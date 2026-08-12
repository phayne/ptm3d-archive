
// Embedded OpenCL kernel source for PTM thermal solver
#define MAX_LAYERS 64

// Stefan-Boltzmann. OpenCL cannot include constants/stefan_boltzmann.h, so this
// literal is checked against ptm::kStefanBoltzmann by the static_assert below
// getEmbeddedKernelSource(); the two cannot drift.
#ifndef SIGMA
#define SIGMA 5.670374419e-8f
#endif

// --------------------------------------------------------------------------
// Regolith conductivity laws. DEVICE MIRROR of
// src/cpp/thermal/conductivity_model.h -- keep the two in step; both are
// checked against the paper's published values by
// src/cpp/tools/test_conductivity_model.cpp.
//
//   cond_model == 0 : Hayne et al. (2017)      k = kc*(1 + R350*T^3)   [DEFAULT]
//   cond_model == 1 : Martinez & Siegler (2021) JGR Planets 126, e2021JE006829
//                     Eq. 10: k = (A1*rho-A2)*k_am(T) + (B1*rho-B2)*T^3
//                     Eq.  9: k_am(T) = Ac + Bc/T^4 + Cc/T^3 + Dc/T^2 + Ec/T
//                                     + Fc*T + Gc*T^2 + Hc*T^3 + Ic*T^4
//
// `rho` is already a per-node buffer on this path (see the kernels below), so
// Martinez needs NO new device buffer. The T floor of 2.7 K is PTM3D's own
// ABSOLUTE_ZERO_K: Eq. 9 diverges negative as T -> 0 and the solver never lets
// a node below the CMB, so the clamp is a domain restriction, not a fudge.
// --------------------------------------------------------------------------
#define PTM_MARTINEZ_TMIN 2.7f

// Eq. 9 without the domain clamp; callers pass t >= PTM_MARTINEZ_TMIN. Split
// out so martinez_k does not re-clamp a value it has already clamped, in a
// loop that runs per node, per substep, per timestep.
inline float martinez_kam_at_or_above(float t) {
    float it = 1.0f / t;
    float it2 = it * it;
    float it3 = it2 * it;
    float it4 = it2 * it2;
    float t2 = t * t;
    float t3 = t2 * t;
    float t4 = t2 * t2;
    return -0.203297f
         + (-11.472f)     * it4
         + ( 22.5793f)    * it3
         + (-14.3084f)    * it2
         + (  3.41742f)   * it
         + (  0.01101f)   * t
         + ( -2.8049e-5f) * t2
         + (  3.35837e-8f)* t3
         + (-1.40021e-11f)* t4;
}

inline float martinez_k(float T, float rho) {
    float t = fmax(T, PTM_MARTINEZ_TMIN);
    float a = 5.0821e-6f  * rho - 5.1e-3f;
    float b = 2.0022e-13f * rho - 1.953e-10f;
    return a * martinez_kam_at_or_above(t) + b * (t * t * t);
}

// cond_model==0 reproduces the historical `kc * (1 + R350*T^3)` expression
// exactly (same operations, same order), so the default path is unchanged.
inline float compute_k_model(float T, float kc, float rho, float R350, int cond_model) {
    if (cond_model == 1) return martinez_k(T, rho);
    float T3 = T * T * T;
    return kc * (1.0f + R350 * T3);
}

inline float compute_cp(float T, float cp0, __constant float* cp_coeffs, int n_coeffs) {
    // Polynomial: cp = cp0 + T * (c0 + T * (c1 + T * (c2 + ...)))
    // Horner form for coefficients: result = c[n-1]; then result = result*T + c[i] for i=n-2..0
    // Final: cp = cp0 + T * result
    if (n_coeffs == 0) return cp0;
    float result = cp_coeffs[n_coeffs - 1];
    for (int i = n_coeffs - 2; i >= 0; --i) {
        result = result * T + cp_coeffs[i];
    }
    // IMPORTANT: cp = cp0 + T * result, NOT cp0 * result!
    return cp0 + T * result;
}

__kernel void materialUpdateKernel(
    __global const float* T,
    __global float* k,
    __global float* cp,
    __global float* kappa,
    __global const float* rho,
    __global const float* kc,
    float R350,
    __constant float* cp_coeffs,
    int n_cp_coeffs,
    float cp0,
    int n_faces,
    int n_layers,
    int cond_model)          // 0 = Hayne 2017 (default), 1 = Martinez 2021
{
    int idx = get_global_id(0);
    int total = n_faces * n_layers;
    if (idx >= total) return;

    float T_val = T[idx];
    float kc_val = kc[idx];
    float rho_val = rho[idx];

    float k_val = compute_k_model(T_val, kc_val, rho_val, R350, cond_model);
    float cp_val = compute_cp(T_val, cp0, cp_coeffs, n_cp_coeffs);
    if (cp_val <= 0.0f) cp_val = 800.0f;

    k[idx] = k_val;
    cp[idx] = cp_val;
    kappa[idx] = k_val / (rho_val * cp_val);
}

__kernel void combinedThermalKernel(
    __global float* T_global,
    __global float* k_global,
    __global float* cp_global,
    __global float* kappa_global,
    __global const float* rho,
    __global const float* kc,
    __global const float* flux,
    __global const float* emissivity,
    __global const float* z,
    __global const float* T_deep,
    __global const float* Qb,
    float R350,
    __constant float* cp_coeffs,
    int n_cp_coeffs,
    float cp0,
    float sigma,
    float dt,
    int n_faces,
    int n_layers,
    int cond_model)          // 0 = Hayne 2017 (default), 1 = Martinez 2021
{
    int face = get_global_id(0);
    if (face >= n_faces) return;
    if (n_layers > MAX_LAYERS) return;

    float T_local[MAX_LAYERS];
    float k_local[MAX_LAYERS];
    float cp_local[MAX_LAYERS];
    float kappa_local[MAX_LAYERS];
    float a[MAX_LAYERS];
    float b[MAX_LAYERS];
    float c[MAX_LAYERS];
    float d[MAX_LAYERS];

    // Load and compute material properties
    for (int i = 0; i < n_layers; ++i) {
        int idx = i * n_faces + face;
        float T_val = T_global[idx];
        float kc_val = kc[idx];
        float rho_val = rho[idx];

        T_local[i] = T_val;
        k_local[i] = compute_k_model(T_val, kc_val, rho_val, R350, cond_model);
        cp_local[i] = compute_cp(T_val, cp0, cp_coeffs, n_cp_coeffs);
        if (cp_local[i] <= 0.0f) cp_local[i] = 800.0f;
        kappa_local[i] = k_local[i] / (rho_val * cp_local[i]);
    }

    float eps = emissivity[face];
    float Q_surf = flux[face];
    float T_bottom = T_deep[face];
    float Q_bottom = Qb[face];

    // Surface layer
    {
        float dz = z[1] - z[0];
        float dz2 = dz * dz;
        float Fo = kappa_local[0] * dt / dz2;
        float T3 = T_local[0] * T_local[0] * T_local[0];
        float rad = 4.0f * eps * sigma * T3 * dz / k_local[0];

        a[0] = 0.0f;
        b[0] = 1.0f + Fo + 0.5f * rad;
        c[0] = -Fo;

        float Q_rad_old = eps * sigma * T_local[0] * T3;
        float rho_0 = rho[face];
        d[0] = (1.0f - Fo) * T_local[0] + Fo * T_local[1]
             + dt * (Q_surf - Q_rad_old) / (rho_0 * cp_local[0] * dz);
    }

    // Interior layers - flux-based formulation matching CPU code
    for (int i = 1; i < n_layers - 1; ++i) {
        float dz_minus = z[i] - z[i - 1];
        float dz_plus = z[i + 1] - z[i];
        float dz_avg = 0.5f * (dz_minus + dz_plus);

        // Grid coefficients (matching CPU g1, g2 formulation)
        float g1 = 1.0f / (dz_minus * dz_avg);  // coupling to i-1
        float g2 = 1.0f / (dz_plus * dz_avg);   // coupling to i+1

        // Flux-based: use k[i-1] for lower interface, k[i] for upper
        float alpha = g1 * k_local[i - 1];
        float beta = g2 * k_local[i];
        float inv_rhocp = dt / (rho[i * n_faces + face] * cp_local[i]);

        a[i] = -0.5f * inv_rhocp * alpha;
        b[i] = 1.0f + 0.5f * inv_rhocp * (alpha + beta);
        c[i] = -0.5f * inv_rhocp * beta;

        // RHS: Crank-Nicolson explicit part
        d[i] = T_local[i] + 0.5f * inv_rhocp * (
            alpha * T_local[i - 1]
            - (alpha + beta) * T_local[i]
            + beta * T_local[i + 1]);
    }

    // Bottom layer
    {
        int i = n_layers - 1;
        int idx = i * n_faces + face;
        float dz = z[i] - z[i - 1];
        float dz2 = dz * dz;
        float Fo = kappa_local[i] * dt / dz2;

        a[i] = -0.5f * Fo;
        b[i] = 1.0f + 0.5f * Fo;
        c[i] = 0.0f;

        d[i] = 0.5f * Fo * T_local[i - 1]
             + (1.0f - 0.5f * Fo) * T_local[i]
             + 0.5f * Fo * T_bottom;

        if (Q_bottom != 0.0f) {
            float rho_n = rho[idx];
            d[i] += dt * Q_bottom / (rho_n * cp_local[i] * dz);
        }
    }

    // Thomas algorithm
    for (int i = 1; i < n_layers; ++i) {
        float m = a[i] / b[i - 1];
        b[i] = b[i] - m * c[i - 1];
        d[i] = d[i] - m * d[i - 1];
    }

    T_local[n_layers - 1] = d[n_layers - 1] / b[n_layers - 1];
    for (int i = n_layers - 2; i >= 0; --i) {
        T_local[i] = (d[i] - c[i] * T_local[i + 1]) / b[i];
    }

    // Write back
    for (int i = 0; i < n_layers; ++i) {
        int idx = i * n_faces + face;
        T_global[idx] = T_local[i];
        k_global[idx] = k_local[i];
        cp_global[idx] = cp_local[i];
        kappa_global[idx] = kappa_local[i];
    }
}

// Persistent thermal kernel - processes ALL timesteps on GPU
// Uses internal sub-stepping to maintain numerical stability when output dt is large
__kernel void persistentThermalKernel(
    __global float* T_global,
    __global float* k_global,
    __global float* cp_global,
    __global float* kappa_global,
    __global const float* rho,
    __global const float* kc,
    __global const float* flux_all,
    __global float* output_temps,
    __global const float* emissivity,
    __global const float* z,
    __global const float* T_deep,
    __global const float* Qb,
    float R350,
    __constant float* cp_coeffs,
    int n_cp_coeffs,
    float cp0,
    float sigma,
    float dt_output,  // OUTPUT timestep (may be large, e.g., hours)
    int n_faces,
    int n_layers,
    int n_timesteps,
    // Schorghofer & Taylor (2007) ice-stability accumulator (optional).
    // When do_accumulate != 0, ⟨E(T(z,t))⟩ is summed per face × layer
    // for every output timestep in [accum_start, accum_end).
    __global float* avg_E_accum,
    int do_accumulate,
    int accum_start,   // first GLOBAL output timestep at which to accumulate
    int accum_end,     // one past last GLOBAL output timestep to accumulate
    // Phase R (R2): global timestep offset for in-solver dispatch
    // self-chunking. A logical session of N output timesteps may be split by
    // the host into sub-dispatches [0,s),[s,2s),... with a clFinish between
    // them to keep each command buffer under the machine watchdog ceiling.
    // Each sub-dispatch runs a LOCAL loop t in [0,n_timesteps) but reads flux
    // from / writes output to the GLOBAL slot (t_offset + t), continuing the
    // T(z) profile it inherits in T_global. t_offset == 0 (the un-chunked
    // case) makes every indexed expression textually identical to pre-R2.
    int t_offset,
    // Phase R (R2 fix, PORT-R2-01): per-face frozen von-Neumann substep count.
    // Written on the first sub-dispatch of a session (t_offset==0), read on the
    // rest, so a self-chunked session integrates with the SAME substep count as
    // the un-chunked dispatch would. Re-derived at each cycle entry.
    __global int* n_substeps_frozen,
    // Regolith conductivity law: 0 = Hayne 2017 (default), 1 = Martinez &
    // Siegler 2021. Appended LAST on purpose so the hard-coded arg indices
    // kArgNTimesteps(20) / kArgTOffset(25) above stay valid.
    int cond_model,
    // Per-face surface-BC certificate: max over this session's substeps of the
    // achieved |f| / (4 eps sigma T^3) [K] from the surface Newton. Appended
    // after cond_model for the same reason cond_model was appended last.
    __global float* surf_resid_K,
    // Companion: max Newton steps this face took in any substep. Without it the
    // round-off floor's COST is invisible from outside the kernel -- the
    // residual alone cannot distinguish "exited in 2 steps at the precision
    // limit" from "spun to the 10-step cap and ended at the same place".
    __global int* surf_iters)
{
    int face = get_global_id(0);
    if (face >= n_faces) return;
    if (n_layers > MAX_LAYERS) return;

    float T_local[MAX_LAYERS];
    float k_local[MAX_LAYERS];
    float cp_local[MAX_LAYERS];
    float kappa_local[MAX_LAYERS];
    float a[MAX_LAYERS];
    float b[MAX_LAYERS];
    float c_coef[MAX_LAYERS];
    float d[MAX_LAYERS];
    float kc_local[MAX_LAYERS];
    float rho_local[MAX_LAYERS];

    // Load initial state
    for (int i = 0; i < n_layers; ++i) {
        int idx = i * n_faces + face;
        T_local[i] = T_global[idx];
        kc_local[i] = kc[idx];
        rho_local[i] = rho[idx];
    }

    float eps = emissivity[face];
    float Q_bottom = Qb[face];  // Interior heat flux (W/m²)

    // Geothermal-equilibrium temperature: the coldest a surface radiating
    // against the interior heat flux Qb can sustain in steady darkness,
    // T_geo = (Qb/(eps*sigma))^0.25 (all of Qb exits at the surface as
    // radiation). Used as a physics-based low-side floor at the end of each
    // substep; 0 (disabled) when Qb <= 0.
    float T_geo = (Q_bottom > 0.0f)
        ? native_sqrt(native_sqrt(Q_bottom / (eps * sigma))) : 0.0f;

    // Sub-step count per output interval. Phase R (R2 fix, PORT-R2-01): the
    // von-Neumann count depends on the session-ENTRY T (kappa falls as the
    // surface warms), so a self-chunked session MUST reuse one count across its
    // sub-dispatches or the integration drifts (~0.02 K, ceiling-dependent =>
    // fails bit-identity). Derive-and-FREEZE on the first sub-dispatch
    // (t_offset==0), reuse on the rest. Because t_offset resets to 0 at each
    // cycle's entry, each cycle re-derives from ITS entry T — identical to the
    // pre-R2 single-dispatch-per-cycle behavior (this is what keeps multi-cycle
    // unsplit==pre-R2). The frozen value is the INT; dt is recomputed from the
    // SAME int below (bit-exact), never cached as a float.
    int n_substeps;
    if (t_offset == 0) {
        // Compute stable timestep based on von Neumann criterion (Fo < 0.5)
        // Need to compute kappa first to get stable dt
        // Use initial temperature to estimate kappa
        float max_kappa = 0.0f;
        float min_dz2 = 1e10f;
        for (int i = 0; i < n_layers; ++i) {
            float k_val = compute_k_model(T_local[i], kc_local[i], rho_local[i], R350, cond_model);
            float cp_val = compute_cp(T_local[i], cp0, cp_coeffs, n_cp_coeffs);
            if (cp_val <= 0.0f) cp_val = 800.0f;
            float kappa_val = k_val / (rho_local[i] * cp_val);
            if (kappa_val > max_kappa) max_kappa = kappa_val;

            if (i > 0) {
                float dz = z[i] - z[i-1];
                float dz2 = dz * dz;
                if (dz2 < min_dz2) min_dz2 = dz2;
            }
        }

        // Stable dt: Fo = kappa * dt / dz^2 < 0.49
        float dt_stable = 0.49f * min_dz2 / max_kappa;

        // Clamp dt_stable to reasonable range. The lower bound is the
        // larger of:
        //   - 1 second (legacy floor for fine-event resolution), and
        //   - dt_output / N_SUBSTEPS_MAX, which caps per-output-step
        //     substeps to a value that keeps the persistent kernel below
        //     the macOS GPU-watchdog timeout (~5–10 s default).
        // Without this second floor, very cold checkpoint states (24 K
        // PSR floors) yield tiny cp(T), large kappa = k/(rho·cp), tiny
        // dt_stable per the CFL criterion, and worst-case work-items
        // accumulate enough substeps that the kernel never finishes
        // within the watchdog window — causing the AMD driver to abort
        // with gpusKillClientExt.
        const int N_SUBSTEPS_MAX = 200;  // 432 s / step at dt_output = 86400 s
        float min_dt_stable = dt_output / (float)N_SUBSTEPS_MAX;
        if (min_dt_stable < 1.0f) min_dt_stable = 1.0f;
        dt_stable = fmax(min_dt_stable, fmin(dt_stable, dt_output));

        // Number of sub-steps per output interval (capped above by the
        // dt_stable floor)
        n_substeps = (int)ceil(dt_output / dt_stable);
        n_substeps_frozen[face] = n_substeps;   // FREEZE for later sub-dispatches
    } else {
        n_substeps = n_substeps_frozen[face];    // reuse this session's frozen count
    }
    float dt = dt_output / (float)n_substeps;  // Actual sub-step size (bit-exact)

    // Process ALL output timesteps
    for (int t = 0; t < n_timesteps; ++t) {
        int tg = t_offset + t;   // GLOBAL timestep (== t when un-chunked)
        float Q_surf = flux_all[tg * n_faces + face];

        // Sub-stepping loop for numerical stability
        for (int substep = 0; substep < n_substeps; ++substep) {
            // Material properties (updated each sub-step)
            for (int i = 0; i < n_layers; ++i) {
                float T_val = T_local[i];
                k_local[i] = compute_k_model(T_val, kc_local[i], rho_local[i], R350, cond_model);
                cp_local[i] = compute_cp(T_val, cp0, cp_coeffs, n_cp_coeffs);
                // Guard against invalid cp values (e.g., negative from polynomial at extreme T)
                if (cp_local[i] <= 0.0f || !isfinite(cp_local[i])) {
                    cp_local[i] = 800.0f;  // Fallback to reasonable value
                }
                kappa_local[i] = k_local[i] / (rho_local[i] * cp_local[i]);
            }

            // Step 1: the surface energy balance.
            //
            // THIS IS THE SAME SPECIFICATION AS THE CPU solveSurfaceBC()
            // (src/cpp/thermal/thermal_model.h), evaluated in float32. Seed,
            // relaxation, exit criterion, ulp floor, step limiter and
            // max_iter are identical; the derivation lives in that header and
            // is not repeated here. A change to one MUST be made to the other.
            // Energy balance at surface: Qs + k*(T1-T0)/dz = eps*sigma*T0^4
            {
                float dz = z[1] - z[0];
                float k0 = k_local[0];
                float T1 = T_local[1];

                // Initial guess: the UPPER end of the root bracket.
                //
                // f(T) = Qs + k0*(T1 - T)/dz - eps*sigma*T^4 is strictly
                // decreasing, and f(T_eq)*f(T1) <= 0 for T_eq = (Qs/(eps*sigma))^0.25
                // (f(T_eq) = k0*(T1-T_eq)/dz, f(T1) = eps*sigma*(T_eq^4 - T1^4)),
                // so the unique root is ALWAYS bracketed by [min(T_eq,T1), max(T_eq,T1)].
                // Starting at the upper end also puts Newton on the side where the
                // concave f (f'' = -12*eps*sigma*T^2 < 0) converges monotonically.
                //
                // PTM-115: starting at T_eq unconditionally when Qs > 0 is the
                // defect. As Qs -> 0+, T_eq -> 0, while the step limiter below
                // caps growth at 1.5x per iteration over max_iter=10 -> the
                // iterate can travel at most 1.5^10 = 57.665x from the guess.
                // Any face with 0 < Qs < eps*sigma*(T_root/57.665)^4 (~5e-7 W/m^2
                // for a 100 K root) therefore returned an under-converged,
                // spuriously COLD surface temperature, exactly 1.5^10 * T_eq.
                // Sunlit faces have T_eq >= T1 and are bit-unchanged.
                float T_surf = T1;
                if (Q_surf > 0.0f) {
                    // Equilibrium temperature (sqrt(sqrt(x)) = x^0.25, more accurate)
                    float x = Q_surf / (eps * sigma);
                    float T_eq = native_sqrt(native_sqrt(x));
                    if (T_eq > T_surf) T_surf = T_eq;
                }

                // Exit when the RADIATIVELY NORMALIZED residual is small:
                //   |f| <= max(tol_K*4*eps*sigma*T0^3, 2*|df|*FLT_EPSILON*T0)
                // with omega = 1.0 and max_iter = 10.
                //
                // THE SECOND TERM IS LOAD-BEARING ON THIS PATH. In float32 the
                // residual test's own round-off noise is 1.08e-4 K at 24 K on
                // the moon3d grid -- i.e. AT tol_K -- so without the floor a
                // fully converged iterate cannot certify convergence and spins
                // to the iteration cap on essentially every cold face. The
                // achieved residual is written out per face below so a run that
                // is limited by that floor rather than by tol_K says so.
                float f = 0.0f;
                float h_rad = 0.0f;
                int   iter = 0;
                for (; iter < 10; ++iter) {
                    float T0 = T_surf;
                    float T3 = T0 * T0 * T0;

                    // f = Qs + k*(T1 - T0)/dz - eps*sigma*T0^4
                    f = Q_surf + k0 * (T1 - T0) / dz - eps * sigma * T3 * T0;

                    // df/dT0 = -k/dz - 4*eps*sigma*T0^3
                    float df = -k0 / dz - 4.0f * eps * sigma * T3;

                    if (fabs(df) < 1e-10f) df = -1e-10f;

                    h_rad = 4.0f * eps * sigma * T3;

                    // MIN_ITER = 1: testing before the first step turns a
                    // Qb = 0 column into a continuum of accepted states of
                    // width tol_K, so it stops wherever it happens to be.
                    if (iter > 0) {
                        float tol_res = 1.0e-4f * h_rad;
                        float tol_ulp = 2.0f * fabs(df) * FLT_EPSILON * T0;
                        if (fabs(f) <= tol_res || fabs(f) <= tol_ulp) break;
                    }

                    float dT_nr = f / df;   // relaxation omega = 1.0

                    // Robust-Newton step limiter (globalization, NOT an output
                    // clamp). For very cold, low-conductivity faces (PSR floors)
                    // df = -k/dz - 4 eps sigma T^3 -> 0 as both terms vanish, so
                    // an unguarded Newton step overshoots past the unique
                    // positive root of the monotonic f(T) into NEGATIVE T,
                    // collapsing the column. Capping the COOLING step at 0.5*T0
                    // keeps every iterate strictly positive. The GROWTH side is
                    // 4*T0 rather than 0.5*T0, which at 1.5x/iteration imposed a
                    // factor-57.7 ceiling over 10 iterations; that ceiling is
                    // unreachable from the max(T_eq, T1) seed, and widening it
                    // means the code does not silently depend on that premise.
                    // The final temperature is still set by physics (the root of
                    // f), not bounded to a fixed range.
                    float cool_cap   = 0.5f * T0;
                    float growth_cap = 4.0f * T0;
                    if (dT_nr >  cool_cap)   dT_nr =  cool_cap;
                    if (dT_nr < -growth_cap) dT_nr = -growth_cap;
                    T_surf = T0 - dT_nr;
                }

                // Per-face surface-BC certificate: the worst achieved
                // |f| / (4 eps sigma T^3) [K] over every substep this face
                // takes. Each work-item owns its own element, so there is no
                // race. The host reduces it to a per-run maximum and prints it;
                // that is what keeps the float32 round-off floor from binding
                // silently.
                {
                    float resid = (h_rad > 0.0f) ? fabs(f) / h_rad : fabs(f);
                    if (resid > surf_resid_K[face]) surf_resid_K[face] = resid;
                    if (iter > surf_iters[face]) surf_iters[face] = iter;
                }

                T_local[0] = T_surf;

                // CRITICAL: Recompute surface material properties using NEW surface temperature
                // This matches CPU order: surfTemp() → material refresh → Thomas solve
                // Without this, k_local[0] would use OLD T[0] and the coupling between
                // layers 0 and 1 in the tridiagonal system would be wrong.
                k_local[0] = compute_k_model(T_surf, kc_local[0], rho_local[0], R350, cond_model);
                cp_local[0] = compute_cp(T_surf, cp0, cp_coeffs, n_cp_coeffs);
                if (cp_local[0] <= 0.0f || !isfinite(cp_local[0])) {
                    cp_local[0] = 800.0f;
                }
                kappa_local[0] = k_local[0] / (rho_local[0] * cp_local[0]);
            }

            // Step 2: Apply bottom flux boundary condition (matches CPU botTemp)
            // CPU does: T[n] = T[n-1] + Qb * dz / k
            // This is applied BEFORE the interior solve, using OLD T[n-1]
            {
                int n = n_layers - 1;
                float dz_bottom = z[n] - z[n - 1];
                float k_bottom = k_local[n];
                // Flux BC: ensures heat flux at bottom equals Qb (interior heat flux)
                T_local[n] = T_local[n - 1] + Q_bottom * dz_bottom / k_bottom;
            }

            // Step 3: Interior diffusion using Crank-Nicolson (excluding surface and bottom)
            // Build tridiagonal system for layers 1 to n_layers-2

            // Layer 1 (next to surface) - surface T[0] is known from Newton-Raphson
            // IMPORTANT: CPU code copies T AFTER surfTemp(), so T_old[0] = T_new[0]
            // We must use T_local[0] (new surface temp) for BOTH explicit and implicit parts
            int n_int = n_layers - 2;  // Interior node count (excluding surface and bottom)

            if (n_layers > 2) {
                float dz_minus = z[1] - z[0];
                float dz_plus = z[2] - z[1];
                float dz_avg = 0.5f * (dz_minus + dz_plus);

                float g1 = 1.0f / (dz_minus * dz_avg);
                float g2 = 1.0f / (dz_plus * dz_avg);

                float alpha = g1 * k_local[0];
                float beta = g2 * k_local[1];
                float inv_rhocp = dt / (rho_local[1] * cp_local[1]);

                // a[1] couples to T[0] which is now known - move to RHS
                a[1] = 0.0f;
                b[1] = 1.0f + 0.5f * inv_rhocp * (alpha + beta);
                c_coef[1] = -0.5f * inv_rhocp * beta;

                // RHS: Matches CPU exactly - both explicit and implicit use NEW surface temp
                // CPU does: d[0] = T_old[i] + 0.5*inv_rhocp*(alpha*T_old[i-1] - ...)
                // then:     d[0] -= a[0] * Tptr[0]  which adds +0.5*inv_rhocp*alpha*Tptr[0]
                // Since T_old[0] = Tptr[0] = T_new (surfTemp called before copy), total = inv_rhocp*alpha*T0_new
                d[1] = T_local[1] + inv_rhocp * alpha * T_local[0]
                     - 0.5f * inv_rhocp * (alpha + beta) * T_local[1]
                     + 0.5f * inv_rhocp * beta * T_local[2];
            }

            // Interior layers 2 to n_layers-2 (EXCLUDING bottom layer)
            for (int i = 2; i < n_layers - 1; ++i) {
                float dz_minus = z[i] - z[i - 1];
                float dz_plus = z[i + 1] - z[i];
                float dz_avg = 0.5f * (dz_minus + dz_plus);

                float g1 = 1.0f / (dz_minus * dz_avg);
                float g2 = 1.0f / (dz_plus * dz_avg);

                float alpha = g1 * k_local[i - 1];
                float beta = g2 * k_local[i];
                float inv_rhocp = dt / (rho_local[i] * cp_local[i]);

                a[i] = -0.5f * inv_rhocp * alpha;
                b[i] = 1.0f + 0.5f * inv_rhocp * (alpha + beta);
                c_coef[i] = -0.5f * inv_rhocp * beta;

                d[i] = T_local[i] + 0.5f * inv_rhocp * (
                    alpha * T_local[i - 1]
                    - (alpha + beta) * T_local[i]
                    + beta * T_local[i + 1]);
            }

            // Fold bottom boundary into last interior node's RHS
            // CPU does: d[n_int-1] -= c[n_int-1] * Tptr[n_tot - 1]
            // Our last interior node is at layer n_layers-2
            // T[n_layers-1] was set by flux BC above
            if (n_int > 0) {
                int last_int = n_layers - 2;  // Last interior layer
                d[last_int] -= c_coef[last_int] * T_local[n_layers - 1];
                c_coef[last_int] = 0.0f;  // No longer couples upward
            }

            // Thomas algorithm for interior only (layers 1 to n_layers-2)
            // Layer 0 is already solved via Newton-Raphson
            // Layer n_layers-1 is set by flux BC (computed from T[n_layers-2])
            for (int i = 2; i < n_layers - 1; ++i) {
                float m = a[i] / b[i - 1];
                b[i] = b[i] - m * c_coef[i - 1];
                d[i] = d[i] - m * d[i - 1];
            }

            // Back substitution for interior only (layers 1 to n_layers-2)
            // Start from last interior node, not the bottom boundary
            if (n_int > 0) {
                int last_int = n_layers - 2;
                T_local[last_int] = d[last_int] / b[last_int];
                for (int i = last_int - 1; i >= 1; --i) {
                    T_local[i] = (d[i] - c_coef[i] * T_local[i + 1]) / b[i];
                }
            }

            // Note: Bottom layer T[n_layers-1] was already set by flux BC before solve
            // and remains unchanged during interior solve (not part of tridiagonal system)

            // NaN/Inf guard only (NOT a temperature clamp): the implicit
            // Crank-Nicolson conduction update is unconditionally stable, so
            // valid temperatures are left untouched and set by physics. This
            // recovers solely from a non-finite value (e.g. a degenerate cell),
            // which correct physics never produces, so the sim isn't corrupted.
            for (int i = 0; i < n_layers; ++i) {
                if (!isfinite(T_local[i])) {
                    T_local[i] = 200.0f;  // last-resort recovery, should never fire
                }
            }

            // Geothermal-equilibrium floor (physics-based, NOT an arbitrary
            // [3,500] K clamp). The substep dt is fixed from the initial
            // column and capped at N_SUBSTEPS_MAX for the GPU watchdog, so it
            // under-resolves the ~minute surface skin response. At a sharp
            // sunlit->shadow flux drop on a low-inertia face the upper column
            // cools faster than the under-resolved conduction can resupply
            // heat from below, drifting numerically below the physical night
            // floor (recorded as spurious ~5-24 K minima on faces that are
            // otherwise sunlit, Tavg~100 K). The interior heat flux Qb sets a
            // hard lower bound: in steady darkness the surface radiates Qb, so
            // no node can sustain a temperature below T_geo = (Qb/(eps*sigma))
            // ^0.25 (~24 K here). Real PSR floors converge to T_geo from
            // ABOVE, so this is a no-op for them; only the sub-floor numerical
            // undershoot is corrected. Per-face and physics-set, so it does
            // not bound results to a fixed range or perturb the bulk solution.
            if (T_geo > 0.0f) {
                for (int i = 0; i < n_layers; ++i) {
                    T_local[i] = fmax(T_local[i], T_geo);
                }
            }
        }  // End of sub-stepping loop

        // Store surface temperature after all sub-steps are complete
        output_temps[tg * n_faces + face] = T_local[0];

        // Schorghofer & Taylor (2007) sublimation-rate accumulator (optional).
        // Constants match src/cpp/ice/ice_table.cpp:
        //   PT      = 611.657 Pa  (triple-point pressure)
        //   L_MU/KB = 6139.06 K   (latent heat × molecular mass / Boltzmann)
        //   1/TT    = 1/273.16 K
        //   A       = 6.21e23     ((2π·KB·MU)^-0.5 prefactor)
        // Each face owns its column → no atomics required.
        if (do_accumulate != 0 && tg >= accum_start && tg < accum_end) {
            for (int i = 0; i < n_layers; ++i) {
                float Ti = T_local[i];
                if (Ti > 30.0f) {
                    float arg = -6139.06f * (1.0f / Ti - 1.0f / 273.16f);
                    float pv  = 611.657f * native_exp(arg);
                    float E   = 6.21e23f * pv * native_rsqrt(Ti);
                    avg_E_accum[i * n_faces + face] += E;
                }
            }
        }
    }  // End of output timestep loop

    // Write final state back
    for (int i = 0; i < n_layers; ++i) {
        int idx = i * n_faces + face;
        T_global[idx] = T_local[i];
        k_global[idx] = k_local[i];
        cp_global[idx] = cp_local[i];
        kappa_global[idx] = kappa_local[i];
    }
}
