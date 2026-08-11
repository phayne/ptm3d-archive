/**
 * @file thermal_opencl.cpp
 * @brief OpenCL thermal solver implementation
 */

#ifdef PTM_GPU_OPENCL

// Define PTM_OPENCL_NO_THROW BEFORE including headers to use non-throwing
// CL_CHECK macro. This avoids SIGILL on macOS when exceptions cross shared
// library boundaries with aggressive optimization.
#define PTM_OPENCL_NO_THROW

#include "thermal_opencl.h"
#include "machine_profile.h"   // Phase R (R2): per-host dispatch self-chunk ceiling
#include <fstream>
#include <sstream>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <atomic>
#include <limits>   // eq-extrap: quiet_NaN sentinel for "face not extrapolated"

namespace gpu {
namespace opencl {

// ---------------------------------------------------------------------------
// Phase R (R2): in-solver dispatch self-chunking helpers.
//
// The persistent thermal kernel loops n_timesteps internally in ONE
// clEnqueueNDRangeKernel. A single unbroken dispatch that runs too many
// timesteps is exactly what tripped the macOS+AMD gpusKillClientExt
// command-buffer watchdog on the year-long run (1560 steps in one dispatch vs
// the proven-safe ~240). To stay under the host's watchdog ceiling WITHOUT an
// external wrapper, a logical session of N output timesteps is split into
// sub-dispatches of <= max_timesteps_per_dispatch, with a host clFinish
// between them. State (the T(z) profile) is carried across sub-dispatches in
// the persistent buf_T_; each sub-dispatch reads flux from / writes output to
// the correct GLOBAL timestep slot via the kernel's t_offset arg. The split is
// NUMERICALLY INERT: the exact same per-timestep Crank-Nicolson updates run in
// the exact same order, and clFinish changes no computed value. When
// n_timesteps <= max_timesteps_per_dispatch the split degenerates to a single
// dispatch (t_offset==0), which is the pre-R2 code path bit-for-bit.
// ---------------------------------------------------------------------------

// Kernel-arg indices shared by both persistent-kernel callers (the arg list is
// identical in solvePersistentAllTimesteps and solvePersistentMultipleCycles).
static constexpr int kArgNTimesteps = 20;  // int n_timesteps
static constexpr int kArgTOffset    = 25;  // int t_offset (Phase R, R2 — last arg)

// In-process override of the self-chunk ceiling, set by the auto-calibrate path
// (ptm_thermal_sim.cpp ensureGpuCalibrated) via setDispatchCeilingOverride().
// Checked BEFORE the static cache so the probe's own dispatch — which latches a
// conservative value into `cached` below — cannot defeat the calibrated ceiling.
// -1 = unset. Atomic: the value is written from the (single-threaded) calibrate
// helper and read from the solver worker threads.
static std::atomic<int> g_dispatch_ceiling_override{-1};

void setDispatchCeilingOverride(int max_timesteps_per_dispatch) {
    // The operator's explicit ceiling always wins; never override it.
    if (const char* e = std::getenv("PTM_MAX_TIMESTEPS_PER_DISPATCH")) {
        if (e[0] != '\0') return;
    }
    g_dispatch_ceiling_override.store(
        max_timesteps_per_dispatch > 0 ? max_timesteps_per_dispatch : -1,
        std::memory_order_relaxed);
}

// The self-chunking ceiling: an in-process override (auto-calibrate) wins first;
// otherwise loaded once from the per-host machine profile (R1), or a
// conservative default if none exists, with an env override for ops/testing.
// Fail-loud: the source of the ceiling is announced once.
static int gpuMaxTimestepsPerDispatch() {
    // Auto-calibrate override (silent here — ensureGpuCalibrated does the INFO
    // logging so the ceiling's provenance is announced exactly once, on stdout).
    int ov = g_dispatch_ceiling_override.load(std::memory_order_relaxed);
    if (ov > 0) return ov;
    static int cached = []() -> int {
        if (const char* e = std::getenv("PTM_MAX_TIMESTEPS_PER_DISPATCH")) {
            try {
                int v = std::stoi(e);
                if (v > 0) {
                    std::cerr << "[GPU Thermal] dispatch self-chunk ceiling: " << v
                              << " timesteps/launch (from "
                                 "PTM_MAX_TIMESTEPS_PER_DISPATCH)\n";
                    return v;
                }
            } catch (...) {}
        }
        gpu::MachineProfile p = gpu::loadMachineProfileOrDefault();
        std::cerr << "[GPU Thermal] dispatch self-chunk ceiling: "
                  << p.max_timesteps_per_dispatch << " timesteps/launch ("
                  << (p.valid ? "from host profile" : "conservative default")
                  << ")\n";
        return p.max_timesteps_per_dispatch;
    }();
    return cached;
}

// Enqueue the persistent kernel over global output timesteps [0, n_timesteps),
// self-chunked into sub-dispatches of <= max_ts each with a clFinish between.
// The caller must have set every kernel arg EXCEPT kArgNTimesteps and
// kArgTOffset (this function sets those per sub-dispatch). do_accumulate /
// accum_start / accum_end are in the GLOBAL frame and are honored unchanged.
// Returns CL_SUCCESS, or the first failing clEnqueue/clFinish error.
static cl_int enqueuePersistentSelfChunked(cl_kernel kernel,
                                           cl_command_queue queue,
                                           int n_timesteps, int max_ts,
                                           size_t global_size, size_t local_size) {
    if (max_ts < 1) max_ts = 1;
    int n_sub = (n_timesteps + max_ts - 1) / max_ts;   // ceil
    if (n_sub < 1) n_sub = 1;
    // Wedge-efficacy instrumentation (Fermi acceptance test D): when
    // PTM_LOG_SUBDISPATCH is set, time each sub-dispatch's clFinish so a run
    // can prove every command buffer stayed bounded. Logging only — inert.
    static const bool log_sub = (std::getenv("PTM_LOG_SUBDISPATCH") != nullptr);
    cl_int err = CL_SUCCESS;
    for (int s = 0; s < n_sub; ++s) {
        int g0 = s * max_ts;
        int sub_n = n_timesteps - g0;
        if (sub_n > max_ts) sub_n = max_ts;
        err = clSetKernelArg(kernel, kArgNTimesteps, sizeof(int), &sub_n);
        if (err != CL_SUCCESS) return err;
        err = clSetKernelArg(kernel, kArgTOffset, sizeof(int), &g0);
        if (err != CL_SUCCESS) return err;
        auto t0 = std::chrono::high_resolution_clock::now();
        err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr,
                                     &global_size, &local_size, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) return err;
        // Bounded command buffer: host sync between sub-dispatches. Inert
        // (no computed value changes); for n_sub==1 this is exactly the single
        // enqueue + clFinish the pre-R2 code already performed.
        err = clFinish(queue);
        if (err != CL_SUCCESS) return err;
        if (log_sub) {
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration_cast<std::chrono::duration<double>>(
                            t1 - t0).count() * 1000.0;
            std::cerr << "[GPU Thermal][subdispatch] " << (s + 1) << "/" << n_sub
                      << " steps [" << g0 << "," << (g0 + sub_n) << ") in "
                      << ms << " ms\n";
        }
    }
    return err;
}

// Embedded kernel source (generated from thermal_kernels.cl)
// This avoids needing to distribute a separate .cl file
std::string getEmbeddedKernelSource() {
    return R"CL_SOURCE(
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
    __global float* surf_resid_K)
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
                for (int iter = 0; iter < 10; ++iter) {
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
)CL_SOURCE";
}

// The kernel text above carries its own `#define SIGMA` because OpenCL C cannot
// include a C++ header. These two lines are the tripwire that keeps it honest:
// change ptm::kStefanBoltzmann and the build stops here, change the kernel
// literal and the runtime check below fires. Neither can drift silently.
static_assert(ptm::kStefanBoltzmann == 5.670374419e-8,
              "ptm::kStefanBoltzmann changed. Update the `#define SIGMA` in "
              "getEmbeddedKernelSource() (and re-run "
              "scripts/sync_cl_from_embedded.py), then update this assert.");

namespace {
// Verified once, at kernel build time, against the same literal.
bool kernelSigmaMatchesHost(const std::string& src) {
    return src.find("#define SIGMA 5.670374419e-8f") != std::string::npos;
}
}  // namespace

bool isOpenCLAvailable() {
    cl_uint num_platforms = 0;
    cl_int err = clGetPlatformIDs(0, nullptr, &num_platforms);
    return (err == CL_SUCCESS && num_platforms > 0);
}

void printOpenCLInfo(std::ostream& out) {
    cl_uint num_platforms = 0;
    cl_int err = clGetPlatformIDs(0, nullptr, &num_platforms);

    if (err != CL_SUCCESS || num_platforms == 0) {
        out << "No OpenCL platforms found.\n";
        return;
    }

    std::vector<cl_platform_id> platforms(num_platforms);
    clGetPlatformIDs(num_platforms, platforms.data(), nullptr);

    out << "OpenCL Platforms: " << num_platforms << "\n";

    for (cl_uint p = 0; p < num_platforms; ++p) {
        char buffer[256];
        clGetPlatformInfo(platforms[p], CL_PLATFORM_NAME, sizeof(buffer), buffer, nullptr);
        out << "\nPlatform " << p << ": " << buffer << "\n";

        clGetPlatformInfo(platforms[p], CL_PLATFORM_VENDOR, sizeof(buffer), buffer, nullptr);
        out << "  Vendor: " << buffer << "\n";

        clGetPlatformInfo(platforms[p], CL_PLATFORM_VERSION, sizeof(buffer), buffer, nullptr);
        out << "  Version: " << buffer << "\n";

        // Get devices
        cl_uint num_devices = 0;
        err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, 0, nullptr, &num_devices);
        if (err == CL_SUCCESS && num_devices > 0) {
            std::vector<cl_device_id> devices(num_devices);
            clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, num_devices, devices.data(), nullptr);

            for (cl_uint d = 0; d < num_devices; ++d) {
                clGetDeviceInfo(devices[d], CL_DEVICE_NAME, sizeof(buffer), buffer, nullptr);
                out << "  Device " << d << ": " << buffer << "\n";

                cl_device_type type;
                clGetDeviceInfo(devices[d], CL_DEVICE_TYPE, sizeof(type), &type, nullptr);
                out << "    Type: " << ((type & CL_DEVICE_TYPE_GPU) ? "GPU" : "CPU") << "\n";

                cl_ulong mem;
                clGetDeviceInfo(devices[d], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem), &mem, nullptr);
                out << "    Memory: " << (mem / (1024*1024)) << " MB\n";

                cl_uint units;
                clGetDeviceInfo(devices[d], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(units), &units, nullptr);
                out << "    Compute Units: " << units << "\n";
            }
        }
    }
}

// ThermalOpenCLSolver implementation

ThermalOpenCLSolver::ThermalOpenCLSolver(size_t n_faces, size_t n_layers,
                                         const ThermalOpenCLConfig& config)
    : n_faces_(n_faces)
    , n_faces_active_(n_faces)
    , n_layers_(n_layers)
    , config_(config)
{
}

void ThermalOpenCLSolver::setActiveFaceCount(size_t n) {
    if (n == 0) {
        n_faces_active_ = n_faces_;
        return;
    }
    if (n > n_faces_) {
        std::cerr << "[GPU Thermal] setActiveFaceCount(" << n
                  << ") exceeds capacity " << n_faces_
                  << "; clamping to capacity\n";
        n_faces_active_ = n_faces_;
        return;
    }
    n_faces_active_ = n;
}

ThermalOpenCLSolver::~ThermalOpenCLSolver() {
    release();
}

bool ThermalOpenCLSolver::initialize() {
    if (initialized_) return true;

    // Create OpenCL context with default first-available GPU
    context_ = std::make_unique<OpenCLContext>();
    if (!context_->initialize(true)) {  // Prefer GPU
        if (config_.verbose) {
            std::cerr << "Failed to initialize OpenCL context\n";
        }
        return false;
    }

    if (config_.verbose) {
        printDeviceInfo(context_->deviceInfo());
    }

    // Build kernels and create buffers (non-throwing)
    buildKernels();
    createBuffers();

    // Check if critical resources were created successfully
    if (!program_ || !persistent_kernel_ || !buf_T_ || !buf_rho_) {
        std::cerr << "[GPU Thermal] Failed to initialize OpenCL resources\n";
        release();
        return false;
    }

    initialized_ = true;

    if (config_.verbose) {
        std::cout << "ThermalOpenCLSolver initialized: "
                  << n_faces_ << " faces, " << n_layers_ << " layers\n";
    }

    return true;
}

bool ThermalOpenCLSolver::initialize(const OpenCLDeviceInfo& device_handle) {
    if (initialized_) return true;

    context_ = std::make_unique<OpenCLContext>();
    if (!context_->initialize(device_handle)) {
        if (config_.verbose) {
            std::cerr << "Failed to initialize OpenCL context on device "
                      << device_handle.name << "\n";
        }
        return false;
    }

    if (config_.verbose) {
        printDeviceInfo(context_->deviceInfo());
    }

    buildKernels();
    createBuffers();

    if (!program_ || !persistent_kernel_ || !buf_T_ || !buf_rho_) {
        std::cerr << "[GPU Thermal] Failed to initialize OpenCL resources on device "
                  << device_handle.name << "\n";
        release();
        return false;
    }

    initialized_ = true;

    if (config_.verbose) {
        std::cout << "ThermalOpenCLSolver initialized on " << device_handle.name
                  << ": " << n_faces_ << " faces, " << n_layers_ << " layers\n";
    }

    return true;
}

void ThermalOpenCLSolver::buildKernels() {
    // PTM_SOLVER_EQFIX refusal, NARROWED 2026-08-11.
    //
    // The surface-Newton exit criterion is no longer part of this gate: the
    // kernel below implements the same specification as the CPU
    // solveSurfaceBC, unconditionally. What the gate still asks for is release
    // of the T_geo floor, and this kernel applies that floor every substep
    // with no way to switch it off. Accepting the flag and keeping the floor
    // would be a quiet substitution -- the class of failure PTM-115/116 were.
    // Refuse instead.
    {
        const char* e = std::getenv("PTM_SOLVER_EQFIX");
        if (e && *e && std::strcmp(e, "0") != 0) {
            std::cerr << "\n[GPU Thermal] FATAL: PTM_SOLVER_EQFIX asks for the "
                         "T_geo floor to be released, but the GPU persistent "
                         "kernel applies it unconditionally, every substep.\n"
                         "  Running here under the gate would keep the floor while "
                         "the log claimed it was released, and nothing would say "
                         "so.\n"
                         "  (The surface-Newton criterion is NOT the issue: this "
                         "kernel implements the same specification as the CPU "
                         "solveSurfaceBC as of fix/surface-bc-unification.)\n"
                         "  Use a CPU scalar run, or unset PTM_SOLVER_EQFIX."
                      << std::endl;
            std::exit(1);
        }
    }

    std::string source;
    const std::string embedded = getEmbeddedKernelSource();
    if (!kernelSigmaMatchesHost(embedded)) {
        std::cerr << "\n[GPU Thermal] FATAL: the embedded kernel's Stefan-Boltzmann "
                     "constant does not match ptm::kStefanBoltzmann. The GPU surface "
                     "energy balance and the CPU one would disagree by construction. "
                     "Fix the `#define SIGMA` in getEmbeddedKernelSource().\n";
        std::exit(1);
    }

    // Phase R / Fermi F5: the kernel source is single-sourced from the embedded
    // string (see scripts/sync_cl_from_embedded.py). buildKernels() must NEVER
    // silently swap in a different kernel: it (a) announces which source it
    // compiled, (b) warns loudly instead of silently falling back when a
    // requested kernel_path fails to open, and (c) warns loudly if a loaded
    // external kernel DIVERGES from the validated embedded physics.
    if (!config_.kernel_path.empty()) {
        std::ifstream file(config_.kernel_path);
        if (file.is_open()) {
            std::stringstream buffer;
            buffer << file.rdbuf();
            source = buffer.str();
            std::cout << "[GPU Thermal] compiling kernels from FILE: "
                      << config_.kernel_path << "\n";
            if (source != embedded) {
                std::cerr << "[GPU Thermal] WARNING (F5): external kernel file '"
                          << config_.kernel_path << "' DIFFERS from the validated "
                             "embedded kernel source. The compiled physics may not "
                             "match the byte-identity-validated solver. Leave "
                             "kernel_path empty to use the embedded kernel, or run "
                             "scripts/sync_cl_from_embedded.py to resync the .cl.\n";
            }
        } else {
            std::cerr << "[GPU Thermal] WARNING (F5): kernel_path '"
                      << config_.kernel_path << "' could not be opened; falling "
                         "back to the embedded kernel source.\n";
        }
    }

    // Fall back to embedded source
    if (source.empty()) {
        source = embedded;
        std::cout << "[GPU Thermal] compiling kernels from EMBEDDED source "
                     "(validated default)\n";
    }

    // Build program directly (avoid context_->buildProgram() which throws exceptions
    // that cause SIGILL on macOS when crossing shared library boundaries)
    const char* src = source.c_str();
    size_t len = source.size();

    cl_int err;
    program_ = clCreateProgramWithSource(context_->context(), 1, &src, &len, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "[GPU Thermal] Failed to create program from source: "
                  << clErrorString(err) << "\n";
        program_ = nullptr;
        return;
    }

    std::string options = "-cl-fast-relaxed-math -cl-mad-enable";
    cl_device_id device = context_->device();
    err = clBuildProgram(program_, 1, &device, options.c_str(), nullptr, nullptr);
    if (err != CL_SUCCESS) {
        // Get build log
        size_t log_size = 0;
        clGetProgramBuildInfo(program_, device, CL_PROGRAM_BUILD_LOG,
                             0, nullptr, &log_size);
        if (log_size > 0) {
            std::vector<char> log(log_size);
            clGetProgramBuildInfo(program_, device, CL_PROGRAM_BUILD_LOG,
                                 log_size, log.data(), nullptr);
            std::cerr << "[GPU Thermal] OpenCL program build failed:\n" << log.data() << "\n";
        } else {
            std::cerr << "[GPU Thermal] OpenCL program build failed: "
                      << clErrorString(err) << "\n";
        }
        clReleaseProgram(program_);
        program_ = nullptr;
        return;
    }

    // Create kernels
    material_kernel_ = clCreateKernel(program_, "materialUpdateKernel", &err);
    CL_CHECK(err);
    if (err != CL_SUCCESS) {
        std::cerr << "[GPU Thermal] Failed to create materialUpdateKernel\n";
        return;
    }

    combined_kernel_ = clCreateKernel(program_, "combinedThermalKernel", &err);
    CL_CHECK(err);
    if (err != CL_SUCCESS) {
        std::cerr << "[GPU Thermal] Failed to create combinedThermalKernel\n";
        return;
    }

    persistent_kernel_ = clCreateKernel(program_, "persistentThermalKernel", &err);
    CL_CHECK(err);
    if (err != CL_SUCCESS) {
        std::cerr << "[GPU Thermal] Failed to create persistentThermalKernel\n";
        return;
    }
}

// Reduce the per-face surface-BC certificate to a per-run maximum.
//
// The kernel writes, per face, the largest achieved |f| / (4 eps sigma T^3) [K]
// over every substep it took. That number is the surface Newton's own statement
// of how far from the energy balance it stopped, in kelvin, and it is the only
// way the float32 round-off floor in the exit test can be seen from outside the
// kernel. Read after every dispatch; reported by reportSurfaceBCCertificate().
void ThermalOpenCLSolver::reduceSurfaceBCResidual(cl_command_queue queue) {
    if (!buf_surf_resid_ || n_faces_ == 0) return;
    std::vector<float> resid(n_faces_, 0.0f);
    cl_int rerr = clEnqueueReadBuffer(queue, buf_surf_resid_, CL_TRUE, 0,
                                      n_faces_ * sizeof(float), resid.data(),
                                      0, nullptr, nullptr);
    if (rerr != CL_SUCCESS) {
        std::cerr << "[GPU Thermal] WARNING: could not read the surface-BC "
                     "residual certificate (" << clErrorString(rerr)
                  << "); the surface Newton's achieved accuracy is UNKNOWN for "
                     "this dispatch.\n";
        return;
    }
    for (size_t i = 0; i < resid.size(); ++i) {
        const float r = resid[i];
        if (std::isfinite(r) && r > max_surf_resid_K_) max_surf_resid_K_ = r;
    }
}

void ThermalOpenCLSolver::reportSurfaceBCCertificate() {
    if (max_surf_resid_K_ <= 0.0f) return;
    if (max_surf_resid_K_ <= reported_surf_resid_K_) return;   // nothing new
    reported_surf_resid_K_ = max_surf_resid_K_;
    std::cout << "[GPU Thermal] surface-BC certificate: max achieved "
                 "|f|/(4 eps sigma T^3) = " << max_surf_resid_K_
              << " K (tol_K = 1e-4 K). Values above tol_K are the float32 "
                 "round-off floor of the residual test, not a solver failure -- "
                 "reported so they are never silent.\n";
}

void ThermalOpenCLSolver::createBuffers() {
    cl_context ctx = context_->context();
    cl_int err;

    size_t total = n_faces_ * n_layers_;

    // Allocate device buffers
    buf_T_ = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                           total * sizeof(float), nullptr, &err);
    CL_CHECK(err);

    buf_k_ = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                           total * sizeof(float), nullptr, &err);
    CL_CHECK(err);

    buf_cp_ = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                            total * sizeof(float), nullptr, &err);
    CL_CHECK(err);

    buf_kappa_ = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                               total * sizeof(float), nullptr, &err);
    CL_CHECK(err);

    buf_rho_ = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
                             total * sizeof(float), nullptr, &err);
    CL_CHECK(err);

    buf_kc_ = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
                            total * sizeof(float), nullptr, &err);
    CL_CHECK(err);

    buf_flux_ = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
                              n_faces_ * sizeof(float), nullptr, &err);
    CL_CHECK(err);

    buf_emissivity_ = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
                                    n_faces_ * sizeof(float), nullptr, &err);
    CL_CHECK(err);

    buf_z_ = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
                           n_layers_ * sizeof(float), nullptr, &err);
    CL_CHECK(err);

    buf_T_deep_ = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
                                n_faces_ * sizeof(float), nullptr, &err);
    CL_CHECK(err);

    buf_Qb_ = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
                            n_faces_ * sizeof(float), nullptr, &err);
    CL_CHECK(err);

    // Phase R (R2 fix, Fermi PORT-R2-01): per-face frozen von-Neumann substep
    // count. The persistent kernel derives n_substeps from the session-ENTRY
    // T; a self-chunked session must reuse the SAME count across its
    // sub-dispatches or the integration drifts (the count depends on T, which
    // warms across the session). The kernel writes this on the first
    // sub-dispatch (t_offset==0) and reads it on the rest; the inter-
    // sub-dispatch clFinish makes the write visible. Re-derived at each cycle
    // entry (t_offset resets to 0 per cycle) => identical to pre-R2 behavior.
    buf_n_substeps_frozen_ = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                            n_faces_ * sizeof(int), nullptr, &err);
    CL_CHECK(err);

    // Per-face surface-BC certificate: max achieved |f|/(4 eps sigma T^3) [K].
    // Zeroed at allocation and accumulated as a running max by the kernel; the
    // host reduces it after each dispatch so the float32 round-off floor in the
    // surface Newton can never bind without the run saying so.
    {
        // Zero-initialized at allocation via COPY_HOST_PTR: createBuffers() has
        // no command queue in scope, and the kernel accumulates a running max
        // into it, so a defined starting value is mandatory rather than nice.
        std::vector<float> zeros(n_faces_, 0.0f);
        buf_surf_resid_ = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                         n_faces_ * sizeof(float), zeros.data(), &err);
        CL_CHECK(err);
    }

    // Placeholder for cp_coeffs (will resize when setMaterialParams is called)
    buf_cp_coeffs_ = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
                                   sizeof(float), nullptr, &err);
    CL_CHECK(err);
}

void ThermalOpenCLSolver::uploadMeshData(
    const std::vector<float>& z_grid,
    const std::vector<float>& rho,
    const std::vector<float>& kc,
    const std::vector<float>& emissivity,
    const std::vector<float>& T_deep,
    const std::vector<float>& Qb)
{
    if (!initialized_) {
        throw std::runtime_error("ThermalOpenCLSolver not initialized");
    }

    cl_command_queue queue = context_->queue();

    CL_CHECK(clEnqueueWriteBuffer(queue, buf_z_, CL_FALSE, 0,
                                 z_grid.size() * sizeof(float),
                                 z_grid.data(), 0, nullptr, nullptr));

    CL_CHECK(clEnqueueWriteBuffer(queue, buf_rho_, CL_FALSE, 0,
                                 rho.size() * sizeof(float),
                                 rho.data(), 0, nullptr, nullptr));

    CL_CHECK(clEnqueueWriteBuffer(queue, buf_kc_, CL_FALSE, 0,
                                 kc.size() * sizeof(float),
                                 kc.data(), 0, nullptr, nullptr));

    CL_CHECK(clEnqueueWriteBuffer(queue, buf_emissivity_, CL_FALSE, 0,
                                 emissivity.size() * sizeof(float),
                                 emissivity.data(), 0, nullptr, nullptr));

    CL_CHECK(clEnqueueWriteBuffer(queue, buf_T_deep_, CL_FALSE, 0,
                                 T_deep.size() * sizeof(float),
                                 T_deep.data(), 0, nullptr, nullptr));

    CL_CHECK(clEnqueueWriteBuffer(queue, buf_Qb_, CL_TRUE, 0,
                                 Qb.size() * sizeof(float),
                                 Qb.data(), 0, nullptr, nullptr));
}

void ThermalOpenCLSolver::setMaterialParams(
    float cp0,
    const std::vector<float>& cp_coeffs,
    float R350,
    float sigma,
    int cond_model)
{
    cp0_ = cp0;
    R350_ = R350;
    sigma_ = sigma;
    cond_model_ = cond_model;
    n_cp_coeffs_ = static_cast<int>(cp_coeffs.size());

    if (!cp_coeffs.empty() && initialized_) {
        // Recreate cp_coeffs buffer with correct size
        if (buf_cp_coeffs_) {
            clReleaseMemObject(buf_cp_coeffs_);
        }

        cl_int err;
        buf_cp_coeffs_ = clCreateBuffer(context_->context(), CL_MEM_READ_ONLY,
                                       cp_coeffs.size() * sizeof(float),
                                       nullptr, &err);
        CL_CHECK(err);

        CL_CHECK(clEnqueueWriteBuffer(context_->queue(), buf_cp_coeffs_, CL_TRUE, 0,
                                     cp_coeffs.size() * sizeof(float),
                                     cp_coeffs.data(), 0, nullptr, nullptr));
    }
}

void ThermalOpenCLSolver::uploadInitialTemperature(const std::vector<float>& T_initial) {
    if (!initialized_) {
        throw std::runtime_error("ThermalOpenCLSolver not initialized");
    }

    CL_CHECK(clEnqueueWriteBuffer(context_->queue(), buf_T_, CL_TRUE, 0,
                                 T_initial.size() * sizeof(float),
                                 T_initial.data(), 0, nullptr, nullptr));
}

void ThermalOpenCLSolver::uploadFlux(const std::vector<float>& flux) {
    if (!initialized_) {
        throw std::runtime_error("ThermalOpenCLSolver not initialized");
    }

    CL_CHECK(clEnqueueWriteBuffer(context_->queue(), buf_flux_, CL_TRUE, 0,
                                 flux.size() * sizeof(float),
                                 flux.data(), 0, nullptr, nullptr));
}

void ThermalOpenCLSolver::advanceTimestep(float dt) {
    if (!initialized_) {
        throw std::runtime_error("ThermalOpenCLSolver not initialized");
    }

    auto start = std::chrono::high_resolution_clock::now();

    cl_command_queue queue = context_->queue();
    int n_faces = static_cast<int>(n_faces_active_);
    int n_layers = static_cast<int>(n_layers_);

    // Set kernel arguments
    int arg = 0;
    CL_CHECK(clSetKernelArg(combined_kernel_, arg++, sizeof(cl_mem), &buf_T_));
    CL_CHECK(clSetKernelArg(combined_kernel_, arg++, sizeof(cl_mem), &buf_k_));
    CL_CHECK(clSetKernelArg(combined_kernel_, arg++, sizeof(cl_mem), &buf_cp_));
    CL_CHECK(clSetKernelArg(combined_kernel_, arg++, sizeof(cl_mem), &buf_kappa_));
    CL_CHECK(clSetKernelArg(combined_kernel_, arg++, sizeof(cl_mem), &buf_rho_));
    CL_CHECK(clSetKernelArg(combined_kernel_, arg++, sizeof(cl_mem), &buf_kc_));
    CL_CHECK(clSetKernelArg(combined_kernel_, arg++, sizeof(cl_mem), &buf_flux_));
    CL_CHECK(clSetKernelArg(combined_kernel_, arg++, sizeof(cl_mem), &buf_emissivity_));
    CL_CHECK(clSetKernelArg(combined_kernel_, arg++, sizeof(cl_mem), &buf_z_));
    CL_CHECK(clSetKernelArg(combined_kernel_, arg++, sizeof(cl_mem), &buf_T_deep_));
    CL_CHECK(clSetKernelArg(combined_kernel_, arg++, sizeof(cl_mem), &buf_Qb_));
    CL_CHECK(clSetKernelArg(combined_kernel_, arg++, sizeof(float), &R350_));
    CL_CHECK(clSetKernelArg(combined_kernel_, arg++, sizeof(cl_mem), &buf_cp_coeffs_));
    CL_CHECK(clSetKernelArg(combined_kernel_, arg++, sizeof(int), &n_cp_coeffs_));
    CL_CHECK(clSetKernelArg(combined_kernel_, arg++, sizeof(float), &cp0_));
    CL_CHECK(clSetKernelArg(combined_kernel_, arg++, sizeof(float), &sigma_));
    CL_CHECK(clSetKernelArg(combined_kernel_, arg++, sizeof(float), &dt));
    CL_CHECK(clSetKernelArg(combined_kernel_, arg++, sizeof(int), &n_faces));
    CL_CHECK(clSetKernelArg(combined_kernel_, arg++, sizeof(int), &n_layers));
    CL_CHECK(clSetKernelArg(combined_kernel_, arg++, sizeof(int), &cond_model_));

    // Launch kernel
    size_t global_size = ((n_faces_active_ + config_.work_group_size - 1) /
                          config_.work_group_size) * config_.work_group_size;
    size_t local_size = config_.work_group_size;

    CL_CHECK(clEnqueueNDRangeKernel(queue, combined_kernel_, 1, nullptr,
                                   &global_size, &local_size, 0, nullptr, nullptr));

    // Wait for completion
    CL_CHECK(clFinish(queue));

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    last_kernel_time_ms_ = duration.count() / 1000.0f;
}

void ThermalOpenCLSolver::downloadSurfaceTemps(std::vector<float>& T_surface) {
    if (!initialized_) {
        throw std::runtime_error("ThermalOpenCLSolver not initialized");
    }

    T_surface.resize(n_faces_active_);

    // Surface temps are at layer 0, which is at indices [0, n_faces_active_).
    // (The buffer itself is sized to capacity n_faces_; only the active
    // prefix carries valid data after a kernel launch with the matching
    // n_faces arg.)
    CL_CHECK(clEnqueueReadBuffer(context_->queue(), buf_T_, CL_TRUE, 0,
                                n_faces_active_ * sizeof(float),
                                T_surface.data(), 0, nullptr, nullptr));
}

void ThermalOpenCLSolver::downloadAllTemps(std::vector<float>& T_all) {
    if (!initialized_) {
        throw std::runtime_error("ThermalOpenCLSolver not initialized");
    }

    T_all.resize(n_faces_active_ * n_layers_);

    // SoA layout in the buffer is [layer * n_faces_active_ + face]. The
    // kernel uses n_faces_active_ as the stride too, so the live data is
    // packed in the first n_faces_active_ * n_layers_ floats of the buffer
    // (capacity beyond that is unused junk).
    CL_CHECK(clEnqueueReadBuffer(context_->queue(), buf_T_, CL_TRUE, 0,
                                T_all.size() * sizeof(float),
                                T_all.data(), 0, nullptr, nullptr));
}

void ThermalOpenCLSolver::uploadAllTemps(const std::vector<float>& T_all) {
    if (!initialized_) {
        throw std::runtime_error("ThermalOpenCLSolver not initialized");
    }
    const size_t expected = n_faces_active_ * n_layers_;
    if (T_all.size() != expected) {
        throw std::runtime_error(
            "ThermalOpenCLSolver::uploadAllTemps: size mismatch (got "
            + std::to_string(T_all.size()) + ", expected " + std::to_string(expected) + ")");
    }
    CL_CHECK(clEnqueueWriteBuffer(context_->queue(), buf_T_, CL_TRUE, 0,
                                  T_all.size() * sizeof(float),
                                  T_all.data(), 0, nullptr, nullptr));
}

void ThermalOpenCLSolver::synchronize() {
    if (context_) {
        context_->finish();
    }
}

void ThermalOpenCLSolver::solvePersistentAllTimesteps(
    const std::vector<float>& all_flux,
    std::vector<float>& output_temps,
    int n_timesteps,
    float dt,
    std::vector<float>* avg_E_inout,
    int accum_start,
    int accum_end)
{
    if (!initialized_) {
        std::cerr << "[GPU Thermal] Solver not initialized\n";
        return;
    }

    auto start = std::chrono::high_resolution_clock::now();

    cl_context ctx = context_->context();
    cl_command_queue queue = context_->queue();
    cl_int err;

    // Validate input size
    size_t expected_flux_size = static_cast<size_t>(n_timesteps) * n_faces_active_;
    if (all_flux.size() != expected_flux_size) {
        std::cerr << "[GPU Thermal] all_flux size mismatch: expected "
                  << expected_flux_size << ", got " << all_flux.size() << "\n";
        return;
    }

    // Allocate output
    output_temps.resize(expected_flux_size);

    // Create temporary buffers for all flux data and output temperatures
    cl_mem buf_flux_all = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         all_flux.size() * sizeof(float),
                                         const_cast<float*>(all_flux.data()), &err);
    CL_CHECK(err);

    cl_mem buf_output_temps = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
                                             output_temps.size() * sizeof(float),
                                             nullptr, &err);
    CL_CHECK(err);

    // Set kernel arguments
    int n_faces = static_cast<int>(n_faces_active_);
    int n_layers = static_cast<int>(n_layers_);

    // Verify kernel is valid
    if (!persistent_kernel_) {
        std::cerr << "[GPU Thermal] ERROR: persistent_kernel_ is null!\n";
        clReleaseMemObject(buf_flux_all);
        clReleaseMemObject(buf_output_temps);
        // buf_avg_E is created later, after this point — no release needed here.
        return;
    }

    // Optional Schorghofer accumulator buffer (size n_layers * n_faces_active_).
    // Allocate a real buffer if requested; otherwise a 4-byte dummy so the
    // kernel arg slot is always valid.
    const bool enable_accum = (avg_E_inout != nullptr);
    cl_mem buf_avg_E = nullptr;
    if (enable_accum) {
        if (avg_E_inout->size() != static_cast<size_t>(n_layers_) * n_faces_active_) {
            avg_E_inout->assign(static_cast<size_t>(n_layers_) * n_faces_active_, 0.0f);
        }
        buf_avg_E = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                   avg_E_inout->size() * sizeof(float),
                                   avg_E_inout->data(), &err);
    } else {
        float dummy = 0.0f;
        buf_avg_E = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                   sizeof(float), &dummy, &err);
    }
    CL_CHECK(err);

    int do_accum_int    = enable_accum ? 1 : 0;
    int accum_start_eff = (accum_start < 0) ? 0 : accum_start;
    int accum_end_eff   = (accum_end   < 0) ? n_timesteps : accum_end;

    int arg = 0;
    CL_CHECK(clSetKernelArg(persistent_kernel_, arg++, sizeof(cl_mem), &buf_T_));
    CL_CHECK(clSetKernelArg(persistent_kernel_, arg++, sizeof(cl_mem), &buf_k_));
    CL_CHECK(clSetKernelArg(persistent_kernel_, arg++, sizeof(cl_mem), &buf_cp_));
    CL_CHECK(clSetKernelArg(persistent_kernel_, arg++, sizeof(cl_mem), &buf_kappa_));
    CL_CHECK(clSetKernelArg(persistent_kernel_, arg++, sizeof(cl_mem), &buf_rho_));
    CL_CHECK(clSetKernelArg(persistent_kernel_, arg++, sizeof(cl_mem), &buf_kc_));
    CL_CHECK(clSetKernelArg(persistent_kernel_, arg++, sizeof(cl_mem), &buf_flux_all));
    CL_CHECK(clSetKernelArg(persistent_kernel_, arg++, sizeof(cl_mem), &buf_output_temps));
    CL_CHECK(clSetKernelArg(persistent_kernel_, arg++, sizeof(cl_mem), &buf_emissivity_));
    CL_CHECK(clSetKernelArg(persistent_kernel_, arg++, sizeof(cl_mem), &buf_z_));
    CL_CHECK(clSetKernelArg(persistent_kernel_, arg++, sizeof(cl_mem), &buf_T_deep_));
    CL_CHECK(clSetKernelArg(persistent_kernel_, arg++, sizeof(cl_mem), &buf_Qb_));
    CL_CHECK(clSetKernelArg(persistent_kernel_, arg++, sizeof(float), &R350_));
    CL_CHECK(clSetKernelArg(persistent_kernel_, arg++, sizeof(cl_mem), &buf_cp_coeffs_));
    CL_CHECK(clSetKernelArg(persistent_kernel_, arg++, sizeof(int), &n_cp_coeffs_));
    CL_CHECK(clSetKernelArg(persistent_kernel_, arg++, sizeof(float), &cp0_));
    CL_CHECK(clSetKernelArg(persistent_kernel_, arg++, sizeof(float), &sigma_));
    CL_CHECK(clSetKernelArg(persistent_kernel_, arg++, sizeof(float), &dt));
    CL_CHECK(clSetKernelArg(persistent_kernel_, arg++, sizeof(int), &n_faces));
    CL_CHECK(clSetKernelArg(persistent_kernel_, arg++, sizeof(int), &n_layers));
    CL_CHECK(clSetKernelArg(persistent_kernel_, arg++, sizeof(int), &n_timesteps));
    CL_CHECK(clSetKernelArg(persistent_kernel_, arg++, sizeof(cl_mem), &buf_avg_E));
    CL_CHECK(clSetKernelArg(persistent_kernel_, arg++, sizeof(int), &do_accum_int));
    CL_CHECK(clSetKernelArg(persistent_kernel_, arg++, sizeof(int), &accum_start_eff));
    CL_CHECK(clSetKernelArg(persistent_kernel_, arg++, sizeof(int), &accum_end_eff));
    // Phase R (R2): t_offset arg. Set once here; enqueuePersistentSelfChunked
    // overwrites it per sub-dispatch (kArgTOffset == arg index 25).
    int t_offset_init = 0;
    CL_CHECK(clSetKernelArg(persistent_kernel_, arg++, sizeof(int), &t_offset_init));
    // Phase R (R2 fix, PORT-R2-01): per-face frozen substep-count buffer (arg 26).
    CL_CHECK(clSetKernelArg(persistent_kernel_, arg++, sizeof(cl_mem), &buf_n_substeps_frozen_));
    // Conductivity law (arg 27). Appended AFTER t_offset so kArgTOffset==25 holds.
    CL_CHECK(clSetKernelArg(persistent_kernel_, arg++, sizeof(int), &cond_model_));
    // Per-face surface-BC residual certificate (arg 28), appended last.
    CL_CHECK(clSetKernelArg(persistent_kernel_, arg++, sizeof(cl_mem), &buf_surf_resid_));

    // Launch kernel - one work item per face
    size_t global_size = ((n_faces_active_ + config_.work_group_size - 1) /
                          config_.work_group_size) * config_.work_group_size;
    size_t local_size = config_.work_group_size;

    // Phase R (R2): self-chunk the dispatch to stay under the host watchdog
    // ceiling. n_timesteps <= ceiling degenerates to the pre-R2 single dispatch.
    err = enqueuePersistentSelfChunked(persistent_kernel_, queue, n_timesteps,
                                       gpuMaxTimestepsPerDispatch(),
                                       global_size, local_size);
    if (err != CL_SUCCESS) {
        std::cerr << "[GPU Thermal] Kernel launch/exec failed: " << clErrorString(err) << "\n";
    }

    // Download results
    err = clEnqueueReadBuffer(queue, buf_output_temps, CL_TRUE, 0,
                              output_temps.size() * sizeof(float),
                              output_temps.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "[GPU Thermal] Failed to read output buffer: " << clErrorString(err) << "\n";
    }

    reduceSurfaceBCResidual(queue);
    reportSurfaceBCCertificate();

    // Debug output (only when verbose mode is enabled)
    if (config_.verbose) {
        float min_val = output_temps[0], max_val = output_temps[0];
        int nan_count = 0, zero_count = 0, valid_count = 0;
        for (size_t i = 0; i < std::min(output_temps.size(), (size_t)(n_timesteps * 100)); ++i) {
            float val = output_temps[i];
            if (std::isnan(val)) nan_count++;
            else if (val == 0.0f) zero_count++;
            else {
                valid_count++;
                if (val < min_val) min_val = val;
                if (val > max_val) max_val = val;
            }
        }
        std::cerr << "[GPU Thermal DEBUG] output_temps sample (first 100 faces x " << n_timesteps << " timesteps):\n"
                  << "  NaN count: " << nan_count << ", Zero count: " << zero_count
                  << ", Valid count: " << valid_count << "\n"
                  << "  Valid range: " << min_val << " to " << max_val << " K\n";

        // Show first few values for face 0
        std::cerr << "[GPU Thermal DEBUG] Face 0 temps (first 10 timesteps): ";
        for (int t = 0; t < std::min(10, n_timesteps); ++t) {
            std::cerr << output_temps[t * n_faces_active_ + 0] << " ";
        }
        std::cerr << "\n";

        // Also show later timesteps to track equilibration
        std::cerr << "[GPU Thermal DEBUG] Face 0 temps at later steps: ";
        for (int t : {1000, 2000, 3000, 4000, 5000, 5500, 5800, 5989}) {
            if (t < n_timesteps) {
                std::cerr << "t=" << t << ":" << output_temps[t * n_faces_active_ + 0] << " ";
            }
        }
        std::cerr << "\n";

        // Check flux values for face 0
        std::cerr << "[GPU Thermal DEBUG] Flux sample for face 0: ";
        for (int t : {0, 100, 200, 500, 1000, 2000, 3000}) {
            if (t < n_timesteps) {
                std::cerr << "t=" << t << ":" << all_flux[t * n_faces + 0] << " ";
            }
        }
        std::cerr << "\n";

        // Count zero flux timesteps
        int zero_flux_count = 0;
        for (int t = 0; t < n_timesteps; ++t) {
            if (all_flux[t * n_faces + 0] == 0.0f) zero_flux_count++;
        }
        std::cerr << "[GPU Thermal DEBUG] Face 0: zero flux timesteps = " << zero_flux_count
                  << " / " << n_timesteps << "\n";

        // Read back final T profile to check interior temperatures
        std::vector<float> T_final(n_faces_active_ * n_layers_);
        cl_int read_err = clEnqueueReadBuffer(queue, buf_T_, CL_TRUE, 0,
                                               T_final.size() * sizeof(float),
                                               T_final.data(), 0, nullptr, nullptr);
        if (read_err == CL_SUCCESS) {
            std::cerr << "[GPU Thermal DEBUG] Final T profile for face 0 (all layers): ";
            for (size_t layer = 0; layer < n_layers_; ++layer) {
                std::cerr << T_final[layer * n_faces_active_ + 0] << " ";
            }
            std::cerr << "\n";
        }

        // Debug R350 and other params
        std::cerr << "[GPU Thermal DEBUG] Material params: R350=" << R350_
                  << ", cp0=" << cp0_ << ", sigma=" << sigma_
                  << ", n_cp_coeffs=" << n_cp_coeffs_ << "\n";
    }

    // Read back the Schorghofer accumulator if requested.
    if (enable_accum) {
        cl_int rerr = clEnqueueReadBuffer(queue, buf_avg_E, CL_TRUE, 0,
                                          avg_E_inout->size() * sizeof(float),
                                          avg_E_inout->data(), 0, nullptr, nullptr);
        if (rerr != CL_SUCCESS) {
            std::cerr << "[GPU Thermal] Failed to read avg_E_accum: " << clErrorString(rerr) << "\n";
        }
    }

    // Clean up temporary buffers
    clReleaseMemObject(buf_flux_all);
    clReleaseMemObject(buf_output_temps);
    clReleaseMemObject(buf_avg_E);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    last_kernel_time_ms_ = static_cast<float>(duration.count());

    std::cout << "[GPU Thermal] Persistent kernel processed " << n_timesteps << " timesteps for "
              << n_faces_active_ << " faces in " << last_kernel_time_ms_ << " ms\n";
}

namespace {

// ---------------------------------------------------------------------------
// Pass-stride RRE helpers. Both operate on the SoA device layout
// [layer * n_faces + face] and are pure host arithmetic -- no OpenCL, no
// kernel, nothing that can change when the extrapolation is off.
// ---------------------------------------------------------------------------

// Build T* for every face from the p+1 most recent pass snapshots.
// `ring` is indexed cycle % (p+1); `newest_cycle` says which slot is newest.
// rhat_out[f] = the measured per-face contraction ratio, or NaN for a face
// that was NOT extrapolated (box guard or a failed solve) -- the certificate
// scorer skips those so they cannot pollute the ehat distribution.
bool eqExtrapolateRing(const std::vector<std::vector<float>>& ring,
                       int newest_cycle, size_t n_faces, size_t n_layers,
                       const ptm::eqextrap::Options& opt,
                       std::vector<float>& T_star,
                       std::vector<float>& rhat_out,
                       std::vector<unsigned char>& held_out,
                       ptm::eqextrap::Stats* stats) {
    const int p = opt.order_p;
    const size_t depth = ring.size();
    if (depth != static_cast<size_t>(p) + 1 || n_faces == 0 || n_layers < 2) return false;
    for (const auto& s : ring) {
        if (s.size() != n_faces * n_layers) return false;
    }
    // ordered[s] = S_{newest_cycle - p + s}: oldest first, newest last.
    std::vector<const float*> ordered(depth);
    for (size_t s = 0; s < depth; ++s) {
        const int slot = ptm::eqextrap::ringSlot(newest_cycle, p,
                                                 static_cast<int>(s),
                                                 static_cast<int>(depth));
        ordered[s] = ring[static_cast<size_t>(slot)].data();
    }

    T_star.assign(n_faces * n_layers, 0.0f);
    // rhat is now per face PER BLOCK (N6.2b): a slow deep ACTIVE block must not
    // be masked by a fast surface block in a column-wide norm.
    rhat_out.assign(n_faces * ptm::eqextrap::kNBlocks,
                    std::numeric_limits<float>::quiet_NaN());

    held_out.assign(n_faces * n_layers, 0);
    std::vector<double> X(depth * n_layers);
    std::vector<double> cand(n_layers);
    std::vector<double> s_layer(n_layers);
    std::vector<unsigned char> held(n_layers);
    long long solve_failed = 0, near_degenerate = 0;
    long long converged_window = 0, layers_held = 0, faces_partly_held = 0;
    double d_held_max = 0.0;

    for (size_t f = 0; f < n_faces; ++f) {
        for (size_t s = 0; s < depth; ++s) {
            const float* src = ordered[s];
            for (size_t l = 0; l < n_layers; ++l) {
                X[s * n_layers + l] = static_cast<double>(src[l * n_faces + f]);
            }
        }
        const double* newest = X.data() + static_cast<size_t>(p) * n_layers;

        double cond = 0.0, gamma0 = 0.0;
        const bool ok = ptm::eqextrap::rreColumn(X.data(), p,
                                                 static_cast<int>(n_layers),
                                                 cand.data(), &cond,
                                                 ptm::eqextrap::kRcond,
                                                 &gamma0, s_layer.data());
        bool use_extrap = ok;
        if (!ok) ++solve_failed;

        if (use_extrap) {
            // NEAR-DEGENERATE WINDOW: reported, never acted on.
            {
                double seed_dist = 0.0;
                const double* seed = X.data();      // X[0] = oldest = the seed
                for (size_t l = 0; l < n_layers; ++l)
                    seed_dist = std::max(seed_dist, std::fabs(seed[l] - newest[l]));
                if (ptm::eqextrap::nearDegenerateWindow(gamma0, seed_dist, opt.tol_K))
                    ++near_degenerate;
            }
            // ---- LEVEL 2 (N6.2): per-layer hold on a signal-vs-noise test.
            // A no-op wherever the layer carries signal, so the 55-64x result on
            // the active populations is untouched by construction.
            size_t n_held = 0;
            for (size_t l = 0; l < n_layers; ++l) {
                held[l] = ptm::eqextrap::level2Hold(s_layer[l], opt.hold_K) ? 1 : 0;
                if (held[l]) {
                    cand[l] = newest[l];
                    held_out[l * n_faces + f] = 1;
                    ++n_held;
                }
            }
            layers_held += static_cast<long long>(n_held);
            if (n_held == n_layers)      ++converged_window;
            else if (n_held > 0)         ++faces_partly_held;
            // ---- Displacement tripwire on held layers (N6.2a). Identically 0
            // once Level 2 is in place; it exists to VERIFY the guard, because
            // the residual certificate is irreducibly blind on frozen layers
            // (vanishing one-pass residual AND rhat = 0/0 -> clamp floor).
            for (size_t l = 0; l < n_layers; ++l) {
                if (held[l]) d_held_max = std::max(d_held_max,
                                                   std::fabs(cand[l] - newest[l]));
            }
            // ---- Block-local rhat (N6.2b).
            const double* s1 = X.data() + static_cast<size_t>(p)     * n_layers;
            const double* s2 = X.data() + static_cast<size_t>(p - 1) * n_layers;
            const double* s3 = X.data() + static_cast<size_t>(p - 2) * n_layers;
            double d2[ptm::eqextrap::kNBlocks] = {0, 0, 0};
            double d1[ptm::eqextrap::kNBlocks] = {0, 0, 0};
            for (size_t l = 0; l < n_layers; ++l) {
                const int B = ptm::eqextrap::blockOf(static_cast<int>(l));
                const double a = s1[l] - s2[l], b = s2[l] - s3[l];
                d2[B] += a * a;
                d1[B] += b * b;
            }
            for (int B = 0; B < ptm::eqextrap::kNBlocks; ++B) {
                rhat_out[f * ptm::eqextrap::kNBlocks + B] = static_cast<float>(
                    ptm::eqextrap::clampRhat(std::sqrt(d2[B]), std::sqrt(d1[B]), ptm::eqextrap::kRCap));
            }
            for (size_t l = 0; l < n_layers; ++l) {
                T_star[l * n_faces + f] = static_cast<float>(cand[l]);
            }
        } else {
            // Fall back to the plain state for this face. Uploading S_{N-1}
            // unchanged is exactly what plain cycling would have left on the
            // device, so a refused face is bit-for-bit the plain result.
            for (size_t l = 0; l < n_layers; ++l) {
                T_star[l * n_faces + f] = static_cast<float>(newest[l]);
            }
        }
    }
    if (stats) {
        stats->layers_total      += static_cast<long long>(n_faces) * n_layers;
        stats->solve_failed      += solve_failed;
        stats->near_degenerate   += near_degenerate;
        stats->converged_window  += converged_window;
        stats->layers_held       += layers_held;
        stats->faces_partly_held += faces_partly_held;
        stats->d_held_max         = std::max(stats->d_held_max, d_held_max);
    }
    return true;
}

// ehat = safety * ||Phi(T*) - T*||_inf * rhat/(1-rhat), per face.
void eqScoreCertificate(const std::vector<float>& T_out,
                        const std::vector<float>& T_star,
                        const std::vector<float>& rhat,
                        const std::vector<unsigned char>& held,
                        size_t n_faces, size_t n_layers,
                        const ptm::eqextrap::Options& opt,
                        ptm::eqextrap::Stats* stats) {
    // BLOCK-LOCAL (N6.2b): d_post and rhat are evaluated per depth block
    // [0,15) [15,30) [30,n) and ehat = max_B ehat_B. A column-wide norm lets a
    // fast surface block mask a slow deep ACTIVE block; this stops that.
    if (!stats) return;
    long long certified = 0, refused = 0;
    const int NB = ptm::eqextrap::kNBlocks;
    stats->ehat.reserve(stats->ehat.size() + n_faces);
    for (size_t f = 0; f < n_faces; ++f) {
        if (!std::isfinite(rhat[f * NB])) continue;  // not extrapolated
        double dmax[ptm::eqextrap::kNBlocks] = {0, 0, 0};
        for (size_t l = 0; l < n_layers; ++l) {
            const size_t i = l * n_faces + f;
            const int B = ptm::eqextrap::blockOf(static_cast<int>(l));
            dmax[B] = std::max(dmax[B], std::fabs(static_cast<double>(T_out[i])
                                                - static_cast<double>(T_star[i])));
        }
        // POST-POLISH TRIPWIRE (N8.2), restricted to HELD layers. This is the
        // one with detection power: the T*-based tripwire is structurally zero
        // once the hold fires, so it can only assert that the guard EXECUTED.
        // Restricted to held layers because the 3b population's deep layers move
        // ~0.9 K per pass legitimately -- an all-layers-below-30 version would
        // false-positive on exactly the faces that passed. On a HELD layer the
        // pre-extrapolation state was quiet (<= hold_K), so one more pass may
        // move it by that much plus the deep response to the correctly
        // extrapolated upper column; 0.01 K is 10x that allowance and still
        // catches the observed 0.465 K by 46x.
        {
            double dpost = 0.0;
            bool any_held = false;
            for (size_t l = 0; l < n_layers; ++l) {
                const size_t i = l * n_faces + f;
                if (!held[i]) continue;
                any_held = true;
                dpost = std::max(dpost, std::fabs(static_cast<double>(T_out[i])
                                                - static_cast<double>(T_star[i])));
            }
            if (any_held) stats->d_held_post.push_back(static_cast<float>(dpost));
        }
        double ehat = 0.0;
        for (int B = 0; B < NB; ++B) {
            const double r = static_cast<double>(rhat[f * NB + B]);
            if (!std::isfinite(r)) continue;
            ehat = std::max(ehat, ptm::eqextrap::kSafety * dmax[B] * r / (1.0 - r));
        }
        stats->ehat.push_back(static_cast<float>(ehat));
        if (ehat <= opt.tol_K) ++certified; else ++refused;
    }
    stats->certified    += certified;
    stats->cert_refused += refused;
}

}  // namespace

void ThermalOpenCLSolver::solvePersistentMultipleCycles(
    const std::vector<float>& all_flux,
    std::vector<float>& output_temps,
    int n_timesteps,
    float dt,
    int n_equil_cycles,
    std::vector<float>* avg_E_inout,
    int accum_start,
    int accum_end,
    const ptm::eqextrap::Options* extrap,
    ptm::eqextrap::Stats* extrap_stats)
{
    if (!initialized_) {
        std::cerr << "[GPU Thermal] Solver not initialized\n";
        return;
    }
    if (!persistent_kernel_) {
        std::cerr << "[GPU Thermal] ERROR: persistent_kernel_ is null!\n";
        return;
    }
    if (n_equil_cycles < 0) n_equil_cycles = 0;

    auto start = std::chrono::high_resolution_clock::now();

    cl_context ctx = context_->context();
    cl_command_queue queue = context_->queue();
    cl_int err;

    const size_t expected_flux_size = static_cast<size_t>(n_timesteps) * n_faces_active_;
    if (all_flux.size() != expected_flux_size) {
        std::cerr << "[GPU Thermal] all_flux size mismatch: expected "
                  << expected_flux_size << ", got " << all_flux.size() << "\n";
        return;
    }

    output_temps.resize(expected_flux_size);

    // Upload flux ONCE; reuse for all equilibration cycles plus the output cycle.
    cl_mem buf_flux_all = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         all_flux.size() * sizeof(float),
                                         const_cast<float*>(all_flux.data()), &err);
    CL_CHECK(err);

    // Allocate output buffer ONCE; each cycle overwrites it, only read back the final cycle.
    cl_mem buf_output_temps = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
                                             output_temps.size() * sizeof(float),
                                             nullptr, &err);
    CL_CHECK(err);

    // Optional Schorghofer accumulator (size n_layers * n_faces_active_).
    // Allocate a real buffer if requested; otherwise a 4-byte dummy so the
    // kernel arg slot is always valid.
    const bool enable_accum = (avg_E_inout != nullptr);
    cl_mem buf_avg_E = nullptr;
    if (enable_accum) {
        if (avg_E_inout->size() != static_cast<size_t>(n_layers_) * n_faces_active_) {
            avg_E_inout->assign(static_cast<size_t>(n_layers_) * n_faces_active_, 0.0f);
        }
        buf_avg_E = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                   avg_E_inout->size() * sizeof(float),
                                   avg_E_inout->data(), &err);
    } else {
        float dummy = 0.0f;
        buf_avg_E = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                   sizeof(float), &dummy, &err);
    }
    CL_CHECK(err);

    int accum_start_eff = (accum_start < 0) ? 0 : accum_start;
    int accum_end_eff   = (accum_end   < 0) ? n_timesteps : accum_end;

    // Set kernel arguments ONCE (same for every cycle).
    int n_faces = static_cast<int>(n_faces_active_);
    int n_layers = static_cast<int>(n_layers_);
    int arg = 0;
    CL_CHECK(clSetKernelArg(persistent_kernel_, arg++, sizeof(cl_mem), &buf_T_));
    CL_CHECK(clSetKernelArg(persistent_kernel_, arg++, sizeof(cl_mem), &buf_k_));
    CL_CHECK(clSetKernelArg(persistent_kernel_, arg++, sizeof(cl_mem), &buf_cp_));
    CL_CHECK(clSetKernelArg(persistent_kernel_, arg++, sizeof(cl_mem), &buf_kappa_));
    CL_CHECK(clSetKernelArg(persistent_kernel_, arg++, sizeof(cl_mem), &buf_rho_));
    CL_CHECK(clSetKernelArg(persistent_kernel_, arg++, sizeof(cl_mem), &buf_kc_));
    CL_CHECK(clSetKernelArg(persistent_kernel_, arg++, sizeof(cl_mem), &buf_flux_all));
    CL_CHECK(clSetKernelArg(persistent_kernel_, arg++, sizeof(cl_mem), &buf_output_temps));
    CL_CHECK(clSetKernelArg(persistent_kernel_, arg++, sizeof(cl_mem), &buf_emissivity_));
    CL_CHECK(clSetKernelArg(persistent_kernel_, arg++, sizeof(cl_mem), &buf_z_));
    CL_CHECK(clSetKernelArg(persistent_kernel_, arg++, sizeof(cl_mem), &buf_T_deep_));
    CL_CHECK(clSetKernelArg(persistent_kernel_, arg++, sizeof(cl_mem), &buf_Qb_));
    CL_CHECK(clSetKernelArg(persistent_kernel_, arg++, sizeof(float), &R350_));
    CL_CHECK(clSetKernelArg(persistent_kernel_, arg++, sizeof(cl_mem), &buf_cp_coeffs_));
    CL_CHECK(clSetKernelArg(persistent_kernel_, arg++, sizeof(int), &n_cp_coeffs_));
    CL_CHECK(clSetKernelArg(persistent_kernel_, arg++, sizeof(float), &cp0_));
    CL_CHECK(clSetKernelArg(persistent_kernel_, arg++, sizeof(float), &sigma_));
    CL_CHECK(clSetKernelArg(persistent_kernel_, arg++, sizeof(float), &dt));
    CL_CHECK(clSetKernelArg(persistent_kernel_, arg++, sizeof(int), &n_faces));
    CL_CHECK(clSetKernelArg(persistent_kernel_, arg++, sizeof(int), &n_layers));
    CL_CHECK(clSetKernelArg(persistent_kernel_, arg++, sizeof(int), &n_timesteps));
    CL_CHECK(clSetKernelArg(persistent_kernel_, arg++, sizeof(cl_mem), &buf_avg_E));
    // do_accumulate slot — overridden each cycle below (only the final
    // cycle accumulates).
    int do_accum_zero = 0;
    CL_CHECK(clSetKernelArg(persistent_kernel_, arg++, sizeof(int), &do_accum_zero));
    CL_CHECK(clSetKernelArg(persistent_kernel_, arg++, sizeof(int), &accum_start_eff));
    CL_CHECK(clSetKernelArg(persistent_kernel_, arg++, sizeof(int), &accum_end_eff));
    // Phase R (R2): t_offset arg. Set once here; enqueuePersistentSelfChunked
    // overwrites it per sub-dispatch (kArgTOffset == arg index 25).
    int t_offset_init = 0;
    CL_CHECK(clSetKernelArg(persistent_kernel_, arg++, sizeof(int), &t_offset_init));
    // Phase R (R2 fix, PORT-R2-01): per-face frozen substep-count buffer (arg 26).
    CL_CHECK(clSetKernelArg(persistent_kernel_, arg++, sizeof(cl_mem), &buf_n_substeps_frozen_));
    // Conductivity law (arg 27). Appended AFTER t_offset so kArgTOffset==25 holds.
    CL_CHECK(clSetKernelArg(persistent_kernel_, arg++, sizeof(int), &cond_model_));
    // Per-face surface-BC residual certificate (arg 28), appended last.
    CL_CHECK(clSetKernelArg(persistent_kernel_, arg++, sizeof(cl_mem), &buf_surf_resid_));
    const int kArgIdxDoAccumulate = 22; // index of do_accumulate in the arg list

    size_t global_size = ((n_faces_active_ + config_.work_group_size - 1) /
                          config_.work_group_size) * config_.work_group_size;
    size_t local_size = config_.work_group_size;

    // Phase R (R2): dispatch self-chunk ceiling for THIS session. Each cycle's
    // n_timesteps is split into sub-dispatches of <= max_ts with a clFinish
    // between them, keeping every command buffer under the host watchdog. The
    // split is numerically inert (same CN steps, same order, host sync only);
    // n_timesteps <= max_ts reproduces the pre-R2 single-dispatch-per-cycle
    // path bit-for-bit.
    const int max_ts = gpuMaxTimestepsPerDispatch();

    const int total_cycles = n_equil_cycles + 1;

    // ---- Pass-stride RRE extrapolation (thermal.eq_extrapolation) ----------
    // Snapshots are taken at the TOP of each pass, so at the top of the last
    // pass we hold S_0..S_{N-1}. We extrapolate there and run pass N from T*:
    // that pass is simultaneously the polish, the a-posteriori verification,
    // and the delivered state, which is why the certificate costs zero
    // wall-clock (we were running pass N regardless).
    //
    // COUPLING CONDITION (mandatory). On radiation-coupling iterations >= 2 the
    // eq forcing gains eq_indirect_flux[f], so the fixed point MOVES and
    // snapshots from the previous iteration's map must never be reused. The
    // ring buffer below is a function LOCAL: it is constructed empty on every
    // call and destroyed on return, so it is structurally impossible for a
    // snapshot to survive into another coupling iteration. `ring.assign` below
    // makes that explicit rather than merely true.
    const bool extrap_on = (extrap != nullptr) && extrap->enabled;
    const int  ext_p     = extrap_on ? extrap->order_p : 0;
    const bool ext_ok    = extrap_on && ext_p >= ptm::eqextrap::kMinOrderP
                        && total_cycles >= ext_p + 1 && n_layers_ >= 2;
    const size_t ext_ncol = static_cast<size_t>(n_faces_active_) * n_layers_;
    std::vector<std::vector<float>> ring;        // depth ext_p+1, newest last
    std::vector<float> T_star_host;              // the uploaded extrapolant
    std::vector<float> rhat_host;                // per-face contraction ratio
    std::vector<unsigned char> held_host;        // per face x layer: Level-2 hold mask
    bool extrapolated = false;
    if (extrap_on && extrap_stats) {
        extrap_stats->passes_spent = total_cycles;
        extrap_stats->faces_seen  += static_cast<long long>(n_faces_active_);
        if (!ext_ok) {
            // Not a silent downgrade: the caller has already warned once, and
            // the banner reports the count so the run cannot look extrapolated
            // when it was not.
            extrap_stats->budget_exhausted += static_cast<long long>(n_faces_active_);
        }
    }
    bool ext_live = ext_ok;   // cleared if a readback ever fails, with a warning
    if (ext_live) {
        ring.assign(static_cast<size_t>(ext_p) + 1, std::vector<float>());
    }

    for (int cycle = 0; cycle < total_cycles; ++cycle) {
        if (ext_live) {
            // Snapshot S_cycle from the persistent device state, then -- on the
            // last pass only -- extrapolate and upload T*.
            std::vector<float>& slot = ring[static_cast<size_t>(cycle) % ring.size()];
            bool slot_ok = true;
            try {
                downloadAllTemps(slot);
            } catch (const std::exception& e) {
                std::cerr << "[eq-extrap] WARNING: snapshot readback failed at pass "
                          << cycle << " (" << e.what() << "); running plain cycling\n";
                slot_ok = false;
            }
            if (slot_ok && slot.size() != ext_ncol) {
                std::cerr << "[eq-extrap] WARNING: snapshot size " << slot.size()
                          << " != " << ext_ncol << " at pass " << cycle
                          << "; running plain cycling\n";
                slot_ok = false;
            }
            if (!slot_ok) {
                ext_live = false;
                ring.clear();
                if (extrap_stats) {
                    extrap_stats->budget_exhausted +=
                        static_cast<long long>(n_faces_active_);
                }
            } else if (cycle == total_cycles - 1) {
                // We hold S_0..S_{N-1}. Extrapolate, then upload T* so that the
                // pass about to run starts from it.
                extrapolated = eqExtrapolateRing(ring, cycle, n_faces_active_,
                                                 n_layers_, *extrap,
                                                 T_star_host, rhat_host, held_host,
                                                 extrap_stats);
                if (extrapolated) {
                    try {
                        uploadAllTemps(T_star_host);
                    } catch (const std::exception& e) {
                        std::cerr << "[eq-extrap] WARNING: T* upload failed ("
                                  << e.what() << "); the final pass runs from the"
                                     " plain state instead\n";
                        extrapolated = false;
                        if (extrap_stats) {
                            extrap_stats->budget_exhausted +=
                                static_cast<long long>(n_faces_active_);
                        }
                    }
                }
            }
        }
        // Only accumulate on the FINAL cycle (the "output" cycle). The kernel
        // gates accumulation on the GLOBAL timestep vs [accum_start, accum_end),
        // so a self-chunked final cycle still accumulates each output step
        // exactly once across its sub-dispatches.
        int do_accum_this_cycle = (enable_accum && cycle == total_cycles - 1) ? 1 : 0;
        CL_CHECK(clSetKernelArg(persistent_kernel_, kArgIdxDoAccumulate,
                                 sizeof(int), &do_accum_this_cycle));
        err = enqueuePersistentSelfChunked(persistent_kernel_, queue, n_timesteps,
                                           max_ts, global_size, local_size);
        if (err != CL_SUCCESS) {
            std::cerr << "[GPU Thermal] Kernel launch/exec failed (cycle " << cycle
                      << "): " << clErrorString(err) << "\n";
            break;
        }
    }

    // Certificate. The final pass has just run FROM T*, so buf_T_ now holds
    // Phi(T*) -- the state we are shipping. ehat is evaluated per face against
    // the T* we uploaded, using the contraction ratio measured over the last
    // three snapshots. Nothing here is re-integrated: the "verification pass"
    // is the pass we were going to run anyway.
    if (extrapolated && extrap_stats) {
        std::vector<float> T_out;
        try {
            downloadAllTemps(T_out);
        } catch (const std::exception& e) {
            std::cerr << "[eq-extrap] WARNING: post-polish readback failed ("
                      << e.what() << "); certificate NOT evaluated for this batch\n";
            T_out.clear();
        }
        if (T_out.size() == T_star_host.size()) {
            eqScoreCertificate(T_out, T_star_host, rhat_host, held_host,
                               n_faces_active_, n_layers_, *extrap, extrap_stats);
        } else if (!T_out.empty()) {
            std::cerr << "[eq-extrap] WARNING: post-polish size mismatch ("
                      << T_out.size() << " vs " << T_star_host.size()
                      << "); certificate NOT evaluated for this batch\n";
        }
    }

    // Read output ONCE after the final cycle.
    err = clEnqueueReadBuffer(queue, buf_output_temps, CL_TRUE, 0,
                              output_temps.size() * sizeof(float),
                              output_temps.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "[GPU Thermal] Failed to read output buffer: " << clErrorString(err) << "\n";
    }

    reduceSurfaceBCResidual(queue);
    reportSurfaceBCCertificate();

    // Read back the Schorghofer accumulator if requested.
    if (enable_accum) {
        cl_int rerr = clEnqueueReadBuffer(queue, buf_avg_E, CL_TRUE, 0,
                                          avg_E_inout->size() * sizeof(float),
                                          avg_E_inout->data(), 0, nullptr, nullptr);
        if (rerr != CL_SUCCESS) {
            std::cerr << "[GPU Thermal] Failed to read avg_E_accum: " << clErrorString(rerr) << "\n";
        }
    }

    clReleaseMemObject(buf_flux_all);
    clReleaseMemObject(buf_output_temps);
    clReleaseMemObject(buf_avg_E);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    last_kernel_time_ms_ = static_cast<float>(duration.count());

    std::cout << "[GPU Thermal] Persistent kernel: " << total_cycles << " cycles x "
              << n_timesteps << " timesteps for " << n_faces_active_ << " faces in "
              << last_kernel_time_ms_ << " ms ("
              << (last_kernel_time_ms_ / total_cycles) << " ms/cycle avg)\n";
}

const OpenCLDeviceInfo& ThermalOpenCLSolver::getDeviceInfo() const {
    static OpenCLDeviceInfo empty;
    return context_ ? context_->deviceInfo() : empty;
}

void ThermalOpenCLSolver::release() {
    // Release kernels
    if (material_kernel_) {
        clReleaseKernel(material_kernel_);
        material_kernel_ = nullptr;
    }
    if (thermal_kernel_) {
        clReleaseKernel(thermal_kernel_);
        thermal_kernel_ = nullptr;
    }
    if (combined_kernel_) {
        clReleaseKernel(combined_kernel_);
        combined_kernel_ = nullptr;
    }
    if (persistent_kernel_) {
        clReleaseKernel(persistent_kernel_);
        persistent_kernel_ = nullptr;
    }

    // Release program
    if (program_) {
        clReleaseProgram(program_);
        program_ = nullptr;
    }

    // Release buffers
    if (buf_T_) { clReleaseMemObject(buf_T_); buf_T_ = nullptr; }
    if (buf_k_) { clReleaseMemObject(buf_k_); buf_k_ = nullptr; }
    if (buf_cp_) { clReleaseMemObject(buf_cp_); buf_cp_ = nullptr; }
    if (buf_kappa_) { clReleaseMemObject(buf_kappa_); buf_kappa_ = nullptr; }
    if (buf_rho_) { clReleaseMemObject(buf_rho_); buf_rho_ = nullptr; }
    if (buf_kc_) { clReleaseMemObject(buf_kc_); buf_kc_ = nullptr; }
    if (buf_flux_) { clReleaseMemObject(buf_flux_); buf_flux_ = nullptr; }
    if (buf_emissivity_) { clReleaseMemObject(buf_emissivity_); buf_emissivity_ = nullptr; }
    if (buf_z_) { clReleaseMemObject(buf_z_); buf_z_ = nullptr; }
    if (buf_T_deep_) { clReleaseMemObject(buf_T_deep_); buf_T_deep_ = nullptr; }
    if (buf_Qb_) { clReleaseMemObject(buf_Qb_); buf_Qb_ = nullptr; }
    if (buf_n_substeps_frozen_) { clReleaseMemObject(buf_n_substeps_frozen_); buf_n_substeps_frozen_ = nullptr; }
    if (buf_surf_resid_) { clReleaseMemObject(buf_surf_resid_); buf_surf_resid_ = nullptr; }
    if (buf_cp_coeffs_) { clReleaseMemObject(buf_cp_coeffs_); buf_cp_coeffs_ = nullptr; }

    // Release context (via unique_ptr)
    context_.reset();

    initialized_ = false;
}

}  // namespace opencl
}  // namespace gpu

#endif  // PTM_GPU_OPENCL
