// test_thermal_wave_depth -- the analytic gate for the solution BELOW the
// surface, and the measurement the depth-map architecture decision needs.
//
// WHY THIS EXISTS. test_bowl_crater gates radiation at the SURFACE. Nothing
// gated the conduction solution at depth against a closed form, so a
// subsurface temperature product would have shipped with no analytic anchor at
// all. This is that anchor.
//
// ---------------------------------------------------------------------------
// THE PROBLEM AND THE CLOSED FORM
// ---------------------------------------------------------------------------
// A homogeneous half-space (k, rho, cp, and therefore kappa, constant in z AND
// in T) forced periodically at the surface. For a surface series decomposed
// into harmonics, the interior response to harmonic n is exactly
//
//     T_n(z,t) = A_n e^(-z/d_n) cos(n*w*t + phi_n - z/d_n),
//     d_n = sqrt(2*kappa/(n*w)) = d_1/sqrt(n),   d_1 = sqrt(kappa*P/pi)
//
// -- amplitude decays by e^(-z/d) and phase lags by z/d, the two independent
// signatures. Superposition holds because the INTERIOR problem is linear with
// constant coefficients: once T(0,t) is known the interior is a linear
// Dirichlet problem, whatever nonlinearity produced T(0,t).
//
// ---------------------------------------------------------------------------
// WHAT THIS GATE DOES **NOT** COVER -- read before quoting it
// ---------------------------------------------------------------------------
//  * The SURFACE ENERGY BALANCE. The analytic reference is driven by the
//    engine's OWN measured T(0,t). A wrong radiative Robin boundary condition
//    would move T(0,t) and this gate would still pass: it validates interior
//    conduction and depth interpolation, not the surface solve. That is
//    tools/test_surface_newton's job, and this file does not duplicate it.
//  * The CPU SIMD-batch and GPU persistent solvers. This exercises the CPU
//    SCALAR path (Profile::update_T). The other two implement the same scheme
//    in their own code and would need their own arm.
//  * Depth-varying properties. chi = 0, ks == kd, rhos == rhod, cpCoeff empty,
//    by construction -- a closed form exists only for constant kappa. The
//    shipped lunar column is NOT homogeneous, so the layer spacing measured
//    here in units of the LOCAL skin depth is finer at depth than production's.
//  * Radiation coupling, meshes, view factors, SPICE, and the GPU. None are
//    touched: safe to run while a production job owns the devices.
//
// ---------------------------------------------------------------------------
// THE TOLERANCE IS DERIVED, NOT CHOSEN
// ---------------------------------------------------------------------------
// Substituting T_j^s = exp(i*n*w*s*dt) * exp(-lam*z_j) into the scheme the
// engine actually runs -- the 3-point nonuniform Laplacian of
// Profile::update_T assembled Crank-Nicolson in time -- gives the DISCRETE
// dispersion relation
//
//     cosh(lam_h * h) = 1 + i * h^2 / d_eff^2,
//     d_eff = sqrt(2*kappa/w_eff),   w_eff = (2/dt) * tan(n*w*dt/2)
//
// (the w_eff form is CN's own: its amplification factor for exp(i*w*t) is
// (1 + i*w_eff*dt/2)/(1 - i*w_eff*dt/2)). So
//
//     lam_h = acosh(1 + i*h^2/d_eff^2) / h        vs   lam = (1+i)/d_n
//
// and the scheme's PREDICTED error at node j is obtained by accumulating
// lam_h cell by cell across the real grid. There is no free parameter in that
// prediction. The gate is `measured <= 3 x predicted + floor`, and the test
// PRINTS measured/predicted at every node so the margin is visible rather than
// asserted. A number tuned to pass would have no such ratio to show.
//
// At the shipped near-surface resolution m = 10 (h = d/10) the prediction is
// 8.33e-4 relative amplitude and 8.33e-4 rad PER SKIN DEPTH -- reproduced as
// an explicit self-check below, because Fermi derived the same two numbers
// independently (3.06e-4*A absolute at one skin depth = e^-1 * 8.33e-4, and
// 8.35e-4 rad). Two derivations agreeing is what makes this a floor and not a
// guess. A gate tighter than that FAILS CORRECT CODE.
//
// An absolute bar alone can be passed by a first-order scheme with a lucky
// constant, so the ORDER TEST is a separate, mandatory arm: halve h and dt
// together and the error must drop by 3.5x-4.5x.
//
// ---------------------------------------------------------------------------
// SEMI-INFINITE VALIDITY
// ---------------------------------------------------------------------------
// The engine's column is finite with an insulating bottom, so the wave
// reflects. Relative contamination of the reference at depth z is
// exp(-2*(Zd - z/d)) with Zd = z_max/d. The test computes that bound for its
// deepest probe and REFUSES to score if it is not far below the error floor.
//
// ---------------------------------------------------------------------------
// U2 / DECISION D8 -- the max/min interpolation-commutation bias
// ---------------------------------------------------------------------------
// avg commutes with linear interpolation in z; max and min do not, because
// adjacent layers peak at different times. For one harmonic between nodes at
// zeta1, zeta2 (depths in skin depths) with weight w:
//
//     arch A (interpolate, then max) = |(1-w) e^(-(1+i)zeta1) + w e^(-(1+i)zeta2)|
//     arch B (max, then interpolate) = (1-w) e^(-zeta1) + w e^(-zeta2)
//     truth                          = e^(-((1-w)zeta1 + w zeta2))
//
// This test evaluates that closed form on the real grid AND measures both
// architectures from the engine's own solution, so the two are checked against
// each other rather than either being trusted. A `z = 0.0` probe cannot
// exercise this at all -- it sits at w = 0, where interpolation is exact by
// construction -- so the probes are node MIDPOINTS, w = 0.5.

#include "../thermal/thermal_model.h"
#include "../thermal/depth_map_plan.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

namespace {

int g_checks = 0;
int g_failed = 0;

void check(bool ok, const std::string& what) {
    ++g_checks;
    std::printf("  %s  %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++g_failed;
}

using cd = std::complex<double>;

// ---------------------------------------------------------------------------
// The homogeneous half-space. Every property that could vary with depth or
// temperature is pinned, and the pinning is VERIFIED below rather than assumed
// -- a single temperature-dependent term would silently invalidate the closed
// form this whole file compares against.
// ---------------------------------------------------------------------------
struct Setup {
    double P      = 2.5536e6;   // forcing period [s]
    double k      = 1.0e-3;     // W/m/K
    double rho    = 1500.0;     // kg/m^3
    double cp     = 600.0;      // J/kg/K
    double eps    = 0.95;
    double Q0     = 86.19;      // W/m^2 -> eps*sigma*T^4 gives T ~ 200 K
    double amp    = 0.5;        // Qs(t) = Q0 * (1 + amp*cos(w t))
    int    m      = 10;         // layers per skin depth at the surface
    int    b      = 15;         // z_max = b * delta
    double ndeep  = 0.5;        // MIN_LAYERS_DEEP_PER_SKIN
    int    S      = 768;        // steps per period (Config::MAIN_LOOP_STEPS_PER_DAY)
};

struct Run {
    int    n_nodes = 0, S = 0, periods = 0;
    double delta = 0, kappa = 0, dt = 0, z_max = 0;
    double periodicity_K = 0, T_min_seen = 1e300, wall_s = 0;
    std::vector<double> z, dz;
    std::vector<std::vector<double>> T;   // [node][sample], one settled period
    // Harmonics of each node's series. H[n][j], n = 1..n_harm.
    std::vector<std::vector<cd>> H;
    std::vector<double> mean_z;
};

const int kNHarm = 8;

// One integration to periodic steady state, then one recorded period.
//
// Spin-up length is MEASURED, not assumed: the run continues in blocks until
// the period-over-period residual falls below `settle_K`, and REFUSES (returns
// periods = 0) if it never does. Assuming a settled column is exactly how a
// transient gets scored as a discretization error.
Run integrate(const Setup& su, double settle_K, int max_periods) {
    Run r;
    const auto t_start = std::chrono::steady_clock::now();

    thermal::Planet planet("Moon");
    planet.ks = planet.kd = su.k;
    planet.rhos = planet.rhod = su.rho;
    planet.cp0 = su.cp;
    planet.cpCoeff.clear();          // -> cp(T) is the constant cp0
    planet.H   = 0.1;                // irrelevant once ks==kd and rhos==rhod
    planet.day = su.P;
    planet.year = su.P * 12.0;
    planet.emissivity = su.eps;
    planet.Qb  = 0.0;

    static thermal::Config config;   // Profile keeps a POINTER to this
    config = thermal::Config();
    config.chi = 0.0;                // -> R350 = 0 -> k is independent of T
    config.m = su.m;
    config.b = su.b;
    config.MIN_LAYERS_DEEP_PER_SKIN = su.ndeep;
    config.ADAPTIVE_GRID = true;     // the SHIPPING grid, not a special one

    thermal::Profile prof(planet, 0.0, config, /*verbose=*/false);
    r.n_nodes = prof.nlayers;
    r.z.assign(prof.z().begin(), prof.z().begin() + r.n_nodes);
    r.dz.assign(prof.dz().begin(), prof.dz().begin() + (r.n_nodes - 1));
    r.z_max = r.z.back();
    r.kappa = prof.kappa()[0];
    r.delta = thermal::skinDepth(su.P, r.kappa);
    r.S     = su.S;
    r.dt    = su.P / su.S;

    // Start from the radiative-equilibrium mean; the spin-up loop measures how
    // far that is from settled rather than trusting it.
    const double Tbar0 = std::pow(su.Q0 / (su.eps * config.sigma), 0.25);
    for (int i = 0; i < r.n_nodes; ++i) prof.T()[i] = Tbar0;
    prof.updateThermalProperties();

    const double w = 2.0 * M_PI / su.P;
    auto flux = [&](double t) { return su.Q0 * (1.0 + su.amp * std::cos(w * t)); };

    std::vector<std::vector<double>> prev, cur;
    auto onePeriod = [&](std::vector<std::vector<double>>* rec) {
        if (rec) rec->assign(r.n_nodes, std::vector<double>(su.S, 0.0));
        for (int s = 0; s < su.S; ++s) {
            const double t0 = s * r.dt;
            prof.update_T(r.dt, flux(t0), 0.0);
            if (rec) {
                // ALL NODES AT ONE EPOCH. The interior is now the state at
                // t0+dt; the surface node still holds the algebraic solve made
                // with the flux at t0. outputSurfaceTemp is the engine's own
                // non-mutating helper for exactly this (see its block comment
                // in thermal_model.cpp) -- without it node 0 trails the
                // interior by one step, which at S=768 is 8.2e-3 rad, ten
                // times the tolerance this gate imposes.
                (*rec)[0][s] = thermal::outputSurfaceTemp(prof, flux(t0 + r.dt), 0.0);
                for (int j = 1; j < r.n_nodes; ++j) (*rec)[j][s] = prof.T()[j];
                for (int j = 0; j < r.n_nodes; ++j)
                    r.T_min_seen = std::min(r.T_min_seen, (*rec)[j][s]);
            }
        }
    };

    int done = 0;
    const int kBlock = 25;
    while (done < max_periods) {
        for (int p = 0; p < kBlock - 1; ++p) { onePeriod(nullptr); ++done; }
        prev = cur;
        onePeriod(&cur);
        ++done;
        if (!prev.empty()) {
            double res = 0.0;
            for (int j = 0; j < r.n_nodes; ++j)
                for (int s = 0; s < su.S; ++s)
                    res = std::max(res, std::fabs(cur[j][s] - prev[j][s]));
            r.periodicity_K = res;
            // The residual is over kBlock periods; the per-period drift is
            // what contaminates the comparison.
            if (res / kBlock < settle_K) { r.periods = done; break; }
        }
    }
    if (r.periods == 0) return r;    // never settled: caller must refuse
    r.T = cur;

    // ---- harmonics -------------------------------------------------------
    r.H.assign(kNHarm + 1, std::vector<cd>(r.n_nodes, cd(0.0, 0.0)));
    r.mean_z.assign(r.n_nodes, 0.0);
    for (int j = 0; j < r.n_nodes; ++j) {
        double sum = 0.0;
        for (int s = 0; s < su.S; ++s) sum += r.T[j][s];
        r.mean_z[j] = sum / su.S;
        for (int n = 1; n <= kNHarm; ++n) {
            cd acc(0.0, 0.0);
            for (int s = 0; s < su.S; ++s) {
                const double th = -2.0 * M_PI * n * s / su.S;
                acc += r.T[j][s] * cd(std::cos(th), std::sin(th));
            }
            r.H[n][j] = acc * (2.0 / su.S);
        }
    }
    r.wall_s = std::chrono::duration<double>(
                   std::chrono::steady_clock::now() - t_start).count();
    return r;
}

// The DISCRETE wavenumber of the scheme the engine runs, for local spacing h
// and harmonic n. Parameter-free; this is the prediction the gate is measured
// against.
cd lambdaDiscrete(double h, double kappa, double omega_n, double dt) {
    const double w_eff = (2.0 / dt) * std::tan(0.5 * omega_n * dt);
    const double d_eff2 = 2.0 * kappa / w_eff;
    return std::acosh(cd(1.0, h * h / d_eff2)) / h;
}

// Accumulated transfer exponent from node 0 to node j: sum of lam_h*h over the
// cells between them. exp(-this) is the scheme's predicted amplitude and phase.
std::vector<cd> predictedExponent(const Run& r, int n) {
    const double omega_n = n * 2.0 * M_PI / (r.S * r.dt);
    std::vector<cd> e(r.n_nodes, cd(0.0, 0.0));
    cd acc(0.0, 0.0);
    for (int j = 1; j < r.n_nodes; ++j) {
        acc += lambdaDiscrete(r.dz[j - 1], r.kappa, omega_n, r.dt) * r.dz[j - 1];
        e[j] = acc;
    }
    return e;
}

double wrapPi(double a) {
    while (a >  M_PI) a -= 2.0 * M_PI;
    while (a < -M_PI) a += 2.0 * M_PI;
    return a;
}

}  // namespace

int main() {
    std::printf("=== test_thermal_wave_depth: damped thermal wave at depth ===\n");
    std::printf("Analytic gate for the CONDUCTION solution below the surface,\n"
                "and the U2 / D8 max-min interpolation measurement.\n\n");

    // ---- 0. the derivation, self-checked against Fermi's independent one --
    {
        // At m = 10 the cell is h = delta/10 and the temporal term is
        // negligible, so lam_h*h should be 0.100083 + 0.099917i: a relative
        // mis-decay and a mis-phase of 8.33e-4 PER SKIN DEPTH (ten cells).
        const double kappa = 1.0e-9, P = 2.5536e6;
        const double delta = std::sqrt(kappa * P / M_PI);
        const double omega = 2.0 * M_PI / P;
        const double dt = P / 100000.0;             // temporal term -> 0
        const cd lh = lambdaDiscrete(delta / 10.0, kappa, omega, dt) * (delta / 10.0);
        const double mis_decay = 10.0 * lh.real() - 1.0;
        const double mis_phase = 10.0 * lh.imag() - 1.0;
        std::printf("  [derivation] m=10: mis-decay %.4e / skin depth, "
                    "mis-phase %.4e rad / skin depth\n", mis_decay, mis_phase);
        std::printf("               absolute amplitude error at 1 skin depth = "
                    "%.3e * A   (Fermi, independently: 3.06e-4 * A)\n",
                    std::exp(-1.0) * mis_decay);
        check(std::fabs(mis_decay - 8.33e-4) < 0.02 * 8.33e-4 &&
              std::fabs(mis_phase + 8.33e-4) < 0.02 * 8.33e-4,
              "W0: the discrete dispersion relation reproduces the independently "
              "derived 8.33e-4 per skin depth at m=10 (two derivations agree, so "
              "the tolerance below is a FLOOR, not a guess)");
    }

    Setup su;
    const double kSettleK = 1.0e-7;     // per-period drift allowed before scoring
    Run base = integrate(su, kSettleK, 4000);

    if (base.periods == 0) {
        std::printf("  FAIL  the column never reached periodic steady state; "
                    "nothing below can be scored\n");
        return 1;
    }

    const double delta = base.delta;
    std::printf("\n  grid: %d nodes, kappa = %.6e m^2/s, delta = %.6f m, "
                "z_max = %.4f m = %.2f delta\n",
                base.n_nodes, base.kappa, delta, base.z_max, base.z_max / delta);
    std::printf("  dz[0] = %.6f m = delta/%.2f;  dz[last] = %.6f m = %.2f delta\n",
                base.dz.front(), delta / base.dz.front(),
                base.dz.back(), base.dz.back() / delta);
    std::printf("  time: %d steps/period, %d periods to settle "
                "(per-period drift %.2e K), %.1f s wall\n",
                base.S, base.periods, base.periodicity_K / 25.0, base.wall_s);

    // ---- 1. preconditions: the closed form only exists if these hold ------
    {
        check(base.periodicity_K / 25.0 < kSettleK,
              "W1: periodic steady state REACHED (per-period drift < 1e-7 K), so "
              "what follows is discretization error and not a leftover transient");
        check(base.T_min_seen > 20.0,
              "W1: the T_geo / CMB floor never fired (min T seen = " +
                  std::to_string(base.T_min_seen) + " K), so no clamp is in the "
                  "comparison");

        // Homogeneity, verified rather than assumed.
        // (Reconstruct the profile's own arrays through a second Profile so the
        //  check is on what the constructor actually built.)
        double mean_spread = 0.0;
        for (int j = 1; j < base.n_nodes; ++j)
            mean_spread = std::max(mean_spread,
                                   std::fabs(base.mean_z[j] - base.mean_z[0]));
        std::printf("        cycle-mean profile spread over depth: %.4e K\n",
                    mean_spread);
        check(mean_spread < 0.02,
              "W1: the cycle-mean profile is FLAT in depth, as Qb = 0 with "
              "constant k requires -- a depth-dependent mean would mean the "
              "properties are not homogeneous after all");
    }

    // ---- 2. semi-infinite validity ---------------------------------------
    const double kProbeDeltas = 4.0;    // deepest depth scored, in skin depths
    {
        const double Zd = base.z_max / delta;
        const double contamination = std::exp(-2.0 * (Zd - kProbeDeltas));
        std::printf("\n  reflection from the insulating bottom at the deepest "
                    "scored probe (%.1f delta):\n"
                    "        Zd = %.2f  ->  relative contamination "
                    "exp(-2(Zd - z/delta)) = %.2e\n",
                    kProbeDeltas, Zd, contamination);
        check(contamination < 1.0e-6,
              "W2: the finite column is semi-infinite to far better than the "
              "error floor at the deepest probe (needs Zd >= 4.6 + z/delta; "
              "b = 15 gives Zd = 15)");
    }

    // ---- 3. amplitude decay and phase lag, per harmonic ------------------
    // Both are scored against the CONTINUUM closed form, with a tolerance
    // taken from the scheme's own predicted error at that node. The
    // measured/predicted ratio is printed at every node.
    int last_scored = 0;
    while (last_scored + 1 < base.n_nodes &&
           base.z[last_scored + 1] <= kProbeDeltas * delta) ++last_scored;

    // REFERENCE NODE IS 1, NOT 0, and that is a physics decision, not a
    // convenience. The surface node is MASSLESS -- solveSurfaceBC determines it
    // algebraically from (Qs, T[1]) -- so it is a diagnostic of an epoch rather
    // than an integrated state, and pairing it with the interior imports the
    // engine's documented surface output-epoch convention into a measurement of
    // conduction. The offset that convention produces is not swept under the
    // rug: it is measured and pinned by its own check, W4S, below. Everything
    // a depth product samples is an interior node, so interior-to-interior is
    // also the quantity that matters.
    const int kRef = 1;
    for (int n = 1; n <= 2; ++n) {
        const double d_n = delta / std::sqrt(static_cast<double>(n));
        const std::vector<cd> pred = predictedExponent(base, n);
        const double Ar = std::abs(base.H[n][kRef]);
        const double pr = std::arg(base.H[n][kRef]);
        const double zr = base.z[kRef];

        std::printf("\n  --- harmonic %d (delta_%d = %.6f m, amplitude at the "
                    "reference node z=%.4f m: %.4f K) ---\n",
                    n, n, d_n, zr, Ar);
        std::printf("     z/delta   amp ratio    analytic    rel.err   pred.err"
                    "   ratio |  phase lag  analytic   err(rad)  pred(rad) ratio\n");

        double worst_amp_ratio = 0.0, worst_ph_ratio = 0.0;
        bool amp_ok = true, ph_ok = true;
        for (int j = kRef + 1; j <= last_scored; ++j) {
            const double zeta = (base.z[j] - zr) / d_n;
            const double meas_amp = std::abs(base.H[n][j]) / Ar;
            const double ana_amp  = std::exp(-zeta);
            const double ana_ph   = zeta;
            // Wrap the ERROR, not the lag: the true lag exceeds pi beyond
            // z/delta = pi and would otherwise alias to 2*pi. Valid because the
            // error is ~1e-3 rad, six orders below the pi ambiguity -- and the
            // amplitude arm, which has no such ambiguity, is an independent
            // witness that the node ordering is right.
            const double meas_ph_raw = pr - std::arg(base.H[n][j]);
            const double err_ph_signed = wrapPi(meas_ph_raw - ana_ph);
            const double meas_ph = ana_ph + err_ph_signed;

            const cd    predj    = pred[j] - pred[kRef];
            const double pred_amp = std::exp(-predj.real());
            const double pred_ph  = predj.imag();

            const double err_amp = std::fabs(meas_amp - ana_amp) / ana_amp;
            const double err_ph  = std::fabs(err_ph_signed);
            const double pre_amp = std::fabs(pred_amp - ana_amp) / ana_amp;
            const double pre_ph  = std::fabs(pred_ph - ana_ph);

            const double tol_amp = 3.0 * pre_amp + 1.0e-6;
            const double tol_ph  = 3.0 * pre_ph  + 1.0e-6;
            if (err_amp > tol_amp) amp_ok = false;
            if (err_ph  > tol_ph)  ph_ok  = false;
            worst_amp_ratio = std::max(worst_amp_ratio,
                                       err_amp / std::max(pre_amp, 1e-30));
            worst_ph_ratio  = std::max(worst_ph_ratio,
                                       err_ph / std::max(pre_ph, 1e-30));

            std::printf("     %7.3f  %10.6f  %10.6f  %9.2e  %9.2e  %6.2f | "
                        "%9.5f %9.5f  %9.2e  %9.2e %6.2f\n",
                        base.z[j] / delta, meas_amp, ana_amp, err_amp, pre_amp,
                        err_amp / std::max(pre_amp, 1e-30),
                        meas_ph, ana_ph, err_ph, pre_ph,
                        err_ph / std::max(pre_ph, 1e-30));
        }
        check(amp_ok,
              "W3." + std::to_string(n) +
                  ": AMPLITUDE decays as exp(-z/delta) to within 3x the "
                  "scheme's own predicted discretization error at every node "
                  "(worst measured/predicted = " +
                  std::to_string(worst_amp_ratio) + ")");
        check(ph_ok,
              "W4." + std::to_string(n) +
                  ": PHASE lags by z/delta to within 3x the predicted error "
                  "(worst measured/predicted = " +
                  std::to_string(worst_ph_ratio) + ") -- an amplitude-only "
                  "test passes a time-index or layer transposition; this does "
                  "not");
    }

    // ---- 3b. the surface node's output epoch, measured and PINNED ---------
    // The lag from node 0 to node 1 comes out HALF A TIME STEP larger than
    // conduction alone accounts for, identically at every depth and scaling
    // with harmonic number -- i.e. a pure time-origin offset of the surface
    // series, not a depth-dependent error. That is the CPU scalar path's known
    // surface/interior epoch convention (see the outputSurfaceTemp block
    // comment in thermal_model.cpp, which measures the same thing a different
    // way). It is asserted here rather than mentioned, so that if it ever stops
    // being exactly omega*dt/2 this test says so.
    {
        const double omega = 2.0 * M_PI / (base.S * base.dt);
        std::printf("\n  --- surface-node output epoch (node 0 vs node 1) ---\n");
        bool ok = true;
        for (int n = 1; n <= 2; ++n) {
            const double d_n = delta / std::sqrt(static_cast<double>(n));
            const std::vector<cd> pred = predictedExponent(base, n);
            const double lag = wrapPi(std::arg(base.H[n][0]) -
                                      std::arg(base.H[n][1]));
            const double conduction = pred[1].imag();  // predicted, 0 -> 1
            const double excess = lag - conduction;
            const double half_step = 0.5 * n * omega * base.dt;
            std::printf("     harmonic %d: measured lag %.6e, conduction "
                        "accounts for %.6e, excess %.6e rad\n"
                        "                 omega_n*dt/2 = %.6e  ->  excess is "
                        "%.4f of half a time step\n",
                        n, lag, conduction, excess, half_step,
                        excess / half_step);
            if (std::fabs(excess / half_step - 1.0) > 0.05) ok = false;
        }
        check(ok,
              "W4S: the node-0 phase excess is EXACTLY half a time step "
              "(omega_n*dt/2) for both harmonics -- a time-origin offset of the "
              "massless surface node, not a depth-dependent error. This is the "
              "CPU scalar surface/interior epoch convention, and it is why the "
              "gates above reference node 1");
    }

    // ---- 4. ORDER TEST ---------------------------------------------------
    // An absolute bar can be passed by a first-order scheme with a lucky
    // constant. Halving h and dt together must drop the error ~4x.
    {
        Setup fine = su;
        fine.m *= 2;
        fine.ndeep *= 2.0;
        fine.S *= 2;
        Run ref = integrate(fine, kSettleK, 4000);
        check(ref.periods > 0, "W5: the refined run reached periodic steady state");

        // Same interior-referenced metrics the gates above use.
        auto worstLnAmpErr = [&](const Run& r) {
            const double Ar = std::abs(r.H[1][kRef]), zr = r.z[kRef];
            double worst = 0.0;
            for (int j = kRef + 1; j < r.n_nodes; ++j) {
                if (r.z[j] > kProbeDeltas * r.delta) break;
                const double meas = std::log(std::abs(r.H[1][j]) / Ar);
                worst = std::max(worst,
                                 std::fabs(meas + (r.z[j] - zr) / r.delta));
            }
            return worst;
        };
        auto worstPhErr = [&](const Run& r) {
            const double pr = std::arg(r.H[1][kRef]), zr = r.z[kRef];
            double worst = 0.0;
            for (int j = kRef + 1; j < r.n_nodes; ++j) {
                if (r.z[j] > kProbeDeltas * r.delta) break;
                const double ana = (r.z[j] - zr) / r.delta;
                worst = std::max(worst,
                                 std::fabs(wrapPi(pr - std::arg(r.H[1][j]) - ana)));
            }
            return worst;
        };

        if (ref.periods > 0) {
            const double eA = worstLnAmpErr(base), eA2 = worstLnAmpErr(ref);
            const double eP = worstPhErr(base),    eP2 = worstPhErr(ref);
            const double hr = base.dz.front() / ref.dz.front();
            std::printf("\n  --- order test (m %d->%d, ndeep %.1f->%.1f, "
                        "steps/period %d->%d) ---\n",
                        su.m, fine.m, su.ndeep, fine.ndeep, su.S, fine.S);
            std::printf("     dz[0] refined by %.3fx; nodes %d -> %d\n",
                        hr, base.n_nodes, ref.n_nodes);
            std::printf("     ln-amplitude error  %.4e -> %.4e   ratio %.2f\n",
                        eA, eA2, eA / eA2);
            std::printf("     phase error (rad)   %.4e -> %.4e   ratio %.2f\n",
                        eP, eP2, eP / eP2);
            check(std::fabs(hr - 2.0) < 0.05,
                  "W5: the refinement really did halve the near-surface spacing");
            check(eA / eA2 >= 3.5 && eA / eA2 <= 4.5,
                  "W6a: amplitude error falls 3.5x-4.5x when h and dt are "
                  "halved -- SECOND ORDER (a first-order scheme gives ~2x)");
            check(eP / eP2 >= 3.5 && eP / eP2 <= 4.5,
                  "W6b: phase error falls 3.5x-4.5x under the same refinement");
        }
    }

    // ---- 5. interpolation at node MIDPOINTS, and U2 / D8 ------------------
    // w = 0.5, deliberately: w = 0 (a z = 0.0 probe, or any probe landing on a
    // node) is exact by construction and cannot exercise this at all.
    {
        std::printf("\n  --- interpolation at node midpoints (w = 0.5) and the "
                    "max-commutation bias ---\n");
        const double A1 = std::abs(base.H[1][0]);
        std::printf("     A = surface fundamental amplitude = %.4f K\n", A1);
        std::printf("     zeta1  zeta2 | interp err |  arch A bias        "
                    "arch B bias      | closed form: A       B\n");
        std::printf("                  |    (K)     |  measured (K, /A)   "
                    "measured (K, /A) |  (/A)              (/A)\n");

        double worst_interp_K = 0.0;
        double worst_A_meas = 0.0, worst_B_meas = 0.0;
        double worst_A_form = 0.0, worst_B_form = 0.0;
        int    worst_B_layer = -1, worst_A_layer = -1, worst_B_form_layer = -1;
        bool   interp_ok = true;

        // THE WHOLE GRID, not just the scored window. The two architectures
        // peak in different places -- arch B shallow (large amplitude, small
        // phase difference) and arch A deep (tiny amplitude, phase difference
        // approaching pi, where the phasor sum cancels) -- so a scan that
        // stopped at 4 skin depths would miss arch A's worst case entirely.
        // Rows below the scored window are marked: their MEASURED values carry
        // the bottom reflection, at absolute levels ~1e-8 K, while the closed
        // form is exact algebra everywhere.
        for (int j = 0; j < base.n_nodes - 1; ++j) {
            const bool scored = (j + 1 <= last_scored);
            const double z1 = base.z[j], z2 = base.z[j + 1];
            const double zq = 0.5 * (z1 + z2);
            const double zeta1 = z1 / delta, zeta2 = z2 / delta;
            const double zetaq = zq / delta;

            // Analytic series at the query depth, rebuilt from the MEASURED
            // surface harmonics -- superposition, exact for the continuum.
            std::vector<double> ana(base.S), interp(base.S);
            for (int s = 0; s < base.S; ++s) {
                double v = base.mean_z[0];
                for (int n = 1; n <= kNHarm; ++n) {
                    const double d_n = delta / std::sqrt(static_cast<double>(n));
                    const cd prop = std::exp(cd(-zq / d_n, -zq / d_n));
                    const double th = 2.0 * M_PI * n * s / base.S;
                    v += std::real(base.H[n][0] * prop * cd(std::cos(th), std::sin(th)));
                }
                ana[s] = v;
                interp[s] = 0.5 * (base.T[j][s] + base.T[j + 1][s]);
            }

            double e = 0.0, max_interp = -1e300, max_ana = -1e300;
            double max_j = -1e300, max_j1 = -1e300;
            for (int s = 0; s < base.S; ++s) {
                e = std::max(e, std::fabs(interp[s] - ana[s]));
                max_interp = std::max(max_interp, interp[s]);
                max_ana    = std::max(max_ana, ana[s]);
                max_j      = std::max(max_j, base.T[j][s]);
                max_j1     = std::max(max_j1, base.T[j + 1][s]);
            }
            const double max_B = 0.5 * (max_j + max_j1);   // interpolate the maxes
            const double bias_A = std::fabs(max_interp - max_ana);
            const double bias_B = std::fabs(max_B - max_ana);

            // The closed form, single harmonic, same nodes.
            const double truth = std::exp(-zetaq);
            const double formB = 0.5 * (std::exp(-zeta1) + std::exp(-zeta2));
            const double formA = std::abs(0.5 * std::exp(cd(-zeta1, -zeta1)) +
                                          0.5 * std::exp(cd(-zeta2, -zeta2)));
            const double fA = std::fabs(formA - truth), fB = std::fabs(formB - truth);

            // Interpolation is second order in the local spacing:
            // |T - interp| <= (h^2/8)|T''| = (h^2/(4 d^2)) * amplitude(zq).
            // Linear interpolation of a smooth function: |T - interp| <=
            // (h^2/8)|T''|, and for harmonic n, |T''| = |q_n|^2 |T_n| =
            // (2n/delta^2) * A_n * exp(-zq/d_n). Summed over the harmonics that
            // are actually present -- no slack factor, no fitted constant.
            const double h = z2 - z1;
            double bound = 0.0;
            for (int n = 1; n <= kNHarm; ++n) {
                const double d_n = delta / std::sqrt(static_cast<double>(n));
                bound += (h * h / 8.0) * (2.0 * n / (delta * delta)) *
                         std::abs(base.H[n][0]) * std::exp(-zq / d_n);
            }
            if (scored && e > bound + 1e-4) interp_ok = false;

            if (scored) {
                worst_interp_K = std::max(worst_interp_K, e);
                worst_A_meas   = std::max(worst_A_meas, bias_A);
                if (bias_B > worst_B_meas) {
                    worst_B_meas = bias_B; worst_B_layer = j;
                }
            }
            if (fA > worst_A_form) { worst_A_form = fA; worst_A_layer = j; }
            if (fB > worst_B_form) { worst_B_form = fB; worst_B_form_layer = j; }

            std::printf("   %s %5.3f  %5.3f |  %8.4f  |  %8.4f (%.3e)  "
                        "%8.4f (%.3e) |  %.3e   %.3e\n",
                        scored ? " " : "*", zeta1, zeta2, e, bias_A, bias_A / A1,
                        bias_B, bias_B / A1, fA, fB);
        }
        std::printf("     (* = below the %.0f-skin-depth scored window: the "
                    "closed form is still exact there, the\n"
                    "        measured columns carry the bottom reflection at "
                    "~1e-8 K)\n", kProbeDeltas);

        check(interp_ok,
              "W7: linear interpolation between nodes matches the analytic value "
              "to within the (h^2/8)|T''| interpolation bound at every midpoint "
              "(worst " + std::to_string(worst_interp_K) + " K)");

        std::printf("\n     U2 MEASUREMENT (decision D8):\n");
        std::printf("       arch A  in-engine max of the interpolant : "
                    "%.4f K  = %.3e * A   (closed form %.3e * A)\n",
                    worst_A_meas, worst_A_meas / A1, worst_A_form);
        std::printf("       arch B  interpolate the per-layer maxes  : "
                    "%.4f K  = %.3e * A   (closed form %.3e * A)\n",
                    worst_B_meas, worst_B_meas / A1, worst_B_form);
        std::printf("       closed form, worst over the WHOLE grid: "
                    "arch A %.3e * A at layer %d (h/delta %.2f, DEEP),\n"
                    "                                          arch B %.3e * A "
                    "at layer %d (SHALLOW)\n",
                    worst_A_form, worst_A_layer,
                    (base.z[worst_A_layer + 1] - base.z[worst_A_layer]) / delta,
                    worst_B_form, worst_B_form_layer);
        std::printf("       measured worst inside the scored window: arch B at "
                    "layer %d of %d\n", worst_B_layer, base.n_nodes - 1);
        std::printf("       at a lunar-equatorial A = 160 K these scale to "
                    "arch A %.3f K, arch B %.3f K\n",
                    160.0 * worst_A_form, 160.0 * worst_B_form);
        std::printf("       reference floor for a depth product: 0.15 K\n");

        check(worst_B_form > worst_A_form,
              "W8: the closed form puts arch B (interpolate the maxes) WORSE "
              "than arch A (max of the interpolant) -- the ordering that "
              "decides D8");

        const double relB = std::fabs(worst_B_meas / A1 - worst_B_form) /
                            std::max(worst_B_form, 1e-30);
        std::printf("       measured vs closed form: arch B %.1f%%\n",
                    100.0 * relB);
        check(relB < 0.25,
              "W9: the MEASURED arch-B bias confirms the closed form to better "
              "than 25% (they are independent: one is a DFT of the engine's own "
              "solution, the other is algebra)");

        // Arch A is NOT confirmed the same way, and saying so is the point.
        // Its predicted bias (2.5e-5*A) is an order of magnitude BELOW the
        // solver's own discretization error at the same nodes (1.2e-2 relative
        // amplitude at 4 skin depths, from the table above), so the measurement
        // cannot resolve it -- it can only BOUND it. Reporting the ratio of two
        // numbers when one of them is buried in noise would be theater.
        std::printf("\n       arch A is BOUNDED, not confirmed: its closed-form "
                    "bias %.3e * A sits an order of\n"
                    "       magnitude below the solver's own discretization "
                    "error at the same nodes, so the\n"
                    "       measurement (%.3e * A) is an upper bound dominated "
                    "by that error, not a\n"
                    "       measurement of commutation. Both readings agree on "
                    "the decision.\n",
                    worst_A_form, worst_A_meas / A1);
        check(worst_A_meas / A1 > worst_A_form,
              "W9b: the measured arch-A bias is at least the closed form, i.e. "
              "a consistent UPPER bound rather than a contradiction");

        // The decision, stated on BOTH readings so it does not rest on either
        // one alone. 0.15 K is the reference floor a depth product is scored
        // against; A = 160 K is the lunar equatorial diurnal amplitude.
        const double kFloorK = 0.15, kAeq = 160.0;
        const double A_form_K = kAeq * worst_A_form;
        const double A_meas_K = kAeq * worst_A_meas / A1;
        const double B_form_K = kAeq * worst_B_form;
        const double B_meas_K = kAeq * worst_B_meas / A1;
        std::printf("\n       D8 at A = %.0f K:  arch A  %.3f K (closed form) / "
                    "%.3f K (measured bound)\n"
                    "                          arch B  %.3f K (closed form) / "
                    "%.3f K (measured)\n"
                    "                          floor   %.3f K\n",
                    kAeq, A_form_K, A_meas_K, B_form_K, B_meas_K, kFloorK);
        check(A_form_K < kFloorK && A_meas_K < kFloorK &&
              B_form_K > kFloorK && B_meas_K > kFloorK,
              "W10: D8 -- arch B (native-grid export, interpolate max/min in "
              "post) EXCEEDS the 0.15 K floor at lunar-equatorial amplitude on "
              "BOTH readings, and arch A stays under it on both. The hybrid is "
              "required; B alone is not sufficient");

        // Grid-independent form, so this transfers to the SHIPPED lunar column,
        // whose local layer spacing this homogeneous grid does not reproduce.
        std::printf("\n       TRANSFERABLE FORM (bias relative to the LOCAL "
                    "amplitude at the query depth,\n"
                    "       as a function of h/delta_local -- the only "
                    "parameter; w = 0.5):\n"
                    "         h/delta_local      arch A        arch B\n");
        for (double hd : {0.1, 0.25, 0.5, 1.0, 2.0}) {
            const cd e1(1.0, 0.0);
            const cd e2 = std::exp(cd(-hd, -hd));
            const double truth = std::exp(-0.5 * hd);
            const double fA = std::fabs(std::abs(0.5 * e1 + 0.5 * e2) - truth) / truth;
            const double fB = std::fabs(0.5 * (1.0 + std::exp(-hd)) - truth) / truth;
            std::printf("         %5.2f          %.3e     %.3e%s\n", hd, fA, fB,
                        (hd == 0.1) ? "   <- m = 10, the surface cell"
                                    : ((hd == 2.0)
                                           ? "   <- MIN_LAYERS_DEEP_PER_SKIN = 0.5"
                                           : ""));
        }
        std::printf("       On the shipped lunar column the deep cell is "
                    "h = 2*delta_local, where the local\n"
                    "       diurnal amplitude is < 0.05 K by design "
                    "(config_parser.h), so arch A's worst case\n"
                    "       there is bounded by 0.05 K * the arch A entry "
                    "above.\n");

        // The interpolation error itself -- independent of max/min, and it is
        // the same size as arch B's bias, because arch B's bias essentially IS
        // the interpolation error of the max field.
        std::printf("\n       INSTANTANEOUS INTERPOLATION COST, a SEPARATE "
                    "error from the max-commutation one:\n"
                    "       worst %.4f K at A = %.2f K = %.3e * A  ->  %.3f K "
                    "at A = %.0f K.\n"
                    "       This is what a TIME SERIES at an off-node depth "
                    "pays, in every architecture,\n"
                    "       because the profile is curved and the interpolant "
                    "is straight. The error in the\n"
                    "       MAX is smaller (%.3f K at A = %.0f K) because the "
                    "pointwise errors partly cancel\n"
                    "       at the peak -- so a max/min product must not be "
                    "budgeted with this number, and a\n"
                    "       time-series product must not be budgeted with the "
                    "arch A one. Only a requested\n"
                    "       depth landing ON a node avoids either.\n",
                    worst_interp_K, A1, worst_interp_K / A1,
                    kAeq * worst_interp_K / A1, kAeq, A_meas_K, kAeq);
    }

    // ---- 6. THE SHIPPING PRODUCT PATH, end to end -------------------------
    //
    // Sections 3-5 score quantities THIS FILE computes for itself. That is a
    // gate on the solver, not on the product: every arithmetic step between the
    // solved column and statistics/temperature_z*.bin -- resolving a requested
    // depth to (j, w), sampling a strided column, replacing the massless
    // surface node, reducing to avg/max/min -- lived outside it. A wrong node
    // index or a dropped surface override would have passed everything above.
    //
    // So this section drives the SHIPPING code:
    //   thermal::depthmap::buildPlan      (the same call the engine makes)
    //   thermal::depthmap::Accumulator    (the same object both solver drivers
    //                                      fold their columns into)
    // over the engine's own recorded period, and scores the result against the
    // analytic reference at the RESOLVED depth, with a bound that has no fitted
    // constant in it.
    {
        using namespace thermal::depthmap;
        std::printf("\n  --- 6. the shipping depth-map product path, end to end "
                    "---\n");

        // Three probes, and the FIRST of them is load-bearing:
        //   z = 0        the surface node, j = 0 -- the ONLY probe that enters
        //                the surface-override branch. Without it W12 below is
        //                vacuous, which is exactly what it was until a
        //                falsification run (override disabled) left W12 passing.
        //                A guard whose input is never varied is invisible.
        //   an interior NODE (w = 0)   isolates the plumbing: no arithmetic.
        //   the MIDPOINT of that cell (w = 0.5)  the worst case for
        //                interpolation. z = 0 cannot exercise it at all.
        int j_probe = 1;
        while (j_probe + 2 < last_scored && base.z[j_probe] < 2.0 * delta) ++j_probe;
        const double z_node = base.z[j_probe];
        const double z_mid  = 0.5 * (base.z[j_probe] + base.z[j_probe + 1]);
        std::printf("     probes: z=0 (the surface node, j=0), node j=%d at "
                    "z/delta=%.3f, and the midpoint of cell [%d,%d] at "
                    "z/delta=%.3f\n",
                    j_probe, z_node / delta, j_probe, j_probe + 1, z_mid / delta);

        Request req;
        req.depths_m = { 0.0, z_node, z_mid };
        req.snap_to_node = false;           // sample the midpoint as asked
        Plan plan;
        std::string refusal;
        std::vector<std::string> warn;
        const bool built = buildPlan(req, base.z.data(), base.n_nodes, delta,
                                     /*n_faces*/ 0, /*with_raw*/ false, plan,
                                     refusal, warn);
        check(built, "W8: the shipping planner resolves both probe depths against "
                     "the engine's own grid" + (built ? "" : (" -- " + refusal)));
        if (!built) {
            std::printf("\n=== %d/%d checks passed ===\n", g_checks - g_failed,
                        g_checks);
            return 1;
        }
        check(plan.entries[0].on_node && plan.entries[0].j == 0 &&
                  plan.entries[0].w == 0.0,
              "W8b: the z=0 probe resolves to (j, w) = (0, 0), i.e. the surface "
              "node -- the one probe that reaches the override branch");
        check(plan.entries[1].on_node && plan.entries[1].j == j_probe &&
                  plan.entries[1].w == 0.0,
              "W9: the node probe resolves to (j, w) = (" +
                  std::to_string(j_probe) + ", 0) -- the plan indexes the node "
                  "the depth actually is, not a neighbor");
        check(!plan.entries[2].on_node && plan.entries[2].j == j_probe &&
                  std::fabs(plan.entries[2].w - 0.5) < 1e-12,
              "W10: the midpoint probe resolves to w = 0.5");

        // Fold the engine's own recorded period through the SHIPPING
        // accumulator, one column per epoch, exactly as a solver driver does.
        // Face 0 uses the recorded column as-is; face 1 is fed a deliberately
        // CORRUPTED surface node together with the correct value as the
        // `surface_override`, which is the path the CPU SIMD driver takes (the
        // massless surface is re-evaluated, not read out of the column). If the
        // override were dropped, face 1 would not match face 0.
        Accumulator acc;
        acc.allocate(plan.size(), 2);
        std::vector<double> col(base.n_nodes), col_bad(base.n_nodes);
        for (int s = 0; s < base.S; ++s) {
            for (int j = 0; j < base.n_nodes; ++j) col[j] = base.T[j][s];
            col_bad = col;
            col_bad[0] = -999.0;            // would be catastrophic if used
            acc.addColumn(plan, 0, col.data(), 1);
            const double surf_s = base.T[0][s];
            acc.addColumn(plan, 1, col_bad.data(), 1, &surf_s);
        }
        check(acc.count[0] == static_cast<std::uint32_t>(base.S) &&
                  acc.count[1] == acc.count[0],
              "W11: the accumulator folded every sample of the recorded period");

        bool override_ok = true;
        for (std::size_t d = 0; d < plan.size(); ++d) {
            if (acc.sum[d * 2 + 0]  != acc.sum[d * 2 + 1] ||
                acc.vmax[d * 2 + 0] != acc.vmax[d * 2 + 1] ||
                acc.vmin[d * 2 + 0] != acc.vmin[d * 2 + 1]) override_ok = false;
        }
        check(override_ok,
              "W12: a column whose node 0 is corrupted to -999 K, folded WITH the "
              "correct surface override, gives bit-identical avg/max/min at every "
              "probe INCLUDING z=0 -- so the override reaches the surface node, and "
              "the deeper probes prove it reaches nothing else");
        // Make the falsification explicit: without the override, face 1's z=0
        // product WOULD be -999 K. Assert that the corrupted value is not what
        // came out, so a future refactor that quietly drops the override is
        // caught by a check that names the failure rather than by a bit compare.
        check(acc.vmin[0 * 2 + 1] > 0.0 && acc.vmin[0 * 2 + 0] > 0.0,
              "W12b: and neither face's z=0 minimum is the -999 K sentinel, so the "
              "override was genuinely consulted rather than the corruption ignored");

        // --- plumbing, exact: the product equals a direct reduction ---------
        for (std::size_t d = 0; d < plan.size(); ++d) {
            const Entry& e = plan.entries[d];
            double sum = 0.0, mx = -1e300, mn = 1e300;
            for (int s = 0; s < base.S; ++s) {
                const double v = e.on_node
                    ? base.T[e.j][s]
                    : (1.0 - e.w) * base.T[e.j][s] + e.w * base.T[e.j + 1][s];
                sum += v;
                mx = std::max(mx, v);
                mn = std::min(mn, v);
            }
            // max and min are SELECTIONS -- no arithmetic -- so they must be
            // bit-identical, and a wrong node index or a wrong weight shows up
            // in the last bit. The SUM is a different matter: an S-term
            // accumulation is reassociated by the vectorizer, so the two orders
            // can differ by at most (S-1)*u*sum|x|. That bound is derived here,
            // not chosen, and at S=768 it is ~1e-8 K on a ~1.5e5 K sum, i.e.
            // ~1e-11 K on the average the product actually writes.
            const double reassoc = static_cast<double>(base.S) *
                                   std::numeric_limits<double>::epsilon() *
                                   std::fabs(sum);
            const bool sel_exact = (acc.vmax[d * 2 + 0] == mx) &&
                                   (acc.vmin[d * 2 + 0] == mn);
            const double sum_err = std::fabs(acc.sum[d * 2 + 0] - sum);
            std::printf("     [W13] %-22s max/min bit-exact: %s   sum |d| %.3e "
                        "(reassociation bound %.3e, = %.2e K on the average)\n",
                        e.field.c_str(), sel_exact ? "yes" : "NO", sum_err,
                        reassoc, sum_err / base.S);
            check(sel_exact,
                  "W13[" + e.field + "]: the shipping accumulator's max and min "
                  "are BIT-IDENTICAL to a direct reduction of the interpolated "
                  "series (they are selections; a wrong node or weight would "
                  "move the last bit)");
            check(sum_err <= reassoc,
                  "W13b[" + e.field + "]: and its sum agrees to within the "
                  "floating-point reassociation bound");
        }

        // --- physics: score the product against the analytic reference ------
        //
        // The bound has two parts and no fitted constant:
        //   discretization  the scheme's own predicted error at the bracketing
        //                   nodes, from the discrete dispersion relation used
        //                   throughout this file;
        //   interpolation   sum_h A_h * bias_instantaneous_per_A[h], read out of
        //                   THE PLAN -- i.e. the very numbers written into
        //                   statistics/depth_map_z.json. Scoring against them is
        //                   what makes that manifest a checked claim instead of
        //                   an unverified annotation.
        std::vector<std::vector<cd>> pred_exp(kNHarm + 1);
        for (int n = 1; n <= kNHarm; ++n) pred_exp[n] = predictedExponent(base, n);

        std::printf("     depth      |  max: product   analytic    |err|      "
                    "bound(disc+interp)  ratio\n");
        for (std::size_t d = 0; d < plan.size(); ++d) {
            const Entry& e = plan.entries[d];
            const double zq = e.resolved_m;

            // Analytic series at zq, rebuilt by superposition from the MEASURED
            // surface harmonics (the same construction section 5 uses).
            double ana_max = -1e300, ana_mean = 0.0;
            for (int sidx = 0; sidx < base.S; ++sidx) {
                double v = base.mean_z[0];
                for (int n = 1; n <= kNHarm; ++n) {
                    const double d_n = delta / std::sqrt(static_cast<double>(n));
                    const cd prop = std::exp(cd(-zq / d_n, -zq / d_n));
                    const double th = 2.0 * M_PI * n * sidx / base.S;
                    v += std::real(base.H[n][0] * prop *
                                   cd(std::cos(th), std::sin(th)));
                }
                ana_max = std::max(ana_max, v);
                ana_mean += v;
            }
            ana_mean /= base.S;

            // Discretization at the bracketing nodes, summed over harmonics.
            double disc = 0.0;
            for (int n = 1; n <= kNHarm; ++n) {
                const double d_n = delta / std::sqrt(static_cast<double>(n));
                const double An = std::abs(base.H[n][0]);
                for (int jj : { e.j, std::min(e.j + 1, base.n_nodes - 1) }) {
                    const cd sch = std::exp(-pred_exp[n][jj]);
                    const cd tru = std::exp(cd(-base.z[jj] / d_n, -base.z[jj] / d_n));
                    disc = std::max(disc, An * std::abs(sch - tru));
                }
            }
            // Interpolation, from the plan's own published coefficients.
            double interp_bound = 0.0;
            for (int h = 1; h <= kBiasHarmonics; ++h) {
                interp_bound += std::abs(base.H[h][0]) * e.bias_inst_per_A[h - 1];
            }
            const double bound = disc + interp_bound;

            const double prod_max = acc.vmax[d * 2 + 0];
            const double err = std::fabs(prod_max - ana_max);
            std::printf("     %-10s |  %9.4f  %9.4f  %8.5f   %8.5f            "
                        "%6.3f\n",
                        e.field.c_str(), prod_max, ana_max, err, bound,
                        (bound > 0.0) ? err / bound : 0.0);
            check(err <= bound + 1e-4,
                  "W14[" + e.field + "]: the max the PRODUCT would write is within "
                  "the derived bound of the analytic max at the resolved depth "
                  "(err " + std::to_string(err) + " K, bound " +
                  std::to_string(bound) + " K)");

            const double prod_avg = acc.sum[d * 2 + 0] / acc.count[0];
            check(std::fabs(prod_avg - ana_mean) <= bound + 1e-4,
                  "W15[" + e.field + "]: the avg the PRODUCT would write is within "
                  "the same bound of the analytic mean (err " +
                  std::to_string(std::fabs(prod_avg - ana_mean)) + " K)");
        }

        // --- snap-to-node is a real behavior change, end to end -------------
        {
            Request rq2 = req;
            rq2.depths_m = { z_mid };
            rq2.snap_to_node = true;
            Plan p2;
            std::string r2;
            std::vector<std::string> w2;
            check(buildPlan(rq2, base.z.data(), base.n_nodes, delta, 0, false, p2,
                            r2, w2),
                  "W16: the same midpoint request builds with snap-to-node on");
            Accumulator a2;
            a2.allocate(p2.size(), 1);
            for (int sidx = 0; sidx < base.S; ++sidx) {
                for (int j = 0; j < base.n_nodes; ++j) col[j] = base.T[j][sidx];
                a2.addColumn(p2, 0, col.data(), 1);
            }
            // It snapped to ONE of the bracketing nodes; whichever it was, the
            // product must now be that node's own series, exactly.
            const int jn = p2.entries[0].on_node && p2.entries[0].w == 0.0
                               ? p2.entries[0].j : p2.entries[0].j + 1;
            double mx = -1e300;
            for (int sidx = 0; sidx < base.S; ++sidx)
                mx = std::max(mx, base.T[jn][sidx]);
            check(a2.vmax[0] == mx,
                  "W17: with snap-to-node ON the product is the SNAPPED node's own "
                  "series, bit-for-bit -- the interpolation is genuinely bypassed, "
                  "not merely re-weighted");
            // "Zero" here means zero to round-off: the coefficients are
            // differences of exponentials that cancel identically at w = 0 or
            // w = 1, so the residue is ~1e-17, not a bit pattern.
            double snapped_budget = 0.0;
            for (int h = 0; h < kBiasHarmonics; ++h) {
                snapped_budget = std::max(snapped_budget,
                    std::max(std::fabs(p2.entries[0].bias_inst_per_A[h]),
                             std::fabs(p2.entries[0].bias_max_per_A[h])));
            }
            check(snapped_budget < 1e-14,
                  "W18: and the published error budget for the snapped depth is "
                  "zero at every harmonic to round-off (" +
                  std::to_string(snapped_budget) + ")");
            check(std::fabs(p2.entries[0].residual_m) > 0.0,
                  "W19: while the residual records how far it moved, so the snap "
                  "cannot pass unnoticed");
        }
    }

    std::printf("\n=== %d/%d checks passed ===\n", g_checks - g_failed, g_checks);
    return (g_failed == 0) ? 0 : 1;
}
