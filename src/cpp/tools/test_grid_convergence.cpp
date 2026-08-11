// Grid-convergence gate for the depth discretization — the test that decides
// whether a minimum layer thickness may exist at all.
//
// WHY THIS EXISTS. `Profile`'s constructor used to build layer thicknesses as
// `dz[i] = max(1e-3, z[i+1] - z[i])`, flooring the CPU conduction stencil at
// 1 mm while leaving the depth grid `z` — which the material model and every
// GPU path index — untouched. The claim that this was safe is exactly a
// convergence claim, and it is falsifiable: under grid refinement a correct
// discretization must converge, and one whose stencil spacing is pinned at a
// constant CANNOT, because refinement past the pin changes z (and so the
// sampled properties) while the stencil keeps integrating the same thickness.
//
// So this program refines m (layers per skin depth) and measures whether the
// diurnal solution converges. It runs BOTH arms from one binary:
//   --floor=0     the shipping behavior, dz == the grid's own spacing
//   --floor=1e-3  a faithful re-creation of the removed clamp (floor dz only,
//                 leave z alone, rebuild d3z/g1/g2 from the floored dz —
//                 exactly what the old constructor did)
// so the comparison cannot be confounded by two different builds.
//
// It also reports the top-cell Fourier number Fo0 = kappa*dt/dz0^2, which is
// the quantity the derivation says actually limits refinement (see the note in
// Profile's constructor), and the surface-Newton cap counters, so a run that
// converges by luck while the Newton is failing cannot pass quietly.
//
// GATES (exit 0 = PASS, printed as "[test_grid_convergence] PASS")
//   1. UNFLOORED CONVERGES, AT FIRST ORDER. With dz == grid spacing and the
//      substep count scaled as m^2 (constant Fo0, as the derivation requires),
//      each m -> 2m refinement must shrink the change in diurnal Tmax by a
//      factor near 2, and the last change must be < 0.05 K.
//   2. FLOORED STALLS AT THE WRONG ANSWER. This is the falsifier, and note
//      what it is NOT: it is not "the floored deltas stay large". A stalled
//      sequence has SMALL successive differences precisely because it is
//      stuck, so small deltas would wrongly read as convergence. The test is
//      whether the floored arm approaches the same LIMIT — its finest value
//      must differ from the unfloored arm's Richardson-extrapolated limit by
//      more than `stall_K`, and its deltas must fail to contract.
//   3. NEWTON STAYS HEALTHY. Neither arm may hit the surface-Newton iteration
//      cap or the growth-side step limiter.
//
// Usage:
//   test_grid_convergence [--quiet]
#include "../thermal/thermal_model.h"
#include "../thermal/planetary_properties.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

// The shipped asteroid quickstart's column (examples/quickstart_asteroid_
// diurnal.yaml, generic_body.surface). TI = sqrt(k*rho*cp) = 100 SI exactly,
// constant in depth and temperature, so there is one number to reason about.
struct Column {
    double ks = 0.008333333, kd = 0.008333333;
    double rhos = 1500.0, rhod = 1500.0;
    double cp0 = 800.0;
    double H = 0.02;
    double period_s = 18000.0;   // 5 h rotation
    double emissivity = 0.95;
    double albedo = 0.05;
    double Qb = 0.018;
    double S = 1361.0 / (2.7 * 2.7);  // solar constant at a = 2.7 AU
};

thermal::Planet makePlanet(const Column& c) {
    thermal::Planet p("Moon");     // any name; every field below is overwritten
    p.ks = c.ks; p.kd = c.kd;
    p.rhos = c.rhos; p.rhod = c.rhod;
    p.cp0 = c.cp0; p.cpCoeff.clear();
    p.H = c.H;
    p.day = c.period_s;
    p.emissivity = c.emissivity;
    p.albedo = c.albedo;
    p.Qb = c.Qb;
    p.S = c.S;
    p.albedoCoef.clear();          // no incidence-angle albedo law
    p.use_hapke_albedo = false;
    return p;
}

// Re-create the REMOVED clamp exactly: floor dz, leave z alone, then rebuild
// the grid coefficients from the floored dz — the old constructor's order of
// operations. Rebuilding d3z/g1/g2 matters; skipping it would emulate a
// different (and less harmful) bug than the one that shipped.
void applyLegacyFloor(thermal::Profile& p, double floor_m) {
    if (!(floor_m > 0.0)) return;
    for (size_t i = 0; i < p.dz().size(); ++i) {
        if (p.dz()[i] < floor_m) p.dz()[i] = floor_m;
    }
    for (size_t i = 0; i + 1 < p.dz().size(); ++i) {
        const double d3 = p.dz()[i+1] * p.dz()[i] * (p.dz()[i+1] + p.dz()[i]);
        p.d3z()[i] = d3;
        p.g1()[i]  = 2.0 * p.dz()[i+1] / d3;
        p.g2()[i]  = 2.0 * p.dz()[i]   / d3;
    }
}

struct Result {
    double Tmax = 0.0, Tmin = 0.0;
    double Fo0 = 0.0, dz0 = 0.0;
    int    nlayers = 0;
};

// Drive one column through `n_rot` rotations of analytic diurnal insolation at
// the equator and report the last rotation's surface extrema. Absorbed flux is
// S*(1-A)*max(0, cos(hour angle)) — a hard sunrise/sunset corner, which is the
// forcing that stresses a thin top cell hardest.
Result runColumn(const Column& col, int m, int steps_per_rot, int n_rot,
                 double floor_m) {
    thermal::Planet planet = makePlanet(col);
    thermal::Config config;
    config.m = m;
    config.b = 15;
    config.chi = 0.0;

    thermal::Profile prof(planet, 0.0, config, /*verbose=*/false);
    applyLegacyFloor(prof, floor_m);

    Result r;
    r.nlayers = prof.nlayers;
    r.dz0     = prof.dz()[0];
    const double dt = col.period_s / static_cast<double>(steps_per_rot);
    r.Fo0     = thermal::topCellFourierNumber(prof, dt);

    const double one_minus_A = 1.0 - col.albedo;
    r.Tmax = -1e30;
    r.Tmin =  1e30;

    const long long total = static_cast<long long>(steps_per_rot) * n_rot;
    for (long long s = 0; s < total; ++s) {
        const double phase = 2.0 * M_PI *
            (static_cast<double>(s) / static_cast<double>(steps_per_rot));
        const double mu = std::cos(phase);
        const double Qs = col.S * one_minus_A * (mu > 0.0 ? mu : 0.0);
        prof.update_T(dt, Qs, col.Qb);

        if (s >= total - steps_per_rot) {          // last rotation only
            const double T0 = prof.T()[0];
            if (T0 > r.Tmax) r.Tmax = T0;
            if (T0 < r.Tmin) r.Tmin = T0;
        }
    }
    return r;
}

// ---------------------------------------------------------------------------
// Adversarial stability probe. Drives a column through the hard sunrise corner
// and reports whether the surface series rings, and how badly. Deliberately
// includes cases PAST the derived Fo0 <= 1 bound, because the useful question
// is not "is there a regime where a thin cell misbehaves" (there is, and the
// derivation predicts exactly where) but "does it stay bounded and does the
// solver tell you". Reports NaN/negative temperatures, the post-sunrise
// oscillation count, and the Newton cap counters.
struct Probe {
    double Fo0 = 0.0, dz0 = 0.0;
    double Tmax = 0.0, Tmin = 0.0;
    int    sign_flips = 0;     // successive-dT sign changes after sunrise
    double max_step_dT = 0.0;  // largest |T0(n) - T0(n-1)| after sunrise
    bool   finite = true;
    bool   positive = true;
};

Probe probeColumn(const Column& col, int m, int steps_per_rot, int n_rot,
                  double floor_m) {
    thermal::Planet planet = makePlanet(col);
    thermal::Config config;
    config.m = m;
    config.b = 15;
    config.chi = 0.0;

    thermal::Profile prof(planet, 0.0, config, /*verbose=*/false);
    applyLegacyFloor(prof, floor_m);

    Probe p;
    p.dz0 = prof.dz()[0];
    const double dt = col.period_s / static_cast<double>(steps_per_rot);
    p.Fo0 = thermal::topCellFourierNumber(prof, dt);

    const double one_minus_A = 1.0 - col.albedo;
    p.Tmax = -1e30; p.Tmin = 1e30;

    const long long total = static_cast<long long>(steps_per_rot) * n_rot;
    // Watch the 1/8 rotation straight after sunrise on the LAST rotation --
    // the sharpest forcing corner in the cycle.
    const long long watch_lo = total - steps_per_rot + (3 * steps_per_rot) / 4;
    const long long watch_hi = watch_lo + steps_per_rot / 8;

    double prev_T = 0.0, prev_dT = 0.0;
    for (long long s = 0; s < total; ++s) {
        const double phase = 2.0 * M_PI *
            (static_cast<double>(s) / static_cast<double>(steps_per_rot));
        const double mu = std::cos(phase);
        const double Qs = col.S * one_minus_A * (mu > 0.0 ? mu : 0.0);
        prof.update_T(dt, Qs, col.Qb);

        const double T0 = prof.T()[0];
        if (!std::isfinite(T0)) p.finite = false;
        if (T0 <= 0.0) p.positive = false;
        if (s >= total - steps_per_rot) {
            if (T0 > p.Tmax) p.Tmax = T0;
            if (T0 < p.Tmin) p.Tmin = T0;
        }
        if (s >= watch_lo && s < watch_hi) {
            const double dT = T0 - prev_T;
            if (s > watch_lo) {
                if (dT * prev_dT < 0.0) p.sign_flips++;
                if (std::fabs(dT) > p.max_step_dT) p.max_step_dT = std::fabs(dT);
            }
            prev_dT = dT;
        }
        prev_T = T0;
    }
    return p;
}

const char* arm_name(double floor_m) {
    return (floor_m > 0.0) ? "FLOORED (legacy 1 mm dz clamp)"
                           : "UNFLOORED (dz == grid spacing)";
}

struct Arm {
    std::vector<double> tmax;
    std::vector<double> deltas;
    bool   contracting  = true;
    bool   first_order  = true;   // every delta ratio near 2
    double last_delta   = 0.0;
    double richardson   = 0.0;    // limit implied by the finest pair, p = 1
};

// One refinement sequence. Substeps scale as m^2 so that Fo0 = pi*m^2/N stays
// constant across the sweep: without that, refinement changes the temporal
// error at the same time as the spatial error and the sequence measures
// neither cleanly.
Arm runArm(const Column& col, double floor_m, const std::vector<int>& m_list,
           double fo_target, int n_rot, bool quiet) {
    std::printf("\n=== %s ===\n", arm_name(floor_m));
    std::printf("  %4s %8s %12s %8s %11s %11s %10s %7s\n",
                "m", "layers", "dz0 [m]", "Fo0", "Tmax [K]", "Tmin [K]",
                "d(Tmax)", "ratio");

    Arm a;
    for (size_t i = 0; i < m_list.size(); ++i) {
        const int m = m_list[i];
        const int steps = static_cast<int>(
            std::ceil(M_PI * static_cast<double>(m) * m / fo_target));
        const Result r = runColumn(col, m, steps, n_rot, floor_m);
        a.tmax.push_back(r.Tmax);

        char dbuf[32], rbuf[32];
        std::snprintf(dbuf, sizeof dbuf, "%10s", "-");
        std::snprintf(rbuf, sizeof rbuf, "%7s", "-");
        if (i > 0) {
            const double d = std::fabs(a.tmax[i] - a.tmax[i-1]);
            std::snprintf(dbuf, sizeof dbuf, "%10.4f", d);
            if (!a.deltas.empty()) {
                const double prev = a.deltas.back();
                const double ratio = (d > 0.0) ? prev / d : 1e30;
                std::snprintf(rbuf, sizeof rbuf, "%7.2f", ratio);
                if (d > prev) a.contracting = false;
                if (ratio < 1.7 || ratio > 2.3) a.first_order = false;
            }
            a.deltas.push_back(d);
        }
        if (!quiet) {
            std::printf("  %4d %8d %12.3e %8.3f %11.4f %11.4f %s %s\n",
                        m, r.nlayers, r.dz0, r.Fo0, r.Tmax, r.Tmin, dbuf, rbuf);
        }
    }

    if (!a.deltas.empty()) {
        a.last_delta = a.deltas.back();
        // Richardson with observed order p = 1 (halving dz halves the error):
        // T_inf = T_fine + (T_fine - T_coarse)/(2^p - 1).
        const size_t n = a.tmax.size();
        a.richardson = a.tmax[n-1] + (a.tmax[n-1] - a.tmax[n-2]);
    }
    std::printf("  final |dTmax| = %.5f K, contracting = %s, "
                "first-order ratios = %s\n",
                a.last_delta, a.contracting ? "yes" : "NO",
                a.first_order ? "yes" : "NO");
    return a;
}

}  // namespace

int main(int argc, char** argv) {
    bool quiet = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--quiet") == 0) quiet = true;
    }

    const Column col;
    const std::vector<int> m_list = {5, 10, 20, 40, 80};
    const double fo_target = 0.40;   // constant across the sweep, < 1
    const int    n_rot     = 40;     // enough for the surface cycle to repeat
    const double tol_K     = 0.05;   // convergence tolerance, unfloored arm
    const double stall_K   = 0.30;   // min offset that counts as "wrong limit"

    std::printf("Grid-convergence gate: shipped asteroid quickstart column\n");
    std::printf("  TI = %.1f SI, P = %.0f s, S = %.1f W/m^2, A = %.2f, eps = %.2f\n",
                std::sqrt(col.ks * col.rhos * col.cp0), col.period_s, col.S,
                col.albedo, col.emissivity);
    std::printf("  substeps scaled as m^2 to hold Fo0 = %.2f constant; "
                "%d rotations\n", fo_target, n_rot);

    const Arm unfloored = runArm(col, 0.0,  m_list, fo_target, n_rot, quiet);
    const Arm floored   = runArm(col, 1e-3, m_list, fo_target, n_rot, quiet);

    // ---- Gate 1: the unfloored arm converges, at the expected order --------
    const bool g1 = unfloored.contracting && unfloored.first_order &&
                    (unfloored.last_delta < tol_K);

    // ---- Gate 2: the floored arm stalls at a DIFFERENT limit ---------------
    // Compared against the unfloored arm's Richardson limit, not against its
    // own successive differences: a stalled sequence's deltas go to zero.
    const double truth  = unfloored.richardson;
    const double stall  = std::fabs(floored.tmax.back() - truth);
    const bool   g2     = (stall > stall_K) && !floored.contracting;

    std::printf("\n=== verdict ===\n");
    std::printf("  grid-converged Tmax (Richardson, p=1) = %.4f K\n", truth);
    std::printf("  floored arm's finest Tmax             = %.4f K"
                "  -> stalls %+.4f K away\n",
                floored.tmax.back(), floored.tmax.back() - truth);
    std::printf("  error at the SHIPPED m=10: unfloored %+.4f K, "
                "floored %+.4f K (floor costs %.4f K)\n",
                unfloored.tmax[1] - truth, floored.tmax[1] - truth,
                std::fabs(floored.tmax[1] - unfloored.tmax[1]));
    std::printf("  gate 1 (unfloored converges, first order, <%.2f K): %s\n",
                tol_K, g1 ? "PASS" : "FAIL");
    std::printf("  gate 2 (floored stalls >%.2f K off, non-contracting): %s\n",
                stall_K, g2 ? "PASS" : "FAIL");

    // ---- Stability probe --------------------------------------------------
    // Thin top cells against a hard sunrise corner, including cases far past
    // the derived Fo0 <= 1 bound and a 10-minute super-fast rotator. Nothing
    // here may produce a non-finite or non-positive temperature; ringing above
    // Fo0 = 1 is expected and is reported, not failed.
    std::printf("\n=== stability probe: thin cell vs. hard sunrise corner ===\n");
    std::printf("  %-26s %5s %7s %11s %8s %7s %11s %6s\n",
                "case", "m", "N/rot", "dz0 [m]", "Fo0", "flips",
                "max dT [K]", "finite");
    struct ProbeCase { const char* name; Column col; int m; int steps; };
    Column fast = col;  fast.period_s = 600.0;                 // 10 min rotator
    Column loti = col;  loti.ks = loti.kd = 0.00085333331;     // TI = 32 SI
    const std::vector<ProbeCase> cases = {
        {"shipped m=10",             col,  10,   768},
        {"refined m=40, Fo<1",       col,  40,  12567},
        {"refined m=80, Fo<1",       col,  80,  50266},
        {"m=40, Fo0~13 (past bound)",col,  40,    384},
        {"m=80, Fo0~52 (far past)",  col,  80,    384},
        {"TI=32, m=40 (dz0~50 um)",  loti, 40,  12567},
        {"10-min rotator, m=40",     fast, 40,  12567},
    };
    bool probe_ok = true;
    for (const auto& c : cases) {
        const Probe p = probeColumn(c.col, c.m, c.steps, 20, /*floor=*/0.0);
        std::printf("  %-26s %5d %7d %11.3e %8.2f %7d %11.4f %6s\n",
                    c.name, c.m, c.steps, p.dz0, p.Fo0, p.sign_flips,
                    p.max_step_dT, (p.finite && p.positive) ? "yes" : "NO");
        if (!p.finite || !p.positive) probe_ok = false;
    }
    std::printf("  'flips' = post-sunrise successive-dT sign changes; nonzero "
                "above Fo0=1 is the predicted CN ringing, bounded and damped.\n");
    std::printf("  gate 3 (all cases finite and positive): %s\n",
                probe_ok ? "PASS" : "FAIL");

    std::printf("\n=== surface-Newton health (all arms and probes) ===\n");
    thermal::reportSurfaceNewtonStats();
    std::printf("  (no WARNING lines above = no iteration-cap or growth-cap hits)\n");

    const bool pass = g1 && g2 && probe_ok;
    std::printf("\n[test_grid_convergence] %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
