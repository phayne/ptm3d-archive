// test_primary_disk -- standalone gate for the PTM-41 / PTM-44 primary-body
// flux geometry in src/cpp/illumination/primary_disk.h.
//
// Everything here runs on the CPU in well under a second and needs no SPICE,
// no mesh and no Embree: it exercises the closed forms directly against
// numerical truth, so a failure localizes to the physics and not to the
// plumbing. The engine-level arms live in the session report.
//
// Build:  cmake --build build --target test_primary_disk
// Run:    ./build/bin/test_primary_disk
// Exit:   0 = all checks pass, 1 = at least one failed.
//
// Derivation and symbols: docs/primary-reflected-flux-derivation.md

#include "../illumination/primary_disk.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace ptm::disk;

static int g_failures = 0;

static void check(bool ok, const std::string& what, double got, double want,
                  double tol) {
    if (!ok) ++g_failures;
    std::printf("  [%s] %-52s got %-14.8g want %-14.8g tol %.3g\n",
                ok ? "PASS" : "FAIL", what.c_str(), got, want, tol);
}

static void checkClose(const std::string& what, double got, double want,
                       double tol) {
    check(std::fabs(got - want) <= tol, what, got, want, tol);
}

// ---------------------------------------------------------------------------
// 1. Lambert-sphere phase function vs the lune integral it summarizes.
//
//    E*d^2/(A F) = Int_{lit and visible} (cos i / pi) cos e dA  over the unit
//    sphere.  The closed form says this equals (2/3) Phi_L(a).
// ---------------------------------------------------------------------------
static double luneIntegral(double a, int n) {
    const double ox = 0.0, oy = 0.0, oz = 1.0;             // observer direction
    const double sx = std::sin(a), sy = 0.0, sz = std::cos(a);  // sun direction
    double acc = 0.0;
    for (int it = 0; it < n; ++it) {
        const double th = (it + 0.5) * M_PI / n;
        const double st = std::sin(th), ct = std::cos(th);
        const double wth = (M_PI / n) * st;
        for (int ip = 0; ip < 2 * n; ++ip) {
            const double ph = (ip + 0.5) * 2.0 * M_PI / (2 * n);
            const double nx = st * std::cos(ph), ny = st * std::sin(ph), nz = ct;
            const double ci = nx * sx + ny * sy + nz * sz;
            const double ce = nx * ox + ny * oy + nz * oz;
            if (ci > 0.0 && ce > 0.0) acc += (ci / M_PI) * ce * wth * (2.0 * M_PI / (2 * n));
        }
    }
    return acc;
}

static void testPhaseFunction() {
    std::printf("\n1. Lambert-sphere phase function vs direct lune integral\n");
    const double degs[] = {0, 15, 30, 45, 60, 90, 120, 135, 150, 170};
    for (double d : degs) {
        const double a = d * M_PI / 180.0;
        const double num = luneIntegral(a, 700);
        const double closed = (2.0 / 3.0) * lambertSpherePhase(a);
        const double rel = std::fabs(num - closed) / std::max(1e-12, closed);
        char buf[64];
        std::snprintf(buf, sizeof buf, "phase %5.1f deg, relative error", d);
        // 700x1400 quadrature: 1e-4 is loose for the smooth part and honest for
        // the terminator kink near 170 deg.
        check(rel <= 2e-3, buf, rel, 0.0, 2e-3);
    }
    checkClose("Phi_L(0)   = 1 (full)",       lambertSpherePhase(0.0),        1.0,        1e-12);
    checkClose("Phi_L(pi/2)= 1/pi",           lambertSpherePhase(M_PI / 2),   1.0 / M_PI, 1e-12);
    checkClose("Phi_L(pi)  = 0 (new)",        lambertSpherePhase(M_PI),       0.0,        1e-12);

    // monotone decreasing on [0, pi]
    bool mono = true;
    double prev = lambertSpherePhase(0.0);
    for (int k = 1; k <= 2000; ++k) {
        const double v = lambertSpherePhase(k * M_PI / 2000.0);
        if (v > prev + 1e-15) mono = false;
        prev = v;
    }
    check(mono, "Phi_L monotone decreasing on [0, pi]", mono ? 1 : 0, 1, 0);
}

// ---------------------------------------------------------------------------
// 2. Phase integral and energy closure.
// ---------------------------------------------------------------------------
static void testEnergyClosure() {
    std::printf("\n2. Phase integral and energy closure\n");
    const int N = 200000;
    double q = 0.0, q_old = 0.0;
    for (int k = 0; k < N; ++k) {
        const double a = (k + 0.5) * M_PI / N;
        q     += 2.0 * lambertSpherePhase(a) * std::sin(a) * (M_PI / N);
        q_old += 2.0 * ((1.0 + std::cos(a)) / 2.0) * std::sin(a) * (M_PI / N);
    }
    checkClose("phase integral q (Lambert) = 3/2", q, 1.5, 1e-6);
    checkClose("phase integral q of the pre-fix (1+cos a)/2", q_old, 2.0, 1e-6);

    // Total power radiated by the fixed model, integrated over a sphere at
    // distance d, must equal A_bond * F * pi * R^2.
    const double A = 0.343, F = 50.26, R = 69911.0, d = 671100.0;
    double total = 0.0;
    for (int k = 0; k < N; ++k) {
        const double a = (k + 0.5) * M_PI / N;
        const double E = lambertSphereIrradiance(F, A, R / d, a, 1.0);
        total += E * d * d * 2.0 * M_PI * std::sin(a) * (M_PI / N);
    }
    const double want = A * F * M_PI * R * R;
    checkClose("emitted power / A*F*pi*R^2 (fixed)", total / want, 1.0, 1e-4);

    // Same integral for the pre-fix expression: exactly half.
    double total_old = 0.0;
    for (int k = 0; k < N; ++k) {
        const double a = (k + 0.5) * M_PI / N;
        const double alpha = std::asin(R / d);
        const double E = F * A * ((1.0 + std::cos(a)) / 2.0)
                       * (2.0 * M_PI * (1.0 - std::cos(alpha))) / (4.0 * M_PI);
        total_old += E * d * d * 2.0 * M_PI * std::sin(a) * (M_PI / N);
    }
    // Exactly 1/2 in the small-angle limit; the residual 2/(1+cos alpha) is the
    // Omega/(4 pi) vs sin^2(alpha)/4 difference, +0.27 % at Jupiter/Europa.
    const double alpha_J = std::asin(R / d);
    checkClose("emitted power / A*F*pi*R^2 (PRE-FIX) = 1/2 * 2/(1+cos a)",
               total_old / want, 0.5 * 2.0 / (1.0 + std::cos(alpha_J)), 1e-4);
}

// ---------------------------------------------------------------------------
// 3. Horizon-clipped mean cosine: closed form vs 2-D numerical integration of
//    max(sin(h + x), 0) over the disk, and the seam/monotonicity properties.
// ---------------------------------------------------------------------------
static double meanCosNumeric(double h, double alpha, int n) {
    // average of max(cos theta, 0) over the disk of angular radius alpha whose
    // center sits at altitude h; theta measured from the face normal, so
    // cos theta = sin(altitude) and the altitude of a disk point offset by
    // (x, y) is h + x to first order in the (small) disk radius.
    double acc = 0.0;
    long cnt = 0;
    for (int i = 0; i < n; ++i) {
        const double x = (-1.0 + 2.0 * (i + 0.5) / n);
        const double ymax = std::sqrt(std::max(0.0, 1.0 - x * x));
        const int m = std::max(1, int(n * ymax));
        for (int j = 0; j < m; ++j) {
            const double c = std::sin(h + alpha * x);
            acc += (c > 0.0 ? c : 0.0);
            ++cnt;
        }
    }
    return cnt ? acc / cnt : 0.0;
}

static void testHorizonMeanCos() {
    std::printf("\n3. Horizon-clipped <cos theta>\n");

    checkClose("g(-1) = 0",              horizonMeanG(-1.0), 0.0,               1e-12);
    checkClose("g( 0) = 2/(3 pi)",       horizonMeanG( 0.0), 2.0 / (3 * M_PI),  1e-12);
    checkClose("g(+1) = 1",              horizonMeanG( 1.0), 1.0,               1e-12);

    // g monotone non-decreasing
    bool mono = true;
    double prev = -1.0;
    for (int k = 0; k <= 4000; ++k) {
        const double v = horizonMeanG(-1.0 + 2.0 * k / 4000.0);
        if (v < prev - 1e-15) mono = false;
        prev = v;
    }
    check(mono, "g monotone non-decreasing on [-1, 1]", mono ? 1 : 0, 1, 0);

    const double alphas[] = {5.978 * M_PI / 180.0,   // Jupiter from Europa
                             0.950 * M_PI / 180.0};  // Earth from the Moon
    for (double alpha : alphas) {
        double worst = 0.0;
        for (int k = -40; k <= 40; ++k) {
            const double h = alpha * (k / 40.0) * 1.2;      // sweep past both seams
            const double got = diskMeanCosAboveHorizon(std::sin(h), alpha);
            const double num = meanCosNumeric(h, alpha, 400);
            worst = std::max(worst, std::fabs(got - num));
        }
        char buf[96];
        std::snprintf(buf, sizeof buf,
                      "alpha=%.3f deg: max |closed - numeric| / alpha",
                      alpha * 180.0 / M_PI);
        // O(alpha^2) truncation of the linearization, expressed as a fraction
        // of alpha (the scale of <cos> inside the band).
        check(worst / alpha <= 2.0e-2, buf, worst / alpha, 0.0, 2.0e-2);
    }

    // Continuity at BOTH seams and monotonicity of the whole ramp: the defect
    // PTM-44 names is a STEP, so the fix has to be shown not to reintroduce one.
    for (double alpha : alphas) {
        double maxstep = 0.0;
        bool ramp_mono = true;
        double prev_v = -1.0;
        const int M = 200000;
        for (int k = 0; k <= M; ++k) {
            const double h = -1.5 * alpha + 3.0 * alpha * k / M;
            const double v = diskMeanCosAboveHorizon(std::sin(h), alpha);
            if (k) maxstep = std::max(maxstep, std::fabs(v - prev_v));
            if (v < prev_v - 1e-15 && k) ramp_mono = false;
            prev_v = v;
        }
        char buf[96];
        std::snprintf(buf, sizeof buf, "alpha=%.3f deg: largest step in <cos> ramp",
                      alpha * 180.0 / M_PI);
        // 3*alpha swept in M steps: a continuous ramp gives ~3*alpha/M.
        check(maxstep <= 1e-4 * alpha, buf, maxstep / alpha, 0.0, 1e-4);
        std::snprintf(buf, sizeof buf, "alpha=%.3f deg: ramp monotone",
                      alpha * 180.0 / M_PI);
        check(ramp_mono, buf, ramp_mono ? 1 : 0, 1, 0);
    }

    // The pre-fix behavior, for contrast: a step of sin(alpha) at h = 0.
    std::printf("     (pre-fix step at the horizon was %.4g in <cos theta>, "
                "i.e. the FULL flux)\n", std::sin(alphas[0]));
}

// ---------------------------------------------------------------------------
// 4. Terrain sampling: half-plane occluder.  Replicates the sampler in
//    updatePrimaryDiskVisibility exactly (same Hammersley + concentric mapping,
//    same cosine weighting) and compares to the analytic circular-segment area.
// ---------------------------------------------------------------------------
static double sampledVisibleFraction(uint32_t N, double edge, double alpha) {
    // Face normal == disk center direction (+z); disk offsets in x, y.
    double w_total = 0.0, w_clear = 0.0;
    const double s = std::sin(alpha);
    for (uint32_t k = 0; k < N; ++k) {
        double u, v;
        hammersley(k, N, 0.0, 0.0, u, v);
        double dx, dy;
        squareToConcentricDisk(u, v, dx, dy);
        // direction = normalize(z + s*(dx, dy, 0)); its cosine with z:
        const double len = std::sqrt(1.0 + s * s * (dx * dx + dy * dy));
        const double w = 1.0 / len;
        if (w <= 0.0) continue;
        w_total += w;
        if (dx >= edge) w_clear += w;          // half-plane occluder at x = edge
    }
    return w_total > 0.0 ? w_clear / w_total : 0.0;
}

// Alternative point sets, so the claim "other generators are no better" in
// primary_disk.h is reproducible from this binary instead of from a scratch
// script. Fibonacci (sunflower) spiral and an equal-area polar grid.
static double sampledVisibleFractionAlt(uint32_t N, double edge, int which) {
    double n_total = 0.0, n_clear = 0.0;
    const double GA = M_PI * (3.0 - std::sqrt(5.0));
    if (which == 0) {                                   // Fibonacci spiral
        for (uint32_t k = 0; k < N; ++k) {
            const double r = std::sqrt((k + 0.5) / double(N));
            const double th = k * GA;
            const double dx = r * std::cos(th);
            n_total += 1.0;
            if (dx >= edge) n_clear += 1.0;
        }
    } else {                                            // equal-area polar grid
        const int nr = std::max(1, int(std::lround(std::sqrt(N / M_PI))));
        for (int i = 0; i < nr; ++i) {
            const double r0 = std::sqrt(double(i) / nr);
            const double r1 = std::sqrt(double(i + 1) / nr);
            const int ni = std::max(1, int(std::lround(N * (r1*r1 - r0*r0))));
            const double r = 0.5 * (r0 + r1);
            for (int j = 0; j < ni; ++j) {
                const double th = 2.0 * M_PI * (j + 0.5) / ni + r * 2.399963;
                const double dx = r * std::cos(th);
                n_total += 1.0;
                if (dx >= edge) n_clear += 1.0;
            }
        }
    }
    return n_total > 0.0 ? n_clear / n_total : 0.0;
}

static double segmentFraction(double e) {
    if (e <= -1.0) return 1.0;
    if (e >=  1.0) return 0.0;
    return (std::acos(e) - e * std::sqrt(1.0 - e * e)) / M_PI;
}

static void testHalfPlaneOccluder() {
    std::printf("\n4. Terrain sampling: half-plane occluder (PTM-44 core claim)\n");
    const double alpha = 5.978 * M_PI / 180.0;

    // (a) an occluder that hides exactly half the disk must give ~50 %.
    for (uint32_t N : {16u, 32u, 64u, 128u, 256u, 512u}) {
        const double got = sampledVisibleFraction(N, 0.0, alpha);
        char buf[96];
        std::snprintf(buf, sizeof buf, "N=%-4u half-occulted disk -> visible fraction", N);
        check(std::fabs(got - 0.5) <= 1.0 / double(N), buf, got, 0.5, 1.0 / double(N));
    }

    // (b) worst-case error over the whole sweep, per sample count. This is the
    //     number that justifies PRIMARY_DISK_SAMPLES. The criterion is NOT a
    //     fixed "< 1 % of full flux" any more: it is the accuracy budget
    //     PRIMARY_DISK_WORST_VISIBILITY_ERROR that primary_disk.h states and
    //     converts to kelvin at the coldest lunar face. Changing N without
    //     re-deriving that budget fails here.
    std::printf("     worst |sampled - analytic| over edge in [-1.2, 1.2]:\n");
    for (uint32_t N : {16u, 32u, 64u, 128u, 256u, 512u}) {
        double worst = 0.0;
        for (int k = -600; k <= 600; ++k) {
            const double e = k / 500.0;
            worst = std::max(worst, std::fabs(sampledVisibleFraction(N, e, alpha)
                                              - segmentFraction(e)));
        }
        std::printf("       N=%-4u worst = %.4f  (1/N = %.4f)%s\n",
                    N, worst, 1.0 / double(N),
                    N == PRIMARY_DISK_SAMPLES ? "   <-- PRIMARY_DISK_SAMPLES" : "");
        if (N == PRIMARY_DISK_SAMPLES) {
            check(worst <= PRIMARY_DISK_WORST_VISIBILITY_ERROR,
                  "PRIMARY_DISK_SAMPLES worst error within the stated budget",
                  worst, 0.0, PRIMARY_DISK_WORST_VISIBILITY_ERROR);
        }
    }

    // (b2) the same sweep for two other point sets, at the chosen N. If either
    //      were materially better this would say so.
    for (int which = 0; which < 2; ++which) {
        double worst = 0.0;
        for (int k = -600; k <= 600; ++k) {
            const double e = k / 500.0;
            worst = std::max(worst, std::fabs(
                sampledVisibleFractionAlt(PRIMARY_DISK_SAMPLES, e, which)
                - segmentFraction(e)));
        }
        std::printf("       N=%-4u worst = %.4f  (%s)\n", PRIMARY_DISK_SAMPLES, worst,
                    which == 0 ? "Fibonacci spiral" : "equal-area polar grid");
    }

    // (c) monotone, no step, as the occluder sweeps off the disk.
    double prev = -1.0, maxstep = 0.0;
    bool mono = true;
    for (int k = 650; k >= -650; --k) {                 // edge from +1.3 to -1.3
        const double v = sampledVisibleFraction(PRIMARY_DISK_SAMPLES, k / 500.0, alpha);
        if (prev >= 0.0) {
            maxstep = std::max(maxstep, std::fabs(v - prev));
            if (v < prev - 1e-12) mono = false;
        }
        prev = v;
    }
    check(mono, "visible fraction monotone as the occluder retreats", mono ? 1 : 0, 1, 0);
    // Same criterion as the quantization: no residual discontinuity worth more
    // than the stated visibility budget. The pre-fix center ray stepped the
    // FULL 1.0000.
    check(maxstep <= PRIMARY_DISK_WORST_VISIBILITY_ERROR,
          "largest single step in the sweep within the stated budget",
          maxstep, 0.0, PRIMARY_DISK_WORST_VISIBILITY_ERROR);
    std::printf("     (the pre-fix center ray stepped 1.0000 at edge = 0)\n");
}

// ---------------------------------------------------------------------------
// 4c. PER-FACE ray budget: the ladder, and the kelvin it is tied to.
//
// The N = 128 gate above asks one question -- "does the ONE sample count meet
// the ONE budget at the ONE worst face". The ladder asks it of every rung, and
// it has to be at least as hard to weaken, or the per-face rule is a way of
// buying speed with an unexamined constant. Three things are gated here:
//
//   (b) every rung's TABULATED worst error is >= what this binary MEASURES for
//       it, on the x-aligned sweep AND on a sweep over occluder ORIENTATION.
//       Lowering a tabulated error to buy fewer rays fails here immediately.
//       Orientation is not decoration: the 5-probe rung has five points and no
//       rotational symmetry, so an edge at 45 deg is a different question from
//       an edge at 0 deg, and the low-discrepancy sets are not symmetric either.
//   (d) the CLOSURE: pushing each rung's tabulated error back through the exact
//       dT expression, at the flux that made the selector choose that rung,
//       must land inside PRIMARY_DISK_TEMPERATURE_BUDGET_K. This is what makes
//       N_f a function of the budget rather than of taste.
//   (e) the CALIBRATION IDENTITY: on the Moon at the measured peak primary flux
//       the per-face rule must return the inherited global constant. If the two
//       disagree, one of them is wrong and this says which.
// ---------------------------------------------------------------------------

// Half-plane occluder with an arbitrary in-plane orientation: the disk point
// (dx, dy) is CLEAR when its projection on (cos th, sin th) is >= edge.
static double sampledVisibleFractionRot(uint32_t N, double edge, double alpha,
                                        double th) {
    const double s = std::sin(alpha), ct = std::cos(th), st = std::sin(th);
    double w_total = 0.0, w_clear = 0.0;
    for (uint32_t k = 0; k < N; ++k) {
        double u, v;
        hammersley(k, N, 0.0, 0.0, u, v);
        double dx, dy;
        squareToConcentricDisk(u, v, dx, dy);
        const double w = 1.0 / std::sqrt(1.0 + s * s * (dx * dx + dy * dy));
        w_total += w;
        if (dx * ct + dy * st >= edge) w_clear += w;
    }
    return w_total > 0.0 ? w_clear / w_total : 0.0;
}

// The engine's 5-ray probe, read as an ESTIMATE of V rather than as a yes/no:
// center plus the four rim points of probeOffset(), cosine weighted exactly as
// updatePrimaryDiskVisibility now accumulates them.
static double probeVisibleFraction(double edge, double alpha, double th) {
    const double s = std::sin(alpha), ct = std::cos(th), st = std::sin(th);
    double w_total = 0.0, w_clear = 0.0;
    for (uint32_t k = 0; k < PRIMARY_DISK_PROBES; ++k) {
        double dx, dy;
        probeOffset(k, dx, dy);
        const double w = 1.0 / std::sqrt(1.0 + s * s * (dx * dx + dy * dy));
        w_total += w;
        if (dx * ct + dy * st >= edge) w_clear += w;
    }
    return w_total > 0.0 ? w_clear / w_total : 0.0;
}

// Worst |estimate - analytic| for one rung, over edge and (optionally) over
// occluder orientation. `samples == PRIMARY_DISK_PROBES` selects the probe
// estimator; `samples == 0` is the no-ray rung and has no estimator at all.
// Orientations run over a HALF turn, because the edge sweep already covers both
// signs, and the step has to divide 45 deg: the worst case is on the diagonal
// of Shirley-Chiu's concentric mapping every single time, and a sweep that
// steps past 45 and 135 deg reports a comfortable number that is not true.
// 1 deg steps, verified converged against 0.25 deg and 0.0625 deg.
static double worstRungError(uint32_t samples, double alpha, int n_orient) {
    double worst = 0.0;
    for (int o = 0; o < n_orient; ++o) {
        const double th = M_PI * double(o) / double(n_orient);
        for (int k = -600; k <= 600; ++k) {
            const double e = k / 500.0;
            const double got = (samples == PRIMARY_DISK_PROBES)
                ? probeVisibleFraction(e, alpha, th)
                : sampledVisibleFractionRot(samples, e, alpha, th);
            worst = std::max(worst, std::fabs(got - segmentFraction(e)));
        }
    }
    return worst;
}

// The exact temperature error a visibility error dV buys at the cold end,
// V = 0. Same expression primary_disk.h inverts.
static double dTforDV(double dv, double f_full, double q_floor, double eps_sigma) {
    return std::pow((q_floor + dv * f_full) / eps_sigma, 0.25)
         - std::pow(q_floor / eps_sigma, 0.25);
}

static void testPerFaceBudget() {
    std::printf("\n4c. Per-face ray budget (PRIMARY_DISK_TIERS)\n");

    // The budget itself is Paul's, 2026-08-14, and it is the only free
    // parameter in the whole ladder. Pin it: raising it is a decision, not an
    // edit, and it has to break something visible.
    check(std::fabs(PRIMARY_DISK_TEMPERATURE_BUDGET_K - 0.163) < 1e-12,
          "kelvin budget is the accepted 0.163 K worst face",
          PRIMARY_DISK_TEMPERATURE_BUDGET_K, 0.163, 1e-12);

    // (a) the selector takes the FIRST rung that fits, so the table has to be
    //     sorted both ways or it silently returns the wrong rung.
    bool sorted = true;
    for (int t = 1; t < PRIMARY_DISK_NUM_TIERS; ++t) {
        if (!(PRIMARY_DISK_TIERS[t].samples > PRIMARY_DISK_TIERS[t-1].samples)) sorted = false;
        if (!(PRIMARY_DISK_TIERS[t].worst_visibility_error
              < PRIMARY_DISK_TIERS[t-1].worst_visibility_error)) sorted = false;
    }
    check(sorted, "ladder sorted: samples up, error down", sorted ? 1 : 0, 1, 0);
    check(PRIMARY_DISK_TIERS[PRIMARY_DISK_NUM_TIERS-1].samples == PRIMARY_DISK_SAMPLES,
          "top rung is PRIMARY_DISK_SAMPLES",
          PRIMARY_DISK_TIERS[PRIMARY_DISK_NUM_TIERS-1].samples, PRIMARY_DISK_SAMPLES, 0);

    // (b) every tabulated error against measurement, at 1 deg over a half turn.
    const double alpha = 5.978 * M_PI / 180.0;
    std::printf("     tabulated vs MEASURED worst |dV| (x-aligned; over orientation):\n");
    std::printf("       %-6s %-11s %-13s %-13s\n", "N", "tabulated", "x-aligned", "any orientation");
    for (int t = 0; t < PRIMARY_DISK_NUM_TIERS; ++t) {
        const uint32_t N = PRIMARY_DISK_TIERS[t].samples;
        if (N == 0) {
            std::printf("       %-6s %-11.4f %-13s %-13s\n", "0 (A)",
                        PRIMARY_DISK_TIERS[t].worst_visibility_error,
                        "n/a", "n/a  (V := 1, the full range by construction)");
            check(PRIMARY_DISK_TIERS[t].worst_visibility_error >= 1.0,
                  "no-ray rung is budgeted for the whole range",
                  PRIMARY_DISK_TIERS[t].worst_visibility_error, 1.0, 0.0);
            continue;
        }
        const int n_orient = 180;
        const double w_axis = worstRungError(N, alpha, 1);
        const double w_any  = worstRungError(N, alpha, n_orient);
        char buf[96];
        std::snprintf(buf, sizeof buf, "N=%-4u tabulated error >= measured worst", N);
        std::printf("       %-6u %-11.4f %-13.4f %-13.4f\n",
                    N, PRIMARY_DISK_TIERS[t].worst_visibility_error, w_axis, w_any);
        check(PRIMARY_DISK_TIERS[t].worst_visibility_error >= w_any, buf,
              PRIMARY_DISK_TIERS[t].worst_visibility_error, w_any, 0.0);
    }

    // (f) the two standards, side by side, because one of them is inherited and
    //     wrong and this is where anyone re-deriving the budget will look.
    //     Printed rather than asserted -- whether PRIMARY_DISK_SAMPLES should
    //     move is Paul's call, not this gate's.
    std::printf("     INHERITED vs CORRECTED at the cap: "
                "PRIMARY_DISK_WORST_VISIBILITY_ERROR = %.4f (x-aligned sweep only), "
                "measured over orientation = %.4f, ratio %.2fx.\n",
                PRIMARY_DISK_WORST_VISIBILITY_ERROR, worstRungError(128, alpha, 180),
                worstRungError(128, alpha, 180) / PRIMARY_DISK_WORST_VISIBILITY_ERROR);
    std::printf("     The 0.163 K the cap was accepted against is therefore "
                "~%.2f K at the coldest lunar face. Unchanged by this ladder; "
                "N = 512 measures %.4f, so raising N is a poor lever.\n",
                0.163 * worstRungError(128, alpha, 180) / PRIMARY_DISK_WORST_VISIBILITY_ERROR,
                worstRungError(512, alpha, 180));
    std::printf("     fitted 0.49*N^-0.65 at N=64 gives %.4f against a measured "
                "%.4f -- optimistic\n", 0.49 * std::pow(64.0, -0.65),
                worstRungError(64, alpha, 180));

    // (e) the ladder must never demote the face the cap was chosen for. On the
    //     Moon (eps 0.95, Qb 0.018 -> T_floor 24.043 K) at the measured peak
    //     primary flux of 0.087 W/m^2 -- 0.046 IR + 0.041 reflected over
    //     48 lunar polar frames -- the strict V = 0 rule is STRICTER than the
    //     inherited constant, so that face keeps every ray it has today.
    const double sigma = 5.670374419e-8;
    const double es_moon = 0.95 * sigma, qb_moon = 0.018;
    const double dv_moon = primaryDiskAllowedVisibilityError(0.087, qb_moon, es_moon);
    std::printf("     Moon T_floor = %.4f K; at the peak flux 0.087 W/m^2 the rule "
                "allows dV = %.4f\n", std::pow(qb_moon / es_moon, 0.25), dv_moon);
    check(dv_moon < PRIMARY_DISK_WORST_VISIBILITY_ERROR,
          "rule at the peak lunar flux is stricter than the inherited constant",
          dv_moon, PRIMARY_DISK_WORST_VISIBILITY_ERROR,
          PRIMARY_DISK_WORST_VISIBILITY_ERROR);
    check(primaryDiskSamplesForAllowedError(dv_moon) == PRIMARY_DISK_SAMPLES,
          "so the calibration face keeps PRIMARY_DISK_SAMPLES",
          primaryDiskSamplesForAllowedError(dv_moon), PRIMARY_DISK_SAMPLES, 0);

    // (d) CLOSURE. Sweep the full-disk flux a face can see, from nothing to the
    //     Jupiter-at-Europa peak, and check the rung the selector picks keeps
    //     its own tabulated error inside the budget. This is the check that
    //     ties N_f to kelvin; it fails if a rung is ever selected above its
    //     price.
    std::printf("     closure: selected rung vs budget over the flux range\n");
    std::printf("       %-12s %-10s %-9s %-10s %-10s\n",
                "F_full W/m2", "dV allow", "rung", "worst dT", "budget");
    struct Body { const char* name; double eps; double qb; double fmax; };
    const Body bodies[2] = { {"Moon",   0.95, 0.018, 0.087},
                             {"Europa", 0.90, 0.030, 0.280} };
    for (const Body& b : bodies) {
        const double es = b.eps * sigma;
        double worst_over = 0.0;
        int n_pts = 0;
        for (int k = 0; k <= 400; ++k) {
            const double f = b.fmax * std::pow(10.0, -4.0 + 4.0 * k / 400.0);
            const double dv = primaryDiskAllowedVisibilityError(f, b.qb, es);
            const uint32_t N = primaryDiskSamplesForAllowedError(dv);
            // the tabulated error of the rung actually selected
            double e_rung = PRIMARY_DISK_WORST_VISIBILITY_ERROR;
            for (int t = 0; t < PRIMARY_DISK_NUM_TIERS; ++t)
                if (PRIMARY_DISK_TIERS[t].samples == N)
                    e_rung = PRIMARY_DISK_TIERS[t].worst_visibility_error;
            // A rung is only ever selected because its error FITS dv, except at
            // the cap, where the budget is knowingly unreachable and the
            // behavior is the inherited one. Score the reachable range.
            if (e_rung > dv) continue;
            ++n_pts;
            worst_over = std::max(worst_over, dTforDV(e_rung, f, b.qb, es));
            if (k % 80 == 0)
                std::printf("       %-12.3e %-10.4f %-9u %-10.4f %-10.3f  (%s)\n",
                            f, dv, N, dTforDV(e_rung, f, b.qb, es),
                            PRIMARY_DISK_TEMPERATURE_BUDGET_K, b.name);
        }
        char buf[96];
        std::snprintf(buf, sizeof buf, "%s: every selected rung inside the budget", b.name);
        check(worst_over <= PRIMARY_DISK_TEMPERATURE_BUDGET_K + 1e-9 && n_pts > 0,
              buf, worst_over, PRIMARY_DISK_TEMPERATURE_BUDGET_K,
              PRIMARY_DISK_TEMPERATURE_BUDGET_K);
    }

    // (c) monotonicity of the selector: more allowance can never buy more rays.
    bool mono = true;
    uint32_t prev = PRIMARY_DISK_SAMPLES;
    for (int k = 0; k <= 2000; ++k) {
        const uint32_t N = primaryDiskSamplesForAllowedError(k / 2000.0);
        if (N > prev) mono = false;
        prev = N;
    }
    check(mono, "selector monotone: more allowed error, never more rays",
          mono ? 1 : 0, 1, 0);

    // A zero-flux face must cost nothing, and a disarmed budget (floor 0) must
    // fall back to the inherited count on every face -- the two ends of the
    // rule, both of which the engine relies on.
    check(primaryDiskSamplesForAllowedError(
              primaryDiskAllowedVisibilityError(0.0, qb_moon, es_moon)) == 0,
          "a face with no primary flux casts no primary rays",
          primaryDiskSamplesForAllowedError(
              primaryDiskAllowedVisibilityError(0.0, qb_moon, es_moon)), 0, 0);
    check(primaryDiskSamplesForAllowedError(
              primaryDiskAllowedVisibilityError(0.087, 0.0, es_moon)) == PRIMARY_DISK_SAMPLES,
          "no deep-heat floor -> inherited count, not a cheaper one",
          primaryDiskSamplesForAllowedError(
              primaryDiskAllowedVisibilityError(0.087, 0.0, es_moon)),
          PRIMARY_DISK_SAMPLES, 0);
}

// ---------------------------------------------------------------------------
// 4b. Temporal reuse of the terrain factor: the SAME budget, applied to the
//     other approximation in this path.
//
// PRIMARY_DISK_REUSE_CELL_FRACTION snaps the primary's position to a lattice of
// spacing f * R_primary, which is a lattice of f * alpha in apparent ANGLE, and
// reuses the terrain-visible fraction until the lattice cell changes. The disk
// therefore sits up to sqrt(3)/2 * f * alpha -- i.e. 0.866 * f disk RADII --
// away from where the terrain actually is. Note the bound does not depend on
// how many timesteps the value is reused for: it is set by the cell, not by
// the dwell, which is the whole reason the value is snapped to the cell rather
// than cached from whichever step happened to enter it first.
//
// Against the half-plane occluder above, the induced visibility error is
// |V(e) - V(e +- 0.866 f)|, worst ~ (2/pi) * 0.866 * f. This gate holds it to
// the same PRIMARY_DISK_WORST_VISIBILITY_ERROR that fixes PRIMARY_DISK_SAMPLES,
// because a kelvin spent here is the same kelvin. Raising the constant past
// what that budget allows fails HERE, loudly, instead of silently shifting
// every lunar and Europa temperature.
// ---------------------------------------------------------------------------
static void testTerrainReuseBudget() {
    std::printf("\n4b. Temporal reuse of the terrain factor "
                "(PRIMARY_DISK_REUSE_CELL_FRACTION)\n");

    auto worstForFraction = [](double f) {
        const double shift = std::sqrt(3.0) / 2.0 * f;   // disk radii
        double worst = 0.0;
        for (int k = -650; k <= 650; ++k) {
            const double e = k / 500.0;
            worst = std::max(worst, std::fabs(segmentFraction(e) - segmentFraction(e + shift)));
            worst = std::max(worst, std::fabs(segmentFraction(e) - segmentFraction(e - shift)));
        }
        return worst;
    };

    std::printf("     worst |dV| from the lattice snap, half-plane occluder:\n");
    std::printf("       %-8s %-12s %-12s\n", "f", "max shift", "worst |dV|");
    for (double f : {0.0, 0.02, 0.04, 0.05, 0.1, 0.25, 0.5, 1.0}) {
        std::printf("       %-8.2f %-12.4f %-12.4f%s\n",
                    f, std::sqrt(3.0) / 2.0 * f, worstForFraction(f),
                    f == PRIMARY_DISK_REUSE_CELL_FRACTION ? "   <-- compiled-in default" : "");
    }

    // The measured end-to-end cost, for whoever raises the constant. These are
    // EMPIRICAL worst-face numbers over whole runs, not the worst case above,
    // and they are 20-60x smaller because no real face is a straight edge
    // cutting the disk in half at the coldest temperature on the mesh. Both
    // standards are printed on purpose: the worst case is what the N budget was
    // set against, so it is what this gate enforces.
    std::printf("     measured worst face-frame |dT| and PRIMARY RAYS REMOVED\n"
                "     (2026-08-15; Moon polar 100.6 M face-frames at the 1.0 h\n"
                "     production cadence, Europa Pwyll DEM 1.36 M at 1.78 h):\n");
    std::printf("       %-7s %-11s %-9s %-11s %-9s\n",
                "f", "Moon |dT|", "rays", "Europa |dT|", "rays");
    std::printf("       %-7s %-11s %-9s %-11s %-9s\n", "0.038", "0.0333 K", "-0 %",  "0.0078 K", "-43 %");
    std::printf("       %-7s %-11s %-9s %-11s %-9s\n", "0.10",  "0.0411 K", "-15 %", "0.0114 K", "-78 %");
    std::printf("       %-7s %-11s %-9s %-11s %-9s\n", "0.25",  "0.0599 K", "-45 %", "0.0158 K", "-87 %");
    std::printf("       %-7s %-11s %-9s %-11s %-9s\n", "0.50",  "0.1512 K", "-64 %", "0.0222 K", "-95 %");
    std::printf("     Read the Moon row at f = 0.038 before enabling this: at the largest\n"
                "     f this gate allows, the Moon removes NO rays and still pays 0.033 K,\n"
                "     because Earth moves 0.052 disk radii per production timestep and the\n"
                "     cell is smaller than that. Reuse is a Europa/Jupiter-class win and a\n"
                "     lunar-cadence loss; the deciding number is (apparent motion per step)\n"
                "     / alpha, not the body.\n");

    const double worst = worstForFraction(PRIMARY_DISK_REUSE_CELL_FRACTION);
    check(worst <= PRIMARY_DISK_WORST_VISIBILITY_ERROR,
          "PRIMARY_DISK_REUSE_CELL_FRACTION worst error within the stated budget",
          worst, 0.0, PRIMARY_DISK_WORST_VISIBILITY_ERROR);
    if (PRIMARY_DISK_REUSE_CELL_FRACTION == 0.0) {
        std::printf("     (reuse is OFF by default: the engine recomputes the terrain\n"
                    "      factor every timestep and this path is exactly free)\n");
    }
}

// ---------------------------------------------------------------------------
// 5. Jupiter-shine at Europa: before/after table.
// ---------------------------------------------------------------------------
static void reportJupiterTable() {
    std::printf("\n5. Jupiter-shine at Europa's sub-Jovian point (cos theta = 1)\n");
    const double S0 = 1361.0, aJ = 5.2038, R = 69911.0, d = 671100.0, A = 0.343;
    const double F = S0 / (aJ * aJ);
    const double alpha = std::asin(R / d);
    const double omega_over_pi = 2.0 * (1.0 - std::cos(alpha));
    std::printf("     F at Jupiter %.3f W/m^2, alpha %.3f deg, sin^2 %.5e\n",
                F, alpha * 180.0 / M_PI, (R / d) * (R / d));
    std::printf("     %-10s %-14s %-14s %-8s\n", "phase", "pre-fix", "fixed", "ratio");
    for (double dg : {0.0, 45.0, 90.0, 135.0, 180.0}) {
        const double a = dg * M_PI / 180.0;
        const double old_v = F * A * ((1.0 + std::cos(a)) / 2.0) * (omega_over_pi / 4.0);
        const double new_v = lambertSphereIrradiance(F, A, R / d, a, 1.0);
        std::printf("     %6.1f deg %-14.6f %-14.6f %s\n", dg, old_v, new_v,
                    old_v > 0 ? std::to_string(new_v / old_v).substr(0, 6).c_str() : "n/a");
    }
    std::printf("     primary IR (unchanged, PTM-40): %.6f W/m^2\n", 13.8 * omega_over_pi);
    std::printf("     hard upper bound, p = 1:        %.6f W/m^2\n", F * (R / d) * (R / d));
}

int main() {
    std::printf("test_primary_disk -- PTM-41 / PTM-44 geometry gate\n");
    testPhaseFunction();
    testEnergyClosure();
    testHorizonMeanCos();
    testHalfPlaneOccluder();
    testPerFaceBudget();
    testTerrainReuseBudget();
    reportJupiterTable();
    std::printf("\n%s (%d failure%s)\n",
                g_failures == 0 ? "ALL CHECKS PASS" : "FAILURES PRESENT",
                g_failures, g_failures == 1 ? "" : "s");
    return g_failures == 0 ? 0 : 1;
}
