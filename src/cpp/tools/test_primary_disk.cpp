// test_primary_disk -- standalone gate for the PTM-41 / PTM-44 primary-body
// flux geometry in src/cpp/illumination/primary_disk.h.
//
// Everything here runs on the CPU in well under a second and needs no SPICE,
// no mesh and no Embree: it exercises the closed forms directly against
// numerical truth, so a failure localises to the physics and not to the
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
// 1. Lambert-sphere phase function vs the lune integral it summarises.
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
    // centre sits at altitude h; theta measured from the face normal, so
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
        // O(alpha^2) truncation of the linearisation, expressed as a fraction
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

    // The pre-fix behaviour, for contrast: a step of sin(alpha) at h = 0.
    std::printf("     (pre-fix step at the horizon was %.4g in <cos theta>, "
                "i.e. the FULL flux)\n", std::sin(alphas[0]));
}

// ---------------------------------------------------------------------------
// 4. Terrain sampling: half-plane occluder.  Replicates the sampler in
//    updatePrimaryDiskVisibility exactly (same Hammersley + concentric mapping,
//    same cosine weighting) and compares to the analytic circular-segment area.
// ---------------------------------------------------------------------------
static double sampledVisibleFraction(uint32_t N, double edge, double alpha) {
    // Face normal == disk centre direction (+z); disk offsets in x, y.
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
    //     number that justifies PRIMARY_DISK_SAMPLES: the CRITERION is < 1 % of
    //     full flux.
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
            check(worst < 0.01,
                  "PRIMARY_DISK_SAMPLES worst error < 1 % of full flux",
                  worst, 0.0, 0.01);
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
    // Same criterion as the quantisation: no residual discontinuity worth more
    // than 1 % of full flux. The pre-fix centre ray stepped the FULL 1.0000.
    check(maxstep <= 0.01, "largest single step in the sweep < 1 % of full flux",
          maxstep, 0.0, 0.01);
    std::printf("     (the pre-fix centre ray stepped 1.0000 at edge = 0)\n");
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
    reportJupiterTable();
    std::printf("\n%s (%d failure%s)\n",
                g_failures == 0 ? "ALL CHECKS PASS" : "FAILURES PRESENT",
                g_failures, g_failures == 1 ? "" : "s");
    return g_failures == 0 ? 0 : 1;
}
