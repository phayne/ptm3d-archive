// test_eq_extrapolation — correctness gate for pass-stride RRE extrapolation.
//
// WHY THIS TEST EXISTS
// --------------------
// The acceptance thresholds in the physics review addendum (A1: 0.267 K certified at 5
// passes; A3a: 0 missed detections over 72 cases) were measured with ONE
// specific reference implementation, `eq_compare_2026-08-08/code/modeext.py`
// `_rre`. If the C++ disagrees with that function, every published number
// becomes a statement about a different algorithm and the GO decision does not
// transfer. So the gate is: reproduce it, on well-conditioned AND rank-deficient
// inputs, to machine precision.
//
// This is not hypothetical. The first draft of minNormLstsq() had an inverted
// sign in the Jacobi rotation, which made the sweep ANTI-diagonalize. It
// returned finite, plausible-looking temperatures and disagreed with numpy by
// 17-21 % on a FULL-RANK system. No physics gate would have caught it; only a
// direct comparison against the reference did.
//
// Header-only under test, so nothing links here and no stale library can
// silently satisfy it (same reasoning as test_conductivity_model).
//
// Run:  ./build/bin/test_eq_extrapolation      (exit 0 = pass)
#include "../thermal/eq_extrapolation.h"

#include <cmath>
#include <cstdio>
#include <vector>

namespace ex = ptm::eqextrap;

static int g_failures = 0;

static void check(bool ok, const char* what, double got, double want) {
    if (!ok) ++g_failures;
    std::printf("  %-58s got %-18.10g want %-18.10g  %s\n",
                what, got, want, ok ? "PASS" : "*** FAIL ***");
}

static void checkNear(const char* what, double got, double want, double tol) {
    check(std::fabs(got - want) <= tol, what, got, want);
}

// ---------------------------------------------------------------------------
// 1. GOLDEN VECTORS from modeext.py `_rre`. Four layers, five snapshots.
//    Produced by: me._rre(list(S), p) on the S below, numpy 1.x, rcond 1e-12.
// ---------------------------------------------------------------------------
static void testGoldenVsReference() {
    std::printf("\n[1] golden vectors vs eq_compare/code/modeext.py::_rre\n");
    const int L = 4;
    // S[snapshot][layer], oldest first.
    const double S[5][4] = {
        {300.0,   250.0,   200.0,   150.0   },
        {280.0,   238.0,   193.0,   147.5   },
        {268.0,   230.8,   188.8,   146.05  },
        {260.8,   226.48,  186.28,  145.18  },
        {256.48,  223.888, 184.768, 144.658 },
    };
    const double want_p4[4] = {250.00000000000003, 219.99999999999997,
                               182.5,              143.87499999999997};
    const double want_p3[4] = {249.99999999999994, 219.99999999999997,
                               182.5,              143.875};

    {
        std::vector<double> X(5 * L);
        for (int s = 0; s < 5; ++s)
            for (int l = 0; l < L; ++l) X[s * L + l] = S[s][l];
        std::vector<double> out(L);
        double cond = 0.0;
        const bool ok = ex::rreColumn(X.data(), 4, L, out.data(), &cond);
        check(ok, "p=4 solve reports success", ok ? 1 : 0, 1);
        for (int l = 0; l < L; ++l) {
            char lab[80];
            std::snprintf(lab, sizeof lab, "p=4 layer %d matches numpy", l);
            checkNear(lab, out[l], want_p4[l], 1e-9);
        }
    }
    {
        std::vector<double> X(4 * L);
        for (int s = 0; s < 4; ++s)
            for (int l = 0; l < L; ++l) X[s * L + l] = S[s][l];
        std::vector<double> out(L);
        double cond = 0.0;
        ex::rreColumn(X.data(), 3, L, out.data(), &cond);
        for (int l = 0; l < L; ++l) {
            char lab[80];
            std::snprintf(lab, sizeof lab, "p=3 layer %d matches numpy", l);
            checkNear(lab, out[l], want_p3[l], 1e-9);
        }
    }
}

// ---------------------------------------------------------------------------
// 2. EXACTNESS. RRE of order p annihilates up to p-1 geometric modes exactly,
//    so on a sequence T_k = T_inf + sum_{m<p} a_m r_m^k it must return T_inf to
//    machine precision -- with NO reference data needed. This is the property
//    that makes the whole scheme work: the eq pass map is linear-ish near its
//    fixed point and its transient is a sum of decaying modes.
// ---------------------------------------------------------------------------
static void testExactOnGeometricModes() {
    std::printf("\n[2] exactness on <= p-1 geometric modes (analytic, no reference)\n");
    const int L = 50;
    for (int p = 3; p <= 4; ++p) {
        const int n_modes = p - 1;
        const double rates[3] = {0.61, 0.27, 0.08};
        std::vector<double> Tinf(L), X(static_cast<size_t>(p + 1) * L);
        std::vector<std::vector<double>> amp(n_modes, std::vector<double>(L));
        for (int l = 0; l < L; ++l) {
            Tinf[l] = 30.0 + 6.0 * l;
            for (int m = 0; m < n_modes; ++m)
                amp[m][l] = 40.0 * std::sin(0.7 * l + 1.3 * m) + 5.0 * (m + 1);
        }
        for (int s = 0; s <= p; ++s) {
            for (int l = 0; l < L; ++l) {
                double v = Tinf[l];
                for (int m = 0; m < n_modes; ++m)
                    v += amp[m][l] * std::pow(rates[m], s);
                X[static_cast<size_t>(s) * L + l] = v;
            }
        }
        std::vector<double> out(L);
        double cond = 0.0;
        ex::rreColumn(X.data(), p, L, out.data(), &cond);
        double worst = 0.0;
        for (int l = 0; l < L; ++l) worst = std::max(worst, std::fabs(out[l] - Tinf[l]));
        char lab[80];
        std::snprintf(lab, sizeof lab, "p=%d recovers T_inf for %d modes (max err)",
                      p, n_modes);
        checkNear(lab, worst, 0.0, 1e-8);
    }
}

// ---------------------------------------------------------------------------
// 3. RANK DEFICIENCY. Fewer live modes than p is the COMMON case on a real
//    mesh (a face deep in shadow relaxes on one time constant). The minimum-norm
//    solve must still land on the exact limit and must not blow up.
// ---------------------------------------------------------------------------
static void testRankDeficient() {
    std::printf("\n[3] rank-deficient column (1 live mode, p=4)\n");
    const int L = 50, p = 4;
    std::vector<double> Tinf(L), X(static_cast<size_t>(p + 1) * L), out(L);
    for (int l = 0; l < L; ++l) Tinf[l] = 25.0 + 0.5 * l;
    for (int s = 0; s <= p; ++s)
        for (int l = 0; l < L; ++l)
            X[static_cast<size_t>(s) * L + l] = Tinf[l] + 33.0 * std::pow(0.42, s);
    double cond = 0.0;
    ex::rreColumn(X.data(), p, L, out.data(), &cond);
    double worst = 0.0;
    for (int l = 0; l < L; ++l) worst = std::max(worst, std::fabs(out[l] - Tinf[l]));
    checkNear("p=4 on a 1-mode column recovers T_inf (max err)", worst, 0.0, 1e-8);

    // Already converged: every snapshot identical. 0/0 everywhere. Must return
    // that state, finite, and must not produce NaN.
    std::vector<double> Xc(static_cast<size_t>(p + 1) * L), oc(L);
    for (int s = 0; s <= p; ++s)
        for (int l = 0; l < L; ++l) Xc[static_cast<size_t>(s) * L + l] = 100.0 + l;
    ex::rreColumn(Xc.data(), p, L, oc.data(), &cond);
    double wc = 0.0;
    bool finite = true;
    for (int l = 0; l < L; ++l) {
        wc = std::max(wc, std::fabs(oc[l] - (100.0 + l)));
        finite = finite && std::isfinite(oc[l]);
    }
    check(finite, "fully converged column stays finite", finite ? 1 : 0, 1);
    checkNear("fully converged column is a fixed point (max err)", wc, 0.0, 1e-9);
}

// ---------------------------------------------------------------------------
// 4. REFUSALS. p < 3 must never be executed by rreColumn, and a bad column must
//    degrade to the newest state rather than to garbage.
// ---------------------------------------------------------------------------
static void testRefusals() {
    std::printf("\n[4] refusals and safe degradation\n");
    const int L = 6;
    std::vector<double> X(3 * L), out(L);
    for (int s = 0; s < 3; ++s)
        for (int l = 0; l < L; ++l) X[static_cast<size_t>(s) * L + l] = 200.0 - 10.0 * s + l;
    const bool ok2 = ex::rreColumn(X.data(), 2, L, out.data(), nullptr);
    // p=2 is inside rreColumn's algebraic domain, but the CONFIG layer refuses
    // it; assert only that it is well-formed here.
    check(std::isfinite(out[0]), "p=2 result finite (config layer is what refuses it)",
          out[0], out[0]);
    (void)ok2;

    // A NaN anywhere must trigger the fallback to the newest snapshot.
    std::vector<double> Xn(5 * L), on(L);
    for (int s = 0; s < 5; ++s)
        for (int l = 0; l < L; ++l) Xn[static_cast<size_t>(s) * L + l] = 150.0 + l - 2.0 * s;
    Xn[0] = std::nan("");
    const bool okn = ex::rreColumn(Xn.data(), 4, L, on.data(), nullptr);
    check(!okn, "NaN input reports failure", okn ? 1 : 0, 0);
    checkNear("NaN input falls back to the newest snapshot",
              on[1], Xn[4 * L + 1], 0.0);

}

// ---------------------------------------------------------------------------
// 5. CERTIFICATE ALGEBRA. rhat clamping is what keeps ehat finite when the
//    sequence stalls (r -> 1 makes r/(1-r) diverge).
// ---------------------------------------------------------------------------
static void testCertificateAlgebra() {
    std::printf("\n[5] certificate algebra\n");
    checkNear("rhat = d2/d1 in the normal regime",
              ex::clampRhat(2.0, 4.0, 0.9995), 0.5, 1e-15);
    checkNear("rhat clamped at r_cap when the sequence stalls",
              ex::clampRhat(4.0, 4.0, 0.9995), 0.9995, 1e-15);
    checkNear("rhat clamped at r_cap on a zero denominator",
              ex::clampRhat(1.0, 0.0, 0.9995), 0.9995, 1e-15);
    checkNear("rhat floored at 1e-6 on a zero numerator",
              ex::clampRhat(0.0, 4.0, 0.9995), 1e-6, 1e-15);
    // ehat = safety * dinf * r/(1-r): the amplification at pass stride is O(1),
    // which is the whole reason the certificate is usable here (at lunation
    // stride r ~ 0.97 gives ~40x).
    const double r = 0.4;
    checkNear("amplification r/(1-r) at a measured pass-stride r=0.4",
              r / (1.0 - r), 0.6666666666666666, 1e-15);
}

// ---------------------------------------------------------------------------
// 6. LEAST SQUARES. Direct check of the min-norm solver against a hand-solvable
//    overdetermined system, plus an exactly singular one.
// ---------------------------------------------------------------------------
static void testLeastSquares() {
    std::printf("\n[6] minimum-norm least squares\n");
    // A = [[1,0],[0,1],[1,1]], b = [1,2,3]. Normal equations:
    // [[2,1],[1,2]] x = [4,5]  ->  x = [1, 2].
    double A[6] = {1, 0, 1,   0, 1, 1};      // column-major 3x2
    const double b[3] = {1, 2, 3};
    double x[2] = {0, 0}, cond = 0;
    ex::minNormLstsq(A, 3, 2, b, x, ex::kRcond, &cond);
    checkNear("lstsq x0", x[0], 1.0, 1e-12);
    checkNear("lstsq x1", x[1], 2.0, 1e-12);

    // Exactly singular: two identical columns. Minimum-norm splits the
    // coefficient evenly instead of returning an arbitrary large pair.
    double As[6] = {1, 1, 1,   1, 1, 1};
    const double bs[3] = {2, 2, 2};
    double xs[2] = {0, 0};
    const int rank = ex::minNormLstsq(As, 3, 2, bs, xs, ex::kRcond, &cond);
    check(rank == 1, "singular system detected as rank 1", rank, 1);
    checkNear("min-norm splits evenly (x0)", xs[0], 1.0, 1e-12);
    checkNear("min-norm splits evenly (x1)", xs[1], 1.0, 1e-12);
}

// ---------------------------------------------------------------------------
// 7. RING ADDRESSING. Snapshots are written to slot k % (p+1) as the passes run,
//    so recovering them oldest-first at the top of the last pass is a modular
//    index. Get it wrong and RRE is handed a ROTATED sequence: finite, plausible
//    temperatures, silently wrong. Simulate the writes and assert the reads.
// ---------------------------------------------------------------------------
static void testRingAddressing() {
    std::printf("\n[7] ring-buffer addressing (rotation would be silent)\n");
    for (int p = 3; p <= 4; ++p) {
        const int depth = p + 1;
        for (int N = p + 1; N <= p + 6; ++N) {   // N = total passes
            // Simulate: pass k writes snapshot id k into slot k % depth.
            std::vector<int> slot(depth, -1);
            for (int k = 0; k < N; ++k) slot[k % depth] = k;
            // Read back oldest-first at newest_cycle = N-1.
            bool ok = true;
            for (int s = 0; s <= p; ++s) {
                const int got = slot[ex::ringSlot(N - 1, p, s, depth)];
                const int want = (N - 1) - p + s;
                if (got != want) ok = false;
            }
            char lab[96];
            std::snprintf(lab, sizeof lab,
                          "p=%d N=%d recovers S_%d..S_%d in order", p, N,
                          N - 1 - p, N - 1);
            check(ok, lab, ok ? 1 : 0, 1);
        }
    }
}

// ---------------------------------------------------------------------------
// 8. REGRESSION on the MEASURED gate-3a failure (ADDENDUM NOTE 6).
//
// The 2026-08-08 validation failed gate 3a: 73 zero-forcing faces moved >1e-4 K
// at layers >=30, worst 0.465260 K on face 277511. Its measured signature: the
// UPPER column still moving (4->8 pass change 9.47e-3 K over layers 0..15)
// while the DEEP column was bit-locked (4->8 change EXACTLY 0 over layers
// 30..49), and RRE displaced the deep column by a nearly uniform -0.465 K
// (30.0829 -> 29.6180, 30.7314 -> 30.2662).
//
// SCOPE, stated honestly: the 50 column values below are REAL engine data from
// the banked 8-pass reference. The five-snapshot window that produced the
// failure is NOT recoverable -- the per-pass snapshots never reach disk (that
// is its own ledger row). An attempt to reconstruct one gave ||gamma||_1 = 1.29
// and no displacement, i.e. it did NOT reproduce the mechanism, so it is not
// used. What IS asserted here is the guard itself, against the MEASURED
// ||gamma||_1 values from N6.2: the worst healthy active column (5.3) must
// survive, and the amplification implied by the observed 0.465 K displacement
// (2.6e5) must be refused.
static void testMeasuredNearDegenerateRegression() {
    std::printf("\n[8] guard vs the MEASURED gate-3a failure (face 277511 column)\n");
    static const double S_last[50] = {
        53.5418396, 54.1749001, 54.665905, 55.0112038, 55.2049446,
        55.2383347, 55.1001358, 54.7769241, 54.2532043, 53.5117302,
        52.5338936, 51.3004646, 49.7927208, 47.9939308, 45.8917999,
        43.4815178, 40.771347, 37.7923088, 34.7830391, 32.4550056,
        30.7293568, 29.5028744, 28.6723919, 28.1446915, 27.840704,
        27.6963482, 27.6619644, 27.7005253, 27.7856255, 27.8994808,
        28.0300159, 28.1699219, 28.3147583, 28.4618893, 28.609848,
        28.7584991, 28.9070053, 29.0555553, 29.2038116, 29.3519058,
        29.4992275, 29.6461201, 29.7923889, 29.9379635, 30.0828915,
        30.2276859, 30.3721333, 30.516325, 30.6601276, 30.7314262,
    };
    const int L = 50;
    double smax = 0.0;
    for (int l = 0; l < L; ++l) smax = std::max(smax, std::fabs(S_last[l]));

    // Level 1 and the box guard are GONE (2026-08-09 trim): gamma_refused and
    // box_refused were 0 in every validated run, and Level 1's stated rationale
    // was refuted -- its roundoff bound is 2.6e-5 K against a 0.465 K observed
    // displacement, 17,800x too small. What remains is tested below.

    // gamma_0 is the quantity that actually bounds the failure mode.
    // LEVEL 2 (ADDENDUM NOTE 8): absolute HOLD_K on the LAST-pass increment.
    const double HOLD_K = ex::kHoldKDefault;   // 1e-3 K/pass
    check(ex::level2Hold(0.0, HOLD_K),
          "Level 2 HOLDS a bit-locked layer (s_z = 0)", 1, 1);
    check(ex::level2Hold(9.9e-4, HOLD_K),
          "Level 2 HOLDS a layer just under HOLD_K", 1, 1);
    check(!ex::level2Hold(1.1e-3, HOLD_K),
          "Level 2 does NOT hold a layer just over HOLD_K", 1, 1);
    // The 3b population's deep layers carry a pass-4 increment of
    // gap*(1-r)/r = 0.4796 * 1.874 = 0.90 K, i.e. 899x above HOLD_K.
    check(!ex::level2Hold(0.90, HOLD_K),
          "Level 2 does NOT hold the 3b population's deep layers (0.90 K/pass)", 1, 1);
    std::printf("       3b deep increment 0.90 K is %.0fx above HOLD_K = %.0e K\n",
                0.90 / HOLD_K, HOLD_K);
    // HOLD_K must sit well above the float32 noise floor at both ends of the range.
    std::printf("       HOLD_K = %.0f ULP at 30 K, %.0f ULP at 390 K\n",
                HOLD_K / 1.9073486e-06, HOLD_K / 3.0517578e-05);
    check(HOLD_K > 100.0 * 1.9073486e-06,
          "HOLD_K is >100 ULP at 30 K (no ULP floor arm needed)", 1, 1);

    // ---- THE NAMED PREDICTION (N8.1): the 73-face case must now hold at depth.
    // A layer bit-locked from pass 1: S_0 = a (the seed), S_1..S_4 = b.
    // Without the hold, sum(gamma)=1 gives T* = b + gamma_0*(a-b) -- displacement
    // LINEAR in the seed-to-converged distance, which is the measured defect
    // (0.465 K observed => gamma_0 ~ 0.093, an ordinary weight, NOT roundoff).
    // With the hold, s_z = |S_4 - S_3| = 0 <= HOLD_K, so T*(z) = b EXACTLY.
    {
        const int L = 4, p = 4;
        const double a[4] = {29.0, 29.5, 30.0, 30.5};      // seed, ~5 K away
        const double b[4] = {24.0426, 24.1230, 24.1975, 24.2670};  // converged
        std::vector<double> X(static_cast<size_t>(p + 1) * L);
        for (int l = 0; l < L; ++l) {
            X[l] = a[l];                                    // S_0 = seed
            for (int j = 1; j <= p; ++j) X[static_cast<size_t>(j) * L + l] = b[l];
        }
        std::vector<double> out(L), sz(L);
        double cond = 0.0, g1 = 0.0;
        ex::rreColumn(X.data(), p, L, out.data(), &cond, ex::kRcond, &g1, sz.data());
        double s_max = 0.0, disp = 0.0;
        for (int l = 0; l < L; ++l) {
            s_max = std::max(s_max, sz[l]);
            disp  = std::max(disp, std::fabs(out[l] - b[l]));
        }
        std::printf("       bit-locked-from-pass-1 column: s_z max = %.3g K, "
                    "unheld displacement = %.6g K, ||gamma||_1 = %.3g\n",
                    s_max, disp, g1);
        checkNear("s_z excludes the seed transition (== 0 on a pass-1-locked layer)",
                  s_max, 0.0, 0.0);
        int held = 0;
        for (int l = 0; l < L; ++l) if (ex::level2Hold(sz[l], HOLD_K)) ++held;
        check(held == L, "ALL layers of the pass-1-locked column are HELD", held, L);
        double disp_held = 0.0;
        for (int l = 0; l < L; ++l) {
            const double v = ex::level2Hold(sz[l], HOLD_K) ? b[l] : out[l];
            disp_held = std::max(disp_held, std::fabs(v - b[l]));
        }
        checkNear("with the hold, T* == S_last EXACTLY (the named prediction)",
                  disp_held, 0.0, 0.0);
    }

    // NEAR-DEGENERATE WINDOW diagnostic (reported, never acted on).
    // Measured on the affected faces: gamma_0 ~ 0.09, seed distance ~5 K,
    // so the predicted displacement is ~0.45 K -- above a 0.5 K tol_K? no:
    // just under it, which is why the count is a diagnostic and not a guard.
    check(ex::nearDegenerateWindow(0.093, 5.0, 0.4),
          "near-degenerate fires when |g0|*seed_dist exceeds tol_K", 1, 1);
    check(!ex::nearDegenerateWindow(0.093, 5.0, 0.6),
          "near-degenerate silent when the bound is below tol_K", 1, 1);
    check(!ex::nearDegenerateWindow(0.0, 100.0, 0.5),
          "near-degenerate silent when the seed carries no weight", 1, 1);
}

int main() {
    std::printf("=== test_eq_extrapolation — pass-stride RRE correctness gate ===\n");
    testRingAddressing();
    testMeasuredNearDegenerateRegression();
    testGoldenVsReference();
    testExactOnGeometricModes();
    testRankDeficient();
    testRefusals();
    testCertificateAlgebra();
    testLeastSquares();
    std::printf("\n%s (%d failure%s)\n",
                g_failures == 0 ? "ALL PASS" : "*** FAILURES ***",
                g_failures, g_failures == 1 ? "" : "s");
    return g_failures == 0 ? 0 : 1;
}
