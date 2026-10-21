// test_lateral_network.cpp -- gates for ptm::lateral::Network, the fused all-layer lateral
// stages the engine runs (C1c perf fusion, 2026-10-21). Links libthermal, i.e. the SAME
// OpenMP-compiled object code as ptm_thermal_sim.
//
//   N1  thread invariance: Network at 1 thread and at 8 threads give BIT-IDENTICAL T and
//       increments (fixed 8192-face chunks; the per-block code does not depend on the thread).
//   N2  equivalence to the single-layer path: per layer, Network == backwardEulerStep(BE) and
//       explicitIncrement to round-off. Not bit-for-bit by construction: libthermal is built
//       with -ffast-math, so the compiler may reassociate the vectorized reductions of the two
//       code shapes differently. Bar: max |dT| <= 1e-9 K after 20 coupled steps (reported).
//   N3  the PCG iteration counts agree with the single-layer path (same iterations per layer).
//   N4  a capped solve (rel_tol < 0, unreachable) is reported and applies NOTHING to any layer.
//   N5  closure: per step |sum C dT| / sum |C dT| <= 1e-12 on every layer.
//   N6  implicitCoupled on a zero-area face with every edge inactive: no cap, == the exact column solve.
// Mesh: skewed lattice with > 1 chunk per layer (so chunk boundaries are exercised).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "mesh/lateral_adjacency.h"
#include "thermal/lateral_conduction.h"

using namespace ptm::lateral;

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { std::printf("  FAIL: "); std::printf(__VA_ARGS__); std::printf("\n"); ++g_fail; } } while (0)

struct Mesh { std::vector<double> v; std::vector<int64_t> f; };

static Mesh lattice(int nrows, int ncols, double a, double jitter, unsigned seed) {
    Mesh m;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    const double hy = a * std::sqrt(3.0) / 2.0;
    for (int r = 0; r <= nrows; ++r)
        for (int c = 0; c <= ncols; ++c) {
            double x = c * a + ((r % 2) ? 0.5 * a : 0.0), y = r * hy;
            if (c != 0 && c != ncols && r != 0 && r != nrows) { x += jitter * a * U(rng); y += jitter * hy * U(rng); }
            m.v.insert(m.v.end(), {x, y, 0.0});
        }
    auto id = [&](int r, int c) { return static_cast<int64_t>(r) * (ncols + 1) + c; };
    for (int r = 0; r < nrows; ++r)
        for (int c = 0; c < ncols; ++c) {
            if (r % 2 == 0) {
                m.f.insert(m.f.end(), {id(r, c), id(r, c + 1), id(r + 1, c)});
                m.f.insert(m.f.end(), {id(r, c + 1), id(r + 1, c + 1), id(r + 1, c)});
            } else {
                m.f.insert(m.f.end(), {id(r, c), id(r + 1, c + 1), id(r + 1, c)});
                m.f.insert(m.f.end(), {id(r, c), id(r, c + 1), id(r + 1, c + 1)});
            }
        }
    return m;
}

int main() {
    // 120 x 120 points -> 28,800 faces = 4 chunks per layer; 2.5 mm facets (Moon3D scale).
    Mesh m = lattice(120, 120, 2.5e-3, 0.3, 3);
    Adjacency A = build(m.v, m.f, {});
    const int64_t nf = A.n_faces;
    const std::vector<double> h = {1.6e-3, 3.4e-3, 4.1e-3, 9.0e-3, 2.0e-2, 6.0e-2};
    const int L = static_cast<int>(h.size());
    std::printf("Network gates: %lld faces, %d layers, %lld chunks/layer\n", (long long)nf, L,
                (long long)((nf + 8191) / 8192));
    std::mt19937 rng(17);
    std::uniform_real_distribution<double> UT(40.0, 300.0), Uk(5e-4, 3e-3), Urc(2e5, 1.5e6);
    std::vector<double> T0(L * nf), K(L * nf), RC(L * nf);
    for (int64_t i = 0; i < L * nf; ++i) { T0[i] = UT(rng); K[i] = Uk(rng); RC[i] = Urc(rng); }
    // 30 % of the edges inactive (exercises the activation mask)
    std::vector<uint8_t> act(3 * nf, 1);
    {
        std::mt19937 r2(5);
        std::uniform_real_distribution<double> U(0.0, 1.0);
        for (int64_t s = 0; s < 3 * nf; ++s) {
            const int32_t j = A.nbr[s];
            if (j < 0 || j < s / 3) continue;
            const uint8_t a = U(r2) < 0.7;
            act[s] = a;
            act[3 * static_cast<int64_t>(j) + A.nbr_slot[s]] = a;
        }
    }
    const double tau = 3322.0;   // a production CN substep (Moon day / 768)
    const int nsteps = 20;

    // Reference: single-layer functions, layer by layer (the pre-fusion engine sequence).
    std::vector<double> Tref = T0, incref(L * nf);
    std::vector<int> it_ref;
    Workspace ws;
    SolverOptions opt;
    opt.scheme = Scheme::BackwardEuler;
    opt.rel_tol = 1e-8;
    auto t0 = std::chrono::steady_clock::now();
    for (int step = 0; step < nsteps; ++step) {
        for (int l = 0; l < L; ++l) {
            std::vector<double> G;
            assembleConductance(A, act.data(), &K[l * nf], h[l], G);
            double sq, sa;
            explicitIncrement(A, G, &Tref[l * nf], &RC[l * nf], h[l], tau, &incref[l * nf], &sq, &sa);
            std::vector<double> C(nf);
            for (int64_t i = 0; i < nf; ++i) C[i] = RC[l * nf + i] * A.area[i] * h[l];
            opt.max_iter = iterationCap(maxFourier(A, G, C, tau));
            StepStats st = backwardEulerStep(A, act.data(), &Tref[l * nf], &K[l * nf], &RC[l * nf], h[l], tau, opt, ws);
            it_ref.push_back(st.iterations);
            if (st.capped) { std::printf("  reference capped?!\n"); return 2; }
        }
    }
    const double t_ref = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    auto runNet = [&](int nt, std::vector<double>& T, std::vector<double>& inc, std::vector<int>& its,
                      double* closure, double* wall) {
        Network net;
        net.init(&A, act.data(), h, nt);
        T = T0;
        inc.assign(L * nf, 0.0);
        *closure = 0.0;
        auto a = std::chrono::steady_clock::now();
        for (int step = 0; step < nsteps; ++step) {
            net.assemble(K.data(), RC.data());
            net.explicitIncrement(T.data(), tau, inc.data());
            Network::Stats st = net.implicitBE(T.data(), tau, 1e-8);
            if (st.capped) { std::printf("  network capped?!\n"); std::exit(2); }
            its.push_back(st.iterations);
            *closure = std::max(*closure, st.closure_worst);
        }
        *wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count();
    };
    std::vector<double> T1, i1, T8, i8;
    std::vector<int> its1, its8;
    double c1, c8, w1, w8;
    runNet(1, T1, i1, its1, &c1, &w1);
    runNet(8, T8, i8, its8, &c8, &w8);

    // N1
    int64_t ndiff = 0;
    for (int64_t i = 0; i < L * nf; ++i) ndiff += (T1[i] != T8[i]) || (i1[i] != i8[i]);
    std::printf("N1 thread invariance (1 vs 8 threads): %lld differing values of %lld\n", (long long)ndiff, (long long)(2 * L * nf));
    CHECK(ndiff == 0, "Network results depend on the thread count");
    // N2
    double dT = 0.0, dI = 0.0;
    for (int64_t i = 0; i < L * nf; ++i) { dT = std::max(dT, std::fabs(T1[i] - Tref[i])); dI = std::max(dI, std::fabs(i1[i] - incref[i])); }
    int64_t nbit = 0;
    for (int64_t i = 0; i < L * nf; ++i) nbit += (T1[i] != Tref[i]);
    std::printf("N2 Network vs single-layer path after %d steps: max |dT| = %.3e K, max |d inc| (last step) = %.3e K; "
                "%lld of %lld values not bit-identical\n", nsteps, dT, dI, (long long)nbit, (long long)(L * nf));
    CHECK(dT <= 1e-9 && dI <= 1e-9, "Network differs from the single-layer path beyond round-off");
    // N3: per-step totals over layers
    int64_t itdiff = 0;
    for (int s = 0; s < nsteps; ++s) {
        int r = 0;
        for (int l = 0; l < L; ++l) r += it_ref[s * L + l];
        itdiff += (r != its1[s]);
    }
    std::printf("N3 PCG iterations per step (all layers): reference %d, network %d (step 0); %lld steps differ\n",
                [&] { int r = 0; for (int l = 0; l < L; ++l) r += it_ref[l]; return r; }(), its1[0], (long long)itdiff);
    CHECK(itdiff == 0, "iteration counts differ from the single-layer path");
    // N5
    std::printf("N5 closure worst |sum Q|/sum|Q| = %.2e (1 thread), %.2e (8 threads)\n", c1, c8);
    CHECK(c1 <= 1e-12 && c8 <= 1e-12, "closure");
    // N4
    {
        Network net;
        net.init(&A, act.data(), h, 4);
        std::vector<double> T = T0;
        net.assemble(K.data(), RC.data());
        // rel_tol < 0 can never be met (a 1e-300 bar CAN: at Fo ~ 0.5 CG underflows to an exact-zero
        // residual), so every layer runs to its cap.
        Network::Stats st = net.implicitBE(T.data(), tau, -1.0);
        std::printf("N4 unreachable tolerance: capped=%d layer=%d its=%d (cap %d) residual=%.2e worst_face=%lld; T untouched: %s\n",
                    st.capped, st.capped_layer, st.capped_iterations, st.cap, st.capped_rel_residual,
                    (long long)st.worst_face, T == T0 ? "yes" : "NO");
        CHECK(st.capped && T == T0, "a capped Network solve was applied (or not reported)");
    }
    // N6 implicitCoupled with a ZERO-AREA face and every edge inactive (the zero-coupling companion on a mesh
    // with a degenerate triangle): must not cap, and must equal the exact per-column tridiagonal solve.
    {
        Mesh m2 = lattice(6, 6, 2.5e-3, 0.0, 1);
        const int64_t nv0 = static_cast<int64_t>(m2.v.size() / 3);
        m2.v.insert(m2.v.end(), {0.0, -1e-3, 0.0, 1e-3, -1e-3, 0.0, 2e-3, -1e-3, 0.0});   // collinear: area 0
        m2.f.insert(m2.f.end(), {nv0, nv0 + 1, nv0 + 2});
        Adjacency A2 = build(m2.v, m2.f, {});
        const int64_t nf2 = A2.n_faces;
        const std::vector<double> h2 = {1.6e-3, 3.4e-3, 9.0e-3};
        const int L2 = static_cast<int>(h2.size());
        std::vector<uint8_t> none(3 * nf2, 0);
        std::vector<double> K2(L2 * nf2, 1e-3), RC2(L2 * nf2, 8e5), a(L2 * nf2), b(L2 * nf2), c(L2 * nf2), d(L2 * nf2);
        for (int l = 0; l < L2; ++l)
            for (int64_t i = 0; i < nf2; ++i) {
                const int64_t k = l * nf2 + i;
                a[k] = (l > 0) ? -0.3 : 0.0; c[k] = (l < L2 - 1) ? -0.25 : 0.0; b[k] = 1.6; d[k] = 100.0 + 0.1 * i + l;
            }
        // (a) every edge inactive (the companion: CG never iterates, x0 is exact); (b) every coupled edge ACTIVE,
        // so CG iterates and the zero-area face's row reaches the preconditioner (the failure mode).
        Network net2;
        net2.init(&A2, none.data(), h2, 2);
        net2.assemble(K2.data(), RC2.data());
        std::vector<double> T2(L2 * nf2, 0.0);
        const Network::Stats st2 = net2.implicitCoupled(a.data(), b.data(), c.data(), d.data(), 3322.0, 1e-8, T2.data());
        Network net3;
        net3.init(&A2, nullptr, h2, 2);
        net3.assemble(K2.data(), RC2.data());
        std::vector<double> T3(L2 * nf2, 0.0);
        const Network::Stats st3 = net3.implicitCoupled(a.data(), b.data(), c.data(), d.data(), 3322.0, 1e-8, T3.data());
        bool finite3 = true;
        for (double v : T3) finite3 = finite3 && std::isfinite(v);
        double worst = 0.0;
        for (int64_t i = 0; i < nf2; ++i) {   // reference: Thomas per column
            std::vector<double> cp(L2), dp(L2), x(L2);
            double mm = b[i]; cp[0] = c[i] / mm; dp[0] = d[i] / mm;
            for (int l = 1; l < L2; ++l) { const int64_t k = l * nf2 + i; mm = b[k] - a[k] * cp[l - 1]; cp[l] = c[k] / mm; dp[l] = (d[k] - a[k] * dp[l - 1]) / mm; }
            x[L2 - 1] = dp[L2 - 1];
            for (int l = L2 - 2; l >= 0; --l) x[l] = dp[l] - cp[l] * x[l + 1];
            for (int l = 0; l < L2; ++l) worst = std::max(worst, std::fabs(T2[l * nf2 + i] - x[l]));
            if (i == nf2 - 1)   // the degenerate face is isolated: its column must be the column solve in (b) too
                for (int l = 0; l < L2; ++l) worst = std::max(worst, std::fabs(T3[l * nf2 + i] - x[l]));
        }
        std::printf("N6 zero-area face: degenerate faces %lld; (a) zero coupling capped %d; (b) all edges active capped %d, "
                    "its %d, all finite %d; max |T - column solve| (all faces in a, the degenerate face in b) %.2e K\n",
                    (long long)A2.n_degenerate_faces, st2.capped, st3.capped, st3.iterations, finite3, worst);
        CHECK(A2.n_degenerate_faces >= 1 && !st2.capped && !st3.capped && st3.iterations > 0 && finite3 && worst <= 1e-10,
              "implicitCoupled fails on a zero-area face");
    }
    std::printf("timing (%d steps): single-layer path %.3f s (1 thread), network %.3f s (1 thread), %.3f s (8 threads)\n",
                nsteps, t_ref, w1, w8);
    std::printf(g_fail ? "\nRESULT: FAIL (%d)\n" : "\nRESULT: PASS (all stated bars)\n", g_fail);
    return g_fail ? 1 : 0;
}
