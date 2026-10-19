// test_lateral_conduction.cpp -- standalone unit gates for the C1c lateral operator
// (lateral_adjacency + lateral_conduction). Tiny meshes only; single-threaded.
//
//   A1  adjacency counts: closed tetrahedron, open strip, non-manifold fan, excluded neighbor
//   A2  conductance symmetry: G[s] == G[s'] bit-for-bit on a skewed random mesh
//   G13 energy closure: |sum C dT| <= 1e-12 * sum |C dT| per step, converged AND capped solves;
//       zero flux through the open (adiabatic) boundary => total sum C T unchanged
//   MP  discrete maximum principle (BE, M-matrix): no new extrema
//   EIG solver exactness: an applied step satisfies (C + dt L) T_new = C T_old, checked by an
//       independent residual evaluation (<= 1e-10); the mode's discrete eigenvalue is reported
//   G10s spatial accuracy: free-decay rate of cos(pi y/H) vs kappa q^2 at 3 refinements, dt small
//   G10t time accuracy at the production step dt = 7087 s and dt/4 (reported; see report text)
//   F3  activation rule: symmetric; a small cold facet next to a large warm one is ACTIVE
//
// Exit 0 only if every gate with a stated bar passes. G10t is REPORTED against the 1% bar
// (pass/fail printed) because Fermi's G10 includes it.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "mesh/lateral_adjacency.h"
#include "thermal/lateral_conduction.h"

using namespace ptm::lateral;

static int g_fail = 0;
static double g_last_be_residual = 0.0;
static Scheme g_scheme = Scheme::BackwardEuler;
#define CHECK(cond, ...) do { if (!(cond)) { std::printf("  FAIL: "); std::printf(__VA_ARGS__); std::printf("\n"); ++g_fail; } } while (0)

struct Mesh { std::vector<double> v; std::vector<int64_t> f; };

// Equilateral triangular lattice: nrows rows of height a*sqrt(3)/2, ncols lattice points per
// row; straight top/bottom boundaries (y = 0, y = H), zig-zag left/right.
static Mesh lattice(int nrows, int ncols, double a, double jitter = 0.0, unsigned seed = 1) {
    Mesh m;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    const double hy = a * std::sqrt(3.0) / 2.0;
    for (int r = 0; r <= nrows; ++r)
        for (int c = 0; c <= ncols; ++c) {
            double x = c * a + ((r % 2) ? 0.5 * a : 0.0), y = r * hy;
            const bool edge_row = (r == 0 || r == nrows);
            if (jitter > 0.0) {  // keep the straight top/bottom boundaries straight
                x += jitter * a * U(rng);
                if (!edge_row) y += jitter * hy * U(rng);
            }
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

// Link the left (c = 0) and right (c = ncols) zig-zag boundaries: the lattice is periodic in
// x with period ncols*a, so only the straight top/bottom boundaries remain (adiabatic). d_half
// stays each face's own centroid-to-midpoint distance, which is correct on both sides.
static void makePeriodicX(const Mesh& m, int nrows, int ncols, Adjacency& A) {
    auto rc = [&](int64_t v, int& r, int& c) { r = static_cast<int>(v / (ncols + 1)); c = static_cast<int>(v % (ncols + 1)); };
    std::vector<std::pair<int64_t, int>> left(nrows, {-1, -1}), right(nrows, {-1, -1});
    const int64_t nf = static_cast<int64_t>(m.f.size() / 3);
    for (int64_t f = 0; f < nf; ++f)
        for (int e = 0; e < 3; ++e) {
            if (A.nbr[3 * f + e] >= 0) continue;
            int r0, c0, r1, c1;
            rc(m.f[3 * f + e], r0, c0); rc(m.f[3 * f + (e + 1) % 3], r1, c1);
            if (r0 == r1) continue;                       // top/bottom boundary edge
            const int r = std::min(r0, r1);
            if (c0 == 0 && c1 == 0) left[r] = {f, e};
            if (c0 == ncols && c1 == ncols) right[r] = {f, e};
        }
    for (int r = 0; r < nrows; ++r) {
        const auto L = left[r], R = right[r];
        if (L.first < 0 || R.first < 0) { std::printf("  makePeriodicX: row %d unmatched\n", r); ++g_fail; continue; }
        A.nbr[3 * L.first + L.second] = static_cast<int32_t>(R.first);
        A.nbr[3 * R.first + R.second] = static_cast<int32_t>(L.first);
        A.nbr_slot[3 * L.first + L.second] = static_cast<int8_t>(R.second);
        A.nbr_slot[3 * R.first + R.second] = static_cast<int8_t>(L.second);
        A.n_boundary_edges -= 2; A.n_interior_edges += 1;
    }
}

// Independent check that a step solved BE: || (C + dt L) T_new - C T_old || / || C T_old ||.
static double beResidual(const Adjacency& A, const std::vector<double>& G, const std::vector<double>& C,
                         const std::vector<double>& Told, const std::vector<double>& Tnew, double dt) {
    double rn = 0.0, bn = 0.0;
    for (size_t i = 0; i < Told.size(); ++i) {
        double ax = C[i] * Tnew[i];
        for (int e = 0; e < 3; ++e) {
            const int32_t j = A.nbr[3 * i + e];
            if (j >= 0 && G[3 * i + e] != 0.0) ax += dt * G[3 * i + e] * (Tnew[i] - Tnew[j]);
        }
        const double r = ax - C[i] * Told[i];
        rn += r * r; bn += C[i] * Told[i] * C[i] * Told[i];
    }
    return std::sqrt(rn / bn);
}

static std::vector<double> centroidY(const Mesh& m) {
    const size_t nf = m.f.size() / 3;
    std::vector<double> y(nf);
    for (size_t i = 0; i < nf; ++i)
        y[i] = (m.v[3 * m.f[3 * i] + 1] + m.v[3 * m.f[3 * i + 1] + 1] + m.v[3 * m.f[3 * i + 2] + 1]) / 3.0;
    return y;
}

static void testAdjacency() {
    std::printf("A1 adjacency counts\n");
    {   // closed tetrahedron: 4 faces, 6 interior edges, no boundary
        Mesh t{{0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1}, {0, 2, 1, 0, 1, 3, 1, 2, 3, 0, 3, 2}};
        Adjacency A = build(t.v, t.f, {});
        std::printf("  tetra: %s\n", A.summary().c_str());
        CHECK(A.n_interior_edges == 6 && A.n_boundary_edges == 0, "tetrahedron counts");
    }
    {   // 2x3 lattice strip: E_boundary = 2*ncols + 2*nrows for this lattice
        Mesh s = lattice(2, 3, 1.0);
        Adjacency A = build(s.v, s.f, {});
        std::printf("  strip: %s\n", A.summary().c_str());
        const int64_t nf = static_cast<int64_t>(s.f.size() / 3);
        CHECK(2 * A.n_interior_edges + A.n_boundary_edges == 3 * nf, "edge-slot bookkeeping");
        CHECK(A.n_boundary_edges == 2 * 3 + 2 * 2, "strip boundary edges = %lld", (long long)A.n_boundary_edges);
    }
    {   // non-manifold fan: three triangles on edge (0,1)
        Mesh n{{0, 0, 0, 1, 0, 0, 0.5, 1, 0, 0.5, -1, 0, 0.5, 0, 1}, {0, 1, 2, 1, 0, 3, 0, 1, 4}};
        Adjacency A = build(n.v, n.f, {});
        std::printf("  fan:   %s\n", A.summary().c_str());
        CHECK(A.n_nonmanifold_edges == 1 && A.n_interior_edges == 0, "non-manifold edge couples nobody");
    }
    {   // excluded neighbor
        Mesh s = lattice(1, 1, 1.0);   // 2 faces sharing one edge
        Adjacency A = build(s.v, s.f, {1, 0});
        CHECK(A.n_interior_edges == 0 && A.n_excluded_edges == 1, "excluded neighbor is adiabatic and counted");
        CHECK(A.nbr[0] == -1 && A.nbr[1] == -1 && A.nbr[2] == -1, "excluded neighbor not referenced");
    }
}

static void randomFields(size_t nf, unsigned seed, std::vector<double>& T, std::vector<double>& k,
                         std::vector<double>& rc) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> UT(40.0, 300.0), Uk(5e-4, 3e-3), Urc(2e5, 1.5e6);
    T.resize(nf); k.resize(nf); rc.resize(nf);
    for (size_t i = 0; i < nf; ++i) { T[i] = UT(rng); k[i] = Uk(rng); rc[i] = Urc(rng); }
}

static void testClosureAndMP() {
    std::printf("A2/G13/MP conservation, symmetry, maximum principle (skewed 24x24 lattice, jitter 0.3)\n");
    Mesh m = lattice(24, 24, 2.5e-3, 0.3, 7);
    Adjacency A = build(m.v, m.f, {});
    const size_t nf = m.f.size() / 3;
    std::vector<double> T, k, rc;
    randomFields(nf, 11, T, k, rc);
    std::vector<double> G;
    assembleConductance(A, nullptr, k.data(), 1.0e-3, G);
    int64_t asym = 0;
    for (size_t s = 0; s < 3 * nf; ++s) {
        const int32_t j = A.nbr[s];
        if (j >= 0 && G[s] != G[3 * static_cast<size_t>(j) + A.nbr_slot[s]]) ++asym;
    }
    CHECK(asym == 0, "%lld asymmetric conductance slots", (long long)asym);
    for (int run = 0; run < 3; ++run) {
        const int capped = (run == 1);
        std::vector<double> Tw = T;
        SolverOptions opt;
        if (capped) opt.max_iter = 2;
        if (run == 2) opt.scheme = Scheme::TRBDF2;
        Workspace ws;
        double E0 = 0.0, E1 = 0.0, worst = 0.0;
        for (size_t i = 0; i < nf; ++i) E0 += rc[i] * A.area[i] * 1.0e-3 * Tw[i];
        const double Tmin0 = *std::min_element(Tw.begin(), Tw.end()), Tmax0 = *std::max_element(Tw.begin(), Tw.end());
        double Tmin1 = 1e300, Tmax1 = -1e300;
        bool saw_cap = false;
        for (int step = 0; step < 20; ++step) {
            StepStats st = backwardEulerStep(A, nullptr, Tw.data(), k.data(), rc.data(), 1.0e-3, 7087.0, opt, ws);
            saw_cap = saw_cap || st.capped;
            if (st.sum_abs_Q > 0) worst = std::max(worst, std::fabs(st.sum_Q) / st.sum_abs_Q);
        }
        for (size_t i = 0; i < nf; ++i) {
            E1 += rc[i] * A.area[i] * 1.0e-3 * Tw[i];
            Tmin1 = std::min(Tmin1, Tw[i]); Tmax1 = std::max(Tmax1, Tw[i]);
        }
        std::printf("  %s: max |sum Q|/sum|Q| = %.2e, total energy rel change = %.2e, T range [%.3f, %.3f] -> [%.3f, %.3f]%s\n",
                    capped ? "BE capped (2 its)" : (run == 2 ? "TRBDF2 converged " : "BE converged     "), worst, std::fabs(E1 - E0) / E0, Tmin0, Tmax0, Tmin1, Tmax1,
                    saw_cap ? ", cap reported" : "");
        CHECK(worst <= 1e-12, "energy closure %.2e > 1e-12", worst);
        CHECK(std::fabs(E1 - E0) / E0 <= 1e-12, "total energy drift");
        if (capped) {
            CHECK(saw_cap, "capped solve not reported");
            CHECK(Tw == T, "a capped (unconverged) solve was applied to T");
        }
        else if (run == 0) CHECK(Tmin1 >= Tmin0 - 1e-6 && Tmax1 <= Tmax0 + 1e-6, "maximum principle violated (BE)");
    }
}

// Projection of a field onto cos(q y) (area-weighted, mean removed).
static double modeAmp(const std::vector<double>& T, const std::vector<double>& y, const std::vector<double>& area, double q) {
    double num = 0.0, den = 0.0, mean = 0.0, at = 0.0;
    for (size_t i = 0; i < T.size(); ++i) { mean += area[i] * T[i]; at += area[i]; }
    mean /= at;
    for (size_t i = 0; i < T.size(); ++i) {
        const double c = std::cos(q * y[i]);
        num += area[i] * (T[i] - mean) * c;
        den += area[i] * c * c;
    }
    return num / den;
}

// Free decay of cos(pi y / H). Returns the measured rate over nsteps of dt.
static double decayRate(int nrows, double H, double kval, double rcval, double dt, int nsteps, double* lam_h_out,
                        double* amp_ratio_step) {
    const double a = H / (nrows * std::sqrt(3.0) / 2.0);
    const int ncols = 4;
    Mesh m = lattice(nrows, ncols, a);
    Adjacency A = build(m.v, m.f, {});
    makePeriodicX(m, nrows, ncols, A);
    const size_t nf = m.f.size() / 3;
    std::vector<double> y = centroidY(m), T(nf), k(nf, kval), rc(nf, rcval);
    const double q = M_PI / H;
    for (size_t i = 0; i < nf; ++i) T[i] = 100.0 + 1.0 * std::cos(q * y[i]);
    // Rayleigh quotient of the discrete mode: lambda_h = (x^T L x)/(x^T C x), using the actual G.
    std::vector<double> G;
    const double hk = 1.0;
    assembleConductance(A, nullptr, k.data(), hk, G);
    double xLx = 0.0, xCx = 0.0;
    for (size_t i = 0; i < nf; ++i) {
        const double xi = std::cos(q * y[i]);
        xCx += rcval * A.area[i] * hk * xi * xi;
        for (int e = 0; e < 3; ++e) {
            const int32_t j = A.nbr[3 * i + e];
            if (j >= 0) xLx += 0.5 * G[3 * i + e] * (xi - std::cos(q * y[j])) * (xi - std::cos(q * y[j]));
        }
    }
    *lam_h_out = xLx / xCx;
    SolverOptions opt;
    opt.scheme = g_scheme;
    opt.rel_tol = 1e-13;
    opt.max_iter = 5000;
    Workspace ws;
    const double A0 = modeAmp(T, y, A.area, q);
    double A1 = A0;
    for (int s = 0; s < nsteps; ++s) {
        const std::vector<double> Told = T;
        backwardEulerStep(A, nullptr, T.data(), k.data(), rc.data(), hk, dt, opt, ws);
        if (s == 0 && g_scheme != Scheme::BackwardEuler) A1 = modeAmp(T, y, A.area, q);
        if (s == 0 && g_scheme == Scheme::BackwardEuler) {
            A1 = modeAmp(T, y, A.area, q);
            std::vector<double> C(nf);
            for (size_t i = 0; i < nf; ++i) C[i] = rcval * A.area[i] * hk;
            g_last_be_residual = beResidual(A, G, C, Told, T, dt);
        }
    }
    const double An = modeAmp(T, y, A.area, q);
    *amp_ratio_step = A1 / A0;
    return -std::log(An / A0) / (nsteps * dt);
}

static void testG10() {
    // Cold-regolith-like properties; H = 1 cm half-wavelength (a cm-scale cold trap).
    const double kval = 1.0e-3, rcval = 1100.0 * 600.0, H = 0.01;
    const double kappa = kval / rcval, q = M_PI / H, lam = kappa * q * q;
    std::printf("EIG/G10 free decay of cos(pi y/H), H = 1 cm, kappa = %.3e m^2/s, analytic rate kappa q^2 = %.4e /s\n",
                kappa, lam);
    // Spatial: dt small (lambda dt = 1e-3), 3 refinements (mean edge 2.5 / 1.25 / 0.625 mm)
    for (int nrows : {4, 8, 16}) {
        double lam_h, r1;
        const double dt = 1e-3 / lam;
        const double rate = decayRate(nrows, H, kval, rcval, dt, 400, &lam_h, &r1);
        const double be_pred = 1.0 / (1.0 + lam_h * dt);
        const double rel = rate / lam - 1.0;
        std::printf("  G10s rows=%2d edge=%.3f mm: rate/analytic - 1 = %+.4f%%, discrete eigen/analytic - 1 = %+.4f%%, "
                    "step amp %.10f vs 1/(1+lambda_h dt) %.10f; BE residual of step 1 = %.1e\n",
                    nrows, 1e3 * H / (nrows * std::sqrt(3.0) / 2.0), 100 * rel, 100 * (lam_h / lam - 1.0), r1, be_pred,
                    g_last_be_residual);
        CHECK(g_last_be_residual <= 1e-10, "EIG: step does not satisfy the BE equations (%.1e)", g_last_be_residual);
        if (nrows == 16) CHECK(std::fabs(rel) <= 0.01, "G10s finest refinement %.3f%% > 1%%", 100 * rel);
    }
    // Time: production step and dt/4 on the finest mesh, both schemes. The reference is the
    // mesh's own semi-discrete rate lambda_h (so this isolates the TIME error).
    for (Scheme sc : {Scheme::BackwardEuler, Scheme::TRBDF2}) {
        g_scheme = sc;
        for (double dt : {7087.0, 7087.0 / 4.0}) {
            double lam_h, r1;
            const int nsteps = std::max(1, static_cast<int>(std::lround(4.0 * 7087.0 / dt)));
            const double rate = decayRate(16, H, kval, rcval, dt, nsteps, &lam_h, &r1);
            const double rel = rate / lam_h - 1.0;
            std::printf("  G10t %-6s dt=%7.1f s (lambda dt = %.3f): rate/semi-discrete - 1 = %+.2f%%  [bar 1%%: %s]\n",
                        sc == Scheme::BackwardEuler ? "BE" : "TRBDF2", dt, lam_h * dt, 100 * rel,
                        std::fabs(rel) <= 0.01 ? "pass" : "FAIL");
        }
        // Where does each scheme reach 1%? Scan lambda dt.
        double ld_ok = 0.0;
        for (double ld = 1e-3; ld <= 4.0; ld *= 1.25) {
            double lam_h, r1;
            const double dt = ld / lam;
            const double rate = decayRate(16, H, kval, rcval, dt, 4, &lam_h, &r1);
            if (std::fabs(rate / lam_h - 1.0) <= 0.01) ld_ok = lam_h * dt; else break;
        }
        std::printf("  G10t %-6s: free-decay rate within 1%% up to lambda dt = %.3f\n",
                    sc == Scheme::BackwardEuler ? "BE" : "TRBDF2", ld_ok);
    }
    g_scheme = Scheme::BackwardEuler;
}

static void testActivation() {
    std::printf("F3 activation: min(l_i,l_j) < c*max(D*_i,D*_j), c = 2\n");
    Mesh m = lattice(6, 6, 2.5e-3, 0.2, 3);
    Adjacency A = build(m.v, m.f, {});
    const size_t nf = m.f.size() / 3;
    std::vector<double> D(nf, 0.5e-3);   // warm: D* = 0.5 mm < l/2 -> inactive
    D[nf / 2] = 6.5e-3;                  // one cold facet (110 K): its edges become active
    int64_t n_act = 0;
    std::vector<uint8_t> act = edgeActivation(A, D, 2.0, &n_act);
    int64_t asym = 0, on_cold = 0;
    for (size_t s = 0; s < 3 * nf; ++s) {
        const int32_t j = A.nbr[s];
        if (j < 0) continue;
        if (act[s] != act[3 * static_cast<size_t>(j) + A.nbr_slot[s]]) ++asym;
        if ((s / 3 == nf / 2) && act[s]) ++on_cold;
    }
    int64_t cold_coupled = 0;
    for (int e = 0; e < 3; ++e) cold_coupled += (A.nbr[3 * (nf / 2) + e] >= 0);
    std::printf("  %lld active pairs of %lld; cold facet active edges %lld of %lld coupled\n", (long long)n_act,
                (long long)A.n_interior_edges, (long long)on_cold, (long long)cold_coupled);
    CHECK(asym == 0, "activation not symmetric");
    CHECK(on_cold == cold_coupled && n_act == cold_coupled, "cold facet's edges (and only those) should be active");
    // D* sanity vs Fermi's table: 110 K regolith surface properties -> ~6.5 mm
    const double T = 110.0, ks = 7.4e-4 * (1 + 2.7 * std::pow(T / 350.0, 3));
    const double cp = -3.6125 + 2.7431 * T + 2.3616e-3 * T * T - 1.2340e-5 * T * T * T + 8.9093e-9 * T * T * T * T;
    const double Ds = conductionLength(ks, 1100.0, cp, 0.95, T, 2 * M_PI / (29.530588 * 86400.0));
    std::printf("  D*(110 K) = %.2f mm (expected ~6.5 mm)\n", 1e3 * Ds);
    CHECK(std::fabs(1e3 * Ds - 6.5) < 0.2, "D*(110 K) off");
}

int main() {
    testAdjacency();
    testClosureAndMP();
    testG10();
    testActivation();
    std::printf(g_fail ? "\nRESULT: FAIL (%d)\n" : "\nRESULT: PASS (all stated bars)\n", g_fail);
    return g_fail ? 1 : 0;
}
