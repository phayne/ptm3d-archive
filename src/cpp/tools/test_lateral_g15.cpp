// test_lateral_g15.cpp -- G15 transient stress, SYNTHETIC CHECKERBOARD part (Fermi addendum 2 item 2,
// LATERAL_SIGNOFF_FERMI.md). Runs the engine's own lateral calls (ptm::lateral::Network +
// ThermalBatchSoA::advanceTimestepWithSource + the BE correction, the time-outer driver's balanced split)
// on a triangular lattice of Moon columns.
//
// Set-up: uniform Moon columns at T0 with Qb = 0 and uniform forcing F = eps sigma T0^4 (an exact steady
// state), then a checkerboard contrast +-dT/2 on every capacitive node (up/down triangles: the lattice is
// bipartite, so this is the HIGHEST lateral mode, the stiffest case). Facet size a sets
// Fo = dt_sub kappa / a^2; a = 2.48 mm is the production 509k mesh (Fo multiplier x1); multipliers
// x3 and x10 per Fermi, plus G15_EXTRA_MULT (comma list) to locate where the split breaks.
// Bars per step (Fermi): every capacitive node within [min, max] of its previous-step stencil (column
// +-1, its lateral neighbours, the surface and bottom values) to 0.1 K and never below T_geo; the lateral
// anomaly norm (area-weighted L2 of T minus the area-weighted neighbour mean, per layer) non-increasing
// to 1 %; after 1 day and 1 lunation |T - T_ref| median <= 0.1 K and max <= 1 K against the same split
// at dt/64. Also reported: the drift of the area-weighted mean column temperature against the reference
// (an energy-creating split shows up there first; the checkerboard is mean-zero).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "mesh/lateral_adjacency.h"
#include "thermal/lateral_conduction.h"
#include "thermal/thermal_model.h"

namespace lat = ptm::lateral;
constexpr int BS = thermal::THERMAL_BATCH_SIZE;

struct Lattice { lat::Adjacency adj; std::vector<int> color; int nf; };

static Lattice makeLattice(int nr, int nc, double a) {
    std::vector<double> v;
    std::vector<int64_t> f;
    const double hy = a * std::sqrt(3.0) / 2.0;
    for (int r = 0; r <= nr; ++r)
        for (int c = 0; c <= nc; ++c) v.insert(v.end(), {c * a + ((r % 2) ? 0.5 * a : 0.0), r * hy, 0.0});
    auto id = [&](int r, int c) { return static_cast<int64_t>(r) * (nc + 1) + c; };
    Lattice L;
    for (int r = 0; r < nr; ++r)
        for (int c = 0; c < nc; ++c) {
            if (r % 2 == 0) {
                f.insert(f.end(), {id(r, c), id(r, c + 1), id(r + 1, c)}); L.color.push_back(0);
                f.insert(f.end(), {id(r, c + 1), id(r + 1, c + 1), id(r + 1, c)}); L.color.push_back(1);
            } else {
                f.insert(f.end(), {id(r, c), id(r + 1, c + 1), id(r + 1, c)}); L.color.push_back(1);
                f.insert(f.end(), {id(r, c), id(r, c + 1), id(r + 1, c + 1)}); L.color.push_back(0);
            }
        }
    L.adj = lat::build(v, f, {});
    L.nf = static_cast<int>(f.size() / 3);
    // verify the coloring is a proper 2-coloring (every coupled pair differs)
    for (int i = 0; i < L.nf; ++i)
        for (int e = 0; e < 3; ++e) {
            const int j = L.adj.nbr[3 * i + e];
            if (j >= 0 && L.color[i] == L.color[j]) { std::fprintf(stderr, "coloring not bipartite\n"); std::exit(2); }
        }
    return L;
}

struct Sim {
    Lattice L;
    std::vector<thermal::ThermalBatchSoA> batches;
    lat::Network net;
    std::vector<double> LT, LK, LRC, LSRC, hl;
    int nl = 0, n_int = 0, nb = 0;
    double Fbar = 0.0, T0 = 0.0;
    void gather() {
        const int nf = L.nf;
        for (auto& b : batches) thermal::updateMaterialPropertiesBatchSIMD(b);
        for (int l = 1; l <= nl - 2; ++l)
            for (int j = 0; j < nf; ++j) {
                const auto& b = batches[j / BS];
                const int f = j % BS;
                LT[(l - 1) * nf + j] = b.T[l * BS + f];
                LK[(l - 1) * nf + j] = b.k[l * BS + f];
                LRC[(l - 1) * nf + j] = b.rho[l * BS + f] * b.cp[l * BS + f];
            }
    }
    // the engine's balanced substep (time-outer driver, ptm_thermal_sim.cpp)
    bool lat_off = false;   // G15_LATOFF=1 control arm: the same columns with no lateral stage
    bool mono = false;      // G15_SCHEME=monolithic: vertical CN + BE lateral in one coupled solve
    std::vector<double> LA, LB, LC, LD;
    void substepMono(double tau, double tol) {
        const int nf = L.nf;
        gather();
        net.assemble(LK.data(), LRC.data());
        LA.resize(LT.size()); LB.resize(LT.size()); LC.resize(LT.size()); LD.resize(LT.size());
        for (int bi = 0; bi < nb; ++bi) {
            auto& b = batches[bi];
            alignas(64) double fl[BS];
            for (int f = 0; f < BS; ++f) fl[f] = Fbar;
            b.setFluxContiguous(fl, BS);
            thermal::updateMaterialPropertiesBatchSIMD(b);
            thermal::applySurfaceBCBatch(b, tau);
            thermal::buildTridiagonalSystemBatch(b, tau);
            for (int li = 0; li < n_int; ++li)
                for (int f = 0; f < BS; ++f) {
                    const int j = bi * BS + f;
                    if (j >= nf) continue;
                    LA[li * nf + j] = b.a_coef[li * BS + f];
                    LB[li * nf + j] = b.b_coef[li * BS + f];
                    LC[li * nf + j] = b.c_coef[li * BS + f];
                    LD[li * nf + j] = b.d_coef[li * BS + f];
                }
        }
        const lat::Network::Stats st = net.implicitCoupled(LA.data(), LB.data(), LC.data(), LD.data(), tau, tol, LT.data());
        if (st.capped) { std::fprintf(stderr, "FATAL: coupled PCG capped (%d its, residual %.2e)\n", st.capped_iterations, st.capped_rel_residual); std::exit(1); }
        coupled_its += st.iterations; ++coupled_calls;
        closure_worst = std::max(closure_worst, st.closure_worst);
        for (int l = 1; l <= nl - 2; ++l)
            for (int j = 0; j < nf; ++j) batches[j / BS].T[l * BS + j % BS] = LT[(l - 1) * nf + j];
        for (auto& b : batches) thermal::applyBottomBCBatch(b);
    }
    long coupled_its = 0, coupled_calls = 0;
    double closure_worst = 0.0;
    void substep(double tau, double tol) {
        const int nf = L.nf;
        if (mono && !lat_off) { substepMono(tau, tol); return; }
        if (lat_off) {
            for (int bi = 0; bi < nb; ++bi) {
                auto& b = batches[bi];
                alignas(64) double fl[BS];
                for (int f = 0; f < BS; ++f) fl[f] = Fbar;
                b.setFluxContiguous(fl, BS);
                b.advanceTimestep(tau);
            }
            return;
        }
        gather();
        net.assemble(LK.data(), LRC.data());
        net.explicitIncrement(LT.data(), tau, LSRC.data());
        for (int bi = 0; bi < nb; ++bi) {
            auto& b = batches[bi];
            alignas(64) double fl[BS];
            for (int f = 0; f < BS; ++f) fl[f] = Fbar;
            b.setFluxContiguous(fl, BS);
            alignas(64) double src[thermal::MAX_THERMAL_LAYERS * BS];
            for (int li = 0; li < n_int; ++li)
                for (int f = 0; f < BS; ++f) {
                    const int j = bi * BS + f;
                    src[li * BS + f] = (j < nf) ? LSRC[li * nf + j] : 0.0;
                }
            b.advanceTimestepWithSource(tau, src);
        }
        for (int l = 1; l <= nl - 2; ++l)
            for (int j = 0; j < nf; ++j)
                LT[(l - 1) * nf + j] = batches[j / BS].T[l * BS + j % BS] - LSRC[(l - 1) * nf + j];
        const lat::Network::Stats st = net.implicitBE(LT.data(), tau, tol);
        if (st.capped) { std::fprintf(stderr, "FATAL: lateral PCG capped (layer %d)\n", st.capped_layer + 1); std::exit(1); }
        for (int l = 1; l <= nl - 2; ++l)
            for (int j = 0; j < nf; ++j) batches[j / BS].T[l * BS + j % BS] = LT[(l - 1) * nf + j];
        for (auto& b : batches) thermal::applyBottomBCBatch(b);
    }
    double T(int j, int l) const { return batches[j / BS].T[l * BS + j % BS]; }
};

static void init(Sim& S, int nr, int nc, double a, double T0, double dT) {
    S.L = makeLattice(nr, nc, a);
    thermal::Planet planet("Moon");
    planet.Qb = 0.0;
    thermal::Config config;
    thermal::Profile ref(planet, 0.0, config, false);
    S.nl = static_cast<int>(ref.nlayers);
    S.n_int = S.nl - 2;
    S.nb = (S.L.nf + BS - 1) / BS;
    S.batches.assign(S.nb, thermal::ThermalBatchSoA());
    for (int bi = 0; bi < S.nb; ++bi) {
        auto& b = S.batches[bi];
        b.initialize(S.nl, BS, planet, config);
        b.copyGridFromProfile(ref);
        for (int f = 0; f < BS; ++f) {
            b.Qb[f] = 0.0;
            const int j = bi * BS + f;
            const double sgn = (j < S.L.nf) ? (S.L.color[j] ? 0.5 : -0.5) : 0.0;
            for (int l = 0; l < S.nl; ++l) b.T[l * BS + f] = T0 + ((l >= 1 && l <= S.nl - 2) ? sgn * dT : 0.0);
        }
        b.n_faces = BS;
        thermal::updateMaterialPropertiesBatchSIMD(b);
        thermal::applyBottomBCBatch(b);
    }
    S.T0 = T0;
    S.Fbar = S.batches[0].emissivity[0] * S.batches[0].sigma_sb * std::pow(T0, 4);
    S.hl.resize(S.n_int);
    for (int l = 1; l <= S.nl - 2; ++l) S.hl[l - 1] = 0.5 * (ref.z()[l + 1] - ref.z()[l - 1]);
    const size_t N = static_cast<size_t>(S.n_int) * S.L.nf;
    S.LT.assign(N, 0.0); S.LK = S.LT; S.LRC = S.LT; S.LSRC = S.LT;
    S.net.init(&S.L.adj, nullptr, S.hl, 1);
}

// lateral anomaly norm: sqrt(sum_l sum_i A_i (T_i - nbr-mean_i)^2) over capacitive layers
static double anomaly(const Sim& S) {
    double s = 0.0;
    for (int l = 1; l <= S.nl - 2; ++l)
        for (int i = 0; i < S.L.nf; ++i) {
            double m = 0.0; int n = 0;
            for (int e = 0; e < 3; ++e) { const int j = S.L.adj.nbr[3 * i + e]; if (j >= 0) { m += S.T(j, l); ++n; } }
            if (!n) continue;
            const double d = S.T(i, l) - m / n;
            s += S.L.adj.area[i] * d * d;
        }
    return std::sqrt(s);
}

static double meanT(const Sim& S) {   // area- and h-weighted mean over capacitive nodes
    double s = 0.0, w = 0.0;
    for (int l = 1; l <= S.nl - 2; ++l)
        for (int i = 0; i < S.L.nf; ++i) { const double ww = S.L.adj.area[i] * S.hl[l - 1]; s += ww * S.T(i, l); w += ww; }
    return s / w;
}

struct Result { double its_per_call = 0.0, anomaly_1d = 0.0, global_viol = 0.0; double bound_worst, anomaly_worst_growth, d1_med, d1_max, dL_med, dL_max, mean_drift_1d, mean_drift_L,
                fo_max; long n_bound_viol; bool ok; };

static Result runCase(double T0, double dT, double mult, int n_div_ref) {
    const double a = 2.48e-3 / std::sqrt(mult);
    const int nr = 12, nc = 12;
    Sim S, R;
    init(S, nr, nc, a, T0, dT);
    init(R, nr, nc, a, T0, dT);
    S.lat_off = R.lat_off = std::getenv("G15_LATOFF") && std::getenv("G15_LATOFF")[0] == '1';
    S.mono = R.mono = std::getenv("G15_SCHEME") && std::string(std::getenv("G15_SCHEME")) == "monolithic";
    const double P = 29.530588 * 86400.0, dt = P / 360.0;
    thermal::Planet planet("Moon");
    const int n_sub = std::max(1, static_cast<int>(std::ceil(dt / (planet.day / 768.0))));
    const double tau = dt / n_sub;
    const int steps_day = static_cast<int>(std::lround(86400.0 / dt * 1.0));   // ~12 output steps = 1 day
    Result res{};
    res.ok = true;
    double an_prev = anomaly(S);
    const double an0 = an_prev;   // growth is gated only while the norm is above 1e-6 of its start
    std::vector<double> prevT;
    auto snapshot = [&](const Sim& X) {
        std::vector<double> v(static_cast<size_t>(X.nl) * X.L.nf);
        for (int l = 0; l < X.nl; ++l) for (int i = 0; i < X.L.nf; ++i) v[static_cast<size_t>(l) * X.L.nf + i] = X.T(i, l);
        return v;
    };
    auto compare = [&](double* med, double* mx) {
        std::vector<double> d;
        for (int l = 0; l < S.nl; ++l) for (int i = 0; i < S.L.nf; ++i) d.push_back(std::fabs(S.T(i, l) - R.T(i, l)));
        std::sort(d.begin(), d.end());
        *med = d[d.size() / 2]; *mx = d.back();
    };
    // Fo report: max over layers/faces of tau sum G / C at the start
    S.gather();
    S.net.assemble(S.LK.data(), S.LRC.data());
    {
        lat::Network probe;
        probe.init(&S.L.adj, nullptr, S.hl, 1);
        probe.assemble(S.LK.data(), S.LRC.data());
        std::vector<double> tmp = S.LT;
        res.fo_max = probe.implicitBE(tmp.data(), tau, 1e-13).fo_max;
    }
    const double m0 = meanT(S);
    for (int step = 1; step <= 360; ++step) {
        for (int k = 0; k < n_sub; ++k) {
            prevT = snapshot(S);
            const double gmin = *std::min_element(prevT.begin(), prevT.end()), gmax = *std::max_element(prevT.begin(), prevT.end());
            double gmin_new = 1e300, gmax_new = -1e300;
            S.substep(tau, 1e-12);
            // bounds: each capacitive node within [min, max] of its previous-step stencil +- 0.1 K
            for (int l = 1; l <= S.nl - 2; ++l)
                for (int i = 0; i < S.L.nf; ++i) {
                    double lo = 1e300, hi = -1e300;
                    auto take = [&](int ll, int ii) { const double v = prevT[static_cast<size_t>(ll) * S.L.nf + ii]; lo = std::min(lo, v); hi = std::max(hi, v); };
                    take(l, i); take(l - 1, i); take(l + 1, i); take(0, i); take(S.nl - 1, i);
                    for (int e = 0; e < 3; ++e) { const int j = S.L.adj.nbr[3 * i + e]; if (j >= 0) take(l, j); }
                    lo = std::min(lo, S.T(i, 0)); hi = std::max(hi, S.T(i, 0));   // the new surface value is a BC value
                    const double v = S.T(i, l);
                    gmin_new = std::min(gmin_new, v); gmax_new = std::max(gmax_new, v);
                    const double viol = std::max(lo - v, v - hi);
                    if (viol > 0.1) { ++res.n_bound_viol; }
                    res.bound_worst = std::max(res.bound_worst, viol);
                }
            {
                double smin = 1e300, smax = -1e300;   // the new surface values are boundary values too
                for (int i = 0; i < S.L.nf; ++i) { smin = std::min(smin, S.T(i, 0)); smax = std::max(smax, S.T(i, 0)); }
                const double gv = std::max(std::min(gmin, smin) - gmin_new, gmax_new - std::max(gmax, smax));
                res.global_viol = std::max(res.global_viol, gv);
            }
            const double an = anomaly(S);
            if (an_prev > 1e-6 * an0) res.anomaly_worst_growth = std::max(res.anomaly_worst_growth, an / an_prev - 1.0);
            an_prev = an;
        }
        for (int k = 0; k < n_sub * n_div_ref; ++k) R.substep(tau / n_div_ref, 1e-12);
        if (step == steps_day) { res.anomaly_1d = anomaly(S) / an0; compare(&res.d1_med, &res.d1_max); res.mean_drift_1d = (meanT(S) - m0) - (meanT(R) - m0); }
    }
    compare(&res.dL_med, &res.dL_max);
    res.mean_drift_L = meanT(S) - meanT(R);
    res.its_per_call = S.coupled_calls ? static_cast<double>(S.coupled_its) / S.coupled_calls : 0.0;
    res.ok = res.n_bound_viol == 0 && res.anomaly_worst_growth <= 0.01 && res.d1_med <= 0.1 && res.d1_max <= 1.0 &&
             res.dL_med <= 0.1 && res.dL_max <= 1.0;
    return res;
}

int main() {
    setenv("PTM_SURFACE_BC_TOLK", "1e-10", 1);
    setvbuf(stdout, nullptr, _IOLBF, 0);   // line-buffered: progress is visible through a pipe
    std::vector<double> mults = {1.0, 3.0, 10.0};
    if (const char* e = std::getenv("G15_ONLY_MULT")) mults = {std::atof(e)};
    if (const char* e = std::getenv("G15_EXTRA_MULT")) {
        std::string s(e);
        size_t p = 0;
        while (p < s.size()) { size_t q = s.find(',', p); if (q == std::string::npos) q = s.size(); mults.push_back(std::atof(s.substr(p, q - p).c_str())); p = q + 1; }
    }
    const int ndiv = std::getenv("G15_REF_DIV") ? std::atoi(std::getenv("G15_REF_DIV")) : 64;
    int fails = 0;
    std::printf("G15 synthetic checkerboard (scheme %s; 288 faces; reference = same scheme at dt/%d)\n", std::getenv("G15_LATOFF") ? "LATERAL OFF (control)" : (std::getenv("G15_SCHEME") ? std::getenv("G15_SCHEME") : "balanced split"), ndiv);
    for (double T0 : {60.0, 110.0})
        for (double dT : {10.0, 30.0, 100.0})
            for (double m : mults) {
                if (dT >= 2.0 * T0 - 20.0) continue;   // keep the cold side above ~10 K
                if (std::getenv("G15_ONLY_T") && std::atof(std::getenv("G15_ONLY_T")) != T0) continue;
                if (std::getenv("G15_ONLY_DT") && std::atof(std::getenv("G15_ONLY_DT")) != dT) continue;
                const Result r = runCase(T0, dT, m, ndiv);
                const bool gated = (m == 1.0 || m == 3.0 || m == 10.0);
                if (gated && !r.ok) ++fails;
                std::printf("  T0=%3.0f dT=%5.1f Fo x%-6g (Fo_max %.2e): GLOBAL bound worst %+.3f K | local-stencil viol %ld (worst %+.3f K) | anomaly growth max %+.2e (norm/start at 1 day %.1e)"
                            " | 1 day |dT| med %.2e max %.2e | 1 lunation med %.2e max %.2e | mean drift vs ref 1d %+.3e"
                            " lun %+.3e K | coupled CG its/substep %.1f  [%s%s]\n", T0, dT, m, r.fo_max, r.global_viol, r.n_bound_viol, r.bound_worst, r.anomaly_worst_growth, r.anomaly_1d,
                            r.d1_med, r.d1_max, r.dL_med, r.dL_max, r.mean_drift_1d, r.mean_drift_L, r.its_per_call, r.ok ? "pass" : "FAIL",
                            gated ? "" : ", diagnostic");
            }
    std::printf(fails ? "\nRESULT: FAIL (%d gated cases)\n" : "\nRESULT: PASS (all gated cases)\n", fails);
    return fails ? 1 : 0;
}
