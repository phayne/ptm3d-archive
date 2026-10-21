// test_lateral_g10prime.cpp -- G10' driven-response gate for C1c lateral conduction
// (Fermi addenda 2026-10-18 and 2026-10-19, LATERAL_SIGNOFF_FERMI.md).
//
// WHAT RUNS: the ENGINE's own code. Each CN substep tau is the engine's balanced (Douglas) split,
// built from the same calls ptm_thermal_sim's time-outer driver makes:
//     ptm::lateral::Network::assemble / explicitIncrement   (substep-start k, rho c, T)
//     thermal::ThermalBatchSoA::advanceTimestepWithSource   (vertical CN + explicit lateral source)
//     T* - inc, then ptm::lateral::Network::implicitBE      (BE lateral correction, all layers)
//     thermal::applyBottomBCBatch                           (the bottom node is algebraic)
// with n_sub = ceil(dt / (day/768)) substeps per output step and the driver's midpoint flux
// interpolation, at the production dt = P/360 = 7087 s. (G10P_SPLIT=strang runs the superseded
// Strang split with TR-BDF2 / BE halves, kept as the diagnostic that found the steady-state defect.)
// What this harness does NOT exercise: the engine's per-batch state save/load and its scatter of
// the layer arrays (that plumbing is covered by G0b with lateral OFF and by the engine-vs-harness
// comparison, G10'-engine).
//
// Geometry: a 1-D chain of equal equilateral facets (Neumann ends), so the lateral mode
// x_j = cos(pi (j + 1/2) / N) is an EXACT eigenvector of the discrete lateral operator; the chain
// is sized so the mode's half-wavelength is Lambda/2. This isolates the TIME and SPLITTING error
// (G2/G3 cover the spatial operator).
// Base state: uniform T0 (60 or 110 K) with Qb = 0 and F_bar = eps sigma T0^4 (an exact steady
// state of the column). Forcing perturbation on the q-mode only: dF = a * 0.5 x_j * sin(n w t),
// a = 1e-4 F_bar (n = 0: a constant 0.5 a x_j). Amplitude choice, MEASURED (110 K, 10 mm, n = 0):
// the lateral-OFF arm carries an absolute round-off floor of ~1.4e-11 K (weakly damped column,
// ~2e5 substeps), so its increment error scaled exactly as 1/a: 1.0e-5 / 1.05e-6 / 9.1e-8 / 9.3e-9
// of S at a = 1e-7 / 1e-6 / 1e-5 / 1e-4, while the ON arm's absolute error stayed <= 7.3e-11
// (no nonlinearity visible). At the old a = 1e-6 the floor alone is the size of the 1e-6 bar.
// Reference: the exact periodic solution of the LINEARIZED semi-discrete system (same z-grid, same
// nodal coefficients, same surface two-point balance, lateral eigenvalue mu_i per node), in the
// frequency domain. Lateral OFF = the same with mu = 0.
//
// GATES (Fermi addendum 2, item 3: the bars apply to the LATERAL INCREMENT):
//   Delta = X(ON) - X(OFF) (numerical), Delta_ref = R(ON) - R(OFF); S = |R_surf(ON)|.
//   n = 0:  |Delta - Delta_ref| / S <= 1e-6, surface and node 3   (the steady-state check)
//   n >= 1: amplitude  (|Delta| - |Delta_ref|) / S <= 1 % / 2 % / 5 %  (n = 1 / 3 / 10), surface and node 3
//           phase arg(Delta / Delta_ref) <= 0.5 / 1 / 3 deg, ONLY where |Delta_ref| >= 0.1 S
//   ABSOLUTE surface bars (ON vs R(ON); Fermi: "keep them"): n = 0 rel 1e-6; amp 1/2/5 %, phase
//           0.5/1/3 deg. The absolute NODE-3 error is the production vertical solver's own
//           limit (e.g. -1.12 deg at 60 K, n = 3), logged, not gated.
// The lateral-OFF arm is spun up on the column's own slowest VERTICAL eigenvalue (30 e-folds);
// the ON arm on min(mu) + that eigenvalue. Each arm reports its period-to-period change.
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "mesh/lateral_adjacency.h"
#include "thermal/lateral_conduction.h"
#include "thermal/thermal_model.h"

using cd = std::complex<double>;
namespace lat = ptm::lateral;
constexpr int BS = thermal::THERMAL_BATCH_SIZE;

struct Chain {
    lat::Adjacency adj;
    std::vector<double> x;   // eigenvector
    int N;
};

static Chain makeChain(int N, double Lambda) {
    // one row of N equilateral triangles; centroid spacing along x is a/2, so N cells span
    // N a / 2 = Lambda / 2 (mode m = 1 is a half-wavelength over the chain)
    const double a = Lambda / N;
    const int ncols = N / 2;
    const double hy = a * std::sqrt(3.0) / 2.0;
    std::vector<double> v;
    std::vector<int64_t> f;
    for (int r = 0; r <= 1; ++r)
        for (int c = 0; c <= ncols; ++c) v.insert(v.end(), {c * a + (r ? 0.5 * a : 0.0), r * hy, 0.0});
    auto id = [&](int r, int c) { return static_cast<int64_t>(r) * (ncols + 1) + c; };
    for (int c = 0; c < ncols; ++c) {
        f.insert(f.end(), {id(0, c), id(0, c + 1), id(1, c)});
        f.insert(f.end(), {id(0, c + 1), id(1, c + 1), id(1, c)});
    }
    Chain ch;
    ch.adj = lat::build(v, f, {});
    ch.N = N;
    ch.x.resize(N);
    for (int j = 0; j < N; ++j) ch.x[j] = std::cos(M_PI * (j + 0.5) / N);
    return ch;
}

struct Result { cd surf, node3; };

// Exact periodic (or steady, n = 0) response of the linearized semi-discrete system to a
// surface forcing Re[F_hat e^{i w t}] on the q-mode. Nodal coefficients from a batch at T0.
static Result reference(const thermal::ThermalBatchSoA& b, const std::vector<double>& mu, double eps, double T0,
                        double omega, cd F_hat) {
    const int n = b.n_layers;
    std::vector<cd> lo(n, 0.0), di(n, 0.0), up(n, 0.0), rhs(n, 0.0);
    const double sig = b.sigma_sb, k0 = b.k[0], dz0 = b.dz[0];
    di[0] = 4.0 * eps * sig * T0 * T0 * T0 + k0 / dz0;
    up[0] = -k0 / dz0;
    rhs[0] = F_hat;
    for (int i = 1; i <= n - 2; ++i) {
        const double alpha = b.g1[i - 1] * b.k[(i - 1) * BS];
        const double beta = b.g2[i - 1] * b.k[i * BS];
        const double rc = b.rho[i * BS] * b.cp[i * BS];
        lo[i] = -alpha;
        up[i] = -beta;
        di[i] = cd(alpha + beta + rc * mu[i], omega * rc);
    }
    lo[n - 1] = -1.0;
    di[n - 1] = 1.0;   // T_{n-1} = T_{n-2} (Qb = 0)
    for (int i = 1; i < n; ++i) {
        const cd m = lo[i] / di[i - 1];
        di[i] -= m * up[i - 1];
        rhs[i] -= m * rhs[i - 1];
    }
    std::vector<cd> T(n);
    T[n - 1] = rhs[n - 1] / di[n - 1];
    for (int i = n - 2; i >= 0; --i) T[i] = (rhs[i] - up[i] * T[i + 1]) / di[i];
    return {T[0], T[3]};
}

// Slowest decay rate (1/s) of the linearized lateral-OFF column (surface radiative two-point
// balance, insulating bottom), by inverse iteration on rc dT/dt = -K T over the capacitive nodes
// with the algebraic surface and bottom nodes eliminated.
static double slowestVerticalRate(const thermal::ThermalBatchSoA& b, double eps, double T0) {
    const int n = b.n_layers, m = n - 2;
    std::vector<double> lo(m, 0.0), di(m, 0.0), up(m, 0.0), rc(m);
    const double hs = 4.0 * eps * b.sigma_sb * T0 * T0 * T0, k0 = b.k[0], dz0 = b.dz[0];
    const double s1 = (k0 / dz0) / (hs + k0 / dz0);   // T0 = s1 T1 (+ forcing)
    for (int i = 1; i <= n - 2; ++i) {
        const double alpha = b.g1[i - 1] * b.k[(i - 1) * BS];
        const double beta = b.g2[i - 1] * b.k[i * BS];
        const int r = i - 1;
        rc[r] = b.rho[i * BS] * b.cp[i * BS];
        di[r] = alpha + beta;
        if (i == 1) di[r] -= alpha * s1; else lo[r] = -alpha;
        if (i == n - 2) di[r] -= beta; else up[r] = -beta;   // T_{n-1} = T_{n-2}
    }
    std::vector<double> x(m, 1.0), y(m);
    double lam = 0.0;
    for (int itr = 0; itr < 200; ++itr) {
        std::vector<double> d = di, r(m);
        for (int i = 0; i < m; ++i) r[i] = rc[i] * x[i];
        for (int i = 1; i < m; ++i) { const double w = lo[i] / d[i - 1]; d[i] -= w * up[i - 1]; r[i] -= w * r[i - 1]; }
        y[m - 1] = r[m - 1] / d[m - 1];
        for (int i = m - 2; i >= 0; --i) y[i] = (r[i] - up[i] * y[i + 1]) / d[i];
        double num = 0.0, den = 0.0;
        for (int i = 0; i < m; ++i) { num += x[i] * rc[i] * x[i]; den += x[i] * rc[i] * y[i]; }
        lam = num / den;
        double nrm = 0.0;
        for (double v : y) nrm = std::max(nrm, std::fabs(v));
        for (int i = 0; i < m; ++i) x[i] = y[i] / nrm;
    }
    return lam;
}

struct Case { double T0, Lambda; int harm; };
struct Arm { cd Xs, X3, Rs, R3; double periodic, rate_dt; int periods, n_sub, layers; };

static int envInt(const char* n, int d) { const char* e = std::getenv(n); return (e && *e) ? std::atoi(e) : d; }
static double envD(const char* n, double d) { const char* e = std::getenv(n); return (e && *e) ? std::atof(e) : d; }

enum class Split { Balanced, StrangTRBDF2, StrangBE };

static Arm runArm(const Case& cs, Split split, bool lat_on) {
    const int dtdiv = std::max(1, envInt("G10P_DTDIV", 1));
    thermal::Planet planet("Moon");
    planet.Qb = 0.0;
    thermal::Config config;
    thermal::Profile ref(planet, 0.0, config, false);
    const int nl = static_cast<int>(ref.nlayers);
    const int n_int = nl - 2;
    const int N = 16;
    Chain ch = makeChain(N, cs.Lambda);
    const int nb = (N + BS - 1) / BS;
    std::vector<thermal::ThermalBatchSoA> batches(nb);
    for (auto& b : batches) {
        b.initialize(nl, BS, planet, config);
        b.copyGridFromProfile(ref);
        for (int f = 0; f < BS; ++f) {
            b.Qb[f] = 0.0;
            for (int l = 0; l < nl; ++l) b.T[l * BS + f] = cs.T0;
        }
        b.n_faces = BS;
        thermal::updateMaterialPropertiesBatchSIMD(b);
    }
    const double eps = batches[0].emissivity[0];
    const double Fbar = eps * batches[0].sigma_sb * std::pow(cs.T0, 4);
    const double amp = envD("G10P_AMP", 1e-4) * Fbar;   // see the header: amplitude choice, measured
    const double P = 29.530588 * 86400.0, dt = P / (360.0 * dtdiv);
    const double omega = 2.0 * M_PI * cs.harm / P;
    const int n_sub = std::max(1, static_cast<int>(std::ceil(dt / (planet.day / 768.0))));
    const double dt_sub = dt / n_sub;

    // layer arrays [layer][face], node layers l = 1..nl-2  (the engine's LT/LK/LRC)
    std::vector<double> LT(static_cast<size_t>(n_int) * N), LK(LT.size()), LRC(LT.size()), LSRC(LT.size());
    std::vector<double> hl(n_int);
    for (int l = 1; l <= nl - 2; ++l) hl[l - 1] = 0.5 * (ref.z()[l + 1] - ref.z()[l - 1]);
    auto gather = [&]() {
        for (int l = 1; l <= nl - 2; ++l)
            for (int j = 0; j < N; ++j) {
                const auto& b = batches[j / BS];
                const int f = j % BS;
                LT[(l - 1) * N + j] = b.T[l * BS + f];
                LK[(l - 1) * N + j] = b.k[l * BS + f];
                LRC[(l - 1) * N + j] = b.rho[l * BS + f] * b.cp[l * BS + f];
            }
    };
    auto scatterT = [&]() {
        for (int l = 1; l <= nl - 2; ++l)
            for (int j = 0; j < N; ++j) batches[j / BS].T[l * BS + j % BS] = LT[(l - 1) * N + j];
        for (auto& b : batches) thermal::applyBottomBCBatch(b);
    };

    // lateral eigenvalue per capacitive node (exact eigenvector): mu_i = (L x)_j / (C_j x_j)
    std::vector<double> mu(nl, 0.0);
    gather();
    double mu_min = 1e300;
    for (int l = 1; l <= nl - 2; ++l) {
        std::vector<double> G;
        lat::assembleConductance(ch.adj, nullptr, &LK[(l - 1) * N], hl[l - 1], G);
        double Lx = 0.0;   // at j = 0
        for (int e = 0; e < 3; ++e) {
            const int32_t nb2 = ch.adj.nbr[e];
            if (nb2 >= 0) Lx += G[e] * (ch.x[0] - ch.x[nb2]);
        }
        const double m = Lx / (LRC[(l - 1) * N] * ch.adj.area[0] * hl[l - 1] * ch.x[0]);
        mu[l] = lat_on ? m : 0.0;
        mu_min = std::min(mu_min, m);
    }
    const double lam_v = slowestVerticalRate(batches[0], eps, cs.T0);
    const cd F_hat = (cs.harm == 0) ? cd(0.5 * amp, 0.0) : cd(0.0, -0.5 * amp);   // sin -> Re[-i e^{iwt}]
    const Result R = reference(batches[0], mu, eps, cs.T0, omega, F_hat);

    // the engine's lateral network on this chain (all edges active, 1 thread)
    lat::Network net;
    net.init(&ch.adj, nullptr, hl, 1);
    const double cg_tol = envD("G10P_CGTOL", 1e-13);   // engine: 1e-8; tighter here so CG cannot enter a 1e-6 bar
    auto balancedSubstep = [&](double tau, const double* flA, const double* flB, double al) {
        if (lat_on) {   // substep-start state (the engine gathers it after its material update)
            for (auto& b : batches) thermal::updateMaterialPropertiesBatchSIMD(b);
            gather();
            net.assemble(LK.data(), LRC.data());
            net.explicitIncrement(LT.data(), tau, LSRC.data());
        }
        for (int bi = 0; bi < nb; ++bi) {
            auto& b = batches[bi];
            alignas(64) double fl2[BS];
            for (int f = 0; f < BS; ++f) fl2[f] = (1.0 - al) * flA[bi * BS + f] + al * flB[bi * BS + f];
            b.setFluxContiguous(fl2, BS);
            if (lat_on) {
                alignas(64) double src[thermal::MAX_THERMAL_LAYERS * BS];
                for (int li = 0; li < n_int; ++li)
                    for (int f = 0; f < BS; ++f) src[li * BS + f] = LSRC[li * N + bi * BS + f];
                b.advanceTimestepWithSource(tau, src);
            } else {
                b.advanceTimestep(tau);
            }
        }
        if (lat_on) {
            for (int l = 1; l <= nl - 2; ++l)
                for (int j = 0; j < N; ++j)
                    LT[(l - 1) * N + j] = batches[j / BS].T[l * BS + j % BS] - LSRC[(l - 1) * N + j];
            const lat::Network::Stats st = net.implicitBE(LT.data(), tau, cg_tol);
            if (st.capped) { std::fprintf(stderr, "FATAL: lateral PCG capped (layer %d)\n", st.capped_layer + 1); std::exit(1); }
            scatterT();
        }
    };
    // Strang (superseded; diagnostic only): half lateral | CN substeps | half lateral
    lat::SolverOptions opt;
    opt.scheme = (split == Split::StrangBE) ? lat::Scheme::BackwardEuler : lat::Scheme::TRBDF2;
    opt.rel_tol = cg_tol;
    opt.max_iter = 1000;
    lat::Workspace ws;
    auto lateralHalf = [&](double h_dt) {
        if (!lat_on) return;
        for (auto& b : batches) thermal::updateMaterialPropertiesBatchSIMD(b);
        gather();
        for (int l = 1; l <= nl - 2; ++l) {
            lat::StepStats st = lat::backwardEulerStep(ch.adj, nullptr, &LT[(l - 1) * N], &LK[(l - 1) * N],
                                                       &LRC[(l - 1) * N], hl[l - 1], h_dt, opt, ws);
            if (st.capped) { std::fprintf(stderr, "FATAL: lateral CG capped (layer %d)\n", l); std::exit(1); }
        }
        scatterT();
    };
    auto flux_at = [&](double t, int j) {
        const double s = (cs.harm == 0) ? 1.0 : std::sin(omega * t);
        return Fbar + amp * 0.5 * ch.x[j] * s;
    };
    // spin-up: 30 e-folds of the slowest mode (vertical for OFF; >= mu_min + vertical for ON)
    const double rate = lat_on ? (mu_min + lam_v) : lam_v;
    const double t_spin = envD("G10P_EFOLDS", 30.0) / rate;
    const int steps_per_period = ((cs.harm == 0) ? 360 : 360 / cs.harm) * dtdiv;
    const int n_periods = std::max(3, static_cast<int>(std::ceil(t_spin / (steps_per_period * dt)))) + 2;
    // G10P_ENGINE_DUMP=<dir> (G10'-engine): run G10P_ENGINE_STEPS output steps from the uniform T0 start, with
    // no spin-up, and write what ptm_thermal_sim needs to run the SAME case (chain.ply, flux.bin) plus this
    // harness's surface series and final column for the comparison.
    const char* dump_dir = std::getenv("G10P_ENGINE_DUMP");
    const long n_steps = dump_dir ? std::max(1, envInt("G10P_ENGINE_STEPS", 360))
                                  : static_cast<long>(n_periods) * steps_per_period;
    std::vector<double> dump_surf;
    if (dump_dir) {
        const std::string d(dump_dir);
        // chain.ply: the chain's vertices and faces, %.17g (the engine rebuilds the adjacency from them)
        {
            FILE* f = std::fopen((d + "/chain.ply").c_str(), "w");
            if (!f) { std::fprintf(stderr, "FATAL: cannot write %s/chain.ply\n", d.c_str()); std::exit(2); }
            const int ncols = N / 2, nv = 2 * (ncols + 1);
            const double a = cs.Lambda / N, hy = a * std::sqrt(3.0) / 2.0;
            std::fprintf(f, "ply\nformat ascii 1.0\nelement vertex %d\nproperty double x\nproperty double y\n"
                            "property double z\nelement face %d\nproperty list uchar int vertex_indices\nend_header\n", nv, N);
            for (int r = 0; r <= 1; ++r)
                for (int c = 0; c <= ncols; ++c) std::fprintf(f, "%.17g %.17g %.17g\n", c * a + (r ? 0.5 * a : 0.0), r * hy, 0.0);
            auto id = [&](int r, int c) { return r * (ncols + 1) + c; };
            for (int c = 0; c < ncols; ++c) {
                std::fprintf(f, "3 %d %d %d\n", id(0, c), id(0, c + 1), id(1, c));
                std::fprintf(f, "3 %d %d %d\n", id(0, c + 1), id(1, c + 1), id(1, c));
            }
            std::fclose(f);
        }
        // flux.bin (PTMFLXR1): flux_at(t_k) for k = 0..n_steps (the engine's total_time_steps = n_steps + 1 frames)
        {
            std::ofstream o(d + "/flux.bin", std::ios::binary);
            const uint64_t ns = static_cast<uint64_t>(n_steps) + 1, nf = static_cast<uint64_t>(N);
            o.write("PTMFLXR1", 8);
            o.write(reinterpret_cast<const char*>(&ns), 8);
            o.write(reinterpret_cast<const char*>(&nf), 8);
            for (uint64_t k = 0; k < ns; ++k)
                for (int j = 0; j < N; ++j) { const double v = Fbar + amp * 0.5 * ch.x[j] * ((cs.harm == 0) ? 1.0 : std::sin(omega * k * dt)); o.write(reinterpret_cast<const char*>(&v), 8); }
        }
        for (int j = 0; j < N; ++j)
            dump_surf.push_back(thermal::outputSurfaceTempBatch(batches[j / BS], j % BS, Fbar + amp * 0.5 * ch.x[j] * ((cs.harm == 0) ? 1.0 : 0.0)));
        std::printf("G10'-engine dump: T0=%g L=%g mm n=%d amp=%g F_bar, %ld steps, dt=%.9g s, n_sub=%d -> %s\n",
                    cs.T0, 1e3 * cs.Lambda, cs.harm, amp / Fbar, n_steps, dt, n_sub, dump_dir);
    }
    std::vector<double> qs(steps_per_period), q3(steps_per_period), qs_prev(steps_per_period);
    std::vector<double> flA(N), flB(N);
    for (long s = 0; s < n_steps; ++s) {
        const double t0 = s * dt, t1 = (s + 1) * dt;
        for (int j = 0; j < N; ++j) { flA[j] = flux_at(t0, j); flB[j] = flux_at(t1, j); }
        if (split == Split::Balanced) {
            for (int k = 0; k < n_sub; ++k) balancedSubstep(dt_sub, flA.data(), flB.data(), (k + 0.5) / n_sub);
        } else {
            lateralHalf(0.5 * dt);
            for (int bi = 0; bi < nb; ++bi) {
                auto& b = batches[bi];
                for (int k = 0; k < n_sub; ++k) {
                    const double al = (k + 0.5) / n_sub;
                    alignas(64) double fl[BS];
                    for (int f = 0; f < BS; ++f) fl[f] = (1.0 - al) * flA[bi * BS + f] + al * flB[bi * BS + f];
                    b.setFluxContiguous(fl, BS);
                    b.advanceTimestep(dt_sub);
                }
            }
            lateralHalf(0.5 * dt);
        }
        if (dump_dir)
            for (int j = 0; j < N; ++j) dump_surf.push_back(thermal::outputSurfaceTempBatch(batches[j / BS], j % BS, flux_at(t1, j)));
        // q-mode projections at t1: surface (forcing-epoch re-evaluation, as the driver outputs) and node 3
        double ps = 0.0, p3 = 0.0, xx = 0.0;
        for (int j = 0; j < N; ++j) {
            const auto& b = batches[j / BS];
            const int f = j % BS;
            ps += ch.x[j] * (thermal::outputSurfaceTempBatch(b, f, flux_at(t1, j)) - cs.T0);
            p3 += ch.x[j] * (b.T[3 * BS + f] - cs.T0);
            xx += ch.x[j] * ch.x[j];
        }
        const long k = (s + 1) % steps_per_period;
        if (s + 1 >= n_steps - 2L * steps_per_period && s + 1 < n_steps - steps_per_period) qs_prev[k] = ps / xx;
        if (s + 1 >= n_steps - steps_per_period) { qs[k] = ps / xx; q3[k] = p3 / xx; }
    }
    if (dump_dir) {
        const std::string d(dump_dir);
        std::ofstream o(d + "/harness_surface.bin", std::ios::binary);   // float64 [n_steps + 1][N]
        o.write(reinterpret_cast<const char*>(dump_surf.data()), static_cast<std::streamsize>(dump_surf.size() * 8));
        std::ofstream o2(d + "/harness_final.bin", std::ios::binary);    // float64 [n_layers][N]
        for (int l = 0; l < nl; ++l)
            for (int j = 0; j < N; ++j) { const double v = batches[j / BS].T[l * BS + j % BS]; o2.write(reinterpret_cast<const char*>(&v), 8); }
        o.close(); o2.close();   // std::exit skips the destructors: flush explicitly
        if (!o || !o2) { std::fprintf(stderr, "FATAL: could not write the dump files in %s\n", d.c_str()); std::exit(2); }
        std::exit(0);
    }
    cd Xs = 0.0, X3 = 0.0, Xp = 0.0;
    for (int k = 0; k < steps_per_period; ++k) {
        const double t = k * dt;   // phase of the sample within the period (t1 mod period)
        const cd e = (cs.harm == 0) ? cd(1.0, 0.0) : std::exp(cd(0.0, -omega * t));
        Xs += qs[k] * e; X3 += q3[k] * e; Xp += qs_prev[k] * e;
    }
    const double norm = (cs.harm == 0) ? 1.0 / steps_per_period : 2.0 / steps_per_period;
    Xs *= norm; X3 *= norm; Xp *= norm;
    Arm a;
    a.Xs = Xs; a.X3 = X3; a.Rs = R.surf; a.R3 = R.node3;
    a.periodic = std::abs(Xs - Xp) / std::abs(Xs);
    a.rate_dt = rate * dt;
    a.periods = n_periods; a.n_sub = n_sub; a.layers = nl;
    return a;
}

int main() {
    if (envInt("G10P_PRINT_Z", 0)) {   // the production reference grid (Moon defaults), for C1b / G9
        thermal::Planet planet("Moon");
        thermal::Config config;
        thermal::Profile ref(planet, 0.0, config, false);
        std::printf("# PTM3D CPU-SIMD reference depth grid (Planet Moon, default Config): %zu nodes, z [m]\n", static_cast<size_t>(ref.nlayers));
        for (size_t i = 0; i < ref.nlayers; ++i) std::printf("%zu %.9e\n", i, ref.z()[i]);
        return 0;
    }
    setenv("PTM_SURFACE_BC_TOLK", "1e-13", 1);   // diagnostic override: the Newton tolerance must not enter a 1e-6 bar
    const char* sp = std::getenv("G10P_SPLIT");
    const std::string sps = sp ? sp : "balanced";
    std::vector<Split> splits;
    if (sps == "balanced") splits = {Split::Balanced};
    else if (sps == "strang") splits = {Split::StrangTRBDF2, Split::StrangBE};
    else { std::fprintf(stderr, "G10P_SPLIT must be 'balanced' (default) or 'strang'\n"); return 2; }
    int fails = 0, n_cases = 0, n_phase_gated = 0;
    double worst[2][4] = {{0}};   // [surf/node3][n index] worst normalized increment amplitude error
    for (Split split : splits) {
        std::printf("G10' %s  (production dt = P/360 x 1/%d, CG tol %.0e)\n",
                    split == Split::Balanced ? "BALANCED split (the engine's: Network + advanceTimestepWithSource)"
                                             : (split == Split::StrangTRBDF2 ? "Strang, TR-BDF2 halves (superseded)"
                                                                            : "Strang, BE halves (superseded)"),
                    std::max(1, envInt("G10P_DTDIV", 1)), envD("G10P_CGTOL", 1e-13));
        std::printf("  increment metric: d|D| = (|D|-|Dref|)/S, phase = arg(D/Dref) (gated if |Dref| >= 0.1 S); "
                    "abs = ON vs R(ON), surface\n");
        for (double T0 : {60.0, 110.0})
            for (double L : {1e-3, 3e-3, 1e-2, 3e-2, 1e-1})
                for (int n : {0, 1, 3, 10}) {
                    if (envInt("G10P_ONLY_N", -1) >= 0 && n != envInt("G10P_ONLY_N", -1)) continue;
                    if (envInt("G10P_ONLY_LMM", -1) >= 0 && std::lround(1e3 * L) != envInt("G10P_ONLY_LMM", -1)) continue;
                    if (envInt("G10P_ONLY_T", -1) >= 0 && std::lround(T0) != envInt("G10P_ONLY_T", -1)) continue;
                    const Case cs{T0, L, n};
                    const Arm on = runArm(cs, split, true);
                    const Arm off = runArm(cs, split, false);
                    const double S = std::abs(on.Rs);
                    const cd Ds = on.Xs - off.Xs, Dsr = on.Rs - off.Rs;
                    const cd D3 = on.X3 - off.X3, D3r = on.R3 - off.R3;
                    const int ni = (n == 0) ? 0 : (n == 1) ? 1 : (n == 3) ? 2 : 3;
                    const double bar_a = (n == 0) ? 1e-6 : (n == 1) ? 0.01 : (n == 3) ? 0.02 : 0.05;
                    const double bar_p = (n == 0) ? 1e9 : (n == 1) ? 0.5 : (n == 3) ? 1.0 : 3.0;
                    double es, e3;
                    if (n == 0) { es = std::abs(Ds - Dsr) / S; e3 = std::abs(D3 - D3r) / S; }
                    else { es = (std::abs(Ds) - std::abs(Dsr)) / S; e3 = (std::abs(D3) - std::abs(D3r)) / S; }
                    const bool gs = std::abs(Dsr) >= 0.1 * S, g3 = std::abs(D3r) >= 0.1 * S;
                    const double ps = std::arg(Ds / Dsr) * 180.0 / M_PI, p3 = std::arg(D3 / D3r) * 180.0 / M_PI;
                    // absolute surface (ON vs R(ON))
                    const double as = (n == 0) ? std::abs(on.Xs - on.Rs) / S : std::abs(on.Xs) / S - 1.0;
                    const double aps = (n == 0) ? 0.0 : std::arg(on.Xs / on.Rs) * 180.0 / M_PI;
                    const double a3 = std::abs(on.X3) / std::abs(on.R3) - 1.0;
                    const double ap3 = (n == 0) ? 0.0 : std::arg(on.X3 / on.R3) * 180.0 / M_PI;
                    bool ok = std::fabs(es) <= bar_a && std::fabs(e3) <= bar_a && std::fabs(as) <= bar_a
                              && std::fabs(aps) <= bar_p;
                    if (n > 0 && gs) ok = ok && std::fabs(ps) <= bar_p;
                    if (n > 0 && g3) ok = ok && std::fabs(p3) <= bar_p;
                    n_phase_gated += (n > 0) * (gs + g3);
                    // periodicity must be far inside the bar it would otherwise contaminate
                    const double per = std::max(on.periodic, off.periodic);
                    if (per > 0.1 * bar_a) ok = false;
                    worst[0][ni] = std::max(worst[0][ni], std::fabs(es));
                    worst[1][ni] = std::max(worst[1][ni], std::fabs(e3));
                    ++n_cases;
                    if (split == Split::Balanced && !ok) ++fails;
                    std::printf("  T0=%3.0f L=%5.1fmm n=%2d |Dref|/S surf %.3f node3 %.3f | INC surf d|D| %+.2e ph %+.3f%s"
                                " node3 d|D| %+.2e ph %+.3f%s | ABS surf %+.2e %+.3f deg (node3 %+.2e %+.3f deg, logged)"
                                " | OFF-arm abs surf %+.2e %+.3f deg node3 %+.2e %+.3f deg"
                                " | periods %d/%d per %.0e  [%s]\n",
                                T0, 1e3 * L, n, std::abs(Dsr) / S, std::abs(D3r) / S, es, n ? ps : 0.0,
                                (n && gs) ? "*" : " ", e3, n ? p3 : 0.0, (n && g3) ? "*" : " ", as, aps, a3, ap3,
                                std::abs(off.Xs) / std::abs(off.Rs) - 1.0, n ? std::arg(off.Xs / off.Rs) * 180.0 / M_PI : 0.0,
                                std::abs(off.X3) / std::abs(off.R3) - 1.0, n ? std::arg(off.X3 / off.R3) * 180.0 / M_PI : 0.0,
                                on.periods, off.periods, per, ok ? "pass" : "FAIL");
                }
    }
    std::printf("\n(* = phase gated: |Dref| >= 0.1 S.)  %d cases, %d phase-gated increments.\n", n_cases, n_phase_gated);
    std::printf("worst |increment amplitude error| / S by harmonic n=0/1/3/10:  surface %.1e %.1e %.1e %.1e"
                "  node3 %.1e %.1e %.1e %.1e\n", worst[0][0], worst[0][1], worst[0][2], worst[0][3],
                worst[1][0], worst[1][1], worst[1][2], worst[1][3]);
    std::printf(fails ? "\nRESULT: FAIL (%d balanced-split cases)\n" : "\nRESULT: PASS (all balanced-split cases)\n", fails);
    return fails ? 1 : 0;
}
