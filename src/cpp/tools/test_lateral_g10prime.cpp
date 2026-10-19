// test_lateral_g10prime.cpp -- G10' driven-response gate for C1c lateral conduction
// (Fermi addendum 2026-10-18, LATERAL_SIGNOFF_FERMI.md).
//
// Runs the PRODUCTION CPU SIMD column solver (thermal::ThermalBatchSoA: shared surface Newton,
// CN interior, bottom BC, n_sub = ceil(dt / (day/768)) substeps with the driver's midpoint flux
// interpolation and the forcing-epoch surface re-evaluation) inside the planned engine split:
//     half lateral (dt/2) | CN substeps over dt | half lateral (dt/2),  bottom BC re-applied after
//     each lateral half, lateral on capacitive nodes 1..n-2 only (Fermi R2 + addendum 2).
// Geometry: a 1-D chain of equal equilateral facets (Neumann ends), so the lateral mode
// x_j = cos(pi (j + 1/2) / N) is an EXACT eigenvector of the discrete lateral operator; the
// chain is sized so the mode's half-wavelength is the requested Lambda/2. This isolates the
// TIME and SPLITTING error (G2/G3 cover the spatial operator).
// Base state: uniform T0 (60 or 110 K) with Qb = 0 and F_bar = eps sigma T0^4 (an exact steady
// state of the column). Forcing perturbation on the q-mode only (the uniform part of Fermi's
// F0[1 + 0.5 cos qx] is orthogonal to the q-mode in the linear response and is omitted so the
// deep column need not re-equilibrate):  dF = a * 0.5 x_j * sin(n w t), a = 1e-6 F_bar.
// Reference: the exact periodic solution of the LINEARIZED semi-discrete system (same z-grid,
// same nodal coefficients, same surface two-point balance, lateral eigenvalue mu_i per node),
// solved in the frequency domain (complex tridiagonal). n = 0: steady state.
// Bars (Fermi): n=0 contrast rel err <= 1e-6; n=1 amp 1% / phase 0.5 deg; n=3 2% / 1 deg;
// n=10 5% / 3 deg. Evaluated on the surface temperature and node 3. TR-BDF2 is gated; BE reported.
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
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
    // Thomas
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

struct Case { double T0, Lambda; int harm; };
struct Outcome { double amp_s, ph_s, amp_3, ph_3; bool ok; };

static int envInt(const char* n, int d) { const char* e = std::getenv(n); return (e && *e) ? std::atoi(e) : d; }

static Outcome runCase(const Case& cs, lat::Scheme scheme, bool verbose) {
    const bool nolat = envInt("G10P_NOLAT", 0) != 0;
    const int dtdiv = std::max(1, envInt("G10P_DTDIV", 1));
    thermal::Planet planet("Moon");
    planet.Qb = 0.0;
    thermal::Config config;
    thermal::Profile ref(planet, 0.0, config, false);
    const int nl = static_cast<int>(ref.nlayers);
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
    const double amp = 1e-6 * Fbar;
    const double P = 29.530588 * 86400.0, dt = P / (360.0 * dtdiv);
    const double omega = 2.0 * M_PI * cs.harm / P;
    const int n_sub = std::max(1, static_cast<int>(std::ceil(dt / (planet.day / 768.0))));
    const double dt_sub = dt / n_sub;

    // lateral eigenvalue per capacitive node (exact eigenvector): mu_i = (L x)_j / (C_j x_j)
    lat::Workspace ws;
    std::vector<double> kk(N), rc(N), Tn(N), mu(nl, 0.0);
    auto gather = [&](int l) {
        for (int j = 0; j < N; ++j) {
            const auto& b = batches[j / BS];
            const int f = j % BS;
            kk[j] = b.k[l * BS + f];
            rc[j] = b.rho[l * BS + f] * b.cp[l * BS + f];
            Tn[j] = b.T[l * BS + f];
        }
    };
    for (int l = 1; l <= nl - 2; ++l) {
        gather(l);
        const double h = 0.5 * (ref.z()[l + 1] - ref.z()[l - 1]);
        std::vector<double> G;
        lat::assembleConductance(ch.adj, nullptr, kk.data(), h, G);
        double Lx = 0.0;   // at j = 0
        for (int e = 0; e < 3; ++e) {
            const int32_t nb2 = ch.adj.nbr[e];
            if (nb2 >= 0) Lx += G[e] * (ch.x[0] - ch.x[nb2]);
        }
        mu[l] = nolat ? 0.0 : Lx / (rc[0] * ch.adj.area[0] * h * ch.x[0]);
    }
    const cd F_hat = (cs.harm == 0) ? cd(0.5 * amp, 0.0) : cd(0.0, -0.5 * amp);   // sin -> Re[-i e^{iwt}]
    const Result R = reference(batches[0], mu, eps, cs.T0, omega, F_hat);

    lat::SolverOptions opt;
    opt.scheme = scheme;
    opt.rel_tol = 1e-13;
    opt.max_iter = 1000;
    auto lateralHalf = [&](double h_dt) {
        if (nolat) return;
        for (auto& b : batches) thermal::updateMaterialPropertiesBatchSIMD(b);   // k, rho c at the half-step start
        for (int l = 1; l <= nl - 2; ++l) {
            gather(l);
            const double h = 0.5 * (ref.z()[l + 1] - ref.z()[l - 1]);
            lat::StepStats st = lat::backwardEulerStep(ch.adj, nullptr, Tn.data(), kk.data(), rc.data(), h, h_dt, opt, ws);
            if (st.capped) { std::fprintf(stderr, "FATAL: lateral CG capped (layer %d)\n", l); std::exit(1); }
            for (int j = 0; j < N; ++j) batches[j / BS].T[l * BS + j % BS] = Tn[j];
        }
        for (auto& b : batches) thermal::applyBottomBCBatch(b);
    };
    auto flux_at = [&](double t, int j) {
        const double s = (cs.harm == 0) ? 1.0 : std::sin(omega * t);
        return Fbar + amp * 0.5 * ch.x[j] * s;
    };
    // slowest relaxation: min over capacitive nodes of mu_i; spin-up >= 60 e-folds, whole periods
    double mu_min = nolat ? 1e-6 : 1e300;
    if (!nolat) for (int l = 1; l <= nl - 2; ++l) mu_min = std::min(mu_min, mu[l]);
    const double t_spin = 60.0 / mu_min;
    const int steps_per_period = ((cs.harm == 0) ? 360 : 360 / cs.harm) * dtdiv;
    int n_periods = std::max(3, static_cast<int>(std::ceil(t_spin / (steps_per_period * dt)))) + 2;
    const long n_steps = static_cast<long>(n_periods) * steps_per_period;
    std::vector<double> qs(steps_per_period), q3(steps_per_period), qs_prev(steps_per_period);
    double fl_a[BS], fl_b[BS], fl[BS];
    // G10P_SPLIT=balanced: steady-state-preserving (Douglas / balanced) split per CN substep tau:
    //   (1) vertical CN step (the batch's own sequence) with the lateral flux L T^n added as an
    //       explicit source on capacitive nodes -> T*;
    //   (2) remove that explicit source and take one implicit BE lateral step:
    //       (C + tau L) T^{n+1} = C T* + tau L T^n.
    //   At a steady state both stages vanish identically, so the steady state is EXACT.
    const bool balanced = [] { const char* e = std::getenv("G10P_SPLIT"); return e && std::string(e) == "balanced"; }();
    std::vector<std::vector<double>> srcL(nl), kkL(nl), rcL(nl);
    lat::SolverOptions optBE = opt;
    optBE.scheme = lat::Scheme::BackwardEuler;
    auto balancedSubstep = [&](double tau, const double* flA, const double* flB, double al) {
        for (auto& b : batches) thermal::updateMaterialPropertiesBatchSIMD(b);
        for (int l = 1; l <= nl - 2; ++l) {
            gather(l);
            const double h = 0.5 * (ref.z()[l + 1] - ref.z()[l - 1]);
            std::vector<double> G;
            lat::assembleConductance(ch.adj, nullptr, kk.data(), h, G);
            srcL[l].assign(N, 0.0);
            for (int j = 0; j < N; ++j) {
                double q = 0.0;
                for (int e = 0; e < 3; ++e) {
                    const int32_t jj = ch.adj.nbr[3 * j + e];
                    if (jj >= 0 && G[3 * j + e] != 0.0) q += G[3 * j + e] * (Tn[jj] - Tn[j]);
                }
                srcL[l][j] = tau * q / (rc[j] * ch.adj.area[j] * h);   // explicit lateral increment, K
            }
            kkL[l] = kk; rcL[l] = rc;
        }
        for (int bi = 0; bi < nb; ++bi) {
            auto& b = batches[bi];
            double fl2[BS];
            for (int f = 0; f < BS; ++f) fl2[f] = (1.0 - al) * flA[bi * BS + f] + al * flB[bi * BS + f];
            b.setFluxContiguous(fl2, BS);
            thermal::updateMaterialPropertiesBatchSIMD(b);
            thermal::applySurfaceBCBatch(b, tau);
            thermal::buildTridiagonalSystemBatch(b, tau);
            for (int l = 1; l <= nl - 2; ++l)
                for (int f = 0; f < BS; ++f) b.d_coef[(l - 1) * BS + f] += srcL[l][bi * BS + f];
            thermal::solveThomasBatchSIMD(b);
            thermal::applyBottomBCBatch(b);
        }
        for (int l = 1; l <= nl - 2; ++l) {
            gather(l);
            const double h = 0.5 * (ref.z()[l + 1] - ref.z()[l - 1]);
            for (int j = 0; j < N; ++j) Tn[j] -= srcL[l][j];
            lat::StepStats st = lat::backwardEulerStep(ch.adj, nullptr, Tn.data(), kkL[l].data(), rcL[l].data(), h, tau, optBE, ws);
            if (st.capped) { std::fprintf(stderr, "FATAL: lateral CG capped (layer %d)\n", l); std::exit(1); }
            for (int j = 0; j < N; ++j) batches[j / BS].T[l * BS + j % BS] = Tn[j];
        }
        for (auto& b : batches) thermal::applyBottomBCBatch(b);
    };
    std::vector<double> flA(N), flB(N);
    for (long s = 0; s < n_steps; ++s) {
        const double t0 = s * dt, t1 = (s + 1) * dt;
        if (balanced) {
            for (int j = 0; j < N; ++j) { flA[j] = flux_at(t0, j); flB[j] = flux_at(t1, j); }
            for (int k = 0; k < n_sub; ++k) balancedSubstep(dt_sub, flA.data(), flB.data(), (k + 0.5) / n_sub);
        } else {
        lateralHalf(0.5 * dt);
        for (int bi = 0; bi < nb; ++bi) {
            auto& b = batches[bi];
            for (int f = 0; f < BS; ++f) { fl_a[f] = flux_at(t0, bi * BS + f); fl_b[f] = flux_at(t1, bi * BS + f); }
            for (int k = 0; k < n_sub; ++k) {
                const double al = (k + 0.5) / n_sub;
                for (int f = 0; f < BS; ++f) fl[f] = (1.0 - al) * fl_a[f] + al * fl_b[f];
                b.setFluxContiguous(fl, BS);
                b.advanceTimestep(dt_sub);
            }
        }
        lateralHalf(0.5 * dt);
        }
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
    cd Xs = 0.0, X3 = 0.0, Xp = 0.0;
    for (int k = 0; k < steps_per_period; ++k) {
        const double t = k * dt;   // phase of the sample within the period (t1 mod period)
        const cd e = (cs.harm == 0) ? cd(1.0, 0.0) : std::exp(cd(0.0, -omega * t));
        Xs += qs[k] * e; X3 += q3[k] * e; Xp += qs_prev[k] * e;
    }
    const double norm = (cs.harm == 0) ? 1.0 / steps_per_period : 2.0 / steps_per_period;
    Xs *= norm; X3 *= norm; Xp *= norm;
    const double periodic = std::abs(Xs - Xp) / std::abs(Xs);
    auto amp_err = [](cd a, cd b) { return std::abs(a) / std::abs(b) - 1.0; };
    auto ph_err = [](cd a, cd b) { return std::arg(a / b) * 180.0 / M_PI; };
    Outcome o{amp_err(Xs, R.surf), ph_err(Xs, R.surf), amp_err(X3, R.node3), ph_err(X3, R.node3), true};
    double bar_a = 0, bar_p = 0;
    if (cs.harm == 0) { bar_a = 1e-6; bar_p = 1e9; o.ph_s = o.ph_3 = 0.0; }
    else if (cs.harm == 1) { bar_a = 0.01; bar_p = 0.5; }
    else if (cs.harm == 3) { bar_a = 0.02; bar_p = 1.0; }
    else { bar_a = 0.05; bar_p = 3.0; }
    o.ok = std::fabs(o.amp_s) <= bar_a && std::fabs(o.amp_3) <= bar_a && std::fabs(o.ph_s) <= bar_p && std::fabs(o.ph_3) <= bar_p;
    if (verbose)
        std::printf("  %-6s T0=%3.0f K  L=%5.1f mm  n=%2d  mu_min*dt=%.2e  layers=%d n_sub=%d periods=%d (periodicity %.1e)"
                    "  surf amp %+.3e ph %+.3f deg | node3 amp %+.3e ph %+.3f deg  [%s]\n",
                    std::getenv("G10P_SPLIT") ? "BAL" : (scheme == lat::Scheme::TRBDF2 ? "TRBDF2" : "BE"), cs.T0, 1e3 * cs.Lambda, cs.harm, mu_min * dt, nl, n_sub,
                    n_periods, periodic, o.amp_s, o.ph_s, o.amp_3, o.ph_3, o.ok ? "pass" : "FAIL");
    return o;
}

int main() {
    setenv("PTM_SURFACE_BC_TOLK", "1e-13", 1);   // diagnostic override: the Newton tolerance must not enter a 1e-6 bar
    int fails = 0;
    for (lat::Scheme sc : {lat::Scheme::TRBDF2, lat::Scheme::BackwardEuler}) {
        const char* sp = std::getenv("G10P_SPLIT");
        const bool bal = sp && std::string(sp) == "balanced";
        if (bal && sc != lat::Scheme::TRBDF2) continue;   // balanced split always uses a BE lateral stage
        std::printf("G10' %s\n", bal ? "BALANCED split (Douglas, BE lateral stage per CN substep)"
                                       : (sc == lat::Scheme::TRBDF2 ? "Strang, TR-BDF2 halves (gated)" : "Strang, BE halves (reported)"));
        for (double T0 : {60.0, 110.0})
            for (double L : {1e-3, 3e-3, 1e-2, 3e-2, 1e-1})
                for (int n : {0, 1, 3, 10}) {
                    if (envInt("G10P_ONLY_N", -1) >= 0 && n != envInt("G10P_ONLY_N", -1)) continue;
                    if (envInt("G10P_ONLY_LMM", -1) >= 0 && std::lround(1e3 * L) != envInt("G10P_ONLY_LMM", -1)) continue;
                    if (envInt("G10P_ONLY_T", -1) >= 0 && std::lround(T0) != envInt("G10P_ONLY_T", -1)) continue;
                    const Outcome o = runCase({T0, L, n}, sc, true);
                    if (sc == lat::Scheme::TRBDF2 && !o.ok) ++fails;
                }
    }
    std::printf(fails ? "\nRESULT: FAIL (%d TR-BDF2 cases)\n" : "\nRESULT: PASS (all TR-BDF2 cases)\n", fails);
    return fails ? 1 : 0;
}
