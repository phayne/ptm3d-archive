// Unit tests for --ice-fraction-from-albedo=hapke-intimate (hapke_mixing.h,
// the exposed-area ice layer and ODE, and the Hapke forward albedo model).
// Self-contained; exit code 0 = all checks pass.
//
// Optional argument: path to a per-face Bond albedo .bin (size_t count +
// double[count]); if given, the forward/initialization match is also checked on
// every in-range value of that map.

#include "hapke_mixing.h"
#include "composition_albedo.h"
#include "ice_layer_model.h"
#include "multitimescale_evolution.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <random>
#include <string>
#include <vector>

using namespace ptm::exosphere;

static int g_fail = 0, g_total = 0;
static void check(bool ok, const std::string& name, double value, double limit) {
    ++g_total;
    if (!ok) ++g_fail;
    std::printf("[%s] %-72s value=%.3e limit=%.1e\n", ok ? "PASS" : "FAIL", name.c_str(), value, limit);
}

// Literal copy of the engine's closed form (ptm_thermal_sim.cpp hapkeWFromA0 +
// applyIncidenceAlbedo), kept verbatim so a change on either side is caught.
static double engineWFromA0(double A0) {
    const double gamma = (1.0 - A0) / (1.0 + 2.0 * A0);
    return 1.0 - gamma * gamma;
}
static double engineAh(double w, double mu0) {
    const double gamma = std::sqrt(std::max(0.0, 1.0 - w));
    const double H = (1.0 + 2.0 * mu0) / (1.0 + 2.0 * mu0 * gamma);
    return 1.0 - gamma * H;
}

// Legacy volume-fraction solution (multitimescale_evolution.h, pre-mode branch).
static double legacyConc(double c0, double A, double B, double dt) {
    const double ceq = 1.0 - B / A;
    return ceq - (ceq - c0) * std::exp(-A * dt);
}

// RK4 reference for dphi/dt = B - A f(phi), clamped at phi = 1.
static void rk4(double phi0, double A, double B, double r, double dt, int nsteps,
                double& phi_end, double& mean_f) {
    auto f = [r](double p) { p = std::clamp(p, 0.0, 1.0); return p / (p + r * (1.0 - p)); };
    auto rhs = [&](double p) { return B - A * f(p); };
    double p = phi0, h = dt / nsteps, integ = 0.0;
    for (int k = 0; k < nsteps; ++k) {
        const double k1 = rhs(p), k2 = rhs(p + 0.5 * h * k1), k3 = rhs(p + 0.5 * h * k2), k4 = rhs(p + h * k3);
        const double f1 = f(p), f2 = f(p + 0.5 * h * k1), f3 = f(p + 0.5 * h * k2), f4 = f(p + h * k3);
        integ += h * (f1 + 2 * f2 + 2 * f3 + f4) / 6.0;
        p += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0;
        if (p > 1.0) p = 1.0;
    }
    phi_end = p;
    mean_f = integ / dt;
}

int main(int argc, char** argv) {
    // 1. Gauss-Legendre 64 on [0,1]: exact for polynomials up to degree 127.
    {
        const auto& gl = hapke::gaussLegendre64();
        double err = 0.0;
        for (int k = 0; k <= 127; k += 7) {
            double s = 0.0;
            for (int i = 0; i < 64; ++i) s += gl.wt[i] * std::pow(gl.mu[i], k);
            err = std::max(err, std::abs(s - 1.0 / (k + 1)));
        }
        check(err < 1e-13, "GL64 integrates mu^k exactly (k <= 127)", err, 1e-13);
    }

    // 2. A_h matches the engine closed form; A_h(1; w_engine(A0)) = A0.
    {
        double err_ah = 0.0, err_inv = 0.0;
        for (double A0 = 0.02; A0 < 0.99; A0 += 0.01) {
            const double w = engineWFromA0(A0);
            err_inv = std::max(err_inv, std::abs(hapke::directionalHemisphericAlbedo(w, 1.0) - A0));
            for (double mu = 0.0; mu <= 1.0; mu += 0.05) {
                err_ah = std::max(err_ah, std::abs(hapke::directionalHemisphericAlbedo(w, mu) - engineAh(w, mu)));
            }
        }
        check(err_ah == 0.0, "A_h(w,mu0) bitwise equal to engine applyIncidenceAlbedo form", err_ah, 0.0);
        check(err_inv < 1e-14, "A_h(1; engine hapkeWFromA0(A0)) == A0 (engine closed-form inverse)", err_inv, 1e-14);
    }

    // 3. Bond albedo: 64-point quadrature vs exact integral.
    {
        double err = 0.0;
        for (double w = 0.0; w <= 0.999; w += 0.001) {
            err = std::max(err, std::abs(hapke::sphericalAlbedo(w) - hapke::sphericalAlbedoAnalytic(w)));
        }
        check(err < 1e-12, "A_B(w): GL64 vs exact integral, w in [0, 0.999]", err, 1e-12);
    }

    // 4. Inverse of A_B, isotropic and anisotropic (similarity relation).
    {
        double err_w = 0.0, err_A = 0.0;
        for (double w = 0.001; w <= 0.999; w += 0.007) {
            err_w = std::max(err_w, std::abs(hapke::invertSphericalAlbedo(hapke::sphericalAlbedo(w)) - w));
        }
        for (double A = 0.005; A <= 0.95; A += 0.005) {
            for (double g : {-0.3, 0.0, 0.3, 0.6}) {
                const double w = hapke::singleScatteringAlbedoFromBond(A, g);
                err_A = std::max(err_A, std::abs(hapke::bondAlbedo(w, g) - A));
            }
        }
        check(err_w < 1e-10, "invert A_B: w -> A_B(w) -> w round trip", err_w, 1e-10);
        check(err_A < 1e-12, "endmember w from Bond albedo, g in {-0.3,0,0.3,0.6}", err_A, 1e-12);
        // The Bond-anchored w is NOT the engine's normal-incidence-anchored w.
        const double w_bond = hapke::singleScatteringAlbedoFromBond(0.5, 0.0);
        const double w_eng = engineWFromA0(0.5);
        check(std::abs(w_bond - w_eng) > 1e-3, "Bond-anchored w differs from engine A_h(1)-anchored w (A=0.5)",
              std::abs(w_bond - w_eng), 1e-3);
    }

    // 5. Mixture monotone in f; bisection inverts it; endpoints are the endmembers.
    {
        double min_slope = 1e300, err_inv = 0.0, err_end = 0.0;
        bool monotone = true;
        const double g_pairs[][2] = {{0, 0}, {0.5, 0}, {0, 0.5}, {-0.3, -0.3}, {0.7, 0.2}};
        for (double Ai : {0.70, 0.80, 0.90}) {
            for (double An : {0.01, 0.012, 0.02, 0.04, 0.10}) {
                for (const auto& gp : g_pairs) {
                    const auto m = hapke::makeIntimateMixture(Ai, gp[0], An, gp[1]);
                    err_end = std::max(err_end, std::max(std::abs(m.bond(0.0) - An), std::abs(m.bond(1.0) - Ai)));
                    double prev = m.bond(0.0);
                    for (int k = 1; k <= 2000; ++k) {
                        const double f = k / 2000.0, a = m.bond(f);
                        if (!(a > prev)) monotone = false;
                        min_slope = std::min(min_slope, (a - prev) * 2000.0);
                        prev = a;
                    }
                    for (int k = 1; k < 100; ++k) {
                        const double f = k / 100.0;
                        int cl = 9;
                        err_inv = std::max(err_inv, std::abs(m.invert(m.bond(f), &cl) - f));
                        if (cl != 0) monotone = false;
                    }
                    int c0 = 0, c1 = 0;
                    if (m.invert(An * 0.5, &c0) != 0.0 || c0 != -1) monotone = false;
                    if (m.invert(std::min(1.0, Ai + 0.05), &c1) != 1.0 || c1 != +1) monotone = false;
                }
            }
        }
        check(monotone, "A_B(mixture) strictly increasing in f; clamps flagged", monotone ? 0.0 : 1.0, 0.0);
        check(min_slope > 0.0, "min dA/df over all parameter sets > 0", min_slope, 0.0);
        check(err_inv < 1e-12, "bisection f(A) inverts the mixture", err_inv, 1e-12);
        check(err_end < 1e-12, "mixture endpoints reproduce endmember Bond albedos", err_end, 1e-12);
    }

    // 6. Area <-> volume fraction.
    {
        double err = 0.0;
        for (double r : {0.1, 1.0, 3.0, 10.0}) {
            for (double f = 0.0; f <= 1.0; f += 0.01) {
                err = std::max(err, std::abs(hapke::exposedAreaFromVolume(hapke::volumeFromExposedArea(f, r), r) - f));
            }
        }
        const double phi = hapke::volumeFromExposedArea(0.5, 10.0);
        check(err < 1e-15, "f -> phi -> f round trip", err, 1e-15);
        check(std::abs(phi - 10.0 / 11.0) < 1e-15, "phi(f=0.5, r=10) = 10/11", std::abs(phi - 10.0 / 11.0), 1e-15);
    }

    // 7. Forward model reproduces the map from the initialization (synthetic + optional real map).
    auto forward_gap = [](const std::vector<double>& A_map, double Ai, double An, double gi, double gn,
                          double r, size_t& n_used) {
        CompositionAlbedoConfig cfg;
        cfg.hapke_intimate = true;
        cfg.hapke_mixture = hapke::makeIntimateMixture(Ai, gi, An, gn);
        cfg.grain_ratio = r;
        CompositionAlbedoMapper fwd(cfg);
        double gap = 0.0;
        n_used = 0;
        for (size_t i = 0; i < A_map.size(); ++i) {
            int cl = 0;
            const double f = cfg.hapke_mixture.invert(A_map[i], &cl);
            if (cl != 0) continue;
            const double c = 1.0 - hapke::volumeFromExposedArea(f, r);
            gap = std::max(gap, std::abs(fwd.computeAlbedo(c, i) - A_map[i]));
            ++n_used;
        }
        return gap;
    };
    {
        std::mt19937_64 rng(12345);
        std::uniform_real_distribution<double> U(0.0, 1.0);
        std::vector<double> A(100000);
        for (auto& a : A) a = U(rng);
        size_t n1 = 0, n2 = 0;
        const double g1 = forward_gap(A, 0.80, 0.012, 0.0, 0.0, 10.0, n1);
        const double g2 = forward_gap(A, 0.90, 0.04, 0.5, -0.2, 3.0, n2);
        check(g1 < 1e-6 && n1 > 50000, "forward(c_init) == A_map, base parameters (100k synthetic)", g1, 1e-6);
        check(g2 < 1e-6 && n2 > 50000, "forward(c_init) == A_map, g != 0, r = 3 (100k synthetic)", g2, 1e-6);
        // The legacy linear forward model does NOT reproduce the map (why hapke mode has its own).
        CompositionAlbedoMapper lin(CompositionAlbedoConfig(0.80, 0.012));
        const auto m = hapke::makeIntimateMixture(0.80, 0.0, 0.012, 0.0);
        const double f = m.invert(0.41);
        const double c = 1.0 - hapke::volumeFromExposedArea(f, 10.0);
        const double jump = std::abs(lin.computeAlbedo(c) - 0.41);
        check(jump > 0.05, "legacy linear forward model misses the map at A=0.41 (documented)", jump, 0.05);
    }
    if (argc > 1) {
        std::ifstream in(argv[1], std::ios::binary);
        uint64_t n = 0;
        in.read(reinterpret_cast<char*>(&n), sizeof(n));
        std::vector<double> A(n);
        in.read(reinterpret_cast<char*>(A.data()), static_cast<std::streamsize>(n * sizeof(double)));
        size_t used = 0;
        const double gap = in ? forward_gap(A, 0.80, 0.012, 0.0, 0.0, 10.0, used) : 1.0;
        check(in && gap < 1e-6, "forward(c_init) == A_map on " + std::string(argv[1]).substr(std::string(argv[1]).find_last_of('/') + 1)
              + " (" + std::to_string(used) + " faces)", gap, 1e-6);
    }

    // 8. Ice layer availability: default is bitwise (1 - c); area mode gives f.
    {
        IceLayerModel model(3);
        IceLayerState st(3);
        model.initializeFromConcentrations(st, {0.05, 0.5, 1.0});
        bool legacy_ok = model.getIceAvailabilityFactor(st, 0) == 1.0 - 0.05 &&
                         model.getIceAvailabilityFactor(st, 1) == 1.0 - 0.5 &&
                         model.getIceAvailabilityFactor(st, 2) == 0.0 && !model.usesExposedAreaFraction();
        check(legacy_ok, "default availability is exactly (1 - c)", legacy_ok ? 0.0 : 1.0, 0.0);
        model.setExposedAreaGrainRatio(10.0);
        const double f1 = model.getIceAvailabilityFactor(st, 1);   // phi = 0.5 -> 0.5/5.5
        check(std::abs(f1 - 0.5 / 5.5) < 1e-15, "area-mode availability f(phi=0.5, r=10) = 1/11",
              std::abs(f1 - 0.5 / 5.5), 1e-15);
    }

    // 9. Exposed-area ODE: r = 1 equals the legacy solution; r in {0.3, 0.5, 10} equals RK4,
    //    including D = B (r - 1) + A = 0 (r = 0.5, B = 2A) and D < 0 (r = 0.5, B = 3A).
    {
        double err_r1 = 0.0, err_rk_phi = 0.0, err_rk_f = 0.0;
        bool in_range = true;
        const double As[] = {1e-12, 3e-11, 1e-9};
        const double Bfrac[] = {0.0, 0.2, 0.9, 1.0, 1.5, 2.0, 3.0};
        const double phis[] = {0.02, 0.5, 0.95, 1.0};
        const double dts[] = {3.07e7, 3.07e9, 3.07e11};
        for (double A : As) for (double bf : Bfrac) for (double phi0 : phis) for (double dt : dts) {
            const double B = bf * A;
            const auto s1 = solveExposedAreaStep(phi0, A, B, 1.0, dt);
            double c_leg = std::clamp(legacyConc(1.0 - phi0, A, B, dt), 0.0, 1.0);
            err_r1 = std::max(err_r1, std::abs((1.0 - s1.phi_end) - c_leg));
            for (double r : {0.3, 0.5, 10.0}) {
                const auto st = solveExposedAreaStep(phi0, A, B, r, dt);
                if (!(st.phi_end >= 0.0 && st.phi_end <= 1.0 && st.mean_f >= 0.0 && st.mean_f <= 1.0)) in_range = false;
                if (A * dt <= 30.0) {
                    double pr, fr;
                    rk4(phi0, A, B, r, dt, 20000, pr, fr);
                    err_rk_phi = std::max(err_rk_phi, std::abs(st.phi_end - pr));
                    err_rk_f = std::max(err_rk_f, std::abs(st.mean_f - fr));
                }
            }
        }
        check(err_r1 < 1e-9, "exposed-area ODE with r = 1 reproduces legacy closed form", err_r1, 1e-9);
        check(in_range, "exposed-area ODE: phi_end and mean f stay in [0, 1] (r < 1, D <= 0 included)",
              in_range ? 0.0 : 1.0, 0.0);
        check(err_rk_phi < 1e-7, "exposed-area ODE r in {0.3, 0.5, 10}: phi_end vs RK4", err_rk_phi, 1e-7);
        check(err_rk_f < 1e-7, "exposed-area ODE r in {0.3, 0.5, 10}: time-mean f vs RK4", err_rk_f, 1e-7);
        // Conditioning: a 1e-14 relative change in deposition (transport-sum jitter)
        // must not move the time-mean f by more than 1e-10 (the B t - dphi form alone
        // gives 6e-10 on this grid, and float32-visible run-to-run jitter in smoke runs).
        double worst = 0.0;
        for (double A : {1e-16, 1e-14, 1e-12, 3e-11, 1e-9})
        for (double bf : {0.01, 0.2, 0.9, 1.0, 1.5, 1e2, 1e4}) for (double phi0 : phis) for (double dt : dts)
            for (double r : {0.5, 10.0}) {
                const double B = bf * A;
                const auto a = solveExposedAreaStep(phi0, A, B, r, dt);
                const auto b = solveExposedAreaStep(phi0, A, B * (1.0 + 1e-14), r, dt);
                worst = std::max(worst, std::abs(a.mean_f - b.mean_f));
            }
        check(worst < 1e-10, "exposed-area ODE: mean f well conditioned in deposition (dB/B = 1e-14)", worst, 1e-10);
        // Reviewer's case: r = 0.5, B = 3A (D < 0) must GAIN ice, not wipe it.
        const auto sr = solveExposedAreaStep(0.5, 1e-12, 3e-12, 0.5, 1e6);
        double pr, fr;
        rk4(0.5, 1e-12, 3e-12, 0.5, 1e6, 20000, pr, fr);
        check(std::abs(sr.phi_end - pr) < 1e-12, "r = 0.5, B = 3A, D < 0: phi_end matches RK4 (was -1.25)",
              std::abs(sr.phi_end - pr), 1e-12);
    }

    // 10. Owed row 1: the initial ice availability governs the first-epoch rate.
    //     Diurnal averages are scaled by availability before caching; the cache
    //     keeps the pure-ice potential. First-epoch sublimation must be
    //     availability x pure (was: the pure-ice rate for any composition).
    {
        const double P = 2.0e-9, area = 1.0e6, dt = 3.07e3;   // A dt ~ 1e-7: no depletion within the step
        for (double r : {0.0, 10.0}) {
            IceLayerModel model(2);
            IceLayerState st(2);
            model.initializeFromConcentrations(st, {0.05, 0.40});
            model.setExposedAreaGrainRatio(r);
            MultiTimescaleEvolution evo(2);
            auto& d = evo.getMutableDiurnalAverages();
            std::vector<double> pure = {P, P}, sput = {0.0, 0.0}, areas = {area, area};
            double expect = 0.0;
            for (int i = 0; i < 2; ++i) {
                const double av = model.getIceAvailabilityFactor(st, i);
                d.avg_sublimation_flux[i] = P * av;
                d.avg_deposition_flux[i] = 0.0;
                expect += P * av * area;
            }
            evo.cacheBaseFluxesUnscaled(model, st, pure, sput, areas, 0);
            evo.evolveEpoch(model, st, areas, dt);
            const double got = evo.getStatistics().sublimation_rate;
            check(std::abs(got / expect - 1.0) < 1e-6,
                  std::string("first-epoch sublimation = availability x pure-ice potential (") +
                  (r > 0 ? "exposed-area" : "volume") + " mode)", std::abs(got / expect - 1.0), 1e-6);
        }
    }

    // 11. Owed row 7: a face that starts with NO ice (c = 1, f = 0) keeps its
    //     pure-ice potential in the cache and, once deposition gives it ice,
    //     develops nonzero sublimation.
    {
        const double P = 2.0e-9, dep = 1.0e-10, area = 1.0e6, epoch = 3.07e7;
        for (double r : {0.0, 10.0}) {
            IceLayerModel model(1);
            IceLayerState st(1);
            model.initializeFromConcentrations(st, {1.0});
            model.setExposedAreaGrainRatio(r);
            MultiTimescaleEvolution evo(1);
            auto& d = evo.getMutableDiurnalAverages();
            std::vector<double> pure = {P}, sput = {0.0}, areas = {area};
            d.avg_sublimation_flux[0] = P * model.getIceAvailabilityFactor(st, 0);   // = 0
            d.avg_deposition_flux[0] = dep;
            evo.cacheBaseFluxesUnscaled(model, st, pure, sput, areas, 0);
            evo.evolveEpoch(model, st, areas, epoch);
            const double c1 = st.nonice_concentration[0];
            const double sub1 = evo.getStatistics().sublimation_rate;
            evo.evolveEpoch(model, st, areas, epoch);
            const double sub2 = evo.getStatistics().sublimation_rate;
            const std::string m = (r > 0 ? "exposed-area" : "volume");
            check(c1 < 1.0 && sub1 > 0.0 && sub2 > sub1,
                  "f = 0 face with deposition gains ice and starts sublimating (" + m + " mode)",
                  sub2, 0.0);
            // It relaxes toward the equilibrium where exposed-ice sublimation balances deposition.
            // relaxation time ~ M/P ~ 2e10 s; step well past it
            for (int k = 0; k < 50; ++k) evo.evolveEpoch(model, st, areas, 1.0e12);
            const double bal = evo.getStatistics().sublimation_rate / (dep * area);
            check(std::abs(bal - 1.0) < 1e-3, "... and reaches sublimation = deposition (" + m + " mode)",
                  std::abs(bal - 1.0), 1e-3);
        }
    }

    std::printf("\n%d / %d checks passed\n", g_total - g_fail, g_total);
    return g_fail == 0 ? 0 : 1;
}
