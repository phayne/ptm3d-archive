// Unit tests for the lag-layer diffusion barrier (multiscale). Self-contained;
// exit code 0 = all checks pass.
//
//  (1) One face, no deposition, no sputtering: d + d^2/(2 l) = beta J0 f t to 1e-6,
//      in one step, in many steps, and through the multiscale engine (super-steps).
//  (2) With sputtering and deposition: solveLagFaceStep against an RK4 integration of
//      dd/dt = beta (J0 f/(1 + d/l) + Jp f - B); the equilibrium d* = l (u* - 1),
//      u* = J0 f / (B - Jp f), is approached and never crossed.
//  (3) Frost veneer first, then the barrier: the veneer sublimates unbarriered at
//      Js + Jp with d unchanged; once it is gone the barrier applies from d0.
//      Deposition above Js + Jp grows the veneer and never removes lag. Recycling
//      (E_S < B < Js + Jp): emission = B, state unchanged.
//  (4) Global mass conservation (64-face synthetic system, two source groups):
//      inventory change + (sublimated + sputtered - deposited) = 0 to rounding,
//      every step passes the super-step deposition audit, lag forms on the hot faces and a veneer
//      on the cold ones.
//  (5) t = 0 invariance: at d = 0 the availabilities, the albedo, and the scaled
//      fluxes and deposition are bit-identical to the legacy model.
//  (6) Barrier length (Knudsen, Schorghofer 2008 eq. 3), and the lag albedo limits.

#include "multitimescale_evolution.h"
#include "ice_layer_model.h"
#include "composition_albedo.h"
#include "hapke_mixing.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace exo = ptm::exosphere;

static int g_fail = 0, g_total = 0;
static void check(bool ok, const std::string& name, double value = 0.0, double limit = 0.0) {
    ++g_total;
    if (!ok) ++g_fail;
    std::printf("[%s] %-86s value=%.3e limit=%.1e\n", ok ? "PASS" : "FAIL", name.c_str(), value, limit);
    std::fflush(stdout);
}

static constexpr double kEpoch = 100.0 * exo::timescale_constants::EUROPA_DIURNAL_PERIOD;
static const double kRhoIce = exo::layer_constants::ICE_DENSITY;

static double betaOf(double xn, double phi) { return xn / ((1.0 - xn) * kRhoIce * (1.0 - phi)); }

// RK4 reference for the substrate regime: y = (d, sub emitted).
static void rk4(double Js, double Jp, double B, double f, double beta, double ell, double d0, double T,
                int K, double& d_out, double& sub_out) {
    auto rhs = [&](double d, double& dd, double& ds) {
        const double e = Js * f / (1.0 + d / ell);
        dd = beta * (e + Jp * f - B);
        ds = e;
    };
    double d = d0, s = 0.0;
    const double h = T / K;
    for (int k = 0; k < K; ++k) {
        double k1d, k1s, k2d, k2s, k3d, k3s, k4d, k4s;
        rhs(d, k1d, k1s);
        rhs(d + 0.5 * h * k1d, k2d, k2s);
        rhs(d + 0.5 * h * k2d, k3d, k3s);
        rhs(d + h * k3d, k4d, k4s);
        d += h / 6.0 * (k1d + 2 * k2d + 2 * k3d + k4d);
        s += h / 6.0 * (k1s + 2 * k2s + 2 * k3s + k4s);
    }
    d_out = d;
    sub_out = s;
}

// ---------------------------------------------------------------------------
static void test_analytic_one_face() {
    std::printf("\n(1) one face, no deposition: d + d^2/(2 l) = beta J0 f t\n");
    const double Js = 3e-10, f = 0.8, beta = betaOf(0.05, 0.6), ell = 4.23e-6, M = 45.0;
    double worst = 0.0, worst_mass = 0.0;
    for (double t : {1e3, 1e6, 1e8, 1e9, 1e10, 3e11, 3e12}) {
        const auto st = exo::solveLagFaceStep(Js, 0.0, 0.0, 0.0, 0.0, M, f, beta, ell, t);
        const double lhs = st.d_end + st.d_end * st.d_end / (2.0 * ell);
        worst = std::max(worst, std::abs(lhs / (beta * Js * f * t) - 1.0));
        // Mass: the lag is beta x the ice lost.
        worst_mass = std::max(worst_mass, std::abs(beta * st.sub_int / st.d_end - 1.0));
    }
    check(worst < 1e-6, "one step: (d + d^2/2l) / (beta J0 f t) - 1, t = 1e3 .. 3e12 s", worst, 1e-6);
    check(worst_mass < 1e-9, "one step: d = beta x (ice sublimated)", worst_mass, 1e-9);

    // Many steps compose to the same answer.
    double d = 0.0, Mi = M, sub = 0.0;
    const double T = 3e12;
    const int N = 1000;
    for (int k = 0; k < N; ++k) {
        const auto st = exo::solveLagFaceStep(Js, 0.0, 0.0, d, 0.0, Mi, f, beta, ell, T / N);
        d = st.d_end; Mi = st.M_end; sub += st.sub_int;
    }
    const double lhs = d + d * d / (2.0 * ell);
    check(std::abs(lhs / (beta * Js * f * T) - 1.0) < 1e-6, "1000 steps: (d + d^2/2l) / (beta J0 f t) - 1",
          std::abs(lhs / (beta * Js * f * T) - 1.0), 1e-6);
    check(std::abs(sub / (M - Mi) - 1.0) < 1e-9, "1000 steps: sublimated = ice lost",
          std::abs(sub / (M - Mi) - 1.0), 1e-9);

    // Through the engine: one face, no deposition, engine super-steps (computeSafeEpochCountLag).
    exo::IceLayerConfig cfg;
    exo::IceLayerModel model(1, cfg);
    model.setExposedAreaGrainRatio(10.0);
    exo::IceLayerState state(1);
    const double c0 = 0.05;
    model.initializeFromConcentrations(state, std::vector<double>{c0});
    exo::LagBarrierConfig lc;
    lc.enabled = true;
    lc.barrier_length = ell;
    model.enableLagBarrier(state, lc);
    exo::MultiTimescaleEvolution mte(1);
    std::vector<double> areas{1e6}, sub_v{Js}, sput_v{0.0};
    auto& da = mte.getMutableDiurnalAverages();
    da.avg_sublimation_flux[0] = Js * model.getSublimationAvailability(state, 0);
    da.avg_deposition_flux[0] = 0.0;
    mte.cacheBaseFluxesUnscaled(model, state, sub_v, sput_v, areas, 0);
    const int total = 100000;
    int e = 0, iters = 0;
    while (e < total) {
        mte.applyScaledFluxes(model, state, areas);
        mte.evolveEpoch(model, state, areas, kEpoch);
        int n = mte.computeSafeEpochCountLag(model, state, areas, kEpoch, total - e - 1 > 0 ? total - e - 1 : 1);
        int extra = std::min(n - 1, total - e - 1);
        if (extra > 0) mte.evolveEpoch(model, state, areas, kEpoch * extra);
        e += 1 + std::max(0, extra);
        ++iters;
    }
    const double fe = model.exposedAreaFraction(1.0 - c0);
    const double be = betaOf(c0, lc.lag_porosity);
    const double te = total * kEpoch;
    const double de = state.lag_thickness[0];
    const double rel = std::abs((de + de * de / (2.0 * ell)) / (be * Js * fe * te) - 1.0);
    check(rel < 1e-6, "engine, 1e5 epochs in " + std::to_string(iters) + " steps: (d + d^2/2l)/(beta J0 f t) - 1",
          rel, 1e-6);
    check(iters < 200, "engine: the lag super-step bound grows the steps (iterations < 200)", iters, 200);
    check(state.nonice_concentration[0] == c0, "engine: the mixture's non-ice fraction stays at x_n (no void)");
}

static void test_rk4_reference() {
    std::printf("\n(2) sputtering + deposition: closed form vs RK4; equilibrium approached, not crossed\n");
    const double f = 0.7, beta = betaOf(0.10, 0.6), ell = 5e-6, M = 40.0;
    struct Case { double Js, Jp, B, d0, T; const char* tag; };
    const Case cases[] = {
        {2e-10, 2e-11, 0.0, 0.0, 1e11, "sputter, no deposition"},
        {2e-10, 2e-11, 5e-11, 3e-6, 1e11, "sputter + deposition below the barriered loss"},
        {2e-10, 0.0, 1e-10, 0.0, 1e12, "deposition, approach to d*"},
        {1e-12, 3e-11, 1e-11, 1e-5, 1e11, "sputter-dominated (q > 0)"},
    };
    for (const auto& c : cases) {
        const auto st = exo::solveLagFaceStep(c.Js, c.Jp, c.B, c.d0, 0.0, M, f, beta, ell, c.T);
        double d_ref, s_ref;
        rk4(c.Js, c.Jp, c.B, f, beta, ell, c.d0, c.T, 200000, d_ref, s_ref);
        const double ed = std::abs(st.d_end - d_ref) / std::max(d_ref, 1e-30);
        const double es = std::abs(st.sub_int - s_ref) / std::max(s_ref, 1e-30);
        check(ed < 1e-8, std::string("d_end vs RK4: ") + c.tag, ed, 1e-8);
        check(es < 1e-8, std::string("sublimated vs RK4: ") + c.tag, es, 1e-8);
        const double booked = st.sub_int + st.sput_int - c.B * c.T;
        check(std::abs((M - st.M_end) - booked) <= 1e-12 * M, std::string("substrate change = booked net loss: ") + c.tag,
              std::abs((M - st.M_end) - booked), 1e-12 * M);
    }
    // Equilibrium: u* = a / (B - p).
    const double Js = 2e-10, B = 1e-10;
    const double d_star = ell * (Js * f / B - 1.0);
    const auto st = exo::solveLagFaceStep(Js, 0.0, B, 0.0, 0.0, M, f, beta, ell, 1e16);
    check(st.d_end < d_star && std::abs(st.d_end / d_star - 1.0) < 1e-6,
          "d -> d* = l (J0 f / B - 1) from below after 1e16 s", std::abs(st.d_end / d_star - 1.0), 1e-6);
    const auto s2 = exo::solveLagFaceStep(Js, 0.0, B, d_star, 0.0, M, f, beta, ell, 1e12);
    check(std::abs(s2.d_end - d_star) <= 1e-12 * d_star && std::abs(s2.sub_int - B * 1e12) <= 1e-9 * B * 1e12,
          "at d*: steady (emission = deposition, d fixed)", std::abs(s2.d_end - d_star) / d_star, 1e-12);
}

static void test_veneer_then_barrier() {
    std::printf("\n(3) frost veneer first, then the barrier\n");
    const double Js = 2e-10, Jp = 1e-11, f = 0.6, beta = betaOf(0.08, 0.6), ell = 4e-6, M = 30.0;
    const double d0 = 2e-6, mf0 = 1e-2, Jv = Js + Jp, B = 0.2 * Jv;
    const double tv = mf0 / (Jv - B);
    // Inside the veneer phase: unbarriered emission, d unchanged.
    {
        const auto st = exo::solveLagFaceStep(Js, Jp, B, d0, mf0, M, f, beta, ell, 0.5 * tv);
        check(st.d_end == d0 && st.M_end == M, "veneer phase: lag and substrate untouched");
        check(std::abs(st.mf_end - (mf0 - (Jv - B) * 0.5 * tv)) <= 1e-12 * mf0, "veneer phase: m_f falls at Js + Jp - B",
              std::abs(st.mf_end - (mf0 - (Jv - B) * 0.5 * tv)) / mf0, 1e-12);
        check(std::abs(st.sub_int / (Js * 0.5 * tv) - 1.0) < 1e-12 && std::abs(st.sput_int / (Jp * 0.5 * tv) - 1.0) < 1e-12,
              "veneer phase: sublimation and sputter unbarriered (Js, Jp)");
    }
    // Veneer gone after tv, then the barrier from d0 for 2 tv (B < E_S(d0) here).
    {
        const double ES0 = Js * f / (1.0 + d0 / ell) + Jp * f;
        check(B < ES0, "setup: after the veneer the face is a net source (B < E_S(d0))", B / ES0, 1.0);
        const auto st = exo::solveLagFaceStep(Js, Jp, B, d0, mf0, M, f, beta, ell, 3.0 * tv);
        double d_ref, s_ref;
        rk4(Js, Jp, B, f, beta, ell, d0, 2.0 * tv, 200000, d_ref, s_ref);
        check(st.mf_end == 0.0, "veneer consumed first (m_f = 0 at the end)");
        check(std::abs(st.d_end - d_ref) / (d_ref - d0) < 1e-7, "then the barrier: lag growth from d0 matches RK4",
              std::abs(st.d_end - d_ref) / (d_ref - d0), 1e-7);
        check(std::abs(st.sub_int - (Js * tv + s_ref)) / st.sub_int < 1e-8,
              "sublimated = Js t_v (veneer) + barriered integral", std::abs(st.sub_int - (Js * tv + s_ref)) / st.sub_int, 1e-8);
        // The barrier lowered the rate: mean sublimation after the veneer < unbarriered Js f.
        check(s_ref / (2.0 * tv) < Js * f / (1.0 + d0 / ell) * (1.0 + 1e-12), "barriered rate <= J0 f/(1 + d0/l)");
    }
    // Deposition never removes lag; above Js + Jp the veneer grows.
    {
        const double Bh = 3.0 * Jv, T = 1e9;
        const auto st = exo::solveLagFaceStep(Js, Jp, Bh, d0, 0.0, M, f, beta, ell, T);
        check(st.d_end == d0 && st.M_end == M, "deposition above Js + Jp: lag and substrate untouched");
        check(std::abs(st.mf_end - (Bh - Jv) * T) <= 1e-12 * (Bh - Jv) * T, "veneer grows at B - Js - Jp",
              std::abs(st.mf_end - (Bh - Jv) * T) / ((Bh - Jv) * T), 1e-12);
    }
    // Recycling: E_S < B < Js + Jp at m_f = 0: emission = B, nothing moves.
    {
        const double ES0 = Js * f / (1.0 + d0 / ell) + Jp * f;
        const double Bm = 0.5 * (ES0 + Jv), T = 1e10;
        const auto st = exo::solveLagFaceStep(Js, Jp, Bm, d0, 0.0, M, f, beta, ell, T);
        check(st.d_end == d0 && st.M_end == M && st.mf_end == 0.0 && st.regime_end == 1,
              "recycling: lag, substrate and veneer unchanged");
        check(std::abs((st.sub_int + st.sput_int) / (Bm * T) - 1.0) < 1e-12, "recycling: emission = deposition",
              std::abs((st.sub_int + st.sput_int) / (Bm * T) - 1.0), 1e-12);
        double as, ap;
        exo::lagInstantAvailability(Js, Jp, Bm, true, d0, 0.0, M, f, ell, as, ap);
        check(std::abs((Js * as + Jp * ap) / Bm - 1.0) < 1e-12, "recycling: instantaneous availability emits B",
              std::abs((Js * as + Jp * ap) / Bm - 1.0), 1e-12);
    }
    // Depletion: a thin substrate runs out mid-step; then the face recycles what lands.
    {
        const double Ms = 1e-3, Bs = 1e-12, T = 1e12;
        const auto st = exo::solveLagFaceStep(Js, Jp, Bs, 0.0, 0.0, Ms, f, beta, ell, T);
        check(st.M_end == 0.0, "thin substrate: depleted within the step");
        check(std::abs(st.d_end / (beta * Ms) - 1.0) < 1e-9, "depleted: d = beta x (all the substrate)",
              std::abs(st.d_end / (beta * Ms) - 1.0), 1e-9);
        check(std::abs((st.sub_int + st.sput_int) - (Ms + Bs * T)) <= 1e-9 * (Ms + Bs * T),
              "depleted: emitted = substrate + deposition", std::abs((st.sub_int + st.sput_int) - (Ms + Bs * T)), 1e-9);
    }
}

// ---------------------------------------------------------------------------
struct Sys {
    size_t n = 64;
    std::unique_ptr<exo::IceLayerModel> model;
    std::unique_ptr<exo::IceLayerState> state;
    std::unique_ptr<exo::MultiTimescaleEvolution> mte;
    std::vector<double> areas, sub, sput;
};

static Sys makeSys(bool lag, double ell) {
    Sys s;
    exo::IceLayerConfig cfg;
    cfg.thermal_skin_depth = 0.05;
    s.model = std::make_unique<exo::IceLayerModel>(s.n, cfg);
    s.model->setExposedAreaGrainRatio(10.0);
    s.state = std::make_unique<exo::IceLayerState>(s.n);
    std::vector<double> conc(s.n);
    s.areas.resize(s.n); s.sub.resize(s.n); s.sput.resize(s.n);
    for (size_t i = 0; i < s.n; ++i) {
        const bool hot = i < 16;
        s.areas[i] = 1.0e6 * (1.0 + 0.01 * static_cast<double>(i));
        conc[i] = hot ? 0.03 + 0.005 * static_cast<double>(i) : (i % 7 == 0 ? 0.0 : 0.05);
        s.sub[i] = hot ? 2e-9 * (1.0 + 0.1 * static_cast<double>(i)) : 1e-14;
        s.sput[i] = 0.02 * s.sub[i] + 3e-12;
    }
    s.model->initializeFromConcentrations(*s.state, conc);
    if (lag) {
        exo::LagBarrierConfig lc;
        lc.enabled = true;
        lc.barrier_length = ell;
        s.model->enableLagBarrier(*s.state, lc);
    }
    exo::MultiTimescaleConfig mc;
    s.mte = std::make_unique<exo::MultiTimescaleEvolution>(s.n, mc);
    const double share = 0.765;
    s.mte->setSputterPartition(share, 0.2208, 1.0 - share - 0.2208);
    auto& d = s.mte->getMutableDiurnalAverages();
    double S = 0.0, P = 0.0, w_hot = 0.0, w_cold = 0.0;
    for (size_t i = 0; i < s.n; ++i) {
        const double f = s.model->getIceAvailabilityFactor(*s.state, i);
        S += s.sub[i] * f * s.areas[i];
        P += s.sput[i] * f * s.areas[i];
        (i < 16 ? w_hot : w_cold) += s.areas[i];
    }
    const double rho = 1.0 - 1e-3;
    for (size_t i = 0; i < s.n; ++i) {
        const double share_i = (i < 16) ? 0.05 / w_hot : 0.95 / w_cold;
        const double f = s.model->getIceAvailabilityFactor(*s.state, i);
        d.avg_sublimation_flux[i] = s.sub[i] * f;
        d.avg_sputtering_flux[i] = s.sput[i] * f;
        d.avg_deposition_sublimation_flux[i] = rho * S * share_i;
        d.avg_deposition_sputter_flux[i] = share * P * share_i;
        d.avg_deposition_flux[i] = d.avg_deposition_sublimation_flux[i] + d.avg_deposition_sputter_flux[i];
    }
    s.mte->cacheBaseFluxesUnscaled(*s.model, *s.state, s.sub, s.sput, s.areas, 0);
    return s;
}

static void test_global_mass() {
    std::printf("\n(4) global mass conservation with lag and veneer (64 faces, two source groups)\n");
    for (double ell : {5e-6, 5e-5}) {
        Sys s = makeSys(true, ell);
        std::string err;
        const int total = 20000;
        int e = 0;
        try {
            while (e < total) {
                s.mte->applyScaledFluxes(*s.model, *s.state, s.areas);
                s.mte->evolveEpoch(*s.model, *s.state, s.areas, kEpoch);
                const int rem = total - e - 1;
                int extra = 0;
                if (rem > 0) extra = std::min(rem, s.mte->computeSafeEpochCountLag(*s.model, *s.state, s.areas, kEpoch, rem) - 1);
                if (extra > 0) s.mte->evolveEpoch(*s.model, *s.state, s.areas, kEpoch * extra);
                e += 1 + std::max(0, extra);
            }
        } catch (const std::exception& x) { err = x.what(); }
        const std::string tag = " (l = " + std::to_string(ell * 1e6).substr(0, 4) + " um)";
        check(err.empty(), "every step passes the super-step deposition audit" + tag + (err.empty() ? "" : ": " + err));
        const auto& st = s.mte->getStatistics();
        const double net = st.total_sublimated_mass + st.total_sputtered_mass - st.total_deposited_mass;
        const double resid = (st.lag_inventory_mass - st.lag_inventory_initial_mass) + net;
        const double thr = st.total_sublimated_mass + st.total_sputtered_mass;
        check(std::abs(resid) <= 1e-12 * st.lag_inventory_initial_mass,
              "inventory change + (sublimated + sputtered - deposited) = 0" + tag,
              std::abs(resid) / st.lag_inventory_initial_mass, 1e-12);
        check(std::abs(resid) <= 1e-9 * thr, "... relative to the throughput" + tag, std::abs(resid) / thr, 1e-9);
        // Limit: rounding of the inventory (ulp of ~45 kg/m^2) against one step's flux.
        check(st.max_inventory_error_rel <= 1e-9, "per-face inventory = booked fluxes (max per step)" + tag,
              st.max_inventory_error_rel, 1e-9);
        check(st.max_step_residual_rel <= 1e-6, "step audit: deposited = retained emission" + tag,
              st.max_step_residual_rel, 1e-6);
        check(st.total_layer_overflow_mass == 0.0, "no off-book layer overflow (the veneer holds it)" + tag);
        double d_hot = 1e300, mf_cold = 0.0;
        for (size_t i = 0; i < 16; ++i) d_hot = std::min(d_hot, s.state->lag_thickness[i]);
        for (size_t i = 16; i < s.n; ++i) mf_cold += s.state->frost_veneer[i];
        check(d_hot > 10.0 * ell, "every hot face is lag-sealed (d > 10 l)" + tag, d_hot / ell, 10.0);
        check(mf_cold > 0.0, "cold faces carry a frost veneer" + tag, mf_cold, 0.0);
        const double fates = st.total_sputter_redeposited_mass + st.total_sputter_escaped_mass +
                             st.total_sputter_reservoir_mass;
        check(std::abs(fates / st.total_sputtered_mass - 1.0) < 1e-6, "sputter fates sum to the gross" + tag,
              std::abs(fates / st.total_sputtered_mass - 1.0), 1e-6);
    }
}

static void test_t0_invariance() {
    std::printf("\n(5) t = 0 invariance: lag at d = 0 == legacy, bit for bit\n");
    Sys a = makeSys(false, 5e-6), b = makeSys(true, 5e-6);
    size_t bad = 0;
    for (size_t i = 0; i < a.n; ++i) {
        const double f = a.model->getIceAvailabilityFactor(*a.state, i);
        double as, ap;
        exo::lagInstantAvailability(1.0, 1.0, 0.0, false, 0.0, 0.0, b.state->substrate_ice[i],
                                    b.state->lag_exposed_fraction[i], 5e-6, as, ap);
        if (b.model->getSublimationAvailability(*b.state, i) != f || b.model->getSputterAvailability(*b.state, i) != f ||
            as != f || ap != f || b.state->ice_mass[i] != a.state->ice_mass[i]) ++bad;
    }
    check(bad == 0, "availabilities and ice mass identical at d = 0", bad, 0);
    a.mte->applyScaledFluxes(*a.model, *a.state, a.areas);
    b.mte->applyScaledFluxes(*b.model, *b.state, b.areas);
    const auto& da = a.mte->getDiurnalAverages();
    const auto& db = b.mte->getDiurnalAverages();
    auto same = [](const std::vector<double>& x, const std::vector<double>& y) {
        return x.size() == y.size() && std::memcmp(x.data(), y.data(), x.size() * sizeof(double)) == 0;
    };
    check(same(da.avg_sublimation_flux, db.avg_sublimation_flux) && same(da.avg_sputtering_flux, db.avg_sputtering_flux) &&
          same(da.avg_deposition_flux, db.avg_deposition_flux) && da.total_base_emission == db.total_base_emission,
          "scaled fluxes, deposition and cache totals identical at t = 0");

    // Albedo: hapke-intimate forward model at d = 0.
    exo::CompositionAlbedoConfig ac;
    ac.hapke_intimate = true;
    ac.hapke_mixture = exo::hapke::makeIntimateMixture(0.80, 0.0, 0.3814289033, 0.0);
    ac.grain_ratio = 10.0;
    exo::CompositionAlbedoMapper mapper(ac);
    size_t bad_a = 0;
    for (int k = 0; k <= 1000; ++k) {
        const double c = k / 1000.0;
        if (mapper.computeAlbedoWithLag(c, 0.0, 1e-5) != mapper.computeAlbedo(c)) ++bad_a;
    }
    check(bad_a == 0, "albedo with lag at d = 0 == composition albedo (1001 concentrations)", bad_a, 0);
}

static void test_length_and_albedo() {
    std::printf("\n(6) barrier length and lag albedo\n");
    exo::LagBarrierConfig lc;
    const double l = lc.length();
    const double expect = 4.0 * 3.14159265358979323846 / (8.0 + 3.14159265358979323846) * 1.5 * 5e-6 / 2.0;
    check(std::abs(l / expect - 1.0) < 1e-15 && std::abs(l - 4.2297e-6) < 1e-9,
          "default l = 1.128 [0.6/0.4] 5 um / 2 = 4.23 um", l, 4.23e-6);
    lc.barrier_length = 5e-5;
    check(lc.length() == 5e-5, "explicit --lag-barrier-length overrides the Knudsen value");
    exo::CompositionAlbedoConfig ac;
    ac.hapke_intimate = true;
    ac.hapke_mixture = exo::hapke::makeIntimateMixture(0.80, 0.0, 0.3814289033, 0.0);
    ac.grain_ratio = 10.0;
    exo::CompositionAlbedoMapper mapper(ac);
    const double c = 0.05, A_mix = mapper.computeAlbedo(c), A_lag = mapper.computeAlbedo(1.0);
    check(std::abs(A_lag - 0.3814289033) < 1e-8, "A_lag = the non-ice endmember (P10 = 0.3814)", A_lag, 0.3814289033);
    const double A1 = mapper.computeAlbedoWithLag(c, 1e-5, 1e-5);
    check(std::abs(A1 - (A_lag + (A_mix - A_lag) * std::exp(-1.0))) < 1e-14, "A(d = delta) = A_lag + (A_mix - A_lag)/e",
          std::abs(A1 - (A_lag + (A_mix - A_lag) * std::exp(-1.0))), 1e-14);
    check(std::abs(mapper.computeAlbedoWithLag(c, 1.0, 1e-5) - A_lag) < 1e-14, "thick lag -> A_lag");
}

int main() {
    test_analytic_one_face();
    test_rk4_reference();
    test_veneer_then_barrier();
    test_global_mass();
    test_t0_invariance();
    test_length_and_albedo();
    std::printf("\n%d / %d checks passed\n", g_total - g_fail, g_total);
    return g_fail == 0 ? 0 : 1;
}
