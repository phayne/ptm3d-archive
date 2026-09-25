// Unit tests for the gardened lag with the sputter split (PTM-669). Self-contained;
// exit code 0 = all checks pass.
//
//  (1) Box depth vs scripts/jwst/gardening.py: Lambda_99(t)^2 = K99 t (Costello et al.
//      2021 Eq. 1, Cases A and B; reference values printed by gardening.py).
//  (2) Spencer criterion: at d = 0, beta (a + p_w + p_r - B) <= v_m is absorbing
//      (no lag, the box loses ice); just above it a lag forms; a lag on a sub-critical
//      face is gardened away and the face then stays at d = 0.
//  (3) Steady state: d -> d* with beta [(a + p_w)/u* + p_r - B] = v_m, from below
//      and from above; closed form vs RK4 with gardening and the shielded split.
//  (4) Veneer gardening: dm_f/dt = B - Jv - rho (1 - phi) v_m, the gardened frost
//      goes to the box; lag under the veneer untouched.
//  (5) Regression: v_m = 0, no shield: solveGardenFaceStep == solveLagFaceStep (1e-12).
//  (6) Global mass closure with gardening, entrainment and the split (64 faces):
//      inventory change + (sublimated + sputtered - deposited) - entrained <= 1e-12.
//  (7) t = 0 invariance in garden mode: availabilities bit-identical to the legacy.

#include "multitimescale_evolution.h"
#include "ice_layer_model.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace exo = ptm::exosphere;
namespace gc = ptm::exosphere::gardening_constants;

static int g_fail = 0, g_total = 0;
static void check(bool ok, const std::string& name, double value = 0.0, double limit = 0.0) {
    ++g_total;
    if (!ok) ++g_fail;
    std::printf("[%s] %-88s value=%.3e limit=%.1e\n", ok ? "PASS" : "FAIL", name.c_str(), value, limit);
    std::fflush(stdout);
}

static constexpr double YR = 365.25 * 86400.0;
static constexpr double kEpoch = 100.0 * exo::timescale_constants::EUROPA_DIURNAL_PERIOD;
static constexpr double RHO = exo::layer_constants::ICE_DENSITY;
static double betaOf(double c, double phi = 0.6) { return c / ((1.0 - c) * RHO * (1.0 - phi)); }

static void test_box_depth_vs_python() {
    std::printf("\n(1) box depth vs scripts/jwst/gardening.py (Lambda at lambda = 4.605)\n");
    // python3 -c "import gardening as g; print(repr(float(g.gardening_depth(t, lam=g.LAMBDA_SAT, case=C))))"
    struct Ref { char c; double t_yr, lambda_m; };
    const Ref refs[] = {
        {'A', 1e2, 0.00033203265436657504}, {'A', 1e3, 0.001049979445349829}, {'A', 1e4, 0.0033203265436657498},
        {'A', 1e5, 0.01049979445349829}, {'A', 1e6, 0.033203265436657504},
        {'B', 1e2, 0.0006194165073979799}, {'B', 1e3, 0.0019587669836841535}, {'B', 1e4, 0.0061941650739797995},
        {'B', 1e5, 0.019587669836841536}, {'B', 1e6, 0.061941650739797995},
    };
    double worst = 0.0;
    for (const auto& r : refs) {
        const double K = gc::K(r.c == 'A' ? gc::U_CASE_A : gc::U_CASE_B);
        worst = std::max(worst, std::abs(std::sqrt(K * r.t_yr * YR) / r.lambda_m - 1.0));
    }
    check(worst < 1e-12, "sqrt(K99 t) == gardening_depth(t, lam=4.605), Cases A and B, 1e2..1e6 yr", worst, 1e-12);
    check(std::abs(gc::K(gc::U_CASE_B) * YR / 3.84e-9 - 1.0) < 2e-3 && std::abs(gc::K(gc::U_CASE_A) * YR / 1.10e-9 - 1.0) < 3e-3,
          "K99 = 3.84e-9 (B), 1.10e-9 (A) m^2/yr (Fermi's values, rounded)", gc::K(gc::U_CASE_B) * YR, 3.84e-9);
    exo::LagBarrierConfig lc;
    lc.gardening = true;
    const double h = lc.gardenBoxDepth(1e5 * YR);
    check(std::abs(h - std::sqrt(0.01 * 0.01 + 0.019587669836841536 * 0.019587669836841536)) < 1e-15,
          "h_g(100 kyr) = sqrt(1 cm^2 + Lambda_99^2) (Case B)", h, 0.022);
}

static void test_spencer() {
    std::printf("\n(2) Spencer criterion: d = 0 absorbing iff beta (a + p_w + p_r - B) <= v_m\n");
    const double f = 0.5, c = 0.05, beta = betaOf(c), ell = 5e-6, vm = 0.3e-6 / YR, gv = RHO * 0.4 * vm;
    const double M = 500.0, B = 1e-12, Jw = 2e-12, Jr = 5e-13;
    // Critical sublimation: beta (Js f + (Jw + Jr) f - B) = vm.
    const double Js_c = ((vm / beta + B) / f) - Jw - Jr;
    for (double k : {0.9, 0.999}) {
        const auto st = exo::solveGardenFaceStep(k * Js_c, Jw, Jr, B, 0.0, 0.0, M, f, beta, ell, vm, gv, 1e12);
        const double L = (k * Js_c + Jw + Jr) * f - B;
        check(st.d_end == 0.0 && st.absorbing, "below critical (x" + std::to_string(k).substr(0, 5) + "): d stays 0",
              st.d_end, 0.0);
        check(std::abs((M - st.M_end) / (L * 1e12) - 1.0) < 1e-12, "  the box loses L t (no lag, its c rises)",
              std::abs((M - st.M_end) / (L * 1e12) - 1.0), 1e-12);
    }
    const auto up = exo::solveGardenFaceStep(1.01 * Js_c, Jw, Jr, B, 0.0, 0.0, M, f, beta, ell, vm, gv, 1e12);
    check(up.d_end > 0.0 && !up.absorbing, "above critical (x1.01): a lag forms", up.d_end, 0.0);
    // A lag on a sub-critical face is gardened away, then the face is absorbing.
    const double d0 = 2e-6;
    const auto gone = exo::solveGardenFaceStep(0.5 * Js_c, Jw, Jr, B, d0, 0.0, M, f, beta, ell, vm, gv, 1e12);
    check(gone.d_end == 0.0 && gone.absorbing, "sub-critical face with a lag: gardened to d = 0, then absorbing",
          gone.d_end, 0.0);
    // A thin box runs dry mid-step; the rest of the step is recycling (no chatter).
    const auto dry = exo::solveGardenFaceStep(1.01 * Js_c, Jw, Jr, B, 0.0, 0.0, 9.0, f, beta, ell, vm, gv, 1e12);
    const double em = dry.sub_int + dry.sputw_int + dry.sputr_int;
    check(std::abs((9.0 - dry.M_end + B * 1e12) / (em + 0.0) - 1.0) < 1e-9,
          "box runs dry mid-step: emitted = box ice + deposition (mass)", std::abs((9.0 - dry.M_end + B * 1e12) / em - 1.0), 1e-9);
    // Gardening off: the same sub-critical face grows a lag (the PTM-659 behavior).
    const auto nog = exo::solveGardenFaceStep(0.5 * Js_c, Jw, Jr, B, 0.0, 0.0, M, f, beta, ell, 0.0, 0.0, 1e12);
    check(nog.d_end > 0.0, "v_m = 0: every net source grows a lag", nog.d_end, 0.0);
}

static void rk4(double ap, double pr, double B, double beta, double ell, double vm, double d0, double T, int K,
                double& d_out, double& I_out) {
    auto rhs = [&](double d, double& dd, double& di) {
        const double e = ap / (1.0 + d / ell);
        dd = beta * (e + pr - B) - vm;
        di = e;
    };
    double d = d0, I = 0.0;
    const double h = T / K;
    for (int k = 0; k < K; ++k) {
        double a1, b1, a2, b2, a3, b3, a4, b4;
        rhs(d, a1, b1); rhs(d + 0.5 * h * a1, a2, b2); rhs(d + 0.5 * h * a2, a3, b3); rhs(d + h * a3, a4, b4);
        d += h / 6.0 * (a1 + 2 * a2 + 2 * a3 + a4);
        I += h / 6.0 * (b1 + 2 * b2 + 2 * b3 + b4);
    }
    d_out = d; I_out = I;
}

static void test_steady_state() {
    std::printf("\n(3) steady state d* and closed form vs RK4 (gardening, shielded H2O ejection)\n");
    const double f = 0.6, c = 0.08, beta = betaOf(c), ell = 5e-6, vm = 0.3e-6 / YR, gv = RHO * 0.4 * vm;
    const double Js = 4e-10, Jw = 3e-11, Jr = 4e-12, B = 5e-11, M = 1e5;   // deep box: no exhaustion
    const double ap = (Js + Jw) * f, pr = Jr * f;
    const double u_star = ap / (vm / beta + B - pr);
    const double d_star = ell * (u_star - 1.0);
    check(d_star > 0.0, "setup: d* > 0", d_star, 0.0);
    const auto lo = exo::solveGardenFaceStep(Js, Jw, Jr, B, 0.0, 0.0, M, f, beta, ell, vm, gv, 3e13);
    const auto hi = exo::solveGardenFaceStep(Js, Jw, Jr, B, 3.0 * d_star, 0.0, M, f, beta, ell, vm, gv, 3e13);
    check(lo.d_end < d_star && std::abs(lo.d_end / d_star - 1.0) < 1e-6, "d -> d* from below (1 Myr)",
          std::abs(lo.d_end / d_star - 1.0), 1e-6);
    check(hi.d_end > d_star && std::abs(hi.d_end / d_star - 1.0) < 1e-6, "d -> d* from above (lag gardened back)",
          std::abs(hi.d_end / d_star - 1.0), 1e-6);
    const auto at = exo::solveGardenFaceStep(Js, Jw, Jr, B, d_star, 0.0, M, f, beta, ell, vm, gv, 1e12);
    // At d*, the ice loss equals v_m / beta: the gardening returns the lag as fast as it forms.
    check(std::abs(at.d_end / d_star - 1.0) < 1e-9 && std::abs((M - at.M_end) / (vm / beta * 1e12) - 1.0) < 1e-6,
          "at d*: steady; box ice loss = (v_m / beta) t", std::abs((M - at.M_end) / (vm / beta * 1e12) - 1.0), 1e-6);
    for (double dd0 : {0.0, 0.5 * d_star, 2.0 * d_star}) {
        const double T = 2e10;
        const auto st = exo::solveGardenFaceStep(Js, Jw, Jr, B, dd0, 0.0, M, f, beta, ell, vm, gv, T);
        double dr, Ir;
        rk4(ap, pr, B, beta, ell, vm, dd0, T, 200000, dr, Ir);
        const double ed = std::abs(st.d_end - dr) / d_star;
        const double ei = std::abs((st.sub_int + st.sputw_int) - Ir) / Ir;
        check(ed < 1e-8 && ei < 1e-8, "closed form vs RK4 from d0 = " + std::to_string(dd0 / d_star).substr(0, 4) + " d*",
              std::max(ed, ei), 1e-8);
        check(std::abs(st.sub_int / st.sputw_int - Js / Jw) < 1e-9 * Js / Jw && std::abs(st.sputr_int - pr * T) < 1e-12 * pr * T,
              "  H2O ejection barriered like sublimation; radiolytic part unbarriered");
        const double booked = st.sub_int + st.sputw_int + st.sputr_int - B * T;
        check(std::abs((M - st.M_end) - booked) < 1e-12 * M, "  box change = booked net loss", std::abs((M - st.M_end) - booked), 1e-12 * M);
    }
}

static void test_veneer() {
    std::printf("\n(4) veneer gardening\n");
    const double vm = 1.2e-6 / YR, gv = RHO * 0.4 * vm, Js = 1e-13, Jw = 1e-14, Jr = 1e-15, T = 1e11;
    const double B = Js + Jw + Jr + 3.0 * gv, d0 = 4e-6, mf0 = 1e-3, M = 5.0;
    const auto st = exo::solveGardenFaceStep(Js, Jw, Jr, B, d0, mf0, M, 0.4, betaOf(0.1), 5e-6, vm, gv, T);
    check(std::abs(st.mf_end - (mf0 + (B - Js - Jw - Jr - gv) * T)) < 1e-12 * st.mf_end,
          "dm_f/dt = B - Jv - rho (1 - phi) v_m", std::abs(st.mf_end - (mf0 + (B - Js - Jw - Jr - gv) * T)), 1e-12);
    check(std::abs(st.M_end - (M + gv * T)) < 1e-12 * M && std::abs(st.box_gain - gv * T) < 1e-15,
          "gardened frost goes into the box ice");
    check(st.d_end == d0, "lag under the veneer untouched");
    // Deposition below Jv + gv on a bare face: frost is gardened in as it lands (recycling).
    const double Bm = Js + Jw + Jr + 0.5 * gv;
    const auto r = exo::solveGardenFaceStep(Js, Jw, Jr, Bm, 0.0, 0.0, 0.0, 0.4, betaOf(0.1), 5e-6, vm, gv, T);
    const double emit = r.sub_int + r.sputw_int + r.sputr_int;
    check(r.mf_end == 0.0 && std::abs((emit + r.box_gain) / (Bm * T) - 1.0) < 1e-12,
          "recycling with gardening: emission + frost gardened into the box = deposition",
          std::abs((emit + r.box_gain) / (Bm * T) - 1.0), 1e-12);
}

static void test_regression_vs_lag() {
    std::printf("\n(5) v_m = 0, no shield: garden step == PTM-659 lag step\n");
    double worst = 0.0;
    const double ell = 5e-6;
    struct Case { double Js, Jp, B, d0, mf0, M, f, c, T; };
    const Case cases[] = {
        {3e-10, 0.0, 0.0, 0.0, 0.0, 40.0, 0.7, 0.05, 3e12},
        {2e-10, 2e-11, 5e-11, 3e-6, 0.0, 40.0, 0.7, 0.1, 1e11},
        {2e-10, 0.0, 1e-10, 0.0, 0.0, 40.0, 0.7, 0.1, 1e14},
        {2e-10, 1e-11, 4e-11, 2e-6, 1e-2, 30.0, 0.6, 0.08, 1e10},
        {2e-10, 1e-11, 1.5e-10, 2e-5, 0.0, 30.0, 0.6, 0.08, 1e10},
        {1e-13, 1e-14, 3e-13, 0.0, 0.0, 30.0, 0.6, 0.08, 1e10},
        {2e-10, 1e-11, 1e-12, 0.0, 0.0, 1e-3, 0.6, 0.08, 1e12},
    };
    for (const auto& k : cases) {
        const double beta = betaOf(k.c);
        const auto a = exo::solveLagFaceStep(k.Js, k.Jp, k.B, k.d0, k.mf0, k.M, k.f, beta, ell, k.T);
        const auto b = exo::solveGardenFaceStep(k.Js, 0.7 * k.Jp, 0.3 * k.Jp, k.B, k.d0, k.mf0, k.M, k.f, beta, ell,
                                                0.0, 0.0, k.T, false);
        auto rel = [](double x, double y) { return std::abs(x - y) / std::max({std::abs(x), std::abs(y), 1e-300}); };
        worst = std::max({worst, rel(a.d_end, b.d_end), rel(a.mf_end, b.mf_end), rel(a.M_end, b.M_end),
                          rel(a.sub_int, b.sub_int), rel(a.sput_int, b.sputw_int + b.sputr_int)});
    }
    check(worst < 1e-10, "7 cases (source, deposition, d*, veneer, recycling, veneer growth, depletion)", worst, 1e-10);
}

// ---------------------------------------------------------------------------
struct Sys {
    size_t n = 64;
    std::unique_ptr<exo::IceLayerModel> model;
    std::unique_ptr<exo::IceLayerState> state;
    std::unique_ptr<exo::MultiTimescaleEvolution> mte;
    std::vector<double> areas, sub, sput;
};

static Sys makeSys(bool garden) {
    Sys s;
    exo::IceLayerConfig cfg;
    s.model = std::make_unique<exo::IceLayerModel>(s.n, cfg);
    s.model->setExposedAreaGrainRatio(10.0);
    s.state = std::make_unique<exo::IceLayerState>(s.n);
    std::vector<double> conc(s.n);
    s.areas.resize(s.n); s.sub.resize(s.n); s.sput.resize(s.n);
    for (size_t i = 0; i < s.n; ++i) {
        const bool hot = i < 16;
        s.areas[i] = 1.0e6 * (1.0 + 0.01 * static_cast<double>(i));
        conc[i] = hot ? 0.03 + 0.01 * static_cast<double>(i) : (i % 7 == 0 ? 0.0 : 0.05);
        s.sub[i] = hot ? 5e-10 * (1.0 + 0.3 * static_cast<double>(i)) : 1e-14;
        s.sput[i] = 3e-12;
    }
    s.model->initializeFromConcentrations(*s.state, conc);
    exo::LagBarrierConfig lc;
    lc.enabled = true;
    lc.barrier_length = 5e-6;
    lc.gardening = garden;
    lc.sputter_shield = garden;
    lc.gardening_rate = 1.2e-6 / YR;
    s.model->enableLagBarrier(*s.state, lc);
    exo::MultiTimescaleConfig mc;
    s.mte = std::make_unique<exo::MultiTimescaleEvolution>(s.n, mc);
    const double share = 0.765;
    s.mte->setSputterPartition(share, 0.2208, 1.0 - share - 0.2208);
    auto& d = s.mte->getMutableDiurnalAverages();
    // Per-face composition: 90% H2O ejection (1 - e_H2O = 0.85 of it transported),
    // radiolytic part: 10% escapes, the rest atmospheric loss.
    d.sput_w.assign(s.n, 0.9); d.sput_tw.assign(s.n, 0.85); d.sput_ew.assign(s.n, 0.15); d.sput_er.assign(s.n, 0.1);
    d.sputter_face_transport_frac.assign(s.n, 0.9 * 0.85);
    d.sputter_face_escape_frac.assign(s.n, 0.9 * 0.15 + 0.1 * 0.1);
    d.sputter_face_reservoir_frac.assign(s.n, 1.0 - 0.9 * 0.85 - (0.9 * 0.15 + 0.1 * 0.1));
    std::vector<double> as, ap;
    s.mte->lagAvailabilities(*s.model, *s.state, s.sub, s.sput, as, ap);
    double S = 0.0, P = 0.0, w_hot = 0.0, w_cold = 0.0;
    for (size_t i = 0; i < s.n; ++i) {
        S += s.sub[i] * as[i] * s.areas[i];
        P += s.sput[i] * ap[i] * d.sputter_face_transport_frac[i] * s.areas[i];
        (i < 16 ? w_hot : w_cold) += s.areas[i];
    }
    for (size_t i = 0; i < s.n; ++i) {
        const double share_i = (i < 16) ? 0.05 / w_hot : 0.95 / w_cold;
        d.avg_sublimation_flux[i] = s.sub[i] * as[i];
        d.avg_sputtering_flux[i] = s.sput[i] * ap[i];
        d.avg_deposition_sublimation_flux[i] = (1.0 - 1e-3) * S * share_i;
        d.avg_deposition_sputter_flux[i] = P * share_i;
        d.avg_deposition_flux[i] = d.avg_deposition_sublimation_flux[i] + d.avg_deposition_sputter_flux[i];
    }
    s.mte->cacheBaseFluxesUnscaled(*s.model, *s.state, s.sub, s.sput, s.areas, 0);
    return s;
}

static void test_global_mass() {
    std::printf("\n(6) global mass closure: gardening + entrainment + sputter split (64 faces)\n");
    Sys s = makeSys(true);
    std::string err;
    const int total = 30000;
    int e = 0, iters = 0;
    try {
        while (e < total) {
            s.mte->applyScaledFluxes(*s.model, *s.state, s.areas);
            s.mte->evolveEpoch(*s.model, *s.state, s.areas, kEpoch);
            const int rem = total - e - 1;
            int extra = 0;
            if (rem > 0) extra = std::min(rem, s.mte->computeSafeEpochCountLag(*s.model, *s.state, s.areas, kEpoch, rem) - 1);
            if (extra > 0) s.mte->evolveEpoch(*s.model, *s.state, s.areas, kEpoch * extra);
            e += 1 + std::max(0, extra);
            ++iters;
        }
    } catch (const std::exception& x) { err = x.what(); }
    check(err.empty(), "30 kyr in " + std::to_string(iters) + " steps; every step passes the PTM-644 audit" +
                       (err.empty() ? "" : ": " + err));
    const auto& st = s.mte->getStatistics();
    const double net = st.total_sublimated_mass + st.total_sputtered_mass - st.total_deposited_mass;
    const double resid = (st.lag_inventory_mass - st.lag_inventory_initial_mass) + net - st.garden_entrained_mass;
    check(st.garden_entrained_mass > 0.0, "the growing box entrained ice", st.garden_entrained_mass, 0.0);
    check(std::abs(resid) <= 1e-12 * st.lag_inventory_initial_mass,
          "inventory change + (sub + sput - dep) - entrained = 0", std::abs(resid) / st.lag_inventory_initial_mass, 1e-12);
    check(st.max_inventory_error_rel <= 1e-9, "per-face inventory = booked fluxes + entrainment (max per step)",
          st.max_inventory_error_rel, 1e-9);
    check(st.max_step_residual_rel <= 1e-6, "step audit: deposited = retained emission", st.max_step_residual_rel, 1e-6);
    const double fates = st.total_sputter_redeposited_mass + st.total_sputter_escaped_mass + st.total_sputter_reservoir_mass;
    check(std::abs(fates / st.total_sputtered_mass - 1.0) < 1e-6, "sputter fates sum to the gross",
          std::abs(fates / st.total_sputtered_mass - 1.0), 1e-6);
    const double h = s.model->getLagConfig().gardenBoxDepth(s.state->lag_time);
    check(std::abs(h - std::sqrt(1e-4 + gc::K(gc::U_CASE_B) * s.state->lag_time)) < 1e-15, "box depth follows h_g(t)", h, 0.0);
}

static void test_t0() {
    std::printf("\n(7) t = 0 invariance in garden mode\n");
    Sys g = makeSys(true);
    exo::IceLayerConfig cfg;
    exo::IceLayerModel legacy(64, cfg);
    legacy.setExposedAreaGrainRatio(10.0);
    exo::IceLayerState ls(64);
    std::vector<double> conc(64);
    for (size_t i = 0; i < 64; ++i) conc[i] = g.state->box_c0[i];
    legacy.initializeFromConcentrations(ls, conc);
    std::vector<double> as, ap;
    g.mte->lagAvailabilities(*g.model, *g.state, g.sub, g.sput, as, ap);
    size_t bad = 0;
    for (size_t i = 0; i < 64; ++i) {
        const double f = legacy.getIceAvailabilityFactor(ls, i);
        if (as[i] != f || ap[i] != f || g.state->nonice_concentration[i] != conc[i]) ++bad;
    }
    check(bad == 0, "sublimation and gross sputter availability and c identical at d = 0", bad, 0);
    check(std::abs(g.state->substrate_ice[1] - (1.0 - conc[1]) * RHO * 0.01) < 1e-12,
          "box starts at h_mm = 1 cm (ice mass differs from the 5 cm legacy layer; composition does not)");
}

int main() {
    test_box_depth_vs_python();
    test_spencer();
    test_steady_state();
    test_veneer();
    test_regression_vs_lag();
    test_global_mass();
    test_t0();
    std::printf("\n%d / %d checks passed\n", g_total - g_fail, g_total);
    return g_fail == 0 ? 0 : 1;
}
