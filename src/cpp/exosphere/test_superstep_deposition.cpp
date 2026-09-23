// Unit tests for the multiscale super-step deposition (PTM-644). Self-contained;
// exit code 0 = all checks pass.
//
// A super-step holds deposition fixed while each face's emission follows its ice
// availability. Pre-fix, deposition was held at its START-of-step value, so a step
// in which availability fell deposited more than it emitted. Now deposition is held
// at the global scale whose deposition equals what the kernel retains from the
// step's own emission (a scalar Newton solve per source group).
//
//  (1) Exact step-mean availability (linear mode): stepFace() against quadrature,
//      including a face that fills to pure ice mid-step.
//  (2) Synthetic depleting system, sputtering OFF, linear and exposed-area modes:
//      the legacy step FAILS the conservation audit (deposition exceeds retained
//      emission); the new step passes at <= 1e-6 per step, and the cumulative
//      identity sublimated - deposited = not redeposited - residual closes.
//  (3) The same with sputter transport (two source groups).
//  (4) The audit throws when deposition and the kernel's retention disagree.
//  (5) Constant availability (exact equilibrium, and zero flux): the new step is
//      bit-identical to the legacy step.
//  (6) Step refinement: halving the super-step, the new cumulative totals converge
//      (Cauchy differences shrink) and conserve at every step size; the legacy
//      residual shrinks with the step, i.e. it is a step-size artifact.
//  (7) Ice inventory: the state change equals the booked fluxes (new), which the
//      legacy trapezoid mean violates in linear mode.
//  (8) End to end (ico3 synthetic body, --multiscale, CPU), sputtering OFF and ON:
//      the run completes (the audit throws otherwise) and evolution_summary.txt
//      reports a residual <= 1e-6; the legacy step reports a larger one.

#include "multitimescale_evolution.h"
#include "ice_layer_model.h"
#include "composition_albedo.h"
#include "ptm_exosphere_integration.h"
#include "sputtering_model.h"
#include "../mesh/mesh_utils.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <unistd.h>

namespace exo = ptm::exosphere;
namespace fs = std::filesystem;

static int g_fail = 0, g_total = 0;
static void check(bool ok, const std::string& name, double value = 0.0, double limit = 0.0) {
    ++g_total;
    if (!ok) ++g_fail;
    std::printf("[%s] %-84s value=%.3e limit=%.1e\n", ok ? "PASS" : "FAIL", name.c_str(), value, limit);
    std::fflush(stdout);
}

// ---------------------------------------------------------------------------
// Synthetic system: 16 hot faces that deplete, 48 cold faces that receive.
struct Sys {
    size_t n = 64;
    exo::IceLayerConfig cfg;
    std::unique_ptr<exo::IceLayerModel> model;
    std::unique_ptr<exo::IceLayerState> state;
    std::unique_ptr<exo::MultiTimescaleEvolution> mte;
    std::vector<double> areas;
};

static constexpr double kEpoch = 100.0 * exo::timescale_constants::EUROPA_DIURNAL_PERIOD;

// area_ratio > 0: exposed-area mode with that grain ratio. two: sputter transport.
static Sys makeSys(bool consistent, double area_ratio, bool two, int max_step = 0) {
    Sys s;
    s.cfg.thermal_skin_depth = 0.05;
    s.model = std::make_unique<exo::IceLayerModel>(s.n, s.cfg);
    if (area_ratio > 0.0) s.model->setExposedAreaGrainRatio(area_ratio);
    s.state = std::make_unique<exo::IceLayerState>(s.n);
    std::vector<double> conc(s.n), sub(s.n), sput(s.n);
    s.areas.resize(s.n);
    const double M = s.model->getMaxIceMassPerArea();
    for (size_t i = 0; i < s.n; ++i) {
        s.areas[i] = 1.0e6 * (1.0 + 0.01 * static_cast<double>(i));
        const bool hot = i < 16;
        conc[i] = hot ? 0.10 + 0.01 * static_cast<double>(i) : 0.30;
        // Hot: depletion time M/E ~ 150 yr; cold: negligible emission.
        sub[i] = hot ? M / (4.6e9 * (1.0 + 0.1 * static_cast<double>(i))) : 1e-15;
        sput[i] = two ? 0.2 * sub[i] + 1e-14 : 0.0;
    }
    s.model->initializeFromConcentrations(*s.state, conc);
    exo::MultiTimescaleConfig mc;
    mc.superstep_consistent_deposition = consistent;
    mc.max_superstep_epochs = max_step;
    s.mte = std::make_unique<exo::MultiTimescaleEvolution>(s.n, mc);
    const double share = 0.765;
    if (two) s.mte->setSputterPartition(share, 0.2208, 1.0 - share - 0.2208);

    // Emission at cache time (availability-scaled) and a kernel that redeposits
    // it mostly onto the cold faces, 10% back onto the hot ones, retaining
    // rho = 1 - 1e-3 of the sublimation (the rest is Jeans escape).
    auto& d = s.mte->getMutableDiurnalAverages();
    double S = 0.0, P = 0.0, w_hot = 0.0, w_cold = 0.0;
    for (size_t i = 0; i < s.n; ++i) {
        const double f = s.model->getIceAvailabilityFactor(*s.state, i);
        S += sub[i] * f * s.areas[i];
        P += sput[i] * f * s.areas[i];
        (i < 16 ? w_hot : w_cold) += s.areas[i];
    }
    const double rho = 1.0 - 1e-3;
    for (size_t i = 0; i < s.n; ++i) {
        const double share_i = (i < 16) ? 0.1 / w_hot : 0.9 / w_cold;  // per m^2
        const double ds = rho * S * share_i, dp = share * P * share_i;
        d.avg_sublimation_flux[i] = sub[i] * s.model->getIceAvailabilityFactor(*s.state, i);
        d.avg_sputtering_flux[i] = sput[i] * s.model->getIceAvailabilityFactor(*s.state, i);
        if (two) {
            d.avg_deposition_sublimation_flux[i] = ds;
            d.avg_deposition_sputter_flux[i] = dp;
            d.avg_deposition_flux[i] = ds + dp;
        } else {
            d.avg_deposition_flux[i] = ds;
        }
    }
    s.mte->cacheBaseFluxesUnscaled(*s.model, *s.state, sub, sput, s.areas, 0);
    return s;
}

// Mimic runMultiTimescaleSimulation: per iteration, rescale, one epoch, then a super-step.
static void runSys(Sys& s, int total_epochs, int superstep) {
    int e = 0;
    while (e < total_epochs) {
        s.mte->applyScaledFluxes(*s.model, *s.state, s.areas);
        s.mte->evolveEpoch(*s.model, *s.state, s.areas, kEpoch);
        int extra = std::min(superstep - 1, total_epochs - e - 1);
        if (extra > 0) s.mte->evolveEpoch(*s.model, *s.state, s.areas, kEpoch * extra);
        e += 1 + std::max(0, extra);
    }
}

// ---------------------------------------------------------------------------
static void test_exact_linear_mean() {
    std::printf("\n(1) exact step-mean availability (linear mode) vs quadrature\n");
    exo::IceLayerConfig cfg;
    exo::IceLayerModel m(1, cfg);
    exo::MultiTimescaleEvolution mte(1);
    struct Case { double c0, A, B, dt; const char* tag; };
    const Case cases[] = {
        {0.20, 1e-9, 2e-10, 3e9, "depleting toward f_eq = 0.2"},
        {0.60, 1e-9, 8e-10, 3e9, "recovering toward f_eq = 0.8"},
        {0.30, 1e-9, 1.6e-9, 3e9, "fills to pure ice mid-step (f_eq = 1.6)"},
    };
    for (const auto& c : cases) {
        const auto st = mte.stepFace(m, c.c0, c.A, c.B, c.dt, true);
        const auto tr = mte.stepFace(m, c.c0, c.A, c.B, c.dt, false);
        // Composite Simpson on f(t) = min(1, f_eq + (f0 - f_eq) exp(-A t)).
        const int K = 200000;
        const double f0 = 1.0 - c.c0, feq = c.B / c.A, h = c.dt / K;
        auto f = [&](double t) { return std::min(1.0, feq + (f0 - feq) * std::exp(-c.A * t)); };
        double q = f(0.0) + f(c.dt);
        for (int k = 1; k < K; ++k) q += (k % 2 ? 4.0 : 2.0) * f(k * h);
        q *= h / 3.0 / c.dt;
        check(std::abs(st.mean_avail - q) < 1e-9, std::string("exact mean = quadrature: ") + c.tag,
              std::abs(st.mean_avail - q), 1e-9);
        check(std::abs(tr.mean_avail - q) > 1e-3, std::string("legacy trapezoid is off (control): ") + c.tag,
              std::abs(tr.mean_avail - q), 1e-3);
        check(std::abs(std::max(0.0, st.nonice_new) - std::max(0.0, tr.nonice_new)) < 1e-13,
              std::string("end state agrees with the legacy closed form: ") + c.tag,
              std::abs(std::max(0.0, st.nonice_new) - std::max(0.0, tr.nonice_new)), 1e-13);
    }
}

static void test_depleting(double area_ratio, bool two) {
    const std::string mode = std::string(area_ratio > 0.0 ? "exposed-area" : "linear") +
                             (two ? ", sputter transport" : ", sputtering OFF");
    std::printf("\n(%d) depleting system (%s): legacy fails, new passes\n", two ? 3 : 2, mode.c_str());
    Sys old = makeSys(false, area_ratio, two), neu = makeSys(true, area_ratio, two);
    runSys(old, 400, 50);
    std::string err;
    try { runSys(neu, 400, 50); } catch (const std::exception& e) { err = e.what(); }
    check(err.empty(), "new: every step passes the audit (it throws otherwise)" + (err.empty() ? "" : ": " + err));
    const auto& so = old.mte->getStatistics();
    const auto& sn = neu.mte->getStatistics();
    const double emit_o = so.total_sublimated_mass + so.total_sputtered_mass;
    const double emit_n = sn.total_sublimated_mass + sn.total_sputtered_mass;
    check(so.max_step_residual_rel > 1e-4, "legacy: a step deposits more than it retains from its emission",
          so.max_step_residual_rel, 1e-4);
    check(so.total_step_residual_mass > 1e-4 * emit_o, "legacy: cumulative deposition EXCEEDS retained emission",
          so.total_step_residual_mass / emit_o, 1e-4);
    check(sn.max_step_residual_rel <= 1e-6, "new: max |step residual| / emitted <= 1e-6",
          sn.max_step_residual_rel, 1e-6);
    check(std::abs(sn.total_step_residual_mass) <= 1e-9 * emit_n, "new: cumulative residual / emitted <= 1e-9",
          std::abs(sn.total_step_residual_mass) / emit_n, 1e-9);
    const double lhs = emit_n - sn.total_deposited_mass;
    const double rhs = sn.total_not_redeposited_mass - sn.total_step_residual_mass;
    check(std::abs(lhs - rhs) <= 1e-12 * emit_n, "new: emitted - deposited = not redeposited - residual",
          std::abs(lhs - rhs) / emit_n, 1e-12);
    check(sn.total_deposited_mass <= emit_n - sn.total_not_redeposited_mass + 1e-6 * emit_n,
          "new: deposited <= emitted - (Jeans + escape + reservoir)",
          (sn.total_deposited_mass - (emit_n - sn.total_not_redeposited_mass)) / emit_n, 1e-6);
    check(sn.step_solver_max_iterations <= 8, "new: Newton converges in <= 8 iterations",
          sn.step_solver_max_iterations, 8);
    if (two) {
        const double fates = sn.total_sputter_redeposited_mass + sn.total_sputter_escaped_mass +
                             sn.total_sputter_reservoir_mass;
        check(std::abs(fates / sn.total_sputtered_mass - 1.0) < 1e-6,
              "new: sputter redeposited + escaped + reservoir = gross (cumulative)",
              std::abs(fates / sn.total_sputtered_mass - 1.0), 1e-6);
    }
}

static void test_audit_throws() {
    std::printf("\n(4) the audit throws when deposition disagrees with the kernel's retention\n");
    Sys s = makeSys(true, 0.0, false);
    // Book 1% less retention than the deposition pattern carries.
    s.mte->getMutableDiurnalAverages().total_base_deposition *= 0.99;
    bool threw = false;
    std::string what;
    try { runSys(s, 10, 5); } catch (const std::runtime_error& e) { threw = true; what = e.what(); }
    check(threw && what.find("step conservation audit failed") != std::string::npos,
          "tampered retention -> std::runtime_error (step conservation audit)");
}

static bool sameState(const Sys& a, const Sys& b) {
    auto eq = [](const std::vector<double>& x, const std::vector<double>& y) {
        return x.size() == y.size() && std::memcmp(x.data(), y.data(), x.size() * sizeof(double)) == 0;
    };
    return eq(a.state->nonice_concentration, b.state->nonice_concentration) &&
           eq(a.state->ice_mass, b.state->ice_mass) &&
           eq(a.state->cumulative_sublimated, b.state->cumulative_sublimated) &&
           eq(a.state->cumulative_deposited, b.state->cumulative_deposited) &&
           eq(a.state->cumulative_sputtered, b.state->cumulative_sputtered);
}

static void test_constant_availability() {
    std::printf("\n(5) constant availability: new step bit-identical to the legacy step\n");
    for (int zero = 0; zero < 2; ++zero) {
        Sys r[2];
        for (int k = 0; k < 2; ++k) {
            Sys& s = r[k];
            s.cfg.thermal_skin_depth = 0.05;
            s.model = std::make_unique<exo::IceLayerModel>(s.n, s.cfg);
            s.state = std::make_unique<exo::IceLayerState>(s.n);
            s.model->initializeFromConcentrations(*s.state, std::vector<double>(s.n, 0.5));
            exo::MultiTimescaleConfig mc;
            mc.superstep_consistent_deposition = (k == 1);
            s.mte = std::make_unique<exo::MultiTimescaleEvolution>(s.n, mc);
            s.areas.assign(s.n, 0.0);
            std::vector<double> sub(s.n), sput(s.n, 0.0);
            auto& d = s.mte->getMutableDiurnalAverages();
            for (size_t i = 0; i < s.n; ++i) {
                s.areas[i] = 1e6 * (1.0 + 0.01 * static_cast<double>(i));
                // Each face redeposits exactly what it emits (f0 = 0.5 is the equilibrium).
                sub[i] = zero ? 0.0 : 1e-9 * (1.0 + 0.1 * static_cast<double>(i));
                d.avg_sublimation_flux[i] = sub[i] * 0.5;
                d.avg_deposition_flux[i] = sub[i] * 0.5;
            }
            s.mte->cacheBaseFluxesUnscaled(*s.model, *s.state, sub, sput, s.areas, 0);
            runSys(s, 300, 60);
        }
        const std::string tag = zero ? "zero flux" : "exact equilibrium";
        check(sameState(r[0], r[1]), "state + cumulative fields byte-identical: " + tag);
        check(r[1].mte->getStatistics().step_solver_max_iterations == 0,
              "start-of-step scale accepted without iterating: " + tag,
              r[1].mte->getStatistics().step_solver_max_iterations, 0);
    }
}

static void test_refinement() {
    std::printf("\n(6) super-step refinement (exposed-area mode, sputter transport, 800 epochs)\n");
    const int steps[] = {200, 100, 50, 25, 12};
    double dep_new[5], sub_new[5], res_old[5], dep_old[5];
    for (int k = 0; k < 5; ++k) {
        Sys a = makeSys(true, 2.0, true), b = makeSys(false, 2.0, true);
        runSys(a, 800, steps[k]);
        runSys(b, 800, steps[k]);
        const auto& sa = a.mte->getStatistics();
        const auto& sb = b.mte->getStatistics();
        dep_new[k] = sa.total_deposited_mass;
        sub_new[k] = sa.total_sublimated_mass;
        dep_old[k] = sb.total_deposited_mass;
        res_old[k] = sb.total_step_residual_mass / (sb.total_sublimated_mass + sb.total_sputtered_mass);
        std::printf("    step %3d: new deposited %.9e sublimated %.9e max|res| %.1e | legacy deposited %.9e "
                    "residual %.3e\n", steps[k], dep_new[k], sub_new[k], sa.max_step_residual_rel, dep_old[k],
                    res_old[k]);
        check(sa.max_step_residual_rel <= 1e-6, "new conserves at step " + std::to_string(steps[k]),
              sa.max_step_residual_rel, 1e-6);
    }
    for (int k = 1; k < 4; ++k) {
        const double d1 = std::abs(dep_new[k] - dep_new[k - 1]), d2 = std::abs(dep_new[k + 1] - dep_new[k]);
        check(d2 < 0.75 * d1, "new deposited: Cauchy difference shrinks, step " + std::to_string(steps[k + 1]),
              d2 / d1, 0.75);
    }
    check(std::abs(dep_new[4] / dep_new[3] - 1.0) < 1e-3, "new deposited: last halving moves it < 1e-3",
          std::abs(dep_new[4] / dep_new[3] - 1.0), 1e-3);
    check(res_old[0] > 2.0 * res_old[4] && res_old[4] > 0.0,
          "legacy residual shrinks with the step (a step-size artifact)", res_old[0] / res_old[4], 2.0);
}

static void test_inventory() {
    std::printf("\n(7) ice inventory = booked fluxes\n");
    Sys neu = makeSys(true, 0.0, false), old = makeSys(false, 0.0, false);
    runSys(neu, 400, 50);
    runSys(old, 400, 50);
    check(neu.mte->getStatistics().max_inventory_error_rel < 1e-9, "new (linear): state change = booked fluxes",
          neu.mte->getStatistics().max_inventory_error_rel, 1e-9);
    check(old.mte->getStatistics().max_inventory_error_rel > 1e-4,
          "legacy trapezoid (linear): booked emission disagrees with the state change (control)",
          old.mte->getStatistics().max_inventory_error_rel, 1e-4);
    // A pure-ice face that receives more than it can emit: the excess overflows
    // the layer. Face 0 (half ice) emits; the kernel puts everything (rho = 1) on
    // face 1 (pure ice). Overflow = (step deposition - emission) x area x dt on face 1.
    for (double r : {0.0, 2.0}) {
        exo::IceLayerConfig cfg;
        cfg.thermal_skin_depth = 0.05;
        exo::IceLayerModel m(2, cfg);
        if (r > 0.0) m.setExposedAreaGrainRatio(r);
        exo::IceLayerState st(2);
        m.initializeFromConcentrations(st, {0.5, 0.0});
        exo::MultiTimescaleEvolution e(2);
        const double E0 = 1e-8, E1 = 1e-10, dt = 5 * kEpoch;
        std::vector<double> sub = {E0, E1}, sput = {0.0, 0.0}, areas = {2e6, 1e6};
        auto& d = e.getMutableDiurnalAverages();
        double S = 0.0;
        for (int i = 0; i < 2; ++i) {
            d.avg_sublimation_flux[i] = sub[i] * m.getIceAvailabilityFactor(st, i);
            S += d.avg_sublimation_flux[i] * areas[i];
        }
        d.avg_deposition_flux[0] = 0.0;
        d.avg_deposition_flux[1] = S / areas[1];
        e.cacheBaseFluxesUnscaled(m, st, sub, sput, areas, 0);
        e.evolveEpoch(m, st, areas, dt);
        const double want = (e.getStepDepositionFlux()[1] - E1) * areas[1] * dt;
        const double got = e.getStatistics().total_layer_overflow_mass;
        check(want > 0.0 && std::abs(got / want - 1.0) < 1e-12 && st.nonice_concentration[1] == 0.0,
              std::string("pure-ice overflow = (step deposition - emission) x area x dt (") +
              (r > 0 ? "exposed-area" : "linear") + ")", std::abs(got / want - 1.0), 1e-12);
        check(e.getStatistics().max_step_residual_rel <= 1e-6, std::string("... and the step conserves (") +
              (r > 0 ? "exposed-area" : "linear") + ")", e.getStatistics().max_step_residual_rel, 1e-6);
    }
    // A cached kernel that deposits more than its source is refused.
    {
        Sys s = makeSys(true, 0.0, false);
        auto& d = s.mte->getMutableDiurnalAverages();
        d.total_base_deposition = 1.01 * d.total_base_emission;
        bool threw = false;
        try { runSys(s, 2, 1); } catch (const std::runtime_error& e) {
            threw = std::string(e.what()).find("exceeds its source") != std::string::npos;
        }
        check(threw, "cached deposition > its source -> std::runtime_error");
    }
    Sys area = makeSys(true, 2.0, true);
    runSys(area, 400, 50);
    check(area.mte->getStatistics().max_inventory_error_rel < 1e-9, "new (exposed-area): state change = booked",
          area.mte->getStatistics().max_inventory_error_rel, 1e-9);
}

// ---------------------------------------------------------------------------
// End to end, as in test_sputter_mass_accounting.cpp.
static ptm::Mesh make_icosphere(int levels, double radius) {
    const double t = (1.0 + std::sqrt(5.0)) / 2.0;
    std::vector<std::array<double, 3>> v = {
        {-1, t, 0}, {1, t, 0}, {-1, -t, 0}, {1, -t, 0}, {0, -1, t}, {0, 1, t},
        {0, -1, -t}, {0, 1, -t}, {t, 0, -1}, {t, 0, 1}, {-t, 0, -1}, {-t, 0, 1}};
    std::vector<std::array<int, 3>> f = {
        {0, 11, 5}, {0, 5, 1}, {0, 1, 7}, {0, 7, 10}, {0, 10, 11}, {1, 5, 9}, {5, 11, 4},
        {11, 10, 2}, {10, 7, 6}, {7, 1, 8}, {3, 9, 4}, {3, 4, 2}, {3, 2, 6}, {3, 6, 8},
        {3, 8, 9}, {4, 9, 5}, {2, 4, 11}, {6, 2, 10}, {8, 6, 7}, {9, 8, 1}};
    auto norm = [](std::array<double, 3> a) {
        const double r = std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
        return std::array<double, 3>{a[0] / r, a[1] / r, a[2] / r};
    };
    for (auto& p : v) p = norm(p);
    for (int l = 0; l < levels; ++l) {
        std::map<std::pair<int, int>, int> cache;
        auto mid = [&](int a, int b) {
            const auto key = std::make_pair(std::min(a, b), std::max(a, b));
            auto it = cache.find(key);
            if (it != cache.end()) return it->second;
            v.push_back(norm({v[a][0] + v[b][0], v[a][1] + v[b][1], v[a][2] + v[b][2]}));
            return cache[key] = static_cast<int>(v.size()) - 1;
        };
        std::vector<std::array<int, 3>> nf;
        for (const auto& tri : f) {
            const int ab = mid(tri[0], tri[1]), bc = mid(tri[1], tri[2]), ca = mid(tri[2], tri[0]);
            nf.push_back({tri[0], ab, ca});
            nf.push_back({tri[1], bc, ab});
            nf.push_back({tri[2], ca, bc});
            nf.push_back({ab, bc, ca});
        }
        f = std::move(nf);
    }
    ptm::Mesh m;
    for (auto& p : v) m.vertices.push_back({p[0] * radius, p[1] * radius, p[2] * radius});
    m.faces = f;
    return m;
}

static const int kNt = 8;
static void writeThermal(const fs::path& dir, const ptm::Mesh& mesh, double dT = 0.0) {
    fs::create_directories(dir / "temperature");
    const size_t n = mesh.faces.size();
    for (int t = 0; t < kNt; ++t) {
        std::vector<float> T(n);
        for (size_t f = 0; f < n; ++f) {
            const auto& fc = mesh.faces[f];
            double c[3] = {0.0, 0.0, 0.0};
            for (int k = 0; k < 3; ++k)
                for (int j = 0; j < 3; ++j) c[j] += mesh.vertices[fc[k]][j] / 3.0;
            const double mu = std::abs(c[2]) / std::sqrt(c[0] * c[0] + c[1] * c[1] + c[2] * c[2]);
            T[f] = static_cast<float>(150.0 + dT - 70.0 * mu + 8.0 * std::cos(2.0 * M_PI * t / kNt + 0.37 * f));
        }
        std::ostringstream fn;
        fn << (dir / "temperature").string() << "/temperature_" << std::setfill('0') << std::setw(4) << t << ".bin";
        std::ofstream o(fn.str(), std::ios::binary);
        const uint32_t c = static_cast<uint32_t>(n);
        o.write(reinterpret_cast<const char*>(&c), sizeof(c));
        o.write(reinterpret_cast<const char*>(T.data()), static_cast<std::streamsize>(n * sizeof(float)));
    }
}

static double summaryValue(const std::string& text, const std::string& key) {
    const size_t p = text.find(key);
    if (p == std::string::npos) return std::nan("");
    return std::stod(text.substr(p + key.size()));
}

struct E2E { bool ok = false; std::string err; double residual = NAN, max_rel = NAN, emitted = NAN; };
static E2E runEndToEnd(bool sputtering, bool consistent) {
    E2E r;
    char tmpl[] = "/tmp/test_superstep_deposition_XXXXXX";
    const char* dd = mkdtemp(tmpl);
    if (!dd) { r.err = "mkdtemp"; return r; }
    const fs::path root(dd);
    const ptm::Mesh mesh = make_icosphere(3, 1560.8);
    writeThermal(root / "thermal", mesh);
    auto integ = std::make_unique<exo::PTMExosphereIntegration>((root / "out").string());
    integ->setShowProgress(false);
    integ->setMesh(std::make_shared<ptm::Mesh>(mesh));
    integ->setSuperstepConsistentDeposition(consistent);
    std::streambuf* old = std::cout.rdbuf();
    std::streambuf* olde = std::cerr.rdbuf();
    std::ostringstream sink;
    std::cout.rdbuf(sink.rdbuf());
    std::cerr.rdbuf(sink.rdbuf());
    bool ok = integ->loadTemperatureData((root / "thermal").string(), kNt);
    integ->enableIceLayerModel(0.05);
    try {
        if (sputtering) {
            exo::SputteringConfig sc;
            sc.rate_scale = 50.0;
            integ->enableSputtering(sc);
        }
        if (ok) integ->runMultiTimescaleSimulation(2000, 100, 500);
    } catch (const std::exception& e) {
        r.err = e.what();
    }
    std::cout.rdbuf(old);
    std::cerr.rdbuf(olde);
    r.ok = ok && r.err.empty();
    std::ifstream in(root / "out" / "evolution_summary.txt");
    std::stringstream ss;
    ss << in.rdbuf();
    const std::string s = ss.str();
    r.residual = summaryValue(s, "Step residual (deposited - retained emission): ");
    const size_t p = s.find("; max per step ", s.find("Step residual"));
    if (p != std::string::npos) r.max_rel = std::stod(s.substr(p + 15));
    r.emitted = summaryValue(s, "Total sublimated: ") + summaryValue(s, "Total sputtered: ");
    fs::remove_all(root);
    return r;
}

static void test_end_to_end() {
    std::printf("\n(8) end to end: --multiscale on a synthetic ico3 body\n");
    for (int sp = 0; sp < 2; ++sp) {
        const std::string tag = sp ? "sputtering ON" : "sputtering OFF";
        const E2E n = runEndToEnd(sp == 1, true);
        const E2E o = runEndToEnd(sp == 1, false);
        check(n.ok, "new run completes (" + tag + ")" + (n.err.empty() ? "" : ": " + n.err));
        check(o.ok, "legacy run completes (" + tag + ")" + (o.err.empty() ? "" : ": " + o.err));
        check(std::isfinite(n.max_rel) && n.max_rel <= 1e-6, "new: summary max step residual <= 1e-6 (" + tag + ")",
              n.max_rel, 1e-6);
        check(std::isfinite(o.max_rel) && o.max_rel > 1e-6, "legacy: summary max step residual > 1e-6 (" + tag + ")",
              o.max_rel, 1e-6);
        check(std::isfinite(o.residual) && o.residual > 0.0, "legacy: cumulative deposition exceeds retained (" + tag + ")",
              o.residual / o.emitted, 0.0);
    }
}

// ---------------------------------------------------------------------------
// (9) PTM-640: with albedo coupling, a super-step ends when a face's albedo would
// pass the refresh threshold, so the refresh is never late by more than one epoch's
// drift, and the answer no longer depends on the super-step size.
static double kLimiterDT = -22.0;  // cooler body: one epoch moves the albedo by << threshold
struct LimRun { bool ok = false; std::string err; double max_dev = NAN, max_c = NAN, sub = NAN; };
static LimRun runLimiter(bool limit, int cap, int epochs) {
    LimRun r;
    char tmpl[] = "/tmp/test_superstep_limiter_XXXXXX";
    const char* dd = mkdtemp(tmpl);
    if (!dd) { r.err = "mkdtemp"; return r; }
    const fs::path root(dd);
    const ptm::Mesh mesh = make_icosphere(3, 1560.8);
    writeThermal(root / "thermal", mesh, kLimiterDT);
    auto integ = std::make_unique<exo::PTMExosphereIntegration>((root / "out").string());
    integ->setShowProgress(false);
    integ->setMesh(std::make_shared<ptm::Mesh>(mesh));
    integ->setAlbedoStepLimit(limit);
    if (cap > 0) integ->setMaxSuperstepEpochs(cap);
    std::streambuf* old = std::cout.rdbuf();
    std::streambuf* olde = std::cerr.rdbuf();
    std::ostringstream sink;
    std::cout.rdbuf(sink.rdbuf());
    std::cerr.rdbuf(sink.rdbuf());
    bool ok = integ->loadTemperatureData((root / "thermal").string(), kNt);
    integ->enableIceLayerModel(0.05);
    try {
        // Linear composition albedo 0.65 (ice) .. 0.10; c = 0.05 -> 0.6225 = thermal albedo.
        ok = ok && integ->setThermalAlbedoUniform(0.6225, "uniform 0.6225 (test)") &&
             integ->enableAlbedoCoupling(exo::CompositionAlbedoConfig());
        if (ok) integ->runMultiTimescaleSimulation(epochs, 100, 0);
    } catch (const std::exception& e) {
        r.err = e.what();
    }
    std::cout.rdbuf(old);
    std::cerr.rdbuf(olde);
    r.ok = ok && r.err.empty();
    std::ifstream in(root / "out" / "evolution_summary.txt");
    std::stringstream ss;
    ss << in.rdbuf();
    r.max_dev = summaryValue(ss.str(), "Max albedo deviation at a refresh: ");
    r.sub = summaryValue(ss.str(), "Total sublimated: ");
    double c = 0.0;
    for (size_t i = 0; i < mesh.faces.size(); ++i) c = std::max(c, 1.0 - integ->getIceAvailabilityFactor(static_cast<int>(i)));
    r.max_c = c;
    fs::remove_all(root);
    return r;
}

static void test_albedo_step_limiter() {
    std::printf("\n(9) albedo step limiter (PTM-640): refresh never late, step-size independent\n");
    const int E = 3000;
    const double thr = exo::CompositionAlbedoConfig().albedo_update_threshold;
    LimRun on0 = runLimiter(true, 0, E), on1 = runLimiter(true, 100, E), on2 = runLimiter(true, 25, E);
    LimRun off0 = runLimiter(false, 0, E), off2 = runLimiter(false, 25, E);
    for (const auto* r : {&on0, &on1, &on2, &off0, &off2}) {
        check(r->ok, "run completes" + (r->err.empty() ? "" : ": " + r->err));
    }
    std::printf("    limiter on:  max step none/100/25: max non-ice %.6f %.6f %.6f, max dev at refresh %.4e %.4e %.4e\n",
                on0.max_c, on1.max_c, on2.max_c, on0.max_dev, on1.max_dev, on2.max_dev);
    std::printf("    limiter off: max step none/25:     max non-ice %.6f %.6f, max dev at refresh %.4e %.4e\n",
                off0.max_c, off2.max_c, off0.max_dev, off2.max_dev);
    check(off0.max_dev > 2.0 * thr, "limiter off (control): a refresh fires far past the threshold", off0.max_dev / thr, 2.0);
    check(on0.max_dev < 1.5 * thr, "limiter on: every refresh fires within 1.5 x threshold", on0.max_dev / thr, 1.5);
    const double d_on = std::abs(on0.max_c - on2.max_c), d_off = std::abs(off0.max_c - off2.max_c);
    check(d_off > 1e-3, "limiter off (control): max non-ice depends on the super-step cap", d_off, 1e-3);
    check(d_on < 0.25 * d_off, "limiter on: cap dependence of max non-ice cut by > 4x", d_on / d_off, 0.25);
    check(std::abs(on0.sub / on2.sub - 1.0) < 0.01, "limiter on: cumulative sublimated agrees within 1% across caps",
          std::abs(on0.sub / on2.sub - 1.0), 0.01);
}

int main() {
    std::printf("=== test_superstep_deposition ===\n");
    test_exact_linear_mean();
    test_depleting(0.0, false);
    test_depleting(2.0, false);
    test_depleting(0.0, true);
    test_depleting(2.0, true);
    test_audit_throws();
    test_constant_availability();
    test_refinement();
    test_inventory();
    test_end_to_end();
    test_albedo_step_limiter();
    std::printf("\n%d/%d checks passed\n", g_total - g_fail, g_total);
    return g_fail == 0 ? 0 : 1;
}
