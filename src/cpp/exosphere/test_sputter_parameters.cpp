// Unit tests for the sputtering parameter set (PTM-649). Self-contained; exit
// code 0 = all checks pass.
//
//  (1) e_H2O and T_launch both derive from the binding energy U (Sigmund-Thompson
//      f(E) ~ E / (E + U)^3): closed forms against direct quadrature for several U;
//      U = 0.055 eV gives 0.2354 and 683 K; E_esc matches the exosphere's v_esc.
//  (2) tau(T) is continuous and strictly monotonic from 20 to 300 K, equals 1 at
//      T_ref, has no cutoff, and applies to the radiolytic part only (the H2O part
//      is temperature-independent). The legacy factor (control) has the 50 K
//      cutoff and the 80 K cliff.
//  (3) Renormalization: on a mesh with unequal face areas, pure ice at T_ref
//      delivers the target total (2e27 H2O/s x m_H2O x rate_scale) to 1e-12; the
//      legacy count normalization over a fixed 3.09e13 m^2 misses it (control).
//  (4) Absolute ion map: the map sets pattern and total (molecules cm^-2 s^-1).
//  (5) Ion-map plug-in: feeding the default weights (at any scale) reproduces the
//      default per-face fluxes.
//  (6) Fates: per face, gross = H2O + O2 + H2 = redeposited + escaped + O2 loss;
//      O2 is 100% atmospheric loss, H2 100% escape.
//  (7) End to end (ico3, --multiscale, sputtering with an absolute map): the run
//      completes (the sputter and step audits throw otherwise) and the realized
//      O2 production equals the per-face O2 of the diurnal mean.

#include "exosphere_physics.h"
#include "multitimescale_evolution.h"
#include "ptm_exosphere_integration.h"
#include "sputtering_model.h"
#include "../mesh/mesh_utils.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <regex>
#include <sstream>
#include <string>
#include <vector>
#include <unistd.h>

namespace exo = ptm::exosphere;
namespace fs = std::filesystem;
namespace sc = ptm::exosphere::sputtering_constants;

static int g_fail = 0, g_total = 0;
static void check(bool ok, const std::string& name, double value = 0.0, double limit = 0.0) {
    ++g_total;
    if (!ok) ++g_fail;
    std::printf("[%s] %-84s value=%.4e limit=%.1e\n", ok ? "PASS" : "FAIL", name.c_str(), value, limit);
    std::fflush(stdout);
}

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

struct Geo {
    std::vector<std::array<double, 3>> centers;
    std::vector<double> areas, lat, lon;
};
static Geo geometry(const ptm::Mesh& m) {
    Geo g;
    for (const auto& tri : m.faces) {
        const auto& a = m.vertices[tri[0]];
        const auto& b = m.vertices[tri[1]];
        const auto& c = m.vertices[tri[2]];
        std::array<double, 3> ctr{(a[0] + b[0] + c[0]) / 3, (a[1] + b[1] + c[1]) / 3, (a[2] + b[2] + c[2]) / 3};
        g.centers.push_back(ctr);
        const double r = std::sqrt(ctr[0] * ctr[0] + ctr[1] * ctr[1] + ctr[2] * ctr[2]);
        g.lat.push_back(std::asin(ctr[2] / r) * 180.0 / M_PI);
        g.lon.push_back(std::atan2(ctr[1], ctr[0]) * 180.0 / M_PI);
        const double e1[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
        const double e2[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
        const double cx = e1[1] * e2[2] - e1[2] * e2[1];
        const double cy = e1[2] * e2[0] - e1[0] * e2[2];
        const double cz = e1[0] * e2[1] - e1[1] * e2[0];
        g.areas.push_back(0.5 * std::sqrt(cx * cx + cy * cy + cz * cz));
    }
    return g;
}

static double sumA(const std::vector<double>& x, const std::vector<double>& a) {
    double s = 0.0;
    for (size_t i = 0; i < x.size(); ++i) s += x[i] * a[i];
    return s;
}

// ---------------------------------------------------------------------------
static void test_sigmund_thompson() {
    std::printf("\n(1) e_H2O and T_launch derive from U (Sigmund-Thompson)\n");
    const double Ee = exo::h2oEscapeEnergyEV();
    const double Ee_phys = 0.5 * exo::constants::WATER_MOLECULAR_MASS * exo::constants::EUROPA_ESCAPE_VELOCITY *
                           exo::constants::EUROPA_ESCAPE_VELOCITY / sc::JOULE_PER_EV;
    check(std::abs(Ee / Ee_phys - 1.0) < 1e-12, "E_esc uses the exosphere's v_esc and m_H2O (0.3828 eV)", Ee, 0.3828);
    for (double U : {0.02, 0.055, 0.1, 0.3}) {
        // Direct quadrature of f(E) = E/(E+U)^3 on a mapped grid (E = U t/(1-t)).
        const int N = 400000;
        double tot = 0.0, above = 0.0, m1_bound = 0.0, m0_bound = 0.0;
        for (int k = 0; k < N; ++k) {
            const double t = (k + 0.5) / N;
            const double E = U * t / (1.0 - t), dE = U / ((1.0 - t) * (1.0 - t)) / N;
            const double w = E / std::pow(E + U, 3) * dE;
            tot += w;
            if (E >= Ee) above += w;
            else { m0_bound += w; m1_bound += E * w; }
        }
        const double e_q = above / tot, mean_q = m1_bound / m0_bound;
        const double e_c = exo::sigmundThompsonEscapeFraction(U, Ee);
        const double mean_c = exo::sigmundThompsonMeanBoundEnergy(U, Ee);
        char tag[96];
        std::snprintf(tag, sizeof tag, "U = %.3f eV: escape fraction closed form = quadrature", U);
        check(std::abs(e_c - e_q) < 2e-5, tag, std::abs(e_c - e_q), 2e-5);
        std::snprintf(tag, sizeof tag, "U = %.3f eV: bound mean energy closed form = quadrature", U);
        check(std::abs(mean_c / mean_q - 1.0) < 2e-4, tag, std::abs(mean_c / mean_q - 1.0), 2e-4);
    }
    exo::SputteringConfig c;
    check(std::abs(c.h2oEscapeFraction() - 0.2354) < 1e-4, "U = 0.055 eV: e_H2O = 0.2354", c.h2oEscapeFraction(), 0.2354);
    check(std::abs(c.launchTemperatureK() - 683.0) < 1.0, "U = 0.055 eV: T_launch = 683 K", c.launchTemperatureK(), 683.0);
    exo::SputteringConfig c2;
    c2.binding_energy_eV = 0.1;
    check(c2.h2oEscapeFraction() > c.h2oEscapeFraction() && c2.launchTemperatureK() > c.launchTemperatureK(),
          "raising U raises BOTH e_H2O and T_launch (one parameter, no disagreement)",
          c2.h2oEscapeFraction() - c.h2oEscapeFraction(), 0.0);
    exo::SputteringConfig L;
    L.legacy_params = true;
    L.applyLegacyParams();
    check(L.h2oEscapeFraction() == 0.1 && L.launchTemperatureK() == 683.0,
          "legacy params keep the old 0.10 and 683 K", L.h2oEscapeFraction(), 0.1);
}

static void test_tau() {
    std::printf("\n(2) tau(T): continuous, monotonic, radiolytic part only\n");
    exo::SputteringConfig c;
    check(std::abs(c.radiolyticFactor(c.radiolysis_reference_K) - 1.0) < 1e-15, "tau(T_ref) = 1",
          c.radiolyticFactor(c.radiolysis_reference_K), 1.0);
    double prev = c.radiolyticFactor(20.0), worst_jump = 0.0;
    bool mono = prev > 0.0;
    for (double T = 20.01; T <= 300.0; T += 0.01) {
        const double v = c.radiolyticFactor(T);
        mono = mono && (v > prev);
        // Continuity: relative step bounded by (E_a/k) dT / T^2 (+ rounding).
        const double bound = (c.radiolysis_activation_eV / sc::BOLTZMANN_EV) * 0.01 / (T * T) * 1.01 + 1e-12;
        worst_jump = std::max(worst_jump, std::abs(v / prev - 1.0) / bound);
        prev = v;
    }
    check(mono, "tau strictly increasing and > 0 from 20 to 300 K (no cutoff)", c.radiolyticFactor(20.0), 0.0);
    check(worst_jump <= 1.0, "no jump: every 0.01 K step within the analytic derivative bound", worst_jump, 1.0);

    // Radiolytic part only: H2O part independent of T; radiolytic part scales with tau.
    const ptm::Mesh m = make_icosphere(2, 1560.8e3);
    const Geo g = geometry(m);
    const size_t n = g.areas.size();
    exo::SputteringModel sm(n, c);
    sm.initializeSpatialDistribution(g.centers, g.lat, g.lon, g.areas);
    exo::SputteringState a(n), b(n);
    sm.computeSputteringRates(std::vector<double>(n, 90.0), g.areas, a);
    sm.computeSputteringRates(std::vector<double>(n, 130.0), g.areas, b);
    double worst_h2o = 0.0, worst_rad = 0.0;
    const double ratio = c.radiolyticFactor(130.0) / c.radiolyticFactor(90.0);
    for (size_t i = 0; i < n; ++i) {
        worst_h2o = std::max(worst_h2o, std::abs(b.h2o_sputter_flux[i] / a.h2o_sputter_flux[i] - 1.0));
        const double ra = a.o2_production_flux[i] + a.h2_production_flux[i];
        const double rb = b.o2_production_flux[i] + b.h2_production_flux[i];
        worst_rad = std::max(worst_rad, std::abs(rb / ra / ratio - 1.0));
    }
    check(worst_h2o < 1e-15, "H2O yield is temperature-independent (90 K vs 130 K)", worst_h2o, 1e-15);
    check(worst_rad < 1e-13, "radiolytic O2 + H2 scales with tau(130)/tau(90)", worst_rad, 1e-13);
    // 30 K: still sputtering (no cutoff).
    exo::SputteringState cold(n);
    sm.computeSputteringRates(std::vector<double>(n, 30.0), g.areas, cold);
    check(sumA(cold.h2o_sputter_flux, g.areas) > 0.0, "30 K: H2O is still sputtered (no 50 K cutoff)",
          sumA(cold.h2o_sputter_flux, g.areas), 0.0);

    // Legacy control: 0 below 50 K, 1 from 50-80 K, a cliff at 80 K.
    exo::SputteringConfig L;
    L.legacy_params = true;
    L.applyLegacyParams();
    exo::SputteringModel sl(n, L);
    sl.initializeSpatialDistribution(g.centers, g.lat, g.lon, g.areas);
    exo::SputteringState s49(n), s79(n), s81(n);
    sl.computeSputteringRates(std::vector<double>(n, 49.9), g.areas, s49);
    sl.computeSputteringRates(std::vector<double>(n, 79.9), g.areas, s79);
    sl.computeSputteringRates(std::vector<double>(n, 80.1), g.areas, s81);
    check(s49.sputtering_yield[0] == 0.0 && s79.sputtering_yield[0] == 1.0 && s81.sputtering_yield[0] < 0.61,
          "legacy (control): 0 below 50 K, 1.0 at 79.9 K, ~0.60 at 80.1 K (the cliff)", s81.sputtering_yield[0], 0.61);
}

static void test_renormalization() {
    std::printf("\n(3) renormalization: pure ice at T_ref delivers the target total exactly\n");
    const ptm::Mesh m = make_icosphere(3, 1560.8e3);
    const Geo g = geometry(m);
    const size_t n = g.areas.size();
    const double amin = *std::min_element(g.areas.begin(), g.areas.end());
    const double amax = *std::max_element(g.areas.begin(), g.areas.end());
    check(amax / amin > 1.2, "test mesh has unequal face areas", amax / amin, 1.2);
    for (double rs : {1.0, 3.7}) {
        exo::SputteringConfig c;
        c.rate_scale = rs;
        exo::SputteringModel sm(n, c);
        sm.initializeSpatialDistribution(g.centers, g.lat, g.lon, g.areas);
        exo::SputteringState st(n);
        sm.computeSputteringRates(std::vector<double>(n, c.radiolysis_reference_K), g.areas, st);
        const double target = sc::GLOBAL_H2O_RATE * sc::H2O_MASS * rs;
        const double got = sumA(st.gross_removal_flux, g.areas);
        char tag[96];
        std::snprintf(tag, sizeof tag, "rate_scale %.1f: sum(gross x area) = 2e27 x m_H2O x rate_scale", rs);
        check(std::abs(got / target - 1.0) < 1e-12, tag, std::abs(got / target - 1.0), 1e-12);
        check(std::abs(sumA(st.h2o_sputter_flux, g.areas) / (c.h2o_fraction * target) - 1.0) < 1e-12,
              "H2O part = f_H2O x target (Cassidy split 0.90)",
              sumA(st.h2o_sputter_flux, g.areas) / target, c.h2o_fraction);
        double wmean = 0.0;
        for (size_t i = 0; i < n; ++i) wmean += sm.getSpatialWeight(i) * g.areas[i];
        wmean /= sm.meshArea();
        check(std::abs(wmean - 1.0) < 1e-12, "ion weight: area-weighted mean = 1", std::abs(wmean - 1.0), 1e-12);
    }
    // Legacy control: count normalization over 3.09e13 m^2 does not give the target.
    exo::SputteringConfig L;
    L.legacy_params = true;
    L.applyLegacyParams();
    exo::SputteringModel sl(n, L);
    sl.initializeSpatialDistribution(g.centers, g.lat, g.lon, g.areas);
    exo::SputteringState st(n);
    sl.computeSputteringRates(std::vector<double>(n, 100.0), g.areas, st);
    const double got = sumA(st.gross_removal_flux, g.areas);
    const double target = sc::GLOBAL_H2O_RATE * sc::H2O_MASS;
    check(std::abs(got / target - 1.0) > 1e-4, "legacy (control): count normalization misses the target",
          got / target - 1.0, 1e-4);
}

static void test_absolute_map() {
    std::printf("\n(4) absolute ion map sets pattern and total\n");
    const ptm::Mesh m = make_icosphere(3, 1560.8e3);
    const Geo g = geometry(m);
    const size_t n = g.areas.size();
    std::vector<double> R(n);
    for (size_t i = 0; i < n; ++i) R[i] = 1e11 * (1.0 + 0.5 * std::cos(g.lon[i] * M_PI / 180.0));  // cm^-2 s^-1
    exo::SputteringConfig c;
    exo::SputteringModel sm(n, c);
    sm.setIonMap(R, true, "synthetic absolute");
    sm.initializeSpatialDistribution(g.centers, g.lat, g.lon, g.areas);
    exo::SputteringState st(n);
    sm.computeSputteringRates(std::vector<double>(n, c.radiolysis_reference_K), g.areas, st);
    const double expect = sumA(R, g.areas) * 1e4;  // molecules/s
    check(std::abs(sm.ionTotalRate() / expect - 1.0) < 1e-14, "absolute map: total = sum(R x area x 1e4)",
          sm.ionTotalRate() / expect - 1.0, 1e-14);
    double worst = 0.0;
    for (size_t i = 0; i < n; ++i) {
        worst = std::max(worst, std::abs(st.gross_removal_flux[i] / (R[i] * 1e4 * sc::H2O_MASS) - 1.0));
    }
    check(worst < 1e-14, "absolute map: per-face gross at T_ref = R x 1e4 x m_H2O", worst, 1e-14);
}

static void test_plugin_reproduces_default() {
    std::printf("\n(5) ion-map plug-in fed the default weights reproduces the default\n");
    const ptm::Mesh m = make_icosphere(3, 1560.8e3);
    const Geo g = geometry(m);
    const size_t n = g.areas.size();
    exo::SputteringConfig c;
    exo::SputteringModel def(n, c);
    def.initializeSpatialDistribution(g.centers, g.lat, g.lon, g.areas);
    std::vector<double> T(n);
    for (size_t i = 0; i < n; ++i) T[i] = 80.0 + 60.0 * std::abs(std::cos(g.lat[i] * M_PI / 180.0));
    exo::SputteringState s0(n);
    def.computeSputteringRates(T, g.areas, s0);
    for (double scale : {1.0, 7.3}) {
        std::vector<double> w(n);
        for (size_t i = 0; i < n; ++i) w[i] = scale * def.getSpatialWeight(i);
        exo::SputteringModel pl(n, c);
        pl.setIonMap(w, false, "default weights");
        pl.initializeSpatialDistribution(g.centers, g.lat, g.lon, g.areas);
        exo::SputteringState s1(n);
        pl.computeSputteringRates(T, g.areas, s1);
        double worst = 0.0;
        for (size_t i = 0; i < n; ++i) {
            worst = std::max(worst, std::abs(s1.gross_removal_flux[i] / s0.gross_removal_flux[i] - 1.0));
        }
        char tag[96];
        std::snprintf(tag, sizeof tag, "weights x %.1f: per-face gross matches the default", scale);
        check(worst < 1e-13, tag, worst, 1e-13);
    }
    bool threw = false;
    try {
        exo::SputteringModel bad(n, c);
        bad.setIonMap(std::vector<double>(n - 1, 1.0), false, "short");
        bad.initializeSpatialDistribution(g.centers, g.lat, g.lon, g.areas);
    } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "a map with the wrong face count is refused");
    threw = false;
    try {
        std::vector<double> neg(n, 1.0);
        neg[3] = -1.0;
        exo::SputteringModel bad(n, c);
        bad.setIonMap(neg, false, "negative");
        bad.initializeSpatialDistribution(g.centers, g.lat, g.lon, g.areas);
    } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "a negative weight is refused");
    threw = false;
    try {
        exo::SputteringModel bad(n, c);
        bad.initializeSpatialDistribution(g.centers, g.lat, g.lon);  // no areas
    } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "the current set refuses to normalize without face areas");
}

static void test_fates() {
    std::printf("\n(6) fates: O2 = atmospheric loss, H2 escapes, closure per face\n");
    const ptm::Mesh m = make_icosphere(2, 1560.8e3);
    const Geo g = geometry(m);
    const size_t n = g.areas.size();
    exo::SputteringConfig c;
    exo::SputteringModel sm(n, c);
    sm.initializeSpatialDistribution(g.centers, g.lat, g.lon, g.areas);
    std::vector<double> T(n);
    for (size_t i = 0; i < n; ++i) T[i] = 40.0 + 100.0 * (i % 11) / 10.0;
    exo::SputteringState st(n);
    sm.computeSputteringRates(T, g.areas, st);
    double w_prod = 0.0, w_fate = 0.0, w_o2 = 0.0, w_ratio = 0.0;
    const double stoich = sc::O2_MASS / (2.0 * sc::H2_MASS);
    const double e = c.h2oEscapeFraction();
    for (size_t i = 0; i < n; ++i) {
        const double G = st.gross_removal_flux[i];
        w_prod = std::max(w_prod, std::abs(st.h2o_sputter_flux[i] + st.o2_production_flux[i] + st.h2_production_flux[i] - G) / G);
        w_fate = std::max(w_fate, std::abs(st.redeposition_flux[i] + st.total_mass_loss[i] + st.reservoir_flux[i] - G) / G);
        w_o2 = std::max(w_o2, std::abs(st.reservoir_flux[i] - st.o2_production_flux[i]) / G +
                              std::abs(st.total_mass_loss[i] - e * st.h2o_sputter_flux[i] - st.h2_production_flux[i]) / G);
        w_ratio = std::max(w_ratio, std::abs(st.o2_production_flux[i] / st.h2_production_flux[i] / stoich - 1.0));
    }
    check(w_prod < 1e-14, "per face: H2O + O2 + H2 = gross", w_prod, 1e-14);
    check(w_fate < 1e-14, "per face: redeposited + escaped + atmospheric loss = gross", w_fate, 1e-14);
    check(w_o2 < 1e-14, "O2 is 100% atmospheric loss; escape = e_H2O x H2O + all H2", w_o2, 1e-14);
    check(w_ratio < 1e-12, "O2 : H2 stays stoichiometric at every temperature", w_ratio, 1e-12);
    check(std::abs(c.transportShare() + c.escapeShare() + c.reservoirShare() - 1.0) < 1e-15,
          "global shares at T_ref sum to 1 (0.6881 / 0.2231 / 0.0888)", c.transportShare(), 0.6881);
}

// ---------------------------------------------------------------------------
static const int kNt = 8;
static void writeThermal(const fs::path& dir, const ptm::Mesh& mesh) {
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
            T[f] = static_cast<float>(150.0 - 70.0 * mu + 8.0 * std::cos(2.0 * M_PI * t / kNt + 0.37 * f));
        }
        std::ostringstream fn;
        fn << (dir / "temperature").string() << "/temperature_" << std::setfill('0') << std::setw(4) << t << ".bin";
        std::ofstream o(fn.str(), std::ios::binary);
        const uint32_t c = static_cast<uint32_t>(n);
        o.write(reinterpret_cast<const char*>(&c), sizeof(c));
        o.write(reinterpret_cast<const char*>(T.data()), static_cast<std::streamsize>(n * sizeof(float)));
    }
}

static void test_end_to_end() {
    std::printf("\n(7) end to end: --multiscale, sputtering with an absolute ion map (ico3)\n");
    char tmpl[] = "/tmp/test_sputter_parameters_XXXXXX";
    const char* dd = mkdtemp(tmpl);
    if (!dd) { check(false, "mkdtemp"); return; }
    const fs::path root(dd);
    const ptm::Mesh mesh = make_icosphere(3, 1560.8);
    writeThermal(root / "thermal", mesh);
    const size_t n = mesh.faces.size();
    // Absolute map with a trailing-apex patch: 50x Cassidy's mean flux there.
    std::vector<double> R(n);
    const double mean_cm = sc::GLOBAL_H2O_RATE / 3.06e13 / 1e4;
    for (size_t f = 0; f < n; ++f) {
        const auto& fc = mesh.faces[f];
        double c[3] = {0.0, 0.0, 0.0};
        for (int k = 0; k < 3; ++k)
            for (int j = 0; j < 3; ++j) c[j] += mesh.vertices[fc[k]][j] / 3.0;
        const double lon = std::atan2(c[1], c[0]);
        R[f] = mean_cm * (5.0 + 45.0 * std::max(0.0, std::cos(lon + M_PI / 2.0)));
    }
    const fs::path map = root / "abs_map.bin";
    {
        std::ofstream o(map, std::ios::binary);
        const uint64_t c = n;
        o.write(reinterpret_cast<const char*>(&c), sizeof(c));
        o.write(reinterpret_cast<const char*>(R.data()), static_cast<std::streamsize>(n * sizeof(double)));
    }
    auto integ = std::make_unique<exo::PTMExosphereIntegration>((root / "out").string());
    integ->setShowProgress(false);
    integ->setMesh(std::make_shared<ptm::Mesh>(mesh));
    std::streambuf* old = std::cout.rdbuf();
    std::ostringstream sink;
    std::cout.rdbuf(sink.rdbuf());
    bool ok = integ->loadTemperatureData((root / "thermal").string(), kNt);
    integ->enableIceLayerModel(0.05);
    exo::SputteringConfig cfg;
    cfg.ion_map_path = map.string();
    cfg.ion_map_absolute = true;
    std::string err;
    try {
        integ->enableSputtering(cfg);
        if (ok) integ->runMultiTimescaleSimulation(1000, 100, 500);
    } catch (const std::exception& e) {
        err = e.what();
    }
    std::cout.rdbuf(old);
    check(ok && err.empty(), "run completes (sputter and step audits throw otherwise)" + (err.empty() ? "" : ": " + err));
    const std::string log = sink.str();
    const std::regex re("\\[sputter-rates\\] [^:]*: realized gross ([0-9.e+-]+) kg/s = ([0-9.]+) x the target "
                        "([0-9.e+-]+) kg/s .*O2 production ([0-9.e+-]+) kg/s");
    int lines = 0;
    double target = 0.0;
    for (auto it = std::sregex_iterator(log.begin(), log.end(), re); it != std::sregex_iterator(); ++it) {
        target = std::stod((*it)[3]);
        ++lines;
    }
    check(lines >= 2, "realized-gross and O2 lines were logged", lines, 2);
    // Target = the absolute map's own total, not Cassidy's.
    // (The engine rescales the mesh to meters; recompute the map total with its areas.)
    const double m_scale = 1000.0;
    double map_total = 0.0;
    for (size_t f = 0; f < n; ++f) {
        const auto& t = mesh.faces[f];
        const auto& a = mesh.vertices[t[0]];
        const auto& b = mesh.vertices[t[1]];
        const auto& c = mesh.vertices[t[2]];
        const double e1[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
        const double e2[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
        const double cx = e1[1] * e2[2] - e1[2] * e2[1], cy = e1[2] * e2[0] - e1[0] * e2[2],
                     cz = e1[0] * e2[1] - e1[1] * e2[0];
        map_total += R[f] * 1e4 * 0.5 * std::sqrt(cx * cx + cy * cy + cz * cz) * m_scale * m_scale;
    }
    check(std::abs(target / (map_total * sc::H2O_MASS) - 1.0) < 1e-5,
          "logged target = the absolute map's total x m_H2O (printed to 7 digits)",
          std::abs(target / (map_total * sc::H2O_MASS) - 1.0), 1e-5);
    fs::remove_all(root);
}

int main() {
    std::printf("=== test_sputter_parameters ===\n");
    test_sigmund_thompson();
    test_tau();
    test_renormalization();
    test_absolute_map();
    test_plugin_reproduces_default();
    test_fates();
    test_end_to_end();
    std::printf("\n%d/%d checks passed\n", g_total - g_fail, g_total);
    return g_fail == 0 ? 0 : 1;
}
