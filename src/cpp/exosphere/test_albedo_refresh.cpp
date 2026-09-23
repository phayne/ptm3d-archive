// Unit tests for the --albedo-coupling temperature refresh (albedo_refresh.h) and
// the thermal-run albedo reference in PTMExosphereIntegration.
// Self-contained; exit code 0 = all checks pass.
//
//  (a) slow drift below the threshold per pass fires a refresh once the
//      CUMULATIVE change since the last refresh passes the threshold;
//  (b) no drift is lost across refreshes: the final temperature equals the
//      rescale from the true thermal albedo to the final albedo;
//  (c) the t = 0 thermal-vs-composition mismatch warning fires (and does not
//      fire when the two agree), and coupling refuses without a thermal albedo;
//  (d) hapke-intimate honors a pinned non-ice endmember (--nonice-albedo) on a
//      uniform map.
//
// The legacy rule (compare to the previous pass, skip faces below the threshold)
// is emulated verbatim from ptm_exosphere_integration.cpp at 323cf11 (:1663 and
// :1686) to show each check discriminates.

#include "albedo_refresh.h"
#include "composition_albedo.h"
#include "ptm_exosphere_integration.h"
#include "../mesh/mesh_utils.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <unistd.h>

namespace exo = ptm::exosphere;

static int g_fail = 0, g_total = 0;
static void check(bool ok, const std::string& name, double value = 0.0, double limit = 0.0) {
    ++g_total;
    if (!ok) ++g_fail;
    std::printf("[%s] %-78s value=%.3e limit=%.1e\n", ok ? "PASS" : "FAIL", name.c_str(), value, limit);
    std::fflush(stdout);
}

// ---------------------------------------------------------------------------
// Synthetic drift: 4 faces, 3 timesteps.
// ---------------------------------------------------------------------------
static const std::vector<double> kThermal = {0.50, 0.60, 0.40, 0.55};
static const std::vector<double> kT0 = {100.0, 110.0, 120.0};

// Per-pass drift: face 0 darkens 0.004/pass (below the 0.01 threshold per pass),
// face 1 darkens 0.001/pass (never triggers alone), face 2 is fixed, face 3
// oscillates +/-0.003.
static std::vector<double> albedoAtPass(int p, double d0 = -0.004) {
    return {kThermal[0] + d0 * p, kThermal[1] - 0.001 * p, kThermal[2],
            kThermal[3] + ((p % 2) ? 0.003 : 0.0)};
}
static std::vector<std::vector<double>> freshTemperatures() {
    return std::vector<std::vector<double>>(kThermal.size(), kT0);
}

// Legacy (323cf11) per-pass rule, verbatim in effect: previous = last pass,
// trigger on the per-pass max change, rescale only faces above threshold.
struct LegacyRefresh {
    std::vector<double> previous, current;
    int fired = 0;
    void seed(const std::vector<double>& a) { previous = a; current = a; }
    void pass(const std::vector<double>& a, std::vector<std::vector<double>>& T, double thr) {
        previous = current;                                   // :1663
        current = a;
        const double m = exo::CompositionAlbedoMapper::maxAlbedoChange(previous, current);
        if (!(m > thr)) return;
        ++fired;
        for (size_t f = 0; f < current.size(); ++f) {
            if (std::abs(current[f] - previous[f]) < thr) continue;   // :1686
            const double s = std::pow((1.0 - current[f]) / (1.0 - previous[f]), 0.25);
            for (double& t : T[f]) t *= s;
        }
    }
};

static double maxRelErrVsThermalRescale(const std::vector<std::vector<double>>& T,
                                        const std::vector<double>& A_target) {
    double e = 0.0;
    for (size_t f = 0; f < T.size(); ++f) {
        const double s = exo::AlbedoRefreshTracker::scaleFactor(kThermal[f], A_target[f]);
        for (size_t t = 0; t < kT0.size(); ++t) {
            e = std::max(e, std::abs(T[f][t] / (kT0[t] * s) - 1.0));
        }
    }
    return e;
}

static void test_tracker() {
    const double thr = 0.01;

    // (a) first refresh fires at pass 3 (cumulative 0.012 > 0.01; 0.004 and 0.008 do not).
    {
        exo::AlbedoRefreshTracker tr;
        tr.seed(kThermal);
        auto T = freshTemperatures();
        int first = -1, n_fired = 0;
        for (int p = 1; p <= 30; ++p) {
            const auto r = tr.update(albedoAtPass(p), T, thr);
            if (r.fired) { ++n_fired; if (first < 0) first = p; }
        }
        check(first == 3, "(a) 0.004/pass drift: first refresh fires at pass 3 (cum 0.012 > 0.01)", first, 3);
        check(n_fired == 10, "(a) 30 passes of 0.004/pass: 10 refreshes (passes 3, 6, ..., 30)", n_fired, 10);

        LegacyRefresh lg;
        lg.seed(kThermal);
        auto TL = freshTemperatures();
        for (int p = 1; p <= 30; ++p) lg.pass(albedoAtPass(p), TL, thr);
        check(lg.fired == 0, "(a) discriminator: legacy per-pass rule never fires on the same drift",
              lg.fired, 0);

        // (b) pass 30 fired: T equals the rescale from the thermal albedo to A(30), every face.
        const double e = maxRelErrVsThermalRescale(T, albedoAtPass(30));
        check(e < 1e-6, "(b) final T == T_thermal * ((1-A_final)/(1-A_thermal))^1/4, all faces", e, 1e-6);
        double eb = 0.0;
        for (size_t f = 0; f < kThermal.size(); ++f) eb = std::max(eb, std::abs(tr.baseline()[f] - albedoAtPass(30)[f]));
        check(eb == 0.0, "(b) baseline == A_final on every face after the refresh", eb, 0.0);
        const double eL = maxRelErrVsThermalRescale(TL, albedoAtPass(30));
        check(eL > 1e-3, "(b) discriminator: legacy final T misses the thermal rescale (face 0 drift 0.12)",
              eL, 1e-3);
    }

    // (b) sub-threshold faces ride along: face 0 fires every pass (0.012/pass) while
    // face 1 drifts 0.001/pass. The legacy rule dropped face 1's drift for good.
    {
        exo::AlbedoRefreshTracker tr;
        tr.seed(kThermal);
        auto T = freshTemperatures();
        LegacyRefresh lg;
        lg.seed(kThermal);
        auto TL = freshTemperatures();
        for (int p = 1; p <= 20; ++p) {
            tr.update(albedoAtPass(p, -0.012), T, thr);
            lg.pass(albedoAtPass(p, -0.012), TL, thr);
        }
        const double e = maxRelErrVsThermalRescale(T, albedoAtPass(20, -0.012));
        check(e < 1e-6, "(b) every face corrected at each refresh (face 1 at 0.001/pass included)", e, 1e-6);
        const double eL = maxRelErrVsThermalRescale(TL, albedoAtPass(20, -0.012));
        check(eL > 1e-3, "(b) discriminator: legacy discards face 1's sub-threshold drift", eL, 1e-3);
    }

    // (b) between refreshes: T matches the baseline, and the pending drift is bounded.
    {
        exo::AlbedoRefreshTracker tr;
        tr.seed(kThermal);
        auto T = freshTemperatures();
        for (int p = 1; p <= 31; ++p) tr.update(albedoAtPass(p), T, thr);
        const double e = maxRelErrVsThermalRescale(T, tr.baseline());
        check(e < 1e-6, "(b) after a non-firing pass, T == thermal rescale to the last-refresh albedo", e, 1e-6);
        const double pending = tr.maxDeviation(albedoAtPass(31));
        check(pending <= thr, "(b) pending drift is kept (bounded by the threshold), not discarded", pending, thr);
        // One more refresh brings it home exactly.
        const auto r = tr.update(albedoAtPass(31), T, 0.0);
        const double e2 = maxRelErrVsThermalRescale(T, albedoAtPass(31));
        check(r.fired && e2 < 1e-6, "(b) forcing a refresh then reaches the thermal rescale to A_final", e2, 1e-6);
    }

    // Singular baseline is reported, not silently rescaled.
    {
        exo::AlbedoRefreshTracker tr;
        tr.seed({1.0, 0.5});
        std::vector<std::vector<double>> T = {{100.0}, {100.0}};
        const auto r = tr.update({0.9, 0.4}, T, 0.01);
        check(r.fired && r.faces_singular == 1 && r.faces_rescaled == 1 && T[0][0] == 100.0 &&
                  tr.baseline()[0] == 1.0,
              "singular baseline (A_ref = 1): face skipped, counted, baseline kept", double(r.faces_singular), 1);
    }
}

// ---------------------------------------------------------------------------
// Class-level checks on a km-unit ico4 (5120 faces).
// ---------------------------------------------------------------------------
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

// Run fn with std::cout captured; returns the captured text.
template <class F>
static std::string captureCout(F&& fn) {
    std::ostringstream buf;
    std::streambuf* old = std::cout.rdbuf(buf.rdbuf());
    fn();
    std::cout.rdbuf(old);
    return buf.str();
}

static void writeBin(const std::string& path, const std::vector<double>& a) {
    std::ofstream o(path, std::ios::binary);
    const uint64_t n = a.size();
    o.write(reinterpret_cast<const char*>(&n), sizeof(n));
    o.write(reinterpret_cast<const char*>(a.data()), static_cast<std::streamsize>(a.size() * sizeof(double)));
}

static void test_integration(const std::filesystem::path& root) {
    const ptm::Mesh mesh = make_icosphere(4, 1560.8);
    const size_t n = mesh.faces.size();

    auto make = [&](const std::string& tag) {
        auto integ = std::make_unique<exo::PTMExosphereIntegration>((root / tag).string());
        integ->setShowProgress(false);
        integ->setMesh(std::make_shared<ptm::Mesh>(mesh));
        integ->enableIceLayerModel(0.05);
        return integ;
    };

    // (c) refuse without a thermal albedo.
    {
        auto integ = make("refuse");
        bool ok = true;
        const std::string out = captureCout([&] { ok = integ->enableAlbedoCoupling(exo::CompositionAlbedoConfig()); });
        check(!ok && !integ->isAlbedoCouplingEnabled(),
              "(c) enableAlbedoCoupling refuses when no thermal-run albedo was declared");
        bool bad1 = integ->setThermalAlbedoUniform(1.0, "test");
        bool bad2 = integ->setThermalAlbedoUniform(std::numeric_limits<double>::quiet_NaN(), "test");
        check(!bad1 && !bad2, "(c) thermal albedo 1.0 and NaN are refused");
        writeBin((root / "short.bin").string(), std::vector<double>(n - 1, 0.5));
        check(!integ->loadThermalAlbedoMap((root / "short.bin").string()),
              "(c) thermal albedo map with the wrong face count is refused");
    }

    // (c) mismatch warning: thermal 0.5 vs linear-law composition 0.65*0.95 + 0.10*0.05 = 0.6225.
    {
        auto integ = make("mismatch");
        check(integ->setThermalAlbedoUniform(0.5, "uniform 0.5 (test)"), "(c) set thermal albedo 0.5");
        bool ok = false;
        const std::string out = captureCout([&] { ok = integ->enableAlbedoCoupling(exo::CompositionAlbedoConfig()); });
        const bool warned = out.find("WARNING: t = 0 albedo MISMATCH (thermal vs composition)") != std::string::npos;
        check(ok && warned, "(c) thermal 0.5 vs composition 0.6225: t = 0 mismatch WARNING printed");
        check(out.find(std::to_string(n) + " of " + std::to_string(n) + " faces") != std::string::npos,
              "(c) warning counts every face as mismatched", double(n), double(n));
        double eb = 0.0, ec = 0.0;
        for (size_t i = 0; i < n; ++i) {
            eb = std::max(eb, std::abs(integ->getAlbedoAtLastRefresh()[i] - 0.5));
            ec = std::max(ec, std::abs(integ->getFaceAlbedo(static_cast<int>(i)) - 0.6225));
        }
        check(eb == 0.0, "(c) temperature reference seeded from the THERMAL albedo (0.5), not composition", eb, 0.0);
        check(ec < 1e-12, "(c) composition albedo at t = 0 is 0.6225", ec, 1e-12);
    }
    {
        auto integ = make("match");
        integ->setThermalAlbedoUniform(0.6225, "uniform 0.6225 (test)");
        bool ok = false;
        const std::string out = captureCout([&] { ok = integ->enableAlbedoCoupling(exo::CompositionAlbedoConfig()); });
        check(ok && out.find("WARNING") == std::string::npos &&
                  out.find("t = 0 check: composition albedo matches") != std::string::npos,
              "(c) thermal 0.6225 == composition: no warning, match reported");
    }
    {
        // Just inside / outside the 1e-3 tolerance.
        auto a = make("tol_in");
        a->setThermalAlbedoUniform(0.6225 - 0.9e-3, "t");
        const std::string oa = captureCout([&] { a->enableAlbedoCoupling(exo::CompositionAlbedoConfig()); });
        auto b = make("tol_out");
        b->setThermalAlbedoUniform(0.6225 - 1.1e-3, "t");
        const std::string ob = captureCout([&] { b->enableAlbedoCoupling(exo::CompositionAlbedoConfig()); });
        check(oa.find("WARNING") == std::string::npos && ob.find("WARNING") != std::string::npos,
              "(c) warning tolerance is 1e-3 (0.9e-3 silent, 1.1e-3 warns)");
    }

    // Thermal map with an invalid value: refused where it is the temperature reference...
    {
        auto integ = make("badmap_linear");
        std::vector<double> m(n, 0.5);
        m[7] = -1.0;
        writeBin((root / "bad.bin").string(), m);
        bool ok = true;
        captureCout([&] {
            integ->loadThermalAlbedoMap((root / "bad.bin").string());
            ok = integ->enableAlbedoCoupling(exo::CompositionAlbedoConfig());
        });
        check(!ok, "(c) thermal map value -1 on a referenced face: coupling refused");
    }
    // ...but accepted on a hapke-intimate fill face, which is anchored and never uses it.
    {
        auto integ = make("badmap_hapke");
        std::vector<double> m(n, 0.6225);
        m[7] = -1.0;
        m[8] = std::numeric_limits<double>::quiet_NaN();
        const std::string map = (root / "fill.bin").string();
        writeBin(map, m);
        exo::HapkeIceFractionConfig cfg;
        cfg.albedo_map_path = map;
        cfg.fill_fits_path = "none";
        cfg.nonice_albedo = 0.3814;
        exo::CompositionAlbedoConfig acfg;
        bool ok1 = false, ok2 = false, ok3 = false;
        const std::string out = captureCout([&] {
            ok1 = integ->initializeConcentrationsFromHapkeIntimate(cfg, acfg);
            ok2 = integ->loadThermalAlbedoMap(map);
            ok3 = integ->enableAlbedoCoupling(acfg);
        });
        const auto& base = integ->getAlbedoAtLastRefresh();
        const bool anchored_ok = base.size() == n && std::isfinite(base[7]) && std::isfinite(base[8]) &&
                                 std::abs(base[7] - integ->getFaceAlbedo(7)) == 0.0;
        check(ok1 && ok2 && ok3 && anchored_ok &&
                  out.find("Anchored hapke-intimate faces (fill or clamped): 2 of") != std::string::npos,
              "(c) hapke fill faces with -1/NaN: accepted, anchored at composition albedo");
    }

    // (d) hapke-intimate, uniform map 0.6225, pinned non-ice endmember 0.3814.
    {
        auto integ = make("hapke_pinned");
        const std::string map = (root / "uniform_0.6225.bin").string();
        writeBin(map, std::vector<double>(n, 0.6225));
        exo::HapkeIceFractionConfig cfg;
        cfg.albedo_map_path = map;
        cfg.fill_fits_path = "none";
        cfg.nonice_albedo = 0.3814;
        exo::CompositionAlbedoConfig acfg;
        bool ok = false;
        const std::string out = captureCout([&] { ok = integ->initializeConcentrationsFromHapkeIntimate(cfg, acfg); });
        check(ok, "(d) hapke-intimate init on a uniform map with --nonice-albedo 0.3814");
        const double e_non = std::abs(acfg.hapke_mixture.bond(0.0) - 0.3814);
        check(e_non < 1e-9, "(d) pinned non-ice endmember used: A_B(f = 0) == 0.3814", e_non, 1e-9);
        check(out.find("(explicit --nonice-albedo)") != std::string::npos,
              "(d) banner reports the explicit endmember, not a map percentile");
        integ->loadThermalAlbedoMap(map);
        const std::string out2 = captureCout([&] { ok = integ->enableAlbedoCoupling(acfg); });
        double ec = 0.0;
        for (size_t i = 0; i < n; ++i) ec = std::max(ec, std::abs(integ->getFaceAlbedo(static_cast<int>(i)) - 0.6225));
        check(ok && ec < 1e-6, "(d) forward model reproduces the uniform map at t = 0", ec, 1e-6);
        check(out2.find("WARNING") == std::string::npos && out2.find("Anchored") == std::string::npos,
              "(d) no mismatch and no anchored faces (every face inverted)");

        // Without the pin the endmember is the map's P10 = 0.6225 itself.
        auto integ2 = make("hapke_p10");
        exo::HapkeIceFractionConfig cfg2 = cfg;
        cfg2.nonice_albedo = -1.0;
        exo::CompositionAlbedoConfig acfg2;
        bool ok2 = false;
        captureCout([&] { ok2 = integ2->initializeConcentrationsFromHapkeIntimate(cfg2, acfg2); });
        const double a_non2 = ok2 ? acfg2.hapke_mixture.bond(0.0) : 0.6225;
        check(ok2 && std::abs(a_non2 - 0.6225) < 1e-9,
              "(d) contrast: unpinned, a uniform map's P10 endmember is the map value (degenerate)",
              std::abs(a_non2 - 0.6225), 1e-9);
    }
}

int main() {
    std::printf("=== test_albedo_refresh ===\n");
    test_tracker();
    char tmpl[] = "/tmp/test_albedo_refresh_XXXXXX";
    const char* d = mkdtemp(tmpl);
    if (!d) { std::printf("mkdtemp failed\n"); return 2; }
    const std::filesystem::path root(d);
    test_integration(root);
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
    std::printf("\n%d / %d checks passed\n", g_total - g_fail, g_total);
    return g_fail == 0 ? 0 : 1;
}
