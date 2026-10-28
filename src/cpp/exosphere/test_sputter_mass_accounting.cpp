// Unit tests for the sputter mass accounting and the transport-kernel
// options it relies on. Self-contained; exit code 0 = all checks pass.
//
//  (1) Stoichiometry: radiolytic O2 : H2 mass ratio = m_O2 / (2 m_H2) (7.94, 8:1
//      nominal); product fractions and fate shares each sum to 1. The legacy split
//      (0.10 / 0.05, ratio 2) is kept only behind legacy_escape_only.
//  (2) Per face: gross = H2O + O2 + H2 and gross = escaped + transported + reservoir.
//  (3) Kernel: a bound population (Jeans escape off) deposits exactly its source,
//      even at the 683 K sputter launch temperature; with Jeans on it would lose
//      mass a second time. A hot call after a cold one reaches the antipode only
//      with a call-local index (the persistent index keeps its first radius).
//  (4) Deposition rescale (c): after availability changes that hit sublimation and
//      sputtering differently, deposition <= sublimation + sputter transport source
//      to 1e-6, and deposition from sublimation <= sublimation. The old single
//      (sublimation + sputtering) rescale is run on the same case and shown to
//      violate it.
//  (5) End to end (ico3 synthetic body, --multiscale, sputtering, CPU): every audit
//      line satisfies gross = redeposited + escaped + reservoir to 1e-6 and the
//      cumulative decomposition in evolution_summary.txt closes.

#include "exosphere_physics.h"
#include "multitimescale_evolution.h"
#include "ptm_exosphere_integration.h"
#include "sputtering_model.h"
#include "../mesh/mesh_utils.h"

#include <algorithm>
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

static int g_fail = 0, g_total = 0;
static void check(bool ok, const std::string& name, double value = 0.0, double limit = 0.0) {
    ++g_total;
    if (!ok) ++g_fail;
    std::printf("[%s] %-82s value=%.3e limit=%.1e\n", ok ? "PASS" : "FAIL", name.c_str(), value, limit);
    std::fflush(stdout);
}

// Same construction as test_multiscale_snapshot_time.cpp.
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
    std::vector<std::array<double, 3>> centers;  // meters, on the sphere
    std::vector<double> areas;                   // m^2 (flat triangles)
};
static Geo geometry(const ptm::Mesh& m, double R) {
    Geo g;
    for (const auto& tri : m.faces) {
        const auto& a = m.vertices[tri[0]];
        const auto& b = m.vertices[tri[1]];
        const auto& c = m.vertices[tri[2]];
        std::array<double, 3> ctr{(a[0] + b[0] + c[0]) / 3, (a[1] + b[1] + c[1]) / 3, (a[2] + b[2] + c[2]) / 3};
        const double r = std::sqrt(ctr[0] * ctr[0] + ctr[1] * ctr[1] + ctr[2] * ctr[2]);
        for (int d = 0; d < 3; ++d) ctr[d] *= R / r;
        g.centers.push_back(ctr);
        const double e1[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
        const double e2[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
        const double cx = e1[1] * e2[2] - e1[2] * e2[1];
        const double cy = e1[2] * e2[0] - e1[0] * e2[2];
        const double cz = e1[0] * e2[1] - e1[1] * e2[0];
        g.areas.push_back(0.5 * std::sqrt(cx * cx + cy * cy + cz * cz));
    }
    return g;
}

// ---------------------------------------------------------------------------
static void test_stoichiometry() {
    std::printf("\n(1) stoichiometry and share closure\n");
    exo::SputteringConfig c;
    const double ratio = c.o2Fraction() / c.h2Fraction();
    const double expect = exo::sputtering_constants::O2_MASS / (2.0 * exo::sputtering_constants::H2_MASS);
    check(std::abs(ratio / expect - 1.0) < 1e-12, "O2:H2 mass ratio = m_O2 / (2 m_H2)", ratio, expect);
    check(std::abs(ratio / 8.0 - 1.0) < 0.01, "O2:H2 mass ratio within 1% of the nominal 8:1", ratio, 8.0);
    const double sum_products = c.h2o_fraction + c.o2Fraction() + c.h2Fraction();
    check(std::abs(sum_products - 1.0) < 1e-15, "H2O + O2 + H2 fractions sum to 1", std::abs(sum_products - 1.0), 1e-15);
    const double sum_fates = c.transportShare() + c.escapeShare() + c.reservoirShare();
    check(std::abs(sum_fates - 1.0) < 1e-15, "transport + escape + reservoir shares sum to 1", std::abs(sum_fates - 1.0), 1e-15);
    check(std::abs(c.escapeShare() + c.reservoirShare() - (1.0 - c.h2o_fraction * (1.0 - c.h2oEscapeFraction()))) < 1e-15,
          "surface loss share = 1 - f_H2O (1 - e_H2O) (current set, at T_ref: 0.3119)",
          c.escapeShare() + c.reservoirShare(), 0.3119);
    {
        exo::SputteringConfig lp;  // legacy sputtering parameter set (the pre-2026-09-23 defaults)
        lp.legacy_params = true;
        lp.applyLegacyParams();
        check(std::abs(lp.escapeShare() + lp.reservoirShare() - (1.0 - 0.85 * 0.9)) < 1e-15,
              "legacy params: surface loss share = 1 - 0.85 x 0.9 = 0.235", lp.escapeShare() + lp.reservoirShare(), 0.235);
    }
    exo::SputteringConfig h = c;
    h.h2o_fraction = 0.6;  // the H2O fraction is the parameter; O2/H2 follow
    check(std::abs(h.o2Fraction() / h.h2Fraction() / expect - 1.0) < 1e-12,
          "ratio stays stoichiometric when --sputter-h2o-fraction changes", h.o2Fraction() / h.h2Fraction(), expect);

    exo::SputteringConfig L;
    L.legacy_escape_only = true;
    L.applyLegacyParams();  // what the CLI does for --sputter-legacy-escape-only
    check(L.o2Fraction() == 0.10 && L.h2Fraction() == 0.05, "legacy split is the old 0.10 / 0.05 (ratio 2)",
          L.o2Fraction() / L.h2Fraction(), 2.0);
    check(std::abs(L.escapeShare() - 0.2225) < 1e-15, "legacy escaping share = 22.25% of gross", L.escapeShare(), 0.2225);

    bool threw = false;
    try { exo::SputteringConfig b; b.o2_escape_fraction = 1.5; b.validate(); } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "validate() refuses an escape fraction outside [0, 1]");
    threw = false;
    try { exo::SputteringConfig b; b.launch_temperature_K = 0.0; b.validate(); } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "validate() refuses a non-positive launch temperature");
    threw = false;
    try { exo::SputteringConfig b; b.legacy_escape_only = true; b.h2o_fraction = 0.7; b.validate(); } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "validate() refuses a changed H2O fraction under the fixed legacy split");
    threw = false;
    try { exo::MultiTimescaleEvolution m(4); m.setSputterPartition(0.5, 0.2, 0.2); } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "setSputterPartition() refuses shares that do not sum to 1");
}

// ---------------------------------------------------------------------------
static void test_per_face_fates() {
    std::printf("\n(2) per-face product and fate closure\n");
    const ptm::Mesh m = make_icosphere(2, 1560.8e3);
    const Geo g = geometry(m, 1560.8e3);
    const size_t n = g.areas.size();
    std::vector<double> T(n);
    for (size_t i = 0; i < n; ++i) T[i] = 60.0 + 80.0 * (i % 7) / 6.0;  // spans the 80 K yield threshold
    for (int legacy = 0; legacy < 2; ++legacy) {
        exo::SputteringConfig c;
        c.legacy_escape_only = (legacy == 1);
        if (legacy) c.applyLegacyParams();
        exo::SputteringModel sm(n, c);
        std::vector<double> lat, lon;
        sm.initializeSpatialDistribution(g.centers, lat, lon, g.areas);
        exo::SputteringState st(n);
        sm.computeSputteringRates(T, g.areas, st);
        double worst_products = 0.0, worst_fates = 0.0, gross_total = 0.0, removal_total = 0.0;
        for (size_t i = 0; i < n; ++i) {
            const double G = st.gross_removal_flux[i];
            gross_total += G * g.areas[i];
            removal_total += st.surfaceRemovalFlux(c.legacy_escape_only)[i] * g.areas[i];
            if (G <= 0.0) continue;
            worst_products = std::max(worst_products,
                std::abs(st.h2o_sputter_flux[i] + st.o2_production_flux[i] + st.h2_production_flux[i] - G) / G);
            worst_fates = std::max(worst_fates,
                std::abs(st.total_mass_loss[i] + st.redeposition_flux[i] + st.reservoir_flux[i] - G) / G);
        }
        const std::string tag = legacy ? "legacy: " : "new: ";
        check(worst_products < 1e-14, tag + "per face, H2O + O2 + H2 = gross", worst_products, 1e-14);
        check(worst_fates < 1e-14, tag + "per face, escaped + transported + reservoir = gross", worst_fates, 1e-14);
        if (!legacy) {
            check(std::abs(removal_total / gross_total - 1.0) < 1e-15,
                  "new: the surface loses the GROSS sputtered mass", removal_total / gross_total, 1.0);
        } else {
            // Documents the defect the legacy flag reproduces.
            check(std::abs(removal_total / gross_total - 0.2225) < 1e-12,
                  "legacy: the surface loses only 22.25% of gross (not mass-conserving)",
                  removal_total / gross_total, 0.2225);
        }
    }
}

// ---------------------------------------------------------------------------
static void test_kernel_bound_population() {
    std::printf("\n(3) transport kernel: bound population and call-local index\n");
    const double R = exo::constants::EUROPA_RADIUS;
    const ptm::Mesh m = make_icosphere(3, R);
    const Geo g = geometry(m, R);
    const size_t n = g.areas.size();
    std::vector<double> src(n, 0.0);
    for (size_t i = 0; i < n; ++i) src[i] = (i % 3 == 0) ? g.areas[i] * 1e-12 : 0.0;
    const std::vector<double> T_hot(n, 683.0);

    exo::BallisticTransport bt_off;
    std::vector<double> dep;
    exo::TransportCallOptions off;
    off.apply_jeans_escape = false;
    exo::TransportTotals t_off;
    bt_off.computeTransportProduct(g.centers, g.areas, T_hot, src, dep, 1e-10, off, &t_off);
    double dep_sum = 0.0;
    for (double d : dep) dep_sum += d;
    check(std::abs(dep_sum / t_off.source - 1.0) < 1e-12, "Jeans off at 683 K: deposited = source",
          std::abs(dep_sum / t_off.source - 1.0), 1e-12);
    check(t_off.jeans_escaped == 0.0 && t_off.unscattered == 0.0, "Jeans off: no Jeans loss, nothing unscattered",
          t_off.jeans_escaped + t_off.unscattered, 0.0);

    exo::BallisticTransport bt_on;
    exo::TransportTotals t_on;
    bt_on.computeTransportProduct(g.centers, g.areas, T_hot, src, dep, 1e-10, exo::TransportCallOptions(), &t_on);
    check(t_on.jeans_escaped > 0.0, "Jeans on at 683 K would remove the bound population a second time",
          t_on.jeans_escaped / t_on.source, 0.0);
    check(std::abs((t_on.deposited + t_on.jeans_escaped) / t_on.source - 1.0) < 1e-12,
          "totals close: source = deposited + Jeans + unscattered",
          std::abs((t_on.deposited + t_on.jeans_escaped + t_on.unscattered) / t_on.source - 1.0), 1e-12);

    // Persistent index built by a COLD call; then a hot point source.
    size_t s = 0, anti = 0;
    double best = 2.0;
    for (size_t i = 0; i < n; ++i) {
        const double c = (g.centers[i][0] * g.centers[0][0] + g.centers[i][1] * g.centers[0][1] +
                          g.centers[i][2] * g.centers[0][2]) / (R * R);
        if (c < best) { best = c; anti = i; }
    }
    std::vector<double> point(n, 0.0);
    point[s] = 1.0;
    const std::vector<double> T_cold(n, 110.0);
    for (int priv = 0; priv < 2; ++priv) {
        exo::BallisticTransport bt;
        std::vector<double> d0;
        bt.computeTransportProduct(g.centers, g.areas, T_cold, point, d0);  // builds the small index
        exo::TransportCallOptions o;
        o.apply_jeans_escape = false;
        o.private_index_if_needed = (priv == 1);
        std::streambuf* old = std::cerr.rdbuf();
        std::ostringstream err;
        std::cerr.rdbuf(err.rdbuf());
        bt.computeTransportProduct(g.centers, g.areas, T_hot, point, d0, 1e-10, o, nullptr);
        std::cerr.rdbuf(old);
        const bool warned = err.str().find("transport kernel cutoff") != std::string::npos;
        if (priv) {
            check(d0[anti] > 0.0 && !warned, "hot call with a call-local index reaches the antipode", d0[anti], 0.0);
        } else {
            check(d0[anti] == 0.0 && warned, "hot call on the cold persistent index is truncated AND warns", d0[anti], 0.0);
        }
    }
}

// ---------------------------------------------------------------------------
// Two populations: sublimation on faces [0, n/2), sputtering on [n/2, n).
// Deposition at cache time is a mass-conserving spread of each source.
static void test_deposition_rescale() {
    std::printf("\n(4) deposition rescale follows its own source (mass accounting, part c)\n");
    const size_t n = 200;
    std::vector<double> A(n, 1.0), avail0(n, 1.0), avail1(n, 1.0);
    std::vector<double> sub(n, 0.0), sput(n, 0.0);
    for (size_t i = 0; i < n / 2; ++i) sub[i] = 2.0e-9;
    for (size_t i = n / 2; i < n; ++i) sput[i] = 1.0e-9;
    for (size_t i = 0; i < n / 2; ++i) avail1[i] = 0.2;  // sublimating faces lose ice; sputtered ones do not
    const double share = exo::SputteringConfig().transportShare();
    double S0 = 0, P0 = 0;
    for (size_t i = 0; i < n; ++i) { S0 += sub[i]; P0 += sput[i]; }

    // New accounting.
    {
        exo::DiurnalAverages d(n);
        d.sputter_transport = true;
        d.sputter_transport_share = share;
        for (size_t i = 0; i < n; ++i) {
            d.avg_sublimation_flux[i] = sub[i];
            d.avg_sputtering_flux[i] = sput[i];
            d.avg_deposition_sublimation_flux[i] = S0 / n;             // uniform spread, sums to S0
            d.avg_deposition_sputter_flux[i] = share * P0 / n;         // sums to share * P0
            d.avg_deposition_flux[i] = d.avg_deposition_sublimation_flux[i] + d.avg_deposition_sputter_flux[i];
        }
        d.cacheBaseFluxesUnscaled(sub, sput, avail0, A);
        d.applyScaledFluxes(avail1, A);
        double s = 0, p = 0, dep = 0, dep_sub = 0;
        for (size_t i = 0; i < n; ++i) {
            s += d.avg_sublimation_flux[i];
            p += d.avg_sputtering_flux[i];
            dep += d.avg_deposition_flux[i];
            dep_sub += d.avg_deposition_sublimation_flux[i];
        }
        const double excess = (dep - (s + share * p)) / (s + share * p);
        check(excess <= 1e-6, "new: deposition <= sublimation + redeposited sputter source", excess, 1e-6);
        check(dep_sub <= s * (1.0 + 1e-6), "new: deposition from sublimation <= sublimation", dep_sub / s - 1.0, 1e-6);
    }
    // Old single (sublimation + sputtering) rescale on the same case, where the
    // sputtering term is the legacy escaping share and deposition is from
    // sublimation alone: it must show the artifact (discriminating).
    {
        const double esc = 0.2225;
        std::vector<double> sput_old(n);
        for (size_t i = 0; i < n; ++i) sput_old[i] = esc * sput[i];
        exo::DiurnalAverages d(n);
        for (size_t i = 0; i < n; ++i) {
            d.avg_sublimation_flux[i] = sub[i];
            d.avg_sputtering_flux[i] = sput_old[i];
            d.avg_deposition_flux[i] = S0 / n;
        }
        d.cacheBaseFluxesUnscaled(sub, sput_old, avail0, A);
        d.applyScaledFluxes(avail1, A);
        double s = 0, dep = 0;
        for (size_t i = 0; i < n; ++i) { s += d.avg_sublimation_flux[i]; dep += d.avg_deposition_flux[i]; }
        check(dep > s * (1.0 + 1e-3), "old rescale (control): deposition EXCEEDS sublimation on this case",
              dep / s - 1.0, 1e-3);
    }
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

static double summaryValue(const std::string& text, const std::string& key) {
    const size_t p = text.find(key);
    if (p == std::string::npos) return std::nan("");
    return std::stod(text.substr(p + key.size()));
}

static void test_end_to_end(bool legacy_params) {
    std::printf("\n(5) end to end: --multiscale with sputtering on a synthetic ico3 body (%s parameters)\n",
                legacy_params ? "legacy" : "current");
    char tmpl[] = "/tmp/test_sputter_mass_accounting_XXXXXX";
    const char* dd = mkdtemp(tmpl);
    if (!dd) { check(false, "mkdtemp"); return; }
    const fs::path root(dd);
    const ptm::Mesh mesh = make_icosphere(3, 1560.8);
    writeThermal(root / "thermal", mesh);

    auto integ = std::make_unique<exo::PTMExosphereIntegration>((root / "out").string());
    integ->setShowProgress(false);
    integ->setMesh(std::make_shared<ptm::Mesh>(mesh));
    std::streambuf* old = std::cout.rdbuf();
    std::ostringstream sink;
    std::cout.rdbuf(sink.rdbuf());
    bool ok = integ->loadTemperatureData((root / "thermal").string(), kNt);
    integ->enableIceLayerModel(0.05);
    exo::SputteringConfig sc;
    sc.rate_scale = 50.0;  // make sputtering comparable to sublimation on this small body
    if (legacy_params) {
        sc.legacy_params = true;
        sc.applyLegacyParams();
    }
    std::string err;
    try {
        integ->enableSputtering(sc);
        if (ok) integ->runMultiTimescaleSimulation(2000, 100, 500);
    } catch (const std::exception& e) {
        err = e.what();
    }
    std::cout.rdbuf(old);
    check(ok && err.empty(), "run completes (the audit throws on a closure failure)" + (err.empty() ? "" : ": " + err));

    const std::string log = sink.str();
    const std::regex re("\\[sputter-audit\\] [^:]*: gross ([0-9.e+-]+) = redeposited ([0-9.e+-]+) \\+ escaped "
                        "([0-9.e+-]+) \\+ reservoir ([0-9.e+-]+)");
    int lines = 0;
    double worst = 0.0;
    for (auto it = std::sregex_iterator(log.begin(), log.end(), re); it != std::sregex_iterator(); ++it) {
        const double G = std::stod((*it)[1]), D = std::stod((*it)[2]), E = std::stod((*it)[3]), Rr = std::stod((*it)[4]);
        worst = std::max(worst, std::abs(G - (D + E + Rr)) / G);
        ++lines;
    }
    check(lines >= 2, "audit lines were printed (full diurnal + snapshots)", lines, 2);
    // Printed to 7 significant digits, so the parsed closure is limited by print rounding.
    check(worst < 1e-6, "every audit: gross = redeposited + escaped + reservoir", worst, 1e-6);

    std::ifstream in(root / "out" / "evolution_summary.txt");
    std::stringstream ss;
    ss << in.rdbuf();
    const std::string s = ss.str();
    const double sub = summaryValue(s, "Total sublimated: "), dep = summaryValue(s, "Total deposited: ");
    const double spu = summaryValue(s, "Total sputtered: "), red = summaryValue(s, "sputter redeposited (ballistic transport): ");
    const double esc = summaryValue(s, "sputter escaped (all products): ");
    const double res = legacy_params ? summaryValue(s, "sputter reservoir (non-escaping O2 + H2): ")
                                     : summaryValue(s, "sputter atmospheric loss (O2: ionization/dissociation/pickup): ");
    const double jea = summaryValue(s, "Sublimation Jeans escape: ");
    const double lag = summaryValue(s, "Super-step lag (deposition in excess of step-mean sources): ");
    const bool have = std::isfinite(sub + dep + spu + red + esc + res + jea + lag);
    check(have, "evolution_summary.txt carries the sputter fates, Jeans and lag terms");
    if (have) {
        const double lhs = sub + spu - dep, rhs = esc + res + jea - lag;
        // Summary values are printed to 4 significant digits.
        check(std::abs(lhs - rhs) <= 2e-3 * (sub + spu), "cumulative: sublimated + sputtered - deposited = "
              "escaped + reservoir + Jeans - lag", std::abs(lhs - rhs) / (sub + spu), 2e-3);
        if (legacy_params) {
            exo::SputteringConfig lp;
            lp.legacy_params = true;
            lp.applyLegacyParams();
            check(std::abs((esc + res) / spu - lp.escapeShare() - lp.reservoirShare()) < 2e-3,
                  "cumulative sputter surface loss = 23.5% of gross", (esc + res) / spu, 0.235);
        } else {
            // Current set: the composition follows tau(T) per face, so the realized share
            // differs from the T_ref share; it must still close (above) and be a share.
            check((esc + res) / spu > 0.0 && (esc + res) / spu < 1.0 && std::abs(red + esc + res - spu) <= 2e-3 * spu,
                  "cumulative: redeposited + escaped + atmospheric loss = gross (current set)",
                  std::abs(red + esc + res - spu) / spu, 2e-3);
        }
    }
    fs::remove_all(root);
}

int main() {
    std::printf("=== test_sputter_mass_accounting ===\n");
    test_stoichiometry();
    test_per_face_fates();
    test_kernel_bound_population();
    test_deposition_rescale();
    test_end_to_end(true);
    test_end_to_end(false);
    std::printf("\n%d/%d checks passed\n", g_total - g_fail, g_total);
    return g_fail == 0 ? 0 : 1;
}
