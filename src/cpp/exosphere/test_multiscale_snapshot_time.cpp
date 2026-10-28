// Unit test for multiscale snapshot timing.
// Self-contained; exit code 0 = all checks pass.
//
// A multiscale snapshot labeled L must be the state after exactly L epochs, with
// every field taken from that one state, and time_steps.txt must list the true
// time of every snapshot written. Before the fix the export ran after the epoch's
// super-step: state fields (concentration, ice mass, cumulative fluxes) were up to
// a whole export interval past the label while fluxes, temperatures and bond_albedo
// were at the label; time_steps.txt held (L + 1) * dt_epoch and omitted the final
// snapshot.
//
// Checks (on a synthetic ico3 body, CPU):
//  (a) snapshot L of a long run is byte-identical to the final snapshot of a run
//      that stops at exactly L epochs (the state after L epochs, by construction);
//  (b) within one snapshot, emission_flux / exposed ice fraction(concentration) is
//      the same per-face potential flux at every label (fluxes and composition are
//      from the same state);
//  (c) with albedo coupling, bond_albedo equals the forward albedo of the exported
//      concentration;
//  (d) time_steps.txt has one entry per snapshot file, equal to L * dt_epoch, and
//      includes the final label (num_epochs); the atmospheric-mass CSV agrees;
//      --export-interval 0 writes the final state only.

#include "composition_albedo.h"
#include "ptm_exosphere_integration.h"
#include "multitimescale_evolution.h"
#include "ice_layer_model.h"
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
    std::printf("[%s] %-78s value=%.3e limit=%.1e\n", ok ? "PASS" : "FAIL", name.c_str(), value, limit);
    std::fflush(stdout);
}

// Same construction as test_albedo_refresh.cpp.
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

static const int kNt = 8;           // diurnal timesteps
static const int kCycles = 100;     // diurnal cycles per epoch

// Synthetic thermal output: a warm equatorial belt (sublimating on these time
// scales) over a cold body, with a diurnal swing.
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
            const double z = c[2], r = std::sqrt(c[0] * c[0] + c[1] * c[1] + c[2] * c[2]);
            const double mu = std::abs(z) / r;                         // |sin(lat)|
            const double base = 150.0 - 70.0 * mu;                     // 150 K equator, 80 K poles
            const double swing = 8.0 * std::cos(2.0 * M_PI * (t / double(kNt)) + 0.37 * f);
            T[f] = static_cast<float>(base + swing);
        }
        std::ostringstream fn;
        fn << (dir / "temperature").string() << "/temperature_" << std::setfill('0') << std::setw(4) << t << ".bin";
        std::ofstream o(fn.str(), std::ios::binary);
        const uint32_t c = static_cast<uint32_t>(n);
        o.write(reinterpret_cast<const char*>(&c), sizeof(c));
        o.write(reinterpret_cast<const char*>(T.data()), static_cast<std::streamsize>(n * sizeof(float)));
    }
}

static std::vector<float> readField(const fs::path& out, const std::string& name, int label, bool* ok = nullptr) {
    std::ostringstream fn;
    fn << (out / name).string() << "/" << name << "_" << std::setfill('0') << std::setw(4) << label << ".bin";
    std::ifstream in(fn.str(), std::ios::binary);
    std::vector<float> v;
    if (!in) { if (ok) *ok = false; return v; }
    uint32_t c = 0;
    in.read(reinterpret_cast<char*>(&c), sizeof(c));
    v.resize(c);
    in.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(c * sizeof(float)));
    if (ok) *ok = static_cast<bool>(in);
    return v;
}

static std::string slurp(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    std::ostringstream s;
    s << in.rdbuf();
    return s.str();
}

static std::vector<int> labelsOf(const fs::path& out, const std::string& name) {
    std::vector<int> labels;
    if (!fs::exists(out / name)) return labels;
    for (const auto& e : fs::directory_iterator(out / name)) {
        const std::string s = e.path().stem().string();         // name_LLLL
        labels.push_back(std::stoi(s.substr(name.size() + 1)));
    }
    std::sort(labels.begin(), labels.end());
    return labels;
}

static std::vector<double> readTimeSteps(const fs::path& out) {
    std::ifstream in(out / "time_steps.txt");
    std::vector<double> t;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        t.push_back(std::stod(line));
    }
    return t;
}

struct RunOpts { int epochs; int interval; bool coupling; };

static bool run(const fs::path& out, const fs::path& thermal, const ptm::Mesh& mesh, const RunOpts& o) {
    fs::remove_all(out);
    auto integ = std::make_unique<exo::PTMExosphereIntegration>(out.string());
    integ->setShowProgress(false);
    integ->setMesh(std::make_shared<ptm::Mesh>(mesh));
    std::streambuf* old = std::cout.rdbuf();
    std::ostringstream sink;
    std::cout.rdbuf(sink.rdbuf());
    bool ok = integ->loadTemperatureData(thermal.string(), kNt);
    integ->enableIceLayerModel(0.05);
    if (ok && o.coupling) {
        ok = integ->setThermalAlbedoUniform(0.6225, "test") &&
             integ->enableAlbedoCoupling(exo::CompositionAlbedoConfig());
    }
    if (ok) integ->runMultiTimescaleSimulation(o.epochs, kCycles, o.interval);
    std::cout.rdbuf(old);
    return ok;
}

int main() {
    std::printf("=== test_multiscale_snapshot_time ===\n");
    char tmpl[] = "/tmp/test_multiscale_snapshot_time_XXXXXX";
    const char* d = mkdtemp(tmpl);
    if (!d) { std::printf("mkdtemp failed\n"); return 2; }
    const fs::path root(d);

    const ptm::Mesh mesh = make_icosphere(3, 1560.8);
    const size_t n = mesh.faces.size();
    writeThermal(root / "thermal", mesh);
    const double dt_epoch = kCycles * exo::timescale_constants::EUROPA_DIURNAL_PERIOD;

    // Long run: 25 epochs, export every 10 -> labels 0, 10, 20, 25 (final).
    const RunOpts longrun{25, 10, false};
    const fs::path L = root / "long";
    check(run(L, root / "thermal", mesh, longrun), "long run (25 epochs, interval 10) completes");
    const std::vector<int> labels = labelsOf(L, "nonice_concentration");
    const std::vector<int> want = {0, 10, 20, 25};
    check(labels == want, "(d) snapshot labels are 0, 10, 20 and the final state 25",
          double(labels.size()), double(want.size()));

    // (a) label 10 and 20 == final snapshot of runs stopped at 10 and 20 epochs.
    const std::vector<std::string> state_fields = {
        "nonice_concentration", "ice_mass_kg_m2", "cumulative_sublimated_kg_m2",
        "cumulative_deposited_kg_m2", "emission_flux_kg_m2_s", "deposition_flux_kg_m2_s",
        "vapor_pressure_Pa", "temperature_mean_K"};
    for (int stop : {10, 20}) {
        const fs::path S = root / ("stop" + std::to_string(stop));
        check(run(S, root / "thermal", mesh, RunOpts{stop, 10, false}),
              "short run to " + std::to_string(stop) + " epochs completes");
        size_t n_diff = 0;
        for (const auto& f : state_fields) {
            bool ok1 = true, ok2 = true;
            const auto a = readField(L, f, stop, &ok1);
            const auto b = readField(S, f, stop, &ok2);
            if (!ok1 || !ok2 || a != b) {
                ++n_diff;
                std::printf("       field %s differs (long label %d vs short final)\n", f.c_str(), stop);
            }
        }
        check(n_diff == 0, "(a) long-run snapshot " + std::to_string(stop) +
              " == state of a run stopped at " + std::to_string(stop) + " epochs (8 fields, bytes)",
              double(n_diff), 0.0);
    }

    // The evolution must actually move the composition between labels, or (a)/(b)
    // prove nothing.
    {
        const auto c0 = readField(L, "nonice_concentration", 0);
        const auto c25 = readField(L, "nonice_concentration", 25);
        // Label 25 is written by the fixed engine only; fall back to the last
        // snapshot present so liveness is measured either way.
        const auto last = labels.empty() ? 0 : labels.back();
        const auto cl = c25.empty() ? readField(L, "nonice_concentration", last) : c25;
        double dmax = 0.0;
        for (size_t i = 0; i < n && i < c0.size() && i < cl.size(); ++i)
            dmax = std::max(dmax, double(std::abs(cl[i] - c0[i])));
        check(dmax > 1e-2, "composition evolves between the first and last snapshot (test is live)", dmax, 1e-2);
        const auto c0_init = readField(L, "nonice_concentration", 0);
        double e0 = 0.0;
        for (float c : c0_init) e0 = std::max(e0, std::abs(double(c) - 0.05));
        check(e0 < 1e-7, "(a) snapshot 0 is the initial state (5% non-ice everywhere)", e0, 1e-7);
    }

    // (b) emission / exposed fraction(conc) is the same potential flux at every label.
    {
        exo::IceLayerModel model(n);
        exo::IceLayerState st(n);
        model.initializeUniform(st, 0.05);
        std::vector<double> pot0;
        double worst = 0.0;
        size_t compared = 0;
        // Over the snapshots actually written, so a missing file cannot stand in for
        // a mixed-time one.
        for (int lab : labels) {
            bool ok1 = true, ok2 = true;
            const auto c = readField(L, "nonice_concentration", lab, &ok1);
            const auto e = readField(L, "emission_flux_kg_m2_s", lab, &ok2);
            if (!ok1 || !ok2 || c.size() != n || e.size() != n) {
                std::printf("       snapshot %d missing or short\n", lab);
                worst = 1.0;
                continue;
            }
            std::vector<double> pot(n, 0.0);
            for (size_t i = 0; i < n; ++i) {
                st.nonice_concentration[i] = c[i];
                model.updateIceMass(st, i);
                const double avail = model.getIceAvailabilityFactor(st, i);
                pot[i] = (avail > 1e-3) ? e[i] / avail : -1.0;
            }
            if (pot0.empty()) { pot0 = pot; continue; }
            for (size_t i = 0; i < n; ++i) {
                if (pot[i] < 0.0 || pot0[i] <= 1e-20) continue;
                worst = std::max(worst, std::abs(pot[i] / pot0[i] - 1.0));
                ++compared;
            }
        }
        check(labels.size() >= 3 && compared > n && worst < 1e-5 && pot0.size() == n,
              "(b) emission_flux / availability(conc) identical at every label (same-state fields)",
              worst, 1e-5);
    }

    // (c) coupling: bond_albedo is the forward albedo of the exported concentration.
    {
        const fs::path C = root / "coupled";
        check(run(C, root / "thermal", mesh, RunOpts{25, 10, true}), "coupled run completes");
        exo::CompositionAlbedoMapper mapper{exo::CompositionAlbedoConfig()};
        double worst = 0.0;
        const std::vector<int> clabels = labelsOf(C, "bond_albedo");
        for (int lab : clabels) {
            bool ok1 = true, ok2 = true;
            const auto c = readField(C, "nonice_concentration", lab, &ok1);
            const auto a = readField(C, "bond_albedo", lab, &ok2);
            if (!ok1 || !ok2 || c.size() != n || a.size() != n) {
                std::printf("       coupled snapshot %d missing or short\n", lab);
                worst = 1.0;
                continue;
            }
            for (size_t i = 0; i < n && i < c.size() && i < a.size(); ++i) {
                const double fwd = mapper.computeAlbedo(double(c[i]), static_cast<int>(i));
                worst = std::max(worst, std::abs(fwd - double(a[i])));
            }
        }
        check(clabels.size() >= 3 && worst < 1e-6,
              "(c) bond_albedo_L == forward albedo of nonice_concentration_L", worst, 1e-6);
    }

    // (d) time_steps.txt: complete, one entry per snapshot, at L * dt_epoch.
    {
        const auto t = readTimeSteps(L);
        check(t.size() == want.size(), "(d) time_steps.txt has one entry per snapshot (4)",
              double(t.size()), double(want.size()));
        double worst = 0.0;
        for (size_t k = 0; k < t.size() && k < want.size(); ++k)
            worst = std::max(worst, std::abs(t[k] - want[k] * dt_epoch));
        check(t.size() == want.size() && worst <= 0.05,
              "(d) time_steps.txt entry k == label_k * dt_epoch (to the 0.1 s print precision)", worst, 0.05);
        const std::string md = slurp(L / "metadata.txt");
        check(md.find("time_steps=4\n") != std::string::npos, "(d) metadata.txt time_steps=4");

        std::ifstream csv(L / "atmosphere" / "atmospheric_mass.csv");
        std::string line;
        std::getline(csv, line);  // header
        std::vector<std::pair<int, double>> rows;
        while (std::getline(csv, line)) {
            std::stringstream ss(line);
            std::string a, b;
            std::getline(ss, a, ',');
            std::getline(ss, b, ',');
            rows.emplace_back(std::stoi(a), std::stod(b));
        }
        bool labels_ok = rows.size() == want.size();
        double wt = 0.0;
        for (size_t k = 0; labels_ok && k < rows.size(); ++k) {
            labels_ok = rows[k].first == want[k];
            wt = std::max(wt, std::abs(rows[k].second / (want[k] * dt_epoch + 1.0) - want[k] * dt_epoch / (want[k] * dt_epoch + 1.0)));
        }
        check(labels_ok && wt < 1e-6, "(d) atmospheric_mass.csv rows: label L at time L * dt_epoch", wt, 1e-6);
    }

    // (d) --export-interval 0: the final state only, as documented.
    {
        const fs::path Z = root / "final_only";
        check(run(Z, root / "thermal", mesh, RunOpts{25, 0, false}), "export-interval 0 run completes");
        const auto zl = labelsOf(Z, "nonice_concentration");
        const auto zt = readTimeSteps(Z);
        check(zl == std::vector<int>{25} && zt.size() == 1 && std::abs(zt[0] - 25 * dt_epoch) <= 0.05,
              "(d) --export-interval 0 writes exactly the final state (label 25, one time entry)",
              double(zl.size()), 1.0);
        // Export labels clamp super-steps, so only a run with the same partition
        // (interval 25 on 25 epochs) must match byte for byte.
        const fs::path Q = root / "interval25";
        check(run(Q, root / "thermal", mesh, RunOpts{25, 25, false}), "export-interval 25 run completes");
        bool ok1 = true, ok2 = true;
        const auto cz = readField(Z, "nonice_concentration", 25, &ok1);
        check(cz == readField(Q, "nonice_concentration", 25, &ok2) && ok1 && ok2,
              "(d) its final state == an interval-25 run's final snapshot (same partition, bytes)");
        const auto c10 = readField(L, "nonice_concentration", 25);
        double d = 0.0;
        for (size_t i = 0; i < cz.size() && i < c10.size(); ++i) d = std::max(d, double(std::abs(cz[i] - c10[i])));
        std::printf("       info: interval 0 vs interval 10 final concentration max |diff| = %.3e "
                    "(super-step partition, not an export effect)\n", d);
    }

    std::error_code ec;
    fs::remove_all(root, ec);
    std::printf("\n%d / %d checks passed\n", g_total - g_fail, g_total);
    return g_fail == 0 ? 0 : 1;
}
