/*
 * Regression tests for the exosphere review blockers EXO-01 and EXO-02.
 *
 * EXO-01: besselK0(x) must match reference K0 to 1e-6 relative on both
 *         branches, be continuous across the x = 2 branch switch, and give
 *         the analytic mean hop <xi> = 8/pi for the areal kernel xi^2 K0(xi).
 * EXO-02: face areas must be in m^2 whatever the mesh length unit. Checked
 *         end to end: a km-unit icosphere with uniform T must report
 *         total_sublimation_rate_kg_s = Phi(T) * sum(A [m^2]) in
 *         atmospheric_mass.csv, through BOTH setMesh() and loadMesh(); and a
 *         mesh that is not a closed sphere must be refused.
 *
 * Uses only public API present before the fix, so the same file builds
 * against the unfixed code (where it fails).
 *
 * Exit code 0 = all PASS, 1 = at least one FAIL.
 */

#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <unistd.h>

#include "exosphere_physics.h"
#include "ptm_exosphere_integration.h"
#include "../mesh/mesh_utils.h"

namespace exo = ptm::exosphere;
namespace fs = std::filesystem;

static int g_failures = 0;

static void check(bool ok, const std::string& what) {
    std::cout << (ok ? "  PASS  " : "  FAIL  ") << what << std::endl;
    if (!ok) ++g_failures;
}

// ---------------------------------------------------------------------------
// EXO-01
// ---------------------------------------------------------------------------
static void test_k0_values() {
    std::cout << "\n[EXO-01] K0 reference values (scipy.special.k0, cross-checked against\n"
                 "         the integral of exp(-x cosh u) to <= 4.4e-16)\n";
    const std::vector<std::pair<double, double>> ref = {
        {0.1, 2.4270690247020164},
        {0.5, 0.9244190712276656},
        {1.0, 0.42102443824070823},
        {1.9, 0.12884597927604755},
        {2.0, 0.1138938727495334},
        {2.1, 0.10078374088996692},
        {5.0, 0.0036910983340425942},
        {15.0, 9.819536482396435e-08},
    };
    for (const auto& [x, k] : ref) {
        const double got = exo::besselK0(x);
        const double rel = std::abs(got / k - 1.0);
        std::ostringstream os;
        os << "K0(" << x << ") = " << std::setprecision(12) << got << "  ref " << k
           << "  rel err " << std::scientific << std::setprecision(2) << rel << " (tol 1e-6)";
        check(rel < 1e-6, os.str());
    }
}

static void test_k0_continuity() {
    std::cout << "\n[EXO-01] continuity and monotonicity across the x = 2 branch switch\n";
    const double below = exo::besselK0(2.0);
    const double above = exo::besselK0(std::nextafter(2.0, 3.0));
    const double jump = std::abs(above / below - 1.0);
    std::ostringstream os;
    os << "K0(2) / K0(2+ulp) - 1 = " << std::scientific << std::setprecision(2) << jump
       << " (tol 1e-6)";
    check(jump < 1e-6, os.str());

    bool monotone = true;
    double prev = exo::besselK0(1.5);
    for (int i = 1; i <= 1000; ++i) {
        const double x = 1.5 + i * 1e-3;
        const double v = exo::besselK0(x);
        if (!(v < prev)) { monotone = false; break; }
        prev = v;
    }
    check(monotone, "K0 strictly decreasing on [1.5, 2.5] (step 1e-3)");
}

// Mean hop for the kernel as the transport code uses it: probability per unit
// AREA proportional to d * K0(d / d_char), so the ring-weighted density is
// xi^2 K0(xi) and <xi> = int xi^3 K0 / int xi^2 K0 = 4 / (pi/2) = 8/pi.
static void test_mean_hop() {
    std::cout << "\n[EXO-01] mean ballistic hop of the areal kernel xi^2 K0(xi)\n";
    const double xmax = 40.0;
    const int n = 400000;  // composite Simpson, even
    const double h = xmax / n;
    double m2 = 0.0, m3 = 0.0;
    for (int i = 0; i <= n; ++i) {
        const double x = i * h;
        const double w = (i == 0 || i == n) ? 1.0 : ((i % 2) ? 4.0 : 2.0);
        const double k = (x > 0.0) ? exo::besselK0(x) : 0.0;  // x^2 K0 -> 0 at 0
        m2 += w * x * x * k;
        m3 += w * x * x * x * k;
    }
    const double mean = m3 / m2;
    const double exact = 8.0 / M_PI;
    std::ostringstream os;
    os << "<xi> = <d>/d_char = " << std::setprecision(6) << mean << "  exact 8/pi = " << exact
       << "  ratio " << (mean / exact) << " (tol 1e-4)";
    check(std::abs(mean / exact - 1.0) < 1e-4, os.str());
}

// ---------------------------------------------------------------------------
// EXO-02
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

// Independent reference: flat-triangle area of the mesh rescaled to meters.
static double total_area_m2(const ptm::Mesh& m, double to_m) {
    double s = 0.0;
    for (const auto& tri : m.faces) {
        const auto& a = m.vertices[tri[0]];
        const auto& b = m.vertices[tri[1]];
        const auto& c = m.vertices[tri[2]];
        const double e1[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
        const double e2[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
        const double cx = e1[1] * e2[2] - e1[2] * e2[1];
        const double cy = e1[2] * e2[0] - e1[0] * e2[2];
        const double cz = e1[0] * e2[1] - e1[1] * e2[0];
        s += 0.5 * std::sqrt(cx * cx + cy * cy + cz * cz);
    }
    return s * to_m * to_m;
}

static void write_uniform_temperature(const fs::path& dir, size_t n_faces, float T) {
    fs::create_directories(dir / "temperature");
    std::ofstream f(dir / "temperature" / "temperature_0000.bin", std::ios::binary);
    const uint32_t n = static_cast<uint32_t>(n_faces);
    f.write(reinterpret_cast<const char*>(&n), sizeof(n));
    std::vector<float> t(n_faces, T);
    f.write(reinterpret_cast<const char*>(t.data()), static_cast<std::streamsize>(t.size() * 4));
}

// Same binary layout as the delivered exosphere mesh.ply (double xyz, uchar/int list).
static void write_ply(const fs::path& path, const ptm::Mesh& m) {
    std::ofstream f(path, std::ios::binary);
    f << "ply\nformat binary_little_endian 1.0\n"
      << "element vertex " << m.vertices.size() << "\n"
      << "property double x\nproperty double y\nproperty double z\n"
      << "element face " << m.faces.size() << "\n"
      << "property list uchar int vertex_indices\nend_header\n";
    for (const auto& p : m.vertices) f.write(reinterpret_cast<const char*>(p.data()), 24);
    for (const auto& tri : m.faces) {
        const unsigned char three = 3;
        f.write(reinterpret_cast<const char*>(&three), 1);
        f.write(reinterpret_cast<const char*>(tri.data()), 12);
    }
}

static double read_total_sublimation(const fs::path& out_dir) {
    std::ifstream f(out_dir / "atmosphere" / "atmospheric_mass.csv");
    std::string header, row;
    if (!std::getline(f, header) || !std::getline(f, row)) return std::nan("");
    std::vector<std::string> cols;
    std::stringstream ss(row);
    for (std::string c; std::getline(ss, c, ',');) cols.push_back(c);
    return cols.size() > 7 ? std::stod(cols[7]) : std::nan("");  // total_sublimation_rate_kg_s
}

static void test_reported_rate(const std::string& path_name, bool via_file) {
    const double R_km = exo::constants::EUROPA_RADIUS / 1000.0;
    const ptm::Mesh mesh_km = make_icosphere(5, R_km);  // 20480 faces, vertices in km
    const double T = 120.0;
    const double expected = exo::VaporPressure::calculateSublimationFlux(T) *
                            total_area_m2(mesh_km, 1000.0);

    const fs::path root = fs::temp_directory_path() /
                          ("ptm_exo_area_test_" + path_name + "_" + std::to_string(::getpid()));
    fs::remove_all(root);
    write_uniform_temperature(root, mesh_km.faces.size(), static_cast<float>(T));

    double got = std::nan("");
    std::string failure;
    try {
        exo::PTMExosphereIntegration integ((root / "out").string());
        integ.setShowProgress(false);
        bool ok = true;
        if (via_file) {
            write_ply(root / "mesh.ply", mesh_km);
            ok = integ.loadMesh((root / "mesh.ply").string());
        } else {
            integ.setMesh(std::make_shared<ptm::Mesh>(mesh_km));
        }
        if (!ok) failure = "loadMesh returned false";
        else if (!integ.loadTemperatureData(root.string(), 1)) failure = "loadTemperatureData failed";
        else {
            integ.simulateTimestep(0, 3600.0);
            integ.exportAtmosphericMass(0, 0.0);
            got = read_total_sublimation(root / "out");
        }
    } catch (const std::exception& e) {
        failure = std::string("exception: ") + e.what();
    }
    fs::remove_all(root);

    std::ostringstream os;
    os << path_name << ": km-unit ico5, T = " << T << " K: reported total sublimation "
       << std::scientific << std::setprecision(6) << got << " kg/s, expected Phi(T) sum(A_m2) = "
       << expected << " (ratio " << std::setprecision(4) << (got / expected) << ", tol 1e-5)";
    if (!failure.empty()) os << " [" << failure << "]";
    check(failure.empty() && std::abs(got / expected - 1.0) < 1e-5, os.str());
}

static void test_open_mesh_refused() {
    const double R_km = exo::constants::EUROPA_RADIUS / 1000.0;
    ptm::Mesh full = make_icosphere(4, R_km);
    ptm::Mesh north;
    north.vertices = full.vertices;
    for (const auto& tri : full.faces) {
        const double zc = full.vertices[tri[0]][2] + full.vertices[tri[1]][2] + full.vertices[tri[2]][2];
        if (zc > 0.0) north.faces.push_back(tri);
    }
    bool refused = false;
    try {
        exo::PTMExosphereIntegration integ(
            (fs::temp_directory_path() / ("ptm_exo_open_" + std::to_string(::getpid()))).string());
        integ.setMesh(std::make_shared<ptm::Mesh>(north));
    } catch (const std::exception& e) {
        refused = true;
        std::cout << "        (refusal message: " << e.what() << ")" << std::endl;
    }
    fs::remove_all(fs::temp_directory_path() / ("ptm_exo_open_" + std::to_string(::getpid())));
    check(refused, "setMesh refuses an open hemisphere (sum A = 0.5 x 4 pi R^2)");
}

int main() {
    std::cout << "Exosphere EXO-01 / EXO-02 regression tests\n";
    test_k0_values();
    test_k0_continuity();
    test_mean_hop();
    std::cout << "\n[EXO-02] reported absolute rate uses m^2 areas for a km-unit mesh\n";
    test_reported_rate("setMesh", false);
    test_reported_rate("loadMesh", true);
    test_open_mesh_refused();
    std::cout << "\n" << (g_failures == 0 ? "ALL PASS" : "FAILURES: " + std::to_string(g_failures))
              << std::endl;
    return g_failures == 0 ? 0 : 1;
}
