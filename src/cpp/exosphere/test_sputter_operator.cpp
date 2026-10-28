// the fixed-temperature transport operator (built once, reused) against
// the direct per-call kernel of BallisticTransport::computeTransportProduct.
//
//   (1) the kernel table's interpolation error is small;
//   (2) raw deposition and calibration match the direct kernel face by face,
//       at the sputter launch temperature (677 K, all pairs) and at 150 K with
//       Jeans escape (a kernel cutoff below pi, taper active);
//   (3) the finalized deposition matches a full direct call, and conserves mass;
//   (4) the face-local delta update equals a full application (linearity);
//   (5) the result does not depend on the thread count (bit-identical);
//   (6) timing, direct vs operator.
//
// Usage: test_sputter_operator [ico_levels=5]

#include "exosphere_physics.h"
#include "../mesh/mesh_utils.h"

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace exo = ptm::exosphere;

static int g_fail = 0, g_total = 0;
static void check(bool ok, const std::string& name, double value = 0.0, double limit = 0.0) {
    ++g_total;
    if (!ok) ++g_fail;
    std::printf("[%s] %-82s value=%.3e limit=%.1e\n", ok ? "PASS" : "FAIL", name.c_str(), value, limit);
    std::fflush(stdout);
}

// Same construction as test_sputter_mass_accounting.cpp.
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
    std::vector<double> areas;
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

// max_j |a_j - b_j| / max_j |b_j| (a field-scale relative error) and the
// face-by-face max |a_j / b_j - 1| over faces with |b_j| > 1e-6 max |b|.
struct Diff { double field = 0.0, face = 0.0; };
static Diff diff(const std::vector<double>& a, const std::vector<double>& b) {
    double bmax = 0.0, dmax = 0.0, rmax = 0.0;
    for (double v : b) bmax = std::max(bmax, std::abs(v));
    for (size_t j = 0; j < a.size(); ++j) {
        dmax = std::max(dmax, std::abs(a[j] - b[j]));
        if (std::abs(b[j]) > 1e-6 * bmax) rmax = std::max(rmax, std::abs(a[j] / b[j] - 1.0));
    }
    Diff d;
    d.field = (bmax > 0.0) ? dmax / bmax : dmax;
    d.face = rmax;
    return d;
}
static double sum(const std::vector<double>& v) {
    double s = 0.0;
    for (double x : v) s += x;
    return s;
}

static void run_case(const Geo& g, double T, bool jeans, const char* label, bool timing) {
    const size_t n = g.centers.size();
    std::printf("\n=== %s: %zu faces, T = %.1f K, Jeans escape %s ===\n", label, n, T, jeans ? "on" : "off");
    // A non-uniform, sputter-like source: a hemispheric contrast plus noise.
    std::mt19937_64 rng(12345);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    std::vector<double> e1(n), e2(n), temps(n, T);
    for (size_t i = 0; i < n; ++i) {
        const double x = g.centers[i][0] / exo::constants::EUROPA_RADIUS;
        e1[i] = g.areas[i] * (1.0 + 2.0 * std::max(0.0, x)) * (0.5 + U(rng));
    }
    // A second field that differs on ~5% of the sources.
    e2 = e1;
    std::vector<char> changed(n, 0);
    for (size_t i = 0; i < n; ++i) {
        if (U(rng) < 0.05) { e2[i] *= 0.3 + 1.4 * U(rng); changed[i] = 1; }
    }

    exo::BallisticTransport bt;
    exo::TransportCallOptions raw_opt;
    raw_opt.apply_jeans_escape = jeans;
    raw_opt.private_index_if_needed = true;
    raw_opt.raw = true;
    std::vector<double> cal_direct, raw_direct, tmp;
    raw_opt.calibration_out = &cal_direct;
    auto t0 = std::chrono::steady_clock::now();
    bt.computeTransportProduct(g.centers, g.areas, temps, e1, raw_direct, 1e-10, raw_opt);
    const double s_direct = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    exo::FixedKernelTransportOperator op;
    op.build(g.centers, g.areas, T, jeans, bt.bodyRadius(), 1e-10);
    std::printf("  operator build %.3f s, %.2f MB, table %zu nodes, max interpolation error %.2e\n",
                op.buildSeconds(), op.memoryBytes() / 1048576.0, op.tableNodes(), op.tableMaxRelError());
    check(op.tableMaxRelError() < 1e-6, "kernel table: max relative interpolation error < 1e-6", op.tableMaxRelError(), 1e-6);

    std::vector<double> raw_op;
    exo::TransportTotals tt;
    t0 = std::chrono::steady_clock::now();
    op.applyRaw(e1, nullptr, raw_op, &tt);
    const double s_op = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    // The direct kernel's own floating-point noise floor: the same call on a
    // rigidly rotated copy of the mesh (physically a no-op). Its float cosines
    // round differently, and near pairs are sensitive to that (one float ulp of
    // cos at theta ~ 0.015 rad is ~3e-4 of theta).
    Geo rot = g;
    {
        const double ang = 0.7318, ax[3] = {0.2672612419, 0.5345224838, 0.8017837257};  // (1,2,3)/sqrt(14)
        const double c = std::cos(ang), sn = std::sin(ang), C = 1.0 - c;
        const double Rm[3][3] = {
            {c + ax[0]*ax[0]*C, ax[0]*ax[1]*C - ax[2]*sn, ax[0]*ax[2]*C + ax[1]*sn},
            {ax[1]*ax[0]*C + ax[2]*sn, c + ax[1]*ax[1]*C, ax[1]*ax[2]*C - ax[0]*sn},
            {ax[2]*ax[0]*C - ax[1]*sn, ax[2]*ax[1]*C + ax[0]*sn, c + ax[2]*ax[2]*C}};
        for (auto& p : rot.centers) {
            const std::array<double, 3> q = p;
            for (int r = 0; r < 3; ++r) p[r] = Rm[r][0]*q[0] + Rm[r][1]*q[1] + Rm[r][2]*q[2];
        }
    }
    std::vector<double> raw_rot, cal_rot;
    {
        exo::BallisticTransport bt_rot;
        exo::TransportCallOptions o = raw_opt;
        o.calibration_out = &cal_rot;
        bt_rot.computeTransportProduct(rot.centers, rot.areas, temps, e1, raw_rot, 1e-10, o);
    }
    const Diff noise = diff(raw_rot, raw_direct), noise_c = diff(cal_rot, cal_direct);
    std::printf("  direct-kernel noise floor (rigid rotation): raw %.2e, calibration %.2e per face\n",
                noise.face, noise_c.face);

    const Diff dr = diff(raw_op, raw_direct), dc = diff(op.calibration(), cal_direct);
    const double lim_r = std::max(1e-6, 2.0 * noise.face), lim_c = std::max(1e-6, 2.0 * noise_c.face);
    check(dr.face < lim_r, "raw deposition vs the direct kernel: max per-face rel. diff (< max(1e-6, 2 x noise))",
          dr.face, lim_r);
    check(dc.face < lim_c, "calibration vs the direct kernel: max per-face rel. diff (< max(1e-6, 2 x noise))",
          dc.face, lim_c);
    // Every source scatters exactly (1 - escape) of its emission.
    const double target_total = sum(e1) - tt.jeans_escaped;
    check(std::abs(tt.deposited / target_total - 1.0) < 1e-12,
          "operator raw total = emission - Jeans escape (per-source normalization)",
          std::abs(tt.deposited / target_total - 1.0), 1e-12);
    check(tt.unscattered == 0.0, "no unscattered emission", tt.unscattered, 0.0);
    if (!jeans) check(tt.jeans_escaped == 0.0, "no Jeans escape when it is off", tt.jeans_escaped, 0.0);

    // (3) finalized deposition vs a full direct call.
    exo::TransportCallOptions full_opt;
    full_opt.apply_jeans_escape = jeans;
    full_opt.private_index_if_needed = true;
    std::vector<double> dep_direct, dep_op;
    bt.computeTransportProduct(g.centers, g.areas, temps, e1, dep_direct, 1e-10, full_opt);
    exo::BallisticTransport::finalizeTransport(g.areas, raw_op, op.calibration(), dep_op);
    const Diff dd = diff(dep_op, dep_direct);
    check(dd.face < lim_r, "finalized deposition vs a full direct call: max per-face rel. diff", dd.face, lim_r);
    check(std::abs(sum(dep_op) / tt.deposited - 1.0) < 1e-12,
          "finalized deposition conserves the raw total (row bias + mass fix)",
          std::abs(sum(dep_op) / tt.deposited - 1.0), 1e-12);
    check(std::abs(sum(dep_op) / sum(dep_direct) - 1.0) < 1e-9,
          "finalized total vs the direct call", std::abs(sum(dep_op) / sum(dep_direct) - 1.0), 1e-9);

    // (4) delta update: R(e1) + K (e2 - e1) over the changed sources = K e2.
    std::vector<double> delta(n, 0.0), r_delta, raw_e2;
    for (size_t i = 0; i < n; ++i) if (changed[i]) delta[i] = e2[i] - e1[i];
    op.applyRaw(delta, &changed, r_delta);
    op.applyRaw(e2, nullptr, raw_e2);
    std::vector<double> upd(n);
    for (size_t j = 0; j < n; ++j) upd[j] = raw_op[j] + r_delta[j];
    const Diff du = diff(upd, raw_e2);
    check(du.face < 1e-12, "delta update R(e1) + K(e2 - e1) = K e2 (linearity)", du.face, 1e-12);
    // A masked application only sees masked sources.
    std::vector<double> masked_e1, e1_only(n, 0.0);
    for (size_t i = 0; i < n; ++i) if (changed[i]) e1_only[i] = e1[i];
    std::vector<double> r_only;
    op.applyRaw(e1, &changed, masked_e1);
    op.applyRaw(e1_only, nullptr, r_only);
    check(diff(masked_e1, r_only).field == 0.0, "mask selects exactly the masked sources (bit-identical)",
          diff(masked_e1, r_only).field, 0.0);

    // (5) thread-count invariance.
    const int nt = omp_get_max_threads();
    omp_set_num_threads(1);
    std::vector<double> r1;
    op.applyRaw(e1, nullptr, r1);
    exo::FixedKernelTransportOperator op1;
    op1.build(g.centers, g.areas, T, jeans, bt.bodyRadius(), 1e-10);
    const std::vector<double> cal1 = op1.calibration();  // built at 1 thread
    omp_set_num_threads(std::max(4, nt));
    std::vector<double> r4;
    op.applyRaw(e1, nullptr, r4);
    omp_set_num_threads(nt);
    bool same = (r1 == r4) && (cal1 == op.calibration());
    check(same, "1 thread vs 4 threads: raw product and calibration bit-identical", same ? 0.0 : 1.0, 0.0);

    if (timing) {
        std::printf("  timing (%d threads): direct raw call (with calibration) %.3f s; operator build (2 sweeps)"
                    " %.3f s, application %.3f s (%.1fx per call)\n",
                    omp_get_max_threads(), s_direct, op.buildSeconds(), s_op, s_direct / std::max(s_op, 1e-9));
    }
}

int main(int argc, char** argv) {
    const int levels = (argc > 1) ? std::atoi(argv[1]) : 5;
    const double R = exo::constants::EUROPA_RADIUS;
    const Geo g = geometry(make_icosphere(levels, R), R);
    // The sputter launch temperature (U = 0.054 eV) and a sublimation-like one.
    run_case(g, 677.3, false, "sputter launch temperature", true);
    run_case(g, 150.0, true, "150 K with Jeans escape (cutoff < pi)", true);
    std::printf("\n%d / %d checks passed\n", g_total - g_fail, g_total);
    return g_fail == 0 ? 0 : 1;
}
