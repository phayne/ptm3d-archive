// Gate for the LIT-FACE SEED ANCHOR (src/cpp/thermal/eq_seed.h).
//
// Why this file exists: the seed's surface anchor was T_w = ((<F>+Qb)/(eps
// sigma))^0.25, which is the exact fourth moment and NOT the cycle mean the
// conductive profile hangs from. On the shipped lunar 80N crater that is
// +45.06 K mean / 83.15 K max. The replacement is a lumped-substrate anchor
// whose two limits are exact boundary conditions -- so they are UNIT TESTS
// here, not statistics: they hold, or the code is wrong.
//
// Build: cmake --build build --target test_eq_anchor
// Run:   ./build/bin/test_eq_anchor              (exit 0 = pass)
//        ./build/bin/test_eq_anchor --bench      per-face cost
//        ./build/bin/test_eq_anchor --score in.bin out.csv
//                                   score a banked forcing field (see
//                                   readCase() for the format)

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "../thermal/eq_seed.h"
#include "../thermal/thermal_model.h"

namespace {

int g_fail = 0;

void check(bool ok, const std::string& what, const std::string& detail = "") {
    std::printf("  [%s] %s%s%s\n", ok ? "PASS" : "FAIL", what.c_str(),
                detail.empty() ? "" : " -- ", detail.c_str());
    if (!ok) ++g_fail;
}

const double kMoonDay = 29.530588 * 86400.0;
const double kMoonCp[4] = {2.7431, 2.3616e-3, -1.2340e-5, 8.9093e-9};
const double kMoonCp0   = -3.6125;

// A lunar-like graded column: k_c(z), rho(z) exponential with H, as
// thermal_model.cpp builds them.
struct Column {
    std::vector<double> z, kc, rho;
    explicit Column(int n = 50, double z_max = 1.0, double H = 0.06,
                    double ks = 7.4e-4, double kd = 3.4e-3,
                    double rhos = 1100.0, double rhod = 1800.0) {
        z.resize(n); kc.resize(n); rho.resize(n);
        const double r = std::pow(z_max / 3.3e-3, 1.0 / (n - 2));
        double dz = 3.3e-3;
        z[0] = 0.0;
        for (int i = 1; i < n; ++i) { z[i] = z[i - 1] + dz; dz *= r; }
        const double scale = z_max / z[n - 1];
        for (int i = 0; i < n; ++i) z[i] *= scale;
        for (int i = 0; i < n; ++i) {
            const double df = std::exp(-z[i] / H);
            kc[i]  = kd   + (ks   - kd  ) * df;
            rho[i] = rhod + (rhos - rhod) * df;
        }
    }
};

struct Case {
    thermal::EqSeedInput   in;
    thermal::EqSeedForcing fc;
};

Case moonCase(const Column& c, const std::vector<double>& F, double Qb = 0.018) {
    Case k;
    k.in.Qb         = Qb;
    k.in.emissivity = 0.95;
    k.in.sigma      = thermal::SIGMA_SB;
    k.in.z    = c.z.data();
    k.in.kc   = c.kc.data();
    k.in.rho  = c.rho.data();
    k.in.n    = static_cast<int>(c.z.size());
    k.in.R350 = 2.7 / (350.0 * 350.0 * 350.0);
    k.in.cond = thermal::ConductivityModel::Hayne2017;
    double s = 0.0;
    for (double v : F) s += v;
    k.in.mean_absorbed_flux = s / static_cast<double>(F.size());
    k.fc.flux       = F.data();
    k.fc.n_samples  = static_cast<int>(F.size());
    k.fc.span_s     = kMoonDay;
    k.fc.cycle_s    = kMoonDay;
    k.fc.cp0        = kMoonCp0;
    k.fc.cp_coeff   = kMoonCp;
    k.fc.n_cp_coeff = 4;
    k.in.forcing    = &k.fc;
    return k;
}

std::vector<double> cosineForcing(int n, double peak, double bias = 0.0) {
    std::vector<double> F(n);
    for (int i = 0; i < n; ++i) {
        const double ph = 2.0 * M_PI * i / n;
        const double s  = std::cos(ph) - bias;
        F[i] = peak * (s > 0.0 ? s : 0.0);
    }
    return F;
}

// ---------------------------------------------------------------------------
double coldBracket(const std::vector<double>& F, double Qb, double a) {
    double s = 0.0;
    for (double v : F) s += std::sqrt(std::sqrt((std::max(0.0, v) + Qb) / a));
    return s / static_cast<double>(F.size());
}
double warmBracket(const std::vector<double>& F, double Qb, double a) {
    double s = 0.0;
    for (double v : F) s += std::max(0.0, v);
    return std::sqrt(std::sqrt((s / static_cast<double>(F.size()) + Qb) / a));
}

// ===========================================================================
// A1  THE TWO LIMITS -- boundary conditions, so they hold exactly
// ===========================================================================
void testLimits() {
    std::printf("\nA1  exact limits of the lumped anchor (G->0 cold, G->inf warm)\n");
    const double eps = 0.95, sig = thermal::SIGMA_SB, a = eps * sig;
    const double Qb = 0.018;
    struct Shape { const char* name; std::vector<double> F; };
    std::vector<Shape> shapes = {
        {"cos, 118 samples",        cosineForcing(118, 1361.0 * 0.88)},
        {"cos-bias 0.5, 96",        cosineForcing(96,  1361.0 * 0.88, 0.5)},
        {"grazing (bias 0.93), 118",cosineForcing(118, 1361.0 * 0.88, 0.93)},
        {"flash 4/256",             std::vector<double>()},
        {"prime length 97",         cosineForcing(97,  600.0)},
    };
    shapes[3].F.assign(256, 0.0);
    for (int i = 120; i < 124; ++i) shapes[3].F[i] = 900.0;

    double worst0 = 0.0, worstInf = 0.0;
    for (const auto& s : shapes) {
        double Tw = 0.0, Tc = 0.0;
        int it = 0;
        const double t0 = thermal::eqSeedLumpedAnchor(
            s.F.data(), 1, static_cast<int>(s.F.size()), Qb, eps, sig, 0.0,
            &Tw, &Tc, &it);
        const double tinf = thermal::eqSeedLumpedAnchor(
            s.F.data(), 1, static_cast<int>(s.F.size()), Qb, eps, sig, 1e12,
            &Tw, &Tc, &it);
        const double e0 = std::fabs(t0 - coldBracket(s.F, Qb, a));
        const double ei = std::fabs(tinf - warmBracket(s.F, Qb, a));
        worst0   = std::max(worst0, e0);
        worstInf = std::max(worstInf, ei);
        std::printf("      %-28s  G=0 err %.3e K   G=1e12 err %.3e K"
                    "  [T_c %.3f  T_w %.3f]\n",
                    s.name, e0, ei, Tc, Tw);
    }
    check(worst0 <= 1e-9, "G = 0 returns the cold bracket to <= 1e-9 K",
          "worst " + std::to_string(worst0));
    check(worstInf <= 1e-9, "G -> inf returns T_w to <= 1e-9 K",
          "worst " + std::to_string(worstInf));

    // Monotone in G, and inside the bracket for every intermediate G.
    const std::vector<double>& F = shapes[0].F;
    double prev = -1.0;
    bool mono = true, inside = true;
    const double Tc = coldBracket(F, Qb, a), Tw = warmBracket(F, Qb, a);
    for (double G : {0.0, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 1e3, 1e6}) {
        double dummy1, dummy2;
        int it;
        const double T = thermal::eqSeedLumpedAnchor(
            F.data(), 1, static_cast<int>(F.size()), Qb, eps, sig, G,
            &dummy1, &dummy2, &it);
        if (T < prev - 1e-12) mono = false;
        if (T < Tc - 1e-9 || T > Tw + 1e-9) inside = false;
        prev = T;
    }
    check(mono,   "T_d is monotone non-decreasing in G");
    check(inside, "T_d stays inside [cold bracket, warm bracket] for all G");
}

// ===========================================================================
// A2  ZERO SWING IS BYTE-IDENTICAL TO TODAY
// ===========================================================================
void testZeroSwing() {
    std::printf("\nA2  zero-swing faces are bit-for-bit the pre-anchor seed\n");
    Column c;
    for (double Fv : {0.0, 1e-9, 3.2, 140.0, 1361.0}) {
        std::vector<double> F(118, Fv);
        Case k = moonCase(c, F);
        std::vector<double> Ta(c.z.size()), Tb(c.z.size());
        const auto with = thermal::eqSeedColumn(k.in, Ta.data());
        k.in.forcing = nullptr;                       // exactly today's call
        const auto without = thermal::eqSeedColumn(k.in, Tb.data());
        bool same = (with.T_surface == without.T_surface);
        for (size_t i = 0; i < Ta.size(); ++i) same = same && (Ta[i] == Tb[i]);
        check(same, "flat F = " + std::to_string(Fv) +
                        " W/m^2: identical bits at every node",
              std::string("status=") +
                  thermal::eqSeedAnchorStatusName(with.anchor_status));
    }
    // ... and the explicit opt-out is byte-identical for ANY forcing.
    std::vector<double> F = cosineForcing(118, 1200.0);
    Case k = moonCase(c, F);
    k.in.anchor = thermal::EqSeedAnchor::Warm;
    std::vector<double> Ta(c.z.size()), Tb(c.z.size());
    const auto with = thermal::eqSeedColumn(k.in, Ta.data());
    k.in.forcing = nullptr;
    const auto without = thermal::eqSeedColumn(k.in, Tb.data());
    bool same = (with.T_surface == without.T_surface);
    for (size_t i = 0; i < Ta.size(); ++i) same = same && (Ta[i] == Tb[i]);
    check(same, "seed_anchor: warm is byte-identical on a full-swing face");
}

// ===========================================================================
// FFT and admittance -- the two derived ingredients
// ===========================================================================
void testSpectrum() {
    std::printf("\nspectrum: one-sided power vs a direct DFT, and Parseval\n");
    double worst_dft = 0.0, worst_var = 0.0;
    for (int n : {2, 3, 8, 16, 17, 96, 97, 118, 120, 256, 360, 1439}) {
        std::vector<double> x(n);
        unsigned s = 12345u + n;
        for (int i = 0; i < n; ++i) {
            s = s * 1664525u + 1013904223u;
            x[i] = 100.0 * (static_cast<double>(s >> 8) / 16777216.0) +
                   50.0 * std::cos(2.0 * M_PI * 3.0 * i / n);
        }
        std::vector<double> P;
        thermal::eqSeedOneSidedPower(x.data(), 1, n, P);
        // direct DFT reference
        double worst = 0.0;
        const int nb = n / 2 + 1;
        for (int k = 1; k < nb; ++k) {
            double re = 0.0, im = 0.0;
            for (int j = 0; j < n; ++j) {
                const double ang = -2.0 * M_PI * j * k / n;
                re += x[j] * std::cos(ang);
                im += x[j] * std::sin(ang);
            }
            re /= n; im /= n;
            double pk = 2.0 * (re * re + im * im);
            if (n % 2 == 0 && k == nb - 1) pk *= 0.5;
            worst = std::max(worst, std::fabs(pk - P[k]));
        }
        // Parseval: sum_{k>=1} P_k == var(x)
        double m = 0.0;
        for (double v : x) m += v;
        m /= n;
        double var = 0.0;
        for (double v : x) var += (v - m) * (v - m);
        var /= n;
        double S0 = 0.0;
        for (int k = 1; k < nb; ++k) S0 += P[k];
        const double rel = (var > 0.0) ? std::fabs(S0 - var) / var : 0.0;
        worst_dft = std::max(worst_dft, worst);
        worst_var = std::max(worst_var, rel);
    }
    check(worst_dft < 1e-8, "power spectrum matches a direct DFT (n = 2..1439)",
          "max abs " + std::to_string(worst_dft));
    check(worst_var < 1e-12, "Parseval: sum_{n>=1} P_n == var(F)",
          "max rel " + std::to_string(worst_var));
}

void testAdmittance() {
    std::printf("\nadmittance: layered recursion -> sqrt(k rho cp) when uniform\n");
    const int n = 400;
    std::vector<double> z(n), kc(n), rho(n);
    const double kv = 1.2e-3, rv = 1500.0;
    const double omega = 2.0 * M_PI / kMoonDay;
    const double cpv = kMoonCp0 + 200.0 * (kMoonCp[0] + 200.0 * (kMoonCp[1] +
                       200.0 * (kMoonCp[2] + 200.0 * kMoonCp[3])));
    const double kappa = kv / (rv * cpv);
    const double zmax = 30.0 * std::sqrt(2.0 * kappa / omega);
    for (int i = 0; i < n; ++i) {
        z[i] = zmax * i / (n - 1);
        kc[i] = kv; rho[i] = rv;
    }
    thermal::EqSeedInput in;
    in.z = z.data(); in.kc = kc.data(); in.rho = rho.data(); in.n = n;
    in.R350 = 0.0; in.cond = thermal::ConductivityModel::Hayne2017;
    thermal::EqSeedForcing fc;
    fc.cp0 = kMoonCp0; fc.cp_coeff = kMoonCp; fc.n_cp_coeff = 4;
    const double gam = thermal::eqSeedGammaEff(in, fc, omega, 200.0);
    const double exact = std::sqrt(kv * rv * cpv);
    check(std::fabs(gam - exact) / exact < 1e-6,
          "homogeneous column: Gamma_eff == sqrt(k rho cp)",
          "got " + std::to_string(gam) + " vs " + std::to_string(exact));

    // cp <= 0 must be refused, not substituted.
    thermal::EqSeedForcing bad = fc;
    bad.cp0 = -1e6; bad.n_cp_coeff = 0; bad.cp_coeff = nullptr;
    check(thermal::eqSeedGammaEff(in, bad, omega, 200.0) < 0.0,
          "cp <= 0 is refused (negative return), never substituted");
}

// ===========================================================================
// The regime gates -- each must FIRE, and each must report WHY
// ===========================================================================
void testGates() {
    std::printf("\ngates: every fallback fires and names itself\n");
    Column c;
    std::vector<double> F = cosineForcing(118, 1200.0);
    Case base = moonCase(c, F);
    {
        Case k = base; k.in.forcing = &k.fc;
        k.fc.radiation_coupled = true;
        const auto r = thermal::eqSeedColumn(k.in, nullptr);
        check(r.anchor_status == thermal::EqSeedAnchorStatus::FallbackRadiationCoupled &&
              r.T_surface == r.T_warm, "radiation-coupled -> T_w, reported");
    }
    {
        Case k = base; k.in.forcing = &k.fc;
        k.fc.latent_heat = true;
        const auto r = thermal::eqSeedColumn(k.in, nullptr);
        check(r.anchor_status == thermal::EqSeedAnchorStatus::FallbackLatentHeat,
              "latent heat -> T_w, reported");
    }
    {
        std::vector<double> Ff(118, 0.0);
        Ff[40] = 800.0; Ff[41] = 900.0;             // 2 lit samples
        Case k = moonCase(c, Ff); k.in.forcing = &k.fc;
        const auto r0 = thermal::eqSeedColumn(k.in, nullptr);
        check(r0.anchor_status == thermal::EqSeedAnchorStatus::LumpedInputLimited &&
              r0.n_lit_samples == 2,
              "2 lit samples -> DISCLOSED input-limited, anchor still runs");
        k.fc.min_lit_samples = 4;
        const auto r = thermal::eqSeedColumn(k.in, nullptr);
        check(r.anchor_status == thermal::EqSeedAnchorStatus::FallbackFewLitSamples &&
              r.T_surface == r.T_warm,
              "min_lit_samples = 4 restores the refusal, reported");
    }
    {
        std::vector<double> Ff(118, 0.0);
        for (int i = 40; i < 46; ++i) Ff[i] = 800.0;   // 6 lit samples
        Case k = moonCase(c, Ff); k.in.forcing = &k.fc;
        const auto r = thermal::eqSeedColumn(k.in, nullptr);
        check(r.anchor_status == thermal::EqSeedAnchorStatus::Lumped,
              "6 lit samples -> the anchor runs");
    }
    {
        Case k = base; k.in.forcing = &k.fc;
        k.fc.span_s = 0.3 * kMoonDay;
        const auto r = thermal::eqSeedColumn(k.in, nullptr);
        check(r.anchor_status == thermal::EqSeedAnchorStatus::FallbackShortSpan,
              "span < 1 cycle -> not a cycle mean, reported");
    }
    {
        // 8 lunations of diurnal forcing with a strong seasonal envelope.
        const int ncyc = 8, ns = 8 * 118;
        std::vector<double> Fs(ns);
        for (int i = 0; i < ns; ++i) {
            const double season = 0.5 + 0.5 * std::cos(2.0 * M_PI * i / ns);
            const double d = std::cos(2.0 * M_PI * ncyc * i / ns);
            Fs[i] = 1200.0 * season * (d > 0.0 ? d : 0.0);
        }
        Case k = moonCase(c, Fs); k.in.forcing = &k.fc;
        k.fc.span_s = ncyc * kMoonDay;
        const auto r = thermal::eqSeedColumn(k.in, nullptr);
        check(r.anchor_status == thermal::EqSeedAnchorStatus::FallbackTwoTimescale,
              "diurnal + annual -> two-timescale, reported",
              "sub-diurnal variance fraction " +
                  std::to_string(r.subdiurnal_variance_fraction));
    }
    {
        // The SAME 8 lunations with no seasonal envelope must NOT fall back.
        const int ncyc = 8, ns = 8 * 118;
        std::vector<double> Fs(ns);
        for (int i = 0; i < ns; ++i) {
            const double d = std::cos(2.0 * M_PI * ncyc * i / ns);
            Fs[i] = 1200.0 * (d > 0.0 ? d : 0.0);
        }
        Case k = moonCase(c, Fs); k.in.forcing = &k.fc;
        k.fc.span_s = ncyc * kMoonDay;
        const auto r = thermal::eqSeedColumn(k.in, nullptr);
        check(r.anchor_status == thermal::EqSeedAnchorStatus::Lumped,
              "8 identical lunations -> single timescale, the anchor runs",
              "sub-diurnal variance fraction " +
                  std::to_string(r.subdiurnal_variance_fraction));
    }
}

// ===========================================================================
// Sanity: the anchor moves a lit face DOWN, and dark faces not at all
// ===========================================================================
void testDirection() {
    std::printf("\ndirection: Jensen says the anchor can only move T DOWN\n");
    Column c;
    bool ok = true;
    std::string worst;
    for (double bias : {0.0, 0.3, 0.6, 0.9}) {
        std::vector<double> F = cosineForcing(118, 1361.0 * 0.88, bias);
        Case k = moonCase(c, F);
        const auto r = thermal::eqSeedColumn(k.in, nullptr);
        Case k2 = moonCase(c, F);
        const auto r2 = thermal::eqSeedColumn(k2.in, nullptr);
        (void)r2;
        k.in.forcing = &k.fc;
        const auto ra = thermal::eqSeedColumn(k.in, nullptr);
        if (!(ra.T_surface <= r.T_surface + 1e-12)) ok = false;
        std::printf("      bias %.2f: T_w %8.3f  T_c %8.3f  T_d %8.3f"
                    "  (drop %6.3f K)  G %.4g\n",
                    bias, ra.T_warm, ra.T_cold, ra.T_surface,
                    ra.T_warm - ra.T_surface, ra.G);
    }
    check(ok, "the lumped anchor never sits above the warm bracket", worst);
}

// ===========================================================================
// A5  COST
// ===========================================================================
void bench() {
    std::printf("\nA5  cost per face\n");
    Column c;
    for (int ns : {48, 118, 360, 1440}) {
        std::vector<double> F = cosineForcing(ns, 1361.0 * 0.88, 0.2);
        Case k = moonCase(c, F);
        k.fc.span_s = kMoonDay;
        const int reps = std::max(50, 20000 / ns);
        // warm the plan cache
        thermal::eqSeedColumn(k.in, nullptr);
        std::vector<double> T(c.z.size());
        const auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < reps; ++i) thermal::eqSeedColumn(k.in, T.data());
        const auto t1 = std::chrono::high_resolution_clock::now();
        k.in.forcing = nullptr;
        const auto t2 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < reps; ++i) thermal::eqSeedColumn(k.in, T.data());
        const auto t3 = std::chrono::high_resolution_clock::now();
        const double us_a = std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
        const double us_w = std::chrono::duration<double, std::micro>(t3 - t2).count() / reps;
        std::printf("      %5d samples:  anchor %8.2f us/face   warm-only %6.3f us/face\n",
                    ns, us_a, us_w);
    }
}

// ===========================================================================
// --score: run the anchor over a banked forcing field
// ===========================================================================
struct Bank {
    int32_t n_samples = 0, n_faces = 0, n_nodes = 0, n_cp = 0;
    double span_s = 0, cycle_s = 0, eps = 0, sigma = 0, Qb = 0, R350 = 0, cp0 = 0;
    std::vector<double> cp, z, kc, rho, F;   // F is [sample][face]
};

bool readBank(const std::string& path, Bank& b) {
    std::ifstream fh(path, std::ios::binary);
    if (!fh) return false;
    auto rd = [&](void* p, size_t n) { fh.read(reinterpret_cast<char*>(p), n); };
    rd(&b.n_samples, 4); rd(&b.n_faces, 4); rd(&b.n_nodes, 4); rd(&b.n_cp, 4);
    rd(&b.span_s, 8); rd(&b.cycle_s, 8); rd(&b.eps, 8); rd(&b.sigma, 8);
    rd(&b.Qb, 8); rd(&b.R350, 8); rd(&b.cp0, 8);
    b.cp.resize(b.n_cp);   rd(b.cp.data(),  8 * b.n_cp);
    b.z.resize(b.n_nodes); rd(b.z.data(),   8 * b.n_nodes);
    b.kc.resize(b.n_nodes); rd(b.kc.data(), 8 * b.n_nodes);
    b.rho.resize(b.n_nodes); rd(b.rho.data(), 8 * b.n_nodes);
    b.F.resize(static_cast<size_t>(b.n_samples) * b.n_faces);
    rd(b.F.data(), 8 * b.F.size());
    return static_cast<bool>(fh);
}

int score(const std::string& in_path, const std::string& out_path,
          int min_lit) {
    Bank b;
    if (!readBank(in_path, b)) {
        std::fprintf(stderr, "cannot read bank '%s'\n", in_path.c_str());
        return 2;
    }
    std::fprintf(stderr, "[score] %d faces x %d samples, %d nodes, span %.6g s,"
                         " cycle %.6g s\n",
                 b.n_faces, b.n_samples, b.n_nodes, b.span_s, b.cycle_s);
    std::ofstream out(out_path);
    out << "face,T_warm,T_cold,T_anchor,T_seed,G,gamma_eff,n_samples,n_lit,status\n";
    out.precision(17);
    const auto t0 = std::chrono::high_resolution_clock::now();
    for (int f = 0; f < b.n_faces; ++f) {
        thermal::EqSeedInput in;
        in.Qb = b.Qb; in.emissivity = b.eps; in.sigma = b.sigma;
        in.z = b.z.data(); in.kc = b.kc.data(); in.rho = b.rho.data();
        in.n = b.n_nodes; in.R350 = b.R350;
        in.cond = thermal::ConductivityModel::Hayne2017;
        thermal::EqSeedForcing fc;
        fc.flux = b.F.data() + f;
        fc.stride = static_cast<size_t>(b.n_faces);
        fc.n_samples = b.n_samples;
        fc.span_s = b.span_s;
        fc.cycle_s = b.cycle_s;
        fc.cp0 = b.cp0;
        fc.cp_coeff = b.n_cp ? b.cp.data() : nullptr;
        fc.n_cp_coeff = b.n_cp;
        fc.min_lit_samples = min_lit;
        double s = 0.0;
        for (int t = 0; t < b.n_samples; ++t)
            s += b.F[static_cast<size_t>(t) * b.n_faces + f];
        in.mean_absorbed_flux = s / b.n_samples;
        in.forcing = &fc;
        const auto r = thermal::eqSeedColumn(in, nullptr);
        out << f << ',' << r.T_warm << ',' << r.T_cold << ',' << r.T_anchor
            << ',' << r.T_surface << ',' << r.G << ',' << r.gamma_eff << ',' << r.n_samples << ','
            << r.n_lit_samples << ','
            << thermal::eqSeedAnchorStatusName(r.anchor_status) << '\n';
    }
    const auto t1 = std::chrono::high_resolution_clock::now();
    std::fprintf(stderr, "[score] %.3f s total, %.1f us/face\n",
                 std::chrono::duration<double>(t1 - t0).count(),
                 std::chrono::duration<double, std::micro>(t1 - t0).count() /
                     b.n_faces);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc >= 2 && std::strcmp(argv[1], "--score") == 0) {
        if (argc < 4) {
            std::fprintf(stderr, "--score in.bin out.csv [min_lit_samples]\n");
            return 2;
        }
        return score(argv[2], argv[3], (argc >= 5) ? std::atoi(argv[4]) : 0);
    }
    std::printf("=== lit-face seed anchor gate (eq_seed.h) ===\n");
    testLimits();
    testZeroSwing();
    testSpectrum();
    testAdmittance();
    testGates();
    testDirection();
    if (argc >= 2 && std::strcmp(argv[1], "--bench") == 0) bench();
    std::printf("\n%s (%d failure%s)\n", g_fail ? "FAILED" : "ALL PASS",
                g_fail, g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}
