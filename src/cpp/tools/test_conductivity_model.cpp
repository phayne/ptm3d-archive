/**
 * @file test_conductivity_model.cpp
 * @brief Anchor tests for the selectable regolith conductivity laws.
 *
 * Three things are checked, in order of how badly each would burn a campaign:
 *
 *  A. Martinez & Siegler (2021) Eq. 9/10 reproduce values computed
 *     independently from the paper's published coefficients, at temperatures
 *     spanning ultracold PSR floors (10 K) to lunar noon (400 K).
 *  B. The float32 OpenCL device mirror in
 *     `src/cpp/gpu/thermal_opencl.cpp::getEmbeddedKernelSource()` agrees with
 *     the double-precision host implementation to within float32 tolerance.
 *     This test carries its own byte-for-byte copy of the device expression, so
 *     a drift between the two files is caught here rather than in a production
 *     run. (An actual OpenCL execution test would need a device; this checks
 *     the ALGEBRA, which is what silently drifts.)
 *  C. The reference dispatch form `conductivityK(Hayne2017, ...)` is bit-for-bit
 *     identical to the historical inline expression `kc * (1 + R350*T^3)`. NOTE
 *     the solvers do not call it -- they branch outside their inner loops so the
 *     Hayne SIMD code stays textually unchanged -- so this pins the WRITTEN-DOWN
 *     law, not the solver. Only a byte-identical production run proves the solver.
 *
 * Reference values in test A were computed independently in Python from the
 * paper's Table (under Eq. 9) and Eq. 10 coefficients, NOT by running this
 * code. See docs/martinez-conductivity-model.md.
 */

#include "../thermal/conductivity_model.h"

#include <cmath>
#include <cstdio>
#include <string>

using thermal::ConductivityModel;

static int g_failures = 0;
static int g_checks   = 0;

static void check_rel(const char* what, double got, double expect, double rtol) {
    ++g_checks;
    const double denom = (std::abs(expect) > 0.0) ? std::abs(expect) : 1.0;
    const double rel   = std::abs(got - expect) / denom;
    const bool ok = (rel <= rtol);
    if (!ok) ++g_failures;
    std::printf("  [%s] %-42s got %.10g  expect %.10g  rel %.3e (tol %.1e)\n",
                ok ? "PASS" : "FAIL", what, got, expect, rel, rtol);
}

static void check_exact(const char* what, double got, double expect) {
    ++g_checks;
    const bool ok = (got == expect);
    if (!ok) ++g_failures;
    std::printf("  [%s] %-42s got %.17g  expect %.17g  %s\n",
                ok ? "PASS" : "FAIL", what, got, expect,
                ok ? "(bit-identical)" : "(DIFFERS)");
}

// ---------------------------------------------------------------------------
// Byte-for-byte transcription of the OpenCL device functions. Kept literally in
// float so the comparison exercises the same rounding the GPU performs.
// If you edit thermal_opencl.cpp's martinez_kam/martinez_k, edit these too --
// this test exists precisely to make that omission loud.
// ---------------------------------------------------------------------------
#define PTM_MARTINEZ_TMIN 2.7f
static float device_martinez_kam_at_or_above(float t) {
    float it = 1.0f / t;
    float it2 = it * it;
    float it3 = it2 * it;
    float it4 = it2 * it2;
    float t2 = t * t;
    float t3 = t2 * t;
    float t4 = t2 * t2;
    return -0.203297f
         + (-11.472f)     * it4
         + ( 22.5793f)    * it3
         + (-14.3084f)    * it2
         + (  3.41742f)   * it
         + (  0.01101f)   * t
         + ( -2.8049e-5f) * t2
         + (  3.35837e-8f)* t3
         + (-1.40021e-11f)* t4;
}
static float device_martinez_k(float T, float rho) {
    float t = std::fmax(T, PTM_MARTINEZ_TMIN);
    float a = 5.0821e-6f  * rho - 5.1e-3f;
    float b = 2.0022e-13f * rho - 1.953e-10f;
    return a * device_martinez_kam_at_or_above(t) + b * (t * t * t);
}

int main() {
    std::printf("=== PTM3D conductivity-model anchor tests ===\n\n");

    // -----------------------------------------------------------------------
    // A. Martinez Eq. 9 -- k_am(T), amorphous endmember [W m^-1 K^-1]
    //    Independent reference values (Python, double precision, paper coeffs).
    // -----------------------------------------------------------------------
    std::printf("A1. Martinez & Siegler (2021) Eq. 9  k_am(T)\n");
    struct KamRef { double T, k; };
    // Reference values from 40-digit exact-decimal arithmetic on the paper's
    // coefficients (Python `decimal`), so the tolerance below tests THIS code's
    // evaluation order, not the reference's rounding.
    const KamRef kam_ref[] = {
        {  10.0, 0.124121643679      },  // ultracold; paper quotes +-15% below 10 K
        {  30.0, 0.2014922033841852  },  // deep PSR floor
        {  60.0, 0.4164853050395556  },  // PSR wall / shadowed slope
        { 100.0, 0.68216231458       },  // cold-trap threshold region
        { 150.0, 0.9455104629883333  },
        { 200.0, 1.1397414452425     },  // paper's +-3% band starts here
        { 250.0, 1.279632297113368   },
        { 300.0, 1.379869142631852   },  // amorphous SiO2 ~1.4 W/m/K at 300 K
        { 350.0, 1.453630404118973   },
        { 400.0, 1.512220514853438   },  // lunar noon
    };
    for (const auto& r : kam_ref) {
        char buf[64];
        std::snprintf(buf, sizeof buf, "k_am(%.0f K)", r.T);
        check_rel(buf, thermal::martinezKam(r.T), r.k, 1e-12);
    }

    // -----------------------------------------------------------------------
    // A2. Martinez Eq. 10 -- k_eff(T, rho). rho = 1300 kg/m^3 is the paper's
    //     own reference density in Figures 1 and 3.
    // -----------------------------------------------------------------------
    std::printf("\nA2. Martinez & Siegler (2021) Eq. 10  k_eff(T, rho)\n");
    struct KRef { double T, rho, k; };
    const KRef k_ref[] = {
        {  30.0, 1100.0, 9.946707624129984e-05},   // Moon surface node
        { 100.0, 1100.0, 0.0003594130044617198},
        { 250.0, 1100.0, 0.001017135261597655 },
        {  30.0, 1300.0, 0.0003053489696050534},
        { 100.0, 1300.0, 0.001092820424247123 },
        { 250.0, 1300.0, 0.002943466621029625 },
        { 300.0, 1300.0, 0.00383371223327769  },  // Fig. 3 "New 1300"
        { 400.0, 1300.0, 0.00643761201634512  },  // Fig. 3 "New 1300", right edge
        {  30.0, 1800.0, 0.0008200537030144371},  // Moon deep node
        { 100.0, 1800.0, 0.002926338973710632 },
        { 250.0, 1800.0, 0.007759295019609549 },
    };
    for (const auto& r : k_ref) {
        char buf[64];
        std::snprintf(buf, sizeof buf, "k(%.0f K, %.0f kg/m3)", r.T, r.rho);
        check_rel(buf, thermal::martinezK(r.T, r.rho), r.k, 1e-12);
    }

    // -----------------------------------------------------------------------
    // A3. Domain guards, measured rather than asserted by fiat.
    // -----------------------------------------------------------------------
    std::printf("\nA3. Domain guards\n");
    ++g_checks;
    {
        const bool ok = (std::abs(thermal::kMartinezRhoMin - 1003.5221660337263) < 1e-6);
        if (!ok) ++g_failures;
        std::printf("  [%s] %-42s %.10f kg/m3\n", ok ? "PASS" : "FAIL",
                    "Eq.10 solid-term sign flip density", thermal::kMartinezRhoMin);
    }
    ++g_checks;
    {
        // Positive k across every temperature PTM3D can produce (its own floor
        // is ABSOLUTE_ZERO_K = 2.7 K) at the lunar density span.
        bool ok = true;
        double worst_T = 0, worst_k = 1e30;
        for (double rho = 1100.0; rho <= 1800.0 + 1e-9; rho += 25.0) {
            for (double T = 2.7; T <= 500.0; T += 0.1) {
                const double k = thermal::martinezK(T, rho);
                if (k < worst_k) { worst_k = k; worst_T = T; }
                if (!(k > 0.0) || !std::isfinite(k)) ok = false;
            }
        }
        if (!ok) ++g_failures;
        std::printf("  [%s] %-42s min k = %.6e at T = %.1f K\n", ok ? "PASS" : "FAIL",
                    "k > 0 over T in [2.7,500], rho 1100-1800", worst_k, worst_T);
    }
    ++g_checks;
    {
        // The T-clamp is a domain restriction, not a silent rescue: below the
        // floor the law must return exactly its value AT the floor.
        const bool ok = (thermal::martinezK(1.0, 1300.0) ==
                         thermal::martinezK(thermal::kMartinezTMin, 1300.0));
        if (!ok) ++g_failures;
        std::printf("  [%s] %-42s\n", ok ? "PASS" : "FAIL",
                    "T < 2.7 K clamps to the T=2.7 K value");
    }
    ++g_checks;
    {
        const bool ok = !thermal::martinezDensityIsValid(900.0) &&
                        !thermal::martinezDensityIsValid(1000.0) &&
                         thermal::martinezDensityIsValid(1100.0);
        if (!ok) ++g_failures;
        std::printf("  [%s] %-42s\n", ok ? "PASS" : "FAIL",
                    "density validity gate rejects rho < 1003.5");
    }

    // -----------------------------------------------------------------------
    // B. Host (double) vs OpenCL device mirror (float32).
    // -----------------------------------------------------------------------
    std::printf("\nB. Host double vs OpenCL float32 device mirror\n");
    {
        double worst = 0.0, worst_T = 0.0, worst_rho = 0.0;
        for (double rho = 1100.0; rho <= 1800.0 + 1e-9; rho += 50.0) {
            for (double T = 3.0; T <= 420.0; T += 0.5) {
                const double h = thermal::martinezK(T, rho);
                const double d = static_cast<double>(
                    device_martinez_k(static_cast<float>(T), static_cast<float>(rho)));
                const double rel = std::abs(d - h) / std::abs(h);
                if (rel > worst) { worst = rel; worst_T = T; worst_rho = rho; }
            }
        }
        ++g_checks;
        // 1e-4 leaves room for float32 cancellation in Eq. 9 (the sum of terms
        // is ~8x the result at 400 K) without letting an algebra drift through.
        const bool ok = (worst <= 1e-4);
        if (!ok) ++g_failures;
        std::printf("  [%s] %-42s worst rel %.3e at T=%.1f K, rho=%.0f\n",
                    ok ? "PASS" : "FAIL", "device mirror matches host",
                    worst, worst_T, worst_rho);
    }

    // -----------------------------------------------------------------------
    // C. Hayne default path is bit-for-bit the historical expression.
    // -----------------------------------------------------------------------
    std::printf("\nC. Hayne 2017 default path bit-identity\n");
    {
        const double R350 = 2.7 / (350.0 * 350.0 * 350.0);
        for (double T : {30.0, 100.0, 250.0, 400.0}) {
            for (double kc : {7.4e-4, 1.5e-3, 3.4e-3}) {
                const double legacy = kc * (1.0 + R350 * (T * T * T));
                char buf[64];
                std::snprintf(buf, sizeof buf, "k(%.0f K, kc=%.1e)", T, kc);
                check_exact(buf,
                            thermal::conductivityK(ConductivityModel::Hayne2017,
                                                   T, kc, /*rho=*/1300.0, R350),
                            legacy);
            }
        }
    }

    // -----------------------------------------------------------------------
    // D. Config-string parsing: valid names accepted, junk REJECTED (a silent
    //    fallback to the default would misattribute a whole campaign).
    // -----------------------------------------------------------------------
    std::printf("\nD. conductivity_model string parsing\n");
    {
        ConductivityModel m;
        struct Case { const char* s; bool ok; ConductivityModel want; };
        const Case cases[] = {
            {"",              true,  ConductivityModel::Hayne2017},
            {"hayne2017",     true,  ConductivityModel::Hayne2017},
            {"Hayne_2017",    true,  ConductivityModel::Hayne2017},
            {"martinez2021",  true,  ConductivityModel::Martinez2021},
            {"Martinez-2021", true,  ConductivityModel::Martinez2021},
            {"MARTINEZ 2021", true,  ConductivityModel::Martinez2021},
            {"martinez2019",  false, ConductivityModel::Hayne2017},
            {"woodsrobinson", false, ConductivityModel::Hayne2017},
            {"true",          false, ConductivityModel::Hayne2017},
        };
        for (const auto& c : cases) {
            m = ConductivityModel::Martinez2021;   // poison, so a no-write shows
            const bool got = thermal::parseConductivityModel(c.s, m);
            ++g_checks;
            const bool ok = (got == c.ok) && (!c.ok || m == c.want);
            if (!ok) ++g_failures;
            std::printf("  [%s] parse(\"%s\") -> %s%s\n", ok ? "PASS" : "FAIL", c.s,
                        got ? "accepted " : "REJECTED",
                        got ? thermal::conductivityModelName(m) : "");
        }
    }

    std::printf("\n=== %d checks, %d failures ===\n", g_checks, g_failures);
    return (g_failures == 0) ? 0 : 1;
}
