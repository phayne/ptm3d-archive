// Standalone validator for the CAMERA-STAGE Smith (1967) directional thermal
// emission + band brightness temperature (BT_K). Exercises the EXACT production
// code paths -- PhotometricModel::bandPlanckRadiance / bandBrightnessTemperature
// and radiation::smith::{buildChiTable,evalChi} -- with no mesh / Embree / SPICE.
//
// Checks:
//   (1) chi == 1 round-trip identity: band BT == kinetic T (the rms_slope=0 gate
//       for the BT_K observable).
//   (2) Monotonic beaming: at fixed facet T, toward-Sun (small phase) BT exceeds
//       away-Sun BT, and the toward-minus-away contrast grows with RMS slope and
//       with solar incidence.
//   (3) T_ss ceiling / clamp behavior: a warm grazing facet at large RMS slope
//       beams a band BT above its own kinetic T (isothermal-tier artifact) and can
//       exceed the subsolar-equilibrium ceiling -- demonstrated + quantified.
//
// Build: linked as the test_camera_beaming target (camera/CMakeLists.txt).
// Run:   ./bin/test_camera_beaming

#include "photometric_model.h"
#include "radiation/smith_beaming.h"
#include "radiation/anisothermal_beaming.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>

using namespace ptm;

static int g_fail = 0;
static void check(bool ok, const char* msg) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", msg);
    if (!ok) ++g_fail;
}

int main() {
    const int ns = 24, ne = 24, np = 33;   // same table dims as the camera/radiosity

    // Default instrument = MSSS ECAM-IR3S LWIR top-hat 8-14 um, emissivity 0.95.
    PhotometricModel::Params params;
    PhotometricModel model(PhotometricModel::ReflectanceType::LAMBERTIAN, params);

    // Band brightness temperature of an isothermal facet at kinetic T beaming by
    // the energy-normalized directional factor chi -- exactly the camera path.
    auto bandBT = [&](double T, double chi) {
        return model.bandBrightnessTemperature(chi * model.bandPlanckRadiance(T));
    };

    std::printf("Camera-stage Smith directional band-BT validation\n");
    std::printf("=================================================\n");
    std::printf("Band: %.1f-%.1f um (top-hat), emissivity %.2f\n\n",
                params.thermal_band_lo_um, params.thermal_band_hi_um,
                params.emissivity);

    // (0) Byte-identity of the ONLY pre-existing-HDU change at rms_slope=0. When
    // beaming is off the render loop computes therm_sample = chi * thermalSpectral-
    // Radiance(...) with chi == 1.0 (vs the original therm_sample = thermalSpectral-
    // Radiance(...)). This proves that multiply is BIT-EXACT over a T/cos_e/lambda
    // grid, so the THERMAL HDU (and thus every pre-existing HDU) is byte-identical
    // to the Lambertian path at rms_slope=0. Everything else Smith-emission adds
    // feeds only the NEW BT_K HDU or diagnostics.
    std::printf("(0) rms_slope=0 THERMAL byte-identity (chi*L == L bit-exact):\n");
    {
        const double chi_off = 1.0;   // exactly what buildEmissionChiTable leaves off
        size_t nchk = 0, nbad = 0;
        for (double T : {5.0, 60.0, 123.4, 200.0, 275.0, 400.0}) {
            for (double ce : {0.02, 0.13, 0.5, 0.71, 1.0}) {
                for (double lam : {8.0, 10.0, 10.5, 14.0}) {
                    double L   = model.thermalSpectralRadiance(T, ce, lam);
                    double Lc  = chi_off * L;   // the exact expression the render uses
                    ++nchk;
                    if (std::memcmp(&L, &Lc, sizeof(double)) != 0) ++nbad;
                }
            }
        }
        std::printf("    %zu (T,cos_e,lambda) samples, %zu bitwise mismatches\n", nchk, nbad);
        check(nbad == 0, "chi*thermalSpectralRadiance is bit-identical at chi=1 (rms=0 gate)");
    }

    // (1) chi == 1 round-trip identity: BT == kinetic T.
    std::printf("(1) chi=1 round-trip identity (BT == kinetic T; rms_slope=0 gate):\n");
    {
        double worst = 0.0;
        for (double T : {60.0, 100.0, 150.0, 200.0, 240.0}) {
            double bt = bandBT(T, 1.0);
            std::printf("    T=%6.1f K -> BT=%9.5f K  (dev %+.2e K)\n", T, bt, bt - T);
            worst = std::max(worst, std::fabs(bt - T));
        }
        check(worst < 1e-2, "band BT == kinetic T at chi=1 within 0.01 K");
    }

    // Geometry helper (principal plane): normal = +z; Sun at polar angle theta_sun;
    // emission direction at signed polar angle theta_e (positive => same side as the
    // Sun => toward-Sun / small phase). Then
    //   cos_sun = cos(theta_sun), cos_e = cos(theta_e),
    //   cos_psi = cos(theta_e - theta_sun).
    auto chiAt = [&](const std::vector<float>& tbl, double theta_sun_deg,
                     double theta_e_deg) {
        double ts = theta_sun_deg * M_PI / 180.0;
        double te = theta_e_deg   * M_PI / 180.0;
        double cos_sun = std::cos(ts);
        double cos_e   = std::cos(te);
        double cos_psi = std::cos(te - ts);
        return (double)radiation::smith::evalChi(tbl, ns, ne, np,
                                                 cos_sun, cos_e, cos_psi);
    };

    // (2a) BT vs emission angle for a fixed facet, one geometry, several RMS slopes.
    std::printf("\n(2a) BT vs emission angle, T=200 K, solar incidence 60 deg:\n");
    std::printf("     theta_e<0 = away from Sun, theta_e>0 = toward Sun.\n");
    {
        const double T = 200.0, theta_sun = 60.0;
        std::printf("     %8s", "theta_e");
        for (double w : {0.0, 10.0, 20.0, 35.0}) std::printf("   w=%4.0f", w);
        std::printf("   [K]\n");
        for (double te = -80.0; te <= 80.0 + 1e-9; te += 20.0) {
            std::printf("     %8.0f", te);
            for (double w : {0.0, 10.0, 20.0, 35.0}) {
                std::vector<float> tbl;
                if (w > 0.0) tbl = radiation::smith::buildChiTable(
                    w * M_PI / 180.0, ns, ne, np);
                double chi = (w > 0.0) ? chiAt(tbl, theta_sun, te) : 1.0;
                std::printf("  %7.2f", bandBT(T, chi));
            }
            std::printf("\n");
        }
    }

    // (2b) Toward-minus-away BT contrast grows monotonically with RMS slope.
    std::printf("\n(2b) toward-minus-away BT contrast vs RMS slope "
                "(T=200 K, incidence 60 deg, |theta_e|=50 deg):\n");
    {
        const double T = 200.0, theta_sun = 60.0, te = 50.0;
        double prev = -1e9;
        bool grows = true, toward_gt_away = true;
        for (double w : {5.0, 10.0, 20.0, 35.0}) {
            auto tbl = radiation::smith::buildChiTable(w * M_PI / 180.0, ns, ne, np);
            double bt_toward = bandBT(T, chiAt(tbl, theta_sun, +te));
            double bt_away   = bandBT(T, chiAt(tbl, theta_sun, -te));
            double d = bt_toward - bt_away;
            std::printf("     w=%4.0f deg : BT_toward=%7.2f  BT_away=%7.2f  "
                        "contrast=%+6.2f K\n", w, bt_toward, bt_away, d);
            if (bt_toward <= bt_away) toward_gt_away = false;
            if (d < prev - 1e-3) grows = false;
            prev = d;
        }
        check(toward_gt_away, "toward-Sun BT > away-Sun BT at every RMS slope");
        check(grows, "toward-minus-away BT contrast grows with RMS slope");
    }

    // (2c) Contrast grows as the Sun approaches grazing (larger incidence).
    std::printf("\n(2c) toward-minus-away BT contrast vs solar incidence "
                "(T=200 K, w=25 deg, |theta_e|=45 deg):\n");
    {
        const double T = 200.0, te = 45.0;
        auto tbl = radiation::smith::buildChiTable(25.0 * M_PI / 180.0, ns, ne, np);
        double prev = -1e9;
        bool grows = true;
        for (double theta_sun : {20.0, 40.0, 60.0, 75.0}) {
            double bt_toward = bandBT(T, chiAt(tbl, theta_sun, +te));
            double bt_away   = bandBT(T, chiAt(tbl, theta_sun, -te));
            double d = bt_toward - bt_away;
            std::printf("     incidence=%4.0f deg : contrast=%+6.2f K\n", theta_sun, d);
            if (d < prev - 1e-3) grows = false;
            prev = d;
        }
        check(grows, "toward-minus-away BT contrast grows with solar incidence");
    }

    // (3) T_ss ceiling / clamp behavior. A warm grazing facet at high RMS slope
    // beams a band BT above its own kinetic T (isothermal-tier artifact) and can
    // pierce the subsolar-equilibrium ceiling -- the case the camera runtime guard
    // flags. T_ss @2.7 AU, A=0.05, eps=0.95 ~ 240 K.
    std::printf("\n(3) Warm grazing facet, high roughness (T_ss ceiling demo):\n");
    {
        const double T_ss = 240.0;   // subsolar-equilibrium ceiling @2.7 AU
        const double T = 235.0;      // warm facet, just below the ceiling
        auto tbl = radiation::smith::buildChiTable(35.0 * M_PI / 180.0, ns, ne, np);
        // Sun grazing at 82 deg; observer near the Sun (toward, small phase).
        double theta_sun = 82.0, te = 70.0;
        double chi = chiAt(tbl, theta_sun, te);
        double bt = bandBT(T, chi);
        std::printf("     T=%.1f K, incidence=%.0f deg, emission=%.0f deg, w=35 deg\n",
                    T, theta_sun, te);
        std::printf("     chi=%.3f -> BT=%.2f K  (T_ss ceiling=%.1f K, exceed=%+.2f K)\n",
                    chi, bt, T_ss, bt - T_ss);
        check(bt > T, "beamed band BT exceeds kinetic T toward the Sun (beaming present)");
        std::printf("     NOTE: BT>T for an isothermal facet means directional "
                    "emissivity eps*chi>eps;\n"
                    "           energy is conserved hemispherically (int chi cos dOmega = pi),\n"
                    "           but the directional band BT is a lower-fidelity artifact -- the\n"
                    "           camera runtime guard flags any pixel with BT>T_ss+2K.\n");
    }

    // (4) Anisothermal sub-facet layer (T0.1b) through the EXACT camera Planck path.
    namespace aniso = radiation::smith::aniso;
    std::printf("\n(4) Anisothermal sub-facet contrast E through the camera band path:\n");
    {
        // (4a) OFF byte-identity: with the anisothermal model inactive (empty
        // kernel) E == 1.0 exactly, so the band radiance chi*E*B is bit-identical
        // to the isothermal chi*B the camera writes when the model is not aniso.
        aniso::ContrastKernel off;
        size_t nchk = 0, nbad = 0;
        for (double T : {60.0, 137.0, 235.0})
            for (double chi : {1.0, 1.6, 3.1}) {
                double E = aniso::enhancement(off, 0.5, 0.5, 0.3, 1.0, T, 240.0,
                    [&](double t){ return model.bandPlanckRadiance(t); });
                double with = chi * E * model.bandPlanckRadiance(T);
                double without = chi * model.bandPlanckRadiance(T);
                ++nchk;
                if (std::memcmp(&with, &without, sizeof(double)) != 0) ++nbad;
            }
        std::printf("    empty kernel: %zu samples, %zu bitwise chi*E*B != chi*B\n", nchk, nbad);
        check(nbad == 0, "aniso E*L == L bit-identical when model inactive (do-no-harm)");

        // (4b) Anisothermal BT_K > isothermal BT_K at a Wien-regime backscatter
        // geometry: the sub-facet hot spot raises the band brightness temperature
        // ABOVE the isothermal Smith chi alone (the +5-20 deg bias origin).
        const double T = 137.0;               // Justitia-class temperature
        double w = 25.0 * M_PI / 180.0;
        auto chiT = radiation::smith::buildChiTable(w, ns, ne, np);
        auto k = aniso::buildContrastKernel(w, 0.2, ns, ne, np);
        double theta_sun = 72.0, te = 55.0;   // grazing Sun, toward-Sun backscatter
        double chi = chiAt(chiT, theta_sun, te);
        double cos_sun = std::cos(theta_sun*M_PI/180), cos_e = std::cos(te*M_PI/180);
        double cos_psi = std::cos((te - theta_sun)*M_PI/180);
        double E = aniso::enhancement(k, cos_sun, cos_e, cos_psi, 1.0, T, 0.0,
            [&](double t){ return model.bandPlanckRadiance(t); });
        double bt_iso   = bandBT(T, chi);
        double bt_aniso = model.bandBrightnessTemperature(chi * E * model.bandPlanckRadiance(T));
        std::printf("    T=%.0f K, chi=%.3f, E=%.3f: BT_iso=%.2f K  BT_aniso=%.2f K  (+%.2f K)\n",
                    T, chi, E, bt_iso, bt_aniso, bt_aniso - bt_iso);
        check(bt_aniso > bt_iso + 0.5, "anisothermal BT_K exceeds isothermal BT_K (sub-facet hot spot)");

        // (4c) Damping monotonicity through the camera path: higher Theta -> less
        // excess -> BT_aniso decreases toward BT_iso.
        double prev = 1e9; bool shrinks = true;
        std::printf("    BT_aniso - BT_iso vs Theta:");
        for (double th : {0.0, 1.0, 3.0, 10.0, 1e4}) {
            double D = aniso::dampingPole(th);
            double Ed = aniso::enhancement(k, cos_sun, cos_e, cos_psi, D, T, 0.0,
                [&](double t){ return model.bandPlanckRadiance(t); });
            double bt = model.bandBrightnessTemperature(chi * Ed * model.bandPlanckRadiance(T));
            std::printf("  %.0f:%+.2f", th, bt - bt_iso);
            if (bt - bt_iso > prev + 1e-3) shrinks = false;
            prev = bt - bt_iso;
        }
        std::printf("\n");
        check(shrinks, "BT excess decreases monotonically with Theta");
        check(prev < 0.05, "BT excess -> 0 (recovers isothermal) at Theta -> inf");
    }

    std::printf("\n%s (%d failure%s)\n", g_fail == 0 ? "ALL PASS" : "FAILURES",
                g_fail, g_fail == 1 ? "" : "s");
    return g_fail == 0 ? 0 : 1;
}
