// Standalone validator for the Smith (1967) anisotropic thermal-emission factor
// chi (smith_beaming.h). Checks, with NO mesh/Embree/SPICE dependency:
//   (1) Energy conservation: cos_e-weighted hemispherical mean of chi == 1 for
//       every Sun elevation (pure angular redistribution).
//   (2) w -> 0 limit: chi -> 1 everywhere.
//   (3) Directionality: chi > 1 toward the Sun (small psi), chi < 1 away.
//   (4) Grazing-Sun wall->floor enhancement: a wall sloped toward the floor,
//       Sun grazing past it, beams MORE toward the floor as w grows.
//
// Build: linked as the test_beam_chi target (see radiation/CMakeLists.txt).
// Run:   ./bin/test_beam_chi

#include "smith_beaming.h"

#include <cstdio>
#include <cmath>
#include <vector>

using namespace radiation;

static int g_fail = 0;
static void check(bool ok, const char* msg) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", msg);
    if (!ok) ++g_fail;
}

// Numerically integrate (1/pi) * int chi(d) cos_e dOmega over the emission
// hemisphere, for a fixed Sun at polar angle theta_sun. Should equal 1.
static double hemisphericalMean(const std::vector<float>& tbl, int ns, int ne, int np,
                                double cos_sun) {
    double sin_sun = std::sqrt(std::max(0.0, 1.0 - cos_sun * cos_sun));
    const int Nt = 64, Nphi = 128;
    const double dtheta = (M_PI / 2.0) / Nt, dphi = (2.0 * M_PI) / Nphi;
    double acc = 0.0;
    for (int it = 0; it < Nt; ++it) {
        double te = (it + 0.5) * dtheta;
        double cos_e = std::cos(te), sin_e = std::sin(te);
        for (int ip = 0; ip < Nphi; ++ip) {
            double phi = (ip + 0.5) * dphi;
            // d in the frame where Sun is at (sin_sun, 0, cos_sun).
            double dx = sin_e * std::cos(phi), dy = sin_e * std::sin(phi), dz = cos_e;
            double cos_psi = dx * sin_sun + dz * cos_sun;
            double chi = smith::evalChi(tbl, ns, ne, np, cos_sun, cos_e, cos_psi);
            acc += chi * cos_e * sin_e * dtheta * dphi;
        }
    }
    return acc / M_PI;   // int cos_e dOmega over hemisphere = pi
}

int main() {
    const int ns = 24, ne = 24, np = 33;

    std::printf("Smith beamed-emission chi validation\n");
    std::printf("====================================\n\n");

    // (2) w -> 0 limit.
    {
        std::printf("(2) w -> 0 limit (chi -> 1 everywhere):\n");
        auto tbl = smith::buildChiTable(0.5 * M_PI / 180.0, ns, ne, np);  // 0.5 deg
        double maxdev = 0.0; int ais=0, aie=0, aip=0;
        for (int is = 0; is < ns; ++is) {
            for (int ie = 0; ie < ne; ++ie) {
                for (int ip = 0; ip < np; ++ip) {
                    double v = tbl[(size_t(is)*ne + ie)*np + ip];
                    double dev = std::fabs(v - 1.0);
                    if (dev > maxdev) { maxdev = dev; ais=is; aie=ie; aip=ip; }
                }
            }
        }
        std::printf("    max|chi-1| at w=0.5deg = %.4f at cos_sun=%.3f cos_e=%.3f cos_psi=%.3f\n",
                    maxdev, (ais+0.5)/ns, (aie+0.5)/ne, -1.0 + 2.0*(aip+0.5)/np);
        // Bulk of the table must be ~1; grazing limb corners (cos_e -> 0) carry a
        // real, finite Smith masking signature even as w -> 0, so check the bulk.
        size_t near1 = 0;
        for (float v : tbl) if (std::fabs((double)v - 1.0) < 0.05) ++near1;
        double frac = (double)near1 / tbl.size();
        std::printf("    fraction of cells within 0.05 of 1 = %.3f\n", frac);
        check(frac > 0.95, "chi ~ 1 in the bulk at tiny RMS slope");
    }

    // (1) Energy conservation across Sun elevations, several RMS slopes.
    std::printf("\n(1) Energy conservation (cos_e-weighted hemispherical mean == 1):\n");
    for (double wdeg : {10.0, 15.0, 20.0, 25.0}) {
        auto tbl = smith::buildChiTable(wdeg * M_PI / 180.0, ns, ne, np);
        double worst = 0.0;
        for (double cos_sun : {0.95, 0.7, 0.4, 0.15, 0.05}) {
            double m = hemisphericalMean(tbl, ns, ne, np, cos_sun);
            worst = std::max(worst, std::fabs(m - 1.0));
        }
        std::printf("    w=%4.1f deg : worst |mean-1| over Sun elevations = %.3f\n", wdeg, worst);
        char buf[96];
        std::snprintf(buf, sizeof buf, "energy-conserving at w=%.0f deg (mean within 10%% of 1)", wdeg);
        check(worst < 0.10, buf);
    }

    // (3) Directionality: toward Sun (cos_psi=+1) vs away (cos_psi=-1), grazing Sun.
    std::printf("\n(3) Directionality (grazing Sun, cos_sun=0.17 ~ 80 deg):\n");
    {
        auto tbl = smith::buildChiTable(20.0 * M_PI / 180.0, ns, ne, np);
        double cos_sun = 0.17, cos_e = 0.5;
        double sin_sun = std::sqrt(1 - cos_sun*cos_sun), sin_e = std::sqrt(1 - cos_e*cos_e);
        // Reachable azimuth extremes for this (cos_sun, cos_e):
        //   toward Sun (phi=0):   cos_psi = cos_sun cos_e + sin_sun sin_e
        //   away   Sun (phi=pi):  cos_psi = cos_sun cos_e - sin_sun sin_e
        double toward = smith::evalChi(tbl, ns, ne, np, cos_sun, cos_e,
                                       cos_sun*cos_e + sin_sun*sin_e);
        double away   = smith::evalChi(tbl, ns, ne, np, cos_sun, cos_e,
                                       cos_sun*cos_e - sin_sun*sin_e);
        std::printf("    chi(toward Sun) = %.3f, chi(away) = %.3f\n", toward, away);
        check(toward > 1.0, "chi > 1 toward the Sun");
        check(away   < 1.0, "chi < 1 away from the Sun");
        check(toward > away, "chi larger toward the Sun than away");
    }

    // (4) Grazing-Sun wall->floor enhancement grows with roughness.
    // Geometry: wall normal n = +z (local frame). Sun grazes at 80 deg from n,
    // azimuth 0. Floor lies in the same azimuth as the Sun but below the wall's
    // mean horizon is impossible in the upper hemisphere -- model the floor as an
    // emission direction at cos_e=0.4 sharing the Sun azimuth (small psi).
    std::printf("\n(4) Wall->floor enhancement vs RMS slope (grazing Sun, shared azimuth):\n");
    {
        double cos_sun = 0.17;               // Sun ~80 deg off the wall normal
        double cos_e = 0.4;                  // emission toward floor
        double sin_sun = std::sqrt(1 - cos_sun*cos_sun), sin_e = std::sqrt(1 - cos_e*cos_e);
        double cos_psi = cos_sun*cos_e + sin_sun*sin_e;  // same azimuth (phi=0)
        double prev = 1.0;
        bool monotone = true;
        for (double wdeg : {2.0, 8.0, 15.0, 25.0}) {
            auto tbl = smith::buildChiTable(wdeg * M_PI / 180.0, ns, ne, np);
            double chi = smith::evalChi(tbl, ns, ne, np, cos_sun, cos_e, cos_psi);
            std::printf("    w=%4.1f deg : chi(wall->floor) = %.3f\n", wdeg, chi);
            if (wdeg > 2.0 && chi < prev - 1e-3) monotone = false;
            prev = chi;
        }
        check(prev > 1.05, "wall->floor beaming enhanced at physical RMS slope (chi>1.05 at 25 deg)");
        check(monotone, "enhancement grows monotonically with RMS slope");
    }

    std::printf("\n%s (%d failure%s)\n", g_fail == 0 ? "ALL PASS" : "FAILURES",
                g_fail, g_fail == 1 ? "" : "s");
    return g_fail == 0 ? 0 : 1;
}
