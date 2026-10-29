// Standalone gate for the lunar albedo-map registration tripwire.
//
// The tripwire's job is to catch a mis-registered albedo map at load time. Its
// two failure modes are equally serious:
//   - a MISS lets a rolled map through silently (the original defect: five months
//     of lunar runs with the albedo on the wrong terrain, invisible to every
//     scalar summary), and
//   - a FALSE ALARM on a correct map, a regional mesh, or a non-lunar body would
//     train everyone to ignore it.
//
// So this test drives both directions. It builds synthetic meshes rather than
// reading data files, so it runs anywhere and pins the LOGIC, not the shipped bins.
//
// Build: cmake --build build --target test_albedo_registration_check
// Run:   build/bin/test_albedo_registration_check

#include "albedo_registration_check.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

constexpr double kPi = 3.14159265358979323846;

int g_failures = 0;

void check(bool cond, const std::string& what) {
    std::printf("  [%s] %s\n", cond ? "PASS" : "FAIL", what.c_str());
    if (!cond) ++g_failures;
}

// A roughly uniform global sphere of face centers (equirectangular lattice --
// denser near the poles than an icosphere, which is irrelevant here: the check
// only ever uses nearest-face lookups).
void makeGlobalMesh(std::vector<double>& lat, std::vector<double>& lon, double step_deg) {
    lat.clear();
    lon.clear();
    for (double la = -89.0; la <= 89.0; la += step_deg) {
        for (double lo = 0.0; lo < 360.0; lo += step_deg) {
            lat.push_back(la);
            lon.push_back(lo);
        }
    }
}

// A small regional patch, the shape of an A3GT site or a moon3d crater mesh.
void makeRegionalMesh(std::vector<double>& lat, std::vector<double>& lon,
                      double lat0, double lon0, double half_width_deg) {
    lat.clear();
    lon.clear();
    for (double dla = -half_width_deg; dla <= half_width_deg; dla += 0.05) {
        for (double dlo = -half_width_deg; dlo <= half_width_deg; dlo += 0.05) {
            lat.push_back(lat0 + dla);
            lon.push_back(lon0 + dlo);
        }
    }
}

// Synthetic lunar Bond albedo: dark maria on the nearside, bright highlands
// everywhere else. Crude, but it carries the one property the check tests --
// the physical ordering of mare and highland brightness.
double synthAlbedo(double lat, double lon) {
    struct Mare { double lat, lon, radius; };
    static const Mare maria[] = {
        {28.0, 17.5, 12.0}, {32.8, 344.4, 14.0}, {17.0, 58.4, 9.0},
        {20.0, 302.0, 18.0}, {8.5, 31.4, 12.0},
    };
    while (lon < 0.0) lon += 360.0;
    while (lon >= 360.0) lon -= 360.0;
    for (const Mare& m : maria) {
        double dlon = lon - m.lon;
        while (dlon > 180.0) dlon -= 360.0;
        while (dlon < -180.0) dlon += 360.0;
        const double dlat = lat - m.lat;
        // Approximate angular distance is adequate at these separations.
        const double d = std::sqrt(dlat * dlat +
                                   dlon * dlon * std::cos(lat * kPi / 180.0) *
                                       std::cos(lat * kPi / 180.0));
        if (d < m.radius) return 0.08;  // mare
    }
    return 0.16;  // highland
}

enum class Transform { kCorrect, kLonRoll180, kLatFlip };

std::vector<double> bakeAlbedo(const std::vector<double>& lat,
                               const std::vector<double>& lon, Transform t) {
    std::vector<double> a(lat.size());
    for (size_t i = 0; i < lat.size(); ++i) {
        switch (t) {
            case Transform::kCorrect:
                a[i] = synthAlbedo(lat[i], lon[i]);
                break;
            case Transform::kLonRoll180:  // the albedo-map longitude-roll
                a[i] = synthAlbedo(lat[i], lon[i] + 180.0);
                break;
            case Transform::kLatFlip:  // the albedo generator latitude flip
                a[i] = synthAlbedo(-lat[i], lon[i]);
                break;
        }
    }
    return a;
}

}  // namespace

int main() {
    std::printf("test_albedo_registration_check: lunar albedo map registration tripwire\n");

    std::vector<double> glat, glon;
    makeGlobalMesh(glat, glon, 1.0);
    std::printf("\nglobal mesh: %zu faces\n", glat.size());

    // 1. A correct map must pass quietly.
    {
        const auto a = bakeAlbedo(glat, glon, Transform::kCorrect);
        const auto r = ptm::checkAlbedoMapRegistration("Moon", glat, glon, a, "synthetic correct");
        check(r.applicable, "correct map: check applies to a global lunar mesh");
        check(!r.suspect, "correct map: NOT flagged");
        check(r.contrast > 0.02, "correct map: bright-dark contrast is positive");
        check(r.ns_contrast > 0.0, "correct map: north/south contrast is positive");
    }

    // 2. A 180-degree longitude roll must be caught.
    {
        const auto a = bakeAlbedo(glat, glon, Transform::kLonRoll180);
        const auto r = ptm::checkAlbedoMapRegistration("Moon", glat, glon, a, "synthetic lon+180");
        check(r.applicable && r.suspect, "longitude roll: FLAGGED");
    }

    // 3. A latitude flip must be caught. This is the one the bright-dark
    //    contrast alone barely sees, which is why the north/south test exists.
    {
        const auto a = bakeAlbedo(glat, glon, Transform::kLatFlip);
        const auto r = ptm::checkAlbedoMapRegistration("Moon", glat, glon, a, "synthetic -lat");
        check(r.applicable && r.suspect, "latitude flip: FLAGGED");
        check(r.ns_contrast < 0.0, "latitude flip: north/south contrast goes negative");
    }

    // 4. Non-lunar bodies must never be touched. Europa's FITS registers correctly
    //    through the same loader, so a blanket correction there is a regression.
    {
        const auto a = bakeAlbedo(glat, glon, Transform::kLonRoll180);
        for (const char* body : {"Europa", "Mars", "Bennu", ""}) {
            const auto r = ptm::checkAlbedoMapRegistration(body, glat, glon, a, "synthetic");
            check(!r.applicable && !r.suspect,
                  std::string("body '") + body + "': check does not apply, stays silent");
        }
    }

    // 4b. The body name is normalized inside the check, so a caller cannot
    //     silently disable the guard by passing the config's raw casing.
    {
        const auto a = bakeAlbedo(glat, glon, Transform::kLonRoll180);
        for (const char* body : {"moon", "MOON", "mOoN"}) {
            const auto r = ptm::checkAlbedoMapRegistration(body, glat, glon, a, "synthetic");
            check(r.applicable && r.suspect,
                  std::string("body '") + body + "': normalized, still checked");
        }
    }

    // 5. A regional mesh cannot resolve the reference sites, so the check must
    //    DECLINE rather than judge off whatever face is nearest. Without this,
    //    every A3GT polar site and every moon3d crater run would false-alarm.
    {
        struct Site { double lat, lon; const char* name; };
        const Site sites[] = {
            {-88.1, 6.3, "A3GT-like south polar site"},
            {-85.0, 30.0, "moon3d-like polar crater"},
            {0.0, 90.0, "equatorial nearside patch"},
        };
        for (const Site& s : sites) {
            std::vector<double> rlat, rlon;
            makeRegionalMesh(rlat, rlon, s.lat, s.lon, 1.0);
            const auto a = bakeAlbedo(rlat, rlon, Transform::kLonRoll180);
            const auto r = ptm::checkAlbedoMapRegistration("Moon", rlat, rlon, a, "regional");
            check(!r.applicable && !r.suspect,
                  std::string("regional mesh (") + s.name + "): declines to judge, no false alarm");
        }
    }

    // 6. Degenerate inputs must not crash or fire.
    {
        const std::vector<double> empty;
        const auto r = ptm::checkAlbedoMapRegistration("Moon", empty, empty, empty, "empty");
        check(!r.applicable && !r.suspect, "empty mesh: no verdict, no crash");

        // A uniform field has zero contrast, so the check flags it. That is correct
        // for a field that claims to be a map -- but a deliberate uniform-albedo run
        // is not a map at all, so BOTH call sites gate this on a map having actually
        // loaded. Without that gate, every use_albedo_map:false lunar run would warn.
        std::vector<double> a(glat.size(), 0.12);
        const auto r2 = ptm::checkAlbedoMapRegistration("Moon", glat, glon, a, "uniform");
        check(r2.applicable && r2.suspect,
              "uniform albedo: flagged (callers must gate on a map having loaded)");
    }

    std::printf("\n%s (%d failure%s)\n", g_failures == 0 ? "ALL PASS" : "FAILURES",
                g_failures, g_failures == 1 ? "" : "s");
    return g_failures == 0 ? 0 : 1;
}
