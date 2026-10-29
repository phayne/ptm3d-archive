#include "albedo_registration_check.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

namespace ptm {
namespace {

constexpr double kPi = 3.14159265358979323846;

struct ReferenceSite {
    double lat_deg;
    double lon_deg;  // east longitude, IAU
    const char* name;
};

// Lunar reference sites, IAU_MOON. Bond albedo from the same LRO-heritage
// mapping the shipped map is derived from: maria ~0.07-0.09, highlands ~0.15-0.17.
//
// Aristarchus is deliberately NOT used as a bright anchor: it is sub-pixel at the
// map's 1 deg/pixel sampling and reads dark, which would produce a false alarm.
const ReferenceSite kMoonDark[] = {
    {28.0, 17.5, "Mare Serenitatis"},
    {32.8, 344.4, "Mare Imbrium"},
    {17.0, 58.4, "Mare Crisium"},
    {20.0, 302.0, "Oceanus Procellarum"},
    {8.5, 31.4, "Mare Tranquillitatis"},
};
const ReferenceSite kMoonBright[] = {
    {0.0, 180.0, "farside highlands (180E)"},
    {-4.4, 202.6, "Korolev region"},
    {0.0, 160.0, "farside highlands (160E)"},
    {20.0, 150.0, "farside highlands (150E)"},
    {-43.3, 348.8, "Tycho ejecta"},
};

// Latitude axis: the bright/dark contrast alone detects a latitude flip only
// weakly (measured +0.0141 against a +0.02 gate — 0.006 of headroom). Every
// kMoonDark site is in the NORTHERN hemisphere, so reflecting those same sites
// across the equator at unchanged longitude gives southern highlands: dark north,
// bright south. A latitude flip swaps them, turning a measured +0.0546 into
// -0.0546 — a clean sign change with an order of magnitude more margin.
//
// The northern half of that pair is exactly kMoonDark, so only the mirrors need
// their own queries; the "north" mean IS the dark mean.

// The sampled albedos ARE the evidence in this message, so format them
// explicitly rather than inheriting whatever precision the ambient stream
// happens to be carrying (which rounded them to "0.2" and "-0.0" in testing).
std::string fmt(double v, int digits = 4) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(digits) << v;
    return os.str();
}

void unitVector(double lat_deg, double lon_deg, double v[3]) {
    const double la = lat_deg * kPi / 180.0;
    const double lo = lon_deg * kPi / 180.0;
    v[0] = std::cos(la) * std::cos(lo);
    v[1] = std::cos(la) * std::sin(lo);
    v[2] = std::sin(la);
}

/// One query point and the albedo the mesh actually carries there.
struct Sample {
    double albedo = 0.0;
    bool resolved = false;  ///< a face center lay within the separation tolerance
};

/// Nearest-face lookup for every query point in a SINGLE pass over the faces
/// (which may number millions), rather than one full sweep per query.
std::vector<Sample> sampleQueries(const std::vector<double>& q_lat,
                                  const std::vector<double>& q_lon,
                                  const std::vector<double>& lat,
                                  const std::vector<double>& lon,
                                  const std::vector<double>& albedo) {
    const size_t nq = q_lat.size();
    const double cos_tol = std::cos(kMaxAnchorSeparationDeg * kPi / 180.0);

    std::vector<double> qv(nq * 3);
    std::vector<double> best_dot(nq, -2.0);
    std::vector<size_t> best_face(nq, 0);
    for (size_t s = 0; s < nq; ++s) {
        unitVector(q_lat[s], q_lon[s], &qv[s * 3]);
    }

    for (size_t f = 0; f < albedo.size(); ++f) {
        double p[3];
        unitVector(lat[f], lon[f], p);
        for (size_t s = 0; s < nq; ++s) {
            const double dot =
                p[0] * qv[s * 3] + p[1] * qv[s * 3 + 1] + p[2] * qv[s * 3 + 2];
            if (dot > best_dot[s]) {
                best_dot[s] = dot;
                best_face[s] = f;
            }
        }
    }

    std::vector<Sample> out(nq);
    for (size_t s = 0; s < nq; ++s) {
        out[s].resolved = (best_dot[s] >= cos_tol);
        if (out[s].resolved) out[s].albedo = albedo[best_face[s]];
    }
    return out;
}

/// Mean over the resolved samples in [begin, end); `n` receives how many counted.
double meanResolved(const std::vector<Sample>& s, size_t begin, size_t end, int& n) {
    double sum = 0.0;
    n = 0;
    for (size_t i = begin; i < end; ++i) {
        if (s[i].resolved) {
            sum += s[i].albedo;
            ++n;
        }
    }
    return (n > 0) ? sum / n : 0.0;
}

}  // namespace

AlbedoRegistrationReport checkAlbedoMapRegistration(
    const std::string& body_name,
    const std::vector<double>& face_lat_deg,
    const std::vector<double>& face_lon_deg,
    const std::vector<double>& face_albedos,
    const std::string& source_label) {
    AlbedoRegistrationReport report;

    // Normalize here rather than at each call site: the config value arrives in
    // whatever case the YAML used, and the surrounding code already repeats this
    // proper-casing half a dozen times. Doing it inside the check means a caller
    // cannot silently disable the guard by passing "moon".
    std::string body = body_name;
    if (!body.empty()) {
        std::transform(body.begin(), body.end(), body.begin(), ::tolower);
        body[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(body[0])));
    }

    // Only bodies with a published dark/bright signature can be checked. Every
    // other body (Europa, Mars, generic asteroids) returns silently.
    if (body != "Moon") {
        return report;
    }

    const size_t n = face_albedos.size();
    if (n == 0 || face_lat_deg.size() < n || face_lon_deg.size() < n) {
        return report;  // no per-face geometry: nothing defensible to check
    }

    const size_t n_dark = sizeof(kMoonDark) / sizeof(kMoonDark[0]);
    const size_t n_bright = sizeof(kMoonBright) / sizeof(kMoonBright[0]);

    // Query layout: [dark | bright | southern mirrors of the dark sites].
    std::vector<double> q_lat, q_lon;
    for (size_t i = 0; i < n_dark; ++i) {
        q_lat.push_back(kMoonDark[i].lat_deg);
        q_lon.push_back(kMoonDark[i].lon_deg);
    }
    for (size_t i = 0; i < n_bright; ++i) {
        q_lat.push_back(kMoonBright[i].lat_deg);
        q_lon.push_back(kMoonBright[i].lon_deg);
    }
    for (size_t i = 0; i < n_dark; ++i) {
        q_lat.push_back(-kMoonDark[i].lat_deg);
        q_lon.push_back(kMoonDark[i].lon_deg);
    }

    const std::vector<Sample> s =
        sampleQueries(q_lat, q_lon, face_lat_deg, face_lon_deg, face_albedos);

    int dark_n = 0, bright_n = 0, south_n = 0;
    const size_t o_bright = n_dark;
    const size_t o_south = n_dark + n_bright;
    const double dark_mean = meanResolved(s, 0, o_bright, dark_n);
    const double bright_mean = meanResolved(s, o_bright, o_south, bright_n);
    const double south_mean = meanResolved(s, o_south, s.size(), south_n);
    // The northern half of the latitude pair is exactly the dark set.
    const double north_mean = dark_mean;
    const int north_n = dark_n;

    // A regional mesh does not contain the reference sites. Decline to judge
    // rather than reporting a false positive off whatever face happens to be
    // nearest to a point the mesh does not cover.
    if (dark_n < kMinResolvedAnchors || bright_n < kMinResolvedAnchors) {
        return report;
    }

    report.applicable = true;
    report.dark_mean = dark_mean;
    report.bright_mean = bright_mean;
    report.contrast = bright_mean - dark_mean;
    report.dark_n = dark_n;
    report.bright_n = bright_n;

    const bool contrast_bad = (report.contrast < kMinAlbedoContrast);

    // The latitude test needs both hemispheres resolved to mean anything.
    const bool ns_usable = (north_n >= kMinResolvedAnchors && south_n >= kMinResolvedAnchors);
    report.ns_applicable = ns_usable;
    report.north_mare_mean = north_mean;
    report.south_mirror_mean = south_mean;
    report.ns_contrast = south_mean - north_mean;
    const bool lat_bad = ns_usable && (report.ns_contrast < 0.0);

    report.latitude_suspect = lat_bad;
    report.suspect = contrast_bad || lat_bad;

    if (report.suspect) {
        // Deliberately NOT attributing an axis here. Measured on real fixtures, a
        // pure longitude roll also drives the north/south statistic negative
        // (-0.0185), and a pure latitude flip lands just under the contrast gate
        // (+0.0187) — so neither statistic alone identifies the axis, and naming
        // one would be a guess dressed as a diagnosis. Attribution belongs in
        // verify_moon_albedo_registration.py, which has the source FITS and can
        // score all four candidate transforms against it.
        std::cerr
            << "\n*** WARNING: lunar albedo map looks MISREGISTERED ***\n"
            << "    source: " << source_label << "\n"
            << "    Maria sample " << fmt(dark_mean) << ", highlands sample " << fmt(bright_mean)
            << " -> contrast " << fmt(report.contrast) << " (must be positive, normally ~+0.08).\n";
        if (ns_usable) {
            std::cerr << "    Northern maria " << fmt(north_mean) << " vs their southern mirrors "
                      << fmt(south_mean) << " -> " << fmt(report.ns_contrast)
                      << " (must be positive, normally ~+0.05).\n";
        }
        std::cerr
            << "    The maria are the DARK terrain (Bond albedo ~0.07-0.09) and the highlands\n"
            << "    the BRIGHT terrain (~0.15-0.17). Reading that ordering inverted means the\n"
            << "    map is rolled relative to IAU_MOON. Two known defects do this: a\n"
            << "    180-degree longitude roll of the source FITS, and a latitude flip\n"
            << "    baked in by the map generator. These two statistics cannot tell them apart\n"
            << "    on their own -- run the verify script below, which scores the map against\n"
            << "    the source FITS under every candidate transform and names the axis.\n"
            << "    Consequence: the albedo VALUE distribution is unchanged, so the mean albedo\n"
            << "    and the effective absorbed albedo look normal, but the albedo is applied to\n"
            << "    the WRONG TERRAIN. Surface temperatures are misplaced by several K locally\n"
            << "    (measured: +8 K at the maria, -7 K on the farside) with no global signature.\n"
            << "    This run is CONTINUING with the map as given -- it is not corrected here,\n"
            << "    because the fix belongs in the data.\n"
            << "    To repair: python3 scripts/fetch_data.py --dataset albedo-maps (the archive\n"
            << "    ships corrected maps), then confirm with\n"
            << "    scripts/verify_moon_albedo_registration.py.\n\n";
    } else {
        std::cout << "[albedo] Lunar map registration check: maria " << fmt(dark_mean)
                  << " < highlands " << fmt(bright_mean) << " (contrast " << fmt(report.contrast);
        if (ns_usable) {
            std::cout << ", N/S " << fmt(report.ns_contrast);
        }
        std::cout << ", " << dark_n << "/" << bright_n << " reference sites resolved) -- OK"
                  << std::endl;
    }

    return report;
}

}  // namespace ptm
