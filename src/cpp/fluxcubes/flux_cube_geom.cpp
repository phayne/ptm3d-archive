// flux_cube_geom.cpp -- ptm_flux_cube_geom
// ---------------------------------------------------------------------------
// One-time geometry build for the flux-cube product: place a regular grid of
// virtual measurement cubes on a site's terrain and, for each of the six
// oriented faces of each cube, ray-trace the cosine-weighted view-factor row
// over the CORE mesh plus the sky and occluder-ring fractions.
//
// CPU only. No GPU, no OpenCL context, no GPU lock -- it is safe to run
// concurrently with production GPU windows. The one scheduling rule is that it
// must not overlap a view-factor ASSEMBLE step (those peak above 300 GB RSS);
// this tool's own footprint is a few GB.
//
// WHY A SEPARATE BINARY
// --------------------
// The cache must be buildable before, or concurrently with, the GPU windows
// that consume it; it must be independently testable; and the engine must never
// build it inline, because a 7,542-row hemisphere sweep starting unannounced
// inside a production window is the kind of surprise that ends campaigns. The
// engine LOADS this file and refuses if the key does not match -- it never
// rebuilds (the never-rebuild rule).
//
// ESTIMATOR
// ---------
// Rays are drawn cosine-weighted about each patch normal, so every ray carries
// equal weight and
//     F[c][p][j] = (rays hitting core face j) / N
//     f_sky      = (rays escaping)            / N
//     f_ring     = (rays hitting the ring)    / N
// with F, f_sky and f_ring summing to exactly 1. For Lambertian emitters of
// radiosity B_j the irradiance on the patch is exactly E = sum_j F[j] B_j.
//
// Ring hits are counted in their OWN channel and are never folded into sky: the
// ring carries no thermal solution, so it has no radiosity to contribute, and
// quietly calling it "sky" would delete a real IR source from the budget.
// ---------------------------------------------------------------------------

#include "flux_cube_format.h"
#include "../mesh/mesh_utils.h"

#include <embree4/rtcore.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <cstddef>
#include <cstdlib>
#include <stdexcept>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace ptm::fluxcubes;

namespace {

constexpr double kPi = 3.14159265358979323846;

struct Vec3 {
    double x = 0, y = 0, z = 0;
    Vec3() = default;
    Vec3(double a, double b, double c) : x(a), y(b), z(c) {}
    Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vec3 operator*(double s) const { return {x * s, y * s, z * s}; }
};
inline double dot(const Vec3& a, const Vec3& b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return {a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x};
}
inline double norm(const Vec3& a) { return std::sqrt(dot(a, a)); }
inline Vec3 unit(const Vec3& a) { const double n = norm(a); return n > 0 ? a * (1.0 / n) : a; }

// Unit direction from geodetic-style lat/lon. ONE spelling, called by both the
// site-frame construction and the named-location resolver. Writing the same
// three trig lines in two places is not equivalent: the optimizer is free to
// contract them differently at each site, and the two results then differ by an
// ulp -- which showed up as a ~2e-11 m residue in a named location placed at the
// site's own coordinates, breaking the grid-vs-named bit-identity check. Marked
// noinline so both callers demonstrably execute the same instructions.
__attribute__((noinline))
Vec3 dirFromLatLon(double lat_deg, double lon_deg) {
    const double la = lat_deg * kPi / 180.0;
    const double lo = lon_deg * kPi / 180.0;
    return unit(Vec3(std::cos(la) * std::cos(lo),
                     std::cos(la) * std::sin(lo),
                     std::sin(la)));
}

// Equivalent edge of an equilateral triangle of the given area:
// A = (sqrt(3)/4) e^2  =>  e = sqrt(4A/sqrt(3)). This is the definition the
// site-relative height factor multiplies, and it is stated in the file header
// so the number in a product can be re-derived.
inline double equivalentEdge(double area) {
    return std::sqrt(4.0 * area / std::sqrt(3.0));
}

uint64_t fnv1a64File(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return 0;
    uint64_t h = 1469598103934665603ULL;
    char buf[65536];
    while (f.read(buf, sizeof(buf)) || f.gcount() > 0) {
        const std::streamsize n = f.gcount();
        for (std::streamsize i = 0; i < n; ++i) {
            h ^= static_cast<uint8_t>(buf[i]);
            h *= 1099511628211ULL;
        }
    }
    return h;
}

// --- Sampling ------------------------------------------------------------
// Radical inverse base 2 (van der Corput), the stratified base sequence.
inline double radicalInverse2(uint32_t bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return static_cast<double>(bits) * 2.3283064365386963e-10;
}

// Shirley-Chiu concentric mapping of the unit square to the unit disk. Lower
// distortion than the polar map, which is why the engine's own primary-disk
// sampler uses it -- and why its worst-case error sits on the 45/135 degree
// diagonal, which is exactly what the per-patch Cranley-Patterson rotation
// below decorrelates. Without that rotation every one of the thousands of
// patches on a regular lattice would reuse the identical point grain and the
// flux map would carry a coherent stripe rather than white noise.
inline void concentricDisk(double u, double v, double& dx, double& dy) {
    const double a = 2.0 * u - 1.0, b = 2.0 * v - 1.0;
    if (a == 0.0 && b == 0.0) { dx = 0.0; dy = 0.0; return; }
    double r, phi;
    if (std::fabs(a) > std::fabs(b)) { r = a; phi = (kPi / 4.0) * (b / a); }
    else                             { r = b; phi = (kPi / 2.0) - (kPi / 4.0) * (a / b); }
    dx = r * std::cos(phi);
    dy = r * std::sin(phi);
}

struct Frame { Vec3 t, bt, n; };
Frame buildFrame(const Vec3& n) {
    Frame f;
    f.n = n;
    const Vec3 a = (std::fabs(n.z) < 0.9) ? Vec3(0, 0, 1) : Vec3(1, 0, 0);
    f.t  = unit(cross(a, n));
    f.bt = cross(n, f.t);
    return f;
}

struct Args {
    std::string core, ring, out;
    double center_lat = 0.0, center_lon = 0.0;
    bool   have_center = false;
    double spacing_m = 25.0, radius_m = 500.0;
    double height_factor = 0.1, edge_factor = 0.1;
    double height_m = -1.0, edge_m = -1.0;   // <0 => use the factor
    uint32_t n_rays = 16384;
    uint64_t seed = 0;
    double truncate_rel = 0.0;
    int threads = 8;
    std::string locations;        // CSV: name,lat_deg,lon_deg[,height_m[,edge_m]]
};

// A named location, resolved to the site frame by the same drop-ray path the
// lattice uses -- deliberately the same path, so a named cube placed at a
// lattice cube's own coordinates lands on bit-identical geometry.
struct NamedLoc {
    std::string name;
    double lat = 0.0, lon = 0.0;
    double height_m = -1.0;   // <0 => use the cache-level height
    double edge_m = -1.0;     // <0 => use the cache-level edge
};

void usage() {
    std::printf(
"ptm_flux_cube_geom -- build the flux-cube geometry cache (.fcg)\n\n"
"  --core <ply>            CORE mesh (required)\n"
"  --ring <ply>            occluder ring mesh (optional but recommended)\n"
"  --out  <file.fcg>       output cache (required)\n"
"  --center-lat <deg>      site center latitude   (default: core mesh centroid)\n"
"  --center-lon <deg>      site center longitude\n"
"  --spacing <m>           lattice spacing        (default 25)\n"
"  --radius <m>            grid radius (default 500; <0 = NO lattice,\n"
"                          named locations only)\n"
"  --height-factor <f>     cube center height = f * local median facet edge (default 0.1)\n"
"  --height-m <m>          ABSOLUTE height override; wins over --height-factor\n"
"  --edge-factor <f>       cube edge = f * local median facet edge (default 0.1)\n"
"  --edge-m <m>            ABSOLUTE edge override; wins over --edge-factor\n"
"  --n-rays <N>            rays per patch         (default 16384)\n"
"  --seed <n>              Cranley-Patterson seed (default 0)\n"
"  --truncate-rel <f>      drop partners below f of the row max (default 0 = keep all)\n"
"  --threads <n>           OpenMP threads         (default 8)\n"
"  --locations <csv>       named locations, one per line:\n"
"                            name,lat_deg,lon_deg[,height_m[,edge_m]]\n"
"                          Appended AFTER the lattice (use --radius 0 for\n"
"                          named locations only). Blank lines and # comments\n"
"                          are skipped; a malformed line is a hard error.\n");
}

bool parseArgs(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto next = [&](double& d) { if (i + 1 < argc) { d = std::atof(argv[++i]); return true; } return false; };
        if      (s == "--core" && i + 1 < argc) a.core = argv[++i];
        else if (s == "--ring" && i + 1 < argc) a.ring = argv[++i];
        else if (s == "--out"  && i + 1 < argc) a.out  = argv[++i];
        else if (s == "--center-lat") { double d; if (!next(d)) return false; a.center_lat = d; a.have_center = true; }
        else if (s == "--center-lon") { double d; if (!next(d)) return false; a.center_lon = d; a.have_center = true; }
        else if (s == "--spacing")       { if (!next(a.spacing_m)) return false; }
        else if (s == "--radius")        { if (!next(a.radius_m)) return false; }
        else if (s == "--height-factor") { if (!next(a.height_factor)) return false; }
        else if (s == "--height-m")      { if (!next(a.height_m)) return false; }
        else if (s == "--edge-factor")   { if (!next(a.edge_factor)) return false; }
        else if (s == "--edge-m")        { if (!next(a.edge_m)) return false; }
        else if (s == "--truncate-rel")  { if (!next(a.truncate_rel)) return false; }
        else if (s == "--n-rays" && i + 1 < argc) a.n_rays  = static_cast<uint32_t>(std::atoll(argv[++i]));
        else if (s == "--seed"   && i + 1 < argc) a.seed    = static_cast<uint64_t>(std::atoll(argv[++i]));
        else if (s == "--threads"&& i + 1 < argc) a.threads = std::atoi(argv[++i]);
        else if (s == "--locations" && i + 1 < argc) a.locations = argv[++i];
        else if (s == "-h" || s == "--help") { usage(); std::exit(0); }
        else { std::cerr << "unknown argument: " << s << "\n"; return false; }
    }
    return !a.core.empty() && !a.out.empty();
}

// Parse the named-location CSV. A malformed line is a HARD ERROR, never a skip:
// silently dropping a requested location would change the cube set between runs
// of the same configuration, which is the same failure mode the lattice's
// miss-is-fatal rule exists to prevent.
bool parseLocations(const std::string& path, std::vector<NamedLoc>& out, std::string& err) {
    std::ifstream f(path);
    if (!f) { err = "cannot open --locations file '" + path + "'"; return false; }
    std::string line;
    int lineno = 0;
    while (std::getline(f, line)) {
        ++lineno;
        // strip comments and whitespace
        const size_t h = line.find('#');
        if (h != std::string::npos) line = line.substr(0, h);
        size_t b = line.find_first_not_of(" \t\r\n");
        if (b == std::string::npos) continue;
        size_t e2 = line.find_last_not_of(" \t\r\n");
        line = line.substr(b, e2 - b + 1);
        if (line.empty()) continue;
        std::vector<std::string> tok;
        std::string cur;
        std::istringstream ss(line);
        while (std::getline(ss, cur, ',')) {
            size_t p0 = cur.find_first_not_of(" \t");
            size_t p1 = cur.find_last_not_of(" \t");
            tok.push_back(p0 == std::string::npos ? std::string() : cur.substr(p0, p1 - p0 + 1));
        }
        // Header row, at ANY column count. Keying this on tok.size()==3 meant a
        // header that also declared the optional height/edge columns was parsed
        // as data and the run died on "could not parse a numeric field".
        if (!tok.empty() && tok[0] == "name") continue;
        if (tok.size() < 3 || tok.size() > 5) {
            err = path + ":" + std::to_string(lineno) +
                  ": expected name,lat_deg,lon_deg[,height_m[,edge_m]] -- got " +
                  std::to_string(tok.size()) + " field(s)";
            return false;
        }
        NamedLoc L;
        L.name = tok[0];
        if (L.name.empty()) { err = path + ":" + std::to_string(lineno) + ": empty name"; return false; }
        if (L.name.size() >= static_cast<size_t>(kNameLen)) {
            err = path + ":" + std::to_string(lineno) + ": name '" + L.name + "' exceeds " +
                  std::to_string(kNameLen - 1) + " characters";
            return false;
        }
        try {
            L.lat = std::stod(tok[1]);
            L.lon = std::stod(tok[2]);
            if (tok.size() >= 4 && !tok[3].empty()) L.height_m = std::stod(tok[3]);
            if (tok.size() >= 5 && !tok[4].empty()) L.edge_m = std::stod(tok[4]);
        } catch (const std::exception&) {
            err = path + ":" + std::to_string(lineno) + ": could not parse a numeric field";
            return false;
        }
        if (L.lat < -90.0 || L.lat > 90.0) {
            err = path + ":" + std::to_string(lineno) + ": latitude " + tok[1] + " out of range";
            return false;
        }
        for (const NamedLoc& q : out)
            if (q.name == L.name) {
                err = path + ":" + std::to_string(lineno) + ": duplicate location name '" +
                      L.name + "' -- outputs are keyed by name, so names must be unique";
                return false;
            }
        out.push_back(L);
    }
    return true;
}

void embreeError(void*, RTCError code, const char* str) {
    std::cerr << "[flux_cube_geom] Embree error " << code << ": " << (str ? str : "") << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    Args args;
    if (!parseArgs(argc, argv, args)) { usage(); return 2; }

#ifdef _OPENMP
    if (args.threads > 0) omp_set_num_threads(args.threads);
#endif

    std::printf("=== ptm_flux_cube_geom ===\n");

    std::vector<NamedLoc> named;
    if (!args.locations.empty()) {
        std::string lerr;
        if (!parseLocations(args.locations, named, lerr)) {
            std::cerr << "FATAL: " << lerr << "\n";
            return 2;
        }
        std::printf("locations : %s -> %zu named entr%s\n", args.locations.c_str(),
                    named.size(), named.size() == 1 ? "y" : "ies");
    }

    // ---------------------------------------------------------------- meshes
    ptm::MeshUtils core_utils;
    // importTriangulatedMesh is the PLY entry point the engine itself uses for
    // mesh.import_file (ptm_thermal_sim.cpp:14683). importMeshFromFile is the
    // PTMTOPO binary loader and will reject a PLY.
    std::shared_ptr<ptm::Mesh> core;
    if (core_utils.importTriangulatedMesh(args.core, 1.0)) core = core_utils.getMesh();
    if (!core || core->faces.empty()) {
        std::cerr << "FATAL: could not load core mesh '" << args.core << "'\n";
        return 3;
    }
    const size_t n_core_v = core->vertices.size();
    const size_t n_core_f = core->faces.size();
    std::printf("core mesh : %s\n          %zu vertices, %zu faces\n",
                args.core.c_str(), n_core_v, n_core_f);

    std::vector<std::array<double,3>> ring_v;
    std::vector<std::array<int,3>>    ring_f;
    uint64_t ring_hash = 0;
    if (!args.ring.empty()) {
        ptm::MeshUtils ring_utils;
        std::shared_ptr<ptm::Mesh> ring;
        if (ring_utils.importTriangulatedMesh(args.ring, 1.0)) ring = ring_utils.getMesh();
        if (!ring || ring->faces.empty()) {
            std::cerr << "FATAL: --ring given but '" << args.ring << "' did not load. "
                         "Refusing to build a cache with a silently missing occluder: the "
                         "ring changes every sky fraction in the file.\n";
            return 3;
        }
        ring_v = ring->vertices;
        ring_f = ring->faces;
        ring_hash = fnv1a64File(args.ring);
        std::printf("ring mesh : %s\n          %zu vertices, %zu faces (hash %016llx)\n",
                    args.ring.c_str(), ring_v.size(), ring_f.size(),
                    static_cast<unsigned long long>(ring_hash));
    } else {
        std::printf("ring mesh : NONE (f_ring will be 0 everywhere by construction)\n");
    }

    // Face geometry: centroid, normal, area, and the geometry hash the engine's
    // own view-factor gate uses, so the cache key is the same identity test.
    std::vector<Vec3> fc(n_core_f), fn(n_core_f);
    std::vector<double> farea(n_core_f);
    double geomhash = 0.0;
    for (size_t i = 0; i < n_core_f; ++i) {
        const auto& f = core->faces[i];
        const Vec3 a(core->vertices[f[0]][0], core->vertices[f[0]][1], core->vertices[f[0]][2]);
        const Vec3 b(core->vertices[f[1]][0], core->vertices[f[1]][1], core->vertices[f[1]][2]);
        const Vec3 c(core->vertices[f[2]][0], core->vertices[f[2]][1], core->vertices[f[2]][2]);
        fc[i] = (a + b + c) * (1.0 / 3.0);
        const Vec3 n = cross(b - a, c - a);
        const double m = norm(n);
        farea[i] = 0.5 * m;
        Vec3 nn = (m > 0) ? n * (1.0 / m) : Vec3(0, 0, 1);
        if (dot(nn, unit(fc[i])) < 0) nn = nn * -1.0;   // outward
        fn[i] = nn;
    }
    for (size_t i = 0; i < n_core_f; ++i)
        geomhash += farea[i] + static_cast<double>(i + 1) * farea[i] * 1e-9;
    std::printf("geomhash  : %.10g   (matches the PTM_HMATRIX_VF_REPLACE mesh-identity test)\n",
                geomhash);

    // ------------------------------------------------------------- site frame
    Vec3 C;
    if (args.have_center) {
        // Direction from lat/lon; radius resolved by the drop ray below.
        C = dirFromLatLon(args.center_lat, args.center_lon);
    } else {
        Vec3 s(0, 0, 0);
        for (size_t i = 0; i < n_core_f; ++i) s = s + fc[i];
        C = unit(s * (1.0 / static_cast<double>(n_core_f)));
        args.center_lat = std::asin(C.z) * 180.0 / kPi;
        args.center_lon = std::atan2(C.y, C.x) * 180.0 / kPi;
        std::printf("center    : (not given) using core centroid direction "
                    "lat=%.6f lon=%.6f\n", args.center_lat, args.center_lon);
    }
    // U = C, NOT unit(C): dirFromLatLon already normalized, and the centroid
    // branch below normalizes its own average. Re-normalizing here divided by a
    // |C| that differs from 1 by an ulp and put the site frame one bit away from
    // the direction a named location computes from the same lat/lon.
    const Vec3 U = C;                             // local vertical (plumb line)
    const Vec3 zbody(0, 0, 1);
    const double zdotu = dot(zbody, U);
    if (std::fabs(zdotu) > std::cos(0.1 * kPi / 180.0)) {
        std::cerr << "FATAL: site is within 0.1 deg of the spin pole (|z.U| = " << zdotu
                  << "); local north is undefined and the face naming would be arbitrary. "
                     "Refusing rather than emitting a frame nobody can bolt hardware to.\n";
        return 3;
    }
    const Vec3 N = unit(zbody - U * zdotu);       // local north
    const Vec3 E = cross(N, U);                   // local east; (E,N,U) right-handed
    std::printf("frame     : lat=%.6f lon=%.6f  |z.U|=%.6f (%.2f deg from pole)\n",
                args.center_lat, args.center_lon, std::fabs(zdotu),
                90.0 - std::acos(std::fabs(zdotu)) * 180.0 / kPi);

    // Mean core radius, to give the drop rays a sane start altitude.
    double r_mean = 0.0, r_max = 0.0, r_min = 1e300;
    for (size_t i = 0; i < n_core_f; ++i) {
        const double r = norm(fc[i]);
        r_mean += r; r_max = std::max(r_max, r); r_min = std::min(r_min, r);
    }
    r_mean /= static_cast<double>(n_core_f);
    const double relief = r_max - r_min;
    std::printf("relief    : r_min=%.3f r_max=%.3f (relief %.3f m), r_mean=%.3f m\n",
                r_min, r_max, relief, r_mean);

    // --------------------------------------------- local facet length scale
    // The site-relative height factor multiplies the MEDIAN EQUIVALENT EDGE of
    // the CORE faces whose centroid falls inside the grid disk. Defined here,
    // measured here, and written into the file -- never assumed from the
    // mesh-wide statistics, which at a refined site are several times coarser
    // than the terrain the cubes actually sit on.
    const Vec3 C_surf_guess = U * r_mean;
    const double kLocalScaleDisk =
        (args.radius_m >= 50.0) ? args.radius_m : 50.0;
    std::vector<double> local_edges;
    local_edges.reserve(n_core_f / 4 + 16);
    for (size_t i = 0; i < n_core_f; ++i) {
        const Vec3 d = fc[i] - C_surf_guess;
        const double de = dot(d, E), dn = dot(d, N);
        // The local length scale is a property of the TERRAIN, not of the
        // lattice, so it is measured over a sensible neighborhood even when the
        // lattice is tiny or absent. Without the floor, --radius 0 (one origin
        // cube) and --radius <0 (named locations only) sample an empty disk and
        // the tool aborts with "no core faces inside the grid disk".
        if (de * de + dn * dn <= kLocalScaleDisk * kLocalScaleDisk)
            local_edges.push_back(equivalentEdge(farea[i]));
    }
    if (local_edges.empty()) {
        std::cerr << "FATAL: no core faces within " << kLocalScaleDisk
                  << " m of the requested center. The grid does not overlap the "
                     "core mesh; check --center-lat/--center-lon.\n";
        return 3;
    }
    std::nth_element(local_edges.begin(), local_edges.begin() + local_edges.size() / 2,
                     local_edges.end());
    const double local_median_edge = local_edges[local_edges.size() / 2];
    std::printf("local scale: %zu core faces within %.0f m, median equivalent edge %.4f m%s\n",
                local_edges.size(), kLocalScaleDisk, local_median_edge,
                (kLocalScaleDisk != args.radius_m)
                    ? "  (neighborhood floored at 50 m: the terrain scale is not the lattice's)"
                    : "");

    const bool height_abs = (args.height_m > 0.0);
    const bool edge_abs   = (args.edge_m   > 0.0);
    const double height = height_abs ? args.height_m : args.height_factor * local_median_edge;
    const double edge   = edge_abs   ? args.edge_m   : args.edge_factor   * local_median_edge;
    std::printf("height    : %.4f m  (%s)\n", height,
                height_abs ? "ABSOLUTE override --height-m"
                           : "site-relative: height_factor x local median edge");
    std::printf("edge      : %.4f m  (%s)\n", edge,
                edge_abs ? "ABSOLUTE override --edge-m"
                         : "site-relative: edge_factor x local median edge");
    if (!height_abs && height < 0.02) {
        std::printf("  NOTE: the resolved height is below 2 cm. That is legitimate on a very "
                    "fine mesh, but confirm it is what you meant -- --height-m overrides it.\n");
    }

    // ----------------------------------------------------------- Embree scene
    RTCDevice dev = rtcNewDevice("");
    if (!dev) { std::cerr << "FATAL: could not create Embree device\n"; return 3; }
    rtcSetDeviceErrorFunction(dev, embreeError, nullptr);
    RTCScene scene = rtcNewScene(dev);
    rtcSetSceneFlags(scene, RTC_SCENE_FLAG_ROBUST);
    rtcSetSceneBuildQuality(scene, RTC_BUILD_QUALITY_HIGH);

    // Recenter on the SITE, not the mesh bounding box: Embree's vertex buffer is
    // float32, and at a selenocentric magnitude of 1.74e6 m that is a ~0.1 m
    // quantum -- larger than the cube height itself. Referred to the site, the
    // grid rim sits at 500 m where float32 resolves ~0.03 mm.
    const Vec3 org = U * r_mean;
    auto attachMesh = [&](const std::vector<std::array<double,3>>& V,
                          const std::vector<std::array<int,3>>& F) {
        RTCGeometry g = rtcNewGeometry(dev, RTC_GEOMETRY_TYPE_TRIANGLE);
        rtcSetGeometryBuildQuality(g, RTC_BUILD_QUALITY_HIGH);
        float* vb = static_cast<float*>(rtcSetNewGeometryBuffer(
            g, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, sizeof(float) * 3, V.size()));
        for (size_t i = 0; i < V.size(); ++i) {
            vb[i*3+0] = static_cast<float>(V[i][0] - org.x);
            vb[i*3+1] = static_cast<float>(V[i][1] - org.y);
            vb[i*3+2] = static_cast<float>(V[i][2] - org.z);
        }
        uint32_t* ib = static_cast<uint32_t*>(rtcSetNewGeometryBuffer(
            g, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, sizeof(uint32_t) * 3, F.size()));
        for (size_t i = 0; i < F.size(); ++i) {
            ib[i*3+0] = static_cast<uint32_t>(F[i][0]);
            ib[i*3+1] = static_cast<uint32_t>(F[i][1]);
            ib[i*3+2] = static_cast<uint32_t>(F[i][2]);
        }
        rtcCommitGeometry(g);
        const unsigned id = rtcAttachGeometry(scene, g);
        rtcReleaseGeometry(g);
        return id;
    };
    const unsigned geom_core = attachMesh(core->vertices, core->faces);
    unsigned geom_ring = 0xFFFFFFFFu;
    if (!ring_f.empty()) geom_ring = attachMesh(ring_v, ring_f);
    rtcCommitScene(scene);
    std::printf("embree    : scene built (core geomID=%u", geom_core);
    if (geom_ring != 0xFFFFFFFFu) std::printf(", ring geomID=%u", geom_ring);
    std::printf(")\n");

    auto castRay = [&](const Vec3& o, const Vec3& d, float tnear, float tfar,
                       unsigned& gid, unsigned& pid, float& thit) {
        RTCRayHit rh{};
        rh.ray.org_x = static_cast<float>(o.x - org.x);
        rh.ray.org_y = static_cast<float>(o.y - org.y);
        rh.ray.org_z = static_cast<float>(o.z - org.z);
        rh.ray.dir_x = static_cast<float>(d.x);
        rh.ray.dir_y = static_cast<float>(d.y);
        rh.ray.dir_z = static_cast<float>(d.z);
        rh.ray.tnear = tnear;
        rh.ray.tfar  = tfar;
        rh.ray.mask  = 0xFFFFFFFF;
        rh.hit.geomID = RTC_INVALID_GEOMETRY_ID;
        rh.hit.primID = RTC_INVALID_GEOMETRY_ID;
        rtcIntersect1(scene, &rh);
        gid = rh.hit.geomID; pid = rh.hit.primID; thit = rh.ray.tfar;
    };

    // ------------------------------------------------------------- placement
    const int n_lat = static_cast<int>(std::floor(args.radius_m / args.spacing_m));
    std::vector<CubeRecord> cubes;
    std::vector<Vec3> centers;
    const double H_hi = relief + 100.0;
    int n_miss = 0, n_ring_hit = 0;
    double max_tilt = 0.0;

    // ONE placement routine for both the lattice and the named locations. That
    // is deliberate and it is what the byte-identity discriminator tests: a
    // named location given a lattice cube's own coordinates must produce a
    // bit-identical cube, which can only hold if neither path has its own copy
    // of the drop ray, the frame projection, or the slope/aspect arithmetic.
    std::vector<std::string> names;
    const bool want_names = !named.empty();
    auto place = [&](double x, double y, double h_use, double e_use,
                     int gi, int gj, const char* what) -> bool {
        const Vec3 start = org + E * x + N * y + U * H_hi;
        unsigned gid, pid; float t;
        castRay(start, U * -1.0, 0.0f, static_cast<float>(H_hi + relief + 200.0), gid, pid, t);
        if (gid == RTC_INVALID_GEOMETRY_ID) {
            if (n_miss < 5)
                std::cerr << "FATAL: " << what << " at (" << x << ", " << y
                          << ") m found no terrain below it.\n";
            ++n_miss; return false;
        }
        if (gid != geom_core) {
            if (n_ring_hit < 5)
                std::cerr << "FATAL: " << what << " at (" << x << ", " << y
                          << ") m landed on the OCCLUDER RING, which carries no thermal "
                             "solution -- there is no T_face and no radiosity there.\n";
            ++n_ring_hit; return false;
        }
        const Vec3 P = start + (U * -1.0) * static_cast<double>(t);
        CubeRecord cr{};
        cr.grid_i = gi; cr.grid_j = gj;
        cr.x_m = x; cr.y_m = y;
        const Vec3 xc = P + U * h_use;
        cr.center[0] = xc.x; cr.center[1] = xc.y; cr.center[2] = xc.z;
        cr.face_id = static_cast<int32_t>(pid);
        cr.edge_m = static_cast<float>(e_use);
        const Vec3& nj = fn[pid];
        cr.slope_deg  = static_cast<float>(std::acos(std::max(-1.0, std::min(1.0, dot(nj, U)))) * 180.0 / kPi);
        double asp = std::atan2(dot(nj, E), dot(nj, N)) * 180.0 / kPi;
        if (asp < 0) asp += 360.0;
        cr.aspect_deg = static_cast<float>(asp);
        cr.area_m2 = static_cast<float>(farea[pid]);
        cr.elev_m  = static_cast<float>(dot(P - org, U));
        cr.height_achieved_m = h_use;
        cubes.push_back(cr);
        centers.push_back(xc);
        max_tilt = std::max(max_tilt,
            std::acos(std::max(-1.0, std::min(1.0, dot(U, unit(xc))))) * 180.0 / kPi);
        return true;
    };

    // radius < 0 means NO lattice at all -- named locations only. (radius == 0
    // still yields the single origin cube, which is a different, useful thing.)
    if (args.radius_m >= 0.0) {
        for (int i = -n_lat; i <= n_lat; ++i) {
            for (int j = -n_lat; j <= n_lat; ++j) {
                const double x = i * args.spacing_m, y = j * args.spacing_m;
                if (x * x + y * y > args.radius_m * args.radius_m) continue;
                if (place(x, y, height, edge, i, j, "lattice point") && want_names)
                    names.emplace_back();      // lattice cubes carry an empty name
            }
        }
    } else if (named.empty()) {
        std::cerr << "FATAL: --radius < 0 disables the lattice, but no --locations were given, "
                     "so there would be no cubes at all.\n";
        rtcReleaseScene(scene); rtcReleaseDevice(dev);
        return 3;
    }
    const size_t n_lattice = cubes.size();

    // --- named locations, appended after the lattice -----------------------
    for (const NamedLoc& L : named) {
        // lat/lon -> the SITE frame, via the same basis the lattice uses. At the
        // site's own lat/lon this yields exactly (0, 0), so a named location
        // placed there reproduces the lattice origin cube bit for bit.
        // dirFromLatLon already returns a unit vector and the site frame now
        // takes U = C directly, so ONE call is the whole story on both paths.
        // (An earlier version compensated by normalizing twice here to match a
        // double-normalized U -- symptom-matching. The clean fix is to stop
        // normalizing twice on the other side, which is what U = C does.)
        const Vec3 d = dirFromLatLon(L.lat, L.lon);
        // Difference the DIRECTIONS first, then scale. The algebraically
        // identical `d * r_mean - org` subtracts two ~1.7e6 m vectors to get a
        // ~0 m result -- catastrophic cancellation that left a ~1e-11 m residue
        // in the recorded offset even when the directions agreed exactly, and
        // broke the grid-vs-named bit-identity check. Differencing first makes
        // the site-center case exactly zero instead of nearly zero.
        const Vec3 rel = (d - U) * r_mean;
        const double x = dot(rel, E), y = dot(rel, N);
        const double rr = std::sqrt(x * x + y * y);
        if (rr > args.radius_m + 1e-9)
            std::printf("  NOTE: named location '%s' sits %.1f m from the site center, "
                        "outside the %.1f m grid radius. That is allowed -- named locations are "
                        "not constrained by the lattice -- but it must be inside the CORE mesh.\n",
                        L.name.c_str(), rr, args.radius_m);
        const double h_use = (L.height_m > 0.0) ? L.height_m : height;
        const double e_use = (L.edge_m   > 0.0) ? L.edge_m   : edge;
        const std::string what = "named location '" + L.name + "'";
        if (place(x, y, h_use, e_use, kNamedGridIndex, kNamedGridIndex, what.c_str()))
            names.push_back(L.name);
    }
    if (n_miss || n_ring_hit) {
        std::cerr << "FATAL: " << n_miss << " lattice point(s) missed the terrain and "
                  << n_ring_hit << " landed on the ring. Refusing to silently drop cubes: the "
                     "cube count must not change between runs of the same configuration.\n";
        rtcReleaseScene(scene); rtcReleaseDevice(dev);
        return 3;
    }
    const size_t n_cubes = cubes.size();
    std::printf("placement : %zu cubes = %zu on a %.1f m lattice inside r <= %.1f m",
                n_cubes, n_lattice, args.spacing_m, args.radius_m);
    if (n_cubes > n_lattice) std::printf(" + %zu named location(s)", n_cubes - n_lattice);
    std::printf("  (frame max tilt %.4f deg)\n", max_tilt);
    for (size_t k = n_lattice; k < n_cubes; ++k)
        std::printf("            '%s': E=%+.2f N=%+.2f m, face %d, h=%.4f m, edge=%.4f m\n",
                    names[k].c_str(), cubes[k].x_m, cubes[k].y_m,
                    cubes[k].face_id, cubes[k].height_achieved_m, cubes[k].edge_m);

    // ---------------------------------------------------------- hemisphere sweep
    const std::array<Vec3, kNumFaces> face_normals = {
        U, U * -1.0, N, N * -1.0, E, E * -1.0
    };
    const size_t n_patch = n_cubes * kNumFaces;
    std::vector<PatchRecord> patches(n_patch);
    std::vector<std::vector<int32_t>> rows_c(n_patch);
    std::vector<std::vector<float>>   rows_v(n_patch);

    const uint32_t N_rays = args.n_rays;
    const uint32_t N_half = N_rays / 2;
    const float tfar = static_cast<float>(4.0 * (r_max - r_min) + 4.0 * args.radius_m + 2.0e5);

    std::printf("sweep     : %zu patches x %u rays = %.3f M rays "
                "(Shirley-Chiu + per-patch Cranley-Patterson, seed %llu)\n",
                n_patch, N_rays,
                static_cast<double>(n_patch) * N_rays / 1e6,
                static_cast<unsigned long long>(args.seed));

    #pragma omp parallel for schedule(dynamic, 8)
    for (long long p = 0; p < static_cast<long long>(n_patch); ++p) {
        const size_t ci = static_cast<size_t>(p) / kNumFaces;
        const int    fi = static_cast<int>(static_cast<size_t>(p) % kNumFaces);
        const Vec3&  n  = face_normals[fi];
        const Vec3&  o  = centers[ci];
        const Frame  fr = buildFrame(n);

        // Per-patch Cranley-Patterson offsets. Deterministic in (seed, patch),
        // so the cache is reproducible, but decorrelated between patches so the
        // shared point grain cannot print a stripe across the lattice.
        std::mt19937_64 rng(args.seed * 0x9E3779B97F4A7C15ULL + static_cast<uint64_t>(p) + 1ULL);
        std::uniform_real_distribution<double> uni(0.0, 1.0);
        const double cp_a[2] = {uni(rng), uni(rng)};   // half-row A
        const double cp_b[2] = {uni(rng), uni(rng)};   // half-row B, independent

        std::map<int32_t, uint32_t> hits_a, hits_b;
        uint32_t sky_a = 0, sky_b = 0, ring_a = 0, ring_b = 0;

        for (uint32_t k = 0; k < N_rays; ++k) {
            const bool half_a = (k < N_half);
            const double* cp = half_a ? cp_a : cp_b;
            const uint32_t kk = half_a ? k : (k - N_half);
            const uint32_t nn = half_a ? N_half : (N_rays - N_half);
            double u = (static_cast<double>(kk) + 0.5) / static_cast<double>(nn) + cp[0];
            double v = radicalInverse2(kk) + cp[1];
            u -= std::floor(u); v -= std::floor(v);

            double dx, dy;
            concentricDisk(u, v, dx, dy);
            const double dz = std::sqrt(std::max(0.0, 1.0 - dx * dx - dy * dy));
            const Vec3 d = unit(fr.t * dx + fr.bt * dy + fr.n * dz);

            unsigned gid, pid; float t;
            castRay(o, d, 0.0f, tfar, gid, pid, t);
            if (gid == RTC_INVALID_GEOMETRY_ID) { if (half_a) ++sky_a;  else ++sky_b;  }
            else if (gid == geom_core)          { if (half_a) ++hits_a[static_cast<int32_t>(pid)];
                                                  else       ++hits_b[static_cast<int32_t>(pid)]; }
            else                                { if (half_a) ++ring_a; else ++ring_b; }
        }

        // Combined row = A + B, normalized by the full ray count.
        std::map<int32_t, uint32_t> hits = hits_a;
        for (const auto& kv : hits_b) hits[kv.first] += kv.second;

        const double invN = 1.0 / static_cast<double>(N_rays);
        std::vector<std::pair<int32_t, float>> row;
        row.reserve(hits.size());
        double f_max = 0.0, sumF = 0.0;
        for (const auto& kv : hits) {
            const double F = kv.second * invN;
            row.emplace_back(kv.first, static_cast<float>(F));
            f_max = std::max(f_max, F);
            sumF += F;
        }
        // Optional lossy truncation. Default 0 => this loop does nothing. When it
        // is on, the dropped weight is measured and reported, never assumed small.
        double dropped = 0.0;
        if (args.truncate_rel > 0.0) {
            const double cut = args.truncate_rel * f_max;
            std::vector<std::pair<int32_t, float>> kept;
            kept.reserve(row.size());
            for (const auto& e : row) {
                if (e.second >= cut) kept.push_back(e);
                else dropped += e.second;
            }
            row.swap(kept);
            sumF -= dropped;
        }

        // Partners needed to reach half the terrain weight.
        std::vector<float> vals;
        vals.reserve(row.size());
        for (const auto& e : row) vals.push_back(e.second);
        std::sort(vals.begin(), vals.end(), std::greater<float>());
        double acc = 0.0; uint32_t n50 = 0;
        for (float v : vals) { acc += v; ++n50; if (acc >= 0.5 * sumF) break; }

        // Split-half total variation: 0.5 * sum_j |F_a[j] - F_b[j]|, each half-row
        // normalized by its own ray count. This is the MC accuracy metric, and it
        // bounds the flux disagreement directly: |E_a - E_b| <= TV * max_j B_j.
        const double invA = 1.0 / static_cast<double>(N_half);
        const double invB = 1.0 / static_cast<double>(N_rays - N_half);
        double tv = 0.0;
        {
            // Union of the two half-rows' keys; a face hit by only one half still
            // contributes its full weight to the disagreement, which is the point.
            for (const auto& kv : hits) {
                const auto ia = hits_a.find(kv.first);
                const auto ib = hits_b.find(kv.first);
                const double fa = (ia == hits_a.end() ? 0.0 : ia->second * invA);
                const double fb = (ib == hits_b.end() ? 0.0 : ib->second * invB);
                tv += std::fabs(fa - fb);
            }
            tv *= 0.5;
        }

        PatchRecord& pr = patches[static_cast<size_t>(p)];
        pr.row_len  = static_cast<uint32_t>(row.size());
        pr.n_faces_50pct = n50;
        pr.f_sky  = static_cast<float>((sky_a  + sky_b ) * invN);
        pr.f_ring = static_cast<float>((ring_a + ring_b) * invN);
        pr.f_max  = static_cast<float>(f_max);
        pr.closure_residual = static_cast<float>(std::fabs(sumF + dropped
                                + (sky_a + sky_b) * invN + (ring_a + ring_b) * invN - 1.0));
        pr.splithalf_tv = static_cast<float>(tv);
        pr.splithalf_fsky = static_cast<float>(std::fabs(sky_a * invA - sky_b * invB));

        rows_c[static_cast<size_t>(p)].reserve(row.size());
        rows_v[static_cast<size_t>(p)].reserve(row.size());
        for (const auto& e : row) {
            rows_c[static_cast<size_t>(p)].push_back(e.first);
            rows_v[static_cast<size_t>(p)].push_back(e.second);
        }
    }

    // --------------------------------------------------------------- reporting
    std::printf("\n--- per face orientation (p50 / p99 / max) ---\n");
    std::printf("%-8s %10s %10s %10s %10s %10s %10s\n",
                "face", "row_len", "f_sky", "f_ring", "f_max", "n50", "splithalf");
    double worst_closure = 0.0;
    for (int f = 0; f < kNumFaces; ++f) {
        std::vector<double> rl, fs, fr_, fm, n5, sh;
        for (size_t c = 0; c < n_cubes; ++c) {
            const PatchRecord& pr = patches[c * kNumFaces + f];
            rl.push_back(pr.row_len); fs.push_back(pr.f_sky); fr_.push_back(pr.f_ring);
            fm.push_back(pr.f_max);   n5.push_back(pr.n_faces_50pct); sh.push_back(pr.splithalf_tv);
            worst_closure = std::max(worst_closure, static_cast<double>(pr.closure_residual));
        }
        auto q = [](std::vector<double>& v, double p) {
            if (v.empty()) return 0.0;
            size_t k = static_cast<size_t>(p * (v.size() - 1));
            std::nth_element(v.begin(), v.begin() + k, v.end());
            return v[k];
        };
        std::vector<double> rl2 = rl, fs2 = fs, fr2 = fr_, fm2 = fm, n52 = n5, sh2 = sh;
        std::printf("%-8s %10.0f %10.4f %10.5f %10.4f %10.0f %10.2e\n",
                    faceName(f), q(rl2, 0.5), q(fs2, 0.5), q(fr2, 0.5),
                    q(fm2, 0.5), q(n52, 0.5), q(sh2, 0.5));
        std::vector<double> rl3 = rl, fm3 = fm, sh3 = sh, fs3 = fs;
        std::printf("%-8s %10.0f %10.4f %10s %10.4f %10s %10.2e   <- p99\n",
                    "", q(rl3, 0.99), q(fs3, 0.99), "", q(fm3, 0.99), "", q(sh3, 0.99));
    }
    std::printf("\nworst closure residual over all patches: %.3e "
                "(bookkeeping check; the estimator sums to 1 by construction)\n", worst_closure);
    if (worst_closure > 1e-6) {
        std::cerr << "FATAL: closure residual " << worst_closure
                  << " is far above float round-off. A ray channel is being dropped or "
                     "double counted; the rows are not trustworthy.\n";
        rtcReleaseScene(scene); rtcReleaseDevice(dev);
        return 3;
    }

    // f_ring disclosure -- measured, never assumed zero.
    double fring_max = 0.0;
    for (const auto& pr : patches) fring_max = std::max(fring_max, static_cast<double>(pr.f_ring));
    std::printf("max f_ring over all patches: %.6f%s\n", fring_max,
                fring_max > 1e-4
                    ? "   <- NON-ZERO: the grid sees ring geometry, which carries NO radiosity. "
                      "That IR source is missing from the product; report it."
                    : "   (grid sits inside the core, as intended)");

    // ------------------------------------------------------------------ write
    uint64_t total_nnz = 0;
    for (size_t p = 0; p < n_patch; ++p) {
        patches[p].row_offset = total_nnz;
        total_nnz += patches[p].row_len;
    }

    char json[4096];
    std::snprintf(json, sizeof(json),
        "{\"schema\":1,\"tool\":\"ptm_flux_cube_geom\","
        "\"core_mesh\":\"%s\",\"ring_mesh\":\"%s\","
        "\"units\":\"W m-2 incident (no albedo/emissivity applied)\","
        "\"height_definition\":\"plumb line along the site vertical U; "
        "height = height_factor * median equivalent edge sqrt(4A/sqrt(3)) of core faces "
        "inside the grid disk, unless height_m overrides\","
        "\"face_order\":[\"zenith\",\"nadir\",\"north\",\"south\",\"east\",\"west\"],"
        "\"frame\":\"site-local ENU, right-handed, E x N = U\"}",
        args.core.c_str(), args.ring.c_str());
    const uint32_t json_len = static_cast<uint32_t>(std::strlen(json));

    FcgHeader h{};
    std::memcpy(h.magic, "FCG1", 4);
    h.version = 1;
    h.n_cubes = static_cast<uint32_t>(n_cubes);
    h.n_faces_per_cube = kNumFaces;
    h.total_nnz = total_nnz;
    h.geomhash_core = geomhash;
    h.n_faces_core = n_core_f;
    h.ring_hash = ring_hash;
    h.center_lat_deg = args.center_lat;
    h.center_lon_deg = args.center_lon;
    h.spacing_m = args.spacing_m;
    h.radius_m = args.radius_m;
    h.height_m = height;
    h.edge_m = edge;
    h.n_rays = N_rays;
    h.sampler_id = kSamplerShirleyChiuCP;
    h.seed = args.seed;
    h.truncate_rel = args.truncate_rel;
    h.height_factor = height_abs ? 0.0 : args.height_factor;
    h.edge_factor   = edge_abs   ? 0.0 : args.edge_factor;
    h.local_median_edge_m = local_median_edge;
    h.frame_origin[0] = org.x; h.frame_origin[1] = org.y; h.frame_origin[2] = org.z;
    h.frame_east[0]  = E.x; h.frame_east[1]  = E.y; h.frame_east[2]  = E.z;
    h.frame_north[0] = N.x; h.frame_north[1] = N.y; h.frame_north[2] = N.z;
    h.frame_up[0]    = U.x; h.frame_up[1]    = U.y; h.frame_up[2]    = U.z;
    h.frame_max_tilt_deg = max_tilt;
    h.json_len = json_len;
    h.has_names = want_names ? 1u : 0u;

    std::ofstream out(args.out, std::ios::binary);
    if (!out) { std::cerr << "FATAL: cannot open '" << args.out << "' for writing\n"; return 3; }
    out.write(reinterpret_cast<const char*>(&h), sizeof(h));
    out.write(reinterpret_cast<const char*>(cubes.data()),
              static_cast<std::streamsize>(cubes.size() * sizeof(CubeRecord)));
    out.write(reinterpret_cast<const char*>(patches.data()),
              static_cast<std::streamsize>(patches.size() * sizeof(PatchRecord)));
    for (size_t p = 0; p < n_patch; ++p)
        out.write(reinterpret_cast<const char*>(rows_c[p].data()),
                  static_cast<std::streamsize>(rows_c[p].size() * sizeof(int32_t)));
    for (size_t p = 0; p < n_patch; ++p)
        out.write(reinterpret_cast<const char*>(rows_v[p].data()),
                  static_cast<std::streamsize>(rows_v[p].size() * sizeof(float)));
    out.write(json, json_len);
    // Names LAST, after the JSON: every reader written against the original
    // layout parses everything before this at unchanged offsets and ignores
    // the tail, so adding named locations did not break the format.
    if (want_names) {
        std::vector<char> nb(static_cast<size_t>(n_cubes) * kNameLen, 0);
        for (size_t k = 0; k < n_cubes && k < names.size(); ++k)
            std::strncpy(&nb[k * kNameLen], names[k].c_str(), kNameLen - 1);
        out.write(nb.data(), static_cast<std::streamsize>(nb.size()));
    }
    out.close();

    const double mb = (sizeof(FcgHeader) + cubes.size() * sizeof(CubeRecord)
                       + patches.size() * sizeof(PatchRecord)
                       + total_nnz * (sizeof(int32_t) + sizeof(float)) + json_len) / 1048576.0;
    std::printf("\nwrote %s  (%zu cubes, %zu patches, %llu nnz, %.1f MB)\n",
                args.out.c_str(), n_cubes, n_patch,
                static_cast<unsigned long long>(total_nnz), mb);

    rtcReleaseScene(scene);
    rtcReleaseDevice(dev);
    return 0;
}
