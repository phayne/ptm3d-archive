// test_shadow_depth.cpp
// ---------------------------------------------------------------------------
// The falsifier for the shadow-depth pass, on SYNTHETIC geometry whose umbral
// ceiling is a closed form. No mesh file, no Embree, no SPICE, no GPU, no
// engine -- the pass takes its ray casting as a callback, so this file supplies
// a deterministic Moller-Trumbore caster over triangles it built itself and
// knows the truth analytically.
//
// ⭐ THE CLOSED FORMS CARRY theta_sun, AND THAT IS THE POINT OF THIS FILE.
//
//   STEP WALL   flat plane, one vertical step of height Z at x = 0, Sun at
//               elevation eps. For a face on the LOW side at downrange
//               distance d:
//                   h* = max(0, Z - d * tan(eps + theta_sun))
//
//   CYLINDRICAL PIT   radius R, depth D. A floor face at distance s from the
//               rim along the anti-solar azimuth:
//                   h* = max(0, D - s * tan(eps + theta_sun))
//
// theta_sun is the SOLAR ANGULAR RADIUS, asin(R_sun / distance) = 0.2666 deg at
// 1 AU. "Not even partially visible" means the entire disk is occluded, so the
// ray that decides the ceiling is the one to the UPPER LIMB, at elevation
// eps + theta_sun, not the one to the disk center. A closed form written with
// eps alone disagrees with correct code by exactly the solar angular radius --
// d*sec^2(eps)*theta_sun, which is 0.19 m at d = 20 m and eps = 45 deg, ten
// times the search resolution -- and it would look like a bug in the code
// rather than in the test.
//
// What each arm can falsify:
//
//  A. STEP WALL, four elevations x six downrange distances. Falsifies the
//     search ladder, the vertical convention, the theta_sun term and the
//     tan() sign all at once. A point-sun implementation fails every cell
//     whose d is large enough for d*theta_sun to exceed the resolution, and
//     the test prints the point-sun prediction beside the umbral one so the
//     discrimination is visible rather than asserted.
//
//  B. STEP WALL, Z ABOVE the search ceiling. h* > h_max must be the SENTINEL
//     (+inf), not h_max and not a number. Falsifies an implementation that
//     clamps.
//
//  C. CYLINDRICAL PIT, four elevations x four azimuths. The azimuth is the
//     part arm A cannot test: a step wall is azimuth-degenerate, so an
//     implementation that ignores the Sun's azimuth passes A and fails C.
//     Deep pit (D > h_max) forces the sentinel from a second direction.
//
//  D. CLEAR SKY. No occluder at all -> exactly 0.0f at every face, at every
//     elevation. Falsifies an implementation that reports the surface offset
//     as a ceiling.
//
//  E. RESOLUTION. The reported value must lie within half the final bracket of
//     the closed form. Asserted as a bound, not as a tolerance chosen after
//     seeing the answer.
//
//  F-K. NEGATIVE CONTROLS, each of which MUST be refused: no ray caster, no
//     scene origin offset, a tnear too coarse for the search, a point sun
//     (solar_radius_km = 0), an inconsistent time axis, and a batched caster
//     that returns a short result vector.
//
// Exit code = number of failures.
// ---------------------------------------------------------------------------

#include "../fileio/frame_naming.h"
#include "../illumination/shadow_depth_pass.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

using namespace ptm::shadowdepth;

namespace {

using V3 = std::array<double, 3>;

constexpr double kPi = 3.14159265358979323846;
// The engine's own constants, in km. constants::SUN_RADIUS is 696000 and the
// test places the Sun at exactly 1 AU so theta_sun is reproducible here and in
// the pass from the same two numbers.
constexpr double kSunRadiusKm = 696000.0;
constexpr double kAuKm        = 149597870.7;
// The body radius the synthetic patch is lifted to, so that the RADIAL vertical
// the pass uses is a real radial and not a special case. Lunar, in meters.
constexpr double kBodyRadiusM = 1737400.0;

V3 sub(const V3& a, const V3& b) { return {a[0]-b[0], a[1]-b[1], a[2]-b[2]}; }
double dot(const V3& a, const V3& b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }
V3 cross(const V3& a, const V3& b) {
    return {a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]};
}
double len(const V3& a) { return std::sqrt(dot(a, a)); }
V3 unit(const V3& a) { const double l = len(a); return {a[0]/l, a[1]/l, a[2]/l}; }

// -------------------------------------------------------------------------
// The synthetic world.
//
// Local frame: X is UP (the body radial, since the patch sits at x = R on the
// +X axis), Y is EAST, Z is NORTH. A local point (u, e, n) with u the height
// above the datum maps to the mesh frame as (R + u, e, n). Over a patch a few
// tens of meters across the radial tilts by at most |e,n|/R = 1.7e-5 rad, which
// moves a 5 m ceiling by under 0.5 mm -- two orders below the resolution, and
// it is a real curvature the pass is right to carry.
// -------------------------------------------------------------------------
struct Tri { V3 a, b, c; };

V3 world(double u, double e, double n) { return {kBodyRadiusM + u, e, n}; }

// A flat quad at height u spanning [e0,e1] x [n0,n1].
void addQuad(std::vector<Tri>& T, double u, double e0, double e1,
             double n0, double n1) {
    T.push_back({world(u, e0, n0), world(u, e1, n0), world(u, e1, n1)});
    T.push_back({world(u, e0, n0), world(u, e1, n1), world(u, e0, n1)});
}

// A vertical quad in the plane e = e_const, spanning [u0,u1] x [n0,n1].
void addWallE(std::vector<Tri>& T, double e_const, double u0, double u1,
              double n0, double n1) {
    T.push_back({world(u0, e_const, n0), world(u1, e_const, n0), world(u1, e_const, n1)});
    T.push_back({world(u0, e_const, n0), world(u1, e_const, n1), world(u0, e_const, n1)});
}

// STEP WALL. The high side (u = Z) occupies e < 0; the low side (u = 0)
// occupies e > 0. The riser is the vertical wall at e = 0 from u = 0 to Z. The
// Sun is placed toward -e, so a face at e = d on the low side is shadowed by
// the top edge of the riser at downrange distance d.
std::vector<Tri> stepWall(double Z, double half) {
    std::vector<Tri> T;
    addQuad (T, Z, -half, 0.0, -half, half);       // high plateau
    addQuad (T, 0.0, 0.0, half, -half, half);      // low plain
    addWallE(T, 0.0, 0.0, Z, -half, half);         // the riser
    return T;
}

// CYLINDRICAL PIT of radius R and depth D in a plane at u = 0: the floor is a
// disk at u = -D, the wall is the cylinder e^2 + n^2 = R^2 from u = -D to 0,
// and the exterior is an annulus at u = 0 out to `half`.
std::vector<Tri> cylindricalPit(double R, double D, double half, int nseg = 256) {
    std::vector<Tri> T;
    const V3 floor_c = world(-D, 0.0, 0.0);
    for (int k = 0; k < nseg; ++k) {
        const double a0 = 2.0*kPi*k/nseg, a1 = 2.0*kPi*(k+1)/nseg;
        const double e0 = R*std::cos(a0), n0 = R*std::sin(a0);
        const double e1 = R*std::cos(a1), n1 = R*std::sin(a1);
        // floor fan
        T.push_back({floor_c, world(-D, e0, n0), world(-D, e1, n1)});
        // cylinder wall
        T.push_back({world(-D, e0, n0), world(0.0, e0, n0), world(0.0, e1, n1)});
        T.push_back({world(-D, e0, n0), world(0.0, e1, n1), world(-D, e1, n1)});
        // exterior annulus out to `half`
        const double E0 = half*std::cos(a0), N0 = half*std::sin(a0);
        const double E1 = half*std::cos(a1), N1 = half*std::sin(a1);
        T.push_back({world(0.0, e0, n0), world(0.0, E0, N0), world(0.0, E1, N1)});
        T.push_back({world(0.0, e0, n0), world(0.0, E1, N1), world(0.0, e1, n1)});
    }
    return T;
}

// Moller-Trumbore, with the SAME tnear semantics the engine's batched path
// uses: a hit closer than tnear is not a hit. That is what makes the negative
// control on tnear meaningful rather than decorative.
//
// ⛔ THE BARYCENTRIC TESTS CARRY AN EPSILON, AND IT IS LOAD-BEARING, NOT TIDINESS.
// With strict `u < 0 || u > 1` the two triangles that share an edge both reject a
// ray that passes exactly through that edge, and the ray escapes through a seam
// that does not exist in the geometry. MEASURED here before the epsilon was added:
// the cylindrical-pit arm reported a 0.30 m ceiling where the closed form is
// 2.53 m, at exactly the four (azimuth, offset) cells whose escape ray leaves
// through a polygon VERTEX of the 256-gon wall -- and only those. That was the
// FIXTURE, not the pass: Embree's triangle intersector is watertight by
// construction (Woop et al. 2013), so the engine's own ray path has no such seam,
// and a test that fails on its own caster would have sent someone hunting in the
// search ladder. `kEdgeEps` admits a shared-edge ray to BOTH triangles, which is
// harmless for an occlusion query -- the answer is "blocked" either way.
struct Caster {
    static constexpr double kEdgeEps = 1e-9;

    std::vector<Tri> tris;
    double tnear = 0.01;
    mutable uint64_t rays = 0;

    bool hit(const V3& o, const V3& d) const {
        for (const Tri& t : tris) {
            const V3 e1 = sub(t.b, t.a), e2 = sub(t.c, t.a);
            const V3 p = cross(d, e2);
            const double det = dot(e1, p);
            if (std::fabs(det) < 1e-12) continue;
            const double inv = 1.0 / det;
            const V3 s = sub(o, t.a);
            const double u = dot(s, p) * inv;
            if (u < -kEdgeEps || u > 1.0 + kEdgeEps) continue;
            const V3 q = cross(s, e1);
            const double v = dot(d, q) * inv;
            if (v < -kEdgeEps || u + v > 1.0 + kEdgeEps) continue;
            const double tt = dot(e2, q) * inv;
            if (tt > tnear) return true;
        }
        return false;
    }
    void batch(const std::vector<V3>& o, const std::vector<V3>& d,
               std::vector<uint8_t>& blocked) const {
        blocked.assign(o.size(), 0);
        rays += o.size();
        #pragma omp parallel for schedule(static)
        for (long long i = 0; i < static_cast<long long>(o.size()); ++i)
            blocked[static_cast<size_t>(i)] = hit(o[static_cast<size_t>(i)],
                                                  d[static_cast<size_t>(i)]) ? 1 : 0;
    }
};

// Sun position in KM, at elevation `elev_deg` above the local horizontal and
// azimuth `az_deg` measured from +EAST toward +NORTH. The local horizontal is
// the plane perpendicular to the radial at the patch, i.e. the (e, n) plane.
V3 sunPositionKm(double elev_deg, double az_deg) {
    const double el = elev_deg * kPi / 180.0;
    const double az = az_deg * kPi / 180.0;
    const V3 dir = {std::sin(el), std::cos(el)*std::cos(az), std::cos(el)*std::sin(az)};
    // The patch origin, in km, plus 1 AU along `dir`.
    return {kBodyRadiusM*1e-3 + kAuKm*dir[0], kAuKm*dir[1], kAuKm*dir[2]};
}

// theta_sun as the pass computes it: asin(R_sun / |sun - probe|). The probe sits
// within a few meters of the patch origin, so 1 AU is exact to 12 digits here.
double thetaSun() { return std::asin(kSunRadiusKm / kAuKm); }

struct Harness {
    std::vector<V3> centers, normals;
    std::vector<double> times{0.0};
    std::vector<std::string> tstr{"2028-08-03T22:21:00.000"};
    std::vector<V3> sun;
    Caster caster;

    PassInputs inputs(const std::string& out) {
        PassInputs in;
        in.face_centers = &centers;
        in.face_normals = &normals;
        in.n_faces = centers.size();
        in.times = &times;
        in.time_strings = &tstr;
        in.equilibration_steps = 0;
        in.total_time_steps = static_cast<int>(times.size());
        in.sun_positions = &sun;
        in.mesh_to_km = 1.0e-3;            // the synthetic mesh is in METERS
        in.solar_radius_km = kSunRadiusKm;
        in.solar_disk_samples = 128;
        in.h_max_m = 5.0;
        in.march_steps = 16;
        in.bisect_rounds = 4;
        in.surface_offset_m = 0.0;         // the caster has its own tnear
        in.scene_has_offset = true;
        in.resolved_tnear_m = caster.tnear;
        Caster* c = &caster;
        in.occluded_batch = [c](const std::vector<V3>& o, const std::vector<V3>& d,
                                std::vector<uint8_t>& b) { c->batch(o, d, b); };
        in.output_dir = out;
        in.write_frames = true;
        in.monotonicity_faces = 0;
        in.monotonicity_epochs = 0;
        return in;
    }
};

// Read back the frame the pass wrote: uint32 face count, then float32[F].
bool readFrame(const std::string& path, std::vector<float>& v) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    uint32_t n = 0;
    f.read(reinterpret_cast<char*>(&n), sizeof(uint32_t));
    v.resize(n);
    f.read(reinterpret_cast<char*>(v.data()),
           static_cast<std::streamsize>(n * sizeof(float)));
    return static_cast<bool>(f);
}

int g_fails = 0;

void checkNear(const char* what, double got, double want, double tol) {
    const double d = std::fabs(got - want);
    const bool ok = (d <= tol);
    std::printf("  %-52s got %9.4f  want %9.4f  |d| %8.2e  tol %7.4f  %s\n",
                what, got, want, d, tol, ok ? "PASS" : "FAIL");
    if (!ok) ++g_fails;
}

void checkTrue(const char* what, bool ok) {
    std::printf("  %-52s %s\n", what, ok ? "PASS" : "FAIL");
    if (!ok) ++g_fails;
}

void checkRefused(const char* what, bool ran, const std::string& err) {
    std::printf("  %-52s %s\n", what, ran ? "FAIL (accepted!)" : "PASS (refused)");
    if (ran) ++g_fails; else std::printf("      reason: %.150s\n", err.c_str());
}

const double kBracket = 5.0 / (16.0 * 16.0);   // h_max/(march*2^bisect) = 0.0195 m

}  // namespace

int main() {
    std::printf("\n=== shadow depth: falsifier on synthetic geometry with closed-form "
                "umbral ceilings ===\n");
    const double th = thetaSun();
    std::printf("solar angular radius theta_sun = %.6f deg (%.6e rad) from "
                "asin(%.0f km / %.4f km)\n", th*180.0/kPi, th, kSunRadiusKm, kAuKm);
    const std::string base = "output/shadow_depth/unit";

    // ------------------------------------------------------------------ ARM A
    {
        std::printf("\n--- ARM A: step wall, h* = max(0, Z - d*tan(eps + theta_sun)) ---\n");
        const double Z = 4.0, half = 60.0;
        const double elevs[] = {2.0, 6.0, 15.0, 45.0};
        const double dists[] = {0.5, 1.0, 2.0, 5.0, 10.0, 20.0};
        for (double eps_deg : elevs) {
            Harness H;
            H.caster.tnear = 0.001;
            H.caster.tris = stepWall(Z, half);
            for (double d : dists) {
                H.centers.push_back(world(0.0, d, 0.0));
                H.normals.push_back(unit(world(0.0, d, 0.0)));   // the plain is level
            }
            // Sun toward -EAST (azimuth 180 deg) so the riser at e = 0 shadows
            // the low plain at e > 0.
            H.sun.assign(1, sunPositionKm(eps_deg, 180.0));
            char dir[160];
            std::snprintf(dir, sizeof(dir), "%s/armA_e%02.0f", base.c_str(), eps_deg);
            auto in = H.inputs(dir);
            std::string err;
            if (!runShadowDepthPass(in, err)) {
                std::printf("  FAIL: %s\n", err.c_str()); ++g_fails; continue;
            }
            std::vector<float> v;
            if (!readFrame(ptm::resolveFrameFile(std::string(dir) + "/shadow_depth", "shadow_depth", 0), v)) {
                std::printf("  FAIL: no frame written\n"); ++g_fails; continue;
            }
            const double eps = eps_deg * kPi / 180.0;
            std::printf("  eps = %5.1f deg\n", eps_deg);
            for (size_t i = 0; i < v.size(); ++i) {
                const double d = dists[i];
                const double want  = std::max(0.0, Z - d*std::tan(eps + th));
                const double point = std::max(0.0, Z - d*std::tan(eps));   // the WRONG form
                char lbl[80];
                std::snprintf(lbl, sizeof(lbl), "d = %5.1f m  (point-sun form: %.4f)",
                              d, point);
                checkNear(lbl, v[i], want, 0.5*kBracket + 1e-3);
            }
        }
        // The discrimination, stated as a number rather than asserted: at the
        // largest downrange distance the two forms differ by far more than the
        // bracket, so a point-sun implementation cannot pass this arm.
        const double eps = 45.0 * kPi / 180.0;
        const double gap = 20.0 * (std::tan(eps + th) - std::tan(eps));
        std::printf("  DISCRIMINATION: at d = 20 m, eps = 45 deg the umbral and point-sun\n"
                    "  ceilings differ by %.4f m = %.1f x the %.4f m bracket.\n",
                    gap, gap / kBracket, kBracket);
        checkTrue("arm A discriminates a point-sun implementation", gap > 2.0 * kBracket);
    }

    // ------------------------------------------------------------------ ARM B
    {
        std::printf("\n--- ARM B: Z above the search ceiling -> SENTINEL, not a clamp ---\n");
        const double Z = 12.0, half = 60.0;      // 12 m step, h_max is 5 m
        Harness H;
        H.caster.tnear = 0.001;
        H.caster.tris = stepWall(Z, half);
        // d = 0.5 m: h* = 12 - 0.5*tan(6.27 deg) = 11.945 m, far above h_max.
        H.centers.push_back(world(0.0, 0.5, 0.0));
        H.normals.push_back(unit(world(0.0, 0.5, 0.0)));
        H.sun.assign(1, sunPositionKm(6.0, 180.0));
        const std::string dir = base + "/armB";
        auto in = H.inputs(dir);
        std::string err;
        if (!runShadowDepthPass(in, err)) { std::printf("  FAIL: %s\n", err.c_str()); ++g_fails; }
        else {
            std::vector<float> v;
            readFrame(ptm::resolveFrameFile(dir + "/shadow_depth", "shadow_depth", 0), v);
            checkTrue("Z >> h_max reports +inf (not h_max, not a number)",
                      v.size() == 1 && std::isinf(v[0]) && v[0] > 0.0f);
            // The exact bit pattern, because downstream readers key on it.
            uint32_t bits = 0; std::memcpy(&bits, &v[0], 4);
            char lbl[96];
            std::snprintf(lbl, sizeof(lbl), "sentinel bit pattern is 0x7F800000 (got 0x%08X)", bits);
            checkTrue(lbl, bits == 0x7F800000u);
        }
    }

    // ------------------------------------------------------------------ ARM C
    {
        std::printf("\n--- ARM C: cylindrical pit, h* = max(0, D - s*tan(eps + theta_sun)) ---\n");
        const double R = 8.0, D = 3.0, half = 60.0;
        const double elevs[] = {2.0, 6.0, 15.0, 45.0};
        const double azis[]  = {0.0, 90.0, 200.0, 315.0};
        for (double eps_deg : elevs) {
            for (double az_deg : azis) {
                Harness H;
                H.caster.tnear = 0.001;
                H.caster.tris = cylindricalPit(R, D, half);
                // Floor faces on the anti-solar side, i.e. displaced AWAY from
                // the Sun's azimuth, so the shadowing rim is the sunward one at
                // horizontal distance s along the Sun azimuth.
                const double az = az_deg * kPi / 180.0;
                const V3 to_sun_h = {0.0, std::cos(az), std::sin(az)};
                const double offs[] = {0.0, 2.0, 4.0, 6.0};
                std::vector<double> s_list;
                for (double off : offs) {
                    // face at -off along the Sun azimuth; the sunward rim is
                    // then s = R + off away along that azimuth.
                    H.centers.push_back(world(-D, -off*to_sun_h[1], -off*to_sun_h[2]));
                    H.normals.push_back(unit(world(-D, -off*to_sun_h[1], -off*to_sun_h[2])));
                    s_list.push_back(R + off);
                }
                H.sun.assign(1, sunPositionKm(eps_deg, az_deg));
                char dir[200];
                std::snprintf(dir, sizeof(dir), "%s/armC_e%02.0f_a%03.0f",
                              base.c_str(), eps_deg, az_deg);
                auto in = H.inputs(dir);
                std::string err;
                if (!runShadowDepthPass(in, err)) {
                    std::printf("  FAIL: %s\n", err.c_str()); ++g_fails; continue;
                }
                std::vector<float> v;
                readFrame(ptm::resolveFrameFile(std::string(dir) + "/shadow_depth", "shadow_depth", 0), v);
                const double eps = eps_deg * kPi / 180.0;
                for (size_t i = 0; i < v.size(); ++i) {
                    const double s = s_list[i];
                    const double want = std::max(0.0, D - s*std::tan(eps + th));
                    char lbl[96];
                    std::snprintf(lbl, sizeof(lbl), "eps %4.1f  az %5.1f  s = %4.1f m",
                                  eps_deg, az_deg, s);
                    checkNear(lbl, v[i], want, 0.5*kBracket + 1e-3);
                }
            }
        }
    }

    // -------------------------------------------------------------- ARM C-deep
    {
        std::printf("\n--- ARM C-deep: pit deeper than the search ceiling -> SENTINEL ---\n");
        const double R = 8.0, D = 9.0, half = 60.0;
        Harness H;
        H.caster.tnear = 0.001;
        H.caster.tris = cylindricalPit(R, D, half);
        H.centers.push_back(world(-D, 0.0, 0.0));
        H.normals.push_back(unit(world(-D, 0.0, 0.0)));
        H.sun.assign(1, sunPositionKm(2.0, 45.0));
        const std::string dir = base + "/armCdeep";
        auto in = H.inputs(dir);
        std::string err;
        if (!runShadowDepthPass(in, err)) { std::printf("  FAIL: %s\n", err.c_str()); ++g_fails; }
        else {
            std::vector<float> v;
            readFrame(ptm::resolveFrameFile(dir + "/shadow_depth", "shadow_depth", 0), v);
            checkTrue("deep pit floor reports the +inf sentinel",
                      v.size() == 1 && std::isinf(v[0]));
        }
    }

    // ------------------------------------------------------------------ ARM D
    {
        std::printf("\n--- ARM D: clear sky -> exactly 0.0f (the surface offset is not a "
                    "ceiling) ---\n");
        for (double eps_deg : {2.0, 6.0, 15.0, 45.0}) {
            Harness H;
            H.caster.tnear = 0.001;
            H.caster.tris.clear();                     // nothing to block anything
            for (double d : {0.5, 5.0, 20.0}) {
                H.centers.push_back(world(0.0, d, 0.0));
                H.normals.push_back(unit(world(0.0, d, 0.0)));
            }
            H.sun.assign(1, sunPositionKm(eps_deg, 180.0));
            char dir[160];
            std::snprintf(dir, sizeof(dir), "%s/armD_e%02.0f", base.c_str(), eps_deg);
            auto in = H.inputs(dir);
            std::string err;
            if (!runShadowDepthPass(in, err)) { std::printf("  FAIL: %s\n", err.c_str()); ++g_fails; continue; }
            std::vector<float> v;
            readFrame(ptm::resolveFrameFile(std::string(dir) + "/shadow_depth", "shadow_depth", 0), v);
            bool all_zero = !v.empty();
            for (float x : v) all_zero = all_zero && (x == 0.0f);
            char lbl[80];
            std::snprintf(lbl, sizeof(lbl), "eps = %4.1f deg: every face exactly 0.0f", eps_deg);
            checkTrue(lbl, all_zero);
        }
    }

    // ----------------------------------------------------- NEGATIVE CONTROLS
    std::printf("\n--- NEGATIVE CONTROLS (each MUST be refused) ---\n");
    {
        Harness H;
        H.caster.tris = stepWall(4.0, 60.0);
        H.centers.push_back(world(0.0, 1.0, 0.0));
        H.normals.push_back(unit(world(0.0, 1.0, 0.0)));
        H.sun.assign(1, sunPositionKm(6.0, 180.0));
        {
            auto in = H.inputs(base + "/negA"); in.occluded_batch = nullptr;
            std::string e; checkRefused("no batched ray caster", runShadowDepthPass(in, e), e);
        }
        {
            auto in = H.inputs(base + "/negB"); in.scene_has_offset = false;
            std::string e; checkRefused("scene origin offset NOT applied (the tnear trap)",
                                        runShadowDepthPass(in, e), e);
        }
        {
            auto in = H.inputs(base + "/negC"); in.resolved_tnear_m = 1.74;  // no offset => 1.74 m
            std::string e; checkRefused("tnear 1.74 m against a 5 m search",
                                        runShadowDepthPass(in, e), e);
        }
        {
            auto in = H.inputs(base + "/negD"); in.resolved_tnear_m = -1.0;
            std::string e; checkRefused("caller did not report the resolved tnear",
                                        runShadowDepthPass(in, e), e);
        }
        {
            auto in = H.inputs(base + "/negE"); in.solar_radius_km = 0.0;
            std::string e; checkRefused("point sun (solar_radius_km = 0)",
                                        runShadowDepthPass(in, e), e);
        }
        {
            // eq 0 + 1 output > 0 total: the exported epochs would index past the
            // end of the ephemeris.
            auto in = H.inputs(base + "/negF"); in.total_time_steps = 0;
            std::string e; checkRefused("output epochs exceed total_time_steps",
                                        runShadowDepthPass(in, e), e);
        }
        {
            // POSITIVE control on the same relation, and it is not a formality:
            // a CHUNKED run integrates one grid point past its exported window
            // (thermal.checkpoint_lookahead_steps: 1), so total = eq + output + 1.
            // An `==` assertion here refuses every chunked window -- measured on a
            // 2-chunk crater80N run before this was fixed.
            auto in = H.inputs(base + "/posF");
            in.total_time_steps = 2;               // eq 0 + 1 output + 1 lookahead
            H.sun.push_back(H.sun.front());        // ephemeris covers the extra step
            std::string e;
            const bool ran = runShadowDepthPass(in, e);
            std::printf("  %-52s %s\n", "chunk checkpoint lookahead is ACCEPTED",
                        ran ? "PASS" : "FAIL (refused)");
            if (!ran) { ++g_fails; std::printf("      reason: %.150s\n", e.c_str()); }
            H.sun.pop_back();
        }
        {
            auto in = H.inputs(base + "/negG");
            in.occluded_batch = [](const std::vector<V3>&, const std::vector<V3>&,
                                   std::vector<uint8_t>& b) { b.assign(1, 1); };
            std::string e; checkRefused("caster returns a SHORT result vector",
                                        runShadowDepthPass(in, e), e);
        }
        {
            auto in = H.inputs(base + "/negH");
            std::vector<V3> bad_centers{{0.0, 0.0, 0.0}};   // exactly at the body center
            std::vector<V3> bad_normals{{1.0, 0.0, 0.0}};
            in.face_centers = &bad_centers; in.face_normals = &bad_normals; in.n_faces = 1;
            std::string e; checkRefused("face centroid at the body center (no radial)",
                                        runShadowDepthPass(in, e), e);
        }
    }

    std::printf("\n==== test_shadow_depth: %d failure(s) ====\n\n", g_fails);
    return g_fails;
}
