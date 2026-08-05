#include "illumination_model.h"
#include "primary_disk.h"
#include "../constants/physics_constants.h"
#include "../constants/numeric_constants.h"
#include "../constants/constants.h"  // For ptm_thermal::constants
#include "../thermal/thermal_model.h"
#include "../thermal/planetary_properties.h"
#include "../orbital/kepler_propagator.h"
#include "../orbital/body_fixed_frame.h"
#include "../orbital/generic_body.h"
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <vector>
#include <map>
#include <unordered_map>
#include <random>
#include <limits>
#include <cassert>
#include <array>
#include <mutex>
#include <set>
#include <string>

// Conditionally include x86-specific SIMD headers
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <emmintrin.h>  // SSE2 intrinsics
#include <immintrin.h>  // AVX/AVX2/AVX-512 intrinsics
#endif

#include <filesystem>  // For std::filesystem::exists
#include "config.h"    // For SPICE_KERNELS_DIR

// Embree includes
#include <embree4/rtcore.h>
#include <embree4/rtcore_ray.h>
#include <embree4/rtcore_common.h>

// SPICE includes
#include <SpiceUsr.h>  // Angle brackets to use system include path

// GPU BVH ray tracer (optional)
#ifdef PTM_GPU_OPENCL
#include "../gpu/bvh_opencl.h"
#include "../gpu/bvh_gpu_pool.h"
#endif

namespace constants = ptm_thermal::constants;  

namespace numeric = ptm::numeric;             // For numeric constants

namespace {
// Helper: convert km vectors from SPICE to meters (mesh units)
inline std::array<double,3> kmToMeters(const std::array<double,3>& vkm) {
    return { vkm[0]*1000.0, vkm[1]*1000.0, vkm[2]*1000.0 };
}
// Helper: convert meters to km when needed for irradiance
inline std::array<double,3> metersToKm(const std::array<double,3>& vm) {
    return { vm[0]/1000.0, vm[1]/1000.0, vm[2]/1000.0 };
}
// PTM-36: km -> MESH-NATIVE scale factor for the primary-body (eclipse) shadow sphere.
//
// PTM3D mesh units are NOT uniform: global icosphere meshes are built in KM
// (mesh_utils.cpp createIcosphereMesh scales vertices to body_radius_km_) while
// DEM/site meshes are in METERS. raySphereIntersect() is unit-agnostic -- the ray
// origin, the sphere center and the sphere RADIUS must all share one unit -- so the
// primary position and primary_radius_ (always km, see setPrimaryBody) must be
// converted with the SAME factor. Converting only the center (kmToMeters) shrank
// Jupiter's shadow 1000x and deleted every Europa eclipse; same for Earth/Moon.
// Uses the identical |face_center| > 10000 discriminator already trusted by
// calculatePrimaryIRBatch and calculatePrimaryReflectedBatch.
inline double kmToNativeScale(const std::array<double,3>& face_center) {
    const double mag = std::sqrt(face_center[0]*face_center[0]
                               + face_center[1]*face_center[1]
                               + face_center[2]*face_center[2]);
    return (mag > 10000.0) ? 1000.0 : 1.0;   // meters : kilometers
}

// PTM-45(ii): ONE definition of the penumbral "heavy" solar-disk sample count.
//
// It used to be spelled three different ways -- single face max(64, baseN*4)
// (:3082), CPU batch max(64, baseN) (:3836), GPU batch max(64, N) (:5017) --
// which agree at the default N_SOLAR_SAMPLES = 9 (all 64) and diverge above 16
// samples, at which point the single-face and batch paths would sample the
// SAME geometry differently. The *4 spelling is adopted because "heavy" should
// scale with the requested base count; it is the finer of the two, so
// unifying on it can only improve the batch paths, never coarsen them.
inline uint32_t heavySolarSamples(int num_samples) {
    const uint32_t baseN = static_cast<uint32_t>(std::max(1, num_samples));
    return std::max<uint32_t>(64u, baseN * 4u);
}

// PTM-45(i) / rule 7: an unrecognized primary body is a silent substitution in
// four different places. Say it ONCE per (body, context) instead of never.
inline void warnUnknownPrimaryOnce(const std::string& body,
                                   const char* context,
                                   const char* consequence) {
    static std::mutex mtx;
    static std::set<std::string> seen;
    const std::string key = body + "|" + context;
    {
        std::lock_guard<std::mutex> lk(mtx);
        if (!seen.insert(key).second) return;
    }
    std::cerr << "[illumination] WARNING: primary body '" << body
              << "' is not one of EARTH / JUPITER / SATURN / MARS. " << context
              << ": " << consequence
              << " (warned once; see docs/KNOWN_LIMITATIONS.md, PTM-45)." << std::endl;
}

// PTM-44 / Fermi F1: the horizon-oriented 5-point probe found nothing above the
// face's horizon while the analytic clip still says the disk is (barely) lit.
// Rule 7: the fallback keeps the analytic value, and it announces itself.
inline void warnPrimaryProbeUnderflowOnce() {
    static std::once_flag once;
    std::call_once(once, [](){
        std::cerr << "[illumination] NOTE: primary-disk probe found no "
                     "above-horizon sample on a face whose analytic horizon "
                     "clip is still positive (a ~sin^3(alpha)/2 sliver at the "
                     "bottom of the band). Keeping the analytic, unoccluded "
                     "value there; the flux involved is ~0. Warned once."
                  << std::endl;
    });
}

// PTM-38: does the primary body overlap ANY part of the finite solar disk seen
// from this ray origin?  True in the penumbra AND in the umbra; false when the
// Sun is completely clear of the primary.
//
// The center-ray fast paths test only the ray to the disk CENTER, which the
// primary does not cover until roughly half the disk is already gone.  Left
// ungated, a face 40 % obscured takes the fast path and reports FULL sun, so no
// amount of fixing the sampling denominator can produce a penumbra.  This test
// is a strict superset of the center-ray test (a center hit implies
// sep < ang_primary < ang_primary + disk_ang), so vetoing the fast path on it
// only ever routes MORE faces into the finite-disk sampling that already
// handles partial obscuration correctly -- and it is exactly false whenever no
// eclipse is in progress, which keeps every non-eclipse run byte-identical.
//
// All lengths in mesh-native units (see kmToNativeScale); disk_ang in radians.
inline bool primaryDiskOverlap(const std::array<double,3>& ray_origin,
                               const std::array<double,3>& sun_dir,
                               const std::array<double,3>& primary_pos_n,
                               double primary_radius_n,
                               double sun_dist_n,
                               double disk_ang)
{
    const std::array<double,3> pc = { primary_pos_n[0] - ray_origin[0],
                                      primary_pos_n[1] - ray_origin[1],
                                      primary_pos_n[2] - ray_origin[2] };
    const double d = std::sqrt(pc[0]*pc[0] + pc[1]*pc[1] + pc[2]*pc[2]);
    if (d <= primary_radius_n) return true;      // inside the primary (degenerate)
    if (d >= sun_dist_n)       return false;     // primary is farther than the Sun
    const double along = pc[0]*sun_dir[0] + pc[1]*sun_dir[1] + pc[2]*sun_dir[2];
    if (along <= 0.0)          return false;     // primary is behind the face
    double cs = along / d;
    if (cs >  1.0) cs =  1.0;
    if (cs < -1.0) cs = -1.0;
    const double sep   = std::acos(cs);                        // Sun-center vs primary-center
    const double ang_p = std::asin(primary_radius_n / d);      // primary angular radius
    return sep < (ang_p + disk_ang);
}
} // anonymous namespace

namespace {
inline double solarDiskHorizonVisibility(double altitude_rad, double disk_rad) {
    if (disk_rad <= 0.0) return 0.0;
    if (altitude_rad >= disk_rad) return 1.0;
    if (altitude_rad <= -disk_rad) return 0.0;
    const double r = disk_rad;
    const double r2 = r*r;
    if (altitude_rad >= 0.0) {
        double y = altitude_rad;
        double A_cut = r2 * std::acos(y / r) - y * std::sqrt(std::max(0.0, r2 - y*y));
        return 1.0 - A_cut / (M_PI * r2);
    } else {
        double y = -altitude_rad;
        double A_vis = r2 * std::acos(y / r) - y * std::sqrt(std::max(0.0, r2 - y*y));
        return A_vis / (M_PI * r2);
    }
}
// Limb attenuation factor (empirical smooth projection correction)
inline double limbProjectionFactor(double altitude_rad, double disk_rad) {
    if (altitude_rad >= disk_rad) return 1.0;
    if (altitude_rad <= -disk_rad) return 0.0;
    double x = (altitude_rad + disk_rad) / (2.0 * disk_rad); // 0→1 across partial phase
    // Smooth cubic easing (better than linear near edges)
    return std::clamp(x*x*(3.0 - 2.0*x), 0.0, 1.0);
}

inline uint32_t mix32(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352dU;
    x ^= x >> 15; x *= 0x846ca68bU;
    x ^= x >> 16; return x;
}
inline uint32_t hashFaceSeed(const std::array<double,3>& c, const std::array<double,3>& n,
                             const std::array<double,3>& sun_pos) {
    auto q = [&](double v)->uint32_t {
        int64_t i = static_cast<int64_t>(std::llround(v * 1e-3)); // mm→m bucket
        return mix32(static_cast<uint32_t>(i) ^ static_cast<uint32_t>(i>>32));
    };
    uint32_t h = 2166136261u;
    h = mix32(h ^ q(c[0])); h = mix32(h ^ q(c[1])); h = mix32(h ^ q(c[2]));
    h = mix32(h ^ q(n[0])); h = mix32(h ^ q(n[1])); h = mix32(h ^ q(n[2]));
    h = mix32(h ^ q(sun_pos[0])); h = mix32(h ^ q(sun_pos[1])); h = mix32(h ^ q(sun_pos[2]));
    return h;
}
// The QMC disk sampler now lives in primary_disk.h so the solar-disk path and
// the PTM-44 primary-disk path share ONE definition (and so the standalone
// test_primary_disk gate can exercise the same arithmetic). Definitions moved
// verbatim; every call site below is unchanged.
using ptm::disk::vanDerCorput;
using ptm::disk::hammersley;
using ptm::disk::squareToConcentricDisk;
inline void rotateBasis(const std::array<double,3>& u, const std::array<double,3>& v,
                        double ang, std::array<double,3>& ur, std::array<double,3>& vr) {
    double c = std::cos(ang), s = std::sin(ang);
    ur = { c*u[0] - s*v[0], c*u[1] - s*v[1], c*u[2] - s*v[2] };
    vr = { s*u[0] + c*v[0], s*u[1] + c*v[1], s*u[2] + c*v[2] };
}
} // anonymous namespace

namespace ptm {

// Thread-local storage definitions
thread_local std::vector<std::array<double, 3>> IlluminationModel::tls_ray_origins;
thread_local std::vector<std::array<double, 3>> IlluminationModel::tls_ray_directions;
thread_local std::vector<int> IlluminationModel::tls_ray_results;
thread_local size_t IlluminationModel::tls_batch_size = 0;

thread_local IlluminationModel::RayBatch IlluminationModel::tls_batch;

// Error handling function for Embree
namespace {
inline void embreeErrorFunction(void* userPtr, RTCError error, const char* str) {
    std::cerr << "[Embree] Error " << int(error) << ": " << (str ? str : "(null)") << std::endl;
}
}

IlluminationModel::IlluminationModel(bool debug_output) 
    : embree_device_(nullptr), embree_scene_(nullptr), debug_output_enabled_(debug_output), show_progress_(true) {
    // Default initialization
    mesh_resolution_ = 32;
    time_steps_ = 24;
    albedo_ = 0.5;
    
    // Get kernels path from config
    kernels_path_ = ptm_thermal::config::SPICE_KERNELS_DIR;

    // NOTE: Do NOT initialize SPICE here. At construction the target body is not
    // yet known, so meta_kernel_filename_ is still the "europa.tm" default. Eagerly
    // furnsh'ing it (a) loads the wrong metakernel for every non-Europa body, and
    // (b) — because a partial/failed load latches a SPICE error and the one-shot
    // `spice_initialized` guard — permanently blocks the correct per-body metakernel
    // from ever loading, silently producing a garbage epoch (BUGS.md PTM-27).
    // SPICE is initialized lazily/explicitly AFTER setPlanetaryBody()/setGenericBody()
    // has set the correct metakernel (see initialize_spice() callers).
}

IlluminationModel::~IlluminationModel() {
    cleanupEmbree();
    cleanupSpice();
}

void IlluminationModel::setMeshResolution(int resolution) {
    mesh_resolution_ = resolution;
}

void IlluminationModel::setTimeSteps(int steps) {
    time_steps_ = steps;
}

void IlluminationModel::setExternalTimeGrid(const std::vector<double>& ets_tdb) {
    if (precomputed_positions_loaded_) {
        std::cerr << "[IlluminationModel] Warning: setExternalTimeGrid ignored because "
                     "a precomputed positions file is already loaded." << std::endl;
        return;
    }
    external_ets_ = ets_tdb;
    std::cout << "[IlluminationModel] External ET grid set: " << external_ets_.size()
              << " samples";
    if (!external_ets_.empty()) {
        const double span_s = external_ets_.back() - external_ets_.front();
        std::cout << " spanning " << span_s << " s (" << span_s / 86400.0 << " days)";
    }
    std::cout << std::endl;
}

void IlluminationModel::clearExternalTimeGrid() {
    external_ets_.clear();
}

void IlluminationModel::setAlbedo(double albedo) {
    albedo_ = albedo;
}

// Embree error callback definition
static void EmbreeErrorCallback(void* /*userPtr*/, RTCError error, const char* str) {
    std::cerr << "[Embree] Error " << int(error) << ": " << (str ? str : "(null)") << std::endl;
}

bool IlluminationModel::initializeEmbree() {
    if (!embree_device_) {
        embree_device_ = rtcNewDevice("");
        if (!embree_device_) {
            std::cerr << "Failed to create Embree device" << std::endl;
            return false;
        }
        rtcSetDeviceErrorFunction(embree_device_, EmbreeErrorCallback, nullptr);
    }
    return true;
}

void IlluminationModel::cleanupEmbree() {
    // Release Embree resources
    if (embree_scene_) {
        rtcReleaseScene(embree_scene_);
        embree_scene_ = nullptr;
    }
    
    if (embree_device_) {
        rtcReleaseDevice(embree_device_);
        embree_device_ = nullptr;
    }
}

bool IlluminationModel::initializeScene(
    const std::vector<std::array<double, 3>>& vertices,
    const std::vector<std::array<int, 3>>& faces)
{
    try {
        if (vertices.empty() || faces.empty()) {
            std::cerr << "[IlluminationModel] initializeScene: empty mesh" << std::endl;
            return false;
        }

        // Occluder-only geometry: append the ring/annulus triangles to the
        // ray-tracing BVH so distant topography shadows the core. They are
        // shadow-only (boolean rtcOccluded path returns no face id), never
        // thermally solved / radiosity-coupled / output. When there are no
        // occluders, Vv/Ff alias the inputs verbatim → byte-identical to
        // the pre-occluder behavior. See docs/extended-occluder-architecture.md
        // PTM-18: interior-hole seal gate (env; also settable via setSealInteriorHoles). Compute the
        // fill BEFORE the merge so the fills (CORE-vertex indices) append after core [+ occluder ring]
        // in primID space. Default OFF -> no fill -> byte-identical (below preserves the input alias).
        if (const char* sh = std::getenv("PTM_ILLUM_SEAL_HOLES"))
            if (sh[0]=='1'||sh[0]=='t'||sh[0]=='T'||sh[0]=='y'||sh[0]=='Y') seal_interior_holes_ = true;
        std::vector<std::array<int, 3>> seal_faces;
        if (seal_interior_holes_)
            seal_faces = buildInteriorHoleFill(vertices, faces);

        std::vector<std::array<double, 3>> verts_merged;
        std::vector<std::array<int, 3>>    faces_merged;
        const std::vector<std::array<double, 3>>* Vsrc = &vertices;
        const std::vector<std::array<int, 3>>*    Fsrc = &faces;
        const bool have_occ = (!occluder_faces_.empty() && !occluder_vertices_.empty());
        if (have_occ || !seal_faces.empty()) {
            verts_merged = vertices;
            faces_merged = faces;
            faces_merged.reserve(faces.size()
                                 + (have_occ ? occluder_faces_.size() : 0)
                                 + seal_faces.size());
            if (have_occ) {
                const int voff = static_cast<int>(vertices.size());
                verts_merged.insert(verts_merged.end(),
                                    occluder_vertices_.begin(),
                                    occluder_vertices_.end());
                for (const auto& f : occluder_faces_)
                    faces_merged.push_back({ f[0] + voff, f[1] + voff, f[2] + voff });
                std::cout << "[IlluminationModel] + " << occluder_faces_.size()
                          << " occluder triangles (shadow-only; core faces="
                          << faces.size() << ")\n";
            }
            // Interior-hole fills reference CORE vertices (indices < vertices.size()); append verbatim
            // as shadow-only occluders after the core [+ ring].
            for (const auto& f : seal_faces) faces_merged.push_back(f);
            Vsrc = &verts_merged;
            Fsrc = &faces_merged;
        }
        const std::vector<std::array<double, 3>>& Vv = *Vsrc;
        const std::vector<std::array<int, 3>>&    Ff = *Fsrc;

        if (!embree_device_) {
            embree_device_ = rtcNewDevice("");
            if (!embree_device_) {
                std::cerr << "[IlluminationModel] Failed to create Embree device" << std::endl;
                return false;
            }
            // FIX: use the canonical callback with exact signature
            rtcSetDeviceErrorFunction(embree_device_, EmbreeErrorCallback, nullptr);
        }

        if (embree_scene_) {
            rtcReleaseScene(embree_scene_);
            embree_scene_ = nullptr;
        }

        embree_scene_ = rtcNewScene(embree_device_);
        if (!embree_scene_) {
            std::cerr << "[IlluminationModel] Failed to create Embree scene" << std::endl;
            return false;
        }

        rtcSetSceneFlags(embree_scene_, RTC_SCENE_FLAG_ROBUST | RTC_SCENE_FLAG_COMPACT);
        rtcSetSceneBuildQuality(embree_scene_, RTC_BUILD_QUALITY_HIGH);

        // FIX: avoid infinity initializers (compatible with -ffinite-math-only)
        std::array<double,3> bb_min = { std::numeric_limits<double>::max(),
                                        std::numeric_limits<double>::max(),
                                        std::numeric_limits<double>::max() };
        std::array<double,3> bb_max = { -std::numeric_limits<double>::max(),
                                        -std::numeric_limits<double>::max(),
                                        -std::numeric_limits<double>::max() };

        for (const auto& v : Vv) {
            bb_min[0] = std::min(bb_min[0], v[0]); bb_max[0] = std::max(bb_max[0], v[0]);
            bb_min[1] = std::min(bb_min[1], v[1]); bb_max[1] = std::max(bb_max[1], v[1]);
            bb_min[2] = std::min(bb_min[2], v[2]); bb_max[2] = std::max(bb_max[2], v[2]);
        }

        scene_origin_offset_ = {
            0.5 * (bb_min[0] + bb_max[0]),
            0.5 * (bb_min[1] + bb_max[1]),
            0.5 * (bb_min[2] + bb_max[2])
        };
        scene_has_offset_ = true;

        // Closed-vs-open mesh detection (topology, not shape). Count boundary
        // edges: an undirected edge shared by exactly one triangle is a boundary
        // edge. 0 boundary edges => watertight/closed; any => open patch. Uses the
        // CORE faces only (an occluder ring is legitimately open and must NOT flip
        // the body pre-check on for a closed core). O(edges) with a hash set.
        // Gates the analytic body-occlusion pre-check on the CPU AND GPU paths;
        // see the mesh_is_open_ note in the header and memory
        // ptm-mu-gate-open-mesh-selfocclusion-fail.
        {
            const auto t_topo0 = std::chrono::steady_clock::now();
            // Map each undirected edge (min,max vertex id) -> incidence count.
            std::unordered_map<uint64_t, uint32_t> edge_count;
            edge_count.reserve(faces.size() * 3);
            auto edge_key = [](int a, int b) -> uint64_t {
                uint32_t lo = static_cast<uint32_t>(a < b ? a : b);
                uint32_t hi = static_cast<uint32_t>(a < b ? b : a);
                return (static_cast<uint64_t>(lo) << 32) | static_cast<uint64_t>(hi);
            };
            for (const auto& f : faces) {
                ++edge_count[edge_key(f[0], f[1])];
                ++edge_count[edge_key(f[1], f[2])];
                ++edge_count[edge_key(f[2], f[0])];
            }
            size_t boundary_edges = 0;
            for (const auto& kv : edge_count) {
                if (kv.second == 1) ++boundary_edges;
            }
            mesh_is_open_ = (boundary_edges > 0);

            // Body-level occluding-sphere radius for the OPEN-mesh pre-check:
            // min over core-face centroid radii (native units, matching runtime
            // |fc|). See the body_radius_occ_native_ note in the header. Only
            // meaningful/used when mesh_is_open_; harmless to compute for closed.
            double r_min = std::numeric_limits<double>::max();
            for (const auto& f : faces) {
                const auto& a = vertices[f[0]];
                const auto& b = vertices[f[1]];
                const auto& c = vertices[f[2]];
                const double cx = (a[0] + b[0] + c[0]) / 3.0;
                const double cy = (a[1] + b[1] + c[1]) / 3.0;
                const double cz = (a[2] + b[2] + c[2]) / 3.0;
                const double rc = std::sqrt(cx*cx + cy*cy + cz*cz);
                if (rc < r_min) r_min = rc;
            }
            body_radius_occ_native_ = (r_min == std::numeric_limits<double>::max())
                                          ? 0.0 : r_min;

            const auto t_topo1 = std::chrono::steady_clock::now();
            const double topo_ms =
                std::chrono::duration<double, std::milli>(t_topo1 - t_topo0).count();
            std::cout << "[IlluminationModel] mesh topology: "
                      << (mesh_is_open_ ? "OPEN patch" : "CLOSED (watertight)")
                      << " (" << boundary_edges << " boundary edges / "
                      << edge_count.size() << " unique edges, "
                      << faces.size() << " core faces; detect "
                      << topo_ms << " ms). Analytic body-occlusion pre-check "
                      << (mesh_is_open_ ? "ENABLED (open: sole self-occlusion guard)."
                                        : "DISABLED (closed: Embree self-occludes).")
                      << " R_occ(body) = " << body_radius_occ_native_
                      << " (native units).\n";
        }

        // PTM-13: build the local-horizon (skyline) field for the grazing DIRECT-sun shadow
        // cull IF requested. Default OFF -> byte-identical (no build, no cull). Env knobs:
        // PTM_ILLUM_HORIZON_CULL=1 enable; PTM_ILLUM_HORIZON_SLACK_DEG / _CELL_SCALE tune it.
        // The cull is azimuth-specific and resolution-independent (see horizon_field.h); it is
        // the physics fix for the mm-relief PSR leak that the single Embree shadow ray hops.
        {
            if (const char* he = std::getenv("PTM_ILLUM_HORIZON_CULL"))
                if (he[0]=='1'||he[0]=='t'||he[0]=='T'||he[0]=='y'||he[0]=='Y') use_horizon_cull_ = true;
            if (const char* se = std::getenv("PTM_ILLUM_HORIZON_SLACK_DEG")) horizon_slack_deg_ = std::atof(se);
            if (const char* ce = std::getenv("PTM_ILLUM_HORIZON_CELL_SCALE")) horizon_cell_scale_ = std::atof(ce);
        }
        if (use_horizon_cull_) {
            const size_t Fh = Ff.size();
            // Local up = outward radial (direction of the selenocentric bbox center for a
            // recentered planetary patch). A SINGLE global up is valid only for a small patch
            // (curvature negligible) -- gated on angular extent below; warned + disabled otherwise.
            std::array<double,3> up = scene_has_offset_ ? scene_origin_offset_
                                                        : std::array<double,3>{{0.0,0.0,1.0}};
            { double L = std::sqrt(up[0]*up[0]+up[1]*up[1]+up[2]*up[2]); if (L>0){up[0]/=L;up[1]/=L;up[2]/=L;} }
            const double body_r = scene_has_offset_
                ? std::sqrt(scene_origin_offset_[0]*scene_origin_offset_[0]
                          + scene_origin_offset_[1]*scene_origin_offset_[1]
                          + scene_origin_offset_[2]*scene_origin_offset_[2]) : 0.0;
            std::vector<std::array<double,3>> fcs(Fh);
            std::vector<double> pitches(Fh, 0.0);
            std::array<double,3> bmin{{1e300,1e300,1e300}}, bmax{{-1e300,-1e300,-1e300}};
            for (size_t f = 0; f < Fh; ++f) {
                const auto& a = Vv[Ff[f][0]]; const auto& b = Vv[Ff[f][1]]; const auto& c = Vv[Ff[f][2]];
                std::array<double,3> ctr{{(a[0]+b[0]+c[0])/3.0,(a[1]+b[1]+c[1])/3.0,(a[2]+b[2]+c[2])/3.0}};
                if (scene_has_offset_) { ctr[0]-=scene_origin_offset_[0]; ctr[1]-=scene_origin_offset_[1]; ctr[2]-=scene_origin_offset_[2]; }
                fcs[f] = ctr;
                for (int k=0;k<3;++k){ bmin[k]=std::min(bmin[k],ctr[k]); bmax[k]=std::max(bmax[k],ctr[k]); }
                const double dx=a[0]-b[0], dy=a[1]-b[1], dz=a[2]-b[2];
                pitches[f] = std::sqrt(dx*dx+dy*dy+dz*dz);
            }
            const double extent = std::max({bmax[0]-bmin[0], bmax[1]-bmin[1], bmax[2]-bmin[2]});
            std::nth_element(pitches.begin(), pitches.begin()+Fh/2, pitches.end());
            const double med_pitch = (Fh>0 && pitches[Fh/2]>0.0) ? pitches[Fh/2] : 1.0;
            const double cell = horizon_cell_scale_ * med_pitch;
            const double gate = (body_r>0.0) ? body_r*(0.25*M_PI/180.0) : 1e300;  // 0.25 deg of arc
            if (body_r>0.0 && extent>gate) {
                std::cout << "[IlluminationModel] PTM-13 horizon cull DISABLED: mesh extent " << extent
                          << " m exceeds the single-up validity limit " << gate << " m (0.25 deg arc at "
                          << "body radius " << body_r << " m); curvature would invalidate the global-up "
                          << "skyline. Keeping the Embree shadow test.\n";
                use_horizon_cull_ = false;
            } else {
                horizon_direct_.setSlackDeg(horizon_slack_deg_);
                horizon_direct_.build(fcs, up, cell);
                std::cout << "[IlluminationModel] PTM-13 local-horizon direct-shadow cull ENABLED (cell="
                          << cell*1000.0 << " mm, slack=" << horizon_slack_deg_ << " deg, " << Fh
                          << " faces, azimuth-specific skyline; resolution-independent).\n";
                if (use_gpu_ray_tracer_)
                    std::cout << "[IlluminationModel] WARNING: the horizon cull applies on the CPU "
                                 "illumination path only; GPU ray tracing is active, so the cull will "
                                 "NOT engage. Set illumination.use_gpu:false to apply it (or add the GPU mirror).\n";
            }
        }

        // PTM-13 P1: build the topological self-skip adjacency (source + N-ring edge neighbors) for
        // the grazing DIRECT-sun shadow ray IF requested. Default OFF -> byte-identical (no build, no
        // re-trace). Env knobs: PTM_ILLUM_TOPO_SELFSKIP=1 enable; PTM_ILLUM_SELFSKIP_RINGS=<n> depth.
        // When ON, the CPU direct-solar occlusion starts the ray AT the face (eps_n=eps_d=0, tnear=0)
        // and skips the source + its <= N edge rings by primID, so the ~mm-scale local rim occluder
        // (present in the mesh) is detected instead of hopped by the 1 cm offset. See BUGS.md PTM-13.
        {
            if (const char* se = std::getenv("PTM_ILLUM_TOPO_SELFSKIP"))
                if (se[0]=='1'||se[0]=='t'||se[0]=='T'||se[0]=='y'||se[0]=='Y') use_topo_selfskip_ = true;
            if (const char* re = std::getenv("PTM_ILLUM_SELFSKIP_RINGS")) {
                int rr = std::atoi(re); if (rr >= 1) selfskip_rings_ = rr;
            }
        }
        if (use_topo_selfskip_) {
            const auto t_ss0 = std::chrono::steady_clock::now();
            buildSelfSkipAdjacency(Ff, faces.size());
            const auto t_ss1 = std::chrono::steady_clock::now();
            const size_t ncore = faces.size();
            const double mean_nbr = (ncore > 0)
                ? static_cast<double>(face_skip_nbr_.size()) / static_cast<double>(ncore) : 0.0;
            std::cout << "[IlluminationModel] PTM-13 topological self-skip ENABLED (rings="
                      << selfskip_rings_ << ", " << ncore << " core faces, mean " << mean_nbr
                      << " skip-ids/face incl self; ray starts AT face, eps_n=eps_d=0, tnear=0; build "
                      << std::chrono::duration<double, std::milli>(t_ss1 - t_ss0).count()
                      << " ms). The mm-scale rim occluder is detected topologically instead of hopped "
                         "by the 1 cm offset.\n";
            if (use_gpu_ray_tracer_)
                std::cout << "[IlluminationModel] WARNING: topological self-skip applies on the CPU "
                             "illumination path only; GPU ray tracing is active, so it will NOT engage. "
                             "Set illumination.use_gpu:false to apply it (or add the GPU mirror). The "
                             "GPU batch shadow origin additionally uses eps_d=8x "
                             "(see docs/KNOWN_LIMITATIONS.md, PTM-13).\n";
        }

        // PTM-13 rise-epsilon: build the per-core-face facet scale for the out-of-plane-rise
        // self-relief discriminator IF requested. Default OFF -> byte-identical (no build, no
        // re-trace). Env knobs: PTM_ILLUM_RISE_EPS=1 enable; PTM_ILLUM_RISE_KEPS=<k> the facet
        // fraction (h_eps = k * local edge; default 0.2 -> ~0.5 mm at a 2.5 mm facet). Independent
        // of the self-skip flag: it separates self-relief from a real occluder by out-of-plane RISE
        // (self-relief ~0.12 mm vs the ~2.47 mm rim the 1 cm offset hops), not by distance or ring.
        {
            if (const char* re = std::getenv("PTM_ILLUM_RISE_EPS"))
                if (re[0]=='1'||re[0]=='t'||re[0]=='T'||re[0]=='y'||re[0]=='Y') use_rise_eps_ = true;
            if (const char* ke = std::getenv("PTM_ILLUM_RISE_KEPS")) {
                double kk = std::atof(ke); if (kk > 0.0) rise_keps_ = kk;
            }
            // PTM-13 OPTION 3 multi-hit refinement (sub-mode). RISE_EPS2 implies RISE_EPS so the base
            // build + CPU routing engage; the OR layer lives inside traceOccludedRiseEps.
            if (const char* r2 = std::getenv("PTM_ILLUM_RISE_EPS2"))
                if (r2[0]=='1'||r2[0]=='t'||r2[0]=='T'||r2[0]=='y'||r2[0]=='Y') { use_rise_eps2_ = true; use_rise_eps_ = true; }
            if (const char* be = std::getenv("PTM_ILLUM_RISE_BARY_EPS")) {
                double bb = std::atof(be); if (bb >= 0.0) rise_bary_eps_ = bb;
            }
            // PTM-13 Class-2 fix: absolute self-relief floor override (meters). >=0 replaces the
            // edge-proportional h_eps with a fixed floor (0 => exact source-triangle-only exclusion).
            // Default unset -> rise_habs_ stays -1.0 -> byte-identical (edge-proportional floor).
            if (const char* ha = std::getenv("PTM_ILLUM_RISE_HABS")) {
                double hh = std::atof(ha); if (hh >= 0.0) rise_habs_ = hh;
            }
        }
        if (use_rise_eps_) {
            const size_t ncore = faces.size();
            face_edge_local_.assign(ncore, 0.0f);
            auto elen = [](const std::array<double,3>& p, const std::array<double,3>& q) {
                const double dx=p[0]-q[0], dy=p[1]-q[1], dz=p[2]-q[2];
                return std::sqrt(dx*dx+dy*dy+dz*dz);
            };
            double sum_e = 0.0;
            for (size_t c = 0; c < ncore; ++c) {
                const auto& a = Vv[Ff[c][0]]; const auto& b = Vv[Ff[c][1]]; const auto& d = Vv[Ff[c][2]];
                const double em = (elen(a,b) + elen(b,d) + elen(d,a)) / 3.0;
                face_edge_local_[c] = static_cast<float>(em);
                sum_e += em;
            }
            const double mean_e = (ncore > 0) ? sum_e / static_cast<double>(ncore) : 0.0;
            std::cout << "[IlluminationModel] PTM-13 rise-epsilon self-relief discriminator ENABLED (k="
                      << rise_keps_ << ", mean facet edge " << mean_e*1000.0 << " mm -> mean h_eps "
                      << rise_keps_*mean_e*1000.0 << " mm, " << ncore << " core faces; ray marches from "
                         "the centroid, a hit counts as a real occluder iff it rises >= h_eps out of the "
                         "source face plane). Separates ~0.12 mm self-relief from the ~2.47 mm rim the "
                         "1 cm offset hops.\n";
            if (rise_habs_ >= 0.0) {
                std::cout << "[IlluminationModel] PTM-13 Class-2 fix: ABSOLUTE self-relief floor "
                             "OVERRIDE h_eps = " << rise_habs_*1000.0 << " mm (env PTM_ILLUM_RISE_HABS), "
                             "replacing the edge-proportional " << rise_keps_ << "*edge (~"
                          << rise_keps_*mean_e*1000.0 << " mm). h_abs=0 => exact source-triangle-only "
                             "self-exclusion (resolved relief shadows; Paul's scale ruling). Phase-2 "
                             "knob table: 0.2*edge kills 0.9% of grazing occlusions, h_abs=0 kills 100% "
                             "with 0 acne.\n";
            }
            if (use_rise_eps2_) {
                // OPTION 3: store the recentered double vertices of every core face so the OR layer can
                // recompute a float64-interior test + the hit-triangle max-vertex rise per hit.
                face_verts_local_.assign(ncore, {{{{0,0,0}},{{0,0,0}},{{0,0,0}}}});
                for (size_t c = 0; c < ncore; ++c) {
                    for (int k = 0; k < 3; ++k) {
                        const auto& vv = Vv[Ff[c][k]];
                        face_verts_local_[c][k] = {{ vv[0] - scene_origin_offset_[0],
                                                     vv[1] - scene_origin_offset_[1],
                                                     vv[2] - scene_origin_offset_[2] }};
                    }
                }
                std::cout << "[IlluminationModel] PTM-13 rise-epsilon MULTI-HIT refinement (OPTION 3) "
                             "ENABLED (bary_eps=" << rise_bary_eps_ << "): a hit ALSO blocks iff it is a "
                             "float64-INTERIOR crossing whose TRIANGLE max-vertex rise >= h_eps (catches "
                             "the near-coplanar occluder crossed near its shared edge; STEP-1 recall "
                             "24%->84%, ~0 incremental acne). Strict superset of the base rise-eps blocks.\n";
            }
            if (use_topo_selfskip_)
                std::cout << "[IlluminationModel] NOTE: both rise-epsilon and topological self-skip are "
                             "enabled; rise-epsilon takes precedence on the CPU direct-solar path.\n";
            if (use_gpu_ray_tracer_)
                std::cout << "[IlluminationModel] WARNING: rise-epsilon applies on the CPU illumination "
                             "path only; GPU ray tracing is active, so it will NOT engage. Set "
                             "illumination.use_gpu:false to apply it (or add the GPU mirror).\n";
        }

        RTCGeometry geom = rtcNewGeometry(embree_device_, RTC_GEOMETRY_TYPE_TRIANGLE);
        if (!geom) {
            std::cerr << "[IlluminationModel] Failed to create Embree geometry" << std::endl;
            return false;
        }
        rtcSetGeometryBuildQuality(geom, RTC_BUILD_QUALITY_HIGH);
        rtcSetGeometryMask(geom, 0xFFFFFFFFu);

        const size_t V = Vv.size();
        const size_t F = Ff.size();

        float* vbuf = static_cast<float*>(rtcSetNewGeometryBuffer(
            geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3,
            sizeof(float) * 3, V));
        if (!vbuf) {
            std::cerr << "[IlluminationModel] Failed to allocate vertex buffer" << std::endl;
            rtcReleaseGeometry(geom);
            return false;
        }
        for (size_t i = 0; i < V; ++i) {
            vbuf[i*3 + 0] = static_cast<float>(Vv[i][0] - scene_origin_offset_[0]);
            vbuf[i*3 + 1] = static_cast<float>(Vv[i][1] - scene_origin_offset_[1]);
            vbuf[i*3 + 2] = static_cast<float>(Vv[i][2] - scene_origin_offset_[2]);
        }

        uint32_t* ibuf = static_cast<uint32_t*>(rtcSetNewGeometryBuffer(
            geom, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3,
            sizeof(uint32_t) * 3, F));
        if (!ibuf) {
            std::cerr << "[IlluminationModel] Failed to allocate index buffer" << std::endl;
            rtcReleaseGeometry(geom);
            return false;
        }
        for (size_t i = 0; i < F; ++i) {
            ibuf[i*3 + 0] = static_cast<uint32_t>(Ff[i][0]);
            ibuf[i*3 + 1] = static_cast<uint32_t>(Ff[i][1]);
            ibuf[i*3 + 2] = static_cast<uint32_t>(Ff[i][2]);
        }

        rtcCommitGeometry(geom);
        rtcAttachGeometry(embree_scene_, geom);
        rtcReleaseGeometry(geom);

        rtcCommitScene(embree_scene_);

        std::cout << "[IlluminationModel] Embree scene ready. V=" << V
                  << " F=" << F
                  << " origin shift=("
                  << scene_origin_offset_[0] << ", "
                  << scene_origin_offset_[1] << ", "
                  << scene_origin_offset_[2] << ")\n";

#ifdef PTM_GPU_OPENCL
        // Build and upload BVH to GPU if GPU ray tracing is enabled
        if (use_gpu_ray_tracer_ && gpu_ray_pool_ && gpu_ray_pool_->isReady()) {
            try {
                auto bvh_start = std::chrono::high_resolution_clock::now();

                // Build BVH with SAH (CPU-side, replicated across all pool devices)
                gpu_bvh_ = std::make_unique<gpu::opencl::BVHBuilder>();
                gpu::opencl::BVHConfig bvh_config;
                bvh_config.max_triangles_per_leaf = 4;
                bvh_config.use_sah = true;
                bvh_config.verbose = false;

                // Convert vertices to local coordinates (same as Embree)
                std::vector<std::array<float, 3>> local_vertices(V);
                for (size_t i = 0; i < V; ++i) {
                    local_vertices[i] = {{
                        static_cast<float>(Vv[i][0] - scene_origin_offset_[0]),
                        static_cast<float>(Vv[i][1] - scene_origin_offset_[1]),
                        static_cast<float>(Vv[i][2] - scene_origin_offset_[2])
                    }};
                }

                // Convert indices
                std::vector<std::array<uint32_t, 3>> indices(F);
                for (size_t i = 0; i < F; ++i) {
                    indices[i] = {{
                        static_cast<uint32_t>(Ff[i][0]),
                        static_cast<uint32_t>(Ff[i][1]),
                        static_cast<uint32_t>(Ff[i][2])
                    }};
                }

                // Try to load BVH from disk cache first (saves the ~500 ms
                // SAH build on 5 M-face lunar meshes). If load fails or file
                // is missing, build from scratch and persist for next run.
                bool loaded_from_cache = false;
                if (!bvh_cache_file_.empty() && bvh_cache_file_ != "off") {
                    if (gpu_bvh_->read(bvh_cache_file_)) {
                        loaded_from_cache = (gpu_bvh_->numTriangles() == F);
                        if (!loaded_from_cache) {
                            std::cout << "[IlluminationModel] BVH cache "
                                      << bvh_cache_file_ << " mismatched mesh size ("
                                      << gpu_bvh_->numTriangles() << " vs " << F
                                      << "); rebuilding\n";
                            gpu_bvh_ = std::make_unique<gpu::opencl::BVHBuilder>();
                        }
                    }
                }
                if (!loaded_from_cache) {
                    gpu_bvh_->build(local_vertices, indices, bvh_config);
                    if (!bvh_cache_file_.empty() && bvh_cache_file_ != "off") {
                        if (gpu_bvh_->write(bvh_cache_file_)) {
                            std::cout << "[IlluminationModel] BVH cache written to "
                                      << bvh_cache_file_ << "\n";
                        }
                    }
                }

                auto bvh_end = std::chrono::high_resolution_clock::now();
                double bvh_time = std::chrono::duration<double>(bvh_end - bvh_start).count();

                std::cout << "[IlluminationModel] BVH "
                          << (loaded_from_cache ? "loaded from cache: " : "built: ")
                          << gpu_bvh_->numNodes()
                          << " nodes in " << std::fixed << std::setprecision(2)
                          << bvh_time << "s\n";

                // Upload mesh and BVH to every pool device.
                auto upload_start = std::chrono::high_resolution_clock::now();
                gpu_ray_pool_->uploadMesh(local_vertices, indices);
                gpu_ray_pool_->uploadBVH(*gpu_bvh_);
                gpu_mesh_uploaded_ = gpu_ray_pool_->isReady();
                auto upload_end = std::chrono::high_resolution_clock::now();
                double upload_time = std::chrono::duration<double>(upload_end - upload_start).count();

                std::cout << "[IlluminationModel] GPU mesh/BVH uploaded to "
                          << gpu_ray_pool_->numDevices() << " device(s) in "
                          << std::fixed << std::setprecision(2) << upload_time << "s\n";
            } catch (const std::exception& e) {
                std::cerr << "[IlluminationModel] GPU BVH setup failed: " << e.what()
                          << ", falling back to CPU\n";
                gpu_bvh_.reset();
                gpu_mesh_uploaded_ = false;
                use_gpu_ray_tracer_ = false;
            }
        }
#endif

        return true;
    } catch (const std::exception& e) {
        std::cerr << "[IlluminationModel] initializeScene exception: " << e.what() << std::endl;
        return false;
    } catch (...) {
        std::cerr << "[IlluminationModel] initializeScene unknown exception" << std::endl;
        return false;
    }
}

std::vector<int> IlluminationModel::intersectsFirst(
    const std::vector<std::array<double, 3>>& origins,
    const std::vector<std::array<double, 3>>& directions)
{
    const size_t n = std::min(origins.size(), directions.size());
    std::vector<int> face_ids(n, -1);
    if (!embree_scene_ || n == 0) return face_ids;

    auto estimate_tnear = [](const std::array<double,3>& org_local) -> float {
        const double r = std::sqrt(org_local[0]*org_local[0] +
                                   org_local[1]*org_local[1] +
                                   org_local[2]*org_local[2]);
        return static_cast<float>(std::max(1e-3, r * 1e-7));
    };

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        // Shift to Embree local space for robustness
        const std::array<double,3> org_local = {
            origins[i][0] - (scene_has_offset_ ? scene_origin_offset_[0] : 0.0),
            origins[i][1] - (scene_has_offset_ ? scene_origin_offset_[1] : 0.0),
            origins[i][2] - (scene_has_offset_ ? scene_origin_offset_[2] : 0.0)
        };

        // Normalize direction
        const double dl = std::sqrt(directions[i][0]*directions[i][0] +
                                    directions[i][1]*directions[i][1] +
                                    directions[i][2]*directions[i][2]);
        const double inv_dl = (dl > 0.0) ? (1.0 / dl) : 1.0;

        RTCRayHit rayhit{};
        rayhit.ray.org_x = static_cast<float>(org_local[0]);
        rayhit.ray.org_y = static_cast<float>(org_local[1]);
        rayhit.ray.org_z = static_cast<float>(org_local[2]);
        rayhit.ray.dir_x = static_cast<float>(directions[i][0] * inv_dl);
        rayhit.ray.dir_y = static_cast<float>(directions[i][1] * inv_dl);
        rayhit.ray.dir_z = static_cast<float>(directions[i][2] * inv_dl);

        rayhit.ray.tnear = estimate_tnear(org_local);
        rayhit.ray.tfar  = 1.0e30f;
        rayhit.ray.time  = 0.0f;
        rayhit.ray.mask  = 0xFFFFFFFFu;
        rayhit.ray.flags = 0;

        rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
        rayhit.hit.primID = RTC_INVALID_GEOMETRY_ID;

        RTCIntersectArguments args;
        rtcInitIntersectArguments(&args);
        args.flags = RTC_RAY_QUERY_FLAG_INCOHERENT;

        rtcIntersect1(embree_scene_, &rayhit, &args);

        if (rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID) {
            face_ids[i] = static_cast<int>(rayhit.hit.primID);
        }
    }

    return face_ids;
}

// SIMD version for improved performance when tracing multiple rays
std::vector<int> IlluminationModel::intersectsFirstSIMD(
    const std::vector<std::array<double, 3>>& origins,
    const std::vector<std::array<double, 3>>& directions)
{
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    const size_t n = std::min(origins.size(), directions.size());
    std::vector<int> face_ids(n, -1);
    if (!embree_scene_ || n == 0) return face_ids;

    auto estimate_tnear = [](float ox, float oy, float oz) -> float {
        const double r = std::sqrt(double(ox)*ox + double(oy)*oy + double(oz)*oz);
        return static_cast<float>(std::max(1e-3, r * 1e-7));
    };

    size_t processed = 0;
    while (processed < n) {
        const size_t packet_size = std::min<size_t>(8, n - processed);

        alignas(32) int valid[8];
        std::memset(valid, 0, sizeof(valid));

        alignas(32) RTCRayHit8 rayhit{};
        for (size_t lane = 0; lane < packet_size; ++lane) {
            valid[lane] = -1;
            const size_t idx = processed + lane;

            // Shift to Embree local space
            const float org_x = static_cast<float>(origins[idx][0] - (scene_has_offset_ ? scene_origin_offset_[0] : 0.0));
            const float org_y = static_cast<float>(origins[idx][1] - (scene_has_offset_ ? scene_origin_offset_[1] : 0.0));
            const float org_z = static_cast<float>(origins[idx][2] - (scene_has_offset_ ? scene_origin_offset_[2] : 0.0));

            // Normalize direction
            const double dl = std::sqrt(directions[idx][0]*directions[idx][0] +
                                        directions[idx][1]*directions[idx][1] +
                                        directions[idx][2]*directions[idx][2]);
            const double inv_dl = (dl > 0.0) ? (1.0 / dl) : 1.0;

            rayhit.ray.org_x[lane] = org_x;
            rayhit.ray.org_y[lane] = org_y;
            rayhit.ray.org_z[lane] = org_z;

            rayhit.ray.dir_x[lane] = static_cast<float>(directions[idx][0] * inv_dl);
            rayhit.ray.dir_y[lane] = static_cast<float>(directions[idx][1] * inv_dl);
            rayhit.ray.dir_z[lane] = static_cast<float>(directions[idx][2] * inv_dl);

            rayhit.ray.tnear[lane] = estimate_tnear(org_x, org_y, org_z);
            rayhit.ray.tfar[lane]  = 1.0e30f;
            rayhit.ray.mask[lane]  = 0xFFFFFFFFu;
            rayhit.ray.flags[lane] = 0;
            rayhit.ray.time[lane]  = 0.0f;

            rayhit.hit.geomID[lane] = RTC_INVALID_GEOMETRY_ID;
            rayhit.hit.primID[lane] = RTC_INVALID_GEOMETRY_ID;
            rayhit.hit.instID[0][lane] = RTC_INVALID_GEOMETRY_ID;
        }

        RTCIntersectArguments args;
        rtcInitIntersectArguments(&args);
        args.flags = RTC_RAY_QUERY_FLAG_INCOHERENT;

        rtcIntersect8(valid, embree_scene_, &rayhit, &args);

        for (size_t lane = 0; lane < packet_size; ++lane) {
            const size_t idx = processed + lane;
            if (rayhit.hit.geomID[lane] != RTC_INVALID_GEOMETRY_ID) {
                face_ids[idx] = static_cast<int>(rayhit.hit.primID[lane]);
            }
        }

        processed += packet_size;
    }

    return face_ids;
#else
    // Fallback to scalar implementation
    return intersectsFirst(origins, directions);
#endif
}

void IlluminationModel::initRayBatch() {
    // Clear previous results
    tls_ray_results.clear();
    tls_batch_size = 0;
    
    // Ensure vectors are preallocated for efficiency
    if (tls_ray_origins.size() < MAX_BATCH_SIZE) {
        tls_ray_origins.resize(MAX_BATCH_SIZE);
        tls_ray_directions.resize(MAX_BATCH_SIZE);
    }
}

// Initialize the batch collection system
void IlluminationModel::initRayBatchCollection() {
    // Clear thread-local batch
    tls_batch.origins.clear();
    tls_batch.directions.clear();
    tls_batch.results.clear();
    tls_batch.indices.clear();
    tls_batch.current_size = 0;
    tls_batch.processed = false;
    
    // Preallocate memory for efficiency
    tls_batch.origins.reserve(COLLECTION_BATCH_SIZE);
    tls_batch.directions.reserve(COLLECTION_BATCH_SIZE);
    tls_batch.indices.reserve(COLLECTION_BATCH_SIZE);
    
    // Clear shared batches under lock protection
    {
        std::lock_guard<std::mutex> lock(batch_mutex_);
        // Don't actually delete the RayBatch objects, just clear the pointers
        shared_batches_.clear();
        batches_being_processed_ = false;
    }
}

// Add a ray to the thread-local batch
size_t IlluminationModel::addRayToBatch(const std::array<double, 3>& origin, const std::array<double, 3>& direction) {
    // Add ray to thread-local batch
    size_t batch_index = tls_batch.current_size;
    
    tls_batch.origins.push_back(origin);
    tls_batch.directions.push_back(direction);
    tls_batch.current_size++;
    
    // Check if the thread-local batch is full
    if (tls_batch.current_size >= COLLECTION_BATCH_SIZE) {
        // Only submit batch if it's not already being processed
        if (!tls_batch.processed) {
            std::lock_guard<std::mutex> lock(batch_mutex_);
            
            // If not currently processing batches, add this batch
            if (!batches_being_processed_) {
                shared_batches_.push_back(&tls_batch);
                tls_batch.processed = true;
                
                // If we have enough batches or the batch is large enough,
                // start processing
                if (shared_batches_.size() >= 4 || tls_batch.current_size >= 64) {
                    processRayBatches();
                }
            }
        }
    }
    
    return batch_index;
}

// Process all collected ray batches in SIMD-friendly chunks
void IlluminationModel::processRayBatches() {
    std::vector<std::array<double, 3>> all_origins;
    std::vector<std::array<double, 3>> all_directions;
    std::vector<RayBatch*> batches_to_process;
    
    // Under lock: mark that we're processing and collect all batches
    {
        std::lock_guard<std::mutex> lock(batch_mutex_);
        
        if (shared_batches_.empty()) {
            return;
        }
        
        batches_being_processed_ = true;
        batches_to_process = shared_batches_;
        shared_batches_.clear();
    }
    
    // Calculate total rays
    size_t total_rays = 0;
    for (auto* batch : batches_to_process) {
        total_rays += batch->current_size;
    }
    
    // Allocate enough space for all rays
    all_origins.reserve(total_rays);
    all_directions.reserve(total_rays);
    
    // Combine all ray batches
    for (auto* batch : batches_to_process) {
        all_origins.insert(all_origins.end(), batch->origins.begin(), batch->origins.begin() + batch->current_size);
        all_directions.insert(all_directions.end(), batch->directions.begin(), batch->directions.begin() + batch->current_size);
    }
    
    // Process rays with optimal SIMD batching
    std::vector<int> all_results = traceRays(all_origins, all_directions);
    
    // Distribute results back to thread-local batches
    size_t result_offset = 0;
    for (auto* batch : batches_to_process) {
        batch->results.resize(batch->current_size);
        std::copy(all_results.begin() + result_offset, 
                  all_results.begin() + result_offset + batch->current_size, 
                  batch->results.begin());
        result_offset += batch->current_size;
        batch->processed = true;
    }
    
    // Mark processing as done
    {
        std::lock_guard<std::mutex> lock(batch_mutex_);
        batches_being_processed_ = false;
    }
}

// Get the hit result for a ray in the thread-local batch
bool IlluminationModel::getRayHitResult(size_t batch_index) {
    // Check if the batch has been processed
    if (!tls_batch.processed) {
        // Process the batch now if needed
        if (tls_batch.current_size > 0) {
            std::vector<std::array<double, 3>> origins(tls_batch.origins.begin(), 
                                                     tls_batch.origins.begin() + tls_batch.current_size);
            std::vector<std::array<double, 3>> directions(tls_batch.directions.begin(), 
                                                        tls_batch.directions.begin() + tls_batch.current_size);
            
            tls_batch.results = traceRays(origins, directions);
            tls_batch.processed = true;
        } else {
            return false;  // Empty batch
        }
    }
    
    // Check if the index is valid
    if (batch_index < tls_batch.results.size()) {
        return tls_batch.results[batch_index] >= 0;
    }
    
    return false;
}

void IlluminationModel::processRayBatch() {
    if (tls_batch_size == 0) return;
    
    // Trace all rays in the batch at once
    tls_ray_results = traceRays(
        std::vector<std::array<double, 3>>(tls_ray_origins.begin(), tls_ray_origins.begin() + tls_batch_size),
        std::vector<std::array<double, 3>>(tls_ray_directions.begin(), tls_ray_directions.begin() + tls_batch_size)
    );
    
    // Reset batch counter but keep results
    tls_batch_size = 0;
}

std::vector<uint8_t> IlluminationModel::traceOccluded(
    const std::vector<std::array<double, 3>>& origins,
    const std::vector<std::array<double, 3>>& directions,
    bool /*coherent*/)
{
    const size_t n = origins.size();
    std::vector<uint8_t> blocked(n, 0);
    if (n == 0) return blocked;

#ifdef PTM_GPU_OPENCL
    // Use GPU ray tracer if available
    if (use_gpu_ray_tracer_ && gpu_ray_pool_ && gpu_mesh_uploaded_) {
        return traceOccludedGPU(origins, directions);
    }
#endif

    // Fall back to CPU Embree
    if (!embree_scene_) return blocked;

    auto estimate_tnear = [&](const std::array<double,3>& org_local) -> float {
        const double r = std::sqrt(org_local[0]*org_local[0] +
                                   org_local[1]*org_local[1] +
                                   org_local[2]*org_local[2]);
        const double ulp = std::numeric_limits<double>::epsilon() * r;
        const double scaled = std::max(1e-6 * r, 0.01);  // 1 ppm of radius, min 1 cm
        return static_cast<float>(std::max(16.0 * ulp, scaled));
    };

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        // Shift to Embree local space (scene origin removal)
        const std::array<double,3> org_local_d = {
            origins[i][0] - (scene_has_offset_ ? scene_origin_offset_[0] : 0.0),
            origins[i][1] - (scene_has_offset_ ? scene_origin_offset_[1] : 0.0),
            origins[i][2] - (scene_has_offset_ ? scene_origin_offset_[2] : 0.0)
        };

        // Normalize direction
        const double dl = std::sqrt(directions[i][0]*directions[i][0] +
                                    directions[i][1]*directions[i][1] +
                                    directions[i][2]*directions[i][2]);
        const double inv_dl = (dl > 0.0) ? (1.0 / dl) : 1.0;

        RTCRay ray{};
        ray.org_x = static_cast<float>(org_local_d[0]);
        ray.org_y = static_cast<float>(org_local_d[1]);
        ray.org_z = static_cast<float>(org_local_d[2]);
        ray.dir_x = static_cast<float>(directions[i][0] * inv_dl);
        ray.dir_y = static_cast<float>(directions[i][1] * inv_dl);
        ray.dir_z = static_cast<float>(directions[i][2] * inv_dl);
        ray.tnear = estimate_tnear(org_local_d);
        ray.tfar  = 1.0e30f;
        ray.mask  = 0xFFFFFFFFu;
        ray.flags = 0;
        ray.time  = 0.0f;

        RTCOccludedArguments args;                 // CHANGE: use occluded args
        rtcInitOccludedArguments(&args);           // CHANGE: init occluded args
        args.flags = RTC_RAY_QUERY_FLAG_INCOHERENT;

        rtcOccluded1(embree_scene_, &ray, &args);  // CHANGE: pass RTCOccludedArguments*
        blocked[i] = static_cast<uint8_t>(ray.tfar < 0.0f ? 1 : 0);
    }
    return blocked;
}

// PTM-13 P1: build the per-core-face self-skip CSR. For each CORE face c in [0, n_core), collect
// c plus every face reachable within selfskip_rings_ edge-neighbor hops over the merged mesh
// (primID space), sorted, into face_skip_nbr_[face_skip_off_[c] .. face_skip_off_[c+1]). This lets
// the direct-solar occlusion re-trace skip self/near-neighbor hits by primID so it can start the
// ray AT the face and still detect the mm-scale local occluder. Called only when the flag is set.
void IlluminationModel::buildSelfSkipAdjacency(
    const std::vector<std::array<int, 3>>& faces_all, size_t n_core)
{
    face_skip_off_.clear();
    face_skip_nbr_.clear();
    const size_t F = faces_all.size();
    if (F == 0 || n_core == 0) return;

    // Undirected-edge -> (up to two) incident faces, over the merged mesh.
    auto ekey = [](int a, int b) -> uint64_t {
        uint32_t lo = static_cast<uint32_t>(a < b ? a : b);
        uint32_t hi = static_cast<uint32_t>(a < b ? b : a);
        return (static_cast<uint64_t>(lo) << 32) | static_cast<uint64_t>(hi);
    };
    std::unordered_map<uint64_t, std::pair<int,int>> edge_faces;
    edge_faces.reserve(F * 3);
    for (size_t f = 0; f < F; ++f) {
        const auto& t = faces_all[f];
        const int e[3][2] = { {t[0],t[1]}, {t[1],t[2]}, {t[2],t[0]} };
        for (int k = 0; k < 3; ++k) {
            const uint64_t key = ekey(e[k][0], e[k][1]);
            auto it = edge_faces.find(key);
            if (it == edge_faces.end()) edge_faces.emplace(key, std::make_pair(static_cast<int>(f), -1));
            else if (it->second.second < 0) it->second.second = static_cast<int>(f);
            // >2 faces on an edge (non-manifold): keep the first two; extra sharing is rare here.
        }
    }

    // 1-ring (edge-adjacent) neighbors for every face, as a fixed [F][3] table (-1 = boundary).
    std::vector<std::array<int,3>> nb1(F, {{-1,-1,-1}});
    for (size_t f = 0; f < F; ++f) {
        const auto& t = faces_all[f];
        const int e[3][2] = { {t[0],t[1]}, {t[1],t[2]}, {t[2],t[0]} };
        for (int k = 0; k < 3; ++k) {
            auto it = edge_faces.find(ekey(e[k][0], e[k][1]));
            int other = -1;
            if (it != edge_faces.end()) {
                if (it->second.first == static_cast<int>(f)) other = it->second.second;
                else if (it->second.second == static_cast<int>(f)) other = it->second.first;
            }
            nb1[f][k] = other;
        }
    }

    // BFS to selfskip_rings_ per core face, using a per-face generation stamp (no set allocation).
    face_skip_off_.assign(n_core + 1, 0);
    std::vector<int> stamp(F, -1), acc, frontier, nextf;
    for (size_t c = 0; c < n_core; ++c) {
        face_skip_off_[c] = static_cast<uint32_t>(face_skip_nbr_.size());
        acc.clear(); frontier.clear();
        const int gen = static_cast<int>(c);
        stamp[c] = gen; acc.push_back(gen); frontier.push_back(gen);
        for (int r = 0; r < selfskip_rings_; ++r) {
            nextf.clear();
            for (int f : frontier) {
                for (int k = 0; k < 3; ++k) {
                    const int nb = nb1[f][k];
                    if (nb >= 0 && stamp[nb] != gen) { stamp[nb] = gen; acc.push_back(nb); nextf.push_back(nb); }
                }
            }
            frontier.swap(nextf);
        }
        std::sort(acc.begin(), acc.end());
        for (int x : acc) face_skip_nbr_.push_back(x);
    }
    face_skip_off_[n_core] = static_cast<uint32_t>(face_skip_nbr_.size());
}

// PTM-13 P1: occlusion test with topological self-skip. Per ray, march via rtcIntersect1, skipping
// any hit whose primID is the source face or one of its <= selfskip_rings_ edge-neighbors, until the
// first NON-neighbor hit (=> blocked) or no hit (=> clear). The callers pass origins AT the face and
// zero geometric offset, so the ~mm local occluder present in the mesh is detected rather than hopped
// by the 1 cm offset. CPU Embree only; used when use_topo_selfskip_ is set.
std::vector<uint8_t> IlluminationModel::traceOccludedSelfSkip(
    const std::vector<std::array<double, 3>>& origins,
    const std::vector<std::array<double, 3>>& directions,
    const std::vector<int>& src_faces)
{
    const size_t n = origins.size();
    std::vector<uint8_t> blocked(n, 0);
    if (n == 0 || !embree_scene_) return blocked;

    auto in_skip = [&](int src, unsigned int prim) -> bool {
        if (src < 0) return false;
        if (static_cast<size_t>(src) + 1 >= face_skip_off_.size()) return static_cast<int>(prim) == src;
        const uint32_t b = face_skip_off_[static_cast<size_t>(src)];
        const uint32_t e = face_skip_off_[static_cast<size_t>(src) + 1];
        return std::binary_search(face_skip_nbr_.begin() + b, face_skip_nbr_.begin() + e,
                                  static_cast<int>(prim));
    };

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        // Shift into Embree-local (recentered) space, as traceOccluded does.
        const std::array<double,3> org_local = {
            origins[i][0] - (scene_has_offset_ ? scene_origin_offset_[0] : 0.0),
            origins[i][1] - (scene_has_offset_ ? scene_origin_offset_[1] : 0.0),
            origins[i][2] - (scene_has_offset_ ? scene_origin_offset_[2] : 0.0)
        };
        const double dl = std::sqrt(directions[i][0]*directions[i][0] +
                                    directions[i][1]*directions[i][1] +
                                    directions[i][2]*directions[i][2]);
        const double inv_dl = (dl > 0.0) ? (1.0 / dl) : 1.0;
        const float dx = static_cast<float>(directions[i][0] * inv_dl);
        const float dy = static_cast<float>(directions[i][1] * inv_dl);
        const float dz = static_cast<float>(directions[i][2] * inv_dl);
        const int src = (i < src_faces.size()) ? src_faces[i] : -1;

        float tnear = 0.0f;
        bool occ = false;
        // Bounded march: the skip set is small (~1 + 3*rings), so a few iterations suffice; the
        // cap guards degenerate/non-manifold fans. tnear strictly increases => no infinite loop.
        for (int iter = 0; iter < 64; ++iter) {
            RTCRayHit rh{};
            rh.ray.org_x = static_cast<float>(org_local[0]);
            rh.ray.org_y = static_cast<float>(org_local[1]);
            rh.ray.org_z = static_cast<float>(org_local[2]);
            rh.ray.dir_x = dx; rh.ray.dir_y = dy; rh.ray.dir_z = dz;
            rh.ray.tnear = tnear;
            rh.ray.tfar  = 1.0e30f;
            rh.ray.mask  = 0xFFFFFFFFu;
            rh.ray.flags = 0;
            rh.ray.time  = 0.0f;
            rh.hit.geomID = RTC_INVALID_GEOMETRY_ID;
            rh.hit.primID = RTC_INVALID_GEOMETRY_ID;

            RTCIntersectArguments args;
            rtcInitIntersectArguments(&args);
            args.flags = RTC_RAY_QUERY_FLAG_INCOHERENT;
            rtcIntersect1(embree_scene_, &rh, &args);

            if (rh.hit.geomID == RTC_INVALID_GEOMETRY_ID) { occ = false; break; }  // clear to the sun
            const unsigned int prim = rh.hit.primID;
            const float t_hit = rh.ray.tfar;  // Embree writes the hit distance into ray.tfar
            if (in_skip(src, prim)) {
                const float t_next = t_hit + std::max(t_hit * 1.0e-4f, 1.0e-6f);
                if (!(t_next > tnear)) { occ = false; break; }  // no forward progress -> treat as clear
                tnear = t_next;
                continue;
            }
            occ = true; break;  // first non-neighbor hit is a real occluder
        }
        blocked[i] = occ ? 1u : 0u;
    }
    return blocked;
}

// PTM-13 rise-epsilon occlusion test. Per ray, march via rtcIntersect1 from the TRUE face centroid
// (callers pass origins AT the face, eps_n=eps_d=0) and reject a hit as self-relief iff its
// out-of-plane rise above the source face's own tangent plane, dz = (P_hit - c_src)*n_src, is below
// the facet-scaled floor h_eps = rise_keps_ * face_edge_local_[src]; the FIRST hit with dz >= h_eps
// is a real occluder (=> blocked), no hit => clear. The origin IS the centroid so c_src = org_local
// and dz = t_hit * (dir_hat . n_src). The origin primID is skipped (dz~0 there) for float robustness.
// This separates the ~0.12 mm coplanar self-relief from the ~2.47 mm real rim occluder that the 1 cm
// offset hops (BUGS.md PTM-13); it is resolution-independent (h_eps scales with the facet). CPU
// Embree only; used when use_rise_eps_ is set.
// OPTION 3 (use_rise_eps2_): the single hit-POINT dz misses a near-coplanar occluder CROSSED near its
// shared edge (dz<h_eps) whose TRIANGLE rises >= h_eps farther along. When set, a would-be-skip hit
// STILL blocks iff it is a genuine float64-INTERIOR crossing (double Moller-Trumbore, bary_min >
// rise_bary_eps_) of a triangle whose max-vertex rise >= h_eps. The interior test rejects the float32
// edge-clip artifacts that are the entire acne source; it is a strict superset of the base blocks.
std::vector<uint8_t> IlluminationModel::traceOccludedRiseEps(
    const std::vector<std::array<double, 3>>& origins,
    const std::vector<std::array<double, 3>>& directions,
    const std::vector<int>& src_faces,
    const std::vector<std::array<double, 3>>& src_normals)
{
    const size_t n = origins.size();
    std::vector<uint8_t> blocked(n, 0);
    if (n == 0 || !embree_scene_) return blocked;

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        // Shift into Embree-local (recentered) space, as traceOccluded / self-skip do. Under
        // rise-epsilon the callers set eps_n=eps_d=0, so this origin IS the source centroid c_src.
        const std::array<double,3> org_local = {
            origins[i][0] - (scene_has_offset_ ? scene_origin_offset_[0] : 0.0),
            origins[i][1] - (scene_has_offset_ ? scene_origin_offset_[1] : 0.0),
            origins[i][2] - (scene_has_offset_ ? scene_origin_offset_[2] : 0.0)
        };
        const double dl = std::sqrt(directions[i][0]*directions[i][0] +
                                    directions[i][1]*directions[i][1] +
                                    directions[i][2]*directions[i][2]);
        const double inv_dl = (dl > 0.0) ? (1.0 / dl) : 1.0;
        const double ddx = directions[i][0] * inv_dl;
        const double ddy = directions[i][1] * inv_dl;
        const double ddz = directions[i][2] * inv_dl;

        const int src = (i < src_faces.size()) ? src_faces[i] : -1;

        // Source face effective outward normal (unit); defensive re-normalize.
        std::array<double,3> ns = (i < src_normals.size()) ? src_normals[i]
                                                           : std::array<double,3>{{0.0,0.0,0.0}};
        {
            const double nl = std::sqrt(ns[0]*ns[0] + ns[1]*ns[1] + ns[2]*ns[2]);
            if (nl > 0.0) { ns[0]/=nl; ns[1]/=nl; ns[2]/=nl; }
        }
        // Per-facet self-relief floor: h_eps = rise_keps_ * local edge. Fallback 0 (unknown src)
        // leaves only the origin-primID skip, so an unknown source still cannot self-light on its
        // own facet; in practice src>=0 and face_edge_local_ is populated for every core face.
        // Absolute floor override (Class-2 fix) when rise_habs_ >= 0, else the edge-proportional floor.
        const double h_eps = (rise_habs_ >= 0.0)
                           ? rise_habs_
                           : ((src >= 0 && static_cast<size_t>(src) < face_edge_local_.size())
                              ? rise_keps_ * static_cast<double>(face_edge_local_[src]) : 0.0);
        // Ray-along component of the source normal: dz = t_hit * (dir_hat . n_src).
        const double dir_dot_n = ddx*ns[0] + ddy*ns[1] + ddz*ns[2];

        float tnear = 0.0f;   // start AT the centroid (~machine eps); src-primID + dz test guard self-hits
        bool occ = false;
        // Bounded march: at most a handful of near-coplanar self-relief hits; the cap guards
        // degenerate/non-manifold fans. tnear strictly increases => no infinite loop.
        for (int iter = 0; iter < 64; ++iter) {
            RTCRayHit rh{};
            rh.ray.org_x = static_cast<float>(org_local[0]);
            rh.ray.org_y = static_cast<float>(org_local[1]);
            rh.ray.org_z = static_cast<float>(org_local[2]);
            rh.ray.dir_x = static_cast<float>(ddx);
            rh.ray.dir_y = static_cast<float>(ddy);
            rh.ray.dir_z = static_cast<float>(ddz);
            rh.ray.tnear = tnear;
            rh.ray.tfar  = 1.0e30f;
            rh.ray.mask  = 0xFFFFFFFFu;
            rh.ray.flags = 0;
            rh.ray.time  = 0.0f;
            rh.hit.geomID = RTC_INVALID_GEOMETRY_ID;
            rh.hit.primID = RTC_INVALID_GEOMETRY_ID;

            RTCIntersectArguments args;
            rtcInitIntersectArguments(&args);
            args.flags = RTC_RAY_QUERY_FLAG_INCOHERENT;
            rtcIntersect1(embree_scene_, &rh, &args);

            if (rh.hit.geomID == RTC_INVALID_GEOMETRY_ID) { occ = false; break; }  // clear to the sun
            const unsigned int prim = rh.hit.primID;
            const float t_hit = rh.ray.tfar;  // Embree writes the hit distance into ray.tfar
            // Out-of-plane rise of this hit above the source face's own tangent plane.
            const double dz = static_cast<double>(t_hit) * dir_dot_n;
            // Self-relief: the origin's own facet (dz~0; skipped by primID for float robustness) or
            // any hit rising less than h_eps out of the plane. Skip it and march past it.
            bool self_or_low = (static_cast<int>(prim) == src) || (dz < h_eps);
            // OPTION 3 OR layer: a hit the single hit-POINT rise would skip STILL blocks iff it is a
            // genuine float64-INTERIOR crossing of a triangle whose max-vertex out-of-plane rise
            // >= h_eps -- i.e. the near-coplanar occluder crossed near its shared edge (dz<h_eps) whose
            // TRIANGLE rises higher farther along. The interior test (double Moller-Trumbore, bary_min
            // > rise_bary_eps_) rejects the float32 edge-clip artifacts that are the whole acne source
            // (STEP-1: acne set is 100% edge-clip, bary<0; real occluders 89% interior). Never removes a
            // block; only promotes a would-be-skip to a block -> strict superset, byte-identical when off.
            if (self_or_low && use_rise_eps2_ && static_cast<int>(prim) != src &&
                static_cast<size_t>(prim) < face_verts_local_.size()) {
                const auto& tv = face_verts_local_[prim];
                const std::array<double,3> A = tv[0], B = tv[1], Cc = tv[2];
                // hit-triangle max-vertex rise above the source plane (org_local IS c_src)
                double tri_max = -1.0e300;
                for (int k = 0; k < 3; ++k) {
                    const double rk = (tv[k][0]-org_local[0])*ns[0]
                                    + (tv[k][1]-org_local[1])*ns[1]
                                    + (tv[k][2]-org_local[2])*ns[2];
                    if (rk > tri_max) tri_max = rk;
                }
                if (tri_max >= h_eps) {
                    // double Moller-Trumbore of the marched ray vs this triangle -> barycentric interior
                    const double e1x=B[0]-A[0], e1y=B[1]-A[1], e1z=B[2]-A[2];
                    const double e2x=Cc[0]-A[0], e2y=Cc[1]-A[1], e2z=Cc[2]-A[2];
                    const double px=ddy*e2z-ddz*e2y, py=ddz*e2x-ddx*e2z, pz=ddx*e2y-ddy*e2x;
                    const double det=e1x*px+e1y*py+e1z*pz;
                    if (std::fabs(det) > 1.0e-30) {
                        const double invd=1.0/det;
                        const double sx=org_local[0]-A[0], sy=org_local[1]-A[1], sz=org_local[2]-A[2];
                        const double u=(sx*px+sy*py+sz*pz)*invd;
                        const double qx=sy*e1z-sz*e1y, qy=sz*e1x-sx*e1z, qz=sx*e1y-sy*e1x;
                        const double vv=(ddx*qx+ddy*qy+ddz*qz)*invd;
                        const double bmin=std::min(u, std::min(vv, 1.0-u-vv));
                        if (bmin > rise_bary_eps_) self_or_low = false;  // real interior occluder
                    }
                }
            }
            if (self_or_low) {
                const float t_next = t_hit + std::max(t_hit * 1.0e-4f, 1.0e-6f);
                if (!(t_next > tnear)) { occ = false; break; }  // no forward progress -> treat as clear
                tnear = t_next;
                continue;
            }
            occ = true; break;  // first hit rising >= h_eps out of the plane is a real occluder
        }
        blocked[i] = occ ? 1u : 0u;
    }
    return blocked;
}

// PTM-18: fan-triangulate the SMALL INTERIOR boundary-edge loops of the core mesh (missing-triangle
// holes / T-junction cracks) that let a low grazing sun thread through to shadowed floor faces via the
// finite-disk penumbra path (the "crater-center flash"). The outer patch perimeter (the boundary
// component whose spatial extent spans the mesh) is NOT a hole and is excluded. Only clean closed loops
// are filled; open/non-manifold components are skipped WITH A WARNING (no silent no-op). Returns
// occluder-only triangles in CORE vertex index space; initializeScene appends them to the shadow BVH.
std::vector<std::array<int, 3>> IlluminationModel::buildInteriorHoleFill(
    const std::vector<std::array<double, 3>>& vertices,
    const std::vector<std::array<int, 3>>& faces) const
{
    std::vector<std::array<int, 3>> fills;
    if (faces.empty() || vertices.empty()) return fills;

    // 1. boundary edges = undirected edges incident to exactly one triangle.
    auto edge_key = [](int a, int b) -> uint64_t {
        uint32_t lo = static_cast<uint32_t>(a < b ? a : b);
        uint32_t hi = static_cast<uint32_t>(a < b ? b : a);
        return (static_cast<uint64_t>(lo) << 32) | static_cast<uint64_t>(hi);
    };
    std::unordered_map<uint64_t, uint32_t> ecount;
    ecount.reserve(faces.size() * 3);
    for (const auto& f : faces) {
        ++ecount[edge_key(f[0], f[1])];
        ++ecount[edge_key(f[1], f[2])];
        ++ecount[edge_key(f[2], f[0])];
    }
    std::vector<std::array<int, 2>> bedges;
    for (const auto& kv : ecount) {
        if (kv.second == 1)
            bedges.push_back({ static_cast<int>(kv.first >> 32),
                               static_cast<int>(kv.first & 0xffffffffu) });
    }
    if (bedges.empty()) return fills;  // watertight core -> nothing to seal

    // 2. connected components of the boundary edges (union-find over boundary vertices).
    std::unordered_map<int, int> parent;
    for (const auto& e : bedges) {
        if (!parent.count(e[0])) parent[e[0]] = e[0];
        if (!parent.count(e[1])) parent[e[1]] = e[1];
    }
    auto find = [&](int x) -> int { while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; } return x; };
    for (const auto& e : bedges) { int ra = find(e[0]), rb = find(e[1]); if (ra != rb) parent[ra] = rb; }
    std::unordered_map<int, std::vector<std::array<int, 2>>> comp;
    for (const auto& e : bedges) comp[find(e[0])].push_back(e);

    // Full-mesh bbox diagonal for the outer-perimeter test (spatial extent, resolution-independent).
    std::array<double, 3> mn = { std::numeric_limits<double>::max(),
                                 std::numeric_limits<double>::max(),
                                 std::numeric_limits<double>::max() };
    std::array<double, 3> mx = { std::numeric_limits<double>::lowest(),
                                 std::numeric_limits<double>::lowest(),
                                 std::numeric_limits<double>::lowest() };
    for (const auto& v : vertices)
        for (int k = 0; k < 3; ++k) { mn[k] = std::min(mn[k], v[k]); mx[k] = std::max(mx[k], v[k]); }
    const double meshD = std::sqrt((mx[0]-mn[0])*(mx[0]-mn[0]) + (mx[1]-mn[1])*(mx[1]-mn[1]) + (mx[2]-mn[2])*(mx[2]-mn[2]));

    int n_holes = 0, n_skip_outer = 0, n_skip_open = 0;
    for (auto& kv : comp) {
        const auto& edges = kv.second;
        // component vertex adjacency + spatial extent
        std::unordered_map<int, std::vector<int>> adj;
        std::array<double, 3> cmn = { std::numeric_limits<double>::max(), std::numeric_limits<double>::max(), std::numeric_limits<double>::max() };
        std::array<double, 3> cmx = { std::numeric_limits<double>::lowest(), std::numeric_limits<double>::lowest(), std::numeric_limits<double>::lowest() };
        for (const auto& e : edges) { adj[e[0]].push_back(e[1]); adj[e[1]].push_back(e[0]); }
        for (const auto& av : adj) {
            const auto& v = vertices[av.first];
            for (int k = 0; k < 3; ++k) { cmn[k] = std::min(cmn[k], v[k]); cmx[k] = std::max(cmx[k], v[k]); }
        }
        const double compD = std::sqrt((cmx[0]-cmn[0])*(cmx[0]-cmn[0]) + (cmx[1]-cmn[1])*(cmx[1]-cmn[1]) + (cmx[2]-cmn[2])*(cmx[2]-cmn[2]));
        if (meshD > 0.0 && compD > 0.5 * meshD) { ++n_skip_outer; continue; }  // outer patch perimeter, not a hole

        // A clean closed simple loop has every vertex at degree exactly 2.
        bool closed = (adj.size() >= 3 && adj.size() <= 4096);
        for (const auto& av : adj) if (av.second.size() != 2) { closed = false; break; }
        if (!closed) { ++n_skip_open; continue; }

        // Walk the loop in order, then fan-triangulate from the first vertex.
        std::vector<int> loop; loop.reserve(adj.size());
        int start = adj.begin()->first, prev = -1, cur = start;
        while (true) {
            loop.push_back(cur);
            int nx = -1;
            for (int w : adj[cur]) if (w != prev) { nx = w; break; }
            if (nx < 0 || nx == start) break;
            prev = cur; cur = nx;
            if (loop.size() > adj.size() + 1) break;  // guard degenerate fans
        }
        if (loop.size() != adj.size()) { ++n_skip_open; continue; }  // not a single clean loop
        for (size_t k = 1; k + 1 < loop.size(); ++k)
            fills.push_back({ loop[0], loop[k], loop[k + 1] });
        ++n_holes;
    }

    std::cout << "[IlluminationModel] PTM-18 interior hole-seal ENABLED: sealed " << n_holes
              << " interior hole(s) -> " << fills.size() << " shadow-only fill triangle(s); excluded "
              << n_skip_outer << " outer/spanning boundary component(s); skipped " << n_skip_open
              << " open/non-manifold boundary component(s) (NOT sealed -- a residual grazing-sun flash "
                 "may persist there; see docs/KNOWN_LIMITATIONS.md, PTM-18). Default OFF -> "
                 "byte-identical.\n";
    return fills;
}

// Batch ray tracing with 16-wide AVX-512 packets for maximum performance on supported hardware
std::vector<int> IlluminationModel::intersectsFirstAVX512(
    const std::vector<std::array<double, 3>>& origins,
    const std::vector<std::array<double, 3>>& directions) {

    // Ensure Embree is initialized
    if (!embree_scene_) {
        std::cerr << "Embree scene not initialized" << std::endl;
        return std::vector<int>(origins.size(), -1);
    }
    
    // 
    size_t num_rays = origins.size();
    std::vector<int> face_ids(num_rays, -1);  // Initialize all to -1 (no hit)

#if defined(__x86_64__) || defined(_M_X64) && defined(__AVX512F__) && defined(__AVX512DQ__)
    // Process rays in SIMD packets of 16
    size_t processed = 0;
    while (processed < num_rays) {
        // Number of rays in this packet (up to 16)
        size_t packet_size = std::min(size_t(16), num_rays - processed);
        
        // Initialize SIMD ray packet
        alignas(64) int valid[16];
        alignas(64) RTCRayHit16 rayhit;
        
        // Clear valid array
        std::memset(valid, 0, sizeof(valid));
        
        // Set up rays for this packet
        for (size_t i = 0; i < packet_size; ++i) {
            valid[i] = -1;  // Mark ray as active
            
            size_t idx = processed + i;
            
            // Set ray origin and direction
            rayhit.ray.org_x[i] = origins[idx][0];
            rayhit.ray.org_y[i] = origins[idx][1];
            rayhit.ray.org_z[i] = origins[idx][2];
            rayhit.ray.dir_x[i] = directions[idx][0];
            rayhit.ray.dir_y[i] = directions[idx][1];
            rayhit.ray.dir_z[i] = directions[idx][2];
            
            // Other ray parameters
            rayhit.ray.tnear[i] = 1.0e-4f;
            rayhit.ray.tfar[i]  = 1.0e30f;
            rayhit.ray.mask[i] = 0xFFFFFFFF;
            rayhit.ray.flags[i] = 0;
            rayhit.ray.time[i] = 0.0f;
            
            // Initialize hit fields
            rayhit.hit.geomID[i] = RTC_INVALID_GEOMETRY_ID;
            rayhit.hit.primID[i] = RTC_INVALID_GEOMETRY_ID;
            rayhit.hit.instID[0][i] = RTC_INVALID_GEOMETRY_ID;
        }
        
        // Trace ray packet
        RTCIntersectArguments args;
        rtcInitIntersectArguments(&args);
        rtcIntersect16(valid, embree_scene_, &rayhit, &args);
        
        // Process results
        for (size_t i = 0; i < packet_size; ++i) {
            size_t idx = processed + i;
            if (rayhit.hit.geomID[i] != RTC_INVALID_GEOMETRY_ID) {
                face_ids[idx] = static_cast<int>(rayhit.hit.primID[i]);
            }
        }
        
        processed += packet_size;
    }
    
    return face_ids;

#else
    // Fallback to SIMD or non-SIMD version on platforms without AVX-512
    #if defined(__x86_64__) || defined(_M_X64)
        return intersectsFirstSIMD(origins, directions);
    #else
        return intersectsFirst(origins, directions);
    #endif
#endif
}

// Auto-select best ray tracing method based on hardware and ray count
std::vector<int> IlluminationModel::traceRays(
    const std::vector<std::array<double, 3>>& origins,
    const std::vector<std::array<double, 3>>& directions) {
    
    size_t num_rays = origins.size();
    
#if defined(__x86_64__) || defined(_M_X64)
    // Use different methods based on ray count and CPU features
    if (num_rays >= 16 && isAVX512Supported()) {
        // std::cerr << "Using AVX-512 ray tracing for " << num_rays << " rays" << std::endl;
        return intersectsFirstAVX512(origins, directions);
    } else if (num_rays >= 8) {
        // std::cerr << "Using SIMD ray tracing for " << num_rays << " rays" << std::endl;
        return intersectsFirstSIMD(origins, directions);
    } else {
        // std::cerr << "Using scalar ray tracing for " << num_rays << " rays" << std::endl;
        return intersectsFirst(origins, directions);
    }
#else
    // On non-x86 architectures, always use the scalar version
        // std::cerr << "Using scalar ray tracing (non-x86 architecture) for " << num_rays << " rays" << std::endl;
    return intersectsFirst(origins, directions);
#endif
}

// Helper to check if AVX-512 is supported
bool IlluminationModel::isAVX512Supported() {
    static bool checked = false;
    static bool avx512_supported = false;
    
    if (!checked) {
        checked = true;
        
#if defined(__x86_64__) || defined(_M_X64)
    #if defined(__AVX512F__) && defined(__AVX512DQ__)
        // Compile-time check passed
        
        // Additional runtime detection could be added here if needed
        // This ensures the binary has AVX512 instructions AND the CPU supports them
        
        // On Intel CPUs with AVX-512, also check for thermal throttling issues
        // by trying vectorized code on a small test case first
        static bool tested = false;
        static bool avx512_works = false;
        
        if (!tested) {
            // Simple runtime test to verify AVX-512 actually works
            tested = true;
            avx512_works = true;  // Default to true, could add actual test here
        }
        
        avx512_supported = avx512_works;
        
        // TEMPORARY: Always output SIMD support info regardless of debug flag
        std::cerr << "AVX-512 support detected: " << (avx512_supported ? "Yes" : "No") << std::endl;
    #else
        // TEMPORARY: Always output SIMD support info regardless of debug flag
        std::cerr << "AVX-512 not supported at compile time" << std::endl;
        avx512_supported = false;
    #endif
#else
        // TEMPORARY: Always output SIMD support info regardless of debug flag
        std::cerr << "Non-x86 architecture detected, AVX-512 not supported" << std::endl;
        avx512_supported = false;
#endif

        // TEMPORARY: Print detailed SIMD support information
        printSIMDInfoTemporary();
    }
    
    return avx512_supported;
}

// TEMPORARY: Helper function to print detailed SIMD information
void IlluminationModel::printSIMDInfoTemporary() {
    std::cerr << "=== SIMD Support Information ===" << std::endl;

#if defined(__x86_64__) || defined(_M_X64)
    std::cerr << "x86-64 architecture detected" << std::endl;

    #if defined(__SSE2__)
        std::cerr << "SSE2: Supported" << std::endl;
    #else
        std::cerr << "SSE2: Not supported" << std::endl;
    #endif
    
    #if defined(__AVX__)
        std::cerr << "AVX: Supported" << std::endl;
    #else
        std::cerr << "AVX: Not supported" << std::endl;
    #endif
    
    #if defined(__AVX2__)
        std::cerr << "AVX2: Supported" << std::endl;
    #else
        std::cerr << "AVX2: Not supported" << std::endl;
    #endif
    
    #if defined(__AVX512F__)
        std::cerr << "AVX-512F: Supported" << std::endl;
    #else
        std::cerr << "AVX-512F: Not supported" << std::endl;
    #endif
    
    #if defined(__AVX512DQ__)
        std::cerr << "AVX-512DQ: Supported" << std::endl;
    #else
        std::cerr << "AVX-512DQ: Not supported" << std::endl;
    #endif
#else
    std::cerr << "Non-x86 architecture detected" << std::endl;
#endif

    std::cerr << "===============================" << std::endl;
}

// Clean up SPICE resources
void IlluminationModel::cleanupSpice() {
    // Unload all kernels
    unload_c("all");
}

bool IlluminationModel::calculateIllumination(
    const std::vector<std::array<double, 3>>& mesh_vertices,
    const std::vector<std::array<int, 3>>& mesh_faces,
    const std::string& start_time,
    double duration_days) {
    
    static bool warned = false;
    if (!warned) {
        std::cerr << "Warning: Using non-optimized illumination calculation path. "
                  << "Consider using calculateIlluminationOptimized instead." << std::endl;
        warned = true;
    }
    
    // Initialize SPICE for ephemeris calculations
    if (!precomputed_positions_loaded_ && !initialize_spice()) { 
        std::cerr << "Failed to initialize SPICE" << std::endl;
        return false;
    }
    
    // Initialize Embree for ray tracing
    if (!initializeEmbree()) {
        std::cerr << "Failed to initialize Embree" << std::endl;
        return false;
    }
    
    // Set up Embree scene with the mesh
    if (!initializeScene(mesh_vertices, mesh_faces)) {
        std::cerr << "Failed to initialize Embree scene" << std::endl;
        return false;
    }
    
    // Skip position calculations if precomputed positions were loaded
    if (!precomputed_positions_loaded_) {
        times_.resize(time_steps_);
        time_strings_.resize(time_steps_);

        if (!external_ets_.empty()) {
            // External (possibly non-uniform) ET grid supplied by caller
            if (static_cast<int>(external_ets_.size()) != time_steps_) {
                std::cerr << "[IlluminationModel] ERROR: external ET grid size ("
                          << external_ets_.size() << ") != time_steps ("
                          << time_steps_ << ")" << std::endl;
                return false;
            }
            for (int t = 0; t < time_steps_; ++t) {
                times_[t] = external_ets_[t];
                char time_str[41];
                et2utc_c(times_[t], "ISOC", 3, sizeof(time_str), time_str);
                time_strings_[t] = std::string(time_str);
            }
        } else {
            // Uniform grid (legacy behavior)
            double start_et = convertTimeToET(start_time);
            double duration_seconds = duration_days * constants::SECONDS_PER_DAY;
            double step_size = duration_seconds / time_steps_;
            for (int t = 0; t < time_steps_; ++t) {
                double et = start_et + t * step_size;
                times_[t] = et;
                char time_str[41];
                et2utc_c(et, "ISOC", 3, sizeof(time_str), time_str);
                time_strings_[t] = std::string(time_str);
            }
        }

        // Compute Sun and Primary positions for each time step
        sun_positions_.resize(time_steps_);
        primary_positions_.resize(time_steps_);

        for (int t = 0; t < time_steps_; ++t) {
            sun_positions_[t] = getSunPosition(times_[t]);
            primary_positions_[t] = getPrimaryPosition(times_[t]);
        }
    }

    // Calculate face normals and centers
    std::vector<std::array<double, 3>> face_normals(mesh_faces.size());
    std::vector<std::array<double, 3>> face_centers(mesh_faces.size());

    for (size_t f = 0; f < mesh_faces.size(); ++f) {
        const auto& face = mesh_faces[f];
        const auto& v0 = mesh_vertices[face[0]];
        const auto& v1 = mesh_vertices[face[1]];
        const auto& v2 = mesh_vertices[face[2]];

        face_normals[f] = calculateFaceNormal(v0, v1, v2);
        face_centers[f] = {{
            (v0[0] + v1[0] + v2[0]) / 3.0,
            (v0[1] + v1[1] + v2[1]) / 3.0,
            (v0[2] + v1[2] + v2[2]) / 3.0
        }};
    }
    
    // Resize illumination values array
    illumination_values_.resize(mesh_faces.size() * time_steps_, 0.0);
    
    // Process each time step
    for (int t = 0; t < time_steps_; ++t) {
        const auto& sun_pos = sun_positions_[t];
        const auto& primary_pos = primary_positions_[t];

        // Use calculateSolarIlluminationOptimized for each face
        #pragma omp parallel for schedule(dynamic)
        for (size_t f = 0; f < mesh_faces.size(); ++f) {
            // This calls our optimized method which includes Primary eclipse checking
            double illumination = calculateSolarIlluminationWithFiniteDisk(
                face_centers[f],
                face_normals[f],
                sun_pos,
                primary_pos,
                constants::N_SOLAR_SAMPLES,  // Number of solar samples (sqrt(N) should be integer)
                static_cast<int>(f)          // PTM-13 P1: source core-face id for topological self-skip
            );
            
            // Store illumination value
            illumination_values_[f * time_steps_ + t] = illumination;
        }
    }
    
    return true;
}

bool IlluminationModel::calculateIlluminationOptimized(
    const std::vector<std::array<double, 3>>& mesh_vertices,
    const std::vector<std::array<int, 3>>& mesh_faces,
    const std::string& start_time,
    double duration_days) {
    
    debugPrint("Debug: Entering calculateIlluminationOptimized");
    
    // Check for very large meshes and adjust thread count accordingly
    size_t face_count = mesh_faces.size();
    bool is_large_mesh = face_count > 1000000;
    bool is_very_large_mesh = face_count > 10000000;
    
    // Auto-adjust thread count for large meshes to prevent memory pressure
    int original_thread_count = 0;
    #ifdef _OPENMP
    original_thread_count = omp_get_max_threads();
    if (is_very_large_mesh) {
        int adjusted_threads = std::min(8, original_thread_count);
        debugPrint("Very large mesh detected (", face_count, " faces). Limiting to ", 
                  adjusted_threads, " threads to reduce memory pressure.");
        omp_set_num_threads(adjusted_threads);
    } else if (is_large_mesh) {
        int adjusted_threads = std::min(16, original_thread_count);
        debugPrint("Large mesh detected (", face_count, " faces). Limiting to ", 
                  adjusted_threads, " threads.");
        omp_set_num_threads(adjusted_threads);
    }
    #endif
    
    // Initialize SPICE for ephemeris calculations
    if (!precomputed_positions_loaded_ && !initialize_spice()) {
        std::cerr << "Failed to initialize SPICE" << std::endl;
        return false;
    }
    
    debugPrint("Debug: SPICE initialization complete");
    
    // Initialize Embree for ray tracing
    debugPrint("Debug: Starting Embree initialization");
    if (!initializeEmbree()) {
        std::cerr << "Failed to initialize Embree" << std::endl;
        return false;
    }
    debugPrint("Debug: Embree initialization complete");
    
    // Set up Embree scene with the mesh
    debugPrint("Debug: Starting scene initialization with " + std::to_string(mesh_vertices.size()) + " vertices and " + std::to_string(mesh_faces.size()) + " faces");
    
    try {
        if (!initializeScene(mesh_vertices, mesh_faces)) {
            debugPrint("Debug: Failed to initialize Embree scene");
            return false;
        }
        debugPrint("Debug: Scene initialization complete");
    } catch (const std::exception& e) {
        debugPrint("Debug: Exception during scene initialization: " + std::string(e.what()));
        return false;
    } catch (...) {
        debugPrint("Debug: Unknown exception during scene initialization");
        return false;
    }
    
    // Skip position calculations if precomputed positions were loaded
    if (!precomputed_positions_loaded_) {
        debugPrint("Debug: Calculating celestial body positions");
        // Initialize time array
        times_.resize(time_steps_);
        time_strings_.resize(time_steps_);

        if (!external_ets_.empty()) {
            if (static_cast<int>(external_ets_.size()) != time_steps_) {
                std::cerr << "[IlluminationModel] ERROR: external ET grid size ("
                          << external_ets_.size() << ") != time_steps ("
                          << time_steps_ << ")" << std::endl;
                return false;
            }
            for (int t = 0; t < time_steps_; ++t) {
                times_[t] = external_ets_[t];
                char time_str[41];
                et2utc_c(times_[t], "ISOC", 3, sizeof(time_str), time_str);
                time_strings_[t] = std::string(time_str);
            }
        } else {
            double start_et = convertTimeToET(start_time);
            double duration_seconds = duration_days * constants::SECONDS_PER_DAY;
            double step_size = duration_seconds / time_steps_;
            for (int t = 0; t < time_steps_; ++t) {
                double et = start_et + t * step_size;
                times_[t] = et;
                char time_str[41];
                et2utc_c(et, "ISOC", 3, sizeof(time_str), time_str);
                time_strings_[t] = std::string(time_str);
            }
        }

        // Compute Sun and Primary positions for each time step
        debugPrint("Debug: Resizing position arrays for " + std::to_string(time_steps_) + " time steps");
        sun_positions_.resize(time_steps_);
        primary_positions_.resize(time_steps_);

        debugPrint("Debug: Computing Sun and Primary positions");
        for (int t = 0; t < time_steps_; ++t) {
            try {
                sun_positions_[t] = getSunPosition(times_[t]);
                primary_positions_[t] = getPrimaryPosition(times_[t]);
            } catch (const std::exception& e) {
                debugPrint("Debug: Exception during position calculation at time step " + std::to_string(t) + ": " + std::string(e.what()));
                return false;
            } catch (...) {
                debugPrint("Debug: Unknown exception during position calculation at time step " + std::to_string(t));
                return false;
            }
        }
        debugPrint("Debug: Sun and Primary positions computed successfully");
    }
    
    debugPrint("Debug: Calculating face normals and centers");
    // Calculate face normals and centers (parallel) WITH outward enforcement
    std::vector<std::array<double, 3>> face_normals(mesh_faces.size());
    std::vector<std::array<double, 3>> face_centers(mesh_faces.size());

    bool strict_horizon = true;
    if (const char* ev = std::getenv("PTM_DISABLE_STRICT_HORIZON")) {
        if (std::atoi(ev) != 0) strict_horizon = false;
    }

    // Choose a reasonable OpenMP chunk size (avoid using undeclared 'chunk_size')
    int omp_chunk = static_cast<int>(std::max<size_t>(1, chunk_size_ / 64));
    if (omp_chunk < 32) omp_chunk = 32;
    #pragma omp parallel for schedule(dynamic, omp_chunk)
    for (size_t f = 0; f < mesh_faces.size(); ++f) {
        try {
            const auto& face = mesh_faces[f];
            const auto& v0 = mesh_vertices[face[0]];
            const auto& v1 = mesh_vertices[face[1]];
            const auto& v2 = mesh_vertices[face[2]];

            // Center
            std::array<double,3> center = {
                (v0[0] + v1[0] + v2[0]) / 3.0,
                (v0[1] + v1[1] + v2[1]) / 3.0,
                (v0[2] + v1[2] + v2[2]) / 3.0
            };

            // Normal (raw)
            std::array<double,3> e1 = { v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2] };
            std::array<double,3> e2 = { v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2] };
            std::array<double,3> n = {
                e1[1]*e2[2] - e1[2]*e2[1],
                e1[2]*e2[0] - e1[0]*e2[2],
                e1[0]*e2[1] - e1[1]*e2[0]
            };
            double len = std::sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]);
            if (len > 0.0) {
                n[0] /= len; n[1] /= len; n[2] /= len;
            } else {
                n = {0.0,0.0,1.0};
            }

            // Enforce outward orientation: dot(center, normal) must be > 0
            double dcn = center[0]*n[0] + center[1]*n[1] + center[2]*n[2];
            if (dcn < 0.0) {
                n[0] = -n[0]; n[1] = -n[1]; n[2] = -n[2];
            }

            face_centers[f] = center;
            face_normals[f] = n;
        } catch (...) {
            face_centers[f] = {0.0,0.0,0.0};
            face_normals[f] = {0.0,0.0,1.0};
        }
    }
    
    debugPrint("Debug: Finished calculating normals and centers");
    
    debugPrint("Debug: Resizing result arrays for " + std::to_string(mesh_faces.size()) + " faces and " + std::to_string(time_steps_) + " time steps");
    
    // Pre-calculate the total size needed
    size_t total_faces = mesh_faces.size();
    size_t total_values = total_faces * time_steps_;
    
    // Check if we need to use chunked storage (over 500 million elements)
    const size_t chunked_threshold = 500000000;
    
    try {
        if (total_values > chunked_threshold) {
            std::cout << "Using chunked storage for very large mesh (" 
                      << total_faces << " faces, " << time_steps_ << " time steps)" << std::endl;
            
            // Initialize chunked storage
            initializeChunkedStorage(total_faces, time_steps_);
            
            // Clear standard arrays to save memory
            illumination_values_.clear();
            direct_solar_values_.clear();
            primary_reflected_values_.clear();
            primary_ir_values_.clear();
            
            // Shrink to fit to release memory
            illumination_values_.shrink_to_fit();
            direct_solar_values_.shrink_to_fit();
            primary_reflected_values_.shrink_to_fit();
            primary_ir_values_.shrink_to_fit();
        } else {
            // Regular storage approach
            using_chunked_storage_ = false;
            
            // Check for size overflow to prevent integer overflow
            if (mesh_faces.size() > std::numeric_limits<size_t>::max() / time_steps_) {
                throw std::overflow_error("Mesh size causes integer overflow in array allocation");
            }
            
            // Resize arrays with zero initialization
            illumination_values_.resize(total_values, 0.0);
            direct_solar_values_.resize(total_values, 0.0);
            primary_reflected_values_.resize(total_values, 0.0);
            primary_ir_values_.resize(total_values, 0.0);
        }
        
        debugPrint("Debug: Arrays resized successfully");
    } catch (const std::exception& e) {
        debugPrint("Debug: Exception during array resizing: " + std::string(e.what()));
        std::cerr << "Failed to allocate memory for illumination arrays: " << e.what() << std::endl;
        std::cerr << "Try using a lower resolution mesh or running with fewer time steps" << std::endl;
        return false;
    }
    
    debugPrint("Debug: Starting illumination calculations for " + std::to_string(time_steps_) + " time steps");
    
    // Create progress bar
    thermal::ProgressBar progressBar(time_steps_, "Illumination calculation", 50, show_progress_);
    
    // For very large meshes, use smaller thread-local batches to reduce memory pressure
    size_t batch_size = is_very_large_mesh ? 16 : (is_large_mesh ? 32 : COLLECTION_BATCH_SIZE);
    
    // Process each time step with batch processing
    for (int t = 0; t < time_steps_; ++t) {
        debugPrint("Debug: Processing time step " + std::to_string(t+1) + " of " + std::to_string(time_steps_));
        const auto& sun_pos = sun_positions_[t];
        const auto& primary_pos = primary_positions_[t];
        
        // Temporary vectors to store values for this time step
        // For large meshes, allocate these in chunks to reduce peak memory usage
        const size_t chunk_faces = is_very_large_mesh ? 1000000 : mesh_faces.size();
        
        std::vector<double> timestep_direct_solar(mesh_faces.size(), 0.0);
        std::vector<double> timestep_primary_reflected(mesh_faces.size(), 0.0);
        std::vector<double> timestep_primary_ir(mesh_faces.size(), 0.0);
        
        // Initialize thread-local ray batching system
        initRayBatchCollection();
        
        // Calculate direct solar illumination
        debugPrint("Debug: Calculating direct solar illumination for time step " + std::to_string(t+1));
        try {
#ifdef PTM_GPU_OPENCL
            // Use GPU batched version when GPU ray tracing is enabled
            if (use_gpu_ray_tracer_ && gpu_ray_pool_ && gpu_mesh_uploaded_) {
                calculateDirectSolarBatchWithFiniteDiskGPU(
                    face_centers,
                    face_normals,
                    sun_pos,
                    primary_pos,
                    timestep_direct_solar,
                    constants::N_SOLAR_SAMPLES
                );
            } else {
                calculateDirectSolarBatchWithFiniteDisk(
                    face_centers,
                    face_normals,
                    sun_pos,
                    primary_pos,
                    timestep_direct_solar,
                    constants::N_SOLAR_SAMPLES
                );
            }
#else
            calculateDirectSolarBatchWithFiniteDisk(
                face_centers,
                face_normals,
                sun_pos,
                primary_pos,
                timestep_direct_solar,
                constants::N_SOLAR_SAMPLES
            );
#endif
            debugPrint("Debug: Direct solar calculation complete for time step " + std::to_string(t+1));
        } catch (const std::exception& e) {
            debugPrint("Debug: Exception during direct solar calculation at time step " + std::to_string(t) + ": " + std::string(e.what()));
            std::cerr << "Exception during direct solar calculation: " << e.what() << std::endl;
            // Continue with zeros rather than failing completely
            std::fill(timestep_direct_solar.begin(), timestep_direct_solar.end(), 0.0);
        } catch (...) {
            debugPrint("Debug: Unknown exception during direct solar calculation at time step " + std::to_string(t));
            std::cerr << "Unknown exception during direct solar calculation" << std::endl;
            // Continue with zeros rather than failing completely
            std::fill(timestep_direct_solar.begin(), timestep_direct_solar.end(), 0.0);
        }

        // Calculate Primary reflected solar (if enabled)
        if (include_primary_reflection_ && !primary_is_sun_) {
            debugPrint("Debug: Calculating Primary reflection for time step " + std::to_string(t+1));
            try {
                calculatePrimaryReflectedBatch(
                    face_centers,
                    face_normals,
                    sun_pos,
                    primary_pos,
                    timestep_primary_reflected
                );
                debugPrint("Debug: Primary reflection calculation complete for time step " + std::to_string(t+1));
            } catch (const std::exception& e) {
                debugPrint("Debug: Exception during Primary reflection calculation at time step " + std::to_string(t) + ": " + std::string(e.what()));
                std::cerr << "Exception during Primary reflection calculation: " << e.what() << std::endl;
                // Continue with zeros rather than failing completely
                std::fill(timestep_primary_reflected.begin(), timestep_primary_reflected.end(), 0.0);
            } catch (...) {
                debugPrint("Debug: Unknown exception during Primary reflection calculation at time step " + std::to_string(t));
                std::cerr << "Unknown exception during Primary reflection calculation" << std::endl;
                // Continue with zeros rather than failing completely
                std::fill(timestep_primary_reflected.begin(), timestep_primary_reflected.end(), 0.0);
            }
        } else {
            // Fill with zeros when primary is Sun or reflection is disabled
            std::fill(timestep_primary_reflected.begin(), timestep_primary_reflected.end(), 0.0);
        }

        // Calculate Primary IR heating
        if (!primary_is_sun_) {
            debugPrint("Debug: Calculating Primary IR for time step " + std::to_string(t+1));
            try {
                calculatePrimaryIRBatch(
                    face_centers,
                    face_normals,
                    primary_pos,
                    timestep_primary_ir
                );
                debugPrint("Debug: Primary IR calculation complete for time step " + std::to_string(t+1));
            } catch (const std::exception& e) {
                debugPrint("Debug: Exception during Primary IR calculation at time step " + std::to_string(t) + ": " + std::string(e.what()));
                std::cerr << "Exception during Primary IR calculation at time step " + std::to_string(t)
                        << " with exception: " + std::string(e.what()) << std::endl;
                std::cerr << "Caught exception in calculatePrimaryIRBatch at time step " + std::to_string(t)
                        << ": " + std::string(e.what()) << std::endl;
                // Continue with zeros rather than failing completely
                std::fill(timestep_primary_ir.begin(), timestep_primary_ir.end(), 0.0);
            } catch (...) {
                std::cerr << "Unknown exception during Primary IR calculation at time step " + std::to_string(t) << std::endl;
                // Continue with zeros rather than failing completely
                std::fill(timestep_primary_ir.begin(), timestep_primary_ir.end(), 0.0);
            }
        } else {
            // Fill with zeros when primary is Sun
            std::fill(timestep_primary_ir.begin(), timestep_primary_ir.end(), 0.0);
        }
        
        // Calculate total illumination
        std::vector<double> timestep_total(mesh_faces.size(), 0.0);
        for (size_t i = 0; i < timestep_total.size(); ++i) {
            // Total = Direct + Reflected
            timestep_total[i] = timestep_direct_solar[i] + 
                              (include_primary_reflection_ ? timestep_primary_reflected[i] : 0.0);
        }
        
        // Copy batch results to main storage in face-major order
        // Process in smaller chunks for better cache behavior
        const size_t copy_chunk_size = 4096;
        for (size_t chunk_start = 0; chunk_start < mesh_faces.size(); chunk_start += copy_chunk_size) {
            size_t chunk_end = std::min(chunk_start + copy_chunk_size, mesh_faces.size());
            
            #pragma omp parallel for schedule(static)
            for (size_t f = chunk_start; f < chunk_end; ++f) {
                if (using_chunked_storage_) {
                    storeChunkedValue(illumination_chunks_, f, t, timestep_total[f]);
                    storeChunkedValue(direct_solar_chunks_, f, t, timestep_direct_solar[f]);
                    storeChunkedValue(primary_reflected_chunks_, f, t, timestep_primary_reflected[f]);
                    storeChunkedValue(primary_ir_chunks_, f, t, timestep_primary_ir[f]);
                } else {
                    size_t index = f * time_steps_ + t;
                    if (index < illumination_values_.size()) {  // Safety check
                        illumination_values_[index] = timestep_total[f];  // Store correct total
                        direct_solar_values_[index] = timestep_direct_solar[f];
                        primary_reflected_values_[index] = timestep_primary_reflected[f];
                        primary_ir_values_[index] = timestep_primary_ir[f];
                    }
                }
            }
        }
        
        // Update progress bar
        progressBar.update(t + 1);
    }
    
    // Ensure progress bar is completed
    progressBar.finish();
    
    // Restore original thread count if we modified it
    #ifdef _OPENMP
    if ((is_large_mesh || is_very_large_mesh) && original_thread_count > 0) {
        omp_set_num_threads(original_thread_count);
        debugPrint("Debug: Restored original thread count: ", original_thread_count);
    }
    #endif
    
    debugPrint("Debug: Illumination calculation completed successfully");
    return true;
}

std::map<int, std::vector<double>> IlluminationModel::calculateLocationIllumination(
    const std::vector<std::pair<double, double>>& locations,
    const std::string& start_time,
    double duration_days) {
    
    // Initialize SPICE for ephemeris calculations (if not already initialized)
    if (!precomputed_positions_loaded_ && !initialize_spice()) { 
        std::cerr << "Failed to initialize SPICE" << std::endl;
        return {};
    }
    
    // If positions not already loaded, calculate them
    if (sun_positions_.empty() || primary_positions_.empty()) {
        // Initialize time array
        times_.resize(time_steps_);
        time_strings_.resize(time_steps_);
        sun_positions_.resize(time_steps_);
        primary_positions_.resize(time_steps_);

        if (!external_ets_.empty() &&
            static_cast<int>(external_ets_.size()) == time_steps_) {
            for (int t = 0; t < time_steps_; ++t) {
                times_[t] = external_ets_[t];
                char time_str[41];
                et2utc_c(times_[t], "ISOC", 3, sizeof(time_str), time_str);
                time_strings_[t] = std::string(time_str);
                sun_positions_[t] = getSunPosition(times_[t]);
                primary_positions_[t] = getPrimaryPosition(times_[t]);
            }
        } else {
            double start_et = convertTimeToET(start_time);
            double duration_seconds = duration_days * constants::SECONDS_PER_DAY;
            double step_size = duration_seconds / time_steps_;
            for (int t = 0; t < time_steps_; ++t) {
                double et = start_et + t * step_size;
                times_[t] = et;
                char time_str[41];
                et2utc_c(et, "ISOC", 3, sizeof(time_str), time_str);
                time_strings_[t] = std::string(time_str);
                sun_positions_[t] = getSunPosition(times_[t]);
                primary_positions_[t] = getPrimaryPosition(times_[t]);
            }
        }
    }

    // Result container
    std::map<int, std::vector<double>> location_illumination;
    
    // Convert locations to 3D positions and normals
    std::vector<std::array<double, 3>> positions;
    std::vector<std::array<double, 3>> normals;
    
    for (const auto& [lat, lon] : locations) {
        // Convert lat/lon to 3D position on Body's surface
        std::array<double, 3> position = cartesianFromLatLon(lat, lon, constants::BODY_RADIUS);
        positions.push_back(position);
        
        // Normal vector at this position (points outward from center)
        normals.push_back(normalize(position));
    }
    
    // For each time step
    for (int t = 0; t < times_.size(); ++t) {
        const auto& sun_pos = sun_positions_[t];
        const auto& primary_pos = primary_positions_[t];

        // Process all locations for this time step
        for (size_t loc_idx = 0; loc_idx < locations.size(); ++loc_idx) {
            const auto& position = positions[loc_idx];
            const auto& normal = normals[loc_idx];
            
            // Calculate illumination with our advanced model
            double illumination = calculateSolarIlluminationWithFiniteDisk(
                position,
                normal,
                sun_pos,
                primary_pos,
                constants::N_SOLAR_SAMPLES  // Number of solar samples (sqrt(N) should be integer)
            );
            
            // Store result
            if (location_illumination.find(loc_idx) == location_illumination.end()) {
                location_illumination[loc_idx] = std::vector<double>(times_.size(), 0.0);
            }
            
            location_illumination[loc_idx][t] = illumination;
        }
    }
    
    return location_illumination;
}

const std::vector<double>& IlluminationModel::getIllumination() const {
    return illumination_values_;
}

const std::vector<double>& IlluminationModel::getTimes() const {
    return times_;
}

std::vector<std::pair<double, double>> IlluminationModel::getPrimaryIRFluxForFace(int face_idx) const {
    std::vector<std::pair<double, double>> result;
    result.reserve(times_.size());
    
    for (size_t t = 0; t < times_.size(); ++t) {
        double ir_flux;
        if (using_chunked_storage_) {
            ir_flux = getChunkedValue(primary_ir_chunks_, face_idx, t);
        } else {
            size_t index = face_idx * time_steps_ + t;
            ir_flux = (index < primary_ir_values_.size()) ? 
                primary_ir_values_[index] : 0.0;
        }
        result.emplace_back(times_[t], ir_flux);
    }
    
    return result;
}

std::vector<std::array<double, 3>> IlluminationModel::getSunPositions() const {
    return sun_positions_;
}

std::vector<std::array<double, 3>> IlluminationModel::getPrimaryPositions() const {
    return primary_positions_;
}

// Static utility methods
std::array<double, 3> IlluminationModel::cartesianFromLatLon(double lat, double lon, double radius) {
    double lat_rad = lat * constants::DEG_TO_RAD;
    double lon_rad = lon * constants::DEG_TO_RAD;
    
    return {{
        radius * std::cos(lat_rad) * std::cos(lon_rad),
        radius * std::cos(lat_rad) * std::sin(lon_rad),
        radius * std::sin(lat_rad)
    }};
}

std::pair<double, double> IlluminationModel::latLonFromCartesian(double x, double y, double z) {
    double radius = std::sqrt(x*x + y*y + z*z);
    double lat_rad = std::asin(z / radius);
    double lon_rad = std::atan2(y, x);
    
    return {
        lat_rad * constants::RAD_TO_DEG,
        lon_rad * constants::RAD_TO_DEG
    };
}

bool IlluminationModel::initialize_spice(bool verbose) {
    static bool spice_initialized = false;

    // Resolve the kernel directory if it was not set explicitly:
    // $PTM_SPICE_KERNELS, then the compiled-in SPICE_KERNELS_DIR (CMake cache
    // variable; default <repo>/data/kernels). Fail loud rather than fall back
    // to a developer-machine path (portability audit P0, PORT-02).
    if (kernels_path_.empty()) {
        const char* env_kernels = std::getenv("PTM_SPICE_KERNELS");
        if (env_kernels && *env_kernels) {
            kernels_path_ = env_kernels;
        } else {
            kernels_path_ = ptm_thermal::config::SPICE_KERNELS_DIR;
        }
    }
    if (kernels_path_.empty()) {
        std::cerr << "Error: SPICE kernel directory is unresolved (empty).\n"
                  << "  Resolution order: setKernelsPath()/config, $PTM_SPICE_KERNELS,\n"
                  << "  compiled-in SPICE_KERNELS_DIR (CMake; default <repo>/data/kernels).\n"
                  << "  Populate the kernel directory with: python3 scripts/fetch_kernels.py\n"
                  << "  (see INSTALL.md, section \"SPICE kernel data\")." << std::endl;
        return false;
    }

    // If using Keplerian propagation, skip full SPICE initialization
    // but still load the leapseconds kernel for time conversion
    if (use_keplerian_) {
        if (!spice_initialized) {
            try {
                erract_c("SET", 0, const_cast<char*>("RETURN"));
                errprt_c("SET", 0, const_cast<char*>("NONE"));

                // Load just the leapseconds kernel for str2et_c time conversion
                std::string lsk_path = kernels_path_ + "/lsk/naif0012.tls";
                if (std::filesystem::exists(lsk_path)) {
                    furnsh_c(lsk_path.c_str());
                    if (verbose) {
                        std::cout << "Keplerian mode: loaded leapseconds kernel for time conversion" << std::endl;
                    }
                } else {
                    std::cout << "Keplerian mode: leapseconds kernel not found at " << lsk_path
                              << " (time conversion may use approximation)" << std::endl;
                }
                spice_initialized = true;
            } catch (const std::exception& e) {
                std::cerr << "SPICE LSK initialization error: " << e.what() << std::endl;
            }
        }
        std::cout << "Using Keplerian orbit propagation for " << target_body_name_ << std::endl;
        return true;
    }

    if (!spice_initialized) {
        try {
            // Reset error handling
            erract_c("SET", 0, const_cast<char*>("RETURN"));
            errprt_c("SET", 0, const_cast<char*>("NONE"));

            std::cout << "Initializing SPICE with meta-kernel" << std::endl;
            
            // Use the meta-kernel filename from properties
            std::string meta_kernel = kernels_path_ + "/" + meta_kernel_filename_;

            // A Europa/Galilean metakernel (europa.tm) references jup365.bsp,
            // which the default Moon kernel fetch omits. Point the user at the
            // fetch set that actually includes it, so the bare command in the
            // hint does not leave them failing on the same missing kernel.
            const bool galilean_metakernel =
                (meta_kernel_filename_.find("europa") != std::string::npos);
            const char* fetch_bodies_flag = galilean_metakernel ? " --bodies europa" : "";

            if (!std::filesystem::exists(meta_kernel)) {
                std::cerr << "Error: SPICE meta-kernel not found: " << meta_kernel << "\n"
                          << "  Kernel dir resolution: setKernelsPath()/config, $PTM_SPICE_KERNELS,\n"
                          << "  compiled-in SPICE_KERNELS_DIR. Fetch kernels + metakernels with:\n"
                          << "  python3 scripts/fetch_kernels.py" << fetch_bodies_flag
                          << "  (see INSTALL.md \"SPICE kernel data\")."
                          << std::endl;
                // Try falling back to default meta-kernel if different
                if (meta_kernel_filename_ != "europa.tm") {
                    std::cerr << "Attempting to fall back to default europa.tm" << std::endl;
                    meta_kernel = kernels_path_ + "/europa.tm";
                    
                    if (!std::filesystem::exists(meta_kernel)) {
                        std::cerr << "Error: Default SPICE meta-kernel not found: " << meta_kernel << std::endl;
                        return false;
                    }
                } else {
                    return false;
                }
            }
            
            furnsh_c(meta_kernel.c_str());

            // Fail loud if furnsh could not load the metakernel or any kernel it
            // references. Under erract=RETURN a missing file (e.g. europa.tm's
            // jup365.bsp, which the default Moon kernel fetch omits) does NOT throw:
            // it latches a SPICE error and leaves subsequent str2et_c/spkpos_c as
            // silent no-ops, producing a garbage epoch. Do not cache a failed init
            // (leave spice_initialized false) and return false so the caller aborts
            // instead of running with an uninitialized epoch (BUGS.md PTM-27).
            if (failed_c()) {
                SpiceChar short_msg[81] = {0};
                SpiceChar long_msg[1841] = {0};
                getmsg_c("SHORT", 81, short_msg);
                getmsg_c("LONG", 1841, long_msg);
                reset_c();
                std::cerr << "Error: SPICE furnsh_c failed while loading metakernel '"
                          << meta_kernel << "': " << short_msg << "\n"
                          << "  " << long_msg << "\n"
                          << "  A referenced kernel is missing or unresolved. Populate the kernel\n"
                          << "  directory with:  python3 scripts/fetch_kernels.py" << fetch_bodies_flag << "\n"
                          << "  (see INSTALL.md, section \"SPICE kernel data\")." << std::endl;
                return false;
            }

            // Only print detailed messages if verbose mode is enabled
            if (verbose) {
                std::cout << "Loaded SPICE meta-kernel: " << meta_kernel_filename_ << std::endl;
                std::cout << "Set planetary body to " << target_body_name_
                        << " with primary " << primary_body_name_
                        << " using meta-kernel " << meta_kernel_filename_ << std::endl;
            }
            // Only run diagnostic checks if verbose mode is enabled
            if (verbose) {
                debugSpice();
            }

            spice_initialized = true;
            return true;
        } catch (const std::exception& e) {
            std::cerr << "SPICE initialization error: " << e.what() << std::endl;
            return false;
        }
    }
    return true;
}

double IlluminationModel::convertTimeToET(const std::string& time_str) {
    // Convert time string to SPICE ephemeris time
    double et;
    str2et_c(time_str.c_str(), &et);
    return et;
}

std::array<double, 3> IlluminationModel::getSunPosition(double et) {
    if (use_keplerian_ && keplerian_propagator_ && body_fixed_frame_) {
        // Keplerian mode: compute body position from orbital elements,
        // then get Sun direction in body-fixed frame
        double t_sec = keplerian_propagator_->etToSecondsFromEpoch(et);
        auto body_pos = keplerian_propagator_->getPosition(t_sec);
        return body_fixed_frame_->getSunInBodyFixed(t_sec, body_pos);
    }

    // SPICE mode
    double position[3];
    double lt;

    // Use the configurable target body and frame
    spkpos_c("SUN", et, target_body_frame_.c_str(), "LT+S",
             target_body_name_.c_str(), position, &lt);

    // Check if SPICE failed (e.g. SPK data doesn't cover this epoch).
    // Fall back to Keplerian propagation for out-of-range times.
    if (failed_c()) {
        reset_c();
        if (keplerian_propagator_ && body_fixed_frame_) {
            double t_sec = keplerian_propagator_->etToSecondsFromEpoch(et);
            auto body_pos = keplerian_propagator_->getPosition(t_sec);
            return body_fixed_frame_->getSunInBodyFixed(t_sec, body_pos);
        }
        return {{0.0, 0.0, 0.0}};
    }

    // For generic SPICE bodies (HORIZONS SPK), we get the Sun position in
    // ECLIPJ2000 (inertial) and must rotate to body-fixed using the spin state.
    // Built-in bodies use IAU body-fixed frames directly from SPICE.
    if (body_fixed_frame_ && !use_keplerian_) {
        // spkpos_c returns Sun relative to body in ECLIPJ2000.
        // getSunInBodyFixed expects body position relative to Sun in ecliptic J2000.
        // body_pos_ecliptic = -(Sun relative to body) = body relative to Sun.
        std::array<double, 3> body_pos = {{-position[0], -position[1], -position[2]}};

        // Convert SPICE et to seconds from spin epoch
        double spin_epoch_et = (body_fixed_frame_->spinState().epoch_jd - 2451545.0) * 86400.0;
        double t_sec = et - spin_epoch_et;

        return body_fixed_frame_->getSunInBodyFixed(t_sec, body_pos);
    }

    return {{position[0], position[1], position[2]}};
}

std::array<double, 3> IlluminationModel::getPrimaryPosition(double et) {
    // For heliocentric bodies without a primary, return zero vector
    if (!has_primary_body_ || use_keplerian_) {
        return {{0.0, 0.0, 0.0}};
    }

    // SPICE mode
    double position[3];
    double lt;

    // Use the configurable primary and target bodies
    spkpos_c(primary_body_name_.c_str(), et, target_body_frame_.c_str(), "LT+S",
             target_body_name_.c_str(), position, &lt);
    checkSpiceError("getPrimaryPosition");

    return {{position[0], position[1], position[2]}};
}

// Helper method for SPICE in IlluminationModel class
void IlluminationModel::checkSpiceError(const std::string& operation) {
    if (failed_c()) {
        SpiceChar short_msg[81];
        SpiceChar long_msg[241];
        
        getmsg_c("SHORT", 80, short_msg);
        getmsg_c("LONG", 240, long_msg);
        
        std::cerr << "SPICE error in " << operation << ": " << short_msg << std::endl;
        std::cerr << long_msg << std::endl;
        
        // Clear the error state
        reset_c();
    }
}

bool IlluminationModel::debugSpice() {
    std::cout << "==== SPICE Debug Information ====" << std::endl;
    
    // Test body ID lookups
    SpiceInt sunID, europaID, jupiterID;
    SpiceBoolean found;
    
    // Try all possible name formats
    bodn2c_c("SUN", &sunID, &found);
    std::cout << "Body lookup 'SUN': " << (found ? "Found (ID: " + std::to_string(sunID) + ")" : "Not found") << std::endl;
    
    bodn2c_c("10", &sunID, &found);
    std::cout << "Body lookup '10': " << (found ? "Found (ID: " + std::to_string(sunID) + ")" : "Not found") << std::endl;
    
    bodn2c_c("EUROPA", &europaID, &found);
    std::cout << "Body lookup 'EUROPA': " << (found ? "Found (ID: " + std::to_string(europaID) + ")" : "Not found") << std::endl;
    
    bodn2c_c("502", &europaID, &found);
    std::cout << "Body lookup '502': " << (found ? "Found (ID: " + std::to_string(europaID) + ")" : "Not found") << std::endl;
    
    // Test frame lookups
    SpiceInt j2000ID, iauEuropaID;
    
    namfrm_c("J2000", &j2000ID);
    std::cout << "Frame lookup 'J2000': " << (j2000ID != 0 ? "Found (ID: " + std::to_string(j2000ID) + ")" : "Not found") << std::endl;
    
    namfrm_c("IAU_EUROPA", &iauEuropaID);
    std::cout << "Frame lookup 'IAU_EUROPA': " << (iauEuropaID != 0 ? "Found (ID: " + std::to_string(iauEuropaID) + ")" : "Not found") << std::endl;
    
    std::cout << "==== End Debug Information ====" << std::endl;
    return true;
}

bool IlluminationModel::loadPositionsFromFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Error: Unable to open positions file " << filename << std::endl;
        return false;
    }
    
    std::string line;
    int line_number = 0;
    
    // Skip header lines that start with #
    while (std::getline(file, line)) {
        line_number++;
        if (!line.empty() && line[0] != '#') {
            break;
        }
    }
    
    // Reset file position
    file.clear();
    file.seekg(0);
    
    // Skip header lines again
    for (int i = 0; i < line_number - 1; i++) {
        std::getline(file, line);
    }
    
    // Parse data
    int expected_time_steps = time_steps_;
    sun_positions_.clear();
    primary_positions_.clear();
    times_.clear();
    
    int time_index;
    double et, sun_x, sun_y, sun_z, primary_x, primary_y, primary_z;
    char comma;

    while (file >> time_index >> comma >> et >> comma >> sun_x >> comma >> sun_y >> comma >>
           sun_z >> comma >> primary_x >> comma >> primary_y >> comma >> primary_z) {

        times_.push_back(et);
        sun_positions_.push_back({{sun_x, sun_y, sun_z}});
        primary_positions_.push_back({{primary_x, primary_y, primary_z}});
    }
    
    if (sun_positions_.size() != expected_time_steps) {
        std::cerr << "Warning: Expected " << expected_time_steps << " time steps, but loaded " 
                  << sun_positions_.size() << std::endl;
    }
    
    // Convert ephemeris times to strings
    time_strings_.resize(times_.size());
    for (size_t i = 0; i < times_.size(); i++) {
        // Format time as ISO string (simplified approach)
        std::stringstream ss;
        ss << "Time " << i;
        time_strings_[i] = ss.str();
    }
    
    precomputed_positions_loaded_ = true;
    std::cout << "Loaded " << sun_positions_.size() << " positions from file." << std::endl;
    return true;
}

double IlluminationModel::dotProduct(const std::array<double, 3>& v1, const std::array<double, 3>& v2) const {
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

std::array<double, 3> IlluminationModel::normalize(const std::array<double, 3>& v) const {
    double length = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    
    if (length < constants::EPSILON) {
        return {{0.0, 0.0, 0.0}}; // Return zero vector if the input is too short
    }
    
    return {{
        v[0] / length,
        v[1] / length,
        v[2] / length
    }};
}

std::array<double, 3> IlluminationModel::calculateFaceNormal(
    const std::array<double, 3>& v1,
    const std::array<double, 3>& v2,
    const std::array<double, 3>& v3) {
    
    // Calculate two edges of the triangle
    std::array<double, 3> edge1 = {{
        v2[0] - v1[0],
        v2[1] - v1[1],
        v2[2] - v1[2]
    }};
    
    std::array<double, 3> edge2 = {{
        v3[0] - v1[0],
        v3[1] - v1[1],
        v3[2] - v1[2]
    }};
    
    // Calculate cross product of edges
    std::array<double, 3> normal = {{
        edge1[1] * edge2[2] - edge1[2] * edge2[1],
        edge1[2] * edge2[0] - edge1[0] * edge2[2],
        edge1[0] * edge2[1] - edge1[1] * edge2[0]
    }};
    
    // Normalize the resulting vector
    return normalize(normal);
}

double IlluminationModel::calculateShadowFlux(
    const std::array<double, 3>& face_center,
    const std::array<double, 3>& face_normal,
    const std::array<double, 3>& sun_position) const {
    
    // Calculate sun direction and distance
    std::array<double, 3> sun_rel = {
        sun_position[0] - face_center[0],
        sun_position[1] - face_center[1],
        sun_position[2] - face_center[2]
    };
    
    double sun_dist = vectorLength(sun_rel);
    
    if (sun_dist < constants::EPSILON) {
        return 0.0;
    }
    
    std::array<double, 3> sun_dir = {
        sun_rel[0] / sun_dist,
        sun_rel[1] / sun_dist,
        sun_rel[2] / sun_dist
    };
    
    // Calculate solar elevation angle (angle from horizontal plane)
    // For a spherical body, the local "up" is the normalized face center
    std::array<double, 3> up_dir = face_center;
    double body_radius = vectorLength(up_dir);
    if (body_radius > constants::EPSILON) {
        up_dir[0] /= body_radius;
        up_dir[1] /= body_radius;
        up_dir[2] /= body_radius;
    }
    
    // Solar elevation is the angle between sun direction and horizontal plane
    // sin(elevation) = dot product of sun direction with local up
    double sin_elevation = dotProduct(sun_dir, up_dir);
    
    // If sun is below local horizon, no shadow flux
    if (sin_elevation <= 0.0) {
        return 0.0;
    }
    
    // Calculate the f parameter based on crater geometry
    // From the paper: f = 1/(1 + D²/4) where D is diameter/depth ratio
    double D = crater_diameter_depth_ratio_;
    double f = 1.0 / (1.0 + D * D / 4.0);
    
    // Calculate solar flux at body distance
    double sun_dist_au = sun_dist / constants::KM_PER_AU;
    double solar_flux = constants::SOLAR_CONSTANT / (sun_dist_au * sun_dist_au);
    
    // Use the body's albedo and emissivity
    double A = body_albedo_;
    double e = body_emissivity_;
    
    double shadow_flux = 0.0;
    
    if (D > 10.0) {
        // Use simplified equation (10) for large D (shallow craters)
        // σT⁴ = F₀ sin(e₀) * (1-A)/(D²/4) * [1 + A(1-f)/e]
        double D2_4 = D * D / 4.0;
        double term1 = (1.0 - A) / D2_4;
        double term2 = 1.0 + A * (1.0 - f) / e;
        shadow_flux = solar_flux * sin_elevation * term1 * term2;
    } else {
        // Use full equation (8) for more accurate calculation
        double Af = A * f;
        double denominator_solar = 1.0 - Af;
        
        // Solar scattering term
        double solar_term = solar_flux * sin_elevation * (1.0 - A) * f / denominator_solar;
        
        // Multiple scattering enhancement
        double scattering_enhancement = 1.0 + Af / denominator_solar;
        
        // For the thermal term, we'd need wall temperatures
        // For now, we'll use a simplified approach
        double ir_contribution = 0.0;
        
        shadow_flux = solar_term * scattering_enhancement + ir_contribution;
    }
    
    return std::max(0.0, shadow_flux);
}


// Implementation of the finite solar disk model with shadow flux
double IlluminationModel::calculateSolarIlluminationWithFiniteDisk(
    const std::array<double, 3>& face_center,
    const std::array<double, 3>& face_normal,
    const std::array<double, 3>& sun_position,        // SPICE in km
    const std::array<double, 3>& primary_position,    // SPICE in km
    int num_samples,
    int src_face)                                     // PTM-13 P1: source core-face id (-1 = no self-skip)
{
    // PTM-13 P1: topological self-skip is active only with a valid mesh source face id (the lat/lon
    // point path passes -1). Default OFF -> byte-identical. PTM-13 rise-epsilon is the independent
    // rise-based discriminator (takes precedence when both are set). Either starts the ray AT the face.
    const bool selfskip = use_topo_selfskip_ && src_face >= 0;
    const bool riseeps  = use_rise_eps_ && src_face >= 0;
    const bool at_face  = selfskip || riseeps;
    // Convert SPICE positions to meters for geometric tests
    const std::array<double,3> sun_pos_m     = kmToMeters(sun_position);
    // PTM-36: the eclipse sphere must live in MESH-NATIVE units (km for global
    // icosphere meshes, m for DEM/site meshes) -- center AND radius together.
    const double km_to_native = kmToNativeScale(face_center);
    const std::array<double,3> primary_pos_n = {
        primary_position[0] * km_to_native,
        primary_position[1] * km_to_native,
        primary_position[2] * km_to_native
    };
    const double primary_radius_n = primary_radius_ * km_to_native;

    // Geometry in meters: vector from face to Sun (meters)
    std::array<double, 3> sun_rel_m = {
        sun_pos_m[0] - face_center[0],
        sun_pos_m[1] - face_center[1],
        sun_pos_m[2] - face_center[2]
    };
    double sun_dist_m = vectorLength(sun_rel_m);
    if (sun_dist_m <= constants::EPSILON) {
        return enable_shadow_flux_ ? calculateShadowFlux(face_center, face_normal, sun_pos_m) : 0.0;
    }
    std::array<double, 3> sun_dir = { sun_rel_m[0]/sun_dist_m, sun_rel_m[1]/sun_dist_m, sun_rel_m[2]/sun_dist_m };
    // PTM-36: ray parameters t from raySphereIntersect are in MESH-NATIVE units
    // (sun_dir is a unit vector), so the "nearer than the Sun" bound must be too.
    const double sun_dist_n = sun_dist_m * (km_to_native / 1000.0);

    // Solar angular radius uses km distance
    const std::array<double,3> face_center_km = metersToKm(face_center);
    std::array<double,3> sun_rel_km = {
        sun_position[0] - face_center_km[0],
        sun_position[1] - face_center_km[1],
        sun_position[2] - face_center_km[2]
    };
    double sun_dist_km = std::sqrt(sun_rel_km[0]*sun_rel_km[0] + sun_rel_km[1]*sun_rel_km[1] + sun_rel_km[2]*sun_rel_km[2]);
    double s = constants::SUN_RADIUS / sun_dist_km; // SUN_RADIUS is in km
    if (s > 1.0) s = 1.0;
    const double disk_ang   = std::asin(s);
    const double disk_scale = std::sin(disk_ang);

    // Finite-disk-aware outward-facing test
    double cos_inc0 = dotProduct(face_normal, sun_dir);
    if (cos_inc0 <= -std::sin(disk_ang)) {
        return enable_shadow_flux_ ? calculateShadowFlux(face_center, face_normal, sun_pos_m) : 0.0;
    }

    // Analytical body-occlusion pre-check (same logic as batch/GPU paths).
    // TOPOLOGY GATE (not shape): run this cull ONLY on OPEN patch meshes. On a
    // CLOSED (watertight) mesh the Embree BVH holds the full body and self-
    // occludes, so this analytic sphere test is unnecessary and over-culls
    // sunward-tilted grazing facets on shaped bodies -- disable it and let Embree
    // do all occlusion (matching the GPU path, correct on closed bodies). On an
    // OPEN micro-patch (lunar/Europa) the BVH has no far-side geometry, so shadow
    // rays would pass THROUGH the bulk body; here the analytic body-sphere is the
    // SOLE self-occlusion guard.
    //
    // Cull only when the body SPHERE actually lies BETWEEN the face and the sun,
    // i.e. the forward ray (origin=fc, dir=sun) intersects the body sphere at
    // t>0. This forward test -- NOT a bare |fc x sun| < R_occ line test -- is
    // what distinguishes a genuine far-side/behind-the-limb face from a sunward-
    // tilted face just past the geometric terminator whose grazing ray only comes
    // near the near limb going AWAY from the body. The old bare-line test (with a
    // 0.998*|fc| inset that masked it) wrongly culled mu>0 sunward facets once the
    // inset was tightened. R_occ = body-level radius (min core-face centroid
    // radius), which is relief-adaptive and never exceeds a face's own radius, so
    // no lit face is culled. Ungated by mu -- the forward-intersection condition
    // already excludes lit sunward facets. See mesh_is_open_ / body_radius_occ_
    // notes and memory ptm-mu-gate-open-mesh-selfocclusion-fail.
    if (mesh_is_open_ && body_radius_occ_native_ > 0.0) {
        const double fc_dot_sun = face_center[0]*sun_dir[0] + face_center[1]*sun_dir[1] + face_center[2]*sun_dir[2];
        if (fc_dot_sun < 0.0) {  // cheap early-out; a forward hit implies this
            const std::array<double,3> body_center = {0.0, 0.0, 0.0};
            double tb1, tb2;
            if (raySphereIntersect(face_center, sun_dir, body_center,
                                   body_radius_occ_native_, tb1, tb2)) {
                if (tb1 > 0.0 || tb2 > 0.0) {
                    return enable_shadow_flux_ ? calculateShadowFlux(face_center, face_normal, sun_pos_m) : 0.0;
                }
            }
        }
    }

    // PTM-13: local-horizon (skyline) cull of the grazing DIRECT sun. Azimuth-specific;
    // catches the over-tessellation/grazing leak where the single Embree ray hops the mm
    // rim. Gated (default OFF -> byte-identical). CPU path; built in initializeScene().
    if (use_horizon_cull_ && horizon_direct_.built()) {
        const std::array<double,3> p_local = {
            face_center[0] - (scene_has_offset_ ? scene_origin_offset_[0] : 0.0),
            face_center[1] - (scene_has_offset_ ? scene_origin_offset_[1] : 0.0),
            face_center[2] - (scene_has_offset_ ? scene_origin_offset_[2] : 0.0)
        };
        if (horizon_direct_.sunBelowLocalHorizon(p_local, sun_dir))
            return enable_shadow_flux_ ? calculateShadowFlux(face_center, face_normal, sun_pos_m) : 0.0;
    }

    // Local up and elevation
    std::array<double, 3> up = face_center;
    double r_up = vectorLength(up);
    if (r_up > constants::EPSILON) { up[0]/=r_up; up[1]/=r_up; up[2]/=r_up; }
    const double sin_elev = dotProduct(sun_dir, up);

    // Robust origin offsets (meters). PTM-13 P1/rise-epsilon: under self-skip or rise-epsilon the ray
    // starts AT the face (eps_n=eps_d=0) and self/self-relief hits are rejected by primID / out-of-plane
    // rise. Default OFF -> unchanged (2x/1x).
    const double eps_base = calculateSafeRayOffset(face_center, face_normal);
    const double eps_n = at_face ? 0.0 : 2.0 * eps_base;
    const double eps_d = at_face ? 0.0 : 1.0 * eps_base;

    std::array<double, 3> ray_origin_base = {
        face_center[0] + face_normal[0] * eps_n + sun_dir[0] * eps_d,
        face_center[1] + face_normal[1] * eps_n + sun_dir[1] * eps_d,
        face_center[2] + face_normal[2] * eps_n + sun_dir[2] * eps_d
    };

    // PTM-38: is the primary anywhere on the solar disk (penumbra OR umbra)?
    // See primaryDiskOverlap(). False outside an eclipse -> byte-identical.
    const bool eclipse_partial = (!primary_is_sun_) &&
        primaryDiskOverlap(ray_origin_base, sun_dir, primary_pos_n, primary_radius_n,
                           sun_dist_n, disk_ang);

    // Center-ray fast path; use meters for eclipse/horizon
    // PTM-38: !eclipse_partial vetoes the fast path for the WHOLE eclipse. The
    // old center-ray test broke the penumbra at both ends: before the primary
    // covered the disk center it returned FULL sun, and the moment it did it
    // returned ZERO -- while the limb of the Sun was still visible. Falling
    // through to the finite-disk sampler is correct at both ends, and in full
    // umbra the sampler returns exactly the same 0 / shadow-flux this branch did
    // (every sample is eclipse-skipped, so sum_cos == 0).
    if (!eclipse_partial && sin_elev > 2.0 * std::sin(disk_ang) && cos_inc0 > 0.10) {
        if (use_geometric_horizon_ && checkLunarHorizon(ray_origin_base, sun_dir, vectorLength(face_center))) {
            return enable_shadow_flux_ ? calculateShadowFlux(face_center, face_normal, sun_pos_m) : 0.0;
        }
        // Terrain occlusion via Embree. PTM-13: rise-epsilon / self-skip re-trace when active (default OFF).
        {
            std::vector<std::array<double,3>> o(1, ray_origin_base);
            std::vector<std::array<double,3>> d(1, sun_dir);
            std::vector<uint8_t> occ;
            if (riseeps)
                occ = traceOccludedRiseEps(o, d, std::vector<int>(1, src_face),
                                           std::vector<std::array<double,3>>(1, face_normal));
            else if (selfskip)
                occ = traceOccludedSelfSkip(o, d, std::vector<int>(1, src_face));
            else
                occ = traceOccluded(o, d, /*coherent=*/true);
            if (!occ.empty() && occ[0] == 0) {
                double sun_dist_au = sun_dist_km / constants::KM_PER_AU;
                double solar_irr = constants::SOLAR_CONSTANT / (sun_dist_au * sun_dist_au);
                return solar_irr * cos_inc0;
            }
        }
        // If center ray blocked, fall back to finite-disk sampling
    }

    // Disk basis with decorrelation
    std::array<double, 3> u, v; createCoordinateSystem(sun_dir, u, v);
    uint32_t seed = hashFaceSeed(face_center, face_normal, sun_position);
    double theta = (seed & 0xFFFFu) * (2.0*M_PI / 65536.0);
    std::array<double,3> ur, vr; rotateBasis(u, v, theta, ur, vr);
    double uoff = ((seed >> 16) & 0xFFu) / 256.0;
    double voff = ((seed >> 24) & 0xFFu) / 256.0;

    // Irradiance with km
    double sun_dist_au = sun_dist_km / constants::KM_PER_AU;
    double solar_irr = constants::SOLAR_CONSTANT / (sun_dist_au * sun_dist_au);

    const uint32_t baseN   = static_cast<uint32_t>(std::max(1, num_samples));
    uint32_t heavyN        = heavySolarSamples(num_samples);   // PTM-45(ii)
    // PTM-38: in the penumbra, 8 samples quantize the eclipsed fraction at
    // ~12.5 % (~21 K near half-obscuration). Start heavy when the primary
    // overlaps the disk. Outside an eclipse this is the old initialN exactly.
    uint32_t initialN      = eclipse_partial ? heavyN : std::max<uint32_t>(8, baseN);

    auto sample_and_accumulate = [&](uint32_t N, double& sum_cos_out,
                                     uint32_t& total_out, uint32_t& visible_out,
                                     uint32_t& blocked_visible_out)
    {
        sum_cos_out = 0.0;
        total_out = visible_out = blocked_visible_out = 0;

        std::vector<std::array<double, 3>> dirs;
        std::vector<uint8_t> skip;
        dirs.reserve(N);
        skip.reserve(N);

        for (uint32_t k = 0; k < N; ++k) {
            double su, sv; hammersley(k, N, uoff, voff, su, sv);
            double dx, dy; squareToConcentricDisk(su, sv, dx, dy);
            if (dx*dx + dy*dy > 1.0) continue;

            const double ox = disk_scale * dx, oy = disk_scale * dy;
            std::array<double, 3> sd = {
                sun_dir[0] + ox * ur[0] + oy * vr[0],
                sun_dir[1] + ox * ur[1] + oy * vr[1],
                sun_dir[2] + ox * ur[2] + oy * vr[2]
            };
            double l = vectorLength(sd);
            if (l <= constants::EPSILON) continue;
            sd[0]/=l; sd[1]/=l; sd[2]/=l;

            bool sskip = false;

            // Eclipse clip
            if (!primary_is_sun_) {
                double t1, t2;
                if (raySphereIntersect(ray_origin_base, sd, primary_pos_n, primary_radius_n, t1, t2)) {
                    if ((t1 > 0.0 && t1 < sun_dist_n) || (t2 > 0.0 && t2 < sun_dist_n)) sskip = true;
                }
            }

            skip.push_back(sskip ? 1 : 0);
            if (!sskip) dirs.push_back(sd);
            total_out++;
        }

        // Start each survivor ray slightly along its own direction too (eps_d=0 under self-skip)
        std::vector<std::array<double,3>> origins;
        origins.reserve(dirs.size());
        for (const auto& sd : dirs) {
            origins.push_back({ ray_origin_base[0] + sd[0]*eps_d,
                                ray_origin_base[1] + sd[1]*eps_d,
                                ray_origin_base[2] + sd[2]*eps_d });
        }

        // Terrain occlusion for survivors. PTM-13: rise-epsilon / self-skip re-trace (default OFF).
        std::vector<uint8_t> occ;
        if (riseeps)
            occ = traceOccludedRiseEps(origins, dirs, std::vector<int>(dirs.size(), src_face),
                                       std::vector<std::array<double,3>>(dirs.size(), face_normal));
        else if (selfskip)
            occ = traceOccludedSelfSkip(origins, dirs, std::vector<int>(dirs.size(), src_face));
        else
            occ = traceOccluded(origins, dirs, /*coherent=*/true);

        size_t m = 0;
        for (uint32_t sidx = 0; sidx < total_out; ++sidx) {
            if (skip[sidx]) continue;
            const bool blocked = (m < occ.size()) && (occ[m] != 0);
            const auto& sd = dirs[m];
            m++;
            if (blocked) {
                blocked_visible_out++;
            } else {
                double c = dotProduct(face_normal, sd);
                if (c > 0.0) { sum_cos_out += c; visible_out++; }
            }
        }
    };

    // First pass
    double sum_cos = 0.0;
    uint32_t total_samples = 0, visible_cnt = 0, blocked_visible_cnt = 0;
    sample_and_accumulate(initialN, sum_cos, total_samples, visible_cnt, blocked_visible_cnt);

    if (total_samples == 0) {
        return enable_shadow_flux_ ? calculateShadowFlux(face_center, face_normal, sun_pos_m) : 0.0;
    }

    // Upsample if partial terrain occlusion detected
    const bool partial_by_topo = (visible_cnt > 0 && blocked_visible_cnt > 0 && blocked_visible_cnt < visible_cnt);
    if (partial_by_topo && heavyN > initialN) {
        double sc2 = 0.0; uint32_t ts2=0, v2=0, bv2=0;
        sample_and_accumulate(heavyN, sc2, ts2, v2, bv2);
        if (ts2 > 0) { sum_cos = sc2; total_samples = ts2; }
    }

    double flux = solar_irr * (sum_cos / std::max<uint32_t>(1, total_samples));
    if (flux <= 0.0 && enable_shadow_flux_) {
        flux = calculateShadowFlux(face_center, face_normal, sun_pos_m);
    }
    return flux;
}

bool IlluminationModel::checkLunarHorizon(
    const std::array<double, 3>& ray_origin,
    const std::array<double, 3>& ray_direction,
    double lunar_radius) const {
    
    // Ray-sphere intersection with lunar body centered at origin
    std::array<double, 3> lunar_center = {{0.0, 0.0, 0.0}};
    double t1, t2;
    
    // Use existing ray-sphere intersection method
    if (raySphereIntersect(ray_origin, ray_direction, lunar_center, lunar_radius, t1, t2)) {
        // Check if intersection occurs in front of the ray origin
        // t1 is the near intersection, t2 is the far intersection
        
        // For horizon checking, we care about the nearest positive intersection
        double nearest_t = -1.0;
        if (t1 > 0.001) {  // Small epsilon to avoid self-intersection
            nearest_t = t1;
        } else if (t2 > 0.001) {
            nearest_t = t2;
        }
        
        if (nearest_t > 0.0) {
            // Ray hits lunar surface before reaching infinity
            return true;  // Blocked by lunar horizon
        }
    }
    
    return false;  // Ray escapes to space - sun is visible
}

bool IlluminationModel::calculateIlluminationFromMesh(
    const std::shared_ptr<Mesh>& mesh,
    const std::string& start_time,
    double duration_days) {
    
    // Ensure the mesh has pre-calculated face properties
    if (mesh->face_normals.empty() || mesh->face_centers.empty()) {
        std::cerr << "Error: Mesh face properties not calculated. Please call calculateFaceProperties first." << std::endl;
        return false;
    }
    
    // CRITICAL FIX: Create corrected copies of normals and centers
    // We need to ensure normals point outward for topographic meshes
    std::vector<std::array<double, 3>> corrected_normals = mesh->face_normals;
    const std::vector<std::array<double, 3>>& face_centers = mesh->face_centers;

    // PTM-Class1 fix (env PTM_ILLUM_TRUE_NORMALS): replace the per-face radial flip -- which REVERSES
    // genuine overhang faces (true winding normal radial-negative), manufacturing mu>0 and spurious
    // direct sun on the underside of ledges / rock-base slope breaks -- with a topology-propagated
    // consistent orientation. On a consistently-wound manifold the winding normal IS the true outward
    // (up to a per-connected-component global sign, resolved by majority radial-outward vote, which
    // equals seeding from a definitely-up face and BFS across shared edges). The radial test only
    // breaks ties per disconnected component. Default OFF -> byte-identical (radial flip below runs
    // verbatim). Phase-2 census (scripts/ptm_illum_deepdive/phase2_overhang_census.csv): winding
    // globally consistent (0/763,839 inconsistent manifold edges); 15,476 overhangs (3.0%), 90.1% of
    // the >380K radoff hot tail. Applies to the ILLUMINATION normal only (direct-solar cos_inc + the
    // rise-eps source normal); thermal-emission and view-factor normals are unchanged (radoff energy
    // balance depends on the normal only via direct_solar, so this is the complete radoff fix; the VF
    // path would need a rebuild and is out of scope).
    bool use_true_normals = false;
    if (const char* tn = std::getenv("PTM_ILLUM_TRUE_NORMALS"))
        if (tn[0]=='1'||tn[0]=='t'||tn[0]=='T'||tn[0]=='y'||tn[0]=='Y') use_true_normals = true;

    if (use_true_normals) {
        const size_t nfc = mesh->faces.size();
        // 1. raw winding normals from the mesh's own vertex order (BEFORE any flip)
        std::vector<std::array<double,3>> nwv(nfc);
        for (size_t i = 0; i < nfc; ++i) {
            const auto& f = mesh->faces[i];
            const auto& a = mesh->vertices[f[0]];
            const auto& b = mesh->vertices[f[1]];
            const auto& c = mesh->vertices[f[2]];
            std::array<double,3> e1 = {{b[0]-a[0], b[1]-a[1], b[2]-a[2]}};
            std::array<double,3> e2 = {{c[0]-a[0], c[1]-a[1], c[2]-a[2]}};
            std::array<double,3> n = {{ e1[1]*e2[2]-e1[2]*e2[1],
                                        e1[2]*e2[0]-e1[0]*e2[2],
                                        e1[0]*e2[1]-e1[1]*e2[0] }};
            double L = std::sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]);
            if (L > 0) { n[0]/=L; n[1]/=L; n[2]/=L; }
            nwv[i] = n;
        }
        // 2. union-find connected components over shared edges + winding-consistency audit
        std::vector<int> parent(nfc);
        for (size_t i = 0; i < nfc; ++i) parent[i] = static_cast<int>(i);
        auto uf_find = [&parent](int x){ while (parent[x]!=x){ parent[x]=parent[parent[x]]; x=parent[x]; } return x; };
        auto uf_union = [&](int a,int b){ int ra=uf_find(a), rb=uf_find(b); if (ra!=rb) parent[ra]=rb; };
        std::unordered_map<uint64_t,std::pair<int,int>> emap;  // undirected edge -> (last face, dir)
        emap.reserve(nfc*3);
        size_t inconsistent = 0;
        for (size_t i = 0; i < nfc; ++i) {
            const auto& f = mesh->faces[i];
            const int vv[3] = { f[0], f[1], f[2] };
            for (int e = 0; e < 3; ++e) {
                int s = vv[e], d = vv[(e+1)%3];
                int dir = (s < d) ? 1 : -1;
                uint32_t lo = static_cast<uint32_t>(s < d ? s : d);
                uint32_t hi = static_cast<uint32_t>(s < d ? d : s);
                uint64_t key = (static_cast<uint64_t>(lo) << 32) | hi;
                auto it = emap.find(key);
                if (it == emap.end()) { emap.emplace(key, std::make_pair(static_cast<int>(i), dir)); }
                else {
                    uf_union(static_cast<int>(i), it->second.first);
                    if (it->second.second == dir) ++inconsistent;  // same directed traversal => inconsistent
                    it->second.first = static_cast<int>(i);        // track last (tolerate >2 fans)
                }
            }
        }
        // 3. per-component majority-outward global sign (== seed-from-up-face BFS on a consistent mesh)
        std::unordered_map<int,long> vote;
        for (size_t i = 0; i < nfc; ++i) {
            const auto& c = face_centers[i];
            double dr = c[0]*nwv[i][0] + c[1]*nwv[i][1] + c[2]*nwv[i][2];
            vote[uf_find(static_cast<int>(i))] += (dr >= 0.0) ? 1 : -1;
        }
        // 4. winding-consistency gate: a junk (non-manifold / inconsistent) winding cannot be oriented
        //    topologically -> WARN LOUDLY and fall back to the radial flip (never silently proceed).
        if (inconsistent > nfc / 100) {
            std::cout << "[IlluminationModel] PTM-Class1 TRUE normals REFUSED: winding INCONSISTENT ("
                      << inconsistent << " same-direction manifold half-edges > 1% of " << nfc
                      << " faces); falling back to the radial-flip orientation.\n";
            use_true_normals = false;
        } else {
            int overhang = 0;
            for (size_t i = 0; i < nfc; ++i) {
                double S = (vote[uf_find(static_cast<int>(i))] >= 0) ? 1.0 : -1.0;
                corrected_normals[i] = {{ S*nwv[i][0], S*nwv[i][1], S*nwv[i][2] }};
                const auto& c = face_centers[i];
                if (c[0]*corrected_normals[i][0] + c[1]*corrected_normals[i][1]
                        + c[2]*corrected_normals[i][2] < 0.0)
                    ++overhang;
            }
            std::cout << "[IlluminationModel] PTM-Class1 TRUE (winding-consistent) normals ENABLED "
                         "(env PTM_ILLUM_TRUE_NORMALS): winding "
                      << (inconsistent==0 ? "GLOBALLY CONSISTENT" : "near-consistent") << " ("
                      << inconsistent << " inconsistent half-edges); " << overhang << " overhang faces ("
                      << std::fixed << std::setprecision(2)
                      << (100.0*overhang/std::max<size_t>(nfc,1)) << "%) keep their true radial-INWARD "
                         "normal instead of being flipped outward -> darkens spurious direct sun on "
                         "ledge undersides / rock-base slope breaks (Class 1). Illumination normal only; "
                         "thermal/VF normals unchanged.\n";
        }
    }

    // Radial-flip orientation (default; and the fallback when TRUE normals are refused). Byte-identical
    // to the historical path whenever PTM_ILLUM_TRUE_NORMALS is unset.
    if (!use_true_normals) {
        // Check and fix normal orientations for topographic meshes
        // This is essential for correct primary IR and reflection calculations
        int flipped_count = 0;
        for (size_t i = 0; i < corrected_normals.size(); i++) {
            const auto& center = face_centers[i];
            auto& normal = corrected_normals[i];

            // For a mesh centered at the origin, the face center vector
            // should point in the same general direction as the normal (both outward)
            double dot = center[0] * normal[0] +
                         center[1] * normal[1] +
                         center[2] * normal[2];

            // If dot product is negative, the normal points inward
            if (dot < 0) {
                // Flip the normal to point outward
                normal[0] = -normal[0];
                normal[1] = -normal[1];
                normal[2] = -normal[2];
                flipped_count++;
            }
        }

        if (flipped_count > 0) {
            std::cout << "Corrected " << flipped_count << " inward-pointing normals ("
                      << std::fixed << std::setprecision(1)
                      << (100.0 * flipped_count / corrected_normals.size())
                      << "% of faces)" << std::endl;
        }
    }
    
    // Initialize Embree for ray tracing
    if (!initializeEmbree() || !initializeScene(mesh->vertices, mesh->faces)) {
        std::cerr << "Failed to initialize ray tracing." << std::endl;
        return false;
    }
    
    // Get the number of faces for allocation
    size_t num_faces = mesh->faces.size();
    
    // Start timer for performance measurement
    auto start_time_point = std::chrono::high_resolution_clock::now();
    
    // Initialize SPICE if not already done
    if (!initialize_spice(debug_output_enabled_)) {
        std::cerr << "Failed to initialize SPICE." << std::endl;
        return false;
    }
    
    // Set up time steps for simulation
    // External grid (non-uniform) overrides the uniform grid when set.
    const bool use_external_grid = !external_ets_.empty() &&
        static_cast<int>(external_ets_.size()) == time_steps_;
    double et_start = 0.0;
    double et_end = 0.0;
    double et_step = 0.0;
    if (use_external_grid) {
        et_start = external_ets_.front();
        et_end   = external_ets_.back();
        et_step  = 0.0;  // not used in this branch
    } else {
        et_start = convertTimeToET(start_time);
        et_end = et_start + duration_days * constants::SECONDS_PER_DAY;
        et_step = (et_end - et_start) / (time_steps_ - 1);
    }
    
    // Check if we should use chunked storage for very large meshes
    using_chunked_storage_ = (num_faces * time_steps_ > 100000000); // Over 100M elements
    
    // Initialize storage for results
    if (using_chunked_storage_) {
        initializeChunkedStorage(num_faces, time_steps_);
    } else {
        // Resize all result vectors
        illumination_values_.resize(num_faces * time_steps_, 0.0);
        direct_solar_values_.resize(num_faces * time_steps_, 0.0);
        primary_reflected_values_.resize(num_faces * time_steps_, 0.0);
        primary_ir_values_.resize(num_faces * time_steps_, 0.0);
    }
    
    // Resize and calculate ephemeris data
    times_.resize(time_steps_);
    time_strings_.resize(time_steps_);
    sun_positions_.resize(time_steps_);
    primary_positions_.resize(time_steps_);
    
    
    // std::ostringstream progress_prefix;
    // progress_prefix << "Computing " << time_steps_ << " time steps: ";
    // thermal::ProgressBar progress_bar(time_steps_, progress_prefix.str(), 50, show_progress_);
    
    // Generate time steps and get positions
    for (int t = 0; t < time_steps_; t++) {
        // Calculate ephemeris time
        double et = use_external_grid
            ? external_ets_[t]
            : (et_start + t * et_step);
        times_[t] = et;

        // Convert ET to string
        char time_string[100];
        et2utc_c(et, "ISOC", 3, 100, time_string);
        time_strings_[t] = std::string(time_string);

        // Get positions of sun and primary body relative to target body
        sun_positions_[t] = getSunPosition(et);
        primary_positions_[t] = getPrimaryPosition(et);

        //progress_bar.update(t + 1);
    }

    // progress_bar.finish();

    // Calculate illumination for each time step with CORRECTED face properties
    thermal::ProgressBar illum_progress(time_steps_, "Calculating illumination: ", 50, show_progress_);
    
    for (int t = 0; t < time_steps_; t++) {
        // Get positions for this time step
        const auto& sun_position = sun_positions_[t];
        const auto& primary_position = primary_positions_[t];
        
        // Initialize temporary arrays for batch processing
        std::vector<double> direct_solar_batch(num_faces, 0.0);
        std::vector<double> primary_reflected_batch(num_faces, 0.0);
        std::vector<double> primary_ir_batch(num_faces, 0.0);
        
        // CRITICAL: Use corrected_normals instead of mesh->face_normals

        // Calculate direct solar illumination with finite disk model
#ifdef PTM_GPU_OPENCL
        // Use GPU batched version when GPU ray tracing is enabled
        if (use_gpu_ray_tracer_ && gpu_ray_pool_ && gpu_mesh_uploaded_) {
            calculateDirectSolarBatchWithFiniteDiskGPU(
                face_centers,                    // Use original face centers
                corrected_normals,                // Use CORRECTED normals
                sun_position,
                primary_position,
                direct_solar_batch
            );
        } else {
            calculateDirectSolarBatchWithFiniteDisk(
                face_centers,                    // Use original face centers
                corrected_normals,                // Use CORRECTED normals
                sun_position,
                primary_position,
                direct_solar_batch
            );
        }
#else
        calculateDirectSolarBatchWithFiniteDisk(
            face_centers,                    // Use original face centers
            corrected_normals,                // Use CORRECTED normals
            sun_position,
            primary_position,
            direct_solar_batch
        );
#endif
        
        // Calculate primary IR heating
        calculatePrimaryIRBatch(
            face_centers,                    // Use original face centers
            corrected_normals,                // Use CORRECTED normals
            primary_position, 
            primary_ir_batch
        );
        
        // Calculate primary-reflected solar light if enabled
        if (include_primary_reflection_) {
            calculatePrimaryReflectedBatch(
                face_centers,                // Use original face centers
                corrected_normals,            // Use CORRECTED normals
                sun_position, 
                primary_position, 
                primary_reflected_batch
            );
        }
        
        // Store results
        if (using_chunked_storage_) {
            // Store in chunked storage for large meshes
            for (size_t f = 0; f < num_faces; f++) {
                // Compute total illumination
                double total_illumination = direct_solar_batch[f] + 
                                          primary_reflected_batch[f];
                
                // Store using the chunked storage methods
                storeChunkedValue(illumination_chunks_, f, t, total_illumination);
                storeChunkedValue(direct_solar_chunks_, f, t, direct_solar_batch[f]);
                storeChunkedValue(primary_reflected_chunks_, f, t, primary_reflected_batch[f]);
                storeChunkedValue(primary_ir_chunks_, f, t, primary_ir_batch[f]);
            }
        } else {
            // Store in regular arrays
            for (size_t f = 0; f < num_faces; f++) {
                size_t idx = f * time_steps_ + t;
                
                // Compute total illumination
                double total_illumination = direct_solar_batch[f] + 
                                          primary_reflected_batch[f];
                // Note: Primary IR is typically not included in "illumination"
                // but stored separately
                
                illumination_values_[idx] = total_illumination;
                direct_solar_values_[idx] = direct_solar_batch[f];
                primary_reflected_values_[idx] = primary_reflected_batch[f];
                primary_ir_values_[idx] = primary_ir_batch[f];
            }
        }
        
        illum_progress.update(t + 1);
    }
    
    illum_progress.finish();
    
    // Calculate statistics for reporting
    double total_flux = 0.0;
    double max_flux = 0.0;
    double min_flux = std::numeric_limits<double>::max();
    double total_direct = 0.0;
    double total_reflected = 0.0;
    double total_ir = 0.0;
    double max_direct = 0.0;
    double max_reflected = 0.0;
    double max_ir = 0.0;
    
    size_t non_zero_count = 0;
    
    for (size_t f = 0; f < num_faces; f++) {
        for (int t = 0; t < time_steps_; t++) {
            double direct_val, reflected_val, ir_val;
            
            if (using_chunked_storage_) {
                direct_val = getDirectSolarValue(f, t);
                reflected_val = getPrimaryReflectedValue(f, t);
                ir_val = getPrimaryIRValue(f, t);
            } else {
                size_t idx = f * time_steps_ + t;
                direct_val = direct_solar_values_[idx];
                reflected_val = primary_reflected_values_[idx];
                ir_val = primary_ir_values_[idx];
            }
            
            double total = direct_val + reflected_val + ir_val;
            
            if (total > 0) {
                total_flux += total;
                total_direct += direct_val;
                total_reflected += reflected_val;
                total_ir += ir_val;
                non_zero_count++;
                
                max_flux = std::max(max_flux, total);
                max_direct = std::max(max_direct, direct_val);
                max_reflected = std::max(max_reflected, reflected_val);
                max_ir = std::max(max_ir, ir_val);
                
                if (total < min_flux) {
                    min_flux = total;
                }
            }
        }
    }
    
    // Report results
    double avg_flux = (non_zero_count > 0) ? total_flux / non_zero_count : 0.0;
    double avg_direct = (non_zero_count > 0) ? total_direct / non_zero_count : 0.0;
    double avg_reflected = (non_zero_count > 0) ? total_reflected / non_zero_count : 0.0;
    double avg_ir = (non_zero_count > 0) ? total_ir / non_zero_count : 0.0;
    
    // End timing
    auto end_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time_point - start_time_point);
    
    std::cout << "Illumination calculation completed in " 
              << std::fixed << std::setprecision(3) 
              << duration.count() / 1000.0 << " seconds." << std::endl;
    
    // Print summary statistics if not suppressed
    if (show_progress_) {
        std::cout << "Results:" << std::endl;
        std::cout << "  Average total flux: " << std::fixed << std::setprecision(9) 
                  << avg_flux << " W/m^2" << std::endl;
        std::cout << "    - Direct solar: " << avg_direct << " W/m^2" << std::endl;
        std::cout << "    - Primary reflected: " << avg_reflected << " W/m^2" << std::endl;
        std::cout << "    - Primary IR: " << avg_ir << " W/m^2" << std::endl;
        std::cout << "  Maximum total flux: " << max_flux << " W/m^2" << std::endl;
        std::cout << "    - Direct solar: " << max_direct << " W/m^2" << std::endl;
        std::cout << "    - Primary reflected: " << max_reflected << " W/m^2" << std::endl;
        std::cout << "    - Primary IR: " << max_ir << " W/m^2" << std::endl;
        std::cout << "  Minimum flux (non-zero): " 
                  << ((min_flux < std::numeric_limits<double>::max()) ? min_flux : 0.0) 
                  << " W/m^2" << std::endl;
    }
    
    return true;
}

// Batch processing for direct solar illumination with finite disk model
void IlluminationModel::calculateDirectSolarBatchWithFiniteDisk(
    const std::vector<std::array<double, 3>>& face_centers,
    const std::vector<std::array<double, 3>>& face_normals,
    const std::array<double, 3>& sun_position,
    const std::array<double, 3>& primary_position,
    std::vector<double>& direct_solar_values,
    int num_samples)
{
    const size_t n = face_centers.size();
    direct_solar_values.assign(n, 0.0);
    if (n == 0) return;

    // Convert SPICE positions from km to meters for direction computation
    // (SPICE returns km; the Sun is far enough that the mesh-unit mismatch on
    // global km meshes is a ~1e-9 direction error, and sun_dist_km below is exact)
    const std::array<double,3> sun_pos_m = kmToMeters(sun_position);
    // PTM-36: the eclipse sphere must live in MESH-NATIVE units (km for global
    // icosphere meshes, m for DEM/site meshes) -- center AND radius together.
    const double km_to_native = kmToNativeScale(face_centers[0]);
    const std::array<double,3> primary_pos_n = {
        primary_position[0] * km_to_native,
        primary_position[1] * km_to_native,
        primary_position[2] * km_to_native
    };
    const double primary_radius_n = primary_radius_ * km_to_native;

    auto hash32 = [](uint32_t x) -> uint32_t {
        x ^= x >> 17; x *= 0xed5ad4bbU; x ^= x >> 11; x *= 0xac4c1b51U; x ^= x >> 15; x *= 0x31848babU; x ^= x >> 14;
        return x;
    };
    auto u01 = [&](uint32_t s) -> double { return (hash32(s) & 0xFFFFFFU) / double(0x1000000U); };

    // Pre-compute a conservative body radius for the analytical body-
    // occlusion pre-check below. Use the MEAN face-center radius so that
    // faces genuinely elevated above the mean sphere (mountain tops near
    // the terminator) can still pass to the Embree terrain check.
    double body_radius_m_ = 0.0;
    {
        double sum_r = 0.0;
        for (size_t i = 0; i < n; ++i) {
            sum_r += vectorLength(face_centers[i]);
        }
        body_radius_m_ = sum_r / static_cast<double>(n);
    }

    #pragma omp parallel for schedule(dynamic, 256)
    for (size_t i = 0; i < n; ++i) {
        const auto& fc = face_centers[i];
        const auto& fn = face_normals[i];

        // Sun vector from this face (both in meters)
        std::array<double, 3> sun_rel = {
            sun_pos_m[0] - fc[0],
            sun_pos_m[1] - fc[1],
            sun_pos_m[2] - fc[2]
        };
        double sun_dist_m = vectorLength(sun_rel);
        if (sun_dist_m <= constants::EPSILON) { direct_solar_values[i] = 0.0; continue; }
        std::array<double, 3> sun_dir = { sun_rel[0]/sun_dist_m, sun_rel[1]/sun_dist_m, sun_rel[2]/sun_dist_m };
        // PTM-36: ray parameters t are in MESH-NATIVE units (sun_dir is unit).
        const double sun_dist_n = sun_dist_m * (km_to_native / 1000.0);

        // Incidence cosine
        double cos_inc0 = dotProduct(fn, sun_dir);

        // Solar disk geometry (SUN_RADIUS and sun_dist_km both in km)
        double sun_dist_km = sun_dist_m / 1000.0;
        double s = constants::SUN_RADIUS / sun_dist_km;
        if (s > 1.0) s = 1.0;
        const double disk_ang   = std::asin(s);
        const double disk_scale = std::sin(disk_ang); // equals s

        // Finite-disk horizon gate:
        // Entire disk below horizon if n·s <= -sin(disk_ang)
        if (cos_inc0 <= -std::sin(disk_ang)) {
            direct_solar_values[i] = enable_shadow_flux_ ? calculateShadowFlux(fc, fn, sun_pos_m) : 0.0;
            continue;
        }

        // Analytical body-occlusion pre-check. A face whose center is behind
        // the terminator plane (dot(fc, sun_dir) < 0) AND whose shadow ray
        // passes through the occluding sphere (closest approach < R_occ) cannot
        // see the sun regardless of terrain-slope normal orientation. This
        // catches nightside faces that Embree's mesh trace misses, which
        // otherwise produce spurious illumination up to ~8° past the
        // terminator on topographic meshes.
        //
        // TOPOLOGY GATE (not shape): run this cull ONLY on OPEN patch meshes. On a
        // CLOSED (watertight) mesh the Embree BVH self-occludes, so the pre-check
        // is unnecessary and over-culls sunward-tilted grazing facets on shaped
        // bodies -- disable it and let Embree do all occlusion. On an OPEN micro-
        // patch the BVH has no far-side geometry, so the analytic body-sphere is
        // the SOLE self-occlusion guard.
        //
        // Cull only when the body SPHERE lies BETWEEN the face and the sun: the
        // forward ray (origin=fc, dir=sun) must intersect the body sphere at t>0.
        // A bare |fc x sun| < R_occ line test culls sunward-tilted (mu>0) facets
        // just past the terminator whose ray only grazes the near limb going away
        // from the body -- the forward-intersection condition excludes them. R_occ
        // = body-level radius (min core-face centroid radius), relief-adaptive and
        // never larger than a face's own radius, so no lit face is culled. See
        // mesh_is_open_ / body_radius_occ_ notes and memory
        // ptm-mu-gate-open-mesh-selfocclusion-fail.
        if (mesh_is_open_ && body_radius_occ_native_ > 0.0) {
            const double fc_dot_sun = fc[0]*sun_dir[0] + fc[1]*sun_dir[1] + fc[2]*sun_dir[2];
            if (fc_dot_sun < 0.0) {  // cheap early-out; a forward hit implies this
                const std::array<double,3> body_center = {0.0, 0.0, 0.0};
                double tb1, tb2;
                if (raySphereIntersect(fc, sun_dir, body_center,
                                       body_radius_occ_native_, tb1, tb2)) {
                    if (tb1 > 0.0 || tb2 > 0.0) {
                        direct_solar_values[i] = enable_shadow_flux_
                            ? calculateShadowFlux(fc, fn, sun_pos_m) : 0.0;
                        continue;
                    }
                }
            }
        }

        // PTM-13: local-horizon (skyline) cull of the grazing DIRECT sun (mirror of the
        // single-face path). Gated (default OFF -> byte-identical). CPU batch path.
        if (use_horizon_cull_ && horizon_direct_.built()) {
            const std::array<double,3> p_local = {
                fc[0] - (scene_has_offset_ ? scene_origin_offset_[0] : 0.0),
                fc[1] - (scene_has_offset_ ? scene_origin_offset_[1] : 0.0),
                fc[2] - (scene_has_offset_ ? scene_origin_offset_[2] : 0.0)
            };
            if (horizon_direct_.sunBelowLocalHorizon(p_local, sun_dir)) {
                direct_solar_values[i] = enable_shadow_flux_ ? calculateShadowFlux(fc, fn, sun_pos_m) : 0.0;
                continue;
            }
        }

        // Robust origin. PTM-13 P1/rise-epsilon: with topological self-skip OR rise-epsilon the ray
        // starts AT the face (eps_n=eps_d=0) and self/self-relief hits are rejected by primID / out-of-
        // plane rise, so a real mm-scale local occluder is detected instead of hopped by the 1 cm
        // offset. Default OFF -> unchanged (2x/1x).
        const bool at_face = use_topo_selfskip_ || use_rise_eps_;
        const double eps_base = calculateSafeRayOffset(fc, fn);
        const double eps_n = at_face ? 0.0 : 2.0 * eps_base;
        const double eps_d = at_face ? 0.0 : 1.0 * eps_base;

        std::array<double, 3> ray_o = {
            fc[0] + fn[0] * eps_n + sun_dir[0] * eps_d,
            fc[1] + fn[1] * eps_n + sun_dir[1] * eps_d,
            fc[2] + fn[2] * eps_n + sun_dir[2] * eps_d
        };

        // PTM-38: is the primary anywhere on the solar disk (penumbra OR umbra)?
        // Computed once per face and used twice: to veto the center-ray fast path
        // (which would otherwise report full sun for any face less than ~50 %
        // obscured) and to escalate the disk sampling below. False whenever no
        // eclipse is in progress -> byte-identical outside eclipses.
        const bool eclipse_partial = (!primary_is_sun_) &&
            primaryDiskOverlap(ray_o, sun_dir, primary_pos_n, primary_radius_n,
                               sun_dist_n, disk_ang);

        // Center-ray fast path only if center is above horizon
        if (cos_inc0 > 0.0) {
            // PTM-38: any overlap of the primary with the solar disk sends this
            // face to the finite-disk sampler. The old center-ray-only test is a
            // strict subset of eclipse_partial, so this is never less
            // conservative than before.
            bool blocked = eclipse_partial;

            if (!blocked) {
                std::vector<std::array<double,3>> o(1, ray_o), d(1, sun_dir);
                std::vector<uint8_t> occ;
                if (use_rise_eps_)
                    occ = traceOccludedRiseEps(o, d, std::vector<int>(1, static_cast<int>(i)),
                                               std::vector<std::array<double,3>>(1, fn));
                else if (use_topo_selfskip_)
                    occ = traceOccludedSelfSkip(o, d, std::vector<int>(1, static_cast<int>(i)));
                else
                    occ = traceOccluded(o, d, /*coherent=*/true);
                if (!occ.empty() && occ[0] == 0) {
                    double sun_dist_au = sun_dist_km / constants::KM_PER_AU;
                    double solar_irr = constants::SOLAR_CONSTANT / (sun_dist_au * sun_dist_au);
                    direct_solar_values[i] = solar_irr * cos_inc0;
                    continue;
                }
            }
            // if blocked, fall through to finite-disk sampling
        }
        // If cos_inc0 <= 0 but disk is partially visible, we also fall through to sampling

        // Finite disk sampling (decorrelated), single Embree packet query
        std::array<double, 3> u, v;
        createCoordinateSystem(sun_dir, u, v);

        // Per-face decorrelation to reduce spatial banding
        uint32_t seed = 0x9E3779B1u ^ uint32_t(i);
        double theta = u01(seed) * 2.0 * M_PI;
        std::array<double,3> ur, vr; rotateBasis(u, v, theta, ur, vr);
        double uoff = u01(seed ^ 0x85EBCA6Bu);
        double voff = u01(seed ^ 0xC2B2AE35u);

        // PTM-38: the default 9 samples (constants::N_SOLAR_SAMPLES) quantize the
        // eclipsed fraction at ~11 %, ~21 K near half-obscuration. Escalate to the same heavy count
        // the single-face path uses (illumination_model.cpp heavyN) whenever the
        // primary overlaps the disk. Untouched -> byte-identical -- outside an
        // eclipse.
        const uint32_t baseN = static_cast<uint32_t>(std::max(8, num_samples));
        // PTM-45(ii): ONE heavyN definition, shared with the single-face and
        // GPU batch paths. Identical at the default 9 samples (all 64).
        const uint32_t N = eclipse_partial ? heavySolarSamples(num_samples) : baseN;
        std::vector<std::array<double, 3>> dirs; dirs.reserve(N);
        std::vector<uint8_t> skip; skip.reserve(N);

        // Generate directions on the solar disk and filter only by eclipse (terrain via Embree)
        for (uint32_t k = 0; k < N; ++k) {
            double su, sv; hammersley(k, N, uoff, voff, su, sv);
            double dx, dy; squareToConcentricDisk(su, sv, dx, dy);
            if (dx*dx + dy*dy > 1.0) continue;

            const double ox = disk_scale * dx, oy = disk_scale * dy;
            std::array<double, 3> sd = {
                sun_dir[0] + ox * ur[0] + oy * vr[0],
                sun_dir[1] + ox * ur[1] + oy * vr[1],
                sun_dir[2] + ox * ur[2] + oy * vr[2]
            };
            double l = vectorLength(sd);
            if (l <= constants::EPSILON) continue;
            sd[0]/=l; sd[1]/=l; sd[2]/=l;

            bool sskip = false;

            // Eclipse clip (before occlusion) — mesh-native units
            if (!primary_is_sun_) {
                double t1, t2;
                if (raySphereIntersect(ray_o, sd, primary_pos_n, primary_radius_n, t1, t2)) {
                    if ((t1 > 0.0 && t1 < sun_dist_n) || (t2 > 0.0 && t2 < sun_dist_n)) sskip = true;
                }
            }

            skip.push_back(sskip ? 1 : 0);
            if (!sskip) dirs.push_back(sd);
        }

        // No valid samples (full eclipse)
        if (dirs.empty()) {
            direct_solar_values[i] = enable_shadow_flux_ ? calculateShadowFlux(fc, fn, sun_pos_m) : 0.0;
            continue;
        }

        // Start each sample a bit along its own direction as well (eps_d=0 under self-skip -> at fc)
        std::vector<std::array<double,3>> origins; origins.reserve(dirs.size());
        for (const auto& sd : dirs) {
            origins.push_back({ ray_o[0] + sd[0]*eps_d,
                                ray_o[1] + sd[1]*eps_d,
                                ray_o[2] + sd[2]*eps_d });
        }

        // Terrain occlusion for survivors. PTM-13: rise-epsilon / self-skip re-trace (default OFF -> traceOccluded).
        std::vector<uint8_t> occ;
        if (use_rise_eps_)
            occ = traceOccludedRiseEps(origins, dirs, std::vector<int>(dirs.size(), static_cast<int>(i)),
                                       std::vector<std::array<double,3>>(dirs.size(), fn));
        else if (use_topo_selfskip_)
            occ = traceOccludedSelfSkip(origins, dirs, std::vector<int>(dirs.size(), static_cast<int>(i)));
        else
            occ = traceOccluded(origins, dirs, /*coherent=*/true);

        // Accumulate cosine-weighted contributions for unblocked samples
        double sum_cos = 0.0;
        uint32_t total_out = 0;
        size_t m = 0;
        for (uint32_t k = 0; k < skip.size(); ++k) {
            if (skip[k]) continue;
            const bool blocked = (m < occ.size()) && (occ[m] != 0);
            const auto& sd = dirs[m];
            m++;
            total_out++;
            if (!blocked) {
                double c = dotProduct(fn, sd);
                if (c > 0.0) sum_cos += c;
            }
        }

        if (total_out == 0) {
            direct_solar_values[i] = enable_shadow_flux_ ? calculateShadowFlux(fc, fn, sun_pos_m) : 0.0;
            continue;
        }

        double sun_dist_au = sun_dist_km / constants::KM_PER_AU;
        double solar_irr = constants::SOLAR_CONSTANT / (sun_dist_au * sun_dist_au);
        // PTM-38: normalize by the ATTEMPTED disk samples, not the survivors.
        // total_out counts only the samples that were NOT eclipse-skipped, so a
        // face with 95 % of the solar disk behind the primary renormalized its
        // 5 % of surviving samples back up to FULL SUN -- the eclipse was a step
        // function at totality instead of a ramp through the penumbra.
        // Terrain-blocked samples were always (correctly) kept in the
        // denominator; only the eclipse-skipped ones were deleted. skip.size()
        // == total_out whenever nothing is eclipse-skipped, so this is exactly a
        // no-op outside an eclipse.
        double flux = solar_irr * (sum_cos / double(skip.size()));
        if (flux <= 0.0 && enable_shadow_flux_) {
            flux = calculateShadowFlux(fc, fn, sun_pos_m);
        }
        direct_solar_values[i] = flux;
    }
}


// Calculate safe ray offset based on face center and normal
double IlluminationModel::calculateSafeRayOffset(
    const std::array<double, 3>& face_center, 
    const std::array<double, 3>& /*face_normal*/) const 
{
    // Scale-aware offset in meters: robust across global meshes and DEM roughness.
    // The radius MUST be measured in the same recentered frame the Embree/BVH scene
    // lives in (scene_origin_offset_ is subtracted in initializeScene), exactly as the
    // sibling estimate_tnear lambda in traceOccluded does. Using the raw selenocentric
    // center here inflated the offset ~175x (1.74 m vs ~1 cm) on lunar meshes, lifting
    // grazing-sun shadow rays ~3.5 m and lowering a 200 m-distant rim's apparent horizon
    // by ~1 deg -> spurious illumination of permanently-shadowed crater floors.
    const std::array<double, 3> c_local = {
        face_center[0] - (scene_has_offset_ ? scene_origin_offset_[0] : 0.0),
        face_center[1] - (scene_has_offset_ ? scene_origin_offset_[1] : 0.0),
        face_center[2] - (scene_has_offset_ ? scene_origin_offset_[2] : 0.0)
    };
    const double r = vectorLength(c_local);                         // local radius [m]
    const double ulp = std::numeric_limits<double>::epsilon() * r;  // machine epsilon scaled
    const double scaled = std::max(1e-6 * r, 0.01);                 // 1 ppm of radius, min 1 cm
    return std::max(16.0 * ulp, scaled);
}

std::array<double, 3> IlluminationModel::vectorSubtract(
    const std::array<double, 3>& v1,
    const std::array<double, 3>& v2) {
    return {{
        v1[0] - v2[0],
        v1[1] - v2[1],
        v1[2] - v2[2]
    }};
}

std::array<double, 3> IlluminationModel::crossProduct(
    const std::array<double, 3>& v1,
    const std::array<double, 3>& v2) const {
    return {{
        v1[1] * v2[2] - v1[2] * v2[1],
        v1[2] * v2[0] - v1[0] * v2[2],
        v1[0] * v2[1] - v1[1] * v2[0]
    }};
}

double IlluminationModel::vectorLength(const std::array<double, 3>& v) const {
    return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

bool IlluminationModel::raySphereIntersect(
    const std::array<double, 3>& ray_origin,
    const std::array<double, 3>& ray_direction,
    const std::array<double, 3>& sphere_center,
    double sphere_radius,
    double& t1,
    double& t2) {
    
    // Compute relative position of sphere to ray origin
    std::array<double, 3> rel_pos = {{
        ray_origin[0] - sphere_center[0],
        ray_origin[1] - sphere_center[1],
        ray_origin[2] - sphere_center[2]
    }};
    
    // Ray-sphere intersection coefficients
    double b = 2.0 * (ray_direction[0] * rel_pos[0] + 
                     ray_direction[1] * rel_pos[1] + 
                     ray_direction[2] * rel_pos[2]);
    
    double c = rel_pos[0] * rel_pos[0] +
               rel_pos[1] * rel_pos[1] + 
               rel_pos[2] * rel_pos[2] - 
               sphere_radius * sphere_radius;
    
    double discriminant = b*b - 4.0*c;
    
    if (discriminant > 0.0) {
        t1 = (-b - std::sqrt(discriminant)) / 2.0;
        t2 = (-b + std::sqrt(discriminant)) / 2.0;
        return true;
    }
    
    t1 = t2 = 0.0;
    return false;
}

void IlluminationModel::createCoordinateSystem(
    const std::array<double, 3>& direction,
    std::array<double, 3>& u,
    std::array<double, 3>& v) {
    
    // Choose a suitable up vector for creating the basis
    std::array<double, 3> up = {0.0, 0.0, 1.0};
    
    // If direction is nearly parallel to up, use a different up vector
    if (std::abs(direction[2]) > 0.9) {
        up = {0.0, 1.0, 0.0};
    }
    
    // Implement cross product directly instead of calling member function
    u[0] = up[1] * direction[2] - up[2] * direction[1];
    u[1] = up[2] * direction[0] - up[0] * direction[2];
    u[2] = up[0] * direction[1] - up[1] * direction[0];
    
    // Normalize directly
    double len = std::sqrt(u[0]*u[0] + u[1]*u[1] + u[2]*u[2]);
    if (len > constants::EPSILON) {
        u[0] /= len;
        u[1] /= len;
        u[2] /= len;
    }
    
    // Second cross product directly
    v[0] = direction[1] * u[2] - direction[2] * u[1];
    v[1] = direction[2] * u[0] - direction[0] * u[2];
    v[2] = direction[0] * u[1] - direction[1] * u[0];
}

// ===========================================================================
// PTM-44: fractional visibility of the PRIMARY's finite disk.
//
// Both primary-flux paths used to decide visibility with ONE ray to the disk
// centre and to zero the flux the instant cos(theta) <= 0. The primary is not a
// point -- Jupiter's angular RADIUS from Europa is 5.98 deg, Earth's from the
// Moon 0.95 deg -- so a crater rim sweeping across it switched a face between
// 100 % and 0 % with a hard step, and a face whose normal fell 0.1 deg past the
// primary's horizon lost all of it. The finite SOLAR disk already gets this
// treatment (:3727 uses the identical `cos_inc0 <= -sin(disk_ang)` criterion);
// this brings the primary paths level.
//
// The result is <cos theta> over the disk, eq. (8) of
// docs/primary-reflected-flux-derivation.md, factored as
//
//     <cos theta> = A(h, alpha) * V_terrain
//
//   A         analytic horizon clip, ptm::disk::diskMeanCosAboveHorizon.
//             EXACTLY cos(theta_0) when the whole disk is above the horizon, so
//             the arithmetic for those faces is bit-identical to the old code.
//   V_terrain cosine-weighted unoccluded fraction of the disk, from rays.
//
// Cost control (fast-path criterion, documented in the derivation note 7.3):
//   * whole disk below the horizon        -> 0, NO rays;
//   * 5-ray probe (centre + 4 rim points, above-horizon directions only)
//     unanimous                           -> V_terrain = 1 or 0, no more rays;
//   * probe split                         -> PRIMARY_DISK_SAMPLES rays; see
//                                            primary_disk.h for the MEASURED
//                                            quantisation that fixes the count.
// On a convex body (every icosphere) an above-horizon ray from an offset origin
// always escapes, so global meshes pay 5 rays per face instead of 1 and get the
// analytic horizon ramp for free. PTM_PRIMARY_DISK_ALWAYS_SAMPLE=1 disables the
// probe short-circuit for validation.
//
// Results land in the memo members and are consumed by BOTH batch functions:
// visibility depends only on (mesh, primary position) -- not on the Sun -- and
// the two batches are always called back to back with the same arguments
// (:2223/:2251 and :3530/:3539), so one evaluation serves both. Both call sites
// are outside every OpenMP region, so the memo needs no lock.
// ===========================================================================
void IlluminationModel::updatePrimaryDiskVisibility(
    const std::vector<std::array<double, 3>>& face_centers,
    const std::vector<std::array<double, 3>>& face_normals,
    const std::array<double, 3>& primary_position) {

    const size_t num_faces = face_centers.size();

    if (primary_vis_key_n_ == num_faces &&
        primary_vis_key_centers_ == static_cast<const void*>(face_centers.data()) &&
        primary_vis_key_normals_ == static_cast<const void*>(face_normals.data()) &&
        primary_vis_key_pos_[0] == primary_position[0] &&
        primary_vis_key_pos_[1] == primary_position[1] &&
        primary_vis_key_pos_[2] == primary_position[2] &&
        primary_vis_mean_cos_.size() == num_faces) {
        return;                                   // memo hit
    }

    // The memo key is published only AFTER the loop completes: if the loop
    // throws, the next call must recompute rather than serve a half-filled
    // array (the two batch callers catch and zero-fill, which would otherwise
    // hide it).
    primary_vis_key_centers_ = nullptr;
    primary_vis_key_normals_ = nullptr;
    primary_vis_key_n_       = static_cast<size_t>(-1);
    primary_vis_mean_cos_.assign(num_faces, 0.0);
    primary_vis_dist_km_.assign(num_faces, 0.0);
    if (num_faces == 0) return;

    // Same |face_center| > 10000 unit discriminator the rest of this file uses
    // (PTM-37 makes it visible and refuses the ambiguous cases at startup).
    const bool centers_in_meters = (vectorLength(face_centers[0]) > 10000.0);
    const double km_to_native = centers_in_meters ? 1000.0 : 1.0;

    static const bool always_sample = [](){
        const char* e = std::getenv("PTM_PRIMARY_DISK_ALWAYS_SAMPLE");
        return e && std::atoi(e) != 0;
    }();
    // Opt-in branch census. Without it there is no way to tell from a run's
    // outputs whether the escalation path ever fired, and "it must have" is not
    // evidence.
    static const bool report_stats = [](){
        const char* e = std::getenv("PTM_PRIMARY_DISK_STATS");
        return e && std::atoi(e) != 0;
    }();
    size_t n_below = 0, n_full = 0, n_dark = 0, n_sampled = 0;
    // Fermi F1 accounting: faces whose analytic clip A is POSITIVE but which end
    // at zero anyway, and why. On a closed convex mesh the limb legitimately
    // blocks grazing rays, so this is not expected to be zero -- but it must be
    // attributable to rays, never to a probe that simply missed the crescent.
    size_t n_zero_by_rays = 0, n_probe_underflow = 0;

    const uint32_t N_SAMP = ptm::disk::PRIMARY_DISK_SAMPLES;

    #pragma omp parallel for schedule(dynamic) \
        reduction(+:n_below, n_full, n_dark, n_sampled, n_zero_by_rays, \
                    n_probe_underflow)
    for (size_t i = 0; i < num_faces; ++i) {
        const auto& fn = face_normals[i];

        std::array<double, 3> fc_km = centers_in_meters
            ? metersToKm(face_centers[i]) : face_centers[i];

        const std::array<double, 3> rel = {
            primary_position[0] - fc_km[0],
            primary_position[1] - fc_km[1],
            primary_position[2] - fc_km[2]
        };
        const double dist = vectorLength(rel);
        if (dist <= constants::EPSILON) continue;
        primary_vis_dist_km_[i] = dist;

        const std::array<double, 3> dir = { rel[0]/dist, rel[1]/dist, rel[2]/dist };
        const double c0 = dotProduct(fn, dir);

        double sin_alpha = primary_radius_ / dist;
        if (sin_alpha > 1.0) sin_alpha = 1.0;          // degenerate: inside the primary
        const double alpha = std::asin(sin_alpha);

        // Whole disk below the face's own horizon plane -> no flux, no rays.
        if (c0 <= -sin_alpha) { ++n_below; continue; }

        const double A = ptm::disk::diskMeanCosAboveHorizon(c0, alpha);
        if (A <= 0.0) { ++n_below; continue; }

        if (!embree_scene_) {                          // no BVH -> nothing occludes
            primary_vis_mean_cos_[i] = A;
            ++n_full;
            continue;
        }

        const double safe_offset = calculateSafeRayOffset(face_centers[i], fn);
        const std::array<double, 3> ray_origin_local = {
            face_centers[i][0] + fn[0]*safe_offset - (scene_has_offset_ ? scene_origin_offset_[0] : 0.0),
            face_centers[i][1] + fn[1]*safe_offset - (scene_has_offset_ ? scene_origin_offset_[1] : 0.0),
            face_centers[i][2] + fn[2]*safe_offset - (scene_has_offset_ ? scene_origin_offset_[2] : 0.0)
        };
        // Stop the ray at the primary's near surface, exactly as the single
        // centre ray did. Mesh-native units.
        const float tfar_dist = static_cast<float>((dist - primary_radius_) * km_to_native);

        // Fermi F1: the probe basis is oriented to the HORIZON, not arbitrarily.
        // u points along the horizon gradient, so probe (+1, 0) is the disk's
        // HIGHEST point. With an arbitrary basis (createCoordinateSystem) all
        // five probes can miss a thin above-horizon crescent, probes_up comes
        // back 0, and the face is zeroed even though the analytic clip A says
        // it is lit -- measured at 10851 of 1082503 A>0 face-steps on the
        // Europa arm before this fix.
        std::array<double, 3> u, v;
        {
            std::array<double, 3> up = { fn[0] - c0*dir[0],
                                         fn[1] - c0*dir[1],
                                         fn[2] - c0*dir[2] };
            const double ul = vectorLength(up);
            if (ul > 1.0e-12) {
                u = { up[0]/ul, up[1]/ul, up[2]/ul };
                v = crossProduct(dir, u);          // orthonormal: |dir x u| = 1
            } else {
                // fn parallel to dir: no horizon gradient exists, and the disk
                // is entirely above the horizon anyway (|c0| = 1), so any basis
                // will do.
                createCoordinateSystem(dir, u, v);
            }
        }

        // Direction toward the disk point offset by (dx, dy) in units of the
        // disk's angular radius. Same construction (and same sin(alpha) scale)
        // as the solar-disk sampler at :3868.
        auto disk_dir = [&](double dx, double dy) {
            std::array<double, 3> sd = {
                dir[0] + sin_alpha*(dx*u[0] + dy*v[0]),
                dir[1] + sin_alpha*(dx*u[1] + dy*v[1]),
                dir[2] + sin_alpha*(dx*u[2] + dy*v[2])
            };
            const double l = vectorLength(sd);
            if (l > constants::EPSILON) { sd[0]/=l; sd[1]/=l; sd[2]/=l; }
            return sd;
        };

        auto occluded = [&](const std::array<double, 3>& sd) {
            RTCRayHit rayhit;
            rayhit.ray.org_x = ray_origin_local[0];
            rayhit.ray.org_y = ray_origin_local[1];
            rayhit.ray.org_z = ray_origin_local[2];
            rayhit.ray.dir_x = sd[0];
            rayhit.ray.dir_y = sd[1];
            rayhit.ray.dir_z = sd[2];
            rayhit.ray.tnear = 1.0e-4f;
            rayhit.ray.tfar  = tfar_dist;
            rayhit.ray.mask  = 0xFFFFFFFF;
            rayhit.ray.flags = 0;
            rayhit.ray.time  = 0.0f;
            rayhit.hit.geomID    = RTC_INVALID_GEOMETRY_ID;
            rayhit.hit.primID    = RTC_INVALID_GEOMETRY_ID;
            rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
            RTCIntersectArguments args;
            rtcInitIntersectArguments(&args);
            rtcIntersect1(embree_scene_, &rayhit, &args);
            return rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID;
        };

        // ---- cheap probe: centre + 4 rim points, above-horizon only ---------
        uint32_t probes_up = 0, probes_clear = 0;
        for (uint32_t k = 0; k < ptm::disk::PRIMARY_DISK_PROBES; ++k) {
            double dx, dy;
            ptm::disk::probeOffset(k, dx, dy);
            const std::array<double, 3> sd = disk_dir(dx, dy);
            if (dotProduct(fn, sd) <= 0.0) continue;   // carries no flux anyway
            ++probes_up;
            if (!occluded(sd)) ++probes_clear;
        }

        // Everything below is a SHORT-CIRCUIT of the sampler, so all of it is
        // gated on !always_sample: PTM_PRIMARY_DISK_ALWAYS_SAMPLE has to mean
        // "always sample", or the ALWAYS_SAMPLE arm is not an upper bound on
        // the fast path's error (Fermi F1).
        if (!always_sample && probes_up == 0) {
            // With the horizon-oriented basis this is confined to a sliver of
            // width ~sin^3(alpha)/2 in cos(theta_0) at the very bottom of the
            // band (5.6e-4 of 0.208 at Jupiter/Europa), where the probe rim
            // sits at atan(sin alpha) rather than alpha and so falls just
            // short of the true rim. A is essentially zero there. Keep the
            // analytic value rather than discarding it -- "no probe above the
            // horizon" is absence of terrain evidence, not evidence of
            // blockage -- and say so once. It is NOT counted as n_below, so
            // the census stays comparable with the geometric count.
            warnPrimaryProbeUnderflowOnce();
            primary_vis_mean_cos_[i] = A;
            ++n_full;
            ++n_probe_underflow;
            continue;
        }
        if (!always_sample) {
            if (probes_up > 0 && probes_clear == probes_up) {
                primary_vis_mean_cos_[i] = A;          // fully visible
                ++n_full;
                continue;
            }
            if (probes_clear == 0) {
                ++n_dark;
                ++n_zero_by_rays;
                continue;                              // fully occluded -> 0
            }
        }
        ++n_sampled;

        // ---- escalate: cosine-weighted unoccluded fraction of the disk ------
        // V is a RATIO over the same sample set, so the O(alpha^2) bias of
        // sampling a projected disk cancels and V == 1 reproduces A exactly.
        double w_total = 0.0, w_clear = 0.0;
        for (uint32_t k = 0; k < N_SAMP; ++k) {
            double su, sv;
            ptm::disk::hammersley(k, N_SAMP, 0.0, 0.0, su, sv);
            double dx, dy;
            ptm::disk::squareToConcentricDisk(su, sv, dx, dy);
            const std::array<double, 3> sd = disk_dir(dx, dy);
            const double w = dotProduct(fn, sd);
            if (w <= 0.0) continue;                    // below the horizon: no flux
            w_total += w;
            if (!occluded(sd)) w_clear += w;
        }
        if (w_total <= 0.0) { ++n_zero_by_rays; continue; }
        if (w_clear <= 0.0) ++n_zero_by_rays;
        primary_vis_mean_cos_[i] = A * (w_clear / w_total);
    }

    primary_vis_key_centers_ = static_cast<const void*>(face_centers.data());
    primary_vis_key_normals_ = static_cast<const void*>(face_normals.data());
    primary_vis_key_pos_     = primary_position;
    primary_vis_key_n_       = num_faces;

    if (report_stats) {
        std::cerr << "[illumination] primary disk: " << num_faces << " faces -- "
                  << n_below << " below horizon, " << n_full << " fully visible, "
                  << n_dark << " fully occluded, " << n_sampled << " sampled ("
                  << N_SAMP << " rays each); A>0 zeroed BY RAYS " << n_zero_by_rays
                  << ", probe-underflow fallbacks " << n_probe_underflow
                  << std::endl;
    }
}

void IlluminationModel::calculatePrimaryIRBatch(
    const std::vector<std::array<double, 3>>& face_centers,
    const std::vector<std::array<double, 3>>& face_normals,
    const std::array<double, 3>& primary_position,
    std::vector<double>& ir_values) {
    
    size_t num_faces = face_centers.size();
    ir_values.assign(num_faces, 0.0);

    // Skip if primary is the sun
    if (primary_is_sun_) {
        return;
    }

    // Get primary IR flux based on body
    double primary_ir_flux = 0.0;
    if (primary_body_name_ == "EARTH") {
        primary_ir_flux = constants::EARTH_IR_FLUX;
    } else if (primary_body_name_ == "JUPITER") {
        primary_ir_flux = constants::JUPITER_IR_FLUX;
    } else if (primary_body_name_ == "SATURN") {
        primary_ir_flux = constants::SATURN_IR_FLUX;
    } else if (primary_body_name_ == "MARS") {
        primary_ir_flux = constants::MARS_IR_FLUX;
    } else {
        // PTM-45(i): unknown primary body -> no IR flux. Behavior unchanged;
        // the single-face path now agrees, and both say so.
        warnUnknownPrimaryOnce(primary_body_name_, "primary IR heating (batch)",
                               "contributing 0 W/m^2 for every face");
        return;
    }
    
    // PTM-44: one evaluation of the finite disk's visibility, shared with
    // calculatePrimaryReflectedBatch. Replaces the per-face centre ray AND the
    // hard cos_angle <= 0 cut that used to live in this loop.
    updatePrimaryDiskVisibility(face_centers, face_normals, primary_position);

    // Process each face in parallel
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < num_faces; ++i) {
        const double mean_cos = primary_vis_mean_cos_[i];
        if (mean_cos <= 0.0) {
            continue;    // below the horizon, fully occluded, or degenerate
        }
        ir_values[i] = calculatePrimaryHeating(mean_cos, primary_vis_dist_km_[i]);
    }
}

// Complete implementation of calculatePrimaryReflectedBatch with unit conversion fix
void IlluminationModel::calculatePrimaryReflectedBatch(
    const std::vector<std::array<double, 3>>& face_centers,
    const std::vector<std::array<double, 3>>& face_normals,
    const std::array<double, 3>& sun_position,
    const std::array<double, 3>& primary_position,
    std::vector<double>& reflected_values) {
    
    size_t num_faces = face_centers.size();
    reflected_values.assign(num_faces, 0.0);
    
    // Skip if primary is the sun
    if (primary_is_sun_) {
        return;
    }
    
    // Get primary bond albedo based on body
    double primary_albedo = 0.3;  // Default
    if (primary_body_name_ == "EARTH") {
        primary_albedo = constants::EARTH_BOND_ALBEDO;
    } else if (primary_body_name_ == "JUPITER") {
        primary_albedo = constants::JUPITER_BOND_ALBEDO;
    } else if (primary_body_name_ == "SATURN") {
        primary_albedo = constants::SATURN_BOND_ALBEDO;
    } else if (primary_body_name_ == "MARS") {
        primary_albedo = constants::MARS_BOND_ALBEDO;
    } else {
        // PTM-45(i), same class: a substituted Bond albedo. Value UNCHANGED
        // (both reflected paths already agree on 0.3); it just no longer
        // happens silently.
        warnUnknownPrimaryOnce(primary_body_name_, "primary reflected light (batch)",
                               "substituting a Bond albedo of 0.3");
    }

    // PTM-44: shared finite-disk visibility (memoized -- the IR batch that runs
    // beside this one gets the same evaluation, not a second set of rays).
    updatePrimaryDiskVisibility(face_centers, face_normals, primary_position);

    // Detect if face centers are in meters or kilometers (needed only to put
    // the face centre in km for the phase-angle geometry).
    const bool centers_in_meters =
        (!face_centers.empty() && vectorLength(face_centers[0]) > 10000.0);

    // Process each face in parallel
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < num_faces; ++i) {
        const double mean_cos = primary_vis_mean_cos_[i];
        if (mean_cos <= 0.0) {
            continue;    // below the horizon, fully occluded, or degenerate
        }
        const double primary_dist = primary_vis_dist_km_[i];
        if (primary_dist <= constants::EPSILON) {
            continue;
        }

        const std::array<double, 3> face_center_km = centers_in_meters
            ? metersToKm(face_centers[i]) : face_centers[i];
        const std::array<double, 3> primary_rel = {
            primary_position[0] - face_center_km[0],
            primary_position[1] - face_center_km[1],
            primary_position[2] - face_center_km[2]
        };
        const std::array<double, 3> primary_dir = {
            primary_rel[0] / primary_dist,
            primary_rel[1] / primary_dist,
            primary_rel[2] / primary_dist
        };

        reflected_values[i] = calculatePrimaryReflectedLight(
            mean_cos, primary_dir, sun_position, primary_position, primary_dist);
    }
}

bool IlluminationModel::traceRaySingle(const Ray& ray) {
    // Create vectors with a single ray
    std::vector<std::array<double, 3>> origins = {ray.origin};
    std::vector<std::array<double, 3>> directions = {ray.direction};
    
    // Use existing ray tracing infrastructure to perform the intersection test
    std::vector<int> result = traceRays(origins, directions);
    
    // If the result indicates a hit (not -1), there was an intersection
    return !result.empty() && result[0] >= 0;
}

// `mean_cos` is <cos theta> over the primary's disk (PTM-44): the horizon clip
// and the terrain-visible fraction are already folded in by
// updatePrimaryDiskVisibility, and it equals the old dot(normal, primary_dir)
// exactly whenever the whole disk is above the horizon and unoccluded.
double IlluminationModel::calculatePrimaryHeating(
    double mean_cos,
    double primary_distance) {

    // If Primary is below the horizon or fully hidden, no heating
    if (mean_cos <= 0.0) {
        return 0.0;
    }

    // Calculate the angular size of Primary as seen from Body
    // Use the class member variable for primary radius instead of constant
    double primary_angular_radius = std::asin(primary_radius_ / primary_distance);
    double solid_angle = 2 * M_PI * (1.0 - std::cos(primary_angular_radius));

    // PTM-40: view factor from a differential area to a sphere.
    //
    // The correct factor is F_dA->sphere = sin^2(alpha) * cos(theta), which for
    // a sphere of uniform radiosity J gives E = J * F (by reciprocity,
    // A_dA * F_dA->S = A_S * F_S->dA). Since Omega = 2*pi*(1 - cos alpha), that
    // is Omega/pi to better than 0.3 % for alpha <= 6 deg (Jupiter from Europa,
    // the largest primary disk PTM3D models).
    //
    // The old divisor was 4*pi, i.e. Omega/(4*pi) = the fraction of the FULL
    // SPHERE of directions the primary covers. That is not an irradiance view
    // factor, and it made every primary-IR flux 4x too low: Jupiter at the
    // sub-Jovian Europa face read 0.037 W/m^2 instead of 0.150, and Earth at the
    // sub-Earth lunar face 0.017 instead of 0.066.
    //
    // The IR flux constants this multiplies are EMITTER-SURFACE fluxes, which is
    // what J has to be: JUPITER_IR_FLUX 13.8 = sigma*(124.5 K)^4 and
    // EARTH_IR_FLUX 240 = sigma*(255 K)^4, both matching the body's effective
    // temperature, not any at-target irradiance. (SATURN_IR_FLUX and
    // MARS_IR_FLUX do NOT match their own stated temperatures -- see BUGS PTM-43;
    // Titan/Enceladus/Phobos-class runs are affected by that, not by this fix.)
    double view_factor = solid_angle / M_PI * mean_cos;
    
    // Use appropriate IR flux based on the primary body
    double primary_ir_flux = 0.0;
    
    if (primary_is_sun_) {
        // For Sun as primary, use solar constant scaled by distance
        double dist_au = primary_distance / constants::KM_PER_AU;
        primary_ir_flux = constants::SOLAR_CONSTANT / (dist_au * dist_au);
    } 
    else if (primary_body_name_ == "JUPITER") {
        primary_ir_flux = constants::JUPITER_IR_FLUX;
    }
    else if (primary_body_name_ == "EARTH") {
        primary_ir_flux = constants::EARTH_IR_FLUX;
    }
    else if (primary_body_name_ == "SATURN") {
        primary_ir_flux = constants::SATURN_IR_FLUX;
    }
    else if (primary_body_name_ == "MARS") {
        primary_ir_flux = constants::MARS_IR_FLUX;
    }
    else {
        // PTM-45(i): unknown primary body.
        //
        // This used to fall through to primary_temp = 250 K and emit
        // sigma*250^4 = 221 W/m^2 with no warning, while the batch path
        // (calculatePrimaryIRBatch) returned 0.0 for the SAME body. Two paths,
        // two answers, both silent. The four per-body branches that preceded
        // the 250 K line -- JUPITER / EARTH / SATURN / MARS -- were dead: the
        // if-chain above already claims all four, so nothing could reach them.
        // That dead code is the only reason the 250 K default never produced a
        // wrong number.
        //
        // Unified on the batch path's behavior (0.0), and now it warns.
        warnUnknownPrimaryOnce(primary_body_name_, "primary IR heating",
                               "contributing 0 W/m^2, matching the batch path "
                               "(it used to silently assume 250 K here)");
        return 0.0;
    }
    
    // Apply the appropriate IR flux, scaled by view factor
    return primary_ir_flux * view_factor;
}

// PTM-41: disk-integrated reflected irradiance from a LAMBERT SPHERE primary.
//
//     E = p * F * sin^2(alpha) * Phi_L(a) * <cos theta>
//     p       = (2/3) * A_bond          (Lambert sphere: phase integral q = 3/2)
//     Phi_L(a)= [sin a + (pi - a) cos a] / pi
//
// Full derivation, limits and the numerical check:
// docs/primary-reflected-flux-derivation.md.
//
// What this replaces, and why the previous "the two errors cancel" note was
// only half right. The old expression was
//
//     E_old = F * A_bond * [(1+cos a)/2] * (Omega/(4 pi)) * cos theta
//           = F * (A_bond/4) * sin^2(alpha) * Phi_old(a) * cos theta
//
// The 1/4 in Omega/(4 pi) does belong to an ISOTROPIC re-radiator -- an
// isotropic sphere has Phi == 1 and phase integral q = 4, so p_iso = A_bond/4.
// But the code then multiplied that isotropic normalisation by a NON-isotropic
// phase function whose own phase integral is
//
//     q_old = 2 * Int_0^pi [(1+cos a)/2] sin a da = 2
//
// so the implied total reflected power was F*pi*R^2 * A_bond * (2/4), i.e.
// EXACTLY HALF the sunlight the primary reflects. The pre-fix term was not just
// mis-shaped in phase, it broke energy conservation by a factor 2, and at full
// phase it was 8/3 = 2.667x low (ratio truth/old = (8/3) Phi_L/Phi_old, which
// falls to 1.70 at quadrature and crosses 1 near a = 125 deg -- near NEW phase
// the old code over-estimated). The declined blanket x4 of PTM-40 would have
// given 2x the correct total power; the radiosity and the phase function had to
// be repaired together, which is what this does. With p = (2/3) A_bond and
// Phi_L, q = 3/2 and the energy budget closes exactly.
//
// Model choice: a Lambert sphere, energy-consistent with the *_BOND_ALBEDO
// constants. Jupiter's measured V-band geometric albedo (~0.52) would be 2.3x
// brighter at full phase, but pairing it with A_bond = 0.343 implies q = 0.66,
// which is not Jupiter's phase integral, and it is a V-band number applied to a
// bolometric flux. A body-specific measured (p, Phi) pair is a separate ticket.
double IlluminationModel::calculatePrimaryReflectedLight(
    double mean_cos,
    const std::array<double, 3>& primary_direction,
    const std::array<double, 3>& sun_position,
    const std::array<double, 3>& primary_position,
    double primary_distance) {

    // If Primary is below the horizon or fully hidden, no reflection.
    // mean_cos is <cos theta> over the disk (PTM-44); it equals the old
    // dot(normal, primary_dir) for a fully visible disk.
    if (mean_cos <= 0.0) {
        return 0.0;
    }

    // Get vector from Primary to Sun
    std::array<double, 3> primary_to_sun = {
        sun_position[0] - primary_position[0],
        sun_position[1] - primary_position[1],
        sun_position[2] - primary_position[2]
    };
    double sun_dist = vectorLength(primary_to_sun);
    if (sun_dist <= constants::EPSILON || primary_distance <= constants::EPSILON) {
        return 0.0;
    }

    // Normalize
    std::array<double, 3> sun_direction = {
        primary_to_sun[0] / sun_dist,
        primary_to_sun[1] / sun_dist,
        primary_to_sun[2] / sun_dist
    };

    // Phase angle: Sun-primary-face angle. primary_direction points *toward*
    // the primary, so -primary_direction is the primary->face direction.
    // a = 0 is "full" primary (and, on the sub-primary hemisphere, local
    // midnight); a = pi is "new".
    double cos_phase = -dotProduct(sun_direction, primary_direction);
    cos_phase = std::max(-1.0, std::min(1.0, cos_phase));
    const double phase_angle = std::acos(cos_phase);

    // Get Primary's albedo based on the current primary body
    double primary_albedo = 0.0;

    if (primary_body_name_ == "JUPITER") {
        primary_albedo = constants::JUPITER_BOND_ALBEDO;
    }
    else if (primary_body_name_ == "EARTH") {
        primary_albedo = constants::EARTH_BOND_ALBEDO;
    }
    else if (primary_body_name_ == "SATURN") {
        primary_albedo = constants::SATURN_BOND_ALBEDO;
    }
    else if (primary_body_name_ == "MARS") {
        primary_albedo = constants::MARS_BOND_ALBEDO;
    }
    else {
        // PTM-45(i), same class as the batch path above: substituted value,
        // UNCHANGED at 0.3, now announced.
        warnUnknownPrimaryOnce(primary_body_name_, "primary reflected light",
                               "substituting a Bond albedo of 0.3");
        primary_albedo = 0.3; // Default albedo for other bodies
    }

    // Angular radius of the primary seen from THIS FACE. The pre-fix code used
    // vectorLength(primary_position), i.e. the BODY-CENTRE distance, while the
    // IR path next door used the face-relative one; that inconsistency was
    // worth 0.5 % at Europa and 0.9 % on a lunar DEM patch.
    double sin_angular_radius = primary_radius_ / primary_distance;
    if (sin_angular_radius > 1.0) sin_angular_radius = 1.0;

    // Solar flux at the primary
    double sun_dist_au = sun_dist / constants::KM_PER_AU;
    double solar_flux_at_primary = constants::SOLAR_CONSTANT / (sun_dist_au * sun_dist_au);

    return ptm::disk::lambertSphereIrradiance(solar_flux_at_primary,
                                              primary_albedo,
                                              sin_angular_radius,
                                              phase_angle,
                                              mean_cos);
}

// Helper function to map body names to SPICE IDs
std::string IlluminationModel::getSpiceIdForBody(const std::string& body_name) {
    // Common SPICE IDs for solar system bodies
    static const std::map<std::string, std::string> spice_ids = {
        {"SUN", "10"},
        {"MERCURY", "199"},
        {"VENUS", "299"},
        {"EARTH", "399"},
        {"MOON", "301"},
        {"MARS", "499"},
        {"JUPITER", "599"},
        {"EUROPA", "502"},
        {"GANYMEDE", "503"},
        {"CALLISTO", "504"},
        {"IO", "501"},
        {"SATURN", "699"},
        {"TITAN", "606"},
        {"ENCELADUS", "602"},
        {"URANUS", "799"},
        {"NEPTUNE", "899"},
        {"PLUTO", "999"}
    };
    
    auto it = spice_ids.find(body_name);
    if (it != spice_ids.end()) {
        return it->second;
    }
    
    // Default to Europa if not found
    // Replace debugPrint with std::cout since this is a static method
    std::cout << "Warning: No SPICE ID found for body: " << body_name 
              << ". Using Europa's ID as default." << std::endl;
    return "502";
}

// Set planetary body from enum
void IlluminationModel::setPlanetaryBody(thermal::PlanetaryBody body) {
    // Get properties for this body to access meta-kernel info
    thermal::ThermophysicalProperties props = thermal::PlanetaryData::getProperties(body);
    
    // Set meta-kernel from properties
    meta_kernel_filename_ = props.spice_metakernel;

    // Set albedo from planetary properties for consistency
    setAlbedo(props.albedo);
    
    use_keplerian_ = false;  // Built-in bodies use SPICE

    switch (body) {
        case thermal::PlanetaryBody::EUROPA:
            setTargetBody("EUROPA");
            setPrimaryBody("JUPITER");
            primary_radius_ = constants::JUPITER_RADIUS;
            primary_is_sun_ = false;
            has_primary_body_ = true;
            break;
        case thermal::PlanetaryBody::MOON:
            setTargetBody("MOON");
            setPrimaryBody("EARTH");
            primary_radius_ = constants::EARTH_RADIUS;
            primary_is_sun_ = false;
            has_primary_body_ = true;
            break;
        case thermal::PlanetaryBody::MERCURY:
            setTargetBody("MERCURY");
            setPrimaryBody("SUN");
            primary_is_sun_ = true;
            has_primary_body_ = false;
            break;
        case thermal::PlanetaryBody::ENCELADUS:
            setTargetBody("ENCELADUS");
            setPrimaryBody("SATURN");
            primary_radius_ = constants::SATURN_RADIUS;
            primary_is_sun_ = false;
            has_primary_body_ = true;
            break;
        case thermal::PlanetaryBody::TITAN:
            setTargetBody("TITAN");
            setPrimaryBody("SATURN");
            primary_radius_ = constants::SATURN_RADIUS;
            primary_is_sun_ = false;
            has_primary_body_ = true;
            break;
        case thermal::PlanetaryBody::MARS:
            setTargetBody("MARS");
            setPrimaryBody("SUN");
            primary_is_sun_ = true;
            has_primary_body_ = false;
            break;
        case thermal::PlanetaryBody::CERES:
            setTargetBody("CERES");
            setPrimaryBody("SUN");
            primary_is_sun_ = true;
            has_primary_body_ = false;
            break;
        default:
            setTargetBody("EUROPA"); // Default
            setPrimaryBody("JUPITER");
            has_primary_body_ = true;
            break;
    }
    
    // Use standard cout since debugPrint is non-static
    // std::cout << "Set planetary body to " << target_body_name_ 
    //           << " with primary " << primary_body_name_
    //           << " using meta-kernel " << meta_kernel_filename_ << std::endl;
}

void IlluminationModel::setGenericBody(const orbital::GenericBody& body) {
    target_body_name_ = body.name;
    has_primary_body_ = false;  // Heliocentric bodies have no primary
    primary_is_sun_ = true;

    // Set albedo from body properties
    setAlbedo(body.thermo.albedo);

    // Always create BodyFixedFrame from spin state for generic bodies.
    // HORIZONS SPK files don't include body-fixed frame data, so we need
    // the spin state to rotate SPICE J2000 positions to body-fixed.
    body_fixed_frame_ = std::make_unique<orbital::BodyFixedFrame>(body.spin);

    if (body.ephemeris_source == orbital::GenericBody::KEPLERIAN) {
        // Keplerian mode
        use_keplerian_ = true;
        keplerian_propagator_ = std::make_unique<orbital::KeplerianPropagator>(body.orbit);

        // No SPICE frame or ID needed
        target_body_frame_ = "J2000";  // Placeholder (not used in Keplerian mode)
        target_body_id_ = "0";

        std::cout << "Configured generic body '" << body.name
                  << "' with Keplerian propagation (a=" << body.orbit.semi_major_axis_au
                  << " AU, e=" << body.orbit.eccentricity
                  << ", P_rot=" << body.spin.spin_period_s / 3600.0 << " h)" << std::endl;
    } else {
        // SPICE mode with user-provided metakernel
        use_keplerian_ = false;
        meta_kernel_filename_ = body.spice_metakernel;

        // Also create Keplerian propagator as fallback for times outside SPK coverage.
        // Equilibration orbits can extend before the SPK start time.
        keplerian_propagator_ = std::make_unique<orbital::KeplerianPropagator>(body.orbit);

        // Use the NAIF body ID for SPICE queries (e.g. "2000269" for asteroid 269).
        // SPICE doesn't recognize human-readable names for small bodies from SPK files.
        if (!body.spice_body_id.empty()) {
            target_body_name_ = body.spice_body_id;
        }

        // Use ecliptic J2000 frame for SPICE position queries.
        // We then rotate to body-fixed using BodyFixedFrame (from spin state).
        target_body_frame_ = "ECLIPJ2000";

        std::cout << "Configured generic body '" << body.name
                  << "' with SPICE ephemeris (kernel: " << body.spice_metakernel
                  << ", NAIF ID: " << target_body_name_
                  << ", Keplerian fallback: enabled)" << std::endl;
    }
}

void IlluminationModel::initializeChunkedStorage(size_t total_faces, int time_steps) {
    using_chunked_storage_ = true;
    
    // Calculate number of chunks needed
    size_t num_chunks = (total_faces + chunk_size_ - 1) / chunk_size_;
    
    // Initialize chunk vectors
    illumination_chunks_.resize(num_chunks);
    direct_solar_chunks_.resize(num_chunks);
    primary_reflected_chunks_.resize(num_chunks);
    primary_ir_chunks_.resize(num_chunks);
    
    // Initialize each chunk
    for (size_t i = 0; i < num_chunks; i++) {
        // Calculate size of this chunk
        size_t this_chunk_size = (i == num_chunks - 1) ? 
            (total_faces - i * chunk_size_) * time_steps : 
            chunk_size_ * time_steps;
            
        illumination_chunks_[i].resize(this_chunk_size, 0.0);
        direct_solar_chunks_[i].resize(this_chunk_size, 0.0);
        primary_reflected_chunks_[i].resize(this_chunk_size, 0.0);
        primary_ir_chunks_[i].resize(this_chunk_size, 0.0);
    }
    
    std::cout << "Initialized chunked storage with " << num_chunks 
              << " chunks of up to " << chunk_size_ << " faces each" << std::endl;
}

void IlluminationModel::storeChunkedValue(std::vector<std::vector<double>>& chunks, 
                                         size_t face_idx, size_t time_idx, double value) {
    size_t chunk_idx = face_idx / chunk_size_;
    size_t local_face_idx = face_idx % chunk_size_;
    size_t local_idx = local_face_idx * time_steps_ + time_idx;
    
    // Bounds checking
    if (chunk_idx >= chunks.size()) {
        debugPrint("Warning: chunk_idx out of bounds in storeChunkedValue: ", 
                  chunk_idx, " >= ", chunks.size());
        return;
    }
    
    if (local_idx >= chunks[chunk_idx].size()) {
        debugPrint("Warning: local_idx out of bounds in storeChunkedValue: ", 
                  local_idx, " >= ", chunks[chunk_idx].size());
        return;
    }
    
    chunks[chunk_idx][local_idx] = value;
}

double IlluminationModel::getChunkedValue(const std::vector<std::vector<double>>& chunks, 
                                         size_t face_idx, size_t time_idx) const {
    if (!using_chunked_storage_) {
        // If we're not using chunked storage, this is an error
        return 0.0;
    }
    
    size_t chunk_idx = face_idx / chunk_size_;
    size_t local_face_idx = face_idx % chunk_size_;
    size_t local_idx = local_face_idx * time_steps_ + time_idx;
    
    // Bounds checking
    if (chunk_idx >= chunks.size()) {
        return 0.0;
    }
    
    if (local_idx >= chunks[chunk_idx].size()) {
        return 0.0;
    }
    
    return chunks[chunk_idx][local_idx];
}

// Accessor methods implementation
double IlluminationModel::getIlluminationValue(size_t face_idx, size_t time_idx) const {
    if (using_chunked_storage_) {
        return getChunkedValue(illumination_chunks_, face_idx, time_idx);
    } else if (face_idx * time_steps_ + time_idx < illumination_values_.size()) {
        return illumination_values_[face_idx * time_steps_ + time_idx];
    }
    return 0.0;
}

double IlluminationModel::getDirectSolarValue(size_t face_idx, size_t time_idx) const {
    if (using_chunked_storage_) {
        return getChunkedValue(direct_solar_chunks_, face_idx, time_idx);
    } else if (face_idx * time_steps_ + time_idx < direct_solar_values_.size()) {
        return direct_solar_values_[face_idx * time_steps_ + time_idx];
    }
    return 0.0;
}

double IlluminationModel::getPrimaryReflectedValue(size_t face_idx, size_t time_idx) const {
    if (using_chunked_storage_) {
        return getChunkedValue(primary_reflected_chunks_, face_idx, time_idx);
    } else if (face_idx * time_steps_ + time_idx < primary_reflected_values_.size()) {
        return primary_reflected_values_[face_idx * time_steps_ + time_idx];
    }
    return 0.0;
}

double IlluminationModel::getPrimaryIRValue(size_t face_idx, size_t time_idx) const {
    if (using_chunked_storage_) {
        return getChunkedValue(primary_ir_chunks_, face_idx, time_idx);
    } else if (face_idx * time_steps_ + time_idx < primary_ir_values_.size()) {
        return primary_ir_values_[face_idx * time_steps_ + time_idx];
    }
    return 0.0;
}

//=============================================================================
// GPU ray tracing support
//=============================================================================

#ifdef PTM_GPU_OPENCL

bool IlluminationModel::enableGPURayTracing(bool enable) {
    if (!enable) {
        use_gpu_ray_tracer_ = false;
        return true;
    }

    if (use_gpu_ray_tracer_ && gpu_ray_pool_ && gpu_ray_pool_->isReady()) {
        // Already enabled
        return true;
    }

    return initializeGPURayTracer();
}

bool IlluminationModel::initializeGPURayTracer() {
    try {
        // Resolve the device list from configureGPURayTracingDevices() state.
        auto all_gpus = gpu::opencl::OpenCLContext::enumerateGPUs(/*only_available=*/true);
        std::vector<gpu::opencl::OpenCLDeviceInfo> selected;
        if (!gpu_device_ids_.empty()) {
            for (int id : gpu_device_ids_) {
                if (id < 0 || static_cast<size_t>(id) >= all_gpus.size()) {
                    std::cerr << "Warning: invalid illumination gpu_device_id " << id
                              << " (only " << all_gpus.size() << " GPUs enumerated); ignoring"
                              << std::endl;
                    continue;
                }
                selected.push_back(all_gpus[id]);
            }
        } else if (gpu_devices_count_ <= 0) {
            // 0 / "auto" → use all available GPUs.
            selected = all_gpus;
        } else if (gpu_devices_count_ == 1) {
            if (!all_gpus.empty()) selected.push_back(all_gpus.front());
        } else {
            const int n = std::min(gpu_devices_count_,
                                   static_cast<int>(all_gpus.size()));
            for (int i = 0; i < n; ++i) selected.push_back(all_gpus[i]);
        }

        if (selected.empty()) {
            std::cerr << "Warning: no GPUs available for ray tracing" << std::endl;
            gpu_ray_pool_.reset();
            use_gpu_ray_tracer_ = false;
            return false;
        }

        gpu_ray_pool_ = std::make_unique<gpu::opencl::BVHGPUPool>(
            std::move(selected), gpu_inter_batch_pause_ms_, /*verbose=*/false);
        if (!gpu_ray_pool_->initialize()) {
            std::cerr << "Warning: BVHGPUPool::initialize() failed; "
                         "falling back to CPU" << std::endl;
            gpu_ray_pool_.reset();
            use_gpu_ray_tracer_ = false;
            return false;
        }

        std::cout << "GPU ray tracer initialized across "
                  << gpu_ray_pool_->numDevices() << " device(s):" << std::endl;
        for (size_t k = 0; k < gpu_ray_pool_->numDevices(); ++k) {
            const auto& info = gpu_ray_pool_->deviceAt(k);
            std::cout << "  [" << k << "] " << info.name << " ("
                      << (info.global_mem_size / (1024 * 1024)) << " MB)"
                      << std::endl;
        }
        if (gpu_inter_batch_pause_ms_ > 0) {
            std::cout << "  Inter-batch pause: " << gpu_inter_batch_pause_ms_
                      << " ms (per device)" << std::endl;
        }

        use_gpu_ray_tracer_ = true;
        gpu_mesh_uploaded_ = false;  // Will upload mesh when needed
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error initializing GPU ray tracer: " << e.what() << std::endl;
        gpu_ray_pool_.reset();
        use_gpu_ray_tracer_ = false;
        return false;
    }
}

void IlluminationModel::configureGPURayTracingDevices(
    int gpu_devices,
    const std::vector<int>& gpu_device_ids,
    int inter_batch_pause_ms)
{
    gpu_devices_count_         = gpu_devices;
    gpu_device_ids_            = gpu_device_ids;
    gpu_inter_batch_pause_ms_  = inter_batch_pause_ms;
}

void IlluminationModel::uploadMeshToGPU() {
    if (!gpu_ray_pool_ || gpu_mesh_uploaded_) return;

    // We need the mesh vertices and indices that were passed to initializeScene
    // These need to be stored or re-extracted. For now, we'll use Embree's scene
    // info to get geometry bounds and assume the mesh data is available.

    // This function should be called after initializeScene() which has the mesh data.
    // The mesh needs to be passed again or stored. For proper integration,
    // we'll modify initializeScene to also handle GPU upload.

    // For now, mark as not uploaded - actual upload happens in initializeScene
    gpu_mesh_uploaded_ = false;
}

IlluminationModel::GPURayTracerStats IlluminationModel::getGPURayTracerStats() const {
    GPURayTracerStats stats;
    if (gpu_ray_pool_) {
        const auto agg = gpu_ray_pool_->getAggregatedStats();
        stats.rays_traced    = agg.rays_traced;
        stats.total_time_ms  = agg.total_time_ms;
        stats.kernel_time_ms = agg.kernel_time_ms;
    }
    return stats;
}

std::vector<uint8_t> IlluminationModel::traceOccludedGPU(
    const std::vector<std::array<double, 3>>& origins,
    const std::vector<std::array<double, 3>>& directions)
{
    const size_t n = origins.size();
    std::vector<uint8_t> blocked(n, 0);

    if (!gpu_ray_pool_ || !gpu_ray_pool_->isReady() || n == 0) {
        // Fall back to CPU
        return traceOccluded(origins, directions, false);
    }

    // Apply scene origin offset if present (same as CPU path)
    std::vector<std::array<double, 3>> local_origins(n);
    for (size_t i = 0; i < n; ++i) {
        local_origins[i] = {
            origins[i][0] - (scene_has_offset_ ? scene_origin_offset_[0] : 0.0),
            origins[i][1] - (scene_has_offset_ ? scene_origin_offset_[1] : 0.0),
            origins[i][2] - (scene_has_offset_ ? scene_origin_offset_[2] : 0.0)
        };
    }

    // Normalize directions (same as CPU path)
    std::vector<std::array<double, 3>> norm_directions(n);
    for (size_t i = 0; i < n; ++i) {
        double dl = std::sqrt(directions[i][0] * directions[i][0] +
                              directions[i][1] * directions[i][1] +
                              directions[i][2] * directions[i][2]);
        double inv_dl = (dl > 0.0) ? (1.0 / dl) : 1.0;
        norm_directions[i] = {
            directions[i][0] * inv_dl,
            directions[i][1] * inv_dl,
            directions[i][2] * inv_dl
        };
    }

    // The pool serializes per-device internally; no outer mutex needed.
    try {
        gpu_ray_pool_->traceShadowRays(local_origins, norm_directions, blocked);
    } catch (const std::exception& e) {
        std::cerr << "GPU ray trace failed: " << e.what() << ", falling back to CPU" << std::endl;
        return traceOccluded(origins, directions, false);
    }

    return blocked;
}

void IlluminationModel::traceOccludedBatchedGPU(
    const std::vector<float>& ray_org_x,
    const std::vector<float>& ray_org_y,
    const std::vector<float>& ray_org_z,
    const std::vector<float>& ray_dir_x,
    const std::vector<float>& ray_dir_y,
    const std::vector<float>& ray_dir_z,
    std::vector<uint8_t>& occluded)
{
    if (!gpu_ray_pool_ || !gpu_ray_pool_->isReady()) {
        occluded.assign(ray_org_x.size(), 1);  // Assume all blocked on failure
        return;
    }

    // Single GPU dispatch - no mutex needed as caller handles synchronization
    gpu_ray_pool_->traceShadowRays(ray_org_x, ray_org_y, ray_org_z,
                                      ray_dir_x, ray_dir_y, ray_dir_z,
                                      occluded);
}

void IlluminationModel::calculateDirectSolarBatchWithFiniteDiskGPU(
    const std::vector<std::array<double, 3>>& face_centers,
    const std::vector<std::array<double, 3>>& face_normals,
    const std::array<double, 3>& sun_position,
    const std::array<double, 3>& primary_position,
    std::vector<double>& direct_solar_values,
    int num_samples)
{
    const size_t n = face_centers.size();
    direct_solar_values.assign(n, 0.0);
    if (n == 0) return;

    // Convert SPICE positions from km to meters for direction computation
    const std::array<double,3> sun_pos_m = kmToMeters(sun_position);
    // PTM-36: the eclipse sphere must live in MESH-NATIVE units (km for global
    // icosphere meshes, m for DEM/site meshes) -- center AND radius together.
    // Kept identical to the CPU batch path so CPU/GPU stay equivalent.
    const double km_to_native = kmToNativeScale(face_centers[0]);
    const std::array<double,3> primary_pos_n = {
        primary_position[0] * km_to_native,
        primary_position[1] * km_to_native,
        primary_position[2] * km_to_native
    };
    const double primary_radius_n = primary_radius_ * km_to_native;

    // Hash function for decorrelation
    auto hash32 = [](uint32_t x) -> uint32_t {
        x ^= x >> 17; x *= 0xed5ad4bbU; x ^= x >> 11; x *= 0xac4c1b51U; x ^= x >> 15; x *= 0x31848babU; x ^= x >> 14;
        return x;
    };
    auto u01 = [&](uint32_t s) -> double { return (hash32(s) & 0xFFFFFFU) / double(0x1000000U); };

    const uint32_t N = std::max(8, num_samples);
    // PTM-38: penumbral faces are sampled heavily (mirrors the CPU paths exactly
    // so CPU and GPU stay equivalent). Only faces the primary actually overlaps
    // pay for it, so a run without an eclipse is byte-identical.
    const uint32_t heavyN = heavySolarSamples(num_samples);   // PTM-45(ii)

    // ==== PHASE 1: Generate all rays (parallel, CPU) ====
    // We'll collect all rays from all faces, then batch them to GPU.
    // For each face, we may generate 1 (center ray) or N (disk samples) rays.
    // Track which rays belong to which face using per-face ray ranges.

    // Pre-compute per-face info: sun direction, incidence angle, ray origin
    struct FaceInfo {
        std::array<double, 3> sun_dir;
        std::array<double, 3> ray_origin;
        double sun_dist_m;   // meters — for the ray geometry
        double sun_dist_km;  // km — for angular size and irradiance
        double sun_dist_n;   // mesh-native units — far bound for eclipse checks (PTM-36)
        double cos_inc0;
        double disk_scale;
        bool below_horizon;     // Entire disk below horizon
        bool eclipse_overlap;   // PTM-38: primary overlaps ANY part of the solar
                                // disk (penumbra or umbra), not just the center ray
        bool needs_sampling;    // Needs finite-disk sampling
        bool precheck_occluded; // Analytic body-occlusion pre-check culled it
                                // (OPEN meshes only; see below)
    };
    std::vector<FaceInfo> face_info(n);

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        const auto& fc = face_centers[i];
        const auto& fn = face_normals[i];
        FaceInfo& fi = face_info[i];

        // Sun vector from this face (both in meters)
        std::array<double, 3> sun_rel = {
            sun_pos_m[0] - fc[0],
            sun_pos_m[1] - fc[1],
            sun_pos_m[2] - fc[2]
        };
        fi.sun_dist_m = vectorLength(sun_rel);
        fi.sun_dist_km = fi.sun_dist_m / 1000.0;
        fi.sun_dist_n = fi.sun_dist_m * (km_to_native / 1000.0);  // PTM-36

        if (fi.sun_dist_m <= constants::EPSILON) {
            fi.below_horizon = true;
            fi.needs_sampling = false;
            fi.precheck_occluded = false;
            continue;
        }

        fi.sun_dir = { sun_rel[0]/fi.sun_dist_m, sun_rel[1]/fi.sun_dist_m, sun_rel[2]/fi.sun_dist_m };
        fi.cos_inc0 = dotProduct(fn, fi.sun_dir);

        // Solar disk geometry (SUN_RADIUS in km, use km distance)
        double s = constants::SUN_RADIUS / fi.sun_dist_km;
        if (s > 1.0) s = 1.0;
        fi.disk_scale = s;  // sin(asin(s)) = s

        // Check if entire disk is below horizon
        if (fi.cos_inc0 <= -s) {
            fi.below_horizon = true;
            fi.needs_sampling = false;
            fi.precheck_occluded = false;
            continue;
        }
        fi.below_horizon = false;
        fi.precheck_occluded = false;

        // Analytic body-occlusion pre-check, TOPOLOGY-gated to OPEN meshes only,
        // mirroring the CPU paths EXACTLY so CPU and GPU stay equivalent. On a
        // CLOSED mesh the GPU BVH self-occludes (no pre-check needed, historical
        // correct behavior); on an OPEN micro-patch the BVH has no far-side
        // geometry, so without this the GPU path leaks sun through the bulk body.
        // Cull only when the body sphere lies BETWEEN the face and the sun
        // (forward ray-sphere hit at t>0); R_occ = body-level radius. This forward
        // test excludes sunward-tilted (mu>0) facets just past the terminator so
        // no lit face is lost. A culled face generates NO rays and routes to
        // shadow-flux/zero. See memory ptm-mu-gate-open-mesh-selfocclusion-fail.
        if (mesh_is_open_ && body_radius_occ_native_ > 0.0) {
            const double fc_dot_sun = fc[0]*fi.sun_dir[0] + fc[1]*fi.sun_dir[1] + fc[2]*fi.sun_dir[2];
            if (fc_dot_sun < 0.0) {  // cheap early-out; a forward hit implies this
                const std::array<double,3> body_center = {0.0, 0.0, 0.0};
                double tb1, tb2;
                if (raySphereIntersect(fc, fi.sun_dir, body_center,
                                       body_radius_occ_native_, tb1, tb2)) {
                    if (tb1 > 0.0 || tb2 > 0.0) {
                        fi.precheck_occluded = true;
                        fi.needs_sampling = false;
                        fi.eclipse_overlap = false;
                        continue;
                    }
                }
            }
        }

        // Compute robust ray origin
        const double eps_base = calculateSafeRayOffset(fc, fn);
        const double eps_n = 2.0 * eps_base;
        const double eps_d = 8.0 * eps_base;

        fi.ray_origin = {
            fc[0] + fn[0] * eps_n + fi.sun_dir[0] * eps_d,
            fc[1] + fn[1] * eps_n + fi.sun_dir[1] * eps_d,
            fc[2] + fn[2] * eps_n + fi.sun_dir[2] * eps_d
        };

        // PTM-38: eclipse test on the WHOLE solar disk, not just the center ray
        // (mesh-native units). The center-ray-only test reported full sun for
        // every face less than ~50 % obscured. Strict superset of the old test.
        fi.eclipse_overlap = false;
        if (!primary_is_sun_ && fi.cos_inc0 > 0.0) {
            fi.eclipse_overlap = primaryDiskOverlap(fi.ray_origin, fi.sun_dir,
                                                    primary_pos_n, primary_radius_n,
                                                    fi.sun_dist_n,
                                                    std::asin(fi.disk_scale));
        }

        // Determine if we need sampling (if center is below horizon or might be blocked)
        fi.needs_sampling = (fi.cos_inc0 <= 0.0) || fi.eclipse_overlap;
    }

    // ==== Collect rays for all faces ====
    // We'll use a two-pass approach:
    // Pass 1: Count rays per face to compute offsets
    // Pass 2: Fill in ray data

    // Ray counts: center_ray (1) or disk_samples (up to N)
    std::vector<size_t> ray_counts(n);
    std::vector<size_t> ray_offsets(n + 1);

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        const FaceInfo& fi = face_info[i];
        if (fi.below_horizon || fi.precheck_occluded) {
            ray_counts[i] = 0;
        } else if (!fi.needs_sampling && fi.cos_inc0 > 0.0 && !fi.eclipse_overlap) {
            // Center ray fast path - just 1 ray
            ray_counts[i] = 1;
        } else {
            // Disk sampling - estimate N rays (actual may be less due to eclipse filtering).
            // PTM-38: penumbral faces are sampled with heavyN; this expression must
            // stay identical to the per-face Nf used in PHASE 1b or the ray offsets
            // computed from it will not line up.
            ray_counts[i] = fi.eclipse_overlap ? heavyN : N;
        }
    }

    // Compute prefix sum for ray offsets
    ray_offsets[0] = 0;
    for (size_t i = 0; i < n; ++i) {
        ray_offsets[i + 1] = ray_offsets[i] + ray_counts[i];
    }
    const size_t total_rays_estimate = ray_offsets[n];

    // Allocate ray buffers (SoA format for GPU)
    std::vector<float> ray_org_x(total_rays_estimate);
    std::vector<float> ray_org_y(total_rays_estimate);
    std::vector<float> ray_org_z(total_rays_estimate);
    std::vector<float> ray_dir_x(total_rays_estimate);
    std::vector<float> ray_dir_y(total_rays_estimate);
    std::vector<float> ray_dir_z(total_rays_estimate);

    // Track actual ray counts and eclipse flags per face
    // These will be used in result processing
    struct FaceRayInfo {
        size_t ray_start;       // Starting index in global ray array
        size_t ray_count;       // Actual number of rays generated
        std::vector<uint8_t> eclipse_skip;  // Which disk samples were eclipsed (not traced)
        std::vector<std::array<double, 3>> ray_dirs;  // Ray directions for result processing
    };
    std::vector<FaceRayInfo> face_ray_info(n);

    // ==== PHASE 1b: Generate rays (parallel) ====
    #pragma omp parallel for schedule(dynamic, 256)
    for (size_t i = 0; i < n; ++i) {
        const FaceInfo& fi = face_info[i];
        FaceRayInfo& fri = face_ray_info[i];
        fri.ray_start = ray_offsets[i];
        fri.ray_count = 0;

        if (fi.below_horizon || fi.precheck_occluded) continue;

        const auto& fc = face_centers[i];
        const auto& fn = face_normals[i];

        // Apply scene origin offset for GPU (same as traceOccludedGPU)
        auto apply_offset = [this](const std::array<double, 3>& p) -> std::array<double, 3> {
            return {
                p[0] - (scene_has_offset_ ? scene_origin_offset_[0] : 0.0),
                p[1] - (scene_has_offset_ ? scene_origin_offset_[1] : 0.0),
                p[2] - (scene_has_offset_ ? scene_origin_offset_[2] : 0.0)
            };
        };

        if (!fi.needs_sampling && fi.cos_inc0 > 0.0 && !fi.eclipse_overlap) {
            // Center ray fast path - single ray
            size_t idx = fri.ray_start;
            auto local_origin = apply_offset(fi.ray_origin);

            ray_org_x[idx] = static_cast<float>(local_origin[0]);
            ray_org_y[idx] = static_cast<float>(local_origin[1]);
            ray_org_z[idx] = static_cast<float>(local_origin[2]);
            ray_dir_x[idx] = static_cast<float>(fi.sun_dir[0]);
            ray_dir_y[idx] = static_cast<float>(fi.sun_dir[1]);
            ray_dir_z[idx] = static_cast<float>(fi.sun_dir[2]);

            fri.ray_count = 1;
            fri.ray_dirs.push_back(fi.sun_dir);
            fri.eclipse_skip.clear();
        } else {
            // Finite disk sampling
            std::array<double, 3> u, v;
            createCoordinateSystem(fi.sun_dir, u, v);

            // Per-face decorrelation
            uint32_t seed = 0x9E3779B1u ^ uint32_t(i);
            double theta = u01(seed) * 2.0 * M_PI;
            std::array<double, 3> ur = {
                u[0] * std::cos(theta) + v[0] * std::sin(theta),
                u[1] * std::cos(theta) + v[1] * std::sin(theta),
                u[2] * std::cos(theta) + v[2] * std::sin(theta)
            };
            std::array<double, 3> vr = {
                -u[0] * std::sin(theta) + v[0] * std::cos(theta),
                -u[1] * std::sin(theta) + v[1] * std::cos(theta),
                -u[2] * std::sin(theta) + v[2] * std::cos(theta)
            };
            double uoff = u01(seed ^ 0x85EBCA6Bu);
            double voff = u01(seed ^ 0xC2B2AE35u);

            // PTM-38: heavy sampling for penumbral faces (must match ray_counts[i])
            const uint32_t Nf = fi.eclipse_overlap ? heavyN : N;
            fri.eclipse_skip.reserve(Nf);
            fri.ray_dirs.reserve(Nf);

            size_t idx = fri.ray_start;
            const double eps_base = calculateSafeRayOffset(fc, fn);
            const double eps_d = 8.0 * eps_base;

            for (uint32_t k = 0; k < Nf; ++k) {
                // Hammersley sampling
                double su = double(k) / double(Nf);
                double sv = 0.0;
                uint32_t bits = k;
                double inv = 0.5;
                while (bits) {
                    sv += (bits & 1) * inv;
                    bits >>= 1;
                    inv *= 0.5;
                }
                su = std::fmod(su + uoff, 1.0);
                sv = std::fmod(sv + voff, 1.0);

                // Square to concentric disk mapping
                double a = 2.0 * su - 1.0;
                double b = 2.0 * sv - 1.0;
                double dx, dy;
                if (a * a > b * b) {
                    double r = a;
                    double phi = (M_PI / 4.0) * (b / a);
                    dx = r * std::cos(phi);
                    dy = r * std::sin(phi);
                } else if (b != 0.0) {
                    double r = b;
                    double phi = (M_PI / 2.0) - (M_PI / 4.0) * (a / b);
                    dx = r * std::cos(phi);
                    dy = r * std::sin(phi);
                } else {
                    dx = dy = 0.0;
                }

                if (dx*dx + dy*dy > 1.0) continue;

                const double ox = fi.disk_scale * dx;
                const double oy = fi.disk_scale * dy;
                std::array<double, 3> sd = {
                    fi.sun_dir[0] + ox * ur[0] + oy * vr[0],
                    fi.sun_dir[1] + ox * ur[1] + oy * vr[1],
                    fi.sun_dir[2] + ox * ur[2] + oy * vr[2]
                };
                double l = vectorLength(sd);
                if (l <= constants::EPSILON) continue;
                sd[0] /= l; sd[1] /= l; sd[2] /= l;

                // Check eclipse (mesh-native units)
                bool eclipsed = false;
                if (!primary_is_sun_) {
                    double t1, t2;
                    if (raySphereIntersect(fi.ray_origin, sd, primary_pos_n, primary_radius_n, t1, t2)) {
                        if ((t1 > 0.0 && t1 < fi.sun_dist_n) || (t2 > 0.0 && t2 < fi.sun_dist_n)) {
                            eclipsed = true;
                        }
                    }
                }

                fri.eclipse_skip.push_back(eclipsed ? 1 : 0);
                fri.ray_dirs.push_back(sd);

                if (!eclipsed) {
                    // Generate ray origin offset along sample direction
                    std::array<double, 3> sample_origin = {
                        fi.ray_origin[0] + sd[0] * eps_d,
                        fi.ray_origin[1] + sd[1] * eps_d,
                        fi.ray_origin[2] + sd[2] * eps_d
                    };
                    auto local_origin = apply_offset(sample_origin);

                    ray_org_x[idx] = static_cast<float>(local_origin[0]);
                    ray_org_y[idx] = static_cast<float>(local_origin[1]);
                    ray_org_z[idx] = static_cast<float>(local_origin[2]);
                    ray_dir_x[idx] = static_cast<float>(sd[0]);
                    ray_dir_y[idx] = static_cast<float>(sd[1]);
                    ray_dir_z[idx] = static_cast<float>(sd[2]);

                    idx++;
                    fri.ray_count++;
                }
            }
        }
    }

    // Compute actual total rays
    size_t total_rays = 0;
    for (size_t i = 0; i < n; ++i) {
        total_rays += face_ray_info[i].ray_count;
    }

    if (total_rays == 0) {
        // All faces in shadow or below horizon
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            if (enable_shadow_flux_) {
                direct_solar_values[i] = calculateShadowFlux(face_centers[i], face_normals[i], sun_pos_m);
            }
        }
        return;
    }

    // Compact ray arrays to remove unused slots
    // Build mapping from original indices to compacted indices
    std::vector<size_t> compact_offsets(n + 1);
    compact_offsets[0] = 0;
    for (size_t i = 0; i < n; ++i) {
        compact_offsets[i + 1] = compact_offsets[i] + face_ray_info[i].ray_count;
    }

    std::vector<float> compact_org_x(total_rays);
    std::vector<float> compact_org_y(total_rays);
    std::vector<float> compact_org_z(total_rays);
    std::vector<float> compact_dir_x(total_rays);
    std::vector<float> compact_dir_y(total_rays);
    std::vector<float> compact_dir_z(total_rays);

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        const FaceRayInfo& fri = face_ray_info[i];
        if (fri.ray_count == 0) continue;

        size_t src = fri.ray_start;
        size_t dst = compact_offsets[i];

        for (size_t j = 0; j < fri.ray_count; ++j) {
            compact_org_x[dst + j] = ray_org_x[src + j];
            compact_org_y[dst + j] = ray_org_y[src + j];
            compact_org_z[dst + j] = ray_org_z[src + j];
            compact_dir_x[dst + j] = ray_dir_x[src + j];
            compact_dir_y[dst + j] = ray_dir_y[src + j];
            compact_dir_z[dst + j] = ray_dir_z[src + j];
        }
        // Update face_ray_info to use compacted indices
        const_cast<FaceRayInfo&>(fri).ray_start = dst;
    }

    // ==== PHASE 2: GPU ray tracing (single batch; pool partitions across devices) ====
    std::vector<uint8_t> occluded;
    traceOccludedBatchedGPU(compact_org_x, compact_org_y, compact_org_z,
                             compact_dir_x, compact_dir_y, compact_dir_z,
                             occluded);

    // ==== PHASE 3: Process results (parallel) ====
    #pragma omp parallel for schedule(dynamic, 256)
    for (size_t i = 0; i < n; ++i) {
        const FaceInfo& fi = face_info[i];
        const FaceRayInfo& fri = face_ray_info[i];
        const auto& fc = face_centers[i];
        const auto& fn = face_normals[i];

        if (fi.below_horizon || fi.precheck_occluded) {
            // Below horizon, or culled by the analytic body-occlusion pre-check
            // (OPEN meshes) -- no rays traced; treat as shadowed.
            direct_solar_values[i] = enable_shadow_flux_ ? calculateShadowFlux(fc, fn, sun_pos_m) : 0.0;
            continue;
        }

        if (fri.ray_count == 0) {
            // All samples were eclipsed
            direct_solar_values[i] = enable_shadow_flux_ ? calculateShadowFlux(fc, fn, sun_pos_m) : 0.0;
            continue;
        }

        double sun_dist_au = fi.sun_dist_km / constants::KM_PER_AU;
        double solar_irr = constants::SOLAR_CONSTANT / (sun_dist_au * sun_dist_au);

        if (!fi.needs_sampling && fi.cos_inc0 > 0.0 && !fi.eclipse_overlap) {
            // Center ray result
            bool blocked = (fri.ray_start < occluded.size()) && (occluded[fri.ray_start] != 0);
            if (!blocked) {
                direct_solar_values[i] = solar_irr * fi.cos_inc0;
            } else {
                direct_solar_values[i] = enable_shadow_flux_ ? calculateShadowFlux(fc, fn, sun_pos_m) : 0.0;
            }
        } else {
            // Process disk sampling results
            double sum_cos = 0.0;
            uint32_t total_valid = 0;
            size_t occ_idx = fri.ray_start;

            for (size_t j = 0; j < fri.eclipse_skip.size(); ++j) {
                if (fri.eclipse_skip[j]) continue;  // Eclipsed, not traced

                bool blocked = (occ_idx < occluded.size()) && (occluded[occ_idx] != 0);
                occ_idx++;
                total_valid++;

                if (!blocked) {
                    double c = dotProduct(fn, fri.ray_dirs[j]);
                    if (c > 0.0) sum_cos += c;
                }
            }

            if (total_valid == 0) {
                direct_solar_values[i] = enable_shadow_flux_ ? calculateShadowFlux(fc, fn, sun_pos_m) : 0.0;
            } else {
                // PTM-38: normalize by the ATTEMPTED disk samples, not the
                // survivors. total_valid counts only the non-eclipse-skipped
                // samples, so a 95 %-obscured face renormalized its surviving
                // 5 % back up to full sun. Terrain-blocked samples were always
                // kept in the denominator; only eclipse-skipped ones were
                // dropped. Identical to total_valid outside an eclipse. Mirrors
                // the CPU batch path exactly.
                double flux = solar_irr * (sum_cos / double(fri.eclipse_skip.size()));
                if (flux <= 0.0 && enable_shadow_flux_) {
                    flux = calculateShadowFlux(fc, fn, sun_pos_m);
                }
                direct_solar_values[i] = flux;
            }
        }
    }
}

#endif  // PTM_GPU_OPENCL

} // namespace ptm