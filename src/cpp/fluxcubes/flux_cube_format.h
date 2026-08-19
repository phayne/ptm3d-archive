// flux_cube_format.h
// ---------------------------------------------------------------------------
// On-disk contracts for the flux-cube product:
//
//   .fcg  geometry cache   -- written by ptm_flux_cube_geom, read by the engine
//   .fcb  flux output      -- written by the engine, read by the HDF5 packager
//
// Both are little-endian, self-describing, and versioned. The machine-critical
// fields live in a FIXED BINARY struct so no C++ side ever needs a JSON parser;
// a trailing JSON blob carries provenance strings for humans and for the Python
// packager, and the C++ engine echoes it without parsing it.
//
// A flux cube is a MEASUREMENT FIXTURE, not a body. It does not occlude, emit,
// scatter, or appear in the view-factor matrix, and cubes never see each other.
// Everything reported is INCIDENT irradiance in W/m^2 -- no albedo, no
// emissivity, no cube material is applied anywhere. Consumers bolt their own
// optical model on top.
// ---------------------------------------------------------------------------
#ifndef PTM_FLUX_CUBE_FORMAT_H
#define PTM_FLUX_CUBE_FORMAT_H

#include <cstdint>

namespace ptm {
namespace fluxcubes {

// ---------------------------------------------------------------------------
// Face axis. FIXED ORDER, NEVER REORDERED -- engineers bolt hardware to this.
// Normals are in the site-local frame (E, N, U) defined at the grid center.
//   0 zenith +U   1 nadir -U   2 north +N   3 south -N   4 east +E   5 west -E
// ---------------------------------------------------------------------------
enum FaceIndex : int {
    kFaceZenith = 0, kFaceNadir = 1, kFaceNorth = 2,
    kFaceSouth  = 3, kFaceEast  = 4, kFaceWest  = 5,
    kNumFaces = 6
};

inline const char* faceName(int f) {
    switch (f) {
        case kFaceZenith: return "zenith";
        case kFaceNadir:  return "nadir";
        case kFaceNorth:  return "north";
        case kFaceSouth:  return "south";
        case kFaceEast:   return "east";
        case kFaceWest:   return "west";
        default:          return "invalid";
    }
}

// ---------------------------------------------------------------------------
// Component axis. FIXED ORDER. Index 5 (total) is the sum of 0..4.
// ---------------------------------------------------------------------------
enum ComponentIndex : int {
    kCompDirect      = 0,   // direct solar beam
    kCompTerrainVis  = 1,   // terrain-scattered visible
    kCompTerrainIR   = 2,   // terrain thermal IR
    kCompPrimaryRefl = 3,   // primary body (e.g. Earth) reflected sunlight
    kCompPrimaryIR   = 4,   // primary body thermal IR
    kCompTotal       = 5,
    kNumComponents = 6
};

inline const char* componentName(int c) {
    switch (c) {
        case kCompDirect:      return "direct";
        case kCompTerrainVis:  return "terrain_vis";
        case kCompTerrainIR:   return "terrain_ir";
        case kCompPrimaryRefl: return "primary_refl";
        case kCompPrimaryIR:   return "primary_ir";
        case kCompTotal:       return "total";
        default:               return "invalid";
    }
}

// Sampler identifiers. The cache key includes this, so changing the sampler
// invalidates caches instead of silently mixing point sets.
enum SamplerId : uint32_t {
    kSamplerShirleyChiuCP = 1   // stratified Shirley-Chiu + per-patch Cranley-Patterson
};

#pragma pack(push, 1)

// ---------------------------------------------------------------------------
// .fcg -- geometry cache header. Followed by, in order:
//   CubeRecord   cubes  [n_cubes]
//   PatchRecord  patches[n_cubes * kNumFaces]      (cube-major, face-inner)
//   int32_t      row_cols[total_nnz]               (CORE face ids)
//   float        row_vals[total_nnz]               (view factors F[c][p][j])
//   char         json[json_len]                    (provenance, not parsed by C++)
//   char         names[n_cubes][kNameLen]          ONLY when has_names == 1
// ---------------------------------------------------------------------------
struct FcgHeader {
    char     magic[4];            // "FCG1"
    uint32_t version;             // 1
    uint32_t n_cubes;
    uint32_t n_faces_per_cube;    // == kNumFaces
    uint64_t total_nnz;

    // --- cache key: all of these must match the run, or the engine REFUSES.
    double   geomhash_core;       // sum_i (A_i + (i+1)*A_i*1e-9), the same hash the
                                  // PTM_HMATRIX_VF_REPLACE gate uses
    uint64_t n_faces_core;
    uint64_t ring_hash;           // FNV-1a 64 over the occluder PLY bytes; 0 = no ring.
                                  // NOT cryptographic -- a cache key, not a signature.
    double   center_lat_deg;
    double   center_lon_deg;
    double   spacing_m;
    double   radius_m;
    double   height_m;            // ACHIEVED plumb-line height actually used
    double   edge_m;              // ACHIEVED cube edge (metadata + probe spread only)
    uint32_t n_rays;              // per patch
    uint32_t sampler_id;
    uint64_t seed;
    double   truncate_rel;        // 0 = keep every hit face (recommended)

    // --- how the geometry was resolved (recorded, so a file explains itself)
    double   height_factor;       // 0 when an absolute height_m override was used
    double   edge_factor;         // 0 when an absolute edge_m override was used
    double   local_median_edge_m; // the length scale the factors multiplied

    // --- site frame, computed not assumed
    double   frame_origin[3];     // C projected to the surface, body-fixed meters
    double   frame_east[3];
    double   frame_north[3];
    double   frame_up[3];
    double   frame_max_tilt_deg;  // max angle between the site U and each cube's own radial

    uint32_t json_len;
    // 1 => a names block of n_cubes * kNameLen bytes is appended AFTER the JSON
    // blob. Placed last on purpose: every reader written against the original
    // layout parses header, cubes, patches, rows and JSON at unchanged offsets
    // and simply ignores the trailing bytes, so adding named locations did not
    // break the format or require a version bump. (This slot was `reserved`.)
    uint32_t has_names;
};

// Maximum length of a named-location label, including the terminating NUL.
constexpr int kNameLen = 32;

// Sentinel grid index marking a cube that came from a NAMED LOCATION rather
// than the lattice. Both grid_i and grid_j are set to it, so the two kinds are
// distinguishable without consulting the names block.
constexpr int32_t kNamedGridIndex = -2147483647 - 1;

// One cube: placement and the mesh face beneath it.
struct CubeRecord {
    // For a LATTICE cube these are the grid indices and x = i*spacing,
    // y = j*spacing. For a NAMED LOCATION both are set to kNamedGridIndex
    // (-2147483648) so the two kinds are distinguishable without consulting the
    // names block, and x_m/y_m carry its resolved site-frame offsets.
    int32_t  grid_i, grid_j;
    double   x_m, y_m;            // site-frame offsets from C (east, north)
    double   center[3];           // cube center, body-fixed meters
    int32_t  face_id;             // CORE mesh face beneath (from the drop ray)
    float    edge_m;              // PER-CUBE cube edge; sets the shadow-probe
                                  // spread. Equals the cache-level edge for
                                  // lattice cubes; a named location may override
                                  // it. (This slot was an unused pad, so adding
                                  // it changed no offset and no struct size.)
    float    slope_deg;           // acos(n_j . U)
    float    aspect_deg;          // atan2(n_j . E, n_j . N), clockwise from north
    float    area_m2;             // area of the underlying face
    float    elev_m;              // (P_k - C) . U, terrain elevation relative to C
    double   height_achieved_m;   // |x_c - P_k| along U (equals header height_m)
};

// One oriented differential patch (one face of one cube).
struct PatchRecord {
    uint64_t row_offset;          // index into row_cols / row_vals
    uint32_t row_len;             // number of distinct CORE partner faces
    uint32_t n_faces_50pct;       // partners needed to reach 50% of the terrain weight
    float    f_sky;               // cosine-weighted fraction escaping to space
    float    f_ring;              // fraction hitting the occluder ring (NO radiosity there)
    float    f_max;               // largest single view-factor entry
    float    closure_residual;    // |sum_j F + f_sky + f_ring - 1|; bookkeeping, see below
    float    splithalf_tv;        // 0.5*sum_j |F_a[j] - F_b[j]| between two independent
                                  // half-rows. This is the MC error metric that matters:
                                  // |E_a - E_b| <= splithalf_tv * max_j B_j, a real flux bound.
    float    splithalf_fsky;      // |f_sky_a - f_sky_b|
};

// ---------------------------------------------------------------------------
// .fcb -- flux output header. Followed by, in order:
//   double flux   [n_cubes][kNumFaces][kNumComponents][n_steps]
//   double t_face [n_cubes][n_steps]
//   uint8  penumbral[n_cubes][n_steps]
//   float  f_sky  [n_cubes][kNumFaces]
//   float  f_ring [n_cubes][kNumFaces]
//   float  f_max  [n_cubes][kNumFaces]
//   CubeRecord cubes[n_cubes]
//   double epoch_et [n_steps]
//   char   epoch_utc[n_steps][24]
//   char   json[json_len]
//   char   names[n_cubes][kNameLen]        ONLY when has_names == 1
// float64 for flux is deliberate: the dynamic range at a polar site runs from
// ~0.01 W/m^2 (PSR Earthshine) to ~1000 W/m^2 (noon), and the PSR end is the
// end that sizes a radiator.
// ---------------------------------------------------------------------------
struct FcbHeader {
    char     magic[4];            // "FCB1"
    uint32_t version;             // 1
    uint32_t n_cubes;
    uint32_t n_faces_per_cube;    // == kNumFaces
    uint32_t n_components;        // == kNumComponents
    uint32_t n_steps;             // output steps only; equilibration EXCLUDED
    uint64_t equilibration_steps;
    uint64_t total_time_steps;
    double   height_m;
    double   edge_m;
    double   spacing_m;
    double   radius_m;
    double   frame_origin[3];
    double   frame_east[3];
    double   frame_north[3];
    double   frame_up[3];
    uint32_t json_len;
    // 1 => a names block of n_cubes * kNameLen bytes follows the JSON, same
    // trailing-block convention as the .fcg. (This slot was `reserved`.)
    uint32_t has_names;
};

#pragma pack(pop)

// A note on closure_residual, so nobody reads more into it than is there:
// the estimator is F[j] = (rays hitting j)/N, f_sky = (escaping)/N,
// f_ring = (hitting ring)/N, so the three channels sum to exactly 1 by
// construction. The residual therefore tests BOOKKEEPING -- a dropped or
// double-counted channel -- and not sampling accuracy. splithalf_tv is the
// quantity that measures accuracy, and it is the one the ray budget is set from.

}  // namespace fluxcubes
}  // namespace ptm

#endif  // PTM_FLUX_CUBE_FORMAT_H
