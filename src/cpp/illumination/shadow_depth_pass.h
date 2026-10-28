// shadow_depth_pass.h
// ---------------------------------------------------------------------------
// SHADOW DEPTH: the maximum height above the local surface at which the Sun is
// not even partially visible, per face and per output epoch.
//
// THE DEFINITION, in Paul's words (2026-09), which are exact:
//   "The maximum height above the local surface where the Sun is not even
//    partially visible, where height is measured along the vertical (radial to
//    the Moon body center) rather than surface normal."
//
// Two halves of that sentence do all the work:
//
//  1. "NOT EVEN PARTIALLY VISIBLE" is the UMBRAL ceiling, not the penumbral
//     one. The quantity is the greatest h at which the visible solar fraction
//     is exactly zero; one height step above it a sliver of limb is showing.
//     The Sun is a finite disk of angular radius ~0.2666 deg at 1 AU, so this
//     is NOT a single ray to the disk center -- the penumbral eclipse removed exactly that
//     test from the engine's direct-solar path because it "broke the penumbra
//     at both ends". The disk test here is the engine's own ladder, called
//     through illumination/solar_disk_visibility.h, which the flux-cube pass
//     also calls. One spelling, two products.
//
//     WHAT THE ANGULAR RADIUS IS WORTH, so nobody drops it as a detail: the
//     umbral ceiling behind an occluder at downrange distance d is
//         h* = max(0, Z - d*tan(eps + theta_sun))
//     and omitting theta_sun moves the answer by d*sec^2(eps)*theta_sun --
//     0.19 m at d = 20 m and eps = 45 deg, nearly ten times the search
//     resolution. A closed form written with eps alone disagrees with correct
//     code by exactly that, and it looks like a bug in the code.
//
//  2. "ALONG THE VERTICAL (RADIAL TO THE BODY CENTER)" -- not along the face
//     normal. The vertical at a face is the unit vector from the body center
//     through the face centroid, which the engine already has (the mesh frame
//     is body-centered; illumination_model.cpp measures a face's radius as
//     |face_center|). It is exact, needs no site frame, and works on a global
//     mesh. It also agrees with how ptm_flux_cube_geom places its cubes, so
//     the two products stack by construction instead of by coincidence.
//
// WHY THIS IS A STANDALONE PASS
// -----------------------------
// Same reason flux_cube_pass is: the per-step illumination loop has three
// dispatch routes (single-face, CPU batch, GPU batch) and hanging a height
// search inside it means three insertion points that must agree forever. After
// the illumination phase has populated its series, all three routes have
// converged onto one set of arrays, and that confluence is what makes a pass
// route-blind.
//
// EVERYTHING IS INJECTED, AND IT REFUSES RATHER THAN SUBSTITUTES
// --------------------------------------------------------------
// Ray casting, the per-face ray offset, and the resolved Embree `tnear` are
// passed in by the caller. The module is therefore unit-testable against
// synthetic geometry with no Embree, no SPICE and no GPU, and it cannot grow a
// second spelling of an engine formula. A missing ray caster, a scene with no
// origin offset, a `tnear` too coarse for the search, a zero solar radius, an
// inconsistent time axis -- each is a hard error with a reason, never a quiet
// downgrade.
//
// ⛔ THE `tnear` TRAP, which is the single thing most likely to produce a
// plausible and completely wrong field. The batched occlusion path computes its
// ray `tnear` from the distance to the SCENE origin
// (illumination_model.cpp:1164-1171, `max(1e-6*r, 0.01)`). With the scene origin
// offset applied, r is meters and tnear is the 1 cm floor. WITHOUT it,
// r ~ 1.737e6 m on a lunar mesh and tnear ~ 1.74 m -- every ray would skip the
// first 1.74 m of terrain, and a 0-5 m vertical search would return confident
// garbage with no error anywhere. That is the
// ptm-shadow-ray-1cm-offset-floor-psr-leak class recurring, so this pass
// ASSERTS the offset, PRINTS the resolved tnear, and REFUSES when
// tnear > h_max_m/100.
// ---------------------------------------------------------------------------
#ifndef PTM_SHADOW_DEPTH_PASS_H
#define PTM_SHADOW_DEPTH_PASS_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <vector>

namespace ptm {
namespace shadowdepth {

// Sentinel encoding of the per-face, per-epoch field. float32 throughout.
//   0.0f      the Sun is at least partially visible AT the surface
//   finite >0 the umbral ceiling, meters above the surface along the vertical
//   +inf      no clear height was found at or below h_max_m. This is the
//             MAJORITY value inside a permanently shadowed region BY DESIGN --
//             a PSR floor is umbral all the way up, and inventing a number for
//             it would be inventing data. Report it structurally, through
//             shadow_unresolved_frac.
//   NaN       the face was not evaluated at this epoch
// `NaN` is deliberately NOT used for "never lit". There is no number for that;
// the fractions say it.
constexpr float kUnresolved = std::numeric_limits<float>::infinity();

struct PassInputs {
    // --- geometry, in the MESH's own frame (body-centered) ----------------
    const std::vector<std::array<double, 3>>* face_centers = nullptr;
    // Normals are used ONLY by the consistency gate, to separate the population
    // where this field and `direct_solar` legitimately differ (a facet tilted
    // away from the Sun under a clear sky has zero direct flux while a person
    // standing on it is in sunlight). They never enter the height search.
    const std::vector<std::array<double, 3>>* face_normals = nullptr;
    size_t n_faces = 0;

    // --- time axis --------------------------------------------------------
    // MUST be the same (times, time_strings) pair handed to writeTimeStepsFile.
    // Re-deriving the grid from start_time + i*dt is the output-epoch drift defect.
    const std::vector<double>* times = nullptr;
    const std::vector<std::string>* time_strings = nullptr;
    int equilibration_steps = 0;
    int total_time_steps = 0;

    // --- ephemeris, indexed by ABSOLUTE step, in KILOMETERS ---------------
    const std::vector<std::array<double, 3>>* sun_positions = nullptr;
    // ⛔ UNITS. sun_positions is km; the mesh frame is METERS for lunar/Europa
    // DEMs and KILOMETERS for asteroid shape models. Subtracting one from the
    // other without this factor points the ray into the ground and every probe
    // reads as occluded -- a field that is entirely sentinel, with no error.
    double mesh_to_km = 1.0;
    // Solar radius in km. 0 is REFUSED: the umbra is defined by the finite
    // disk, and a point sun would silently compute the penumbral ceiling
    // instead, which is a different (smaller) number everywhere.
    double solar_radius_km = 0.0;
    int solar_disk_samples = 128;   // escalation count when the 5 probes split

    // --- the engine's own direct-solar field, for the consistency gate ----
    // FLAT with stride total_time_steps: [f*total_time_steps + t]. Optional:
    // null disables the gate, and the manifest SAYS the gate did not run rather
    // than implying it passed.
    const std::vector<double>* direct_solar = nullptr;

    // --- the search ladder ------------------------------------------------
    // MARCH then BISECT, not pure bisection. Bisection assumes the
    // blocked-below / clear-above transition is single-valued; an overhang
    // breaks that, and a bisection over a non-monotone predicate converges
    // confidently on the wrong crossing. The march finds the FIRST clear step
    // from the surface upward -- which is the contiguous umbral column a person
    // standing there is actually inside -- and only then is bisection safe,
    // because it runs inside a bracket the march proved.
    double h_max_m = 5.0;      // crew mobility and thermal constraints (Paul)
    int march_steps = 16;      // h_max/16 = 0.3125 m rungs at the default
    int bisect_rounds = 4;     // final bracket h_max/256 = 0.0195 m
    // The h = 0 rung is taken this far above the centroid along the VERTICAL. A
    // ray leaving the centroid exactly is decided by the facet's own plane;
    // this is the same 1 cm scale the engine uses to escape a source facet
    // (calculateSafeRayOffset), applied along the product's own vertical so it
    // never enters the height coordinate for h above it.
    double surface_offset_m = 0.01;

    // --- injected ray casting --------------------------------------------
    // BATCHED, deliberately. The single-ray probeOccluded the flux-cube pass
    // uses routes through intersectsFirst one ray at a time; that is fine for
    // its ~1e6 probes and is not fine for the ~1e8 this pass casts. REQUIRED;
    // a null here is a hard error, never "assume unshadowed".
    std::function<void(const std::vector<std::array<double, 3>>& origins,
                       const std::vector<std::array<double, 3>>& directions,
                       std::vector<uint8_t>& blocked)> occluded_batch;
    // Did the ray scene have its origin offset applied? See the tnear note in
    // the file header. False is REFUSED.
    bool scene_has_offset = false;
    // The tnear the batched path will ACTUALLY use, for a representative origin
    // in this scene, in mesh units. Printed and gated, never assumed.
    double resolved_tnear_m = -1.0;

    // --- output ------------------------------------------------------------
    std::string output_dir;         // run directory
    bool write_frames = true;       // shadow_depth/shadow_depth_<index>.bin (>= 6 digits)
    bool write_raw_stats = false;   // statistics/_raw side files for chunk merge
    std::string provenance_json;    // engine sha / md5 / config, echoed verbatim
    // Monotonicity audit: how many faces and epochs to re-scan above the first
    // clear rung. Sampled, because a full audit doubles the ray budget; the
    // count of failures is REPORTED, never zeroed.
    int monotonicity_faces = 10000;
    int monotonicity_epochs = 10;
};

// Returns true on success. On failure returns false and fills `err` with a
// reason the caller must surface as a hard error.
bool runShadowDepthPass(const PassInputs& in, std::string& err);

}  // namespace shadowdepth
}  // namespace ptm

#endif  // PTM_SHADOW_DEPTH_PASS_H
