// flux_cube_pass.h
// ---------------------------------------------------------------------------
// The engine-side flux-cube evaluation: a POST-SOLVE pass, run once after the
// radiation coupling loop has converged, immediately before the location
// export.
//
// WHY POST-SOLVE AND NOT IN THE COUPLING LOOP
// -------------------------------------------
//  1. The sweep is the hot loop. A per-iteration cube dot would run once per
//     coupling iteration for no benefit -- only the converged answer is a
//     product.
//  2. Inside the loop the band series are Picard-blended ((1-w)*old + w*new),
//     and B_ir carries an emission term that moves with T between iterations.
//     A blended B is not the B of any state.
//  3. The loop has TWO dispatch sites (single-timestep and multi-GPU pool).
//     Evaluating there means two insertion points that must agree forever --
//     the PTM-106c "two spellings of one formula" defect class. After the loop
//     both routes have written the same blended series, and that confluence is
//     what makes this pass route-blind.
//
// EVERYTHING IS INJECTED
// ----------------------
// The pass takes its ray casting, its primary-body physics, its solar constant
// and its incidence-albedo function as callbacks rather than reaching for the
// engine's globals. Two reasons: the module is unit-testable against synthetic
// geometry with no illumination model and no SPICE, and every function it uses
// is the ENGINE'S OWN, passed in by the caller, rather than a second spelling
// written here.
//
// It REFUSES rather than substituting. A missing cache, a cache key that does
// not match the run, a gray (non-band-separated) radiosity solve, an
// unreconstructible face, a missing ray tracer -- each is a hard error with a
// reason. A run that was asked for a product and produced none must fail, not
// succeed quietly.
// ---------------------------------------------------------------------------
#ifndef PTM_FLUX_CUBE_PASS_H
#define PTM_FLUX_CUBE_PASS_H

#include <array>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>
#include <cstddef>

namespace ptm {
namespace fluxcubes {

struct PassInputs {
    // --- cache + identity (the key; a mismatch is refused, never rebuilt) ---
    std::string cache_file;
    double geomhash_core = 0.0;     // sum_i (A_i + (i+1)*A_i*1e-9) over CORE faces
    size_t n_faces_core = 0;
    // FNV-1a 64 of the occluder PLY the RUN is using. The cache records the hash
    // of the ring it was built against, and the ring changes every sky fraction
    // in the file -- so a cache built against a different ring is as wrong as one
    // built against a different core, and until this was checked it loaded
    // silently. 0 => the caller could not compute one; then the check is skipped
    // and the pass SAYS SO rather than implying it verified something.
    uint64_t ring_hash = 0;

    // --- time axis -------------------------------------------------------
    // MUST be the same (times, time_strings) pair that was handed to
    // writeTimeStepsFile for this run. Re-deriving the grid from
    // start_time + i*dt is exactly the PTM-250/251 defect (the output epoch
    // grid divides the COMBINED equilibration+output span, and the stride
    // export path filters). The pass asserts its row count against these.
    const std::vector<double>* times = nullptr;
    const std::vector<std::string>* time_strings = nullptr;
    int equilibration_steps = 0;
    int total_time_steps = 0;

    // --- fields, indexed by ABSOLUTE step ---------------------------------
    // direct_solar is FLAT with stride total_time_steps: [f*total_time_steps + t].
    // Do not copy the neighboring location-export call's stride; it passes the
    // OUTPUT count against arrays strided by the total, which is a live defect.
    const std::vector<double>* direct_solar = nullptr;
    const std::vector<std::vector<double>>* surface_temps = nullptr;          // [f][t]
    const std::vector<std::vector<double>>* indirect_solar_series = nullptr;  // [f][t]
    const std::vector<std::vector<double>>* indirect_ir_series = nullptr;     // [f][t]

    // --- per-face static properties --------------------------------------
    std::vector<double> emissivity;    // operative emissivity the solver used
    std::vector<double> bond_albedo;   // diffuse OPERATOR reflectance (setSolarAlbedo)

    // --- ephemeris, indexed by ABSOLUTE step ------------------------------
    const std::vector<std::array<double, 3>>* sun_positions = nullptr;
    const std::vector<std::array<double, 3>>* primary_positions = nullptr;
    double primary_radius_km = 0.0;    // 0 => the primary is treated as a POINT
                                       // source and the header says so.
    int primary_disk_samples = 128;
    // ⛔ UNITS. sun_positions / primary_positions are in KILOMETERS, while the
    // mesh (and therefore every cube center) is in the mesh's own frame --
    // METERS for lunar/Europa DEMs, kilometers for asteroid shape models. The
    // engine handles this explicitly (illumination_model.cpp:5099, metersToKm
    // gated on centers_in_meters) and so must this pass: subtracting a meters
    // position from a kilometers one produces a direction pointing INTO the
    // ground, every disk sample reads as occluded, and the primary-body flux
    // collapses to exactly zero without any error. That is not hypothetical --
    // it is what this field was added to fix.
    double mesh_to_km = 1.0;           // multiply a mesh coordinate to get km

    // --- injected engine functions ---------------------------------------
    // occluded(origin, unit_direction) -> true when terrain blocks the ray.
    // REQUIRED; a null here is a hard error, never "assume unshadowed".
    std::function<bool(const std::array<double, 3>&, const std::array<double, 3>&)> occluded;
    // The engine's own solarConstForStep(t, S) with S already bound.
    std::function<double(int)> solar_const;
    // The engine's own applyIncidenceAlbedo(base_A, direct_flux, S) with the
    // PTM-106 coefficients already bound. Null => the source albedo falls back
    // to the Bond value, which is the legacy (pre-PTM-106) behavior and is
    // recorded in the header rather than assumed.
    std::function<double(double, double, double)> source_albedo;
    // The engine's own primary-body physics, taking mean_cos (not a normal).
    std::function<double(double, double)> primary_heating;
    std::function<double(double, const std::array<double, 3>&,
                         const std::array<double, 3>&,
                         const std::array<double, 3>&, double)> primary_reflected;

    // --- output ----------------------------------------------------------
    std::string output_dir;            // run directory; files land in flux_cubes/
    bool write_binary = true;
    bool write_csv = true;
    int  write_csv_max_cubes = 0;      // 0 = every cube
    int  shadow_probe_rays = 5;        // 1 = center only; 5 = center + 4 rim
    // FINITE SOLAR DISK. The Sun is not a point: its angular radius is ~0.26 deg
    // at 1 AU, so a shadow edge is smeared by dist*tan(alpha) -- 0.045 m at a
    // 10 m occluder, 0.45 m at 100 m. A binary point-sun test therefore returns
    // a hard 0 or 1 where the engine resolves a partial disk, and the error is
    // the FULL beam (up to ~1.2 kW/m^2) at the epochs where a cube straddles an
    // edge. 0 => point sun (the old behavior), recorded in the header.
    double solar_radius_km = 0.0;
    int    solar_disk_samples = 128;   // escalation count when the 5-probe splits
    std::string provenance_json;       // engine sha / md5 / config, echoed verbatim
    // Names the CONFIG asked for. The pass verifies the cache actually contains
    // every one and REFUSES otherwise -- editing the YAML without rebuilding the
    // cache would otherwise yield a product silently describing the previous set
    // of locations, which is the exact failure mode naming them was meant to
    // prevent. Empty => no check.
    std::vector<std::string> expected_location_names;
};

// Returns true on success. On failure returns false and fills `err` with a
// reason the caller must surface as a hard error.
bool runFluxCubePass(const PassInputs& in, std::string& err);

}  // namespace fluxcubes
}  // namespace ptm

#endif  // PTM_FLUX_CUBE_PASS_H
