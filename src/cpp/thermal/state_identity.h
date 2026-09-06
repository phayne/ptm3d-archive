// state_identity.h -- WHAT A THERMAL-STATE CHECKPOINT WAS INTEGRATED WITH.
//
// The PTMT v2 trailer (thermal/thermal_state_io.h) carries a 64-bit
// `config_identity_hash` so a restore can refuse a checkpoint that belongs to a
// different physics. This file is the ONE definition of what that hash covers.
//
// ---------------------------------------------------------------------------
// WHY THIS HASHES THE RESOLVED OBJECTS AND NOT THE CONFIG FILE
// ---------------------------------------------------------------------------
// On a built-in body (Moon / Europa / Mercury) the whole `thermophysical:` YAML
// block is DECORATIVE -- BUGS.md PTM-67. `ks`, `kd`, `H`, `rhos`, `rhod`,
// `cp0`, `Qb`, `chi`, `albedo` and `emissivity` under that block are read by
// nothing; only the `*_override` keys are live. Hashing the FILE would
// therefore do exactly the wrong thing twice over: two runs whose configs
// differ only in inert text would refuse each other, and two runs with
// genuinely different physics would compare equal.
//
// So the hash is taken over the RESOLVED `Planet` -- after
// applyConfiguredThermo() has folded in every live override -- plus the
// `Config` members that shape the depth grid and the conduction law. That is
// the same stance ptm_eq_certify already takes when it refuses to read those
// dead scalars (ptm_eq_certify.cpp:29-33).
//
// ---------------------------------------------------------------------------
// ONE NAMED SET, READ BY BOTH THE HASH AND THE REPORT
// ---------------------------------------------------------------------------
// identityFields() is the single list. hashOf() folds it and text() prints it;
// neither has its own idea of what a property is. A property added to the list
// is therefore automatically covered by the guard AND named in the mismatch
// report -- which is the failure mode PTM-53's `thermoOverrideRequested()`
// exists to prevent for the override forwarding, and this copies that stance
// deliberately.
//
// ---------------------------------------------------------------------------
// WHAT THE HASH DOES **NOT** COVER -- read before quoting it as "identical"
// ---------------------------------------------------------------------------
//   * PER-FACE MAPS. An albedo map, a thermal-inertia map, an H map or a
//     per-face Qb mask (thermal.heat_flow_mask_file) changes the physics face
//     by face and is NOT in this hash. Two runs that load DIFFERENT maps but
//     agree on every scalar below produce the SAME hash and will restore each
//     other's checkpoints. `refgrid_H` (the median H a loaded map publishes) is
//     hashed and does move with the map, so a wholly different map is usually
//     -- not always -- caught by that one number.
//   * THE MESH. Covered separately by `mesh_geomhash` in the trailer.
//   * VIEW FACTORS / radiation coupling configuration.
//   * The time window, the output cadence, file paths, and anything else that
//     legitimately differs between two chunks of ONE run. Those must NOT be in
//     the hash or every chunk hand-off would refuse.
//
// Anything in that list is a real gap in the guard, and callers print this
// boundary rather than implying the hash is complete.

#ifndef PTM_THERMAL_STATE_IDENTITY_H
#define PTM_THERMAL_STATE_IDENTITY_H

#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#include "thermal_model.h"

namespace thermal {
namespace state_id {

// %.17g -- a round-trippable rendering of a double, so the hash is a pure
// function of the VALUE and not of a print precision someone may widen later.
inline std::string num(double v) {
    char buf[40];
    std::snprintf(buf, sizeof(buf), "%.17g", v);
    return buf;
}

inline std::string num(int v) {
    char buf[24];
    std::snprintf(buf, sizeof(buf), "%d", v);
    return buf;
}

inline std::string vec(const std::vector<double>& v) {
    std::string s = "[";
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (i) s += ",";
        s += num(v[i]);
    }
    return s + "]";
}

// THE NAMED SET. Add a property here and it is hashed AND reported, both.
// `refgrid_H` is passed in rather than read off the Planet because it is
// published by main() into the Config property bag (refGridHOverride) and is
// what every solver path actually builds its shared grid on.
inline std::vector<std::pair<std::string, std::string>>
identityFields(const Planet& planet, const Config& config, double refgrid_H) {
    std::vector<std::pair<std::string, std::string>> f;
    f.reserve(32);
    // --- body identity and the resolved thermophysical scalars -------------
    f.emplace_back("body",              planet.name);
    f.emplace_back("albedo",            num(planet.albedo));
    f.emplace_back("emissivity",        num(planet.emissivity));
    f.emplace_back("day_s",             num(planet.day));
    f.emplace_back("year_s",            num(planet.year));
    f.emplace_back("rAU",               num(planet.rAU));
    f.emplace_back("S",                 num(planet.S));
    f.emplace_back("Qb",                num(planet.Qb));
    f.emplace_back("ks",                num(planet.ks));
    f.emplace_back("kd",                num(planet.kd));
    f.emplace_back("rhos",              num(planet.rhos));
    f.emplace_back("rhod",              num(planet.rhod));
    f.emplace_back("H",                 num(planet.H));
    f.emplace_back("cp0",               num(planet.cp0));
    f.emplace_back("cpCoeff",           vec(planet.cpCoeff));
    f.emplace_back("albedoCoef",        vec(planet.albedoCoef));
    f.emplace_back("body_radius_km",    num(planet.body_radius_km));
    f.emplace_back("use_hapke_albedo",  num(planet.use_hapke_albedo ? 1 : 0));
    f.emplace_back("hapke_albedo_w",    num(planet.hapke_albedo_w));
    // --- the depth grid and the conduction law -----------------------------
    f.emplace_back("chi",               num(config.chi));
    f.emplace_back("sigma",             num(config.sigma));
    f.emplace_back("S0",                num(config.S0));
    f.emplace_back("grid_m",            num(config.m));
    f.emplace_back("grid_n",            num(config.n));
    f.emplace_back("grid_b",            num(config.b));
    f.emplace_back("adaptive_grid",     num(config.ADAPTIVE_GRID ? 1 : 0));
    f.emplace_back("R_MAX",             num(config.R_MAX));
    f.emplace_back("min_layers_deep_per_skin",
                                        num(config.MIN_LAYERS_DEEP_PER_SKIN));
    f.emplace_back("min_total_layers_for_deep_coarsening",
                                        num(config.MIN_TOTAL_LAYERS_FOR_DEEP_COARSENING));
    f.emplace_back("res_decay_skins",   num(config.RES_DECAY_SKINS));
    f.emplace_back("conductivity_model",
                   num(static_cast<int>(config.conductivity_model)));
    f.emplace_back("refgrid_H",         num(refgrid_H));
    return f;
}

// Canonical rendering: one `key=value` per line, in list order. This is what
// the hash is taken over, and it is also what a log prints so a mismatch can be
// diffed by eye instead of by re-running both sides.
inline std::string text(const Planet& planet, const Config& config,
                        double refgrid_H) {
    std::string s;
    for (const auto& kv : identityFields(planet, config, refgrid_H)) {
        s += kv.first;
        s += '=';
        s += kv.second;
        s += '\n';
    }
    return s;
}

// FNV-1a, 64-bit, over the canonical rendering. Deterministic across builds and
// machines: no pointers, no hash_combine on a std::hash whose seed the standard
// library is free to change.
inline std::uint64_t hashOf(const std::string& canonical) {
    std::uint64_t h = 1469598103934665603ull;
    for (unsigned char c : canonical) {
        h ^= static_cast<std::uint64_t>(c);
        h *= 1099511628211ull;
    }
    // 0 is the trailer's "not recorded" sentinel, so it must never be a real
    // hash value. The odds are 2^-64 and the fix is one bit.
    return h ? h : 1ull;
}

inline std::uint64_t hashOf(const Planet& planet, const Config& config,
                            double refgrid_H) {
    return hashOf(text(planet, config, refgrid_H));
}

// The mesh-identity number. Same formula the view-factor pipeline already uses
// for its geomhash gate (radiation/vf_dump_sparse.cpp:302-310 and
// fluxcubes/flux_cube_geom.cpp:379-396), so a checkpoint and a VF matrix
// disagree about the mesh in the SAME arithmetic rather than in two.
//
// Rigid rotation (latitude_override) preserves every face area, so a rotated
// mesh hashes identically -- which is correct: the columns are the same
// columns. A rescaled or re-triangulated mesh does not.
inline double meshGeomHash(const std::vector<double>& face_areas) {
    double area_sum = 0.0, weighted = 0.0;
    for (std::size_t i = 0; i < face_areas.size(); ++i) {
        area_sum += face_areas[i];
        weighted += face_areas[i] * static_cast<double>(i + 1);
    }
    return area_sum + weighted * 1e-9;
}

}  // namespace state_id
}  // namespace thermal

#endif  // PTM_THERMAL_STATE_IDENTITY_H
