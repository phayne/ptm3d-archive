#include "config_parser.h"
#include "../thermal/conductivity_model.h"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>

#ifdef HAVE_YAML_CPP
#include <yaml-cpp/yaml.h>
#endif

// std::sort, for the PTM-67 unread-key report (deterministic key order).
#include <algorithm>

namespace ptm {

#ifdef HAVE_YAML_CPP

// Helper template for safe YAML node access with defaults
template<typename T>
T getWithDefault(const YAML::Node& node, const std::string& key, const T& defaultVal) {
    if (node && node[key]) return node[key].as<T>();
    return defaultVal;
}

namespace {
// ---------------- Data-path resolution (PTM_DATA_ROOT contract) ----------------
// One contract for every data-path config key (see docs/ENVIRONMENT.md and
// data/README.md):
//   1. An ABSOLUTE path in the config always wins (back-compat with campaign
//      YAMLs).
//   2. If $PTM_DATA_ROOT is set, a relative "data/..." reference resolves
//      under it (PTM_DATA_ROOT points AT the data directory).
//   3. Otherwise the repository's data/ directory; the repository root is
//      found by walking up from the YAML file's directory (then from the
//      current working directory) to the directory containing
//      data/data_manifest.json.
//   4. If a REQUIRED input still cannot be found, stop with a clear error
//      that names the dataset and the exact fetch command, rather than
//      failing later with a bare file-open error.

// Map a "data/..." path prefix to the dataset name in data/data_manifest.json.
std::string datasetForPath(const std::string& p) {
    static const struct { const char* prefix; const char* dataset; } kMap[] = {
        {"data/topo/",            "moon-topo"},
        {"data/property_maps/",   "moon-property-maps"},
        {"data/albedo_maps/",     "albedo-maps"},
        {"data/dem/validation/",  "validation-meshes"},
        {"data/dem/copernicus/",  "copernicus-dem"},
        {"data/meshes/moon3d",    "moon3d-mesh"},
    };
    for (const auto& m : kMap)
        if (p.rfind(m.prefix, 0) == 0) return m.dataset;
    return "";
}

std::filesystem::path findRepoRootForData(const std::string& yaml_file) {
    std::error_code ec;
    auto probe = [&ec](std::filesystem::path dir) -> std::filesystem::path {
        for (int i = 0; i < 10 && !dir.empty(); ++i) {
            if (std::filesystem::exists(dir / "data" / "data_manifest.json", ec))
                return dir;
            auto parent = dir.parent_path();
            if (parent == dir) break;
            dir = parent;
        }
        return {};
    };
    auto abs = std::filesystem::absolute(yaml_file, ec);
    if (!ec) {
        auto r = probe(abs.parent_path());
        if (!r.empty()) return r;
    }
    auto cwd = std::filesystem::current_path(ec);
    if (!ec) return probe(cwd);
    return {};
}

// Resolve one path-valued config entry. `must_exist` = the key is a required
// input in this configuration (its feature is enabled and the value non-empty).
std::string resolveDataPath(const std::string& value,
                            const std::filesystem::path& repo_root,
                            bool must_exist,
                            const char* key_name) {
    if (value.empty()) return value;
    std::filesystem::path p(value);
    if (p.is_absolute()) return value;                      // rule 1
    std::error_code ec;
    const bool is_data_rel = (value.rfind("data/", 0) == 0);
    const char* env_root = std::getenv("PTM_DATA_ROOT");
    std::filesystem::path resolved;
    if (is_data_rel && env_root && *env_root) {             // rule 2
        resolved = std::filesystem::path(env_root) / value.substr(5);
    } else if (std::filesystem::exists(p, ec)) {
        return value;                                       // resolves from cwd as before
    } else if (is_data_rel && !repo_root.empty()) {         // rule 3
        resolved = repo_root / value;
    } else {
        resolved = p;
    }
    if (std::filesystem::exists(resolved, ec))
        return resolved.lexically_normal().string();
    if (must_exist) {                                       // rule 4
        const std::string ds = datasetForPath(value);
        std::cerr << "ERROR: " << key_name << " = '" << value << "' not found";
        if (is_data_rel && env_root && *env_root)
            std::cerr << " under PTM_DATA_ROOT=" << env_root;
        else if (!repo_root.empty())
            std::cerr << " under " << (repo_root / "data").string()
                      << " (and PTM_DATA_ROOT is not set)";
        else
            std::cerr << " (PTM_DATA_ROOT not set; no repository data/ directory located)";
        std::cerr << ".\n";
        if (!ds.empty())
            std::cerr << "  Fetch it with: python3 scripts/fetch_data.py --dataset "
                      << ds << "\n";
        else if (value.rfind("data/kernels", 0) == 0)
            std::cerr << "  Fetch it with: python3 scripts/fetch_kernels.py\n";
        else if (value.rfind("data/external/", 0) == 0)
            std::cerr << "  This path is an external-dataset placeholder (not distributed);"
                         " see data/external/README.md.\n";
        std::cerr << "  Data location contract: data/README.md, docs/ENVIRONMENT.md"
                  << std::endl;
        std::exit(1);
    }
    return resolved.lexically_normal().string();
}

// ---------------------------------------------------------------------------
// PTM-67: report, ONCE and in ONE line, every key under `thermophysical:` that
// this parser does not read.
//
// Most of the block is decorative. `ks`, `kd`, `H`, `rhos`, `rhod`, `cp0`,
// `Qb`, `chi`, `albedo` and `emissivity` are read only under
// `generic_body.surface` -- under `thermophysical:` they are dropped here, at
// the parser, with nothing printed. Every `use_default_<key>` flag except
// `use_default_albedo` is likewise unread, which is worse than merely inert:
// the flag implies an implemented "false" branch that does not exist.
// Measured 2026-08-04: 19 distinct dead keys across the 132 shipped configs;
// `use_default_rhos: false` + `rhos: 900.0` ran at rhos = 1100 (the Moon
// default) and the run's own banner said so.
//
// This WARNS; it does not fail, and it does NOT alias the dead keys to the
// live ones. Making them live would change results for every config in the
// repository and needs its own gate. Naming them costs nothing and ends the
// silence, which is the actual defect.
void warnUnreadThermophysicalKeys(const YAML::Node& thermo) {
    if (!thermo || !thermo.IsMap()) return;

    // The keys loadFromYAML actually consumes from root["thermophysical"].
    // INVARIANT: every `getWithDefault(thermo, "K", ...)` in the
    // `root["thermophysical"]` block must appear here. There is no way to
    // derive this list -- the reads are individual statements -- so it is
    // maintained by hand and this comment is the contract.
    //
    // NOTE `conductivity_model` is listed although THIS branch does not yet
    // read it: `feat/martinez-conductivity` adds its reader to the same block,
    // and the released binary is the merged tree. On a hygiene-only build the
    // key is therefore accepted without comment -- the one deliberate
    // false-negative in this list, recorded here rather than left to be
    // rediscovered.
    static const char* const kRead[] = {
        "albedo_hapke_w",
        "albedo_model",
        "albedo_override",
        "conductivity_model",
        "constant_thermal_inertia",
        "emissivity_override",
        "incidence_angle_albedo",
        "kd_override",
        "ks_override",
        "use_default_albedo",
    };

    std::vector<std::string> dead;
    for (const auto& kv : thermo) {
        std::string key;
        try { key = kv.first.as<std::string>(); } catch (...) { continue; }
        bool is_read = false;
        for (const char* r : kRead) { if (key == r) { is_read = true; break; } }
        if (!is_read) dead.push_back(key);
    }
    if (dead.empty()) return;

    std::sort(dead.begin(), dead.end());   // deterministic output
    std::ostringstream line;
    line << "[config] WARNING: " << dead.size()
         << " key(s) under `thermophysical:` are read by nothing and have NO effect: ";
    for (size_t i = 0; i < dead.size(); ++i)
        line << (i ? ", " : "") << dead[i];
    line << " -- editing them changes no result. Named properties are honored only under "
            "`generic_body.surface:`; on a built-in body use the `*_override` keys. "
            "See docs/KNOWN_LIMITATIONS.md (PTM-67).";
    std::cerr << line.str() << std::endl;
}
} // namespace

// ---------------------------------------------------------------------------
// PTM-42: apply the config's `engine_env:` block to the process environment.
//
// Static-initialization audit (2026-08-04, before writing this): every
// std::getenv() call under src/cpp/ (105 of them) sits inside a function body.
// The only cached read is a function-local static,
// gpu/thermal_opencl.cpp:114 `static const bool log_sub = ...`, which is
// latched on FIRST CALL inside the GPU dispatch loop, not at load time. There
// is no namespace-scope object whose constructor reads the environment, so no
// variable is consumed before main(). Applying here -- ahead of loadFromYAML,
// which itself reads PTM_DATA_ROOT -- is therefore early enough for all of
// them. If that audit ever stops holding, the correct response is to move the
// call earlier, not to widen a tolerance.
// ---------------------------------------------------------------------------
int ConfigParser::applyEngineEnv(const std::string& filename) {
    YAML::Node root;
    try {
        root = YAML::LoadFile(filename);
    } catch (const std::exception& e) {
        // Do not fail here: loadFromYAML reports the parse error a moment
        // later with its own message. But do not stay silent either -- a
        // config we cannot read is a config whose engine_env we did not apply.
        std::cerr << "[engine_env] WARNING: cannot read '" << filename
                  << "' (" << e.what() << "); NO engine_env variables applied."
                  << std::endl;
        return 0;
    }

    if (!root["engine_env"]) return 0;   // no block: nothing declared, nothing to say
    const YAML::Node& block = root["engine_env"];
    if (!block.IsMap()) {
        std::cerr << "[engine_env] WARNING: 'engine_env' in " << filename
                  << " is not a mapping; ignoring it. Expected 'KEY: value' pairs."
                  << std::endl;
        return 0;
    }

    std::cout << "[engine_env] " << block.size() << " variable(s) declared in "
              << filename << std::endl;

    int applied = 0, conflicts = 0;
    for (auto it = block.begin(); it != block.end(); ++it) {
        std::string key;
        std::string val;
        try {
            key = it->first.as<std::string>();
            val = it->second.as<std::string>();
        } catch (const std::exception&) {
            std::cerr << "[engine_env]   WARNING: skipping a non-scalar entry in "
                         "engine_env (keys and values must be plain scalars)."
                      << std::endl;
            continue;
        }
        if (key.empty()) {
            std::cerr << "[engine_env]   WARNING: skipping an empty key." << std::endl;
            continue;
        }

        const char* prior = std::getenv(key.c_str());
        if (prior != nullptr && val != prior) {
            // The process environment wins. Say so loudly: this is exactly the
            // situation where a run silently gets an environment its config
            // does not describe.
            std::cerr << "[engine_env]   CONFLICT: " << key << " = '" << prior
                      << "' in the process environment overrides the config's '"
                      << val << "'. The PROCESS value is used." << std::endl;
            ++conflicts;
            continue;
        }
        if (prior != nullptr) {
            std::cout << "[engine_env]   " << key << "=" << val
                      << " (already set in the environment, identical)" << std::endl;
            continue;
        }
        if (::setenv(key.c_str(), val.c_str(), /*overwrite=*/1) != 0) {
            std::cerr << "[engine_env]   WARNING: setenv(" << key
                      << ") failed; this variable is NOT active." << std::endl;
            continue;
        }
        std::cout << "[engine_env]   " << key << "=" << val << " (applied)" << std::endl;
        ++applied;
    }
    std::cout << "[engine_env] applied " << applied << ", conflicts " << conflicts
              << std::endl;
    return applied;
}

SimulationConfig ConfigParser::loadFromYAML(const std::string& filename) {
    SimulationConfig config;
    try {
        YAML::Node root = YAML::LoadFile(filename);

        // Parse Time section
        if (root["simulation"]) {
            const auto& sim = root["simulation"];
            config.time.start_time = getWithDefault(sim, "start_time", config.time.start_time);
            config.time.duration_days = getWithDefault(sim, "duration_days", config.time.duration_days);
            config.time.timesteps = getWithDefault(sim, "timesteps", config.time.timesteps);
            config.time.body = getWithDefault(sim, "body", config.time.body);
        }
        
        // Parse Mesh section
        if (root["mesh"]) {
            const auto& mesh = root["mesh"];
            config.mesh.resolution = getWithDefault(mesh, "resolution", config.mesh.resolution);
            config.mesh.import_file = getWithDefault(mesh, "import_file", config.mesh.import_file);
            config.mesh.occluder_file = getWithDefault(mesh, "occluder_file", config.mesh.occluder_file);
            config.mesh.scale_factor = getWithDefault(mesh, "scale_factor", config.mesh.scale_factor);
            config.mesh.export_basic = getWithDefault(mesh, "export_basic", config.mesh.export_basic);
            config.mesh.use_topographic = getWithDefault(mesh, "use_topographic", config.mesh.use_topographic);
            config.mesh.topo_directory = getWithDefault(mesh, "topo_directory", config.mesh.topo_directory);
            config.mesh.latitude_override = getWithDefault(mesh, "latitude_override", config.mesh.latitude_override);
            config.mesh.longitude_override = getWithDefault(mesh, "longitude_override", config.mesh.longitude_override);
        }
        
        // Parse Thermal section
        if (root["thermal"]) {
            const auto& thermal = root["thermal"];
            config.thermal.run_thermal_mesh = getWithDefault(thermal, "run_thermal_mesh", config.thermal.run_thermal_mesh);
            config.thermal.threads = getWithDefault(thermal, "threads", config.thermal.threads);
            // Depth-grid resolution. m = layers per upper skin depth; n = layer
            // growth factor; b = skin depths to bottom layer. These were
            // previously read only from --thermal-{m,n,b} CLI flags, with the
            // YAML keys silently ignored — fixed here.
            config.thermal.grid_m = getWithDefault(thermal, "thermal_m", config.thermal.grid_m);
            config.thermal.grid_n = getWithDefault(thermal, "thermal_n", config.thermal.grid_n);
            config.thermal.grid_b = getWithDefault(thermal, "thermal_b", config.thermal.grid_b);
            config.thermal.equilibration_orbits = getWithDefault(thermal, "equilibration_orbits", config.thermal.equilibration_orbits);
            config.thermal.equilibration_steps_per_day = getWithDefault(thermal, "equilibration_steps_per_day", config.thermal.equilibration_steps_per_day);
            config.thermal.equilibration_samples_per_diurnal_cycle = getWithDefault(thermal, "equilibration_samples_per_diurnal_cycle", config.thermal.equilibration_samples_per_diurnal_cycle);
            config.thermal.equilibration_cycles = getWithDefault(thermal, "equilibration_cycles", config.thermal.equilibration_cycles);
            config.thermal.equilibration_target_K = getWithDefault(thermal, "equilibration_target_K", config.thermal.equilibration_target_K);
            if (thermal["eq_extrapolation"]) {
                const auto& ex = thermal["eq_extrapolation"];
                config.thermal.eq_extrapolation.method =
                    getWithDefault(ex, "method",  config.thermal.eq_extrapolation.method);
                config.thermal.eq_extrapolation.order_p =
                    getWithDefault(ex, "order_p", config.thermal.eq_extrapolation.order_p);
                config.thermal.eq_extrapolation.tol_K =
                    getWithDefault(ex, "tol_K",   config.thermal.eq_extrapolation.tol_K);
                // safety and r_cap were retired as user knobs; their values are
                // now internal constants. Warn rather than ignore silently.
                for (const char* k : {"safety", "r_cap"}) {
                    if (ex[k]) {
                        std::cerr << "Warning: thermal.eq_extrapolation." << k
                                  << " is no longer a knob and is IGNORED; the value is now "
                                     "an internal constant (safety 3.0, r_cap 0.9995).\n";
                    }
                }
                config.thermal.eq_extrapolation.hold_K =
                    getWithDefault(ex, "hold_K",  config.thermal.eq_extrapolation.hold_K);
            }
            config.thermal.compute_statistics = getWithDefault(thermal, "compute_statistics", config.thermal.compute_statistics);
            config.thermal.use_gpu = getWithDefault(thermal, "use_gpu", config.thermal.use_gpu);
            config.thermal.gpu_devices = getWithDefault(thermal, "gpu_devices", config.thermal.gpu_devices);
            if (thermal["gpu_device_ids"] && thermal["gpu_device_ids"].IsSequence()) {
                config.thermal.gpu_device_ids.clear();
                for (const auto& v : thermal["gpu_device_ids"]) {
                    config.thermal.gpu_device_ids.push_back(v.as<int>());
                }
            }
            // gpu_batch_size is a size_t in config but YAML deserializes as int — accept both.
            if (thermal["gpu_batch_size"]) {
                config.thermal.gpu_batch_size =
                    static_cast<size_t>(thermal["gpu_batch_size"].as<long long>());
            }
            config.thermal.gpu_inter_batch_pause_ms = getWithDefault(
                thermal, "gpu_inter_batch_pause_ms", config.thermal.gpu_inter_batch_pause_ms);
            config.thermal.auto_calibrate_gpu = getWithDefault(
                thermal, "auto_calibrate_gpu", config.thermal.auto_calibrate_gpu);
            config.thermal.thermal_inertia_map = getWithDefault(thermal, "thermal_inertia_map", config.thermal.thermal_inertia_map);
            config.thermal.h_parameter_map = getWithDefault(thermal, "h_parameter_map", config.thermal.h_parameter_map);
            config.thermal.use_default_property_map = getWithDefault(thermal, "use_default_property_map", config.thermal.use_default_property_map);
            config.thermal.property_maps_directory = getWithDefault(thermal, "property_maps_directory", config.thermal.property_maps_directory);
            config.thermal.enable_shadow_flux = getWithDefault(thermal, "enable_shadow_flux", config.thermal.enable_shadow_flux);
            config.thermal.crater_dd_ratio = getWithDefault(thermal, "crater_dd_ratio", config.thermal.crater_dd_ratio);
            config.thermal.integration_days = getWithDefault(thermal, "integration_days", config.thermal.integration_days);
            config.thermal.integration_steps = getWithDefault(thermal, "integration_steps", config.thermal.integration_steps);
            config.thermal.integration_offset_days = getWithDefault(thermal, "integration_offset_days", config.thermal.integration_offset_days);
            config.thermal.integration_offset_steps = getWithDefault(thermal, "integration_offset_steps", config.thermal.integration_offset_steps);
            config.thermal.geothermal_heat_flow = getWithDefault(thermal, "geothermal_heat_flow", config.thermal.geothermal_heat_flow);
            config.thermal.heat_flow_mask_file = getWithDefault(thermal, "heat_flow_mask_file", config.thermal.heat_flow_mask_file);
            config.thermal.heat_flow_mask_value = getWithDefault(thermal, "heat_flow_mask_value", config.thermal.heat_flow_mask_value);
            config.thermal.initial_state_file = getWithDefault(thermal, "initial_state_file", config.thermal.initial_state_file);
            config.thermal.final_state_file   = getWithDefault(thermal, "final_state_file",   config.thermal.final_state_file);
            config.thermal.checkpoint_every_output_steps = getWithDefault(thermal, "checkpoint_every_output_steps", config.thermal.checkpoint_every_output_steps);
            config.thermal.indirect_mean_out_file = getWithDefault(thermal, "indirect_mean_out_file", config.thermal.indirect_mean_out_file);
            config.thermal.eq_indirect_flux_file  = getWithDefault(thermal, "eq_indirect_flux_file",  config.thermal.eq_indirect_flux_file);
            config.thermal.compute_ice_stability = getWithDefault(thermal, "compute_ice_stability", config.thermal.compute_ice_stability);
            if (thermal["ice_stability"]) {
                const auto& is_node = thermal["ice_stability"];
                config.thermal.ice_stability.flux_threshold_kgm2s =
                    getWithDefault(is_node, "flux_threshold_kgm2s", config.thermal.ice_stability.flux_threshold_kgm2s);
                config.thermal.ice_stability.grain_hop_length_um =
                    getWithDefault(is_node, "grain_hop_length_um", config.thermal.ice_stability.grain_hop_length_um);
                config.thermal.ice_stability.geothermal_gradient_K_per_m =
                    getWithDefault(is_node, "geothermal_gradient_K_per_m", config.thermal.ice_stability.geothermal_gradient_K_per_m);
                config.thermal.ice_stability.sublimation_enthalpy_kJmol =
                    getWithDefault(is_node, "sublimation_enthalpy_kJmol", config.thermal.ice_stability.sublimation_enthalpy_kJmol);
                config.thermal.ice_stability.n_depth_subsamples =
                    getWithDefault(is_node, "n_depth_subsamples", config.thermal.ice_stability.n_depth_subsamples);
            }
        }
        
        // Parse Illumination section
        if (root["illumination"]) {
            const auto& n = root["illumination"];
            config.illumination.include_primary_reflection =
                getWithDefault(n, "include_primary_reflection", config.illumination.include_primary_reflection);
            config.illumination.use_batch_processing =
                getWithDefault(n, "use_batch_processing", config.illumination.use_batch_processing);
            config.illumination.use_albedo_map =
                getWithDefault(n, "use_albedo_map", config.illumination.use_albedo_map);
            config.illumination.albedo_map =
                getWithDefault(n, "albedo_map", config.illumination.albedo_map);
            config.illumination.albedo_maps_directory =
                getWithDefault(n, "albedo_maps_directory", config.illumination.albedo_maps_directory);
            config.illumination.positions_file =
                getWithDefault(n, "positions_file", config.illumination.positions_file);
            config.illumination.use_gpu = getWithDefault(n, "use_gpu",
                config.illumination.use_gpu);
            config.illumination.gpu_devices = getWithDefault(n, "gpu_devices",
                config.illumination.gpu_devices);
            if (n["gpu_device_ids"] && n["gpu_device_ids"].IsSequence()) {
                config.illumination.gpu_device_ids.clear();
                for (const auto& v : n["gpu_device_ids"]) {
                    config.illumination.gpu_device_ids.push_back(v.as<int>());
                }
            }
            config.illumination.gpu_inter_batch_pause_ms = getWithDefault(n,
                "gpu_inter_batch_pause_ms", config.illumination.gpu_inter_batch_pause_ms);
            config.illumination.bvh_cache_file = getWithDefault(n,
                "bvh_cache_file", config.illumination.bvh_cache_file);
        }
        
        // Parse Output section
        if (root["output"]) {
            const auto& output = root["output"];
            config.output.directory = getWithDefault(output, "directory", config.output.directory);
            config.output.format = getWithDefault(output, "format", config.output.format);
            config.output.vtk_mode = getWithDefault(output, "vtk_mode", config.output.vtk_mode);
            config.output.include_primary_ir = getWithDefault(output, "include_primary_ir", config.output.include_primary_ir);
            config.output.thermal_vtk_file = getWithDefault(output, "thermal_vtk_file", config.output.thermal_vtk_file);
            config.output.complete_vtk_file = getWithDefault(output, "complete_vtk_file", config.output.complete_vtk_file);
            config.output.locations_file = getWithDefault(output, "locations_file", config.output.locations_file);
            config.output.series_output = getWithDefault(output, "series_output", config.output.series_output);
            config.output.location_data = getWithDefault(output, "location_data", config.output.location_data);
            config.output.depths_file = getWithDefault(output, "depths_file", config.output.depths_file);
            config.output.skip_time_series = getWithDefault(output, "skip_time_series", config.output.skip_time_series);
            config.output.write_raw_stats = getWithDefault(output, "write_raw_stats", config.output.write_raw_stats);
            config.output.skip_mesh = getWithDefault(output, "skip_mesh", config.output.skip_mesh);
        }
        
        // Parse Radiation section
        if (root["radiation"]) {
            const auto& rad = root["radiation"];
            config.radiation.enabled = getWithDefault(rad, "enabled", config.radiation.enabled);
            config.radiation.exchange_selection = getWithDefault(rad, "exchange_selection", config.radiation.exchange_selection);
            config.radiation.min_shadow_fraction = getWithDefault(rad, "min_shadow_fraction", config.radiation.min_shadow_fraction);
            config.radiation.verbose = getWithDefault(rad, "verbose", config.radiation.verbose);
            config.radiation.use_gpu = getWithDefault(rad, "use_gpu", config.radiation.use_gpu);
            config.radiation.view_factor_file = getWithDefault(rad, "view_factor_file", config.radiation.view_factor_file);
            config.radiation.max_coupling_iterations = getWithDefault(rad, "max_coupling_iterations", config.radiation.max_coupling_iterations);
            config.radiation.coupling_temp_threshold = getWithDefault(rad, "coupling_temp_threshold", config.radiation.coupling_temp_threshold);
            config.radiation.coupling_flux_threshold = getWithDefault(rad, "coupling_flux_threshold", config.radiation.coupling_flux_threshold);
            config.radiation.coupling_face_fraction = getWithDefault(rad, "coupling_face_fraction", config.radiation.coupling_face_fraction);
            config.radiation.coupling_select_mode = getWithDefault(rad, "coupling_select_mode", config.radiation.coupling_select_mode);
            config.radiation.coupling_select_safety = getWithDefault(rad, "coupling_select_safety", config.radiation.coupling_select_safety);
            config.radiation.coupling_mode = getWithDefault(rad, "coupling_mode", config.radiation.coupling_mode);
            config.radiation.coupling_select_dt = getWithDefault(rad, "coupling_select_dt", config.radiation.coupling_select_dt);
            config.radiation.use_hmatrix = getWithDefault(rad, "use_hmatrix", config.radiation.use_hmatrix);
            config.radiation.use_aca = getWithDefault(rad, "use_aca", config.radiation.use_aca);
            config.radiation.aca_tolerance = getWithDefault(rad, "aca_tolerance", config.radiation.aca_tolerance);
            config.radiation.aca_max_rank = getWithDefault(rad, "aca_max_rank", config.radiation.aca_max_rank);
            config.radiation.hmatrix_threads = getWithDefault(rad, "hmatrix_threads", config.radiation.hmatrix_threads);
            // Adaptive neighbor selection
            config.radiation.adaptive_neighbors = getWithDefault(rad, "adaptive_neighbors", config.radiation.adaptive_neighbors);
            config.radiation.captured_energy_target = getWithDefault(rad, "captured_energy_target", config.radiation.captured_energy_target);
            config.radiation.max_neighbors = getWithDefault(rad, "max_neighbors", config.radiation.max_neighbors);
            config.radiation.min_neighbors = getWithDefault(rad, "min_neighbors", config.radiation.min_neighbors);
            config.radiation.use_cluster_cull = getWithDefault(rad, "use_cluster_cull", config.radiation.use_cluster_cull);
            config.radiation.use_monte_carlo_vf = getWithDefault(rad, "use_monte_carlo_vf", config.radiation.use_monte_carlo_vf);
            config.radiation.mc_rays_per_face = getWithDefault(rad, "mc_rays_per_face", config.radiation.mc_rays_per_face);
            // Anisotropic roughness-beamed emission (Smith 1967); 0 = Lambertian.
            config.radiation.rms_slope_deg = getWithDefault(rad, "rms_slope_deg", config.radiation.rms_slope_deg);
            config.radiation.beam_vf_threshold = getWithDefault(rad, "beam_vf_threshold", config.radiation.beam_vf_threshold);
            // Radiance-weighted VF pruning epsilon [W/m^2] (assembly-time; 0 = off).
            // Distinct from visibility_threshold. The runners propagate it to the
            // assembler via PTM_RADIANCE_PRUNE_EPS; the sim only logs it.
            config.radiation.radiance_prune_eps = getWithDefault(rad, "radiance_prune_eps", config.radiation.radiance_prune_eps);
            // Band-separated (visible + thermal-IR) radiosity (D1). Default ON
            // since the D2 flip (config_parser.h: band_separated = true); set
            // false to recover the gray solve. See two-band-radiosity-derivation.tex.
            config.radiation.band_separated = getWithDefault(rad, "band_separated", config.radiation.band_separated);

            // Multi-GPU radiosity dispatch (mirrors thermal pool YAML keys).
            config.radiation.gpu_devices = getWithDefault(rad, "gpu_devices", config.radiation.gpu_devices);
            if (rad["gpu_device_ids"] && rad["gpu_device_ids"].IsSequence()) {
                config.radiation.gpu_device_ids.clear();
                for (const auto& v : rad["gpu_device_ids"]) {
                    config.radiation.gpu_device_ids.push_back(v.as<int>());
                }
            }
            config.radiation.gpu_batch_size = getWithDefault(rad, "gpu_batch_size", config.radiation.gpu_batch_size);
            config.radiation.gpu_inter_batch_pause_ms = getWithDefault(rad, "gpu_inter_batch_pause_ms", config.radiation.gpu_inter_batch_pause_ms);

            // Resolve relative view_factor_file against the YAML file's directory so
            // the cache path is cwd-independent (GUI subprocess cwd is not necessarily
            // the repo root, which was silently breaking the VF cache).
            if (!config.radiation.view_factor_file.empty()) {
                std::filesystem::path vf_path(config.radiation.view_factor_file);
                if (vf_path.is_relative()) {
                    std::error_code ec;
                    std::filesystem::path yaml_dir =
                        std::filesystem::absolute(filename, ec).parent_path();
                    if (!ec) {
                        config.radiation.view_factor_file =
                            (yaml_dir / vf_path).lexically_normal().string();
                    }
                }
            }
        }

        // Parse Generic Body section
        if (root["generic_body"]) {
            const auto& gb = root["generic_body"];
            config.generic_body.name = getWithDefault(gb, "name", config.generic_body.name);
            config.generic_body.config_file = getWithDefault(gb, "config_file", config.generic_body.config_file);
            config.generic_body.ephemeris_source = getWithDefault(gb, "ephemeris_source", config.generic_body.ephemeris_source);
            config.generic_body.spice_metakernel = getWithDefault(gb, "spice_metakernel", config.generic_body.spice_metakernel);
            config.generic_body.naif_id = getWithDefault(gb, "naif_id", config.generic_body.naif_id);

            // Shape subsection
            if (gb["shape"]) {
                const auto& shape = gb["shape"];
                config.generic_body.shape_type = getWithDefault(shape, "type", config.generic_body.shape_type);
                config.generic_body.radius_km = getWithDefault(shape, "radius_km", config.generic_body.radius_km);
                config.generic_body.mesh_file = getWithDefault(shape, "mesh_file", config.generic_body.mesh_file);
                config.generic_body.mesh_scale = getWithDefault(shape, "mesh_scale", config.generic_body.mesh_scale);
                if (shape["semi_axes_km"] && shape["semi_axes_km"].IsSequence() && shape["semi_axes_km"].size() == 3) {
                    config.generic_body.semi_axis_a_km = shape["semi_axes_km"][0].as<double>();
                    config.generic_body.semi_axis_b_km = shape["semi_axes_km"][1].as<double>();
                    config.generic_body.semi_axis_c_km = shape["semi_axes_km"][2].as<double>();
                }
            }

            // Orbit subsection
            if (gb["orbit"]) {
                const auto& orbit = gb["orbit"];
                config.generic_body.semi_major_axis_au = getWithDefault(orbit, "semi_major_axis_au", config.generic_body.semi_major_axis_au);
                config.generic_body.eccentricity = getWithDefault(orbit, "eccentricity", config.generic_body.eccentricity);
                config.generic_body.inclination_deg = getWithDefault(orbit, "inclination_deg", config.generic_body.inclination_deg);
                config.generic_body.lon_ascending_node_deg = getWithDefault(orbit, "lon_ascending_node_deg", config.generic_body.lon_ascending_node_deg);
                config.generic_body.arg_perihelion_deg = getWithDefault(orbit, "arg_perihelion_deg", config.generic_body.arg_perihelion_deg);
                config.generic_body.mean_anomaly_deg = getWithDefault(orbit, "mean_anomaly_deg", config.generic_body.mean_anomaly_deg);
                config.generic_body.orbital_epoch = getWithDefault(orbit, "epoch", config.generic_body.orbital_epoch);
            }

            // Spin subsection
            if (gb["spin"]) {
                const auto& spin = gb["spin"];
                config.generic_body.spin_period_hours = getWithDefault(spin, "period_hours", config.generic_body.spin_period_hours);
                config.generic_body.pole_ra_deg = getWithDefault(spin, "pole_ra_deg", config.generic_body.pole_ra_deg);
                config.generic_body.pole_dec_deg = getWithDefault(spin, "pole_dec_deg", config.generic_body.pole_dec_deg);
                config.generic_body.prime_meridian_deg = getWithDefault(spin, "prime_meridian_deg", config.generic_body.prime_meridian_deg);
            }

            // Surface/thermophysical subsection
            if (gb["surface"]) {
                const auto& surf = gb["surface"];
                config.generic_body.albedo = getWithDefault(surf, "albedo", config.generic_body.albedo);
                config.generic_body.emissivity = getWithDefault(surf, "emissivity", config.generic_body.emissivity);
                config.generic_body.ks = getWithDefault(surf, "ks", config.generic_body.ks);
                config.generic_body.kd = getWithDefault(surf, "kd", config.generic_body.kd);
                config.generic_body.rhos = getWithDefault(surf, "rhos", config.generic_body.rhos);
                config.generic_body.rhod = getWithDefault(surf, "rhod", config.generic_body.rhod);
                config.generic_body.H_scale = getWithDefault(surf, "H", config.generic_body.H_scale);
                config.generic_body.cp0 = getWithDefault(surf, "cp0", config.generic_body.cp0);
                config.generic_body.Qb = getWithDefault(surf, "Qb", config.generic_body.Qb);
                config.generic_body.chi = getWithDefault(surf, "chi", config.generic_body.chi);
            }
        }

        // Parse Thermophysical section (GUI-written overrides)
        if (root["thermophysical"]) {
            const auto& thermo = root["thermophysical"];
            // PTM-67: say which of these keys are decorative BEFORE reading the
            // ones that are not. One line, whatever the count.
            warnUnreadThermophysicalKeys(thermo);
            bool use_default_albedo = getWithDefault(thermo, "use_default_albedo", true);
            config.thermophysical_albedo_explicit = !use_default_albedo;
            config.thermophysical.albedo_explicit = !use_default_albedo;
            config.thermophysical.constant_thermal_inertia = getWithDefault(
                thermo, "constant_thermal_inertia",
                config.thermophysical.constant_thermal_inertia);
            // Optional scalar property overrides (sensitivity sweeps). Distinct
            // keys (suffixed _override) so existing configs that carry ignored
            // GUI scalars are unaffected; sentinel <0 leaves the body default.
            config.thermophysical.albedo_override = getWithDefault(
                thermo, "albedo_override", config.thermophysical.albedo_override);
            config.thermophysical.emissivity_override = getWithDefault(
                thermo, "emissivity_override", config.thermophysical.emissivity_override);
            config.thermophysical.ks_override = getWithDefault(
                thermo, "ks_override", config.thermophysical.ks_override);
            config.thermophysical.kd_override = getWithDefault(
                thermo, "kd_override", config.thermophysical.kd_override);

            // Directional-hemispheric (Hapke) albedo gate. Two equivalent opt-in
            // spellings: the previously-dead `incidence_angle_albedo` bool, or
            // `albedo_model: hapke`. Default OFF -> the Hayne-polynomial/constant
            // path is untouched and baseline runs stay byte-identical.
            bool inc_alb = getWithDefault(thermo, "incidence_angle_albedo", false);
            std::string albedo_model =
                getWithDefault<std::string>(thermo, "albedo_model", std::string{});
            for (auto& c : albedo_model) c = std::tolower(c);
            config.thermophysical.albedo_hapke = (inc_alb || albedo_model == "hapke");
            config.thermophysical.albedo_hapke_w = getWithDefault(
                thermo, "albedo_hapke_w", config.thermophysical.albedo_hapke_w);

            // Regolith conductivity law. Default "hayne2017" = the historical
            // path (byte-identical). An unrecognized value must be FATAL.
            //
            // Deliberately NOT a `throw`, and the reason CHANGED during this
            // release -- read both halves before touching it.
            //
            // When this was written, ptm_thermal_sim.cpp caught every
            // loadFromYAML exception, printed "Warning: Failed to load config
            // file ... Continuing with default configuration", and ran the
            // WHOLE simulation on built-in defaults with exit code 0
            // (BUGS.md PTM-66). A throw here would have been downgraded to a
            // warning and the run would have proceeded quietly on Hayne.
            // PTM-66 is FIXED on fix/release-hygiene in this same release, so
            // that is no longer true: a throw would now be fatal.
            //
            // The two-step is kept anyway, deliberately: the parser names the
            // offending key and the valid values while it still has the file
            // in hand, and `main` issues one FATAL and one exit code. Folding
            // it into a throw would replace a specific diagnosis with a generic
            // "failed to load config file". If you change this to a throw,
            // check that the message survives.
            {
                std::string cm = getWithDefault<std::string>(
                    thermo, "conductivity_model",
                    config.thermophysical.conductivity_model);
                thermal::ConductivityModel probe;
                if (!thermal::parseConductivityModel(cm, probe)) {
                    std::cerr << "ERROR: unknown thermophysical.conductivity_model '"
                              << cm << "'. Valid values: hayne2017 (default) | "
                                 "martinez2021." << std::endl;
                }
                config.thermophysical.conductivity_model = cm;
            }
        }

        // Parse Advanced section
        if (root["advanced"]) {
            const auto& adv = root["advanced"];
            config.advanced.verbose = getWithDefault(adv, "verbose", config.advanced.verbose);
            config.advanced.default_visible_albedo = getWithDefault(adv, "default_visible_albedo", config.advanced.default_visible_albedo);
        }

        // ---- Apply the PTM_DATA_ROOT resolution contract to every data-path
        // key (rules + actionable errors: see the helpers above). Required
        // inputs are only enforced when their feature is actually in use, so
        // configs that merely carry a default directory keep working.
        {
            const auto repo_root = findRepoRootForData(filename);
            auto& C = config;
            C.mesh.import_file = resolveDataPath(
                C.mesh.import_file, repo_root, true, "mesh.import_file");
            C.mesh.occluder_file = resolveDataPath(
                C.mesh.occluder_file, repo_root, true, "mesh.occluder_file");
            C.mesh.topo_directory = resolveDataPath(
                C.mesh.topo_directory, repo_root, C.mesh.use_topographic,
                "mesh.topo_directory");
            C.thermal.thermal_inertia_map = resolveDataPath(
                C.thermal.thermal_inertia_map, repo_root, true,
                "thermal.thermal_inertia_map");
            C.thermal.h_parameter_map = resolveDataPath(
                C.thermal.h_parameter_map, repo_root, true,
                "thermal.h_parameter_map");
            C.thermal.heat_flow_mask_file = resolveDataPath(
                C.thermal.heat_flow_mask_file, repo_root, true,
                "thermal.heat_flow_mask_file");
            // Property/albedo map DIRECTORIES: resolve, but never hard-fail on
            // the directory alone — the runtime falls back to uniform defaults
            // when maps are absent (existing behavior), except the albedo dir
            // when the config explicitly turns the albedo map on.
            C.thermal.property_maps_directory = resolveDataPath(
                C.thermal.property_maps_directory, repo_root, false,
                "thermal.property_maps_directory");
            C.illumination.albedo_maps_directory = resolveDataPath(
                C.illumination.albedo_maps_directory, repo_root,
                C.illumination.use_albedo_map, "illumination.albedo_maps_directory");
            C.illumination.albedo_map = resolveDataPath(
                C.illumination.albedo_map, repo_root,
                C.illumination.use_albedo_map, "illumination.albedo_map");
            // Caches / outputs (created on demand): resolve only.
            C.illumination.bvh_cache_file = resolveDataPath(
                C.illumination.bvh_cache_file, repo_root, false,
                "illumination.bvh_cache_file");
            // view_factor_file already has config-dir-relative resolution above
            // (it is created on first run, so no existence requirement).
            C.output.locations_file = resolveDataPath(
                C.output.locations_file, repo_root, true, "output.locations_file");
            C.output.depths_file = resolveDataPath(
                C.output.depths_file, repo_root, true, "output.depths_file");
            // Generic-body shape mesh: resolve without an existence requirement
            // (the Horizons pipeline can populate it after config load).
            C.generic_body.mesh_file = resolveDataPath(
                C.generic_body.mesh_file, repo_root, false, "generic_body.mesh_file");
        }

        std::cout << "Successfully loaded configuration from: " << filename << std::endl;

    } catch (const YAML::Exception& e) {
        last_error_ = std::string("YAML parsing error: ") + e.what();
        throw std::runtime_error(last_error_);
    } catch (const std::exception& e) {
        last_error_ = std::string("Error loading config: ") + e.what();
        throw std::runtime_error(last_error_);
    }
    
    return config;
}

bool ConfigParser::saveToYAML(const std::string& filename, const SimulationConfig& config) {
    try {
        YAML::Emitter out;
        out << YAML::BeginMap;
        
        // Simulation section
        out << YAML::Key << "simulation" << YAML::Value << YAML::BeginMap;
        out << YAML::Key << "start_time" << YAML::Value << config.time.start_time;
        out << YAML::Key << "duration_days" << YAML::Value << config.time.duration_days;
        out << YAML::Key << "timesteps" << YAML::Value << config.time.timesteps;
        out << YAML::Key << "body" << YAML::Value << config.time.body;
        out << YAML::EndMap;
        
        // Mesh section
        out << YAML::Key << "mesh" << YAML::Value << YAML::BeginMap;
        out << YAML::Key << "resolution" << YAML::Value << config.mesh.resolution;
        if (!config.mesh.import_file.empty())
            out << YAML::Key << "import_file" << YAML::Value << config.mesh.import_file;
        if (!config.mesh.occluder_file.empty())
            out << YAML::Key << "occluder_file" << YAML::Value << config.mesh.occluder_file;
        out << YAML::Key << "scale_factor" << YAML::Value << config.mesh.scale_factor;
        out << YAML::Key << "export_basic" << YAML::Value << config.mesh.export_basic;
        out << YAML::Key << "use_topographic" << YAML::Value << config.mesh.use_topographic;
        if (!config.mesh.topo_directory.empty())
            out << YAML::Key << "topo_directory" << YAML::Value << config.mesh.topo_directory;
        if (config.mesh.latitude_override > -900.0)
            out << YAML::Key << "latitude_override" << YAML::Value << config.mesh.latitude_override;
        if (config.mesh.longitude_override > -900.0)
            out << YAML::Key << "longitude_override" << YAML::Value << config.mesh.longitude_override;
        out << YAML::EndMap;
        
        // Thermal section
        out << YAML::Key << "thermal" << YAML::Value << YAML::BeginMap;
        out << YAML::Key << "run_thermal_mesh" << YAML::Value << config.thermal.run_thermal_mesh;
        out << YAML::Key << "threads" << YAML::Value << config.thermal.threads;
        out << YAML::Key << "thermal_m" << YAML::Value << config.thermal.grid_m;
        out << YAML::Key << "thermal_n" << YAML::Value << config.thermal.grid_n;
        out << YAML::Key << "thermal_b" << YAML::Value << config.thermal.grid_b;
        out << YAML::Key << "equilibration_orbits" << YAML::Value << config.thermal.equilibration_orbits;
        out << YAML::Key << "equilibration_steps_per_day" << YAML::Value << config.thermal.equilibration_steps_per_day;
        out << YAML::Key << "equilibration_samples_per_diurnal_cycle" << YAML::Value << config.thermal.equilibration_samples_per_diurnal_cycle;
        // Only emit when explicitly set — an unset value means "solver legacy
        // default", and writing -1 into every template would invite confusion
        // with the real pass count (which is 2, not 1).
        if (config.thermal.equilibration_cycles > 0)
            out << YAML::Key << "equilibration_cycles" << YAML::Value << config.thermal.equilibration_cycles;
        out << YAML::Key << "equilibration_target_K" << YAML::Value << config.thermal.equilibration_target_K;
        // Same rule: only round-trip the extrapolation block when it is armed,
        // so an OFF run's re-serialized config stays free of the knob entirely.
        if (config.thermal.eq_extrapolation.method != "off") {
            out << YAML::Key << "eq_extrapolation" << YAML::Value << YAML::BeginMap;
            out << YAML::Key << "method"  << YAML::Value << config.thermal.eq_extrapolation.method;
            out << YAML::Key << "order_p" << YAML::Value << config.thermal.eq_extrapolation.order_p;
            out << YAML::Key << "tol_K"   << YAML::Value << config.thermal.eq_extrapolation.tol_K;
            out << YAML::Key << "hold_K"  << YAML::Value << config.thermal.eq_extrapolation.hold_K;
            out << YAML::EndMap;
        }
        out << YAML::Key << "compute_statistics" << YAML::Value << config.thermal.compute_statistics;
        // Multi-GPU knobs (only emit when not at defaults — keep templates clean).
        if (config.thermal.gpu_devices != 1)
            out << YAML::Key << "gpu_devices" << YAML::Value << config.thermal.gpu_devices;
        if (!config.thermal.gpu_device_ids.empty()) {
            out << YAML::Key << "gpu_device_ids" << YAML::Value << YAML::Flow << YAML::BeginSeq;
            for (int id : config.thermal.gpu_device_ids) out << id;
            out << YAML::EndSeq;
        }
        if (config.thermal.gpu_batch_size != 0)
            out << YAML::Key << "gpu_batch_size" << YAML::Value
                << static_cast<long long>(config.thermal.gpu_batch_size);
        if (config.thermal.gpu_inter_batch_pause_ms != 0)
            out << YAML::Key << "gpu_inter_batch_pause_ms" << YAML::Value
                << config.thermal.gpu_inter_batch_pause_ms;
        if (!config.thermal.auto_calibrate_gpu)   // default true → emit only when opted out
            out << YAML::Key << "auto_calibrate_gpu" << YAML::Value
                << config.thermal.auto_calibrate_gpu;
        if (!config.thermal.thermal_inertia_map.empty())
            out << YAML::Key << "thermal_inertia_map" << YAML::Value << config.thermal.thermal_inertia_map;
        if (!config.thermal.h_parameter_map.empty())
            out << YAML::Key << "h_parameter_map" << YAML::Value << config.thermal.h_parameter_map;
        if (config.thermal.use_default_property_map)
            out << YAML::Key << "use_default_property_map" << YAML::Value << config.thermal.use_default_property_map;
        if (config.thermal.property_maps_directory != "data/property_maps")
            out << YAML::Key << "property_maps_directory" << YAML::Value << config.thermal.property_maps_directory;
        out << YAML::Key << "enable_shadow_flux" << YAML::Value << config.thermal.enable_shadow_flux;
        out << YAML::Key << "crater_dd_ratio" << YAML::Value << config.thermal.crater_dd_ratio;
        if (config.thermal.integration_days > 0)
            out << YAML::Key << "integration_days" << YAML::Value << config.thermal.integration_days;
        if (config.thermal.geothermal_heat_flow > 0)
            out << YAML::Key << "geothermal_heat_flow" << YAML::Value << config.thermal.geothermal_heat_flow;
        if (!config.thermal.heat_flow_mask_file.empty())
            out << YAML::Key << "heat_flow_mask_file" << YAML::Value << config.thermal.heat_flow_mask_file;
        if (!config.thermal.heat_flow_mask_file.empty())
            out << YAML::Key << "heat_flow_mask_value" << YAML::Value << config.thermal.heat_flow_mask_value;
        if (!config.thermal.initial_state_file.empty())
            out << YAML::Key << "initial_state_file" << YAML::Value << config.thermal.initial_state_file;
        if (!config.thermal.final_state_file.empty())
            out << YAML::Key << "final_state_file" << YAML::Value << config.thermal.final_state_file;
        if (config.thermal.compute_ice_stability) {
            out << YAML::Key << "compute_ice_stability" << YAML::Value << config.thermal.compute_ice_stability;
            out << YAML::Key << "ice_stability" << YAML::Value << YAML::BeginMap;
            out << YAML::Key << "flux_threshold_kgm2s" << YAML::Value << config.thermal.ice_stability.flux_threshold_kgm2s;
            out << YAML::Key << "grain_hop_length_um" << YAML::Value << config.thermal.ice_stability.grain_hop_length_um;
            out << YAML::Key << "geothermal_gradient_K_per_m" << YAML::Value << config.thermal.ice_stability.geothermal_gradient_K_per_m;
            out << YAML::Key << "sublimation_enthalpy_kJmol" << YAML::Value << config.thermal.ice_stability.sublimation_enthalpy_kJmol;
            out << YAML::Key << "n_depth_subsamples" << YAML::Value << config.thermal.ice_stability.n_depth_subsamples;
            out << YAML::EndMap;
        }
        out << YAML::EndMap;

        // Illumination section
        out << YAML::Key << "illumination" << YAML::Value << YAML::BeginMap;
        out << YAML::Key << "include_primary_reflection" << YAML::Value << config.illumination.include_primary_reflection;
        out << YAML::Key << "use_batch_processing" << YAML::Value << config.illumination.use_batch_processing;
        out << YAML::Key << "use_albedo_map" << YAML::Value << config.illumination.use_albedo_map;
        if (!config.illumination.albedo_map.empty())
            out << YAML::Key << "albedo_map" << YAML::Value << config.illumination.albedo_map;
        out << YAML::Key << "albedo_maps_directory" << YAML::Value << config.illumination.albedo_maps_directory;
        if (!config.illumination.positions_file.empty())
            out << YAML::Key << "positions_file" << YAML::Value << config.illumination.positions_file;
        // Illumination GPU knobs (only emit when not at defaults).
        if (config.illumination.use_gpu)
            out << YAML::Key << "use_gpu" << YAML::Value << config.illumination.use_gpu;
        if (config.illumination.gpu_devices != 1)
            out << YAML::Key << "gpu_devices" << YAML::Value << config.illumination.gpu_devices;
        if (!config.illumination.gpu_device_ids.empty()) {
            out << YAML::Key << "gpu_device_ids" << YAML::Value << YAML::Flow << YAML::BeginSeq;
            for (int id : config.illumination.gpu_device_ids) out << id;
            out << YAML::EndSeq;
        }
        if (config.illumination.gpu_inter_batch_pause_ms != 0)
            out << YAML::Key << "gpu_inter_batch_pause_ms" << YAML::Value
                << config.illumination.gpu_inter_batch_pause_ms;
        if (!config.illumination.bvh_cache_file.empty())
            out << YAML::Key << "bvh_cache_file" << YAML::Value
                << config.illumination.bvh_cache_file;
        out << YAML::EndMap;
        
        // Output section
        out << YAML::Key << "output" << YAML::Value << YAML::BeginMap;
        out << YAML::Key << "directory" << YAML::Value << config.output.directory;
        out << YAML::Key << "format" << YAML::Value << config.output.format;
        out << YAML::Key << "vtk_mode" << YAML::Value << config.output.vtk_mode;
        out << YAML::Key << "include_primary_ir" << YAML::Value << config.output.include_primary_ir;
        if (!config.output.thermal_vtk_file.empty())
            out << YAML::Key << "thermal_vtk_file" << YAML::Value << config.output.thermal_vtk_file;
        if (!config.output.complete_vtk_file.empty())
            out << YAML::Key << "complete_vtk_file" << YAML::Value << config.output.complete_vtk_file;
        if (!config.output.locations_file.empty())
            out << YAML::Key << "locations_file" << YAML::Value << config.output.locations_file;
        if (!config.output.series_output.empty())
            out << YAML::Key << "series_output" << YAML::Value << config.output.series_output;
        if (!config.output.location_data.empty())
            out << YAML::Key << "location_data" << YAML::Value << config.output.location_data;
        if (!config.output.depths_file.empty())
            out << YAML::Key << "depths_file" << YAML::Value << config.output.depths_file;
        if (config.output.skip_time_series)
            out << YAML::Key << "skip_time_series" << YAML::Value << config.output.skip_time_series;
        if (config.output.write_raw_stats)
            out << YAML::Key << "write_raw_stats" << YAML::Value << config.output.write_raw_stats;
        if (config.output.skip_mesh)
            out << YAML::Key << "skip_mesh" << YAML::Value << config.output.skip_mesh;
        out << YAML::EndMap;

        // Advanced section
        out << YAML::Key << "advanced" << YAML::Value << YAML::BeginMap;
        out << YAML::Key << "verbose" << YAML::Value << config.advanced.verbose;
        out << YAML::EndMap;
        
        out << YAML::EndMap;
        
        std::ofstream fout(filename);
        if (!fout) {
            last_error_ = "Cannot open file for writing: " + filename;
            return false;
        }
        fout << out.c_str();
        return true;
        
    } catch (const std::exception& e) {
        last_error_ = std::string("Error saving config: ") + e.what();
        return false;
    }
}

#else

// Fallback implementation without YAML support
SimulationConfig ConfigParser::loadFromYAML(const std::string& filename) {
    last_error_ = "YAML support not available. Please install yaml-cpp library.";
    throw std::runtime_error(last_error_);
}

// PTM-42 fallback. Without yaml-cpp we cannot read the block at all -- but a
// silently unapplied engine_env is the whole bug, so say so.
int ConfigParser::applyEngineEnv(const std::string& filename) {
    std::cerr << "[engine_env] WARNING: this build has no YAML support, so any "
                 "engine_env block in " << filename << " is NOT applied. "
                 "Export those variables yourself (or use "
                 "scripts/ptm_run_with_engine_env.py)." << std::endl;
    return 0;
}

bool ConfigParser::saveToYAML(const std::string& filename, const SimulationConfig& config) {
    last_error_ = "YAML support not available. Please install yaml-cpp library.";
    return false;
}

#endif // HAVE_YAML_CPP

void ConfigParser::overrideFromCommandLine(SimulationConfig& config, int argc, char** argv) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        // Time parameters
        if (arg == "--start" && i + 1 < argc) {
            config.time.start_time = argv[++i];
        } else if (arg == "--duration" && i + 1 < argc) {
            config.time.duration_days = std::stod(argv[++i]);
        } else if (arg == "--timesteps" && i + 1 < argc) {
            config.time.timesteps = std::stoi(argv[++i]);
        } else if ((arg == "--body" || arg == "-b") && i + 1 < argc) {
            config.time.body = argv[++i];
        }
        // Mesh parameters
        else if (arg == "--resolution" && i + 1 < argc) {
            config.mesh.resolution = std::stoi(argv[++i]);
        } else if (arg == "--mesh-file" && i + 1 < argc) {
            config.mesh.import_file = argv[++i];
        } else if (arg == "--occluder-file" && i + 1 < argc) {
            config.mesh.occluder_file = argv[++i];
        } else if (arg == "--mesh-scale" && i + 1 < argc) {
            config.mesh.scale_factor = std::stod(argv[++i]);
        } else if (arg == "--export-basic-mesh") {
            config.mesh.export_basic = true;
        }
        // Thermal parameters
        else if (arg == "--thermal-mesh") {
            config.thermal.run_thermal_mesh = true;
        } else if (arg == "--thermal-threads" && i + 1 < argc) {
            config.thermal.threads = std::stoi(argv[++i]);
        } else if (arg == "--equilibration-orbits" && i + 1 < argc) {
            config.thermal.equilibration_orbits = std::stod(argv[++i]);
        } else if (arg == "--equilibration-steps-per-day" && i + 1 < argc) {
            config.thermal.equilibration_steps_per_day = std::stod(argv[++i]);
        } else if (arg == "--equilibration-samples-per-diurnal" && i + 1 < argc) {
            config.thermal.equilibration_samples_per_diurnal_cycle = std::stod(argv[++i]);
        } else if (arg == "--equilibration-cycles" && i + 1 < argc) {
            config.thermal.equilibration_cycles = std::stoi(argv[++i]);
        } else if (arg == "--no-statistics") {
            config.thermal.compute_statistics = false;
        } else if (arg == "--thermal-inertia-map" && i + 1 < argc) {
            config.thermal.thermal_inertia_map = argv[++i];
        } else if (arg == "--h-parameter-map" && i + 1 < argc) {
            config.thermal.h_parameter_map = argv[++i];
        } else if (arg == "--use-default-property-map") {
            config.thermal.use_default_property_map = true;
        } else if (arg == "--property-maps-directory" && i + 1 < argc) {
            config.thermal.property_maps_directory = argv[++i];
        } else if (arg == "--shadow-flux") {
            config.thermal.enable_shadow_flux = true;
        } else if (arg == "--crater-dd-ratio" && i + 1 < argc) {
            config.thermal.crater_dd_ratio = std::stod(argv[++i]);
        } else if (arg == "--thermal-debug") {
            config.thermal.debug = true;
        } else if (arg == "--thermal-debug-lat" && i + 1 < argc) {
            config.thermal.debug_lat_threshold = std::stod(argv[++i]);
        } else if (arg == "--thermal-m" && i + 1 < argc) {
            config.thermal.grid_m = std::stoi(argv[++i]);
        } else if (arg == "--thermal-n" && i + 1 < argc) {
            config.thermal.grid_n = std::stoi(argv[++i]);
        } else if (arg == "--thermal-b" && i + 1 < argc) {
            config.thermal.grid_b = std::stoi(argv[++i]);
        } else if (arg == "--geothermal-heat-flow" && i + 1 < argc) {
            config.thermal.geothermal_heat_flow = std::stod(argv[++i]);
        } else if (arg == "--heat-flow-mask" && i + 1 < argc) {
            config.thermal.heat_flow_mask_file = argv[++i];
        } else if (arg == "--heat-flow-mask-value" && i + 1 < argc) {
            config.thermal.heat_flow_mask_value = std::stod(argv[++i]);
        }
        // Illumination parameters
        else if (arg == "--no-primary-reflection") {
            config.illumination.include_primary_reflection = false;
        } else if (arg == "--no-batch") {
            config.illumination.use_batch_processing = false;
        } else if (arg == "--no-albedo-map") {
            config.illumination.use_albedo_map = false;
        } else if (arg == "--albedo-map" && i + 1 < argc) {
            config.illumination.albedo_map = argv[++i];
        } else if (arg == "--albedo-maps-directory" && i + 1 < argc) {
            config.illumination.albedo_maps_directory = argv[++i];
        } else if (arg == "--positions" && i + 1 < argc) {
            config.illumination.positions_file = argv[++i];
        }
        // Output parameters
        else if (arg == "--output" && i + 1 < argc) {
            config.output.directory = argv[++i];
        } else if (arg == "--output-format" && i + 1 < argc) {
            config.output.format = argv[++i];
        } else if (arg == "--vtk-mode" && i + 1 < argc) {
            config.output.vtk_mode = argv[++i];
        } else if (arg == "--include-primary-ir") {
            config.output.include_primary_ir = true;
        } else if (arg == "--locations" && i + 1 < argc) {
            config.output.locations_file = argv[++i];
        }
        // Generic body parameters
        else if (arg == "--generic-body-config" && i + 1 < argc) {
            config.generic_body.config_file = argv[++i];
        } else if (arg == "--ephemeris-source" && i + 1 < argc) {
            config.generic_body.ephemeris_source = argv[++i];
        } else if (arg == "--body-radius" && i + 1 < argc) {
            config.generic_body.radius_km = std::stod(argv[++i]);
        } else if (arg == "--orbit-sma" && i + 1 < argc) {
            config.generic_body.semi_major_axis_au = std::stod(argv[++i]);
        } else if (arg == "--orbit-ecc" && i + 1 < argc) {
            config.generic_body.eccentricity = std::stod(argv[++i]);
        } else if (arg == "--orbit-inc" && i + 1 < argc) {
            config.generic_body.inclination_deg = std::stod(argv[++i]);
        } else if (arg == "--orbit-lan" && i + 1 < argc) {
            config.generic_body.lon_ascending_node_deg = std::stod(argv[++i]);
        } else if (arg == "--orbit-argp" && i + 1 < argc) {
            config.generic_body.arg_perihelion_deg = std::stod(argv[++i]);
        } else if (arg == "--orbit-ma" && i + 1 < argc) {
            config.generic_body.mean_anomaly_deg = std::stod(argv[++i]);
        } else if (arg == "--orbit-epoch" && i + 1 < argc) {
            config.generic_body.orbital_epoch = argv[++i];
        } else if (arg == "--spin-period" && i + 1 < argc) {
            config.generic_body.spin_period_hours = std::stod(argv[++i]);
        } else if (arg == "--spin-pole-ra" && i + 1 < argc) {
            config.generic_body.pole_ra_deg = std::stod(argv[++i]);
        } else if (arg == "--spin-pole-dec" && i + 1 < argc) {
            config.generic_body.pole_dec_deg = std::stod(argv[++i]);
        } else if (arg == "--chi" && i + 1 < argc) {
            config.generic_body.chi = std::stod(argv[++i]);
        }
        // Advanced parameters
        else if (arg == "--verbose") {
            config.advanced.verbose = true;
        }
    }
}

bool ConfigParser::generateTemplate(const std::string& filename) {
    std::ofstream out(filename);
    if (!out) {
        std::cerr << "Cannot create file: " << filename << std::endl;
        return false;
    }
    
    out << "# PTM3D Simulation Configuration Template\n";
    out << "# This file contains all available configuration parameters\n";
    out << "# with their default values and descriptions.\n\n";
    
    out << "# ========== Simulation Parameters ==========\n";
    out << "simulation:\n";
    out << "  start_time: \"2025-01-01T00:00:00\"  # ISO 8601 format\n";
    out << "  duration_days: 3.55                # Simulation duration\n";
    out << "  timesteps: 72                       # Number of output timesteps\n";
    out << "  body: \"Europa\"                      # Target body (Europa, Moon, Mercury, etc.)\n\n";
    
    out << "# ========== Mesh Configuration ==========\n";
    out << "mesh:\n";
    out << "  resolution: 5                       # Global mesh resolution\n";
    out << "  import_file: \"\"                     # Import custom mesh (OBJ, PLY, VTK)\n";
    out << "  occluder_file: \"\"                   # Optional occluder-only ring/annulus PLY (shadow ray-tracing only; not solved/output)\n";
    out << "  scale_factor: 1.0                   # Scaling for imported meshes\n";
    out << "  export_basic: false                 # Export basic mesh to VTK\n";
    out << "  use_topographic: false              # Use pre-computed topographic mesh\n";
    out << "  topo_directory: \"\"                  # Directory containing topographic meshes\n";
    out << "                                      # (default: data/topo, with per-body subdir)\n\n";
    
    out << "# ========== Thermal Model ==========\n";
    out << "thermal:\n";
    out << "  run_thermal_mesh: true              # Run thermal model\n";
    out << "  threads: 0                          # Number of threads (0=auto)\n";
    out << "  equilibration_orbits: 10            # Day/night cycles for equilibration\n";
    out << "  equilibration_steps_per_day: -1     # Explicit eq cadence (steps/Earth-day); -1 = auto\n";
    out << "  equilibration_samples_per_diurnal_cycle: 120  # Accuracy target when eq_steps_per_day<=0\n";
    out << "                                      # Calibrated on Moon ico3 vs legacy reference:\n";
    out << "                                      #   120 -> ~4 K max err (Balanced, default)\n";
    out << "                                      #   240 -> ~3 K max err (High accuracy)\n";
    out << "                                      #    30 -> ~7 K max err (Fast)\n";
    out << "                                      #   <=0 -> legacy (match output cadence)\n";
    out << "  # equilibration_cycles: 2           # PASSES over the eq forcing array (GPU solver only).\n";
    out << "  # equilibration_target_K: 0.1        # accuracy the spin-up ESTIMATE targets (report-only).\n";
    out << "                                      # Effective eq = cycles x equilibration_orbits diurnal\n";
    out << "                                      # cycles, at 1x illumination cost and 1x flux RAM.\n";
    out << "                                      # Unset = the solver legacy value, which is 2 (not 1).\n";
    out << "  # eq_extrapolation:                  # Pass-stride RRE on the equilibration state. DEFAULT OFF.\n";
    out << "  #   method: off                      #   off | rre\n";
    out << "  #   order_p: 4                       #   3 or 4; needs equilibration_cycles >= order_p + 1\n";
    out << "  #   tol_K: 0.5                       #   certificate reporting threshold (state ships regardless)\n";
    out << "  #   hold_K: 1.0e-3                   #   hold a layer whose last-pass increment is below this (K)\n";
    out << "  compute_statistics: true            # Compute temperature statistics\n";
    out << "  compute_ice_stability: false        # Schorghofer & Taylor (2007) depth-resolved ice-stability output\n";
    out << "  ice_stability:\n";
    out << "    flux_threshold_kgm2s: 3.0e-14     # J0 (~1 mm/Ga survival rate)\n";
    out << "    grain_hop_length_um: 75.0         # ell, regolith grain diameter\n";
    out << "    geothermal_gradient_K_per_m: 0.0  # extra g on top of solver T(z); 0 = trust solver\n";
    out << "    sublimation_enthalpy_kJmol: 51.058 # Q (Schorghofer Eq. 4)\n";
    out << "    n_depth_subsamples: -1            # -1 = use every native depth node\n";
    out << "  thermal_inertia_map: \"\"             # Path to per-face TI map (binary float32, n_faces values)\n";
    out << "  h_parameter_map: \"\"                 # Path to per-face H-parameter map (alternative to TI map)\n";
    out << "  use_default_property_map: false     # Auto-discover bundled icosphere map for the body\n";
    out << "  property_maps_directory: \"data/property_maps\"  # Root for bundled maps (<dir>/<body>/...)\n";
    out << "  enable_shadow_flux: false           # Enable crater flux enhancement\n";
    out << "  crater_dd_ratio: 8.0                # Crater diameter/depth ratio\n";
    out << "  integration_days: -1                # Integration window (-1=use output window)\n";
    out << "  integration_steps: -1               # Integration steps (-1=auto)\n";
    out << "  integration_offset_days: 0.0        # Offset for integration window\n";
    out << "  integration_offset_steps: 0         # Offset steps for integration\n\n";
    
    out << "# ========== Illumination ==========\n";
    out << "illumination:\n";
    out << "  include_primary_reflection: true    # Include reflected sunlight from primary\n";
    out << "  use_batch_processing: true          # Use SIMD batch processing\n";
    out << "  use_albedo_map: true                 # Auto-discover spatially-varying albedo maps\n";
    out << "  albedo_map: \"\"                      # Custom albedo map (FITS)\n";
    out << "  albedo_maps_directory: \"data/albedo_maps\"  # Albedo maps directory\n";
    out << "  positions_file: \"\"                  # Body positions file\n\n";
    
    out << "# ========== Output Configuration ==========\n";
    out << "output:\n";
    out << "  directory: \"output\"                 # Output directory\n";
    out << "  format: \"binary\"                    # Output format (binary or vtk)\n";
    out << "  vtk_mode: \"both\"                    # VTK mode (thermal, complete, both)\n";
    out << "  include_primary_ir: false           # Include primary IR in VTK\n";
    out << "  thermal_vtk_file: \"\"                # Custom thermal VTK filename\n";
    out << "  complete_vtk_file: \"\"               # Custom complete VTK filename\n";
    out << "  locations_file: \"\"                  # Specific lat/lon points file\n";
    out << "  depths_file: \"\"                     # Optional depths file (m); write per-location depth series\n";
    out << "  series_output: \"\"                   # Location time series output filename\n";
    out << "  location_data: \"\"                   # Custom location data\n";
    out << "  skip_time_series: false             # Skip per-step binary files (saves disk; statistics still computed)\n";
    out << "  write_raw_stats: false              # Write merge-friendly raw stats (statistics/_raw/*.bin) for chunked runs\n";
    out << "  skip_mesh: false                    # Skip writing mesh.ply (used by chunked workflow on chunks 1+)\n\n";

    out << "# ========== Advanced Options ==========\n";
    out << "advanced:\n";
    out << "  verbose: false                      # Verbose output\n";
    out << "  default_visible_albedo: -1.0        # Override albedo (-1=use planet default)\n";
    out << "  chi: -1.0                           # Radiative conductivity parameter (-1=use default 2.7)\n";
    
    std::cout << "Template configuration file created: " << filename << std::endl;
    return true;
}

bool ConfigParser::validate(const SimulationConfig& config) {
    bool valid = true;
    
    // Validate time parameters
    if (config.time.duration_days <= 0) {
        std::cerr << "Error: duration_days must be positive" << std::endl;
        valid = false;
    }
    if (config.time.timesteps <= 0) {
        std::cerr << "Error: timesteps must be positive" << std::endl;
        valid = false;
    }
    
    // Validate mesh parameters
    if (config.mesh.resolution < 1) {
        std::cerr << "Error: mesh resolution must be >= 1" << std::endl;
        valid = false;
    }
    if (config.mesh.scale_factor <= 0) {
        std::cerr << "Error: mesh scale_factor must be positive" << std::endl;
        valid = false;
    }
    
    // Validate thermal parameters
    if (config.thermal.threads < 0) {
        std::cerr << "Error: thermal threads cannot be negative (use 0 for auto)" << std::endl;
        valid = false;
    }
    if (config.thermal.equilibration_orbits < 0.0) {
        std::cerr << "Error: equilibration_orbits cannot be negative" << std::endl;
        valid = false;
    }
    // equilibration_cycles: -1 = unset (solver legacy default), otherwise >= 1.
    if (config.thermal.equilibration_cycles != -1 &&
        config.thermal.equilibration_cycles < 1) {
        std::cerr << "Error: equilibration_cycles must be >= 1 "
                     "(or -1 for the solver default); got "
                  << config.thermal.equilibration_cycles << std::endl;
        valid = false;
    }
    // eq_extrapolation. Every rejection here is FATAL rather than a silent
    // downgrade: an unrecognized method or an order the study showed to be
    // HARMFUL must not be quietly replaced with something that runs.
    {
        const auto& ex = config.thermal.eq_extrapolation;
        if (ex.method != "off" && ex.method != "rre") {
            std::cerr << "Error: thermal.eq_extrapolation.method must be 'off' or 'rre'; got '"
                      << ex.method << "'" << std::endl;
            valid = false;
        }
        if (ex.method == "rre") {
            if (ex.order_p < 3) {
                std::cerr << "Error: thermal.eq_extrapolation.order_p must be >= 3; got "
                          << ex.order_p << ". order_p=2 is not merely weak, it is HARMFUL "
                             "(measured 54.82 K worst-column error vs 31.88 K for plain "
                             "2-pass cycling): the seed sits outside the linear regime and a "
                             "degree-1 annihilator cannot remove it. Refusing rather than "
                             "silently raising it." << std::endl;
                valid = false;
            }
            if (ex.tol_K <= 0.0) {
                std::cerr << "Error: thermal.eq_extrapolation.tol_K must be > 0; got "
                          << ex.tol_K << std::endl;
                valid = false;
            }
            if (!(ex.hold_K > 0.0)) {
                std::cerr << "Error: thermal.eq_extrapolation.hold_K must be > 0; got "
                          << ex.hold_K << std::endl;
                valid = false;
            }
        }
    }
    // equilibration_steps_per_day: allow -1 (sentinel = match output cadence) or strictly positive.
    // Reject 0 and negative-but-not-minus-one values.
    if (config.thermal.equilibration_steps_per_day != -1.0 &&
        config.thermal.equilibration_steps_per_day <= 0.0) {
        std::cerr << "Error: equilibration_steps_per_day must be > 0 "
                     "(or -1 to match output cadence); got "
                  << config.thermal.equilibration_steps_per_day << std::endl;
        valid = false;
    }

    // Per-face thermal property maps: H and TI are mutually exclusive (the
    // model is parameterized on H, and we only want one input source of truth).
    if (!config.thermal.h_parameter_map.empty() &&
        !config.thermal.thermal_inertia_map.empty()) {
        std::cerr << "Error: cannot set both h_parameter_map and "
                     "thermal_inertia_map (choose one)\n";
        valid = false;
    }

    // Validate output format
    if (config.output.format != "binary" && config.output.format != "vtk") {
        std::cerr << "Error: output format must be 'binary' or 'vtk'" << std::endl;
        valid = false;
    }
    
    // Validate VTK mode
    if (config.output.vtk_mode != "thermal" &&
        config.output.vtk_mode != "complete" &&
        config.output.vtk_mode != "both") {
        std::cerr << "Error: vtk_mode must be 'thermal', 'complete', or 'both'" << std::endl;
        valid = false;
    }

    return valid;
}

} // namespace ptm