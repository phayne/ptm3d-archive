#include "config_parser.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>

#ifdef HAVE_YAML_CPP
#include <yaml-cpp/yaml.h>
#endif

namespace ptm {

#ifdef HAVE_YAML_CPP

// Helper template for safe YAML node access with defaults
template<typename T>
T getWithDefault(const YAML::Node& node, const std::string& key, const T& defaultVal) {
    if (node && node[key]) return node[key].as<T>();
    return defaultVal;
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
            config.mesh.use_topographic = getWithDefault(mesh, "use_topographic", config.mesh.use_topographic);
            config.mesh.topo_directory = getWithDefault(mesh, "topo_directory", config.mesh.topo_directory);
            config.mesh.import_file = getWithDefault(mesh, "import_file", config.mesh.import_file);
            config.mesh.scale_factor = getWithDefault(mesh, "scale_factor", config.mesh.scale_factor);
            config.mesh.export_basic = getWithDefault(mesh, "export_basic", config.mesh.export_basic);
        }
        
        // Parse Thermal section
        if (root["thermal"]) {
            const auto& thermal = root["thermal"];
            config.thermal.run_thermal_mesh = getWithDefault(thermal, "run_thermal_mesh", config.thermal.run_thermal_mesh);
            config.thermal.threads = getWithDefault(thermal, "threads", config.thermal.threads);
            config.thermal.equilibration_orbits = getWithDefault(thermal, "equilibration_orbits", config.thermal.equilibration_orbits);
            config.thermal.compute_statistics = getWithDefault(thermal, "compute_statistics", config.thermal.compute_statistics);
            config.thermal.thermal_inertia_map = getWithDefault(thermal, "thermal_inertia_map", config.thermal.thermal_inertia_map);
            config.thermal.enable_shadow_flux = getWithDefault(thermal, "enable_shadow_flux", config.thermal.enable_shadow_flux);
            config.thermal.crater_dd_ratio = getWithDefault(thermal, "crater_dd_ratio", config.thermal.crater_dd_ratio);
            config.thermal.integration_days = getWithDefault(thermal, "integration_days", config.thermal.integration_days);
            config.thermal.integration_steps = getWithDefault(thermal, "integration_steps", config.thermal.integration_steps);
            config.thermal.integration_offset_days = getWithDefault(thermal, "integration_offset_days", config.thermal.integration_offset_days);
            config.thermal.integration_offset_steps = getWithDefault(thermal, "integration_offset_steps", config.thermal.integration_offset_steps);
        }
        
        // Parse Illumination section
        if (root["illumination"]) {
            const auto& n = root["illumination"];
            config.illumination.include_primary_reflection =
                getWithDefault(n, "include_primary_reflection", config.illumination.include_primary_reflection);
            config.illumination.use_batch_processing =
                getWithDefault(n, "use_batch_processing", config.illumination.use_batch_processing);
            config.illumination.albedo_map =
                getWithDefault(n, "albedo_map", config.illumination.albedo_map);
            config.illumination.albedo_maps_directory =
                getWithDefault(n, "albedo_maps_directory", config.illumination.albedo_maps_directory);
            config.illumination.positions_file =
                getWithDefault(n, "positions_file", config.illumination.positions_file);
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
            config.output.no_maps = getWithDefault(output, "no_maps", config.output.no_maps);
            config.output.locations_file = getWithDefault(output, "locations_file", config.output.locations_file);
            config.output.series_output = getWithDefault(output, "series_output", config.output.series_output);
            config.output.location_data = getWithDefault(output, "location_data", config.output.location_data);
            config.output.depths_file = getWithDefault(output, "depths_file", config.output.depths_file);
        }
        
        // Parse Visualization section
        if (root["visualization"]) {
            const auto& vis = root["visualization"];
            config.visualization.create = getWithDefault(vis, "create", config.visualization.create);
            config.visualization.width = getWithDefault(vis, "width", config.visualization.width);
            config.visualization.height = getWithDefault(vis, "height", config.visualization.height);
            config.visualization.fps = getWithDefault(vis, "fps", config.visualization.fps);
            config.visualization.camera_distance = getWithDefault(vis, "camera_distance", config.visualization.camera_distance);
            config.visualization.camera_latitude = getWithDefault(vis, "camera_latitude", config.visualization.camera_latitude);
            config.visualization.camera_longitude = getWithDefault(vis, "camera_longitude", config.visualization.camera_longitude);
            config.visualization.emission_angle = getWithDefault(vis, "emission_angle", config.visualization.emission_angle);
            config.visualization.azimuth = getWithDefault(vis, "azimuth", config.visualization.azimuth);
            config.visualization.save_frames = getWithDefault(vis, "save_frames", config.visualization.save_frames);
        }
        
        // Parse Advanced section
        if (root["advanced"]) {
            const auto& adv = root["advanced"];
            config.advanced.verbose = getWithDefault(adv, "verbose", config.advanced.verbose);
            config.advanced.default_visible_albedo = getWithDefault(adv, "default_visible_albedo", config.advanced.default_visible_albedo);
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
        out << YAML::Key << "use_topographic" << YAML::Value << config.mesh.use_topographic;
        if (!config.mesh.topo_directory.empty())
            out << YAML::Key << "topo_directory" << YAML::Value << config.mesh.topo_directory;
        if (!config.mesh.import_file.empty())
            out << YAML::Key << "import_file" << YAML::Value << config.mesh.import_file;
        out << YAML::Key << "scale_factor" << YAML::Value << config.mesh.scale_factor;
        out << YAML::Key << "export_basic" << YAML::Value << config.mesh.export_basic;
        out << YAML::EndMap;
        
        // Thermal section
        out << YAML::Key << "thermal" << YAML::Value << YAML::BeginMap;
        out << YAML::Key << "run_thermal_mesh" << YAML::Value << config.thermal.run_thermal_mesh;
        out << YAML::Key << "threads" << YAML::Value << config.thermal.threads;
        out << YAML::Key << "equilibration_orbits" << YAML::Value << config.thermal.equilibration_orbits;
        out << YAML::Key << "compute_statistics" << YAML::Value << config.thermal.compute_statistics;
        if (!config.thermal.thermal_inertia_map.empty())
            out << YAML::Key << "thermal_inertia_map" << YAML::Value << config.thermal.thermal_inertia_map;
        out << YAML::Key << "enable_shadow_flux" << YAML::Value << config.thermal.enable_shadow_flux;
        out << YAML::Key << "crater_dd_ratio" << YAML::Value << config.thermal.crater_dd_ratio;
        if (config.thermal.integration_days > 0)
            out << YAML::Key << "integration_days" << YAML::Value << config.thermal.integration_days;
        out << YAML::EndMap;
        
        // Illumination section
        out << YAML::Key << "illumination" << YAML::Value << YAML::BeginMap;
        out << YAML::Key << "include_primary_reflection" << YAML::Value << config.illumination.include_primary_reflection;
        out << YAML::Key << "use_batch_processing" << YAML::Value << config.illumination.use_batch_processing;
        if (!config.illumination.albedo_map.empty())
            out << YAML::Key << "albedo_map" << YAML::Value << config.illumination.albedo_map;
        out << YAML::Key << "albedo_maps_directory" << YAML::Value << config.illumination.albedo_maps_directory;
        if (!config.illumination.positions_file.empty())
            out << YAML::Key << "positions_file" << YAML::Value << config.illumination.positions_file;
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
        out << YAML::Key << "no_maps" << YAML::Value << config.output.no_maps;
        if (!config.output.locations_file.empty())
            out << YAML::Key << "locations_file" << YAML::Value << config.output.locations_file;
        if (!config.output.series_output.empty())
            out << YAML::Key << "series_output" << YAML::Value << config.output.series_output;
        if (!config.output.location_data.empty())
            out << YAML::Key << "location_data" << YAML::Value << config.output.location_data;
        if (!config.output.depths_file.empty())
            out << YAML::Key << "depths_file" << YAML::Value << config.output.depths_file;
        out << YAML::EndMap;

        // Visualization section
        out << YAML::Key << "visualization" << YAML::Value << YAML::BeginMap;
        out << YAML::Key << "create" << YAML::Value << config.visualization.create;
        out << YAML::Key << "width" << YAML::Value << config.visualization.width;
        out << YAML::Key << "height" << YAML::Value << config.visualization.height;
        out << YAML::Key << "fps" << YAML::Value << config.visualization.fps;
        out << YAML::Key << "camera_distance" << YAML::Value << config.visualization.camera_distance;
        out << YAML::Key << "camera_latitude" << YAML::Value << config.visualization.camera_latitude;
        out << YAML::Key << "camera_longitude" << YAML::Value << config.visualization.camera_longitude;
        out << YAML::Key << "save_frames" << YAML::Value << config.visualization.save_frames;
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
        } else if (arg == "--topo") {
            config.mesh.use_topographic = true;
        } else if (arg == "--topo-dir" && i + 1 < argc) {
            config.mesh.topo_directory = argv[++i];
        } else if (arg == "--mesh-file" && i + 1 < argc) {
            config.mesh.import_file = argv[++i];
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
            config.thermal.equilibration_orbits = std::stoi(argv[++i]);
        } else if (arg == "--no-statistics") {
            config.thermal.compute_statistics = false;
        } else if (arg == "--thermal-inertia-map" && i + 1 < argc) {
            config.thermal.thermal_inertia_map = argv[++i];
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
        }
        // Illumination parameters
        else if (arg == "--no-primary-reflection") {
            config.illumination.include_primary_reflection = false;
        } else if (arg == "--no-batch") {
            config.illumination.use_batch_processing = false;
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
        } else if (arg == "--no-maps") {
            config.output.no_maps = true;
        } else if (arg == "--locations" && i + 1 < argc) {
            config.output.locations_file = argv[++i];
        }
        // Visualization parameters
        else if (arg == "--visualization") {
            config.visualization.create = true;
        } else if (arg == "--animation-width" && i + 1 < argc) {
            config.visualization.width = std::stoi(argv[++i]);
        } else if (arg == "--animation-height" && i + 1 < argc) {
            config.visualization.height = std::stoi(argv[++i]);
        } else if (arg == "--camera-distance" && i + 1 < argc) {
            config.visualization.camera_distance = std::stod(argv[++i]);
        } else if (arg == "--camera-lat" && i + 1 < argc) {
            config.visualization.camera_latitude = std::stod(argv[++i]);
        } else if (arg == "--camera-lon" && i + 1 < argc) {
            config.visualization.camera_longitude = std::stod(argv[++i]);
        } else if (arg == "--emission-angle" && i + 1 < argc) {
            config.visualization.emission_angle = std::stod(argv[++i]);
        } else if (arg == "--azimuth" && i + 1 < argc) {
            config.visualization.azimuth = std::stod(argv[++i]);
        } else if (arg == "--save-frames") {
            config.visualization.save_frames = true;
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
    out << "  use_topographic: false              # Use pre-computed topographic meshes\n";
    out << "  topo_directory: \"\"                  # Directory for topographic meshes\n";
    out << "  import_file: \"\"                     # Import custom mesh (OBJ, PLY, VTK)\n";
    out << "  scale_factor: 1.0                   # Scaling for imported meshes\n";
    out << "  export_basic: false                 # Export basic mesh to VTK\n\n";
    
    out << "# ========== Thermal Model ==========\n";
    out << "thermal:\n";
    out << "  run_thermal_mesh: true              # Run thermal model\n";
    out << "  threads: 0                          # Number of threads (0=auto)\n";
    out << "  equilibration_orbits: 10            # Day/night cycles for equilibration\n";
    out << "  compute_statistics: true            # Compute temperature statistics\n";
    out << "  thermal_inertia_map: \"\"             # Path to thermal inertia map\n";
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
    out << "  no_maps: false                      # Disable map generation\n";
    out << "  locations_file: \"\"                  # Specific lat/lon points file\n";
    out << "  depths_file: \"\"                     # Optional depths file (m); write per-location depth series\n";
    out << "  series_output: \"\"                   # Location time series output filename\n";
    out << "  location_data: \"\"                   # Custom location data\n";
    
    out << "# ========== Visualization ==========\n";
    out << "visualization:\n";
    out << "  create: false                       # Generate visualization\n";
    out << "  width: 1024                         # Animation width\n";
    out << "  height: 768                         # Animation height\n";
    out << "  fps: 30                             # Frames per second\n";
    out << "  camera_distance: 5.0                # Camera distance\n";
    out << "  camera_latitude: 30.0               # Camera latitude (degrees)\n";
    out << "  camera_longitude: 0.0               # Camera longitude (degrees)\n";
    out << "  emission_angle: -1.0                # Emission angle (-1=use camera pos)\n";
    out << "  azimuth: -1.0                       # Azimuth angle (-1=use camera pos)\n";
    out << "  save_frames: false                  # Save individual frames\n\n";
    
    out << "# ========== Advanced Options ==========\n";
    out << "advanced:\n";
    out << "  verbose: false                      # Verbose output\n";
    out << "  default_visible_albedo: -1.0        # Override albedo (-1=use planet default)\n";
    
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
    if (config.thermal.equilibration_orbits < 0) {
        std::cerr << "Error: equilibration_orbits cannot be negative" << std::endl;
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
    
    // Validate visualization parameters
    if (config.visualization.width <= 0 || config.visualization.height <= 0) {
        std::cerr << "Error: visualization dimensions must be positive" << std::endl;
        valid = false;
    }
    if (config.visualization.fps <= 0) {
        std::cerr << "Error: fps must be positive" << std::endl;
        valid = false;
    }
    
    return valid;
}

} // namespace ptm