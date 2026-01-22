/**
 * viewshed_cli.cpp
 * 
 * Command-line interface for PTM3D viewshed analysis
 * 
 * Usage:
 *   viewshed_cli <dem_file> <locations_file> <output_dir> <format>
 * 
 * Arguments:
 *   dem_file       - Input DEM (PLY, OBJ, VTK, or GeoTIFF)
 *   locations_file - CSV file with viewpoint locations
 *   output_dir     - Directory for output files
 *   format         - Output format: vtk, hdf5, or both
 * 
 * Optional flags:
 *   --range <km>      - Maximum viewing range (default: 10 km)
 *   --height <m>      - Observer height above surface (default: 2 m)
 *   --horizon         - Compute horizon profiles
 *   --threads <n>     - Number of threads (default: auto)
 *   --waypoints       - Treat locations as connected waypoints
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <string>
#include <filesystem>
#include <chrono>
#include <algorithm>
#include <cctype>

#include "viewshed_analyzer.h"

namespace fs = std::filesystem;
using namespace ptm;

// Structure for command-line arguments
struct CLIArgs {
    std::string dem_file;
    std::string locations_file;
    std::string output_dir;
    std::string format;
    double max_range_km = 10.0;
    double observer_height = 2.0;
    bool compute_horizon = false;
    int num_threads = 0;
    bool treat_as_waypoints = false;
    bool verbose = false;
};

// Print usage information
void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name 
              << " <dem_file> <locations_file> <output_dir> <format> [options]\n\n";
    std::cout << "Arguments:\n";
    std::cout << "  dem_file       - Input DEM (PLY, OBJ, VTK, or GeoTIFF)\n";
    std::cout << "  locations_file - CSV file with viewpoint locations\n";
    std::cout << "                   Format: label,latitude,longitude,height\n";
    std::cout << "  output_dir     - Directory for output files\n";
    std::cout << "  format         - Output format: vtk, hdf5, or both\n\n";
    std::cout << "Options:\n";
    std::cout << "  --range <km>   - Maximum viewing range (default: 10 km)\n";
    std::cout << "  --height <m>   - Default observer height if not in CSV (default: 2 m)\n";
    std::cout << "  --horizon      - Compute horizon profiles\n";
    std::cout << "  --threads <n>  - Number of threads (default: auto)\n";
    std::cout << "  --waypoints    - Treat locations as connected waypoints/traverse\n";
    std::cout << "  --verbose      - Enable verbose output\n";
    std::cout << "  --help         - Show this help message\n\n";
    std::cout << "Example:\n";
    std::cout << "  " << program_name << " lunar_dem.ply viewpoints.csv ./output vtk --range 20 --waypoints\n\n";
    std::cout << "CSV Format Example:\n";
    std::cout << "  Site1,-87.5,5.0,2.0\n";
    std::cout << "  Site2,-87.3,10.0,5.0\n";
    std::cout << "  Site3,-88.0,15.0,2.0\n";
}

// Parse command-line arguments
bool parseArguments(int argc, char* argv[], CLIArgs& args) {
    if (argc < 5) {
        return false;
    }
    
    // Required arguments
    args.dem_file = argv[1];
    args.locations_file = argv[2];
    args.output_dir = argv[3];
    args.format = argv[4];
    
    // Convert format to lowercase
    std::transform(args.format.begin(), args.format.end(), args.format.begin(), ::tolower);
    
    // Validate format
    if (args.format != "vtk" && args.format != "hdf5" && args.format != "both") {
        std::cerr << "Error: Invalid format '" << args.format 
                  << "'. Must be 'vtk', 'hdf5', or 'both'\n";
        return false;
    }
    
    // Parse optional arguments
    for (int i = 5; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--range" && i + 1 < argc) {
            args.max_range_km = std::stod(argv[++i]);
        } else if (arg == "--height" && i + 1 < argc) {
            args.observer_height = std::stod(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            args.num_threads = std::stoi(argv[++i]);
        } else if (arg == "--horizon") {
            args.compute_horizon = true;
        } else if (arg == "--waypoints") {
            args.treat_as_waypoints = true;
        } else if (arg == "--verbose") {
            args.verbose = true;
        } else if (arg == "--help") {
            return false;
        } else {
            std::cerr << "Warning: Unknown option '" << arg << "'\n";
        }
    }
    
    return true;
}

// Load viewpoint locations from CSV file
std::vector<ViewLocation> loadLocationsFromCSV(const std::string& filename, double default_height) {
    std::vector<ViewLocation> locations;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open locations file: " << filename << std::endl;
        return locations;
    }
    
    std::string line;
    int line_num = 0;
    
    // Skip header if present
    if (std::getline(file, line)) {
        line_num++;
        // Check if first line is a header
        if (line.find("label") != std::string::npos || 
            line.find("lat") != std::string::npos ||
            line.find("Label") != std::string::npos) {
            // Skip header, get next line
            if (!std::getline(file, line)) {
                return locations;
            }
            line_num++;
        }
        
        // Process first data line
        do {
            std::istringstream iss(line);
            std::string label;
            double lat, lon, height = default_height;
            
            // Try to parse: label,lat,lon[,height]
            if (std::getline(iss, label, ',')) {
                std::string lat_str, lon_str, height_str;
                
                if (std::getline(iss, lat_str, ',') && std::getline(iss, lon_str, ',')) {
                    try {
                        lat = std::stod(lat_str);
                        lon = std::stod(lon_str);
                        
                        // Optional height
                        if (std::getline(iss, height_str, ',')) {
                            height = std::stod(height_str);
                        }
                        
                        // Remove quotes from label if present
                        if (!label.empty() && label.front() == '"') {
                            label = label.substr(1);
                        }
                        if (!label.empty() && label.back() == '"') {
                            label.pop_back();
                        }
                        
                        // Create ViewLocation
                        ViewLocation loc(lat, lon, height, label);
                        locations.push_back(loc);
                        
                    } catch (const std::exception& e) {
                        std::cerr << "Warning: Error parsing line " << line_num 
                                  << ": " << e.what() << std::endl;
                    }
                }
            }
            line_num++;
        } while (std::getline(file, line));
    }
    
    file.close();
    
    std::cout << "Loaded " << locations.size() << " viewpoint locations from " 
              << filename << std::endl;
    
    return locations;
}

// Create output filenames
std::string createOutputFilename(const std::string& output_dir, 
                                const std::string& prefix,
                                const std::string& label,
                                const std::string& extension) {
    fs::path dir(output_dir);
    std::string safe_label = label;
    
    // Replace spaces and special characters in label
    std::replace(safe_label.begin(), safe_label.end(), ' ', '_');
    std::replace(safe_label.begin(), safe_label.end(), '/', '_');
    std::replace(safe_label.begin(), safe_label.end(), '\\', '_');
    
    std::string filename = prefix + "_" + safe_label + extension;
    return (dir / filename).string();
}

// Print viewshed statistics
void printViewshedStats(const ViewshedResult& result, bool verbose = false) {
    std::cout << "\n  Statistics for " << result.location.label << ":\n";
    std::cout << "    Location: (" << std::fixed << std::setprecision(4) 
              << result.location.latitude << "°, " 
              << result.location.longitude << "°) @ "
              << result.location.height << "m\n";
    std::cout << "    Visible faces: " << result.num_visible_faces 
              << " (" << std::setprecision(1) 
              << (100.0 * result.num_visible_faces / result.visibility.size()) 
              << "%)\n";
    std::cout << "    Visible area: " << std::setprecision(2) 
              << result.visible_area_km2 << " km²\n";
    std::cout << "    Max range: " << result.max_visible_range_km << " km\n";
    
    if (verbose) {
        std::cout << "    Mean range: " << result.mean_visible_range_km << " km\n";
        std::cout << "    Bounds: [" << std::setprecision(2)
                  << result.visibility_bounds[0] << "°, "
                  << result.visibility_bounds[1] << "°] lat, ["
                  << result.visibility_bounds[2] << "°, "
                  << result.visibility_bounds[3] << "°] lon\n";
        std::cout << "    Computation time: " << std::setprecision(0)
                  << result.computation_time_ms << " ms\n";
    }
}

// Main function
int main(int argc, char* argv[]) {
    // Parse command-line arguments
    CLIArgs args;
    if (!parseArguments(argc, argv, args)) {
        printUsage(argv[0]);
        return 1;
    }
    
    // Print header
    std::cout << "============================================================\n";
    std::cout << "             PTM3D Viewshed Analysis Tool\n";
    std::cout << "============================================================\n\n";
    
    // Validate input files
    if (!fs::exists(args.dem_file)) {
        std::cerr << "Error: DEM file not found: " << args.dem_file << std::endl;
        return 1;
    }
    
    if (!fs::exists(args.locations_file)) {
        std::cerr << "Error: Locations file not found: " << args.locations_file << std::endl;
        return 1;
    }
    
    // Create output directory if needed
    fs::create_directories(args.output_dir);
    
    // Initialize viewshed analyzer
    std::cout << "Loading DEM: " << args.dem_file << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    ViewshedAnalyzer analyzer;
    if (!analyzer.initialize(args.dem_file)) {
        std::cerr << "Error: Failed to initialize viewshed analyzer" << std::endl;
        return 1;
    }
    
    auto load_time = std::chrono::high_resolution_clock::now();
    double load_duration = std::chrono::duration<double>(load_time - start_time).count();
    
    auto mesh = analyzer.getMesh();
    std::cout << "  DEM loaded in " << std::fixed << std::setprecision(2) 
              << load_duration << " seconds\n";
    std::cout << "  Mesh: " << mesh->vertices.size() << " vertices, " 
              << mesh->faces.size() << " faces\n\n";
    
    // Load viewpoint locations
    auto locations = loadLocationsFromCSV(args.locations_file, args.observer_height);
    if (locations.empty()) {
        std::cerr << "Error: No valid locations found in file" << std::endl;
        return 1;
    }
    
    // Set up viewshed parameters
    ViewshedParams params;
    params.max_range_km = args.max_range_km;
    params.compute_horizon = args.compute_horizon;
    params.num_threads = args.num_threads;
    
    std::cout << "Computing viewsheds:\n";
    std::cout << "  Locations: " << locations.size() << "\n";
    std::cout << "  Max range: " << args.max_range_km << " km\n";
    std::cout << "  Mode: " << (args.treat_as_waypoints ? "Waypoints/Traverse" : "Individual points") << "\n";
    std::cout << "  Output format: " << args.format << "\n";
    std::cout << "  Output directory: " << args.output_dir << "\n\n";
    
    // Compute viewsheds
    std::cout << "Processing viewsheds...\n";
    auto compute_start = std::chrono::high_resolution_clock::now();
    
    auto results = analyzer.computeMultipleViewsheds(locations, params);
    
    auto compute_end = std::chrono::high_resolution_clock::now();
    double compute_duration = std::chrono::duration<double>(compute_end - compute_start).count();
    
    // Print statistics
    std::cout << "\nViewshed computation complete (" << std::fixed << std::setprecision(2) 
              << compute_duration << " seconds)\n";
    
    // Print individual statistics
    for (const auto& result : results) {
        printViewshedStats(result, args.verbose);
    }
    
    // Export results
    std::cout << "\nExporting results...\n";
    
    if (args.format == "vtk" || args.format == "both") {
        if (args.treat_as_waypoints) {
            // Export as PVD time series for waypoints
            std::string pvd_file = createOutputFilename(args.output_dir, "traverse", "viewsheds", ".pvd");
            if (analyzer.exportToPVD(pvd_file, results, true)) {
                std::cout << "  Exported traverse to: " << pvd_file << std::endl;
            }
            
            // Also export combined coverage
            ViewshedResult combined;
            combined.visibility.resize(results[0].visibility.size(), 0.0f);
            combined.location.label = "Combined_Coverage";
            
            // Combine all viewsheds
            for (const auto& r : results) {
                for (size_t i = 0; i < r.visibility.size(); ++i) {
                    if (r.visibility[i] > 0) {
                        combined.visibility[i] = 1.0f;
                    }
                }
            }
            
            // Compute statistics for combined
            int visible_count = 0;
            for (float v : combined.visibility) {
                if (v > 0) visible_count++;
            }
            combined.num_visible_faces = visible_count;
            
            std::string combined_file = createOutputFilename(args.output_dir, "combined", "coverage", ".vtk");
            if (analyzer.exportToVTK(combined_file, combined, false)) {
                std::cout << "  Exported combined coverage to: " << combined_file << std::endl;
                std::cout << "    Total coverage: " << visible_count << " faces ("
                          << std::fixed << std::setprecision(1)
                          << (100.0 * visible_count / combined.visibility.size()) << "%)\n";
            }
        } else {
            // Export individual VTK files
            for (const auto& result : results) {
                std::string vtk_file = createOutputFilename(args.output_dir, "viewshed", 
                                                           result.location.label, ".vtk");
                if (analyzer.exportToVTK(vtk_file, result, true)) {
                    std::cout << "  Exported: " << vtk_file << std::endl;
                }
            }
        }
    }
    
#ifdef HAS_HDF5
    if (args.format == "hdf5" || args.format == "both") {
        std::string hdf5_file = createOutputFilename(args.output_dir, "viewsheds", "data", ".h5");
        if (analyzer.exportToHDF5(hdf5_file, results)) {
            std::cout << "  Exported HDF5 data to: " << hdf5_file << std::endl;
        }
    }
#else
    if (args.format == "hdf5" || args.format == "both") {
        std::cout << "  Warning: HDF5 support not compiled in\n";
    }
#endif
    
    // Export horizon profiles if computed
    if (args.compute_horizon) {
        for (const auto& result : results) {
            if (!result.horizon_profile.empty()) {
                std::string horizon_file = createOutputFilename(args.output_dir, "horizon", 
                                                               result.location.label, ".csv");
                std::ofstream hfile(horizon_file);
                if (hfile.is_open()) {
                    hfile << "azimuth_deg,elevation_deg\n";
                    for (const auto& [az, el] : result.horizon_profile) {
                        hfile << az << "," << el << "\n";
                    }
                    hfile.close();
                    std::cout << "  Exported horizon: " << horizon_file << std::endl;
                }
            }
        }
    }
    
    // Summary for waypoints/traverse
    if (args.treat_as_waypoints && results.size() > 1) {
        std::cout << "\nTraverse Summary:\n";
        
        // Calculate cumulative statistics
        std::set<int> all_visible_faces;
        double total_distance = 0.0;
        
        for (const auto& result : results) {
            for (size_t i = 0; i < result.visibility.size(); ++i) {
                if (result.visibility[i] > 0) {
                    all_visible_faces.insert(i);
                }
            }
        }
        
        // Calculate traverse distance
        for (size_t i = 1; i < results.size(); ++i) {
            double dlat = results[i].location.latitude - results[i-1].location.latitude;
            double dlon = results[i].location.longitude - results[i-1].location.longitude;
            // Simple approximation for small distances
            double dist_deg = std::sqrt(dlat*dlat + dlon*dlon);
            total_distance += dist_deg * 111.0; // ~111 km per degree at equator
        }
        
        std::cout << "  Waypoints: " << results.size() << "\n";
        std::cout << "  Approximate traverse distance: " << std::fixed << std::setprecision(2) 
                  << total_distance << " km\n";
        std::cout << "  Total unique faces visible: " << all_visible_faces.size() 
                  << " (" << std::setprecision(1) 
                  << (100.0 * all_visible_faces.size() / mesh->faces.size()) << "%)\n";
    }
    
    // Final timing
    auto end_time = std::chrono::high_resolution_clock::now();
    double total_duration = std::chrono::duration<double>(end_time - start_time).count();
    
    std::cout << "\n============================================================\n";
    std::cout << "Analysis complete!\n";
    std::cout << "Total execution time: " << std::fixed << std::setprecision(2) 
              << total_duration << " seconds\n";
    std::cout << "Results saved to: " << args.output_dir << "\n";
    std::cout << "============================================================\n";
    
    return 0;
}