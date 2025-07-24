#include "../mesh/mesh_utils.h"
#include "../illumination/illumination_model.h"
#include "../constants/constants.h"
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sstream> 
#include <filesystem>
#include "config.h"
#include <cstdlib>
#include <string>
#include <SpiceUsr.h>  // Include SPICE functions

#include "../thermal/thermal_model.h"
#include "../thermal/planetary_properties.h"
#include "../maps/spatial_map.h"

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace ptm;
using namespace ptm_thermal::constants;
using namespace ptm_thermal::config;

using namespace thermal;

// Constants for albedo processing
namespace {
    // Minimum physically plausible albedo for icy surfaces
    // Ice has albedo typically 0.5+, but allowing down to 0.05 for "dirty" ice
    constexpr double MIN_VALID_ALBEDO = 0.01;
    
    // Maximum physically plausible albedo (pure white ice/snow)
    constexpr double MAX_VALID_ALBEDO = 0.99;
    
    // Default visible albedo if no valid value is found
    constexpr double DEFAULT_ICE_ALBEDO = 0.55;
    
    // Default IR albedo used for thermal calculations
    constexpr double DEFAULT_IR_ALBEDO = 0.1;
}

// Thread-safe output suppressor class
class OutputSuppressor {
private:
    std::streambuf* original_buffer;
    bool active;
    static std::ofstream null_stream;
    static int ref_count;
    static std::mutex mtx;

public:
    OutputSuppressor() : active(false), original_buffer(nullptr) {
        suppress();
    }
    
    void suppress() {
        std::lock_guard<std::mutex> lock(mtx);
        if (!active) {
            if (ref_count == 0) {
                // Only open the null stream once
                null_stream.open("/dev/null");
            }
            ref_count++;
            original_buffer = std::cout.rdbuf();
            std::cout.rdbuf(null_stream.rdbuf());
            active = true;
        }
    }
    
    void resume() {
        std::lock_guard<std::mutex> lock(mtx);
        if (active) {
            std::cout.rdbuf(original_buffer);
            ref_count--;
            if (ref_count == 0) {
                null_stream.close();
            }
            active = false;
        }
    }
    
    ~OutputSuppressor() {
        resume();
    }
};

// Initialize static members
std::ofstream OutputSuppressor::null_stream;
int OutputSuppressor::ref_count = 0;
std::mutex OutputSuppressor::mtx;

// Thread-safe stderr progress bar class
class StderrProgressBar {
private:
    int total;
    int barWidth;
    std::string prefix;
    bool showProgress;
    std::mutex mtx; // Add mutex for thread safety
    int last_percentage; // Track last percentage to reduce updates

public:
    StderrProgressBar(int total, const std::string& prefix, int barWidth, bool showProgress)
        : total(total), barWidth(barWidth), prefix(prefix), showProgress(showProgress), last_percentage(-1) {}
    
    void update(int current) {
        if (!showProgress) return;
        
        // Calculate current percentage
        int percentage = static_cast<int>((static_cast<float>(current) / total) * 100.0);
        
        // Only update if percentage has changed to reduce mutex contention
        if (percentage == last_percentage && percentage != 100)
            return;
        
        // Lock mutex for thread-safe stderr access
        std::lock_guard<std::mutex> lock(mtx);
        
        // Only update if percentage has changed (double-check after lock)
        if (percentage != last_percentage || percentage == 100) {
            last_percentage = percentage;
            
            float progress = static_cast<float>(current) / total;
            int pos = static_cast<int>(barWidth * progress);
            
            std::cerr << "\r" << prefix << " [";
            for (int i = 0; i < barWidth; ++i) {
                if (i < pos) std::cerr << "=";
                else if (i == pos) std::cerr << ">";
                else std::cerr << " ";
            }
            std::cerr << "] " << percentage << "%";
            std::cerr.flush();
        }
    }
    
    void finish() {
        if (!showProgress) return;
        
        // Lock mutex for thread-safe stderr access
        std::lock_guard<std::mutex> lock(mtx);
        
        std::cerr << "\r" << prefix << " [";
        for (int i = 0; i < barWidth; ++i) {
            std::cerr << "=";
        }
        std::cerr << "] 100%" << std::endl;
    }
};

// Function declarations (prototypes)
bool exportLocationIllumination(const std::string&, const std::string&, const std::shared_ptr<Mesh>&,
    const std::vector<double>&, const std::vector<double>&, const std::vector<double>&,
    const std::vector<double>&, const std::vector<std::string>&, int);
    
bool exportBodyPositions(const std::string&, const std::vector<std::array<double, 3>>&,
    const std::vector<std::array<double, 3>>&, const std::vector<std::string>&);
    
bool runVisualization(const std::string&, const std::string&, const std::string&, 
    const std::string&, const std::string&, const std::string&, const std::string&,
    const std::string&, const std::string&, double, double, double, double, double, int, bool, bool, int, int);
    
bool processLocations(const std::string&, const std::string&, IlluminationModel&);

bool runThermalModel(const std::string&, const std::vector<double>&, const std::vector<double>&,
    double, double, const std::string&, int, const Config&, bool);
    
std::vector<double> runThermalModelForFace(const std::vector<double>&, const std::vector<double>&,
    double, const thermal::Config&, int);
    
bool runThermalModelsForAllFaces(const std::string&, const std::shared_ptr<Mesh>&, 
    const std::vector<double>&, const std::vector<double>&, const std::vector<double>&,
    const std::vector<double>&, int, const thermal::Config&, double, double,
    std::vector<std::vector<double>>*, int, int, int);  // Add int parameter for thermal_threads
    
void saveCheckpoint(const std::string&, const std::vector<size_t>&, 
    const std::vector<std::vector<double>>&);
    
bool addThermalDataToVTK(const std::string&, const std::shared_ptr<Mesh>&,
    const std::vector<std::vector<double>>&, const std::vector<double>&);

void fixOutOfRangeAlbedos(
    std::vector<double>& face_albedos,
    const std::shared_ptr<Mesh>& mesh,
    double default_visible_albedo);

// Add to function declarations section
std::string calculateAdjustedStartTime(const std::string& iso_time, double days_offset);

// Add this class near the top of the file, after the includes
class StdoutRedirector {
private:
    std::streambuf* original_buffer;
    std::ostringstream null_stream;
    bool active;

public:
    StdoutRedirector() : active(false), original_buffer(nullptr) {}
    
    void redirect() {
        if (!active) {
            original_buffer = std::cout.rdbuf();
            std::cout.rdbuf(null_stream.rdbuf());
            active = true;
        }
    }
    
    void restore() {
        if (active && original_buffer) {
            std::cout.rdbuf(original_buffer);
            active = false;
        }
    }
    
    ~StdoutRedirector() {
        restore(); // Always restore on destruction
    }
}; 

std::vector<double> calculateAbsorbedFlux(
    const std::vector<double>& direct_solar_values,
    const std::vector<double>& primary_reflected_values, 
    const std::vector<double>& primary_ir_values,
    int time_steps,
    const std::shared_ptr<ptm::SpatialMap>& albedo_map,
    const std::shared_ptr<Mesh>& mesh,
    double default_visible_albedo,
    double default_ir_albedo,
    const thermal::Config& config) {
    
    // Calculate IR emissivity from IR albedo
    double ir_emissivity = 1.0 - default_ir_albedo;
    
    // Size of output array (same as input)
    size_t n_values = direct_solar_values.size();
    
    // Validate input arrays have expected size
    if (n_values == 0 || primary_reflected_values.size() != n_values || primary_ir_values.size() != n_values) {
        std::cerr << "Error: Input arrays must have matching sizes in calculateAbsorbedFlux" << std::endl;
        std::cerr << "direct_solar: " << direct_solar_values.size() 
                  << ", primary_reflected: " << primary_reflected_values.size()
                  << ", primary_ir: " << primary_ir_values.size() << std::endl;
        return std::vector<double>(); // Return empty vector to signal error
    }
    
    size_t num_faces = n_values / time_steps;
    
    // Validate face count makes sense
    if (num_faces * time_steps != n_values) {
        std::cerr << "Error: Input array size not divisible by time_steps in calculateAbsorbedFlux" << std::endl;
        return std::vector<double>();
    }
    
    // Validate mesh has expected face count
    if (mesh->faces.size() < num_faces) {
        std::cerr << "Error: Mesh has fewer faces (" << mesh->faces.size() 
                  << ") than expected (" << num_faces << ")" << std::endl;
        return std::vector<double>();
    }
    
    // Safety check: ensure latitude/longitude arrays are correctly sized
    if (mesh->latitudes.size() < num_faces || mesh->longitudes.size() < num_faces) {
        std::cerr << "Error: Mesh latitude/longitude arrays too small. "
                  << "Expected: " << num_faces 
                  << ", Got: " << mesh->latitudes.size() << "/" << mesh->longitudes.size() << std::endl;
        return std::vector<double>();
    }
    
    std::vector<double> absorbed_flux(n_values);
    
    // Pre-calculate albedo for each face
    std::vector<double> face_albedos(num_faces, default_visible_albedo);
    
    // Check if we're using a custom mesh
    bool is_custom_mesh = false;
    try {
        auto it = config.custom_properties.find("is_custom_mesh");
        if (it != config.custom_properties.end()) {
            is_custom_mesh = (it->second == "1" || it->second == "true");
        }
    } catch (const std::exception& e) {
        std::cout << "WARNING: Error checking custom mesh status: " << e.what() << std::endl;
    }
    
    std::cout << "Using " << (is_custom_mesh ? "custom" : "standard") << " mesh mode" << std::endl;
    
    // If user explicitly provided an albedo map file, always try to use it first
    bool explicit_albedo_map = config.custom_properties.count("albedo_map") > 0;
    bool using_precomputed = false;
    
    // For custom mesh without explicit albedo map, use global albedo map or default value
    if (is_custom_mesh && !explicit_albedo_map) {
        // Try to load global albedo map if not already provided
        std::shared_ptr<ptm::SpatialMap> global_albedo_map = albedo_map;
        
        // If no albedo map was provided, try to load the global map
        if (!global_albedo_map) {
            // Get the albedo maps directory
            std::string albedo_maps_dir = "data/albedo_maps";
            if (config.custom_properties.count("albedo_maps_directory") > 0) {
                albedo_maps_dir = config.getProperty("albedo_maps_directory");
            } else if (const char* env_dir = std::getenv("EUROPA_ALBEDO_MAPS_DIR")) {
                albedo_maps_dir = env_dir;
            }
            
            // Ensure directory has trailing slash
            if (!albedo_maps_dir.empty() && albedo_maps_dir.back() != '/' && albedo_maps_dir.back() != '\\') {
                albedo_maps_dir += '/';
            }
            
            // Try loading the global albedo map
            std::string global_map_path = albedo_maps_dir + "Europa-0.5deg-BondAlbedo.fits";
            std::cout << "Loading global albedo map from: " << global_map_path << std::endl;

            global_albedo_map = std::make_shared<ptm::SpatialMap>();
            if (!global_albedo_map->loadFromFITS(global_map_path)) {
                std::cout << "Failed to load global albedo map, falling back to default value" << std::endl;
                global_albedo_map = nullptr;
            } else {
                std::cout << "Successfully loaded global albedo map" << std::endl;
            }
        }
        
        // Calculate average albedo from the mesh center if global map is available
        if (global_albedo_map && global_albedo_map->isValid()) {
            // Quick calculation - just calculate mesh center once
            double avg_lat = 0.0, avg_lon = 0.0;
            size_t count = 0;
            
            // Sample points to calculate center (use max ~1000 points for efficiency)
            size_t stride = std::max(size_t(1), mesh->faces.size() / 1000);
            for (size_t f = 0; f < std::min(mesh->faces.size(), num_faces); f += stride) {
                // Bounds check for both arrays
                if (f < mesh->latitudes.size() && f < mesh->longitudes.size()) {
                    avg_lat += mesh->latitudes[f];
                    avg_lon += mesh->longitudes[f];
                    count++;
                }
            }
            
            if (count > 0) {
                avg_lat /= count;
                avg_lon /= count;
                
                // Normalize longitude to 0-360 range for map lookup
                while (avg_lon < 0)
                    avg_lon += 360.0;
                while (avg_lon >= 360.0)
                    avg_lon -= 360.0;
                
                // Get albedo from the global map at this point (note lat sign flip for map)
                double mesh_center_albedo = global_albedo_map->getValue(avg_lon, -avg_lat);
                
                if (mesh_center_albedo != global_albedo_map->getMissingValue() && 
                    mesh_center_albedo >= MIN_VALID_ALBEDO && mesh_center_albedo <= MAX_VALID_ALBEDO) {
                    // Valid albedo value - use it for all faces
                    std::fill(face_albedos.begin(), face_albedos.end(), mesh_center_albedo);
                    
                    std::cout << "Using global map albedo: " << mesh_center_albedo 
                              << " for custom mesh based on center location ("
                              << avg_lat << "°N, " << avg_lon << "°E)" << std::endl;
                    
                    // Skip all other albedo processing steps
                    goto albedo_calculation_complete;
                } else {
                    std::cout << "Invalid albedo value from global map: " << mesh_center_albedo 
                              << ", falling back to default" << std::endl;
                }
            }
        }
        
        // If we get here, we couldn't use the global map, so use the default albedo
        std::cout << "Using constant albedo " << default_visible_albedo 
                  << " for custom mesh (no valid global albedo found)" << std::endl;
    } 
    else if (!is_custom_mesh) {
        // Standard mesh - try to load pre-computed albedo map first
        int mesh_resolution = 4; // Default value
        try {
            // Get mesh resolution from configuration
            auto it = config.custom_properties.find("mesh_resolution");
            if (it != config.custom_properties.end()) {
                mesh_resolution = std::stoi(it->second);
            }
        } catch (const std::exception& e) {
            std::cout << "WARNING: Error reading mesh_resolution: " << e.what() 
                    << ", using default value 4" << std::endl;
        }
        
        // Get albedo map directory from config/environment
        std::string albedo_maps_dir = "data/albedo_maps";
        
        // Check if a custom path is provided in configuration
        if (config.custom_properties.count("albedo_maps_directory") > 0) {
            albedo_maps_dir = config.getProperty("albedo_maps_directory");
        } 
        // Otherwise check environment variable
        else if (const char* env_dir = std::getenv("EUROPA_ALBEDO_MAPS_DIR")) {
            albedo_maps_dir = env_dir;
        }
        
        // Ensure the directory path ends with a separator if needed
        if (!albedo_maps_dir.empty() && albedo_maps_dir.back() != '/' && albedo_maps_dir.back() != '\\') {
            albedo_maps_dir += '/';
        }
        
        std::string body_name = config.getProperty("target_body_name", "Europa");
        // Convert to proper case (first letter uppercase, rest lowercase)
        if (!body_name.empty()) {
            std::transform(body_name.begin(), body_name.end(), body_name.begin(), ::tolower);
            body_name[0] = std::toupper(body_name[0]);
        }

        std::string precomputed_albedo_path = albedo_maps_dir + "/" + body_name + "_Albedo_res" + 
                                    std::to_string(mesh_resolution) + ".bin";
                                            
        std::cout << "Looking for pre-computed albedo map: " << precomputed_albedo_path << std::endl;
        
        // Try to load the pre-computed albedo map
        bool albedo_loaded = false;
        if (std::filesystem::exists(precomputed_albedo_path)) {
            try {
                std::ifstream albedo_file(precomputed_albedo_path, std::ios::binary);
                
                if (albedo_file.is_open()) {
                    // First read the number of faces from header
                    size_t stored_num_faces = 0;
                    albedo_file.read(reinterpret_cast<char*>(&stored_num_faces), sizeof(stored_num_faces));
                    
                    // Verify the number of faces matches what we expect
                    if (stored_num_faces == num_faces) {
                        // Read albedo values directly into face_albedos array
                        albedo_file.read(reinterpret_cast<char*>(face_albedos.data()), 
                                      num_faces * sizeof(double));
                        
                        std::cout << "Successfully loaded pre-computed albedo map for "
                                << num_faces << " faces" << std::endl;
                        albedo_loaded = true;
                        using_precomputed = true;
                    } else {
                        std::cout << "Pre-computed albedo map has incorrect face count: expected " 
                                << num_faces << ", got " << stored_num_faces << std::endl;
                    }
                    
                    albedo_file.close();
                }
            } catch (const std::exception& e) {
                std::cout << "Error loading pre-computed albedo map: " << e.what() << std::endl;
            }
        } else {
            std::cout << "Pre-computed albedo map not found at: " << precomputed_albedo_path << std::endl;
        }
        
        // If pre-computed map failed, try to use global albedo map
        if (!albedo_loaded && albedo_map && albedo_map->isValid()) {
            std::cout << "Using global albedo map for standard mesh" << std::endl;
            
            // Map each face to its corresponding albedo from the global map
            #pragma omp parallel for schedule(dynamic, 1000)
            for (size_t f = 0; f < num_faces; f++) {
                // Add explicit bounds checking
                if (f < mesh->latitudes.size() && f < mesh->longitudes.size()) {
                    double lat = mesh->latitudes[f];
                    double lon = mesh->longitudes[f];
                    
                    // Normalize longitude to 0-360 range for map lookup
                    while (lon < 0.0) lon += 360.0;
                    while (lon >= 360.0) lon -= 360.0;
                    
                    try {
                        // Note: Albedo map typically uses opposite latitude convention (north is positive)
                        double albedo_value = albedo_map->getValue(lon, -lat);
                        
                        // Check if the value is valid
                        if (albedo_value != albedo_map->getMissingValue() && 
                            albedo_value >= MIN_VALID_ALBEDO && 
                            albedo_value <= MAX_VALID_ALBEDO) {
                            face_albedos[f] = albedo_value;
                        }
                    } catch (const std::exception& e) {
                        // If any error occurs during map lookup, just use the default value
                        // (face_albedos already initialized with default_visible_albedo)
                        #pragma omp critical
                        {
                            std::cerr << "Error getting albedo for face " << f 
                                    << " (lat=" << lat << ", lon=" << lon << "): " 
                                    << e.what() << std::endl;
                        }
                    }
                }
            }
            
            // Fix any invalid albedo values by interpolating from neighbors
            fixOutOfRangeAlbedos(face_albedos, mesh, default_visible_albedo);
        }
        // If neither pre-computed nor global map worked, use default albedo
        else if (!albedo_loaded) {
            std::cout << "No albedo map available for standard mesh, using constant albedo: " 
                    << default_visible_albedo << std::endl;
            // face_albedos already initialized with default_visible_albedo
        }
    }

    // If we already have valid albedo values (from any source), we can skip the rest
    if (using_precomputed) {
        std::cout << "Using pre-computed albedo map" << std::endl;
        goto albedo_calculation_complete;
    }
    // Otherwise, check if an explicit albedo map was provided
    else if (explicit_albedo_map && albedo_map && albedo_map->isValid()) {
        std::cout << "Using provided albedo map" << std::endl;
        
        // Apply albedo map to each face
        #pragma omp parallel for schedule(dynamic, 1000)
        for (size_t f = 0; f < num_faces; f++) {
            // Add explicit bounds checking
            if (f < mesh->latitudes.size() && f < mesh->longitudes.size()) {
                double lat = mesh->latitudes[f];
                double lon = mesh->longitudes[f];
                
                // Normalize longitude to 0-360 range for map lookup
                while (lon < 0.0) lon += 360.0;
                while (lon >= 360.0) lon -= 360.0;
                
                try {
                    // Note: Albedo map typically uses opposite latitude convention (north is positive)
                    double albedo_value = albedo_map->getValue(lon, -lat);
                    
                    // Check if the value is valid
                    if (albedo_value != albedo_map->getMissingValue() && 
                        albedo_value >= MIN_VALID_ALBEDO && 
                        albedo_value <= MAX_VALID_ALBEDO) {
                        face_albedos[f] = albedo_value;
                    }
                } catch (const std::exception& e) {
                    // If any error occurs during map lookup, just use the default value
                    // (face_albedos already initialized with default_visible_albedo)
                    #pragma omp critical
                    {
                        std::cerr << "Error getting albedo for face " << f 
                                << " (lat=" << lat << ", lon=" << lon << "): " 
                                << e.what() << std::endl;
                    }
                }
            }
        }
        
        // Fix any invalid albedo values by interpolating from neighbors
        fixOutOfRangeAlbedos(face_albedos, mesh, default_visible_albedo);
        goto albedo_calculation_complete;
    }

    // If we get here, we're using the default constant albedo for all faces
    std::cout << "Using constant albedo " << default_visible_albedo << " for all faces" << std::endl;

albedo_calculation_complete:
    // Final validation to ensure all albedo values are valid
    int invalid_count = 0;
    for (size_t f = 0; f < num_faces; f++) {
        if (std::isnan(face_albedos[f]) || 
            std::isinf(face_albedos[f]) || 
            face_albedos[f] < MIN_VALID_ALBEDO || 
            face_albedos[f] > MAX_VALID_ALBEDO) {
            
            face_albedos[f] = default_visible_albedo;
            invalid_count++;
        }
    }
    
    if (invalid_count > 0) {
        std::cout << "Warning: Fixed " << invalid_count << " invalid albedo values" << std::endl;
    }
    
    // Apply albedo to calculate absorbed flux for each face at each time step
    #pragma omp parallel for schedule(dynamic, 1000)
    for (size_t f = 0; f < num_faces; f++) {
        double visible_albedo = face_albedos[f];
        
        for (size_t t = 0; t < time_steps; t++) {
            size_t idx = f * time_steps + t;
            
            // Add explicit bounds checking
            if (idx >= n_values) {
                #pragma omp critical
                {
                    std::cerr << "Error: Index out of bounds in calculateAbsorbedFlux: " 
                              << idx << " >= " << n_values << std::endl;
                }
                continue;
            }
            
            // Apply appropriate albedo to each component
            double absorbed_direct = direct_solar_values[idx] * (1.0 - visible_albedo);
            double absorbed_reflected = primary_reflected_values[idx] * (1.0 - visible_albedo);
            double absorbed_ir = primary_ir_values[idx] * ir_emissivity;
            
            // Sum all components
            absorbed_flux[idx] = absorbed_direct + absorbed_reflected + absorbed_ir;
        }
    }
    
    return absorbed_flux;
}

// Function to run the thermal model with matched times
std::vector<double> runThermalModelWithMatchedTimes(
    const std::vector<double>& flux_values,
    const std::vector<double>& times,
    double latitude,
    const thermal::Config& config,
    int equilibration_steps) {
    
    try {
        // Replace StdoutRedirector with thread-safe OutputSuppressor
        OutputSuppressor suppressor;
        // No need to explicitly call suppress() as constructor does it
        
        // Create thermal model with proper parameters
        thermal::Planet europa = thermal::Planet(thermal::PlanetaryBody::EUROPA);
        
        // Create local arrays to avoid sharing across threads
        Eigen::MatrixXd flux_arr(flux_values.size(), 2);
        for (size_t i = 0; i < flux_values.size(); i++) {
            flux_arr(i, 0) = times[i];
            flux_arr(i, 1) = flux_values[i];
        }
        
        // Restrict critical section to just check for empty arrays
        #pragma omp critical (thermal_model_creation)
        {
            if (flux_values.empty() || times.empty()) {
                throw std::runtime_error("Empty flux or time arrays");
            }
        }
        
        // Create the model with correct constructor parameters
        thermal::Model thermalModel(europa, latitude * M_PI/180.0, times, flux_arr, config, false);
        
        // Run the model
        thermalModel.run();
        
        // Extract surface temperatures (thread-local)
        std::vector<double> surface_temps(thermalModel.T.rows());
        for (int i = 0; i < thermalModel.T.rows(); i++) {
            surface_temps[i] = thermalModel.T(i, 0);
        }
        
        // OutputSuppressor will automatically resume output in its destructor
        return surface_temps;
    }
    catch (const std::exception& e) {
        #pragma omp critical
        {
            std::cerr << "Exception in runThermalModelWithMatchedTimes: " << e.what() << std::endl;
        }
        // Create a fallback temperature array instead of failing
        return std::vector<double>(flux_values.size(), 100.0);  // Default 100K temperature
    }
    catch (...) {
        #pragma omp critical
        {
            std::cerr << "Unknown exception in runThermalModelWithMatchedTimes" << std::endl;
        }
        return std::vector<double>(flux_values.size(), 100.0);
    }
}

bool exportLocationIllumination(
    const std::string& output_file,
    const std::string& locations_file,
    const std::shared_ptr<Mesh>& mesh,
    const std::vector<double>& illumination_values,
    const std::vector<double>& direct_solar_values,      // Added component
    const std::vector<double>& primary_reflected_values, // Added component
    const std::vector<double>& primary_ir_values,        // Added IR component
    const std::vector<std::string>& time_strings,
    int time_steps) {
    
    // Read locations from file
    std::cout << "Attempting to open locations file: " << locations_file << std::endl;
    std::ifstream input(locations_file);
    if (!input.is_open()) {
        std::cerr << "Failed to open locations file: " << locations_file << std::endl;
        return false;
    }
    std::cout << "Successfully opened locations file" << std::endl;
    
    // Parse locations
    std::vector<std::pair<double, double>> locations;
    std::vector<std::string> location_names;
    
    std::string line;
    while (std::getline(input, line)) {
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') {
            continue;
        }
        
        std::istringstream iss(line);
        double lat, lon;
        std::string name;
        
        if (!(iss >> lat >> lon)) {
            std::cerr << "Error parsing location: " << line << std::endl;
            continue;
        }
        
        // Get name if present, otherwise use coordinates
        std::getline(iss >> std::ws, name);
        if (name.empty()) {
            std::ostringstream name_ss;
            name_ss << "Location_" << lat << "_" << lon;
            name = name_ss.str();
        }
        
        locations.emplace_back(lat, lon);
        location_names.push_back(name);
    }
    
    if (locations.empty()) {
        std::cerr << "No valid locations found in file" << std::endl;
        return false;
    }
    
    // Find closest mesh facet for each location
    std::vector<size_t> facet_indices(locations.size());
    
    for (size_t i = 0; i < locations.size(); i++) {
        const auto& location = locations[i];
        double min_distance = std::numeric_limits<double>::max();
        size_t closest_facet = 0;
        
        // Find the facet with the closest center to our location
        for (size_t f = 0; f < mesh->faces.size(); f++) {
            double facet_lat = mesh->latitudes[f];
            double facet_lon = mesh->longitudes[f];
            
            // Simple spherical distance calculation
            double dlat = facet_lat - location.first;
            double dlon = facet_lon - location.second;
            while (dlon > 180.0) dlon -= 360.0;
            while (dlon < -180.0) dlon += 360.0;
            
            double distance = std::sqrt(dlat*dlat + dlon*dlon);
            
            if (distance < min_distance) {
                min_distance = distance;
                closest_facet = f;
            }
        }
        
        facet_indices[i] = closest_facet;
        std::cout << "Location " << location_names[i] << " (" << location.first << ", " << location.second 
                  << ") mapped to facet " << closest_facet << " at (" 
                  << mesh->latitudes[closest_facet] << ", " << mesh->longitudes[closest_facet] << ")" << std::endl;
    }
    
    // Open output file
    std::ofstream output(output_file);
    if (!output.is_open()) {
        std::cerr << "Failed to open output file: " << output_file << std::endl;
        return false;
    }
    
    // Modify the CSV header section
    output << "Time";
    for (const auto& name : location_names) {
        output << "," << name << "_Total";
        output << "," << name << "_DirectSolar";
        output << "," << name << "_PrimaryReflected";
        output << "," << name << "_PrimaryIR";
    }
    output << std::endl;
    
    // Modify the data output section
    for (int t = 0; t < time_steps; t++) {
        // Write time
        output << time_strings[t];
        
        // Write illumination components for each location
        for (size_t i = 0; i < locations.size(); i++) {
            size_t facet_idx = facet_indices[i];
            size_t idx = facet_idx * time_steps + t;
            
            // Write total illumination (solar direct + reflected)
            output << "," << illumination_values[idx];
            
            // Write direct solar component
            output << "," << direct_solar_values[idx];
            
            // Write Primary reflected solar component
            output << "," << primary_reflected_values[idx];
            
            // Write Primary IR component
            output << "," << primary_ir_values[idx];
        }
        output << std::endl;
    }
    
    std::cout << "Location illumination data written to " << output_file << std::endl;
    return true;
}

// Add this function to export body positions
bool exportBodyPositions(
    const std::string& output_file,
    const std::vector<std::array<double, 3>>& sun_positions,
    const std::vector<std::array<double, 3>>& primary_positions,
    const std::vector<std::string>& time_strings) {
    
    std::ofstream output(output_file);
    if (!output.is_open()) {
        std::cerr << "Failed to open body positions file: " << output_file << std::endl;
        return false;
    }
    
    // Write CSV header
    output << "time,sun_x,sun_y,sun_z,primary_x,primary_y,primary_z" << std::endl;

    // Write positions for each time step
    for (size_t i = 0; i < time_strings.size(); i++) {
        const auto& sun_pos = sun_positions[i];
        const auto& primary_pos = primary_positions[i];
        
        output << time_strings[i] << ","
               << sun_pos[0] << "," << sun_pos[1] << "," << sun_pos[2] << ","
               << primary_pos[0] << "," << primary_pos[1] << "," << primary_pos[2]
               << std::endl;
    }
    
    std::cout << "Body positions written to " << output_file << std::endl;
    return true;
}

std::vector<double> calculateAbsorbedFlux(
    const std::vector<double>& direct_solar_values,
    const std::vector<double>& primary_reflected_values, 
    const std::vector<double>& primary_ir_values,
    int time_steps,
    const std::shared_ptr<ptm::SpatialMap>& albedo_map,
    const std::shared_ptr<Mesh>& mesh,
    double default_visible_albedo,
    double default_ir_albedo,
    const thermal::Config& config); // Default param for backward compatibility
    
bool runVisualization(
    const std::string& illumination_vtk_file, 
    const std::string& output_video,
    const std::string& output_ir_video,
    const std::string& thermal_vtk_file,
    const std::string& output_thermal_video,
    const std::string& output_maps_dir,
    const std::string& positions_file,
    const std::string& location_data_file,
    const std::string& location_plots_dir,
    double camera_distance,
    double camera_latitude,
    double camera_longitude,
    double emission_angle,
    double azimuth,
    int fps,
    bool save_frames,
    bool skip_maps,
    int width = 1024,
    int height = 768) {
    
    // Debug output to verify parameters are received correctly
    std::cout << "Starting visualization with camera: lat=" << camera_latitude
              << "°, lon=" << camera_longitude 
              << "°, distance=" << camera_distance << std::endl;
    
    // Check if thermal VTK file exists
    bool thermal_file_exists = !thermal_vtk_file.empty() && std::filesystem::exists(thermal_vtk_file);
    
    if (!thermal_vtk_file.empty()) {
        if (thermal_file_exists) {
            std::cout << "Thermal VTK file found at: " << thermal_vtk_file << std::endl;
        } else {
            std::cout << "Warning: Thermal VTK file specified but not found at: " << thermal_vtk_file << std::endl;
        }
    }
    
    // Check if location data file exists
    bool location_data_exists = false;
    if (!location_data_file.empty()) {
        location_data_exists = std::filesystem::exists(location_data_file);
        if (location_data_exists) {
            std::cout << "Location data file found at: " << location_data_file << std::endl;
        } else {
            std::cout << "Warning: Location data file specified but not found at: " << location_data_file << std::endl;
        }
    }
    
    // Path to the Python script
    std::string script_path = "../src/python/europa_visualization.py";
    
    // Check if the script exists
    if (!std::filesystem::exists(script_path)) {
        std::cerr << "Error: Visualization script not found at: " << script_path << std::endl;
        return false;
    }
    
    // Check multiple possible module locations
    std::vector<std::string> possible_module_paths = {
        "../build/src/python/bindings",
        "../src/python",
        ".",
        "./build/src/python/bindings"
    };

    std::string pythonpath = "";
    for (const auto& path : possible_module_paths) {
        // Check if directory exists
        if (std::filesystem::exists(path)) {
            pythonpath += path + ":";
        }
    }

    // Remove trailing colon
    if (!pythonpath.empty()) {
        pythonpath.pop_back();
    }

    // Get conda environment Python if available
    std::string python_cmd = "python3";
    const char* conda_prefix = std::getenv("CONDA_PREFIX");
    if (conda_prefix != nullptr) {
        std::string conda_python = std::string(conda_prefix) + "/bin/python";
        if (std::filesystem::exists(conda_python)) {
            python_cmd = conda_python;
            std::cout << "Using conda environment Python: " << python_cmd << std::endl;
        }
    }

    // Construct command with PYTHONPATH
    std::string command;
    if (!pythonpath.empty()) {
        command = "PYTHONPATH=" + pythonpath + " " + python_cmd + " " + script_path;
    } else {
        command = python_cmd + " " + script_path;
    }
    
    // Illumination VTK for solar/IR
    command += " --vtk \"" + illumination_vtk_file + "\"";
    
    // Add thermal VTK file path if available
    if (thermal_file_exists) {
        command += " --thermal-vtk \"" + thermal_vtk_file + "\"";
    }
    
    if (!output_video.empty()) {
        command += " --output-video \"" + output_video + "\"";
    }
    
    // Add the IR video parameter
    if (!output_ir_video.empty()) {
        command += " --output-ir-video \"" + output_ir_video + "\"";
    }
    
    // Simply add the thermal video if requested
    if (!output_thermal_video.empty()) {
        command += " --output-thermal-video \"" + output_thermal_video + "\"";
    }
    
    // Only pass the output maps directory if we're not skipping maps
    if (!skip_maps && !output_maps_dir.empty()) {
        command += " --output-maps \"" + output_maps_dir + "\"";
    }
    
    // Add width and height parameters to the command
    command += " --width " + std::to_string(width) + " --height " + std::to_string(height);
    
    // Debug output to verify camera parameters before building command
    std::cout << "Camera parameters for Python: lat=" << camera_latitude 
              << ", lon=" << camera_longitude 
              << ", distance=" << camera_distance << std::endl;
    
    // Add camera or emission angle parameters
    if (emission_angle >= 0.0 && azimuth >= 0.0) {
        command += " --emission-angle " + std::to_string(emission_angle);
        command += " --azimuth " + std::to_string(azimuth);
    } else {
        // Explicit conversion to string to avoid any potential issues
        std::stringstream lat_ss, lon_ss, dist_ss;
        lat_ss << std::fixed << std::setprecision(6) << camera_latitude;
        lon_ss << std::fixed << std::setprecision(6) << camera_longitude;
        dist_ss << std::fixed << std::setprecision(6) << camera_distance;
        
        command += " --camera-distance " + dist_ss.str();
        command += " --camera-lat " + lat_ss.str();
        command += " --camera-lon " + lon_ss.str();
    }
    
    command += " --fps " + std::to_string(fps);
    
    // Add positions file to command
    if (!positions_file.empty()) {
        command += " --positions \"" + positions_file + "\"";
    }
    
    // Add location data file if it exists
    if (location_data_exists) {
        command += " --location-data \"" + location_data_file + "\"";
        
        // Only add location plots directory if we have valid location data
        if (!location_plots_dir.empty()) {
            // Make sure the directory exists
            std::filesystem::create_directories(location_plots_dir);
            command += " --output-location-plots \"" + location_plots_dir + "\"";
        } else if (!output_video.empty()) {
            // If no explicit plots directory provided, use a default location
            std::string default_plots_dir = std::filesystem::path(output_video).parent_path().string() + "/location_plots";
            std::filesystem::create_directories(default_plots_dir);
            command += " --output-location-plots \"" + default_plots_dir + "\"";
            std::cout << "Using default location plots directory: " << default_plots_dir << std::endl;
        }
    }
    
    // Add save frames argument to command if provided
    if (save_frames) {
        command += " --save-frames";
    }
    
    // We still need the skip_maps flag in case other code paths create maps
    if (skip_maps) {
        command += " --skip-maps";
    }
    
    // Print the full command for debugging
    std::cout << "Running visualization command: " << command << std::endl;
    
    // Execute the command
    int result = std::system(command.c_str());
    
    if (result != 0) {
        std::cerr << "Warning: Visualization command returned non-zero code: " << result << std::endl;
        return false;
    }
    
    if (location_data_exists) {
        std::cout << "Location data processed. Check for plots in the output directory." << std::endl;
    }
    
    return true;
}

// Example of using the improved function with a location file
bool processLocations(const std::string& location_file, 
                       const std::string& output_file,
                       IlluminationModel& model) {
    // Read locations from file
    std::vector<std::pair<double, double>> locations;
    std::vector<std::string> location_names;
    
    std::ifstream file(location_file);
    if (!file) {
        std::cerr << "Error: Could not open location file: " << location_file << std::endl;
        return false;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;  // Skip comments and empty lines
        
        std::istringstream iss(line);
        double lat, lon;
        if (iss >> lat >> lon) {
            locations.push_back({lat, lon});
            
            // Get name if present
            std::string name;
            std::getline(iss >> std::ws, name);  // Read rest of line as name
            if (name.empty()) {
                name = "Location_" + std::to_string(locations.size());
            }
            location_names.push_back(name);
        }
    }
    
    if (locations.empty()) {
        std::cerr << "No valid locations found in file" << std::endl;
        return false;
    }
    
    // Calculate illumination for all locations
    auto results = model.calculateLocationIllumination(
        locations, "2020-01-01T00:00:00", 1.0);
    
    // Write results to CSV file
    std::ofstream out_file(output_file);
    if (!out_file) {
        std::cerr << "Error: Could not open output file: " << output_file << std::endl;
        return false;
    }
    
    // Write header
    out_file << "Location,Latitude,Longitude";
    for (size_t t = 0; t < model.getTimes().size(); t++) {
        out_file << ",Time_" << t;
    }
    out_file << std::endl;
    
    // Write data
    for (size_t i = 0; i < locations.size(); i++) {
        out_file << location_names[i] << "," 
                << locations[i].first << "," 
                << locations[i].second;
        
        for (double value : results[i]) {
            out_file << "," << value;
        }
        out_file << std::endl;
    }
    
    return true;
}

bool runThermalModel(
    const std::string& output_file,
    const std::vector<double>& flux_values,
    const std::vector<double>& times,
    double latitude,
    double longitude,
    const std::string& location_name,
    int days,  // We'll keep this parameter for backwards compatibility
    const Config& config,
    bool verbose = false) {
    
    try {
        std::cout << "Running thermal model for location: " 
                  << location_name << " (Lat: " << latitude << ", Lon: " << longitude << ")" << std::endl;
        
        // Create the planet model (Europa by default)
        Planet europa("Europa");
        
        // Create a flux array in Eigen format
        Eigen::MatrixXd flux_arr(flux_values.size(), 2);
        for (size_t i = 0; i < flux_values.size(); i++) {
            flux_arr(i, 0) = times[i];           // Time (seconds)
            flux_arr(i, 1) = flux_values[i];      // Flux (W/m²)
        }
        
        // FIXED: Use vector of times instead of integer days parameter
        Model thermalModel(europa, latitude * M_PI/180.0, times, flux_arr, config, verbose);
        
        // Run the thermal model
        auto start_time = std::chrono::high_resolution_clock::now();
        thermalModel.run();
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time).count();
        
        std::cout << "Thermal model completed in " << duration_ms << " ms" << std::endl;
        
        // Write results to CSV file
        std::ofstream outfile(output_file);
        if (!outfile.is_open()) {
            std::cerr << "Failed to open output file: " << output_file << std::endl;
            return false;
        }
        
        // Write header
        outfile << "Time(hours),";
        for (int i = 0; i < thermalModel.profile.z().size(); i++) {
            outfile << "T_" << thermalModel.profile.z()[i] << "m";
            if (i < thermalModel.profile.z().size() - 1) outfile << ",";
        }
        outfile << std::endl;
        
        // Write data
        for (int i = 0; i < thermalModel.T.rows(); i++) {
            outfile << thermalModel.lt(i) / 3600.0 << ","; // Time in hours
            for (int j = 0; j < thermalModel.T.cols(); j++) {
                outfile << thermalModel.T(i, j);
                if (j < thermalModel.T.cols() - 1) outfile << ",";
            }
            outfile << std::endl;
        }
        outfile.close();
        
        // Print summary statistics
        double min_temp = thermalModel.T.col(0).minCoeff();
        double max_temp = thermalModel.T.col(0).maxCoeff();
        double avg_temp = thermalModel.T.col(0).mean();

        std::cout << "Surface Temperature Summary for " << location_name << ":" << std::endl;
        std::cout << "  Minimum: " << min_temp << " K" << std::endl;
        std::cout << "  Maximum: " << max_temp << " K" << std::endl;
        std::cout << "  Average: " << avg_temp << " K" << std::endl;
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error in thermal model: " << e.what() << std::endl;
        return false;
    }
}

// Fix the runThermalModelForFace function to ensure time alignment
std::vector<double> runThermalModelForFace(
    const std::vector<double>& flux_values,
    const std::vector<double>& times,
    double latitude,
    const thermal::Config& config,
    int equilibration_steps) {
    
    try {
        // Create planet model
        Planet europa("Europa");
        
        // Create flux array in the expected format
        Eigen::MatrixXd flux_arr(flux_values.size(), 2);
        for (size_t i = 0; i < flux_values.size(); i++) {
            flux_arr(i, 0) = times[i];
            flux_arr(i, 1) = flux_values[i];
        }
        
        // SIMPLIFIED APPROACH: Create thermal model with all time points
        Model thermalModel(europa, latitude * M_PI/180.0, times, flux_arr, config, false);
        
        // Do NOT use internal equilibration filtering - let the model run continuously
        // for all time points, just like the illumination model did
        thermalModel.run();
        
        // Return ALL temperatures for manual filtering later
        std::vector<double> surface_temps(thermalModel.T.rows());
        for (int i = 0; i < thermalModel.T.rows(); i++) {
            surface_temps[i] = thermalModel.T(i, 0);  // Column 0 = surface
        }
        
        return surface_temps;
    } catch (const std::exception& e) {
        return {};
    }
}

// Function to run thermal models for all mesh faces
bool runThermalModelsForAllFaces(
    const std::string& thermal_output_dir,
    const std::shared_ptr<Mesh>& mesh,
    const std::vector<double>& direct_solar_values,
    const std::vector<double>& primary_reflected_values,
    const std::vector<double>& primary_ir_values,
    const std::vector<double>& times,
    int time_steps,
    const thermal::Config& config,
    double default_visible_albedo,
    double ir_albedo,
    std::vector<std::vector<double>>* surface_temps_out,
    int equilibration_steps,
    int total_time_steps,
    int thermal_threads) {
    
    // Load albedo map if specified
    std::shared_ptr<ptm::SpatialMap> albedo_map;
    std::string albedo_map_path;

    // Check if the property exists in the config
    if (config.custom_properties.find("albedo_map") != config.custom_properties.end()) {
        albedo_map_path = config.getProperty("albedo_map");
        
        if (!albedo_map_path.empty()) {
            albedo_map = std::make_shared<ptm::SpatialMap>();
            if (albedo_map->loadFromFITS(albedo_map_path)) {
                std::cout << "Using albedo map: " << albedo_map_path << std::endl;
                
                // Print some statistics about the map
                std::cout << "  - Map resolution: " << albedo_map->getWidth() << "x" << albedo_map->getHeight() << std::endl;
                std::cout << "  - Albedo range: " << albedo_map->getMin() << " to " << albedo_map->getMax() << std::endl;
                std::cout << "  - Coordinate range: Lon [" << albedo_map->getLonMin() << "," << albedo_map->getLonMax() 
                          << "], Lat [" << albedo_map->getLatMin() << "," << albedo_map->getLatMax() << "]" << std::endl;
            } else {
                std::cout << "Warning: Failed to load albedo map: " << albedo_map_path << std::endl;
                std::cout << "Error: " << albedo_map->getLastError() << std::endl;
                albedo_map.reset();
            }
        }
    }
    
    // Use illumination model's albedo instead of constants::BODY_ALBEDO
    //double visible_albedo = illumination_model->getAlbedo();
    ir_albedo = DEFAULT_IR_ALBEDO;

    // Calculate total absorbed flux at each face and time
    std::vector<double> absorbed_flux = calculateAbsorbedFlux(
        direct_solar_values,            // Use parameter instead of illumination_model->getDirectSolarValues()
        primary_reflected_values,       // Use parameter instead of illumination_model->getPrimaryReflectedValues()
        primary_ir_values,              // Use parameter instead of illumination_model->getPrimaryIRFlux()
        time_steps,
        albedo_map,
        mesh,
        default_visible_albedo,         // Use parameter instead of illumination_model->getAlbedo()
        ir_albedo,                      // Use parameter as is (don't redefine)
        config);
    
    size_t n_faces = mesh->faces.size();
    
    // Print initial message
    std::cerr << "\nRunning thermal models for " << n_faces << " mesh faces..." << std::endl;
    
    // Start timing - Add timer here
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // CHUNKED PROCESSING - for large meshes
    // Process faces in chunks to limit peak memory usage
    const size_t CHUNK_SIZE = 100000; // Process 100k faces at a time 
    const size_t NUM_CHUNKS = (n_faces + CHUNK_SIZE - 1) / CHUNK_SIZE;
    
    // Initialize results container if output is requested
    if (surface_temps_out) {
        surface_temps_out->clear();
        surface_temps_out->resize(n_faces);
    }
    
    // Atomic counter for progress tracking
    std::atomic<int> completed_faces_atomic(0);
    
    // Process in chunks to reduce memory pressure
    bool success = true;
    
    // Create progress bar for thermal calculation - using cerr for output
    thermal::ProgressBar progressBar(n_faces, "Thermal Simulation", 50, true);
    
    for (size_t chunk = 0; chunk < NUM_CHUNKS; chunk++) {
        size_t start_face = chunk * CHUNK_SIZE;
        size_t end_face = std::min((chunk + 1) * CHUNK_SIZE, n_faces);
        size_t chunk_size = end_face - start_face;
        
        // Show chunk info only if there are multiple chunks
        if (NUM_CHUNKS > 1) {
            std::cerr << "Processing chunk " << (chunk + 1) << " of " << NUM_CHUNKS 
                      << " (faces " << start_face << " to " << (end_face - 1) << ")" << std::endl;
        }
        
        // Temporary storage for this chunk
        std::vector<std::vector<double>> chunk_results(chunk_size);
        
        // Initialize each temperature array with default value of 50.0K (same as original code)
        for (size_t i = 0; i < chunk_size; i++) {
            chunk_results[i].resize(total_time_steps, 50.0);
        }
        
        // Reset counter for this chunk
        completed_faces_atomic.store(0);
        
        // Process faces in this chunk in parallel
        #pragma omp parallel for schedule(dynamic, 1) num_threads(thermal_threads)
        for (size_t local_idx = 0; local_idx < chunk_size; local_idx++) {
            size_t f = start_face + local_idx;
            
            try {
                // Process each face
                std::vector<double> face_flux(total_time_steps);
                for (int t = 0; t < total_time_steps; t++) {
                    size_t idx = f * total_time_steps + t;
                    if (idx < absorbed_flux.size()) {
                        face_flux[t] = absorbed_flux[idx];
                    } else {
                        face_flux[t] = 0.0; // Safety fallback
                    }
                }

                // Run thermal model for this face
                std::vector<double> temps = runThermalModelWithMatchedTimes(
                    face_flux,
                    times,
                    mesh->latitudes[f],
                    config,
                    equilibration_steps
                );

                // Store results locally in the chunk
                chunk_results[local_idx] = std::move(temps);
                
                // Update progress
                int completed = ++completed_faces_atomic;
                if (completed % 100 == 0 || completed == static_cast<int>(chunk_size)) {
                    // Update the progress bar with global progress
                    progressBar.update(start_face + completed);
                }
            } catch (const std::exception& e) {
                #pragma omp critical
                {
                    std::cerr << "Error processing face " << f << ": " << e.what() << std::endl;
                    success = false;
                }
            } catch (...) {
                #pragma omp critical
                {
                    std::cerr << "Unknown exception in thermal model for face " << f << std::endl;
                    success = false;
                }
            }
        }
        
        // Copy chunk results to main output if requested
        if (surface_temps_out) {
            for (size_t local_idx = 0; local_idx < chunk_size; local_idx++) {
                size_t f = start_face + local_idx;
                (*surface_temps_out)[f] = std::move(chunk_results[local_idx]);
            }
        }
        
        // Force cleanup of chunk memory before starting next chunk
        chunk_results.clear();
        chunk_results.shrink_to_fit();
        
        // Add a newline between chunks if there are multiple
        if (NUM_CHUNKS > 1 && chunk < NUM_CHUNKS - 1) {
            std::cerr << std::endl;
        }
    }
    
    // Ensure progress bar is completed
    progressBar.finish();
    
    // End timing and calculate duration
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    double seconds = duration.count() / 1000.0;
    
    // Print timing information using cerr to match illumination code
    std::cerr << "Thermal computation completed in " << std::fixed << std::setprecision(3) 
              << seconds << " seconds (" << duration.count() << " ms)" << std::endl;
    
    if (success) {
        std::cerr << "\nThermal models completed successfully!" << std::endl;
    } else {
        std::cerr << "\nThermal models completed with errors!" << std::endl;
    }
    
    return success;
}

// Function to export thermal results to VTK format - complete rewrite
bool addThermalDataToVTK(
    const std::string& vtk_file,
    const std::shared_ptr<Mesh>& mesh,
    const std::vector<std::vector<double>>& surface_temps,
    const std::vector<double>& times) {
    
    // Read the entire original file
    std::ifstream infile(vtk_file);
    if (!infile.is_open()) {
        std::cerr << "Failed to open VTK file: " << vtk_file << std::endl;
        return false;
    }
    
    std::vector<std::string> content;
    std::string line;
    bool found_cell_data = false;
    int cell_data_line = -1;
    int cell_count = 0;
    
    // Read and parse the file
    while (std::getline(infile, line)) {
        content.push_back(line);
        if (line.find("CELL_DATA") != std::string::npos) {
            found_cell_data = true;
            cell_data_line = content.size() - 1;
            std::istringstream iss(line);
            std::string dummy;
            iss >> dummy >> cell_count;
        }
    }
    infile.close();
    
    // Make sure we found the CELL_DATA section
    if (!found_cell_data) {
        std::cerr << "Could not find CELL_DATA section in VTK file" << std::endl;
        return false;
    }
    
    // Create temperature data blocks
    std::vector<std::string> temp_data;
    for (size_t t = 0; t < times.size(); t++) {
        temp_data.push_back(std::string("SCALARS temperature_") + std::to_string(t) + " float 1");
        temp_data.push_back("LOOKUP_TABLE default");
        
        // Write one value per face
        for (size_t f = 0; f < mesh->faces.size(); f++) {
            double temp = 50.0;  // Default value
            if (f < surface_temps.size() && t < surface_temps[f].size()) {
                temp = surface_temps[f][t];
                if (std::isnan(temp) || std::isinf(temp)) {
                    temp = 50.0;  // Safety for visualization
                }
            }
            temp_data.push_back(std::to_string(temp));
        }
    }
    
    // Find the end of all existing CELL_DATA sections to insert our data
    int insert_pos = cell_data_line + 1;
    bool in_scalars = false;
    for (size_t i = cell_data_line + 1; i < content.size(); i++) {
        if (content[i].find("SCALARS") != std::string::npos) {
            in_scalars = true;
            continue;
        }
        
        if (in_scalars && content[i].find("LOOKUP_TABLE") != std::string::npos) {
            // Skip the next cell_count lines (the data values)
            i += cell_count;
            in_scalars = false;
            insert_pos = i + 1;  // Insert after this block
            continue;
        }
        
        // If we find a new section, stop
        if (content[i].find("POINT_DATA") != std::string::npos ||
            content[i].find("FIELD") != std::string::npos) {
            insert_pos = i;
            break;
        }
    }
    
    // Insert our temperature data at the proper position
    content.insert(content.begin() + insert_pos, temp_data.begin(), temp_data.end());
    
    // Write the modified file
    std::ofstream outfile(vtk_file);
    if (!outfile.is_open()) {
        std::cerr << "Failed to open output VTK file for writing" << std::endl;
        return false;
    }
    
    for (const auto& line : content) {
        outfile << line << std::endl;
    }
    
    outfile.close();
    std::cout << "Temperature data added to VTK file: " << vtk_file << std::endl;
    std::cout << "Thermal VTK file generated successfully" << std::endl;
    return true;
}

// Add checkpoint saving
void saveCheckpoint(
    const std::string& checkpoint_file,
    const std::vector<size_t>& completed_indices,
    const std::vector<std::vector<double>>& surface_temps) {
    
    std::ofstream out(checkpoint_file, std::ios::binary);
    if (!out.is_open()) {
        std::cerr << "Failed to save checkpoint" << std::endl;
        return;
    }
    
    // Write number of completed faces
    size_t count = completed_indices.size();
    out.write(reinterpret_cast<const char*>(&count), sizeof(count));
    
    // Write completed face indices
    for (size_t idx : completed_indices) {
        out.write(reinterpret_cast<const char*>(&idx), sizeof(idx));
        
        // Write temperature data for this face
        size_t temp_count = surface_temps[idx].size();
        out.write(reinterpret_cast<const char*>(&temp_count), sizeof(temp_count));
        
        out.write(reinterpret_cast<const char*>(surface_temps[idx].data()), 
                 temp_count * sizeof(double));
    }
    
    std::cout << "Checkpoint saved with " << count << " faces" << std::endl;
}

// Helper function to adjust start time
std::string calculateAdjustedStartTime(const std::string& iso_time, double days_offset) {
    // Parse the ISO time string to get seconds since J2000
    double et;
    str2et_c(iso_time.c_str(), &et);
    
    // Apply the offset in seconds
    double adjusted_et = et + days_offset * 86400.0;
    
    // Convert back to ISO format
    char time_string[51];
    et2utc_c(adjusted_et, "ISOC", 0, 50, time_string);
    
    return std::string(time_string);
}

bool exportLocationIlluminationAndTemperature(
    const std::string& output_file,
    const std::string& locations_file,
    const std::shared_ptr<Mesh>& mesh,
    const std::vector<double>& illumination_values,
    const std::vector<double>& direct_solar_values,
    const std::vector<double>& primary_reflected_values,
    const std::vector<double>& primary_ir_values,
    const std::vector<std::vector<double>>& surface_temps,  // Add surface temperature data
    const std::vector<std::string>& time_strings,
    int time_steps) {
    
    // Reuse existing code to read locations and find facets
    std::cout << "Attempting to open locations file: " << locations_file << std::endl;
    std::ifstream input(locations_file);
    if (!input.is_open()) {
        std::cerr << "Failed to open locations file: " << locations_file << std::endl;
        return false;
    }
    
    // Parse locations
    std::vector<std::pair<double, double>> locations;
    std::vector<std::string> location_names;
    
    std::string line;
    while (std::getline(input, line)) {
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') {
            continue;
        }
        
        std::istringstream iss(line);
        double lat, lon;
        std::string name;
        
        if (!(iss >> lat >> lon)) {
            std::cerr << "Error parsing location: " << line << std::endl;
            continue;
        }
        
        // Get name if present, otherwise use coordinates
        std::getline(iss >> std::ws, name);
        if (name.empty()) {
            std::ostringstream name_ss;
            name_ss << "Location_" << lat << "_" << lon;
            name = name_ss.str();
        }
        
        locations.emplace_back(lat, lon);
        location_names.push_back(name);
    }
    
    // Find closest mesh facet for each location (same as before)
    std::vector<size_t> facet_indices(locations.size());
    for (size_t i = 0; i < locations.size(); i++) {
        const auto& location = locations[i];
        double min_distance = std::numeric_limits<double>::max();
        size_t closest_facet = 0;
        
        // Find the facet with the closest center to our location
        for (size_t f = 0; f < mesh->faces.size(); f++) {
            double facet_lat = mesh->latitudes[f];
            double facet_lon = mesh->longitudes[f];
            
            // Simple spherical distance calculation
            double dlat = facet_lat - location.first;
            double dlon = facet_lon - location.second;
            while (dlon > 180.0) dlon -= 360.0;
            while (dlon < -180.0) dlon += 360.0;
            
            double distance = std::sqrt(dlat*dlat + dlon*dlon);
            
            if (distance < min_distance) {
                min_distance = distance;
                closest_facet = f;
            }
        }
        
        facet_indices[i] = closest_facet;
        std::cout << "Location " << location_names[i] << " (" << location.first << ", " << location.second 
                  << ") mapped to facet " << closest_facet << " at (" 
                  << mesh->latitudes[closest_facet] << ", " << mesh->longitudes[closest_facet] << ")" << std::endl;
    }
    
    // Open output file
    std::ofstream output(output_file);
    if (!output.is_open()) {
        std::cerr << "Failed to open output file: " << output_file << std::endl;
        return false;
    }
    
    // Modified CSV header to include temperature
    output << "Time";
    for (const auto& name : location_names) {
        output << "," << name << "_Total";
        output << "," << name << "_DirectSolar";
        output << "," << name << "_PrimaryReflected";
        output << "," << name << "_PrimaryIR";
        output << "," << name << "_Temperature";  // Add temperature column
    }
    output << std::endl;
    
    // Write data including temperature
    for (int t = 0; t < time_steps; t++) {
        output << time_strings[t];
        
        for (size_t i = 0; i < locations.size(); i++) {
            size_t facet_idx = facet_indices[i];
            size_t idx = facet_idx * time_steps + t;
            
            // Write illumination data
            output << "," << illumination_values[idx];
            output << "," << direct_solar_values[idx];
            output << "," << primary_reflected_values[idx];
            output << "," << primary_ir_values[idx];
            
            // Add temperature data if available
            if (facet_idx < surface_temps.size() && t < surface_temps[facet_idx].size()) {
                output << "," << surface_temps[facet_idx][t];
            } else {
                output << ",50.0";  // Default temperature if not available
            }
        }
        output << std::endl;
    }
    
    std::cout << "Location illumination and temperature data written to " << output_file << std::endl;
    return true;
}

// Add this declaration to the function prototypes section
std::vector<double> runThermalModelWithMatchedTimes(
    const std::vector<double>& flux_values,
    const std::vector<double>& times,
    double latitude,
    const thermal::Config& config,
    int equilibration_steps);

// Write a complete VTK file with both illumination and thermal data
bool writeCompleteVTK(
    const std::string& output_file,
    const std::shared_ptr<Mesh>& mesh,
    const std::vector<double>& illumination_values,
    const std::vector<double>& primary_ir_values,
    const std::vector<std::vector<double>>& surface_temps,
    const std::vector<double>& times,
    const std::vector<std::string>& time_strings,
    int time_steps,
    const std::string& source_mesh_file,
    bool include_primary_ir = false) {
    
    std::ofstream file(output_file);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file for writing: " << output_file << std::endl;
        return false;
    }
    
    // Write VTK header with custom mesh info
    std::string vtk_header = "Planetary Thermal Simulation";
    if (!source_mesh_file.empty()) {
        vtk_header += " - Using custom mesh: " + 
            std::filesystem::path(source_mesh_file).filename().string();
    }
    
    file << "# vtk DataFile Version 3.0\n";
    file << vtk_header << "\n";
    file << "ASCII\n";
    file << "DATASET UNSTRUCTURED_GRID\n";
    
    // Write vertices
    file << "POINTS " << mesh->vertices.size() << " float\n";
    for (const auto& v : mesh->vertices) {
        file << v[0] << " " << v[1] << " " << v[2] << "\n";
    }
    
    // Write cells (faces)
    file << "CELLS " << mesh->faces.size() << " " << mesh->faces.size() * 4 << "\n";
    for (const auto& f : mesh->faces) {
        file << "3 " << f[0] << " " << f[1] << " " << f[2] << "\n";
    }
    
    // Write cell types (all triangles)
    file << "CELL_TYPES " << mesh->faces.size() << "\n";
    for (size_t i = 0; i < mesh->faces.size(); i++) {
        file << "5\n"; // VTK_TRIANGLE
    }
    
    // Write FIELD data for time values
    file << "FIELD FieldData 2\n";
    
    // Write time values
    file << "TIME_STEPS " << 1 << " " << time_steps << " float\n";
    for (int t = 0; t < time_steps; t++) {
        file << times[t] << "\n"; 
    }
    
    // Write time strings
    file << "TIME_STRINGS " << 1 << " " << time_steps << " string\n";
    for (int t = 0; t < time_steps; t++) {
        file << time_strings[t] << "\n";
    }
    
    // Write cell data
    file << "CELL_DATA " << mesh->faces.size() << "\n";
    
    // Add latitude/longitude data first (same as illumination file)
    file << "SCALARS latitude float 1\n";
    file << "LOOKUP_TABLE default\n";
    for (size_t f = 0; f < mesh->faces.size(); f++) {
        file << mesh->latitudes[f] << "\n";
    }
    
    file << "SCALARS longitude float 1\n";
    file << "LOOKUP_TABLE default\n";
    for (size_t f = 0; f < mesh->faces.size(); f++) {
        file << mesh->longitudes[f] << "\n";
    }

    // Write illumination data for each time step
    for (int t = 0; t < time_steps; t++) {
        // Total illumination
        file << "SCALARS illumination_" << t << " float 1\n";
        file << "LOOKUP_TABLE default\n";
        for (size_t f = 0; f < mesh->faces.size(); f++) {
            size_t idx = f * time_steps + t;
            if (idx < illumination_values.size()) {
                file << illumination_values[idx] << "\n";
            } else {
                file << "0.0\n";
            }
        }

        // Only write primary IR data if explicitly requested
        if (include_primary_ir) {
            file << "SCALARS primary_ir_" << t << " float 1\n";
            file << "LOOKUP_TABLE default\n";
            for (size_t f = 0; f < mesh->faces.size(); f++) {
                size_t idx = f * time_steps + t;
                if (idx < primary_ir_values.size()) {
                    file << primary_ir_values[idx] << "\n";
                } else {
                    file << "0.0\n";
                }
            }
        }
        
        // Surface temperatures (if available)
        if (!surface_temps.empty() && t < time_steps) {
            file << "SCALARS temperature_" << t << " float 1\n";
            file << "LOOKUP_TABLE default\n";
            for (size_t f = 0; f < mesh->faces.size(); f++) {
                if (f < surface_temps.size() && t < surface_temps[f].size()) {
                    file << surface_temps[f][t] << "\n";
                } else {
                    file << "0.0\n";
                }
            }
        }
    }
    
    file.close();
    std::cout << "Complete VTK file with " << time_steps << " time steps written to: " << output_file << std::endl;
    return true;
}

// Write temperature data to a separate VTK file in the same format as illumination data
bool writeThermalVTK(
    const std::string& output_file,
    const std::shared_ptr<Mesh>& mesh,
    const std::vector<std::vector<double>>& surface_temps,
    const std::vector<double>& times,
    const std::vector<std::string>& time_strings,
    int time_steps,
    bool include_equilibration = false) {

    // Determine actual number of time steps to write
    int start_idx = include_equilibration ? 0 : (times.size() - time_steps);
    int output_time_steps = include_equilibration ? times.size() : time_steps;
    
    // Status message - no verbose check, always print
    std::cout << "Generating thermal VTK file: " << output_file << std::endl;
    std::cout << "Writing " << output_time_steps << " time steps" 
              << (include_equilibration ? " (including equilibration)" : " (excluding equilibration)") << std::endl;
    
    std::ofstream file(output_file);
    if (!file.is_open()) {
        std::cerr << "Failed to open thermal output file: " << output_file << std::endl;
        return false;
    }
    
    // Write VTK header
    file << "# vtk DataFile Version 3.0\n";
    file << "Planetary Thermal Simulation\n";
    file << "ASCII\n";
    file << "DATASET UNSTRUCTURED_GRID\n";
    
    // Write vertices
    file << "POINTS " << mesh->vertices.size() << " float\n";
    for (const auto& v : mesh->vertices) {
        file << v[0] << " " << v[1] << " " << v[2] << "\n";
    }
    
    // Write faces
    file << "CELLS " << mesh->faces.size() << " " << mesh->faces.size() * 4 << "\n";
    for (const auto& f : mesh->faces) {
        file << "3 " << f[0] << " " << f[1] << " " << f[2] << "\n";
    }
    
    // Cell types (5 = VTK_TRIANGLE)
    file << "CELL_TYPES " << mesh->faces.size() << "\n";
    for (size_t i = 0; i < mesh->faces.size(); i++) {
        file << "5\n";
    }
    
    // Write FIELD data for time values
    file << "FIELD FieldData 2\n";
    
    // Write time values - use output_time_steps and start_idx for correct selection
    file << "TIME_STEPS " << 1 << " " << output_time_steps << " float\n";
    for (int t = 0; t < output_time_steps; t++) {
        file << times[start_idx + t] << "\n"; 
    }
    
    // Write time strings - use output_time_steps and start_idx for correct selection
    file << "TIME_STRINGS " << 1 << " " << output_time_steps << " string\n";
    for (int t = 0; t < output_time_steps; t++) {
        file << time_strings[start_idx + t] << "\n";
    }
    
    // Write cell data
    file << "CELL_DATA " << mesh->faces.size() << "\n";
    
    // Add latitude/longitude data first (same as illumination file)
    file << "SCALARS latitude float 1\n";
    file << "LOOKUP_TABLE default\n";
    for (size_t f = 0; f < mesh->faces.size(); f++) {
        file << mesh->latitudes[f] << "\n";
    }
    
    file << "SCALARS longitude float 1\n";
    file << "LOOKUP_TABLE default\n";
    for (size_t f = 0; f < mesh->faces.size(); f++) {
        file << mesh->longitudes[f] << "\n";
    }
    
    // For detection of temperature range
    double min_temp = std::numeric_limits<double>::max();
    double max_temp = std::numeric_limits<double>::lowest();
    int valid_temp_count = 0;
    int nan_inf_count = 0;
    
    // For each time step, write a temperature array - use output_time_steps and start_idx
    for (int t = 0; t < output_time_steps; t++) {
        // Calculate the actual time index using start_idx
        int time_idx = start_idx + t;
        
        // Temperature for this time step
        file << "SCALARS temperature_" << t << " float 1\n";
        file << "LOOKUP_TABLE default\n";
        for (size_t f = 0; f < mesh->faces.size(); f++) {
            // Use -1.0 as marker for invalid data
            double temp = -1.0;  
            
            if (f < surface_temps.size() && time_idx < surface_temps[f].size()) {
                temp = surface_temps[f][time_idx];
                
                // Only filter out non-numeric values
                if (std::isnan(temp) || std::isinf(temp)) {
                    temp = -1.0;
                    nan_inf_count++;
                } else {
                    // Track temperature statistics for valid temperatures only
                    min_temp = std::min(min_temp, temp);
                    max_temp = std::max(max_temp, temp);
                    valid_temp_count++;
                }
            }
            
            // Write temperature value
            file << std::fixed << std::setprecision(6) << temp << "\n";
        }
    }
    
    file.close();
    
    // Print diagnostic information about temperature data
    if (valid_temp_count > 0) {
        std::cout << "Temperature data statistics:" << std::endl;
        std::cout << "  Valid values: " << valid_temp_count << std::endl;
        std::cout << "  NaN/Inf values replaced: " << nan_inf_count << std::endl;
        std::cout << "  Temperature range: " << min_temp << "K to " << max_temp << "K" << std::endl;
    } else {
        std::cout << "WARNING: No valid temperature values were written to VTK!" << std::endl;
    }
    
    std::cout << "Temperature data written to: " << output_file << std::endl;
    std::cout << "Thermal VTK file generated successfully" << std::endl;
    return true;
}

// Fix unreasonable albedo values
void fixOutOfRangeAlbedos(
    std::vector<double>& face_albedos,
    const std::shared_ptr<Mesh>& mesh,
    double default_visible_albedo) {
    
    const size_t num_faces = face_albedos.size();
    if (num_faces == 0) return;
    
    // Define reasonable bounds for albedo
    const double min_reasonable_albedo = 0.01;  // Minimum reasonable albedo for body's surface
    const double max_reasonable_albedo = 0.99;  // Maximum reasonable albedo for body's surface
    
    // Identify faces with unreasonable albedo values
    std::vector<size_t> problematic_faces;
    for (size_t f = 0; f < num_faces; f++) {
        // Check if albedo is outside reasonable range or set to default
        if (face_albedos[f] == default_visible_albedo || 
            face_albedos[f] < min_reasonable_albedo || 
            face_albedos[f] > max_reasonable_albedo) {
            problematic_faces.push_back(f);
        }
    }
    
    if (problematic_faces.empty()) {
        std::cout << "No unreasonable albedo values found." << std::endl;
        return;
    }
    
    std::cout << "Found " << problematic_faces.size() << " faces with unreasonable albedo values ("
              << (100.0 * problematic_faces.size() / num_faces) << "%)." << std::endl;
    
    // Fix each problematic face using valid neighbors
    int fixed_count = 0;
    
    #pragma omp parallel for reduction(+:fixed_count)
    for (size_t i = 0; i < problematic_faces.size(); i++) {
        size_t f = problematic_faces[i];
        
        // Find valid neighbors with reasonable albedos
        std::vector<std::pair<size_t, double>> valid_neighbors; // face_id, distance
        
        for (size_t other_f = 0; other_f < num_faces; other_f++) {
            // Skip if it's the same face or another problematic face
            if (other_f == f || 
                face_albedos[other_f] == default_visible_albedo || 
                face_albedos[other_f] < min_reasonable_albedo || 
                face_albedos[other_f] > max_reasonable_albedo) {
                continue;
            }
            
            // Calculate spherical distance between face centers
            double dlat = mesh->latitudes[f] - mesh->latitudes[other_f];
            double dlon = mesh->longitudes[f] - mesh->longitudes[other_f];
            
            // Handle longitude wrapping
            if (dlon > 180.0) dlon -= 360.0;
            if (dlon < -180.0) dlon += 360.0;
            
            // Simple spherical distance approximation
            double dist = std::sqrt(dlat*dlat + dlon*dlon);
            
            // Only consider neighbors within a reasonable distance (30 degrees)
            if (dist <= 30.0) {
                valid_neighbors.push_back({other_f, dist});
            }
        }
        
        // If we found valid neighbors, compute weighted average
        if (!valid_neighbors.empty()) {
            // Sort by distance (closest first)
            std::sort(valid_neighbors.begin(), valid_neighbors.end(), 
                    [](const auto& a, const auto& b) { return a.second < b.second; });
            
            // Take up to 12 closest neighbors
            const size_t max_neighbors = 12;
            if (valid_neighbors.size() > max_neighbors) {
                valid_neighbors.resize(max_neighbors);
            }
            
            // Calculate inverse-distance weighted average
            double sum_weights = 0.0;
            double weighted_sum = 0.0;
            
            for (const auto& [neighbor_f, dist] : valid_neighbors) {
                // Add a small epsilon to avoid division by zero
                double weight = 1.0 / (dist + 0.001);
                sum_weights += weight;
                weighted_sum += weight * face_albedos[neighbor_f];
            }
            
            // Update albedo with weighted average
            face_albedos[f] = weighted_sum / sum_weights;
            fixed_count++;
        }
    }
    
    // Report results
    std::cout << "Fixed " << fixed_count << " faces with unreasonable albedo values ("
              << (100.0 * fixed_count / problematic_faces.size()) << "% of problematic faces)." << std::endl;
    
    // Check if any problematic faces remain
    int remaining_bad = problematic_faces.size() - fixed_count;
    if (remaining_bad > 0) {
        std::cout << "Warning: " << remaining_bad << " faces still have unreasonable albedo values." << std::endl;
        std::cout << "Consider increasing the search radius or using a different fixing strategy." << std::endl;
    }
}

// Add this function to your declarations section
bool validateMesh(const std::shared_ptr<ptm::Mesh>& mesh);

// Add the implementation at an appropriate location
bool validateMesh(const std::shared_ptr<ptm::Mesh>& mesh) {
    if (mesh->vertices.empty() || mesh->faces.empty()) {
        std::cerr << "Error: Mesh has no vertices or faces" << std::endl;
        return false;
    }
    
    // Check if face normals were calculated
    if (mesh->face_normals.size() != mesh->faces.size()) {
        std::cerr << "Error: Face normals were not properly calculated" << std::endl;
        return false;
    }
    
    // Check if lat/lon mappings were calculated
    if (mesh->latitudes.size() != mesh->faces.size() || 
        mesh->longitudes.size() != mesh->faces.size()) {
        std::cerr << "Error: Latitude/longitude mappings were not properly calculated" << std::endl;
        return false;
    }
    
    // Mesh appears valid
    return true;
}

// In your function that displays usage information
void showUsage(const char* programName) {
    std::cout << "Usage: " << programName << " [options]\n\n";
    
    std::cout << "Time and Simulation Options:\n";
    std::cout << "  --body, -b <name>      Set the planetary body (EUROPA, MOON, MERCURY, etc.)" << std::endl;
    std::cout << "  --start TIME         : Start time in ISO format (default: 2025-01-01T00:00:00)\n";
    std::cout << "  --duration DAYS      : Simulation duration in days (default: 3.55)\n";
    std::cout << "  --timesteps N        : Number of time steps (default: 72)\n";
    
    std::cout << "\nMesh Options:\n";
    std::cout << "  --mesh-resolution <n> : Set mesh resolution (default: 5)\n";
    std::cout << "  --mesh-file <file>    : Import triangulated mesh from file (OBJ, PLY, VTK)\n";
    std::cout << "  --mesh-scale <factor> : Scale factor for imported mesh (default: 1.0)\n";
    std::cout << "  --export-basic-mesh   : Export basic mesh to VTK format\n";
    
    std::cout << "\nThermal Model Options:\n";
    std::cout << "  --thermal-mesh        : Run thermal model for all mesh faces\n";
    std::cout << "  --thermal-threads N   : Number of parallel threads for thermal calculation (default: auto)\n";
    std::cout << "  --equilibration-orbits N : Number of orbits for thermal equilibration (default: 10)\n";
    
    std::cout << "\nIllumination Options:\n";
    std::cout << "  --no-primary-reflection : Disable reflected solar radiation from primary body\n";
    std::cout << "  --albedo-map FILE     : Path to albedo map file\n";
    std::cout << "  --thermal-inertia-map FILE : Path to thermal inertia map file\n";
    std::cout << "  --albedo-maps-directory DIR : Path to albedo maps directory (default: data/albedo_maps)\n";
    std::cout << "  --no-batch            : Disable SIMD batch processing for illumination\n";
    
    std::cout << "\nOutput Options:\n";
    std::cout << "  --output DIR          : Output directory (default: ./output)\n";
    std::cout << "  --output-format FORMAT  : Output format: 'binary' (default) or 'vtk'" << std::endl;
    std::cout << "  --include-primary-ir    : Include primary IR data in VTK files (increases file size)" << std::endl;
    std::cout << "  --vtk-mode MODE        : VTK output mode: 'both' (default), 'thermal', or 'complete'" << std::endl;
    std::cout << "  --thermal-vtk-file FILE  : Custom filename for thermal VTK output\n";
    std::cout << "  --complete-vtk-file FILE : Custom filename for complete VTK output\n";    
    std::cout << "  --positions FILE      : Export body positions to specified file\n";
    std::cout << "  --locations FILE      : Extract illumination for specific lat/lon points\n";
    std::cout << "  --location-data FILE  : Specify custom location data file for visualization\n";  // Add this line
    std::cout << "  --series-output FILE  : Output file for location illumination data (default: illumination_series.csv)\n";
    
    std::cout << "\nVisualization Options:\n";
    std::cout << "  --visualization       : Create visualization of the results\n";
    std::cout << "  --camera-distance N   : Camera distance for visualization (default: 5.0)\n";
    std::cout << "  --camera-lat N        : Camera latitude for visualization (default: 30.0)\n";
    std::cout << "  --camera-lon N        : Camera longitude for visualization (default: 0.0)\n";
    std::cout << "  --emission-angle N    : Alternative to camera position - set emission angle\n";
    std::cout << "  --azimuth N           : Used with emission-angle - set azimuth angle\n";
    std::cout << "  --animation-width N   : Set animation width in pixels (default: 1024)\n";
    std::cout << "  --animation-height N  : Set animation height in pixels (default: 768)\n";
    std::cout << "  --no-maps             : Don't generate cylindrical map projections\n";
    std::cout << "  --save-frames         : Save individual frames during visualization\n";
    
    std::cout << "\nOther Options:\n";
    std::cout << "  --verbose             : Enable verbose output\n";
    std::cout << "  --help                : Display this help message and exit\n";
}

std::string createDefaultLocationsFile(const std::string& output_dir) {
    // Ensure output directory exists
    try {
        if (!std::filesystem::exists(output_dir)) {
            std::filesystem::create_directories(output_dir);
        }
        
        // Create a default locations file with some useful points
        std::string filename = output_dir + "/default_locations.txt";
        std::ofstream file(filename);
        if (!file.is_open()) {
            return "";
        }
        
        // Format: latitude longitude location_name (space-separated)
        file << "# lat lon name\n";
        file << "0.0 0.0 Sub_Jupiter\n";      // Equator at 0° longitude (sub-Jupiter point)
        file << "0.0 90.0 Trailing\n";        // Equator at 90°E (trailing hemisphere)
        file << "0.0 180.0 Anti_Jupiter\n";   // Equator at 180° (anti-Jupiter point)
        file << "0.0 270.0 Leading\n";        // Equator at 270°E (leading hemisphere)
        file << "45.0 0.0 North_45\n";        // 45°N at 0° longitude
        file << "-45.0 0.0 South_45\n";       // 45°S at 0° longitude
        file << "85.0 0.0 North_Pole\n";      // Near North Pole
        file << "-85.0 0.0 South_Pole\n";     // Near South Pole
        
        file.close();
        return filename;
    } catch (const std::exception& e) {
        std::cerr << "Error creating default locations file: " << e.what() << std::endl;
        return "";
    }
}

std::string generateSimulationFilename(
    const std::string& output_dir,
    const std::string& body_name,
    const std::string& start_time,
    const std::string& suffix) {
    
    // Clean the start time to be filesystem-friendly
    std::string clean_time = start_time;
    std::replace(clean_time.begin(), clean_time.end(), ':', '-');
    std::replace(clean_time.begin(), clean_time.end(), ' ', '_');
    std::replace(clean_time.begin(), clean_time.end(), 'T', '_');
    
    // Trim to a reasonable length (first 10 chars)
    if (clean_time.length() > 10) {
        clean_time = clean_time.substr(0, 10);
    }
    
    // Format: output_dir/ptm_BODY_STARTTIME_suffix.vtk
    return output_dir + "/ptm_" + body_name + "_" + clean_time + "_" + suffix + ".vtk";
}

// Function to write binary data for a single time step
bool writeBinaryTimeStepFile(
    const std::string& filename,
    const std::vector<float>& data) {
    
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file for writing: " << filename << std::endl;
        return false;
    }
    
    // Write number of elements as header (4-byte uint32)
    uint32_t num_elements = static_cast<uint32_t>(data.size());
    file.write(reinterpret_cast<char*>(&num_elements), sizeof(uint32_t));
    
    // Write data as 32-bit floats
    file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
    
    return file.good();
}

// Function to write time steps metadata file
bool writeTimeStepsFile(
    const std::string& filename,
    const std::vector<double>& times,
    const std::vector<std::string>& time_strings) {
    
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open time steps file for writing: " << filename << std::endl;
        return false;
    }
    
    // Write header
    file << "# Time steps metadata\n";
    file << "# Column 1: Time elapsed from start (seconds)\n";
    file << "# Column 2: UTC time\n";
    
    // Write time data
    for (size_t i = 0; i < times.size(); i++) {
        file << std::fixed << std::setprecision(1) << times[i] << "\t" 
             << time_strings[i] << std::endl;
    }
    
    return file.good();
}

// Function to export mesh to binary PLY format
bool exportMeshToPLY(
    const std::string& filename,
    const std::shared_ptr<Mesh>& mesh) {
    
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open PLY file for writing: " << filename << std::endl;
        return false;
    }
    
    // Write PLY header (ASCII part)
    file << "ply\n";
    file << "format binary_little_endian 1.0\n";
    file << "element vertex " << mesh->vertices.size() << "\n";
    file << "property float x\n";
    file << "property float y\n";
    file << "property float z\n";
    file << "element face " << mesh->faces.size() << "\n";
    file << "property list uchar int vertex_indices\n";
    file << "property float nx\n";
    file << "property float ny\n";
    file << "property float nz\n";
    file << "property float latitude\n";
    file << "property float longitude\n";
    file << "end_header\n";
    
    // Write vertex data
    for (const auto& vertex : mesh->vertices) {
        float x = static_cast<float>(vertex[0]);
        float y = static_cast<float>(vertex[1]);
        float z = static_cast<float>(vertex[2]);
        file.write(reinterpret_cast<char*>(&x), sizeof(float));
        file.write(reinterpret_cast<char*>(&y), sizeof(float));
        file.write(reinterpret_cast<char*>(&z), sizeof(float));
    }
    
    // Write face data
    for (size_t i = 0; i < mesh->faces.size(); i++) {
        const auto& face = mesh->faces[i];
        const auto& normal = mesh->face_normals[i];
        
        // Number of vertices in the face (always 3 for triangles)
        unsigned char num_vertices = 3;
        file.write(reinterpret_cast<char*>(&num_vertices), sizeof(unsigned char));
        
        // Vertex indices
        int v1 = face[0];
        int v2 = face[1];
        int v3 = face[2];
        file.write(reinterpret_cast<char*>(&v1), sizeof(int));
        file.write(reinterpret_cast<char*>(&v2), sizeof(int));
        file.write(reinterpret_cast<char*>(&v3), sizeof(int));
        
        // Face normal
        float nx = static_cast<float>(normal[0]);
        float ny = static_cast<float>(normal[1]);
        float nz = static_cast<float>(normal[2]);
        file.write(reinterpret_cast<char*>(&nx), sizeof(float));
        file.write(reinterpret_cast<char*>(&ny), sizeof(float));
        file.write(reinterpret_cast<char*>(&nz), sizeof(float));
        
        // Latitude and longitude
        float lat = static_cast<float>(mesh->latitudes[i]);
        float lon = static_cast<float>(mesh->longitudes[i]);
        file.write(reinterpret_cast<char*>(&lat), sizeof(float));
        file.write(reinterpret_cast<char*>(&lon), sizeof(float));
    }
    
    return file.good();
}

bool exportDataAsBinary(
    const std::string& output_dir,
    const std::shared_ptr<Mesh>& mesh,
    const std::vector<double>& illumination_values,
    const std::vector<double>& primary_ir_values,
    const std::vector<std::vector<double>>& surface_temps,
    const std::vector<double>& times,
    const std::vector<std::string>& time_strings,
    int time_steps,
    bool is_custom_mesh,
    bool include_equilibration = false) {
    
    // Determine actual number of time steps to write
    int start_idx = include_equilibration ? 0 : (times.size() - time_steps);
    int output_time_steps = include_equilibration ? times.size() : time_steps;
    
    // Status message
    std::cout << "Generating binary output in: " << output_dir << std::endl;
    std::cout << "Writing " << output_time_steps << " time steps" 
              << (include_equilibration ? " (including equilibration)" : " (excluding equilibration)") << std::endl;
    
    // Calculate total number of files to be written for progress tracking
    int total_files = 2; // Mesh and time steps files
    total_files += output_time_steps * 3; // Illumination, primary IR, and temperature files for each time step
    
    // Create progress bar for file writing
    thermal::ProgressBar progressBar(total_files, "Writing output files", 50, true);
    int files_written = 0;
    
    // Create output directories
    std::string illum_dir = output_dir + "/illumination";
    std::string ir_dir = output_dir + "/primary_ir";
    std::string temp_dir = output_dir + "/temperature";
    
    std::filesystem::create_directories(illum_dir);
    std::filesystem::create_directories(ir_dir);
    std::filesystem::create_directories(temp_dir);
    
    // Export mesh if using global model (not custom mesh)
    if (!is_custom_mesh) {
        std::string mesh_file = output_dir + "/mesh.ply";
        std::cout << "Exporting mesh to " << mesh_file << std::endl;
        if (!exportMeshToPLY(mesh_file, mesh)) {
            std::cerr << "Failed to export mesh" << std::endl;
            return false;
        }
    }
    progressBar.update(++files_written);
    
    // Always export mesh in VTK-compatible format for the converter
    std::string vtk_mesh_file = output_dir + "/mesh.ply";
    if (is_custom_mesh || !std::filesystem::exists(vtk_mesh_file)) {
        std::cout << "Exporting mesh in VTK-compatible format: " << vtk_mesh_file << std::endl;
        
        try {
            std::ofstream file(vtk_mesh_file, std::ios::binary);
            if (!file.is_open()) {
                std::cerr << "Failed to open file for writing: " << vtk_mesh_file << std::endl;
                // Continue even if this fails - it's just for visualization
            }
            else {
                // Write header (8 bytes of identifier)
                const char* header = "EUROPMSH";
                file.write(header, 8);
                
                // Write vertex count (4-byte integer)
                int vertex_count = static_cast<int>(mesh->vertices.size());
                file.write(reinterpret_cast<const char*>(&vertex_count), sizeof(int));
                
                // Write vertices (each vertex is 3 floats = 12 bytes)
                for (const auto& vertex : mesh->vertices) {
                    float x = static_cast<float>(vertex[0]);
                    float y = static_cast<float>(vertex[1]);
                    float z = static_cast<float>(vertex[2]);
                    file.write(reinterpret_cast<const char*>(&x), sizeof(float));
                    file.write(reinterpret_cast<const char*>(&y), sizeof(float));
                    file.write(reinterpret_cast<const char*>(&z), sizeof(float));
                }
                
                // Write face count (4-byte integer)
                int face_count = static_cast<int>(mesh->faces.size());
                file.write(reinterpret_cast<const char*>(&face_count), sizeof(int));
                
                // Write faces (each face is 3 integers = 12 bytes)
                for (const auto& face : mesh->faces) {
                    int v1 = face[0];
                    int v2 = face[1];
                    int v3 = face[2];
                    file.write(reinterpret_cast<const char*>(&v1), sizeof(int));
                    file.write(reinterpret_cast<const char*>(&v2), sizeof(int));
                    file.write(reinterpret_cast<const char*>(&v3), sizeof(int));
                }
                
                // Write additional data needed by vtk_converter.py
                // Write normals
                for (const auto& normal : mesh->face_normals) {
                    float nx = static_cast<float>(normal[0]);
                    float ny = static_cast<float>(normal[1]);
                    float nz = static_cast<float>(normal[2]);
                    file.write(reinterpret_cast<const char*>(&nx), sizeof(float));
                    file.write(reinterpret_cast<const char*>(&ny), sizeof(float));
                    file.write(reinterpret_cast<const char*>(&nz), sizeof(float));
                }
                
                // Write latitudes and longitudes
                for (const auto& lat : mesh->latitudes) {
                    float lat_f = static_cast<float>(lat);
                    file.write(reinterpret_cast<const char*>(&lat_f), sizeof(float));
                }
                
                for (const auto& lon : mesh->longitudes) {
                    float lon_f = static_cast<float>(lon);
                    file.write(reinterpret_cast<const char*>(&lon_f), sizeof(float));
                }
                
                file.close();
                std::cout << "Mesh exported successfully in VTK-compatible format" << std::endl;
            }
        }
        catch (const std::exception& e) {
            std::cerr << "Error exporting mesh: " << e.what() << std::endl;
            // Continue even if this fails - it's just for visualization
        }
    }
    
    // Export filtered time steps metadata
    std::string time_file = output_dir + "/time_steps.txt";
    std::cout << "Writing time steps metadata to " << time_file << std::endl;
    
    // Use only the relevant time steps
    std::vector<double> filtered_times(times.begin() + start_idx, times.begin() + start_idx + output_time_steps);
    std::vector<std::string> filtered_time_strings(time_strings.begin() + start_idx, 
                                                 time_strings.begin() + start_idx + output_time_steps);
    
    if (!writeTimeStepsFile(time_file, filtered_times, filtered_time_strings)) {
        std::cerr << "Failed to write time steps file" << std::endl;
        return false;
    }
    progressBar.update(++files_written);
    
    // Get number of faces
    size_t num_faces = mesh->faces.size();
    
    // Process each time step
    for (int t = 0; t < output_time_steps; t++) {
        // Format time step number with leading zeros
        std::ostringstream time_str;
        time_str << std::setw(4) << std::setfill('0') << (t + 1);
        std::string time_step_str = time_str.str();
        
        // Get the actual time index
        int actual_time_idx = start_idx + t;
        
        // Convert illumination data to float
        std::vector<float> illum_data(num_faces);
        for (size_t f = 0; f < num_faces; f++) {
            size_t idx = f * times.size() + actual_time_idx;
            if (idx < illumination_values.size()) {
                illum_data[f] = static_cast<float>(illumination_values[idx]);
            } else {
                illum_data[f] = 0.0f;
            }
        }
        
        // Export illumination data
        std::string illum_file = illum_dir + "/illumination_" + time_step_str + ".bin";
        if (!writeBinaryTimeStepFile(illum_file, illum_data)) {
            std::cerr << "Failed to write illumination data for time step " << t << std::endl;
            return false;
        }
        progressBar.update(++files_written);
        
        // Convert primary IR data to float
        std::vector<float> ir_data(num_faces);
        for (size_t f = 0; f < num_faces; f++) {
            size_t idx = f * times.size() + actual_time_idx;
            if (idx < primary_ir_values.size()) {
                ir_data[f] = static_cast<float>(primary_ir_values[idx]);
            } else {
                ir_data[f] = 0.0f;
            }
        }
        
        // Export primary IR data
        std::string ir_file = ir_dir + "/primary_ir_" + time_step_str + ".bin";
        if (!writeBinaryTimeStepFile(ir_file, ir_data)) {
            std::cerr << "Failed to write primary IR data for time step " << t << std::endl;
            return false;
        }
        progressBar.update(++files_written);
        
        // Convert temperature data to float
        std::vector<float> temp_data(num_faces);
        for (size_t f = 0; f < num_faces; f++) {
            if (f < surface_temps.size() && actual_time_idx < surface_temps[f].size()) {
                temp_data[f] = static_cast<float>(surface_temps[f][actual_time_idx]);
            } else {
                temp_data[f] = 0.0f;
            }
        }
        
        // Export temperature data
        std::string temp_file = temp_dir + "/temperature_" + time_step_str + ".bin";
        if (!writeBinaryTimeStepFile(temp_file, temp_data)) {
            std::cerr << "Failed to write temperature data for time step " << t << std::endl;
            return false;
        }
        progressBar.update(++files_written);
    }
    
    // Ensure progress bar is completed
    progressBar.finish();
    
    std::cout << "Successfully exported all data in binary format" << std::endl;
    return true;
}

int main(int argc, char** argv) {
    std::cout << "Planetary Thermal-Illumination Coupled Model" << std::endl;

    // Simulation parameters
    std::string start_time = "2025-01-01T00:00:00";
    double duration_days = 3.55; // Body's orbital period
    int mesh_resolution = 4;     // Resolution level for icosphere
    int time_steps = 72;         // Number of time steps (every ~1.2 hours)
    std::string output_dir = "./output";
    std::string positions_file = "";
    bool use_batch_processing = true;
    bool export_locations = false;
    std::string locations_file = "";
    std::string series_output = "illumination_series.csv";
    std::string location_plots_dir = ""; // Directory for location plots
    bool create_visualization = false;
    bool include_primary_reflection = true;  // Default to true (include reflection)
    bool no_maps = false;  
    bool save_frames = false;
    bool include_primary_ir_in_vtk = false;  // Default to excluding primary IR data
    std::string vtk_output_mode = "both";    // Default to both thermal and complete VTK files
    std::string output_format = "binary";  // Default to binary format for better performance and smaller files

    // Create the thermal config instance
    thermal::Config config;

    // Add mesh input options 
    std::string mesh_file = "";
    double mesh_scale_factor = 1.0;
    
    // Parameter to control verbosity 
    bool verbose = false;

    // Thermal model parameters
    bool run_thermal_mesh = false;  // Flag to run thermal model for entire mesh

    // Thermal multiprocessing parameters
    int thermal_threads = 0;  // 0 means use OMP default (usually all cores)

    // Command-line parameter for equilibration orbits
    int equilibration_orbits = 10;  // Default to 10 Body orbits for equilibration

    // Command line options for maps
    std::string albedo_map_file = "";
    std::string thermal_inertia_map_file = "";
    std::string albedo_maps_directory = "data/albedo_maps";

    // VTK files
    std::string thermal_vtk_file = "";    // Custom thermal VTK filename
    std::string complete_vtk_file = "";   // Custom complete VTK filename

    // Camera settings
    double camera_distance = 5.0;
    double camera_latitude = 30.0;
    double camera_longitude = 0.0;
    double emission_angle = -1.0;  // -1 means not specified
    double azimuth = -1.0;         // -1 means not specified

    // New command line option:
    bool export_basic_mesh = false;

    // Add animation width and height parameters
    int animation_width = 1024;  // Default width
    int animation_height = 768;  // Default height

    thermal::Planet planet("Europa");  // Default to Europa
    std::shared_ptr<ptm::IlluminationModel> illumination_model;

    // Add location data file argument
    std::string location_data_arg = "";

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--start" && i + 1 < argc) {
            start_time = argv[++i];
        } else if (arg == "--duration" && i + 1 < argc) {
            duration_days = std::stod(argv[++i]);
        } else if (arg == "--resolution" && i + 1 < argc) {
            mesh_resolution = std::stoi(argv[++i]);
        } else if (arg == "--timesteps" && i + 1 < argc) {
            time_steps = std::stoi(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            output_dir = argv[++i];
        } else if (arg == "--include-primary-ir") {
            include_primary_ir_in_vtk = true;
            std::cout << "Including primary IR data in VTK files" << std::endl;
        } else if (arg == "--vtk-mode" && i + 1 < argc) {
            vtk_output_mode = argv[++i];
            // Validate the mode
            if (vtk_output_mode != "both" && vtk_output_mode != "thermal" && vtk_output_mode != "complete") {
                std::cerr << "Error: vtk-mode must be 'both', 'thermal', or 'complete'" << std::endl;
                showUsage(argv[0]);
                return 1;
            }
            std::cout << "VTK output mode set to: " << vtk_output_mode << std::endl;
        } else if (arg == "--output-format" && i + 1 < argc) {
            std::string format = argv[++i];
            if (format == "binary" || format == "vtk") {
                output_format = format;
                std::cout << "Output format set to: " << output_format << std::endl;
            } else {
                std::cerr << "Error: output-format must be 'binary' or 'vtk'" << std::endl;
                showUsage(argv[0]);
                return 1;
            }
        } else if (arg == "--thermal-vtk-file" && i + 1 < argc) {
            thermal_vtk_file = argv[++i];
            std::cout << "Using custom thermal VTK file: " << thermal_vtk_file << std::endl;
        } else if (arg == "--complete-vtk-file" && i + 1 < argc) {
            complete_vtk_file = argv[++i];
            std::cout << "Using custom complete VTK file: " << complete_vtk_file << std::endl;
        } else if (arg == "--positions" && i + 1 < argc) {
            positions_file = argv[++i];
        } else if (arg == "--no-batch") {
            use_batch_processing = false;
        } else if (arg == "--locations" && i + 1 < argc) {
            export_locations = true;
            locations_file = argv[++i];
        } else if (arg == "--series-output" && i + 1 < argc) {
            series_output = argv[++i];
        } else if (arg == "--visualization") {
            create_visualization = true;
        } else if (arg == "--no-primary-reflection") {
            include_primary_reflection = false;
        } else if (arg == "--no-maps") {
            no_maps = true;
        } else if (arg == "--save-frames") {
            save_frames = true;
        } else if (arg == "--verbose") {
            verbose = true;
        } else if (arg == "--thermal-mesh") {
            run_thermal_mesh = true;
        } else if (arg == "--thermal-threads" && i + 1 < argc) {
            thermal_threads = std::stoi(argv[++i]);
        } else if (arg == "--equilibration-orbits" && i + 1 < argc) {
            equilibration_orbits = std::stoi(argv[++i]);
        } else if (arg == "--albedo-map" && i + 1 < argc) {
            albedo_map_file = argv[++i];
            continue;
        } else if (arg == "--thermal-inertia-map" && i + 1 < argc) {
            thermal_inertia_map_file = argv[++i];
            continue;
        } else if (arg == "--albedo-maps-directory" && i + 1 < argc) {
            albedo_maps_directory = argv[++i];
            std::cout << "Using albedo maps directory: " << albedo_maps_directory << std::endl;
        } else if (arg == "--mesh-file" && i + 1 < argc) {
            mesh_file = argv[++i];
        } else if (arg == "--mesh-scale" && i + 1 < argc) {
            mesh_scale_factor = std::stod(argv[++i]);
        } else if (arg == "--emission-angle" && i + 1 < argc) {
            emission_angle = std::stod(argv[++i]);
            std::cout << "Setting emission angle to: " << emission_angle << "°" << std::endl;
        } else if (arg == "--azimuth" && i + 1 < argc) {
            azimuth = std::stod(argv[++i]);
            std::cout << "Setting azimuth angle to: " << azimuth << "°" << std::endl;
        } else if (arg == "--export-basic-mesh") {
            export_basic_mesh = true;
        } else if (arg == "--animation-width" && i + 1 < argc) {
            animation_width = std::stoi(argv[++i]);
            std::cout << "Setting animation width to: " << animation_width << " pixels" << std::endl;
        } else if (arg == "--animation-height" && i + 1 < argc) {
            animation_height = std::stoi(argv[++i]);
            std::cout << "Setting animation height to: " << animation_height << " pixels" << std::endl;
        } else if (arg == "--camera-lat" && i + 1 < argc) {
            camera_latitude = std::stod(argv[++i]);
            std::cout << "Setting camera latitude to: " << camera_latitude << " degrees" << std::endl;
        } else if (arg == "--camera-lon" && i + 1 < argc) {
            camera_longitude = std::stod(argv[++i]);
            std::cout << "Setting camera longitude to: " << camera_longitude << " degrees" << std::endl;
        } else if (arg == "--location-data" && i + 1 < argc) {
            location_data_arg = argv[++i];
        } else if (arg == "--body" || arg == "-b") {
            if (i + 1 < argc) {
                std::string body_name = argv[++i];
                std::transform(body_name.begin(), body_name.end(), body_name.begin(), ::toupper);
                
                // Set the planetary body in config for thermal properties
                thermal::PlanetaryBody body_enum = thermal::PlanetaryBody::EUROPA; // Default
                
                // Map string to enum
                if (body_name == "MOON") body_enum = thermal::PlanetaryBody::MOON;
                else if (body_name == "MERCURY") body_enum = thermal::PlanetaryBody::MERCURY;
                else if (body_name == "EUROPA") body_enum = thermal::PlanetaryBody::EUROPA;
                else if (body_name == "ENCELADUS") body_enum = thermal::PlanetaryBody::ENCELADUS;
                else if (body_name == "TITAN") body_enum = thermal::PlanetaryBody::TITAN;
                else if (body_name == "MARS") body_enum = thermal::PlanetaryBody::MARS;
                else if (body_name == "CERES") body_enum = thermal::PlanetaryBody::CERES;
                else {
                    std::cerr << "Warning: Unrecognized body name '" << body_name 
                            << "'. Using Europa as default." << std::endl;
                }
                
                // Update both thermal model planet and illumination model settings
                planet = thermal::Planet(body_enum);
                
                // Create illumination model instance if not created yet
                if (!illumination_model) {
                    illumination_model = std::make_shared<ptm::IlluminationModel>();
                }
                
                // Set the planetary body in the illumination model
                illumination_model->setPlanetaryBody(body_enum);
                config.setProperty("target_body_name", illumination_model->getTargetBodyName());
                config.setProperty("default_albedo", std::to_string(illumination_model->getAlbedo()));

                std::cerr << "Using planetary body: " << body_name << std::endl;
                std::cerr << "  Target: " << illumination_model->getTargetBodyName() << std::endl;
                std::cerr << "  Primary: " << illumination_model->getPrimaryBodyName() << std::endl;
                std::cerr << "  Albedo: " << illumination_model->getAlbedo() << std::endl;
            } else {
                std::cerr << "Error: --body requires a parameter" << std::endl;
                showUsage(argv[0]);
                return 1;
            }
        } else if (arg == "--help") {
            showUsage(argv[0]);
            return 0;
        }
    }
    
    // Create output directory if it doesn't exist
    if (!std::filesystem::exists(output_dir)) {
        std::filesystem::create_directories(output_dir);
    }
    
    // Start timing
    auto start_time_point = std::chrono::high_resolution_clock::now();
    
    // Create mesh
    ptm::MeshUtils meshUtils;

    if (!mesh_file.empty()) {
        std::cout << "Importing triangulated mesh from: " << mesh_file << std::endl;
        if (!meshUtils.importTriangulatedMesh(mesh_file, mesh_scale_factor)) {
            std::cerr << "Failed to import mesh. Using default icosphere instead." << std::endl;
            meshUtils.setMeshResolution(mesh_resolution);
        } else {
            std::cout << "Successfully imported mesh with " << meshUtils.getMesh()->vertices.size() 
                      << " vertices and " << meshUtils.getMesh()->faces.size() << " faces" << std::endl;
            
            // If scale factor was provided and is not 1.0, add a note about it
            if (mesh_scale_factor != 1.0) {
                std::cout << "Applied scale factor: " << mesh_scale_factor << std::endl;
            }
            
            // Validate the mesh
            if (!validateMesh(meshUtils.getMesh())) {
                std::cerr << "Warning: Imported mesh has issues that may affect simulation quality." << std::endl;
            }
        }
    } else {
        // Use the standard icosphere mesh
        meshUtils.setMeshResolution(mesh_resolution);
    }

    // Rest of main() stays the same, using meshUtils.getMesh() to access the mesh
    auto mesh = meshUtils.getMesh();
    
    // Validate the mesh
    if (!validateMesh(mesh)) {
        std::cerr << "Mesh validation failed. Exiting." << std::endl;
        return 1;
    }
    
    // Export mesh to VTK for visualization
    if (export_basic_mesh) {
        std::string mesh_filename = output_dir + "/ptm_mesh.vtk";
        std::cout << "Exporting mesh to VTK: " << mesh_filename << std::endl;
        if (!meshUtils.exportMeshToVTK(mesh_filename)) {
            std::cerr << "Failed to export mesh to VTK" << std::endl;
        } else {
            std::cout << "Mesh exported to " << mesh_filename << std::endl;
        }
    }
    
    // Initialize illumination model
    std::cout << "Setting up illumination model..." << std::endl;
    if (!illumination_model) {
        illumination_model = std::make_shared<ptm::IlluminationModel>();
    }
    // If body was specified via command line, it's already set
    // Otherwise, initialize using the thermal::Planet that was created
    if (planet.name != "Custom") {
        // Determine body enum from name
        thermal::PlanetaryBody body_enum = thermal::PlanetaryBody::EUROPA; // Default
        
        if (planet.name == "Moon") body_enum = thermal::PlanetaryBody::MOON;
        else if (planet.name == "Mercury") body_enum = thermal::PlanetaryBody::MERCURY;
        else if (planet.name == "Europa") body_enum = thermal::PlanetaryBody::EUROPA;
        else if (planet.name == "Enceladus") body_enum = thermal::PlanetaryBody::ENCELADUS;
        else if (planet.name == "Titan") body_enum = thermal::PlanetaryBody::TITAN;
        else if (planet.name == "Mars") body_enum = thermal::PlanetaryBody::MARS;
        else if (planet.name == "Ceres") body_enum = thermal::PlanetaryBody::CERES;
        
        // Set the illumination model to use the same body
        illumination_model->setPlanetaryBody(body_enum);
        
        std::cout << "Illumination model using: " << illumination_model->getTargetBodyName() 
              << " (Primary: " << illumination_model->getPrimaryBodyName() 
              << ", Meta-kernel: " << illumination_model->getMetaKernel() << ")" << std::endl;
    }
    double default_visible_albedo = illumination_model->getAlbedo();
    illumination_model->initialize_spice(verbose);
    illumination_model->setTimeSteps(time_steps);
    illumination_model->setAlbedo(default_visible_albedo);
    illumination_model->setKernelsPath(SPICE_KERNELS_DIR);
    illumination_model->setIncludePrimaryReflection(include_primary_reflection);

    // Load positions if available
    if (!positions_file.empty()) {
        if (!illumination_model->loadPositionsFromFile(positions_file)) {
            std::cerr << "Failed to load positions from file: " << positions_file << std::endl;
            return 1;
        }
    }
    
    // Calculate total simulation time including equilibration
    double equilibration_days = equilibration_orbits * 3.55; // multiple of orbital period
    double total_simulation_days = equilibration_days + duration_days;

    // Calculate adjusted start time that includes equilibration period
    std::string adjustedStartTime = calculateAdjustedStartTime(start_time, -equilibration_days);

    // Calculate time steps needed for proper temporal resolution
    // This ensures requested time_steps applies ONLY to the main simulation duration
    int equilibration_steps = static_cast<int>(time_steps * (equilibration_days / duration_days));
    int total_time_steps = equilibration_steps + time_steps;

    // Update the illumination model with the total time steps
    illumination_model->setTimeSteps(total_time_steps);

    // Run illumination model for the full period but keep track of where the requested interval starts
    std::cout << "Running illumination simulation for " << total_simulation_days 
              << " days (including " << equilibration_days << " days equilibration)" 
              << " starting at " << adjustedStartTime << std::endl;

    // Add ray tracing timer start
    auto ray_tracing_start = std::chrono::high_resolution_clock::now();

    bool success;
    if (use_batch_processing) {
        // Use the batch optimized method - faster with SIMD
        success = illumination_model->calculateIlluminationOptimized(
            mesh->vertices,
            mesh->faces,
            adjustedStartTime,
            total_simulation_days
        );
    } else {
        // Use the regular method - slower but may be more compatible
        success = illumination_model->calculateIllumination(
            mesh->vertices,
            mesh->faces,
            adjustedStartTime,
            total_simulation_days
        );
    }

    // Add ray tracing timer end and reporting
    auto ray_tracing_end = std::chrono::high_resolution_clock::now();
    auto ray_tracing_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        ray_tracing_end - ray_tracing_start).count();
    double ray_tracing_sec = ray_tracing_ms / 1000.0;

    std::cout << "Ray tracing computation completed in " << std::fixed << std::setprecision(3) 
              << ray_tracing_sec << " seconds (" << ray_tracing_ms << " ms)" << std::endl;

    if (!success) {
        std::cerr << "Illumination simulation failed!" << std::endl;
        return 1;
    }
    
    // Get full results
    const auto& illumination_values = illumination_model->getIllumination();
    const auto& direct_solar_values = illumination_model->getDirectSolarValues();
    const auto& primary_reflected_values = illumination_model->getPrimaryReflectedValues();
    const auto& primary_ir_values = illumination_model->getPrimaryIRFlux();
    const auto& times = illumination_model->getTimes();
    const auto& time_strings = illumination_model->getTimeStrings();
    const auto& sun_positions = illumination_model->getSunPositions();
    const auto& primary_positions = illumination_model->getPrimaryPositions();

    // Calculate the number of faces and other setup parameters
    size_t num_faces = mesh->faces.size();

    // DON'T filter illumination yet - run thermal model with full dataset first
    std::vector<std::vector<double>> surface_temps;
    std::vector<std::vector<double>> filtered_surface_temps;

    if (run_thermal_mesh) {
        // Configure OpenMP and setup thermal model as before
        #ifdef _OPENMP
        if (thermal_threads > 0) {
            omp_set_num_threads(thermal_threads);
            std::cout << "Using " << thermal_threads << " OpenMP threads for thermal calculations" << std::endl;
        } else {
            std::cout << "Using " << omp_get_max_threads() << " OpenMP threads (auto)" << std::endl;
        }
        #else
        std::cout << "OpenMP not available, running in single-threaded mode" << std::endl;
        #endif
        
        // Set up thermal model configuration
        thermal::Config thermal_config;
        thermal_config.m = 15;
        thermal_config.n = 20;
        thermal_config.b = 5;
        thermal_config.F = 0.5;
        thermal_config.NPERDAY = 24;
        thermal_config.NYEARSEQ = 2;
        thermal_config.sigma = 5.67e-8;
        thermal_config.chi = 2.7e-7;

        // Add albedo maps directory to config properties
        thermal_config.setProperty("albedo_maps_directory", albedo_maps_directory);

        // If using thermal model, add the maps to configuration
        if (!albedo_map_file.empty()) {
            thermal_config.setProperty("albedo_map", albedo_map_file);
        }
        
        if (!thermal_inertia_map_file.empty()) {
            thermal_config.setProperty("thermal_inertia_map", thermal_inertia_map_file);
        }
        
        // Add information about mesh type and resolution to thermal_config
        thermal_config.custom_properties["is_custom_mesh"] = (mesh_file.empty()) ? "0" : "1";
        thermal_config.custom_properties["mesh_resolution"] = std::to_string(mesh_resolution);

        // Debug output to verify properties were set correctly
        if (verbose) {
            std::cout << "Thermal config properties:" << std::endl;
            std::cout << " - is_custom_mesh: " << ((mesh_file.empty()) ? "0" : "1") << std::endl;
            std::cout << " - mesh_resolution: " << mesh_resolution << std::endl;
            std::cout << " - Standard mesh: " << (mesh_file.empty() ? "YES" : "NO") << std::endl;
        }
        std::vector<double> direct_solar_values_full;
        std::vector<double> primary_reflected_values_full;
        std::vector<double> primary_ir_values_full;

        // Check if illumination model is using chunked storage by checking if any of the arrays are empty
        bool using_chunked_storage = illumination_model->getDirectSolarValues().empty() || 
                                    illumination_model->getPrimaryReflectedValues().empty() || 
                                    illumination_model->getPrimaryIRFlux().empty();

        if (using_chunked_storage) {
            std::cout << "Detected chunked storage in illumination model. Extracting data for thermal calculations..." << std::endl;
            
            // Get the number of faces and time steps
            size_t num_faces = mesh->faces.size();
            size_t total_values = num_faces * total_time_steps;
            
            // Allocate space for the extracted values
            direct_solar_values_full.resize(total_values);
            primary_reflected_values_full.resize(total_values);
            primary_ir_values_full.resize(total_values);
            
            // Calculate reporting interval for progress updates
            size_t report_interval = std::max(size_t(1), num_faces / 20); // Report ~20 times
            
            // Extract values from chunked storage with progress reporting
            for (size_t f = 0; f < num_faces; f++) {
                // Show progress periodically
                if (f % report_interval == 0 || f == num_faces - 1) {
                    std::cout << "\rExtracting data from chunked storage: " 
                            << std::fixed << std::setprecision(1) 
                            << (100.0 * f / num_faces) << "%" << std::flush;
                }
                
                for (size_t t = 0; t < total_time_steps; t++) {
                    size_t idx = f * total_time_steps + t;
                    
                    // Use accessor methods to get values from chunked storage
                    // These methods handle chunking internally
                    direct_solar_values_full[idx] = illumination_model->getDirectSolarValue(f, t);
                    primary_reflected_values_full[idx] = illumination_model->getPrimaryReflectedValue(f, t);
                    primary_ir_values_full[idx] = illumination_model->getPrimaryIRValue(f, t);
                }
            }
            std::cout << std::endl; // End the progress line
        } else {
            // If not using chunked storage, use the regular arrays
            direct_solar_values_full = illumination_model->getDirectSolarValues();
            primary_reflected_values_full = illumination_model->getPrimaryReflectedValues();
            primary_ir_values_full = illumination_model->getPrimaryIRFlux();
        }

        // Now use our extracted values instead of the direct accessors
        bool thermal_success = runThermalModelsForAllFaces(
            output_dir,
            mesh,
            direct_solar_values_full,         // Use our extracted values
            primary_reflected_values_full,    // Use our extracted values 
            primary_ir_values_full,           // Use our extracted values
            times,                           // Use FULL times
            total_time_steps,                // Use FULL time step count
            thermal_config,
            default_visible_albedo,
            0.02,
            &surface_temps,
            equilibration_steps,
            total_time_steps,
            thermal_threads                  // Pass thermal_threads as the last parameter
        );
        
        if (!thermal_success) {
            std::cerr << "Error during thermal mesh simulation" << std::endl;
            return 1;
        }
    }

    // NOW filter BOTH illumination and thermal data together using the same approach 
    std::vector<double> filtered_illumination_values;
    std::vector<double> filtered_direct_solar_values;
    std::vector<double> filtered_primary_reflected_values;
    std::vector<double> filtered_primary_ir_values;
    std::vector<double> filtered_times;
    std::vector<std::string> filtered_time_strings;
    std::vector<std::array<double, 3>> filtered_sun_positions;
    std::vector<std::array<double, 3>> filtered_primary_positions;

    // Properly calculate array sizes and time step counts
    const int filtered_time_steps = time_steps;
    const size_t filtered_array_size = num_faces * filtered_time_steps;

    // First allocate arrays to appropriate sizes with proper initialization
    filtered_illumination_values.resize(filtered_array_size, 0.0);
    filtered_direct_solar_values.resize(filtered_array_size, 0.0);
    filtered_primary_reflected_values.resize(filtered_array_size, 0.0);
    filtered_primary_ir_values.resize(filtered_array_size, 0.0);

    // Clear time-related arrays
    filtered_times.clear();
    filtered_time_strings.clear();
    filtered_sun_positions.clear();
    filtered_primary_positions.clear();

    // Reserve capacity for better performance
    filtered_times.reserve(filtered_time_steps);
    filtered_time_strings.reserve(filtered_time_steps);
    filtered_sun_positions.reserve(filtered_time_steps);
    filtered_primary_positions.reserve(filtered_time_steps);

    // Process time arrays first
    std::cout << "Filtering " << total_time_steps << " time steps to " << filtered_time_steps 
            << " output time steps (skipping " << equilibration_steps << " equilibration steps)" << std::endl;

    // Extract the time-related arrays safely
    for (int t = equilibration_steps; t < total_time_steps && t < times.size(); t++) {
        filtered_times.push_back(times[t]);
        
        if (t < time_strings.size()) {
            filtered_time_strings.push_back(time_strings[t]);
        }
        
        if (t < sun_positions.size()) {
            filtered_sun_positions.push_back(sun_positions[t]);
        }
        
        if (t < primary_positions.size()) {
            filtered_primary_positions.push_back(primary_positions[t]);
        }
    }

    // For large meshes, process in batches to avoid excessive logging
    const size_t reporting_interval = std::max(size_t(1), num_faces / 20); // Report ~20 times
    std::cout << "Filtering illumination data..." << std::endl;

    // Filter illumination data with robust bounds checking
    for (size_t f = 0; f < num_faces; f++) {
        // Report progress periodically
        if (f % reporting_interval == 0 || f == num_faces - 1) {
            std::cout << "\rFiltering face " << f + 1 << "/" << num_faces << " (" 
                    << std::fixed << std::setprecision(1) << (100.0 * f / num_faces) 
                    << "%)" << std::flush;
        }
        
        // Process each time step for this face
        for (int t = equilibration_steps; t < total_time_steps; t++) {
            // New time index relative to filtered data
            int new_t = t - equilibration_steps;
            
            // Calculate source and destination indices
            size_t src_idx = f * total_time_steps + t;
            size_t dst_idx = f * filtered_time_steps + new_t;
            
            // Get direct illumination model references for consistency
            const auto& illumination_values = illumination_model->getIllumination();
            const auto& direct_solar_values = illumination_model->getDirectSolarValues();
            const auto& primary_reflected_values = illumination_model->getPrimaryReflectedValues();
            const auto& primary_ir_values = illumination_model->getPrimaryIRFlux();
            
            // Robust bounds checking for both source and destination arrays
            if (src_idx >= direct_solar_values.size() || dst_idx >= filtered_array_size) {
                continue; // Skip this index safely
            }
            
            // Now we can safely copy values with verified indices
            filtered_direct_solar_values[dst_idx] = direct_solar_values[src_idx];
            filtered_primary_reflected_values[dst_idx] = primary_reflected_values[src_idx];
            filtered_primary_ir_values[dst_idx] = primary_ir_values[src_idx];
            
            // Illumination is the sum of direct and reflected
            filtered_illumination_values[dst_idx] = 
                direct_solar_values[src_idx] + primary_reflected_values[src_idx];
                
            // Add additional bounds check for illumination values if they're separate from
            // the direct and reflected components
            if (illumination_values.size() > 0 && src_idx < illumination_values.size()) {
                // Override with actual illumination if available 
                // (in case it's not just the sum of direct and reflected)
                filtered_illumination_values[dst_idx] = illumination_values[src_idx];
            }
        }
    }
    std::cout << std::endl; // End progress line

    // Now filter thermal data with the same careful approach
    if (run_thermal_mesh && !surface_temps.empty()) {
        std::cout << "Filtering thermal data..." << std::endl;
        filtered_surface_temps.resize(surface_temps.size());
        
        for (size_t f = 0; f < surface_temps.size(); f++) {
            if (f % reporting_interval == 0 || f == surface_temps.size() - 1) {
                std::cout << "\rFiltering thermal face " << f + 1 << "/" << surface_temps.size() 
                        << " (" << std::fixed << std::setprecision(1) 
                        << (100.0 * f / surface_temps.size()) << "%)" << std::flush;
            }
            
            // Initialize with safe default temperature
            filtered_surface_temps[f].resize(filtered_time_steps, 50.0); 
            
            // Only copy valid indices
            for (int t = 0; t < filtered_time_steps; t++) {
                int src_t = t + equilibration_steps;
                
                // Check array bounds before accessing
                if (src_t < surface_temps[f].size()) {
                    filtered_surface_temps[f][t] = surface_temps[f][src_t];
                }
            }
        }
        std::cout << std::endl; // End progress line
    }

    // Now write body positions file with filtered data
    std::string positions_output = output_dir + "/body_positions.csv";
    exportBodyPositions(
        positions_output,
        filtered_sun_positions,
        filtered_primary_positions,
        filtered_time_strings
    );

    // Update the VTK output section with this more efficient approach
    std::string illumination_vtk_file = output_dir + "/ptm_illumination.vtk";
    if (thermal_vtk_file.empty()) {
        thermal_vtk_file = output_dir + "/ptm_thermal.vtk";
    }
    if (complete_vtk_file.empty()) {
        complete_vtk_file = output_dir + "/ptm_complete.vtk";
    }

    bool data_write_success = false;

    // Handle output based on format selection (binary vs VTK)
    if (output_format == "vtk") {
        // VTK format output - use existing code paths
        if (!run_thermal_mesh) {
            // For illumination-only runs, write just the illumination VTK
            std::cout << "Writing illumination data to " << illumination_vtk_file << "..." << std::endl;
            MeshUtils mesh_utils;
            data_write_success = mesh_utils.exportIlluminationToVTK(
                illumination_vtk_file,
                mesh,
                filtered_illumination_values,
                filtered_primary_ir_values,
                filtered_times,
                filtered_time_strings,
                time_steps);
        } else {
            // Generate dynamic filenames if not specified by user
            if (thermal_vtk_file.empty()) {
                thermal_vtk_file = generateSimulationFilename(
                    output_dir, 
                    planet.name,
                    start_time, 
                    "thermal"
                );
            }

            if (complete_vtk_file.empty()) {
                complete_vtk_file = generateSimulationFilename(
                    output_dir, 
                    planet.name,
                    start_time, 
                    "complete"
                );
            }

            // Conditionally write VTK files based on the selected mode
            if (vtk_output_mode == "both" || vtk_output_mode == "complete") {
                if (!writeCompleteVTK(
                    complete_vtk_file, mesh, 
                    illumination_values, primary_ir_values, surface_temps, 
                    times, time_strings, total_time_steps, mesh_file,
                    include_primary_ir_in_vtk)) {
                    std::cerr << "Failed to write complete VTK file" << std::endl;
                } else {
                    std::cout << "Complete data written to: " << complete_vtk_file << std::endl;
                    data_write_success = true;
                }
            }

            if (vtk_output_mode == "both" || vtk_output_mode == "thermal") {
                if (!writeThermalVTK(
                    thermal_vtk_file, mesh, surface_temps, 
                    times, time_strings, time_steps, false)) {
                    std::cerr << "Failed to write thermal VTK file" << std::endl;
                } else {
                    std::cout << "Thermal data written to: " << thermal_vtk_file << std::endl;
                    data_write_success = true;
                }
            }
        }
    } else {
        // Binary output format (default)
        std::cout << "Using binary output format for data files" << std::endl;

        // Determine which data to write based on simulation type
        if (!run_thermal_mesh) {
            // For illumination-only runs, we still export the mesh and illumination data
            data_write_success = exportDataAsBinary(
                output_dir,
                mesh,
                filtered_illumination_values,
                filtered_primary_ir_values,
                std::vector<std::vector<double>>(), // Empty temperature data
                filtered_times,
                filtered_time_strings,
                time_steps,
                !mesh_file.empty(), // is_custom_mesh if mesh_file is provided
                false // explicitly exclude equilibration steps
            );
        } else {
            // For thermal runs, export filtered data (excluding equilibration steps)
            data_write_success = exportDataAsBinary(
                output_dir,
                mesh,
                filtered_illumination_values, // Use filtered illumination values
                filtered_primary_ir_values,   // Use filtered primary IR values
                filtered_surface_temps,       // Use filtered surface temperatures
                filtered_times,               // Use filtered times
                filtered_time_strings,        // Use filtered time strings
                time_steps,                   // Use main simulation time steps only
                !mesh_file.empty(),           // is_custom_mesh if mesh_file is provided
                false                         // explicitly exclude equilibration steps
            );
        }

        if (data_write_success) {
            std::cout << "Data successfully written in binary format to: " << output_dir << std::endl;
        }
    }

    // Common error message for both formats
    if (!data_write_success) {
        std::cerr << "Warning: Failed to write output data files" << std::endl;
    }

    // Set the location data file path for visualization
    std::string location_data_file = "";
    if (export_locations) {
        // Use the provided location file or the default
        std::string locations_file = config.getProperty("locations_file", output_dir + "/default_locations.txt");
        
        // Check if the file exists
        if (!std::filesystem::exists(locations_file)) {
            // Create a default locations file with a few representative points
            locations_file = createDefaultLocationsFile(output_dir);
            if (locations_file.empty()) {
                std::cout << "Note: No locations file provided and unable to create default. Skipping location-specific data export." << std::endl;
                export_locations = false;
            } else {
                std::cout << "Created default locations file: " << locations_file << std::endl;
            }
        }
        
        if (export_locations) {
            // Only export if we have a valid locations file
            std::string series_output = config.getProperty("series_output", output_dir + "/illumination_series.csv");
            if (exportLocationIlluminationAndTemperature(
                    series_output, locations_file, mesh,
                    illumination_values, direct_solar_values, primary_reflected_values, primary_ir_values,
                    surface_temps, time_strings, time_steps)) {
                std::cout << "Location illumination and temperature data exported to " << series_output << std::endl;
                location_data_file = series_output;
            } else {
                std::cout << "Note: Location data export was skipped." << std::endl;
            }
        }
    }

    // Now run visualization with the appropriate VTK file
    if (create_visualization) {
        std::string vtk_for_visualization = complete_vtk_file;
        
        // For the case where only illumination was run, use the illumination VTK
        if (!run_thermal_mesh) {
            vtk_for_visualization = illumination_vtk_file;
        }
        
        runVisualization(
            vtk_for_visualization,  // Use the appropriate file
            output_dir + "/body_solar_flux.mp4",
            output_dir + "/body_primary_ir.mp4",
            data_write_success ? thermal_vtk_file : "",  
            output_dir + "/body_surface_temp.mp4",
            output_dir + "/maps",
            output_dir + "/body_positions.csv",
            location_data_file,  // Pass the correct location data file path
            output_dir + "/location_plots", // Use this instead of undefined location_plots_dir
            camera_distance,  // camera_distance
            camera_latitude,  // camera_latitude
            camera_longitude, // camera_longitude
            emission_angle,   // emission_angle
            azimuth,          // azimuth
            10,               // fps
            save_frames,
            no_maps,
            animation_width,  // animation_width
            animation_height  // animation_height
        );
    }
    
    // Calculate some statistics
    double total_solar_flux = 0.0;
    double total_reflected_flux = 0.0;
    double total_ir_flux = 0.0;
    double max_solar_flux = 0.0;
    double max_reflected_flux = 0.0;
    double max_ir_flux = 0.0;

    for (size_t i = 0; i < illumination_values.size(); ++i) {
        total_solar_flux += illumination_model->getDirectSolarValues()[i];
        total_reflected_flux += illumination_model->getPrimaryReflectedValues()[i];
        total_ir_flux += illumination_model->getPrimaryIRFlux()[i];
        
        max_solar_flux = std::max(max_solar_flux, illumination_model->getDirectSolarValues()[i]);
        max_reflected_flux = std::max(max_reflected_flux, illumination_model->getPrimaryReflectedValues()[i]);
        max_ir_flux = std::max(max_ir_flux, illumination_model->getPrimaryIRFlux()[i]);
    }

    double avg_solar_flux = total_solar_flux / illumination_values.size();
    double avg_reflected_flux = total_reflected_flux / illumination_values.size();
    double avg_ir_flux = total_ir_flux / illumination_values.size();

    double total_flux = 0.0;
    double max_flux = 0.0;
    double min_flux = std::numeric_limits<double>::max();
    
    for (const auto& flux : illumination_values) {
        total_flux += flux;
        max_flux = std::max(max_flux, flux);
        if (flux > 0.0) { // Only consider non-zero (illuminated areas)
            min_flux = std::min(min_flux, flux);
        }
    }
    
    double avg_flux = total_flux / illumination_values.size();
    
    // End timing
    auto end_time_point = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time_point - start_time_point).count();

    // Convert to seconds with decimal precision for display
    double duration_sec = duration_ms / 1000.0;

    // Print summary
    std::cout << "Simulation completed in " << std::fixed << std::setprecision(3) 
              << duration_sec << " seconds (" << duration_ms << " ms)" << std::endl;
    std::cout << "Results:" << std::endl;
    std::cout << "  Average total flux: " << avg_flux << " W/m²" << std::endl;
    std::cout << "    - Direct solar: " << avg_solar_flux << " W/m²" << std::endl;
    std::cout << "    - Primary reflected: " << avg_reflected_flux << " W/m²" << std::endl;
    std::cout << "    - Primary IR: " << avg_ir_flux << " W/m²" << std::endl;
    std::cout << "  Maximum total flux: " << max_flux << " W/m²" << std::endl;
    std::cout << "    - Direct solar: " << max_solar_flux << " W/m²" << std::endl;
    std::cout << "    - Primary reflected: " << max_reflected_flux << " W/m²" << std::endl;
    std::cout << "    - Primary IR: " << max_ir_flux << " W/m²" << std::endl;
    std::cout << "  Minimum flux (non-zero): " << min_flux << " W/m²" << std::endl;
    std::cout << "  Output written to " << output_dir << std::endl;

    if (export_locations) {
        if (locations_file.empty()) {
            std::cerr << "Error: --locations parameter is required when --export-locations is used" << std::endl;
        } else {
            std::string location_output = output_dir + "/illumination_series.csv";
            std::cout << "Exporting location illumination data to: " << location_output << std::endl;
            
            bool export_success = exportLocationIlluminationAndTemperature(
                location_output,
                locations_file,
                mesh,
                filtered_illumination_values,
                filtered_direct_solar_values,
                filtered_primary_reflected_values,
                filtered_primary_ir_values,
                filtered_surface_temps, // Include temperature data
                filtered_time_strings,
                time_steps
            );
            
            if (!export_success) {
                std::cerr << "Failed to export location data" << std::endl;
            }
        }
    }

    if (run_thermal_mesh && !filtered_surface_temps.empty()) {
        // Check if we've already exported location data
        if (!export_locations) {
            // Only export here if not already exported above
            std::string default_locations = createDefaultLocationsFile(output_dir);
            std::cout << "Exporting location data to: " << series_output << std::endl;
            bool export_success = exportLocationIlluminationAndTemperature(
                series_output,
                locations_file.empty() ? default_locations : locations_file,
                mesh,
                filtered_illumination_values,
                filtered_direct_solar_values,
                filtered_primary_reflected_values,
                filtered_primary_ir_values,
                filtered_surface_temps,
                filtered_time_strings,
                time_steps
            );
            
            if (!export_success) {
                std::cerr << "Failed to export location illumination and temperature data" << std::endl;
            }
        }
    }

    return 0;
}