#include "../mesh/mesh_utils.h"
#include "../maps/spatial_map.h"
#include "../constants/constants.h"
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <filesystem>
#include <atomic>
#include <mutex>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace ptm;

// Constants for albedo processing
namespace {
    constexpr double MIN_REASONABLE_ALBEDO = 0.15;  // Minimum reasonable albedo for Europa's surface
    constexpr double MAX_REASONABLE_ALBEDO = 0.90;  // Maximum reasonable albedo for Europa
    constexpr double DEFAULT_ALBEDO = 0.55;         // Default value if no valid albedo found
}

// Progress bar utility function
void printProgressBar(size_t current, size_t total, int barWidth = 50) {
    float progress = static_cast<float>(current) / total;
    int pos = static_cast<int>(barWidth * progress);
    
    std::cout << "\r[";
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    
    std::cout << "] " << int(progress * 100.0) << "% (" << current << "/" << total << ")" << std::flush;
}

void fixOutOfRangeAlbedos(std::vector<double>& face_albedos, const std::shared_ptr<Mesh>& mesh);
void exportAlbedoMapToFile(const std::string& filename, const std::vector<double>& albedos);
std::vector<double> importAlbedoMapFromFile(const std::string& filename);

/**
 * This utility pre-computes albedo maps for different mesh resolutions.
 * It loads a high-resolution albedo FITS map and calculates the appropriate
 * albedo for each face in meshes of different resolutions.
 */
int main(int argc, char** argv) {
    // Parse arguments
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <high_res_albedo.fits> <output_dir> [min_res] [max_res]" << std::endl;
        std::cout << "  min_res: Minimum resolution level (default: 3)" << std::endl;
        std::cout << "  max_res: Maximum resolution level (default: 7)" << std::endl;
        return 1;
    }
    
    std::string input_map = argv[1];
    std::string output_dir = argv[2];
    int min_resolution = (argc > 3) ? std::stoi(argv[3]) : 3;
    int max_resolution = (argc > 4) ? std::stoi(argv[4]) : 7;
    
    // Create output directory if it doesn't exist
    std::filesystem::create_directories(output_dir);
    
    // Load high-resolution albedo map
    std::cout << "Loading high-resolution albedo map: " << input_map << std::endl;
    SpatialMap albedo_map;
    if (!albedo_map.loadFromFITS(input_map)) {
        std::cerr << "Failed to load albedo map: " << albedo_map.getLastError() << std::endl;
        return 1;
    }
    
    std::cout << "Loaded map: " << albedo_map.getWidth() << "x" << albedo_map.getHeight() << " pixels" << std::endl;
    std::cout << "Longitude range: " << albedo_map.getLonMin() << " to " << albedo_map.getLonMax() << " degrees" << std::endl;
    std::cout << "Latitude range: " << albedo_map.getLatMin() << " to " << albedo_map.getLatMax() << " degrees" << std::endl;
    std::cout << "Data range: " << albedo_map.getMin() << " to " << albedo_map.getMax() << std::endl;
    std::cout << "Map resolution: " << albedo_map.getWidth() << "x" << albedo_map.getHeight() << std::endl;
    std::cout << "Value range: " << albedo_map.getMin() << " to " << albedo_map.getMax() << std::endl;
    
    // Process each resolution level
    for (int resolution = min_resolution; resolution <= max_resolution; resolution++) {
        std::cout << "\nGenerating mesh for resolution " << resolution << "..." << std::endl;
        
        // Create mesh at this resolution
        MeshUtils mesh_utils;
        mesh_utils.setMeshResolution(resolution);
        auto mesh = mesh_utils.getMesh();
        
        size_t num_faces = mesh->faces.size();
        std::cout << "Mesh has " << num_faces << " faces" << std::endl;
        
        // Pre-calculate albedo for each face
        std::vector<double> face_albedos(num_faces, DEFAULT_ALBEDO);
        int direct_averaging_count = 0;
        
        // First pass: Try to get average albedo within each triangle
        std::cout << "Performing triangle-based averaging..." << std::endl;
        
        // Thread-safe progress tracking
        std::atomic<size_t> completed(0);
        std::mutex progress_mutex;
        const size_t total_faces = mesh->faces.size();
        const size_t update_frequency = std::max(size_t(100), total_faces / 1000);
        
        #pragma omp parallel
        {
            // Ensure each thread has its own local count
            #pragma omp single
            direct_averaging_count = 0;
            
            // Thread-local storage for face albedos
            std::vector<double> local_face_albedos(num_faces, DEFAULT_ALBEDO);
            
            // Get the number of faces in the mesh
            size_t num_faces = mesh->faces.size();
            // Thread-local counter
            int local_count = 0;
            
            // Parallel loop over faces
            #pragma omp for schedule(dynamic, 100) nowait
            for (size_t f = 0; f < num_faces; f++) {
                const auto& face = mesh->faces[f];
                
                // Skip invalid faces
                if (face[0] >= mesh->vertices.size() || 
                    face[1] >= mesh->vertices.size() || 
                    face[2] >= mesh->vertices.size()) {
                    continue;
                }
                
                // Use face center coordinates directly - NO COORDINATE SHIFTS
                double lat = mesh->latitudes[f];  // Use face center latitude
                double lon = mesh->longitudes[f]; // Use face center longitude
                
                // Normalize longitude to [0, 360) for map lookup
                while (lon < 0.0) {
                    lon += 360.0;
                }
                while (lon >= 360.0) {
                    lon -= 360.0;
                }
                
                // Special case for faces near the prime meridian to avoid interpolation artifacts
                if (lon < 1.0 || lon > 359.0) {
                    // For points very close to the prime meridian, sample at multiple nearby points
                    // and average the results to avoid discontinuities
                    double sum_albedo = 0.0;
                    int count = 0;
                    
                    // Sample at 5 points: the center and 4 points slightly offset
                    for (double offset : {0.0, -0.5, 0.5, -0.25, 0.25}) {
                        double sample_lon = lon + offset;
                        
                        // Renormalize
                        while (sample_lon < 0.0) {
                            sample_lon += 360.0;
                        }
                        while (sample_lon >= 360.0) {
                            sample_lon -= 360.0;
                        }
                        
                        double map_albedo = albedo_map.getValue(sample_lon, lat);
                        
                        if (map_albedo != albedo_map.getMissingValue() && 
                            map_albedo >= 0.0 && map_albedo <= 1.0) {
                            sum_albedo += map_albedo;
                            count++;
                        }
                    }
                    
                    // If we got valid samples, use the average
                    if (count > 0) {
                        face_albedos[f] = sum_albedo / count;
                        local_count++;
                    }
                } else {
                    // Use regular interpolation for points away from the prime meridian
                    double map_albedo = albedo_map.getValue(lon, lat);
                    
                    // Store valid albedo values
                    if (map_albedo != albedo_map.getMissingValue() && 
                        map_albedo >= 0.0 && map_albedo <= 1.0) {
                        face_albedos[f] = map_albedo;
                        local_count++;
                    }
                }
                
                // Update progress
                size_t current = ++completed;
                if (current % update_frequency == 0 || current == total_faces) {
                    std::lock_guard<std::mutex> lock(progress_mutex);
                    printProgressBar(current, total_faces);
                }
            }
            
            // Combine local counts
            #pragma omp atomic
            direct_averaging_count += local_count;
        }
        
        std::cout << std::endl;  // End the progress bar line
        
        // Report statistics
        std::cout << "Triangle-based averaging found valid albedos for " << direct_averaging_count 
                  << " of " << num_faces << " faces ("
                  << (100.0 * direct_averaging_count / num_faces) << "%)" << std::endl;
        
        // Second pass: For faces without valid averages, try bilinear interpolation at face centers
        if (direct_averaging_count < num_faces) {
            int interpolation_count = 0;
            
            std::cout << "Performing bilinear interpolation at face centers..." << std::endl;
            for (size_t f = 0; f < num_faces; f++) {
                if (face_albedos[f] == DEFAULT_ALBEDO) {
                    // Get face center lat/lon - FLIP LATITUDE for map lookup
                    double lat = mesh->latitudes[f]; // Negate latitude
                    double lon = mesh->longitudes[f];
                    
                    // Normalize longitude to [0,360] range for map lookup
                    while (lon < 0) lon += 360.0;
                    while (lon >= 360.0) lon -= 360.0;
                    
                    // Get interpolated albedo value
                    double map_albedo = albedo_map.getValue(lon, lat);
                    
                    // Use map value if physically valid (0-1 range)
                    if (map_albedo != albedo_map.getMissingValue() && 
                        map_albedo >= 0.0 && map_albedo <= 1.0) {
                        face_albedos[f] = map_albedo;
                        interpolation_count++;
                    }
                }
            }
            
            std::cout << "Bilinear interpolation added albedos for " << interpolation_count 
                      << " more faces" << std::endl;
            
            // Third pass: Fill remaining gaps with nearest neighbor values
            int remaining_gaps = num_faces - direct_averaging_count - interpolation_count;
            if (remaining_gaps > 0) {
                std::cout << "Filling remaining " << remaining_gaps << " gaps with nearest neighbors..." << std::endl;
                
                std::vector<bool> has_valid_albedo(num_faces);
                for (size_t f = 0; f < num_faces; f++) {
                    has_valid_albedo[f] = (face_albedos[f] != DEFAULT_ALBEDO);
                }
                
                int filled_count = 0;
                for (size_t f = 0; f < num_faces; f++) {
                    if (!has_valid_albedo[f]) {
                        // Find nearest face with a valid albedo
                        double min_distance = std::numeric_limits<double>::max();
                        size_t nearest_face = f;
                        
                        for (size_t other_f = 0; other_f < num_faces; other_f++) {
                            if (has_valid_albedo[other_f]) {
                                // Calculate spherical distance between face centers
                                double dlat = mesh->latitudes[f] - mesh->latitudes[other_f];
                                double dlon = mesh->longitudes[f] - mesh->longitudes[other_f];
                                
                                // Handle longitude wrapping at 180°
                                if (dlon > 180.0) dlon -= 360.0;
                                if (dlon < -180.0) dlon += 360.0;
                                
                                // Simple spherical distance approximation
                                double dist = std::sqrt(dlat*dlat + dlon*dlon);
                                
                                if (dist < min_distance) {
                                    min_distance = dist;
                                    nearest_face = other_f;
                                }
                            }
                        }
                        
                        // Copy albedo from nearest valid face
                        if (nearest_face != f) {
                            face_albedos[f] = face_albedos[nearest_face];
                            filled_count++;
                        }
                    }
                }
                
                std::cout << "Nearest neighbor filling added albedos for " << filled_count 
                          << " remaining faces" << std::endl;
            }
        }
        
        // Fix any out-of-range albedo values
        std::cout << "\nPerforming final quality check on albedo values..." << std::endl;
        fixOutOfRangeAlbedos(face_albedos, mesh);
        
        // Add detailed numerical validation to prevent any potential issues
        std::cout << "\nPerforming detailed numerical validation..." << std::endl;
        int numerical_issues_fixed = 0;
        const double MIN_SAFE_ALBEDO = 0.01;  // Absolute minimum to prevent div/0 errors

        for (size_t f = 0; f < face_albedos.size(); f++) {
            // Check for exact zeros (dangerous for division), NaN, infinity
            if (face_albedos[f] == 0.0 || std::isnan(face_albedos[f]) || 
                std::isinf(face_albedos[f]) || face_albedos[f] < 0.0) {
                
                face_albedos[f] = MIN_SAFE_ALBEDO;  // Use minimal safe value
                numerical_issues_fixed++;
            }
            
            // Ensure all values are within a physically safe range
            if (face_albedos[f] < MIN_SAFE_ALBEDO) {
                face_albedos[f] = MIN_SAFE_ALBEDO;
                numerical_issues_fixed++;
            }
            else if (face_albedos[f] > 1.0) {
                face_albedos[f] = 1.0;
                numerical_issues_fixed++;
            }
        }

        if (numerical_issues_fixed > 0) {
            std::cout << "WARNING: Fixed " << numerical_issues_fixed 
                    << " potential numerical instability issues!" << std::endl;
        } else {
            std::cout << "No numerical stability issues found." << std::endl;
        }
        
        // Perform final validation to catch missing values
        std::cout << "Performing final validation to catch missing values..." << std::endl;
        int missing_values_fixed = 0;
        for (size_t f = 0; f < face_albedos.size(); f++) {
            // Check for missing values or other invalid values
            if (face_albedos[f] == albedo_map.getMissingValue() || 
                std::isnan(face_albedos[f]) || 
                std::isinf(face_albedos[f]) ||
                face_albedos[f] < 0.0 || 
                face_albedos[f] > 1.0) {
                    
                // Replace with default albedo
                face_albedos[f] = DEFAULT_ALBEDO;
                missing_values_fixed++;
            }
        }

        if (missing_values_fixed > 0) {
            std::cout << "WARNING: Fixed " << missing_values_fixed << " missing or invalid values!" << std::endl;
        } else {
            std::cout << "No missing or invalid values found." << std::endl;
        }
        
        // Compute statistics about the final albedo distribution
        double min_albedo = 1.0;
        double max_albedo = 0.0;
        double sum_albedo = 0.0;
        
        for (size_t f = 0; f < num_faces; f++) {
            min_albedo = std::min(min_albedo, face_albedos[f]);
            max_albedo = std::max(max_albedo, face_albedos[f]);
            sum_albedo += face_albedos[f];
        }
        
        double mean_albedo = sum_albedo / num_faces;
        
        std::cout << "Final albedo range: " << min_albedo << " to " << max_albedo 
                  << " (mean: " << mean_albedo << ")" << std::endl;
        
        // Save the face albedo map to a file
        std::string output_file = output_dir + "/Europa_Albedo_res" + std::to_string(resolution) + ".bin";
        exportAlbedoMapToFile(output_file, face_albedos);
        std::cout << "Saved face albedo map to " << output_file << std::endl;
    }
    
    std::cout << "\nComplete! Generated albedo maps for mesh resolutions " 
              << min_resolution << " to " << max_resolution << std::endl;
    
    return 0;
}

// Fix out-of-range albedo values
void fixOutOfRangeAlbedos(std::vector<double>& face_albedos, const std::shared_ptr<Mesh>& mesh) {
    const size_t num_faces = face_albedos.size();
    if (num_faces == 0) return;
    
    // Identify faces with unreasonable albedo values
    std::vector<size_t> problematic_faces;
    for (size_t f = 0; f < num_faces; f++) {
        if (face_albedos[f] < MIN_REASONABLE_ALBEDO || face_albedos[f] > MAX_REASONABLE_ALBEDO) {
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
            if (other_f == f || 
                face_albedos[other_f] < MIN_REASONABLE_ALBEDO || 
                face_albedos[other_f] > MAX_REASONABLE_ALBEDO) {
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
    }
}

// Export the face albedo map to a binary file
void exportAlbedoMapToFile(const std::string& filename, const std::vector<double>& albedos) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return;
    }
    
    // Write the number of elements as a header
    size_t num_faces = albedos.size();
    file.write(reinterpret_cast<char*>(&num_faces), sizeof(num_faces));
    
    // Write all albedo values
    file.write(reinterpret_cast<const char*>(albedos.data()), num_faces * sizeof(double));
}

// Import a face albedo map from a binary file
std::vector<double> importAlbedoMapFromFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open file for reading: " << filename << std::endl;
        return {};
    }
    
    // Read the number of elements
    size_t num_faces;
    file.read(reinterpret_cast<char*>(&num_faces), sizeof(num_faces));
    
    // Read all albedo values
    std::vector<double> albedos(num_faces);
    file.read(reinterpret_cast<char*>(albedos.data()), num_faces * sizeof(double));
    
    return albedos;
}