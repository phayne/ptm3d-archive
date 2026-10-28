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
#include <cmath>
#include <fitsio.h>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace ptm;

// Constants for albedo processing
namespace {
    constexpr double MIN_REASONABLE_ALBEDO = 0.01;  // Minimum reasonable albedo
    constexpr double MAX_REASONABLE_ALBEDO = 0.99;  // Maximum reasonable albedo
    constexpr double DEFAULT_ALBEDO = 0.5;         // Default value if no valid albedo found
}

// Longitude range for handling the prime meridian edge cases
double calculateLongitudeBuffer(int resolution) {
    // For an icosphere at resolution N, triangles are approximately 360°/(4^N * 5) in angular size
    // We want buffer to be 2-3x a typical triangle width to ensure smooth transitions
    double base_triangle_size = 360.0 / (std::pow(4, resolution) * 5);
    double buffer = std::min(5.0, std::max(0.5, base_triangle_size * 2.5));
    
    // Print info about the buffer being used
    //std::cout << "Using longitude boundary buffer of " << buffer
    //          << "° for mesh resolution " << resolution << std::endl;

    return buffer;
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


bool validateFITSFile(const std::string& filename) {
    std::cout << "Validating FITS file: " << filename << std::endl;
    
    // Check if file exists and is readable
    if (!std::filesystem::exists(filename)) {
        std::cerr << "Error: FITS file does not exist: " << filename << std::endl;
        return false;
    }
    
    std::ifstream testFile(filename, std::ios::binary);
    if (!testFile) {
        std::cerr << "Error: Cannot open FITS file: " << filename << std::endl;
        return false;
    }
    
    // Check file size (should be at least 2880 bytes for a valid FITS file)
    testFile.seekg(0, std::ios::end);
    std::streamsize fileSize = testFile.tellg();
    if (fileSize < 2880) {
        std::cerr << "Error: FITS file too small, possibly corrupted: " << filename << std::endl;
        return false;
    }
    
    // Check FITS header signature
    testFile.seekg(0, std::ios::beg);
    char header[80];
    testFile.read(header, 80);
    if (strncmp(header, "SIMPLE  =", 9) != 0) {
        std::cerr << "Error: Invalid FITS header (missing SIMPLE keyword)" << std::endl;
        return false;
    }
    
    std::cout << "FITS file validation passed. File size: " << fileSize << " bytes" << std::endl;
    return true;
}

bool directLoadFITSToSpatialMap(SpatialMap& map, const std::string& filename) {
    fitsfile* fptr = nullptr;
    int status = 0;
    
    // Open the FITS file
    fits_open_file(&fptr, filename.c_str(), READONLY, &status);
    if (status) {
        char error_msg[80];
        fits_get_errstatus(status, error_msg);
        std::cerr << "CFITSIO Error opening file: " << error_msg << std::endl;
        return false;
    }
    
    // Get image dimensions
    long naxes[2] = {0, 0};
    int naxis = 0;
    fits_get_img_dim(fptr, &naxis, &status);
    fits_get_img_size(fptr, 2, naxes, &status);
    
    if (status || naxis != 2) {
        std::cerr << "Error: Failed to get image dimensions" << std::endl;
        fits_close_file(fptr, &status);
        return false;
    }
    
    long width = naxes[0];
    long height = naxes[1];
    
    std::cout << "FITS dimensions: " << width << "x" << height << std::endl;
    
    // Read WCS information for coordinate mapping
    double crpix1 = 0, crpix2 = 0;  // Reference pixel
    double crval1 = 0, crval2 = 0;  // Value at reference pixel
    double cdelt1 = 0, cdelt2 = 0;  // Pixel size/increment
    
    // Get WCS keywords (with fallback defaults for lunar maps)
    fits_read_key(fptr, TDOUBLE, "CRPIX1", &crpix1, nullptr, &status);
    if (status) {
        status = 0;
        crpix1 = width / 2.0;  // Default to center pixel
    }
    
    fits_read_key(fptr, TDOUBLE, "CRPIX2", &crpix2, nullptr, &status);
    if (status) {
        status = 0;
        crpix2 = height / 2.0;  // Default to center pixel
    }
    
    fits_read_key(fptr, TDOUBLE, "CRVAL1", &crval1, nullptr, &status);
    if (status) {
        status = 0;
        crval1 = 0.0;  // Default to 0 longitude at center
    }
    
    fits_read_key(fptr, TDOUBLE, "CRVAL2", &crval2, nullptr, &status);
    if (status) {
        status = 0;
        crval2 = 0.0;  // Default to 0 latitude at center
    }
    
    fits_read_key(fptr, TDOUBLE, "CDELT1", &cdelt1, nullptr, &status);
    if (status) {
        status = 0;
        cdelt1 = 1.0;  // Default to 1 degree per pixel
    }
    
    fits_read_key(fptr, TDOUBLE, "CDELT2", &cdelt2, nullptr, &status);
    if (status) {
        status = 0;
        cdelt2 = 1.0;  // Default to 1 degree per pixel
    }
    
    // Calculate coordinate ranges (matches FITSReader::readWCS convention)
    double lon_min = crval1 - (crpix1 - 1) * cdelt1;
    double lon_max = lon_min + width * cdelt1;
    double lat_min = crval2 - (crpix2 - 1) * cdelt2;
    double lat_max = lat_min + height * cdelt2;
    
    std::cout << "Coordinate ranges from FITS:" << std::endl;
    std::cout << "  Longitude: " << lon_min << " to " << lon_max << " degrees" << std::endl;
    std::cout << "  Latitude: " << lat_min << " to " << lat_max << " degrees" << std::endl;
    
    // Allocate memory for image data
    std::vector<float> image_data(width * height);
    
    // Read the image data
    long fpixel[2] = {1, 1};  // FITS uses 1-based indexing
    int anynul = 0;
    float nullval = 0;
    
    // Try to read as float first
    status = 0;
    fits_read_pix(fptr, TFLOAT, fpixel, width * height, &nullval,
                  image_data.data(), &anynul, &status);
                  
    if (status) {
        status = 0;
        std::cout << "Float read failed, trying as double..." << std::endl;
        
        // Try reading as double if float read fails
        std::vector<double> double_data(width * height);
        fits_read_pix(fptr, TDOUBLE, fpixel, width * height, &nullval,
                      double_data.data(), &anynul, &status);
                      
        if (status) {
            std::cerr << "Failed to read image data" << std::endl;
            fits_close_file(fptr, &status);
            return false;
        }
        
        // Convert double to float
        for (size_t i = 0; i < double_data.size(); i++) {
            image_data[i] = static_cast<float>(double_data[i]);
        }
    }
    
    // Close the FITS file
    fits_close_file(fptr, &status);
    
    // Check for NaN or invalid values and calculate statistics
    float min_val = std::numeric_limits<float>::max();
    float max_val = -std::numeric_limits<float>::max();
    int nan_count = 0;
    
    for (size_t i = 0; i < image_data.size(); i++) {
        if (std::isnan(image_data[i])) {
            image_data[i] = 0.0f;  // Replace NaN with 0
            nan_count++;
        } else {
            min_val = std::min(min_val, image_data[i]);
            max_val = std::max(max_val, image_data[i]);
        }
    }
    
    if (nan_count > 0) {
        std::cout << "Warning: Replaced " << nan_count << " NaN values with 0" << std::endl;
    }
    
    std::cout << "Data range: " << min_val << " to " << max_val << std::endl;
    
    // Manually initialize the SpatialMap with the data we've loaded
    bool success = map.initializeFromData(
        image_data.data(), 
        width, 
        height, 
        lon_min, 
        lon_max, 
        lat_min, 
        lat_max
    );
    
    if (!success) {
        std::cerr << "Failed to initialize SpatialMap with loaded data" << std::endl;
        return false;
    }
    
    std::cout << "Successfully loaded FITS data directly into SpatialMap" << std::endl;
    return true;
}

// Function to safely load FITS file with error handling
bool safelyLoadFITSFile(SpatialMap& map, const std::string& filename) {
    try {
        // First validate the file
        if (!validateFITSFile(filename)) {
            return false;
        }
        
        std::cout << "Attempting to load FITS file..." << std::endl;
        
        // Step-by-step FITS loading with additional validation
        // 1. First check if we can open the file with CFITSIO directly
        fitsfile* fptr = nullptr;
        int status = 0;
        
        std::cout << "Opening FITS file with CFITSIO..." << std::endl;
        fits_open_file(&fptr, filename.c_str(), READONLY, &status);
        
        if (status) {
            char error_msg[80];
            fits_get_errstatus(status, error_msg);
            std::cerr << "CFITSIO Error: " << error_msg << std::endl;
            return false;
        }
        
        // 2. Get basic information about the FITS file
        int num_hdus = 0;
        fits_get_num_hdus(fptr, &num_hdus, &status);
        
        if (status) {
            char error_msg[80];
            fits_get_errstatus(status, error_msg);
            std::cerr << "CFITSIO Error getting HDU count: " << error_msg << std::endl;
            fits_close_file(fptr, &status);
            return false;
        }
        
        std::cout << "FITS file has " << num_hdus << " HDUs" << std::endl;
        
        // 3. Move to the primary HDU (image data)
        int hdu_type = 0;
        fits_movabs_hdu(fptr, 1, &hdu_type, &status);
        
        if (status) {
            char error_msg[80];
            fits_get_errstatus(status, error_msg);
            std::cerr << "CFITSIO Error moving to primary HDU: " << error_msg << std::endl;
            fits_close_file(fptr, &status);
            return false;
        }
        
        // 4. Check that this is an image HDU
        if (hdu_type != IMAGE_HDU) {
            std::cerr << "Error: Primary HDU is not an image (" << hdu_type << ")" << std::endl;
            fits_close_file(fptr, &status);
            return false;
        }
        
        // 5. Get the dimensions of the image
        int num_dims = 0;
        fits_get_img_dim(fptr, &num_dims, &status);
        
        if (status || num_dims != 2) {
            std::cerr << "Error: FITS image is not 2D (dimensions: " << num_dims << ")" << std::endl;
            fits_close_file(fptr, &status);
            return false;
        }
        
        long dims[2] = {0, 0};
        fits_get_img_size(fptr, 2, dims, &status);
        
        if (status) {
            char error_msg[80];
            fits_get_errstatus(status, error_msg);
            std::cerr << "CFITSIO Error getting image size: " << error_msg << std::endl;
            fits_close_file(fptr, &status);
            return false;
        }
        
        std::cout << "FITS image dimensions: " << dims[0] << "x" << dims[1] << std::endl;
        
        // Close the FITS file - we've validated it, now let SpatialMap load it
        fits_close_file(fptr, &status);
        
        // 6. Now let SpatialMap attempt to load the file
        std::cout << "Validated FITS structure, now loading into SpatialMap..." << std::endl;
        
        // Try loading with verbose output for debugging
        bool result = map.loadFromFITS(filename);
        
        if (!result) {
            std::cerr << "Failed to load FITS file: " << map.getLastError() << std::endl;
            return false;
        }
        
        // Validate data after loading
        if (map.getWidth() <= 0 || map.getHeight() <= 0) {
            std::cerr << "Error: Loaded map has invalid dimensions: " 
                      << map.getWidth() << "x" << map.getHeight() << std::endl;
            return false;
        }
        
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Exception during FITS loading: " << e.what() << std::endl;
        return false;
    }
    catch (...) {
        std::cerr << "Unknown exception during FITS loading" << std::endl;
        return false;
    }
}

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

    // Primary path: use SpatialMap::loadFromFITS (correct WCS via FITSReader)
    // Fallback: directLoadFITSToSpatialMap with manual WCS handling
    if (!safelyLoadFITSFile(albedo_map, input_map)) {
        std::cout << "Standard FITS loading failed, trying direct approach..." << std::endl;
        if (!directLoadFITSToSpatialMap(albedo_map, input_map)) {
            std::cerr << "Fatal error: Could not load the albedo map. Exiting." << std::endl;
            return 1;
        }
    }

    // Add additional validation after loading
    if (albedo_map.getMin() < 0.0 || albedo_map.getMax() > 1.0) {
        std::cout << "Warning: Albedo values outside expected range [0,1]." << std::endl;
        std::cout << "Map range: " << albedo_map.getMin() << " to " << albedo_map.getMax() << std::endl;
        std::cout << "Values will be clamped during processing." << std::endl;
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
                
                // Use face center coordinates directly
                double lat = mesh->latitudes[f];  // Use face center latitude
                double lon = mesh->longitudes[f]; // Use face center longitude
                
                // Normalize longitude to [0, 360) for map lookup
                while (lon < 0.0) {
                    lon += 360.0;
                }
                while (lon >= 360.0) {
                    lon -= 360.0;
                }
                
                // Use the improved bilinear interpolation which now properly handles
                // wraparound at the 0°/360° boundary for cylindrical maps.
                // Sample at the face's physical +latitude (north positive) to match every
                // runtime sampling path (thermal ptm_thermal_sim.cpp, thermal_model.cpp,
                // albedo_manager.cpp all use getValue(lon, +lat)). The previous
                // getValue(lon, -lat) flipped the hemisphere in the baked per-face bins.
                double map_albedo = albedo_map.getValue(lon, lat);
                
                // Store valid albedo values
                if (map_albedo != albedo_map.getMissingValue() && 
                    map_albedo >= 0.0 && map_albedo <= 1.0) {
                    face_albedos[f] = map_albedo;
                    local_count++;
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
                    // Get face center lat/lon (physical +latitude, north positive)
                    double lat = mesh->latitudes[f];
                    double lon = mesh->longitudes[f];

                    // Normalize longitude to [0,360] range for map lookup
                    while (lon < 0) lon += 360.0;
                    while (lon >= 360.0) lon -= 360.0;

                    // Get interpolated albedo value at the physical +latitude (the albedo generator latitude flip fix)
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
        // Extract body name from the input file or use a command-line parameter
        std::string body_name = "Europa"; // Default to Europa

        // Try to determine the body from the input file name
        std::filesystem::path input_path(input_map);
        std::string filename = input_path.filename().string();
        std::transform(filename.begin(), filename.end(), filename.begin(), ::tolower);

        // Check for known body names in the input filename
        if (filename.find("lunar") != std::string::npos || filename.find("moon") != std::string::npos) {
            body_name = "Moon";
        } else if (filename.find("mercury") != std::string::npos) {
            body_name = "Mercury";
        } else if (filename.find("europa") != std::string::npos) {
            body_name = "Europa";
        } else if (filename.find("mars") != std::string::npos) {
            body_name = "Mars";
        } else if (filename.find("enceladus") != std::string::npos) {
            body_name = "Enceladus";
        } else if (filename.find("titan") != std::string::npos) {
            body_name = "Titan";
        } else if (filename.find("ceres") != std::string::npos) {
            body_name = "Ceres";
        }

        std::cout << "Generating albedo map for body: " << body_name << std::endl;
        std::string output_file = output_dir + "/" + body_name + "_Albedo_res" + std::to_string(resolution) + ".bin";
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
    try {
        // Create directory if it doesn't exist
        std::filesystem::path filePath(filename);
        std::filesystem::create_directories(filePath.parent_path());
        
        // Open file with explicit error checking
        std::ofstream file(filename, std::ios::binary);
        if (!file) {
            std::cerr << "Failed to open file for writing: " << filename << std::endl;
            return;
        }
        
        // Write the number of elements as a header
        size_t num_faces = albedos.size();
        if (num_faces == 0) {
            std::cerr << "Warning: Exporting empty albedo map" << std::endl;
        }
        
        file.write(reinterpret_cast<char*>(&num_faces), sizeof(num_faces));
        
        // Check if write succeeded
        if (!file.good()) {
            std::cerr << "Error writing header to file: " << filename << std::endl;
            return;
        }
        
        // Write all albedo values
        file.write(reinterpret_cast<const char*>(albedos.data()), num_faces * sizeof(double));
        
        // Check if write succeeded
        if (!file.good()) {
            std::cerr << "Error writing data to file: " << filename << std::endl;
            return;
        }
        
        // Make sure all data is flushed to disk
        file.flush();
        file.close();
        
        // Verify file was written correctly
        if (std::filesystem::exists(filename)) {
            std::cout << "Successfully wrote " << num_faces << " albedo values to " << filename << std::endl;
        } else {
            std::cerr << "File writing verification failed: " << filename << std::endl;
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Exception during file export: " << e.what() << std::endl;
    }
}

// Import a face albedo map from a binary file
std::vector<double> importAlbedoMapFromFile(const std::string& filename) {
    try {
        // Check if file exists
        if (!std::filesystem::exists(filename)) {
            std::cerr << "File does not exist: " << filename << std::endl;
            return {};
        }
        
        // Get file size for validation
        std::uintmax_t fileSize = std::filesystem::file_size(filename);
        if (fileSize < sizeof(size_t)) {
            std::cerr << "File too small to be valid: " << filename << std::endl;
            return {};
        }
        
        // Open file with explicit error checking
        std::ifstream file(filename, std::ios::binary);
        if (!file) {
            std::cerr << "Failed to open file for reading: " << filename << std::endl;
            return {};
        }
        
        // Read the number of elements
        size_t num_faces = 0;
        file.read(reinterpret_cast<char*>(&num_faces), sizeof(num_faces));
        
        // Validate num_faces against file size
        size_t expected_size = sizeof(size_t) + num_faces * sizeof(double);
        if (expected_size != fileSize) {
            std::cerr << "File size mismatch. Expected: " << expected_size 
                      << ", Actual: " << fileSize << std::endl;
            return {};
        }
        
        // Allocate memory for albedo values
        std::vector<double> albedos(num_faces);
        
        // Read all albedo values
        file.read(reinterpret_cast<char*>(albedos.data()), num_faces * sizeof(double));
        
        // Check if read succeeded
        if (!file.good() && !file.eof()) {
            std::cerr << "Error reading data from file: " << filename << std::endl;
            return {};
        }
        
        std::cout << "Successfully read " << num_faces << " albedo values from " << filename << std::endl;
        return albedos;
    }
    catch (const std::exception& e) {
        std::cerr << "Exception during file import: " << e.what() << std::endl;
        return {};
    }
}