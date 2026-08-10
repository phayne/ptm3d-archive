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
#include <gdal.h>
#include <gdal_priv.h>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace ptm;

// Constants for elevation processing
namespace {
    // Minimum reasonable elevation adjustment relative to mean radius
    constexpr double MIN_REASONABLE_ELEVATION = -30000.0;  // -30 km
    // Maximum reasonable elevation adjustment relative to mean radius
    constexpr double MAX_REASONABLE_ELEVATION = 30000.0;   // +30 km
    // Default elevation if no valid value found
    constexpr double DEFAULT_ELEVATION = 0.0;              // No adjustment
    // Mean lunar radius in meters
    constexpr double MEAN_LUNAR_RADIUS = 1737400.0;        // 1737.4 km
}

// Progress bar utility function - reused from albedo_map_generator
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

// Fix out-of-range elevation values
void fixOutOfRangeElevations(
    std::vector<double>& vertex_elevations,
    const std::shared_ptr<Mesh>& mesh,
    double default_elevation = DEFAULT_ELEVATION);

// // Export the mesh with elevations to a binary file
// void exportMeshToFile(
//     const std::string& filename,
//     const std::shared_ptr<Mesh>& mesh,
//     const std::vector<double>& elevations);

// // Import a pre-computed mesh from a binary file
// std::shared_ptr<Mesh> importMeshFromFile(
//     const std::string& filename,
//     std::vector<double>& elevations);

// Helper function to check if coordinate ranges are reasonable
bool areCoordinatesReasonable(double lon_min, double lon_max, double lat_min, double lat_max) {
    // Check if the longitude range is within ±360°
    if (std::abs(lon_min) > 360 || std::abs(lon_max) > 360) {
        return false;
    }
    
    // Check if the latitude range is within ±90°
    if (std::abs(lat_min) > 90 || std::abs(lat_max) > 90) {
        return false;
    }
    
    // Check if the ranges are non-zero and have reasonable extents
    if (std::abs(lon_max - lon_min) < 0.1 || std::abs(lat_max - lat_min) < 0.1) {
        return false;
    }
    
    return true;
}

// Load DEM using GDAL library
bool loadDEMToSpatialMap(SpatialMap& map, const std::string& filename,
                         double mean_radius = 1737400.0) {
    // Initialize GDAL
    GDALAllRegister();
    
    // Open the dataset (C API)
    GDALDatasetH dataset = GDALOpen(filename.c_str(), GA_ReadOnly);
    if (!dataset) {
        std::cerr << "Error: Failed to open DEM: " << filename << std::endl;
        return false;
    }
    
    // Get dimensions
    int width  = GDALGetRasterXSize(dataset);
    int height = GDALGetRasterYSize(dataset);

    // Check if the DEM is too large to load at once (>2GB or >1 billion pixels)
    const size_t MAX_PIXELS = 1000000000; // 1 billion pixels
    const size_t pixel_count = static_cast<size_t>(width) * height;
    
    std::cout << "DEM dimensions: " << width << " x " << height << " (" 
              << pixel_count << " pixels)" << std::endl;
    
    // Get geotransform information
    double geotransform[6];
    if (GDALGetGeoTransform(dataset, geotransform) != CE_None) {
        std::cerr << "Error: Failed to read GeoTransform" << std::endl;
        GDALClose(dataset);
        return false;
    }
    
    // Print the geotransform for debugging
    std::cout << "Geotransform values:" << std::endl;
    std::cout << "  Origin X: " << geotransform[0] << std::endl;
    std::cout << "  Pixel Width: " << geotransform[1] << std::endl;
    std::cout << "  Row Rotation: " << geotransform[2] << std::endl;
    std::cout << "  Origin Y: " << geotransform[3] << std::endl;
    std::cout << "  Column Rotation: " << geotransform[4] << std::endl;
    std::cout << "  Pixel Height: " << geotransform[5] << std::endl;
    
    // Calculate coordinate ranges
    double lon_min = geotransform[0];
    double lon_max = lon_min + width * geotransform[1];
    double lat_max = geotransform[3];
    double lat_min = lat_max + height * geotransform[5];

    // Auto-detect projected (metric) Equidistant / SimpleCylindrical
    // coordinates and convert to degrees. The LOLA SLDEM2015 / LDEM_118m
    // products from PDS ship with projected meters at 0° standard parallel:
    //   x_meters = R * lon_radians ;  y_meters = R * lat_radians.
    // Triggered when the raw bounds are obviously not lat/lon (any |coord|
    // exceeding 720°), which is a safe heuristic since a global lat/lon
    // raster fits within ±180° lon / ±90° lat.
    if (std::abs(lon_min) > 720.0 || std::abs(lon_max) > 720.0 ||
        std::abs(lat_min) > 720.0 || std::abs(lat_max) > 720.0) {
        const double R = mean_radius;     // meters; passed via argv[5]
        const double rad_to_deg = 180.0 / M_PI;
        std::cout << "Detected projected (metric) coordinates; converting to "
                  << "degrees using R = " << R << " m\n";
        lon_min = lon_min * rad_to_deg / R;
        lon_max = lon_max * rad_to_deg / R;
        lat_min = lat_min * rad_to_deg / R;
        lat_max = lat_max * rad_to_deg / R;
        // GeoTIFF bounds can exceed exact ±90 / ±180 by a fraction of a pixel
        // due to mean-radius rounding in the projection definition. Clamp so
        // the downstream reasonableness check doesn't reject a global raster.
        lat_min = std::max(-90.0, std::min(90.0, lat_min));
        lat_max = std::max(-90.0, std::min(90.0, lat_max));
        lon_min = std::max(-360.0, std::min(360.0, lon_min));
        lon_max = std::max(-360.0, std::min(360.0, lon_max));
    }

    std::cout << "Initial coordinate range calculation:" << std::endl;
    std::cout << "  Longitude: " << lon_min << " to " << lon_max << " degrees" << std::endl;
    std::cout << "  Latitude: " << lat_min << " to " << lat_max << " degrees" << std::endl;
    
    // Check if the coordinates are reasonable
    if (!areCoordinatesReasonable(lon_min, lon_max, lat_min, lat_max)) {
        GDALClose(dataset);
        return false;
    }
    
    // Example band access via C API wherever needed later:
    // GDALRasterBandH band = GDALGetRasterBand(dataset, 1);
    // if (!band) { GDALClose(dataset); return false; }
    // ... GDALRasterIO(...) ...

    // If DEM is too large, create a downsampled version
    if (pixel_count > MAX_PIXELS) {
        int downsample_factor = static_cast<int>(std::ceil(std::sqrt(static_cast<double>(pixel_count) / MAX_PIXELS)));
        int new_width = width / downsample_factor;
        int new_height = height / downsample_factor;
        
        std::cout << "DEM is too large to load in memory. Downsampling by factor " 
                  << downsample_factor << " to " << new_width << " x " << new_height << std::endl;
        
        // Get the first band (assumed to be elevation)
        GDALRasterBandH band = GDALGetRasterBand(dataset, 1);
        if (!band) {
            std::cerr << "Error: Failed to get raster band 1" << std::endl;
            GDALClose(dataset);
            return false;
        }
        
        // Get no data value
        int has_nodata = 0;
        double nodata = GDALGetRasterNoDataValue(band, &has_nodata);
        if (!has_nodata) {
            nodata = -9999.0; // Default no data value
        }
        
        // Allocate memory for downsampled data
        std::vector<float> downsampled_data(new_width * new_height, nodata);
        
        // Process the DEM in tiles
        const int TILE_SIZE = 1024; // Tile size in pixels
        std::cout << "Processing DEM in tiles..." << std::endl;
        
        int num_tiles_x = (width + TILE_SIZE - 1) / TILE_SIZE;
        int num_tiles_y = (height + TILE_SIZE - 1) / TILE_SIZE;
        int total_tiles = num_tiles_x * num_tiles_y;
        int tiles_processed = 0;
        
        // For each tile
        for (int ty = 0; ty < num_tiles_y; ty++) {
            for (int tx = 0; tx < num_tiles_x; tx++) {
                // Calculate tile bounds
                int x_start = tx * TILE_SIZE;
                int y_start = ty * TILE_SIZE;
                int x_end = std::min(x_start + TILE_SIZE, width);
                int y_end = std::min(y_start + TILE_SIZE, height);
                int tile_width = x_end - x_start;
                int tile_height = y_end - y_start;
                
                // Allocate memory for tile data
                std::vector<float> tile_data(tile_width * tile_height);
                
                // Read tile data
                CPLErr err = GDALRasterIO(band, GF_Read, x_start, y_start, tile_width, tile_height,
                                          tile_data.data(), tile_width, tile_height, GDT_Float32, 0, 0);
                
                if (err != CE_None) {
                    std::cerr << "Error reading DEM tile data." << std::endl;
                    GDALClose(dataset);
                    return false;
                }
                
                // Downsample tile data to the output array
                for (int y = 0; y < tile_height; y += downsample_factor) {
                    for (int x = 0; x < tile_width; x += downsample_factor) {
                        // Get average of non-nodata values in this downsample cell
                        int valid_count = 0;
                        float sum = 0.0f;
                        
                        for (int sy = 0; sy < downsample_factor && y + sy < tile_height; sy++) {
                            for (int sx = 0; sx < downsample_factor && x + sx < tile_width; sx++) {
                                float val = tile_data[(y + sy) * tile_width + (x + sx)];
                                if (val != nodata && !std::isnan(val)) {
                                    sum += val;
                                    valid_count++;
                                }
                            }
                        }
                        
                        // Output coordinates in downsampled grid
                        int out_x = (x_start + x) / downsample_factor;
                        int out_y = (y_start + y) / downsample_factor;
                        
                        // Set output value
                        if (valid_count > 0 && out_x < new_width && out_y < new_height) {
                            downsampled_data[out_y * new_width + out_x] = sum / valid_count;
                        }
                    }
                }
                
                // Update progress
                tiles_processed++;
                if (tiles_processed % 10 == 0 || tiles_processed == total_tiles) {
                    int percent = (100 * tiles_processed) / total_tiles;
                    std::cout << "\rProcessing tiles: " << percent << "% complete (" 
                              << tiles_processed << "/" << total_tiles << ")" << std::flush;
                }
            }
        }
        
        std::cout << std::endl;
        
        // Check and normalize downsampled data
        float min_val = std::numeric_limits<float>::max();
        float max_val = -std::numeric_limits<float>::max();
        int nan_count = 0;
        
        for (size_t i = 0; i < downsampled_data.size(); i++) {
            if (std::isnan(downsampled_data[i]) || downsampled_data[i] == nodata) {
                downsampled_data[i] = nodata;
                nan_count++;
            } else {
                min_val = std::min(min_val, downsampled_data[i]);
                max_val = std::max(max_val, downsampled_data[i]);
            }
        }
        
        if (nan_count > 0) {
            std::cout << "Warning: " << nan_count << " nodata/NaN values in downsampled DEM." << std::endl;
        }
        
        std::cout << "Downsampled DEM elevation range: " << min_val << " to " << max_val << " meters" << std::endl;
        
        // Initialize the spatial map with downsampled data
        bool success = map.initializeFromData(
            downsampled_data.data(), 
            new_width, 
            new_height, 
            lon_min, 
            lon_max, 
            lat_min, 
            lat_max, 
            MapProjection::EQUIRECTANGULAR,
            nodata
        );
        
        // Clean up
        GDALClose(dataset);
        
        // Validate that the map has reasonable coordinates
        if (success) {
            std::cout << "Initialized DEM with coordinate range:" << std::endl;
            std::cout << "  Longitude: " << map.getLonMin() << " to " << map.getLonMax() << " degrees" << std::endl;
            std::cout << "  Latitude: " << map.getLatMin() << " to " << map.getLatMax() << " degrees" << std::endl;
        }
        
        return success;
    } else {
        // Original code for manageable DEM sizes with the coordinate fixes
        // Get the first band (assumed to be elevation)
        GDALRasterBandH band = GDALGetRasterBand(dataset, 1);
        if (!band) {
            std::cerr << "Error: Failed to get raster band 1" << std::endl;
            GDALClose(dataset);
            return false;
        }
        
        // Get no data value
        int has_nodata = 0;
        double nodata = GDALGetRasterNoDataValue(band, &has_nodata);
        if (!has_nodata) {
            nodata = -9999.0; // Default no data value
        }
        
        // Read the entire band into memory
        std::vector<float> data(width * height);
        CPLErr err = GDALRasterIO(band, GF_Read, 0, 0, width, height, data.data(),
                                  width, height, GDT_Float32, 0, 0);
        
        if (err != CE_None) {
            std::cerr << "Error reading DEM data." << std::endl;
            GDALClose(dataset);
            return false;
        }
        
        // Check and normalize data
        float min_val = std::numeric_limits<float>::max();
        float max_val = -std::numeric_limits<float>::max();
        int nan_count = 0;
        
        for (size_t i = 0; i < data.size(); i++) {
            if (std::isnan(data[i]) || (has_nodata && data[i] == nodata)) {
                data[i] = nodata;  // Mark as no data
                nan_count++;
            } else {
                min_val = std::min(min_val, data[i]);
                max_val = std::max(max_val, data[i]);
            }
        }
        
        if (nan_count > 0) {
            std::cout << "Warning: " << nan_count << " nodata/NaN values in DEM." << std::endl;
        }
        
        std::cout << "DEM elevation range: " << min_val << " to " << max_val << " meters" << std::endl;
        
        // Initialize the spatial map
        bool success = map.initializeFromData(
            data.data(), 
            width, 
            height, 
            lon_min, 
            lon_max, 
            lat_min, 
            lat_max, 
            MapProjection::EQUIRECTANGULAR,
            nodata
        );
        
        // Clean up
        GDALClose(dataset);
        
        // Validate that the map has reasonable coordinates
        if (success) {
            std::cout << "Initialized DEM with coordinate range:" << std::endl;
            std::cout << "  Longitude: " << map.getLonMin() << " to " << map.getLonMax() << " degrees" << std::endl;
            std::cout << "  Latitude: " << map.getLatMin() << " to " << map.getLatMax() << " degrees" << std::endl;
        }
        
        return success;
    }
}

// Calculate longitude buffer for edge handling - shared with albedo_map_generator
double calculateLongitudeBuffer(int resolution) {
    // For an icosphere at resolution N, triangles are approximately 360°/(4^N * 5) in angular size
    // We want buffer to be 2-3x a typical triangle width to ensure smooth transitions
    double base_triangle_size = 360.0 / (std::pow(4, resolution) * 5);
    double buffer = std::min(5.0, std::max(0.5, base_triangle_size * 2.5));
    
    std::cout << "Using longitude boundary buffer of " << buffer 
              << "° for mesh resolution " << resolution << std::endl;
    
    return buffer;
}

/**
 * This utility pre-computes mesh files with accurate elevations for different resolutions.
 * It loads a high-resolution DEM and calculates the appropriate elevation
 * for each vertex in meshes of different resolutions.
 */
int main(int argc, char** argv) {
    // Parse arguments
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <dem_file.tif> <output_dir> [min_res] [max_res] [body_radius] [elevation_scale]" << std::endl;
        std::cout << "  dem_file.tif: High-resolution DEM file (TIFF/GeoTIFF)" << std::endl;
        std::cout << "  output_dir: Output directory for generated meshes" << std::endl;
        std::cout << "  min_res: Minimum resolution level (default: 3)" << std::endl;
        std::cout << "  max_res: Maximum resolution level (default: 7)" << std::endl;
        std::cout << "  body_radius: Mean radius of body in meters (default: 1737400 for Moon)" << std::endl;
        std::cout << "  elevation_scale: Scaling factor for elevation values (default: 0.5)" << std::endl;
        std::cout << "                   0.5 is the LOLA SLDEM2015 / LDEM_118m DN-to-meter factor." << std::endl;
        std::cout << "                   Override only if your input DEM is already pre-scaled to meters." << std::endl;
        return 1;
    }

    std::string input_dem = argv[1];
    std::string output_dir = argv[2];
    int min_resolution = (argc > 3) ? std::stoi(argv[3]) : 3;
    int max_resolution = (argc > 4) ? std::stoi(argv[4]) : 7;
    double mean_radius = (argc > 5) ? std::stod(argv[5]) : MEAN_LUNAR_RADIUS;
    // Default 0.5 reflects the LOLA SLDEM2015 / LDEM_118m DN-to-meter factor
    // (16-bit signed integers store elevation in 0.5 m increments).
    double elevation_scale = (argc > 6) ? std::stod(argv[6]) : 0.5;

    std::cout << "Using elevation scaling factor: " << elevation_scale << "x";
    if (elevation_scale != 0.5) {
        std::cout << "  WARNING: non-default (LOLA DN factor is 0.5)";
    }
    std::cout << std::endl;
    
    // Create output directory if it doesn't exist
    std::filesystem::create_directories(output_dir);
    
    // Load high-resolution DEM
    std::cout << "Loading high-resolution DEM: " << input_dem << std::endl;
    SpatialMap dem_map;

    if (!loadDEMToSpatialMap(dem_map, input_dem, mean_radius)) {
        std::cerr << "Fatal error: Could not load the DEM. Exiting." << std::endl;
        return 1;
    }
    
    std::cout << "Loaded DEM: " << dem_map.getWidth() << "x" << dem_map.getHeight() << " pixels" << std::endl;
    std::cout << "Longitude range: " << dem_map.getLonMin() << " to " << dem_map.getLonMax() << " degrees" << std::endl;
    std::cout << "Latitude range: " << dem_map.getLatMin() << " to " << dem_map.getLatMax() << " degrees" << std::endl;
    std::cout << "Elevation range: " << dem_map.getMin() << " to " << dem_map.getMax() << " meters" << std::endl;
    
    // Process each resolution level
    for (int resolution = min_resolution; resolution <= max_resolution; resolution++) {
        std::cout << "\nGenerating mesh for resolution " << resolution << "..." << std::endl;
        
        // Create base mesh at this resolution
        MeshUtils meshUtils;
        meshUtils.setMeshResolution(resolution);
        auto mesh = meshUtils.getMesh();
        
        // Vertex elevations - start with zeros
        std::vector<double> vertex_elevations(mesh->vertices.size(), 0.0);
        
        // Calculate the longitude buffer based on resolution
        double lon_buffer = calculateLongitudeBuffer(resolution);
        
        // Phase 1: Estimate elevation for each vertex using DEM
        std::cout << "Mapping DEM elevations to vertices..." << std::endl;
        
        int valid_elevations = 0;
        
        #pragma omp parallel for reduction(+:valid_elevations)
        for (size_t v = 0; v < mesh->vertices.size(); v++) {
            // Get the lat/lon of this vertex
            auto lat_lon = meshUtils.latLonFromCartesian(
                mesh->vertices[v][0], 
                mesh->vertices[v][1], 
                mesh->vertices[v][2]
            );
            
            double lat = lat_lon.first;
            double lon = lat_lon.second;
            
            // Use the improved bilinear interpolation which now properly handles
            // wraparound at the 0°/360° boundary for cylindrical maps
            double elevation = dem_map.getValue(lon, lat);
            
            if (elevation != dem_map.getMissingValue() && 
                elevation >= MIN_REASONABLE_ELEVATION && 
                elevation <= MAX_REASONABLE_ELEVATION) {
                vertex_elevations[v] = elevation;
                valid_elevations++;
            }
        }
        
        std::cout << "Found valid elevations for " << valid_elevations << " of " 
                  << mesh->vertices.size() << " vertices (" 
                  << (100.0 * valid_elevations / mesh->vertices.size()) << "%)" << std::endl;
        
        // Phase 2: Fill gaps using nearest neighbors
        if (valid_elevations < mesh->vertices.size()) {
            std::cout << "Filling " << (mesh->vertices.size() - valid_elevations) 
                      << " vertices with missing elevations..." << std::endl;
            
            // Create an array of missing vertices and valid vertices
            std::vector<size_t> missing_vertices;
            std::vector<size_t> valid_vertices;
            
            for (size_t v = 0; v < mesh->vertices.size(); v++) {
                if (vertex_elevations[v] == 0.0) {
                    missing_vertices.push_back(v);
                } else {
                    valid_vertices.push_back(v);
                }
            }
            
            // For each missing vertex, find the nearest valid vertex
            int filled = 0;
            
            #pragma omp parallel for reduction(+:filled)
            for (size_t i = 0; i < missing_vertices.size(); i++) {
                size_t missing_idx = missing_vertices[i];
                double min_distance = std::numeric_limits<double>::max();
                double best_elevation = DEFAULT_ELEVATION;
                
                // Find the nearest valid vertex
                for (size_t j = 0; j < valid_vertices.size(); j++) {
                    size_t valid_idx = valid_vertices[j];
                    
                    // Calculate Euclidean distance
                    double dx = mesh->vertices[missing_idx][0] - mesh->vertices[valid_idx][0];
                    double dy = mesh->vertices[missing_idx][1] - mesh->vertices[valid_idx][1];
                    double dz = mesh->vertices[missing_idx][2] - mesh->vertices[valid_idx][2];
                    double distance = std::sqrt(dx*dx + dy*dy + dz*dz);
                    
                    if (distance < min_distance) {
                        min_distance = distance;
                        best_elevation = vertex_elevations[valid_idx];
                    }
                }
                
                // Assign the best elevation
                vertex_elevations[missing_idx] = best_elevation;
                filled++;
                
                // Show progress
                if (i % 1000 == 0 || i == missing_vertices.size() - 1) {
                    #pragma omp critical
                    {
                        printProgressBar(i + 1, missing_vertices.size());
                    }
                }
            }
            
            std::cout << "\nFilled " << filled << " vertices with nearest-neighbor interpolation." << std::endl;
        }
        
        // Phase 3: Apply the elevations to create the final mesh
        std::cout << "Applying elevations to create final mesh..." << std::endl;
        
        // Make a copy of the original vertices for safety
        auto original_vertices = mesh->vertices;
        
        // Apply the elevation to each vertex
        for (size_t v = 0; v < mesh->vertices.size(); v++) {
            // Get the normalized direction vector (x,y,z)
            double x = original_vertices[v][0];
            double y = original_vertices[v][1];
            double z = original_vertices[v][2];
            double length = std::sqrt(x*x + y*y + z*z);
            
            // Normalize
            x /= length;
            y /= length;
            z /= length;
            
            // Apply mean radius plus elevation with optional scaling
            double radius = mean_radius + (vertex_elevations[v] * elevation_scale);
            
            // Set the new vertex position
            mesh->vertices[v][0] = x * radius;
            mesh->vertices[v][1] = y * radius;
            mesh->vertices[v][2] = z * radius;
        }
        
        // Calculate vertex radius statistics for verification
        double min_radius = std::numeric_limits<double>::max();
        double max_radius = -std::numeric_limits<double>::max();
        double sum_radius = 0.0;
        
        for (const auto& v : mesh->vertices) {
            double r = std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
            min_radius = std::min(min_radius, r);
            max_radius = std::max(max_radius, r);
            sum_radius += r;
        }
        
        double mean_vertex_radius = sum_radius / mesh->vertices.size();
        double radius_range = max_radius - min_radius;
        
        std::cout << "Vertex radius statistics:" << std::endl;
        std::cout << "  Min radius: " << min_radius << " meters" << std::endl;
        std::cout << "  Max radius: " << max_radius << " meters" << std::endl;
        std::cout << "  Mean radius: " << mean_vertex_radius << " meters" << std::endl;
        std::cout << "  Radius range: " << radius_range << " meters (" 
                  << (100.0 * radius_range / mean_vertex_radius) << "%)" << std::endl;
        
        // Recalculate face properties with the new elevations.
        // setupMesh() already populated face_normals (radial) and face_centers
        // (from un-displaced unit-icosphere vertices). Without clearing those
        // arrays, calculateFaceProperties' skip-cache short-circuits and we
        // ship stale data: normals stay purely radial (no slope tilt) and
        // face_centers stay at the wrong scale. Clearing forces a clean
        // recompute from the now-displaced vertex positions.
        mesh->face_normals.clear();
        mesh->face_centers.clear();
        meshUtils.calculateFaceProperties(false);
        
        // Extract the body name from the input file or use a command-line parameter
        std::string body_name = "Moon"; // Default to Moon
        
        // Try to determine the body from the input file name
        std::filesystem::path input_path(input_dem);
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
        
        std::cout << "Generating mesh for body: " << body_name << std::endl;
        // 2026-04 naming: topo_<body>_ico<N>.bin (parallels hpar_<body>_ico<N>.bin
        // and ti273k_<body>_ico<N>.bin under data/property_maps/<body>/).
        std::string body_lower = body_name;
        std::transform(body_lower.begin(), body_lower.end(), body_lower.begin(),
                       [](unsigned char c){ return std::tolower(c); });
        std::string output_file = output_dir + "/topo_" + body_lower
                                + "_ico" + std::to_string(resolution) + ".bin";

        // Apply elevation_scale to the elevations array as well, so the file
        // stores elevations in meters (matching the geometry of the vertex
        // positions written above). Canonical Moon_Topo_res9.bin (the
        // Artemis-validated reference) follows this convention.
        std::vector<double> scaled_elevations(vertex_elevations.size());
        for (size_t i = 0; i < vertex_elevations.size(); ++i) {
            scaled_elevations[i] = vertex_elevations[i] * elevation_scale;
        }

        // Save the mesh with (scaled) elevations
        meshUtils.exportMeshToFile(output_file, mesh, scaled_elevations);
        std::cout << "Saved topographic mesh to " << output_file << std::endl;
    }
    
    std::cout << "\nComplete! Generated elevation-accurate meshes for resolutions " 
              << min_resolution << " to " << max_resolution << std::endl;
    
    return 0;
}

// Fix unreasonable elevation values
void fixOutOfRangeElevations(
    std::vector<double>& vertex_elevations,
    const std::shared_ptr<Mesh>& mesh,
    double default_elevation) {
    
    const size_t num_vertices = vertex_elevations.size();
    if (num_vertices == 0) return;
    
    // Identify vertices with unreasonable elevation values
    std::vector<size_t> problematic_vertices;
    for (size_t v = 0; v < num_vertices; v++) {
        if (vertex_elevations[v] < MIN_REASONABLE_ELEVATION || 
            vertex_elevations[v] > MAX_REASONABLE_ELEVATION) {
            problematic_vertices.push_back(v);
        }
    }
    
    if (problematic_vertices.empty()) {
        std::cout << "No unreasonable elevation values found." << std::endl;
        return;
    }
    
    std::cout << "Found " << problematic_vertices.size() << " vertices with unreasonable elevation values ("
              << (100.0 * problematic_vertices.size() / num_vertices) << "%)." << std::endl;
    
    // Fix each problematic vertex using valid neighbors
    int fixed_count = 0;
    
    #pragma omp parallel for reduction(+:fixed_count)
    for (size_t i = 0; i < problematic_vertices.size(); i++) {
        size_t v = problematic_vertices[i];
        
        // Find valid neighbors with reasonable elevations
        std::vector<std::pair<size_t, double>> valid_neighbors; // vertex_id, distance
        
        for (size_t other_v = 0; other_v < num_vertices; other_v++) {
            if (other_v == v || 
                vertex_elevations[other_v] < MIN_REASONABLE_ELEVATION || 
                vertex_elevations[other_v] > MAX_REASONABLE_ELEVATION) {
                continue;
            }
            
            // Calculate Euclidean distance between vertices
            double dx = mesh->vertices[v][0] - mesh->vertices[other_v][0];
            double dy = mesh->vertices[v][1] - mesh->vertices[other_v][1];
            double dz = mesh->vertices[v][2] - mesh->vertices[other_v][2];
            double distance = std::sqrt(dx*dx + dy*dy + dz*dz);
            
            valid_neighbors.push_back({other_v, distance});
        }
        
        // Sort by distance
        std::sort(valid_neighbors.begin(), valid_neighbors.end(),
                 [](const auto& a, const auto& b) { return a.second < b.second; });
        
        // Take the average of the nearest K neighbors, or fewer if not enough are available
        const int K = 5;
        double sum_elevation = 0.0;
        int count = 0;
        
        for (size_t j = 0; j < valid_neighbors.size() && j < K; j++) {
            sum_elevation += vertex_elevations[valid_neighbors[j].first];
            count++;
        }
        
        if (count > 0) {
            vertex_elevations[v] = sum_elevation / count;
            fixed_count++;
        } else {
            vertex_elevations[v] = default_elevation;
            fixed_count++;
        }
    }
    
    // Report results
    std::cout << "Fixed " << fixed_count << " vertices with unreasonable elevation values ("
              << (100.0 * fixed_count / problematic_vertices.size()) << "% of problematic vertices)." << std::endl;
    
    // Check if any problematic vertices remain
    int remaining_bad = problematic_vertices.size() - fixed_count;
    if (remaining_bad > 0) {
        std::cout << "Warning: " << remaining_bad << " vertices could not be fixed and were set to default elevation." << std::endl;
    }
}

// // Export the mesh with elevations to a binary file
// void exportMeshToFile(
//     const std::string& filename,
//     const std::shared_ptr<Mesh>& mesh,
//     const std::vector<double>& elevations) {
    
//     try {
//         std::ofstream file(filename, std::ios::binary);
//         if (!file.is_open()) {
//             throw std::runtime_error("Could not open file for writing: " + filename);
//         }
        
//         // Write a magic header to identify the file format
//         const char* magic = "PTMTOPO";
//         file.write(magic, 7);
        
//         // Write the mesh resolution as metadata
//         int resolution = std::stoi(mesh->getProperty("resolution", "0"));
//         file.write(reinterpret_cast<const char*>(&resolution), sizeof(resolution));
        
//         // Write the number of vertices and faces
//         size_t num_vertices = mesh->vertices.size();
//         size_t num_faces = mesh->faces.size();
//         file.write(reinterpret_cast<const char*>(&num_vertices), sizeof(num_vertices));
//         file.write(reinterpret_cast<const char*>(&num_faces), sizeof(num_faces));
        
//         // Write all vertices
//         for (const auto& vertex : mesh->vertices) {
//             file.write(reinterpret_cast<const char*>(vertex.data()), 3 * sizeof(double));
//         }
        
//         // Write all faces
//         for (const auto& face : mesh->faces) {
//             file.write(reinterpret_cast<const char*>(face.data()), 3 * sizeof(int));
//         }
        
//         // Write vertex elevations
//         file.write(reinterpret_cast<const char*>(elevations.data()), elevations.size() * sizeof(double));
        
//         // Write face normals
//         for (const auto& normal : mesh->face_normals) {
//             file.write(reinterpret_cast<const char*>(normal.data()), 3 * sizeof(double));
//         }
        
//         // Write face centers
//         for (const auto& center : mesh->face_centers) {
//             file.write(reinterpret_cast<const char*>(center.data()), 3 * sizeof(double));
//         }
        
//         // Write latitudes and longitudes
//         file.write(reinterpret_cast<const char*>(mesh->latitudes.data()), mesh->latitudes.size() * sizeof(double));
//         file.write(reinterpret_cast<const char*>(mesh->longitudes.data()), mesh->longitudes.size() * sizeof(double));
        
//         file.close();
//     }
//     catch (const std::exception& e) {
//         std::cerr << "Error exporting mesh to file: " << e.what() << std::endl;
//     }
// }

// // Import a pre-computed mesh from a binary file
// std::shared_ptr<Mesh> importMeshFromFile(
//     const std::string& filename,
//     std::vector<double>& elevations) {
    
//     try {
//         std::ifstream file(filename, std::ios::binary);
//         if (!file.is_open()) {
//             throw std::runtime_error("Could not open file for reading: " + filename);
//         }
        
//         auto mesh = std::make_shared<Mesh>();
        
//         // Read and verify the magic header
//         char magic[8] = {0};
//         file.read(magic, 7);
//         if (std::string(magic) != "PTMTOPO") {
//             throw std::runtime_error("Invalid topographic mesh file format: " + filename);
//         }
        
//         // Read the mesh resolution
//         int resolution;
//         file.read(reinterpret_cast<char*>(&resolution), sizeof(resolution));
//         mesh->setProperty("resolution", std::to_string(resolution));
        
//         // Read the number of vertices and faces
//         size_t num_vertices, num_faces;
//         file.read(reinterpret_cast<char*>(&num_vertices), sizeof(num_vertices));
//         file.read(reinterpret_cast<char*>(&num_faces), sizeof(num_faces));
        
//         // Read all vertices
//         mesh->vertices.resize(num_vertices);
//         for (auto& vertex : mesh->vertices) {
//             file.read(reinterpret_cast<char*>(vertex.data()), 3 * sizeof(double));
//         }
        
//         // Read all faces
//         mesh->faces.resize(num_faces);
//         for (auto& face : mesh->faces) {
//             file.read(reinterpret_cast<char*>(face.data()), 3 * sizeof(int));
//         }
        
//         // Read vertex elevations
//         elevations.resize(num_vertices);
//         file.read(reinterpret_cast<char*>(elevations.data()), elevations.size() * sizeof(double));
        
//         // Read face normals
//         mesh->face_normals.resize(num_faces);
//         for (auto& normal : mesh->face_normals) {
//             file.read(reinterpret_cast<char*>(normal.data()), 3 * sizeof(double));
//         }
        
//         // Read face centers
//         mesh->face_centers.resize(num_faces);
//         for (auto& center : mesh->face_centers) {
//             file.read(reinterpret_cast<char*>(center.data()), 3 * sizeof(double));
//         }
        
//         // Read latitudes and longitudes
//         mesh->latitudes.resize(num_faces);
//         mesh->longitudes.resize(num_faces);
//         file.read(reinterpret_cast<char*>(mesh->latitudes.data()), mesh->latitudes.size() * sizeof(double));
//         file.read(reinterpret_cast<char*>(mesh->longitudes.data()), mesh->longitudes.size() * sizeof(double));
        
//         file.close();
        
//         // Mark this as a topographic mesh
//         mesh->setProperty("is_topographic", "true");
        
//         return mesh;
//     }
//     catch (const std::exception& e) {
//         std::cerr << "Error importing mesh from file: " << e.what() << std::endl;
//         return nullptr;
//     }
// }