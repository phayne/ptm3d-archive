// dtm_advanced_processor.cpp
// Advanced DTM processing with Region of Interest extraction and adaptive radial refinement
// For focused analysis of landing sites and features with optimized mesh resolution
//
// PORT RECORD AND STATUS -- READ BEFORE USING
// -------------------------------------------
// Ported into the PTM3D tree from the standalone heat3d/dtm working directory
// (source md5 e61aac129b19b5aeef3e2fd1dacf48b5). The classes below
// (ROIExtractor, AdaptiveMeshRefiner) are ported unchanged.
//
// This program's main() is an UNFINISHED PIPELINE and has never produced a
// usable mesh. Two pieces were left as comment stubs by the original author:
//
//   1. "Convert ROI to initial mesh" -- nothing turns the ROIExtractor result
//      into vertices/triangles, so the original computed it and threw it away.
//      This port does not run the extraction at all (see the ROI branch in
//      main() for why).
//   2. "Create initial mesh from DTM" -- when no input mesh exists, none is
//      built, so generateAdaptiveMesh() is handed EMPTY vertex and triangle
//      vectors.
//
// In addition, the getElevation callback is a placeholder that returns 0.0f for
// every query, so any subdivision it did drive would sit on a flat datum.
//
// The consequence in the original: exportToPLY() wrote a PLY with 0 vertices and
// 0 faces, printed "Processing completed successfully!", and returned 0. This
// port keeps the algorithms byte-for-byte but REFUSES to write an empty mesh and
// exits non-zero, so the failure is visible instead of silent.
//
// For a working DTM -> mesh path with region-of-interest extraction and radial
// density falloff, use dtm_to_ply_delaunay (--roi-center/--roi-radius/
// --roi-max-points/--roi-core-frac/--density-exp). See
// docs/dem-to-mesh-workflow.md.

#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <string>
#include <cmath>
#include <algorithm>
#include <functional>
#include <memory>
#include <limits>
#include <iomanip>
#include <filesystem>
#include <queue>
#include <unordered_map>
#include <gdal.h>
#include <gdal_priv.h>
#include <cpl_conv.h>

// Constants
constexpr double MOON_RADIUS = 1737400.0;  // Mean lunar radius in meters
constexpr double DEG_TO_RAD = M_PI / 180.0;
constexpr double RAD_TO_DEG = 180.0 / M_PI;

// =============================================================================
// Region of Interest (ROI) Extractor
// =============================================================================

class ROIExtractor {
public:
    struct ROIParams {
        double center_lat;
        double center_lon;
        double radius_km;           // Radius in kilometers
        int max_points;            // Maximum number of points to extract
        bool use_geodesic;         // Use geodesic distance (true) or simple angular (false)
    };
    
    struct ROIResult {
        std::vector<float> elevation_data;
        int width;
        int height;
        double lon_min, lon_max;
        double lat_min, lat_max;
        double pixel_size_lon;
        double pixel_size_lat;
        double nodata_value;
    };

private:
    ROIParams params_;
    double body_radius_;
    
    // Calculate geodesic distance between two points on sphere
    double geodesicDistance(double lat1, double lon1, double lat2, double lon2) const {
        double lat1_rad = lat1 * DEG_TO_RAD;
        double lon1_rad = lon1 * DEG_TO_RAD;
        double lat2_rad = lat2 * DEG_TO_RAD;
        double lon2_rad = lon2 * DEG_TO_RAD;
        
        // Haversine formula
        double dlat = lat2_rad - lat1_rad;
        double dlon = lon2_rad - lon1_rad;
        
        double a = sin(dlat/2) * sin(dlat/2) +
                   cos(lat1_rad) * cos(lat2_rad) *
                   sin(dlon/2) * sin(dlon/2);
        double c = 2 * atan2(sqrt(a), sqrt(1-a));
        
        return body_radius_ * c / 1000.0;  // Return in km
    }
    
public:
    ROIExtractor(double body_radius = MOON_RADIUS) : body_radius_(body_radius) {}
    
    void setParams(const ROIParams& params) {
        params_ = params;
    }
    
    // Extract ROI from DTM data
    ROIResult extractFromDTM(GDALDataset* dataset) {
        ROIResult result;
        
        // Get DTM dimensions and geotransform
        int dtm_width = dataset->GetRasterXSize();
        int dtm_height = dataset->GetRasterYSize();
        
        double geotransform[6];
        dataset->GetGeoTransform(geotransform);
        
        double dtm_lon_min = geotransform[0];
        double dtm_pixel_size_lon = geotransform[1];
        double dtm_lat_max = geotransform[3];
        double dtm_pixel_size_lat = geotransform[5];  // Usually negative
        
        // Get elevation band
        GDALRasterBand* band = dataset->GetRasterBand(1);
        int has_nodata = 0;
        result.nodata_value = band->GetNoDataValue(&has_nodata);
        if (!has_nodata) {
            result.nodata_value = -9999.0;
        }
        
        // Calculate approximate bounding box for ROI
        double angular_radius = (params_.radius_km / (body_radius_ / 1000.0)) * RAD_TO_DEG;
        
        double roi_lat_min = params_.center_lat - angular_radius;
        double roi_lat_max = params_.center_lat + angular_radius;
        double roi_lon_min = params_.center_lon - angular_radius / cos(params_.center_lat * DEG_TO_RAD);
        double roi_lon_max = params_.center_lon + angular_radius / cos(params_.center_lat * DEG_TO_RAD);
        
        // Convert to pixel coordinates
        int px_min = std::max(0, (int)((roi_lon_min - dtm_lon_min) / dtm_pixel_size_lon));
        int px_max = std::min(dtm_width - 1, (int)((roi_lon_max - dtm_lon_min) / dtm_pixel_size_lon));
        int py_min = std::max(0, (int)((roi_lat_max - dtm_lat_max) / dtm_pixel_size_lat));
        int py_max = std::min(dtm_height - 1, (int)((roi_lat_min - dtm_lat_max) / dtm_pixel_size_lat));
        
        int roi_width = px_max - px_min + 1;
        int roi_height = py_max - py_min + 1;
        
        std::cout << "Initial ROI dimensions: " << roi_width << " x " << roi_height 
                  << " = " << (roi_width * roi_height) << " points" << std::endl;
        
        // Calculate downsampling if needed
        int downsample = 1;
        if (params_.max_points > 0 && roi_width * roi_height > params_.max_points) {
            downsample = (int)ceil(sqrt((double)(roi_width * roi_height) / params_.max_points));
            std::cout << "Applying downsample factor: " << downsample 
                      << " to meet max_points constraint" << std::endl;
        }
        
        // Calculate final dimensions
        result.width = (roi_width + downsample - 1) / downsample;
        result.height = (roi_height + downsample - 1) / downsample;
        result.elevation_data.resize(result.width * result.height);
        
        // Calculate actual bounds
        result.lon_min = dtm_lon_min + px_min * dtm_pixel_size_lon;
        result.lon_max = dtm_lon_min + px_max * dtm_pixel_size_lon;
        result.lat_max = dtm_lat_max + py_min * dtm_pixel_size_lat;
        result.lat_min = dtm_lat_max + py_max * dtm_pixel_size_lat;
        result.pixel_size_lon = dtm_pixel_size_lon * downsample;
        result.pixel_size_lat = dtm_pixel_size_lat * downsample;
        
        // Read and filter data
        std::vector<float> temp_data(roi_width * roi_height);
        CPLErr err = band->RasterIO(GF_Read, px_min, py_min, roi_width, roi_height,
                                    temp_data.data(), roi_width, roi_height, 
                                    GDT_Float32, 0, 0);
        
        if (err != CE_None) {
            std::cerr << "Error reading ROI data from DTM" << std::endl;
            return result;
        }
        
        // Downsample and apply distance filter
        int valid_points = 0;
        for (int y = 0; y < result.height; y++) {
            for (int x = 0; x < result.width; x++) {
                // Source pixel in ROI
                int src_x = x * downsample;
                int src_y = y * downsample;
                
                // Calculate lat/lon of this point
                double lon = result.lon_min + x * result.pixel_size_lon;
                double lat = result.lat_max + y * result.pixel_size_lat;
                
                // Check distance from center
                double dist_km = params_.use_geodesic ? 
                    geodesicDistance(params_.center_lat, params_.center_lon, lat, lon) :
                    angular_radius * sqrt(pow((lat - params_.center_lat), 2) + 
                                        pow((lon - params_.center_lon) * cos(lat * DEG_TO_RAD), 2));
                
                if (dist_km <= params_.radius_km) {
                    // Average values in downsample window
                    float sum = 0;
                    int count = 0;
                    
                    for (int dy = 0; dy < downsample && src_y + dy < roi_height; dy++) {
                        for (int dx = 0; dx < downsample && src_x + dx < roi_width; dx++) {
                            float val = temp_data[(src_y + dy) * roi_width + (src_x + dx)];
                            if (val != result.nodata_value && !std::isnan(val)) {
                                sum += val;
                                count++;
                            }
                        }
                    }
                    
                    if (count > 0) {
                        result.elevation_data[y * result.width + x] = sum / count;
                        valid_points++;
                    } else {
                        result.elevation_data[y * result.width + x] = result.nodata_value;
                    }
                } else {
                    result.elevation_data[y * result.width + x] = result.nodata_value;
                }
            }
        }
        
        std::cout << "Extracted ROI: " << result.width << " x " << result.height 
                  << " with " << valid_points << " valid points within radius" << std::endl;
        
        return result;
    }
};

// =============================================================================
// Adaptive Radial Mesh Refinement
// =============================================================================

class AdaptiveMeshRefiner {
public:
    struct RefinementParams {
        double center_lat;
        double center_lon;
        double inner_radius_km;     // Full resolution within this radius
        double outer_radius_km;     // Zero resolution beyond this radius
        int max_faces;             // Target maximum number of faces
        double min_edge_length_m;   // Minimum edge length in meters
        double max_edge_length_m;   // Maximum edge length in meters
        bool use_elevation_gradient; // Consider elevation gradient in refinement
    };
    
    struct Vertex {
        double x, y, z;
        double lat, lon;
        float elevation;
        int id;
    };
    
    struct Triangle {
        int v0, v1, v2;
        double center_lat, center_lon;
        double area;
        double distance_from_center;
        int level;  // Subdivision level
    };

private:
    RefinementParams params_;
    double body_radius_;
    std::vector<Vertex> vertices_;
    std::vector<Triangle> triangles_;
    int next_vertex_id_;
    
    // Calculate distance from center for refinement weight
    double distanceFromCenter(double lat, double lon) const {
        double lat1 = params_.center_lat * DEG_TO_RAD;
        double lon1 = params_.center_lon * DEG_TO_RAD;
        double lat2 = lat * DEG_TO_RAD;
        double lon2 = lon * DEG_TO_RAD;
        
        double dlat = lat2 - lat1;
        double dlon = lon2 - lon1;
        
        double a = sin(dlat/2) * sin(dlat/2) +
                   cos(lat1) * cos(lat2) * sin(dlon/2) * sin(dlon/2);
        double c = 2 * atan2(sqrt(a), sqrt(1-a));
        
        return body_radius_ * c / 1000.0;  // km
    }
    
    // Calculate refinement weight based on distance
    double getRefinementWeight(double distance_km) const {
        if (distance_km <= params_.inner_radius_km) {
            return 1.0;  // Full refinement
        } else if (distance_km >= params_.outer_radius_km) {
            return 0.0;  // No refinement
        } else {
            // Linear interpolation (could use other functions)
            double t = (distance_km - params_.inner_radius_km) / 
                      (params_.outer_radius_km - params_.inner_radius_km);
            return 1.0 - t;  // Linear falloff
            // Alternative: Gaussian falloff
            // return exp(-t * t * 2.0);
        }
    }
    
    // Calculate edge length in meters
    double edgeLength(const Vertex& v1, const Vertex& v2) const {
        double dx = v2.x - v1.x;
        double dy = v2.y - v1.y;
        double dz = v2.z - v1.z;
        return sqrt(dx*dx + dy*dy + dz*dz);
    }
    
    // Add a new vertex at midpoint
    int addMidpoint(int v1_id, int v2_id, 
                    std::function<float(double, double)> getElevation) {
        const Vertex& v1 = vertices_[v1_id];
        const Vertex& v2 = vertices_[v2_id];
        
        // Calculate midpoint in lat/lon space
        double mid_lat = (v1.lat + v2.lat) / 2.0;
        double mid_lon = (v1.lon + v2.lon) / 2.0;
        
        // Get elevation at midpoint
        float elevation = getElevation(mid_lat, mid_lon);
        if (std::isnan(elevation)) elevation = (v1.elevation + v2.elevation) / 2.0;
        
        // Convert to Cartesian
        double radius = body_radius_ + elevation;
        double lat_rad = mid_lat * DEG_TO_RAD;
        double lon_rad = mid_lon * DEG_TO_RAD;
        
        Vertex v;
        v.lat = mid_lat;
        v.lon = mid_lon;
        v.elevation = elevation;
        v.x = radius * cos(lat_rad) * cos(lon_rad);
        v.y = radius * cos(lat_rad) * sin(lon_rad);
        v.z = radius * sin(lat_rad);
        v.id = next_vertex_id_++;
        
        vertices_.push_back(v);
        return v.id;
    }
    
public:
    AdaptiveMeshRefiner(double body_radius = MOON_RADIUS) 
        : body_radius_(body_radius), next_vertex_id_(0) {}
    
    void setParams(const RefinementParams& params) {
        params_ = params;
    }
    
    // Generate adaptive mesh from regular grid
    void generateAdaptiveMesh(const std::vector<Vertex>& input_vertices,
                             const std::vector<Triangle>& input_triangles,
                             std::function<float(double, double)> getElevation) {
        vertices_ = input_vertices;
        triangles_ = input_triangles;
        next_vertex_id_ = vertices_.size();
        
        // Update vertex IDs
        for (size_t i = 0; i < vertices_.size(); i++) {
            vertices_[i].id = i;
        }
        
        // Calculate initial triangle properties
        for (auto& tri : triangles_) {
            updateTriangleProperties(tri);
        }
        
        std::cout << "Starting adaptive refinement..." << std::endl;
        std::cout << "  Initial: " << vertices_.size() << " vertices, " 
                  << triangles_.size() << " faces" << std::endl;
        
        // Priority queue for triangles to subdivide (largest area * weight first)
        using PriorityPair = std::pair<double, int>;
        auto compare = [](const PriorityPair& a, const PriorityPair& b) {
            return a.first < b.first;  // Max heap
        };
        std::priority_queue<PriorityPair, 
                           std::vector<PriorityPair>,
                           decltype(compare)> subdivision_queue(compare);
        
        // Initialize queue with triangles needing subdivision
        for (size_t i = 0; i < triangles_.size(); i++) {
            double priority = calculateSubdivisionPriority(triangles_[i]);
            if (priority > 0) {
                subdivision_queue.push({priority, i});
            }
        }
        
        // Subdivide triangles adaptively
        int subdivisions = 0;
        while (!subdivision_queue.empty() && triangles_.size() < params_.max_faces) {
            auto [priority, tri_idx] = subdivision_queue.top();
            subdivision_queue.pop();
            
            // Skip if triangle was already modified
            if (tri_idx >= triangles_.size()) continue;
            
            Triangle& tri = triangles_[tri_idx];
            
            // Check if subdivision is still needed
            if (!shouldSubdivide(tri)) continue;
            
            // Subdivide triangle into 4 smaller triangles
            subdivideTriangle(tri_idx, getElevation, subdivision_queue);
            subdivisions++;
            
            if (subdivisions % 100 == 0) {
                std::cout << "\r  Progress: " << vertices_.size() << " vertices, " 
                          << triangles_.size() << " faces" << std::flush;
            }
        }
        
        std::cout << "\n  Final: " << vertices_.size() << " vertices, " 
                  << triangles_.size() << " faces" << std::endl;
        std::cout << "  Performed " << subdivisions << " subdivisions" << std::endl;
    }
    
    // Calculate subdivision priority for a triangle
    double calculateSubdivisionPriority(const Triangle& tri) {
        // Get refinement weight based on distance from center
        double weight = getRefinementWeight(tri.distance_from_center);
        if (weight <= 0) return 0;
        
        // Consider triangle area and edge lengths
        const Vertex& v0 = vertices_[tri.v0];
        const Vertex& v1 = vertices_[tri.v1];
        const Vertex& v2 = vertices_[tri.v2];
        
        double edge0 = edgeLength(v0, v1);
        double edge1 = edgeLength(v1, v2);
        double edge2 = edgeLength(v2, v0);
        double max_edge = std::max({edge0, edge1, edge2});
        
        // Priority based on weighted edge length
        double target_edge = params_.min_edge_length_m + 
            (params_.max_edge_length_m - params_.min_edge_length_m) * (1.0 - weight);
        
        if (max_edge > target_edge) {
            return max_edge * weight;
        }
        
        return 0;
    }
    
    // Check if triangle should be subdivided
    bool shouldSubdivide(const Triangle& tri) {
        if (triangles_.size() >= params_.max_faces) return false;
        
        double weight = getRefinementWeight(tri.distance_from_center);
        if (weight <= 0) return false;
        
        const Vertex& v0 = vertices_[tri.v0];
        const Vertex& v1 = vertices_[tri.v1];
        const Vertex& v2 = vertices_[tri.v2];
        
        double edge0 = edgeLength(v0, v1);
        double edge1 = edgeLength(v1, v2);
        double edge2 = edgeLength(v2, v0);
        double max_edge = std::max({edge0, edge1, edge2});
        
        double target_edge = params_.min_edge_length_m + 
            (params_.max_edge_length_m - params_.min_edge_length_m) * (1.0 - weight);
        
        return max_edge > target_edge;
    }
    
    // Subdivide a triangle into 4 smaller triangles
    template<typename QueueType>
    void subdivideTriangle(int tri_idx, 
                           std::function<float(double, double)> getElevation,
                           QueueType& queue) {
        Triangle& tri = triangles_[tri_idx];
        
        // Create midpoint vertices
        int m01 = addMidpoint(tri.v0, tri.v1, getElevation);
        int m12 = addMidpoint(tri.v1, tri.v2, getElevation);
        int m20 = addMidpoint(tri.v2, tri.v0, getElevation);
        
        // Create 4 new triangles
        Triangle t0 = {tri.v0, m01, m20, 0, 0, 0, 0, tri.level + 1};
        Triangle t1 = {tri.v1, m12, m01, 0, 0, 0, 0, tri.level + 1};
        Triangle t2 = {tri.v2, m20, m12, 0, 0, 0, 0, tri.level + 1};
        Triangle t3 = {m01, m12, m20, 0, 0, 0, 0, tri.level + 1};
        
        // Update properties
        updateTriangleProperties(t0);
        updateTriangleProperties(t1);
        updateTriangleProperties(t2);
        updateTriangleProperties(t3);
        
        // Replace original triangle with first new one
        triangles_[tri_idx] = t0;
        
        // Add other three triangles
        int idx1 = triangles_.size();
        triangles_.push_back(t1);
        int idx2 = triangles_.size();
        triangles_.push_back(t2);
        int idx3 = triangles_.size();
        triangles_.push_back(t3);
        
        // Add to subdivision queue if needed
        auto addToQueue = [&](const Triangle& t, int idx) {
            double priority = calculateSubdivisionPriority(t);
            if (priority > 0) {
                queue.push({priority, idx});
            }
        };
        
        addToQueue(t0, tri_idx);
        addToQueue(t1, idx1);
        addToQueue(t2, idx2);
        addToQueue(t3, idx3);
    }
    
    // Update triangle properties (center, area, distance)
    void updateTriangleProperties(Triangle& tri) {
        const Vertex& v0 = vertices_[tri.v0];
        const Vertex& v1 = vertices_[tri.v1];
        const Vertex& v2 = vertices_[tri.v2];
        
        // Calculate center
        tri.center_lat = (v0.lat + v1.lat + v2.lat) / 3.0;
        tri.center_lon = (v0.lon + v1.lon + v2.lon) / 3.0;
        
        // Calculate area (cross product)
        double ax = v1.x - v0.x;
        double ay = v1.y - v0.y;
        double az = v1.z - v0.z;
        double bx = v2.x - v0.x;
        double by = v2.y - v0.y;
        double bz = v2.z - v0.z;
        
        double cx = ay * bz - az * by;
        double cy = az * bx - ax * bz;
        double cz = ax * by - ay * bx;
        
        tri.area = 0.5 * sqrt(cx*cx + cy*cy + cz*cz);
        
        // Calculate distance from refinement center
        tri.distance_from_center = distanceFromCenter(tri.center_lat, tri.center_lon);
    }
    
    // Get the refined mesh
    void getResults(std::vector<Vertex>& out_vertices, 
                   std::vector<Triangle>& out_triangles) {
        out_vertices = vertices_;
        out_triangles = triangles_;
    }
    
    // Export to PLY format
    bool exportToPLY(const std::string& filename) {
        std::cout << "\nExporting adaptive mesh to PLY: " << filename << std::endl;
        
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file for writing: " << filename << std::endl;
            return false;
        }
        
        // Write header
        file << "ply\n";
        file << "format binary_little_endian 1.0\n";
        file << "comment Adaptive radially-refined mesh for PTM3D\n";
        file << "comment Center: " << params_.center_lat << ", " << params_.center_lon << "\n";
        file << "comment Inner radius: " << params_.inner_radius_km << " km\n";
        file << "comment Outer radius: " << params_.outer_radius_km << " km\n";
        
        file << "element vertex " << vertices_.size() << "\n";
        file << "property float x\n";
        file << "property float y\n";
        file << "property float z\n";
        file << "property float nx\n";
        file << "property float ny\n";
        file << "property float nz\n";
        
        file << "element face " << triangles_.size() << "\n";
        file << "property list uchar int vertex_indices\n";
        file << "property float latitude\n";
        file << "property float longitude\n";
        file << "property int subdivision_level\n";
        file << "end_header\n";
        
        // Write vertices
        for (const auto& v : vertices_) {
            float x = static_cast<float>(v.x);
            float y = static_cast<float>(v.y);
            float z = static_cast<float>(v.z);
            file.write(reinterpret_cast<const char*>(&x), sizeof(float));
            file.write(reinterpret_cast<const char*>(&y), sizeof(float));
            file.write(reinterpret_cast<const char*>(&z), sizeof(float));
            
            // Normal
            double len = sqrt(x*x + y*y + z*z);
            float nx = (len > 0) ? static_cast<float>(x / len) : 0.0f;
            float ny = (len > 0) ? static_cast<float>(y / len) : 0.0f;
            float nz = (len > 0) ? static_cast<float>(z / len) : 0.0f;
            file.write(reinterpret_cast<const char*>(&nx), sizeof(float));
            file.write(reinterpret_cast<const char*>(&ny), sizeof(float));
            file.write(reinterpret_cast<const char*>(&nz), sizeof(float));
        }
        
        // Write faces
        for (const auto& t : triangles_) {
            unsigned char num_verts = 3;
            file.write(reinterpret_cast<const char*>(&num_verts), sizeof(unsigned char));
            
            int indices[3] = {t.v0, t.v1, t.v2};
            file.write(reinterpret_cast<const char*>(indices), 3 * sizeof(int));
            
            float lat = static_cast<float>(t.center_lat);
            float lon = static_cast<float>(t.center_lon);
            file.write(reinterpret_cast<const char*>(&lat), sizeof(float));
            file.write(reinterpret_cast<const char*>(&lon), sizeof(float));
            
            int level = t.level;
            file.write(reinterpret_cast<const char*>(&level), sizeof(int));
        }
        
        file.close();
        
        // Print statistics
        printMeshStatistics();
        
        return true;
    }
    
    void printMeshStatistics() {
        // Calculate subdivision level statistics
        std::unordered_map<int, int> level_counts;
        for (const auto& t : triangles_) {
            level_counts[t.level]++;
        }
        
        std::cout << "Mesh statistics:" << std::endl;
        std::cout << "  Total vertices: " << vertices_.size() << std::endl;
        std::cout << "  Total faces: " << triangles_.size() << std::endl;
        std::cout << "  Subdivision levels:" << std::endl;
        for (const auto& [level, count] : level_counts) {
            std::cout << "    Level " << level << ": " << count << " faces" << std::endl;
        }
        
        // Calculate resolution distribution
        double min_edge = std::numeric_limits<double>::max();
        double max_edge = 0;
        double total_edge = 0;
        int edge_count = 0;
        
        for (const auto& t : triangles_) {
            double edges[3] = {
                edgeLength(vertices_[t.v0], vertices_[t.v1]),
                edgeLength(vertices_[t.v1], vertices_[t.v2]),
                edgeLength(vertices_[t.v2], vertices_[t.v0])
            };
            
            for (double e : edges) {
                min_edge = std::min(min_edge, e);
                max_edge = std::max(max_edge, e);
                total_edge += e;
                edge_count++;
            }
        }
        
        std::cout << "  Edge length range: " << min_edge << " to " << max_edge << " m" << std::endl;
        std::cout << "  Average edge length: " << (total_edge / edge_count) << " m" << std::endl;
    }
};

// =============================================================================
// Main Processing Pipeline
// =============================================================================

int main(int argc, char* argv[]) {
    std::cout << "==============================================\n";
    std::cout << "   Advanced DTM Processor with ROI and\n";
    std::cout << "   Adaptive Radial Mesh Refinement\n";
    std::cout << "==============================================\n\n";
    
    std::cout << "STATUS: adaptive refinement in this tool is NOT IMPLEMENTED.\n";
    std::cout << "        main() never builds the initial mesh the refiner needs, so this\n";
    std::cout << "        program cannot produce a usable PLY. It now fails loudly instead\n";
    std::cout << "        of writing an empty mesh and reporting success.\n";
    std::cout << "        Working alternative: dtm_to_ply_delaunay --roi-center <lat> <lon>\n";
    std::cout << "        --roi-radius <km> --roi-max-points <n> --roi-core-frac <f>\n";
    std::cout << "        (see docs/dem-to-mesh-workflow.md).\n\n";

    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <input_dtm.tif> <output_mesh.ply> [options]\n";
        std::cout << "\nROI Extraction Options:\n";
        std::cout << "  --roi-center <lat> <lon>  : Center of region of interest\n";
        std::cout << "  --roi-radius <km>         : Radius of ROI in kilometers\n";
        std::cout << "  --roi-max-points <n>      : Maximum points in ROI\n";
        std::cout << "\nAdaptive Refinement Options:\n";
        std::cout << "  --refine-center <lat> <lon> : Center for refinement\n";
        std::cout << "  --refine-inner <km>         : Inner radius (full resolution)\n";
        std::cout << "  --refine-outer <km>         : Outer radius (min resolution)\n";
        std::cout << "  --refine-max-faces <n>      : Maximum number of faces\n";
        std::cout << "  --refine-min-edge <m>       : Minimum edge length in meters\n";
        std::cout << "  --refine-max-edge <m>       : Maximum edge length in meters\n";
        std::cout << "\nGeneral Options:\n";
        std::cout << "  --radius <meters>          : Body radius (default: 1737400)\n";
        std::cout << "  --initial-downsample <n>   : Initial downsampling (default: 10)\n";
        std::cout << "\nExamples:\n";
        std::cout << "  # Extract 10km ROI around specific location\n";
        std::cout << "  " << argv[0] << " dtm.tif roi_mesh.ply --roi-center -84.5 30.0 --roi-radius 10\n";
        std::cout << "  # Adaptive refinement with 1km inner, 20km outer radius\n";
        std::cout << "  " << argv[0] << " dtm.tif adaptive_mesh.ply --refine-center -84.5 30.0 ";
        std::cout << "--refine-inner 1 --refine-outer 20 --refine-max-faces 100000\n";
        return 1;
    }
    
    std::string input_file = argv[1];
    std::string output_file = argv[2];
    
    // Parse command line options
    bool use_roi = false;
    ROIExtractor::ROIParams roi_params;
    // Center/radius were left uninitialized in the original; a --roi-radius
    // without --roi-center (or vice versa) read indeterminate values.
    roi_params.center_lat = 0.0;
    roi_params.center_lon = 0.0;
    roi_params.radius_km = 0.0;
    roi_params.use_geodesic = true;
    roi_params.max_points = 0;  // No limit by default

    bool use_refinement = false;
    AdaptiveMeshRefiner::RefinementParams refine_params;
    // Same here: center and the two radii were uninitialized in the original.
    refine_params.center_lat = 0.0;
    refine_params.center_lon = 0.0;
    refine_params.inner_radius_km = 0.0;
    refine_params.outer_radius_km = 0.0;
    refine_params.max_faces = 100000;
    refine_params.min_edge_length_m = 10.0;
    refine_params.max_edge_length_m = 1000.0;
    refine_params.use_elevation_gradient = false;
    
    double body_radius = MOON_RADIUS;
    int initial_downsample = 10;
    
    for (int i = 3; i < argc; i++) {
        std::string arg = argv[i];
        
        // ROI options
        if (arg == "--roi-center" && i + 2 < argc) {
            use_roi = true;
            roi_params.center_lat = std::stod(argv[++i]);
            roi_params.center_lon = std::stod(argv[++i]);
        } else if (arg == "--roi-radius" && i + 1 < argc) {
            use_roi = true;
            roi_params.radius_km = std::stod(argv[++i]);
        } else if (arg == "--roi-max-points" && i + 1 < argc) {
            roi_params.max_points = std::stoi(argv[++i]);
        }
        // Refinement options
        else if (arg == "--refine-center" && i + 2 < argc) {
            use_refinement = true;
            refine_params.center_lat = std::stod(argv[++i]);
            refine_params.center_lon = std::stod(argv[++i]);
        } else if (arg == "--refine-inner" && i + 1 < argc) {
            use_refinement = true;
            refine_params.inner_radius_km = std::stod(argv[++i]);
        } else if (arg == "--refine-outer" && i + 1 < argc) {
            use_refinement = true;
            refine_params.outer_radius_km = std::stod(argv[++i]);
        } else if (arg == "--refine-max-faces" && i + 1 < argc) {
            refine_params.max_faces = std::stoi(argv[++i]);
        } else if (arg == "--refine-min-edge" && i + 1 < argc) {
            refine_params.min_edge_length_m = std::stod(argv[++i]);
        } else if (arg == "--refine-max-edge" && i + 1 < argc) {
            refine_params.max_edge_length_m = std::stod(argv[++i]);
        }
        // General options
        else if (arg == "--radius" && i + 1 < argc) {
            body_radius = std::stod(argv[++i]);
        } else if (arg == "--initial-downsample" && i + 1 < argc) {
            initial_downsample = std::stoi(argv[++i]);
            // Parsed, stored, and never read: the initial mesh it would have
            // strided is one of the two unfinished stubs described at the top
            // of this file.
            std::cerr << "WARNING: --initial-downsample is parsed but NEVER USED "
                         "(value " << initial_downsample << " has no effect).\n";
        } else {
            std::cerr << "WARNING: ignoring unrecognized or incomplete argument: "
                      << arg << "\n";
        }
    }
    
    // Initialize GDAL
    GDALAllRegister();
    
    // Open DTM dataset
    GDALDataset* dataset = (GDALDataset*)GDALOpen(input_file.c_str(), GA_ReadOnly);
    if (!dataset) {
        std::cerr << "Error: Could not open DTM file: " << input_file << std::endl;
        return 1;
    }
    
    std::cout << "Loaded DTM: " << input_file << std::endl;
    std::cout << "  Dimensions: " << dataset->GetRasterXSize() << " x " 
              << dataset->GetRasterYSize() << std::endl;
    
    // Process DTM based on options
    std::vector<AdaptiveMeshRefiner::Vertex> vertices;
    std::vector<AdaptiveMeshRefiner::Triangle> triangles;
    bool wrote_output = false;

    // Lambda for getting elevation from DTM
    auto getElevation = [](double lat, double lon) -> float {
        // Implementation would query the dataset
        // For now, return a placeholder
        //
        // UNFINISHED: this returns 0.0 for every query, so every subdivision
        // midpoint would land on a flat datum regardless of the terrain. It is
        // unreachable in practice because the initial-mesh stub below refuses
        // first, but it must be written before this tool can be trusted.
        // Whoever writes it: restore the `[&dataset]` capture, dropped here
        // only because an unused capture is a compiler warning.
        (void)lat;
        (void)lon;
        return 0.0f;
    };
    
    if (use_roi) {
        std::cout << "\n--- Extracting Region of Interest ---" << std::endl;
        std::cout << "Center: " << roi_params.center_lat << ", " << roi_params.center_lon << std::endl;
        std::cout << "Radius: " << roi_params.radius_km << " km" << std::endl;
        
        ROIExtractor extractor(body_radius);
        extractor.setParams(roi_params);

        // Convert ROI to initial mesh
        // ... (conversion code here)
        //
        // UNFINISHED: the stub above was never written, so whatever
        // extractFromDTM(dataset) produced was discarded and
        // `vertices`/`triangles` stayed empty. extractFromDTM is therefore
        // deliberately NOT called here -- running it would burn time on a
        // result nothing consumes, and it carries a latent defect while it is
        // dead code: it computes `roi_width * roi_height` as int*int and sizes
        // a vector from that, which overflows past 2^31 samples (a large ROI
        // over a large raster). Widen that arithmetic before wiring it up.
        std::cerr << "ERROR: ROI extraction is configured, but the ROI-to-mesh "
                     "conversion step is\n"
                     "       NOT IMPLEMENTED, so no mesh can be produced from "
                     "the extracted region.\n"
                     "       Use dtm_to_ply_delaunay --roi-center/--roi-radius/"
                     "--roi-max-points\n"
                     "       instead; see docs/dem-to-mesh-workflow.md.\n";
    }

    if (use_refinement) {
        std::cout << "\n--- Applying Adaptive Refinement ---" << std::endl;
        std::cout << "Center: " << refine_params.center_lat << ", " 
                  << refine_params.center_lon << std::endl;
        std::cout << "Inner radius: " << refine_params.inner_radius_km << " km" << std::endl;
        std::cout << "Outer radius: " << refine_params.outer_radius_km << " km" << std::endl;
        std::cout << "Max faces: " << refine_params.max_faces << std::endl;
        
        AdaptiveMeshRefiner refiner(body_radius);
        refiner.setParams(refine_params);
        
        // If no input mesh, create initial regular mesh
        if (vertices.empty()) {
            // Create initial mesh from DTM
            // ... (initial mesh generation)
            //
            // UNFINISHED: the stub above was never written. Without it the
            // refiner receives an empty mesh, subdivides nothing, and exports a
            // 0-vertex / 0-face PLY. Refuse rather than write that file.
            std::cerr << "\nERROR: adaptive refinement needs an initial mesh, and the\n"
                         "       'create initial mesh from DTM' step is NOT IMPLEMENTED.\n"
                         "       Refusing to write an empty mesh to: " << output_file << "\n"
                         "       Use dtm_to_ply_delaunay with --roi-center/--roi-radius/\n"
                         "       --roi-max-points/--roi-core-frac instead; it produces a real\n"
                         "       radially graded ROI mesh. See docs/dem-to-mesh-workflow.md.\n";
            GDALClose(dataset);
            return 2;
        }

        refiner.generateAdaptiveMesh(vertices, triangles, getElevation);
        // Unreachable while the initial-mesh stub above refuses first, but the
        // moment that stub is implemented an unwritable path or a full disk
        // must not fall through to "Processing completed successfully!" -- that
        // is the exact silent success this file was ported to eliminate.
        if (!refiner.exportToPLY(output_file)) {
            std::cerr << "\nERROR: failed to write the refined mesh to "
                      << output_file << "\n";
            GDALClose(dataset);
            return 1;
        }
        wrote_output = true;
    }

    // Clean up
    GDALClose(dataset);

    // Nothing above wrote a mesh (no --refine-* given, or only --roi-* which
    // cannot reach an output). Do not report success for an absent file.
    if (!wrote_output) {
        std::cerr << "\nERROR: no output mesh was written to " << output_file << ".\n"
                     "       This tool's mesh pipeline is unfinished (see the STATUS banner\n"
                     "       above). Nothing was produced.\n";
        return 2;
    }

    std::cout << "\nProcessing completed successfully!" << std::endl;
    std::cout << "Output file: " << output_file << std::endl;

    return 0;
}