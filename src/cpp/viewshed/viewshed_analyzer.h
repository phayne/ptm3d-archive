/**
 * ViewshedAnalyzer - Viewshed analysis module for PTM3D
 * 
 * Purpose: Compute viewsheds from specified observation points on lunar terrain
 * for mission planning and traverse optimization.
 * 
 * Features:
 * - Multi-format DEM support (PLY, OBJ, VTK, GeoTIFF)
 * - Single and batch viewshed computation
 * - Optimized ray tracing using Embree with AVX-512 support
 * - Multiple output formats (VTK/PVD time series, HDF5)
 * - Viewpoint visualization in output files
 * 
 * Author: PTM3D Development Team
 * Date: 2025
 */

#ifndef VIEWSHED_ANALYZER_H
#define VIEWSHED_ANALYZER_H

#include <vector>
#include <array>
#include <string>
#include <memory>
#include <map>
#include <set>
#include <thread>
#include <mutex>

// PTM3D includes
#include "../mesh/mesh_utils.h"
#include "../illumination/illumination_model.h"

// External includes
#include <embree4/rtcore.h>

#ifdef HAS_HDF5
#include <H5Cpp.h>
#endif

namespace ptm {

/**
 * Viewing location specification
 */
struct ViewLocation {
    enum Type {
        LAT_LON,        // Geographic coordinates
        FACE_INDEX,     // Direct mesh face index
        CARTESIAN       // Cartesian coordinates
    };
    
    Type type;
    double latitude;    // degrees
    double longitude;   // degrees
    double height;      // meters above surface
    int face_index;     // if type == FACE_INDEX
    std::array<double, 3> position; // if type == CARTESIAN
    std::string label;  // Optional label for this location
    std::string utc_timestamp;  // Optional UTC timestamp for this waypoint

    ViewLocation() : type(LAT_LON), latitude(0), longitude(0), height(2.0), face_index(-1) {}

    // Constructor for lat/lon
    ViewLocation(double lat, double lon, double h = 2.0, const std::string& lbl = "")
        : type(LAT_LON), latitude(lat), longitude(lon), height(h), face_index(-1), label(lbl) {}
};

/**
 * Viewshed computation parameters
 */
struct ViewshedParams {
    double max_range_km = 100.0;        // Maximum viewing range
    double angular_resolution = 0.1;    // degrees between ray samples
    double target_elevation = 0.0;      // Target height for visibility
    bool compute_horizon = true;        // Compute horizon profile
    int num_threads = 0;                // 0 = auto-detect
    
    // Ray tracing parameters
    double ray_offset_factor = 1e-4;   // Surface offset to avoid self-intersection
    bool use_avx512 = true;            // Use AVX-512 if available
    int batch_size = 16384;            // Ray batch size for processing
    
    // Filtering parameters
    double min_elevation_angle = -50.0; // Minimum elevation angle to consider
    double max_elevation_angle = 50.0;  // Maximum elevation angle to consider
};

/**
 * Viewshed result for a single observation point
 */
struct ViewshedResult {
    ViewLocation location;
    std::vector<float> visibility;      // Per-face visibility (0 or 1)
    std::vector<float> distance;        // Distance to visible faces (km)
    std::vector<float> elevation_angle; // Elevation angle to faces (degrees)
    std::vector<float> azimuth;        // Azimuth to faces (degrees)
    std::vector<float> emission_angle;  // Emission angle: LOS vs face normal (degrees)
    std::vector<float> incidence_cos;   // cos(emission_angle) for Lambertian calcs
    std::vector<float> face_area_m2;    // Geometric face area (m^2)

    // Viewpoint face index (for visualization)
    int viewpoint_face_index = -1;
    
    // Horizon profile (if computed)
    std::vector<std::pair<double, double>> horizon_profile; // (azimuth, elevation)
    
    // Statistics
    int num_visible_faces = 0;
    double visible_area_km2 = 0.0;
    double max_visible_range_km = 0.0;
    double mean_visible_range_km = 0.0;
    std::array<double, 4> visibility_bounds; // min_lat, max_lat, min_lon, max_lon
    
    // Quality metrics
    int rays_cast = 0;
    int rays_hit = 0;
    double computation_time_ms = 0.0;
};

/**
 * Main viewshed analyzer class
 */
class ViewshedAnalyzer {
public:
    ViewshedAnalyzer();
    ~ViewshedAnalyzer();
    
    // Initialization
    bool initialize(const std::string& dem_file, double body_radius = 1737400.0);
    bool initializeFromMesh(std::shared_ptr<Mesh> mesh);
    
    // Single viewshed computation
    ViewshedResult computeViewshed(
        const ViewLocation& location,
        const ViewshedParams& params = ViewshedParams());
    
    // Batch viewshed computation (for waypoints/traverse)
    std::vector<ViewshedResult> computeMultipleViewsheds(
        const std::vector<ViewLocation>& locations,
        const ViewshedParams& params = ViewshedParams());
    
    // Analyze visibility into specific region
    ViewshedResult analyzeRegionVisibility(
        const ViewLocation& observer,
        const std::vector<int>& target_faces,
        const ViewshedParams& params = ViewshedParams());
    
    // Find optimal viewing locations for target region
    std::vector<ViewLocation> findOptimalViewpoints(
        const std::vector<int>& target_faces,  // Faces we want to observe
        int num_viewpoints = 10,
        const ViewshedParams& params = ViewshedParams());
    
    // Output methods
    bool exportToVTK(
        const std::string& filename,
        const ViewshedResult& result,
        bool mark_viewpoint = true) const;
    
    bool exportBatchToVTK(
        const std::string& base_filename,
        const std::vector<ViewshedResult>& results,
        bool mark_viewpoints = true) const;
    
    bool exportToPVD(
        const std::string& pvd_filename,
        const std::vector<ViewshedResult>& results,
        bool mark_viewpoints = true) const;
    
#ifdef HAS_HDF5
    bool exportToHDF5(
        const std::string& filename,
        const std::vector<ViewshedResult>& results) const;
#endif
    
    // Utility methods
    std::vector<int> getFacesInRegion(
        double center_lat, double center_lon, 
        double radius_km) const;
    
    std::vector<int> getFacesBelowTemperature(
        double max_temp_k,
        const std::vector<double>& temperatures) const;
    
    std::vector<int> getFacesFromMask(
        const std::vector<bool>& mask) const;
    
    bool loadFaceSelectionFromFile(const std::string& filename);
    bool saveFaceSelectionToFile(
        const std::string& filename,
        const std::vector<int>& faces) const;
    
    // Load viewpoint locations from CSV file
    std::vector<ViewLocation> loadViewpointsFromFile(
        const std::string& filename) const;
    
    // Getters
    std::shared_ptr<Mesh> getMesh() const { return mesh_; }
    bool isInitialized() const { return initialized_; }
    size_t getNumFaces() const { return mesh_ ? mesh_->faces.size() : 0; }
    
    // Find closest mesh face to a lat/lon location
    int findClosestFace(double lat, double lon) const;
    
private:
    // Core computation methods
    void computeSingleViewshed(
        const std::array<double, 3>& observer_pos,
        int observer_face_idx,
        ViewshedResult& result,
        const ViewshedParams& params);
    
    void castRaysToFaces(
        const std::array<double, 3>& observer_pos,
        const std::vector<int>& target_faces,
        std::vector<int>& visible_faces,
        std::vector<float>& distances,
        const ViewshedParams& params);
    
    void computeHorizonProfile(
        const std::array<double, 3>& observer_pos,
        ViewshedResult& result,
        const ViewshedParams& params);
    
    // Ray tracing helpers
    std::array<double, 3> calculateSafeRayOrigin(
        const std::array<double, 3>& surface_point,
        const std::array<double, 3>& normal,
        double offset_factor) const;
    
    bool isRayBlocked(
        const std::array<double, 3>& origin,
        const std::array<double, 3>& direction,
        double max_distance) const;
    
    // Cast rays; SIMD if available
    std::vector<int> batchRaycast(
        const std::vector<std::array<double, 3>>& origins,
        const std::vector<std::array<double, 3>>& directions,
        bool use_simd = true) const;

    // Overload with per-ray max distance (tfar) in meters
    std::vector<int> batchRaycast(
        const std::vector<std::array<double, 3>>& origins,
        const std::vector<std::array<double, 3>>& directions,
        const std::vector<double>& target_distances,
        bool use_simd = true) const;

    // Optional: return both hit face and distance; used by some call sites
    std::vector<std::pair<int, double>> batchRaycastWithDistance(
        const std::vector<std::array<double, 3>>& origins,
        const std::vector<std::array<double, 3>>& directions,
        const std::vector<double>& target_distances,
        bool use_simd = true) const;
    
    // Coordinate transformations
    std::array<double, 3> latLonToCartesian(
        double lat, double lon, double elevation = 0.0) const;
    
    std::pair<double, double> cartesianToLatLon(
        const std::array<double, 3>& position) const;
    
    ViewLocation convertToFaceIndex(const ViewLocation& location) const;
    
    // Statistics computation
    void computeStatistics(ViewshedResult& result) const;
    
    // Threading utilities
    void processViewshedBatch(
        const std::vector<ViewLocation>& locations,
        std::vector<ViewshedResult>& results,
        const ViewshedParams& params,
        size_t start_idx,
        size_t end_idx);
    
private:
    // Member variables
    std::shared_ptr<Mesh> mesh_;
    std::shared_ptr<MeshUtils> mesh_utils_;
    std::unique_ptr<IlluminationModel> illumination_model_;
    
    RTCDevice embree_device_;
    RTCScene embree_scene_;
    
    bool initialized_;
    double body_radius_;
    
    // Thread pool for batch processing
    mutable std::mutex result_mutex_;
    int hardware_threads_;
    bool has_avx512_;
    
    // Performance monitoring
    mutable size_t total_rays_cast_;
    mutable size_t total_rays_hit_;
    mutable double total_computation_time_;
    
    // Coordinate frame detection and transforms
    enum class CoordFrame { ECEF, LOCAL_ENU };
    CoordFrame coord_frame_ = CoordFrame::ECEF;
    // ENU anchor (origin) and transforms if using LOCAL_ENU
    double enu_ref_lat_ = 0.0, enu_ref_lon_ = 0.0;
    std::array<double,3> enu_ref_ecef_{0.0,0.0,0.0};
    std::array<std::array<double,3>,3> R_ecef2enu_{{ {1,0,0},{0,1,0},{0,0,1} }};
    std::array<std::array<double,3>,3> R_enu2ecef_{{ {1,0,0},{0,1,0},{0,0,1} }};

    // Helpers to set ENU basis and convert between ECEF/ENU
    void setLocalENUFrame(double ref_lat_deg, double ref_lon_deg);
    static std::array<double,3> ecefFromLatLon(double lat_deg, double lon_deg, double alt_m, double radius);
    static void enuAxesFromLatLon(double lat_deg, double lon_deg,
                                  std::array<double,3>& e_east,
                                  std::array<double,3>& e_north,
                                  std::array<double,3>& e_up);
    std::array<double,3> ecefToEnu(const std::array<double,3>& ecef) const;
    std::array<double,3> enuToEcef(const std::array<double,3>& enu) const;
};

/**
 * Utility functions for viewshed analysis
 */
namespace ViewshedUtils {
    // Compute viewshed overlap between multiple observation points
    std::vector<float> computeViewshedOverlap(
        const std::vector<ViewshedResult>& results);
    
    // Find minimum set of viewpoints to cover target area
    std::vector<int> findMinimumCoveringSet(
        const std::vector<ViewshedResult>& results,
        const std::vector<int>& target_faces,
        double coverage_threshold = 0.95);
    
    // Rank viewpoints by coverage of target region
    std::vector<std::pair<int, double>> rankByTargetVisibility(
        const std::vector<ViewshedResult>& results,
        const std::vector<int>& target_faces);
    
    // Compute cumulative coverage along traverse (waypoints)
    std::vector<float> computeCumulativeCoverage(
        const std::vector<ViewshedResult>& traverse_results);
    
    // Generate intermediate waypoints between two locations
    std::vector<ViewLocation> interpolateWaypoints(
        const ViewLocation& start,
        const ViewLocation& end,
        int num_points = 10);
}

} // namespace ptm

#endif // VIEWSHED_ANALYZER_H