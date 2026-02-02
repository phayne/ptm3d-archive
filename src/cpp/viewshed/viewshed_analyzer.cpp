/**
 * ViewshedAnalyzer Implementation
 * High-performance viewshed computation for lunar terrain analysis
 * 
 * Simplified version focusing on general region-based visibility analysis
 * with support for viewpoint visualization in outputs
 */

#include "viewshed_analyzer.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iomanip>
#include <future>
#include <filesystem>
#include <limits>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

// For CPU feature detection
#ifdef __x86_64__
#include <cpuid.h>
#include <immintrin.h>
#endif

namespace fs = std::filesystem;

namespace ptm {

// Internal helpers for VTU/PVD writers
namespace {
static std::string ensure_vtu_extension(const std::string& path);
static bool writeVTU_UnstructuredGrid(
    const std::string& vtu_path,
    const std::shared_ptr<Mesh>& mesh,
    const ViewshedResult& result,
    double time_value,
    bool mark_viewpoint);
}

// Constants
constexpr double MOON_RADIUS = 1737400.0;  // meters
constexpr double DEG_TO_RAD = M_PI / 180.0;
constexpr double RAD_TO_DEG = 180.0 / M_PI;
constexpr double EPSILON = 1e-10;

// ------------ NEW: geometry helpers to ensure outward normals and unit consistency ------------
namespace {

inline double vec_len(const std::array<double,3>& v) {
    return std::sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
}
inline std::array<double,3> vec_norm(const std::array<double,3>& v) {
    double L = vec_len(v);
    if (L <= 0.0) return {0,0,1};
    return {v[0]/L, v[1]/L, v[2]/L};
}
inline std::array<double,3> vec_cross(const std::array<double,3>& a, const std::array<double,3>& b) {
    return { a[1]*b[2]-a[2]*b[1],
             a[2]*b[0]-a[0]*b[2],
             a[0]*b[1]-a[1]*b[0] };
}
inline double vec_dot(const std::array<double,3>& a, const std::array<double,3>& b) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

// Compute centers/normals if missing and ensure normals point outward (away from origin)
static void ensure_face_props_outward(std::shared_ptr<Mesh>& mesh) {
    const size_t nF = mesh->faces.size();
    if (mesh->face_centers.size() != nF) mesh->face_centers.assign(nF, {0,0,0});
    if (mesh->face_normals.size() != nF) mesh->face_normals.assign(nF, {0,0,1});

    for (size_t i = 0; i < nF; ++i) {
        const auto& f = mesh->faces[i];
        const auto& a = mesh->vertices[f[0]];
        const auto& b = mesh->vertices[f[1]];
        const auto& c = mesh->vertices[f[2]];

        // center
        std::array<double,3> center = { (a[0]+b[0]+c[0])/3.0,
                                        (a[1]+b[1]+c[1])/3.0,
                                        (a[2]+b[2]+c[2])/3.0 };
        mesh->face_centers[i] = center;

        // normal (unnormalized)
        std::array<double,3> ab = { b[0]-a[0], b[1]-a[1], b[2]-a[2] };
        std::array<double,3> ac = { c[0]-a[0], c[1]-a[1], c[2]-a[2] };
        std::array<double,3> n = vec_cross(ab, ac);
        // If degenerate, keep default
        if (vec_len(n) <= 0.0) n = {0,0,1};
        n = vec_norm(n);

        // Flip inward normals: outward means dot(center, normal) > 0
        if (vec_dot(center, n) < 0.0) {
            n[0] = -n[0]; n[1] = -n[1]; n[2] = -n[2];
        }
        mesh->face_normals[i] = n;
    }
}

// Estimate mesh mean radius (for unit consistency and offsets)
static double estimate_mesh_radius(const std::shared_ptr<Mesh>& mesh) {
    if (!mesh || mesh->vertices.empty()) return MOON_RADIUS;
    long double sum = 0.0L;
    size_t cnt = 0;
    // Sample vertices (all or strided for speed)
    const size_t step = std::max<size_t>(1, mesh->vertices.size() / 100000); // sample up to 1e5
    for (size_t i = 0; i < mesh->vertices.size(); i += step) {
        const auto& v = mesh->vertices[i];
        sum += std::sqrt((long double)v[0]*v[0] + (long double)v[1]*v[1] + (long double)v[2]*v[2]);
        ++cnt;
    }
    if (cnt == 0) return MOON_RADIUS;
    return static_cast<double>(sum / cnt);
}

} // anonymous
// ------------------------------------------------------------------------------------------------

// ------------ helpers for ENU/ECEF transforms ------------
static inline std::array<double,3> matTmulVec(const std::array<std::array<double,3>,3>& M,
                                              const std::array<double,3>& v) {
    return {
        M[0][0]*v[0] + M[0][1]*v[1] + M[0][2]*v[2],
        M[1][0]*v[0] + M[1][1]*v[1] + M[1][2]*v[2],
        M[2][0]*v[0] + M[2][1]*v[1] + M[2][2]*v[2]
    };
}
static inline std::array<std::array<double,3>,3> matTranspose(const std::array<std::array<double,3>,3>& A) {
    return {{
        {A[0][0], A[1][0], A[2][0]},
        {A[0][1], A[1][1], A[2][1]},
        {A[0][2], A[1][2], A[2][2]}
    }};
}

void ViewshedAnalyzer::enuAxesFromLatLon(double lat_deg, double lon_deg,
                                         std::array<double,3>& e_east,
                                         std::array<double,3>& e_north,
                                         std::array<double,3>& e_up) {
    const double lat = lat_deg * DEG_TO_RAD;
    const double lon = lon_deg * DEG_TO_RAD;
    e_up    = { std::cos(lat)*std::cos(lon), std::cos(lat)*std::sin(lon), std::sin(lat) };
    e_east  = { -std::sin(lon), std::cos(lon), 0.0 };
    e_north = { -std::sin(lat)*std::cos(lon), -std::sin(lat)*std::sin(lon), std::cos(lat) };
}

std::array<double,3> ViewshedAnalyzer::ecefFromLatLon(double lat_deg, double lon_deg, double alt_m, double radius) {
    const double lat = lat_deg * DEG_TO_RAD;
    const double lon = lon_deg * DEG_TO_RAD;
    const double r = radius + alt_m;
    return { r * std::cos(lat) * std::cos(lon),
             r * std::cos(lat) * std::sin(lon),
             r * std::sin(lat) };
}

void ViewshedAnalyzer::setLocalENUFrame(double ref_lat_deg, double ref_lon_deg) {
    enu_ref_lat_ = ref_lat_deg;
    enu_ref_lon_ = ref_lon_deg;
    enu_ref_ecef_ = ecefFromLatLon(ref_lat_deg, ref_lon_deg, 0.0, body_radius_);
    std::array<double,3> eE, eN, eU;
    enuAxesFromLatLon(ref_lat_deg, ref_lon_deg, eE, eN, eU);
    // Rows of R_ecef2enu are ENU basis in ECEF
    R_ecef2enu_ = {{ eE, eN, eU }};
    R_enu2ecef_ = matTranspose(R_ecef2enu_);
}

std::array<double,3> ViewshedAnalyzer::ecefToEnu(const std::array<double,3>& ecef) const {
    std::array<double,3> d = { ecef[0]-enu_ref_ecef_[0], ecef[1]-enu_ref_ecef_[1], ecef[2]-enu_ref_ecef_[2] };
    return matTmulVec(R_ecef2enu_, d);
}

std::array<double,3> ViewshedAnalyzer::enuToEcef(const std::array<double,3>& enu) const {
    auto d = matTmulVec(R_enu2ecef_, enu);
    return { d[0]+enu_ref_ecef_[0], d[1]+enu_ref_ecef_[1], d[2]+enu_ref_ecef_[2] };
}

// ------------------------------------------------------------------------------------------------

// Constructor
ViewshedAnalyzer::ViewshedAnalyzer() 
    : initialized_(false),
      body_radius_(MOON_RADIUS),
      embree_device_(nullptr),
      embree_scene_(nullptr),
      hardware_threads_(std::thread::hardware_concurrency()),
      has_avx512_(false),
      total_rays_cast_(0),
      total_rays_hit_(0),
      total_computation_time_(0.0) {
    
    // Detect AVX-512 support
#ifdef __x86_64__
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid_max(0, nullptr) >= 7) {
        __cpuid_count(7, 0, eax, ebx, ecx, edx);
        has_avx512_ = (ebx & (1 << 16)) != 0;  // AVX512F
    }
#endif
    
    if (has_avx512_) {
        std::cout << "AVX-512 support detected and enabled" << std::endl;
    }
}

// Destructor
ViewshedAnalyzer::~ViewshedAnalyzer() {
    if (embree_scene_) {
        rtcReleaseScene(embree_scene_);
    }
    if (embree_device_) {
        rtcReleaseDevice(embree_device_);
    }
}

// Initialize from DEM file
bool ViewshedAnalyzer::initialize(const std::string& dem_file, double body_radius) {
    body_radius_ = body_radius;
    
    // Create mesh utilities
    mesh_utils_ = std::make_shared<MeshUtils>();
    
    // Determine file type and load accordingly
    std::string extension = dem_file.substr(dem_file.find_last_of(".") + 1);
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
    
    bool success = false;
    
    if (extension == "tif" || extension == "tiff" || extension == "geotiff") {
        // Use DTM importer for GeoTIFF
        success = mesh_utils_->importDEMData(dem_file, true, body_radius);
    } else if (extension == "ply" || extension == "obj" || extension == "vtk" || 
               extension == "vtu" || extension == "vtp") {
        // Use mesh importer for standard formats
        success = mesh_utils_->importTriangulatedMesh(dem_file);
    } else {
        std::cerr << "Unsupported file format: " << extension << std::endl;
        return false;
    }
    
    if (!success) {
        std::cerr << "Failed to load DEM/mesh from: " << dem_file << std::endl;
        return false;
    }
    
    mesh_ = mesh_utils_->getMesh();
    if (!mesh_ || mesh_->vertices.empty() || mesh_->faces.empty()) {
        std::cerr << "Invalid mesh data" << std::endl;
        return false;
    }

    // Preliminary radius before any calibration
    const double mesh_R_pre = estimate_mesh_radius(mesh_);

    // Detect coordinate frame: ECEF-like if near body radius, else LOCAL_ENU
    if (mesh_R_pre > 0.5 * body_radius_ && mesh_R_pre < 2.0 * body_radius_) {
        coord_frame_ = CoordFrame::ECEF;
    } else {
        coord_frame_ = CoordFrame::LOCAL_ENU;
    }

    // Ensure face centers/normals (orientation adjust later per frame)
    ensure_face_props_outward(mesh_);

    // Calibrate radius only for ECEF meshes
    if (coord_frame_ == CoordFrame::ECEF) {
        if (mesh_R_pre > 0.0) {
            double ratio = mesh_R_pre / body_radius_;
            if (ratio < 0.8 || ratio > 1.2) {
                std::cerr << "Warning: body_radius (" << body_radius_
                          << ") differs from mesh radius (" << mesh_R_pre
                          << "); using mesh radius." << std::endl;
                body_radius_ = mesh_R_pre;
            }
        }
    } else {
        // LOCAL_ENU: set ENU anchor from mesh mean lat/lon if available
        double mean_lat = 0.0, mean_lon = 0.0;
        if (!mesh_->latitudes.empty() && !mesh_->longitudes.empty()
            && mesh_->latitudes.size() == mesh_->faces.size()
            && mesh_->longitudes.size() == mesh_->faces.size()) {
            for (size_t i = 0; i < mesh_->faces.size(); ++i) {
                mean_lat += mesh_->latitudes[i];
                mean_lon += mesh_->longitudes[i];
            }
            mean_lat /= mesh_->faces.size();
            mean_lon /= mesh_->faces.size();
        }
        setLocalENUFrame(mean_lat, mean_lon);
    }

    // Initialize illumination model for ray tracing
    illumination_model_ = std::make_unique<IlluminationModel>();
    
    // Initialize Embree
    embree_device_ = rtcNewDevice(nullptr);
    if (!embree_device_) {
        std::cerr << "Failed to create Embree device" << std::endl;
        return false;
    }
    
    // Create Embree scene
    embree_scene_ = rtcNewScene(embree_device_);
    RTCGeometry geom = rtcNewGeometry(embree_device_, RTC_GEOMETRY_TYPE_TRIANGLE);
    
    // Set vertices
    float* vertices = (float*)rtcSetNewGeometryBuffer(
        geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3,
        3 * sizeof(float), mesh_->vertices.size());
    
    for (size_t i = 0; i < mesh_->vertices.size(); ++i) {
        vertices[3*i + 0] = static_cast<float>(mesh_->vertices[i][0]);
        vertices[3*i + 1] = static_cast<float>(mesh_->vertices[i][1]);
        vertices[3*i + 2] = static_cast<float>(mesh_->vertices[i][2]);
    }
    
    // Set triangles
    unsigned* triangles = (unsigned*)rtcSetNewGeometryBuffer(
        geom, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3,
        3 * sizeof(unsigned), mesh_->faces.size());
    
    for (size_t i = 0; i < mesh_->faces.size(); ++i) {
        triangles[3*i + 0] = mesh_->faces[i][0];
        triangles[3*i + 1] = mesh_->faces[i][1];
        triangles[3*i + 2] = mesh_->faces[i][2];
    }
    
    rtcCommitGeometry(geom);
    rtcAttachGeometry(embree_scene_, geom);
    rtcReleaseGeometry(geom);
    rtcCommitScene(embree_scene_);
    
    initialized_ = true;
    
    std::cout << "ViewshedAnalyzer initialized with " << mesh_->vertices.size() 
              << " vertices and " << mesh_->faces.size() << " faces" << std::endl;
    std::cout << "Hardware: " << hardware_threads_ << " threads, "
              << (has_avx512_ ? "AVX-512 enabled" : "AVX-512 disabled") << std::endl;
    
    return true;
}

// Initialize from existing mesh
bool ViewshedAnalyzer::initializeFromMesh(std::shared_ptr<Mesh> mesh) {
    if (!mesh || mesh->vertices.empty() || mesh->faces.empty()) {
        std::cerr << "Invalid mesh provided" << std::endl;
        return false;
    }
    mesh_ = mesh;
    // Detect frame with current body_radius_
    const double mesh_R_pre = estimate_mesh_radius(mesh_);
    coord_frame_ = (mesh_R_pre > 0.5 * body_radius_ && mesh_R_pre < 2.0 * body_radius_)
                   ? CoordFrame::ECEF : CoordFrame::LOCAL_ENU;

    ensure_face_props_outward(mesh_);

    if (coord_frame_ == CoordFrame::ECEF) {
        if (mesh_R_pre > 0.0) {
            double ratio = mesh_R_pre / body_radius_;
            if (ratio < 0.8 || ratio > 1.2) body_radius_ = mesh_R_pre;
        }
    } else {
        double mean_lat = 0.0, mean_lon = 0.0;
        if (!mesh_->latitudes.empty() && !mesh_->longitudes.empty()
            && mesh_->latitudes.size() == mesh_->faces.size()
            && mesh_->longitudes.size() == mesh_->faces.size()) {
            for (size_t i = 0; i < mesh_->faces.size(); ++i) {
                mean_lat += mesh_->latitudes[i];
                mean_lon += mesh_->longitudes[i];
            }
            mean_lat /= mesh_->faces.size();
            mean_lon /= mesh_->faces.size();
        }
        setLocalENUFrame(mean_lat, mean_lon);
    }

    // Initialize Embree (similar to above)
    embree_device_ = rtcNewDevice(nullptr);
    if (!embree_device_) {
        std::cerr << "Failed to create Embree device" << std::endl;
        return false;
    }
    
    embree_scene_ = rtcNewScene(embree_device_);
    RTCGeometry geom = rtcNewGeometry(embree_device_, RTC_GEOMETRY_TYPE_TRIANGLE);
    
    float* vertices = (float*)rtcSetNewGeometryBuffer(
        geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3,
        3 * sizeof(float), mesh_->vertices.size());
    
    for (size_t i = 0; i < mesh_->vertices.size(); ++i) {
        vertices[3*i + 0] = static_cast<float>(mesh_->vertices[i][0]);
        vertices[3*i + 1] = static_cast<float>(mesh_->vertices[i][1]);
        vertices[3*i + 2] = static_cast<float>(mesh_->vertices[i][2]);
    }
    
    unsigned* triangles = (unsigned*)rtcSetNewGeometryBuffer(
        geom, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3,
        3 * sizeof(unsigned), mesh_->faces.size());
    
    for (size_t i = 0; i < mesh_->faces.size(); ++i) {
        triangles[3*i + 0] = mesh_->faces[i][0];
        triangles[3*i + 1] = mesh_->faces[i][1];
        triangles[3*i + 2] = mesh_->faces[i][2];
    }
    
    rtcCommitGeometry(geom);
    rtcAttachGeometry(embree_scene_, geom);
    rtcReleaseGeometry(geom);
    rtcCommitScene(embree_scene_);
    
    initialized_ = true;
    return true;
}

// Compute single viewshed
ViewshedResult ViewshedAnalyzer::computeViewshed(
    const ViewLocation& location,
    const ViewshedParams& params) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    ViewshedResult result;
    result.location = location;
    
    if (!initialized_) {
        std::cerr << "ViewshedAnalyzer not initialized" << std::endl;
        return result;
    }
    
    // Convert location to Cartesian coordinates and find corresponding face
    std::array<double, 3> observer_pos;
    int observer_face_idx = -1;
    
    switch (location.type) {
        case ViewLocation::LAT_LON: {
            // Find closest face by lat/lon on the mesh (robust to frame)
            observer_face_idx = findClosestFace(location.latitude, location.longitude);
            if (observer_face_idx >= 0 && observer_face_idx < static_cast<int>(mesh_->faces.size())) {
                // Place observer ON the mesh: face center + outward normal * height (meters)
                const auto& center = mesh_->face_centers[observer_face_idx];
                // Ensure outward normal (frame-aware)
                std::array<double,3> fn = mesh_->face_normals[observer_face_idx];
                if (coord_frame_ == CoordFrame::ECEF) {
                    if ((center[0]*fn[0] + center[1]*fn[1] + center[2]*fn[2]) < 0.0) {
                        fn = { -fn[0], -fn[1], -fn[2] };
                    }
                } else {
                    if (fn[2] < 0.0) {
                        fn = { -fn[0], -fn[1], -fn[2] };
                    }
                }
                observer_pos = {
                    center[0] + fn[0] * location.height,
                    center[1] + fn[1] * location.height,
                    center[2] + fn[2] * location.height
                };
            } else {
                // Fallback: convert lat/lon to Cartesian if no face found
                observer_pos = latLonToCartesian(location.latitude, location.longitude, location.height);
            }
            break;
        }
            
        case ViewLocation::FACE_INDEX: {
            observer_face_idx = location.face_index;
            if (observer_face_idx >= 0 && observer_face_idx < static_cast<int>(mesh_->faces.size())) {
                const auto& center = mesh_->face_centers[observer_face_idx];
                const auto& normal = mesh_->face_normals[observer_face_idx];
                observer_pos = {
                    center[0] + normal[0] * location.height,
                    center[1] + normal[1] * location.height,
                    center[2] + normal[2] * location.height
                };
            } else {
                std::cerr << "Invalid face index: " << observer_face_idx << std::endl;
                return result;
            }
            break;
        }
            
        case ViewLocation::CARTESIAN: {
            observer_pos = location.position;
            // Find closest face to this position for orientation/marking
            double min_dist = std::numeric_limits<double>::max();
            for (size_t i = 0; i < mesh_->face_centers.size(); ++i) {
                const auto& center = mesh_->face_centers[i];
                double dx = center[0] - observer_pos[0];
                double dy = center[1] - observer_pos[1];
                double dz = center[2] - observer_pos[2];
                double dist = dx*dx + dy*dy + dz*dz;
                if (dist < min_dist) {
                    min_dist = dist;
                    observer_face_idx = static_cast<int>(i);
                }
            }
            break;
        }
    }
    
    // Store viewpoint face index for visualization
    result.viewpoint_face_index = observer_face_idx;
    
    // Initialize result arrays
    result.visibility.resize(mesh_->faces.size(), 0.0f);
    result.distance.resize(mesh_->faces.size(), -1.0f);
    result.elevation_angle.resize(mesh_->faces.size(), 0.0f);
    result.azimuth.resize(mesh_->faces.size(), 0.0f);
    result.emission_angle.resize(mesh_->faces.size(), -1.0f);
    result.incidence_cos.resize(mesh_->faces.size(), 0.0f);
    result.face_area_m2.resize(mesh_->faces.size(), 0.0f);
    
    // Perform viewshed computation
    computeSingleViewshed(observer_pos, observer_face_idx, result, params);
    
    // Compute horizon profile if requested
    if (params.compute_horizon) {
        computeHorizonProfile(observer_pos, result, params);
    }
    
    // Compute statistics
    computeStatistics(result);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    result.computation_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    return result;
}

// Core single viewshed computation
void ViewshedAnalyzer::computeSingleViewshed(
    const std::array<double, 3>& observer_pos,
    int observer_face_idx,
    ViewshedResult& result,
    const ViewshedParams& params) {

    const double max_range_m = params.max_range_km * 1000.0;
    std::cout << "Computing viewshed from face index " << observer_face_idx
              << " with max range " << params.max_range_km << " km" << std::endl;
    std::cout << "Angular resolution: " << params.angular_resolution << " deg" << std::endl;
    
    // Helpers
    auto norm3 = [](const std::array<double,3>& v){
        double L = std::sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
        return (L > 0.0) ? std::array<double,3>{v[0]/L, v[1]/L, v[2]/L} : std::array<double,3>{0.0,0.0,1.0};
    };
    auto dot3 = [](const std::array<double,3>& a, const std::array<double,3>& b){
        return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
    };
    auto cross3 = [](const std::array<double,3>& a, const std::array<double,3>& b){
        return std::array<double,3>{
            a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0]
        };
    };

    // Outward radial at observer
    std::array<double,3> r_hat = (coord_frame_ == CoordFrame::ECEF)
    ? vec_norm(observer_pos)
    : std::array<double,3>{0.0, 0.0, 1.0};

    // Observer normal (use face normal if available; flip to outward by dot with position)
    std::array<double, 3> observer_normal = r_hat;
    if (observer_face_idx >= 0 && static_cast<size_t>(observer_face_idx) < mesh_->face_normals.size()) {
        observer_normal = mesh_->face_normals[observer_face_idx];
        if (std::abs(observer_normal[0])+std::abs(observer_normal[1])+std::abs(observer_normal[2]) < 1e-12)
            observer_normal = r_hat;
        if (vec_dot(observer_normal, observer_pos) < 0.0) {
            observer_normal[0] = -observer_normal[0];
            observer_normal[1] = -observer_normal[1];
            observer_normal[2] = -observer_normal[2];
        }
        observer_normal = vec_norm(observer_normal);
    }

    // Local ENU frame
    std::array<double,3> up = observer_normal;
    std::array<double,3> ez = {0.0, 0.0, 1.0};
    std::array<double,3> ex = {1.0, 0.0, 0.0};
    std::array<double,3> east = vec_cross(ez, up);
    if (std::abs(east[0])+std::abs(east[1])+std::abs(east[2]) < 1e-10) east = vec_cross(ex, up);
    east = vec_norm(east);
    std::array<double,3> north = vec_norm(vec_cross(up, east));

    // Use mesh radius for offset scale to tolerate unit mismatches
    const double R_for_offset = std::max(body_radius_, estimate_mesh_radius(mesh_));
    const double offset_factor = (params.ray_offset_factor > 0.0) ? params.ray_offset_factor : 1e-4;
    const double safe_offset = std::max(1e-6 * R_for_offset, offset_factor * R_for_offset);

    // Offset origin outward
    std::array<double, 3> ray_origin = {
        observer_pos[0] + observer_normal[0] * safe_offset,
        observer_pos[1] + observer_normal[1] * safe_offset,
        observer_pos[2] + observer_normal[2] * safe_offset
    };

    // Prepare batches
    std::vector<std::array<double, 3>> ray_origins;
    std::vector<std::array<double, 3>> ray_directions;
    std::vector<double> target_distances;
    std::vector<int> face_indices;

    ray_origins.reserve(mesh_->faces.size());
    ray_directions.reserve(mesh_->faces.size());
    target_distances.reserve(mesh_->faces.size());
    face_indices.reserve(mesh_->faces.size());

    for (size_t face_idx = 0; face_idx < mesh_->faces.size(); ++face_idx) {
        const auto& face_center = mesh_->face_centers[face_idx];

        // Direction and distance FROM ray_origin to target center
        std::array<double, 3> d = {
            face_center[0] - ray_origin[0],
            face_center[1] - ray_origin[1],
            face_center[2] - ray_origin[2]
        };
        double L = vec_len(d);
        if (L <= 0.0) continue;

        if (max_range_m > 0.0 && L > max_range_m) continue;

        if (static_cast<int>(face_idx) == observer_face_idx) {
            result.visibility[face_idx] = 1.0f;
            result.distance[face_idx] = 0.0f;
            continue;
        }

        // Normalize
        d[0] /= L; d[1] /= L; d[2] /= L;

        // Outward-oriented face normal (robust even if mesh normals are inconsistent)
        std::array<double,3> fn = mesh_->face_normals[face_idx];
        if (coord_frame_ == CoordFrame::ECEF) {
            if (vec_dot(fn, face_center) < 0.0) { fn[0] = -fn[0]; fn[1] = -fn[1]; fn[2] = -fn[2]; }
        } else {
            if (fn[2] < 0.0) { fn[0] = -fn[0]; fn[1] = -fn[1]; fn[2] = -fn[2]; }
        }
        fn = vec_norm(fn);

        // Backface culling against ray origin: require fn · (-d) > 0 (i.e., fn·d < 0)
        if (fn[0]*(-d[0]) + fn[1]*(-d[1]) + fn[2]*(-d[2]) <= 0.0) continue;

        // Add to batch
        ray_origins.push_back(ray_origin);
        ray_directions.push_back(d);
        target_distances.push_back(L);
        face_indices.push_back(static_cast<int>(face_idx));

        if (ray_origins.size() >= params.batch_size) {
            auto hits = batchRaycast(ray_origins, ray_directions, target_distances,
                                     params.use_avx512 && has_avx512_);

            for (size_t i = 0; i < hits.size(); ++i) {
                int fidx = face_indices[i];
                if (hits[i] == fidx) {
                    result.visibility[fidx] = 1.0f;

                    const auto& c = mesh_->face_centers[fidx];
                    double vx = c[0] - observer_pos[0];
                    double vy = c[1] - observer_pos[1];
                    double vz = c[2] - observer_pos[2];
                    const double vlen = std::sqrt(vx*vx + vy*vy + vz*vz);
                    result.distance[fidx] = static_cast<float>(vlen / 1000.0);

                    const double v_up = vx*up[0] + vy*up[1] + vz*up[2];
                    const double v_h = std::sqrt(std::max(0.0, vlen*vlen - v_up*v_up));
                    result.elevation_angle[fidx] = static_cast<float>(std::atan2(v_up, v_h) * RAD_TO_DEG);

                    if (result.elevation_angle[fidx] < params.min_elevation_angle ||
                        result.elevation_angle[fidx] > params.max_elevation_angle) {
                        result.visibility[fidx] = 0.0f;
                        result.distance[fidx] = -1.0f;
                        continue;
                    }

                    const double v_e = vx*east[0] + vy*east[1] + vz*east[2];
                    const double v_n = vx*north[0] + vy*north[1] + vz*north[2];
                    double az = std::atan2(v_e, v_n) * RAD_TO_DEG;
                    if (az < 0) az += 360.0;
                    result.azimuth[fidx] = static_cast<float>(az);

                    // Emission angle: angle between -LOS and outward face normal
                    const auto& face_n = mesh_->face_normals[fidx];
                    double dx = vx/vlen, dy = vy/vlen, dz = vz/vlen;
                    double cos_em = -(dx*face_n[0] + dy*face_n[1] + dz*face_n[2]);
                    cos_em = std::max(0.0, std::min(1.0, cos_em));
                    result.emission_angle[fidx] = static_cast<float>(std::acos(cos_em) * RAD_TO_DEG);
                    result.incidence_cos[fidx] = static_cast<float>(cos_em);
                }
            }
            ray_origins.clear();
            ray_directions.clear();
            target_distances.clear();
            face_indices.clear();
        }
    }

    // Process remaining
    if (!ray_origins.empty()) {
        auto hits = batchRaycast(ray_origins, ray_directions, target_distances,
                                 params.use_avx512 && has_avx512_);
        for (size_t i = 0; i < hits.size(); ++i) {
            int fidx = face_indices[i];
            if (hits[i] == fidx) {
                result.visibility[fidx] = 1.0f;

                const auto& c = mesh_->face_centers[fidx];
                double vx = c[0] - observer_pos[0];
                double vy = c[1] - observer_pos[1];
                double vz = c[2] - observer_pos[2];
                const double vlen = std::sqrt(vx*vx + vy*vy + vz*vz);
                result.distance[fidx] = static_cast<float>(vlen / 1000.0);

                const double v_up = vx*up[0] + vy*up[1] + vz*up[2];
                const double v_h = std::sqrt(std::max(0.0, vlen*vlen - v_up*v_up));
                result.elevation_angle[fidx] = static_cast<float>(std::atan2(v_up, v_h) * RAD_TO_DEG);

                if (result.elevation_angle[fidx] < params.min_elevation_angle ||
                    result.elevation_angle[fidx] > params.max_elevation_angle) {
                    result.visibility[fidx] = 0.0f;
                    result.distance[fidx] = -1.0f;
                    continue;
                }

                const double v_e = vx*east[0] + vy*east[1] + vz*east[2];
                const double v_n = vx*north[0] + vy*north[1] + vz*north[2];
                double az = std::atan2(v_e, v_n) * RAD_TO_DEG;
                if (az < 0) az += 360.0;
                result.azimuth[fidx] = static_cast<float>(az);

                // Emission angle: angle between -LOS and outward face normal
                const auto& face_n = mesh_->face_normals[fidx];
                double dx = vx/vlen, dy = vy/vlen, dz = vz/vlen;
                double cos_em = -(dx*face_n[0] + dy*face_n[1] + dz*face_n[2]);
                cos_em = std::max(0.0, std::min(1.0, cos_em));
                result.emission_angle[fidx] = static_cast<float>(std::acos(cos_em) * RAD_TO_DEG);
                result.incidence_cos[fidx] = static_cast<float>(cos_em);
            }
        }
    }

    result.rays_cast = static_cast<int>(mesh_->faces.size());
    result.rays_hit = static_cast<int>(std::count(result.visibility.begin(), result.visibility.end(), 1.0f));
}

// Batch raycast with target distances using Embree
std::vector<std::pair<int, double>> ViewshedAnalyzer::batchRaycastWithDistance(
    const std::vector<std::array<double, 3>>& origins,
    const std::vector<std::array<double, 3>>& directions,
    const std::vector<double>& target_distances,
    bool use_simd) const {
    
    std::vector<std::pair<int, double>> hits(origins.size(), {-1, 0.0});
    
    // Process rays one by one (simplified for clarity - SIMD version would be similar)
    for (size_t i = 0; i < origins.size(); ++i) {
        RTCRayHit rayhit;
        rayhit.ray.org_x = origins[i][0];
        rayhit.ray.org_y = origins[i][1];
        rayhit.ray.org_z = origins[i][2];
        
        rayhit.ray.dir_x = directions[i][0];
        rayhit.ray.dir_y = directions[i][1];
        rayhit.ray.dir_z = directions[i][2];
        
        rayhit.ray.tnear = 1e-4f;
        // CRITICAL: Set tfar to slightly beyond target distance to catch the target face
        rayhit.ray.tfar = target_distances[i] + 1.0f; // Add 1 meter buffer
        rayhit.ray.mask = 0xFFFFFFFF;
        rayhit.ray.flags = 0;
        
        rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
        
        RTCIntersectArguments args;
        rtcInitIntersectArguments(&args);
        rtcIntersect1(embree_scene_, &rayhit, &args);
        
        if (rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID) {
            hits[i].first = rayhit.hit.primID;
            hits[i].second = rayhit.ray.tfar; // Distance to hit
        }
    }
    
    return hits;
}

// Batch raycast using Embree with per-ray tfar limits (meters)
std::vector<int> ViewshedAnalyzer::batchRaycast(
    const std::vector<std::array<double, 3>>& origins,
    const std::vector<std::array<double, 3>>& directions,
    const std::vector<double>& target_distances,
    bool use_simd) const
{
    std::vector<int> hit_faces(origins.size(), -1);

    if (use_simd && has_avx512_) {
#ifdef __AVX512F__
        for (size_t i = 0; i < origins.size(); i += 16) {
            const size_t batch_size = std::min<size_t>(16, origins.size() - i);

            alignas(64) RTCRayHit16 rayhit;
            alignas(64) int valid[16];
            std::memset(valid, 0, sizeof(valid));

            for (size_t j = 0; j < batch_size; ++j) {
                valid[j] = -1;

                rayhit.ray.org_x[j] = static_cast<float>(origins[i + j][0]);
                rayhit.ray.org_y[j] = static_cast<float>(origins[i + j][1]);
                rayhit.ray.org_z[j] = static_cast<float>(origins[i + j][2]);

                rayhit.ray.dir_x[j] = static_cast<float>(directions[i + j][0]);
                rayhit.ray.dir_y[j] = static_cast<float>(directions[i + j][1]);
                rayhit.ray.dir_z[j] = static_cast<float>(directions[i + j][2]);

                rayhit.ray.tnear[j] = 1e-4f;

                float tfar = 1e20f;
                if (i + j < target_distances.size()) {
                    // Slightly beyond target so the target face can be reported
                    tfar = static_cast<float>(target_distances[i + j] * 1.01);
                }
                rayhit.ray.tfar[j] = tfar;

                rayhit.ray.mask[j] = 0xFFFFFFFF;
                rayhit.ray.flags[j] = 0;
                rayhit.hit.geomID[j] = RTC_INVALID_GEOMETRY_ID;
            }

            RTCIntersectArguments args;
            rtcInitIntersectArguments(&args);
            rtcIntersect16(valid, embree_scene_, &rayhit, &args);

            for (size_t j = 0; j < batch_size; ++j) {
                if (rayhit.hit.geomID[j] != RTC_INVALID_GEOMETRY_ID) {
                    hit_faces[i + j] = rayhit.hit.primID[j];
                }
            }
        }
#else
        // Compile-time AVX512 not available; fall back to scalar
        use_simd = false;
#endif
    }

    if (!use_simd || !has_avx512_) {
        #pragma omp parallel for
        for (size_t i = 0; i < origins.size(); ++i) {
            RTCRayHit rayhit;
            rayhit.ray.org_x = static_cast<float>(origins[i][0]);
            rayhit.ray.org_y = static_cast<float>(origins[i][1]);
            rayhit.ray.org_z = static_cast<float>(origins[i][2]);

            rayhit.ray.dir_x = static_cast<float>(directions[i][0]);
            rayhit.ray.dir_y = static_cast<float>(directions[i][1]);
            rayhit.ray.dir_z = static_cast<float>(directions[i][2]);

            rayhit.ray.tnear = 1e-4f;
            float tfar = 1e20f;
            if (i < target_distances.size()) {
                tfar = static_cast<float>(target_distances[i] + 1.0f);
            }
            rayhit.ray.tfar = tfar;
            rayhit.ray.mask = 0xFFFFFFFF;
            rayhit.ray.flags = 0;

            rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;

            RTCIntersectArguments args;
            rtcInitIntersectArguments(&args);
            rtcIntersect1(embree_scene_, &rayhit, &args);

            if (rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID) {
                hit_faces[i] = rayhit.hit.primID;
            }
        }
    }

    return hit_faces;
}

// Backward-compatible overload without per-ray distances (uses very large tfar)
std::vector<int> ViewshedAnalyzer::batchRaycast(
    const std::vector<std::array<double, 3>>& origins,
    const std::vector<std::array<double, 3>>& directions,
    bool use_simd) const
{
    // Use large tfar for all rays
    std::vector<double> tfar(origins.size(), 1e20);
    return batchRaycast(origins, directions, tfar, use_simd);
}

// Compute horizon profile
void ViewshedAnalyzer::computeHorizonProfile(
    const std::array<double, 3>& observer_pos,
    ViewshedResult& result,
    const ViewshedParams& params) {
    
    const int num_azimuth_samples = static_cast<int>(360.0 / params.angular_resolution);
    result.horizon_profile.reserve(num_azimuth_samples);

    // Build local frame at observer
    auto norm3 = [](const std::array<double,3>& v){
        double L = std::sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
        return (L > 0.0) ? std::array<double,3>{v[0]/L, v[1]/L, v[2]/L} : std::array<double,3>{0.0,0.0,1.0};
    };
    auto cross3 = [](const std::array<double,3>& a, const std::array<double,3>& b){
        return std::array<double,3>{
            a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0]
        };
    };

    // Outward normal at observer
    std::array<double,3> up = (coord_frame_ == CoordFrame::ECEF)
    ? norm3(observer_pos)
    : std::array<double,3>{0.0, 0.0, 1.0};
    // Robust east/north
    std::array<double,3> ez = {0.0, 0.0, 1.0};
    std::array<double,3> ex = {1.0, 0.0, 0.0};
    std::array<double,3> east = cross3(ez, up);
    if (std::abs(east[0]) + std::abs(east[1]) + std::abs(east[2]) < 1e-10) {
        east = cross3(ex, up);
    }
    east = norm3(east);
    std::array<double,3> north = cross3(up, east);
    north = norm3(north);

    // Safe origin above the surface
    const double safe_offset = std::max(1e-6 * body_radius_, 1e-4 * body_radius_);
    const std::array<double,3> origin_out = {
        observer_pos[0] + up[0]*safe_offset,
        observer_pos[1] + up[1]*safe_offset,
        observer_pos[2] + up[2]*safe_offset
    };
    
    for (int i = 0; i < num_azimuth_samples; ++i) {
        double azimuth = i * params.angular_resolution;
        double az_rad = azimuth * DEG_TO_RAD;

        // Unit vector on tangent plane pointing at this azimuth
        std::array<double,3> dir_az = {
            std::cos(az_rad) * north[0] + std::sin(az_rad) * east[0],
            std::cos(az_rad) * north[1] + std::sin(az_rad) * east[1],
            std::cos(az_rad) * north[2] + std::sin(az_rad) * east[2]
        };

        double max_elevation = -90.0;
        
        // Sweep elevation from below to above horizon
        for (double elevation = -10.0; elevation <= 45.0; elevation += 0.5) {
            double el_rad = elevation * DEG_TO_RAD;

            // Ray direction in local frame: tilt up by elevation
            std::array<double,3> direction = {
                std::cos(el_rad) * dir_az[0] + std::sin(el_rad) * up[0],
                std::cos(el_rad) * dir_az[1] + std::sin(el_rad) * up[1],
                std::cos(el_rad) * dir_az[2] + std::sin(el_rad) * up[2]
            };

            if (isRayBlocked(origin_out, direction, params.max_range_km * 1000.0)) {
                max_elevation = elevation;
            }
        }
        
        result.horizon_profile.push_back({azimuth, max_elevation});
    }
}

// Check if a ray is blocked
bool ViewshedAnalyzer::isRayBlocked(
    const std::array<double, 3>& origin,
    const std::array<double, 3>& direction,
    double max_distance) const {
    
    RTCRayHit rayhit;
    rayhit.ray.org_x = origin[0];
    rayhit.ray.org_y = origin[1];
    rayhit.ray.org_z = origin[2];
    
    rayhit.ray.dir_x = direction[0];
    rayhit.ray.dir_y = direction[1];
    rayhit.ray.dir_z = direction[2];
    
    rayhit.ray.tnear = 1e-4f;
    rayhit.ray.tfar = max_distance;
    rayhit.ray.mask = 0xFFFFFFFF;
    rayhit.ray.flags = 0;
    
    rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
    
    RTCIntersectArguments args;
    rtcInitIntersectArguments(&args);
    rtcIntersect1(embree_scene_, &rayhit, &args);
    
    return (rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID);
}

// Compute statistics
void ViewshedAnalyzer::computeStatistics(ViewshedResult& result) const {
    result.num_visible_faces = 0;
    result.visible_area_km2 = 0.0;
    result.max_visible_range_km = 0.0;
    double total_range = 0.0;
    int visible_count = 0;
    
    // Initialize bounds
    result.visibility_bounds[0] = 90.0;   // min_lat
    result.visibility_bounds[1] = -90.0;  // max_lat
    result.visibility_bounds[2] = 180.0;  // min_lon
    result.visibility_bounds[3] = -180.0; // max_lon
    
    for (size_t i = 0; i < result.visibility.size(); ++i) {
        if (result.visibility[i] > 0) {
            result.num_visible_faces++;
            double area_m2 = mesh_utils_->getFaceArea(i);
            result.face_area_m2[i] = static_cast<float>(area_m2);
            result.visible_area_km2 += area_m2 / 1e6;
            
            if (result.distance[i] > result.max_visible_range_km) {
                result.max_visible_range_km = result.distance[i];
            }
            
            total_range += result.distance[i];
            visible_count++;
            
            // Update bounds
            double lat = mesh_->latitudes[i];
            double lon = mesh_->longitudes[i];
            
            if (lat < result.visibility_bounds[0]) result.visibility_bounds[0] = lat;
            if (lat > result.visibility_bounds[1]) result.visibility_bounds[1] = lat;
            if (lon < result.visibility_bounds[2]) result.visibility_bounds[2] = lon;
            if (lon > result.visibility_bounds[3]) result.visibility_bounds[3] = lon;
        }
    }
    
    if (visible_count > 0) {
        result.mean_visible_range_km = total_range / visible_count;
    }
}

// Find closest mesh face to a lat/lon location (robust to missing lat/lon arrays)
int ViewshedAnalyzer::findClosestFace(double lat, double lon) const {
    if (!mesh_ || mesh_->faces.empty()) {
        return -1;
    }

    const bool have_ll = (!mesh_->latitudes.empty() && !mesh_->longitudes.empty()
                          && mesh_->latitudes.size() == mesh_->faces.size()
                          && mesh_->longitudes.size() == mesh_->faces.size());

    int closest_face = -1;
    double min_distance = std::numeric_limits<double>::max();
    const double lat_rad = lat * DEG_TO_RAD;
    const double cos_lat = std::max(1e-6, std::cos(lat_rad)); // avoid zero at poles

    for (size_t i = 0; i < mesh_->faces.size(); ++i) {
        double face_lat, face_lon;

        if (have_ll) {
            face_lat = mesh_->latitudes[i];
            face_lon = mesh_->longitudes[i];
        } else {
            // Derive lat/lon from face center using current frame
            std::array<double,3> ecef = mesh_->face_centers[i];
            if (coord_frame_ == CoordFrame::LOCAL_ENU) {
                ecef = enuToEcef(mesh_->face_centers[i]);
            }
            const double rxy = std::hypot(ecef[0], ecef[1]);
            const double r = std::hypot(rxy, ecef[2]);
            if (r <= 0.0) continue;
            face_lat = std::asin(ecef[2] / r) * RAD_TO_DEG;
            face_lon = std::atan2(ecef[1], ecef[0]) * RAD_TO_DEG;
        }

        double dlat = face_lat - lat;
        double dlon = face_lon - lon;

        // Handle longitude wrap-around
        if (dlon > 180.0) dlon -= 360.0;
        if (dlon < -180.0) dlon += 360.0;

        // Weight longitude difference by latitude to account for convergence at poles
        dlon *= cos_lat;

        const double dist_sq = dlat * dlat + dlon * dlon;

        if (dist_sq < min_distance) {
            min_distance = dist_sq;
            closest_face = static_cast<int>(i);
        }
    }

    return closest_face;
}

// Coordinate transformations
std::array<double, 3> ViewshedAnalyzer::latLonToCartesian(
    double lat, double lon, double elevation) const {
    if (coord_frame_ == CoordFrame::ECEF) {
        return ecefFromLatLon(lat, lon, elevation, body_radius_);
    } else {
        // Compute ENU of the target w.r.t. ENU reference
        auto ecef = ecefFromLatLon(lat, lon, elevation, body_radius_);
        return ecefToEnu(ecef);
    }
}

std::pair<double, double> ViewshedAnalyzer::cartesianToLatLon(
    const std::array<double, 3>& position) const {
    std::array<double,3> ecef = position;
    if (coord_frame_ == CoordFrame::LOCAL_ENU) {
        ecef = enuToEcef(position);
    }
    double r = std::sqrt(ecef[0]*ecef[0] + ecef[1]*ecef[1] + ecef[2]*ecef[2]);
    double lat = std::asin(ecef[2] / r) * RAD_TO_DEG;
    double lon = std::atan2(ecef[1], ecef[0]) * RAD_TO_DEG;
    return {lat, lon};
}

// Batch computation
std::vector<ViewshedResult> ViewshedAnalyzer::computeMultipleViewsheds(
    const std::vector<ViewLocation>& locations,
    const ViewshedParams& params) {
    
    std::vector<ViewshedResult> results(locations.size());
    
    int num_threads = params.num_threads > 0 ? params.num_threads : hardware_threads_;
    
#ifdef _OPENMP
    omp_set_num_threads(num_threads);
    #pragma omp parallel for schedule(dynamic)
#endif
    for (size_t i = 0; i < locations.size(); ++i) {
        results[i] = computeViewshed(locations[i], params);
        
        // Progress reporting
        #pragma omp critical
        {
            std::cout << "\rComputing viewsheds: " << (i + 1) << "/" << locations.size()
                     << " (" << std::fixed << std::setprecision(1) 
                     << (100.0 * (i + 1) / locations.size()) << "%)" << std::flush;
        }
    }
    
    std::cout << std::endl;
    return results;
}

// Analyze visibility into specific region
ViewshedResult ViewshedAnalyzer::analyzeRegionVisibility(
    const ViewLocation& observer,
    const std::vector<int>& target_faces,
    const ViewshedParams& params) {
    
    // Compute normal viewshed
    ViewshedResult result = computeViewshed(observer, params);
    
    // Count visibility into target region
    int visible_in_region = 0;
    double visible_region_area = 0.0;
    double total_region_area = 0.0;
    
    for (int face_idx : target_faces) {
        if (face_idx >= 0 && face_idx < result.visibility.size()) {
            double face_area = mesh_utils_->getFaceArea(face_idx);
            total_region_area += face_area / 1e6;  // Convert to km²
            
            if (result.visibility[face_idx] > 0) {
                visible_in_region++;
                visible_region_area += face_area / 1e6;
            }
        }
    }
    
    // Report region-specific statistics
    std::cout << "Region visibility analysis:\n";
    std::cout << "  Target region: " << target_faces.size() << " faces\n";
    std::cout << "  Visible in region: " << visible_in_region 
              << " (" << std::fixed << std::setprecision(1)
              << (100.0 * visible_in_region / target_faces.size()) << "%)\n";
    std::cout << "  Visible region area: " << std::setprecision(2) 
              << visible_region_area << " km²\n";
    std::cout << "  Total region area: " << total_region_area << " km²\n";
    std::cout << "  Coverage: " << std::setprecision(1)
              << (100.0 * visible_region_area / total_region_area) << "%\n";
    
    return result;
}

// Get faces in circular region
std::vector<int> ViewshedAnalyzer::getFacesInRegion(
    double center_lat, double center_lon, double radius_km) const {
    
    std::vector<int> faces;
    
    if (!mesh_) return faces;
    
    // Convert center to Cartesian
    auto center = latLonToCartesian(center_lat, center_lon, 0.0);
    double radius_m = radius_km * 1000.0;
    
    for (size_t i = 0; i < mesh_->face_centers.size(); ++i) {
        const auto& face_center = mesh_->face_centers[i];
        
        double dx = face_center[0] - center[0];
        double dy = face_center[1] - center[1];
        double dz = face_center[2] - center[2];
        double distance = std::sqrt(dx*dx + dy*dy + dz*dz);
        
        if (distance <= radius_m) {
            faces.push_back(i);
        }
    }
    
    return faces;
}

// Get faces from boolean mask
std::vector<int> ViewshedAnalyzer::getFacesFromMask(const std::vector<bool>& mask) const {
    std::vector<int> faces;
    
    for (size_t i = 0; i < mask.size() && i < mesh_->faces.size(); ++i) {
        if (mask[i]) {
            faces.push_back(i);
        }
    }
    
    return faces;
}

// Load viewpoints from CSV file
std::vector<ViewLocation> ViewshedAnalyzer::loadViewpointsFromFile(
    const std::string& filename) const {
    
    std::vector<ViewLocation> locations;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return locations;
    }
    
    std::string line;
    int line_num = 0;
    
    while (std::getline(file, line)) {
        line_num++;
        
        // Skip comments and empty lines
        if (line.empty() || line[0] == '#') continue;
        
        // Check if header
        if (line.find("label") != std::string::npos || 
            line.find("lat") != std::string::npos) {
            continue;  // Skip header
        }
        
        std::istringstream iss(line);
        std::string label;
        double lat, lon, height = 2.0;
        
        // Parse CSV line
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
                    
                    ViewLocation loc(lat, lon, height, label);
                    locations.push_back(loc);
                    
                } catch (const std::exception& e) {
                    std::cerr << "Error parsing line " << line_num << ": " << e.what() << std::endl;
                }
            }
        }
    }
    
    file.close();
    return locations;
}

// Export to VTK with viewpoint marking
bool ViewshedAnalyzer::exportToVTK(
    const std::string& filename,
    const ViewshedResult& result,
    bool mark_viewpoint) const
{
    if (!mesh_ || mesh_->vertices.empty() || mesh_->faces.empty()) {
        std::cerr << "exportToVTK: mesh empty\n";
        return false;
    }
    std::filesystem::path out = filename;
    if (!out.has_extension() || out.extension() == ".vtk" || out.extension() == ".VTK") {
        out.replace_extension(".vtu"); // ensure VTU for ParaView PVD compatibility
    }
    if (!out.parent_path().empty()) {
        std::error_code ec;
        std::filesystem::create_directories(out.parent_path(), ec);
    }
    return writeVTU_UnstructuredGrid(out.string(), mesh_, result, /*time_value*/ 0.0, mark_viewpoint);
}

// Export a ParaView PVD collection referencing .vtu time steps
bool ViewshedAnalyzer::exportToPVD(
    const std::string& pvd_filename,
    const std::vector<ViewshedResult>& results,
    bool mark_viewpoints) const
{
    if (!mesh_ || mesh_->faces.empty() || results.empty()) {
        std::cerr << "exportToPVD: invalid inputs\n";
        return false;
    }

    namespace fs = std::filesystem;
    fs::path pvd_path = pvd_filename;
    if (!pvd_path.has_extension() || pvd_path.extension() != ".pvd") {
        pvd_path.replace_extension(".pvd");
    }
    if (!pvd_path.parent_path().empty()) {
        std::error_code ec;
        fs::create_directories(pvd_path.parent_path(), ec);
    }

    const std::string stem = pvd_path.stem().string(); // base name for steps
    std::vector<std::string> step_files;
    step_files.reserve(results.size());

    for (size_t i = 0; i < results.size(); ++i) {
        std::ostringstream name;
        name << stem << "_" << std::setfill('0') << std::setw(4) << i << ".vtu";
        fs::path vtu_path = pvd_path.parent_path() / name.str();

        double time_value = static_cast<double>(i); // simple increasing timestep
        if (!writeVTU_UnstructuredGrid(vtu_path.string(), mesh_, results[i], time_value, mark_viewpoints)) {
            std::cerr << "exportToPVD: failed to write " << vtu_path << "\n";
            return false;
        }
        step_files.push_back(name.str()); // relative path
    }

    std::ofstream pvd(pvd_path.string(), std::ios::binary);
    if (!pvd) {
        std::cerr << "exportToPVD: cannot open " << pvd_path << "\n";
        return false;
    }

    pvd << "<?xml version=\"1.0\"?>\n";
    pvd << "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    pvd << "  <Collection>\n";
    for (size_t i = 0; i < step_files.size(); ++i) {
        pvd << "    <DataSet timestep=\"" << i << "\" group=\"\" part=\"0\" file=\"" << step_files[i] << "\"/>\n";
    }
    pvd << "  </Collection>\n";
    pvd << "</VTKFile>\n";

    return true;
}

#ifdef HAS_HDF5
// Export to HDF5
bool ViewshedAnalyzer::exportToHDF5(
    const std::string& filename,
    const std::vector<ViewshedResult>& results) const {
    
    try {
        // Create HDF5 file
        H5::H5File file(filename, H5F_ACC_TRUNC);
        
        // Write mesh data (once)
        {
            H5::Group mesh_group = file.createGroup("/mesh");
            
            // Write vertices
            hsize_t vert_dims[2] = {mesh_->vertices.size(), 3};
            H5::DataSpace vert_space(2, vert_dims);
            H5::DataSet vert_dataset = mesh_group.createDataSet("vertices", 
                H5::PredType::NATIVE_DOUBLE, vert_space);
            
            std::vector<double> vert_data;
            for (const auto& v : mesh_->vertices) {
                vert_data.push_back(v[0]);
                vert_data.push_back(v[1]);
                vert_data.push_back(v[2]);
            }
            vert_dataset.write(vert_data.data(), H5::PredType::NATIVE_DOUBLE);
            
            // Write faces
            hsize_t face_dims[2] = {mesh_->faces.size(), 3};
            H5::DataSpace face_space(2, face_dims);
            H5::DataSet face_dataset = mesh_group.createDataSet("faces", 
                H5::PredType::NATIVE_INT, face_space);
            
            std::vector<int> face_data;
            for (const auto& f : mesh_->faces) {
                face_data.push_back(f[0]);
                face_data.push_back(f[1]);
                face_data.push_back(f[2]);
            }
            face_dataset.write(face_data.data(), H5::PredType::NATIVE_INT);

            // Write per-face geographic data
            const size_t n_faces = mesh_->faces.size();
            hsize_t face1d_dims[1] = {n_faces};
            H5::DataSpace face1d_space(1, face1d_dims);

            if (mesh_->latitudes.size() == n_faces) {
                H5::DataSet lat_mesh = mesh_group.createDataSet("face_latitudes",
                    H5::PredType::NATIVE_DOUBLE, face1d_space);
                lat_mesh.write(mesh_->latitudes.data(), H5::PredType::NATIVE_DOUBLE);
            }
            if (mesh_->longitudes.size() == n_faces) {
                H5::DataSet lon_mesh = mesh_group.createDataSet("face_longitudes",
                    H5::PredType::NATIVE_DOUBLE, face1d_space);
                lon_mesh.write(mesh_->longitudes.data(), H5::PredType::NATIVE_DOUBLE);
            }
            if (mesh_->elevations.size() == n_faces) {
                H5::DataSet elev_mesh = mesh_group.createDataSet("face_elevations",
                    H5::PredType::NATIVE_DOUBLE, face1d_space);
                elev_mesh.write(mesh_->elevations.data(), H5::PredType::NATIVE_DOUBLE);
            }
            if (mesh_->face_normals.size() == n_faces) {
                hsize_t norm_dims[2] = {n_faces, 3};
                H5::DataSpace norm_space(2, norm_dims);
                H5::DataSet norm_mesh = mesh_group.createDataSet("face_normals",
                    H5::PredType::NATIVE_DOUBLE, norm_space);
                std::vector<double> norm_data;
                norm_data.reserve(n_faces * 3);
                for (const auto& n : mesh_->face_normals) {
                    norm_data.push_back(n[0]);
                    norm_data.push_back(n[1]);
                    norm_data.push_back(n[2]);
                }
                norm_mesh.write(norm_data.data(), H5::PredType::NATIVE_DOUBLE);
            }
        }

        // Write viewshed results
        H5::Group viewsheds_group = file.createGroup("/viewsheds");
        
        for (size_t i = 0; i < results.size(); ++i) {
            const auto& result = results[i];
            
            std::string group_name = "viewshed_" + std::to_string(i);
            H5::Group vgroup = viewsheds_group.createGroup(group_name);
            
            // Write visibility
            hsize_t vis_dims[1] = {result.visibility.size()};
            H5::DataSpace vis_space(1, vis_dims);
            H5::DataSet vis_dataset = vgroup.createDataSet("visibility", 
                H5::PredType::NATIVE_FLOAT, vis_space);
            vis_dataset.write(result.visibility.data(), H5::PredType::NATIVE_FLOAT);
            
            // Write distance
            H5::DataSet dist_dataset = vgroup.createDataSet("distance", 
                H5::PredType::NATIVE_FLOAT, vis_space);
            dist_dataset.write(result.distance.data(), H5::PredType::NATIVE_FLOAT);
            
            // Write elevation angles
            H5::DataSet elev_dataset = vgroup.createDataSet("elevation_angle", 
                H5::PredType::NATIVE_FLOAT, vis_space);
            elev_dataset.write(result.elevation_angle.data(), H5::PredType::NATIVE_FLOAT);
            
            // Write azimuth
            H5::DataSet az_dataset = vgroup.createDataSet("azimuth",
                H5::PredType::NATIVE_FLOAT, vis_space);
            az_dataset.write(result.azimuth.data(), H5::PredType::NATIVE_FLOAT);

            // Write emission angle
            if (!result.emission_angle.empty()) {
                H5::DataSet em_dataset = vgroup.createDataSet("emission_angle",
                    H5::PredType::NATIVE_FLOAT, vis_space);
                em_dataset.write(result.emission_angle.data(), H5::PredType::NATIVE_FLOAT);
            }

            // Write incidence cosine
            if (!result.incidence_cos.empty()) {
                H5::DataSet ic_dataset = vgroup.createDataSet("incidence_cos",
                    H5::PredType::NATIVE_FLOAT, vis_space);
                ic_dataset.write(result.incidence_cos.data(), H5::PredType::NATIVE_FLOAT);
            }

            // Write face area
            if (!result.face_area_m2.empty()) {
                H5::DataSet fa_dataset = vgroup.createDataSet("face_area_m2",
                    H5::PredType::NATIVE_FLOAT, vis_space);
                fa_dataset.write(result.face_area_m2.data(), H5::PredType::NATIVE_FLOAT);
            }

            // Write viewpoint face index
            hsize_t scalar_dims[1] = {1};
            H5::DataSpace scalar_space(1, scalar_dims);
            H5::DataSet vp_dataset = vgroup.createDataSet("viewpoint_face_index", 
                H5::PredType::NATIVE_INT, scalar_space);
            vp_dataset.write(&result.viewpoint_face_index, H5::PredType::NATIVE_INT);
            
            // Write location metadata
            H5::Group loc_group = vgroup.createGroup("location");
            
            H5::DataSet lat_dataset = loc_group.createDataSet("latitude", 
                H5::PredType::NATIVE_DOUBLE, scalar_space);
            lat_dataset.write(&result.location.latitude, H5::PredType::NATIVE_DOUBLE);
            
            H5::DataSet lon_dataset = loc_group.createDataSet("longitude", 
                H5::PredType::NATIVE_DOUBLE, scalar_space);
            lon_dataset.write(&result.location.longitude, H5::PredType::NATIVE_DOUBLE);
            
            H5::DataSet height_dataset = loc_group.createDataSet("height", 
                H5::PredType::NATIVE_DOUBLE, scalar_space);
            height_dataset.write(&result.location.height, H5::PredType::NATIVE_DOUBLE);
            
            // Write label as attribute
            H5::StrType str_type(H5::PredType::C_S1, result.location.label.size() + 1);
            H5::Attribute label_attr = loc_group.createAttribute("label", str_type, scalar_space);
            label_attr.write(str_type, result.location.label);

            // Write UTC timestamp as attribute (if present)
            if (!result.location.utc_timestamp.empty()) {
                H5::StrType ts_type(H5::PredType::C_S1, result.location.utc_timestamp.size() + 1);
                H5::Attribute ts_attr = loc_group.createAttribute("utc_timestamp", ts_type, scalar_space);
                ts_attr.write(ts_type, result.location.utc_timestamp);
            }

            // Write statistics
            H5::Group stats_group = vgroup.createGroup("statistics");
            
            H5::DataSet nvis_dataset = stats_group.createDataSet("num_visible_faces", 
                H5::PredType::NATIVE_INT, scalar_space);
            nvis_dataset.write(&result.num_visible_faces, H5::PredType::NATIVE_INT);
            
            H5::DataSet area_dataset = stats_group.createDataSet("visible_area_km2", 
                H5::PredType::NATIVE_DOUBLE, scalar_space);
            area_dataset.write(&result.visible_area_km2, H5::PredType::NATIVE_DOUBLE);
            
            H5::DataSet maxr_dataset = stats_group.createDataSet("max_visible_range_km", 
                H5::PredType::NATIVE_DOUBLE, scalar_space);
            maxr_dataset.write(&result.max_visible_range_km, H5::PredType::NATIVE_DOUBLE);
            
            H5::DataSet meanr_dataset = stats_group.createDataSet("mean_visible_range_km", 
                H5::PredType::NATIVE_DOUBLE, scalar_space);
            meanr_dataset.write(&result.mean_visible_range_km, H5::PredType::NATIVE_DOUBLE);
            
            // Write visibility bounds
            hsize_t bounds_dims[1] = {4};
            H5::DataSpace bounds_space(1, bounds_dims);
            H5::DataSet bounds_dataset = stats_group.createDataSet("visibility_bounds", 
                H5::PredType::NATIVE_DOUBLE, bounds_space);
            bounds_dataset.write(result.visibility_bounds.data(), H5::PredType::NATIVE_DOUBLE);
            
            // Write horizon profile if present
            if (!result.horizon_profile.empty()) {
                H5::Group horizon_group = vgroup.createGroup("horizon_profile");
                
                std::vector<double> azimuths, elevations;
                for (const auto& [az, el] : result.horizon_profile) {
                    azimuths.push_back(az);
                    elevations.push_back(el);
                }
                
                hsize_t horizon_dims[1] = {azimuths.size()};
                H5::DataSpace horizon_space(1, horizon_dims);
                
                H5::DataSet az_prof_dataset = horizon_group.createDataSet("azimuth", 
                    H5::PredType::NATIVE_DOUBLE, horizon_space);
                az_prof_dataset.write(azimuths.data(), H5::PredType::NATIVE_DOUBLE);
                
                H5::DataSet el_prof_dataset = horizon_group.createDataSet("elevation", 
                    H5::PredType::NATIVE_DOUBLE, horizon_space);
                el_prof_dataset.write(elevations.data(), H5::PredType::NATIVE_DOUBLE);
            }
        }
        
        file.close();
        
        std::cout << "Exported " << results.size() << " viewsheds to HDF5: " 
                  << filename << std::endl;
        
        return true;
        
    } catch (const H5::Exception& e) {
        std::cerr << "HDF5 error: " << e.getCDetailMsg() << std::endl;
        return false;
    }
}
#endif

// Internal helpers for VTU/PVD writing
namespace {

// Ensure we always write .vtu extension
static std::string ensure_vtu_extension(const std::string& path) {
    std::filesystem::path p(path);
    if (!p.has_extension() || p.extension() != ".vtu") {
        p.replace_extension(".vtu");
    }
    return p.string();
}

// Minimal, robust VTU (ASCII) writer for UnstructuredGrid with triangle cells
static bool writeVTU_UnstructuredGrid(
    const std::string& vtu_path,
    const std::shared_ptr<Mesh>& mesh,
    const ViewshedResult& result,
    double time_value = 0.0,
    bool mark_viewpoint = true)
{
    if (!mesh || mesh->vertices.empty() || mesh->faces.empty()) {
        std::cerr << "writeVTU_UnstructuredGrid: empty mesh\n";
        return false;
    }

    const size_t nPts = mesh->vertices.size();
    const size_t nCells = mesh->faces.size();

    // Build connectivity, offsets, types
    std::ostringstream conn, offs, types;
    int offset = 0;
    for (const auto& f : mesh->faces) {
        conn << f[0] << " " << f[1] << " " << f[2] << "\n";
        offset += 3;
        offs << offset << "\n";
        types << 5 << "\n"; // VTK_TRIANGLE = 5
    }

    // Points (Float64)
    std::ostringstream pts;
    pts.setf(std::ios::scientific);
    pts.precision(15);
    for (const auto& p : mesh->vertices) {
        pts << p[0] << " " << p[1] << " " << p[2] << "\n";
    }

    // Writer for Float32 cell arrays (pad/validate size)
    auto writeFloatArray = [&](const char* name, const std::vector<float>& data) -> std::string {
        std::ostringstream ss;
        ss.setf(std::ios::scientific);
        ss.precision(7);
        ss << "<DataArray type=\"Float32\" Name=\"" << name << "\" format=\"ascii\">\n";
        if (data.size() == nCells) {
            for (size_t i = 0; i < nCells; ++i) ss << data[i] << "\n";
        } else {
            for (size_t i = 0; i < nCells; ++i) ss << 0.0f << "\n";
        }
        ss << "</DataArray>\n";
        return ss.str();
    };

    // Optional viewpoint mask
    std::vector<float> vp_mask;
    if (mark_viewpoint && result.viewpoint_face_index >= 0) {
        vp_mask.assign(nCells, 0.0f);
        const size_t idx = static_cast<size_t>(result.viewpoint_face_index);
        if (idx < nCells) vp_mask[idx] = 1.0f;
    }

    // FieldData (TimeValue and LocationLabel if present)
    std::ostringstream fdata;
    fdata << "<FieldData>\n";
    fdata << "  <DataArray type=\"Float64\" Name=\"TimeValue\" NumberOfTuples=\"1\" format=\"ascii\">\n";
    fdata << "    " << std::setprecision(16) << time_value << "\n";
    fdata << "  </DataArray>\n";
    if (!result.location.label.empty()) {
        fdata << "  <DataArray type=\"String\" Name=\"LocationLabel\" NumberOfTuples=\"1\" format=\"ascii\">\n";
        fdata << "    " << result.location.label << "\n";
        fdata << "  </DataArray>\n";
    }
    fdata << "</FieldData>\n";

    // Assemble VTU XML
    std::ostringstream xml;
    xml << "<?xml version=\"1.0\"?>\n";
    xml << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    xml << "  <UnstructuredGrid>\n";
    xml << "    <Piece NumberOfPoints=\"" << nPts << "\" NumberOfCells=\"" << nCells << "\">\n";
    xml << "      <Points>\n";
    xml << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    xml << pts.str();
    xml << "        </DataArray>\n";
    xml << "      </Points>\n";
    xml << "      <Cells>\n";
    xml << "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
    xml << conn.str();
    xml << "        </DataArray>\n";
    xml << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
    xml << offs.str();
    xml << "        </DataArray>\n";
    xml << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
    xml << types.str();
    xml << "        </DataArray>\n";
    xml << "      </Cells>\n";
    xml << "      <CellData Scalars=\"visibility\">\n";
    xml << writeFloatArray("visibility", result.visibility);
    xml << writeFloatArray("distance_km", result.distance);
    xml << writeFloatArray("elevation_deg", result.elevation_angle);
    xml << writeFloatArray("azimuth_deg", result.azimuth);
    xml << writeFloatArray("emission_angle_deg", result.emission_angle);
    xml << writeFloatArray("incidence_cos", result.incidence_cos);
    xml << writeFloatArray("face_area_m2", result.face_area_m2);
    if (!vp_mask.empty()) {
        xml << writeFloatArray("viewpoint_face", vp_mask);
    }
    xml << "      </CellData>\n";
    xml << fdata.str();
    xml << "    </Piece>\n";
    xml << "  </UnstructuredGrid>\n";
    xml << "</VTKFile>\n";

    // Ensure directory exists and write file
    std::filesystem::path outp = ensure_vtu_extension(vtu_path);
    if (!outp.parent_path().empty()) {
        std::error_code ec;
        std::filesystem::create_directories(outp.parent_path(), ec);
    }
    std::ofstream out(outp.string(), std::ios::binary);
    if (!out) {
        std::cerr << "writeVTU_UnstructuredGrid: cannot open " << outp << "\n";
        return false;
    }
    const auto s = xml.str();
    out.write(s.data(), static_cast<std::streamsize>(s.size()));
    return true;
}

} // anonymous namespace

} // namespace ptm