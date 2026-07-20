/**
 * H-Matrix Profiling Tool
 *
 * Identifies computational bottlenecks in the H-matrix view factor computation.
 * Instruments each phase to measure time spent in:
 * - Face data precomputation
 * - Quadtree construction
 * - View factor computation (ray tracing + geometry)
 * - SVD compression
 * - Block construction overhead
 */

#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <cmath>
#include <atomic>

#include <embree4/rtcore.h>

#include "hmatrix_view_factor.h"
#include "radiation_config.h"
#include "../mesh/mesh_utils.h"

namespace radiation {
namespace profile {

// Global counters for profiling (atomic for thread safety)
std::atomic<size_t> g_ray_trace_count{0};
std::atomic<size_t> g_view_factor_calls{0};
std::atomic<size_t> g_culled_backface{0};
std::atomic<size_t> g_culled_grazing{0};
std::atomic<size_t> g_culled_distance{0};
std::atomic<size_t> g_svd_calls{0};
std::atomic<size_t> g_svd_total_elements{0};

// Timing accumulators (in microseconds)
std::atomic<long long> g_time_ray_trace_us{0};
std::atomic<long long> g_time_geometry_us{0};
std::atomic<long long> g_time_svd_us{0};

void resetCounters() {
    g_ray_trace_count = 0;
    g_view_factor_calls = 0;
    g_culled_backface = 0;
    g_culled_grazing = 0;
    g_culled_distance = 0;
    g_svd_calls = 0;
    g_svd_total_elements = 0;
    g_time_ray_trace_us = 0;
    g_time_geometry_us = 0;
    g_time_svd_us = 0;
}

void printCounters() {
    std::cout << "\n=== PROFILING RESULTS ===" << std::endl;
    std::cout << "View factor calls:     " << g_view_factor_calls.load() << std::endl;
    std::cout << "  Culled (backface):   " << g_culled_backface.load()
              << " (" << (100.0 * g_culled_backface.load() / std::max(size_t(1), g_view_factor_calls.load())) << "%)" << std::endl;
    std::cout << "  Culled (grazing):    " << g_culled_grazing.load()
              << " (" << (100.0 * g_culled_grazing.load() / std::max(size_t(1), g_view_factor_calls.load())) << "%)" << std::endl;
    std::cout << "  Culled (distance):   " << g_culled_distance.load()
              << " (" << (100.0 * g_culled_distance.load() / std::max(size_t(1), g_view_factor_calls.load())) << "%)" << std::endl;
    std::cout << "  Ray traces executed: " << g_ray_trace_count.load()
              << " (" << (100.0 * g_ray_trace_count.load() / std::max(size_t(1), g_view_factor_calls.load())) << "%)" << std::endl;
    std::cout << "\nSVD compressions:      " << g_svd_calls.load() << std::endl;
    std::cout << "  Total elements:      " << g_svd_total_elements.load() << std::endl;
    std::cout << "\nTime breakdown:" << std::endl;
    std::cout << "  Geometry checks:     " << (g_time_geometry_us.load() / 1000.0) << " ms" << std::endl;
    std::cout << "  Ray tracing:         " << (g_time_ray_trace_us.load() / 1000.0) << " ms" << std::endl;
    std::cout << "  SVD compression:     " << (g_time_svd_us.load() / 1000.0) << " ms" << std::endl;
}

// Generate a bowl crater mesh for testing
std::shared_ptr<ptm::Mesh> generateBowlCraterMesh(
    double diameter, double depth_to_diameter, int n_rings, int n_azimuth) {

    auto mesh = std::make_shared<ptm::Mesh>();

    double radius = diameter / 2.0;
    double depth = diameter * depth_to_diameter;

    // Parabolic bowl: z = depth * (r/radius)^2 - depth
    // So z ranges from -depth (center) to 0 (rim)

    // Generate vertices
    // Center vertex
    mesh->vertices.push_back({{0.0, 0.0, -depth}});

    // Ring vertices
    for (int ring = 1; ring <= n_rings; ++ring) {
        double r = radius * ring / n_rings;
        double z = depth * (r / radius) * (r / radius) - depth;

        for (int az = 0; az < n_azimuth; ++az) {
            double theta = 2.0 * M_PI * az / n_azimuth;
            mesh->vertices.push_back({{
                r * std::cos(theta),
                r * std::sin(theta),
                z
            }});
        }
    }

    // Generate faces
    // Center fan
    for (int az = 0; az < n_azimuth; ++az) {
        int next_az = (az + 1) % n_azimuth;
        mesh->faces.push_back({{0, 1 + az, 1 + next_az}});
    }

    // Ring strips
    for (int ring = 1; ring < n_rings; ++ring) {
        int base_inner = 1 + (ring - 1) * n_azimuth;
        int base_outer = 1 + ring * n_azimuth;

        for (int az = 0; az < n_azimuth; ++az) {
            int next_az = (az + 1) % n_azimuth;

            int v0 = base_inner + az;
            int v1 = base_inner + next_az;
            int v2 = base_outer + az;
            int v3 = base_outer + next_az;

            mesh->faces.push_back({{v0, v2, v1}});
            mesh->faces.push_back({{v1, v2, v3}});
        }
    }

    return mesh;
}

RTCScene createEmbreeScene(RTCDevice device, const std::shared_ptr<ptm::Mesh>& mesh) {
    RTCScene scene = rtcNewScene(device);

    RTCGeometry geom = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);

    float* vertices = static_cast<float*>(rtcSetNewGeometryBuffer(
        geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3,
        3 * sizeof(float), mesh->vertices.size()));

    for (size_t i = 0; i < mesh->vertices.size(); ++i) {
        vertices[3 * i + 0] = static_cast<float>(mesh->vertices[i][0]);
        vertices[3 * i + 1] = static_cast<float>(mesh->vertices[i][1]);
        vertices[3 * i + 2] = static_cast<float>(mesh->vertices[i][2]);
    }

    unsigned* indices = static_cast<unsigned*>(rtcSetNewGeometryBuffer(
        geom, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3,
        3 * sizeof(unsigned), mesh->faces.size()));

    for (size_t i = 0; i < mesh->faces.size(); ++i) {
        indices[3 * i + 0] = static_cast<unsigned>(mesh->faces[i][0]);
        indices[3 * i + 1] = static_cast<unsigned>(mesh->faces[i][1]);
        indices[3 * i + 2] = static_cast<unsigned>(mesh->faces[i][2]);
    }

    rtcCommitGeometry(geom);
    rtcAttachGeometry(scene, geom);
    rtcReleaseGeometry(geom);
    rtcCommitScene(scene);

    return scene;
}

struct ProfilingResult {
    int num_faces;
    double total_time_ms;
    double precompute_time_ms;
    double quadtree_time_ms;
    double block_build_time_ms;

    // From counters
    size_t view_factor_calls;
    size_t ray_traces;
    size_t culled_total;
    double cull_rate_percent;

    double geometry_time_ms;
    double ray_trace_time_ms;
    double svd_time_ms;

    size_t svd_calls;
};

ProfilingResult runProfiling(int n_rings, int n_azimuth, double tolerance) {
    ProfilingResult result;
    resetCounters();

    // Generate mesh
    double diameter = 1000.0;
    double depth_to_diameter = 0.2;
    auto mesh = generateBowlCraterMesh(diameter, depth_to_diameter, n_rings, n_azimuth);
    result.num_faces = static_cast<int>(mesh->faces.size());

    std::cout << "\n--- Profiling " << result.num_faces << " faces ---" << std::endl;

    // Create Embree scene
    RTCDevice device = rtcNewDevice(nullptr);
    RTCScene scene = createEmbreeScene(device, mesh);

    // Configure
    RadiationConfig config;
    config.enabled = true;
    config.verbose = false;
    config.visibility_threshold = 1e-6;

    HMatrixConfig hconfig;
    hconfig.compression_tolerance = tolerance;
    hconfig.verbose = false;
    hconfig.max_tree_depth = 10;
    hconfig.min_block_size = 64;

    // Build with timing
    auto t_start = std::chrono::high_resolution_clock::now();

    HMatrixViewFactor hmatrix;
    hmatrix.build(mesh, device, scene, config, hconfig);

    auto t_end = std::chrono::high_resolution_clock::now();
    result.total_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    // Collect results
    result.view_factor_calls = g_view_factor_calls.load();
    result.ray_traces = g_ray_trace_count.load();
    result.culled_total = g_culled_backface.load() + g_culled_grazing.load() + g_culled_distance.load();
    result.cull_rate_percent = 100.0 * result.culled_total / std::max(size_t(1), result.view_factor_calls);

    result.geometry_time_ms = g_time_geometry_us.load() / 1000.0;
    result.ray_trace_time_ms = g_time_ray_trace_us.load() / 1000.0;
    result.svd_time_ms = g_time_svd_us.load() / 1000.0;
    result.svd_calls = g_svd_calls.load();

    // Cleanup
    rtcReleaseScene(scene);
    rtcReleaseDevice(device);

    return result;
}

} // namespace profile
} // namespace radiation

int main(int argc, char** argv) {
    std::cout << "╔═══════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║  H-Matrix View Factor Profiler                                 ║" << std::endl;
    std::cout << "║  Identifies computational bottlenecks                          ║" << std::endl;
    std::cout << "╚═══════════════════════════════════════════════════════════════╝" << std::endl;

    std::vector<radiation::profile::ProfilingResult> results;

    // Test different mesh sizes
    std::vector<std::pair<int, int>> mesh_sizes = {
        {10, 24},   // ~480 faces
        {15, 36},   // ~1080 faces
        {20, 48},   // ~1920 faces
        {25, 60},   // ~3000 faces
    };

    double tolerance = 1e-2;

    for (const auto& [n_rings, n_azimuth] : mesh_sizes) {
        auto result = radiation::profile::runProfiling(n_rings, n_azimuth, tolerance);
        results.push_back(result);
        radiation::profile::printCounters();
    }

    // Summary table
    std::cout << "\n\n========== PROFILING SUMMARY ==========" << std::endl;
    std::cout << std::setw(10) << "Faces"
              << std::setw(12) << "Total(ms)"
              << std::setw(12) << "VF Calls"
              << std::setw(12) << "Ray Traces"
              << std::setw(10) << "Cull %"
              << std::setw(12) << "Geom(ms)"
              << std::setw(12) << "Ray(ms)"
              << std::setw(12) << "SVD(ms)"
              << std::setw(10) << "SVD#"
              << std::endl;
    std::cout << std::string(102, '-') << std::endl;

    for (const auto& r : results) {
        std::cout << std::setw(10) << r.num_faces
                  << std::setw(12) << std::fixed << std::setprecision(1) << r.total_time_ms
                  << std::setw(12) << r.view_factor_calls
                  << std::setw(12) << r.ray_traces
                  << std::setw(10) << std::setprecision(1) << r.cull_rate_percent
                  << std::setw(12) << std::setprecision(1) << r.geometry_time_ms
                  << std::setw(12) << r.ray_trace_time_ms
                  << std::setw(12) << r.svd_time_ms
                  << std::setw(10) << r.svd_calls
                  << std::endl;
    }

    // Analysis
    std::cout << "\n\n========== BOTTLENECK ANALYSIS ==========" << std::endl;
    if (!results.empty()) {
        const auto& r = results.back();  // Use largest mesh
        double accounted = r.geometry_time_ms + r.ray_trace_time_ms + r.svd_time_ms;
        double overhead = r.total_time_ms - accounted;

        std::cout << "For " << r.num_faces << " faces:" << std::endl;
        std::cout << "  Geometry checks: " << std::setprecision(1) << (100.0 * r.geometry_time_ms / r.total_time_ms) << "% of total" << std::endl;
        std::cout << "  Ray tracing:     " << (100.0 * r.ray_trace_time_ms / r.total_time_ms) << "% of total" << std::endl;
        std::cout << "  SVD compression: " << (100.0 * r.svd_time_ms / r.total_time_ms) << "% of total" << std::endl;
        std::cout << "  Other overhead:  " << (100.0 * overhead / r.total_time_ms) << "% of total" << std::endl;

        std::cout << "\nCulling effectiveness:" << std::endl;
        std::cout << "  " << r.cull_rate_percent << "% of view factor pairs culled before ray tracing" << std::endl;
        std::cout << "  Only " << r.ray_traces << " ray traces needed out of " << r.view_factor_calls << " pairs" << std::endl;
    }

    std::cout << "\nProfiling completed." << std::endl;
    return 0;
}
