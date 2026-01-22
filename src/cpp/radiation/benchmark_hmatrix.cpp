/**
 * @file benchmark_hmatrix.cpp
 * @brief Benchmark comparing H-matrix vs sparse view factor performance
 *
 * Tests memory usage, compression ratios, and MVP performance for
 * different mesh sizes based on Potter et al. (2023) approach.
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <cmath>
#include <memory>
#include <iomanip>
#include <chrono>

#include "../mesh/mesh_utils.h"
#include "radiation_config.h"
#include "view_factor.h"
#include "hmatrix_view_factor.h"
#include "radiation_exchange.h"

#include <embree4/rtcore.h>

namespace radiation {
namespace benchmark {

constexpr double PI = 3.14159265358979323846;

/**
 * @brief Generate a bowl-shaped crater mesh
 */
std::shared_ptr<ptm::Mesh> generateBowlCraterMesh(
    double diameter,
    double depth_to_diameter,
    int n_rings,
    int n_azimuth)
{
    auto mesh = std::make_shared<ptm::Mesh>();

    double d = depth_to_diameter * diameter;
    double r_rim = diameter / 2.0;
    double R = (d * d + r_rim * r_rim) / (2.0 * d);
    double theta_max = std::asin(r_rim / R);
    double z_center = R - d;

    // Center vertex (crater floor)
    mesh->vertices.push_back({{0.0, 0.0, -d}});

    // Generate rings
    for (int ring = 1; ring <= n_rings; ++ring) {
        double t = static_cast<double>(ring) / n_rings;
        double theta = t * theta_max;
        double r = R * std::sin(theta);
        double z = z_center - R * std::cos(theta);

        for (int az = 0; az < n_azimuth; ++az) {
            double phi = 2.0 * PI * az / n_azimuth;
            mesh->vertices.push_back({{r * std::cos(phi), r * std::sin(phi), z}});
        }
    }

    // Generate faces - center to first ring
    for (int az = 0; az < n_azimuth; ++az) {
        int next_az = (az + 1) % n_azimuth;
        mesh->faces.push_back({{0, 1 + az, 1 + next_az}});
    }

    // Connect subsequent rings
    for (int ring = 1; ring < n_rings; ++ring) {
        int ring_start = 1 + (ring - 1) * n_azimuth;
        int next_ring_start = 1 + ring * n_azimuth;

        for (int az = 0; az < n_azimuth; ++az) {
            int next_az = (az + 1) % n_azimuth;
            int v0 = ring_start + az;
            int v1 = ring_start + next_az;
            int v2 = next_ring_start + az;
            int v3 = next_ring_start + next_az;

            mesh->faces.push_back({{v0, v2, v1}});
            mesh->faces.push_back({{v1, v2, v3}});
        }
    }

    // Compute face properties
    mesh->face_centers.resize(mesh->faces.size());
    mesh->face_normals.resize(mesh->faces.size());

    for (size_t i = 0; i < mesh->faces.size(); ++i) {
        const auto& f = mesh->faces[i];
        const auto& v0 = mesh->vertices[f[0]];
        const auto& v1 = mesh->vertices[f[1]];
        const auto& v2 = mesh->vertices[f[2]];

        mesh->face_centers[i] = {{
            (v0[0] + v1[0] + v2[0]) / 3.0,
            (v0[1] + v1[1] + v2[1]) / 3.0,
            (v0[2] + v1[2] + v2[2]) / 3.0
        }};

        std::array<double, 3> e1 = {{v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2]}};
        std::array<double, 3> e2 = {{v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2]}};
        mesh->face_normals[i] = {{
            e1[1]*e2[2] - e1[2]*e2[1],
            e1[2]*e2[0] - e1[0]*e2[2],
            e1[0]*e2[1] - e1[1]*e2[0]
        }};

        double len = std::sqrt(
            mesh->face_normals[i][0]*mesh->face_normals[i][0] +
            mesh->face_normals[i][1]*mesh->face_normals[i][1] +
            mesh->face_normals[i][2]*mesh->face_normals[i][2]);

        if (len > 1e-10) {
            mesh->face_normals[i][0] /= len;
            mesh->face_normals[i][1] /= len;
            mesh->face_normals[i][2] /= len;
        }

        // Orient normal toward sphere center
        const auto& c = mesh->face_centers[i];
        double dot = mesh->face_normals[i][0] * (0 - c[0]) +
                     mesh->face_normals[i][1] * (0 - c[1]) +
                     mesh->face_normals[i][2] * (z_center - c[2]);
        if (dot < 0) {
            mesh->face_normals[i][0] *= -1;
            mesh->face_normals[i][1] *= -1;
            mesh->face_normals[i][2] *= -1;
        }
    }

    return mesh;
}

/**
 * @brief Create Embree scene from mesh
 */
RTCScene createEmbreeScene(RTCDevice device, const std::shared_ptr<ptm::Mesh>& mesh) {
    RTCScene scene = rtcNewScene(device);

    RTCGeometry geom = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);

    float* vertices = static_cast<float*>(
        rtcSetNewGeometryBuffer(geom, RTC_BUFFER_TYPE_VERTEX, 0,
                                RTC_FORMAT_FLOAT3, 3 * sizeof(float),
                                mesh->vertices.size()));

    for (size_t i = 0; i < mesh->vertices.size(); ++i) {
        vertices[3*i + 0] = static_cast<float>(mesh->vertices[i][0]);
        vertices[3*i + 1] = static_cast<float>(mesh->vertices[i][1]);
        vertices[3*i + 2] = static_cast<float>(mesh->vertices[i][2]);
    }

    unsigned int* indices = static_cast<unsigned int*>(
        rtcSetNewGeometryBuffer(geom, RTC_BUFFER_TYPE_INDEX, 0,
                                RTC_FORMAT_UINT3, 3 * sizeof(unsigned int),
                                mesh->faces.size()));

    for (size_t i = 0; i < mesh->faces.size(); ++i) {
        indices[3*i + 0] = static_cast<unsigned int>(mesh->faces[i][0]);
        indices[3*i + 1] = static_cast<unsigned int>(mesh->faces[i][1]);
        indices[3*i + 2] = static_cast<unsigned int>(mesh->faces[i][2]);
    }

    rtcCommitGeometry(geom);
    rtcAttachGeometry(scene, geom);
    rtcReleaseGeometry(geom);
    rtcCommitScene(scene);

    return scene;
}

struct BenchmarkResult {
    size_t n_faces;
    double sparse_build_time_ms;
    double hmatrix_build_time_ms;
    size_t sparse_memory_bytes;
    size_t hmatrix_memory_bytes;
    double compression_ratio;
    double sparse_mvp_time_ms;
    double hmatrix_mvp_time_ms;
    double mvp_speedup;
    size_t num_zero_blocks;
    size_t num_sparse_blocks;
    size_t num_lowrank_blocks;
    size_t num_hierarchical_blocks;
};

/**
 * @brief Run benchmark for a given mesh size
 */
BenchmarkResult runBenchmark(int n_rings, int n_azimuth, double hmatrix_tolerance) {
    BenchmarkResult result;

    double diameter = 1000.0;
    double depth_to_diameter = 0.2;

    std::cout << "\n--- Benchmark: " << n_rings << " rings x " << n_azimuth << " azimuth ---" << std::endl;

    // Generate mesh
    auto mesh = generateBowlCraterMesh(diameter, depth_to_diameter, n_rings, n_azimuth);
    result.n_faces = mesh->faces.size();
    std::cout << "  Faces: " << result.n_faces << std::endl;

    // Create Embree device and scene
    RTCDevice device = rtcNewDevice(nullptr);
    RTCScene scene = createEmbreeScene(device, mesh);

    // Configure radiation
    RadiationConfig config;
    config.enabled = true;
    config.verbose = false;
    config.max_neighbors = 5000;
    config.search_radius_factor = 50.0;
    config.visibility_threshold = 1e-8;

    // ========== Sparse View Factor Benchmark ==========
    std::cout << "  Computing sparse view factors..." << std::flush;
    auto t1 = std::chrono::high_resolution_clock::now();

    ViewFactorCalculator calculator(mesh, device, scene, config);
    ViewFactorMatrix sparse_F = calculator.computeAll();

    auto t2 = std::chrono::high_resolution_clock::now();
    result.sparse_build_time_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    result.sparse_memory_bytes = sparse_F.memoryUsage();

    std::cout << " done (" << result.sparse_build_time_ms << " ms)" << std::endl;
    std::cout << "    Non-zeros: " << sparse_F.numNonZeros() << std::endl;
    std::cout << "    Memory: " << (result.sparse_memory_bytes / 1024.0 / 1024.0) << " MB" << std::endl;

    // ========== H-Matrix View Factor Benchmark ==========
    std::cout << "  Computing H-matrix view factors (tol=" << hmatrix_tolerance << ")..." << std::flush;
    t1 = std::chrono::high_resolution_clock::now();

    HMatrixConfig hconfig;
    hconfig.compression_tolerance = hmatrix_tolerance;
    hconfig.max_tree_depth = 10;
    hconfig.min_block_size = 64;
    hconfig.max_svd_rank = 50;
    hconfig.verbose = false;

    HMatrixViewFactor hmatrix_F;
    hmatrix_F.build(mesh, device, scene, config, hconfig);

    t2 = std::chrono::high_resolution_clock::now();
    result.hmatrix_build_time_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    result.hmatrix_memory_bytes = hmatrix_F.memoryBytes();

    std::cout << " done (" << result.hmatrix_build_time_ms << " ms)" << std::endl;
    std::cout << "    Memory: " << (result.hmatrix_memory_bytes / 1024.0 / 1024.0) << " MB" << std::endl;

    result.compression_ratio = static_cast<double>(result.sparse_memory_bytes) / result.hmatrix_memory_bytes;
    std::cout << "    Compression ratio: " << result.compression_ratio << "x" << std::endl;

    // Print H-matrix statistics
    hmatrix_F.printStatistics();

    // ========== MVP Benchmark ==========
    std::vector<double> x(result.n_faces, 1.0);
    std::vector<double> y_sparse(result.n_faces);
    std::vector<double> y_hmatrix(result.n_faces);

    int n_mvp_iterations = 10;

    // Sparse MVP
    std::cout << "  Benchmarking sparse MVP (" << n_mvp_iterations << " iterations)..." << std::flush;
    t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_mvp_iterations; ++i) {
        sparse_F.matvec(x, y_sparse);
    }
    t2 = std::chrono::high_resolution_clock::now();
    result.sparse_mvp_time_ms = std::chrono::duration<double, std::milli>(t2 - t1).count() / n_mvp_iterations;
    std::cout << " " << result.sparse_mvp_time_ms << " ms/MVP" << std::endl;

    // H-matrix MVP
    std::cout << "  Benchmarking H-matrix MVP (" << n_mvp_iterations << " iterations)..." << std::flush;
    t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_mvp_iterations; ++i) {
        hmatrix_F.matvec(x, y_hmatrix);
    }
    t2 = std::chrono::high_resolution_clock::now();
    result.hmatrix_mvp_time_ms = std::chrono::duration<double, std::milli>(t2 - t1).count() / n_mvp_iterations;
    std::cout << " " << result.hmatrix_mvp_time_ms << " ms/MVP" << std::endl;

    result.mvp_speedup = result.sparse_mvp_time_ms / result.hmatrix_mvp_time_ms;
    std::cout << "    MVP speedup: " << result.mvp_speedup << "x" << std::endl;

    // Check accuracy
    double max_error = 0.0;
    double max_rel_error = 0.0;
    for (size_t i = 0; i < result.n_faces; ++i) {
        double error = std::abs(y_sparse[i] - y_hmatrix[i]);
        double rel_error = error / std::max(std::abs(y_sparse[i]), 1e-10);
        max_error = std::max(max_error, error);
        max_rel_error = std::max(max_rel_error, rel_error);
    }
    std::cout << "    MVP max absolute error: " << max_error << std::endl;
    std::cout << "    MVP max relative error: " << (max_rel_error * 100) << "%" << std::endl;

    // Cleanup
    rtcReleaseScene(scene);
    rtcReleaseDevice(device);

    return result;
}

} // namespace benchmark
} // namespace radiation

int main(int argc, char** argv) {
    std::cout << "╔═══════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║  H-Matrix View Factor Benchmark                                ║" << std::endl;
    std::cout << "║  Based on Potter et al. (2023) Hierarchical Matrices          ║" << std::endl;
    std::cout << "╚═══════════════════════════════════════════════════════════════╝" << std::endl;

    std::vector<radiation::benchmark::BenchmarkResult> results;

    // Test different mesh sizes (start small to avoid memory issues)
    std::vector<std::pair<int, int>> mesh_sizes = {
        {10, 24},   // ~480 faces
        {15, 36},   // ~1080 faces
        {20, 48},   // ~1920 faces
        {25, 60},   // ~3000 faces
    };

    double tolerance = 1e-2;

    for (const auto& [n_rings, n_azimuth] : mesh_sizes) {
        auto result = radiation::benchmark::runBenchmark(n_rings, n_azimuth, tolerance);
        results.push_back(result);
    }

    // Print summary table
    std::cout << "\n\n========== BENCHMARK SUMMARY ==========" << std::endl;
    std::cout << std::setw(10) << "Faces"
              << std::setw(15) << "Sparse(MB)"
              << std::setw(15) << "H-Matrix(MB)"
              << std::setw(12) << "Compress"
              << std::setw(15) << "Sparse MVP"
              << std::setw(15) << "H-Mat MVP"
              << std::setw(12) << "Speedup"
              << std::endl;
    std::cout << std::string(94, '-') << std::endl;

    for (const auto& r : results) {
        std::cout << std::setw(10) << r.n_faces
                  << std::setw(15) << std::fixed << std::setprecision(3)
                  << (r.sparse_memory_bytes / 1024.0 / 1024.0)
                  << std::setw(15) << (r.hmatrix_memory_bytes / 1024.0 / 1024.0)
                  << std::setw(12) << std::setprecision(2) << r.compression_ratio << "x"
                  << std::setw(15) << std::setprecision(2) << r.sparse_mvp_time_ms << " ms"
                  << std::setw(15) << r.hmatrix_mvp_time_ms << " ms"
                  << std::setw(12) << r.mvp_speedup << "x"
                  << std::endl;
    }

    // Test different tolerances (using smaller mesh)
    std::cout << "\n\n========== TOLERANCE STUDY (1500 faces) ==========" << std::endl;
    std::vector<double> tolerances = {1e-1, 5e-2, 1e-2, 5e-3, 1e-3};

    for (double tol : tolerances) {
        std::cout << "\nTolerance: " << tol << std::endl;
        auto result = radiation::benchmark::runBenchmark(15, 36, tol);
    }

    // Test serialization
    std::cout << "\n\n========== SERIALIZATION TEST ==========" << std::endl;
    {
        // Build H-matrix
        double diameter = 1000.0;
        double depth_to_diameter = 0.2;
        auto mesh = radiation::benchmark::generateBowlCraterMesh(diameter, depth_to_diameter, 15, 36);

        RTCDevice device = rtcNewDevice(nullptr);
        RTCScene scene = radiation::benchmark::createEmbreeScene(device, mesh);

        radiation::RadiationConfig config;
        config.enabled = true;
        config.verbose = true;

        radiation::HMatrixConfig hconfig;
        hconfig.compression_tolerance = 0.01;
        hconfig.verbose = true;

        radiation::HMatrixViewFactor hmatrix;
        hmatrix.build(mesh, device, scene, config, hconfig);

        // Test save
        std::string filename = "/tmp/test_hmatrix.bin";
        std::cout << "\nSaving H-matrix to " << filename << "..." << std::endl;
        auto t1 = std::chrono::high_resolution_clock::now();
        bool saved = hmatrix.writeBinary(filename);
        auto t2 = std::chrono::high_resolution_clock::now();
        double save_time = std::chrono::duration<double, std::milli>(t2 - t1).count();
        std::cout << "  Save " << (saved ? "succeeded" : "FAILED") << " in " << save_time << " ms" << std::endl;

        // Test load
        std::cout << "\nLoading H-matrix from " << filename << "..." << std::endl;
        radiation::HMatrixViewFactor hmatrix_loaded;
        t1 = std::chrono::high_resolution_clock::now();
        bool loaded = hmatrix_loaded.readBinary(filename);
        t2 = std::chrono::high_resolution_clock::now();
        double load_time = std::chrono::duration<double, std::milli>(t2 - t1).count();
        std::cout << "  Load " << (loaded ? "succeeded" : "FAILED") << " in " << load_time << " ms" << std::endl;

        if (saved && loaded) {
            // Verify loaded matrix produces same results
            std::vector<double> x(mesh->faces.size(), 1.0);
            std::vector<double> y_orig(mesh->faces.size());
            std::vector<double> y_loaded(mesh->faces.size());

            hmatrix.matvec(x, y_orig);
            hmatrix_loaded.matvec(x, y_loaded);

            double max_diff = 0.0;
            for (size_t i = 0; i < y_orig.size(); ++i) {
                max_diff = std::max(max_diff, std::abs(y_orig[i] - y_loaded[i]));
            }
            std::cout << "  MVP difference (orig vs loaded): " << max_diff << std::endl;
            std::cout << "  Serialization test " << (max_diff < 1e-10 ? "PASSED" : "FAILED") << std::endl;
        }

        rtcReleaseScene(scene);
        rtcReleaseDevice(device);
    }

    // Compare Randomized SVD vs JacobiSVD
    std::cout << "\n\n========== SVD METHOD COMPARISON ==========" << std::endl;
    {
        double diameter = 1000.0;
        double depth_to_diameter = 0.2;
        auto mesh = radiation::benchmark::generateBowlCraterMesh(diameter, depth_to_diameter, 20, 48);
        std::cout << "Testing with " << mesh->faces.size() << " faces" << std::endl;

        RTCDevice device = rtcNewDevice(nullptr);
        RTCScene scene = radiation::benchmark::createEmbreeScene(device, mesh);

        radiation::RadiationConfig config;
        config.enabled = true;
        config.verbose = false;

        // Test Randomized SVD (default)
        {
            radiation::HMatrixConfig hconfig;
            hconfig.compression_tolerance = 0.01;
            hconfig.verbose = false;
            hconfig.svd_method = radiation::SVDMethod::Randomized;

            radiation::HMatrixViewFactor hmatrix;
            auto t1 = std::chrono::high_resolution_clock::now();
            hmatrix.build(mesh, device, scene, config, hconfig);
            auto t2 = std::chrono::high_resolution_clock::now();
            double build_time = std::chrono::duration<double, std::milli>(t2 - t1).count();

            // Test MVP accuracy
            std::vector<double> x(mesh->faces.size(), 1.0);
            std::vector<double> y(mesh->faces.size(), 0.0);
            hmatrix.matvec(x, y);
            double sum = 0.0;
            for (double v : y) sum += v;

            std::cout << "  Randomized SVD: " << std::fixed << std::setprecision(2)
                      << build_time << " ms, MVP sum = " << sum << std::endl;
        }

        // Test JacobiSVD
        {
            radiation::HMatrixConfig hconfig;
            hconfig.compression_tolerance = 0.01;
            hconfig.verbose = false;
            hconfig.svd_method = radiation::SVDMethod::Jacobi;

            radiation::HMatrixViewFactor hmatrix;
            auto t1 = std::chrono::high_resolution_clock::now();
            hmatrix.build(mesh, device, scene, config, hconfig);
            auto t2 = std::chrono::high_resolution_clock::now();
            double build_time = std::chrono::duration<double, std::milli>(t2 - t1).count();

            // Test MVP accuracy
            std::vector<double> x(mesh->faces.size(), 1.0);
            std::vector<double> y(mesh->faces.size(), 0.0);
            hmatrix.matvec(x, y);
            double sum = 0.0;
            for (double v : y) sum += v;

            std::cout << "  JacobiSVD:      " << std::fixed << std::setprecision(2)
                      << build_time << " ms, MVP sum = " << sum << std::endl;
        }

        rtcReleaseScene(scene);
        rtcReleaseDevice(device);
    }

    std::cout << "\nBenchmark completed." << std::endl;
    return 0;
}
