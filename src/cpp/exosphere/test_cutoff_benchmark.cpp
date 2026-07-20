/*
 * Benchmark for kernel cutoff optimization
 * Compares cached matrix build with and without cutoff
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <array>
#include <cmath>
#include <chrono>
#include "exosphere_physics.h"

using namespace ptm::exosphere;

// Generate an icosahedral mesh with N subdivisions
void generateIcosahedralMesh(size_t n_faces,
                             std::vector<std::array<double, 3>>& face_centers,
                             std::vector<double>& face_areas) {
    face_centers.clear();
    face_areas.clear();

    // Simple Fibonacci lattice for quasi-uniform points on sphere
    double R = constants::EUROPA_RADIUS;
    double total_area = 4.0 * M_PI * R * R;
    double area_per_face = total_area / n_faces;

    double golden_ratio = (1.0 + std::sqrt(5.0)) / 2.0;

    for (size_t i = 0; i < n_faces; ++i) {
        double theta = 2.0 * M_PI * i / golden_ratio;
        double phi = std::acos(1.0 - 2.0 * (i + 0.5) / n_faces);

        std::array<double, 3> center;
        center[0] = R * std::sin(phi) * std::cos(theta);
        center[1] = R * std::sin(phi) * std::sin(theta);
        center[2] = R * std::cos(phi);

        face_centers.push_back(center);
        face_areas.push_back(area_per_face);
    }
}

// Generate temperature distribution (subsolar heating)
void generateTemperatures(const std::vector<std::array<double, 3>>& face_centers,
                         std::vector<double>& temperatures,
                         double T_min, double T_max) {
    temperatures.resize(face_centers.size());
    double R = constants::EUROPA_RADIUS;

    for (size_t i = 0; i < face_centers.size(); ++i) {
        double x = face_centers[i][0];
        double cosine_factor = std::max(0.0, x / R);
        temperatures[i] = T_min + (T_max - T_min) * std::pow(cosine_factor, 0.25);
    }
}

int main(int argc, char* argv[]) {
    // Parse arguments
    size_t n_faces = 5000;  // Default mesh size
    if (argc > 1) {
        n_faces = std::atoi(argv[1]);
    }

    std::cout << "========================================" << std::endl;
    std::cout << " Kernel Cutoff Optimization Benchmark" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "\nMesh size: " << n_faces << " faces" << std::endl;
    std::cout << "Total pairs: " << (n_faces * (n_faces - 1)) << std::endl;

    // Generate mesh
    std::vector<std::array<double, 3>> face_centers;
    std::vector<double> face_areas;
    generateIcosahedralMesh(n_faces, face_centers, face_areas);

    // Generate temperature distribution
    std::vector<double> temperatures;
    generateTemperatures(face_centers, temperatures, 50.0, 130.0);

    std::cout << "\nTemperature range: [" << 50.0 << ", " << 130.0 << "] K" << std::endl;

    // Create transport object
    BallisticTransport transport;

    std::cout << "\n--- Test 1: With Cutoff (default) ---" << std::endl;
    {
        auto start = std::chrono::high_resolution_clock::now();
        auto matrix = transport.buildTransportMatrixCached(face_centers, face_areas, temperatures);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "Total time: " << duration.count() << " ms" << std::endl;
        std::cout << "Non-zeros: " << matrix.nonZeros() << std::endl;
        std::cout << "Density: " << std::fixed << std::setprecision(4)
                  << (100.0 * matrix.nonZeros() / (n_faces * n_faces)) << "%" << std::endl;
    }

    std::cout << "\n--- Test 2: Without Cutoff (EXO_NO_CUTOFF=1) ---" << std::endl;
    {
        setenv("EXO_NO_CUTOFF", "1", 1);

        auto start = std::chrono::high_resolution_clock::now();
        auto matrix = transport.buildTransportMatrixCached(face_centers, face_areas, temperatures);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "Total time: " << duration.count() << " ms" << std::endl;
        std::cout << "Non-zeros: " << matrix.nonZeros() << std::endl;

        unsetenv("EXO_NO_CUTOFF");
    }

    std::cout << "\n--- Test 3: With lower xi cutoff (EXO_XI_CUTOFF=10) ---" << std::endl;
    {
        setenv("EXO_XI_CUTOFF", "10", 1);

        auto start = std::chrono::high_resolution_clock::now();
        auto matrix = transport.buildTransportMatrixCached(face_centers, face_areas, temperatures);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "Total time: " << duration.count() << " ms" << std::endl;
        std::cout << "Non-zeros: " << matrix.nonZeros() << std::endl;

        unsetenv("EXO_XI_CUTOFF");
    }

    std::cout << "\n--- Test 4: Spatial Index (O(N*k) complexity) ---" << std::endl;
    {
        // Create new transport object for fresh spatial index
        BallisticTransport transport2;

        auto start = std::chrono::high_resolution_clock::now();
        auto matrix = transport2.buildTransportMatrixIndexed(face_centers, face_areas, temperatures);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "Total time: " << duration.count() << " ms" << std::endl;
        std::cout << "Non-zeros: " << matrix.nonZeros() << std::endl;
        std::cout << "Spatial index memory: " << std::fixed << std::setprecision(2)
                  << transport2.getSpatialIndexMemory() / (1024.0 * 1024.0) << " MB" << std::endl;
    }

    std::cout << "\n--- Test 5: Spatial Index with xi=10 ---" << std::endl;
    {
        setenv("EXO_XI_CUTOFF", "10", 1);

        BallisticTransport transport3;

        auto start = std::chrono::high_resolution_clock::now();
        auto matrix = transport3.buildTransportMatrixIndexed(face_centers, face_areas, temperatures);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "Total time: " << duration.count() << " ms" << std::endl;
        std::cout << "Non-zeros: " << matrix.nonZeros() << std::endl;

        unsetenv("EXO_XI_CUTOFF");
    }

    std::cout << "\n--- Test 6: Smooth Kernel (no shell binning) ---" << std::endl;
    {
        BallisticTransport transport4;

        auto start = std::chrono::high_resolution_clock::now();
        auto matrix = transport4.buildTransportMatrixSmooth(face_centers, face_areas, temperatures);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "Total time: " << duration.count() << " ms" << std::endl;
        std::cout << "Non-zeros: " << matrix.nonZeros() << std::endl;
    }

    std::cout << "\n--- Test 7: Smooth Kernel with xi=10 ---" << std::endl;
    {
        setenv("EXO_XI_CUTOFF", "10", 1);

        BallisticTransport transport5;

        auto start = std::chrono::high_resolution_clock::now();
        auto matrix = transport5.buildTransportMatrixSmooth(face_centers, face_areas, temperatures);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "Total time: " << duration.count() << " ms" << std::endl;
        std::cout << "Non-zeros: " << matrix.nonZeros() << std::endl;

        unsetenv("EXO_XI_CUTOFF");
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << " Benchmark Complete" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
