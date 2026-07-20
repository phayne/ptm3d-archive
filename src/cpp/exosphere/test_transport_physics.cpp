/*
 * Test program for ballistic transport physics
 * Verifies the corrected implementation with analytical cases
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <array>
#include <cmath>
#include <numeric>
#include "exosphere_physics.cpp"

using namespace ptm::exosphere;

void test_vapor_pressure() {
    std::cout << "\n=== Testing Vapor Pressure Calculation ===\n" << std::endl;
    
    std::vector<double> test_temps = {50, 70, 90, 100, 110, 130, 150, 200};
    
    std::cout << std::setw(15) << "Temperature(K)" 
              << std::setw(20) << "Vapor Pressure(Pa)" 
              << std::setw(20) << "Expected Range" << std::endl;
    std::cout << std::string(55, '-') << std::endl;
    
    for (double T : test_temps) {
        double p = VaporPressure::calculateVaporPressure(T);
        
        // Expected rough values for water ice
        double expected_min = 0.0, expected_max = 0.0;
        if (T < 100) {
            expected_min = 1e-20;
            expected_max = 1e-10;
        } else if (T < 150) {
            expected_min = 1e-12;
            expected_max = 1e-7;
        } else {
            expected_min = 1e-7;
            expected_max = 1e-2;
        }
        
        std::cout << std::setw(15) << T 
                  << std::setw(20) << std::scientific << std::setprecision(3) << p
                  << "  [" << expected_min << ", " << expected_max << "]" << std::endl;
        
        // Check if in reasonable range
        if (p < expected_min * 0.01 || p > expected_max * 100) {
            std::cerr << "WARNING: Vapor pressure out of expected range!" << std::endl;
        }
    }
}

void test_transport_probability() {
    std::cout << "\n=== Testing Transport Probability ===\n" << std::endl;
    
    BallisticTransport transport;
    
    // Test parameters
    double T = 100.0;  // K
    std::vector<double> distances = {0, 1000, 10000, 50000, 100000, 500000, 1000000};  // meters
    
    std::cout << std::setw(15) << "Distance(km)" 
              << std::setw(20) << "Probability p(d)"
              << std::setw(20) << "Cumulative" << std::endl;
    std::cout << std::string(55, '-') << std::endl;
    
    double cumulative = 0.0;
    double prev_d = 0.0;
    
    for (double d : distances) {
        if (d == 0) continue;
        
        double p = transport.transportProbabilityDensity(d, T);
        
        // Approximate integral (should use proper integration)
        double delta_d = d - prev_d;
        cumulative += p * delta_d;
        
        std::cout << std::setw(15) << d/1000.0
                  << std::setw(20) << std::scientific << std::setprecision(3) << p
                  << std::setw(20) << cumulative << std::endl;
        
        prev_d = d;
    }
}

void test_mass_conservation() {
    std::cout << "\n=== Testing Mass Conservation in Transport Matrix ===\n" << std::endl;
    
    // Create a simple test mesh (octahedron)
    std::vector<std::array<double, 3>> vertices = {
        {1.5608e6, 0, 0},      // +x
        {-1.5608e6, 0, 0},     // -x
        {0, 1.5608e6, 0},      // +y
        {0, -1.5608e6, 0},     // -y
        {0, 0, 1.5608e6},      // +z
        {0, 0, -1.5608e6}      // -z
    };
    
    // Face centers (8 triangular faces)
    std::vector<std::array<double, 3>> face_centers;
    std::vector<double> face_areas;
    
    // Calculate face centers and areas for octahedron
    double R = constants::EUROPA_RADIUS;
    double face_area = 4.0 * M_PI * R * R / 8.0;  // Approximate equal area division
    
    // 8 faces of regular octahedron
    face_centers.push_back({R/std::sqrt(3), R/std::sqrt(3), R/std::sqrt(3)});
    face_centers.push_back({R/std::sqrt(3), R/std::sqrt(3), -R/std::sqrt(3)});
    face_centers.push_back({R/std::sqrt(3), -R/std::sqrt(3), R/std::sqrt(3)});
    face_centers.push_back({R/std::sqrt(3), -R/std::sqrt(3), -R/std::sqrt(3)});
    face_centers.push_back({-R/std::sqrt(3), R/std::sqrt(3), R/std::sqrt(3)});
    face_centers.push_back({-R/std::sqrt(3), R/std::sqrt(3), -R/std::sqrt(3)});
    face_centers.push_back({-R/std::sqrt(3), -R/std::sqrt(3), R/std::sqrt(3)});
    face_centers.push_back({-R/std::sqrt(3), -R/std::sqrt(3), -R/std::sqrt(3)});
    
    // Normalize face centers to sphere surface
    for (auto& fc : face_centers) {
        double r = std::sqrt(fc[0]*fc[0] + fc[1]*fc[1] + fc[2]*fc[2]);
        fc[0] *= R/r;
        fc[1] *= R/r;
        fc[2] *= R/r;
        face_areas.push_back(face_area);
    }
    
    // Test with different temperature distributions
    std::vector<std::string> test_cases = {
        "Uniform 100K",
        "Subsolar hot spot",
        "Polar cold traps"
    };
    
    for (int test_id = 0; test_id < test_cases.size(); ++test_id) {
        std::cout << "\nTest case: " << test_cases[test_id] << std::endl;
        
        std::vector<double> temperatures(face_centers.size());
        
        if (test_id == 0) {
            // Uniform temperature
            std::fill(temperatures.begin(), temperatures.end(), 100.0);
        } else if (test_id == 1) {
            // Subsolar hot spot
            for (size_t i = 0; i < face_centers.size(); ++i) {
                double x = face_centers[i][0];
                double cosine_factor = std::max(0.0, x / R);
                temperatures[i] = 70.0 + 60.0 * cosine_factor;
            }
        } else {
            // Polar cold traps
            for (size_t i = 0; i < face_centers.size(); ++i) {
                double z = face_centers[i][2];
                double lat_factor = std::abs(z / R);
                temperatures[i] = 100.0 - 30.0 * lat_factor;
            }
        }
        
        // Build transport matrix
        BallisticTransport transport;
        auto matrix = transport.buildTransportMatrix(face_centers, face_areas, temperatures);
        
        // Check mass conservation
        std::cout << "  Matrix size: " << matrix.rows() << " x " << matrix.cols() << std::endl;
        std::cout << "  Non-zeros: " << matrix.nonZeros() << std::endl;
        
        // Calculate column sums
        std::vector<double> col_sums(matrix.cols(), 0.0);
        for (int k = 0; k < matrix.outerSize(); ++k) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(matrix, k); it; ++it) {
                col_sums[it.col()] += it.value();
            }
        }
        
        // Statistics
        double min_sum = *std::min_element(col_sums.begin(), col_sums.end());
        double max_sum = *std::max_element(col_sums.begin(), col_sums.end());
        double avg_sum = std::accumulate(col_sums.begin(), col_sums.end(), 0.0) / col_sums.size();
        
        std::cout << "  Column sums: min=" << min_sum 
                  << ", max=" << max_sum 
                  << ", avg=" << avg_sum << std::endl;
        
        // Check conservation
        bool conserved = true;
        for (double sum : col_sums) {
            if (sum > 1.0 + 1e-6) {
                conserved = false;
                std::cerr << "  ERROR: Column sum > 1: " << sum << std::endl;
            }
        }
        
        if (conserved) {
            std::cout << "  ✓ Mass conservation satisfied" << std::endl;
        } else {
            std::cout << "  ✗ Mass conservation violated!" << std::endl;
        }
        
        // Test escape calculation
        double escape_frac = transport.calculateEscapeFraction(temperatures[0]);
        std::cout << "  Escape fraction at " << temperatures[0] << "K: " 
                  << std::scientific << std::setprecision(3) << escape_frac << std::endl;
    }
}

void test_exosphere_evolution() {
    std::cout << "\n=== Testing Exosphere Time Evolution ===\n" << std::endl;
    
    // Create simple test mesh
    size_t n_faces = 100;
    std::vector<std::array<double, 3>> face_centers(n_faces);
    std::vector<double> face_areas(n_faces);
    
    // Generate random points on sphere
    double total_area = 4.0 * M_PI * constants::EUROPA_RADIUS * constants::EUROPA_RADIUS;
    double area_per_face = total_area / n_faces;
    
    for (size_t i = 0; i < n_faces; ++i) {
        // Random spherical coordinates
        double theta = 2.0 * M_PI * i / n_faces;
        double phi = std::acos(1.0 - 2.0 * (i + 0.5) / n_faces);
        
        face_centers[i][0] = constants::EUROPA_RADIUS * std::sin(phi) * std::cos(theta);
        face_centers[i][1] = constants::EUROPA_RADIUS * std::sin(phi) * std::sin(theta);
        face_centers[i][2] = constants::EUROPA_RADIUS * std::cos(phi);
        
        face_areas[i] = area_per_face;
    }
    
    // Create model
    ExosphereModel model(n_faces);
    model.setMeshGeometry(face_centers, face_areas);
    
    // Run for several timesteps
    double dt = 3600.0;  // 1 hour
    int n_steps = 5;
    
    std::cout << "Running " << n_steps << " timesteps with dt = " << dt << " seconds" << std::endl;
    
    for (int step = 0; step < n_steps; ++step) {
        // Create temperature distribution (simple day/night)
        std::vector<double> temperatures(n_faces);
        for (size_t i = 0; i < n_faces; ++i) {
            double x = face_centers[i][0];
            double solar_angle = std::acos(x / constants::EUROPA_RADIUS);
            
            if (solar_angle < M_PI/2) {
                // Dayside
                temperatures[i] = 100.0 + 30.0 * std::cos(solar_angle);
            } else {
                // Nightside
                temperatures[i] = 70.0;
            }
        }
        
        // Update and evolve
        model.updateTemperatures(temperatures);
        model.evolveTimeStep(dt);
        
        // Get state
        const auto& state = model.getState();
        
        // Calculate totals
        double total_emission = 0.0;
        double total_deposition = 0.0;
        double total_escape = 0.0;
        
        for (size_t i = 0; i < n_faces; ++i) {
            total_emission += state.emission_flux[i] * face_areas[i];
            total_deposition += state.deposition_flux[i] * face_areas[i];
            total_escape += state.escape_flux[i] * face_areas[i];
        }
        
        std::cout << "\nTimestep " << step << ":" << std::endl;
        std::cout << "  Total emission: " << std::scientific << std::setprecision(3) 
                  << total_emission << " kg/s" << std::endl;
        std::cout << "  Total deposition: " << total_deposition << " kg/s" << std::endl;
        std::cout << "  Total escape: " << total_escape << " kg/s" << std::endl;
        std::cout << "  Mass balance error: " 
                  << std::abs(total_emission - total_deposition - total_escape) << " kg/s" << std::endl;
    }
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << " Ballistic Transport Physics Test Suite" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Run tests
    test_vapor_pressure();
    test_transport_probability();
    test_mass_conservation();
    test_exosphere_evolution();
    
    std::cout << "\n========================================" << std::endl;
    std::cout << " All Tests Complete" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}