/*
 * Test program to analyze equilibrium bias in exosphere transport
 *
 * This tests what happens when we iterate the transport matrix to equilibrium
 * with uniform initial conditions. Any systematic bias should show up as
 * pentagon-adjacent faces having different equilibrium mass.
 */

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <array>
#include <cmath>
#include <set>
#include <algorithm>
#include <numeric>
#include <Eigen/Sparse>
#include "exosphere_physics.h"

using namespace ptm::exosphere;

// Simple PLY reader for binary little-endian format
bool readPLYMesh(const std::string& filename,
                 std::vector<std::array<double, 3>>& vertices,
                 std::vector<std::array<int, 3>>& faces) {

    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Cannot open PLY file: " << filename << std::endl;
        return false;
    }

    std::string line;
    int num_vertices = 0;
    int num_faces = 0;
    bool in_header = true;

    while (in_header && std::getline(file, line)) {
        if (line.find("element vertex") != std::string::npos) {
            sscanf(line.c_str(), "element vertex %d", &num_vertices);
        } else if (line.find("element face") != std::string::npos) {
            sscanf(line.c_str(), "element face %d", &num_faces);
        } else if (line == "end_header") {
            in_header = false;
        }
    }

    vertices.resize(num_vertices);
    for (int i = 0; i < num_vertices; ++i) {
        double coords[6];
        file.read(reinterpret_cast<char*>(coords), 6 * sizeof(double));
        vertices[i] = {coords[0], coords[1], coords[2]};
    }

    faces.resize(num_faces);
    for (int i = 0; i < num_faces; ++i) {
        unsigned char count;
        file.read(reinterpret_cast<char*>(&count), 1);
        int indices[3];
        file.read(reinterpret_cast<char*>(indices), 3 * sizeof(int));
        faces[i] = {indices[0], indices[1], indices[2]};
        float lat, lon;
        file.read(reinterpret_cast<char*>(&lat), sizeof(float));
        file.read(reinterpret_cast<char*>(&lon), sizeof(float));
    }

    return true;
}

void calculateFaceGeometry(const std::vector<std::array<double, 3>>& vertices,
                           const std::vector<std::array<int, 3>>& faces,
                           std::vector<std::array<double, 3>>& face_centers,
                           std::vector<double>& face_areas) {

    size_t num_faces = faces.size();
    face_centers.resize(num_faces);
    face_areas.resize(num_faces);

    for (size_t i = 0; i < num_faces; ++i) {
        const auto& f = faces[i];
        const auto& v0 = vertices[f[0]];
        const auto& v1 = vertices[f[1]];
        const auto& v2 = vertices[f[2]];

        face_centers[i] = {
            (v0[0] + v1[0] + v2[0]) / 3.0,
            (v0[1] + v1[1] + v2[1]) / 3.0,
            (v0[2] + v1[2] + v2[2]) / 3.0
        };

        double e1[3] = {v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2]};
        double e2[3] = {v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2]};
        double cross[3] = {
            e1[1]*e2[2] - e1[2]*e2[1],
            e1[2]*e2[0] - e1[0]*e2[2],
            e1[0]*e2[1] - e1[1]*e2[0]
        };
        face_areas[i] = 0.5 * std::sqrt(cross[0]*cross[0] + cross[1]*cross[1] + cross[2]*cross[2]);
    }
}

void identifyPentagonFaces(const std::vector<std::array<int, 3>>& faces,
                           std::set<int>& pentagon_vertices,
                           std::vector<bool>& is_pentagon_face,
                           std::vector<int>& pentagon_vertex_count) {

    std::map<int, int> vertex_face_count;
    for (const auto& face : faces) {
        for (int v : face) {
            vertex_face_count[v]++;
        }
    }

    for (const auto& [v, count] : vertex_face_count) {
        if (count == 5) {
            pentagon_vertices.insert(v);
        }
    }

    is_pentagon_face.resize(faces.size(), false);
    pentagon_vertex_count.resize(faces.size(), 0);

    for (size_t i = 0; i < faces.size(); ++i) {
        int pent_count = 0;
        for (int v : faces[i]) {
            if (pentagon_vertices.count(v)) pent_count++;
        }
        pentagon_vertex_count[i] = pent_count;
        is_pentagon_face[i] = (pent_count > 0);
    }
}

int main(int argc, char* argv[]) {
    std::cout << "========================================" << std::endl;
    std::cout << " Equilibrium Bias Analysis" << std::endl;
    std::cout << "========================================" << std::endl;

    std::string mesh_file = "/Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/europa/heat3d/ptm/build/output/europa_high_res/mesh.ply";
    if (argc > 1) mesh_file = argv[1];

    // Load mesh
    std::vector<std::array<double, 3>> vertices;
    std::vector<std::array<int, 3>> faces;
    if (!readPLYMesh(mesh_file, vertices, faces)) return 1;

    std::cout << "Loaded: " << vertices.size() << " vertices, " << faces.size() << " faces" << std::endl;

    std::vector<std::array<double, 3>> face_centers;
    std::vector<double> face_areas;
    calculateFaceGeometry(vertices, faces, face_centers, face_areas);

    std::set<int> pentagon_vertices;
    std::vector<bool> is_pentagon_face;
    std::vector<int> pentagon_vertex_count;
    identifyPentagonFaces(faces, pentagon_vertices, is_pentagon_face, pentagon_vertex_count);

    size_t n = faces.size();

    // Use uniform temperature to isolate geometric effects
    std::vector<double> temperatures(n, 100.0);

    // Build transport matrix with smooth kernel
    std::cout << "\nBuilding transport matrix with smooth kernel..." << std::endl;
    BallisticTransport transport;
    auto M = transport.buildTransportMatrixSmooth(face_centers, face_areas, temperatures, 1e-12);

    std::cout << "Matrix: " << n << " x " << n << ", " << M.nonZeros() << " non-zeros" << std::endl;

    // === Test 1: Uniform emission, check deposition flux ===
    std::cout << "\n=== Test 1: Uniform Emission Test ===" << std::endl;
    std::cout << "All faces emit flux = 1.0 kg/(m²·s)" << std::endl;

    // Emission vector: emission(i) = flux * area_i
    Eigen::VectorXd emission(n);
    for (size_t i = 0; i < n; ++i) {
        emission(i) = 1.0 * face_areas[i];  // 1.0 kg/(m²·s) * area
    }

    // Deposition
    Eigen::VectorXd deposition = M * emission;

    // Deposition flux
    Eigen::VectorXd deposition_flux(n);
    for (size_t i = 0; i < n; ++i) {
        deposition_flux(i) = deposition(i) / face_areas[i];
    }

    // Analyze by face type
    std::vector<double> flux_pent, flux_regular;
    for (size_t i = 0; i < n; ++i) {
        if (is_pentagon_face[i]) {
            flux_pent.push_back(deposition_flux(i));
        } else {
            flux_regular.push_back(deposition_flux(i));
        }
    }

    auto mean = [](const std::vector<double>& v) {
        return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    };

    double flux_pent_mean = mean(flux_pent);
    double flux_reg_mean = mean(flux_regular);

    std::cout << "Deposition flux (should be ~1.0 for all):" << std::endl;
    std::cout << "  Pentagon-adjacent: " << std::fixed << std::setprecision(6) << flux_pent_mean << std::endl;
    std::cout << "  Regular faces:     " << flux_reg_mean << std::endl;
    std::cout << "  Ratio:             " << (flux_pent_mean / flux_reg_mean) << std::endl;
    std::cout << "  Bias:              " << std::setprecision(4) << ((flux_pent_mean / flux_reg_mean - 1.0) * 100.0) << "%" << std::endl;

    // === Test 2: Iterate to equilibrium ===
    std::cout << "\n=== Test 2: Equilibrium Iteration ===" << std::endl;
    std::cout << "Starting with uniform mass, iterating transport..." << std::endl;

    // Start with uniform mass per unit area
    Eigen::VectorXd mass_per_area(n);
    for (size_t i = 0; i < n; ++i) {
        mass_per_area(i) = 1.0;  // 1 kg/m²
    }

    // Iterate: mass evolves as net_flux = deposition_flux - emission_flux
    // For equilibrium analysis, assume emission_flux ∝ mass_per_area (sublimation-limited)

    int max_iters = 1000;
    double dt = 0.01;  // Relative time step

    for (int iter = 0; iter < max_iters; ++iter) {
        // Emission proportional to current mass
        Eigen::VectorXd emission_iter(n);
        for (size_t i = 0; i < n; ++i) {
            emission_iter(i) = mass_per_area(i) * face_areas[i];
        }

        // Deposition
        Eigen::VectorXd deposition_iter = M * emission_iter;

        // Update mass
        for (size_t i = 0; i < n; ++i) {
            double dep_flux = deposition_iter(i) / face_areas[i];
            double em_flux = mass_per_area(i);  // emission flux = mass per area (normalized)
            mass_per_area(i) += dt * (dep_flux - em_flux);
            if (mass_per_area(i) < 0) mass_per_area(i) = 0;
        }

        // Print progress every 100 iterations
        if ((iter + 1) % 200 == 0) {
            std::vector<double> mass_pent, mass_regular;
            for (size_t i = 0; i < n; ++i) {
                if (is_pentagon_face[i]) {
                    mass_pent.push_back(mass_per_area(i));
                } else {
                    mass_regular.push_back(mass_per_area(i));
                }
            }
            double m_pent = mean(mass_pent);
            double m_reg = mean(mass_regular);
            std::cout << "  Iter " << std::setw(4) << (iter+1)
                      << ": pent=" << std::setprecision(6) << m_pent
                      << ", reg=" << m_reg
                      << ", ratio=" << (m_pent/m_reg) << std::endl;
        }
    }

    // Final analysis
    std::vector<double> final_mass_pent, final_mass_regular;
    for (size_t i = 0; i < n; ++i) {
        if (is_pentagon_face[i]) {
            final_mass_pent.push_back(mass_per_area(i));
        } else {
            final_mass_regular.push_back(mass_per_area(i));
        }
    }

    double final_pent = mean(final_mass_pent);
    double final_reg = mean(final_mass_regular);

    std::cout << "\nEquilibrium mass per area:" << std::endl;
    std::cout << "  Pentagon-adjacent: " << final_pent << std::endl;
    std::cout << "  Regular faces:     " << final_reg << std::endl;
    std::cout << "  Ratio:             " << (final_pent / final_reg) << std::endl;
    std::cout << "  Bias:              " << ((final_pent / final_reg - 1.0) * 100.0) << "%" << std::endl;

    // === Test 3: Analyze what M * area gives us ===
    std::cout << "\n=== Test 3: Row Sum Analysis (M * areas) ===" << std::endl;

    // If emission is uniform flux (1), then emission(i) = area_i
    // deposition(j) = sum_i M[j,i] * area_i
    // This should equal area_j for uniform equilibrium

    Eigen::VectorXd areas_vec(n);
    for (size_t i = 0; i < n; ++i) {
        areas_vec(i) = face_areas[i];
    }

    Eigen::VectorXd weighted_row_sum = M * areas_vec;

    // Compute ratio weighted_row_sum / area for each face
    std::vector<double> ratio_pent, ratio_regular;
    for (size_t i = 0; i < n; ++i) {
        double ratio = weighted_row_sum(i) / face_areas[i];
        if (is_pentagon_face[i]) {
            ratio_pent.push_back(ratio);
        } else {
            ratio_regular.push_back(ratio);
        }
    }

    double r_pent = mean(ratio_pent);
    double r_reg = mean(ratio_regular);

    std::cout << "Ratio (M*areas)[j] / area[j] (should be 1.0):" << std::endl;
    std::cout << "  Pentagon-adjacent: " << r_pent << std::endl;
    std::cout << "  Regular faces:     " << r_reg << std::endl;
    std::cout << "  Pent/Reg ratio:    " << (r_pent / r_reg) << std::endl;
    std::cout << "  Bias:              " << ((r_pent / r_reg - 1.0) * 100.0) << "%" << std::endl;

    // === Test 4: Compute "effective receiving area" for each face ===
    std::cout << "\n=== Test 4: Effective Receiving Area ===" << std::endl;

    // For face j, the effective receiving area is:
    // A_eff[j] = sum_i M[j,i] * A_i * (A_j / A_j) = sum_i M[j,i] * A_i
    // At equilibrium, we want A_eff[j] = A_j for all j

    // The issue is that M[j,i] contains A_j in numerator
    // M[j,i] = p[j,i] * A_j / sum_k(p[k,i] * A_k)
    //
    // So: sum_i M[j,i] * A_i = sum_i [p[j,i] * A_j / norm_i] * A_i
    //                        = A_j * sum_i [p[j,i] * A_i / norm_i]
    //
    // For this to equal A_j, we need sum_i [p[j,i] * A_i / norm_i] = 1

    // This is NOT guaranteed! Let's compute it.
    std::vector<double> sum_factor_pent, sum_factor_regular;
    for (size_t j = 0; j < n; ++j) {
        double sum_factor = weighted_row_sum(j) / face_areas[j];
        if (is_pentagon_face[j]) {
            sum_factor_pent.push_back(sum_factor);
        } else {
            sum_factor_regular.push_back(sum_factor);
        }
    }

    // This is the same as Test 3, but let's also look at the distribution
    auto compute_stats = [](const std::vector<double>& v) {
        double sum = std::accumulate(v.begin(), v.end(), 0.0);
        double mean = sum / v.size();
        double var = 0;
        for (double x : v) var += (x - mean) * (x - mean);
        double std = std::sqrt(var / v.size());
        double mn = *std::min_element(v.begin(), v.end());
        double mx = *std::max_element(v.begin(), v.end());
        return std::make_tuple(mean, std, mn, mx);
    };

    auto [sf_p_mean, sf_p_std, sf_p_min, sf_p_max] = compute_stats(sum_factor_pent);
    auto [sf_r_mean, sf_r_std, sf_r_min, sf_r_max] = compute_stats(sum_factor_regular);

    std::cout << "Sum factor = sum_i[p[j,i] * A_i / norm_i] (should be 1.0):" << std::endl;
    std::cout << "  Pentagon-adjacent: mean=" << sf_p_mean << ", std=" << sf_p_std
              << ", range=[" << sf_p_min << ", " << sf_p_max << "]" << std::endl;
    std::cout << "  Regular faces:     mean=" << sf_r_mean << ", std=" << sf_r_std
              << ", range=[" << sf_r_min << ", " << sf_r_max << "]" << std::endl;

    std::cout << "\n========================================" << std::endl;
    std::cout << " Analysis Complete" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
