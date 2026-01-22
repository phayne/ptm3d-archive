/*
 * Test program to analyze and fix pentagon vertex bias in transport matrix
 *
 * Pentagon vertices in an icosahedral mesh have:
 * - 5 adjacent faces instead of 6
 * - Different local geometry affecting transport patterns
 * - Systematically different face areas in their vicinity
 *
 * This test analyzes the bias and implements a correction.
 */

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <array>
#include <cmath>
#include <set>
#include <map>
#include <algorithm>
#include <numeric>
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

    // Parse header
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

    // Read vertices (x, y, z, nx, ny, nz as doubles)
    vertices.resize(num_vertices);
    for (int i = 0; i < num_vertices; ++i) {
        double coords[6];
        file.read(reinterpret_cast<char*>(coords), 6 * sizeof(double));
        vertices[i] = {coords[0], coords[1], coords[2]};
    }

    // Read faces (uchar count, 3 ints, lat, lon as floats)
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

// Calculate face centers and areas
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

        double e1[3] = {v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]};
        double e2[3] = {v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]};
        double cross[3] = {
            e1[1]*e2[2] - e1[2]*e2[1],
            e1[2]*e2[0] - e1[0]*e2[2],
            e1[0]*e2[1] - e1[1]*e2[0]
        };
        face_areas[i] = 0.5 * std::sqrt(cross[0]*cross[0] + cross[1]*cross[1] + cross[2]*cross[2]);
    }
}

// Identify pentagonal vertices and faces adjacent to them
void identifyPentagonalStructure(const std::vector<std::array<int, 3>>& faces,
                                  size_t num_vertices,
                                  std::set<int>& pentagon_vertices,
                                  std::vector<bool>& is_pentagon_face,
                                  std::vector<int>& face_pentagon_count) {

    // Count how many faces each vertex belongs to
    std::vector<int> vertex_face_count(num_vertices, 0);
    for (const auto& face : faces) {
        vertex_face_count[face[0]]++;
        vertex_face_count[face[1]]++;
        vertex_face_count[face[2]]++;
    }

    // Pentagon vertices have exactly 5 adjacent faces
    pentagon_vertices.clear();
    for (size_t v = 0; v < num_vertices; ++v) {
        if (vertex_face_count[v] == 5) {
            pentagon_vertices.insert(v);
        }
    }

    // Mark faces that touch pentagon vertices
    is_pentagon_face.resize(faces.size(), false);
    face_pentagon_count.resize(faces.size(), 0);

    for (size_t i = 0; i < faces.size(); ++i) {
        const auto& face = faces[i];
        int pcount = 0;
        for (int v = 0; v < 3; ++v) {
            if (pentagon_vertices.count(face[v]) > 0) {
                pcount++;
            }
        }
        face_pentagon_count[i] = pcount;
        is_pentagon_face[i] = (pcount > 0);
    }

    std::cout << "\nPentagonal structure analysis:" << std::endl;
    std::cout << "  Total vertices: " << num_vertices << std::endl;
    std::cout << "  Pentagon vertices: " << pentagon_vertices.size() << std::endl;

    // Count faces by pentagon proximity
    int faces_0_pent = 0, faces_1_pent = 0, faces_2_pent = 0, faces_3_pent = 0;
    for (size_t i = 0; i < faces.size(); ++i) {
        if (face_pentagon_count[i] == 0) faces_0_pent++;
        else if (face_pentagon_count[i] == 1) faces_1_pent++;
        else if (face_pentagon_count[i] == 2) faces_2_pent++;
        else faces_3_pent++;
    }

    std::cout << "  Faces with 0 pentagon vertices: " << faces_0_pent << std::endl;
    std::cout << "  Faces with 1 pentagon vertex: " << faces_1_pent << std::endl;
    std::cout << "  Faces with 2 pentagon vertices: " << faces_2_pent << std::endl;
    std::cout << "  Faces with 3 pentagon vertices: " << faces_3_pent << std::endl;
}

// Analyze transport matrix for pentagon bias
void analyzeTransportBias(const Eigen::SparseMatrix<double>& matrix,
                          const std::vector<bool>& is_pentagon_face,
                          const std::vector<int>& face_pentagon_count,
                          const std::vector<double>& face_areas,
                          const std::string& method_name) {

    size_t n = matrix.rows();

    // Collect statistics by face type
    std::vector<double> row_sums_pent, row_sums_regular;
    std::vector<double> col_sums_pent, col_sums_regular;
    std::vector<int> nnz_pent, nnz_regular;
    std::vector<double> areas_pent, areas_regular;

    // Compute row and column sums
    std::vector<double> row_sums(n, 0.0);
    std::vector<double> col_sums(n, 0.0);
    std::vector<int> nnz_per_row(n, 0);

    for (int k = 0; k < matrix.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(matrix, k); it; ++it) {
            row_sums[it.row()] += it.value();
            col_sums[it.col()] += it.value();
            nnz_per_row[it.row()]++;
        }
    }

    // Separate by pentagon adjacency
    for (size_t i = 0; i < n; ++i) {
        if (is_pentagon_face[i]) {
            row_sums_pent.push_back(row_sums[i]);
            col_sums_pent.push_back(col_sums[i]);
            nnz_pent.push_back(nnz_per_row[i]);
            areas_pent.push_back(face_areas[i]);
        } else {
            row_sums_regular.push_back(row_sums[i]);
            col_sums_regular.push_back(col_sums[i]);
            nnz_regular.push_back(nnz_per_row[i]);
            areas_regular.push_back(face_areas[i]);
        }
    }

    // Compute statistics
    auto stats = [](const std::vector<double>& v) -> std::tuple<double, double, double, double> {
        if (v.empty()) return {0, 0, 0, 0};
        double sum = std::accumulate(v.begin(), v.end(), 0.0);
        double mean = sum / v.size();
        double var = 0.0;
        for (double x : v) var += (x - mean) * (x - mean);
        var /= v.size();
        double mn = *std::min_element(v.begin(), v.end());
        double mx = *std::max_element(v.begin(), v.end());
        return {mean, std::sqrt(var), mn, mx};
    };

    auto stats_int = [](const std::vector<int>& v) -> std::tuple<double, double, int, int> {
        if (v.empty()) return {0, 0, 0, 0};
        double sum = std::accumulate(v.begin(), v.end(), 0.0);
        double mean = sum / v.size();
        double var = 0.0;
        for (int x : v) var += (x - mean) * (x - mean);
        var /= v.size();
        int mn = *std::min_element(v.begin(), v.end());
        int mx = *std::max_element(v.begin(), v.end());
        return {mean, std::sqrt(var), mn, mx};
    };

    std::cout << "\n=== " << method_name << " Pentagon Bias Analysis ===" << std::endl;

    // Face areas
    auto [area_p_mean, area_p_std, area_p_min, area_p_max] = stats(areas_pent);
    auto [area_r_mean, area_r_std, area_r_min, area_r_max] = stats(areas_regular);
    std::cout << "\nFace areas (m²):" << std::endl;
    std::cout << "  Pentagon-adj (" << areas_pent.size() << "): mean=" << std::scientific
              << area_p_mean << ", std=" << area_p_std << std::endl;
    std::cout << "  Regular (" << areas_regular.size() << "): mean=" << area_r_mean
              << ", std=" << area_r_std << std::endl;
    std::cout << "  Ratio (pent/reg): " << std::fixed << std::setprecision(4)
              << (area_p_mean / area_r_mean) << std::endl;

    // Row sums (outgoing transport)
    auto [row_p_mean, row_p_std, row_p_min, row_p_max] = stats(row_sums_pent);
    auto [row_r_mean, row_r_std, row_r_min, row_r_max] = stats(row_sums_regular);
    std::cout << "\nRow sums (outgoing transport):" << std::endl;
    std::cout << "  Pentagon-adj: mean=" << std::setprecision(6) << row_p_mean
              << ", std=" << row_p_std
              << ", range=[" << row_p_min << ", " << row_p_max << "]" << std::endl;
    std::cout << "  Regular: mean=" << row_r_mean
              << ", std=" << row_r_std
              << ", range=[" << row_r_min << ", " << row_r_max << "]" << std::endl;
    std::cout << "  Ratio (pent/reg): " << (row_p_mean / row_r_mean) << std::endl;

    // Column sums (incoming transport / mass conservation)
    auto [col_p_mean, col_p_std, col_p_min, col_p_max] = stats(col_sums_pent);
    auto [col_r_mean, col_r_std, col_r_min, col_r_max] = stats(col_sums_regular);
    std::cout << "\nColumn sums (incoming transport):" << std::endl;
    std::cout << "  Pentagon-adj: mean=" << col_p_mean
              << ", std=" << col_p_std << std::endl;
    std::cout << "  Regular: mean=" << col_r_mean
              << ", std=" << col_r_std << std::endl;
    std::cout << "  Ratio (pent/reg): " << (col_p_mean / col_r_mean) << std::endl;

    // Non-zeros per row (neighbor count)
    auto [nnz_p_mean, nnz_p_std, nnz_p_min, nnz_p_max] = stats_int(nnz_pent);
    auto [nnz_r_mean, nnz_r_std, nnz_r_min, nnz_r_max] = stats_int(nnz_regular);
    std::cout << "\nNon-zeros per row (neighbor count):" << std::endl;
    std::cout << "  Pentagon-adj: mean=" << std::setprecision(1) << nnz_p_mean
              << ", std=" << nnz_p_std
              << ", range=[" << nnz_p_min << ", " << nnz_p_max << "]" << std::endl;
    std::cout << "  Regular: mean=" << nnz_r_mean
              << ", std=" << nnz_r_std
              << ", range=[" << nnz_r_min << ", " << nnz_r_max << "]" << std::endl;
    std::cout << "  Ratio (pent/reg): " << (nnz_p_mean / nnz_r_mean) << std::endl;

    // Bias assessment
    double row_bias = std::abs(row_p_mean / row_r_mean - 1.0) * 100.0;
    double col_bias = std::abs(col_p_mean / col_r_mean - 1.0) * 100.0;
    double nnz_bias = std::abs(nnz_p_mean / nnz_r_mean - 1.0) * 100.0;

    std::cout << "\nBias Summary:" << std::endl;
    std::cout << "  Row sum bias: " << std::setprecision(2) << row_bias << "%" << std::endl;
    std::cout << "  Column sum bias: " << col_bias << "%" << std::endl;
    std::cout << "  Neighbor count bias: " << nnz_bias << "%" << std::endl;

    if (row_bias > 5.0 || col_bias > 5.0) {
        std::cout << "  ⚠ SIGNIFICANT PENTAGON BIAS DETECTED!" << std::endl;
    } else if (row_bias > 1.0 || col_bias > 1.0) {
        std::cout << "  ⚡ Minor pentagon bias present." << std::endl;
    } else {
        std::cout << "  ✓ Pentagon bias within acceptable limits." << std::endl;
    }
}

// Generate temperature distribution
void generateTemperatures(const std::vector<std::array<double, 3>>& face_centers,
                          std::vector<double>& temperatures) {
    size_t n = face_centers.size();
    temperatures.resize(n);

    double R = std::sqrt(face_centers[0][0]*face_centers[0][0] +
                        face_centers[0][1]*face_centers[0][1] +
                        face_centers[0][2]*face_centers[0][2]);

    for (size_t i = 0; i < n; ++i) {
        double x = face_centers[i][0];
        double cosine_factor = std::max(0.0, x / R);
        temperatures[i] = 50.0 + 90.0 * std::pow(cosine_factor, 0.25);
    }
}

int main(int argc, char* argv[]) {
    std::cout << "========================================" << std::endl;
    std::cout << " Pentagon Vertex Bias Analysis" << std::endl;
    std::cout << "========================================" << std::endl;

    std::string mesh_file = "/Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/europa/heat3d/ptm/build/output/europa_high_res/mesh.ply";
    if (argc > 1) {
        mesh_file = argv[1];
    }

    // Load mesh
    std::cout << "\nLoading mesh: " << mesh_file << std::endl;
    std::vector<std::array<double, 3>> vertices;
    std::vector<std::array<int, 3>> faces;

    if (!readPLYMesh(mesh_file, vertices, faces)) {
        return 1;
    }

    std::cout << "Loaded: " << vertices.size() << " vertices, " << faces.size() << " faces" << std::endl;

    // Calculate geometry
    std::vector<std::array<double, 3>> face_centers;
    std::vector<double> face_areas;
    calculateFaceGeometry(vertices, faces, face_centers, face_areas);

    // Identify pentagonal structure
    std::set<int> pentagon_vertices;
    std::vector<bool> is_pentagon_face;
    std::vector<int> face_pentagon_count;
    identifyPentagonalStructure(faces, vertices.size(), pentagon_vertices,
                                 is_pentagon_face, face_pentagon_count);

    // Generate temperatures
    std::vector<double> temperatures;
    generateTemperatures(face_centers, temperatures);

    // Test smooth kernel
    std::cout << "\n--- Test: Smooth Kernel (xi=15) ---" << std::endl;
    {
        setenv("EXO_XI_CUTOFF", "15", 1);
        BallisticTransport transport;
        auto matrix = transport.buildTransportMatrixSmooth(face_centers, face_areas, temperatures);
        analyzeTransportBias(matrix, is_pentagon_face, face_pentagon_count, face_areas, "Smooth Kernel");
        unsetenv("EXO_XI_CUTOFF");
    }

    // Test shell-based kernel
    std::cout << "\n--- Test: Shell-based Kernel (xi=15) ---" << std::endl;
    {
        setenv("EXO_XI_CUTOFF", "15", 1);
        BallisticTransport transport;
        auto matrix = transport.buildTransportMatrixCached(face_centers, face_areas, temperatures);
        analyzeTransportBias(matrix, is_pentagon_face, face_pentagon_count, face_areas, "Shell-based Kernel");
        unsetenv("EXO_XI_CUTOFF");
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << " Analysis Complete" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
