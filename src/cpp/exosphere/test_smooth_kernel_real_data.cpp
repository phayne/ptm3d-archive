/*
 * Test smooth kernel on real simulation data
 * Compares shell-based vs smooth kernel transport matrices
 * Uses actual mesh and temperature data from europa_high_res simulation
 */

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <array>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <numeric>
#include "exosphere_physics.h"

using namespace ptm::exosphere;

// Simple PLY reader for binary little-endian format
bool readPLYMesh(const std::string& filename,
                 std::vector<std::array<double, 3>>& vertices,
                 std::vector<std::array<int, 3>>& faces,
                 std::vector<float>& face_latitudes,
                 std::vector<float>& face_longitudes) {

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

    std::cout << "PLY header: " << num_vertices << " vertices, " << num_faces << " faces" << std::endl;

    // Read vertices (x, y, z, nx, ny, nz as doubles)
    vertices.resize(num_vertices);
    for (int i = 0; i < num_vertices; ++i) {
        double coords[6];  // x, y, z, nx, ny, nz
        file.read(reinterpret_cast<char*>(coords), 6 * sizeof(double));
        vertices[i] = {coords[0], coords[1], coords[2]};
    }

    // Read faces (uchar count, 3 ints, lat, lon as floats)
    faces.resize(num_faces);
    face_latitudes.resize(num_faces);
    face_longitudes.resize(num_faces);

    for (int i = 0; i < num_faces; ++i) {
        unsigned char count;
        file.read(reinterpret_cast<char*>(&count), 1);

        int indices[3];
        file.read(reinterpret_cast<char*>(indices), 3 * sizeof(int));
        faces[i] = {indices[0], indices[1], indices[2]};

        file.read(reinterpret_cast<char*>(&face_latitudes[i]), sizeof(float));
        file.read(reinterpret_cast<char*>(&face_longitudes[i]), sizeof(float));
    }

    return true;
}

// Calculate face centers and areas from vertex data
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

        // Face center is centroid of triangle
        face_centers[i] = {
            (v0[0] + v1[0] + v2[0]) / 3.0,
            (v0[1] + v1[1] + v2[1]) / 3.0,
            (v0[2] + v1[2] + v2[2]) / 3.0
        };

        // Calculate area using cross product
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

// Generate synthetic temperature distribution (subsolar heating pattern)
void generateTemperatureDistribution(const std::vector<std::array<double, 3>>& face_centers,
                                     std::vector<double>& temperatures,
                                     double T_min = 50.0, double T_max = 140.0) {

    size_t n = face_centers.size();
    temperatures.resize(n);

    // Estimate radius from first face center
    double R = std::sqrt(face_centers[0][0]*face_centers[0][0] +
                         face_centers[0][1]*face_centers[0][1] +
                         face_centers[0][2]*face_centers[0][2]);

    for (size_t i = 0; i < n; ++i) {
        // Subsolar point at +x direction
        double x = face_centers[i][0];
        double cosine_factor = std::max(0.0, x / R);

        // T^4 ~ cos(solar_zenith_angle) for thermal equilibrium
        temperatures[i] = T_min + (T_max - T_min) * std::pow(cosine_factor, 0.25);
    }
}

// Analyze transport matrix for geometric artifacts
void analyzeMatrixForArtifacts(const Eigen::SparseMatrix<double>& matrix,
                               const std::vector<std::array<double, 3>>& face_centers,
                               const std::string& method_name) {

    size_t n = matrix.rows();
    double R = constants::EUROPA_RADIUS;

    // Calculate statistics per face
    std::vector<double> row_sums(n, 0.0);
    std::vector<double> col_sums(n, 0.0);
    std::vector<int> nnz_per_row(n, 0);

    // Analyze by latitude band
    const int num_bands = 9;  // 20-degree bands
    std::vector<double> band_avg_nnz(num_bands, 0.0);
    std::vector<int> band_count(num_bands, 0);

    for (int k = 0; k < matrix.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(matrix, k); it; ++it) {
            row_sums[it.row()] += it.value();
            col_sums[it.col()] += it.value();
            nnz_per_row[it.row()]++;
        }
    }

    // Get latitude for each face (from z coordinate)
    for (size_t i = 0; i < n; ++i) {
        double z = face_centers[i][2];
        double r = std::sqrt(face_centers[i][0]*face_centers[i][0] +
                            face_centers[i][1]*face_centers[i][1] +
                            face_centers[i][2]*face_centers[i][2]);
        double lat = std::asin(z / r) * 180.0 / M_PI;

        int band = static_cast<int>((lat + 90.0) / 20.0);
        band = std::max(0, std::min(num_bands - 1, band));

        band_avg_nnz[band] += nnz_per_row[i];
        band_count[band]++;
    }

    // Compute statistics
    double min_row_sum = *std::min_element(row_sums.begin(), row_sums.end());
    double max_row_sum = *std::max_element(row_sums.begin(), row_sums.end());
    double avg_row_sum = std::accumulate(row_sums.begin(), row_sums.end(), 0.0) / n;

    double min_col_sum = *std::min_element(col_sums.begin(), col_sums.end());
    double max_col_sum = *std::max_element(col_sums.begin(), col_sums.end());
    double avg_col_sum = std::accumulate(col_sums.begin(), col_sums.end(), 0.0) / n;

    int min_nnz = *std::min_element(nnz_per_row.begin(), nnz_per_row.end());
    int max_nnz = *std::max_element(nnz_per_row.begin(), nnz_per_row.end());
    double avg_nnz = std::accumulate(nnz_per_row.begin(), nnz_per_row.end(), 0.0) / n;

    // Calculate variance of nnz (indicator of artifacts)
    double var_nnz = 0.0;
    for (size_t i = 0; i < n; ++i) {
        var_nnz += (nnz_per_row[i] - avg_nnz) * (nnz_per_row[i] - avg_nnz);
    }
    var_nnz /= n;
    double std_nnz = std::sqrt(var_nnz);

    std::cout << "\n=== " << method_name << " Analysis ===" << std::endl;
    std::cout << "Matrix size: " << n << " x " << n << std::endl;
    std::cout << "Total non-zeros: " << matrix.nonZeros() << std::endl;
    std::cout << "Density: " << std::fixed << std::setprecision(4)
              << (100.0 * matrix.nonZeros() / (n * n)) << "%" << std::endl;

    std::cout << "\nRow sums: min=" << std::setprecision(6) << min_row_sum
              << ", max=" << max_row_sum << ", avg=" << avg_row_sum << std::endl;
    std::cout << "Column sums: min=" << min_col_sum
              << ", max=" << max_col_sum << ", avg=" << avg_col_sum << std::endl;

    std::cout << "\nNon-zeros per row: min=" << min_nnz
              << ", max=" << max_nnz << ", avg=" << std::setprecision(1) << avg_nnz << std::endl;
    std::cout << "Std dev of nnz/row: " << std::setprecision(2) << std_nnz
              << " (lower = more uniform = fewer artifacts)" << std::endl;

    std::cout << "\nNon-zeros by latitude band:" << std::endl;
    for (int b = 0; b < num_bands; ++b) {
        double lat_min = -90.0 + b * 20.0;
        double lat_max = lat_min + 20.0;
        double avg = band_count[b] > 0 ? band_avg_nnz[b] / band_count[b] : 0;
        std::cout << "  [" << std::setw(4) << lat_min << ", " << std::setw(4) << lat_max
                  << "): " << std::setw(6) << band_count[b] << " faces, avg nnz = "
                  << std::setprecision(1) << avg << std::endl;
    }
}

// Export matrix weights for visualization
void exportMatrixWeightsForVisualization(const Eigen::SparseMatrix<double>& matrix,
                                         const std::vector<std::array<double, 3>>& face_centers,
                                         const std::string& output_file,
                                         size_t source_face = 0) {

    std::ofstream file(output_file);
    if (!file.is_open()) {
        std::cerr << "Cannot open output file: " << output_file << std::endl;
        return;
    }

    file << "# Transport weights from face " << source_face << std::endl;
    file << "# face_id,lat,lon,weight" << std::endl;

    double R = constants::EUROPA_RADIUS;

    // Get weights from column (what this face receives from others)
    // or row (what this face sends to others)
    for (Eigen::SparseMatrix<double>::InnerIterator it(matrix, source_face); it; ++it) {
        size_t j = it.row();
        double w = it.value();

        double r = std::sqrt(face_centers[j][0]*face_centers[j][0] +
                            face_centers[j][1]*face_centers[j][1] +
                            face_centers[j][2]*face_centers[j][2]);
        double lat = std::asin(face_centers[j][2] / r) * 180.0 / M_PI;
        double lon = std::atan2(face_centers[j][1], face_centers[j][0]) * 180.0 / M_PI;

        file << j << "," << lat << "," << lon << "," << w << std::endl;
    }

    std::cout << "Exported weights to: " << output_file << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "========================================" << std::endl;
    std::cout << " Smooth Kernel Test on Real Data" << std::endl;
    std::cout << "========================================" << std::endl;

    // Default paths
    const char* dir_env = std::getenv("PTM_TEST_EUROPA_RUN");
    std::string output_dir = (argc > 1) ? argv[1] : (dir_env ? dir_env : "output/europa_high_res/");
    if (!output_dir.empty() && output_dir.back() != '/') output_dir += '/';
    std::string mesh_file = output_dir + "mesh.ply";

    if (argc > 1) {
        mesh_file = argv[1];
    }
    if (argc > 2) {
        output_dir = argv[2];
    }

    // Load mesh
    std::cout << "\nLoading mesh from: " << mesh_file << std::endl;

    std::vector<std::array<double, 3>> vertices;
    std::vector<std::array<int, 3>> faces;
    std::vector<float> face_latitudes;
    std::vector<float> face_longitudes;

    if (!readPLYMesh(mesh_file, vertices, faces, face_latitudes, face_longitudes)) {
        std::cerr << "Failed to load mesh!" << std::endl;
        return 1;
    }

    // Calculate face geometry
    std::vector<std::array<double, 3>> face_centers;
    std::vector<double> face_areas;
    calculateFaceGeometry(vertices, faces, face_centers, face_areas);

    std::cout << "Calculated geometry for " << face_centers.size() << " faces" << std::endl;

    // Generate temperature distribution
    std::vector<double> temperatures;
    generateTemperatureDistribution(face_centers, temperatures);

    double T_min = *std::min_element(temperatures.begin(), temperatures.end());
    double T_max = *std::max_element(temperatures.begin(), temperatures.end());
    std::cout << "Temperature range: [" << T_min << ", " << T_max << "] K" << std::endl;

    // Test different methods
    BallisticTransport transport;

    // Test 1: Shell-based with cutoff (current method)
    std::cout << "\n--- Test 1: Shell-based with xi=15 cutoff ---" << std::endl;
    {
        setenv("EXO_XI_CUTOFF", "15", 1);

        auto start = std::chrono::high_resolution_clock::now();
        auto matrix_shell = transport.buildTransportMatrixCached(face_centers, face_areas, temperatures);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "Build time: " << duration.count() << " ms" << std::endl;
        analyzeMatrixForArtifacts(matrix_shell, face_centers, "Shell-based (xi=15)");

        // Export for visualization
        exportMatrixWeightsForVisualization(matrix_shell, face_centers,
            output_dir + "shell_weights_face0.csv", 0);
        exportMatrixWeightsForVisualization(matrix_shell, face_centers,
            output_dir + "shell_weights_face1000.csv", 1000);

        unsetenv("EXO_XI_CUTOFF");
    }

    // Test 2: Smooth kernel (new method)
    std::cout << "\n--- Test 2: Smooth kernel with xi=15 cutoff ---" << std::endl;
    {
        setenv("EXO_XI_CUTOFF", "15", 1);

        BallisticTransport transport2;

        auto start = std::chrono::high_resolution_clock::now();
        auto matrix_smooth = transport2.buildTransportMatrixSmooth(face_centers, face_areas, temperatures);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "Build time: " << duration.count() << " ms" << std::endl;
        analyzeMatrixForArtifacts(matrix_smooth, face_centers, "Smooth kernel (xi=15)");

        // Export for visualization
        exportMatrixWeightsForVisualization(matrix_smooth, face_centers,
            output_dir + "smooth_weights_face0.csv", 0);
        exportMatrixWeightsForVisualization(matrix_smooth, face_centers,
            output_dir + "smooth_weights_face1000.csv", 1000);

        unsetenv("EXO_XI_CUTOFF");
    }

    // Test 3: Spatial index with smooth kernel
    std::cout << "\n--- Test 3: Spatial index + smooth kernel (xi=10) ---" << std::endl;
    {
        setenv("EXO_XI_CUTOFF", "10", 1);

        BallisticTransport transport3;

        auto start = std::chrono::high_resolution_clock::now();
        auto matrix_indexed = transport3.buildTransportMatrixSmooth(face_centers, face_areas, temperatures);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "Build time: " << duration.count() << " ms" << std::endl;
        analyzeMatrixForArtifacts(matrix_indexed, face_centers, "Smooth kernel (xi=10)");

        unsetenv("EXO_XI_CUTOFF");
    }

    // Test 4: Compare specific weight distributions
    std::cout << "\n--- Test 4: Weight distribution comparison ---" << std::endl;
    {
        setenv("EXO_XI_CUTOFF", "15", 1);

        BallisticTransport t_shell, t_smooth;
        auto m_shell = t_shell.buildTransportMatrixCached(face_centers, face_areas, temperatures);
        auto m_smooth = t_smooth.buildTransportMatrixSmooth(face_centers, face_areas, temperatures);

        // Pick a face near equator and one near pole
        size_t equator_face = 0;
        size_t pole_face = 0;
        double max_z = -1e10, min_abs_z = 1e10;

        for (size_t i = 0; i < face_centers.size(); ++i) {
            double z = std::abs(face_centers[i][2]);
            double r = std::sqrt(face_centers[i][0]*face_centers[i][0] +
                                face_centers[i][1]*face_centers[i][1] +
                                face_centers[i][2]*face_centers[i][2]);
            double sin_lat = z / r;

            if (sin_lat > max_z) {
                max_z = sin_lat;
                pole_face = i;
            }
            if (std::abs(face_centers[i][2]) < min_abs_z) {
                min_abs_z = std::abs(face_centers[i][2]);
                equator_face = i;
            }
        }

        std::cout << "\nEquator face " << equator_face << " comparison:" << std::endl;

        // Count non-zeros and sum weights for each matrix
        int nnz_shell = 0, nnz_smooth = 0;
        double sum_shell = 0.0, sum_smooth = 0.0;

        for (Eigen::SparseMatrix<double>::InnerIterator it(m_shell, equator_face); it; ++it) {
            nnz_shell++;
            sum_shell += it.value();
        }
        for (Eigen::SparseMatrix<double>::InnerIterator it(m_smooth, equator_face); it; ++it) {
            nnz_smooth++;
            sum_smooth += it.value();
        }

        std::cout << "  Shell: " << nnz_shell << " neighbors, sum = " << sum_shell << std::endl;
        std::cout << "  Smooth: " << nnz_smooth << " neighbors, sum = " << sum_smooth << std::endl;

        std::cout << "\nPole face " << pole_face << " comparison:" << std::endl;

        nnz_shell = nnz_smooth = 0;
        sum_shell = sum_smooth = 0.0;

        for (Eigen::SparseMatrix<double>::InnerIterator it(m_shell, pole_face); it; ++it) {
            nnz_shell++;
            sum_shell += it.value();
        }
        for (Eigen::SparseMatrix<double>::InnerIterator it(m_smooth, pole_face); it; ++it) {
            nnz_smooth++;
            sum_smooth += it.value();
        }

        std::cout << "  Shell: " << nnz_shell << " neighbors, sum = " << sum_shell << std::endl;
        std::cout << "  Smooth: " << nnz_smooth << " neighbors, sum = " << sum_smooth << std::endl;

        unsetenv("EXO_XI_CUTOFF");
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << " Test Complete" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
