/*
 * PTM Exosphere Model - CLI wrapper
 *
 * This file provides a compact, robust CLI that delegates core work to
 * PTMExosphereIntegration (defined in ptm_exosphere_integration.*).
 * The previous file had many partially-applied edits; we replace the
 * top-level definitions with a minimal, maintainable surface.
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <vector>
#include <array>
#include <filesystem>
#include <chrono>
#include <cmath>
#include <numeric>
#include <memory>
#include <algorithm>
#include <cstdlib>
#include <limits>

#include "ptm_exosphere_integration.h"
#include "sputtering_model.h"
#include "exosphere_physics.h"
#include "ice_crystallization_model.h"
#include "composition_albedo.h"
#include "spherical_index.h"
#include "../mesh/mesh_utils.h"

// Physical constants used here
namespace exo_const = ptm::exosphere::constants;

// Minimal Mesh view used by this CLI (we mostly hand off to MeshUtils/PTM integration)
struct MeshView {
    std::vector<std::array<double,3>> vertices;
    std::vector<std::array<int,3>> faces;
    std::vector<double> face_lat;
    std::vector<double> face_lon;
};

// Modified Bessel function K0: use the single shared implementation in
// exosphere_physics.h. (A private duplicate lived here and carried the same
// x <= 2 defect, EXO-01; do not reintroduce a local copy.)
using ptm::exosphere::besselK0;

// Load mesh using MeshUtils; return nullptr on error.
static std::shared_ptr<ptm::Mesh> load_mesh_with_meshutils(const std::string& mesh_path)
{
    ptm::MeshUtils mu;
    try {
        if (mesh_path.empty()) {
            // Default to a small icosphere if no path is provided
            mu.setMeshResolution(4);
            mu.setupMesh();
            return mu.getMesh();
        }
        if (!std::filesystem::exists(mesh_path)) {
            std::cerr << "Mesh file not found: " << mesh_path << std::endl;
            return nullptr;
        }
        // Preferred robust loader (supports PTM PLY/VTK)
        if (mu.loadMeshFromFile(mesh_path)) {
            return mu.getMesh();
        }
        // Fallback: attempt import of triangulated mesh
        if (mu.importTriangulatedMesh(mesh_path, 1.0)) {
            return mu.getMesh();
        }
        std::cerr << "Unsupported mesh format or failed to import: " << mesh_path << std::endl;
        return nullptr;
    } catch (const std::exception& e) {
        std::cerr << "MeshUtils exception while loading mesh '" << mesh_path << "': " << e.what() << std::endl;
        return nullptr;
    }
}

// Ensure face centers (if present in MeshUtils) are exactly on nominal spherical radius
static void ensure_face_centers_on_sphere(std::shared_ptr<ptm::Mesh> m,
                                          double target_radius = exo_const::EUROPA_RADIUS)
{
    if (!m) return;
    if (m->face_centers.empty()) return;
    for (auto &fc : m->face_centers) {
        double r = std::sqrt(fc[0]*fc[0] + fc[1]*fc[1] + fc[2]*fc[2]);
        if (r <= 0.0) continue;
        fc[0] *= target_radius / r;
        fc[1] *= target_radius / r;
        fc[2] *= target_radius / r;
    }
}

// Progress bar utility
class ProgressBar {
private:
    int total_steps_;
    int current_step_;
    int bar_width_;
    std::chrono::steady_clock::time_point start_time_;
    
public:
    ProgressBar(int total_steps, int bar_width = 50) 
        : total_steps_(total_steps), current_step_(0), bar_width_(bar_width) {
        start_time_ = std::chrono::steady_clock::now();
    }
    
    void update(int step) {
        current_step_ = step;
        float progress = static_cast<float>(current_step_) / total_steps_;
        
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time_).count();
        int eta_seconds = (elapsed > 0 && current_step_ > 0) ?
                          static_cast<int>((elapsed / static_cast<float>(current_step_)) * (total_steps_ - current_step_)) : 0;
        
        std::cout << "\r[";
        int pos = static_cast<int>(bar_width_ * progress);
        for (int i = 0; i < bar_width_; ++i) {
            if (i < pos) std::cout << "=";
            else if (i == pos) std::cout << ">";
            else std::cout << " ";
        }
        std::cout << "] " << std::fixed << std::setprecision(1) << (progress * 100.0) << "% ";
        
        if (eta_seconds > 0) {
            int minutes = eta_seconds / 60;
            int seconds = eta_seconds % 60;
            std::cout << "ETA: " << minutes << "m" << std::setfill('0') << std::setw(2) << seconds << "s ";
        }
        std::cout << std::flush;
    }
    
    void finish() {
        update(total_steps_);
        std::cout << std::endl;
    }
};

#if 0

class ExosphereVTUExporter {
private:
    std::shared_ptr<Mesh> mesh_;
    std::string output_dir_;
    std::string vtu_dir_;
    
public:
    ExosphereVTUExporter(const std::shared_ptr<Mesh>& mesh, const std::string& output_dir)
        : mesh_(mesh), output_dir_(output_dir) {
        // Create vtu subdirectory
        vtu_dir_ = output_dir_ + "/vtu";
        std::filesystem::create_directories(vtu_dir_);
    }
    
    void exportTimestep(int timestep, 
                       const std::vector<double>& temperature,
                       const std::vector<double>& vapor_pressure,
                       const std::vector<double>& emission_flux,
                       const std::vector<double>& deposition_flux,
                       const std::vector<double>& net_flux,
                       const std::vector<double>& accumulated_mass,
                       double time_value) {
        
        std::stringstream filename;
        filename << vtu_dir_ << "/exosphere_" << std::setfill('0') << std::setw(4) << timestep << ".vtu";
        
        std::ofstream file(filename.str());
        if (!file.is_open()) {
            std::cerr << "Failed to open VTU file: " << filename.str() << std::endl;
            return;
        }
        
        file << "<?xml version=\"1.0\"?>\n";
        file << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\">\n";
        file << "  <UnstructuredGrid>\n";
        file << "    <Piece NumberOfPoints=\"" << mesh_->vertices.size() 
             << "\" NumberOfCells=\"" << mesh_->faces.size() << "\">\n";
        
        // Points
        file << "      <Points>\n";
        file << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";
        for (const auto& v : mesh_->vertices) {
            file << "          " << v[0] << " " << v[1] << " " << v[2] << "\n";
        }
        file << "        </DataArray>\n";
        file << "      </Points>\n";
        
        // Cell data
        file << "      <CellData>\n";
        writeDataArray(file, "Temperature", "Float64", temperature, "K");
        writeDataArray(file, "VaporPressure", "Float64", vapor_pressure, "Pa");
        writeDataArray(file, "EmissionFlux", "Float64", emission_flux, "kg/m2/s");
        writeDataArray(file, "DepositionFlux", "Float64", deposition_flux, "kg/m2/s");
        writeDataArray(file, "NetFlux", "Float64", net_flux, "kg/m2/s");
        writeDataArray(file, "AccumulatedMass", "Float64", accumulated_mass, "kg/m2");
        
        // Ice thickness
        std::vector<double> ice_thickness(accumulated_mass.size());
        const double ice_density = 917.0; // kg/m3
        for (size_t i = 0; i < accumulated_mass.size(); ++i) {
            ice_thickness[i] = accumulated_mass[i] / ice_density;
        }
        writeDataArray(file, "IceThickness", "Float64", ice_thickness, "m");
        
        file << "      </CellData>\n";
        
        // Cells
        file << "      <Cells>\n";
        
        file << "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
        for (const auto& f : mesh_->faces) {
            file << "          " << f[0] << " " << f[1] << " " << f[2] << "\n";
        }
        file << "        </DataArray>\n";
        
        file << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
        for (size_t i = 1; i <= mesh_->faces.size(); ++i) {
            file << "          " << (i * 3) << "\n";
        }
        file << "        </DataArray>\n";
        
        file << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
        for (size_t i = 0; i < mesh_->faces.size(); ++i) {
            file << "          5\n";  // VTK_TRIANGLE
        }
        file << "        </DataArray>\n";
        
        file << "      </Cells>\n";
        file << "    </Piece>\n";
        file << "  </UnstructuredGrid>\n";
        file << "</VTKFile>\n";
        
        file.close();
    }
    
    void createPVDFile(int num_timesteps, const std::vector<double>& time_values) {
        // Put PVD file in the main output directory, not in vtu subdirectory
        std::string pvd_filename = output_dir_ + "/exosphere.pvd";
        
        std::ofstream file(pvd_filename);
        if (!file.is_open()) {
            std::cerr << "Failed to create PVD file: " << pvd_filename << std::endl;
            return;
        }
        
        file << "<?xml version=\"1.0\"?>\n";
        file << "<VTKFile type=\"Collection\" version=\"0.1\">\n";
        file << "  <Collection>\n";
        
        for (int t = 0; t < num_timesteps; ++t) {
            double time_val = (t < time_values.size()) ? time_values[t] : t * 3600.0;
            // Reference VTU files in the vtu subdirectory
            file << "    <DataSet timestep=\"" << time_val 
                 << "\" file=\"vtu/exosphere_" << std::setfill('0') << std::setw(4) << t 
                 << ".vtu\"/>\n";
        }
        
        file << "  </Collection>\n";
        file << "</VTKFile>\n";
        
        file.close();
        std::cout << "Created PVD file: " << pvd_filename << std::endl;
    }
    
private:
    void writeDataArray(std::ofstream& file, const std::string& name, 
                       const std::string& type, const std::vector<double>& data,
                       const std::string& units = "") {
        file << "        <DataArray type=\"" << type << "\" Name=\"" << name << "\"";
        if (!units.empty()) {
            file << " units=\"" << units << "\"";
        }
        file << " format=\"ascii\">\n";
        
        for (const auto& val : data) {
            file << "          " << std::scientific << std::setprecision(8) << val << "\n";
        }
        file << "        </DataArray>\n";
    }
};

// Ensure face centers are consistently on sphere surface
void ensureFaceCentersOnSphere(std::vector<std::array<double, 3>>& face_centers,
                               double target_radius = constants::EUROPA_RADIUS) {
    for (auto& fc : face_centers) {
        double r = std::sqrt(fc[0]*fc[0] + fc[1]*fc[1] + fc[2]*fc[2]);
        if (r > 0.0 && std::abs(r - target_radius) > 1e-6) {
            double scale = target_radius / r;
            fc[0] *= scale;
            fc[1] *= scale;
            fc[2] *= scale;
        }
    }
}

// Main simulator class
class ExosphereSimulator {
private:
    std::shared_ptr<Mesh> mesh_;
    std::vector<double> face_areas_;
    std::vector<std::array<double, 3>> face_centers_;
    std::unique_ptr<ExosphereVTUExporter> vtu_exporter_;
    
    // Time-series data
    std::vector<std::vector<double>> temperatures_;
    std::vector<std::vector<double>> vapor_pressure_;
    std::vector<std::vector<double>> vapor_density_;
    std::vector<std::vector<double>> thermal_velocity_;
    std::vector<std::vector<double>> emission_flux_;
    std::vector<std::vector<double>> deposition_flux_;
    std::vector<std::vector<double>> net_flux_;
    std::vector<std::vector<double>> accumulated_mass_;
    std::vector<double> time_values_;

    std::string output_dir_;  // Store output directory
    
    // Transport matrix
    Eigen::SparseMatrix<double> transport_matrix_;
    bool transport_matrix_valid_ = false;

    // Spatial index for emission flux smoothing (removes pentagon mesh artifacts)
    ptm::exosphere::SphericalSpatialIndex spatial_index_;
    bool spatial_index_built_ = false;
    double mean_face_theta_ = 0.0;  // cached mean angular face size

    // Model parameters
    double dt_ = 3600.0;  // 1 hour timestep
    double initial_equatorial_thickness_ = 1e-4;  // 0.1 mm
    double initial_polar_thickness_ = 1e-3;       // 1 mm

    // NEW: per-rebuild jitter controls
    int rebuild_id_ = 0;
    double phase_jitter_amp_ = 0.05; // radians (~2.9°), tiny jitter to decorrelate in time

    // Verify transport matrix consistency
    void verifyTransportMatrixConsistency() {
        std::cout << "\nVerifying transport matrix consistency..." << std::endl;
        
        // Check that all face centers are at the same radius
        double radius_sum = 0.0;
        double radius_sq_sum = 0.0;
        
        for (const auto& fc : face_centers_) {
            double r = std::sqrt(fc[0]*fc[0] + fc[1]*fc[1] + fc[2]*fc[2]);
            radius_sum += r;
            radius_sq_sum += r * r;
        }
        
        double mean_radius = radius_sum / face_centers_.size();
        double variance = (radius_sq_sum / face_centers_.size()) - (mean_radius * mean_radius);
        double std_dev = std::sqrt(variance);
        
        std::cout << "  Face center radius statistics:" << std::endl;
        std::cout << "    Mean: " << mean_radius / 1000 << " km" << std::endl;
        std::cout << "    Std dev: " << std_dev << " m (" 
                  << (std_dev / mean_radius * 100) << "%)" << std::endl;
        
        // Sample some distances to check symmetry
        std::cout << "  Distance symmetry check (sampling 100 pairs):" << std::endl;
        double max_asymmetry = 0.0;
        for (int test = 0; test < 100; ++test) {
            size_t i = rand() % face_centers_.size();
            size_t j = rand() % face_centers_.size();
            if (i == j) continue;
            
            double d_ij = calculateGreatCircleDistance(face_centers_[i], face_centers_[j]);
            double d_ji = calculateGreatCircleDistance(face_centers_[j], face_centers_[i]);
            double asymmetry = std::abs(d_ij - d_ji);
            max_asymmetry = std::max(max_asymmetry, asymmetry);
        }
        std::cout << "    Maximum asymmetry: " << max_asymmetry << " m" << std::endl;
        
        if (max_asymmetry > 1e-6) {
            std::cerr << "WARNING: Distance calculations show asymmetry!" << std::endl;
        }
    }

    void exportMassBalance(const std::string& output_dir) {
        std::ofstream file(output_dir + "/mass_balance_summary.txt");
        
        file << "# Europa Exosphere Mass Balance Summary\n";
        file << "# Time_Step Time(s) Total_Emission(kg/s) Total_Deposition(kg/s) "
             << "Total_Escape(kg/s) Escape_Fraction Total_Ice_Mass(kg)\n";
        
        for (size_t t = 0; t < temperatures_.size(); ++t) {
            double total_emission = 0.0;
            double total_deposition = 0.0;
            double total_ice = 0.0;
            
            for (size_t i = 0; i < mesh_->faces.size(); ++i) {
                total_emission += emission_flux_[t][i] * face_areas_[i];
                total_deposition += deposition_flux_[t][i] * face_areas_[i];
                total_ice += accumulated_mass_[t][i] * face_areas_[i];
            }
            
            double total_escape = total_emission - total_deposition;
            double escape_fraction = (total_emission > 0) ? 
                (total_escape / total_emission) : 0.0;
            
            file << t << " " << time_values_[t] << " " 
                 << std::scientific << total_emission << " " 
                 << total_deposition << " " << total_escape << " "
                 << std::fixed << escape_fraction << " "
                 << std::scientific << total_ice << "\n";
        }
        
        file.close();
        std::cout << "Mass balance summary exported to: " << output_dir << "/mass_balance_summary.txt" << std::endl;
    }

    void diagnoseTransportPatterns(const std::string& output_dir, size_t timestep = 0) {
        std::cout << "\n=== TRANSPORT PATTERN DIAGNOSTICS (Timestep " << timestep << ") ===" << std::endl;
        
        size_t n_faces = mesh_->faces.size();
        
        // 1. Analyze mesh topology
        std::cout << "\n1. Mesh Topology Analysis:" << std::endl;
        
        // Count face connectivity
        std::vector<int> neighbor_count(n_faces, 0);
        std::vector<std::vector<size_t>> neighbors(n_faces);
        
        for (size_t i = 0; i < n_faces; ++i) {
            const auto& face_i = mesh_->faces[i];
            for (size_t j = 0; j < n_faces; ++j) {
                if (i == j) continue;
                const auto& face_j = mesh_->faces[j];
                
                // Check for shared vertices (edge neighbors)
                int shared = 0;
                for (int vi = 0; vi < 3; ++vi) {
                    for (int vj = 0; vj < 3; ++vj) {
                        if (face_i[vi] == face_j[vj]) shared++;
                    }
                }
                
                if (shared >= 2) {  // Share an edge
                    neighbor_count[i]++;
                    neighbors[i].push_back(j);
                }
            }
        }
        
        // Find pentagonal vertices (5 faces meet) - these are the 12 icosahedron vertices
        std::set<int> pentagonal_vertices;
        std::vector<int> vertex_face_count(mesh_->vertices.size(), 0);
        
        for (const auto& face : mesh_->faces) {
            vertex_face_count[face[0]]++;
            vertex_face_count[face[1]]++;
            vertex_face_count[face[2]]++;
        }
        
        for (size_t v = 0; v < mesh_->vertices.size(); ++v) {
            if (vertex_face_count[v] == 5) {
                pentagonal_vertices.insert(v);
            }
        }
        
        std::cout << "  Total faces: " << n_faces << std::endl;
        std::cout << "  Pentagonal vertices (5 faces meet): " << pentagonal_vertices.size() 
                  << " (should be 12 for icosphere)" << std::endl;
        
        // Identify faces at pentagonal vertices
        std::vector<bool> is_at_pentagon(n_faces, false);
        for (size_t i = 0; i < n_faces; ++i) {
            const auto& face = mesh_->faces[i];
            for (int v = 0; v < 3; ++v) {
                if (pentagonal_vertices.count(face[v]) > 0) {
                    is_at_pentagon[i] = true;
                    break;
                }
            }
        }
        
        int pentagon_faces = std::count(is_at_pentagon.begin(), is_at_pentagon.end(), true);
        std::cout << "  Faces touching pentagonal vertices: " << pentagon_faces << std::endl;
        
        // 2. Analyze transport matrix structure
        std::cout << "\n2. Transport Matrix Analysis:" << std::endl;
        
        // Check column sums (mass conservation)
        std::vector<double> column_sums(n_faces, 0.0);
        std::vector<double> escape_fractions(n_faces, 0.0);
        
        for (int k = 0; k < transport_matrix_.outerSize(); ++k) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(transport_matrix_, k); it; ++it) {
                column_sums[it.col()] += it.value();
            }
        }
        
        // Calculate escape fractions
        for (size_t i = 0; i < n_faces; ++i) {
            escape_fractions[i] = 1.0 - column_sums[i];
        }
        
        // Statistics
        double sum_min = *std::min_element(column_sums.begin(), column_sums.end());
        double sum_max = *std::max_element(column_sums.begin(), column_sums.end());
        double sum_avg = std::accumulate(column_sums.begin(), column_sums.end(), 0.0) / n_faces;
        
        std::cout << "  Column sums (should be ≤1 for conservation):" << std::endl;
        std::cout << "    Min: " << sum_min << std::endl;
        std::cout << "    Max: " << sum_max << std::endl;
        std::cout << "    Avg: " << sum_avg << std::endl;
        
        // Find problematic faces
        int problematic = 0;
        for (size_t i = 0; i < n_faces; ++i) {
            if (column_sums[i] > 1.01 || column_sums[i] < 0.0) {
                problematic++;
            }
        }
        
        if (problematic > 0) {
            std::cout << "  WARNING: " << problematic 
                      << " faces violate mass conservation!" << std::endl;
        }
        
        // 3. Analyze flux patterns
        std::cout << "\n3. Flux Pattern Analysis (Timestep " << timestep << "):" << std::endl;
        
        // Check for patterns in neighbor differences
        double max_neighbor_diff = 0.0;
        size_t max_diff_face = 0;
        double dep_avg = 0.0;

        if (timestep < deposition_flux_.size() && !deposition_flux_[timestep].empty()) {
            const auto& dep_flux = deposition_flux_[timestep];
            const auto& em_flux = emission_flux_[timestep];
            
            // Calculate statistics
            double dep_min = *std::min_element(dep_flux.begin(), dep_flux.end());
            double dep_max = *std::max_element(dep_flux.begin(), dep_flux.end());
            dep_avg = std::accumulate(dep_flux.begin(), dep_flux.end(), 0.0) / n_faces;
            
            std::cout << "  Deposition flux: min=" << dep_min 
                      << ", max=" << dep_max 
                      << ", avg=" << dep_avg << std::endl;
            
            // Compare pentagon vs regular faces
            double pentagon_dep = 0.0, regular_dep = 0.0;
            double pentagon_em = 0.0, regular_em = 0.0;
            int n_pentagon = 0, n_regular = 0;
            
            for (size_t i = 0; i < n_faces; ++i) {
                if (is_at_pentagon[i]) {
                    pentagon_dep += dep_flux[i];
                    pentagon_em += em_flux[i];
                    n_pentagon++;
                } else {
                    regular_dep += dep_flux[i];
                    regular_em += em_flux[i];
                    n_regular++;
                }
            }
            
            if (n_pentagon > 0) {
                pentagon_dep /= n_pentagon;
                pentagon_em /= n_pentagon;
            }
            if (n_regular > 0) {
                regular_dep /= n_regular;
                regular_em /= n_regular;
            }
            
            std::cout << "  Pentagon faces: avg deposition=" << pentagon_dep 
                      << ", avg emission=" << pentagon_em << std::endl;
            std::cout << "  Regular faces: avg deposition=" << regular_dep 
                      << ", avg emission=" << regular_em << std::endl;
            std::cout << "  Pentagon/Regular ratio: deposition=" 
                      << (regular_dep > 0 ? pentagon_dep/regular_dep : 0) 
                      << ", emission=" 
                      << (regular_em > 0 ? pentagon_em/regular_em : 0) << std::endl;
            
            for (size_t i = 0; i < n_faces; ++i) {
                for (size_t j : neighbors[i]) {
                    double diff = std::abs(dep_flux[i] - dep_flux[j]);
                    if (diff > max_neighbor_diff) {
                        max_neighbor_diff = diff;
                        max_diff_face = i;
                    }
                }
            }
            
            std::cout << "  Maximum neighbor flux difference: " << max_neighbor_diff 
                      << " at face " << max_diff_face << std::endl;
        }
        
        // 4. Export diagnostic data
        std::cout << "\n4. Exporting Diagnostic Data..." << std::endl;
        
        // Export neighbor connectivity
        std::ofstream neighbor_file(output_dir + "/mesh_neighbors.txt");
        neighbor_file << "# Face_id Num_neighbors Neighbor_ids Is_at_pentagon Lat Lon\n";
        for (size_t i = 0; i < n_faces; ++i) {
            neighbor_file << i << " " << neighbor_count[i] << " ";
            for (size_t j : neighbors[i]) {
                neighbor_file << j << " ";
            }
            neighbor_file << (is_at_pentagon[i] ? "1" : "0") << " ";
            if (!mesh_->latitudes.empty()) {
                neighbor_file << mesh_->latitudes[i] << " " << mesh_->longitudes[i];
            }
            neighbor_file << "\n";
        }
        neighbor_file.close();
        
        // Export transport matrix diagnostics
        std::ofstream matrix_file(output_dir + "/transport_diagnostics.txt");
        matrix_file << "# Face_id Column_sum Escape_fraction Temperature";
        if (timestep < deposition_flux_.size()) {
            matrix_file << " Deposition_flux Emission_flux Net_flux";
        }
        matrix_file << "\n";
        
        for (size_t i = 0; i < n_faces; ++i) {
            matrix_file << i << " " << column_sums[i] << " " << escape_fractions[i] << " ";
            if (timestep < temperatures_.size()) {
                matrix_file << temperatures_[timestep][i];
            } else {
                matrix_file << "0";
            }
            
            if (timestep < deposition_flux_.size()) {
                matrix_file << " " << deposition_flux_[timestep][i] 
                           << " " << emission_flux_[timestep][i]
                           << " " << net_flux_[timestep][i];
            }
            matrix_file << "\n";
        }
        matrix_file.close();
        
        // Export sparse matrix structure (sample)
        std::ofstream sparse_file(output_dir + "/transport_matrix_sparse.txt");
        sparse_file << "# row col value (first 1000 non-zeros only)\n";
        int count = 0;
        for (int k = 0; k < transport_matrix_.outerSize() && count < 1000; ++k) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(transport_matrix_, k); it && count < 1000; ++it) {
                if (it.value() > 1e-10) {
                    sparse_file << it.row() << " " << it.col() << " " << it.value() << "\n";
                    count++;
                }
            }
        }
        sparse_file.close();
        
        std::cout << "  Diagnostic files exported to: " << output_dir << std::endl;
        
        // 5. Pattern detection summary
        std::cout << "\n5. Pattern Detection Summary:" << std::endl;
        
        if (pentagon_faces > 0 && timestep < deposition_flux_.size()) {
            double pentagon_dep = 0.0, regular_dep = 0.0;
            int n_pentagon = 0, n_regular = 0;
            
            for (size_t i = 0; i < n_faces; ++i) {
                if (is_at_pentagon[i]) {
                    pentagon_dep += deposition_flux_[timestep][i];
                    n_pentagon++;
                } else {
                    regular_dep += deposition_flux_[timestep][i];
                    n_regular++;
                }
            }
            
            if (n_pentagon > 0) pentagon_dep /= n_pentagon;
            if (n_regular > 0) regular_dep /= n_regular;
            
            double ratio = (regular_dep > 0) ? pentagon_dep / regular_dep : 0;
            
            if (std::abs(ratio - 1.0) > 0.2) {
                std::cout << "  ⚠ MESH TOPOLOGY ARTIFACT DETECTED!" << std::endl;
                std::cout << "    Pentagon/regular flux ratio: " << ratio << std::endl;
                std::cout << "    The icosphere mesh topology is affecting transport." << std::endl;
            }
            
            if (max_neighbor_diff > dep_avg * 0.5) {
                std::cout << "  ⚠ SHARP DISCONTINUITIES DETECTED!" << std::endl;
                std::cout << "    Neighbor differences exceed 50% of average flux." << std::endl;
                std::cout << "    This suggests numerical issues in transport calculation." << std::endl;
            }
            
            if (problematic > n_faces * 0.01) {
                std::cout << "  ⚠ MASS CONSERVATION ISSUES!" << std::endl;
                std::cout << "    More than 1% of faces violate conservation." << std::endl;
            }
        }
    }
    
public:
    ExosphereSimulator(std::shared_ptr<Mesh> mesh) : mesh_(mesh) {
        calculateFaceProperties();
    }
    
    void calculateFaceProperties() {
        size_t n_faces = mesh_->faces.size();
        face_centers_.resize(n_faces);
        face_areas_.resize(n_faces);
        
        for (size_t i = 0; i < n_faces; ++i) {
            const auto& face = mesh_->faces[i];
            const auto& v0 = mesh_->vertices[face[0]];
            const auto& v1 = mesh_->vertices[face[1]];
            const auto& v2 = mesh_->vertices[face[2]];
            // Face center (as before)
            face_centers_[i][0] = (v0[0] + v1[0] + v2[0]) / 3.0;
            face_centers_[i][1] = (v0[1] + v1[1] + v2[1]) / 3.0;
            face_centers_[i][2] = (v0[2] + v1[2] + v2[2]) / 3.0;

            // Spherical triangle area (replaces planar cross-product area)
            face_areas_[i] = calculateSphericalTriangleArea(v0, v1, v2);
        }
        
        // CRITICAL: Ensure all face centers are on the sphere surface
        ensureFaceCentersOnSphere(face_centers_, constants::EUROPA_RADIUS);
        
        // Update the mesh's face_centers to match
        mesh_->face_centers = face_centers_;
        
        // Verify consistency
        double min_r = 1e10, max_r = 0.0;
        double min_area = 1e10, max_area = 0.0;
        for (size_t i = 0; i < n_faces; ++i) {
            double r = std::sqrt(face_centers_[i][0]*face_centers_[i][0] + 
                               face_centers_[i][1]*face_centers_[i][1] + 
                               face_centers_[i][2]*face_centers_[i][2]);
            min_r = std::min(min_r, r);
            max_r = std::max(max_r, r);
            min_area = std::min(min_area, face_areas_[i]);
            max_area = std::max(max_area, face_areas_[i]);
        }
        
        double variation = (max_r - min_r) / min_r * 100.0;
        
        // Calculate and report mesh statistics
        double total_area = std::accumulate(face_areas_.begin(), face_areas_.end(), 0.0);
        double theoretical_area = 4.0 * M_PI * constants::EUROPA_RADIUS * constants::EUROPA_RADIUS;
        
        std::cout << "\nMesh statistics:" << std::endl;
        std::cout << "  Number of faces: " << n_faces << std::endl;
        std::cout << "  Total surface area: " << total_area / 1e6 << " km²" << std::endl;
        std::cout << "  Theoretical sphere area: " << theoretical_area / 1e6 << " km²" << std::endl;
        std::cout << "  Area ratio: " << total_area / theoretical_area << std::endl;
        std::cout << "  Face areas: min=" << min_area << " m², max=" << max_area << " m²" << std::endl;
        std::cout << "  Face center radii: min=" << min_r / 1000 << " km, max=" 
                << max_r / 1000 << " km" << std::endl;
        std::cout << "  Radius variation: " << std::fixed << std::setprecision(8) 
                << variation << "%" << std::endl;
        
        if (variation > 1e-6) {
            std::cerr << "WARNING: Face centers show radius variation > 1e-6%!" << std::endl;
        }

        // Build spatial index for emission flux smoothing
        spatial_index_.build(face_centers_);
        spatial_index_built_ = true;

        // Cache mean angular face size (used for smoothing radius)
        const double R = constants::EUROPA_RADIUS;
        mean_face_theta_ = 0.0;
        for (size_t i = 0; i < n_faces; ++i) {
            mean_face_theta_ += std::sqrt(std::max(face_areas_[i], 0.0) / M_PI) / R;
        }
        mean_face_theta_ /= n_faces;
    }

    // Helper function to calculate spherical triangle area
    double calculateSphericalTriangleArea(const std::array<double, 3>& v0,
                                        const std::array<double, 3>& v1,
                                        const std::array<double, 3>& v2) {
        // Normalize vertices to unit sphere
        auto normalize = [](const std::array<double, 3>& v) -> std::array<double, 3> {
            double r = std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
            return {{v[0]/r, v[1]/r, v[2]/r}};
        };
        
        auto n0 = normalize(v0);
        auto n1 = normalize(v1);
        auto n2 = normalize(v2);
        
        // Calculate angles of the spherical triangle using dot and cross products
        auto angle = [](const std::array<double, 3>& a, const std::array<double, 3>& b, 
                    const std::array<double, 3>& c) -> double {
            // Vector from b to a
            std::array<double, 3> ba = {{a[0] - b[0], a[1] - b[1], a[2] - b[2]}};
            // Vector from b to c  
            std::array<double, 3> bc = {{c[0] - b[0], c[1] - b[1], c[2] - b[2]}};
            
            // For unit vectors on sphere, use the spherical angle formula
            double dot_product = a[0]*c[0] + a[1]*c[1] + a[2]*c[2];
            dot_product = std::max(-1.0, std::min(1.0, dot_product));
            
            // Cross product for determining angle
            std::array<double, 3> cross = {{
                a[1]*b[2] - a[2]*b[1],
                a[2]*b[0] - a[0]*b[2],
                a[0]*b[1] - a[1]*b[0]
            }};
            
            std::array<double, 3> cross2 = {{
                c[1]*b[2] - c[2]*b[1],
                c[2]*b[0] - c[0]*b[2],
                c[0]*b[1] - c[1]*b[0]
            }};
            
            double dot_cross = cross[0]*cross2[0] + cross[1]*cross2[1] + cross[2]*cross2[2];
            double norm_cross = std::sqrt((cross[0]*cross[0] + cross[1]*cross[1] + cross[2]*cross[2]) *
                                        (cross2[0]*cross2[0] + cross2[1]*cross2[1] + cross2[2]*cross2[2]));
            
            if (norm_cross > 0) {
                double cos_angle = dot_cross / norm_cross;
                cos_angle = std::max(-1.0, std::min(1.0, cos_angle));
                return std::acos(cos_angle);
            }
            return 0.0;
        };
        
        // Calculate the three angles of the spherical triangle
        double angle_A = angle(n1, n0, n2);
        double angle_B = angle(n2, n1, n0);
        double angle_C = angle(n0, n2, n1);
        
        // Spherical excess E = A + B + C - π
        double E = angle_A + angle_B + angle_C - M_PI;
        
        // Area = R² * E
        double area = constants::EUROPA_RADIUS * constants::EUROPA_RADIUS * E;
        
        // Fallback to planar approximation if spherical calculation fails
        if (area <= 0.0 || std::isnan(area) || std::isinf(area)) {
            // Calculate planar triangle area
            double dx1 = v1[0] - v0[0], dy1 = v1[1] - v0[1], dz1 = v1[2] - v0[2];
            double dx2 = v2[0] - v0[0], dy2 = v2[1] - v0[1], dz2 = v2[2] - v0[2];
            
            double cx = dy1 * dz2 - dz1 * dy2;
            double cy = dz1 * dx2 - dx1 * dz2;
            double cz = dx1 * dy2 - dy1 * dx2;
            
            area = 0.5 * std::sqrt(cx*cx + cy*cy + cz*cz);
        }
        
        return area;
    }
    
    // Calculate vapor pressure for Europa temperatures
    double calculateVaporPressure(double T) {
        if (T <= 0.0) return 0.0;

        // Marti & Mauersberger (1993) formula for low temperatures
        // Valid for Europa's range (50-150K)
        // NOTE: Formula uses base-10 logarithm: log10(P[Pa]) = -2663.5/T + 12.537
        double log10_p = -2663.5 / T + 12.537;
        return std::pow(10.0, log10_p);
    }
    
    // Calculate great circle distance
    double calculateGreatCircleDistance(const std::array<double, 3>& p1, 
                                   const std::array<double, 3>& p2) {
        // Calculate radii
        double r1 = std::sqrt(p1[0]*p1[0] + p1[1]*p1[1] + p1[2]*p1[2]);
        double r2 = std::sqrt(p2[0]*p2[0] + p2[1]*p2[1] + p2[2]*p2[2]);
        
        if (r1 <= 0.0 || r2 <= 0.0) return 0.0;
        
        // Normalize to unit vectors
        std::array<double, 3> n1 = {p1[0]/r1, p1[1]/r1, p1[2]/r1};
        std::array<double, 3> n2 = {p2[0]/r2, p2[1]/r2, p2[2]/r2};
        
        // Dot product gives cosine of angle
        double dot = n1[0]*n2[0] + n1[1]*n2[1] + n1[2]*n2[2];
        
        // Clamp to avoid numerical issues with acos
        dot = std::max(-1.0, std::min(1.0, dot));
        
        // For consistency, always use the nominal Europa radius
        // This ensures transport calculations are consistent even if
        // face centers have tiny variations
        return constants::EUROPA_RADIUS * std::acos(dot);
    }
    
    // --- ADD: small Gauss-Legendre helper (n=6) for self-cap integral ---
    double integrateSelfCap(const double T, const double face_area) const {
        if (T <= 0.0 || face_area <= 0.0) return 0.0;
        constexpr int N = 6;
        // Gauss-Legendre nodes (x) and weights (w) on [-1,1]
        static const double x[N] = {
            -0.9324695142031520, -0.6612093864662645, -0.2386191860831969,
             0.2386191860831969,  0.6612093864662645,  0.9324695142031520
        };
        static const double w[N] = {
            0.1713244923791704, 0.3607615730481386, 0.4679139345726910,
            0.4679139345726910, 0.3607615730481386, 0.1713244923791704
        };

        // Convert area to angular cap radius θ_cap via A = π R^2 θ_cap^2  (small-angle approx)
        double R = constants::EUROPA_RADIUS;
        double theta_cap = std::sqrt(std::max(face_area, 0.0) / (M_PI * R * R));
        theta_cap = std::min(theta_cap, 0.25); // safety clamp (very small faces)

        auto p_density = [&](double d)->double {
            if (d <= 0.0) return 0.0;
            double sigma2 = constants::BOLTZMANN_K * T / constants::WATER_MOLECULAR_MASS;
            double sigma4 = sigma2 * sigma2;
            double xi = constants::EUROPA_GRAVITY * d / (2.0 * sigma2);
            if (xi > 50.0) return 0.0;
            return (constants::EUROPA_GRAVITY * constants::EUROPA_GRAVITY * d / (4.0 * sigma4)) * besselK0(xi);
        };

        double a = 0.0, b = theta_cap;
        double integral = 0.0;
        for (int k = 0; k < N; ++k) {
            double t = 0.5 * ((b - a) * x[k] + (b + a)); // map to [0, θ_cap]
            double d = R * t;
            double pd = p_density(d);
            double jac = 0.5 * (b - a);
            // Surface element over spherical ring: 2π R^2 sin t dt
            integral += w[k] * pd * (2.0 * M_PI * R * R * std::sin(t)) * jac;
        }
        return integral; // raw probability mass contribution over the face cap (unnormalized)
    }

    void buildTransportMatrix(const std::vector<double>& temperatures) {
        size_t n_faces = mesh_->faces.size();

        // Check if smooth kernel is requested via environment variable
        bool use_smooth_kernel = false;
        if (const char* envp = std::getenv("EXO_USE_SMOOTH_KERNEL")) {
            use_smooth_kernel = (std::atoi(envp) != 0);
        }

        if (use_smooth_kernel) {
            // Use the optimized smooth kernel from exosphere_physics.cpp
            std::cout << "\nBuilding transport matrix with SMOOTH kernel (no shell binning)..." << std::endl;

            ptm::exosphere::BallisticTransport transport;
            transport_matrix_ = transport.buildTransportMatrixSmooth(
                face_centers_, face_areas_, temperatures);

            std::cout << "  Transport matrix built (smooth kernel):" << std::endl;
            std::cout << "    Non-zeros: " << transport_matrix_.nonZeros() << std::endl;
            std::cout << "  ✓ Smooth kernel - no shell binning artifacts\n";
            transport_matrix_valid_ = true;
            return;
        }

        // Original shell-based method
        std::vector<Eigen::Triplet<double>> triplets;
        triplets.reserve(n_faces * 64); // slightly smaller (no sort buffer)

        std::cout << "\nBuilding transport matrix with Bessel kernel (ring-density corrected, no-sort)..." << std::endl;

        const double R = constants::EUROPA_RADIUS;
        auto faceAngularRadius = [R](double A){ return std::sqrt(std::max(A,0.0)/M_PI)/R; };

        // Precompute unit centers & angular radii (once)
        std::vector<std::array<double,3>> unit_centers(n_faces);
        std::vector<double> theta_face(n_faces);
        for (size_t i = 0; i < n_faces; ++i) {
            const auto& fc = face_centers_[i];
            double invr = 1.0 / std::sqrt(fc[0]*fc[0]+fc[1]*fc[1]+fc[2]*fc[2]);
            unit_centers[i] = {fc[0]*invr, fc[1]*invr, fc[2]*invr};
            theta_face[i] = faceAngularRadius(face_areas_[i]);
        }

        // Allow runtime tuning of shell angular width factor (default 2.5)
        double shell_factor = 2.5;
        if (const char* envp = std::getenv("EXO_SHELL_WIDTH_FACTOR")) {
            double v = std::atof(envp);
            if (v > 0.1 && v < 10.0) shell_factor = v;
        }

        bool disable_ring_corr = false;
        if (const char* envp = std::getenv("EXO_DISABLE_RING_CORR")) {
            disable_ring_corr = (std::atoi(envp) != 0);
        }

        // Precompute mean theta (unchanged)
        double mean_theta = 0.0;
        for (double th : theta_face) mean_theta += th;
        mean_theta /= std::max<size_t>(1, theta_face.size());
        const double dtheta_base = std::max(1e-6, shell_factor * mean_theta);
        const int max_shells = static_cast<int>(std::ceil(M_PI / dtheta_base)) + 2;

        auto p_density = [&](double d, double T)->double {
            if (d <= 0.0 || T <= 0.0) return 0.0;
            double sigma2 = constants::BOLTZMANN_K * T / constants::WATER_MOLECULAR_MASS;
            double xi = constants::EUROPA_GRAVITY * d / (2.0 * sigma2);
            if (xi > 50.0) return 0.0;  // physical tail cutoff
            double sigma4 = sigma2 * sigma2;
            return (constants::EUROPA_GRAVITY * constants::EUROPA_GRAVITY * d / (4.0 * sigma4)) * besselK0(xi);
        };

        // Stats
        double min_col = 1.0, max_col = 0.0;
        int emitting = 0;
        double escape_accum = 0.0;

        // Thread-local buffers reused to avoid realloc
        #pragma omp parallel
        {
            std::vector<Eigen::Triplet<double>> local;
            local.reserve(8192);

            std::vector<double> shell_disc_area(max_shells);
            std::vector<double> shell_cont_area(max_shells);
            std::vector<std::vector<std::pair<int,double>>> shell_entries(max_shells);

            double local_min = 1.0, local_max = 0.0;
            int local_emit = 0;
            double local_escape = 0.0;

            #pragma omp for schedule(dynamic)
            for (size_t i = 0; i < n_faces; ++i) {
                double T_i = temperatures[i];
                if (T_i <= 0.0) {
                    local.emplace_back((int)i,(int)i,1.0);
                    continue;
                }
                local_emit++;

                // Escape fraction (unchanged logic)
                double vth = std::sqrt(2.0 * constants::BOLTZMANN_K * T_i / constants::WATER_MOLECULAR_MASS);
                double ratio = constants::EUROPA_ESCAPE_VELOCITY / vth;
                double escape_frac = 0.0;
                if (ratio < 5.0) {
                    escape_frac = std::exp(-ratio*ratio)*(1.0+ratio*ratio)/(2.0*ratio*ratio);
                }
                local_escape += escape_frac;

                // Self-cap integral
                double self_cap_mass = integrateSelfCap(T_i, face_areas_[i]);

                // Reset shell accumulators
                int used_shells = 0;
                std::fill(shell_disc_area.begin(), shell_disc_area.end(), 0.0);
                for (int s = 0; s < max_shells; ++s) {
                    shell_entries[s].clear();
                }

                // PASS 1: Compute θ_max and bin simultaneously (no sort)
                double theta_max_local = 0.0;

                const auto& uc_i = unit_centers[i];
                // Unrolled loop (process 4 targets) - keeps semantics identical
                size_t j = 0;
                for (; j + 3 < n_faces; j += 4) {
                    if (j == i || j+1 == i || j+2 == i || j+3 == i) break;
                    // Manual inline for each to keep precision (acos remains scalar)
                    for (int k = 0; k < 4; ++k) {
                        size_t jj = j + k;
                        if (jj == i) continue;
                        const auto& uc_j = unit_centers[jj];
                        double dotp = uc_i[0]*uc_j[0] + uc_i[1]*uc_j[1] + uc_i[2]*uc_j[2];
                        dotp = std::max(-1.0, std::min(1.0, dotp));
                        double theta = std::acos(dotp);
                        theta_max_local = std::max(theta_max_local, theta);
                        int s = (int)std::floor(theta / dtheta_base);
                        if (s >= max_shells) s = max_shells - 1;
                        shell_disc_area[s] += face_areas_[jj];
                        // store (face index, distance) later compute kernel
                        shell_entries[s].emplace_back((int)jj, R * theta);
                        if (s+1 > used_shells) used_shells = s+1;
                    }
                }
                // Remaining (or break due to self)
                for (; j < n_faces; ++j) {
                    if (j == i) continue;
                    const auto& uc_j = unit_centers[j];
                    double dotp = uc_i[0]*uc_j[0] + uc_i[1]*uc_j[1] + uc_i[2]*uc_j[2];
                    dotp = std::max(-1.0, std::min(1.0, dotp));
                    double theta = std::acos(dotp);
                    theta_max_local = std::max(theta_max_local, theta);
                    int s = (int)std::floor(theta / dtheta_base);
                    if (s >= max_shells) s = max_shells - 1;
                    shell_disc_area[s] += face_areas_[j];
                    shell_entries[s].emplace_back((int)j, R * theta);
                    if (s+1 > used_shells) used_shells = s+1;
                }

                // PASS 2: Compute continuous shell areas only up to used_shells
                for (int s = 0; s < used_shells; ++s) {
                    double theta_inner = s * dtheta_base;
                    double theta_outer = (s+1) * dtheta_base;
                    if (theta_outer > M_PI) theta_outer = M_PI;
                    shell_cont_area[s] =
                        2.0 * M_PI * R * R * (std::cos(theta_inner) - std::cos(theta_outer));
                }

                // PASS 3: Accumulate off-diagonal raw weights
                double sum_off_raw = 0.0;
                std::vector<std::pair<int,double>> buffer;
                buffer.reserve(used_shells * 8);

                for (int s = 0; s < used_shells; ++s) {
                    if (shell_entries[s].empty()) continue;
                    double correction = 1.0;
                    if (!disable_ring_corr &&
                        shell_disc_area[s] > 0.0 &&
                        shell_cont_area[s] > 0.0) {
                        correction = shell_cont_area[s] / shell_disc_area[s];
                    }
                    for (auto& rec : shell_entries[s]) {
                        int jidx = rec.first;
                        double d = rec.second;
                        double pd = p_density(d, T_i);
                        if (pd <= 0.0) continue;
                        double w_raw = pd * face_areas_[jidx] * correction;
                        if (w_raw > 0.0 && std::isfinite(w_raw)) {
                            buffer.emplace_back(jidx, w_raw);
                            sum_off_raw += w_raw;
                        }
                    }
                }

                double total_raw = self_cap_mass + sum_off_raw;
                double target = 1.0 - escape_frac;

                if (total_raw <= 0.0) {
                    local.emplace_back((int)i,(int)i,target);
                    local_min = std::min(local_min, target);
                    local_max = std::max(local_max, target);
                    continue;
                }
                double scale = target / total_raw;

                // Self
                double self_weight = self_cap_mass * scale;
                local.emplace_back((int)i,(int)i,self_weight);
                double col_sum = self_weight;

                for (auto& pr : buffer) {
                    double v = pr.second * scale;
                    if (v > 0.0) {
                        local.emplace_back(pr.first,(int)i,v);
                        col_sum += v;
                    }
                }

                local_min = std::min(local_min, col_sum);
                local_max = std::max(local_max, col_sum);
            }

            #pragma omp critical
            {
                triplets.insert(triplets.end(), local.begin(), local.end());
                min_col = std::min(min_col, local_min);
                max_col = std::max(max_col, local_max);
                emitting += local_emit;
                escape_accum += local_escape;
            }
        }

        transport_matrix_ = Eigen::SparseMatrix<double>(n_faces, n_faces);
        transport_matrix_.setFromTriplets(triplets.begin(), triplets.end());
        transport_matrix_.makeCompressed();

        std::cout << "  Transport matrix built:" << std::endl;
        std::cout << "    Emitting faces: " << emitting << "/" << n_faces << std::endl;
        std::cout << "    Avg escape fraction: " << (emitting? escape_accum/emitting:0.0) << std::endl;
        std::cout << "    Non-zeros: " << transport_matrix_.nonZeros() << std::endl;
        std::cout << "    Column sum range: [" << std::fixed << std::setprecision(6)
                  << min_col << ", " << max_col << "]" << std::endl;
        if (std::fabs(min_col-1.0)>1e-6 || std::fabs(max_col-1.0)>1e-6)
            std::cerr << "  WARNING: column normalization drift\n";
        else
            std::cout << "  ✓ Mass conservation (ring-density corrected, no-sort)\n";
        transport_matrix_valid_ = true;
    }

    // Calculate escape fraction using Jeans escape formula
    double calculateEscapeFraction(double temperature_K) {
        if (temperature_K <= 0.0) return 0.0;
        
        // Jeans escape parameter: λ = (m * v_esc²) / (2kT)
        double lambda = (constants::WATER_MOLECULAR_MASS * constants::EUROPA_ESCAPE_VELOCITY * 
                        constants::EUROPA_ESCAPE_VELOCITY) / 
                    (2.0 * constants::BOLTZMANN_K * temperature_K);
        
        // Jeans escape rate
        if (lambda > 10.0) {
            // Cold regime - use asymptotic formula
            return std::exp(-lambda) * std::sqrt(2.0 * lambda / M_PI);
        } else {
            // Warm regime - use full formula
            double f_esc = std::exp(-lambda) * (1.0 + lambda) / std::sqrt(M_PI * lambda);
            // Cap at reasonable maximum
            return std::min(f_esc, 0.5);
        }
    }
    
    // Set initial ice distribution
    void setIceParameters(double equatorial_mm, double polar_mm) {
        initial_equatorial_thickness_ = equatorial_mm * 1e-3;  // Convert mm to m
        initial_polar_thickness_ = polar_mm * 1e-3;
        
        size_t n_faces = mesh_->faces.size();
        
        // Initialize first timestep's accumulated mass
        if (accumulated_mass_.empty() || accumulated_mass_[0].empty()) {
            accumulated_mass_.resize(1);
            accumulated_mass_[0].resize(n_faces);
        }
        
        std::cout << "\n========================================" << std::endl;
        std::cout << "Latitudinal Ice Distribution Initialized" << std::endl;
        std::cout << "========================================\n" << std::endl;
        
        std::cout << "Input Parameters:" << std::endl;
        std::cout << "  Equatorial thickness: " << equatorial_mm * 1000.0 << " μm" << std::endl;
        std::cout << "  Polar thickness: " << polar_mm * 1000.0 << " μm" << std::endl;
        
        double total_mass = 0.0;
        double min_thickness = 1e10, max_thickness = 0.0;
        
        for (size_t i = 0; i < n_faces; ++i) {
            // Get latitude
            double lat = mesh_->latitudes.empty() ? 0.0 : mesh_->latitudes[i];
            double abs_lat = std::abs(lat);
            
            // Cosine-squared distribution
            double thickness = initial_equatorial_thickness_ + 
                              (initial_polar_thickness_ - initial_equatorial_thickness_) * 
                              std::pow(std::sin(abs_lat * M_PI / 180.0), 2);
            
            double mass = thickness * constants::ICE_DENSITY;
            accumulated_mass_[0][i] = mass;
            
            min_thickness = std::min(min_thickness, thickness);
            max_thickness = std::max(max_thickness, thickness);
            total_mass += mass * face_areas_[i];
        }
        
        std::cout << "\nActual Distribution:" << std::endl;
        std::cout << "  Minimum thickness: " << min_thickness * 1e6 << " μm" << std::endl;
        std::cout << "  Maximum thickness: " << max_thickness * 1e6 << " μm" << std::endl;
        std::cout << "  Ratio (max/min): " << std::fixed << std::setprecision(2) 
                  << max_thickness / min_thickness << std::endl;
        
        std::cout << "\nGlobal Statistics:" << std::endl;
        std::cout << "  Total ice mass: " << std::scientific << std::setprecision(4) 
                  << total_mass << " kg" << std::endl;
        std::cout << "========================================\n" << std::endl;
    }
    
    // Read temperature files and initialize data structures
    bool loadTemperatures(const std::string& ptm_dir, int timesteps) {
        size_t n_faces = mesh_->faces.size();
        
        // Resize storage for all timesteps
        temperatures_.resize(timesteps);
        vapor_pressure_.resize(timesteps);
        vapor_density_.resize(timesteps);
        thermal_velocity_.resize(timesteps);
        emission_flux_.resize(timesteps);
        deposition_flux_.resize(timesteps);
        net_flux_.resize(timesteps);
        accumulated_mass_.resize(timesteps);
        time_values_.resize(timesteps);
        
        // Only initialize timestep 0, not all future timesteps
        // Future timesteps will be calculated during simulation
        for (int t = 0; t < timesteps; ++t) {
            temperatures_[t].resize(n_faces);
            vapor_pressure_[t].resize(n_faces);
            vapor_density_[t].resize(n_faces);
            thermal_velocity_[t].resize(n_faces);
            emission_flux_[t].resize(n_faces);
            deposition_flux_[t].resize(n_faces);
            net_flux_[t].resize(n_faces);
            accumulated_mass_[t].resize(n_faces, 0.0);  // Initialize to zero
            
            // Build filename
            std::stringstream filename;
            filename << ptm_dir << "/temperature/temperature_" 
                    << std::setfill('0') << std::setw(4) << t << ".bin";
            
            std::ifstream file(filename.str(), std::ios::binary);
            if (!file.is_open()) {
                std::cerr << "Failed to open temperature file: " << filename.str() << std::endl;
                return false;
            }
            
            // Read header
            uint32_t count;
            file.read(reinterpret_cast<char*>(&count), sizeof(uint32_t));
            
            if (count != n_faces) {
                std::cerr << "Face count mismatch: expected " << n_faces 
                        << ", got " << count << std::endl;
                return false;
            }
            
            // Read as float (PTM3D format)
            std::vector<float> float_data(count);
            file.read(reinterpret_cast<char*>(float_data.data()), count * sizeof(float));
            
            // Convert to double and validate
            for (size_t i = 0; i < count; ++i) {
                double T = static_cast<double>(float_data[i]);
                
                // Validate temperature range for Europa
                if (T < 50.0 || T > 200.0) {
                    T = std::max(50.0, std::min(200.0, T));  // Clamp to reasonable range
                }
                
                temperatures_[t][i] = T;
            }
            
            file.close();
            time_values_[t] = t * dt_;
        }
        
        // Set initial ice distribution ONLY for timestep 0
        // This was already done in setIceParameters
        
        // Report temperature statistics
        double min_T = 1e10, max_T = 0.0, avg_T = 0.0;
        for (const auto& T : temperatures_[0]) {
            min_T = std::min(min_T, T);
            max_T = std::max(max_T, T);
            avg_T += T;
        }
        avg_T /= n_faces;
        
        std::cout << "Temperature statistics (first timestep):" << std::endl;
        std::cout << "  Min: " << min_T << " K" << std::endl;
        std::cout << "  Max: " << max_T << " K" << std::endl;
        std::cout << "  Avg: " << avg_T << " K" << std::endl;
        
        // Build transport matrix using first timestep temperatures
        // NEW: increment rebuild id before building to apply initial jitter
        rebuild_id_++;
        buildTransportMatrix(temperatures_[0]);
        
        return true;
    }

    // Run the simulation
    void runSimulation(const std::string& output_dir) {
        output_dir_ = output_dir;  // Store for later use
        size_t n_faces = mesh_->faces.size();
        size_t n_timesteps = temperatures_.size();
        
        std::cout << "\n=== Running Exosphere Simulation ===" << std::endl;
        
        ProgressBar progress(n_timesteps);
        
        // Track running ice mass (starts with initial distribution from setIceParameters)
        std::vector<double> current_ice_mass = accumulated_mass_[0];
        
        // Track temperature changes for transport matrix updates
        double last_max_T = 0.0;
        double last_min_T = 1000.0;
        double last_avg_T = 0.0;
        
        for (size_t t = 0; t < n_timesteps; ++t) {
            progress.update(t);
            
            // Always rebuild (optionally skip if env disables)
            bool skip_rebuild = false;
            if (const char* envp = std::getenv("EXO_NO_REBUILD")) {
                skip_rebuild = (std::atoi(envp) != 0);
            }
            if (!skip_rebuild) {
                rebuild_id_++;
                buildTransportMatrix(temperatures_[t]); // per-step matrix
            }
            
            // PHASE 1: Calculate emission for each face and store in arrays
            // Use parallel for WITHOUT reduction for array writes
            #pragma omp parallel for
            for (size_t i = 0; i < n_faces; ++i) {
                double T = temperatures_[t][i];
                
                // Check if there's ice available using the running ice mass
                constexpr double MIN_MASS = constants::MIN_ICE_THICKNESS * constants::ICE_DENSITY;
                bool has_ice = (current_ice_mass[i] > MIN_MASS);
                
                if (T > 0.0 && has_ice) {
                    // Calculate vapor properties
                    vapor_pressure_[t][i] = calculateVaporPressure(T);
                    vapor_density_[t][i] = vapor_pressure_[t][i] * 0.01801528 / (8.314462618 * T);
                    thermal_velocity_[t][i] = std::sqrt(8.0 * constants::BOLTZMANN_K * T / 
                                                    (M_PI * constants::WATER_MOLECULAR_MASS));
                    
                    // Emission flux (kg/m²/s)
                    emission_flux_[t][i] = vapor_density_[t][i] * thermal_velocity_[t][i] / 4.0;
                    
                    // Limit emission to available ice
                    double max_emission = current_ice_mass[i] / dt_;
                    if (emission_flux_[t][i] > max_emission) {
                        emission_flux_[t][i] = max_emission;
                    }
                } else {
                    // No emission if no temperature or no ice
                    vapor_pressure_[t][i] = 0.0;
                                       vapor_density_[t][i] = 0.0;
                    thermal_velocity_[t][i] = 0.0;
                    emission_flux_[t][i] = 0.0;
                }
            }
            
            // PHASE 2: Calculate totals from stored arrays
            // Separate loop with reduction for scalar sums
            double total_emission = 0.0;
            int emitting_faces = 0;
            
            #pragma omp parallel for reduction(+:total_emission,emitting_faces)
            for (size_t i = 0; i < n_faces; ++i) {
                if (emission_flux_[t][i] > 0.0) {
                    total_emission += emission_flux_[t][i] * face_areas_[i];
                    emitting_faces++;
                }
            }
            
            // PHASE 3: Smooth emission flux and apply transport matrix
            // Area-weighted spatial smoothing removes mesh-topology artifacts
            // (pentagon vs hexagon faces) while preserving physical gradients.
            // Same approach as deposition smoothing in computeTransportProduct().
            Eigen::VectorXd emission_vec(n_faces);
            {
                double smooth_radius = 3.0 * mean_face_theta_;
                float cos_smooth = static_cast<float>(std::cos(smooth_radius));
                std::vector<double> em_smooth(n_faces);

                #pragma omp parallel
                {
                    std::vector<std::pair<size_t, float>> neighbors;
                    neighbors.reserve(64);

                    #pragma omp for schedule(dynamic)
                    for (size_t i = 0; i < n_faces; ++i) {
                        spatial_index_.queryNeighborsByCosTheta(
                            i, cos_smooth, neighbors);

                        double wsum = face_areas_[i];
                        double em_sum = emission_flux_[t][i] * face_areas_[i];
                        for (const auto& [k, cos_theta] : neighbors) {
                            em_sum += emission_flux_[t][k] * face_areas_[k];
                            wsum += face_areas_[k];
                        }
                        em_smooth[i] = em_sum / wsum;
                    }
                }

                for (size_t i = 0; i < n_faces; ++i) {
                    emission_vec(i) = em_smooth[i] * face_areas_[i] * dt_;
                }
            }

            Eigen::VectorXd deposition_vec = transport_matrix_ * emission_vec;
            
            // PHASE 4: Update ice mass and calculate fluxes
            // Use parallel for WITHOUT reduction for array writes
            #pragma omp parallel for
            for (size_t i = 0; i < n_faces; ++i) {
                // Calculate deposition flux
                deposition_flux_[t][i] = deposition_vec(i) / (face_areas_[i] * dt_);
                
                // Calculate net flux
                net_flux_[t][i] = deposition_flux_[t][i] - emission_flux_[t][i];
                
                // Update running ice mass
                current_ice_mass[i] += net_flux_[t][i] * dt_;
                current_ice_mass[i] = std::max(0.0, current_ice_mass[i]);
                
                // Store in accumulated_mass array for this timestep
                accumulated_mass_[t][i] = current_ice_mass[i];
            }
            
            // PHASE 5: Calculate totals for reporting
            double total_deposition = 0.0;
            #pragma omp parallel for reduction(+:total_deposition)
            for (size_t i = 0; i < n_faces; ++i) {
                total_deposition += deposition_flux_[t][i] * face_areas_[i];
            }
            
            // Calculate escape
            double total_escape = total_emission - total_deposition;
            double escape_fraction = (total_emission > 0) ? (total_escape / total_emission) : 0.0;
            
            // Print progress every 10 timesteps
            if (t % 10 == 0 || t == n_timesteps - 1) {
                // Count faces with remaining ice
                int faces_with_ice = 0;
                double total_ice_mass = 0.0;
                
                #pragma omp parallel for reduction(+:faces_with_ice,total_ice_mass)
                for (size_t i = 0; i < n_faces; ++i) {
                    if (current_ice_mass[i] > constants::MIN_ICE_THICKNESS * constants::ICE_DENSITY) {
                        faces_with_ice++;
                    }
                    total_ice_mass += current_ice_mass[i] * face_areas_[i];
                }
                
                std::cout << "\nTimestep " << t << ": "
                        << "T_max=" << std::fixed << std::setprecision(1) 
                        << *std::max_element(temperatures_[t].begin(), temperatures_[t].end()) << "K, "
                        << "Emission=" << std::scientific << std::setprecision(3) 
                        << total_emission << " kg/s, "
                        << "Deposition=" << total_deposition << " kg/s, "
                        << "Escape=" << std::fixed << std::setprecision(2) 
                        << (escape_fraction * 100.0) << "%, "
                        << "Emitting faces=" << emitting_faces << "/" << n_faces 
                        << ", Faces with ice=" << faces_with_ice << "/" << n_faces
                        << ", Total ice=" << std::scientific << std::setprecision(3) 
                        << total_ice_mass << " kg" << std::endl;
                       }
        }
        
        progress.finish();
        std::cout << "\nSimulation complete!" << std::endl;
        
        // Verify data preservation for VTU export

        std::cout << "\nVerifying data preservation for VTU export:" << std::endl;
        for (size_t t = 0; t < std::min(size_t(5), n_timesteps); ++t) {
            double max_emission = *std::max_element(emission_flux_[t].begin(), emission_flux_[t].end());
            double max_deposition = *std::max_element(deposition_flux_[t].begin(), deposition_flux_[t].end());
            double avg_emission = 0.0, avg_deposition = 0.0;
            int nonzero_emission = 0, nonzero_deposition = 0;
            
            for (size_t i = 0; i < n_faces; ++i) {
                if (emission_flux_[t][i] > 0.0) {
                    avg_emission += emission_flux_[t][i];
                    nonzero_emission++;
                }
                if (deposition_flux_[t][i] > 0.0) {
                    avg_deposition += deposition_flux_[t][i];
                    nonzero_deposition++;
                }
            }
            
            if (nonzero_emission > 0) avg_emission /= nonzero_emission;
            if (nonzero_deposition > 0) avg_deposition /= nonzero_deposition;
            
            std::cout << "  Timestep " << t << ":" << std::endl;
            std::cout << "    Emission: max=" << std::scientific << std::setprecision(3) << max_emission 
                    << " kg/(m²·s), avg=" << avg_emission 
                    << " kg/(m²·s), nonzero=" << nonzero_emission << " faces" << std::endl;
            std::cout << "    Deposition: max=" << max_deposition 
                    << " kg/(m²·s), avg=" << avg_deposition 
                    << " kg/(m²·s), nonzero=" << nonzero_deposition << " faces" << std::endl;
        }

        // Run diagnostics on first and last timesteps
        diagnoseTransportPatterns(output_dir, 0);  // First timestep
        if (n_timesteps > 1) {
            diagnoseTransportPatterns(output_dir, n_timesteps - 1);  // Last timestep
        }
    }
    
    // Export results
    void exportResults(const std::string& output_dir) {
        std::cout << "\nExporting results..." << std::endl;
        
        // Initialize VTU exporter with the base output directory
        // The exporter will create the vtu subdirectory internally
        vtu_exporter_ = std::make_unique<ExosphereVTUExporter>(mesh_, output_dir);
        
        std::cout << "Writing VTU files..." << std::endl;
        ProgressBar vtu_progress(temperatures_.size());
        
        for (size_t t = 0; t < temperatures_.size(); ++t) {
            vtu_exporter_->exportTimestep(
                t,
                temperatures_[t],
                vapor_pressure_[t],
                emission_flux_[t],
                deposition_flux_[t],
                net_flux_[t],
                accumulated_mass_[t],
                time_values_[t]
            );
            vtu_progress.update(t + 1);
        }
        vtu_progress.finish();
        
        vtu_exporter_->createPVDFile(temperatures_.size(), time_values_);
        
        exportMassBalance(output_dir);
        
        std::cout << "Results exported to " << output_dir << std::endl;

        // Create output directory if it doesn't exist
        std::filesystem::create_directories(output_dir);
        
        // Run final diagnostics
        diagnoseTransportPatterns(output_dir, temperatures_.size() - 1);
    }
    
    
};

// Load mesh from PTM3D binary PLY format with unit conversion
std::shared_ptr<Mesh> loadMesh(const std::string& mesh_file, const std::string& ptm_dir) {
    std::string ply_file = mesh_file.empty() ? ptm_dir + "/mesh.ply" : mesh_file;

    std::cout << "Loading mesh from: " << ply_file << std::endl;

    // 1) Preferred path: use PTM MeshUtils (handles PTM PLY/VTK and metadata)
    try {
        ptm::MeshUtils mu;
        if (mu.loadMeshFromFile(ply_file)) {
            auto pm = mu.getMesh();
            if (pm && !pm->faces.empty()) {
                auto mesh = std::make_shared<Mesh>();
                mesh->vertices     = pm->vertices;
                mesh->faces        = pm->faces;
                mesh->latitudes    = pm->latitudes;
                mesh->longitudes   = pm->longitudes;
                mesh->normals      = pm->face_normals;
                mesh->face_centers = pm->face_centers;

                // Ensure centers are exactly on the sphere
                ensureFaceCentersOnSphere(mesh->face_centers, constants::EUROPA_RADIUS);

                std::cout << "Successfully loaded mesh via MeshUtils: "
                          << mesh->vertices.size() << " vertices, "
                          << mesh->faces.size()    << " faces" << std::endl;
                return mesh;
            }
        } else {
            std::cout << "MeshUtils could not read mesh; falling back to local PLY reader" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cout << "MeshUtils load failed (" << e.what() << "); falling back to local PLY reader" << std::endl;
    }

    // 2) Fallback: existing PLY reader (will parse header and read vertices/faces)
    // NOTE: Keep original implementation intact below.
    std::ifstream file(ply_file, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open mesh file: " << ply_file << std::endl;
        return nullptr;
    }

    std::string line;
    int n_vertices = 0, n_faces = 0;
    bool has_normals = false, has_face_latlon = false;

    // Parse header
    while (std::getline(file, line)) {
        if (line.find("element vertex") != std::string::npos) {
            sscanf(line.c_str(), "element vertex %d", &n_vertices);
        } else if (line.find("element face") != std::string::npos) {
            sscanf(line.c_str(), "element face %d", &n_faces);
        } else if (line.find("property float nx") != std::string::npos) {
            has_normals = true;
        } else if (line.find("property float latitude") != std::string::npos) {
           
            has_face_latlon = true;
        } else if (line == "end_header") {
            break;
        }
    }
    
    std::cout << "  Found " << n_vertices << " vertices and " << n_faces << " faces" << std::endl;
    std::cout << "  Has normals: " << (has_normals ? "yes" : "no") << std::endl;
    std::cout << "  Has face lat/lon: " << (has_face_latlon ? "yes" : "no") << std::endl;

    // Read vertices
    auto mesh = std::make_shared<Mesh>();
    mesh->vertices.resize(n_vertices);
    if (has_normals) mesh->normals.resize(n_vertices);
    
    // CRITICAL FIX: Detect units and convert
    bool vertices_in_km = false;
    
    for (int i = 0; i < n_vertices; ++i) {
        float x, y, z;
        file.read(reinterpret_cast<char*>(&x), sizeof(float));
        file.read(reinterpret_cast<char*>(&y), sizeof(float));
        file.read(reinterpret_cast<char*>(&z), sizeof(float));
        
        // Check if vertices are in km (radius around 1560) or meters (radius around 1.56e6)
        if (i == 0) {
            double radius = std::sqrt(x*x + y*y + z*z);
            if (radius < 10000) {  // Likely in km
                vertices_in_km = true;
                std::cout << "  Detected vertices in kilometers, converting to meters" << std::endl;
            }
        }
        
        // Convert to meters if needed
        double scale = vertices_in_km ? 1000.0 : 1.0;
        mesh->vertices[i] = {x * scale, y * scale, z * scale};
        
        if (has_normals) {
            float nx, ny, nz;
            file.read(reinterpret_cast<char*>(&nx), sizeof(float));
            file.read(reinterpret_cast<char*>(&ny), sizeof(float));
            file.read(reinterpret_cast<char*>(&nz), sizeof(float));
            mesh->normals[i] = {nx, ny, nz};
        }
    }
    
    // Read faces (fallback parser expects triangles; if not, it will error)
    mesh->faces.resize(n_faces);
    if (has_face_latlon) {
        mesh->latitudes.resize(n_faces);
        mesh->longitudes.resize(n_faces);
    }
    for (int i = 0; i < n_faces; ++i) {
        unsigned char n_verts;
        file.read(reinterpret_cast<char*>(&n_verts), sizeof(unsigned char));
        if (n_verts !=  3) {
            throw std::runtime_error("Non-triangular face found");
        }
        
        int v0, v1, v2;
        file.read(reinterpret_cast<char*>(&v0), sizeof(int));
        file.read(reinterpret_cast<char*>(&v1), sizeof(int));
        file.read(reinterpret_cast<char*>(&v2), sizeof(int));
        mesh->faces[i] = {v0, v1, v2};
        
        if (has_face_latlon) {
            float lat, lon;
            file.read(reinterpret_cast<char*>(&lat), sizeof(float));
            file.read(reinterpret_cast<char*>(&lon), sizeof(float));
            mesh->latitudes[i] = lat;
            mesh->longitudes[i] = lon;
        }
    }
    
    file.close();

    std::cout << "Successfully loaded mesh with " << mesh->vertices.size()
              << " vertices and " << mesh->faces.size() << " faces" << std::endl;

    return mesh;
}

#endif

// Main function
// Cross-check the declared thermal-run albedo against the thermal run's own
// config_used.yaml (a verbatim copy of the config it read): a uniform declaration
// against `use_albedo_map: true`, or a map against `use_albedo_map: false`, is
// almost certainly the wrong temperature reference. A line-based read of one key,
// not a YAML parser, so a contradiction WARNS rather than refuses.
static void checkThermalConfigAlbedoMode(const std::string& ptm_dir, bool declared_map) {
    const std::string cfg = ptm_dir + "/config_used.yaml";
    std::ifstream in(cfg);
    if (!in) {
        std::cout << "  (no " << cfg << " to cross-check the thermal-run albedo against)" << std::endl;
        return;
    }
    std::string line;
    while (std::getline(in, line)) {
        const size_t hash = line.find('#');
        if (hash != std::string::npos) line.erase(hash);
        const size_t k = line.find("use_albedo_map:");
        if (k == std::string::npos) continue;
        std::string v = line.substr(k + std::string("use_albedo_map:").size());
        v.erase(0, v.find_first_not_of(" \t\"'"));
        v.erase(v.find_last_not_of(" \t\r\"'") + 1);
        const bool uses_map = (v == "true" || v == "True" || v == "yes");
        const bool no_map = (v == "false" || v == "False" || v == "no");
        if ((uses_map && !declared_map) || (no_map && declared_map)) {
            std::cout << "\n********************************************************************************\n"
                      << "WARNING: the declared thermal-run albedo is a " << (declared_map ? "per-face map" : "uniform value")
                      << ", but " << cfg << "\n  says use_albedo_map: " << v
                      << ". The temperature reference is probably wrong; check --thermal-albedo /\n"
                      << "  --thermal-albedo-map against the thermal config.\n"
                      << "********************************************************************************" << std::endl;
        } else if (uses_map || no_map) {
            std::cout << "  Thermal config cross-check: use_albedo_map: " << v << " is consistent with the "
                      << "declared " << (declared_map ? "per-face map" : "uniform value") << std::endl;
        }
        return;
    }
    // config_parser.h: illumination.use_albedo_map defaults to true (auto-discovered map).
    std::cout << "  Thermal config cross-check: " << cfg << " has no use_albedo_map key (the thermal "
              << "default is true: an auto-discovered map, when one exists)"
              << (declared_map ? "" : "; WARNING: a uniform thermal albedo was declared, check it")
              << std::endl;
}

int main(int argc, char** argv)
{
    std::string ptm_dir;
    std::string output_dir = "./exosphere_output";
    std::string mesh_file;
    int timesteps = 0;
    double equatorial_mm = 0.0;
    double polar_mm = 0.0;

    // Sputtering options
    bool sputtering_enabled = false;
    double sputtering_rate_scale = 1.0;
    double sputtering_hemisphere_contrast = 3.0;
    // Sputter mass accounting / products (PTM-642). Defaults live in SputteringConfig.
    ptm::exosphere::SputteringConfig sputter_cfg;
    std::string sputter_physics_flag;   // first sputter-physics flag seen (needs --sputtering)
    std::vector<std::pair<std::string, double>> sputter_overrides;  // numeric sputter flags, in order

    // Ice layer model options (lag deposit tracking)
    bool ice_layer_enabled = false;
    double ice_layer_dust_fraction = 0.05;  // 5% dust by volume

    // Multi-timescale evolution options
    bool multiscale_enabled = false;
    int multiscale_epochs = 1000;           // Default: ~1000 years
    int multiscale_cycles_per_epoch = 100;  // 100 diurnal cycles ≈ 1 year
    int multiscale_export_interval = 100;   // Export every 100 epochs
    // Not user-settable (M3): runMultiTimescaleSimulation() always super-steps
    // with the closed-form solution and recomputes the diurnal cycle only on the
    // first epoch or when albedo coupling changes temperatures. The CLI rejects
    // --no-analytical and --cache-refresh rather than accept and ignore them.
    bool analytical_acceleration = true;
    int cache_refresh_interval = 0;
    double convergence_threshold = -1.0;    // <0 = use default (1e-6)
    double flux_convergence_threshold = -1.0; // <0 = use default (0.01 = 1%)
    int diurnal_stride = 1;                 // Use every Nth timestep in diurnal averaging
    bool superstep_legacy_deposition = false;  // PTM-644: reproduce the pre-fix super-step
    bool no_albedo_step_limit = false;         // PTM-640: reproduce the unlimited super-step
    bool albedo_refresh_global = false;        // PTM-657: full recompute at every refresh
    double albedo_refresh_tol = 1e-2;          // PTM-657 face-local refresh tolerance
    double albedo_refresh_fraction = 0.0;      // PTM-657 faces past this x threshold are rescaled
    int max_superstep_epochs = 0;              // 0 = no cap
    std::string superstep_flag;                // first super-step flag seen (needs --multiscale)
    // Lag-layer diffusion barrier (multiscale; on by default, --no-lag-barrier = legacy).
    ptm::exosphere::LagBarrierConfig lag_cfg;
    bool no_lag_barrier = false;
    std::string lag_param_flag;                // first lag parameter flag seen

    // Albedo-composition coupling options
    bool albedo_coupling_enabled = false;
    ptm::exosphere::CompositionAlbedoConfig albedo_config;
    // Bond albedo the thermal run used (the albedo the loaded temperatures correspond
    // to). Required in albedo-coupling mode: --thermal-albedo A (uniform; the older
    // --initial-albedo is an alias) or --thermal-albedo-map F (per-face .bin).
    // hapke-intimate defaults to its --ice-fraction-albedo-map. No silent default.
    double thermal_albedo = std::numeric_limits<double>::quiet_NaN();
    std::string thermal_albedo_flag;     // which flag set thermal_albedo
    std::string thermal_albedo_map_path;
    std::string albedo_nonice_map_path;  // FITS/binary albedo map for per-face non-ice endmember
    std::string albedo_maps_dir;         // Directory for auto-discovery of albedo map
    bool init_from_albedo_map = false;   // Initialize concentrations from albedo map
    double albedo_percentile_low = 5.0;  // Lower percentile for concentration scaling
    double albedo_percentile_high = 95.0; // Upper percentile for concentration scaling

    // Ice fraction from albedo (--ice-fraction-from-albedo=uniform|percentile|hapke-intimate).
    // uniform (default) = the legacy uniform --dust-fraction on every face.
    std::string ice_fraction_mode;        // empty = not given (uniform)
    ptm::exosphere::HapkeIceFractionConfig hapke_cfg;
    std::string hapke_param_given;        // first hapke-only flag seen (for mode checks)
    std::string legacy_albedo_flag_given; // first legacy albedo-model flag seen

    // Ice crystallization model options
    bool crystallization_enabled = false;
    ptm::exosphere::CrystallizationConfig cryst_config;

    // Simple argument parsing
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--ptm-output" && i+1 < argc) { ptm_dir = argv[++i]; }
        else if (a == "--output" && i+1 < argc) { output_dir = argv[++i]; }
        else if (a == "--mesh" && i+1 < argc) { mesh_file = argv[++i]; }
        else if (a == "--timesteps" && i+1 < argc) { timesteps = std::stoi(argv[++i]); }
        else if (a == "--equatorial-ice" && i+1 < argc) { equatorial_mm = std::stod(argv[++i]); }
        else if (a == "--polar-ice" && i+1 < argc) { polar_mm = std::stod(argv[++i]); }
        // Sputtering options
        else if (a == "--sputtering") { sputtering_enabled = true; }
        else if (a == "--no-sputtering") { sputtering_enabled = false; }
        else if (a == "--sputtering-rate" && i+1 < argc) {
            sputtering_enabled = true;
            sputtering_rate_scale = std::stod(argv[++i]);
        }
        else if (a == "--sputter-legacy-escape-only") {
            if (sputter_physics_flag.empty()) sputter_physics_flag = a;
            sputter_cfg.legacy_escape_only = true;
        }
        else if (a == "--sputter-legacy-params") {
            if (sputter_physics_flag.empty()) sputter_physics_flag = a;
            sputter_cfg.legacy_params = true;
        }
        else if ((a == "--sputter-h2o-fraction" || a == "--sputter-escape-h2o" ||
                  a == "--sputter-escape-o2" || a == "--sputter-escape-h2" ||
                  a == "--sputter-launch-temperature" || a == "--sputter-binding-energy" ||
                  a == "--sputter-radiolysis-ea" || a == "--sputter-radiolysis-tref" ||
                  a == "--sputter-radiolysis-q0" ||
                  a == "--sputter-polar-enhancement") && i+1 < argc) {
            if (sputter_physics_flag.empty()) sputter_physics_flag = a;
            // Applied after parsing, once the parameter set is known (PTM-649).
            sputter_overrides.emplace_back(a, std::stod(argv[++i]));
        }
        else if (a == "--sputter-o2-map-absolute" && i+1 < argc) {
            if (sputter_physics_flag.empty()) sputter_physics_flag = a;
            sputter_cfg.o2_map_path = argv[++i];
        }
        else if ((a == "--sputter-ion-map" || a == "--sputter-ion-map-absolute") && i+1 < argc) {
            if (sputter_physics_flag.empty()) sputter_physics_flag = a;
            if (!sputter_cfg.ion_map_path.empty()) {
                std::cerr << "Error: give only one of --sputter-ion-map and --sputter-ion-map-absolute" << std::endl;
                return 1;
            }
            sputter_cfg.ion_map_path = argv[++i];
            sputter_cfg.ion_map_absolute = (a == "--sputter-ion-map-absolute");
        }
        else if (a == "--sputtering-contrast" && i+1 < argc) {
            sputtering_hemisphere_contrast = std::stod(argv[++i]);
        }
        // Ice layer model options
        else if (a == "--ice-layer") { ice_layer_enabled = true; }
        else if (a == "--no-ice-layer") { ice_layer_enabled = false; }
        else if (a == "--dust-fraction" && i+1 < argc) {
            ice_layer_enabled = true;
            ice_layer_dust_fraction = std::stod(argv[++i]);
        }
        // Multi-timescale evolution options
        else if (a == "--multiscale") {
            multiscale_enabled = true;
            ice_layer_enabled = true;  // Multi-timescale requires ice layer model
        }
        else if (a == "--epochs" && i+1 < argc) {
            multiscale_enabled = true;
            ice_layer_enabled = true;
            multiscale_epochs = std::stoi(argv[++i]);
        }
        else if (a == "--cycles-per-epoch" && i+1 < argc) {
            multiscale_cycles_per_epoch = std::stoi(argv[++i]);
        }
        else if (a == "--export-interval" && i+1 < argc) {
            multiscale_export_interval = std::stoi(argv[++i]);
        }
        else if (a == "--no-analytical" || a == "--cache-refresh") {
            std::cerr << "Error: " << a << " is not implemented and is refused.\n"
                      << "  The multi-timescale integrator always uses closed-form super-stepping\n"
                      << "  and recomputes the diurnal cycle only on the first epoch or when\n"
                      << "  --albedo-coupling changes temperatures by more than\n"
                      << "  --albedo-update-threshold. There is no periodic or adaptive cache\n"
                      << "  refresh and no way to disable super-stepping. (Earlier builds accepted\n"
                      << "  these flags and silently ignored them.)" << std::endl;
            return 1;
        }
        else if (a == "--diurnal-stride" && i+1 < argc) {
            diurnal_stride = std::stoi(argv[++i]);
        }
        else if (a == "--convergence-threshold" && i+1 < argc) {
            convergence_threshold = std::stod(argv[++i]);
        }
        else if (a == "--flux-convergence" && i+1 < argc) {
            flux_convergence_threshold = std::stod(argv[++i]);
        }
        else if (a == "--albedo-refresh-global") {
            albedo_refresh_global = true;
            if (superstep_flag.empty()) superstep_flag = a;
        }
        else if (a == "--albedo-refresh-tolerance" && i+1 < argc) {
            albedo_refresh_tol = std::stod(argv[++i]);
            if (!(albedo_refresh_tol >= 0.0 && albedo_refresh_tol < 0.1)) {
                std::cerr << "Error: --albedo-refresh-tolerance must be in [0, 0.1)" << std::endl;
                return 1;
            }
            if (superstep_flag.empty()) superstep_flag = a;
        }
        else if (a == "--albedo-refresh-fraction" && i+1 < argc) {
            albedo_refresh_fraction = std::stod(argv[++i]);
            if (!(albedo_refresh_fraction >= 0.0 && albedo_refresh_fraction <= 1.0)) {
                std::cerr << "Error: --albedo-refresh-fraction must be in [0, 1]" << std::endl;
                return 1;
            }
            if (superstep_flag.empty()) superstep_flag = a;
        }
        else if (a == "--no-albedo-step-limit") {
            no_albedo_step_limit = true;
            if (superstep_flag.empty()) superstep_flag = a;
        }
        else if (a == "--no-lag-barrier") {
            no_lag_barrier = true;
        }
        else if ((a == "--lag-barrier-length" || a == "--lag-grain-radius" || a == "--lag-porosity" ||
                  a == "--lag-tortuosity" || a == "--lag-optical-depth" || a == "--lag-source-refresh-tol")
                 && i+1 < argc) {
            if (lag_param_flag.empty()) lag_param_flag = a;
            const double v = std::stod(argv[++i]);
            bool ok = std::isfinite(v);
            if (a == "--lag-barrier-length") { ok = ok && v > 0.0; lag_cfg.barrier_length = v; }
            else if (a == "--lag-grain-radius") { ok = ok && v > 0.0; lag_cfg.grain_radius = v; }
            else if (a == "--lag-porosity") { ok = ok && v > 0.0 && v < 1.0; lag_cfg.lag_porosity = v; }
            else if (a == "--lag-tortuosity") { ok = ok && v >= 1.0; lag_cfg.tortuosity = v; }
            else if (a == "--lag-optical-depth") { ok = ok && v > 0.0; lag_cfg.optical_depth = v; }
            else { ok = ok && v > 0.0 && v < 1.0; lag_cfg.source_refresh_tol = v; }
            if (!ok) {
                std::cerr << "Error: " << a << " " << v << " is out of range (lengths [m] > 0; porosity in (0, 1); "
                          << "tortuosity >= 1; refresh tolerance in (0, 1))" << std::endl;
                return 1;
            }
        }
        else if (a == "--superstep-legacy-deposition") {
            superstep_legacy_deposition = true;
            if (superstep_flag.empty()) superstep_flag = a;
        }
        else if (a == "--max-superstep-epochs" && i+1 < argc) {
            max_superstep_epochs = std::stoi(argv[++i]);
            if (max_superstep_epochs < 1) {
                std::cerr << "Error: --max-superstep-epochs must be >= 1" << std::endl;
                return 1;
            }
            if (superstep_flag.empty()) superstep_flag = a;
        }
        // Albedo-composition coupling options
        else if (a == "--albedo-coupling") {
            albedo_coupling_enabled = true;
            ice_layer_enabled = true;  // Albedo coupling requires ice layer model
        }
        else if (a == "--albedo-ice" && i+1 < argc) {
            if (legacy_albedo_flag_given.empty()) legacy_albedo_flag_given = a;
            albedo_coupling_enabled = true;
            ice_layer_enabled = true;
            albedo_config.albedo_ice = std::stod(argv[++i]);
        }
        else if (a == "--albedo-nonice" && i+1 < argc) {
            if (legacy_albedo_flag_given.empty()) legacy_albedo_flag_given = a;
            albedo_coupling_enabled = true;
            ice_layer_enabled = true;
            albedo_config.albedo_nonice = std::stod(argv[++i]);
        }
        else if (a == "--albedo-update-threshold" && i+1 < argc) {
            albedo_config.albedo_update_threshold = std::stod(argv[++i]);
        }
        else if ((a == "--thermal-albedo" || a == "--initial-albedo") && i+1 < argc) {
            if (!thermal_albedo_flag.empty()) {
                std::cerr << "Error: " << a << " given after " << thermal_albedo_flag
                          << " (they are the same setting; give one)" << std::endl;
                return 1;
            }
            thermal_albedo_flag = a;
            thermal_albedo = std::stod(argv[++i]);
        }
        else if (a == "--thermal-albedo-map" && i+1 < argc) {
            thermal_albedo_map_path = argv[++i];
        }
        else if (a == "--albedo-nonice-map" && i+1 < argc) {
            if (legacy_albedo_flag_given.empty()) legacy_albedo_flag_given = a;
            albedo_nonice_map_path = argv[++i];
            albedo_coupling_enabled = true;
            ice_layer_enabled = true;
        }
        else if (a == "--albedo-maps-dir" && i+1 < argc) {
            if (legacy_albedo_flag_given.empty()) legacy_albedo_flag_given = a;
            albedo_maps_dir = argv[++i];
            albedo_coupling_enabled = true;
            ice_layer_enabled = true;
        }
        else if (a == "--init-from-albedo-map") {
            init_from_albedo_map = true;
            albedo_coupling_enabled = true;
            ice_layer_enabled = true;
        }
        else if (a.rfind("--ice-fraction-from-albedo", 0) == 0) {
            if (a == "--ice-fraction-from-albedo" && i+1 < argc) {
                ice_fraction_mode = argv[++i];
            } else if (a.rfind("--ice-fraction-from-albedo=", 0) == 0) {
                ice_fraction_mode = a.substr(std::string("--ice-fraction-from-albedo=").size());
            } else {
                std::cerr << "Error: --ice-fraction-from-albedo needs a value "
                          << "(uniform|percentile|hapke-intimate)" << std::endl;
                return 1;
            }
        }
        else if ((a == "--ice-albedo" || a == "--nonice-albedo" || a == "--g-ice" ||
                  a == "--g-nonice" || a == "--grain-ratio" ||
                  a == "--nonice-albedo-percentile") && i+1 < argc) {
            const double v = std::stod(argv[++i]);
            if (a == "--ice-albedo") hapke_cfg.ice_albedo = v;
            else if (a == "--nonice-albedo") {
                if (!(v >= 0.0)) {
                    std::cerr << "Error: --nonice-albedo must be >= 0" << std::endl;
                    return 1;
                }
                hapke_cfg.nonice_albedo = v;
            }
            else if (a == "--nonice-albedo-percentile") hapke_cfg.nonice_percentile = v;
            else if (a == "--g-ice") hapke_cfg.g_ice = v;
            else if (a == "--g-nonice") hapke_cfg.g_nonice = v;
            else hapke_cfg.grain_ratio = v;
            if (hapke_param_given.empty()) hapke_param_given = a;
        }
        else if (a == "--fill-ice-fraction" || a.rfind("--fill-ice-fraction=", 0) == 0) {
            std::string v;
            if (a == "--fill-ice-fraction") {
                if (i + 1 >= argc) {
                    std::cerr << "Error: --fill-ice-fraction needs a value (map-value|uniform)" << std::endl;
                    return 1;
                }
                v = argv[++i];
            } else {
                v = a.substr(std::string("--fill-ice-fraction=").size());
            }
            if (v == "map-value") hapke_cfg.fill_from_map_value = true;
            else if (v == "uniform") hapke_cfg.fill_from_map_value = false;
            else {
                std::cerr << "Error: --fill-ice-fraction must be map-value or uniform (got '" << v << "')"
                          << std::endl;
                return 1;
            }
            if (hapke_param_given.empty()) hapke_param_given = "--fill-ice-fraction";
        }
        else if ((a == "--ice-fraction-albedo-map" || a == "--albedo-fill-fits") && i+1 < argc) {
            if (a == "--ice-fraction-albedo-map") hapke_cfg.albedo_map_path = argv[++i];
            else hapke_cfg.fill_fits_path = argv[++i];
            if (hapke_param_given.empty()) hapke_param_given = a;
        }
        else if (a == "--albedo-percentile-low" && i+1 < argc) {
            if (legacy_albedo_flag_given.empty()) legacy_albedo_flag_given = a;
            albedo_percentile_low = std::stod(argv[++i]);
        }
        else if (a == "--albedo-percentile-high" && i+1 < argc) {
            if (legacy_albedo_flag_given.empty()) legacy_albedo_flag_given = a;
            albedo_percentile_high = std::stod(argv[++i]);
        }
        // Ice crystallization model options
        else if (a == "--crystallization") {
            crystallization_enabled = true;
            ice_layer_enabled = true;  // Crystallization requires ice layer model
        }
        else if (a == "--initial-crystallinity" && i+1 < argc) {
            crystallization_enabled = true;
            ice_layer_enabled = true;
            cryst_config.initial_crystalline_fraction = std::stod(argv[++i]);
        }
        else if (a == "--activation-energy" && i+1 < argc) {
            cryst_config.activation_energy = std::stod(argv[++i]);
        }
        else if (a == "--amorphization-dose" && i+1 < argc) {
            cryst_config.amorphization_dose = std::stod(argv[++i]);
        }
        else if (a == "--equatorial-porosity" && i+1 < argc) {
            cryst_config.equatorial_porosity = std::stod(argv[++i]);
        }
        else if (a == "--polar-porosity" && i+1 < argc) {
            cryst_config.polar_porosity = std::stod(argv[++i]);
        }
        else if (a == "--dose-hemisphere-contrast" && i+1 < argc) {
            cryst_config.hemisphere_dose_contrast = std::stod(argv[++i]);
        }
        else if (a == "--dose-polar-enhancement" && i+1 < argc) {
            cryst_config.polar_dose_enhancement = std::stod(argv[++i]);
        }
        else if (a == "--base-dose-rate" && i+1 < argc) {
            cryst_config.base_dose_rate = std::stod(argv[++i]);
        }
        else if (a == "--help" || a == "-h") {
            std::cout << "Usage: " << argv[0] << " --ptm-output DIR --output DIR --timesteps N [OPTIONS]\n\n";
            std::cout << "Required:\n";
            std::cout << "  --ptm-output DIR       Directory containing PTM3D temperature output\n";
            std::cout << "  --output DIR           Output directory for exosphere results\n";
            std::cout << "  --timesteps N          Number of timesteps to simulate\n\n";
            std::cout << "Optional:\n";
            std::cout << "  --mesh FILE            Mesh file (default: ptm_output/mesh.ply)\n";
            std::cout << "  --equatorial-ice MM    Initial equatorial ice thickness [mm]\n";
            std::cout << "  --polar-ice MM         Initial polar ice thickness [mm]\n\n";
            std::cout << "Sputtering options:\n";
            std::cout << "  --sputtering           Enable magnetospheric plasma sputtering\n";
            std::cout << "  --no-sputtering        Disable sputtering (sublimation only, default)\n";
            std::cout << "  --sputtering-rate S    Sputtering rate scale factor (default: 1.0)\n";
            std::cout << "                         1.0 = Cassidy et al. 2013 rates\n";
            std::cout << "  --sputtering-contrast C  Trailing/leading hemisphere ratio (default: 3.0)\n";
            std::cout << "  --sputter-h2o-fraction F  Mass fraction of gross sputtered mass leaving as H2O at\n";
            std::cout << "                         T_ref (default 0.90, Cassidy et al. 2013; legacy 0.85); the\n";
            std::cout << "                         rest is radiolytic O2 + H2 (2 H2O -> 2 H2 + O2)\n";
            std::cout << "  --sputter-binding-energy U  Effective binding energy [eV] of the Sigmund energy\n";
            std::cout << "                         distribution (default 0.054, Plainaki et al. 2010); sets the\n";
            std::cout << "                         H2O escape fraction (0.232) AND launch temperature (673 K)\n";
            std::cout << "  --sputter-radiolysis-ea E / --sputter-radiolysis-q0 Q / --sputter-radiolysis-tref T\n";
            std::cout << "                         tau(T) on the radiolytic part: [1 + Q e^(-E/kT)] /\n";
            std::cout << "                         [1 + Q e^(-E/kTref)] (default 0.06 eV, 1000, 100 K;\n";
            std::cout << "                         Cassidy et al. 2013 p. 71)\n";
            std::cout << "  --sputter-o2-map-absolute FILE  Per-face O2 production (O2 cm^-2 s^-1, e.g.\n";
            std::cout << "                         Addison et al. 2022); H2 stoichiometric, H2O = gross - O2 - H2,\n";
            std::cout << "                         no tau(T) (the map carries its own)\n";
            std::cout << "  --sputter-polar-enhancement E  Linear polar factor of the default cosine pattern\n";
            std::cout << "                         (default 1 = none; 1.2 in the legacy set; not from Cassidy)\n";
            std::cout << "  --sputter-ion-map FILE  Per-face ion weights (uint64 count + float64), normalized\n";
            std::cout << "                         to an area-weighted mean of 1 at Cassidy's 2e27 H2O/s\n";
            std::cout << "  --sputter-ion-map-absolute FILE  Per-face H2O-equivalent molecules cm^-2 s^-1 at\n";
            std::cout << "                         T_ref (e.g. Addison et al. 2022); sets pattern AND total\n";
            std::cout << "  --sputter-legacy-params  Pre-2026-09-23 parameters (f 0.85; escape 0.10/0.90/0.95;\n";
            std::cout << "                         683 K; stepped tau(T) on the whole yield, 50 K cutoff;\n";
            std::cout << "                         count-normalized weights). Reproduces earlier products\n";
            std::cout << "  --sputter-escape-h2o F / --sputter-escape-o2 F / --sputter-escape-h2 F\n";
            std::cout << "  --sputter-launch-temperature K  Legacy parameters only (the current set derives them)\n";
            std::cout << "  --sputter-legacy-escape-only  Pre-PTM-642 accounting: remove only the escaping\n";
            std::cout << "                         share, no transport, 0.10/0.05 O2/H2 split (NOT mass-\n";
            std::cout << "                         conserving; reproduces old products; implies\n";
            std::cout << "                         --sputter-legacy-params)\n\n";
            std::cout << "Ice layer model options (lag deposit tracking):\n";
            std::cout << "  --ice-layer            Enable multi-layer ice model with lag deposits\n";
            std::cout << "  --no-ice-layer         Disable ice layer model (default)\n";
            std::cout << "  --dust-fraction F      Dust volume fraction in ice (default: 0.05 = 5%%)\n";
            std::cout << "                         Controls lag deposit accumulation rate\n\n";
            std::cout << "Multi-timescale evolution options (geological timescales):\n";
            std::cout << "  --multiscale           Enable multi-timescale simulation\n";
            std::cout << "  --epochs N             Number of epochs to simulate (default: 1000)\n";
            std::cout << "                         Each epoch = cycles_per_epoch * 3.55 days\n";
            std::cout << "  --cycles-per-epoch N   Diurnal cycles per epoch (default: 100 ~ 1 year)\n";
            std::cout << "  --export-interval N    Export VTU every N epochs (default: 100, 0=final only)\n";
            std::cout << "  (--no-analytical and --cache-refresh are not implemented and are refused;\n";
            std::cout << "   the diurnal cycle is recomputed only on epoch 0 and on albedo-coupling\n";
            std::cout << "   temperature updates)\n";
            std::cout << "  --convergence-threshold T  Max concentration change per epoch for convergence\n";
            std::cout << "                         (default: 1e-6)\n";
            std::cout << "  --flux-convergence T   Flux imbalance fraction for convergence (default: 0.01 = 1%%)\n";
            std::cout << "  --max-superstep-epochs N  Cap one super-step at N epochs (default: none; for\n";
            std::cout << "                         step-size refinement checks)\n";
            std::cout << "  --albedo-refresh-global  Recompute the full transport at every albedo refresh\n";
            std::cout << "                         (pre-PTM-657; slower). Default: face-local update of the\n";
            std::cout << "                         sources whose emission/temperature moved\n";
            std::cout << "  --albedo-refresh-tolerance T  Face-local refresh: sources left stale may differ\n";
            std::cout << "                         from the current field by at most T of the totals\n";
            std::cout << "                         (default 1e-2; 0 = update every changed source)\n";
            std::cout << "  --albedo-refresh-fraction F  Face-local refresh: rescale the temperatures of faces\n";
            std::cout << "                         whose albedo moved by more than F x the threshold (default\n";
            std::cout << "                         0 = every face that moved, as the global refresh; F > 0 is\n";
            std::cout << "                         faster but re-bases faces at different times, face-scale\n";
            std::cout << "                         jitter that the albedo feedback amplifies)\n";
            std::cout << "  --no-albedo-step-limit  Do not end super-steps when a face's albedo would pass the\n";
            std::cout << "                         refresh threshold (pre-PTM-640; reproduces earlier products)\n";
            std::cout << "  --no-lag-barrier       Legacy well-mixed layer: lost ice becomes non-ice volume and\n";
            std::cout << "                         nothing limits sublimation but the exposed ice area (for\n";
            std::cout << "                         comparison; default: the lag-layer diffusion barrier)\n";
            std::cout << "  Lag barrier (default on in --multiscale): sublimation J_0 f / (1 + d/l) through a lag\n";
            std::cout << "  of thickness d left by the lost ice; sputtering is not barriered; deposition beyond\n";
            std::cout << "  the face's loss builds an unbarriered frost veneer; the lag darkens the face,\n";
            std::cout << "  A = A_lag + (A_mix - A_lag) exp(-d/delta_opt). Thermal inertia is unchanged.\n";
            std::cout << "  --lag-barrier-length L  l [m] directly (default: Knudsen, 1.128 [phi/(1-phi)] r / tau,\n";
            std::cout << "                         Schorghofer 2008 ApJ 682:697 eq. 3; 4.23e-6 m at the defaults)\n";
            std::cout << "  --lag-grain-radius R   Non-ice grain radius r [m] (default 5e-6)\n";
            std::cout << "  --lag-porosity P       Lag porosity phi_lag (default 0.6)\n";
            std::cout << "  --lag-tortuosity T     Lag tortuosity tau (default 2)\n";
            std::cout << "  --lag-optical-depth D  Albedo e-folding lag thickness delta_opt [m] (default 1e-5)\n";
            std::cout << "  --lag-source-refresh-tol X  Refresh the transport when the emission pattern has\n";
            std::cout << "                         moved by X (total variation; default 0.02)\n";
            std::cout << "  --superstep-legacy-deposition  Hold deposition at its start-of-step value over\n";
            std::cout << "                         each super-step (pre-PTM-644; not mass-conserving;\n";
            std::cout << "                         reproduces earlier products only)\n";
            std::cout << "  --diurnal-stride N     Use every Nth timestep in diurnal averaging (default: 1)\n";
            std::cout << "                         stride=4 gives ~4x faster diurnal computation, <2%% error\n\n";
            std::cout << "Albedo-composition coupling options (thermal segregation feedback):\n";
            std::cout << "  --albedo-coupling      Enable composition-dependent albedo feedback\n";
            std::cout << "  --albedo-ice A         Pure ice Bond albedo endmember (default: 0.65)\n";
            std::cout << "  --albedo-nonice A      Non-ice lag deposit Bond albedo (default: 0.10)\n";
            std::cout << "  --albedo-nonice-map F  FITS/binary albedo map for per-face non-ice endmember\n";
            std::cout << "                         Overrides --albedo-nonice with spatially-varying values\n";
            std::cout << "  --albedo-maps-dir DIR  Auto-discover albedo map in directory\n";
            std::cout << "  --albedo-update-threshold T  Refresh the temperatures when any face's albedo has\n";
            std::cout << "                         moved by more than T since the last refresh (default: 0.01)\n";
            std::cout << "  --thermal-albedo A     uniform Bond albedo the thermal run used; the temperature\n";
            std::cout << "                         reference (--initial-albedo is an alias)\n";
            std::cout << "  --thermal-albedo-map F per-face Bond albedo .bin the thermal run used\n";
            std::cout << "                         (uint64 count + double[count]). Albedo coupling requires\n";
            std::cout << "                         one of the two; hapke-intimate defaults to its\n";
            std::cout << "                         --ice-fraction-albedo-map. There is no silent default\n\n";
            std::cout << "Ice fraction from albedo (optional coupling of initial ice to the albedo map):\n";
            std::cout << "  --ice-fraction-from-albedo=MODE\n";
            std::cout << "      uniform         (default) uniform --dust-fraction on every face\n";
            std::cout << "      percentile      legacy rank mapping of the map (5th-95th percentile);\n";
            std::cout << "                      not a physical inversion. Same as --init-from-albedo-map\n";
            std::cout << "      hapke-intimate  invert the per-face Bond albedo with an intimate Hapke\n";
            std::cout << "                      mixture (w by extinction, g by scattering cross section);\n";
            std::cout << "                      sublimation scales with the EXPOSED ice area fraction f,\n";
            std::cout << "                      phi_ice = f r / (f r + 1 - f). Requires --multiscale.\n";
            std::cout << "  --ice-fraction-albedo-map F  per-face Bond albedo .bin used by the thermal run\n";
            std::cout << "                      (required for hapke-intimate)\n";
            std::cout << "  --albedo-fill-fits F  FITS map used to flag faces that mixed in fill pixels\n";
            std::cout << "                      (default: Europa-0.5deg-BondAlbedo.fits next to the .bin;\n";
            std::cout << "                      'none' skips fill detection, with a warning)\n";
            std::cout << "  --fill-ice-fraction M  initial composition of fill faces (hapke-intimate):\n";
            std::cout << "      map-value       (default) invert the face's own map value, the albedo the\n";
            std::cout << "                      thermal run used there, so composition and temperature\n";
            std::cout << "                      agree; faces with no usable value get 1 - --dust-fraction\n";
            std::cout << "      uniform         every fill face gets exposed ice fraction 1 - --dust-fraction\n";
            std::cout << "                      (pre-PTM-637 behavior; disagrees with the thermal albedo)\n";
            std::cout << "  --ice-albedo A     ice endmember Bond albedo (default: 0.80, Fermi's base case)\n";
            std::cout << "  --nonice-albedo A  pin the non-ice endmember Bond albedo (e.g. 0.3814); overrides\n";
            std::cout << "                     the default, which is the map's area-weighted percentile below.\n";
            std::cout << "                     Pin it when the map is uniform (its percentile is the map value)\n";
            std::cout << "  --nonice-albedo-percentile P  percentile of the map (area-weighted, fill\n";
            std::cout << "                     faces excluded) used as the non-ice endmember (default: 10)\n";
            std::cout << "  --g-ice G          ice asymmetry parameter (default: 0)\n";
            std::cout << "  --g-nonice G       non-ice asymmetry parameter (default: 0)\n";
            std::cout << "  --grain-ratio R    D_ice / D_nonice (default: 10)\n\n";
            std::cout << "Ice crystallization model options:\n";
            std::cout << "  --crystallization      Enable ice crystallization tracking\n";
            std::cout << "                         (amorphous/crystalline competition)\n";
            std::cout << "  --initial-crystallinity F  Initial crystalline fraction [0-1] (default: 0.0)\n";
            std::cout << "  --activation-energy E  Crystallization activation energy [J/mol]\n";
            std::cout << "                         (default: 62000, Mitchell et al. 2017)\n";
            std::cout << "  --amorphization-dose D Dose threshold [eV/molecule] (default: 3.0)\n";
            std::cout << "  --equatorial-porosity P  Equatorial porosity [0-1] (default: 0.5)\n";
            std::cout << "  --polar-porosity P     Polar porosity [0-1] (default: 0.2)\n";
            std::cout << "  --dose-hemisphere-contrast C  Trailing/leading dose ratio (default: 5.0)\n";
            std::cout << "  --dose-polar-enhancement P    Pole/equator dose ratio (default: 2.5)\n";
            std::cout << "  --base-dose-rate R     Base dose rate [eV/(mol s)] (default: 2.3e-6)\n";
            return 0;
        } else {
            std::cerr << "Unknown arg: " << a << std::endl;
            return 1;
        }
    }

    // Resolve --ice-fraction-from-albedo. uniform leaves every legacy path untouched.
    if (init_from_albedo_map) {
        if (!ice_fraction_mode.empty() && ice_fraction_mode != "percentile") {
            std::cerr << "Error: --init-from-albedo-map (an alias for "
                      << "--ice-fraction-from-albedo=percentile) conflicts with "
                      << "--ice-fraction-from-albedo=" << ice_fraction_mode << std::endl;
            return 1;
        }
        ice_fraction_mode = "percentile";
    }
    if (ice_fraction_mode.empty()) ice_fraction_mode = "uniform";
    if (ice_fraction_mode != "uniform" && ice_fraction_mode != "percentile" &&
        ice_fraction_mode != "hapke-intimate") {
        std::cerr << "Error: unknown --ice-fraction-from-albedo mode '" << ice_fraction_mode
                  << "' (expected uniform|percentile|hapke-intimate)" << std::endl;
        return 1;
    }
    if (ice_fraction_mode != "hapke-intimate" && !hapke_param_given.empty()) {
        std::cerr << "Error: " << hapke_param_given << " is used only by "
                  << "--ice-fraction-from-albedo=hapke-intimate (mode is " << ice_fraction_mode
                  << ")" << std::endl;
        return 1;
    }
    if (ice_fraction_mode == "percentile") {
        // Identical to the legacy --init-from-albedo-map.
        init_from_albedo_map = true;
        albedo_coupling_enabled = true;
        ice_layer_enabled = true;
    }
    if (!lag_param_flag.empty() && no_lag_barrier) {
        std::cerr << "Error: " << lag_param_flag << " sets a lag-barrier parameter but --no-lag-barrier "
                  << "switches the barrier off" << std::endl;
        return 1;
    }
    if (!lag_param_flag.empty() && !multiscale_enabled) {
        std::cerr << "Error: " << lag_param_flag << " requires --multiscale (the lag barrier is a "
                  << "multiscale model)" << std::endl;
        return 1;
    }
    if (!superstep_flag.empty() && !multiscale_enabled) {
        std::cerr << "Error: " << superstep_flag << " requires --multiscale" << std::endl;
        return 1;
    }
    if (ice_fraction_mode == "hapke-intimate") {
        if (!multiscale_enabled) {
            std::cerr << "Error: --ice-fraction-from-albedo=hapke-intimate requires --multiscale. "
                      << "The standard diurnal path does not apply the ice availability factor "
                      << "to the emitted flux, so the exposed ice fraction would have no effect."
                      << std::endl;
            return 1;
        }
        if (hapke_cfg.albedo_map_path.empty()) {
            std::cerr << "Error: --ice-fraction-from-albedo=hapke-intimate requires "
                      << "--ice-fraction-albedo-map <per-face Bond albedo .bin> (the map the "
                      << "thermal run used, e.g. data/albedo_maps/Europa_Albedo_res6.bin)" << std::endl;
            return 1;
        }
        if (!legacy_albedo_flag_given.empty()) {
            std::cerr << "Error: " << legacy_albedo_flag_given << " belongs to the legacy linear "
                      << "albedo model; hapke-intimate uses --ice-albedo/--nonice-albedo/--g-ice/"
                      << "--g-nonice/--grain-ratio and its own forward model" << std::endl;
            return 1;
        }
        if (!(hapke_cfg.grain_ratio > 0.0)) {
            std::cerr << "Error: --grain-ratio must be > 0" << std::endl;
            return 1;
        }
        hapke_cfg.fill_exposed_ice_fraction = 1.0 - std::clamp(ice_layer_dust_fraction, 0.0, 1.0);
        albedo_coupling_enabled = true;
        ice_layer_enabled = true;
    }

    if (!thermal_albedo_flag.empty() && !thermal_albedo_map_path.empty()) {
        std::cerr << "Error: " << thermal_albedo_flag << " and --thermal-albedo-map both declare the "
                  << "thermal-run albedo; give one" << std::endl;
        return 1;
    }

    if (ptm_dir.empty()) {
        std::cerr << "Error: --ptm-output is required\n";
        return 1;
    }
    if (timesteps <= 0) {
        std::cerr << "Error: --timesteps N is required and must be > 0\n";
        return 1;
    }

    std::cout << "PTM3D output directory: " << ptm_dir << std::endl;
    std::cout << "Exosphere output directory: " << output_dir << std::endl;
    std::cout << "Number of timesteps: " << timesteps << std::endl;

    // Load mesh (prefer user-specified mesh_file; otherwise try PTM directory mesh.ply)
    std::string mesh_path = mesh_file.empty() ? (ptm_dir + "/mesh.ply") : mesh_file;
    std::cout << "Loading mesh: " << mesh_path << "..." << std::flush;
    auto t_mesh_start = std::chrono::steady_clock::now();
    auto mesh = load_mesh_with_meshutils(mesh_path);
    if (!mesh) {
        std::cerr << "\nFailed to load mesh: " << mesh_path << std::endl;
        return 1;
    }
    ensure_face_centers_on_sphere(mesh);
    auto t_mesh_end = std::chrono::steady_clock::now();
    double mesh_sec = std::chrono::duration<double>(t_mesh_end - t_mesh_start).count();
    std::cout << " " << mesh->faces.size() << " faces (" << std::fixed
              << std::setprecision(1) << mesh_sec << " s)" << std::endl;

    // Create integration instance
    ptm::exosphere::PTMExosphereIntegration integrator(output_dir);

    // Let integrator use the mesh from MeshUtils or load from path using its public API
    if (!mesh_file.empty()) {
        // Prefer integrator loading when a file path is explicitly provided
        if (!integrator.loadMesh(mesh_file)) {
            std::cerr << "Failed to load mesh via integrator: " << mesh_file << std::endl;
            return 1;
        }
    } else {
        // Use the mesh we already loaded (throws on unusable geometry, EXO-02)
        try {
            integrator.setMesh(mesh);
        } catch (const std::exception& e) {
            std::cerr << e.what() << std::endl;
            return 1;
        }
    }

    // Load temperatures from PTM output directory
    std::cout << "Loading temperature data (" << timesteps << " timesteps)..." << std::flush;
    auto t_temp_start = std::chrono::steady_clock::now();
    if (!integrator.loadTemperatureData(ptm_dir, timesteps)) {
        std::cerr << "\nFailed to load temperature data from: " << ptm_dir << std::endl;
        return 1;
    }
    auto t_temp_end = std::chrono::steady_clock::now();
    double temp_sec = std::chrono::duration<double>(t_temp_end - t_temp_start).count();
    std::cout << " done (" << std::fixed << std::setprecision(1) << temp_sec << " s)" << std::endl;

    // Sputter-physics flags without sputtering would be silently ignored.
    if (!sputter_physics_flag.empty() && !sputtering_enabled) {
        std::cerr << "Error: " << sputter_physics_flag << " requires --sputtering" << std::endl;
        return 1;
    }

    // Sputter parameter set (PTM-649). The legacy set (--sputter-legacy-params, implied
    // by --sputter-legacy-escape-only) takes the old values, then the overrides. The
    // current set DERIVES the H2O escape fraction and launch temperature from the
    // binding energy U and fixes the O2/H2 fates, so those overrides are refused.
    if (sputter_cfg.legacyParams()) {
        sputter_cfg.applyLegacyParams();
    }
    for (const auto& [flag, v] : sputter_overrides) {
        const bool legacy_only = (flag == "--sputter-escape-h2o" || flag == "--sputter-escape-o2" ||
                                  flag == "--sputter-escape-h2" || flag == "--sputter-launch-temperature");
        const bool current_only = (flag == "--sputter-binding-energy" || flag == "--sputter-radiolysis-ea" ||
                                   flag == "--sputter-radiolysis-tref" || flag == "--sputter-radiolysis-q0");
        if (legacy_only && !sputter_cfg.legacyParams()) {
            std::cerr << "Error: " << flag << " applies to the legacy sputter parameters only. The current set "
                      << "derives the H2O escape fraction and launch temperature from the binding energy "
                      << "(--sputter-binding-energy, Sigmund-Thompson) so they cannot disagree; O2 is "
                      << "atmospheric loss and H2 escapes. Add --sputter-legacy-params to set it." << std::endl;
            return 1;
        }
        if (current_only && sputter_cfg.legacyParams()) {
            std::cerr << "Error: " << flag << " has no effect with the legacy sputter parameters" << std::endl;
            return 1;
        }
        if (flag == "--sputter-h2o-fraction") sputter_cfg.h2o_fraction = v;
        else if (flag == "--sputter-escape-h2o") sputter_cfg.h2o_escape_fraction = v;
        else if (flag == "--sputter-escape-o2") sputter_cfg.o2_escape_fraction = v;
        else if (flag == "--sputter-escape-h2") sputter_cfg.h2_escape_fraction = v;
        else if (flag == "--sputter-launch-temperature") {
            sputter_cfg.launch_temperature_K = v;
            sputter_cfg.launch_temperature_given = true;
        }
        else if (flag == "--sputter-binding-energy") sputter_cfg.binding_energy_eV = v;
        else if (flag == "--sputter-radiolysis-ea") sputter_cfg.radiolysis_activation_eV = v;
        else if (flag == "--sputter-radiolysis-tref") sputter_cfg.radiolysis_reference_K = v;
        else if (flag == "--sputter-radiolysis-q0") sputter_cfg.radiolysis_q0 = v;
        else if (flag == "--sputter-polar-enhancement") sputter_cfg.polar_enhancement = v;
    }
    if (!sputter_cfg.o2_map_path.empty() && sputter_cfg.legacyParams()) {
        std::cerr << "Error: --sputter-o2-map-absolute needs the current sputter parameters" << std::endl;
        return 1;
    }
    if (!sputter_cfg.ion_map_path.empty() && sputter_cfg.legacyParams()) {
        std::cerr << "Error: --sputter-ion-map / --sputter-ion-map-absolute need the current sputter "
                  << "parameters (not --sputter-legacy-params / --sputter-legacy-escape-only)" << std::endl;
        return 1;
    }
    if (sputter_cfg.legacyParams() && sputtering_enabled) {
        std::cerr << "WARNING [PTM-649]: legacy sputter parameters (f_H2O 0.85, escape 0.10/0.90/0.95, "
                  << "stepped tau(T) on the whole yield with a 50 K cutoff, count-normalized weights). "
                  << "Reproduces earlier products only." << std::endl;
    }

    // Enable sputtering if requested
    if (sputtering_enabled) {
        std::cout << "\nEnabling magnetospheric plasma sputtering:" << std::endl;
        std::cout << "  Rate scale factor: " << sputtering_rate_scale << std::endl;
        std::cout << "  Hemisphere contrast: " << sputtering_hemisphere_contrast << std::endl;
        sputter_cfg.rate_scale = sputtering_rate_scale;
        sputter_cfg.hemisphere_contrast = sputtering_hemisphere_contrast;
        try {
            integrator.enableSputtering(sputter_cfg);
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            return 1;
        }
        if (!multiscale_enabled && !sputter_cfg.legacy_escape_only) {
            std::cerr << "WARNING [PTM-642]: the per-timestep (non --multiscale) path still removes only "
                      << "the escaping share of sputtered mass and does not transport the rest; the "
                      << "mass-conserving sputter accounting is implemented in --multiscale only." << std::endl;
        }
    } else {
        std::cout << "\nSputtering disabled (sublimation-only mode)" << std::endl;
    }

    // Enable ice layer model if requested
    if (ice_layer_enabled) {
        std::cout << "\nEnabling ice layer model (lag deposit tracking):" << std::endl;
        std::cout << "  Dust volume fraction: " << (ice_layer_dust_fraction * 100) << "%" << std::endl;
        integrator.enableIceLayerModel(ice_layer_dust_fraction);
    }

    // Ice fraction from albedo: startup banner (every mode) and the hapke-intimate
    // initialization (before crystallization, whose state initializeFromConcentrations
    // would otherwise reset).
    std::cout << "\nIce fraction from albedo: " << ice_fraction_mode;
    if (ice_fraction_mode == "uniform") {
        if (ice_layer_enabled) {
            std::cout << " (default): non-ice volume fraction " << ice_layer_dust_fraction
                      << " on every face" << std::endl;
        } else {
            std::cout << " (default): ice layer model off" << std::endl;
        }
    } else if (ice_fraction_mode == "percentile") {
        std::cout << " (legacy --init-from-albedo-map): rank mapping between the P"
                  << albedo_percentile_low << " and P" << albedo_percentile_high
                  << " albedo percentiles. NOTE: this is not a physical inversion" << std::endl;
    } else {
        std::cout << " (intimate Hapke inversion of the Bond albedo map)" << std::endl;
        if (!integrator.initializeConcentrationsFromHapkeIntimate(hapke_cfg, albedo_config)) {
            std::cerr << "Error: hapke-intimate ice-fraction initialization failed" << std::endl;
            return 1;
        }
    }

    // Enable crystallization model if requested (requires ice layer model)
    if (crystallization_enabled) {
        std::cout << "\nEnabling ice crystallization model:" << std::endl;
        cryst_config.enabled = true;
        integrator.enableCrystallizationModel(cryst_config);
    }

    // Enable albedo-composition coupling if requested (requires ice layer model)
    if (albedo_coupling_enabled) {
        // Load per-face non-ice endmember map before enabling coupling
        // (legacy linear model; hapke-intimate refuses these flags above)
        if (!albedo_nonice_map_path.empty() || !albedo_maps_dir.empty()) {
            integrator.loadNoniceAlbedoMap(albedo_nonice_map_path, albedo_maps_dir);
        }

        // Initialize concentrations from albedo map if requested
        // (must come after loadNoniceAlbedoMap and enableIceLayerModel)
        if (init_from_albedo_map) {
            integrator.initializeConcentrationsFromAlbedoMap(
                albedo_percentile_low, albedo_percentile_high);
        }

        // Temperature reference: the albedo the thermal run used. Refuse rather than guess.
        bool thermal_ok = false;
        if (!thermal_albedo_map_path.empty()) {
            std::cout << "\nThermal-run albedo: per-face map " << thermal_albedo_map_path
                      << " (--thermal-albedo-map)" << std::endl;
            thermal_ok = integrator.loadThermalAlbedoMap(thermal_albedo_map_path);
        } else if (!thermal_albedo_flag.empty()) {
            std::cout << "\nThermal-run albedo: uniform " << thermal_albedo << " (" << thermal_albedo_flag
                      << ")" << std::endl;
            thermal_ok = integrator.setThermalAlbedoUniform(thermal_albedo, "uniform " +
                std::to_string(thermal_albedo) + " (" + thermal_albedo_flag + ")");
        } else if (ice_fraction_mode == "hapke-intimate") {
            std::cout << "\nThermal-run albedo: per-face map " << hapke_cfg.albedo_map_path
                      << " (the hapke-intimate --ice-fraction-albedo-map, documented as the map the "
                      << "thermal run used; override with --thermal-albedo-map/--thermal-albedo)" << std::endl;
            thermal_ok = integrator.loadThermalAlbedoMap(hapke_cfg.albedo_map_path);
        } else {
            std::cerr << "Error: albedo coupling (mode " << ice_fraction_mode << ") needs the Bond albedo "
                      << "the thermal run used, because the temperatures are rescaled relative to it. "
                      << "Pass --thermal-albedo A (uniform thermal albedo, e.g. 0.5 for the Europa "
                      << "uniform-ice configs) or --thermal-albedo-map F (the per-face .bin the thermal "
                      << "run used). The exosphere will not guess it." << std::endl;
            return 1;
        }
        if (!thermal_ok) {
            std::cerr << "Error: could not set the thermal-run albedo" << std::endl;
            return 1;
        }
        checkThermalConfigAlbedoMode(ptm_dir, !thermal_albedo_map_path.empty() ||
            (thermal_albedo_flag.empty() && ice_fraction_mode == "hapke-intimate"));
        if (!integrator.enableAlbedoCoupling(albedo_config)) {
            std::cerr << "Error: albedo coupling could not be enabled" << std::endl;
            return 1;
        }
        if (!multiscale_enabled) {
            std::cout << "WARNING: albedo coupling refreshes temperatures only in --multiscale mode. "
                      << "This standard-mode run exports bond_albedo but keeps the thermal-run "
                      << "temperatures unchanged." << std::endl;
        }
    } else if (!thermal_albedo_flag.empty() || !thermal_albedo_map_path.empty()) {
        std::cerr << "Error: " << (thermal_albedo_map_path.empty() ? thermal_albedo_flag
                                                                   : std::string("--thermal-albedo-map"))
                  << " is used only with albedo coupling (--albedo-coupling or an "
                  << "--ice-fraction-from-albedo mode other than uniform)" << std::endl;
        return 1;
    }

    // Run simulation based on mode
    if (multiscale_enabled) {
        // Multi-timescale evolution over geological timescales
        std::cout << "\nRunning multi-timescale evolution:" << std::endl;
        std::cout << "  Epochs: " << multiscale_epochs << std::endl;
        std::cout << "  Cycles per epoch: " << multiscale_cycles_per_epoch << std::endl;
        std::cout << "  Export interval: " << multiscale_export_interval << " epochs" << std::endl;
        std::cout << "  Diurnal-cycle refresh: epoch 0"
                  << (albedo_coupling_enabled ? " + albedo-coupling temperature updates" : " only")
                  << " (closed-form super-stepping always on)" << std::endl;

        // Calculate total simulation time for display
        double diurnal_period = 3.5512 * 24.0 * 3600.0;  // seconds
        double epoch_duration = multiscale_cycles_per_epoch * diurnal_period;
        double total_years = multiscale_epochs * epoch_duration / (365.25 * 24.0 * 3600.0);
        std::cout << "  Total simulation time: " << std::scientific << std::setprecision(2)
                  << total_years << " years" << std::endl;

        // Set convergence thresholds if user specified them
        if (convergence_threshold >= 0.0) {
            integrator.setConvergenceThreshold(convergence_threshold);
        }
        if (flux_convergence_threshold >= 0.0) {
            integrator.setFluxConvergenceThreshold(flux_convergence_threshold);
        }
        integrator.setSuperstepConsistentDeposition(!superstep_legacy_deposition);
        if (sputtering_enabled && sputter_cfg.legacy_escape_only && !no_albedo_step_limit) {
            no_albedo_step_limit = true;
            std::cerr << "WARNING [PTM-640]: --sputter-legacy-escape-only implies --no-albedo-step-limit "
                      << "(reproduces pre-fix products)" << std::endl;
        }
        if (no_albedo_step_limit && albedo_coupling_enabled) {
            std::cerr << "WARNING [PTM-640]: --no-albedo-step-limit: super-steps are not limited by the albedo "
                      << "change, so the temperature refresh can lag by a whole step (the answer then depends "
                      << "on the step size). Reproduces earlier products only." << std::endl;
        }
        integrator.setAlbedoStepLimit(!no_albedo_step_limit);
        if (sputtering_enabled && sputter_cfg.legacy_escape_only) albedo_refresh_global = true;
        integrator.setLocalAlbedoRefresh(!albedo_refresh_global, albedo_refresh_tol, albedo_refresh_fraction);
        if (!albedo_refresh_global && albedo_refresh_fraction > 0.0 && albedo_coupling_enabled) {
            std::cerr << "WARNING [PTM-657]: --albedo-refresh-fraction " << albedo_refresh_fraction
                      << ": faces are re-based at different times, which injects face-scale temperature "
                      << "jitter; the albedo-sublimation feedback is linearly unstable on warm faces "
                      << "(PTM-658) and amplifies it, so local extremes can differ from the coherent "
                      << "refresh (phase 0 cosine case: max non-ice 35.9% vs 18.2%)." << std::endl;
        }
        integrator.setMaxSuperstepEpochs(max_superstep_epochs);
        // Lag-layer diffusion barrier (default on). The legacy reproduction modes imply
        // --no-lag-barrier (they reproduce pre-fix products); say so.
        if (!no_lag_barrier && (superstep_legacy_deposition ||
                                (sputtering_enabled && sputter_cfg.legacy_escape_only))) {
            if (!lag_param_flag.empty()) {
                std::cerr << "Error: " << lag_param_flag << " needs the lag barrier, which the legacy reproduction "
                          << "flags (--superstep-legacy-deposition, --sputter-legacy-escape-only) switch off"
                          << std::endl;
                return 1;
            }
            no_lag_barrier = true;
            std::cerr << "WARNING [lag]: " << (superstep_legacy_deposition ? "--superstep-legacy-deposition"
                                                                           : "--sputter-legacy-escape-only")
                      << " implies --no-lag-barrier (reproduces pre-lag products)" << std::endl;
        }
        lag_cfg.enabled = !no_lag_barrier;
        integrator.setLagBarrier(lag_cfg);
        if (diurnal_stride > 1) {
            integrator.setDiurnalStride(diurnal_stride);
            std::cout << "  Diurnal accumulation stride: " << diurnal_stride << std::endl;
        }
        try {
            integrator.runMultiTimescaleSimulation(multiscale_epochs,
                                                    multiscale_cycles_per_epoch,
                                                    multiscale_export_interval,
                                                    analytical_acceleration,
                                                    cache_refresh_interval);
        } catch (const std::exception& e) {
            // e.g., an invalid EXO_DEP_SMOOTH_DEG: fail loudly with the reason.
            std::cerr << "Error: " << e.what() << std::endl;
            return 1;
        }
    } else {
        if (ice_layer_enabled && !no_lag_barrier) {
            std::cerr << "WARNING [lag]: the lag-layer diffusion barrier is a --multiscale model; this "
                      << "per-timestep run uses the legacy well-mixed ice layer" << std::endl;
        }
        // Standard diurnal simulation
        std::cout << "\nRunning standard diurnal simulation..." << std::endl;
        integrator.runSimulation(3600.0, true);
    }

    std::cout << "Exosphere simulation finished. Results in: " << output_dir << std::endl;
    return 0;
}