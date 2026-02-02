#include "ptm_exosphere_integration.h"
#include "exosphere_physics.h"
#include "sputtering_model.h"
#include "ice_layer_model.h"
#include "ice_crystallization_model.h"
#include "multitimescale_evolution.h"
#include "../mesh/mesh_utils.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include <numeric>
#include <iomanip>
#include <sstream>
#include <cmath>  // For M_PI and other math functions
#include <chrono>

namespace ptm {
namespace exosphere {

PTMExosphereIntegration::PTMExosphereIntegration(const std::string& output_dir)
    : output_dir_(output_dir) {
    std::filesystem::create_directories(output_dir_);
    // Ensure unbuffered immediate output for progress heartbeats
    std::cout.setf(std::ios::unitbuf);
    std::cerr.setf(std::ios::unitbuf);
}

PTMExosphereIntegration::~PTMExosphereIntegration() {
    // Destructor defined here where ExosphereModel is complete
}

void PTMExosphereIntegration::setMesh(const std::shared_ptr<Mesh>& mesh) {
    mesh_ = mesh;

    if (!mesh_ || mesh_->faces.empty()) {
        std::cerr << "PTMExosphereIntegration::setMesh: invalid mesh (null or no faces)\n";
        return;
    }

    const size_t n_faces = mesh_->faces.size();
    face_areas_.assign(n_faces, 0.0);
    face_centers_.assign(n_faces, {0.0, 0.0, 0.0});
    face_normals_.assign(n_faces, {0.0, 0.0, 0.0});

    const bool have_vertices = !mesh_->vertices.empty();

    // If the mesh already provides face centers, copy and normalize them to Europa's radius
    if (mesh_->face_centers.size() == n_faces) {
        for (size_t i = 0; i < n_faces; ++i) {
            auto fc = mesh_->face_centers[i];
            const double r = std::sqrt(fc[0]*fc[0] + fc[1]*fc[1] + fc[2]*fc[2]);
            if (r > 0.0) {
                const double s = ptm::exosphere::constants::EUROPA_RADIUS / r;
                fc[0] *= s; fc[1] *= s; fc[2] *= s;
            }
            face_centers_[i] = fc;
        }
    }

    // Compute face areas (and centers if not provided) from vertex geometry when available
    if (have_vertices) {
        for (size_t i = 0; i < n_faces; ++i) {
            const auto& f = mesh_->faces[i];
            const auto& v0 = mesh_->vertices[f[0]];
            const auto& v1 = mesh_->vertices[f[1]];
            const auto& v2 = mesh_->vertices[f[2]];

            // Area via cross product
            const std::array<double,3> e1{ v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2] };
            const std::array<double,3> e2{ v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2] };
            const std::array<double,3> cr{
                e1[1]*e2[2] - e1[2]*e2[1],
                e1[2]*e2[0] - e1[0]*e2[2],
                e1[0]*e2[1] - e1[1]*e2[0]
            };
            face_areas_[i] = 0.5 * std::sqrt(cr[0]*cr[0] + cr[1]*cr[1] + cr[2]*cr[2]);

            // Normal (unnormalized cross) for optional diagnostics
            face_normals_[i] = cr;

            // If we didn’t copy provided centers above, compute them and normalize to sphere radius
            if (mesh_->face_centers.size() != n_faces) {
                std::array<double,3> c{
                    (v0[0] + v1[0] + v2[0]) / 3.0,
                    (v0[1] + v1[1] + v2[1]) / 3.0,
                    (v0[2] + v1[2] + v2[2]) / 3.0
                };
                const double r = std::sqrt(c[0]*c[0] + c[1]*c[1] + c[2]*c[2]);
                if (r > 0.0) {
                    const double s = ptm::exosphere::constants::EUROPA_RADIUS / r;
                    c[0] *= s; c[1] *= s; c[2] *= s;
                }
                face_centers_[i] = c;
            }
        }
    }

    // Fallback area if vertices are missing or degenerate
    if (!have_vertices) {
        const double total = 4.0 * M_PI *
            ptm::exosphere::constants::EUROPA_RADIUS * ptm::exosphere::constants::EUROPA_RADIUS;
        const double equal_area = total / static_cast<double>(n_faces);
        std::fill(face_areas_.begin(), face_areas_.end(), equal_area);

        // If centers are still all zero, use any provided mesh centers
        if (mesh_->face_centers.size() == n_faces) {
            for (size_t i = 0; i < n_faces; ++i) {
                face_centers_[i] = mesh_->face_centers[i];
            }
        }
    }

    // Initialize exosphere model and provide mesh geometry
    exosphere_model_ = std::make_unique<ExosphereModel>(n_faces);
    exosphere_model_->setMeshGeometry(face_centers_, face_areas_);

    // Cache sizes for getters
    num_faces_ = static_cast<int>(n_faces);
    matrix_built_ = false;
}

bool PTMExosphereIntegration::loadMesh(const std::string& mesh_file) {
        MeshUtils mesh_utils;

        if (mesh_file.empty()) {
            // Use default icosphere mesh
            mesh_utils.setMeshResolution(4); // Same as PTM default
            mesh_utils.setupMesh();
        } else {
            // Load custom mesh - check if file exists first
            if (!std::filesystem::exists(mesh_file)) {
                std::cerr << "Mesh file does not exist: " << mesh_file << std::endl;
                return false;
            }

            // Try to load mesh from file using MeshUtils
            if (!mesh_utils.loadMeshFromFile(mesh_file)) {
                std::cerr << "Failed to load mesh from file: " << mesh_file << std::endl;
                std::cerr << "Falling back to default mesh" << std::endl;
                mesh_utils.setMeshResolution(4);
                mesh_utils.setupMesh();
            }
        }

        mesh_ = mesh_utils.getMesh();
        
        if (!mesh_ || mesh_->faces.empty()) {
            std::cerr << "Invalid mesh!" << std::endl;
            return false;
        }
        
        std::cout << "Loaded mesh with " << mesh_->faces.size() << " faces" << std::endl;
        
        // Calculate face areas and centers from vertices and faces
        face_areas_.resize(mesh_->faces.size());
        face_centers_.resize(mesh_->faces.size());
        
        for (size_t i = 0; i < mesh_->faces.size(); ++i) {
            const auto& face = mesh_->faces[i];
            const auto& v0 = mesh_->vertices[face[0]];
            const auto& v1 = mesh_->vertices[face[1]];
            const auto& v2 = mesh_->vertices[face[2]];
            
            // Calculate face center (average of vertices)
            face_centers_[i][0] = (v0[0] + v1[0] + v2[0]) / 3.0;
            face_centers_[i][1] = (v0[1] + v1[1] + v2[1]) / 3.0;
            face_centers_[i][2] = (v0[2] + v1[2] + v2[2]) / 3.0;
            
            // Project center to sphere surface (for spherical bodies)
            double r = std::sqrt(face_centers_[i][0] * face_centers_[i][0] + 
                                face_centers_[i][1] * face_centers_[i][1] + 
                                face_centers_[i][2] * face_centers_[i][2]);
            if (r > 0) {
                face_centers_[i][0] *= constants::EUROPA_RADIUS / r;
                face_centers_[i][1] *= constants::EUROPA_RADIUS / r;
                face_centers_[i][2] *= constants::EUROPA_RADIUS / r;
            }
            
            // Calculate area using cross product
            std::array<double, 3> edge1 = {v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]};
            std::array<double, 3> edge2 = {v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]};
            
            std::array<double, 3> cross = {
                edge1[1] * edge2[2] - edge1[2] * edge2[1],
                edge1[2] * edge2[0] - edge1[0] * edge2[2],
                edge1[0] * edge2[1] - edge1[1] * edge2[0]
            };
            
            face_areas_[i] = 0.5 * std::sqrt(cross[0] * cross[0] + 
                                             cross[1] * cross[1] + 
                                             cross[2] * cross[2]);
        }
        
        // Initialize exosphere model
        size_t num_faces = mesh_->faces.size();
        exosphere_model_ = std::make_unique<ExosphereModel>(num_faces);

        // Set mesh geometry - use mesh's face_centers if available, otherwise use calculated
        if (!mesh_->face_centers.empty()) {
            exosphere_model_->setMeshGeometry(mesh_->face_centers, face_areas_);
        } else {
            exosphere_model_->setMeshGeometry(face_centers_, face_areas_);
        }

        // IMPORTANT: Set num_faces_ so sputtering and other methods work correctly
        num_faces_ = static_cast<int>(num_faces);
        num_timesteps_ = 0;  // Will be set when temperature data is loaded
        matrix_built_ = false;

        return true;
    }
    
    // Load temperature data from PTM binary output
    bool PTMExosphereIntegration::loadTemperatureData(const std::string& temperature_dir, int num_timesteps) {
        surface_temperatures_.clear();
        surface_temperatures_.resize(mesh_->faces.size());
        
        for (int t = 0; t < num_timesteps; ++t) {
            std::stringstream filename;
            filename << temperature_dir << "/temperature/temperature_" 
                    << std::setfill('0') << std::setw(4) << t << ".bin";  // Changed from setw(3) to setw(4)
            
            std::ifstream file(filename.str(), std::ios::binary);
            if (!file.is_open()) {
                std::cerr << "Failed to open temperature file: " << filename.str() << std::endl;
                return false;
            }
            
            // Read header (number of faces)
            uint32_t num_faces;
            file.read(reinterpret_cast<char*>(&num_faces), sizeof(uint32_t));
            
            if (num_faces != mesh_->faces.size()) {
                std::cerr << "Mesh size mismatch in temperature file!" << std::endl;
                return false;
            }
            
            // Read temperature data
            std::vector<float> temp_data(num_faces);
            file.read(reinterpret_cast<char*>(temp_data.data()), num_faces * sizeof(float));
            
            // Store in our array
            for (size_t f = 0; f < num_faces; ++f) {
                surface_temperatures_[f].push_back(static_cast<double>(temp_data[f]));
            }
            
            file.close();
        }
        
        std::cout << "Loaded temperature data for " << num_timesteps << " timesteps" << std::endl;
        return true;
    }
    
    // Run exosphere simulation for one timestep
    void PTMExosphereIntegration::simulateTimestep(int timestep_idx, double dt_seconds) {
        // Extract temperatures for this timestep
        if (show_progress_) {
            std::cerr << "\r[t=" << timestep_idx << "] copying temperatures...                " << std::flush;
        }
        std::vector<double> current_temps(mesh_->faces.size());
        for (size_t f = 0; f < mesh_->faces.size(); ++f) {
            current_temps[f] = surface_temperatures_[f][timestep_idx];
        }

        // Update exosphere model with surface temperatures directly
        // (new model uses surface temps without thermal insulation correction)
        if (show_progress_) {
            std::cerr << "\r[t=" << timestep_idx << "] updating physics (vapor/transport)... " << std::flush;
        }
        exosphere_model_->updateTemperatures(current_temps);

        // Evolve one timestep for sublimation/transport
        if (show_progress_) {
            std::cerr << "\r[t=" << timestep_idx << "] evolving mass state...                 " << std::flush;
        }
        exosphere_model_->evolveTimeStep(dt_seconds);

        // Get state for further processing
        auto& exo_state = exosphere_model_->getMutableState();

        // Apply ice layer physics if enabled (concentration-based model)
        if (ice_layer_enabled_ && ice_layer_model_ && ice_layer_state_) {
            if (show_progress_) {
                std::cerr << "\r[t=" << timestep_idx << "] applying ice layer physics...           " << std::flush;
            }

            // Get ice availability factor (0 if depleted, 1 if ice present)
            // This determines whether sublimation can occur
            std::vector<double> ice_availability(mesh_->faces.size());
            for (size_t i = 0; i < mesh_->faces.size(); ++i) {
                ice_availability[i] = ice_layer_model_->getIceAvailabilityFactor(*ice_layer_state_, i);
            }

            // Apply sublimation (increases non-ice concentration)
            // Actual sublimation is scaled by ice availability
            ice_layer_model_->applySublimation(*ice_layer_state_, exo_state.emission_flux, dt_seconds);

            // Apply deposition (fills void space, decreases non-ice concentration)
            ice_layer_model_->applyDeposition(*ice_layer_state_, exo_state.deposition_flux, dt_seconds);

            // Update ice mass for all faces after sublimation and deposition
            for (size_t i = 0; i < mesh_->faces.size(); ++i) {
                ice_layer_model_->updateIceMass(*ice_layer_state_, i);
            }

            // Sync ice mass back to exosphere state for consistency
            for (size_t i = 0; i < mesh_->faces.size(); ++i) {
                exo_state.accumulated_mass[i] = ice_layer_state_->ice_mass[i];
            }
            exo_state.updateIceThickness();
        }

        // Apply sputtering if enabled
        if (sputtering_enabled_ && sputtering_model_ && sputtering_state_) {
            if (show_progress_) {
                std::cerr << "\r[t=" << timestep_idx << "] applying sputtering...                  " << std::flush;
            }

            // Compute sputtering rates based on current temperatures
            sputtering_model_->computeSputteringRates(current_temps, face_areas_, *sputtering_state_);

            // Apply sputtering with ice layer model if enabled
            if (ice_layer_enabled_ && ice_layer_model_ && ice_layer_state_) {
                // Apply sputtering through ice layer model (uses same thermally active layer)
                ice_layer_model_->applySputtering(
                    *ice_layer_state_,
                    sputtering_state_->total_mass_loss,
                    sputtering_model_->getConfig().h2o_escape_fraction,
                    dt_seconds
                );

                // Update ice mass for all faces after sputtering
                for (size_t i = 0; i < mesh_->faces.size(); ++i) {
                    ice_layer_model_->updateIceMass(*ice_layer_state_, i);
                }

                // Sync back to exosphere state
                for (size_t i = 0; i < mesh_->faces.size(); ++i) {
                    exo_state.accumulated_mass[i] = ice_layer_state_->ice_mass[i];
                }
            } else {
                // Apply sputtering mass loss directly to accumulated ice
                sputtering_model_->applySputteringMassLoss(
                    *sputtering_state_,
                    exo_state.accumulated_mass,
                    face_areas_,
                    dt_seconds
                );
            }

            // Update ice thickness after sputtering
            exo_state.updateIceThickness();
        }

        // Evolve ice crystallinity if crystallization model is enabled
        if (crystallization_enabled_ && crystallization_model_ && ice_layer_state_) {
            if (show_progress_) {
                std::cerr << "\r[t=" << timestep_idx << "] evolving crystallinity...                 " << std::flush;
            }

            // Evolve crystallinity using current temperatures
            crystallization_model_->evolveAllFaces(
                dt_seconds, current_temps, ice_layer_state_->crystalline_fraction);

            // Apply depositional amorphization: newly deposited ice is amorphous
            for (size_t i = 0; i < mesh_->faces.size(); ++i) {
                double deposited = exo_state.deposition_flux[i] * dt_seconds;
                if (deposited > 0.0 && ice_layer_state_->ice_mass[i] > 0.0) {
                    IceCrystallizationModel::applyDepositionalAmorphization(
                        deposited, ice_layer_state_->ice_mass[i],
                        ice_layer_state_->crystalline_fraction[i]);
                }
            }

            // Accumulate radiation dose
            for (size_t i = 0; i < mesh_->faces.size(); ++i) {
                ice_layer_state_->cumulative_dose[i] +=
                    ice_layer_state_->radiation_dose_rate[i] * dt_seconds;
            }
        }

        if (show_progress_) {
            std::cerr << "\r[t=" << timestep_idx << "] timestep complete.                      " << std::flush;
        }
    }
    
    // Run complete simulation
    void PTMExosphereIntegration::runSimulation(double dt_seconds, bool export_all_steps) {
        size_t num_timesteps = surface_temperatures_.empty() ? 0 : surface_temperatures_[0].size();
        if (num_timesteps == 0) {
            std::cerr << "No temperature timesteps loaded; nothing to simulate.\n";
            return;
        }

        std::cout << "Running exosphere simulation for " << num_timesteps << " timesteps..." << std::endl;

        // Build time values vector for PVD file
        std::vector<double> time_values(num_timesteps);
        for (size_t t = 0; t < num_timesteps; ++t) {
            time_values[t] = t * dt_seconds;
        }

        const int bar_width = 40;
        auto t0 = std::chrono::steady_clock::now();

        for (size_t t = 0; t < num_timesteps; ++t) {
            // Print a per-step header immediately so users see activity
            if (show_progress_) {
                std::cerr << "\rStep " << (t + 1) << "/" << num_timesteps << " starting...                     " << std::flush;
            }

            auto s0 = std::chrono::steady_clock::now();
            simulateTimestep(static_cast<int>(t), dt_seconds);
            auto s1 = std::chrono::steady_clock::now();

            if (export_all_steps || t == num_timesteps - 1) {
                if (show_progress_) {
                    std::cerr << "\rStep " << (t + 1) << "/" << num_timesteps << " exporting...                   " << std::flush;
                }
                // Export VTU file (for ParaView)
                exportTimestepVTU(static_cast<int>(t), time_values[t]);
                // Also export CSV (for backward compatibility)
                exportTimestep(static_cast<int>(t));
            }
            if (t % 10 == 0) {
                if (show_progress_) {
                    std::cerr << "\rStep " << (t + 1) << "/" << num_timesteps << " analyzing...                   " << std::flush;
                }
                analyzeMassBalance(static_cast<int>(t));
            }

            // Loop-level progress bar and ETA
            if (show_progress_) {
                double prog = static_cast<double>(t + 1) / static_cast<double>(num_timesteps);
                auto now = std::chrono::steady_clock::now();
                const double elapsed_s_total = std::chrono::duration_cast<std::chrono::seconds>(now - t0).count();
                const double step_s = std::chrono::duration_cast<std::chrono::seconds>(s1 - s0).count();
                const double eta_s = (prog > 0.0) ? elapsed_s_total * (1.0 - prog) / prog : 0.0;

                std::cerr << "\r[";
                int pos = static_cast<int>(prog * bar_width);
                for (int i = 0; i < bar_width; ++i) {
                    if (i < pos) std::cerr << '=';
                    else if (i == pos) std::cerr << '>';
                    else std::cerr << ' ';
                }
                std::cerr << "] " << std::fixed << std::setprecision(1) << (prog * 100.0) << "% "
                          << "step " << (t + 1) << "/" << num_timesteps
                          << " (" << static_cast<long>(step_s) << "s), "
                          << "ETA: " << static_cast<long>(eta_s) << "s" << std::flush;
            }
        }
        if (show_progress_) std::cerr << std::endl;

        std::cout << "Simulation complete!" << std::endl;

        // Create PVD file to tie together all VTU timesteps
        if (export_all_steps) {
            createPVDFile(static_cast<int>(num_timesteps), time_values);
        }

        exportFinalAnalysis();
    }
    
    // Analyze mass balance
    void PTMExosphereIntegration::analyzeMassBalance(int timestep_idx) {
        const auto& state = exosphere_model_->getState();

        mass_balance_.total_sublimation_rate = 0.0;
        mass_balance_.total_deposition_rate = 0.0;
        mass_balance_.polar_deposition_rate = 0.0;
        mass_balance_.equatorial_loss_rate = 0.0;
        double total_escape_rate = 0.0;

        for (size_t i = 0; i < mesh_->faces.size(); ++i) {
            double area = face_areas_[i];  // Use our calculated face areas

            // Total emission (sublimation) and deposition rates
            // These should be nearly equal since the transport matrix conserves mass
            mass_balance_.total_sublimation_rate += state.emission_flux[i] * area;
            mass_balance_.total_deposition_rate += state.deposition_flux[i] * area;
            total_escape_rate += state.escape_flux[i] * area;

            // Latitude-dependent analysis
            double lat = 0.0;
            if (!mesh_->latitudes.empty() && i < mesh_->latitudes.size()) {
                lat = mesh_->latitudes[i];
            } else {
                // Calculate latitude from face center if not available
                double z = face_centers_[i][2];  // assuming z is up
                double r = std::sqrt(face_centers_[i][0] * face_centers_[i][0] +
                                   face_centers_[i][1] * face_centers_[i][1] +
                                   face_centers_[i][2] * face_centers_[i][2]);
                if (r > 0) {
                    lat = std::asin(z / r) * 180.0 / M_PI;  // Convert to degrees
                }
            }

            // Net flux = deposition - emission (positive = net gain, negative = net loss)
            double net_flux = state.net_flux[i];

            if (std::abs(lat) > 70.0) {  // Polar regions
                if (net_flux > 0) {
                    mass_balance_.polar_deposition_rate += net_flux * area;
                }
            } else if (std::abs(lat) < 30.0) {  // Equatorial regions
                if (net_flux < 0) {
                    mass_balance_.equatorial_loss_rate += std::abs(net_flux) * area;
                }
            }
        }

        // Net loss = escape only (since transport conserves mass internally)
        // The difference between sublimation and deposition should be ~escape
        mass_balance_.net_loss_rate = total_escape_rate;
    }
    
    // Export timestep data
    void PTMExosphereIntegration::exportTimestep(int timestep_idx) {
        std::stringstream filename;
        filename << output_dir_ << "/exosphere_t" 
                << std::setfill('0') << std::setw(3) << timestep_idx << ".csv";
        
        exosphere_model_->exportState(filename.str());
    }
    
    // Export final analysis
    void PTMExosphereIntegration::exportFinalAnalysis() {
        std::string filename = output_dir_ + "/mass_balance_analysis.txt";
        std::ofstream file(filename);

        if (!file.is_open()) {
            std::cerr << "Failed to open analysis file!" << std::endl;
            return;
        }

        file << "Europa Exosphere Mass Balance Analysis\n";
        file << "=======================================\n\n";

        file << "Global Mass Flux Rates:\n";
        file << "  Total sublimation rate: " << std::scientific << std::setprecision(3)
             << mass_balance_.total_sublimation_rate << " kg/s\n";
        file << "  Total deposition rate:  " << mass_balance_.total_deposition_rate << " kg/s\n";
        file << "  Atmospheric escape rate: " << mass_balance_.net_loss_rate << " kg/s\n";

        // Mass balance check: sublimation should equal deposition + escape
        double balance_error = mass_balance_.total_sublimation_rate -
                               (mass_balance_.total_deposition_rate + mass_balance_.net_loss_rate);
        double balance_error_pct = 100.0 * balance_error / mass_balance_.total_sublimation_rate;
        file << "\nMass Balance Check:\n";
        file << "  Sublimation - (Deposition + Escape) = " << balance_error << " kg/s\n";
        file << "  Relative error: " << std::fixed << std::setprecision(4) << balance_error_pct << "%\n";

        file << "\nLatitude-dependent Redistribution:\n";
        file << "  Polar (|lat|>70°) net deposition: " << std::scientific << std::setprecision(3)
             << mass_balance_.polar_deposition_rate << " kg/s\n";
        file << "  Equatorial (|lat|<30°) net loss:  " << mass_balance_.equatorial_loss_rate << " kg/s\n";

        // Calculate global water loss timescale based on escape only
        double europa_surface_area = 4.0 * M_PI * constants::EUROPA_RADIUS * constants::EUROPA_RADIUS;
        double ice_thickness = 1.0; // 1 meter of ice
        double ice_density = 917.0; // kg/m^3
        double total_ice_mass = europa_surface_area * ice_thickness * ice_density;

        file << "\nIce Loss Timescale (escape only):\n";
        if (mass_balance_.net_loss_rate > 0) {
            double loss_timescale_years = total_ice_mass / (mass_balance_.net_loss_rate * 365.25 * 86400);
            file << "  1m global ice layer: " << std::scientific << std::setprecision(3)
                 << loss_timescale_years << " years\n";
        } else {
            file << "  Negligible escape (timescale effectively infinite)\n";
        }

        file.close();
        std::cout << "Analysis exported to: " << filename << std::endl;
    }
    
    // Export transport matrix for visualization
    void PTMExosphereIntegration::exportTransportMatrix(const std::string& filename) {
        // Get transport matrix from model (would need to expose this in ExosphereModel)
        // For now, create a simple connectivity matrix
        
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open transport matrix file!" << std::endl;
            return;
        }
        
        file << "# Transport matrix (fraction of mass from face i to face j)\n";
        file << "# Format: i, j, transport_fraction\n";
        
        // This would output the actual transport matrix
        // For demonstration, showing structure
        size_t num_faces = mesh_->faces.size();
        for (size_t i = 0; i < num_faces; i += 100) {  // Sample every 100th face
            for (size_t j = 0; j < num_faces; j += 100) {
                // Calculate approximate transport
                double distance = exosphere_model_->getState().thermal_velocity[i] > 0 ? 
                    std::sqrt(std::pow(mesh_->face_centers[i][0] - mesh_->face_centers[j][0], 2) +
                             std::pow(mesh_->face_centers[i][1] - mesh_->face_centers[j][1], 2) +
                             std::pow(mesh_->face_centers[i][2] - mesh_->face_centers[j][2], 2)) : 0.0;
                
                if (distance > 0 && distance < constants::EUROPA_RADIUS * 0.5) {
                    double transport = std::exp(-distance / (constants::EUROPA_RADIUS * 0.1));
                    if (transport > 0.01) {
                        file << i << ", " << j << ", " << transport << "\n";
                    }
                }
            }
        }
        
        file.close();
        std::cout << "Transport matrix exported to: " << filename << std::endl;
    }

    // Export timestep data as binary files (matching ptm_thermal_sim format)
    // Use Python vtk_converter_exosphere.py to convert to VTU
    // Field names include units for clarity in ParaView
    void PTMExosphereIntegration::exportTimestepVTU(int timestep_idx, double time_value) {
        const auto& state = exosphere_model_->getState();
        size_t num_faces = mesh_->faces.size();

        // Helper lambda to write binary data file (same format as ptm_thermal_sim)
        auto writeBinaryArray = [&](const std::string& name, const std::vector<double>& data) {
            std::string dir = output_dir_ + "/" + name;
            std::filesystem::create_directories(dir);

            std::stringstream filename;
            filename << dir << "/" << name << "_" << std::setfill('0') << std::setw(4) << timestep_idx << ".bin";

            std::ofstream file(filename.str(), std::ios::binary);
            if (!file.is_open()) {
                std::cerr << "Failed to open binary file: " << filename.str() << std::endl;
                return;
            }

            // Header: uint32 face count
            uint32_t count = static_cast<uint32_t>(data.size());
            file.write(reinterpret_cast<const char*>(&count), sizeof(count));

            // Data: float32 values
            std::vector<float> float_data(data.size());
            for (size_t i = 0; i < data.size(); ++i) {
                float_data[i] = static_cast<float>(data[i]);
            }
            file.write(reinterpret_cast<const char*>(float_data.data()), float_data.size() * sizeof(float));

            file.close();
        };

        // Temperature handling depends on export mode
        if (multiscale_export_mode_ && multiscale_evolution_) {
            // Multiscale mode: export min/max/mean temperatures from diurnal averages
            const auto& diurnal_avg = multiscale_evolution_->getDiurnalAverages();
            writeBinaryArray("temperature_mean_K", diurnal_avg.avg_temperature);
            writeBinaryArray("temperature_min_K", diurnal_avg.min_temperature);
            writeBinaryArray("temperature_max_K", diurnal_avg.max_temperature);
        } else {
            // Diurnal mode: export instantaneous temperature
            std::vector<double> temperatures(num_faces);
            if (!surface_temperatures_.empty() &&
                surface_temperatures_.size() == num_faces &&
                !surface_temperatures_[0].empty() &&
                timestep_idx < static_cast<int>(surface_temperatures_[0].size())) {
                for (size_t f = 0; f < num_faces; ++f) {
                    temperatures[f] = surface_temperatures_[f][timestep_idx];
                }
            }
            writeBinaryArray("temperature_K", temperatures);
        }

        // Write flux data with units in names
        writeBinaryArray("vapor_pressure_Pa", state.vapor_pressure);
        writeBinaryArray("emission_flux_kg_m2_s", state.emission_flux);
        writeBinaryArray("deposition_flux_kg_m2_s", state.deposition_flux);
        writeBinaryArray("net_flux_kg_m2_s", state.net_flux);
        writeBinaryArray("accumulated_mass_kg_m2", state.accumulated_mass);
        writeBinaryArray("ice_thickness_m", state.ice_thickness);
        writeBinaryArray("escape_flux_kg_m2_s", state.escape_flux);

        // Write sputtering data if enabled
        if (sputtering_enabled_ && sputtering_state_) {
            writeBinaryArray("sputtering_flux_kg_m2_s", sputtering_state_->h2o_sputter_flux);
            writeBinaryArray("sputtering_yield", sputtering_state_->sputtering_yield);
            writeBinaryArray("sputtering_mass_loss_kg_m2_s", sputtering_state_->total_mass_loss);

            // Also export spatial weight (ion flux distribution) on first timestep
            if (timestep_idx == 0 && sputtering_model_) {
                std::vector<double> spatial_weights(num_faces);
                for (size_t i = 0; i < num_faces; ++i) {
                    spatial_weights[i] = sputtering_model_->getSpatialWeight(i);
                }
                writeBinaryArray("sputtering_spatial_weight", spatial_weights);
            }
        }

        // Write ice layer data if enabled (concentration-based model)
        if (ice_layer_enabled_ && ice_layer_state_) {
            // Primary output: non-ice concentration [dimensionless, 0-1 where 1 = no ice]
            writeBinaryArray("nonice_concentration", ice_layer_state_->nonice_concentration);
            // Ice mass per unit area [kg/m²]
            writeBinaryArray("ice_mass_kg_m2", ice_layer_state_->ice_mass);
            // Cumulative fluxes [kg/m²]
            writeBinaryArray("cumulative_sublimated_kg_m2", ice_layer_state_->cumulative_sublimated);
            writeBinaryArray("cumulative_sputtered_kg_m2", ice_layer_state_->cumulative_sputtered);
            writeBinaryArray("cumulative_deposited_kg_m2", ice_layer_state_->cumulative_deposited);
        }

        // Write crystallization data if enabled
        if (crystallization_enabled_ && crystallization_model_ && ice_layer_state_) {
            writeBinaryArray("crystalline_fraction", ice_layer_state_->crystalline_fraction);
            writeBinaryArray("porosity", ice_layer_state_->porosity);
            writeBinaryArray("radiation_dose_rate_eV_mol_s", ice_layer_state_->radiation_dose_rate);
            writeBinaryArray("cumulative_dose_eV_mol", ice_layer_state_->cumulative_dose);

            // Compute and export crystallization timescale [years] for visualization
            // Uses diurnal-averaged temperature if available, otherwise instantaneous
            std::vector<double> cryst_timescale_yr(num_faces);
            const double* temps = nullptr;
            std::vector<double> temp_buf;
            if (multiscale_export_mode_ && multiscale_evolution_) {
                temps = multiscale_evolution_->getDiurnalAverages().avg_temperature.data();
            } else if (!surface_temperatures_.empty() &&
                       timestep_idx < static_cast<int>(surface_temperatures_[0].size())) {
                temp_buf.resize(num_faces);
                for (size_t f = 0; f < num_faces; ++f) {
                    temp_buf[f] = surface_temperatures_[f][timestep_idx];
                }
                temps = temp_buf.data();
            }
            if (temps) {
                for (size_t i = 0; i < num_faces; ++i) {
                    cryst_timescale_yr[i] = crystallization_model_->crystallizationTimescale_yr(
                        temps[i], i);
                }
            }
            writeBinaryArray("crystallization_timescale_yr", cryst_timescale_yr);
        }

        // Export atmospheric mass summary
        exportAtmosphericMass(timestep_idx, time_value);

        // On first timestep, also export the mesh as PLY (for Python converter)
        if (timestep_idx == 0) {
            exportMeshPLY();
        }
    }

    // Export mesh in binary PLY format (matching ptm_thermal_sim format)
    void PTMExosphereIntegration::exportMeshPLY() {
        std::string filename = output_dir_ + "/mesh.ply";
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Failed to open mesh file: " << filename << std::endl;
            return;
        }

        size_t num_vertices = mesh_->vertices.size();
        size_t num_faces = mesh_->faces.size();

        // Write PLY header
        file << "ply\n";
        file << "format binary_little_endian 1.0\n";
        file << "element vertex " << num_vertices << "\n";
        file << "property double x\n";
        file << "property double y\n";
        file << "property double z\n";
        file << "element face " << num_faces << "\n";
        file << "property list uchar int vertex_indices\n";
        file << "property float latitude\n";
        file << "property float longitude\n";
        file << "end_header\n";

        // Write vertices (double precision)
        for (const auto& v : mesh_->vertices) {
            double x = v[0], y = v[1], z = v[2];
            file.write(reinterpret_cast<const char*>(&x), sizeof(double));
            file.write(reinterpret_cast<const char*>(&y), sizeof(double));
            file.write(reinterpret_cast<const char*>(&z), sizeof(double));
        }

        // Write faces with lat/lon
        for (size_t i = 0; i < num_faces; ++i) {
            const auto& f = mesh_->faces[i];
            unsigned char count = 3;
            file.write(reinterpret_cast<const char*>(&count), sizeof(count));

            int idx0 = static_cast<int>(f[0]);
            int idx1 = static_cast<int>(f[1]);
            int idx2 = static_cast<int>(f[2]);
            file.write(reinterpret_cast<const char*>(&idx0), sizeof(int));
            file.write(reinterpret_cast<const char*>(&idx1), sizeof(int));
            file.write(reinterpret_cast<const char*>(&idx2), sizeof(int));

            // Calculate lat/lon from face center
            float lat = 0.0f, lon = 0.0f;
            if (i < face_centers_.size()) {
                double x = face_centers_[i][0];
                double y = face_centers_[i][1];
                double z = face_centers_[i][2];
                double r = std::sqrt(x*x + y*y + z*z);
                if (r > 0) {
                    lat = static_cast<float>(std::asin(z / r) * 180.0 / M_PI);
                    lon = static_cast<float>(std::atan2(y, x) * 180.0 / M_PI);
                }
            }
            file.write(reinterpret_cast<const char*>(&lat), sizeof(float));
            file.write(reinterpret_cast<const char*>(&lon), sizeof(float));
        }

        file.close();
        std::cout << "Exported mesh to: " << filename << std::endl;
    }

    // Create time_steps.txt and metadata.txt for Python VTK converter
    void PTMExosphereIntegration::createPVDFile(int num_timesteps, const std::vector<double>& time_values) {
        // Write time_steps.txt (matching ptm_thermal_sim format)
        std::string time_steps_file = output_dir_ + "/time_steps.txt";
        std::ofstream ts_file(time_steps_file);
        if (ts_file.is_open()) {
            ts_file << "# Time steps metadata for exosphere simulation\n";
            ts_file << "# Column 1: Time elapsed from start (seconds)\n";
            for (int t = 0; t < num_timesteps; ++t) {
                double time_val = (t < static_cast<int>(time_values.size())) ? time_values[t] : t * 3600.0;
                ts_file << std::fixed << std::setprecision(1) << time_val << "\n";
            }
            ts_file.close();
            std::cout << "Created time_steps.txt: " << time_steps_file << std::endl;
        }

        // Write metadata.txt (for Python converter)
        std::string metadata_file = output_dir_ + "/metadata.txt";
        std::ofstream md_file(metadata_file);
        if (md_file.is_open()) {
            md_file << "faces=" << num_faces_ << "\n";
            md_file << "time_steps=" << num_timesteps << "\n";
            md_file << "components=temperature,vapor_pressure,emission_flux,deposition_flux,net_flux,accumulated_mass,ice_thickness,escape_flux\n";
            md_file.close();
            std::cout << "Created metadata.txt: " << metadata_file << std::endl;
        }

        std::cout << "\nTo convert binary data to VTU format, run:\n";
        std::cout << "  python src/python/vtk_converter_exosphere.py --input-dir " << output_dir_ << " --output-dir " << output_dir_ << "\n";
    }

    // Export atmospheric mass summary (column densities, vapor pressures for H2O, O2, H2)
    void PTMExosphereIntegration::exportAtmosphericMass(int timestep_idx, double time_value) {
        const auto& state = exosphere_model_->getState();
        size_t num_faces = mesh_->faces.size();

        // Calculate global mean values
        double total_area = 0.0;
        double weighted_h2o_column_mass = 0.0;  // kg/m² (same as vapor density integrated over scale height)
        double weighted_h2o_vapor_pressure = 0.0;  // Pa

        for (size_t i = 0; i < num_faces; ++i) {
            double area = face_areas_[i];
            total_area += area;

            // H2O column mass: vapor density * scale height
            // For a Maxwellian atmosphere: column = vapor_density * kT / (m*g)
            // But since emission_flux = vapor_density * v_th / 4, we can estimate
            // For now, use vapor_density directly as a proxy (it's in kg/m³)
            // Column mass = vapor_density * H where H = kT/(mg) is scale height
            // For H2O at 100K: H ≈ kT/(m*g) = 1.38e-23 * 100 / (2.99e-26 * 1.314) ≈ 35 km
            double T = 100.0;  // Use nominal temp for scale height estimate
            if (multiscale_export_mode_ && multiscale_evolution_) {
                const auto& diurnal_avg = multiscale_evolution_->getDiurnalAverages();
                if (i < diurnal_avg.avg_temperature.size()) {
                    T = std::max(50.0, diurnal_avg.avg_temperature[i]);
                }
            }
            double scale_height = constants::BOLTZMANN_K * T /
                                  (constants::WATER_MOLECULAR_MASS * constants::EUROPA_GRAVITY);
            double vapor_density = state.vapor_pressure[i] * constants::WATER_MOLAR_MASS /
                                   (constants::GAS_CONSTANT_R * T);
            double column_mass = vapor_density * scale_height;  // kg/m²

            weighted_h2o_column_mass += column_mass * area;
            weighted_h2o_vapor_pressure += state.vapor_pressure[i] * area;
        }

        // Compute global means
        double mean_h2o_column_mass = (total_area > 0) ? weighted_h2o_column_mass / total_area : 0.0;
        double mean_h2o_vapor_pressure = (total_area > 0) ? weighted_h2o_vapor_pressure / total_area : 0.0;

        // For O2 and H2, use approximate ratios based on photodissociation/radiolysis products
        // Typical ratios from literature: O2/H2O ~ 10-100, H2/H2O ~ 1-10
        // These are rough approximations; exact values depend on many factors
        double o2_production_ratio = 0.5;  // O2 produced per H2O lost (rough estimate)
        double h2_production_ratio = 1.0;  // H2 produced per H2O lost (rough estimate)

        // O2 and H2 have longer lifetimes, so they accumulate
        // For a rough estimate, scale by molecular mass ratio for vapor pressure
        double mean_o2_vapor_pressure = mean_h2o_vapor_pressure * o2_production_ratio *
                                        (constants::WATER_MOLAR_MASS / 0.032);  // O2 = 32 g/mol
        double mean_h2_vapor_pressure = mean_h2o_vapor_pressure * h2_production_ratio *
                                        (constants::WATER_MOLAR_MASS / 0.002);  // H2 = 2 g/mol

        // Create output directory
        std::string atm_dir = output_dir_ + "/atmosphere";
        std::filesystem::create_directories(atm_dir);

        // Append to atmospheric mass CSV file
        std::string csv_file = atm_dir + "/atmospheric_mass.csv";
        bool write_header = !std::filesystem::exists(csv_file);

        std::ofstream file(csv_file, std::ios::app);
        if (!file.is_open()) {
            std::cerr << "Failed to open atmospheric mass file: " << csv_file << std::endl;
            return;
        }

        if (write_header) {
            file << "timestep,time_s,time_years,"
                 << "h2o_column_mass_kg_m2,h2o_vapor_pressure_Pa,"
                 << "o2_vapor_pressure_Pa,h2_vapor_pressure_Pa\n";
        }

        double time_years = time_value / timescale_constants::SECONDS_PER_YEAR;

        file << timestep_idx << ","
             << std::scientific << std::setprecision(6) << time_value << ","
             << std::scientific << std::setprecision(6) << time_years << ","
             << std::scientific << std::setprecision(6) << mean_h2o_column_mass << ","
             << std::scientific << std::setprecision(6) << mean_h2o_vapor_pressure << ","
             << std::scientific << std::setprecision(6) << mean_o2_vapor_pressure << ","
             << std::scientific << std::setprecision(6) << mean_h2_vapor_pressure << "\n";

        file.close();
    }

    // =========================================================================
    // Sputtering Model Interface Implementation
    // =========================================================================

    void PTMExosphereIntegration::enableSputtering(double rate_scale, double hemisphere_contrast) {
        if (num_faces_ == 0) {
            std::cerr << "PTMExosphereIntegration::enableSputtering: no faces loaded yet!" << std::endl;
            return;
        }

        // Check that face_centers_ is populated
        if (face_centers_.empty() || face_centers_.size() != num_faces_) {
            std::cerr << "PTMExosphereIntegration::enableSputtering: face_centers_ not initialized!" << std::endl;
            std::cerr << "  face_centers_.size() = " << face_centers_.size() << ", num_faces_ = " << num_faces_ << std::endl;
            return;
        }

        // Check that face_areas_ is populated
        if (face_areas_.empty() || face_areas_.size() != num_faces_) {
            std::cerr << "PTMExosphereIntegration::enableSputtering: face_areas_ not initialized!" << std::endl;
            std::cerr << "  face_areas_.size() = " << face_areas_.size() << ", num_faces_ = " << num_faces_ << std::endl;
            return;
        }

        // Create configuration
        SputteringConfig config;
        config.enabled = true;
        config.rate_scale = rate_scale;
        config.hemisphere_contrast = hemisphere_contrast;

        // Create sputtering model
        sputtering_model_ = std::make_unique<SputteringModel>(num_faces_, config);
        sputtering_state_ = std::make_unique<SputteringState>(num_faces_);

        // Compute latitudes and longitudes from face centers
        std::vector<double> latitudes(num_faces_);
        std::vector<double> longitudes(num_faces_);

        for (size_t i = 0; i < num_faces_; ++i) {
            const auto& pos = face_centers_[i];
            double r = std::sqrt(pos[0]*pos[0] + pos[1]*pos[1] + pos[2]*pos[2]);

            if (r > 0) {
                latitudes[i] = std::asin(pos[2] / r) * 180.0 / M_PI;
                longitudes[i] = std::atan2(pos[1], pos[0]) * 180.0 / M_PI;
            }
        }

        // Initialize spatial distribution
        std::cout << "Initializing sputtering spatial distribution..." << std::endl;
        std::cout << "  num_faces_ = " << num_faces_ << std::endl;
        std::cout << "  face_centers_.size() = " << face_centers_.size() << std::endl;
        std::cout << "  latitudes.size() = " << latitudes.size() << std::endl;
        std::cout << "  longitudes.size() = " << longitudes.size() << std::endl;

        sputtering_model_->initializeSpatialDistribution(face_centers_, latitudes, longitudes);

        // Only enable if initialization succeeded
        if (sputtering_model_->isSpatialInitialized()) {
            sputtering_enabled_ = true;
            sputtering_model_->printConfig();

            std::cout << "Sputtering model enabled with rate_scale=" << rate_scale
                      << ", hemisphere_contrast=" << hemisphere_contrast << std::endl;
        } else {
            sputtering_enabled_ = false;
            std::cerr << "ERROR: Sputtering spatial distribution failed to initialize!" << std::endl;
            std::cerr << "  Sputtering will be DISABLED for this run." << std::endl;
        }
    }

    void PTMExosphereIntegration::disableSputtering() {
        sputtering_enabled_ = false;
        if (sputtering_model_) {
            sputtering_model_->setEnabled(false);
        }
        std::cout << "Sputtering model disabled (sublimation-only mode)" << std::endl;
    }

    void PTMExosphereIntegration::setSputteringRateScale(double scale) {
        if (sputtering_model_) {
            sputtering_model_->setRateScale(scale);
            std::cout << "Sputtering rate scale set to " << scale << std::endl;
        }
    }

    double PTMExosphereIntegration::getSputteringEscapedMass() const {
        if (sputtering_model_) {
            return sputtering_model_->getTotalEscapedMass();
        }
        return 0.0;
    }

    double PTMExosphereIntegration::getSputteringFlux(int face_idx) const {
        if (sputtering_state_ && face_idx >= 0 && face_idx < static_cast<int>(num_faces_)) {
            return sputtering_state_->total_mass_loss[face_idx];
        }
        return 0.0;
    }

    // =========================================================================
    // Ice Layer Model Interface Implementation
    // =========================================================================

    void PTMExosphereIntegration::enableIceLayerModel(double initial_nonice_concentration) {
        if (num_faces_ == 0) {
            std::cerr << "PTMExosphereIntegration::enableIceLayerModel: no faces loaded yet!" << std::endl;
            return;
        }

        // Create configuration for concentration-based thermally active layer model
        IceLayerConfig config;
        config.initial_nonice_concentration = std::clamp(initial_nonice_concentration, 0.0, 1.0);
        // Use default thermal skin depth from PTM thermophysical properties
        config.thermal_skin_depth = layer_constants::DEFAULT_THERMAL_SKIN_DEPTH;

        // Create model and state
        ice_layer_model_ = std::make_unique<IceLayerModel>(num_faces_, config);
        ice_layer_state_ = std::make_unique<IceLayerState>(num_faces_);

        // Initialize with uniform concentration as specified by user
        // The thermal layer model assumes ice fills the thermal skin depth (5cm by default).
        // Using the exosphere model's thin surface ice (0.1-1mm) would give incorrect
        // initial concentrations since it's much thinner than the thermal layer.
        ice_layer_model_->initializeUniform(*ice_layer_state_, config.initial_nonice_concentration);

        // Sync ice mass back to exosphere state for consistency
        if (exosphere_model_) {
            auto& exo_state = exosphere_model_->getMutableState();
            for (size_t i = 0; i < num_faces_; ++i) {
                exo_state.accumulated_mass[i] = ice_layer_state_->ice_mass[i];
            }
            exo_state.updateIceThickness();
        }

        ice_layer_enabled_ = true;
        ice_layer_model_->printConfig();

        std::cout << "Ice layer model enabled with initial_nonice_concentration="
                  << (config.initial_nonice_concentration * 100.0) << "%" << std::endl;
    }

    void PTMExosphereIntegration::enableCrystallizationModel(const CrystallizationConfig& config) {
        if (num_faces_ == 0) {
            std::cerr << "PTMExosphereIntegration::enableCrystallizationModel: no faces loaded yet!" << std::endl;
            return;
        }
        if (!ice_layer_enabled_ || !ice_layer_state_) {
            std::cerr << "PTMExosphereIntegration::enableCrystallizationModel: "
                      << "ice layer model must be enabled first!" << std::endl;
            return;
        }

        // Create crystallization model
        crystallization_model_ = std::make_unique<IceCrystallizationModel>(num_faces_, config);

        // Compute latitudes and longitudes from face centers
        std::vector<double> latitudes(num_faces_);
        std::vector<double> longitudes(num_faces_);

        for (size_t i = 0; i < num_faces_; ++i) {
            const auto& pos = face_centers_[i];
            double r = std::sqrt(pos[0]*pos[0] + pos[1]*pos[1] + pos[2]*pos[2]);

            if (r > 0) {
                latitudes[i] = std::asin(pos[2] / r) * 180.0 / M_PI;
                longitudes[i] = std::atan2(pos[1], pos[0]) * 180.0 / M_PI;
                if (longitudes[i] < 0) longitudes[i] += 360.0;  // Convert to 0-360 east
            }
        }

        // Initialize porosity and radiation dose rate distributions
        crystallization_model_->initializePorosity(latitudes);
        crystallization_model_->initializeRadiationDoseRate(latitudes, longitudes);

        // Initialize crystalline fraction in ice layer state
        crystallization_model_->initializeCrystallineFraction(ice_layer_state_->crystalline_fraction);

        // Copy porosity and dose rate to ice layer state for export
        const auto& porosity = crystallization_model_->getPorosity();
        const auto& dose_rate = crystallization_model_->getRadiationDoseRate();
        for (size_t i = 0; i < num_faces_; ++i) {
            ice_layer_state_->porosity[i] = porosity[i];
            ice_layer_state_->radiation_dose_rate[i] = dose_rate[i];
        }

        crystallization_enabled_ = true;
        crystallization_model_->printConfig();

        std::cout << "Ice crystallization model enabled." << std::endl;
    }

    void PTMExosphereIntegration::disableIceLayerModel() {
        ice_layer_enabled_ = false;
        std::cout << "Ice layer model disabled (simple ice tracking)" << std::endl;
    }

    double PTMExosphereIntegration::getNoniceConcentration(int face_idx) const {
        if (ice_layer_state_ && face_idx >= 0 && face_idx < static_cast<int>(num_faces_)) {
            return ice_layer_state_->nonice_concentration[face_idx];
        }
        return 0.0;
    }

    double PTMExosphereIntegration::getIceMass(int face_idx) const {
        if (ice_layer_state_ && face_idx >= 0 && face_idx < static_cast<int>(num_faces_)) {
            return ice_layer_state_->ice_mass[face_idx];
        }
        return 0.0;
    }

    double PTMExosphereIntegration::getIceAvailabilityFactor(int face_idx) const {
        if (ice_layer_model_ && ice_layer_state_ && face_idx >= 0 && face_idx < static_cast<int>(num_faces_)) {
            return ice_layer_model_->getIceAvailabilityFactor(*ice_layer_state_, face_idx);
        }
        return 1.0;
    }

    double PTMExosphereIntegration::getCumulativeSublimated(int face_idx) const {
        if (ice_layer_state_ && face_idx >= 0 && face_idx < static_cast<int>(num_faces_)) {
            return ice_layer_state_->cumulative_sublimated[face_idx];
        }
        return 0.0;
    }

    double PTMExosphereIntegration::getCumulativeSputtered(int face_idx) const {
        if (ice_layer_state_ && face_idx >= 0 && face_idx < static_cast<int>(num_faces_)) {
            return ice_layer_state_->cumulative_sputtered[face_idx];
        }
        return 0.0;
    }

    double PTMExosphereIntegration::getCumulativeDeposited(int face_idx) const {
        if (ice_layer_state_ && face_idx >= 0 && face_idx < static_cast<int>(num_faces_)) {
            return ice_layer_state_->cumulative_deposited[face_idx];
        }
        return 0.0;
    }

    // =========================================================================
    // Multi-Timescale Evolution Implementation
    // =========================================================================

    void PTMExosphereIntegration::runMultiTimescaleSimulation(int num_epochs,
                                                               int cycles_per_epoch,
                                                               int export_interval,
                                                               bool analytical_acceleration,
                                                               int cache_refresh_interval) {
        // Validate prerequisites
        if (!ice_layer_enabled_ || !ice_layer_model_ || !ice_layer_state_) {
            std::cerr << "Multi-timescale simulation requires ice layer model. "
                      << "Call enableIceLayerModel() first." << std::endl;
            return;
        }

        size_t num_timesteps = surface_temperatures_.empty() ? 0 : surface_temperatures_[0].size();
        if (num_timesteps == 0) {
            std::cerr << "No temperature data loaded for multi-timescale simulation." << std::endl;
            return;
        }

        // Set export mode flag for temperature output format
        multiscale_export_mode_ = true;

        // Clear any existing atmospheric mass CSV (will be recreated fresh)
        std::string atm_csv = output_dir_ + "/atmosphere/atmospheric_mass.csv";
        if (std::filesystem::exists(atm_csv)) {
            std::filesystem::remove(atm_csv);
        }

        // Create multi-timescale evolution controller
        MultiTimescaleConfig config;
        config.num_epochs = num_epochs;
        config.cycles_per_epoch = cycles_per_epoch;

        // Adaptive export interval: if user specified 0 or default 100, compute adaptive interval
        // to produce ~10-20 output files regardless of epoch count
        int effective_export_interval = export_interval;
        if (export_interval <= 0) {
            // User specified 0 = final only
            effective_export_interval = 0;
        } else if (export_interval == 100 && num_epochs <= 200) {
            // Default interval with small epoch count - make adaptive
            // Target: ~10-20 output files
            const int target_exports = 10;
            effective_export_interval = std::max(1, num_epochs / target_exports);
        } else {
            effective_export_interval = export_interval;
        }
        config.export_interval = effective_export_interval;
        config.show_progress = show_progress_;

        // Note: analytical_acceleration and cache_refresh_interval parameters are
        // kept for API compatibility but no longer used in the new concentration model

        multiscale_evolution_ = std::make_unique<MultiTimescaleEvolution>(num_faces_, config);
        multiscale_enabled_ = true;

        config.print();

        // Calculate epoch duration
        double diurnal_period = timescale_constants::EUROPA_DIURNAL_PERIOD;
        double epoch_duration = cycles_per_epoch * diurnal_period;
        double dt_seconds = diurnal_period / num_timesteps;

        std::cout << "\nStarting multi-timescale simulation:" << std::endl;
        std::cout << "  Total simulation time: " << std::scientific << std::setprecision(2)
                  << config.totalTimeYears() << " years" << std::endl;
        std::cout << "  Diurnal timesteps: " << num_timesteps << std::endl;
        std::cout << "  Timestep: " << dt_seconds << " seconds" << std::endl;

        auto t0 = std::chrono::steady_clock::now();

        // Track cache statistics
        int full_diurnal_count = 0;
        int cached_epoch_count = 0;

        // Main epoch loop with flux caching optimization
        for (int epoch = 0; epoch < num_epochs; ++epoch) {
            // Step 1: Either compute full diurnal cycle or use cached fluxes
            bool need_full_diurnal = multiscale_evolution_->needsCacheRefresh(
                *ice_layer_model_, *ice_layer_state_, epoch);

            if (need_full_diurnal) {
                // Run full diurnal cycle (expensive, ~10 minutes)
                if (!computeDiurnalAverages(dt_seconds)) {
                    std::cerr << "Failed to compute diurnal averages at epoch " << epoch << std::endl;
                    break;
                }
                // Cache the computed base fluxes for future epochs
                multiscale_evolution_->cacheBaseFluxes(*ice_layer_model_, *ice_layer_state_, epoch);
                full_diurnal_count++;

                // Note: Cache refresh is silent now - info is printed at the end
            } else {
                // Use cached fluxes with ice availability scaling (fast, milliseconds)
                // Scale base fluxes by ice availability factor
                multiscale_evolution_->applyScaledFluxes(*ice_layer_model_, *ice_layer_state_);
                cached_epoch_count++;
            }

            // Step 2: Evolve concentration using (possibly scaled) averaged fluxes
            evolveEpoch(epoch, epoch_duration);

            // Step 3: Update statistics
            multiscale_evolution_->updateStatistics(*ice_layer_model_, *ice_layer_state_, epoch_duration);

            // Progress reporting
            if (show_progress_ && (epoch % timescale_constants::PROGRESS_INTERVAL == 0 || epoch == num_epochs - 1)) {
                multiscale_evolution_->printProgress();
            }

            // Export at intervals (use effective_export_interval, not original parameter)
            if (effective_export_interval > 0 && (epoch % effective_export_interval == 0 || epoch == num_epochs - 1)) {
                // Sync ice layer state and averaged fluxes to exosphere model before export
                if (exosphere_model_ && multiscale_evolution_) {
                    auto& exo_state = exosphere_model_->getMutableState();
                    const auto& diurnal_avg = multiscale_evolution_->getDiurnalAverages();

                    for (size_t i = 0; i < num_faces_; ++i) {
                        // Sync ice mass from ice layer state
                        if (ice_layer_enabled_ && ice_layer_state_) {
                            exo_state.accumulated_mass[i] = ice_layer_state_->ice_mass[i];
                        }

                        // Sync averaged fluxes from multi-timescale evolution
                        // These are the diurnal-averaged values scaled by ice availability
                        exo_state.emission_flux[i] = diurnal_avg.avg_sublimation_flux[i];
                        exo_state.deposition_flux[i] = diurnal_avg.avg_deposition_flux[i];
                        exo_state.net_flux[i] = diurnal_avg.avg_net_flux[i];
                        exo_state.escape_flux[i] = diurnal_avg.avg_escape_flux[i];
                    }
                    exo_state.updateIceThickness();
                }

                // Export VTU with epoch index
                double time_value = multiscale_evolution_->getStatistics().elapsed_time_years *
                                   timescale_constants::SECONDS_PER_YEAR;
                exportTimestepVTU(epoch, time_value);
            }

            // Check convergence
            if (multiscale_evolution_->getStatistics().converged) {
                std::cout << "\nConverged at epoch " << epoch << std::endl;
                break;
            }
        }

        if (show_progress_) std::cerr << std::endl;

        auto t1 = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration_cast<std::chrono::seconds>(t1 - t0).count();

        std::cout << "\nMulti-timescale simulation complete!" << std::endl;
        std::cout << "  Wall-clock time: " << elapsed << " seconds ("
                  << std::fixed << std::setprecision(1) << (elapsed / 60.0) << " minutes)" << std::endl;
        std::cout << "  Cache statistics:" << std::endl;
        std::cout << "    Full diurnal computations: " << full_diurnal_count << std::endl;
        std::cout << "    Cached (scaled) epochs: " << cached_epoch_count << std::endl;
        if (full_diurnal_count > 0) {
            double speedup = static_cast<double>(full_diurnal_count + cached_epoch_count) / full_diurnal_count;
            std::cout << "    Effective speedup: " << std::setprecision(1) << speedup << "x" << std::endl;
        }

        // Print final statistics
        multiscale_evolution_->getStatistics().print();

        // Export summary
        exportEvolutionSummary();

        // Create PVD file for exported epochs (use effective_export_interval)
        if (effective_export_interval > 0) {
            int num_exports = (num_epochs + effective_export_interval - 1) / effective_export_interval;
            std::vector<double> time_values;
            for (int i = 0; i < num_exports; ++i) {
                int ep = i * effective_export_interval;
                if (ep >= num_epochs) ep = num_epochs - 1;
                double years = (ep + 1) * epoch_duration / timescale_constants::SECONDS_PER_YEAR;
                time_values.push_back(years * timescale_constants::SECONDS_PER_YEAR);
            }
            createPVDFile(num_exports, time_values);
        }

        // Reset export mode flag
        multiscale_export_mode_ = false;
    }

    bool PTMExosphereIntegration::computeDiurnalAverages(double dt_seconds) {
        if (!multiscale_evolution_) return false;

        size_t num_timesteps = surface_temperatures_[0].size();

        // Reset accumulator
        multiscale_evolution_->resetDiurnalAccumulator();

        // Skip transport matrix rebuilds during diurnal averaging - we only need fluxes
        // The transport matrix for deposition will use the pre-built matrix
        exosphere_model_->setSkipMatrixRebuild(true);
        exosphere_model_->setVerbose(false);

        // Run through one diurnal cycle, accumulating fluxes
        for (size_t t = 0; t < num_timesteps; ++t) {
            // Extract temperatures for this timestep
            std::vector<double> current_temps(mesh_->faces.size());
            for (size_t f = 0; f < mesh_->faces.size(); ++f) {
                current_temps[f] = surface_temperatures_[f][t];
            }

            // Use surface temperatures directly (new model has no thermal insulation)
            exosphere_model_->updateTemperatures(current_temps);

            // Get current fluxes (without evolving mass - we just need the rates)
            const auto& state = exosphere_model_->getState();

            // Get sputtering fluxes if enabled
            std::vector<double> sputtering_flux;
            if (sputtering_enabled_ && sputtering_model_ && sputtering_state_) {
                sputtering_model_->computeSputteringRates(current_temps, face_areas_, *sputtering_state_);
                sputtering_flux = sputtering_state_->total_mass_loss;
            }

            // Accumulate into diurnal averages
            multiscale_evolution_->accumulateDiurnalStep(
                state.emission_flux,
                state.deposition_flux,
                sputtering_flux,
                state.escape_flux,
                current_temps
            );
        }

        // Finalize averages
        multiscale_evolution_->finalizeDiurnalAverages(static_cast<int>(num_timesteps));

        // Restore normal matrix rebuild and verbosity settings
        exosphere_model_->setSkipMatrixRebuild(false);
        exosphere_model_->setVerbose(show_progress_);

        return true;
    }

    void PTMExosphereIntegration::evolveEpoch(int epoch_idx, double epoch_duration) {
        if (!multiscale_evolution_ || !ice_layer_model_ || !ice_layer_state_) return;

        // Use the multi-timescale evolution controller to advance concentration
        multiscale_evolution_->evolveEpoch(*ice_layer_model_, *ice_layer_state_,
                                            face_areas_, epoch_duration);

        // Evolve ice crystallinity over the epoch using diurnal-averaged temperature
        // The analytical ODE solution handles the large epoch timestep gracefully
        if (crystallization_enabled_ && crystallization_model_) {
            const auto& diurnal_avg = multiscale_evolution_->getDiurnalAverages();

            crystallization_model_->evolveAllFaces(
                epoch_duration, diurnal_avg.avg_temperature,
                ice_layer_state_->crystalline_fraction);

            // Apply depositional amorphization from epoch-averaged deposition
            for (size_t i = 0; i < num_faces_; ++i) {
                double deposited = diurnal_avg.avg_deposition_flux[i] * epoch_duration;
                if (deposited > 0.0 && ice_layer_state_->ice_mass[i] > 0.0) {
                    IceCrystallizationModel::applyDepositionalAmorphization(
                        deposited, ice_layer_state_->ice_mass[i],
                        ice_layer_state_->crystalline_fraction[i]);
                }
            }

            // Accumulate radiation dose over epoch
            for (size_t i = 0; i < num_faces_; ++i) {
                ice_layer_state_->cumulative_dose[i] +=
                    ice_layer_state_->radiation_dose_rate[i] * epoch_duration;
            }
        }

        // Update ice mass for all faces based on updated concentrations
        for (size_t i = 0; i < num_faces_; ++i) {
            ice_layer_model_->updateIceMass(*ice_layer_state_, i);
        }
    }

    void PTMExosphereIntegration::exportEvolutionSummary() {
        if (!multiscale_evolution_) return;

        std::string filename = output_dir_ + "/evolution_summary.txt";
        std::ofstream file(filename);

        if (!file.is_open()) {
            std::cerr << "Failed to open evolution summary file!" << std::endl;
            return;
        }

        const auto& stats = multiscale_evolution_->getStatistics();
        const auto& config = multiscale_evolution_->getConfig();

        file << "Europa Ice Evolution Summary\n";
        file << "==============================\n\n";

        file << "Simulation Parameters:\n";
        file << "  Epochs: " << config.num_epochs << "\n";
        file << "  Cycles per epoch: " << config.cycles_per_epoch << "\n";
        file << "  Total time: " << std::scientific << std::setprecision(3)
             << config.totalTimeYears() << " years\n\n";

        file << "Final State:\n";
        file << "  Elapsed time: " << stats.elapsed_time_years << " years\n";
        file << "  Max non-ice concentration: " << std::fixed << std::setprecision(2)
             << (stats.max_nonice_concentration * 100.0) << " %\n";
        file << "  Mean non-ice concentration: " << std::fixed << std::setprecision(2)
             << (stats.mean_nonice_concentration * 100.0) << " %\n";
        file << "  Faces depleted (no ice): " << stats.faces_depleted << " / " << num_faces_ << "\n";
        file << "  Converged: " << (stats.converged ? "yes" : "no") << "\n\n";

        file << "Mass Balance:\n";
        file << "  Total sublimated: " << std::scientific << std::setprecision(3)
             << stats.total_sublimated_mass << " kg\n";
        file << "  Total deposited: " << stats.total_deposited_mass << " kg\n";
        file << "  Total sputtered: " << stats.total_sputtered_mass << " kg\n";
        file << "  Total escaped: " << stats.total_escaped_mass << " kg\n\n";

        // Latitude-dependent analysis of non-ice concentration
        file << "Latitude Distribution (final non-ice concentration):\n";
        std::vector<double> lat_bins(18, 0.0);  // 10-degree bins
        std::vector<int> lat_counts(18, 0);

        for (size_t i = 0; i < num_faces_; ++i) {
            double z = face_centers_[i][2];
            double r = std::sqrt(face_centers_[i][0] * face_centers_[i][0] +
                                face_centers_[i][1] * face_centers_[i][1] +
                                face_centers_[i][2] * face_centers_[i][2]);
            double lat = (r > 0) ? std::asin(z / r) * 180.0 / M_PI : 0.0;

            int bin = static_cast<int>((lat + 90.0) / 10.0);
            bin = std::clamp(bin, 0, 17);

            lat_bins[bin] += ice_layer_state_->nonice_concentration[i];
            lat_counts[bin]++;
        }

        for (int b = 0; b < 18; ++b) {
            double lat_center = -85.0 + b * 10.0;
            double avg_conc = (lat_counts[b] > 0) ? lat_bins[b] / lat_counts[b] : 0.0;
            file << "  " << std::setw(4) << lat_center << "°: "
                 << std::fixed << std::setprecision(2) << (avg_conc * 100.0) << " %\n";
        }

        // Crystallization state if enabled
        if (crystallization_enabled_ && crystallization_model_ && ice_layer_state_) {
            file << "\nCrystallization State:\n";
            double sum_phi = 0.0, min_phi = 1.0, max_phi = 0.0;
            size_t n_cryst = 0, n_amorph = 0;
            for (size_t i = 0; i < num_faces_; ++i) {
                double phi = ice_layer_state_->crystalline_fraction[i];
                sum_phi += phi;
                min_phi = std::min(min_phi, phi);
                max_phi = std::max(max_phi, phi);
                if (phi > 0.9) n_cryst++;
                if (phi < 0.1) n_amorph++;
            }
            file << "  Mean crystalline fraction: " << std::fixed << std::setprecision(2)
                 << (sum_phi / num_faces_ * 100.0) << " %\n";
            file << "  Range: " << (min_phi * 100.0) << " % - " << (max_phi * 100.0) << " %\n";
            file << "  Mostly crystalline (>90%): " << n_cryst << " faces\n";
            file << "  Mostly amorphous (<10%): " << n_amorph << " faces\n";
        }

        file << "\n";
        file << "Units:\n";
        file << "  nonice_concentration: dimensionless [0-1], 0 = pure ice, 1 = no ice\n";
        file << "  ice_mass: kg/m² (ice mass per unit surface area)\n";
        file << "  crystalline_fraction: dimensionless [0-1], 0 = amorphous, 1 = crystalline\n";
        file << "  porosity: dimensionless [0-1], volume fraction of pore space\n";
        file << "  radiation_dose_rate: eV/(molecule s)\n";
        file << "  Fluxes: kg/m²/s\n";
        file << "  Cumulative values: kg/m²\n";

        file.close();
        std::cout << "Evolution summary exported to: " << filename << std::endl;
    }

} // namespace exosphere
} // namespace ptm