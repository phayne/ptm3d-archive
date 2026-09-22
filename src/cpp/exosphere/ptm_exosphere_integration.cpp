#include "ptm_exosphere_integration.h"
#include "exosphere_physics.h"
#include "sputtering_model.h"
#include "ice_layer_model.h"
#include "ice_crystallization_model.h"
#include "multitimescale_evolution.h"
#include "composition_albedo.h"
#include "../mesh/mesh_utils.h"
#include "maps/albedo_manager.h"
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
#include <stdexcept>

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

// Compute exosphere face geometry in SI units (m, m^2) from a mesh in any
// length unit (EXO-02). The production ico6 mesh stores vertices in km; face
// centers were already projected to meters but areas were taken from the raw
// vertices, so every absolute mass/rate was 1e6 too small and every
// area-derived angle (smoothing radius, self-cap, shell width) 1000x too
// small. Both mesh-loading paths now go through this one routine.
//
// Units: a single uniform scale s = R_Europa / <|v|> converts the native
// length unit to meters (s = 1000 for a km mesh); areas scale by s^2. Face
// centers are projected onto the sphere of radius R_Europa.
//
// Hard gate: |sum(A) - 4 pi R^2| / (4 pi R^2) must be < 1e-3. The exosphere is
// a closed-sphere model; a mesh that fails this is either in unexpected units
// or not a global sphere, and must not run silently.
bool computeExosphereFaceGeometry(const Mesh& mesh,
                                  std::vector<std::array<double,3>>& centers,
                                  std::vector<double>& areas,
                                  std::vector<std::array<double,3>>* normals,
                                  std::string& error) {
    const double R = constants::EUROPA_RADIUS;
    const size_t n_faces = mesh.faces.size();
    centers.assign(n_faces, {0.0, 0.0, 0.0});
    areas.assign(n_faces, 0.0);
    if (normals) normals->assign(n_faces, {0.0, 0.0, 0.0});
    if (n_faces == 0) { error = "mesh has no faces"; return false; }

    auto projectToSphere = [R](std::array<double,3> c, bool& ok) {
        const double r = std::sqrt(c[0]*c[0] + c[1]*c[1] + c[2]*c[2]);
        ok = (r > 0.0 && std::isfinite(r));
        if (ok) { const double k = R / r; c[0] *= k; c[1] *= k; c[2] *= k; }
        return c;
    };
    const bool have_centers = (mesh.face_centers.size() == n_faces);

    if (!mesh.vertices.empty()) {
        double r_sum = 0.0;
        for (const auto& v : mesh.vertices) {
            r_sum += std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
        }
        const double r_mean = r_sum / static_cast<double>(mesh.vertices.size());
        if (!(r_mean > 0.0) || !std::isfinite(r_mean)) {
            error = "mean vertex radius is not positive and finite";
            return false;
        }
        const double s = R / r_mean;   // native length unit -> m
        std::cout << "[exosphere] mesh units: mean vertex radius " << std::setprecision(8)
                  << r_mean << " (native) -> scale " << s << " to meters" << std::endl;

        for (size_t i = 0; i < n_faces; ++i) {
            const auto& f = mesh.faces[i];
            const auto& v0 = mesh.vertices[f[0]];
            const auto& v1 = mesh.vertices[f[1]];
            const auto& v2 = mesh.vertices[f[2]];
            const std::array<double,3> e1{ v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2] };
            const std::array<double,3> e2{ v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2] };
            const std::array<double,3> cr{
                e1[1]*e2[2] - e1[2]*e2[1],
                e1[2]*e2[0] - e1[0]*e2[2],
                e1[0]*e2[1] - e1[1]*e2[0]
            };
            areas[i] = 0.5 * std::sqrt(cr[0]*cr[0] + cr[1]*cr[1] + cr[2]*cr[2]) * s * s;
            if (normals) (*normals)[i] = cr;   // unnormalized, diagnostics only

            const std::array<double,3> c = have_centers ? mesh.face_centers[i]
                : std::array<double,3>{ (v0[0]+v1[0]+v2[0]) / 3.0,
                                        (v0[1]+v1[1]+v2[1]) / 3.0,
                                        (v0[2]+v1[2]+v2[2]) / 3.0 };
            bool ok = false;
            centers[i] = projectToSphere(c, ok);
            if (!ok) { error = "face " + std::to_string(i) + " has a zero-radius center"; return false; }
        }
    } else {
        // No vertices: equal-area faces on the sphere (already m^2).
        if (!have_centers) { error = "mesh has neither vertices nor face centers"; return false; }
        const double equal_area = 4.0 * M_PI * R * R / static_cast<double>(n_faces);
        std::cout << "[exosphere] WARNING: mesh has no vertices; using equal face areas "
                  << equal_area << " m^2" << std::endl;
        for (size_t i = 0; i < n_faces; ++i) {
            areas[i] = equal_area;
            bool ok = false;
            centers[i] = projectToSphere(mesh.face_centers[i], ok);
            if (!ok) { error = "face " + std::to_string(i) + " has a zero-radius center"; return false; }
        }
    }

    double total = 0.0;
    for (double a : areas) total += a;
    const double sphere = 4.0 * M_PI * R * R;
    const double rel = (total - sphere) / sphere;
    std::cout << "[exosphere] total face area " << std::scientific << std::setprecision(6)
              << total << " m^2 = 4 pi R^2 x (1 " << (rel >= 0 ? "+ " : "- ")
              << std::abs(rel) << ")" << std::defaultfloat << std::endl;
    if (!(std::abs(rel) < 1e-3)) {
        std::ostringstream os;
        os << "total face area " << total << " m^2 differs from 4 pi R_Europa^2 = " << sphere
           << " m^2 by a relative " << rel << " (limit 1e-3): mesh is in unexpected units "
           << "or is not a closed global sphere";
        error = os.str();
        return false;
    }
    return true;
}

void PTMExosphereIntegration::setMesh(const std::shared_ptr<Mesh>& mesh) {
    mesh_ = mesh;

    if (!mesh_ || mesh_->faces.empty()) {
        std::cerr << "PTMExosphereIntegration::setMesh: invalid mesh (null or no faces)\n";
        return;
    }

    const size_t n_faces = mesh_->faces.size();
    std::string error;
    if (!computeExosphereFaceGeometry(*mesh_, face_centers_, face_areas_, &face_normals_, error)) {
        throw std::runtime_error("FATAL: exosphere mesh geometry: " + error);
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

            // PTM-17: a REQUESTED mesh that fails to load is FATAL. Never silently fall
            // back to a default icosphere for a specified mesh_file (that would run on the
            // wrong geometry). The empty-mesh_file procedural path above is unaffected.
            if (!mesh_utils.loadMeshFromFile(mesh_file)) {
                std::cerr << "FATAL: Failed to load mesh from file: " << mesh_file
                          << " (refusing to substitute a default icosphere)." << std::endl;
                return false;
            }
        }

        mesh_ = mesh_utils.getMesh();
        
        if (!mesh_ || mesh_->faces.empty()) {
            std::cerr << "Invalid mesh!" << std::endl;
            return false;
        }
        
        std::cout << "Loaded mesh with " << mesh_->faces.size() << " faces" << std::endl;
        
        // Face geometry in SI units, shared with setMesh() (EXO-02). The model
        // always receives the meter-scaled centers: the raw mesh_->face_centers
        // are in the mesh's native unit (km for the production meshes).
        std::string geometry_error;
        if (!computeExosphereFaceGeometry(*mesh_, face_centers_, face_areas_, &face_normals_,
                                          geometry_error)) {
            std::cerr << "FATAL: exosphere mesh geometry: " << geometry_error << std::endl;
            return false;
        }

        size_t num_faces = mesh_->faces.size();
        exosphere_model_ = std::make_unique<ExosphereModel>(num_faces);
        exosphere_model_->setMeshGeometry(face_centers_, face_areas_);

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

        // Pre-allocate to avoid repeated push_back reallocations
        for (size_t f = 0; f < mesh_->faces.size(); ++f) {
            surface_temperatures_[f].reserve(num_timesteps);
        }

        int progress_interval = std::max(1, num_timesteps / 4);

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

            if ((t + 1) % progress_interval == 0 || t == num_timesteps - 1) {
                std::cerr << "\r  Loading temperatures: " << (t + 1) << "/" << num_timesteps
                          << " (" << (100 * (t + 1) / num_timesteps) << "%)" << std::flush;
            }
        }
        std::cerr << std::endl;

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

            // Get ice fraction (continuous 0 to 1, scales sublimation rate)
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
                    // face_centers_ (meters), not mesh_->face_centers (native
                    // mesh unit, km for production meshes): EXO-02 unit class.
                    std::sqrt(std::pow(face_centers_[i][0] - face_centers_[j][0], 2) +
                             std::pow(face_centers_[i][1] - face_centers_[j][1], 2) +
                             std::pow(face_centers_[i][2] - face_centers_[j][2], 2)) : 0.0;
                
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

        // Write composition-dependent albedo if coupling is enabled
        if (albedo_coupling_enabled_ && !current_face_albedos_.empty()) {
            writeBinaryArray("bond_albedo", current_face_albedos_);
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

        // Calculate global mean atmospheric values
        double total_area = 0.0;
        double weighted_h2o_column_mass = 0.0;
        double weighted_h2o_vapor_pressure = 0.0;

        for (size_t i = 0; i < num_faces; ++i) {
            double area = face_areas_[i];
            total_area += area;

            // Get temperature for this face (diurnal-averaged in multiscale mode)
            double T = 100.0;
            if (multiscale_export_mode_ && multiscale_evolution_) {
                const auto& diurnal_avg = multiscale_evolution_->getDiurnalAverages();
                if (i < diurnal_avg.avg_temperature.size()) {
                    T = std::max(50.0, diurnal_avg.avg_temperature[i]);
                }
            }

            // Gate vapor pressure by ice availability (no ice = no local vapor source)
            double vp = state.vapor_pressure[i];
            if (ice_layer_enabled_ && ice_layer_model_ && ice_layer_state_) {
                vp *= ice_layer_model_->getIceAvailabilityFactor(*ice_layer_state_, i);
            }

            // Column mass = vapor_density * scale_height
            // Scale height H = kT / (m*g) for a Maxwellian atmosphere
            double scale_height = constants::BOLTZMANN_K * T /
                                  (constants::WATER_MOLECULAR_MASS * constants::EUROPA_GRAVITY);
            double vapor_density = vp * constants::WATER_MOLAR_MASS /
                                   (constants::GAS_CONSTANT_R * T);
            double column_mass = vapor_density * scale_height;

            weighted_h2o_column_mass += column_mass * area;
            weighted_h2o_vapor_pressure += vp * area;
        }

        double mean_h2o_column_mass = (total_area > 0) ? weighted_h2o_column_mass / total_area : 0.0;
        double mean_h2o_vapor_pressure = (total_area > 0) ? weighted_h2o_vapor_pressure / total_area : 0.0;

        // O2 and H2 approximate ratios (from photodissociation/radiolysis)
        double o2_production_ratio = 0.5;
        double h2_production_ratio = 1.0;
        double mean_o2_vapor_pressure = mean_h2o_vapor_pressure * o2_production_ratio *
                                        (constants::WATER_MOLAR_MASS / 0.032);
        double mean_h2_vapor_pressure = mean_h2o_vapor_pressure * h2_production_ratio *
                                        (constants::WATER_MOLAR_MASS / 0.002);

        // Compute mass balance rates from diurnal-averaged fluxes
        double total_sublimation_rate = 0.0;
        double total_deposition_rate = 0.0;
        double total_sputtering_rate = 0.0;
        if (multiscale_export_mode_ && multiscale_evolution_) {
            const auto& diurnal_avg = multiscale_evolution_->getDiurnalAverages();
            for (size_t i = 0; i < num_faces; ++i) {
                double area = face_areas_[i];
                total_sublimation_rate += diurnal_avg.avg_sublimation_flux[i] * area;
                total_deposition_rate += diurnal_avg.avg_deposition_flux[i] * area;
                total_sputtering_rate += diurnal_avg.avg_sputtering_flux[i] * area;
            }
        } else {
            for (size_t i = 0; i < num_faces; ++i) {
                double area = face_areas_[i];
                total_sublimation_rate += state.emission_flux[i] * area;
                total_deposition_rate += state.deposition_flux[i] * area;
            }
        }
        double net_mass_loss_rate = total_sublimation_rate + total_sputtering_rate - total_deposition_rate;

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
                 << "o2_vapor_pressure_Pa,h2_vapor_pressure_Pa,"
                 << "total_sublimation_rate_kg_s,total_deposition_rate_kg_s,"
                 << "net_mass_loss_rate_kg_s\n";
        }

        double time_years = time_value / timescale_constants::SECONDS_PER_YEAR;

        file << timestep_idx << ","
             << std::scientific << std::setprecision(6) << time_value << ","
             << std::scientific << std::setprecision(6) << time_years << ","
             << std::scientific << std::setprecision(6) << mean_h2o_column_mass << ","
             << std::scientific << std::setprecision(6) << mean_h2o_vapor_pressure << ","
             << std::scientific << std::setprecision(6) << mean_o2_vapor_pressure << ","
             << std::scientific << std::setprecision(6) << mean_h2_vapor_pressure << ","
             << std::scientific << std::setprecision(6) << total_sublimation_rate << ","
             << std::scientific << std::setprecision(6) << total_deposition_rate << ","
             << std::scientific << std::setprecision(6) << net_mass_loss_rate << "\n";

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
    // Albedo-Composition Coupling Implementation
    // =========================================================================

    void PTMExosphereIntegration::enableAlbedoCoupling(const CompositionAlbedoConfig& config) {
        if (num_faces_ == 0) {
            std::cerr << "PTMExosphereIntegration::enableAlbedoCoupling: no faces loaded yet!" << std::endl;
            return;
        }
        if (!ice_layer_enabled_ || !ice_layer_state_) {
            std::cerr << "PTMExosphereIntegration::enableAlbedoCoupling: "
                      << "ice layer model must be enabled first!" << std::endl;
            return;
        }

        albedo_mapper_ = std::make_unique<CompositionAlbedoMapper>(config);

        // Pass per-face non-ice endmember map if already loaded
        if (!nonice_albedo_map_.empty()) {
            albedo_mapper_->setNoniceEndmemberMap(nonice_albedo_map_);
        }

        // Initialize per-face albedos from current composition
        current_face_albedos_.resize(num_faces_);
        previous_face_albedos_.resize(num_faces_);

        albedo_mapper_->computeAlbedos(ice_layer_state_->nonice_concentration,
                                       current_face_albedos_);
        previous_face_albedos_ = current_face_albedos_;

        albedo_coupling_enabled_ = true;
        config.print();

        std::cout << "Albedo-composition coupling enabled." << std::endl;
        std::cout << "  Initial albedo from thermal sim: " << initial_albedo_ << std::endl;
        std::cout << "  Current mean composition-albedo: " << std::fixed << std::setprecision(4)
                  << std::accumulate(current_face_albedos_.begin(), current_face_albedos_.end(), 0.0)
                     / num_faces_ << std::endl;
    }

    void PTMExosphereIntegration::disableAlbedoCoupling() {
        albedo_coupling_enabled_ = false;
        std::cout << "Albedo-composition coupling disabled." << std::endl;
    }

    void PTMExosphereIntegration::loadNoniceAlbedoMap(
        const std::string& map_path, const std::string& maps_dir)
    {
        if (!mesh_) {
            std::cerr << "loadNoniceAlbedoMap: mesh must be loaded first!" << std::endl;
            return;
        }

        ptm::AlbedoManager mgr;
        // Initialize first (sets initialized_=true, needed for getAlbedosForMesh)
        mgr.initialize("Europa", "");
        bool loaded = false;
        if (!map_path.empty()) {
            loaded = mgr.loadFromFile(map_path);
            if (!loaded) {
                std::cerr << "Failed to load albedo map: " << map_path << std::endl;
                return;
            }
        } else if (!maps_dir.empty()) {
            loaded = mgr.initialize("Europa", maps_dir);
        }

        if (loaded && mgr.hasMap()) {
            nonice_albedo_map_ = mgr.getAlbedosForMesh(mesh_);
            auto [mn, mx] = std::minmax_element(nonice_albedo_map_.begin(),
                                                 nonice_albedo_map_.end());
            std::cout << "Loaded per-face non-ice albedo endmembers from map ("
                      << nonice_albedo_map_.size() << " faces, range ["
                      << std::fixed << std::setprecision(4) << *mn << ", " << *mx
                      << "])" << std::endl;

            // Pass to mapper if already configured
            if (albedo_mapper_) {
                albedo_mapper_->setNoniceEndmemberMap(nonice_albedo_map_);
            }
        } else {
            std::cerr << "No albedo map available for non-ice endmember." << std::endl;
        }
    }

    void PTMExosphereIntegration::initializeConcentrationsFromAlbedoMap(
        double percentile_low, double percentile_high)
    {
        if (nonice_albedo_map_.empty()) {
            std::cerr << "initializeConcentrationsFromAlbedoMap: "
                      << "albedo map not loaded (call loadNoniceAlbedoMap first)" << std::endl;
            return;
        }
        if (!ice_layer_model_ || !ice_layer_state_) {
            std::cerr << "initializeConcentrationsFromAlbedoMap: "
                      << "ice layer model not enabled (call enableIceLayerModel first)" << std::endl;
            return;
        }

        // Compute per-face concentrations from the observed albedo map
        std::vector<double> concentrations;
        CompositionAlbedoMapper::computeConcentrationsFromAlbedoMap(
            nonice_albedo_map_, percentile_low, percentile_high, concentrations);

        // Set concentrations on the ice layer model
        ice_layer_model_->initializeFromConcentrations(*ice_layer_state_, concentrations);

        // Set per-face initial albedos = observed map albedos
        // (these are the albedos the PTM3D temperatures correspond to)
        initial_face_albedos_ = nonice_albedo_map_;

        // Sync ice mass to exosphere state
        if (exosphere_model_) {
            auto& exo_state = exosphere_model_->getMutableState();
            for (size_t i = 0; i < static_cast<size_t>(num_faces_); ++i) {
                exo_state.accumulated_mass[i] = ice_layer_state_->ice_mass[i];
            }
            exo_state.updateIceThickness();
        }

        std::cout << "Initialized per-face concentrations from albedo map." << std::endl;
    }

    double PTMExosphereIntegration::getFaceAlbedo(int face_idx) const {
        if (albedo_coupling_enabled_ && face_idx >= 0 &&
            face_idx < static_cast<int>(current_face_albedos_.size())) {
            return current_face_albedos_[face_idx];
        }
        if (!initial_face_albedos_.empty() && face_idx >= 0 &&
            face_idx < static_cast<int>(initial_face_albedos_.size())) {
            return initial_face_albedos_[face_idx];
        }
        return initial_albedo_;
    }

    double PTMExosphereIntegration::updateAlbedoFromComposition() {
        if (!albedo_mapper_ || !ice_layer_state_) return 0.0;

        // Save previous albedos for change detection
        previous_face_albedos_ = current_face_albedos_;

        // Compute new albedos from current composition
        albedo_mapper_->computeAlbedos(ice_layer_state_->nonice_concentration,
                                       current_face_albedos_);

        // Compute maximum change
        return CompositionAlbedoMapper::maxAlbedoChange(previous_face_albedos_,
                                                        current_face_albedos_);
    }

    void PTMExosphereIntegration::updateTemperaturesForAlbedoChange() {
        if (surface_temperatures_.empty()) return;

        size_t num_timesteps = surface_temperatures_[0].size();
        double threshold = albedo_mapper_->getConfig().albedo_update_threshold;
        int faces_updated = 0;

        // Scale temperatures for faces with significant albedo change
        // T_new = T_old * ((1 - A_new) / (1 - A_old))^(1/4)
        #pragma omp parallel for reduction(+:faces_updated)
        for (size_t f = 0; f < static_cast<size_t>(num_faces_); ++f) {
            double albedo_change = std::abs(current_face_albedos_[f] - previous_face_albedos_[f]);
            if (albedo_change < threshold) continue;

            double old_absorptivity = 1.0 - previous_face_albedos_[f];
            double new_absorptivity = 1.0 - current_face_albedos_[f];

            // Avoid division by zero
            if (old_absorptivity < 1e-10) continue;

            double ratio = new_absorptivity / old_absorptivity;
            double scale = std::pow(ratio, 0.25);

            for (size_t t = 0; t < num_timesteps; ++t) {
                surface_temperatures_[f][t] *= scale;
            }
            faces_updated++;
        }

        if (show_progress_ && faces_updated > 0) {
            std::cout << "  Albedo coupling: updated temperatures for " << faces_updated
                      << " / " << num_faces_ << " faces" << std::endl;
        }
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

        // Apply user-specified convergence thresholds (if set)
        if (convergence_threshold_ >= 0.0) {
            config.convergence_threshold = convergence_threshold_;
        }
        if (flux_convergence_threshold_ >= 0.0) {
            config.flux_convergence_threshold = flux_convergence_threshold_;
        }

        // analytical_acceleration and cache_refresh_interval are kept in the
        // signature for API compatibility only (M3). Nothing implements them:
        // super-stepping is always closed-form, and the diurnal cycle is
        // recomputed only when the cache is empty (epoch 0) or albedo coupling
        // forces it. Refuse non-default values instead of ignoring them.
        if (!analytical_acceleration || cache_refresh_interval != 0) {
            throw std::invalid_argument(
                "runMultiTimescaleSimulation: analytical_acceleration=false and "
                "cache_refresh_interval!=0 are not implemented (they were previously "
                "accepted and silently ignored)");
        }

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
        int total_super_stepped = 0;  // Total epochs advanced via super-stepping
        int max_super_step = 0;       // Largest single super-step taken

        // Track albedo coupling statistics
        int albedo_temp_refresh_count = 0;

        // Main epoch loop with flux caching and super-stepping optimization
        for (int epoch = 0; epoch < num_epochs; ) {
            // Step 0 (optional): Update albedo from current composition
            bool force_cache_refresh = false;
            if (albedo_coupling_enabled_ && albedo_mapper_) {
                double max_albedo_change = updateAlbedoFromComposition();

                // Update statistics
                auto& stats = const_cast<EvolutionStatistics&>(multiscale_evolution_->getStatistics());
                stats.max_albedo_change = max_albedo_change;
                double albedo_sum = 0.0;
                for (size_t i = 0; i < static_cast<size_t>(num_faces_); ++i) {
                    albedo_sum += current_face_albedos_[i];
                }
                stats.mean_albedo = albedo_sum / num_faces_;

                if (max_albedo_change > albedo_mapper_->getConfig().albedo_update_threshold) {
                    // Albedo changed significantly - update temperatures and force cache refresh
                    updateTemperaturesForAlbedoChange();
                    force_cache_refresh = true;
                    albedo_temp_refresh_count++;
                    stats.albedo_temp_refreshes = albedo_temp_refresh_count;
                }
            }

            // Step 1: Either compute full diurnal cycle or use cached fluxes
            bool need_full_diurnal = force_cache_refresh ||
                multiscale_evolution_->needsCacheRefresh(
                    *ice_layer_model_, *ice_layer_state_, face_areas_, epoch);

            if (need_full_diurnal) {
                // Run full diurnal cycle (expensive)
                if (!computeDiurnalAverages(dt_seconds)) {
                    std::cerr << "Failed to compute diurnal averages at epoch " << epoch << std::endl;
                    break;
                }
                // Cache the computed base fluxes for future epochs
                multiscale_evolution_->cacheBaseFluxes(*ice_layer_model_, *ice_layer_state_, face_areas_, epoch);
                full_diurnal_count++;
            } else {
                // Use cached fluxes with ice availability scaling (fast, milliseconds)
                multiscale_evolution_->applyScaledFluxes(*ice_layer_model_, *ice_layer_state_, face_areas_);
                cached_epoch_count++;
            }

            // Step 2: Evolve one epoch to establish concentration_change_rate_
            evolveEpoch(epoch, epoch_duration);

            // Step 3: Compute safe super-step size.
            // After evolving one epoch, concentration_change_rate_ is set.
            // Predict how many MORE epochs can elapse before the next face depletes.
            int safe_extra = 0;
            {
                int remaining = num_epochs - epoch - 1;
                if (remaining > 0) {
                    // Limit super-step to not overshoot export intervals
                    int max_step = remaining;
                    if (effective_export_interval > 0) {
                        int next_export = ((epoch / effective_export_interval) + 1) * effective_export_interval;
                        max_step = std::min(max_step, next_export - epoch - 1);
                    }

                    if (max_step > 0) {
                        safe_extra = multiscale_evolution_->computeSafeEpochCount(
                            *ice_layer_state_, face_areas_, epoch_duration, max_step) - 1;
                        // -1 because computeSafeEpochCount returns total including the one we already did
                        safe_extra = std::max(0, safe_extra);
                    }

                    if (safe_extra > 0) {
                        // Save per-epoch stats before super-step overwrites them
                        double saved_max_change = multiscale_evolution_->getStatistics().max_concentration_change;
                        double saved_flux_imbalance = multiscale_evolution_->getStatistics().flux_imbalance_fraction;

                        // Super-step: evolve concentration forward by safe_extra more epochs
                        // using the same cached/computed fluxes
                        evolveEpoch(epoch + 1, epoch_duration * safe_extra);
                        total_super_stepped += safe_extra;
                        max_super_step = std::max(max_super_step, safe_extra + 1);
                        cached_epoch_count += safe_extra;

                        // Restore per-epoch stats for meaningful progress reporting
                        auto& stats = const_cast<EvolutionStatistics&>(multiscale_evolution_->getStatistics());
                        stats.max_concentration_change = saved_max_change;
                        stats.flux_imbalance_fraction = saved_flux_imbalance;
                    }
                }
            }

            int step_size = 1 + safe_extra;

            // Step 4: Update statistics (accounts for all epochs in this super-step)
            multiscale_evolution_->updateStatistics(*ice_layer_model_, *ice_layer_state_,
                                                     epoch_duration, step_size);

            // Progress reporting
            if (show_progress_) {
                multiscale_evolution_->printProgress();
            }

            // Export at intervals
            int next_epoch = epoch + step_size;
            if (effective_export_interval > 0 &&
                (epoch % effective_export_interval == 0 || next_epoch >= num_epochs)) {
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
                        exo_state.emission_flux[i] = diurnal_avg.avg_sublimation_flux[i];
                        exo_state.deposition_flux[i] = diurnal_avg.avg_deposition_flux[i];
                        exo_state.net_flux[i] = diurnal_avg.avg_net_flux[i];
                        exo_state.escape_flux[i] = diurnal_avg.avg_escape_flux[i];

                        // Recompute atmospheric properties from diurnal-averaged temperature
                        double T = std::max(50.0, diurnal_avg.avg_temperature[i]);
                        double ice_avail = (ice_layer_enabled_ && ice_layer_model_ && ice_layer_state_)
                            ? ice_layer_model_->getIceAvailabilityFactor(*ice_layer_state_, i)
                            : 1.0;
                        exo_state.vapor_pressure[i] = VaporPressure::calculateVaporPressure(T) * ice_avail;
                        exo_state.vapor_density[i] = VaporPressure::calculateVaporDensity(
                            exo_state.vapor_pressure[i], T);
                        exo_state.thermal_velocity[i] = VaporPressure::calculateThermalVelocity(T) * ice_avail;
                    }
                    exo_state.updateIceThickness();
                }

                // Export VTU with epoch index
                double time_value = multiscale_evolution_->getStatistics().elapsed_time_years *
                                   timescale_constants::SECONDS_PER_YEAR;
                exportTimestepVTU(epoch, time_value);
            }

            // Advance epoch counter by the super-step size
            epoch += step_size;
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
        if (total_super_stepped > 0) {
            std::cout << "    Super-stepped epochs: " << total_super_stepped
                      << " (max step: " << max_super_step << ")" << std::endl;
        }
        if (full_diurnal_count > 0) {
            double speedup = static_cast<double>(full_diurnal_count + cached_epoch_count) / full_diurnal_count;
            std::cout << "    Effective speedup: " << std::setprecision(1) << speedup << "x" << std::endl;
        }
        if (albedo_coupling_enabled_) {
            std::cout << "  Albedo coupling:" << std::endl;
            std::cout << "    Temperature refreshes: " << albedo_temp_refresh_count << std::endl;
            double mean_a = std::accumulate(current_face_albedos_.begin(),
                                            current_face_albedos_.end(), 0.0) / num_faces_;
            std::cout << "    Final mean albedo: " << std::fixed << std::setprecision(4) << mean_a << std::endl;
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

        // Subsample timesteps for faster accumulation.
        // Stride > 1 uses every Nth timestep with equal weight.
        // Error is small for smooth diurnal curves: <2% for stride=4.
        int stride = diurnal_accumulation_stride_;
        size_t effective_timesteps = (num_timesteps + stride - 1) / stride;

        std::cout << "  Computing diurnal averages (" << effective_timesteps
                  << "/" << num_timesteps << " timesteps";
        if (stride > 1) std::cout << ", stride=" << stride;
        std::cout << ")..." << std::flush;
        auto t_diurnal_start = std::chrono::steady_clock::now();

        // Reset accumulator
        multiscale_evolution_->resetDiurnalAccumulator();

        // Skip matrix rebuilds during the per-timestep diurnal loop.
        // The full transport matrix will be built ONCE after averaging, using
        // the diurnal-averaged temperatures.
        exosphere_model_->setSkipMatrixRebuild(true);
        exosphere_model_->setVerbose(false);

        // Pre-allocate buffers reused across all timesteps
        size_t nf = mesh_->faces.size();
        std::vector<double> current_temps(nf);
        std::vector<double> zero_deposition(num_faces_, 0.0);
        std::vector<double> sputtering_flux;
        // Per-timestep crystallization rate buffer (only populated when the
        // crystallization model is active; left untouched otherwise).
        std::vector<double> k_cryst_step;
        const std::vector<double>* porosity_per_face = nullptr;
        if (crystallization_enabled_ && crystallization_model_) {
            k_cryst_step.assign(nf, 0.0);
            porosity_per_face = &crystallization_model_->getPorosity();
        }

        // Run through one diurnal cycle, accumulating fluxes.
        // Deposition will be computed after finalization from the escape fraction.
        int accumulated_count = 0;
        for (size_t t = 0; t < num_timesteps; t += stride) {
            // Extract temperatures for this timestep
            for (size_t f = 0; f < nf; ++f) {
                current_temps[f] = surface_temperatures_[f][t];
            }

            // Compute emission, escape, and vapor properties for this timestep
            exosphere_model_->updateTemperatures(current_temps);

            const auto& state = exosphere_model_->getState();

            // Get sputtering fluxes if enabled
            if (sputtering_enabled_ && sputtering_model_ && sputtering_state_) {
                sputtering_model_->computeSputteringRates(current_temps, face_areas_, *sputtering_state_);
                sputtering_flux = sputtering_state_->total_mass_loss;
            }

            // Accumulate emission, sputtering, escape, and temperature.
            // Deposition passed as zeros — computed from escape fraction below.
            multiscale_evolution_->accumulateDiurnalStep(
                state.emission_flux,
                zero_deposition,
                sputtering_flux,
                state.escape_flux,
                current_temps
            );

            // Accumulate <k_cryst(T(t))> per face for the multi-timescale
            // crystallization step (Eq. D34 with the time-averaged Avrami rate).
            // Lock-step with accumulateDiurnalStep so the averaging denominator
            // matches.
            if (crystallization_enabled_ && crystallization_model_ && porosity_per_face) {
                #pragma omp parallel for
                for (size_t f = 0; f < nf; ++f) {
                    k_cryst_step[f] = crystallization_model_->crystallizationRate(
                        current_temps[f], (*porosity_per_face)[f]);
                }
                multiscale_evolution_->accumulateCrystallizationRate(k_cryst_step);
            }

            accumulated_count++;
        }

        // Finalize emission/sputtering/escape/temperature averages
        multiscale_evolution_->finalizeDiurnalAverages(accumulated_count);

        auto t_diurnal_end = std::chrono::steady_clock::now();
        double diurnal_sec = std::chrono::duration<double>(t_diurnal_end - t_diurnal_start).count();
        std::cout << " done (" << std::fixed << std::setprecision(1) << diurnal_sec << " s)" << std::endl;

        // Compute diurnal-averaged deposition using matrix-free transport.
        // This evaluates dep = M * emission on-the-fly without forming the
        // full sparse matrix (~17 GB for 1.3M faces). Uses ~90 MB instead.
        auto& diurnal_avg = multiscale_evolution_->getMutableDiurnalAverages();

        std::cout << "  Computing transport product (matrix-free)..." << std::flush;
        auto t_transport_start = std::chrono::steady_clock::now();

        // Prepare area-weighted emission vector
        std::vector<double> avg_emission(num_faces_);
        for (size_t i = 0; i < static_cast<size_t>(num_faces_); ++i) {
            avg_emission[i] = diurnal_avg.avg_sublimation_flux[i] * face_areas_[i];
        }

        // Matrix-free: two-pass source-based scatter. The kernel is evaluated
        // at each source's EMISSION-WEIGHTED temperature, not its diurnal mean
        // (M1): hops are launched when the face is warm.
        {
            double w_sum = 0.0, tw_sum = 0.0, tm_sum = 0.0;
            for (size_t i = 0; i < static_cast<size_t>(num_faces_); ++i) {
                w_sum += avg_emission[i];
                tw_sum += avg_emission[i] * diurnal_avg.emission_weighted_temperature[i];
                tm_sum += avg_emission[i] * diurnal_avg.avg_temperature[i];
            }
            if (w_sum > 0.0) {
                std::cout << "\n  [exosphere] transport temperature (emission-weighted over sources): "
                          << std::fixed << std::setprecision(2) << (tw_sum / w_sum)
                          << " K vs diurnal-mean " << (tm_sum / w_sum) << " K; "
                          << diurnal_avg.emission_weighted_fallback_count
                          << " non-emitting faces use the diurnal mean" << std::endl;
            }
        }
        std::vector<double> avg_dep;
        exosphere_model_->computeTransportProduct(
            diurnal_avg.emission_weighted_temperature, avg_emission, avg_dep);

        for (size_t i = 0; i < static_cast<size_t>(num_faces_); ++i) {
            diurnal_avg.avg_deposition_flux[i] = avg_dep[i] / face_areas_[i];
        }

        auto t_transport_end = std::chrono::steady_clock::now();
        double transport_sec = std::chrono::duration<double>(t_transport_end - t_transport_start).count();
        std::cout << "  Transport complete (" << std::fixed << std::setprecision(1)
                  << transport_sec << " s)" << std::endl;

        // Smooth sublimation (and sputtering) to match the transport's spatial
        // resolution. The transport post-smoothing averages deposition over
        // 20 x the mean face angular radius (about 8 deg on ico6; logged by
        // computeTransportProduct), but sublimation is computed per-face from
        // temperature. (Before EXO-02 the km^2 areas made that radius ~0.008
        // deg, i.e., no smoothing at all.)
        // Sharp temperature boundaries from albedo map pixels (~0.5° resolution)
        // create sharp sublimation features that appear as artifacts in net flux.
        // Smoothing both sides ensures consistent spatial resolution.
        // Area-weighted smoothing does not conserve sum(flux * area) on its own,
        // so each smoothed field is rescaled to its pre-smoothing total. Both
        // sublimation and sputtering go through the same path (EXO-08: the
        // sputtering field used to be smoothed without this rescale).
        auto smoothPreservingTotal = [&](std::vector<double>& field) {
            double total_before = 0.0;
            for (size_t i = 0; i < static_cast<size_t>(num_faces_); ++i) {
                total_before += field[i] * face_areas_[i];
            }
            std::vector<double> smoothed(num_faces_);
            exosphere_model_->smoothField(field, smoothed);
            double total_after = 0.0;
            for (size_t i = 0; i < static_cast<size_t>(num_faces_); ++i) {
                total_after += smoothed[i] * face_areas_[i];
            }
            if (total_after > 0.0) {
                const double scale = total_before / total_after;
                for (size_t i = 0; i < static_cast<size_t>(num_faces_); ++i) {
                    smoothed[i] *= scale;
                }
            }
            field = std::move(smoothed);
        };
        smoothPreservingTotal(diurnal_avg.avg_sublimation_flux);
        if (sputtering_enabled_) {
            smoothPreservingTotal(diurnal_avg.avg_sputtering_flux);
        }

        // Update net flux consistently
        #pragma omp parallel for
        for (size_t i = 0; i < static_cast<size_t>(num_faces_); ++i) {
            diurnal_avg.avg_net_flux[i] = diurnal_avg.avg_deposition_flux[i]
                                        - diurnal_avg.avg_sublimation_flux[i]
                                        - diurnal_avg.avg_sputtering_flux[i];
        }

        return true;
    }

    void PTMExosphereIntegration::evolveEpoch(int epoch_idx, double epoch_duration) {
        if (!multiscale_evolution_ || !ice_layer_model_ || !ice_layer_state_) return;

        // Use the multi-timescale evolution controller to advance concentration
        multiscale_evolution_->evolveEpoch(*ice_layer_model_, *ice_layer_state_,
                                            face_areas_, epoch_duration);

        // Evolve ice crystallinity over the epoch using the time-averaged
        // crystallization rate <k_cryst(T(t))>, NOT k_cryst(<T>). The latter
        // is biased by Jensen's inequality near the crystallization knife-edge
        // (k_cryst is exponential in T) and underpredicts steady-state phi_C
        // by orders of magnitude on warm-equator faces.
        if (crystallization_enabled_ && crystallization_model_) {
            const auto& diurnal_avg = multiscale_evolution_->getDiurnalAverages();

            crystallization_model_->evolveAllFacesWithRate(
                epoch_duration, diurnal_avg.avg_crystallization_rate,
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

        if (albedo_coupling_enabled_ && !current_face_albedos_.empty()) {
            file << "Albedo-Composition Coupling:\n";
            double min_a = *std::min_element(current_face_albedos_.begin(), current_face_albedos_.end());
            double max_a = *std::max_element(current_face_albedos_.begin(), current_face_albedos_.end());
            double mean_a = std::accumulate(current_face_albedos_.begin(),
                                            current_face_albedos_.end(), 0.0) / num_faces_;
            file << "  Initial albedo: " << std::fixed << std::setprecision(4) << initial_albedo_ << "\n";
            file << "  Final mean albedo: " << mean_a << "\n";
            file << "  Final albedo range: " << min_a << " - " << max_a << "\n";
            file << "  Ice endmember: " << albedo_mapper_->getConfig().albedo_ice << "\n";
            file << "  Non-ice endmember: " << albedo_mapper_->getConfig().albedo_nonice << "\n";
            file << "  Temperature refreshes: " << stats.albedo_temp_refreshes << "\n\n";
        }

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