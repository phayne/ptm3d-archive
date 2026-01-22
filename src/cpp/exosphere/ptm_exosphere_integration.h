#ifndef PTM_EXOSPHERE_INTEGRATION_H
#define PTM_EXOSPHERE_INTEGRATION_H

#include <memory>
#include <vector>
#include <array>
#include <string>
#include <Eigen/Dense>
#include <Eigen/Sparse>

// Forward declarations
namespace ptm {
    struct Mesh;

    namespace exosphere {

        // Forward declaration so we can hold a unique_ptr in the header
        class ExosphereModel;
        class SputteringModel;
        struct SputteringConfig;
        struct SputteringState;
        class IceLayerModel;
        struct IceLayerState;
        struct IceLayerConfig;
        class MultiTimescaleEvolution;
        struct MultiTimescaleConfig;

        /**
         * PTM Exosphere Integration
         * 
         * This class provides integration between PTM3D thermal output
         * and the ballistic exosphere transport model.
         * 
         * Workflow:
         * 1. Load mesh geometry (same as used by PTM3D)
         * 2. Load temperature data from PTM3D output
         * 3. Calculate vapor pressures and emission fluxes
         * 4. Simulate ballistic transport between faces
         * 5. Track mass balance over time
         */
        class PTMExosphereIntegration {
        private:
            std::shared_ptr<Mesh> mesh_;
            std::string output_dir_;
            
            // Mesh properties
            std::vector<double> face_areas_;
            std::vector<std::array<double, 3>> face_centers_;
            std::vector<std::array<double, 3>> face_normals_;
            
            // Temperature data from PTM3D [face][time]
            std::vector<std::vector<double>> surface_temperatures_;
            std::vector<double> time_points_;
            int num_timesteps_;
            int num_faces_;
            
            // Physical parameters
            const double g_europa_ = 1.314;  // m/s^2
            const double dt_default_ = 3600.0;  // 1 hour timestep
            
            // Mass balance tracking
            std::vector<double> surface_ice_mass_;     // kg per face
            std::vector<double> emission_rates_;       // kg/s per face  
            std::vector<double> deposition_rates_;     // kg/s per face
            std::vector<double> net_flux_rates_;       // kg/s per face (deposition - emission)
            
            // Transport matrix (sparse for efficiency)
            Eigen::SparseMatrix<double> transport_matrix_;
            bool matrix_built_;
            
            // Analysis results
            struct MassBalance {
                double total_sublimation_rate;   // kg/s
                double total_deposition_rate;    // kg/s
                double total_escape_rate;        // kg/s
                double net_loss_rate;            // kg/s

                // Detailed rates used by analysis and logging
                double polar_deposition_rate = 0.0;   // kg/s
                double equatorial_loss_rate = 0.0;    // kg/s

                // Optional fractions (kept for compatibility)
                double polar_deposition_fraction = 0.0;
                double equatorial_loss_fraction = 0.0;

                std::vector<double> latitude_distribution;
                std::vector<double> longitude_distribution;
            };
            
            MassBalance mass_balance_;

            // Owns the physics model (fix: add missing member)
            std::unique_ptr<ExosphereModel> exosphere_model_;

            // Sputtering model (optional, can be enabled/disabled)
            std::unique_ptr<SputteringModel> sputtering_model_;
            std::unique_ptr<SputteringState> sputtering_state_;
            bool sputtering_enabled_ = false;

            // Ice layer model (optional, for concentration tracking in thermally active layer)
            std::unique_ptr<IceLayerModel> ice_layer_model_;
            std::unique_ptr<IceLayerState> ice_layer_state_;
            bool ice_layer_enabled_ = false;

            // Multi-timescale evolution (optional, for geological timescales)
            std::unique_ptr<MultiTimescaleEvolution> multiscale_evolution_;
            bool multiscale_enabled_ = false;

            // Export mode flag (determines temperature output format)
            bool multiscale_export_mode_ = false;

            // Private methods
            void calculateFaceProperties();
            double calculateGreatCircleDistance(int face_i, int face_j);
            double calculateBallisticProbability(double distance, double temperature);
            double calculateEscapeFraction(double temperature);
            void buildTransportMatrix(double avg_temperature);
            double besselK0(double x);
            double besselI0(double x);

            bool show_progress_ = true;
            
        public:
            /**
             * Constructor
             * @param output_dir Directory for exosphere output files
             */
            PTMExosphereIntegration(const std::string& output_dir = "./exosphere_output");
            
            /**
             * Destructor
             */
            ~PTMExosphereIntegration();
            
            /**
             * Set the mesh (should be same mesh used by PTM3D)
             * @param mesh Shared pointer to mesh structure
             */
            void setMesh(const std::shared_ptr<Mesh>& mesh);
            
            /**
             * Load mesh from file
             * @param mesh_file Path to mesh file (PLY, OBJ, or PTM format)
             * @return true if successful
             */
            bool loadMesh(const std::string& mesh_file);
            
            /**
             * Load temperature data from PTM3D binary output
             * @param temperature_dir Directory containing temperature_XXXX.bin files
             * @param num_timesteps Number of timesteps to load
             * @return true if successful
             */
            bool loadTemperatureData(const std::string& temperature_dir, int num_timesteps);
            
            /**
             * Load temperature data from VTK files
             * @param vtk_dir Directory containing VTK files
             * @param num_timesteps Number of timesteps to load
             * @return true if successful
             */
            bool loadTemperatureDataVTK(const std::string& vtk_dir, int num_timesteps);
            
            /**
             * Run exosphere simulation for one timestep
             * @param timestep_idx Index of timestep to process
             * @param dt_seconds Time step in seconds (default 3600)
             */
            void simulateTimestep(int timestep_idx, double dt_seconds = 3600.0);
            
            /**
             * Run complete simulation for all loaded timesteps
             * @param dt_seconds Time step in seconds
             * @param export_all_steps Export results at each timestep
             */
            void runSimulation(double dt_seconds = 3600.0, bool export_all_steps = false);
            
            /**
             * Analyze mass balance at a given timestep
             * @param timestep_idx Timestep index
             */
            void analyzeMassBalance(int timestep_idx);
            
            /**
             * Export results for a specific timestep
             * @param timestep_idx Timestep index
             */
            void exportTimestep(int timestep_idx);
            
            /**
             * Export final analysis and summary
             */
            void exportFinalAnalysis();
            
            /**
             * Export transport matrix for visualization
             * @param filename Output filename
             */
            void exportTransportMatrix(const std::string& filename);

            /**
             * Export timestep data as binary files for VTK conversion
             * @param timestep_idx Timestep index
             * @param time_value Time value in seconds
             */
            void exportTimestepVTU(int timestep_idx, double time_value);

            /**
             * Export mesh in PLY format for VTK conversion
             */
            void exportMeshPLY();

            /**
             * Create time_steps.txt and metadata for Python VTK converter
             * @param num_timesteps Total number of timesteps
             * @param time_values Vector of time values
             */
            void createPVDFile(int num_timesteps, const std::vector<double>& time_values);

            /**
             * Export atmospheric mass summary (column densities, vapor pressures)
             * @param timestep_idx Timestep or epoch index
             * @param time_value Time value in seconds
             */
            void exportAtmosphericMass(int timestep_idx, double time_value);

            /**
             * Get emission rate for a specific face
             * @param face_idx Face index
             * @return Emission rate in kg/s
             */
            double getEmissionRate(int face_idx) const {
                return (face_idx < num_faces_) ? emission_rates_[face_idx] : 0.0;
            }
            
            /**
             * Get deposition rate for a specific face
             * @param face_idx Face index
             * @return Deposition rate in kg/s
             */
            double getDepositionRate(int face_idx) const {
                return (face_idx < num_faces_) ? deposition_rates_[face_idx] : 0.0;
            }
            
            /**
             * Get net mass flux for a specific face
             * @param face_idx Face index
             * @return Net flux in kg/s (positive = gain, negative = loss)
             */
            double getNetFlux(int face_idx) const {
                return (face_idx < num_faces_) ? net_flux_rates_[face_idx] : 0.0;
            }
            
            /**
             * Get accumulated surface ice mass for a specific face
             * @param face_idx Face index
             * @return Mass in kg
             */
            double getSurfaceIceMass(int face_idx) const {
                return (face_idx < num_faces_) ? surface_ice_mass_[face_idx] : 0.0;
            }
            
            /**
             * Get the mass balance summary
             * @return MassBalance structure
             */
            const MassBalance& getMassBalance() const { return mass_balance_; }

            // Progress bar toggle
            void setShowProgress(bool enabled) { show_progress_ = enabled; }

            // =========================================================
            // Sputtering Model Interface
            // =========================================================

            /**
             * Enable sputtering with default or custom configuration
             * @param rate_scale Global rate scaling factor (1.0 = Cassidy et al. 2013 rates)
             * @param hemisphere_contrast Trailing/leading flux ratio (default 3.0)
             */
            void enableSputtering(double rate_scale = 1.0, double hemisphere_contrast = 3.0);

            /**
             * Disable sputtering (sublimation-only mode)
             */
            void disableSputtering();

            /**
             * Check if sputtering is enabled
             */
            bool isSputteringEnabled() const { return sputtering_enabled_; }

            /**
             * Set sputtering rate scale factor
             * @param scale Multiplier for global sputtering rate (1.0 = nominal)
             */
            void setSputteringRateScale(double scale);

            /**
             * Get sputtering statistics
             * @return Total mass lost to sputtering escape [kg]
             */
            double getSputteringEscapedMass() const;

            /**
             * Get sputtering flux for a specific face
             * @param face_idx Face index
             * @return Sputtering mass loss flux [kg/m²/s]
             */
            double getSputteringFlux(int face_idx) const;

            // =========================================================
            // Ice Layer Model Interface (thermally active layer with concentration tracking)
            // =========================================================

            /**
             * Enable ice layer model with concentration tracking in thermally active layer
             * @param initial_nonice_concentration Initial non-ice concentration [0-1], default 0.01 (1%)
             */
            void enableIceLayerModel(double initial_nonice_concentration = 0.01);

            /**
             * Disable ice layer model (use simple ice tracking)
             */
            void disableIceLayerModel();

            /**
             * Check if ice layer model is enabled
             */
            bool isIceLayerEnabled() const { return ice_layer_enabled_; }

            /**
             * Get non-ice concentration for a face [dimensionless, 0-1]
             * @param face_idx Face index
             * @return Non-ice concentration (0 = pure ice, 1 = no ice/depleted)
             */
            double getNoniceConcentration(int face_idx) const;

            /**
             * Get ice mass for a face [kg/m²]
             * @param face_idx Face index
             * @return Ice mass per unit area
             */
            double getIceMass(int face_idx) const;

            /**
             * Get ice availability factor for a face
             * @param face_idx Face index
             * @return 1.0 if ice available, 0.0 if depleted
             */
            double getIceAvailabilityFactor(int face_idx) const;

            /**
             * Get cumulative ice sublimated for a face [kg/m²]
             */
            double getCumulativeSublimated(int face_idx) const;

            /**
             * Get cumulative ice sputtered for a face [kg/m²]
             */
            double getCumulativeSputtered(int face_idx) const;

            /**
             * Get cumulative ice deposited for a face [kg/m²]
             */
            double getCumulativeDeposited(int face_idx) const;

            // =========================================================
            // Multi-Timescale Evolution Interface
            // =========================================================

            /**
             * Run multi-timescale simulation using operator splitting
             *
             * This enables simulation over geological timescales (kyr to Myr)
             * by separating fast diurnal dynamics from slow lag evolution.
             *
             * @param num_epochs Number of epochs to simulate
             * @param cycles_per_epoch Diurnal cycles per epoch (default: 100 ≈ 1 year)
             * @param export_interval Export VTU every N epochs (0 = final only)
             * @param analytical_acceleration Use analytical solution for slow regions
             * @param cache_refresh_interval Force diurnal recomputation every N epochs
             *                               (0 = adaptive refresh only, fastest)
             */
            void runMultiTimescaleSimulation(int num_epochs,
                                              int cycles_per_epoch = 100,
                                              int export_interval = 100,
                                              bool analytical_acceleration = true,
                                              int cache_refresh_interval = 0);

            /**
             * Run a single diurnal cycle and compute time-averaged fluxes
             * Internal method used by multi-timescale simulation
             *
             * @param dt_seconds Timestep within diurnal cycle
             * @return True if successful
             */
            bool computeDiurnalAverages(double dt_seconds);

            /**
             * Evolve lag thickness for one epoch using averaged fluxes
             * Internal method used by multi-timescale simulation
             *
             * @param epoch_idx Current epoch index
             * @param epoch_duration Duration of epoch in seconds
             */
            void evolveEpoch(int epoch_idx, double epoch_duration);

            /**
             * Export multi-timescale evolution summary
             */
            void exportEvolutionSummary();
        };

    } // namespace exosphere
} // namespace ptm

#endif // PTM_EXOSPHERE_INTEGRATION_H