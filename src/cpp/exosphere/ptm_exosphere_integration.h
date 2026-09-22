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
        class IceCrystallizationModel;
        struct CrystallizationConfig;
        class MultiTimescaleEvolution;
        struct MultiTimescaleConfig;
        class CompositionAlbedoMapper;
        struct CompositionAlbedoConfig;

        /**
         * Parameters of --ice-fraction-from-albedo=hapke-intimate.
         * Endmember Bond albedos and asymmetry parameters, the grain-size ratio
         * r = D_ice / D_nonice, the per-face Bond albedo map (the thermal run's
         * own <Body>_Albedo_res<N>.bin), the FITS map used to flag faces whose
         * baked value mixed in fill pixels, and the exposed ice fraction given
         * to those faces.
         */
        struct HapkeIceFractionConfig {
            double ice_albedo = 0.80;
            double nonice_albedo = 0.012;
            double g_ice = 0.0;
            double g_nonice = 0.0;
            double grain_ratio = 10.0;
            std::string albedo_map_path;     // per-face .bin (size_t count + double[count])
            std::string fill_fits_path;      // "" = sibling FITS next to the .bin; "none" = skip
            double fill_exposed_ice_fraction = 0.95;
        };

        /**
         * Face centers [m, on the sphere of radius R_Europa] and areas [m^2] from
         * a mesh in any length unit (EXO-02). Returns false with a message in
         * `error` if the geometry is unusable, including when the total area
         * differs from 4 pi R_Europa^2 by 1e-2 or more (relative; catches unit
         * errors and non-global meshes, admits coarse flat-faceted icospheres).
         */
        bool computeExosphereFaceGeometry(const Mesh& mesh,
                                          std::vector<std::array<double,3>>& centers,
                                          std::vector<double>& areas,
                                          std::vector<std::array<double,3>>* normals,
                                          std::string& error);

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

            // Ice crystallization model (optional, tracks amorphous/crystalline transition)
            std::unique_ptr<IceCrystallizationModel> crystallization_model_;
            bool crystallization_enabled_ = false;

            // Multi-timescale evolution (optional, for geological timescales)
            std::unique_ptr<MultiTimescaleEvolution> multiscale_evolution_;
            bool multiscale_enabled_ = false;

            // Export mode flag (determines temperature output format)
            bool multiscale_export_mode_ = false;

            // Albedo-composition coupling (optional, for thermal segregation feedback)
            bool albedo_coupling_enabled_ = false;
            std::unique_ptr<CompositionAlbedoMapper> albedo_mapper_;
            std::vector<double> current_face_albedos_;    // per-face current albedo
            std::vector<double> previous_face_albedos_;   // for change detection
            double initial_albedo_ = 0.5;                 // original albedo from PTM thermal sim

            // Per-face non-ice albedo endmembers from spatial map
            std::vector<double> nonice_albedo_map_;

            // Per-face initial albedos from PTM3D (for temperature scaling reference)
            std::vector<double> initial_face_albedos_;

            // hapke-intimate mode: availability is the exposed ice AREA fraction and
            // the diurnal averages are scaled by it before transport and caching.
            bool exposed_area_mode_ = false;
            // Per-face initialization class: 0 = inverted, 1 = fill (uniform),
            // 2 = clamped at f = 0, 3 = clamped at f = 1.
            std::vector<double> ice_fraction_init_class_;
            // Unscaled (pure-ice) diurnal fluxes, the cache base in this mode.
            std::vector<double> unscaled_sublimation_;
            std::vector<double> unscaled_sputtering_;

            // Albedo coupling internal methods
            double updateAlbedoFromComposition();
            void updateTemperaturesForAlbedoChange();

            // Private methods
            void calculateFaceProperties();
            double calculateGreatCircleDistance(int face_i, int face_j);
            double calculateBallisticProbability(double distance, double temperature);
            double calculateEscapeFraction(double temperature);
            void buildTransportMatrix(double avg_temperature);
            // (Never-defined besselK0/besselI0 member declarations removed:
            // they shadowed the shared ptm::exosphere::besselK0 inside this
            // class. Use exosphere_physics.h.)

            bool show_progress_ = true;

            // Convergence thresholds (set via CLI, applied to MultiTimescaleConfig)
            double convergence_threshold_ = -1.0;       // <0 means use default
            double flux_convergence_threshold_ = -1.0;   // <0 means use default

            // Diurnal accumulation stride: use every Nth timestep for averaging.
            // stride=1 uses all timesteps (default), stride=4 uses every 4th (~4x faster).
            int diurnal_accumulation_stride_ = 1;

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
             * Enable ice crystallization model
             * Tracks amorphous/crystalline ice transition using Mitchell et al. (2017)
             * Avrami kinetics and Nordheim et al. (2022) radiation dose model.
             * Requires ice layer model to be enabled first.
             *
             * @param config Crystallization model configuration
             */
            void enableCrystallizationModel(const CrystallizationConfig& config);

            /**
             * Disable ice layer model (use simple ice tracking)
             */
            void disableIceLayerModel();

            /**
             * Check if ice layer model is enabled
             */
            bool isIceLayerEnabled() const { return ice_layer_enabled_; }

            /**
             * Check if crystallization model is enabled
             */
            bool isCrystallizationEnabled() const { return crystallization_enabled_; }

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
             * Get ice fraction for a face (continuous ice availability)
             * @param face_idx Face index
             * @return Ice fraction [0, 1]: 1.0 = pure ice, 0.0 = fully depleted
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
            // Albedo-Composition Coupling Interface
            // =========================================================

            /**
             * Enable albedo-composition feedback coupling
             *
             * When enabled, non-ice concentration from the ice layer model is
             * mapped to surface albedo via linear mixing between ice and non-ice
             * endmembers. Temperature arrays are scaled to reflect the new albedo
             * using radiative equilibrium: T_new = T_old * ((1-A_new)/(1-A_old))^(1/4)
             *
             * This captures the positive feedback loop driving thermal segregation:
             * sublimation -> darker surface -> higher T -> more sublimation
             *
             * @param config Albedo coupling configuration (endmember values, threshold)
             */
            void enableAlbedoCoupling(const CompositionAlbedoConfig& config);

            /**
             * Disable albedo-composition coupling
             */
            void disableAlbedoCoupling();

            /**
             * Check if albedo coupling is enabled
             */
            bool isAlbedoCouplingEnabled() const { return albedo_coupling_enabled_; }

            /**
             * Get current albedo for a face
             * @param face_idx Face index
             * @return Current Bond albedo (composition-dependent if coupling enabled)
             */
            double getFaceAlbedo(int face_idx) const;

            /**
             * Set the initial (reference) albedo used by the PTM thermal simulation
             * This is needed to compute the temperature scaling ratio
             * @param albedo The Bond albedo used in the original thermal simulation
             */
            void setInitialAlbedo(double albedo) { initial_albedo_ = albedo; }

            /**
             * Load spatially-varying non-ice albedo endmembers from a FITS/binary map.
             * When set, the mixing model becomes per-face:
             *   A(c, face) = A_ice*(1-c) + A_map(face)*c
             * instead of using a uniform albedo_nonice scalar.
             *
             * @param map_path Path to albedo map file (FITS or binary)
             * @param maps_dir Directory for auto-discovery (used if map_path empty)
             */
            void loadNoniceAlbedoMap(const std::string& map_path,
                                     const std::string& maps_dir = "");

            /**
             * Pre-load per-face non-ice concentrations from the observed albedo
             * map so albedo-coupling runs start at "current Europa" rather than
             * uniform ice + 1000-yr spin-up. Requires loadNoniceAlbedoMap and
             * enableIceLayerModel to have been called first.
             *
             * @param percentile_low  Darkest percentile mapped to c=1 (e.g. 5)
             * @param percentile_high Brightest percentile mapped to c=0 (e.g. 95)
             */
            void initializeConcentrationsFromAlbedoMap(double percentile_low,
                                                       double percentile_high);

            /**
             * --ice-fraction-from-albedo=hapke-intimate. Invert the per-face Bond
             * albedo map for the ice extinction (exposed-area) fraction f with the
             * intimate Hapke mixture, convert to volume (phi = f r / (f r + 1 - f)),
             * seed the ice layer (c = 1 - phi), and switch the ice layer to
             * exposed-area availability. Faces whose map value mixed in fill pixels
             * get fill_exposed_ice_fraction. Fills albedo_config with the matching
             * Hapke forward model and enforces the forward/initialization gate
             * (max |A_forward(c_init) - A_map| < 1e-6 on inverted faces).
             * Requires enableIceLayerModel first.
             * @return false (with a message on stderr) on any failure
             */
            bool initializeConcentrationsFromHapkeIntimate(const HapkeIceFractionConfig& cfg,
                                                           CompositionAlbedoConfig& albedo_config);

            // =========================================================
            // Multi-Timescale Evolution Interface
            // =========================================================

            /**
             * Set convergence threshold for multi-timescale simulation
             * @param threshold Max concentration change per epoch for convergence
             */
            void setConvergenceThreshold(double threshold) { convergence_threshold_ = threshold; }

            /**
             * Set flux imbalance convergence threshold
             * @param threshold Flux imbalance fraction [0-1] for convergence (e.g. 0.01 = 1%)
             */
            void setFluxConvergenceThreshold(double threshold) { flux_convergence_threshold_ = threshold; }

            /**
             * Set diurnal accumulation stride for multi-timescale simulation
             * @param stride Use every Nth timestep (1 = all, 4 = every 4th, etc.)
             */
            void setDiurnalStride(int stride) { diurnal_accumulation_stride_ = std::max(1, stride); }

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