/*
 * Thermally Active Layer Ice Model for Europa Exosphere Simulation
 *
 * Tracks ice concentration within a fixed thermally active layer (thermal skin depth).
 *
 * Physical model:
 * - The thermally active layer has a fixed thickness defined by the thermal skin depth
 * - Ice sublimates uniformly from this layer, increasing the non-ice concentration
 * - Sublimation/sputtering rates scale with ice fraction (1 - nonice_concentration),
 *   providing self-limiting depletion as ice is consumed
 * - When non-ice concentration reaches 1.0, no ice remains and sublimation stops
 * - Deposited ice fills void space, decreasing non-ice concentration
 *
 * Key simplifications:
 * - Ice temperature = surface temperature (from thermal model)
 * - No vertical stratification within the thermally active layer
 */

#ifndef ICE_LAYER_MODEL_H
#define ICE_LAYER_MODEL_H

#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip>

namespace ptm {
namespace exosphere {

// Physical constants for thermally active layer model
namespace layer_constants {
    // Ice density [kg/m³]
    constexpr double ICE_DENSITY = 917.0;

    // Non-ice material (dust/regolith) density [kg/m³]
    constexpr double DUST_DENSITY = 1500.0;

    // Default thermal skin depth [m]
    // This defines the thermally active layer thickness
    // Typical values for Europa: 1-10 cm depending on thermal properties
    constexpr double DEFAULT_THERMAL_SKIN_DEPTH = 0.05;  // 5 cm

    // Default initial non-ice concentration (volume fraction)
    // 0 = pure ice, 1 = no ice
    constexpr double DEFAULT_INITIAL_NONICE_CONCENTRATION = 0.01;  // 1% non-ice initially

    // Minimum concentration change to track (numerical threshold)
    constexpr double MIN_CONCENTRATION_CHANGE = 1e-12;

    // Maximum non-ice concentration (when all ice is gone)
    constexpr double MAX_NONICE_CONCENTRATION = 1.0;
}

/**
 * Configuration for thermally active layer model
 */
struct IceLayerConfig {
    // Thermal skin depth (thermally active layer thickness) [m]
    double thermal_skin_depth = layer_constants::DEFAULT_THERMAL_SKIN_DEPTH;

    // Initial non-ice concentration (volume fraction, 0-1)
    double initial_nonice_concentration = layer_constants::DEFAULT_INITIAL_NONICE_CONCENTRATION;

    // Constructor with defaults
    IceLayerConfig() = default;
};

/**
 * Per-face ice layer state
 *
 * Units are documented for each field.
 */
struct IceLayerState {
    // Primary state: non-ice concentration in thermally active layer
    // Range: 0 (pure ice) to 1 (no ice remaining)
    std::vector<double> nonice_concentration;  // [dimensionless, 0-1]

    // Ice mass per unit area in thermally active layer [kg/m²]
    // Derived from: ice_mass = (1 - nonice_concentration) * thermal_skin_depth * ICE_DENSITY
    std::vector<double> ice_mass;  // [kg/m²]

    // Cumulative tracking (for analysis)
    std::vector<double> cumulative_sublimated;  // Total ice lost to sublimation [kg/m²]
    std::vector<double> cumulative_sputtered;   // Total ice lost to sputtering [kg/m²]
    std::vector<double> cumulative_deposited;   // Total ice gained from deposition [kg/m²]
    std::vector<double> cumulative_escaped;     // Total mass escaped to space [kg/m²]

    // Ice crystallinity state (managed by IceCrystallizationModel)
    std::vector<double> crystalline_fraction;    // [0-1], 0 = fully amorphous, 1 = fully crystalline
    std::vector<double> porosity;                // [0-1], volume fraction of pore space
    std::vector<double> radiation_dose_rate;     // [eV/(molecule·s)] energy deposition rate
    std::vector<double> cumulative_dose;         // [eV/molecule] total accumulated radiation dose

    explicit IceLayerState(size_t num_faces) {
        nonice_concentration.resize(num_faces, 0.0);
        ice_mass.resize(num_faces, 0.0);

        cumulative_sublimated.resize(num_faces, 0.0);
        cumulative_sputtered.resize(num_faces, 0.0);
        cumulative_deposited.resize(num_faces, 0.0);
        cumulative_escaped.resize(num_faces, 0.0);

        crystalline_fraction.resize(num_faces, 0.0);
        porosity.resize(num_faces, 0.0);
        radiation_dose_rate.resize(num_faces, 0.0);
        cumulative_dose.resize(num_faces, 0.0);
    }

    void reset() {
        std::fill(nonice_concentration.begin(), nonice_concentration.end(), 0.0);
        std::fill(ice_mass.begin(), ice_mass.end(), 0.0);
        std::fill(cumulative_sublimated.begin(), cumulative_sublimated.end(), 0.0);
        std::fill(cumulative_sputtered.begin(), cumulative_sputtered.end(), 0.0);
        std::fill(cumulative_deposited.begin(), cumulative_deposited.end(), 0.0);
        std::fill(cumulative_escaped.begin(), cumulative_escaped.end(), 0.0);
        std::fill(crystalline_fraction.begin(), crystalline_fraction.end(), 0.0);
        // Note: porosity and radiation_dose_rate are NOT reset - they are set during initialization
        std::fill(cumulative_dose.begin(), cumulative_dose.end(), 0.0);
    }
};

/**
 * Thermally Active Layer Ice Model
 *
 * Manages ice concentration within a fixed thermally active layer.
 * Sublimation and sputtering increase non-ice concentration.
 * Deposition fills void space, decreasing non-ice concentration.
 */
class IceLayerModel {
private:
    IceLayerConfig config_;
    size_t num_faces_;

    // Maximum ice mass per unit area in the thermally active layer [kg/m²]
    // This is the ice mass when concentration = 0 (pure ice)
    double max_ice_mass_per_area_;

    // Exposed-area availability (--ice-fraction-from-albedo=hapke-intimate).
    // 0 (default) = legacy: availability is the ice VOLUME fraction (1 - c).
    // r > 0 = availability is the exposed ice AREA fraction
    //   f = phi / (phi + r (1 - phi)),  phi = 1 - c,  r = D_ice / D_nonice.
    double exposed_area_grain_ratio_ = 0.0;

public:
    /**
     * Constructor
     * @param num_faces Number of mesh faces
     * @param config Model configuration
     */
    explicit IceLayerModel(size_t num_faces, const IceLayerConfig& config = IceLayerConfig())
        : config_(config), num_faces_(num_faces) {
        // Compute maximum ice mass per unit area in the layer
        max_ice_mass_per_area_ = config_.thermal_skin_depth * layer_constants::ICE_DENSITY;
    }

    /**
     * Initialize ice distribution
     * Sets initial non-ice concentration for all faces
     *
     * @param state Ice layer state to initialize
     * @param initial_ice_mass Initial ice mass per face [kg/m²] (from exosphere model)
     */
    void initialize(IceLayerState& state, const std::vector<double>& initial_ice_mass) {
        if (initial_ice_mass.size() != num_faces_) {
            std::cerr << "IceLayerModel: size mismatch in initialize!" << std::endl;
            return;
        }

        state.reset();

        #pragma omp parallel for
        for (size_t i = 0; i < num_faces_; ++i) {
            // If there's initial ice, set concentration based on how much ice there is
            // relative to the maximum possible in the thermal layer
            if (initial_ice_mass[i] > 0) {
                // Ice mass determines ice fraction in the layer
                double ice_fraction = std::min(initial_ice_mass[i] / max_ice_mass_per_area_, 1.0);
                // Non-ice concentration is 1 minus ice fraction, but at least the initial dust content
                state.nonice_concentration[i] = std::max(
                    1.0 - ice_fraction,
                    config_.initial_nonice_concentration
                );
            } else {
                // No ice: non-ice concentration = 1
                state.nonice_concentration[i] = layer_constants::MAX_NONICE_CONCENTRATION;
            }

            // Update ice mass from concentration
            updateIceMass(state, i);
        }
    }

    /**
     * Initialize with uniform non-ice concentration
     * @param state Ice layer state to initialize
     * @param nonice_conc Initial non-ice concentration (0-1)
     */
    void initializeUniform(IceLayerState& state, double nonice_conc) {
        state.reset();

        double conc = std::clamp(nonice_conc, 0.0, layer_constants::MAX_NONICE_CONCENTRATION);

        #pragma omp parallel for
        for (size_t i = 0; i < num_faces_; ++i) {
            state.nonice_concentration[i] = conc;
            updateIceMass(state, i);
        }
    }

    /**
     * Initialize from a per-face non-ice concentration field.
     *
     * Used to seed the ice-layer state from an external map (e.g. derived from
     * the observed Galileo Bond-albedo distribution via
     * CompositionAlbedoMapper::computeConcentrationsFromAlbedoMap). Each
     * concentration is clamped to [0, MAX_NONICE_CONCENTRATION], the per-face
     * ice mass is then derived from the remaining ice fraction.
     *
     * @param state          Ice layer state to initialize
     * @param concentrations Per-face initial non-ice concentrations [0-1]
     */
    void initializeFromConcentrations(IceLayerState& state,
                                      const std::vector<double>& concentrations) {
        if (concentrations.size() != num_faces_) {
            std::cerr << "IceLayerModel::initializeFromConcentrations: size mismatch ("
                      << concentrations.size() << " vs " << num_faces_ << ")"
                      << std::endl;
            return;
        }

        state.reset();

        #pragma omp parallel for
        for (size_t i = 0; i < num_faces_; ++i) {
            state.nonice_concentration[i] = std::clamp(
                concentrations[i], 0.0, layer_constants::MAX_NONICE_CONCENTRATION);
            updateIceMass(state, i);
        }
    }

    /**
     * Update ice mass from non-ice concentration for a single face
     * @param state Ice layer state
     * @param face_idx Face index
     */
    void updateIceMass(IceLayerState& state, size_t face_idx) const {
        double ice_fraction = 1.0 - state.nonice_concentration[face_idx];
        state.ice_mass[face_idx] = ice_fraction * max_ice_mass_per_area_;
    }

    /**
     * Update ice mass for all faces
     * @param state Ice layer state
     */
    void updateAllIceMass(IceLayerState& state) const {
        #pragma omp parallel for
        for (size_t i = 0; i < num_faces_; ++i) {
            updateIceMass(state, i);
        }
    }

    /**
     * Check if ice is available at a face (for sublimation/sputtering)
     * @param state Ice layer state
     * @param face_idx Face index
     * @return true if ice is available (concentration < 1)
     */
    bool hasIce(const IceLayerState& state, size_t face_idx) const {
        return state.nonice_concentration[face_idx] < layer_constants::MAX_NONICE_CONCENTRATION;
    }

    /**
     * Get ice availability factor as continuous ice fraction [0, 1]
     * Sublimation/sputtering scale proportionally with exposed ice area.
     * As non-ice material accumulates, less surface area is available for
     * sublimation, creating self-limiting behavior at depleted faces.
     * @param state Ice layer state
     * @param face_idx Face index
     * @return (1 - nonice_concentration): 1.0 for pure ice, 0.0 for fully depleted
     */
    double getIceAvailabilityFactor(const IceLayerState& state, size_t face_idx) const {
        double nonice = state.nonice_concentration[face_idx];
        if (nonice >= layer_constants::MAX_NONICE_CONCENTRATION) return 0.0;
        if (exposed_area_grain_ratio_ > 0.0) {
            return exposedAreaFraction(1.0 - nonice);
        }
        return 1.0 - nonice;
    }

    /**
     * Switch availability from ice volume fraction to exposed ice area fraction.
     * @param grain_ratio r = D_ice / D_nonice (> 0); 0 restores the legacy
     *        volume-fraction availability.
     */
    void setExposedAreaGrainRatio(double grain_ratio) {
        exposed_area_grain_ratio_ = std::max(0.0, grain_ratio);
    }

    /** True when availability is the exposed ice area fraction. */
    bool usesExposedAreaFraction() const { return exposed_area_grain_ratio_ > 0.0; }

    /** Grain-size ratio r = D_ice / D_nonice of the exposed-area mode (0 = off). */
    double getExposedAreaGrainRatio() const { return exposed_area_grain_ratio_; }

    /**
     * Exposed ice area fraction for ice volume fraction phi (geometric optics,
     * equal Q_ext): f = phi / (phi + r (1 - phi)). Requires the exposed-area mode.
     */
    double exposedAreaFraction(double phi) const {
        const double r = exposed_area_grain_ratio_;
        phi = std::clamp(phi, 0.0, 1.0);
        const double den = phi + r * (1.0 - phi);
        return (den > 0.0) ? phi / den : 0.0;
    }

    /**
     * Apply ice loss from sublimation
     * Increases non-ice concentration as ice is removed from the layer
     *
     * @param state Ice layer state (modified in place)
     * @param sublimation_flux Mass flux from sublimation [kg/(m²·s)] per face
     * @param dt Time step [s]
     */
    void applySublimation(IceLayerState& state,
                          const std::vector<double>& sublimation_flux,
                          double dt) {
        #pragma omp parallel for
        for (size_t i = 0; i < num_faces_; ++i) {
            // Only sublimate if ice is present
            if (!hasIce(state, i)) continue;

            double ice_loss = sublimation_flux[i] * dt;  // [kg/m²]

            // Can't lose more ice than available
            ice_loss = std::min(ice_loss, state.ice_mass[i]);

            if (ice_loss > layer_constants::MIN_CONCENTRATION_CHANGE * max_ice_mass_per_area_) {
                // Track cumulative loss
                state.cumulative_sublimated[i] += ice_loss;

                // Compute new non-ice concentration
                // Ice lost -> void space created -> non-ice concentration increases
                double conc_increase = ice_loss / max_ice_mass_per_area_;
                state.nonice_concentration[i] += conc_increase;

                // Clamp to maximum
                if (state.nonice_concentration[i] >= layer_constants::MAX_NONICE_CONCENTRATION) {
                    state.nonice_concentration[i] = layer_constants::MAX_NONICE_CONCENTRATION;
                }

                // Update ice mass
                updateIceMass(state, i);
            }
        }
    }

    /**
     * Apply ice loss from sputtering
     * Similar to sublimation: increases non-ice concentration
     *
     * @param state Ice layer state (modified in place)
     * @param sputtering_flux Mass flux from sputtering [kg/(m²·s)] per face
     * @param escape_fraction Fraction of sputtered ice that escapes (vs redeposits)
     * @param dt Time step [s]
     */
    void applySputtering(IceLayerState& state,
                         const std::vector<double>& sputtering_flux,
                         double escape_fraction,
                         double dt) {
        #pragma omp parallel for
        for (size_t i = 0; i < num_faces_; ++i) {
            // Only sputter if ice is present
            if (!hasIce(state, i)) continue;

            double ice_loss = sputtering_flux[i] * dt;  // [kg/m²]

            // Can't lose more ice than available
            ice_loss = std::min(ice_loss, state.ice_mass[i]);

            if (ice_loss > layer_constants::MIN_CONCENTRATION_CHANGE * max_ice_mass_per_area_) {
                // Track cumulative loss
                state.cumulative_sputtered[i] += ice_loss;
                state.cumulative_escaped[i] += ice_loss * escape_fraction;

                // Compute new non-ice concentration
                double conc_increase = ice_loss / max_ice_mass_per_area_;
                state.nonice_concentration[i] += conc_increase;

                // Clamp to maximum
                if (state.nonice_concentration[i] >= layer_constants::MAX_NONICE_CONCENTRATION) {
                    state.nonice_concentration[i] = layer_constants::MAX_NONICE_CONCENTRATION;
                }

                // Update ice mass
                updateIceMass(state, i);
            }
        }
    }

    /**
     * Apply ice deposition
     * Deposited ice fills void space, decreasing non-ice concentration
     *
     * @param state Ice layer state (modified in place)
     * @param deposition_flux Mass flux from deposition [kg/(m²·s)] per face
     * @param dt Time step [s]
     */
    void applyDeposition(IceLayerState& state,
                         const std::vector<double>& deposition_flux,
                         double dt) {
        #pragma omp parallel for
        for (size_t i = 0; i < num_faces_; ++i) {
            double ice_gain = deposition_flux[i] * dt;  // [kg/m²]

            if (ice_gain > layer_constants::MIN_CONCENTRATION_CHANGE * max_ice_mass_per_area_) {
                // Track cumulative deposition
                state.cumulative_deposited[i] += ice_gain;

                // Deposited ice fills void space, decreasing non-ice concentration
                // But we can only fill up to the layer capacity
                double current_void_fraction = state.nonice_concentration[i];

                if (current_void_fraction > 0) {
                    // How much ice can we add before filling all voids?
                    double max_fillable_mass = current_void_fraction * max_ice_mass_per_area_;

                    // Actual ice that goes into the layer (can't exceed void space)
                    double ice_filled = std::min(ice_gain, max_fillable_mass);

                    // Decrease non-ice concentration
                    double conc_decrease = ice_filled / max_ice_mass_per_area_;
                    state.nonice_concentration[i] -= conc_decrease;

                    // Clamp to minimum (pure ice)
                    if (state.nonice_concentration[i] < 0) {
                        state.nonice_concentration[i] = 0.0;
                    }

                    // Update ice mass
                    updateIceMass(state, i);
                }
                // If concentration is already 0 (pure ice), excess deposition
                // would overflow the layer - in this simplified model we ignore overflow
            }
        }
    }

    /**
     * Configuration access
     */
    const IceLayerConfig& getConfig() const { return config_; }
    void setConfig(const IceLayerConfig& config) {
        config_ = config;
        // Recompute max ice mass
        max_ice_mass_per_area_ = config_.thermal_skin_depth * layer_constants::ICE_DENSITY;
    }

    /**
     * Set thermal skin depth [m]
     */
    void setThermalSkinDepth(double depth) {
        config_.thermal_skin_depth = std::max(depth, 0.001);  // Minimum 1 mm
        max_ice_mass_per_area_ = config_.thermal_skin_depth * layer_constants::ICE_DENSITY;
    }

    /**
     * Get thermal skin depth [m]
     */
    double getThermalSkinDepth() const { return config_.thermal_skin_depth; }

    /**
     * Get maximum ice mass per unit area [kg/m²]
     */
    double getMaxIceMassPerArea() const { return max_ice_mass_per_area_; }

    /**
     * Print configuration summary
     */
    void printConfig() const {
        std::cout << "\n=== Thermally Active Layer Ice Model ===" << std::endl;
        std::cout << "  Thermal skin depth: " << (config_.thermal_skin_depth * 100) << " cm" << std::endl;
        std::cout << "  Max ice mass per area: " << max_ice_mass_per_area_ << " kg/m²" << std::endl;
        std::cout << "  Initial non-ice concentration: "
                  << (config_.initial_nonice_concentration * 100) << "%" << std::endl;
        std::cout << "  Ice density: " << layer_constants::ICE_DENSITY << " kg/m³" << std::endl;
    }

    /**
     * Print statistics for current state
     */
    void printStatistics(const IceLayerState& state) const {
        double total_ice_mass = 0.0;
        double mean_concentration = 0.0;
        double max_concentration = 0.0;
        double min_concentration = 1.0;
        size_t faces_with_ice = 0;
        size_t faces_depleted = 0;

        for (size_t i = 0; i < num_faces_; ++i) {
            total_ice_mass += state.ice_mass[i];
            mean_concentration += state.nonice_concentration[i];
            max_concentration = std::max(max_concentration, state.nonice_concentration[i]);
            min_concentration = std::min(min_concentration, state.nonice_concentration[i]);

            if (hasIce(state, i)) {
                faces_with_ice++;
            } else {
                faces_depleted++;
            }
        }

        mean_concentration /= num_faces_;

        std::cout << "\n=== Ice Layer Statistics ===" << std::endl;
        std::cout << "  Total ice mass (summed): " << std::scientific << std::setprecision(3)
                  << total_ice_mass << " kg/m²" << std::endl;
        std::cout << "  Non-ice concentration:" << std::endl;
        std::cout << "    Mean: " << std::fixed << std::setprecision(4)
                  << (mean_concentration * 100) << "%" << std::endl;
        std::cout << "    Range: " << (min_concentration * 100) << "% - "
                  << (max_concentration * 100) << "%" << std::endl;
        std::cout << "  Faces with ice: " << faces_with_ice << " / " << num_faces_ << std::endl;
        std::cout << "  Faces depleted: " << faces_depleted << " / " << num_faces_ << std::endl;
        std::cout << std::fixed;
    }
};

} // namespace exosphere
} // namespace ptm

#endif // ICE_LAYER_MODEL_H
