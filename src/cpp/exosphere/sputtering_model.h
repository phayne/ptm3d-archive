/*
 * Europa Magnetospheric Plasma Sputtering Model
 *
 * Physics-based model of ion sputtering at Europa's surface based on:
 * - Cassidy et al. (2013) - Magnetospheric ion sputtering spatial distribution
 * - Famá et al. (2008) - Low-energy ion sputtering yield formulas
 * - Teolis et al. (2017) - Temperature-dependent sputtering yields
 *
 * Key features:
 * - Spatially-varying ion flux (trailing hemisphere enhancement)
 * - Temperature-dependent sputtering yields
 * - Separate tracking of H2O, O2, and H2 production
 * - Configurable parameters for sensitivity studies
 */

#ifndef SPUTTERING_MODEL_H
#define SPUTTERING_MODEL_H

#include <vector>
#include <array>
#include <cmath>
#include <string>
#include <iostream>

namespace ptm {
namespace exosphere {

// Sputtering-related physical constants
namespace sputtering_constants {
    // Global sputtering rates from Cassidy et al. (2013)
    constexpr double GLOBAL_H2O_RATE = 2.0e27;    // molecules/s (total Europa)
    constexpr double GLOBAL_O2_RATE = 1.0e26;     // molecules/s
    constexpr double GLOBAL_H2_RATE = 2.0e26;     // molecules/s

    // Europa surface area for rate normalization
    constexpr double EUROPA_SURFACE_AREA = 3.09e13;  // m² (4πR²)

    // Molecular masses
    constexpr double H2O_MASS = 2.99151e-26;  // kg
    constexpr double O2_MASS = 5.3135e-26;    // kg
    constexpr double H2_MASS = 3.3474e-27;    // kg

    // Avogadro's number for conversions
    constexpr double AVOGADRO = 6.02214076e23;

    // Temperature dependence parameters (Teolis et al. 2017)
    constexpr double TEMP_ACTIVATION_ENERGY = 0.06;  // eV
    constexpr double TEMP_PREFACTOR = 1000.0;        // q0 parameter
    constexpr double BOLTZMANN_EV = 8.617333e-5;     // eV/K

    // Reference temperature for yield normalization
    constexpr double REFERENCE_TEMP = 100.0;  // K (typical Europa dayside)

    // Temperature below which sputtering yield is constant
    constexpr double LOW_TEMP_THRESHOLD = 80.0;  // K
}

/**
 * Configuration for sputtering model parameters
 */
struct SputteringConfig {
    // Enable/disable sputtering
    bool enabled = true;

    // Global rate scaling factor (for sensitivity studies)
    double rate_scale = 1.0;

    // Trailing/leading hemisphere contrast ratio
    // Based on Cassidy et al. (2013): trailing apex ~3x leading apex
    double hemisphere_contrast = 3.0;

    // Latitude dependence - polar enhancement factor for energetic ions
    // Energetic ions preferentially precipitate at higher latitudes
    double polar_enhancement = 1.2;

    // Temperature dependence enable (Teolis et al. 2017)
    bool temperature_dependent = true;

    // Fraction of sputtered material that escapes vs redeposits
    // From Cassidy: most H2O redeposits, but O2/H2 largely escape
    double h2o_escape_fraction = 0.1;   // 10% escapes
    double o2_escape_fraction = 0.9;    // 90% escapes
    double h2_escape_fraction = 0.95;   // 95% escapes

    // Species production ratios (Cassidy et al. 2013)
    // Relative to total sputtered mass
    double h2o_fraction = 0.85;  // Most sputtering produces H2O
    double o2_fraction = 0.10;   // Some O2 from radiolysis
    double h2_fraction = 0.05;   // Small H2 component

    // Minimum temperature for sputtering (below this, surface is "cold trapped")
    double min_sputtering_temp = 50.0;  // K

    // Constructor with defaults
    SputteringConfig() = default;

    // Convenience method to disable sputtering
    static SputteringConfig disabled() {
        SputteringConfig cfg;
        cfg.enabled = false;
        return cfg;
    }
};

/**
 * Sputtering state tracking per face
 */
struct SputteringState {
    std::vector<double> ion_flux;           // ions/(m²·s) - spatial distribution
    std::vector<double> sputtering_yield;   // molecules/ion (temperature-dependent)
    std::vector<double> h2o_sputter_flux;   // kg/(m²·s) - H2O sputtered
    std::vector<double> o2_production_flux; // kg/(m²·s) - O2 produced
    std::vector<double> h2_production_flux; // kg/(m²·s) - H2 produced
    std::vector<double> total_mass_loss;    // kg/(m²·s) - net mass removed
    std::vector<double> redeposition_flux;  // kg/(m²·s) - H2O that redeposits locally

    explicit SputteringState(size_t num_faces) {
        ion_flux.resize(num_faces, 0.0);
        sputtering_yield.resize(num_faces, 0.0);
        h2o_sputter_flux.resize(num_faces, 0.0);
        o2_production_flux.resize(num_faces, 0.0);
        h2_production_flux.resize(num_faces, 0.0);
        total_mass_loss.resize(num_faces, 0.0);
        redeposition_flux.resize(num_faces, 0.0);
    }

    void reset() {
        std::fill(ion_flux.begin(), ion_flux.end(), 0.0);
        std::fill(sputtering_yield.begin(), sputtering_yield.end(), 0.0);
        std::fill(h2o_sputter_flux.begin(), h2o_sputter_flux.end(), 0.0);
        std::fill(o2_production_flux.begin(), o2_production_flux.end(), 0.0);
        std::fill(h2_production_flux.begin(), h2_production_flux.end(), 0.0);
        std::fill(total_mass_loss.begin(), total_mass_loss.end(), 0.0);
        std::fill(redeposition_flux.begin(), redeposition_flux.end(), 0.0);
    }
};

/**
 * Main sputtering model class
 *
 * Computes spatially and temperature-dependent sputtering rates
 * based on magnetospheric plasma bombardment physics.
 */
class SputteringModel {
private:
    SputteringConfig config_;
    size_t num_faces_;

    // Pre-computed spatial distribution (normalized)
    std::vector<double> spatial_weight_;
    bool spatial_initialized_ = false;

    // Reference flux for normalization (ions/m²/s at trailing apex)
    double reference_ion_flux_ = 0.0;

    // Statistics
    double total_sputtered_mass_ = 0.0;
    double total_escaped_mass_ = 0.0;
    double total_redeposited_mass_ = 0.0;

public:
    /**
     * Constructor
     * @param num_faces Number of mesh faces
     * @param config Sputtering configuration
     */
    explicit SputteringModel(size_t num_faces, const SputteringConfig& config = SputteringConfig())
        : config_(config), num_faces_(num_faces) {
        spatial_weight_.resize(num_faces_, 1.0);
    }

    /**
     * Initialize spatial distribution based on face coordinates
     * Must be called before computing sputtering rates
     *
     * @param face_centers Face center positions [x, y, z] in body-fixed frame
     * @param face_latitudes Face latitudes in degrees (optional, computed if empty)
     * @param face_longitudes Face longitudes in degrees (optional, computed if empty)
     */
    void initializeSpatialDistribution(
        const std::vector<std::array<double, 3>>& face_centers,
        const std::vector<double>& face_latitudes = {},
        const std::vector<double>& face_longitudes = {});

    /**
     * Compute sputtering rates for current temperatures
     *
     * @param temperatures Surface temperatures [K] per face
     * @param face_areas Face areas [m²] per face
     * @param state Output sputtering state (modified in place)
     */
    void computeSputteringRates(
        const std::vector<double>& temperatures,
        const std::vector<double>& face_areas,
        SputteringState& state) const;

    /**
     * Apply sputtering mass loss to accumulated ice
     *
     * @param state Current sputtering state
     * @param accumulated_mass Ice mass per face [kg/m²] (modified in place)
     * @param face_areas Face areas [m²]
     * @param dt Time step [s]
     * @return Total mass lost to escape [kg]
     */
    double applySputteringMassLoss(
        const SputteringState& state,
        std::vector<double>& accumulated_mass,
        const std::vector<double>& face_areas,
        double dt);

    /**
     * Get the spatial flux weight for a face (for visualization/analysis)
     */
    double getSpatialWeight(size_t face_idx) const {
        return (face_idx < spatial_weight_.size()) ? spatial_weight_[face_idx] : 0.0;
    }

    /**
     * Configuration access
     */
    const SputteringConfig& getConfig() const { return config_; }
    void setConfig(const SputteringConfig& config) { config_ = config; }

    /**
     * Enable/disable sputtering at runtime
     */
    void setEnabled(bool enabled) { config_.enabled = enabled; }
    bool isEnabled() const { return config_.enabled; }

    /**
     * Check if spatial distribution has been initialized
     */
    bool isSpatialInitialized() const { return spatial_initialized_; }

    /**
     * Scale factor for rate adjustment
     */
    void setRateScale(double scale) { config_.rate_scale = scale; }
    double getRateScale() const { return config_.rate_scale; }

    /**
     * Statistics
     */
    double getTotalSputteredMass() const { return total_sputtered_mass_; }
    double getTotalEscapedMass() const { return total_escaped_mass_; }
    double getTotalRedeposited() const { return total_redeposited_mass_; }
    void resetStatistics() {
        total_sputtered_mass_ = 0.0;
        total_escaped_mass_ = 0.0;
        total_redeposited_mass_ = 0.0;
    }

    /**
     * Print configuration summary
     */
    void printConfig() const;

private:
    /**
     * Compute longitude-dependent flux factor
     * Trailing hemisphere (270°W) has peak flux, leading (90°W) has minimum
     *
     * @param longitude West longitude in degrees (0-360)
     * @return Flux factor (0.33 to 1.0 for default contrast=3)
     */
    double computeLongitudeFluxFactor(double longitude) const;

    /**
     * Compute latitude-dependent flux factor
     * Slight polar enhancement for energetic ion precipitation
     *
     * @param latitude Latitude in degrees (-90 to 90)
     * @return Flux factor (1.0 to polar_enhancement)
     */
    double computeLatitudeFluxFactor(double latitude) const;

    /**
     * Compute temperature-dependent sputtering yield enhancement
     * Based on Teolis et al. (2017): Y(T) = Y0 * [1 + q0*exp(-Q/kT)]
     *
     * @param temperature Surface temperature [K]
     * @return Yield enhancement factor (≥1.0)
     */
    double computeTemperatureYieldFactor(double temperature) const;

    /**
     * Convert latitude/longitude to west longitude convention
     * Europa uses west longitude (0-360°W) with trailing apex at 270°W
     */
    static double toWestLongitude(double lon_deg);
};

// ============================================================================
// Inline implementations for performance-critical functions
// ============================================================================

inline double SputteringModel::computeLongitudeFluxFactor(double longitude) const {
    // Trailing apex at 270°W has maximum flux
    // Leading apex at 90°W has minimum flux (~1/contrast of maximum)
    // Use cosine distribution centered on trailing apex

    double lon_rad = longitude * M_PI / 180.0;
    double trailing_apex_rad = 270.0 * M_PI / 180.0;

    // Cosine of angle from trailing apex (shifted so trailing = 0)
    double cos_angle = std::cos(lon_rad - trailing_apex_rad);

    // Map from [-1, 1] to [1/contrast, 1]
    // When cos_angle = 1 (at trailing), factor = 1
    // When cos_angle = -1 (at leading), factor = 1/contrast
    double min_factor = 1.0 / config_.hemisphere_contrast;
    double factor = min_factor + (1.0 - min_factor) * (cos_angle + 1.0) / 2.0;

    return factor;
}

inline double SputteringModel::computeLatitudeFluxFactor(double latitude) const {
    // Slight enhancement at poles due to energetic ion precipitation
    // Thermal plasma is more equatorial, energetic ions reach poles
    double abs_lat = std::abs(latitude);

    // Linear interpolation from equator (factor=1) to pole (factor=polar_enhancement)
    double factor = 1.0 + (config_.polar_enhancement - 1.0) * (abs_lat / 90.0);

    return factor;
}

inline double SputteringModel::computeTemperatureYieldFactor(double temperature) const {
    if (!config_.temperature_dependent || temperature <= 0.0) {
        return 1.0;
    }

    // Below threshold, yield is constant (cold surface)
    if (temperature < sputtering_constants::LOW_TEMP_THRESHOLD) {
        return 1.0;
    }

    // Teolis et al. (2017): Y(T) = Y0 * [1 + q0*exp(-Q/kT)]
    // Normalized so factor = 1 at reference temperature
    double q0 = sputtering_constants::TEMP_PREFACTOR;
    double Q = sputtering_constants::TEMP_ACTIVATION_ENERGY;
    double kT = sputtering_constants::BOLTZMANN_EV * temperature;
    double kT_ref = sputtering_constants::BOLTZMANN_EV * sputtering_constants::REFERENCE_TEMP;

    double yield_T = 1.0 + q0 * std::exp(-Q / kT);
    double yield_ref = 1.0 + q0 * std::exp(-Q / kT_ref);

    return yield_T / yield_ref;
}

inline double SputteringModel::toWestLongitude(double lon_deg) {
    // Convert to 0-360 west longitude
    double west_lon = std::fmod(-lon_deg + 360.0, 360.0);
    if (west_lon < 0) west_lon += 360.0;
    return west_lon;
}

} // namespace exosphere
} // namespace ptm

#endif // SPUTTERING_MODEL_H
