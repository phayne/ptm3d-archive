/*
 * Ice Crystallization Model for Europa Exosphere Simulation
 *
 * Tracks the competition between thermal recrystallization and radiation
 * amorphization of water ice on Europa's surface.
 *
 * Physics:
 * - Freshly deposited or radiation-processed ice is amorphous
 * - Thermal energy drives recrystallization (amorphous -> cubic crystalline)
 * - Magnetospheric ion bombardment drives amorphization (crystalline -> amorphous)
 * - The equilibrium crystalline fraction depends on temperature and radiation dose
 *
 * Key references:
 * - Mitchell et al. (2017) Icarus: Avrami kinetics for thermal recrystallization
 * - Famá et al. (2010) Surface Science: Radiation amorphization cross-sections
 * - Nordheim et al. (2022) PSJ: Magnetospheric ion flux maps for Europa
 * - Cartwright et al. (2025) PSJ: JWST observations of crystalline ice distribution
 */

#ifndef ICE_CRYSTALLIZATION_MODEL_H
#define ICE_CRYSTALLIZATION_MODEL_H

#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip>

namespace ptm {
namespace exosphere {

// Physical constants for crystallization model
namespace crystallization_constants {
    // Gas constant [J/(mol·K)]
    constexpr double R_GAS = 8.314;

    // Activation energy for crystallization [J/mol]
    // Mitchell et al. (2017), Table 2, Stage 2: 60-64 kJ/mol
    constexpr double DEFAULT_ACTIVATION_ENERGY = 62000.0;  // 62 kJ/mol (mean)

    // Pre-exponential factor K_0^(1/n) [s^-1]
    // Mitchell et al. (2017), Table 2
    constexpr double DEFAULT_PRE_EXPONENTIAL = 3.9e19;

    // Avrami exponent range (porosity-dependent)
    // Mitchell et al. (2017): n decreases from ~4 (compact) to ~1.5 (porous)
    constexpr double N_LOW_POROSITY = 4.0;    // Compact ice (porosity ~ 0)
    constexpr double N_HIGH_POROSITY = 1.5;   // Highly porous ice (porosity ~ 0.7)
    constexpr double POROSITY_MAX = 0.7;      // Maximum porosity for interpolation

    // Amorphization dose threshold [eV per (16 amu) molecule]
    // Famá et al. (2010), Leto & Baratta (2003)
    constexpr double DEFAULT_AMORPHIZATION_DOSE = 3.0;

    // Base radiation dose rate at trailing hemisphere equator [eV/(molecule·s)]
    // Calibrated from Cartwright et al. (2025): amorphization timescale ~ 15 days
    // dose_rate = E_a / tau = 3.0 / (15 * 86400) ≈ 2.3e-6
    constexpr double DEFAULT_BASE_DOSE_RATE = 2.3e-6;

    // Hemisphere dose contrast (trailing/leading)
    // Nordheim et al. (2022): S+ ion flux ratio ~ 5:1
    constexpr double DEFAULT_HEMISPHERE_DOSE_CONTRAST = 5.0;

    // Polar dose enhancement factor (pole/equator)
    // Nordheim et al. (2022): O+ and S+ at 1 keV show ~2.5x enhancement
    constexpr double DEFAULT_POLAR_DOSE_ENHANCEMENT = 2.5;

    // Default porosity values
    // Cartwright et al. (2025): millimeter observations, 40-70% at low latitudes
    constexpr double DEFAULT_EQUATORIAL_POROSITY = 0.50;
    constexpr double DEFAULT_POLAR_POROSITY = 0.20;

    // Minimum temperature for crystallization to occur [K]
    // Below this, rate is effectively zero (negligible on geological timescales)
    constexpr double MIN_CRYSTALLIZATION_TEMP = 80.0;

    // Seconds per year (for timescale conversions)
    constexpr double SECONDS_PER_YEAR = 365.25 * 24.0 * 3600.0;
}

/**
 * Configuration for ice crystallization model
 */
struct CrystallizationConfig {
    bool enabled = false;

    // --- Mitchell et al. (2017) Avrami kinetics parameters ---

    // Activation energy [J/mol]
    double activation_energy = crystallization_constants::DEFAULT_ACTIVATION_ENERGY;

    // Pre-exponential factor K_0^(1/n) [s^-1]
    double pre_exponential = crystallization_constants::DEFAULT_PRE_EXPONENTIAL;

    // Avrami exponent for compact ice (porosity ~ 0)
    double n_low_porosity = crystallization_constants::N_LOW_POROSITY;

    // Avrami exponent for porous ice (porosity ~ 0.7)
    double n_high_porosity = crystallization_constants::N_HIGH_POROSITY;

    // --- Famá et al. (2010) / Cartwright et al. (2025) amorphization parameters ---

    // Dose threshold for amorphization [eV/molecule]
    double amorphization_dose = crystallization_constants::DEFAULT_AMORPHIZATION_DOSE;

    // Base radiation dose rate at trailing equator [eV/(molecule·s)]
    double base_dose_rate = crystallization_constants::DEFAULT_BASE_DOSE_RATE;

    // Trailing/leading hemisphere dose contrast
    double hemisphere_dose_contrast = crystallization_constants::DEFAULT_HEMISPHERE_DOSE_CONTRAST;

    // Polar/equatorial dose enhancement factor
    double polar_dose_enhancement = crystallization_constants::DEFAULT_POLAR_DOSE_ENHANCEMENT;

    // --- Porosity parameters ---

    // Equatorial porosity (from millimeter observations, Cartwright et al. 2025)
    double equatorial_porosity = crystallization_constants::DEFAULT_EQUATORIAL_POROSITY;

    // Polar porosity
    double polar_porosity = crystallization_constants::DEFAULT_POLAR_POROSITY;

    // --- Initial state ---

    // Initial crystalline fraction (0 = fully amorphous, 1 = fully crystalline)
    double initial_crystalline_fraction = 0.0;

    // Minimum temperature for crystallization [K]
    double min_crystallization_temp = crystallization_constants::MIN_CRYSTALLIZATION_TEMP;
};


/**
 * Ice Crystallization Model
 *
 * Tracks the crystalline fraction of surface ice per mesh face,
 * evolving under the competition between thermal recrystallization
 * and radiation-driven amorphization.
 */
class IceCrystallizationModel {
private:
    CrystallizationConfig config_;
    size_t num_faces_;

    // Pre-computed per-face values
    std::vector<double> radiation_dose_rate_;  // [eV/(molecule·s)] per face
    std::vector<double> porosity_;             // [0-1] per face

public:
    /**
     * Constructor
     * @param num_faces Number of mesh faces
     * @param config Model configuration
     */
    explicit IceCrystallizationModel(size_t num_faces,
                                      const CrystallizationConfig& config = CrystallizationConfig())
        : config_(config), num_faces_(num_faces) {
        radiation_dose_rate_.resize(num_faces_, 0.0);
        porosity_.resize(num_faces_, config_.equatorial_porosity);
    }

    // --- Initialization ---

    /**
     * Initialize porosity distribution from face coordinates
     * Uses latitude-dependent model calibrated to Cartwright et al. (2025) observations
     *
     * @param latitudes Face center latitudes [degrees]
     */
    void initializePorosity(const std::vector<double>& latitudes) {
        for (size_t i = 0; i < num_faces_; ++i) {
            double abs_lat = std::abs(latitudes[i]);
            double t = abs_lat / 90.0;  // 0 at equator, 1 at pole

            // Linear interpolation between equatorial and polar porosity
            porosity_[i] = config_.equatorial_porosity
                         + (config_.polar_porosity - config_.equatorial_porosity) * t;

            // Clamp to valid range
            porosity_[i] = std::clamp(porosity_[i], 0.05, 0.70);
        }
    }

    /**
     * Initialize radiation dose rate distribution from face coordinates
     * Parametric model calibrated to Nordheim et al. (2022) ion flux maps
     * and Cartwright et al. (2025) ~15-day amorphization timescale
     *
     * @param latitudes Face center latitudes [degrees]
     * @param longitudes Face center longitudes [degrees, 0-360 east]
     */
    void initializeRadiationDoseRate(const std::vector<double>& latitudes,
                                     const std::vector<double>& longitudes) {
        for (size_t i = 0; i < num_faces_; ++i) {
            radiation_dose_rate_[i] = computeRadiationDoseRate(latitudes[i], longitudes[i]);
        }
    }

    /**
     * Initialize crystalline fraction for all faces
     * @param crystalline_fraction Vector to initialize (must be pre-allocated)
     */
    void initializeCrystallineFraction(std::vector<double>& crystalline_fraction) const {
        for (size_t i = 0; i < num_faces_; ++i) {
            crystalline_fraction[i] = config_.initial_crystalline_fraction;
        }
    }

    // --- Core Physics ---

    /**
     * Compute Avrami exponent as function of porosity
     * Mitchell et al. (2017): n decreases from ~4 (compact) to ~1.5 (porous)
     *
     * @param porosity Volume fraction of pore space [0-1]
     * @return Avrami exponent n
     */
    inline double avramiExponent(double porosity) const {
        double t = std::clamp(porosity / crystallization_constants::POROSITY_MAX, 0.0, 1.0);
        return config_.n_low_porosity
             + (config_.n_high_porosity - config_.n_low_porosity) * t;
    }

    /**
     * Compute thermal crystallization rate [1/s]
     * Mitchell et al. (2017) Avrami kinetics with Arrhenius temperature dependence
     *
     * K_eff = n * K_0^(1/n) * exp(-Delta_H / (R*T))
     *
     * @param temperature_K Surface temperature [K]
     * @param porosity Volume fraction of pore space [0-1]
     * @return Effective first-order crystallization rate [1/s]
     */
    inline double crystallizationRate(double temperature_K, double porosity) const {
        if (temperature_K < config_.min_crystallization_temp) return 0.0;

        double n = avramiExponent(porosity);
        double k_inv_n = config_.pre_exponential
                       * std::exp(-config_.activation_energy
                                / (crystallization_constants::R_GAS * temperature_K));

        // Effective first-order rate from Avrami kinetics
        // For the linearized ODE d(phi)/dt = k_eff * (1 - phi), k_eff ≈ n * k^(1/n)
        return n * k_inv_n;
    }

    /**
     * Compute radiation amorphization rate [1/s]
     * Based on Famá et al. (2010): rate = dose_rate / E_amorphization
     *
     * @param dose_rate Energy deposition rate [eV/(molecule·s)]
     * @return Amorphization rate [1/s]
     */
    inline double amorphizationRate(double dose_rate) const {
        if (config_.amorphization_dose <= 0.0) return 0.0;
        return dose_rate / config_.amorphization_dose;
    }

    /**
     * Compute radiation dose rate at a given location
     * Parametric model from Nordheim et al. (2022) flux patterns
     *
     * @param latitude_deg Latitude [degrees, -90 to +90]
     * @param longitude_deg East longitude [degrees, 0-360]
     * @return Dose rate [eV/(molecule·s)]
     */
    double computeRadiationDoseRate(double latitude_deg, double longitude_deg) const {
        // Longitude factor: trailing hemisphere enhanced
        // Trailing apex at 90°E (= 270°W in IAU west-positive). Cassidy et al.
        // (2013) and Nordheim et al. (2022) place the trailing apex 90° east of
        // sub-Jupiter; the run's mesh.ply (IAU east-positive from SPICE) and
        // Galileo Bond-albedo map are consistent with this convention. The
        // previous value of 270.0 here was 180° offset from the data and has
        // been corrected.
        // Nordheim et al. (2022): S+ trailing/leading ratio ~5:1
        double trailing_angle_rad = (longitude_deg - 90.0) * M_PI / 180.0;
        double cos_trailing = std::cos(trailing_angle_rad);

        // Map cos from [-1, +1] to [1/contrast, 1]
        double min_factor = 1.0 / config_.hemisphere_dose_contrast;
        double lon_factor = min_factor + (1.0 - min_factor) * std::max(0.0, (cos_trailing + 1.0) / 2.0);

        // Latitude factor: polar enhancement
        // Nordheim et al. (2022): O+ at 1 keV shows ~2.5x pole/equator
        double abs_lat = std::abs(latitude_deg);
        double lat_factor = 1.0 + (config_.polar_dose_enhancement - 1.0)
                          * std::pow(abs_lat / 90.0, 1.5);

        return config_.base_dose_rate * lon_factor * lat_factor;
    }

    /**
     * Evolve crystalline fraction for one face over a timestep
     *
     * Solves the competition ODE analytically:
     *   d(phi_C)/dt = k_cryst * (1 - phi_C) - k_amorph * phi_C
     *
     * Solution: phi_C(t) = phi_eq + (phi_C(0) - phi_eq) * exp(-(k_cryst + k_amorph)*dt)
     * where phi_eq = k_cryst / (k_cryst + k_amorph)
     *
     * @param dt Time step [s]
     * @param temperature_K Surface temperature [K]
     * @param face_idx Face index (for pre-computed porosity and dose rate)
     * @param crystalline_fraction Current crystalline fraction [0-1], modified in place
     */
    void evolveCrystallinity(double dt, double temperature_K, size_t face_idx,
                             double& crystalline_fraction) const {
        double porosity = porosity_[face_idx];
        double dose_rate = radiation_dose_rate_[face_idx];

        double k_cryst = crystallizationRate(temperature_K, porosity);
        double k_amorph = amorphizationRate(dose_rate);

        double k_total = k_cryst + k_amorph;
        if (k_total < 1e-30) return;  // No change if both rates are negligible

        // Equilibrium crystalline fraction
        double phi_eq = k_cryst / k_total;

        // Analytical solution for the linear ODE
        double decay = std::exp(-k_total * dt);
        crystalline_fraction = phi_eq + (crystalline_fraction - phi_eq) * decay;

        // Clamp to valid range
        crystalline_fraction = std::clamp(crystalline_fraction, 0.0, 1.0);
    }

    /**
     * Evolve crystallinity for all faces
     *
     * @param dt Time step [s]
     * @param temperatures Per-face surface temperatures [K]
     * @param crystalline_fraction Per-face crystalline fractions [0-1], modified in place
     */
    void evolveAllFaces(double dt, const std::vector<double>& temperatures,
                        std::vector<double>& crystalline_fraction) const {
        #pragma omp parallel for
        for (size_t i = 0; i < num_faces_; ++i) {
            evolveCrystallinity(dt, temperatures[i], i, crystalline_fraction[i]);
        }
    }

    /**
     * Evolve crystallinity for all faces using a pre-computed crystallization
     * rate per face — the time-averaged Avrami rate <k_cryst(T(t))>.
     *
     * This is the correct way to apply the analytical D34 ODE solution over
     * an epoch when temperature varies on a much faster timescale than the
     * epoch (the multi-timescale operator-splitting case). Calling
     * evolveAllFaces with the diurnal-mean temperature instead is biased by
     * Jensen's inequality because k_cryst is exponential in T:
     * &lt;k_cryst(T)&gt; can exceed k_cryst(&lt;T&gt;) by orders of magnitude near
     * Europa's crystallization knife-edge (~125 K).
     *
     * @param dt Time step [s] (typically one full epoch)
     * @param k_cryst_per_face Per-face time-averaged crystallization rate [s^-1]
     * @param crystalline_fraction Per-face crystalline fractions [0-1], modified in place
     */
    void evolveAllFacesWithRate(double dt,
                                const std::vector<double>& k_cryst_per_face,
                                std::vector<double>& crystalline_fraction) const {
        #pragma omp parallel for
        for (size_t i = 0; i < num_faces_; ++i) {
            double k_cryst = k_cryst_per_face[i];
            double k_amorph = amorphizationRate(radiation_dose_rate_[i]);
            double k_total = k_cryst + k_amorph;
            if (k_total < 1e-30) continue;

            double phi_eq = k_cryst / k_total;
            double decay = std::exp(-k_total * dt);
            double phi = phi_eq + (crystalline_fraction[i] - phi_eq) * decay;
            crystalline_fraction[i] = std::clamp(phi, 0.0, 1.0);
        }
    }

    /**
     * Apply depositional amorphization
     * Freshly deposited ice at Europa temperatures is amorphous.
     * The crystalline fraction is diluted by mass-weighted mixing.
     *
     * @param deposited_mass Mass of newly deposited ice [kg/m^2]
     * @param existing_ice_mass Existing ice mass before deposition [kg/m^2]
     * @param crystalline_fraction Current crystalline fraction, modified in place
     */
    static void applyDepositionalAmorphization(double deposited_mass,
                                               double existing_ice_mass,
                                               double& crystalline_fraction) {
        if (deposited_mass <= 0.0 || existing_ice_mass <= 0.0) return;

        // Deposited ice is fully amorphous (crystalline_fraction = 0)
        // New fraction is mass-weighted average
        double total_mass = existing_ice_mass + deposited_mass;
        crystalline_fraction = crystalline_fraction * existing_ice_mass / total_mass;
    }

    // --- Accessors ---

    const CrystallizationConfig& getConfig() const { return config_; }
    const std::vector<double>& getPorosity() const { return porosity_; }
    const std::vector<double>& getRadiationDoseRate() const { return radiation_dose_rate_; }

    double getPorosity(size_t face_idx) const { return porosity_[face_idx]; }
    double getRadiationDoseRate(size_t face_idx) const { return radiation_dose_rate_[face_idx]; }

    /**
     * Compute crystallization timescale [years] for a face
     * @param temperature_K Surface temperature [K]
     * @param face_idx Face index
     * @return Crystallization timescale in years (1/k_cryst), or -1 if rate is zero
     */
    double crystallizationTimescale_yr(double temperature_K, size_t face_idx) const {
        double k = crystallizationRate(temperature_K, porosity_[face_idx]);
        if (k < 1e-30) return -1.0;
        return 1.0 / (k * crystallization_constants::SECONDS_PER_YEAR);
    }

    /**
     * Compute amorphization timescale [years] for a face
     * @param face_idx Face index
     * @return Amorphization timescale in years (1/k_amorph)
     */
    double amorphizationTimescale_yr(size_t face_idx) const {
        double k = amorphizationRate(radiation_dose_rate_[face_idx]);
        if (k < 1e-30) return -1.0;
        return 1.0 / (k * crystallization_constants::SECONDS_PER_YEAR);
    }

    // --- Diagnostics ---

    /**
     * Print model configuration
     */
    void printConfig() const {
        std::cout << "\n=== Ice Crystallization Model ===" << std::endl;
        std::cout << "  Activation energy: " << config_.activation_energy / 1000.0 << " kJ/mol"
                  << " (Mitchell et al. 2017)" << std::endl;
        std::cout << "  Pre-exponential K0^(1/n): " << std::scientific
                  << config_.pre_exponential << " s^-1" << std::endl;
        std::cout << "  Avrami exponent range: " << std::fixed << std::setprecision(1)
                  << config_.n_high_porosity << " (porous) to "
                  << config_.n_low_porosity << " (compact)" << std::endl;
        std::cout << "  Amorphization dose: " << config_.amorphization_dose
                  << " eV/molecule (Fama et al. 2010)" << std::endl;
        std::cout << "  Base dose rate: " << std::scientific << config_.base_dose_rate
                  << " eV/(molecule s)" << std::endl;
        std::cout << "  Hemisphere dose contrast: " << std::fixed << std::setprecision(1)
                  << config_.hemisphere_dose_contrast << ":1 (Nordheim et al. 2022)" << std::endl;
        std::cout << "  Polar dose enhancement: " << config_.polar_dose_enhancement
                  << "x" << std::endl;
        std::cout << "  Porosity range: " << std::setprecision(0)
                  << (config_.polar_porosity * 100) << "% (poles) to "
                  << (config_.equatorial_porosity * 100) << "% (equator)" << std::endl;
        std::cout << "  Initial crystalline fraction: " << std::setprecision(2)
                  << (config_.initial_crystalline_fraction * 100) << "%" << std::endl;

        // Example timescales
        std::cout << "\n  Crystallization timescales (low porosity / high porosity):" << std::endl;
        double temps[] = {86.0, 100.0, 110.0, 120.0, 132.0};
        for (double T : temps) {
            double k_low = crystallizationRate(T, config_.polar_porosity);
            double k_high = crystallizationRate(T, config_.equatorial_porosity);
            auto format_time = [](double k) -> std::string {
                if (k < 1e-30) return ">> Gyr";
                double tau_s = 1.0 / k;
                double tau_yr = tau_s / (365.25 * 24 * 3600);
                if (tau_yr > 1e9) return "> Gyr";
                if (tau_yr > 1e6) {
                    char buf[32];
                    snprintf(buf, sizeof(buf), "%.1f Myr", tau_yr / 1e6);
                    return buf;
                }
                if (tau_yr > 1) {
                    char buf[32];
                    snprintf(buf, sizeof(buf), "%.1f yr", tau_yr);
                    return buf;
                }
                double tau_days = tau_s / 86400.0;
                if (tau_days > 1) {
                    char buf[32];
                    snprintf(buf, sizeof(buf), "%.1f days", tau_days);
                    return buf;
                }
                double tau_hr = tau_s / 3600.0;
                char buf[32];
                snprintf(buf, sizeof(buf), "%.1f hours", tau_hr);
                return buf;
            };
            std::cout << "    T=" << std::fixed << std::setprecision(0) << T << " K: "
                      << format_time(k_low) << " / " << format_time(k_high) << std::endl;
        }

        // Amorphization timescale at trailing equator
        double k_amorph = amorphizationRate(config_.base_dose_rate);
        if (k_amorph > 0) {
            double tau_amorph_days = 1.0 / (k_amorph * 86400.0);
            std::cout << "  Amorphization timescale (trailing equator): "
                      << std::setprecision(1) << tau_amorph_days << " days" << std::endl;
        }

        std::cout << std::fixed;
    }

    /**
     * Print crystallinity statistics
     */
    void printStatistics(const std::vector<double>& crystalline_fraction) const {
        double sum = 0.0, min_val = 1.0, max_val = 0.0;
        size_t n_crystalline = 0, n_amorphous = 0, n_mixed = 0;

        for (size_t i = 0; i < num_faces_; ++i) {
            double phi = crystalline_fraction[i];
            sum += phi;
            min_val = std::min(min_val, phi);
            max_val = std::max(max_val, phi);

            if (phi > 0.9) n_crystalline++;
            else if (phi < 0.1) n_amorphous++;
            else n_mixed++;
        }

        double mean = sum / num_faces_;

        std::cout << "\n=== Crystallinity Statistics ===" << std::endl;
        std::cout << "  Mean crystalline fraction: " << std::fixed << std::setprecision(2)
                  << (mean * 100) << "%" << std::endl;
        std::cout << "  Range: " << (min_val * 100) << "% - " << (max_val * 100) << "%" << std::endl;
        std::cout << "  Mostly crystalline (>90%): " << n_crystalline << " faces" << std::endl;
        std::cout << "  Mostly amorphous (<10%): " << n_amorphous << " faces" << std::endl;
        std::cout << "  Mixed: " << n_mixed << " faces" << std::endl;
    }
};

} // namespace exosphere
} // namespace ptm

#endif // ICE_CRYSTALLIZATION_MODEL_H
