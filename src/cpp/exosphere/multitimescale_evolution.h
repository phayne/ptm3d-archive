/*
 * Multi-Timescale Ice Evolution via Operator Splitting
 *
 * Enables simulation of ice redistribution over geological timescales (kyr to Myr)
 * by separating the fast diurnal dynamics from slow concentration evolution.
 *
 * Algorithm:
 * 1. Run one full diurnal cycle to compute time-averaged fluxes
 * 2. Use averaged fluxes to evolve non-ice concentration over many diurnal cycles (one "epoch")
 * 3. Repeat for desired number of epochs
 *
 * Key approximations:
 * - Transport matrix is fixed (computed once at start)
 * - Ice presence determined by non-ice concentration (concentration < 1 means ice present)
 * - No sublimation suppression - ice sublimates at full rate when present
 *
 * References:
 * - Operator splitting: Strang (1968), LeVeque (2002)
 */

#ifndef MULTITIMESCALE_EVOLUTION_H
#define MULTITIMESCALE_EVOLUTION_H

#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <functional>
#include "ice_layer_model.h"

namespace ptm {
namespace exosphere {

// Physical constants for timescale calculations
namespace timescale_constants {
    // Europa diurnal period [seconds]
    constexpr double EUROPA_DIURNAL_PERIOD = 3.5512 * 24.0 * 3600.0;  // 3.55 days

    // Seconds per Earth year
    constexpr double SECONDS_PER_YEAR = 365.25 * 24.0 * 3600.0;

    // Default diurnal cycles per epoch (~ 1 Earth year)
    constexpr int DEFAULT_CYCLES_PER_EPOCH = 100;

    // Minimum epochs between progress reports (1 = every epoch, updates are cheap)
    constexpr int PROGRESS_INTERVAL = 1;

    // Convergence threshold for non-ice concentration change per epoch
    constexpr double CONVERGENCE_THRESHOLD = 1e-10;  // dimensionless
}

/**
 * Configuration for multi-timescale evolution
 */
struct MultiTimescaleConfig {
    // Number of diurnal cycles per epoch
    int cycles_per_epoch = timescale_constants::DEFAULT_CYCLES_PER_EPOCH;

    // Total number of epochs to simulate
    int num_epochs = 1000;  // ~1000 years with default cycles_per_epoch

    // Export interval (epochs between VTU exports, 0 = final only)
    int export_interval = 100;

    // Progress reporting
    bool show_progress = true;

    // Convergence threshold for early termination (max concentration change per epoch)
    double convergence_threshold = timescale_constants::CONVERGENCE_THRESHOLD;

    // Constructor with defaults
    MultiTimescaleConfig() = default;

    // Convenience: compute total simulation time
    double totalTimeYears() const {
        double epoch_duration = cycles_per_epoch * timescale_constants::EUROPA_DIURNAL_PERIOD;
        return num_epochs * epoch_duration / timescale_constants::SECONDS_PER_YEAR;
    }

    // Print configuration
    void print() const {
        std::cout << "\n=== Multi-Timescale Evolution Configuration ===" << std::endl;
        std::cout << "  Cycles per epoch: " << cycles_per_epoch << std::endl;
        std::cout << "  Number of epochs: " << num_epochs << std::endl;
        std::cout << "  Total simulation time: " << std::scientific << std::setprecision(2)
                  << totalTimeYears() << " years" << std::endl;
        std::cout << "  Export interval: " << export_interval << " epochs" << std::endl;
        std::cout << "  Convergence threshold: " << convergence_threshold << std::endl;
        std::cout << std::fixed;
    }
};

/**
 * Time-averaged flux data from diurnal cycle
 * These are the "slow" variables that drive long-term evolution
 *
 * Units documented for each field.
 */
struct DiurnalAverages {
    // Current effective fluxes (diurnal averages)
    std::vector<double> avg_sublimation_flux;   // Time-averaged emission [kg/(m²·s)]
    std::vector<double> avg_deposition_flux;    // Time-averaged deposition [kg/(m²·s)]
    std::vector<double> avg_sputtering_flux;    // Time-averaged sputtering [kg/(m²·s)]
    std::vector<double> avg_net_flux;           // Time-averaged net change [kg/(m²·s)]
    std::vector<double> avg_escape_flux;        // Time-averaged escape [kg/(m²·s)]
    std::vector<double> avg_temperature;        // Time-averaged temperature [K]

    // Cached base fluxes (computed once when ice is present everywhere)
    std::vector<double> base_sublimation_flux;  // Unscaled sublimation [kg/(m²·s)]
    std::vector<double> base_sputtering_flux;   // Unscaled sputtering [kg/(m²·s)]
    std::vector<double> base_deposition_flux;   // Base deposition [kg/(m²·s)]

    // Cache for ice availability factors at last computation
    std::vector<double> cached_ice_availability;

    // Peak values (for diagnostics)
    std::vector<double> max_temperature;  // [K]
    std::vector<double> min_temperature;  // [K]

    // Cache status
    bool cache_valid = false;
    int cache_epoch = -1;

    // Total emission at cache time (for scaling deposition)
    double total_base_emission = 0.0;  // [kg/s] (summed over all faces)

    explicit DiurnalAverages(size_t num_faces) {
        avg_sublimation_flux.resize(num_faces, 0.0);
        avg_deposition_flux.resize(num_faces, 0.0);
        avg_sputtering_flux.resize(num_faces, 0.0);
        avg_net_flux.resize(num_faces, 0.0);
        avg_escape_flux.resize(num_faces, 0.0);
        avg_temperature.resize(num_faces, 0.0);

        base_sublimation_flux.resize(num_faces, 0.0);
        base_sputtering_flux.resize(num_faces, 0.0);
        base_deposition_flux.resize(num_faces, 0.0);

        cached_ice_availability.resize(num_faces, 1.0);

        max_temperature.resize(num_faces, 0.0);
        min_temperature.resize(num_faces, 1000.0);
    }

    void reset() {
        std::fill(avg_sublimation_flux.begin(), avg_sublimation_flux.end(), 0.0);
        std::fill(avg_deposition_flux.begin(), avg_deposition_flux.end(), 0.0);
        std::fill(avg_sputtering_flux.begin(), avg_sputtering_flux.end(), 0.0);
        std::fill(avg_net_flux.begin(), avg_net_flux.end(), 0.0);
        std::fill(avg_escape_flux.begin(), avg_escape_flux.end(), 0.0);
        std::fill(avg_temperature.begin(), avg_temperature.end(), 0.0);
        std::fill(max_temperature.begin(), max_temperature.end(), 0.0);
        std::fill(min_temperature.begin(), min_temperature.end(), 1000.0);
        // Note: base_* and cached_* are NOT reset - they persist across epochs
    }

    // Finalize averages after accumulation
    void finalize(int num_timesteps) {
        double inv_n = 1.0 / static_cast<double>(num_timesteps);
        for (size_t i = 0; i < avg_sublimation_flux.size(); ++i) {
            avg_sublimation_flux[i] *= inv_n;
            avg_deposition_flux[i] *= inv_n;
            avg_sputtering_flux[i] *= inv_n;
            avg_net_flux[i] *= inv_n;
            avg_escape_flux[i] *= inv_n;
            avg_temperature[i] *= inv_n;
        }
    }

    // Store current averages as base (cache) values
    // ice_availability: 1.0 where ice is present, 0.0 where depleted
    void cacheBaseFluxes(const std::vector<double>& ice_availability) {
        total_base_emission = 0.0;
        for (size_t i = 0; i < avg_sublimation_flux.size(); ++i) {
            // Store current fluxes as base values
            // If ice is not available, the flux was already 0, so base is still valid
            double avail = std::max(ice_availability[i], 1e-10);

            base_sublimation_flux[i] = avg_sublimation_flux[i] / avail;
            base_sputtering_flux[i] = avg_sputtering_flux[i] / avail;
            base_deposition_flux[i] = avg_deposition_flux[i];

            // Sum total base emission
            total_base_emission += base_sublimation_flux[i] + base_sputtering_flux[i];

            cached_ice_availability[i] = ice_availability[i];
        }
        cache_valid = true;
    }

    // Apply ice availability factors to cached base fluxes (fast operation)
    void applyScaledFluxes(const std::vector<double>& ice_availability) {
        // First, compute total current emission to get global deposition scaling
        double total_current_emission = 0.0;
        for (size_t i = 0; i < avg_sublimation_flux.size(); ++i) {
            total_current_emission += base_sublimation_flux[i] * ice_availability[i]
                                    + base_sputtering_flux[i] * ice_availability[i];
        }

        // Deposition scales with total emission
        double deposition_scale = (total_base_emission > 1e-30)
                                ? total_current_emission / total_base_emission
                                : 1.0;

        #pragma omp parallel for
        for (size_t i = 0; i < avg_sublimation_flux.size(); ++i) {
            // Scale emission by ice availability (0 if no ice, 1 if ice present)
            avg_sublimation_flux[i] = base_sublimation_flux[i] * ice_availability[i];
            avg_sputtering_flux[i] = base_sputtering_flux[i] * ice_availability[i];

            // Scale deposition by global emission change
            avg_deposition_flux[i] = base_deposition_flux[i] * deposition_scale;

            // Update net flux
            avg_net_flux[i] = avg_deposition_flux[i] - avg_sublimation_flux[i];
        }
    }

    // Check if cache refresh is needed based on ice availability changes
    bool needsRefresh(const std::vector<double>& ice_availability, double threshold) const {
        if (!cache_valid) return true;

        for (size_t i = 0; i < ice_availability.size(); ++i) {
            double delta = std::abs(ice_availability[i] - cached_ice_availability[i]);
            if (delta > threshold) {
                return true;
            }
        }
        return false;
    }
};

/**
 * Statistics for long-term evolution tracking
 *
 * Units documented for each field.
 */
struct EvolutionStatistics {
    int current_epoch = 0;
    double elapsed_time_years = 0.0;  // [years]

    // Global mass balance
    double total_sublimated_mass = 0.0;    // [kg]
    double total_deposited_mass = 0.0;     // [kg]
    double total_sputtered_mass = 0.0;     // [kg]
    double total_escaped_mass = 0.0;       // [kg]

    // Non-ice concentration statistics
    double max_nonice_concentration = 0.0;   // [dimensionless, 0-1]
    double mean_nonice_concentration = 0.0;  // [dimensionless, 0-1]
    size_t faces_with_ice = 0;               // count
    size_t faces_depleted = 0;               // count (concentration = 1)

    // Convergence tracking
    double max_concentration_change = 0.0;  // [dimensionless] max change this epoch
    bool converged = false;

    void print() const {
        std::cout << "\n--- Evolution Statistics (Epoch " << current_epoch << ") ---" << std::endl;
        std::cout << "  Elapsed time: " << std::scientific << std::setprecision(3)
                  << elapsed_time_years << " years" << std::endl;
        std::cout << "  Non-ice concentration:" << std::endl;
        std::cout << "    Max: " << std::fixed << std::setprecision(4)
                  << (max_nonice_concentration * 100) << "%" << std::endl;
        std::cout << "    Mean: " << (mean_nonice_concentration * 100) << "%" << std::endl;
        std::cout << "  Faces with ice: " << faces_with_ice << std::endl;
        std::cout << "  Faces depleted (no ice): " << faces_depleted << std::endl;
        std::cout << "  Max concentration change: " << std::scientific
                  << max_concentration_change << " /epoch" << std::endl;
        std::cout << "  Converged: " << (converged ? "yes" : "no") << std::endl;
        std::cout << std::fixed;
    }
};

/**
 * Multi-timescale evolution controller
 *
 * Manages the operator splitting between fast diurnal dynamics
 * and slow non-ice concentration evolution.
 */
class MultiTimescaleEvolution {
private:
    MultiTimescaleConfig config_;
    size_t num_faces_;

    // Diurnal averages (computed once per epoch refresh)
    std::unique_ptr<DiurnalAverages> diurnal_avg_;

    // Evolution statistics
    EvolutionStatistics stats_;

    // Concentration change rates for each face [1/s]
    std::vector<double> concentration_change_rate_;

public:
    /**
     * Constructor
     * @param num_faces Number of mesh faces
     * @param config Multi-timescale configuration
     */
    explicit MultiTimescaleEvolution(size_t num_faces,
                                      const MultiTimescaleConfig& config = MultiTimescaleConfig())
        : config_(config), num_faces_(num_faces) {
        diurnal_avg_ = std::make_unique<DiurnalAverages>(num_faces);
        concentration_change_rate_.resize(num_faces, 0.0);
    }

    /**
     * Accumulate flux data from a single diurnal timestep
     * Call this during the diurnal cycle simulation
     *
     * @param emission_flux Current emission flux [kg/(m²·s)]
     * @param deposition_flux Current deposition flux [kg/(m²·s)]
     * @param sputtering_flux Current sputtering flux [kg/(m²·s)] (can be empty)
     * @param escape_flux Current escape flux [kg/(m²·s)]
     * @param temperatures Current surface temperatures [K]
     */
    void accumulateDiurnalStep(const std::vector<double>& emission_flux,
                                const std::vector<double>& deposition_flux,
                                const std::vector<double>& sputtering_flux,
                                const std::vector<double>& escape_flux,
                                const std::vector<double>& temperatures) {
        #pragma omp parallel for
        for (size_t i = 0; i < num_faces_; ++i) {
            diurnal_avg_->avg_sublimation_flux[i] += emission_flux[i];
            diurnal_avg_->avg_deposition_flux[i] += deposition_flux[i];

            if (!sputtering_flux.empty()) {
                diurnal_avg_->avg_sputtering_flux[i] += sputtering_flux[i];
            }

            diurnal_avg_->avg_escape_flux[i] += escape_flux[i];
            diurnal_avg_->avg_net_flux[i] += (deposition_flux[i] - emission_flux[i]);
            diurnal_avg_->avg_temperature[i] += temperatures[i];

            diurnal_avg_->max_temperature[i] = std::max(diurnal_avg_->max_temperature[i], temperatures[i]);
            diurnal_avg_->min_temperature[i] = std::min(diurnal_avg_->min_temperature[i], temperatures[i]);
        }
    }

    /**
     * Finalize diurnal averages after one complete diurnal cycle
     * @param num_timesteps Number of timesteps in the diurnal cycle
     */
    void finalizeDiurnalAverages(int num_timesteps) {
        diurnal_avg_->finalize(num_timesteps);
    }

    /**
     * Reset diurnal accumulator for next epoch
     */
    void resetDiurnalAccumulator() {
        diurnal_avg_->reset();
    }

    /**
     * Check if flux cache is valid
     */
    bool isCacheValid() const {
        return diurnal_avg_->cache_valid;
    }

    /**
     * Cache current base fluxes for future scaling
     * Call this after first diurnal computation
     */
    void cacheBaseFluxes(const IceLayerModel& ice_model,
                         const IceLayerState& ice_state,
                         int epoch) {
        // Compute ice availability factors (1 if ice present, 0 if depleted)
        std::vector<double> ice_availability(num_faces_);
        for (size_t i = 0; i < num_faces_; ++i) {
            ice_availability[i] = ice_model.getIceAvailabilityFactor(ice_state, i);
        }

        diurnal_avg_->cacheBaseFluxes(ice_availability);
        diurnal_avg_->cache_epoch = epoch;
    }

    /**
     * Apply scaled fluxes using cached base values
     * Much faster than recomputing diurnal cycle
     */
    void applyScaledFluxes(const IceLayerModel& ice_model,
                           const IceLayerState& ice_state) {
        // Compute current ice availability factors
        std::vector<double> ice_availability(num_faces_);
        for (size_t i = 0; i < num_faces_; ++i) {
            ice_availability[i] = ice_model.getIceAvailabilityFactor(ice_state, i);
        }

        diurnal_avg_->applyScaledFluxes(ice_availability);
    }

    /**
     * Check if cache refresh is needed
     */
    bool needsCacheRefresh(const IceLayerModel& ice_model,
                           const IceLayerState& ice_state,
                           int current_epoch) const {
        if (!diurnal_avg_->cache_valid) return true;

        // Compute current ice availability
        std::vector<double> ice_availability(num_faces_);
        for (size_t i = 0; i < num_faces_; ++i) {
            ice_availability[i] = ice_model.getIceAvailabilityFactor(ice_state, i);
        }

        // Check for significant changes in ice availability
        return diurnal_avg_->needsRefresh(ice_availability, 0.1);  // 10% threshold
    }

    /**
     * Evolve non-ice concentration for one epoch using time-averaged fluxes
     *
     * This is the core of the operator splitting: use averaged fluxes
     * to advance the non-ice concentration over many diurnal cycles.
     *
     * @param ice_model Ice layer model
     * @param ice_state Ice layer state (modified in place)
     * @param face_areas Face areas [m²]
     * @param epoch_duration Duration of one epoch [s]
     */
    void evolveEpoch(const IceLayerModel& ice_model,
                     IceLayerState& ice_state,
                     const std::vector<double>& face_areas,
                     double epoch_duration) {
        double max_ice_mass = ice_model.getMaxIceMassPerArea();
        double max_change = 0.0;

        #pragma omp parallel for reduction(max:max_change)
        for (size_t i = 0; i < num_faces_; ++i) {
            double conc_before = ice_state.nonice_concentration[i];

            // Get averaged fluxes for this face
            double sublimation = diurnal_avg_->avg_sublimation_flux[i];
            double sputtering = diurnal_avg_->avg_sputtering_flux[i];
            double deposition = diurnal_avg_->avg_deposition_flux[i];

            // Net ice loss rate [kg/(m²·s)]
            // Positive = ice loss, Negative = ice gain
            double net_ice_loss_rate = sublimation + sputtering - deposition;

            // Convert to concentration change rate [1/s]
            // concentration increases when ice is lost
            double conc_rate = net_ice_loss_rate / max_ice_mass;
            concentration_change_rate_[i] = conc_rate;

            // Update concentration
            double conc_change = conc_rate * epoch_duration;
            ice_state.nonice_concentration[i] += conc_change;

            // Clamp to valid range [0, 1]
            if (ice_state.nonice_concentration[i] < 0) {
                ice_state.nonice_concentration[i] = 0.0;
            }
            if (ice_state.nonice_concentration[i] > layer_constants::MAX_NONICE_CONCENTRATION) {
                ice_state.nonice_concentration[i] = layer_constants::MAX_NONICE_CONCENTRATION;
            }

            // Update ice mass from concentration
            ice_model.updateIceMass(ice_state, i);

            // Track cumulative fluxes
            ice_state.cumulative_sublimated[i] += sublimation * epoch_duration;
            ice_state.cumulative_sputtered[i] += sputtering * epoch_duration;
            ice_state.cumulative_deposited[i] += deposition * epoch_duration;

            // Track maximum change
            double change = std::abs(ice_state.nonice_concentration[i] - conc_before);
            max_change = std::max(max_change, change);
        }

        // Update statistics
        stats_.max_concentration_change = max_change;
        stats_.converged = (max_change < config_.convergence_threshold);

        // Update global mass balance
        double epoch_sublimated = 0.0, epoch_deposited = 0.0, epoch_sputtered = 0.0;
        for (size_t i = 0; i < num_faces_; ++i) {
            epoch_sublimated += diurnal_avg_->avg_sublimation_flux[i] * face_areas[i];
            epoch_deposited += diurnal_avg_->avg_deposition_flux[i] * face_areas[i];
            epoch_sputtered += diurnal_avg_->avg_sputtering_flux[i] * face_areas[i];
        }
        stats_.total_sublimated_mass += epoch_sublimated * epoch_duration;
        stats_.total_deposited_mass += epoch_deposited * epoch_duration;
        stats_.total_sputtered_mass += epoch_sputtered * epoch_duration;
    }

    /**
     * Update statistics after epoch evolution
     */
    void updateStatistics(const IceLayerModel& ice_model,
                          const IceLayerState& ice_state,
                          double epoch_duration) {
        stats_.current_epoch++;
        stats_.elapsed_time_years += epoch_duration / timescale_constants::SECONDS_PER_YEAR;

        double max_conc = 0.0, total_conc = 0.0;
        size_t with_ice = 0, depleted = 0;

        for (size_t i = 0; i < num_faces_; ++i) {
            double conc = ice_state.nonice_concentration[i];
            max_conc = std::max(max_conc, conc);
            total_conc += conc;

            if (ice_model.hasIce(ice_state, i)) {
                with_ice++;
            } else {
                depleted++;
            }
        }

        stats_.max_nonice_concentration = max_conc;
        stats_.mean_nonice_concentration = total_conc / num_faces_;
        stats_.faces_with_ice = with_ice;
        stats_.faces_depleted = depleted;
    }

    /**
     * Get current statistics
     */
    const EvolutionStatistics& getStatistics() const { return stats_; }

    /**
     * Get diurnal averages (for export/analysis)
     */
    const DiurnalAverages& getDiurnalAverages() const { return *diurnal_avg_; }

    /**
     * Get configuration
     */
    const MultiTimescaleConfig& getConfig() const { return config_; }
    void setConfig(const MultiTimescaleConfig& config) { config_ = config; }

    /**
     * Get concentration change rate for a face [1/s]
     */
    double getConcentrationChangeRate(size_t face_idx) const {
        return (face_idx < concentration_change_rate_.size()) ? concentration_change_rate_[face_idx] : 0.0;
    }

    /**
     * Print progress with visual progress bar
     */
    void printProgress() const {
        if (!config_.show_progress) return;

        int epoch = stats_.current_epoch;
        int total = config_.num_epochs;
        double pct = 100.0 * epoch / total;
        double years = stats_.elapsed_time_years;

        // Build progress bar (30 chars wide)
        const int bar_width = 30;
        int filled = static_cast<int>(bar_width * epoch / total);
        std::string bar(filled, '=');
        if (filled < bar_width) bar += '>';
        bar.resize(bar_width, ' ');

        // Format years nicely
        std::string years_str;
        if (years < 1e-3) {
            years_str = std::to_string(static_cast<int>(years * 365.25 * 24)) + " hr";
        } else if (years < 1.0) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(2) << (years * 365.25) << " days";
            years_str = oss.str();
        } else if (years < 1000.0) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(1) << years << " yr";
            years_str = oss.str();
        } else if (years < 1e6) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(2) << (years / 1000.0) << " kyr";
            years_str = oss.str();
        } else {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(2) << (years / 1e6) << " Myr";
            years_str = oss.str();
        }

        // Format concentration
        std::string conc_str;
        double conc_pct = stats_.max_nonice_concentration * 100.0;
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << conc_pct << "%";
        conc_str = oss.str();

        std::cerr << "\r[" << bar << "] "
                  << std::setw(5) << std::fixed << std::setprecision(1) << pct << "% | "
                  << "Epoch " << std::setw(5) << epoch << "/" << total << " | "
                  << std::setw(10) << years_str << " | "
                  << "nonice: " << std::setw(8) << conc_str
                  << "   " << std::flush;
    }

    /**
     * Reset evolution state for new simulation
     */
    void reset() {
        diurnal_avg_->reset();
        std::fill(concentration_change_rate_.begin(), concentration_change_rate_.end(), 0.0);
        stats_ = EvolutionStatistics();
    }
};

} // namespace exosphere
} // namespace ptm

#endif // MULTITIMESCALE_EVOLUTION_H
