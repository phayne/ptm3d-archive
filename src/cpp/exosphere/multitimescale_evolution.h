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
 * Key physics:
 * - Ice availability is continuous: sublimation scales with ice fraction (1 - nonice_concentration)
 * - Self-limiting depletion: as ice is consumed, sublimation rate decreases proportionally
 * - Analytical exponential evolution: d(nonice)/dt = A*(1-nonice) - B is solved exactly,
 *   enabling accurate super-stepping over many epochs without numerical error
 * - Each face reaches equilibrium where reduced sublimation balances incoming deposition
 *
 * Key approximations:
 * - Transport matrix is fixed (computed once at start)
 * - Deposition is treated as constant within super-steps (globally scaled by total emission)
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
    constexpr double CONVERGENCE_THRESHOLD = 1e-6;  // dimensionless

    // Flux imbalance threshold for mass balance convergence
    constexpr double FLUX_CONVERGENCE_THRESHOLD = 0.01;  // 1% imbalance
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

    // Flux imbalance threshold for mass balance convergence
    // Converges when |sublimation + sputtering - deposition| / (sublimation + sputtering) < threshold
    double flux_convergence_threshold = timescale_constants::FLUX_CONVERGENCE_THRESHOLD;

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
        std::cout << "  Flux convergence threshold: " << (flux_convergence_threshold * 100.0) << "%" << std::endl;
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

    // Time-averaged crystallization rate <k_cryst(T(t))> [s^-1].
    // Distinct from k_cryst(<T>) because k_cryst is exponential in T; using
    // <k_cryst(T)> avoids a Jensen's-inequality bias of many orders of
    // magnitude near the crystallization knife-edge (~125 K). Populated only
    // when the crystallization model is active; left at zero otherwise.
    std::vector<double> avg_crystallization_rate;

    // Cached base fluxes (theoretical full-rate, normalized by ice fraction at cache time)
    // evolveEpoch() uses these directly with analytical exponential evolution
    std::vector<double> base_sublimation_flux;  // Full-rate sublimation [kg/(m²·s)]
    std::vector<double> base_sputtering_flux;   // Full-rate sputtering [kg/(m²·s)]
    std::vector<double> base_deposition_flux;   // Base deposition [kg/(m²·s)]

    // Cache for ice availability (ice fraction) at last full diurnal computation
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
        avg_crystallization_rate.resize(num_faces, 0.0);

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
        std::fill(avg_crystallization_rate.begin(), avg_crystallization_rate.end(), 0.0);
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
            avg_crystallization_rate[i] *= inv_n;
        }
    }

    // Store current averages as base (cache) values
    // ice_availability: continuous ice fraction [0, 1] per face
    // face_areas: per-face areas [m²] for proper area-weighted global emission
    void cacheBaseFluxes(const std::vector<double>& ice_availability,
                         const std::vector<double>& face_areas) {
        total_base_emission = 0.0;
        for (size_t i = 0; i < avg_sublimation_flux.size(); ++i) {
            // Store current fluxes as base values, normalized by ice fraction.
            // Floor at 1e-3 to avoid extreme amplification when ice fraction
            // is very small (base flux would be at most 1000× raw flux).
            double avail = std::max(ice_availability[i], 1e-3);

            base_sublimation_flux[i] = avg_sublimation_flux[i] / avail;
            base_sputtering_flux[i] = avg_sputtering_flux[i] / avail;
            base_deposition_flux[i] = avg_deposition_flux[i];

            // Area-weighted total base emission [kg/s]
            total_base_emission += (base_sublimation_flux[i] + base_sputtering_flux[i]) * face_areas[i];

            cached_ice_availability[i] = ice_availability[i];
        }
        cache_valid = true;
    }

    // Apply ice fraction scaling to cached base fluxes (fast operation).
    // Sets avg_* fields for diagnostics/export. evolveEpoch() uses base_* directly
    // with analytical exponential evolution for the actual concentration update.
    void applyScaledFluxes(const std::vector<double>& ice_availability,
                           const std::vector<double>& face_areas) {
        size_t n = avg_sublimation_flux.size();

        // Pass 1: Compute scaled emission per face and global total (fused with reduction)
        double total_current_emission = 0.0;

        #pragma omp parallel for reduction(+:total_current_emission)
        for (size_t i = 0; i < n; ++i) {
            double sublim_scaled = base_sublimation_flux[i] * ice_availability[i];
            double sput_scaled = base_sputtering_flux[i] * ice_availability[i];
            avg_sublimation_flux[i] = sublim_scaled;
            avg_sputtering_flux[i] = sput_scaled;
            total_current_emission += (sublim_scaled + sput_scaled) * face_areas[i];
        }

        // Deposition scales with total emission (both now area-weighted [kg/s])
        double deposition_scale = (total_base_emission > 1e-30)
                                ? total_current_emission / total_base_emission
                                : 1.0;

        // Pass 2: Apply deposition scaling and compute net flux
        #pragma omp parallel for
        for (size_t i = 0; i < n; ++i) {
            avg_deposition_flux[i] = base_deposition_flux[i] * deposition_scale;
            avg_net_flux[i] = avg_deposition_flux[i] - avg_sublimation_flux[i] - avg_sputtering_flux[i];
        }
    }

    // Check if cache refresh is needed based on ice fraction changes.
    // Uses emission-weighted mean change: a face that contributes 0.1% of global
    // emission can deplete without triggering a global recomputation.
    // With continuous ice fractions, changes are gradual, so refresh triggers
    // less frequently than with the old binary model.
    bool needsRefresh(const std::vector<double>& ice_availability,
                      const std::vector<double>& face_areas,
                      double threshold) const {
        if (!cache_valid) return true;

        double weighted_change = 0.0;
        double total_weight = 0.0;
        int faces_changed = 0;

        for (size_t i = 0; i < ice_availability.size(); ++i) {
            double delta = std::abs(ice_availability[i] - cached_ice_availability[i]);
            // Weight by base emission contribution (area × flux)
            double weight = (base_sublimation_flux[i] + base_sputtering_flux[i]) * face_areas[i];
            weighted_change += delta * weight;
            total_weight += weight;
            if (delta > 0.01) faces_changed++;
        }

        double mean_change = (total_weight > 1e-30) ? weighted_change / total_weight : 0.0;

        // Refresh if emission-weighted mean change exceeds threshold
        // OR if >5% of faces changed significantly (structural change)
        return (mean_change > threshold) ||
               (faces_changed > static_cast<int>(0.05 * ice_availability.size()));
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

    // Mass balance rates (per epoch, for convergence diagnostics)
    double sublimation_rate = 0.0;          // [kg/s] global sublimation rate this epoch
    double deposition_rate = 0.0;           // [kg/s] global deposition rate this epoch
    double sputtering_rate = 0.0;           // [kg/s] global sputtering rate this epoch
    double net_mass_loss_rate = 0.0;        // [kg/s] net loss (sublim + sput - depos)
    double flux_imbalance_fraction = 0.0;   // |net| / (sublim + sput), dimensionless

    // Albedo coupling statistics
    double max_albedo_change = 0.0;       // [dimensionless] max albedo change this epoch
    double mean_albedo = 0.0;             // [dimensionless] mean albedo across all faces
    int albedo_temp_refreshes = 0;        // count of temperature refreshes triggered by albedo

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
        if (sublimation_rate > 0.0 || deposition_rate > 0.0) {
            std::cout << "  Mass balance:" << std::endl;
            std::cout << "    Sublimation: " << std::scientific << std::setprecision(3)
                      << sublimation_rate << " kg/s" << std::endl;
            std::cout << "    Deposition:  " << deposition_rate << " kg/s" << std::endl;
            if (sputtering_rate > 0.0) {
                std::cout << "    Sputtering:  " << sputtering_rate << " kg/s" << std::endl;
            }
            std::cout << "    Net loss:    " << net_mass_loss_rate << " kg/s" << std::endl;
            std::cout << "    Flux imbalance: " << std::fixed << std::setprecision(2)
                      << (flux_imbalance_fraction * 100.0) << "%" << std::endl;
            std::cout << "  Cumulative mass balance:" << std::endl;
            std::cout << "    Total sublimated: " << std::scientific << std::setprecision(3)
                      << total_sublimated_mass << " kg" << std::endl;
            std::cout << "    Total deposited:  " << total_deposited_mass << " kg" << std::endl;
            if (total_sputtered_mass > 0.0) {
                std::cout << "    Total sputtered:  " << total_sputtered_mass << " kg" << std::endl;
            }
            std::cout << "    Total escaped:    " << total_escaped_mass << " kg" << std::endl;
        }
        if (mean_albedo > 0.0) {
            std::cout << "  Albedo coupling:" << std::endl;
            std::cout << "    Mean albedo: " << std::fixed << std::setprecision(4)
                      << mean_albedo << std::endl;
            std::cout << "    Max albedo change: " << std::scientific
                      << max_albedo_change << " /epoch" << std::endl;
            std::cout << "    Temperature refreshes: " << albedo_temp_refreshes << std::endl;
        }
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

    // Reusable buffer for ice availability queries (avoids per-epoch allocation)
    mutable std::vector<double> ice_availability_buf_;

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
        ice_availability_buf_.resize(num_faces, 0.0);
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
            double sput = (!sputtering_flux.empty()) ? sputtering_flux[i] : 0.0;
            diurnal_avg_->avg_net_flux[i] += (deposition_flux[i] - emission_flux[i] - sput);
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
     * Accumulate per-face crystallization rate at one diurnal timestep.
     *
     * Must be called once per diurnal timestep (and once per stride if
     * subsampling is in use), exactly in lock-step with accumulateDiurnalStep,
     * so that the averaging denominator matches. After the diurnal cycle
     * completes, finalizeDiurnalAverages divides by the timestep count to
     * yield <k_cryst(T(t))> per face.
     *
     * @param k_cryst_per_face Pre-computed k_cryst(T(t), porosity) [s^-1]
     */
    void accumulateCrystallizationRate(const std::vector<double>& k_cryst_per_face) {
        #pragma omp parallel for
        for (size_t i = 0; i < num_faces_; ++i) {
            diurnal_avg_->avg_crystallization_rate[i] += k_cryst_per_face[i];
        }
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
                         const std::vector<double>& face_areas,
                         int epoch) {
        // Compute ice fraction per face (continuous, 0 = depleted, 1 = pure ice)
        for (size_t i = 0; i < num_faces_; ++i) {
            ice_availability_buf_[i] = ice_model.getIceAvailabilityFactor(ice_state, i);
        }

        diurnal_avg_->cacheBaseFluxes(ice_availability_buf_, face_areas);
        diurnal_avg_->cache_epoch = epoch;
    }

    /**
     * Apply scaled fluxes using cached base values
     * Much faster than recomputing diurnal cycle
     */
    void applyScaledFluxes(const IceLayerModel& ice_model,
                           const IceLayerState& ice_state,
                           const std::vector<double>& face_areas) {
        // Compute current ice availability factors
        for (size_t i = 0; i < num_faces_; ++i) {
            ice_availability_buf_[i] = ice_model.getIceAvailabilityFactor(ice_state, i);
        }

        diurnal_avg_->applyScaledFluxes(ice_availability_buf_, face_areas);
    }

    /**
     * Check if cache refresh is needed.
     *
     * The transport product depends on TEMPERATURE (kernel shape), not ice
     * availability. Ice availability changes are handled cheaply by
     * applyScaledFluxes(), which scales cached base fluxes. A full diurnal
     * + transport recomputation (~200s for 1.3M faces) is only needed when:
     *   (a) Cache is not yet valid (first epoch)
     *   (b) Temperatures changed (albedo coupling — handled via force_cache_refresh
     *       in the caller)
     *
     * Ice availability changes do NOT require transport recomputation.
     */
    bool needsCacheRefresh(const IceLayerModel& ice_model,
                           const IceLayerState& ice_state,
                           const std::vector<double>& face_areas,
                           int current_epoch) const {
        if (!diurnal_avg_->cache_valid) return true;

        // Transport product depends on temperature, not ice availability.
        // applyScaledFluxes() handles ice depletion correctly by scaling
        // emission to 0 and reducing deposition proportionally.
        // Temperature changes are signaled via force_cache_refresh from
        // the albedo coupling logic in the caller.
        return false;
    }

    /**
     * Compute the maximum number of epochs that can be safely super-stepped.
     *
     * Uses the emission-weighted criterion: faces are sorted by time-to-depletion,
     * and we accumulate the weighted impact of each depletion until the cumulative
     * emission-weighted change exceeds the cache refresh threshold (10%).
     *
     * This allows jumping past many low-impact depletion events (cold polar faces
     * with negligible emission) while stopping before a high-impact face depletes.
     *
     * Also limits step size to avoid overshooting concentration bounds.
     *
     * @param ice_state Current ice layer state
     * @param face_areas Per-face areas [m²]
     * @param epoch_duration Duration of one epoch [s]
     * @param max_epochs Maximum epochs to step (remaining in simulation)
     * @return Number of safe epochs [1, max_epochs]
     */
    int computeSafeEpochCount(const IceLayerState& ice_state,
                              const std::vector<double>& face_areas,
                              double epoch_duration,
                              int max_epochs) const {
        if (max_epochs <= 1 || epoch_duration <= 0.0) return 1;

        // Compute total base emission for weighting
        double total_emission_weight = 0.0;
        for (size_t i = 0; i < num_faces_; ++i) {
            total_emission_weight += (diurnal_avg_->base_sublimation_flux[i] +
                                      diurnal_avg_->base_sputtering_flux[i]) * face_areas[i];
        }

        if (total_emission_weight < 1e-30) {
            // No emission at all → safe to step to the end
            return max_epochs;
        }

        // Build list of (time_to_event, emission_weight) for faces with significant rate changes
        struct FaceEvent {
            double time;      // Time until ice fraction changes significantly
            double weight;    // Emission weight of this face
        };
        std::vector<FaceEvent> events;
        events.reserve(num_faces_);

        double max_time = max_epochs * epoch_duration;

        for (size_t i = 0; i < num_faces_; ++i) {
            double rate = concentration_change_rate_[i];
            double conc = ice_state.nonice_concentration[i];

            if (rate > 0.0) {
                // Face losing ice → will deplete
                double room = layer_constants::MAX_NONICE_CONCENTRATION - conc;
                if (room > 1e-10) {
                    double t = room / rate;
                    if (t < max_time) {
                        double w = (diurnal_avg_->base_sublimation_flux[i] +
                                    diurnal_avg_->base_sputtering_flux[i]) * face_areas[i];
                        events.push_back({t, w});
                    }
                }
            } else if (rate < 0.0) {
                // Face gaining ice → ice fraction increasing
                // Track when concentration drops enough to significantly change sublimation rate
                if (conc > 0.01) {
                    // Time for concentration to drop by 0.01 (1% ice fraction change)
                    double t = 0.01 / (-rate);
                    if (t > 0.0 && t < max_time) {
                        double w = (diurnal_avg_->base_deposition_flux[i]) * face_areas[i];
                        events.push_back({t, w});
                    }
                }
            }
        }

        if (events.empty()) {
            // No events within the time window → safe to step all the way
            return max_epochs;
        }

        // Sort events by time
        std::sort(events.begin(), events.end(),
                  [](const FaceEvent& a, const FaceEvent& b) { return a.time < b.time; });

        // Walk through events, accumulating weighted change until threshold
        double cumulative_weighted_change = 0.0;
        double threshold = 0.1;  // 10% emission-weighted change
        double safe_time = max_time;

        for (const auto& event : events) {
            cumulative_weighted_change += event.weight / total_emission_weight;

            if (cumulative_weighted_change > threshold) {
                // This event pushes us over the threshold
                safe_time = event.time;
                break;
            }

            // Also check: if >5% of faces changed, stop (structural change)
            // (Approximate: count events so far)
        }

        // Convert safe time to epoch count (conservative: floor)
        int safe_epochs = static_cast<int>(safe_time / epoch_duration);

        // Clamp to [1, max_epochs]
        safe_epochs = std::max(1, std::min(safe_epochs, max_epochs));

        return safe_epochs;
    }

    /**
     * Evolve non-ice concentration using analytical exponential solution
     *
     * Solves the ODE: d(nonice)/dt = A*(1-nonice) - B exactly, where
     * A = base_emission/max_ice_mass (emission rate constant) and
     * B = deposition/max_ice_mass (deposition rate). This provides
     * accurate evolution over arbitrarily long super-steps.
     *
     * Each face approaches equilibrium nonice_eq = 1 - B/A, where
     * sublimation (reduced by ice fraction) balances deposition.
     *
     * @param ice_model Ice layer model
     * @param ice_state Ice layer state (modified in place)
     * @param face_areas Face areas [m²]
     * @param epoch_duration Duration of one epoch [s] (may be multiple epochs for super-stepping)
     */
    void evolveEpoch(const IceLayerModel& ice_model,
                     IceLayerState& ice_state,
                     const std::vector<double>& face_areas,
                     double epoch_duration) {
        double max_ice_mass = ice_model.getMaxIceMassPerArea();
        double max_change = 0.0;
        double epoch_sublimated = 0.0, epoch_deposited = 0.0, epoch_sputtered = 0.0;

        #pragma omp parallel for reduction(max:max_change) \
            reduction(+:epoch_sublimated,epoch_deposited,epoch_sputtered)
        for (size_t i = 0; i < num_faces_; ++i) {
            double conc_before = ice_state.nonice_concentration[i];

            // Base (unscaled) emission rates and deposition
            double base_sublim = diurnal_avg_->base_sublimation_flux[i];
            double base_sput = diurnal_avg_->base_sputtering_flux[i];
            double deposition = diurnal_avg_->avg_deposition_flux[i];

            // Analytical exponential evolution:
            // The ODE is: d(nonice)/dt = [base_emission * (1-nonice) - deposition] / max_ice_mass
            // where sublimation/sputtering scale with ice fraction (1-nonice).
            // Let A = (base_sublim + base_sput) / max_ice_mass  [emission rate constant]
            //     B = deposition / max_ice_mass                  [deposition rate]
            // ODE: d(nonice)/dt = A*(1-nonice) - B = A - B - A*nonice
            // Solution: nonice(t) = nonice_eq - (nonice_eq - nonice_0) * exp(-A*t)
            //   where nonice_eq = 1 - B/A  (equilibrium concentration)
            double A = (base_sublim + base_sput) / max_ice_mass;
            double B = deposition / max_ice_mass;

            double nonice_new;
            if (A > 1e-30) {
                double nonice_eq = 1.0 - B / A;
                double decay = std::exp(-A * epoch_duration);
                nonice_new = nonice_eq - (nonice_eq - conc_before) * decay;
            } else {
                // No emission: pure deposition (linear)
                nonice_new = conc_before - B * epoch_duration;
            }

            // Effective rates for mass balance tracking (time-averaged over epoch)
            double ice_frac_before = 1.0 - conc_before;
            double ice_frac_after = 1.0 - std::clamp(nonice_new, 0.0, 1.0);
            double avg_ice_frac = 0.5 * (ice_frac_before + ice_frac_after);
            double eff_sublimation = base_sublim * avg_ice_frac;
            double eff_sputtering = base_sput * avg_ice_frac;

            // Accumulate area-weighted mass balance
            epoch_sublimated += eff_sublimation * face_areas[i];
            epoch_deposited += deposition * face_areas[i];
            epoch_sputtered += eff_sputtering * face_areas[i];

            // Store concentration change rate (instantaneous, for super-step estimation)
            double net_rate = A * (1.0 - conc_before) - B;
            concentration_change_rate_[i] = net_rate;

            // Clamp to valid range [0, 1]
            ice_state.nonice_concentration[i] = std::clamp(nonice_new, 0.0,
                layer_constants::MAX_NONICE_CONCENTRATION);

            // Update ice mass from concentration
            ice_model.updateIceMass(ice_state, i);

            // Track cumulative fluxes
            ice_state.cumulative_sublimated[i] += eff_sublimation * epoch_duration;
            ice_state.cumulative_sputtered[i] += eff_sputtering * epoch_duration;
            ice_state.cumulative_deposited[i] += deposition * epoch_duration;

            // Track maximum change
            double change = std::abs(ice_state.nonice_concentration[i] - conc_before);
            max_change = std::max(max_change, change);
        }

        // Update statistics
        stats_.max_concentration_change = max_change;
        stats_.total_sublimated_mass += epoch_sublimated * epoch_duration;
        stats_.total_deposited_mass += epoch_deposited * epoch_duration;
        stats_.total_sputtered_mass += epoch_sputtered * epoch_duration;

        // Net mass loss = escape to space (sublimation + sputtering that isn't redeposited)
        double epoch_net_loss = epoch_sublimated + epoch_sputtered - epoch_deposited;
        stats_.total_escaped_mass += std::max(0.0, epoch_net_loss) * epoch_duration;

        // Mass balance rates (for convergence diagnostics)
        stats_.sublimation_rate = epoch_sublimated;
        stats_.deposition_rate = epoch_deposited;
        stats_.sputtering_rate = epoch_sputtered;
        stats_.net_mass_loss_rate = epoch_net_loss;

        double total_emission = epoch_sublimated + epoch_sputtered;
        stats_.flux_imbalance_fraction = (total_emission > 1e-30)
            ? std::abs(total_emission - epoch_deposited) / total_emission
            : 0.0;

        // Multi-criteria convergence:
        // (a) Concentration change below threshold, OR
        // (b) Flux imbalance small AND concentration change rate is low
        stats_.converged = (max_change < config_.convergence_threshold)
            || (stats_.flux_imbalance_fraction < config_.flux_convergence_threshold
                && max_change < 1e-4);
    }

    /**
     * Update statistics after epoch evolution
     * @param epoch_count Number of epochs advanced (for super-stepping, default 1)
     */
    void updateStatistics(const IceLayerModel& ice_model,
                          const IceLayerState& ice_state,
                          double epoch_duration,
                          int epoch_count = 1) {
        stats_.current_epoch += epoch_count;
        stats_.elapsed_time_years += (epoch_duration * epoch_count) / timescale_constants::SECONDS_PER_YEAR;

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
    DiurnalAverages& getMutableDiurnalAverages() { return *diurnal_avg_; }

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
        {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(2) << conc_pct << "%";
            conc_str = oss.str();
        }

        // Format flux imbalance
        std::string flux_str;
        {
            double flux_pct = stats_.flux_imbalance_fraction * 100.0;
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(2) << flux_pct << "%";
            flux_str = oss.str();
        }

        std::cerr << "\r[" << bar << "] "
                  << std::setw(5) << std::fixed << std::setprecision(1) << pct << "% | "
                  << "Epoch " << std::setw(5) << epoch << "/" << total << " | "
                  << std::setw(10) << years_str << " | "
                  << "nonice: " << std::setw(8) << conc_str << " | "
                  << "flux: " << std::setw(7) << flux_str
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
