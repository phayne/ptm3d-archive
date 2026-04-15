/*
 * Composition-to-Albedo Mapping for Exosphere-Thermal Coupling
 *
 * Maps non-ice concentration (from IceLayerModel) to surface Bond albedo
 * using a linear mixing model between pure ice and non-ice endmembers.
 *
 * Physical basis:
 * - Pure water ice on Europa has high albedo (~0.5-0.9 depending on grain size)
 * - Non-ice lag deposits (silicates, salts, organics) have low albedo (~0.05-0.15)
 * - As ice sublimes, non-ice material concentrates at the surface
 * - Lower albedo -> higher absorbed solar flux -> higher temperature -> more sublimation
 * - This positive feedback drives thermal segregation of volatiles
 *
 * The linear mixing model is:
 *   A(c) = A_ice * (1 - c) + A_nonice * c
 * where c = nonice_concentration [0, 1]
 *
 * Endmember values are configurable at runtime via --albedo-ice and --albedo-nonice.
 */

#ifndef COMPOSITION_ALBEDO_H
#define COMPOSITION_ALBEDO_H

#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>

namespace ptm {
namespace exosphere {

/**
 * Configuration for composition-albedo mapping
 */
struct CompositionAlbedoConfig {
    // Pure water ice Bond albedo (endmember at nonice_concentration = 0)
    double albedo_ice = 0.65;

    // Non-ice lag deposit Bond albedo (endmember at nonice_concentration = 1)
    double albedo_nonice = 0.10;

    // Minimum albedo change to trigger temperature update
    double albedo_update_threshold = 0.01;

    CompositionAlbedoConfig() = default;

    CompositionAlbedoConfig(double a_ice, double a_nonice, double threshold = 0.01)
        : albedo_ice(a_ice), albedo_nonice(a_nonice), albedo_update_threshold(threshold) {}

    void print() const {
        std::cout << "\n=== Composition-Albedo Coupling Configuration ===" << std::endl;
        std::cout << "  Ice albedo (endmember): " << albedo_ice << std::endl;
        std::cout << "  Non-ice albedo (endmember): " << albedo_nonice << std::endl;
        std::cout << "  Update threshold: " << albedo_update_threshold << std::endl;
    }
};

/**
 * Maps non-ice concentration to surface albedo via linear mixing
 */
class CompositionAlbedoMapper {
private:
    CompositionAlbedoConfig config_;
    std::vector<double> nonice_endmember_map_;  // Optional per-face non-ice albedo

public:
    CompositionAlbedoMapper() = default;

    explicit CompositionAlbedoMapper(const CompositionAlbedoConfig& config)
        : config_(config) {}

    /**
     * Set per-face non-ice albedo endmembers from a spatial albedo map.
     * When set, A(c, face) = A_ice*(1-c) + A_map[face]*c instead of
     * using the uniform albedo_nonice scalar.
     */
    void setNoniceEndmemberMap(const std::vector<double>& map) {
        nonice_endmember_map_ = map;
    }

    /** Check if per-face non-ice endmember map is loaded */
    bool hasNoniceMap() const { return !nonice_endmember_map_.empty(); }

    /**
     * Compute albedo for a single face from its non-ice concentration
     * @param nonice_concentration Non-ice volume fraction [0, 1]
     * @param face_idx Face index (used for per-face endmember map)
     * @return Bond albedo
     */
    double computeAlbedo(double nonice_concentration, size_t face_idx = 0) const {
        double c = std::clamp(nonice_concentration, 0.0, 1.0);
        double a_nonice = (!nonice_endmember_map_.empty() && face_idx < nonice_endmember_map_.size())
            ? nonice_endmember_map_[face_idx]
            : config_.albedo_nonice;
        return config_.albedo_ice * (1.0 - c) + a_nonice * c;
    }

    /**
     * Compute albedos for all faces
     * @param nonice_concentration Per-face non-ice concentration [0, 1]
     * @param albedos_out Output per-face albedos (resized if needed)
     */
    void computeAlbedos(const std::vector<double>& nonice_concentration,
                        std::vector<double>& albedos_out) const {
        albedos_out.resize(nonice_concentration.size());

        #pragma omp parallel for
        for (size_t i = 0; i < nonice_concentration.size(); ++i) {
            albedos_out[i] = computeAlbedo(nonice_concentration[i], i);
        }
    }

    /**
     * Compute maximum albedo change between two albedo arrays
     * @param old_albedos Previous per-face albedos
     * @param new_albedos Current per-face albedos
     * @return Maximum absolute albedo change across all faces
     */
    static double maxAlbedoChange(const std::vector<double>& old_albedos,
                                  const std::vector<double>& new_albedos) {
        double max_change = 0.0;
        for (size_t i = 0; i < old_albedos.size(); ++i) {
            max_change = std::max(max_change, std::abs(new_albedos[i] - old_albedos[i]));
        }
        return max_change;
    }

    /**
     * Compute per-face nonice_concentration from observed Bond albedo map
     * using percentile-based linear scaling.
     *
     * Bright faces (albedo near P_high) → c ≈ 0 (pure ice)
     * Dark faces  (albedo near P_low)  → c ≈ 1 (no ice)
     *
     * c(face) = (A_high - A_obs(face)) / (A_high - A_low), clamped [0, 1]
     *
     * @param observed_albedos Per-face observed Bond albedos from FITS map
     * @param percentile_low Lower percentile (e.g., 5 → darkest 5% get c=1)
     * @param percentile_high Upper percentile (e.g., 95 → brightest 5% get c=0)
     * @param concentrations_out Output per-face nonice_concentration [0-1]
     */
    static void computeConcentrationsFromAlbedoMap(
        const std::vector<double>& observed_albedos,
        double percentile_low,
        double percentile_high,
        std::vector<double>& concentrations_out)
    {
        size_t n = observed_albedos.size();
        concentrations_out.resize(n);

        if (n == 0) return;

        // Compute percentiles by sorting a copy
        std::vector<double> sorted_albedos(observed_albedos);
        std::sort(sorted_albedos.begin(), sorted_albedos.end());

        // Percentile indices (linear interpolation)
        double idx_low = (percentile_low / 100.0) * (n - 1);
        double idx_high = (percentile_high / 100.0) * (n - 1);

        size_t i_low = static_cast<size_t>(idx_low);
        size_t i_high = static_cast<size_t>(idx_high);
        double frac_low = idx_low - i_low;
        double frac_high = idx_high - i_high;

        double A_low = sorted_albedos[i_low] * (1.0 - frac_low)
                      + sorted_albedos[std::min(i_low + 1, n - 1)] * frac_low;
        double A_high = sorted_albedos[i_high] * (1.0 - frac_high)
                       + sorted_albedos[std::min(i_high + 1, n - 1)] * frac_high;

        double range = A_high - A_low;
        if (range < 1e-10) {
            // Degenerate case: all albedos similar
            std::fill(concentrations_out.begin(), concentrations_out.end(), 0.5);
            std::cout << "  Warning: albedo range too small for percentile mapping" << std::endl;
            return;
        }

        // Map each face: bright → c=0 (ice), dark → c=1 (no ice)
        double sum = 0.0;
        size_t n_at_zero = 0, n_at_one = 0;
        #pragma omp parallel for reduction(+:sum,n_at_zero,n_at_one)
        for (size_t i = 0; i < n; ++i) {
            double c = (A_high - observed_albedos[i]) / range;
            c = std::clamp(c, 0.0, 1.0);
            concentrations_out[i] = c;
            sum += c;
            if (c <= 0.0) n_at_zero++;
            if (c >= 1.0) n_at_one++;
        }

        std::cout << "\n=== Albedo-to-Concentration Mapping ===" << std::endl;
        std::cout << "  Percentile bounds: P" << percentile_low << " = " << A_low
                  << ", P" << percentile_high << " = " << A_high << std::endl;
        std::cout << "  Concentration: mean=" << std::fixed << std::setprecision(4)
                  << (sum / n * 100.0) << "%, min="
                  << (*std::min_element(concentrations_out.begin(), concentrations_out.end()) * 100.0)
                  << "%, max="
                  << (*std::max_element(concentrations_out.begin(), concentrations_out.end()) * 100.0)
                  << "%" << std::endl;
        std::cout << "  Faces at c=0 (pure ice): " << n_at_zero
                  << ", faces at c=1 (no ice): " << n_at_one << std::endl;
    }

    const CompositionAlbedoConfig& getConfig() const { return config_; }
    void setConfig(const CompositionAlbedoConfig& config) { config_ = config; }
};

} // namespace exosphere
} // namespace ptm

#endif // COMPOSITION_ALBEDO_H
