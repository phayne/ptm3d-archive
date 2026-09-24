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
 * - Ice availability is continuous: sublimation scales with ice fraction (1 - nonice_concentration),
 *   or, in the exposed-area mode (--ice-fraction-from-albedo=hapke-intimate), with the exposed
 *   ice area fraction f(1 - nonice_concentration); see solveExposedAreaStep()
 * - Self-limiting depletion: as ice is consumed, sublimation rate decreases proportionally
 * - Analytical exponential evolution: d(nonice)/dt = A*(1-nonice) - B is solved exactly,
 *   enabling accurate super-stepping over many epochs without numerical error
 * - Each face reaches equilibrium where reduced sublimation balances incoming deposition
 *
 * Key approximations:
 * - Transport matrix is fixed (computed once at start)
 * - Deposition is constant within a step at the global scale (per source group) whose
 *   deposition equals what the kernel retains from the step's own emission (PTM-644)
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
#include <limits>
#include <stdexcept>
#include <string>
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
 * One closed-form step of the exposed-area ice ODE
 * (--ice-fraction-from-albedo=hapke-intimate).
 *
 * With ice volume fraction phi = 1 - nonice_concentration and exposed ice area
 * fraction f(phi) = phi / (phi + r (1 - phi)), r = D_ice / D_nonice:
 *   dphi/dt = B - A f(phi)
 * A = base emission / max ice mass, B = deposition / max ice mass [1/s].
 * Separable: dt = (r - (r-1) phi) dphi / (E - D phi), D = B (r - 1) + A, E = B r,
 * integrated in closed form (see t_of below; finite for any sign of D, so r < 1
 * is covered). The interior equilibrium is phi_eq = E / D (A f(phi_eq) = B) when
 * D > E; otherwise phi rises to 1. phi_end is found by bisection on the
 * monotone t(phi). For r = 1 this reduces to the legacy
 * exponential solution. phi is clamped at 1 (deposition overflow is ignored,
 * as in the legacy model).
 *
 * mean_f is the time-mean exposed fraction, so that base_emission * mean_f is
 * the mean emission over the step: int f dt, from whichever of the exact balance
 * (B t - dphi)/A or the closed-form int phi/(E - D phi) dphi is well conditioned
 * (the trapezoid rule when phi moves by < 1e-8).
 */
struct ExposedAreaStep {
    double phi_end = 0.0;
    double mean_f = 0.0;
};

inline ExposedAreaStep solveExposedAreaStep(double phi0, double A, double B,
                                            double r, double dt) {
    ExposedAreaStep out;
    phi0 = std::clamp(phi0, 0.0, 1.0);
    auto f_of = [r](double p) {
        const double den = p + r * (1.0 - p);
        return (den > 0.0) ? p / den : 0.0;
    };
    const double f0 = f_of(phi0);
    if (dt <= 0.0) { out.phi_end = phi0; out.mean_f = f0; return out; }
    if (A <= 1e-30) {
        // No emission: pure deposition (linear), as in the legacy branch.
        out.phi_end = std::min(1.0, phi0 + B * dt);
        out.mean_f = 0.5 * (f0 + f_of(out.phi_end));
        return out;
    }

    const double rate0 = B - A * f0;           // dphi/dt at phi0
    if (rate0 == 0.0) { out.phi_end = phi0; out.mean_f = f0; return out; }

    // t(phi) = int_{phi0}^{phi} (r - (r-1)p) / (E - D p) dp,  D = B (r-1) + A,  E = B r.
    // Written with s = p - phi0, eps = D / g0 (g0 = E - D phi0, same sign as rate0) as
    //   t = [n0 L(s) - (r-1) M(s)] / g0,   n0 = r - (r-1) phi0,
    //   L = int ds/(1 - eps s),  M = int s ds/(1 - eps s),
    // which stays finite and accurate for any sign of D, including D -> 0 (r < 1).
    const double D = B * (r - 1.0) + A;
    const double E = B * r;
    const double g0 = E - D * phi0;
    const double n0 = r - (r - 1.0) * phi0;
    const double eps = D / g0;
    // L(s) = int_0^s du/(1 - eps u), M(s) = int_0^s u du/(1 - eps u).
    auto LM = [&](double sd, double& L, double& M) {
        const double y = eps * sd;             // < 1 along the trajectory
        if (std::abs(y) < 1e-3) {
            L = sd * (1.0 + y / 2.0 + y * y / 3.0 + y * y * y / 4.0);
            M = sd * sd * (0.5 + y / 3.0 + y * y / 4.0 + y * y * y / 5.0);
        } else {
            L = -std::log1p(-y) / eps;
            M = (L - sd) / eps;
        }
    };
    auto t_of = [&](double p) {
        double L, M;
        LM(p - phi0, L, M);
        return (n0 * L - (r - 1.0) * M) / g0;
    };
    // Exposure integral: f dt = f (dt/dphi) dphi = phi / (E - D phi) dphi, so
    //   int_0^t f dt' = (phi0 L(s) + M(s)) / g0.
    auto f_integral = [&](double p) {
        double L, M;
        LM(p - phi0, L, M);
        return (phi0 * L + M) / g0;
    };
    // Interior equilibrium phi_eq = E / D (A f(phi_eq) = B) exists only for D > E.
    const bool interior_eq = (D > 0.0) && (E < D);

    double lo, hi;
    bool clamped_at_one = false;
    double t_one = 0.0;
    if (rate0 > 0.0) {
        // phi increases toward phi_eq, or toward 1 when there is no interior equilibrium.
        lo = phi0;
        if (interior_eq) {
            hi = E / D;
        } else {
            hi = 1.0;
            if (E - D > 0.0) {                 // dphi/dt > 0 at phi = 1: reached in finite time
                t_one = t_of(1.0);
                if (dt >= t_one) clamped_at_one = true;
            }
        }
    } else {
        // phi decreases toward phi_eq (rate0 < 0 implies D > E / phi0 > 0).
        lo = std::max(0.0, E / D); hi = phi0;
    }

    // int f dt over [0, t] to phi: two exact, complementary forms.
    //   balance: (B t - (phi - phi0)) / A   -- cancels when A f << B (deposition-
    //            dominated growth, far from equilibrium);
    //   direct:  f_integral(phi)            -- log-singular at phi_eq.
    // Use the balance unless it loses more than 3 digits to cancellation; in that
    // regime phi is far from phi_eq and the direct form is well conditioned.
    auto exposure = [&](double p, double t) {
        const double dphi = p - phi0;
        const double bal = B * t - dphi;
        if (std::abs(bal) > 1e-3 * std::max(std::abs(B * t), std::abs(dphi))) return bal / A;
        return f_integral(p);
    };

    if (clamped_at_one) {
        out.phi_end = 1.0;
        // Interior segment [0, t_one], then f = 1 for the rest of the step.
        out.mean_f = std::clamp((exposure(1.0, t_one) + (dt - t_one)) / dt, f0, 1.0);
        return out;
    }

    for (int it = 0; it < 200 && hi - lo > 1e-16; ++it) {
        const double m = 0.5 * (lo + hi);
        const bool before = t_of(m) < dt;      // m is reached before dt
        if (rate0 > 0.0) { if (before) lo = m; else hi = m; }
        else             { if (before) hi = m; else lo = m; }
    }
    out.phi_end = 0.5 * (lo + hi);

    const double f1 = f_of(out.phi_end);
    const double f_lo = std::min(f0, f1), f_hi = std::max(f0, f1);
    // When phi barely moves, its bisection-resolved change (~1e-16) cannot carry
    // the integral; f is then constant to < r * 1e-8 and the trapezoid is exact
    // to that order.
    if (std::abs(out.phi_end - phi0) < 1e-8) {
        out.mean_f = 0.5 * (f0 + f1);
    } else {
        out.mean_f = std::clamp(exposure(out.phi_end, dt) / dt, f_lo, f_hi);
    }
    return out;
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

    // PTM-644: deposition over a step follows the emission produced over that step
    // (see MultiTimescaleEvolution::evolveEpoch). false reproduces the pre-fix
    // operator splitting: deposition held at its start-of-step value while the
    // sources follow the ice availability, and (linear availability mode) the
    // trapezoid mean of the availability.
    bool superstep_consistent_deposition = true;

    // Upper bound on one super-step, in epochs (0 = none). Used for step-size
    // refinement checks; the physical bound is computeSafeEpochCount().
    int max_superstep_epochs = 0;

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
        std::cout << "  Step deposition: "
                  << (superstep_consistent_deposition
                      ? "consistent with the step's emission (PTM-644)"
                      : "LEGACY start-of-step value (PTM-644 not applied; not mass-conserving)")
                  << std::endl;
        if (max_superstep_epochs > 0) {
            std::cout << "  Max super-step: " << max_superstep_epochs << " epochs" << std::endl;
        }
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

    // Emission-weighted temperature T_w,i = sum_t Phi_i(t) T_i(t) / sum_t Phi_i(t) [K]
    // (M1). The ballistic kernel's hop length scales with the temperature of
    // the molecules actually launched (d_char = 2 k T / (m g)), and sublimation
    // is exponential in T, so the launch-weighted temperature is far above the
    // diurnal mean (146 K vs 117 K on the delivered ico6 product). Using
    // avg_temperature for transport shortened every hop by ~1.25x. Same
    // Jensen's-inequality correction as avg_crystallization_rate above.
    // Faces with zero emission over the whole cycle launch nothing; for them
    // the diurnal mean is used and counted in emission_weighted_fallback_count.
    std::vector<double> emission_weighted_temperature;
    std::vector<double> emission_weight_sum;        // accumulator: sum_t Phi_i(t)
    size_t emission_weighted_fallback_count = 0;

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

    // PTM-642 sputter mass accounting. When sputter_transport is set,
    // avg_sputtering_flux / base_sputtering_flux are the GROSS sputter removal,
    // a fraction sputter_transport_share of it (the non-escaping H2O) enters the
    // ballistic transport, and deposition is kept as two parts, each rescaled by
    // the ratio of ITS OWN source (sublimation, sputter) to its cache-time value.
    // When it is clear (sputtering off, or --sputter-legacy-escape-only), the
    // original single (sublimation + sputtering) rescale runs unchanged.
    bool sputter_transport = false;
    double sputter_transport_share = 0.0;   // of gross: non-escaping H2O -> transport
    double sputter_escape_share = 0.0;      // of gross: escaping, all products
    double sputter_reservoir_share = 0.0;   // of gross: non-escaping O2 + H2
    std::vector<double> avg_deposition_sublimation_flux;  // [kg/(m²·s)] from sublimation
    std::vector<double> avg_deposition_sputter_flux;      // [kg/(m²·s)] from sputtered H2O
    std::vector<double> base_deposition_sublimation_flux;
    std::vector<double> base_deposition_sputter_flux;
    double total_base_sublimation = 0.0;  // [kg/s] sublimation source at cache time
    double total_base_sputter = 0.0;      // [kg/s] GROSS sputter removal at cache time

    // PTM-649. Per-face fate fractions of the gross sputter flux (diurnal means),
    // set when the composition varies with temperature (current parameter set:
    // tau(T) on the radiolytic part only). Empty = the global shares above.
    std::vector<double> sputter_face_transport_frac;
    std::vector<double> sputter_face_escape_frac;
    std::vector<double> sputter_face_reservoir_frac;
    double total_base_sputter_transport = 0.0;  // [kg/s] sputter transport source at cache time
    bool perFaceSputter() const { return !sputter_face_transport_frac.empty(); }
    double sputterTransportFrac(size_t i) const {
        return perFaceSputter() ? sputter_face_transport_frac[i] : sputter_transport_share;
    }
    double sputterEscapeFrac(size_t i) const {
        return perFaceSputter() ? sputter_face_escape_frac[i] : sputter_escape_share;
    }
    double sputterReservoirFrac(size_t i) const {
        return perFaceSputter() ? sputter_face_reservoir_frac[i] : sputter_reservoir_share;
    }

    // PTM-644. Deposition = base deposition x a global scale per source group
    // (one group, total emission, without sputter transport; sublimation and
    // sputter with it). The scales below are the CURRENT (start-of-step) values
    // set by cacheBaseFluxesUnscaled() (1) and applyScaledFluxes(); evolveEpoch()
    // replaces them over a step by the scales consistent with the step's
    // emission. The base deposition totals (area-weighted, summed in face order)
    // give each group's retention: deposited / emitted at cache time (sublimation:
    // 1 - Jeans - unscattered fraction; sputter: the non-escaping H2O share).
    double deposition_scale_total = 1.0;
    double deposition_scale_sub = 1.0;
    double deposition_scale_sput = 1.0;
    double total_base_deposition = 0.0;      // [kg/s]
    double total_base_deposition_sub = 0.0;  // [kg/s]
    double total_base_deposition_sput = 0.0; // [kg/s]

    explicit DiurnalAverages(size_t num_faces) {
        avg_sublimation_flux.resize(num_faces, 0.0);
        avg_deposition_flux.resize(num_faces, 0.0);
        avg_sputtering_flux.resize(num_faces, 0.0);
        avg_net_flux.resize(num_faces, 0.0);
        avg_escape_flux.resize(num_faces, 0.0);
        avg_temperature.resize(num_faces, 0.0);
        avg_crystallization_rate.resize(num_faces, 0.0);
        emission_weighted_temperature.resize(num_faces, 0.0);
        emission_weight_sum.resize(num_faces, 0.0);

        base_sublimation_flux.resize(num_faces, 0.0);
        base_sputtering_flux.resize(num_faces, 0.0);
        base_deposition_flux.resize(num_faces, 0.0);

        cached_ice_availability.resize(num_faces, 1.0);

        avg_deposition_sublimation_flux.resize(num_faces, 0.0);
        avg_deposition_sputter_flux.resize(num_faces, 0.0);
        base_deposition_sublimation_flux.resize(num_faces, 0.0);
        base_deposition_sputter_flux.resize(num_faces, 0.0);

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
        std::fill(emission_weighted_temperature.begin(), emission_weighted_temperature.end(), 0.0);
        std::fill(emission_weight_sum.begin(), emission_weight_sum.end(), 0.0);
        emission_weighted_fallback_count = 0;
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
        // Emission-weighted temperature (ratio of sums; no 1/n needed).
        size_t fallback = 0;
        for (size_t i = 0; i < emission_weighted_temperature.size(); ++i) {
            if (emission_weight_sum[i] > 0.0) {
                emission_weighted_temperature[i] /= emission_weight_sum[i];
            } else {
                emission_weighted_temperature[i] = avg_temperature[i];
                ++fallback;
            }
        }
        emission_weighted_fallback_count = fallback;
    }

    // Cache base (pure-ice potential) fluxes. On entry the avg_* fields have
    // ALREADY been scaled by the per-face ice availability, and
    // avg_deposition_flux comes from that scaled emission. The unscaled pure-ice
    // potential fluxes are stored directly as base_* (no availability floor), so
    // the exposure factor is applied dynamically and a face that holds no ice
    // today resumes sublimating once deposition gives it ice. total_base_emission
    // is the emission that produced base_deposition_flux, so the deposition scale
    // is 1 at cache time.
    //
    // (This replaced cacheBaseFluxes(), which divided the pure-ice diurnal
    // emission by the availability as though it were already scaled. That made
    // the first-epoch rate the pure-ice rate for any initial composition and left
    // deposition and sublimation out of step by the availability factor.)
    void cacheBaseFluxesUnscaled(const std::vector<double>& unscaled_sublimation,
                                 const std::vector<double>& unscaled_sputtering,
                                 const std::vector<double>& ice_availability,
                                 const std::vector<double>& face_areas) {
        total_base_emission = 0.0;
        total_base_sublimation = 0.0;
        total_base_sputter = 0.0;
        for (size_t i = 0; i < avg_sublimation_flux.size(); ++i) {
            base_sublimation_flux[i] = unscaled_sublimation[i];
            base_sputtering_flux[i] = unscaled_sputtering.empty() ? 0.0 : unscaled_sputtering[i];
            base_deposition_flux[i] = avg_deposition_flux[i];
            total_base_emission += (base_sublimation_flux[i] + base_sputtering_flux[i])
                                   * ice_availability[i] * face_areas[i];
            cached_ice_availability[i] = ice_availability[i];
            if (sputter_transport) {
                base_deposition_sublimation_flux[i] = avg_deposition_sublimation_flux[i];
                base_deposition_sputter_flux[i] = avg_deposition_sputter_flux[i];
                total_base_sublimation += base_sublimation_flux[i] * ice_availability[i] * face_areas[i];
                total_base_sputter += base_sputtering_flux[i] * ice_availability[i] * face_areas[i];
            }
        }
        total_base_sputter_transport = 0.0;
        if (sputter_transport && perFaceSputter()) {
            for (size_t i = 0; i < avg_sublimation_flux.size(); ++i) {
                total_base_sputter_transport += base_sputtering_flux[i] * sputter_face_transport_frac[i]
                                                * ice_availability[i] * face_areas[i];
            }
        }
        deposition_scale_total = deposition_scale_sub = deposition_scale_sput = 1.0;
        total_base_deposition = total_base_deposition_sub = total_base_deposition_sput = 0.0;
        for (size_t i = 0; i < avg_sublimation_flux.size(); ++i) {
            total_base_deposition += base_deposition_flux[i] * face_areas[i];
            total_base_deposition_sub += base_deposition_sublimation_flux[i] * face_areas[i];
            total_base_deposition_sput += base_deposition_sputter_flux[i] * face_areas[i];
        }
        cache_valid = true;
    }

    // Apply ice fraction scaling to cached base fluxes (fast operation).
    // Sets avg_* fields for diagnostics/export. evolveEpoch() uses base_* directly
    // with analytical exponential evolution for the actual concentration update.
    void applyScaledFluxes(const std::vector<double>& ice_availability,
                           const std::vector<double>& face_areas) {
        size_t n = avg_sublimation_flux.size();

        if (sputter_transport) {
            // PTM-642 (c): each deposition part follows the source that produced
            // it. The old single (sublimation + sputtering) ratio let deposition
            // exceed sublimation whenever the two sources lost ice availability
            // at different rates (sublimation is equatorial, sputtering trailing).
            // Face-order (deterministic) sums: an OpenMP reduction here made the
            // deposition scale, and so every face's state, vary by an ulp with the
            // thread count and run to run (PTM-645; flaked the PTM-644 equilibrium test).
            #pragma omp parallel for
            for (size_t i = 0; i < n; ++i) {
                avg_sublimation_flux[i] = base_sublimation_flux[i] * ice_availability[i];
                avg_sputtering_flux[i] = base_sputtering_flux[i] * ice_availability[i];
            }
            double s_now = 0.0, p_now = 0.0;
            for (size_t i = 0; i < n; ++i) {
                s_now += avg_sublimation_flux[i] * face_areas[i];
                p_now += avg_sputtering_flux[i] * face_areas[i];
            }
            const double scale_s = (total_base_sublimation > 1e-30) ? s_now / total_base_sublimation : 1.0;
            double scale_p = (total_base_sputter > 1e-30) ? p_now / total_base_sputter : 1.0;
            if (perFaceSputter()) {
                // The sputter deposition follows ITS source, the per-face transported share.
                double c_now = 0.0;
                for (size_t i = 0; i < n; ++i) {
                    c_now += avg_sputtering_flux[i] * sputter_face_transport_frac[i] * face_areas[i];
                }
                scale_p = (total_base_sputter_transport > 1e-30) ? c_now / total_base_sputter_transport : 1.0;
            }
            deposition_scale_sub = scale_s;
            deposition_scale_sput = scale_p;
            #pragma omp parallel for
            for (size_t i = 0; i < n; ++i) {
                avg_deposition_sublimation_flux[i] = base_deposition_sublimation_flux[i] * scale_s;
                avg_deposition_sputter_flux[i] = base_deposition_sputter_flux[i] * scale_p;
                avg_deposition_flux[i] = avg_deposition_sublimation_flux[i] + avg_deposition_sputter_flux[i];
                avg_net_flux[i] = avg_deposition_flux[i] - avg_sublimation_flux[i] - avg_sputtering_flux[i];
            }
            return;
        }

        // Pass 1: Compute scaled emission per face and global total (fused with reduction)
        double total_current_emission = 0.0;

        #pragma omp parallel for
        for (size_t i = 0; i < n; ++i) {
            avg_sublimation_flux[i] = base_sublimation_flux[i] * ice_availability[i];
            avg_sputtering_flux[i] = base_sputtering_flux[i] * ice_availability[i];
        }
        for (size_t i = 0; i < n; ++i) {  // face order: deterministic (PTM-645)
            total_current_emission += (avg_sublimation_flux[i] + avg_sputtering_flux[i]) * face_areas[i];
        }

        // Deposition scales with total emission (both now area-weighted [kg/s])
        double deposition_scale = (total_base_emission > 1e-30)
                                ? total_current_emission / total_base_emission
                                : 1.0;
        deposition_scale_total = deposition_scale;

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

    // PTM-642 sputter fates (set only when the new accounting is active; then
    // sputtering_rate is the GROSS removal). Rates are per step [kg/s]; the
    // redeposited rate is the start-of-step deposition the step used, while
    // sputtering_rate is averaged over the step's ice availability, so the two
    // differ by the operator-splitting lag of a super-step.
    bool sputter_partition_active = false;
    double sputter_redeposited_rate = 0.0;
    double sputter_escaped_rate = 0.0;
    double sputter_reservoir_rate = 0.0;
    double sublimation_deposition_rate = 0.0;
    double total_sputter_redeposited_mass = 0.0;  // [kg]
    double total_sputter_escaped_mass = 0.0;      // [kg]
    double total_sputter_reservoir_mass = 0.0;    // [kg]
    // Decomposition of the cumulative balance (new accounting only):
    // sublimated + sputtered - deposited = sputter escaped + reservoir
    //   + sublimation Jeans escape - super-step lag.
    // The lag is deposition in excess of the step-mean sources: a super-step
    // holds deposition at its start-of-step value while sublimation and
    // sputtering follow the step-mean ice availability (PTM-644).
    double total_sublimation_jeans_mass = 0.0;    // [kg]
    double total_superstep_lag_mass = 0.0;        // [kg]

    // PTM-644 step conservation audit (all multiscale modes; it throws only in
    // the consistent mode). Per step: deposited = retained emission, where each
    // source group retains rho_k = kernel deposition / kernel source at cache time.
    // Cumulative identity: sublimated + sputtered - deposited
    //   = not_redeposited - step_residual.
    bool superstep_consistent = false;
    double total_step_residual_mass = 0.0;     // [kg] deposited - retained emission
    double total_not_redeposited_mass = 0.0;   // [kg] emitted - retained (Jeans, unscattered, sputter escape + reservoir)
    double max_step_residual_rel = 0.0;        // max |residual| / emitted over steps
    double last_step_residual_rel = 0.0;
    double total_layer_overflow_mass = 0.0;    // [kg] deposition onto pure-ice faces beyond the layer
    double total_external_deposition_mass = 0.0;  // [kg] deposition with no cached source (synthetic only)
    double total_inventory_error_mass = 0.0;   // [kg] sum |state change - booked fluxes| (numerical)
    double max_inventory_error_rel = 0.0;      // per step, / emitted mass
    double last_step_scale_sub = 1.0, last_step_scale_sput = 0.0;
    long long step_audit_count = 0;
    long long steps_evolved = 0;
    long long step_solver_evaluations = 0;
    int step_solver_max_iterations = 0;
    // PTM-640 albedo step limiter (predictions do not change the state).
    long long step_solver_slope_fallbacks = 0;  // FD slope outside [-1, 0): used -1
    long long step_solver_loose = 0;            // accepted at |residual| <= 1e-8 after stagnation
    long long predictions = 0;
    long long prediction_evaluations = 0;
    long long steps_albedo_limited = 0;

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
                std::cout << "    Sputtering:  " << sputtering_rate << " kg/s"
                          << (sputter_partition_active ? " (gross removal)" : " (escaping share only; legacy)")
                          << std::endl;
            }
            if (sputter_partition_active) {
                std::cout << "      deposition from sublimation: " << sublimation_deposition_rate << " kg/s" << std::endl;
                std::cout << "      sputter redeposited:         " << sputter_redeposited_rate << " kg/s" << std::endl;
                std::cout << "      sputter escaped:             " << sputter_escaped_rate << " kg/s" << std::endl;
                std::cout << "      sputter reservoir (O2+H2):   " << sputter_reservoir_rate << " kg/s" << std::endl;
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
            if (sputter_partition_active) {
                std::cout << "      sputter redeposited: " << total_sputter_redeposited_mass << " kg" << std::endl;
                std::cout << "      sputter escaped:     " << total_sputter_escaped_mass << " kg" << std::endl;
                std::cout << "      sputter reservoir:   " << total_sputter_reservoir_mass << " kg" << std::endl;
                std::cout << "    Sublimation Jeans escape: " << total_sublimation_jeans_mass << " kg" << std::endl;
                std::cout << "    Super-step lag (deposition in excess of step-mean sources; PTM-644): "
                          << total_superstep_lag_mass << " kg" << std::endl;
                std::cout << "    (sublimated + sputtered - deposited = sputter escaped + reservoir + "
                          << "Jeans - lag)" << std::endl;
            }
            std::cout << "    Total escaped:    " << total_escaped_mass << " kg" << std::endl;
            if (step_audit_count > 0) {
                std::cout << "  Step conservation (PTM-644" << (superstep_consistent ? "" : "; LEGACY, not enforced")
                          << "): not redeposited " << total_not_redeposited_mass << " kg, residual "
                          << total_step_residual_mass << " kg (max step " << max_step_residual_rel
                          << " rel), layer overflow " << total_layer_overflow_mass << " kg" << std::endl;
            }
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

    // Per-face scratch for evolveEpoch()'s face-order sums (PTM-644).
    struct StepBuffers {
        std::vector<double> emit_sub, emit_sput, dep_sub, dep_sput, deposition, overflow, inventory_err;
        std::vector<double> sput_transport, sput_escape, sput_reservoir;
        void resize(size_t n) {
            if (emit_sub.size() == n) return;
            for (auto* v : {&emit_sub, &emit_sput, &dep_sub, &dep_sput, &deposition, &overflow, &inventory_err,
                            &sput_transport, &sput_escape, &sput_reservoir}) {
                v->assign(n, 0.0);
            }
        }
    };
    StepBuffers step_buffers_;
    bool predict_skip_solve_ = false;

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
            diurnal_avg_->emission_weighted_temperature[i] += emission_flux[i] * temperatures[i];
            diurnal_avg_->emission_weight_sum[i] += emission_flux[i];

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
     * Cache the pure-ice potential base fluxes (all modes). The diurnal averages
     * must already be scaled by the ice availability factor.
     */
    void cacheBaseFluxesUnscaled(const IceLayerModel& ice_model,
                                 const IceLayerState& ice_state,
                                 const std::vector<double>& unscaled_sublimation,
                                 const std::vector<double>& unscaled_sputtering,
                                 const std::vector<double>& face_areas,
                                 int epoch) {
        for (size_t i = 0; i < num_faces_; ++i) {
            ice_availability_buf_[i] = ice_model.getIceAvailabilityFactor(ice_state, i);
        }
        diurnal_avg_->cacheBaseFluxesUnscaled(unscaled_sublimation, unscaled_sputtering,
                                              ice_availability_buf_, face_areas);
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
     *
     * NOTE (M3): this returns true only while the cache is empty. There is no
     * periodic or concentration-threshold refresh; the CLI refuses
     * --cache-refresh and --no-analytical. Appendix D.7.4/D.7.6 describing an
     * "adaptive cache refresh" is inaccurate (doc fix owed).
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
     * One face over one step with deposition held at B (PTM-644 helper).
     * conc_before: non-ice concentration at the start; A, B [1/s] as in
     * evolveEpoch(). mean_avail is the step-mean ice availability, so that
     * base_emission * mean_avail is the step-mean emission.
     */
    struct FaceStep {
        double nonice_new = 0.0;   // unclamped end-of-step concentration
        double mean_avail = 0.0;   // step-mean availability
        double net_rate = 0.0;     // d(nonice)/dt at the start [1/s]
    };
    FaceStep stepFace(const IceLayerModel& ice_model, double conc_before, double A, double B,
                      double dt, bool exact_linear_mean) const {
        FaceStep out;
        if (ice_model.usesExposedAreaFraction()) {
            // Exposed-area mode: d(phi)/dt = B - A f(phi), solved in closed form.
            const double phi0 = 1.0 - conc_before;
            const ExposedAreaStep st =
                solveExposedAreaStep(phi0, A, B, ice_model.getExposedAreaGrainRatio(), dt);
            out.nonice_new = 1.0 - st.phi_end;
            out.mean_avail = st.mean_f;
            out.net_rate = A * ice_model.exposedAreaFraction(phi0) - B;
            return out;
        }
        if (exact_linear_mean) {
            // Well-conditioned closed form (PTM-644). With f = 1 - nonice, y = A t and
            // rate0 = B - A f0 (df/dt at the start):
            //   f(t) - f0 = rate0 t phi1(y),               phi1(y) = (1 - e^-y) / y
            //   int_0^t f = t [f0 + rate0 t psi(y)],       psi(y) = (y - 1 + e^-y) / y^2
            // The legacy form nonice_eq - (nonice_eq - c0) e^-y loses ulp(B/A) when a
            // face barely emits (A t << 1, B/A >> 1: every cold receiving face), and
            // its trapezoid mean is not the mean of an exponential. f is clamped at 1
            // (pure ice; deposition beyond it overflows the layer).
            auto phi1 = [](double y) {
                return (y < 1e-8) ? 1.0 - 0.5 * y : -std::expm1(-y) / y;
            };
            auto psi = [](double y) {
                if (y < 1e-2) return 0.5 - y / 6.0 + y * y / 24.0 - y * y * y / 120.0 + y * y * y * y / 720.0;
                return (y + std::expm1(-y)) / (y * y);
            };
            const double f0 = 1.0 - conc_before;
            const double rate0 = B - A * f0;
            double nonice_new;
            if (A > 1e-30 && B <= 2.0 * A) {
                // |nonice_eq| <= 1: the legacy closed form is well conditioned; kept so
                // that faces at (or near) equilibrium evolve bit for bit as before.
                const double nonice_eq = 1.0 - B / A;
                nonice_new = nonice_eq - (nonice_eq - conc_before) * std::exp(-A * dt);
            } else {
                nonice_new = conc_before - rate0 * dt * phi1(A * dt);
            }
            double f_end = 1.0 - std::clamp(nonice_new, 0.0, 1.0);
            out.nonice_new = nonice_new;
            out.net_rate = A * (1.0 - conc_before) - B;
            if (rate0 == 0.0) {
                // Exact equilibrium: f is constant; the trapezoid of the end points keeps
                // such faces bit-identical to the legacy step. (An absolute |df| < 1e-8
                // switch here made the step mean discontinuous in B on nearly depleted
                // faces, f ~ 1e-6, and stalled the deposition solve; PTM-640 work.)
                out.mean_avail = 0.5 * (f0 + f_end);
                return out;
            }
            double integral;
            if (nonice_new >= 0.0) {
                integral = dt * (f0 + rate0 * dt * psi(A * dt));
            } else {
                // f reaches 1 at t1 (rate0 > 0 and B > A there), then stays at 1.
                const double t1 = (A > 0.0) ? -std::log1p(-A * (1.0 - f0) / rate0) / A
                                            : (1.0 - f0) / rate0;
                integral = t1 * (f0 + rate0 * t1 * psi(A * t1)) + (dt - t1);
            }
            const double lo = std::min(f0, f_end), hi = std::max(f0, f_end);
            out.mean_avail = std::clamp(integral / dt, lo, hi);
            return out;
        }
        if (A > 1e-30) {
            double nonice_eq = 1.0 - B / A;
            double decay = std::exp(-A * dt);
            out.nonice_new = nonice_eq - (nonice_eq - conc_before) * decay;
        } else {
            // No emission: pure deposition (linear)
            out.nonice_new = conc_before - B * dt;
        }
        double ice_frac_before = 1.0 - conc_before;
        double ice_frac_after = 1.0 - std::clamp(out.nonice_new, 0.0, 1.0);
        out.mean_avail = 0.5 * (ice_frac_before + ice_frac_after);
        out.net_rate = A * (1.0 - conc_before) - B;
        return out;
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
     * Deposition over the step (PTM-644). Deposition is the transport kernel's
     * output at cache time times a global scale per source group (s_sub, s_sput;
     * one group without sputter transport), because the kernel is linear in its
     * source and the pattern is fixed until the next full diurnal. Pre-fix, the
     * scale was frozen at its start-of-step value while every face's emission
     * followed its own availability trajectory, so a step deposited more than it
     * emitted whenever availability fell (Europa uniform-hapke phase 0, sputtering
     * off: +0.24% cumulative, 2.1e11 kg of ice from nothing; the lag halved with
     * each halving of the step). Now B is held at the scale whose deposition equals what the
     * kernel retains from the step's actual emission:
     *     s_k * S_k,base = sum_i a_i E_k,i <f_i>(B(s))      for each group k,
     * where <f_i> is the exact step-mean availability for that constant B. This
     * is the implicit (self-consistent) coupling of the source and its deposition
     * over the step; within the step the timing of deposition is approximated by
     * its mean, so the pattern error is O(the step's change in emission), which
     * computeSafeEpochCount() bounds, and it vanishes under step refinement
     * (--max-superstep-epochs). (Temperatures and the albedo-coupling refresh are
     * still evaluated only at step boundaries; that lag is PTM-640, not this.)
     * The mass balance is exact for every step:
     *     deposited_k = rho_k * emitted_k,  rho_k = D_k,base / S_k,base
     * (sublimation: 1 - Jeans - unscattered; sputter: non-escaping H2O share).
     * The scalar system is solved by Newton on s (1-2 unknowns; finite-difference
     * Jacobian; it is affine in s in the linear availability mode). All sums that
     * feed the solve are taken in face order, so the result is reproducible.
     * The step audit throws above 1e-6 relative. When the start-of-step scale is
     * already consistent (constant availability), it is kept bit for bit.
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
        evolveEpochImpl(ice_model, ice_state, face_areas, epoch_duration, nullptr);
    }

    /**
     * PTM-640: the non-ice concentrations evolveEpoch() WOULD produce over
     * epoch_duration from the current state (same deposition solve and closed
     * form), without changing the state or the statistics.
     */
    void predictConcentrations(const IceLayerModel& ice_model,
                               const IceLayerState& ice_state,
                               const std::vector<double>& face_areas,
                               double epoch_duration,
                               std::vector<double>& nonice_out,
                               bool solve_deposition = true) {
        // solve_deposition = false (PTM-657): hold deposition at its start-of-step
        // scale instead of solving the step-consistent scale; ~7x cheaper, and
        // adequate for the albedo step limiter, which only needs where the albedo
        // is heading (the step itself is still solved).
        predict_skip_solve_ = !solve_deposition;
        const EvolutionStatistics saved = stats_;
        nonice_out.resize(num_faces_);
        evolveEpochImpl(ice_model, const_cast<IceLayerState&>(ice_state), face_areas, epoch_duration,
                        &nonice_out);
        const long long evals = stats_.step_solver_evaluations - saved.step_solver_evaluations;
        stats_ = saved;
        stats_.prediction_evaluations += evals;
        ++stats_.predictions;
        predict_skip_solve_ = false;
    }

private:
    void evolveEpochImpl(const IceLayerModel& ice_model,
                         IceLayerState& ice_state,
                         const std::vector<double>& face_areas,
                         double epoch_duration,
                         std::vector<double>* predict_only) {
        const DiurnalAverages& d = *diurnal_avg_;
        double max_ice_mass = ice_model.getMaxIceMassPerArea();
        const bool consistent = config_.superstep_consistent_deposition;
        const bool two = d.sputter_transport;
        double max_change = 0.0;
        double epoch_sublimated = 0.0, epoch_deposited = 0.0, epoch_sputtered = 0.0;
        step_buffers_.resize(num_faces_);

        // Source groups: 0 = sublimation (+ sputtering when it is not transported), 1 = sputter.
        // Group 1 is the sputter transport source: the gross, or (per-face composition,
        // PTM-649) the per-face transported share of it.
        const bool per_face = two && d.perFaceSputter();
        const double S_base[2] = {two ? d.total_base_sublimation : d.total_base_emission,
                                  two ? (per_face ? d.total_base_sputter_transport : d.total_base_sputter) : 0.0};
        auto w1 = [&](size_t i) { return per_face ? d.sputter_face_transport_frac[i] : 1.0; };
        const double D_base[2] = {two ? d.total_base_deposition_sub : d.total_base_deposition,
                                  two ? d.total_base_deposition_sput : 0.0};
        const bool active[2] = {S_base[0] > 1e-30, two && S_base[1] > 1e-30};
        double s[2] = {two ? d.deposition_scale_sub : d.deposition_scale_total,
                       two ? d.deposition_scale_sput : 0.0};
        auto dep_at = [&](size_t i, const double* sc) {
            return two ? d.base_deposition_sublimation_flux[i] * sc[0] + d.base_deposition_sputter_flux[i] * sc[1]
                       : d.base_deposition_flux[i] * sc[0];
        };

        // Step-mean emission per group [kg/s] for deposition scales sc (face-order sums).
        auto evaluate = [&](const double* sc, double* G) {
            #pragma omp parallel for
            for (size_t i = 0; i < num_faces_; ++i) {
                const double bs = d.base_sublimation_flux[i], bp = d.base_sputtering_flux[i];
                const FaceStep st = stepFace(ice_model, ice_state.nonice_concentration[i],
                                             (bs + bp) / max_ice_mass, dep_at(i, sc) / max_ice_mass,
                                             epoch_duration, true);
                step_buffers_.emit_sub[i] = bs * st.mean_avail * face_areas[i];
                step_buffers_.emit_sput[i] = bp * w1(i) * st.mean_avail * face_areas[i];
            }
            G[0] = G[1] = 0.0;
            for (size_t i = 0; i < num_faces_; ++i) G[0] += step_buffers_.emit_sub[i];
            for (size_t i = 0; i < num_faces_; ++i) G[1] += step_buffers_.emit_sput[i];
            if (!two) { G[0] += G[1]; G[1] = 0.0; }
            ++stats_.step_solver_evaluations;
        };
        auto residual = [&](const double* G, const double* sc, double* g) {
            for (int k = 0; k < 2; ++k) g[k] = active[k] ? G[k] / S_base[k] - sc[k] : 0.0;
        };
        auto converged_at = [&](const double* g, const double* sc) {
            for (int k = 0; k < 2; ++k) {
                if (active[k] && !(std::abs(g[k]) <= 1e-12 * std::max(sc[k], 1e-6))) return false;
            }
            return true;
        };
        auto gnorm = [](const double* g) { return std::max(std::abs(g[0]), std::abs(g[1])); };

        // The kernel cannot retain more than its source: rho_sub <= 1 (Jeans and
        // unscattered losses only), rho_sput <= the non-escaping H2O share.
        if (consistent) {
            const double rho_max[2] = {1.0, (two && !per_face) ? d.sputter_transport_share : 1.0};
            for (int k = 0; k < 2; ++k) {
                if (active[k] && D_base[k] > S_base[k] * rho_max[k] * (1.0 + 1e-6)) {
                    std::ostringstream m;
                    m << std::scientific << std::setprecision(6) << "evolveEpoch (PTM-644): cached deposition "
                      << D_base[k] << " kg/s exceeds its source's retainable share " << S_base[k] * rho_max[k]
                      << " kg/s (group " << k << ")";
                    throw std::runtime_error(m.str());
                }
            }
        }

        int iterations = 0;
        if (consistent && (active[0] || active[1]) && !(predict_only && predict_skip_solve_)) {
            double G[2], g[2];
            evaluate(s, G);
            residual(G, s, g);
            while (!converged_at(g, s)) {
                // Stagnation: the scale no longer moves (a kink at the root). Accept
                // when the residual is far inside the 1e-6 step audit, count it.
                if (iterations >= 8 && gnorm(g) <= 1e-8) {
                    ++stats_.step_solver_loose;
                    break;
                }
                if (++iterations > 40) {
                    std::ostringstream m;
                    m << std::scientific << std::setprecision(6)
                      << "evolveEpoch (PTM-644): step deposition scale did not converge in 40 Newton iterations"
                      << " (s = " << s[0] << ", " << s[1] << "; residual " << g[0] << ", " << g[1] << ")";
                    throw std::runtime_error(m.str());
                }
                // Finite-difference Jacobian of g (inactive groups: identity, g = 0).
                double J[2][2] = {{1.0, 0.0}, {0.0, 1.0}};
                for (int k = 0; k < 2; ++k) {
                    if (!active[k]) continue;
                    double sp[2] = {s[0], s[1]}, Gp[2], gp[2];
                    const double h = 1e-6 * std::max(s[k], 1e-3);
                    sp[k] += h;
                    evaluate(sp, Gp);
                    residual(Gp, sp, gp);
                    for (int j = 0; j < 2; ++j) J[j][k] = active[j] ? (gp[j] - g[j]) / h : (j == k ? 1.0 : 0.0);
                }
                // g_k = G_k/S_k - s_k decreases in s_k with slope in [-1, 0) (more
                // deposition, more ice, more emission, but never more than deposited).
                // A finite difference across a kink in the per-face closed forms can
                // violate that; fall back to the fixed-point slope -1 for that group.
                for (int k = 0; k < 2; ++k) {
                    if (active[k] && !(J[k][k] < 0.0 && J[k][k] >= -1.0 - 1e-6)) {
                        J[k][k] = -1.0;
                        J[k][1 - k] = 0.0;
                        ++stats_.step_solver_slope_fallbacks;
                    }
                }
                const double det = J[0][0] * J[1][1] - J[0][1] * J[1][0];
                if (!(std::abs(det) > 0.0) || !std::isfinite(det)) {
                    throw std::runtime_error("evolveEpoch (PTM-644): singular Jacobian in the step deposition solve");
                }
                const double dx0 = (-g[0] * J[1][1] + g[1] * J[0][1]) / det;
                const double dx1 = (-g[1] * J[0][0] + g[0] * J[1][0]) / det;
                // Damped update (g is monotone and nearly affine; damping only guards clamps).
                double lambda = 1.0, sn[2], Gn[2], gn[2];
                for (int ls = 0; ; ++ls) {
                    sn[0] = active[0] ? std::max(0.0, s[0] + lambda * dx0) : s[0];
                    sn[1] = active[1] ? std::max(0.0, s[1] + lambda * dx1) : s[1];
                    evaluate(sn, Gn);
                    residual(Gn, sn, gn);
                    if (gnorm(gn) < gnorm(g) || converged_at(gn, sn) || ls >= 30) break;
                    lambda *= 0.5;
                }
                s[0] = sn[0]; s[1] = sn[1]; g[0] = gn[0]; g[1] = gn[1];
            }
        }
        if (predict_only) {
            // Prediction: the end-of-step concentrations only (state untouched).
            #pragma omp parallel for
            for (size_t i = 0; i < num_faces_; ++i) {
                const double deposition = consistent ? dep_at(i, s) : d.avg_deposition_flux[i];
                const double A = (d.base_sublimation_flux[i] + d.base_sputtering_flux[i]) / max_ice_mass;
                const FaceStep st = stepFace(ice_model, ice_state.nonice_concentration[i], A,
                                             deposition / max_ice_mass, epoch_duration, consistent);
                (*predict_only)[i] = std::clamp(st.nonice_new, 0.0, layer_constants::MAX_NONICE_CONCENTRATION);
            }
            return;
        }
        stats_.step_solver_max_iterations = std::max(stats_.step_solver_max_iterations, iterations);
        ++stats_.steps_evolved;

        #pragma omp parallel for reduction(max:max_change) \
            reduction(+:epoch_sublimated,epoch_deposited,epoch_sputtered)
        for (size_t i = 0; i < num_faces_; ++i) {
            double conc_before = ice_state.nonice_concentration[i];

            // Base (unscaled) emission rates and deposition
            double base_sublim = d.base_sublimation_flux[i];
            double base_sput = d.base_sputtering_flux[i];
            double deposition = consistent ? dep_at(i, s) : d.avg_deposition_flux[i];

            // Analytical exponential evolution:
            // The ODE is: d(nonice)/dt = [base_emission * (1-nonice) - deposition] / max_ice_mass
            // where sublimation/sputtering scale with ice fraction (1-nonice).
            // Let A = (base_sublim + base_sput) / max_ice_mass  [emission rate constant]
            //     B = deposition / max_ice_mass                  [deposition rate]
            // ODE: d(nonice)/dt = A*(1-nonice) - B = A - B - A*nonice
            // Solution: nonice(t) = nonice_eq - (nonice_eq - nonice_0) * exp(-A*t)
            //   where nonice_eq = 1 - B/A  (equilibrium concentration)
            // (Exposed-area mode: d(phi)/dt = B - A f(phi); see stepFace().)
            double A = (base_sublim + base_sput) / max_ice_mass;
            double B = deposition / max_ice_mass;

            const FaceStep st = stepFace(ice_model, conc_before, A, B, epoch_duration, consistent);
            const double nonice_new = st.nonice_new;
            const double eff_sublimation = base_sublim * st.mean_avail;
            const double eff_sputtering = base_sput * st.mean_avail;

            // Accumulate area-weighted mass balance
            epoch_sublimated += eff_sublimation * face_areas[i];
            epoch_deposited += deposition * face_areas[i];
            epoch_sputtered += eff_sputtering * face_areas[i];

            // Store concentration change rate (instantaneous, for super-step estimation)
            concentration_change_rate_[i] = st.net_rate;

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

            // PTM-644 audit terms, summed in face order below.
            step_buffers_.deposition[i] = deposition;
            step_buffers_.emit_sub[i] = eff_sublimation * face_areas[i];
            step_buffers_.emit_sput[i] = eff_sputtering * face_areas[i];
            if (per_face) {
                step_buffers_.sput_transport[i] = eff_sputtering * d.sputter_face_transport_frac[i] * face_areas[i];
                step_buffers_.sput_escape[i] = eff_sputtering * d.sputter_face_escape_frac[i] * face_areas[i];
                step_buffers_.sput_reservoir[i] = eff_sputtering * d.sputter_face_reservoir_frac[i] * face_areas[i];
            }
            step_buffers_.dep_sub[i] = (two ? (consistent ? d.base_deposition_sublimation_flux[i] * s[0]
                                                          : d.avg_deposition_sublimation_flux[i])
                                            : deposition) * face_areas[i];
            step_buffers_.dep_sput[i] = two ? deposition * face_areas[i] - step_buffers_.dep_sub[i] : 0.0;
            // Ice inventory: the state change must equal (deposition - emission) x dt,
            // except deposition that overflows a pure-ice face.
            const double dphi = conc_before - ice_state.nonice_concentration[i];
            const double booked = (B - A * st.mean_avail) * epoch_duration;
            const bool pure_ice_end = (ice_state.nonice_concentration[i] <= 0.0);
            const double over = (pure_ice_end && booked > dphi) ? booked - dphi : 0.0;
            step_buffers_.overflow[i] = over * max_ice_mass * face_areas[i];
            step_buffers_.inventory_err[i] = std::abs(booked - over - dphi) * max_ice_mass * face_areas[i];
        }

        // Face-order (reproducible) sums of the step.
        // Gs, Gp, Ds, Dp [kg/s]; over, inv_err [kg] over the step.
        double Gs = 0.0, Gp = 0.0, Ds = 0.0, Dp = 0.0, over = 0.0, inv_err = 0.0;
        double Gc = 0.0, Ge = 0.0, Gr = 0.0;  // per-face sputter fates [kg/s] (PTM-649)
        if (per_face) {
            for (size_t i = 0; i < num_faces_; ++i) {
                Gc += step_buffers_.sput_transport[i];
                Ge += step_buffers_.sput_escape[i];
                Gr += step_buffers_.sput_reservoir[i];
            }
        }
        for (size_t i = 0; i < num_faces_; ++i) {
            Gs += step_buffers_.emit_sub[i];
            Gp += step_buffers_.emit_sput[i];
            Ds += step_buffers_.dep_sub[i];
            Dp += step_buffers_.dep_sput[i];
            over += step_buffers_.overflow[i];
            inv_err += step_buffers_.inventory_err[i];
        }
        if (consistent) {
            // Stats from the reproducible sums (the OpenMP reductions above vary at
            // ~1e-16 run to run; PTM-645).
            epoch_sublimated = Gs;
            epoch_sputtered = Gp;
            epoch_deposited = Ds + Dp;
        }

        // Step conservation audit (PTM-644): deposited = retained share of what
        // the step emitted, per group.
        const double rho0 = active[0] ? D_base[0] / S_base[0] : 0.0;
        const double rho1 = active[1] ? D_base[1] / S_base[1] : 0.0;
        const double emitted = Gs + Gp;
        const double retained = two ? rho0 * Gs + rho1 * (per_face ? Gc : Gp) : rho0 * emitted;
        // A group with deposition but no source at cache time (only in synthetic
        // setups: the kernel conserves its source) is an external supply; it is
        // reported separately and excluded from the audit, never silently.
        const double external = ((!active[0] && D_base[0] > 0.0) ? Ds : 0.0) +
                                ((!active[1] && D_base[1] > 0.0) ? Dp : 0.0);
        if (external > 0.0 && stats_.total_external_deposition_mass == 0.0) {
            std::cerr << "WARNING [PTM-644]: deposition with no cached source (external supply, "
                      << std::scientific << std::setprecision(3) << external
                      << " kg/s); it is excluded from the step conservation audit and reported as "
                      << "external deposition" << std::fixed << std::endl;
        }
        stats_.total_external_deposition_mass += external * epoch_duration;
        const double deposited = Ds + Dp - external;
        const double step_residual = deposited - retained;
        const double step_rel = (emitted > 0.0) ? step_residual / emitted
                              : (deposited > 0.0 ? std::numeric_limits<double>::infinity() : 0.0);
        const double inv_rel = (emitted > 0.0) ? inv_err / (emitted * epoch_duration) : 0.0;
        stats_.superstep_consistent = consistent;
        stats_.last_step_residual_rel = step_rel;
        stats_.max_step_residual_rel = std::max(stats_.max_step_residual_rel, std::abs(step_rel));
        stats_.max_inventory_error_rel = std::max(stats_.max_inventory_error_rel, inv_rel);
        stats_.total_step_residual_mass += step_residual * epoch_duration;
        stats_.total_not_redeposited_mass += (emitted - retained) * epoch_duration;
        stats_.total_layer_overflow_mass += over;      // already [kg] over the step
        stats_.total_inventory_error_mass += inv_err;  // already [kg] over the step
        stats_.last_step_scale_sub = s[0];
        stats_.last_step_scale_sput = s[1];
        stats_.step_audit_count++;
        if (consistent && !(std::abs(step_rel) <= 1e-6)) {
            std::ostringstream m;
            m << std::scientific << std::setprecision(6)
              << "evolveEpoch (PTM-644): step conservation audit failed: deposited " << deposited
              << " kg/s vs retained emission " << retained << " kg/s (emitted " << emitted
              << "), residual " << step_rel << " relative (limit 1e-6)";
            throw std::runtime_error(m.str());
        }

        // Update statistics
        stats_.max_concentration_change = max_change;
        stats_.total_sublimated_mass += epoch_sublimated * epoch_duration;
        stats_.total_deposited_mass += epoch_deposited * epoch_duration;
        stats_.total_sputtered_mass += epoch_sputtered * epoch_duration;

        // Net surface mass loss = sublimation + sputtering - deposition. With the PTM-642
        // accounting this is escape + atmospheric reservoir (+ Jeans), minus the super-step lag.
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

        if (d.sputter_transport && consistent) {
            // Deposition is the retained share of the step's own sources, so the
            // Jeans term is (1 - rho_sub) x sublimated and the lag is the audit residual.
            stats_.total_sublimation_jeans_mass += (1.0 - rho0) * Gs * epoch_duration;
            stats_.total_superstep_lag_mass += step_residual * epoch_duration;
            stats_.sputter_redeposited_rate = Dp;
            stats_.sublimation_deposition_rate = Ds;
            stats_.sputter_escaped_rate = per_face ? Ge : d.sputter_escape_share * epoch_sputtered;
            stats_.sputter_reservoir_rate = per_face ? Gr : d.sputter_reservoir_share * epoch_sputtered;
            stats_.total_sputter_redeposited_mass += Dp * epoch_duration;
            stats_.total_sputter_escaped_mass += stats_.sputter_escaped_rate * epoch_duration;
            stats_.total_sputter_reservoir_mass += stats_.sputter_reservoir_rate * epoch_duration;
        } else if (d.sputter_transport) {
            // Legacy (pre-PTM-644) diagnostics, unchanged. Separate pass so the reductions above are untouched.
            double dep_sput = 0.0, dep_sub = 0.0, s0 = 0.0;
            for (size_t i = 0; i < num_faces_; ++i) {
                dep_sput += d.avg_deposition_sputter_flux[i] * face_areas[i];
                dep_sub += d.avg_deposition_sublimation_flux[i] * face_areas[i];
                s0 += d.avg_sublimation_flux[i] * face_areas[i];  // start of step
            }
            // Deposition from sublimation had the step followed the step-mean source.
            const double dep_sub_mean = (s0 > 0.0) ? dep_sub * (epoch_sublimated / s0) : 0.0;
            const double lag_rate = (dep_sub - dep_sub_mean)
                + (dep_sput - d.sputter_transport_share * epoch_sputtered);
            stats_.total_sublimation_jeans_mass += (epoch_sublimated - dep_sub_mean) * epoch_duration;
            stats_.total_superstep_lag_mass += lag_rate * epoch_duration;
            stats_.sputter_redeposited_rate = dep_sput;
            stats_.sublimation_deposition_rate = dep_sub;
            stats_.sputter_escaped_rate = d.sputter_escape_share * epoch_sputtered;
            stats_.sputter_reservoir_rate = d.sputter_reservoir_share * epoch_sputtered;
            stats_.total_sputter_redeposited_mass += dep_sput * epoch_duration;
            stats_.total_sputter_escaped_mass += stats_.sputter_escaped_rate * epoch_duration;
            stats_.total_sputter_reservoir_mass += stats_.sputter_reservoir_rate * epoch_duration;
        }

        // Multi-criteria convergence:
        // (a) Concentration change below threshold, OR
        // (b) Flux imbalance small AND concentration change rate is low
        stats_.converged = (max_change < config_.convergence_threshold)
            || (stats_.flux_imbalance_fraction < config_.flux_convergence_threshold
                && max_change < 1e-4);
    }

public:
    /**
     * Deposition flux [kg/(m²·s)] that the last evolveEpoch() step used, per face
     * (the step-consistent value; the start-of-step value in legacy mode).
     */
    const std::vector<double>& getStepDepositionFlux() const { return step_buffers_.deposition; }

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
     * Activate the PTM-642 sputter mass accounting (call before the first
     * epoch). Shares are fractions of the gross sputtered mass and must sum to 1.
     */
    void setSputterPartition(double transport_share, double escape_share, double reservoir_share) {
        const double sum = transport_share + escape_share + reservoir_share;
        if (!(std::abs(sum - 1.0) <= 1e-12) || transport_share < 0.0 || escape_share < 0.0 ||
            reservoir_share < 0.0) {
            throw std::invalid_argument("setSputterPartition: shares must be >= 0 and sum to 1 (sum = " +
                                        std::to_string(sum) + ")");
        }
        diurnal_avg_->sputter_transport = true;
        diurnal_avg_->sputter_transport_share = transport_share;
        diurnal_avg_->sputter_escape_share = escape_share;
        diurnal_avg_->sputter_reservoir_share = reservoir_share;
        stats_.sputter_partition_active = true;
    }

    /**
     * Global sputter mass audit of the CURRENT flux state [kg/s].
     * gross = redeposited + escaped + reservoir must close to 1e-6 (the caller throws);
     * redeposited is MEASURED from the deposition field (the kernel output after
     * rescaling), not assumed. Also checks the PTM-642 (c) regression:
     * deposition <= sublimation + sputter transport source.
     */
    struct SputterAudit {
        double gross = 0.0, transport_source = 0.0, redeposited = 0.0, escaped = 0.0, reservoir = 0.0;
        double sublimation = 0.0, deposition = 0.0, deposition_from_sublimation = 0.0;
        double closure_rel = 0.0;       // |gross - (redeposited + escaped + reservoir)| / gross
        double transport_rel = 0.0;     // |transport_source - redeposited| / transport_source
        double dep_excess_rel = 0.0;    // (deposition - sublimation - transport_source) / (sublimation + transport_source)
    };
    SputterAudit sputterFluxAudit(const std::vector<double>& face_areas) const {
        SputterAudit a;
        const auto& d = *diurnal_avg_;
        for (size_t i = 0; i < num_faces_; ++i) {
            a.gross += d.avg_sputtering_flux[i] * face_areas[i];
            a.redeposited += d.avg_deposition_sputter_flux[i] * face_areas[i];
            a.sublimation += d.avg_sublimation_flux[i] * face_areas[i];
            a.deposition += d.avg_deposition_flux[i] * face_areas[i];
            a.deposition_from_sublimation += d.avg_deposition_sublimation_flux[i] * face_areas[i];
        }
        if (d.perFaceSputter()) {
            for (size_t i = 0; i < num_faces_; ++i) {
                const double g = d.avg_sputtering_flux[i] * face_areas[i];
                a.transport_source += d.sputter_face_transport_frac[i] * g;
                a.escaped += d.sputter_face_escape_frac[i] * g;
                a.reservoir += d.sputter_face_reservoir_frac[i] * g;
            }
        } else {
            a.transport_source = d.sputter_transport_share * a.gross;
            a.escaped = d.sputter_escape_share * a.gross;
            a.reservoir = d.sputter_reservoir_share * a.gross;
        }
        if (a.gross > 0.0) {
            a.closure_rel = std::abs(a.gross - (a.redeposited + a.escaped + a.reservoir)) / a.gross;
        }
        if (a.transport_source > 0.0) {
            a.transport_rel = std::abs(a.transport_source - a.redeposited) / a.transport_source;
        }
        const double src = a.sublimation + a.transport_source;
        if (src > 0.0) a.dep_excess_rel = (a.deposition - src) / src;
        return a;
    }

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
