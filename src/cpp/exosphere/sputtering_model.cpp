/*
 * Europa Magnetospheric Plasma Sputtering Model - Implementation
 *
 * See sputtering_model.h for documentation and references.
 */

#include "sputtering_model.h"
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <stdexcept>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ptm {
namespace exosphere {

void SputteringModel::initializeSpatialDistribution(
    const std::vector<std::array<double, 3>>& face_centers,
    const std::vector<double>& face_latitudes,
    const std::vector<double>& face_longitudes,
    const std::vector<double>& face_areas)
{
    if (face_centers.size() != num_faces_) {
        std::cerr << "SputteringModel: face_centers size mismatch!" << std::endl;
        return;
    }
    const bool legacy = config_.legacyParams();
    if (legacy && hasIonMap()) {
        throw std::invalid_argument("sputtering: --sputter-ion-map / --sputter-ion-map-absolute need the "
                                    "current parameter set (not --sputter-legacy-params)");
    }
    if (!legacy && face_areas.size() != num_faces_) {
        throw std::invalid_argument("sputtering: the area-weighted ion normalization needs the face areas "
                                    "(initializeSpatialDistribution got " + std::to_string(face_areas.size()) +
                                    " for " + std::to_string(num_faces_) + " faces)");
    }
    if (hasO2Map() && legacy) {
        throw std::invalid_argument("sputtering: --sputter-o2-map-absolute needs the current parameter set");
    }
    if (hasO2Map() && o2_map_values_.size() != num_faces_) {
        throw std::invalid_argument("sputtering: O2 map " + o2_map_label_ + " has " +
                                    std::to_string(o2_map_values_.size()) + " values; the mesh has " +
                                    std::to_string(num_faces_) + " faces");
    }
    if (hasIonMap() && ion_map_values_.size() != num_faces_) {
        throw std::invalid_argument("sputtering: ion map " + ion_map_label_ + " has " +
                                    std::to_string(ion_map_values_.size()) + " values; the mesh has " +
                                    std::to_string(num_faces_) + " faces");
    }

    spatial_weight_.resize(num_faces_);

    // Compute latitude and longitude if not provided
    std::vector<double> latitudes = face_latitudes;
    std::vector<double> longitudes = face_longitudes;

    if (latitudes.empty() || longitudes.empty()) {
        latitudes.resize(num_faces_);
        longitudes.resize(num_faces_);

        #pragma omp parallel for
        for (size_t i = 0; i < num_faces_; ++i) {
            const auto& pos = face_centers[i];
            double r = std::sqrt(pos[0]*pos[0] + pos[1]*pos[1] + pos[2]*pos[2]);

            if (r > 0) {
                // Latitude: angle from equatorial plane
                latitudes[i] = std::asin(pos[2] / r) * 180.0 / M_PI;

                // Longitude: angle in x-y plane (standard convention)
                longitudes[i] = std::atan2(pos[1], pos[0]) * 180.0 / M_PI;
            } else {
                latitudes[i] = 0.0;
                longitudes[i] = 0.0;
            }
        }
    }

    if (!legacy) {
        // Current set (PTM-649): area-weighted normalization, exact total.
        std::vector<double> raw(num_faces_);
        if (hasIonMap()) {
            raw = ion_map_values_;
        } else {
            for (size_t i = 0; i < num_faces_; ++i) {
                raw[i] = computeLongitudeFluxFactor(toWestLongitude(longitudes[i])) *
                         computeLatitudeFluxFactor(latitudes[i]);
            }
        }
        double area = 0.0, weighted = 0.0;
        for (size_t i = 0; i < num_faces_; ++i) {
            if (!std::isfinite(raw[i]) || raw[i] < 0.0 || !(face_areas[i] > 0.0)) {
                throw std::invalid_argument("sputtering: ion weight/area at face " + std::to_string(i) +
                                            " is not finite and >= 0 (weight " + std::to_string(raw[i]) +
                                            ", area " + std::to_string(face_areas[i]) + ")");
            }
            area += face_areas[i];
            weighted += raw[i] * face_areas[i];
        }
        if (!(weighted > 0.0)) {
            throw std::invalid_argument("sputtering: the ion map integrates to zero");
        }
        mesh_area_ = area;
        ion_molecule_flux_.resize(num_faces_);
        if (hasIonMap() && ion_map_absolute_) {
            // molecules cm^-2 s^-1 -> m^-2 s^-1; the map sets the total.
            ion_total_rate_ = weighted * sputtering_constants::CM2_PER_M2;
            for (size_t i = 0; i < num_faces_; ++i) {
                ion_molecule_flux_[i] = raw[i] * sputtering_constants::CM2_PER_M2;
            }
        } else {
            ion_total_rate_ = sputtering_constants::GLOBAL_H2O_RATE;
            for (size_t i = 0; i < num_faces_; ++i) {
                ion_molecule_flux_[i] = ion_total_rate_ * raw[i] / weighted;
            }
        }
        if (hasO2Map()) {
            // PTM-650: radiolytic O2 (and stoichiometric H2) from the map, taken out
            // of the H2O-equivalent gross; it must fit inside the gross on every face.
            const double m_o2 = sputtering_constants::O2_MASS, m_h2o = sputtering_constants::H2O_MASS;
            const double rad_per_o2 = m_o2 + 2.0 * sputtering_constants::H2_MASS;  // kg per O2 produced
            o2_map_total_rate_ = 0.0;
            double worst = 0.0;
            size_t worst_face = 0;
            for (size_t i = 0; i < num_faces_; ++i) {
                const double v = o2_map_values_[i];
                if (!std::isfinite(v) || v < 0.0) {
                    throw std::invalid_argument("sputtering: O2 map value at face " + std::to_string(i) +
                                                " is not finite and >= 0");
                }
                o2_map_total_rate_ += v * sputtering_constants::CM2_PER_M2 * face_areas[i];
                const double gross = m_h2o * ion_molecule_flux_[i];
                const double rad = rad_per_o2 * v * sputtering_constants::CM2_PER_M2;
                const double r = (gross > 0.0) ? rad / gross : (rad > 0.0 ? INFINITY : 0.0);
                if (r > worst) { worst = r; worst_face = i; }
            }
            if (worst > 1.0) {
                throw std::invalid_argument("sputtering: O2 map " + o2_map_label_ + " puts more radiolytic mass than "
                                            "the H2O-equivalent gross at face " + std::to_string(worst_face) +
                                            " (ratio " + std::to_string(worst) + ")");
            }
            std::cout << "  Radiolytic part from " << o2_map_label_ << ": " << std::scientific << std::setprecision(6)
                      << o2_map_total_rate_ << " O2/s = " << o2_map_total_rate_ * m_o2
                      << " kg/s at full exposure (x rate_scale); H2 stoichiometric; tau(T) NOT applied "
                      << "(the map carries its own T dependence); max per-face (O2 + H2)/gross "
                      << std::fixed << std::setprecision(4) << worst << std::endl;
        }
        // Diagnostic weight: area-weighted mean 1.
        for (size_t i = 0; i < num_faces_; ++i) spatial_weight_[i] = raw[i] * area / weighted;
        reference_ion_flux_ = ion_total_rate_ / area *
                              *std::max_element(spatial_weight_.begin(), spatial_weight_.end());
        spatial_initialized_ = true;
        const double min_w = *std::min_element(spatial_weight_.begin(), spatial_weight_.end());
        const double max_w = *std::max_element(spatial_weight_.begin(), spatial_weight_.end());
        std::cout << "Sputtering ion flux (area-weighted normalization, PTM-649):" << std::endl;
        std::cout << "  Pattern: " << (hasIonMap() ? ion_map_label_ :
                     "cosine " + std::to_string(config_.hemisphere_contrast) +
                     ":1 trailing/leading (Cassidy et al. 2013 apex ratio ~3, p. 64/69; the cosine form "
                     "approximates their Fig. 7 map) x linear polar " + std::to_string(config_.polar_enhancement) +
                     (config_.polar_enhancement == 1.0 ? " (none)" : " (not from Cassidy)")) << std::endl;
        std::cout << "  Total at T_ref " << std::fixed << std::setprecision(1) << config_.radiolysis_reference_K
                  << " K on pure ice: " << std::scientific << std::setprecision(6) << ion_total_rate_
                  << " H2O-equivalent/s x rate_scale " << config_.rate_scale << " = "
                  << ion_total_rate_ * config_.rate_scale * sputtering_constants::H2O_MASS << " kg/s ("
                  << (hasIonMap() && ion_map_absolute_ ? "set by the absolute map" : "Cassidy et al. 2013 target")
                  << "; availability and tau(T) change the realized gross)" << std::endl;
        std::cout << "  Mesh area " << area << " m^2; weight range " << std::fixed << std::setprecision(3)
                  << min_w << " - " << max_w << std::endl;
        return;
    }

    // Compute spatial weights based on hemisphere and latitude
    double total_weight = 0.0;

    #pragma omp parallel for reduction(+:total_weight)
    for (size_t i = 0; i < num_faces_; ++i) {
        // Convert to west longitude for Europa convention
        double west_lon = toWestLongitude(longitudes[i]);

        // Compute flux factors
        double lon_factor = computeLongitudeFluxFactor(west_lon);
        double lat_factor = computeLatitudeFluxFactor(latitudes[i]);

        spatial_weight_[i] = lon_factor * lat_factor;
        total_weight += spatial_weight_[i];
    }

    // Normalize weights so they sum to num_faces_ (i.e., mean = 1.0)
    if (total_weight > 0) {
        double norm = num_faces_ / total_weight;
        #pragma omp parallel for
        for (size_t i = 0; i < num_faces_; ++i) {
            spatial_weight_[i] *= norm;
        }
    }

    // Compute reference ion flux at trailing apex
    // Global rate / surface area gives average flux
    // Trailing apex has maximum weight, so reference = average * max_weight / mean_weight
    double avg_flux = sputtering_constants::GLOBAL_H2O_RATE / sputtering_constants::EUROPA_SURFACE_AREA;

    // Convert to mass flux and find reference
    // The spatial_weight_ is already normalized so mean = 1.0
    double max_weight = *std::max_element(spatial_weight_.begin(), spatial_weight_.end());
    reference_ion_flux_ = avg_flux * max_weight;

    spatial_initialized_ = true;

    // Report initialization
    double min_weight = *std::min_element(spatial_weight_.begin(), spatial_weight_.end());
    std::cout << "Sputtering spatial distribution initialized:" << std::endl;
    std::cout << "  Weight range: " << std::fixed << std::setprecision(3)
              << min_weight << " - " << max_weight << std::endl;
    std::cout << "  Trailing/leading ratio: " << (max_weight / min_weight) << std::endl;
}

void SputteringModel::computeSputteringRates(
    const std::vector<double>& temperatures,
    const std::vector<double>& face_areas,
    SputteringState& state) const
{
    if (!config_.enabled) {
        state.reset();
        return;
    }

    if (!spatial_initialized_) {
        std::cerr << "SputteringModel: spatial distribution not initialized!" << std::endl;
        state.reset();
        return;
    }

    if (temperatures.size() != num_faces_ || face_areas.size() != num_faces_) {
        std::cerr << "SputteringModel: size mismatch in computeSputteringRates!" << std::endl;
        state.reset();
        return;
    }

    if (!config_.legacyParams()) {
        // Current set (PTM-649): H2O part temperature-independent, radiolytic part x tau(T).
        const double f = config_.h2o_fraction;
        const double m_o2 = sputtering_constants::O2_MASS, m_2h2 = 2.0 * sputtering_constants::H2_MASS;
        const double o2_of_rad = m_o2 / (m_o2 + m_2h2);
        const double e_h2o = config_.h2oEscapeFraction();
        const double e_o2 = config_.o2EscapeFraction(), e_h2 = config_.h2EscapeFraction();
        const double scale = sputtering_constants::H2O_MASS * config_.rate_scale;
        const bool o2_map = hasO2Map();
        const double o2_map_scale = sputtering_constants::O2_MASS * sputtering_constants::CM2_PER_M2 *
                                    config_.rate_scale;
        #pragma omp parallel for
        for (size_t i = 0; i < num_faces_; ++i) {
            const double base = scale * ion_molecule_flux_[i];            // kg m^-2 s^-1 at T_ref
            double h2o, rad, o2;
            if (o2_map) {
                o2 = o2_map_scale * o2_map_values_[i];
                rad = o2 / o2_of_rad;
                h2o = base - rad;
            } else {
                const double tau = config_.radiolyticFactor(temperatures[i]);
                h2o = base * f;
                rad = base * (1.0 - f) * tau;
                o2 = rad * o2_of_rad;
            }
            const double h2 = rad - o2;
            const double gross = h2o + rad;
            state.sputtering_yield[i] = (base > 0.0) ? gross / base : 0.0;  // gross / (gross at T_ref)
            state.h2o_sputter_flux[i] = h2o;
            state.o2_production_flux[i] = o2;
            state.h2_production_flux[i] = h2;
            state.gross_removal_flux[i] = gross;
            state.total_mass_loss[i] = h2o * e_h2o + o2 * e_o2 + h2 * e_h2;
            state.redeposition_flux[i] = h2o * (1.0 - e_h2o);
            state.reservoir_flux[i] = o2 * (1.0 - e_o2) + h2 * (1.0 - e_h2);
            state.ion_flux[i] = gross / sputtering_constants::H2O_MASS;  // H2O-equivalent molecules m^-2 s^-1
        }
        return;
    }

    // Global average sputtering flux (molecules/m²/s)
    double avg_molecule_flux = sputtering_constants::GLOBAL_H2O_RATE / sputtering_constants::EUROPA_SURFACE_AREA;

    // Apply rate scaling
    avg_molecule_flux *= config_.rate_scale;

    // Convert to mass flux (kg/m²/s)
    double avg_mass_flux = avg_molecule_flux * sputtering_constants::H2O_MASS;
    const double o2_fraction = config_.o2Fraction();
    const double h2_fraction = config_.h2Fraction();

    #pragma omp parallel for
    for (size_t i = 0; i < num_faces_; ++i) {
        double T = temperatures[i];

        // Check minimum temperature
        if (T < config_.min_sputtering_temp) {
            state.ion_flux[i] = 0.0;
            state.sputtering_yield[i] = 0.0;
            state.h2o_sputter_flux[i] = 0.0;
            state.o2_production_flux[i] = 0.0;
            state.h2_production_flux[i] = 0.0;
            state.total_mass_loss[i] = 0.0;
            state.redeposition_flux[i] = 0.0;
            state.gross_removal_flux[i] = 0.0;
            state.reservoir_flux[i] = 0.0;
            continue;
        }

        // Spatial weighting (already normalized to mean=1)
        double spatial_factor = spatial_weight_[i];

        // Temperature-dependent yield enhancement
        double temp_factor = computeTemperatureYieldFactor(T);

        // Store yield factor for diagnostics
        state.sputtering_yield[i] = temp_factor;

        // Total sputtering mass flux for this face
        double total_sputter_flux = avg_mass_flux * spatial_factor * temp_factor;

        // Partition into products (mass fractions sum to 1; O2/H2 stoichiometric
        // unless legacy_escape_only)
        double h2o_flux = total_sputter_flux * config_.h2o_fraction;
        double o2_flux = total_sputter_flux * o2_fraction;
        double h2_flux = total_sputter_flux * h2_fraction;

        // Store species fluxes
        state.h2o_sputter_flux[i] = h2o_flux;
        state.o2_production_flux[i] = o2_flux;
        state.h2_production_flux[i] = h2_flux;
        state.gross_removal_flux[i] = total_sputter_flux;

        // Compute mass loss (what escapes to space)
        double h2o_escape = h2o_flux * config_.h2o_escape_fraction;
        double o2_escape = o2_flux * config_.o2_escape_fraction;
        double h2_escape = h2_flux * config_.h2_escape_fraction;

        // Net mass loss to space
        state.total_mass_loss[i] = h2o_escape + o2_escape + h2_escape;

        // Non-escaping H2O. New accounting: the multiscale path launches it into
        // the ballistic transport kernel (runMultiTimescaleSimulation). Legacy:
        // diagnostic only; it was never removed or transported (the old comment
        // here claimed the transport matrix handled it; it did not; PTM-642).
        state.redeposition_flux[i] = h2o_flux * (1.0 - config_.h2o_escape_fraction);

        // Non-escaping O2 + H2: non-condensable, an atmospheric reservoir.
        state.reservoir_flux[i] = o2_flux * (1.0 - config_.o2_escape_fraction)
                                + h2_flux * (1.0 - config_.h2_escape_fraction);

        // Ion flux for diagnostics (convert back to molecules/m²/s)
        state.ion_flux[i] = (total_sputter_flux / sputtering_constants::H2O_MASS) * spatial_factor;
    }
}

double SputteringModel::applySputteringMassLoss(
    const SputteringState& state,
    std::vector<double>& accumulated_mass,
    const std::vector<double>& face_areas,
    double dt)
{
    if (!config_.enabled) {
        return 0.0;
    }

    if (accumulated_mass.size() != num_faces_ || face_areas.size() != num_faces_) {
        std::cerr << "SputteringModel: size mismatch in applySputteringMassLoss!" << std::endl;
        return 0.0;
    }

    double total_sputtered = 0.0;
    double total_escaped = 0.0;
    double total_redeposited = 0.0;

    #pragma omp parallel for reduction(+:total_sputtered,total_escaped,total_redeposited)
    for (size_t i = 0; i < num_faces_; ++i) {
        // Total mass sputtered from this face this timestep
        double sputtered_mass = state.h2o_sputter_flux[i] * dt;

        // Can't sputter more than available ice
        // accumulated_mass is in kg/m², multiply by area for total kg
        double available_mass = accumulated_mass[i];

        if (sputtered_mass > available_mass) {
            sputtered_mass = available_mass;
        }

        // Apply mass loss (sputtering removes ice from surface)
        accumulated_mass[i] -= sputtered_mass;

        // Ensure non-negative
        if (accumulated_mass[i] < 0.0) {
            accumulated_mass[i] = 0.0;
        }

        // Track statistics
        total_sputtered += sputtered_mass * face_areas[i];

        // Escaped mass (fraction that doesn't return)
        double escaped = sputtered_mass * config_.h2oEscapeFraction();
        total_escaped += escaped * face_areas[i];

        // Redeposited mass (H2O that returns via ballistic transport)
        // Note: This is handled by the transport matrix, so we just track it
        double redeposited = sputtered_mass * (1.0 - config_.h2oEscapeFraction());
        total_redeposited += redeposited * face_areas[i];
    }

    // Also add O2 and H2 escape to total escaped
    #pragma omp parallel for reduction(+:total_escaped)
    for (size_t i = 0; i < num_faces_; ++i) {
        total_escaped += state.o2_production_flux[i] * config_.o2EscapeFraction() * dt * face_areas[i];
        total_escaped += state.h2_production_flux[i] * config_.h2EscapeFraction() * dt * face_areas[i];
    }

    // Update cumulative statistics
    total_sputtered_mass_ += total_sputtered;
    total_escaped_mass_ += total_escaped;
    total_redeposited_mass_ += total_redeposited;

    return total_escaped;
}

void SputteringModel::printConfig() const {
    std::cout << "\n=== Sputtering Model Configuration ===" << std::endl;
    std::cout << "  Enabled: " << (config_.enabled ? "yes" : "no") << std::endl;

    if (!config_.enabled) {
        std::cout << "  (Sputtering disabled - no further parameters apply)" << std::endl;
        return;
    }
    const bool legacy = config_.legacyParams();
    std::cout << "  Parameter set: " << (legacy ? "LEGACY (--sputter-legacy-params; pre-2026-09-23)"
                                                : "current (PTM-649; citations in sputtering_model.h)") << std::endl;
    std::cout << "  Rate scale factor: " << config_.rate_scale << std::endl;
    if (!legacy && hasIonMap()) {
        std::cout << "  Ion map: " << ion_map_label_ << (ion_map_absolute_ ? " (absolute)" : " (pattern)") << std::endl;
    } else {
        std::cout << "  Hemisphere contrast: " << config_.hemisphere_contrast
                  << " (trailing/leading ratio)" << std::endl;
        std::cout << "  Polar enhancement: " << config_.polar_enhancement << std::endl;
    }
    std::cout << "  Temperature dependent: " << (config_.temperature_dependent ? "yes" : "no") << std::endl;
    if (legacy) {
        std::cout << "  Min sputtering temp: " << config_.min_sputtering_temp << " K" << std::endl;
    } else {
        if (hasO2Map()) {
            std::cout << "  Radiolytic part: " << o2_map_label_ << " (no tau(T); the map carries its own)" << std::endl;
        } else {
            std::cout << "  tau(T) on the radiolytic part only: [1 + q0 e^(-E_a/kT)] / [1 + q0 e^(-E_a/kT_ref)], q0 "
                      << config_.radiolysis_q0 << ", E_a " << config_.radiolysis_activation_eV << " eV, T_ref "
                      << config_.radiolysis_reference_K << " K (Cassidy et al. 2013 p. 71; no cutoff)" << std::endl;
        }
        std::cout << "  Binding energy U: " << config_.binding_energy_eV << " eV -> H2O escaping fraction "
                  << std::setprecision(4) << config_.h2oEscapeFraction() << " and launch temperature "
                  << std::setprecision(1) << config_.launchTemperatureK() << " K (Sigmund-Thompson, E_esc "
                  << std::setprecision(4) << h2oEscapeEnergyEV() << " eV)" << std::endl;
    }

    std::cout << "\n  Species fractions" << (legacy ? "" : " (at T_ref)") << ":" << std::endl;
    std::cout << "    H2O: " << (config_.h2o_fraction * 100) << "%" << std::endl;
    std::cout << "    O2:  " << (config_.o2Fraction() * 100) << "%" << std::endl;
    std::cout << "    H2:  " << (config_.h2Fraction() * 100) << "%" << std::endl;
    std::cout << "    O2:H2 mass ratio: " << (config_.o2Fraction() / config_.h2Fraction())
              << (config_.legacy_escape_only ? " (legacy, non-stoichiometric)"
                                              : " (radiolytic stoichiometry 2 H2O -> 2 H2 + O2)")
              << std::endl;

    std::cout << "\n  Escape fractions:" << std::endl;
    std::cout << "    H2O: " << (config_.h2oEscapeFraction() * 100) << "%" << std::endl;
    if (legacy) {
        std::cout << "    O2:  " << (config_.o2EscapeFraction() * 100) << "%" << std::endl;
    } else {
        std::cout << "    O2:  0% escape; 100% atmospheric loss (bound; ionization/dissociation/pickup)" << std::endl;
    }
    std::cout << "    H2:  " << (config_.h2EscapeFraction() * 100) << "%" << std::endl;

    // Compute expected global rates
    const double rate = legacy ? sputtering_constants::GLOBAL_H2O_RATE : ion_total_rate_;
    double h2o_total = rate * sputtering_constants::H2O_MASS * config_.rate_scale;
    std::cout << "\n  Expected global rates (with current settings" << (legacy ? "" : ", pure ice at T_ref") << "):"
              << std::endl;
    std::cout << "    Total sputtered (H2O-equivalent): " << std::scientific << std::setprecision(2)
              << (rate * config_.rate_scale) << " molecules/s" << std::endl;
    std::cout << "    Gross mass rate: " << h2o_total << " kg/s" << std::endl;
    std::cout << "    H2O escape rate: " << h2o_total * config_.h2o_fraction * config_.h2oEscapeFraction()
              << " kg/s" << std::endl;
    if (!legacy) {
        const double o2_rate = hasO2Map() ? o2_map_total_rate_ * sputtering_constants::O2_MASS * config_.rate_scale
                                          : h2o_total * config_.o2Fraction();
        std::cout << "    O2 production: " << o2_rate << " kg/s (JADE: 12 +/- 6 kg/s, Szalay et al. 2024)"
                  << std::endl;
    }
    std::cout << std::fixed << std::setprecision(4);
    if (config_.legacy_escape_only) {
        std::cout << "\n  Mass accounting: LEGACY escape-only (--sputter-legacy-escape-only). Only the "
                  << (100.0 * config_.escapeShare()) << "% escaping share leaves the surface; the other "
                  << (100.0 * (1.0 - config_.escapeShare())) << "% is neither removed nor transported "
                  << "(NOT mass-conserving; PTM-642)." << std::endl;
    } else {
        std::cout << "\n  Mass accounting (share of gross sputtered mass" << (legacy ? "" : ", at T_ref") << "):"
                  << std::endl;
        std::cout << "    Redeposited via ballistic transport (non-escaping H2O): "
                  << (100.0 * config_.transportShare()) << "%, launch temperature "
                  << std::setprecision(1) << config_.launchTemperatureK() << " K" << std::endl;
        std::cout << std::setprecision(4);
        std::cout << "    Escaped (all products):                  " << (100.0 * config_.escapeShare()) << "%" << std::endl;
        std::cout << (legacy ? "    Atmospheric reservoir (non-escaping O2 + H2, non-condensable): "
                             : "    Atmospheric loss (O2; ionization/dissociation/pickup): ")
                  << (100.0 * config_.reservoirShare()) << "%" << std::endl;
    }
    std::cout << std::fixed;
}

void SputteringConfig::validate() const {
    auto in01 = [](double v, const char* name) {
        if (!(v >= 0.0 && v <= 1.0)) {
            throw std::invalid_argument(std::string("sputtering: ") + name + " = " +
                                        std::to_string(v) + " is outside [0, 1]");
        }
    };
    in01(h2o_fraction, "h2o_fraction (--sputter-h2o-fraction)");
    in01(h2o_escape_fraction, "h2o_escape_fraction (--sputter-escape-h2o)");
    in01(o2_escape_fraction, "o2_escape_fraction (--sputter-escape-o2)");
    in01(h2_escape_fraction, "h2_escape_fraction (--sputter-escape-h2)");
    if (!(launch_temperature_K > 0.0) || !std::isfinite(launch_temperature_K)) {
        throw std::invalid_argument("sputtering: launch_temperature_K (--sputter-launch-temperature) = " +
                                    std::to_string(launch_temperature_K) + " must be a positive temperature");
    }
    if (legacy_escape_only && launch_temperature_given) {
        throw std::invalid_argument("sputtering: --sputter-launch-temperature has no effect with "
                                    "--sputter-legacy-escape-only (nothing is transported)");
    }
    auto positive = [](double v, const char* name) {
        if (!(v > 0.0) || !std::isfinite(v)) {
            throw std::invalid_argument(std::string("sputtering: ") + name + " = " + std::to_string(v) +
                                        " must be positive and finite");
        }
    };
    positive(binding_energy_eV, "binding_energy_eV (--sputter-binding-energy)");
    positive(radiolysis_reference_K, "radiolysis_reference_K (--sputter-radiolysis-tref)");
    if (!(radiolysis_activation_eV >= 0.0) || !std::isfinite(radiolysis_activation_eV)) {
        throw std::invalid_argument("sputtering: radiolysis_activation_eV (--sputter-radiolysis-ea) = " +
                                    std::to_string(radiolysis_activation_eV) + " must be >= 0");
    }
    positive(polar_enhancement, "polar_enhancement (--sputter-polar-enhancement)");
    if (!(radiolysis_q0 >= 0.0) || !std::isfinite(radiolysis_q0)) {
        throw std::invalid_argument("sputtering: radiolysis_q0 (--sputter-radiolysis-q0) = " +
                                    std::to_string(radiolysis_q0) + " must be >= 0");
    }
    if (!(hemisphere_contrast >= 1.0)) {
        throw std::invalid_argument("sputtering: hemisphere_contrast (--sputtering-contrast) = " +
                                    std::to_string(hemisphere_contrast) + " must be >= 1");
    }
    if (legacy_escape_only &&
        std::abs(h2o_fraction + legacy_o2_fraction + legacy_h2_fraction - 1.0) > 1e-12) {
        throw std::invalid_argument("sputtering: legacy_escape_only uses the fixed 0.10 O2 / 0.05 H2 "
                                    "split, so h2o_fraction must stay 0.85");
    }
}

} // namespace exosphere
} // namespace ptm
