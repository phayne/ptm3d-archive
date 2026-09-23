/*
 * Europa Magnetospheric Plasma Sputtering Model
 *
 * Gross sputtering (ice removed) per face, in H2O-equivalent mass:
 *
 *   gross_i(T) = m_H2O * Phi_i * [ f + (1 - f) * tau(T_i) ]
 *
 *   Phi_i   ion-driven H2O-equivalent molecule flux at T_ref. Default:
 *           N_tot * w_i / A, with the geometric (ion-access) weight w_i renormalized
 *           so that sum_i w_i a_i = A (area-weighted mean 1, A = the mesh's total
 *           area). The target N_tot is delivered exactly by a pure-ice surface at
 *           T_ref; ice availability and tau(T) then reduce or raise the realized
 *           gross. We renormalize the geometric weight, not the realized gross: the
 *           magnetosphere sets the ion flux, the surface sets the yield.
 *           Alternatives: --sputter-ion-map FILE (weights, renormalized the same
 *           way) or --sputter-ion-map-absolute FILE (Phi_i itself, cm^-2 s^-1).
 *   f       H2O mass fraction of the gross at T_ref (--sputter-h2o-fraction).
 *   tau(T)  radiolytic (O2 + H2) yield relative to T_ref, on the radiolytic part
 *           ONLY (Cassidy et al. 2013 p. 71: the H2O rate is nearly independent of T
 *           at 80-130 K; the T dependence arises mostly from O2 and H2):
 *             tau(T) = [1 + q0 exp(-E_a/kT)] / [1 + q0 exp(-E_a/kT_ref)],
 *           the O2 yield law of Cassidy et al. 2013 p. 71 (q0 = 1000, E_a = 0.06 eV,
 *           after Teolis et al. 2010). Continuous and monotonic, no cutoff (the
 *           legacy factor was 0 below 50 K, 1 from 50-80 K, 0.599 at 80 K, and applied
 *           to the whole yield; PTM-649).
 *   --sputter-o2-map-absolute FILE: the radiolytic part comes from a per-face O2
 *           production map instead (O2 molecules cm^-2 s^-1; H2 stoichiometric;
 *           H2O = gross - O2 - H2). tau(T) is NOT applied to it: the Addison et al.
 *           2022 O2 map already carries the same law at a diurnal-mean surface
 *           temperature profile (GENTOo O2 source: parameters.cpp:201-202 o2_q0 =
 *           1.0e3, o2_Q = 0.06 eV; defines.h:55 TEMPERATUREPROFILE).
 *
 * Products and fates. Radiolytic O2 : H2 is stoichiometric (2 H2O -> 2 H2 + O2;
 * Cassidy p. 71: "the H2 production rate is exactly twice that of the O2").
 * H2O leaves with the Sigmund distribution f(E) ~ E / (E + U)^3 (Plainaki et al.
 * 2010 Eq. 2 in its T_m -> infinity limit), which fixes BOTH the escaping fraction
 * and the launch temperature of the redepositing part, from the effective surface
 * binding energy U (--sputter-binding-energy):
 *   e_H2O  = 1 - [E_esc / (E_esc + U)]^2                (fraction above E_esc)
 *   T_launch = <E | E < E_esc> / (2 k)                   (kernel mean energy 2kT)
 * with E_esc = m_H2O v_esc^2 / 2 = 0.3828 eV (v_esc = 2025 m/s; Plainaki: 0.38 eV).
 * U = 0.054 eV gives e_H2O = 0.2320 and T_launch = 677 K. H2 escapes (100%); O2 is
 * gravitationally bound (Cassidy p. 71) but non-condensable and is lost from the
 * ice budget by ionization, dissociation and pickup (100% "atmospheric loss",
 * not escape).
 *
 * Citations, checked 2026-09-23 against the PDFs in Paul's library
 * (VERIFIED = value found at the locator; DIFFERS = the source gives another value):
 *   N_tot = 2e27 H2O/s    Cassidy et al. 2013, PSS 77, 64: abstract (p. 64),     VERIFIED
 *                         Table 2 "Total" (p. 70); H2O-equivalent incl. O2/H2 (p. 66)
 *   f = 0.90              Cassidy p. 71: O2 + H2 "~10% by mass"; Table 2 O2       VERIFIED
 *                         1e26/s (p. 70), H2 2e26/s (p. 71)
 *   Pattern: 3:1          Cassidy abstract (p. 64) "decreases to about 1/3rd of   VERIFIED (amplitude)
 *     trailing/leading    the peak value at the leading hemisphere apex"; p. 69
 *                         "about a factor of 3 difference" between the apexes
 *   Pattern: cosine form  Cassidy gives a computed map (Fig. 7, p. 69), no         DIFFERS (approximation)
 *                         analytic form; cos(theta) from the trailing apex is the
 *                         cold-ion flux only (p. 70). The map also rises with
 *                         latitude on the leading hemisphere (Fig. 7 caption),
 *                         which the cosine omits (PTM-652).
 *   Polar 1.2 (linear)    NOT in Cassidy, and no such number in Nordheim 2022      NOT-FOUND; default 1.0
 *                         (below). Default is now 1 (BASE = Cassidy).              (legacy set keeps 1.2)
 *   tau(T): q0 = 1000,    Cassidy p. 71 O2-yield equation (after Teolis et al.     VERIFIED (Teolis 2010
 *     E_a = 0.06 eV       2010); total-yield law q0 = 220, E_a = 0.06 eV: Cassidy   not checked)
 *                         Eq. 2 (p. 66) = Fama et al. 2008 Sec. 3.1 (p. 158)
 *   T_ref = 100 K         Addison et al. 2022 fixed surface T (GENTOo source)      VERIFIED (as used)
 *   U = 0.054 eV          Plainaki et al. 2010, Icarus 210, 385: Table 1 and       VERIFIED (0.055 NOT
 *                         Sec. 2.2.1.3 (p. 388), after Boring et al. 1984 and      FOUND: not in Johnson
 *                         Haring et al. 1984                                       & Sittler 1990 GRL)
 *   e_H2O                 Plainaki p. 388 and abstract: ~22% for 0.054 eV (their   DIFFERS slightly (0.2320
 *                         Eq. 2 with the T_m cutoff, 100 keV S+); Cassidy p. 69:   here, T_m -> inf);
 *                         "~1/3 of sputtered H2O" escapes                          Cassidy DIFFERS (1/3)
 *   f(E) ~ E/(E+U)^3      Plainaki Eq. 2 (p. 388), after Sigmund 1969              VERIFIED (limit form)
 *   O2 bound, H2 escapes  Cassidy p. 71 (after Smyth & Marconi 2006)              VERIFIED
 *   O2 loss channels      Teolis et al. 2017, Icarus 284, 18, Sec. 2 (pp. 21-22):   VERIFIED
 *                         ionization/dissociation/charge exchange rates
 *   JADE O2 12 +/- 6 kg/s Szalay et al. 2024, Nat. Astron.: abstract and Table 1    VERIFIED
 *                         (2.2 +/- 1.2e26 /s, inferred from H2 loss x 2:1)
 *   Addison 2022 maps     Zenodo 10.5281/zenodo.5708206 (GENTOo)               data VERIFIED (md5)
 *   Addison et al. 2022, JGR Space Phys. 127, e2021JA030136:
 *     units R [cm^-2 s^-1]   pp. 21, 30 (H2O-equivalent and O2 sputtering rates)    VERIFIED
 *     H2O total, draped      p. 25: P_T ~ 5.9e27 /s (uniform fields 1.16e28, p. 24)  DIFFERS: the shipped
 *                            p. 12: yields x 0.3 for regolith sticking; the GENTOo   map is 1.98e28 = 5.9e27
 *                            source has no such factor                               / 0.3 (PTM-654)
 *     H2O yield T-indep.     p. 11 (below 140 K, after Baragiola 2003); the GENTOo   DIFFERS (source keeps
 *                            H2O source still applies 1 + 220 e^(-0.06/kT) at 100 K  1.21 at <= 100 keV)
 *     O2 total, thin layer   p. 32: 1.5e26 - 4.9e26 /s over five T profiles; map     VERIFIED (in range)
 *                            (diurnal-mean profile) 2.65e26 /s = 14.1 kg/s
 *     O2 yield T law         Eqs. 3 and 5 (p. 14): (1 + q0 e^(-E_A/kT)), q0 = 1000,   VERIFIED
 *                            E_A = 0.06 eV (after Johnson et al. 2009)
 *     O2 delayed T response  p. 6 (after Teolis et al. 2005): diurnal-mean T         VERIFIED
 *     energy quadrature      discrete energies (p. 10); the integration rule is      NOT-FOUND
 *                            not stated
 *   Nordheim et al. 2022, PSJ 3:5: high-latitude and polar regions receive the   1.2 NOT-FOUND (ion
 *                            largest ion flux (abstract p. 1; p. 14); no linear     access maps only; no
 *                            polar sputtering factor is given                       sputtering factor)
 *
 * --sputter-legacy-params restores the pre-2026-09-23 parameter set (f 0.85,
 * escape 0.10 / 0.90 / 0.95, launch 683 K, the stepped tau on the whole yield with
 * a 50 K cutoff, polar 1.2, count-normalized weights over a fixed 3.09e13 m^2).
 * --sputter-legacy-escape-only implies it.
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

    // Temperature below which sputtering yield is constant (legacy parameters only)
    constexpr double LOW_TEMP_THRESHOLD = 80.0;  // K

    // Current parameter set (see the header comment for citations and status).
    constexpr double EUROPA_ESCAPE_VELOCITY = 2025.0;          // m/s (= constants::EUROPA_ESCAPE_VELOCITY)
    constexpr double JOULE_PER_EV = 1.602176634e-19;
    constexpr double BOLTZMANN_J = 1.380649e-23;               // J/K
    constexpr double DEFAULT_BINDING_ENERGY_EV = 0.054;        // U (Plainaki 2010 Table 1)
    constexpr double DEFAULT_H2O_FRACTION = 0.90;              // f at T_ref (Cassidy split)
    constexpr double RADIOLYSIS_ACTIVATION_EV = 0.06;          // E_a of tau(T) (Cassidy 2013 p. 71)
    constexpr double RADIOLYSIS_Q0 = 1000.0;                   // q0 of tau(T) (Cassidy 2013 p. 71)
    constexpr double RADIOLYSIS_REFERENCE_T = 100.0;           // K, T_ref of tau(T) and of the maps
    constexpr double CM2_PER_M2 = 1.0e4;
}

/**
 * Sigmund-Thompson H2O ejecta, f(E) ~ E / (E + U)^3 (PTM-649).
 * escapeFraction: share with E >= E_esc, 1 - [E_esc / (E_esc + U)]^2.
 * meanBoundEnergy: mean E over [0, E_esc) [same unit as U, E_esc].
 */
inline double sigmundThompsonEscapeFraction(double U, double E_esc) {
    const double x = E_esc / (E_esc + U);
    return 1.0 - x * x;
}
inline double sigmundThompsonMeanBoundEnergy(double U, double E_esc) {
    // int_0^Ee E^2/(E+U)^3 dE / int_0^Ee E/(E+U)^3 dE, with y = E + U.
    const double y1 = E_esc + U;
    const double num = std::log(y1 / U) + 2.0 * U / y1 - 2.0 - U * U / (2.0 * y1 * y1) + 0.5;
    const double den = (1.0 - sigmundThompsonEscapeFraction(U, E_esc)) / (2.0 * U);
    return num / den;
}
inline double h2oEscapeEnergyEV() {
    return 0.5 * sputtering_constants::H2O_MASS * sputtering_constants::EUROPA_ESCAPE_VELOCITY *
           sputtering_constants::EUROPA_ESCAPE_VELOCITY / sputtering_constants::JOULE_PER_EV;
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

    // Latitude dependence - linear polar enhancement factor. NOT in Cassidy et al.
    // 2013 (checked 2026-09-23); appendix D describes it as a simplification of
    // Nordheim et al. 2022. Default 1 (none: BASE uses Cassidy's pattern); the
    // legacy parameter set restores 1.2. --sputter-polar-enhancement.
    double polar_enhancement = 1.0;

    // Temperature dependence enable (Teolis et al. 2017)
    bool temperature_dependent = true;

    // Mass accounting (PTM-642). The gross sputtered mass (Cassidy et al. 2013,
    // GLOBAL_H2O_RATE x H2O_MASS ~ 60 kg/s H2O-equivalent at rate_scale 1) is ice
    // REMOVED from the face. It leaves as three products:
    //   H2O (fraction h2o_fraction), and radiolytic O2 + H2 (the remainder),
    // and each product has an escaping share. The non-escaping H2O is launched on
    // ballistic hops and redeposits through the transport kernel (multiscale path).
    // The non-escaping O2 and H2 are non-condensable at Europa surface temperatures
    // (O2 sublimation is ~30 K, H2 far lower), so they never return as ice: they
    // are booked as an atmospheric reservoir, which in steady state is lost by
    // ionization/dissociation/pickup at the rate it is supplied. Surface ice loss
    // from sputtering = escaped + reservoir = gross x (1 - h2o_fraction x (1 - h2o_escape)).
    //
    // legacy_escape_only = true restores the pre-PTM-642 accounting: only the
    // escaping share is removed from the surface, the non-escaping ~78% is neither
    // removed nor transported (NOT mass-conserving), the species split is the
    // non-stoichiometric 0.85/0.10/0.05, and the multiscale deposition rescale uses
    // the (sublimation + sputtering) ratio. Kept only to reproduce old products.
    bool legacy_escape_only = false;

    // Escaping share of each product. PROVENANCE: legacy defaults of this model,
    // attributed in the original code to Cassidy et al. (2013); not traced to a
    // specific table there. They are qualitatively consistent with Smyth & Marconi
    // (2006, Icarus 181) and Johnson et al. (2009, in "Europa", Univ. Arizona
    // Press): sputtered H2O mostly returns to the surface, H2 escapes readily, and
    // O2 is lost from the atmosphere rather than condensing. They are inputs, not
    // derived here; a Sigmund-Thompson energy distribution with U = 0.055 eV would
    // put ~24% of sputtered H2O above the escape energy, not 10%. Override with
    // --sputter-escape-h2o / -o2 / -h2.
    double h2o_escape_fraction = 0.1;   // 10% escapes
    double o2_escape_fraction = 0.9;    // 90% escapes
    double h2_escape_fraction = 0.95;   // 95% escapes

    // Mass fraction of the gross sputtered mass that leaves as H2O molecules
    // (--sputter-h2o-fraction), at T_ref in the current parameter set (away from
    // T_ref the radiolytic part follows tau(T)). The remainder is radiolytic O2 + H2
    // in the stoichiometric mass ratio of 2 H2O -> 2 H2 + O2, i.e.
    // m_O2 : 2 m_H2 = 7.94 : 1 with this file's molecular masses (8:1 nominal).
    // 0.90 = Cassidy et al. 2013 (UNVERIFIED); the legacy set uses 0.85.
    double h2o_fraction = sputtering_constants::DEFAULT_H2O_FRACTION;

    // Parameter set (PTM-649). false = the current set described in the header;
    // true = the pre-2026-09-23 set (--sputter-legacy-params; implied by
    // legacy_escape_only). applyLegacyParams() sets the legacy values.
    bool legacy_params = false;
    bool legacyParams() const { return legacy_params || legacy_escape_only; }

    // Surface binding energy U [eV] of the Sigmund-Thompson distribution; sets the
    // H2O escaping fraction AND the launch temperature (current set only).
    double binding_energy_eV = sputtering_constants::DEFAULT_BINDING_ENERGY_EV;
    // tau(T) = [1 + q0 e^(-E_a/kT)] / [1 + q0 e^(-E_a/kT_ref)] on the radiolytic part (current set).
    double radiolysis_activation_eV = sputtering_constants::RADIOLYSIS_ACTIVATION_EV;
    double radiolysis_reference_K = sputtering_constants::RADIOLYSIS_REFERENCE_T;
    double radiolysis_q0 = sputtering_constants::RADIOLYSIS_Q0;

    // Derived (current set) or stored (legacy set) fate parameters. Use these, not
    // the raw fields, which hold the legacy values.
    double h2oEscapeFraction() const {
        return legacyParams() ? h2o_escape_fraction
                              : sigmundThompsonEscapeFraction(binding_energy_eV, h2oEscapeEnergyEV());
    }
    double o2EscapeFraction() const { return legacyParams() ? o2_escape_fraction : 0.0; }  // O2: atmospheric loss
    double h2EscapeFraction() const { return legacyParams() ? h2_escape_fraction : 1.0; }  // H2 escapes
    double launchTemperatureK() const {
        if (legacyParams()) return launch_temperature_K;
        return sigmundThompsonMeanBoundEnergy(binding_energy_eV, h2oEscapeEnergyEV()) *
               sputtering_constants::JOULE_PER_EV / (2.0 * sputtering_constants::BOLTZMANN_J);
    }
    // Radiolytic yield relative to T_ref (current set). Continuous, monotonic, > 0.
    double radiolyticFactor(double T) const {
        if (!temperature_dependent) return 1.0;
        const double k_ev = sputtering_constants::BOLTZMANN_EV;
        if (!(T > 0.0)) return 1.0 / (1.0 + radiolysis_q0 * std::exp(-radiolysis_activation_eV /
                                                                    (k_ev * radiolysis_reference_K)));
        return (1.0 + radiolysis_q0 * std::exp(-radiolysis_activation_eV / (k_ev * T))) /
               (1.0 + radiolysis_q0 * std::exp(-radiolysis_activation_eV / (k_ev * radiolysis_reference_K)));
    }

    // Restore the pre-2026-09-23 parameter values (callers set legacy_params too).
    void applyLegacyParams() {
        h2o_fraction = 0.85;
        h2o_escape_fraction = 0.1;
        o2_escape_fraction = 0.9;
        h2_escape_fraction = 0.95;
        launch_temperature_K = 683.0;
        min_sputtering_temp = 50.0;
        polar_enhancement = 1.2;
    }

    // Legacy (non-stoichiometric) O2 / H2 mass fractions: O2:H2 = 2:1 by mass
    // against 7.94:1 from radiolysis. Used ONLY when legacy_escape_only is set.
    double legacy_o2_fraction = 0.10;
    double legacy_h2_fraction = 0.05;

    // Launch temperature [K] the ballistic kernel uses for the non-escaping
    // sputtered H2O (--sputter-launch-temperature). The kernel is the thermal
    // (Maxwellian-flux) hop kernel, whose mean launch energy is 2kT. Sputtered H2O
    // follows a Sigmund-Thompson distribution f(E) ~ E / (E + U)^3 with
    // U = 0.055 eV (the value commonly used for H2O ice, e.g. Plainaki et al. 2012,
    // after Johnson 1990). Truncated at the H2O escape energy
    // (0.5 m v_esc^2 = 0.383 eV) its mean energy is 0.118 eV, so T = 0.118 eV / 2k
    // = 683 K. At 683 K the hop scale 2kT/(mg) is ~480 km (0.31 R_Europa), where
    // the kernel's flat, uniform-gravity form underestimates hop range; the
    // redistribution PATTERN is therefore approximate. The global mass accounting
    // does not depend on this value.
    double launch_temperature_K = 683.0;
    bool launch_temperature_given = false;  // set by the CLI; rejected with legacy (unused there)

    // Optional per-face ion map (current set; read by PTMExosphereIntegration::
    // enableSputtering): uint64 count + float64 per face, face order.
    // ion_map_absolute = false: weights (--sputter-ion-map, renormalized to an
    // area-weighted mean of 1 at the N_tot target); true: H2O-equivalent molecules
    // cm^-2 s^-1 at T_ref (--sputter-ion-map-absolute), which set the total too.
    std::string ion_map_path;
    bool ion_map_absolute = false;
    // Optional per-face radiolytic O2 production map (current set;
    // --sputter-o2-map-absolute): O2 molecules cm^-2 s^-1, same file format. When
    // given, O2 comes from the map (no tau(T)), H2 is stoichiometric to it, and
    // H2O = gross - O2 - H2.
    std::string o2_map_path;

    // Minimum temperature for sputtering (legacy parameter set only; the current
    // set has no cutoff: sputtering is not thermally activated)
    double min_sputtering_temp = 50.0;  // K

    // Product mass fractions of the gross sputtered mass (sum to 1).
    double o2Fraction() const {
        if (legacy_escape_only) return legacy_o2_fraction;
        const double m_o2 = sputtering_constants::O2_MASS;
        const double m_2h2 = 2.0 * sputtering_constants::H2_MASS;
        return (1.0 - h2o_fraction) * m_o2 / (m_o2 + m_2h2);
    }
    double h2Fraction() const {
        if (legacy_escape_only) return legacy_h2_fraction;
        const double m_o2 = sputtering_constants::O2_MASS;
        const double m_2h2 = 2.0 * sputtering_constants::H2_MASS;
        return (1.0 - h2o_fraction) * m_2h2 / (m_o2 + m_2h2);
    }

    // Shares of the gross sputtered mass by fate (new accounting; sum to 1).
    // transport: non-escaping H2O, redeposited through the ballistic kernel.
    // escape:    escaping share of every product, lost to space.
    // reservoir: non-escaping O2 + H2 (non-condensable), never returned as ice.
    // In the current set these are the shares at T_ref; per face the composition
    // follows tau(T), and the multiscale path uses per-face shares (PTM-649).
    // "reservoir" is the atmospheric loss of the non-condensable products (current
    // set: all O2, lost by ionization/dissociation/pickup).
    double transportShare() const { return h2o_fraction * (1.0 - h2oEscapeFraction()); }
    double escapeShare() const {
        return h2o_fraction * h2oEscapeFraction() + o2Fraction() * o2EscapeFraction()
             + h2Fraction() * h2EscapeFraction();
    }
    double reservoirShare() const {
        return o2Fraction() * (1.0 - o2EscapeFraction()) + h2Fraction() * (1.0 - h2EscapeFraction());
    }

    // Throws std::invalid_argument on an out-of-range parameter.
    void validate() const;

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
    std::vector<double> total_mass_loss;    // kg/(m²·s) - escaping share, all products (lost to space)
    // kg/(m²·s) - non-escaping H2O. New accounting: the ballistic transport source
    // (multiscale path). Legacy: diagnostic only, never removed or transported.
    std::vector<double> redeposition_flux;
    std::vector<double> gross_removal_flux; // kg/(m²·s) - all sputtered mass (ice removed)
    std::vector<double> reservoir_flux;     // kg/(m²·s) - non-escaping O2 + H2 (non-condensable)

    explicit SputteringState(size_t num_faces) {
        ion_flux.resize(num_faces, 0.0);
        sputtering_yield.resize(num_faces, 0.0);
        h2o_sputter_flux.resize(num_faces, 0.0);
        o2_production_flux.resize(num_faces, 0.0);
        h2_production_flux.resize(num_faces, 0.0);
        total_mass_loss.resize(num_faces, 0.0);
        redeposition_flux.resize(num_faces, 0.0);
        gross_removal_flux.resize(num_faces, 0.0);
        reservoir_flux.resize(num_faces, 0.0);
    }

    void reset() {
        std::fill(ion_flux.begin(), ion_flux.end(), 0.0);
        std::fill(sputtering_yield.begin(), sputtering_yield.end(), 0.0);
        std::fill(h2o_sputter_flux.begin(), h2o_sputter_flux.end(), 0.0);
        std::fill(o2_production_flux.begin(), o2_production_flux.end(), 0.0);
        std::fill(h2_production_flux.begin(), h2_production_flux.end(), 0.0);
        std::fill(total_mass_loss.begin(), total_mass_loss.end(), 0.0);
        std::fill(redeposition_flux.begin(), redeposition_flux.end(), 0.0);
        std::fill(gross_removal_flux.begin(), gross_removal_flux.end(), 0.0);
        std::fill(reservoir_flux.begin(), reservoir_flux.end(), 0.0);
    }

    // Per-face flux the SURFACE loses to sputtering: the gross removal (new
    // accounting) or the escaping share only (legacy).
    const std::vector<double>& surfaceRemovalFlux(bool legacy_escape_only) const {
        return legacy_escape_only ? total_mass_loss : gross_removal_flux;
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

    // Current parameter set (PTM-649): H2O-equivalent molecule flux per face at
    // T_ref [molecules m^-2 s^-1] before rate_scale, and its source.
    std::vector<double> ion_molecule_flux_;
    std::vector<double> ion_map_values_;   // optional per-face map (weights or absolute)
    bool ion_map_absolute_ = false;
    std::string ion_map_label_;
    double mesh_area_ = 0.0;               // [m^2]
    double ion_total_rate_ = 0.0;          // [molecules/s] at T_ref, before rate_scale
    std::vector<double> o2_map_values_;    // optional O2 map [cm^-2 s^-1]
    std::string o2_map_label_;
    double o2_map_total_rate_ = 0.0;       // [O2 molecules/s], before rate_scale

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
        const std::vector<double>& face_longitudes = {},
        const std::vector<double>& face_areas = {});

    /**
     * Optional per-face ion map, set BEFORE initializeSpatialDistribution (current
     * parameter set only). absolute = false: weights (any scale; renormalized to an
     * area-weighted mean of 1, total N_tot = 2e27 x rate_scale). absolute = true:
     * H2O-equivalent molecules cm^-2 s^-1 at T_ref, which set pattern AND total.
     */
    void setIonMap(const std::vector<double>& values, bool absolute, const std::string& label) {
        ion_map_values_ = values;
        ion_map_absolute_ = absolute;
        ion_map_label_ = label;
    }
    bool hasIonMap() const { return !ion_map_values_.empty(); }
    /** Optional per-face O2 production map [O2 molecules cm^-2 s^-1] (see o2_map_path). */
    void setO2Map(const std::vector<double>& values, const std::string& label) {
        o2_map_values_ = values;
        o2_map_label_ = label;
    }
    bool hasO2Map() const { return !o2_map_values_.empty(); }
    /** Total O2 production of the map over the mesh [molecules/s], before rate_scale. */
    double o2MapTotalRate() const { return o2_map_total_rate_; }
    bool ionMapAbsolute() const { return ion_map_absolute_; }
    /** Total H2O-equivalent rate at T_ref over the mesh [molecules/s], before rate_scale. */
    double ionTotalRate() const { return ion_total_rate_; }
    double meshArea() const { return mesh_area_; }
    /** Per-face H2O-equivalent molecule flux at T_ref [m^-2 s^-1], before rate_scale. */
    double ionMoleculeFlux(size_t i) const {
        return (i < ion_molecule_flux_.size()) ? ion_molecule_flux_[i] : 0.0;
    }

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
