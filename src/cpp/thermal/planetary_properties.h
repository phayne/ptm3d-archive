#ifndef PLANETARY_PROPERTIES_H
#define PLANETARY_PROPERTIES_H

#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <iostream>

namespace thermal {

// Enumeration of planetary bodies the CONFIGURATION SURFACE names.
//
// ⚠️ Membership here is NOT the same thing as being implemented. Only the
// three bodies listed in kImplementedBodies below have thermophysical
// properties in this file; ENCELADUS, TITAN, MARS and CERES reach
// getProperties()'s `default:` arm and come back with EUROPA's radius, day,
// albedo, conductivity and metakernel. The enum keeps them because
// IlluminationModel::setPlanetaryBody switches on all seven, and because a
// config that names one must be REFUSED by name rather than silently mapped
// to some other enumerator (see isImplementedBody).
enum class PlanetaryBody {
    MOON,
    MERCURY,
    EUROPA,
    ENCELADUS,
    TITAN,
    MARS,
    CERES,
    CUSTOM
};

// Structure to hold all thermophysical properties
struct ThermophysicalProperties {
    // Orbital and rotational parameters
    //
    // `day` is the SOLAR (synodic) day in seconds -- sunrise to sunrise -- NOT
    // the sidereal rotation period. Everything downstream treats it as the
    // period of the solar forcing: the hour angle (thermal_model.cpp
    // surfFlux), the diurnal skin depth and hence the depth grid
    // (skinDepth(planet.day, ...)), the timestep (getTimeStep), the
    // equilibration block length (ptm_thermal_sim `equilibration_days`), the
    // spectral equilibrator's `cycle_s`, and the output cadence. On a
    // synchronous rotator (Europa) and on the Moon the distinction is either
    // nil or already folded in; on Mercury it is a factor of three, and
    // getting it wrong there was BUGS.md ENG-MERCURY-SOLAR-DAY.
    double day;           // Solar (synodic) day (seconds)
    double year;          // Orbital period (seconds)
    double rAU;           // Distance from Sun (AU)
    double S;             // Solar constant at orbit (W/m²)

    // Surface properties
    double albedo;        // Bond albedo
    double emissivity;    // Thermal emissivity
    double Qb;            // Internal heat flux (W/m²)

    // Thermal properties
    double ks;            // Surface thermal conductivity (W/m/K)
    double kd;            // Deep thermal conductivity (W/m/K)
    double rhos;          // Surface density (kg/m³)
    double rhod;          // Deep density (kg/m³)
    double H;             // Scale height for density/conductivity change (m)
    double cp0;           // Reference heat capacity (J/kg/K)

    // Variable property coefficients
    std::vector<double> cpCoeff;     // Heat capacity temperature coefficients
    std::vector<double> albedoCoef;  // Albedo variation coefficients

    // SPICE configuration
    std::string spice_metakernel;  // Meta-kernel filename for SPICE

    // Body physical parameters (for mesh generation and generic body support)
    double body_radius_km = 0.0;  // Mean radius (km). 0 = use mesh as-is
};

// ---------------------------------------------------------------------------
// THE IMPLEMENTED-BODY REGISTRY -- the one authority on which `simulation.body`
// values this engine can actually run without a `generic_body:` block.
//
// Anything not on this list has no properties here. Until 2026-08-26 such a
// name did not fail: getPropertiesByName() printed one `Note:` line on stderr
// and returned EUROPA's properties, so a config saying `body: Mars` ran to
// COMPLETION with exit 0, Europa's 1560.8 km radius, Europa's europa.tm
// metakernel, a 3.55-day synodic banner, and 0.000000000 W/m^2 of flux on every
// one of its faces -- sixteen statistics binaries of a simulation of nothing.
// Measured on four arms (Mars, Ceres, Titan, Enceladus). `ptm_thermal_sim` now
// refuses such a config at parse time (BUGS.md ENG-BODY-UNBACKED-RUN).
//
// Adding a body means adding a getXxxProperties() function AND a row here AND
// regenerating src/python/ptm/body_capabilities.py; the parity test
// src/python/tests/test_body_capabilities.py fails until all three agree.
inline const std::vector<std::string>& implementedBodies() {
    static const std::vector<std::string> kImplementedBodies = {
        "Moon", "Europa", "Mercury"
    };
    return kImplementedBodies;
}

// Case-insensitive membership in implementedBodies(). Returns the CANONICAL
// spelling through `canonical` when non-null, so callers report "Moon" rather
// than echoing back whatever case the config used.
inline bool isImplementedBody(const std::string& name, std::string* canonical = nullptr) {
    std::string lower_name = name;
    std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    for (const auto& body : implementedBodies()) {
        std::string lower_body = body;
        std::transform(lower_body.begin(), lower_body.end(), lower_body.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (lower_name == lower_body) {
            if (canonical) *canonical = body;
            return true;
        }
    }
    return false;
}

// "Moon, Europa or Mercury" -- for the message that refuses everything else.
inline std::string implementedBodiesList() {
    const auto& bodies = implementedBodies();
    std::string out;
    for (size_t i = 0; i < bodies.size(); ++i) {
        if (i > 0) out += (i + 1 == bodies.size()) ? " or " : ", ";
        out += bodies[i];
    }
    return out;
}

// Default properties for supported planetary bodies
class PlanetaryData {
public:
    // Get properties for a specific body
    static ThermophysicalProperties getProperties(PlanetaryBody body) {
        switch (body) {
            case PlanetaryBody::EUROPA:
                return getEuropaProperties();
            case PlanetaryBody::MOON:
                return getMoonProperties();
            case PlanetaryBody::MERCURY:
                return getMercuryProperties();
            // Add other bodies as needed
            default:
                return getEuropaProperties(); // Default to Europa
        }
    }
    
    // Get properties by name
    static ThermophysicalProperties getPropertiesByName(const std::string& name) {
        // Convert to lowercase for case-insensitive comparison
        std::string lower_name = name;
        for (auto& c : lower_name) c = std::tolower(c);
        
        if (lower_name == "europa") return getProperties(PlanetaryBody::EUROPA);
        if (lower_name == "moon") return getProperties(PlanetaryBody::MOON);
        if (lower_name == "mercury") return getProperties(PlanetaryBody::MERCURY);
        // Add other bodies as needed

        // Not an implemented body. The ONLY legitimate caller left here is the
        // generic-body path: `thermal::Planet` is constructed from the config's
        // body NAME early in main(), before the `generic_body:` block has been
        // read, so an asteroid's Planet is briefly seeded with these
        // placeholders and then overwritten field by field from the generic
        // body. Every other route is refused by name at config parse
        // (isImplementedBody / ENG-BODY-UNBACKED-RUN), which is what this
        // warning could not do on its own -- it was one stderr line in front of
        // a complete, exit-0, all-zero-flux simulation.
        static bool warned = false;
        if (!warned) {
            std::cerr << "Note: body '" << name << "' is not one of the "
                      << "implemented bodies (" << implementedBodiesList()
                      << "); seeding placeholder properties that the "
                         "generic_body: block must overwrite. If this run has "
                         "no generic_body: block it is already being refused."
                      << std::endl;
            warned = true;
        }
        return getProperties(PlanetaryBody::EUROPA);
    }

private:
    // Europa properties
    static ThermophysicalProperties getEuropaProperties() {
        ThermophysicalProperties props;
        
        // Orbital parameters
        props.day = 3.55 * 24 * 3600;    // Europa's day in seconds (synchronous)
        props.year = 3.55 * 24 * 3600;   // Same as day for synchronous rotation
        props.rAU = 5.2;                 // Distance from Sun in AU
        props.S = 52.0;                  // Solar constant at Europa's orbit
        
        // Surface properties
        props.albedo = 0.5;              // Bond albedo
        props.emissivity = 0.9;          // Thermal emissivity
        props.Qb = 0.03;                 // Internal heat flux (W/m²)
        
        // Thermal properties for Europa icy regolith
        // Target TI ~ 60 SI units: TI = sqrt(k * rho * cp)
        // At T=100K with cp~750 J/kg/K: TI = sqrt(0.0096 * 500 * 750) ≈ 60 SI
        props.ks = 0.0096;               // Surface thermal conductivity (W/m/K) - porous ice
        props.kd = 0.05;                 // Deep thermal conductivity (W/m/K)
        props.rhos = 500.0;              // Surface density (kg/m³) - porous ice regolith
        props.rhod = 700.0;              // Deep density (kg/m³)
        props.H = 0.07;                  // Scale height (m)
        props.cp0 = 0.0;                 // Reference heat capacity (J/kg/K) - using linear T model

        // Variable property coefficients for water ice
        // Ice heat capacity: cp ≈ 7.49*T for T < ~270K (Giauque & Stout 1936)
        // This gives cp(100K) ≈ 749 J/kg/K, cp(130K) ≈ 974 J/kg/K
        // Empty cpCoeff means constant cp; single value means cp = cp0 + c0*T
        props.cpCoeff = {7.49};          // Linear temperature coefficient for ice (J/kg/K²)
        props.albedoCoef = {};           // No incidence-dependent albedo for Europa
        
        // SPICE configuration
        props.spice_metakernel = "europa.tm";

        // Body size
        props.body_radius_km = 1560.8;

        return props;
    }

    // Moon properties - Hayne et al. (2017) JGR Planets, Table A1
    // "Global Regolith Thermophysical Properties of the Moon From the Diviner
    // Lunar Radiometer Experiment"
    static ThermophysicalProperties getMoonProperties() {
        ThermophysicalProperties props;

        // Orbital parameters
        props.day = 29.530588 * 24 * 3600;    // Synodic lunar day in seconds
        props.year = 365.25 * 24 * 3600;      // Earth year in seconds
        props.rAU = 1.0;                      // Same distance as Earth
        props.S = 1361.0;                     // Solar constant at 1 AU (W/m²)

        // Surface properties (Hayne et al. 2017 Table A1)
        props.albedo = 0.12;                  // Hemispherical albedo A0 at normal incidence (theta=0)
        props.emissivity = 0.95;              // Thermal emissivity
        props.Qb = 0.018;                     // Internal heat flux (W/m²), Langseth et al. 1976

        // Thermal properties (Hayne et al. 2017 Table A1)
        props.ks = 7.4e-4;                    // Surface thermal conductivity Kc (W/m/K)
        props.kd = 3.4e-3;                    // Deep thermal conductivity Kd (W/m/K)
        props.rhos = 1100.0;                  // Surface density rho_s (kg/m³)
        props.rhod = 1800.0;                  // Deep density rho_d (kg/m³)
        props.H = 0.06;                       // H-parameter, e-folding depth (m)
        props.cp0 = -3.6125;                  // Heat capacity constant term c0 (J/kg/K)

        // Heat capacity polynomial coefficients (Hayne et al. 2017 Table A1)
        // cp(T) = c0 + c1*T + c2*T^2 + c3*T^3 + c4*T^4
        // cpCoeff = {c1, c2, c3, c4} (c0 is stored in cp0)
        props.cpCoeff = {2.7431, 2.3616e-3, -1.2340e-5, 8.9093e-9};

        // Albedo variation coefficients (Hayne et al. 2017 Table A1)
        props.albedoCoef = {0.06, 0.25};      // A(theta) = A0 + a*(theta/(pi/4))^3 + b*(theta/(pi/2))^8

        // SPICE configuration
        props.spice_metakernel = "moon.tm";

        // Body size
        props.body_radius_km = 1737.4;

        return props;
    }

    // Mercury properties
    static ThermophysicalProperties getMercuryProperties() {
        ThermophysicalProperties props;

        // Orbital parameters
        //
        // ⚠️ `day` is the SOLAR (synodic) day -- sunrise to sunrise -- on every
        // body in this file, because that is the period the thermal model is
        // forced at: it sets the hour angle, the diurnal skin depth, the
        // timestep, the equilibration block length and the output cadence
        // (thermal_model.cpp: surfFlux/getTimeStep/skinDepth, ptm_thermal_sim
        // equilibration_days). Mercury is the one body where that differs from
        // the rotation period, and until 2026-08-26 this line carried the
        // SIDEREAL rotation, 58.646 d, while the banner printed it as
        // "Synodic day (solar day)". Every derived quantity was therefore a
        // third of its true length: `thermal.equilibration_orbits: 50` bought
        // 50 sidereal rotations of spin-up rather than 50 solar days, and the
        // diurnal skin depth was sqrt(3) = 1.73x too shallow.
        //
        // Mercury is in a 3:2 spin-orbit resonance, so the solar day is
        //   P_solar = 1 / |1/P_sidereal - 1/P_orbital| = 3 * P_sidereal
        //           = 175.938 d  = exactly two Mercury years.
        // Written as the product below rather than as one opaque literal so
        // that the resonance is visible in the source, and cross-checked
        // against the two periods by the static_assert underneath so the three
        // numbers cannot drift apart. Same expression as
        // orbital/generic_body.h's synodic-day derivation.
        //
        // ⛔ THIS CHANGES MERCURY PHYSICS. No shipped config, tutorial or
        // dataset selects Mercury (0 of 78 run configs, `ALBEDO_BODIES` and
        // `TOPO_BODIES` exclude it), so the blast radius is the tutorial and
        // testing surface only -- but a Mercury run on this binary is not
        // comparable with one from before. BUGS.md ENG-MERCURY-SOLAR-DAY.
        constexpr double kMercurySiderealRotationDays = 58.646;
        constexpr double kMercuryOrbitalPeriodDays    = 87.969;
        constexpr double kMercurySolarDayDays         = 3.0 * kMercurySiderealRotationDays;
        static_assert(
            kMercurySolarDayDays * (1.0 / kMercurySiderealRotationDays
                                    - 1.0 / kMercuryOrbitalPeriodDays) > 0.9999 &&
            kMercurySolarDayDays * (1.0 / kMercurySiderealRotationDays
                                    - 1.0 / kMercuryOrbitalPeriodDays) < 1.0001,
            "Mercury's solar day must equal 1/(1/P_sidereal - 1/P_orbital); "
            "one of the three periods was edited without the others");

        props.day = kMercurySolarDayDays * 24 * 3600;   // solar day, 175.938 d
        props.year = kMercuryOrbitalPeriodDays * 24 * 3600; // Mercury year in seconds
        props.rAU = 0.387;               // Average distance from Sun in AU
        props.S = 9126.6;                // Solar constant at Mercury's orbit
        
        // Surface properties
        props.albedo = 0.07;             // Bond albedo
        props.emissivity = 0.93;         // Thermal emissivity
        props.Qb = 0.02;                 // Internal heat flux (W/m²)
        
        // Thermal properties
        props.ks = 0.002;                // Surface thermal conductivity (W/m/K)
        props.kd = 2.0;                  // Deep thermal conductivity (W/m/K)
        props.rhos = 1800.0;             // Surface density (kg/m³)
        props.rhod = 2200.0;             // Deep density (kg/m³)
        props.H = 0.1;                   // Scale height (m)
        props.cp0 = 800.0;               // Reference heat capacity (J/kg/K)
        
        // Variable property coefficients
        props.cpCoeff = {0.25, 0.0};     // Heat capacity temperature coefficients
        props.albedoCoef = {0.02, 0.0};  // Albedo variation coefficients

        // SPICE configuration
        props.spice_metakernel = "mercury.tm";

        // Body size
        props.body_radius_km = 2439.7;

        return props;
    }
};

} // namespace thermal

#endif // PLANETARY_PROPERTIES_H