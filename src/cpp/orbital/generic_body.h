#ifndef GENERIC_BODY_H
#define GENERIC_BODY_H

#include "orbital_elements.h"
#include "spin_state.h"
#include "body_shape.h"
#include "../thermal/planetary_properties.h"
#include <string>

namespace orbital {

// High-level container that bundles all information needed to define
// a generic planetary body for thermal simulation.
struct GenericBody {
    std::string name = "CustomBody";

    // Shape
    BodyShape shape;

    // Orbit (heliocentric)
    OrbitalElements orbit;

    // Rotation state
    SpinState spin;

    // Thermophysical properties (reuses existing struct)
    thermal::ThermophysicalProperties thermo;

    // Ephemeris source
    enum EphemerisSource { KEPLERIAN, SPICE };
    EphemerisSource ephemeris_source = KEPLERIAN;
    std::string spice_metakernel;   // Used when ephemeris_source == SPICE
    std::string spice_body_id;      // NAIF ID for SPICE queries (e.g. "2000269")

    // Populate ThermophysicalProperties derived fields from orbital/spin data
    void updateDerivedProperties() {
        // Rotation period → synodic day
        double sidereal_day_s = spin.spin_period_s;
        double orbital_period_s = orbit.orbitalPeriod();

        // Synodic day = 1 / |1/P_sid - 1/P_orb|
        if (sidereal_day_s > 0.0 && orbital_period_s > 0.0 &&
            std::abs(sidereal_day_s - orbital_period_s) > 1.0) {
            thermo.day = 1.0 / std::abs(1.0 / sidereal_day_s - 1.0 / orbital_period_s);
        } else {
            thermo.day = sidereal_day_s;  // Synchronous or very close
        }

        thermo.year = orbital_period_s;
        thermo.rAU = orbit.semi_major_axis_au;  // Mean distance

        // Solar constant at mean distance
        constexpr double SOLAR_CONSTANT_1AU = 1361.0;  // W/m^2
        if (thermo.rAU > 0.0) {
            thermo.S = SOLAR_CONSTANT_1AU / (thermo.rAU * thermo.rAU);
        }

        // Set SPICE metakernel if using SPICE
        if (ephemeris_source == SPICE && !spice_metakernel.empty()) {
            thermo.spice_metakernel = spice_metakernel;
        }
    }

    // Validate that all required fields are populated
    bool validate(std::string& error_msg) const {
        if (name.empty()) {
            error_msg = "Body name is empty";
            return false;
        }
        if (!shape.isValid()) {
            error_msg = "Invalid shape parameters for body '" + name + "'";
            return false;
        }
        if (orbit.semi_major_axis_au <= 0.0) {
            error_msg = "Semi-major axis must be positive for body '" + name + "'";
            return false;
        }
        if (orbit.eccentricity < 0.0 || orbit.eccentricity >= 1.0) {
            error_msg = "Eccentricity must be in [0, 1) for body '" + name + "'";
            return false;
        }
        if (spin.spin_period_s <= 0.0) {
            error_msg = "Spin period must be positive for body '" + name + "'";
            return false;
        }
        if (thermo.albedo < 0.0 || thermo.albedo > 1.0) {
            error_msg = "Albedo must be in [0, 1] for body '" + name + "'";
            return false;
        }
        if (thermo.emissivity <= 0.0 || thermo.emissivity > 1.0) {
            error_msg = "Emissivity must be in (0, 1] for body '" + name + "'";
            return false;
        }
        if (ephemeris_source == SPICE && spice_metakernel.empty()) {
            error_msg = "SPICE ephemeris source requires a metakernel for body '" + name + "'";
            return false;
        }
        return true;
    }

    // Create default thermophysical properties for a rocky asteroid
    static GenericBody defaultRockyAsteroid() {
        GenericBody body;
        body.name = "RockyAsteroid";
        body.shape.type = BodyShape::SPHERICAL;
        body.shape.radius_km = 50.0;

        body.orbit.semi_major_axis_au = 2.5;
        body.orbit.eccentricity = 0.1;
        body.orbit.inclination_deg = 5.0;

        body.spin.spin_period_s = 6.0 * 3600.0;  // 6 hours
        body.spin.pole_ra_deg = 0.0;
        body.spin.pole_dec_deg = 90.0;  // Pole perpendicular to ecliptic

        body.thermo.albedo = 0.1;
        body.thermo.emissivity = 0.95;
        body.thermo.Qb = 0.0;
        body.thermo.ks = 0.01;
        body.thermo.kd = 0.5;
        body.thermo.rhos = 1500.0;
        body.thermo.rhod = 2500.0;
        body.thermo.H = 0.05;
        body.thermo.cp0 = 800.0;
        body.thermo.cpCoeff = {};
        body.thermo.albedoCoef = {};

        body.updateDerivedProperties();
        return body;
    }
};

} // namespace orbital

#endif // GENERIC_BODY_H
