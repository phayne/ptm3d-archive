#ifndef ORBITAL_ELEMENTS_H
#define ORBITAL_ELEMENTS_H

#include <cmath>

namespace orbital {

// Standard gravitational parameters (m^3/s^2)
namespace gm {
    constexpr double SUN = 1.32712440018e20;     // GM_Sun
    constexpr double JUPITER = 1.26686534e17;    // GM_Jupiter
    constexpr double SATURN = 3.7931187e16;      // GM_Saturn
    constexpr double EARTH = 3.986004418e14;     // GM_Earth
} // namespace gm

// Keplerian orbital elements (heliocentric, ecliptic J2000)
struct OrbitalElements {
    double semi_major_axis_au = 1.0;              // Semi-major axis (AU)
    double eccentricity = 0.0;                    // Orbital eccentricity [0, 1)
    double inclination_deg = 0.0;                 // Orbital inclination (degrees)
    double lon_ascending_node_deg = 0.0;          // Longitude of ascending node, Omega (degrees)
    double arg_perihelion_deg = 0.0;              // Argument of perihelion, omega (degrees)
    double mean_anomaly_at_epoch_deg = 0.0;       // Mean anomaly at epoch M0 (degrees)
    double epoch_jd = 2451545.0;                  // Julian date of epoch (default J2000.0)
    double gm_central = gm::SUN;                 // GM of central body (m^3/s^2)

    // Compute semi-major axis in meters
    double semiMajorAxisMeters() const {
        return semi_major_axis_au * AU_TO_M;
    }

    // Compute orbital period from Kepler's third law (seconds)
    double orbitalPeriod() const {
        double a_m = semiMajorAxisMeters();
        return 2.0 * M_PI * std::sqrt(a_m * a_m * a_m / gm_central);
    }

    // Compute mean motion (radians/second)
    double meanMotion() const {
        double a_m = semiMajorAxisMeters();
        return std::sqrt(gm_central / (a_m * a_m * a_m));
    }

    // AU to meters conversion
    static constexpr double AU_TO_M = 1.495978707e11;
    static constexpr double AU_TO_KM = 1.495978707e8;
};

} // namespace orbital

#endif // ORBITAL_ELEMENTS_H
