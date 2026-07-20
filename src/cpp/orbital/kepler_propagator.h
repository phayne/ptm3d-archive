#ifndef KEPLER_PROPAGATOR_H
#define KEPLER_PROPAGATOR_H

#include "orbital_elements.h"
#include <array>
#include <cmath>

namespace orbital {

// Two-body Keplerian orbit propagator.
// Computes position of a body in the ecliptic J2000 frame given
// orbital elements and time since epoch.
class KeplerianPropagator {
public:
    explicit KeplerianPropagator(const OrbitalElements& elements);

    // Get position in ecliptic J2000 frame (km) at time t seconds from epoch
    std::array<double, 3> getPosition(double t_seconds) const;

    // Get heliocentric distance in AU at time t seconds from epoch
    double getHeliocentricDistanceAU(double t_seconds) const;

    // Get true anomaly (radians) at time t seconds from epoch
    double getTrueAnomaly(double t_seconds) const;

    // Get orbital radius (km) at time t seconds from epoch
    double getOrbitalRadius(double t_seconds) const;

    // Get orbital period (seconds)
    double getOrbitalPeriod() const { return elements_.orbitalPeriod(); }

    // Access elements
    const OrbitalElements& elements() const { return elements_; }

    // Convert Julian date to seconds from epoch
    double jdToSecondsFromEpoch(double jd) const {
        return (jd - elements_.epoch_jd) * 86400.0;
    }

    // Convert ephemeris time (seconds past J2000) to seconds from our epoch
    double etToSecondsFromEpoch(double et) const {
        // ET is seconds past J2000.0 (JD 2451545.0)
        double jd = 2451545.0 + et / 86400.0;
        return jdToSecondsFromEpoch(jd);
    }

private:
    OrbitalElements elements_;

    // Mean motion (rad/s)
    double mean_motion_;

    // Semi-major axis (km)
    double a_km_;

    // Precomputed rotation matrix: orbital plane → ecliptic J2000
    // R = Rz(-Omega) * Rx(-i) * Rz(-omega)
    std::array<std::array<double, 3>, 3> rotation_matrix_;

    // Precompute the rotation matrix from orbital elements
    void precomputeRotationMatrix();

    // Solve Kepler's equation M = E - e*sin(E) for eccentric anomaly E.
    // Uses Newton-Raphson iteration.
    double solveKeplerEquation(double M, double e) const;

    // Convert eccentric anomaly to true anomaly
    double eccentricToTrueAnomaly(double E, double e) const;

    // Compute orbital radius from eccentric anomaly
    double radiusFromEccentricAnomaly(double E) const;

    // Compute mean anomaly at time t (seconds from epoch)
    double meanAnomalyAtTime(double t_seconds) const;
};

} // namespace orbital

#endif // KEPLER_PROPAGATOR_H
