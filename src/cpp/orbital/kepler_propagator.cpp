#include "kepler_propagator.h"
#include <stdexcept>
#include <string>
#include <algorithm>

namespace orbital {

KeplerianPropagator::KeplerianPropagator(const OrbitalElements& elements)
    : elements_(elements)
{
    if (elements_.eccentricity < 0.0 || elements_.eccentricity >= 1.0) {
        throw std::invalid_argument(
            "KeplerianPropagator: eccentricity must be in [0, 1), got " +
            std::to_string(elements_.eccentricity));
    }
    if (elements_.semi_major_axis_au <= 0.0) {
        throw std::invalid_argument(
            "KeplerianPropagator: semi-major axis must be positive, got " +
            std::to_string(elements_.semi_major_axis_au) + " AU");
    }

    a_km_ = elements_.semi_major_axis_au * OrbitalElements::AU_TO_KM;
    mean_motion_ = elements_.meanMotion();

    precomputeRotationMatrix();
}

void KeplerianPropagator::precomputeRotationMatrix() {
    // Convert angles to radians
    const double Omega = elements_.lon_ascending_node_deg * M_PI / 180.0;
    const double i = elements_.inclination_deg * M_PI / 180.0;
    const double omega = elements_.arg_perihelion_deg * M_PI / 180.0;

    // Precompute trig
    const double cO = std::cos(Omega), sO = std::sin(Omega);
    const double ci = std::cos(i),     si = std::sin(i);
    const double cw = std::cos(omega), sw = std::sin(omega);

    // Rotation matrix: orbital plane (x=perihelion, y=90deg ahead) → ecliptic J2000
    // This is the standard Rz(-Omega) * Rx(-i) * Rz(-omega) composition.
    // Column 0: direction of perihelion in ecliptic
    // Column 1: 90 degrees ahead of perihelion in ecliptic
    // Column 2: orbit normal in ecliptic
    rotation_matrix_[0][0] =  cO * cw - sO * sw * ci;
    rotation_matrix_[0][1] = -cO * sw - sO * cw * ci;
    rotation_matrix_[0][2] =  sO * si;

    rotation_matrix_[1][0] =  sO * cw + cO * sw * ci;
    rotation_matrix_[1][1] = -sO * sw + cO * cw * ci;
    rotation_matrix_[1][2] = -cO * si;

    rotation_matrix_[2][0] =  sw * si;
    rotation_matrix_[2][1] =  cw * si;
    rotation_matrix_[2][2] =  ci;
}

double KeplerianPropagator::solveKeplerEquation(double M, double e) const {
    // Normalize M to [0, 2*pi)
    M = std::fmod(M, 2.0 * M_PI);
    if (M < 0.0) M += 2.0 * M_PI;

    // Initial guess: E = M for small e, Markley's starter for larger e
    double E = M;
    if (e > 0.1) {
        // Better initial guess for moderate eccentricity
        E = M + e * std::sin(M) * (1.0 + e * std::cos(M));
    }

    // Newton-Raphson iteration
    // f(E) = E - e*sin(E) - M = 0
    // f'(E) = 1 - e*cos(E)
    constexpr int MAX_ITER = 30;
    constexpr double TOL = 1.0e-15;

    for (int iter = 0; iter < MAX_ITER; ++iter) {
        double sinE = std::sin(E);
        double cosE = std::cos(E);
        double f = E - e * sinE - M;
        double fp = 1.0 - e * cosE;

        // Halley's method for faster convergence
        double fpp = e * sinE;
        double dE = -f / (fp - 0.5 * f * fpp / fp);

        E += dE;

        if (std::abs(dE) < TOL) {
            return E;
        }
    }

    // If we get here, convergence failed (should not happen for e < 1)
    return E;
}

double KeplerianPropagator::eccentricToTrueAnomaly(double E, double e) const {
    // tan(f/2) = sqrt((1+e)/(1-e)) * tan(E/2)
    double half_f = std::atan2(
        std::sqrt(1.0 + e) * std::sin(E / 2.0),
        std::sqrt(1.0 - e) * std::cos(E / 2.0)
    );
    return 2.0 * half_f;
}

double KeplerianPropagator::radiusFromEccentricAnomaly(double E) const {
    return a_km_ * (1.0 - elements_.eccentricity * std::cos(E));
}

double KeplerianPropagator::meanAnomalyAtTime(double t_seconds) const {
    double M0 = elements_.mean_anomaly_at_epoch_deg * M_PI / 180.0;
    return M0 + mean_motion_ * t_seconds;
}

double KeplerianPropagator::getTrueAnomaly(double t_seconds) const {
    double M = meanAnomalyAtTime(t_seconds);
    double E = solveKeplerEquation(M, elements_.eccentricity);
    return eccentricToTrueAnomaly(E, elements_.eccentricity);
}

double KeplerianPropagator::getOrbitalRadius(double t_seconds) const {
    double M = meanAnomalyAtTime(t_seconds);
    double E = solveKeplerEquation(M, elements_.eccentricity);
    return radiusFromEccentricAnomaly(E);
}

double KeplerianPropagator::getHeliocentricDistanceAU(double t_seconds) const {
    return getOrbitalRadius(t_seconds) / OrbitalElements::AU_TO_KM;
}

std::array<double, 3> KeplerianPropagator::getPosition(double t_seconds) const {
    // Step 1: Compute mean anomaly at time t
    double M = meanAnomalyAtTime(t_seconds);

    // Step 2: Solve Kepler's equation for eccentric anomaly
    double E = solveKeplerEquation(M, elements_.eccentricity);

    // Step 3: Position in orbital plane
    double r = radiusFromEccentricAnomaly(E);
    double f = eccentricToTrueAnomaly(E, elements_.eccentricity);
    double x_orb = r * std::cos(f);
    double y_orb = r * std::sin(f);
    // z_orb = 0 (in-plane)

    // Step 4: Rotate to ecliptic J2000 using precomputed matrix
    std::array<double, 3> pos;
    pos[0] = rotation_matrix_[0][0] * x_orb + rotation_matrix_[0][1] * y_orb;
    pos[1] = rotation_matrix_[1][0] * x_orb + rotation_matrix_[1][1] * y_orb;
    pos[2] = rotation_matrix_[2][0] * x_orb + rotation_matrix_[2][1] * y_orb;

    return pos;
}

} // namespace orbital
