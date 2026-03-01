#ifndef SPIN_STATE_H
#define SPIN_STATE_H

#include <cmath>

namespace orbital {

// Rotational state of a body (IAU convention)
// Pole orientation in equatorial J2000, with conversion to ecliptic
struct SpinState {
    double spin_period_s = 86400.0;           // Sidereal rotation period (seconds)
    double pole_ra_deg = 0.0;                 // Right ascension of north pole (degrees, J2000 equatorial)
    double pole_dec_deg = 90.0;               // Declination of north pole (degrees, J2000 equatorial)
    double prime_meridian_deg = 0.0;          // Prime meridian W0 at epoch (degrees)
    double prime_meridian_rate_deg_per_day = 0.0; // dW/dt (degrees/day), derived from spin_period if 0
    double epoch_jd = 2451545.0;              // Reference epoch (default J2000.0)

    // Compute rotation rate in degrees/day from spin period
    double rotationRateDegPerDay() const {
        if (prime_meridian_rate_deg_per_day != 0.0) {
            return prime_meridian_rate_deg_per_day;
        }
        if (spin_period_s > 0.0) {
            return 360.0 / (spin_period_s / 86400.0);
        }
        return 0.0;
    }

    // Compute rotation rate in radians/second
    double rotationRateRadPerSec() const {
        if (spin_period_s > 0.0) {
            return 2.0 * M_PI / spin_period_s;
        }
        return 0.0;
    }

    // Earth's obliquity for equatorial-to-ecliptic conversion (J2000, radians)
    static constexpr double OBLIQUITY_J2000_RAD = 23.4392911 * M_PI / 180.0;
};

} // namespace orbital

#endif // SPIN_STATE_H
