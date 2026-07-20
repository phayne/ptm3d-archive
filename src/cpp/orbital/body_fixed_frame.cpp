#include "body_fixed_frame.h"
#include <cmath>
#include <stdexcept>

namespace orbital {

BodyFixedFrame::BodyFixedFrame(const SpinState& spin)
    : spin_(spin)
{
    rotation_rate_rad_s_ = spin_.rotationRateRadPerSec();
    computeSpinPoleEcliptic();
    computeReferenceAxes();
}

void BodyFixedFrame::computeSpinPoleEcliptic() {
    // Convert pole RA/Dec (equatorial J2000) to Cartesian equatorial
    double ra_rad = spin_.pole_ra_deg * M_PI / 180.0;
    double dec_rad = spin_.pole_dec_deg * M_PI / 180.0;

    double x_eq = std::cos(dec_rad) * std::cos(ra_rad);
    double y_eq = std::cos(dec_rad) * std::sin(ra_rad);
    double z_eq = std::sin(dec_rad);

    // Rotate from equatorial J2000 to ecliptic J2000
    // Rotation about x-axis by obliquity epsilon
    // x_ecl = x_eq
    // y_ecl = y_eq * cos(eps) + z_eq * sin(eps)
    // z_ecl = -y_eq * sin(eps) + z_eq * cos(eps)
    double eps = SpinState::OBLIQUITY_J2000_RAD;
    double ce = std::cos(eps);
    double se = std::sin(eps);

    spin_pole_ecliptic_[0] = x_eq;
    spin_pole_ecliptic_[1] = y_eq * ce + z_eq * se;
    spin_pole_ecliptic_[2] = -y_eq * se + z_eq * ce;

    normalize(spin_pole_ecliptic_);
}

void BodyFixedFrame::computeReferenceAxes() {
    // We need a reference x-axis perpendicular to the spin pole.
    // Use the vernal equinox direction (ecliptic x-axis) as seed,
    // then Gram-Schmidt to get a perpendicular vector.
    std::array<double, 3> seed = {1.0, 0.0, 0.0};

    // If spin pole is nearly aligned with ecliptic x, use ecliptic y instead
    if (std::abs(dot(spin_pole_ecliptic_, seed)) > 0.9) {
        seed = {0.0, 1.0, 0.0};
    }

    // ref_x = seed - (seed . pole) * pole, then normalize
    double proj = dot(seed, spin_pole_ecliptic_);
    ref_x_ecliptic_[0] = seed[0] - proj * spin_pole_ecliptic_[0];
    ref_x_ecliptic_[1] = seed[1] - proj * spin_pole_ecliptic_[1];
    ref_x_ecliptic_[2] = seed[2] - proj * spin_pole_ecliptic_[2];
    normalize(ref_x_ecliptic_);

    // ref_y = pole x ref_x
    ref_y_ecliptic_ = cross(spin_pole_ecliptic_, ref_x_ecliptic_);
    normalize(ref_y_ecliptic_);

    // Now rotate ref_x and ref_y by the prime meridian angle at epoch
    // so that at t=0, body-fixed x-axis points to the prime meridian
    double W0 = spin_.prime_meridian_deg * M_PI / 180.0;
    if (W0 != 0.0) {
        double cw = std::cos(W0);
        double sw = std::sin(W0);
        // Rotate about the spin pole by W0
        std::array<double, 3> new_x, new_y;
        for (int k = 0; k < 3; ++k) {
            new_x[k] = cw * ref_x_ecliptic_[k] + sw * ref_y_ecliptic_[k];
            new_y[k] = -sw * ref_x_ecliptic_[k] + cw * ref_y_ecliptic_[k];
        }
        ref_x_ecliptic_ = new_x;
        ref_y_ecliptic_ = new_y;
    }
}

double BodyFixedFrame::getRotationAngle(double t_seconds) const {
    // W(t) = rotation_rate * t (prime meridian angle at epoch already baked into ref axes)
    return rotation_rate_rad_s_ * t_seconds;
}

std::array<double, 3> BodyFixedFrame::eclipticToBodyFixed(
    const std::array<double, 3>& vec_ecliptic,
    double t_seconds) const
{
    // Rotation angle since epoch
    double W = getRotationAngle(t_seconds);
    double cW = std::cos(W);
    double sW = std::sin(W);

    // Body-fixed axes in ecliptic frame at time t:
    // x_bf = cos(W) * ref_x + sin(W) * ref_y   (rotated by W about pole)
    // y_bf = -sin(W) * ref_x + cos(W) * ref_y
    // z_bf = spin_pole

    // Project vec_ecliptic onto time-dependent body-fixed axes
    // First project onto ref_x and ref_y (static), then apply rotation
    double vx_ref = dot(vec_ecliptic, ref_x_ecliptic_);
    double vy_ref = dot(vec_ecliptic, ref_y_ecliptic_);
    double vz = dot(vec_ecliptic, spin_pole_ecliptic_);

    // Apply rotation: body-fixed x = cos(W)*vx_ref + sin(W)*vy_ref
    //                  body-fixed y = -sin(W)*vx_ref + cos(W)*vy_ref
    std::array<double, 3> result;
    result[0] = cW * vx_ref + sW * vy_ref;
    result[1] = -sW * vx_ref + cW * vy_ref;
    result[2] = vz;

    return result;
}

std::array<double, 3> BodyFixedFrame::bodyFixedToEcliptic(
    const std::array<double, 3>& vec_body_fixed,
    double t_seconds) const
{
    double W = getRotationAngle(t_seconds);
    double cW = std::cos(W);
    double sW = std::sin(W);

    // Inverse rotation: ref_x component = cos(W)*bf_x - sin(W)*bf_y
    //                   ref_y component = sin(W)*bf_x + cos(W)*bf_y
    double vx_ref = cW * vec_body_fixed[0] - sW * vec_body_fixed[1];
    double vy_ref = sW * vec_body_fixed[0] + cW * vec_body_fixed[1];
    double vz = vec_body_fixed[2];

    // Reconstruct ecliptic vector from ref_x, ref_y, pole components
    std::array<double, 3> result;
    for (int k = 0; k < 3; ++k) {
        result[k] = vx_ref * ref_x_ecliptic_[k]
                   + vy_ref * ref_y_ecliptic_[k]
                   + vz * spin_pole_ecliptic_[k];
    }
    return result;
}

std::array<double, 3> BodyFixedFrame::getSunInBodyFixed(
    double t_seconds,
    const std::array<double, 3>& body_pos_ecliptic) const
{
    // Sun position relative to body = -body_pos (since body_pos is body relative to Sun)
    std::array<double, 3> sun_ecliptic = {
        -body_pos_ecliptic[0],
        -body_pos_ecliptic[1],
        -body_pos_ecliptic[2]
    };

    return eclipticToBodyFixed(sun_ecliptic, t_seconds);
}

std::array<double, 3> BodyFixedFrame::cross(
    const std::array<double, 3>& a,
    const std::array<double, 3>& b)
{
    return {
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    };
}

double BodyFixedFrame::dot(
    const std::array<double, 3>& a,
    const std::array<double, 3>& b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

void BodyFixedFrame::normalize(std::array<double, 3>& v) {
    double mag = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if (mag > 0.0) {
        v[0] /= mag;
        v[1] /= mag;
        v[2] /= mag;
    }
}

} // namespace orbital
