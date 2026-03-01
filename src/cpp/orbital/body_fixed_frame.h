#ifndef BODY_FIXED_FRAME_H
#define BODY_FIXED_FRAME_H

#include "spin_state.h"
#include <array>

namespace orbital {

// Transforms vectors between ecliptic J2000 and body-fixed frames.
// The body-fixed frame is defined by:
//   - z-axis: spin pole direction (from RA/Dec of north pole)
//   - x-axis: prime meridian direction, rotating with time
//   - y-axis: completes right-handed frame
//
// This replicates what SPICE's IAU_<body> frame does, but without
// requiring SPICE frame kernels.
class BodyFixedFrame {
public:
    explicit BodyFixedFrame(const SpinState& spin);

    // Transform a vector from ecliptic J2000 to body-fixed frame at time t
    // (seconds from the spin state epoch)
    std::array<double, 3> eclipticToBodyFixed(
        const std::array<double, 3>& vec_ecliptic,
        double t_seconds) const;

    // Transform a vector from body-fixed frame to ecliptic J2000 at time t
    std::array<double, 3> bodyFixedToEcliptic(
        const std::array<double, 3>& vec_body_fixed,
        double t_seconds) const;

    // Get the Sun's position in body-fixed frame at time t.
    // body_pos_ecliptic: position of body relative to Sun in ecliptic J2000 (km).
    // Returns: Sun position relative to body in body-fixed frame (km).
    std::array<double, 3> getSunInBodyFixed(
        double t_seconds,
        const std::array<double, 3>& body_pos_ecliptic) const;

    // Get spin pole direction in ecliptic J2000 (unit vector)
    const std::array<double, 3>& spinPoleEcliptic() const { return spin_pole_ecliptic_; }

    // Get rotation angle W(t) at time t (radians)
    double getRotationAngle(double t_seconds) const;

    // Access spin state
    const SpinState& spinState() const { return spin_; }

private:
    SpinState spin_;

    // Spin pole direction in ecliptic J2000 (unit vector)
    std::array<double, 3> spin_pole_ecliptic_;

    // Reference x-axis in ecliptic (prime meridian at epoch, before rotation)
    // This is an arbitrary direction perpendicular to the spin pole
    std::array<double, 3> ref_x_ecliptic_;

    // Reference y-axis in ecliptic (completes frame)
    std::array<double, 3> ref_y_ecliptic_;

    // Rotation rate (rad/s)
    double rotation_rate_rad_s_;

    // Compute spin pole in ecliptic J2000 from equatorial RA/Dec
    void computeSpinPoleEcliptic();

    // Set up the reference frame axes (perpendicular to spin pole)
    void computeReferenceAxes();

    // Helper: 3D cross product
    static std::array<double, 3> cross(
        const std::array<double, 3>& a,
        const std::array<double, 3>& b);

    // Helper: 3D dot product
    static double dot(
        const std::array<double, 3>& a,
        const std::array<double, 3>& b);

    // Helper: normalize vector
    static void normalize(std::array<double, 3>& v);
};

} // namespace orbital

#endif // BODY_FIXED_FRAME_H
