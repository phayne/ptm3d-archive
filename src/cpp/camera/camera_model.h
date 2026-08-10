#pragma once

#include <array>
#include <vector>
#include <string>
#include <cmath>

namespace ptm {

struct CameraConfig {
    enum class Mode { GENERIC, SPICE_IK };
    Mode mode = Mode::GENERIC;

    // Generic mode: camera position in body-fixed frame (km)
    std::array<double, 3> position{0.0, 0.0, 2000.0};

    // Alternative: specify position as lat/lon/alt
    bool use_latlon = false;
    double latitude_deg = 0.0;
    double longitude_deg = 0.0;
    double altitude_km = 100.0;

    // Look-at target. FIXED (default) uses the look_at XYZ point directly.
    // LATLON computes the surface point from (lat, lon) each frame.
    // LIMB_SUN / LIMB_PRIMARY track the point on the visible limb closest to the
    // Sun / Earth (computed per frame from SPICE positions).
    enum class LookAtMode { FIXED, LATLON, LIMB_SUN, LIMB_PRIMARY };
    LookAtMode look_at_mode = LookAtMode::FIXED;

    std::array<double, 3> look_at{0.0, 0.0, 0.0};
    bool use_direction = false;  // If true, look_at is treated as a direction vector

    // For LATLON mode: surface point at this lat/lon (alt = 0).
    double look_at_lat_deg = 0.0;
    double look_at_lon_deg = 0.0;

    // Up vector hint. In EXPLICIT mode (default) up_vector is used as-is for the
    // LOOK_AT Gram-Schmidt.
    //
    // VELOCITY_POLE: up_vector recomputed per frame as the body-fixed axis
    //   perpendicular to the spacecraft velocity and closest to the body
    //   spin-axis positive direction (north pole, = body-fixed +Z by IAU/SPICE
    //   convention): up = normalize(pole - (pole.v_hat) v_hat).
    //
    // BORESIGHT_POLE (EMA default): the pole projected off the camera BORESIGHT
    //   rather than the velocity: up = normalize(pole - (pole.b_hat) b_hat),
    //   where b_hat is the look direction (camera_pos -> look_at). This keeps
    //   celestial north exactly in the image plane, independent of the flyby
    //   velocity. Falls back to a fixed body axis when the pole is within 5 deg
    //   of the boresight, forces north-up sign (up.pole > 0), and preserves a
    //   right-handed (right, up, forward) frame so the image is not mirrored.
    //   See CameraSimulator::renderFrame.
    enum class UpMode { EXPLICIT, VELOCITY_POLE, BORESIGHT_POLE };
    UpMode up_mode = UpMode::EXPLICIT;
    std::array<double, 3> up_vector{0.0, 0.0, 1.0};

    // Optics — pinhole geometry
    double focal_length_mm = 50.0;
    double sensor_width_mm = 36.0;   // 35mm full-frame default
    double sensor_height_mm = 24.0;
    int nx = 1024;
    int ny = 1024;
    double fov_x_deg = -1.0;  // If > 0, overrides focal_length

    // Realistic-camera extensions. All default to pinhole behavior (zero
    // distortion, principal point at image center, square pixels) so existing
    // configurations are bit-identical.
    //
    // Principal point (pixel coordinates, 0-indexed, sensor top-left origin).
    // < 0 means "use image center" ((nx-1)/2, (ny-1)/2).
    double cx = -1.0;
    double cy = -1.0;

    // Anisotropic pixel pitch (mm/pixel). If BOTH > 0, these override the
    // sensor_width_mm / sensor_height_mm derivation and allow non-square pixels.
    double pixel_pitch_x_mm = -1.0;
    double pixel_pitch_y_mm = -1.0;

    // Brown-Conrady distortion (OpenCV convention). All zero = pure pinhole.
    //   Forward model (ideal → distorted normalized coords):
    //     x_d = x*(1 + k1*r² + k2*r⁴ + k3*r⁶) + 2*p1*x*y + p2*(r² + 2x²)
    //     y_d = y*(1 + k1*r² + k2*r⁴ + k3*r⁶) + p1*(r² + 2y²) + 2*p2*x*y
    // The ray generator uses the inverse (iterative fixed-point).
    double k1 = 0.0, k2 = 0.0, k3 = 0.0;   // radial
    double p1 = 0.0, p2 = 0.0;              // tangential

    // Orientation source. Controls how R_cam→world is built in configure().
    // LOOK_AT is the legacy/default path.
    enum class Pointing { LOOK_AT, QUATERNION, ROTATION_MATRIX, SPICE_CK };
    Pointing pointing = Pointing::LOOK_AT;

    // Quaternion (w, x, y, z) mapping camera frame → body-fixed frame.
    // Used only when pointing == QUATERNION.
    std::array<double, 4> quaternion{1.0, 0.0, 0.0, 0.0};

    // Explicit 3x3 rotation matrix R_cam→world (row-major).
    // Used only when pointing == ROTATION_MATRIX.
    std::array<std::array<double, 3>, 3> rotation_matrix{{
        {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}
    }};

    // SPICE IK mode
    std::string instrument_name;
    std::string spacecraft_name;
    std::string ik_kernel;
    std::string spk_kernel;
    std::string ck_kernel;
};

class CameraModel {
public:
    CameraModel() = default;

    // Configure from generic parameters
    void configure(const CameraConfig& config, double body_radius_km = 0.0);

    // Configure from SPICE instrument kernel at a given ephemeris time
    bool configureFromSPICE(const CameraConfig& config, double et,
                            const std::string& target_body,
                            const std::string& target_frame);

    // Generate a single ray for pixel (px, py), 0-indexed, center of pixel
    // Returns ray origin and direction in body-fixed frame (km)
    struct Ray {
        std::array<double, 3> origin;
        std::array<double, 3> direction;
    };
    Ray generateRay(int px, int py) const;

    // Generate a ray for pixel (px, py) with a sub-pixel offset.
    // subpx, subpy should be in [-0.5, +0.5] for samples inside the pixel.
    // Used for N×N supersampling (anti-aliasing).
    Ray generateRaySubpixel(int px, int py, double subpx, double subpy) const;

    // Generate all rays (row-major: py * nx + px)
    void generateAllRays(std::vector<std::array<double, 3>>& origins,
                         std::vector<std::array<double, 3>>& directions) const;

    // Convert lat/lon/alt to body-fixed XYZ (km). Public so that
    // CameraSimulator::renderFrame() can use it for dynamic look-at.
    static std::array<double, 3> latlonaltToXYZ(double lat_deg, double lon_deg,
                                                 double alt_km, double body_radius_km);

    int width() const { return nx_; }
    int height() const { return ny_; }
    const std::array<double, 3>& position() const { return position_; }
    const std::array<double, 3>& forward() const { return forward_; }
    double fovXDeg() const { return fov_x_deg_; }
    double fovYDeg() const { return fov_y_deg_; }

    // Principal point (1-indexed FITS convention) and focal length in pixels.
    // Exposed for FITS WCS writing.
    double principalPointX() const { return cx_; }
    double principalPointY() const { return cy_; }
    double focalLengthPxX()  const { return f_x_px_; }
    double focalLengthPxY()  const { return f_y_px_; }
    double pixelPitchXMm()   const { return pixel_pitch_x_mm_; }
    double pixelPitchYMm()   const { return pixel_pitch_y_mm_; }
    double focalLengthMm()   const { return focal_length_mm_; }

    // True iff any distortion coefficient is non-zero.
    bool hasDistortion() const {
        return k1_ != 0.0 || k2_ != 0.0 || k3_ != 0.0 || p1_ != 0.0 || p2_ != 0.0;
    }
    double k1() const { return k1_; }
    double k2() const { return k2_; }
    double k3() const { return k3_; }
    double p1() const { return p1_; }
    double p2() const { return p2_; }

    // SPICE-IK metadata (populated after configureFromSPICE; empty otherwise).
    const std::string& instrumentFrame() const { return instrument_frame_; }
    int instrumentId() const { return instrument_naif_id_; }

private:
    std::array<double, 3> position_{0, 0, 0};
    std::array<double, 3> forward_{0, 0, -1};
    std::array<double, 3> right_{1, 0, 0};
    std::array<double, 3> up_{0, 1, 0};
    int nx_ = 1024;
    int ny_ = 1024;
    double fov_x_rad_ = 0.0;
    double fov_y_rad_ = 0.0;
    double fov_x_deg_ = 0.0;
    double fov_y_deg_ = 0.0;
    double half_width_tan_ = 0.0;
    double half_height_tan_ = 0.0;

    // Cached realistic-camera intrinsics, populated in configure().
    double focal_length_mm_ = 50.0;
    double pixel_pitch_x_mm_ = 0.0;
    double pixel_pitch_y_mm_ = 0.0;
    double f_x_px_ = 0.0;   // focal length in pixels (x axis)
    double f_y_px_ = 0.0;
    double cx_ = 0.0;       // principal point (pixels, 0-indexed)
    double cy_ = 0.0;
    double k1_ = 0.0, k2_ = 0.0, k3_ = 0.0;
    double p1_ = 0.0, p2_ = 0.0;
    bool has_distortion_ = false;

    // SPICE-IK metadata
    std::string instrument_frame_;
    int instrument_naif_id_ = 0;

    // Helper: invert Brown-Conrady forward model (distorted → ideal normalized
    // coords) via 5-iteration fixed-point. Matches OpenCV undistortPoints.
    void undistortNormalized(double x_d, double y_d,
                             double& x_u, double& y_u) const;

    // Helper: build (right_, up_, forward_) from a quaternion (w, x, y, z) or
    // from an explicit rotation matrix (camera frame → body-fixed frame).
    void setAxesFromQuaternion(const std::array<double, 4>& q);
    void setAxesFromRotationMatrix(const std::array<std::array<double, 3>, 3>& R);
};

// Vector math utilities
namespace vec {
    inline double dot(const std::array<double, 3>& a, const std::array<double, 3>& b) {
        return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
    }
    inline double length(const std::array<double, 3>& v) {
        return std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    }
    inline std::array<double, 3> normalize(const std::array<double, 3>& v) {
        double L = length(v);
        if (L <= 0.0) return {0, 0, 1};
        return {v[0]/L, v[1]/L, v[2]/L};
    }
    inline std::array<double, 3> cross(const std::array<double, 3>& a, const std::array<double, 3>& b) {
        return { a[1]*b[2] - a[2]*b[1],
                 a[2]*b[0] - a[0]*b[2],
                 a[0]*b[1] - a[1]*b[0] };
    }
    inline std::array<double, 3> sub(const std::array<double, 3>& a, const std::array<double, 3>& b) {
        return {a[0]-b[0], a[1]-b[1], a[2]-b[2]};
    }
    inline std::array<double, 3> add(const std::array<double, 3>& a, const std::array<double, 3>& b) {
        return {a[0]+b[0], a[1]+b[1], a[2]+b[2]};
    }
    inline std::array<double, 3> scale(const std::array<double, 3>& v, double s) {
        return {v[0]*s, v[1]*s, v[2]*s};
    }
} // namespace vec

} // namespace ptm
