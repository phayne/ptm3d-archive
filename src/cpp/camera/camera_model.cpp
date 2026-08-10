#include "camera_model.h"

#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>

// SPICE (for configureFromSPICE)
extern "C" {
#include "SpiceUsr.h"
}

namespace ptm {

std::array<double, 3> CameraModel::latlonaltToXYZ(double lat_deg, double lon_deg,
                                                    double alt_km, double body_radius_km) {
    constexpr double DEG_TO_RAD = M_PI / 180.0;
    double lat = lat_deg * DEG_TO_RAD;
    double lon = lon_deg * DEG_TO_RAD;
    double r = body_radius_km + alt_km;
    return {
        r * std::cos(lat) * std::cos(lon),
        r * std::cos(lat) * std::sin(lon),
        r * std::sin(lat)
    };
}

void CameraModel::configure(const CameraConfig& config, double body_radius_km) {
    nx_ = config.nx;
    ny_ = config.ny;

    // ------------------------------------------------------------------
    // Position (body-fixed frame, mesh units as passed in)
    // ------------------------------------------------------------------
    if (config.use_latlon) {
        position_ = latlonaltToXYZ(config.latitude_deg, config.longitude_deg,
                                   config.altitude_km, body_radius_km);
    } else {
        position_ = config.position;
    }

    // ------------------------------------------------------------------
    // Orientation (R_cam→world via forward_/right_/up_)
    // Camera convention: +Z boresight (forward), +X right, +Y up.
    // ------------------------------------------------------------------
    switch (config.pointing) {
        case CameraConfig::Pointing::QUATERNION:
            setAxesFromQuaternion(config.quaternion);
            break;

        case CameraConfig::Pointing::ROTATION_MATRIX:
            setAxesFromRotationMatrix(config.rotation_matrix);
            break;

        case CameraConfig::Pointing::SPICE_CK:
            // SPICE-driven orientation is applied per frame via configureFromSPICE;
            // during the LOOK_AT-less bootstrap fall back to identity and rely on
            // the simulator to call configureFromSPICE before rendering.
            if (config.rotation_matrix != std::array<std::array<double, 3>, 3>{{
                    {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0} }}) {
                setAxesFromRotationMatrix(config.rotation_matrix);
            } else {
                forward_ = {0.0, 0.0, 1.0};
                right_   = {1.0, 0.0, 0.0};
                up_      = {0.0, 1.0, 0.0};
            }
            break;

        case CameraConfig::Pointing::LOOK_AT:
        default: {
            if (config.use_direction) {
                forward_ = vec::normalize(config.look_at);
            } else {
                forward_ = vec::normalize(vec::sub(config.look_at, position_));
            }
            // Gram-Schmidt: right = forward x up, then recompute up
            right_ = vec::normalize(vec::cross(forward_, config.up_vector));
            if (vec::length(right_) < 1e-10) {
                // Degenerate case: forward parallel to up. Pick an alt up.
                std::array<double, 3> alt_up = {0, 1, 0};
                if (std::abs(vec::dot(forward_, alt_up)) > 0.99) {
                    alt_up = {1, 0, 0};
                }
                right_ = vec::normalize(vec::cross(forward_, alt_up));
            }
            up_ = vec::cross(right_, forward_);
            break;
        }
    }

    // ------------------------------------------------------------------
    // Intrinsics. Supports three parameterizations:
    //   (a) explicit FOV (fov_x_deg > 0) — sensor dimensions are reconstructed
    //       from FOV + focal length for pixel-pitch computation.
    //   (b) anisotropic pixel pitch (pixel_pitch_x_mm and _y_mm both > 0).
    //   (c) sensor_width_mm / sensor_height_mm (legacy default).
    // In all cases we compute focal length in pixels along each axis and cache
    // the principal point.
    // ------------------------------------------------------------------
    focal_length_mm_ = config.focal_length_mm;

    double sensor_w_mm = config.sensor_width_mm;
    double sensor_h_mm = config.sensor_height_mm;

    if (config.pixel_pitch_x_mm > 0.0 && config.pixel_pitch_y_mm > 0.0) {
        pixel_pitch_x_mm_ = config.pixel_pitch_x_mm;
        pixel_pitch_y_mm_ = config.pixel_pitch_y_mm;
        sensor_w_mm = pixel_pitch_x_mm_ * nx_;
        sensor_h_mm = pixel_pitch_y_mm_ * ny_;
    } else {
        pixel_pitch_x_mm_ = sensor_w_mm / std::max(1, nx_);
        pixel_pitch_y_mm_ = sensor_h_mm / std::max(1, ny_);
    }

    if (config.fov_x_deg > 0.0) {
        fov_x_rad_ = config.fov_x_deg * M_PI / 180.0;
        // Re-derive sensor width from FOV so pixel pitch is consistent.
        sensor_w_mm = 2.0 * focal_length_mm_ * std::tan(fov_x_rad_ / 2.0);
        pixel_pitch_x_mm_ = sensor_w_mm / std::max(1, nx_);
        // Keep pixel_pitch_y aspect-proportional unless user supplied it.
        if (!(config.pixel_pitch_y_mm > 0.0)) {
            pixel_pitch_y_mm_ = pixel_pitch_x_mm_;  // assume square pixels
            sensor_h_mm = pixel_pitch_y_mm_ * ny_;
        }
    } else {
        fov_x_rad_ = 2.0 * std::atan(sensor_w_mm / (2.0 * focal_length_mm_));
    }
    fov_y_rad_ = 2.0 * std::atan(sensor_h_mm / (2.0 * focal_length_mm_));

    fov_x_deg_ = fov_x_rad_ * 180.0 / M_PI;
    fov_y_deg_ = fov_y_rad_ * 180.0 / M_PI;
    half_width_tan_  = std::tan(fov_x_rad_ / 2.0);
    half_height_tan_ = std::tan(fov_y_rad_ / 2.0);

    f_x_px_ = focal_length_mm_ / pixel_pitch_x_mm_;
    f_y_px_ = focal_length_mm_ / pixel_pitch_y_mm_;

    // Default principal point is at the gridline between the central pixels
    // (nx/2, ny/2). This keeps legacy pinhole rays bit-identical with the old
    // u = (2*(px+0.5)/nx - 1)*tan(fov/2) formulation: at px = nx/2-1 the ray
    // direction is u = -1/nx * tan, and with the new formula (0.5 - nx/2)/f_x
    // yields the same value because f_x = nx/(2*tan(fov/2)).
    cx_ = (config.cx < 0.0) ? 0.5 * nx_ : config.cx;
    cy_ = (config.cy < 0.0) ? 0.5 * ny_ : config.cy;

    // Distortion coefficients (Brown-Conrady). Caching allows the hot path to
    // short-circuit for pinhole cameras.
    k1_ = config.k1; k2_ = config.k2; k3_ = config.k3;
    p1_ = config.p1; p2_ = config.p2;
    has_distortion_ = hasDistortion();

    std::cout << "[CameraModel] Position: (" << std::fixed << std::setprecision(1)
              << position_[0] << ", " << position_[1] << ", " << position_[2] << ")" << std::endl;
    std::cout << "[CameraModel] Forward: (" << std::setprecision(4)
              << forward_[0] << ", " << forward_[1] << ", " << forward_[2] << ")" << std::endl;
    std::cout << "[CameraModel] FOV: " << std::setprecision(2) << fov_x_deg_ << " x " << fov_y_deg_
              << " deg, " << nx_ << " x " << ny_ << " pixels" << std::endl;
    std::cout << "[CameraModel] Principal point: (" << std::setprecision(2)
              << cx_ << ", " << cy_ << ") px; focal length "
              << std::setprecision(3) << focal_length_mm_ << " mm, pitch ("
              << pixel_pitch_x_mm_ << ", " << pixel_pitch_y_mm_ << ") mm/px"
              << std::endl;
    if (has_distortion_) {
        std::cout << "[CameraModel] Brown-Conrady distortion: k=("
                  << k1_ << ", " << k2_ << ", " << k3_ << "), p=("
                  << p1_ << ", " << p2_ << ")" << std::endl;
    }
}

void CameraModel::setAxesFromRotationMatrix(
        const std::array<std::array<double, 3>, 3>& R) {
    // Columns of R_cam→world are the camera basis vectors expressed in world.
    // Convention: camera +X = right, +Y = up, +Z = boresight (forward).
    right_   = { R[0][0], R[1][0], R[2][0] };
    up_      = { R[0][1], R[1][1], R[2][1] };
    forward_ = { R[0][2], R[1][2], R[2][2] };
    right_   = vec::normalize(right_);
    up_      = vec::normalize(up_);
    forward_ = vec::normalize(forward_);
}

void CameraModel::setAxesFromQuaternion(const std::array<double, 4>& q) {
    // q = (w, x, y, z). Normalize and build the 3x3 rotation matrix, then
    // delegate.
    double w = q[0], x = q[1], y = q[2], z = q[3];
    double n = std::sqrt(w*w + x*x + y*y + z*z);
    if (n <= 0.0) { w = 1.0; x = y = z = 0.0; n = 1.0; }
    w /= n; x /= n; y /= n; z /= n;

    std::array<std::array<double, 3>, 3> R{{
        { 1 - 2*(y*y + z*z),  2*(x*y - z*w),     2*(x*z + y*w)     },
        { 2*(x*y + z*w),      1 - 2*(x*x + z*z), 2*(y*z - x*w)     },
        { 2*(x*z - y*w),      2*(y*z + x*w),     1 - 2*(x*x + y*y) }
    }};
    setAxesFromRotationMatrix(R);
}

void CameraModel::undistortNormalized(double x_d, double y_d,
                                       double& x_u, double& y_u) const {
    // 5-iteration fixed-point inversion of the Brown-Conrady forward model.
    // Matches OpenCV's undistortPoints.
    x_u = x_d;
    y_u = y_d;
    for (int iter = 0; iter < 5; ++iter) {
        const double r2 = x_u * x_u + y_u * y_u;
        const double r4 = r2 * r2;
        const double r6 = r4 * r2;
        const double radial = 1.0 + k1_ * r2 + k2_ * r4 + k3_ * r6;
        const double dx_tan = 2.0 * p1_ * x_u * y_u + p2_ * (r2 + 2.0 * x_u * x_u);
        const double dy_tan = p1_ * (r2 + 2.0 * y_u * y_u) + 2.0 * p2_ * x_u * y_u;
        // forward(x_u, y_u) = radial * (x_u, y_u) + (dx_tan, dy_tan) should equal (x_d, y_d)
        // ⇒ solve for (x_u, y_u): (x_u, y_u) = ((x_d - dx_tan), (y_d - dy_tan)) / radial
        if (radial == 0.0) break;
        x_u = (x_d - dx_tan) / radial;
        y_u = (y_d - dy_tan) / radial;
    }
}

bool CameraModel::configureFromSPICE(const CameraConfig& config, double et,
                                      const std::string& /*target_body*/,
                                      const std::string& target_frame) {
    if (config.instrument_name.empty()) {
        std::cerr << "[CameraModel] configureFromSPICE: instrument_name is empty"
                  << std::endl;
        return false;
    }

    // --- Resolve instrument NAIF ID from name ---
    SpiceInt inst_id = 0;
    SpiceBoolean found = SPICEFALSE;
    bodn2c_c(config.instrument_name.c_str(), &inst_id, &found);
    if (!found) {
        std::cerr << "[CameraModel] Unknown instrument name: "
                  << config.instrument_name
                  << " (check IK / FK kernels are loaded)" << std::endl;
        return false;
    }
    instrument_naif_id_ = static_cast<int>(inst_id);

    // --- Query FOV geometry from IK ---
    constexpr SpiceInt ROOM = 10;
    SpiceChar shape[32];
    SpiceChar frame[64];
    SpiceDouble bsight[3];
    SpiceInt n_bounds = 0;
    SpiceDouble bounds[10][3];

    getfov_c(inst_id, ROOM, sizeof(shape), sizeof(frame),
             shape, frame, bsight, &n_bounds, bounds);
    if (failed_c()) {
        char msg[1024];
        getmsg_c("LONG", sizeof(msg), msg);
        std::cerr << "[CameraModel] getfov_c failed: " << msg << std::endl;
        reset_c();
        return false;
    }
    instrument_frame_ = frame;

    // --- Derive half-FOV along image x/y axes from corner bounds ---
    // Bounds are unit vectors in the instrument frame, pointing to FOV corners.
    // Project each onto (X_cam=(1,0,0)) and (Y_cam=(0,1,0)) and find max |angle|.
    // For a RECTANGLE shape this gives the full half-FOV; other shapes fall back
    // to circumscribing rectangle.
    double max_tan_x = 0.0, max_tan_y = 0.0;
    for (SpiceInt i = 0; i < n_bounds; ++i) {
        const double bx = bounds[i][0];
        const double by = bounds[i][1];
        const double bz = bounds[i][2];
        if (std::abs(bz) < 1e-12) continue;
        const double tx = std::abs(bx / bz);
        const double ty = std::abs(by / bz);
        if (tx > max_tan_x) max_tan_x = tx;
        if (ty > max_tan_y) max_tan_y = ty;
    }
    if (max_tan_x > 0.0) {
        fov_x_rad_ = 2.0 * std::atan(max_tan_x);
        half_width_tan_ = max_tan_x;
    }
    if (max_tan_y > 0.0) {
        fov_y_rad_ = 2.0 * std::atan(max_tan_y);
        half_height_tan_ = max_tan_y;
    }
    fov_x_deg_ = fov_x_rad_ * 180.0 / M_PI;
    fov_y_deg_ = fov_y_rad_ * 180.0 / M_PI;

    // Re-derive pixel pitches from IK-provided FOV so intrinsics stay consistent.
    if (fov_x_rad_ > 0.0 && focal_length_mm_ > 0.0) {
        const double sensor_w_mm = 2.0 * focal_length_mm_ * std::tan(fov_x_rad_ / 2.0);
        pixel_pitch_x_mm_ = sensor_w_mm / std::max(1, nx_);
        f_x_px_ = focal_length_mm_ / pixel_pitch_x_mm_;
    }
    if (fov_y_rad_ > 0.0 && focal_length_mm_ > 0.0) {
        const double sensor_h_mm = 2.0 * focal_length_mm_ * std::tan(fov_y_rad_ / 2.0);
        pixel_pitch_y_mm_ = sensor_h_mm / std::max(1, ny_);
        f_y_px_ = focal_length_mm_ / pixel_pitch_y_mm_;
    }

    // --- Per-frame orientation: R_inst→target at ET ---
    SpiceDouble rot[3][3];
    pxform_c(frame, target_frame.c_str(), et, rot);
    if (failed_c()) {
        char msg[1024];
        getmsg_c("LONG", sizeof(msg), msg);
        std::cerr << "[CameraModel] pxform_c(" << frame << " → "
                  << target_frame << ") failed: " << msg << std::endl;
        reset_c();
        return false;
    }

    // rot is row-major 3x3 rotating a vector from instrument frame to target
    // frame. Apply to the instrument-frame axes (+X right, +Y up, boresight).
    auto apply = [&rot](const std::array<double, 3>& v) -> std::array<double, 3> {
        return {
            rot[0][0]*v[0] + rot[0][1]*v[1] + rot[0][2]*v[2],
            rot[1][0]*v[0] + rot[1][1]*v[1] + rot[1][2]*v[2],
            rot[2][0]*v[0] + rot[2][1]*v[1] + rot[2][2]*v[2]
        };
    };
    const std::array<double, 3> x_cam = {1.0, 0.0, 0.0};
    const std::array<double, 3> y_cam = {0.0, 1.0, 0.0};
    const std::array<double, 3> bs    = {bsight[0], bsight[1], bsight[2]};

    right_   = vec::normalize(apply(x_cam));
    up_      = vec::normalize(apply(y_cam));
    forward_ = vec::normalize(apply(bs));

    return true;
}

CameraModel::Ray CameraModel::generateRay(int px, int py) const {
    // Delegates to the sub-pixel variant at the pixel center.
    return generateRaySubpixel(px, py, 0.0, 0.0);
}

CameraModel::Ray CameraModel::generateRaySubpixel(int px, int py,
                                                   double subpx, double subpy) const {
    // Pixel coordinates including sub-pixel jitter. (subpx, subpy) in [-0.5, +0.5].
    const double x_px = px + 0.5 + subpx;
    const double y_px = py + 0.5 + subpy;

    // Normalized sensor coordinates (distorted): y axis flipped so image +y
    // corresponds to world +up (image top-left origin, FITS convention).
    const double x_d = (x_px - cx_) / f_x_px_;
    const double y_d = -((y_px - cy_) / f_y_px_);

    // Recover ideal (undistorted) normalized coords. Pinhole fast path skips
    // the inversion loop entirely when no distortion is configured.
    double x_u, y_u;
    if (has_distortion_) {
        undistortNormalized(x_d, y_d, x_u, y_u);
    } else {
        x_u = x_d;
        y_u = y_d;
    }

    // World-space ray direction: dir = x_u * right + y_u * up + forward
    // (camera frame: +Z boresight, +X right, +Y up).
    std::array<double, 3> dir = {
        forward_[0] + x_u * right_[0] + y_u * up_[0],
        forward_[1] + x_u * right_[1] + y_u * up_[1],
        forward_[2] + x_u * right_[2] + y_u * up_[2]
    };
    dir = vec::normalize(dir);

    return {position_, dir};
}

void CameraModel::generateAllRays(std::vector<std::array<double, 3>>& origins,
                                   std::vector<std::array<double, 3>>& directions) const {
    const size_t n = static_cast<size_t>(nx_) * ny_;
    origins.resize(n);
    directions.resize(n);

    #pragma omp parallel for schedule(static)
    for (int py = 0; py < ny_; ++py) {
        for (int px = 0; px < nx_; ++px) {
            size_t idx = static_cast<size_t>(py) * nx_ + px;
            Ray ray = generateRay(px, py);
            origins[idx] = ray.origin;
            directions[idx] = ray.direction;
        }
    }
}

} // namespace ptm
