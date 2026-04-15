#include "camera_model.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>

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

    // Determine camera position
    if (config.use_latlon) {
        position_ = latlonaltToXYZ(config.latitude_deg, config.longitude_deg,
                                   config.altitude_km, body_radius_km);
    } else {
        position_ = config.position;
    }

    // Compute camera axes
    if (config.use_direction) {
        forward_ = vec::normalize(config.look_at);
    } else {
        forward_ = vec::normalize(vec::sub(config.look_at, position_));
    }

    // Gram-Schmidt: right = forward x up, then recompute up
    right_ = vec::normalize(vec::cross(forward_, config.up_vector));

    // Handle degenerate case: forward parallel to up
    if (vec::length(right_) < 1e-10) {
        // Pick an arbitrary perpendicular
        std::array<double, 3> alt_up = {0, 1, 0};
        if (std::abs(vec::dot(forward_, alt_up)) > 0.99) {
            alt_up = {1, 0, 0};
        }
        right_ = vec::normalize(vec::cross(forward_, alt_up));
    }
    up_ = vec::cross(right_, forward_);

    // Compute FOV
    if (config.fov_x_deg > 0.0) {
        fov_x_rad_ = config.fov_x_deg * M_PI / 180.0;
    } else {
        fov_x_rad_ = 2.0 * std::atan(config.sensor_width_mm / (2.0 * config.focal_length_mm));
    }
    double aspect = static_cast<double>(ny_) / static_cast<double>(nx_);
    fov_y_rad_ = 2.0 * std::atan(std::tan(fov_x_rad_ / 2.0) * aspect);

    fov_x_deg_ = fov_x_rad_ * 180.0 / M_PI;
    fov_y_deg_ = fov_y_rad_ * 180.0 / M_PI;
    half_width_tan_ = std::tan(fov_x_rad_ / 2.0);
    half_height_tan_ = std::tan(fov_y_rad_ / 2.0);

    std::cout << "[CameraModel] Position: (" << std::fixed << std::setprecision(1)
              << position_[0] << ", " << position_[1] << ", " << position_[2] << ")" << std::endl;
    std::cout << "[CameraModel] Forward: (" << std::setprecision(4)
              << forward_[0] << ", " << forward_[1] << ", " << forward_[2] << ")" << std::endl;
    std::cout << "[CameraModel] FOV: " << std::setprecision(2) << fov_x_deg_ << " x " << fov_y_deg_
              << " deg, " << nx_ << " x " << ny_ << " pixels" << std::endl;
}

bool CameraModel::configureFromSPICE(const CameraConfig& /*config*/, double /*et*/,
                                      const std::string& /*target_body*/,
                                      const std::string& /*target_frame*/) {
    // Phase 4: SPICE IK support
    std::cerr << "[CameraModel] SPICE IK mode not yet implemented" << std::endl;
    return false;
}

CameraModel::Ray CameraModel::generateRay(int px, int py) const {
    // Normalized device coordinates: [-1, 1] from left to right, top to bottom
    double u = (2.0 * (px + 0.5) / nx_ - 1.0) * half_width_tan_;
    double v = (1.0 - 2.0 * (py + 0.5) / ny_) * half_height_tan_;

    // Ray direction in world space
    std::array<double, 3> dir = {
        forward_[0] + u * right_[0] + v * up_[0],
        forward_[1] + u * right_[1] + v * up_[1],
        forward_[2] + u * right_[2] + v * up_[2]
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
