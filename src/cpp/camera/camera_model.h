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

    // Look-at point or direction
    std::array<double, 3> look_at{0.0, 0.0, 0.0};
    bool use_direction = false;  // If true, look_at is treated as a direction vector

    // Up vector hint
    std::array<double, 3> up_vector{0.0, 0.0, 1.0};

    // Optics
    double focal_length_mm = 50.0;
    double sensor_width_mm = 36.0;   // 35mm full-frame default
    double sensor_height_mm = 24.0;
    int nx = 1024;
    int ny = 1024;
    double fov_x_deg = -1.0;  // If > 0, overrides focal_length

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

    // Generate all rays (row-major: py * nx + px)
    void generateAllRays(std::vector<std::array<double, 3>>& origins,
                         std::vector<std::array<double, 3>>& directions) const;

    int width() const { return nx_; }
    int height() const { return ny_; }
    const std::array<double, 3>& position() const { return position_; }
    const std::array<double, 3>& forward() const { return forward_; }
    double fovXDeg() const { return fov_x_deg_; }
    double fovYDeg() const { return fov_y_deg_; }

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

    // Convert lat/lon/alt to body-fixed XYZ
    static std::array<double, 3> latlonaltToXYZ(double lat_deg, double lon_deg,
                                                 double alt_km, double body_radius_km);
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
