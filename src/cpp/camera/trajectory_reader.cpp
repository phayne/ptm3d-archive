#include "trajectory_reader.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cmath>

// SPICE headers
extern "C" {
#include "SpiceUsr.h"
}

namespace ptm {

// ============================================================================
// SPICE SPK loading
// ============================================================================

bool TrajectoryReader::loadSPK(const std::string& spk_file,
                                const std::string& spacecraft_id,
                                const std::string& target_body,
                                const std::string& target_frame) {
    // Load the SPK kernel
    furnsh_c(spk_file.c_str());

    // Check for SPICE errors
    if (failed_c()) {
        char msg[1024];
        getmsg_c("LONG", sizeof(msg), msg);
        std::cerr << "[TrajectoryReader] SPICE error loading " << spk_file << ": " << msg << std::endl;
        reset_c();
        return false;
    }

    use_spice_ = true;
    spice_spacecraft_id_ = spacecraft_id;
    spice_target_body_ = target_body;
    spice_target_frame_ = target_frame;

    std::cout << "[TrajectoryReader] Loaded SPK: " << spk_file
              << " (spacecraft=" << spacecraft_id << ")" << std::endl;
    return true;
}

std::array<double, 3> TrajectoryReader::getPositionSPICE(double et) const {
    SpiceDouble state[6];
    SpiceDouble lt;

    spkpos_c(spice_spacecraft_id_.c_str(), et,
             spice_target_frame_.c_str(), "NONE",
             spice_target_body_.c_str(), state, &lt);

    if (failed_c()) {
        char msg[1024];
        getmsg_c("LONG", sizeof(msg), msg);
        std::cerr << "[TrajectoryReader] SPICE spkpos_c error: " << msg << std::endl;
        reset_c();
        return {0, 0, 0};
    }

    return {state[0], state[1], state[2]};
}

// ============================================================================
// HORIZONS vector table loading
// ============================================================================

bool TrajectoryReader::loadHORIZONS(const std::string& filename,
                                     const std::string& /*target_body*/,
                                     const std::string& /*target_frame*/) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "[TrajectoryReader] Cannot open HORIZONS file: " << filename << std::endl;
        return false;
    }

    points_.clear();
    bool in_data = false;
    std::string line;

    while (std::getline(file, line)) {
        // HORIZONS uses $$SOE and $$EOE markers
        if (line.find("$$SOE") != std::string::npos) {
            in_data = true;
            continue;
        }
        if (line.find("$$EOE") != std::string::npos) {
            break;
        }

        if (!in_data) continue;
        if (line.empty()) continue;

        // Parse HORIZONS vector table format:
        // JDTDB, Calendar_Date, X, Y, Z, VX, VY, VZ, LT, RG, RR
        // We need to handle the multi-line format where each record spans 2-4 lines

        // Simple format: ET X Y Z VX VY VZ (one line per record, already in body-fixed km)
        std::istringstream iss(line);
        double et, x, y, z, vx, vy, vz;
        if (iss >> et >> x >> y >> z >> vx >> vy >> vz) {
            TrajectoryPoint pt;
            pt.et = et;
            pt.position = {x, y, z};
            pt.velocity = {vx, vy, vz};
            points_.push_back(pt);
        }
    }

    if (points_.empty()) {
        std::cerr << "[TrajectoryReader] No data points found in " << filename << std::endl;
        return false;
    }

    // Sort by ET
    std::sort(points_.begin(), points_.end(),
              [](const TrajectoryPoint& a, const TrajectoryPoint& b) { return a.et < b.et; });

    std::cout << "[TrajectoryReader] Loaded " << points_.size() << " HORIZONS points, ET range: ["
              << points_.front().et << ", " << points_.back().et << "]" << std::endl;
    return true;
}

// ============================================================================
// PROP_MAN format loading
// ============================================================================

bool TrajectoryReader::loadPropMan(const std::string& filename,
                                    const std::string& target_body,
                                    const std::string& target_frame,
                                    const std::string& epoch_utc) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "[TrajectoryReader] Cannot open PROP_MAN file: " << filename << std::endl;
        return false;
    }

    points_.clear();

    // Parse header line
    std::string header;
    std::getline(file, header);  // "PROP_MAN 11.0"

    // Parse epoch line: year, epoch_sec, ?, duration, frame, units...
    std::string epoch_line;
    std::getline(file, epoch_line);
    std::istringstream epoch_iss(epoch_line);
    int year;
    double epoch_sec;
    epoch_iss >> year >> epoch_sec;

    // Convert epoch: seconds from start of year to ET
    // Build epoch string: "YYYY-01-01T00:00:00"
    std::string epoch_str;
    if (!epoch_utc.empty()) {
        epoch_str = epoch_utc;
    } else {
        std::ostringstream oss;
        oss << year << "-01-01T00:00:00";
        epoch_str = oss.str();
    }

    SpiceDouble epoch_et;
    str2et_c(epoch_str.c_str(), &epoch_et);
    if (failed_c()) {
        char msg[1024];
        getmsg_c("LONG", sizeof(msg), msg);
        std::cerr << "[TrajectoryReader] SPICE str2et_c error for epoch: " << msg << std::endl;
        reset_c();
        return false;
    }

    // Build target body frame string for rotation
    std::string frame = target_frame.empty() ? ("IAU_" + target_body) : target_frame;

    // Parse data lines: MET(sec), X(ft), Y(ft), Z(ft), VX(fps), VY(fps), VZ(fps), mass(lbm)
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::istringstream iss(line);
        double met, x_ft, y_ft, z_ft, vx_fps, vy_fps, vz_fps, mass;
        if (!(iss >> met >> x_ft >> y_ft >> z_ft >> vx_fps >> vy_fps >> vz_fps >> mass)) continue;

        double et = epoch_et + met;

        // Convert from feet to km (M50/J2000 inertial frame)
        double x_km = x_ft * FEET_TO_KM;
        double y_km = y_ft * FEET_TO_KM;
        double z_km = z_ft * FEET_TO_KM;
        double vx_kms = vx_fps * FPS_TO_KMS;
        double vy_kms = vy_fps * FPS_TO_KMS;
        double vz_kms = vz_fps * FPS_TO_KMS;

        // Rotate from J2000 to body-fixed frame using SPICE
        SpiceDouble rot[3][3];
        pxform_c("J2000", frame.c_str(), et, rot);

        if (failed_c()) {
            reset_c();
            continue;
        }

        // Apply rotation
        TrajectoryPoint pt;
        pt.et = et;
        pt.position = {
            rot[0][0]*x_km + rot[0][1]*y_km + rot[0][2]*z_km,
            rot[1][0]*x_km + rot[1][1]*y_km + rot[1][2]*z_km,
            rot[2][0]*x_km + rot[2][1]*y_km + rot[2][2]*z_km
        };
        pt.velocity = {
            rot[0][0]*vx_kms + rot[0][1]*vy_kms + rot[0][2]*vz_kms,
            rot[1][0]*vx_kms + rot[1][1]*vy_kms + rot[1][2]*vz_kms,
            rot[2][0]*vx_kms + rot[2][1]*vy_kms + rot[2][2]*vz_kms
        };
        points_.push_back(pt);
    }

    if (points_.empty()) {
        std::cerr << "[TrajectoryReader] No data points parsed from " << filename << std::endl;
        return false;
    }

    std::sort(points_.begin(), points_.end(),
              [](const TrajectoryPoint& a, const TrajectoryPoint& b) { return a.et < b.et; });

    std::cout << "[TrajectoryReader] Loaded " << points_.size()
              << " PROP_MAN points (converted to " << frame << ")" << std::endl;
    return true;
}

// ============================================================================
// Interpolation
// ============================================================================

double TrajectoryReader::startET() const {
    if (use_spice_ || points_.empty()) return 0.0;
    return points_.front().et;
}

double TrajectoryReader::endET() const {
    if (use_spice_ || points_.empty()) return 0.0;
    return points_.back().et;
}

std::pair<int, int> TrajectoryReader::findBracketingPoints(double et) const {
    if (points_.empty()) return {-1, -1};
    if (et <= points_.front().et) return {0, 0};
    if (et >= points_.back().et) {
        int last = static_cast<int>(points_.size()) - 1;
        return {last, last};
    }

    auto it = std::lower_bound(points_.begin(), points_.end(), et,
        [](const TrajectoryPoint& p, double t) { return p.et < t; });
    int upper = static_cast<int>(std::distance(points_.begin(), it));
    return {upper - 1, upper};
}

std::array<double, 3> TrajectoryReader::getPosition(double et) const {
    if (use_spice_) {
        return getPositionSPICE(et);
    }

    auto [lo, hi] = findBracketingPoints(et);
    if (lo < 0) return {0, 0, 0};
    if (lo == hi) return points_[lo].position;

    // Linear interpolation (sufficient for 1-second spacing)
    double t_lo = points_[lo].et;
    double t_hi = points_[hi].et;
    double dt = t_hi - t_lo;
    if (dt <= 0.0) return points_[lo].position;

    double alpha = (et - t_lo) / dt;
    const auto& p0 = points_[lo].position;
    const auto& p1 = points_[hi].position;

    return {
        (1.0 - alpha) * p0[0] + alpha * p1[0],
        (1.0 - alpha) * p0[1] + alpha * p1[1],
        (1.0 - alpha) * p0[2] + alpha * p1[2]
    };
}

std::array<double, 3> TrajectoryReader::getVelocity(double et) const {
    if (use_spice_) {
        // For SPICE mode, compute velocity from finite differences
        double h = 1.0;  // 1 second
        auto p1 = getPositionSPICE(et + h);
        auto p0 = getPositionSPICE(et - h);
        return {(p1[0]-p0[0])/(2*h), (p1[1]-p0[1])/(2*h), (p1[2]-p0[2])/(2*h)};
    }

    auto [lo, hi] = findBracketingPoints(et);
    if (lo < 0) return {0, 0, 0};
    if (lo == hi) return points_[lo].velocity;

    double t_lo = points_[lo].et;
    double t_hi = points_[hi].et;
    double dt = t_hi - t_lo;
    if (dt <= 0.0) return points_[lo].velocity;

    double alpha = (et - t_lo) / dt;
    const auto& v0 = points_[lo].velocity;
    const auto& v1 = points_[hi].velocity;

    return {
        (1.0 - alpha) * v0[0] + alpha * v1[0],
        (1.0 - alpha) * v0[1] + alpha * v1[1],
        (1.0 - alpha) * v0[2] + alpha * v1[2]
    };
}

} // namespace ptm
