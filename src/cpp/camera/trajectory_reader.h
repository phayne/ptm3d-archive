#pragma once

#include <string>
#include <vector>
#include <array>

namespace ptm {

struct TrajectoryPoint {
    double et;                        // SPICE ephemeris time (seconds past J2000)
    std::array<double, 3> position;   // Body-fixed XYZ (km)
    std::array<double, 3> velocity;   // Body-fixed XYZ (km/s)
};

class TrajectoryReader {
public:
    TrajectoryReader() = default;

    // Load SPICE SPK for spacecraft
    // spacecraft_id: NAIF ID string (e.g., "-1024" for Artemis II)
    // target_body: target body name (e.g., "MOON")
    // target_frame: body-fixed frame (e.g., "IAU_MOON")
    bool loadSPK(const std::string& spk_file,
                 const std::string& spacecraft_id,
                 const std::string& target_body = "MOON",
                 const std::string& target_frame = "IAU_MOON");

    // Load HORIZONS vector table (text format)
    // Expects columns: JDTDB, Calendar_Date, X, Y, Z, VX, VY, VZ
    // or simple: ET, X, Y, Z, VX, VY, VZ (km, km/s, in body-fixed frame)
    bool loadHORIZONS(const std::string& filename,
                      const std::string& target_body = "MOON",
                      const std::string& target_frame = "IAU_MOON");

    // Load NASA PROP_MAN format
    // Columns: MET(sec), X(ft), Y(ft), Z(ft), VX(fps), VY(fps), VZ(fps), mass(lbm)
    // Coordinates in M50 (J2000) inertial frame
    bool loadPropMan(const std::string& filename,
                     const std::string& target_body = "MOON",
                     const std::string& target_frame = "IAU_MOON",
                     const std::string& epoch_utc = "");

    // Get position at a given ET (Hermite interpolation)
    std::array<double, 3> getPosition(double et) const;

    // Get velocity at a given ET
    std::array<double, 3> getVelocity(double et) const;

    // Get all loaded trajectory points
    const std::vector<TrajectoryPoint>& getTrajectory() const { return points_; }

    // Time range
    double startET() const;
    double endET() const;
    bool isLoaded() const { return !points_.empty(); }

    // SPICE mode: query position directly via spkpos_c
    bool usesSPICE() const { return use_spice_; }
    std::array<double, 3> getPositionSPICE(double et) const;

private:
    std::vector<TrajectoryPoint> points_;
    bool use_spice_ = false;
    std::string spice_spacecraft_id_;
    std::string spice_target_body_;
    std::string spice_target_frame_;

    // Unit conversions
    static constexpr double FEET_TO_KM = 0.0003048;
    static constexpr double FPS_TO_KMS = 0.0003048;

    // Hermite interpolation between trajectory points
    std::pair<int, int> findBracketingPoints(double et) const;
};

} // namespace ptm
