#pragma once

#include "camera_model.h"
#include "photometric_model.h"
#include "ptm_output_reader.h"
#include "trajectory_reader.h"

#include <embree4/rtcore.h>
#include <string>
#include <vector>
#include <array>

namespace ptm {

struct CameraSimConfig {
    // Camera
    CameraConfig camera;

    // Photometry
    PhotometricModel::ReflectanceType reflectance_type = PhotometricModel::ReflectanceType::LOMMEL_SEELIGER;
    PhotometricModel::Params photometric_params;

    // Render mode
    enum class RenderMode { REFLECTED, THERMAL, BOTH };
    RenderMode render_mode = RenderMode::REFLECTED;

    // PTM3D output
    std::string ptm_output_dir;

    // Render times (UTC strings)
    std::vector<std::string> render_times;

    // Trajectory (optional — overrides static camera position)
    std::string trajectory_file;
    std::string trajectory_format = "spk";  // "spk", "horizons", "propman"
    std::string trajectory_spacecraft = "-1024";
    std::string trajectory_epoch_utc;  // For PROP_MAN

    // SPICE
    std::string kernels_path;
    std::string meta_kernel;
    std::string target_body = "MOON";
    std::string target_frame = "IAU_MOON";

    // Output
    std::string output_dir = "camera_output";
    bool output_fits = true;
    bool output_png = false;
};

struct PixelData {
    float radiance = 0.0f;
    float thermal_radiance = 0.0f;
    int32_t face_index = -1;
    float incidence_angle = 0.0f;  // degrees
    float emission_angle = 0.0f;   // degrees
    float phase_angle = 0.0f;      // degrees
    float temperature = 0.0f;      // K
    float distance_km = 0.0f;
    float latitude = 0.0f;         // degrees
    float longitude = 0.0f;        // degrees
};

class CameraSimulator {
public:
    CameraSimulator();
    ~CameraSimulator();

    bool initialize(const CameraSimConfig& config);

    // Render a single frame at the given UTC time
    std::vector<PixelData> renderFrame(const std::string& utc_time);

    // Render all configured frames
    void renderAllFrames();

    // Output writers
    bool writeFITS(const std::string& filename, const std::vector<PixelData>& pixels,
                   int nx, int ny, const std::string& utc_time, double et) const;
    bool writePNG(const std::string& filename, const std::vector<PixelData>& pixels,
                  int nx, int ny) const;

private:
    CameraSimConfig config_;
    CameraModel camera_;
    PhotometricModel photometry_;
    PTMOutputReader output_reader_;
    TrajectoryReader trajectory_;

    // Embree
    RTCDevice embree_device_ = nullptr;
    RTCScene embree_scene_ = nullptr;
    std::array<double, 3> scene_origin_offset_{0.0, 0.0, 0.0};

    bool spice_initialized_ = false;

    bool initializeEmbree();
    void cleanupEmbree();
    bool initializeSPICE();

    // Get sun direction in body-fixed frame at given ET
    std::array<double, 3> getSunPosition(double et) const;

    // Convert UTC to ET
    double utcToET(const std::string& utc) const;

    // Unit scale: km_to_mesh_ converts km to mesh units
    // If mesh is in meters, km_to_mesh_ = 1000.0
    // If mesh is in km, km_to_mesh_ = 1.0
    double km_to_mesh_ = 1000.0;
};

} // namespace ptm
