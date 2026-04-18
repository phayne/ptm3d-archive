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

    // Albedo map (per-face, binary: uint64 count + double[count])
    std::string albedo_map_file;

    // Background celestial bodies (rendered for sky pixels)
    bool render_sun = false;
    bool render_primary = false;   // Render the primary body (planet for moons)

    // Primary body configuration. For moons, the primary is the parent planet
    // (Moon→Earth, Europa→Jupiter). Auto-detected from target_body NAIF ID
    // when primary_body is empty. For planets/asteroids, no primary exists.
    std::string primary_body;       // Empty = auto-detect; explicit overrides
    double primary_radius_km = 0;   // 0 = query SPICE bodvrd_c at runtime
    double primary_bond_albedo = -1; // <0 = lookup from built-in table

    // Sub-pixel supersampling (anti-aliasing). 1 = no AA (legacy, one ray at pixel
    // center). N>1 = cast N*N rays per pixel on a uniform sub-pixel grid and average.
    // This approximates a real camera integrating scene radiance over its pixel IFOV,
    // removing per-face aliasing (sparkling) when small triangles subtend <~1 pixel.
    // Default N=2 matches a real camera's pixel integration and is recommended for
    // animations on dense meshes (ico7+). Set to 1 for bit-identical legacy output.
    int supersample = 2;

    // Smooth shading. When true, the surface normal and albedo are interpolated
    // linearly across each triangle using Embree barycentric hit coordinates
    // (Phong-style, rather than flat-shaded). This removes the step discontinuity
    // at face boundaries that causes the bulk of the remaining sparkle when a
    // moving camera samples different faces frame-to-frame. Enabled by default.
    // Pass --no-smooth-shading on the CLI to disable.
    bool smooth_shading = true;

    // Camera YAML definition file (optional). If provided, its contents populate
    // the camera, photometric, and SPICE IK fields below; CLI flags still override.
    std::string camera_yaml_file;

    // SPICE Instrument Kernel (IK) and C-Kernel (CK). When CameraConfig::pointing
    // == SPICE_CK, these are loaded via furnsh_c and the camera orientation is
    // driven per-frame by pxform_c(instrument_frame, target_frame, et).
    std::string ik_kernel;
    std::string ck_kernel;

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

    // Resolved primary body (populated by resolvePrimaryBody after SPICE init)
    std::string primary_body_name_;
    double primary_radius_km_ = 0;
    double primary_bond_albedo_ = 0;

    bool initializeEmbree();
    void cleanupEmbree();
    bool initializeSPICE();
    void resolvePrimaryBody();
    static double lookupBondAlbedo(const std::string& body);

    // Per-frame orientation update. Calls configureFromSPICE when the camera's
    // pointing mode is SPICE_CK, otherwise a no-op (static pointing is handled
    // by camera_.configure() once per frame). Must be called with the frame's ET
    // before camera_.configure(). Returns false on SPICE lookup failure.
    bool updateOrientation(double et);

    // Get body position in target body-fixed frame at given ET (km)
    std::array<double, 3> getSunPosition(double et) const;
    std::array<double, 3> getBodyPosition(const std::string& body, double et) const;

    // Convert UTC to ET
    double utcToET(const std::string& utc) const;

    // Celestial body rendering for sky pixels
    // Returns radiance for a ray that intersects a distant sphere (Sun or primary)
    struct CelestialBodyParams {
        std::array<double, 3> pos_km;  // Position in body-fixed frame (km)
        double radius_km;              // Body radius (km)
        double albedo;                 // Bond albedo (for reflected-light bodies)
        bool is_emitter;               // true = Sun (self-luminous), false = reflected
    };

    // Test if ray hits a distant sphere; if so, compute radiance
    // cam_pos_km: camera position in km
    // ray_dir: unit ray direction
    // sun_dir: unit direction to Sun from the body center (for phase computation)
    // solar_flux: solar irradiance at the body's distance (W/m^2)
    // Returns radiance (W/m^2/sr), or 0 if miss
    static double renderCelestialBody(const std::array<double, 3>& cam_pos_km,
                                       const std::array<double, 3>& ray_dir,
                                       const CelestialBodyParams& body,
                                       const std::array<double, 3>& sun_dir_from_body,
                                       double solar_flux_at_body);

    // Unit scale: km_to_mesh_ converts km to mesh units
    // If mesh is in meters, km_to_mesh_ = 1000.0
    // If mesh is in km, km_to_mesh_ = 1.0
    double km_to_mesh_ = 1000.0;

    // Per-face albedo map (empty = use uniform albedo from config)
    std::vector<double> face_albedos_;
    bool loadAlbedoMap(const std::string& filepath);

    // Per-vertex smooth-shading attributes (populated lazily when
    // config_.smooth_shading is true). Area-weighted averages of adjacent
    // face quantities. Sized to mesh->vertices.size() once populated.
    std::vector<std::array<double, 3>> vertex_normals_;   // unit vectors
    std::vector<double>                vertex_albedos_;   // per-vertex albedo
    bool smooth_attrs_ready_ = false;
    void computeSmoothShadingAttributes();
};

} // namespace ptm
