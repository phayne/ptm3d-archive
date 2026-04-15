#include "camera_simulator.h"

#include <iostream>
#include <fstream>
#include <filesystem>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <sstream>

// SPICE
extern "C" {
#include "SpiceUsr.h"
}

// CFITSIO
#include <fitsio.h>

// Embree
#include <embree4/rtcore.h>
#include <embree4/rtcore_ray.h>

namespace fs = std::filesystem;

namespace ptm {

CameraSimulator::CameraSimulator() = default;

CameraSimulator::~CameraSimulator() {
    cleanupEmbree();
}

// ============================================================================
// Initialization
// ============================================================================

bool CameraSimulator::initialize(const CameraSimConfig& config) {
    config_ = config;

    // Initialize SPICE
    if (!initializeSPICE()) {
        std::cerr << "[CameraSimulator] SPICE initialization failed" << std::endl;
        return false;
    }

    // Load PTM3D output
    if (!config_.ptm_output_dir.empty()) {
        if (!output_reader_.loadFromDirectory(config_.ptm_output_dir)) {
            std::cerr << "[CameraSimulator] Failed to load PTM output from "
                      << config_.ptm_output_dir << std::endl;
            return false;
        }
    } else {
        std::cerr << "[CameraSimulator] No PTM output directory specified" << std::endl;
        return false;
    }

    // Initialize Embree scene from mesh
    if (!initializeEmbree()) {
        std::cerr << "[CameraSimulator] Embree initialization failed" << std::endl;
        return false;
    }

    // Initialize photometric model
    photometry_ = PhotometricModel(config_.reflectance_type, config_.photometric_params);
    std::cout << "[CameraSimulator] Photometric model: "
              << PhotometricModel::reflectanceTypeName(config_.reflectance_type) << std::endl;

    // Load trajectory if specified
    if (!config_.trajectory_file.empty()) {
        if (config_.trajectory_format == "spk") {
            if (!trajectory_.loadSPK(config_.trajectory_file,
                                     config_.trajectory_spacecraft,
                                     config_.target_body,
                                     config_.target_frame)) {
                std::cerr << "[CameraSimulator] Failed to load SPK trajectory" << std::endl;
                return false;
            }
        } else if (config_.trajectory_format == "horizons") {
            if (!trajectory_.loadHORIZONS(config_.trajectory_file,
                                          config_.target_body,
                                          config_.target_frame)) {
                return false;
            }
        } else if (config_.trajectory_format == "propman") {
            if (!trajectory_.loadPropMan(config_.trajectory_file,
                                         config_.target_body,
                                         config_.target_frame,
                                         config_.trajectory_epoch_utc)) {
                return false;
            }
        }
    }

    // Create output directory
    fs::create_directories(config_.output_dir);

    return true;
}

bool CameraSimulator::initializeSPICE() {
    if (spice_initialized_) return true;

    // Set SPICE error handling to RETURN mode (don't abort on errors)
    erract_c("SET", 0, const_cast<char*>("RETURN"));
    errprt_c("SET", 0, const_cast<char*>("NONE"));

    // Helper: load a single kernel file with error check
    auto loadKernel = [](const std::string& path, const char* label) -> bool {
        if (!fs::exists(path)) {
            std::cerr << "[CameraSimulator] " << label << " not found: " << path << std::endl;
            return false;
        }
        furnsh_c(path.c_str());
        if (failed_c()) {
            char msg[1024];
            getmsg_c("LONG", sizeof(msg), msg);
            std::cerr << "[CameraSimulator] Error loading " << label << ": " << msg << std::endl;
            reset_c();
            return false;
        }
        return true;
    };

    // Load meta-kernel if specified
    if (!config_.meta_kernel.empty()) {
        furnsh_c(config_.meta_kernel.c_str());
        if (failed_c()) {
            char msg[1024];
            getmsg_c("LONG", sizeof(msg), msg);
            std::cerr << "[CameraSimulator] Meta-kernel error (will try individual kernels): " << msg << std::endl;
            reset_c();
        }
    }

    // Always try to load essential kernels from kernels_path
    if (!config_.kernels_path.empty()) {
        loadKernel(config_.kernels_path + "/lsk/naif0012.tls", "Leapseconds");
        loadKernel(config_.kernels_path + "/pck/pck00011.tpc", "PCK");

        // Try loading planetary ephemerides
        for (const auto& spk_name : {"de440s.bsp", "de440.bsp", "de442.bsp"}) {
            std::string spk_path = config_.kernels_path + "/spk/" + spk_name;
            if (fs::exists(spk_path)) {
                loadKernel(spk_path, spk_name);
                break;
            }
        }

        // Load GM constants if available
        std::string gm = config_.kernels_path + "/pck/gm_de431.tpc";
        if (fs::exists(gm)) {
            loadKernel(gm, "GM constants");
        }

        // Load Jupiter SPKs if available (for Europa/Galilean moons)
        for (const auto& jup_spk : {"jup365.bsp", "jup363.bsp", "jup346.bsp", "jup344.bsp"}) {
            std::string path = config_.kernels_path + "/spk/" + jup_spk;
            if (fs::exists(path)) {
                loadKernel(path, jup_spk);
            }
        }
    }

    spice_initialized_ = true;
    std::cout << "[CameraSimulator] SPICE initialized" << std::endl;
    return true;
}

bool CameraSimulator::initializeEmbree() {
    auto mesh = output_reader_.getMesh();
    if (!mesh || mesh->vertices.empty()) {
        std::cerr << "[CameraSimulator] No mesh loaded" << std::endl;
        return false;
    }

    embree_device_ = rtcNewDevice(nullptr);
    if (!embree_device_) {
        std::cerr << "[CameraSimulator] Failed to create Embree device" << std::endl;
        return false;
    }

    embree_scene_ = rtcNewScene(embree_device_);
    RTCGeometry geom = rtcNewGeometry(embree_device_, RTC_GEOMETRY_TYPE_TRIANGLE);

    // Compute scene origin offset (bounding box center) for float precision
    {
        double min_x = std::numeric_limits<double>::max(), max_x = std::numeric_limits<double>::lowest();
        double min_y = min_x, max_y = max_x;
        double min_z = min_x, max_z = max_x;
        for (const auto& v : mesh->vertices) {
            if (v[0] < min_x) min_x = v[0]; if (v[0] > max_x) max_x = v[0];
            if (v[1] < min_y) min_y = v[1]; if (v[1] > max_y) max_y = v[1];
            if (v[2] < min_z) min_z = v[2]; if (v[2] > max_z) max_z = v[2];
        }
        scene_origin_offset_ = { 0.5*(min_x+max_x), 0.5*(min_y+max_y), 0.5*(min_z+max_z) };
    }

    // Set vertices (shifted by offset)
    float* vertices = (float*)rtcSetNewGeometryBuffer(
        geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3,
        3 * sizeof(float), mesh->vertices.size());

    for (size_t i = 0; i < mesh->vertices.size(); ++i) {
        vertices[3*i + 0] = static_cast<float>(mesh->vertices[i][0] - scene_origin_offset_[0]);
        vertices[3*i + 1] = static_cast<float>(mesh->vertices[i][1] - scene_origin_offset_[1]);
        vertices[3*i + 2] = static_cast<float>(mesh->vertices[i][2] - scene_origin_offset_[2]);
    }

    // Set triangle indices
    unsigned* triangles = (unsigned*)rtcSetNewGeometryBuffer(
        geom, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3,
        3 * sizeof(unsigned), mesh->faces.size());

    for (size_t i = 0; i < mesh->faces.size(); ++i) {
        triangles[3*i + 0] = static_cast<unsigned>(mesh->faces[i][0]);
        triangles[3*i + 1] = static_cast<unsigned>(mesh->faces[i][1]);
        triangles[3*i + 2] = static_cast<unsigned>(mesh->faces[i][2]);
    }

    rtcCommitGeometry(geom);
    rtcAttachGeometry(embree_scene_, geom);
    rtcReleaseGeometry(geom);
    rtcCommitScene(embree_scene_);

    // Auto-detect mesh scale: compute average vertex distance from origin
    // If > 10000, assume meters; if < 10000, assume km
    {
        double sum_r = 0.0;
        for (const auto& v : mesh->vertices) {
            sum_r += std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
        }
        double avg_r = sum_r / mesh->vertices.size();
        if (avg_r > 10000.0) {
            km_to_mesh_ = 1000.0;  // mesh in meters
            std::cout << "[CameraSimulator] Mesh scale: meters (avg radius = "
                      << std::fixed << std::setprecision(0) << avg_r << " m)" << std::endl;
        } else {
            km_to_mesh_ = 1.0;  // mesh in km
            std::cout << "[CameraSimulator] Mesh scale: km (avg radius = "
                      << std::fixed << std::setprecision(1) << avg_r << " km)" << std::endl;
        }
    }

    std::cout << "[CameraSimulator] Embree scene: " << mesh->vertices.size()
              << " vertices, " << mesh->faces.size() << " faces" << std::endl;
    return true;
}

void CameraSimulator::cleanupEmbree() {
    if (embree_scene_) { rtcReleaseScene(embree_scene_); embree_scene_ = nullptr; }
    if (embree_device_) { rtcReleaseDevice(embree_device_); embree_device_ = nullptr; }
}

double CameraSimulator::utcToET(const std::string& utc) const {
    SpiceDouble et;
    str2et_c(utc.c_str(), &et);
    if (failed_c()) {
        char msg[1024];
        getmsg_c("LONG", sizeof(msg), msg);
        std::cerr << "[CameraSimulator] str2et_c error for '" << utc << "': " << msg << std::endl;
        reset_c();
        return 0.0;
    }
    return et;
}

std::array<double, 3> CameraSimulator::getSunPosition(double et) const {
    SpiceDouble pos[3];
    SpiceDouble lt;
    spkpos_c("SUN", et, config_.target_frame.c_str(), "LT+S",
             config_.target_body.c_str(), pos, &lt);

    if (failed_c()) {
        char msg[1024];
        getmsg_c("LONG", sizeof(msg), msg);
        std::cerr << "[CameraSimulator] getSunPosition error: " << msg << std::endl;
        reset_c();
        return {0, 0, 0};
    }

    return {pos[0], pos[1], pos[2]};
}

// ============================================================================
// Rendering
// ============================================================================

std::vector<PixelData> CameraSimulator::renderFrame(const std::string& utc_time) {
    double et = utcToET(utc_time);
    if (et == 0.0) return {};

    auto mesh = output_reader_.getMesh();
    if (!mesh) return {};

    // Determine camera position (in km from SPICE/trajectory, convert to mesh units)
    std::array<double, 3> cam_pos_km;
    if (trajectory_.isLoaded() || trajectory_.usesSPICE()) {
        cam_pos_km = trajectory_.getPosition(et);
    } else {
        cam_pos_km = config_.camera.position;
    }

    // Convert camera position to mesh units for ray casting
    CameraConfig cam_config = config_.camera;
    cam_config.position = {
        cam_pos_km[0] * km_to_mesh_,
        cam_pos_km[1] * km_to_mesh_,
        cam_pos_km[2] * km_to_mesh_
    };
    // Scale look-at to mesh units too
    cam_config.look_at = {
        config_.camera.look_at[0] * km_to_mesh_,
        config_.camera.look_at[1] * km_to_mesh_,
        config_.camera.look_at[2] * km_to_mesh_
    };
    camera_.configure(cam_config);
    auto cam_pos = camera_.position();  // Now in mesh units

    // Get sun position (from SPICE, in km) — we only need direction, not absolute position
    auto sun_pos_km = getSunPosition(et);
    double sun_dist_km = vec::length(sun_pos_km);
    auto sun_dir = vec::normalize(sun_pos_km);

    // Solar flux at this distance (inverse square law)
    constexpr double SOLAR_CONSTANT = 1361.0;  // W/m^2 at 1 AU
    constexpr double KM_PER_AU = 1.495978707e8;
    double sun_dist_au = sun_dist_km / KM_PER_AU;
    double solar_flux = SOLAR_CONSTANT / (sun_dist_au * sun_dist_au);

    // Load temperature data if needed
    std::vector<float> temperatures;
    if (config_.render_mode == CameraSimConfig::RenderMode::THERMAL ||
        config_.render_mode == CameraSimConfig::RenderMode::BOTH) {
        if (output_reader_.hasTemperature()) {
            temperatures = output_reader_.getTemperatureAtET(et);
        }
    }

    // Generate all camera rays
    int nx = camera_.width();
    int ny = camera_.height();
    size_t n_pixels = static_cast<size_t>(nx) * ny;

    std::vector<std::array<double, 3>> ray_origins, ray_directions;
    camera_.generateAllRays(ray_origins, ray_directions);

    // Allocate output
    std::vector<PixelData> pixels(n_pixels);

    std::cout << "[CameraSimulator] Rendering " << nx << "x" << ny << " at " << utc_time
              << " (ET=" << std::fixed << std::setprecision(3) << et << ")" << std::endl;

    // Camera direction for emission angle computation
    auto cam_dir_to_body = vec::normalize(vec::sub({0,0,0}, cam_pos));

    // Ray cast all pixels
    #pragma omp parallel for schedule(dynamic, 64)
    for (size_t i = 0; i < n_pixels; ++i) {
        const auto& origin = ray_origins[i];
        const auto& dir = ray_directions[i];

        // Apply scene origin offset for Embree (mesh vertices are in the body-fixed
        // frame, typically meters for the mesh; camera is in km — we need consistent units)
        // The mesh is stored in its native units; camera positions are in km.
        // Convert camera position to mesh units (mesh vertices are in the same
        // coordinate system as stored in the PLY file).
        double ox = origin[0] - scene_origin_offset_[0];
        double oy = origin[1] - scene_origin_offset_[1];
        double oz = origin[2] - scene_origin_offset_[2];

        RTCRayHit rayhit{};
        rayhit.ray.org_x = static_cast<float>(ox);
        rayhit.ray.org_y = static_cast<float>(oy);
        rayhit.ray.org_z = static_cast<float>(oz);
        rayhit.ray.dir_x = static_cast<float>(dir[0]);
        rayhit.ray.dir_y = static_cast<float>(dir[1]);
        rayhit.ray.dir_z = static_cast<float>(dir[2]);

        double r_local = std::sqrt(ox*ox + oy*oy + oz*oz);
        rayhit.ray.tnear = static_cast<float>(std::max(1e-6 * r_local, 0.01));
        rayhit.ray.tfar = 1e20f;
        rayhit.ray.mask = 0xFFFFFFFF;
        rayhit.ray.flags = 0;
        rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;

        RTCIntersectArguments args;
        rtcInitIntersectArguments(&args);
        rtcIntersect1(embree_scene_, &rayhit, &args);

        PixelData& px = pixels[i];

        if (rayhit.hit.geomID == RTC_INVALID_GEOMETRY_ID) {
            // Miss — sky pixel
            px.face_index = -1;
            continue;
        }

        int face_id = static_cast<int>(rayhit.hit.primID);
        px.face_index = face_id;
        px.distance_km = static_cast<float>(rayhit.ray.tfar / km_to_mesh_);

        // Face properties
        const auto& normal = mesh->face_normals[face_id];

        // Compute viewing geometry
        // View direction: from surface to camera
        auto view_dir = vec::normalize(vec::sub(origin, mesh->face_centers[face_id]));

        // cos(emission angle) = dot(normal, view_dir)
        double cos_e = vec::dot(normal, view_dir);
        cos_e = std::max(0.0, std::min(1.0, cos_e));

        // cos(incidence angle) = dot(normal, sun_dir)
        double cos_i = vec::dot(normal, sun_dir);
        cos_i = std::max(-1.0, std::min(1.0, cos_i));

        // Phase angle = angle between sun direction and view direction at the surface
        double cos_alpha = vec::dot(sun_dir, view_dir);
        cos_alpha = std::max(-1.0, std::min(1.0, cos_alpha));
        double alpha_rad = std::acos(cos_alpha);

        px.emission_angle = static_cast<float>(std::acos(cos_e) * 180.0 / M_PI);
        px.incidence_angle = static_cast<float>(std::acos(std::max(0.0, cos_i)) * 180.0 / M_PI);
        px.phase_angle = static_cast<float>(alpha_rad * 180.0 / M_PI);

        // Latitude/longitude
        if (face_id < static_cast<int>(mesh->latitudes.size())) {
            px.latitude = static_cast<float>(mesh->latitudes[face_id]);
            px.longitude = static_cast<float>(mesh->longitudes[face_id]);
        }

        // Reflected radiance
        if (config_.render_mode == CameraSimConfig::RenderMode::REFLECTED ||
            config_.render_mode == CameraSimConfig::RenderMode::BOTH) {
            px.radiance = static_cast<float>(
                photometry_.reflectedRadiance(std::max(0.0, cos_i), cos_e, alpha_rad, solar_flux));
        }

        // Thermal radiance
        if (config_.render_mode == CameraSimConfig::RenderMode::THERMAL ||
            config_.render_mode == CameraSimConfig::RenderMode::BOTH) {
            if (!temperatures.empty() && face_id < static_cast<int>(temperatures.size())) {
                float T = temperatures[face_id];
                px.temperature = T;
                if (T > 0.0f) {
                    px.thermal_radiance = static_cast<float>(
                        photometry_.thermalSpectralRadiance(T, cos_e,
                            config_.photometric_params.thermal_wavelength_um));
                }
            }
        }
    }

    // Count hits
    int hits = 0;
    for (const auto& px : pixels) {
        if (px.face_index >= 0) hits++;
    }
    std::cout << "[CameraSimulator] " << hits << "/" << n_pixels
              << " pixels hit surface (" << std::fixed << std::setprecision(1)
              << (100.0 * hits / n_pixels) << "%)" << std::endl;

    return pixels;
}

void CameraSimulator::renderAllFrames() {
    for (size_t i = 0; i < config_.render_times.size(); ++i) {
        const auto& utc = config_.render_times[i];
        std::cout << "\n=== Frame " << (i+1) << "/" << config_.render_times.size()
                  << " ===" << std::endl;

        auto pixels = renderFrame(utc);
        if (pixels.empty()) {
            std::cerr << "[CameraSimulator] Frame " << i << " failed" << std::endl;
            continue;
        }

        double et = utcToET(utc);

        // Build output filename
        std::ostringstream oss;
        oss << config_.output_dir << "/frame_" << std::setw(4) << std::setfill('0') << i;

        if (config_.output_fits) {
            std::string fits_path = oss.str() + ".fits";
            writeFITS(fits_path, pixels, camera_.width(), camera_.height(), utc, et);
        }

        if (config_.output_png) {
            std::string png_path = oss.str() + ".pgm";  // PGM for now (no PNG library)
            writePNG(png_path, pixels, camera_.width(), camera_.height());
        }
    }
}

// ============================================================================
// FITS output
// ============================================================================

bool CameraSimulator::writeFITS(const std::string& filename,
                                 const std::vector<PixelData>& pixels,
                                 int nx, int ny,
                                 const std::string& utc_time, double et) const {
    fitsfile* fptr = nullptr;
    int status = 0;

    // Remove existing file
    std::string bang_path = "!" + filename;
    fits_create_file(&fptr, bang_path.c_str(), &status);
    if (status) {
        std::cerr << "[CameraSimulator] FITS create error: " << status << std::endl;
        return false;
    }

    long naxes[2] = {nx, ny};
    size_t n = static_cast<size_t>(nx) * ny;

    // Helper lambda: write a float image extension
    auto writeFloatImage = [&](const char* extname, const std::vector<float>& data) {
        fits_create_img(fptr, FLOAT_IMG, 2, naxes, &status);
        fits_write_img(fptr, TFLOAT, 1, static_cast<long>(n),
                       const_cast<float*>(data.data()), &status);
        fits_write_key(fptr, TSTRING, "EXTNAME", const_cast<char*>(extname), nullptr, &status);
    };

    // Helper: write an int32 image extension
    auto writeIntImage = [&](const char* extname, const std::vector<int32_t>& data) {
        fits_create_img(fptr, LONG_IMG, 2, naxes, &status);
        fits_write_img(fptr, TINT, 1, static_cast<long>(n),
                       const_cast<int32_t*>(data.data()), &status);
        fits_write_key(fptr, TSTRING, "EXTNAME", const_cast<char*>(extname), nullptr, &status);
    };

    // Extract per-field arrays
    std::vector<float> radiance(n), thermal_rad(n), incidence(n), emission(n);
    std::vector<float> phase(n), temperature(n), distance(n), lat(n), lon(n);
    std::vector<int32_t> face_idx(n);

    for (size_t i = 0; i < n; ++i) {
        const auto& px = pixels[i];
        radiance[i] = px.radiance;
        thermal_rad[i] = px.thermal_radiance;
        face_idx[i] = px.face_index;
        incidence[i] = px.incidence_angle;
        emission[i] = px.emission_angle;
        phase[i] = px.phase_angle;
        temperature[i] = px.temperature;
        distance[i] = px.distance_km;
        lat[i] = px.latitude;
        lon[i] = px.longitude;
    }

    // HDU 0: Primary — radiance
    writeFloatImage("RADIANCE", radiance);

    // Add header keywords to primary
    {
        char utc_buf[64];
        strncpy(utc_buf, utc_time.c_str(), sizeof(utc_buf)-1);
        utc_buf[sizeof(utc_buf)-1] = '\0';
        fits_update_key(fptr, TSTRING, "UTC_TIME", utc_buf, "Render time (UTC)", &status);
        fits_update_key(fptr, TDOUBLE, "ET_TIME", const_cast<double*>(&et), "Ephemeris time (s past J2000)", &status);

        auto cam_pos = camera_.position();
        fits_update_key(fptr, TDOUBLE, "CAM_X", &cam_pos[0], "Camera X (km, body-fixed)", &status);
        fits_update_key(fptr, TDOUBLE, "CAM_Y", &cam_pos[1], "Camera Y (km, body-fixed)", &status);
        fits_update_key(fptr, TDOUBLE, "CAM_Z", &cam_pos[2], "Camera Z (km, body-fixed)", &status);

        double fovx = camera_.fovXDeg();
        double fovy = camera_.fovYDeg();
        fits_update_key(fptr, TDOUBLE, "FOV_X", &fovx, "Horizontal FOV (deg)", &status);
        fits_update_key(fptr, TDOUBLE, "FOV_Y", &fovy, "Vertical FOV (deg)", &status);

        char target_buf[64];
        strncpy(target_buf, config_.target_body.c_str(), sizeof(target_buf)-1);
        target_buf[sizeof(target_buf)-1] = '\0';
        fits_update_key(fptr, TSTRING, "TARGET", target_buf, "Target body", &status);

        auto photmod = PhotometricModel::reflectanceTypeName(config_.reflectance_type);
        char mod_buf[64];
        strncpy(mod_buf, photmod.c_str(), sizeof(mod_buf)-1);
        mod_buf[sizeof(mod_buf)-1] = '\0';
        fits_update_key(fptr, TSTRING, "PHOTMOD", mod_buf, "Photometric model", &status);
    }

    // Extension HDUs
    writeFloatImage("THERMAL", thermal_rad);
    writeIntImage("FACE_IDX", face_idx);
    writeFloatImage("INC_ANG", incidence);
    writeFloatImage("EMI_ANG", emission);
    writeFloatImage("PHA_ANG", phase);
    writeFloatImage("TEMP_K", temperature);
    writeFloatImage("DIST_KM", distance);
    writeFloatImage("LATITUDE", lat);
    writeFloatImage("LONGITUDE", lon);

    fits_close_file(fptr, &status);

    if (status) {
        std::cerr << "[CameraSimulator] FITS write error: " << status << std::endl;
        return false;
    }

    std::cout << "[CameraSimulator] Wrote FITS: " << filename << std::endl;
    return true;
}

bool CameraSimulator::writePNG(const std::string& filename,
                                const std::vector<PixelData>& pixels,
                                int nx, int ny) const {
    // Write a simple PGM (Portable GrayMap) — no external PNG library needed
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[CameraSimulator] Cannot open " << filename << std::endl;
        return false;
    }

    // Find radiance range for normalization
    float max_rad = 0.0f;
    for (const auto& px : pixels) {
        if (px.radiance > max_rad) max_rad = px.radiance;
    }
    if (max_rad <= 0.0f) max_rad = 1.0f;

    // PGM header
    file << "P5\n" << nx << " " << ny << "\n255\n";

    // Pixel data: gamma-corrected 8-bit
    for (size_t i = 0; i < pixels.size(); ++i) {
        float val = pixels[i].radiance / max_rad;
        val = std::pow(std::max(0.0f, val), 1.0f / 2.2f);  // gamma correction
        uint8_t byte = static_cast<uint8_t>(std::min(255.0f, val * 255.0f));
        file.write(reinterpret_cast<const char*>(&byte), 1);
    }

    std::cout << "[CameraSimulator] Wrote PGM: " << filename << std::endl;
    return true;
}

} // namespace ptm
