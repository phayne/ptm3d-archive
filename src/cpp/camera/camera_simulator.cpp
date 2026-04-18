#include "camera_simulator.h"

#include <iostream>
#include <fstream>
#include <filesystem>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <unordered_map>

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

    // Resolve primary body (planet for moons) from SPICE
    resolvePrimaryBody();

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

    // Load albedo map if specified
    if (!config_.albedo_map_file.empty()) {
        if (!loadAlbedoMap(config_.albedo_map_file)) {
            std::cerr << "[CameraSimulator] Warning: failed to load albedo map, using uniform albedo" << std::endl;
        }
    }

    // If smooth shading requested, compute per-vertex normals and albedos.
    // (Requires mesh loaded by initializeEmbree + albedo map loaded above.)
    if (config_.smooth_shading) {
        computeSmoothShadingAttributes();
    }

    // Create output directory
    fs::create_directories(config_.output_dir);

    return true;
}

bool CameraSimulator::loadAlbedoMap(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[CameraSimulator] Cannot open albedo map: " << filepath << std::endl;
        return false;
    }

    uint64_t count = 0;
    file.read(reinterpret_cast<char*>(&count), sizeof(uint64_t));
    if (!file.good() || count == 0 || count > 100000000) {
        std::cerr << "[CameraSimulator] Invalid albedo map count: " << count << std::endl;
        return false;
    }

    face_albedos_.resize(count);
    file.read(reinterpret_cast<char*>(face_albedos_.data()), count * sizeof(double));
    if (!file.good()) {
        std::cerr << "[CameraSimulator] Error reading albedo map data" << std::endl;
        face_albedos_.clear();
        return false;
    }

    // Compute stats
    double min_a = *std::min_element(face_albedos_.begin(), face_albedos_.end());
    double max_a = *std::max_element(face_albedos_.begin(), face_albedos_.end());
    double sum_a = 0.0;
    for (double a : face_albedos_) sum_a += a;
    double mean_a = sum_a / count;

    std::cout << "[CameraSimulator] Loaded albedo map: " << count << " faces, range ["
              << std::fixed << std::setprecision(4) << min_a << ", " << max_a
              << "], mean " << mean_a << std::endl;

    return true;
}

void CameraSimulator::computeSmoothShadingAttributes() {
    auto mesh = output_reader_.getMesh();
    if (!mesh) return;
    const size_t nv = mesh->vertices.size();
    const size_t nf = mesh->faces.size();

    vertex_normals_.assign(nv, std::array<double, 3>{0.0, 0.0, 0.0});
    std::vector<double> w_normal(nv, 0.0);

    const bool has_albedo = (!face_albedos_.empty() && face_albedos_.size() == nf);

    // Only allocate vertex albedos when we have per-face albedo data to
    // propagate. When empty, the render loop falls through to the uniform
    // albedo path, avoiding a "zero-albedo" black-image bug.
    std::vector<double> w_albedo;
    if (has_albedo) {
        vertex_albedos_.assign(nv, 0.0);
        w_albedo.assign(nv, 0.0);
    } else {
        vertex_albedos_.clear();
    }

    // Accumulate area-weighted face attributes onto each of the face's three vertices.
    // Single-threaded to avoid contention on shared vertex accumulators (0.5 s for
    // 5.2M faces).
    for (size_t f = 0; f < nf; ++f) {
        const auto& tri = mesh->faces[f];
        const int v0 = tri[0], v1 = tri[1], v2 = tri[2];
        const auto& p0 = mesh->vertices[v0];
        const auto& p1 = mesh->vertices[v1];
        const auto& p2 = mesh->vertices[v2];
        // Area from cross product magnitude / 2
        const std::array<double, 3> e1 = {p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2]};
        const std::array<double, 3> e2 = {p2[0]-p0[0], p2[1]-p0[1], p2[2]-p0[2]};
        const std::array<double, 3> cr = vec::cross(e1, e2);
        const double area = 0.5 * vec::length(cr);
        if (area <= 0.0) continue;

        const auto& fn = mesh->face_normals[f];
        for (int k = 0; k < 3; ++k) {
            const int vi = tri[k];
            vertex_normals_[vi][0] += fn[0] * area;
            vertex_normals_[vi][1] += fn[1] * area;
            vertex_normals_[vi][2] += fn[2] * area;
            w_normal[vi] += area;

            if (has_albedo) {
                vertex_albedos_[vi] += face_albedos_[f] * area;
                w_albedo[vi] += area;
            }
        }
    }

    // Normalize
    size_t n_normal = 0, n_albedo = 0;
    for (size_t v = 0; v < nv; ++v) {
        if (w_normal[v] > 0.0) {
            vertex_normals_[v] = vec::normalize(vertex_normals_[v]);
            ++n_normal;
        }
        if (has_albedo && w_albedo[v] > 0.0) {
            vertex_albedos_[v] /= w_albedo[v];
            ++n_albedo;
        }
    }

    smooth_attrs_ready_ = true;
    std::cout << "[CameraSimulator] Smooth-shading attributes: "
              << n_normal << " vertex normals, "
              << n_albedo << " vertex albedos populated" << std::endl;
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

    // Load Instrument Kernel (IK) and C-Kernel (CK) if supplied. These provide
    // FOV / boresight / frame (IK) and per-time attitude (CK) for SPICE-driven
    // pointing. Failures are logged but not fatal — manual pointing still works.
    if (!config_.ik_kernel.empty()) {
        loadKernel(config_.ik_kernel, "Instrument Kernel (IK)");
    }
    if (!config_.ck_kernel.empty()) {
        loadKernel(config_.ck_kernel, "C-Kernel (CK)");
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

double CameraSimulator::lookupBondAlbedo(const std::string& body) {
    static const std::unordered_map<std::string, double> table = {
        {"EARTH", 0.306}, {"JUPITER", 0.343}, {"SATURN", 0.342},
        {"MARS", 0.250}, {"VENUS", 0.770}, {"NEPTUNE", 0.290},
        {"URANUS", 0.300},
    };
    auto it = table.find(body);
    return (it != table.end()) ? it->second : 0.12;
}

void CameraSimulator::resolvePrimaryBody() {
    // If the user explicitly set a primary body, use it.
    if (!config_.primary_body.empty()) {
        primary_body_name_ = config_.primary_body;
    } else {
        // Auto-detect: for a moon with NAIF ID NXX (N=1-9, XX=01-98),
        // the parent planet is N*100 + 99 (e.g., 301 → 399 = EARTH).
        SpiceInt target_id = 0;
        SpiceBoolean found = SPICEFALSE;
        bodn2c_c(config_.target_body.c_str(), &target_id, &found);
        if (failed_c()) { reset_c(); found = SPICEFALSE; }

        if (found && target_id > 100 && (target_id % 100) != 99) {
            SpiceInt primary_id = (target_id / 100) * 100 + 99;
            char name[64];
            SpiceBoolean found2 = SPICEFALSE;
            bodc2n_c(primary_id, sizeof(name), name, &found2);
            if (failed_c()) { reset_c(); found2 = SPICEFALSE; }
            if (found2) primary_body_name_ = name;
        }
        // Planets (ID % 100 == 99) and small bodies: no primary
    }

    if (primary_body_name_.empty()) {
        std::cout << "[CameraSimulator] No primary body for target "
                  << config_.target_body << std::endl;
        return;
    }

    // Resolve radius from SPICE (bodvrd_c)
    if (config_.primary_radius_km > 0) {
        primary_radius_km_ = config_.primary_radius_km;
    } else {
        SpiceInt n = 0;
        SpiceDouble radii[3];
        bodvrd_c(primary_body_name_.c_str(), "RADII", 3, &n, radii);
        if (!failed_c() && n >= 1) {
            primary_radius_km_ = radii[0];  // equatorial radius
        } else {
            reset_c();
            primary_radius_km_ = 6371.0;  // fallback
            std::cerr << "[CameraSimulator] Could not query radius for "
                      << primary_body_name_ << "; using fallback" << std::endl;
        }
    }

    // Resolve albedo
    if (config_.primary_bond_albedo >= 0) {
        primary_bond_albedo_ = config_.primary_bond_albedo;
    } else {
        primary_bond_albedo_ = lookupBondAlbedo(primary_body_name_);
    }

    std::cout << "[CameraSimulator] Primary body: " << primary_body_name_
              << " (radius " << std::fixed << std::setprecision(0)
              << primary_radius_km_ << " km, albedo "
              << std::setprecision(3) << primary_bond_albedo_ << ")"
              << std::endl;
}

bool CameraSimulator::updateOrientation(double et) {
    // Static pointing (LOOK_AT / QUATERNION / ROTATION_MATRIX) is handled inside
    // camera_.configure() each frame — nothing to do here.
    if (config_.camera.pointing != CameraConfig::Pointing::SPICE_CK) {
        return true;
    }

    // SPICE_CK: query instrument-frame → target-frame rotation at ET and set
    // the camera axes. camera_.configure() has already set position / intrinsics;
    // here we overwrite forward_/right_/up_ with the IK+CK-driven values.
    return camera_.configureFromSPICE(config_.camera, et,
                                       config_.target_body, config_.target_frame);
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

std::array<double, 3> CameraSimulator::getBodyPosition(const std::string& body, double et) const {
    SpiceDouble pos[3];
    SpiceDouble lt;
    spkpos_c(body.c_str(), et, config_.target_frame.c_str(), "LT+S",
             config_.target_body.c_str(), pos, &lt);

    if (failed_c()) {
        char msg[1024];
        getmsg_c("LONG", sizeof(msg), msg);
        std::cerr << "[CameraSimulator] getBodyPosition(" << body << ") error: " << msg << std::endl;
        reset_c();
        return {0, 0, 0};
    }

    return {pos[0], pos[1], pos[2]};
}

double CameraSimulator::renderCelestialBody(
    const std::array<double, 3>& cam_pos_km,
    const std::array<double, 3>& ray_dir,
    const CelestialBodyParams& body,
    const std::array<double, 3>& sun_dir_from_body,
    double solar_flux_at_body)
{
    // Vector from camera to body center
    auto to_body = vec::sub(body.pos_km, cam_pos_km);
    double dist = vec::length(to_body);
    if (dist <= body.radius_km) return 0.0;  // inside the body

    auto to_body_dir = vec::normalize(to_body);

    // Check if ray direction is within the angular radius of the body
    double cos_ang = vec::dot(ray_dir, to_body_dir);
    double sin_ang_body = body.radius_km / dist;
    double cos_ang_body = std::sqrt(1.0 - sin_ang_body * sin_ang_body);

    if (cos_ang < cos_ang_body) return 0.0;  // ray misses the body

    // Ray hits the body disk — compute where on the disk
    // Project ray onto the body: find intersection with sphere
    // Parametric: P = cam + t * ray_dir
    // |P - body_center|^2 = R^2
    // Solve quadratic: t^2 + 2*b*t + c = 0
    // where b = dot(cam - center, dir), c = |cam - center|^2 - R^2
    auto oc = vec::sub(cam_pos_km, body.pos_km);
    double b = vec::dot(oc, ray_dir);
    double c = vec::dot(oc, oc) - body.radius_km * body.radius_km;
    double discriminant = b * b - c;
    if (discriminant < 0.0) return 0.0;

    double t = -b - std::sqrt(discriminant);  // nearest intersection
    if (t < 0.0) return 0.0;

    // Hit point on the body surface
    auto hit = vec::add(cam_pos_km, vec::scale(ray_dir, t));
    // Normal at hit point (outward from body center)
    auto normal = vec::normalize(vec::sub(hit, body.pos_km));

    if (body.is_emitter) {
        // Sun: uniform emitter
        // Solar spectral radiance approximation: L = F_sun / pi (at 1 AU, the Sun
        // subtends a solid angle, and F = L * pi * (R/d)^2)
        // More precisely: the Sun's surface radiance is L = SOLAR_CONSTANT / (pi * sin^2(angular_radius))
        // At 1 AU: sin(angular_rad) = R_sun/1AU, L = 1361 / (pi * (R_sun/1AU)^2)
        // But since we're rendering it as seen from the camera, we want a fixed
        // brightness that looks like the Sun. Use the solar disk radiance:
        // L_sun = F_sun_at_1AU / (pi * Omega_sun) where Omega_sun = pi*(R_sun/1AU)^2
        // L_sun = 1361 / (pi^2 * (6.96e5/1.496e8)^2) ≈ 2.01e7 W/m^2/sr
        // But for display purposes, just use a high constant.
        constexpr double SUN_SURFACE_RADIANCE = 2.01e7;  // W/m^2/sr
        return SUN_SURFACE_RADIANCE;
    } else {
        // Reflected light body (Earth, etc.)
        // Lambertian sphere: L = albedo/pi * cos(incidence) * F_sun
        // where incidence = angle between surface normal and sun direction
        double cos_i = vec::dot(normal, sun_dir_from_body);
        if (cos_i <= 0.0) return 0.0;  // nightside — no reflected light

        // Lambertian disk radiance
        return (body.albedo / M_PI) * cos_i * solar_flux_at_body;
    }
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

    // Compute dynamic look-at (km, body-fixed) before scaling to mesh units.
    // For FIXED mode, config_.camera.look_at is used as-is. For dynamic modes,
    // the look-at is recomputed each frame from the current geometry.
    std::array<double, 3> look_at_km = config_.camera.look_at;
    const double body_r_km = (km_to_mesh_ > 1.0)
        ? vec::length(mesh->vertices[0]) / km_to_mesh_   // mesh in meters
        : vec::length(mesh->vertices[0]);                 // mesh in km

    switch (config_.camera.look_at_mode) {
        case CameraConfig::LookAtMode::LATLON: {
            look_at_km = CameraModel::latlonaltToXYZ(
                config_.camera.look_at_lat_deg,
                config_.camera.look_at_lon_deg,
                0.0, body_r_km);
            break;
        }
        case CameraConfig::LookAtMode::LIMB_SUN:
        case CameraConfig::LookAtMode::LIMB_PRIMARY: {
            // Compute the point on the visible limb closest to the Sun or
            // the primary body (e.g., Earth for Moon, Jupiter for Europa).
            std::array<double, 3> target_pos_km;
            if (config_.camera.look_at_mode == CameraConfig::LookAtMode::LIMB_SUN) {
                target_pos_km = getSunPosition(et);
            } else {
                target_pos_km = getBodyPosition(primary_body_name_, et);
            }
            const double dist_km = vec::length(cam_pos_km);
            if (dist_km > body_r_km) {
                // Direction from SC to the target body (Sun or primary)
                auto target_from_sc = vec::sub(target_pos_km, cam_pos_km);
                auto target_dir = vec::normalize(target_from_sc);
                // Direction from SC to body center
                auto moon_dir = vec::normalize(vec::scale(cam_pos_km, -1.0));
                // Perpendicular component of target direction (in the plane of
                // the SC–body-center line and the target)
                double proj = vec::dot(target_dir, moon_dir);
                auto perp = vec::sub(target_dir, vec::scale(moon_dir, proj));
                double perp_len = vec::length(perp);
                if (perp_len > 1e-10) {
                    perp = vec::scale(perp, 1.0 / perp_len);
                } else {
                    // Target is directly behind/in-front of the body — pick
                    // an arbitrary perpendicular direction.
                    perp = {0.0, 0.0, 1.0};
                    if (std::abs(vec::dot(perp, moon_dir)) > 0.99)
                        perp = {0.0, 1.0, 0.0};
                    auto tmp = vec::cross(moon_dir, perp);
                    perp = vec::normalize(tmp);
                }
                // Limb half-angle: angle from SC-to-body-center to any tangent
                double limb_angle = std::acos(body_r_km / dist_km);
                // Limb point direction from body center
                auto sc_from_body = vec::normalize(cam_pos_km);
                auto limb_dir = vec::add(
                    vec::scale(sc_from_body, std::cos(limb_angle)),
                    vec::scale(perp, std::sin(limb_angle)));
                look_at_km = vec::scale(vec::normalize(limb_dir), body_r_km);
            }
            // else: camera inside body (shouldn't happen) — fall back to origin
            break;
        }
        default:  // FIXED
            break;
    }

    // Scale look-at to mesh units
    cam_config.look_at = {
        look_at_km[0] * km_to_mesh_,
        look_at_km[1] * km_to_mesh_,
        look_at_km[2] * km_to_mesh_
    };
    camera_.configure(cam_config);
    // If pointing is SPICE-driven, override the axes with the IK+CK lookup for
    // this ET. Position / intrinsics are kept from the configure() above.
    if (cam_config.pointing == CameraConfig::Pointing::SPICE_CK) {
        if (!updateOrientation(et)) {
            std::cerr << "[CameraSimulator] SPICE-CK orientation lookup failed at "
                      << utc_time << " — falling back to look-at" << std::endl;
        }
    }
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

    // Set up celestial body rendering (Sun, primary body) for sky pixels
    constexpr double SUN_RADIUS_KM = 696000.0;

    CelestialBodyParams sun_body{}, primary_body{};
    bool have_sun_body = false, have_primary_body = false;
    std::array<double, 3> sun_dir_from_primary = {0, 0, 0};
    double solar_flux_at_primary = 0.0;

    if (config_.render_sun && sun_dist_km > 0.0) {
        sun_body.pos_km = sun_pos_km;
        sun_body.radius_km = SUN_RADIUS_KM;
        sun_body.is_emitter = true;
        have_sun_body = true;

        double sun_ang_deg = std::asin(SUN_RADIUS_KM / sun_dist_km) * 180.0 / M_PI;
        std::cout << "[CameraSimulator] Sun: distance " << std::fixed << std::setprecision(0)
                  << sun_dist_km << " km, angular diameter "
                  << std::setprecision(3) << 2.0 * sun_ang_deg << " deg" << std::endl;
    }

    if (config_.render_primary && !primary_body_name_.empty()) {
        auto prim_pos_km = getBodyPosition(primary_body_name_, et);
        double prim_dist_km = vec::length(prim_pos_km);
        if (prim_dist_km > 0.0) {
            primary_body.pos_km = prim_pos_km;
            primary_body.radius_km = primary_radius_km_;
            primary_body.albedo = primary_bond_albedo_;
            primary_body.is_emitter = false;
            have_primary_body = true;

            // Sun direction as seen from the primary (for phase computation)
            auto sun_from_prim = vec::sub(sun_pos_km, prim_pos_km);
            double sun_prim_dist = vec::length(sun_from_prim);
            sun_dir_from_primary = vec::normalize(sun_from_prim);

            // Solar flux at the primary's distance from Sun
            double sun_prim_au = sun_prim_dist / KM_PER_AU;
            solar_flux_at_primary = SOLAR_CONSTANT / (sun_prim_au * sun_prim_au);

            // Phase angle of the primary as seen from the camera
            auto cam_to_prim = vec::normalize(vec::sub(prim_pos_km, cam_pos_km));
            double prim_phase_cos = vec::dot(sun_dir_from_primary, vec::scale(cam_to_prim, -1.0));
            double prim_phase_deg = std::acos(std::max(-1.0, std::min(1.0, prim_phase_cos))) * 180.0 / M_PI;

            double prim_ang_deg = std::asin(primary_radius_km_ / prim_dist_km) * 180.0 / M_PI;
            std::cout << "[CameraSimulator] " << primary_body_name_
                      << ": distance " << std::fixed << std::setprecision(0)
                      << prim_dist_km << " km, angular diameter "
                      << std::setprecision(3) << 2.0 * prim_ang_deg << " deg, phase "
                      << std::setprecision(1) << prim_phase_deg << " deg" << std::endl;
        }
    }

    // Load primary-body reflected light (e.g., Earthshine for Moon, Jupitershine
    // for Europa). PTM3D stores this in the "primary_reflected" output field.
    std::vector<float> primary_reflected;
    if (output_reader_.hasPrimaryReflected()) {
        primary_reflected = output_reader_.getPrimaryReflectedAtET(et);
        if (!primary_reflected.empty()) {
            float max_es = *std::max_element(primary_reflected.begin(), primary_reflected.end());
            int n_nonzero = 0;
            for (float v : primary_reflected) if (v > 0.0f) n_nonzero++;
            std::cout << "[CameraSimulator] " << (primary_body_name_.empty() ? "Primary" : primary_body_name_)
                      << "-shine: " << n_nonzero << " illuminated faces, "
                      << "max " << std::setprecision(4) << max_es << " W/m^2" << std::endl;
        }
    }

    // Load PTM3D direct solar irradiance for reflected-light coupling.
    // When available, this replaces independent shadow computation — the
    // camera sim uses PTM3D's validated illumination (finite-disk sun model,
    // terrain occlusion, body self-shadowing) instead of re-deriving it.
    std::vector<float> direct_solar;
    if (output_reader_.hasDirectSolar()) {
        direct_solar = output_reader_.getDirectSolarAtET(et);
        if (!direct_solar.empty()) {
            int n_lit = 0;
            for (float v : direct_solar) if (v > 0.0f) n_lit++;
            std::cout << "[CameraSimulator] PTM3D direct_solar: "
                      << n_lit << " / " << direct_solar.size()
                      << " faces illuminated" << std::endl;
        }
    }

    // Load temperature data if needed
    std::vector<float> temperatures;
    if (config_.render_mode == CameraSimConfig::RenderMode::THERMAL ||
        config_.render_mode == CameraSimConfig::RenderMode::BOTH) {
        if (output_reader_.hasTemperature()) {
            temperatures = output_reader_.getTemperatureAtET(et);
        }
    }

    // Supersampling factor (N×N rays per pixel). 1 = legacy point sampling.
    const int N = std::max(1, config_.supersample);
    const double inv_N = 1.0 / static_cast<double>(N);
    const double n_total_samples = static_cast<double>(N) * static_cast<double>(N);

    int nx = camera_.width();
    int ny = camera_.height();
    size_t n_pixels = static_cast<size_t>(nx) * ny;

    // Allocate output
    std::vector<PixelData> pixels(n_pixels);

    std::cout << "[CameraSimulator] Rendering " << nx << "x" << ny << " at " << utc_time
              << " (ET=" << std::fixed << std::setprecision(3) << et << ")";
    if (N > 1) {
        std::cout << " | supersample " << N << "x" << N << " = "
                  << (N * N) << " rays/pixel";
    }
    std::cout << std::endl;

    // Camera position in mesh frame (for ray origin — same for all sub-rays).
    const std::array<double, 3> cam_origin = camera_.position();
    const double cam_ox = cam_origin[0] - scene_origin_offset_[0];
    const double cam_oy = cam_origin[1] - scene_origin_offset_[1];
    const double cam_oz = cam_origin[2] - scene_origin_offset_[2];
    const double cam_r_local = std::sqrt(cam_ox*cam_ox + cam_oy*cam_oy + cam_oz*cam_oz);
    const float  ray_tnear = static_cast<float>(std::max(1e-6 * cam_r_local, 0.01));

    // Parallelize over output pixels; each pixel sequentially casts N*N sub-rays.
    #pragma omp parallel for schedule(dynamic, 64)
    for (size_t i = 0; i < n_pixels; ++i) {
        const int py = static_cast<int>(i / nx);
        const int px_idx = static_cast<int>(i % nx);

        // Per-pixel accumulators
        double sum_rad = 0.0;          // reflected + primary-body shine (visible)
        double sum_therm = 0.0;        // thermal spectral radiance
        double sum_inc = 0.0;          // incidence angle (deg)
        double sum_emi = 0.0;          // emission angle (deg)
        double sum_pha = 0.0;          // phase angle (deg)
        double sum_temp = 0.0;         // surface temperature (K)
        double sum_dist_km = 0.0;      // camera-to-surface distance (km)
        double sum_lat = 0.0;
        double sum_lon = 0.0;
        int n_surface_hits = 0;        // number of sub-rays that hit the surface
        int n_sun_hits = 0;
        int n_primary_hits = 0;
        double sum_sun_rad = 0.0;      // accumulated sun-disk radiance (across sun-hit samples)
        double sum_primary_rad = 0.0;    // accumulated earth-disk radiance
        int first_face_id = -1;        // for traceability in FACE_IDX output

        for (int sy = 0; sy < N; ++sy) {
            for (int sx = 0; sx < N; ++sx) {
                const double dx = (sx + 0.5) * inv_N - 0.5;  // (-0.5, +0.5) sub-pixel offset
                const double dy = (sy + 0.5) * inv_N - 0.5;
                auto ray = camera_.generateRaySubpixel(px_idx, py, dx, dy);
                const auto& dir = ray.direction;

                RTCRayHit rayhit{};
                rayhit.ray.org_x = static_cast<float>(cam_ox);
                rayhit.ray.org_y = static_cast<float>(cam_oy);
                rayhit.ray.org_z = static_cast<float>(cam_oz);
                rayhit.ray.dir_x = static_cast<float>(dir[0]);
                rayhit.ray.dir_y = static_cast<float>(dir[1]);
                rayhit.ray.dir_z = static_cast<float>(dir[2]);
                rayhit.ray.tnear = ray_tnear;
                rayhit.ray.tfar = 1e20f;
                rayhit.ray.mask = 0xFFFFFFFF;
                rayhit.ray.flags = 0;
                rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;

                RTCIntersectArguments args;
                rtcInitIntersectArguments(&args);
                rtcIntersect1(embree_scene_, &rayhit, &args);

                // Classify hit. A "sky" sub-ray is a true miss, a back-face hit (far-side
                // punch-through due to Embree float precision at the limb), or an
                // out-of-range face ID. The back-face check MUST use the flat (geometric)
                // face normal — that's what Embree actually intersected. The smooth
                // (shading) normal can differ at the terminator and must not be used for
                // culling.
                bool treat_as_sky = (rayhit.hit.geomID == RTC_INVALID_GEOMETRY_ID);
                int face_id = -1;
                std::array<double, 3> flat_normal{};    // geometric — used for culling
                std::array<double, 3> normal{};         // shading — used for photometry (smooth if enabled)
                std::array<double, 3> view_dir{};
                double cos_e_raw = 0.0;
                double face_albedo_smooth = 0.0;
                bool have_smooth_albedo = false;
                if (!treat_as_sky) {
                    face_id = static_cast<int>(rayhit.hit.primID);
                    if (face_id < 0 || face_id >= static_cast<int>(mesh->face_normals.size())) {
                        treat_as_sky = true;
                    } else {
                        flat_normal = mesh->face_normals[face_id];
                        view_dir = vec::normalize(vec::sub(ray.origin, mesh->face_centers[face_id]));

                        // Back-face / limb-punch-through cull using flat normal.
                        const double cos_e_flat = vec::dot(flat_normal, view_dir);
                        if (cos_e_flat <= 1e-4) {
                            treat_as_sky = true;
                        } else {
                            // Keep this hit. Set the shading normal: smooth if enabled,
                            // else flat.
                            if (config_.smooth_shading && smooth_attrs_ready_ &&
                                face_id < static_cast<int>(mesh->faces.size())) {
                                const auto& tri = mesh->faces[face_id];
                                const double hu = static_cast<double>(rayhit.hit.u);
                                const double hv = static_cast<double>(rayhit.hit.v);
                                const double hw = 1.0 - hu - hv;
                                const auto& n0 = vertex_normals_[tri[0]];
                                const auto& n1 = vertex_normals_[tri[1]];
                                const auto& n2 = vertex_normals_[tri[2]];
                                normal = vec::normalize(std::array<double, 3>{
                                    hw * n0[0] + hu * n1[0] + hv * n2[0],
                                    hw * n0[1] + hu * n1[1] + hv * n2[1],
                                    hw * n0[2] + hu * n1[2] + hv * n2[2]});
                                if (!vertex_albedos_.empty()) {
                                    face_albedo_smooth =
                                        hw * vertex_albedos_[tri[0]] +
                                        hu * vertex_albedos_[tri[1]] +
                                        hv * vertex_albedos_[tri[2]];
                                    have_smooth_albedo = true;
                                }
                            } else {
                                normal = flat_normal;
                            }
                            cos_e_raw = vec::dot(normal, view_dir);
                            // If smooth normal grazes below horizon, clamp to near-zero
                            // so photometric model still returns ~0 radiance, but DON'T
                            // reclassify as sky (the face IS visible geometrically).
                            if (cos_e_raw < 0.0) cos_e_raw = 0.0;
                        }
                    }
                }

                if (treat_as_sky) {
                    // Sky sub-ray: check Sun and Earth background.
                    if (have_sun_body) {
                        double L = renderCelestialBody(cam_pos_km, dir, sun_body,
                                                        {0,0,0}, 0.0);
                        if (L > 0.0) {
                            sum_sun_rad += L;
                            n_sun_hits++;
                            continue;
                        }
                    }
                    if (have_primary_body) {
                        double L = renderCelestialBody(cam_pos_km, dir, primary_body,
                                                        sun_dir_from_primary,
                                                        solar_flux_at_primary);
                        if (L > 0.0) {
                            sum_primary_rad += L;
                            n_primary_hits++;
                            continue;
                        }
                    }
                    // Empty sky: contributes 0 radiance to the pixel integral.
                    continue;
                }

                // Valid front-facing surface hit.
                double cos_e = std::min(1.0, cos_e_raw);
                double cos_i = std::max(0.0, vec::dot(normal, sun_dir));

                double cos_alpha = vec::dot(sun_dir, view_dir);
                cos_alpha = std::max(-1.0, std::min(1.0, cos_alpha));
                double alpha_rad = std::acos(cos_alpha);

                // Effective albedo for this sub-ray. Prefer smooth (barycentric-
                // interpolated vertex albedo) if available; fall back to the per-face
                // albedo map; fall back to the default config albedo.
                double effective_albedo;
                bool have_albedo = true;
                if (have_smooth_albedo) {
                    effective_albedo = face_albedo_smooth;
                } else if (!face_albedos_.empty() && face_id < static_cast<int>(face_albedos_.size())) {
                    effective_albedo = face_albedos_[face_id];
                } else {
                    effective_albedo = config_.photometric_params.albedo;
                    have_albedo = false;
                }

                // Reflected radiance — coupled to PTM3D illumination.
                // When PTM3D direct_solar is available, use it for shadow/
                // occlusion determination instead of independent ray tracing.
                // PTM3D's direct_solar = S × cos(i_flat) × shadow_occlusion,
                // already encoding the finite-disk sun model, terrain shadows,
                // and body self-occlusion validated in the illumination model.
                double rad_sample = 0.0;
                if (config_.render_mode == CameraSimConfig::RenderMode::REFLECTED ||
                    config_.render_mode == CameraSimConfig::RenderMode::BOTH) {
                    if (!direct_solar.empty() && face_id < static_cast<int>(direct_solar.size())) {
                        // PTM3D-coupled path: use precomputed illumination
                        const float ptm_flux = direct_solar[face_id];
                        if (ptm_flux > 0.0f && cos_i > 1e-6) {
                            double shadow_frac = std::min(1.0,
                                static_cast<double>(ptm_flux) / (solar_flux * cos_i));
                            if (have_albedo || have_smooth_albedo) {
                                rad_sample = shadow_frac * photometry_.reflectedRadiance(
                                    cos_i, cos_e, alpha_rad,
                                    solar_flux, effective_albedo);
                            } else {
                                rad_sample = shadow_frac * photometry_.reflectedRadiance(
                                    cos_i, cos_e, alpha_rad, solar_flux);
                            }
                        }
                        // else: face is in shadow per PTM3D → rad_sample stays 0
                    } else {
                        // Fallback when PTM3D direct_solar is not available:
                        // geometric cos_i without shadow test.
                        if (have_albedo || have_smooth_albedo) {
                            rad_sample = photometry_.reflectedRadiance(
                                cos_i, cos_e, alpha_rad,
                                solar_flux, effective_albedo);
                        } else {
                            rad_sample = photometry_.reflectedRadiance(
                                cos_i, cos_e, alpha_rad, solar_flux);
                        }
                    }
                }

                // Primary-body reflected light (Lambertian) for this face.
                if (!primary_reflected.empty() && face_id < static_cast<int>(primary_reflected.size())) {
                    float es_flux = primary_reflected[face_id];
                    if (es_flux > 0.0f) {
                        rad_sample += effective_albedo / M_PI * es_flux;
                    }
                }

                // Thermal radiance
                double therm_sample = 0.0;
                float  T_sample = 0.0f;
                if (!temperatures.empty() && face_id < static_cast<int>(temperatures.size())) {
                    T_sample = temperatures[face_id];
                    if ((config_.render_mode == CameraSimConfig::RenderMode::THERMAL ||
                         config_.render_mode == CameraSimConfig::RenderMode::BOTH) &&
                         T_sample > 0.0f) {
                        therm_sample = photometry_.thermalSpectralRadiance(
                            T_sample, cos_e, config_.photometric_params.thermal_wavelength_um);
                    }
                }

                // Accumulate this sub-ray's contribution.
                sum_rad += rad_sample;
                sum_therm += therm_sample;
                sum_emi  += std::acos(cos_e) * 180.0 / M_PI;
                sum_inc  += std::acos(std::max(0.0, cos_i)) * 180.0 / M_PI;
                sum_pha  += alpha_rad * 180.0 / M_PI;
                sum_temp += static_cast<double>(T_sample);
                sum_dist_km += rayhit.ray.tfar / km_to_mesh_;
                if (face_id < static_cast<int>(mesh->latitudes.size())) {
                    sum_lat += mesh->latitudes[face_id];
                    sum_lon += mesh->longitudes[face_id];
                }
                if (first_face_id < 0) first_face_id = face_id;
                n_surface_hits++;
            }  // sx
        }  // sy

        // Write averaged pixel values. Radiances are integrals over the pixel IFOV,
        // so they divide by n_total_samples (sky sub-samples contribute 0).
        // Per-face geometric quantities (angles, temperature, distance, lat/lon) are
        // meaningful only where the ray hit surface — average over n_surface_hits.
        PixelData& px = pixels[i];

        if (n_surface_hits > 0) {
            px.face_index = first_face_id;
            px.radiance         = static_cast<float>(sum_rad   / n_total_samples);
            px.thermal_radiance = static_cast<float>(sum_therm / n_total_samples);
            const double ih = 1.0 / static_cast<double>(n_surface_hits);
            px.incidence_angle  = static_cast<float>(sum_inc  * ih);
            px.emission_angle   = static_cast<float>(sum_emi  * ih);
            px.phase_angle      = static_cast<float>(sum_pha  * ih);
            px.temperature      = static_cast<float>(sum_temp * ih);
            px.distance_km      = static_cast<float>(sum_dist_km * ih);
            px.latitude         = static_cast<float>(sum_lat  * ih);
            px.longitude        = static_cast<float>(sum_lon  * ih);
        } else if (n_sun_hits > 0) {
            // Fully-sky pixel dominated by Sun sub-samples.
            px.face_index = -2;
            px.radiance   = static_cast<float>(sum_sun_rad / n_total_samples);
        } else if (n_primary_hits > 0) {
            // Fully-sky pixel dominated by Earth sub-samples.
            px.face_index = -3;
            px.radiance   = static_cast<float>(sum_primary_rad / n_total_samples);
        } else {
            // Empty sky.
            px.face_index = -1;
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

        // ----- Camera intrinsics -----
        double focal_mm = camera_.focalLengthMm();
        double pitch_x  = camera_.pixelPitchXMm();
        double pitch_y  = camera_.pixelPitchYMm();
        fits_update_key(fptr, TDOUBLE, "FOCLEN",  &focal_mm, "Focal length (mm)",       &status);
        fits_update_key(fptr, TDOUBLE, "PIXPITX", &pitch_x,  "Pixel pitch X (mm/px)",   &status);
        fits_update_key(fptr, TDOUBLE, "PIXPITY", &pitch_y,  "Pixel pitch Y (mm/px)",   &status);

        // Brown-Conrady distortion coefficients (always written; zero = pinhole).
        char dist_buf[] = "BROWN-CONRADY";
        fits_update_key(fptr, TSTRING, "DISTMODL", dist_buf,
                        "Distortion model", &status);
        double k1 = camera_.k1(), k2 = camera_.k2(), k3 = camera_.k3();
        double p1 = camera_.p1(), p2 = camera_.p2();
        fits_update_key(fptr, TDOUBLE, "DISTK1", &k1, "Radial distortion k1",     &status);
        fits_update_key(fptr, TDOUBLE, "DISTK2", &k2, "Radial distortion k2",     &status);
        fits_update_key(fptr, TDOUBLE, "DISTK3", &k3, "Radial distortion k3",     &status);
        fits_update_key(fptr, TDOUBLE, "DISTP1", &p1, "Tangential distortion p1", &status);
        fits_update_key(fptr, TDOUBLE, "DISTP2", &p2, "Tangential distortion p2", &status);

        // SPICE-IK metadata (when SPICE_CK pointing is active).
        if (!camera_.instrumentFrame().empty()) {
            char frm_buf[64];
            strncpy(frm_buf, camera_.instrumentFrame().c_str(), sizeof(frm_buf)-1);
            frm_buf[sizeof(frm_buf)-1] = '\0';
            fits_update_key(fptr, TSTRING, "INSTFRM", frm_buf,
                            "SPICE instrument frame",  &status);
            int nid = camera_.instrumentId();
            fits_update_key(fptr, TINT, "INSTID", &nid,
                            "SPICE NAIF instrument ID", &status);
        }

        // ----- Body-fixed tangent WCS (LON--TAN / LAT--TAN) -----
        // Derive CRVAL/CRPIX/CD from the per-pixel (lat, lon) data. Use a wider
        // central-difference stencil so the CD matrix averages across discrete
        // mesh-face boundaries (coarse meshes have 4–10 px per face, which makes
        // 1-px forward differences unreliable). The stencil spans a fraction of
        // the image size but is clamped so it stays away from the limb.
        const int ix0 = static_cast<int>(std::lround(camera_.principalPointX() - 0.5));
        const int iy0 = static_cast<int>(std::lround(camera_.principalPointY() - 0.5));
        const int stencil = std::max(1, std::min(nx, ny) / 32);  // ~3-32 px

        auto pixelAt = [&](int px, int py) -> const PixelData* {
            if (px < 0 || px >= nx || py < 0 || py >= ny) return nullptr;
            const PixelData& p = pixels[static_cast<size_t>(py) * nx + px];
            if (p.face_index < 0) return nullptr;  // sky miss
            return &p;
        };
        const PixelData* pC = pixelAt(ix0, iy0);
        const PixelData* pR = pixelAt(ix0 + stencil, iy0);
        const PixelData* pL = pixelAt(ix0 - stencil, iy0);
        const PixelData* pU = pixelAt(ix0, iy0 - stencil);
        const PixelData* pD = pixelAt(ix0, iy0 + stencil);

        if (pC && pR && pL && pU && pD) {
            auto unwrap = [](double lon, double ref) {
                double d = lon - ref;
                if (d >  180.0) lon -= 360.0;
                if (d < -180.0) lon += 360.0;
                return lon;
            };
            const double lonC = pC->longitude;
            const double lonR = unwrap(pR->longitude, lonC);
            const double lonL = unwrap(pL->longitude, lonC);
            const double lonU = unwrap(pU->longitude, lonC);
            const double lonD = unwrap(pD->longitude, lonC);

            double crval1 = lonC;
            double crval2 = pC->latitude;
            // CRPIX in 1-indexed FITS convention; principalPointX/Y is 0-indexed
            // with pixel-center convention 0.5, so the conversion is +0.5.
            double crpix1 = camera_.principalPointX() + 0.5;
            double crpix2 = camera_.principalPointY() + 0.5;
            char ctype1[] = "LON--TAN";
            char ctype2[] = "LAT--TAN";
            char cunit[]  = "deg";

            // Central differences averaged over 2*stencil pixels.
            const double inv_stencil2 = 0.5 / static_cast<double>(stencil);
            double cd1_1 = (lonR - lonL) * inv_stencil2;
            double cd1_2 = (lonU - lonD) * inv_stencil2;  // image +y = down; up - down
            double cd2_1 = (pR->latitude - pL->latitude) * inv_stencil2;
            double cd2_2 = (pU->latitude - pD->latitude) * inv_stencil2;

            fits_update_key(fptr, TSTRING, "CTYPE1", ctype1, "Gnomonic (TAN) longitude", &status);
            fits_update_key(fptr, TSTRING, "CTYPE2", ctype2, "Gnomonic (TAN) latitude",  &status);
            fits_update_key(fptr, TSTRING, "CUNIT1", cunit,  "Units",                    &status);
            fits_update_key(fptr, TSTRING, "CUNIT2", cunit,  "Units",                    &status);
            fits_update_key(fptr, TDOUBLE, "CRVAL1", &crval1, "Longitude of reference",  &status);
            fits_update_key(fptr, TDOUBLE, "CRVAL2", &crval2, "Latitude of reference",   &status);
            fits_update_key(fptr, TDOUBLE, "CRPIX1", &crpix1, "Reference pixel X (1-indexed)", &status);
            fits_update_key(fptr, TDOUBLE, "CRPIX2", &crpix2, "Reference pixel Y (1-indexed)", &status);
            fits_update_key(fptr, TDOUBLE, "CD1_1",  &cd1_1, "Transformation matrix",    &status);
            fits_update_key(fptr, TDOUBLE, "CD1_2",  &cd1_2, "Transformation matrix",    &status);
            fits_update_key(fptr, TDOUBLE, "CD2_1",  &cd2_1, "Transformation matrix",    &status);
            fits_update_key(fptr, TDOUBLE, "CD2_2",  &cd2_2, "Transformation matrix",    &status);
            int stncl = stencil;
            fits_update_key(fptr, TINT, "WCSSTNCL", &stncl,
                            "CD matrix stencil (pixels)", &status);
        }
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
