#include "camera_yaml.h"

#include <iostream>

#ifdef HAVE_YAML_CPP
#include <yaml-cpp/yaml.h>
#endif

namespace ptm {

#ifdef HAVE_YAML_CPP

namespace {

template <typename T>
T getOr(const YAML::Node& node, const std::string& key, const T& defaultVal) {
    if (node && node[key]) return node[key].as<T>();
    return defaultVal;
}

template <typename T, size_t N>
bool readArray(const YAML::Node& node, const std::string& key,
               std::array<T, N>& out) {
    if (!node || !node[key]) return false;
    const auto& arr = node[key];
    if (!arr.IsSequence() || arr.size() != N) return false;
    for (size_t i = 0; i < N; ++i) out[i] = arr[i].as<T>();
    return true;
}

} // namespace

bool loadCameraYaml(const std::string& filepath, CameraSimConfig& config) {
    try {
        YAML::Node root = YAML::LoadFile(filepath);
        if (!root["camera"]) {
            std::cerr << "[camera_yaml] Missing top-level 'camera' key in "
                      << filepath << std::endl;
            return false;
        }
        const auto& cam = root["camera"];

        // ----- Intrinsics -----
        config.camera.focal_length_mm = getOr(cam, "focal_length_mm",
                                              config.camera.focal_length_mm);

        if (cam["sensor_mm"] && cam["sensor_mm"].IsSequence()
                             && cam["sensor_mm"].size() == 2) {
            config.camera.sensor_width_mm  = cam["sensor_mm"][0].as<double>();
            config.camera.sensor_height_mm = cam["sensor_mm"][1].as<double>();
        }
        if (cam["pixel_pitch_mm"] && cam["pixel_pitch_mm"].IsSequence()
                                   && cam["pixel_pitch_mm"].size() == 2) {
            config.camera.pixel_pitch_x_mm = cam["pixel_pitch_mm"][0].as<double>();
            config.camera.pixel_pitch_y_mm = cam["pixel_pitch_mm"][1].as<double>();
        }

        config.camera.fov_x_deg = getOr(cam, "fov_x_deg", config.camera.fov_x_deg);
        config.camera.nx        = getOr(cam, "nx", config.camera.nx);
        config.camera.ny        = getOr(cam, "ny", config.camera.ny);

        if (cam["principal_point"] && cam["principal_point"].IsSequence()
                                    && cam["principal_point"].size() == 2) {
            config.camera.cx = cam["principal_point"][0].as<double>();
            config.camera.cy = cam["principal_point"][1].as<double>();
        }

        // ----- Distortion -----
        if (cam["distortion"]) {
            const auto& dist = cam["distortion"];
            const std::string model = getOr<std::string>(dist, "model", "brown_conrady");
            if (model != "brown_conrady" && model != "pinhole") {
                std::cerr << "[camera_yaml] Unsupported distortion model '"
                          << model << "' (only 'brown_conrady' and 'pinhole' are "
                          << "supported)." << std::endl;
                return false;
            }
            config.camera.k1 = getOr(dist, "k1", config.camera.k1);
            config.camera.k2 = getOr(dist, "k2", config.camera.k2);
            config.camera.k3 = getOr(dist, "k3", config.camera.k3);
            config.camera.p1 = getOr(dist, "p1", config.camera.p1);
            config.camera.p2 = getOr(dist, "p2", config.camera.p2);
        }

        // ----- Pointing -----
        if (cam["pointing"]) {
            const auto& pt = cam["pointing"];
            const std::string mode = getOr<std::string>(pt, "mode", "look_at");
            if (mode == "look_at") {
                config.camera.pointing = CameraConfig::Pointing::LOOK_AT;
                readArray<double, 3>(pt, "look_at", config.camera.look_at);
                readArray<double, 3>(pt, "up",      config.camera.up_vector);
                config.camera.use_direction =
                    getOr(pt, "use_direction", config.camera.use_direction);
                // Optional dynamic up convention (CLI flags still override).
                const std::string up_mode =
                    getOr<std::string>(pt, "up_mode", std::string{});
                if (up_mode == "explicit") {
                    config.camera.up_mode = CameraConfig::UpMode::EXPLICIT;
                } else if (up_mode == "velocity_pole") {
                    config.camera.up_mode = CameraConfig::UpMode::VELOCITY_POLE;
                } else if (up_mode == "boresight_pole") {
                    config.camera.up_mode = CameraConfig::UpMode::BORESIGHT_POLE;
                } else if (!up_mode.empty()) {
                    std::cerr << "[camera_yaml] Unknown pointing.up_mode '"
                              << up_mode << "' (expected: explicit | "
                              << "velocity_pole | boresight_pole)" << std::endl;
                    return false;
                }
            } else if (mode == "quaternion") {
                config.camera.pointing = CameraConfig::Pointing::QUATERNION;
                if (!readArray<double, 4>(pt, "quaternion",
                                          config.camera.quaternion)) {
                    std::cerr << "[camera_yaml] pointing.mode=quaternion requires "
                              << "a 4-element 'quaternion: [w, x, y, z]' entry."
                              << std::endl;
                    return false;
                }
            } else if (mode == "rotation_matrix") {
                config.camera.pointing = CameraConfig::Pointing::ROTATION_MATRIX;
                if (pt["rotation_matrix"] && pt["rotation_matrix"].IsSequence()
                                          && pt["rotation_matrix"].size() == 3) {
                    for (size_t r = 0; r < 3; ++r) {
                        const auto& row = pt["rotation_matrix"][r];
                        if (!row.IsSequence() || row.size() != 3) {
                            std::cerr << "[camera_yaml] rotation_matrix must be 3x3"
                                      << std::endl;
                            return false;
                        }
                        for (size_t c = 0; c < 3; ++c) {
                            config.camera.rotation_matrix[r][c] =
                                row[c].as<double>();
                        }
                    }
                } else {
                    std::cerr << "[camera_yaml] pointing.mode=rotation_matrix "
                              << "requires a 3x3 'rotation_matrix'." << std::endl;
                    return false;
                }
            } else if (mode == "spice_ck") {
                config.camera.pointing = CameraConfig::Pointing::SPICE_CK;
                config.camera.instrument_name =
                    getOr<std::string>(pt, "instrument_name",
                                       config.camera.instrument_name);
            } else {
                std::cerr << "[camera_yaml] Unknown pointing.mode '" << mode
                          << "' (expected: look_at | quaternion | "
                          << "rotation_matrix | spice_ck)" << std::endl;
                return false;
            }
        }

        // ----- Camera position (optional — trajectory usually sets this) -----
        if (cam["position_km"] && cam["position_km"].IsSequence()
                                && cam["position_km"].size() == 3) {
            config.camera.position = {
                cam["position_km"][0].as<double>(),
                cam["position_km"][1].as<double>(),
                cam["position_km"][2].as<double>(),
            };
        }
        if (cam["latlon_alt"] && cam["latlon_alt"].IsSequence()
                               && cam["latlon_alt"].size() == 3) {
            config.camera.use_latlon     = true;
            config.camera.latitude_deg   = cam["latlon_alt"][0].as<double>();
            config.camera.longitude_deg  = cam["latlon_alt"][1].as<double>();
            config.camera.altitude_km    = cam["latlon_alt"][2].as<double>();
        }

        // ----- SPICE (IK/CK) -----
        if (cam["spice"]) {
            const auto& sp = cam["spice"];
            config.camera.instrument_name =
                getOr<std::string>(sp, "instrument_name",
                                   config.camera.instrument_name);
            config.ik_kernel =
                getOr<std::string>(sp, "ik_kernel", config.ik_kernel);
            config.ck_kernel =
                getOr<std::string>(sp, "ck_kernel", config.ck_kernel);
            config.target_body =
                getOr<std::string>(sp, "target_body", config.target_body);
            config.target_frame =
                getOr<std::string>(sp, "target_frame", config.target_frame);
            config.primary_body =
                getOr<std::string>(sp, "primary_body", config.primary_body);
            config.primary_bond_albedo =
                getOr(sp, "primary_bond_albedo", config.primary_bond_albedo);
        }

        // ----- Photometry (optional) -----
        if (cam["photometry"]) {
            const auto& ph = cam["photometry"];
            const std::string refl =
                getOr<std::string>(ph, "reflectance", std::string{});
            if (!refl.empty()) {
                if      (refl == "lambertian")       config.reflectance_type = PhotometricModel::ReflectanceType::LAMBERTIAN;
                else if (refl == "lommel_seeliger")  config.reflectance_type = PhotometricModel::ReflectanceType::LOMMEL_SEELIGER;
                else if (refl == "minnaert")         config.reflectance_type = PhotometricModel::ReflectanceType::MINNAERT;
                else if (refl == "hapke")            config.reflectance_type = PhotometricModel::ReflectanceType::HAPKE;
                else {
                    std::cerr << "[camera_yaml] Unknown reflectance model '"
                              << refl << "'" << std::endl;
                    return false;
                }
            }
            config.photometric_params.albedo =
                getOr(ph, "albedo", config.photometric_params.albedo);
            config.photometric_params.emissivity =
                getOr(ph, "emissivity", config.photometric_params.emissivity);
            config.photometric_params.ls_w =
                getOr(ph, "ls_w", config.photometric_params.ls_w);
            config.photometric_params.minnaert_k =
                getOr(ph, "minnaert_k", config.photometric_params.minnaert_k);

            // Hapke parameters. w is the single-scattering albedo (calibrated to
            // the body's geometric albedo via the opposition-excluded disk
            // integral); B0/h/g/theta_bar are the phase-dependence structural
            // params (Fermi's class-based provisional defaults — override here).
            config.photometric_params.hapke_w =
                getOr(ph, "hapke_w", config.photometric_params.hapke_w);
            config.photometric_params.hapke_B0 =
                getOr(ph, "hapke_B0", config.photometric_params.hapke_B0);
            config.photometric_params.hapke_h =
                getOr(ph, "hapke_h", config.photometric_params.hapke_h);
            config.photometric_params.hapke_g =
                getOr(ph, "hapke_g", config.photometric_params.hapke_g);
            config.photometric_params.hapke_theta_bar =
                getOr(ph, "hapke_theta_bar",
                      config.photometric_params.hapke_theta_bar);
        }

        std::cout << "[camera_yaml] Loaded camera definition from "
                  << filepath << std::endl;
        return true;
    } catch (const YAML::Exception& e) {
        std::cerr << "[camera_yaml] YAML parse error in " << filepath << ": "
                  << e.what() << std::endl;
        return false;
    }
}

#else  // !HAVE_YAML_CPP

bool loadCameraYaml(const std::string& filepath, CameraSimConfig& /*config*/) {
    std::cerr << "[camera_yaml] yaml-cpp not available at build time — cannot "
              << "load " << filepath << std::endl;
    return false;
}

#endif

} // namespace ptm
