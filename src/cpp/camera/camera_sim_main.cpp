#include "camera_simulator.h"
#include "camera_yaml.h"
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>

namespace fs = std::filesystem;

void printUsage(const char* prog) {
    std::cout << "PTM3D Camera Simulator\n\n"
              << "Usage: " << prog << " [options]\n\n"
              << "Required:\n"
              << "  --ptm-output DIR          PTM3D binary output directory\n"
              << "  --time UTC                Render time (UTC string), can specify multiple\n"
              << "\n"
              << "Camera (Generic mode):\n"
              << "  --camera-yaml FILE        Load camera definition from YAML (intrinsics,\n"
              << "                             distortion, pointing, SPICE IK). CLI flags\n"
              << "                             below still override YAML values.\n"
              << "  --cam-pos X Y Z           Camera position in body-fixed frame (km)\n"
              << "  --cam-latlon LAT LON ALT  Camera position as lat(deg) lon(deg) alt(km)\n"
              << "  --look-at X Y Z           Look-at point (km) [default: 0 0 0]\n"
              << "  --look-at-latlon LAT LON  Track a surface point at (lat, lon) degrees.\n"
              << "                             The look-at XYZ is recomputed each frame.\n"
              << "  --look-at-limb TARGET     Track the limb point closest to TARGET,\n"
              << "                             where TARGET is 'sun' or 'primary' (or\n"
              << "                             'earth' as alias). For eclipse animations.\n"
              << "  --up X Y Z                Up vector [default: 0 0 1]\n"
              << "  --up-velocity-pole        Per-frame up = body axis perpendicular to the\n"
              << "                             spacecraft velocity and closest to the spin-axis\n"
              << "                             positive direction (north pole). Overrides --up.\n"
              << "  --quaternion W X Y Z      Override pointing with quaternion (camera→body)\n"
              << "  --focal-mm F              Focal length in mm [default: 50]\n"
              << "  --sensor-mm W H           Sensor dimensions in mm [default: 36 24]\n"
              << "  --pixel-pitch PX PY       Anisotropic pixel pitch in mm (overrides sensor)\n"
              << "  --pixels NX NY            Image dimensions [default: 1024 1024]\n"
              << "  --fov-deg F               Horizontal FOV in degrees (overrides focal length)\n"
              << "  --principal-point CX CY   Principal point in pixels (default: image center)\n"
              << "  --distortion K1 K2 K3 P1 P2\n"
              << "                             Brown-Conrady distortion (OpenCV convention).\n"
              << "                             Defaults 0 → pure pinhole. Enable to simulate\n"
              << "                             radial/tangential lens distortion.\n"
              << "  --supersample N           Sub-pixel supersampling (N*N rays/pixel).\n"
              << "                             1 = legacy point sampling (bit-identical).\n"
              << "                             2 (default) = 4 rays/pixel; removes face-\n"
              << "                             aliasing. 4 = 16 rays/pixel (high quality).\n"
              << "  --smooth-shading          Interpolate normals & albedo across each triangle\n"
              << "                             using barycentric hit coords (Phong-style).\n"
              << "                             [default: on]\n"
              << "  --no-smooth-shading       Disable smooth shading (use flat per-face shading).\n"
              << "\n"
              << "Photometry:\n"
              << "  --photometric MODEL       lambertian|lommel_seeliger|minnaert|hapke [default: lommel_seeliger]\n"
              << "  --render-mode MODE        reflected|thermal|both [default: reflected]\n"
              << "  --albedo A                Surface albedo [default: 0.12]\n"
              << "  --albedo-map FILE         Per-face albedo map (binary: uint64 count + double[count])\n"
              << "  --ls-w W                  Lommel-Seeliger single-scattering albedo [default: 0.25]\n"
              << "  --minnaert-k K            Minnaert limb-darkening parameter [default: 0.5]\n"
              << "  --hapke W B0 H G THETA    Hapke parameters [default: 0.25 1.0 0.05 -0.3 20.0]\n"
              << "  --emissivity E            Thermal emissivity [default: 0.95]\n"
              << "  --thermal-wavelength UM   Thermal wavelength in microns [default: 10.0]\n"
              << "  --emission-phase TYPE     lambertian|flat|polynomial [default: lambertian]\n"
              << "\n"
              << "Trajectory:\n"
              << "  --trajectory FILE         Trajectory file (SPK, HORIZONS, or PROP_MAN)\n"
              << "  --traj-format FMT         spk|horizons|propman [default: spk]\n"
              << "  --traj-spacecraft ID      NAIF spacecraft ID [default: -1024]\n"
              << "  --traj-epoch UTC          Epoch for PROP_MAN time offset\n"
              << "\n"
              << "SPICE:\n"
              << "  --meta-kernel FILE        SPICE meta-kernel file\n"
              << "  --kernels-path DIR        SPICE kernels directory\n"
              << "  --target-body NAME        Target body [default: MOON]\n"
              << "  --target-frame FRAME      Body-fixed frame [default: IAU_MOON]\n"
              << "  --ik-kernel FILE          SPICE Instrument Kernel (IK); populates FOV\n"
              << "                             and boresight via getfov_c.\n"
              << "  --ck-kernel FILE          SPICE C-Kernel (CK); drives per-frame\n"
              << "                             camera orientation via pxform_c.\n"
              << "  --instrument NAME         Instrument name for SPICE_CK pointing mode\n"
              << "                             (e.g. ARTEMIS_II_CAM, LRO_LROCNACL).\n"
              << "\n"
              << "Background:\n"
              << "  --render-sun              Render the Sun disk in sky pixels\n"
              << "  --render-primary          Render the primary body (auto-detected planet\n"
              << "                             for moons, e.g. Earth for Moon, Jupiter for\n"
              << "                             Europa) with correct phase in sky pixels.\n"
              << "  --render-earth            Alias for --render-primary (backward compat)\n"
              << "  --primary-body NAME       Override auto-detected primary body name\n"
              << "  --primary-albedo A        Override Bond albedo of primary body\n"
              << "\n"
              << "Output:\n"
              << "  --output-dir DIR          Output directory [default: camera_output]\n"
              << "  --fits                    Write FITS output [default: on]\n"
              << "  --pgm                     Write PGM grayscale image\n"
              << "  --no-fits                 Disable FITS output\n"
              << std::endl;
}

int main(int argc, char* argv[]) {
    ptm::CameraSimConfig config;

    // First pass: locate --camera-yaml and apply it before any CLI overrides.
    // This lets the user point to a reusable instrument definition and then
    // tweak individual fields on the command line.
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--camera-yaml" && i+1 < argc) {
            config.camera_yaml_file = argv[i+1];
            if (!ptm::loadCameraYaml(config.camera_yaml_file, config)) {
                std::cerr << "Failed to load camera YAML: "
                          << config.camera_yaml_file << std::endl;
                return 1;
            }
            break;
        }
    }

    // Parse command-line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        }
        else if (arg == "--camera-yaml" && i+1 < argc) {
            // Already handled in the first pass; skip its value.
            ++i;
        }
        else if (arg == "--ptm-output" && i+1 < argc) {
            config.ptm_output_dir = argv[++i];
        }
        else if (arg == "--time" && i+1 < argc) {
            config.render_times.push_back(argv[++i]);
        }
        // Camera position
        else if (arg == "--cam-pos" && i+3 < argc) {
            config.camera.position[0] = std::stod(argv[++i]);
            config.camera.position[1] = std::stod(argv[++i]);
            config.camera.position[2] = std::stod(argv[++i]);
        }
        else if (arg == "--cam-latlon" && i+3 < argc) {
            config.camera.use_latlon = true;
            config.camera.latitude_deg = std::stod(argv[++i]);
            config.camera.longitude_deg = std::stod(argv[++i]);
            config.camera.altitude_km = std::stod(argv[++i]);
        }
        else if (arg == "--look-at" && i+3 < argc) {
            config.camera.look_at_mode = ptm::CameraConfig::LookAtMode::FIXED;
            config.camera.look_at[0] = std::stod(argv[++i]);
            config.camera.look_at[1] = std::stod(argv[++i]);
            config.camera.look_at[2] = std::stod(argv[++i]);
        }
        else if (arg == "--look-at-latlon" && i+2 < argc) {
            config.camera.look_at_mode = ptm::CameraConfig::LookAtMode::LATLON;
            config.camera.look_at_lat_deg = std::stod(argv[++i]);
            config.camera.look_at_lon_deg = std::stod(argv[++i]);
        }
        else if (arg == "--look-at-limb" && i+1 < argc) {
            std::string target = argv[++i];
            if (target == "sun") {
                config.camera.look_at_mode = ptm::CameraConfig::LookAtMode::LIMB_SUN;
            } else if (target == "primary" || target == "earth") {
                config.camera.look_at_mode = ptm::CameraConfig::LookAtMode::LIMB_PRIMARY;
            } else {
                std::cerr << "Unknown --look-at-limb target: " << target
                          << " (expected: sun, primary)" << std::endl;
            }
        }
        else if (arg == "--up" && i+3 < argc) {
            config.camera.up_vector[0] = std::stod(argv[++i]);
            config.camera.up_vector[1] = std::stod(argv[++i]);
            config.camera.up_vector[2] = std::stod(argv[++i]);
        }
        else if (arg == "--up-velocity-pole") {
            config.camera.up_mode = ptm::CameraConfig::UpMode::VELOCITY_POLE;
        }
        else if (arg == "--focal-mm" && i+1 < argc) {
            config.camera.focal_length_mm = std::stod(argv[++i]);
        }
        else if (arg == "--sensor-mm" && i+2 < argc) {
            config.camera.sensor_width_mm = std::stod(argv[++i]);
            config.camera.sensor_height_mm = std::stod(argv[++i]);
        }
        else if (arg == "--pixels" && i+2 < argc) {
            config.camera.nx = std::stoi(argv[++i]);
            config.camera.ny = std::stoi(argv[++i]);
        }
        else if (arg == "--fov-deg" && i+1 < argc) {
            config.camera.fov_x_deg = std::stod(argv[++i]);
        }
        else if (arg == "--pixel-pitch" && i+2 < argc) {
            config.camera.pixel_pitch_x_mm = std::stod(argv[++i]);
            config.camera.pixel_pitch_y_mm = std::stod(argv[++i]);
        }
        else if (arg == "--principal-point" && i+2 < argc) {
            config.camera.cx = std::stod(argv[++i]);
            config.camera.cy = std::stod(argv[++i]);
        }
        else if (arg == "--distortion" && i+5 < argc) {
            config.camera.k1 = std::stod(argv[++i]);
            config.camera.k2 = std::stod(argv[++i]);
            config.camera.k3 = std::stod(argv[++i]);
            config.camera.p1 = std::stod(argv[++i]);
            config.camera.p2 = std::stod(argv[++i]);
        }
        else if (arg == "--quaternion" && i+4 < argc) {
            config.camera.pointing = ptm::CameraConfig::Pointing::QUATERNION;
            config.camera.quaternion[0] = std::stod(argv[++i]);
            config.camera.quaternion[1] = std::stod(argv[++i]);
            config.camera.quaternion[2] = std::stod(argv[++i]);
            config.camera.quaternion[3] = std::stod(argv[++i]);
        }
        else if (arg == "--instrument" && i+1 < argc) {
            config.camera.instrument_name = argv[++i];
        }
        else if (arg == "--ik-kernel" && i+1 < argc) {
            config.ik_kernel = argv[++i];
        }
        else if (arg == "--ck-kernel" && i+1 < argc) {
            config.ck_kernel = argv[++i];
        }
        // Photometry
        else if (arg == "--photometric" && i+1 < argc) {
            std::string model = argv[++i];
            if (model == "lambertian") config.reflectance_type = ptm::PhotometricModel::ReflectanceType::LAMBERTIAN;
            else if (model == "lommel_seeliger") config.reflectance_type = ptm::PhotometricModel::ReflectanceType::LOMMEL_SEELIGER;
            else if (model == "minnaert") config.reflectance_type = ptm::PhotometricModel::ReflectanceType::MINNAERT;
            else if (model == "hapke") config.reflectance_type = ptm::PhotometricModel::ReflectanceType::HAPKE;
            else std::cerr << "Unknown photometric model: " << model << std::endl;
        }
        else if (arg == "--render-mode" && i+1 < argc) {
            std::string mode = argv[++i];
            if (mode == "reflected") config.render_mode = ptm::CameraSimConfig::RenderMode::REFLECTED;
            else if (mode == "thermal") config.render_mode = ptm::CameraSimConfig::RenderMode::THERMAL;
            else if (mode == "both") config.render_mode = ptm::CameraSimConfig::RenderMode::BOTH;
        }
        else if (arg == "--albedo" && i+1 < argc) {
            config.photometric_params.albedo = std::stod(argv[++i]);
        }
        else if (arg == "--albedo-map" && i+1 < argc) {
            config.albedo_map_file = argv[++i];
        }
        else if (arg == "--ls-w" && i+1 < argc) {
            config.photometric_params.ls_w = std::stod(argv[++i]);
        }
        else if (arg == "--minnaert-k" && i+1 < argc) {
            config.photometric_params.minnaert_k = std::stod(argv[++i]);
        }
        else if (arg == "--hapke" && i+5 < argc) {
            config.photometric_params.hapke_w = std::stod(argv[++i]);
            config.photometric_params.hapke_B0 = std::stod(argv[++i]);
            config.photometric_params.hapke_h = std::stod(argv[++i]);
            config.photometric_params.hapke_g = std::stod(argv[++i]);
            config.photometric_params.hapke_theta_bar = std::stod(argv[++i]);
        }
        else if (arg == "--emissivity" && i+1 < argc) {
            config.photometric_params.emissivity = std::stod(argv[++i]);
        }
        else if (arg == "--thermal-wavelength" && i+1 < argc) {
            config.photometric_params.thermal_wavelength_um = std::stod(argv[++i]);
        }
        else if (arg == "--emission-phase" && i+1 < argc) {
            std::string type = argv[++i];
            if (type == "lambertian") config.photometric_params.emission_phase = ptm::PhotometricModel::EmissionPhaseType::LAMBERTIAN;
            else if (type == "flat") config.photometric_params.emission_phase = ptm::PhotometricModel::EmissionPhaseType::FLAT;
            else if (type == "polynomial") config.photometric_params.emission_phase = ptm::PhotometricModel::EmissionPhaseType::POLYNOMIAL;
        }
        // Trajectory
        else if (arg == "--trajectory" && i+1 < argc) {
            config.trajectory_file = argv[++i];
        }
        else if (arg == "--traj-format" && i+1 < argc) {
            config.trajectory_format = argv[++i];
        }
        else if (arg == "--traj-spacecraft" && i+1 < argc) {
            config.trajectory_spacecraft = argv[++i];
        }
        else if (arg == "--traj-epoch" && i+1 < argc) {
            config.trajectory_epoch_utc = argv[++i];
        }
        // SPICE
        else if (arg == "--meta-kernel" && i+1 < argc) {
            config.meta_kernel = argv[++i];
        }
        else if (arg == "--kernels-path" && i+1 < argc) {
            config.kernels_path = argv[++i];
        }
        else if (arg == "--target-body" && i+1 < argc) {
            config.target_body = argv[++i];
        }
        else if (arg == "--target-frame" && i+1 < argc) {
            config.target_frame = argv[++i];
        }
        // Background celestial bodies
        else if (arg == "--render-sun") {
            config.render_sun = true;
        }
        else if (arg == "--render-primary" || arg == "--render-earth") {
            config.render_primary = true;
        }
        else if (arg == "--primary-body" && i+1 < argc) {
            config.primary_body = argv[++i];
        }
        else if (arg == "--primary-albedo" && i+1 < argc) {
            config.primary_bond_albedo = std::stod(argv[++i]);
        }
        else if (arg == "--supersample" && i+1 < argc) {
            int n = std::stoi(argv[++i]);
            if (n < 1 || n > 8) {
                std::cerr << "Error: --supersample must be in [1, 8]; got " << n << std::endl;
                return 1;
            }
            config.supersample = n;
        }
        else if (arg == "--smooth-shading") {
            config.smooth_shading = true;
        }
        else if (arg == "--no-smooth-shading") {
            config.smooth_shading = false;
        }
        // Output
        else if (arg == "--output-dir" && i+1 < argc) {
            config.output_dir = argv[++i];
        }
        else if (arg == "--fits") {
            config.output_fits = true;
        }
        else if (arg == "--no-fits") {
            config.output_fits = false;
        }
        else if (arg == "--pgm") {
            config.output_png = true;
        }
        else {
            std::cerr << "Unknown argument: " << arg << std::endl;
        }
    }

    // Validate
    if (config.ptm_output_dir.empty()) {
        std::cerr << "Error: --ptm-output is required" << std::endl;
        printUsage(argv[0]);
        return 1;
    }
    if (config.render_times.empty()) {
        std::cerr << "Error: at least one --time is required" << std::endl;
        printUsage(argv[0]);
        return 1;
    }

    // Default kernels path
    if (config.kernels_path.empty()) {
        config.kernels_path = "/Users/paha3326/software/cspice/data";
    }

    // Run
    ptm::CameraSimulator simulator;
    if (!simulator.initialize(config)) {
        std::cerr << "Initialization failed" << std::endl;
        return 1;
    }

    simulator.renderAllFrames();

    std::cout << "\nDone." << std::endl;
    return 0;
}
