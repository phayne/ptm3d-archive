#include "albedo_manager.h"
#include "spatial_map.h"
#include "../mesh/mesh_utils.h"
#include "../thermal/planetary_properties.h"

#include <fstream>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <filesystem>

namespace ptm {

AlbedoManager::AlbedoManager()
    : default_albedo_(0.1),
      has_map_(false),
      initialized_(false),
      map_width_(0),
      map_height_(0),
      lat_min_(-90.0),
      lat_max_(90.0),
      lon_min_(0.0),
      lon_max_(360.0),
      map_min_(0.0),
      map_max_(1.0) {
}

AlbedoManager::~AlbedoManager() = default;

double AlbedoManager::getCanonicalAlbedo(const std::string& body_name) {
    // Use the canonical values from planetary_properties.h
    std::string lower_name = body_name;
    std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(), ::tolower);

    if (lower_name == "moon") {
        return 0.12;  // From planetary_properties.h getMoonProperties()
    } else if (lower_name == "europa") {
        return 0.5;   // From planetary_properties.h getEuropaProperties()
    } else if (lower_name == "mercury") {
        return 0.07;  // From planetary_properties.h getMercuryProperties()
    } else if (lower_name == "mars") {
        return 0.25;  // Mars Bond albedo
    } else if (lower_name == "enceladus") {
        return 0.81;  // Enceladus is very bright
    } else if (lower_name == "titan") {
        return 0.22;  // Titan Bond albedo
    } else if (lower_name == "ceres") {
        return 0.09;  // Ceres Bond albedo
    } else {
        // Default fallback
        return 0.1;
    }
}

bool AlbedoManager::initialize(const std::string& body_name, const std::string& albedo_maps_dir) {
    body_name_ = body_name;
    maps_dir_ = albedo_maps_dir;

    // Get canonical default albedo for this body
    default_albedo_ = getCanonicalAlbedo(body_name);

    std::cout << "[AlbedoManager] Initializing for body: " << body_name_ << std::endl;
    std::cout << "[AlbedoManager] Default albedo: " << default_albedo_ << std::endl;
    std::cout << "[AlbedoManager] Maps directory: " << maps_dir_ << std::endl;

    // Try to load the best available map
    has_map_ = tryLoadBestMap(body_name, albedo_maps_dir);

    if (has_map_) {
        std::cout << "[AlbedoManager] Albedo map loaded successfully" << std::endl;
        std::cout << "[AlbedoManager] Map dimensions: " << map_width_ << " x " << map_height_ << std::endl;
        std::cout << "[AlbedoManager] Albedo range: " << map_min_ << " to " << map_max_ << std::endl;
    } else {
        std::cout << "[AlbedoManager] No albedo map found, using constant albedo: "
                  << default_albedo_ << std::endl;
    }

    initialized_ = true;
    return true;  // Always succeed - we can fall back to default
}

bool AlbedoManager::tryLoadBestMap(const std::string& body_name, const std::string& maps_dir) {
    // Normalize body name to proper case
    std::string proper_name = body_name;
    if (!proper_name.empty()) {
        std::transform(proper_name.begin(), proper_name.end(), proper_name.begin(), ::tolower);
        proper_name[0] = std::toupper(proper_name[0]);
    }

    // Ensure maps_dir has trailing separator
    std::string dir = maps_dir;
    if (!dir.empty() && dir.back() != '/' && dir.back() != '\\') {
        dir += '/';
    }

    // Priority order for FITS files (try multiple naming conventions):
    // 1. {Body}-0.5deg-BondAlbedo.fits (standard high-res)
    // 2. {Body}-albedo.fits (simple naming)
    // 3. {Body}-1deg-BondAlbedo.fits (lower resolution)
    // 4. Binary files for standard icosphere meshes

    // FITS filename patterns to try (in priority order)
    std::vector<std::string> fits_patterns = {
        dir + proper_name + "-0.5deg-BondAlbedo.fits",
        dir + proper_name + "-BondAlbedo.fits",
        dir + proper_name + "-albedo.fits",
        dir + proper_name + "-1deg-BondAlbedo.fits"
    };

    for (const auto& fits_path : fits_patterns) {
        if (std::filesystem::exists(fits_path)) {
            std::cout << "[AlbedoManager] Found FITS map: " << fits_path << std::endl;
            if (loadFITSMap(fits_path)) {
                return true;
            }
        }
    }

    // Try binary maps at different resolutions (prefer higher resolution)
    // Note: Binary maps only work for standard icosphere meshes
    for (int res = 8; res >= 3; --res) {
        std::string bin_path = dir + proper_name + "_Albedo_res" + std::to_string(res) + ".bin";
        if (std::filesystem::exists(bin_path)) {
            std::cout << "[AlbedoManager] Found binary map: " << bin_path << std::endl;
            if (loadBinaryMap(bin_path)) {
                return true;
            }
        }
    }

    return false;
}

bool AlbedoManager::loadFromFile(const std::string& filepath) {
    if (filepath.empty()) {
        return false;
    }

    // Determine file type by extension
    std::string ext = filepath.substr(filepath.find_last_of('.'));
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    if (ext == ".fits" || ext == ".fit") {
        return loadFITSMap(filepath);
    } else if (ext == ".bin") {
        return loadBinaryMap(filepath);
    } else {
        std::cerr << "[AlbedoManager] Unknown file format: " << ext << std::endl;
        return false;
    }
}

bool AlbedoManager::loadBinaryMap(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        std::cerr << "[AlbedoManager] Cannot open binary file: " << filepath << std::endl;
        return false;
    }

    // Binary format: size_t num_values, then double[num_values]
    // These are per-face values for standard icosphere, not a lat/lon grid
    // We need to convert them to a lat/lon grid for interpolation

    size_t num_values;
    file.read(reinterpret_cast<char*>(&num_values), sizeof(size_t));

    if (num_values == 0 || num_values > 100000000) {
        std::cerr << "[AlbedoManager] Invalid binary file size: " << num_values << std::endl;
        return false;
    }

    // For now, we'll create a simple lat/lon grid by averaging values
    // This is a temporary approach - ideally we'd use the icosphere directly

    // Read the raw values
    std::vector<double> values(num_values);
    file.read(reinterpret_cast<char*>(values.data()), num_values * sizeof(double));

    if (!file) {
        std::cerr << "[AlbedoManager] Error reading binary file" << std::endl;
        return false;
    }

    // Compute statistics
    map_min_ = *std::min_element(values.begin(), values.end());
    map_max_ = *std::max_element(values.begin(), values.end());

    // For binary icosphere maps, we'll use a simplified approach:
    // Create a 360x180 lat/lon grid and compute average albedo in each cell
    // This loses some detail but works for any mesh

    map_width_ = 360;   // 1 degree resolution
    map_height_ = 180;
    lat_min_ = -90.0;
    lat_max_ = 90.0;
    lon_min_ = 0.0;
    lon_max_ = 360.0;

    // Since we don't have the icosphere geometry here, we'll store the raw values
    // and use nearest-neighbor for now. This should be improved later.
    // For now, just use the mean value as a placeholder
    double mean_albedo = 0.0;
    for (double v : values) {
        mean_albedo += v;
    }
    mean_albedo /= values.size();

    // Create a uniform grid with the mean value
    // TODO: Properly project icosphere values to lat/lon grid
    map_data_.assign(map_width_ * map_height_, mean_albedo);

    std::cout << "[AlbedoManager] Binary map: " << num_values << " values, mean albedo = "
              << mean_albedo << std::endl;
    std::cout << "[AlbedoManager] WARNING: Binary icosphere maps not fully supported yet, "
              << "using mean albedo for all locations" << std::endl;

    has_map_ = true;
    return true;
}

bool AlbedoManager::loadFITSMap(const std::string& filepath) {
    try {
        // Use SpatialMap to load FITS - keep it as a member for direct interpolation
        std::cout << "[AlbedoManager] Creating SpatialMap for FITS loading..." << std::endl;
        spatial_map_ = std::make_shared<SpatialMap>();

        std::cout << "[AlbedoManager] Loading FITS file: " << filepath << std::endl;
        if (!spatial_map_->loadFromFITS(filepath, MapFillPolicy::bondAlbedo())) {
            std::cerr << "[AlbedoManager] Failed to load FITS file: " << filepath << std::endl;
            std::cerr << "[AlbedoManager] Error: " << spatial_map_->getLastError() << std::endl;
            spatial_map_.reset();
            return false;
        }

        std::cout << "[AlbedoManager] FITS loaded, extracting metadata..." << std::endl;

        // Store map metadata
        map_width_ = spatial_map_->getWidth();
        map_height_ = spatial_map_->getHeight();
        lat_min_ = spatial_map_->getLatMin();
        lat_max_ = spatial_map_->getLatMax();
        lon_min_ = spatial_map_->getLonMin();
        lon_max_ = spatial_map_->getLonMax();
        map_min_ = spatial_map_->getMin();
        map_max_ = spatial_map_->getMax();

        std::cout << "[AlbedoManager] FITS map loaded: " << map_width_ << "x" << map_height_
                  << ", albedo range: " << map_min_ << " to " << map_max_ << std::endl;

        has_map_ = true;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[AlbedoManager] Exception loading FITS: " << e.what() << std::endl;
        spatial_map_.reset();
        return false;
    } catch (...) {
        std::cerr << "[AlbedoManager] Unknown exception loading FITS" << std::endl;
        spatial_map_.reset();
        return false;
    }
}

double AlbedoManager::interpolate(double lat_deg, double lon_deg) const {
    if (!has_map_) {
        return default_albedo_;
    }

    // If we have a SpatialMap (from FITS), use its interpolation
    if (spatial_map_ && spatial_map_->isValid()) {
        // SpatialMap::getValue(lon, lat) - note the order!
        double albedo = spatial_map_->getValue(lon_deg, lat_deg);

        // Check for missing value
        if (albedo == spatial_map_->getMissingValue() ||
            !std::isfinite(albedo) || albedo < 0.0 || albedo > 1.0) {
            return default_albedo_;
        }
        return albedo;
    }

    // Fall back to map_data_ interpolation (for binary maps)
    if (map_data_.empty()) {
        return default_albedo_;
    }

    // Normalize longitude to [0, 360)
    while (lon_deg < 0.0) lon_deg += 360.0;
    while (lon_deg >= 360.0) lon_deg -= 360.0;

    // Clamp latitude to [-90, 90]
    lat_deg = std::max(-90.0, std::min(90.0, lat_deg));

    // Convert to grid coordinates
    double x = (lon_deg - lon_min_) / (lon_max_ - lon_min_) * (map_width_ - 1);
    double y = (lat_deg - lat_min_) / (lat_max_ - lat_min_) * (map_height_ - 1);

    // Bilinear interpolation
    int x0 = static_cast<int>(std::floor(x));
    int y0 = static_cast<int>(std::floor(y));
    int x1 = std::min(x0 + 1, map_width_ - 1);
    int y1 = std::min(y0 + 1, map_height_ - 1);

    x0 = std::max(0, x0);
    y0 = std::max(0, y0);

    double fx = x - x0;
    double fy = y - y0;

    double v00 = map_data_[y0 * map_width_ + x0];
    double v10 = map_data_[y0 * map_width_ + x1];
    double v01 = map_data_[y1 * map_width_ + x0];
    double v11 = map_data_[y1 * map_width_ + x1];

    double albedo = v00 * (1 - fx) * (1 - fy) +
                    v10 * fx * (1 - fy) +
                    v01 * (1 - fx) * fy +
                    v11 * fx * fy;

    // Clamp to valid range
    return std::max(0.0, std::min(1.0, albedo));
}

double AlbedoManager::getAlbedoAtLocation(double lat_deg, double lon_deg) const {
    if (!initialized_) {
        std::cerr << "[AlbedoManager] Warning: not initialized, returning default" << std::endl;
        return default_albedo_;
    }

    if (!has_map_) {
        return default_albedo_;
    }

    return interpolate(lat_deg, lon_deg);
}

std::vector<double> AlbedoManager::getAlbedosForMesh(const std::shared_ptr<Mesh>& mesh) const {
    if (!mesh) {
        return std::vector<double>();
    }

    size_t num_faces = mesh->faces.size();
    std::vector<double> albedos(num_faces, default_albedo_);

    if (!initialized_) {
        std::cerr << "[AlbedoManager] Warning: not initialized, using default albedo" << std::endl;
        return albedos;
    }

    if (!has_map_) {
        // No map, return constant default
        return albedos;
    }

    // Check if mesh has lat/lon data
    bool has_latlon = (mesh->latitudes.size() == num_faces &&
                       mesh->longitudes.size() == num_faces);

    // Check if mesh has face centers
    bool has_centers = (mesh->face_centers.size() == num_faces);

    if (has_latlon) {
        // Use pre-computed lat/lon
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < num_faces; ++i) {
            albedos[i] = interpolate(mesh->latitudes[i], mesh->longitudes[i]);
        }
    } else if (has_centers) {
        // Compute lat/lon from Cartesian coordinates
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < num_faces; ++i) {
            const auto& center = mesh->face_centers[i];
            double x = center[0];
            double y = center[1];
            double z = center[2];

            // Assuming body-centered coordinates with z as polar axis
            double r = std::sqrt(x*x + y*y + z*z);
            if (r < 1e-10) {
                albedos[i] = default_albedo_;
                continue;
            }

            double lat_deg = std::asin(z / r) * 180.0 / M_PI;
            double lon_deg = std::atan2(y, x) * 180.0 / M_PI;
            if (lon_deg < 0) lon_deg += 360.0;

            albedos[i] = interpolate(lat_deg, lon_deg);
        }
    } else {
        // Need to compute face centers from vertices
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < num_faces; ++i) {
            const auto& face = mesh->faces[i];
            const auto& v0 = mesh->vertices[face[0]];
            const auto& v1 = mesh->vertices[face[1]];
            const auto& v2 = mesh->vertices[face[2]];

            double cx = (v0[0] + v1[0] + v2[0]) / 3.0;
            double cy = (v0[1] + v1[1] + v2[1]) / 3.0;
            double cz = (v0[2] + v1[2] + v2[2]) / 3.0;

            double r = std::sqrt(cx*cx + cy*cy + cz*cz);
            if (r < 1e-10) {
                albedos[i] = default_albedo_;
                continue;
            }

            double lat_deg = std::asin(cz / r) * 180.0 / M_PI;
            double lon_deg = std::atan2(cy, cx) * 180.0 / M_PI;
            if (lon_deg < 0) lon_deg += 360.0;

            albedos[i] = interpolate(lat_deg, lon_deg);
        }
    }

    return albedos;
}

}  // namespace ptm
