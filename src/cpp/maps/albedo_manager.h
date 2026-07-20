#ifndef ALBEDO_MANAGER_H
#define ALBEDO_MANAGER_H

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <array>

namespace ptm {

class Mesh;       // Forward declaration
class SpatialMap; // Forward declaration

/**
 * @brief Centralized manager for albedo data across all planetary bodies
 *
 * This class provides a unified interface for loading and interpolating albedo
 * maps to arbitrary mesh faces. It supports:
 * - Binary pre-computed albedo maps (for standard icosphere meshes)
 * - FITS global albedo maps (for arbitrary meshes)
 * - Fallback to planetary body default albedo
 *
 * Usage:
 *   AlbedoManager manager;
 *   manager.initialize("Moon", "data/albedo_maps");
 *   std::vector<double> face_albedos = manager.getAlbedosForMesh(mesh);
 */
class AlbedoManager {
public:
    AlbedoManager();
    ~AlbedoManager();

    /**
     * @brief Initialize the albedo manager for a specific planetary body
     * @param body_name Name of the planetary body (e.g., "Moon", "Europa")
     * @param albedo_maps_dir Directory containing albedo map files
     * @return true if initialization successful (map loaded or default available)
     */
    bool initialize(const std::string& body_name, const std::string& albedo_maps_dir = "data/albedo_maps");

    /**
     * @brief Load a specific albedo map file (FITS or binary)
     * @param filepath Path to the albedo map file
     * @return true if loading successful
     */
    bool loadFromFile(const std::string& filepath);

    /**
     * @brief Get interpolated albedo values for all mesh faces
     * @param mesh The mesh to interpolate albedos to
     * @return Vector of albedo values, one per face
     */
    std::vector<double> getAlbedosForMesh(const std::shared_ptr<Mesh>& mesh) const;

    /**
     * @brief Get interpolated albedo for a single lat/lon location
     * @param lat_deg Latitude in degrees
     * @param lon_deg Longitude in degrees
     * @return Interpolated albedo value
     */
    double getAlbedoAtLocation(double lat_deg, double lon_deg) const;

    /**
     * @brief Get the default albedo for the current planetary body
     * @return Default Bond albedo
     */
    double getDefaultAlbedo() const { return default_albedo_; }

    /**
     * @brief Check if a valid albedo map is loaded
     * @return true if map data is available
     */
    bool hasMap() const { return has_map_; }

    /**
     * @brief Get the name of the current planetary body
     */
    const std::string& getBodyName() const { return body_name_; }

    /**
     * @brief Get map statistics
     */
    double getMapMin() const { return map_min_; }
    double getMapMax() const { return map_max_; }
    int getMapWidth() const { return map_width_; }
    int getMapHeight() const { return map_height_; }

    /**
     * @brief Get the canonical default albedo for a planetary body
     * @param body_name Name of the planetary body
     * @return Default Bond albedo for that body
     */
    static double getCanonicalAlbedo(const std::string& body_name);

private:
    // Load different map formats
    bool loadBinaryMap(const std::string& filepath);
    bool loadFITSMap(const std::string& filepath);

    // Try to find and load the best available map for the body
    bool tryLoadBestMap(const std::string& body_name, const std::string& maps_dir);

    // Bilinear interpolation on the loaded map
    double interpolate(double lat_deg, double lon_deg) const;

    // Member variables
    std::string body_name_;
    std::string maps_dir_;
    double default_albedo_;
    bool has_map_;
    bool initialized_;

    // Map data - either raw data or SpatialMap for FITS
    std::vector<double> map_data_;      // For binary maps
    std::shared_ptr<SpatialMap> spatial_map_;  // For FITS maps (uses SpatialMap's interpolation)
    int map_width_;   // Number of longitude bins
    int map_height_;  // Number of latitude bins
    double lat_min_, lat_max_;  // Latitude range (degrees)
    double lon_min_, lon_max_;  // Longitude range (degrees)
    double map_min_, map_max_;  // Albedo value range
};

}  // namespace ptm

#endif  // ALBEDO_MANAGER_H
