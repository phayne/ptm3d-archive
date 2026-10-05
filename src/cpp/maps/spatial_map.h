#ifndef SPATIAL_MAP_H
#define SPATIAL_MAP_H

#include <vector>
#include <string>
#include <memory>
#include <array>
#include <unordered_map>  // Add this include
#include <mutex>
#include <limits>
#include <cmath>
#include <cstddef>
#include "fits_reader.h"

namespace ptm {

enum class MapProjection {
    EQUIRECTANGULAR,  // Simple lat/lon grid
    SINUSOIDAL,       // Area-preserving projection
    CUSTOM            // User-defined projection
};

// Which raster values are fill (no data) beyond the value the file itself
// declares (FITS BLANK/MISSING/NODATA). Applied once at load time: every pixel
// it classifies as fill is rewritten to the map's missing value, so the seam
// healing, the samplers and the statistics all see a single sentinel.
struct MapFillPolicy {
    // Exact-match sentinels configured for a particular map, for files whose
    // header does not declare the value they actually use.
    std::vector<double> fill_values;
    // Physical range of the quantity. A value outside [valid_min, valid_max]
    // cannot be data and is treated as fill. Unbounded by default.
    double valid_min = -std::numeric_limits<double>::infinity();
    double valid_max = std::numeric_limits<double>::infinity();

    // A Bond albedo lies in [0, 1]. This is what catches the Europa
    // Bond-albedo maps' -1.0 gap fill, which their headers misdeclare as
    // MISSING = -9999 (see data/albedo_maps/README.md).
    static MapFillPolicy bondAlbedo() {
        MapFillPolicy p;
        p.valid_min = 0.0;
        p.valid_max = 1.0;
        return p;
    }
};

// Per-category pixel counts from applying a MapFillPolicy at load time.
struct MapFillCounts {
    size_t declared = 0;      // equal to a fill value the header declares
    size_t configured = 0;    // equal to a MapFillPolicy::fill_values entry
    size_t out_of_range = 0;  // outside [valid_min, valid_max]
    size_t non_finite = 0;    // NaN or infinite
    size_t total() const { return declared + configured + out_of_range + non_finite; }
};

// What a single interpolated sample drew on.
struct MapSampleInfo {
    int fill_corners = 0;        // bilinear corners with nonzero weight that were fill
    double fill_weight = 0.0;    // their total bilinear weight (0 = clean sample)
    bool no_valid_support = false;  // every weighted corner was fill: value is missing
};

class SpatialMap {
public:
    // Constructors
    SpatialMap();
    
    // Create an empty map with dimensions
    SpatialMap(int width, int height, 
              double lon_min, double lon_max, 
              double lat_min, double lat_max,
              MapProjection proj = MapProjection::EQUIRECTANGULAR,
              double missing_val = -9999.0);
              
    // Create a map with data
    SpatialMap(int width, int height, 
              double lon_min, double lon_max, 
              double lat_min, double lat_max,
              const std::vector<double>& data,
              MapProjection proj = MapProjection::EQUIRECTANGULAR,
              double missing_val = -9999.0);
    
    // Initialize the map with data
    bool initializeFromData(
        const float* data,
        int width,
        int height,
        double lon_min,
        double lon_max,
        double lat_min,
        double lat_max,
        MapProjection proj = MapProjection::EQUIRECTANGULAR,
        double missing_val = -9999.0,
        const MapFillPolicy& policy = MapFillPolicy());
    
    // Load from file. Fill pixels (header-declared, configured, out of the
    // policy's physical range, or non-finite) are rewritten to the missing
    // value and reported on stdout; getFillCounts() returns the tally.
    bool loadFromFITS(const std::string& filename,
                      const MapFillPolicy& policy = MapFillPolicy());
    bool loadFromBinary(const std::string& filename);
    
    // Save to file
    bool saveToFITS(const std::string& filename) const;
    bool saveToCSV(const std::string& filename) const;
    
    // Get interpolated value at specific location. Fill pixels never enter a
    // sample: the bilinear weights are renormalized over the valid corners, and
    // the missing value is returned when no weighted corner is valid. Pass
    // `info` to learn whether (and how much) fill the sample had to skip.
    double getValue(double lon, double lat) const;
    double getValue(double lon, double lat, MapSampleInfo* info) const;
    
    // Create a resampled version of the map
    std::shared_ptr<SpatialMap> resample(int new_width, int new_height) const;
    
    // Basic getters
    int getWidth() const { return width_; }
    int getHeight() const { return height_; }
    double getLonMin() const { return lon_min_; }
    double getLonMax() const { return lon_max_; }
    double getLatMin() const { return lat_min_; }
    double getLatMax() const { return lat_max_; }
    double getMissingValue() const { return missing_value_; }
    MapProjection getProjection() const { return projection_; }
    const MapFillCounts& getFillCounts() const { return fill_counts_; }
    
    // Statistics
    double getMin() const;
    double getMax() const;
    double getMean() const;
    double getStdDev() const;
    
    // Access raw data
    const std::vector<double>& getData() const { return data_; }
    std::vector<double>& getData() { return data_; }
    
    // Set/get a specific pixel
    void setValue(int x, int y, double value);
    double getValue(int x, int y) const;
    
    // Utility functions
    void fill(double value);
    void applyOffset(double offset);
    void applyScale(double scale);
    void applyScaleAndOffset(double scale, double offset);
    
    // Check if the map has valid data
    bool isValid() const { return !data_.empty() && width_ > 0 && height_ > 0; }
    
    // Get the last error message
    std::string getLastError() const { return last_error_; }

    // Set coordinates
    void setCoordinates(double lon_min, double lon_max, double lat_min, double lat_max);

    // Mean of a 7-point barycentric sample of a triangular face. Each point goes
    // through getValue(), so fill is excluded the same way. Not used by
    // albedo_map_generator, which samples face centers.
    double getAverageValueInTriangle(
        double lat1, double lon1,
        double lat2, double lon2,
        double lat3, double lon3) const;

    // Helper method for 3D triangle containment test
    bool isPointInTriangle3D(
        const std::array<double, 3>& p,
        const std::array<double, 3>& a,
        const std::array<double, 3>& b,
        const std::array<double, 3>& c) const;

private:
    std::vector<double> data_;
    int width_;
    int height_;
    double lon_min_;
    double lon_max_;
    double lat_min_;
    double lat_max_;
    double missing_value_;
    MapProjection projection_;
    MapFillCounts fill_counts_;
    mutable std::string last_error_;
    
    // Add these new members to support caching triangle calculations
    mutable std::unordered_map<size_t, double> triangle_albedo_cache;
    mutable std::mutex cache_mutex_;
    
    // Add this function for triangle hashing
    size_t hashTriangle(double lat1, double lon1, 
                        double lat2, double lon2, 
                        double lat3, double lon3) const;

    // Helper methods
    bool isValidCoordinate(double lon, double lat) const;
    void normalizeCoordinates(double& lon, double& lat) const;
    int getIndex(int x, int y) const;
    bool isValidIndex(int index) const;
    bool isFill(double v) const { return !std::isfinite(v) || v == missing_value_; }
    double bilinearInterpolate(double x, double y, MapSampleInfo* info = nullptr) const;
    // Rewrite every fill pixel of data_ to missing_value_ and tally it.
    // `single_precision`: the raster was stored as float, so sentinels are matched
    // at float precision.
    void applyFillPolicy(const std::vector<double>& declared_fill_values,
                         const MapFillPolicy& policy, bool single_precision);
    // One stdout line with the fill tally, and a WARNING when most of the map is
    // fill (wrong units or sentinel): the load still succeeds, but never silently.
    void reportFillCounts(const std::string& label, const std::string& missing_source,
                          const MapFillPolicy& policy) const;
};

// This function is used outside the class
bool isPointInSphericalTriangle(
    double lat, double lon,
    double lat1, double lon1, 
    double lat2, double lon2, 
    double lat3, double lon3);

} // namespace ptm

#endif // SPATIAL_MAP_H