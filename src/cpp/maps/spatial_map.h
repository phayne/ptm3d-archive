#ifndef SPATIAL_MAP_H
#define SPATIAL_MAP_H

#include <vector>
#include <string>
#include <memory>
#include <array>
#include <unordered_map>  // Add this include
#include <mutex>
#include "fits_reader.h"

namespace ptm {

enum class MapProjection {
    EQUIRECTANGULAR,  // Simple lat/lon grid
    SINUSOIDAL,       // Area-preserving projection
    CUSTOM            // User-defined projection
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
        double missing_val = -9999.0);
    
    // Load from file
    bool loadFromFITS(const std::string& filename);
    bool loadFromBinary(const std::string& filename);
    
    // Save to file
    bool saveToFITS(const std::string& filename) const;
    bool saveToCSV(const std::string& filename) const;
    
    // Get interpolated value at specific location
    double getValue(double lon, double lat) const;
    
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

    // New method to get average value in a triangular mesh face
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
    double bilinearInterpolate(double x, double y) const;
};

// This function is used outside the class
bool isPointInSphericalTriangle(
    double lat, double lon,
    double lat1, double lon1, 
    double lat2, double lon2, 
    double lat3, double lon3);

} // namespace ptm

#endif // SPATIAL_MAP_H