#include "spatial_map.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <sstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <mutex>

namespace ptm {

SpatialMap::SpatialMap() : 
    width_(0), 
    height_(0), 
    lon_min_(0.0),
    lon_max_(0.0),
    lat_min_(0.0),
    lat_max_(0.0),
    missing_value_(-9999.0),
    projection_(MapProjection::EQUIRECTANGULAR) {
}

SpatialMap::SpatialMap(int width, int height, 
                     double lon_min, double lon_max, 
                     double lat_min, double lat_max,
                     MapProjection proj,
                     double missing_val) :
    width_(width),
    height_(height),
    lon_min_(lon_min),
    lon_max_(lon_max),
    lat_min_(lat_min),
    lat_max_(lat_max),
    missing_value_(missing_val),
    projection_(proj) {
    
    // Initialize data with missing values
    data_.resize(width * height, missing_val);
}

SpatialMap::SpatialMap(int width, int height, 
                     double lon_min, double lon_max, 
                     double lat_min, double lat_max,
                     const std::vector<double>& data,
                     MapProjection proj,
                     double missing_val) :
    width_(width),
    height_(height),
    lon_min_(lon_min),
    lon_max_(lon_max),
    lat_min_(lat_min),
    lat_max_(lat_max),
    missing_value_(missing_val),
    projection_(proj),
    data_(data) {
    
    // Check data size
    if (data.size() != static_cast<size_t>(width * height)) {
        last_error_ = "Data size does not match dimensions";
        throw std::invalid_argument(last_error_);
    }
}

namespace {

// Detect and mark uniform fill values in edge columns of cylindrical maps.
// Some FITS files have sentinel/fill values (e.g. -1.0) in columns 0 and W-1
// that are not flagged by the FITS MISSING keyword. This detects such columns
// by checking if all values are identical and far outside the range of the
// neighboring interior column.
inline void detect_edge_fill_values(std::vector<double>& data,
                                    int width, int height,
                                    double missing_value) {
    if (width < 4 || height <= 0) return;

    auto check_and_fix_column = [&](int col) {
        const int neighbor_col = (col == 0) ? 1 : (width - 2);

        // Check if this column is uniformly a single value
        double col_val = data[col];
        if (!std::isfinite(col_val) || col_val == missing_value) return; // already bad
        bool all_same = true;
        for (int y = 1; y < height; ++y) {
            if (data[y * width + col] != col_val) {
                all_same = false;
                break;
            }
        }
        if (!all_same) return;

        // Compute range of the neighboring interior column
        double nmin = std::numeric_limits<double>::max();
        double nmax = std::numeric_limits<double>::lowest();
        for (int y = 0; y < height; ++y) {
            double v = data[y * width + neighbor_col];
            if (std::isfinite(v) && v != missing_value) {
                nmin = std::min(nmin, v);
                nmax = std::max(nmax, v);
            }
        }
        if (nmin > nmax) return; // no valid neighbor data

        // If the uniform value is far outside the neighbor range, mark as missing
        double margin = 0.5 * (nmax - nmin);
        if (col_val < nmin - margin || col_val > nmax + margin) {
            for (int y = 0; y < height; ++y) {
                data[y * width + col] = missing_value;
            }
            std::cout << "Detected fill values in edge column " << col
                      << " (uniform value " << col_val
                      << "), marked as missing" << std::endl;
        }
    };

    check_and_fix_column(0);
    check_and_fix_column(width - 1);
}

// Heal edge columns of a cylindrical (360-degree) map using linear interpolation.
// When edge columns contain missing/invalid data, interpolates smoothly from
// the nearest valid interior columns on each side of the seam. When edge
// columns already contain valid data, they are left unchanged.
inline void heal_cylindrical_seam(std::vector<double>& data,
                                  int width, int height,
                                  double missing_value) {
    if (width < 4 || height <= 0) return;
    const int W = width;

    auto is_bad = [&](double v) -> bool {
        return !std::isfinite(v) || v == missing_value;
    };

    for (int y = 0; y < height; ++y) {
        const int i0   = y * W + 0;       // col 0 (first)
        const int i1   = y * W + 1;       // col 1
        const int iLm1 = y * W + (W - 2); // col W-2
        const int iL   = y * W + (W - 1); // col W-1 (last)

        double v0   = data[i0];
        double v1   = data[i1];
        double vLm1 = data[iLm1];
        double vL   = data[iL];

        bool v0_bad  = is_bad(v0);
        bool v1_ok   = !is_bad(v1);
        bool vLm1_ok = !is_bad(vLm1);
        bool vL_bad  = is_bad(vL);

        // Both edges bad, both interior neighbors OK: linear interpolation
        // across the 2-pixel seam gap (3 equal intervals: vLm1 -> W-1 -> 0 -> v1)
        if (v0_bad && vL_bad && v1_ok && vLm1_ok) {
            data[iL] = (2.0 / 3.0) * vLm1 + (1.0 / 3.0) * v1;
            data[i0] = (1.0 / 3.0) * vLm1 + (2.0 / 3.0) * v1;
        }
        // Only col 0 is bad
        else if (v0_bad && !vL_bad) {
            if (v1_ok) {
                data[i0] = 0.5 * (vL + v1);
            } else {
                data[i0] = vL;
            }
        }
        // Only col W-1 is bad
        else if (vL_bad && !v0_bad) {
            if (vLm1_ok) {
                data[iL] = 0.5 * (v0 + vLm1);
            } else {
                data[iL] = v0;
            }
        }
        // Both edges valid: leave them unchanged
    }
}

} // anonymous namespace

bool SpatialMap::loadFromFITS(const std::string& filename) {
    try {
        // Use the FITS reader to load data
        std::vector<double> fileData;
        FITSHeader header;
        
        // Create FITS reader instance
        FITSReader reader;
        if (!reader.read(filename, fileData, header)) {
            last_error_ = "Failed to read FITS file: " + filename + 
                         " - " + reader.getLastError();
            return false;
        }
        
        // Update map properties
        width_ = header.width;
        height_ = header.height;
        lon_min_ = header.lon_min;
        lon_max_ = header.lon_max;
        
        // CORRECTED: Handle FITS latitude convention
        // FITS files often have latitude stored with +90° at the top (y=0)
        // If the header indicates this convention, we need to flip the data
        bool fits_lat_flipped = (header.lat_min < header.lat_max); // Standard FITS convention
        
        if (fits_lat_flipped) {
            // Keep the latitude bounds as they are in the header
            lat_min_ = header.lat_min;
            lat_max_ = header.lat_max;
            
            // Flip the data rows to match our internal coordinate system
            // where lat_min is at the bottom and lat_max is at the top
            std::vector<double> flipped_data(fileData.size());
            for (int y = 0; y < height_; y++) {
                int flipped_y = height_ - 1 - y;
                for (int x = 0; x < width_; x++) {
                    int original_idx = y * width_ + x;
                    int flipped_idx = flipped_y * width_ + x;
                    flipped_data[flipped_idx] = fileData[original_idx];
                }
            }
            data_ = std::move(flipped_data);
        } else {
            // Data is already in the correct orientation
            lat_min_ = header.lat_min;
            lat_max_ = header.lat_max;
            data_ = std::move(fileData);
        }
        
        missing_value_ = header.missing_value;
        
        // Default to equirectangular projection unless specified otherwise
        projection_ = MapProjection::EQUIRECTANGULAR;

        // Detect fill values in edge columns, then heal the seam
        {
            const double period = std::abs(lon_max_ - lon_min_);
            const bool is_cyl = (std::abs(period - 360.0) < 2.0) || (period > 350.0);
            if (is_cyl && width_ >= 4 && height_ > 0) {
                detect_edge_fill_values(data_, width_, height_, missing_value_);
                heal_cylindrical_seam(data_, width_, height_, missing_value_);
            }
        }
        
        std::cout << "Loaded map: " << width_ << "x" << height_ << " pixels" << std::endl;
        std::cout << "Longitude range: " << lon_min_ << " to " << lon_max_ << " degrees" << std::endl;
        std::cout << "Latitude range: " << lat_min_ << " to " << lat_max_ << " degrees" << std::endl;
        std::cout << "Data range: " << getMin() << " to " << getMax() << std::endl;
        
        return true;
    }
    catch (const std::exception& e) {
        last_error_ = "Exception during FITS load: " + std::string(e.what());
        return false;
    }
}

bool SpatialMap::saveToFITS(const std::string& filename) const {
    if (!isValid()) {
        last_error_ = "Cannot save invalid map to FITS";
        return false;
    }
    
    try {
        // Prepare FITS header
        FITSHeader header;
        header.width = width_;
        header.height = height_;
        header.lon_min = lon_min_;
        header.lon_max = lon_max_;
        header.lat_min = lat_min_;
        header.lat_max = lat_max_;
        header.missing_value = missing_value_;
        
        // Save using FITS writer
        FITSReader writer;
        if (!writer.write(filename, data_, header)) {
            last_error_ = "Failed to write FITS file: " + filename + 
                         " - " + writer.getLastError();
            return false;
        }
        
        return true;
    }
    catch (const std::exception& e) {
        last_error_ = "Exception during FITS save: " + std::string(e.what());
        return false;
    }
}

bool SpatialMap::saveToCSV(const std::string& filename) const {
    if (!isValid()) {
        last_error_ = "Cannot save invalid map to CSV";
        return false;
    }
    
    try {
        std::ofstream file(filename);
        if (!file.is_open()) {
            last_error_ = "Failed to open file for writing: " + filename;
            return false;
        }
        
        // Write header
        file << "lon,lat,value" << std::endl;
        
        // Write data
        for (int y = 0; y < height_; ++y) {
            // Calculate latitude for this row
            double lat = lat_min_ + (lat_max_ - lat_min_) * y / (height_ - 1);
            
            for (int x = 0; x < width_; ++x) {
                // Calculate longitude for this column
                double lon = lon_min_ + (lon_max_ - lon_min_) * x / (width_ - 1);
                
                // Get value at this position
                double value = getValue(x, y);
                
                // Write to file
                file << lon << "," << lat << "," << value << std::endl;
            }
        }
        
        return true;
    }
    catch (const std::exception& e) {
        last_error_ = "Exception during CSV save: " + std::string(e.what());
        return false;
    }
}

double SpatialMap::getValue(double lon, double lat) const {
    if (!isValid()) {
        return missing_value_;
    }

    // Normalize (handles cylindrical wrap)
    normalizeCoordinates(lon, lat);

    // Cylindrical check
    const double period = std::abs(lon_max_ - lon_min_);
    const bool is_cyl = (std::abs(period - 360.0) < 2.0) || (period > 350.0);

    // Map to pixel coordinates (x: column index in [0, width))
    double x;
    if (is_cyl) {
        // Use center-based mapping with strict wrap into [0,width)
        const double W = static_cast<double>(width_);
        x = (lon - lon_min_) / period * W;
        x = std::fmod(x, W);
        if (x < 0.0) x += W;  // ensure non-negative
    } else {
        // x in [0, width-1] for non-cylindrical
        x = (lon - lon_min_) / (lon_max_ - lon_min_) * static_cast<double>(width_ - 1);
        if (x < 0.0) x = 0.0;
        if (x > static_cast<double>(width_ - 1)) x = static_cast<double>(width_ - 1);
    }

    // y in [0, height-1] with standard latitude mapping
    double y = (lat_max_ - lat) / (lat_max_ - lat_min_) * static_cast<double>(height_ - 1);
    if (y < 0.0) y = 0.0;
    if (y > height_ - 1.0) y = height_ - 1.0;

    return bilinearInterpolate(x, y);
}

double SpatialMap::getValue(int x, int y) const {
    int index = getIndex(x, y);
    if (!isValidIndex(index)) {
        return missing_value_;
    }
    return data_[index];
}

void SpatialMap::setValue(int x, int y, double value) {
    int index = getIndex(x, y);
    if (!isValidIndex(index)) {
        return;
    }
    data_[index] = value;
}

std::shared_ptr<SpatialMap> SpatialMap::resample(int new_width, int new_height) const {
    if (!isValid() || new_width <= 0 || new_height <= 0) {
        throw std::invalid_argument("Invalid parameters for resampling");
    }
    
    // Create new map
    auto result = std::make_shared<SpatialMap>(new_width, new_height, lon_min_, lon_max_, lat_min_, lat_max_, projection_, missing_value_);
    
    // Resample using bilinear interpolation
    for (int y = 0; y < new_height; ++y) {
        // Calculate relative position in the new grid
        double v = static_cast<double>(y) / (new_height - 1);
        
        // Map to the original grid space
        double lat = lat_max_ - v * (lat_max_ - lat_min_);
        
        for (int x = 0; x < new_width; ++x) {
            // Calculate relative position in the new grid
            double u = static_cast<double>(x) / (new_width - 1);
            
            // Map to the original grid space
            double lon = lon_min_ + u * (lon_max_ - lon_min_);
            
            // Get interpolated value
            double value = getValue(lon, lat);
            result->setValue(x, y, value);
        }
    }
    
    return result;
}

double SpatialMap::getMin() const {
    if (!isValid()) {
        return missing_value_;
    }
    
    double min_val = std::numeric_limits<double>::max();
    bool found = false;
    
    for (double value : data_) {
        if (value != missing_value_) {
            min_val = std::min(min_val, value);
            found = true;
        }
    }
    
    return found ? min_val : missing_value_;
}

double SpatialMap::getMax() const {
    if (!isValid()) {
        return missing_value_;
    }
    
    double max_val = std::numeric_limits<double>::lowest();
    bool found = false;
    
    for (double value : data_) {
        if (value != missing_value_) {
            max_val = std::max(max_val, value);
            found = true;
        }
    }
    
    return found ? max_val : missing_value_;
}

double SpatialMap::getMean() const {
    if (!isValid()) {
        return missing_value_;
    }
    
    double sum = 0.0;
    int count = 0;
    
    for (double value : data_) {
        if (value != missing_value_) {
            sum += value;
            count++;
        }
    }
    
    return (count > 0) ? (sum / count) : missing_value_;
}

double SpatialMap::getStdDev() const {
    if (!isValid()) {
        return missing_value_;
    }
    
    double mean = getMean();
    if (mean == missing_value_) {
        return missing_value_;
    }
    
    double sum_squared_diff = 0.0;
    int count = 0;
    
    for (double value : data_) {
        if (value != missing_value_) {
            double diff = value - mean;
            sum_squared_diff += diff * diff;
            count++;
        }
    }
    
    return (count > 1) ? std::sqrt(sum_squared_diff / (count - 1)) : 0.0;
}

void SpatialMap::fill(double value) {
    std::fill(data_.begin(), data_.end(), value);
}

void SpatialMap::applyOffset(double offset) {
    for (double& value : data_) {
        if (value != missing_value_) {
            value += offset;
        }
    }
}

void SpatialMap::applyScale(double scale) {
    for (double& value : data_) {
        if (value != missing_value_) {
            value *= scale;
        }
    }
}

void SpatialMap::applyScaleAndOffset(double scale, double offset) {
    for (double& value : data_) {
        if (value != missing_value_) {
            value = value * scale + offset;
        }
    }
}

void SpatialMap::setCoordinates(double lon_min, double lon_max, 
                               double lat_min, double lat_max) {
    lon_min_ = lon_min;
    lon_max_ = lon_max;
    lat_min_ = lat_min;
    lat_max_ = lat_max;
    
    std::cout << "Map coordinates set to:" << std::endl;
    std::cout << " - Longitude: " << lon_min_ << "° to " << lon_max_ << "°" << std::endl;
    std::cout << " - Latitude: " << lat_min_ << "° to " << lat_max_ << "°" << std::endl;
}

bool isPointInSphericalTriangle(
    double lat, double lon,
    double lat1, double lon1, 
    double lat2, double lon2, 
    double lat3, double lon3) {
    
    // Convert lat/lon to radians for calculation
    lat *= M_PI / 180.0;
    lon *= M_PI / 180.0;
    lat1 *= M_PI / 180.0;
    lon1 *= M_PI / 180.0;
    lat2 *= M_PI / 180.0;
    lon2 *= M_PI / 180.0;
    lat3 *= M_PI / 180.0;
    lon3 *= M_PI / 180.0;
    
    // Convert to 3D Cartesian coordinates (unit sphere)
    std::array<double, 3> p = {
        std::cos(lat) * std::cos(lon),
        std::cos(lat) * std::sin(lon),
        std::sin(lat)
    };
    
    std::array<double, 3> a = {
        std::cos(lat1) * std::cos(lon1),
        std::cos(lat1) * std::sin(lon1),
        std::sin(lat1)
    };
    
    std::array<double, 3> b = {
        std::cos(lat2) * std::cos(lon2),
        std::cos(lat2) * std::sin(lon2),
        std::sin(lat2)
    };
    
    std::array<double, 3> c = {
        std::cos(lat3) * std::cos(lon3),
        std::cos(lat3) * std::sin(lon3),
        std::sin(lat3)
    };
    
    // Calculate barycentric coordinates using spherical triangle method
    auto cross = [](const std::array<double, 3>& u, const std::array<double, 3>& v) {
        return std::array<double, 3>{
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0]
        };
    };
    
    auto magnitude = [](const std::array<double, 3>& v) {
        return std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    };
    
    auto area = [&cross, &magnitude](const std::array<double, 3>& u, 
                                     const std::array<double, 3>& v,
                                     const std::array<double, 3>& w) {
        // Area of spherical triangle using Girard's formula
        auto u_norm = magnitude(u);
        auto v_norm = magnitude(v);
        auto w_norm = magnitude(w);
        
        // Normalize vectors
        std::array<double, 3> u_hat = {u[0]/u_norm, u[1]/u_norm, u[2]/u_norm};
        std::array<double, 3> v_hat = {v[0]/v_norm, v[1]/v_norm, v[2]/v_norm};
        std::array<double, 3> w_hat = {w[0]/w_norm, w[1]/w_norm, w[2]/w_norm};
        
        // Calculate interior angles
        double cos_a = u_hat[0]*v_hat[0] + u_hat[1]*v_hat[1] + u_hat[2]*v_hat[2];
        double cos_b = v_hat[0]*w_hat[0] + v_hat[1]*w_hat[1] + v_hat[2]*w_hat[2];
        double cos_c = w_hat[0]*u_hat[0] + w_hat[1]*u_hat[1] + w_hat[2]*u_hat[2];
        
        double a = std::acos(std::clamp(cos_a, -1.0, 1.0));
        double b = std::acos(std::clamp(cos_b, -1.0, 1.0));
        double c = std::acos(std::clamp(cos_c, -1.0, 1.0));
        
        // Girard's formula: area = A + B + C - π
        return a + b + c - M_PI;
    };
    
    // Calculate areas of subtriangles
    double areaABC = area(a, b, c);
    double areaPBC = area(p, b, c);
    double areaPCA = area(p, c, a);
    double areaPAB = area(p, a, b);
    
    // Check if sum of subtriangle areas equals the full triangle area
    const double epsilon = 1e-10;
    double areaDiff = std::abs(areaABC - (areaPBC + areaPCA + areaPAB));
    
    return areaDiff < epsilon;
}

bool SpatialMap::initializeFromData(
    const float* data,
    int width,
    int height,
    double lon_min,
    double lon_max,
    double lat_min,
    double lat_max,
    MapProjection proj,
    double missing_val) {
    
    // Update member variables
    width_ = width;
    height_ = height;
    lon_min_ = lon_min;
    lon_max_ = lon_max;
    lat_min_ = lat_min;
    lat_max_ = lat_max;
    projection_ = proj;
    missing_value_ = missing_val;
    
    // Resize and copy the data, converting from float to double
    data_.resize(static_cast<size_t>(width) * static_cast<size_t>(height));
    for (int i = 0; i < width * height; ++i) {
        float v = data[i];
        data_[static_cast<size_t>(i)] = static_cast<double>(v);
    }

    // Detect fill values in edge columns, then heal the seam
    {
        const double period = std::abs(lon_max_ - lon_min_);
        const bool is_cyl = (std::abs(period - 360.0) < 2.0) || (period > 350.0);
        if (is_cyl && width_ >= 4 && height_ > 0) {
            detect_edge_fill_values(data_, width_, height_, missing_value_);
            heal_cylindrical_seam(data_, width_, height_, missing_value_);
        }
    }
    
    return true;
}

double SpatialMap::getAverageValueInTriangle(
    double lat1, double lon1,
    double lat2, double lon2,
    double lat3, double lon3) const {

    // Seam-aware unwrapping around lon1
    auto unwrap = [](double base, double& l) {
        double d = l - base;
        while (d > 180.0) d -= 360.0;
        while (d < -180.0) d += 360.0;
        l = base + d;
    };
    double L1 = lon1, L2 = lon2, L3 = lon3;
    unwrap(L1, L2);
    unwrap(L1, L3);

    // Barycentric multi-sample (7-point) inside the spherical triangle
    // We linearly interpolate lat/lon in the locally unwrapped space (small triangles -> good approx)
    struct Bary { double a,b,c; }; // weights sum to 1
    static const Bary samples[] = {
        {1.0/3.0, 1.0/3.0, 1.0/3.0},  // centroid
        {0.5, 0.5, 0.0}, {0.5, 0.0, 0.5}, {0.0, 0.5, 0.5}, // edge midpoints
        {0.6, 0.2, 0.2}, {0.2, 0.6, 0.2}, {0.2, 0.2, 0.6}  // interior points
    };

    auto interp = [&](const Bary& w, double& lat, double& lon) {
        // Interpolate in unwrapped lon space (L1,L2,L3)
        lon = w.a*L1 + w.b*L2 + w.c*L3;
        lat = w.a*lat1 + w.b*lat2 + w.c*lat3;
        // Wrap lon back into map domain (normalizeCoordinates handles cylindrical)
        double lon_norm = lon, lat_norm = lat;
        normalizeCoordinates(lon_norm, lat_norm);
        lon = lon_norm; lat = lat_norm;
    };

    double sum = 0.0;
    int n = 0;
    for (const auto& w : samples) {
        double slat, slon;
        interp(w, slat, slon);
        double v = getValue(slon, slat);
        if (v != missing_value_ && std::isfinite(v)) {
            sum += v;
            ++n;
        }
    }

    if (n == 0) {
        // Fallback to (wrapped) centroid as before
        double centroid_lat = (lat1 + lat2 + lat3) / 3.0;
        double centroid_lon = (L1 + L2 + L3) / 3.0;

        // Wrap back into map domain
        double lon_c = centroid_lon, lat_c = centroid_lat;
        normalizeCoordinates(lon_c, lat_c);
        return getValue(lon_c, lat_c);
    }

    return sum / static_cast<double>(n);
}

bool SpatialMap::isPointInTriangle3D(
    const std::array<double, 3>& p,
    const std::array<double, 3>& a,
    const std::array<double, 3>& b, 
    const std::array<double, 3>& c) const {
    
    // Calculate vectors
    std::array<double, 3> v0 = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
    std::array<double, 3> v1 = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
    std::array<double, 3> v2 = {p[0] - a[0], p[1] - a[1], p[2] - a[2]};
    
    // Calculate dot products
    double dot00 = v0[0]*v0[0] + v0[1]*v0[1] + v0[2]*v0[2];
    double dot01 = v0[0]*v1[0] + v0[1]*v1[1] + v0[2]*v1[2];
    double dot02 = v0[0]*v2[0] + v0[1]*v2[1] + v0[2]*v2[2];
    double dot11 = v1[0]*v1[0] + v1[1]*v1[1] + v1[2]*v1[2];
    double dot12 = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2];
    
    // Calculate barycentric coordinates
    double invDenom = 1.0 / (dot00 * dot11 - dot01 * dot01);
    double u = (dot11 * dot02 - dot01 * dot12) * invDenom;
    double v = (dot00 * dot12 - dot01 * dot02) * invDenom;
    
    // Check if point is inside triangle
    return (u >= 0) && (v >= 0) && (u + v <= 1);
}

bool SpatialMap::isValidIndex(int index) const {
    return index >= 0 && index < static_cast<int>(data_.size());
}

bool SpatialMap::isValidCoordinate(double lon, double lat) const {
    // First normalize coordinates to our map's range
    normalizeCoordinates(lon, lat);
    
    // Then check if they're within bounds
    return lon >= lon_min_ && lon <= lon_max_ && 
           lat >= lat_min_ && lat <= lat_max_;
}

void SpatialMap::normalizeCoordinates(double& lon, double& lat) const {
    // Treat near-360° longitude spans as cylindrical
    const double period = std::abs(lon_max_ - lon_min_);
    const bool is_cyl = (std::abs(period - 360.0) < 2.0) || (period > 350.0);

    if (is_cyl) {
        // Bring lon into [lon_min_, lon_min_ + period)
        if (period > 0.0) {
            const double k = std::floor((lon - lon_min_) / period);
            lon = lon - k * period;
            // Guard numerical edge hitting upper bound
            if (lon >= lon_min_ + period) lon = std::nextafter(lon_min_ + period, lon_min_);
            if (lon < lon_min_) lon += period;
        }
    } else {
        // Clamp for non-cylindrical ranges
        if (lon < lon_min_) lon = lon_min_;
        if (lon > lon_max_) lon = lon_max_;
    }

    // Clamp latitude to physical range and map bounds
    lat = std::max(-90.0, std::min(90.0, lat));
    if (lat < lat_min_) lat = lat_min_;
    if (lat > lat_max_) lat = lat_max_;
}

int SpatialMap::getIndex(int x, int y) const {
    if (x < 0 || x >= width_ || y < 0 || y >= height_) {
        return -1; // Invalid index
    }
    
    // Assuming row-major order storage
    return y * width_ + x;
}

double SpatialMap::bilinearInterpolate(double x, double y) const {
    // Clamp y into [0, H-1]
    if (y < 0.0) y = 0.0;
    if (y > height_ - 1.0) y = height_ - 1.0;

    // Periodicity in x
    const double period = std::abs(lon_max_ - lon_min_);
    const bool is_cylindrical = (std::abs(period - 360.0) < 2.0) || (period > 350.0);

    if (is_cylindrical) {
        // Wrap x robustly into [0, width)
        const double W = static_cast<double>(width_);
        x = std::fmod(x, W);
        if (x < 0.0) x += W;
    } else {
        if (x < 0.0) x = 0.0;
        if (x > width_ - 1.0) x = width_ - 1.0;
    }

    // Integer/fractional parts
    int x0 = static_cast<int>(std::floor(x));
    int y0 = static_cast<int>(std::floor(y));
    double xd = x - static_cast<double>(x0);
    double yd = y - static_cast<double>(y0);

    // Neighbor indices (wrap x1 in cylindrical case)
    int x1 = is_cylindrical ? ((x0 + 1) % width_) : std::min(x0 + 1, width_ - 1);
    int y1 = std::min(y0 + 1, height_ - 1);

    // Sample four corners directly (integer overload)
    double v00 = getValue(x0, y0);
    double v10 = getValue(x1, y0);
    double v01 = getValue(x0, y1);
    double v11 = getValue(x1, y1);

    // Standard bilinear if all valid
    if (v00 != missing_value_ && v10 != missing_value_ &&
        v01 != missing_value_ && v11 != missing_value_) {
        double i1 = v00 * (1.0 - xd) + v10 * xd;
        double i2 = v01 * (1.0 - xd) + v11 * xd;
        return i1 * (1.0 - yd) + i2 * yd;
    }

    // Graceful fallback: weighted average of available neighbors
    int nvalid = 0;
    double acc = 0.0;
    auto add_if = [&](double v, double w) {
        if (v != missing_value_ && std::isfinite(v)) { acc += v * w; ++nvalid; }
    };
    add_if(v00, (1.0 - xd) * (1.0 - yd));
    add_if(v10, xd * (1.0 - yd));
    add_if(v01, (1.0 - xd) * yd);
    add_if(v11, xd * yd);
    return (nvalid > 0) ? acc : missing_value_;
}

size_t SpatialMap::hashTriangle(double lat1, double lon1, 
                                 double lat2, double lon2, 
                                 double lat3, double lon3) const {
    // Quantize and hash coordinates for cache lookup
    const double precision = 0.1; // Quantization precision
    size_t hash = 0;
    
    auto quantize = [precision](double val) -> int {
        return static_cast<int>(std::round(val / precision));
    };
    
    // Combine quantized coordinates
    std::vector<int> coords = {
        quantize(lat1), quantize(lon1), 
        quantize(lat2), quantize(lon2), 
        quantize(lat3), quantize(lon3)
    };
    
    // Use boost hash_combine-like algorithm
    for (int val : coords) {
        hash ^= std::hash<int>{}(val) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    }
    
    return hash;
}

} // namespace ptm