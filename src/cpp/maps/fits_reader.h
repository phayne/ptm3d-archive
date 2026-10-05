#ifndef FITS_READER_H
#define FITS_READER_H

#include <string>
#include <vector>
#include <memory>

namespace ptm {

struct FITSHeader {
    int width;
    int height;
    double lon_min;
    double lon_max;
    double lat_min;
    double lat_max;
    // Fill (no-data) value of the raster. Read from the header's BLANK, MISSING,
    // or NODATA keyword (first one present, in that order); -9999 only when the
    // header declares none of them. missing_value_source names where it came from.
    double missing_value;
    std::string missing_value_source;
    // Every fill value the header declares, in keyword order. A file may carry
    // more than one (for example BLANK and MISSING); all of them are fill.
    std::vector<double> declared_fill_values;
    // True for a single-precision floating-point image (BITPIX = -32). Its pixels
    // are floats promoted to double, so a declared fill value such as -9999.99
    // must be compared at float precision to match them.
    bool single_precision;
    
    FITSHeader() : width(0), height(0), 
                  lon_min(0.0), lon_max(360.0), 
                  lat_min(-90.0), lat_max(90.0),
                  missing_value(-9999.0),
                  missing_value_source("default"),
                  single_precision(false) {}
};

// FITS file reader class
class FITSReader {
public:
    // Read FITS file into data structure
    bool read(const std::string& filename, std::vector<double>& data, FITSHeader& header);
    
    // Write data structure to FITS file
    bool write(const std::string& filename, const std::vector<double>& data, 
              const FITSHeader& header);
    
    // Get last error message
    std::string getLastError() const { return last_error_; }
    
    // Helper functions for WCS and data reading
    bool readWCS(void* fptr, FITSHeader& header);
    bool readImageData(void* fptr, std::vector<double>& data, const FITSHeader& header);
    
    // Format FITS error from status code
    static std::string formatFitsError(int status);

private:
    std::string last_error_;
};

} // namespace ptm

#endif // FITS_READER_H