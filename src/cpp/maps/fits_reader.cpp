#include "fits_reader.h"
#include <fitsio.h>  // CFITSIO library
#include <cmath>
#include <sstream>
#include <iostream>

namespace ptm {

bool FITSReader::read(const std::string& filename, std::vector<double>& data, FITSHeader& header) {
    fitsfile* fptr;
    int status = 0;
    
    // Open FITS file
    fits_open_file(&fptr, filename.c_str(), READONLY, &status);
    if (status) {
        char err_text[32];
        fits_get_errstatus(status, err_text);
        last_error_ = "Error opening FITS file: " + std::string(err_text);
        return false;
    }
    
    // Read image dimensions
    int naxis;
    long naxes[2];
    fits_get_img_dim(fptr, &naxis, &status);
    if (status || naxis != 2) {
        if (!status) status = -1;
        last_error_ = "Error: FITS file is not a 2D image";
        fits_close_file(fptr, &status);
        return false;
    }
    
    fits_get_img_size(fptr, 2, naxes, &status);
    if (status) {
        char err_text[32];
        fits_get_errstatus(status, err_text);
        last_error_ = "Error getting image dimensions: " + std::string(err_text);
        fits_close_file(fptr, &status);
        return false;
    }
    
    // Set basic header info
    header.width = static_cast<int>(naxes[0]);
    header.height = static_cast<int>(naxes[1]);
    
    // Read WCS information for lat/lon bounds
    if (!readWCS(fptr, header)) {
        // If WCS keys not found, use default 0-360, -90-90
        header.lon_min = 0.0;
        header.lon_max = 360.0;
        header.lat_min = -90.0;
        header.lat_max = 90.0;
        
        // Reset status from WCS read attempt
        status = 0;
    }
    
    // Check for BLANK or MISSING keyword
    double blank_val;
    char comment[FLEN_COMMENT];
    if (fits_read_key(fptr, TDOUBLE, "BLANK", &blank_val, comment, &status) == 0 ||
        fits_read_key(fptr, TDOUBLE, "MISSING", &blank_val, comment, &status) == 0 ||
        fits_read_key(fptr, TDOUBLE, "NODATA", &blank_val, comment, &status) == 0) {
        header.missing_value = blank_val;
    } else {
        header.missing_value = -9999.0;  // Default
        status = 0; // Reset status
    }
    
    // Read the image data
    if (!readImageData(fptr, data, header)) {
        fits_close_file(fptr, &status);
        return false;
    }
    
    // Close the file
    int close_status = 0;
    fits_close_file(fptr, &close_status);
    
    return true;
}

bool FITSReader::write(const std::string& filename, const std::vector<double>& data, 
                      const FITSHeader& header) {
    fitsfile* fptr;
    int status = 0;
    
    // Create new FITS file
    fits_create_file(&fptr, filename.c_str(), &status);
    if (status) {
        char err_text[32];
        fits_get_errstatus(status, err_text);
        last_error_ = "Error creating FITS file: " + std::string(err_text);
        return false;
    }
    
    // Create image HDU
    long naxes[2] = {header.width, header.height};
    fits_create_img(fptr, DOUBLE_IMG, 2, naxes, &status);
    if (status) {
        char err_text[32];
        fits_get_errstatus(status, err_text);
        last_error_ = "Error creating FITS image: " + std::string(err_text);
        fits_close_file(fptr, &status);
        return false;
    }
    
    // Write WCS keywords
    double cdelt1 = (header.lon_max - header.lon_min) / header.width;
    double cdelt2 = (header.lat_max - header.lat_min) / header.height;
    double crpix1 = 1.0;  // Reference pixel (1-based in FITS)
    double crpix2 = 1.0;
    double crval1 = header.lon_min;  // Value at reference pixel
    double crval2 = header.lat_min;
    
    // Add WCS headers
    fits_update_key(fptr, TDOUBLE, "CRVAL1", &crval1, "Longitude at reference pixel", &status);
    fits_update_key(fptr, TDOUBLE, "CRVAL2", &crval2, "Latitude at reference pixel", &status);
    fits_update_key(fptr, TDOUBLE, "CDELT1", &cdelt1, "Longitude step", &status);
    fits_update_key(fptr, TDOUBLE, "CDELT2", &cdelt2, "Latitude step", &status);
    fits_update_key(fptr, TDOUBLE, "CRPIX1", &crpix1, "Reference pixel X", &status);
    fits_update_key(fptr, TDOUBLE, "CRPIX2", &crpix2, "Reference pixel Y", &status);
    
    // Add missing value keyword
    double missing = header.missing_value;
    fits_update_key(fptr, TDOUBLE, "MISSING", &missing, "Missing data value", &status);
    
    // Add descriptive keywords
    std::string desc = "Geographic map";
    fits_update_key(fptr, TSTRING, "CONTENT", const_cast<char*>(desc.c_str()), 
                   "Map description", &status);
    
    // Write image data
    long fpixel[2] = {1, 1};  // Start from first pixel (FITS uses 1-based indexing)
    fits_write_pix(fptr, TDOUBLE, fpixel, header.width * header.height, 
                  const_cast<double*>(data.data()), &status);
    if (status) {
        char err_text[32];
        fits_get_errstatus(status, err_text);
        last_error_ = "Error writing FITS data: " + std::string(err_text);
        fits_close_file(fptr, &status);
        return false;
    }
    
    // Close file
    int close_status = 0;
    fits_close_file(fptr, &close_status);
    
    return (status == 0);
}

bool FITSReader::readWCS(void* fptr_void, FITSHeader& header) {
    fitsfile* fptr = static_cast<fitsfile*>(fptr_void);
    int status = 0;
    char comment[FLEN_COMMENT];
    
    // WCS parameters
    double crval1, crval2, cdelt1, cdelt2, crpix1, crpix2;
    
    // Try to read WCS keywords
    fits_read_key(fptr, TDOUBLE, "CRVAL1", &crval1, comment, &status);
    fits_read_key(fptr, TDOUBLE, "CRVAL2", &crval2, comment, &status);
    fits_read_key(fptr, TDOUBLE, "CDELT1", &cdelt1, comment, &status);
    fits_read_key(fptr, TDOUBLE, "CDELT2", &cdelt2, comment, &status);
    fits_read_key(fptr, TDOUBLE, "CRPIX1", &crpix1, comment, &status);
    fits_read_key(fptr, TDOUBLE, "CRPIX2", &crpix2, comment, &status);
    
    if (status) {
        // WCS keywords not found or error reading them
        // Reset status for further operations
        status = 0;
        
        // Check file name to see if it's likely a planetary map
        if (header.width > 0 && header.height > 0) {
            // For planetary maps, assume global coverage unless proven otherwise
            std::cout << "No standard WCS found. Assuming global planetary map covering:" << std::endl;
            std::cout << " - Longitude: 0° to 360° East" << std::endl;
            std::cout << " - Latitude: -90° to 90° North" << std::endl;
            
            header.lon_min = 0.0;
            header.lon_max = 360.0;
            header.lat_min = -90.0;
            header.lat_max = 90.0;
            
            return true;
        }
        
        last_error_ = "Failed to read WCS information and cannot determine map bounds";
        return false;
    }
    
    // Calculate bounds from WCS
    header.lon_min = crval1 - (crpix1 - 1) * cdelt1;
    header.lon_max = header.lon_min + header.width * cdelt1;
    header.lat_min = crval2 - (crpix2 - 1) * cdelt2;
    header.lat_max = header.lat_min + header.height * cdelt2;
    
    // Sanity check - if the coordinates seem too small for a global map
    if (std::abs(header.lon_max - header.lon_min) < 1.0 && 
        std::abs(header.lat_max - header.lat_min) < 1.0) {
        
        std::cout << "Warning: Coordinate range appears very small." << std::endl;
        std::cout << "  Original range: Lon [" << header.lon_min << "," << header.lon_max 
                  << "], Lat [" << header.lat_min << "," << header.lat_max << "]" << std::endl;
        std::cout << "  Assuming global planetary map instead." << std::endl;
        
        header.lon_min = 0.0;
        header.lon_max = 360.0;
        header.lat_min = -90.0;
        header.lat_max = 90.0;
    }
    
    return true;
}

bool FITSReader::readImageData(void* fptr_void, std::vector<double>& data, const FITSHeader& header) {
    fitsfile* fptr = static_cast<fitsfile*>(fptr_void);
    int status = 0;
    
    // Allocate memory for data
    data.resize(header.width * header.height);
    
    // Read image data (FITS uses 1-based indexing)
    long fpixel[2] = {1, 1};
    fits_read_pix(fptr, TDOUBLE, fpixel, header.width * header.height, 
                 const_cast<double*>(&header.missing_value), data.data(), nullptr, &status);
    
    if (status) {
        char err_text[32];
        fits_get_errstatus(status, err_text);
        last_error_ = "Error reading FITS pixel data: " + std::string(err_text);
        return false;
    }
    
    return true;
}

// Add this function to handle and report CFITSIO errors
std::string FITSReader::formatFitsError(int status) {
    if (status == 0) return "";
    
    char error_text[32];
    fits_get_errstatus(status, error_text);
    return std::string(error_text);
}

} // namespace ptm