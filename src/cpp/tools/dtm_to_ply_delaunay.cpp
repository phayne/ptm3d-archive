/**
 * DTM to PLY Converter with Delaunay Triangulation
 *
 * This version uses true 2D Delaunay triangulation on projected coordinates,
 * with edge length filtering to handle variable-density point clouds.
 * Solves the disconnected triangle problem that occurs with adaptive radial thinning.
 *
 * Ported into the PTM3D tree from the standalone heat3d/dtm working directory
 * (source md5 878b9e95c456a535e6abcf9a98a6bd27). NO ARITHMETIC CHANGED: the
 * output PLY is byte-identical to the pre-port binary on both a full-raster and
 * an ROI conversion. The differences are all disclosure or refusal:
 *
 *   - <unordered_set> is now included explicitly (it was reached transitively).
 *   - --south-pole-coords warns that it is a no-op (see below).
 *   - an unrecognized or incomplete argument warns instead of being dropped.
 *   - a zero-face mesh is REFUSED (exit 2) instead of written as a ~360-byte
 *     PLY under a "Success!" message.
 *   - the sliver percentage no longer divides by zero faces.
 *
 * See docs/dem-to-mesh-workflow.md for the full port record.
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <limits>
#include <gdal_priv.h>
#include <cpl_conv.h>
#include <ogr_spatialref.h>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <set>
#include <array>
#include <sstream>
#include <cstdint>
#include <cstring>
#include <random>
#include <queue>

// CDT library for Delaunay triangulation
#include "CDT/include/CDT.h"

// Constants for lunar calculations
const double MOON_RADIUS = 1737400.0; // meters
const double DEG_TO_RAD = M_PI / 180.0;
const double RAD_TO_DEG = 180.0 / M_PI;

// Structure for 3D vertices
struct Vertex {
    double x, y, z;        // Cartesian coordinates
    double lat, lon;       // Geographic coordinates
    double elevation;      // Elevation value
    float nx, ny, nz;      // Normal vector
    int grid_x, grid_y;    // Original grid position
    double proj_x, proj_y; // 2D projected coordinates for Delaunay triangulation (meters)
    size_t index;          // Vertex index in array
};

// Structure for mesh faces
struct Face {
    uint32_t v1, v2, v3;   // Vertex indices
    double lat, lon;       // Face center coordinates
};

// Structure for polar stereographic projection parameters
struct PolarStereoParams {
    bool is_polar_stereo = false;
    bool is_south_pole = false;
    double central_meridian = 0.0;
    double false_easting = 0.0;
    double false_northing = 0.0;
    double scale_factor = 1.0;
    double semi_major = MOON_RADIUS;
};

// Structure for projected coordinate system parameters
struct ProjectionInfo {
    bool is_projected = false;
    bool is_polar_stereo = false;
    bool is_equirectangular = false;
    double body_radius = MOON_RADIUS;
    double lat_origin = 0.0;
    double lon_origin = 0.0;
    double std_parallel = 0.0;
};

// DTM Reader class
class DTMReader {
public:
    struct DTMInfo {
        int width, height;
        double lon_min, lon_max;
        double lat_min, lat_max;
        double pixel_size_lon, pixel_size_lat;
        float nodata_value;
        double min_elevation, max_elevation;
        int valid_pixels;
        PolarStereoParams polar_params;
        ProjectionInfo proj_info;
    };

private:
    GDALDataset* dataset_;
    GDALRasterBand* band_;
    DTMInfo info_;
    double geotransform_[6];
    OGRCoordinateTransformation* coord_transform_;
    bool swap_axes_;  // Flag to swap x/y axes for polar stereographic
    double raster_scale_;   // GDAL scale factor (e.g., 0.5 for LOLA DEMs)
    double raster_offset_;  // GDAL offset (e.g., 1737400.0 for LOLA radius DEMs)

    void equirectangularToLatLon(double x, double y, double& lat, double& lon) {
        const auto& p = info_.proj_info;
        double cos_lat_ts = cos(p.std_parallel * DEG_TO_RAD);
        lon = p.lon_origin + (x / (p.body_radius * cos_lat_ts)) * RAD_TO_DEG;
        lat = p.lat_origin + (y / p.body_radius) * RAD_TO_DEG;
        while (lon > 180.0) lon -= 360.0;
        while (lon < -180.0) lon += 360.0;
    }

    void polarStereoToLatLon(double x, double y, double& lat, double& lon) {
        const auto& p = info_.polar_params;

        double e_prime = x - p.false_easting;
        double n_prime = y - p.false_northing;

        double rho = sqrt(e_prime * e_prime + n_prime * n_prime);

        if (rho < 1e-10) {
            lat = p.is_south_pole ? -90.0 : 90.0;
            lon = p.central_meridian;
            return;
        }

        // Polar stereographic inverse formulas
        if (p.is_south_pole) {
            // South pole inverse transformation
            // lat = 2 * atan(rho / (2*R)) - 90°
            double c = 2.0 * atan(rho / (2.0 * p.semi_major * p.scale_factor));
            lat = c * RAD_TO_DEG - 90.0;
            
            // For standard south polar stereographic with prime meridian "up":
            // - Positive n_prime (north/up) corresponds to 0° longitude direction
            // - Positive e_prime (east/right) corresponds to 90°E longitude direction
            // - Negative n_prime (south/down) corresponds to 180° longitude direction
            // - Negative e_prime (west/left) corresponds to 270°E longitude direction
            // Therefore: lon = atan2(e_prime, n_prime)
            lon = p.central_meridian + atan2(e_prime, n_prime) * RAD_TO_DEG;
        } else {
            // North pole inverse transformation
            double c = 2.0 * atan(rho / (2.0 * p.semi_major * p.scale_factor));
            lat = 90.0 - c * RAD_TO_DEG;
            
            // For north pole, the orientation is different
            lon = p.central_meridian + atan2(e_prime, -n_prime) * RAD_TO_DEG;
        }
        
        // Normalize longitude to [-180, 180]
        while (lon < -180.0) lon += 360.0;
        while (lon > 180.0) lon -= 360.0;
    }

public:
    DTMReader() : dataset_(nullptr), band_(nullptr), coord_transform_(nullptr), swap_axes_(false),
                   raster_scale_(1.0), raster_offset_(0.0) {
        GDALAllRegister();
        CPLSetConfigOption("PROJ_IGNORE_CELESTIAL_BODY", "YES");
    }
    
    void setSwapAxes(bool swap) {
        swap_axes_ = swap;
    }
    
    ~DTMReader() {
        if (coord_transform_) {
            OGRCoordinateTransformation::DestroyCT(coord_transform_);
        }
        if (dataset_) {
            GDALClose(dataset_);
        }
    }
    
    bool load(const std::string& filename, bool skip_bounds = false) {
        dataset_ = (GDALDataset*)GDALOpen(filename.c_str(), GA_ReadOnly);
        if (!dataset_) {
            std::cerr << "Error: Could not open file: " << filename << std::endl;
            return false;
        }

        band_ = dataset_->GetRasterBand(1);
        info_.width = band_->GetXSize();
        info_.height = band_->GetYSize();
        std::cout << "Image size: " << info_.width << " x " << info_.height << " pixels\n";
        dataset_->GetGeoTransform(geotransform_);
        
        const char* wkt = dataset_->GetProjectionRef();
        if (wkt && strlen(wkt) > 0) {
            OGRSpatialReference srs;
            srs.importFromWkt(&wkt);

            // Check if projected
            if (srs.IsProjected()) {
                info_.proj_info.is_projected = true;
                info_.proj_info.body_radius = srs.GetSemiMajor(nullptr);
                if (info_.proj_info.body_radius <= 0) {
                    info_.proj_info.body_radius = MOON_RADIUS;
                }
            }

            const char* projName = srs.GetAttrValue("PROJECTION");
            if (projName && (strstr(projName, "Equirectangular") || strstr(projName, "Equidistant_Cylindrical"))) {
                info_.proj_info.is_equirectangular = true;
                info_.proj_info.lon_origin = srs.GetProjParm(SRS_PP_CENTRAL_MERIDIAN, 0.0);
                info_.proj_info.lat_origin = srs.GetProjParm(SRS_PP_LATITUDE_OF_ORIGIN, 0.0);
                info_.proj_info.std_parallel = srs.GetProjParm(SRS_PP_STANDARD_PARALLEL_1, 0.0);

                std::cout << "Detected EQUIRECTANGULAR projection:\n";
                std::cout << "  Longitude of origin: " << info_.proj_info.lon_origin << "°\n";
                std::cout << "  Standard parallel: " << info_.proj_info.std_parallel << "°\n";
            } else if (projName && strstr(projName, "Polar_Stereographic")) {
                info_.polar_params.is_polar_stereo = true;
                info_.proj_info.is_polar_stereo = true;
                info_.polar_params.central_meridian = srs.GetProjParm(SRS_PP_CENTRAL_MERIDIAN, 0.0);
                double lat_origin = srs.GetProjParm(SRS_PP_LATITUDE_OF_ORIGIN, 0.0);
                info_.polar_params.is_south_pole = (lat_origin < 0);
                info_.polar_params.false_easting = srs.GetProjParm(SRS_PP_FALSE_EASTING, 0.0);
                info_.polar_params.false_northing = srs.GetProjParm(SRS_PP_FALSE_NORTHING, 0.0);
                info_.polar_params.scale_factor = srs.GetProjParm(SRS_PP_SCALE_FACTOR, 1.0);
                info_.polar_params.semi_major = info_.proj_info.body_radius;

                std::cout << "Detected POLAR STEREOGRAPHIC projection:\n";
                std::cout << "  Pole: " << (info_.polar_params.is_south_pole ? "SOUTH" : "NORTH") << "\n";
                std::cout << "  Central meridian: " << info_.polar_params.central_meridian << "°\n";
            }
        }
        
        int has_nodata;
        double nodata = band_->GetNoDataValue(&has_nodata);
        info_.nodata_value = has_nodata ? nodata : -9999.0f;
        if (has_nodata) {
            std::cout << "NoData value: " << nodata << "\n";
        } else {
            std::cout << "No NoData value declared in GeoTIFF (using default: -9999)\n";
        }

        // Read raster scale and offset from metadata (e.g., PDS SCALING_FACTOR/OFFSET)
        // GDAL stores these but does NOT auto-apply them to RasterIO reads.
        // Physical value = DN * scale + offset
        int has_scale = 0, has_offset = 0;
        double gdal_scale = band_->GetScale(&has_scale);
        double gdal_offset = band_->GetOffset(&has_offset);
        if (has_scale && gdal_scale != 0.0 && gdal_scale != 1.0) {
            raster_scale_ = gdal_scale;
            std::cout << "Raster scale factor: " << raster_scale_ << "\n";
        }
        if (has_offset && gdal_offset != 0.0) {
            raster_offset_ = gdal_offset;
            std::cout << "Raster offset: " << raster_offset_ << "\n";
        }
        if (raster_scale_ != 1.0 || raster_offset_ != 0.0) {
            std::cout << "  Physical value = DN * " << raster_scale_ << " + " << raster_offset_ << "\n";
            std::cout << "  NOTE: For elevation DEMs, this converts DN to height or radius.\n";
            std::cout << "  The offset (" << raster_offset_ << ") will be IGNORED if it represents\n";
            std::cout << "  a reference radius, since body_radius is added separately.\n";
            // If the offset is close to a planetary radius (> 100 km), it represents
            // the reference sphere radius. In that case, we only apply the scale factor
            // to get elevation above the datum, not the full radius.
            if (std::abs(raster_offset_) > 100000.0) {
                std::cout << "  Offset appears to be a reference radius ("
                          << raster_offset_ / 1000.0 << " km). Using scale only.\n";
                raster_offset_ = 0.0;  // Don't add reference radius; body_radius handles it
            }
        }

        if (!skip_bounds) {
            std::cout << "Calculating geographic bounds...\n";
            calculateBounds();
        } else {
            std::cout << "Skipping bounds calculation (ROI mode)\n";
            // Set dummy bounds
            info_.lat_min = -90.0;
            info_.lat_max = 90.0;
            info_.lon_min = -180.0;
            info_.lon_max = 180.0;
        }

        return true;
    }
    
    void calculateBounds() {
        if (info_.proj_info.is_projected) {
            // Projected coordinate system - convert corners to lat/lon
            double x_min = geotransform_[0];
            double x_max = geotransform_[0] + info_.width * geotransform_[1];
            double y_max = geotransform_[3];
            double y_min = geotransform_[3] + info_.height * geotransform_[5];

            double lat1, lon1, lat2, lon2, lat3, lon3, lat4, lon4;

            if (info_.polar_params.is_polar_stereo) {
                // Sample multiple points for polar stereographic
                info_.lon_min = 180.0;
                info_.lon_max = -180.0;
                info_.lat_min = 90.0;
                info_.lat_max = -90.0;

                // Use very coarse sampling for large files to speed up loading
                int sample_step = std::max(1000, std::min(info_.width, info_.height) / 20);

                for (int y = 0; y < info_.height; y += sample_step) {
                    for (int x = 0; x < info_.width; x += sample_step) {
                        double lat, lon;
                        if (pixelToLatLon(x, y, lat, lon)) {
                            info_.lon_min = std::min(info_.lon_min, lon);
                            info_.lon_max = std::max(info_.lon_max, lon);
                            info_.lat_min = std::min(info_.lat_min, lat);
                            info_.lat_max = std::max(info_.lat_max, lat);
                        }
                    }
                }
            } else if (info_.proj_info.is_equirectangular) {
                // Convert corners for equirectangular
                equirectangularToLatLon(x_min, y_min, lat1, lon1);
                equirectangularToLatLon(x_max, y_min, lat2, lon2);
                equirectangularToLatLon(x_min, y_max, lat3, lon3);
                equirectangularToLatLon(x_max, y_max, lat4, lon4);

                info_.lat_min = std::min({lat1, lat2, lat3, lat4});
                info_.lat_max = std::max({lat1, lat2, lat3, lat4});
                info_.lon_min = std::min({lon1, lon2, lon3, lon4});
                info_.lon_max = std::max({lon1, lon2, lon3, lon4});
            }
        } else {
            // Unprojected geographic coordinates
            info_.lon_min = geotransform_[0];
            info_.lon_max = geotransform_[0] + info_.width * geotransform_[1];
            info_.lat_max = geotransform_[3];
            info_.lat_min = geotransform_[3] + info_.height * geotransform_[5];
        }

        std::cout << "Geographic bounds:\n";
        std::cout << "  Longitude: " << info_.lon_min << "° to " << info_.lon_max << "°\n";
        std::cout << "  Latitude: " << info_.lat_min << "° to " << info_.lat_max << "°\n";
    }
    
    bool pixelToLatLon(int px, int py, double& lat, double& lon) {
        double proj_x = geotransform_[0] + px * geotransform_[1] + py * geotransform_[2];
        double proj_y = geotransform_[3] + px * geotransform_[4] + py * geotransform_[5];

        if (info_.proj_info.is_projected) {
            if (info_.polar_params.is_polar_stereo) {
                if (swap_axes_) {
                    polarStereoToLatLon(proj_y, proj_x, lat, lon);
                } else {
                    polarStereoToLatLon(proj_x, proj_y, lat, lon);
                }
            } else if (info_.proj_info.is_equirectangular) {
                equirectangularToLatLon(proj_x, proj_y, lat, lon);
            } else {
                // Unknown projected system
                return false;
            }
        } else {
            // Unprojected geographic coordinates
            lon = proj_x;
            lat = proj_y;
        }
        return true;
    }

    // Convert from lat/lon to polar stereographic (x,y)
    void latLonToPolarStereo(double lat, double lon, double& x, double& y) {
        const auto& p = info_.polar_params;
        
        double lat_rad = lat * DEG_TO_RAD;
        double lon_rad = lon * DEG_TO_RAD;
        double central_meridian_rad = p.central_meridian * DEG_TO_RAD;
        
        if (p.is_south_pole) {
            // South polar stereographic projection
            // Using the standard USGS/IAU formulation
            double t = tan(M_PI/4.0 + lat_rad/2.0);
            double rho = 2.0 * p.semi_major * p.scale_factor / t;
            
            double lambda = lon_rad - central_meridian_rad;
            x = rho * sin(lambda);
            y = -rho * cos(lambda);
        } else {
            // North polar stereographic
            double t = tan(M_PI/4.0 - lat_rad/2.0);
            double rho = 2.0 * p.semi_major * p.scale_factor * t;
            
            double lambda = lon_rad - central_meridian_rad;
            x = rho * sin(lambda);
            y = rho * cos(lambda);
        }
        
        // Apply false easting and northing
        x += p.false_easting;
        y += p.false_northing;
    }

    // Convert lat/lon to pixel coordinates
    bool latLonToPixel(double lat, double lon, int& px, int& py) {
        if (info_.polar_params.is_polar_stereo) {
            // Convert lat/lon to projected coordinates
            double x, y;
            latLonToPolarStereo(lat, lon, x, y);
            
            // Convert projected coordinates to pixel using inverse geotransform
            double det = geotransform_[1] * geotransform_[5] - geotransform_[2] * geotransform_[4];
            if (fabs(det) < 1e-10) {
                // Simplified case when GT[2] and GT[4] are 0
                px = (int)((x - geotransform_[0]) / geotransform_[1] + 0.5);
                py = (int)((y - geotransform_[3]) / geotransform_[5] + 0.5);
            } else {
                // General case
                double dx = x - geotransform_[0];
                double dy = y - geotransform_[3];
                px = (int)((dx * geotransform_[5] - dy * geotransform_[2]) / det + 0.5);
                py = (int)((dy * geotransform_[1] - dx * geotransform_[4]) / det + 0.5);
            }
            
            return (px >= 0 && px < info_.width && py >= 0 && py < info_.height);
        } else {
            // Simple equirectangular case
            px = (int)((lon - geotransform_[0]) / geotransform_[1] + 0.5);
            py = (int)((lat - geotransform_[3]) / geotransform_[5] + 0.5);
            
            return (px >= 0 && px < info_.width && py >= 0 && py < info_.height);
        }
    }
    
    float getElevation(int x, int y) {
        if (x < 0 || x >= info_.width || y < 0 || y >= info_.height) {
            return info_.nodata_value;
        }

        float value = info_.nodata_value;
        CPLErr err = band_->RasterIO(GF_Read, x, y, 1, 1, &value, 1, 1, GDT_Float32, 0, 0);
        if (err != CE_None) {
            return info_.nodata_value;
        }
        // Apply scale and offset from raster metadata (e.g., PDS SCALING_FACTOR)
        // This converts raw DN to physical elevation units (meters)
        if (value != info_.nodata_value) {
            value = static_cast<float>(value * raster_scale_ + raster_offset_);
        }
        return value;
    }
    
    const DTMInfo& getInfo() const { return info_; }

    // Get geotransform array for coordinate conversion
    const double* getGeoTransform() const { return geotransform_; }
};

// Mesh Generator with proximity-based triangulation
class MeshGenerator {
private:
    DTMReader& dtm_reader_;
    std::vector<Vertex> vertices_;
    std::vector<Face> faces_;
    double body_radius_;
    bool use_south_pole_coords_;  // Flag for south pole coordinate corrections
    
    // ROI parameters
    bool use_roi_;
    double roi_center_lat_;
    double roi_center_lon_;
    double roi_radius_km_;
    int roi_max_points_;
    double roi_core_frac_;  // Fraction of ROI radius kept at full resolution

    // Edge length threshold for Delaunay triangulation (0 = auto-compute)
    double max_edge_m_;

    // Density falloff exponent (1.0 = linear, 2.0 = quadratic, 0.5 = sqrt)
    double density_exp_;

    // Random number generator for probabilistic sampling
    std::mt19937 rng_;
    std::uniform_real_distribution<double> uniform_dist_;

    // QC sweep parameters
    bool qc_enabled_ = true;
    double qc_sliver_angle_ = 5.0;       // degrees
    int qc_max_hole_edges_ = 20;          // max boundary loop size to fill

    // Edge adjacency types for QC sweep
    using EdgeKey = std::pair<uint32_t, uint32_t>;
    struct EdgeKeyHash {
        size_t operator()(const EdgeKey& e) const {
            return std::hash<uint64_t>()((uint64_t)e.first << 32 | e.second);
        }
    };
    struct EdgeData {
        size_t face[2];
        uint8_t count = 0;
        void add(size_t fi) {
            if (count < 2) face[count] = fi;
            count++;
        }
    };
    using EdgeFaceMap = std::unordered_map<EdgeKey, EdgeData, EdgeKeyHash>;

    static EdgeKey makeEdgeKey(uint32_t a, uint32_t b) {
        return a < b ? EdgeKey{a, b} : EdgeKey{b, a};
    }

    EdgeFaceMap buildEdgeToFaceMap() const {
        EdgeFaceMap efm;
        efm.reserve(faces_.size() * 3 / 2);
        for (size_t fi = 0; fi < faces_.size(); ++fi) {
            const Face& f = faces_[fi];
            efm[makeEdgeKey(f.v1, f.v2)].add(fi);
            efm[makeEdgeKey(f.v2, f.v3)].add(fi);
            efm[makeEdgeKey(f.v3, f.v1)].add(fi);
        }
        return efm;
    }

    // Returns +1 if the face traverses edge (a,b) as a->b, -1 if b->a, 0 if edge not in face
    static int edgeDirection(const Face& f, uint32_t a, uint32_t b) {
        if (f.v1 == a && f.v2 == b) return +1;
        if (f.v2 == a && f.v3 == b) return +1;
        if (f.v3 == a && f.v1 == b) return +1;
        if (f.v1 == b && f.v2 == a) return -1;
        if (f.v2 == b && f.v3 == a) return -1;
        if (f.v3 == b && f.v1 == a) return -1;
        return 0;
    }

    // Compute 3D face area (half the cross product magnitude)
    double faceArea(const Face& f) const {
        const Vertex& a = vertices_[f.v1];
        const Vertex& b = vertices_[f.v2];
        const Vertex& c = vertices_[f.v3];
        double ux = b.x - a.x, uy = b.y - a.y, uz = b.z - a.z;
        double vx = c.x - a.x, vy = c.y - a.y, vz = c.z - a.z;
        double cx = uy * vz - uz * vy;
        double cy = uz * vx - ux * vz;
        double cz = ux * vy - uy * vx;
        return 0.5 * std::sqrt(cx * cx + cy * cy + cz * cz);
    }

    void removeSliverTriangles(double min_angle_deg) {
        std::cout << "QC sweep: removing slivers (min_angle < "
                  << std::fixed << std::setprecision(1) << min_angle_deg << ")...\n";
        size_t faces_before = faces_.size();
        size_t total_removed = 0;

        for (int pass = 0; pass < 2; ++pass) {
            EdgeFaceMap efm = buildEdgeToFaceMap();
            std::vector<bool> marked(faces_.size(), false);
            size_t removed_this_pass = 0;

            for (size_t fi = 0; fi < faces_.size(); ++fi) {
                const Face& f = faces_[fi];
                if (f.v1 == f.v2 || f.v2 == f.v3 || f.v3 == f.v1) {
                    marked[fi] = true;
                    removed_this_pass++;
                    continue;
                }
                double min_a = computeMinAngle(vertices_[f.v1], vertices_[f.v2], vertices_[f.v3]);
                if (min_a < min_angle_deg) {
                    // Check if this is a boundary sliver (has at least one boundary edge)
                    bool is_boundary = false;
                    EdgeKey edges[3] = {
                        makeEdgeKey(f.v1, f.v2),
                        makeEdgeKey(f.v2, f.v3),
                        makeEdgeKey(f.v3, f.v1)
                    };
                    for (int e = 0; e < 3; ++e) {
                        auto it = efm.find(edges[e]);
                        if (it != efm.end() && it->second.count == 1) {
                            is_boundary = true;
                            break;
                        }
                    }
                    if (is_boundary) {
                        marked[fi] = true;
                        removed_this_pass++;
                    }
                }
            }

            if (removed_this_pass == 0) break;

            // Compact faces
            std::vector<Face> new_faces;
            new_faces.reserve(faces_.size() - removed_this_pass);
            for (size_t fi = 0; fi < faces_.size(); ++fi) {
                if (!marked[fi]) new_faces.push_back(faces_[fi]);
            }
            faces_ = std::move(new_faces);
            total_removed += removed_this_pass;
        }

        std::cout << "  Removed " << total_removed << " boundary slivers\n";
        std::cout << "  Faces: " << faces_before << " -> " << faces_.size() << "\n";
    }

    void fixNormalOrientation() {
        std::cout << "QC sweep: fixing normal orientation (BFS flood-fill)...\n";
        EdgeFaceMap efm = buildEdgeToFaceMap();

        std::vector<bool> visited(faces_.size(), false);
        size_t num_components = 0;
        size_t num_flipped = 0;

        for (size_t seed = 0; seed < faces_.size(); ++seed) {
            if (visited[seed]) continue;

            // Find the largest-area unvisited face in this component as seed
            // (Do a quick scan from the current seed forward)
            size_t best_seed = seed;
            double best_area = faceArea(faces_[seed]);
            // Scan a bounded window to find a better seed without O(N^2)
            size_t scan_limit = std::min(seed + 10000, faces_.size());
            for (size_t si = seed + 1; si < scan_limit; ++si) {
                if (!visited[si]) {
                    double a = faceArea(faces_[si]);
                    if (a > best_area) { best_area = a; best_seed = si; }
                }
            }

            // Orient seed using dot-product-with-centroid heuristic
            {
                Face& sf = faces_[best_seed];
                const Vertex& a = vertices_[sf.v1];
                const Vertex& b = vertices_[sf.v2];
                const Vertex& c = vertices_[sf.v3];
                double ux = b.x - a.x, uy = b.y - a.y, uz = b.z - a.z;
                double vx = c.x - a.x, vy = c.y - a.y, vz = c.z - a.z;
                double nx = uy * vz - uz * vy;
                double ny = uz * vx - ux * vz;
                double nz = ux * vy - uy * vx;
                double cx = (a.x + b.x + c.x) / 3.0;
                double cy = (a.y + b.y + c.y) / 3.0;
                double cz = (a.z + b.z + c.z) / 3.0;
                if (nx * cx + ny * cy + nz * cz < 0) {
                    std::swap(sf.v2, sf.v3);
                    num_flipped++;
                }
            }

            // BFS from seed
            std::queue<size_t> bfs;
            visited[best_seed] = true;
            bfs.push(best_seed);
            num_components++;

            while (!bfs.empty()) {
                size_t fi = bfs.front();
                bfs.pop();
                const Face& f = faces_[fi];

                uint32_t verts[3] = {f.v1, f.v2, f.v3};
                for (int e = 0; e < 3; ++e) {
                    uint32_t ea = verts[e];
                    uint32_t eb = verts[(e + 1) % 3];
                    EdgeKey ek = makeEdgeKey(ea, eb);

                    auto it = efm.find(ek);
                    if (it == efm.end()) continue;
                    const EdgeData& ed = it->second;
                    if (ed.count != 2) continue; // skip boundary or non-manifold

                    size_t neighbor = (ed.face[0] == fi) ? ed.face[1] : ed.face[0];
                    if (visited[neighbor]) continue;

                    // Check winding consistency: current face traverses ea->eb
                    // Neighbor should traverse eb->ea for consistent orientation
                    int cur_dir = edgeDirection(faces_[fi], ea, eb);
                    int nbr_dir = edgeDirection(faces_[neighbor], ea, eb);
                    if (cur_dir != 0 && nbr_dir != 0 && cur_dir == nbr_dir) {
                        // Same direction = inconsistent; flip neighbor
                        std::swap(faces_[neighbor].v2, faces_[neighbor].v3);
                        num_flipped++;
                    }

                    visited[neighbor] = true;
                    bfs.push(neighbor);
                }
            }
        }

        std::cout << "  Found " << num_components << " connected component(s), flipped "
                  << num_flipped << " faces\n";
    }

    void fillSmallHoles(int max_loop_edges) {
        if (max_loop_edges <= 0) return;
        std::cout << "QC sweep: filling small holes (max loop size: " << max_loop_edges << " edges)...\n";

        EdgeFaceMap efm = buildEdgeToFaceMap();

        // Collect boundary edges (count == 1) with their directed orientation
        // For a boundary edge, the single adjacent face gives us the correct winding
        // Store as directed edges: for boundary edge (a,b) with face f,
        // store the direction that the face does NOT traverse (the "open" side)
        std::unordered_map<uint32_t, std::vector<uint32_t>> boundary_adj;
        // Also store directed boundary edges for loop tracing
        std::unordered_map<uint32_t, std::vector<uint32_t>> boundary_next;

        for (auto& [ek, ed] : efm) {
            if (ed.count != 1) continue;
            // This is a boundary edge
            const Face& f = faces_[ed.face[0]];
            // Find the directed traversal of this edge in the face
            int dir = edgeDirection(f, ek.first, ek.second);
            // The boundary loop goes in the opposite direction of the face winding
            // (the "free" side of the edge)
            uint32_t from, to;
            if (dir > 0) {
                // Face goes first->second, boundary loop goes second->first
                from = ek.second;
                to = ek.first;
            } else {
                from = ek.first;
                to = ek.second;
            }
            boundary_next[from].push_back(to);
        }

        // Trace boundary loops
        std::unordered_set<uint32_t> used_starts;
        size_t holes_filled = 0;
        size_t holes_skipped = 0;
        size_t verts_added = 0;
        size_t faces_added = 0;

        for (auto& [start, _] : boundary_next) {
            if (used_starts.count(start)) continue;

            // Try to trace a loop from this vertex
            std::vector<uint32_t> loop;
            loop.push_back(start);
            uint32_t current = start;
            bool found_loop = false;

            for (int step = 0; step < max_loop_edges + 1; ++step) {
                auto it = boundary_next.find(current);
                if (it == boundary_next.end() || it->second.empty()) break;

                // Pick the first unused neighbor
                uint32_t next = UINT32_MAX;
                for (uint32_t n : it->second) {
                    if (n == start && loop.size() >= 3) {
                        next = n;
                        found_loop = true;
                        break;
                    }
                    // Don't revisit intermediate vertices
                    bool in_loop = false;
                    for (size_t li = 1; li < loop.size(); ++li) {
                        if (loop[li] == n) { in_loop = true; break; }
                    }
                    if (!in_loop && next == UINT32_MAX) next = n;
                }

                if (found_loop) break;
                if (next == UINT32_MAX) break;

                loop.push_back(next);
                current = next;
            }

            if (!found_loop) continue;

            // Mark all vertices in this loop as used
            for (uint32_t v : loop) used_starts.insert(v);

            int loop_size = (int)loop.size();
            if (loop_size > max_loop_edges) {
                holes_skipped++;
                continue;
            }

            if (loop_size == 3) {
                // Simple triangle fill
                Face nf;
                nf.v1 = loop[0];
                nf.v2 = loop[1];
                nf.v3 = loop[2];
                const Vertex& a = vertices_[nf.v1];
                const Vertex& b = vertices_[nf.v2];
                const Vertex& c = vertices_[nf.v3];
                nf.lat = (a.lat + b.lat + c.lat) / 3.0;
                nf.lon = (a.lon + b.lon + c.lon) / 3.0;
                faces_.push_back(nf);
                faces_added++;
            } else {
                // Fan triangulation from centroid
                Vertex centroid = {};
                for (uint32_t vi : loop) {
                    const Vertex& v = vertices_[vi];
                    centroid.x += v.x; centroid.y += v.y; centroid.z += v.z;
                    centroid.lat += v.lat; centroid.lon += v.lon;
                    centroid.elevation += v.elevation;
                    centroid.proj_x += v.proj_x; centroid.proj_y += v.proj_y;
                }
                double inv = 1.0 / loop_size;
                centroid.x *= inv; centroid.y *= inv; centroid.z *= inv;
                centroid.lat *= inv; centroid.lon *= inv;
                centroid.elevation *= inv;
                centroid.proj_x *= inv; centroid.proj_y *= inv;
                centroid.nx = 0; centroid.ny = 0; centroid.nz = 0;
                centroid.grid_x = -1; centroid.grid_y = -1;
                centroid.index = vertices_.size();

                uint32_t ci = (uint32_t)vertices_.size();
                vertices_.push_back(centroid);
                verts_added++;

                for (int i = 0; i < loop_size; ++i) {
                    Face nf;
                    nf.v1 = ci;
                    nf.v2 = loop[i];
                    nf.v3 = loop[(i + 1) % loop_size];
                    const Vertex& a = vertices_[nf.v1];
                    const Vertex& b = vertices_[nf.v2];
                    const Vertex& c = vertices_[nf.v3];
                    nf.lat = (a.lat + b.lat + c.lat) / 3.0;
                    nf.lon = (a.lon + b.lon + c.lon) / 3.0;
                    faces_.push_back(nf);
                    faces_added++;
                }
            }
            holes_filled++;
        }

        std::cout << "  Found " << (holes_filled + holes_skipped) << " boundary loops, filled "
                  << holes_filled << " holes";
        if (verts_added > 0 || faces_added > 0) {
            std::cout << " (added " << verts_added << " vertices, " << faces_added << " faces)";
        }
        std::cout << "\n";
        if (holes_skipped > 0) {
            std::cout << "  Skipped " << holes_skipped << " holes (> " << max_loop_edges << " edges)\n";
        }
    }

    void runQualityControlSweep() {
        std::cout << "\n--- Mesh Quality Control Sweep ---\n";
        if (qc_sliver_angle_ > 0) {
            removeSliverTriangles(qc_sliver_angle_);
        }
        fixNormalOrientation();
        if (qc_max_hole_edges_ > 0) {
            fillSmallHoles(qc_max_hole_edges_);
        }
        std::cout << "--- QC sweep complete ---\n";
    }

    // Compute minimum angle in a triangle (in degrees)
    double computeMinAngle(const Vertex& a, const Vertex& b, const Vertex& c) {
        // Use projected 2D coordinates for angle computation
        // Vector AB and AC
        double abx = b.proj_x - a.proj_x;
        double aby = b.proj_y - a.proj_y;
        double acx = c.proj_x - a.proj_x;
        double acy = c.proj_y - a.proj_y;

        // Vector BA and BC
        double bax = a.proj_x - b.proj_x;
        double bay = a.proj_y - b.proj_y;
        double bcx = c.proj_x - b.proj_x;
        double bcy = c.proj_y - b.proj_y;

        // Vector CA and CB
        double cax = a.proj_x - c.proj_x;
        double cay = a.proj_y - c.proj_y;
        double cbx = b.proj_x - c.proj_x;
        double cby = b.proj_y - c.proj_y;

        // Compute angles using dot product: cos(angle) = (u·v) / (|u||v|)
        auto angle_between = [](double ux, double uy, double vx, double vy) -> double {
            double dot = ux * vx + uy * vy;
            double mag_u = std::sqrt(ux * ux + uy * uy);
            double mag_v = std::sqrt(vx * vx + vy * vy);
            if (mag_u < 1e-12 || mag_v < 1e-12) return 0.0;
            double cos_angle = dot / (mag_u * mag_v);
            // Clamp to [-1, 1] to handle numerical errors
            cos_angle = std::max(-1.0, std::min(1.0, cos_angle));
            return std::acos(cos_angle) * RAD_TO_DEG;
        };

        double angle_a = angle_between(abx, aby, acx, acy);
        double angle_b = angle_between(bax, bay, bcx, bcy);
        double angle_c = angle_between(cax, cay, cbx, cby);

        return std::min({angle_a, angle_b, angle_c});
    }
    
    void latLonToCartesian(double lat, double lon, double elev, double& x, double& y, double& z) {
        double lat_rad = lat * DEG_TO_RAD;
        double lon_rad = lon * DEG_TO_RAD;
        double r = body_radius_ + elev;
        
        // Standard SPICE lunar body-fixed coordinate system:
        // +X: through 0°N, 0°E (prime meridian at equator)
        // +Y: through 0°N, 90°E
        // +Z: through north pole (90°N)
        // 
        // This is the standard spherical to Cartesian conversion:
        x = r * cos(lat_rad) * cos(lon_rad);
        y = r * cos(lat_rad) * sin(lon_rad);
        z = r * sin(lat_rad);
        
        // At south pole (lat = -90°):
        // - z = r * sin(-90°) = -r (correctly negative)
        // - x and y approach 0 as lat approaches ±90°
    }
    
    double geodesicDistance(double lat1, double lon1, double lat2, double lon2, double radius) {
        double dlat = (lat2 - lat1) * DEG_TO_RAD;
        double dlon = (lon2 - lon1) * DEG_TO_RAD;
        double a = sin(dlat/2) * sin(dlat/2) +
                   cos(lat1 * DEG_TO_RAD) * cos(lat2 * DEG_TO_RAD) *
                   sin(dlon/2) * sin(dlon/2);
        double c = 2 * atan2(sqrt(a), sqrt(1-a));
        return radius * c / 1000.0;  // Return in km
    }
    
    void calculateNormal(const Vertex& a, const Vertex& b, const Vertex& c, 
                        float& nx, float& ny, float& nz) {
        double ux = b.x - a.x;
        double uy = b.y - a.y;
        double uz = b.z - a.z;
        
        double vx = c.x - a.x;
        double vy = c.y - a.y;
        double vz = c.z - a.z;
        
        nx = uy * vz - uz * vy;
        ny = uz * vx - ux * vz;
        nz = ux * vy - uy * vx;
        
        float len = sqrt(nx*nx + ny*ny + nz*nz);
        if (len > 0) {
            nx /= len;
            ny /= len;
            nz /= len;
        }
    }
    
    // Build triangulation using Delaunay algorithm with edge length filtering
    // Edge filtering is optional: set max_edge_m <= 0 to disable
    void buildDelaunayTriangulation(double max_edge_m) {
        if (vertices_.size() < 3) return;

        std::cout << "Building Delaunay triangulation (" << vertices_.size() << " points)...\n";

        // Build CDT point vector from vertices' 2D projections
        std::vector<CDT::V2d<double>> points;
        points.reserve(vertices_.size());
        for (const auto& v : vertices_) {
            points.push_back(CDT::V2d<double>(v.proj_x, v.proj_y));
        }

        // Compute Delaunay triangulation
        CDT::Triangulation<double> cdt;
        cdt.insertVertices(points);
        cdt.eraseSuperTriangle();  // Remove infinite bounding triangle

        std::cout << "  Delaunay produced " << cdt.triangles.size() << " triangles\n";

        bool use_edge_filter = (max_edge_m > 0);
        if (use_edge_filter) {
            std::cout << "  Edge threshold: " << max_edge_m << " m\n";
        } else {
            std::cout << "  Edge filtering: DISABLED (keeping all triangles)\n";
        }

        // Process triangles
        faces_.clear();
        faces_.reserve(cdt.triangles.size());
        int filtered_edge = 0;
        int filtered_degenerate = 0;

        // Statistics for triangle quality
        int slivers_10 = 0;  // min angle < 10°
        int slivers_20 = 0;  // min angle < 20°
        double min_angle_seen = 180.0;
        double max_edge_seen = 0.0;

        // Lambda for squared edge length calculation (fast, no sqrt)
        auto edge_len_sq = [](const Vertex& p, const Vertex& q) {
            double dx = p.proj_x - q.proj_x;
            double dy = p.proj_y - q.proj_y;
            return dx*dx + dy*dy;
        };

        double threshold_sq = max_edge_m * max_edge_m;
        const double DEGENERATE_AREA_SQ = 1e-12;  // Skip truly degenerate triangles

        for (const auto& tri : cdt.triangles) {
            size_t i0 = static_cast<size_t>(tri.vertices[0]);
            size_t i1 = static_cast<size_t>(tri.vertices[1]);
            size_t i2 = static_cast<size_t>(tri.vertices[2]);

            const Vertex& a = vertices_[i0];
            const Vertex& b = vertices_[i1];
            const Vertex& c = vertices_[i2];

            // Check all edge lengths
            double d_ab_sq = edge_len_sq(a, b);
            double d_bc_sq = edge_len_sq(b, c);
            double d_ca_sq = edge_len_sq(c, a);

            // Check edge length threshold (if enabled)
            if (use_edge_filter) {
                if (d_ab_sq > threshold_sq || d_bc_sq > threshold_sq || d_ca_sq > threshold_sq) {
                    filtered_edge++;
                    continue;
                }
            }

            // Skip degenerate triangles (near-zero area)
            // Area² = 0.25 * |AB × AC|² using 2D cross product
            double abx = b.proj_x - a.proj_x, aby = b.proj_y - a.proj_y;
            double acx = c.proj_x - a.proj_x, acy = c.proj_y - a.proj_y;
            double cross = abx * acy - aby * acx;
            double area_sq = 0.25 * cross * cross;
            if (area_sq < DEGENERATE_AREA_SQ) {
                filtered_degenerate++;
                continue;
            }

            // Track statistics for kept triangles only
            double max_edge_this = std::sqrt(std::max({d_ab_sq, d_bc_sq, d_ca_sq}));
            max_edge_seen = std::max(max_edge_seen, max_edge_this);
            double min_angle = computeMinAngle(a, b, c);
            min_angle_seen = std::min(min_angle_seen, min_angle);
            if (min_angle < 10.0) slivers_10++;
            if (min_angle < 20.0) slivers_20++;

            // Add face with proper normal orientation
            Face f;
            f.v1 = i0;
            f.v2 = i1;
            f.v3 = i2;

            // Normal orientation is fixed globally by QC sweep (BFS flood-fill)

            f.lat = (a.lat + b.lat + c.lat) / 3.0;
            f.lon = (a.lon + b.lon + c.lon) / 3.0;

            faces_.push_back(f);
        }

        std::cout << "  Kept " << faces_.size() << " triangles\n";
        if (filtered_edge > 0) {
            std::cout << "  Filtered " << filtered_edge << " long-edge triangles\n";
        }
        if (filtered_degenerate > 0) {
            std::cout << "  Filtered " << filtered_degenerate << " degenerate triangles\n";
        }
        std::cout << "  Triangle quality: min_angle=" << std::fixed << std::setprecision(1)
                  << min_angle_seen << "°, max_edge=" << std::setprecision(0) << max_edge_seen << "m\n";
        if (faces_.empty()) {
            // Every candidate triangle was filtered. Reporting "nan%" here
            // buries the one line that matters at the one moment it matters.
            std::cout << "  Slivers: no triangles survived filtering\n";
        } else {
            std::cout << "  Slivers: " << slivers_10 << " triangles <10°, "
                      << slivers_20 << " triangles <20° ("
                      << std::setprecision(1) << (100.0 * slivers_20 / faces_.size()) << "%)\n";
        }
    }

public:
    MeshGenerator(DTMReader& reader, double radius = MOON_RADIUS)
        : dtm_reader_(reader), body_radius_(radius),
          use_south_pole_coords_(false), use_roi_(false), roi_max_points_(0),
          roi_core_frac_(0.15), max_edge_m_(std::numeric_limits<double>::quiet_NaN()),
          density_exp_(2.0), rng_(42), uniform_dist_(0.0, 1.0) {}


    void setSouthPoleCoords(bool use_sp) { use_south_pole_coords_ = use_sp; }

    void setMaxEdge(double max_edge_m) { max_edge_m_ = max_edge_m; }

    void setDensityExp(double exp) { density_exp_ = exp; }

    void setQCEnabled(bool enabled) { qc_enabled_ = enabled; }
    void setQCSliverAngle(double deg) { qc_sliver_angle_ = deg; }
    void setQCMaxHoleEdges(int n) { qc_max_hole_edges_ = n; }

    void setROI(double center_lat, double center_lon, double radius_km, int max_points = 0, double core_frac = 0.15) {
        use_roi_ = true;
        roi_center_lat_ = center_lat;
        roi_center_lon_ = center_lon;
        roi_radius_km_ = radius_km;
        roi_max_points_ = max_points;
        roi_core_frac_ = core_frac;
    }
    
    bool generateMesh(int downsample_factor) {
        const auto& info = dtm_reader_.getInfo();
        if (downsample_factor < 1) downsample_factor = 1;
        
        vertices_.clear();
        faces_.clear();
        
        // Progressive ROI sampling: density decreases with distance from ROI center
        // Uses probabilistic sampling for smooth density transitions
        double beta = 0.0;
        double core_radius_km = 0.0;
        int anchor_x = -1, anchor_y = -1;
        bool thin_entire_domain = false;
        const double p_exp = density_exp_;  // Configurable density falloff exponent

        if (use_roi_ && roi_max_points_ > 0) {
            std::cout << "Finding ROI center pixel (coarse search)...\n";
            // Find pixel coordinates of ROI center using coarse sampling
            int search_step = std::max(100, std::max(info.width, info.height) / 200);
            for (int y = 0; y < info.height && anchor_x < 0; y += search_step) {
                for (int x = 0; x < info.width; x += search_step) {
                    double lat, lon;
                    if (!dtm_reader_.pixelToLatLon(x, y, lat, lon)) continue;
                    double d_km = geodesicDistance(roi_center_lat_, roi_center_lon_, lat, lon, body_radius_);
                    if (d_km < roi_radius_km_ * 0.5) {  // Within ROI vicinity
                        anchor_x = x;
                        anchor_y = y;
                        break;
                    }
                }
            }
            if (anchor_x < 0) {
                std::cout << "  Warning: ROI center not found, using image center\n";
                anchor_x = info.width / 2;
                anchor_y = info.height / 2;
            } else {
                std::cout << "  Found anchor at pixel (" << anchor_x << ", " << anchor_y << ")\n";
            }

            core_radius_km = roi_radius_km_ * roi_core_frac_;

            // Check if core alone exceeds budget - if so, thin entire domain
            int estimate_sample = std::max(5, std::min(info.width, info.height) / 1000);

            // Quick estimate of core points
            int core_estimate = 0;
            for (int y = 0; y < info.height; y += estimate_sample) {
                for (int x = 0; x < info.width; x += estimate_sample) {
                    double lat, lon;
                    if (!dtm_reader_.pixelToLatLon(x, y, lat, lon)) continue;
                    double d_km = geodesicDistance(roi_center_lat_, roi_center_lon_, lat, lon, body_radius_);
                    if (d_km <= core_radius_km) core_estimate++;
                }
            }
            core_estimate = core_estimate * (estimate_sample * estimate_sample) / (downsample_factor * downsample_factor);

            if (core_estimate > roi_max_points_) {
                thin_entire_domain = true;
                core_radius_km = 0.0;  // No protected core
                std::cout << "  Core alone (" << core_estimate << " pts) exceeds budget - thinning entire ROI\n";
            }

            // Lambda to estimate kept points for a given beta using probabilistic model
            auto estimate_kept = [&](double beta_val) -> int {
                double expected_kept = 0.0;
                int total_in_roi = 0;

                // Sample a subset of the ROI to estimate
                for (int y = 0; y < info.height; y += estimate_sample) {
                    for (int x = 0; x < info.width; x += estimate_sample) {
                        double lat, lon;
                        if (!dtm_reader_.pixelToLatLon(x, y, lat, lon)) continue;
                        double d_km = geodesicDistance(roi_center_lat_, roi_center_lon_, lat, lon, body_radius_);
                        if (d_km > roi_radius_km_) continue;

                        total_in_roi++;

                        // Calculate normalized distance for density
                        double norm_dist = thin_entire_domain ?
                            (d_km / roi_radius_km_) :
                            ((d_km - core_radius_km) / (roi_radius_km_ - core_radius_km));

                        // Calculate keep probability (continuous, not discrete stride)
                        double keep_prob = 1.0;
                        if (thin_entire_domain || d_km > core_radius_km) {
                            if (norm_dist > 0) {
                                // Density falls off as 1 / (1 + beta * norm_dist^p)^2
                                double density_factor = 1.0 + beta_val * std::pow(norm_dist, p_exp);
                                keep_prob = 1.0 / (density_factor * density_factor);
                            }
                        }

                        expected_kept += keep_prob;
                    }
                }

                // Scale up: we sampled every estimate_sample pixels, actual collection is every downsample_factor
                if (total_in_roi == 0) return 0;
                double scale = (double)(estimate_sample * estimate_sample) / (double)(downsample_factor * downsample_factor);
                return (int)(expected_kept * scale);
            };

            std::cout << "Computing adaptive beta parameter (binary search)...\n";

            // Binary search for optimal beta
            double beta_lo = 0.0;
            double beta_hi = 1.0;

            // Find upper bound
            int kept_at_zero = estimate_kept(0.0);
            std::cout << "  Points with beta=0: " << kept_at_zero << "\n";

            for (int iter = 0; iter < 25; ++iter) {
                int kept_hi = estimate_kept(beta_hi);
                std::cout << "  beta=" << beta_hi << " -> " << kept_hi << " points\n";
                if (kept_hi <= roi_max_points_ || beta_hi > 1e6) break;
                beta_hi *= 2.0;
            }

            // Binary search
            beta = beta_hi;
            for (int iter = 0; iter < 40; ++iter) {
                double beta_mid = 0.5 * (beta_lo + beta_hi);
                int kept_mid = estimate_kept(beta_mid);
                if (kept_mid > roi_max_points_) {
                    beta_lo = beta_mid;
                } else {
                    beta_hi = beta_mid;
                    beta = beta_mid;
                }
                if (std::fabs(beta_hi - beta_lo) / (beta_lo + 1e-9) < 1e-4) break;
            }

            std::cout << "  Beta search complete: beta=" << beta << "\n";

            std::cout << "Progressive ROI probabilistic sampling:\n";
            if (thin_entire_domain) {
                std::cout << "  Mode: Entire ROI thinned (core exceeded budget)\n";
            } else {
                std::cout << "  Core radius: " << core_radius_km << " km (fraction: " << roi_core_frac_ << ")\n";
            }
            std::cout << "  Outer radius: " << roi_radius_km_ << " km\n";
            std::cout << "  Beta: " << beta << ", density exponent: " << p_exp << "\n";
            std::cout << "  Density at boundary: " << (1.0 / ((1.0 + beta) * (1.0 + beta))) * 100.0 << "% of center\n";
            std::cout << "  Estimated points: " << estimate_kept(beta) << " / " << roi_max_points_ << "\n";
        }

        // Collect vertices with progressive sampling
        std::cout << "Collecting vertices...\n";
        int processed = 0;
        int skipped_nodata = 0;
        int skipped_roi = 0;
        int skipped_stride = 0;

        for (int y = 0; y < info.height; y += downsample_factor) {
            for (int x = 0; x < info.width; x += downsample_factor) {
                processed++;

                double lat, lon;
                if (!dtm_reader_.pixelToLatLon(x, y, lat, lon)) continue;

                // ROI filtering with progressive stride
                if (use_roi_) {
                    double d_km = geodesicDistance(roi_center_lat_, roi_center_lon_, lat, lon, body_radius_);
                    if (d_km > roi_radius_km_) {
                        skipped_roi++;
                        continue;
                    }

                    // Apply probabilistic thinning if max_points is set
                    // This gives smooth density transitions instead of discrete bands
                    if (roi_max_points_ > 0) {
                        double norm_dist = thin_entire_domain ?
                            (d_km / roi_radius_km_) :
                            ((d_km - core_radius_km) / (roi_radius_km_ - core_radius_km));

                        double keep_prob = 1.0;
                        if (thin_entire_domain || d_km > core_radius_km) {
                            if (norm_dist > 0) {
                                // Density falls off as 1 / (1 + beta * norm_dist^p)^2
                                double density_factor = 1.0 + beta * std::pow(norm_dist, p_exp);
                                keep_prob = 1.0 / (density_factor * density_factor);
                            }
                        }

                        // Probabilistic sampling: keep point with probability keep_prob
                        if (uniform_dist_(rng_) >= keep_prob) {
                            skipped_stride++;
                            continue;
                        }
                    }
                }

                float elev = dtm_reader_.getElevation(x, y);

                // NoData handling: skip non-finite values, values matching
                // the declared NoData (with tolerance for float round-trip),
                // and extreme values near ±FLT_MAX (common SOCET Set fill)
                bool is_nodata = !std::isfinite(elev) || std::abs(elev) > 1e37f;
                if (!is_nodata) {
                    float diff = std::abs(elev - info.nodata_value);
                    float magnitude = std::max(std::abs(elev), std::abs(info.nodata_value));
                    if (magnitude > 0 && diff / magnitude < 1e-5f) {
                        is_nodata = true;
                    }
                }
                if (is_nodata) {
                    skipped_nodata++;
                    continue;
                }

                Vertex v;
                v.lat = lat;
                v.lon = lon;
                v.elevation = elev;
                latLonToCartesian(lat, lon, elev, v.x, v.y, v.z);
                v.nx = v.ny = v.nz = 0.0f;
                v.grid_x = x;
                v.grid_y = y;
                // Compute projected coordinates for Delaunay triangulation
                // Uses the DTM's native coordinate reference system (meters for projected CRS)
                const double* gt = dtm_reader_.getGeoTransform();
                v.proj_x = gt[0] + x * gt[1] + y * gt[2];
                v.proj_y = gt[3] + x * gt[4] + y * gt[5];
                v.index = vertices_.size();
                vertices_.push_back(v);
            }
        }
        
        std::cout << "Processed " << processed << " points\n";
        std::cout << "Created " << vertices_.size() << " vertices\n";
        std::cout << "Skipped " << skipped_nodata << " NoData points\n";
        if (use_roi_) {
            std::cout << "Skipped " << skipped_roi << " points outside ROI\n";
            if (roi_max_points_ > 0) {
                std::cout << "Skipped " << skipped_stride << " points due to probabilistic thinning\n";
            }
        }

        // Compute edge threshold for Delaunay triangulation
        // NaN = auto-compute, 0 = disable filtering, >0 = use specified value
        double edge_threshold = max_edge_m_;
        if (std::isnan(edge_threshold)) {
            // Auto-compute based on pixel size and density at ROI boundary
            const double* gt = dtm_reader_.getGeoTransform();
            double pixel_size_m = std::abs(gt[1]);  // meters per pixel

            // At ROI boundary, density factor = 1 + beta, so spacing ~ sqrt(density_factor^2) * base_spacing
            double max_density_factor = 1.0;
            if (use_roi_ && roi_max_points_ > 0 && beta > 0) {
                max_density_factor = 1.0 + beta;  // At normalized distance = 1.0
            }

            // Allow edges up to 3x the expected point spacing at boundary
            edge_threshold = pixel_size_m * downsample_factor * max_density_factor * 3.0;
            std::cout << "Auto-computed edge threshold: " << edge_threshold << " m "
                      << "(pixel=" << pixel_size_m << "m, max_density_factor=" << max_density_factor << ")\n";
        } else if (edge_threshold == 0.0) {
            std::cout << "Edge filtering: DISABLED by user (--max-edge 0)\n";
        } else {
            std::cout << "Edge threshold: " << edge_threshold << " m (user-specified)\n";
        }

        // Build Delaunay triangulation with edge filtering
        buildDelaunayTriangulation(edge_threshold);

        // Post-processing quality control sweep
        if (qc_enabled_) {
            runQualityControlSweep();
        }

        // Calculate vertex normals (sized after QC, which may add vertices)
        std::vector<float> accx(vertices_.size(), 0), accy(vertices_.size(), 0), accz(vertices_.size(), 0);
        for (const auto& f : faces_) {
            const Vertex& a = vertices_[f.v1];
            const Vertex& b = vertices_[f.v2];
            const Vertex& c = vertices_[f.v3];
            float nx, ny, nz;
            calculateNormal(a, b, c, nx, ny, nz);
            
            accx[f.v1] += nx; accy[f.v1] += ny; accz[f.v1] += nz;
            accx[f.v2] += nx; accy[f.v2] += ny; accz[f.v2] += nz;
            accx[f.v3] += nx; accy[f.v3] += ny; accz[f.v3] += nz;
        }
        
        for (size_t i = 0; i < vertices_.size(); ++i) {
            double m = std::sqrt(accx[i]*accx[i] + accy[i]*accy[i] + accz[i]*accz[i]);
            if (m > 0) {
                vertices_[i].nx = accx[i] / m;
                vertices_[i].ny = accy[i] / m;
                vertices_[i].nz = accz[i] / m;
            }
        }
        
        return true;
    }
    
    size_t vertexCount() const { return vertices_.size(); }
    size_t faceCount() const { return faces_.size(); }

    bool exportToPLY(const std::string& filename) {
        std::cout << "Exporting mesh to PLY: " << filename << "\n";
        std::cout << "  Vertices: " << vertices_.size() << "\n";
        std::cout << "  Faces: " << faces_.size() << "\n";
        
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file for writing: " << filename << std::endl;
            return false;
        }
        
        // Write PLY header
        file << "ply\n";
        file << "format binary_little_endian 1.0\n";
        file << "comment DTM to PLY with Delaunay triangulation for PTM3D\n";
        file << "element vertex " << vertices_.size() << "\n";
        file << "property double x\n";
        file << "property double y\n";
        file << "property double z\n";
        file << "property float nx\n";
        file << "property float ny\n";
        file << "property float nz\n";
        file << "property double latitude\n";
        file << "property double longitude\n";
        file << "property double elevation\n";
        file << "element face " << faces_.size() << "\n";
        file << "property list uchar int vertex_indices\n";
        file << "end_header\n";
        
        // Write vertices (double xyz, float normals, double lat/lon/elev)
        for (const auto& v : vertices_) {
            double coords[3] = {v.x, v.y, v.z};
            file.write(reinterpret_cast<const char*>(coords), sizeof(coords));
            float normals[3] = {v.nx, v.ny, v.nz};
            file.write(reinterpret_cast<const char*>(normals), sizeof(normals));
            double geo[3] = {v.lat, v.lon, v.elevation};
            file.write(reinterpret_cast<const char*>(geo), sizeof(geo));
        }
        
        // Write faces (count + 3 vertex indices only)
        for (const auto& f : faces_) {
            unsigned char count = 3;
            file.write(reinterpret_cast<const char*>(&count), sizeof(count));

            int indices[3] = {
                static_cast<int>(f.v1),
                static_cast<int>(f.v2),
                static_cast<int>(f.v3)
            };
            file.write(reinterpret_cast<const char*>(indices), sizeof(indices));
        }
        
        file.close();
        return true;
    }
};

// Main function
int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " input.tif output.ply [options]\n";
        std::cout << "Options:\n";
        std::cout << "  --downsample N         : Downsample factor (default: 1)\n";
        std::cout << "  --roi-center lat lon   : ROI center coordinates\n";
        std::cout << "  --roi-radius km        : ROI radius in kilometers\n";
        std::cout << "  --roi-max-points N     : Maximum points in ROI\n";
        std::cout << "  --roi-core-frac f      : Fraction of ROI radius at full res (default: 0.15)\n";
        std::cout << "  --density-exp p        : Density falloff exponent (default: 2.0)\n";
        std::cout << "                           1.0 = linear, 2.0 = quadratic, 0.5 = sqrt\n";
        std::cout << "                           Lower values = gentler falloff\n";
        std::cout << "  --max-edge meters      : Maximum triangle edge length\n";
        std::cout << "                           0 = disable edge filtering (no holes)\n";
        std::cout << "                           >0 = filter triangles with edges longer than this\n";
        std::cout << "                           default: auto-compute based on point density\n";
        std::cout << "  --body-radius meters   : Body radius (default: 1737400)\n";
        std::cout << "  --swap-axes            : Swap X/Y axes in polar stereographic projection\n";
        std::cout << "  --south-pole-coords    : ACCEPTED BUT NOT IMPLEMENTED (no-op).\n";
        std::cout << "                           The flag is parsed and stored, but no south-pole\n";
        std::cout << "                           coordinate correction is ever applied. It has\n";
        std::cout << "                           never done anything; meshes built with and\n";
        std::cout << "                           without it are identical. Use --swap-axes for a\n";
        std::cout << "                           real polar-stereographic axis swap.\n";
        std::cout << "  --no-qc                : Disable post-processing quality control sweep\n";
        std::cout << "  --qc-sliver-angle DEG  : Min angle for sliver removal (default: 5.0)\n";
        std::cout << "  --qc-max-hole N        : Max boundary loop edges to fill (default: 20, 0=disable)\n";
        return 1;
    }
    
    std::string input_path = argv[1];
    std::string output_path = argv[2];
    
    int downsample_factor = 1;
    double body_radius = MOON_RADIUS;
    bool body_radius_set = false;
    bool use_roi = false;
    double roi_center_lat = 0, roi_center_lon = 0;
    double roi_radius_km = 0;
    int roi_max_points = 0;
    double roi_core_frac = 0.15;
    double density_exp = 2.0;  // Density falloff exponent
    double max_edge_m = std::numeric_limits<double>::quiet_NaN();  // NaN = auto-compute
    bool swap_axes = false;
    bool south_pole_coords = false;
    bool qc_enabled = true;
    double qc_sliver_angle = 5.0;
    int qc_max_hole = 20;
    
    // Parse command line arguments
    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--downsample" && i + 1 < argc) {
            downsample_factor = std::atoi(argv[++i]);
        } else if (arg == "--swap-axes") {
            swap_axes = true;
        } else if (arg == "--south-pole-coords") {
            south_pole_coords = true;
        } else if (arg == "--body-radius" && i + 1 < argc) {
            body_radius = std::atof(argv[++i]);
            body_radius_set = true;
        } else if (arg == "--roi-center" && i + 2 < argc) {
            use_roi = true;
            roi_center_lat = std::atof(argv[++i]);
            roi_center_lon = std::atof(argv[++i]);
        } else if (arg == "--roi-radius" && i + 1 < argc) {
            roi_radius_km = std::atof(argv[++i]);
        } else if (arg == "--roi-max-points" && i + 1 < argc) {
            roi_max_points = (int)std::atof(argv[++i]);
        } else if (arg == "--roi-core-frac" && i + 1 < argc) {
            roi_core_frac = std::atof(argv[++i]);
            if (roi_core_frac < 0.0) roi_core_frac = 0.0;
            if (roi_core_frac > 1.0) roi_core_frac = 1.0;
        } else if (arg == "--density-exp" && i + 1 < argc) {
            density_exp = std::atof(argv[++i]);
            if (density_exp < 0.1) density_exp = 0.1;  // Prevent too small values
        } else if (arg == "--max-edge" && i + 1 < argc) {
            max_edge_m = std::atof(argv[++i]);
            if (max_edge_m < 0) max_edge_m = 0;  // Negative means disable (same as 0)
        } else if (arg == "--no-qc") {
            qc_enabled = false;
        } else if (arg == "--qc-sliver-angle" && i + 1 < argc) {
            qc_sliver_angle = std::atof(argv[++i]);
        } else if (arg == "--qc-max-hole" && i + 1 < argc) {
            qc_max_hole = std::atoi(argv[++i]);
        } else {
            // Unrecognized token, or a known flag missing its value. Either way
            // the request cannot be honored as written, so say so rather than
            // dropping it silently -- a typo used to change the mesh with no
            // trace in the log.
            std::cerr << "WARNING: ignoring unrecognized or incomplete argument: "
                      << arg << "\n";
        }
    }
    
    std::cout << "==============================================\n";
    std::cout << "   DTM to PLY Converter with Delaunay Triangulation\n";
    std::cout << "==============================================\n";

    // --south-pole-coords is parsed and stored but never read by the mesh
    // generator: it is a no-op. Say so instead of silently ignoring it.
    if (south_pole_coords) {
        std::cerr << "WARNING: --south-pole-coords is accepted but NOT IMPLEMENTED "
                     "(it is a no-op).\n"
                     "         The output mesh is identical with and without this flag.\n"
                     "         For a polar-stereographic axis swap, use --swap-axes.\n";
    }

    // Create DTM reader and load GeoTIFF
    DTMReader dtm_reader;
    dtm_reader.setSwapAxes(swap_axes);  // Apply axis swap setting

    std::cout << "Loading GeoTIFF: " << input_path << std::endl;
    if (!dtm_reader.load(input_path, use_roi)) {
        return 1;
    }

    // Auto-detect body radius from GeoTIFF CRS if not explicitly set
    if (!body_radius_set) {
        double gdal_radius = dtm_reader.getInfo().proj_info.body_radius;
        if (gdal_radius > 0) {
            body_radius = gdal_radius;
            std::cout << "Using body radius from GeoTIFF CRS: " << body_radius << " m\n";
        }
    }

    // Create mesh generator (after load so body radius is resolved)
    MeshGenerator generator(dtm_reader, body_radius);
    generator.setSouthPoleCoords(south_pole_coords);  // Apply south pole coordinate system
    generator.setMaxEdge(max_edge_m);
    generator.setDensityExp(density_exp);
    generator.setQCEnabled(qc_enabled);
    generator.setQCSliverAngle(qc_sliver_angle);
    generator.setQCMaxHoleEdges(qc_max_hole);

    if (use_roi) {
        generator.setROI(roi_center_lat, roi_center_lon, roi_radius_km, roi_max_points, roi_core_frac);
    }
    
    // Generate mesh
    if (!generator.generateMesh(downsample_factor)) {
        std::cerr << "Error: Mesh generation failed\n";
        return 1;
    }

    // Refuse to write an empty mesh. Reaching zero faces here is always a
    // request that could not be honored -- an ROI center off the raster, an
    // ROI radius smaller than a pixel, a --max-edge below the point spacing,
    // or an all-NoData region. Writing the file anyway and printing "Success!"
    // hands the caller a ~360-byte PLY that loads cleanly and simulates
    // nothing, and callers (the GUI included) key on the exit code.
    if (generator.faceCount() == 0 || generator.vertexCount() == 0) {
        std::cerr << "\nERROR: the mesh is empty ("
                  << generator.vertexCount() << " vertices, "
                  << generator.faceCount() << " faces).\n"
                     "       Refusing to write: " << output_path << "\n"
                     "       Common causes:\n"
                     "         - --roi-center is not on this raster (check the "
                     "sign of the latitude)\n"
                     "         - --roi-radius is smaller than the pixel size\n"
                     "         - --max-edge is below the point spacing, so every "
                     "triangle was filtered\n"
                     "         - the sampled region is entirely NoData\n"
                     "       See the 'Skipped ... points' and 'Filtered ... "
                     "triangles' counts above.\n";
        return 2;
    }

    // Export to PLY
    if (!generator.exportToPLY(output_path)) {
        return 1;
    }

    std::cout << "Success!\n";
    return 0;
}