#ifndef MESH_UTILS_H
#define MESH_UTILS_H

#include <vector>
#include <array>
#include <map>
#include <memory>
#include <string>
#include "../constants/constants.h"

namespace ptm {

// Mesh structure definition
struct Mesh {
    std::vector<std::array<double, 3>> vertices;
    std::vector<std::array<int, 3>> faces;
    std::vector<std::array<double, 3>> face_normals;
    std::vector<std::array<double, 3>> face_centers;
    std::vector<double> latitudes;
    std::vector<double> longitudes;
    std::vector<double> elevations; // Added to store elevation data

    // Add these new methods:
    void setProperty(const std::string& key, const std::string& value) {
        properties[key] = value;
    }

    std::string getProperty(const std::string& key, const std::string& default_value = "") const {
        auto it = properties.find(key);
        if (it != properties.end()) {
            return it->second;
        }
        return default_value;
    }

private:
    // Existing members...
    std::map<std::string, std::string> properties;  // Add this property map
};

class MeshUtils {
public:
    MeshUtils();
    
    // Mesh creation and setup methods
    void setMeshResolution(int resolution);
    std::shared_ptr<Mesh> getMesh() const;
    void setupMesh();
    
    // I/O operations
    bool exportMeshToVTK(const std::string& filename) const;
    bool loadMeshFromFile(const std::string& filename);
    
    // New function to import a pre-triangulated mesh
    bool importTriangulatedMesh(const std::string& filename, double scale_factor = 1.0);
    
    double getFaceArea(int face_idx) const;
    std::array<double, 3> getFaceNormal(int face_idx) const;
    std::pair<double, double> getFaceLatLon(int face_idx) const;

    // Check if a location is within the DEM domain
    bool isLocationInDEMDomain(double lat, double lon) const;
    
    // Get DEM bounds (min/max lat/lon)
    std::array<double, 4> getDEMBounds() const;

    bool exportIlluminationToVTK(
        const std::string& filename,
        const std::shared_ptr<Mesh>& mesh,
        const std::vector<double>& illumination_values,
        const std::vector<double>& primary_ir_values, 
        const std::vector<double>& times,
        const std::vector<std::string>& time_strings, 
        int time_steps);
        
    bool exportMultipleVTKFiles(
        const std::string& base_filename,
        const std::shared_ptr<Mesh>& mesh,
        const std::vector<double>& illumination_values,
        const std::vector<double>& times,
        const std::vector<std::string>& time_strings,
        int time_steps);

    // DEM Data Handling Functions
    bool importDEMData(
        const std::string& filename,
        bool createMesh = true,
        double radius = 0.0);
        
    bool createMeshFromDEM(
        const std::vector<double>& latitudes,
        const std::vector<double>& longitudes,
        const std::vector<double>& elevations,
        double radius = 0.0);

    // Helper functions for DEM handling
    std::array<double, 3> latLonElevToCartesian(double lat, double lon, double elev, double radius);
    std::vector<std::array<int, 3>> delaunayTriangulation(
        const std::vector<std::array<double, 2>>& points);
    bool isInCircumcircle(
        const std::array<double, 2>& point,
        const std::array<double, 2>& v1,
        const std::array<double, 2>& v2,
        const std::array<double, 2>& v3);

private:
    void createIcosphereMesh(int resolution, Mesh& mesh);
    int getMidpoint(int a, int b, 
                    std::vector<std::array<double, 3>>& vertices, 
                    std::map<std::pair<int, int>, int>& cache);
    void calculateFaceProperties(bool auto_orient_normals = true);
    void mapFacesToLatLon();
    bool determineMeshHemisphere();
    
    void normalizeVector(std::array<double, 3>& vec);

    int resolution_;
    std::shared_ptr<Mesh> mesh_;
};

} // namespace ptm

#endif // MESH_UTILS_H