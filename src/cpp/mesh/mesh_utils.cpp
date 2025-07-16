#include "mesh_utils.h"
#include <iostream>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <fstream>
#include <sstream>
#include <iomanip>
#include <limits>
#include "../constants/constants.h"
#include "../constants/numeric_constants.h"  // Direct include for RAD_TO_DEG
#include "../thermal/thermal_model.h"

namespace constants = ptm_thermal::constants;

namespace {
    // Local fallback constants in case the includes don't work
    const double PI = 3.14159265358979323846;
    const double RAD_TO_DEG_LOCAL = 180.0 / PI;

    // Add this function to determine system endianness
    bool is_little_endian() {
        const uint16_t value = 0x0001;
        return *reinterpret_cast<const uint8_t*>(&value) == 0x01;
    }
    
    // PLY property type mapping
    enum class PLYPropertyType {
        CHAR, UCHAR, SHORT, USHORT, INT, UINT, FLOAT, DOUBLE, UNKNOWN
    };
    
    struct PLYProperty {
        std::string name;
        PLYPropertyType type;
        
        // For list properties
        bool is_list;
        PLYPropertyType count_type;
        PLYPropertyType item_type;
    };
    
    struct PLYElement {
        std::string name;
        int count;
        std::vector<PLYProperty> properties;
    };
    
    // Convert string type to PLYPropertyType
    PLYPropertyType getPLYPropertyType(const std::string& type_str) {
        if (type_str == "char") return PLYPropertyType::CHAR;
        if (type_str == "uchar") return PLYPropertyType::UCHAR;
        if (type_str == "short") return PLYPropertyType::SHORT;
        if (type_str == "ushort") return PLYPropertyType::USHORT;
        if (type_str == "int") return PLYPropertyType::INT;
        if (type_str == "uint") return PLYPropertyType::UINT;
        if (type_str == "float") return PLYPropertyType::FLOAT;
        if (type_str == "double") return PLYPropertyType::DOUBLE;
        return PLYPropertyType::UNKNOWN;
    }
    
    // Get size of PLY property type
    size_t getPLYPropertyTypeSize(PLYPropertyType type) {
        switch (type) {
            case PLYPropertyType::CHAR: return sizeof(char);
            case PLYPropertyType::UCHAR: return sizeof(unsigned char);
            case PLYPropertyType::SHORT: return sizeof(short);
            case PLYPropertyType::USHORT: return sizeof(unsigned short);
            case PLYPropertyType::INT: return sizeof(int);
            case PLYPropertyType::UINT: return sizeof(unsigned int);
            case PLYPropertyType::FLOAT: return sizeof(float);
            case PLYPropertyType::DOUBLE: return sizeof(double);
            default: return 0;
        }
    }
    
    // Read a value of the specified PLY type
    double readPLYPropertyValue(std::ifstream& file, PLYPropertyType type, bool is_little_endian_file) {
        switch (type) {
            case PLYPropertyType::CHAR: {
                char val;
                file.read(reinterpret_cast<char*>(&val), sizeof(char));
                return static_cast<double>(val);
            }
            case PLYPropertyType::UCHAR: {
                unsigned char val;
                file.read(reinterpret_cast<char*>(&val), sizeof(unsigned char));
                return static_cast<double>(val);
            }
            case PLYPropertyType::SHORT: {
                int16_t val;
                file.read(reinterpret_cast<char*>(&val), sizeof(int16_t));
                if (is_little_endian() != is_little_endian_file) {
                    // Swap bytes if necessary
                    val = (val >> 8) | (val << 8);
                }
                return static_cast<double>(val);
            }
            case PLYPropertyType::USHORT: {
                uint16_t val;
                file.read(reinterpret_cast<char*>(&val), sizeof(uint16_t));
                if (is_little_endian() != is_little_endian_file) {
                    // Swap bytes if necessary
                    val = (val >> 8) | (val << 8);
                }
                return static_cast<double>(val);
            }
            case PLYPropertyType::INT: {
                int32_t val;
                file.read(reinterpret_cast<char*>(&val), sizeof(int32_t));
                if (is_little_endian() != is_little_endian_file) {
                    // Swap bytes if necessary
                    val = ((val & 0xFF) << 24) | ((val & 0xFF00) << 8) | 
                           ((val & 0xFF0000) >> 8) | ((val >> 24) & 0xFF);
                }
                return static_cast<double>(val);
            }
            case PLYPropertyType::UINT: {
                uint32_t val;
                file.read(reinterpret_cast<char*>(&val), sizeof(uint32_t));
                if (is_little_endian() != is_little_endian_file) {
                    // Swap bytes if necessary
                    val = ((val & 0xFF) << 24) | ((val & 0xFF00) << 8) | 
                           ((val & 0xFF0000) >> 8) | ((val >> 24) & 0xFF);
                }
                return static_cast<double>(val);
            }
            case PLYPropertyType::FLOAT: {
                float val;
                file.read(reinterpret_cast<char*>(&val), sizeof(float));
                if (is_little_endian() != is_little_endian_file) {
                    // Swap bytes if necessary
                    uint32_t* val_bits = reinterpret_cast<uint32_t*>(&val);
                    *val_bits = ((*val_bits & 0xFF) << 24) | ((*val_bits & 0xFF00) << 8) | 
                               ((*val_bits & 0xFF0000) >> 8) | ((*val_bits >> 24) & 0xFF);
                }
                return static_cast<double>(val);
            }
            case PLYPropertyType::DOUBLE: {
                double val;
                file.read(reinterpret_cast<char*>(&val), sizeof(double));
                if (is_little_endian() != is_little_endian_file) {
                    // Swap bytes if necessary
                    uint64_t* val_bits = reinterpret_cast<uint64_t*>(&val);
                    *val_bits = ((*val_bits & 0xFF) << 56) | ((*val_bits & 0xFF00) << 40) | 
                               ((*val_bits & 0xFF0000) << 24) | ((*val_bits & 0xFF000000) << 8) |
                               ((*val_bits & 0xFF00000000) >> 8) | ((*val_bits & 0xFF0000000000) >> 24) |
                               ((*val_bits & 0xFF000000000000) >> 40) | ((*val_bits & 0xFF00000000000000) >> 56);
                }
                return val;
            }
            default:
                std::cerr << "Unknown PLY property type!" << std::endl;
                return 0.0;
        }
    }
    
    // Read a PLY list property count value
    int readPLYListCount(std::ifstream& file, PLYPropertyType count_type, bool is_little_endian_file) {
        switch (count_type) {
            case PLYPropertyType::CHAR: return static_cast<int>(readPLYPropertyValue(file, PLYPropertyType::CHAR, is_little_endian_file));
            case PLYPropertyType::UCHAR: return static_cast<int>(readPLYPropertyValue(file, PLYPropertyType::UCHAR, is_little_endian_file));
            case PLYPropertyType::SHORT: return static_cast<int>(readPLYPropertyValue(file, PLYPropertyType::SHORT, is_little_endian_file));
            case PLYPropertyType::USHORT: return static_cast<int>(readPLYPropertyValue(file, PLYPropertyType::USHORT, is_little_endian_file));
            case PLYPropertyType::INT: return static_cast<int>(readPLYPropertyValue(file, PLYPropertyType::INT, is_little_endian_file));
            case PLYPropertyType::UINT: return static_cast<int>(readPLYPropertyValue(file, PLYPropertyType::UINT, is_little_endian_file));
            default:
                std::cerr << "Invalid count type for PLY list property!" << std::endl;
                return 0;
        }
    }
}

namespace ptm {

MeshUtils::MeshUtils() {
    // Initialize with default values
    resolution_ = 3;
    mesh_ = std::make_shared<Mesh>();
}

void MeshUtils::setMeshResolution(int resolution) {
    resolution_ = resolution;
    setupMesh();
}

std::shared_ptr<Mesh> MeshUtils::getMesh() const {
    return mesh_;
}

void MeshUtils::setupMesh() {
    // Create an icosphere mesh for the body
    createIcosphereMesh(resolution_, *mesh_);
    
    // Calculate face centers and normals
    calculateFaceProperties();
    
    // Map faces to lat/lon coordinates
    mapFacesToLatLon();
    
    std::cout << "Mesh created with " << mesh_->vertices.size() << " vertices and " 
              << mesh_->faces.size() << " faces" << std::endl;
}

void MeshUtils::createIcosphereMesh(int resolution, Mesh& mesh) {
    // Start with icosahedron vertices
    const double t = (1.0 + std::sqrt(5.0)) / 2.0;
    const double norm = std::sqrt(1.0 + t * t);
    
    // Create the 12 vertices of the icosahedron
    std::vector<std::array<double, 3>> vertices = {
        {-1.0/norm, t/norm, 0},
        {1.0/norm, t/norm, 0},
        {-1.0/norm, -t/norm, 0},
        {1.0/norm, -t/norm, 0},
        {0, -1.0/norm, t/norm},
        {0, 1.0/norm, t/norm},
        {0, -1.0/norm, -t/norm},
        {0, 1.0/norm, -t/norm},
        {t/norm, 0, -1.0/norm},
        {t/norm, 0, 1.0/norm},
        {-t/norm, 0, -1.0/norm},
        {-t/norm, 0, 1.0/norm}
    };
    
    // Define the 20 triangular faces of the icosahedron
    std::vector<std::array<int, 3>> faces = {
        {0, 11, 5}, {0, 5, 1}, {0, 1, 7}, {0, 7, 10}, {0, 10, 11},
        {1, 5, 9}, {5, 11, 4}, {11, 10, 2}, {10, 7, 6}, {7, 1, 8},
        {3, 9, 4}, {3, 4, 2}, {3, 2, 6}, {3, 6, 8}, {3, 8, 9},
        {4, 9, 5}, {2, 4, 11}, {6, 2, 10}, {8, 6, 7}, {9, 8, 1}
    };
    
    // Refine mesh by subdivision
    for (int i = 0; i < resolution; i++) {
        std::vector<std::array<int, 3>> new_faces;
        std::map<std::pair<int, int>, int> midpoint_cache;
        
        for (const auto& face : faces) {
            int a = face[0];
            int b = face[1];
            int c = face[2];
            
            // Get midpoints
            int ab = getMidpoint(a, b, vertices, midpoint_cache);
            int bc = getMidpoint(b, c, vertices, midpoint_cache);
            int ca = getMidpoint(c, a, vertices, midpoint_cache);
            
            // Create 4 new triangles
            new_faces.push_back({a, ab, ca});
            new_faces.push_back({b, bc, ab});
            new_faces.push_back({c, ca, bc});
            new_faces.push_back({ab, bc, ca});
        }
        
        faces = new_faces;
    }
    
    // Scale vertices to Body radius
    const double radius = constants::BODY_RADIUS;
    for (auto& v : vertices) {
        double scale = radius / std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        v[0] *= scale;
        v[1] *= scale;
        v[2] *= scale;
    }
    
    // Store the results in the mesh
    mesh.vertices = std::move(vertices);
    mesh.faces = std::move(faces);
}

void MeshUtils::normalizeVector(std::array<double, 3>& vec) {
    double length = std::sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);
    if (length > 0) {
        vec[0] /= length;
        vec[1] /= length;
        vec[2] /= length;
    }
}

int MeshUtils::getMidpoint(int a, int b, 
                          std::vector<std::array<double, 3>>& vertices, 
                          std::map<std::pair<int, int>, int>& cache) {
    // Ensure consistent order
    if (b < a) std::swap(a, b);
    
    // Check if we've already calculated this midpoint
    auto key = std::make_pair(a, b);
    auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
    }
    
    // Calculate midpoint
    std::array<double, 3> midpoint = {
        (vertices[a][0] + vertices[b][0]) * 0.5,
        (vertices[a][1] + vertices[b][1]) * 0.5,
        (vertices[a][2] + vertices[b][2]) * 0.5
    };
    
    // Normalize to unit sphere
    normalizeVector(midpoint);
    
    // Add new vertex
    int idx = vertices.size();
    vertices.push_back(midpoint);
    cache[key] = idx;
    
    return idx;
}

void MeshUtils::calculateFaceProperties() {
    mesh_->face_normals.resize(mesh_->faces.size());
    mesh_->face_centers.resize(mesh_->faces.size());
    
    #pragma omp parallel for
    for (size_t i = 0; i < mesh_->faces.size(); i++) {
        const auto& face = mesh_->faces[i];
        const auto& v1 = mesh_->vertices[face[0]];
        const auto& v2 = mesh_->vertices[face[1]];
        const auto& v3 = mesh_->vertices[face[2]];
        
        // Calculate face center
        mesh_->face_centers[i] = {{
            (v1[0] + v2[0] + v3[0]) / 3.0,
            (v1[1] + v2[1] + v3[1]) / 3.0,
            (v1[2] + v2[2] + v3[2]) / 3.0
        }};
        
        // Calculate face normal
        std::array<double, 3> edge1 = {{v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2]}};
        std::array<double, 3> edge2 = {{v3[0] - v1[0], v3[1] - v1[1], v3[2] - v1[2]}};
        
        // Cross product
        std::array<double, 3> normal = {{
            edge1[1] * edge2[2] - edge1[2] * edge2[1],
            edge1[2] * edge2[0] - edge1[0] * edge2[2],
            edge1[0] * edge2[1] - edge1[1] * edge2[0]
        }};
        
        // Normalize
        normalizeVector(normal);
        
        mesh_->face_normals[i] = normal;
    }
}

void MeshUtils::mapFacesToLatLon() {
    mesh_->latitudes.resize(mesh_->faces.size());
    mesh_->longitudes.resize(mesh_->faces.size());
    
    for (size_t i = 0; i < mesh_->faces.size(); i++) {
        const auto& center = mesh_->face_centers[i];
        double r = std::sqrt(center[0] * center[0] + center[1] * center[1] + center[2] * center[2]);
        
        // Simpler implementation
        double lat = std::asin(center[2] / r) * constants::RAD_TO_DEG;
        double lon = std::atan2(center[1], center[0]) * constants::RAD_TO_DEG;
        
        mesh_->latitudes[i] = lat;
        mesh_->longitudes[i] = lon;
    }
}

bool MeshUtils::exportMeshToVTK(const std::string& filename) const {
    std::cout << "Exporting mesh to VTK: " << filename << std::endl;
    
    try {
        std::ofstream vtk(filename);
        if (!vtk.is_open()) {
            std::cerr << "Failed to open file: " << filename << std::endl;
            return false;
        }
        
        // VTK file header
        vtk << "# vtk DataFile Version 3.0\n";
        vtk << "Body Mesh\n";
        vtk << "ASCII\n";
        vtk << "DATASET UNSTRUCTURED_GRID\n";
        
        // Write points
        vtk << "POINTS " << mesh_->vertices.size() << " float\n";
        for (const auto& v : mesh_->vertices) {
            vtk << v[0] << " " << v[1] << " " << v[2] << "\n";
        }
        
        // Write cells (faces)
        vtk << "CELLS " << mesh_->faces.size() << " " << mesh_->faces.size() * 4 << "\n";
        for (const auto& f : mesh_->faces) {
            vtk << "3 " << f[0] << " " << f[1] << " " << f[2] << "\n";
        }
        
        // Cell types (all triangles)
        vtk << "CELL_TYPES " << mesh_->faces.size() << "\n";
        for (size_t i = 0; i < mesh_->faces.size(); i++) {
            vtk << "5\n"; // VTK_TRIANGLE
        }
        
        // Cell data
        vtk << "CELL_DATA " << mesh_->faces.size() << "\n";
        
        // Latitude and longitude
        vtk << "SCALARS latitude float 1\n";
        vtk << "LOOKUP_TABLE default\n";
        for (const auto& lat : mesh_->latitudes) {
            vtk << lat << "\n";
        }
        
        vtk << "SCALARS longitude float 1\n";
        vtk << "LOOKUP_TABLE default\n";
        for (const auto& lon : mesh_->longitudes) {
            vtk << lon << "\n";
        }
        
        vtk.close();
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Error exporting to VTK: " << e.what() << std::endl;
        return false;
    }
}

bool MeshUtils::importTriangulatedMesh(const std::string& filename, double scale_factor) {
    std::cout << "Importing pre-triangulated mesh from: " << filename << std::endl;
    
    // Load the mesh from file
    bool success = loadMeshFromFile(filename);
    
    if (!success) {
        std::cerr << "Failed to load mesh from file: " << filename << std::endl;
        return false;
    }
    
    // Apply scale factor if needed
    if (scale_factor != 1.0) {
        std::cout << "Applying scale factor: " << scale_factor << std::endl;
        for (auto& vertex : mesh_->vertices) {
            vertex[0] *= scale_factor;
            vertex[1] *= scale_factor;
            vertex[2] *= scale_factor;
        }
    }
    
    // Calculate face properties and map to lat/lon coordinates
    calculateFaceProperties();
    mapFacesToLatLon();
    
    std::cout << "Successfully imported mesh with " << mesh_->vertices.size() << " vertices and " 
              << mesh_->faces.size() << " faces" << std::endl;
    
    return true;
}

bool MeshUtils::exportIlluminationToVTK(
    const std::string& filename,
    const std::shared_ptr<Mesh>& mesh,
    const std::vector<double>& illumination_values,
    const std::vector<double>& primary_ir_values,
    const std::vector<double>& times,
    const std::vector<std::string>& time_strings,
    int time_steps) {
    
    try {
        std::ofstream vtk(filename);
        if (!vtk.is_open()) {
            std::cerr << "Error: Could not open file for writing: " << filename << std::endl;
            return false;
        }
        
        std::cerr << "Writing illumination VTK file: " << filename << std::endl;
        
        // VTK file header
        vtk << "# vtk DataFile Version 3.0\n";
        vtk << "Planetary Illumination Simulation\n";
        vtk << "ASCII\n";
        vtk << "DATASET UNSTRUCTURED_GRID\n";
        
        // Write vertices
        vtk << "POINTS " << mesh->vertices.size() << " float\n";
        for (const auto& v : mesh->vertices) {
            vtk << v[0] << " " << v[1] << " " << v[2] << "\n";
        }
        
        // Write cells (faces)
        vtk << "CELLS " << mesh->faces.size() << " " << mesh->faces.size() * 4 << "\n";
        for (const auto& f : mesh->faces) {
            vtk << "3 " << f[0] << " " << f[1] << " " << f[2] << "\n";
        }
        
        // Write cell types (all triangles)
        vtk << "CELL_TYPES " << mesh->faces.size() << "\n";
        for (size_t i = 0; i < mesh->faces.size(); i++) {
            vtk << "5\n"; // VTK_TRIANGLE
        }
        // Write FIELD data for time values
        vtk << "FIELD FieldData 2\n";
        
        // Write time values
        vtk << "TIME_STEPS " << 1 << " " << time_steps << " float\n";
        for (int t = 0; t < time_steps; t++) {
            vtk << times[t] << "\n"; 
        }
        
        // Write time strings
        vtk << "TIME_STRINGS " << 1 << " " << time_steps << " string\n";
        for (int t = 0; t < time_steps; t++) {
            vtk << time_strings[t] << "\n";
        }
        
        // Write cell data
        vtk << "CELL_DATA " << mesh->faces.size() << "\n";
        
        // Add latitude/longitude data first (same as illumination file)
        vtk << "SCALARS latitude float 1\n";
        vtk << "LOOKUP_TABLE default\n";
        for (size_t f = 0; f < mesh->faces.size(); f++) {
            vtk << mesh->latitudes[f] << "\n";
        }
        
        vtk << "SCALARS longitude float 1\n";
        vtk << "LOOKUP_TABLE default\n";
        for (size_t f = 0; f < mesh->faces.size(); f++) {
            vtk << mesh->longitudes[f] << "\n";
        }

        // Simple progress output to stderr
        std::cerr << "Writing illumination data..." << std::endl;
        for (int t = 0; t < time_steps; t++) {
            // Total illumination
            vtk << "SCALARS illumination_" << t << " float 1\n";
            vtk << "LOOKUP_TABLE default\n";
            for (size_t f = 0; f < mesh->faces.size(); f++) {
                size_t index = f * time_steps + t;
                double illum_value = (index < illumination_values.size()) ? illumination_values[index] : 0.0;
                vtk << illum_value << "\n";
            }

            // Primary IR
            vtk << "SCALARS primary_ir_" << t << " float 1\n";
            vtk << "LOOKUP_TABLE default\n";
            for (size_t f = 0; f < mesh->faces.size(); f++) {
                size_t index = f * time_steps + t;
                double ir_value = (index < primary_ir_values.size()) ? primary_ir_values[index] : 0.0;
                vtk << ir_value << "\n";
            }
            
            // Simple progress output to stderr
            if (t % 10 == 0 || t == time_steps - 1) {
                std::cerr << "\rProgress: " << (t + 1) << "/" << time_steps 
                          << " (" << (100 * (t + 1) / time_steps) << "%)" << std::flush;
            }
        }
        std::cerr << std::endl << "Illumination data written successfully." << std::endl;
        
        vtk.close();
        std::cerr << "Illumination data exported to VTK file: " << filename << std::endl;
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Error exporting to VTK: " << e.what() << std::endl;
        return false;
    }
}

bool MeshUtils::loadMeshFromFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file: " << filename << std::endl;
        return false;
    }
    
    // Clear existing mesh data
    mesh_->vertices.clear();
    mesh_->faces.clear();
    
    std::string line;
    std::string file_extension = filename.substr(filename.find_last_of(".") + 1);
    std::transform(file_extension.begin(), file_extension.end(), file_extension.begin(), ::tolower);
    
    if (file_extension == "obj") {
        std::cout << "Loading OBJ file: " << filename << std::endl;

        // Temporary vectors to store the data
        std::vector<std::array<double, 3>> vertices;
        std::vector<std::array<int, 3>> faces;

        // Read OBJ file line by line
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            std::string type;
            iss >> type;

            if (type == "v") {
                // Vertex data
                double x, y, z;
                if (!(iss >> x >> y >> z)) {
                    std::cerr << "Error: Failed to parse vertex data at line: " << line << std::endl;
                    continue;
                }
                vertices.push_back({x, y, z});
            }
            else if (type == "f") {
                // Face data - handle basic face definitions (vertices only)
                std::string v1_str, v2_str, v3_str;
                int v1, v2, v3;

                // Try to read 3 face indices (might include texture/normal indices)
                if (!(iss >> v1_str >> v2_str >> v3_str)) {
                    std::cerr << "Error: Failed to parse face data at line: " << line << std::endl;
                    continue;
                }

                // Extract vertex indices (OBJ is 1-indexed, so subtract 1)
                // Format could be: v, v/vt, v/vt/vn, or v//vn
                std::istringstream v1_iss(v1_str);
                std::istringstream v2_iss(v2_str);
                std::istringstream v3_iss(v3_str);

                std::string index_str;
                
                // Parse v1
                std::getline(v1_iss, index_str, '/');
                v1 = std::stoi(index_str) - 1;  // Convert to 0-indexed

                // Parse v2
                std::getline(v2_iss, index_str, '/');
                v2 = std::stoi(index_str) - 1;  // Convert to 0-indexed
                
                // Parse v3
                std::getline(v3_iss, index_str, '/');
                v3 = std::stoi(index_str) - 1;  // Convert to 0-indexed

                // Check if indices are valid
                if (v1 < 0 || v1 >= static_cast<int>(vertices.size()) ||
                    v2 < 0 || v2 >= static_cast<int>(vertices.size()) ||
                    v3 < 0 || v3 >= static_cast<int>(vertices.size())) {
                    std::cerr << "Error: Invalid vertex index in face: " << line << std::endl;
                    continue;
                }

                faces.push_back({v1, v2, v3});
            }
            // Ignore other OBJ elements (vt, vn, etc.)
        }

        // Check if we have valid data
        if (vertices.empty() || faces.empty()) {
            std::cerr << "Error: No valid vertices or faces found in OBJ file" << std::endl;
            return false;
        }

        // Copy data to mesh structure
        mesh_->vertices = std::move(vertices);
        mesh_->faces = std::move(faces);

        std::cout << "Successfully loaded " << mesh_->vertices.size() << " vertices and " 
                  << mesh_->faces.size() << " faces from OBJ file" << std::endl;
    }
    else if (file_extension == "ply") {
        std::cout << "Loading PLY file: " << filename << std::endl;
        
        // Variables for PLY parsing
        std::string format = "ascii";  // Default format
        bool is_binary = false;
        std::vector<PLYElement> elements;
        PLYElement* current_element = nullptr;
        
        // Read header
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            
            std::istringstream iss(line);
            std::string token;
            iss >> token;
            
            if (token == "format") {
                iss >> format;
                is_binary = (format == "binary_little_endian" || format == "binary_big_endian");
                std::cout << "PLY format: " << format << std::endl;
            }
            else if (token == "element") {
                std::string element_name;
                int count;
                iss >> element_name >> count;
                
                elements.push_back({element_name, count, {}});
                current_element = &elements.back();
                
                std::cout << "Found " << count << " " << element_name << "s" << std::endl;
            }
            else if (token == "property" && current_element) {
                std::string type_str, name;
                iss >> type_str;
                
                if (type_str == "list") {
                    // Handle list property (for faces)
                    std::string count_type_str, item_type_str;
                    iss >> count_type_str >> item_type_str >> name;
                    
                    PLYProperty prop;
                    prop.name = name;
                    prop.is_list = true;
                    prop.count_type = getPLYPropertyType(count_type_str);
                    prop.item_type = getPLYPropertyType(item_type_str);
                    
                    current_element->properties.push_back(prop);
                } else {
                    // Regular property
                    iss >> name;
                    
                    PLYProperty prop;
                    prop.name = name;
                    prop.is_list = false;
                    prop.type = getPLYPropertyType(type_str);
                    
                    current_element->properties.push_back(prop);
                }
            }
            else if (token == "end_header") {
                // Done with header
                break;
            }
        }
        
        // Find vertex and face elements
        PLYElement* vertex_element = nullptr;
        PLYElement* face_element = nullptr;
        for (auto& element : elements) {
            if (element.name == "vertex") {
                vertex_element = &element;
            } else if (element.name == "face") {
                face_element = &element;
            }
        }
        
        if (!vertex_element || !face_element) {
            std::cerr << "Error: PLY file missing vertex or face elements" << std::endl;
            return false;
        }
        
        // Find coordinate properties for vertices
        int x_index = -1, y_index = -1, z_index = -1;
        for (size_t i = 0; i < vertex_element->properties.size(); i++) {
            const auto& prop = vertex_element->properties[i];
            if (prop.name == "x") x_index = i;
            else if (prop.name == "y") y_index = i;
            else if (prop.name == "z") z_index = i;
        }
        
        if (x_index == -1 || y_index == -1 || z_index == -1) {
            std::cerr << "Error: PLY file missing x, y, or z coordinate properties" << std::endl;
            return false;
        }
        
        // Find vertex indices property for faces
        int vertex_indices_index = -1;
        for (size_t i = 0; i < face_element->properties.size(); i++) {
            const auto& prop = face_element->properties[i];
            if (prop.is_list && (prop.name == "vertex_indices" || prop.name == "vertex_index")) {
                vertex_indices_index = i;
                break;
            }
        }
        
        if (vertex_indices_index == -1) {
            std::cerr << "Error: PLY file missing vertex indices property for faces" << std::endl;
            return false;
        }
        
        // Resize mesh arrays
        mesh_->vertices.resize(vertex_element->count);
        mesh_->faces.resize(face_element->count);
        
        bool is_little_endian_file = (format == "binary_little_endian");
        
        // Read vertex data
        if (is_binary) {
            // Binary format
            for (int i = 0; i < vertex_element->count; i++) {
                double coords[3] = {0.0, 0.0, 0.0};
                
                for (size_t j = 0; j < vertex_element->properties.size(); j++) {
                    const auto& prop = vertex_element->properties[j];
                    double value = readPLYPropertyValue(file, prop.type, is_little_endian_file);
                    
                    if (j == x_index) coords[0] = value;
                    else if (j == y_index) coords[1] = value;
                    else if (j == z_index) coords[2] = value;
                }
                
                mesh_->vertices[i] = {coords[0], coords[1], coords[2]};
            }
        } else {
            // ASCII format - existing code
            for (int i = 0; i < vertex_element->count; i++) {
                if (!std::getline(file, line) || line.empty()) {
                    std::cerr << "Error: Unexpected end of file while reading vertices" << std::endl;
                    return false;
                }
                
                std::istringstream iss(line);
                std::vector<double> values(vertex_element->properties.size());
                for (size_t j = 0; j < vertex_element->properties.size(); j++) {
                    if (!(iss >> values[j])) {
                        std::cerr << "Error: Failed to parse vertex data at line: " << line << std::endl;
                        return false;
                    }
                }
                
                mesh_->vertices[i] = {values[x_index], values[y_index], values[z_index]};
            }
        }
        
        // Read face data
        int valid_faces = 0;
        if (is_binary) {
            // Binary format
            for (int i = 0; i < face_element->count; i++) {
                // For each face property
                std::vector<int> vertex_indices;
                
                for (size_t j = 0; j < face_element->properties.size(); j++) {
                    const auto& prop = face_element->properties[j];
                    
                    if (j == vertex_indices_index) {
                        // Read list property for vertex indices
                        int count = readPLYListCount(file, prop.count_type, is_little_endian_file);
                        
                        if (count != 3) {
                            std::cerr << "Warning: Non-triangular face detected (" << count 
                                      << " vertices) at face " << i << ". Only triangular meshes are supported." << std::endl;
                            
                            // Skip the indices
                            for (int k = 0; k < count; k++) {
                                readPLYPropertyValue(file, prop.item_type, is_little_endian_file);
                            }
                            continue;
                        }
                        
                        // Read the three vertex indices
                        vertex_indices.resize(count);
                        for (int k = 0; k < count; k++) {
                            vertex_indices[k] = static_cast<int>(readPLYPropertyValue(file, prop.item_type, is_little_endian_file));
                        }
                    } else {
                        // Skip other properties
                        readPLYPropertyValue(file, prop.type, is_little_endian_file);
                    }
                }
                
                if (vertex_indices.size() == 3) {
                    if (valid_faces < face_element->count) {
                        mesh_->faces[valid_faces++] = {vertex_indices[0], vertex_indices[1], vertex_indices[2]};
                    }
                }
            }
        } else {
            // ASCII format - existing code
            for (int i = 0; i < face_element->count; i++) {
                if (!std::getline(file, line) || line.empty()) {
                    std::cerr << "Error: Unexpected end of file while reading faces" << std::endl;
                    return false;
                }
                
                std::istringstream iss(line);
                int vertex_count_in_face;
                if (!(iss >> vertex_count_in_face)) {
                    std::cerr << "Error: Failed to parse face data at line: " << line << std::endl;
                    return false;
                }
                
                if (vertex_count_in_face != 3) {
                    std::cerr << "Warning: Non-triangular face detected (" << vertex_count_in_face 
                              << " vertices) at face " << i << ". Only triangular meshes are supported." << std::endl;
                    continue;
                }
                
                int v0, v1, v2;
                if (!(iss >> v0 >> v1 >> v2)) {
                    std::cerr << "Error: Failed to parse face indices at line: " << line << std::endl;
                    return false;
                }
                
                if (valid_faces < face_element->count) {
                    mesh_->faces[valid_faces++] = {v0, v1, v2};
                }
            }
        }
        
        // Resize faces array if we skipped some non-triangular faces
        if (valid_faces < face_element->count) {
            mesh_->faces.resize(valid_faces);
        }
        
        std::cout << "Successfully loaded " << mesh_->vertices.size() << " vertices and " 
                  << mesh_->faces.size() << " faces from PLY file" << std::endl;
    }
    else if (file_extension == "vtk") {
        std::cout << "Loading VTK file: " << filename << std::endl;
        
        // Variables to track VTK file structure
        bool found_points = false;
        bool found_cells = false;
        int point_count = 0;
        int cell_count = 0;
        std::string data_type;
        
        // Temporary vectors to store data
        std::vector<std::array<double, 3>> vertices;
        std::vector<std::array<int, 3>> faces;
        
        // Parse VTK file header
        for (int i = 0; i < 5; i++) {
            if (!std::getline(file, line)) {
                std::cerr << "Error: Incomplete VTK file header" << std::endl;
                return false;
            }
        }
        
        // Parse dataset format
        if (!std::getline(file, line)) {
            std::cerr << "Error: Missing dataset format" << std::endl;
            return false;
        }
        std::istringstream dataset_iss(line);
        std::string dataset_keyword, dataset_type;
        dataset_iss >> dataset_keyword >> dataset_type;
        if (dataset_keyword != "DATASET" || 
            (dataset_type != "UNSTRUCTURED_GRID" && dataset_type != "POLYDATA")) {
            std::cerr << "Error: Unsupported dataset type: " << dataset_type << std::endl;
            std::cerr << "Only UNSTRUCTURED_GRID and POLYDATA are supported" << std::endl;
            return false;
        }
        
        // Read data sections
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            std::string section;
            iss >> section;
            
            if (section == "POINTS") {
                int count;
                std::string type;
                if (!(iss >> count >> type)) {
                    std::cerr << "Error: Failed to parse POINTS header: " << line << std::endl;
                    return false;
                }
                
                point_count = count;
                found_points = true;
                vertices.resize(point_count);
                
                // Read point coordinates
                for (int i = 0; i < point_count; i++) {
                    double x, y, z;
                    if (!(file >> x >> y >> z)) {
                        std::cerr << "Error: Failed to parse point data at index " << i << std::endl;
                        return false;
                    }
                    vertices[i] = {x, y, z};
                }
                
                // Consume newline
                std::getline(file, line);
            }
            else if (section == "CELLS" || section == "POLYGONS") {
                int count, total_size;
                if (!(iss >> count >> total_size)) {
                    std::cerr << "Error: Failed to parse CELLS/POLYGONS header: " << line << std::endl;
                    return false;
                }
                
                cell_count = count;
                found_cells = true;
                faces.reserve(cell_count); // Pre-allocate for efficiency
                
                // Read cell data
                for (int i = 0; i < cell_count; i++) {
                    int vertex_count, v1, v2, v3;
                    if (!(file >> vertex_count)) {
                        std::cerr << "Error: Failed to parse cell data at index " << i << std::endl;
                        return false;
                    }
                    
                    if (vertex_count == 3) {
                        // Triangle
                        if (!(file >> v1 >> v2 >> v3)) {
                            std::cerr << "Error: Failed to parse triangle indices at cell " << i << std::endl;
                            return false;
                        }
                        faces.push_back({v1, v2, v3});
                    }
                    else {
                        // Skip non-triangular face
                        std::cerr << "Warning: Skipping non-triangular face with " << vertex_count 
                                  << " vertices at index " << i << std::endl;
                        
                        // Skip the vertex indices for this face
                        for (int j = 0; j < vertex_count; j++) {
                            int dummy;
                            if (!(file >> dummy)) {
                                std::cerr << "Error: Failed to parse index " << j 
                                          << " of non-triangular face " << i << std::endl;
                                return false;
                            }
                        }
                    }
                }
                
                // Consume newline
                std::getline(file, line);
            }
            else if (section == "CELL_TYPES") {
                // Skip cell types section, we only care about triangles (type 5)
                int count;
                if (!(iss >> count)) {
                    std::cerr << "Error: Failed to parse CELL_TYPES header: " << line << std::endl;
                    return false;
                }
                
                // Skip the cell type values
                for (int i = 0; i < count; i++) {
                    int dummy;
                    if (!(file >> dummy)) {
                        std::cerr << "Error: Failed to parse cell type at index " << i << std::endl;
                        return false;
                    }
                }
                
                // Consume newline
                std::getline(file, line);
            }
            // Skip other sections
        }
        
        // Check if we have valid data
        if (!found_points || !found_cells || vertices.empty() || faces.empty()) {
            std::cerr << "Error: Missing data in VTK file. Found points: " << found_points
                      << ", found cells: " << found_cells 
                      << ", vertex count: " << vertices.size()
                      << ", face count: " << faces.size() << std::endl;
            return false;
        }
        
        // Copy data to mesh structure
        mesh_->vertices = std::move(vertices);
        mesh_->faces = std::move(faces);
        
        std::cout << "Successfully loaded " << mesh_->vertices.size() << " vertices and " 
                  << mesh_->faces.size() << " faces from VTK file" << std::endl;
    }
    else {
        std::cerr << "Unsupported file format: " << file_extension << std::endl;
        return false;
    }
    
    if (mesh_->vertices.empty() || mesh_->faces.empty()) {
        std::cerr << "Failed to load valid mesh from file: " << filename << std::endl;
        return false;
    }
    
    std::cout << "Loaded mesh with " << mesh_->vertices.size() << " vertices and " 
              << mesh_->faces.size() << " faces from " << filename << std::endl;
    
    // Update face normals, centers, etc.
    calculateFaceProperties();
    mapFacesToLatLon();
    
    return true;
}

} // namespace ptm