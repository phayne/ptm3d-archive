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
#include <filesystem>  // std::filesystem::path etc. (used below); libstdc++
                       // does not pull this in transitively the way libc++ does
#include "../constants/constants.h"
#include "../constants/numeric_constants.h"  // Direct include for RAD_TO_DEG
#include "../thermal/thermal_model.h"

#ifdef PTM_USE_VTK_READER
// VTK includes (guarded)
#include <vtkSmartPointer.h>
#include <vtkDataSetReader.h>            // legacy .vtk
#include <vtkXMLPolyDataReader.h>        // .vtp
#include <vtkXMLUnstructuredGridReader.h>// .vtu
#include <vtkPolyData.h>
#include <vtkUnstructuredGrid.h>
#include <vtkGeometryFilter.h>
#include <vtkTriangleFilter.h>
#include <vtkCellArray.h>
#include <vtkCell.h>
#include <vtkPoints.h>
#if VTK_MAJOR_VERSION >= 9
#include <vtkCellArrayIterator.h>
#endif
#include <vtkCleanPolyData.h>

// Forward declarations so the symbol is visible where used
namespace ptm {
    static bool loadMeshFromVTKWithVTK(const std::string& filename,
                                       std::vector<std::array<double,3>>& vertices,
                                       std::vector<std::array<int,3>>& faces);
}
#endif

std::shared_ptr<ptm::Mesh> ptm::MeshUtils::importMeshFromFile(
    const std::string& filename,
    std::vector<double>& elevations) {
    
    try {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Error importing mesh from file: cannot open " << filename << std::endl;
            return nullptr;
        }
        
        auto mesh = std::make_shared<ptm::Mesh>();
        
        // Read and verify the magic header (exact 7-byte match)
        char magic[7];
        if (!file.read(reinterpret_cast<char*>(magic), 7)) {
            std::cerr << "Error importing mesh: failed to read header from " << filename << std::endl;
            return nullptr;
        }
        if (std::string(magic, 7) != std::string("PTMTOPO", 7)) {
            std::cerr << "Error importing mesh: invalid header (expected 'PTMTOPO') in " << filename << std::endl;
            return nullptr;
        }
        
        // Read the mesh resolution
        int resolution = 0;
        if (!file.read(reinterpret_cast<char*>(&resolution), sizeof(resolution))) {
            std::cerr << "Error importing mesh: failed reading resolution from " << filename << std::endl;
            return nullptr;
        }
        mesh->setProperty("resolution", std::to_string(resolution));
        
        // Read the number of vertices and faces
        size_t num_vertices = 0, num_faces = 0;
        if (!file.read(reinterpret_cast<char*>(&num_vertices), sizeof(num_vertices)) ||
            !file.read(reinterpret_cast<char*>(&num_faces), sizeof(num_faces))) {
            std::cerr << "Error importing mesh: failed reading counts from " << filename << std::endl;
            return nullptr;
        }
        
        // Read all vertices
        mesh->vertices.resize(num_vertices);
        for (auto& vertex : mesh->vertices) {
            if (!file.read(reinterpret_cast<char*>(vertex.data()), sizeof(double)*3)) {
                std::cerr << "Error importing mesh: failed reading vertex data\n";
                return nullptr;
            }
        }
        
        // Read all faces
        mesh->faces.resize(num_faces);
        for (auto& face : mesh->faces) {
            if (!file.read(reinterpret_cast<char*>(face.data()), sizeof(int)*3)) {
                std::cerr << "Error importing mesh: failed reading face data\n";
                return nullptr;
            }
        }
        
        // Read vertex elevations
        elevations.resize(num_vertices);
        if (!file.read(reinterpret_cast<char*>(elevations.data()), elevations.size() * sizeof(double))) {
            std::cerr << "Error importing mesh: failed reading elevations\n";
            return nullptr;
        }
        
        // Read face normals
        mesh->face_normals.resize(num_faces);
        for (auto& normal : mesh->face_normals) {
            if (!file.read(reinterpret_cast<char*>(normal.data()), sizeof(double)*3)) {
                std::cerr << "Error importing mesh: failed reading normals\n";
                return nullptr;
            }
        }
        
        // Read face centers
        mesh->face_centers.resize(num_faces);
        for (auto& center : mesh->face_centers) {
            if (!file.read(reinterpret_cast<char*>(center.data()), sizeof(double)*3)) {
                std::cerr << "Error importing mesh: failed reading centers\n";
                return nullptr;
            }
        }
        
        // Read latitudes and longitudes
        mesh->latitudes.resize(num_faces);
        mesh->longitudes.resize(num_faces);
        if (!file.read(reinterpret_cast<char*>(mesh->latitudes.data()), mesh->latitudes.size() * sizeof(double)) ||
            !file.read(reinterpret_cast<char*>(mesh->longitudes.data()), mesh->longitudes.size() * sizeof(double))) {
            std::cerr << "Error importing mesh: failed reading lat/lon arrays\n";
            return nullptr;
        }
        
        file.close();
        
        // Mark this as a topographic mesh
        mesh->setProperty("is_topographic", "true");

        return mesh;
    }
    catch (const std::exception& e) {
        std::cerr << "Error importing mesh from file: " << e.what() << std::endl;
        return nullptr;
    }
}

// Also add the complementary exportMeshToFile function
void ptm::MeshUtils::exportMeshToFile(
    const std::string& filename,
    const std::shared_ptr<ptm::Mesh>& mesh,
    const std::vector<double>& elevations) {
    
    try {
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file for writing: " + filename);
        }
        
        // Write a magic header to identify the file format
        const char* magic = "PTMTOPO";
        file.write(magic, 7);
        
        // Write the mesh resolution as metadata
        int resolution = std::stoi(mesh->getProperty("resolution", "0"));
        file.write(reinterpret_cast<const char*>(&resolution), sizeof(resolution));
        
        // Write the number of vertices and faces
        size_t num_vertices = mesh->vertices.size();
        size_t num_faces = mesh->faces.size();
        file.write(reinterpret_cast<const char*>(&num_vertices), sizeof(num_vertices));
        file.write(reinterpret_cast<const char*>(&num_faces), sizeof(num_faces));
        
        // Write all vertices
        for (const auto& vertex : mesh->vertices) {
            file.write(reinterpret_cast<const char*>(vertex.data()), 3 * sizeof(double));
        }
        
        // Write all faces
        for (const auto& face : mesh->faces) {
            file.write(reinterpret_cast<const char*>(face.data()), 3 * sizeof(int));
        }
        
        // Write vertex elevations
        file.write(reinterpret_cast<const char*>(elevations.data()), elevations.size() * sizeof(double));
        
        // Write face normals
        for (const auto& normal : mesh->face_normals) {
            file.write(reinterpret_cast<const char*>(normal.data()), 3 * sizeof(double));
        }
        
        // Write face centers
        for (const auto& center : mesh->face_centers) {
            file.write(reinterpret_cast<const char*>(center.data()), 3 * sizeof(double));
        }
        
        // Write latitudes and longitudes
        file.write(reinterpret_cast<const char*>(mesh->latitudes.data()), mesh->latitudes.size() * sizeof(double));
        file.write(reinterpret_cast<const char*>(mesh->longitudes.data()), mesh->longitudes.size() * sizeof(double));
        
        file.close();
    }
    catch (const std::exception& e) {
        std::cerr << "Error exporting mesh to file: " << e.what() << std::endl;
    }
}

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
    // Defensive: a prior failed loadTopographicMesh used to null mesh_; if
    // any other path leaves mesh_ unset, allocate a fresh one rather than
    // dereferencing null below.
    if (!mesh_) {
        mesh_ = std::make_shared<Mesh>();
    }
    // Create an icosphere mesh for the body
    createIcosphereMesh(resolution_, *mesh_);
    
    // Calculate face centers and normals
    // For standard mesh, we don't need auto-orientation as normals are already correct
    calculateFaceProperties(false);

    // For icosphere meshes (smooth spheres), replace geometric triangle normals
    // with radial normals (normalized face center direction). The geometric normal
    // of a triangle on a discretized sphere differs systematically from the true
    // surface normal, with pentagon-adjacent faces having a larger deviation than
    // hexagonal faces. This creates a ~0.5 K temperature bias at icosahedral
    // pentagon vertices. Using radial normals eliminates this artifact.
    std::cout << "Using radial normals for icosphere mesh (smooth sphere)" << std::endl;
    #pragma omp parallel for
    for (size_t i = 0; i < mesh_->face_centers.size(); i++) {
        const auto& c = mesh_->face_centers[i];
        double r = std::sqrt(c[0]*c[0] + c[1]*c[1] + c[2]*c[2]);
        if (r > 0) {
            mesh_->face_normals[i] = {{c[0]/r, c[1]/r, c[2]/r}};
        }
    }

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
    
    // Scale vertices to body radius (use runtime value if set, else compile-time default)
    const double radius = (body_radius_km_ > 0.0) ? body_radius_km_ : constants::BODY_RADIUS;
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

void MeshUtils::setBodyRadius(double radius_km) {
    body_radius_km_ = radius_km;
}

void MeshUtils::createEllipsoidMesh(int resolution, double a_km, double b_km, double c_km) {
    // Generate a unit icosphere then scale by direction-dependent ellipsoidal radius.
    // a_km, b_km, c_km are the semi-axes along X, Y, Z respectively.
    Mesh& mesh = *mesh_;

    // Build unit icosphere (vertices on unit sphere)
    // Temporarily set body_radius_km_ to 0 so createIcosphereMesh uses
    // the unit normalization then we'll scale ourselves.
    double saved_radius = body_radius_km_;
    body_radius_km_ = 1.0;  // Unit sphere
    createIcosphereMesh(resolution, mesh);
    body_radius_km_ = saved_radius;

    // Now vertices are on a unit sphere. Scale each vertex by the
    // direction-dependent ellipsoidal radius: (a*nx, b*ny, c*nz)
    for (auto& v : mesh.vertices) {
        double r = std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
        if (r > 0.0) {
            double nx = v[0] / r;
            double ny = v[1] / r;
            double nz = v[2] / r;
            v[0] = a_km * nx;
            v[1] = b_km * ny;
            v[2] = c_km * nz;
        }
    }

    // Recompute face properties for the ellipsoidal geometry
    calculateFaceProperties(false);
    mapFacesToLatLon();

    std::cout << "Ellipsoidal mesh created with semi-axes (" << a_km << ", " << b_km
              << ", " << c_km << ") km: " << mesh.vertices.size() << " vertices, "
              << mesh.faces.size() << " faces" << std::endl;
}

void MeshUtils::ensureOutwardNormals() {
    if (!mesh_ || mesh_->face_normals.empty() || mesh_->face_centers.empty()) {
        return;
    }
    
    std::cout << "Ensuring face normals point outward..." << std::endl;
    
    int flipped_count = 0;
    
    #pragma omp parallel for reduction(+:flipped_count)
    for (size_t i = 0; i < mesh_->face_normals.size(); i++) {
        const auto& center = mesh_->face_centers[i];
        auto& normal = mesh_->face_normals[i];
        
        // For a mesh centered at the origin, the face center vector
        // should point in the same general direction as the normal
        // (both should point outward from the center)
        double dot = center[0] * normal[0] + 
                     center[1] * normal[1] + 
                     center[2] * normal[2];
        
        // If dot product is negative, the normal points inward
        if (dot < 0) {
            // Flip the normal
            normal[0] = -normal[0];
            normal[1] = -normal[1];
            normal[2] = -normal[2];
            flipped_count++;
        }
    }
    
    if (flipped_count > 0) {
        std::cout << "Flipped " << flipped_count << " face normals to point outward ("
                  << (100.0 * flipped_count / mesh_->face_normals.size()) 
                  << "% of faces)" << std::endl;
    }
    
    // Mark that normals have been oriented
    mesh_->setProperty("normals_auto_oriented", "true");
}

// Updated loadTopographicMesh function using the robust method:
bool ptm::MeshUtils::loadTopographicMesh(const std::string& filename) {
    std::cout << "Loading topographic mesh from: " << filename << std::endl;

    // Load into a local first; only commit to mesh_ on success so that a
    // partial-read failure (e.g. truncated lat/lon arrays) does not destroy
    // the existing mesh_ and leave the object in a half-broken state. The
    // fallback path in setupMesh() then has a valid mesh_ to repopulate.
    auto loaded = importMeshFromFile(filename, vertex_elevations_);

    if (!loaded) {
        std::cerr << "Failed to load topographic mesh from: " << filename << std::endl;
        return false;
    }
    mesh_ = loaded;
    
    // Retrieve the resolution from the mesh properties
    std::string res_str = mesh_->getProperty("resolution", "0");
    try {
        resolution_ = std::stoi(res_str);
    } catch (const std::exception& e) {
        std::cerr << "Warning: Could not parse resolution from mesh properties: " << e.what() << std::endl;
        resolution_ = 0;
    }
    
    // For topographic meshes, ensure we have valid face properties.
    // The topo binary file stores pre-computed normals and centers, so
    // this only fires if they're somehow missing.
    if (mesh_->face_normals.empty() || mesh_->face_centers.empty()) {
        calculateFaceProperties(false);
    }
    ensureOutwardNormals();
    
    // Make sure we have lat/lon coordinates
    if (mesh_->latitudes.empty() || mesh_->longitudes.empty()) {
        mapFacesToLatLon();
    }
    
    std::cout << "Successfully loaded topographic mesh with " << mesh_->vertices.size() 
              << " vertices and " << mesh_->faces.size() << " faces (resolution " 
              << resolution_ << ")" << std::endl;
    
    return true;
}

bool ptm::MeshUtils::checkForTopographicMesh(const std::string& directory, int resolution, const std::string& body_name) {
    namespace fs = std::filesystem;

    auto trim = [](std::string s) {
        auto issp = [](unsigned char c){ return std::isspace(c); };
        s.erase(s.begin(), std::find_if(s.begin(), s.end(), [&](unsigned char c){ return !issp(c); }));
        s.erase(std::find_if(s.rbegin(), s.rend(), [&](unsigned char c){ return !issp(c); }).base(), s.end());
        return s;
    };
    auto expand_path = [&](std::string p)->fs::path {
        p = trim(p);
        if (p.empty()) return fs::path();
        if (p[0] == '~') {
            const char* home = std::getenv("HOME");
            if (home && std::strlen(home) > 0) p = std::string(home) + p.substr(1);
        }
        std::string out;
        out.reserve(p.size());
        for (size_t i=0;i<p.size();){
            if (p[i] == '$' && i+1 < p.size() && p[i+1] == '{') {
                size_t j = p.find('}', i+2);
                if (j != std::string::npos) {
                    std::string var = p.substr(i+2, j-(i+2));
                    const char* val = std::getenv(var.c_str());
                    if (val) out += val;
                    i = j+1;
                    continue;
                }
            }
            out.push_back(p[i++]);
        }
        return fs::path(out);
    };
    auto to_lower = [](std::string s){
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return std::tolower(c); });
        return s;
    };

    fs::path dir = expand_path(directory);
    if (dir.empty()) {
        dir = fs::path("data") / "dem";
    }
    std::error_code ec;
    if (!fs::exists(dir, ec) || !fs::is_directory(dir, ec)) {
        // project-relative fallback
        fs::path alt = fs::path("src") / "cpp" / "data" / "topo";
        if (fs::exists(alt, ec) && fs::is_directory(alt, ec)) {
            dir = alt;
        }
    }

    const std::string res_str = std::to_string(resolution);
    // Modern (2026-04) layout: <dir>/<body>/topo_<body>_ico<N>.bin
    const std::string ico_suffix = std::string("_ico") + res_str + ".bin";
    // Legacy layout (kept for back-compat): <dir>/<Body>_Topo_res<N>.bin
    const std::string suffix_exact = std::string("_Topo_res") + res_str + ".bin";

    // Normalize body name variants
    std::string bn = trim(body_name);
    std::string bn_upper = bn; std::transform(bn_upper.begin(), bn_upper.end(), bn_upper.begin(), ::toupper);
    std::string bn_lower = bn; std::transform(bn_lower.begin(), bn_lower.end(), bn_lower.begin(), ::tolower);
    std::string bn_cap = bn;
    if (!bn_cap.empty()) {
        bn_cap[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(bn_cap[0])));
        for (size_t i=1;i<bn_cap.size();++i) bn_cap[i] = static_cast<char>(std::tolower(static_cast<unsigned char>(bn_cap[i])));
    }

    // Direct candidates (fast path), in priority order:
    //   1) <dir>/<body>/topo_<body>_ico<N>.bin       (modern, body subdir)
    //   2) <dir>/topo_<body>_ico<N>.bin              (modern, flat)
    //   3) <dir>/<Body>_Topo_res<N>.bin              (legacy)
    std::vector<fs::path> direct_candidates = {
        dir / bn_lower / ("topo_" + bn_lower + ico_suffix),
        dir / ("topo_" + bn_lower + ico_suffix),
        dir / (bn + suffix_exact),
        dir / (bn_cap + suffix_exact),
        dir / (bn_upper + suffix_exact),
        dir / (bn_lower + suffix_exact)
    };

    fs::path found;
    for (const auto& cand : direct_candidates) {
        if (fs::exists(cand, ec) && fs::is_regular_file(cand, ec)) {
            found = cand;
            break;
        }
    }

    // Directory scan with relaxed matching:
    // 1) exact case-insensitive equality (Moon_Topo_res6.bin)
    // 2) contains both "<body>" and "_topo_res<res>.bin" (allow separators)
    // 3) if unique file matches "_topo_res<res>.bin", accept it as last resort
    if (found.empty() && fs::exists(dir, ec)) {
        const std::string target_eq = to_lower(bn_lower + suffix_exact);
        const std::string target_suffix = to_lower(std::string("_topo_res") + res_str + ".bin");

        fs::path candidate_contains;
        fs::path candidate_suffix_only;
        size_t contains_count = 0;
        size_t suffix_only_count = 0;

        for (const auto& entry : fs::directory_iterator(dir, ec)) {
            if (ec) break;
            if (!entry.is_regular_file(ec)) continue;
            const std::string name = entry.path().filename().string();
            const std::string lname = to_lower(name);

            // Equality
            if (lname == target_eq) {
                found = entry.path();
                break;
            }
            // Contains body + suffix (e.g., "moon-topo_res6.bin")
            if (lname.find(bn_lower) != std::string::npos &&
                lname.find(target_suffix) != std::string::npos) {
                candidate_contains = entry.path();
                contains_count++;
                continue;
            }
            // Suffix-only fallback (unique)
            if (lname.size() >= target_suffix.size() &&
                lname.rfind(target_suffix) == lname.size() - target_suffix.size()) {
                candidate_suffix_only = entry.path();
                suffix_only_count++;
            }
        }

        if (found.empty() && contains_count == 1) {
            found = candidate_contains;
        }
        if (found.empty() && contains_count == 0 && suffix_only_count == 1) {
            found = candidate_suffix_only;
        }

        // Diagnostics to aid troubleshooting
        if (found.empty()) {
            std::cout << "Topography scan: no exact match '" << (bn + suffix_exact)
                      << "'. Candidates with '" << bn << "'+" << suffix_exact << ": "
                      << contains_count << ", suffix-only: " << suffix_only_count << "\n";
        }
    }

    if (!found.empty()) {
        std::cout << "Found pre-computed topographic mesh: " << found.string() << std::endl;
        if (this->loadTopographicMesh(found.string())) {
            return true;
        } else {
            std::cerr << "Failed to load topographic mesh from: " << found.string()
                      << " — falling back to standard mesh.\n";
            return false;
        }
    }

    std::cout << "No pre-computed topographic mesh found for resolution " << resolution
              << " in " << dir.string() << ", will generate standard mesh." << std::endl;
    return false;
}

void ptm::MeshUtils::normalizeVector(std::array<double, 3>& vec) {
    double length = std::sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);
    if (length > 0) {
        vec[0] /= length;
        vec[1] /= length;
        vec[2] /= length;
    }
}

int ptm::MeshUtils::getMidpoint(int a, int b, 
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

void ptm::MeshUtils::mapFacesToLatLon() {
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

bool ptm::MeshUtils::determineMeshHemisphere() {
    if (!mesh_ || mesh_->face_centers.empty()) {
        return true; // Default to northern hemisphere if no data
    }
    
    // Calculate average latitude of all face centers
    double total_latitude = 0.0;
    int valid_points = 0;
    
    for (size_t i = 0; i < mesh_->face_centers.size(); i++) {
        const auto& center = mesh_->face_centers[i];
        double r = std::sqrt(center[0] * center[0] + center[1] * center[1] + center[2] * center[2]);
        
        // Calculate latitude in degrees
        double lat = std::asin(center[2] / r) * constants::RAD_TO_DEG;
        
        // Only count valid latitudes
        if (!std::isnan(lat) && std::abs(lat) <= 90.0) {
            total_latitude += lat;
            valid_points++;
        }
    }
    
    // Return true for north hemisphere (positive average latitude)
    // Return false for south hemisphere (negative average latitude)
    bool is_north = true;
    
    if (valid_points > 0) {
        double avg_latitude = total_latitude / valid_points;
        is_north = (avg_latitude >= 0.0);
        
        std::cout << "Mesh hemisphere detection: Average latitude is " 
                  << std::fixed << std::setprecision(2) << avg_latitude 
                  << "° (" << (is_north ? "Northern" : "Southern") << " hemisphere)" << std::endl;
    } else {
        std::cout << "Warning: Could not determine mesh hemisphere (no valid points)" << std::endl;
    }
    
    return is_north;
}

// Add a flag to handle hemisphere detection
void ptm::MeshUtils::calculateFaceProperties(bool auto_orient_normals) {

    // Check if face properties have already been calculated
    bool already_calculated = (!mesh_->face_normals.empty() && 
                            !mesh_->face_centers.empty() &&
                            mesh_->face_normals.size() == mesh_->faces.size() &&
                            mesh_->face_centers.size() == mesh_->faces.size());
    
    if (already_calculated && auto_orient_normals) {
        std::string orientation_status = mesh_->getProperty("normals_auto_oriented", "false");
        if (orientation_status == "true") {
            std::cout << "Face properties already calculated and oriented, skipping recalculation" << std::endl;
            return;
        }
    }
    if (already_calculated && !auto_orient_normals) {
        std::cout << "Face properties already calculated, skipping recalculation" << std::endl;
        return;
    }

    mesh_->face_normals.resize(mesh_->faces.size());
    mesh_->face_centers.resize(mesh_->faces.size());
    
    // Compute face centers and normals once
    #pragma omp parallel for
    for (size_t i = 0; i < mesh_->faces.size(); i++) {
        const auto& face = mesh_->faces[i];
        const auto& v1 = mesh_->vertices[face[0]];
        const auto& v2 = mesh_->vertices[face[1]];
        const auto& v3 = mesh_->vertices[face[2]];
        
        // Face center
        mesh_->face_centers[i] = {{
            (v1[0] + v2[0] + v3[0]) / 3.0,
            (v1[1] + v2[1] + v3[1]) / 3.0,
            (v1[2] + v2[2] + v3[2]) / 3.0
        }};
        
        // Face normal
        std::array<double, 3> e1 = {{v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2]}};
        std::array<double, 3> e2 = {{v3[0] - v1[0], v3[1] - v1[1], v3[2] - v1[2]}};
        std::array<double, 3> n = {{
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0]
        }};
        normalizeVector(n);
        mesh_->face_normals[i] = n;
    }
    
    // Auto-orient normals to point outward from the mesh center
    // This uses the robust dot product method: if (center · normal) < 0, flip the normal
    if (auto_orient_normals) {
        // Determine hemisphere for informational purposes only
        bool is_north_hemisphere = determineMeshHemisphere();
        std::cout << "Mesh hemisphere detection: "
                  << (is_north_hemisphere ? "Northern" : "Southern") << " hemisphere" << std::endl;

        // Use robust outward-pointing normal check for all faces
        int flipped_count = 0;
        #pragma omp parallel for reduction(+:flipped_count)
        for (size_t i = 0; i < mesh_->face_normals.size(); i++) {
            const auto& center = mesh_->face_centers[i];
            auto& normal = mesh_->face_normals[i];

            // For a mesh on a planetary surface, the face center vector (from origin)
            // should point in the same general direction as the outward-pointing normal
            double dot = center[0] * normal[0] +
                         center[1] * normal[1] +
                         center[2] * normal[2];

            // If dot product is negative, the normal points inward - flip it
            if (dot < 0) {
                normal[0] = -normal[0];
                normal[1] = -normal[1];
                normal[2] = -normal[2];
                flipped_count++;
            }
        }

        if (flipped_count > 0) {
            std::cout << "Flipped " << flipped_count << " face normals to point outward ("
                      << std::fixed << std::setprecision(1)
                      << (100.0 * flipped_count / mesh_->face_normals.size())
                      << "% of faces)" << std::endl;
        } else {
            std::cout << "All face normals already point outward" << std::endl;
        }

        mesh_->setProperty("normals_auto_oriented", "true");
    }
}

bool ptm::MeshUtils::exportMeshToVTK(const std::string& filename) const {
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
        vtk << "POINTS " << mesh_->vertices.size() << " double\n";
        vtk << std::setprecision(17);
        for (const auto& v : mesh_->vertices) {
            vtk << v[0] << " " << v[1] << " " << v[2] << "\n";
        }
        vtk << std::setprecision(6);
        
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

bool ptm::MeshUtils::importTriangulatedMesh(const std::string& filename, double scale_factor) {
    /* 
    std::cout << "Importing pre-triangulated mesh from: " << filename << std::endl;
    */
    
    // Check if this mesh has already been imported
    if (mesh_ && !mesh_->vertices.empty() && mesh_->getProperty("source_file") == filename) {
        std::cout << "Mesh from " << filename << " already imported, skipping re-import" << std::endl;
        return true;
    }

    // Load the mesh from file
    bool success = loadMeshFromFile(filename);
    
    if (!success) {
        std::cerr << "Failed to load mesh from file: " << filename << std::endl;
        return false;
    }
    
    // Apply scale factor if needed
    if (scale_factor != 1.0) {
        for (auto& vertex : mesh_->vertices) {
            vertex[0] *= scale_factor;
            vertex[1] *= scale_factor;
            vertex[2] *= scale_factor;
        }
        //std::cout << "Applied scale factor of " << scale_factor << " to mesh" << std::endl;
    }
    
    // Set a property to indicate this is a custom mesh
    mesh_->setProperty("is_custom_mesh", "true");
    
    // Set the source file property
    mesh_->setProperty("source_file", filename);
    
    // Calculate face properties only if not already done
    if (mesh_->face_normals.empty() || mesh_->face_centers.empty()) {
        calculateFaceProperties(true);
    } else {
        //std::cout << "Face properties already calculated, skipping recalculation" << std::endl;
    }
    // Map faces to lat/lon coordinates if not already done
    if (mesh_->latitudes.empty() || mesh_->longitudes.empty()) {
        mapFacesToLatLon();
    }
    
    /*
    std::cout << "Successfully imported mesh with " << mesh_->vertices.size() << " vertices and " 
              << mesh_->faces.size() << " faces" << std::endl;
    */
    
    return true;
}

bool ptm::MeshUtils::exportIlluminationToVTK(
    const std::string& filename,
    const std::shared_ptr<Mesh>& mesh,
    const std::vector<double>& illumination_values,
    const std::vector<double>& primary_reflected_values,
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
        vtk << "POINTS " << mesh->vertices.size() << " double\n";
        vtk << std::setprecision(17);
        for (const auto& v : mesh->vertices) {
            vtk << v[0] << " " << v[1] << " " << v[2] << "\n";
        }
        vtk << std::setprecision(6);
        
        // Write cells (faces)
        vtk << "CELLS " << mesh->faces.size() << " " << mesh->faces.size() * 4 << "\n";
        for (const auto& f : mesh->faces) {
            vtk << "3 " << f[0] << " " << f[1] << " " << f[2] << "\n";
        }
        
        // Write cell types
        vtk << "CELL_TYPES " << mesh->faces.size() << "\n";
        for (size_t i = 0; i < mesh->faces.size(); i++) {
            vtk << "5\n"; // VTK_TRIANGLE
        }
        
        // Write cell data
        vtk << "CELL_DATA " << mesh->faces.size() << "\n";
        
        // Write lat/lon
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
        
        // Write data for each time step
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
            
            // Primary reflected - ADD THIS
            vtk << "SCALARS primary_reflected_" << t << " float 1\n";
            vtk << "LOOKUP_TABLE default\n";
            for (size_t f = 0; f < mesh->faces.size(); f++) {
                size_t index = f * time_steps + t;
                double reflected_value = (index < primary_reflected_values.size()) ? primary_reflected_values[index] : 0.0;
                vtk << reflected_value << "\n";
            }
            
            // Primary IR
            vtk << "SCALARS primary_ir_" << t << " float 1\n";
            vtk << "LOOKUP_TABLE default\n";
            for (size_t f = 0; f < mesh->faces.size(); f++) {
                size_t index = f * time_steps + t;
                double ir_value = (index < primary_ir_values.size()) ? primary_ir_values[index] : 0.0;
                vtk << ir_value << "\n";
            }
            
            // Progress output
            if (t % 10 == 0 || t == time_steps - 1) {
                std::cerr << "\rProgress: " << (t + 1) << "/" << time_steps 
                          << " (" << (100 * (t + 1) / time_steps) << "%)" << std::flush;
            }
        }
        std::cerr << std::endl << "Illumination data written successfully." << std::endl;
        
        vtk.close();
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error writing illumination VTK file: " << e.what() << std::endl;
        return false;
    }
}

bool ptm::MeshUtils::loadMeshFromFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);  // OPEN IN BINARY (works for ASCII too)
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

#ifdef PTM_USE_VTK_READER
    // Try robust VTK reader first (replicates PyVista)
    if (file_extension == "vtk" || file_extension == "vtu" || file_extension == "vtp") {
        std::cout << "Loading VTK file (VTK reader): " << filename << std::endl;
        std::vector<std::array<double,3>> vtx;
        std::vector<std::array<int,3>> fcs;
        if (loadMeshFromVTKWithVTK(filename, vtx, fcs)) {
            mesh_->vertices = std::move(vtx);
            mesh_->faces    = std::move(fcs);
            if (mesh_->vertices.empty() || mesh_->faces.empty()) {
                std::cerr << "VTK reader returned empty mesh for: " << filename << std::endl;
                return false;
            }
            std::cout << "Successfully loaded " << mesh_->vertices.size()
                      << " vertices and " << mesh_->faces.size()
                      << " faces via VTK reader" << std::endl;

            // Finalize (normals/centers/latlon) to match existing behavior
            calculateFaceProperties(true);
            mapFacesToLatLon();
            std::cout << "Loaded mesh with " << mesh_->vertices.size() << " vertices and "
                      << mesh_->faces.size() << " faces from " << filename << std::endl;
            return true;
        } else {
            std::cerr << "VTK reader failed for " << filename << ", falling back to legacy parser..." << std::endl;
            // fall through to legacy parser below
        }
    }
#endif

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
        /*
        std::cout << "Loading PLY file: " << filename << std::endl;
        */
        
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

        bool found_points = false, found_cells = false;
        int point_count = 0, cell_count = 0;
        std::string data_type;

        std::vector<std::array<double, 3>> vertices;
        std::vector<std::array<int, 3>> faces;

        // Detect legacy VTK mode (ASCII/BINARY) while scanning for DATASET
        file.clear();
        file.seekg(0);
        bool dataset_found = false;
        bool legacy_binary = false;

        auto trim = [](const std::string& s)->std::string {
            size_t a=0,b=s.size(); while (a<b && std::isspace((unsigned char)s[a])) ++a;
            while (b>a && std::isspace((unsigned char)s[b-1])) --b;
            return s.substr(a,b-a);
        };

        std::string line;
        while (std::getline(file, line)) {
            auto low = line; std::transform(low.begin(), low.end(), low.begin(), ::tolower);
            if (low.find("ascii") != std::string::npos) legacy_binary = false;
            if (low.find("binary") != std::string::npos) legacy_binary = true;
            if (low.rfind("dataset", 0) == 0 || low.find("dataset ") != std::string::npos) {
                dataset_found = true;
                break;
            }
        }
        if (!dataset_found) {
            std::cerr << "Error: Missing DATASET line in VTK file" << std::endl;
            return false;
        }

        auto is_le = []() {
            const uint16_t x = 1; return *reinterpret_cast<const uint8_t*>(&x) == 1;
        };
        auto bswap32 = [](uint32_t v)->uint32_t {
            return ((v & 0x000000FFu) << 24) | ((v & 0x0000FF00u) << 8) |
                   ((v & 0x00FF0000u) >> 8)  | ((v & 0xFF000000u) >> 24);
        };
        auto bswap64 = [](uint64_t v)->uint64_t {
            return ((v & 0x00000000000000FFULL) << 56) |
                   ((v & 0x000000000000FF00ULL) << 40) |
                   ((v & 0x0000000000FF0000ULL) << 24) |
                   ((v & 0x00000000FF000000ULL) << 8)  |
                   ((v & 0x000000FF00000000ULL) >> 8)  |
                   ((v & 0x0000FF0000000000ULL) >> 24) |
                   ((v & 0x00FF000000000000ULL) >> 40) |
                   ((v & 0xFF00000000000000ULL) >> 56);
        };

        // Continue parsing sections
        // Reset to start for section parsing
        file.clear();
        file.seekg(0);
        while (std::getline(file, line)) {
            std::string header = trim(line);
            if (header.empty()) continue;

            std::istringstream iss(header);
            std::string section;
            iss >> section;

            if (section == "POINTS") {
                int count; std::string type;
                if (!(iss >> count >> type)) {
                    std::cerr << "Error: Failed to parse POINTS header: " << header << std::endl;
                    return false;
                }
                point_count = count;
                found_points = true;
                vertices.resize(point_count);

                if (!legacy_binary) {
                    // ASCII POINTS
                    for (int i = 0; i < point_count; ++i) {
                        double x, y, z;
                        if (!(file >> x >> y >> z)) {
                            std::cerr << "Error: Failed to parse point data at index " << i << std::endl;
                            return false;
                        }
                        vertices[i] = {x, y, z};
                    }
                    std::getline(file, line); // consume EOL
                } else {
                    // BINARY POINTS (float or double, big-endian by spec but in practice may vary)
                    const bool host_le = is_le();
                    const size_t comps = static_cast<size_t>(point_count) * 3;
                    if (type == "float") {
                        std::vector<uint32_t> buf(comps);
                        file.read(reinterpret_cast<char*>(buf.data()), comps * sizeof(uint32_t));
                        if (file.gcount() != static_cast<std::streamsize>(comps * sizeof(uint32_t))) {
                            std::cerr << "Error: Short read in POINTS float data" << std::endl;
                            return false;
                        }
                        for (size_t i = 0; i < comps; ++i) {
                            uint32_t w = buf[i];
                            // legacy expects big-endian; swap on little-endian hosts
                            if (host_le) w = bswap32(w);
                            float fv;
                            std::memcpy(&fv, &w, sizeof(float));
                            size_t vi = i / 3, cj = i % 3;
                            if (cj == 0) vertices[vi][0] = fv;
                            else if (cj == 1) vertices[vi][1] = fv;
                            else vertices[vi][2] = fv;
                        }
                    } else if (type == "double") {
                        std::vector<uint64_t> buf(comps);
                        file.read(reinterpret_cast<char*>(buf.data()), comps * sizeof(uint64_t));
                        if (file.gcount() != static_cast<std::streamsize>(comps * sizeof(uint64_t))) {
                            std::cerr << "Error: Short read in POINTS double data" << std::endl;
                            return false;
                        }
                        for (size_t i = 0; i < comps; ++i) {
                            uint64_t w = buf[i];
                            if (host_le) w = bswap64(w);
                            double dv;
                            std::memcpy(&dv, &w, sizeof(double));
                            size_t vi = i / 3, cj = i % 3;
                            if (cj == 0) vertices[vi][0] = dv;
                            else if (cj == 1) vertices[vi][1] = dv;
                            else vertices[vi][2] = dv;
                        }
                    } else {
                        std::cerr << "Error: Unsupported POINTS type in VTK: " << type << std::endl;
                        return false;
                    }
                }
            }
            else if (section == "CELLS" || section == "POLYGONS") {
                int count, total_size;
                if (!(iss >> count >> total_size)) {
                    std::cerr << "Error: Failed to parse CELLS/POLYGONS header: " << header << std::endl;
                    return false;
                }
                cell_count = count;
                found_cells = true;
                faces.clear();
                faces.reserve(cell_count);

                // Try ASCII parsing first, regardless of header declaration
                {
                    std::streampos save_pos = file.tellg();
                    bool ascii_ok = true;
                    std::vector<std::array<int,3>> faces_ascii;
                    faces_ascii.reserve(cell_count);
                    for (int i = 0; i < cell_count; ++i) {
                        std::string cell_line;
                        if (!std::getline(file, cell_line)) { ascii_ok = false; break; }
                        cell_line = trim(cell_line);
                        if (cell_line.empty()) { ascii_ok = false; break; }
                        std::istringstream cs(cell_line);
                        int nverts = 0; cs >> nverts;
                        if (!cs || nverts < 3) { ascii_ok = false; break; }
                        if (nverts == 3) {
                            int v0, v1, v2;
                            if (!(cs >> v0 >> v1 >> v2)) { ascii_ok = false; break; }
                            faces_ascii.push_back({v0, v1, v2});
                        } else {
                            // skip rest of indices for non-tri cells
                            for (int j=0;j<nverts;++j) { int dummy; if (!(cs >> dummy)) { ascii_ok = false; break; } }
                            // triangulate simple fan if we want; for now, skip non-tri to keep behavior consistent
                            // ascii_ok = false; break;
                        }
                    }
                    if (ascii_ok && !faces_ascii.empty()) {
                        faces.swap(faces_ascii);
                        continue; // parsed successfully as ASCII
                    }
                    // restore and attempt binary parsing
                    file.clear();
                    file.seekg(save_pos);
                }

                // Binary parsing fallback: try big-endian 32-bit, then little 32-bit, then big 64-bit, then little 64-bit
                auto parse_cells_binary = [&](bool swap_bytes, bool use64)->bool {
                    std::vector<std::array<int,3>> out;
                    out.reserve(cell_count);
                    const size_t elem_size = use64 ? sizeof(uint64_t) : sizeof(uint32_t);
                    for (int i = 0; i < cell_count; ++i) {
                        // read nverts
                        uint64_t nverts64 = 0;
                        uint32_t nverts32 = 0;
                        if (use64) {
                            if (!file.read(reinterpret_cast<char*>(&nverts64), sizeof(uint64_t))) return false;
                            if (swap_bytes) nverts64 = bswap64(nverts64);
                            if (nverts64 > 1024) return false; // sanity
                        } else {
                            if (!file.read(reinterpret_cast<char*>(&nverts32), sizeof(uint32_t))) return false;
                            if (swap_bytes) nverts32 = bswap32(nverts32);
                            if (nverts32 > 1024) return false; // sanity
                        }
                        const uint64_t nverts = use64 ? nverts64 : nverts32;
                        if (nverts < 3) return false;

                        // Read indices block
                        if (nverts == 3) {
                            int v[3] = {0,0,0};
                            for (int k=0;k<3;++k) {
                                if (use64) {
                                    uint64_t idx; if (!file.read(reinterpret_cast<char*>(&idx), sizeof(uint64_t))) return false;
                                    if (swap_bytes) idx = bswap64(idx);
                                    v[k] = static_cast<int>(idx);
                                } else {
                                    uint32_t idx; if (!file.read(reinterpret_cast<char*>(&idx), sizeof(uint32_t))) return false;
                                    if (swap_bytes) idx = bswap32(idx);
                                    v[k] = static_cast<int>(idx);
                                }
                            }
                            out.push_back({v[0], v[1], v[2]});
                        } else {
                            // Skip non-tri cells quickly
                            const std::streamoff skip = static_cast<std::streamoff>(nverts * elem_size);
                            if (!file.seekg(skip, std::ios::cur)) return false;
                        }
                    }
                    if (!out.empty()) faces.swap(out);
                    return !out.empty();
                };

                std::streampos cells_data_pos = file.tellg();

                // Big-endian 32-bit (legacy spec)
                file.clear(); file.seekg(cells_data_pos);
                if (parse_cells_binary(/*swap_bytes=*/is_le(), /*use64=*/false)) continue;

                // Little-endian 32-bit
                file.clear(); file.seekg(cells_data_pos);
                if (parse_cells_binary(/*swap_bytes=*/!is_le(), /*use64=*/false)) continue;

                // Big-endian 64-bit
               
                file.clear(); file.seekg(cells_data_pos);
                if (parse_cells_binary(/*swap_bytes=*/is_le(), /*use64=*/true)) continue;

                // Little-endian 64-bit
                file.clear(); file.seekg(cells_data_pos);
                if (parse_cells_binary(/*swap_bytes=*/!is_le(), /*use64=*/true)) continue;

                std::cerr << "Error: Failed to parse CELLS/POLYGONS (robust fallback exhausted)" << std::endl;
                return false;
            }
            else if (section == "CELL_TYPES") {
                // Skip types; triangles expected
                int count;
                if (!(iss >> count)) {
                    std::cerr << "Error: Failed to parse CELL_TYPES header: " << header << std::endl;
                    return false;
                }
                if (!legacy_binary) {
                    for (int i=0;i<count;++i) {
                        int dummy;
                        if (!(file >> dummy)) { std::cerr << "Error: Failed to parse cell type at index " << i << std::endl; return false; }
                    }
                    std::getline(file, line);
                } else {
                    const bool host_le = is_le();
                    std::vector<uint32_t> types(static_cast<size_t>(count));
                    file.read(reinterpret_cast<char*>(types.data()), types.size() * sizeof(uint32_t));
                    if (file.gcount() != static_cast<std::streamsize>(types.size() * sizeof(uint32_t))) {
                        std::cerr << "Error: Short read CELL_TYPES binary" << std::endl;
                        return false;
                    }
                    if (host_le) {
                        for (auto& v : types) v = bswap32(v);
                    }
                }
            }
            // Ignore other sections
        }

        if (!found_points || !found_cells || vertices.empty() || faces.empty()) {
            std::cerr << "Error: Missing data in VTK file. Found points: " << found_points
                      << ", found cells: " << found_cells
                      << ", vertex count: " << vertices.size()
                      << ", face count: " << faces.size() << std::endl;
            return false;
        }

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
    
    /*
    std::cout << "Loaded mesh with " << mesh_->vertices.size() << " vertices and " 
              << mesh_->faces.size() << " faces from " << filename << std::endl;
    */
    
    // Update face normals, centers, etc.
    calculateFaceProperties(true);
    mapFacesToLatLon();
    
    return true;
}

std::pair<double, double> ptm::MeshUtils::latLonFromCartesian(double x, double y, double z) const {
    // Calculate radius
    double radius = std::sqrt(x*x + y*y + z*z);
    
    // Calculate latitude (declination) in radians, then convert to degrees
    double lat = std::asin(z / radius) * constants::RAD_TO_DEG;
    
    // Calculate longitude in radians, then convert to degrees
    double lon = std::atan2(y, x) * constants::RAD_TO_DEG;
    
    // Adjust longitude to [0, 360) range
    if (lon < 0) {
        lon += 360.0;
    }
    
    return std::make_pair(lat, lon);
}

double ptm::MeshUtils::getFaceArea(int face_idx) const {
    if (!mesh_ || face_idx < 0 || static_cast<size_t>(face_idx) >= mesh_->faces.size())
        return 0.0;

    // Prefer geometric area from vertices if available
    if (!mesh_->vertices.empty()) {
        const auto& f = mesh_->faces[face_idx];
        const auto& v0 = mesh_->vertices[f[0]];
        const auto& v1 = mesh_->vertices[f[1]];
        const auto& v2 = mesh_->vertices[f[2]];

        const std::array<double,3> e1{ v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2] };
        const std::array<double,3> e2{ v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2] };
        const std::array<double,3> cr{
            e1[1]*e2[2] - e1[2]*e2[1],
            e1[2]*e2[0] - e1[0]*e2[2],
            e1[0]*e2[1] - e1[1]*e2[0]
        };
        const double mag = std::sqrt(cr[0]*cr[0] + cr[1]*cr[1] + cr[2]*cr[2]);
        return 0.5 * mag;
    }

    // Fallback: estimate from spherical cap if only centers are known (very rough)
    // Assume roughly equal-area tessellation on a sphere of unit radius
    if (!mesh_->face_centers.empty()) {
        const double total_area = 4.0 * M_PI;
        return total_area / static_cast<double>(mesh_->face_centers.size());
    }

    return 0.0;
}

bool ptm::MeshUtils::importDEMData(const std::string& filename, bool /*createMesh*/, double /*radius*/) {
    // Minimal stub to satisfy linkage; full DEM->mesh path is handled by dedicated tools.
    // Viewshed workflows should load triangulated meshes (PLY/OBJ/VTK) instead.
    std::cerr << "MeshUtils::importDEMData not implemented in this build for: " << filename
              << " (use a triangulated mesh file: .ply/.obj/.vtu)\n";
    return false;
}
} // namespace ptm

#ifdef PTM_USE_VTK_READER
namespace ptm {

// Convert vtkPolyData to local mesh buffers (triangulated, cleaned)
static bool vtkToMesh(vtkPolyData* poly,
                      std::vector<std::array<double,3>>& outVerts,
                      std::vector<std::array<int,3>>& outFaces)
{
    if (!poly) return false;

    // Clean and normalize polydata (similar to PyVista preprocessing)
    vtkSmartPointer<vtkCleanPolyData> clean = vtkSmartPointer<vtkCleanPolyData>::New();
    clean->SetInputData(poly);
    clean->ConvertLinesToPointsOff();
    clean->ConvertPolysToLinesOff();
    clean->ConvertStripsToPolysOn();
    clean->Update();
    vtkPolyData* pdata = clean->GetOutput();
    if (!pdata) return false;

    // Triangulate polygons
    vtkSmartPointer<vtkTriangleFilter> tri = vtkSmartPointer<vtkTriangleFilter>::New();
    tri->SetInputData(pdata);
    tri->Update();
    vtkPolyData* tpoly = tri->GetOutput();
    if (!tpoly) return false;

    vtkPoints* pts = tpoly->GetPoints();
    if (!pts || pts->GetNumberOfPoints() <= 0) return false;

    const vtkIdType npts = pts->GetNumberOfPoints();
    outVerts.resize(static_cast<size_t>(npts));
    for (vtkIdType i = 0; i < npts; ++i) {
        double p[3]; pts->GetPoint(i, p);
        outVerts[static_cast<size_t>(i)] = { p[0], p[1], p[2] };
    }

    vtkCellArray* polys = tpoly->GetPolys();
    if (!polys) return false;

    outFaces.clear();
#if VTK_MAJOR_VERSION >= 9
    vtkSmartPointer<vtkCellArrayIterator> it = polys->NewIterator();
    for (it->GoToFirstCell(); !it->IsDoneWithTraversal(); it->GoToNextCell()) {
        vtkIdType nCellPts;
        const vtkIdType* cellPts = nullptr;
        it->GetCurrentCell(nCellPts, cellPts);
        if (nCellPts == 3) {
            outFaces.push_back({ static_cast<int>(cellPts[0]),
                                 static_cast<int>(cellPts[1]),
                                 static_cast<int>(cellPts[2]) });
        }
    }
#else
    vtkIdType nCellPts;
    const vtkIdType* cellPts = nullptr;
    polys->InitTraversal();
    while (polys->GetNextCell(nCellPts, cellPts)) {
        if (nCellPts == 3) {
            outFaces.push_back({ static_cast<int>(cellPts[0]),
                                 static_cast<int>(cellPts[1]),
                                 static_cast<int>(cellPts[2]) });
        }
    }
#endif
    return !outFaces.empty();
}

// Read .vtk/.vtu/.vtp via VTK (mirrors PyVista path)
static bool loadMeshFromVTKWithVTK(const std::string& filename,
                                   std::vector<std::array<double,3>>& vertices,
                                   std::vector<std::array<int,3>>& faces)
{
    // Decide reader by extension
    std::string ext;
    if (auto p = filename.find_last_of('.'); p != std::string::npos) {
        ext = filename.substr(p + 1);
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    }

    vtkSmartPointer<vtkPolyData> poly;
    if (ext == "vtp") {
        vtkSmartPointer<vtkXMLPolyDataReader> r = vtkSmartPointer<vtkXMLPolyDataReader>::New();
        r->SetFileName(filename.c_str());
        if (!r->CanReadFile(filename.c_str())) return false;
        r->Update();
        poly = r->GetOutput();
    } else if (ext == "vtu") {
        vtkSmartPointer<vtkXMLUnstructuredGridReader> r = vtkSmartPointer<vtkXMLUnstructuredGridReader>::New();
        r->SetFileName(filename.c_str());
        if (!r->CanReadFile(filename.c_str())) return false;
        r->Update();
        vtkSmartPointer<vtkGeometryFilter> gf = vtkSmartPointer<vtkGeometryFilter>::New();
        gf->SetInputData(r->GetOutput());
        gf->Update();
        poly = gf->GetOutput();
    } else { // legacy .vtk
        vtkSmartPointer<vtkDataSetReader> r = vtkSmartPointer<vtkDataSetReader>::New();
        r->SetFileName(filename.c_str());
        r->Update();

        vtkPolyData* p = r->GetPolyDataOutput();
        if (!p) {
            vtkUnstructuredGrid* ug = r->GetUnstructuredGridOutput();
            if (!ug) return false;
            vtkSmartPointer<vtkGeometryFilter> gf = vtkSmartPointer<vtkGeometryFilter>::New();
            gf->SetInputData(ug);
            gf->Update();
            poly = gf->GetOutput();
        } else {
            poly = p;
        }
    }

    if (!poly || poly->GetNumberOfPoints() == 0) return false;
    vertices.clear();
    faces.clear();
    return vtkToMesh(poly, vertices, faces);
}

} // namespace ptm
#endif