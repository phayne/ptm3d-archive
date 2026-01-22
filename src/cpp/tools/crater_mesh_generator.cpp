/**
 * @file crater_mesh_generator.cpp
 * @brief Generate bowl-shaped crater meshes for testing radiation exchange
 *
 * Creates PLY mesh files of spherical section craters that can be imported
 * into ptm_thermal_sim for radiation exchange testing.
 *
 * The mesh is generated on a uniform rectangular grid for better triangle
 * quality, then the crater bowl shape is applied as a height field.
 * Face normals point INTO the crater (concave surface).
 *
 * Usage:
 *   crater_mesh_generator [options]
 *
 * Options:
 *   --diameter D        Crater diameter in meters (default: 1000)
 *   --depth-ratio R     Depth-to-diameter ratio (default: 0.2)
 *   --latitude LAT      Crater latitude in degrees (default: 85)
 *   --resolution N      Grid points per crater diameter (default: 40)
 *   --output FILE       Output PLY filename (default: crater_mesh.ply)
 *   --body-radius R     Planetary body radius in meters (default: 1737400 for Moon)
 *   --surrounding F     Surrounding terrain factor (default: 2.0)
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <cmath>
#include <string>
#include <iomanip>
#include <algorithm>

constexpr double PI = 3.14159265358979323846;

struct CraterParams {
    double diameter = 1000.0;           // meters
    double depth_to_diameter = 0.2;     // d/D ratio
    double latitude_deg = 85.0;         // degrees
    int resolution = 40;                // grid points per crater diameter
    double body_radius = 1737400.0;     // Moon radius in meters
    double surrounding_factor = 2.0;    // Total domain = factor * diameter
    std::string output_file = "crater_mesh.ply";
};

void printUsage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n\n";
    std::cout << "Generate bowl-shaped crater mesh for radiation exchange testing.\n\n";
    std::cout << "Options:\n";
    std::cout << "  --diameter D        Crater diameter in meters (default: 1000)\n";
    std::cout << "  --depth-ratio R     Depth-to-diameter ratio (default: 0.2)\n";
    std::cout << "  --latitude LAT      Crater latitude in degrees (default: 85)\n";
    std::cout << "  --resolution N      Grid points per crater diameter (default: 40)\n";
    std::cout << "  --output FILE       Output PLY filename (default: crater_mesh.ply)\n";
    std::cout << "  --body-radius R     Planetary body radius in meters (default: 1737400)\n";
    std::cout << "  --surrounding F     Domain size as multiple of diameter (default: 2.0)\n";
    std::cout << "  --help              Show this help message\n";
}

/**
 * @brief Compute crater bowl height at given (x,y) position
 *
 * Uses spherical section geometry. Returns 0 for points outside crater rim.
 * Returns negative values (depth) for points inside crater.
 */
double craterHeight(double x, double y, double r_rim, double depth, double R_sphere, double z_center) {
    double r = std::sqrt(x * x + y * y);

    if (r >= r_rim) {
        // Outside crater - flat terrain at z = 0
        return 0.0;
    }

    // Inside crater - spherical bowl
    // Sphere equation: x² + y² + (z - z_center)² = R²
    // Solving for z: z = z_center - sqrt(R² - r²)
    double r_sq = r * r;
    double R_sq = R_sphere * R_sphere;

    if (r_sq >= R_sq) {
        // Shouldn't happen if geometry is correct, but clamp anyway
        return -depth;
    }

    double z = z_center - std::sqrt(R_sq - r_sq);
    return z;
}

bool generateCraterMesh(const CraterParams& params,
                        std::vector<std::array<double, 3>>& vertices,
                        std::vector<std::array<int, 3>>& faces) {

    double d = params.depth_to_diameter * params.diameter;
    double r_rim = params.diameter / 2.0;

    // Sphere radius from geometry: R = (d² + r²) / (2d)
    double R = (d * d + r_rim * r_rim) / (2.0 * d);

    // Sphere center is above the floor at z = R - d
    double z_center = R - d;

    std::cout << "Crater geometry:\n";
    std::cout << "  Diameter: " << params.diameter << " m\n";
    std::cout << "  Depth: " << d << " m\n";
    std::cout << "  d/D ratio: " << params.depth_to_diameter << "\n";
    std::cout << "  Sphere radius R: " << R << " m\n";
    std::cout << "  Latitude: " << params.latitude_deg << " deg\n";

    vertices.clear();
    faces.clear();

    // =========================================
    // Generate rectangular grid
    // =========================================

    // Domain extends from -half_domain to +half_domain in x and y
    double half_domain = params.surrounding_factor * r_rim;

    // Grid spacing based on resolution (points per crater diameter)
    double spacing = params.diameter / params.resolution;

    // Number of grid points in each direction
    int n_points = static_cast<int>(std::ceil(2.0 * half_domain / spacing)) + 1;

    // Adjust spacing to fit domain exactly
    spacing = 2.0 * half_domain / (n_points - 1);

    std::cout << "\nGrid parameters:\n";
    std::cout << "  Domain: " << (2.0 * half_domain) << " x " << (2.0 * half_domain) << " m\n";
    std::cout << "  Grid: " << n_points << " x " << n_points << " points\n";
    std::cout << "  Spacing: " << spacing << " m\n";

    // Generate vertices on rectangular grid
    for (int j = 0; j < n_points; ++j) {
        double y = -half_domain + j * spacing;
        for (int i = 0; i < n_points; ++i) {
            double x = -half_domain + i * spacing;
            double z = craterHeight(x, y, r_rim, d, R, z_center);
            vertices.push_back({{x, y, z}});
        }
    }

    // =========================================
    // Generate triangular faces
    // =========================================
    // Create two triangles per grid cell
    // Face winding is set so normals point UPWARD (into the crater bowl)
    // For a concave surface viewed from above, this means normals point
    // toward the viewer (out of the page), which is correct for a crater
    // that should receive illumination from above.

    for (int j = 0; j < n_points - 1; ++j) {
        for (int i = 0; i < n_points - 1; ++i) {
            // Vertex indices for this grid cell
            int v00 = j * n_points + i;           // bottom-left
            int v10 = j * n_points + (i + 1);     // bottom-right
            int v01 = (j + 1) * n_points + i;     // top-left
            int v11 = (j + 1) * n_points + (i + 1); // top-right

            // Two triangles per cell
            // Winding order: counter-clockwise when viewed from above (+z)
            // This makes normals point upward (toward +z in local frame)
            // which is INTO the crater bowl (correct for concave surface)
            faces.push_back({{v00, v10, v11}});  // lower-right triangle
            faces.push_back({{v00, v11, v01}});  // upper-left triangle
        }
    }

    size_t n_crater_faces = 0;
    size_t n_flat_faces = 0;

    // Count faces inside vs outside crater
    for (const auto& f : faces) {
        // Check if any vertex is inside crater
        bool in_crater = false;
        for (int vi = 0; vi < 3; ++vi) {
            const auto& v = vertices[f[vi]];
            double r = std::sqrt(v[0] * v[0] + v[1] * v[1]);
            if (r < r_rim) {
                in_crater = true;
                break;
            }
        }
        if (in_crater) n_crater_faces++;
        else n_flat_faces++;
    }

    // =========================================
    // Apply latitude rotation (SPICE coordinate system)
    // =========================================
    // SPICE body-fixed coordinates:
    //   z-axis = rotation axis (pole)
    //   x-axis = prime meridian at equator
    //   y-axis = completes right-handed system
    //
    // Crater is generated in local frame with bowl axis along +z.
    // The local +z direction is the surface normal (pointing away from body center).
    //
    // To place crater at latitude λ on the body surface:
    // - At latitude 90° (north pole): local +z aligns with body +z, no rotation needed
    // - At latitude 0° (equator): local +z aligns with body +x (prime meridian)
    // - Rotation is around the y-axis by angle (90° - λ)

    // Rotation angle: tilt from pole by (90 - latitude) degrees
    double lat_rad = params.latitude_deg * PI / 180.0;
    double tilt_angle = (90.0 - params.latitude_deg) * PI / 180.0;
    double cos_tilt = std::cos(tilt_angle);
    double sin_tilt = std::sin(tilt_angle);

    if (std::abs(params.latitude_deg - 90.0) > 1e-6) {
        // Rotation around y-axis
        // x' = x * cos(θ) + z * sin(θ)
        // y' = y
        // z' = -x * sin(θ) + z * cos(θ)
        for (auto& v : vertices) {
            double x_new = v[0] * cos_tilt + v[2] * sin_tilt;
            double z_new = -v[0] * sin_tilt + v[2] * cos_tilt;
            v[0] = x_new;
            v[2] = z_new;
        }
    }

    // =========================================
    // Translate to body surface
    // =========================================
    // Place the crater center at the body surface at the specified latitude.
    // Position on body: (R*cos(lat), 0, R*sin(lat)) for longitude = 0
    // This ensures the mesh hemisphere detection works correctly.

    double center_x = params.body_radius * std::cos(lat_rad);
    double center_z = params.body_radius * std::sin(lat_rad);

    std::cout << "\nBody surface positioning:\n";
    std::cout << "  Body radius: " << params.body_radius << " m\n";
    std::cout << "  Center position: (" << center_x << ", 0, " << center_z << ") m\n";

    for (auto& v : vertices) {
        v[0] += center_x;
        v[2] += center_z;
    }

    std::cout << "\nMesh statistics:\n";
    std::cout << "  Total vertices: " << vertices.size() << "\n";
    std::cout << "  Total faces: " << faces.size() << "\n";
    std::cout << "  Crater faces: " << n_crater_faces << "\n";
    std::cout << "  Flat terrain faces: " << n_flat_faces << "\n";

    // Compute triangle quality statistics
    double min_aspect = 1e10, max_aspect = 0.0, sum_aspect = 0.0;
    for (const auto& f : faces) {
        const auto& v0 = vertices[f[0]];
        const auto& v1 = vertices[f[1]];
        const auto& v2 = vertices[f[2]];

        // Edge lengths
        double e0 = std::sqrt(std::pow(v1[0]-v0[0], 2) + std::pow(v1[1]-v0[1], 2) + std::pow(v1[2]-v0[2], 2));
        double e1 = std::sqrt(std::pow(v2[0]-v1[0], 2) + std::pow(v2[1]-v1[1], 2) + std::pow(v2[2]-v1[2], 2));
        double e2 = std::sqrt(std::pow(v0[0]-v2[0], 2) + std::pow(v0[1]-v2[1], 2) + std::pow(v0[2]-v2[2], 2));

        double max_edge = std::max({e0, e1, e2});
        double min_edge = std::min({e0, e1, e2});
        double aspect = max_edge / min_edge;

        min_aspect = std::min(min_aspect, aspect);
        max_aspect = std::max(max_aspect, aspect);
        sum_aspect += aspect;
    }

    std::cout << "\nTriangle quality (aspect ratio = max_edge/min_edge):\n";
    std::cout << "  Min aspect ratio: " << min_aspect << " (ideal = 1.0)\n";
    std::cout << "  Max aspect ratio: " << max_aspect << "\n";
    std::cout << "  Mean aspect ratio: " << (sum_aspect / faces.size()) << "\n";

    return true;
}

bool writePLY(const std::string& filename,
              const std::vector<std::array<double, 3>>& vertices,
              const std::vector<std::array<int, 3>>& faces) {

    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Error: Cannot open file for writing: " << filename << "\n";
        return false;
    }

    // Write PLY header
    file << "ply\n";
    file << "format ascii 1.0\n";
    file << "comment Generated by crater_mesh_generator\n";
    file << "comment Rectangular grid with spherical bowl crater\n";
    file << "comment Face normals point INTO crater (concave surface)\n";
    file << "element vertex " << vertices.size() << "\n";
    file << "property float x\n";
    file << "property float y\n";
    file << "property float z\n";
    file << "element face " << faces.size() << "\n";
    file << "property list uchar int vertex_indices\n";
    file << "end_header\n";

    // Write vertices
    file << std::fixed << std::setprecision(6);
    for (const auto& v : vertices) {
        file << v[0] << " " << v[1] << " " << v[2] << "\n";
    }

    // Write faces
    for (const auto& f : faces) {
        file << "3 " << f[0] << " " << f[1] << " " << f[2] << "\n";
    }

    file.close();
    std::cout << "\nMesh written to: " << filename << "\n";
    return true;
}

int main(int argc, char** argv) {
    CraterParams params;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "--diameter" && i + 1 < argc) {
            params.diameter = std::stod(argv[++i]);
        } else if (arg == "--depth-ratio" && i + 1 < argc) {
            params.depth_to_diameter = std::stod(argv[++i]);
        } else if (arg == "--latitude" && i + 1 < argc) {
            params.latitude_deg = std::stod(argv[++i]);
        } else if (arg == "--resolution" && i + 1 < argc) {
            params.resolution = std::stoi(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            params.output_file = argv[++i];
        } else if (arg == "--body-radius" && i + 1 < argc) {
            params.body_radius = std::stod(argv[++i]);
        } else if (arg == "--surrounding" && i + 1 < argc) {
            params.surrounding_factor = std::stod(argv[++i]);
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            printUsage(argv[0]);
            return 1;
        }
    }

    std::cout << "=== Crater Mesh Generator ===\n\n";

    std::vector<std::array<double, 3>> vertices;
    std::vector<std::array<int, 3>> faces;

    if (!generateCraterMesh(params, vertices, faces)) {
        std::cerr << "Error generating crater mesh\n";
        return 1;
    }

    if (!writePLY(params.output_file, vertices, faces)) {
        std::cerr << "Error writing PLY file\n";
        return 1;
    }

    std::cout << "\nTo use with ptm_thermal_sim:\n";
    std::cout << "  ./ptm_thermal_sim --mesh-file " << params.output_file
              << " --thermal-mesh --radiation-exchange --rad-verbose --body Moon\n";

    return 0;
}
