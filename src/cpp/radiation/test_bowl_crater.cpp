/**
 * @file test_bowl_crater.cpp
 * @brief Validation tests for radiation exchange using bowl-shaped crater geometry
 *
 * Based on Ingersoll, Svitek, & Murray (1992) "Stability of polar frosts in
 * spherical bowl-shaped craters on the Moon, Mercury, and Mars" Icarus 100, 40-47.
 *
 * The bowl-shaped crater is modeled as a spherical section (cap) with:
 * - Depth/diameter ratio (d/D) typically 0.2 for lunar simple craters
 * - The crater is placed at various latitudes to test permanent shadow conditions
 *
 * Key validation targets:
 * 1. View factor reciprocity: A_i * F_ij = A_j * F_ji
 * 2. View factor closure: sum_j F_ij + F_i,sky = 1
 * 3. Energy conservation in radiosity solver
 * 4. Comparison with analytical solutions for spherical cavities
 */

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <cmath>
#include <memory>
#include <iomanip>
#include <chrono>
#include <atomic>
#include <exception>

#include "../mesh/mesh_utils.h"
#include "radiation_config.h"
#include "view_factor.h"
#include "radiation_exchange.h"

#ifdef PTM_GPU_OPENCL
#include "../gpu/radiosity_opencl.h"
#include "../constants/stefan_boltzmann.h"  // ptm::kStefanBoltzmann (the single definition)
#endif

namespace radiation {
namespace test {

// Physical constants
constexpr double PI = 3.14159265358979323846;
constexpr double SIGMA_SB = ptm::kStefanBoltzmann;  // [W/m^2/K^4]

/**
 * @brief Running tally of the pass/fail verdicts, so main() can set the exit status.
 *
 * Every sub-test records its verdict here and main() returns nonzero if any of
 * them failed. Before this existed the program printed "FAILED" and still
 * returned 0, so a broken Ingersoll gate was invisible to CI, to rebuild.sh,
 * and to anyone running it from a shell.
 *
 * Per latitude case the recorded assertions are: the Ingersoll floor view
 * factor, reciprocity, closure, and -- where a device exists -- GPU vs CPU.
 *
 * A SKIPPED arm is counted separately and does NOT fail the run. The only arm
 * that can skip is the GPU-vs-CPU comparison on a host with no OpenCL device
 * (the GPU=ON portable CI runner is exactly that): a missing device is a fact
 * about the machine, not a radiation regression. An arm that initialized and
 * then produced a wrong answer -- or threw -- is a real failure and is recorded
 * as one.
 */
struct VerdictTally {
    int passed = 0;
    int failed = 0;
    int skipped = 0;
    // Verdicts from arms that MUST run on every host. Counted separately from
    // the total so that the optional GPU arm cannot pad it: if the completeness
    // guard in main() scored checked(), then on a machine with a device the four
    // GPU verdicts would exactly cover the loss of one mandatory assertion, and
    // the guard would stay silent for the regression it exists to catch.
    int required = 0;

    void record(bool ok) {
        if (ok) {
            ++passed;
        } else {
            ++failed;
        }
    }
    void recordRequired(bool ok) {
        ++required;
        record(ok);
    }
    void skip() { ++skipped; }
    int checked() const { return passed + failed; }
};

/**
 * @brief Generate a bowl-shaped crater mesh as a spherical section
 *
 * The crater is modeled as a spherical cap with the opening facing upward.
 * For a crater with depth d and diameter D:
 * - The sphere has radius R = (d^2 + (D/2)^2) / (2*d)
 * - The half-angle of the cap is theta = asin(D / (2*R))
 *
 * @param diameter Crater diameter [m]
 * @param depth_to_diameter Depth-to-diameter ratio (typically 0.2 for lunar craters)
 * @param n_rings Number of concentric rings for mesh subdivision
 * @param n_azimuth Number of azimuthal divisions per ring
 * @param latitude_deg Latitude of crater center [degrees]
 * @return Shared pointer to the generated mesh
 */
std::shared_ptr<ptm::Mesh> generateBowlCraterMesh(
    double diameter,
    double depth_to_diameter,
    int n_rings,
    int n_azimuth,
    double latitude_deg = 0.0)
{
    auto mesh = std::make_shared<ptm::Mesh>();

    double d = depth_to_diameter * diameter;  // Crater depth
    double r_rim = diameter / 2.0;            // Rim radius

    // Sphere radius from depth and diameter
    // R = (d^2 + r_rim^2) / (2*d)
    double R = (d * d + r_rim * r_rim) / (2.0 * d);

    // Half-angle of the spherical cap
    double theta_max = std::asin(r_rim / R);

    // Center of the sphere (below the crater floor)
    // The sphere center is at z = -(R - d) below the rim plane (z=0)
    double z_center = -(R - d);

    // Generate vertices
    // Ring 0 is the center (floor), ring n_rings is the rim
    std::vector<std::array<double, 3>> vertices;
    std::vector<std::array<int, 3>> faces;

    // Center vertex (crater floor)
    // The floor is at z = -d (below the rim which is at z=0)
    // From sphere geometry: floor is at z_center + R (top of sphere minus depth)
    // z_floor = z_center + R - offset... let's compute correctly
    // The sphere has center at z_center = -(R - d)
    // The floor (nadir of crater) is at the lowest point, where theta=0:
    // z_floor = z_center + R = -(R - d) + R = d - wait, that's positive
    // Actually for theta=0 (nadir), the point is at distance R from sphere center
    // in the -z direction (since crater opens upward)
    // z_floor = z_center + R*cos(pi) = z_center - R  <- no, we need cos(0) for nadir
    //
    // Let me reconsider: The crater is a spherical cap with opening facing +z
    // Sphere center is below the floor at z_center = -(R - d)
    // The floor (center of crater) is the point on the sphere closest to +z
    // This is at z = z_center + R = -(R-d) + R = d
    // But that's above the rim (z=0)! That's wrong.
    //
    // Actually, the crater opens UPWARD (toward sky), so the floor is the
    // LOWEST point. The sphere center should be ABOVE the floor.
    //
    // Correct geometry:
    // - Rim is at z = 0
    // - Floor is at z = -d
    // - Sphere center is at height z_center such that:
    //   - Distance from center to rim = R (rim is on sphere)
    //   - rim has coordinates (r_rim, 0)
    //   - |z_center - 0| and r_rim satisfy: r_rim² + z_center² = R²
    //   - Also, the floor at (0, -d) is on the sphere: z_center - (-d) = R
    //   - So z_center = R - d
    //
    // Check: z_center = R - d, and R = (d² + r_rim²)/(2d)
    // z_center = (d² + r_rim²)/(2d) - d = (d² + r_rim² - 2d²)/(2d) = (r_rim² - d²)/(2d)
    //
    // Distance from (z_center) to rim at (r_rim, 0):
    // sqrt(r_rim² + z_center²) should = R
    //
    // Let's verify with d=0.2*D, D=1000, so d=200, r_rim=500:
    // R = (200² + 500²)/(2*200) = (40000 + 250000)/400 = 290000/400 = 725
    // z_center = R - d = 725 - 200 = 525
    // Dist to rim: sqrt(500² + 525²) = sqrt(250000 + 275625) = sqrt(525625) = 725. Correct!
    // Dist to floor at (0, -d): z_center - (-d) = 525 + 200 = 725 = R. Correct!
    //
    // So the sphere center is at z_center = R - d (ABOVE the floor)
    // And the floor is at z = -d

    // Corrected: floor is at z = -d
    vertices.push_back({{0.0, 0.0, -d}});

    // Correct sphere center (above the floor)
    double z_center_correct = R - d;

    // Generate rings from center to rim
    // theta=0 is at the floor (nadir), theta=theta_max is at the rim
    for (int ring = 1; ring <= n_rings; ++ring) {
        double t = static_cast<double>(ring) / n_rings;
        double theta = t * theta_max;  // Polar angle from nadir

        // Point on sphere surface at angle theta from nadir
        // The nadir direction is -z (pointing down from sphere center)
        // At theta=0: point is at (0, 0, z_center - R) = (0, 0, -d) [floor]
        // At theta=theta_max: point is at rim (z=0)
        double r = R * std::sin(theta);  // Radial distance from axis
        double z = z_center_correct - R * std::cos(theta);  // z decreases as theta increases from 0

        for (int az = 0; az < n_azimuth; ++az) {
            double phi = 2.0 * PI * az / n_azimuth;
            double x = r * std::cos(phi);
            double y = r * std::sin(phi);
            vertices.push_back({{x, y, z}});
        }
    }

    // Generate faces
    // Connect center to first ring
    for (int az = 0; az < n_azimuth; ++az) {
        int next_az = (az + 1) % n_azimuth;
        faces.push_back({{0, 1 + az, 1 + next_az}});
    }

    // Connect subsequent rings
    for (int ring = 1; ring < n_rings; ++ring) {
        int ring_start = 1 + (ring - 1) * n_azimuth;
        int next_ring_start = 1 + ring * n_azimuth;

        for (int az = 0; az < n_azimuth; ++az) {
            int next_az = (az + 1) % n_azimuth;

            int v0 = ring_start + az;
            int v1 = ring_start + next_az;
            int v2 = next_ring_start + az;
            int v3 = next_ring_start + next_az;

            // Two triangles per quad
            faces.push_back({{v0, v2, v1}});
            faces.push_back({{v1, v2, v3}});
        }
    }

    // Apply latitude rotation if needed
    if (std::abs(latitude_deg) > 1e-6) {
        double lat_rad = latitude_deg * PI / 180.0;
        double cos_lat = std::cos(lat_rad);
        double sin_lat = std::sin(lat_rad);

        for (auto& v : vertices) {
            double y_new = v[1] * cos_lat - v[2] * sin_lat;
            double z_new = v[1] * sin_lat + v[2] * cos_lat;
            v[1] = y_new;
            v[2] = z_new;
        }
    }

    mesh->vertices = std::move(vertices);
    mesh->faces = std::move(faces);

    // Compute face properties
    mesh->face_centers.resize(mesh->faces.size());
    mesh->face_normals.resize(mesh->faces.size());
    mesh->latitudes.resize(mesh->faces.size());
    mesh->longitudes.resize(mesh->faces.size());

    // Sphere center for normal orientation
    // The sphere center is ABOVE the floor at z = R - d
    double z_center_sphere = R - d;
    // After rotation, we need to track the sphere center
    std::array<double, 3> sphere_center = {{0.0, 0.0, z_center_sphere}};
    if (std::abs(latitude_deg) > 1e-6) {
        double lat_rad = latitude_deg * PI / 180.0;
        double cos_lat = std::cos(lat_rad);
        double sin_lat = std::sin(lat_rad);
        double y_new = sphere_center[1] * cos_lat - sphere_center[2] * sin_lat;
        double z_new = sphere_center[1] * sin_lat + sphere_center[2] * cos_lat;
        sphere_center[1] = y_new;
        sphere_center[2] = z_new;
    }

    for (size_t i = 0; i < mesh->faces.size(); ++i) {
        const auto& f = mesh->faces[i];
        const auto& v0 = mesh->vertices[f[0]];
        const auto& v1 = mesh->vertices[f[1]];
        const auto& v2 = mesh->vertices[f[2]];

        // Center
        mesh->face_centers[i] = {{
            (v0[0] + v1[0] + v2[0]) / 3.0,
            (v0[1] + v1[1] + v2[1]) / 3.0,
            (v0[2] + v1[2] + v2[2]) / 3.0
        }};

        // Normal (cross product)
        std::array<double, 3> e1 = {{v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]}};
        std::array<double, 3> e2 = {{v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]}};
        mesh->face_normals[i] = {{
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0]
        }};

        double len = std::sqrt(
            mesh->face_normals[i][0] * mesh->face_normals[i][0] +
            mesh->face_normals[i][1] * mesh->face_normals[i][1] +
            mesh->face_normals[i][2] * mesh->face_normals[i][2]);

        if (len > 1e-10) {
            mesh->face_normals[i][0] /= len;
            mesh->face_normals[i][1] /= len;
            mesh->face_normals[i][2] /= len;
        }

        // For a bowl crater (concave inside), normals should point TOWARD the sphere center
        // This is the "inward" direction for the crater cavity.
        // When two faces "see" each other inside the bowl, the normals point toward
        // the sphere center, and the line of sight crosses the interior.
        const auto& c = mesh->face_centers[i];
        std::array<double, 3> toward_center = {{
            sphere_center[0] - c[0],
            sphere_center[1] - c[1],
            sphere_center[2] - c[2]
        }};
        double dot = mesh->face_normals[i][0] * toward_center[0] +
                     mesh->face_normals[i][1] * toward_center[1] +
                     mesh->face_normals[i][2] * toward_center[2];

        // If normal points away from sphere center, flip it to point inward
        if (dot < 0) {
            mesh->face_normals[i][0] = -mesh->face_normals[i][0];
            mesh->face_normals[i][1] = -mesh->face_normals[i][1];
            mesh->face_normals[i][2] = -mesh->face_normals[i][2];
        }

        // Latitude/longitude from center position
        double r_total = std::sqrt(c[0]*c[0] + c[1]*c[1] + c[2]*c[2]);
        mesh->latitudes[i] = std::asin(c[2] / std::max(r_total, 1e-10)) * 180.0 / PI;
        mesh->longitudes[i] = std::atan2(c[1], c[0]) * 180.0 / PI;
    }

    return mesh;
}

/**
 * @brief Compute analytical view factor for spherical cavity floor-to-wall
 *
 * For a bowl-shaped crater (spherical section), we need to carefully consider
 * the geometry. The crater has:
 * - A circular opening (rim) at z = 0 with radius r_rim = D/2
 * - A floor at z = -d (depth)
 * - The surface follows a sphere of radius R
 *
 * For a small element dA at the floor center looking at the sky:
 * The opening subtends a solid angle, and the sky view factor is:
 * F_sky = sin²(α) where α is the half-angle to the rim from floor center
 *
 * For floor center at depth d, looking up at rim at radius r_rim:
 * tan(α) = r_rim / d
 * α = atan(r_rim / d) = atan(0.5 / d_over_D) = atan(0.5 / 0.2) = atan(2.5) ≈ 68.2°
 *
 * F_sky = sin²(α) where sin(α) = r_rim / sqrt(r_rim² + d²)
 */
double analyticalFloorToWallViewFactor(double depth_to_diameter) {
    double d_over_D = depth_to_diameter;
    double r_rim = 0.5;  // Normalized by diameter (D=1)
    double d = d_over_D;  // Also normalized by diameter

    // Half-angle from floor center to rim edge
    // sin(α) = r_rim / sqrt(r_rim² + d²)
    double dist_to_rim = std::sqrt(r_rim * r_rim + d * d);
    double sin_alpha = r_rim / dist_to_rim;

    // Sky view factor = sin²(α)
    // This is the standard formula for view factor from a differential area
    // to a coaxial disk at distance d with radius r
    double F_sky = sin_alpha * sin_alpha;

    // Wall view factor
    double F_wall = 1.0 - F_sky;

    return F_wall;
}

/**
 * @brief Compute sky view factor from floor center
 */
double analyticalFloorSkyViewFactor(double depth_to_diameter) {
    double d_over_D = depth_to_diameter;
    double r_rim = 0.5;  // Normalized by diameter
    double d = d_over_D;

    // sin(α) = r_rim / sqrt(r_rim² + d²)
    double dist_to_rim = std::sqrt(r_rim * r_rim + d * d);
    double sin_alpha = r_rim / dist_to_rim;

    // Sky view factor = sin²(α)
    return sin_alpha * sin_alpha;
}

/**
 * @brief Test view factor reciprocity
 * Checks that A_i * F_ij = A_j * F_ji for all pairs
 *
 * NOTE ON WHAT THIS DOES AND DOES NOT PROVE. This harness does not call the
 * production view-factor assembler: it fills F itself from the point-to-point
 * kernel F_ij = (A_j / pi) * cos_i * cos_j / r^2 (see runBowlCraterTest). For
 * that expression A_i * F_ij and A_j * F_ji are the same product of the same
 * factors, so reciprocity is an algebraic identity and the residual is at
 * floating-point rounding level. The check is kept because it is cheap and it
 * would still catch an indexing, transpose, or storage regression in
 * ViewFactorMatrix -- but it is NOT evidence that the production assembler is
 * reciprocal. Do not cite it as such.
 */
bool testReciprocity(const ViewFactorMatrix& F,
                     const std::vector<double>& areas,
                     double tolerance = 0.01,
                     double min_significant = 1e-5) {
    bool passed = true;
    double max_error = 0.0;
    int error_count = 0;
    int small_value_count = 0;

    for (size_t i = 0; i < F.numRows(); ++i) {
        std::vector<int> cols;
        std::vector<double> vals;
        F.getRow(static_cast<int>(i), cols, vals);

        for (size_t k = 0; k < cols.size(); ++k) {
            int j = cols[k];
            double F_ij = vals[k];
            double F_ji = F.get(j, static_cast<int>(i));

            double A_i_F_ij = areas[i] * F_ij;
            double A_j_F_ji = areas[j] * F_ji;

            // Skip very small view factors (numerical noise)
            if (F_ij < min_significant && F_ji < min_significant) {
                small_value_count++;
                continue;
            }

            double error = std::abs(A_i_F_ij - A_j_F_ji) /
                          std::max(1e-10, 0.5 * (A_i_F_ij + A_j_F_ji));

            if (error > tolerance) {
                error_count++;
                max_error = std::max(max_error, error);
                if (error_count <= 5) {
                    std::cerr << "  Reciprocity violation: faces " << i << "-" << j
                              << " (F_ij=" << F_ij << ", F_ji=" << F_ji << ")"
                              << ", error = " << (error * 100) << "%" << std::endl;
                }
            }
        }
    }

    // Recompute the unconditional max relative residual across ALL significant
    // pairs, so the validation log always carries a number for the methods paper.
    double overall_max_err = 0.0;
    double overall_sum_err = 0.0;
    size_t overall_count = 0;
    for (size_t i = 0; i < F.numRows(); ++i) {
        std::vector<int> cols2;
        std::vector<double> vals2;
        F.getRow(static_cast<int>(i), cols2, vals2);
        for (size_t k = 0; k < cols2.size(); ++k) {
            int j = cols2[k];
            double F_ij = vals2[k];
            double F_ji = F.get(j, static_cast<int>(i));
            if (F_ij < min_significant && F_ji < min_significant) continue;
            double e = std::abs(areas[i] * F_ij - areas[j] * F_ji) /
                       std::max(1e-30, 0.5 * (areas[i] * F_ij + areas[j] * F_ji));
            overall_max_err = std::max(overall_max_err, e);
            overall_sum_err += e;
            ++overall_count;
        }
    }
    double overall_mean_err = overall_count ? (overall_sum_err / overall_count) : 0.0;

    if (error_count > 0) {
        std::cerr << "  Total reciprocity violations: " << error_count
                  << " (skipped " << small_value_count << " small values)"
                  << ", max error = " << (max_error * 100) << "%" << std::endl;
        passed = false;
    } else {
        std::cout << "  Checked " << (F.numNonZeros() - small_value_count)
                  << " significant pairs (skipped " << small_value_count << " small)" << std::endl;
    }
    std::cout << "  Max relative residual (full scan): "
              << (overall_max_err * 100) << "%"
              << "   mean: " << (overall_mean_err * 100) << "%"
              << "   over " << overall_count << " significant pairs" << std::endl;

    return passed;
}

/**
 * @brief Test view factor closure (sum rule)
 * For each face: sum_j F_ij + F_i,sky = 1
 *
 * The reference passed in must be the ANALYTIC sky view factor per face. For a
 * spherical bowl that reference is exact and uniform: every interior point of a
 * sphere sees a patch with view factor A_patch / (4 pi R^2), and the aperture is
 * radiatively equivalent to the complementary cap, so
 *     F_sky = 1 - f,   f = d^2 / (d^2 + r_rim^2)
 * for every face of the cap regardless of where it sits. Any per-face spread in
 * 1 - sum_j F_ij is therefore numerical view-factor error, which is exactly what
 * this test measures. (See the Ingersoll spherical-Lambert property; the same
 * identity is what makes analyticalFloorSkyViewFactor() a whole-cap value and
 * not just a floor-center value.)
 *
 * The max/mean deviation is printed unconditionally, so the log carries the
 * number whether or not the run trips the tolerance.
 */
bool testClosure(const ViewFactorMatrix& F,
                 const std::vector<double>& expected_sky_view,
                 double tolerance = 0.05) {
    bool passed = true;
    auto row_sums = F.rowSums();

    // A short reference vector would read out of bounds below and hand back a
    // garbage verdict. Fail loudly instead: a mis-sized reference is a defect in
    // the caller, and a gate that cannot be trusted must not report PASSED.
    if (expected_sky_view.size() != row_sums.size()) {
        std::cerr << "  Closure test cannot run: reference has "
                  << expected_sky_view.size() << " entries, matrix has "
                  << row_sums.size() << " rows" << std::endl;
        return false;
    }

    double max_error = 0.0;
    double sum_error = 0.0;
    size_t worst_face = 0;
    int error_count = 0;

    for (size_t i = 0; i < row_sums.size(); ++i) {
        double F_terrain = row_sums[i];
        double F_sky_computed = 1.0 - F_terrain;
        double F_sky_expected = expected_sky_view[i];

        double error = std::abs(F_sky_computed - F_sky_expected) /
                      std::max(1e-10, F_sky_expected);

        if (error > max_error) {
            max_error = error;
            worst_face = i;
        }
        sum_error += error;

        if (error > tolerance) {
            error_count++;
            if (error_count <= 5) {
                std::cerr << "  Closure violation: face " << i
                          << ", F_sky computed = " << F_sky_computed
                          << ", expected = " << F_sky_expected
                          << ", error = " << (error * 100) << "%" << std::endl;
            }
        }
    }

    double mean_error = row_sums.empty()
                            ? 0.0
                            : (sum_error / static_cast<double>(row_sums.size()));
    std::cout << "  Max relative F_sky deviation: " << (max_error * 100) << "%"
              << " (face " << worst_face << ")"
              << "   mean: " << (mean_error * 100) << "%"
              << "   over " << row_sums.size() << " faces" << std::endl;
    std::cout << "  Tolerance: " << (tolerance * 100) << "%" << std::endl;

    if (error_count > 0) {
        std::cerr << "  Total closure violations: " << error_count
                  << " / " << row_sums.size()
                  << ", max error (all faces) = " << (max_error * 100) << "%" << std::endl;
        passed = false;
    }

    return passed;
}

/**
 * @brief Run complete validation test for bowl-shaped crater
 *
 * @param tally Receives one verdict per assertion this case evaluates.
 */
void runBowlCraterTest(double diameter,
                       double depth_to_diameter,
                       int n_rings,
                       int n_azimuth,
                       double latitude_deg,
                       VerdictTally& tally) {
    std::cout << "\n============================================================" << std::endl;
    std::cout << "Bowl Crater Test" << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << "Parameters:" << std::endl;
    std::cout << "  Diameter: " << diameter << " m" << std::endl;
    std::cout << "  Depth/Diameter: " << depth_to_diameter << std::endl;
    std::cout << "  Latitude: " << latitude_deg << " deg" << std::endl;
    std::cout << "  Mesh: " << n_rings << " rings x " << n_azimuth << " azimuth" << std::endl;

    // Generate mesh
    auto t_start = std::chrono::high_resolution_clock::now();
    auto mesh = generateBowlCraterMesh(diameter, depth_to_diameter, n_rings, n_azimuth, latitude_deg);
    auto t_mesh = std::chrono::high_resolution_clock::now();

    std::cout << "\nMesh generated:" << std::endl;
    std::cout << "  Vertices: " << mesh->vertices.size() << std::endl;
    std::cout << "  Faces: " << mesh->faces.size() << std::endl;

    // Debug: print some vertex positions to verify geometry
    double d = depth_to_diameter * diameter;
    double r_rim = diameter / 2.0;
    double R = (d * d + r_rim * r_rim) / (2.0 * d);
    std::cout << "\nGeometry verification:" << std::endl;
    std::cout << "  Depth d = " << d << " m" << std::endl;
    std::cout << "  Rim radius r_rim = " << r_rim << " m" << std::endl;
    std::cout << "  Sphere radius R = " << R << " m" << std::endl;
    std::cout << "  Sphere center z = " << (R - d) << " m" << std::endl;
    std::cout << "  Floor vertex (0): (" << mesh->vertices[0][0] << ", "
              << mesh->vertices[0][1] << ", " << mesh->vertices[0][2] << ")" << std::endl;
    if (mesh->vertices.size() > 1) {
        size_t rim_start = 1 + (n_rings - 1) * n_azimuth;
        std::cout << "  Rim vertex (" << rim_start << "): (" << mesh->vertices[rim_start][0] << ", "
                  << mesh->vertices[rim_start][1] << ", " << mesh->vertices[rim_start][2] << ")" << std::endl;
    }

    // Debug: check normal orientations
    std::cout << "\nNormal orientations (first 5 faces):" << std::endl;
    for (size_t i = 0; i < std::min(size_t(5), mesh->faces.size()); ++i) {
        std::cout << "  Face " << i << ": center=("
                  << mesh->face_centers[i][0] << ", " << mesh->face_centers[i][1] << ", " << mesh->face_centers[i][2]
                  << ") normal=(" << mesh->face_normals[i][0] << ", " << mesh->face_normals[i][1] << ", " << mesh->face_normals[i][2] << ")" << std::endl;
    }

    // Compute face areas
    std::vector<double> areas(mesh->faces.size());
    for (size_t i = 0; i < mesh->faces.size(); ++i) {
        const auto& f = mesh->faces[i];
        const auto& v0 = mesh->vertices[f[0]];
        const auto& v1 = mesh->vertices[f[1]];
        const auto& v2 = mesh->vertices[f[2]];

        std::array<double, 3> e1 = {{v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]}};
        std::array<double, 3> e2 = {{v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]}};
        std::array<double, 3> cross = {{
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0]
        }};
        areas[i] = 0.5 * std::sqrt(cross[0]*cross[0] + cross[1]*cross[1] + cross[2]*cross[2]);
    }

    double total_area = 0.0;
    for (double a : areas) total_area += a;
    std::cout << "  Total area: " << total_area << " m^2" << std::endl;

    // Analytical predictions
    double F_wall_analytical = analyticalFloorToWallViewFactor(depth_to_diameter);
    double F_sky_analytical = analyticalFloorSkyViewFactor(depth_to_diameter);
    std::cout << "\nAnalytical predictions (floor center):" << std::endl;
    std::cout << "  F_floor_to_wall = " << F_wall_analytical << std::endl;
    std::cout << "  F_floor_to_sky  = " << F_sky_analytical << std::endl;

    // Configure radiation exchange
    RadiationConfig config;
    config.enabled = true;
    config.verbose = true;
    config.max_neighbors = 1000;
    config.search_radius_factor = 20.0;  // Large enough to cover entire crater
    config.visibility_threshold = 1e-6;
    config.subdivision_level = 2;
    config.adaptive_ns_threshold = 30.0;

    // Create view factor calculator (without Embree for this simple test)
    // We'll compute view factors directly since the crater is a simple concave surface
    // where all faces can see each other

    std::cout << "\nComputing view factors..." << std::endl;

    // For this test, we compute view factors without ray tracing
    // since the crater is fully concave - all faces can see all other faces
    std::vector<int> all_rows, all_cols;
    std::vector<double> all_vals;

    auto t_vf_start = std::chrono::high_resolution_clock::now();

    // Debug counters
    std::atomic<long long> total_pairs{0};
    std::atomic<long long> cos_rejected{0};
    std::atomic<long long> threshold_rejected{0};
    std::atomic<long long> accepted{0};

    #pragma omp parallel
    {
        std::vector<int> local_rows, local_cols;
        std::vector<double> local_vals;
        long long local_total = 0, local_cos_rej = 0, local_thresh_rej = 0, local_acc = 0;

        #pragma omp for schedule(dynamic)
        for (size_t i = 0; i < mesh->faces.size(); ++i) {
            const auto& c_i = mesh->face_centers[i];
            const auto& n_i = mesh->face_normals[i];

            for (size_t j = 0; j < mesh->faces.size(); ++j) {
                if (i == j) continue;
                local_total++;

                const auto& c_j = mesh->face_centers[j];
                const auto& n_j = mesh->face_normals[j];

                // Vector from i to j
                std::array<double, 3> r_vec = {{
                    c_j[0] - c_i[0],
                    c_j[1] - c_i[1],
                    c_j[2] - c_i[2]
                }};
                double r2 = r_vec[0]*r_vec[0] + r_vec[1]*r_vec[1] + r_vec[2]*r_vec[2];
                double r_len = std::sqrt(r2);

                if (r_len < 1e-10) continue;

                // Normalized direction
                std::array<double, 3> r_hat = {{r_vec[0]/r_len, r_vec[1]/r_len, r_vec[2]/r_len}};

                // Cosines
                // cos_i: angle between n_i and direction toward j (should be positive if i "sees" j)
                // cos_j: angle between n_j and direction toward i (should be positive if j "sees" i)
                double cos_i = n_i[0]*r_hat[0] + n_i[1]*r_hat[1] + n_i[2]*r_hat[2];
                double cos_j = -(n_j[0]*r_hat[0] + n_j[1]*r_hat[1] + n_j[2]*r_hat[2]);

                // Debug: print first pair that fails
                if (i == 0 && j == 1 && latitude_deg == 0.0) {
                    #pragma omp critical
                    {
                        std::cout << "\nDebug: Face 0 -> Face 1" << std::endl;
                        std::cout << "  c_i = (" << c_i[0] << ", " << c_i[1] << ", " << c_i[2] << ")" << std::endl;
                        std::cout << "  n_i = (" << n_i[0] << ", " << n_i[1] << ", " << n_i[2] << ")" << std::endl;
                        std::cout << "  c_j = (" << c_j[0] << ", " << c_j[1] << ", " << c_j[2] << ")" << std::endl;
                        std::cout << "  n_j = (" << n_j[0] << ", " << n_j[1] << ", " << n_j[2] << ")" << std::endl;
                        std::cout << "  r_hat = (" << r_hat[0] << ", " << r_hat[1] << ", " << r_hat[2] << ")" << std::endl;
                        std::cout << "  cos_i = " << cos_i << ", cos_j = " << cos_j << std::endl;
                    }
                }

                // Both must be positive (facing each other)
                if (cos_i <= 0 || cos_j <= 0) {
                    local_cos_rej++;
                    continue;
                }

                // View factor
                double F_ij = (areas[j] / PI) * (cos_i * cos_j) / r2;

                if (F_ij > config.visibility_threshold) {
                    local_rows.push_back(static_cast<int>(i));
                    local_cols.push_back(static_cast<int>(j));
                    local_vals.push_back(F_ij);
                    local_acc++;
                } else {
                    local_thresh_rej++;
                }
            }
        }

        total_pairs += local_total;
        cos_rejected += local_cos_rej;
        threshold_rejected += local_thresh_rej;
        accepted += local_acc;

        #pragma omp critical
        {
            all_rows.insert(all_rows.end(), local_rows.begin(), local_rows.end());
            all_cols.insert(all_cols.end(), local_cols.begin(), local_cols.end());
            all_vals.insert(all_vals.end(), local_vals.begin(), local_vals.end());
        }
    }

    std::cout << "  Total pairs evaluated: " << total_pairs.load() << std::endl;
    std::cout << "  Rejected by cos check: " << cos_rejected.load()
              << " (" << (100.0 * cos_rejected.load() / total_pairs.load()) << "%)" << std::endl;
    std::cout << "  Rejected by threshold: " << threshold_rejected.load() << std::endl;
    std::cout << "  Accepted: " << accepted.load() << std::endl;

    ViewFactorMatrix F;
    F.buildFromTriplets(mesh->faces.size(), all_rows, all_cols, all_vals);

    auto t_vf_end = std::chrono::high_resolution_clock::now();
    auto vf_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_vf_end - t_vf_start).count();

    std::cout << "View factor computation: " << vf_ms << " ms" << std::endl;
    std::cout << "  Non-zeros: " << F.numNonZeros() << std::endl;
    std::cout << "  Density: " << (100.0 * F.numNonZeros() / (F.numRows() * F.numCols())) << "%" << std::endl;

    // Compute row sums (terrain view factors)
    auto row_sums = F.rowSums();

    // Find floor face (first face, at center)
    int floor_face = 0;
    double F_terrain_floor = row_sums[0];
    double F_sky_floor = 1.0 - F_terrain_floor;

    std::cout << "\nFloor face (index 0) view factors:" << std::endl;
    std::cout << "  F_terrain (computed) = " << F_terrain_floor << std::endl;
    std::cout << "  F_sky (computed)     = " << F_sky_floor << std::endl;
    std::cout << "  F_wall (analytical)  = " << F_wall_analytical << std::endl;
    std::cout << "  F_sky (analytical)   = " << F_sky_analytical << std::endl;

    double floor_error = std::abs(F_terrain_floor - F_wall_analytical) / F_wall_analytical;
    std::cout << "  Error: " << (floor_error * 100) << "%" << std::endl;

    // --- Ingersoll floor view-factor gate ------------------------------------
    // This number was computed and printed since the file was written (8178972,
    // 2026-01-22) and never compared against anything, so the headline Ingersoll
    // comparison could drift arbitrarily without the program noticing. It is now
    // a verdict.
    //
    // WHICH quantity: the relative error of the FLOOR face's terrain view factor
    // sum_j F_0j against the analytic spherical-cap value f = d^2/(d^2+r_rim^2).
    // The denominator is f (~0.138 at d/D=0.2), not F_sky (~0.862), so the same
    // absolute view-factor error reads (1-f)/f = 6.25x larger here than it does
    // in the closure test below. Both are printed so the two cannot be confused.
    //
    // WHY 1%: the residual is the flat-triangle faceting of a curved cap, a
    // known and understood discretization effect that shrinks as 1/sqrt(N) --
    // not a defect in the assembler. At this mesh (20 rings x 36 azimuth) it
    // sits near 0.49%, so 1% is roughly a factor of two of headroom: loose
    // enough that faceting alone cannot trip it, tight enough that a real
    // regression in the kernel, the areas, or the row sums will.
    const double kIngersollFloorTolerance = 0.01;
    const double floor_error_vs_sky =
        std::abs(F_sky_floor - F_sky_analytical) / F_sky_analytical;
    bool ingersoll_passed = (floor_error <= kIngersollFloorTolerance);
    std::cout << "\n--- Ingersoll Floor View Factor Test ---" << std::endl;
    std::cout << "  Quantity: relative error of face " << floor_face
              << "'s terrain view factor sum_j F_ij vs analytic f = 1 - F_sky"
              << std::endl;
    std::cout << "  Error: " << (floor_error * 100) << "%"
              << "   tolerance: " << (kIngersollFloorTolerance * 100) << "%"
              << std::endl;
    std::cout << "  Same absolute error referenced to F_sky: "
              << (floor_error_vs_sky * 100) << "%" << std::endl;
    std::cout << "  Result: " << (ingersoll_passed ? "PASSED" : "FAILED") << std::endl;
    tally.recordRequired(ingersoll_passed);

    // Test reciprocity
    std::cout << "\n--- Reciprocity Test (analytic matrix; near-tautology) ---" << std::endl;
    std::cout << "  Note: this harness fills F itself from"
                 " F_ij = (A_j/pi) cos_i cos_j / r^2, so A_i F_ij and A_j F_ji"
                 " are the same product and reciprocity holds by construction,"
                 " at rounding level. It guards ViewFactorMatrix indexing and"
                 " storage; it does not validate the production assembler."
              << std::endl;
    // Debug: check a specific pair that fails
    if (latitude_deg == 0.0) {
        int i = 36, j = 107;
        if (static_cast<size_t>(i) < mesh->faces.size() && static_cast<size_t>(j) < mesh->faces.size()) {
            double F_ij = F.get(i, j);
            double F_ji = F.get(j, i);
            std::cout << "  Debug reciprocity: F_" << i << "_" << j << " = " << F_ij
                      << ", F_" << j << "_" << i << " = " << F_ji << std::endl;
            std::cout << "  Areas: A_" << i << " = " << areas[i] << ", A_" << j << " = " << areas[j] << std::endl;
            std::cout << "  A_i*F_ij = " << (areas[i] * F_ij) << ", A_j*F_ji = " << (areas[j] * F_ji) << std::endl;
        }
    }
    bool reciprocity_passed = testReciprocity(F, areas, 0.05);
    std::cout << "  Result: " << (reciprocity_passed ? "PASSED" : "FAILED") << std::endl;
    tally.recordRequired(reciprocity_passed);

    // Expected sky view factor for every face.
    //
    // This array used to hold 0.5 * (1 + n_z), the sky view factor of a tilted
    // plane on an OPEN plain. That is the wrong geometry for a bowl: it returns
    // 1.0 at the crater floor, where the true value is F_sky = 1 - f = 0.862 at
    // d/D = 0.2, a 16% error. It was never consumed -- testClosure() was defined
    // and never called, in any commit, back to the file's first (8178972) -- so
    // nothing ever noticed. Feeding the placeholder to a live gate would have
    // failed the run for a defect in the reference, not in the code under test.
    //
    // The correct reference is exact and UNIFORM over a spherical cap: every
    // interior point of a sphere sees the aperture with the same view factor.
    // Both floor and rim faces must read analyticalFloorSkyViewFactor(d/D).
    std::vector<double> expected_sky_view(mesh->faces.size(), F_sky_analytical);

    // Statistics on sky view factors
    double min_sky = 1.0, max_sky = 0.0, avg_sky = 0.0;
    for (size_t i = 0; i < row_sums.size(); ++i) {
        double F_sky = 1.0 - row_sums[i];
        min_sky = std::min(min_sky, F_sky);
        max_sky = std::max(max_sky, F_sky);
        avg_sky += F_sky;
    }
    avg_sky /= row_sums.size();

    std::cout << "\n--- Sky View Factor Statistics ---" << std::endl;
    std::cout << "  Min: " << min_sky << std::endl;
    std::cout << "  Max: " << max_sky << std::endl;
    std::cout << "  Avg: " << avg_sky << std::endl;
    std::cout << "  Analytic (uniform over the cap): " << F_sky_analytical << std::endl;

    // --- Closure gate --------------------------------------------------------
    // sum_j F_ij + F_i,sky = 1, checked per face against the analytic uniform
    // cap value. Where the Ingersoll gate above scores ONE face (the floor),
    // this scores all of them, so it is the check that sees a defect confined to
    // the rim or to one ring.
    //
    // Same caveat as reciprocity: F here is the harness's own point-to-point
    // kernel, not the production ViewFactorCalculator. What this gate scores is
    // that kernel plus the mesh generator. It is a guard against a regression in
    // either; it is not a validation of the production assembler.
    //
    // WHY 1%: same faceting argument as the Ingersoll gate, and the same 1% line
    // Paul set for it. The peak per-face deviation at this mesh is about 0.31%
    // (min 0.859417 / max 0.863588 vs analytic 0.862069), so the gate carries
    // roughly 3x headroom over the known discretization error. Note this error
    // is referenced to F_sky, so it reads 6.25x SMALLER than the same absolute
    // view-factor error does in the Ingersoll gate above.
    //
    // The reported worst-face INDEX is not stable and is not meant to be: the
    // outer ring's 72 faces are related by the mesh's azimuthal symmetry and
    // carry the same deviation, so which one wins is decided by rounding and
    // moves with the latitude rotation and with the compiler's ISA. The VALUE
    // is what the gate scores, and it is identical across both.
    const double kClosureTolerance = 0.01;
    std::cout << "\n--- Closure Test (sum_j F_ij + F_sky = 1) ---" << std::endl;
    bool closure_passed = testClosure(F, expected_sky_view, kClosureTolerance);
    std::cout << "  Result: " << (closure_passed ? "PASSED" : "FAILED") << std::endl;
    tally.recordRequired(closure_passed);

    // Test radiosity solver
    std::cout << "\n--- Radiosity Solver Test ---" << std::endl;

    std::vector<double> emissivity(mesh->faces.size(), 0.95);
    RadiositySolver solver(F, emissivity, areas);

    RadiationConfig solver_config;
    solver_config.max_iterations = 100;
    solver_config.convergence_tolerance = 0.001;
    solver_config.use_acceleration = true;
    solver.setConfig(solver_config);

    // Test with uniform temperature
    std::vector<double> temperatures(mesh->faces.size(), 200.0);  // 200 K
    std::vector<double> solar_flux(mesh->faces.size(), 0.0);      // No solar (shadow test)

    auto t_rad_start = std::chrono::high_resolution_clock::now();
    auto indirect_flux = solver.solve(temperatures, solar_flux);
    auto t_rad_end = std::chrono::high_resolution_clock::now();
    auto rad_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_rad_end - t_rad_start).count();

    std::cout << "  Solver time: " << rad_ms << " ms" << std::endl;
    std::cout << "  Iterations: " << solver.getLastIterations() << std::endl;
    std::cout << "  Residual: " << solver.getLastResidual() << std::endl;
    std::cout << "  Converged: " << (solver.didConverge() ? "YES" : "NO") << std::endl;

    // Check energy balance
    auto balance = solver.checkEnergyBalance(temperatures, solar_flux, indirect_flux);
    std::cout << "\n--- Energy Balance ---" << std::endl;
    std::cout << "  Total emission: " << balance[0] << " W" << std::endl;
    std::cout << "  Total absorption: " << balance[1] << " W" << std::endl;
    std::cout << "  Imbalance: " << balance[2] << " W ("
              << (100.0 * balance[2] / std::max(balance[0], 1e-10)) << "%)" << std::endl;

    // Print indirect flux statistics
    double min_flux = 1e10, max_flux = 0.0, avg_flux = 0.0;
    for (double f : indirect_flux) {
        min_flux = std::min(min_flux, f);
        max_flux = std::max(max_flux, f);
        avg_flux += f;
    }
    avg_flux /= indirect_flux.size();

    std::cout << "\n--- Indirect Flux Statistics ---" << std::endl;
    std::cout << "  Min: " << min_flux << " W/m^2" << std::endl;
    std::cout << "  Max: " << max_flux << " W/m^2" << std::endl;
    std::cout << "  Avg: " << avg_flux << " W/m^2" << std::endl;

    // Expected thermal emission at 200 K
    double expected_emission = 0.95 * SIGMA_SB * std::pow(200.0, 4);
    std::cout << "  Expected emission (200 K): " << expected_emission << " W/m^2" << std::endl;

#ifdef PTM_GPU_OPENCL
    // Test GPU radiosity solver
    std::cout << "\n--- GPU Radiosity Solver Test ---" << std::endl;

    gpu::opencl::RadiosityOpenCLConfig gpu_config;
    gpu_config.max_iterations = 100;
    gpu_config.convergence_tolerance = 0.001;
    gpu_config.verbose = true;

    size_t n_faces = mesh->faces.size();
    gpu::opencl::RadiosityOpenCLSolver gpu_solver(n_faces, gpu_config);

    std::cout << "  Initializing GPU solver..." << std::flush;
    if (!gpu_solver.initialize()) {
        // Deliberately NOT the token "FAILED". A host with no OpenCL device
        // (the GPU=ON portable CI runner) lands here, and the CI log grep must
        // be able to tell that apart from a GPU arm that ran and disagreed with
        // the CPU. Print an unambiguous SKIPPED verdict instead.
        std::cout << " unavailable" << std::endl;
        std::cout << "  Note: no usable OpenCL device, or kernel compilation was rejected"
                  << " by the driver" << std::endl;
        std::cout << "  GPU vs CPU validation: SKIPPED (no OpenCL device)" << std::endl;
        tally.skip();
    } else {
        std::cout << " done" << std::endl;

        try {
            // Upload view factors
            std::cout << "  Uploading view factors..." << std::flush;
            gpu_solver.uploadViewFactors(F.values(), F.colIndices(), F.rowPointers());
            std::cout << " done" << std::endl;

            // Convert emissivity to float
            std::vector<float> emissivity_f(n_faces);
            std::vector<float> reflectivity_f(n_faces);
            for (size_t i = 0; i < n_faces; ++i) {
                emissivity_f[i] = static_cast<float>(emissivity[i]);
                reflectivity_f[i] = 1.0f - emissivity_f[i];
            }
            std::cout << "  Uploading face properties..." << std::flush;
            gpu_solver.uploadFaceProperties(emissivity_f, reflectivity_f);
            std::cout << " done" << std::endl;

            // Convert input to float
            std::vector<float> temps_f(n_faces);
            std::vector<float> solar_f(n_faces);
            for (size_t i = 0; i < n_faces; ++i) {
                temps_f[i] = static_cast<float>(temperatures[i]);
                solar_f[i] = static_cast<float>(solar_flux[i]);
            }

            // Solve on GPU
            std::cout << "  Running GPU solver..." << std::flush;
            std::vector<float> gpu_indirect_f;
            auto t_gpu_start = std::chrono::high_resolution_clock::now();
            gpu_solver.solve(temps_f, solar_f, gpu_indirect_f);
            auto t_gpu_end = std::chrono::high_resolution_clock::now();
            std::cout << " done" << std::endl;
            auto gpu_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_gpu_end - t_gpu_start).count();

        std::cout << "  GPU Solver time: " << gpu_ms << " ms" << std::endl;
        std::cout << "  GPU Iterations: " << gpu_solver.getLastIterations() << std::endl;
        std::cout << "  GPU Residual: " << gpu_solver.getLastResidual() << std::endl;
        std::cout << "  GPU Converged: " << (gpu_solver.didConverge() ? "YES" : "NO") << std::endl;

        // Compare GPU vs CPU results
        double max_diff = 0.0;
        double avg_diff = 0.0;
        for (size_t i = 0; i < n_faces; ++i) {
            double diff = std::abs(static_cast<double>(gpu_indirect_f[i]) - indirect_flux[i]);
            max_diff = std::max(max_diff, diff);
            avg_diff += diff;
        }
        avg_diff /= n_faces;

        std::cout << "\n  GPU vs CPU comparison:" << std::endl;
        std::cout << "    Max difference: " << max_diff << " W/m^2" << std::endl;
        std::cout << "    Avg difference: " << avg_diff << " W/m^2" << std::endl;
        std::cout << "    Relative error: " << (100.0 * max_diff / avg_flux) << "%" << std::endl;

        bool gpu_passed = (max_diff < avg_flux * 0.01);
        if (gpu_passed) {
            std::cout << "  GPU vs CPU validation: PASSED (< 1% relative error)" << std::endl;
        } else {
            std::cout << "  GPU vs CPU validation: FAILED (>= 1% relative error)" << std::endl;
        }
        tally.record(gpu_passed);
        } catch (const std::exception& e) {
            // The solver initialized, so a device exists: this is the GPU arm
            // running and faulting, not an absent device. It is a failure.
            std::cerr << "\n  GPU test error: " << e.what() << std::endl;
            std::cout << "  GPU vs CPU validation: FAILED (exception during the GPU arm)"
                      << std::endl;
            tally.record(false);
        }
    }
#endif

    std::cout << "\n============================================================" << std::endl;
}

} // namespace test
} // namespace radiation

int main(int argc, char** argv) {
    std::cout << "╔═══════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║  Bowl-Shaped Crater Radiation Exchange Validation Test        ║" << std::endl;
    std::cout << "║  Based on Ingersoll, Svitek, & Murray (1992)                  ║" << std::endl;
    std::cout << "╚═══════════════════════════════════════════════════════════════╝" << std::endl;

    // Test parameters
    double diameter = 1000.0;  // 1 km crater
    double depth_to_diameter = 0.2;  // Typical for lunar simple craters
    int n_rings = 20;
    int n_azimuth = 36;

    // Equator, then 70 deg, 80 deg, and 85 deg (near pole).
    const double latitudes_deg[] = {0.0, 70.0, 80.0, 85.0};
    const int n_cases =
        static_cast<int>(sizeof(latitudes_deg) / sizeof(latitudes_deg[0]));

    // Assertions every latitude case must record, regardless of hardware:
    // Ingersoll floor view factor, reciprocity, closure. These go through
    // recordRequired() and are scored below against tally.required, NOT against
    // tally.checked(): the GPU-vs-CPU arm is optional (it skips on a host with
    // no device), and if it counted toward the total then on a machine WITH a
    // device its four verdicts would exactly mask the loss of one mandatory
    // assertion and the guard would never fire.
    //
    // These 12 are 3 distinct assertions repeated 4 times, not 12 independent
    // ones. The latitude parameter rigidly rotates the mesh and the view-factor
    // matrix is rotation-invariant, so all four cases produce bit-identical
    // Ingersoll and closure numbers. Latitude varies the illumination and
    // radiosity part of the case, not these gates.
    const int n_assertions_per_case = 3;

    radiation::test::VerdictTally tally;
    for (int c = 0; c < n_cases; ++c) {
        radiation::test::runBowlCraterTest(diameter, depth_to_diameter, n_rings,
                                           n_azimuth, latitudes_deg[c], tally);
    }

    // Test with different depth/diameter ratios
    std::cout << "\n\n========== Depth/Diameter Ratio Study ==========" << std::endl;
    for (double dd : {0.1, 0.15, 0.2, 0.25, 0.3}) {
        double F_wall = radiation::test::analyticalFloorToWallViewFactor(dd);
        double F_sky = radiation::test::analyticalFloorSkyViewFactor(dd);
        std::cout << "d/D = " << dd << ": F_wall = " << F_wall
                  << ", F_sky = " << F_sky << std::endl;
    }

    std::cout << "\nTest completed." << std::endl;

    // ---- Verdict -------------------------------------------------------------
    // Printed on ONE greppable line, and reflected in the exit status. CI greps
    // for "[test_bowl_crater] PASS".
    //
    // The PASS line must not contain the substring "fail" in ANY case. Callers
    // scan this output for failure words, and one of them
    // (scripts/ptm_r2_acceptance.py, test C) upper-cases the whole stream before
    // looking -- so a summary reading "0 failed" would report a green run as a
    // failure. Say what passed and what skipped; do not name a zero count of
    // failures.
    if (tally.required < n_assertions_per_case * n_cases) {
        std::cout << "[test_bowl_crater] FAIL: only " << tally.required
                  << " mandatory verdicts were recorded, expected "
                  << (n_assertions_per_case * n_cases) << " ("
                  << n_assertions_per_case << " assertions x " << n_cases
                  << " latitudes) -- one of the Ingersoll, reciprocity, or"
                     " closure assertions is no longer running, so this program"
                     " is not testing what it claims. Verdicts of every kind: "
                  << tally.checked() << "." << std::endl;
        return 1;
    }
    if (tally.failed > 0) {
        std::cout << "[test_bowl_crater] FAIL: " << tally.failed << " of "
                  << tally.checked() << " verdicts failed (" << tally.passed
                  << " passed, " << tally.skipped << " skipped, " << n_cases
                  << " latitudes)." << std::endl;
        return 1;
    }
    std::cout << "[test_bowl_crater] PASS: " << tally.passed << " of "
              << tally.checked() << " verdicts passed (" << tally.skipped
              << " skipped, " << n_cases << " latitudes)." << std::endl;
    return 0;
}
