#include "face_selection.h"

#include <embree4/rtcore.h>
#include <embree4/rtcore_ray.h>

#include <cmath>
#include <algorithm>

namespace radiation {

std::vector<double> computeHorizonAngles(
    const std::shared_ptr<ptm::Mesh>& mesh,
    void* embree_device,
    void* embree_scene,
    int num_azimuth_samples) {

    RTCScene scene = static_cast<RTCScene>(embree_scene);
    size_t n = mesh->faces.size();
    std::vector<double> max_angles(n, 0.0);

    if (!scene) return max_angles;

    // Precompute face centers and normals
    std::vector<std::array<double, 3>> centers(n);
    std::vector<std::array<double, 3>> normals(n);

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        const auto& face = mesh->faces[i];
        const auto& v0 = mesh->vertices[face[0]];
        const auto& v1 = mesh->vertices[face[1]];
        const auto& v2 = mesh->vertices[face[2]];

        centers[i] = {{
            (v0[0] + v1[0] + v2[0]) / 3.0,
            (v0[1] + v1[1] + v2[1]) / 3.0,
            (v0[2] + v1[2] + v2[2]) / 3.0
        }};

        std::array<double, 3> e1 = {{v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]}};
        std::array<double, 3> e2 = {{v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]}};

        normals[i] = {{
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0]
        }};

        double len = std::sqrt(normals[i][0] * normals[i][0] +
                               normals[i][1] * normals[i][1] +
                               normals[i][2] * normals[i][2]);
        if (len > 1e-10) {
            normals[i][0] /= len;
            normals[i][1] /= len;
            normals[i][2] /= len;
        }

        // Ensure outward-pointing
        double dot = centers[i][0] * normals[i][0] +
                     centers[i][1] * normals[i][1] +
                     centers[i][2] * normals[i][2];
        if (dot < 0) {
            normals[i][0] = -normals[i][0];
            normals[i][1] = -normals[i][1];
            normals[i][2] = -normals[i][2];
        }
    }

    // For each face, trace rays in multiple azimuth directions at varying elevations
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < n; ++i) {
        const auto& center = centers[i];
        const auto& normal = normals[i];

        // Create local coordinate system
        // u = arbitrary perpendicular to normal
        // v = normal x u
        std::array<double, 3> u, v;

        if (std::abs(normal[0]) < 0.9) {
            u = {{1.0, 0.0, 0.0}};
        } else {
            u = {{0.0, 1.0, 0.0}};
        }

        // u = u - (u.n)n (make perpendicular)
        double un = u[0] * normal[0] + u[1] * normal[1] + u[2] * normal[2];
        u[0] -= un * normal[0];
        u[1] -= un * normal[1];
        u[2] -= un * normal[2];

        double u_len = std::sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
        u[0] /= u_len;
        u[1] /= u_len;
        u[2] /= u_len;

        // v = n x u
        v = {{
            normal[1] * u[2] - normal[2] * u[1],
            normal[2] * u[0] - normal[0] * u[2],
            normal[0] * u[1] - normal[1] * u[0]
        }};

        double max_horizon = 0.0;

        // Sample azimuth angles
        for (int az = 0; az < num_azimuth_samples; ++az) {
            double azimuth = 2.0 * M_PI * az / num_azimuth_samples;
            double cos_az = std::cos(azimuth);
            double sin_az = std::sin(azimuth);

            // Horizontal direction in local coords
            std::array<double, 3> horiz = {{
                cos_az * u[0] + sin_az * v[0],
                cos_az * u[1] + sin_az * v[1],
                cos_az * u[2] + sin_az * v[2]
            }};

            // Binary search for horizon angle
            double low = 0.0;
            double high = 90.0;

            while (high - low > 1.0) {
                double mid = (low + high) / 2.0;
                double elev_rad = mid * M_PI / 180.0;
                double cos_el = std::cos(elev_rad);
                double sin_el = std::sin(elev_rad);

                // Direction: cos(elev)*horiz + sin(elev)*normal
                std::array<double, 3> dir = {{
                    cos_el * horiz[0] + sin_el * normal[0],
                    cos_el * horiz[1] + sin_el * normal[1],
                    cos_el * horiz[2] + sin_el * normal[2]
                }};

                // Offset origin slightly above surface
                double offset = 0.01;
                std::array<double, 3> origin = {{
                    center[0] + offset * normal[0],
                    center[1] + offset * normal[1],
                    center[2] + offset * normal[2]
                }};

                // Create ray
                RTCRayHit rayhit;
                rayhit.ray.org_x = static_cast<float>(origin[0]);
                rayhit.ray.org_y = static_cast<float>(origin[1]);
                rayhit.ray.org_z = static_cast<float>(origin[2]);
                rayhit.ray.dir_x = static_cast<float>(dir[0]);
                rayhit.ray.dir_y = static_cast<float>(dir[1]);
                rayhit.ray.dir_z = static_cast<float>(dir[2]);
                rayhit.ray.tnear = 0.0f;
                rayhit.ray.tfar = 1e10f;
                rayhit.ray.mask = 0xFFFFFFFF;
                rayhit.ray.flags = 0;
                rayhit.ray.time = 0.0f;
                rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;

                rtcIntersect1(scene, &rayhit);

                if (rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID) {
                    // Hit terrain - horizon is higher
                    low = mid;
                } else {
                    // No hit - horizon is lower
                    high = mid;
                }
            }

            max_horizon = std::max(max_horizon, low);
        }

        max_angles[i] = max_horizon;
    }

    return max_angles;
}

std::vector<int> findPermanentlyShadowedFaces(
    const std::vector<std::vector<double>>& illumination_series,
    double threshold) {

    std::vector<int> psr_faces;

    for (size_t i = 0; i < illumination_series.size(); ++i) {
        const auto& series = illumination_series[i];
        if (series.empty()) continue;

        bool always_dark = true;
        for (double flux : series) {
            if (flux > threshold) {
                always_dark = false;
                break;
            }
        }

        if (always_dark) {
            psr_faces.push_back(static_cast<int>(i));
        }
    }

    return psr_faces;
}

std::vector<int> findDepressionFaces(
    const std::shared_ptr<ptm::Mesh>& mesh,
    double depth_threshold) {

    std::vector<int> depression_faces;
    size_t n = mesh->faces.size();

    // Compute face centers and radial distances
    std::vector<double> radii(n);
    double mean_radius = 0.0;

    for (size_t i = 0; i < n; ++i) {
        const auto& face = mesh->faces[i];
        const auto& v0 = mesh->vertices[face[0]];
        const auto& v1 = mesh->vertices[face[1]];
        const auto& v2 = mesh->vertices[face[2]];

        double cx = (v0[0] + v1[0] + v2[0]) / 3.0;
        double cy = (v0[1] + v1[1] + v2[1]) / 3.0;
        double cz = (v0[2] + v1[2] + v2[2]) / 3.0;

        radii[i] = std::sqrt(cx * cx + cy * cy + cz * cz);
        mean_radius += radii[i];
    }
    mean_radius /= n;

    // Find faces below mean radius by threshold
    for (size_t i = 0; i < n; ++i) {
        double relative_depth = (mean_radius - radii[i]) / mean_radius;
        if (relative_depth > depth_threshold) {
            depression_faces.push_back(static_cast<int>(i));
        }
    }

    return depression_faces;
}

} // namespace radiation
