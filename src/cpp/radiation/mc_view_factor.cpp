#include "mc_view_factor.h"
#include "../thermal/thermal_model.h"  // ProgressBar

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <unordered_map>
#include <utility>
#include <vector>

#include <embree4/rtcore.h>
#include <embree4/rtcore_ray.h>
#include <omp.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace radiation {

namespace {

// Build an orthonormal tangent frame (u, v, n) at a face.
// Matches the IlluminationModel::createCoordinateSystem convention.
inline void buildFrame(const std::array<double, 3>& n,
                       std::array<double, 3>& u,
                       std::array<double, 3>& v) {
    std::array<double, 3> up = {{0.0, 0.0, 1.0}};
    if (std::abs(n[2]) > 0.9) up = {{0.0, 1.0, 0.0}};

    // u = up × n
    u[0] = up[1] * n[2] - up[2] * n[1];
    u[1] = up[2] * n[0] - up[0] * n[2];
    u[2] = up[0] * n[1] - up[1] * n[0];
    double ulen = std::sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
    if (ulen > 1e-12) {
        u[0] /= ulen;
        u[1] /= ulen;
        u[2] /= ulen;
    }

    // v = n × u
    v[0] = n[1] * u[2] - n[2] * u[1];
    v[1] = n[2] * u[0] - n[0] * u[2];
    v[2] = n[0] * u[1] - n[1] * u[0];
}

// Cosine-weighted hemisphere sample in local (u, v, n) frame.
// Malley's method: sample uniform disk, lift to sphere.
inline void cosineHemisphere(double xi1, double xi2,
                             double& x, double& y, double& z) {
    double r = std::sqrt(xi1);
    double phi = 2.0 * M_PI * xi2;
    x = r * std::cos(phi);
    y = r * std::sin(phi);
    z = std::sqrt(std::max(0.0, 1.0 - xi1));
}

}  // namespace

ViewFactorMatrix computeMonteCarlo(
    const std::shared_ptr<ptm::Mesh>& mesh,
    RTCDevice /*device*/,
    RTCScene scene,
    const RadiationConfig& config,
    int rays_per_face) {

    const size_t n_faces = mesh->faces.size();
    const int max_neighbors = std::max(1, config.max_neighbors);

    std::cout << "Monte Carlo view factor computation:" << std::endl;
    std::cout << "  Faces: " << n_faces << std::endl;
    std::cout << "  Rays per face: " << rays_per_face << std::endl;
    std::cout << "  Top-K compaction cap: " << max_neighbors << " neighbors/face" << std::endl;

    // ----- Precompute per-face geometry -----
    std::vector<std::array<double, 3>> centers(n_faces);
    std::vector<std::array<double, 3>> normals(n_faces);
    std::vector<std::array<double, 3>> us(n_faces), vs(n_faces);
    std::vector<double> areas(n_faces);

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n_faces; ++i) {
        const auto& face = mesh->faces[i];
        const auto& v0 = mesh->vertices[face[0]];
        const auto& v1 = mesh->vertices[face[1]];
        const auto& v2 = mesh->vertices[face[2]];

        centers[i] = {{(v0[0] + v1[0] + v2[0]) / 3.0,
                       (v0[1] + v1[1] + v2[1]) / 3.0,
                       (v0[2] + v1[2] + v2[2]) / 3.0}};

        std::array<double, 3> e1{{v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]}};
        std::array<double, 3> e2{{v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]}};
        std::array<double, 3> nrm{{e1[1] * e2[2] - e1[2] * e2[1],
                                    e1[2] * e2[0] - e1[0] * e2[2],
                                    e1[0] * e2[1] - e1[1] * e2[0]}};
        double L = std::sqrt(nrm[0] * nrm[0] + nrm[1] * nrm[1] + nrm[2] * nrm[2]);
        areas[i] = 0.5 * L;
        if (L > 1e-20) {
            normals[i] = {{nrm[0] / L, nrm[1] / L, nrm[2] / L}};
        } else {
            normals[i] = {{0.0, 0.0, 1.0}};
        }

        // Orient outward (matches PTM3D and view_factor.cpp convention).
        double rd = normals[i][0] * centers[i][0]
                  + normals[i][1] * centers[i][1]
                  + normals[i][2] * centers[i][2];
        if (rd < 0.0) {
            normals[i][0] = -normals[i][0];
            normals[i][1] = -normals[i][1];
            normals[i][2] = -normals[i][2];
        }

        buildFrame(normals[i], us[i], vs[i]);
    }

    // Ray-origin offset to avoid self-intersection.
    double avg_area = 0.0;
    for (double a : areas) avg_area += a;
    avg_area /= static_cast<double>(n_faces);
    const double eps = 1e-4 * std::sqrt(std::max(avg_area, 1e-20));

    // Recenter ray origins to the mesh bounding-box center, matching the Embree
    // scene recentering in RadiationExchangeManager::initializeScene(). The scene
    // vertices are shifted by this same bbox center for float32 precision; if the
    // MC ray origins stayed at raw selenocentric coordinates (|Z|~1.7e6 m) they
    // would start ~1.7e6 m outside the recentered BVH and EVERY ray would miss,
    // producing an empty view-factor matrix (no radiation coupling). View factors
    // are translation-invariant, so this constant shift is exact. The H-matrix
    // builder already applies the identical shift; the MC path previously did not.
    std::array<double, 3> bbmin = mesh->vertices[0];
    std::array<double, 3> bbmax = mesh->vertices[0];
    for (const auto& vtx : mesh->vertices) {
        for (int k = 0; k < 3; ++k) {
            bbmin[k] = std::min(bbmin[k], vtx[k]);
            bbmax[k] = std::max(bbmax[k], vtx[k]);
        }
    }
    const std::array<double, 3> origin_shift = {{0.5 * (bbmin[0] + bbmax[0]),
                                                 0.5 * (bbmin[1] + bbmax[1]),
                                                 0.5 * (bbmin[2] + bbmax[2])}};

    // ----- Per-face histograms: hit_face -> count -----
    // Written by one thread per source (OMP for is work-partitioned by source),
    // so no cross-thread contention on individual maps.
    std::vector<std::unordered_map<int, int>> histograms(n_faces);
    std::vector<int> sky_counts(n_faces, 0);

    std::atomic<size_t> progress{0};
    thermal::ProgressBar pb(static_cast<int>(n_faces), "MC view factor: ", 50, true);
    pb.update(0);

    auto t0 = std::chrono::high_resolution_clock::now();

    #pragma omp parallel
    {
        // Thread-local PRNG, deterministic across runs given same thread count.
        std::mt19937_64 rng(0xc0ffeeULL + static_cast<uint64_t>(omp_get_thread_num()));
        std::uniform_real_distribution<double> U(0.0, 1.0);

        #pragma omp for schedule(dynamic, 8)
        for (size_t src = 0; src < n_faces; ++src) {
            if (areas[src] <= 0.0) continue;

            auto& hist = histograms[src];
            hist.reserve(static_cast<size_t>(max_neighbors) * 2);

            const auto& c = centers[src];
            const auto& n = normals[src];
            const auto& u = us[src];
            const auto& v = vs[src];

            const float org_x = static_cast<float>(c[0] - origin_shift[0] + eps * n[0]);
            const float org_y = static_cast<float>(c[1] - origin_shift[1] + eps * n[1]);
            const float org_z = static_cast<float>(c[2] - origin_shift[2] + eps * n[2]);

            int sky_local = 0;
            const int src_int = static_cast<int>(src);

            for (int chunk = 0; chunk < rays_per_face; chunk += 16) {
                const int pkt_size = std::min(16, rays_per_face - chunk);

                alignas(64) int valid[16];
                alignas(64) RTCRayHit16 rh;
                std::memset(valid, 0, sizeof(valid));

                for (int r = 0; r < pkt_size; ++r) {
                    valid[r] = -1;

                    const double xi1 = U(rng);
                    const double xi2 = U(rng);
                    double lx, ly, lz;
                    cosineHemisphere(xi1, xi2, lx, ly, lz);

                    // World direction: lx*u + ly*v + lz*n
                    const double dx = lx * u[0] + ly * v[0] + lz * n[0];
                    const double dy = lx * u[1] + ly * v[1] + lz * n[1];
                    const double dz = lx * u[2] + ly * v[2] + lz * n[2];

                    rh.ray.org_x[r] = org_x;
                    rh.ray.org_y[r] = org_y;
                    rh.ray.org_z[r] = org_z;
                    rh.ray.dir_x[r] = static_cast<float>(dx);
                    rh.ray.dir_y[r] = static_cast<float>(dy);
                    rh.ray.dir_z[r] = static_cast<float>(dz);
                    rh.ray.tnear[r] = 1.0e-4f;
                    rh.ray.tfar[r] = 1.0e30f;
                    rh.ray.mask[r] = 0xFFFFFFFFu;
                    rh.ray.flags[r] = 0;
                    rh.ray.time[r] = 0.0f;
                    rh.hit.geomID[r] = RTC_INVALID_GEOMETRY_ID;
                    rh.hit.primID[r] = RTC_INVALID_GEOMETRY_ID;
                    rh.hit.instID[0][r] = RTC_INVALID_GEOMETRY_ID;
                }

                // Pad remaining slots as inactive.
                for (int r = pkt_size; r < 16; ++r) {
                    valid[r] = 0;
                    rh.ray.org_x[r] = 0.0f;
                    rh.ray.org_y[r] = 0.0f;
                    rh.ray.org_z[r] = 0.0f;
                    rh.ray.dir_x[r] = 0.0f;
                    rh.ray.dir_y[r] = 0.0f;
                    rh.ray.dir_z[r] = 1.0f;
                    rh.ray.tnear[r] = 0.0f;
                    rh.ray.tfar[r] = -1.0f;
                    rh.ray.mask[r] = 0xFFFFFFFFu;
                    rh.ray.flags[r] = 0;
                    rh.ray.time[r] = 0.0f;
                    rh.hit.geomID[r] = RTC_INVALID_GEOMETRY_ID;
                    rh.hit.primID[r] = RTC_INVALID_GEOMETRY_ID;
                    rh.hit.instID[0][r] = RTC_INVALID_GEOMETRY_ID;
                }

                RTCIntersectArguments args;
                rtcInitIntersectArguments(&args);
                rtcIntersect16(valid, scene, &rh, &args);

                for (int r = 0; r < pkt_size; ++r) {
                    if (rh.hit.geomID[r] != RTC_INVALID_GEOMETRY_ID) {
                        const int hit = static_cast<int>(rh.hit.primID[r]);
                        if (hit != src_int) {
                            hist[hit]++;
                        }
                        // else: self-hit from near-origin numerical noise, drop.
                    } else {
                        ++sky_local;
                    }
                }
            }

            sky_counts[src] = sky_local;

            const size_t done = progress.fetch_add(1, std::memory_order_relaxed) + 1;
            // Update progress bar every ~0.1% or at completion to avoid lock thrash.
            if ((done & 0xFFFu) == 0 || done == n_faces) {
                #pragma omp critical
                pb.update(static_cast<int>(done));
            }
        }
    }
    pb.finish();

    auto t1 = std::chrono::high_resolution_clock::now();
    const double wall_tracing = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "  Ray tracing + accumulation: " << wall_tracing << " s" << std::endl;

    // ----- Top-K compaction + triplet assembly -----
    const double inv_n_rays = 1.0 / static_cast<double>(rays_per_face);
    const double min_value = std::max(config.visibility_threshold, 0.0);

    std::vector<int> row_indices;
    std::vector<int> col_indices;
    std::vector<double> values;

    size_t faces_at_cap = 0;
    double sky_frac_sum = 0.0;
    size_t total_unique = 0;

    // Rough reservation: assume ~max_neighbors per source on average.
    row_indices.reserve(n_faces * static_cast<size_t>(max_neighbors) / 2);
    col_indices.reserve(row_indices.capacity());
    values.reserve(row_indices.capacity());

    std::vector<std::pair<int, int>> entries;  // (count, hit_face)

    for (size_t src = 0; src < n_faces; ++src) {
        auto& hist = histograms[src];
        int sky = sky_counts[src];
        total_unique += hist.size();

        entries.clear();
        entries.reserve(hist.size());
        for (const auto& kv : hist) {
            entries.emplace_back(kv.second, kv.first);
        }
        // Free the hashmap now — we have what we need.
        std::unordered_map<int, int>().swap(hist);

        if (static_cast<int>(entries.size()) > max_neighbors) {
            ++faces_at_cap;
            std::partial_sort(
                entries.begin(),
                entries.begin() + max_neighbors,
                entries.end(),
                [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
                    return a.first > b.first;
                });
            // Tail counts are reassigned to the sky accumulator so total
            // probability mass is conserved (Σ F + F_sky = 1 exactly at the
            // estimator level; truncation shows up only in F_sky).
            for (size_t k = static_cast<size_t>(max_neighbors); k < entries.size(); ++k) {
                sky += entries[k].first;
            }
            entries.resize(static_cast<size_t>(max_neighbors));
        }

        for (const auto& e : entries) {
            const double val = static_cast<double>(e.first) * inv_n_rays;
            if (val < min_value) continue;  // prune sub-threshold terms
            row_indices.push_back(static_cast<int>(src));
            col_indices.push_back(e.second);
            values.push_back(val);
        }

        sky_frac_sum += static_cast<double>(sky) * inv_n_rays;
    }

    const size_t nnz = values.size();
    std::cout << "  CSR entries: " << nnz
              << "  (" << (nnz * 12.0 / (1024.0 * 1024.0)) << " MB @12B/entry)" << std::endl;
    std::cout << "  Faces at max_neighbors cap: " << faces_at_cap
              << " / " << n_faces << std::endl;
    std::cout << "  Mean unique targets per source: "
              << (static_cast<double>(total_unique) / static_cast<double>(n_faces)) << std::endl;
    std::cout << "  Mean sky fraction (F_sky): "
              << (sky_frac_sum / static_cast<double>(n_faces)) << std::endl;

    ViewFactorMatrix vf;
    vf.buildFromTriplets(n_faces, row_indices, col_indices, values);

    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "  CSR assembly: "
              << std::chrono::duration<double>(t2 - t1).count() << " s" << std::endl;
    std::cout << "  MC view factor total: "
              << std::chrono::duration<double>(t2 - t0).count() << " s" << std::endl;

    return vf;
}

}  // namespace radiation
