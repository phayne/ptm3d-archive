#include "mc_view_factor.h"
#include "nearfield_quadrature.h"
#include "../thermal/thermal_model.h"  // ProgressBar

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
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
    int rays_per_face,
    const std::array<double, 3>& scene_origin_shift) {

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

    // Prefer the mesh's own face data when it carries it, exactly as
    // ViewFactorCalculator::precomputeFaceData does. The fallback below orients
    // normals by "away from the body center at the origin", which is right for
    // a body-centered mesh and WRONG for anything else -- on a gate cap built
    // about the origin it points every normal out of the back of the surface,
    // and the gather then reports 100 percent sky and an empty matrix with no
    // error. The two builders must not disagree about which way a face looks.
    const bool use_mesh_data = (mesh->face_centers.size() == n_faces &&
                                mesh->face_normals.size() == n_faces);
    if (!use_mesh_data) {
        std::cout << "  Face normals: derived here and oriented AWAY FROM THE ORIGIN "
                     "(the mesh carries none). Valid only for a body-centered mesh."
                  << std::endl;
    }

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

        if (use_mesh_data) {
            centers[i] = mesh->face_centers[i];
            normals[i] = mesh->face_normals[i];
        } else {
            // Orient outward (matches view_factor.cpp's own fallback).
            double rd = normals[i][0] * centers[i][0]
                      + normals[i][1] * centers[i][1]
                      + normals[i][2] * centers[i][2];
            if (rd < 0.0) {
                normals[i][0] = -normals[i][0];
                normals[i][1] = -normals[i][1];
                normals[i][2] = -normals[i][2];
            }
        }

        buildFrame(normals[i], us[i], vs[i]);
    }

    // Ray-origin offset to avoid self-intersection.
    double avg_area = 0.0;
    for (double a : areas) avg_area += a;
    avg_area /= static_cast<double>(n_faces);
    const double eps = 1e-4 * std::sqrt(std::max(avg_area, 1e-20));

    // The caller's mesh -> scene translation. NOT recomputed here: this used to
    // derive the bounding-box center itself and assume the caller had shifted
    // the scene identically, which is true for RadiationExchangeManager and
    // false for every harness. A mismatch does not error, it silently produces
    // a matrix from rays that started outside the geometry.
    const std::array<double, 3>& origin_shift = scene_origin_shift;

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

    // ----- Reciprocity, then the near field --------------------------------
    //
    // Two things the ray gather cannot give on its own, both of which every
    // other builder in the engine gives by construction. PTM_MC_LEGACY=1
    // restores the raw gather; it warns, and it is the negative control for
    // gate arm [M], not a production setting.
    const bool mc_legacy = [] {
        const char* e = std::getenv("PTM_MC_LEGACY");
        const bool want = (e && std::string(e) == "1");
        if (want) {
            std::cout << "  WARNING: PTM_MC_LEGACY=1 -- raw ray gather. NOT reciprocal, "
                         "and touching pairs keep the centroid-launch near-field bias. "
                         "This is a control, not a production setting." << std::endl;
        }
        return want;
    }();

    // F_ij as the gather measured it, per source. Kept as a dense-per-row map so
    // the near-field pass can overwrite an entry the gather already has.
    std::vector<std::unordered_map<int, double>> F(n_faces);
    {
        const double inv_n = 1.0 / static_cast<double>(rays_per_face);
        for (size_t src = 0; src < n_faces; ++src) {
            F[src].reserve(histograms[src].size() * 2);
            for (const auto& kv : histograms[src])
                F[src][kv.first] = static_cast<double>(kv.second) * inv_n;
        }
    }

    size_t sym_pairs = 0;
    double sym_worst_rel = 0.0;
    if (!mc_legacy) {
        // Reciprocity. The gather is a per-source estimator, so A_i F_ij and
        // A_j F_ji are two independent measurements of the same physical
        // quantity and disagree at the sampling noise. Combine them:
        //
        //     F_ij <- (A_i F_ij + A_j F_ji) / (2 A_i)
        //
        // which restores A_i F_ij == A_j F_ji exactly and is the lower-variance
        // estimate of both. The divisor is 2 unconditionally: a direction the
        // gather never sampled is a MEASURED ZERO, not a missing datum. That is
        // the view-factor divisor ruling, and dividing by "the directions that reported" is
        // exactly the bug it was.
        std::vector<std::unordered_map<int, double>> Fs(n_faces);
        for (size_t i = 0; i < n_faces; ++i) {
            for (const auto& kv : F[i]) {
                const int j = kv.first;
                if (j < 0 || static_cast<size_t>(j) >= n_faces) continue;
                if (Fs[i].count(j)) continue;  // already done from j's side
                const double fij = kv.second;
                auto it = F[j].find(static_cast<int>(i));
                const double fji = (it == F[j].end()) ? 0.0 : it->second;
                const double num = areas[i] * fij + areas[j] * fji;
                const double sij = (areas[i] > 0.0) ? num / (2.0 * areas[i]) : 0.0;
                const double sji = (areas[j] > 0.0) ? num / (2.0 * areas[j]) : 0.0;
                Fs[i][j] = sij;
                Fs[j][static_cast<int>(i)] = sji;
                ++sym_pairs;
                const double denom = std::max(std::abs(fij), 1e-300);
                sym_worst_rel = std::max(sym_worst_rel, std::abs(sij - fij) / denom);
            }
        }
        F.swap(Fs);
    }

    // Near field. The gather launches every ray from the face CENTROID, so a
    // touching pair is estimated by the same point-source sampler as a distant
    // one and carries M1's one-sided bias -- the bias the in-engine builders
    // removed on 2026-08-23. Inside the SHARED window the shared quadrature
    // replaces the gather's value outright.
    //
    // The pairs are enumerated from the GEOMETRY, not from the gather's
    // histogram: a near pair the gather happened to miss is a sampling failure,
    // and taking the histogram as the pair set would let that failure survive
    // the fix.
    size_t near_pairs = 0, near_replaced = 0, near_blocked = 0;
    double near_worst_rel = 0.0;
    if (!mc_legacy) {
        double smax = 0.0;
        for (size_t i = 0; i < n_faces; ++i) smax = std::max(smax, std::sqrt(areas[i]));
        const double reach = std::max(config.nearfield_window_factor * 2.0 * smax, 1e-12);

        // Uniform grid over face centers at the window's own reach, so every
        // near partner of a face is in its own cell or one of the 26 around it.
        std::array<double, 3> gmin = centers[0], gmax = centers[0];
        for (size_t i = 1; i < n_faces; ++i)
            for (int k = 0; k < 3; ++k) {
                gmin[k] = std::min(gmin[k], centers[i][k]);
                gmax[k] = std::max(gmax[k], centers[i][k]);
            }
        std::array<long long, 3> dim{};
        for (int k = 0; k < 3; ++k)
            dim[k] = std::max(1LL, static_cast<long long>((gmax[k] - gmin[k]) / reach) + 1);
        auto cell_of = [&](size_t i) {
            std::array<long long, 3> c{};
            for (int k = 0; k < 3; ++k) {
                long long v = static_cast<long long>((centers[i][k] - gmin[k]) / reach);
                c[k] = std::min(std::max(v, 0LL), dim[k] - 1);
            }
            return c;
        };
        auto key_of = [&](const std::array<long long, 3>& c) {
            return (c[2] * dim[1] + c[1]) * dim[0] + c[0];
        };
        std::unordered_map<long long, std::vector<int>> grid;
        grid.reserve(n_faces);
        for (size_t i = 0; i < n_faces; ++i)
            grid[key_of(cell_of(i))].push_back(static_cast<int>(i));

        const int order = config.nearfield_quadrature_order;
        const std::array<double, 3> no_shift{{0.0, 0.0, 0.0}};

        for (size_t i = 0; i < n_faces; ++i) {
            if (areas[i] <= 0.0) continue;
            const auto ci = cell_of(i);
            for (long long dz = -1; dz <= 1; ++dz)
            for (long long dy = -1; dy <= 1; ++dy)
            for (long long dx = -1; dx <= 1; ++dx) {
                const std::array<long long, 3> c{{ci[0] + dx, ci[1] + dy, ci[2] + dz}};
                if (c[0] < 0 || c[1] < 0 || c[2] < 0 ||
                    c[0] >= dim[0] || c[1] >= dim[1] || c[2] >= dim[2]) continue;
                auto it = grid.find(key_of(c));
                if (it == grid.end()) continue;
                for (int jj : it->second) {
                    const size_t j = static_cast<size_t>(jj);
                    if (j <= i || areas[j] <= 0.0) continue;
                    const double rx = centers[j][0] - centers[i][0];
                    const double ry = centers[j][1] - centers[i][1];
                    const double rz = centers[j][2] - centers[i][2];
                    const double r = std::sqrt(rx * rx + ry * ry + rz * rz);
                    if (!isNearFieldPair(r, areas[i], areas[j],
                                         config.nearfield_window_factor)) continue;
                    ++near_pairs;

                    // Same occlusion convention as the other builders: ONE
                    // canonical centroid-to-centroid ray through the shared
                    // rule. Ray origins are recentered like the gather's.
                    const std::array<double, 3> pa{{centers[i][0] - origin_shift[0],
                                                    centers[i][1] - origin_shift[1],
                                                    centers[i][2] - origin_shift[2]}};
                    const std::array<double, 3> pb{{centers[j][0] - origin_shift[0],
                                                    centers[j][1] - origin_shift[1],
                                                    centers[j][2] - origin_shift[2]}};
                    const VisibilitySegment seg =
                        visibilityRay(pa, pb, areas[i], areas[j], no_shift);
                    if (seg.unresolvable) noteVisibilityUnresolvable();
                    bool visible = true;
                    if (!seg.auto_visible && seg.tfar > 0.0) {
                        RTCRayHit rh;
                        rh.ray.org_x = static_cast<float>(seg.origin[0]);
                        rh.ray.org_y = static_cast<float>(seg.origin[1]);
                        rh.ray.org_z = static_cast<float>(seg.origin[2]);
                        rh.ray.dir_x = static_cast<float>(seg.dir[0]);
                        rh.ray.dir_y = static_cast<float>(seg.dir[1]);
                        rh.ray.dir_z = static_cast<float>(seg.dir[2]);
                        rh.ray.tnear = 0.0f;
                        rh.ray.tfar = static_cast<float>(seg.tfar);
                        rh.ray.mask = 0xFFFFFFFFu;
                        rh.ray.flags = 0;
                        rh.ray.time = 0.0f;
                        rh.hit.geomID = RTC_INVALID_GEOMETRY_ID;
                        rh.hit.primID = RTC_INVALID_GEOMETRY_ID;
                        rh.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
                        rtcIntersect1(scene, &rh);
                        visible = (rh.hit.geomID == RTC_INVALID_GEOMETRY_ID);
                    }
                    if (!visible) {
                        ++near_blocked;
                        F[i].erase(static_cast<int>(j));
                        F[j].erase(static_cast<int>(i));
                        continue;
                    }

                    const auto& ta = mesh->faces[i];
                    const auto& tb = mesh->faces[j];
                    const double g = nearFieldPairIntegral(
                        mesh->vertices[ta[0]], mesh->vertices[ta[1]], mesh->vertices[ta[2]],
                        mesh->vertices[tb[0]], mesh->vertices[tb[1]], mesh->vertices[tb[2]],
                        normals[i], normals[j], areas[i], areas[j], order);
                    if (!(g > 0.0)) {
                        F[i].erase(static_cast<int>(j));
                        F[j].erase(static_cast<int>(i));
                        continue;
                    }
                    const double fij = g / areas[i];
                    const double fji = g / areas[j];
                    auto old_it = F[i].find(static_cast<int>(j));
                    if (old_it != F[i].end() && old_it->second > 0.0)
                        near_worst_rel = std::max(near_worst_rel,
                                                  std::abs(fij - old_it->second) / old_it->second);
                    F[i][static_cast<int>(j)] = fij;
                    F[j][static_cast<int>(i)] = fji;
                    ++near_replaced;
                }
            }
        }
    }

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

    std::vector<std::pair<double, int>> entries;  // (F_ij, target face)

    size_t sky_clamped = 0;
    double sky_worst_negative = 0.0;

    for (size_t src = 0; src < n_faces; ++src) {
        auto& row = F[src];
        total_unique += row.size();

        entries.clear();
        entries.reserve(row.size());
        for (const auto& kv : row) entries.emplace_back(kv.second, kv.first);
        std::unordered_map<int, double>().swap(row);
        std::unordered_map<int, int>().swap(histograms[src]);

        if (static_cast<int>(entries.size()) > max_neighbors) {
            ++faces_at_cap;
            std::partial_sort(
                entries.begin(), entries.begin() + max_neighbors, entries.end(),
                [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
                    return a.first > b.first;
                });
            entries.resize(static_cast<size_t>(max_neighbors));
        }
        // Deterministic row order: an unordered_map's iteration order is not a
        // property of the data, and the CSR built from it would not be either.
        std::sort(entries.begin(), entries.end(),
                  [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
                      return a.second < b.second;
                  });

        double row_sum = 0.0;
        for (const auto& e : entries) {
            if (e.first < min_value) continue;  // prune sub-threshold terms
            row_indices.push_back(static_cast<int>(src));
            col_indices.push_back(e.second);
            values.push_back(e.first);
            row_sum += e.first;
        }

        // The sky channel is now DERIVED, not accumulated. It has to be: the
        // near-field replacement and the reciprocity combination both change
        // row sums, and carrying the gather's own miss count alongside them
        // would report a closure the matrix does not have. Sum F + F_sky = 1 by
        // construction, and a row that would need a negative sky is counted and
        // reported rather than clamped in silence.
        double sky_frac = 1.0 - row_sum;
        if (sky_frac < 0.0) {
            ++sky_clamped;
            sky_worst_negative = std::min(sky_worst_negative, sky_frac);
            sky_frac = 0.0;
        }
        sky_frac_sum += sky_frac;
    }

    const size_t nnz = values.size();
    std::cout << "  CSR entries: " << nnz
              << "  (" << (nnz * 12.0 / (1024.0 * 1024.0)) << " MB @12B/entry)" << std::endl;
    std::cout << "  Faces at max_neighbors cap: " << faces_at_cap
              << " / " << n_faces << std::endl;
    std::cout << "  Mean unique targets per source: "
              << (static_cast<double>(total_unique) / static_cast<double>(n_faces)) << std::endl;
    std::cout << "  Mean sky fraction (F_sky = 1 - sum_j F_ij): "
              << (sky_frac_sum / static_cast<double>(n_faces)) << std::endl;
    if (mc_legacy) {
        std::cout << "  Reciprocity: NOT enforced (PTM_MC_LEGACY=1). Near field: NONE."
                  << std::endl;
    } else {
        std::cout << "  Reciprocity: " << sym_pairs
                  << " pair(s) combined (A_i F_ij == A_j F_ji exactly); worst single-"
                     "direction move " << (100.0 * sym_worst_rel) << "%" << std::endl;
        std::cout << "  Near field: " << near_replaced << " of " << near_pairs
                  << " in-window pair(s) taken from the shared quadrature at order "
                  << config.nearfield_quadrature_order << " (" << near_blocked
                  << " occluded); worst change vs the gather "
                  << (100.0 * near_worst_rel) << "%" << std::endl;
    }
    if (sky_clamped > 0) {
        std::cout << "  WARNING: " << sky_clamped << " row(s) had sum_j F_ij > 1 and their "
                     "sky fraction was clamped to zero; worst 1 - sum = "
                  << sky_worst_negative << ". Energy is NOT conserved on those rows."
                  << std::endl;
    }
    reportNearFieldBoundClamps("Monte Carlo builder", config.nearfield_quadrature_order);
    reportVisibilityUnresolvable("Monte Carlo builder");

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
