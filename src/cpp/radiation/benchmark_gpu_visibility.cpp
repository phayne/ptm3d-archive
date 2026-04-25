/**
 * @file benchmark_gpu_visibility.cpp
 * @brief Rays-per-second micro-benchmark: CPU Embree 16-wide packets vs GPU OpenCL BVH.
 *
 * Loads a triangulated mesh (PLY), generates N realistic view-factor-style shadow
 * rays (source face -> nearby target face, backface-culled, origin offset along
 * normal, tmax = |dir| - eps), and times both engines on the same ray set.
 *
 * This is a prototype for Tier 3 (GPU-dispatched visibility) of the view-factor
 * optimisation effort. It does NOT modify the production VF build path; it only
 * measures whether GPU dispatch of shadow rays is worth pursuing on this system.
 *
 * Usage: benchmark_gpu_visibility <mesh.ply> [num_rays=1000000] [runs=3]
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <embree4/rtcore.h>
#include <embree4/rtcore_ray.h>
#include <omp.h>

#include "../mesh/mesh_utils.h"

#ifdef PTM_GPU_OPENCL
#include "../gpu/bvh_opencl.h"
#endif

namespace {

// --- AVX-512 detection (copy of view_factor.cpp pattern) -----------------
bool hasAVX512() {
#if defined(__x86_64__) || defined(_M_X64)
    unsigned int eax, ebx, ecx, edx;
#ifdef _MSC_VER
    int cpuInfo[4];
    __cpuidex(cpuInfo, 7, 0);
    ebx = cpuInfo[1];
#else
    __asm__ __volatile__("cpuid"
                         : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
                         : "a"(7), "c"(0));
#endif
    return (ebx & (1 << 16)) != 0;
#else
    return false;
#endif
}

// --- Ray definition ---------------------------------------------------------
struct Ray {
    std::array<double, 3> origin;
    std::array<double, 3> dir;    // normalized
    float tmax;                   // tfar along dir
};

// --- Embree scene construction (adapted from benchmark_hmatrix.cpp) ----
RTCScene createEmbreeScene(RTCDevice device, const std::shared_ptr<ptm::Mesh>& mesh) {
    RTCScene scene = rtcNewScene(device);
    RTCGeometry geom = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);

    float* verts = static_cast<float*>(rtcSetNewGeometryBuffer(
        geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3,
        3 * sizeof(float), mesh->vertices.size()));
    for (size_t i = 0; i < mesh->vertices.size(); ++i) {
        verts[3 * i + 0] = static_cast<float>(mesh->vertices[i][0]);
        verts[3 * i + 1] = static_cast<float>(mesh->vertices[i][1]);
        verts[3 * i + 2] = static_cast<float>(mesh->vertices[i][2]);
    }

    unsigned int* idx = static_cast<unsigned int*>(rtcSetNewGeometryBuffer(
        geom, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3,
        3 * sizeof(unsigned int), mesh->faces.size()));
    for (size_t i = 0; i < mesh->faces.size(); ++i) {
        idx[3 * i + 0] = static_cast<unsigned int>(mesh->faces[i][0]);
        idx[3 * i + 1] = static_cast<unsigned int>(mesh->faces[i][1]);
        idx[3 * i + 2] = static_cast<unsigned int>(mesh->faces[i][2]);
    }

    rtcCommitGeometry(geom);
    rtcAttachGeometry(scene, geom);
    rtcReleaseGeometry(geom);
    rtcCommitScene(scene);
    return scene;
}

// --- Face geometry ---------------------------------------------------------
struct FaceGeom {
    std::vector<std::array<double, 3>> centers;
    std::vector<std::array<double, 3>> normals;
    std::vector<double> areas;
};

FaceGeom buildFaceGeom(const std::shared_ptr<ptm::Mesh>& mesh) {
    size_t n = mesh->faces.size();
    FaceGeom g;
    g.centers.resize(n);
    g.normals.resize(n);
    g.areas.resize(n);

    auto sub = [](const std::array<double, 3>& a, const std::array<double, 3>& b) {
        return std::array<double, 3>{{a[0] - b[0], a[1] - b[1], a[2] - b[2]}};
    };
    auto cross = [](const std::array<double, 3>& a, const std::array<double, 3>& b) {
        return std::array<double, 3>{{a[1]*b[2] - a[2]*b[1],
                                       a[2]*b[0] - a[0]*b[2],
                                       a[0]*b[1] - a[1]*b[0]}};
    };
    auto length = [](const std::array<double, 3>& v) {
        return std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    };

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        const auto& f = mesh->faces[i];
        const auto& v0 = mesh->vertices[f[0]];
        const auto& v1 = mesh->vertices[f[1]];
        const auto& v2 = mesh->vertices[f[2]];
        g.centers[i] = {{(v0[0] + v1[0] + v2[0]) / 3.0,
                         (v0[1] + v1[1] + v2[1]) / 3.0,
                         (v0[2] + v1[2] + v2[2]) / 3.0}};
        auto e1 = sub(v1, v0);
        auto e2 = sub(v2, v0);
        auto nrm = cross(e1, e2);
        double L = length(nrm);
        g.areas[i] = 0.5 * L;
        if (L > 0) {
            g.normals[i] = {{nrm[0] / L, nrm[1] / L, nrm[2] / L}};
        } else {
            g.normals[i] = {{0, 0, 1}};
        }
        // Orient outward (away from origin, matching PTM convention)
        double rd = g.normals[i][0] * g.centers[i][0] +
                    g.normals[i][1] * g.centers[i][1] +
                    g.normals[i][2] * g.centers[i][2];
        if (rd < 0) {
            g.normals[i][0] = -g.normals[i][0];
            g.normals[i][1] = -g.normals[i][1];
            g.normals[i][2] = -g.normals[i][2];
        }
    }
    return g;
}

// --- Simple uniform spatial grid for nearest-face queries -------------------
struct FaceGrid {
    std::array<double, 3> mn{{0, 0, 0}};
    std::array<double, 3> mx{{0, 0, 0}};
    std::array<double, 3> cs{{1, 1, 1}};     // cell size
    std::array<int, 3> dims{{1, 1, 1}};
    std::vector<std::vector<int>> cells;

    int cellIdx(const std::array<double, 3>& p) const {
        int ix = std::max(0, std::min(dims[0] - 1,
                                      static_cast<int>((p[0] - mn[0]) / cs[0])));
        int iy = std::max(0, std::min(dims[1] - 1,
                                      static_cast<int>((p[1] - mn[1]) / cs[1])));
        int iz = std::max(0, std::min(dims[2] - 1,
                                      static_cast<int>((p[2] - mn[2]) / cs[2])));
        return (iz * dims[1] + iy) * dims[0] + ix;
    }

    void build(const std::vector<std::array<double, 3>>& centers, double cell_hint) {
        mn = centers[0];
        mx = centers[0];
        for (const auto& c : centers) {
            for (int k = 0; k < 3; ++k) {
                mn[k] = std::min(mn[k], c[k]);
                mx[k] = std::max(mx[k], c[k]);
            }
        }
        for (int k = 0; k < 3; ++k) {
            double extent = std::max(1e-12, mx[k] - mn[k]);
            // pad slightly to keep boundary faces in-grid
            mx[k] += 1e-6 * extent;
            mn[k] -= 1e-6 * extent;
            extent = mx[k] - mn[k];
            dims[k] = std::max(1, static_cast<int>(std::ceil(extent / std::max(1e-12, cell_hint))));
            cs[k] = extent / dims[k];
        }
        cells.assign(static_cast<size_t>(dims[0]) * dims[1] * dims[2], {});
        for (size_t i = 0; i < centers.size(); ++i) {
            cells[cellIdx(centers[i])].push_back(static_cast<int>(i));
        }
    }

    // Append all face indices within [cx ± r] to out.
    void query(const std::array<double, 3>& p, double r, std::vector<int>& out) const {
        std::array<int, 3> lo{}, hi{};
        for (int k = 0; k < 3; ++k) {
            lo[k] = std::max(0, static_cast<int>((p[k] - r - mn[k]) / cs[k]));
            hi[k] = std::min(dims[k] - 1, static_cast<int>((p[k] + r - mn[k]) / cs[k]));
        }
        for (int iz = lo[2]; iz <= hi[2]; ++iz) {
            for (int iy = lo[1]; iy <= hi[1]; ++iy) {
                for (int ix = lo[0]; ix <= hi[0]; ++ix) {
                    int idx = (iz * dims[1] + iy) * dims[0] + ix;
                    const auto& cell = cells[idx];
                    out.insert(out.end(), cell.begin(), cell.end());
                }
            }
        }
    }
};

// --- Ray generation: realistic VF shadow rays ------------------------------
// Strategy: sample random source face; use spatial grid to find candidates
// within search_radius; shuffle; accept first target that passes backface cull.
// Matches the candidate distribution of view_factor.cpp's findNeighbors path.
std::vector<Ray> generateVFRays(const FaceGeom& g, size_t num_rays, uint64_t seed) {
    const size_t n = g.centers.size();
    const double EPS_REL = 1e-4;

    double avg_area = 0.0;
    for (double a : g.areas) avg_area += a;
    avg_area /= n;
    double sqrt_area = std::sqrt(std::max(avg_area, 1e-20));

    // Bounding-box diameter
    std::array<double, 3> mn = g.centers[0], mx = g.centers[0];
    for (const auto& c : g.centers) {
        for (int k = 0; k < 3; ++k) {
            mn[k] = std::min(mn[k], c[k]);
            mx[k] = std::max(mx[k], c[k]);
        }
    }
    double dx = mx[0] - mn[0], dy = mx[1] - mn[1], dz = mx[2] - mn[2];
    double diag = std::sqrt(dx*dx + dy*dy + dz*dz);
    // Mirror view_factor.cpp's adaptive radius (face-based ~10*sqrt(A),
    // capped at a fraction of the mesh extent so we don't cross the whole scene).
    double search_r = std::min(20.0 * sqrt_area, 0.10 * diag);
    if (search_r <= 0) search_r = 0.05 * diag;

    double eps = EPS_REL * sqrt_area;

    // Build grid using the search radius as cell hint.
    FaceGrid grid;
    grid.build(g.centers, search_r);

    std::vector<Ray> rays;
    rays.reserve(num_rays);

    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<size_t> face_dist(0, n - 1);

    std::vector<int> cands;
    cands.reserve(4096);

    size_t attempts = 0;
    const size_t max_attempts = num_rays * 8 + 1000;  // bound total work
    while (rays.size() < num_rays && attempts < max_attempts) {
        ++attempts;
        size_t i = face_dist(rng);
        const auto& c_i = g.centers[i];
        const auto& n_i = g.normals[i];

        cands.clear();
        grid.query(c_i, search_r, cands);
        if (cands.empty()) continue;

        // Shuffle lightly: pick a random start index and walk forward.
        std::uniform_int_distribution<size_t> cand_dist(0, cands.size() - 1);
        size_t start = cand_dist(rng);
        bool accepted = false;
        for (size_t k = 0; k < cands.size() && !accepted; ++k) {
            int j = cands[(start + k) % cands.size()];
            if (static_cast<size_t>(j) == i) continue;

            const auto& c_j = g.centers[j];
            const auto& n_j = g.normals[j];

            std::array<double, 3> r_vec = {{c_j[0] - c_i[0],
                                            c_j[1] - c_i[1],
                                            c_j[2] - c_i[2]}};
            double r2 = r_vec[0]*r_vec[0] + r_vec[1]*r_vec[1] + r_vec[2]*r_vec[2];
            if (r2 < 1e-20) continue;
            double r = std::sqrt(r2);
            if (r > search_r) continue;

            std::array<double, 3> r_hat = {{r_vec[0] / r, r_vec[1] / r, r_vec[2] / r}};
            double dot_ni = n_i[0]*r_hat[0] + n_i[1]*r_hat[1] + n_i[2]*r_hat[2];
            double dot_nj = n_j[0]*r_hat[0] + n_j[1]*r_hat[1] + n_j[2]*r_hat[2];
            if (dot_ni <= 0.0 || dot_nj >= 0.0) continue;

            Ray ray;
            ray.origin = {{c_i[0] + eps * n_i[0],
                           c_i[1] + eps * n_i[1],
                           c_i[2] + eps * n_i[2]}};
            ray.dir = r_hat;
            ray.tmax = static_cast<float>(r - 2.0 * eps);
            if (ray.tmax <= 0) continue;
            rays.push_back(ray);
            accepted = true;
        }
    }
    if (rays.size() < num_rays) {
        std::cerr << "[warn] Only generated " << rays.size() << "/" << num_rays
                  << " valid rays (" << attempts << " attempts). Using "
                  << rays.size() << "." << std::endl;
    }
    std::cout << "  search_radius=" << search_r
              << " m, sqrt(avg_area)=" << sqrt_area
              << " m, diag=" << diag << " m." << std::endl;
    return rays;
}

// --- CPU 16-wide Embree packet visibility (same pattern as view_factor.cpp) -
void cpuVisibility16(RTCScene scene, const std::vector<Ray>& rays,
                     std::vector<uint8_t>& occluded_out) {
    const size_t n = rays.size();
    occluded_out.assign(n, 0);

#pragma omp parallel
    {
        alignas(64) float org_x[16], org_y[16], org_z[16];
        alignas(64) float dir_x[16], dir_y[16], dir_z[16];
        alignas(64) float tnear[16], tfar[16];
        alignas(64) int valid[16];

#pragma omp for schedule(static)
        for (size_t base = 0; base < n; base += 16) {
            size_t fill = std::min<size_t>(16, n - base);
            for (int j = 0; j < 16; ++j) {
                if (static_cast<size_t>(j) < fill) {
                    const auto& r = rays[base + j];
                    org_x[j] = static_cast<float>(r.origin[0]);
                    org_y[j] = static_cast<float>(r.origin[1]);
                    org_z[j] = static_cast<float>(r.origin[2]);
                    dir_x[j] = static_cast<float>(r.dir[0]);
                    dir_y[j] = static_cast<float>(r.dir[1]);
                    dir_z[j] = static_cast<float>(r.dir[2]);
                    tnear[j] = 0.0f;
                    tfar[j] = r.tmax;
                    valid[j] = -1;
                } else {
                    org_x[j] = org_y[j] = org_z[j] = 0.0f;
                    dir_x[j] = dir_y[j] = dir_z[j] = 0.0f;
                    tnear[j] = 0.0f;
                    tfar[j] = -1.0f;
                    valid[j] = 0;
                }
            }

            RTCRayHit16 rh;
            for (int j = 0; j < 16; ++j) {
                rh.ray.org_x[j] = org_x[j];
                rh.ray.org_y[j] = org_y[j];
                rh.ray.org_z[j] = org_z[j];
                rh.ray.dir_x[j] = dir_x[j];
                rh.ray.dir_y[j] = dir_y[j];
                rh.ray.dir_z[j] = dir_z[j];
                rh.ray.tnear[j] = tnear[j];
                rh.ray.tfar[j] = tfar[j];
                rh.ray.mask[j] = 0xFFFFFFFF;
                rh.ray.flags[j] = 0;
                rh.ray.time[j] = 0.0f;
                rh.ray.id[j] = 0;
                rh.hit.geomID[j] = RTC_INVALID_GEOMETRY_ID;
                rh.hit.primID[j] = RTC_INVALID_GEOMETRY_ID;
                rh.hit.instID[0][j] = RTC_INVALID_GEOMETRY_ID;
            }
            rtcIntersect16(valid, scene, &rh);
            for (size_t j = 0; j < fill; ++j) {
                occluded_out[base + j] =
                    (rh.hit.geomID[j] != RTC_INVALID_GEOMETRY_ID) ? 1 : 0;
            }
        }
    }
}

// --- CPU 8-wide fallback ---------------------------------------------------
void cpuVisibility8(RTCScene scene, const std::vector<Ray>& rays,
                    std::vector<uint8_t>& occluded_out) {
    const size_t n = rays.size();
    occluded_out.assign(n, 0);

#pragma omp parallel
    {
        alignas(32) float org_x[8], org_y[8], org_z[8];
        alignas(32) float dir_x[8], dir_y[8], dir_z[8];
        alignas(32) float tnear[8], tfar[8];
        alignas(32) int valid[8];

#pragma omp for schedule(static)
        for (size_t base = 0; base < n; base += 8) {
            size_t fill = std::min<size_t>(8, n - base);
            for (int j = 0; j < 8; ++j) {
                if (static_cast<size_t>(j) < fill) {
                    const auto& r = rays[base + j];
                    org_x[j] = static_cast<float>(r.origin[0]);
                    org_y[j] = static_cast<float>(r.origin[1]);
                    org_z[j] = static_cast<float>(r.origin[2]);
                    dir_x[j] = static_cast<float>(r.dir[0]);
                    dir_y[j] = static_cast<float>(r.dir[1]);
                    dir_z[j] = static_cast<float>(r.dir[2]);
                    tnear[j] = 0.0f;
                    tfar[j] = r.tmax;
                    valid[j] = -1;
                } else {
                    org_x[j] = org_y[j] = org_z[j] = 0.0f;
                    dir_x[j] = dir_y[j] = dir_z[j] = 0.0f;
                    tnear[j] = 0.0f;
                    tfar[j] = -1.0f;
                    valid[j] = 0;
                }
            }

            RTCRayHit8 rh;
            for (int j = 0; j < 8; ++j) {
                rh.ray.org_x[j] = org_x[j];
                rh.ray.org_y[j] = org_y[j];
                rh.ray.org_z[j] = org_z[j];
                rh.ray.dir_x[j] = dir_x[j];
                rh.ray.dir_y[j] = dir_y[j];
                rh.ray.dir_z[j] = dir_z[j];
                rh.ray.tnear[j] = tnear[j];
                rh.ray.tfar[j] = tfar[j];
                rh.ray.mask[j] = 0xFFFFFFFF;
                rh.ray.flags[j] = 0;
                rh.ray.time[j] = 0.0f;
                rh.ray.id[j] = 0;
                rh.hit.geomID[j] = RTC_INVALID_GEOMETRY_ID;
                rh.hit.primID[j] = RTC_INVALID_GEOMETRY_ID;
                rh.hit.instID[0][j] = RTC_INVALID_GEOMETRY_ID;
            }
            rtcIntersect8(valid, scene, &rh);
            for (size_t j = 0; j < fill; ++j) {
                occluded_out[base + j] =
                    (rh.hit.geomID[j] != RTC_INVALID_GEOMETRY_ID) ? 1 : 0;
            }
        }
    }
}

void cpuVisibility(RTCScene scene, const std::vector<Ray>& rays,
                   std::vector<uint8_t>& occluded_out, bool use_avx512) {
    if (use_avx512) {
        cpuVisibility16(scene, rays, occluded_out);
    } else {
        cpuVisibility8(scene, rays, occluded_out);
    }
}

double sampleAgree(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
    if (a.size() != b.size() || a.empty()) return 0.0;
    size_t agree = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] == b[i]) ++agree;
    }
    return static_cast<double>(agree) / static_cast<double>(a.size());
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <mesh.ply> [num_rays=1000000] [runs=3]" << std::endl;
        return 1;
    }
    std::string mesh_file = argv[1];
    size_t num_rays = 1'000'000;
    int runs = 3;
    if (argc >= 3) num_rays = std::stoull(argv[2]);
    if (argc >= 4) runs = std::atoi(argv[3]);

    std::cout << "==== GPU vs CPU Shadow Ray Benchmark ====" << std::endl;
    std::cout << "Mesh: " << mesh_file << std::endl;
    std::cout << "Rays: " << num_rays << ", runs: " << runs << std::endl;
    std::cout << "OpenMP threads: " << omp_get_max_threads() << std::endl;

    // Load mesh
    auto load_start = std::chrono::high_resolution_clock::now();
    ptm::MeshUtils mu;
    if (!mu.importTriangulatedMesh(mesh_file)) {
        std::cerr << "[error] Failed to import mesh." << std::endl;
        return 2;
    }
    auto mesh = mu.getMesh();
    auto load_end = std::chrono::high_resolution_clock::now();
    double load_ms = std::chrono::duration<double, std::milli>(load_end - load_start).count();
    std::cout << "Mesh loaded in " << load_ms << " ms: "
              << mesh->vertices.size() << " vertices, "
              << mesh->faces.size() << " faces." << std::endl;

    // Face geometry
    FaceGeom geom = buildFaceGeom(mesh);

    // Embree scene
    std::cout << "Building Embree scene..." << std::endl;
    auto embree_t0 = std::chrono::high_resolution_clock::now();
    RTCDevice device = rtcNewDevice(nullptr);
    RTCScene scene = createEmbreeScene(device, mesh);
    auto embree_t1 = std::chrono::high_resolution_clock::now();
    double embree_build_ms = std::chrono::duration<double, std::milli>(embree_t1 - embree_t0).count();
    std::cout << "Embree scene built in " << embree_build_ms << " ms." << std::endl;

    bool use_avx512 = hasAVX512();
    std::cout << "SIMD mode: " << (use_avx512 ? "AVX-512 (16-wide)" : "AVX2 (8-wide)") << std::endl;

#ifdef PTM_GPU_OPENCL
    std::cout << "Building GPU BVH (CPU-side, SAH)..." << std::endl;
    auto gpu_build_t0 = std::chrono::high_resolution_clock::now();
    gpu::opencl::BVHBuilder bvh_builder;
    gpu::opencl::BVHConfig bvh_cfg;
    bvh_cfg.verbose = false;
    bvh_builder.build(mesh->vertices, mesh->faces, bvh_cfg);
    auto gpu_build_t1 = std::chrono::high_resolution_clock::now();
    double gpu_bvh_ms = std::chrono::duration<double, std::milli>(gpu_build_t1 - gpu_build_t0).count();
    std::cout << "GPU BVH built in " << gpu_bvh_ms << " ms ("
              << bvh_builder.numNodes() << " nodes)." << std::endl;

    std::cout << "Initialising GPU ray tracer..." << std::endl;
    gpu::opencl::BVHRayTracer gpu_rt;
    if (!gpu_rt.initialize()) {
        std::cerr << "[error] GPU ray tracer init failed." << std::endl;
        return 3;
    }
    auto gpu_upload_t0 = std::chrono::high_resolution_clock::now();
    gpu_rt.uploadMesh(mesh->vertices, mesh->faces);
    gpu_rt.uploadBVH(bvh_builder);
    auto gpu_upload_t1 = std::chrono::high_resolution_clock::now();
    double gpu_upload_ms = std::chrono::duration<double, std::milli>(gpu_upload_t1 - gpu_upload_t0).count();
    std::cout << "GPU mesh+BVH uploaded in " << gpu_upload_ms << " ms." << std::endl;
#else
    std::cout << "[warn] Built without PTM_GPU_OPENCL; GPU path disabled." << std::endl;
#endif

    // Ray generation
    std::cout << "Generating " << num_rays << " realistic VF-style shadow rays..." << std::endl;
    auto ray_t0 = std::chrono::high_resolution_clock::now();
    std::vector<Ray> rays = generateVFRays(geom, num_rays, /*seed=*/42);
    auto ray_t1 = std::chrono::high_resolution_clock::now();
    double ray_ms = std::chrono::duration<double, std::milli>(ray_t1 - ray_t0).count();
    std::cout << "Generated " << rays.size() << " rays in " << ray_ms << " ms." << std::endl;
    if (rays.empty()) {
        std::cerr << "[error] No rays generated." << std::endl;
        rtcReleaseScene(scene);
        rtcReleaseDevice(device);
        return 4;
    }

    // Pre-build SoA buffers once for GPU (transfer time is measured per-run).
#ifdef PTM_GPU_OPENCL
    std::vector<float> gpu_org_x(rays.size()), gpu_org_y(rays.size()), gpu_org_z(rays.size());
    std::vector<float> gpu_dir_x(rays.size()), gpu_dir_y(rays.size()), gpu_dir_z(rays.size());
    std::vector<float> gpu_tmax(rays.size());
    for (size_t i = 0; i < rays.size(); ++i) {
        gpu_org_x[i] = static_cast<float>(rays[i].origin[0]);
        gpu_org_y[i] = static_cast<float>(rays[i].origin[1]);
        gpu_org_z[i] = static_cast<float>(rays[i].origin[2]);
        gpu_dir_x[i] = static_cast<float>(rays[i].dir[0]);
        gpu_dir_y[i] = static_cast<float>(rays[i].dir[1]);
        gpu_dir_z[i] = static_cast<float>(rays[i].dir[2]);
        gpu_tmax[i] = rays[i].tmax;
    }
#endif

    // Warm-up runs (not timed) so we measure steady state.
    std::vector<uint8_t> cpu_occ, gpu_occ;
    std::cout << "Warming up..." << std::endl;
    cpuVisibility(scene, rays, cpu_occ, use_avx512);
#ifdef PTM_GPU_OPENCL
    gpu_rt.traceShadowRays(gpu_org_x, gpu_org_y, gpu_org_z,
                           gpu_dir_x, gpu_dir_y, gpu_dir_z,
                           gpu_tmax, gpu_occ);
    double agreement = sampleAgree(cpu_occ, gpu_occ);
    std::cout << "Agreement (CPU vs GPU occluded bits): "
              << std::fixed << std::setprecision(4)
              << (agreement * 100.0) << "%" << std::endl;
#endif

    // Timed runs
    std::vector<double> cpu_times_ms, gpu_times_ms;
    for (int r = 0; r < runs; ++r) {
        auto t0 = std::chrono::high_resolution_clock::now();
        cpuVisibility(scene, rays, cpu_occ, use_avx512);
        auto t1 = std::chrono::high_resolution_clock::now();
        cpu_times_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());

#ifdef PTM_GPU_OPENCL
        auto g0 = std::chrono::high_resolution_clock::now();
        gpu_rt.traceShadowRays(gpu_org_x, gpu_org_y, gpu_org_z,
                               gpu_dir_x, gpu_dir_y, gpu_dir_z,
                               gpu_tmax, gpu_occ);
        auto g1 = std::chrono::high_resolution_clock::now();
        gpu_times_ms.push_back(std::chrono::duration<double, std::milli>(g1 - g0).count());
#endif
    }

    auto median = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };
    double cpu_ms = median(cpu_times_ms);
    double cpu_rps = rays.size() / (cpu_ms / 1000.0);

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\n==== Results (median of " << runs << " runs) ====" << std::endl;
    std::cout << "CPU Embree (" << (use_avx512 ? "rtcIntersect16" : "rtcIntersect8")
              << " + " << omp_get_max_threads() << " OpenMP threads):" << std::endl;
    std::cout << "  " << cpu_ms << " ms  ->  "
              << (cpu_rps / 1e6) << " M rays/sec" << std::endl;

#ifdef PTM_GPU_OPENCL
    double gpu_ms = median(gpu_times_ms);
    double gpu_rps = rays.size() / (gpu_ms / 1000.0);
    std::cout << "GPU OpenCL BVH traceShadowRays (incl. CPU->GPU transfer):" << std::endl;
    std::cout << "  " << gpu_ms << " ms  ->  "
              << (gpu_rps / 1e6) << " M rays/sec" << std::endl;
    const auto& stats = gpu_rt.getStats();
    double gpu_kernel_ms_per_run = (runs > 0) ? stats.kernel_time_ms / (runs + 1) : 0.0;  // +1 warmup
    double gpu_upload_ms_per_run = (runs > 0) ? stats.upload_time_ms / (runs + 1) : 0.0;
    std::cout << "  Breakdown (averaged): upload " << gpu_upload_ms_per_run
              << " ms, kernel " << gpu_kernel_ms_per_run << " ms." << std::endl;
    double speedup = cpu_ms / gpu_ms;
    std::cout << "\nGPU speedup vs CPU: " << speedup << "x "
              << (speedup >= 1.0 ? "(GPU faster)" : "(GPU slower)") << std::endl;
    // Also report kernel-only speedup (excluding CPU->GPU transfer), since
    // transfer overhead is amortisable if we keep rays resident.
    if (gpu_kernel_ms_per_run > 0) {
        double kernel_rps = rays.size() / (gpu_kernel_ms_per_run / 1000.0);
        double kernel_speedup = cpu_ms / gpu_kernel_ms_per_run;
        std::cout << "GPU kernel-only: " << gpu_kernel_ms_per_run << " ms -> "
                  << (kernel_rps / 1e6) << " M rays/sec ("
                  << kernel_speedup << "x vs CPU)" << std::endl;
    }
#endif

    rtcReleaseScene(scene);
    rtcReleaseDevice(device);
    return 0;
}
