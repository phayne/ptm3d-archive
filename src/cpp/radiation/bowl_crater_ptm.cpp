/**
 * @file bowl_crater_ptm.cpp
 * @brief Ingersoll bowl-crater validation driven by the PRODUCTION PTM3D
 *        radiation module (Embree ray-traced view factors + RadiositySolver),
 *        not a simplified concave-only reimplementation.
 *
 * For each depth/diameter ratio it builds a spherical-cap bowl embedded in a
 * square flat apron, computes occluded view factors with ViewFactorCalculator
 * (so the shadowed floor exchanges IR only with walls it can actually see),
 * ray-traces the Sun shadow at each solar elevation, iterates the per-face
 * radiative-equilibrium temperature using the production RadiositySolver, and
 * writes per-face temperatures + the floor T / Q_ind sweep as .npy files for the
 * comparison figure.
 *
 * Usage: bowl_crater_ptm <out_dir> [dd1 dd2 ...]
 */
#include "../illumination/embree_isa_report.h"  // one-time Embree ISA banner

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <array>
#include <string>
#include <cmath>
#include <memory>
#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <numeric>
#include <chrono>

#include <embree4/rtcore.h>

#include "../mesh/mesh_utils.h"
#include "radiation_config.h"
#include "view_factor.h"
#include "radiation_exchange.h"
#include "../constants/stefan_boltzmann.h"  // ptm::kStefanBoltzmann (the single definition)

namespace {

constexpr double PI = 3.14159265358979323846;
// The Ingersoll validation gate must score the engine with the engine's own
// constant; it used to carry a sixth value that no solver ever used.
constexpr double SIGMA = ptm::kStefanBoltzmann;
constexpr double SOLAR = 1361.0;
constexpr double ALBEDO = 0.12;
constexpr double EMISS = 0.95;

// ---- minimal .npy writer (little-endian, C order) ------------------------
template <typename T>
void write_npy(const std::string& path, const std::vector<T>& data,
               const std::vector<size_t>& shape, const char* dtype) {
    std::ofstream f(path, std::ios::binary);
    std::string sh = "(";
    for (size_t i = 0; i < shape.size(); ++i)
        sh += std::to_string(shape[i]) + (shape.size() == 1 ? "," : (i + 1 < shape.size() ? ", " : ""));
    sh += ")";
    std::string hdr = "{'descr': '" + std::string(dtype) +
                      "', 'fortran_order': False, 'shape': " + sh + ", }";
    size_t base = 10 + hdr.size() + 1;          // magic(8)+ver? -> 6+2 len +hdr+\n
    size_t total = 10 + hdr.size() + 1;
    size_t pad = (64 - (total % 64)) % 64;
    hdr.append(pad, ' ');
    hdr += '\n';
    uint16_t hlen = static_cast<uint16_t>(hdr.size());
    const char magic[8] = {'\x93', 'N', 'U', 'M', 'P', 'Y', '\x01', '\x00'};
    f.write(magic, 8);
    f.write(reinterpret_cast<const char*>(&hlen), 2);
    f.write(hdr.data(), hdr.size());
    f.write(reinterpret_cast<const char*>(data.data()),
            static_cast<std::streamsize>(data.size() * sizeof(T)));
    (void)base;
}

// ---- bowl + square flat apron mesh ---------------------------------------
struct Built {
    std::shared_ptr<ptm::Mesh> mesh;
    std::vector<double> areas;
    std::vector<int> mask;       // 0 wall, 1 floor, 2 flat
    double d, r_rim, R, z_c;
};

Built build_mesh(double dd, double diameter, int n_rings, int n_az,
                 double flat_extent, int n_flat_rings) {
    double d = dd * diameter, r_rim = diameter / 2.0;
    double R = (r_rim * r_rim + d * d) / (2.0 * d);
    double z_c = R - d, theta_max = std::asin(r_rim / R);

    std::vector<std::array<double, 3>> V;
    V.push_back({{0.0, 0.0, -d}});
    for (int k = 1; k <= n_rings; ++k) {
        double th = theta_max * k / n_rings;
        double rho = R * std::sin(th), z = z_c - R * std::cos(th);
        for (int j = 0; j < n_az; ++j) {
            double phi = 2.0 * PI * j / n_az;
            V.push_back({{rho * std::cos(phi), rho * std::sin(phi), z}});
        }
    }
    int rim0 = 1 + (n_rings - 1) * n_az;
    double H = flat_extent * r_rim;
    std::vector<int> flat_starts;
    for (int s = 1; s <= n_flat_rings; ++s) {
        double t = double(s) / n_flat_rings;
        flat_starts.push_back(int(V.size()));
        for (int j = 0; j < n_az; ++j) {
            double phi = 2.0 * PI * j / n_az, c = std::cos(phi), sn = std::sin(phi);
            double sc = H / std::max(std::abs(c), std::abs(sn));
            V.push_back({{(1 - t) * r_rim * c + t * sc * c,
                          (1 - t) * r_rim * sn + t * sc * sn, 0.0}});
        }
    }

    std::vector<std::array<int, 3>> Fc;
    for (int j = 0; j < n_az; ++j)
        Fc.push_back({{0, 1 + j, 1 + (j + 1) % n_az}});
    for (int k = 1; k < n_rings; ++k) {
        int bi = 1 + (k - 1) * n_az, bo = 1 + k * n_az;
        for (int j = 0; j < n_az; ++j) {
            int j1 = (j + 1) % n_az;
            Fc.push_back({{bi + j, bo + j, bo + j1}});
            Fc.push_back({{bi + j, bo + j1, bi + j1}});
        }
    }
    int n_bowl_faces = int(Fc.size());
    std::vector<int> ring_starts; ring_starts.push_back(rim0);
    for (int s : flat_starts) ring_starts.push_back(s);
    for (size_t r = 0; r + 1 < ring_starts.size(); ++r) {
        int a0 = ring_starts[r], b0 = ring_starts[r + 1];
        for (int j = 0; j < n_az; ++j) {
            int j1 = (j + 1) % n_az;
            Fc.push_back({{a0 + j, b0 + j, b0 + j1}});
            Fc.push_back({{a0 + j, b0 + j1, a0 + j1}});
        }
    }

    auto mesh = std::make_shared<ptm::Mesh>();
    mesh->vertices = V;
    mesh->faces = Fc;
    size_t nf = Fc.size();
    mesh->face_centers.resize(nf);
    mesh->face_normals.resize(nf);
    std::vector<double> areas(nf);
    std::vector<int> mask(nf);
    std::array<double, 3> ctr_sphere = {{0, 0, z_c}};
    for (size_t i = 0; i < nf; ++i) {
        const auto& v0 = V[Fc[i][0]]; const auto& v1 = V[Fc[i][1]]; const auto& v2 = V[Fc[i][2]];
        std::array<double, 3> c = {{(v0[0] + v1[0] + v2[0]) / 3, (v0[1] + v1[1] + v2[1]) / 3,
                                    (v0[2] + v1[2] + v2[2]) / 3}};
        mesh->face_centers[i] = c;
        std::array<double, 3> e1 = {{v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]}};
        std::array<double, 3> e2 = {{v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]}};
        std::array<double, 3> nrm = {{e1[1] * e2[2] - e1[2] * e2[1],
                                      e1[2] * e2[0] - e1[0] * e2[2],
                                      e1[0] * e2[1] - e1[1] * e2[0]}};
        double len = std::sqrt(nrm[0] * nrm[0] + nrm[1] * nrm[1] + nrm[2] * nrm[2]);
        areas[i] = 0.5 * len;
        for (int k = 0; k < 3; ++k) nrm[k] /= std::max(len, 1e-30);
        bool flat = int(i) >= n_bowl_faces;
        if (flat) {
            if (nrm[2] < 0) for (int k = 0; k < 3; ++k) nrm[k] = -nrm[k];   // +z up
            mask[i] = 2;
        } else {
            // inward (toward open bowl == toward sphere center above)
            std::array<double, 3> tc = {{ctr_sphere[0] - c[0], ctr_sphere[1] - c[1], ctr_sphere[2] - c[2]}};
            if (nrm[0] * tc[0] + nrm[1] * tc[1] + nrm[2] * tc[2] < 0)
                for (int k = 0; k < 3; ++k) nrm[k] = -nrm[k];
            double rr = std::hypot(c[0], c[1]);
            mask[i] = (rr < 0.18 * r_rim) ? 1 : 0;
        }
        mesh->face_normals[i] = nrm;
    }
    return {mesh, areas, mask, d, r_rim, R, z_c};
}

// ---- Embree scene --------------------------------------------------------
RTCScene make_scene(RTCDevice dev, const ptm::Mesh& m, bool with_occluders = true) {
    RTCScene scene = rtcNewScene(dev);
    if (!with_occluders) {
        // Empty scene: every occlusion ray misses -> assume-all-visible. For a
        // fully-concave bowl this is the geometrically ideal (no self-occlusion).
        rtcCommitScene(scene);
        return scene;
    }
    RTCGeometry g = rtcNewGeometry(dev, RTC_GEOMETRY_TYPE_TRIANGLE);
    size_t nv = m.vertices.size(), nf = m.faces.size();
    float* vb = static_cast<float*>(rtcSetNewGeometryBuffer(
        g, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, 3 * sizeof(float), nv));
    for (size_t i = 0; i < nv; ++i) {
        vb[3 * i] = float(m.vertices[i][0]);
        vb[3 * i + 1] = float(m.vertices[i][1]);
        vb[3 * i + 2] = float(m.vertices[i][2]);
    }
    unsigned* ib = static_cast<unsigned*>(rtcSetNewGeometryBuffer(
        g, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, 3 * sizeof(unsigned), nf));
    for (size_t i = 0; i < nf; ++i) {
        ib[3 * i] = unsigned(m.faces[i][0]);
        ib[3 * i + 1] = unsigned(m.faces[i][1]);
        ib[3 * i + 2] = unsigned(m.faces[i][2]);
    }
    rtcCommitGeometry(g);
    rtcAttachGeometry(scene, g);
    rtcReleaseGeometry(g);
    rtcCommitScene(scene);
    return scene;
}

// shadow ray: true if the Sun is blocked from this face
bool shadowed(RTCScene scene, const std::array<double, 3>& c,
              const std::array<double, 3>& n, const std::array<double, 3>& s, double eps) {
    RTCRay ray;
    ray.org_x = float(c[0] + eps * n[0]);
    ray.org_y = float(c[1] + eps * n[1]);
    ray.org_z = float(c[2] + eps * n[2]);
    ray.dir_x = float(s[0]); ray.dir_y = float(s[1]); ray.dir_z = float(s[2]);
    ray.tnear = 0.0f; ray.tfar = 1.0e30f;
    ray.mask = 0xFFFFFFFFu; ray.flags = 0; ray.time = 0.0f;
    RTCOccludedArguments oa; rtcInitOccludedArguments(&oa);
    rtcOccluded1(scene, &ray, &oa);
    return ray.tfar < 0.0f;
}

// ---- measurement-harness helpers (env-gated; production untouched) -------
// Environment knobs (all optional; defaults reproduce prior behavior):
//   PTM_BOWL_NRINGS / PTM_BOWL_NAZ / PTM_BOWL_NFLAT  : mesh resolution
//   PTM_BOWL_VISTHRESH                               : visibility_threshold
//   PTM_BOWL_WINDOW                                  : nearfield_window_factor
//                     (0 => force pure M1/centroid; huge => force pure M2/subdivision)
//                     Replaced PTM_BOWL_NSTHRESH, which set the RETIRED
//                     adaptive_ns_threshold. The two are different quantities;
//                     setting the old name is now a hard error, not a silent
//                     no-op, because a benchmark that quietly ran the default
//                     window would be reported as one that ran the requested one.
//   PTM_BOWL_NOOCCLUSION=1                           : build an EMPTY Embree scene
//                     (assume-all-visible; ideal for a fully-concave bowl)
//   PTM_BOWL_RECIP=1                                 : symmetrize A_i F_ij = A_j F_ji
//   PTM_BOWL_VFONLY=1                                : F_sky stats only, skip beta sweep + npy
//   PTM_BOWL_KPROBE=m1|m2_legacy|refined             : compute floor F_sky directly from
//                     geometry with a chosen kernel + identical occlusion (bypasses
//                     ViewFactorCalculator) so the KERNEL is isolated cleanly.
//                     ⚠ "m2_legacy" is the RETIRED one-sided (receiver-only)
//                     subdivision rule. Production's near-field M2 has been the
//                     symmetric double-area quadrature since 2026-08-23, which
//                     here is "refined". The spelling "m2" is still accepted and
//                     still means the legacy kernel, but it warns, and the CSV
//                     always records "m2_onesided_legacy" so a probe row can
//                     never be confused with a production "m2" row.
//   PTM_BOWL_QUADORDER=<m>                           : radiation.nearfield_quadrature_order
//                     for the production path (default = the shipped value). With
//                     PTM_BOWL_WINDOW=1e12 every pair goes to M2, so vf_seconds
//                     then measures the near-field kernel alone.
//   PTM_BOWL_RESULTS=<path.csv>                       : append one result row (default
//                     output/bowl_conv/results.csv)
// Moon3D R1 benchmark (C3) knobs -- unset = prior behavior, byte for byte:
//   PTM_BOWL_WRITE_PLY=<prefix>  : write the mesh as <prefix>_dd<dd>.ply (binary, double
//                     xyz, winding = the harness's inward/up normals so mc_ref_pp's
//                     back-face cull sees front faces) + <prefix>_dd<dd>_mask.npy, then
//                     SKIP the view factors and the solve for that d/D. The PLY is the
//                     input for an offline mc_ref_pp sweep + assemble_vf_haworth.py.
//   PTM_BOWL_VF_CSR=<path.csr>   : LOAD the view factors from an assembled CSR (the
//                     production estimator) instead of building the exact dense
//                     matrix. Exactly one d/D per run; the CSR must carry this mesh's
//                     area hash (same check as PTM_HMATRIX_VF_REPLACE) and pass its
//                     invariants (no row > 1, no negative entry), else FATAL.
//   PTM_BOWL_ESTIMATOR=exact|sparse_production : `exact` (default) is the gate's dense
//                     matrix (max_neighbors = nf). `sparse_production` builds with the
//                     SHIPPED RadiationConfig defaults (20 / 500 / 95%, adaptive), i.e.
//                     what a run with radiation enabled and no VF file would build.
//                     PTM_BOWL_VISTHRESH / PTM_BOWL_WINDOW are not applied there.
//   PTM_BOWL_WRITE_CSR=<path.csr>: after BUILDING F (exact or sparse_production), write it
//                     in the assembler's CSR format (float32 values, area hash), so the
//                     PTM_BOWL_VF_CSR load path can be round-trip tested against the gate.
//   PTM_BOWL_DUMP=<prefix>       : with PTM_BOWL_RADIOSITY=1, write per-face <prefix>_T.npy,
//                     _qind.npy (+ _qvis.npy/_qir.npy when two-band), _mask.npy and
//                     _areas.npy at the converged state, for per-face accuracy metrics.
const char* env_c(const char* k) { return std::getenv(k); }
int    env_int(const char* k, int d)    { const char* v = env_c(k); return v ? std::atoi(v) : d; }
double env_dbl(const char* k, double d) { const char* v = env_c(k); return v ? std::atof(v) : d; }
bool   env_flag(const char* k)          { const char* v = env_c(k); return v && v[0] && v[0] != '0'; }
std::string env_s(const char* k, const std::string& d) { const char* v = env_c(k); return v ? std::string(v) : d; }

// Centroid-to-centroid visibility, mirroring ViewFactorCalculator::isVisible
// offset logic (relative+absolute offsets, clamped to 10% of distance), but
// specialized to the harness Embree scene. Returns true if p1 sees p2.
bool ray_visible(RTCScene scene, const std::array<double, 3>& p1,
                 const std::array<double, 3>& p2) {
    std::array<double, 3> d = {{p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]}};
    double dist = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    if (dist < 1e-10) return true;
    for (int k = 0; k < 3; ++k) d[k] /= dist;
    double relative_offset = dist * 0.001, absolute_offset = 0.5;
    double offset = std::min(std::max(relative_offset, absolute_offset), dist * 0.1);
    double end_margin = std::min(std::max(dist * 0.001, absolute_offset), dist * 0.1);
    double tfar = dist - offset - end_margin;
    if (tfar < 0.1) return true;   // matches production isVisible (view_factor.cpp:1460)
    RTCRay ray;
    ray.org_x = float(p1[0] + offset * d[0]);
    ray.org_y = float(p1[1] + offset * d[1]);
    ray.org_z = float(p1[2] + offset * d[2]);
    ray.dir_x = float(d[0]); ray.dir_y = float(d[1]); ray.dir_z = float(d[2]);
    ray.tnear = 0.0f; ray.tfar = float(tfar);
    ray.mask = 0xFFFFFFFFu; ray.flags = 0; ray.time = 0.0f;
    RTCOccludedArguments oa; rtcInitOccludedArguments(&oa);
    rtcOccluded1(scene, &ray, &oa);
    return !(ray.tfar < 0.0f);
}

// Sub-triangle centroids of triangle (p0,p1,p2), m^2 congruent sub-triangles.
// Same barycentric scheme as HMatrixViewFactor::computeViewFactorRefined.
void tri_subcentroids(const std::array<double, 3>& p0, const std::array<double, 3>& p1,
                      const std::array<double, 3>& p2, int m,
                      std::vector<std::array<double, 3>>& out) {
    out.clear();
    if (m < 1) m = 1;
    const std::array<double, 3> e1 = {{p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]}};
    const std::array<double, 3> e2 = {{p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]}};
    auto at = [&](double u, double v) {
        return std::array<double, 3>{{p0[0] + u * e1[0] + v * e2[0],
                                      p0[1] + u * e1[1] + v * e2[1],
                                      p0[2] + u * e1[2] + v * e2[2]}};
    };
    const double inv3m = 1.0 / (3.0 * m);
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < m - i; ++j) {
            out.push_back(at((3.0 * i + 1.0) * inv3m, (3.0 * j + 1.0) * inv3m));
            if (j < m - 1 - i)
                out.push_back(at((3.0 * i + 2.0) * inv3m, (3.0 * j + 2.0) * inv3m));
        }
}

// Harness-local view-factor kernels (double precision), used by the KPROBE mode
// so the KERNEL can be isolated with identical occlusion/pair-set.
//   vf_m1        mirrors production M1 (centroid) exactly.
//   vf_refined   mirrors production near-field M2 as of 2026-08-23 (symmetric
//                double-area sub-centroid quadrature) up to the sub-centroid
//                COUNT: production takes m from radiation.nearfield_quadrature_
//                order (default 3), this harness picks m adaptively from
//                proximity, as HMatrixViewFactor::computeViewFactorRefined does.
//   vf_m2_legacy_onesided  is the RETIRED receiver-only rule. It no longer
//                exists anywhere in production; it is kept so the pre-fix
//                near field can still be reproduced for attribution.
double vf_m1(const std::array<double, 3>& ci, const std::array<double, 3>& ni,
             const std::array<double, 3>& cj, const std::array<double, 3>& nj, double Aj) {
    std::array<double, 3> rv = {{cj[0] - ci[0], cj[1] - ci[1], cj[2] - ci[2]}};
    double r2 = rv[0] * rv[0] + rv[1] * rv[1] + rv[2] * rv[2];
    if (r2 < 1e-20) return 0.0;
    double r = std::sqrt(r2);
    double cos_i = (ni[0] * rv[0] + ni[1] * rv[1] + ni[2] * rv[2]) / r;
    double cos_j = -(nj[0] * rv[0] + nj[1] * rv[1] + nj[2] * rv[2]) / r;
    if (cos_i < 1e-6 || cos_j < 1e-6) return 0.0;
    return (Aj / M_PI) * cos_i * cos_j / r2;
}
// RETIRED M2: source = point at the centroid of i, integrate over subdivided
// receiver j. Removed from production 2026-08-23 (view_factor.cpp) -- it read
// -21.7% mass-weighted on edge/vertex-touching pairs. Kept here for attribution
// only; do not reintroduce it as "M2".
double vf_m2_legacy_onesided(const std::array<double, 3>& ci, const std::array<double, 3>& ni,
             const std::array<double, 3>& nj, double Aj,
             const std::vector<std::array<double, 3>>& sub_j) {
    double s = 0.0; int n = 0;
    for (const auto& pj : sub_j) {
        std::array<double, 3> rv = {{pj[0] - ci[0], pj[1] - ci[1], pj[2] - ci[2]}};
        double r2 = rv[0] * rv[0] + rv[1] * rv[1] + rv[2] * rv[2];
        if (r2 < 1e-20) continue;
        double r = std::sqrt(r2);
        double cos_i = (ni[0] * rv[0] + ni[1] * rv[1] + ni[2] * rv[2]) / r;
        double cos_j = -(nj[0] * rv[0] + nj[1] * rv[1] + nj[2] * rv[2]) / r;
        if (cos_i < 1e-6 || cos_j < 1e-6) continue;
        s += cos_i * cos_j / r2; ++n;
    }
    if (n == 0) return 0.0;
    return (Aj / M_PI) * s / double(sub_j.size());
}
// Refined: area-to-area double quadrature (both faces subdivided).
double vf_refined(const std::array<double, 3>& ni, const std::array<double, 3>& nj,
                  double Ai, double Aj,
                  const std::vector<std::array<double, 3>>& sub_i,
                  const std::vector<std::array<double, 3>>& sub_j) {
    if (sub_i.empty() || sub_j.empty()) return 0.0;
    double wi = Ai / double(sub_i.size()), wj = Aj / double(sub_j.size());
    double sum = 0.0;
    for (const auto& p : sub_i)
        for (const auto& q : sub_j) {
            std::array<double, 3> rv = {{q[0] - p[0], q[1] - p[1], q[2] - p[2]}};
            double r2 = rv[0] * rv[0] + rv[1] * rv[1] + rv[2] * rv[2];
            if (r2 < 1e-18) continue;
            double rinv = 1.0 / std::sqrt(r2);
            double ci = (ni[0] * rv[0] + ni[1] * rv[1] + ni[2] * rv[2]) * rinv;
            double cj = -(nj[0] * rv[0] + nj[1] * rv[1] + nj[2] * rv[2]) * rinv;
            if (ci > 0.0 && cj > 0.0) sum += ci * cj / (M_PI * r2) * wi * wj;
        }
    return sum / Ai;
}

// ---- R1 benchmark helpers --------------------------------------------------
// Same mesh-identity hash as PTM_HMATRIX_VF_REPLACE (radiation_exchange.cpp) and
// assemble_vf_haworth.py's GEOMHASH.
double area_hash(const std::vector<double>& areas) {
    double h = 0.0;
    for (size_t i = 0; i < areas.size(); ++i)
        h += areas[i] + static_cast<double>(i + 1) * areas[i] * 1e-9;
    return h;
}

// Load an assembled CSR (int32 n, float64 geomhash, int64 indptr[n+1], int64 nnz,
// int32 cols[nnz], float32 data[nnz]) into F. FATAL on any mismatch: a bowl
// accuracy number computed on a foreign or corrupt matrix must never be printed.
void load_csr_or_die(const std::string& path, const std::vector<double>& areas,
                     radiation::ViewFactorMatrix& F) {
    std::ifstream f(path, std::ios::binary);
    int32_t n = 0; double gh = 0.0;
    if (f) { f.read(reinterpret_cast<char*>(&n), 4); f.read(reinterpret_cast<char*>(&gh), 8); }
    if (!f) { std::cerr << "FATAL: PTM_BOWL_VF_CSR: cannot read " << path << "\n"; std::exit(3); }
    if (static_cast<size_t>(n) != areas.size()) {
        std::cerr << "FATAL: PTM_BOWL_VF_CSR: CSR has " << n << " faces, mesh " << areas.size()
                  << " (wrong mesh or wrong PTM_BOWL_NRINGS/NAZ/NFLAT).\n";
        std::exit(3);
    }
    const double mh = area_hash(areas);
    if (std::fabs(gh - mh) > 1e-3 * std::fabs(mh) + 1e-9) {
        std::cerr << "FATAL: PTM_BOWL_VF_CSR: geometry hash " << gh << " != mesh " << mh
                  << ". The CSR was built for a different mesh.\n";
        std::exit(3);
    }
    std::vector<int64_t> ip(static_cast<size_t>(n) + 1);
    f.read(reinterpret_cast<char*>(ip.data()), static_cast<std::streamsize>(ip.size() * 8));
    int64_t nnz = 0;
    f.read(reinterpret_cast<char*>(&nnz), 8);
    std::vector<int32_t> c32(static_cast<size_t>(nnz));
    std::vector<float> v32(static_cast<size_t>(nnz));
    f.read(reinterpret_cast<char*>(c32.data()), static_cast<std::streamsize>(nnz * 4));
    f.read(reinterpret_cast<char*>(v32.data()), static_cast<std::streamsize>(nnz * 4));
    if (!f || ip.back() != nnz) {
        std::cerr << "FATAL: PTM_BOWL_VF_CSR: truncated or inconsistent CSR " << path << "\n";
        std::exit(3);
    }
    size_t nover = 0; float vmin = 0.0f;
    for (int32_t i = 0; i < n; ++i) {
        double rs = 0.0;
        for (int64_t k = ip[i]; k < ip[i + 1]; ++k) { rs += v32[k]; vmin = std::min(vmin, v32[k]); }
        if (rs > 1.0 + 1e-6) ++nover;
    }
    if (nover > 0 || vmin < -1e-9f) {
        std::cerr << "FATAL: PTM_BOWL_VF_CSR: invariants violated (rows>1=" << nover
                  << ", min entry=" << vmin << ").\n";
        std::exit(3);
    }
    std::vector<int> cols(c32.begin(), c32.end());
    std::vector<double> vals(v32.begin(), v32.end());
    F.setFromCSR(static_cast<size_t>(n), std::move(ip), std::move(cols), std::move(vals));
    std::cout << "  [PTM_BOWL_VF_CSR] loaded " << path << ": N=" << n << " nnz=" << nnz
              << ", mesh hash OK" << std::endl;
}

void write_csr_or_die(const std::string& path, const std::vector<double>& areas,
                      const radiation::ViewFactorMatrix& F) {
    std::ofstream f(path, std::ios::binary);
    const int32_t n = static_cast<int32_t>(F.numRows());
    const double gh = area_hash(areas);
    const auto& ip = F.rowPointers();
    const int64_t nnz = static_cast<int64_t>(F.numNonZeros());
    std::vector<int32_t> c32(F.colIndices().begin(), F.colIndices().end());
    std::vector<float> v32(F.values().begin(), F.values().end());
    f.write(reinterpret_cast<const char*>(&n), 4);
    f.write(reinterpret_cast<const char*>(&gh), 8);
    f.write(reinterpret_cast<const char*>(ip.data()), static_cast<std::streamsize>(ip.size() * 8));
    f.write(reinterpret_cast<const char*>(&nnz), 8);
    f.write(reinterpret_cast<const char*>(c32.data()), static_cast<std::streamsize>(nnz * 4));
    f.write(reinterpret_cast<const char*>(v32.data()), static_cast<std::streamsize>(nnz * 4));
    if (!f) { std::cerr << "FATAL: cannot write " << path << "\n"; std::exit(3); }
    std::cout << "  [PTM_BOWL_WRITE_CSR] wrote " << path << " (N=" << n << " nnz=" << nnz
              << ", float32 values)" << std::endl;
}

// Binary PLY (double xyz, uchar+int32 faces). Each face is wound so that its
// geometric normal agrees with the harness normal (inward on the bowl, +z on the
// apron): mc_ref_pp uses the RAW winding normal and culls back-face hits.
void write_bowl_ply(const std::string& path, const ptm::Mesh& m) {
    std::ofstream f(path, std::ios::binary);
    f << "ply\nformat binary_little_endian 1.0\nelement vertex " << m.vertices.size()
      << "\nproperty double x\nproperty double y\nproperty double z\nelement face "
      << m.faces.size() << "\nproperty list uchar int vertex_indices\nend_header\n";
    for (const auto& v : m.vertices) f.write(reinterpret_cast<const char*>(v.data()), 24);
    size_t flipped = 0;
    for (size_t i = 0; i < m.faces.size(); ++i) {
        auto fc = m.faces[i];
        const auto& a = m.vertices[fc[0]]; const auto& b = m.vertices[fc[1]];
        const auto& c = m.vertices[fc[2]];
        const double e1[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
        const double e2[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
        const double cr[3] = {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2],
                              e1[0] * e2[1] - e1[1] * e2[0]};
        const auto& nm = m.face_normals[i];
        if (cr[0] * nm[0] + cr[1] * nm[1] + cr[2] * nm[2] < 0.0) { std::swap(fc[1], fc[2]); ++flipped; }
        const unsigned char three = 3;
        f.write(reinterpret_cast<const char*>(&three), 1);
        const int32_t idx[3] = {fc[0], fc[1], fc[2]};
        f.write(reinterpret_cast<const char*>(idx), 12);
    }
    if (!f) { std::cerr << "FATAL: cannot write " << path << "\n"; std::exit(3); }
    std::cout << "  [PTM_BOWL_WRITE_PLY] wrote " << path << " (" << m.faces.size() << " faces, "
              << flipped << " rewound to match the harness normals)" << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
    std::string out_dir = (argc > 1) ? argv[1] : "output/bowl_ptm";
    std::vector<double> dds;
    for (int i = 2; i < argc; ++i) dds.push_back(std::atof(argv[i]));
    if (dds.empty()) dds = {0.01, 0.1, 0.2};

    // ~4x finer (n_rings*n_az). The exact dense view factors for a fully-concave
    // bowl have ~0.5*nf^2 non-zeros: the CSR uses int32 indices (overflow >65k
    // faces) and assembly cost balloons, so exact-dense VF caps near here. The
    // 4-sample shadow anti-aliasing (in solve_T) is what sharpens the terminator.
    // Resolution is env-overridable for the convergence study (defaults unchanged).
    const int n_rings = env_int("PTM_BOWL_NRINGS", 72);
    const int n_az = env_int("PTM_BOWL_NAZ", 192);
    const int n_flat_rings = env_int("PTM_BOWL_NFLAT", 14);
    const double diameter = 1.0, flat_extent = 1.7;

    // Measurement-harness knobs (env). Default reproduces prior behavior.
    const double vis_thresh = env_dbl("PTM_BOWL_VISTHRESH", 5e-10);
    if (env_c("PTM_BOWL_NSTHRESH")) {
        std::fprintf(stderr,
                     "FATAL: PTM_BOWL_NSTHRESH was removed with adaptive_ns_threshold "
                     "(2026-08-23). The near/far switch is now r < w (sqrt(A_i) + "
                     "sqrt(A_j)); use PTM_BOWL_WINDOW=w. The old default 5 is w = 2.5 "
                     "on an equal-area pair, and nothing at all on an unequal one.\n");
        std::exit(2);
    }
    const double window_factor = env_dbl("PTM_BOWL_WINDOW", 4.0);
    const bool no_occ = env_flag("PTM_BOWL_NOOCCLUSION");
    const bool recip = env_flag("PTM_BOWL_RECIP");
    const bool vf_only = env_flag("PTM_BOWL_VFONLY");
    const std::string kprobe = env_s("PTM_BOWL_KPROBE", "");   // "", m1, m2_legacy, refined
    const std::string results_path = env_s("PTM_BOWL_RESULTS", out_dir + "/results.csv");
    // Near-field quadrature order for the production path. Default = whatever
    // RadiationConfig ships, so an unset variable reproduces prior behavior.
    const int quad_order = env_int("PTM_BOWL_QUADORDER",
                                   radiation::RadiationConfig{}.nearfield_quadrature_order);
    if (quad_order < 1 || quad_order > radiation::kMaxConfigNearFieldQuadOrder) {
        std::cerr << "FATAL: PTM_BOWL_QUADORDER=" << quad_order << " is outside 1.."
                  << radiation::kMaxConfigNearFieldQuadOrder << "." << std::endl;
        return 1;
    }

    // R1 benchmark knobs (see the header). Validated up front so a typo FATALs
    // before any work instead of silently running the default estimator.
    const std::string write_ply = env_s("PTM_BOWL_WRITE_PLY", "");
    const std::string vf_csr = env_s("PTM_BOWL_VF_CSR", "");
    const std::string estimator = env_s("PTM_BOWL_ESTIMATOR", "exact");
    const std::string dump_prefix = env_s("PTM_BOWL_DUMP", "");
    const std::string write_csr = env_s("PTM_BOWL_WRITE_CSR", "");
    if (!write_csr.empty() && (!vf_csr.empty() || !kprobe.empty() || dds.size() != 1)) {
        std::cerr << "FATAL: PTM_BOWL_WRITE_CSR needs a BUILT matrix (no PTM_BOWL_VF_CSR / "
                     "PTM_BOWL_KPROBE) and exactly one d/D." << std::endl;
        return 1;
    }
    if (estimator != "exact" && estimator != "sparse_production") {
        std::cerr << "FATAL: PTM_BOWL_ESTIMATOR=" << estimator
                  << " (expected exact | sparse_production)." << std::endl;
        return 1;
    }
    if (!vf_csr.empty() && (env_c("PTM_BOWL_ESTIMATOR") || !kprobe.empty())) {
        std::cerr << "FATAL: PTM_BOWL_VF_CSR loads the matrix; it cannot be combined with "
                     "PTM_BOWL_ESTIMATOR or PTM_BOWL_KPROBE." << std::endl;
        return 1;
    }
    if (!vf_csr.empty() && dds.size() != 1) {
        std::cerr << "FATAL: PTM_BOWL_VF_CSR is one mesh; pass exactly one d/D (got "
                  << dds.size() << ")." << std::endl;
        return 1;
    }
    if (estimator == "sparse_production" && (env_c("PTM_BOWL_VISTHRESH") || env_c("PTM_BOWL_WINDOW")))
        std::cerr << "[bowl] WARNING: PTM_BOWL_ESTIMATOR=sparse_production uses the shipped "
                     "RadiationConfig; PTM_BOWL_VISTHRESH / PTM_BOWL_WINDOW are NOT applied."
                  << std::endl;
    if (!dump_prefix.empty() && !env_flag("PTM_BOWL_RADIOSITY"))
        std::cerr << "[bowl] WARNING: PTM_BOWL_DUMP is written only in PTM_BOWL_RADIOSITY=1 mode; "
                     "nothing will be dumped." << std::endl;

    // CSV header (only if the file does not yet exist).
    {
        std::ifstream chk(results_path);
        if (!chk.good()) {
            std::ofstream h(results_path, std::ios::app);
            h << "mode,dd,n_rings,n_az,nf,n_floor,visthresh,windowfactor,occ,recip,kernel,"
                 "analytic,mean_aw,mean,min,max,std,mean_bias_pct,peak_dev_pct,vf_seconds\n";
        }
    }

    int idx = 0;
    for (double dd : dds) {
        std::cout << "\n=== PTM3D bowl crater d/D=" << dd << " ===" << std::endl;
        Built b = build_mesh(dd, diameter, n_rings, n_az, flat_extent, n_flat_rings);
        size_t nf = b.mesh->faces.size();
        double mesh_eps = 1e-4 * b.r_rim;

        if (!write_ply.empty()) {
            char tag[64];
            std::snprintf(tag, sizeof(tag), "_dd%.4g", dd);
            write_bowl_ply(write_ply + tag + ".ply", *b.mesh);
            write_npy(write_ply + tag + "_mask.npy", b.mask, {nf}, "<i4");
            std::cout << "  faces=" << nf << "  (PTM_BOWL_WRITE_PLY: mesh only, no VF/solve)"
                      << std::endl;
            ++idx;
            continue;
        }

        RTCDevice dev = rtcNewDevice(nullptr);
        // The Ingersoll gate is exactly where someone checks a correctness or
        // performance claim, so it must say which Embree it used too.
        ptm::reportEmbreeIsaOnce(dev);
        RTCScene scene = make_scene(dev, *b.mesh, !no_occ);

        double alpha = std::atan(0.5 / dd);
        double fsky_an = std::sin(alpha) * std::sin(alpha);
        double bcrit = std::atan(2.0 * dd) * 180.0 / PI;

        // Per-floor-face sky view factor (1 - terrain row sum). Pure geometry.
        std::vector<double> fsky_face;   // per floor face
        std::vector<double> fsky_area;   // matching floor-face areas
        std::string kernel_label;
        double vf_seconds = 0.0;
        radiation::ViewFactorMatrix F;   // production matrix (empty in kprobe mode)
        std::vector<double> rsums;
        auto t0 = std::chrono::high_resolution_clock::now();

        if (kprobe.empty() && !vf_csr.empty()) {
            // ---- R1: externally assembled CSR (the production MC estimator) ----
            load_csr_or_die(vf_csr, b.areas, F);
            rsums = F.rowSums();
            std::vector<double> col_wsum;
            if (recip) F.matvecTranspose(b.areas, col_wsum);
            for (size_t i = 0; i < nf; ++i) {
                if (b.mask[i] != 1) continue;
                double rs = rsums[i];
                if (recip) rs = 0.5 * rs + 0.5 * col_wsum[i] / std::max(b.areas[i], 1e-30);
                fsky_face.push_back(1.0 - rs);
                fsky_area.push_back(b.areas[i]);
            }
            kernel_label = "csr";
        } else if (kprobe.empty() && estimator == "sparse_production") {
            // ---- R1: the sparse point-kernel builder at SHIPPED defaults ----
            radiation::RadiationConfig cfg;   // 20 / 500 / 95%, adaptive, as shipped
            cfg.enabled = true;
            cfg.verbose = false;
            radiation::ViewFactorCalculator calc(b.mesh, dev, scene, cfg);
            F = calc.computeAll();
            rsums = F.rowSums();
            std::vector<double> col_wsum;
            if (recip) F.matvecTranspose(b.areas, col_wsum);
            for (size_t i = 0; i < nf; ++i) {
                if (b.mask[i] != 1) continue;
                double rs = rsums[i];
                if (recip) rs = 0.5 * rs + 0.5 * col_wsum[i] / std::max(b.areas[i], 1e-30);
                fsky_face.push_back(1.0 - rs);
                fsky_area.push_back(b.areas[i]);
            }
            kernel_label = "sparse_production";
            if (!write_csr.empty()) write_csr_or_die(write_csr, b.areas, F);
        } else if (kprobe.empty()) {
            // ---- production path: ViewFactorCalculator.computeAll() ----
            radiation::RadiationConfig cfg;
            cfg.enabled = true;
            cfg.verbose = false;
            cfg.visibility_threshold = vis_thresh;
            cfg.max_neighbors = int(nf);             // EXACT: floor must see ALL walls
            cfg.search_radius_factor = 1000.0;
            cfg.method = radiation::RadiationConfig::ViewFactorMethod::CENTROID;
            cfg.adaptive_neighbors = false;
            cfg.nearfield_window_factor = window_factor;  // 0 => pure M1; huge => pure M2
            // Near-field quadrature order. Paired with PTM_BOWL_WINDOW=1e12
            // (every pair goes to M2) this makes vf_seconds a clean measurement
            // of the near-field kernel alone, which is what the order actually
            // costs: m^4 evaluations per directed pair.
            cfg.nearfield_quadrature_order = quad_order;
            cfg.captured_energy_target = 1.0;

            radiation::ViewFactorCalculator calc(b.mesh, dev, scene, cfg);
            F = calc.computeAll();
            rsums = F.rowSums();
            if (!write_csr.empty()) write_csr_or_die(write_csr, b.areas, F);

            std::vector<double> col_wsum;            // (F^T areas)_i = sum_j A_j F_ji
            if (recip) F.matvecTranspose(b.areas, col_wsum);

            for (size_t i = 0; i < nf; ++i) {
                if (b.mask[i] != 1) continue;
                double rs = rsums[i];
                if (recip) rs = 0.5 * rs + 0.5 * col_wsum[i] / std::max(b.areas[i], 1e-30);
                fsky_face.push_back(1.0 - rs);
                fsky_area.push_back(b.areas[i]);
            }
            kernel_label = (window_factor <= 0.0) ? "m1"
                         : (window_factor >= 1e12 ? "m2" : "adaptive");
        } else {
            // ---- kernel-probe path: floor F_sky directly from geometry, with a
            //      chosen kernel + identical occlusion (isolates the kernel). ----
            // The CSV `kernel` column must never carry a name that also means
            // something else in the production path. Production writes "m2" for
            // the SYMMETRIC double-area rule; the probe's "m2" is the retired
            // one-sided kernel, so it is recorded under its own name.
            if (kprobe == "m2" || kprobe == "m2_legacy") {
                if (kprobe == "m2") {
                    std::cerr << "[bowl] WARNING: PTM_BOWL_KPROBE=m2 selects the RETIRED "
                                 "one-sided near-field kernel, which production dropped on "
                                 "2026-08-23. Production's near-field M2 is now the "
                                 "symmetric double-area rule -- that is PTM_BOWL_KPROBE="
                                 "refined. Recording this row as m2_onesided_legacy."
                              << std::endl;
                }
                kernel_label = "m2_onesided_legacy";
            } else {
                kernel_label = kprobe;
            }
            std::vector<int> floor_idx;
            for (size_t i = 0; i < nf; ++i) if (b.mask[i] == 1) floor_idx.push_back(int(i));
            fsky_face.assign(floor_idx.size(), 0.0);
            fsky_area.assign(floor_idx.size(), 0.0);
            const auto& C = b.mesh->face_centers;
            const auto& N = b.mesh->face_normals;
            const auto& V = b.mesh->vertices;
            const auto& Fc = b.mesh->faces;

            auto kernel_ij = [&](int i, int j) -> double {
                if (kprobe == "m1")
                    return vf_m1(C[i], N[i], C[j], N[j], b.areas[j]);
                // proximity-adaptive subdivision count (mirrors refined kernel)
                std::array<double, 3> rv = {{C[j][0] - C[i][0], C[j][1] - C[i][1], C[j][2] - C[i][2]}};
                double r0 = std::sqrt(rv[0] * rv[0] + rv[1] * rv[1] + rv[2] * rv[2]);
                double dchar = std::sqrt(b.areas[i]) + std::sqrt(b.areas[j]);
                int m = (r0 > 1e-12) ? int(std::ceil(4.0 * dchar / r0)) : 16;
                m = std::max(1, std::min(16, m));
                std::vector<std::array<double, 3>> sj;
                tri_subcentroids(V[Fc[j][0]], V[Fc[j][1]], V[Fc[j][2]], m, sj);
                if (kprobe == "m2" || kprobe == "m2_legacy")
                    return vf_m2_legacy_onesided(C[i], N[i], N[j], b.areas[j], sj);
                std::vector<std::array<double, 3>> si;
                tri_subcentroids(V[Fc[i][0]], V[Fc[i][1]], V[Fc[i][2]], m, si);
                return vf_refined(N[i], N[j], b.areas[i], b.areas[j], si, sj);
            };

            #pragma omp parallel for schedule(dynamic, 8)
            for (long fi = 0; fi < (long)floor_idx.size(); ++fi) {
                int i = floor_idx[fi];
                double rs = 0.0;
                for (size_t j = 0; j < nf; ++j) {
                    if (int(j) == i) continue;
                    if (!ray_visible(scene, C[i], C[j])) continue;
                    double Fij = kernel_ij(i, int(j));
                    if (recip) {
                        double Fji = kernel_ij(int(j), i);
                        Fij = 0.5 * Fij + 0.5 * b.areas[j] * Fji / std::max(b.areas[i], 1e-30);
                    }
                    rs += Fij;
                }
                fsky_face[fi] = 1.0 - rs;
                fsky_area[fi] = b.areas[i];
            }
        }

        vf_seconds = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - t0).count();

        // ---- floor F_sky statistics ----
        size_t n_floor = fsky_face.size();
        double Afl = 0.0, mean_aw = 0.0, mean_p = 0.0;
        double fmin = 1e300, fmax = -1e300;
        for (size_t k = 0; k < n_floor; ++k) {
            mean_aw += fsky_area[k] * fsky_face[k];
            Afl += fsky_area[k];
            mean_p += fsky_face[k];
            fmin = std::min(fmin, fsky_face[k]);
            fmax = std::max(fmax, fsky_face[k]);
        }
        mean_aw /= std::max(Afl, 1e-30);
        mean_p /= std::max<size_t>(n_floor, 1);
        double var = 0.0;
        for (size_t k = 0; k < n_floor; ++k) { double d = fsky_face[k] - mean_p; var += d * d; }
        double sd = n_floor ? std::sqrt(var / n_floor) : 0.0;
        double mean_bias_pct = 100.0 * (mean_aw - fsky_an) / fsky_an;
        double peak_dev_pct = 100.0 * std::max(std::abs(fmin - fsky_an),
                                               std::abs(fmax - fsky_an)) / fsky_an;

        std::cout << std::fixed << std::setprecision(6)
                  << "  faces=" << nf << " n_floor=" << n_floor
                  << " kernel=" << kernel_label << (no_occ ? " NOOCC" : "")
                  << (recip ? " RECIP" : "") << "\n"
                  << "  analytic sin^2a=" << fsky_an
                  << "  mean_aw=" << mean_aw << "  mean=" << mean_p
                  << "  min=" << fmin << "  max=" << fmax << "  std=" << sd << "\n"
                  << "  mean_bias=" << std::setprecision(4) << mean_bias_pct
                  << "%  peak_dev=" << peak_dev_pct << "%  beta_c="
                  << std::setprecision(2) << bcrit << " deg  (" << std::setprecision(1)
                  << vf_seconds << " s)"
                  << std::endl << std::defaultfloat << std::flush;

        {
            std::ofstream r(results_path, std::ios::app);
            r << std::setprecision(10)
              << (kprobe.empty() ? "computeAll" : "kprobe") << "," << dd << ","
              << n_rings << "," << n_az << "," << nf << "," << n_floor << ","
              << vis_thresh << "," << window_factor << "," << (no_occ ? 0 : 1) << ","
              << (recip ? 1 : 0) << "," << kernel_label << "," << fsky_an << ","
              << mean_aw << "," << mean_p << "," << fmin << "," << fmax << "," << sd << ","
              << mean_bias_pct << "," << peak_dev_pct << "," << vf_seconds << "\n";
        }

        if (vf_only || !kprobe.empty()) {
            rtcReleaseScene(scene);
            rtcReleaseDevice(dev);
            ++idx;
            continue;   // VF-only / kernel-probe: skip the beta sweep + npy render
        }

        // The render/equilibrium path reuses the production VF matrix F (and its
        // row sums) computed above -- identical to the original driver behavior.
        double Fsky_floor = mean_aw;   // area-weighted floor F_sky, computed above

        // Equilibrium uses a fast SpMV on the PTM3D ray-traced view factors --
        // the single-bounce gray-body limit of the production RadiositySolver
        // (q_ind = ε² σ Σ_j F_ij T_j⁴). At ε=0.95 the omitted multiple-scatter of
        // reflected IR is <0.3%; the full solver agreed at lower resolution. This
        // makes the 10x-finer render mesh tractable.
        const auto& F_val = F.values();
        const auto& F_col = F.colIndices();
        const auto& F_rp = F.rowPointers();
        auto matvec = [&](const std::vector<double>& x, std::vector<double>& out) {
            #pragma omp parallel for schedule(static)
            for (long i = 0; i < (long)nf; ++i) {
                double s = 0.0;
                for (int64_t k = F_rp[i]; k < F_rp[i + 1]; ++k) s += F_val[k] * x[F_col[k]];
                out[i] = s;
            }
        };
        // 4-sample lit fraction (centroid + 3 edge midpoints) -> anti-aliased
        // shadow terminator. Returns mu * (lit samples / 4), or 0 if back-facing.
        auto lit_mu = [&](size_t i, const std::array<double, 3>& s) {
            const auto& nrm = b.mesh->face_normals[i];
            double mu = nrm[0] * s[0] + nrm[1] * s[1] + nrm[2] * s[2];
            if (mu <= 0.0) return 0.0;
            const auto& fc = b.mesh->faces[i];
            const auto& v0 = b.mesh->vertices[fc[0]];
            const auto& v1 = b.mesh->vertices[fc[1]];
            const auto& v2 = b.mesh->vertices[fc[2]];
            std::array<std::array<double, 3>, 4> P = {{b.mesh->face_centers[i], {}, {}, {}}};
            for (int k = 0; k < 3; ++k) {
                P[1][k] = 0.5 * (v0[k] + v1[k]);
                P[2][k] = 0.5 * (v1[k] + v2[k]);
                P[3][k] = 0.5 * (v0[k] + v2[k]);
            }
            int lit = 0;
            for (auto& p : P) if (!shadowed(scene, p, nrm, s, mesh_eps)) ++lit;
            return mu * (lit / 4.0);
        };

        // ---- Two-band β-sweep + render fields for the Moon3D Fig-7 composite --
        // Env-gated (PTM_BOWL_FIG7=1). Reuses the PRODUCTION two-band
        // RadiositySolver (PTM_RAD_TWO_BAND=1 + PTM_SOLAR_ALBEDO=A) on the exact
        // ray-traced VF F: for ONE mesh build it sweeps solar elevation over the
        // shadowed range, recording per β/β_c the shadow (PSR) area-weighted mean
        // T, the numerical shadow (indirect) flux εσ<T⁴>_PSR (= the absorbed
        // indirect flux at equilibrium), and the Ingersoll & Svitek (1992) Eq-8
        // analytic shadow flux; and writes the full per-face equilibrium
        // temperature field at the render β/β_c values to .npy for the 3D renders.
        //   PTM_BOWL_FIG7_NB    : sweep β points over [fmin, ftail] (default 18)
        //   PTM_BOWL_FIG7_FMIN  : min β/β_c (default 0.10)
        //   PTM_BOWL_FIG7_FMAX  : max β/β_c (default 0.92)
        //   PTM_BOWL_FIG7_FTAIL : split point for the dense tail (default = fmax)
        //   PTM_BOWL_FIG7_NBTAIL: extra points in the dense tail (ftail, fmax]
        //                         (default 0 => a single uniform [fmin, fmax] grid).
        //                         The tail densifies the near-critical onset where
        //                         the PSR sunward edge starts catching direct sun.
        //   PTM_BOWL_FIG7_RB    : comma-separated render β/β_c (default "0.30,0.85")
        //   PTM_BOWL_ALBEDO / PTM_SOLAR_ALBEDO : Bond albedo A (Moon 0.12)
        //   PTM_BOWL_FIG7_SWEEP : sweep CSV path (default out_dir+"/fig7_sweep.csv")
        //   PTM_BOWL_GS         : Gauss-Seidel (default Jacobi; two-band GS≈Jacobi)
        if (env_flag("PTM_BOWL_FIG7")) {
            const double A_dir = env_dbl("PTM_BOWL_ALBEDO", ALBEDO);
            const int    NBs   = env_int("PTM_BOWL_FIG7_NB", 18);
            const double fmin  = env_dbl("PTM_BOWL_FIG7_FMIN", 0.10);
            const double fmax  = env_dbl("PTM_BOWL_FIG7_FMAX", 0.92);
            const double ftail = env_dbl("PTM_BOWL_FIG7_FTAIL", fmax);
            const int    NBtail = env_int("PTM_BOWL_FIG7_NBTAIL", 0);
            std::vector<double> rfr;                       // render β/β_c fractions
            {
                std::string rs = env_s("PTM_BOWL_FIG7_RB", "0.30,0.85");
                size_t p = 0;
                while (p <= rs.size()) {
                    size_t c = rs.find(',', p);
                    std::string tok = rs.substr(p, (c == std::string::npos) ? c : c - p);
                    if (!tok.empty()) rfr.push_back(std::atof(tok.c_str()));
                    if (c == std::string::npos) break;
                    p = c + 1;
                }
            }
            const std::string sweep_csv =
                env_s("PTM_BOWL_FIG7_SWEEP", out_dir + "/fig7_sweep.csv");

            std::vector<double> emiss(nf, EMISS);
            radiation::RadiositySolver solver(F, emiss, b.areas);
            radiation::RadiationConfig scfg;
            scfg.max_iterations = 200;
            scfg.convergence_tolerance = 1e-7;
            scfg.use_acceleration = env_flag("PTM_BOWL_GS");   // default Jacobi
            scfg.use_gpu = false;
            solver.setConfig(scfg);
            const bool tb_on = solver.isTwoBand();

            // One two-band radiative-equilibrium Picard solve at a given beta.
            // Returns the full per-face T; fills Tf_num (area-wt PSR mean T),
            // flux_num (area-wt PSR mean εσT⁴ = numerical absorbed indirect flux
            // at equilibrium) and floor_lit_max (max direct solar on any PSR face;
            // > 0 near β_c when the finite PSR's sunward edge starts catching sun).
            auto solve_beta = [&](double beta_deg, std::vector<double>& T,
                                  double& Tf_num, double& flux_num,
                                  double& floor_lit_max, int& fp_it) {
                const double beta = beta_deg * PI / 180.0;
                const std::array<double, 3> s = {{std::cos(beta), 0.0, std::sin(beta)}};
                std::vector<double> solar_inc(nf, 0.0), direct_abs(nf, 0.0);
                #pragma omp parallel for schedule(dynamic, 64)
                for (long i = 0; i < (long)nf; ++i) {
                    double lm = lit_mu(size_t(i), s);
                    solar_inc[i]  = SOLAR * lm;
                    direct_abs[i] = (1.0 - A_dir) * SOLAR * lm;
                }
                floor_lit_max = 0.0;
                for (size_t i = 0; i < nf; ++i)
                    if (b.mask[i] == 1)
                        floor_lit_max = std::max(floor_lit_max, solar_inc[i]);
                if (T.size() != nf) T.assign(nf, 200.0);
                fp_it = 0;
                for (int it = 0; it < 800; ++it) {
                    std::vector<double> ind = solver.solve(T, solar_inc);
                    double maxd = 0.0;
                    for (size_t i = 0; i < nf; ++i) {
                        double rhs = direct_abs[i] + ind[i];
                        double Tn = std::pow(std::max(rhs, 0.0) / (EMISS * SIGMA), 0.25);
                        maxd = std::max(maxd, std::abs(Tn - T[i]));
                        T[i] = 0.5 * (T[i] + Tn);
                    }
                    fp_it = it + 1;
                    if (maxd < 5e-4) break;
                }
                double sT = 0.0, sQ = 0.0, sA = 0.0;
                for (size_t i = 0; i < nf; ++i) if (b.mask[i] == 1) {
                    sT += b.areas[i] * T[i];
                    sQ += b.areas[i] * EMISS * SIGMA * std::pow(T[i], 4);
                    sA += b.areas[i];
                }
                Tf_num   = sT / std::max(sA, 1e-30);
                flux_num = sQ / std::max(sA, 1e-30);
            };

            // β/β_c grid (ascending, warm-started) merged with the render fracs.
            // Main uniform grid over [fmin, ftail] + optional dense tail (ftail, fmax].
            std::vector<double> fracs;
            const double main_hi = (NBtail > 0) ? ftail : fmax;
            for (int i = 0; i < NBs; ++i)
                fracs.push_back(fmin + (main_hi - fmin) * i / std::max(NBs - 1, 1));
            for (int i = 1; i <= NBtail; ++i)
                fracs.push_back(ftail + (fmax - ftail) * i / NBtail);
            for (double rf : rfr)
                if (std::find_if(fracs.begin(), fracs.end(), [&](double x) {
                        return std::abs(x - rf) < 1e-9; }) == fracs.end())
                    fracs.push_back(rf);
            std::sort(fracs.begin(), fracs.end());

            {
                std::ifstream chk(sweep_csv);
                bool need = !chk.good();
                std::ofstream r(sweep_csv, std::ios::app);
                if (need)
                    r << "dd,nf,n_floor,A,eps,two_band,betafrac,beta_deg,bcrit_deg,"
                         "floor_lit_max,Tf_num,flux_num,Tf_eq8,flux_eq8,dT_pct,dflux_pct\n";
            }

            // Mesh geometry (shared by both render betas): write once per d/D.
            const std::string pre = out_dir + "/fig7_dd" + std::to_string(idx);
            {
                std::vector<double> vflat(b.mesh->vertices.size() * 3);
                for (size_t i = 0; i < b.mesh->vertices.size(); ++i)
                    for (int k = 0; k < 3; ++k)
                        vflat[3 * i + k] = b.mesh->vertices[i][k];
                std::vector<int> fflat(nf * 3);
                for (size_t i = 0; i < nf; ++i)
                    for (int k = 0; k < 3; ++k)
                        fflat[3 * i + k] = b.mesh->faces[i][k];
                write_npy(pre + "_verts.npy", vflat, {b.mesh->vertices.size(), 3}, "<f8");
                write_npy(pre + "_faces.npy", fflat, {nf, 3}, "<i4");
                write_npy(pre + "_mask.npy", b.mask, {nf}, "<i4");
            }

            std::cout << std::defaultfloat << std::setprecision(6)
                      << "  [FIG7] dd=" << dd << "  bcrit=" << bcrit
                      << " deg  two_band=" << (tb_on ? 1 : 0) << "  A=" << A_dir
                      << "  nf=" << nf << "  sweep " << fracs.size() << " points\n"
                      << std::flush;

            std::vector<double> T;   // warm-started across ascending beta
            int ridx = 0;
            for (double fr : fracs) {
                const double beta_deg = fr * bcrit;
                double Tf_num = 0.0, flux_num = 0.0, floor_lit_max = 0.0;
                int fp_it = 0;
                solve_beta(beta_deg, T, Tf_num, flux_num, floor_lit_max, fp_it);

                const double beta = beta_deg * PI / 180.0;
                const double f = 1.0 - fsky_an, A = A_dir;
                const double sigT4_eq8 = SOLAR * std::sin(beta)
                                         * f * (1.0 - A) / (1.0 - A * f)
                                         * (1.0 + A * (1.0 - f) / EMISS);
                const double Tf_eq8   = std::pow(sigT4_eq8 / SIGMA, 0.25);
                const double flux_eq8 = EMISS * sigT4_eq8;
                const double dT_pct    = 100.0 * (Tf_num  - Tf_eq8)  / Tf_eq8;
                const double dflux_pct = 100.0 * (flux_num - flux_eq8) / flux_eq8;

                {
                    std::ofstream r(sweep_csv, std::ios::app);
                    r << std::setprecision(10)
                      << dd << "," << nf << "," << n_floor << "," << A_dir << ","
                      << EMISS << "," << (tb_on ? 1 : 0) << "," << fr << ","
                      << beta_deg << "," << bcrit << "," << floor_lit_max << ","
                      << Tf_num << "," << flux_num << "," << Tf_eq8 << ","
                      << flux_eq8 << "," << dT_pct << "," << dflux_pct << "\n";
                }
                std::cout << std::fixed << std::setprecision(4)
                    << "    b/bc=" << fr << "  beta=" << beta_deg << " deg"
                    << "  lit_max=" << floor_lit_max
                    << "  Tf_num=" << Tf_num << "  Tf_eq8=" << Tf_eq8
                    << "  dflux=" << dflux_pct << " %  (it " << fp_it << ")\n"
                    << std::defaultfloat << std::flush;

                for (double rf : rfr) if (std::abs(fr - rf) < 1e-9) {
                    write_npy(pre + "_Trender" + std::to_string(ridx) + ".npy",
                              T, {nf}, "<f8");
                    std::ofstream m(pre + "_render" + std::to_string(ridx) + ".txt");
                    m << "betafrac " << rf << "\nbeta_deg " << beta_deg
                      << "\nfloor_lit_max " << floor_lit_max << "\n";
                    ++ridx;
                }
            }

            std::ofstream meta(pre + "_meta.txt");
            meta << "dd " << dd << "\nbcrit " << bcrit
                 << "\nfsky_analytic " << fsky_an
                 << "\nn_render " << ridx << "\n";
            meta.close();
            std::cout << "  [FIG7] wrote " << pre << "_*.npy (+ sweep -> "
                      << sweep_csv << ")" << std::endl;

            rtcReleaseScene(scene);
            rtcReleaseDevice(dev);
            ++idx;
            continue;
        }

        // ---- Full production-RadiositySolver shadow-flux vs Ingersoll Eq 8 ----
        // Env-gated (PTM_BOWL_RADIOSITY=1). Drives the bowl through the PRODUCTION
        // radiation::RadiositySolver (thermal IR + reflected-solar radiosity, all
        // bounces) at a fixed solar elevation and compares the shadowed central-
        // floor temperature/flux to Ingersoll (1992) Eq 8 (the exact spherical-
        // Lambert shadow-floor sigma T^4). NOT the driver's IR-only single-bounce
        // render path. Isolation toggles:
        //   PTM_BOWL_ALBEDO   : visible Bond albedo A for DIRECT absorption (1-A);
        //                       also the A used in the Eq-8 reference. Default 0.12.
        //   PTM_BOWL_SCATTER  : 1 = scatter solar through the solver; 0 = no visible
        //                       scattering (A=0 limit: solver solar source zeroed).
        //   PTM_SOLAR_ALBEDO  : (read INSIDE RadiositySolver::initCommon) unset =>
        //                       gray reflected-solar source rho=1-eps; set => Bond
        //                       albedo A. The inter-face F*B multiple-scatter term
        //                       is ALWAYS 1-eps regardless.
        //   PTM_BOWL_BOUNCES  : 1 = single bounce (Jacobi, 1 iter); else converged.
        //   PTM_BOWL_BETA     : solar elevation [deg] (absolute); overrides BETAFRAC.
        //   PTM_BOWL_BETAFRAC : solar elevation as a fraction of bcrit. Default 0.6.
        //   PTM_BOWL_RAD_RESULTS : CSV path (default out_dir + "/radiosity_eq8.csv").
        if (env_flag("PTM_BOWL_RADIOSITY")) {
            const double A_dir  = env_dbl("PTM_BOWL_ALBEDO", ALBEDO);
            const bool   do_scat = !(env_c("PTM_BOWL_SCATTER") &&
                                     env_c("PTM_BOWL_SCATTER")[0] == '0');
            const int    bounces = env_int("PTM_BOWL_BOUNCES", 100);
            const double betafrac = env_dbl("PTM_BOWL_BETAFRAC", 0.6);
            const double beta_deg = env_c("PTM_BOWL_BETA")
                                    ? env_dbl("PTM_BOWL_BETA", 5.0)
                                    : betafrac * bcrit;
            const double beta = beta_deg * PI / 180.0;
            const std::array<double, 3> s = {{std::cos(beta), 0.0, std::sin(beta)}};
            const std::string rad_csv =
                env_s("PTM_BOWL_RAD_RESULTS", out_dir + "/radiosity_eq8.csv");

            // Production radiosity solver on the exact ray-traced VF matrix F.
            // Jacobi (matvec is OMP-parallel); the radiosity gain rho*Sigma_F ~
            // (1-eps)(1-Fsky) ~ 0.007 so a handful of iterations fully converges.
            std::vector<double> emiss(nf, EMISS);
            radiation::RadiositySolver solver(F, emiss, b.areas);
            radiation::RadiationConfig scfg;
            scfg.max_iterations = (bounces <= 1) ? 1 : 200;
            scfg.convergence_tolerance = 1e-7;
            // Default Jacobi (clean bounce counting); PTM_BOWL_GS=1 switches to
            // Gauss-Seidel, which is the PRODUCTION sparse path (use_acceleration
            // defaults true there) -- lets the two-band GS kernel be exercised too.
            scfg.use_acceleration = env_flag("PTM_BOWL_GS");
            scfg.use_gpu = false;
            solver.setConfig(scfg);

            // Incident direct solar per face (S * mu, 4-sample shadow AA). The
            // solver reflects rho_solar * this; direct absorbed = (1-A) * this.
            std::vector<double> solar_inc(nf, 0.0), direct_abs(nf, 0.0);
            #pragma omp parallel for schedule(dynamic, 64)
            for (long i = 0; i < (long)nf; ++i) {
                double lm = lit_mu(size_t(i), s);
                solar_inc[i]  = SOLAR * lm;
                direct_abs[i] = (1.0 - A_dir) * SOLAR * lm;
            }
            // A=0 limit / scatter toggle: zero the solver's solar source so no
            // visible light is scattered into shadow (walls still emit IR).
            std::vector<double> solver_solar =
                do_scat ? solar_inc : std::vector<double>(nf, 0.0);

            // Guard: the central floor (mask==1) must be fully shadowed at this
            // beta, else the comparison to the Eq-8 shadow temperature is
            // contaminated by direct sun on floor faces.
            double floor_lit_max = 0.0;
            for (size_t i = 0; i < nf; ++i)
                if (b.mask[i] == 1) floor_lit_max = std::max(floor_lit_max, solar_inc[i]);
            if (floor_lit_max > 0.0)
                std::cout << "  [RADIOSITY] WARNING: central floor not fully shadowed"
                          << " (max incident " << floor_lit_max << " W/m^2)" << std::endl;

            // Picard fixed point: eps sigma T_i^4 = direct_abs_i + indirect_i(T).
            // Timed (cost check, test D): this loop is where the radiosity solve
            // dominates, so the gray-vs-two-band wall-cost delta shows up here.
            std::vector<double> T(nf, 200.0);
            int fp_it = 0; double fp_maxd = 0.0;
            auto picard_t0 = std::chrono::high_resolution_clock::now();
            for (int it = 0; it < 800; ++it) {
                std::vector<double> ind = solver.solve(T, solver_solar);
                double maxd = 0.0;
                for (size_t i = 0; i < nf; ++i) {
                    double rhs = direct_abs[i] + ind[i];
                    double Tn = std::pow(std::max(rhs, 0.0) / (EMISS * SIGMA), 0.25);
                    maxd = std::max(maxd, std::abs(Tn - T[i]));
                    T[i] = 0.5 * (T[i] + Tn);
                }
                fp_it = it + 1; fp_maxd = maxd;
                if (maxd < 5e-4) break;
            }
            const double picard_seconds = std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - picard_t0).count();

            // Shadowed central-floor area-weighted mean T + emitted flux.
            double sT = 0.0, sA = 0.0;
            for (size_t i = 0; i < nf; ++i)
                if (b.mask[i] == 1) { sT += b.areas[i] * T[i]; sA += b.areas[i]; }
            const double Tf_ptm   = sT / std::max(sA, 1e-30);
            const double flux_ptm = EMISS * SIGMA * std::pow(Tf_ptm, 4);

            // Ingersoll (1992) Eq 8 reference: sigma T^4 = F0 sin(e0) *
            //   f(1-A)/(1-Af) * [1 + A(1-f)/eps].  f = 1 - F_sky (geometry).
            const double f = 1.0 - fsky_an;
            const double A = A_dir;
            const double sigT4_eq8 = SOLAR * std::sin(beta)
                                     * f * (1.0 - A) / (1.0 - A * f)
                                     * (1.0 + A * (1.0 - f) / EMISS);
            const double Tf_eq8   = std::pow(sigT4_eq8 / SIGMA, 0.25);
            const double flux_eq8 = EMISS * sigT4_eq8;   // eps sigma T_eq8^4

            const double dT_pct    = 100.0 * (Tf_ptm  - Tf_eq8)   / Tf_eq8;
            const double dflux_pct = 100.0 * (flux_ptm - flux_eq8) / flux_eq8;
            const char* solar_alb_env = env_c("PTM_SOLAR_ALBEDO");

            std::cout << std::fixed << std::setprecision(4)
                << "  [RADIOSITY vs Eq8] dd=" << dd
                << "  beta=" << beta_deg << " deg (bcrit " << bcrit << ")"
                << "  A_dir=" << A_dir << "  scatter=" << (do_scat ? 1 : 0)
                << "  solar_alb=" << (solar_alb_env ? solar_alb_env : "gray(1-eps)")
                << "  two_band=" << (solver.isTwoBand() ? 1 : 0)
                << "  bounces=" << (bounces <= 1 ? "1" : "converged") << "\n"
                << "    floor_lit_max=" << floor_lit_max << " (should be 0)"
                << "  fp_iters=" << fp_it << "  maxd=" << fp_maxd
                << "  picard_s=" << picard_seconds << "\n"
                << "    Tf_ptm=" << Tf_ptm << " K   Tf_eq8=" << Tf_eq8
                << " K   dT=" << dT_pct << " %\n"
                << "    flux_ptm=" << flux_ptm << "   flux_eq8=" << flux_eq8
                << " W/m^2   dflux=" << dflux_pct << " %"
                << std::endl << std::defaultfloat << std::flush;

            {
                std::ifstream chk(rad_csv);
                bool need_hdr = !chk.good();
                std::ofstream r(rad_csv, std::ios::app);
                if (need_hdr)
                    r << "dd,nf,n_floor,beta_deg,bcrit_deg,A_dir,scatter,solar_alb,two_band,"
                         "bounces,floor_lit_max,fp_iters,Tf_ptm,Tf_eq8,dT_pct,"
                         "flux_ptm,flux_eq8,dflux_pct,picard_s\n";
                r << std::setprecision(10)
                  << dd << "," << nf << "," << n_floor << "," << beta_deg << ","
                  << bcrit << "," << A_dir << "," << (do_scat ? 1 : 0) << ","
                  << (solar_alb_env ? solar_alb_env : "gray") << ","
                  << (solver.isTwoBand() ? 1 : 0) << ","
                  << (bounces <= 1 ? 1 : 200) << "," << floor_lit_max << ","
                  << fp_it << "," << Tf_ptm << "," << Tf_eq8 << "," << dT_pct << ","
                  << flux_ptm << "," << flux_eq8 << "," << dflux_pct << ","
                  << picard_seconds << "\n";
            }

            // ---- Energy-closure + band-separated indirect-flux report ----------
            // (C) Reflected + absorbed = incident? First-bounce accounting of the
            // incident solar. The gray band reflects (1-eps) into the enclosure and
            // the surface BC absorbs (1-A): sum (1-A)+(1-eps) < 1 leaks A-(1-eps).
            // Two-band reflects the Bond A: (1-A)+A = 1 exactly (leak gone).
            const bool tb_on = solver.isTwoBand();
            double P_inc = 0.0, P_dir_abs = 0.0;
            for (size_t i = 0; i < nf; ++i) {
                P_inc     += b.areas[i] * solar_inc[i];
                P_dir_abs += b.areas[i] * (1.0 - A_dir) * solar_inc[i];
            }
            const double P_refl_2band = A_dir * P_inc;                         // Bond A reflected
            const double P_refl_gray  = (1.0 - EMISS) * P_inc;                 // (1-eps)*P_inc
            const double leak_gray    = P_inc - (P_dir_abs + P_refl_gray);     // (A-(1-eps))*P_inc
            const double leak_2band   = P_inc - (P_dir_abs + A_dir * P_inc);   // == 0 by construction
            // Multi-bounce closure of the reflected visible: launched A*Q_solar =
            // (absorbed as Q_ind^vis over all faces) + (escaped to sky). Uses the
            // band-separated output when two-band is active.
            double P_ind_vis = 0.0, P_ind_ir = 0.0;
            double q_vis_floor = 0.0, q_ir_floor = 0.0, a_floor = 0.0;
            if (tb_on) {
                std::vector<double> ind_final = solver.solve(T, solver_solar);  // refresh bands at converged T
                const auto& qvis = solver.lastIndirectVisible();
                const auto& qir  = solver.lastIndirectIR();
                for (size_t i = 0; i < nf; ++i) {
                    P_ind_vis += b.areas[i] * qvis[i];
                    P_ind_ir  += b.areas[i] * qir[i];
                    if (b.mask[i] == 1) {
                        q_vis_floor += b.areas[i] * qvis[i];
                        q_ir_floor  += b.areas[i] * qir[i];
                        a_floor     += b.areas[i];
                    }
                }
                q_vis_floor /= std::max(a_floor, 1e-30);
                q_ir_floor  /= std::max(a_floor, 1e-30);
            }
            const double P_escaped = A_dir * P_inc - P_ind_vis;  // reflected visible not absorbed

            std::cout << std::fixed << std::setprecision(6)
                << "  [ENERGY] two_band=" << (tb_on ? 1 : 0) << "  A=" << A_dir
                << "  eps=" << EMISS << "\n"
                << "    P_incident=" << P_inc << "  P_direct_abs=" << P_dir_abs
                << " (frac " << P_dir_abs / std::max(P_inc, 1e-30) << ")\n"
                << "    reflected_into_enclosure: two_band=" << P_refl_2band
                << " (frac " << A_dir << ")   gray=" << P_refl_gray
                << " (frac " << (1.0 - EMISS) << ")\n"
                << "    LEAK (incident - direct_abs - reflected):  gray="
                << leak_gray / std::max(P_inc, 1e-30) * 100.0 << " %   two_band="
                << leak_2band / std::max(P_inc, 1e-30) * 100.0 << " %\n";
            if (tb_on) {
                std::cout
                << "    band-split floor Q_ind: vis=" << q_vis_floor
                << "  ir=" << q_ir_floor << " W/m^2   (sum=" << (q_vis_floor + q_ir_floor) << ")\n"
                << "    multibounce: reflected_vis=" << (A_dir * P_inc)
                << " = absorbed_vis=" << P_ind_vis << " + escaped_sky=" << P_escaped
                << "  (residual " << (A_dir * P_inc - P_ind_vis - P_escaped) << ")\n";
            }
            std::cout << std::defaultfloat << std::flush;

            {
                const std::string en_csv = env_s("PTM_BOWL_ENERGY_RESULTS",
                                                  out_dir + "/radiosity_energy.csv");
                std::ifstream chk(en_csv);
                bool need_hdr = !chk.good();
                std::ofstream e(en_csv, std::ios::app);
                if (need_hdr)
                    e << "dd,nf,A,eps,two_band,P_incident,P_direct_abs,P_refl_2band,"
                         "P_refl_gray,leak_gray_pct,leak_2band_pct,q_vis_floor,q_ir_floor,"
                         "P_ind_vis,P_escaped_sky\n";
                e << std::setprecision(10)
                  << dd << "," << nf << "," << A << "," << EMISS << "," << (tb_on ? 1 : 0) << ","
                  << P_inc << "," << P_dir_abs << "," << P_refl_2band << "," << P_refl_gray << ","
                  << leak_gray / std::max(P_inc, 1e-30) * 100.0 << ","
                  << leak_2band / std::max(P_inc, 1e-30) * 100.0 << ","
                  << q_vis_floor << "," << q_ir_floor << "," << P_ind_vis << "," << P_escaped << "\n";
            }

            if (!dump_prefix.empty()) {
                // Per-face state at the converged T, for per-face accuracy metrics
                // (Q_ind RMS vs a high-ray-budget reference at the same N). One
                // extra solve AFTER every printed number, so nothing above moves.
                std::vector<double> qind = solver.solve(T, solver_solar);
                write_npy(dump_prefix + "_T.npy", T, {nf}, "<f8");
                write_npy(dump_prefix + "_qind.npy", qind, {nf}, "<f8");
                write_npy(dump_prefix + "_mask.npy", b.mask, {nf}, "<i4");
                write_npy(dump_prefix + "_areas.npy", b.areas, {nf}, "<f8");
                if (solver.isTwoBand()) {
                    write_npy(dump_prefix + "_qvis.npy", solver.lastIndirectVisible(), {nf}, "<f8");
                    write_npy(dump_prefix + "_qir.npy", solver.lastIndirectIR(), {nf}, "<f8");
                }
                std::cout << "  [PTM_BOWL_DUMP] wrote " << dump_prefix << "_*.npy" << std::endl;
            }

            rtcReleaseScene(scene);
            rtcReleaseDevice(dev);
            ++idx;
            continue;   // radiosity-vs-Eq8 mode: skip the IR-only render/sweep
        }

        auto solve_T = [&](double beta_deg, std::vector<double>& T) {
            double beta = beta_deg * PI / 180.0;
            std::array<double, 3> s = {{std::cos(beta), 0.0, std::sin(beta)}};
            std::vector<double> direct_over(nf, 0.0);     // direct_abs / (ε σ)
            #pragma omp parallel for schedule(dynamic, 64)
            for (long i = 0; i < (long)nf; ++i)
                direct_over[i] = (1.0 - ALBEDO) * SOLAR * lit_mu(i, s) / (EMISS * SIGMA);
            if (T.size() != nf) T.assign(nf, 200.0);
            std::vector<double> T4(nf), FT4(nf);
            for (int it = 0; it < 400; ++it) {
                for (size_t i = 0; i < nf; ++i) { double t = T[i]; T4[i] = t * t * t * t; }
                matvec(T4, FT4);
                double maxd = 0.0;
                for (size_t i = 0; i < nf; ++i) {
                    double Tn = std::pow(direct_over[i] + EMISS * FT4[i], 0.25);
                    maxd = std::max(maxd, std::abs(Tn - T[i]));
                    T[i] = 0.5 * (T[i] + Tn);
                }
                if (maxd < 5e-4) break;
            }
        };

        auto floor_mean = [&](const std::vector<double>& T) {
            double s = 0, a = 0;
            for (size_t i = 0; i < nf; ++i) if (b.mask[i] == 1) { s += b.areas[i] * T[i]; a += b.areas[i]; }
            return s / std::max(a, 1e-30);
        };
        auto twall_eff = [&](const std::vector<double>& T) {
            double s = 0, a = 0;
            for (size_t i = 0; i < nf; ++i) if (b.mask[i] == 0) { s += b.areas[i] * std::pow(T[i], 4); a += b.areas[i]; }
            return std::pow(s / std::max(a, 1e-30), 0.25);
        };
        auto floor_F_terrain = [&]() {
            double s = 0, a = 0;
            for (size_t i = 0; i < nf; ++i) if (b.mask[i] == 1) { s += b.areas[i] * rsums[i]; a += b.areas[i]; }
            return s / std::max(a, 1e-30);
        };
        auto qind_meas = [&](const std::vector<double>& T) {
            std::vector<double> T4(nf), FT4(nf);
            for (size_t i = 0; i < nf; ++i) { double t = T[i]; T4[i] = t * t * t * t; }
            matvec(T4, FT4);
            double s = 0, a = 0;
            for (size_t i = 0; i < nf; ++i) if (b.mask[i] == 1) {
                s += b.areas[i] * (EMISS * EMISS * SIGMA * FT4[i]); a += b.areas[i];
            }
            return s / std::max(a, 1e-30);
        };
        double Ff = floor_F_terrain();

        // render solves at two fixed solar elevations (one render row each)
        std::vector<double> T_r1, T_r10;
        solve_T(1.0, T_r1);
        solve_T(10.0, T_r10);

        // sweep (floor T + Q_ind + wall T_eff for the smooth analytic curve)
        const int NB = 14;
        std::vector<double> bsweep(NB), Tfm(NB), Tfi(NB), Qm(NB), Qa(NB), Twa(NB);
        std::vector<double> T;                       // warm-started across β
        for (int i = 0; i < NB; ++i) {
            double bd = 0.05 + (bcrit * 0.92 - 0.05) * i / (NB - 1);
            bsweep[i] = bd;
            solve_T(bd, T);
            double Tw = twall_eff(T);
            Twa[i] = Tw;
            Tfm[i] = floor_mean(T);
            Tfi[i] = Tw * std::pow(1.0 - fsky_an, 0.25);
            Qm[i] = qind_meas(T);
            Qa[i] = EMISS * SIGMA * std::pow(Tw, 4) * Ff;
        }

        // write outputs
        std::string pre = out_dir + "/dd" + std::to_string(idx);
        std::vector<double> vflat(b.mesh->vertices.size() * 3);
        for (size_t i = 0; i < b.mesh->vertices.size(); ++i)
            for (int k = 0; k < 3; ++k) vflat[3 * i + k] = b.mesh->vertices[i][k];
        std::vector<int> fflat(nf * 3);
        for (size_t i = 0; i < nf; ++i) for (int k = 0; k < 3; ++k) fflat[3 * i + k] = b.mesh->faces[i][k];
        write_npy(pre + "_verts.npy", vflat, {b.mesh->vertices.size(), 3}, "<f8");
        write_npy(pre + "_faces.npy", fflat, {nf, 3}, "<i4");
        write_npy(pre + "_mask.npy", b.mask, {nf}, "<i4");
        write_npy(pre + "_Trender1.npy", T_r1, {nf}, "<f8");
        write_npy(pre + "_Trender10.npy", T_r10, {nf}, "<f8");
        write_npy(pre + "_bsweep.npy", bsweep, {size_t(NB)}, "<f8");
        write_npy(pre + "_Tfmesh.npy", Tfm, {size_t(NB)}, "<f8");
        write_npy(pre + "_Tfing.npy", Tfi, {size_t(NB)}, "<f8");
        write_npy(pre + "_Tw.npy", Twa, {size_t(NB)}, "<f8");
        write_npy(pre + "_Qm.npy", Qm, {size_t(NB)}, "<f8");
        write_npy(pre + "_Qa.npy", Qa, {size_t(NB)}, "<f8");
        std::ofstream meta(pre + "_meta.txt");
        meta << "dd " << dd << "\nbcrit " << bcrit << "\nfsky_analytic " << fsky_an
             << "\nfloor_Fterrain " << Ff << "\nfloor_Fsky_mesh " << Fsky_floor << "\n";
        meta.close();
        std::cout << "  wrote " << pre << "_*.npy" << std::endl;

        rtcReleaseScene(scene);
        rtcReleaseDevice(dev);
        ++idx;
    }
    std::cout << "\nDone." << std::endl;
    return 0;
}
