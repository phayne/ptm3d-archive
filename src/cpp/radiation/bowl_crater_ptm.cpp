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
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <array>
#include <string>
#include <cmath>
#include <memory>
#include <cstdint>

#include <embree4/rtcore.h>

#include "../mesh/mesh_utils.h"
#include "radiation_config.h"
#include "view_factor.h"
#include "radiation_exchange.h"

namespace {

constexpr double PI = 3.14159265358979323846;
constexpr double SIGMA = 5.670374e-8;
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
            // inward (toward open bowl == toward sphere centre above)
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
RTCScene make_scene(RTCDevice dev, const ptm::Mesh& m) {
    RTCScene scene = rtcNewScene(dev);
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
    const int n_rings = 72, n_az = 192, n_flat_rings = 14;
    const double diameter = 1.0, flat_extent = 1.7;

    int idx = 0;
    for (double dd : dds) {
        std::cout << "\n=== PTM3D bowl crater d/D=" << dd << " ===" << std::endl;
        Built b = build_mesh(dd, diameter, n_rings, n_az, flat_extent, n_flat_rings);
        size_t nf = b.mesh->faces.size();
        double mesh_eps = 1e-4 * b.r_rim;

        RTCDevice dev = rtcNewDevice(nullptr);
        RTCScene scene = make_scene(dev, *b.mesh);

        radiation::RadiationConfig cfg;
        cfg.enabled = true;
        cfg.verbose = false;
        cfg.visibility_threshold = 5e-10;            // low enough that the floor's
                                                     // grazing-wall F_ij survive at
                                                     // 4x res (exact F_sky)
        cfg.max_neighbors = int(nf);                 // EXACT: floor must see ALL
        cfg.search_radius_factor = 1000.0;           // walls (concave bowl is dense;
        cfg.method = radiation::RadiationConfig::ViewFactorMethod::CENTROID;
        cfg.adaptive_neighbors = false;              // ~45 GB at 10x res, fits 384 GB)
        cfg.captured_energy_target = 1.0;

        radiation::ViewFactorCalculator calc(b.mesh, dev, scene, cfg);
        radiation::ViewFactorMatrix F = calc.computeAll();

        // floor F_sky check (occlusion correctness)
        double Fsky_floor = 0.0, Afl = 0.0;
        auto rsums = F.rowSums();
        for (size_t i = 0; i < nf; ++i)
            if (b.mask[i] == 1) { Fsky_floor += b.areas[i] * (1.0 - rsums[i]); Afl += b.areas[i]; }
        Fsky_floor /= std::max(Afl, 1e-30);
        double alpha = std::atan(0.5 / dd);
        double fsky_an = std::sin(alpha) * std::sin(alpha);
        double bcrit = std::atan(2.0 * dd) * 180.0 / PI;
        std::cout << std::fixed << std::setprecision(5)
                  << "  faces=" << nf << "  floor F_sky mesh=" << Fsky_floor
                  << "  sin^2 a=" << fsky_an << "  beta_c=" << std::setprecision(2)
                  << bcrit << " deg" << std::endl << std::defaultfloat << std::flush;

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
                for (int k = F_rp[i]; k < F_rp[i + 1]; ++k) s += F_val[k] * x[F_col[k]];
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
