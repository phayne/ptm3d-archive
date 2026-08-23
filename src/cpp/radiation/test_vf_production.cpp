/**
 * @file test_vf_production.cpp
 * @brief Closed-form gate on the PRODUCTION in-engine view-factor path.
 *
 * WHY THIS EXISTS
 * ---------------
 * `test_bowl_crater` is described project-wide as "the Ingersoll gate", but it
 * builds its view factors from its own inline analytic centroid kernel and never
 * calls `radiation::ViewFactorCalculator`. Its reciprocity arm is a tautology
 * (the harness fills F from a symmetric formula), and FC-08 -- a factor-of-two
 * far-field inflation in the OFFLINE assembler -- lived for months underneath a
 * green run of it. This file closes the in-engine half of that hole: it drives
 * the real Embree-occluded `ViewFactorCalculator` over geometry whose view
 * factors are known in closed form, and turns a scale error or a closure
 * failure red.
 *
 * The offline sweep+assembler half (`scripts/ptm12_diag/mc_ref_pp.cpp` +
 * `assemble_vf_haworth.py`) is gated separately by
 * `scripts/ptm12_diag/test_vf_assembler_gate.py`. Neither gate covers the other;
 * see the NOT COVERED banner this program prints.
 *
 * THE ARMS
 * --------
 *   [I] INGERSOLL SPHERICAL CAP. For a spherical-cap bowl of depth d and rim
 *       radius r the terrain view factor is UNIFORM over the whole cap,
 *           f = d^2 / (d^2 + r^2),      F_sky = 1 - f,
 *       because a Lambertian sphere sees itself with a position-independent
 *       kernel. Every face must reproduce the same number. Gated at 1%
 *       (Paul's ruling, 2026-08-22) on the terrain view factor, which is the
 *       tighter of the two normalizations -- referenced to F_sky the same
 *       absolute error is smaller by f/(1-f).
 *
 *   [D] COAXIAL PARALLEL DISKS. Two coaxial disks of radii r1, r2 separated by
 *       L have the exact configuration factor
 *           R1 = r1/L, R2 = r2/L, S = 1 + (1 + R2^2)/R1^2,
 *           F_12 = 0.5 * ( S - sqrt(S^2 - 4 (r2/r1)^2) ).
 *       This is the arm that catches a PURE SCALE ERROR. F_12 is not tied to
 *       closure (the disks see mostly sky), so neither the row-sum cap nor any
 *       reciprocity symmetrization can hide a factor on it. The area-weighted
 *       mean of the production row sums over disk 1 IS F_12 by definition, so
 *       the comparison is exact, not a limit.
 *
 *   [R] RECIPROCITY on the PRODUCTION matrix. Here this is NOT a tautology:
 *       `ViewFactorCalculator::computeSubset` computes every (i,j) independently
 *       from the M1/M3 kernel and never symmetrizes. A_i F_ij == A_j F_ji only
 *       if the kernel, the areas and the occlusion agree in both directions.
 *
 *   [S] CLOSURE, sum_j F_ij <= 1. Necessary, never sufficient -- FC-08 passed
 *       it with entries 2x too large. Kept because it is free and because a
 *       row sum above 1 is an energy-fabricating defect in its own right
 *       (PTM-12).
 *
 * CONFIGURATIONS
 * --------------
 * Two, and the difference between them is the point.
 *
 *   "no-truncation" -- `adaptive_neighbors=false`, `max_neighbors=n_faces`.
 *       Production-reachable, not test-only: `--max-neighbors N` on
 *       ptm_thermal_sim sets exactly this pair (ptm_thermal_sim.cpp:15274-15276
 *       and :17605-17607). It isolates the VF ESTIMATOR + occlusion + storage
 *       threshold, which is where a scale error lives. Gated tightly.
 *
 *   "defaults" -- `RadiationConfig{}` untouched: adaptive_neighbors=true,
 *       captured_energy_target=0.95, max_neighbors=500. The greedy selector
 *       stops once 95% of the ranked radiative potential is captured, so this
 *       configuration is EXPECTED to read a few percent LOW, by design. It is
 *       still gated -- with a two-sided band whose upper edge is tight enough
 *       that a 2x inflation is red -- because "expected to be approximate" is
 *       not the same as "ungated".
 *
 * NEGATIVE CONTROLS (a gate that has never been seen to fail is not a gate)
 * ------------------------------------------------------------------------
 * All three are self-tests of the DETECTOR, not statements about production;
 * each perturbs the assembled matrix (or the thresholds) after the production
 * calculator has returned, and each prints a loud banner so a perturbed run can
 * never be quoted as a pass.
 *
 *   PTM_VFGATE_INJECT_SCALE=<x>     multiply every entry with Ns > the adaptive
 *                                   threshold (the FAR FIELD, FC-08's home) by x.
 *   PTM_VFGATE_INJECT_RECIP=<x>     multiply ONE entry by x, breaking A_iF_ij =
 *                                   A_jF_ji for that pair only.
 *   --tol-ingersoll / --tol-disk    tighten a tolerance below the measured
 *                                   residual.
 *
 * Usage: test_vf_production [--refine] [--tol-ingersoll PCT]
 *                           [--tol-disk PCT] [--tol-recip PCT]
 */
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include <embree4/rtcore.h>

#include "../mesh/mesh_utils.h"
#include "radiation_config.h"
#include "view_factor.h"

namespace {

constexpr double PI = 3.14159265358979323846;

// ---------------------------------------------------------------------------
// closed forms
// ---------------------------------------------------------------------------

// Ingersoll & Svitek (1992): terrain view factor of a spherical-cap bowl of
// depth d and rim radius r_rim, uniform over the whole cap.
double ingersollTerrainVF(double d, double r_rim) {
    return (d * d) / (d * d + r_rim * r_rim);
}

// Configuration factor between two coaxial parallel disks (radius r1 -> r2,
// separation L). Standard closed form; exact, no series truncation.
double coaxialDiskF12(double r1, double r2, double L) {
    const double R1 = r1 / L;
    const double R2 = r2 / L;
    const double S = 1.0 + (1.0 + R2 * R2) / (R1 * R1);
    const double disc = S * S - 4.0 * (r2 / r1) * (r2 / r1);
    return 0.5 * (S - std::sqrt(disc));
}

// ---------------------------------------------------------------------------
// geometry -- built as a production ptm::Mesh, consumed by the production
// ViewFactorCalculator through a production-style committed Embree scene.
// ---------------------------------------------------------------------------

struct BuiltMesh {
    std::shared_ptr<ptm::Mesh> mesh;
    std::vector<double> areas;
    std::vector<int> group;   // arm D: 0 = lower disk, 1 = upper disk
};

void finalizeMesh(BuiltMesh& b, const std::array<double, 3>& inward_ref, bool radial_from_ref) {
    auto& m = *b.mesh;
    const size_t nf = m.faces.size();
    m.face_centers.resize(nf);
    m.face_normals.resize(nf);
    b.areas.resize(nf);
    for (size_t i = 0; i < nf; ++i) {
        const auto& v0 = m.vertices[m.faces[i][0]];
        const auto& v1 = m.vertices[m.faces[i][1]];
        const auto& v2 = m.vertices[m.faces[i][2]];
        std::array<double, 3> c = {{(v0[0] + v1[0] + v2[0]) / 3.0,
                                    (v0[1] + v1[1] + v2[1]) / 3.0,
                                    (v0[2] + v1[2] + v2[2]) / 3.0}};
        m.face_centers[i] = c;
        const std::array<double, 3> e1 = {{v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]}};
        const std::array<double, 3> e2 = {{v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]}};
        std::array<double, 3> n = {{e1[1] * e2[2] - e1[2] * e2[1],
                                    e1[2] * e2[0] - e1[0] * e2[2],
                                    e1[0] * e2[1] - e1[1] * e2[0]}};
        const double len = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        b.areas[i] = 0.5 * len;
        for (int k = 0; k < 3; ++k) n[k] /= (len > 0.0 ? len : 1.0);
        if (radial_from_ref) {
            // Spherical cap: the surface normal points at the sphere center.
            const std::array<double, 3> to_ref = {{inward_ref[0] - c[0],
                                                   inward_ref[1] - c[1],
                                                   inward_ref[2] - c[2]}};
            if (n[0] * to_ref[0] + n[1] * to_ref[1] + n[2] * to_ref[2] < 0.0)
                for (int k = 0; k < 3; ++k) n[k] = -n[k];
        }
        m.face_normals[i] = n;
    }
}

// A pure spherical cap -- no apron. The apron in bowl_crater_ptm.cpp is needed
// for the thermal sweep, but it destroys the uniform-F_sky property near the rim
// and this arm scores every face, not only the floor.
BuiltMesh buildSphericalCap(double dd, double diameter, int n_rings, int n_az) {
    const double d = dd * diameter;
    const double r_rim = 0.5 * diameter;
    const double R = (r_rim * r_rim + d * d) / (2.0 * d);
    const double z_c = R - d;                       // sphere center on +z
    const double theta_max = std::asin(r_rim / R);

    BuiltMesh b;
    b.mesh = std::make_shared<ptm::Mesh>();
    auto& V = b.mesh->vertices;
    auto& F = b.mesh->faces;

    V.push_back({{0.0, 0.0, -d}});
    for (int k = 1; k <= n_rings; ++k) {
        const double th = theta_max * k / n_rings;
        const double rho = R * std::sin(th);
        const double z = z_c - R * std::cos(th);
        for (int j = 0; j < n_az; ++j) {
            const double phi = 2.0 * PI * j / n_az;
            V.push_back({{rho * std::cos(phi), rho * std::sin(phi), z}});
        }
    }
    for (int j = 0; j < n_az; ++j)
        F.push_back({{0, 1 + j, 1 + (j + 1) % n_az}});
    for (int k = 1; k < n_rings; ++k) {
        const int bi = 1 + (k - 1) * n_az, bo = 1 + k * n_az;
        for (int j = 0; j < n_az; ++j) {
            const int j1 = (j + 1) % n_az;
            F.push_back({{bi + j, bo + j, bo + j1}});
            F.push_back({{bi + j, bo + j1, bi + j1}});
        }
    }
    b.group.assign(F.size(), 0);
    finalizeMesh(b, {{0.0, 0.0, z_c}}, true);
    return b;
}

// Two coaxial disks: lower at z = 0 facing +z, upper at z = L facing -z.
BuiltMesh buildCoaxialDisks(double r1, double r2, double L, int n_rings, int n_az) {
    BuiltMesh b;
    b.mesh = std::make_shared<ptm::Mesh>();
    auto& V = b.mesh->vertices;
    auto& F = b.mesh->faces;

    for (int disk = 0; disk < 2; ++disk) {
        const double rad = (disk == 0) ? r1 : r2;
        const double z = (disk == 0) ? 0.0 : L;
        const int base = static_cast<int>(V.size());
        const size_t f0 = F.size();
        V.push_back({{0.0, 0.0, z}});
        std::vector<int> ring_start;
        for (int k = 1; k <= n_rings; ++k) {
            ring_start.push_back(static_cast<int>(V.size()));
            // Equal-area rings so no facet dominates the area weighting.
            const double rk = rad * std::sqrt(static_cast<double>(k) / n_rings);
            for (int j = 0; j < n_az; ++j) {
                const double phi = 2.0 * PI * j / n_az;
                V.push_back({{rk * std::cos(phi), rk * std::sin(phi), z}});
            }
        }
        for (int j = 0; j < n_az; ++j)
            F.push_back({{base, ring_start[0] + j, ring_start[0] + (j + 1) % n_az}});
        for (int k = 0; k + 1 < n_rings; ++k) {
            const int a0 = ring_start[k], b0 = ring_start[k + 1];
            for (int j = 0; j < n_az; ++j) {
                const int j1 = (j + 1) % n_az;
                F.push_back({{a0 + j, b0 + j, b0 + j1}});
                F.push_back({{a0 + j, b0 + j1, a0 + j1}});
            }
        }
        for (size_t i = f0; i < F.size(); ++i) b.group.push_back(disk);
    }

    finalizeMesh(b, {{0.0, 0.0, 0.0}}, false);
    // Orient by construction: lower disk +z, upper disk -z. finalizeMesh leaves
    // the winding normal alone here, so fix the sign explicitly rather than
    // relying on the vertex order.
    for (size_t i = 0; i < b.mesh->faces.size(); ++i) {
        const double want = (b.group[i] == 0) ? 1.0 : -1.0;
        if (b.mesh->face_normals[i][2] * want < 0.0)
            for (int k = 0; k < 3; ++k) b.mesh->face_normals[i][k] = -b.mesh->face_normals[i][k];
    }
    return b;
}

RTCScene makeScene(RTCDevice dev, const ptm::Mesh& m) {
    RTCScene scene = rtcNewScene(dev);
    RTCGeometry g = rtcNewGeometry(dev, RTC_GEOMETRY_TYPE_TRIANGLE);
    const size_t nv = m.vertices.size(), nf = m.faces.size();
    float* vb = static_cast<float*>(rtcSetNewGeometryBuffer(
        g, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, 3 * sizeof(float), nv));
    for (size_t i = 0; i < nv; ++i) {
        vb[3 * i + 0] = static_cast<float>(m.vertices[i][0]);
        vb[3 * i + 1] = static_cast<float>(m.vertices[i][1]);
        vb[3 * i + 2] = static_cast<float>(m.vertices[i][2]);
    }
    unsigned* ib = static_cast<unsigned*>(rtcSetNewGeometryBuffer(
        g, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, 3 * sizeof(unsigned), nf));
    for (size_t i = 0; i < nf; ++i) {
        ib[3 * i + 0] = static_cast<unsigned>(m.faces[i][0]);
        ib[3 * i + 1] = static_cast<unsigned>(m.faces[i][1]);
        ib[3 * i + 2] = static_cast<unsigned>(m.faces[i][2]);
    }
    rtcCommitGeometry(g);
    rtcAttachGeometry(scene, g);
    rtcReleaseGeometry(g);
    rtcCommitScene(scene);
    return scene;
}

// ---------------------------------------------------------------------------
// fault injection -- the negative controls. Applied to the assembled matrix,
// AFTER the production calculator has returned; production code is untouched.
// ---------------------------------------------------------------------------

struct Injection {
    double far_field_scale = 1.0;   // PTM_VFGATE_INJECT_SCALE
    double reciprocity_scale = 1.0; // PTM_VFGATE_INJECT_RECIP
    bool active() const { return far_field_scale != 1.0 || reciprocity_scale != 1.0; }
};

Injection readInjection() {
    Injection inj;
    if (const char* s = std::getenv("PTM_VFGATE_INJECT_SCALE")) inj.far_field_scale = std::atof(s);
    if (const char* s = std::getenv("PTM_VFGATE_INJECT_RECIP")) inj.reciprocity_scale = std::atof(s);
    return inj;
}

// A dense working copy of one CSR row set. The meshes here are small by design
// (PTM-19: never rebuild a production .vf), so the extra memory is trivial and
// it makes the reciprocity and injection passes trivial to read.
struct DenseVF {
    size_t n = 0;
    std::vector<double> f;  // row-major n x n
    double& at(size_t i, size_t j) { return f[i * n + j]; }
    double at(size_t i, size_t j) const { return f[i * n + j]; }
};

DenseVF densify(const radiation::ViewFactorMatrix& F) {
    DenseVF d;
    d.n = F.numRows();
    d.f.assign(d.n * d.n, 0.0);
    std::vector<int> cols;
    std::vector<double> vals;
    for (size_t i = 0; i < d.n; ++i) {
        F.getRow(static_cast<int>(i), cols, vals);
        for (size_t k = 0; k < cols.size(); ++k) d.at(i, static_cast<size_t>(cols[k])) = vals[k];
    }
    return d;
}

// Far field == the regime the ADAPTIVE method sends to the M1 centroid kernel,
// i.e. Ns = r / sqrt(A_eff) above adaptive_ns_threshold. That is exactly where
// FC-08's factor of two lived, so the control perturbs the same population.
size_t injectFarFieldScale(DenseVF& d, const BuiltMesh& b, double ns_threshold, double scale) {
    size_t touched = 0;
    const auto& C = b.mesh->face_centers;
    for (size_t i = 0; i < d.n; ++i) {
        for (size_t j = 0; j < d.n; ++j) {
            if (d.at(i, j) == 0.0) continue;
            const double dx = C[j][0] - C[i][0], dy = C[j][1] - C[i][1], dz = C[j][2] - C[i][2];
            const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
            const double A_eff = std::sqrt(b.areas[i] * b.areas[j]);
            if (r / std::sqrt(A_eff) > ns_threshold) {
                d.at(i, j) *= scale;
                ++touched;
            }
        }
    }
    return touched;
}

// Perturb exactly ONE entry, chosen deterministically: the far-field two-sided
// pair carrying the most A_i F_ij mass. Far field on purpose -- that is FC-08's
// regime and the one whose reciprocity arm is gated at machine precision. Which
// pair was hit is printed, so the control is auditable rather than "some entry".
bool injectReciprocityBreak(DenseVF& d, const BuiltMesh& b, double ns_threshold,
                            double scale) {
    const auto& C = b.mesh->face_centers;
    double best = 0.0;
    size_t bi = 0, bj = 0;
    bool found = false;
    for (size_t i = 0; i < d.n; ++i) {
        for (size_t j = 0; j < d.n; ++j) {
            if (i == j || d.at(i, j) <= 0.0 || d.at(j, i) <= 0.0) continue;
            const double dx = C[j][0] - C[i][0], dy = C[j][1] - C[i][1], dz = C[j][2] - C[i][2];
            const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
            if (r / std::sqrt(std::sqrt(b.areas[i] * b.areas[j])) <= ns_threshold) continue;
            const double mass = b.areas[i] * d.at(i, j);
            if (mass > best) { best = mass; bi = i; bj = j; found = true; }
        }
    }
    if (!found) return false;
    d.at(bi, bj) *= scale;
    std::cout << "  injection: perturbed the single far-field entry F(" << bi << ","
              << bj << ") by x" << std::fixed << std::setprecision(4) << scale << "\n";
    return true;
}

// ---------------------------------------------------------------------------
// measurement
// ---------------------------------------------------------------------------

// Verdict printer (defined below with the rest of the reporting helpers).
void verdictBelow(const char* tag, const char* what, double measured, double limit,
                  bool& all_ok);

struct RowStats {
    double mean = 0.0;
    double min = 0.0;
    double max = 0.0;
    size_t n = 0;
};

RowStats rowSumStats(const DenseVF& d, const std::vector<int>& group, int want) {
    RowStats s;
    s.min = 1e300;
    s.max = -1e300;
    double acc = 0.0;
    for (size_t i = 0; i < d.n; ++i) {
        if (want >= 0 && group[i] != want) continue;
        double rs = 0.0;
        for (size_t j = 0; j < d.n; ++j) rs += d.at(i, j);
        acc += rs;
        s.min = std::min(s.min, rs);
        s.max = std::max(s.max, rs);
        ++s.n;
    }
    s.mean = (s.n > 0) ? acc / static_cast<double>(s.n) : 0.0;
    return s;
}

// The area-weighted mean of the row sums over group `want` IS the finite-area
// configuration factor F_12 = (1/A_1) * integral over A_1 of F_dA1->2.
double areaWeightedRowSum(const DenseVF& d, const BuiltMesh& b, int want) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < d.n; ++i) {
        if (b.group[i] != want) continue;
        double rs = 0.0;
        for (size_t j = 0; j < d.n; ++j) rs += d.at(i, j);
        num += b.areas[i] * rs;
        den += b.areas[i];
    }
    return (den > 0.0) ? num / den : 0.0;
}

// Reciprocity is split by the SAME criterion the calculator itself uses to pick
// an estimator: Ns = r / sqrt(A_eff) against adaptive_ns_threshold. Above it the
// pair goes through M1, the centroid kernel A_j cos_i cos_j / (pi r^2), which is
// ANALYTICALLY reciprocal -- A_i F_ij and A_j F_ji are the same expression -- so
// any residual there is an occlusion or storage-threshold asymmetry and is gated
// at machine precision. Below it the pair goes through M2
// (`computeSubdivision`), a one-sided quadrature that integrates over the
// EMITTING face only; it is reciprocal only in the limit of infinite
// subdivision, so its residual is reported and bounded, not required to vanish.
struct RecipResult {
    double worst_far = 0.0;
    double worst_near = 0.0;
    size_t pairs_far = 0;
    size_t pairs_near = 0;
    size_t pairs_one_sided = 0;
    double one_sided_weight = 0.0;   // fraction of the A_iF_ij mass that is one-sided
    double defect_mass = 0.0;        // sum|a-c| / sum max(a,c), over two-sided pairs
};

RecipResult reciprocity(const DenseVF& d, const BuiltMesh& b, double ns_threshold) {
    RecipResult r;
    const auto& C = b.mesh->face_centers;
    double total = 0.0, one = 0.0, absdiff = 0.0, maxsum = 0.0;
    for (size_t i = 0; i < d.n; ++i) {
        for (size_t j = i + 1; j < d.n; ++j) {
            const double a = b.areas[i] * d.at(i, j);
            const double c = b.areas[j] * d.at(j, i);
            if (a == 0.0 && c == 0.0) continue;
            total += std::max(a, c);
            if (a == 0.0 || c == 0.0) {
                ++r.pairs_one_sided;
                one += std::max(a, c);
                continue;
            }
            const double dx = C[j][0] - C[i][0], dy = C[j][1] - C[i][1], dz = C[j][2] - C[i][2];
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz);
            const double A_eff = std::sqrt(b.areas[i] * b.areas[j]);
            const double rel = std::fabs(a - c) / std::max(a, c);
            absdiff += std::fabs(a - c);
            maxsum += std::max(a, c);
            if (dist / std::sqrt(A_eff) > ns_threshold) {
                ++r.pairs_far;
                r.worst_far = std::max(r.worst_far, rel);
            } else {
                ++r.pairs_near;
                r.worst_near = std::max(r.worst_near, rel);
            }
        }
    }
    r.one_sided_weight = (total > 0.0) ? one / total : 0.0;
    r.defect_mass = (maxsum > 0.0) ? absdiff / maxsum : 0.0;
    return r;
}

void reportReciprocity(const char* label, const RecipResult& rr, double tol_far,
                       double tol_near_mass, double tol_near_worst, bool& all_ok,
                       const char* tag_far, const char* tag_mass, const char* tag_near) {
    std::cout << "  reciprocity on the PRODUCTION matrix (" << label << "): "
              << rr.pairs_far << " far-field (M1) pairs, " << rr.pairs_near
              << " near-field (M2) pairs, " << rr.pairs_one_sided << " one-sided ("
              << std::fixed << std::setprecision(3) << 100.0 * rr.one_sided_weight
              << "% of the A_iF_ij mass)\n";
    std::cout << "    NOT a tautology: computeSubset() computes every (i,j)\n"
              << "    independently from the kernel and never symmetrizes.\n";
    std::cout << "    worst relative residual: far-field " << std::scientific
              << std::setprecision(3) << rr.worst_far << ", near-field "
              << rr.worst_near << "; mass-weighted defect " << rr.defect_mass
              << std::defaultfloat << "\n";
    verdictBelow(tag_far, "reciprocity, far field (M1, exact identity)",
                 rr.worst_far, tol_far, all_ok);
    verdictBelow(tag_mass, "reciprocity, mass-weighted defect (M1+M2)",
                 rr.defect_mass, tol_near_mass, all_ok);
    verdictBelow(tag_near, "reciprocity, worst near-field pair (M2 quad.)",
                 rr.worst_near, tol_near_worst, all_ok);
}

// ---------------------------------------------------------------------------
// the run
// ---------------------------------------------------------------------------

radiation::RadiationConfig noTruncationConfig(size_t n_faces) {
    radiation::RadiationConfig cfg;              // production defaults, then:
    cfg.adaptive_neighbors = false;              // == --max-neighbors N on the CLI
    cfg.max_neighbors = static_cast<int>(n_faces);
    cfg.search_radius_factor = 1000.0;           // the extent-based radius already
                                                 // covers the mesh; belt and braces
    cfg.verbose = false;
    return cfg;
}

DenseVF runProductionVF(const BuiltMesh& b, const radiation::RadiationConfig& cfg,
                        const Injection& inj, size_t* injected_entries) {
    RTCDevice dev = rtcNewDevice(nullptr);
    RTCScene scene = makeScene(dev, *b.mesh);
    radiation::ViewFactorMatrix F;
    {
        // The calculator prints a progress bar and a memory-budget block on
        // std::cout. Capture them so the gate's verdict lines stay greppable in
        // a CI log; the captured text is discarded, not suppressed at the
        // source, so nothing in production changes.
        std::ostringstream sink;
        std::streambuf* saved = std::cout.rdbuf(sink.rdbuf());
        radiation::ViewFactorCalculator calc(b.mesh, dev, scene, cfg);
        F = calc.computeAll();
        std::cout.rdbuf(saved);
    }
    DenseVF d = densify(F);
    rtcReleaseScene(scene);
    rtcReleaseDevice(dev);

    if (injected_entries) *injected_entries = 0;
    if (inj.far_field_scale != 1.0) {
        const size_t k = injectFarFieldScale(d, b, cfg.adaptive_ns_threshold, inj.far_field_scale);
        if (injected_entries) *injected_entries = k;
    }
    if (inj.reciprocity_scale != 1.0) {
        if (!injectReciprocityBreak(d, b, cfg.adaptive_ns_threshold, inj.reciprocity_scale)) {
            std::cout << "  INJECTION ERROR: no two-sided far-field entry to perturb.\n";
        }
    }
    return d;
}

void verdict(const char* tag, const char* what, double measured, double closed,
             double tol_pct, bool& all_ok) {
    const double err = (closed != 0.0) ? 100.0 * (measured / closed - 1.0) : 0.0;
    const bool ok = std::fabs(err) <= tol_pct;
    std::cout << "  [" << tag << "] " << std::left << std::setw(46) << what
              << std::right << " measured " << std::fixed << std::setprecision(6) << measured
              << "  closed form " << closed
              << "  err " << std::showpos << std::setprecision(4) << err << "%" << std::noshowpos
              << "  tol " << std::setprecision(2) << tol_pct << "%   "
              << (ok ? "PASS" : "FAILED") << "\n";
    all_ok = all_ok && ok;
}

void verdictBand(const char* tag, const char* what, double ratio,
                 double lo, double hi, bool& all_ok) {
    const bool ok = (ratio >= lo && ratio <= hi);
    std::cout << "  [" << tag << "] " << std::left << std::setw(46) << what
              << std::right << " ratio to closed form " << std::fixed << std::setprecision(6)
              << ratio << "  band [" << std::setprecision(3) << lo << ", " << hi << "]   "
              << (ok ? "PASS" : "FAILED") << "\n";
    all_ok = all_ok && ok;
}

void verdictBelow(const char* tag, const char* what, double measured, double limit,
                  bool& all_ok) {
    const bool ok = (measured <= limit);
    std::cout << "  [" << tag << "] " << std::left << std::setw(46) << what
              << std::right << " measured " << std::scientific << std::setprecision(3) << measured
              << "  limit " << limit << "   " << std::defaultfloat
              << (ok ? "PASS" : "FAILED") << "\n";
    all_ok = all_ok && ok;
}

}  // namespace

int main(int argc, char** argv) {
    bool refine = false;
    double tol_ingersoll = 1.0;      // Paul's ruling, 2026-08-22
    double tol_disk = 1.0;           // measured -0.084% at 1344 faces; the refinement
                                     // curve is -1.01/-0.28/-0.15/-0.08% at
                                     // 168/396/720/1344 faces, i.e. monotone and
                                     // convergent, so 1.0% is ~12x the residual and
                                     // still far below a 2x scale error.
    // Tolerances. Every one is pinned from a measured refinement curve, printed
    // above the verdict it gates; see SESSION-NOTES-2026-08-22-vf-production-gate.md.
    double tol_perface = 2.0;        // measured worst per-face deviation 1.19% at the
                                     // default 2,750-face cap; this is the faceting
                                     // residual of a flat-triangle cap and it SHRINKS
                                     // with resolution (4.08/2.69/1.73/1.19% at
                                     // 342/648/1404/2750 faces).
    double tol_recip = 1.0e-6;       // far field: M1 is reciprocal as an identity;
                                     // measured 1.3e-15, nine orders of headroom.
    double tol_recip_mass = 1.0e-2;  // measured 3.5e-3 on the cap (2.8x headroom).
    double tol_recip_near = 0.60;    // measured worst near-field pair 0.464. M2
                                     // integrates over the EMITTER only, so it is
                                     // reciprocal only in the subdivision limit.
                                     // This is a REGRESSION bound on a known
                                     // non-reciprocal quadrature, not a claim that
                                     // 46% is acceptable -- see the owed ledger row.
    double band_lo = 0.90, band_hi = 1.02;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--refine") refine = true;
        else if (a == "--tol-ingersoll" && i + 1 < argc) tol_ingersoll = std::atof(argv[++i]);
        else if (a == "--tol-disk" && i + 1 < argc) tol_disk = std::atof(argv[++i]);
        else if (a == "--tol-recip" && i + 1 < argc) tol_recip = std::atof(argv[++i]);
        else if (a == "--tol-recip-mass" && i + 1 < argc) tol_recip_mass = std::atof(argv[++i]);
        else if (a == "--tol-recip-near" && i + 1 < argc) tol_recip_near = std::atof(argv[++i]);
        else if (a == "--tol-perface" && i + 1 < argc) tol_perface = std::atof(argv[++i]);
        else if (a == "--band-lo" && i + 1 < argc) band_lo = std::atof(argv[++i]);
        else if (a == "--band-hi" && i + 1 < argc) band_hi = std::atof(argv[++i]);
        else {
            std::cerr << "unknown argument: " << a << "\n";
            return 2;
        }
    }

    const Injection inj = readInjection();

    std::cout << "================================================================\n";
    std::cout << "  test_vf_production -- closed-form gate on the PRODUCTION\n";
    std::cout << "  radiation::ViewFactorCalculator (Embree ray-traced).\n";
    std::cout << "================================================================\n";
    if (inj.active()) {
        std::cout << "\n";
        std::cout << "  ****************************************************************\n";
        std::cout << "  ** FAULT INJECTION ACTIVE -- THIS RUN IS A NEGATIVE CONTROL.  **\n";
        std::cout << "  ** Its verdicts describe the DETECTOR, not the production     **\n";
        std::cout << "  ** view-factor path. A run with this banner may never be      **\n";
        std::cout << "  ** quoted as a pass.                                          **\n";
        std::cout << "  **   far-field scale  x " << std::fixed << std::setprecision(4)
                  << inj.far_field_scale << "\n";
        std::cout << "  **   reciprocity      x " << inj.reciprocity_scale << "\n";
        std::cout << "  ****************************************************************\n";
        std::cerr << "test_vf_production: FAULT INJECTION ACTIVE (negative control)\n";
    }

    bool all_ok = true;

    // ------------------------------------------------------------------ ARM I
    {
        const double dd = 0.2, diameter = 1.0;
        const double d_depth = dd * diameter, r_rim = 0.5 * diameter;
        const double f_exact = ingersollTerrainVF(d_depth, r_rim);

        std::cout << "\n=== ARM I: Ingersoll spherical cap, d/D = " << dd << " ===\n";
        std::cout << "  closed form: terrain VF f = d^2/(d^2+r^2) = "
                  << std::fixed << std::setprecision(6) << f_exact
                  << ",  F_sky = " << (1.0 - f_exact) << "\n";
        std::cout << "  Uniform over the WHOLE cap (Ingersoll & Svitek 1992), so every\n"
                  << "  face must read the same number: any per-face spread is numerical.\n";

        struct Level { int rings, az; };
        std::vector<Level> levels;
        if (refine) levels = {{10, 18}, {14, 24}, {20, 36}, {28, 50}};
        else levels = {{28, 50}};

        for (size_t li = 0; li < levels.size(); ++li) {
            BuiltMesh b = buildSphericalCap(dd, diameter, levels[li].rings, levels[li].az);
            auto cfg = noTruncationConfig(b.mesh->faces.size());
            size_t injected = 0;
            DenseVF d = runProductionVF(b, cfg, inj, &injected);
            const RowStats rs = rowSumStats(d, b.group, -1);
            const double dev_lo = 100.0 * (rs.min / f_exact - 1.0);
            const double dev_hi = 100.0 * (rs.max / f_exact - 1.0);
            const double worst_face = std::max(std::fabs(dev_lo), std::fabs(dev_hi));

            std::cout << "  refinement: " << std::setw(5) << b.mesh->faces.size()
                      << " faces (" << levels[li].rings << "x" << levels[li].az
                      << ")  mean " << std::fixed << std::setprecision(6) << rs.mean
                      << "  err " << std::showpos << std::setprecision(4)
                      << 100.0 * (rs.mean / f_exact - 1.0) << "%" << std::noshowpos
                      << "  per-face [" << std::showpos << dev_lo << "%, " << dev_hi
                      << "%]" << std::noshowpos;
            if (injected) std::cout << "   (injection touched " << injected << " entries)";
            std::cout << "\n";

            if (li + 1 == levels.size()) {
                std::cout << "  config: no-truncation (adaptive_neighbors=false, max_neighbors="
                          << cfg.max_neighbors << ")\n";
                verdict("I1", "terrain VF, mean over every cap face", rs.mean, f_exact,
                        tol_ingersoll, all_ok);
                verdict("I2", "sky VF closure, 1 - sum_j F_ij", 1.0 - rs.mean,
                        1.0 - f_exact, tol_ingersoll, all_ok);
                verdictBelow("I3", "worst PER-FACE deviation from f [%]", worst_face,
                             tol_perface, all_ok);
                const RecipResult rr = reciprocity(d, b, cfg.adaptive_ns_threshold);
                reportReciprocity("spherical cap", rr, tol_recip, tol_recip_mass,
                                  tol_recip_near, all_ok, "R1a", "R1b", "R1c");
                verdictBelow("S1", "max row sum (closure, must be <= 1)", rs.max, 1.0, all_ok);
            }
        }
    }

    // ------------------------------------------------------------------ ARM D
    {
        const double r1 = 1.0, r2 = 1.0, L = 1.0;
        const double f12 = coaxialDiskF12(r1, r2, L);
        std::cout << "\n=== ARM D: coaxial parallel disks, r1 = r2 = " << r1
                  << ", L = " << L << " ===\n";
        std::cout << "  closed form: F_12 = " << std::fixed << std::setprecision(6)
                  << f12 << "\n";
        std::cout << "  This arm is the SCALE detector: F_12 is not tied to closure,\n"
                  << "  so no row-sum cap and no symmetrization can hide a factor on it.\n";

        struct Level { int rings, az; };
        std::vector<Level> levels;
        if (refine) levels = {{4, 12}, {6, 18}, {8, 24}, {11, 32}};
        else levels = {{11, 32}};

        for (size_t li = 0; li < levels.size(); ++li) {
            BuiltMesh b = buildCoaxialDisks(r1, r2, L, levels[li].rings, levels[li].az);
            auto cfg = noTruncationConfig(b.mesh->faces.size());
            size_t injected = 0;
            DenseVF d = runProductionVF(b, cfg, inj, &injected);
            const double meas = areaWeightedRowSum(d, b, 0);
            const double ratio = meas / f12;
            std::cout << "  refinement: " << std::setw(5) << b.mesh->faces.size()
                      << " faces  ->  A-weighted mean row sum " << std::fixed
                      << std::setprecision(6) << meas << "  err " << std::showpos
                      << std::setprecision(4) << 100.0 * (ratio - 1.0) << "%"
                      << std::noshowpos;
            if (injected) std::cout << "   (injection touched " << injected << " entries)";
            std::cout << "\n";

            if (li + 1 == levels.size()) {
                verdict("D1", "disk-pair F_12, no-truncation config", meas, f12,
                        tol_disk, all_ok);
                const RecipResult rr = reciprocity(d, b, cfg.adaptive_ns_threshold);
                std::cout << "  NOTE: on this geometry every surviving pair is far field\n"
                          << "  (" << rr.pairs_far << " far vs " << rr.pairs_near
                          << " near), so R2 exercises only the analytically reciprocal M1\n"
                          << "  kernel. The spherical cap above is what probes M2.\n";
                reportReciprocity("coaxial disks", rr, tol_recip, tol_recip_mass,
                                  tol_recip_near, all_ok, "R2a", "R2b", "R2c");
                const RowStats rs = rowSumStats(d, b.group, -1);
                verdictBelow("S2", "max row sum (closure, must be <= 1)", rs.max, 1.0, all_ok);
            }
        }

        // ---- the same geometry at PRODUCTION DEFAULTS -----------------------
        {
            BuiltMesh b = buildCoaxialDisks(r1, r2, L, 8, 24);
            radiation::RadiationConfig cfg;   // untouched production defaults
            cfg.verbose = false;
            size_t injected = 0;
            DenseVF d = runProductionVF(b, cfg, inj, &injected);
            const double meas = areaWeightedRowSum(d, b, 0);
            const size_t partners_per_face = (b.mesh->faces.size() / 2);
            std::cout << "\n  production DEFAULTS on the same disks ("
                      << b.mesh->faces.size() << " faces, "
                      << partners_per_face << " potential partners/face,\n"
                      << "  so max_neighbors=" << cfg.max_neighbors
                      << " does not bind and captured_energy_target="
                      << std::setprecision(2) << cfg.captured_energy_target
                      << " is the only truncation):\n";
            std::cout << "    A-weighted mean row sum " << std::fixed << std::setprecision(6)
                      << meas << "  ratio to closed form " << (meas / f12) << "\n";
            std::cout << "    The greedy selector stops at " << std::setprecision(0)
                      << 100.0 * cfg.captured_energy_target
                      << "% of the ranked radiative potential, so a few percent low is\n"
                      << "    EXPECTED here and is not a defect. The band still rejects a\n"
                      << "    2x inflation and a coupling collapse.\n";
            verdictBand("D2", "disk-pair F_12, production DEFAULT config",
                        meas / f12, band_lo, band_hi, all_ok);
        }
    }

    // ------------------------------------------------------------------------
    std::cout << "\n=== NOT COVERED BY THIS PROGRAM ===\n";
    std::cout << "  * the OFFLINE sweep + assembler (mc_ref_pp + assemble_vf_haworth.py):\n"
              << "    gated separately by scripts/ptm12_diag/test_vf_assembler_gate.py.\n";
    std::cout << "  * the H-matrix builder (hmatrix_view_factor.cpp) and the Monte-Carlo\n"
              << "    ray-gather builder (mc_view_factor.cpp): different code paths,\n"
              << "    selected by use_hmatrix / use_monte_carlo_vf. UNGATED here.\n";
    std::cout << "  * the GPU BVH visibility path (benchmark_gpu_visibility): the\n"
              << "    calculator above uses Embree on the CPU in every configuration.\n";
    std::cout << "  * the radiosity SOLVE. This program scores geometry only.\n";

    if (inj.active()) {
        std::cout << "\n[test_vf_production] NEGATIVE CONTROL RUN -- verdicts above describe\n"
                  << "[test_vf_production] the detector, not production. Exit status "
                  << (all_ok ? 0 : 1) << ".\n";
        return all_ok ? 0 : 1;
    }

    if (all_ok) {
        std::cout << "\n[test_vf_production] PASS: every closed-form verdict met its tolerance\n"
                  << "                     on the production ViewFactorCalculator.\n";
        return 0;
    }
    std::cout << "\n[test_vf_production] at least one verdict FAILED (see above).\n";
    return 1;
}
