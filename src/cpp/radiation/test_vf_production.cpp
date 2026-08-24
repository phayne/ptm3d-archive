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
 *   [T] TOUCHING PAIR. Two perpendicular unit squares sharing an edge, built as
 *       2 + 2 triangles and pushed through the same production calculator. The
 *       aggregate is a closed form (Howell C-14),
 *           A_1 F_12 = 0.20004378   for l = w = h = 1,
 *       and it is the WORST case for any quadrature on this kernel: the
 *       integrand has an integrable 1/r^2 singularity on the shared edge. Arms
 *       I and D cannot see this -- the cap's near pairs are shallow and the
 *       disks have no near pairs at all -- and edge/vertex-touching pairs carry
 *       17.6% of the near-field coupling mass on a real crater mesh.
 *       Gated as a two-sided BAND on the ratio, not as an accuracy tolerance:
 *       no production-order area quadrature is within a few percent of this
 *       closed form (the error falls only like 1/m), so the arm bounds the
 *       magnitude and pins the SIGN. It prints its own refinement table.
 *
 *   [R] RECIPROCITY on the PRODUCTION matrix. Here this is NOT a tautology:
 *       `ViewFactorCalculator::computeSubset` computes every (i,j) independently
 *       from the M1/M2 kernel and never symmetrizes. A_i F_ij == A_j F_ji only
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
 *                           [--tol-touch-lo R] [--tol-touch-hi R]
 *                           [--quad-order M]
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
#include "hmatrix_view_factor.h"
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

// Two perpendicular rectangles with a COMMON EDGE of length l; rectangle 1 is
// l x w, rectangle 2 is l x h. Howell, "A Catalog of Radiation Heat Transfer
// Configuration Factors", C-14. Exact.
double perpendicularRectF12(double l, double w, double h) {
    const double W = w / l;
    const double H = h / l;
    const double W2 = W * W, H2 = H * H;
    const double t = W * std::atan(1.0 / W) + H * std::atan(1.0 / H) -
                     std::sqrt(H2 + W2) * std::atan(1.0 / std::sqrt(H2 + W2));
    const double A = (1.0 + W2) * (1.0 + H2) / (1.0 + W2 + H2);
    const double B = std::pow(W2 * (1.0 + W2 + H2) / ((1.0 + W2) * (W2 + H2)), W2);
    const double C = std::pow(H2 * (1.0 + H2 + W2) / ((1.0 + H2) * (H2 + W2)), H2);
    return (t + 0.25 * std::log(A * B * C)) / (PI * W);
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

// A CONCAVE WEDGE: two unit right triangles sharing the edge (0,0,0)-(1,0,0) at
// an arbitrary dihedral angle. As the angle closes the exact A_i F_ij rises
// toward A_i and never past it, while a finite-order area quadrature diverges:
// the sample pairs nearest the shared edge have r -> 0 with cos_a cos_b -> 1.
// Order 3 reads F_ij = 1.07 at 45 degrees and 19.2 at 10 degrees against
// references 0.49 and 0.84. This geometry exists so the physical-bound tripwire
// in computeSubdivision is exercised by the gate rather than trusted.
BuiltMesh buildConcaveWedge(double dihedral_deg) {
    const double th = dihedral_deg * PI / 180.0;
    BuiltMesh b;
    b.mesh = std::make_shared<ptm::Mesh>();
    auto& V = b.mesh->vertices;
    auto& F = b.mesh->faces;

    V.push_back({{0.0, 0.0, 0.0}});
    V.push_back({{1.0, 0.0, 0.0}});
    V.push_back({{0.0, 1.0, 0.0}});
    V.push_back({{0.0, std::cos(th), std::sin(th)}});

    F.push_back({{0, 1, 2}});
    b.group.push_back(0);
    F.push_back({{0, 3, 1}});
    b.group.push_back(1);

    finalizeMesh(b, {{0.0, 0.0, 0.0}}, false);
    // Both normals must point INTO the wedge or the pair never couples.
    const auto& C = b.mesh->face_centers;
    for (size_t i = 0; i < 2; ++i) {
        const size_t o = 1 - i;
        const std::array<double, 3> to_other = {{C[o][0] - C[i][0], C[o][1] - C[i][1],
                                                 C[o][2] - C[i][2]}};
        auto& n = b.mesh->face_normals[i];
        if (n[0] * to_other[0] + n[1] * to_other[1] + n[2] * to_other[2] < 0.0)
            for (int k = 0; k < 3; ++k) n[k] = -n[k];
    }
    return b;
}

// Two perpendicular UNIT SQUARES sharing the edge from (0,0,0) to (1,0,0):
// square 1 in the z = 0 plane facing +z (group 0), square 2 in the y = 0 plane
// facing +y (group 1). Two triangles each -- the calculator only ever sees
// triangles, so the closed form has to be reassembled from four directed pairs,
// which is precisely the situation a real mesh presents.
BuiltMesh buildTouchingSquares() {
    BuiltMesh b;
    b.mesh = std::make_shared<ptm::Mesh>();
    auto& V = b.mesh->vertices;
    auto& F = b.mesh->faces;

    V.push_back({{0.0, 0.0, 0.0}});   // 0  shared edge start
    V.push_back({{1.0, 0.0, 0.0}});   // 1  shared edge end
    V.push_back({{1.0, 1.0, 0.0}});   // 2  square 1
    V.push_back({{0.0, 1.0, 0.0}});   // 3  square 1
    V.push_back({{1.0, 0.0, 1.0}});   // 4  square 2
    V.push_back({{0.0, 0.0, 1.0}});   // 5  square 2

    F.push_back({{0, 1, 2}});         // square 1, winding gives +z
    F.push_back({{0, 2, 3}});
    b.group.push_back(0);
    b.group.push_back(0);

    F.push_back({{0, 4, 1}});         // square 2, winding gives +y
    F.push_back({{0, 5, 4}});
    b.group.push_back(1);
    b.group.push_back(1);

    finalizeMesh(b, {{0.0, 0.0, 0.0}}, false);
    // Pin the orientation rather than trusting the winding: square 1 must look
    // +z and square 2 must look +y, or the two never see each other at all.
    for (size_t i = 0; i < F.size(); ++i) {
        const int axis = (b.group[i] == 0) ? 2 : 1;
        if (b.mesh->face_normals[i][axis] < 0.0)
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
// i.e. OUTSIDE radiation::isNearFieldPair at nearfield_window_factor. That is
// exactly where FC-08's factor of two lived, so the control perturbs the same
// population.
size_t injectFarFieldScale(DenseVF& d, const BuiltMesh& b, double window_factor, double scale) {
    size_t touched = 0;
    const auto& C = b.mesh->face_centers;
    for (size_t i = 0; i < d.n; ++i) {
        for (size_t j = 0; j < d.n; ++j) {
            if (d.at(i, j) == 0.0) continue;
            const double dx = C[j][0] - C[i][0], dy = C[j][1] - C[i][1], dz = C[j][2] - C[i][2];
            const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
            const double A_eff = std::sqrt(b.areas[i] * b.areas[j]);
            if (!radiation::isNearFieldPair(r, b.areas[i], b.areas[j], window_factor)) {
                d.at(i, j) *= scale;
                ++touched;
            }
        }
    }
    return touched;
}

// Perturb exactly ONE entry, chosen deterministically: the two-sided pair
// carrying the most A_i F_ij mass. FAR FIELD is preferred -- that is FC-08's
// regime -- but a geometry with no far-field pair at all (arm T: every pair is
// near field) falls back to the near field rather than injecting nothing.
// That fallback is the only control the NEAR-field reciprocity arms have, and
// those are the ones whose tolerance dropped from 0.60 to 1e-6 on 2026-08-23.
// Which pair was hit, and in which regime, is printed; the caller treats
// "found nothing" as a hard error, because a control that silently did not
// perturb anything would report a green run as proof of detection.
bool injectReciprocityBreak(DenseVF& d, const BuiltMesh& b, double window_factor,
                            double scale) {
    const auto& C = b.mesh->face_centers;
    // pass 0 = far field only, pass 1 = anything.
    for (int pass = 0; pass < 2; ++pass) {
        double best = 0.0;
        size_t bi = 0, bj = 0;
        bool found = false;
        bool near = false;
        for (size_t i = 0; i < d.n; ++i) {
            for (size_t j = 0; j < d.n; ++j) {
                if (i == j || d.at(i, j) <= 0.0 || d.at(j, i) <= 0.0) continue;
                const double dx = C[j][0] - C[i][0], dy = C[j][1] - C[i][1],
                             dz = C[j][2] - C[i][2];
                const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
                const bool is_far =
                    !radiation::isNearFieldPair(r, b.areas[i], b.areas[j], window_factor);
                if (pass == 0 && !is_far) continue;
                const double mass = b.areas[i] * d.at(i, j);
                if (mass > best) {
                    best = mass; bi = i; bj = j; found = true; near = !is_far;
                }
            }
        }
        if (!found) continue;
        d.at(bi, bj) *= scale;
        std::cout << "  injection: perturbed the single "
                  << (near ? "NEAR-field (M2)" : "far-field (M1)") << " entry F(" << bi
                  << "," << bj << ") by x" << std::fixed << std::setprecision(4)
                  << scale << "\n";
        return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// measurement
// ---------------------------------------------------------------------------

// Verdict printers (defined below with the rest of the reporting helpers).
void verdictBelow(const char* tag, const char* what, double measured, double limit,
                  bool& all_ok);
void verdictAbove(const char* tag, const char* what, double measured, double floor_v,
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

// The UNNORMALIZED coupling out of group `want`: sum over its faces of
// A_i * sum_j F_ij. For the touching-square pair that quantity IS A_1 F_12, the
// closed form, with no area normalization to argue about.
double areaWeightedCoupling(const DenseVF& d, const BuiltMesh& b, int want) {
    double acc = 0.0;
    for (size_t i = 0; i < d.n; ++i) {
        if (b.group[i] != want) continue;
        double rs = 0.0;
        for (size_t j = 0; j < d.n; ++j) rs += d.at(i, j);
        acc += b.areas[i] * rs;
    }
    return acc;
}

// Reciprocity is split by the SAME criterion the calculator itself uses to pick
// an estimator: radiation::isNearFieldPair at nearfield_window_factor. Outside it the
// pair goes through M1, the centroid kernel A_j cos_i cos_j / (pi r^2), which is
// ANALYTICALLY reciprocal -- A_i F_ij and A_j F_ji are the same expression -- so
// any residual there is an occlusion or storage-threshold asymmetry.
//
// Below the threshold the pair goes through M2 (`computeSubdivision`). Until
// 2026-08-23 that was a ONE-SIDED rule: it placed a point emitter at the
// centroid of one face and integrated over the other, i.e. each direction was a
// midpoint rule in a DIFFERENT variable of the same symmetric double integral.
// (An earlier revision of this comment had the two faces the wrong way round --
// the code integrated over the RECEIVER and treated the EMITTER as a point.
// Corrected 2026-08-23 by Fermi's review.) Its worst near-field pair read 0.4618
// on the cap. M2 is now a symmetric double-area sub-centroid quadrature
// accumulated in canonical min(i,j) order, so BOTH regimes are reciprocal as
// algebraic identities and both are gated at machine precision.
struct RecipResult {
    double worst_far = 0.0;
    double worst_near = 0.0;
    size_t pairs_far = 0;
    size_t pairs_near = 0;
    size_t pairs_one_sided = 0;
    double one_sided_weight = 0.0;   // fraction of the A_iF_ij mass that is one-sided
    double defect_mass = 0.0;        // sum|a-c| / sum max(a,c), over two-sided pairs
    size_t pairs_below_floor = 0;    // skipped by the significance cut, if any
};

// `floor_F` is a SIGNIFICANCE cut on the view factor itself, defaulting to 0
// (score everything) so the sparse arms are unchanged. It exists for the
// H-matrix arm: dense and low-rank blocks store every entry they reconstruct,
// including round-off-scale values on exactly coplanar adjacent facets, whereas
// the sparse builder drops anything below visibility_threshold and never
// carries them at all. Scoring the ratio of two 1e-20 numbers produced a
// "worst near-field pair" of 38 that was entirely noise -- measured pair
// (296,297) on the cap, cos_i = cos_j = 0.00000, F = 8.8e-21 against 1.9e-20,
// sparse builder exactly 0. Pairs cut by the floor are COUNTED and printed, so
// the cut cannot quietly remove a real one.
RecipResult reciprocity(const DenseVF& d, const BuiltMesh& b, double window_factor,
                        double floor_F = 0.0) {
    RecipResult r;
    const auto& C = b.mesh->face_centers;
    double total = 0.0, one = 0.0, absdiff = 0.0, maxsum = 0.0;
    for (size_t i = 0; i < d.n; ++i) {
        for (size_t j = i + 1; j < d.n; ++j) {
            const double a = b.areas[i] * d.at(i, j);
            const double c = b.areas[j] * d.at(j, i);
            if (a == 0.0 && c == 0.0) continue;
            if (std::max(d.at(i, j), d.at(j, i)) < floor_F) { ++r.pairs_below_floor; continue; }
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
            if (!radiation::isNearFieldPair(dist, b.areas[i], b.areas[j], window_factor)) {
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

// 0 = leave the production default (radiation_config.h) alone.
int g_quad_order_override = 0;

// Set when a requested fault injection could not be applied. Fatal at exit.
bool g_injection_failed = false;

void applyQuadOrder(radiation::RadiationConfig& cfg) {
    if (g_quad_order_override > 0) cfg.nearfield_quadrature_order = g_quad_order_override;
}

radiation::RadiationConfig noTruncationConfig(size_t n_faces) {
    radiation::RadiationConfig cfg;              // production defaults, then:
    cfg.adaptive_neighbors = false;              // == --max-neighbors N on the CLI
    cfg.max_neighbors = static_cast<int>(n_faces);
    cfg.search_radius_factor = 1000.0;           // the extent-based radius already
                                                 // covers the mesh; belt and braces
    cfg.verbose = false;
    applyQuadOrder(cfg);
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
        const size_t k = injectFarFieldScale(d, b, cfg.nearfield_window_factor, inj.far_field_scale);
        if (injected_entries) *injected_entries = k;
    }
    if (inj.reciprocity_scale != 1.0) {
        if (!injectReciprocityBreak(d, b, cfg.nearfield_window_factor, inj.reciprocity_scale)) {
            // A control that could not perturb anything proves nothing, and a
            // green exit after it would be read as "the detector works".
            std::cout << "  INJECTION ERROR: no two-sided entry to perturb on this "
                         "geometry. The reciprocity control did NOT run here.\n";
            g_injection_failed = true;
        }
    }
    return d;
}

// The OTHER in-engine builder. Selected in production by radiation.use_hmatrix
// (ptm_thermal_sim also switches to it automatically once the sparse matrix
// would not fit), and until 2026-08-23 gated by nothing at all -- which is how
// its near-field kernel came to be an UNBOUNDED copy of the sparse builder's.
//
// HMatrixViewFactor exposes no row accessor, only matvec, so the dense matrix
// is recovered one COLUMN at a time: y = F e_j gives y_i = F_ij. That is n
// matvecs, which is only affordable because these meshes are deliberately small
// (PTM-19: never rebuild a production .vf). It is also the honest thing to
// score -- matvec is exactly how the radiosity solver consumes this matrix, so
// the arm sees the same numbers the solver would.
// `stress_blocks` shrinks the octree leaf and the ACA floor so that ACA runs on
// SMALL, closely spaced clusters. It exists because at library defaults the ACA
// cross generators never meet a near pair at all -- measured on
// lunar_crater_80N (12,800 faces) the closest pair they evaluate sits at
// r / 4d = 2.04, and on the cap below they see none. An arm that asserts the
// cross generators handle a near pair correctly, run at defaults, would be
// green because it measured NOTHING. See arm [U].
DenseVF runHMatrixVF(const BuiltMesh& b, const radiation::RadiationConfig& cfg,
                     bool no_truncation = false, bool force_dense = false,
                     size_t* cross_near_pairs = nullptr, bool stress_blocks = false) {
    RTCDevice dev = rtcNewDevice(nullptr);
    RTCScene scene = makeScene(dev, *b.mesh);
    DenseVF d;
    d.n = b.mesh->faces.size();
    d.f.assign(d.n * d.n, 0.0);
    {
        // The builder prints its block statistics on std::cout even with
        // verbose off; capture so the verdict lines stay greppable.
        std::ostringstream sink;
        std::streambuf* saved = std::cout.rdbuf(sink.rdbuf());
        radiation::HMatrixViewFactor H;
        radiation::HMatrixConfig hc;             // library defaults
        hc.verbose = false;
        if (no_truncation) {
            // The kernel-isolation control: drive the low-rank tolerances to
            // round-off so ACA and SVD keep every mode. The BLOCK STRUCTURE and
            // every kernel call are unchanged; only the compression is
            // effectively removed, so a difference between this and the
            // production configuration is attributable to compression alone.
            hc.aca_tolerance = 1.0e-14;
            hc.compression_tolerance = 1.0e-14;
        }
        if (stress_blocks) {
            hc.min_block_size = 16;
            hc.aca_min_block_size = 16;
        }
        if (force_dense) {
            // The BLOCK-STRUCTURE-isolation control. min_block_size also sizes
            // the octree leaf, so making it >= n_faces collapses the whole
            // matrix onto ONE block, which buildBlock() then computes through
            // computeViewFactorBlock -- the exact, direct, near-field-refined
            // generator. Same builder, same kernel, same order rule, same
            // grazing cull, same occlusion rays; the only thing removed is the
            // hierarchical decomposition and with it the ACA cross generators.
            // A difference against this reference is therefore attributable to
            // the DECOMPOSITION and to nothing else.
            hc.min_block_size = static_cast<int>(d.n) + 1;
        }
        H.build(b.mesh, dev, scene, cfg, hc);
        if (cross_near_pairs) {
            *cross_near_pairs = radiation::HMatrixViewFactor::crossGeneratorNearPairs();
        }
        std::vector<double> x(d.n, 0.0), y;
        for (size_t j = 0; j < d.n; ++j) {
            std::fill(x.begin(), x.end(), 0.0);
            x[j] = 1.0;
            H.matvec(x, y);
            for (size_t i = 0; i < d.n; ++i) d.at(i, j) = y[i];
        }
        std::cout.rdbuf(saved);
    }
    rtcReleaseScene(scene);
    rtcReleaseDevice(dev);
    return d;
}

// The H-matrix builder's EXACT matrix: every entry from its dense-block
// generator, no hierarchical decomposition, no low-rank approximation. This is
// the reference arm [U] scores the assembled matrix against, so that the only
// variable between them is the block structure.
DenseVF exactHMatrixReference(const BuiltMesh& b, const radiation::RadiationConfig& cfg) {
    RTCDevice dev = rtcNewDevice(nullptr);
    RTCScene scene = makeScene(dev, *b.mesh);
    DenseVF d;
    d.n = b.mesh->faces.size();
    {
        std::ostringstream sink;
        std::streambuf* saved = std::cout.rdbuf(sink.rdbuf());
        radiation::HMatrixViewFactor H;
        radiation::HMatrixConfig hc;
        hc.verbose = false;
        H.build(b.mesh, dev, scene, cfg, hc);
        H.computeDenseReference(d.f);
        std::cout.rdbuf(saved);
    }
    rtcReleaseScene(scene);
    rtcReleaseDevice(dev);
    return d;
}

// Largest relative disagreement between the two in-engine builders, over the
// entries that carry the coupling. `floor_frac` of the largest entry is the
// significance cut: an entry at 1e-12 can disagree by 100% and mean nothing,
// and including it would make the statistic a report on the culling thresholds
// rather than on the kernel.
struct AgreeResult {
    double worst_rel = 0.0;
    double worst_near = 0.0;  // worst among pairs the SPARSE builder treats as M2
    double worst_far = 0.0;
    size_t compared = 0;
    size_t only_sparse = 0;   // significant in sparse, absent from H-matrix
    size_t only_hmat = 0;
    double mass_rel = 0.0;    // sum|a-h| / sum max(a,h), area-weighted
    double worst_ns = 0.0;    // Ns of the worst entry, so the regime is visible
};

AgreeResult agreement(const DenseVF& s, const DenseVF& h, const BuiltMesh& b,
                      double floor_frac, double window_factor) {
    AgreeResult r;
    const auto& C = b.mesh->face_centers;
    double peak = 0.0;
    for (size_t k = 0; k < s.f.size(); ++k) peak = std::max(peak, std::max(s.f[k], h.f[k]));
    const double floor_v = floor_frac * peak;
    double absdiff = 0.0, maxsum = 0.0;
    for (size_t i = 0; i < s.n; ++i) {
        for (size_t j = 0; j < s.n; ++j) {
            const double a = s.at(i, j), g = h.at(i, j);
            if (a < floor_v && g < floor_v) continue;
            absdiff += b.areas[i] * std::fabs(a - g);
            maxsum += b.areas[i] * std::max(a, g);
            if (a == 0.0) { ++r.only_hmat; continue; }
            if (g == 0.0) { ++r.only_sparse; continue; }
            ++r.compared;
            const double rel = std::fabs(a - g) / std::max(a, g);
            const double dx = C[j][0] - C[i][0], dy = C[j][1] - C[i][1], dz = C[j][2] - C[i][2];
            const double rr = std::sqrt(dx * dx + dy * dy + dz * dz);
            const double ns = rr / std::sqrt(std::sqrt(b.areas[i] * b.areas[j]));
            if (rel > r.worst_rel) { r.worst_rel = rel; r.worst_ns = ns; }
            if (radiation::isNearFieldPair(rr, b.areas[i], b.areas[j], window_factor))
                r.worst_near = std::max(r.worst_near, rel);
            else
                r.worst_far = std::max(r.worst_far, rel);
        }
    }
    r.mass_rel = (maxsum > 0.0) ? absdiff / maxsum : 0.0;
    return r;
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

// A FLOOR, for the arms whose failure mode is "this measured nothing".
void verdictAbove(const char* tag, const char* what, double measured, double floor_v,
                  bool& all_ok) {
    const bool ok = (measured >= floor_v);
    std::cout << "  [" << tag << "] " << std::left << std::setw(46) << what
              << std::right << " measured " << std::fixed << std::setprecision(0) << measured
              << "  floor " << floor_v << "   " << std::defaultfloat
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
    double tol_perface = 2.0;        // faceting residual of a flat-triangle cap;
                                     // SHRINKS with resolution (4.08/2.69/1.73/1.19%
                                     // at 342/648/1404/2750 faces under the old
                                     // one-sided M2). The symmetric M2 reads 1.45%
                                     // at the 2,750-face cap -- WORSE, and expected
                                     // to be: the old rule's near-field deficit
                                     // partly canceled the faceting error. 1.38x
                                     // headroom, NOT re-tuned.
    // ARM T band. NOT an accuracy tolerance -- see the banner the arm prints.
    // Measured ratios to the closed form, production calculator, 2026-08-23:
    //   symmetric double-area   m = 1  1.5166   (degenerate: pure centroid)
    //                           m = 2  1.2738
    //                           m = 3  1.1853   <- production default
    //                           m = 4  1.1400
    //   old one-sided rule      level 3  0.8192   level 6  0.9487
    // The edges are chosen by an explicit rule: pass the two production-
    // sanctioned orders (2 and 3), reject order 1 (no near-field treatment at
    // all), reject the old one-sided rule at ANY of its levels, and reject a 2x
    // inflation (2.37). The rule OVER-counts here for a structural reason -- an
    // open rule samples a positively singular kernel at finite r -- so the lower
    // edge sits just below the closed form rather than symmetric about it.
    // A REVIEW NOTE, because the spec for this arm said otherwise: the review
    // that commissioned it predicted "+1.3% after". That number is the
    // EDGE/VERTEX-TOUCHING population on lunar_crater_80N.ply (+1.258%), not
    // this geometry. The review's own reference script (q4_edge.py) prints
    // +18.530% for the double-area rule at m = 3 on perpendicular unit squares,
    // and the production calculator reproduces it to six digits.
    double tol_touch_lo = 0.98, tol_touch_hi = 1.40;
    double tol_recip = 1.0e-6;       // far field: M1 is reciprocal as an identity;
                                     // measured 1.3e-15, nine orders of headroom.
    double tol_recip_mass = 1.0e-6;  // was 1e-2 while M2 was one-sided. The
                                     // symmetric double-area M2 makes A_iF_ij =
                                     // A_jF_ji an algebraic identity in BOTH
                                     // regimes; measured 1.6e-16.
    double tol_recip_near = 1.0e-6;  // was 0.60, a regression bound on a known
                                     // non-reciprocal quadrature (worst pair
                                     // 0.4618). Now measured 6.6e-16: the near
                                     // field is held to the same machine-precision
                                     // standard as the far field, so any future
                                     // one-sided kernel is red on sight.
    double band_lo = 0.90, band_hi = 1.02;
    // ARM H tolerances. Every one is pinned from a measurement printed by the
    // arm itself, on the 1,404-face cap, 2026-08-23. They are DRIFT DETECTORS
    // on the H-matrix builder, not accuracy claims about it.
    //
    // Same 1% ruling as [I1]. The H-matrix reads +0.617% here and the sparse
    // builder reads +0.628% on the SAME mesh, so almost all of it is the cap's
    // faceting residual, common to both; 1.6x headroom.
    double tol_hmat_ingersoll = 1.0;
    // [H4] worst per-entry near-field disagreement with the sparse builder, in
    // the NO-TRUNCATION control. Not zero, and not a defect: the two builders
    // run the same kernel at a different ORDER on close pairs (sparse fixed at
    // nearfield_quadrature_order, H-matrix at max(that, ceil(4d/r))). The
    // measurement is the proof of that attribution -- raise the configured
    // floor toward the adaptive order and the disagreement collapses:
    //     m = 2  0.2811      m = 3  0.1643      m = 8  0.01373
    // 0.40 clears the worst production-sanctioned order (m = 2) by 1.4x and the
    // default (m = 3) by 2.4x. A SECOND copy of the integral, or a lost clamp,
    // moves this by orders of magnitude, not by 40%.
    double tol_hmat_agree = 0.40;
    // [H2b] mass-weighted reciprocity defect in the PRODUCTION configuration.
    // This one is compression, not the kernel: it reads 1.48e-3 with
    // aca_tolerance = 1e-3 and 1.16e-15 with the tolerance driven to 1e-14.
    // 1e-2 is ~7x the measurement and catches a 10x compression degradation.
    double tol_hmat_recip_mass = 1.0e-2;
    // [H4b] area-weighted mass difference against the sparse builder in the
    // production configuration: 0.617% measured (0.156% with compression off).
    // 2% is 3.2x.
    double tol_hmat_mass = 2.0e-2;
    // ARM U tolerances -- the stage-1 unification invariants. Pinned from the
    // arm's own printout on the same 1,404-face cap, 2026-08-23; see the arm.
    double min_cross_near_pairs = 100.0;
    double tol_unify_near = 1.0e-9;
    double tol_unify_mass = 2.0e-2;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--refine") refine = true;
        else if (a == "--tol-ingersoll" && i + 1 < argc) tol_ingersoll = std::atof(argv[++i]);
        else if (a == "--tol-disk" && i + 1 < argc) tol_disk = std::atof(argv[++i]);
        else if (a == "--tol-recip" && i + 1 < argc) tol_recip = std::atof(argv[++i]);
        else if (a == "--tol-recip-mass" && i + 1 < argc) tol_recip_mass = std::atof(argv[++i]);
        else if (a == "--tol-recip-near" && i + 1 < argc) tol_recip_near = std::atof(argv[++i]);
        else if (a == "--tol-perface" && i + 1 < argc) tol_perface = std::atof(argv[++i]);
        else if (a == "--tol-touch-lo" && i + 1 < argc) tol_touch_lo = std::atof(argv[++i]);
        else if (a == "--tol-touch-hi" && i + 1 < argc) tol_touch_hi = std::atof(argv[++i]);
        else if (a == "--tol-hmat-ingersoll" && i + 1 < argc)
            tol_hmat_ingersoll = std::atof(argv[++i]);
        else if (a == "--tol-hmat-agree" && i + 1 < argc)
            tol_hmat_agree = std::atof(argv[++i]);
        else if (a == "--tol-hmat-recip-mass" && i + 1 < argc)
            tol_hmat_recip_mass = std::atof(argv[++i]);
        else if (a == "--tol-hmat-mass" && i + 1 < argc)
            tol_hmat_mass = std::atof(argv[++i]);
        else if (a == "--tol-unify-near" && i + 1 < argc)
            tol_unify_near = std::atof(argv[++i]);
        else if (a == "--tol-unify-mass" && i + 1 < argc)
            tol_unify_mass = std::atof(argv[++i]);
        else if (a == "--min-cross-near" && i + 1 < argc)
            min_cross_near_pairs = std::atof(argv[++i]);
        else if (a == "--quad-order" && i + 1 < argc) {
            // strtol, not atoi: atoi maps both "0" and a typo to 0, which is the
            // "no override" sentinel, so a mistyped order would run at the
            // default and REPORT itself as the default.
            char* end = nullptr;
            const long v = std::strtol(argv[++i], &end, 10);
            if (end == argv[i] || *end != '\0' || v < 1 ||
                v > radiation::kMaxConfigNearFieldQuadOrder) {
                std::cerr << "--quad-order expects an integer 1.."
                          << radiation::kMaxConfigNearFieldQuadOrder << ", got '" << argv[i]
                          << "'\n";
                return 2;
            }
            g_quad_order_override = static_cast<int>(v);
        }
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
    {
        const radiation::RadiationConfig probe;
        const int m = (g_quad_order_override > 0) ? g_quad_order_override
                                                  : probe.nearfield_quadrature_order;
        std::cout << "  near-field quadrature order m = " << m << " ("
                  << (m * m) << " sample points per face, " << (m * m * m * m)
                  << " kernel evaluations per pair)"
                  << (g_quad_order_override > 0 ? "   [--quad-order OVERRIDE]"
                                                : "   [production default]")
                  << "\n";
    }
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
                const RecipResult rr = reciprocity(d, b, cfg.nearfield_window_factor);
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
                const RecipResult rr = reciprocity(d, b, cfg.nearfield_window_factor);
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
            applyQuadOrder(cfg);
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

    // ------------------------------------------------------------------------ ARM T
    {
        const double f12 = perpendicularRectF12(1.0, 1.0, 1.0);
        std::cout << "\n=== ARM T: perpendicular unit squares sharing an edge ===\n";
        std::cout << "  closed form (Howell C-14): A_1 F_12 = " << std::fixed
                  << std::setprecision(8) << f12 << "\n";
        std::cout << "  Built as 2 + 2 triangles, so the aggregate is reassembled from\n"
                  << "  four DIRECTED near-field pairs -- the shape a real mesh presents.\n"
                  << "  Arms I and D cannot see this regime: the cap's near pairs are\n"
                  << "  shallow and the disks have no near pairs at all, yet on a real\n"
                  << "  crater mesh edge/vertex-touching pairs carry 17.6% of the\n"
                  << "  near-field coupling mass.\n";
        std::cout << "\n  READ THIS BEFORE QUOTING THE NUMBER. This arm is a REGRESSION\n"
                  << "  BOUND and a SIGN detector, not an accuracy certificate. The\n"
                  << "  integrand has an integrable 1/r^2 singularity ON the shared edge,\n"
                  << "  so NO production-order area quadrature is close to it: the\n"
                  << "  symmetric double-area rule's error falls only like 1/m, roughly\n"
                  << "  +57%/m, and reaching 2% would need m ~ 30, i.e. ~10^6 kernel\n"
                  << "  evaluations per pair. The refinement table below is printed so\n"
                  << "  the 1/m law is visible and a change in convergence ORDER shows up\n"
                  << "  even when the gated ratio does not move.\n";

        struct Row { int m; double meas; };
        std::vector<Row> table;
        const radiation::RadiationConfig probe;
        const int prod_m = (g_quad_order_override > 0) ? g_quad_order_override
                                                       : probe.nearfield_quadrature_order;
        std::vector<int> orders = {1, 2, 3, 4};
        if (std::find(orders.begin(), orders.end(), prod_m) == orders.end())
            orders.push_back(prod_m);

        DenseVF d_prod;
        BuiltMesh b = buildTouchingSquares();
        radiation::RadiationConfig cfg_prod = noTruncationConfig(b.mesh->faces.size());
        size_t injected = 0;
        std::cout << "\n";
        for (int m : orders) {
            auto cfg = noTruncationConfig(b.mesh->faces.size());
            cfg.nearfield_quadrature_order = m;
            size_t inj_here = 0;
            DenseVF d = runProductionVF(b, cfg, inj, &inj_here);
            const double meas = areaWeightedCoupling(d, b, 0);
            table.push_back({m, meas});
            std::cout << "    m = " << m << " (" << std::setw(4) << (m * m * m * m)
                      << " evals/pair):  A_1 F_12 = " << std::fixed << std::setprecision(6)
                      << meas << "   ratio " << std::setprecision(4) << (meas / f12)
                      << "   err " << std::showpos << std::setprecision(3)
                      << 100.0 * (meas / f12 - 1.0) << "%" << std::noshowpos
                      << "   err*m " << std::setprecision(1)
                      << 100.0 * (meas / f12 - 1.0) * m
                      << (m == prod_m ? "   <-- gated" : "") << "\n";
            if (m == prod_m) { d_prod = d; injected = inj_here; }
        }

        double meas_prod = 0.0;
        for (const auto& r : table) if (r.m == prod_m) meas_prod = r.meas;
        const RecipResult rr = reciprocity(d_prod, b, cfg_prod.nearfield_window_factor);
        std::cout << "  " << b.mesh->faces.size() << " faces; " << rr.pairs_near
                  << " of " << (rr.pairs_near + rr.pairs_far)
                  << " two-sided pairs are near field (M2)";
        if (injected) std::cout << "   (injection touched " << injected << " entries)";
        std::cout << "\n";
        verdictBand("T1", "touching-pair A_1 F_12 / closed form", meas_prod / f12,
                    tol_touch_lo, tol_touch_hi, all_ok);
        reportReciprocity("touching squares", rr, tol_recip, tol_recip_mass,
                          tol_recip_near, all_ok, "R3a", "R3b", "R3c");
        const RowStats rs = rowSumStats(d_prod, b.group, -1);
        verdictBelow("S3", "max row sum (closure, must be <= 1)", rs.max, 1.0, all_ok);
    }

    // ------------------------------------------------------------------------ ARM W
    {
        std::cout << "\n=== ARM W: concave wedge, the physical bound F_ij <= 1 ===\n";
        std::cout << "  Two unit triangles sharing an edge at a closing dihedral. The\n"
                  << "  exact A_i F_ij rises toward A_i and never past it; a finite-order\n"
                  << "  area quadrature DIVERGES, because the sample pairs nearest the\n"
                  << "  shared edge have r -> 0 with cos_a cos_b -> 1. Unbounded, order 3\n"
                  << "  reads 1.07 at 45 deg and 19.2 at 10 deg against references 0.49\n"
                  << "  and 0.84 -- fabricated energy that the downstream row-sum\n"
                  << "  renormalization would absorb silently, zeroing that face's sky\n"
                  << "  view factor. computeSubdivision clamps at min(A_i, A_j) and counts\n"
                  << "  it; this arm is what proves the clamp is there.\n";
        double worst = 0.0;
        for (double deg : {90.0, 45.0, 30.0, 10.0}) {
            BuiltMesh b = buildConcaveWedge(deg);
            auto cfg = noTruncationConfig(b.mesh->faces.size());
            size_t injected = 0;
            DenseVF d = runProductionVF(b, cfg, inj, &injected);
            double mx = 0.0;
            for (size_t i = 0; i < d.n; ++i)
                for (size_t j = 0; j < d.n; ++j) mx = std::max(mx, d.at(i, j));
            worst = std::max(worst, mx);
            std::cout << "    dihedral " << std::setw(3) << static_cast<int>(deg)
                      << " deg:  max F_ij = " << std::fixed << std::setprecision(6) << mx
                      << "\n";
        }
        verdictBelow("W1", "max F_ij over a closing concave wedge", worst, 1.0, all_ok);
    }

    // ------------------------------------------------------------------------ ARM H
    // The H-matrix builder. Skipped under fault injection: the Injection hooks
    // perturb a DenseVF produced by runProductionVF, and this arm does not go
    // through it, so a "control" here would silently be a clean run.
    if (!inj.active()) {
        std::cout << "\n=== ARM H: the H-MATRIX builder, same geometry, same closed forms ===\n";
        std::cout << "  radiation.use_hmatrix selects a SECOND in-engine view-factor\n"
                  << "  builder, and ptm_thermal_sim also switches to it on its own once\n"
                  << "  the sparse matrix would not fit in memory -- so which estimator a\n"
                  << "  production run used has been decided by a memory threshold, with\n"
                  << "  no gate on the other side of it. Until 2026-08-23 its near-field\n"
                  << "  kernel was a private copy: already double-area, but with NO\n"
                  << "  physical bound (F_ij = 4.59 at a 10 degree dihedral, 16.75 at 5),\n"
                  << "  no canonical ordering, and a grazing cut at exactly 0 instead of\n"
                  << "  1e-6. It now calls the same radiation::nearFieldPairIntegral() the\n"
                  << "  sparse builder calls. These arms are what holds that.\n";

        // Same cap as arm [I], one refinement level down (1,404 faces): the
        // H-matrix is densified with one matvec per column, so the cost is
        // O(n * nnz) and the finest level would dominate the gate's runtime.
        BuiltMesh b = buildSphericalCap(0.2, 100.0, 20, 36);
        auto cfg = noTruncationConfig(b.mesh->faces.size());
        const double f_closed = ingersollTerrainVF(0.2 * 100.0, 50.0);

        DenseVF ds = runProductionVF(b, cfg, Injection{}, nullptr);
        DenseVF dh = runHMatrixVF(b, cfg);

        auto meanRowSum = [](const DenseVF& d) {
            double m = 0.0;
            for (size_t i = 0; i < d.n; ++i) {
                double row = 0.0;
                for (size_t j = 0; j < d.n; ++j) row += d.at(i, j);
                m += row;
            }
            return m / static_cast<double>(d.n);
        };
        const double mean_h = meanRowSum(dh);
        const double mean_s = meanRowSum(ds);

        std::cout << "  " << b.mesh->faces.size() << " faces, densified column by column\n";
        std::cout << "  the SPARSE builder on this same mesh reads " << std::fixed
                  << std::setprecision(6) << mean_s << " (" << std::showpos
                  << std::setprecision(4) << 100.0 * (mean_s / f_closed - 1.0) << "%"
                  << std::noshowpos << "), so the faceting residual of the cap is common\n"
                  << "  to both and only the DIFFERENCE below is about the builder\n";
        verdict("H1", "H-matrix terrain VF, mean over cap faces", mean_h, f_closed,
                tol_hmat_ingersoll, all_ok);

        // ------------------------------------------------------------------
        // The kernel and the COMPRESSION are two different questions, and
        // scoring them together produced a number that looked alarming and
        // meant nothing. Measured on this cap, production configuration
        // (aca_tolerance 1e-3): worst near-field reciprocity residual 13.08,
        // worst significant per-entry disagreement with the sparse builder
        // 1.62. The offending pair is two ADJACENT, near-coplanar cap facets
        // with F_ij = 2.2e-5 -- the sparse builder gives both directions
        // 2.222003e-05 exactly, and the H-matrix reconstructs 5.07e-08 and
        // 5.05e-09. Those are not M1 (2.81e-05), not M2, and not zero: they are
        // the low-rank residual left where a rank-truncated block had nothing
        // to say. So the arm runs the builder TWICE and gates the two questions
        // apart:
        //
        //   NO-TRUNCATION control  (aca/svd tolerance 1e-14) -> the kernel.
        //       Gated at machine precision. This is what "the two builders now
        //       share one estimator" has to mean to be worth asserting.
        //   PRODUCTION config      (library defaults)        -> compression.
        //       Gated on the AREA-WEIGHTED quantities, which is what the
        //       radiosity solve actually integrates, and reported per-entry
        //       without a verdict.
        // ------------------------------------------------------------------
        DenseVF dn = runHMatrixVF(b, cfg, /*no_truncation=*/true);

        // visibility_threshold is the sparse builder's own storage floor, so it
        // is the non-arbitrary significance cut here: an entry below it does
        // not exist in a production sparse .vf at all.
        const double floor_F = cfg.visibility_threshold;
        const RecipResult rp = reciprocity(dh, b, cfg.nearfield_window_factor, floor_F);
        const RecipResult rn = reciprocity(dn, b, cfg.nearfield_window_factor, floor_F);
        std::cout << "  H-matrix reciprocity, entries >= visibility_threshold ("
                  << std::scientific << std::setprecision(0) << floor_F << std::defaultfloat
                  << "): " << rp.pairs_far << " far-field, "
                  << rp.pairs_near << " near-field, " << rp.pairs_one_sided
                  << " one-sided, " << rp.pairs_below_floor << " below the floor\n"
                  << "    production  (aca_tolerance 1e-3): worst near " << std::scientific
                  << std::setprecision(3) << rp.worst_near << ", worst far " << rp.worst_far
                  << ", mass-weighted defect " << rp.defect_mass << "\n"
                  << "    no-truncation control:            worst near " << rn.worst_near
                  << ", worst far " << rn.worst_far
                  << ", mass-weighted defect " << rn.defect_mass << "\n"
                  << std::defaultfloat;
        verdictBelow("H2", "H-matrix near-field reciprocity, no truncation",
                     rn.worst_near, tol_recip_near, all_ok);

        // Name the offending pair in the NO-TRUNCATION matrix. A_i F_ij and
        // A_j F_ji leave the shared kernel as ONE number, so a near-field
        // asymmetry that survives with compression switched off means one of
        // the two directions never reached that kernel. The M1 column is the
        // discriminator: a direction that equals its own centroid estimate was
        // produced by a code path that does not refine.
        {
            size_t wi = 0, wj = 0;
            double worst = -1.0;
            const auto& C = b.mesh->face_centers;
            for (size_t i = 0; i < dn.n; ++i) {
                for (size_t j = i + 1; j < dn.n; ++j) {
                    const double a = b.areas[i] * dn.at(i, j);
                    const double c = b.areas[j] * dn.at(j, i);
                    if (a <= 0.0 || c <= 0.0) continue;
                    if (std::max(dn.at(i, j), dn.at(j, i)) < floor_F) continue;
                    const double dx = C[j][0] - C[i][0], dy = C[j][1] - C[i][1],
                                 dz = C[j][2] - C[i][2];
                    const double rr = std::sqrt(dx * dx + dy * dy + dz * dz);
                    if (!radiation::isNearFieldPair(rr, b.areas[i], b.areas[j],
                                                    cfg.nearfield_window_factor)) continue;
                    const double rel = std::fabs(a - c) / std::max(a, c);
                    if (rel > worst) { worst = rel; wi = i; wj = j; }
                }
            }
            if (worst > 0.0) {
                const auto& n_i = b.mesh->face_normals[wi];
                const auto& n_j = b.mesh->face_normals[wj];
                const double dx = C[wj][0] - C[wi][0], dy = C[wj][1] - C[wi][1],
                             dz = C[wj][2] - C[wi][2];
                const double r2 = dx * dx + dy * dy + dz * dz;
                const double r = std::sqrt(r2);
                const double ci = (n_i[0] * dx + n_i[1] * dy + n_i[2] * dz) / r;
                const double cj = -(n_j[0] * dx + n_j[1] * dy + n_j[2] * dz) / r;
                const double d_size = std::sqrt(b.areas[wi]) + std::sqrt(b.areas[wj]);
                std::cout << "    worst near pair (" << wi << "," << wj << "), Ns = "
                          << std::fixed << std::setprecision(3)
                          << r / std::sqrt(std::sqrt(b.areas[wi] * b.areas[wj]))
                          << ", cos_i " << std::setprecision(5) << ci << ", cos_j " << cj
                          << ", r/4d " << std::setprecision(3) << r / (4.0 * d_size) << "\n"
                          << std::scientific << std::setprecision(6)
                          << "      F_ij  H-mat(no trunc) " << dn.at(wi, wj) << "  sparse "
                          << ds.at(wi, wj) << "  M1 " << (b.areas[wj] / PI) * ci * cj / r2
                          << "\n"
                          << "      F_ji  H-mat(no trunc) " << dn.at(wj, wi) << "  sparse "
                          << ds.at(wj, wi) << "  M1 " << (b.areas[wi] / PI) * ci * cj / r2
                          << "\n" << std::defaultfloat;
            }
        }
        verdictBelow("H2b", "H-matrix reciprocity mass defect, production",
                     rp.defect_mass, tol_hmat_recip_mass, all_ok);

        const RowStats rsh = rowSumStats(dh, b.group, -1);
        verdictBelow("H3", "H-matrix max row sum (closure, must be <= 1)", rsh.max, 1.0,
                     all_ok);

        // Cross-agreement with the sparse builder. The two are NOT expected to
        // agree exactly even without compression: different near/far switches
        // (Ns <= 5 against r < 4(sqrt(A_i) + sqrt(A_j))), different quadrature
        // ORDER on close pairs (the H-matrix raises it adaptively above the
        // configured floor), a different caller-level grazing cull (0.01 on the
        // centroid cosines against 1e-6), and a different neighbor-selection
        // rule. What the shared kernel buys is that those are all DECLARED
        // differences in the callers, not a second copy of the integral.
        const AgreeResult ap = agreement(ds, dh, b, 1.0e-3, cfg.nearfield_window_factor);
        const AgreeResult an = agreement(ds, dn, b, 1.0e-3, cfg.nearfield_window_factor);
        std::cout << "  sparse vs H-matrix, entries above 1e-3 of the peak:\n"
                  << "    production     " << ap.compared << " compared, " << ap.only_sparse
                  << " sparse-only, " << ap.only_hmat << " H-matrix-only; area-weighted"
                  << " mass difference " << std::fixed << std::setprecision(4)
                  << 100.0 * ap.mass_rel << "%\n"
                  << "                   worst near " << std::scientific
                  << std::setprecision(3) << ap.worst_near << " (at Ns = " << std::fixed
                  << std::setprecision(2) << ap.worst_ns << "), worst far "
                  << std::scientific << std::setprecision(3) << ap.worst_far << "\n"
                  << "    no truncation  " << an.compared << " compared, " << an.only_sparse
                  << " sparse-only, " << an.only_hmat << " H-matrix-only; area-weighted"
                  << " mass difference " << std::fixed << std::setprecision(4)
                  << 100.0 * an.mass_rel << "%\n"
                  << "                   worst near " << std::scientific
                  << std::setprecision(3) << an.worst_near << ", worst far "
                  << an.worst_far << "\n" << std::defaultfloat;
        verdictBelow("H4", "sparse vs H-matrix worst near entry, no truncation",
                     an.worst_near, tol_hmat_agree, all_ok);
        verdictBelow("H4b", "sparse vs H-matrix area-weighted mass, production",
                     ap.mass_rel, tol_hmat_mass, all_ok);

        // The dihedral probe, on the H-matrix path. Before the port this read
        // 4.59 at 10 degrees on its own adaptive order; the shared kernel's
        // min(A_i, A_j) clamp is what bounds it, and the tripwire on stderr is
        // what makes the clamp audible.
        double worst_h = 0.0;
        for (double deg : {90.0, 45.0, 30.0, 10.0}) {
            BuiltMesh w = buildConcaveWedge(deg);
            auto wcfg = noTruncationConfig(w.mesh->faces.size());
            DenseVF dw = runHMatrixVF(w, wcfg);
            double mx = 0.0;
            for (size_t i = 0; i < dw.n; ++i)
                for (size_t j = 0; j < dw.n; ++j) mx = std::max(mx, dw.at(i, j));
            worst_h = std::max(worst_h, mx);
            std::cout << "    H-matrix, dihedral " << std::setw(3) << static_cast<int>(deg)
                      << " deg:  max F_ij = " << std::fixed << std::setprecision(6) << mx
                      << "\n";
        }
        verdictBelow("H5", "H-matrix max F_ij over a closing concave wedge", worst_h,
                     1.0, all_ok);
    }

    // ------------------------------------------------------------------------
    // ARM U -- one physics per PAIR, whichever block it lands in.
    //
    // Skipped under fault injection for the same reason arm H is: the hooks
    // perturb the dense matrix runProductionVF returns and never reach this
    // builder, so a "control" here would silently be a clean run.
    if (!inj.active()) {
        std::cout << "\n=== ARM U: same pair, same view factor, whichever BLOCK it lands in ===\n";
        std::cout << "  The H-matrix has THREE matrix-entry generators, not one:\n"
                  << "  computeViewFactorBlock (dense/small blocks), and\n"
                  << "  computeViewFactorRow / computeViewFactorColumn, the cross\n"
                  << "  generators ACA samples to build a low-rank factorization. The\n"
                  << "  first refines a close pair with the shared near-field\n"
                  << "  quadrature; until 2026-08-23 the other two did not, so whether a\n"
                  << "  pair got the near-field physics depended on where the OCTREE put\n"
                  << "  it -- not on the pair. This arm compares the production block\n"
                  << "  decomposition against the same builder collapsed onto a single\n"
                  << "  exact block (min_block_size >= n_faces), so the decomposition is\n"
                  << "  the only variable.\n"
                  << "  NOTE: reciprocity cannot see this. Both cross generators evaluate\n"
                  << "  the SAME M1 expression, and M1 satisfies A_i F_ij = A_j F_ji as an\n"
                  << "  algebraic identity, so an all-M1 block is perfectly reciprocal and\n"
                  << "  perfectly wrong. [H2] reads 4e-14 with the gap wide open.\n";

        BuiltMesh b = buildSphericalCap(0.2, 100.0, 20, 36);
        auto cfg = noTruncationConfig(b.mesh->faces.size());

        // The reference: the same builder, same kernel, same occlusion rays, no
        // hierarchical decomposition and no low-rank approximation anywhere.
        // NOT "build with min_block_size >= n_faces" -- that route still runs
        // the single block through tryCompressToLowRank(), which caps the rank
        // at max_svd_rank (50) and reports success even when the requested
        // tolerance was not met. On crater80N that reference came back with
        // NEGATIVE view factors and a fabricated 17% mass difference.
        DenseVF dref = exactHMatrixReference(b, cfg);

        size_t near_exposed = 0, near_exposed_prod = 0;
        DenseVF dblk = runHMatrixVF(b, cfg, /*no_truncation=*/true, /*force_dense=*/false,
                                    &near_exposed, /*stress_blocks=*/true);
        DenseVF dprod = runHMatrixVF(b, cfg, /*no_truncation=*/false, /*force_dense=*/false,
                                     &near_exposed_prod, /*stress_blocks=*/true);
        // What the DEFAULT block layout exposes, reported so the stressed
        // numbers above are never mistaken for production's.
        size_t near_default = 0;
        (void)runHMatrixVF(b, cfg, /*no_truncation=*/true, /*force_dense=*/false,
                           &near_default, /*stress_blocks=*/false);
        const size_t cross_default = radiation::HMatrixViewFactor::crossGeneratorPairs();
        const double margin_default = radiation::HMatrixViewFactor::crossGeneratorMinROver4d();

        // The comparison window is the H-MATRIX's own near-field window,
        // r < 4 (sqrt(A_i) + sqrt(A_j)) -- the one its generators branch on --
        // not the sparse builder's Ns <= 5. Scoring a builder against a rule it
        // does not use would put the near/far-window question (a separate,
        // deliberate difference) inside this verdict.
        struct BlockAgree {
            double worst_near = 0.0;
            double mass_near = 0.0;     // area-weighted sum|a-b| / sum max(a,b)
            size_t near_compared = 0;
            size_t near_only_ref = 0;   // present in the reference, absent from the test
            size_t near_only_test = 0;
        };
        const double floor_F = cfg.visibility_threshold;
        auto compareNear = [&](const DenseVF& ref, const DenseVF& test) {
            BlockAgree g;
            const auto& C = b.mesh->face_centers;
            double absdiff = 0.0, maxsum = 0.0;
            for (size_t i = 0; i < ref.n; ++i) {
                for (size_t j = 0; j < ref.n; ++j) {
                    if (i == j) continue;
                    const double a = ref.at(i, j), t = test.at(i, j);
                    if (a < floor_F && t < floor_F) continue;
                    const double dx = C[j][0] - C[i][0], dy = C[j][1] - C[i][1],
                                 dz = C[j][2] - C[i][2];
                    const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
                    const double d4 = 4.0 * (std::sqrt(b.areas[i]) + std::sqrt(b.areas[j]));
                    if (r >= d4) continue;                       // far field, not this arm
                    absdiff += b.areas[i] * std::fabs(a - t);
                    maxsum += b.areas[i] * std::max(a, t);
                    if (a == 0.0) { ++g.near_only_test; continue; }
                    if (t == 0.0) { ++g.near_only_ref; continue; }
                    ++g.near_compared;
                    g.worst_near = std::max(g.worst_near, std::fabs(a - t) / std::max(a, t));
                }
            }
            g.mass_near = (maxsum > 0.0) ? absdiff / maxsum : 0.0;
            return g;
        };

        const BlockAgree gb = compareNear(dref, dblk);
        const BlockAgree gp = compareNear(dref, dprod);

        std::cout << "  " << b.mesh->faces.size() << " faces; near-field window r < 4d,"
                  << " entries >= visibility_threshold\n"
                  << "    at LIBRARY DEFAULTS the ACA cross generators see " << near_default
                  << " near pairs here (" << cross_default << " pairs of any kind";
        if (cross_default > 0) {
            std::cout << ", closest approach r/4d = " << std::fixed << std::setprecision(3)
                      << margin_default;
        } else {
            std::cout << ", so ACA never runs on this mesh at defaults";
        }
        std::cout << ")\n"
                  << "    -- so this arm runs\n"
                  << "    a STRESSED block layout (min_block_size 16, aca_min_block_size 16)"
                  << " to put real near pairs through them\n"
                  << "    stressed: " << near_exposed << " near pairs (no truncation), "
                  << near_exposed_prod << " (production tolerances)\n"
                  << "    vs the single-block reference, NO TRUNCATION: "
                  << gb.near_compared << " compared, " << gb.near_only_ref
                  << " reference-only, " << gb.near_only_test << " test-only;"
                  << " worst " << std::scientific << std::setprecision(3) << gb.worst_near
                  << ", area-weighted mass " << std::fixed << std::setprecision(4)
                  << 100.0 * gb.mass_near << "%\n"
                  << "    vs the single-block reference, PRODUCTION config:  "
                  << gp.near_compared << " compared, " << gp.near_only_ref
                  << " reference-only, " << gp.near_only_test << " test-only;"
                  << " worst " << std::scientific << std::setprecision(3) << gp.worst_near
                  << ", area-weighted mass " << std::fixed << std::setprecision(4)
                  << 100.0 * gp.mass_near << "%\n" << std::defaultfloat;

        // [U0] is what stops [U1] from being green because it measured nothing.
        // The per-entry arm of the cross-agreement gate shipped VACUOUS once
        // already (no entry reached its resolution threshold, "0 <= 0 PASS").
        verdictAbove("U0", "near pairs reaching the ACA cross generators",
                     static_cast<double>(near_exposed), min_cross_near_pairs, all_ok);
        verdictBelow("U1", "block-decomposition dependence of a near pair, no truncation",
                     gb.worst_near, tol_unify_near, all_ok);
        verdictBelow("U2", "block-decomposition dependence, near-field mass, production",
                     gp.mass_near, tol_unify_mass, all_ok);
    }

    // ------------------------------------------------------------------------
    std::cout << "\n=== NOT COVERED BY THIS PROGRAM ===\n";
    std::cout << "  * the OFFLINE sweep + assembler (mc_ref_pp + assemble_vf_haworth.py):\n"
              << "    gated separately by scripts/ptm12_diag/test_vf_assembler_gate.py,\n"
              << "    and cross-checked against this builder by\n"
              << "    scripts/ptm12_diag/test_vf_cross_agreement.py.\n";
    std::cout << "  * the Monte-Carlo ray-gather builder (mc_view_factor.cpp), selected\n"
              << "    by use_monte_carlo_vf: still UNGATED.\n";
    std::cout << "  * the H-matrix builder's FAR field. Arm H gates its near-field\n"
              << "    kernel, its closure and its agreement with the sparse builder, but\n"
              << "    its ACA low-rank compression is scored only by the printed\n"
              << "    far-field reciprocity residual, not by a verdict.\n";
    std::cout << "  * fault injection does not reach arm H: the hooks perturb the dense\n"
              << "    matrix runProductionVF returns, so arm H is SKIPPED under\n"
              << "    PTM_VFGATE_INJECT_*, not silently passed.\n";
    std::cout << "  * the GPU BVH visibility path (benchmark_gpu_visibility): the\n"
              << "    calculator above uses Embree on the CPU in every configuration.\n";
    std::cout << "  * the radiosity SOLVE. This program scores geometry only.\n";

    if (g_injection_failed) {
        std::cout << "\n[test_vf_production] FAULT INJECTION REQUESTED BUT NOT APPLIED on at\n"
                  << "[test_vf_production] least one arm (see INJECTION ERROR above). A\n"
                  << "[test_vf_production] control that perturbed nothing is not a control.\n";
        std::cerr << "test_vf_production: requested fault injection could not be applied\n";
        return 3;
    }

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
