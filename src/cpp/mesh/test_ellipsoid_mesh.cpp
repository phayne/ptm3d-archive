// test_ellipsoid_mesh -- geometry gate for MeshUtils::createEllipsoidMesh.
//
// Why this test exists
// --------------------
// createEllipsoidMesh() builds a UNIT icosphere and then displaces every vertex
// onto the tri-axial ellipsoid. The face centers and face normals must be
// recomputed from the DISPLACED vertices. calculateFaceProperties() carries a
// skip-cache ("already calculated -> return") that keys only on array SIZE, and
// the displacement does not change the face count -- so when the object already
// carried face properties from an earlier setupMesh() (which is exactly what the
// production call order in ptm_thermal_sim.cpp does), the recompute was skipped
// and the run shipped the UNDISPLACED SPHERE's centers, normals and latitudes.
//
// Three independent checks, each of which fails loudly on its own:
//
//   A. ORDER INDEPENDENCE. The same createEllipsoidMesh() call on a FRESH
//      MeshUtils (empty face arrays, so the skip-cache cannot fire) is the
//      oracle. The production order must agree with it bit-for-bit. This is the
//      check that discriminates the defect: it needs no tolerance and no
//      analytic model, only the observation that a mesh generator's output may
//      not depend on what the object was doing beforehand.
//
//   B. ANALYTIC SURFACE NORMAL. For the ellipsoid x^2/a^2 + y^2/b^2 + z^2/c^2 = 1
//      the outward normal is parallel to grad = (x/a^2, y/b^2, z/c^2). That
//      gradient direction is invariant under a radial rescale of the evaluation
//      point -- grad(lambda*p) = lambda*grad(p) -- so evaluating it at the
//      triangle CENTROID (which sits just inside the surface) gives exactly the
//      analytic normal at the centroid's radial projection onto the surface, with
//      no projection step and no approximation. Discretization error is therefore
//      the only residual, and it must shrink under refinement.
//
//   C. VERTEX CONSISTENCY. Face centers must equal the mean of their own three
//      vertices, and face areas must equal the triangle areas of those vertices.
//      A stale array fails this by construction.
//
// The analytic check also answers the physics question the flag turns on: on a
// SPHERE the radial direction IS the surface normal, which is why substituting
// it is legitimate there. On an ELLIPSOID it is not, and this test prints the
// radial-vs-analytic angle so the size of that error is on the record.
//
// Usage:  test_ellipsoid_mesh [--dump <prefix>] [--resolution N]
//         --dump writes <prefix>_res<N>.csv with per-face geometry (for
//         before/after differencing); the pass/fail gates run regardless.

#include "mesh_utils.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using ptm::Mesh;
using ptm::MeshUtils;

namespace {

// The tri-axial test body. Deliberately strongly non-spherical: a near-sphere
// would make the stale-sphere geometry look almost right and the test blind.
constexpr double kA = 7.0;   // km, semi-axis along X
constexpr double kB = 5.0;   // km, semi-axis along Y
constexpr double kC = 3.0;   // km, semi-axis along Z

double norm3(const std::array<double, 3>& v) {
    return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

std::array<double, 3> unit3(std::array<double, 3> v) {
    const double n = norm3(v);
    if (n > 0.0) { v[0] /= n; v[1] /= n; v[2] /= n; }
    return v;
}

double dot3(const std::array<double, 3>& a, const std::array<double, 3>& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

// Angle in degrees between two unit vectors, computed via the chord length so
// it stays accurate for small angles (acos of a near-1 dot product does not).
double angle_deg(const std::array<double, 3>& u, const std::array<double, 3>& v) {
    const std::array<double, 3> d{{u[0] - v[0], u[1] - v[1], u[2] - v[2]}};
    const std::array<double, 3> s{{u[0] + v[0], u[1] + v[1], u[2] + v[2]}};
    return 2.0 * std::atan2(norm3(d), norm3(s)) * 180.0 / M_PI;
}

// Outward analytic normal of the ellipsoid at (or radially below/above) p.
std::array<double, 3> analytic_normal(const std::array<double, 3>& p) {
    return unit3({{p[0] / (kA * kA), p[1] / (kB * kB), p[2] / (kC * kC)}});
}

double geodetic_lat_deg(const std::array<double, 3>& c) {
    const double r = norm3(c);
    return (r > 0.0) ? std::asin(c[2] / r) * 180.0 / M_PI : 0.0;
}

double lon_deg(const std::array<double, 3>& c) {
    return std::atan2(c[1], c[0]) * 180.0 / M_PI;
}

struct Stats {
    double max = 0.0;
    double mean = 0.0;
    size_t n = 0;
    void add(double v) { max = std::max(max, v); mean += v; ++n; }
    double avg() const { return n ? mean / static_cast<double>(n) : 0.0; }
};

// Build the ellipsoid the way PRODUCTION does: set the body radius, then
// setMeshResolution() (which runs a full setupMesh() and populates the face
// arrays for a SPHERE), then createEllipsoidMesh() at the same resolution.
// Mirrors ptm_thermal_sim.cpp:12334-12460.
std::shared_ptr<Mesh> build_production_order(int resolution) {
    MeshUtils mu;
    mu.setBodyRadius(std::cbrt(kA * kB * kC));   // BodyShape::meanRadiusKm()
    mu.setMeshResolution(resolution);            // -> setupMesh(), fills face arrays
    mu.createEllipsoidMesh(resolution, kA, kB, kC);
    auto m = mu.getMesh();
    // Areas are not stored on Mesh; capture them from the same object.
    return m;
}

// The oracle: identical call on a FRESH object, so the skip-cache cannot fire.
std::shared_ptr<Mesh> build_fresh_order(int resolution) {
    MeshUtils mu;
    mu.setBodyRadius(std::cbrt(kA * kB * kC));
    mu.createEllipsoidMesh(resolution, kA, kB, kC);
    return mu.getMesh();
}

// Triangle area straight from the mesh's own vertices (what getFaceArea does).
double tri_area(const Mesh& m, size_t i) {
    const auto& f = m.faces[i];
    const auto& v0 = m.vertices[f[0]];
    const auto& v1 = m.vertices[f[1]];
    const auto& v2 = m.vertices[f[2]];
    const std::array<double, 3> e1{{v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]}};
    const std::array<double, 3> e2{{v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]}};
    const std::array<double, 3> cr{{e1[1] * e2[2] - e1[2] * e2[1],
                                    e1[2] * e2[0] - e1[0] * e2[2],
                                    e1[0] * e2[1] - e1[1] * e2[0]}};
    return 0.5 * norm3(cr);
}

void dump_csv(const std::string& path, const Mesh& m) {
    std::ofstream out(path);
    if (!out) {
        std::cerr << "  (could not open dump file " << path << ")\n";
        return;
    }
    out << std::setprecision(17);
    out << "face,cx,cy,cz,nx,ny,nz,area,lat_deg,lon_deg\n";
    for (size_t i = 0; i < m.faces.size(); ++i) {
        const auto& c = m.face_centers[i];
        const auto& n = m.face_normals[i];
        out << i << ',' << c[0] << ',' << c[1] << ',' << c[2] << ','
            << n[0] << ',' << n[1] << ',' << n[2] << ','
            << tri_area(m, i) << ','
            << geodetic_lat_deg(c) << ',' << lon_deg(c) << '\n';
    }
    std::cout << "  dumped " << m.faces.size() << " faces -> " << path << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    int resolution = 4;
    std::string dump_prefix;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--dump") == 0 && i + 1 < argc) {
            dump_prefix = argv[++i];
        } else if (std::strcmp(argv[i], "--resolution") == 0 && i + 1 < argc) {
            resolution = std::atoi(argv[++i]);
        } else {
            std::cerr << "usage: test_ellipsoid_mesh [--dump <prefix>] [--resolution N]\n";
            return 2;
        }
    }

    std::cout << "=== test_ellipsoid_mesh ===\n"
              << "Ellipsoid semi-axes a,b,c = " << kA << ", " << kB << ", " << kC << " km"
              << "   (mean radius " << std::cbrt(kA * kB * kC) << " km)\n"
              << "Resolution " << resolution << "\n\n";

    bool ok = true;

    auto prod = build_production_order(resolution);
    auto fresh = build_fresh_order(resolution);
    if (!prod || !fresh || prod->faces.empty()) {
        std::cerr << "FAIL: mesh construction produced no faces\n";
        return 1;
    }

    // ---------------------------------------------------------------- A ----
    // Order independence. No tolerance: the two call orders run the same code
    // on the same inputs, so any difference at all is the skip-cache firing.
    std::cout << "[A] order independence (production order vs fresh object)\n";
    if (prod->faces.size() != fresh->faces.size()) {
        std::cerr << "FAIL: face counts differ (" << prod->faces.size()
                  << " vs " << fresh->faces.size() << ")\n";
        return 1;
    }
    size_t n_center_diff = 0, n_normal_diff = 0, n_lat_diff = 0;
    Stats center_err, normal_err, lat_err;
    for (size_t i = 0; i < prod->faces.size(); ++i) {
        const auto& cp = prod->face_centers[i];
        const auto& cf = fresh->face_centers[i];
        const double dc = norm3({{cp[0] - cf[0], cp[1] - cf[1], cp[2] - cf[2]}});
        if (dc != 0.0) { ++n_center_diff; center_err.add(dc); }

        const double dn = angle_deg(unit3(prod->face_normals[i]), unit3(fresh->face_normals[i]));
        if (prod->face_normals[i] != fresh->face_normals[i]) { ++n_normal_diff; }
        normal_err.add(dn);

        const double dl = std::fabs(prod->latitudes[i] - fresh->latitudes[i]);
        if (dl != 0.0) { ++n_lat_diff; }
        lat_err.add(dl);
    }
    std::cout << "    face_centers differing : " << n_center_diff << " / " << prod->faces.size()
              << "   max |dc| = " << center_err.max << " km,  mean " << center_err.avg() << " km\n"
              << "    face_normals differing : " << n_normal_diff << " / " << prod->faces.size()
              << "   max angle = " << normal_err.max << " deg, mean " << normal_err.avg() << " deg\n"
              << "    latitudes  differing : " << n_lat_diff << " / " << prod->faces.size()
              << "   max |dlat| = " << lat_err.max << " deg, mean " << lat_err.avg() << " deg\n";
    if (n_center_diff || n_normal_diff || n_lat_diff) {
        std::cout << "    -> FAIL: createEllipsoidMesh output depends on prior object state\n";
        ok = false;
    } else {
        std::cout << "    -> PASS\n";
    }

    // Face areas come from getFaceArea(), which reads the VERTICES directly and
    // never consults the cached arrays, so they are expected to match in both
    // orders even when the defect is present. Stated explicitly rather than
    // assumed: this is the array the skip-cache does NOT corrupt.
    size_t n_area_diff = 0;
    for (size_t i = 0; i < prod->faces.size(); ++i)
        if (tri_area(*prod, i) != tri_area(*fresh, i)) ++n_area_diff;
    std::cout << "    face areas differing   : " << n_area_diff << " / " << prod->faces.size()
              << "   (areas are derived from vertices on demand, not cached)\n\n";

    // ---------------------------------------------------------------- B ----
    // Analytic surface normal, evaluated on the PRODUCTION mesh.
    std::cout << "[B] analytic ellipsoid normal, production-order mesh\n";
    Stats vs_analytic, radial_vs_analytic;
    for (size_t i = 0; i < prod->faces.size(); ++i) {
        const auto& c = prod->face_centers[i];
        const auto na = analytic_normal(c);
        vs_analytic.add(angle_deg(unit3(prod->face_normals[i]), na));
        radial_vs_analytic.add(angle_deg(unit3(c), na));
    }
    std::cout << "    stored normal vs analytic : max " << vs_analytic.max
              << " deg, mean " << vs_analytic.avg() << " deg\n"
              << "    RADIAL       vs analytic : max " << radial_vs_analytic.max
              << " deg, mean " << radial_vs_analytic.avg() << " deg"
              << "   <- the error a radial-normal substitution would carry here\n";
    // The gate: the stored normal must be an order of magnitude closer to the
    // analytic normal than the radial direction is. On this 7/5/3 body the
    // radial error is tens of degrees, so a factor of 10 is not a tight fit --
    // it is a floor that only a genuinely surface-fitted normal can clear.
    const double kMargin = 10.0;
    if (!(vs_analytic.max * kMargin < radial_vs_analytic.max)) {
        std::cout << "    -> FAIL: stored normals are not surface-fitted\n";
        ok = false;
    } else {
        std::cout << "    -> PASS (stored normal is "
                  << (radial_vs_analytic.max / std::max(vs_analytic.max, 1e-300))
                  << "x closer to analytic than radial is)\n";
    }

    // Convergence: a genuine discretization residual must fall under refinement.
    // A stale or misassigned normal does not. Ratio is reported, not asserted at
    // a fixed order, because the triangle shape distribution also changes.
    std::cout << "    refinement (production order), max deviation from analytic:\n";
    double prev = 0.0;
    bool monotone = true;
    for (int r = 2; r <= std::max(resolution, 5); ++r) {
        auto m = build_production_order(r);
        Stats s;
        for (size_t i = 0; i < m->faces.size(); ++i)
            s.add(angle_deg(unit3(m->face_normals[i]), analytic_normal(m->face_centers[i])));
        std::cout << "      res " << r << "  (" << m->faces.size() << " faces)  max "
                  << std::setw(12) << s.max << " deg  mean " << std::setw(12) << s.avg() << " deg";
        if (prev > 0.0) {
            const double ratio = prev / std::max(s.max, 1e-300);
            std::cout << "   ratio " << ratio;
            if (ratio < 1.5) monotone = false;
        }
        std::cout << "\n";
        prev = s.max;
    }
    if (!monotone) {
        std::cout << "    -> FAIL: deviation from the analytic normal does not shrink"
                     " under refinement\n";
        ok = false;
    } else {
        std::cout << "    -> PASS (deviation shrinks under refinement)\n";
    }
    std::cout << "\n";

    // ---------------------------------------------------------------- C ----
    std::cout << "[C] internal consistency of the production-order mesh\n";
    Stats centroid_err;
    for (size_t i = 0; i < prod->faces.size(); ++i) {
        const auto& f = prod->faces[i];
        const auto& v0 = prod->vertices[f[0]];
        const auto& v1 = prod->vertices[f[1]];
        const auto& v2 = prod->vertices[f[2]];
        const std::array<double, 3> mid{{(v0[0] + v1[0] + v2[0]) / 3.0,
                                         (v0[1] + v1[1] + v2[1]) / 3.0,
                                         (v0[2] + v1[2] + v2[2]) / 3.0}};
        const auto& c = prod->face_centers[i];
        centroid_err.add(norm3({{c[0] - mid[0], c[1] - mid[1], c[2] - mid[2]}}));
    }
    std::cout << "    |face_center - vertex centroid| : max " << centroid_err.max
              << " km, mean " << centroid_err.avg() << " km\n";
    if (centroid_err.max > 1e-12) {
        std::cout << "    -> FAIL: face centers are not the centroids of their own vertices\n";
        ok = false;
    } else {
        std::cout << "    -> PASS\n";
    }

    // Every vertex must satisfy the ellipsoid equation; this pins the geometry
    // itself, independent of the cached arrays.
    Stats surf_err;
    for (const auto& v : prod->vertices)
        surf_err.add(std::fabs(v[0] * v[0] / (kA * kA) + v[1] * v[1] / (kB * kB) +
                               v[2] * v[2] / (kC * kC) - 1.0));
    std::cout << "    |x^2/a^2+y^2/b^2+z^2/c^2 - 1| over vertices : max " << surf_err.max << "\n";
    if (surf_err.max > 1e-12) {
        std::cout << "    -> FAIL: vertices are not on the requested ellipsoid\n";
        ok = false;
    } else {
        std::cout << "    -> PASS\n";
    }

    // The flag consumers read (PTM-221). On an ellipsoid the surface normal is
    // NOT radial, so the analytic-radial-substitution record must be false and
    // the PTM_ILLUM_TRUE_NORMALS override must be allowed to engage.
    std::cout << "\n[flag] analytic_radial_normals = "
              << (prod->analytic_radial_normals ? "true" : "false")
              << "   (must be false: an ellipsoid's normal is not radial -- see the"
                 " RADIAL-vs-analytic figure above)\n";
    if (prod->analytic_radial_normals) {
        std::cout << "    -> FAIL\n";
        ok = false;
    } else {
        std::cout << "    -> PASS\n";
    }

    if (!dump_prefix.empty()) {
        std::cout << "\n";
        dump_csv(dump_prefix + "_prod_res" + std::to_string(resolution) + ".csv", *prod);
        dump_csv(dump_prefix + "_fresh_res" + std::to_string(resolution) + ".csv", *fresh);
    }

    std::cout << "\n=== " << (ok ? "ALL CHECKS PASSED" : "FAILURES ABOVE") << " ===\n";
    return ok ? 0 : 1;
}
