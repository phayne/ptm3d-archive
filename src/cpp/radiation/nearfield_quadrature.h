#pragma once

// Near-field (M2) view-factor quadrature -- ONE kernel, shared by every builder
// that needs it.
//
// Before 2026-08-23 the sparse in-engine builder (view_factor.cpp) and the
// H-matrix builder (hmatrix_view_factor.cpp) each carried their own copy of
// this integral, and they drifted: the sparse copy was a ONE-SIDED rule (point
// emitter, subdivided receiver) that read -21.7% on edge/vertex-touching pairs,
// while the H-matrix copy was already double-area but had no physical bound, so
// it fabricated energy on a concave crease (F_ij = 4.59 at a 10 degree
// dihedral). Two copies, two different defects, one gate between them. Paul's
// standing priority is one physics across all paths; this header is the small
// version of that -- the estimator now exists once and the callers differ only
// in how they choose the order and where they issue the occlusion ray.
//
// NOT shared, deliberately, because they are genuinely per-builder:
//   * occlusion. The sparse builder issues one canonical centroid-to-centroid
//     Embree ray inside computeSubdivision; the H-matrix batches its rays in
//     the block assembler and cannot ask for one here.
//   * bowl_crater_ptm.cpp keeps a THIRD, harness-local replica on purpose: its
//     KPROBE mode isolates the kernel from the occlusion and pair-set, and it
//     also carries the RETIRED one-sided rule for attribution. It is documented
//     as a replica there and must not be pointed at this header, or the probe
//     stops being an independent reading.

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <limits>

#include "radiation_config.h"

namespace radiation {

// ---------------------------------------------------------------------------
// The near/far switch -- ONE rule, both in-engine builders.
//
//   near  <=>  r < w (sqrt(A_i) + sqrt(A_j)),   w = nearfield_window_factor
//
// r is the centroid separation. The length scale is the SUM of the two face
// sizes, so the pair is judged by the LARGER of the two solid angles, which is
// what actually governs the centroid approximation's error.
//
// Until 2026-08-23 the two builders used different rules and the sparse one
// used the GEOMETRIC MEAN: Ns = r / (A_i A_j)^(1/4) <= adaptive_ns_threshold
// (5). The geometric mean cannot see an area contrast. Measured M1 error
// against the same quadrature run to convergence (m = 32, ~1.05e6 evaluations
// per pair; ~/projects/ada_vfunify_2026-08-23/probe/window_curve.py):
//
//   A_j/A_i        at the OLD cut (Ns = 5)     at THIS cut (w = 4)
//   1              Ns 5,      M1 +3.49%        Ns 8,      M1 +1.38%
//   100            Ns 5,      M1 +16.7%        Ns 12.6,   M1 +2.3%
//   10000          Ns 5,      M1 +150%         Ns 40,     M1 ~+2%
//
// so the old rule handed the far-field estimator pairs it was wrong about by
// 150 percent on a 100:1 linear size contrast -- the core-plus-ring meshes this
// project builds are exactly that. The error is one-signed (M1 always
// OVER-estimates a near pair), so widening the window makes terrain coupling
// SMALLER, not larger.
//
// w = 4 is the H-matrix builder's historical constant, kept because it is the
// SAFE direction: M2 converges to M1 in the far field, so an over-wide window
// costs time and never accuracy, while an over-narrow one is a bias. It leaves
// 1.38% of M1 error at the cut for an equal-area pair.
inline bool isNearFieldPair(double r, double A_i, double A_j, double window_factor) {
    return r < window_factor * (std::sqrt(A_i) + std::sqrt(A_j));
}

// ---------------------------------------------------------------------------
// The occlusion ray -- ONE rule, both in-engine builders.
//
// A centroid-to-centroid visibility segment has to start clear of the source
// facet and stop clear of the target facet, or the ray reports the pair
// occluded by the two faces it is asking about. Until 2026-08-23 both builders
// did that with
//
//     offset = clamp(max(0.001 r, 0.5 m), .., 0.1 r)   at each end, and
//     "a segment shorter than 0.1 m after both margins is VISIBLE"
//
// which is three absolute lengths in meters in a code base whose facets run
// from 2.48 mm (the moon3d polar patch) to 80 m (A3GT rings). On a sub-meter
// mesh the 0.1 m auto-pass swallows the whole segment: EVERY pair closer than
// about 0.125 m -- fifty facet widths -- skipped the occlusion test entirely
// and was recorded visible. See [[ptm-shadow-ray-offset-scale-bug]], and note
// that this is the VIEW-FACTOR path, not the solar-shadow one; the two failed
// the same way for the same reason and must be fixed separately.
//
// The replacement is relative to the geometry, with the constants argued from
// it rather than tuned:
//
//   * The margin's job is to leave the source facet's own footprint. A
//     triangle's centroid is about 0.3 L from its edges for L = sqrt(2 A), so
//     any margin up to ~0.25 L moves the origin WITHIN the source facet's own
//     cell. Nothing can be vaulted there that is not the source facet itself,
//     which is the one thing the margin exists to skip. That is the ceiling.
//     kRayMarginFaceFrac = 0.02 sits well under it and still clears the
//     triangle's float32 footprint on every mesh in this project.
//
//   * Embree traverses in float32. If the margin is smaller than the
//     quantization of the ray origin's own coordinates it does not exist: the
//     cast to float rounds it away. kRayMarginUlps ulps of the larger endpoint
//     coordinate is therefore a FLOOR, not a tuning knob. It is what the old
//     0.5 m actually was -- 4 ulps of a lunar selenocentric coordinate, whose
//     ulp is 0.125 m -- discovered rather than declared.
//
//   * The pair is "touching, assume visible" when the segment left after both
//     margins is below that same float32 floor, i.e. when float32 cannot
//     resolve the gap. Not when it is below a tenth of a meter.
//
// If the floor exceeds the ceiling the scene cannot be ray-traced in float32 at
// those coordinates at all, and the caller is told (unresolvable) rather than
// handed a silently meaningless answer. Recentering the scene is the cure; the
// engine already does it, `vf_dump_sparse` does not.
inline constexpr double kRayMarginFaceFrac = 0.02;   // margin as a fraction of sqrt(2A)
inline constexpr double kRayMarginFaceCap  = 0.25;   // never leave the source facet's cell
inline constexpr double kRayMarginDistCap  = 0.40;   // and never eat the whole segment
inline constexpr double kRayMarginUlps     = 8.0;    // float32 quantization floor

// One float32 ulp at the largest coordinate magnitude the ray will touch.
inline double sceneUlp(const std::array<double, 3>& a,
                       const std::array<double, 3>& b) {
    double m = 0.0;
    for (int k = 0; k < 3; ++k) {
        m = std::max(m, std::abs(a[k]));
        m = std::max(m, std::abs(b[k]));
    }
    const float f = static_cast<float>(m);
    const float nx = std::nextafter(f, std::numeric_limits<float>::infinity());
    const double u = static_cast<double>(nx) - static_cast<double>(f);
    return (u > 0.0) ? u : static_cast<double>(std::numeric_limits<float>::denorm_min());
}

struct VisibilitySegment {
    std::array<double, 3> origin{{0.0, 0.0, 0.0}};  // in SCENE coordinates
    std::array<double, 3> dir{{1.0, 0.0, 0.0}};
    double tfar = -1.0;        // < 0 means: do not trace
    bool auto_visible = false; // segment below the float32 floor: touching
    bool unresolvable = false; // float32 floor exceeds the facet-cell ceiling
};

// Build the occlusion segment for a pair, given each end's face AREA and the
// mesh -> scene translation (zero when the scene is in mesh coordinates).
// `pa`/`pb` are the two centroids in MESH coordinates; the caller is
// responsible for passing them in canonical (lower face index first) order, so
// that the float32 traversal cannot answer differently from the two ends.
// The pre-2026-08-23 absolute-meter rule, kept so the A/B that justifies the
// replacement is runnable on ONE binary instead of resting on a build that no
// longer exists. PTM_VF_RAY_LEGACY=1 selects it; it warns, and it is a control,
// not a production setting.
bool visibilityRayLegacyRequested();

inline VisibilitySegment visibilityRayLegacy(const std::array<double, 3>& pa,
                                             const std::array<double, 3>& pb,
                                             const std::array<double, 3>& scene_shift) {
    VisibilitySegment seg;
    const std::array<double, 3> d = {{pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]}};
    const double r = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    if (!(r > 0.0)) { seg.auto_visible = true; return seg; }
    seg.dir = {{d[0] / r, d[1] / r, d[2] / r}};

    const double absolute_offset = 0.5;
    double offset = std::max(r * 0.001, absolute_offset);
    offset = std::min(offset, r * 0.1);
    double end_margin = std::max(r * 0.001, absolute_offset);
    end_margin = std::min(end_margin, r * 0.1);

    seg.origin = {{pa[0] - scene_shift[0] + offset * seg.dir[0],
                   pa[1] - scene_shift[1] + offset * seg.dir[1],
                   pa[2] - scene_shift[2] + offset * seg.dir[2]}};
    const double tfar = r - offset - end_margin;
    if (tfar < 0.1) { seg.auto_visible = true; seg.tfar = -1.0; return seg; }
    seg.tfar = tfar;
    return seg;
}

inline VisibilitySegment visibilityRay(const std::array<double, 3>& pa,
                                       const std::array<double, 3>& pb,
                                       double area_a, double area_b,
                                       const std::array<double, 3>& scene_shift) {
    if (visibilityRayLegacyRequested())
        return visibilityRayLegacy(pa, pb, scene_shift);

    VisibilitySegment seg;

    const std::array<double, 3> d = {{pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]}};
    const double r = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    if (!(r > 0.0)) { seg.auto_visible = true; return seg; }
    seg.dir = {{d[0] / r, d[1] / r, d[2] / r}};

    const std::array<double, 3> sa = {{pa[0] - scene_shift[0], pa[1] - scene_shift[1],
                                       pa[2] - scene_shift[2]}};
    const std::array<double, 3> sb = {{pb[0] - scene_shift[0], pb[1] - scene_shift[1],
                                       pb[2] - scene_shift[2]}};
    const double floor_len = kRayMarginUlps * sceneUlp(sa, sb);

    auto margin = [&](double area) {
        const double L = std::sqrt(2.0 * std::max(area, 0.0));
        const double ceiling = std::min(kRayMarginFaceCap * L, kRayMarginDistCap * r);
        double m = std::max(kRayMarginFaceFrac * L, floor_len);
        if (m > ceiling) {
            if (floor_len > ceiling) seg.unresolvable = true;
            m = ceiling;
        }
        return m;
    };

    const double ma = margin(area_a);
    const double mb = margin(area_b);

    seg.origin = {{sa[0] + ma * seg.dir[0], sa[1] + ma * seg.dir[1],
                   sa[2] + ma * seg.dir[2]}};
    const double tfar = r - ma - mb;
    if (tfar <= floor_len) { seg.auto_visible = true; seg.tfar = -1.0; return seg; }
    seg.tfar = tfar;
    return seg;
}

// Sample points for one face at a given near-field quadrature order; capacity
// is kMaxNearFieldQuadPoints (radiation_config.h). Deliberately NOT
// value-initialized: two of these are stack-constructed per near pair and only
// p[0 .. count) is ever read, so zeroing the tail would memset ~12 kB per call
// for nothing.
struct SubCentroidSamples {
    int count = 0;
    std::array<std::array<double, 3>, kMaxNearFieldQuadPoints> p;
};

namespace detail {
// Near-field physical-bound tripwire. A_i F_ij can never exceed min(A_i, A_j)
// -- F_ij <= 1 and, by reciprocity, F_ji <= 1. A finite-order area quadrature
// CAN exceed it: as a concave crease closes, the sample pairs nearest the
// shared edge have r -> 0 with cos_a cos_b -> 1 and the estimate diverges,
// while the true value approaches 1 from below. Measured on two unit triangles
// sharing an edge, order 3: F_ij = 0.27 at a 90 degree dihedral against a
// reference 0.21, 1.07 at 45 degrees, 2.24 at 30 degrees, 19.2 at 10 degrees.
// The H-matrix builder's adaptive order only slows the growth (4.59 at 10
// degrees, 16.75 at 5) -- it does not bound it; the clamp does.
//
// These counters exist so that an entry clamped to the physical bound is
// REPORTED rather than absorbed by the downstream row-sum renormalization,
// which would silently rescale the whole row and drive that face's sky view
// factor to zero.
inline std::atomic<long long> g_nearfield_bound_clamps{0};
inline std::atomic<int> g_nearfield_bound_worst_ratio_ppm{0};  // (g/g_max - 1) in ppm

// Pairs whose float32 margin floor exceeded the facet-cell ceiling -- the scene
// coordinates are too large for its facets and no occlusion ray at that pair
// means anything. Reported, never absorbed.
inline std::atomic<long long> g_visibility_unresolvable{0};
}  // namespace detail

// One pair could not be ray-traced in float32 at these scene coordinates.
inline void noteVisibilityUnresolvable() {
    detail::g_visibility_unresolvable.fetch_add(1, std::memory_order_relaxed);
}

// Out-of-line so this header does not pull in <iostream>; see the buffer-
// capacity clamp guard in subCentroidPoints() below.
void nearFieldQuadOrderClampWarning(int order);

// Equal-area sub-triangle centroids on a triangle (m^2 points at order m).
//
// Split the triangle into m^2 congruent sub-triangles by the standard
// barycentric subdivision and take each one's centroid, weight A/m^2:
//   row a holds (m - a) "upward" sub-triangles, centroids at
//     (u, v) = (a/m + 1/(3m), b/m + 1/(3m)),
//   and (m - a - 1) "downward" ones at
//     (u, v) = (a/m + 2/(3m), b/m + 2/(3m)).
// Sum over a of (m - a) + (m - a - 1) = m^2. Second-order accurate, and OPEN:
// no sample lands on an edge or a vertex, so the integrable 1/r^2 singularity
// of an edge-touching pair is never evaluated at r = 0.
//
// The order is clamped to 1..kMaxNearFieldQuadOrder because that is the
// capacity of the caller's stack buffer. ptm_thermal_sim refuses an
// out-of-range CONFIGURED order at startup (against the narrower
// kMaxConfigNearFieldQuadOrder), so this clamp is a buffer-overflow guard, not
// a policy -- but it is still reachable from a caller that builds a
// RadiationConfig directly, and a clamp that changes the physics silently is
// exactly the failure mode this project keeps paying for. So it says so, once.
inline void subCentroidPoints(const std::array<double, 3>& v0,
                              const std::array<double, 3>& v1,
                              const std::array<double, 3>& v2,
                              int order, SubCentroidSamples& out) {
    if (order < 1 || order > kMaxNearFieldQuadOrder) {
        static std::atomic<bool> warned{false};
        if (!warned.exchange(true)) {
            nearFieldQuadOrderClampWarning(order);
        }
    }
    const int m = std::min(std::max(order, 1), kMaxNearFieldQuadOrder);
    const double inv_m = 1.0 / static_cast<double>(m);
    const double third = inv_m / 3.0;

    int idx = 0;
    for (int a = 0; a < m; ++a) {
        const double u_row = static_cast<double>(a) * inv_m;
        for (int b = 0; b < m - a; ++b) {
            const double u = u_row + third;
            const double v = static_cast<double>(b) * inv_m + third;
            const double w = 1.0 - u - v;
            out.p[idx++] = {{u * v0[0] + v * v1[0] + w * v2[0],
                             u * v0[1] + v * v1[1] + w * v2[1],
                             u * v0[2] + v * v1[2] + w * v2[2]}};
        }
        for (int b = 0; b < m - a - 1; ++b) {
            const double u = u_row + 2.0 * third;
            const double v = static_cast<double>(b) * inv_m + 2.0 * third;
            const double w = 1.0 - u - v;
            out.p[idx++] = {{u * v0[0] + v * v1[0] + w * v2[0],
                             u * v0[1] + v * v1[1] + w * v2[1],
                             u * v0[2] + v * v1[2] + w * v2[2]}};
        }
    }
    out.count = idx;  // == m * m
}

// The symmetric double-area sub-centroid quadrature.
//
//   g == A_a F_ab == A_b F_ba
//      = (1/pi) int_{A_a} int_{A_b} cos_a cos_b / r^2 dA_b dA_a
//
// Both faces are sampled at the centroids of m x m equal-area sub-triangles, so
// the accumulated sum is ONE quantity -- the same double integral -- for both
// directions, and reciprocity is an algebraic identity of the estimator rather
// than something it approaches in the refinement limit.
//
// The rule this replaced in the sparse builder was one-sided: a point emitter
// at the centroid of one face against a closed vertex lattice on the other.
// Each direction was then a midpoint rule in a DIFFERENT variable of the same
// symmetric double integral, which is why A_i F_ij and A_j F_ji disagreed by up
// to 46% on a near pair. Measured against a Richardson double-area reference on
// data/meshes/lunar_crater_80N.ply (2026-08-23), the one-sided rule read -3.9
// percent mass-weighted over near-field pairs, -0.09 percent on non-touching
// pairs and -21.7 percent on edge/vertex-touching pairs (17.6 percent of the
// near-field mass); the defect was a MAGNITUDE bias, and the non-reciprocity
// was its symptom. Post-hoc symmetrization of the old rule fixes nothing
// (-3.87 percent). At m = 3 this rule reads +0.32 percent mass-weighted on the
// same population, and +1.26 percent on the edge/vertex-touching subset.
//
// It is NOT uniformly better on every geometry. On two perpendicular unit
// squares sharing an edge -- the analytic worst case, an integrable 1/r^2
// singularity sitting on the shared edge -- the old rule read -18.1 percent and
// this one reads +18.5 percent at m = 3, the same magnitude with the sign
// flipped. No production-order area quadrature is close there; the error falls
// only like 1/m. Arm [T] of test_vf_production bounds it and prints the table.
//
// CALLERS MUST PASS THE PAIR IN A CANONICAL ORDER (face a = min index) if they
// want bit-identical values in both directions. The mathematics is symmetric
// either way, but the SUMMATION order is not, and "reciprocal to 1e-16" is a
// weaker statement than "reciprocal to the last bit".
//
// KNOWN LIMITATION: occlusion is the caller's business and is all-or-nothing
// for the pair. No sample point carries its own visibility, so a partially
// occluded near pair is still fully in or fully out.
//
// n_valid_out, when non-null, receives the number of sample pairs that survived
// the grazing cut. Zero means the pair contributed nothing.
inline double nearFieldPairIntegral(const std::array<double, 3>& a0,
                                    const std::array<double, 3>& a1,
                                    const std::array<double, 3>& a2,
                                    const std::array<double, 3>& b0,
                                    const std::array<double, 3>& b1,
                                    const std::array<double, 3>& b2,
                                    const std::array<double, 3>& n_a,
                                    const std::array<double, 3>& n_b,
                                    double A_a, double A_b, int order,
                                    int* n_valid_out = nullptr) {
    SubCentroidSamples pts_a, pts_b;
    subCentroidPoints(a0, a1, a2, order, pts_a);
    subCentroidPoints(b0, b1, b2, order, pts_b);

    double kernel_sum = 0.0;
    int n_valid = 0;

    for (int ia = 0; ia < pts_a.count; ++ia) {
        const auto& p_a = pts_a.p[ia];
        for (int ib = 0; ib < pts_b.count; ++ib) {
            const auto& p_b = pts_b.p[ib];

            const double dx = p_b[0] - p_a[0];
            const double dy = p_b[1] - p_a[1];
            const double dz = p_b[2] - p_a[2];
            const double r2 = dx * dx + dy * dy + dz * dz;
            if (r2 < 1e-20) continue;

            const double inv_r = 1.0 / std::sqrt(r2);
            // Outward normals: cos_a along +d, cos_b along -d. Swapping the two
            // faces flips d and exchanges the two cosines, so the product is
            // invariant -- this is the symmetry the canonical ordering exploits.
            const double cos_a = (n_a[0] * dx + n_a[1] * dy + n_a[2] * dz) * inv_r;
            const double cos_b = -(n_b[0] * dx + n_b[1] * dy + n_b[2] * dz) * inv_r;

            // Grazing/back-facing cut, the shared kGrazingCosCut. A rejected
            // sample pair contributes ZERO; it is not removed from the domain.
            if (cos_a < kGrazingCosCut || cos_b < kGrazingCosCut) continue;

            kernel_sum += (cos_a * cos_b) / r2;
            ++n_valid;
        }
    }

    if (n_valid_out) *n_valid_out = n_valid;
    if (n_valid == 0) return 0.0;

    // Divide by the TOTAL sample-pair count, never by the survivors. Dividing by
    // the survivors would condition the estimate on detection -- FC-08's error --
    // and inflates the near field by +1.42 percent mass-weighted / +25 percent on
    // the worst pair, while the cut itself discards 0.00000 percent of the
    // coupling mass (measured 2026-08-23). This divisor is correct; do not "fix"
    // it.
    const double n_pairs = static_cast<double>(pts_a.count) *
                           static_cast<double>(pts_b.count);

    double g = A_a * A_b * kernel_sum / (M_PI * n_pairs);

    // Physical bound. F_ab <= 1 and F_ba <= 1, so A_a F_ab <= min(A_a, A_b).
    // This is a conservation bound, not a tuning parameter, and min() is
    // symmetric so the clamp cannot break the reciprocity identity. Exceeding
    // it means the quadrature has failed on a near-singular concave crease --
    // it is counted and reported, never absorbed silently.
    const double g_max = std::min(A_a, A_b);
    if (g > g_max) {
        const double excess_ppm = 1.0e6 * (g / g_max - 1.0);
        const int excess = static_cast<int>(std::min(excess_ppm, 2.0e9));
        int prev = detail::g_nearfield_bound_worst_ratio_ppm.load(std::memory_order_relaxed);
        while (excess > prev &&
               !detail::g_nearfield_bound_worst_ratio_ppm.compare_exchange_weak(prev, excess)) {
        }
        detail::g_nearfield_bound_clamps++;
        g = g_max;
    }

    return g;
}

// Print (and reset) the near-field physical-bound tripwire; prints nothing when
// it did not fire. NOT gated on a verbosity flag: a clamped view factor is an
// energy-conservation event, and the whole point of the counter is that it
// cannot be missed. On stderr, not stdout, because several callers redirect the
// builder's stdout to keep their own output greppable (test_vf_production does)
// and this must survive. `builder` names which VF path fired it, because the
// counters are process-wide and shared between them.
void reportNearFieldBoundClamps(const char* builder, int order);

// Report (and reset) the float32-unresolvable pair count. Prints nothing when
// it did not fire.
void reportVisibilityUnresolvable(const char* builder);

}  // namespace radiation
