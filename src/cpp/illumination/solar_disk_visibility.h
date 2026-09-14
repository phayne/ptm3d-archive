// solar_disk_visibility.h -- the ONE spelling of "how much of the solar disk
// can this point see", shared by every passive measurement product.
//
// WHY THIS IS A HEADER AND NOT A SECOND COPY
// ------------------------------------------
// The Sun is not a point. Its angular radius is ~0.2666 deg at 1 AU, so a
// shadow edge is smeared by dist*tan(theta_sun): 0.045 m at a 10 m occluder,
// 0.45 m at 100 m. PTM-38 removed a center-ray test from the engine's own
// direct-solar path for exactly this reason -- it "broke the penumbra at both
// ends" -- and the flux-cube pass then grew a ladder of its own to say the same
// thing for a cube. The shadow-depth pass needs the identical question asked at
// a point above the surface.
//
// Three spellings of one concept is the PTM-45(ii) / PTM-106c defect class:
// they agree today and drift later. So the ladder lives here, once, and both
// passes call it. It is header-only for the same reason primary_disk.h is: the
// standalone tests exercise exactly the arithmetic the engine runs.
//
// THE LADDER, and what each rung costs
// ------------------------------------
//   alpha <= 0          -> point sun, one binary ray (the legacy behavior)
//   probe unanimous     -> 1.0 or 0.0, no further rays
//   probe split         -> `samples` directions across the disk
//
// The probe set is ptm::disk::probeOffset -- center plus the four rim points at
// radius 1 -- and the escalation set is the engine's own Hammersley /
// Shirley-Chiu concentric-disk pair. Neither is invented here.
//
// ⛔ A UNANIMOUSLY BLOCKED PROBE IS ACCEPTED AS ZERO WITHOUT ESCALATING, and a
// consumer has to know that. Five rays cannot prove a 0.53 deg disk carries no
// clear sliver; they can only fail to find one. Escalating the blocked side as
// well would multiply the ray budget of a permanently shadowed face by ~26x for
// a refinement that is bounded by the probe geometry, and the flux-cube pass has
// always made the same trade (a unanimous 0 there is likewise unescalated). What
// removes the arbitrariness is the BASIS: `basis_ref` lets the caller put two of
// the four rim probes exactly on the disk extremes along the axis it cares
// about, so the rung that matters for that product is sampled exactly rather
// than wherever an arbitrary basis happens to land.
#ifndef PTM_SOLAR_DISK_VISIBILITY_H
#define PTM_SOLAR_DISK_VISIBILITY_H

#include "primary_disk.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <vector>

namespace ptm {
namespace disk {

using Vec3 = std::array<double, 3>;

// Orthonormal basis (t, b) spanning the plane perpendicular to `s_hat`.
//
// `basis_ref` is the vector t is made perpendicular to. Pass {0,0,0} for the
// historical choice -- +Z, or +X when the Sun is within ~26 deg of the pole --
// which is what the flux-cube pass has always used and what keeps it
// bit-identical. Pass the local vertical to get b lying IN the plane containing
// the Sun direction and that vertical, which puts probe offsets (0, +-1) exactly
// on the upper and lower solar limb as seen from a point looking along the
// vertical. That is the pair a height search lives or dies on.
//
// A `basis_ref` parallel to `s_hat` would give a degenerate cross product, so it
// falls back to the historical choice rather than emitting a zero basis.
inline void solarDiskBasis(const Vec3& s_hat, const Vec3& basis_ref,
                           Vec3& t, Vec3& b) {
    auto cross = [](const Vec3& p, const Vec3& q) -> Vec3 {
        return {p[1]*q[2] - p[2]*q[1], p[2]*q[0] - p[0]*q[2], p[0]*q[1] - p[1]*q[0]};
    };
    auto norm = [](const Vec3& p) -> double {
        return std::sqrt(p[0]*p[0] + p[1]*p[1] + p[2]*p[2]);
    };
    auto unit = [&](const Vec3& p) -> Vec3 {
        const double l = norm(p);
        return (l > 0.0) ? Vec3{p[0]/l, p[1]/l, p[2]/l} : Vec3{0.0, 0.0, 0.0};
    };

    Vec3 a{0.0, 0.0, 0.0};
    const double ref_len = norm(basis_ref);
    if (ref_len > 0.0) {
        const Vec3 r = unit(basis_ref);
        // Reject a reference that is (nearly) parallel to the Sun direction:
        // the cross product would be noise, and a noisy basis is worse than the
        // historical one because it is not reproducible.
        const double align = std::fabs(r[0]*s_hat[0] + r[1]*s_hat[1] + r[2]*s_hat[2]);
        if (align < 0.999) a = r;
    }
    if (a[0] == 0.0 && a[1] == 0.0 && a[2] == 0.0) {
        a = (std::fabs(s_hat[2]) < 0.9) ? Vec3{0.0, 0.0, 1.0} : Vec3{1.0, 0.0, 0.0};
    }
    t = unit(cross(a, s_hat));
    b = cross(s_hat, t);
}


// The disk frame at a point: the basis plus the ONE direction formula. Callers
// that batch their ray casting need the directions up front and cannot use the
// callback form below, but they must use the SAME arithmetic -- so it lives
// here, once, and both forms are built on it.
struct DiskFrame {
    Vec3 s_hat{0.0, 0.0, 0.0};
    Vec3 t{0.0, 0.0, 0.0};
    Vec3 b{0.0, 0.0, 0.0};
    double ta = 0.0;              // tan(alpha)

    void set(const Vec3& sun_dir, double alpha, const Vec3& basis_ref) {
        s_hat = sun_dir;
        solarDiskBasis(s_hat, basis_ref, t, b);
        ta = std::tan(alpha);
    }
    Vec3 dir(double dx, double dy) const {
        Vec3 d{ s_hat[0] + t[0]*dx*ta + b[0]*dy*ta,
                s_hat[1] + t[1]*dx*ta + b[1]*dy*ta,
                s_hat[2] + t[2]*dx*ta + b[2]*dy*ta };
        const double l = std::sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2]);
        if (l > 0.0) { d[0] /= l; d[1] /= l; d[2] /= l; }
        return d;
    }
    // The k-th PROBE direction, k in [0, PRIMARY_DISK_PROBES).
    Vec3 probe(uint32_t k) const {
        double dx, dy;
        ptm::disk::probeOffset(k, dx, dy);
        return dir(dx, dy);
    }
    // The i-th ESCALATION direction out of N.
    Vec3 sample(uint32_t i, uint32_t N) const {
        double u, v, dx, dy;
        ptm::disk::hammersley(i, N, 0.0, 0.0, u, v);
        ptm::disk::squareToConcentricDisk(u, v, dx, dy);
        return dir(dx, dy);
    }
};

// Unoccluded fraction of the solar disk seen from `o`, in [0, 1].
//
//   o          ray origin, in the mesh's own frame
//   s_hat      unit direction to the disk CENTER
//   alpha      angular RADIUS of the disk, radians. <= 0 => point sun.
//   samples    escalation count when the probe splits
//   basis_ref  see solarDiskBasis; {0,0,0} for the historical basis
//   occluded   the engine's own ray test, injected
//   split      set when the probe disagreed, i.e. the disk is PARTIALLY
//              occluded -- which is what a `penumbral` flag should mean
inline double solarDiskVisibility(
        const Vec3& o, const Vec3& s_hat, double alpha, int samples,
        const Vec3& basis_ref,
        const std::function<bool(const Vec3&, const Vec3&)>& occluded,
        bool& split) {
    split = false;
    if (alpha <= 0.0) return occluded(o, s_hat) ? 0.0 : 1.0;

    DiskFrame f;
    f.set(s_hat, alpha, basis_ref);

    int clear = 0;
    for (uint32_t k = 0; k < ptm::disk::PRIMARY_DISK_PROBES; ++k)
        if (!occluded(o, f.probe(k))) ++clear;
    if (clear == 0) return 0.0;
    if (clear == static_cast<int>(ptm::disk::PRIMARY_DISK_PROBES)) return 1.0;
    split = true;
    const uint32_t N = static_cast<uint32_t>(std::max(8, samples));
    int hit = 0;
    for (uint32_t i = 0; i < N; ++i)
        if (!occluded(o, f.sample(i, N))) ++hit;
    return static_cast<double>(hit) / static_cast<double>(N);
}

}  // namespace disk
}  // namespace ptm

#endif  // PTM_SOLAR_DISK_VISIBILITY_H
