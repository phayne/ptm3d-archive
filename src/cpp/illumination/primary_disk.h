// primary_disk.h -- finite-disk geometry for the PRIMARY body (Jupiter-shine,
// Earthshine).  Header-only so the standalone gate `test_primary_disk` can
// exercise exactly the arithmetic the engine runs.
//
// PTM-44: the primary is not a point.  Jupiter subtends 5.98 deg of angular
// RADIUS from Europa and Earth 0.95 deg from the Moon, yet both primary-flux
// paths decided visibility with a single ray to the disk CENTRE and zeroed the
// whole flux the instant `cos(theta) <= 0`.  Both are all-or-nothing
// quantisations of a finite source; this header supplies the two pieces that
// replace them.
//
// Full derivation, with the numerical verification: docs/primary-reflected-flux-derivation.md

#ifndef PTM_PRIMARY_DISK_H
#define PTM_PRIMARY_DISK_H

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace ptm {
namespace disk {

// ---------------------------------------------------------------------------
// Quasi-random sampling of a unit disk.
//
// Moved here VERBATIM from the anonymous namespace of illumination_model.cpp so
// the solar-disk sampler and the new primary-disk sampler share one definition
// (PTM-45(ii) was the same class of drift: one concept spelled three ways).
// The engine's call sites are unchanged and the arithmetic is bit-identical.
// ---------------------------------------------------------------------------

// Radical inverse base 2
inline double vanDerCorput(uint32_t n) {
    n = (n << 16) | (n >> 16);
    n = ((n & 0x55555555u) << 1) | ((n & 0xAAAAAAAAu) >> 1);
    n = ((n & 0x33333333u) << 2) | ((n & 0xCCCCCCCCu) >> 2);
    n = ((n & 0x0F0F0F0Fu) << 4) | ((n & 0xF0F0F0F0u) >> 4);
    n = ((n & 0x00FF00FFu) << 8) | ((n & 0xFF00FF00u) >> 8);
    return double(n) * 2.3283064365386963e-10; // /2^32
}

// Hammersley point in [0,1)^2 with Cranley-Patterson rotation
inline void hammersley(uint32_t i, uint32_t N, double uoff, double voff,
                       double& u, double& v) {
    u = std::fmod((double(i) / double(std::max(1u, N))) + uoff, 1.0);
    v = std::fmod(vanDerCorput(i) + voff, 1.0);
}

// Concentric disk mapping (Shirley-Chiu)
inline void squareToConcentricDisk(double u, double v, double& dx, double& dy) {
    double sx = 2.0*u - 1.0;
    double sy = 2.0*v - 1.0;
    if (sx == 0.0 && sy == 0.0) { dx = dy = 0.0; return; }
    double r, phi;
    if (std::abs(sx) > std::abs(sy)) { r = sx; phi = (M_PI/4.0) * (sy/sx); }
    else                             { r = sy; phi = (M_PI/2.0) - (M_PI/4.0) * (sx/sy); }
    dx = r * std::cos(phi);
    dy = r * std::sin(phi);
}

// ---------------------------------------------------------------------------
// Horizon-clipped mean cosine over the primary's disk  (derivation note, eq. 9)
// ---------------------------------------------------------------------------

// g(t) = the mean of max(t + x, 0) over the unit disk, where x is the
// coordinate along the horizon gradient.  Closed form:
//
//   g(t) = [ t*pi/2 + t^2*sqrt(1-t^2) + t*asin(t) + (2/3)*(1-t^2)^{3/2} ] / pi
//
//   g(-1) = 0,  g(0) = 2/(3 pi) = 0.212207,  g(1) = 1
//   g'(t) = [ pi/2 + asin(t) + t*sqrt(1-t^2) ] / pi  >= 0  on [-1,1],
//   g'(-1) = 0 and g'(1) = 1: g is C^1 and monotone, so it joins the two
//   saturated branches without a kink.
inline double horizonMeanG(double t) {
    if (t >= 1.0)  return 1.0;
    if (t <= -1.0) return 0.0;
    const double s = std::sqrt(std::max(0.0, 1.0 - t*t));
    return ( t*(M_PI*0.5) + t*t*s + t*std::asin(t) + (2.0/3.0)*s*s*s ) / M_PI;
}

// <cos theta> over the primary's disk with the part below the face's horizon
// plane removed.  `cos_theta0` is the cosine at the disk CENTRE, `alpha` the
// disk's angular radius in radians.
//
// Branches, and why they join exactly:
//   h = asin(cos_theta0) is the disk centre's altitude above the horizon plane.
//   h >= alpha : the whole disk is up -> the exact answer is cos_theta0, and
//                the arithmetic downstream is BIT-IDENTICAL to the pre-PTM-44
//                code.  This is the overwhelming majority of faces.
//   h <= -alpha: the whole disk is down -> 0, also bit-identical (the old
//                cos <= 0 cut).
//   otherwise  : sin(alpha * g(h/alpha)).  At h = +alpha this is sin(alpha),
//                which is exactly cos_theta0 there; at h = -alpha it is 0.
//                The sin() wrapper is what makes both seams exact -- g is
//                derived from a linearisation of cos(theta) across the disk,
//                and mapping back through sin recovers the leading curvature.
//
// Truncation is O(alpha^2) INSIDE the band only: <= ~1 % of an already small
// partial-band flux at Jupiter/Europa (alpha = 5.98 deg), <= 0.03 % at
// Earth/Moon (alpha = 0.95 deg).
inline double diskMeanCosAboveHorizon(double cos_theta0, double alpha) {
    if (alpha <= 0.0) return std::max(0.0, cos_theta0);
    const double c = std::max(-1.0, std::min(1.0, cos_theta0));
    const double h = std::asin(c);
    if (h >=  alpha) return c;
    if (h <= -alpha) return 0.0;
    return std::sin(alpha * horizonMeanG(h / alpha));
}

// ---------------------------------------------------------------------------
// Sample counts
// ---------------------------------------------------------------------------

// Number of disk samples used when the cheap probe is ambiguous.
//
// The naive expectation is that N samples quantise the visible fraction at 1/N.
// MEASURED against a half-plane occluder (test_primary_disk, section 4) the
// worst-case error of this point set is about 3/N, not 1/N, because a straight
// edge sweeping across a low-discrepancy point set crosses points in clumps:
//
//     N     worst |sampled - analytic|      (Hammersley + concentric)
//     64            0.0369
//    128            0.0161
//    256            0.0123
//    512            0.0064
//
// Fibonacci-spiral and equal-area polar-grid point sets were tried and are no
// better (0.0067 and 0.0167 at N = 512), so this is a property of counting
// rays, not of this particular sequence.  N = 512 is therefore the smallest
// power of two that meets the specified criterion, < 1 % of full flux.
//
// The cost lands ONLY on faces whose probe was split, i.e. the thin band whose
// view of the primary is partly blocked by terrain; a convex global mesh has
// none, and every other face is decided by the 5-ray probe.
constexpr uint32_t PRIMARY_DISK_SAMPLES = 512;

// Cheap probe: disk centre + 4 rim points.  Unanimous probes decide the face
// without sampling (see illumination_model.cpp computePrimaryDiskVisibility).
constexpr uint32_t PRIMARY_DISK_PROBES = 5;

// Offsets, in units of the disk's angular radius, of the probe directions.
// Index 0 is the centre so callers can reuse it as the old centre ray.
inline void probeOffset(uint32_t k, double& dx, double& dy) {
    switch (k) {
        case 0:  dx =  0.0; dy =  0.0; break;
        case 1:  dx =  1.0; dy =  0.0; break;
        case 2:  dx = -1.0; dy =  0.0; break;
        case 3:  dx =  0.0; dy =  1.0; break;
        default: dx =  0.0; dy = -1.0; break;
    }
}

// ---------------------------------------------------------------------------
// Lambert-sphere photometry (PTM-41)
// ---------------------------------------------------------------------------

// Phase function of a Lambert sphere, normalised to Phi(0) = 1:
//     Phi(a) = [ sin a + (pi - a) cos a ] / pi
// Phase integral q = 2 * Int_0^pi Phi sin a da = 3/2, hence the geometric
// albedo of a Lambert sphere is p = A_bond / q = (2/3) A_bond.
inline double lambertSpherePhase(double phase_angle_rad) {
    const double a = std::max(0.0, std::min(M_PI, phase_angle_rad));
    return (std::sin(a) + (M_PI - a) * std::cos(a)) / M_PI;
}

// Geometric albedo implied by a Bond albedo under the Lambert-sphere model.
inline double lambertSphereGeometricAlbedo(double bond_albedo) {
    return (2.0/3.0) * bond_albedo;
}

// Disk-integrated reflected irradiance on a face:
//     E = p * F * sin^2(alpha) * Phi(a) * <cos theta>
// with p from lambertSphereGeometricAlbedo and <cos theta> from
// diskMeanCosAboveHorizon (times any terrain-visible fraction).
inline double lambertSphereIrradiance(double solar_flux_at_primary,
                                      double bond_albedo,
                                      double sin_angular_radius,
                                      double phase_angle_rad,
                                      double mean_cos) {
    if (mean_cos <= 0.0) return 0.0;
    const double s2 = sin_angular_radius * sin_angular_radius;
    return solar_flux_at_primary
         * lambertSphereGeometricAlbedo(bond_albedo)
         * s2
         * lambertSpherePhase(phase_angle_rad)
         * mean_cos;
}

} // namespace disk
} // namespace ptm

#endif // PTM_PRIMARY_DISK_H
