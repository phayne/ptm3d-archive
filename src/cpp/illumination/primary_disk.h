// primary_disk.h -- finite-disk geometry for the PRIMARY body (Jupiter-shine,
// Earthshine).  Header-only so the standalone gate `test_primary_disk` can
// exercise exactly the arithmetic the engine runs.
//
// PTM-44: the primary is not a point.  Jupiter subtends 5.98 deg of angular
// RADIUS from Europa and Earth 0.95 deg from the Moon, yet both primary-flux
// paths decided visibility with a single ray to the disk CENTER and zeroed the
// whole flux the instant `cos(theta) <= 0`.  Both are all-or-nothing
// quantizations of a finite source; this header supplies the two pieces that
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
// plane removed.  `cos_theta0` is the cosine at the disk CENTER, `alpha` the
// disk's angular radius in radians.
//
// Branches, and why they join exactly:
//   h = asin(cos_theta0) is the disk center's altitude above the horizon plane.
//   h >= alpha : the whole disk is up -> the exact answer is cos_theta0, and
//                the arithmetic downstream is BIT-IDENTICAL to the pre-PTM-44
//                code.  This is the overwhelming majority of faces.
//   h <= -alpha: the whole disk is down -> 0, also bit-identical (the old
//                cos <= 0 cut).
//   otherwise  : sin(alpha * g(h/alpha)).  At h = +alpha this is sin(alpha),
//                which is exactly cos_theta0 there; at h = -alpha it is 0.
//                The sin() wrapper is what makes both seams exact -- g is
//                derived from a linearization of cos(theta) across the disk,
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
// The naive expectation is that N samples quantize the visible fraction at 1/N.
// MEASURED against a half-plane occluder (test_primary_disk section 4, edge
// swept in steps of 1/500 of the disk radius) the worst-case error is much
// SHALLOWER than 1/N -- a straight edge crosses low-discrepancy points in
// clumps, so refining N buys less than counting suggests:
//
//     N     worst |sampled - analytic|      1/N
//     16           0.0837                 0.0625
//     32           0.0484                 0.0312
//     64           0.0381                 0.0156
//    128           0.0175                 0.0078
//    256           0.0159                 0.0039
//    512           0.0082                 0.0020
//
// A power-law fit over that range gives worst ~ 0.49 * N^(-0.65), i.e. roughly
// the inverse square root of the sample count, NOT 1/N. (Fermi's independent
// fit on the same data: N^(-0.68).) Two other point sets were tried at N = 512
// and are no better -- Fibonacci spiral 0.0092, equal-area polar grid 0.0174,
// both printed by the same test -- so this is a property of counting rays, not
// of this particular sequence.
//
// N = 128 IS A CHOSEN BUDGET, not the smallest N meeting a fixed criterion.
// The count was 512: the smallest power of two under a purely RELATIVE
// criterion, "worst error < 1 % of full flux" (0.0082). Restated against the
// two things that actually constrain it:
//
//   accuracy  the fitted law gives worst |dV| = 0.49 * 128^-0.65 = 0.021 (the
//             measured 0.0175 above is inside it). The full primary flux on
//             the coldest lunar PSR face is 0.087 W/m^2 -- 0.046 IR + 0.041
//             reflected, measured over 48 MS3 A03MP026 frames -- which lifts
//             that face from the 24.04 K geothermal floor to 37.36 K. A 2.1 %
//             error in the GEOMETRIC visibility factor is therefore worth
//             <= 0.163 K there, and less on every warmer face, against a 0.1 K
//             equilibration target and a ~1 K documented spin-up bias. Paul
//             accepted that bar on 2026-08-14.
//
//   cost      the escalation was 91.5 % of the illumination phase on the MS3
//             lunar polar production config (profiler, 2026-08-14), and
//             accounted for 102 % of that phase's 5.5x slowdown since July.
//             Cost is linear in N over the escalated faces, so this is where
//             the kelvin are cheapest.
//
// This is a budget, so it has to be re-derived, not inherited, if either side
// moves. The kelvin figure is ABSOLUTE and lunar: Jupiter's disk is 6.3x wider
// than Earth's, and a Europa DEM patch escalates ~24 % of its faces against the
// Moon's ~7 %. It was measured rather than assumed, 512 -> 128 end to end over
// 48 frames at production equilibration: worst face-frame |dT| 0.025 K on the
// MS3 lunar polar patch, 0.0025 K on the Europa Pwyll patch, 8e-6 K on a Europa
// ico6 global. Europa comes out TEN TIMES quieter than the Moon despite the
// wider disk, because dT/dF = 1/(4 eps sigma T^3) is ~20x smaller at Europa's
// ~100 K than at a 37 K lunar PSR floor -- the binding case is cold, not big.
// So one N serves both; a Europa-specific value would be a separate decision
// with its own measurement.
//
// The cost lands ONLY on faces whose probe was split, i.e. the thin band whose
// view of the primary is partly blocked by terrain. Audited 2026-08-14:
// updatePrimaryDiskVisibility is the only engine code that reads this constant
// (the one other reader anywhere is the gate below, test_primary_disk.cpp), and
// every non-split outcome -- disk below the horizon, no BVH, probe underflow,
// probe unanimously clear, probe unanimously blocked -- returns before the
// sampler.
constexpr uint32_t PRIMARY_DISK_SAMPLES = 128;

// The worst-case visibility error PRIMARY_DISK_SAMPLES is budgeted for, from
// the fitted law above. This is the number the kelvin argument was made
// against, so test_primary_disk gates the MEASURED sweep on it: N and this
// constant move together, or the gate fails and the budget gets re-derived on
// purpose rather than by accident.
constexpr double PRIMARY_DISK_WORST_VISIBILITY_ERROR = 0.021;

// Cheap probe: disk center + 4 rim points.  Unanimous probes decide the face
// without sampling (see illumination_model.cpp computePrimaryDiskVisibility).
constexpr uint32_t PRIMARY_DISK_PROBES = 5;

// Offsets, in units of the disk's angular radius, of the probe directions.
// Index 0 is the center so callers can reuse it as the old center ray.
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

// Phase function of a Lambert sphere, normalized to Phi(0) = 1:
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
