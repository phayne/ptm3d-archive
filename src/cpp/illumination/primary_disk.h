// primary_disk.h -- finite-disk geometry for the PRIMARY body (Jupiter-shine,
// Earthshine).  Header-only so the standalone gate `test_primary_disk` can
// exercise exactly the arithmetic the engine runs.
//
// the primary is not a point.  Jupiter subtends 5.98 deg of angular
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
// (the primary/eclipse fallbacks were the same class of drift: one concept spelled three ways).
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
//                the arithmetic downstream is BIT-IDENTICAL to the pre-fix
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
// the inverse square root of the sample count, NOT 1/N. (The physics review's independent
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
//             reflected, measured over 48 lunar polar frames -- which lifts
//             that face from the 24.04 K geothermal floor to 37.36 K. A 2.1 %
//             error in the GEOMETRIC visibility factor is therefore worth
//             <= 0.163 K there, and less on every warmer face, against a 0.1 K
//             equilibration target and a ~1 K documented spin-up bias. Paul
//             accepted that bar on 2026-08-14.
//
//   cost      the escalation was 91.5 % of the illumination phase on the
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
// lunar polar patch, 0.0025 K on the Europa Pwyll patch, 8e-6 K on a Europa
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

// ---------------------------------------------------------------------------
// PER-FACE sampling budget
//
// Paul's ruling, 2026-08-16: "primary-disk visibility is a secondary effect
// (except possibly in the very coldest PSRs) that should not be driving
// illumination cost; we should reduce to the minimum number of rays and checks
// for reasonable accuracy without spending a lot of computational cycles on
// it."
//
// PRIMARY_DISK_SAMPLES above is ONE number for every face on every body. It was
// fixed by the single hardest face in the fleet -- the coldest lunar PSR floor,
// where the whole primary flux is worth 13 K -- and then charged to all
// 2,397,890 faces of an A3GT production mesh. Measured on the lunar polar site A window,
// current main: 571,917 faces per epoch escalate to 128 rays, 82.8 M
// primary-disk rays per epoch = 34.5x the entire direct-solar shadow-ray budget,
// ~900 s of a 2,772 s window. The flux those rays resolve peaks at 0.087 W/m^2.
//
// What follows converts the SAME accuracy bar into a per-face ray count. The
// bar is unchanged; only the pricing is.
//
// THE ARITHMETIC. A visibility error dV misplaces dV * F_full watts on a face,
// where F_full is the irradiance that face would receive from a fully visible
// primary disk (analytic horizon clip included -- see the caller). The face
// sits on the radiative balance
//
//     eps sigma T^4 = Q_floor + V * F_full + (everything else, all >= 0)
//
// so the temperature error the misplaced watts buy is, EXACTLY (no
// linearization -- it costs nothing to keep the fourth root),
//
//     dT(dV, V) = [ (Q_floor + (V+dV) F_full)/(eps sigma) ]^(1/4)
//               - [ (Q_floor +  V    F_full)/(eps sigma) ]^(1/4)
//
// which is largest at V = 0, the cold end: the same watt is worth more kelvin
// the colder the face, dT/dF = 1/(4 eps sigma T^3). Taking V = 0 is therefore
// the worst case over the one quantity the sampler has not yet measured, and it
// needs no assumption about the terrain. Setting that equal to the budget and
// solving for dV:
//
//     dV_allowed = eps sigma [ (T_floor + budget)^4 - T_floor^4 ] / F_full
//     T_floor    = (Q_floor / (eps sigma))^(1/4)
//
// WHERE THE TEMPERATURE COMES FROM, and why it is not the running temperature.
// The illumination phase computes the WHOLE window up front, before a single
// thermal step (illumination_model.cpp calculateIlluminationOptimized loops all
// time_steps_), so no thermal state exists to read -- not the previous step's,
// not any step's. Three surrogates were considered and two were rejected on
// physics, not on convenience:
//   * the current epoch's own direct-solar flux -- REJECTED, and it is not a
//     near miss. With a massless surface node the balance is
//     eps sigma T_s^4 = F_abs + k dT/dz|_0, and while a face is HEATING the
//     conductive term is negative, so T_s < T_eq(F_abs). A sunrise face is
//     therefore COLDER than its instantaneous radiative equilibrium, and a
//     surrogate built on it would under-count rays exactly where cold terrain
//     crosses the terminator.
//   * a window-mean insolation census -- REJECTED: a mean over the RUN's window
//     is not a function of geometry alone, so a chunked run and a single-shot
//     run would price the same epoch differently. Chunk invariance is not
//     negotiable here.
//   * Q_floor, the body's deep-heat flux, with V = 0 -- TAKEN. It is a true
//     lower bound on the face temperature (every term dropped is non-negative),
//     it is one scalar, it is a pure function of the body, and it is the same
//     number the engine's own PSR floor already is: T_geo = (Qb/(eps sigma))^0.25
//     = 24.0428 K on the Moon at Qb = 0.018, which is precisely where the A3GT
//     temperature field floors.
// Costed: the strict V = 0 bound is at most one ladder rung more expensive than
// the self-consistent T_eq(Q_floor + V F_full) it replaces, because Q_floor
// dominates the sum on every face where the ladder actually saves anything.
//
// WHERE THE RULE SITS RELATIVE TO THE INHERITED CONSTANT. On the Moon (eps
// 0.95, Qb 0.018 -> T_floor 24.0428 K, budget 0.163 K) it returns
// dV_allowed = 0.0057 at F_full = 0.087 W/m^2, the measured peak lunar primary
// flux. That is 3.7x STRICTER than PRIMARY_DISK_WORST_VISIBILITY_ERROR = 0.021,
// so at and near the peak the ladder asks for more rays than the cap allows and
// simply keeps the cap. The ladder therefore never demotes the face the N = 128
// ruling was made on; it only demotes faces whose primary flux is far below the
// peak, which -- on a lunar polar mesh, where Earth sits within a few degrees of
// the horizon -- is most of them.
// (The 0.021 figure comes from evaluating the same expression at V = 1 instead
// of V = 0, i.e. crediting the face with the very flux whose visibility is in
// question. That is where the accepted 24.04 -> 37.36 K derivation stands; it is
// the right reading for a FULLY VISIBLE cold face, which is exactly the face
// that never escalates. For a face the sampler is actually working on, V is
// unknown and V = 0 is the worst case.)
//
// A face whose budget is not reachable even at PRIMARY_DISK_SAMPLES keeps
// PRIMARY_DISK_SAMPLES -- today's behavior exactly, no accuracy is given up
// anywhere -- and the census counts it so the shortfall is visible rather than
// implied.
constexpr double PRIMARY_DISK_TEMPERATURE_BUDGET_K = 0.163;

// One rung of the ladder. `worst_visibility_error` is MEASURED, not fitted:
// test_primary_disk section 4c sweeps a half-plane occluder across the disk in
// steps of 1/500 of its radius, AT EVERY ORIENTATION, and fails if any rung's
// measured worst exceeds the number tabulated here. The fitted law
// 0.49 * N^-0.65 is not usable as the tabulated value -- at N = 64 it gives
// 0.033 against a measured 0.078.
//
// ⚠️ THE ORIENTATION SWEEP IS NEW, AND IT MOVES EVERY NUMBER IN THIS FILE.
// The table in the PRIMARY_DISK_SAMPLES comment above, and
// PRIMARY_DISK_WORST_VISIBILITY_ERROR with it, come from a sweep of an
// x-ALIGNED half-plane only. Measured 2026-08-16, sweeping the occluder's
// in-plane angle as well (converged: identical at 180, 720 and 2880
// orientations):
//
//     estimator      x-aligned    any orientation   worst angle
//     5-point probe    0.3002          0.3082          (no diagonal structure)
//     N =  16          0.0837          0.1860           45 deg
//     N =  32          0.0484          0.1251          135 deg
//     N =  64          0.0381          0.0778           45 deg
//     N = 128          0.0175          0.0613          135 deg
//     N = 256          0.0159          0.0351           45 deg
//     N = 512          0.0082          0.0300          135 deg
//
// The worst orientation is the DIAGONAL, every time, because that is the
// structure Shirley-Chiu's concentric mapping puts into the point set: the
// square's diagonals map to preferred directions on the disk, and an occluder
// edge lying along one of them cuts the samples along their own grain. A real
// ridge crest is a half-plane at an arbitrary angle, so the diagonal is not a
// contrived case; the x-aligned measurement simply never asked.
//
// Consequences, stated rather than absorbed:
//   * PRIMARY_DISK_SAMPLES = 128 does NOT deliver 0.021 worst-case. It delivers
//     0.0613, so the kelvin it was accepted against -- 0.163 K at the coldest
//     lunar face -- is really ~0.48 K there. That is a property of main as
//     shipped, unchanged by this ladder, and it is Paul's call whether 128 is
//     still the right cap. Nothing here quietly re-tunes it.
//   * N = 512 (the pre-2026-08-14 count) measures 0.0300, i.e. the 512 -> 128
//     decision was 0.0082 -> 0.0175 on paper and 0.0300 -> 0.0613 in fact.
//     Same factor, so that decision stands on its own terms.
//   * The count is the wrong lever for this error. 128 -> 512 buys 2x for 4x
//     the rays because the diagonal structure does not average out with N.
//     `hammersley()` already takes a Cranley-Patterson rotation (uoff, voff)
//     and the engine passes 0, 0 -- so every face on every body reuses the
//     identical point set with the identical grain. A per-face rotation is the
//     obvious repair, it is nearly free, and it changes results, so it needs
//     its own gate and its own measurement. NOT done here.
//
// This ladder prices against the ORIENTATION-WORST column, which is stricter
// than the standard the cap was accepted under. That direction is deliberate:
// a rung is only ever selected when it meets 0.163 K on the corrected numbers,
// so no face is demoted onto an error it cannot afford, and every face the
// corrected numbers cannot satisfy keeps PRIMARY_DISK_SAMPLES -- which is
// exactly what main gives it today.
struct PrimaryDiskTier {
    uint32_t samples;                 // disk samples; see the two sentinels
    double   worst_visibility_error;  // measured worst |V_est - V_true|
};

// Sentinels, both of which mean "cast no ray this face does not already cast":
//   samples == 0                    -> take the analytic clip, V := 1, NO rays
//                                      at all, not even the probe. Only for a
//                                      face that could absorb dV = 1.
//   samples == PRIMARY_DISK_PROBES  -> stop at the 5-ray probe every escalating
//                                      face already casts and take its own
//                                      cosine-weighted clear fraction.
// Ordered by increasing cost / decreasing error; the selector takes the first
// rung that fits, so the table must stay sorted both ways (gated in 4c).
constexpr int PRIMARY_DISK_NUM_TIERS = 6;
constexpr PrimaryDiskTier PRIMARY_DISK_TIERS[PRIMARY_DISK_NUM_TIERS] = {
    {   0, 1.0000 },   // V := 1 with no evidence: the error is the full range
    {   5, 0.3120 },   // 5-probe cosine-weighted fraction   (measured 0.3082)
    {  16, 0.1880 },   // all five MEASURED over occluder     (measured 0.1860)
    {  32, 0.1265 },   // orientation by test_primary_disk    (measured 0.1251)
    {  64, 0.0790 },   // 4c, which FAILS if any of them is   (measured 0.0778)
    { 128, 0.0620 },   // optimistic by so much as a bit      (measured 0.0613)
};
// The ~1.2 % headroom on each rung is deliberate and is not slack in the budget:
// the worst is a max over a staircase, so it moves in the last digit with
// -ffast-math codegen, and a gate that fails on codegen jitter gets disabled
// rather than believed. It errs toward MORE rays.

// Largest visibility error this face can absorb inside the kelvin budget.
// `eps_sigma` is emissivity * Stefan-Boltzmann, passed in rather than taken
// from a constants header so this stays the standalone arithmetic the gate
// links against, and so there is exactly one sigma in play per caller.
// Returns 1.0 (i.e. "anything") when the face receives no primary flux at all.
inline double primaryDiskAllowedVisibilityError(double full_disk_flux_w_m2,
                                                double floor_flux_w_m2,
                                                double eps_sigma) {
    if (!(full_disk_flux_w_m2 > 0.0)) return 1.0;
    if (!(eps_sigma > 0.0))           return 0.0;   // degenerate: buy everything
    const double q_floor = (floor_flux_w_m2 > 0.0) ? floor_flux_w_m2 : 0.0;
    const double t_floor = std::pow(q_floor / eps_sigma, 0.25);
    const double t_hi    = t_floor + PRIMARY_DISK_TEMPERATURE_BUDGET_K;
    const double dq      = eps_sigma * (t_hi*t_hi*t_hi*t_hi - q_floor / eps_sigma);
    const double dv      = dq / full_disk_flux_w_m2;
    return (dv > 1.0) ? 1.0 : dv;
}

// First rung of the ladder that fits, or PRIMARY_DISK_SAMPLES if none does.
inline uint32_t primaryDiskSamplesForAllowedError(double allowed_dv) {
    for (int t = 0; t < PRIMARY_DISK_NUM_TIERS; ++t) {
        if (PRIMARY_DISK_TIERS[t].worst_visibility_error <= allowed_dv) {
            return PRIMARY_DISK_TIERS[t].samples;
        }
    }
    return PRIMARY_DISK_SAMPLES;   // budget unreachable: keep today's behavior
}

// ---------------------------------------------------------------------------
// Temporal reuse of the terrain-visibility factor
// ---------------------------------------------------------------------------

// The terrain-visible fraction V of the primary's disk is the only part of
// <cos theta> that costs rays; the horizon clip A is closed-form and free.
// V is a function of (mesh, apparent position of the primary), and the primary
// barely moves in the body-fixed frame of a tidally locked satellite -- Earth
// librates ~0.057 deg per timestep on the 949-step lunar lunation against a
// 0.95 deg disk radius, Jupiter ~0.09 deg against a 5.98 deg one (both
// MEASURED, see the primary-position census below). Recomputing V every step
// re-answers a question whose input moved by a few percent of its own scale.
//
// This is the quantization of that input. The primary's body-fixed position is
// snapped to a cubic lattice of spacing
//
//     cell = PRIMARY_DISK_REUSE_CELL_FRACTION * primary_radius_        [km]
//
// and V is evaluated at the SNAPPED position, cached, and reused until the
// snapped position changes. Because alpha ~ R_primary / r, a lattice step of
// f * R_primary is a step of f * alpha in apparent ANGLE at any distance -- so
// one dimensionless constant sets the angular tolerance for Earth and Jupiter
// alike, with no per-body table.
//
// Snapping (rather than caching whatever the first step in a cell happened to
// compute) is deliberate and load-bearing: the cached V then depends ONLY on
// the lattice cell, not on when a run started or where its chunk boundaries
// fall. A chunked run, a warm-started run and a single-shot run therefore
// produce identical numbers, which the equivalent history-dependent scheme
// would silently break.
//
// This CHANGES RESULTS -- it is a controlled approximation, not a refactor --
// so it is budgeted exactly like PRIMARY_DISK_SAMPLES above, against Paul's
// 0.163 K worst-face bar, and measured rather than argued. See the commit
// message for the sweep.
//
// 0 disables reuse entirely (recompute every step, the pre-2026-08-15 behavior).
constexpr double PRIMARY_DISK_REUSE_CELL_FRACTION = 0.0;

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
// Lambert-sphere photometry
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
