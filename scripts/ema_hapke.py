#!/usr/bin/env python3
"""EMA Hapke disk-integrated photometry helpers (single-scattering albedo <-> disk).

PROVENANCE / HONESTY (read before trusting or "improving" this):

Recovered 2026-07-09. The only copy of these helpers
lived in /tmp/ema_sweep_driver.py, a scratch module destroyed with /tmp on reboot,
so every `fit` path that imported it was dead. The formula itself survived because
it was written into a comment at scripts/ema_campaign_driver.py:45-48:

    r0 = (1 - gamma) / (1 + gamma),  gamma = sqrt(1 - w)
    disk_p_from_w(w) = w/8 + 0.49*r0 + 0.196*r0**2

The physics review derived and validated it. This file transcribes it verbatim. It is NOT to
be re-derived or "corrected" here -- doing so silently rewrites every sweep
directory name on disk (they encode `p`, see below).

Three things are recorded honestly rather than quietly cleaned up:

1. The label "opposition-EXCLUDED" in that ema_campaign_driver.py comment is WRONG.
   The `+ w/8` term hardwires a single-scattering opposition bracket
   `(1 + B0) * P(0) = 2` (i.e. w/4 halved into the disk integral). The true
   opposition-EXCLUDED isotropic geometric albedo is `(r0/2)*(1 + r0/3)`, which
   the physics review verified against the exact Chandrasekhar H-function integral to < 0.5%.
   The `p` returned here is 1.47x-1.90x that opposition-excluded value over the
   campaign's w range. `disk_p_from_w_physical` below spells out the difference.

2. The coefficients 0.49 / 0.196 are NON-CANONICAL. The analytic
   multiple-scattering values are 0.5 / 0.16667. Their provenance is unknown.
   They change `p` by < 1% across the campaign's w range, so they are kept for
   EXACT reproduction of the existing outputs -- every `ti<TI>_alb<p>` sweep
   directory name on disk depends on the exact `p` these coefficients produce.
   DO NOT "fix" them to the canonical values.

3. The camera renderer uses per-body `B0 = 1.5, g = -0.35`, so its effective
   opposition bracket `(1 + B0) * P(0)` is ~7, while THIS inversion assumes a
   bracket of 1 (via the `+ w/8` hardwiring, which corresponds to bracket 2 in
   the geometric-albedo sense). That convention mismatch between the w->p
   inversion and the BRDF used to render is real; it is tracked separately and
   is NOT this module's job to reconcile.

Pure module: no side effects, no imports beyond `math`.

API:
    _r0(w)                 -> Hapke r0 = (1-gamma)/(1+gamma), gamma = sqrt(1-w)
    disk_p_from_w(w)       -> recovered disk-integrated geometric albedo p (see above)
    w_from_disk_p(p)       -> inverse of disk_p_from_w by bisection (strictly monotonic)
    alb_point(w, scale)    -> (p, w_consistent): p = scale * disk_p_from_w(w),
                              w_consistent back-solved from p. Identity in w at scale == 1.
"""
import math

__all__ = ["_r0", "disk_p_from_w", "w_from_disk_p", "alb_point",
           "disk_p_from_w_physical"]


def _r0(w):
    """Hapke diffusive-reflectance parameter r0 = (1 - gamma) / (1 + gamma),
    with gamma = sqrt(1 - w). Defined for single-scattering albedo w in [0, 1]."""
    gamma = math.sqrt(1.0 - w)
    return (1.0 - gamma) / (1.0 + gamma)


def disk_p_from_w(w):
    """Recovered disk-integrated geometric albedo p(w).

        p = w/8 + 0.49*r0 + 0.196*r0**2,  r0 = _r0(w)

    Verbatim from the physics review (see module docstring for the non-canonical-coefficient
    and opposition-convention caveats). Strictly monotonic increasing on (0, 1),
    which is what makes `w_from_disk_p` a well-posed bisection."""
    r0 = _r0(w)
    return w / 8.0 + 0.49 * r0 + 0.196 * r0 * r0


def w_from_disk_p(p, lo=0.0, hi=1.0, tol=1e-14, max_iter=200):
    """Invert disk_p_from_w by bisection: return w in [lo, hi] with
    disk_p_from_w(w) == p. disk_p_from_w is strictly monotonic increasing on
    (0, 1), so the root is unique. If p lies outside the achievable range the
    bisection converges to the nearer bound (p is effectively clamped)."""
    flo, fhi = disk_p_from_w(lo), disk_p_from_w(hi)
    if p <= flo:
        return lo
    if p >= fhi:
        return hi
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if disk_p_from_w(mid) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def alb_point(w, scale):
    """Return (p, w_consistent) for a sweep albedo point.

        p            = scale * disk_p_from_w(w)
        w_consistent = w_from_disk_p(p)

    `w_consistent` is the single-scattering albedo that reproduces the scaled
    geometric albedo `p`, so a downstream thermal run can be driven with a
    self-consistent w. At scale == 1.0 this is the identity in w
    (w_consistent == w to bisection tolerance)."""
    p = scale * disk_p_from_w(w)
    return p, w_from_disk_p(p)


def disk_p_from_w_physical(w, B0=0.0, P0=1.0):
    """PHYSICAL opposition-aware disk-integrated geometric albedo -- SEPARATE and
    UNUSED by default.

        p_phys = (w/8) * ((1 + B0) * P0 - 1) + 0.5 * r0 * (1 + r0/3)

    The `0.5 * r0 * (1 + r0/3)` term is the true opposition-EXCLUDED isotropic
    geometric albedo (the physics review: matches the exact Chandrasekhar H-function integral
    to < 0.5%); the `(w/8) * ((1 + B0)*P0 - 1)` term adds back only the excess
    opposition brightening beyond the isotropic single-scattering baseline.

    This is provided for reference / future reconciliation of the
    opposition-convention mismatch documented in the module docstring. It must
    NOT be silently swapped in for `disk_p_from_w`: it changes `p` for every `w`
    and therefore invalidates every existing `ti<TI>_alb<p>` sweep directory
    name on disk. Any switch to this convention is a deliberate, campaign-wide
    re-run, not a drop-in."""
    r0 = _r0(w)
    return (w / 8.0) * ((1.0 + B0) * P0 - 1.0) + 0.5 * r0 * (1.0 + r0 / 3.0)
