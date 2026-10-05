"""Periodic latitude-longitude grid, quadrature, harmonics and inertia tensor.

Every area integral in this package (averages, spherical-harmonic
projections, the thin-shell inertia tensor) goes through this module, so
that the quadrature is defined in ONE place.

Grid
----
* Longitude: ``n_lam`` equally spaced points ``lam_j = -pi + j * 2pi/n_lam``,
  j = 0 .. n_lam-1. The +pi column is NOT included: it is the same meridian
  as -pi. With uniform weights this is the periodic trapezoid rule, which is
  exact for every trigonometric polynomial of degree < n_lam. The former
  ``linspace(-pi, pi, n)`` grids counted the +-180 deg meridian twice; that
  double count is what produced a spurious tidal-only b11c of -0.08 km and
  biased 4*t22 by ~+2% (physics review, 2026-10-03).
* Colatitude: ``n_theta`` Gauss-Legendre nodes in cos(theta), with the
  Gauss-Legendre weights. Integrals of polynomials in cos(theta) of degree
  < 2 n_theta are exact, so a uniform field is exactly isotropic (no spurious
  t20, no spurious inertia anisotropy) and the degree <= 2 basis is exactly
  orthogonal. (A cell-centered midpoint rule leaves a spurious t20 of ~1e-3
  of the mean on a 46-row grid -- 23 m for a 23.7 km shell -- enough to fake
  an inertia instability.) Neither pole is a node; for odd n_theta the
  equator is.

Projections are normalized by the DISCRETE norm (sum of Y^2 w), so a field
that is exactly a single harmonic projects onto it with coefficient 1 to
round-off.

Harmonic convention (O&S 1989, unnormalized):
    b = b00 + b10 cos(theta) + (b11c cos(lam) + b11s sin(lam)) sin(theta)
        + t20 P20 + (t22 cos 2lam + t22s sin 2lam) * 3 sin^2(theta) + ...
with P20 = (3 cos^2(theta) - 1)/2 and P22 = 3 sin^2(theta).
"""

import functools

import numpy as np


def periodic_grid(n_theta=181, n_lam=360):
    """Return 1-D (theta, lam) arrays of the grid [rad].

    theta: Gauss-Legendre colatitudes, ascending (north to south).
    lam: -pi + j 2 pi / n_lam, j = 0 .. n_lam - 1.
    ``n_lam`` must be even so that the grid is symmetric under
    lam -> lam + pi (needed for exact cancellation of odd-m terms of a
    pi-periodic field such as the O&S tidal forcing).
    """
    if n_lam % 2:
        raise ValueError(f"n_lam must be even for a pi-symmetric grid, got {n_lam}")
    x, _ = np.polynomial.legendre.leggauss(n_theta)
    theta = np.arccos(x[::-1])
    lam = -np.pi + np.arange(n_lam) * (2.0 * np.pi / n_lam)
    return theta, lam


def check_periodic(lam):
    """Raise if ``lam`` is not a periodic grid (duplicated seam or non-uniform)."""
    lam = np.asarray(lam, dtype=float)
    n = lam.size
    step = 2.0 * np.pi / n
    if not np.allclose(np.diff(lam), step, rtol=0, atol=1e-9):
        raise ValueError(
            "longitude grid is not uniform with spacing 2*pi/n; a grid that "
            "includes both -pi and +pi double-counts the seam meridian. Use "
            "ojakangas_stevenson.grid.periodic_grid().")


def theta_weights(theta):
    """Gauss-Legendre weights for the colatitude nodes (they sum to 2)."""
    theta = np.asarray(theta, dtype=float)
    x, w = np.polynomial.legendre.leggauss(theta.size)
    if not np.allclose(theta, np.arccos(x[::-1]), rtol=0, atol=1e-12):
        raise ValueError("theta must be the Gauss-Legendre grid of periodic_grid()")
    return w[::-1]


@functools.lru_cache(maxsize=16)
def _cached_geometry(n_theta, n_lam):
    theta, lam = periodic_grid(n_theta, n_lam)
    w = theta_weights(theta)[:, None] * np.full(n_lam, 2.0 * np.pi / n_lam)[None, :]
    TH, LA = np.meshgrid(theta, lam, indexing='ij')
    st = np.sin(TH)
    xyz = (st * np.cos(LA), st * np.sin(LA), np.cos(TH))
    for a in (w,) + xyz:
        a.setflags(write=False)
    return theta, lam, w, xyz


def _geometry(theta, lam):
    """Weights and unit vectors for a grid of periodic_grid() (validated, cached)."""
    theta = np.asarray(theta, dtype=float)
    lam = np.asarray(lam, dtype=float)
    check_periodic(lam)
    th0, lm0, w, xyz = _cached_geometry(theta.size, lam.size)
    if not (np.allclose(theta, th0, rtol=0, atol=1e-12)
            and np.allclose(lam, lm0, rtol=0, atol=1e-12)):
        raise ValueError("(theta, lam) is not the grid of periodic_grid(); "
                         "colatitudes must be Gauss-Legendre nodes and longitudes "
                         "must start at -pi without the +pi column")
    return w, xyz


def area_weights(theta, lam):
    """2-D quadrature weights w[i, j] [sr], summing to 4 pi."""
    return _geometry(theta, lam)[0]


def area_mean(field, theta, lam):
    """Area-weighted mean of a (n_theta, n_lam) field."""
    w = area_weights(theta, lam)
    return float(np.sum(field * w) / np.sum(w))


def basis_functions(theta, lam):
    """Unnormalized real harmonics of degree <= 2 on the 2-D grid."""
    TH, LA = np.meshgrid(theta, lam, indexing='ij')
    ct, st = np.cos(TH), np.sin(TH)
    return {
        'b00': np.ones_like(TH),
        'b10': ct,
        'b11c': st * np.cos(LA),
        'b11s': st * np.sin(LA),
        'b20': 0.5 * (3.0 * ct**2 - 1.0),
        'b21c': 3.0 * st * ct * np.cos(LA),
        'b21s': 3.0 * st * ct * np.sin(LA),
        'b22c': 3.0 * st**2 * np.cos(2.0 * LA),
        'b22s': 3.0 * st**2 * np.sin(2.0 * LA),
    }


def harmonics(field, theta, lam):
    """Project ``field`` onto the degree <= 2 basis (same units as field).

    Returns a dict with keys b00, b10, b11c, b11s, b20, b21c, b21s, b22c,
    b22s, plus the O&S Table I combinations ``t20`` (= b20), ``t22``
    (= b22c), ``four_t22`` and ``t20_2t22``.
    """
    w = area_weights(theta, lam)
    out = {}
    for key, Y in basis_functions(theta, lam).items():
        out[key] = float(np.sum(field * Y * w) / np.sum(Y * Y * w))
    out['t20'] = out['b20']
    out['t22'] = out['b22c']
    out['four_t22'] = 4.0 * out['b22c']
    out['t20_2t22'] = out['b20'] + 2.0 * out['b22c']
    return out


def thin_shell_inertia_tensor(surface_density, theta, lam):
    """Geometric 3x3 inertia tensor of a thin surface layer on the unit sphere.

    I_ij = sum sigma (delta_ij - x_i x_j) w, in units of [sigma] * R^4 / R^0
    (multiply by R^4 to get kg m^2 when sigma is a surface density in
    kg/m^2 per unit R^2; for sigma = rho * b this is the uncompensated,
    rigidly supported load). Uses the periodic weights, so a field with
    exact north-south symmetry gives I_xz = I_yz = 0 to round-off.
    """
    w, (x, y, z) = _geometry(theta, lam)
    sw = surface_density * w
    I = np.empty((3, 3))
    I[0, 0] = np.sum((1.0 - x * x) * sw)
    I[1, 1] = np.sum((1.0 - y * y) * sw)
    I[2, 2] = np.sum((1.0 - z * z) * sw)
    I[0, 1] = I[1, 0] = -np.sum(x * y * sw)
    I[0, 2] = I[2, 0] = -np.sum(x * z * sw)
    I[1, 2] = I[2, 1] = -np.sum(y * z * sw)
    return I


def deviatoric(I):
    """Traceless part of a 3x3 tensor."""
    return I - np.trace(I) / 3.0 * np.eye(3)
