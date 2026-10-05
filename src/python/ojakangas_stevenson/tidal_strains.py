"""Tidal strain rates in a thin shell on Europa (Appendix B of O&S 1989).

Implements the strain tensor components from Eqs B.20-B.25 for a thin
incompressible elastic shell overlying a fluid synchronously rotating
satellite in an elliptical orbit.

The key output is the time-averaged squared strain rate <edot_ij * edot_ij>,
which enters the dissipation integral.
"""

import numpy as np
from . import constants as C


def legendre_P2(cos_theta):
    """Second-degree Legendre polynomial P2(x) = (3x^2 - 1) / 2."""
    return 0.5 * (3.0 * cos_theta**2 - 1.0)


def f_function(M, lam):
    """Tidal forcing function f(M, lambda) (Eq B.25).

    f = 3*cos(2*lam)*cos(M) + 4*sin(2*lam)*sin(M)

    Parameters
    ----------
    M : float or ndarray
        Mean anomaly [rad].
    lam : float or ndarray
        Longitude [rad].
    """
    return 3.0 * np.cos(2.0 * lam) * np.cos(M) + 4.0 * np.sin(2.0 * lam) * np.sin(M)


def strain_components(theta, lam, M, g2, g3):
    """Compute strain tensor components at given (theta, lam, M).

    From Eqs B.21-B.24:

    e_tt = g2 * [2*f*cos(2*theta) + 6*cos(2*theta)*cos(M)]
         + g3 * [f*sin^2(theta) - 2*P2(cos theta)*cos(M)]

    e_ll = g2 * [2*(cos^2(theta) - 2)*f + 6*cos^2(theta)*cos(M)]
         + g3 * [f*sin^2(theta) - 2*P2(cos theta)*cos(M)]

    e_tl = g2 * cos(theta) * [8*cos(2*lam)*sin(M) - 6*sin(2*lam)*cos(M)]

    For incompressible body: e_rr = -(e_tt + e_ll) (B.20, verified via B.25).

    Parameters
    ----------
    theta : float or ndarray
        Colatitude [rad].
    lam : float or ndarray
        Longitude [rad].
    M : float or ndarray
        Mean anomaly [rad].
    g2, g3 : float
        Characteristic strain amplitudes.

    Returns
    -------
    e_tt, e_ll, e_tl : float or ndarray
        Strain components (dimensionless).
    """
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    cos_M = np.cos(M)
    sin_M = np.sin(M)

    f = f_function(M, lam)
    P2 = legendre_P2(cos_t)

    # Common subexpression for g3 terms
    g3_term = g3 * (f * sin_t**2 - 2.0 * P2 * cos_M)

    e_tt = g2 * (2.0 * f * np.cos(2.0 * theta) + 6.0 * np.cos(2.0 * theta) * cos_M) + g3_term
    e_ll = g2 * (2.0 * (cos_t**2 - 2.0) * f + 6.0 * cos_t**2 * cos_M) + g3_term
    e_tl = g2 * cos_t * (8.0 * np.cos(2.0 * lam) * sin_M - 6.0 * np.sin(2.0 * lam) * cos_M)

    return e_tt, e_ll, e_tl


def time_averaged_strain_squared(theta, lam, g2=None, g3=None):
    """Compute time-averaged <e_ij * e_ij> over one orbital period.

    For the Maxwell rheology (Eq 3.5), the needed quantity is:
        <edot_ij^2> = n^2 * <e_ij * e_ij>

    where <e_ij * e_ij> = <e_rr^2 + e_tt^2 + e_ll^2 + 2*e_tl^2>
    with e_rr = -(e_tt + e_ll) from incompressibility.

    Each strain component has the form e = c_cos * cos(M) + c_sin * sin(M),
    so <e^2> = (c_cos^2 + c_sin^2) / 2.

    Parameters
    ----------
    theta : float or ndarray
        Colatitude [rad].
    lam : float or ndarray
        Longitude [rad].
    g2 : float, optional
        Characteristic strain amplitude. Default: computed from constants.
    g3 : float, optional
        Strain amplitude. Default: computed from constants.

    Returns
    -------
    float or ndarray
        Time-averaged <e_ij * e_ij> (dimensionless).
        Multiply by n^2 to get <edot_ij * edot_ij> [s^-2].
    """
    if g2 is None:
        g2 = C.g2()
    if g3 is None:
        g3 = C.g3()

    theta = np.asarray(theta, dtype=float)
    lam = np.asarray(lam, dtype=float)

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    cos2t = np.cos(2.0 * theta)
    cos2l = np.cos(2.0 * lam)
    sin2l = np.sin(2.0 * lam)
    P2 = legendre_P2(cos_t)

    # Decompose e_tt = c_tt_cos * cos(M) + c_tt_sin * sin(M)
    # From e_tt: g2*(2*f*cos2t + 6*cos^2*cos(M)) + g3*(f*sin^2 - 2*P2*cos(M))
    # f = 3*cos2l*cosM + 4*sin2l*sinM
    # => f*cos2t = cos2t*(3*cos2l*cosM + 4*sin2l*sinM)
    # cos(M) terms: g2*(2*3*cos2l*cos2t + 6*cos^2) + g3*(3*cos2l*sin^2 - 2*P2)
    # sin(M) terms: g2*(2*4*sin2l*cos2t) + g3*(4*sin2l*sin^2)

    c_tt_cos = g2 * (6.0 * cos2l * cos2t + 6.0 * cos2t) + g3 * (3.0 * cos2l * sin_t**2 - 2.0 * P2)
    c_tt_sin = g2 * (8.0 * sin2l * cos2t) + g3 * (4.0 * sin2l * sin_t**2)

    # Decompose e_ll = c_ll_cos * cos(M) + c_ll_sin * sin(M)
    # e_ll: g2*(2*(cos^2-2)*f + 6*cos^2*cos(M)) + g3*(f*sin^2 - 2*P2*cos(M))
    c_ll_cos = g2 * (6.0 * cos2l * (cos_t**2 - 2.0) + 6.0 * cos_t**2) + g3 * (3.0 * cos2l * sin_t**2 - 2.0 * P2)
    c_ll_sin = g2 * (8.0 * sin2l * (cos_t**2 - 2.0)) + g3 * (4.0 * sin2l * sin_t**2)

    # Decompose e_tl = c_tl_cos * cos(M) + c_tl_sin * sin(M)
    # e_tl = g2 * cos_t * (8*cos2l*sinM - 6*sin2l*cosM)
    c_tl_cos = -6.0 * g2 * cos_t * sin2l
    c_tl_sin = 8.0 * g2 * cos_t * cos2l

    # e_rr = -(e_tt + e_ll) from incompressibility
    c_rr_cos = -(c_tt_cos + c_ll_cos)
    c_rr_sin = -(c_tt_sin + c_ll_sin)

    # <e^2> = (c_cos^2 + c_sin^2) / 2
    avg_err2 = 0.5 * (c_rr_cos**2 + c_rr_sin**2)
    avg_ett2 = 0.5 * (c_tt_cos**2 + c_tt_sin**2)
    avg_ell2 = 0.5 * (c_ll_cos**2 + c_ll_sin**2)
    avg_etl2 = 0.5 * (c_tl_cos**2 + c_tl_sin**2)

    # Total: <e_ij e_ij> = <e_rr^2> + <e_tt^2> + <e_ll^2> + 2*<e_tl^2>
    # (factor of 2 for off-diagonal because e_tl = e_lt)
    return avg_err2 + avg_ett2 + avg_ell2 + 2.0 * avg_etl2


def time_averaged_strain_squared_numerical(theta, lam, g2=None, g3=None,
                                            n_samples=360):
    """Numerical verification of time-averaged strain squared.

    Evaluates strain components at n_samples equally spaced M values
    and computes the average. For validation against the analytic formula.
    """
    if g2 is None:
        g2 = C.g2()
    if g3 is None:
        g3 = C.g3()

    theta = np.asarray(theta, dtype=float)
    lam = np.asarray(lam, dtype=float)

    M_vals = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    total = np.zeros_like(theta + lam, dtype=float)

    for M in M_vals:
        e_tt, e_ll, e_tl = strain_components(theta, lam, M, g2, g3)
        e_rr = -(e_tt + e_ll)  # incompressibility
        total += e_rr**2 + e_tt**2 + e_ll**2 + 2.0 * e_tl**2

    return total / n_samples


def strain_squared_grid(n_theta=181, n_lam=360, g2=None, g3=None):
    """Compute <e_ij * e_ij> on the periodic (theta, lam) grid (grid.periodic_grid).

    Returns
    -------
    theta : ndarray, shape (n_theta,)
    lam : ndarray, shape (n_lam,)
    eij2 : ndarray, shape (n_theta, n_lam)
    """
    if g2 is None:
        g2 = C.g2()
    if g3 is None:
        g3 = C.g3()

    from .grid import periodic_grid
    theta, lam = periodic_grid(n_theta, n_lam)
    THETA, LAM = np.meshgrid(theta, lam, indexing='ij')

    eij2 = time_averaged_strain_squared(THETA, LAM, g2, g3)
    return theta, lam, eij2


def time_averaged_strain_power(theta, lam, exponent, g2=None, g3=None,
                                n_samples=360):
    """Compute time-averaged [0.5 * e_ij * e_ij]^exponent over one orbit.

    Needed for the generalized flow law (Eq 4.7) where the exponent is
    (n_G + 1) / (2 * n_G) which is not 1 in general.

    This must be done numerically since <f(x)^p> != <f(x)>^p for p != 1.

    Parameters
    ----------
    theta, lam : float or ndarray
        Colatitude and longitude [rad].
    exponent : float
        Power to raise (0.5 * e_ij^2) to.
    g2, g3 : float, optional
        Strain amplitudes.
    n_samples : int
        Number of orbital samples for numerical averaging.

    Returns
    -------
    float or ndarray
        <[0.5 * e_ij * e_ij]^exponent>
    """
    if g2 is None:
        g2 = C.g2()
    if g3 is None:
        g3 = C.g3()

    theta = np.asarray(theta, dtype=float)
    lam = np.asarray(lam, dtype=float)

    M_vals = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    total = np.zeros_like(theta + lam, dtype=float)

    for M in M_vals:
        e_tt, e_ll, e_tl = strain_components(theta, lam, M, g2, g3)
        e_rr = -(e_tt + e_ll)  # incompressibility
        eij2 = e_rr**2 + e_tt**2 + e_ll**2 + 2.0 * e_tl**2
        total += (0.5 * eij2) ** exponent

    return total / n_samples
