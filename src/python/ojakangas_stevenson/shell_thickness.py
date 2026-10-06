"""Ice shell thickness solver (Section 2 of O&S 1989, thermal paper).

THE thickness law of this package. Every script that needs b(theta, lam)
must call ``equilibrium_thickness`` (or ``IceShellModel``); do not copy the
formula.

Eq. 2.8 (thermal paper p. 223), with D = integral of q(T) dT/T (Eq. 2.6):

    b = ln(T_M / T_s) / [ (2/a1) D + (H/a1)^2 ]^(1/2)

It follows from integrating Eq. 2.4 once (Eq. 2.5) and taking the gradient
of y = ln T as constant through the shell (Eq. 2.7). Before 2026-10 the code
solved b^2 D/ln + b H = a1 ln instead: it dropped the factor 2 on D and added
H linearly rather than in quadrature. That made the shell ~sqrt(2) too thick
and forced the accidental fit C_R = 0.85; with Eq. 2.8, the paper's
C_R = 0.5 reproduces all of Table I (physics review, 2026-10-03).

``method='exact'`` evaluates the exact first integral of Eqs. 2.4/2.5
instead of approximating y' as constant:

    (dy/dz)^2 at T = (2/a1) D_above(T) + (H/a1)^2,
    D_above(T) = integral_T^T_M q dT'/T' = D * F(T),
    b = integral_{T_s}^{T_M} dT / (T dy/dz),

with F(T) from the rheology (``dissipation_fraction_above``). The exact
shell is ~5-8% thicker than Eq. 2.8 for the Maxwell rheology.
"""

import numpy as np

from . import constants as C
from . import grid as G
from .surface_temperature import OSSurfaceTemperature
from .tidal_strains import (time_averaged_strain_squared,
                            time_averaged_strain_power)
from .rheology import MaxwellRheology, FlowLawRheology

METHODS = ('eq28', 'exact')

# Gauss-Legendre nodes for the exact integral, mapped to [0, 1] in
# s = sqrt(u/U), u = ln(T_M/T): composite rule, 24 panels x 8 nodes.
_GL_X, _GL_W = np.polynomial.legendre.leggauss(8)
_N_PANELS = 24
_EDGES = np.linspace(0.0, 1.0, _N_PANELS + 1)
_S_NODES = np.concatenate([0.5 * (b - a) * _GL_X + 0.5 * (b + a)
                           for a, b in zip(_EDGES[:-1], _EDGES[1:])])
_S_WEIGHTS = np.concatenate([0.5 * (b - a) * _GL_W
                             for a, b in zip(_EDGES[:-1], _EDGES[1:])])


def _check_inputs(T_s, D, H, T_melt):
    T_s = np.asarray(T_s, dtype=float)
    D = np.asarray(D, dtype=float)
    bad_T = ~np.isfinite(T_s) | (T_s <= 0) | (T_s >= T_melt)
    if np.any(bad_T):
        raise ValueError(
            f"{int(np.sum(bad_T))} surface temperature(s) are non-finite or outside "
            f"(0, T_melt={T_melt}) K; the shell thickness is undefined there")
    if np.any(~np.isfinite(D)) or np.any(D < 0):
        raise ValueError("dissipation integral D must be finite and >= 0")
    if np.any((D == 0) & (H == 0)):
        raise ValueError("D = 0 and H = 0: no heat reaches the surface, b is infinite")
    return T_s, D


def equilibrium_thickness(T_s, D, H=0.0, a1=None, T_melt=None,
                          method='eq28', rheology=None):
    """Equilibrium shell thickness b [m] (O&S Eq. 2.8, or its exact form).

    Parameters
    ----------
    T_s : float or ndarray
        Surface temperature [K] (the shell's top boundary; for a time-varying
        surface use the ARITHMETIC time mean, project decision 2026-10-04).
    D : float or ndarray
        Dissipation integral, Eq. 2.6 [W/m^2 in the a1 convention].
    H : float
        Basal (core) heat flux [W/m^2].
    a1 : float, optional
        Conductivity coefficient in k = a1/T [W/m]. Default: active set.
    T_melt : float, optional
        Basal (melting) temperature [K]. Default: 273.
    method : {'eq28', 'exact'}
        'eq28' is the paper's closed form; 'exact' integrates the first
        integral of Eqs. 2.4/2.5 and requires ``rheology``.
    rheology : MaxwellRheology or FlowLawRheology
        Needed for method='exact' (supplies F(T)).

    Raises
    ------
    ValueError
        For an unknown method, a missing rheology, T_s outside (0, T_melt),
        or D = H = 0. Nothing is clipped.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    a1 = C.A1_CONDUCTIVITY if a1 is None else a1
    T_melt = C.T_MELT if T_melt is None else T_melt
    T_s, D = _check_inputs(T_s, D, H, T_melt)
    U = np.log(T_melt / T_s)
    if method == 'eq28':
        return U / np.sqrt(2.0 * D / a1 + (H / a1) ** 2)

    if rheology is None:
        raise ValueError("method='exact' needs the rheology (for F(T))")
    shape = np.broadcast(U, D).shape
    Uf = np.broadcast_to(U, shape).reshape(-1, 1)
    Df = np.broadcast_to(D, shape).reshape(-1, 1)
    s = _S_NODES[None, :]
    u = Uf * s**2                                   # u = ln(T_M/T)
    F = rheology.dissipation_fraction_above(T_melt * np.exp(-u), T_melt)
    grad2 = 2.0 * Df / a1 * F + (H / a1) ** 2       # (dy/dz)^2
    integrand = 2.0 * Uf * s / np.sqrt(grad2)       # du = 2 U s ds
    b = np.sum(integrand * _S_WEIGHTS[None, :], axis=1)
    return b.reshape(shape)


def table1_case(key):
    """(rheology, H) for one of the four Table I cases.

    key: 'maxwell_h0', 'maxwell_h10', 'flow_ng3', 'flow_ng52'.
    """
    if key in ('maxwell_h0', 'maxwell_h10'):
        H = C.H_FLUX_10 if key == 'maxwell_h10' else C.H_FLUX_ZERO
        return MaxwellRheology(), H
    if key == 'flow_ng3':
        return FlowLawRheology(n_G=C.GLEN_EXPONENT_1, n_A=C.ARRHENIUS_EXPONENT,
                               A0_cgs=C.A0_FLOW_1_CGS), C.H_FLUX_ZERO
    if key == 'flow_ng52':
        return FlowLawRheology(n_G=C.GLEN_EXPONENT_2, n_A=C.ARRHENIUS_EXPONENT,
                               A0_cgs=C.A0_FLOW_2_CGS), C.H_FLUX_ZERO
    raise ValueError(f"unknown Table I case {key!r}")


TABLE1_CASES = ('maxwell_h0', 'maxwell_h10', 'flow_ng3', 'flow_ng52')


def dissipation_field(rheology, theta, lam, c_r=None, n_samples=360):
    """Dissipation integral D(theta, lam) [Eq. 2.6] for a rheology.

    ``theta`` and ``lam`` are point coordinates (tidal-frame colatitude and
    longitude) that broadcast against each other; mesh 1-D grid vectors with
    np.meshgrid(..., indexing='ij') first.
    """
    theta, lam = np.broadcast_arrays(np.asarray(theta, dtype=float),
                                     np.asarray(lam, dtype=float))
    g2 = C.g2(c_r=c_r)
    g3 = C.g3(c_r=c_r)
    if isinstance(rheology, MaxwellRheology):
        eij2 = time_averaged_strain_squared(theta, lam, g2, g3)
        return rheology.dissipation_integral(C.MEAN_MOTION**2 * eij2)
    if isinstance(rheology, FlowLawRheology):
        p = rheology.strain_exponent
        sp = time_averaged_strain_power(theta, lam, p, g2, g3, n_samples=n_samples)
        return rheology.dissipation_integral(C.MEAN_MOTION**(2.0 * p) * sp)
    raise TypeError(f"Unknown rheology type: {type(rheology)}")


class IceShellModel:
    """Ojakangas & Stevenson (1989) equilibrium ice shell thickness model.

    Parameters
    ----------
    rheology : MaxwellRheology or FlowLawRheology
        Rheology model to compute the dissipation integral.
    surface_temp_model : object, optional
        Provides ``temperature_2d(theta, lam)`` (preferred) or
        ``temperature(theta)``. Default: O&S Appendix A model.
    a1 : float, optional
        Conductivity coefficient [W/m]. Default: the active parameter set.
    T_melt : float
        Melting temperature [K]. Default: 273.
    H_flux : float
        Basal heat flux from the core [W/m^2]. Default: 0.
    c_r : float, optional
        Tidal correction factor. Default: constants.C_R (= 0.5).
    method : {'eq28', 'exact'}
        Thickness law; see equilibrium_thickness().
    """

    def __init__(self, rheology, surface_temp_model=None,
                 a1=None, T_melt=C.T_MELT, H_flux=0.0,
                 c_r=None, method='eq28'):
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {method!r}")
        self.rheology = rheology
        self.surface_temp = surface_temp_model or OSSurfaceTemperature()
        self.a1 = C.A1_CONDUCTIVITY if a1 is None else a1
        self.T_melt = T_melt
        self.H_flux = H_flux
        self.c_r = c_r
        self.method = method

    def _compute_dissipation(self, theta, lam):
        return dissipation_field(self.rheology, theta, lam, c_r=self.c_r)

    def thickness(self, theta, lam):
        """Shell thickness [m] at colatitude theta, longitude lam [rad].

        Uses the zonal surface temperature T_s(theta); for a 2-D surface
        temperature field use thickness_grid().
        """
        theta = np.asarray(theta, dtype=float)
        lam = np.asarray(lam, dtype=float)
        T_s = self.surface_temp.temperature(theta)
        D = self._compute_dissipation(theta, lam)
        return equilibrium_thickness(T_s, D, self.H_flux, a1=self.a1,
                                     T_melt=self.T_melt, method=self.method,
                                     rheology=self.rheology)

    def thickness_grid(self, n_theta=181, n_lam=360):
        """Thickness on the periodic grid (grid.periodic_grid).

        Returns
        -------
        theta : (n_theta,) colatitude cell centers [rad]
        lam : (n_lam,) longitudes in [-pi, pi) [rad]
        b : (n_theta, n_lam) shell thickness [m]
        """
        theta, lam = G.periodic_grid(n_theta, n_lam)
        THETA, LAM = np.meshgrid(theta, lam, indexing='ij')
        if hasattr(self.surface_temp, 'temperature_2d'):
            T_s = np.asarray(self.surface_temp.temperature_2d(theta, lam), dtype=float)
        else:
            T_s = self.surface_temp.temperature(THETA)
        D = self._compute_dissipation(THETA, LAM)
        b = equilibrium_thickness(T_s, D, self.H_flux, a1=self.a1,
                                  T_melt=self.T_melt, method=self.method,
                                  rheology=self.rheology)
        return theta, lam, b

    def average_thickness(self, n_theta=181, n_lam=360):
        """Area-weighted average shell thickness [m]."""
        theta, lam, b = self.thickness_grid(n_theta, n_lam)
        return G.area_mean(b, theta, lam)

    def harmonics(self, n_theta=181, n_lam=360):
        """Degree <= 2 harmonics of b [m]; see grid.harmonics()."""
        theta, lam, b = self.thickness_grid(n_theta, n_lam)
        return G.harmonics(b, theta, lam)

    def spherical_harmonic_decomposition(self, n_theta=181, n_lam=360):
        """(t20, t22) [m] in the paper's unnormalized convention."""
        h = self.harmonics(n_theta, n_lam)
        return h['t20'], h['t22']

    def moment_differences(self, n_theta=181, n_lam=360, bc_over_d=10.0, normalization='C_body'):
        """Table I combinations and the moment differences of the shell (exact Eq. 2.18 chain).

        Delta(B-A) and Delta(B-C) come from the physical (bottom-loaded,
        compensated) shell tensor, polar_wander.shell_inertia_tensor, with
        b_c = <b> and d = b_c / bc_over_d, divided by the ``normalization`` C of
        polar_wander.nsr_normalizations ('C_body' default, 'I_0', 'C_s').
        Before 2026-10-06 this returned O&S Eqs. 5.9-5.10 as written,
        1e-7 (4 t22 / 1 km) and 1e-7 ((t20 + 2 t22) / 1 km): an order-of-magnitude
        rounding that is 12-18x low against the exact chain normalized to C_body
        (Fermi demo spec sec. 6.2). That form is kept as '*_os_eq59'.
        'polar_wander_unstable' is the no-fossil sign test (t20 + 2 t22 > 0).
        """
        from . import polar_wander as PW
        theta, lam, b = self.thickness_grid(n_theta, n_lam)
        h = G.harmonics(b, theta, lam)
        t20, t22 = h['t20'], h['t22']
        t20_km, t22_km = t20 / 1000.0, t22 / 1000.0
        bc = G.area_mean(b, theta, lam)
        p = PW.OSPWParams(shell_thickness=bc, lithosphere_thickness=bc / bc_over_d)
        I = PW.shell_inertia_tensor(b, theta, lam, p)
        Cn = PW.nsr_normalizations(p, bc)[normalization]
        dBA = float(I[1, 1] - I[0, 0])
        dBC = float(I[1, 1] - I[2, 2])
        return {
            't20': t20,
            't22': t22,
            '4t22_km': 4.0 * t22_km,
            't20_plus_2t22_km': t20_km + 2.0 * t22_km,
            'delta_BA': dBA,
            'delta_BC': dBC,
            'delta_BA_C': dBA / Cn,
            'delta_BC_A': dBC / Cn,
            'normalization': normalization,
            'delta_BA_C_os_eq59': 1.0e-7 * (4.0 * t22_km),
            'delta_BC_A_os_eq59': 1.0e-7 * (t20_km + 2.0 * t22_km),
            'polar_wander_unstable': t20_km + 2.0 * t22_km > 0,
        }
