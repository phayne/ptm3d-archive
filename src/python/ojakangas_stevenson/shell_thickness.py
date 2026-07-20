"""Ice shell thickness solver (Section 2 of O&S 1989).

Computes equilibrium ice shell thickness b(theta, lambda) from Eq 2.8:

    b = a1 * ln(T_M / T_s) / [D(theta, lambda) + H]

where D is the dissipation integral from the rheology model and H is
the basal heat flux from the core.

Also computes second-degree spherical harmonic decomposition and the
resulting contributions to Europa's principal moment differences
(Section 5, Eqs 5.7-5.10).
"""

import numpy as np
from scipy.special import lpmv

from . import constants as C
from .surface_temperature import OSSurfaceTemperature
from .tidal_strains import (time_averaged_strain_squared,
                             time_averaged_strain_power,
                             strain_squared_grid)
from .rheology import MaxwellRheology, FlowLawRheology


class IceShellModel:
    """Ojakangas & Stevenson (1989) equilibrium ice shell thickness model.

    Parameters
    ----------
    rheology : MaxwellRheology or FlowLawRheology
        Rheology model to compute the dissipation integral.
    surface_temp_model : object, optional
        Model providing T_s(theta). Must have a ``temperature(theta)``
        method. If None, uses the default O&S surface temperature.
    a1 : float
        Thermal conductivity parameter [W K / m]. Default: 488.
    T_melt : float
        Melting temperature [K]. Default: 273.
    H_flux : float
        Basal heat flux from the core [W/m^2]. Default: 0.
    """

    def __init__(self, rheology, surface_temp_model=None,
                 a1=C.A1_CONDUCTIVITY, T_melt=C.T_MELT, H_flux=0.0,
                 c_r=None):
        self.rheology = rheology
        self.surface_temp = surface_temp_model or OSSurfaceTemperature()
        self.a1 = a1
        self.T_melt = T_melt
        self.H_flux = H_flux
        self.c_r = c_r  # None means use module-level C.C_R default

        # Cache for strain field
        self._strain_cache = {}

    def _compute_dissipation(self, theta, lam):
        """Compute the dissipation integral D(theta, lam).

        Dispatches to the appropriate method depending on rheology type.
        """
        g2 = C.g2(c_r=self.c_r)
        g3 = C.g3(c_r=self.c_r)

        if isinstance(self.rheology, MaxwellRheology):
            # For Maxwell: D depends on n^2 * <e_ij^2>
            eij2 = time_averaged_strain_squared(theta, lam, g2, g3)
            strain_rate_sq = C.MEAN_MOTION**2 * eij2
            return self.rheology.dissipation_integral(strain_rate_sq)

        elif isinstance(self.rheology, FlowLawRheology):
            # For flow law: D depends on <[0.5 * edot_ij^2]^exponent>
            # which requires numerical time averaging
            exponent = self.rheology.strain_exponent
            # We need edot, so multiply strain by n
            # <[0.5 * (n*e_ij)^2]^p> = n^{2p} * <[0.5 * e_ij^2]^p>
            strain_power = time_averaged_strain_power(
                theta, lam, exponent, g2, g3, n_samples=360)
            # Scale by n^{2*exponent}
            strain_rate_power = C.MEAN_MOTION**(2.0 * exponent) * strain_power
            return self.rheology.dissipation_integral(strain_rate_power)

        else:
            raise TypeError(f"Unknown rheology type: {type(self.rheology)}")

    def thickness(self, theta, lam):
        """Compute shell thickness at (theta, lam).

        Eq 2.8 (implicit): b^2 * D_T / ln(T_M/T_s) + b * H = a1 * ln(T_M/T_s)

        where D_T = integral q(T) dT/T [W/m^3] is the dissipation integral
        from Eq 2.6. The change of variables from z to T introduces a factor
        of b/ln(T_M/T_s), making the equation quadratic in b.

        For H=0: b = ln(T_M/T_s) * sqrt(a1 / D_T)
        For H>0: solve quadratic A*b^2 + B*b + C = 0 with
            A = D_T / ln(T_M/T_s)
            B = H
            C = -a1 * ln(T_M/T_s)

        Parameters
        ----------
        theta : float or ndarray
            Colatitude [rad], range [0, pi].
        lam : float or ndarray
            Longitude [rad].

        Returns
        -------
        float or ndarray
            Shell thickness [m].
        """
        theta = np.asarray(theta, dtype=float)
        lam = np.asarray(lam, dtype=float)

        T_s = self.surface_temp.temperature(theta)
        D_T = self._compute_dissipation(theta, lam)

        ln_ratio = np.log(self.T_melt / T_s)

        # Quadratic: (D_T / ln_ratio) * b^2 + H * b - a1 * ln_ratio = 0
        A = D_T / ln_ratio
        B = self.H_flux
        C = -self.a1 * ln_ratio

        # b = (-B + sqrt(B^2 - 4AC)) / (2A)
        discriminant = B**2 - 4.0 * A * C  # Always positive since A>0, C<0
        b = (-B + np.sqrt(discriminant)) / (2.0 * A)

        # Clamp to avoid negative or absurdly large values
        b = np.clip(b, 0, 200e3)
        return b

    def thickness_grid(self, n_theta=181, n_lam=361):
        """Compute shell thickness on a regular (theta, lam) grid.

        Parameters
        ----------
        n_theta : int
            Number of colatitude points.
        n_lam : int
            Number of longitude points.

        Returns
        -------
        theta : ndarray, shape (n_theta,)
            Colatitude values [rad].
        lam : ndarray, shape (n_lam,)
            Longitude values [rad], range [-pi, pi].
        b : ndarray, shape (n_theta, n_lam)
            Shell thickness [m].
        """
        theta = np.linspace(0.001, np.pi - 0.001, n_theta)
        lam = np.linspace(-np.pi, np.pi, n_lam)
        THETA, LAM = np.meshgrid(theta, lam, indexing='ij')

        b = self.thickness(THETA, LAM)
        return theta, lam, b

    def average_thickness(self, n_theta=181, n_lam=361):
        """Compute area-weighted average shell thickness [m].

        <b> = (1/4pi) * integral b * sin(theta) d(theta) d(lam)
        """
        theta, lam, b = self.thickness_grid(n_theta, n_lam)
        dtheta = theta[1] - theta[0]
        dlam = lam[1] - lam[0]

        # Area element: sin(theta) * d(theta) * d(lam)
        sin_theta = np.sin(theta)
        # Integrate: sum over (theta, lam) with proper weighting
        integral = np.sum(b * sin_theta[:, np.newaxis]) * dtheta * dlam
        # Normalize by total solid angle 4*pi
        return integral / (4.0 * np.pi)

    def spherical_harmonic_decomposition(self, n_theta=361, n_lam=721):
        """Compute second-degree spherical harmonic components of thickness.

        Projects b(theta, lam) onto:
          t20: coefficient of Y20 ~ P20(cos theta) = (3cos^2(theta) - 1)/2
          t22: coefficient of Y22 ~ P22(cos theta)*cos(2*lam) = 3*sin^2(theta)*cos(2*lam)

        Uses the orthogonality relations:
          t20 = (5/(4pi)) * integral b * P20 * sin(theta) d(theta) d(lam)
          t22 = (5/(48pi)) * integral b * P22*cos(2lam) * sin(theta) d(theta) d(lam)

        Note: We use real (unnormalized) spherical harmonics as in the paper.
        The normalization factors come from:
          integral P20^2 * sin(theta) d(theta) d(lam) = 4*pi/5
          integral (P22*cos(2lam))^2 * sin(theta) d(theta) d(lam) = 48*pi/5

        Returns
        -------
        t20 : float
            Zonal (m=0) second-degree component [m].
        t22 : float
            Sectoral (m=2) second-degree component [m].
        """
        theta, lam, b = self.thickness_grid(n_theta, n_lam)
        dtheta = theta[1] - theta[0]
        dlam = lam[1] - lam[0]

        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        # P20(cos theta) = (3*cos^2(theta) - 1) / 2
        P20 = 0.5 * (3.0 * cos_t**2 - 1.0)
        # P22(cos theta) = 3 * sin^2(theta)
        P22 = 3.0 * sin_t**2

        cos_2lam = np.cos(2.0 * lam)

        # t20: project onto P20
        # t20 = (5/4pi) * integral b * P20 * sin(theta) d(theta) d(lam)
        # The lam integral of b*P20 over [0, 2pi] just multiplies by the lam range
        integrand_20 = b * P20[:, np.newaxis] * sin_t[:, np.newaxis]
        t20 = (5.0 / (4.0 * np.pi)) * np.sum(integrand_20) * dtheta * dlam

        # t22: project onto P22 * cos(2*lam)
        # t22 = (5/48pi) * integral b * P22 * cos(2*lam) * sin(theta) d(theta) d(lam)
        integrand_22 = b * P22[:, np.newaxis] * cos_2lam[np.newaxis, :] * sin_t[:, np.newaxis]
        t22 = (5.0 / (48.0 * np.pi)) * np.sum(integrand_22) * dtheta * dlam

        return t20, t22

    def moment_differences(self, n_theta=361, n_lam=721):
        """Compute contributions to Europa's principal moment differences.

        Eqs 5.7-5.10:
          Delta((B-A)/C) ~ (rho_m - rho_c)/rho_m * rho_c/rho_avg * (b/a)^4 * 4*t22
          Delta((B-C)/A) ~ (rho_m - rho_c)/rho_m * rho_c/rho_avg * (b/a)^4 * (t20 + 2*t22)

        Approximate forms (Eqs 5.9-5.10):
          Delta((B-A)/C) ~ 1e-7 * (4*t22 / 1 km)
          Delta((B-C)/A) ~ 1e-7 * ((t20 + 2*t22) / 1 km)

        Returns
        -------
        dict with keys:
            't20' : float, second-degree zonal component [m]
            't22' : float, second-degree sectoral component [m]
            '4t22_km' : float, 4*t22 in km (Table I column)
            't20_plus_2t22_km' : float, (t20 + 2*t22) in km (Table I column)
            'delta_BA_C' : float, contribution to (B-A)/C
            'delta_BC_A' : float, contribution to (B-C)/A
            'polar_wander_unstable' : bool, True if delta_BC_A > 0
        """
        t20, t22 = self.spherical_harmonic_decomposition(n_theta, n_lam)

        t20_km = t20 / 1000.0
        t22_km = t22 / 1000.0

        # Approximate formulas (Eqs 5.9, 5.10)
        delta_BA_C = 1.0e-7 * (4.0 * t22_km)
        delta_BC_A = 1.0e-7 * (t20_km + 2.0 * t22_km)

        return {
            't20': t20,
            't22': t22,
            '4t22_km': 4.0 * t22_km,
            't20_plus_2t22_km': t20_km + 2.0 * t22_km,
            'delta_BA_C': delta_BA_C,
            'delta_BC_A': delta_BC_A,
            'polar_wander_unstable': delta_BC_A > 0,
        }
