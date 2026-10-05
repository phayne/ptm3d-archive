"""Surface temperature model from Ojakangas & Stevenson (1989) Appendix A.

Computes the annual-average radiative equilibrium surface temperature of Europa
as a function of colatitude, accounting for the ~3 deg effective obliquity.
"""

import numpy as np
from . import constants as C


class OSSurfaceTemperature:
    """O&S (1989) Appendix A surface temperature model.

    Parameters
    ----------
    S0 : float
        Solar flux at Jupiter orbit, pi*F_sun [W/m^2]. Default: 50.
    albedo : float
        Bond albedo. Default: 0.5.
    obliquity_deg : float
        Effective obliquity [degrees]. Default: 3.0.
    polar_flux : {'A15', 'legacy-max'}
        Insolation poleward of the Arctic circles: Eq. A.15 (default) or the
        pre-2026-10 max(sin(theta), i) form. See annual_average_flux().
    """

    def __init__(self, S0=C.SOLAR_FLUX_JUPITER, albedo=C.BOND_ALBEDO,
                 obliquity_deg=3.0, polar_flux='A15'):
        if polar_flux not in ('A15', 'legacy-max'):
            raise ValueError(f"polar_flux must be 'A15' or 'legacy-max', got {polar_flux!r}")
        self.S0 = S0
        self.albedo = albedo
        self.obliquity = np.radians(obliquity_deg)
        self.polar_flux = polar_flux

    def annual_average_flux(self, theta):
        """Annual-average insolation F_A(theta) [W/m^2].

        Eq. A.9 equatorward of the Arctic circles (i < theta < pi - i):
            F_A = S0 sin(theta) / pi
        Eq. A.15 poleward of them (theta < i, or pi - theta < i):
            F_A = S0 (i^2 + phi^2)^(1/2) / (sqrt(2) pi),  phi = min(theta, pi-theta)
        A.15 matches A.9 at phi = i to O(i^3). At the pole it gives
        S0 i / (sqrt(2) pi), i.e. T_s = 47.7 K for A = 0.5.

        ``polar_flux='legacy-max'`` reproduces the pre-2026-10 behavior,
        F_A = S0/pi max(sin(theta), i), which gives 52.1 K at the pole. That
        is the "~52 K" the paper's text quotes (p. 237), but it is not what
        Eqs. A.15/A.18 give; the project decision (2026-10-04) is A.15.

        Parameters
        ----------
        theta : float or ndarray
            Colatitude in radians [0, pi]. theta=0 is north pole,
            theta=pi/2 is equator, theta=pi is south pole.

        Returns
        -------
        float or ndarray
            Annual-average insolation [W/m^2].
        """
        theta = np.asarray(theta, dtype=float)
        i = self.obliquity
        flux = self.S0 * np.sin(theta) / np.pi                     # Eq. A.9
        if self.polar_flux == 'legacy-max':
            return np.maximum(flux, self.S0 * i / np.pi)
        phi = np.minimum(theta, np.pi - theta)
        polar = self.S0 * np.sqrt(i**2 + phi**2) / (np.sqrt(2.0) * np.pi)  # Eq. A.15
        return np.where(phi < i, polar, flux)

    def temperature(self, theta):
        """Radiative equilibrium surface temperature T_s(theta) [K].

        Eq A.16-A.17: T_s = [(1-A) * F_A / sigma]^(1/4)

        Parameters
        ----------
        theta : float or ndarray
            Colatitude in radians [0, pi].

        Returns
        -------
        float or ndarray
            Surface temperature [K].
        """
        flux = self.annual_average_flux(theta)
        return ((1.0 - self.albedo) * flux / C.STEFAN_BOLTZMANN) ** 0.25

    def temperature_2d(self, theta, lam):
        """Surface temperature on a 2D (theta, lam) grid.

        In the O&S model, T_s depends only on colatitude (no longitude
        dependence for annual-average insolation). Returns T_s(theta)
        broadcast over lam for interface compatibility.

        Parameters
        ----------
        theta : ndarray, shape (n_theta,) or (n_theta, 1)
            Colatitude [rad].
        lam : ndarray, shape (n_lam,) or (1, n_lam)
            Longitude [rad]. Not used but required for interface.

        Returns
        -------
        ndarray, shape (n_theta, n_lam)
            Surface temperature [K].
        """
        theta = np.atleast_1d(theta)
        lam = np.atleast_1d(lam)
        T = self.temperature(theta)
        # Broadcast T(theta) over longitude dimension
        return np.broadcast_to(T[:, np.newaxis], (len(theta), len(lam)))

    def global_mean_temperature(self, n_theta=2000):
        """Area-weighted global mean surface temperature [K] (arithmetic mean)."""
        from .grid import periodic_grid, theta_weights
        theta, _ = periodic_grid(n_theta, 2)
        w = theta_weights(theta)
        return float(np.sum(self.temperature(theta) * w) / np.sum(w))
