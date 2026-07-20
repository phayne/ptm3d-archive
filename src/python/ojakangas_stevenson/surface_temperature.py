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
    """

    def __init__(self, S0=C.SOLAR_FLUX_JUPITER, albedo=C.BOND_ALBEDO,
                 obliquity_deg=3.0):
        self.S0 = S0
        self.albedo = albedo
        self.obliquity = np.radians(obliquity_deg)

    def annual_average_flux(self, theta):
        """Annual-average insolation F_A(theta) [W/m^2].

        Eq A.9 for mid-latitudes: F_A = S0 * sin(theta) / pi
        Eq A.14 near poles: F_A ~ S0 * i / pi

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

        # Mid-latitude formula (Eq A.9)
        flux = self.S0 * np.sin(theta) / np.pi

        # Near poles: use Eq A.14-A.15 interpolation
        # F_A = S0 * i / pi at the pole
        # Smoothly interpolate using max(sin(theta), f_pole(theta))
        # The polar formula from Eq A.14: F_A ~ S0 * i / pi
        # For smooth transition, use: F_A = S0/pi * max(sin(theta), i)
        # This matches A.9 for theta >> i and gives F_A = S0*i/pi at poles
        flux_min = self.S0 * i / np.pi
        flux = np.maximum(flux, flux_min)

        return flux

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

    def global_mean_temperature(self, n_theta=1000):
        """Area-weighted global mean surface temperature [K].

        <T> = integral T(theta) * sin(theta) d(theta) / 2
        """
        theta = np.linspace(0.001, np.pi - 0.001, n_theta)
        T = self.temperature(theta)
        # Area weight: sin(theta) * d(theta), normalized by integral sin(theta) d(theta) = 2
        weights = np.sin(theta)
        dtheta = theta[1] - theta[0]
        return np.sum(T * weights) * dtheta / 2.0
