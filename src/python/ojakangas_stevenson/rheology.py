"""Rheological models for tidal dissipation in Europa's ice shell.

Implements the dissipation integral from Eq 2.6 for two rheologies:
  - Maxwell viscoelastic (Section 3, Eq 3.5)
  - Generalized flow law / Glen's law (Section 4, Eq 4.7)
"""

import numpy as np
from . import constants as C


class MaxwellRheology:
    """Maxwell viscoelastic rheology (Section 3).

    The dissipation integral (Eq 3.5):

        integral_{T_s}^{T_M} q(T) dT/T
            = 2*mu * <edot_ij^2> * [pi/2 - arctan((n*tau_M)_min)] / (n*l)

    Parameters
    ----------
    mu : float
        Ice rigidity [Pa]. Default: 4e9.
    eta0 : float
        Melt viscosity [Pa s]. Default: 1e13.
    l : float
        Viscosity power-law exponent. Default: 24.
    n : float
        Tidal forcing frequency [rad/s]. Default: Europa's mean motion.
    """

    def __init__(self, mu=C.ICE_RIGIDITY, eta0=C.MELT_VISCOSITY,
                 l=C.VISCOSITY_EXPONENT, n=None):
        self.mu = mu
        self.eta0 = eta0
        self.l = l
        self.n = n if n is not None else C.MEAN_MOTION
        self.ntau_min = self.n * self.eta0 / self.mu

    def dissipation_integral(self, strain_rate_sq):
        """Compute the dissipation integral D for given <edot_ij^2>.

        D = 2*mu * <edot_ij^2> * [pi/2 - arctan(ntau_min)] / (n * l)

        Parameters
        ----------
        strain_rate_sq : float or ndarray
            Time-averaged squared strain rate <edot_ij * edot_ij> [s^-2].
            This is n^2 * <e_ij * e_ij> where <e_ij * e_ij> comes from
            tidal_strains.time_averaged_strain_squared().

        Returns
        -------
        float or ndarray
            Dissipation integral D [W/m^2].
            This enters the shell thickness equation b = a1*ln(T_M/T_s) / (D + H).
        """
        prefactor = 2.0 * self.mu / (self.n * self.l)
        arctan_term = np.pi / 2.0 - np.arctan(self.ntau_min)
        return prefactor * strain_rate_sq * arctan_term

    @property
    def name(self):
        return "Maxwell"

    def __repr__(self):
        return (f"MaxwellRheology(mu={self.mu:.2e} Pa, eta0={self.eta0:.2e} Pa s, "
                f"l={self.l}, ntau_min={self.ntau_min:.2f})")


class FlowLawRheology:
    """Generalized power-law (Glen's law) rheology (Section 4).

    The dissipation integral (Eq 4.7):

        D_T = integral q(T) dT/T = 2*n_G / (A0^{1/n_G} * n_A)
                                    * <[0.5 * edot_ij^2]^{(n_G+1)/(2*n_G)}>

    From Eq 4.5: q = 2 [0.5 edot^2]^p / A^{1/n_G}, with the power-law
    viscosity A = A0 (T/T_M)^{n_A} (Eq 4.6). Integrating q dT/T from T_s
    to T_M using the power-law form (valid near T_M) gives n_G/n_A for the
    temperature integral, yielding D_T above.

    Note: For the flow law, the time average must be computed BEFORE
    raising to the power, since <f^p> != <f>^p for p != 1. The
    time_averaged_strain_power() function in tidal_strains handles this.

    Parameters
    ----------
    n_G : float
        Glen's law exponent. Default: 3.
    n_A : float
        Arrhenius exponent. Default: 61.
    A0_cgs : float
        Flow law prefactor in CGS units [s^-1 (dyn/cm^2)^{-n_G}].
    T_melt : float
        Melting temperature [K]. Default: 273.
    """

    def __init__(self, n_G=C.GLEN_EXPONENT_1, n_A=C.ARRHENIUS_EXPONENT,
                 A0_cgs=C.A0_FLOW_1_CGS, T_melt=C.T_MELT):
        self.n_G = n_G
        self.n_A = n_A
        self.A0_cgs = A0_cgs
        self.T_melt = T_melt

        # Convert A0 from CGS to SI
        # 1 dyn/cm^2 = 0.1 Pa, so (dyn/cm^2)^{-n_G} = (0.1 Pa)^{-n_G} = 10^{n_G} Pa^{-n_G}
        self.A0_si = A0_cgs * (10.0 ** n_G)

        # Exponent for the strain power: (n_G + 1) / (2 * n_G)
        self.strain_exponent = (n_G + 1.0) / (2.0 * n_G)

    def dissipation_integral(self, strain_power_avg):
        """Compute the dissipation integral D_T for the flow law.

        D_T = 2 * n_G / (A0_si^{1/n_G} * n_A) * <[0.5 edot^2]^p>

        Parameters
        ----------
        strain_power_avg : float or ndarray
            Time-averaged <[0.5 * edot_ij^2]^{(n_G+1)/(2*n_G)}> [s^{-(n_G+1)/n_G}].

        Returns
        -------
        float or ndarray
            Dissipation integral D_T [W/m^3].
        """
        prefactor = 2.0 * self.n_G / (self.A0_si ** (1.0 / self.n_G) * self.n_A)
        return prefactor * strain_power_avg

    @property
    def name(self):
        return f"FlowLaw(n_G={self.n_G})"

    def __repr__(self):
        return (f"FlowLawRheology(n_G={self.n_G}, n_A={self.n_A}, "
                f"A0_cgs={self.A0_cgs:.2e}, exponent={self.strain_exponent:.3f})")
