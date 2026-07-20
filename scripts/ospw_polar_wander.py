#!/usr/bin/env python3
"""OSPW overdamped pendulum model for polar wander of Europa's ice shell.

Implements the key physics from Ojakangas & Stevenson (1989), Icarus 81, 242-270:
  - Bottom-loading gravity (iceberg effect) from shell thickness variations
  - Willemann compensation parameter for elastic lithosphere
  - Shell viscous dissipation during TPW (unfractured and fractured cases)
  - Overdamped pendulum equation: psi_ddot = (t/tau)(eps*n^2/2) sin 2psi - (Cf/4eps*I0) psi_dot
  - Analytical solution: tan psi = tan(psi0) * exp(t^2 / tau_f^2)

Usage:
    python scripts/ospw_polar_wander.py                # Default run with figures
    python scripts/ospw_polar_wander.py --fig5          # Reproduce Fig 5 (psi vs t)
    python scripts/ospw_polar_wander.py --fig6          # Reproduce Fig 6 (tau_f vs T_f)
    python scripts/ospw_polar_wander.py --instability   # Check instability criterion
"""

import sys
import os
import argparse
from dataclasses import dataclass, field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'python'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ojakangas_stevenson import constants as C

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'mathtext.fontset': 'cm',
    'axes.linewidth': 0.8,
})

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class OSPWConfig:
    """Parameters for OSPW polar wander model.

    All values SI unless noted. References to equation numbers are from
    Ojakangas & Stevenson (1989), Icarus 81, 242-270.
    """
    # Shell geometry
    shell_thickness: float = 20e3       # b_c [m] (average shell thickness)
    lithosphere_thickness: float = 2e3  # d [m] (elastic lithosphere thickness)
    europa_radius: float = C.EUROPA_RADIUS  # a [m]

    # Densities
    ice_density: float = C.ICE_DENSITY      # rho_c [kg/m^3]
    water_density: float = C.WATER_DENSITY  # rho_m [kg/m^3]
    mean_density: float = C.EUROPA_DENSITY  # rho_bar [kg/m^3]

    # Elastic properties
    ice_rigidity: float = C.ICE_RIGIDITY  # mu [Pa]
    poisson_ratio: float = 0.33           # nu

    # Rheology (Arrhenius viscosity: eta = eta0 * exp[l*(Tm/T - 1)])
    melting_viscosity: float = 3e14       # eta_0 [Pa s] (OSPW p.258: 3e15 poise = 3e14 Pa s)
    activation_param: float = 15.0        # l (dimensionless, proportional to activation energy)
    melting_temp: float = C.T_MELT        # T_m [K]
    surface_temp: float = 100.0           # T_s [K]
    fracture_base_temp: float = 130.0     # T_f [K] (temperature at base of fractured region)

    # Tidal/orbital
    love_number_h: float = 1.26           # h (Ross & Schubert 1987)
    mean_motion: float = C.MEAN_MOTION    # n [rad/s]
    europa_mass: float = C.EUROPA_MASS    # M [kg]

    # Shell thickness harmonics (from polar_wander_cycle.py or Paper I Table I)
    # Default values from Paper I, flow law n_G=3 case (most likely to produce TPW)
    t20_km: float = -0.70    # t_20 [km]
    t22_km: float = 0.24     # t_22 [km]

    @property
    def gamma_tidal(self):
        """gamma = n^2 a^3 / (G M) — ratio of tidal to self-gravity potential."""
        return self.mean_motion**2 * self.europa_radius**3 / (C.G_GRAV * self.europa_mass)

    @property
    def delta_rho(self):
        """Density contrast: rho_water - rho_ice."""
        return self.water_density - self.ice_density

    @property
    def t20_plus_2t22(self):
        """Instability diagnostic: t20 + 2*t22 [km]."""
        return self.t20_km + 2 * self.t22_km


# ---------------------------------------------------------------------------
# Core physics functions
# ---------------------------------------------------------------------------

def compute_Cw(cfg):
    """Willemann compensation parameter for degree-2 loading (Eq 2.11).

    For a thin elastic shell (membrane stress only) on a fluid sphere,
    the compensation parameter measures what fraction of bottom-loaded
    topography is isostatically compensated vs elastically supported.

    C_w = K / (1 + K)

    where K = delta_rho * g * a^2 / ((2/3) * mu * d) is the ratio of
    gravitational restoring stress to membrane elastic stress.

    C_w -> 0: rigid lithosphere (full elastic support)
    C_w -> 1: perfectly compensated (no elastic support)

    For Europa (thin ~2 km lithosphere, large radius): C_w ~ 0.98,
    meaning the iceberg effect dominates for bottom loading.
    """
    g = C.EUROPA_GRAVITY
    a = cfg.europa_radius
    mu = cfg.ice_rigidity
    d = cfg.lithosphere_thickness
    delta_rho = cfg.delta_rho  # rho_water - rho_ice

    # Elastic membrane stress parameter for l=2
    sigma_elastic = (2.0 / 3.0) * mu * d / a**2  # Pa

    # Gravitational restoring stress (buoyancy)
    sigma_gravity = delta_rho * g  # Pa/m, but need to match dimensions

    # Ratio K: gravity / elastic (dimensionless via a^2 factor)
    K = delta_rho * g * a**2 / ((2.0 / 3.0) * mu * d)

    Cw = K / (1.0 + K)
    return Cw


def compute_shell_gravity(t2m_km, cfg, Cw=None):
    """Shell thickness C_2m via bottom-loading formula (Eq 2.18).

    C_2m = Q_2m * [(rho_m - rho_c)/rho_c] * [(1-C_w) + 4*C_w*(b_c/a)*(rho_c/delta_rho)]

    where Q_2m = (3/5) * rho_c * t_2m / (rho_bar * a).

    Parameters
    ----------
    t2m_km : float
        Shell thickness harmonic coefficient [km].
    cfg : OSPWConfig
    Cw : float, optional
        Pre-computed Willemann parameter.

    Returns
    -------
    C2m : float
        Dimensionless gravity coefficient.
    """
    if Cw is None:
        Cw = compute_Cw(cfg)

    t2m = t2m_km * 1e3  # convert to meters
    rho_c = cfg.ice_density
    rho_m = cfg.water_density
    rho_bar = cfg.mean_density
    a = cfg.europa_radius
    bc = cfg.shell_thickness
    delta_rho = cfg.delta_rho

    Q2m = (3.0 / 5.0) * rho_c * t2m / (rho_bar * a)

    # Bottom-loading formula (Eq 2.18)
    # For bottom loading, the elastic support and iceberg terms have OPPOSITE signs:
    #   Elastic support: -(1-C_w)  [rigid confinement below crust, reduces inertia]
    #   Iceberg effect:  +4*C_w*(b_c/a)*(rho_c/delta_rho) [surface rise, increases inertia]
    # The full formula is:
    #   C_2m = Q_2m * (delta_rho/rho_c) * [-(1-C_w) + 4*C_w*(b_c/a)*(rho_c/delta_rho)]
    bracket = -(1.0 - Cw) + 4.0 * Cw * (bc / a) * (rho_c / delta_rho)
    C2m = Q2m * (delta_rho / rho_c) * bracket

    return C2m


def compute_fossil_bulge_gravity(cfg, Cw=None):
    """Fossil bulge C_20, C_22 contributions (Eq 3.10).

    The fossil bulge is the lithospheric memory of the hydrostatic shape.

    s_20)_fossil = -(1/3)*h*gamma*(1-C_w)
    s_22)_fossil = +(1/4)*h*gamma*(1-C_w)

    These are surface deformations that contribute to the gravity field.

    Returns
    -------
    C20_fossil, C22_fossil : float
        Fossil bulge gravity coefficients.
    """
    if Cw is None:
        Cw = compute_Cw(cfg)

    h = cfg.love_number_h
    gamma = cfg.gamma_tidal

    # Fossil surface deformation (Eq 3.8)
    s20_fossil = -(1.0 / 3.0) * h * gamma * (1.0 - Cw)
    s22_fossil = (1.0 / 4.0) * h * gamma * (1.0 - Cw)

    # Convert to gravity coefficients via Q_2m = (3/5)*rho_c*t/(rho_bar*a)
    # But fossil is a surface deformation, not thickness change, so:
    # C_2m = (3/5) * s_2m / a  (simplified for surface load)
    # Actually, Eq 3.10 gives these directly in terms of principal moments.
    # The fossil contributes -(1-C_w) fraction of the hydrostatic C_2m.
    # Hydrostatic: C_20^hydro = -(1/3)*h*gamma, C_22^hydro = (1/4)*h*gamma

    C20_fossil = -(1.0 / 3.0) * h * gamma * (1.0 - Cw)
    C22_fossil = (1.0 / 4.0) * h * gamma * (1.0 - Cw)

    return C20_fossil, C22_fossil


def check_instability(cfg, Cw=None):
    """Test I_yy > I_zz criterion (Eq 3.4/3.11/3.12).

    The instability occurs when C_20 + 2*C_22 > 0 for the shell-only
    contribution (thickness + fossil). When this is satisfied, the axis
    of the intermediate moment becomes the axis of the maximum moment,
    triggering 90 deg TPW about the sub-Jovian axis.

    Returns
    -------
    dict with:
        unstable : bool
        t20_2t22 : float — t_20 + 2*t_22 [km]
        threshold_km : float — critical value of (t20+2t22)
        C20_shell, C22_shell : float — gravity coefficients from thickness
        C20_fossil, C22_fossil : float — fossil bulge gravity coefficients
        C20_total, C22_total : float — total
    """
    if Cw is None:
        Cw = compute_Cw(cfg)

    C20_shell = compute_shell_gravity(cfg.t20_km, cfg, Cw)
    C22_shell = compute_shell_gravity(cfg.t22_km, cfg, Cw)
    C20_fossil, C22_fossil = compute_fossil_bulge_gravity(cfg, Cw)

    C20_total = C20_shell + C20_fossil
    C22_total = C22_shell + C22_fossil

    # Instability: I_yy > I_zz means C_20 + 2*C_22 > 0
    # From Eqs 2.13, 2.14: I_yy - I_zz = Ma^2 * (C_20 + 2*C_22)
    sum_criterion = C20_total + 2 * C22_total

    # Threshold from Eq 3.12: (t20+2t22)/bc > 0.6*(4e10 dyn/cm^2)/mu
    # In SI: 4e10 dyn/cm^2 = 4e9 Pa
    threshold_ratio = 0.6 * 4e9 / cfg.ice_rigidity
    threshold_km = threshold_ratio * cfg.shell_thickness / 1e3

    return {
        'unstable': sum_criterion > 0,
        't20_2t22_km': cfg.t20_plus_2t22,
        'threshold_km': threshold_km,
        'C20_shell': C20_shell,
        'C22_shell': C22_shell,
        'C20_fossil': C20_fossil,
        'C22_fossil': C22_fossil,
        'C20_total': C20_total,
        'C22_total': C22_total,
        'criterion_value': sum_criterion,
        'Cw': Cw,
    }


def compute_I0(cfg):
    """Moment of inertia of thin shell + ocean (Eq 5.29).

    I_0 ~ (8/3) * rho * b_0 * a^4

    We use shell+ocean thickness for consistency with OSPW assumption
    that shell and ocean reorient together.
    """
    rho = cfg.ice_density  # approximate
    a = cfg.europa_radius
    b0 = cfg.shell_thickness
    return (8.0 / 3.0) * rho * b0 * a**4


def compute_t_over_tau(cfg, Cw=None):
    """Normalized principal moment difference t/tau (Eq 5.32).

    (t/tau)_equil ~ (10^-5) * [(t20+2t22)/1 km] - 0.6*(4e10 dyn cm^-2)/mu

    Returns the equilibrium value. During an actual TPW event, t/tau grows
    linearly from 0 as the shell approaches thermal equilibrium.
    """
    if Cw is None:
        Cw = compute_Cw(cfg)

    t20_2t22 = cfg.t20_plus_2t22  # [km]

    # Shell contribution to C20+2C22
    C20_s = compute_shell_gravity(cfg.t20_km, cfg, Cw)
    C22_s = compute_shell_gravity(cfg.t22_km, cfg, Cw)
    # Fossil contribution
    C20_f, C22_f = compute_fossil_bulge_gravity(cfg, Cw)

    # (I_yy - I_zz) / I_0 from the gravity coefficients
    I0 = compute_I0(cfg)
    M = cfg.europa_mass
    a = cfg.europa_radius

    # I_yy - I_zz = Ma^2 * (C20 + 2*C22)  [total, shell + fossil]
    t_over_tau = M * a**2 * ((C20_s + C20_f) + 2 * (C22_s + C22_f)) / I0

    return t_over_tau


def compute_epsilon(cfg, Cw=None):
    """Fossil bulge parameter epsilon = (I_zz - I_xx) / I_0 (Eq 5.33-5.34).

    This is the difference between the maximum and minimum principal moments
    due to the fossil bulge + thickness variations.
    """
    if Cw is None:
        Cw = compute_Cw(cfg)

    C20_s = compute_shell_gravity(cfg.t20_km, cfg, Cw)
    C22_s = compute_shell_gravity(cfg.t22_km, cfg, Cw)
    C20_f, C22_f = compute_fossil_bulge_gravity(cfg, Cw)

    I0 = compute_I0(cfg)
    M = cfg.europa_mass
    a = cfg.europa_radius

    # I_zz - I_xx = Ma^2 * (-C20 + 2*C22) from Eqs 2.12, 2.14
    # Wait -- let me re-derive:
    # I_xx = I' + Ma^2(C20/3 - 2C22)
    # I_zz = I' - (2/3)Ma^2 C20
    # I_zz - I_xx = -Ma^2 C20 - Ma^2(C20/3 - 2C22) = Ma^2(-C20 - C20/3 + 2C22)
    # Actually: I_zz - I_xx = I' - (2/3)Ma^2 C20 - I' - Ma^2(C20/3 - 2C22)
    #                       = -(2/3)Ma^2 C20 - Ma^2 C20/3 + 2Ma^2 C22
    #                       = -Ma^2 C20 + 2Ma^2 C22
    #                       = Ma^2(-C20 + 2C22)
    # But epsilon should be positive for the fossil bulge (oblate shape):
    # For fossil: C20 < 0 (oblate), C22 > 0 (triaxial)
    # So -C20 + 2C22 > 0 => epsilon > 0. Good.

    C20_total = C20_s + C20_f
    C22_total = C22_s + C22_f
    epsilon = M * a**2 * (-C20_total + 2 * C22_total) / I0

    return epsilon


def _viscosity(T, eta0, l, Tm):
    """Arrhenius viscosity (Eq 5.2).

    eta(T) = eta_0 * exp[l * (Tm/T - 1)]
    """
    return eta0 * np.exp(l * (Tm / T - 1.0))


def compute_C0(cfg):
    """Dissipation coefficient for unfractured shell (Eq 5.8).

    C_0 ~ (4*pi*a^2*b_c / l) * (T_m/(T_m-T_s)) * eta_max * gamma^2

    where eta_max is the maximum viscosity in the viscous layer (at the
    elastic-viscous boundary). For an unfractured shell, this is enormous.
    """
    a = cfg.europa_radius
    bc = cfg.shell_thickness
    l = cfg.activation_param
    Tm = cfg.melting_temp
    Ts = cfg.surface_temp
    eta0 = cfg.melting_viscosity
    gamma = cfg.gamma_tidal

    # Maximum viscosity: at the elastic/viscous boundary
    # The elastic layer has Maxwell time tau_M > tau_pw, so the boundary
    # is where tau_M ~ tau_pw. For unfractured shell, this is near the surface.
    # OSPW estimates eta_max from the effective viscous layer properties.
    # Here we use eta_max at the surface (upper bound):
    eta_max = _viscosity(Ts, eta0, l, Tm)

    C0 = (4.0 * np.pi * a**2 * bc / l) * (Tm / (Tm - Ts)) * eta_max * gamma**2
    return C0


def compute_Cf(cfg, Tf=None):
    """Viscous dissipation coefficient for fractured shell (Eq 5.15).

    C_f ~ (4*pi*a^2*b_c / l) * (T_m/(T_m-T_s)) * (Tf/Tm)^2
          * eta_0 * exp[l*(Tm/Tf - 1)] * gamma^2

    The viscous layer extends from the base of the fractured region (T_f)
    down to the base of the shell (T_m). The maximum viscosity is at T_f.

    Parameters
    ----------
    cfg : OSPWConfig
    Tf : float, optional
        Temperature at base of fractures [K]. Default: cfg.fracture_base_temp.
    """
    if Tf is None:
        Tf = cfg.fracture_base_temp

    a = cfg.europa_radius
    bc = cfg.shell_thickness
    l = cfg.activation_param
    Tm = cfg.melting_temp
    Ts = cfg.surface_temp
    eta0 = cfg.melting_viscosity
    gamma = cfg.gamma_tidal

    eta_f = _viscosity(Tf, eta0, l, Tm)
    Cf = (4.0 * np.pi * a**2 * bc / l) * (Tm / (Tm - Ts)) * (Tf / Tm)**2 * eta_f * gamma**2

    return Cf


def compute_tau_f(cfg, Tf=None):
    """Overdamped TPW timescale tau_f (from Eq 5.41).

    Derivation from the overdamped equation (Eq 5.39 with psi_ddot ≈ 0):

        psi_dot ≈ [(t/tau) * eps * n^2 / 2 * sin(2psi)] / [C_f / (4*eps*I_0)]
                = [2 * eps^2 * n^2 * I_0 / C_f] * (t/tau) * sin(2psi)

    For small psi, sin(2psi) ≈ 2psi:
        psi_dot ≈ [4 * eps^2 * n^2 * I_0 / C_f] * (t/tau_th) * psi

    Integrating: psi = psi_0 * exp(t^2/tau_f^2) where

        tau_f = sqrt(C_f * tau_th / (2 * eps^2 * n^2 * I_0))

    This matches Eq 5.41: tan(psi) = tan(psi_0) * exp(t^2/tau_f^2).
    OSPW Fig 6 shows tau_f ~ 2 Myr for T_f ~ 125 K.
    """
    if Tf is None:
        Tf = cfg.fracture_base_temp

    Cf = compute_Cf(cfg, Tf)
    I0 = compute_I0(cfg)
    n = cfg.mean_motion
    eps = compute_epsilon(cfg)
    tau_th = 1e7 * 365.25 * 86400  # thermal diffusion time [s]

    if abs(eps) < 1e-20:
        return np.inf, np.inf

    tau_f_s = np.sqrt(Cf * tau_th / (2.0 * eps**2 * n**2 * I0))
    tau_f_yr = tau_f_s / (365.25 * 86400)

    return tau_f_yr, tau_f_s


def compute_undamped_timescale(cfg, Cw=None):
    """Undamped TPW timescale T (Eq 4.22).

    T ~ (tau / (n * epsilon))^(1/3)

    This is a lower bound on the actual TPW timescale.
    """
    if Cw is None:
        Cw = compute_Cw(cfg)
    eps = compute_epsilon(cfg, Cw)
    n = cfg.mean_motion
    tau_th = 1e7 * 365.25 * 86400  # [s]

    if abs(eps) < 1e-20:
        return np.inf

    T_s = (tau_th / (n * abs(eps)))**(1.0 / 3.0)
    T_yr = T_s / (365.25 * 86400)
    return T_yr


def compute_Maxwell_time(T, cfg):
    """Maxwell relaxation time at temperature T.

    tau_M = eta(T) / mu
    """
    eta = _viscosity(T, cfg.melting_viscosity, cfg.activation_param, cfg.melting_temp)
    return eta / cfg.ice_rigidity


# ---------------------------------------------------------------------------
# Evolution models
# ---------------------------------------------------------------------------

class OSPWEvolution:
    """Overdamped pendulum model for shell TPW (Eqs 5.39-5.41).

    State: psi (tip angle about sub-Jovian axis)
    Physics: psi_ddot = (t/tau)(eps*n^2/2) sin(2psi) - (Cf/(4*eps*I0)) psi_dot

    In the overdamped limit, psi_ddot is negligible and the solution is:
        tan(psi) = tan(psi_0) * exp(t^2 / tau_f^2)

    We provide both:
    - analytical_solution(): the closed-form solution from Eq 5.41
    - run(): numerical integration of the full Eq 5.39
    """

    def __init__(self, cfg, t_over_tau_rate=None):
        """
        Parameters
        ----------
        cfg : OSPWConfig
        t_over_tau_rate : float, optional
            Rate at which (t/tau) grows [1/s]. Default: 1/tau_th.
        """
        self.cfg = cfg
        self.Cw = compute_Cw(cfg)
        self.I0 = compute_I0(cfg)
        self.eps = compute_epsilon(cfg, self.Cw)
        self.Cf = compute_Cf(cfg)
        self.n = cfg.mean_motion
        self.tau_th = 1e7 * 365.25 * 86400  # [s]

        if t_over_tau_rate is None:
            # t/tau grows from 0 to (t/tau)_equil over tau_th
            self.t_rate = 1.0 / self.tau_th  # normalized rate
        else:
            self.t_rate = t_over_tau_rate

        # Damping coefficient: C_f / (4*eps*I_0) from Eq 5.39
        if abs(self.eps) > 1e-20:
            self.damping = self.Cf / (4.0 * abs(self.eps) * self.I0)
        else:
            self.damping = 1e30

    def driving_torque(self, t_since_instability, psi):
        """Driving torque per unit angular acceleration: (t/tau)(eps*n^2/2) sin(2psi).

        t_since_instability is time since t/tau passed through zero [s].
        The t/tau factor grows linearly: t/tau = t_rate * t.
        """
        t_over_tau = self.t_rate * t_since_instability
        return 0.5 * t_over_tau * self.eps * self.n**2 * np.sin(2.0 * psi)

    def analytical_solution(self, t_arr, psi0):
        """Analytical overdamped solution (Eq 5.41).

        tan(psi) = tan(psi_0) * exp(t^2 / tau_f^2)

        Parameters
        ----------
        t_arr : array
            Time since instability onset [years].
        psi0 : float
            Initial tip angle [radians].

        Returns
        -------
        psi : array
            Tip angle [radians].
        """
        tau_f_yr, _ = compute_tau_f(self.cfg)
        t_ratio = (t_arr / tau_f_yr)**2
        tan_psi = np.tan(psi0) * np.exp(t_ratio)
        psi = np.arctan(tan_psi)
        # Ensure psi increases toward pi/2 (not wrapping around)
        psi = np.where(psi < 0, psi + np.pi, psi)
        return psi

    def run(self, t_max_yr, psi0_deg=1.0, dt_yr=None):
        """Numerically integrate the overdamped pendulum (Eq 5.39).

        Uses simple forward Euler for the first-order overdamped equation:
            psi_dot = [(t_rate*t) * eps * n^2 / 2 * sin(2psi)] / damping

        Parameters
        ----------
        t_max_yr : float
            Integration time [years].
        psi0_deg : float
            Initial tip angle [degrees].
        dt_yr : float, optional
            Time step [years]. Default: tau_f / 100.

        Returns
        -------
        dict with 't_yr', 'psi_deg', 'psi_dot' arrays.
        """
        tau_f_yr, _ = compute_tau_f(self.cfg)
        if dt_yr is None:
            dt_yr = max(tau_f_yr / 100, 100)  # at least 100 yr steps
        dt = dt_yr * 365.25 * 86400  # [s]

        psi = np.radians(psi0_deg)
        t = 0.0
        n_steps = int(t_max_yr / dt_yr) + 1

        t_arr = np.zeros(n_steps)
        psi_arr = np.zeros(n_steps)
        psi_dot_arr = np.zeros(n_steps)

        for i in range(n_steps):
            t_arr[i] = t / (365.25 * 86400)  # years
            psi_arr[i] = np.degrees(psi)

            # Overdamped: psi_dot = driving / damping
            driving = self.driving_torque(t, psi)
            psi_dot = driving / self.damping if self.damping > 0 else 0.0
            psi_dot_arr[i] = psi_dot

            psi += psi_dot * dt
            t += dt

            # Clamp to [0, pi/2]
            psi = np.clip(psi, 0, np.pi / 2)

        return {
            't_yr': t_arr,
            'psi_deg': psi_arr,
            'psi_dot': psi_dot_arr,
            'tau_f_yr': tau_f_yr,
        }

    def run_full(self, t_max_yr, psi0_deg=1.0, n_points=2000):
        """Numerically integrate the FULL pendulum equation (Eq 5.39).

        Including the inertial term: psi_ddot = driving - damping*psi_dot

        Uses scipy.integrate.solve_ivp for robust integration.
        """
        from scipy.integrate import solve_ivp

        tau_f_yr, _ = compute_tau_f(self.cfg)
        t_max_s = t_max_yr * 365.25 * 86400

        def rhs(t, y):
            psi, psi_dot = y
            # Clamp psi to prevent divergence
            psi_c = np.clip(psi, 1e-15, np.pi / 2 - 1e-10)
            driving = self.driving_torque(t, psi_c)
            psi_ddot = driving - self.damping * psi_dot
            return [psi_dot, psi_ddot]

        t_eval = np.linspace(0, t_max_s, n_points)
        sol = solve_ivp(rhs, [0, t_max_s],
                        [np.radians(psi0_deg), 0.0],
                        t_eval=t_eval, method='Radau',
                        rtol=1e-8, atol=1e-12)

        return {
            't_yr': sol.t / (365.25 * 86400),
            'psi_deg': np.degrees(np.clip(sol.y[0], 0, np.pi / 2)),
            'psi_dot': sol.y[1],
            'tau_f_yr': tau_f_yr,
        }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_fig5(cfg, save_path=None):
    """Reproduce OSPW Fig 5: psi(t) for various initial angles.

    Shows the analytical overdamped solution for several psi_0 values.
    """
    tau_f_yr, _ = compute_tau_f(cfg)
    t_max = 5 * tau_f_yr  # enough to see full evolution

    t = np.linspace(0, t_max, 2000)

    fig, ax = plt.subplots(figsize=(6, 4))

    psi0_values = [0.1, 1.0, 5.0, 10.0, 20.0, 45.0]
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(psi0_values)))

    evol = OSPWEvolution(cfg)

    for psi0, color in zip(psi0_values, colors):
        psi = evol.analytical_solution(t, np.radians(psi0))
        ax.plot(t / 1e6, np.degrees(psi), color=color,
                label=f'$\\psi_0$ = {psi0}$^\\circ$')

    ax.set_xlabel('Time [Myr]')
    ax.set_ylabel('$\\psi$ [degrees]')
    ax.set_title(f'OSPW Overdamped TPW ($\\tau_f$ = {tau_f_yr/1e6:.2f} Myr)')
    ax.set_ylim(0, 95)
    ax.axhline(90, ls=':', color='gray', alpha=0.5)
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f'Saved: {save_path}')
    return fig


def plot_fig6(cfg, save_path=None):
    """Reproduce OSPW Fig 6: tau_f and tau_M vs T_f.

    Shows the TPW timescale and Maxwell time at the base of the
    fractured region as functions of the fracture-base temperature.
    """
    Tf_range = np.linspace(105, 155, 200)
    tau_f_arr = np.zeros_like(Tf_range)
    tau_M_arr = np.zeros_like(Tf_range)

    for i, Tf in enumerate(Tf_range):
        tau_f_yr, _ = compute_tau_f(cfg, Tf=Tf)
        tau_f_arr[i] = tau_f_yr
        tau_M_arr[i] = compute_Maxwell_time(Tf, cfg) / (365.25 * 86400)  # to years

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.semilogy(Tf_range, tau_f_arr, 'b-', linewidth=2, label='$\\tau_f$ (TPW timescale)')
    ax.semilogy(Tf_range, tau_M_arr, 'r--', linewidth=1.5, label='$\\tau_{Mf}$ (Maxwell time at $T_f$)')
    ax.axhline(2e6, ls=':', color='gray', alpha=0.5, label='$\\sim$2 Myr limit')
    ax.axhspan(1e6, 1e7, alpha=0.1, color='green', label='Feasible TPW window')

    ax.set_xlabel('$T_f$ [K]')
    ax.set_ylabel('Timescale [years]')
    ax.set_title('OSPW Fig 6: TPW Timescale vs Fracture-Base Temperature')
    ax.set_ylim(1e2, 1e10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f'Saved: {save_path}')
    return fig


def plot_instability_check(cfg, save_path=None):
    """Diagnostic plot showing instability analysis."""
    result = check_instability(cfg)
    Cw = result['Cw']

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel 1: Gravity coefficients
    ax = axes[0]
    labels = ['$C_{20}^{shell}$', '$C_{22}^{shell}$',
              '$C_{20}^{fossil}$', '$C_{22}^{fossil}$',
              '$C_{20}^{total}$', '$C_{22}^{total}$']
    values = [result['C20_shell'], result['C22_shell'],
              result['C20_fossil'], result['C22_fossil'],
              result['C20_total'], result['C22_total']]
    colors_list = ['C0', 'C0', 'C1', 'C1', 'C2', 'C2']
    hatches = ['/', '\\', '/', '\\', '/', '\\']

    bars = ax.bar(range(len(labels)), values, color=colors_list, alpha=0.7)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Gravity coefficient')
    ax.set_title(f'$C_w$ = {Cw:.4f}')
    ax.axhline(0, color='k', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 2: Instability criterion
    ax = axes[1]
    ax.text(0.5, 0.7, f'$t_{{20}} + 2t_{{22}}$ = {result["t20_2t22_km"]:.3f} km',
            transform=ax.transAxes, fontsize=14, ha='center')
    ax.text(0.5, 0.5, f'Threshold = {result["threshold_km"]:.3f} km',
            transform=ax.transAxes, fontsize=14, ha='center')
    ax.text(0.5, 0.3, f'$C_{{20}} + 2C_{{22}}$ = {result["criterion_value"]:.3e}',
            transform=ax.transAxes, fontsize=14, ha='center')
    status = 'UNSTABLE (TPW possible)' if result['unstable'] else 'STABLE (no TPW)'
    color = 'red' if result['unstable'] else 'green'
    ax.text(0.5, 0.1, status, transform=ax.transAxes, fontsize=16,
            ha='center', fontweight='bold', color=color)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Instability Criterion (Eq 3.12)')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f'Saved: {save_path}')
    return fig


def plot_numerical_vs_analytical(cfg, save_path=None):
    """Compare numerical integration to analytical overdamped solution."""
    evol = OSPWEvolution(cfg)
    tau_f_yr, _ = compute_tau_f(cfg)
    t_max = 4 * tau_f_yr

    psi0_deg = 5.0
    result_num = evol.run(t_max, psi0_deg=psi0_deg)
    result_full = evol.run_full(t_max, psi0_deg=psi0_deg)

    t_an = np.linspace(0, t_max, 2000)
    psi_an = np.degrees(evol.analytical_solution(t_an, np.radians(psi0_deg)))

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax = axes[0]
    ax.plot(t_an / 1e6, psi_an, 'k-', label='Analytical (Eq 5.41)', linewidth=2)
    ax.plot(result_num['t_yr'] / 1e6, result_num['psi_deg'], 'r--',
            label='Numerical (overdamped)', alpha=0.8)
    ax.plot(result_full['t_yr'] / 1e6, result_full['psi_deg'], 'b:',
            label='Numerical (full Eq 5.39)', alpha=0.8)
    ax.set_ylabel('$\\psi$ [degrees]')
    ax.set_ylim(0, 95)
    ax.axhline(90, ls=':', color='gray', alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title(f'$\\psi_0$ = {psi0_deg}$^\\circ$, $\\tau_f$ = {tau_f_yr/1e6:.2f} Myr')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(result_num['t_yr'] / 1e6,
            np.abs(result_num['psi_deg'] - np.interp(result_num['t_yr'], t_an, psi_an)),
            'r-', label='|Overdamped - Analytical|')
    ax.plot(result_full['t_yr'] / 1e6,
            np.abs(result_full['psi_deg'] - np.interp(result_full['t_yr'], t_an, psi_an)),
            'b-', label='|Full - Analytical|')
    ax.set_xlabel('Time [Myr]')
    ax.set_ylabel('$|\\Delta\\psi|$ [degrees]')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f'Saved: {save_path}')
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def print_diagnostics(cfg):
    """Print all computed OSPW quantities."""
    Cw = compute_Cw(cfg)
    result = check_instability(cfg, Cw)
    t_over_tau = compute_t_over_tau(cfg, Cw)
    eps = compute_epsilon(cfg, Cw)
    I0 = compute_I0(cfg)
    C0 = compute_C0(cfg)
    Cf = compute_Cf(cfg)
    tau_f_yr, _ = compute_tau_f(cfg)
    T_undamped = compute_undamped_timescale(cfg, Cw)
    tau_Mf = compute_Maxwell_time(cfg.fracture_base_temp, cfg)

    print('=' * 60)
    print('OSPW Polar Wander Diagnostics')
    print('=' * 60)
    print(f'\nShell parameters:')
    print(f'  b_c = {cfg.shell_thickness/1e3:.0f} km')
    print(f'  d (lithosphere) = {cfg.lithosphere_thickness/1e3:.1f} km')
    print(f'  rho_ice = {cfg.ice_density:.0f} kg/m^3')
    print(f'  rho_water = {cfg.water_density:.0f} kg/m^3')
    print(f'  mu = {cfg.ice_rigidity:.1e} Pa')
    print(f'  T_s = {cfg.surface_temp:.0f} K')
    print(f'  T_f = {cfg.fracture_base_temp:.0f} K')
    print(f'  eta_0 = {cfg.melting_viscosity:.1e} Pa s')
    print(f'  l = {cfg.activation_param:.0f}')
    print(f'\nThickness harmonics:')
    print(f'  t20 = {cfg.t20_km:.3f} km')
    print(f'  t22 = {cfg.t22_km:.3f} km')
    print(f'  t20 + 2*t22 = {cfg.t20_plus_2t22:.3f} km')
    print(f'\nComputed quantities:')
    print(f'  C_w = {Cw:.6f}')
    print(f'  gamma = {cfg.gamma_tidal:.6e}')
    print(f'  I_0 = {I0:.3e} kg m^2')
    print(f'  t/tau (equil) = {t_over_tau:.3e}')
    print(f'  epsilon = {eps:.3e}')
    print(f'\nGravity coefficients:')
    print(f'  C20_shell = {result["C20_shell"]:.3e}')
    print(f'  C22_shell = {result["C22_shell"]:.3e}')
    print(f'  C20_fossil = {result["C20_fossil"]:.3e}')
    print(f'  C22_fossil = {result["C22_fossil"]:.3e}')
    print(f'  C20_total = {result["C20_total"]:.3e}')
    print(f'  C22_total = {result["C22_total"]:.3e}')
    print(f'  C20+2C22 (criterion) = {result["criterion_value"]:.3e}')
    print(f'\nInstability: {"UNSTABLE" if result["unstable"] else "STABLE"}')
    print(f'\nDissipation coefficients:')
    print(f'  C_0 (unfractured) = {C0:.3e} kg m^2 s^-1')
    print(f'  C_f (fractured, Tf={cfg.fracture_base_temp:.0f}K) = {Cf:.3e} kg m^2 s^-1')
    print(f'  tau_Mf (Maxwell at Tf) = {tau_Mf/(365.25*86400):.1e} yr')
    print(f'\nTimescales:')
    print(f'  tau_f (overdamped TPW) = {tau_f_yr:.3e} yr = {tau_f_yr/1e6:.2f} Myr')
    print(f'  T (undamped TPW) = {T_undamped:.3e} yr')
    print(f'  Ratio tau_f / T_undamped = {tau_f_yr / T_undamped:.0f}')
    print(f'\nOur model\'s tau_TPW = 1e5 yr for comparison')
    print(f'  OSPW tau_f / our tau_TPW = {tau_f_yr / 1e5:.0f}x')
    print('=' * 60)


def main():
    parser = argparse.ArgumentParser(description='OSPW polar wander model')
    parser.add_argument('--fig5', action='store_true', help='Plot Fig 5 (psi vs t)')
    parser.add_argument('--fig6', action='store_true', help='Plot Fig 6 (tau_f vs T_f)')
    parser.add_argument('--instability', action='store_true', help='Check instability')
    parser.add_argument('--compare', action='store_true', help='Compare numerical vs analytical')
    parser.add_argument('--all', action='store_true', help='Run all analyses')
    parser.add_argument('--Tf', type=float, default=130.0, help='Fracture base temp [K]')
    parser.add_argument('--bc', type=float, default=20.0, help='Shell thickness [km]')
    parser.add_argument('--d-lith', type=float, default=2.0, help='Lithosphere thickness [km]')
    parser.add_argument('--t20', type=float, default=-0.70, help='t20 [km]')
    parser.add_argument('--t22', type=float, default=0.24, help='t22 [km]')
    parser.add_argument('--eta0', type=float, default=3e14, help='Melting viscosity [Pa s]')
    parser.add_argument('--outdir', type=str, default='docs/figures/ospw',
                        help='Output directory for figures')
    args = parser.parse_args()

    cfg = OSPWConfig(
        shell_thickness=args.bc * 1e3,
        lithosphere_thickness=args.d_lith * 1e3,
        fracture_base_temp=args.Tf,
        t20_km=args.t20,
        t22_km=args.t22,
        melting_viscosity=args.eta0,
    )

    os.makedirs(args.outdir, exist_ok=True)
    print_diagnostics(cfg)

    do_all = args.all or not (args.fig5 or args.fig6 or args.instability or args.compare)

    if args.fig5 or do_all:
        plot_fig5(cfg, save_path=os.path.join(args.outdir, 'ospw_fig5_psi_vs_t.png'))

    if args.fig6 or do_all:
        plot_fig6(cfg, save_path=os.path.join(args.outdir, 'ospw_fig6_tau_f_vs_Tf.png'))

    if args.instability or do_all:
        plot_instability_check(cfg, save_path=os.path.join(args.outdir, 'ospw_instability.png'))

    if args.compare or do_all:
        plot_numerical_vs_analytical(cfg, save_path=os.path.join(args.outdir, 'ospw_num_vs_analytical.png'))

    plt.close('all')
    print('\nDone.')


if __name__ == '__main__':
    main()
