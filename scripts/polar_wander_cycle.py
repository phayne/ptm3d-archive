#!/usr/bin/env python3
"""Polar wander cycle simulation for Europa's ice shell.

Tests the hypothesis: trailing-hemisphere darkening -> shell thickness
change -> MoI instability -> polar wander -> cycle restarts.

Usage:
    python scripts/polar_wander_cycle.py --sweep-only
    python scripts/polar_wander_cycle.py
    python scripts/polar_wander_cycle.py --tau-dark 1e4 --tau-shell 1e7
    python scripts/polar_wander_cycle.py --survey
    python scripts/polar_wander_cycle.py --all-rheologies
"""

import sys
import os
import argparse
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'python'))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.gridspec import GridSpec

from ojakangas_stevenson import constants as C
from ojakangas_stevenson.surface_temperature import OSSurfaceTemperature
from ojakangas_stevenson.rheology import MaxwellRheology, FlowLawRheology
from ojakangas_stevenson.tidal_strains import (
    time_averaged_strain_squared, time_averaged_strain_power)

from ospw_polar_wander import (
    OSPWConfig, compute_Cw, compute_Cf, compute_I0, compute_tau_f,
)

# ---------------------------------------------------------------------------
# Style (matches plot_ice_shell.py)
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

THICKNESS_CMAP = 'gist_earth'

CASE_STYLES = {
    'maxwell_h0':  ('C0', 'o', '-',  'Maxwell $H$=0'),
    'maxwell_h10': ('C1', 's', '--', 'Maxwell $H$=10'),
    'flow_ng3':    ('C2', '^', '-.',  'Flow $n_G$=3'),
    'flow_ng52':   ('C3', 'D', ':',  'Flow $n_G$=5.2'),
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class PolarWanderConfig:
    """Parameters for the polar wander cycle simulation."""
    A_bright: float = 0.55
    A_dark: float = 0.15
    darkening_power: float = 1.0
    tau_dark: float = 1e4
    tau_shell: float = 1e7
    reorient_threshold: float = 5e-9
    reorient_albedo_reset: float = 1.0
    rheology_case: str = 'maxwell_h0'
    n_theta: int = 181
    n_lam: int = 361
    dt_years: float = 1e3
    t_max_years: float = 1e8
    n_cycles_max: int = 10
    n_darkening_steps: int = 50
    use_fossil_bulge: bool = True
    fossil_bulge_factor: float = 1.5
    tau_visc: float = 3e7
    use_full_inertia: bool = True
    misalignment_threshold_deg: float = 5.0
    tau_tpw: float = 1e5              # TPW relaxation timescale [years]
    # --- TPW dynamics model ---
    tpw_model: str = 'exponential'    # 'exponential', 'ospw', or 'coupled'
    # --- OSPW rheological parameters (ospw/coupled modes) ---
    ospw_shell_thickness: float = 20e3       # b_c [m]
    ospw_lithosphere_thickness: float = 2e3  # d [m]
    ospw_melting_viscosity: float = 3e14     # eta_0 [Pa s]
    ospw_activation_param: float = 15.0      # l (dimensionless)
    ospw_fracture_base_temp: float = 130.0   # T_f [K]
    ospw_surface_temp: float = 100.0         # T_s [K]
    # --- Iceberg amplification ---
    use_iceberg_effect: bool = True          # OSPW bottom-loading amplification
    iceberg_amplification: float = 0.0       # 0 = auto-compute; >0 = override
    # --- B&H interior coupling (coupled mode) ---
    interior_gamma: float = 28.0             # C_i / C_s
    interior_triaxiality: float = 1e-5       # (B_i-A_i)/C_i for rocky interior
    interior_K_G_factor: float = 0.5         # K_G / K_G* (<1 weak, >1 strong)


# ---------------------------------------------------------------------------
# Physics helpers
# ---------------------------------------------------------------------------

def _build_dissipation_grid(case, theta, lam):
    """Pre-compute dissipation integral D(θ,λ) for the chosen rheology."""
    THETA, LAM = np.meshgrid(theta, lam, indexing='ij')
    g2, g3 = C.g2(), C.g3()

    if case.startswith('maxwell'):
        maxwell = MaxwellRheology()
        eij2 = time_averaged_strain_squared(THETA, LAM, g2, g3)
        D = maxwell.dissipation_integral(C.MEAN_MOTION**2 * eij2)
    elif case == 'flow_ng3':
        flow = FlowLawRheology(n_G=C.GLEN_EXPONENT_1, n_A=C.ARRHENIUS_EXPONENT,
                                A0_cgs=C.A0_FLOW_1_CGS)
        exp = flow.strain_exponent
        sp = time_averaged_strain_power(THETA, LAM, exp, g2, g3, n_samples=360)
        D = flow.dissipation_integral(C.MEAN_MOTION**(2*exp) * sp)
    elif case == 'flow_ng52':
        flow = FlowLawRheology(n_G=C.GLEN_EXPONENT_2, n_A=C.ARRHENIUS_EXPONENT,
                                A0_cgs=C.A0_FLOW_2_CGS)
        exp = flow.strain_exponent
        sp = time_averaged_strain_power(THETA, LAM, exp, g2, g3, n_samples=360)
        D = flow.dissipation_integral(C.MEAN_MOTION**(2*exp) * sp)
    else:
        raise ValueError(f'Unknown rheology: {case}')

    H = C.H_FLUX_10 if case == 'maxwell_h10' else 0.0
    return D, H


def _equilibrium_thickness(T_s, D, H):
    """O&S Eq 2.8 equilibrium shell thickness [m]."""
    a1, T_M = C.A1_CONDUCTIVITY, C.T_MELT
    ln_r = np.log(T_M / T_s)
    A = D / ln_r
    disc = H**2 + 4 * A * a1 * ln_r
    return np.clip((-H + np.sqrt(disc)) / (2 * A), 0, 200e3)


def _compute_moments(b, theta, lam):
    """Compute MoI quantities from thickness field b(θ,λ) [m]."""
    dt = theta[1] - theta[0]
    dl = lam[1] - lam[0]
    ct, st = np.cos(theta), np.sin(theta)
    P20 = 0.5 * (3 * ct**2 - 1)
    P22 = 3 * st**2
    c2l = np.cos(2 * lam)

    t20 = 5 / (4 * np.pi) * np.sum(b * P20[:, None] * st[:, None]) * dt * dl
    t22 = 5 / (48 * np.pi) * np.sum(b * P22[:, None] * c2l[None, :] * st[:, None]) * dt * dl

    t20_km = t20 / 1000
    t22_km = t22 / 1000
    return {
        't20': t20,
        't22': t22,
        't20_km': t20_km,
        't22_km': t22_km,
        '4t22_km': 4 * t22_km,
        't20_2t22_km': t20_km + 2 * t22_km,
        'delta_BA_C': 1e-7 * 4 * t22_km,
        'delta_BC_A': 1e-7 * (t20_km + 2 * t22_km),
    }


def _area_mean(field, theta):
    """Area-weighted mean of a field on (θ, λ) grid."""
    dt = theta[1] - theta[0]
    dl = 2 * np.pi / (field.shape[1] - 1)
    return np.sum(field * np.sin(theta)[:, None]) * dt * dl / (4 * np.pi)


def _compute_inertia_tensor(b, theta, lam):
    """Full 3x3 inertia tensor of thin shell b(theta,lam) at radius R.

    Returns I / (rho * R^4) so eigenvalue differences are in units of [m·sr].
    The isotropic part is ~(8pi/3)*<b>; anomalies from thickness variations
    are typically 1e-3 to 1e-2 of this.

    Uses trapezoidal weights for the periodic longitude direction (lambda=-pi
    and lambda=+pi are the same point) to eliminate quadrature anisotropy.
    """
    dt = theta[1] - theta[0]
    dl = lam[1] - lam[0]
    THETA, LAM = np.meshgrid(theta, lam, indexing='ij')
    st, ct = np.sin(THETA), np.cos(THETA)
    sl, cl = np.sin(LAM), np.cos(LAM)
    x, y, z = st * cl, st * sl, ct
    # Trapezoidal weights: half-weight at duplicate longitude endpoints
    lam_w = np.ones(len(lam))
    lam_w[0] = lam_w[-1] = 0.5
    w = b * st * dt * dl * lam_w[None, :]

    I = np.zeros((3, 3))
    I[0, 0] = np.sum((1 - x**2) * w)
    I[1, 1] = np.sum((1 - y**2) * w)
    I[2, 2] = np.sum((1 - z**2) * w)
    I[0, 1] = I[1, 0] = -np.sum(x * y * w)
    I[0, 2] = I[2, 0] = -np.sum(x * z * w)
    I[1, 2] = I[2, 1] = -np.sum(y * z * w)
    return I


def _find_principal_axes(I_tensor, spin_pole):
    """Eigendecompose inertia tensor; find misalignment with spin pole.

    Returns dict: eigenvalues (ascending), v_max, misalignment_deg.
    """
    vals, vecs = np.linalg.eigh(I_tensor)
    v_max = vecs[:, -1]
    if np.dot(v_max, spin_pole) < 0:
        v_max = -v_max
    cos_angle = np.clip(np.dot(v_max, spin_pole), -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(cos_angle))
    return {
        'eigenvalues': vals,
        'v_max': v_max,
        'misalignment_deg': angle_deg,
        'delta_I_max': vals[2] - vals[1],
        'delta_I_all': vals[2] - vals[0],
    }


def _compute_ospw_epsilon_correction(cfg):
    """OSPW correction factor for converting geometric inertia anomaly to
    physical epsilon.

    Our geometric inertia tensor treats the shell as uncompensated ice at
    radius R.  OSPW Eq 2.18 shows the correct inertia anomaly for a
    bottom-loaded, isostatically compensated shell is smaller by factor:

        correction = (delta_rho / rho_ice) * bracket

    where bracket = -(1-C_w) + 4*C_w*(b_c/a)*(rho_c/delta_rho).

    For Europa: correction ~ 0.048.  The geometric tensor gives the correct
    principal axis directions (eigenvectors are scale-invariant), but the
    driving torque for TPW must use the corrected epsilon.
    """
    ospw_cfg = OSPWConfig(
        shell_thickness=cfg.ospw_shell_thickness,
        lithosphere_thickness=cfg.ospw_lithosphere_thickness,
    )
    Cw = compute_Cw(ospw_cfg)
    rho_c = C.ICE_DENSITY
    delta_rho = C.WATER_DENSITY - C.ICE_DENSITY
    a = C.EUROPA_RADIUS
    bc = cfg.ospw_shell_thickness
    bracket = -(1.0 - Cw) + 4.0 * Cw * (bc / a) * (rho_c / delta_rho)
    return (delta_rho / rho_c) * bracket


def _solar_flux_for_spin_pole(theta, lam, spin_pole, flux_func):
    """Solar flux on body-frame grid for arbitrary spin pole direction.

    Returns 2D array of flux [W/m^2] with shape (n_theta, n_lam).
    """
    THETA, LAM = np.meshgrid(theta, lam, indexing='ij')
    x = np.sin(THETA) * np.cos(LAM)
    y = np.sin(THETA) * np.sin(LAM)
    z = np.cos(THETA)
    cos_spin = x * spin_pole[0] + y * spin_pole[1] + z * spin_pole[2]
    theta_spin = np.arccos(np.clip(cos_spin, -1.0, 1.0))
    return flux_func(theta_spin)


def _tidal_coordinates(theta, lam, spin_pole):
    """Convert body-frame (theta, lam) grid to tidal/spin-frame coordinates.

    The O&S tidal strain formulas use theta = colatitude from the spin axis
    and lam = longitude from the sub-Jovian meridian, both measured in the
    spin frame.  When the spin pole is not aligned with the body-frame z-axis
    (i.e. during TPW), the tidal coordinates differ from body-frame coordinates.

    Returns (theta_tidal, lam_tidal) 2D arrays on the body-frame grid.
    """
    THETA, LAM = np.meshgrid(theta, lam, indexing='ij')
    x = np.sin(THETA) * np.cos(LAM)
    y = np.sin(THETA) * np.sin(LAM)
    z = np.cos(THETA)
    s = np.asarray(spin_pole, dtype=float)

    # Tidal colatitude: angle from spin pole
    cos_theta_t = x * s[0] + y * s[1] + z * s[2]
    theta_t = np.arccos(np.clip(cos_theta_t, -1.0, 1.0))

    # Sub-Jovian direction projected onto equatorial plane of spin pole
    sub_j = np.array([1.0, 0.0, 0.0])
    e_sub = sub_j - np.dot(sub_j, s) * s
    norm = np.linalg.norm(e_sub)
    if norm < 1e-10:
        # Degenerate: spin pole nearly along sub-Jovian direction
        e_sub = np.array([0.0, 1.0, 0.0]) - np.dot([0.0, 1.0, 0.0], s) * s
        norm = np.linalg.norm(e_sub)
    e_sub /= norm
    e_trail = np.cross(s, e_sub)

    # Project each grid point onto equatorial plane of spin pole
    r_perp_x = x - cos_theta_t * s[0]
    r_perp_y = y - cos_theta_t * s[1]
    r_perp_z = z - cos_theta_t * s[2]

    cos_lam_t = (r_perp_x * e_sub[0] + r_perp_y * e_sub[1]
                 + r_perp_z * e_sub[2])
    sin_lam_t = (r_perp_x * e_trail[0] + r_perp_y * e_trail[1]
                 + r_perp_z * e_trail[2])
    lam_t = np.arctan2(sin_lam_t, cos_lam_t)

    return theta_t, lam_t


def _build_dissipation_for_spin_pole(case, theta, lam, spin_pole):
    """Compute D(theta, lam) in body frame for given spin pole direction.

    Maps body-frame grid to tidal coordinates, evaluates tidal strain
    there, then returns D on the body-frame grid.
    """
    theta_t, lam_t = _tidal_coordinates(theta, lam, spin_pole)
    g2, g3 = C.g2(), C.g3()

    if case.startswith('maxwell'):
        maxwell = MaxwellRheology()
        eij2 = time_averaged_strain_squared(theta_t, lam_t, g2, g3)
        D = maxwell.dissipation_integral(C.MEAN_MOTION**2 * eij2)
    elif case == 'flow_ng3':
        flow = FlowLawRheology(n_G=C.GLEN_EXPONENT_1, n_A=C.ARRHENIUS_EXPONENT,
                                A0_cgs=C.A0_FLOW_1_CGS)
        exp = flow.strain_exponent
        sp = time_averaged_strain_power(theta_t, lam_t, exp, g2, g3,
                                         n_samples=360)
        D = flow.dissipation_integral(C.MEAN_MOTION**(2*exp) * sp)
    elif case == 'flow_ng52':
        flow = FlowLawRheology(n_G=C.GLEN_EXPONENT_2, n_A=C.ARRHENIUS_EXPONENT,
                                A0_cgs=C.A0_FLOW_2_CGS)
        exp = flow.strain_exponent
        sp = time_averaged_strain_power(theta_t, lam_t, exp, g2, g3,
                                         n_samples=360)
        D = flow.dissipation_integral(C.MEAN_MOTION**(2*exp) * sp)
    else:
        raise ValueError(f'Unknown rheology: {case}')

    H = C.H_FLUX_10 if case == 'maxwell_h10' else 0.0
    return D, H


def _darkening_weight_for_spin_pole(theta, lam, spin_pole, power=1.0):
    """Trailing hemisphere darkening weight for given spin pole direction.

    Returns 2D weight array W(theta, lam) normalized to [0, 1].
    Peaks at the current trailing direction (lam_tidal = pi/2).
    """
    _, lam_tidal = _tidal_coordinates(theta, lam, spin_pole)
    W = np.maximum(0.0, np.sin(lam_tidal)) ** power
    wmax = np.max(W)
    if wmax > 0:
        W /= wmax
    return W


def _rotate_l2(t20, t22):
    """90° rotation about the sub-Jovian (x) axis for unnormalized (t20, t22).

    The code uses b = t20*P20 + t22*P22*cos(2λ) with un-normalized associated
    Legendre polynomials P20 = (3cos²θ-1)/2, P22 = 3sin²θ.  Direct coordinate
    substitution (cosθ' = sinθ sinλ) gives the 2×2 rotation matrix:

        M = | -1/2   -3   |    det(M) = -1,  M² = I.
            | -1/4   1/2  |

    This is a reflection (involution). Only valid for 90° about x; other angles
    mix in Y₂₁ components and break the (t20, t22) subspace closure.
    """
    return -0.5 * t20 - 3.0 * t22, -0.25 * t20 + 0.5 * t22


# ---------------------------------------------------------------------------
# Albedo darkening model
# ---------------------------------------------------------------------------

class AlbedoDarkeningModel:
    """Parameterized trailing-hemisphere darkening."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.theta = np.linspace(0.001, np.pi - 0.001, cfg.n_theta)
        self.lam = np.linspace(-np.pi, np.pi, cfg.n_lam)
        # Trailing hemisphere weight: peaks at λ = +π/2 (90° E)
        w = np.maximum(0, np.sin(self.lam)) ** cfg.darkening_power
        self.w = w / np.max(w)
        # Pre-compute insolation (for default spin pole = z-hat)
        self.os_temp = OSSurfaceTemperature()
        self.flux = self.os_temp.annual_average_flux(self.theta)

    def albedo_field(self, f_dark, spin_pole=None):
        """A(θ,λ) for darkening fraction f ∈ [0,1].

        If spin_pole is provided, darkening peaks at the current trailing
        hemisphere direction (rotates with spin pole).  Otherwise uses the
        default body-frame weight (spin pole = z-hat).
        """
        cfg = self.cfg
        if spin_pole is not None:
            w = _darkening_weight_for_spin_pole(
                self.theta, self.lam, spin_pole, cfg.darkening_power)
            return cfg.A_bright + (cfg.A_dark - cfg.A_bright) * f_dark * w
        return cfg.A_bright + (cfg.A_dark - cfg.A_bright) * f_dark * self.w[None, :]

    def surface_temperature(self, A):
        """T_s(θ,λ) from albedo via radiative equilibrium (spin pole = z-hat)."""
        return ((1.0 - A) * self.flux[:, None] / C.STEFAN_BOLTZMANN) ** 0.25

    def surface_temperature_for_pole(self, A, spin_pole):
        """T_s(θ,λ) using solar flux for arbitrary spin pole direction."""
        flux = _solar_flux_for_spin_pole(
            self.theta, self.lam, spin_pole, self.os_temp.annual_average_flux)
        return ((1.0 - A) * flux / C.STEFAN_BOLTZMANN) ** 0.25


# ---------------------------------------------------------------------------
# Quasi-static sweep
# ---------------------------------------------------------------------------

def quasi_static_sweep(cfg):
    """Sweep darkening fraction 0→1, compute equilibrium MoI at each step."""
    model = AlbedoDarkeningModel(cfg)
    D, H = _build_dissipation_grid(cfg.rheology_case, model.theta, model.lam)
    spin_pole = np.array([0.0, 0.0, 1.0])

    f_values = np.linspace(0, 1, cfg.n_darkening_steps)
    keys = ['f_dark', 'delta_BC_A', 'delta_BA_C', 'A_mean', 'b_avg_km',
            't20_km', 't22_km', 't20_2t22_km', '4t22_km',
            'shell_delta_BC_A', 'fossil_t20_km', 'fossil_t22_km',
            'misalignment_deg', 'I_max', 'I_mid', 'I_min']
    results = {k: np.zeros(len(f_values)) for k in keys}

    # Fossil bulge initialization
    fossil_t20, fossil_t22 = 0.0, 0.0
    b_fossil = None
    A0 = model.albedo_field(0.0)
    T_s0 = model.surface_temperature(A0)
    b0 = _equilibrium_thickness(T_s0, D, H)

    if cfg.use_fossil_bulge:
        if cfg.use_full_inertia:
            b_anomaly = b0 - _area_mean(b0, model.theta)
            b_fossil = (cfg.fossil_bulge_factor - 1.0) * b_anomaly
        else:
            mom0 = _compute_moments(b0, model.theta, model.lam)
            fossil_t20 = -mom0['t20']
            fossil_t22 = -mom0['t22']

    for i, f in enumerate(f_values):
        if cfg.use_full_inertia:
            A = model.albedo_field(f, spin_pole=spin_pole)
        else:
            A = model.albedo_field(f)
        T_s = model.surface_temperature(A)
        b_eq = _equilibrium_thickness(T_s, D, H)
        mom = _compute_moments(b_eq, model.theta, model.lam)

        results['f_dark'][i] = f
        results['shell_delta_BC_A'][i] = mom['delta_BC_A']
        results['A_mean'][i] = _area_mean(A, model.theta)
        results['b_avg_km'][i] = _area_mean(b_eq, model.theta) / 1000

        if cfg.use_full_inertia:
            I_shell = _compute_inertia_tensor(b_eq, model.theta, model.lam)
            if b_fossil is not None:
                I_fossil = _compute_inertia_tensor(
                    b_fossil, model.theta, model.lam)
                I_total = I_shell + I_fossil
            else:
                I_total = I_shell
            pa = _find_principal_axes(I_total, spin_pole)
            results['misalignment_deg'][i] = pa['misalignment_deg']
            results['I_min'][i] = pa['eigenvalues'][0]
            results['I_mid'][i] = pa['eigenvalues'][1]
            results['I_max'][i] = pa['eigenvalues'][2]
            # Also compute l=2 for comparison
            if b_fossil is not None:
                mom_f = _compute_moments(b_fossil, model.theta, model.lam)
                results['fossil_t20_km'][i] = mom_f['t20_km']
                results['fossil_t22_km'][i] = mom_f['t22_km']
                tot_t20_km = mom['t20_km'] + mom_f['t20_km']
                tot_t22_km = mom['t22_km'] + mom_f['t22_km']
            else:
                tot_t20_km = mom['t20_km']
                tot_t22_km = mom['t22_km']
            results['delta_BC_A'][i] = 1e-7 * (tot_t20_km + 2 * tot_t22_km)
            results['delta_BA_C'][i] = 1e-7 * 4 * tot_t22_km
            results['t20_km'][i] = tot_t20_km
            results['t22_km'][i] = tot_t22_km
            results['t20_2t22_km'][i] = tot_t20_km + 2 * tot_t22_km
            results['4t22_km'][i] = 4 * tot_t22_km
        else:
            results['fossil_t20_km'][i] = fossil_t20 / 1000
            results['fossil_t22_km'][i] = fossil_t22 / 1000
            if cfg.use_fossil_bulge:
                tot_t20_km = (mom['t20'] + fossil_t20) / 1000
                tot_t22_km = (mom['t22'] + fossil_t22) / 1000
                results['delta_BC_A'][i] = 1e-7 * (tot_t20_km + 2 * tot_t22_km)
                results['delta_BA_C'][i] = 1e-7 * 4 * tot_t22_km
                results['t20_km'][i] = tot_t20_km
                results['t22_km'][i] = tot_t22_km
                results['t20_2t22_km'][i] = tot_t20_km + 2 * tot_t22_km
                results['4t22_km'][i] = 4 * tot_t22_km
            else:
                results['delta_BC_A'][i] = mom['delta_BC_A']
                results['delta_BA_C'][i] = mom['delta_BA_C']
                results['t20_km'][i] = mom['t20_km']
                results['t22_km'][i] = mom['t22_km']
                results['t20_2t22_km'][i] = mom['t20_2t22_km']
                results['4t22_km'][i] = mom['4t22_km']

    return results


# ---------------------------------------------------------------------------
# Time-dependent evolution
# ---------------------------------------------------------------------------

class ShellEvolution:
    """Time-dependent shell evolution with darkening feedback."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.model = AlbedoDarkeningModel(cfg)
        self.theta = self.model.theta
        self.lam = self.model.lam
        self.D, self.H = _build_dissipation_grid(
            cfg.rheology_case, self.theta, self.lam)

        # Initial state: uniform bright albedo, equilibrium thickness
        A_init = self.model.albedo_field(0.0)
        T_s_init = self.model.surface_temperature(A_init)
        self.b = _equilibrium_thickness(T_s_init, self.D, self.H)
        self.f_dark = 0.0
        self.t = 0.0

        # Spin pole direction in body frame (initially along z-hat)
        self.spin_pole = np.array([0.0, 0.0, 1.0])

        # Cache the spin pole direction used for the current D grid.
        # D is recomputed when the pole moves > _D_update_threshold_deg.
        self._D_spin_pole = self.spin_pole.copy()
        self._D_update_threshold_deg = 1.0

        # Fossil bulge
        self.fossil_t20 = 0.0  # [m] (l=2 mode, for backward compat)
        self.fossil_t22 = 0.0  # [m]
        self.b_fossil = np.zeros_like(self.b)  # 2D field [m]
        if cfg.use_fossil_bulge:
            if cfg.use_full_inertia:
                # fossil = (alpha-1) * anomaly: REINFORCES the shell's stable
                # pattern (C > B > A).  alpha=1 → no fossil; alpha=1.5 → 50%
                # extra stability margin; alpha=2 → double margin.
                b_anomaly = self.b - _area_mean(self.b, self.theta)
                self.b_fossil = (cfg.fossil_bulge_factor - 1.0) * b_anomaly
            else:
                mom0 = _compute_moments(self.b, self.theta, self.lam)
                self.fossil_t20 = -mom0['t20']
                self.fossil_t22 = -mom0['t22']

        # OSPW dynamics state
        self._ospw_cfg = None
        if cfg.tpw_model in ('ospw', 'coupled'):
            self._ospw_cfg = OSPWConfig(
                shell_thickness=cfg.ospw_shell_thickness,
                lithosphere_thickness=cfg.ospw_lithosphere_thickness,
                melting_viscosity=cfg.ospw_melting_viscosity,
                activation_param=cfg.ospw_activation_param,
                fracture_base_temp=cfg.ospw_fracture_base_temp,
                surface_temp=cfg.ospw_surface_temp,
            )
            self._ospw_Cf = compute_Cf(self._ospw_cfg)
            self._ospw_I0 = compute_I0(self._ospw_cfg)
            self._ospw_n = C.MEAN_MOTION
            # Correction factor: geometric ε → OSPW ε (accounts for
            # isostatic compensation + iceberg effect)
            if cfg.use_iceberg_effect:
                self._ospw_eps_correction = _compute_ospw_epsilon_correction(cfg)
            else:
                self._ospw_eps_correction = 1.0

        # B&H interior coupling state
        if cfg.tpw_model == 'coupled':
            self.interior_pole = np.array([0.0, 0.0, 1.0])
            # K_G* = (3/2)(B_i-A_i) n² where (B_i-A_i) = eps_i * C_i
            # C_i = gamma * I_0 (interior MoI ≈ gamma × shell MoI)
            self._C_interior = cfg.interior_gamma * self._ospw_I0
            B_minus_A_interior = cfg.interior_triaxiality * self._C_interior
            KG_star = 1.5 * B_minus_A_interior * C.MEAN_MOTION**2
            self._K_G = cfg.interior_K_G_factor * KG_star

        hist_keys = [
            't_years', 'f_dark', 'delta_BC_A', 'delta_BA_C',
            'A_mean', 'b_avg_km', 't20_2t22_km', '4t22_km',
            'shell_delta_BC_A', 'shell_t20_km', 'shell_t22_km',
            'fossil_t20_km', 'fossil_t22_km',
            'total_t20_km', 'total_t22_km',
            'misalignment_deg', 'spin_pole_theta', 'spin_pole_phi',
            'I_max', 'I_mid', 'I_min', 'fossil_rms_km']
        if cfg.tpw_model in ('ospw', 'coupled'):
            hist_keys.append('tpw_omega_deg_per_Myr')
        if cfg.tpw_model == 'coupled':
            hist_keys.extend(['shell_interior_sep_deg',
                              'interior_pole_theta', 'interior_pole_phi'])
        self.history = {k: [] for k in hist_keys}
        self.history['cycle_events'] = []
        self._record()

    def _record(self):
        mom = _compute_moments(self.b, self.theta, self.lam)
        A = self.model.albedo_field(self.f_dark)
        self.history['t_years'].append(self.t)
        self.history['f_dark'].append(self.f_dark)
        self.history['A_mean'].append(_area_mean(A, self.theta))
        self.history['b_avg_km'].append(_area_mean(self.b, self.theta) / 1000)

        # Shell-only harmonics
        self.history['shell_delta_BC_A'].append(mom['delta_BC_A'])
        self.history['shell_t20_km'].append(mom['t20_km'])
        self.history['shell_t22_km'].append(mom['t22_km'])

        # Full inertia tensor path
        if self.cfg.use_full_inertia:
            I_shell = _compute_inertia_tensor(self.b, self.theta, self.lam)
            I_fossil = _compute_inertia_tensor(self.b_fossil, self.theta, self.lam)
            I_total = I_shell + I_fossil
            pa = _find_principal_axes(I_total, self.spin_pole)
            self.history['misalignment_deg'].append(pa['misalignment_deg'])
            vals = pa['eigenvalues']
            self.history['I_min'].append(vals[0])
            self.history['I_mid'].append(vals[1])
            self.history['I_max'].append(vals[2])
            # Spin pole direction (spherical coords in body frame)
            sp_theta = np.arccos(np.clip(self.spin_pole[2], -1, 1))
            sp_phi = np.arctan2(self.spin_pole[1], self.spin_pole[0])
            self.history['spin_pole_theta'].append(np.degrees(sp_theta))
            self.history['spin_pole_phi'].append(np.degrees(sp_phi))
            # Fossil RMS
            fossil_rms = np.sqrt(_area_mean(self.b_fossil**2, self.theta))
            self.history['fossil_rms_km'].append(fossil_rms / 1000)
            # Also compute l=2 fossil harmonics for comparison
            mom_f = _compute_moments(self.b_fossil, self.theta, self.lam)
            self.history['fossil_t20_km'].append(mom_f['t20_km'])
            self.history['fossil_t22_km'].append(mom_f['t22_km'])
            # Total l=2 harmonics
            tot_t20_km = mom['t20_km'] + mom_f['t20_km']
            tot_t22_km = mom['t22_km'] + mom_f['t22_km']
        else:
            self.history['misalignment_deg'].append(0.0)
            self.history['I_min'].append(0.0)
            self.history['I_mid'].append(0.0)
            self.history['I_max'].append(0.0)
            self.history['spin_pole_theta'].append(0.0)
            self.history['spin_pole_phi'].append(0.0)
            self.history['fossil_rms_km'].append(0.0)
            # Fossil bulge harmonics (l=2 path)
            self.history['fossil_t20_km'].append(self.fossil_t20 / 1000)
            self.history['fossil_t22_km'].append(self.fossil_t22 / 1000)
            # Total (shell + fossil)
            if self.cfg.use_fossil_bulge:
                tot_t20_km = (mom['t20'] + self.fossil_t20) / 1000
                tot_t22_km = (mom['t22'] + self.fossil_t22) / 1000
            else:
                tot_t20_km = mom['t20_km']
                tot_t22_km = mom['t22_km']

        self.history['total_t20_km'].append(tot_t20_km)
        self.history['total_t22_km'].append(tot_t22_km)
        self.history['delta_BC_A'].append(1e-7 * (tot_t20_km + 2 * tot_t22_km))
        self.history['delta_BA_C'].append(1e-7 * 4 * tot_t22_km)
        self.history['t20_2t22_km'].append(tot_t20_km + 2 * tot_t22_km)
        self.history['4t22_km'].append(4 * tot_t22_km)

        # OSPW/coupled mode diagnostics
        if self.cfg.tpw_model in ('ospw', 'coupled'):
            # Record 0 for angular velocity at recording time (actual value
            # is computed during step(); this placeholder keeps arrays aligned)
            self.history['tpw_omega_deg_per_Myr'].append(
                getattr(self, '_last_omega_deg_per_Myr', 0.0))
        if self.cfg.tpw_model == 'coupled':
            cos_sep = np.clip(
                np.dot(self.spin_pole, self.interior_pole), -1, 1)
            self.history['shell_interior_sep_deg'].append(
                np.degrees(np.arccos(cos_sep)))
            int_theta = np.arccos(np.clip(self.interior_pole[2], -1, 1))
            int_phi = np.arctan2(self.interior_pole[1], self.interior_pole[0])
            self.history['interior_pole_theta'].append(np.degrees(int_theta))
            self.history['interior_pole_phi'].append(np.degrees(int_phi))

    def _update_dissipation(self):
        """Recompute D(theta, lam) if the spin pole has drifted significantly.

        Tidal heating depends on spin-frame coordinates (O&S Appendix B), so D
        must be recalculated when the spin pole moves.  To avoid recomputing
        every timestep, D is cached and only updated when the pole has moved
        by more than _D_update_threshold_deg since the last computation.
        """
        if not self.cfg.use_full_inertia:
            return
        cos_a = np.clip(np.dot(self.spin_pole, self._D_spin_pole), -1, 1)
        angle = np.degrees(np.arccos(cos_a))
        if angle > self._D_update_threshold_deg:
            self.D, self.H = _build_dissipation_for_spin_pole(
                self.cfg.rheology_case, self.theta, self.lam, self.spin_pole)
            self._D_spin_pole = self.spin_pole.copy()

    def step(self, dt):
        cfg = self.cfg
        # 0. Update tidal heating if spin pole has drifted (full inertia path)
        self._update_dissipation()
        # 1. Exponential darkening
        self.f_dark = 1.0 - (1.0 - self.f_dark) * np.exp(-dt / cfg.tau_dark)
        # 2-3. Temperature and equilibrium thickness
        if cfg.use_full_inertia:
            A = self.model.albedo_field(self.f_dark, spin_pole=self.spin_pole)
            T_s = self.model.surface_temperature_for_pole(A, self.spin_pole)
        else:
            A = self.model.albedo_field(self.f_dark)
            T_s = self.model.surface_temperature(A)
        b_eq = _equilibrium_thickness(T_s, self.D, self.H)
        # 4. Exponential relaxation of shell toward equilibrium
        self.b = b_eq + (self.b - b_eq) * np.exp(-dt / cfg.tau_shell)
        # 5. Fossil bulge evolution (full inertia path)
        # The fossil relaxes toward (alpha-1)*current_equilibrium_anomaly on
        # timescale tau_visc.  This models continuous lithospheric freezing:
        # the fossil gradually adapts to reinforce the current shell pattern.
        # During stable periods: fossil forms and strengthens stability.
        # After TPW: fossil is "out of date" and evolves toward new target.
        # The lag between darkening (tau_dark) and fossil adaptation (tau_visc)
        # drives the instability-reorientation cycle.
        if cfg.use_fossil_bulge and cfg.use_full_inertia:
            b_eq_anomaly = b_eq - _area_mean(b_eq, self.theta)
            b_target = (cfg.fossil_bulge_factor - 1.0) * b_eq_anomaly
            decay = np.exp(-dt / cfg.tau_visc)
            self.b_fossil = b_target + (self.b_fossil - b_target) * decay
        # 6. Continuous polar wander (full inertia path)
        # The spin pole drifts toward v_max (eigenvector of maximum MoI).
        # Three dynamics models:
        #   exponential: free-parameter relaxation on tau_tpw
        #   ospw: overdamped viscous dissipation from OSPW shell rheology
        #   coupled: ospw + B&H gravity-gradient shell-interior coupling
        if cfg.use_full_inertia:
            I_shell = _compute_inertia_tensor(self.b, self.theta, self.lam)
            I_fossil = _compute_inertia_tensor(self.b_fossil, self.theta, self.lam)
            I_total = I_shell + I_fossil
            pa = _find_principal_axes(I_total, self.spin_pole)
            misalign_rad = np.radians(pa['misalignment_deg'])
            if misalign_rad > 1e-6:
                v_max = pa['v_max']
                dot = np.dot(v_max, self.spin_pole)
                v_perp = v_max - dot * self.spin_pole
                v_perp_norm = np.linalg.norm(v_perp)
                if v_perp_norm > 1e-10:
                    v_perp_hat = v_perp / v_perp_norm
                    if cfg.tpw_model == 'coupled':
                        # coupled mode updates spin_pole directly
                        # (2D vector sum of driving + coupling forces)
                        self._coupled_step(
                            dt, misalign_rad, pa, v_perp_hat)
                    else:
                        if cfg.tpw_model == 'exponential':
                            d_angle = misalign_rad * (
                                1.0 - np.exp(-dt / cfg.tau_tpw))
                        elif cfg.tpw_model == 'ospw':
                            d_angle = self._ospw_step(
                                dt, misalign_rad, pa)
                        else:
                            raise ValueError(
                                f'Unknown tpw_model: {cfg.tpw_model}')
                        self.spin_pole = (
                            self.spin_pole * np.cos(d_angle) +
                            v_perp_hat * np.sin(d_angle))
                        self.spin_pole /= np.linalg.norm(
                            self.spin_pole)
        self.t += dt

    # ------------------------------------------------------------------
    #  TPW dynamics methods
    # ------------------------------------------------------------------

    def _ospw_step(self, dt, misalign_rad, pa):
        """OSPW overdamped torque-balance TPW step.

        In the overdamped limit (ψ̈ ≈ 0):
            ω = driving / damping = 2ε²n²I₀/C_f · sin(2ψ)
        where ε is the OSPW-corrected triaxiality parameter.

        The OSPW correction accounts for isostatic compensation and the
        iceberg effect: ε_OSPW = ε_geometric × (Δρ/ρ_ice) × bracket.
        Without the correction, the geometric ε overestimates the driving
        torque by ~20× (the geometric tensor doesn't account for ocean
        displacement beneath thicker shell regions).
        """
        delta_I = pa['delta_I_all']  # I_max - I_min in I/(ρR⁴) units
        delta_I_phys = delta_I * C.ICE_DENSITY * C.EUROPA_RADIUS**4
        I0 = self._ospw_I0
        # Apply OSPW correction: geometric → physical epsilon
        epsilon = delta_I_phys * self._ospw_eps_correction / I0
        if abs(epsilon) < 1e-20:
            return 0.0
        n = self._ospw_n
        Cf = self._ospw_Cf
        # omega = 2ε²n²I₀/C_f · sin(2ψ) (overdamped balance)
        omega = 2.0 * epsilon**2 * n**2 * I0 / Cf * np.sin(2.0 * misalign_rad)
        # Store for diagnostics
        self._last_omega_deg_per_Myr = np.degrees(omega) * 1e6 * 365.25 * 86400
        dt_s = dt * 365.25 * 86400.0
        d_angle = omega * dt_s
        return min(d_angle, misalign_rad)

    def _coupled_step(self, dt, misalign_rad, pa, v_perp_hat):
        """OSPW dissipation + B&H gravity-gradient shell-interior coupling.

        Two torques act on the shell in different directions:
          1. Driving (toward v_max): ω_drive = 2ε²n²I₀/C_f · sin(2ψ)
          2. Coupling (toward interior): ω_couple = 4ε K_G sin(2δ) / C_f

        The interior stays approximately tidally locked (its libration
        period ~years is unresolvable at our dt ~kyr).  The coupling
        provides a restoring torque preventing the shell from drifting
        too far from the initial synchronous orientation.

        Updates self.spin_pole directly (vector sum of two forces).
        """
        delta_I = pa['delta_I_all']
        delta_I_phys = delta_I * C.ICE_DENSITY * C.EUROPA_RADIUS**4
        I0 = self._ospw_I0
        epsilon = delta_I_phys * self._ospw_eps_correction / I0
        if abs(epsilon) < 1e-20:
            return
        n = self._ospw_n
        Cf = self._ospw_Cf
        dt_s = dt * 365.25 * 86400.0

        # 1. Driving angular velocity toward v_max (same as _ospw_step)
        omega_drive = 2.0 * epsilon**2 * n**2 * I0 / Cf * np.sin(
            2.0 * misalign_rad)

        # 2. Coupling angular velocity toward interior_pole
        #    In overdamped limit: (K_G/I₀) sin(2δ) / γ_damp
        #    = (K_G/I₀) sin(2δ) × (4εI₀/Cf) = 4ε K_G sin(2δ) / Cf
        cos_sep = np.clip(
            np.dot(self.spin_pole, self.interior_pole), -1, 1)
        sep_rad = np.arccos(cos_sep)
        if sep_rad > 1e-10:
            omega_couple = (4.0 * abs(epsilon) * self._K_G
                            * np.sin(2.0 * sep_rad) / Cf)
        else:
            omega_couple = 0.0

        # Direction toward interior_pole on the unit sphere
        dot_ip = np.dot(self.interior_pole, self.spin_pole)
        v_couple = self.interior_pole - dot_ip * self.spin_pole
        v_couple_norm = np.linalg.norm(v_couple)

        # Total displacement: vector sum of driving + coupling
        d_spin = omega_drive * dt_s * v_perp_hat
        if v_couple_norm > 1e-10:
            v_couple_hat = v_couple / v_couple_norm
            d_spin += omega_couple * dt_s * v_couple_hat

        # Limit step size to prevent numerical overshooting
        # (same as _ospw_step's min(d_angle, misalign_rad))
        d_magnitude = np.linalg.norm(d_spin)
        if d_magnitude > misalign_rad and misalign_rad > 1e-10:
            d_spin *= misalign_rad / d_magnitude
            d_magnitude = misalign_rad
        if d_magnitude > 1e-15:
            self.spin_pole = self.spin_pole + d_spin
            self.spin_pole /= np.linalg.norm(self.spin_pole)

        self._last_omega_deg_per_Myr = (
            np.degrees(d_magnitude / dt_s) * 1e6 * 365.25 * 86400)

    def check_reorientation(self):
        cfg = self.cfg

        if cfg.use_full_inertia:
            return self._check_reorientation_full_tensor()
        else:
            return self._check_reorientation_l2()

    def _check_reorientation_full_tensor(self):
        """Track polar wander milestones (drift is continuous, in step()).

        Logs an event each time the spin pole has moved by
        misalignment_threshold_deg since the last logged event.
        Returns True when a milestone is crossed (for recording).
        """
        # Cumulative wander: angle from z-axis (initial orientation)
        pole_colat = np.degrees(np.arccos(
            np.clip(self.spin_pole[2], -1, 1)))
        if not hasattr(self, '_last_milestone_colat'):
            self._last_milestone_colat = 0.0
        delta = abs(pole_colat - self._last_milestone_colat)
        if delta >= self.cfg.misalignment_threshold_deg:
            n = len(self.history['cycle_events']) + 1
            self.history['cycle_events'].append((self.t, n))
            self._last_milestone_colat = pole_colat
            sp = self.spin_pole
            print(f'  Milestone {n}: t={self.t/1e6:.3f} Myr, '
                  f'pole at theta={pole_colat:.1f} deg, '
                  f'phi={np.degrees(np.arctan2(sp[1], sp[0])):.1f} deg')
            return True
        return False

    def _check_reorientation_l2(self):
        """Original l=2 harmonic stability check (backward compat)."""
        cfg = self.cfg
        mom = _compute_moments(self.b, self.theta, self.lam)

        if cfg.use_fossil_bulge:
            tot_t20_km = (mom['t20'] + self.fossil_t20) / 1000
            tot_t22_km = (mom['t22'] + self.fossil_t22) / 1000
            delta_BC_A = 1e-7 * (tot_t20_km + 2 * tot_t22_km)
        else:
            delta_BC_A = mom['delta_BC_A']

        if delta_BC_A > cfg.reorient_threshold:
            n = len(self.history['cycle_events']) + 1
            self.history['cycle_events'].append((self.t, n))

            if cfg.use_fossil_bulge:
                tot_t20 = mom['t20'] + self.fossil_t20
                tot_t22 = mom['t22'] + self.fossil_t22
                t20_rot, t22_rot = _rotate_l2(tot_t20, tot_t22)

                self.f_dark *= (1.0 - cfg.reorient_albedo_reset)
                A_new = self.model.albedo_field(self.f_dark)
                T_s_new = self.model.surface_temperature(A_new)
                self.b = _equilibrium_thickness(T_s_new, self.D, self.H)

                mom_new = _compute_moments(self.b, self.theta, self.lam)
                self.fossil_t20 = t20_rot - mom_new['t20']
                self.fossil_t22 = t22_rot - mom_new['t22']

                post_t20_km = (mom_new['t20'] + self.fossil_t20) / 1000
                post_t22_km = (mom_new['t22'] + self.fossil_t22) / 1000
                post_delta = 1e-7 * (post_t20_km + 2 * post_t22_km)
                if post_delta > cfg.reorient_threshold:
                    print(f'  WARNING: post-rotation still unstable '
                          f'(Delta(B-C)/A = {post_delta:.2e})')
            else:
                self.f_dark *= (1.0 - cfg.reorient_albedo_reset)
                A_new = self.model.albedo_field(self.f_dark)
                T_s_new = self.model.surface_temperature(A_new)
                self.b = _equilibrium_thickness(T_s_new, self.D, self.H)

            return True
        return False

    def run(self, record_every=None):
        cfg = self.cfg
        dt = cfg.dt_years
        n_steps = int(cfg.t_max_years / dt)
        if record_every is None:
            record_every = max(1, int(cfg.tau_dark / (10 * dt)))
        check_every = max(1, int(cfg.tau_dark / dt))
        n_cycles = 0

        for i in range(n_steps):
            self.step(dt)
            if i % record_every == 0:
                self._record()
            if i % check_every == 0:
                if self.check_reorientation():
                    n_cycles += 1
                    self._record()
                    # For l=2 (discrete jumps): stop after n_cycles_max events.
                    # For full tensor (continuous drift): milestones are just
                    # logging, don't stop the simulation.
                    if not cfg.use_full_inertia and n_cycles >= cfg.n_cycles_max:
                        break
        self._record()

        for k in self.history:
            if k != 'cycle_events':
                self.history[k] = np.array(self.history[k])
        return self.history


# ---------------------------------------------------------------------------
# Parameter survey
# ---------------------------------------------------------------------------

def parameter_survey(base_cfg, tau_dark_values=None, tau_shell_values=None):
    """Cycle period vs (τ_dark, τ_shell)."""
    if tau_dark_values is None:
        tau_dark_values = np.logspace(3, 5, 6)
    if tau_shell_values is None:
        tau_shell_values = np.logspace(6, 8, 6)

    periods = np.full((len(tau_dark_values), len(tau_shell_values)), np.nan)

    for i, td in enumerate(tau_dark_values):
        for j, ts in enumerate(tau_shell_values):
            print(f'  survey: tau_dark={td:.0e}, tau_shell={ts:.0e}...')
            cfg = PolarWanderConfig(
                tau_dark=td, tau_shell=ts,
                rheology_case=base_cfg.rheology_case,
                A_bright=base_cfg.A_bright,
                A_dark=base_cfg.A_dark,
                n_theta=91, n_lam=181,
                dt_years=min(td / 5, 1e3),
                t_max_years=5e8,
                n_cycles_max=3,
            )
            evo = ShellEvolution(cfg)
            hist = evo.run()
            events = hist['cycle_events']
            if len(events) >= 2:
                times = [e[0] for e in events]
                periods[i, j] = np.mean(np.diff(times))

    return {
        'tau_dark': tau_dark_values,
        'tau_shell': tau_shell_values,
        'cycle_period': periods,
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig1_quasi_static(sweep, cfg, save_path=None):
    """Δ(B-C)/A and misalignment angle vs darkening fraction."""
    has_tensor = cfg.use_full_inertia and np.any(sweep['misalignment_deg'] > 0)

    if has_tensor:
        fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(5.5, 7), sharex=True)
    else:
        fig, ax1 = plt.subplots(figsize=(5.5, 4))

    f = sweep['f_dark']
    dBC = sweep['delta_BC_A']
    ax1.plot(f, dBC * 1e8, 'C0-', linewidth=2)
    ax1.axhline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    ax1.set_ylabel(r'$\Delta(B\!-\!C)/A$ [$\times 10^{-8}$]')
    ax1.grid(True, alpha=0.2)

    # Find critical f
    idx = np.where(np.diff(np.sign(dBC)))[0]
    if len(idx) > 0:
        f_crit = np.interp(0, dBC[idx[0]:idx[0]+2], f[idx[0]:idx[0]+2])
        ax1.axvline(f_crit, color='r', linestyle=':', linewidth=1)
        ax1.annotate(rf'$f_{{\rm crit}}$ = {f_crit:.2f}',
                     xy=(f_crit, 0), fontsize=10,
                     xytext=(f_crit + 0.08, dBC.max() * 0.6e8),
                     arrowprops=dict(arrowstyle='->', color='r'))

    # Secondary: mean albedo
    ax2 = ax1.twiny()
    A_range = sweep['A_mean']
    ax2.set_xlim(A_range[0], A_range[-1])
    ax2.set_xlabel(r'Mean albedo $\langle A \rangle$')

    if has_tensor:
        ax1.text(-0.08, 1.04, '(a)', transform=ax1.transAxes,
                 fontsize=12, fontweight='bold', va='bottom')
        ax3.plot(f, sweep['misalignment_deg'], 'C3-', linewidth=2)
        ax3.axhline(cfg.misalignment_threshold_deg, color='r',
                     linestyle=':', linewidth=1, label='Threshold')
        ax3.set_xlabel('Darkening fraction $f$')
        ax3.set_ylabel('Misalignment angle [deg]')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.2)
        ax3.text(-0.08, 1.04, '(b)', transform=ax3.transAxes,
                 fontsize=12, fontweight='bold', va='bottom')
    else:
        ax1.set_xlabel('Darkening fraction $f$')

    label = CASE_STYLES.get(cfg.rheology_case, ('', '', '', cfg.rheology_case))[3]
    fb_tag = rf' ($\alpha$={cfg.fossil_bulge_factor})' if cfg.use_fossil_bulge else ''
    ax1.set_title(f'Polar wander stability: {label}{fb_tag}')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f'  Saved {save_path}')
    return fig


def fig2_time_series(history, cfg, save_path=None):
    """Multi-panel time series over wander cycles."""
    has_fossil = cfg.use_fossil_bulge
    has_tensor = cfg.use_full_inertia
    has_ospw = cfg.tpw_model in ('ospw', 'coupled')
    has_coupled = cfg.tpw_model == 'coupled'
    if has_tensor:
        n_panels = 6
    elif has_fossil:
        n_panels = 5
    else:
        n_panels = 4
    # Extra panels for OSPW/coupled diagnostics
    if has_ospw and has_tensor:
        n_panels += 1  # omega panel
    if has_coupled and has_tensor:
        n_panels += 1  # shell-interior separation panel
    fig, axes = plt.subplots(n_panels, 1, figsize=(8, 2 * n_panels), sharex=True)
    t_Myr = history['t_years'] / 1e6
    labels = 'abcdefgh'

    # Panels a-c: darkening, albedo, thickness
    simple_panels = [
        (axes[0], history['f_dark'], 'Darkening $f$', 'C0'),
        (axes[1], history['A_mean'], r'$\langle A \rangle$', 'C1'),
        (axes[2], history['b_avg_km'], r'$\langle b \rangle$ [km]', 'C2'),
    ]

    # Show event markers only for discrete l=2 model (too cluttered for
    # continuous drift milestones)
    show_events = not has_tensor

    for (ax, data, ylabel, color), lbl in zip(simple_panels, labels):
        ax.plot(t_Myr, data, color=color, linewidth=1)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, alpha=0.2)
        ax.text(-0.08, 1.04, f'({lbl})', transform=ax.transAxes,
                fontsize=12, fontweight='bold', va='bottom')
        if show_events:
            for t_event, _ in history['cycle_events']:
                ax.axvline(t_event / 1e6, color='red', linestyle='--',
                           linewidth=0.8, alpha=0.6)

    panel_idx = 3

    # Panel d: Δ(B-C)/A — show shell vs total if fossil bulge
    ax_d = axes[panel_idx]
    ax_d.plot(t_Myr, history['delta_BC_A'] * 1e8, 'C3-', linewidth=1.5,
              label='Total' if has_fossil else None)
    if has_fossil:
        ax_d.plot(t_Myr, history['shell_delta_BC_A'] * 1e8, 'C3--',
                  linewidth=0.8, alpha=0.6, label='Shell only')
        ax_d.legend(fontsize=8, loc='upper left')
    ax_d.axhline(0, color='k', linestyle='--', linewidth=0.5)
    ax_d.set_ylabel(r'$\Delta(B\!-\!C)/A$ [$\times 10^{-8}$]', fontsize=10)
    ax_d.grid(True, alpha=0.2)
    ax_d.text(-0.08, 1.04, f'({labels[panel_idx]})', transform=ax_d.transAxes,
              fontsize=12, fontweight='bold', va='bottom')
    if show_events:
        for t_event, _ in history['cycle_events']:
            ax_d.axvline(t_event / 1e6, color='red', linestyle='--',
                         linewidth=0.8, alpha=0.6)
    panel_idx += 1

    if has_tensor:
        # Panel e: Misalignment angle
        ax_mis = axes[panel_idx]
        ax_mis.plot(t_Myr, history['misalignment_deg'], 'C4-', linewidth=1.5)
        ax_mis.axhline(cfg.misalignment_threshold_deg, color='r',
                       linestyle=':', linewidth=1, alpha=0.7)
        ax_mis.set_ylabel('Misalignment [deg]', fontsize=10)
        ax_mis.grid(True, alpha=0.2)
        ax_mis.text(-0.08, 1.04, f'({labels[panel_idx]})',
                    transform=ax_mis.transAxes,
                    fontsize=12, fontweight='bold', va='bottom')
        panel_idx += 1

        # Panel f: Spin pole colatitude
        ax_sp = axes[panel_idx]
        ax_sp.plot(t_Myr, history['spin_pole_theta'], 'C5-', linewidth=1.5)
        ax_sp.set_ylabel(r'Spin pole $\theta$ [deg]', fontsize=10)
        ax_sp.grid(True, alpha=0.2)
        ax_sp.text(-0.08, 1.04, f'({labels[panel_idx]})',
                   transform=ax_sp.transAxes,
                   fontsize=12, fontweight='bold', va='bottom')
        panel_idx += 1

        # Panel g: TPW angular velocity (ospw/coupled modes)
        if has_ospw and 'tpw_omega_deg_per_Myr' in history:
            ax_om = axes[panel_idx]
            omega = np.array(history['tpw_omega_deg_per_Myr'])
            ax_om.plot(t_Myr, omega, 'C6-', linewidth=1)
            ax_om.set_ylabel(r'$\omega_{\rm TPW}$ [deg/Myr]', fontsize=10)
            ax_om.set_yscale('symlog', linthresh=10.0)
            ax_om.grid(True, alpha=0.2)
            ax_om.text(-0.08, 1.04, f'({labels[panel_idx]})',
                       transform=ax_om.transAxes,
                       fontsize=12, fontweight='bold', va='bottom')
            panel_idx += 1

        # Panel h: shell-interior separation (coupled mode)
        if has_coupled and 'shell_interior_sep_deg' in history:
            ax_sep = axes[panel_idx]
            sep = np.array(history['shell_interior_sep_deg'])
            ax_sep.plot(t_Myr, sep, 'C7-', linewidth=1)
            ax_sep.set_ylabel('Shell-int. sep. [deg]', fontsize=10)
            ax_sep.grid(True, alpha=0.2)
            ax_sep.text(-0.08, 1.04, f'({labels[panel_idx]})',
                       transform=ax_sep.transAxes,
                       fontsize=12, fontweight='bold', va='bottom')
            panel_idx += 1
    elif has_fossil:
        # Panel e (l=2 path): harmonic decomposition
        ax_e = axes[panel_idx]
        ax_e.plot(t_Myr, history['total_t20_km'], 'C0-', linewidth=1,
                  label=r'Total $t_{20}$')
        ax_e.plot(t_Myr, history['total_t22_km'], 'C1-', linewidth=1,
                  label=r'Total $t_{22}$')
        ax_e.plot(t_Myr, history['fossil_t20_km'], 'C0:', linewidth=1,
                  alpha=0.6, label=r'Fossil $t_{20}$')
        ax_e.plot(t_Myr, history['fossil_t22_km'], 'C1:', linewidth=1,
                  alpha=0.6, label=r'Fossil $t_{22}$')
        ax_e.axhline(0, color='k', linestyle='--', linewidth=0.5)
        ax_e.set_ylabel('Harmonics [km]', fontsize=10)
        ax_e.legend(fontsize=7, ncol=2, loc='best')
        ax_e.grid(True, alpha=0.2)
        ax_e.text(-0.08, 1.04, f'({labels[panel_idx]})',
                  transform=ax_e.transAxes,
                  fontsize=12, fontweight='bold', va='bottom')
        for t_event, _ in history['cycle_events']:
            ax_e.axvline(t_event / 1e6, color='red', linestyle='--',
                         linewidth=0.8, alpha=0.6)
        panel_idx += 1

    axes[-1].set_xlabel('Time [Myr]')

    label = CASE_STYLES.get(cfg.rheology_case, ('', '', '', cfg.rheology_case))[3]
    fb_tag = ''
    if has_fossil:
        fb_tag = rf' (fossil $\alpha$={cfg.fossil_bulge_factor})'
    if has_tensor:
        fb_tag += ' [full tensor]'
    if cfg.tpw_model != 'exponential':
        fb_tag += f' [{cfg.tpw_model} TPW]'
    fig.suptitle(f'Polar wander cycle: {label}{fb_tag}',
                 fontsize=13, fontweight='bold', y=1.0)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f'  Saved {save_path}')
    return fig


def fig3_cycle_period(survey, save_path=None):
    """Heatmap of cycle period vs τ_dark and τ_shell."""
    fig, ax = plt.subplots(figsize=(6, 5))

    td = survey['tau_dark']
    ts = survey['tau_shell']
    T_Myr = survey['cycle_period'] / 1e6
    valid = T_Myr[np.isfinite(T_Myr)]
    if len(valid) == 0:
        print('  No valid cycle periods found, skipping fig3')
        plt.close(fig)
        return None

    cs = ax.pcolormesh(td, ts, T_Myr.T,
                       cmap='viridis',
                       norm=colors.LogNorm(vmin=max(0.01, valid.min()),
                                           vmax=valid.max()),
                       shading='nearest')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\tau_{\rm dark}$ [yr]')
    ax.set_ylabel(r'$\tau_{\rm shell}$ [yr]')
    cbar = plt.colorbar(cs, ax=ax)
    cbar.set_label('Cycle period [Myr]')
    ax.set_title('Polar wander cycle period')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f'  Saved {save_path}')
    return fig


def fig4_mollweide_snapshots(cfg, save_path=None):
    """Mollweide snapshots at 4 phases of one cycle."""
    model = AlbedoDarkeningModel(cfg)
    theta, lam = model.theta, model.lam
    lat = np.pi / 2 - theta
    LON, LAT = np.meshgrid(lam, lat, indexing='ij')
    D, H = _build_dissipation_grid(cfg.rheology_case, theta, lam)

    # Four darkening fractions: bright, 25%, 75%, full
    f_values = [0.0, 0.25, 0.75, 1.0]
    labels_top = [r'$f = 0$ (bright)', r'$f = 0.25$',
                  r'$f = 0.75$', r'$f = 1$ (dark)']

    fig = plt.figure(figsize=(7.2, 7.5))
    gs = GridSpec(2, 2, figure=fig, hspace=0.12, wspace=0.02)

    # Collect all thickness data for shared colorbar
    all_b = []
    for f in f_values:
        A = model.albedo_field(f)
        T_s = model.surface_temperature(A)
        b = _equilibrium_thickness(T_s, D, H) / 1000
        all_b.append(b)
    vmin = max(0, min(b.min() for b in all_b))
    vmax = max(b.max() for b in all_b)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)

    for idx, (f, b_km, title) in enumerate(zip(f_values, all_b, labels_top)):
        ax = fig.add_subplot(gs[idx // 2, idx % 2], projection='mollweide')
        ax.pcolormesh(LON.T, LAT.T, b_km, cmap=THICKNESS_CMAP,
                      norm=norm, shading='auto', rasterized=True)
        ax.grid(True, linewidth=0.3, alpha=0.4, color='w')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        mom = _compute_moments(b_km * 1000, theta, lam)
        ax.set_title(f'{title}\n'
                     rf'$\langle b \rangle$ = {_area_mean(b_km*1000, theta)/1000:.1f} km, '
                     rf'$\Delta(B\!-\!C)/A$ = {mom["delta_BC_A"]:.1e}',
                     fontsize=9, pad=4)
        ax.text(0.02, 0.96, f'({chr(97+idx)})', transform=ax.transAxes,
                fontsize=11, fontweight='bold', va='top')

    cbar_ax = fig.add_axes([0.15, 0.04, 0.7, 0.018])
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=THICKNESS_CMAP),
                 cax=cbar_ax, orientation='horizontal',
                 label='Ice shell thickness [km]')

    label = CASE_STYLES.get(cfg.rheology_case, ('', '', '', cfg.rheology_case))[3]
    fig.suptitle(f'Shell thickness at progressive darkening: {label}',
                 fontsize=13, fontweight='bold', y=0.98)
    if save_path:
        fig.savefig(save_path)
        print(f'  Saved {save_path}')
    return fig


def fig5_all_rheologies(save_path=None, A_bright=0.55, A_dark=0.15,
                        use_fossil_bulge=True, use_full_inertia=True,
                        fossil_bulge_factor=1.5):
    """Δ(B-C)/A and misalignment vs f for all 4 rheology cases."""
    show_tensor = use_fossil_bulge and use_full_inertia
    if show_tensor:
        fig, ((ax_tl, ax_tr), (ax_bl, ax_br)) = plt.subplots(
            2, 2, figsize=(11, 9))
    elif use_fossil_bulge:
        fig, (ax_tl, ax_tr) = plt.subplots(1, 2, figsize=(11, 4.5))
        ax_bl = ax_br = None
    else:
        fig, ax_tl = plt.subplots(figsize=(6, 4.5))
        ax_tr = ax_bl = ax_br = None

    for case, (color, marker, ls, label) in CASE_STYLES.items():
        print(f'  sweep: {case}...')
        # Without fossil bulge
        cfg_nofb = PolarWanderConfig(
            rheology_case=case, A_bright=A_bright, A_dark=A_dark,
            use_fossil_bulge=False, use_full_inertia=False)
        sweep_nofb = quasi_static_sweep(cfg_nofb)
        ax_tl.plot(sweep_nofb['f_dark'], sweep_nofb['delta_BC_A'] * 1e8,
                   color=color, marker=marker, markevery=5, markersize=4,
                   linestyle=ls, linewidth=1.5, label=label)

        if ax_tr is not None:
            cfg_fb = PolarWanderConfig(
                rheology_case=case, A_bright=A_bright, A_dark=A_dark,
                use_fossil_bulge=True, use_full_inertia=use_full_inertia,
                fossil_bulge_factor=fossil_bulge_factor)
            sweep_fb = quasi_static_sweep(cfg_fb)
            ax_tr.plot(sweep_fb['f_dark'], sweep_fb['delta_BC_A'] * 1e8,
                       color=color, marker=marker, markevery=5, markersize=4,
                       linestyle=ls, linewidth=1.5, label=label)

            if show_tensor:
                # Bottom-left: misalignment (no fossil)
                cfg_nofb_t = PolarWanderConfig(
                    rheology_case=case, A_bright=A_bright, A_dark=A_dark,
                    use_fossil_bulge=False, use_full_inertia=True)
                sweep_nofb_t = quasi_static_sweep(cfg_nofb_t)
                ax_bl.plot(sweep_nofb_t['f_dark'],
                           sweep_nofb_t['misalignment_deg'],
                           color=color, marker=marker, markevery=5,
                           markersize=4, linestyle=ls, linewidth=1.5,
                           label=label)
                # Bottom-right: misalignment (with fossil)
                ax_br.plot(sweep_fb['f_dark'], sweep_fb['misalignment_deg'],
                           color=color, marker=marker, markevery=5,
                           markersize=4, linestyle=ls, linewidth=1.5,
                           label=label)

    for ax in [ax_tl, ax_tr]:
        if ax is not None:
            ax.axhline(0, color='k', linestyle='--', linewidth=0.8)
            ax.set_xlabel('Darkening fraction $f$')
            ax.set_ylabel(r'$\Delta(B\!-\!C)/A$ [$\times 10^{-8}$]')
            ax.legend(fontsize=8.5, loc='best')
            ax.grid(True, alpha=0.2)

    ax_tl.set_title('Without fossil bulge')
    if ax_tr is not None:
        ax_tr.set_title(rf'With fossil bulge ($\alpha$={fossil_bulge_factor})')

    if show_tensor:
        for ax in [ax_bl, ax_br]:
            ax.set_xlabel('Darkening fraction $f$')
            ax.set_ylabel('Misalignment angle [deg]')
            ax.legend(fontsize=8.5, loc='best')
            ax.grid(True, alpha=0.2)
        ax_bl.set_title('Misalignment (no fossil)')
        ax_br.set_title(rf'Misalignment ($\alpha$={fossil_bulge_factor})')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f'  Saved {save_path}')
    return fig


def fig6_spin_pole_trajectory(history, cfg, save_path=None):
    """Spin pole trajectory on polar stereographic projection (body frame)."""
    if not cfg.use_full_inertia:
        return None

    sp_theta = np.array(history['spin_pole_theta'])  # colatitude [deg]
    sp_phi = np.array(history['spin_pole_phi'])  # longitude [deg]
    t_years = np.array(history['t_years'])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5),
                             subplot_kw={'projection': 'polar'})

    label = CASE_STYLES.get(
        cfg.rheology_case, ('', '', '', cfg.rheology_case))[3]
    fig.suptitle(f'Spin pole trajectory (body frame): {label}',
                 fontsize=12, fontweight='bold')

    for ax_idx, ax in enumerate(axes):
        phi_rad = np.radians(sp_phi)
        r = sp_theta  # colatitude in degrees = radial distance

        # Left: trajectory colored by time
        if ax_idx == 0:
            # Plot trajectory colored by time using scatter
            t_myr = t_years / 1e6
            sc = ax.scatter(phi_rad, r, c=t_myr, cmap='viridis', s=1,
                            zorder=2)
            # Connect with faint line
            ax.plot(phi_rad, r, 'k-', linewidth=0.3, alpha=0.3, zorder=1)
            ax.plot(phi_rad[0], r[0], 'go', markersize=10, label='Start',
                    zorder=5)
            ax.plot(phi_rad[-1], r[-1], 'rs', markersize=8, label='End',
                    zorder=5)
            cb = fig.colorbar(sc, ax=ax, pad=0.1, shrink=0.8)
            cb.set_label('Time [Myr]', fontsize=9)
            ax.set_title('Time-colored trajectory', pad=15, fontsize=10)
            ax.legend(fontsize=8, loc='lower left')

        # Right: trajectory colored by misalignment
        else:
            misalign = np.array(history['misalignment_deg'])
            n = min(len(phi_rad), len(misalign))
            sc = ax.scatter(phi_rad[:n], r[:n], c=misalign[:n],
                            cmap='hot_r', s=1, vmin=0, vmax=90, zorder=2)
            ax.plot(phi_rad, r, 'k-', linewidth=0.3, alpha=0.3, zorder=1)
            cb = fig.colorbar(sc, ax=ax, pad=0.1, shrink=0.8)
            cb.set_label('Misalignment [deg]', fontsize=9)
            ax.set_title('Misalignment-colored', pad=15, fontsize=10)

        # Configure polar axes
        max_r = max(5.0, np.ceil(np.max(sp_theta) / 5) * 5 + 5)
        ax.set_rlim(0, max_r)
        ax.set_rlabel_position(135)
        ax.set_theta_zero_location('E')  # phi=0 = sub-Jovian → right

        # Label body-frame directions
        ax.annotate('Sub-Jovian', xy=(0, max_r), fontsize=7,
                    ha='left', va='bottom')
        ax.annotate('Trailing', xy=(np.pi / 2, max_r), fontsize=7,
                    ha='center', va='bottom')
        ax.annotate('Anti-Jovian', xy=(np.pi, max_r), fontsize=7,
                    ha='right', va='bottom')
        ax.annotate('Leading', xy=(3 * np.pi / 2, max_r), fontsize=7,
                    ha='center', va='top')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f'  Saved {save_path}')
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Europa polar wander cycle simulation')
    parser.add_argument('--save-dir', default='plots/polar_wander')
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--format', default='pdf', choices=['pdf', 'png'])
    parser.add_argument('--rheology', default='maxwell_h0',
                        choices=['maxwell_h0', 'maxwell_h10',
                                 'flow_ng3', 'flow_ng52'])
    parser.add_argument('--tau-dark', type=float, default=1e4)
    parser.add_argument('--tau-shell', type=float, default=1e7)
    parser.add_argument('--A-bright', type=float, default=0.55)
    parser.add_argument('--A-dark', type=float, default=0.15)
    parser.add_argument('--n-cycles', type=int, default=5)
    parser.add_argument('--sweep-only', action='store_true',
                        help='Only run quasi-static sweep (fast)')
    parser.add_argument('--survey', action='store_true',
                        help='Run tau_dark x tau_shell parameter survey')
    parser.add_argument('--all-rheologies', action='store_true',
                        help='Quasi-static sweep for all 4 rheologies')
    parser.add_argument('--no-fossil-bulge', action='store_true',
                        help='Disable fossil bulge (original behavior)')
    parser.add_argument('--fossil-factor', type=float, default=1.5,
                        help='Fossil bulge factor alpha (default: 1.5)')
    parser.add_argument('--tau-visc', type=float, default=3e7,
                        help='Fossil relaxation timescale [yr] (default: 3e7)')
    parser.add_argument('--no-full-inertia', action='store_true',
                        help='Use l=2 harmonics instead of full tensor')
    parser.add_argument('--misalignment-threshold', type=float, default=5.0,
                        help='Min spin pole offset to trigger TPW [deg]')
    parser.add_argument('--tau-tpw', type=float, default=1e5,
                        help='TPW relaxation timescale [yr] (default: 1e5)')
    # TPW dynamics model
    parser.add_argument('--tpw-model', default='exponential',
                        choices=['exponential', 'ospw', 'coupled'],
                        help='TPW dynamics: exponential (free tau), '
                             'ospw (viscous), coupled (ospw+interior)')
    # OSPW parameters
    parser.add_argument('--ospw-Tf', type=float, default=130.0,
                        help='Fracture base temperature [K] (default: 130)')
    parser.add_argument('--ospw-eta0', type=float, default=3e14,
                        help='Melting viscosity [Pa s] (default: 3e14)')
    parser.add_argument('--ospw-l', type=float, default=15.0,
                        help='Arrhenius activation parameter (default: 15)')
    parser.add_argument('--ospw-bc', type=float, default=20.0,
                        help='Shell thickness [km] (default: 20)')
    parser.add_argument('--ospw-d-lith', type=float, default=2.0,
                        help='Lithosphere thickness [km] (default: 2)')
    # Iceberg effect
    parser.add_argument('--no-iceberg', action='store_true',
                        help='Disable iceberg amplification of inertia tensor')
    parser.add_argument('--iceberg-amp', type=float, default=0.0,
                        help='Override iceberg amplification factor (0=auto)')
    # B&H interior coupling
    parser.add_argument('--interior-gamma', type=float, default=28.0,
                        help='C_interior/C_shell ratio (default: 28)')
    parser.add_argument('--interior-triaxiality', type=float, default=1e-5,
                        help='Interior (B-A)/C triaxiality (default: 1e-5)')
    parser.add_argument('--interior-KG-factor', type=float, default=0.5,
                        help='K_G/K_G* coupling strength (default: 0.5 = weak)')
    parser.add_argument('--dt', type=float, default=None,
                        help='Time step [yr] (default: tau_tpw/10 for full '
                             'tensor, 1e3 for l=2)')
    parser.add_argument('--t-max', type=float, default=1e8,
                        help='Total simulation time [yr] (default: 1e8)')
    args = parser.parse_args()

    use_full = not args.no_full_inertia
    if args.dt is not None:
        dt = args.dt
    elif use_full:
        if args.tpw_model == 'exponential':
            dt = args.tau_tpw / 10.0
        else:
            # OSPW/coupled: adapt dt to the slower τ_f timescale
            temp_cfg = OSPWConfig(
                fracture_base_temp=args.ospw_Tf,
                melting_viscosity=args.ospw_eta0,
                activation_param=args.ospw_l,
            )
            tau_f_yr, _ = compute_tau_f(temp_cfg)
            dt = max(1e3, min(tau_f_yr / 100.0, 1e4))
    else:
        dt = 1e3

    cfg = PolarWanderConfig(
        rheology_case=args.rheology,
        tau_dark=args.tau_dark,
        tau_shell=args.tau_shell,
        A_bright=args.A_bright,
        A_dark=args.A_dark,
        dt_years=dt,
        t_max_years=args.t_max,
        n_cycles_max=args.n_cycles,
        use_fossil_bulge=not args.no_fossil_bulge,
        fossil_bulge_factor=args.fossil_factor,
        tau_visc=args.tau_visc,
        use_full_inertia=use_full,
        misalignment_threshold_deg=args.misalignment_threshold,
        tau_tpw=args.tau_tpw,
        tpw_model=args.tpw_model,
        ospw_shell_thickness=args.ospw_bc * 1e3,
        ospw_lithosphere_thickness=args.ospw_d_lith * 1e3,
        ospw_melting_viscosity=args.ospw_eta0,
        ospw_activation_param=args.ospw_l,
        ospw_fracture_base_temp=args.ospw_Tf,
        use_iceberg_effect=not args.no_iceberg,
        iceberg_amplification=args.iceberg_amp,
        interior_gamma=args.interior_gamma,
        interior_triaxiality=args.interior_triaxiality,
        interior_K_G_factor=args.interior_KG_factor,
    )

    plt.rcParams['savefig.dpi'] = args.dpi
    os.makedirs(args.save_dir, exist_ok=True)

    def sp(name):
        return os.path.join(args.save_dir, f'{name}.{args.format}')

    # --- Quasi-static sweep ---
    fb_str = 'ON' if cfg.use_fossil_bulge else 'OFF'
    tensor_str = 'full tensor' if cfg.use_full_inertia else 'l=2 harmonics'
    print(f'Fossil bulge: {fb_str} (alpha={cfg.fossil_bulge_factor})')
    print(f'Inertia model: {tensor_str}')
    if cfg.use_full_inertia:
        print(f'  tau_visc={cfg.tau_visc:.0e} yr, '
              f'misalignment threshold={cfg.misalignment_threshold_deg} deg')
        print(f'  TPW model: {cfg.tpw_model}')
        if cfg.tpw_model == 'exponential':
            print(f'  tau_tpw={cfg.tau_tpw:.0e} yr (free parameter)')
        if cfg.use_iceberg_effect:
            print(f'  Iceberg effect: ON')
    print(f'Quasi-static sweep: {cfg.rheology_case}...')
    sweep = quasi_static_sweep(cfg)

    # Print summary
    idx = np.where(np.diff(np.sign(sweep['delta_BC_A'])))[0]
    if len(idx) > 0:
        f_crit = np.interp(0, sweep['delta_BC_A'][idx[0]:idx[0]+2],
                           sweep['f_dark'][idx[0]:idx[0]+2])
        A_crit = np.interp(f_crit, sweep['f_dark'], sweep['A_mean'])
        print(f'  Critical darkening: f_crit = {f_crit:.3f}, '
              f'A_mean = {A_crit:.3f}')
    else:
        if sweep['delta_BC_A'][0] > 0:
            print('  Already unstable at f=0')
        else:
            print('  Never reaches instability')

    if cfg.use_full_inertia:
        print(f'  f=0: misalignment = {sweep["misalignment_deg"][0]:.2f} deg')
        print(f'  f=1: misalignment = {sweep["misalignment_deg"][-1]:.2f} deg')

    print(f'  f=0: Delta(B-C)/A = {sweep["delta_BC_A"][0]:.2e}')
    print(f'  f=1: Delta(B-C)/A = {sweep["delta_BC_A"][-1]:.2e}')

    fig1_quasi_static(sweep, cfg, save_path=sp('fig1_quasi_static'))

    if args.all_rheologies:
        print('All-rheology sweep...')
        fig5_all_rheologies(save_path=sp('fig5_all_rheologies'),
                            A_bright=args.A_bright, A_dark=args.A_dark,
                            use_fossil_bulge=cfg.use_fossil_bulge,
                            use_full_inertia=cfg.use_full_inertia,
                            fossil_bulge_factor=cfg.fossil_bulge_factor)

    if not args.sweep_only:
        # --- Mollweide snapshots ---
        print('Mollweide snapshots...')
        fig4_mollweide_snapshots(cfg, save_path=sp('fig4_mollweide'))

        # --- Time-dependent simulation ---
        print(f'Time-dependent simulation: tau_dark={cfg.tau_dark:.0e}, '
              f'tau_shell={cfg.tau_shell:.0e}...')
        evo = ShellEvolution(cfg)
        # Print OSPW/coupled diagnostics
        if cfg.tpw_model in ('ospw', 'coupled') and cfg.use_full_inertia:
            tau_f_yr, _ = compute_tau_f(evo._ospw_cfg)
            print(f'  OSPW: C_f={evo._ospw_Cf:.3e} kg m^2/s, '
                  f'tau_f={tau_f_yr/1e6:.2f} Myr')
            print(f'  OSPW epsilon correction: {evo._ospw_eps_correction:.4f}')
            if cfg.tpw_model == 'coupled':
                regime = 'weak' if cfg.interior_K_G_factor < 1.0 else 'strong'
                print(f'  Interior coupling: {regime} '
                      f'(K_G/K_G*={cfg.interior_K_G_factor}, '
                      f'eps_i={cfg.interior_triaxiality:.1e}, '
                      f'gamma={cfg.interior_gamma}, '
                      f'K_G={evo._K_G:.3e} N m)')
        history = evo.run()
        n_events = len(history['cycle_events'])
        print(f'  {n_events} reorientation events')
        if n_events >= 1:
            times = [e[0] for e in history['cycle_events']]
            print(f'  Event times [Myr]: {[f"{t/1e6:.3f}" for t in times]}')
        if n_events >= 2:
            period = np.mean(np.diff(times))
            print(f'  Mean cycle period: {period/1e6:.1f} Myr')

        fig2_time_series(history, cfg, save_path=sp('fig2_time_series'))

        if cfg.use_full_inertia:
            fig6_spin_pole_trajectory(history, cfg,
                                      save_path=sp('fig6_spin_pole'))

    if args.survey:
        print('Parameter survey...')
        survey = parameter_survey(cfg)
        fig3_cycle_period(survey, save_path=sp('fig3_cycle_period'))

    plt.close('all')
    print('Done.')


if __name__ == '__main__':
    main()
