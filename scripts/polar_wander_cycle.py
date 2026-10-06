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
from ojakangas_stevenson import grid as G
from ojakangas_stevenson import polar_wander as PW
from ojakangas_stevenson.surface_temperature import OSSurfaceTemperature
from ojakangas_stevenson.shell_thickness import (
    equilibrium_thickness, dissipation_field, table1_case)

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


# Default psi0 (decision of 2026-10-05): Europa's predicted forced obliquity,
# ~0.1 deg (Bills 2005, Icarus 175, 233-247). UNVERIFIED: the number has not
# been checked against the paper in this repository.
PSI0_OBLIQUITY_RAD = np.radians(0.1)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class PolarWanderConfig:
    """Parameters for the polar wander cycle simulation."""
    A_bright: float = 0.55
    A_dark: float = 0.3814           # Hapke non-ice endmember (decision of 2026-10-04)
    darkening_power: float = 1.0
    tau_dark: float = 1e4
    tau_shell: float = 1e7
    reorient_threshold: float = 5e-9
    reorient_albedo_reset: float = 1.0
    rheology_case: str = 'maxwell_h0'
    n_theta: int = 181
    n_lam: int = 360                 # periodic grid: no duplicated +-180 column
    dt_years: float = 1e3
    t_max_years: float = 1e8
    n_cycles_max: int = 10
    n_darkening_steps: int = 50
    # Fossil bulge: True = BODY-FIXED hydrostatic bulge x (1 - C_w), OSPW
    # Eqs. 3.8-3.10; False = no fossil. (Replaces the unphysical alpha factor.)
    use_fossil_bulge: bool = True
    use_full_inertia: bool = True
    misalignment_threshold_deg: float = 5.0
    tau_tpw: float = 1e5              # TPW relaxation timescale [years]
    # --- TPW dynamics model ---
    tpw_model: str = 'exponential'    # 'exponential', 'ospw', or 'coupled'
    # --- OSPW rheological parameters (ospw/coupled modes) ---
    ospw_shell_thickness: float = None       # b_c [m]; None = initial mean b
    ospw_lithosphere_thickness: float = 2e3  # d [m] (ignored if ospw_bc_over_d is set)
    ospw_bc_over_d: float = None             # e.g. 10: d = b_c/10, as in OSPW Eq. 3.12
    ospw_melting_viscosity: float = 3e14     # eta_0 [Pa s]
    ospw_activation_param: float = 15.0      # l (dimensionless)
    ospw_fracture_base_temp: float = 124.0   # T_f [K] (OSPW p. 264)
    ospw_surface_temp: float = 100.0         # T_s [K]
    # --- Iceberg amplification ---
    use_iceberg_effect: bool = True          # Eq. 2.18 bottom loading (False: rigid load)
    # --- psi0 seed and thickness law ---
    # psi0 = Europa's predicted forced obliquity, ~0.1 deg (Bills 2005, Icarus
    # 175, 233; UNVERIFIED: value not checked against the paper). Bracket x0.1, x10.
    psi0_rad: float = PSI0_OBLIQUITY_RAD
    thickness_method: str = 'eq28'           # 'eq28' or 'exact' (shell_thickness)
    D_update_threshold_deg: float = 0.0      # recompute tidal heating when the pole moves
    # --- 2026-10-05 dynamics fixes (Fermi review of the dynamic TPW runs) ---
    # Tidal torque -3 n^2 (I_j - I_k) from Jupiter (OSPW Eqs. 5.19-5.21):
    # stiffness 4 n^2 eps toward Jupiter (Eqs. 5.26/5.38). False = the
    # pre-2026-10-05 rotational torque only (legacy reproduction).
    tidal_torque: bool = True
    # psi0 re-seeding: 'any_axis' = whenever the pole sits within psi0 of a
    # principal axis that is not the max-moment axis (saddles included);
    # 'max_axis' = the pre-2026-10-05 rule (re-armed only on the max axis,
    # so a pole could park on a saddle).
    seed_rearm: str = 'any_axis'
    seed_rearm_deg: float = 1.0       # 'any_axis': re-arm after the pole moves this far
    # --- convection bracketing proxy (Paul 2026-10-05; Fermi's assessment) ---
    # The shell stays conductive O&S; these knobs bracket a stagnant-lid
    # convecting shell. In a stagnant lid the thickness response per kelvin of
    # surface T is the same as in O&S, but basal flow damps the degree-2
    # anomaly and the tidal heat goes to the convecting layer. So:
    #   conv_fD: weight of the tidal-heating (D) contribution to the
    #            equilibrium anomaly, b_f = B(T_s, <D>) + f_D [B(T_s, D) - B(T_s, <D>)]
    #            (surface-T contribution unchanged);
    #   conv_s:  steady-state flow damping of the whole anomaly,
    #            b_eq = <b_f> + s (b_f - <b_f>)  (degree >= 1 scaled by s exactly).
    # Fermi: s ~ tau_flow/(tau_flow + tau_th) ~ 0.4 (code rheology) to ~0.02
    # (fine grains).
    # conv_flow_tau (default True since 2026-10-05): the same competition that
    # damps the anomaly to s relaxes it faster, 1/tau = 1/tau_th + 1/tau_flow,
    # i.e. tau_eff = s tau_th; False relaxes at the conductive rate (42318df).
    # ⛔ conv_fD is NOT a physical proxy on its own (Fermi F4): the conductive
    # control is a +-6 km cancellation of insolation and tidal terms, and
    # convection would damp the insolation (ocean-interface) term too. Kept
    # as a code option; any value != 1 prints a WARNING. Not reported.
    conv_s: float = 1.0
    conv_fD: float = 1.0
    conv_flow_tau: bool = True
    # Sub-Jovian meridian once the pole has moved: 'projection' = body x
    # projected into the equatorial plane (pre-2026-10-05); 'min_moment' =
    # the minimum-moment axis in the plane perpendicular to the spin
    # (quasi-static libration under the tidal torque; Fermi's option).
    meridian_mode: str = 'projection'
    # --- B&H interior coupling (coupled mode) ---
    interior_gamma: float = 28.0             # C_i / C_s
    interior_triaxiality: float = 1e-5       # (B_i-A_i)/C_i for rocky interior
    interior_K_G_factor: float = 0.5         # K_G / K_G* (<1 weak, >1 strong)


# ---------------------------------------------------------------------------
# Physics helpers
# ---------------------------------------------------------------------------
# The thickness law, the quadrature and the OSPW physics live in the
# ojakangas_stevenson package (shell_thickness, grid, polar_wander). The
# helpers below are thin adapters kept for this script's callers and for
# coupled_shell_simulation.py.

# Smallest principal-moment gap that counts as a real instability when
# seeding psi0, as a fraction of I_0: 1e-12 corresponds to ~0.1 mm of
# t20 + 2 t22 (Eq. 5.32: (t/tau) ~ 1e-5 per km). Below it the principal axes
# are set by round-off, not physics.
SEED_GAP_REL = 1e-12



def _build_dissipation_grid(case, theta, lam):
    """Dissipation integral D(theta, lam) and basal flux H for a Table I case."""
    rheology, H = table1_case(case)
    THETA, LAM = np.meshgrid(theta, lam, indexing='ij')
    return dissipation_field(rheology, THETA, LAM), H


def _equilibrium_thickness(T_s, D, H, method='eq28', rheology=None):
    """O&S Eq. 2.8 equilibrium shell thickness [m] (package implementation)."""
    return equilibrium_thickness(T_s, D, H, method=method, rheology=rheology)


def _compute_moments(b, theta, lam):
    """Degree-2 thickness harmonics of b(theta, lam) [m] (periodic quadrature)."""
    h = G.harmonics(b, theta, lam)
    t20_km, t22_km = h['t20'] / 1000, h['t22'] / 1000
    return {
        't20': h['t20'],
        't22': h['t22'],
        't20_km': t20_km,
        't22_km': t22_km,
        '4t22_km': 4 * t22_km,
        't20_2t22_km': t20_km + 2 * t22_km,
        'delta_BA_C': 1e-7 * 4 * t22_km,
        'delta_BC_A': 1e-7 * (t20_km + 2 * t22_km),
    }


def _periodic_lam(n_lam):
    return G.periodic_grid(2, n_lam)[1]


def _area_mean(field, theta, lam=None):
    """Area-weighted mean of a field on the periodic (theta, lam) grid."""
    if lam is None:
        lam = _periodic_lam(field.shape[1])
    return G.area_mean(field, theta, lam)


def _compute_inertia_tensor(b, theta, lam):
    """Geometric 3x3 inertia tensor of a thin shell b(theta, lam), I/(rho R^4) [m sr].

    Uncompensated (rigidly supported) load; see _physical_inertia_tensor for
    the Eq. 2.18 tensor that drives polar wander.
    """
    return G.thin_shell_inertia_tensor(b, theta, lam)


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


def _ospw_params(cfg, mean_b=None):
    """OSPWParams for this run. b_c = cfg.ospw_shell_thickness, or the current
    area-mean thickness when that is None."""
    bc = cfg.ospw_shell_thickness if cfg.ospw_shell_thickness else mean_b
    if bc is None:
        raise ValueError('OSPW shell thickness unknown: set ospw_shell_thickness '
                         'or pass the mean thickness')
    ratio = getattr(cfg, 'ospw_bc_over_d', None)
    d = bc / ratio if ratio else cfg.ospw_lithosphere_thickness
    return PW.OSPWParams(
        shell_thickness=bc,
        lithosphere_thickness=d,
        melting_viscosity=cfg.ospw_melting_viscosity,
        activation_l=cfg.ospw_activation_param,
        fracture_temp=cfg.ospw_fracture_base_temp,
        surface_temp=cfg.ospw_surface_temp,
    )


def _compute_ospw_epsilon_correction(cfg, mean_b=None):
    """Eq. 2.18 factor converting the geometric (uncompensated) inertia of a
    thickness anomaly to the bottom-loaded, buoyantly compensated value.

    = (drho/rho_c) [-(1 - C_w) + 4 (b_c/a)(1 - (drho/rho_m) C_w)], C_w from
    Eq. 2.11; ~0.0035 for b_c = 20 km. (Before 2026-10 this multiplied the
    iceberg term by rho_c/drho and used the membrane C_w: 14-21x too large.)
    """
    return PW.compensation_factor(_ospw_params(cfg, mean_b))


def _physical_inertia_tensor(b, theta, lam, cfg, p):
    """Deviatoric inertia tensor of the shell thickness field [kg m^2].

    With cfg.use_iceberg_effect (default) this is the Eq. 2.18
    bottom-loading tensor; otherwise the uncompensated ice-load tensor.
    """
    if cfg.use_iceberg_effect:
        return PW.shell_inertia_tensor(b, theta, lam, p)
    geo = G.thin_shell_inertia_tensor(b, theta, lam) * C.ICE_DENSITY * p.radius**4
    return G.deviatoric(geo)


def _fossil_state(cfg, p):
    """Body-fixed hydrostatic fossil (Eqs. 3.8-3.10): physical tensor and its
    thickness-equivalent (t20, t22) [m] (the t2m whose Eq. 2.18 C2m equals the
    fossil's). Zero when cfg.use_fossil_bulge is False."""
    if not cfg.use_fossil_bulge:
        return np.zeros((3, 3)), 0.0, 0.0
    c20, c22 = PW.fossil_c20_c22(p)
    per_m = PW.shell_c2m(1.0, p)
    return PW.fossil_inertia_tensor(p), c20 / per_m, c22 / per_m


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


def _tidal_coordinates(theta, lam, spin_pole, sub_jovian=None):
    """Convert body-frame (theta, lam) grid to tidal/spin-frame coordinates.

    The O&S tidal strain formulas use theta = colatitude from the spin axis
    and lam = longitude from the sub-Jovian meridian, both measured in the
    spin frame.  When the spin pole is not aligned with the body-frame z-axis
    (i.e. during TPW), the tidal coordinates differ from body-frame coordinates.

    ``sub_jovian``: body-frame direction of Jupiter (perpendicular to the
    spin; projected and normalized here). None = body x projected into the
    equatorial plane (meridian_mode 'projection').

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
    sub_j = (np.array([1.0, 0.0, 0.0]) if sub_jovian is None
             else np.asarray(sub_jovian, dtype=float))
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


_D_TABLES = {}


def _flow_law_D_table(case):
    """D(theta_t, lam_t) of a flow-law case on a closed 0.5-degree tidal-frame
    table (the flow-law time average is numerical and too slow to repeat every
    step). Cached per case and parameter set."""
    from scipy.interpolate import RegularGridInterpolator
    key = (case, C.active_parameter_set().name, C.C_R)
    if key not in _D_TABLES:
        rheology, _ = table1_case(case)
        th = np.linspace(0.0, np.pi, 361)
        lm = np.linspace(-np.pi, np.pi, 721)          # closed: includes both seams
        TH, LM = np.meshgrid(th, lm, indexing='ij')
        _D_TABLES[key] = RegularGridInterpolator(
            (th, lm), dissipation_field(rheology, TH, LM), method='linear')
    return _D_TABLES[key]


def _build_dissipation_for_spin_pole(case, theta, lam, spin_pole, sub_jovian=None):
    """Compute D(theta, lam) in body frame for given spin pole direction.

    Maps body-frame grid to tidal coordinates, evaluates tidal strain
    there, then returns D on the body-frame grid. Maxwell cases are evaluated
    analytically; flow-law cases are interpolated from a tidal-frame table.
    """
    theta_t, lam_t = _tidal_coordinates(theta, lam, spin_pole, sub_jovian)
    rheology, H = table1_case(case)
    if case.startswith('maxwell'):
        return dissipation_field(rheology, theta_t, lam_t), H
    pts = np.stack([theta_t.ravel(), lam_t.ravel()], axis=-1)
    return _flow_law_D_table(case)(pts).reshape(theta_t.shape), H


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
        self.theta, self.lam = G.periodic_grid(cfg.n_theta, cfg.n_lam)
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
    """Sweep darkening fraction 0 -> 1, compute equilibrium MoI at each step.

    The fossil bulge (cfg.use_fossil_bulge) is the body-fixed hydrostatic
    bulge of OSPW Eqs. 3.8-3.10. Totals ('t20_km', 't20_2t22_km', ...) add
    its thickness-equivalent harmonics to the shell's.
    """
    model = AlbedoDarkeningModel(cfg)
    D, H = _build_dissipation_grid(cfg.rheology_case, model.theta, model.lam)
    rheology, _ = table1_case(cfg.rheology_case)
    spin_pole = np.array([0.0, 0.0, 1.0])

    f_values = np.linspace(0, 1, cfg.n_darkening_steps)
    keys = ['f_dark', 'delta_BC_A', 'delta_BA_C', 'A_mean', 'b_avg_km',
            't20_km', 't22_km', 't20_2t22_km', '4t22_km',
            'shell_delta_BC_A', 'fossil_t20_km', 'fossil_t22_km',
            'misalignment_deg', 'I_max', 'I_mid', 'I_min']
    results = {k: np.zeros(len(f_values)) for k in keys}

    A0 = model.albedo_field(0.0)
    b0 = _equilibrium_thickness(model.surface_temperature(A0), D, H,
                                cfg.thickness_method, rheology)
    p = _ospw_params(cfg, _area_mean(b0, model.theta, model.lam))
    I_fossil, fossil_t20, fossil_t22 = _fossil_state(cfg, p)

    for i, f in enumerate(f_values):
        A = model.albedo_field(f, spin_pole=spin_pole) if cfg.use_full_inertia \
            else model.albedo_field(f)
        b_eq = _equilibrium_thickness(model.surface_temperature(A), D, H,
                                      cfg.thickness_method, rheology)
        mom = _compute_moments(b_eq, model.theta, model.lam)
        results['f_dark'][i] = f
        results['shell_delta_BC_A'][i] = mom['delta_BC_A']
        results['A_mean'][i] = _area_mean(A, model.theta, model.lam)
        results['b_avg_km'][i] = _area_mean(b_eq, model.theta, model.lam) / 1000
        results['fossil_t20_km'][i] = fossil_t20 / 1000
        results['fossil_t22_km'][i] = fossil_t22 / 1000
        tot_t20_km = (mom['t20'] + fossil_t20) / 1000
        tot_t22_km = (mom['t22'] + fossil_t22) / 1000
        results['t20_km'][i] = tot_t20_km
        results['t22_km'][i] = tot_t22_km
        results['t20_2t22_km'][i] = tot_t20_km + 2 * tot_t22_km
        results['4t22_km'][i] = 4 * tot_t22_km
        results['delta_BC_A'][i] = 1e-7 * (tot_t20_km + 2 * tot_t22_km)
        results['delta_BA_C'][i] = 1e-7 * 4 * tot_t22_km
        if cfg.use_full_inertia:
            I_total = _physical_inertia_tensor(b_eq, model.theta, model.lam, cfg, p) + I_fossil
            pa = _find_principal_axes(I_total, spin_pole)
            results['misalignment_deg'][i] = pa['misalignment_deg']
            results['I_min'][i], results['I_mid'][i], results['I_max'][i] = pa['eigenvalues']
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
        self.rheology, _ = table1_case(cfg.rheology_case)
        self.D, self.H = _build_dissipation_grid(
            cfg.rheology_case, self.theta, self.lam)

        # Initial state: uniform bright albedo, equilibrium thickness
        A_init = self.model.albedo_field(0.0)
        T_s_init = self.model.surface_temperature(A_init)
        self.b = self._b_eq(T_s_init)
        self.f_dark = 0.0
        self.t = 0.0

        # Spin pole direction in body frame (initially along z-hat)
        self.spin_pole = np.array([0.0, 0.0, 1.0])

        # Spin pole direction used for the current D grid. D is recomputed
        # whenever the pole has moved (threshold cfg.D_update_threshold_deg,
        # default 0). A 1-degree threshold (the pre-2026-10 value) held the
        # tidal heating body-fixed for small tilts while insolation and
        # darkening followed the pole; that inconsistency drove a spurious
        # ~3 Myr pole oscillation in slow-instability runs.
        self._D_spin_pole = self.spin_pole.copy()
        self._D_update_threshold_deg = cfg.D_update_threshold_deg

        # Sub-Jovian meridian in the body frame (cfg.meridian_mode). In
        # 'projection' mode it is only recorded; the tidal-frame helpers get
        # None and project body x themselves (the pre-2026-10-05 path).
        if cfg.meridian_mode not in ('projection', 'min_moment'):
            raise ValueError(f'meridian_mode must be projection or min_moment: '
                             f'{cfg.meridian_mode!r}')
        if cfg.meridian_mode == 'min_moment' and cfg.tpw_model != 'ospw':
            raise ValueError(f"meridian_mode='min_moment' needs tpw_model='ospw' "
                             f"(got {cfg.tpw_model!r}; the meridian is updated in the OSPW step)")
        if cfg.meridian_mode == 'min_moment' and not self._supports_meridian:
            raise ValueError(f"meridian_mode='min_moment' is not implemented for "
                             f"{type(self).__name__} (its albedo model ignores the meridian)")
        if getattr(cfg, 'conv_fD', 1.0) != 1.0:
            print(f'  WARNING: conv_fD = {cfg.conv_fD} reweights the tidal term alone; this is '
                  f'not a physical convection proxy (convection would damp the insolation '
                  f'term too; Fermi F4)')
        if cfg.seed_rearm not in ('any_axis', 'max_axis'):
            raise ValueError(f'seed_rearm must be any_axis or max_axis: {cfg.seed_rearm!r}')
        if cfg.tpw_model == 'coupled' and cfg.tidal_torque:
            print("  WARNING: tidal_torque is not implemented for tpw_model='coupled'; "
                  "that mode keeps the rotational torque only")
        self.sub_jovian = np.array([1.0, 0.0, 0.0])
        self._D_sub_jovian = self.sub_jovian.copy()
        self.meridian_jumps = []          # (t, angle_deg) of min_moment jumps > 45 deg

        # OSPW parameters (b_c = initial mean thickness unless fixed in cfg)
        self._ospw_p = _ospw_params(cfg, _area_mean(self.b, self.theta, self.lam))
        # Fossil bulge: BODY-FIXED hydrostatic figure x (1 - C_w), OSPW
        # Eqs. 3.8-3.10, frozen with x toward Jupiter and z along the initial
        # spin axis. It rotates with the shell; nothing relaxes it.
        self.I_fossil, self.fossil_t20, self.fossil_t22 = _fossil_state(cfg, self._ospw_p)

        # OSPW dynamics state
        self._ospw_Cf = PW.dissipation_cf(self._ospw_p)
        self._ospw_I0 = PW.moment_I0(self._ospw_p)
        self._ospw_n = C.MEAN_MOTION
        self._ospw_eps_correction = (PW.compensation_factor(self._ospw_p)
                                     if cfg.use_iceberg_effect else 1.0)

        # psi0 seed (OSPW: psi0 is the value of psi when t/tau passes through
        # zero). Armed while the pole sits on the max-moment axis; fires once
        # when that axis loses its status. See _apply_psi0_seed.
        self._seed_armed = True
        self._last_seed_pole = None
        self.seed_events = []

        # B&H interior coupling state
        if cfg.tpw_model == 'coupled':
            self.interior_pole = np.array([0.0, 0.0, 1.0])
            # K_G* = (3/2)(B_i-A_i) n^2 with (B_i-A_i) = eps_i C_i,
            # C_i = gamma_i I_0. Torque on the shell ~ K_G sin(2 delta).
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
            'I_max', 'I_mid', 'I_min', 't_over_tau']
        if cfg.tpw_model in ('ospw', 'coupled'):
            hist_keys.append('tpw_omega_deg_per_Myr')
        if cfg.tpw_model == 'coupled':
            hist_keys.extend(['shell_interior_sep_deg',
                              'interior_pole_theta', 'interior_pole_phi'])
        self.history = {k: [] for k in hist_keys}
        self.history['cycle_events'] = []
        self._record()

    # -- physics helpers ---------------------------------------------------
    _supports_meridian = False

    def _meridian_arg(self):
        """Sub-Jovian vector for the tidal-frame helpers (None = projection)."""
        return None if self.cfg.meridian_mode == 'projection' else self.sub_jovian

    def _meridian_fn(self, I_total):
        """meridian(s) used inside the tidal-torque integrator."""
        if self.cfg.meridian_mode == 'projection':
            return PW.projected_meridian
        ref = self.sub_jovian.copy()
        return lambda s: PW.meridian_min_moment(I_total, s, ref)

    def _update_meridian(self, I_total):
        if self.cfg.meridian_mode == 'min_moment':
            old = PW.projected_meridian(self.spin_pole, self.sub_jovian)
            self.sub_jovian = PW.meridian_min_moment(I_total, self.spin_pole,
                                                     self.sub_jovian)
            jump = np.degrees(np.arccos(np.clip(abs(old @ self.sub_jovian), 0, 1)))
            if jump > 45.0:
                # the in-plane moments swapped order: the new minimum axis is
                # ~90 deg away and its sub- vs anti-Jovian sign (hence which
                # hemisphere trails) is set by continuity alone
                self.meridian_jumps.append((self.t, float(jump)))
                print(f'  WARNING: min_moment meridian jumped {jump:.0f} deg at '
                      f't={self.t/1e6:.2f} Myr (in-plane moments swapped; trailing '
                      f'hemisphere chosen by sign continuity)')
        else:
            self.sub_jovian = PW.projected_meridian(self.spin_pole)

    def _b_eq(self, T_s):
        """O&S equilibrium thickness, with the convection proxy (conv_fD, conv_s).

        conv_s = conv_fD = 1 returns the plain O&S field (bit-identical path).
        """
        cfg = self.cfg
        b = _equilibrium_thickness(T_s, self.D, self.H, cfg.thickness_method, self.rheology)
        s_, fD = getattr(cfg, 'conv_s', 1.0), getattr(cfg, 'conv_fD', 1.0)
        if s_ == 1.0 and fD == 1.0:
            return b
        if not (0.0 < s_ <= 1.0 and 0.0 <= fD <= 1.0):
            raise ValueError(f'conv_s must be in (0, 1] and conv_fD in [0, 1]: {s_}, {fD}')
        if fD != 1.0:
            D_ref = np.full_like(self.D, _area_mean(self.D, self.theta, self.lam))
            b_ref = _equilibrium_thickness(T_s, D_ref, self.H, cfg.thickness_method,
                                           self.rheology)
            b = b_ref + fD * (b - b_ref)
        if s_ != 1.0:
            m = _area_mean(b, self.theta, self.lam)
            b = m + s_ * (b - m)
        return b

    def total_inertia_tensor(self):
        """Physical (Eq. 2.18) shell tensor + body-fixed fossil [kg m^2]."""
        return (_physical_inertia_tensor(self.b, self.theta, self.lam,
                                         self.cfg, self._ospw_p)
                + self.I_fossil)

    def _apply_psi0_seed(self, I_total):
        """Give the pole a physical offset psi0 when it becomes unstable.

        A pole exactly on a principal axis stays there under Eq. 5.39 (the
        pendulum balanced at the top); before 2026-10 the code waited for
        float round-off (~1e-12 rad in the eigenvectors) to grow, which set
        the "2.4 Myr" onset. OSPW take psi0 to be the angle when t/tau passes
        through zero. Here: when the axis the pole sits on stops being the
        max-moment axis (by more than SEED_GAP_REL * I_0), the pole is moved
        to angle psi0 from that axis toward the max axis and the event is
        logged. Nothing happens if it is already farther than psi0.

        cfg.seed_rearm = 'any_axis' (default): this applies whenever the pole
        sits within psi0 of ANY non-max principal axis, so a pole that
        converges onto a saddle (the mid axis) later in the run is kicked
        off it too. 'max_axis' (pre-2026-10-05): one kick, re-armed only
        while the pole sits on the max axis; a pole could then park on a
        saddle indefinitely (the flow n5.2 "settled at 90 deg" runs).
        """
        psi0 = self.cfg.psi0_rad
        vals, vecs = np.linalg.eigh(I_total)
        c = vecs.T @ self.spin_pole
        a = int(np.argmax(np.abs(c)))
        gap = vals[-1] - vals[a]
        if a == 2 or gap <= SEED_GAP_REL * self._ospw_I0:
            if a == 2:
                self._seed_armed = True
            return
        if psi0 <= 0:
            return
        psi = np.arccos(min(1.0, abs(c[a])))
        if self.cfg.seed_rearm == 'max_axis':
            # pre-2026-10-05: one kick per stay on the max axis. A pole that
            # later converged onto a saddle (mid axis) was never re-seeded.
            if not self._seed_armed:
                return
            self._seed_armed = False
            if psi >= psi0:
                return
        else:
            # 'any_axis': kick whenever the pole sits within psi0 of ANY
            # non-max principal axis, including a saddle reached later. Armed
            # on the max axis (above) or once the pole has moved more than
            # max(seed_rearm_deg, 5 psi0) from the last kick, so a shell that
            # re-centers on a kicked pole cannot turn repeated kicks into a
            # drift of psi0 per step.
            if not self._seed_armed:
                last = self._last_seed_pole
                lim = max(np.radians(self.cfg.seed_rearm_deg), 5.0 * psi0)
                if last is None or np.arccos(np.clip(abs(last @ self.spin_pole), -1, 1)) <= lim:
                    return
                self._seed_armed = True
            psi_acc = np.arctan2(np.linalg.norm(np.delete(c, a)), abs(c[a]))
            if psi_acc >= psi0 * (1.0 - 1e-6):
                return
            self._seed_armed = False
        e_a = np.sign(c[a]) * vecs[:, a]
        e_t = vecs[:, 2] * (np.sign(c[2]) if c[2] != 0 else 1.0)
        self.spin_pole = np.cos(psi0) * e_a + np.sin(psi0) * e_t
        self.spin_pole /= np.linalg.norm(self.spin_pole)
        self._last_seed_pole = self.spin_pole.copy()
        self.seed_events.append((self.t, psi, psi0, gap / self._ospw_I0))
        print(f'  psi0 seed at t={self.t/1e6:.3f} Myr: pole moved from {psi:.2e} '
              f'to {psi0:.2e} rad off the unstable axis '
              f'((I_max - I_spin)/I_0 = {gap/self._ospw_I0:.2e})')

    def _record(self):
        mom = _compute_moments(self.b, self.theta, self.lam)
        A = self.model.albedo_field(self.f_dark)
        self.history['t_years'].append(self.t)
        self.history['f_dark'].append(self.f_dark)
        self.history['A_mean'].append(_area_mean(A, self.theta, self.lam))
        self.history['b_avg_km'].append(_area_mean(self.b, self.theta, self.lam) / 1000)

        # Shell-only harmonics
        self.history['shell_delta_BC_A'].append(mom['delta_BC_A'])
        self.history['shell_t20_km'].append(mom['t20_km'])
        self.history['shell_t22_km'].append(mom['t22_km'])
        self.history['fossil_t20_km'].append(self.fossil_t20 / 1000)
        self.history['fossil_t22_km'].append(self.fossil_t22 / 1000)

        I_total = self.total_inertia_tensor()
        if self.cfg.use_full_inertia:
            pa = _find_principal_axes(I_total, self.spin_pole)
            self.history['misalignment_deg'].append(pa['misalignment_deg'])
            vals = pa['eigenvalues']
            self.history['I_min'].append(vals[0])
            self.history['I_mid'].append(vals[1])
            self.history['I_max'].append(vals[2])
            sp_theta = np.arccos(np.clip(self.spin_pole[2], -1, 1))
            sp_phi = np.arctan2(self.spin_pole[1], self.spin_pole[0])
            self.history['spin_pole_theta'].append(np.degrees(sp_theta))
            self.history['spin_pole_phi'].append(np.degrees(sp_phi))
        else:
            for k in ('misalignment_deg', 'I_min', 'I_mid', 'I_max',
                      'spin_pole_theta', 'spin_pole_phi'):
                self.history[k].append(0.0)
        # (t/tau) of the moment pair that drives Eq. 5.39: max-moment axis
        # minus the moment about the spin axis, over I_0.
        Is = self.spin_pole @ I_total @ self.spin_pole
        self.history['t_over_tau'].append(
            (np.linalg.eigvalsh(I_total)[-1] - Is) / self._ospw_I0)

        # Total (shell + fossil, thickness-equivalent) l=2 harmonics
        tot_t20_km = (mom['t20'] + self.fossil_t20) / 1000
        tot_t22_km = (mom['t22'] + self.fossil_t22) / 1000
        self.history['total_t20_km'].append(tot_t20_km)
        self.history['total_t22_km'].append(tot_t22_km)
        self.history['delta_BC_A'].append(1e-7 * (tot_t20_km + 2 * tot_t22_km))
        self.history['delta_BA_C'].append(1e-7 * 4 * tot_t22_km)
        self.history['t20_2t22_km'].append(tot_t20_km + 2 * tot_t22_km)
        self.history['4t22_km'].append(4 * tot_t22_km)

        if self.cfg.tpw_model in ('ospw', 'coupled'):
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
        must be recalculated when the spin pole moves; it is updated whenever
        the pole has moved by more than _D_update_threshold_deg (default 0:
        every step on which the pole moved at all).
        """
        if not self.cfg.use_full_inertia:
            return
        cos_a = np.clip(np.dot(self.spin_pole, self._D_spin_pole), -1, 1)
        angle = np.degrees(np.arccos(cos_a))
        mer = self._meridian_arg()
        mer_moved = mer is not None and not np.array_equal(mer, self._D_sub_jovian)
        if (angle > self._D_update_threshold_deg and angle > 0.0) or mer_moved:
            self.D, self.H = _build_dissipation_for_spin_pole(
                self.cfg.rheology_case, self.theta, self.lam, self.spin_pole, mer)
            self._D_spin_pole = self.spin_pole.copy()
            if mer is not None:
                self._D_sub_jovian = mer.copy()

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
        b_eq = self._b_eq(T_s)
        # 4. Exponential relaxation of shell toward equilibrium
        self.b = b_eq + (self.b - b_eq) * np.exp(-dt / cfg.tau_shell)
        # 5. The fossil bulge is body-fixed and constant (Eqs. 3.8-3.10).
        # 6. Polar wander (full inertia path)
        if cfg.use_full_inertia:
            self._reorient(dt, self.total_inertia_tensor())
        self.t += dt

    def _reorient(self, dt, I_total):
        """Move the spin pole for one step under the chosen TPW model."""
        cfg = self.cfg
        if cfg.tpw_model == 'ospw':
            self._apply_psi0_seed(I_total)
            self._ospw_step(dt, I_total)
            return
        if cfg.tpw_model == 'coupled':
            self._apply_psi0_seed(I_total)
            self._coupled_step(dt, I_total)
            return
        if cfg.tpw_model != 'exponential':
            raise ValueError(f'Unknown tpw_model: {cfg.tpw_model}')
        # exponential: free-parameter relaxation toward v_max on tau_tpw
        pa = _find_principal_axes(I_total, self.spin_pole)
        misalign_rad = np.radians(pa['misalignment_deg'])
        if misalign_rad <= 1e-6:
            return
        v_max = pa['v_max']
        v_perp = v_max - np.dot(v_max, self.spin_pole) * self.spin_pole
        v_perp_norm = np.linalg.norm(v_perp)
        if v_perp_norm <= 1e-10:
            return
        d_angle = misalign_rad * (1.0 - np.exp(-dt / cfg.tau_tpw))
        self.spin_pole = (self.spin_pole * np.cos(d_angle)
                          + v_perp / v_perp_norm * np.sin(d_angle))
        self.spin_pole /= np.linalg.norm(self.spin_pole)

    # ------------------------------------------------------------------
    #  TPW dynamics methods
    # ------------------------------------------------------------------

    def _ospw_step(self, dt, I_total):
        """OSPW overdamped reorientation, Eq. 5.39 (exact for the frozen tensor).

        psi_dot = (t/tau)(n^2 I_0 / 2 C_f) sin 2psi, (t/tau) I_0 = I_target - I_spin,
        in vector form s_dot = (n^2/C_f)[I s - (s.I s) s]; see
        ojakangas_stevenson.polar_wander.overdamped_step. I_total is the
        physical (Eq. 2.18 + fossil) tensor, so the rate needs no epsilon
        correction. Returns the rotation angle [rad].
        """
        dt_s = dt * 365.25 * 86400.0
        old = self.spin_pole.copy()
        if self.cfg.tidal_torque:
            self.spin_pole, info = PW.overdamped_step_tidal(
                self.spin_pole, I_total, self._ospw_Cf, dt_s, self._meridian_fn(I_total))
        else:
            self.spin_pole, info = PW.overdamped_step(self.spin_pole, I_total,
                                                      self._ospw_Cf, dt_s)
        self._update_meridian(I_total)
        d_angle = np.arccos(np.clip(np.dot(old, self.spin_pole), -1, 1))
        self._last_omega_deg_per_Myr = np.degrees(info['psi_dot']) * 1e6 * 365.25 * 86400
        return d_angle

    def _coupled_step(self, dt, I_total):
        """OSPW dissipation + B&H gravity-gradient shell-interior coupling.

        Overdamped: C_f s_dot = (driving torque) + (coupling torque), with
          driving   n^2 [I s - (s.I s) s]           (Eq. 5.39, vector form)
          coupling  K_G sin(2 delta) toward the interior pole.
        The coupling-torque normalization (K_G = factor x (3/2)(B_i - A_i) n^2)
        is NOT verified against B&H 2023, which treats a single axis only.
        Explicit step, capped at the current misalignment.
        """
        n = self._ospw_n
        Cf = self._ospw_Cf
        dt_s = dt * 365.25 * 86400.0
        s = self.spin_pole
        Is = s @ I_total @ s
        d_spin = (n**2 / Cf) * (I_total @ s - Is * s) * dt_s
        cos_sep = np.clip(np.dot(s, self.interior_pole), -1, 1)
        sep_rad = np.arccos(cos_sep)
        v_couple = self.interior_pole - cos_sep * s
        v_couple_norm = np.linalg.norm(v_couple)
        if sep_rad > 1e-10 and v_couple_norm > 1e-10:
            d_spin = d_spin + (self._K_G * np.sin(2.0 * sep_rad) / Cf) * dt_s \
                * v_couple / v_couple_norm
        pa = _find_principal_axes(I_total, s)
        misalign_rad = np.radians(pa['misalignment_deg'])
        d_magnitude = np.linalg.norm(d_spin)
        if d_magnitude > misalign_rad and misalign_rad > 1e-10:
            d_spin *= misalign_rad / d_magnitude
            d_magnitude = misalign_rad
        if d_magnitude > 1e-15:
            self.spin_pole = s + d_spin
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
        """l=2 harmonic stability check with discrete 90-degree reorientation.

        Unstable when the total (shell + fossil, thickness-equivalent)
        t20 + 2 t22 > 0 by more than cfg.reorient_threshold in Delta(B-C)/A.
        On a 90-degree rotation about the sub-Jovian axis the body-fixed fossil
        rotates with the shell (_rotate_l2); the shell is reset to the new
        equilibrium.
        """
        cfg = self.cfg
        mom = _compute_moments(self.b, self.theta, self.lam)
        tot_t20_km = (mom['t20'] + self.fossil_t20) / 1000
        tot_t22_km = (mom['t22'] + self.fossil_t22) / 1000
        delta_BC_A = 1e-7 * (tot_t20_km + 2 * tot_t22_km)
        if delta_BC_A <= cfg.reorient_threshold:
            return False
        n = len(self.history['cycle_events']) + 1
        self.history['cycle_events'].append((self.t, n))
        self.f_dark *= (1.0 - cfg.reorient_albedo_reset)
        A_new = self.model.albedo_field(self.f_dark)
        self.b = self._b_eq(self.model.surface_temperature(A_new))
        if cfg.use_fossil_bulge:
            self.fossil_t20, self.fossil_t22 = _rotate_l2(self.fossil_t20, self.fossil_t22)
            mom_new = _compute_moments(self.b, self.theta, self.lam)
            post = 1e-7 * ((mom_new['t20'] + self.fossil_t20)
                           + 2 * (mom_new['t22'] + self.fossil_t22)) / 1000
            if post > cfg.reorient_threshold:
                print(f'  WARNING: post-rotation still unstable '
                      f'(Delta(B-C)/A = {post:.2e})')
        return True

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
                n_theta=91, n_lam=180,
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
    fb_tag = ' (hydrostatic fossil)' if cfg.use_fossil_bulge else ' (no fossil)'
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
        fb_tag = ' (hydrostatic fossil)'
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


def fig5_all_rheologies(save_path=None, A_bright=0.55, A_dark=0.3814,
                        use_fossil_bulge=True, use_full_inertia=True):
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
                use_fossil_bulge=True, use_full_inertia=use_full_inertia)
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
        ax_tr.set_title('With hydrostatic fossil bulge')

    if show_tensor:
        for ax in [ax_bl, ax_br]:
            ax.set_xlabel('Darkening fraction $f$')
            ax.set_ylabel('Misalignment angle [deg]')
            ax.legend(fontsize=8.5, loc='best')
            ax.grid(True, alpha=0.2)
        ax_bl.set_title('Misalignment (no fossil)')
        ax_br.set_title('Misalignment (hydrostatic fossil)')

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
    parser.add_argument('--A-dark', type=float, default=0.3814,
                        help='dark endmember Bond albedo (Hapke non-ice endmember)')
    parser.add_argument('--n-cycles', type=int, default=5)
    parser.add_argument('--sweep-only', action='store_true',
                        help='Only run quasi-static sweep (fast)')
    parser.add_argument('--survey', action='store_true',
                        help='Run tau_dark x tau_shell parameter survey')
    parser.add_argument('--all-rheologies', action='store_true',
                        help='Quasi-static sweep for all 4 rheologies')
    parser.add_argument('--no-fossil-bulge', action='store_true',
                        help='Disable fossil bulge (original behavior)')
    parser.add_argument('--psi0', type=float, default=PSI0_OBLIQUITY_RAD,
                        help='psi0 seed [rad] applied when the pole becomes unstable')
    parser.add_argument('--thickness-method', choices=('eq28', 'exact'), default='eq28')
    parser.add_argument('--parameter-set', default='paper', choices=sorted(C.PARAMETER_SETS),
                        help="constants: 'paper' (validation) or 'modern' (science)")
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
    parser.add_argument('--ospw-Tf', type=float, default=124.0,
                        help='Fracture base temperature [K] (default: 124, OSPW p. 264)')
    parser.add_argument('--ospw-eta0', type=float, default=3e14,
                        help='Melting viscosity [Pa s] (default: 3e14)')
    parser.add_argument('--ospw-l', type=float, default=15.0,
                        help='Arrhenius activation parameter (default: 15)')
    parser.add_argument('--ospw-bc', type=float, default=None,
                        help='Shell thickness b_c [km] (default: initial mean b)')
    parser.add_argument('--ospw-d-lith', type=float, default=2.0,
                        help='Lithosphere thickness [km] (default: 2)')
    # Iceberg effect
    parser.add_argument('--no-iceberg', action='store_true',
                        help='Use the rigid (uncompensated) load instead of Eq. 2.18')
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
    C.use_parameter_set(args.parameter_set)
    print(C.describe_active())

    use_full = not args.no_full_inertia
    if args.dt is not None:
        dt = args.dt
    elif use_full:
        if args.tpw_model == 'exponential':
            dt = args.tau_tpw / 10.0
        else:
            # OSPW/coupled: adapt dt to the slower τ_f timescale
            tau_f_yr = PW.tau_f(PW.OSPWParams(
                shell_thickness=(args.ospw_bc or 20.0) * 1e3,
                lithosphere_thickness=args.ospw_d_lith * 1e3,
                fracture_temp=args.ospw_Tf, melting_viscosity=args.ospw_eta0,
                activation_l=args.ospw_l))
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
        psi0_rad=args.psi0,
        thickness_method=args.thickness_method,
        use_full_inertia=use_full,
        misalignment_threshold_deg=args.misalignment_threshold,
        tau_tpw=args.tau_tpw,
        tpw_model=args.tpw_model,
        ospw_shell_thickness=args.ospw_bc * 1e3 if args.ospw_bc else None,
        ospw_lithosphere_thickness=args.ospw_d_lith * 1e3,
        ospw_melting_viscosity=args.ospw_eta0,
        ospw_activation_param=args.ospw_l,
        ospw_fracture_base_temp=args.ospw_Tf,
        use_iceberg_effect=not args.no_iceberg,
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
    print(f'Fossil bulge: {fb_str} (body-fixed hydrostatic, OSPW Eqs. 3.8-3.10)')
    print(f'Inertia model: {tensor_str}')
    if cfg.use_full_inertia:
        print(f'  misalignment threshold={cfg.misalignment_threshold_deg} deg, '
              f'psi0={cfg.psi0_rad:.1e} rad')
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
                            use_full_inertia=cfg.use_full_inertia)

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
            tau_f_yr = PW.tau_f(evo._ospw_p)
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
