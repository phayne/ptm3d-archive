#!/usr/bin/env python3
"""Coupled dynamic simulation of Europa ice-shell evolution + true polar wander.

Closes the loop:  trailing-hemisphere darkening -> lower albedo -> surface
temperature (PTM3D-coupled surrogate) -> O&S equilibrium thickness -> shell
relaxes toward it at the LOCAL thermal-diffusion timescale tau_diff(theta,lam)
-> full 3x3 inertia tensor -> OSPW torque reorients the spin pole -> darkening
pattern rotates with the new pole -> repeat.

Built on top of `polar_wander_cycle.py` (subclasses `ShellEvolution`), reusing
its inertia-tensor / OSPW / reorientation / history machinery unchanged.

Temperature surrogate (multi-rate PTM3D coupling):
    T(theta,lam) = [(1-A)*F_A(theta_tidal)/sigma]^(1/4) + dT_PTM(theta_t, lam_t)
where dT_PTM = (PTM3D uniform-albedo - O&S analytical) is the eclipse +
thermal-inertia correction, held fixed in the tidal frame and rotated with the
spin pole.  In Phase 1 dT_PTM is STATIC (from the existing europa_ico6_uniform-ice
run); Phase 2 adds periodic full-PTM3D refresh (PTM3DRunner).

Physics choices (project decisions of 2026-10-04, after the 2026-10-03 review):
  * Surface-T statistic: the shell's top boundary is the ARITHMETIC time-mean
    surface temperature <T> (not (<T^4>)^(1/4)); dT_PTM is built from it.
  * Thickness: O&S Eq. 2.8 with C_R = 0.5 (ojakangas_stevenson.shell_thickness).
  * Darkening: A_bright -> A_dark = 0.3814 (Hapke non-ice endmember);
    tau_dark is free (--tau-dark), tested from 1e4 to 1e7 yr (the B&H 2021
    Pwyll bound is >~1 Myr).
  * Polar wander: OSPW Eq. 5.39, overdamped, driven by the PHYSICAL (Eq. 2.18
    bottom-loaded) inertia tensor plus an optional body-fixed hydrostatic
    fossil (Eqs. 3.8-3.10). The pole is seeded with a physical offset psi0
    (--psi0) when it becomes unstable, not left to float round-off.
  * Relaxation (2026-10-05): Stefan-limited, tau = rho L b / q per location
    (relaxation_mode='stefan', default; ojakangas_stevenson.polar_wander.
    stefan_relaxation_time). 'diffusive' keeps the earlier b^2/kappa
    (_tau_diff_years), which is 4-5x too fast.
  * Split albedo (albedo_mode='split', 2026-10-05): A = A_intr(body) +
    A_exo(tidal frame, relaxing on tau_exo_yr); template and anchoring in
    ojakangas_stevenson.exogenic_albedo.

Run (needs scipy/matplotlib -- use europa_py39; europa_viz for the animation):
    PY=/Users/paha3326/anaconda3/envs/europa_py39/bin/python
    $PY scripts/coupled_shell_simulation.py --smoke
    $PY scripts/coupled_shell_simulation.py --tau-dark 1e4
    $PY scripts/coupled_shell_simulation.py --tau-dark 1e5
"""
import os
import sys
import json
import argparse
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'python'))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

from ojakangas_stevenson import constants as C
from ojakangas_stevenson.surface_temperature import OSSurfaceTemperature

import polar_wander_cycle as pwc
from polar_wander_cycle import (
    PolarWanderConfig, ShellEvolution, AlbedoDarkeningModel,
    _tidal_coordinates, _area_mean, _ospw_params, _fossil_state,
)
from ojakangas_stevenson import polar_wander as PW
from ojakangas_stevenson import exogenic_albedo as EA
from ojakangas_stevenson import grid as G

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PLOTDIR = os.path.join(REPO, 'plots', 'coupled_shell')
DT_PTM_CACHE = os.path.join(PLOTDIR, 'dT_ptm_static.npz')
# The pre-2026-10 cache: a bare array on the legacy grid
# linspace(0.001, pi-0.001, 361) x linspace(-pi, pi, 721) (both seam columns).
DT_PTM_CACHE_LEGACY = os.path.join(PLOTDIR, 'dT_ptm_static.npy')


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class CoupledConfig(PolarWanderConfig):
    # --- darkening (firm user decisions) ---
    A_bright: float = 0.5
    A_dark: float = 0.3814              # Hapke non-ice endmember (decision of 2026-10-04)
    darkening_power: float = 1.0
    darkening_mode: str = 'cosine'      # 'cosine' (apex spot) or 'sin' (regression)
    tau_dark: float = 1e4               # free; tested 1e4-1e7 yr
    # --- initial condition ---
    use_fossil_bulge: bool = False      # True: body-fixed hydrostatic fossil (Eqs. 3.8-3.10)
    # 'equilibrium_mean', 'equilibrium' (for the t = 0 albedo), 'constant',
    # 'uniform_equilibrium' (for the uniform albedo A_unif of forcing_file,
    # i.e. the forcing is switched on at t = 0), or 'blend' =
    # (1 - init_blend_frac) * uniform_equilibrium + init_blend_frac * equilibrium.
    init_b_mode: str = 'equilibrium_mean'
    init_b_const_m: float = 20e3
    init_blend_frac: float = 0.5
    # Random-phase degree-2 thickness perturbation added to the initial
    # shell: RMS drawn uniformly in init_l2_perturb_km (lo, hi), direction in
    # the 5-D real Y2m space from init_perturb_seed. (0, 0) = none.
    init_l2_perturb_km: tuple = (0.0, 0.0)
    init_perturb_seed: int = 0
    # Pole displaced at t = 0 (after the shell is built for pole = body z) by
    # this angle, toward azimuth init_pole_azimuth_deg (from body x toward y).
    init_pole_offset_deg: float = 0.0
    init_pole_azimuth_deg: float = 0.0
    # --- reorientation ---
    tpw_model: str = 'ospw'
    # --- per-location thermal diffusion ---
    cp_mode: str = 'ice'                # 'ice': cp = 90 + 7.49 T (heat1d planets.py); 'constant'
    cp_const: float = 2000.0            # used only with cp_mode='constant'
    tau_diff_floor_yr: float = 1e3
    tau_diff_cap_yr: float = 5e7
    # 'stefan_eigen' (default, Fermi F6): tau = f rho L b / q, f = exact slowest
    # eigenmode of the conductive Stefan shell (~1.40 at 100 K), per location;
    # 'stefan_sensible': f = 1 + S/L (exact S/L; an UPPER estimate, ~8% slow, 42318df);
    # 'stefan': f = 1 (latent only, 679af84); 'diffusive': b^2/kappa (pre-2026-10-05).
    relaxation_mode: str = 'stefan_eigen'
    # Multiplies the relaxation time before the floor/cap clip (sensitivity tests,
    # e.g. 2.0 for "2 tau"; 1.0 = unchanged). Frozen relaxation: set
    # tau_diff_floor_yr = tau_diff_cap_yr = 1e12 (the clip then binds and warns).
    relaxation_scale: float = 1.0
    # --- PTM3D surrogate / multi-rate ---
    use_ptm3d_surrogate: bool = True
    use_live_ptm3d: bool = False        # Phase 2: periodic full-PTM3D refresh
    # --- forcing file (eq36-derived; see tpw_runs/build_forcing.py) ---
    # npz with dT (tidal-frame table), A_meas / A_unif and R_meas / R_unif on
    # the run grid. When set it replaces the legacy dT_PTM cache.
    forcing_file: str = None
    # 'darkening': spin-frame apex darkening (A_bright -> A_dark);
    # 'static_map': body-fixed albedo field from forcing_file ('measured' or
    # 'uniform'), constant in time, carried with the shell as the pole moves.
    # 'split': A = A_intr(body) + A_exo(tidal frame), see split_* below.
    albedo_mode: str = 'darkening'
    static_field: str = 'measured'
    use_residual: bool = True           # add the body-fixed residual R (static_map)
    # --- split albedo (albedo_mode='split') ---
    # 'full' (A_intr + A_exo), 'exogenic' (A_intr = uniform A_unif),
    # 'intrinsic' (A_exo = 0; the control).
    split_component: str = 'full'
    tau_exo_yr: float = 1e6             # 0 = instantaneous tidal frame; inf = body-fixed
    exo_template: str = 'bh2021'        # see exogenic_albedo.TEMPLATE_FORMS, or 'exo_file'
    # exo_template='exo_file': exosphere-derived template (exogenic_albedo.exo_profile_fit)
    exo_forcing_file: str = None
    exo_case: str = 'high_1M'
    exo_variant: str = 'E2'
    exo_scale_zonal: float = 1.0
    exo_scale_odd: float = 1.0
    exo_sym_S: bool = None             # None: symmetrize S for E3map only
    # exo_variant='E4': E1 + kappa x dose-darkening field (exogenic_albedo.add_dose)
    exo_dose_file: str = None
    exo_dose_scenario: str = 'EN_1mm'
    exo_dose_kappa: float = None
    # exo_template='demo': simple demonstration forcing (exogenic_albedo.demo_fit)
    exo_demo: dict = None              # {'a1':, 'a_b':, 'a22':, 'shape':, 'phi_b':, 'delta':}
    albedo_clamp: tuple = (0.3814, 0.80)  # Hapke non-ice endmember .. ice; None = off
    ptm3d_resolution: int = 5
    ptm3d_refresh_pole_deg: float = 10.0
    ptm3d_refresh_albedo_rms: float = 0.03
    ptm3d_max_calls: int = 40
    # --- grid & stepping (coarser than parent: only low-l harmonics matter) ---
    n_theta: int = 91
    n_lam: int = 180                    # periodic grid (even, no seam column)
    t_max_years: float = 1e8
    dt_years: float = None              # fine-phase dt; auto = tau_dark/20
    n_coarse_steps: int = 3000          # coarse-phase step budget (post-darkening)
    reorient_dtheta_max_deg: float = 3.0  # cap pole rotation per step (resolve TPW)
    n_snapshots: int = 60
    n_history: int = 2000


# ---------------------------------------------------------------------------
# Cosine-apex darkening
# ---------------------------------------------------------------------------
def _darkening_weight_cosine_apex(theta, lam, spin_pole, power=1.0, sub_jovian=None):
    """Apex-centered darkening weight: w = max(0, cos gamma)^power, where gamma
    is the angular distance from the trailing-hemisphere apex.

    In the spin/tidal frame the apex (trailing equator point) is the unit
    vector (0,1,0); for a general point (theta_t, lam_t) the cosine of the
    angular distance is cos(gamma) = sin(theta_t) * sin(lam_t).
    """
    theta_t, lam_t = _tidal_coordinates(theta, lam, spin_pole, sub_jovian)
    cos_gamma = np.sin(theta_t) * np.sin(lam_t)
    return np.maximum(0.0, cos_gamma) ** power


class CosineApexAlbedoModel(AlbedoDarkeningModel):
    """Darkening with an apex-centered cosine spot (or the parent sin form)."""
    meridian = None                     # sub-Jovian vector (None = projection)

    def albedo_field(self, f_dark, spin_pole=None):
        cfg = self.cfg
        if getattr(cfg, 'darkening_mode', 'cosine') == 'sin':
            if self.meridian is not None:
                raise ValueError("darkening_mode='sin' ignores the meridian; "
                                 "use meridian_mode='projection'")
            return super().albedo_field(f_dark, spin_pole)
        if spin_pole is None:
            spin_pole = np.array([0.0, 0.0, 1.0])
        w = _darkening_weight_cosine_apex(
            self.theta, self.lam, spin_pole, cfg.darkening_power, self.meridian)
        return cfg.A_bright + (cfg.A_dark - cfg.A_bright) * f_dark * w


# ---------------------------------------------------------------------------
# Static PTM3D-vs-O&S correction field (the surrogate)
# ---------------------------------------------------------------------------
def build_static_dT_ptm(cache=True):
    """dT_PTM(theta,lam) = T_ptm(uniform A=0.5) - T_os(analytical), in the tidal frame.

    T_ptm is the arithmetic time mean (eclipse_vs_albedo_analysis.gridded_temperature).
    Returns (theta1d, lam1d, dT) with lam1d CLOSED (it includes both -pi and
    +pi) so that a RegularGridInterpolator is periodic-safe. Cached as .npz
    with its own coordinates. A legacy .npy cache (pre-2026-10 grid) is read
    with its legacy coordinates and reported, never silently reinterpreted.
    """
    import eclipse_vs_albedo_analysis as eva
    if cache and os.path.exists(DT_PTM_CACHE):
        d = np.load(DT_PTM_CACHE)
        return d['theta'], d['lam'], d['dT']
    if cache and os.path.exists(DT_PTM_CACHE_LEGACY):
        dT = np.load(DT_PTM_CACHE_LEGACY)
        th = np.linspace(0.001, np.pi - 0.001, dT.shape[0])
        lm = np.linspace(-np.pi, np.pi, dT.shape[1])
        print(f'  WARNING: using LEGACY dT_PTM cache {DT_PTM_CACHE_LEGACY} '
              f'({dT.shape}, built 2026-06 from the March-2026 uniform-ice run); '
              f'delete it to rebuild from current PTM3D output')
        return th, lm, dT
    T_ptm = eva.gridded_temperature(eva.RUNS['no_darkening'])
    T_os = OSSurfaceTemperature(albedo=eva.ALBEDO).temperature_2d(eva.THETA1D, eva.LAM1D)
    dT = T_ptm - T_os
    th, lm = eva.THETA1D, np.append(eva.LAM1D, np.pi)
    dT = np.column_stack([dT, dT[:, 0]])            # close the seam
    if cache:
        os.makedirs(PLOTDIR, exist_ok=True)
        np.savez(DT_PTM_CACHE, theta=th, lam=lm, dT=dT)
    return th, lm, dT


# ---------------------------------------------------------------------------
# Coupled evolution
# ---------------------------------------------------------------------------
class StaticAlbedoModel(AlbedoDarkeningModel):
    """Body-fixed albedo field A(theta, lam), independent of time and pole."""

    def __init__(self, cfg, A_grid):
        super().__init__(cfg)
        self.A_grid = np.asarray(A_grid, dtype=float)

    def albedo_field(self, f_dark, spin_pole=None):
        return self.A_grid


class SplitAlbedoModel(AlbedoDarkeningModel):
    """A = clamp(A_intr(body) + A_exo), A_exo relaxing to a tidal-frame template.

    dA_exo/dt = [A_exo^eq(tidal coordinates) - A_exo] / tau_exo, integrated
    exactly per step for the pole and meridian at the start of the step.
    tau_exo = 0: A_exo = A_exo^eq every step; tau_exo = inf: A_exo frozen in
    the body at its t = 0 value. fit = None: no exogenic part.
    """
    meridian = None

    def __init__(self, cfg, A_intr, fit):
        super().__init__(cfg)
        self.A_intr = np.broadcast_to(np.asarray(A_intr, dtype=float),
                                      (len(self.theta), len(self.lam))).copy()
        self.fit = fit
        self.A_exo = self.equilibrium_exo(np.array([0.0, 0.0, 1.0]))
        self.clamped_fraction = 0.0

    def equilibrium_exo(self, spin_pole):
        if self.fit is None:
            return np.zeros_like(self.A_intr)
        th_t, lm_t = _tidal_coordinates(self.theta, self.lam, spin_pole, self.meridian)
        return EA.template_field(self.fit, th_t, lm_t)

    def advance(self, dt, spin_pole):
        tau = self.cfg.tau_exo_yr
        if self.fit is None or np.isinf(tau):
            return
        eq = self.equilibrium_exo(spin_pole)
        if tau <= 0:
            self.A_exo = eq
        else:
            self.A_exo = eq + (self.A_exo - eq) * np.exp(-dt / tau)

    def albedo_field(self, f_dark, spin_pole=None):
        A = self.A_intr + self.A_exo
        clamp = self.cfg.albedo_clamp
        if clamp is None:
            return A
        out = np.clip(A, clamp[0], clamp[1])
        self.clamped_fraction = float(np.mean(out != A))
        return out


def _random_l2_field(theta, lam, seed, rms_range):
    """Random-phase degree-2 field with RMS uniform in rms_range [m].

    Real Y2m basis normalized to unit RMS on the sphere:
    sqrt(5) P2(z), sqrt(15) xz, sqrt(15) yz, sqrt(15) xy, sqrt(15)(x^2-y^2)/2.
    Returns (field, rms, unit coefficient vector).
    """
    rng = np.random.default_rng(seed)
    u = rng.normal(size=5)
    u /= np.linalg.norm(u)
    rms = rng.uniform(*rms_range)
    TH, LA = np.meshgrid(theta, lam, indexing='ij')
    x, y, z = np.sin(TH) * np.cos(LA), np.sin(TH) * np.sin(LA), np.cos(TH)
    f = [np.sqrt(5.0) * (1.5 * z * z - 0.5), np.sqrt(15.0) * x * z, np.sqrt(15.0) * y * z,
         np.sqrt(15.0) * x * y, np.sqrt(15.0) * 0.5 * (x * x - y * y)]
    return rms * sum(c * fi for c, fi in zip(u, f)), rms, u


def load_forcing(cfg, theta, lam):
    """Read cfg.forcing_file and check it was built on this run's grid."""
    d = np.load(cfg.forcing_file)
    if not (d['theta'].shape == theta.shape and np.allclose(d['theta'], theta)
            and d['lam'].shape == lam.shape and np.allclose(d['lam'], lam)):
        raise ValueError(f'{cfg.forcing_file} was built on a different grid than '
                         f'{len(theta)}x{len(lam)}')
    return d


class CoupledEvolution(ShellEvolution):
    """ShellEvolution with PTM3D-coupled temperature, per-location tau_diff
    relaxation, cosine-apex darkening, and a uniform / no-fossil initial state.
    """

    _supports_meridian = True

    def __init__(self, cfg):
        # Parent sets up grid, D (spin pole = z), OSPW, history, and records.
        super().__init__(cfg)
        self._pole0 = self.spin_pole.copy()
        self._R = None
        self._dT_interp = None
        self._warned_tau_clip = False
        self.exo_fit = None
        self.init_perturbation = None
        forcing = load_forcing(cfg, self.theta, self.lam) if cfg.forcing_file else None
        self._A_unif = float(forcing['A_unif']) if forcing is not None else cfg.A_bright
        if cfg.albedo_mode == 'static_map':
            if forcing is None:
                raise ValueError("albedo_mode='static_map' needs forcing_file")
            if cfg.static_field == 'measured':
                A_grid, R = forcing['A_meas'], forcing['R_meas']
            elif cfg.static_field == 'uniform':
                A_grid = np.full((len(self.theta), len(self.lam)), float(forcing['A_unif']))
                R = forcing['R_unif']
            else:
                raise ValueError(f'static_field must be measured or uniform: {cfg.static_field!r}')
            self.model = StaticAlbedoModel(cfg, A_grid)
            self._R = np.array(R) if cfg.use_residual else None
        elif cfg.albedo_mode == 'darkening':
            # Swap in the cosine-apex darkening model.
            self.model = CosineApexAlbedoModel(cfg)
        elif cfg.albedo_mode == 'split':
            if forcing is None:
                raise ValueError("albedo_mode='split' needs forcing_file")
            A_meas = np.asarray(forcing['A_meas'], dtype=float)
            if cfg.exo_template == 'exo_file':
                if not cfg.exo_forcing_file:
                    raise ValueError("exo_template='exo_file' needs exo_forcing_file")
                self.exo_fit = EA.exo_profile_fit(
                    cfg.exo_forcing_file, cfg.exo_case, cfg.exo_variant, A_meas,
                    self.theta, self.lam, cfg.exo_scale_zonal, cfg.exo_scale_odd,
                    cfg.exo_sym_S)
                if cfg.exo_variant == 'E4':
                    if not cfg.exo_dose_file or cfg.exo_dose_kappa is None:
                        raise ValueError("exo_variant='E4' needs exo_dose_file and exo_dose_kappa")
                    self.exo_fit = EA.add_dose(self.exo_fit, cfg.exo_dose_file,
                                               cfg.exo_dose_scenario, cfg.exo_dose_kappa)
            elif cfg.exo_template == 'demo':
                if cfg.split_component != 'exogenic':
                    raise ValueError("exo_template='demo' is defined for split_component='exogenic'")
                self.exo_fit = EA.demo_fit(self.theta, self.lam, **(cfg.exo_demo or {}))
            else:
                self.exo_fit = EA.fit_template(A_meas, self.theta, self.lam, cfg.exo_template)
            _, A_intr = EA.split_map(A_meas, self.theta, self.lam, self.exo_fit)
            comp = cfg.split_component
            if comp == 'full':
                self.model = SplitAlbedoModel(cfg, A_intr, self.exo_fit)
            elif comp == 'exogenic':
                self.model = SplitAlbedoModel(cfg, self._A_unif, self.exo_fit)
            elif comp == 'intrinsic':
                self.model = SplitAlbedoModel(cfg, A_intr, None)
            else:
                raise ValueError(f'split_component must be full, exogenic or intrinsic: {comp!r}')
            # Residual: the uniform-field R only. The measured-map residual
            # R_meas is an albedo response tied to the whole map and cannot be
            # assigned to the body or the tidal frame, so it is dropped here.
            self._R = np.array(forcing['R_unif']) if cfg.use_residual else None
        else:
            raise ValueError(f'albedo_mode must be darkening, static_map or split: '
                             f'{cfg.albedo_mode!r}')

        # Build the temperature surrogate (dT_PTM interpolator).
        if forcing is not None:
            if not cfg.use_ptm3d_surrogate:
                raise ValueError('forcing_file given but use_ptm3d_surrogate is False')
            self._dT_interp = RegularGridInterpolator(
                (forcing['dT_theta'], forcing['dT_lam']), forcing['dT'])
            self._dT_grid = (forcing['dT_theta'], forcing['dT_lam'], forcing['dT'])
        elif cfg.use_ptm3d_surrogate:
            th1d, lm1d, dT = build_static_dT_ptm()
            self._dT_interp = RegularGridInterpolator(
                (th1d, lm1d), dT, bounds_error=False, fill_value=None)
            self._dT_grid = (th1d, lm1d, dT)

        # Initial condition: uniform b, uniform albedo A_bright, and the
        # fossil chosen by cfg.use_fossil_bulge (default: none).
        self.f_dark = 0.0
        A0 = self.model.albedo_field(0.0, spin_pole=self.spin_pole)
        T0 = self.surface_temperature_coupled(A0, self.spin_pole)
        b_eq0 = self._b_eq(T0)
        if cfg.init_b_mode == 'constant':
            b0 = cfg.init_b_const_m
            self.b = np.full_like(self.b, b0)
        elif cfg.init_b_mode == 'equilibrium_mean':
            b0 = _area_mean(b_eq0, self.theta, self.lam)
            self.b = np.full_like(self.b, b0)
        elif cfg.init_b_mode == 'equilibrium':
            # the thermal-equilibrium shell for the initial pole
            self.b = b_eq0.copy()
            b0 = _area_mean(b_eq0, self.theta, self.lam)
        elif cfg.init_b_mode in ('uniform_equilibrium', 'blend'):
            # equilibrium for the uniform albedo (forcing switched on at t=0),
            # or a partial relaxation toward the forced equilibrium. b_c is
            # the forced-equilibrium mean for every initial condition, so the
            # OSPW parameters do not change with the initial condition.
            # T for the uniform albedo takes the uniform-field residual: with
            # static_map 'measured', self._R is R_meas, which belongs to the map.
            A_u = np.full_like(A0, self._A_unif)
            R_run = self._R
            if forcing is not None:
                self._R = np.array(forcing['R_unif']) if cfg.use_residual else None
            try:
                b_u = self._b_eq(self.surface_temperature_coupled(A_u, self.spin_pole))
            finally:
                self._R = R_run
            f = 0.0 if cfg.init_b_mode == 'uniform_equilibrium' else cfg.init_blend_frac
            self.b = (1.0 - f) * b_u + f * b_eq0
            b0 = _area_mean(b_eq0, self.theta, self.lam)
        else:
            raise ValueError(f'unknown init_b_mode {cfg.init_b_mode!r}')
        if cfg.init_l2_perturb_km and cfg.init_l2_perturb_km[1] > 0:
            dbp, rms, u = _random_l2_field(self.theta, self.lam, cfg.init_perturb_seed,
                                           tuple(1e3 * x for x in cfg.init_l2_perturb_km))
            self.b = self.b + dbp
            h = G.harmonics(dbp, self.theta, self.lam)
            self.init_perturbation = {'seed': cfg.init_perturb_seed, 'rms_km': rms / 1e3,
                                      'u': [float(x) for x in u],
                                      't20_2t22_km': h['t20_2t22'] / 1e3}
        self._b0_uniform = b0
        # OSPW parameters for the uniform start (b_c = b0 unless fixed).
        self._ospw_p = _ospw_params(cfg, b0)
        self.I_fossil, self.fossil_t20, self.fossil_t22 = _fossil_state(cfg, self._ospw_p)
        self._ospw_Cf = PW.dissipation_cf(self._ospw_p)
        self._ospw_I0 = PW.moment_I0(self._ospw_p)
        self._seed_armed = True
        self._last_seed_pole = None
        self.seed_events = []
        if cfg.init_pole_offset_deg:
            d, az = np.radians(cfg.init_pole_offset_deg), np.radians(cfg.init_pole_azimuth_deg)
            self.spin_pole = np.array([np.sin(d) * np.cos(az), np.sin(d) * np.sin(az), np.cos(d)])
            self._pole0 = self.spin_pole.copy()

        # PTM3D refresh bookkeeping (Phase 2).
        self._ptm3d_calls = 0
        self._ptm3d_last_pole = self.spin_pole.copy()
        self._ptm3d_last_albedo = A0.copy()
        self._ptm3d_runner = None

        # Re-initialize history + snapshots for the new initial state.
        for k in self.history:
            if k != 'cycle_events':
                self.history[k] = []
        self.history['cycle_events'] = []
        self.map_snapshots = []
        self._record()

    # -- temperature surrogate --------------------------------------------
    def surface_temperature_coupled(self, A, spin_pole):
        T_solar = self.model.surface_temperature_for_pole(A, spin_pole)
        if self._R is not None:
            T_solar = T_solar + self._R          # body-fixed residual (static_map)
        if self._dT_interp is None:
            return T_solar
        theta_t, lam_t = _tidal_coordinates(self.theta, self.lam, spin_pole,
                                            self._meridian_arg())
        pts = np.stack([theta_t.ravel(), lam_t.ravel()], axis=-1)
        dT = self._dT_interp(pts).reshape(theta_t.shape)
        return T_solar + dT

    # -- per-location thermal diffusion timescale -------------------------
    def _tau_diff_years(self, b, T_s):
        """Local relaxation time scale tau_diff = b^2 / kappa [yr].

        Approximations (kept on purpose; document before relying on them):
          * One e-folding time per location for the whole profile. The real
            response of a conductive shell to a step in T_s or q is a sum of
            modes; the slowest is ~b^2/(pi^2 kappa), so b^2/kappa is ~10x
            longer than the fundamental mode and is an upper-end estimate.
          * kappa = k/(rho c_p), with c_p = 90 + 7.49 T (cp_mode='ice'), is
            evaluated at the SURFACE temperature, where
            k = a1/T + a0 is largest; the basal ice (T -> 273 K) diffuses
            ~2.5x more slowly.
          * Latent heat is ignored: the base of the shell moves only as fast
            as the conductive flux imbalance can melt or freeze ice (a Stefan
            problem), which can be slower than b^2/kappa.
          * Lateral heat flow is ignored (thin-shell assumption, as in O&S).
        The result is clipped to [tau_diff_floor_yr, tau_diff_cap_yr]; with
        the defaults (1e3, 5e7 yr) the clip is inactive for 5-50 km shells.
        """
        cfg = self.cfg
        k = C.A1_CONDUCTIVITY / T_s + C.A0_CONDUCTIVITY        # 488/T + 0.468
        if cfg.cp_mode == 'ice':
            cp = C.ice_heat_capacity(T_s)
        elif cfg.cp_mode == 'constant':
            cp = cfg.cp_const
        else:
            raise ValueError(f"cp_mode must be 'ice' or 'constant', got {cfg.cp_mode!r}")
        kappa = k / (C.ICE_DENSITY * cp)                       # m^2 / s
        tau_s = b ** 2 / kappa
        tau_yr = tau_s / (365.25 * 86400.0)
        return np.clip(tau_yr, cfg.tau_diff_floor_yr, cfg.tau_diff_cap_yr)

    def _tau_relax_years(self, b, T_s):
        """Local shell relaxation time [yr] for cfg.relaxation_mode.

        'stefan_eigen': f rho L b / q with f the exact eigenmode factor for the
        local T_s (ojakangas_stevenson.polar_wander.stefan_relaxation_time;
        ~19 Myr for 30 km). 'stefan_sensible': f = 1 + S/L (upper estimate,
        ~20 Myr). 'stefan': rho L b / q (latent only, ~13 Myr).
        'diffusive': _tau_diff_years (b^2/kappa). All are clipped to
        [tau_diff_floor_yr, tau_diff_cap_yr]; in the Stefan modes a clip that
        binds is reported once. cfg.conv_flow_tau multiplies the result by
        conv_s (option; see PolarWanderConfig).
        """
        cfg = self.cfg
        if cfg.relaxation_mode == 'diffusive':
            return self._tau_diff_years(b, T_s)
        sens = {'stefan_eigen': 'eigen', 'stefan_sensible': 'exact', 'stefan': 'none'}.get(
            cfg.relaxation_mode)
        if sens is None:
            raise ValueError(f"relaxation_mode must be 'stefan_eigen', 'stefan_sensible', "
                             f"'stefan' or 'diffusive', got {cfg.relaxation_mode!r}")
        tau = PW.stefan_relaxation_time(b, T_s, sensible=sens) / (365.25 * 86400.0)
        if cfg.conv_flow_tau:
            tau = tau * cfg.conv_s
        if cfg.relaxation_scale != 1.0:
            if not cfg.relaxation_scale > 0:
                raise ValueError(f'relaxation_scale must be > 0: {cfg.relaxation_scale}')
            tau = tau * cfg.relaxation_scale
        out = np.clip(tau, cfg.tau_diff_floor_yr, cfg.tau_diff_cap_yr)
        if not self._warned_tau_clip and not np.array_equal(out, tau):
            self._warned_tau_clip = True
            print(f'  WARNING: Stefan relaxation time clipped to '
                  f'[{cfg.tau_diff_floor_yr:.0e}, {cfg.tau_diff_cap_yr:.0e}] yr on '
                  f'{np.mean(out != tau):.1%} of the grid (range {tau.min():.2e}-'
                  f'{tau.max():.2e} yr) at t={self.t/1e6:.2f} Myr')
        return out

    # -- one coupled step (mirror of ShellEvolution.step) -----------------
    def step(self, dt):
        cfg = self.cfg
        self.model.meridian = self._meridian_arg()
        self._update_dissipation()
        # 1. Exponential darkening; split albedo: A_exo relaxes toward the
        #    tidal-frame template for the current pole and meridian
        self.f_dark = 1.0 - (1.0 - self.f_dark) * np.exp(-dt / cfg.tau_dark)
        if isinstance(self.model, SplitAlbedoModel):
            self.model.advance(dt, self.spin_pole)
        # 2-3. Temperature (surrogate) and O&S equilibrium thickness
        A = self.model.albedo_field(self.f_dark, spin_pole=self.spin_pole)
        T_s = self.surface_temperature_coupled(A, self.spin_pole)
        b_eq = self._b_eq(T_s)
        # 4. Relax shell toward equilibrium at the LOCAL relaxation timescale
        tau_relax = self._tau_relax_years(self.b, T_s)
        self.b = b_eq + (self.b - b_eq) * np.exp(-dt / tau_relax)
        # 5-6. Physical inertia tensor (Eq. 2.18 shell + body-fixed fossil)
        #      and OSPW reorientation (Eq. 5.39, with the psi0 seed).
        self._reorient(dt, self.total_inertia_tensor())
        self.t += dt

    # -- snapshots ---------------------------------------------------------
    def _capture_snapshot(self):
        A = self.model.albedo_field(self.f_dark, spin_pole=self.spin_pole)
        T = self.surface_temperature_coupled(A, self.spin_pole)
        self.map_snapshots.append(dict(
            t=self.t, f_dark=self.f_dark, A=A.copy(), T=T.copy(),
            b=self.b.copy(), spin_pole=self.spin_pole.copy()))

    # -- Phase 2 hook (no-op when static) ---------------------------------
    def _maybe_refresh_ptm3d(self):
        if not self.cfg.use_live_ptm3d or self._ptm3d_runner is None:
            return
        if self._ptm3d_calls >= self.cfg.ptm3d_max_calls:
            return
        pole_drift = np.degrees(np.arccos(
            np.clip(np.dot(self.spin_pole, self._ptm3d_last_pole), -1, 1)))
        A_now = self.model.albedo_field(self.f_dark, spin_pole=self.spin_pole)
        alb_rms = np.sqrt(_area_mean((A_now - self._ptm3d_last_albedo) ** 2,
                                     self.theta, self.lam))
        if (pole_drift > self.cfg.ptm3d_refresh_pole_deg
                or alb_rms > self.cfg.ptm3d_refresh_albedo_rms):
            self._refresh_dT_ptm(A_now)
            self._ptm3d_last_pole = self.spin_pole.copy()
            self._ptm3d_last_albedo = A_now.copy()

    def _refresh_dT_ptm(self, A_now):
        """Run full PTM3D with the current albedo and update dT_PTM."""
        try:
            th1d, lm1d, dT = self._ptm3d_runner.run_and_dT(
                self, A_now, self.spin_pole)
            self._dT_interp = RegularGridInterpolator(
                (th1d, lm1d), dT, bounds_error=False, fill_value=None)
            self._dT_grid = (th1d, lm1d, dT)
            self._ptm3d_calls += 1
            print(f'  [PTM3D refresh #{self._ptm3d_calls}] t={self.t/1e6:.2f} Myr, '
                  f'dT range {dT.min():+.1f}..{dT.max():+.1f} K')
        except Exception as e:           # keep last good dT_PTM on failure
            print(f'  [PTM3D refresh FAILED at t={self.t/1e6:.2f} Myr: {e}; '
                  f'keeping previous correction]')

    # -- main loop (multi-rate: fine during darkening, coarse after) -------
    def run(self):
        cfg = self.cfg
        t_dark_end = 10.0 * cfg.tau_dark            # darkening ~saturated
        dt_coarse = cfg.t_max_years / cfg.n_coarse_steps
        # Fine step resolves the darkening ramp, but is never coarser than the
        # coarse step: for tau_dark ~ 1e6-1e7 yr, tau_dark/20 would otherwise
        # exceed t_max/n_coarse_steps.
        dt_fine = cfg.dt_years if cfg.dt_years else min(cfg.tau_dark / 20.0, dt_coarse)
        snap_dt = cfg.t_max_years / cfg.n_snapshots
        rec_dt = cfg.t_max_years / cfg.n_history
        next_snap, next_rec, next_check = snap_dt, rec_dt, 5e5

        self._capture_snapshot()
        while self.t < cfg.t_max_years - 1.0:
            dt = dt_fine if self.t < t_dark_end else dt_coarse
            # Cap dt during active reorientation so the pole rotates at most
            # reorient_dtheta_max_deg per step (resolve the fast OSPW rotation,
            # using the previous step's angular velocity as the predictor).
            om = abs(getattr(self, '_last_omega_deg_per_Myr', 0.0))  # deg/Myr
            if om > 1e-9:
                dt = min(dt, cfg.reorient_dtheta_max_deg / om * 1e6)
            dt = min(dt, cfg.t_max_years - self.t)
            self.step(dt)
            if self.t >= next_rec:
                self._record(); next_rec += rec_dt
            if self.t >= next_check:
                self.check_reorientation(); next_check += 5e5
            if self.t >= next_snap:
                self._capture_snapshot(); next_snap += snap_dt
            self._maybe_refresh_ptm3d()
        self._record()
        self._capture_snapshot()
        for k in self.history:
            if k != 'cycle_events':
                self.history[k] = np.array(self.history[k])
        return self.history


# ---------------------------------------------------------------------------
# Outputs: figures + 3D globe animation
# ---------------------------------------------------------------------------
def _mollweide(ax, theta, lam, field, title, cmap, vmin=None, vmax=None, cbar=''):
    lat = np.pi / 2 - theta
    LON, LAT = np.meshgrid(lam, lat)
    pcm = ax.pcolormesh(LON, LAT, field, cmap=cmap, vmin=vmin, vmax=vmax,
                        shading='auto', rasterized=True)
    ax.set_title(title, fontsize=9)
    ax.grid(True, lw=0.3, alpha=0.4); ax.set_xticklabels([]); ax.set_yticklabels([])
    cb = ax.figure.colorbar(pcm, ax=ax, orientation='horizontal', fraction=0.05, pad=0.03)
    cb.set_label(cbar, fontsize=7); cb.ax.tick_params(labelsize=6)


def _apex_dir(spin_pole):
    """Trailing-hemisphere apex direction (body frame) for a given spin pole."""
    s = np.asarray(spin_pole, float)
    sub_j = np.array([1.0, 0.0, 0.0])
    e_sub = sub_j - np.dot(sub_j, s) * s
    n = np.linalg.norm(e_sub)
    if n < 1e-10:
        e_sub = np.array([0.0, 1.0, 0.0]) - np.dot([0.0, 1.0, 0.0], s) * s
        n = np.linalg.norm(e_sub)
    e_sub /= n
    return np.cross(s, e_sub)            # e_trail = apex (theta_t=90, lam_t=90)


def fig_pole_timeseries(hist, cfg, save_path):
    t = hist['t_years'] / 1e6
    colat = hist['spin_pole_theta']
    phi = hist['spin_pole_phi']
    # Unwrapped rotation angle in the (y,z) meridian plane the pole travels in.
    pz = np.cos(np.radians(colat))
    py = np.sin(np.radians(colat)) * np.sin(np.radians(phi))
    rot = np.degrees(np.unwrap(np.arctan2(py, pz)))
    fig, ax = plt.subplots(4, 1, figsize=(9, 9), sharex=True)
    ax[0].plot(t, colat, color='#C44E52'); ax[0].set_ylabel('pole colatitude\nfrom original [deg]')
    ax[0].set_ylim(-5, 185); ax[0].axhline(90, color='0.7', lw=0.6, ls='--')
    ax[1].plot(t, rot, color='#3B6FB0'); ax[1].set_ylabel('cumulative TPW\nrotation [deg]')
    ax[2].plot(t, hist['f_dark'], color='#555', label='darkening $f$')
    axb = ax[2].twinx(); axb.plot(t, hist['b_avg_km'], color='#2A7F4F')
    ax[2].set_ylabel('darkening $f$', color='#555'); axb.set_ylabel(r'$\langle b\rangle$ [km]', color='#2A7F4F')
    ax[3].plot(t, hist['misalignment_deg'], color='#8030A0')
    ax[3].set_ylabel('MoI misalignment\n[deg]'); ax[3].set_xlabel('Time [Myr]')
    for a in ax:
        a.grid(True, lw=0.3, alpha=0.4)
    onset = t[np.argmax(colat > 5)] if (colat > 5).any() else None
    ttl = f'Spin-pole evolution (tau_dark={cfg.tau_dark:.0e} yr, {cfg.rheology_case})'
    if onset:
        ttl += f' -- TPW onset ~{onset:.1f} Myr'
        for a in ax:
            a.axvline(onset, color='0.6', lw=0.7, ls=':')
    fig.suptitle(ttl, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98]); fig.savefig(save_path, dpi=160); plt.close(fig)


def fig_map_snapshots(snaps, theta, lam, key, cmap, label, save_path, n=6):
    idx = np.linspace(0, len(snaps) - 1, n).astype(int)
    vmin = min(snaps[i][key].min() for i in idx)
    vmax = max(snaps[i][key].max() for i in idx)
    fig = plt.figure(figsize=(13, 5.2))
    for k, i in enumerate(idx):
        ax = fig.add_subplot(2, 3, k + 1, projection='mollweide')
        _mollweide(ax, theta, lam, snaps[i][key], f't={snaps[i]["t"]/1e6:.1f} Myr',
                   cmap, vmin, vmax, label)
    fig.suptitle(f'{label} snapshots', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(save_path, dpi=160); plt.close(fig)


def make_globe_animation(snaps, theta, lam, out_stem, pole0, fps=20,
                         A_bright=0.5, A_dark=0.15):
    """3D globe colored by albedo, with the spin-axis vector + original pole."""
    import imageio.v2 as imageio
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    st = np.sin(theta)[:, None]; ct = np.cos(theta)[:, None]
    cl = np.cos(lam)[None, :]; sl = np.sin(lam)[None, :]
    X = st * cl; Y = st * sl; Z = ct * np.ones_like(cl)
    norm = plt.Normalize(vmin=A_dark, vmax=A_bright)   # 0.15->black, 0.5->white
    cmap = plt.get_cmap('gray')
    frames = []
    for s in snaps:
        fc = cmap(norm(s['A']))
        fig = plt.figure(figsize=(5.2, 5.4))
        ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
        ax.plot_surface(X, Y, Z, facecolors=fc, rstride=1, cstride=1,
                        shade=False, antialiased=False, linewidth=0)
        sp = s['spin_pole']
        ax.quiver(0, 0, 0, sp[0], sp[1], sp[2], length=1.6, color='red', lw=2.5,
                  arrow_length_ratio=0.12)
        ax.quiver(0, 0, 0, -sp[0], -sp[1], -sp[2], length=1.6, color='red', lw=2.5,
                  arrow_length_ratio=0.0)
        ax.quiver(0, 0, 0, pole0[0], pole0[1], pole0[2], length=1.45, color='#3070D0',
                  lw=1.3, arrow_length_ratio=0.1, alpha=0.7)
        ap = _apex_dir(sp)
        ax.scatter([1.15 * ap[0]], [1.15 * ap[1]], [1.15 * ap[2]], color='saddlebrown',
                   s=40, marker='o', depthshade=False)
        ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2); ax.set_zlim(-1.2, 1.2)
        ax.set_box_aspect([1, 1, 1]); ax.set_axis_off()
        ax.view_init(elev=12, azim=-88)        # broadside to the (y,z) wander plane
        ax.set_title(f't = {s["t"]/1e6:5.2f} Myr   f = {s["f_dark"]:.2f}', fontsize=10)
        fig.tight_layout()
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        frames.append(buf)
        plt.close(fig)
    gif = str(out_stem) + '.gif'
    imageio.mimsave(gif, frames, duration=1.0 / fps, loop=0)
    mp4 = str(out_stem) + '.mp4'
    try:
        import subprocess
        h, w = frames[0].shape[:2]
        proc = subprocess.Popen(
            ['/usr/local/bin/ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
             '-pix_fmt', 'rgb24', '-s', f'{w}x{h}', '-r', str(fps), '-i', '-',
             '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', mp4],
            stdin=subprocess.PIPE)
        for fr in frames:
            proc.stdin.write(fr.tobytes())
        proc.stdin.close(); proc.wait()
    except Exception as e:
        print(f'  mp4 failed ({e}); gif written')
    print(f'  animation -> {gif} (+ mp4 if ffmpeg)')


def produce_outputs(evo, hist, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    th, lm = evo.theta, evo.lam
    # Persist history + snapshots so figures can be regenerated without re-running.
    np.savez_compressed(
        os.path.join(save_dir, 'run_data.npz'), theta=th, lam=lm,
        snap_t=np.array([s['t'] for s in evo.map_snapshots]),
        snap_A=np.array([s['A'] for s in evo.map_snapshots]),
        snap_b=np.array([s['b'] for s in evo.map_snapshots]),
        snap_pole=np.array([s['spin_pole'] for s in evo.map_snapshots]),
        parameter_set=json.dumps(C.active_parameter_set().as_dict()),
        config=json.dumps({k: getattr(evo.cfg, k) for k in evo.cfg.__dataclass_fields__},
                          default=str),
        seed_events=np.array(evo.seed_events, dtype=float).reshape(-1, 4),
        **{f'h_{k}': hist[k] for k in hist if k != 'cycle_events'})
    fig_pole_timeseries(hist, evo.cfg, os.path.join(save_dir, 'pole_timeseries.png'))
    fig_map_snapshots(evo.map_snapshots, th, lm, 'A', 'gray', 'Bond albedo',
                      os.path.join(save_dir, 'snapshots_albedo.png'))
    fig_map_snapshots(evo.map_snapshots, th, lm, 'T', 'inferno', 'Surface T [K]',
                      os.path.join(save_dir, 'snapshots_temperature.png'))
    fig_map_snapshots(evo.map_snapshots, th, lm, 'b', 'gist_earth', 'Shell thickness [m]',
                      os.path.join(save_dir, 'snapshots_thickness.png'))
    try:
        pwc.fig6_spin_pole_trajectory(hist, evo.cfg,
                                      os.path.join(save_dir, 'pole_trajectory.png'))
    except Exception as e:
        print(f'  (pole_trajectory skipped: {e})')
    make_globe_animation(evo.map_snapshots, th, lm,
                         os.path.join(save_dir, 'globe_animation'), evo._pole0,
                         A_bright=evo.cfg.A_bright, A_dark=evo.cfg.A_dark)
    print(f'  outputs -> {save_dir}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tau-dark', type=float, default=1e4,
                   help='darkening e-folding time [yr]; tested 1e4-1e7')
    p.add_argument('--A-dark', type=float, default=0.3814,
                   help='dark endmember Bond albedo (Hapke non-ice endmember)')
    p.add_argument('--psi0', type=float, default=pwc.PSI0_OBLIQUITY_RAD,
                   help='psi0 seed [rad] applied when the pole becomes unstable')
    p.add_argument('--fossil', choices=('none', 'hydrostatic'), default='none',
                   help='fossil bulge: none, or body-fixed hydrostatic (OSPW Eqs. 3.8-3.10)')
    p.add_argument('--thickness-method', choices=('eq28', 'exact'), default='eq28')
    p.add_argument('--parameter-set', default='paper', choices=sorted(C.PARAMETER_SETS),
                   help="constants: 'paper' (validation) or 'modern' (science)")
    p.add_argument('--t-max', type=float, default=2e7)
    p.add_argument('--rheology', default='maxwell_h0')
    p.add_argument('--cp-mode', default='ice', choices=['ice', 'constant'],
                   help="'ice': cp = 90 + 7.49 T (heat1d planets.py); 'constant': cp_const")
    p.add_argument('--no-ptm3d', action='store_true', help='disable surrogate (dT=0)')
    p.add_argument('--live-ptm3d', action='store_true', help='Phase 2 PTM3D refresh')
    p.add_argument('--smoke', action='store_true', help='fast tiny-grid smoke test')
    p.add_argument('--n-snapshots', type=int, default=150)
    p.add_argument('--no-outputs', action='store_true', help='skip figures/animation')
    p.add_argument('--save-dir', default=None)
    args = p.parse_args()
    C.use_parameter_set(args.parameter_set)
    print(C.describe_active())

    cfg = CoupledConfig(
        tau_dark=args.tau_dark, A_dark=args.A_dark, psi0_rad=args.psi0,
        use_fossil_bulge=(args.fossil == 'hydrostatic'),
        thickness_method=args.thickness_method,
        t_max_years=args.t_max, rheology_case=args.rheology,
        cp_mode=args.cp_mode, use_ptm3d_surrogate=not args.no_ptm3d,
        use_live_ptm3d=args.live_ptm3d, n_snapshots=args.n_snapshots,
        misalignment_threshold_deg=90.0)
    if args.smoke:
        cfg.n_theta, cfg.n_lam = 46, 90
        cfg.t_max_years = 2e5
        cfg.n_snapshots = 12

    tag = (f'tau{cfg.tau_dark:.0e}_{cfg.rheology_case}_{args.fossil}_'
           f'psi{cfg.psi0_rad:.0e}_{args.parameter_set}').replace('+', '')
    save_dir = args.save_dir or os.path.join(PLOTDIR, tag)
    print(f'Coupled run: tau_dark={cfg.tau_dark:.0e} yr, t_max={cfg.t_max_years:.0e} yr, '
          f'grid {cfg.n_theta}x{cfg.n_lam}, surrogate={cfg.use_ptm3d_surrogate}, '
          f'A_dark={cfg.A_dark}, fossil={args.fossil}, psi0={cfg.psi0_rad:.1e} rad')
    import time
    t0 = time.time()
    evo = CoupledEvolution(cfg)
    hist = evo.run()
    colat = hist['spin_pole_theta']
    onset = hist['t_years'][np.argmax(colat > 5)] / 1e6 if (colat > 5).any() else None
    print(f'  done in {time.time()-t0:.0f}s: {len(hist["t_years"])} records, '
          f'{len(evo.map_snapshots)} snapshots, TPW onset='
          f'{f"{onset:.1f} Myr" if onset else "none"}, '
          f'<b> {hist["b_avg_km"][0]:.2f}->{hist["b_avg_km"][-1]:.2f} km')
    if not args.no_outputs:
        produce_outputs(evo, hist, save_dir)
    return evo, hist


if __name__ == '__main__':
    main()
