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
  * Relaxation: tau_diff = b^2/kappa per location -- see _tau_diff_years for
    what that approximation leaves out.

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
    init_b_mode: str = 'equilibrium_mean'  # 'equilibrium_mean' or 'constant'
    init_b_const_m: float = 20e3
    # --- reorientation ---
    tpw_model: str = 'ospw'
    # --- per-location thermal diffusion ---
    cp_mode: str = 'constant'           # 'constant' or 'linear' (7.49*T)
    cp_const: float = 2000.0
    tau_diff_floor_yr: float = 1e3
    tau_diff_cap_yr: float = 5e7
    # --- PTM3D surrogate / multi-rate ---
    use_ptm3d_surrogate: bool = True
    use_live_ptm3d: bool = False        # Phase 2: periodic full-PTM3D refresh
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
def _darkening_weight_cosine_apex(theta, lam, spin_pole, power=1.0):
    """Apex-centered darkening weight: w = max(0, cos gamma)^power, where gamma
    is the angular distance from the trailing-hemisphere apex.

    In the spin/tidal frame the apex (trailing equator point) is the unit
    vector (0,1,0); for a general point (theta_t, lam_t) the cosine of the
    angular distance is cos(gamma) = sin(theta_t) * sin(lam_t).
    """
    theta_t, lam_t = _tidal_coordinates(theta, lam, spin_pole)
    cos_gamma = np.sin(theta_t) * np.sin(lam_t)
    return np.maximum(0.0, cos_gamma) ** power


class CosineApexAlbedoModel(AlbedoDarkeningModel):
    """Darkening with an apex-centered cosine spot (or the parent sin form)."""

    def albedo_field(self, f_dark, spin_pole=None):
        cfg = self.cfg
        if getattr(cfg, 'darkening_mode', 'cosine') == 'sin':
            return super().albedo_field(f_dark, spin_pole)
        if spin_pole is None:
            spin_pole = np.array([0.0, 0.0, 1.0])
        w = _darkening_weight_cosine_apex(
            self.theta, self.lam, spin_pole, cfg.darkening_power)
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
class CoupledEvolution(ShellEvolution):
    """ShellEvolution with PTM3D-coupled temperature, per-location tau_diff
    relaxation, cosine-apex darkening, and a uniform / no-fossil initial state.
    """

    def __init__(self, cfg):
        # Parent sets up grid, D (spin pole = z), OSPW, history, and records.
        super().__init__(cfg)
        # Swap in the cosine-apex darkening model.
        self.model = CosineApexAlbedoModel(cfg)
        self._pole0 = self.spin_pole.copy()

        # Build the temperature surrogate (dT_PTM interpolator).
        self._dT_interp = None
        if cfg.use_ptm3d_surrogate:
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
        else:
            b0 = _area_mean(b_eq0, self.theta, self.lam)
        self.b = np.full_like(self.b, b0)
        self._b0_uniform = b0
        # OSPW parameters for the uniform start (b_c = b0 unless fixed).
        self._ospw_p = _ospw_params(cfg, b0)
        self.I_fossil, self.fossil_t20, self.fossil_t22 = _fossil_state(cfg, self._ospw_p)
        self._ospw_Cf = PW.dissipation_cf(self._ospw_p)
        self._ospw_I0 = PW.moment_I0(self._ospw_p)
        self._seed_armed = True
        self.seed_events = []

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
        if self._dT_interp is None:
            return T_solar
        theta_t, lam_t = _tidal_coordinates(self.theta, self.lam, spin_pole)
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
          * kappa = k/(rho c_p) is evaluated at the SURFACE temperature, where
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
        cp = 7.49 * T_s if cfg.cp_mode == 'linear' else cfg.cp_const
        kappa = k / (C.ICE_DENSITY * cp)                       # m^2 / s
        tau_s = b ** 2 / kappa
        tau_yr = tau_s / (365.25 * 86400.0)
        return np.clip(tau_yr, cfg.tau_diff_floor_yr, cfg.tau_diff_cap_yr)

    # -- one coupled step (mirror of ShellEvolution.step) -----------------
    def step(self, dt):
        cfg = self.cfg
        self._update_dissipation()
        # 1. Exponential darkening
        self.f_dark = 1.0 - (1.0 - self.f_dark) * np.exp(-dt / cfg.tau_dark)
        # 2-3. Temperature (surrogate) and O&S equilibrium thickness
        A = self.model.albedo_field(self.f_dark, spin_pole=self.spin_pole)
        T_s = self.surface_temperature_coupled(A, self.spin_pole)
        b_eq = self._b_eq(T_s)
        # 4. Relax shell toward equilibrium at the LOCAL diffusion timescale
        tau_diff = self._tau_diff_years(self.b, T_s)
        self.b = b_eq + (self.b - b_eq) * np.exp(-dt / tau_diff)
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
    p.add_argument('--psi0', type=float, default=1e-6,
                   help='psi0 seed [rad] applied when the pole becomes unstable')
    p.add_argument('--fossil', choices=('none', 'hydrostatic'), default='none',
                   help='fossil bulge: none, or body-fixed hydrostatic (OSPW Eqs. 3.8-3.10)')
    p.add_argument('--thickness-method', choices=('eq28', 'exact'), default='eq28')
    p.add_argument('--parameter-set', default='paper', choices=sorted(C.PARAMETER_SETS),
                   help="constants: 'paper' (validation) or 'modern' (science)")
    p.add_argument('--t-max', type=float, default=2e7)
    p.add_argument('--rheology', default='maxwell_h0')
    p.add_argument('--cp-mode', default='constant', choices=['constant', 'linear'])
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
