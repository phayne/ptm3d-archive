#!/usr/bin/env python3
"""OSPW overdamped pendulum model for polar wander of Europa's ice shell.

Thin command-line and plotting layer over ``ojakangas_stevenson.polar_wander``,
which holds the physics of Ojakangas & Stevenson (1989b), Icarus 81, 242-270:
  - Eq. 2.11 Willemann compensation; Eq. 2.18 bottom loading + iceberg effect
  - Eqs. 3.8-3.10 hydrostatic fossil bulge; Eqs. 3.11/3.12 stability test
  - Eq. 5.15 fractured-shell damping C_f; Eq. 5.29 I_0
  - Eq. 5.39 pendulum: psi_ddot = 2 (t/tau) eps n^2 sin 2psi - 4 eps (C_f/I_0) psi_dot,
    overdamped: psi_dot = (t/tau)(n^2 I_0 / 2 C_f) sin 2psi, with t/tau growing
    as t / tau, tau ~ 1e12 yr (p. 262) -- NOT tau_th.

Usage:
    python scripts/ospw_polar_wander.py                # Default run with figures
    python scripts/ospw_polar_wander.py --fig5          # Fig 5 (psi vs t)
    python scripts/ospw_polar_wander.py --fig6          # Fig 6 (tau_f vs T_f)
    python scripts/ospw_polar_wander.py --instability   # Check instability criterion
"""

import sys
import os
import argparse
from dataclasses import dataclass
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'python'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ojakangas_stevenson import constants as C
from ojakangas_stevenson import polar_wander as PW

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

YEAR_S = PW.YEAR_S


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class OSPWConfig:
    """Parameters for the OSPW polar wander model (SI units).

    ``None`` for europa_radius / mean_density / europa_mass means "the active
    constants set". See ojakangas_stevenson.polar_wander.OSPWParams for the
    sources of each default.
    """
    shell_thickness: float = 20e3       # b_c [m]
    lithosphere_thickness: float = 2e3  # d [m]
    europa_radius: Optional[float] = None
    ice_density: float = C.ICE_DENSITY
    water_density: float = C.WATER_DENSITY
    mean_density: Optional[float] = None
    ice_rigidity: float = C.ICE_RIGIDITY
    poisson_ratio: float = 0.33
    melting_viscosity: float = 3e14       # eta_0 [Pa s] (p. 256: 3e15 P)
    activation_param: float = 15.0        # l
    melting_temp: float = C.T_MELT
    surface_temp: float = 100.0
    fracture_base_temp: float = 124.0     # T_f (p. 264: "T_f ~ 124 K")
    love_number_h: float = 1.26
    tau_growth_yr: float = 1e12           # tau in t/tau (p. 262)
    europa_mass: Optional[float] = None
    # Thickness harmonics (Table I flow law n_G = 3 gives t20 + 2 t22 = 1.1 km)
    t20_km: float = -0.70
    t22_km: float = 0.24

    def __post_init__(self):
        if self.europa_radius is None:
            self.europa_radius = C.EUROPA_RADIUS
        if self.mean_density is None:
            self.mean_density = C.EUROPA_DENSITY
        if self.europa_mass is None:
            self.europa_mass = C.EUROPA_MASS

    def to_params(self, Tf=None):
        return PW.OSPWParams(
            shell_thickness=self.shell_thickness,
            lithosphere_thickness=self.lithosphere_thickness,
            ice_density=self.ice_density, water_density=self.water_density,
            rigidity=self.ice_rigidity, poisson_ratio=self.poisson_ratio,
            love_h=self.love_number_h, melting_viscosity=self.melting_viscosity,
            activation_l=self.activation_param, melting_temp=self.melting_temp,
            surface_temp=self.surface_temp,
            fracture_temp=self.fracture_base_temp if Tf is None else Tf,
            tau_growth_yr=self.tau_growth_yr, radius=self.europa_radius,
            mean_density=self.mean_density, mass=self.europa_mass)

    @property
    def gamma_tidal(self):
        return PW.gamma_tidal(self.to_params())

    @property
    def delta_rho(self):
        return self.water_density - self.ice_density

    @property
    def t20_plus_2t22(self):
        """Instability diagnostic: t20 + 2*t22 [km]."""
        return self.t20_km + 2 * self.t22_km


# ---------------------------------------------------------------------------
# Physics: wrappers over ojakangas_stevenson.polar_wander
# ---------------------------------------------------------------------------

def compute_Cw(cfg):
    """Willemann compensation parameter, Eq. 2.11."""
    return PW.willemann_cw(cfg.to_params())


def compute_shell_gravity(t2m_km, cfg, Cw=None):
    """C_2m of a thickness harmonic t_2m [km] (Eq. 2.18)."""
    return PW.shell_c2m(t2m_km * 1e3, cfg.to_params(), Cw)


def compute_fossil_bulge_gravity(cfg, Cw=None):
    """Fossil bulge (C20, C22), Eq. 3.10."""
    return PW.fossil_c20_c22(cfg.to_params(), Cw)


def check_instability(cfg, Cw=None):
    """Quasi-static I_yy > I_zz test (Eqs. 3.4, 3.11, 3.12)."""
    p = cfg.to_params()
    Cw = PW.willemann_cw(p) if Cw is None else Cw
    res = PW.stability_test(cfg.t20_plus_2t22 * 1e3, p, Cw)
    C20s = PW.shell_c2m(cfg.t20_km * 1e3, p, Cw)
    C22s = PW.shell_c2m(cfg.t22_km * 1e3, p, Cw)
    C20f, C22f = PW.fossil_c20_c22(p, Cw)
    res.update({
        't20_2t22_km': cfg.t20_plus_2t22,
        'threshold_km': res['threshold_km_exact'],
        'C20_shell': C20s, 'C22_shell': C22s,
        'C20_fossil': C20f, 'C22_fossil': C22f,
        'C20_total': C20s + C20f, 'C22_total': C22s + C22f,
        'criterion_value': (C20s + C20f) + 2 * (C22s + C22f),
    })
    return res


def compute_I0(cfg):
    """I_0 of the shell + ocean, Eq. 5.29."""
    return PW.moment_I0(cfg.to_params())


def compute_t_over_tau(cfg, Cw=None):
    """(t/tau)_equil = (I_yy - I_zz)/I_0 (Eqs. 5.31-5.32), shell + fossil."""
    return PW.t_over_tau_equilibrium(cfg.t20_plus_2t22 * 1e3, cfg.to_params(), Cw)


def compute_epsilon(cfg, Cw=None):
    """epsilon = (I_zz - I_xx)/I_0 (Eqs. 5.33-5.34), shell + fossil."""
    return PW.epsilon(cfg.t20_km * 1e3, cfg.t22_km * 1e3, cfg.to_params(), Cw)


def _viscosity(T, eta0, l, Tm):
    """Arrhenius viscosity (Eq. 5.2)."""
    return eta0 * np.exp(l * (Tm / T - 1.0))


def compute_C0(cfg):
    """Damping coefficient of an unfractured shell, Eq. 5.8 (eta_max at T_s, upper bound)."""
    a, bc, l = cfg.europa_radius, cfg.shell_thickness, cfg.activation_param
    Tm, Ts = cfg.melting_temp, cfg.surface_temp
    eta_max = _viscosity(Ts, cfg.melting_viscosity, l, Tm)
    return (4.0 * np.pi * a**2 * bc / l) * (Tm / (Tm - Ts)) * eta_max * cfg.gamma_tidal**2


def compute_Cf(cfg, Tf=None):
    """Fractured-shell damping coefficient C_f, Eq. 5.15 (with the ln 10)."""
    return PW.dissipation_cf(cfg.to_params(Tf))


def compute_tau_f(cfg, Tf=None):
    """Overdamped TPW time scale tau_f = [C_f tau/(n^2 I_0)]^(1/2), Eq. 5.41.

    Returns (tau_f [yr], tau_f [s]). tau is the growth parameter (~1e12 yr).
    """
    t_yr = PW.tau_f(cfg.to_params(Tf))
    return t_yr, t_yr * YEAR_S


def compute_undamped_timescale(cfg, Cw=None, psi0=np.exp(-1.0)):
    """Undamped TPW time scale, Eq. 4.22 [yr] (psi0 = 1/e drops the log factor)."""
    return PW.undamped_time_scale(compute_epsilon(cfg, Cw), psi0, cfg.to_params())


def compute_Maxwell_time(T, cfg):
    """Maxwell relaxation time eta(T)/mu [s] (Eq. 5.2 viscosity)."""
    eta = _viscosity(T, cfg.melting_viscosity, cfg.activation_param, cfg.melting_temp)
    return eta / cfg.ice_rigidity


# ---------------------------------------------------------------------------
# Evolution model
# ---------------------------------------------------------------------------

class OSPWEvolution:
    """OSPW pendulum for the tip angle psi (Eqs. 5.39-5.41).

    t/tau grows linearly, t/tau = t / tau_growth (tau ~ 1e12 yr, p. 262),
    from the time it passes through zero. Eq. 5.39:
        psi_ddot = 2 (t/tau) eps n^2 sin 2psi - 4 eps (C_f/I_0) psi_dot
    overdamped:
        psi_dot = (t/tau) (n^2 I_0 / 2 C_f) sin 2psi.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.p = cfg.to_params()
        self.Cw = PW.willemann_cw(self.p)
        self.I0 = PW.moment_I0(self.p)
        self.eps = compute_epsilon(cfg, self.Cw)
        self.Cf = PW.dissipation_cf(self.p)
        self.n = C.MEAN_MOTION
        self.tau_s = self.p.tau_growth_yr * YEAR_S

    def t_over_tau(self, t_s):
        return t_s / self.tau_s

    def overdamped_rate(self, t_s, psi):
        """psi_dot [rad/s] of the overdamped Eq. 5.39."""
        return (self.t_over_tau(t_s) * self.n**2 * self.I0 / (2.0 * self.Cf)
                * np.sin(2.0 * psi))

    def analytical_solution(self, t_arr, psi0):
        """Closed-form overdamped solution of Eq. 5.39 (t in years, psi in rad)."""
        return PW.psi_overdamped(t_arr, psi0, self.p)

    def run(self, t_max_yr, psi0_deg=1.0, n_points=2000):
        """Integrate the overdamped Eq. 5.39 (solve_ivp, tight tolerances)."""
        from scipy.integrate import solve_ivp
        t_max_s = t_max_yr * YEAR_S
        sol = solve_ivp(lambda t, y: [self.overdamped_rate(t, y[0])],
                        [0, t_max_s], [np.radians(psi0_deg)],
                        t_eval=np.linspace(0, t_max_s, n_points),
                        method='RK45', rtol=1e-10, atol=1e-14)
        if not sol.success:
            raise RuntimeError(f'overdamped integration failed: {sol.message}')
        return {'t_yr': sol.t / YEAR_S, 'psi_deg': np.degrees(sol.y[0]),
                'tau_f_yr': PW.tau_f(self.p)}

    def run_full(self, t_max_yr, psi0_deg=1.0, n_points=2000):
        """Integrate the full Eq. 5.39, including the inertial term."""
        from scipy.integrate import solve_ivp
        t_max_s = t_max_yr * YEAR_S
        damp = 4.0 * self.eps * self.Cf / self.I0

        def rhs(t, y):
            psi, psi_dot = y
            return [psi_dot, 2.0 * self.t_over_tau(t) * self.eps * self.n**2
                    * np.sin(2.0 * psi) - damp * psi_dot]

        sol = solve_ivp(rhs, [0, t_max_s], [np.radians(psi0_deg), 0.0],
                        t_eval=np.linspace(0, t_max_s, n_points),
                        method='Radau', rtol=1e-8, atol=1e-12)
        if not sol.success:
            raise RuntimeError(f'full integration failed: {sol.message}')
        return {'t_yr': sol.t / YEAR_S, 'psi_deg': np.degrees(sol.y[0]),
                'psi_dot': sol.y[1], 'tau_f_yr': PW.tau_f(self.p)}


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
    parser.add_argument('--Tf', type=float, default=124.0, help='Fracture base temp [K] (p. 264)')
    parser.add_argument('--bc', type=float, default=20.0, help='Shell thickness [km]')
    parser.add_argument('--d-lith', type=float, default=2.0, help='Lithosphere thickness [km]')
    parser.add_argument('--t20', type=float, default=-0.70, help='t20 [km]')
    parser.add_argument('--t22', type=float, default=0.24, help='t22 [km]')
    parser.add_argument('--eta0', type=float, default=3e14, help='Melting viscosity [Pa s]')
    parser.add_argument('--outdir', type=str, default='docs/figures/ospw',
                        help='Output directory for figures')
    parser.add_argument('--parameter-set', default='paper', choices=sorted(C.PARAMETER_SETS),
                        help="constants: 'paper' (validation) or 'modern' (science)")
    args = parser.parse_args()
    C.use_parameter_set(args.parameter_set)
    print(C.describe_active())

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
