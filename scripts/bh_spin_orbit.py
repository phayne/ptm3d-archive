#!/usr/bin/env python3
"""Burnett & Hayne (2023) coupled shell-interior spin-orbit dynamics.

Implements the Van Hoolst model for Europa's differentiated rotation:
  - Two coupled nonlinear pendula (shell + interior)
  - Gravity-gradient coupling K_G
  - MacDonald and Williams-Efroimsky tidal torques
  - Energy integral and zero-velocity curves
  - Critical coupling (K_G*) separating weak/strong regimes

Usage:
    python scripts/bh_spin_orbit.py                # Default: all figures
    python scripts/bh_spin_orbit.py --zvc           # Zero-velocity curves
    python scripts/bh_spin_orbit.py --weak-lock     # Weak coupling tidal locking
    python scripts/bh_spin_orbit.py --strong-lock   # Strong coupling tidal locking
    python scripts/bh_spin_orbit.py --poincare      # Poincare map
    python scripts/bh_spin_orbit.py --param-study   # Parameter study (kappa_i vs kappa_s)

Reference: Burnett & Hayne (2023), Icarus 406, 115731.
"""

import sys
import os
import argparse
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'python'))

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm, colors

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
class BHConfig:
    """Parameters for B&H spin-orbit model.

    Moments of inertia are normalized so C_i = 1.0.
    Angles are in radians, times in orbital periods.
    """
    # Moments of inertia (normalized, C_i = 1)
    C_i: float = 1.0
    gamma: float = 28.0     # C_i / C_s (Europa)
    kappa_i: float = 0.0035  # (B_i - A_i) / C_i
    kappa_s: float = 0.0015  # (B_s - A_s) / C_s

    # Coupling
    K_G_factor: float = 0.5   # K_G_hat / (B_i - A_i)

    # Tidal torque
    D_tilde_i_factor: float = 0.003  # D_tilde_i / K_G
    D_tilde_s_factor: float = 0.0015  # D_tilde_s / K_G
    tidal_model: str = 'williams'   # 'macdonald' or 'williams'
    Gamma: float = 0.04             # Williams-Efroimsky settling parameter

    # Orbital
    eccentricity: float = 0.0094
    mean_motion: float = C.MEAN_MOTION  # n [rad/s]

    @property
    def C_s(self):
        return self.C_i / self.gamma

    @property
    def B_i_minus_A_i(self):
        return self.kappa_i * self.C_i

    @property
    def B_s_minus_A_s(self):
        return self.kappa_s * self.C_s

    @property
    def K_G_hat(self):
        """Normalized coupling: K_G / n^2."""
        return self.K_G_factor * self.B_i_minus_A_i

    @property
    def K_G_critical(self):
        """Critical coupling K_G* = (3/2)(B_i - A_i) (Eq 37)."""
        return 1.5 * self.B_i_minus_A_i

    @property
    def is_weak_coupling(self):
        return self.K_G_hat < self.K_G_critical

    @property
    def D_tilde_i(self):
        return self.D_tilde_i_factor * self.K_G_hat

    @property
    def D_tilde_s(self):
        return self.D_tilde_s_factor * self.K_G_hat

    @property
    def kappa_star(self):
        """Classical tidal locking threshold from Goldreich (1966), Eq 7."""
        e = self.eccentricity
        return (2.0 / 3.0) * (9.5 * np.pi * e**2 / (2 * np.sqrt(2)))**2


# ---------------------------------------------------------------------------
# Energy and zero-velocity curves
# ---------------------------------------------------------------------------

def system_energy(eta_s, eta_i, eta_dot_s, eta_dot_i, cfg):
    """Normalized energy integral E_hat = E/n^2 (Eq 32).

    E_hat = (1/2)*C_i*eta_dot_i^2 + (1/2)*C_s*eta_dot_s^2
          - (3/4)*(B_i-A_i)*cos(2*eta_i) - (3/4)*(B_s-A_s)*cos(2*eta_s)
          - (1/2)*K_G_hat*cos(2*(eta_s - eta_i))
    """
    T = 0.5 * cfg.C_i * eta_dot_i**2 + 0.5 * cfg.C_s * eta_dot_s**2
    V = potential_energy(eta_s, eta_i, cfg)
    return T + V


def potential_energy(eta_s, eta_i, cfg):
    """Potential energy part of the normalized energy (Eq 32)."""
    V = (-(3.0 / 4.0) * cfg.B_i_minus_A_i * np.cos(2 * eta_i)
         - (3.0 / 4.0) * cfg.B_s_minus_A_s * np.cos(2 * eta_s)
         - 0.5 * cfg.K_G_hat * np.cos(2 * (eta_s - eta_i)))
    return V


def critical_energies(cfg):
    """Critical energy values (Eqs 35-36).

    E_crit,1: formed when ZVC intersects at eta_i = k*pi, eta_s = l*pi/2 (odd l)
    E_crit,2: formed at eta_i = k*pi/2 (odd k), eta_s = l*pi/2 (odd l)

    Returns (E_crit_1, E_crit_2).
    """
    BA_i = cfg.B_i_minus_A_i
    BA_s = cfg.B_s_minus_A_s
    K_G = cfg.K_G_hat

    E_crit_1 = -0.75 * BA_i + 0.75 * BA_s + 0.5 * K_G
    E_crit_2 = 0.75 * BA_i + 0.75 * BA_s - 0.5 * K_G

    return E_crit_1, E_crit_2


def zero_velocity_surface(eta_s_grid, eta_i_grid, cfg):
    """Compute potential energy V(eta_s, eta_i) on a 2D grid.

    The ZVC at energy E is the contour V = E.
    """
    ETA_S, ETA_I = np.meshgrid(eta_s_grid, eta_i_grid, indexing='ij')
    V = potential_energy(ETA_S, ETA_I, cfg)
    return V


# ---------------------------------------------------------------------------
# Dynamics
# ---------------------------------------------------------------------------

def rhs_averaged(t, y, cfg):
    """RHS of orbit-averaged equations (Eq 27).

    State: y = [eta_s, eta_i, eta_dot_s, eta_dot_i]
    (dimensionless time tau = n*t, eta_dot = d(eta)/d(tau))

    No tidal torques in the averaged system (energy is conserved).
    """
    eta_s, eta_i, eta_dot_s, eta_dot_i = y
    BA_s = cfg.B_s_minus_A_s
    BA_i = cfg.B_i_minus_A_i
    K_G = cfg.K_G_hat
    C_s = cfg.C_s
    C_i = cfg.C_i

    eta_ddot_s = (-(3.0 / 2.0) * BA_s * np.sin(2 * eta_s)
                  - K_G * np.sin(2 * (eta_s - eta_i))) / C_s
    eta_ddot_i = (-(3.0 / 2.0) * BA_i * np.sin(2 * eta_i)
                  + K_G * np.sin(2 * (eta_s - eta_i))) / C_i

    return [eta_dot_s, eta_dot_i, eta_ddot_s, eta_ddot_i]


def rhs_full(f, y, cfg):
    """RHS of unaveraged dynamics with true anomaly f as independent variable (Eq 38).

    State: y = [psi_s, psi_i, psi_dot_s, psi_dot_i]
    where psi_j = theta_j - f (libration angle), ()' = d/df.
    """
    psi_s, psi_i, psi_dot_s, psi_dot_i = y
    e = cfg.eccentricity
    kappa_s = cfg.kappa_s
    kappa_i = cfg.kappa_i
    K_G = cfg.K_G_hat
    C_s = cfg.C_s
    C_i = cfg.C_i

    # Orbital mechanics
    sin_f = np.sin(f)
    cos_f = np.cos(f)
    r_over_a = (1 - e**2) / (1 + e * cos_f)
    e_sin_f_term = 2 * (1 / r_over_a) * e * sin_f / (1 - e**2)
    grav_term = (1 / r_over_a) / (1 - e**2)

    # Specific angular momentum h = n*a^2*sqrt(1-e^2)
    # f_dot = h/r^2 = n*(1+e*cos_f)^2/(1-e^2)^(3/2)
    f_dot_sq_norm = (1 + e * cos_f)**4 / (1 - e**2)**3  # (f_dot/n)^2

    # Tidal torques
    L_s = tidal_torque(psi_dot_s, cfg, 'shell', f, e)
    L_i = tidal_torque(psi_dot_i, cfg, 'interior', f, e)

    coupling_sin = np.sin(2 * (psi_s - psi_i))

    psi_ddot_s = (e_sin_f_term * (1 + psi_dot_s)
                  - 1.5 * kappa_s * grav_term * np.sin(2 * psi_s)
                  - K_G * coupling_sin / (C_s * f_dot_sq_norm)
                  + L_s / (C_s * f_dot_sq_norm))

    psi_ddot_i = (e_sin_f_term * (1 + psi_dot_i)
                  - 1.5 * kappa_i * grav_term * np.sin(2 * psi_i)
                  + K_G * coupling_sin / (C_i * f_dot_sq_norm)
                  + L_i / (C_i * f_dot_sq_norm))

    return [psi_dot_s, psi_dot_i, psi_ddot_s, psi_ddot_i]


def tidal_torque(psi_dot, cfg, body, f=0, e=None):
    """Compute tidal torque for shell or interior.

    MacDonald (Eq 39): L = -D_tilde * (a/r)^6 * sign(psi_dot)
    Williams-Efroimsky (Eq 46): L = -D_tilde * (a/r)^6 * psi_dot / (n*Gamma)

    Parameters
    ----------
    psi_dot : float
        Libration rate d(psi)/df (dimensionless).
    cfg : BHConfig
    body : str
        'shell' or 'interior'.
    f : float
        True anomaly [rad].
    e : float, optional
        Eccentricity.
    """
    if e is None:
        e = cfg.eccentricity

    D_tilde = cfg.D_tilde_s if body == 'shell' else cfg.D_tilde_i

    # (a/r)^6
    cos_f = np.cos(f)
    a_over_r = (1 + e * cos_f) / (1 - e**2)
    a_over_r_6 = a_over_r**6

    if cfg.tidal_model == 'macdonald':
        # MacDonald: constant-magnitude torque (Eq 39)
        if abs(psi_dot) < 1e-15:
            return 0.0
        return -D_tilde * a_over_r_6 * np.sign(psi_dot)
    elif cfg.tidal_model == 'williams':
        # Williams-Efroimsky correction (Eq 46)
        return -D_tilde * a_over_r_6 * psi_dot / cfg.Gamma
    else:
        raise ValueError(f'Unknown tidal model: {cfg.tidal_model}')


# ---------------------------------------------------------------------------
# Evolution class
# ---------------------------------------------------------------------------

class BHEvolution:
    """Coupled shell-interior spin-orbit dynamics.

    State: [eta_s, eta_i, eta_dot_s, eta_dot_i] (averaged)
    or:    [psi_s, psi_i, psi_dot_s, psi_dot_i] (full dynamics)
    """

    def __init__(self, cfg):
        self.cfg = cfg

    def run_averaged(self, t_max_orbits, eta_s0=0.0, eta_i0=0.0,
                     eta_dot_s0=0.0, eta_dot_i0=0.0, n_points=5000):
        """Integrate orbit-averaged dynamics (Eq 27).

        No tidal torques — energy is conserved.
        Time is in units of orbital periods (tau = n*t, so t_orbits = tau/(2*pi)).
        """
        tau_max = t_max_orbits * 2 * np.pi  # dimensionless time
        t_eval = np.linspace(0, tau_max, n_points)

        def rhs_func(t, y):
            return rhs_averaged(t, y, self.cfg)

        y0 = [eta_s0, eta_i0, eta_dot_s0, eta_dot_i0]
        sol = solve_ivp(rhs_func, [0, tau_max], y0,
                        t_eval=t_eval, method='DOP853',
                        rtol=1e-12, atol=1e-14)

        t_orbits = sol.t / (2 * np.pi)
        E = np.array([system_energy(sol.y[0, i], sol.y[1, i],
                                     sol.y[2, i], sol.y[3, i], self.cfg)
                       for i in range(len(sol.t))])

        return {
            't_orbits': t_orbits,
            'eta_s': sol.y[0],
            'eta_i': sol.y[1],
            'eta_dot_s': sol.y[2],
            'eta_dot_i': sol.y[3],
            'energy': E,
        }

    def run_full(self, n_orbits, psi_s0=0.0, psi_i0=0.0,
                 psi_dot_s0=0.0, psi_dot_i0=0.0, n_points_per_orbit=100):
        """Integrate full unaveraged dynamics (Eq 38) with tidal torques.

        Independent variable is true anomaly f.
        """
        f_max = n_orbits * 2 * np.pi
        n_total = n_orbits * n_points_per_orbit
        f_eval = np.linspace(0, f_max, n_total)

        def rhs_func(f, y):
            return rhs_full(f, y, self.cfg)

        y0 = [psi_s0, psi_i0, psi_dot_s0, psi_dot_i0]
        sol = solve_ivp(rhs_func, [0, f_max], y0,
                        t_eval=f_eval, method='DOP853',
                        rtol=1e-10, atol=1e-13,
                        max_step=0.1)

        # Convert to orbital periods
        t_orbits = sol.t / (2 * np.pi)

        # Compute energy at each point (using averaged formula as approximation)
        # For the full dynamics, use psi as eta approximation
        E = np.array([system_energy(sol.y[0, i], sol.y[1, i],
                                     sol.y[2, i], sol.y[3, i], self.cfg)
                       for i in range(len(sol.t))])

        return {
            't_orbits': t_orbits,
            'f': sol.t,
            'psi_s': sol.y[0],
            'psi_i': sol.y[1],
            'psi_dot_s': sol.y[2],
            'psi_dot_i': sol.y[3],
            'energy': E,
        }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_zero_velocity_curves(cfg, save_path=None):
    """Plot zero-velocity curves for weak and strong coupling (Fig 4).

    Shows potential energy contours that bound accessible states.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    eta_grid = np.linspace(-np.pi, np.pi, 400)
    E_c1, E_c2 = critical_energies(cfg)

    # Left: weak coupling
    cfg_weak = BHConfig(K_G_factor=0.5 * 1.5,  # < K_G*
                        kappa_i=cfg.kappa_i, kappa_s=cfg.kappa_s,
                        gamma=cfg.gamma)
    E_c1_w, E_c2_w = critical_energies(cfg_weak)
    V_weak = zero_velocity_surface(eta_grid, eta_grid, cfg_weak)

    ax = axes[0]
    # Energy levels from below E_c1 to above E_c2
    E_min = potential_energy(0, 0, cfg_weak)
    levels = [E_min * 0.8, E_c1_w, 0.5 * (E_c1_w + E_c2_w), E_c2_w, E_c2_w * 1.5]
    levels = sorted(set(levels))
    cs = ax.contour(np.degrees(eta_grid), np.degrees(eta_grid), V_weak.T,
                    levels=levels, colors=['blue', 'red', 'orange', 'green', 'purple'],
                    linewidths=[1, 2, 1, 2, 1])
    ax.set_xlabel('$\\eta_s$ [deg]')
    ax.set_ylabel('$\\eta_i$ [deg]')
    ax.set_title(f'Weak coupling ($\\hat{{K}}_G$ = {cfg_weak.K_G_factor:.1f}$(B_i-A_i)$)')
    ax.set_aspect('equal')
    ax.axhline(0, ls=':', color='gray', alpha=0.3)
    ax.axvline(0, ls=':', color='gray', alpha=0.3)
    ax.grid(True, alpha=0.2)

    # Right: strong coupling
    cfg_strong = BHConfig(K_G_factor=2.5,  # > K_G*
                          kappa_i=cfg.kappa_i, kappa_s=cfg.kappa_s,
                          gamma=cfg.gamma)
    E_c1_s, E_c2_s = critical_energies(cfg_strong)
    V_strong = zero_velocity_surface(eta_grid, eta_grid, cfg_strong)

    ax = axes[1]
    E_min_s = potential_energy(0, 0, cfg_strong)
    levels_s = [E_min_s * 0.8, E_c2_s, 0.5 * (E_c1_s + E_c2_s), E_c1_s, E_c1_s * 1.5]
    levels_s = sorted(set(levels_s))
    cs = ax.contour(np.degrees(eta_grid), np.degrees(eta_grid), V_strong.T,
                    levels=levels_s, colors=['blue', 'green', 'orange', 'red', 'purple'],
                    linewidths=[1, 2, 1, 2, 1])
    ax.set_xlabel('$\\eta_s$ [deg]')
    ax.set_ylabel('$\\eta_i$ [deg]')
    ax.set_title(f'Strong coupling ($\\hat{{K}}_G$ = {cfg_strong.K_G_factor:.1f}$(B_i-A_i)$)')
    ax.set_aspect('equal')
    ax.axhline(0, ls=':', color='gray', alpha=0.3)
    ax.axvline(0, ls=':', color='gray', alpha=0.3)
    ax.grid(True, alpha=0.2)

    plt.suptitle('Zero-Velocity Curves (B&H Fig 4)', fontsize=13)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f'Saved: {save_path}')
    return fig


def plot_tidal_locking_weak(cfg, save_path=None):
    """Simulate weak coupling tidal locking (B&H Fig 10-11).

    Initially super-synchronous rotation. Core locks first, then shell.
    """
    cfg_weak = BHConfig(
        K_G_factor=0.1, kappa_i=0.0035, kappa_s=0.0015,
        gamma=28.0, eccentricity=0.0094,
        D_tilde_i_factor=0.003, D_tilde_s_factor=0.0015,
        tidal_model='williams', Gamma=0.04,
    )

    evol = BHEvolution(cfg_weak)
    # Start super-synchronous: psi_dot > 0 means theta_dot > f_dot (faster than sync)
    n = cfg_weak.mean_motion
    psi_dot_s0 = 0.5  # dimensionless: d(psi)/df
    psi_dot_i0 = 0.1

    result = evol.run_full(
        n_orbits=2000,
        psi_s0=np.radians(-103),
        psi_i0=np.radians(-81),
        psi_dot_s0=psi_dot_s0,
        psi_dot_i0=psi_dot_i0,
        n_points_per_orbit=50,
    )

    E_c1, E_c2 = critical_energies(cfg_weak)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    ax = axes[0]
    ax.plot(result['t_orbits'], np.degrees(result['psi_s']), 'b-', alpha=0.5, linewidth=0.3, label='Shell')
    ax.plot(result['t_orbits'], np.degrees(result['psi_i']), 'r-', alpha=0.5, linewidth=0.3, label='Interior')
    ax.set_ylabel('$\\psi$ [deg]')
    ax.legend(fontsize=8)
    ax.set_title(f'Weak Coupling: $\\hat{{K}}_G$ = {cfg_weak.K_G_factor}$(B_i-A_i)$ < $\\hat{{K}}_G^*$')

    ax = axes[1]
    ax.plot(result['t_orbits'], result['psi_dot_s'], 'b-', alpha=0.5, linewidth=0.3, label='Shell')
    ax.plot(result['t_orbits'], result['psi_dot_i'], 'r-', alpha=0.5, linewidth=0.3, label='Interior')
    ax.axhline(0, ls=':', color='gray')
    ax.set_ylabel("$\\psi'$")
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.plot(result['t_orbits'], result['energy'], 'k-', linewidth=0.5)
    ax.axhline(E_c1, ls='--', color='blue', alpha=0.7, label=f'$\\hat{{E}}_{{crit,1}}$ = {E_c1:.4f}')
    ax.axhline(E_c2, ls='--', color='red', alpha=0.7, label=f'$\\hat{{E}}_{{crit,2}}$ = {E_c2:.4f}')
    ax.set_ylabel('$\\hat{E}$')
    ax.set_xlabel('Time [orbits]')
    ax.legend(fontsize=8)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f'Saved: {save_path}')
    return fig


def plot_tidal_locking_strong(cfg, save_path=None):
    """Simulate strong coupling tidal locking (B&H Fig 12).

    Shell and core lock simultaneously.
    """
    cfg_strong = BHConfig(
        K_G_factor=2.5, kappa_i=0.0035, kappa_s=0.0015,
        gamma=28.0, eccentricity=0.0094,
        D_tilde_i_factor=0.005, D_tilde_s_factor=0.0005,
        tidal_model='williams', Gamma=0.04,
    )

    evol = BHEvolution(cfg_strong)
    result = evol.run_full(
        n_orbits=800,
        psi_s0=np.radians(10.5),
        psi_i0=0.0,
        psi_dot_s0=0.1,
        psi_dot_i0=0.5,
        n_points_per_orbit=50,
    )

    E_c1, E_c2 = critical_energies(cfg_strong)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    ax = axes[0]
    ax.plot(result['t_orbits'], np.degrees(result['psi_s']), 'b-', alpha=0.5, linewidth=0.3, label='Shell')
    ax.plot(result['t_orbits'], np.degrees(result['psi_i']), 'r-', alpha=0.5, linewidth=0.3, label='Interior')
    ax.set_ylabel('$\\psi$ [deg]')
    ax.legend(fontsize=8)
    ax.set_title(f'Strong Coupling: $\\hat{{K}}_G$ = {cfg_strong.K_G_factor}$(B_i-A_i)$ > $\\hat{{K}}_G^*$')

    ax = axes[1]
    ax.plot(result['t_orbits'], result['psi_dot_s'], 'b-', alpha=0.5, linewidth=0.3, label='Shell')
    ax.plot(result['t_orbits'], result['psi_dot_i'], 'r-', alpha=0.5, linewidth=0.3, label='Interior')
    ax.axhline(0, ls=':', color='gray')
    ax.set_ylabel("$\\psi'$")
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.plot(result['t_orbits'], result['energy'], 'k-', linewidth=0.5)
    ax.axhline(E_c1, ls='--', color='blue', alpha=0.7, label=f'$\\hat{{E}}_{{crit,1}}$ = {E_c1:.4f}')
    ax.axhline(E_c2, ls='--', color='red', alpha=0.7, label=f'$\\hat{{E}}_{{crit,2}}$ = {E_c2:.4f}')
    ax.set_ylabel('$\\hat{E}$')
    ax.set_xlabel('Time [orbits]')
    ax.legend(fontsize=8)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f'Saved: {save_path}')
    return fig


def plot_bounded_librations(cfg, save_path=None):
    """Conservative dynamics: bounded librations (B&H Fig 6 analog).

    No tidal torques. Energy is conserved. Both shell and interior librate.
    """
    cfg_cons = BHConfig(K_G_factor=2.5, kappa_i=0.0035, kappa_s=0.0015, gamma=28.0)
    evol = BHEvolution(cfg_cons)

    # Small initial displacement: both should librate
    result = evol.run_averaged(
        t_max_orbits=500,
        eta_s0=np.radians(15),
        eta_i0=np.radians(5),
        eta_dot_s0=0.0,
        eta_dot_i0=0.0,
        n_points=10000,
    )

    E_c1, E_c2 = critical_energies(cfg_cons)

    fig, axes = plt.subplots(2, 2, figsize=(12, 6))

    ax = axes[0, 0]
    ax.plot(result['t_orbits'], np.degrees(result['eta_s']), 'b-', linewidth=0.5, label='Shell')
    ax.plot(result['t_orbits'], np.degrees(result['eta_i']), 'r-', linewidth=0.5, label='Interior')
    ax.set_ylabel('$\\eta$ [deg]')
    ax.set_xlabel('Time [orbits]')
    ax.legend(fontsize=8)
    ax.set_title('Angle evolution')

    ax = axes[0, 1]
    ax.plot(result['t_orbits'], result['eta_dot_s'], 'b-', linewidth=0.5, label='Shell')
    ax.plot(result['t_orbits'], result['eta_dot_i'], 'r-', linewidth=0.5, label='Interior')
    ax.set_ylabel("$\\eta'$")
    ax.set_xlabel('Time [orbits]')
    ax.legend(fontsize=8)
    ax.set_title('Angular velocity')

    ax = axes[1, 0]
    ax.plot(result['t_orbits'], result['energy'], 'k-', linewidth=0.5)
    ax.axhline(E_c1, ls='--', color='blue', alpha=0.5, label=f'$E_{{c1}}$={E_c1:.5f}')
    ax.axhline(E_c2, ls='--', color='red', alpha=0.5, label=f'$E_{{c2}}$={E_c2:.5f}')
    ax.set_ylabel('$\\hat{E}$')
    ax.set_xlabel('Time [orbits]')
    ax.legend(fontsize=7)
    ax.set_title('Energy (conserved)')

    ax = axes[1, 1]
    ax.plot(np.degrees(result['eta_s']), np.degrees(result['eta_i']),
            'k-', linewidth=0.3, alpha=0.5)
    ax.set_xlabel('$\\eta_s$ [deg]')
    ax.set_ylabel('$\\eta_i$ [deg]')
    ax.set_title('Phase space ($\\eta_s$ vs $\\eta_i$)')
    ax.set_aspect('equal')

    plt.suptitle('Bounded Librations (no tidal torques)', fontsize=13)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f'Saved: {save_path}')
    return fig


def plot_poincare_map(cfg, save_path=None):
    """Poincare map for weak coupling (B&H Fig 9 analog).

    Poincare surface: positive crossings of eta_i = 0 (eta_dot_i > 0).
    """
    cfg_weak = BHConfig(K_G_factor=0.5 * 1.5, kappa_i=0.0035, kappa_s=0.0015, gamma=28.0)
    E_c1, E_c2 = critical_energies(cfg_weak)
    E_poincare = 0.5 * (E_c1 + E_c2)

    evol = BHEvolution(cfg_weak)
    fig, ax = plt.subplots(figsize=(7, 5))

    # Sample different initial conditions at constant energy
    eta_s_inits = np.linspace(0, 2 * np.pi, 30)
    n_orbits = 2000

    for eta_s0 in eta_s_inits:
        # Find eta_dot_i from energy: E = (1/2)*C_i*eta_dot_i^2 + V
        V = potential_energy(eta_s0, 0, cfg_weak)
        KE_i = E_poincare - V
        if KE_i < 0:
            continue
        eta_dot_i0 = np.sqrt(2 * KE_i / cfg_weak.C_i)

        result = evol.run_averaged(
            t_max_orbits=n_orbits,
            eta_s0=eta_s0, eta_i0=0.0,
            eta_dot_s0=0.0, eta_dot_i0=eta_dot_i0,
            n_points=n_orbits * 50,
        )

        # Find Poincare crossings: eta_i crosses 0 with eta_dot_i > 0
        eta_i = result['eta_i']
        crossings = []
        for j in range(1, len(eta_i)):
            # Detect zero crossing (mod 2pi)
            ei_prev = eta_i[j - 1] % (2 * np.pi)
            ei_curr = eta_i[j] % (2 * np.pi)
            # Crossing at 0 or 2pi
            if (ei_prev > np.pi and ei_curr < np.pi) or (ei_prev < 0.1 and ei_curr > 2 * np.pi - 0.1):
                continue  # wrong direction
            if (ei_prev > 2 * np.pi - 0.3 and ei_curr < 0.3) or \
               (ei_prev < 0.3 and ei_curr > 2 * np.pi - 0.3):
                continue  # wrapping artifact
            if abs(ei_curr) < 0.3 or abs(ei_curr - 2 * np.pi) < 0.3:
                if result['eta_dot_i'][j] > 0:
                    crossings.append(j)

        if crossings:
            es_cross = result['eta_s'][crossings] % (2 * np.pi)
            eds_cross = result['eta_dot_s'][crossings]
            ax.plot(np.degrees(es_cross), eds_cross, '.', markersize=0.5,
                    alpha=0.5, color='k')

    ax.set_xlabel('$\\eta_s$ [deg]')
    ax.set_ylabel("$\\eta'_s$")
    ax.set_title(f'Poincar\u00e9 Map (weak coupling, $E$ = {E_poincare:.5f})')
    ax.set_xlim(0, 360)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f'Saved: {save_path}')
    return fig


def plot_parameter_study(cfg, save_path=None):
    """Parameter study: equilibrium spin state vs kappa_i, kappa_s (B&H Fig 16).

    Runs many simulations to map tidal locking vs NSR boundary.
    """
    kappa_star = cfg.kappa_star
    n_grid = 12  # reduced for speed
    kappa_i_range = np.linspace(0.2, 3.0, n_grid) * kappa_star
    kappa_s_range = np.linspace(0.2, 6.0, n_grid) * kappa_star

    E_loss = np.zeros((n_grid, n_grid))

    for i, ki in enumerate(kappa_i_range):
        for j, ks in enumerate(kappa_s_range):
            cfg_run = BHConfig(
                kappa_i=ki, kappa_s=ks, gamma=28.0,
                K_G_factor=0.6, eccentricity=0.0094,
                D_tilde_i_factor=0.01, D_tilde_s_factor=0.005,
                tidal_model='williams', Gamma=0.04,
            )
            evol = BHEvolution(cfg_run)
            try:
                result = evol.run_full(
                    n_orbits=500,
                    psi_s0=0.0, psi_i0=0.0,
                    psi_dot_s0=0.3, psi_dot_i0=0.3,
                    n_points_per_orbit=30,
                )
                E0 = result['energy'][0]
                Ef = result['energy'][-1]
                E_c1, _ = critical_energies(cfg_run)
                denom = E0 - E_c1
                if abs(denom) > 1e-15:
                    E_loss[i, j] = (Ef - E0) / denom
                else:
                    E_loss[i, j] = 0
            except Exception:
                E_loss[i, j] = np.nan

    fig, ax = plt.subplots(figsize=(7, 5))

    ki_norm = kappa_i_range / kappa_star
    ks_norm = kappa_s_range / kappa_star

    # Locked: dE/(E0-Ec1) < -1
    locked = E_loss < -1
    cmap = plt.cm.RdYlGn_r
    im = ax.pcolormesh(ks_norm, ki_norm, E_loss, cmap=cmap,
                       vmin=-2, vmax=0, shading='auto')
    plt.colorbar(im, ax=ax, label='$\\Delta E / (E_0 - E_{c,1})$')

    # Contour at -1 (locking boundary)
    try:
        ax.contour(ks_norm, ki_norm, E_loss, levels=[-1], colors='k', linewidths=2)
    except Exception:
        pass

    ax.set_xlabel('$\\kappa_s / \\kappa^*$')
    ax.set_ylabel('$\\kappa_i / \\kappa^*$')
    ax.set_title('Equilibrium Spin State (B&H Fig 16 analog)')
    ax.axhline(1, ls=':', color='gray', alpha=0.5)
    ax.axvline(1, ls=':', color='gray', alpha=0.5)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f'Saved: {save_path}')
    return fig


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def print_diagnostics(cfg):
    """Print B&H model parameters."""
    E_c1, E_c2 = critical_energies(cfg)

    print('=' * 60)
    print('B&H Spin-Orbit Coupling Diagnostics')
    print('=' * 60)
    print(f'\nMoments of inertia:')
    print(f'  C_i = {cfg.C_i:.4f} (normalized)')
    print(f'  C_s = {cfg.C_s:.6f}')
    print(f'  gamma = C_i/C_s = {cfg.gamma:.1f}')
    print(f'  kappa_i = {cfg.kappa_i:.4f}')
    print(f'  kappa_s = {cfg.kappa_s:.4f}')
    print(f'  B_i - A_i = {cfg.B_i_minus_A_i:.6f}')
    print(f'  B_s - A_s = {cfg.B_s_minus_A_s:.6e}')
    print(f'\nCoupling:')
    print(f'  K_G_hat = {cfg.K_G_hat:.6f} = {cfg.K_G_factor:.1f}*(B_i-A_i)')
    print(f'  K_G* = {cfg.K_G_critical:.6f} = 1.5*(B_i-A_i)')
    print(f'  Regime: {"WEAK" if cfg.is_weak_coupling else "STRONG"} coupling')
    print(f'\nTidal torques:')
    print(f'  Model: {cfg.tidal_model}')
    print(f'  D_tilde_i = {cfg.D_tilde_i:.6e}')
    print(f'  D_tilde_s = {cfg.D_tilde_s:.6e}')
    if cfg.tidal_model == 'williams':
        print(f'  Gamma = {cfg.Gamma}')
    print(f'\nCritical energies:')
    print(f'  E_crit,1 = {E_c1:.6f}')
    print(f'  E_crit,2 = {E_c2:.6f}')
    print(f'  kappa* (Goldreich) = {cfg.kappa_star:.6e}')
    print(f'  kappa_i/kappa* = {cfg.kappa_i/cfg.kappa_star:.2f}')
    print(f'  kappa_s/kappa* = {cfg.kappa_s/cfg.kappa_star:.2f}')
    print('=' * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='B&H spin-orbit coupling model')
    parser.add_argument('--zvc', action='store_true', help='Zero-velocity curves')
    parser.add_argument('--weak-lock', action='store_true', help='Weak coupling locking')
    parser.add_argument('--strong-lock', action='store_true', help='Strong coupling locking')
    parser.add_argument('--bounded', action='store_true', help='Bounded librations')
    parser.add_argument('--poincare', action='store_true', help='Poincare map')
    parser.add_argument('--param-study', action='store_true', help='Parameter study')
    parser.add_argument('--all', action='store_true', help='Run all analyses')
    parser.add_argument('--outdir', type=str, default='docs/figures/bh',
                        help='Output directory')
    args = parser.parse_args()

    cfg = BHConfig()
    os.makedirs(args.outdir, exist_ok=True)
    print_diagnostics(cfg)

    do_all = args.all or not any([args.zvc, args.weak_lock, args.strong_lock,
                                   args.bounded, args.poincare, args.param_study])

    if args.zvc or do_all:
        plot_zero_velocity_curves(cfg, save_path=os.path.join(args.outdir, 'bh_zvc.png'))

    if args.bounded or do_all:
        plot_bounded_librations(cfg, save_path=os.path.join(args.outdir, 'bh_bounded_librations.png'))

    if args.weak_lock or do_all:
        plot_tidal_locking_weak(cfg, save_path=os.path.join(args.outdir, 'bh_weak_locking.png'))

    if args.strong_lock or do_all:
        plot_tidal_locking_strong(cfg, save_path=os.path.join(args.outdir, 'bh_strong_locking.png'))

    if args.poincare or do_all:
        plot_poincare_map(cfg, save_path=os.path.join(args.outdir, 'bh_poincare.png'))

    if args.param_study or do_all:
        plot_parameter_study(cfg, save_path=os.path.join(args.outdir, 'bh_param_study.png'))

    plt.close('all')
    print('\nDone.')


if __name__ == '__main__':
    main()
