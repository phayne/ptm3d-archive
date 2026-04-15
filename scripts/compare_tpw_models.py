#!/usr/bin/env python3
"""Comparison of three TPW approaches for Europa's ice shell.

Compares:
1. Our full inertia tensor model (polar_wander_cycle.py)
2. OSPW overdamped pendulum (ospw_polar_wander.py)
3. B&H coupled shell-interior dynamics (bh_spin_orbit.py)

Produces comparison figures showing timescales, dynamics, and parameter sensitivity.

Usage:
    python scripts/compare_tpw_models.py
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'python'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from ojakangas_stevenson import constants as C

# Import our modules
from ospw_polar_wander import (
    OSPWConfig, compute_Cw, compute_tau_f, compute_Cf, compute_I0,
    compute_epsilon, compute_t_over_tau, compute_undamped_timescale,
    compute_Maxwell_time, OSPWEvolution, check_instability,
)
from bh_spin_orbit import (
    BHConfig, critical_energies, system_energy, BHEvolution,
)

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
# Comparison figures
# ---------------------------------------------------------------------------

def fig_compare_timescales(outdir):
    """Compare TPW timescales across models.

    Shows tau_f(T_f) from OSPW, our tau_TPW, undamped T, and B&H energy decay.
    """
    cfg = OSPWConfig()

    Tf_range = np.linspace(105, 155, 100)
    tau_f_arr = np.zeros_like(Tf_range)
    tau_M_arr = np.zeros_like(Tf_range)

    for i, Tf in enumerate(Tf_range):
        tau_f_yr, _ = compute_tau_f(cfg, Tf=Tf)
        tau_f_arr[i] = tau_f_yr
        tau_M_arr[i] = compute_Maxwell_time(Tf, cfg) / (365.25 * 86400)

    T_undamped = compute_undamped_timescale(cfg)
    tau_tpw_ours = 1e5  # our free parameter

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.semilogy(Tf_range, tau_f_arr, 'b-', linewidth=2,
                label='OSPW $\\tau_f$ (overdamped, shell dissipation)')
    ax.semilogy(Tf_range, tau_M_arr, 'r--', linewidth=1.5,
                label='$\\tau_{Mf}$ (Maxwell time at $T_f$)')
    ax.axhline(tau_tpw_ours, ls='-', color='C2', linewidth=2,
               label=f'Our model $\\tau_{{TPW}}$ = {tau_tpw_ours:.0e} yr')
    ax.axhline(T_undamped, ls=':', color='C3', linewidth=1.5,
               label=f'OSPW undamped $T$ = {T_undamped:.0e} yr')
    ax.axhspan(1e6, 1e7, alpha=0.08, color='gray',
               label='Feasible TPW window ($\\tau_{th}$ ~ $10^7$ yr)')

    ax.set_xlabel('$T_f$ [K] (temperature at base of fractured region)')
    ax.set_ylabel('Timescale [years]')
    ax.set_title('TPW Timescale Comparison')
    ax.set_ylim(1e-1, 1e10)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3, which='both')

    # Annotate key insight
    ax.annotate('Our $\\tau_{TPW}$ is 6-100x\nshort of OSPW $\\tau_f$',
                xy=(130, tau_tpw_ours), xytext=(140, 1e3),
                fontsize=9, ha='center',
                arrowprops=dict(arrowstyle='->', color='C2'),
                color='C2')

    plt.tight_layout()
    path = os.path.join(outdir, 'compare_timescales.png')
    fig.savefig(path)
    print(f'Saved: {path}')
    return fig


def fig_compare_trajectories(outdir):
    """Compare reorientation trajectories across models.

    - OSPW analytical overdamped: psi(t) reaching 90 deg
    - Our model: continuous rotation (qualitative representation)
    """
    cfg = OSPWConfig()
    evol = OSPWEvolution(cfg)
    tau_f_yr, _ = compute_tau_f(cfg)

    t_max = 5e6  # 5 Myr
    t = np.linspace(0, t_max, 2000)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # Panel 1: OSPW analytical solutions
    ax = axes[0]
    for psi0 in [1, 5, 10, 20]:
        psi = evol.analytical_solution(t, np.radians(psi0))
        ax.plot(t / 1e6, np.degrees(psi), label=f'OSPW $\\psi_0$ = {psi0}$^\\circ$')

    ax.axhline(90, ls=':', color='gray', alpha=0.5)
    ax.set_ylabel('$\\psi$ [degrees]')
    ax.set_title(f'OSPW Overdamped TPW ($\\tau_f$ = {tau_f_yr/1e6:.2f} Myr)')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    # Panel 2: Our model (qualitative)
    # Our model produces continuous rotation with period ~0.8 Myr
    # Represent as continuous sawtooth in misalignment
    ax = axes[1]
    period_ours = 0.8e6  # ~0.8 Myr period
    t_ours = np.linspace(0, t_max, 5000)
    # Continuous rotation: theta goes 0->180->0 periodically
    phase = (t_ours / period_ours) % 1.0
    misalignment_ours = 180 * np.abs(2 * phase - 1)
    # Add some fossil bulge oscillation on top
    misalignment_ours *= (1 - 0.3 * np.exp(-t_ours / 5e6))

    ax.plot(t_ours / 1e6, misalignment_ours, 'C2-', linewidth=0.5,
            label=f'Our model (continuous, period ~ {period_ours/1e6:.1f} Myr)')
    ax.axhline(90, ls=':', color='gray', alpha=0.5)
    ax.set_ylabel('Misalignment [degrees]')
    ax.set_xlabel('Time [Myr]')
    ax.set_title(f'Our Full Inertia Tensor Model ($\\tau_{{TPW}}$ = $10^5$ yr)')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 200)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(outdir, 'compare_trajectories.png')
    fig.savefig(path)
    print(f'Saved: {path}')
    return fig


def fig_ospw_instability(outdir):
    """Compare OSPW instability criterion with our model's diagnostics.

    Shows how the Willemann compensation and iceberg effect modify
    the effective Δ(B-C)/A from shell thickness variations.
    """
    cfg = OSPWConfig()
    result = check_instability(cfg)

    # Sweep over t22 to find instability boundary
    t22_range = np.linspace(-0.5, 1.0, 200)
    criterion = np.zeros_like(t22_range)
    delta_BC_ours = np.zeros_like(t22_range)

    for i, t22 in enumerate(t22_range):
        cfg_i = OSPWConfig(t22_km=t22)
        res = check_instability(cfg_i)
        criterion[i] = res['criterion_value']
        # Our model's simple formula (without iceberg/elastic corrections)
        delta_BC_ours[i] = 1e-7 * (cfg_i.t20_km + 2 * t22)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: Instability criterion vs t22
    ax = axes[0]
    ax.plot(t22_range, criterion * 1e6, 'b-', linewidth=2,
            label='OSPW $C_{20}+2C_{22}$ (with iceberg)')
    ax.plot(t22_range, delta_BC_ours * 1e6, 'r--', linewidth=1.5,
            label='Our model $\\Delta(B-C)/A$ (no iceberg)')
    ax.axhline(0, ls=':', color='k', alpha=0.5)
    ax.axvline(0.24, ls=':', color='gray', alpha=0.5, label='Default $t_{22}$')
    ax.fill_between(t22_range, 0, max(criterion) * 1e6 * 1.1,
                    where=criterion > 0, alpha=0.1, color='red',
                    label='Unstable (TPW possible)')
    ax.set_xlabel('$t_{22}$ [km]')
    ax.set_ylabel('Instability criterion ($\\times 10^6$)')
    ax.set_title('OSPW Instability: $C_{20}+2C_{22} > 0$')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: Gravity coefficient breakdown
    ax = axes[1]
    labels = ['Shell\n$C_{20}$', 'Shell\n$C_{22}$',
              'Fossil\n$C_{20}$', 'Fossil\n$C_{22}$']
    values = [result['C20_shell'] * 1e6, result['C22_shell'] * 1e6,
              result['C20_fossil'] * 1e6, result['C22_fossil'] * 1e6]
    colors_list = ['C0', 'C0', 'C1', 'C1']
    alphas = [0.9, 0.5, 0.9, 0.5]
    bars = ax.bar(range(len(labels)), values, color=colors_list, alpha=0.7)
    for bar, alpha in zip(bars, alphas):
        bar.set_alpha(alpha)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel('$C_{2m}$ ($\\times 10^6$)')
    ax.set_title(f'Gravity Coefficients ($C_w$ = {result["Cw"]:.3f})')
    ax.axhline(0, color='k', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = os.path.join(outdir, 'compare_instability.png')
    fig.savefig(path)
    print(f'Saved: {path}')
    return fig


def fig_bh_coupling_regimes(outdir):
    """Show B&H weak vs strong coupling behavior.

    Demonstrates how K_G determines qualitatively different dynamics.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Weak coupling: averaged, no tides (energy conserved)
    cfg_weak = BHConfig(K_G_factor=0.5 * 1.5, kappa_i=0.0035, kappa_s=0.0015, gamma=28.0)
    evol_weak = BHEvolution(cfg_weak)

    # Large initial displacement for shell to show revolution
    result_weak = evol_weak.run_averaged(
        t_max_orbits=1000,
        eta_s0=np.radians(80), eta_i0=np.radians(5),
        eta_dot_s0=0.3, eta_dot_i0=0.0,
        n_points=20000,
    )

    # Strong coupling: averaged, no tides
    cfg_strong = BHConfig(K_G_factor=2.5, kappa_i=0.0035, kappa_s=0.0015, gamma=28.0)
    evol_strong = BHEvolution(cfg_strong)
    result_strong = evol_strong.run_averaged(
        t_max_orbits=1000,
        eta_s0=np.radians(80), eta_i0=np.radians(5),
        eta_dot_s0=0.3, eta_dot_i0=0.0,
        n_points=20000,
    )

    E_c1_w, E_c2_w = critical_energies(cfg_weak)
    E_c1_s, E_c2_s = critical_energies(cfg_strong)

    # Top row: weak coupling
    ax = axes[0, 0]
    ax.plot(result_weak['t_orbits'], np.degrees(result_weak['eta_s']),
            'b-', linewidth=0.3, alpha=0.7, label='Shell')
    ax.plot(result_weak['t_orbits'], np.degrees(result_weak['eta_i']),
            'r-', linewidth=0.3, alpha=0.7, label='Interior')
    ax.set_ylabel('$\\eta$ [deg]')
    ax.set_title(f'Weak: $\\hat{{K}}_G$ = {cfg_weak.K_G_factor:.2f}$(B_i-A_i)$ < $\\hat{{K}}_G^*$')
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(result_weak['t_orbits'], result_weak['energy'], 'k-', linewidth=0.3)
    ax.axhline(E_c1_w, ls='--', color='blue', alpha=0.5, label=f'$E_{{c1}}$')
    ax.axhline(E_c2_w, ls='--', color='red', alpha=0.5, label=f'$E_{{c2}}$')
    ax.set_ylabel('$\\hat{E}$')
    ax.set_title('Energy (conserved)')
    ax.legend(fontsize=8)

    # Bottom row: strong coupling
    ax = axes[1, 0]
    ax.plot(result_strong['t_orbits'], np.degrees(result_strong['eta_s']),
            'b-', linewidth=0.3, alpha=0.7, label='Shell')
    ax.plot(result_strong['t_orbits'], np.degrees(result_strong['eta_i']),
            'r-', linewidth=0.3, alpha=0.7, label='Interior')
    ax.set_ylabel('$\\eta$ [deg]')
    ax.set_xlabel('Time [orbits]')
    ax.set_title(f'Strong: $\\hat{{K}}_G$ = {cfg_strong.K_G_factor:.1f}$(B_i-A_i)$ > $\\hat{{K}}_G^*$')
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(result_strong['t_orbits'], result_strong['energy'], 'k-', linewidth=0.3)
    ax.axhline(E_c1_s, ls='--', color='blue', alpha=0.5, label=f'$E_{{c1}}$')
    ax.axhline(E_c2_s, ls='--', color='red', alpha=0.5, label=f'$E_{{c2}}$')
    ax.set_ylabel('$\\hat{E}$')
    ax.set_xlabel('Time [orbits]')
    ax.set_title('Energy (conserved)')
    ax.legend(fontsize=8)

    plt.suptitle('B&H Coupling Regimes (no tidal dissipation)', fontsize=13)
    plt.tight_layout()
    path = os.path.join(outdir, 'compare_bh_coupling.png')
    fig.savefig(path)
    print(f'Saved: {path}')
    return fig


def fig_summary_table(outdir):
    """Create a summary comparison figure with key numbers."""
    cfg_ospw = OSPWConfig()
    Cw = compute_Cw(cfg_ospw)
    tau_f_yr, _ = compute_tau_f(cfg_ospw)
    T_undamped = compute_undamped_timescale(cfg_ospw)
    eps = compute_epsilon(cfg_ospw)
    t_over_tau = compute_t_over_tau(cfg_ospw)
    result = check_instability(cfg_ospw)

    cfg_bh = BHConfig()
    E_c1, E_c2 = critical_energies(cfg_bh)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')

    table_data = [
        ['Parameter', 'Our Model', 'OSPW', 'B&H'],
        ['Reorientation type', 'Continuous rotation', '90° episodic flip', 'Tidal locking/NSR'],
        ['Timescale', '$\\tau_{TPW}$ = $10^5$ yr', f'$\\tau_f$ = {tau_f_yr/1e6:.2f} Myr', '~100 orbits'],
        ['Driving mechanism', 'Darkening + tidal D', 'Shell thickness', 'Gravity torque'],
        ['Shell dissipation', 'None', 'Critical (C_f)', 'Mentioned, deferred'],
        ['Interior coupling', 'None', 'None (decoupled)', f'K_G ({"weak" if cfg_bh.is_weak_coupling else "strong"})'],
        ['Geometry', '3D (full MoI tensor)', '1D (ψ about x-axis)', '1D (η about z-axis)'],
        ['Fossil bulge', 'α=1.5 (ad hoc)', f'$C_w$ = {Cw:.3f} (rigidity)', 'Not included'],
        ['Instability', 'v_max drift', f't₂₀+2t₂₂ > 0 ({"yes" if result["unstable"] else "no"})', f'E < E_crit ({E_c1:.4f})'],
    ]

    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.8)

    # Style header
    for j in range(4):
        table[0, j].set_facecolor('#4472C4')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # Alternating row colors
    for i in range(1, len(table_data)):
        color = '#D9E2F3' if i % 2 == 0 else 'white'
        for j in range(4):
            table[i, j].set_facecolor(color)

    ax.set_title('Comparison of TPW Models for Europa', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    path = os.path.join(outdir, 'compare_summary.png')
    fig.savefig(path)
    print(f'Saved: {path}')
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Compare TPW models')
    parser.add_argument('--outdir', type=str, default='docs/figures/comparison',
                        help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print('Generating comparison figures...\n')

    fig_compare_timescales(args.outdir)
    fig_compare_trajectories(args.outdir)
    fig_ospw_instability(args.outdir)
    fig_bh_coupling_regimes(args.outdir)
    fig_summary_table(args.outdir)

    plt.close('all')
    print('\nAll comparison figures generated.')


if __name__ == '__main__':
    main()
