#!/usr/bin/env python3
"""Publication-quality ice shell thickness plots.

Generates 2D Mollweide projections, 3D globe views, O&S vs PTM3D comparisons,
and latitude profiles for the Ojakangas & Stevenson (1989) ice shell model.

Usage:
    python scripts/plot_ice_shell.py [--save-dir plots] [--dpi 300]
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'python'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm, colors, ticker, patheffects
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D

from scipy.special import lpmv, gammaln

from ojakangas_stevenson import constants as C
from ojakangas_stevenson.surface_temperature import OSSurfaceTemperature
from ojakangas_stevenson.ptm3d_coupling import PTM3DSurfaceTemperature
from ojakangas_stevenson.rheology import MaxwellRheology, FlowLawRheology
from ojakangas_stevenson.shell_thickness import IceShellModel
from ojakangas_stevenson.tidal_strains import (
    time_averaged_strain_squared, time_averaged_strain_power)

# ---------------------------------------------------------------------------
# Global style
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
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.minor.width': 0.4,
    'ytick.minor.width': 0.4,
})

# Consistent colormaps
THICKNESS_CMAP = 'gist_earth'
DIFF_CMAP = 'RdBu_r'
TEMP_CMAP = 'inferno'

# Panel labels
PANEL_LABELS = 'abcdefghijkl'


def _panel_label(ax, label, x=-0.02, y=1.04, fontsize=12, **kwargs):
    """Add (a), (b), etc. label to an axes."""
    if hasattr(ax, 'get_zlim'):
        # 3D axes: use fig.text instead
        pos = ax.get_position()
        ax.figure.text(pos.x0 + x * pos.width, pos.y1 + (y - 1) * pos.height,
                       f'({label})', fontsize=fontsize, fontweight='bold',
                       va='bottom', ha='right', **kwargs)
    else:
        ax.text(x, y, f'({label})', transform=ax.transAxes,
                fontsize=fontsize, fontweight='bold', va='bottom', ha='right',
                **kwargs)


# ---------------------------------------------------------------------------
# Data computation helpers
# ---------------------------------------------------------------------------

def compute_grids(surface_temp, n_theta=361, n_lam=721):
    """Compute thickness grids for all 4 rheology cases.

    Returns dict of {case_name: (theta, lam, b_km, avg_km, moments)}.
    """
    theta = np.linspace(0.001, np.pi - 0.001, n_theta)
    lam = np.linspace(-np.pi, np.pi, n_lam)
    THETA, LAM = np.meshgrid(theta, lam, indexing='ij')

    g2, g3 = C.g2(), C.g3()
    a1, T_M = C.A1_CONDUCTIVITY, C.T_MELT

    # Surface temperature field
    if hasattr(surface_temp, 'temperature_2d'):
        T_s = surface_temp.temperature_2d(theta, lam)
    else:
        T_s = surface_temp.temperature(THETA)

    # Dissipation integrals
    eij2 = time_averaged_strain_squared(THETA, LAM, g2, g3)
    maxwell = MaxwellRheology()
    flow3 = FlowLawRheology(n_G=C.GLEN_EXPONENT_1, n_A=C.ARRHENIUS_EXPONENT,
                             A0_cgs=C.A0_FLOW_1_CGS)
    flow52 = FlowLawRheology(n_G=C.GLEN_EXPONENT_2, n_A=C.ARRHENIUS_EXPONENT,
                              A0_cgs=C.A0_FLOW_2_CGS)

    strain_rate_sq = C.MEAN_MOTION**2 * eij2
    D_maxwell = maxwell.dissipation_integral(strain_rate_sq)

    exp3 = flow3.strain_exponent
    exp52 = flow52.strain_exponent
    sp3 = time_averaged_strain_power(THETA, LAM, exp3, g2, g3, n_samples=360)
    sp52 = time_averaged_strain_power(THETA, LAM, exp52, g2, g3, n_samples=360)
    D_flow3 = flow3.dissipation_integral(C.MEAN_MOTION**(2*exp3) * sp3)
    D_flow52 = flow52.dissipation_integral(C.MEAN_MOTION**(2*exp52) * sp52)

    def _solve_b(T_s, D, H):
        ln_r = np.log(T_M / T_s)
        A = D / ln_r
        disc = H**2 + 4*A*a1*ln_r
        return np.clip((-H + np.sqrt(disc)) / (2*A), 0, 200e3)

    def _avg(b):
        dt, dl = theta[1]-theta[0], lam[1]-lam[0]
        return np.sum(b * np.sin(theta)[:, None]) * dt * dl / (4*np.pi)

    def _moments(b):
        dt, dl = theta[1]-theta[0], lam[1]-lam[0]
        ct, st = np.cos(theta), np.sin(theta)
        P20 = 0.5*(3*ct**2 - 1)
        P22 = 3*st**2
        c2l = np.cos(2*lam)
        i20 = b * P20[:, None] * st[:, None]
        t20 = 5/(4*np.pi) * np.sum(i20) * dt * dl
        i22 = b * P22[:, None] * c2l[None, :] * st[:, None]
        t22 = 5/(48*np.pi) * np.sum(i22) * dt * dl
        return {'t20': t20, 't22': t22,
                '4t22_km': 4*t22/1000, 't20_2t22_km': (t20+2*t22)/1000}

    cases_def = [
        ('Maxwell, $H$=0',      'maxwell_h0',    D_maxwell, 0.0),
        ('Maxwell, $H$=10',     'maxwell_h10',   D_maxwell, C.H_FLUX_10),
        ('Flow law $n_G$=3',    'flow_ng3',      D_flow3,   0.0),
        ('Flow law $n_G$=5.2',  'flow_ng52',     D_flow52,  0.0),
    ]

    results = {}
    for label, key, D, H in cases_def:
        b = _solve_b(T_s, D, H)
        b_km = b / 1000
        avg_km = _avg(b) / 1000
        mom = _moments(b)
        results[key] = {
            'label': label, 'theta': theta, 'lam': lam, 'b_km': b_km,
            'avg_km': avg_km, 'moments': mom, 'T_s': T_s,
        }
    return results


def sh_smooth_temperature(T_2d, theta, lam, l_max):
    """Smooth a 2D temperature field by truncating its SH expansion at l_max.

    Parameters
    ----------
    T_2d : (n_theta, n_lam) array
        Temperature on a regular (theta, lam) grid.
    theta : (n_theta,) array
        Colatitude values (0 to pi).
    lam : (n_lam,) array
        Longitude values (-pi to pi), with lam[-1] = lam[0] + 2*pi
        (periodic endpoint included).
    l_max : int
        Maximum spherical harmonic degree to retain.

    Returns
    -------
    T_smooth : (n_theta, n_lam) array, same shape as T_2d
    """
    dt = theta[1] - theta[0]
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # Remove repeated periodic endpoint for FFT
    T_periodic = T_2d[:, :-1]
    N = T_periodic.shape[1]

    # FFT along longitude axis
    T_fft = np.fft.rfft(T_periodic, axis=1)
    f_c = np.real(T_fft) * 2.0 / N   # cosine coefficients
    f_s = -np.imag(T_fft) * 2.0 / N  # sine coefficients
    f_c[:, 0] /= 2.0                  # m=0 has no factor-of-2

    # Phase correction: lambda grid starts at -pi, so FFT sees
    # cos(m * index) = cos(m(lam + pi)) = (-1)^m cos(m*lam)
    for m in range(f_c.shape[1]):
        phase = (-1)**m
        f_c[:, m] *= phase
        f_s[:, m] *= phase

    # Legendre projection and reconstruction
    coeffs = {}
    for m in range(min(l_max + 1, f_c.shape[1])):
        for l in range(m, l_max + 1):
            Plm = lpmv(m, l, cos_t)
            log_norm = (np.log(2*l + 1) - np.log(2)
                        + gammaln(l - m + 1) - gammaln(l + m + 1))
            N_lm = np.exp(log_norm)
            a_lm = N_lm * np.sum(f_c[:, m] * Plm * sin_t) * dt
            b_lm = (N_lm * np.sum(f_s[:, m] * Plm * sin_t) * dt
                     if m > 0 else 0.0)
            coeffs[(l, m)] = (a_lm, b_lm)

    T_smooth = np.zeros_like(T_2d)
    for (l, m), (a_lm, b_lm) in coeffs.items():
        Plm = lpmv(m, l, cos_t)
        if m == 0:
            T_smooth += a_lm * Plm[:, None]
        else:
            cos_ml = np.cos(m * lam)
            sin_ml = np.sin(m * lam)
            T_smooth += ((a_lm * cos_ml[None, :] + b_lm * sin_ml[None, :])
                         * Plm[:, None])
    return T_smooth


def compute_grids_smoothed(surface_temp, l_max_values, n_theta=361, n_lam=721):
    """Compute thickness grids for multiple l_max values.

    Returns dict of {l_max: results_dict}, where each results_dict has the same
    structure as compute_grids() output.
    """
    theta = np.linspace(0.001, np.pi - 0.001, n_theta)
    lam = np.linspace(-np.pi, np.pi, n_lam)
    THETA, LAM = np.meshgrid(theta, lam, indexing='ij')

    g2, g3 = C.g2(), C.g3()
    a1, T_M = C.A1_CONDUCTIVITY, C.T_MELT

    # Full temperature field
    if hasattr(surface_temp, 'temperature_2d'):
        T_s_full = surface_temp.temperature_2d(theta, lam)
    else:
        T_s_full = surface_temp.temperature(THETA)

    # Dissipation integrals (independent of temperature)
    eij2 = time_averaged_strain_squared(THETA, LAM, g2, g3)
    maxwell = MaxwellRheology()
    flow3 = FlowLawRheology(n_G=C.GLEN_EXPONENT_1, n_A=C.ARRHENIUS_EXPONENT,
                             A0_cgs=C.A0_FLOW_1_CGS)
    flow52 = FlowLawRheology(n_G=C.GLEN_EXPONENT_2, n_A=C.ARRHENIUS_EXPONENT,
                              A0_cgs=C.A0_FLOW_2_CGS)

    strain_rate_sq = C.MEAN_MOTION**2 * eij2
    D_maxwell = maxwell.dissipation_integral(strain_rate_sq)

    exp3 = flow3.strain_exponent
    exp52 = flow52.strain_exponent
    sp3 = time_averaged_strain_power(THETA, LAM, exp3, g2, g3, n_samples=360)
    sp52 = time_averaged_strain_power(THETA, LAM, exp52, g2, g3, n_samples=360)
    D_flow3 = flow3.dissipation_integral(C.MEAN_MOTION**(2*exp3) * sp3)
    D_flow52 = flow52.dissipation_integral(C.MEAN_MOTION**(2*exp52) * sp52)

    def _solve_b(T_s, D, H):
        ln_r = np.log(T_M / T_s)
        A = D / ln_r
        disc = H**2 + 4*A*a1*ln_r
        return np.clip((-H + np.sqrt(disc)) / (2*A), 0, 200e3)

    def _avg(b):
        dt, dl = theta[1]-theta[0], lam[1]-lam[0]
        return np.sum(b * np.sin(theta)[:, None]) * dt * dl / (4*np.pi)

    def _moments(b):
        dt, dl = theta[1]-theta[0], lam[1]-lam[0]
        ct, st = np.cos(theta), np.sin(theta)
        P20 = 0.5*(3*ct**2 - 1)
        P22 = 3*st**2
        c2l = np.cos(2*lam)
        i20 = b * P20[:, None] * st[:, None]
        t20 = 5/(4*np.pi) * np.sum(i20) * dt * dl
        i22 = b * P22[:, None] * c2l[None, :] * st[:, None]
        t22 = 5/(48*np.pi) * np.sum(i22) * dt * dl
        return {'t20': t20, 't22': t22,
                '4t22_km': 4*t22/1000, 't20_2t22_km': (t20+2*t22)/1000}

    def _sh_decomp(b):
        """Full SH decomposition for b11c, b11s, b20, b22c, b22s."""
        dt, dl = theta[1]-theta[0], lam[1]-lam[0]
        ct, st = np.cos(theta), np.sin(theta)
        # b11c, b11s: l=1, m=1
        P11 = st
        c1l = np.cos(lam)
        s1l = np.sin(lam)
        i11c = b * P11[:, None] * c1l[None, :] * st[:, None]
        i11s = b * P11[:, None] * s1l[None, :] * st[:, None]
        b11c = 3/(4*np.pi) * np.sum(i11c) * dt * dl / 1000
        b11s = 3/(4*np.pi) * np.sum(i11s) * dt * dl / 1000
        # b20: l=2, m=0
        P20 = 0.5*(3*ct**2 - 1)
        i20 = b * P20[:, None] * st[:, None]
        b20 = 5/(4*np.pi) * np.sum(i20) * dt * dl / 1000
        # b22c, b22s: l=2, m=2
        P22 = 3*st**2
        c2l = np.cos(2*lam)
        s2l = np.sin(2*lam)
        i22c = b * P22[:, None] * c2l[None, :] * st[:, None]
        i22s = b * P22[:, None] * s2l[None, :] * st[:, None]
        b22c = 5/(48*np.pi) * np.sum(i22c) * dt * dl / 1000
        b22s = 5/(48*np.pi) * np.sum(i22s) * dt * dl / 1000
        return {'b11c': b11c, 'b11s': b11s, 'b20': b20,
                'b22c': b22c, 'b22s': b22s}

    cases_def = [
        ('Maxwell, $H$=0',      'maxwell_h0',    D_maxwell, 0.0),
        ('Maxwell, $H$=10',     'maxwell_h10',   D_maxwell, C.H_FLUX_10),
        ('Flow law $n_G$=3',    'flow_ng3',      D_flow3,   0.0),
        ('Flow law $n_G$=5.2',  'flow_ng52',     D_flow52,  0.0),
    ]

    all_results = {}
    for l_max in l_max_values:
        print(f'  l_max = {l_max}...')
        if l_max is None:
            T_s = T_s_full
        else:
            T_s = sh_smooth_temperature(T_s_full, theta, lam, l_max)

        results = {}
        for label, key, D, H in cases_def:
            b = _solve_b(T_s, D, H)
            b_km = b / 1000
            avg_km = _avg(b) / 1000
            mom = _moments(b)
            sh = _sh_decomp(b)
            results[key] = {
                'label': label, 'theta': theta, 'lam': lam, 'b_km': b_km,
                'avg_km': avg_km, 'moments': mom, 'sh': sh, 'T_s': T_s,
            }
        all_results[l_max] = results
    return all_results


# ---------------------------------------------------------------------------
# Figure 7: SH convergence plot
# ---------------------------------------------------------------------------

def fig7_sh_convergence(all_results, save_path=None):
    """Convergence of geophysical quantities vs l_max."""
    l_max_values = sorted(k for k in all_results if k is not None)
    # Add "full" as a large l_max for plotting
    full_results = all_results.get(None, {})

    fig, axes = plt.subplots(2, 4, figsize=(14, 6.5), sharex=True)
    fig.subplots_adjust(hspace=0.32, wspace=0.38)

    quantities = [
        ('avg_km', r'$\langle b \rangle$ [km]', lambda r: r['avg_km']),
        ('b11c', r'$b_{11}^c$ [km]', lambda r: r['sh']['b11c']),
        ('b11s', r'$b_{11}^s$ [km]', lambda r: r['sh']['b11s']),
        ('b20', r'$b_{20}$ [km]', lambda r: r['sh']['b20']),
        ('b22c', r'$b_{22}^c$ [km]', lambda r: r['sh']['b22c']),
        ('4t22', r'$4t_{22}$ [km]', lambda r: r['moments']['4t22_km']),
        ('t20_2t22', r'$t_{20}\!+\!2t_{22}$ [km]',
         lambda r: r['moments']['t20_2t22_km']),
        ('dBA_C', r'$\Delta(B\!-\!A)/C$',
         lambda r: r['moments']['4t22_km'] * 1e-7),
    ]

    case_styles = {
        'maxwell_h0': ('C0', 'o', 'Maxwell $H$=0'),
        'maxwell_h10': ('C1', 's', 'Maxwell $H$=10'),
        'flow_ng3': ('C2', '^', 'Flow $n_G$=3'),
        'flow_ng52': ('C3', 'D', 'Flow $n_G$=5.2'),
    }

    for ax_idx, (qkey, ylabel, getter) in enumerate(quantities):
        ax = axes.flat[ax_idx]
        for case_key, (color, marker, clabel) in case_styles.items():
            vals = [getter(all_results[lm][case_key]) for lm in l_max_values]
            ax.plot(l_max_values, vals, color=color, marker=marker,
                    markersize=4, linewidth=1.2, label=clabel)
            # Plot full-resolution as dashed line
            if full_results:
                full_val = getter(full_results[case_key])
                ax.axhline(full_val, color=color, linestyle='--',
                           linewidth=0.7, alpha=0.5)

        ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(labelsize=8)
        if ax_idx >= 4:
            ax.set_xlabel(r'$\ell_{\max}$')
        _panel_label(ax, PANEL_LABELS[ax_idx], x=-0.02, y=1.06, fontsize=10)

    # Single legend at bottom
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle('Ice shell geophysical quantities vs SH truncation degree',
                 fontsize=13, fontweight='bold', y=1.01)

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        print(f'  Saved {save_path}')
    return fig


# ---------------------------------------------------------------------------
# Figure 8: Smoothed temperature + thickness Mollweide (single l_max)
# ---------------------------------------------------------------------------

def fig8_smoothed_mollweide(all_results, l_max_show, results_os,
                            save_path=None):
    """Thickness Mollweide: O&S vs SH-smoothed albedo map, plus difference."""
    smoothed = all_results.get(l_max_show, {})
    if not smoothed or not results_os:
        print(f'  Skipping fig8: l_max={l_max_show} or O&S not available')
        return None

    case_key = 'maxwell_h0'
    r_os = results_os[case_key]
    r_smooth = smoothed[case_key]

    theta = r_os['theta']
    lam = r_os['lam']
    lat = np.pi/2 - theta
    LON, LAT = np.meshgrid(lam, lat, indexing='ij')

    fig = plt.figure(figsize=(7.2, 8.8))
    gs = GridSpec(3, 1, figure=fig, hspace=0.18, height_ratios=[1, 1, 1])

    b_os = r_os['b_km']
    b_smooth = r_smooth['b_km']
    diff = b_smooth - b_os

    # Shared thickness range
    all_b = np.concatenate([b_os.ravel(), b_smooth.ravel()])
    vmin_b = max(0, np.nanpercentile(all_b[np.isfinite(all_b)], 0.5))
    vmax_b = np.nanpercentile(all_b[np.isfinite(all_b)], 99.5)
    norm_b = colors.Normalize(vmin=vmin_b, vmax=vmax_b)

    # Diff range
    norm_d = colors.TwoSlopeNorm(vmin=-2.5, vcenter=0, vmax=2.5)

    panels = [
        ('O&S analytical $T_s$', b_os, THICKNESS_CMAP, norm_b, 'km'),
        (rf'Spatially varying albedo, $\ell_{{\max}}$ = {l_max_show}',
         b_smooth, THICKNESS_CMAP, norm_b, 'km'),
        (rf'Spatially varying albedo ($\ell_{{\max}}$={l_max_show}) $-$ O&S',
         diff, DIFF_CMAP, norm_d, '$\\Delta b$ [km]'),
    ]

    for idx, (title, data, cmap, norm, clabel) in enumerate(panels):
        ax = fig.add_subplot(gs[idx], projection='mollweide')
        cs = ax.pcolormesh(LON.T, LAT.T, data, cmap=cmap, norm=norm,
                           shading='auto', rasterized=True)
        ax.grid(True, linewidth=0.3, alpha=0.4,
                color='w' if idx < 2 else 'gray')
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        # Inset colorbar
        bbox = ax.get_position()
        cb_ax = fig.add_axes([bbox.x1 + 0.01, bbox.y0 + 0.02,
                              0.015, bbox.height - 0.04])
        cbar = fig.colorbar(cs, cax=cb_ax)
        cbar.set_label(clabel, fontsize=9)
        cbar.ax.tick_params(labelsize=8)

        # Stats in title
        if idx < 2:
            r = r_os if idx == 0 else r_smooth
            mom = r['moments']
            title += (f"\n"
                      rf"$\langle b \rangle$ = {r['avg_km']:.1f} km, "
                      rf"$4t_{{22}}$ = {mom['4t22_km']:.2f} km")
        ax.set_title(title, fontsize=10, pad=4)
        _panel_label(ax, PANEL_LABELS[idx], x=-0.02, y=1.05, fontsize=12)

    fig.suptitle(f'Ice shell thickness: {r_os["label"]}',
                 fontsize=13, fontweight='bold', y=0.99)

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        print(f'  Saved {save_path}')
    return fig


# ---------------------------------------------------------------------------
# Figure 1: 2D Mollweide projections (all 4 cases)
# ---------------------------------------------------------------------------

def fig1_mollweide_panels(results, save_path=None):
    """Four-panel Mollweide projection of shell thickness."""
    fig = plt.figure(figsize=(7.2, 7.5))
    gs = GridSpec(2, 2, figure=fig, hspace=0.05, wspace=0.02)

    # Common colorbar range
    all_b = np.concatenate([r['b_km'].ravel() for r in results.values()])
    vmin = max(0, np.nanpercentile(all_b[np.isfinite(all_b)], 0.5))
    vmax = np.nanpercentile(all_b[np.isfinite(all_b)], 99.5)

    norm = colors.Normalize(vmin=vmin, vmax=vmax)

    axes = []
    for idx, key in enumerate(['maxwell_h0', 'maxwell_h10', 'flow_ng3', 'flow_ng52']):
        r = results[key]
        ax = fig.add_subplot(gs[idx // 2, idx % 2], projection='mollweide')
        axes.append(ax)

        theta, lam, b_km = r['theta'], r['lam'], r['b_km']
        # Convert colatitude to latitude for Mollweide
        lat = np.pi/2 - theta  # rad, [-pi/2, pi/2]
        LON, LAT = np.meshgrid(lam, lat, indexing='ij')

        # Mollweide expects (lon, lat) with lon in [-pi, pi]
        cs = ax.pcolormesh(LON.T, LAT.T, b_km, cmap=THICKNESS_CMAP,
                           norm=norm, shading='auto', rasterized=True)

        ax.grid(True, linewidth=0.3, alpha=0.4, color='w')
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        avg = r['avg_km']
        mom = r['moments']
        title = (f"{r['label']}\n"
                 rf"$\langle b \rangle$ = {avg:.1f} km, "
                 rf"$4t_{{22}}$ = {mom['4t22_km']:.1f} km")
        ax.set_title(title, fontsize=10, pad=4)
        _panel_label(ax, PANEL_LABELS[idx], x=0.02, y=0.96, fontsize=11)

    # Shared colorbar
    cbar_ax = fig.add_axes([0.15, 0.04, 0.7, 0.018])
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=THICKNESS_CMAP),
                        cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Ice shell thickness [km]')

    fig.suptitle('Equilibrium ice shell thickness (PTM3D surface temperatures)',
                 fontsize=13, fontweight='bold', y=0.98)

    if save_path:
        fig.savefig(save_path)
        print(f'  Saved {save_path}')
    return fig


# ---------------------------------------------------------------------------
# Figure 2: 3D globe views
# ---------------------------------------------------------------------------

def _make_globe_data(theta, lam, b_km, n_lat=180, n_lon=361):
    """Prepare 3D globe data from thickness grid with longitude wrap."""
    step_t = max(1, len(theta) // n_lat)
    step_l = max(1, len(lam) // n_lon)
    th_c = theta[::step_t]
    lm_c = lam[::step_l]
    b_c = b_km[::step_t, ::step_l]

    # Wrap longitude to close the seam
    lm_c = np.append(lm_c, lm_c[0] + 2*np.pi)
    b_c = np.column_stack([b_c, b_c[:, 0]])

    lat_c = np.pi/2 - th_c
    LON, LAT = np.meshgrid(lm_c, lat_c)

    # Exaggerate radial variation for visibility
    R0 = 1.0
    b_mean = np.mean(b_c)
    b_range = b_c.max() - b_c.min()
    scale = 0.12 / b_range if b_range > 0 else 0
    R = R0 + (b_c - b_mean) * scale

    X = R * np.cos(LAT) * np.cos(LON)
    Y = R * np.cos(LAT) * np.sin(LON)
    Z = R * np.sin(LAT)
    return X, Y, Z, b_c


def fig2_globe_views(results, case_key='flow_ng3', save_path=None):
    """3D globe views: sub-Jovian, anti-Jovian, north pole, south pole."""
    r = results[case_key]
    theta, lam, b_km = r['theta'], r['lam'], r['b_km']

    X, Y, Z, b_c = _make_globe_data(theta, lam, b_km)

    vmin = max(0, np.nanpercentile(b_c, 1))
    vmax = np.nanpercentile(b_c, 99)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    facecolors = plt.get_cmap(THICKNESS_CMAP)(norm(b_c))

    views = [
        ('Sub-Jovian',    0,   0),
        ('Anti-Jovian',   0, 180),
        ('North pole',   90,   0),
        ('South pole',  -90,   0),
    ]

    fig = plt.figure(figsize=(7.2, 7.2))
    gs = GridSpec(2, 2, figure=fig, hspace=0.02, wspace=0.02)

    for idx, (view_name, elev, azim) in enumerate(views):
        ax = fig.add_subplot(gs[idx // 2, idx % 2], projection='3d',
                             computed_zorder=False)
        ax.plot_surface(X, Y, Z, facecolors=facecolors,
                        rstride=1, cstride=1, shade=False,
                        antialiased=False, rasterized=True)
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_zlim(-1.1, 1.1)
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        ax.set_box_aspect([1, 1, 1])
        ax.set_title(view_name, fontsize=10, pad=-8)
        _panel_label(ax, PANEL_LABELS[idx], x=0.08, y=0.92)

    # Shared colorbar
    sm = cm.ScalarMappable(norm=norm, cmap=THICKNESS_CMAP)
    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.018])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Ice shell thickness [km]')

    fig.suptitle(f"Ice shell thickness: {r['label']}, "
                 rf"$\langle b \rangle$ = {r['avg_km']:.1f} km",
                 fontsize=13, fontweight='bold', y=0.97)

    if save_path:
        fig.savefig(save_path)
        print(f'  Saved {save_path}')
    return fig


# ---------------------------------------------------------------------------
# Figure 3: O&S vs PTM3D comparison
# ---------------------------------------------------------------------------

def fig3_os_vs_ptm3d(results_os, results_ptm, case_key='flow_ng3',
                     save_path=None):
    """Three-panel Mollweide: O&S, PTM3D, difference."""
    r_os = results_os[case_key]
    r_ptm = results_ptm[case_key]

    theta = r_os['theta']
    lam = r_os['lam']
    lat = np.pi/2 - theta
    LON, LAT = np.meshgrid(lam, lat, indexing='ij')

    b_os = r_os['b_km']
    b_ptm = r_ptm['b_km']
    diff = b_ptm - b_os

    # Shared thickness range
    all_b = np.concatenate([b_os.ravel(), b_ptm.ravel()])
    vmin = max(0, np.nanpercentile(all_b[np.isfinite(all_b)], 0.5))
    vmax = np.nanpercentile(all_b[np.isfinite(all_b)], 99.5)
    norm_b = colors.Normalize(vmin=vmin, vmax=vmax)

    # Diff range: fixed ±1 km
    norm_d = colors.TwoSlopeNorm(vmin=-1.0, vcenter=0, vmax=1.0)

    fig = plt.figure(figsize=(7.2, 8.8))
    gs = GridSpec(3, 1, figure=fig, hspace=0.18,
                  height_ratios=[1, 1, 1])

    panels = [
        ('O&S analytical $T_s$', b_os, THICKNESS_CMAP, norm_b, 'km'),
        ('PTM3D $T_s$ (2D)', b_ptm, THICKNESS_CMAP, norm_b, 'km'),
        ('PTM3D $-$ O&S', diff, DIFF_CMAP, norm_d, '$\\Delta b$ [km]'),
    ]

    for idx, (title, data, cmap, norm, clabel) in enumerate(panels):
        ax = fig.add_subplot(gs[idx], projection='mollweide')
        cs = ax.pcolormesh(LON.T, LAT.T, data, cmap=cmap, norm=norm,
                           shading='auto', rasterized=True)
        ax.grid(True, linewidth=0.3, alpha=0.4, color='w' if idx < 2 else 'gray')
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        # Inset colorbar
        bbox = ax.get_position()
        cb_ax = fig.add_axes([bbox.x1 + 0.01, bbox.y0 + 0.02,
                              0.015, bbox.height - 0.04])
        cbar = fig.colorbar(cs, cax=cb_ax)
        cbar.set_label(clabel, fontsize=9)
        cbar.ax.tick_params(labelsize=8)

        ax.set_title(title, fontsize=11, pad=4)
        _panel_label(ax, PANEL_LABELS[idx], x=-0.02, y=1.05, fontsize=12)

    avg_os = r_os['avg_km']
    avg_ptm = r_ptm['avg_km']
    fig.suptitle(
        f"{r_os['label']}: O&S "
        rf"$\langle b \rangle$={avg_os:.1f} km vs PTM3D "
        rf"$\langle b \rangle$={avg_ptm:.1f} km",
        fontsize=12, fontweight='bold', y=0.99)

    if save_path:
        fig.savefig(save_path)
        print(f'  Saved {save_path}')
    return fig


# ---------------------------------------------------------------------------
# Figure 4: Thickness profiles vs latitude
# ---------------------------------------------------------------------------

def fig4_latitude_profiles(results_os, results_ptm, save_path=None):
    """Thickness vs latitude at key longitudes, all 4 cases."""
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0), sharex=True)
    axes = axes.flatten()

    lon_vals = {
        r'$\lambda$=0 (sub-Jovian)': 0,
        r'$\lambda$=90$\degree$': np.pi/2,
        r'$\lambda$=180$\degree$ (anti-Jovian)': np.pi,
        r'$\lambda$=270$\degree$': 3*np.pi/2 - 2*np.pi,  # = -pi/2
    }

    lon_colors = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a']

    for idx, key in enumerate(['maxwell_h0', 'maxwell_h10', 'flow_ng3', 'flow_ng52']):
        ax = axes[idx]
        r_os = results_os[key]
        r_ptm = results_ptm[key]

        theta_os = r_os['theta']
        lat_deg_os = 90 - np.degrees(theta_os)

        # Zonally averaged profiles
        b_zonal_os = np.mean(r_os['b_km'], axis=1)
        b_zonal_ptm = np.mean(r_ptm['b_km'], axis=1)

        ax.plot(lat_deg_os, b_zonal_os, 'k-', linewidth=1.8,
                label='O&S (zonal avg)', zorder=5)
        ax.plot(lat_deg_os, b_zonal_ptm, 'k--', linewidth=1.8,
                label='PTM3D (zonal avg)', zorder=5)

        # Longitude slices for PTM3D
        lam_arr = r_ptm['lam']
        for (lon_label, lon_val), col in zip(lon_vals.items(), lon_colors):
            li = np.argmin(np.abs(lam_arr - lon_val))
            ax.plot(lat_deg_os, r_ptm['b_km'][:, li], '-', color=col,
                    linewidth=0.9, alpha=0.7, label=lon_label)

        ax.set_ylabel('Thickness [km]')
        ax.set_title(r_os['label'], fontsize=10)
        ax.grid(True, alpha=0.2)
        ax.set_xlim(-90, 90)
        _panel_label(ax, PANEL_LABELS[idx])

        if idx == 0:
            ax.legend(fontsize=7.5, loc='upper center', ncol=2,
                      framealpha=0.85, handlelength=1.5)

    axes[2].set_xlabel('Latitude [deg]')
    axes[3].set_xlabel('Latitude [deg]')

    fig.suptitle('Ice shell thickness profiles', fontsize=13,
                 fontweight='bold', y=1.01)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
        print(f'  Saved {save_path}')
    return fig


# ---------------------------------------------------------------------------
# Figure 5: Surface temperature comparison (O&S vs PTM3D, Mollweide)
# ---------------------------------------------------------------------------

def fig5_surface_temperature(temp_os, temp_ptm, save_path=None):
    """Two-panel Mollweide of surface temperature: O&S vs PTM3D."""
    n_theta, n_lam = 361, 721
    theta = np.linspace(0.001, np.pi - 0.001, n_theta)
    lam = np.linspace(-np.pi, np.pi, n_lam)
    THETA, LAM = np.meshgrid(theta, lam, indexing='ij')
    lat = np.pi/2 - theta
    LON, LAT = np.meshgrid(lam, lat, indexing='ij')

    T_os = temp_os.temperature(THETA)
    T_ptm = temp_ptm.temperature_2d(theta, lam)

    vmin = min(np.nanmin(T_os), np.nanmin(T_ptm))
    vmax = max(np.nanmax(T_os), np.nanmax(T_ptm))
    norm = colors.Normalize(vmin=vmin, vmax=vmax)

    # Diff
    diff = T_ptm - T_os
    d_abs = np.nanmax(np.abs(diff[np.isfinite(diff)]))
    norm_d = colors.TwoSlopeNorm(vmin=-d_abs, vcenter=0, vmax=d_abs)

    fig = plt.figure(figsize=(7.2, 8.8))
    gs = GridSpec(3, 1, figure=fig, hspace=0.18, height_ratios=[1, 1, 1])

    panels_data = [
        ('O&S analytical', T_os, TEMP_CMAP, norm, '$T_s$ [K]'),
        ('PTM3D', T_ptm, TEMP_CMAP, norm, '$T_s$ [K]'),
        ('PTM3D $-$ O&S', diff, DIFF_CMAP, norm_d, '$\\Delta T$ [K]'),
    ]

    for idx, (title, data, cmap, nm, clabel) in enumerate(panels_data):
        ax = fig.add_subplot(gs[idx], projection='mollweide')
        cs = ax.pcolormesh(LON.T, LAT.T, data, cmap=cmap, norm=nm,
                           shading='auto', rasterized=True)
        ax.grid(True, linewidth=0.3, alpha=0.4, color='w' if idx < 2 else 'gray')
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        bbox = ax.get_position()
        cb_ax = fig.add_axes([bbox.x1 + 0.01, bbox.y0 + 0.02,
                              0.015, bbox.height - 0.04])
        cbar = fig.colorbar(cs, cax=cb_ax)
        cbar.set_label(clabel, fontsize=9)
        cbar.ax.tick_params(labelsize=8)

        ax.set_title(title, fontsize=11, pad=4)
        _panel_label(ax, PANEL_LABELS[idx], x=-0.02, y=1.05, fontsize=12)

    fig.suptitle('Annual-average surface temperature',
                 fontsize=13, fontweight='bold', y=0.99)

    if save_path:
        fig.savefig(save_path)
        print(f'  Saved {save_path}')
    return fig


# ---------------------------------------------------------------------------
# Figure 6: 3D globe comparison (O&S vs PTM3D side by side)
# ---------------------------------------------------------------------------

def fig6_globe_comparison(results_os, results_ptm, case_key='flow_ng3',
                          save_path=None):
    """Side-by-side 3D globes: O&S left, PTM3D right, two viewing angles."""
    datasets = [
        ('O&S', results_os[case_key]),
        ('PTM3D', results_ptm[case_key]),
    ]
    views = [
        ('Sub-Jovian', 10, 0),
        ('Trailing', 10, -90),
    ]

    # Common color range
    all_b = np.concatenate([d['b_km'].ravel() for _, d in datasets])
    vmin = max(0, np.nanpercentile(all_b[np.isfinite(all_b)], 0.5))
    vmax = np.nanpercentile(all_b[np.isfinite(all_b)], 99.5)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)

    fig = plt.figure(figsize=(7.2, 7.2))
    gs = GridSpec(2, 2, figure=fig, hspace=0.0, wspace=0.0)

    panel_idx = 0
    for row, (view_name, elev, azim) in enumerate(views):
        for col, (ds_name, r) in enumerate(datasets):
            theta, lam, b_km = r['theta'], r['lam'], r['b_km']
            X, Y, Z, b_c = _make_globe_data(theta, lam, b_km)
            facecolors = plt.get_cmap(THICKNESS_CMAP)(norm(b_c))

            ax = fig.add_subplot(gs[row, col], projection='3d',
                                 computed_zorder=False)
            ax.plot_surface(X, Y, Z, facecolors=facecolors,
                            rstride=1, cstride=1, shade=False,
                            antialiased=False, rasterized=True)
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-1.1, 1.1)
            ax.set_zlim(-1.1, 1.1)
            ax.view_init(elev=elev, azim=azim)
            ax.set_axis_off()
            ax.set_box_aspect([1, 1, 1])

            if row == 0:
                ax.set_title(f'{ds_name}\n'
                             rf"$\langle b \rangle$={r['avg_km']:.1f} km",
                             fontsize=10, pad=-5)

            _panel_label(ax, PANEL_LABELS[panel_idx], x=0.08, y=0.92)
            panel_idx += 1

    # Row labels on right
    fig.text(0.98, 0.72, 'Sub-Jovian view', fontsize=10, rotation=270,
             va='center', ha='center')
    fig.text(0.98, 0.32, 'Trailing view', fontsize=10, rotation=270,
             va='center', ha='center')

    # Shared colorbar
    sm = cm.ScalarMappable(norm=norm, cmap=THICKNESS_CMAP)
    cbar_ax = fig.add_axes([0.15, 0.04, 0.7, 0.018])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Ice shell thickness [km]')

    label = results_os[case_key]['label']
    fig.suptitle(f'Ice shell thickness: {label}',
                 fontsize=13, fontweight='bold', y=0.98)

    if save_path:
        fig.savefig(save_path)
        print(f'  Saved {save_path}')
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_all(save_dir='plots', dpi=300,
                 ptm3d_temp_dir='output/europa_global_ico7_no-albedo/temperature',
                 ptm3d_mesh='output/europa_global_ico7_no-albedo/mesh.ply',
                 fmt='pdf'):
    """Generate all figures. Returns (results_os, results_ptm)."""
    plt.rcParams['savefig.dpi'] = dpi
    os.makedirs(save_dir, exist_ok=True)

    print('Loading PTM3D data...')
    ptm = PTM3DSurfaceTemperature.from_binary_files(ptm3d_temp_dir, ptm3d_mesh)
    print('Symmetrizing N-S...')
    ptm.symmetrize_ns()
    os_temp = OSSurfaceTemperature()

    print('Computing O&S thickness grids...')
    results_os = compute_grids(os_temp)
    print('Computing PTM3D thickness grids...')
    results_ptm = compute_grids(ptm)

    print('Generating figures...')
    def sp(name):
        return os.path.join(save_dir, f'{name}.{fmt}')

    fig1_mollweide_panels(results_ptm, save_path=sp('fig1_thickness_mollweide'))
    fig2_globe_views(results_ptm, case_key='flow_ng3', save_path=sp('fig2_globe_flow_ng3'))
    fig2_globe_views(results_ptm, case_key='maxwell_h0', save_path=sp('fig2b_globe_maxwell_h0'))
    fig3_os_vs_ptm3d(results_os, results_ptm, case_key='flow_ng3', save_path=sp('fig3_os_vs_ptm3d_flow'))
    fig3_os_vs_ptm3d(results_os, results_ptm, case_key='maxwell_h0', save_path=sp('fig3b_os_vs_ptm3d_maxwell'))
    fig4_latitude_profiles(results_os, results_ptm, save_path=sp('fig4_profiles'))
    fig5_surface_temperature(os_temp, ptm, save_path=sp('fig5_surface_temperature'))
    fig6_globe_comparison(results_os, results_ptm, case_key='flow_ng3', save_path=sp('fig6_globe_comparison'))

    plt.close('all')
    print('Done.')
    return results_os, results_ptm


def generate_smoothed(save_dir='plots/smoothed', dpi=300,
                      ptm3d_temp_dir='output/europa_global_ico7/temperature',
                      ptm3d_mesh='output/europa_global_ico7/mesh.ply',
                      fmt='pdf',
                      l_max_values=None):
    """Generate SH-smoothed analysis figures and print results table."""
    if l_max_values is None:
        l_max_values = [2, 4, 8, 16, 32, None]

    plt.rcParams['savefig.dpi'] = dpi
    os.makedirs(save_dir, exist_ok=True)

    print('Loading PTM3D data...')
    ptm = PTM3DSurfaceTemperature.from_binary_files(ptm3d_temp_dir, ptm3d_mesh)
    print('Symmetrizing N-S...')
    ptm.symmetrize_ns()

    print('Computing O&S thickness grids...')
    os_temp = OSSurfaceTemperature()
    results_os = compute_grids(os_temp)

    print('Computing smoothed thickness grids...')
    all_results = compute_grids_smoothed(ptm, l_max_values)

    # Print results table
    print('\n=== SH-Smoothed Results ===\n')
    for case_key in ['maxwell_h0', 'maxwell_h10', 'flow_ng3', 'flow_ng52']:
        label = all_results[l_max_values[0]][case_key]['label']
        print(f'\n{label}:')
        hdr = (f'{"l_max":>8s} {"avg":>7s} {"b11c":>8s} {"b11s":>8s} '
               f'{"b20":>8s} {"b22c":>8s} {"4t22":>7s} {"t20+2t22":>9s} '
               f'{"D(B-A)/C":>12s} {"D(B-C)/A":>12s}')
        print(hdr)
        for lm in l_max_values:
            r = all_results[lm][case_key]
            lm_str = 'full' if lm is None else str(lm)
            sh = r['sh']
            mom = r['moments']
            dBA = mom['4t22_km'] * 1e-7
            dBC = mom['t20_2t22_km'] * 1e-7
            print(f'{lm_str:>8s} {r["avg_km"]:7.2f} {sh["b11c"]:+8.3f} '
                  f'{sh["b11s"]:+8.3f} {sh["b20"]:+8.3f} {sh["b22c"]:+8.3f} '
                  f'{mom["4t22_km"]:7.2f} {mom["t20_2t22_km"]:+9.2f} '
                  f'{dBA:12.2e} {dBC:12.2e}')

    print('\nGenerating figures...')
    def sp(name):
        return os.path.join(save_dir, f'{name}.{fmt}')

    fig7_sh_convergence(all_results, save_path=sp('fig7_sh_convergence'))
    # Use l_max=4 as representative (essentially converged)
    l_max_repr = 4
    if l_max_repr in all_results:
        fig8_smoothed_mollweide(all_results, l_max_repr, results_os,
                                save_path=sp('fig8_smoothed_mollweide'))

    plt.close('all')
    print('Done.')
    return all_results


def main():
    parser = argparse.ArgumentParser(description='Generate ice shell plots')
    parser.add_argument('--save-dir', default='plots',
                        help='Output directory')
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--format', default='pdf', choices=['pdf', 'png'],
                        help='Output format')
    parser.add_argument('--ptm3d-temp-dir',
                        default='output/europa_global_ico7_no-albedo/temperature')
    parser.add_argument('--ptm3d-mesh',
                        default='output/europa_global_ico7_no-albedo/mesh.ply')
    parser.add_argument('--smoothed', action='store_true',
                        help='Generate SH-smoothed analysis (albedo map)')
    args = parser.parse_args()

    if args.smoothed:
        generate_smoothed(save_dir=args.save_dir, dpi=args.dpi, fmt=args.format,
                          ptm3d_temp_dir=args.ptm3d_temp_dir,
                          ptm3d_mesh=args.ptm3d_mesh)
    else:
        generate_all(save_dir=args.save_dir, dpi=args.dpi, fmt=args.format,
                     ptm3d_temp_dir=args.ptm3d_temp_dir,
                     ptm3d_mesh=args.ptm3d_mesh)


if __name__ == '__main__':
    main()
