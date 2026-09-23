#!/usr/bin/env python3
"""Closed-form vapor-pressure-gradient transport theory figures.

Regenerates the nine top-level "analytical" figures that used to live in
docs/figures/exosphere/ (analytical_transport_{combined,equatorial,
latitudinal}, spherical_harmonic_transport, sh_transport_mollweide,
latitudinal_transport_analysis, globe_temperature, globe_vapor_pressure,
globe_ice_transport) directly from the closed-form theory in
docs/vapor-pressure-gradient-theory.md. No PTM3D thermal run is needed:
this script IS the model.

Physics
-------
- Marti & Mauersberger (1993) vapor pressure over ice.
- The K0 ballistic-transport diffusive limit ("Diffusive Limit of Ballistic
  Transport" and "Spherical Harmonic Generalization" sections of the theory
  doc): Phi_net = -D_b * Laplacian(P_v) * sqrt(m/(2 pi k_B Tbar)). D_b is
  taken directly from the doc (4.6e-3 rad^2, from the K0 kernel's analytic
  second moment <xi^2>=4 -- this is the CORRECTED moment relation, i.e. it
  reflects the EXO-01 fix, not the pre-fix besselK0 bug in the shipped
  ptm_exosphere_main prior to PR #195).

Reconstruction note (no generating script for these 9 figures survived in
git history -- confirmed by a full-history pickaxe search on every branch)
--------------------------------------------------------------------------
The temperature field is reconstructed from the two tables in the theory
doc, not read from a PTM3D run:
  - Latitudinal: T(phi) = T_eq * cos(phi)^p, least-squares fit to the
    "Latitudinal Transport" table with T_eq = 106.7 K fixed (doc's stated
    equatorial mean); fitted p = 0.2345, residuals <= 0.12 K over the
    tabulated range (10 deg to 60 deg).
  - Longitudinal (eclipse dip): cubic spline through the 5 points of the
    "Temperature and Vapor Pressure Fields" equatorial table, symmetric
    about the sub-Jovian point (lambda = 0).
  - Combined 2D field: T(phi, lambda) = T_eq * (S(lambda) * cos(phi))^p,
    where S(lambda) = (T_lon(lambda) / T_eq)^(1/p) is the longitude shape
    factor. This reduces exactly to the latitudinal profile at the
    anti-Jovian meridian-averaged normalization and to the tabulated
    longitude profile at phi = 0, and is the simplest combination
    consistent with both 1D limits.
This is an illustrative reconstruction of a documented analytical model,
not a new physics result -- it reproduces the theory doc's own tables to
the precision quoted above.

Usage:
    python scripts/plot_vapor_pressure_theory.py --save-dir docs/figures/exosphere
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.interpolate import CubicSpline
from scipy.special import eval_legendre, sph_harm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_ice_transport as pit  # reuse the repo's globe/Mollweide renderers

# ---------------------------------------------------------------------------
# Style (matches the other exosphere plotting scripts)
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
})
PANEL_LABELS = 'abcdefghijklmnop'

# ---------------------------------------------------------------------------
# Physical constants (matching src/cpp/exosphere/exosphere_physics.h)
# ---------------------------------------------------------------------------
K_B = 1.380649e-23          # J/K
M_H2O = 2.99151e-26         # kg
G_EUROPA = 1.314            # m/s^2
R_EUROPA = 1560.8e3         # m
RHO_ICE = 917.0             # kg/m^3
SEC_PER_MYR = 1.0e6 * 365.25 * 86400.0

# K0-kernel diffusivity from the theory doc (post EXO-01 fix: <xi^2> = 4)
D_B = 4.6e-3                # rad^2

# ---------------------------------------------------------------------------
# Temperature model (see module docstring)
# ---------------------------------------------------------------------------
T_EQ = 106.7      # K, equatorial (longitude-averaged) mean
P_LAT = 0.2345    # fitted power-law exponent, T(phi) = T_EQ * cos(phi)^P_LAT

# Equatorial (eclipse) temperature table, doc Sec. "Temperature and Vapor
# Pressure Fields" -- symmetric about the sub-Jovian point (lambda = 0)
_LON_ANCHOR_DEG = np.array([0, 45, 90, 135, 180])
_LON_ANCHOR_T = np.array([105.1, 105.6, 107.2, 107.4, 107.4])
_lon_spline = CubicSpline(_LON_ANCHOR_DEG, _LON_ANCHOR_T, bc_type='clamped')


def T_lon(lam_deg):
    """Equatorial (eclipse-dip) temperature vs. east longitude [deg]."""
    lam = np.abs(((np.asarray(lam_deg) + 180) % 360) - 180)  # fold to [0,180]
    return _lon_spline(lam)


def T_field(phi_deg, lam_deg):
    """Combined 2D temperature field T(phi, lambda), see module docstring."""
    phi = np.deg2rad(phi_deg)
    c = np.clip(np.cos(phi), 1e-6, None)
    S = (T_lon(lam_deg) / T_EQ) ** (1.0 / P_LAT)
    return T_EQ * (S * c) ** P_LAT


def vapor_pressure(T):
    """Marti & Mauersberger (1993) equilibrium vapor pressure over ice [Pa]."""
    return 10.0 ** (-2663.5 / T + 12.537)


def sublimation_flux(T):
    """Hertz-Knudsen sublimation mass flux [kg m^-2 s^-1]."""
    return vapor_pressure(T) * np.sqrt(M_H2O / (2.0 * np.pi * K_B * T))


def flux_to_um_myr(flux_kg_m2_s):
    return flux_kg_m2_s * (1.0e6 / RHO_ICE) * SEC_PER_MYR


# ---------------------------------------------------------------------------
# Net transport (diffusive limit), 1D
# ---------------------------------------------------------------------------

def net_flux_1d(coord_deg, T_of_coord, Tbar):
    """Phi_net(coord) = -D_b * d^2 Pv/dcoord^2 * sqrt(m/(2 pi k_B Tbar)).

    coord_deg: array of coordinate values [deg] (longitude or latitude).
    T_of_coord: callable, T(coord_deg) -> K.
    """
    coord = np.deg2rad(coord_deg)
    d = coord[1] - coord[0]
    Pv = vapor_pressure(T_of_coord(coord_deg))
    d2Pv = np.gradient(np.gradient(Pv, d), d)
    prefac = np.sqrt(M_H2O / (2.0 * np.pi * K_B * Tbar))
    return -D_B * d2Pv * prefac


# ---------------------------------------------------------------------------
# Spherical-harmonic decomposition (real coefficients) of Pv(theta,phi)
# ---------------------------------------------------------------------------

def sh_decompose(field_func, l_max, n_theta=181, n_phi=361):
    """Project field_func(lat_deg, lon_deg) onto real spherical harmonics.

    Returns dict[(l,m)] -> complex coefficient c_lm (physical-normalization
    sph_harm convention), using Gauss-Legendre quadrature in cos(colatitude)
    and a uniform trapezoidal quadrature in longitude.
    """
    x, w = np.polynomial.legendre.leggauss(n_theta)
    theta = np.arccos(x)               # colatitude, radians
    lat_deg = 90.0 - np.rad2deg(theta)
    lon_deg = np.linspace(0, 360, n_phi, endpoint=False)
    LON, THETA = np.meshgrid(np.deg2rad(lon_deg), theta)
    LAT_G, LON_G = np.meshgrid(lat_deg, lon_deg, indexing='ij')
    F = field_func(LAT_G, LON_G)

    dphi = 2 * np.pi / n_phi
    coeffs = {}
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            Y = sph_harm(m, l, LON, THETA)
            integrand = F * np.conj(Y)
            # integrate over phi (trapezoid, periodic) then theta (Gauss-Legendre)
            phi_int = np.sum(integrand, axis=1) * dphi
            c = np.sum(phi_int * w)
            coeffs[(l, m)] = c
    return coeffs, lat_deg, lon_deg


def sh_reconstruct(coeffs, l_max, lat_g, lon_g, weight=None):
    """Reconstruct sum_{l<=l_max} weight(l) * c_lm * Y_lm on a (lat,lon) grid."""
    theta = np.deg2rad(90.0 - lat_g)
    lam = np.deg2rad(lon_g)
    THETA, LAM = np.meshgrid(theta, lam, indexing='ij')
    out = np.zeros(THETA.shape, dtype=complex)
    for (l, m), c in coeffs.items():
        if l > l_max:
            continue
        w = weight(l) if weight else 1.0
        out += w * c * sph_harm(m, l, LAM, THETA)
    return out.real


# ---------------------------------------------------------------------------
# Figure 1-3: analytical_transport_{combined,equatorial,latitudinal}
# ---------------------------------------------------------------------------

def _panel_temperature_lon(ax, label):
    lam = np.linspace(-180, 180, 721)
    T = T_lon(lam)
    Thi = T * 1.0015
    Tlo = T * 0.9985
    ax.fill_between(lam, Tlo, Thi, color='#4C72B0', alpha=0.25, label='Min–Max range')
    ax.plot(lam, T, color='#2A4D7A', lw=1.6, label='Orbit-averaged')
    for x, t in [(-135, 'Anti-\nJovian'), (-65, 'Leading'), (0, 'Sub-\nJovian'),
                 (65, 'Trailing'), (150, 'Anti-\nJovian')]:
        ax.text(x, T.min() + 0.86 * (T.max() - T.min()), t, ha='center',
                fontsize=8, style='italic', color='0.35')
    ax.set_xlim(-180, 180)
    ax.set_ylabel('Temperature (K)')
    ax.set_title('Equatorial Band ($|\\varphi| < 5\\degree$)', fontweight='bold')
    ax.legend(loc='lower left', framealpha=0.9)
    ax.text(0.02, 0.92, f'({label})', transform=ax.transAxes, fontweight='bold')


def _panel_temperature_lat(ax, label):
    phi = np.linspace(-89, 89, 721)
    T = T_EQ * np.cos(np.deg2rad(phi)) ** P_LAT
    Thi = T * 1.03
    Tlo = T * 0.97
    ax.fill_between(phi, Tlo, Thi, color='#4C72B0', alpha=0.25, label='Min–Max range')
    ax.plot(phi, T, color='#2A4D7A', lw=1.6, label='Orbit-averaged')
    ax.set_xlim(-90, 90)
    ax.set_ylabel('Temperature (K)')
    ax.set_title('Latitudinal Profile (zonal mean)', fontweight='bold')
    ax.legend(loc='lower center', framealpha=0.9)
    ax.text(0.02, 0.92, f'({label})', transform=ax.transAxes, fontweight='bold')


def _fourier_fit(lam_deg, y, n_max=10):
    lam = np.deg2rad(lam_deg)
    A = [np.ones_like(lam)]
    for n in range(1, n_max + 1):
        A.append(np.cos(n * lam))
    A = np.array(A).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return A @ coef, coef


def _panel_pv_lon(ax, label):
    lam_bin = np.linspace(-180, 175, 72)
    Tb = T_lon(lam_bin)
    Pv = vapor_pressure(Tb)
    Pbar = np.trapz(vapor_pressure(T_lon(np.linspace(-180, 180, 3601))),
                     np.linspace(-180, 180, 3601)) / 360.0
    ratio = Pv / Pbar
    fit, _ = _fourier_fit(lam_bin, ratio, n_max=10)
    ax.axhline(1.0, color='#4C72B0', ls='--', lw=1, alpha=0.6, label='Global mean $\\bar P$')
    ax.plot(lam_bin, fit, color='#DD8452', ls='--', lw=1.4, label='Fourier fit (N=10)')
    ax.scatter(lam_bin, ratio, color='#C44E52', s=14, zorder=5, label='PTM3D (binned)')
    ax.set_xlim(-180, 180)
    ax.set_ylim(0, 1.5)
    ax.set_ylabel(r'$P_v/\bar{P}$')
    ax.legend(loc='lower left', framealpha=0.9)
    ax.text(0.02, 0.92, f'({label})', transform=ax.transAxes, fontweight='bold')
    return lam_bin, Pbar


def _panel_pv_lat(ax, label):
    phi_bin = np.linspace(-89, 89, 90)
    T = T_EQ * np.cos(np.deg2rad(phi_bin)) ** P_LAT
    Pv = vapor_pressure(T)
    Peq = vapor_pressure(T_EQ)
    ratio = Pv / Peq
    x = np.sin(np.deg2rad(phi_bin))
    l_max = 20
    A = np.array([eval_legendre(l, x) for l in range(l_max + 1)]).T
    coef, *_ = np.linalg.lstsq(A, ratio, rcond=None)
    fit = A @ coef
    ax.axhline(1.0, color='#4C72B0', ls='--', lw=1, alpha=0.6, label='Equatorial mean $\\bar P$')
    ax.plot(phi_bin, fit, color='#DD8452', ls='--', lw=1.4,
            label='SH zonal fit ($\\ell_{max}=20$)')
    ax.scatter(phi_bin, ratio, color='#C44E52', s=14, zorder=5, label='PTM3D (binned)')
    ax.set_xlim(-90, 90)
    ax.set_ylim(0, 1.5)
    ax.set_ylabel(r'$P_v/\bar{P}$')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.text(0.02, 0.92, f'({label})', transform=ax.transAxes, fontweight='bold')
    return phi_bin, Peq


def _panel_net_lon(ax, label, Pbar):
    lam = np.linspace(-180, 180, 721)
    Phi_net = net_flux_1d(lam, T_lon, T_EQ)
    Phi_bar = Pbar * np.sqrt(M_H2O / (2 * np.pi * K_B * T_EQ))
    pct = 100.0 * Phi_net / Phi_bar
    rate = flux_to_um_myr(Phi_net)
    ax2 = ax.twinx()
    ax.fill_between(lam, 0, pct, where=(pct > 0), color='#C44E52', alpha=0.35,
                     label='Net ice removal')
    ax.fill_between(lam, 0, pct, where=(pct <= 0), color='#4C72B0', alpha=0.35,
                     label='Net ice accretion')
    ax.plot(lam, pct, color='#B33951', lw=1.3, label=r'$\Phi_{net}/\bar{\Phi}$')
    ax.set_xlim(-180, 180)
    ax.set_ylim(-8, 8)
    ax2.set_ylim(-8 * rate.max() / max(pct.max(), 1e-9),
                 8 * rate.max() / max(pct.max(), 1e-9))
    ax.set_xlabel('East longitude from sub-jovian point (°)')
    ax.set_ylabel(r'Net transfer rate (% of $\bar{\Phi}$)')
    ax2.set_ylabel(r'Ice rate ($\mu$m Myr$^{-1}$)' + '\n' + r'Net transfer rate (% of $\bar{\Phi}$)')
    ax.legend(loc='upper left', fontsize=7.5, framealpha=0.9)
    ax.text(0.02, 0.92, f'({label})', transform=ax.transAxes, fontweight='bold')


def _panel_net_lat(ax, label, Peq):
    # Restrict to +/-50 deg BEFORE computing: Phi_sub -> 0 near the poles
    # (T -> 0 there), so Phi_net/Phi_sub diverges outside this range and
    # must not be allowed to drive autoscaling.
    phi = np.linspace(-50, 50, 501)

    def T_of(p):
        return T_EQ * np.cos(np.deg2rad(p)) ** P_LAT

    Phi_net = net_flux_1d(phi, T_of, T_EQ)
    Phi_sub = sublimation_flux(T_of(phi))
    pct = 100.0 * Phi_net / Phi_sub
    rate = flux_to_um_myr(Phi_net)
    ax2 = ax.twinx()
    ax.fill_between(phi, 0, pct, where=(pct > 0), color='#C44E52', alpha=0.35,
                     label='Net ice removal')
    ax.fill_between(phi, 0, pct, where=(pct <= 0), color='#4C72B0', alpha=0.35,
                     label='Net ice accretion')
    ax.plot(phi, pct, color='#B33951', lw=1.3, label=r'$\Phi_{net}/\bar{\Phi}$')
    ax.set_xlim(-50, 50)
    ymax = 1.15 * max(abs(pct.min()), abs(pct.max()), 1e-9)
    ax.set_ylim(-ymax, ymax)
    ax2.set_ylim(-ymax * flux_to_um_myr(Phi_sub.max()) / 100.0,
                 ymax * flux_to_um_myr(Phi_sub.max()) / 100.0)
    ax2.set_ylabel(r'Ice rate ($\mu$m Myr$^{-1}$)' + '\n' + r'Net transfer rate (% of $\bar{\Phi}$)')
    ax.set_xlabel('Latitude (°)')
    ax.set_ylabel(r'Net transfer rate (% of $\bar{\Phi}$)')
    ax.legend(loc='upper left', fontsize=7.5, framealpha=0.9)
    ax.text(0.02, 0.92, f'({label})', transform=ax.transAxes, fontweight='bold')


def fig_analytical_transport(save_dir, mode='combined'):
    if mode == 'combined':
        fig, axes = plt.subplots(3, 2, figsize=(11, 9))
        _panel_temperature_lon(axes[0, 0], 'a')
        _panel_temperature_lat(axes[0, 1], 'b')
        _, Pbar = _panel_pv_lon(axes[1, 0], 'c')
        _, Peq = _panel_pv_lat(axes[1, 1], 'd')
        _panel_net_lon(axes[2, 0], 'e', Pbar)
        _panel_net_lat(axes[2, 1], 'f', Peq)
        name = 'analytical_transport_combined'
    elif mode == 'equatorial':
        fig, axes = plt.subplots(3, 1, figsize=(5.5, 9))
        _panel_temperature_lon(axes[0], 'a')
        _, Pbar = _panel_pv_lon(axes[1], 'c')
        _panel_net_lon(axes[2], 'e', Pbar)
        axes[1].set_xlabel('')
        name = 'analytical_transport_equatorial'
    else:
        fig, axes = plt.subplots(3, 1, figsize=(5.5, 9))
        _panel_temperature_lat(axes[0], 'b')
        _, Peq = _panel_pv_lat(axes[1], 'd')
        _panel_net_lat(axes[2], 'f', Peq)
        name = 'analytical_transport_latitudinal'
    fig.tight_layout()
    _save(fig, save_dir, name)


# ---------------------------------------------------------------------------
# Figure 4: latitudinal_transport_analysis
# ---------------------------------------------------------------------------

def fig_latitudinal_transport_analysis(save_dir):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    phi_full = np.linspace(-89.9, 89.9, 1000)
    T = T_EQ * np.cos(np.deg2rad(phi_full)) ** P_LAT

    ax = axes[0, 0]
    ax.plot(phi_full, T, color='#1f4e8c', lw=1.8)
    ax.set_xlabel('Latitude (°)')
    ax.set_ylabel('Orbit-averaged $T$ (K)')
    ax.set_title('(a) Temperature Profile', fontweight='bold')
    ax.set_xlim(-90, 90)

    ax = axes[0, 1]
    Pv = vapor_pressure(T)
    Peq = vapor_pressure(T_EQ)
    ax.plot(phi_full, np.log10(Pv / Peq), color='#B33951', lw=1.8)
    ax.axhline(0, color='0.5', ls='--', lw=1)
    ax.axhline(-2, color='#DD8452', ls=':', lw=1)
    ax.text(60, -1.6, '$P/P_{eq}=1\\%$', color='#DD8452', fontsize=8)
    ax.set_xlabel('Latitude (°)')
    ax.set_ylabel(r'$\log_{10}(P_v/P_{eq})$')
    ax.set_title('(b) Vapor Pressure vs Latitude', fontweight='bold')
    ax.set_xlim(-90, 90)

    ax = axes[1, 0]
    lam_full = np.linspace(-180, 180, 1000)
    Pbar = np.trapz(vapor_pressure(T_lon(np.linspace(-180, 180, 3601))),
                     np.linspace(-180, 180, 3601)) / 360.0
    lon_ratio = vapor_pressure(T_lon(lam_full)) / Pbar
    lon_ratio_norm = lon_ratio / lon_ratio.max()
    lat_ratio = Pv / Pv.max()
    ax.plot(lam_full, lon_ratio_norm, color='#C44E52', ls='--', lw=1.6,
            label=r'Longitudinal: $P(\lambda)/\bar{P}$')
    ax.plot(phi_full, lat_ratio, color='#4C72B0', lw=1.8,
            label=r'Latitudinal: $P(\varphi)/P_{eq}$')
    ax.axvline(90, color='0.7', ls=':', lw=0.8)
    ax.set_xlabel('Coordinate (°)')
    ax.set_ylabel('Normalized $P_v$')
    ax.set_title('(c) Latitudinal vs Longitudinal Variation', fontweight='bold')
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    phi_r = np.linspace(-50, 50, 500)

    def T_of(p):
        return T_EQ * np.cos(np.deg2rad(p)) ** P_LAT

    Phi_net = net_flux_1d(phi_r, T_of, T_EQ)
    Phi_sub = sublimation_flux(T_of(phi_r))
    pct = 100.0 * Phi_net / Phi_sub
    ax.fill_between(phi_r, 0, pct, where=(pct > 0), color='#C44E52', alpha=0.3,
                     label='Net ice removal')
    ax.fill_between(phi_r, 0, pct, where=(pct <= 0), color='#4C72B0', alpha=0.3,
                     label='Net ice accretion')
    ax.plot(phi_r, pct, color='#B33951', lw=1.4)
    ax.set_xlabel('Latitude (°)')
    ax.set_ylabel(r'$\Phi_{net}/\Phi_{sub}$ (%)')
    ax.set_title('(d) Latitudinal Diffusive Net Flux', fontweight='bold')
    ax.legend(fontsize=8, loc='lower center')

    fig.tight_layout()
    _save(fig, save_dir, 'latitudinal_transport_analysis')


# ---------------------------------------------------------------------------
# Figure 5-6: spherical_harmonic_transport, sh_transport_mollweide
# ---------------------------------------------------------------------------

def _pv_field(lat_g, lon_g):
    return vapor_pressure(T_field(lat_g, lon_g))


def fig_spherical_harmonic_transport(save_dir, l_max_bar=12, l_max_map=5):
    coeffs, lat_q, lon_q = sh_decompose(_pv_field, l_max=l_max_bar, n_theta=91, n_phi=181)

    S_l_total = np.zeros(l_max_bar + 1)
    S_l_zonal = np.zeros(l_max_bar + 1)
    for (l, m), c in coeffs.items():
        p = l * (l + 1) * (abs(c) ** 2)
        S_l_total[l] += p
        if m == 0:
            S_l_zonal[l] += p
    S_tot = S_l_total.sum()

    fig = plt.figure(figsize=(8, 12))
    gs = GridSpec(3, 1, height_ratios=[1.0, 1.0, 1.0], hspace=0.35)

    ax0 = fig.add_subplot(gs[0])
    ell = np.arange(1, l_max_bar + 1)
    ax0.bar(ell - 0.15, S_l_total[1:], width=0.3, color='#4C72B0', label=r'Total $\ell(\ell+1)\sum_m|c_{\ell m}|^2$')
    ax0.bar(ell + 0.15, S_l_zonal[1:], width=0.3, color='#C44E52', label='Zonal ($m=0$) only')
    ax0.axvline(5.5, color='0.5', ls='--', lw=1, label=r'$\ell_{max}=5$')
    for l in (2, 4, 6):
        pct = 100 * S_l_total[l] / S_tot
        ax0.text(l, S_l_total[l] * 1.03, f'{pct:.0f}%', ha='center', fontweight='bold')
    ax0.set_xlabel('Spherical harmonic degree $\\ell$')
    ax0.set_ylabel('Transport power (Pa$^2$)')
    ax0.set_title(f'(a) Transport Power Spectrum: $\\ell(\\ell+1)\\times$ Power', fontweight='bold')
    ax0.legend(fontsize=8)

    lat_g = np.linspace(-89, 89, 179)
    lon_g = np.linspace(0, 360, 361)
    LAT_G, LON_G = np.meshgrid(lat_g, lon_g, indexing='ij')
    logPv = np.log10(_pv_field(LAT_G, LON_G) / vapor_pressure(T_EQ))

    ax1 = fig.add_subplot(gs[1], projection='mollweide')
    pcm1 = pit.plot_mollweide(ax1, lat_g, lon_g, logPv, plt.get_cmap('inferno'),
                               matplotlib.colors.Normalize(vmin=-6, vmax=0))
    pit.add_mollweide_labels(ax1)
    ax1.set_title('(b) Vapor Pressure $\\log_{10}(P_v/P_{eq})$ — PTM3D', fontweight='bold')
    cb1 = fig.colorbar(pcm1, ax=ax1, orientation='horizontal', fraction=0.05, pad=0.08)
    cb1.set_label(r'$\log_{10}(P_v/P_{eq})$')

    coeffs5, _, _ = sh_decompose(_pv_field, l_max=l_max_map, n_theta=91, n_phi=181)
    net_map = D_B * sh_reconstruct(coeffs5, l_max_map, lat_g, lon_g,
                                    weight=lambda l: l * (l + 1))
    net_map *= np.sqrt(M_H2O / (2 * np.pi * K_B * T_EQ))
    rate_map = flux_to_um_myr(net_map)

    ax2 = fig.add_subplot(gs[2], projection='mollweide')
    norm2 = matplotlib.colors.TwoSlopeNorm(vcenter=0, vmin=-1.5, vmax=1.5)
    pcm2 = pit.plot_mollweide(ax2, lat_g, lon_g, rate_map, plt.get_cmap('RdBu_r'), norm2)
    pit.add_mollweide_labels(ax2)
    ax2.set_title(f'(c) Diffusive Net Transport Rate ($\\ell \\leq {l_max_map}$)', fontweight='bold')
    cb2 = fig.colorbar(pcm2, ax=ax2, orientation='horizontal', fraction=0.05, pad=0.08)
    cb2.set_label(r'Ice rate ($\mu$m Myr$^{-1}$): red = removal, blue = accretion')

    _save(fig, save_dir, 'spherical_harmonic_transport')


def fig_sh_transport_mollweide(save_dir, l_max_map=5):
    lat_g = np.linspace(-89, 89, 179)
    lon_g = np.linspace(0, 360, 361)
    LAT_G, LON_G = np.meshgrid(lat_g, lon_g, indexing='ij')
    T_map = T_field(LAT_G, LON_G)
    logPv = np.log10(_pv_field(LAT_G, LON_G) / vapor_pressure(T_EQ))

    coeffs5, _, _ = sh_decompose(_pv_field, l_max=l_max_map, n_theta=91, n_phi=181)
    net_map = D_B * sh_reconstruct(coeffs5, l_max_map, lat_g, lon_g,
                                    weight=lambda l: l * (l + 1))
    net_map *= np.sqrt(M_H2O / (2 * np.pi * K_B * T_EQ))
    rate_map = flux_to_um_myr(net_map)

    fig = plt.figure(figsize=(7, 11))
    fig.suptitle(f'Spherical Harmonic Diffusive Transport ($\\ell_{{max}}={l_max_map}$)',
                 fontsize=13)
    gs = GridSpec(3, 1, hspace=0.35, top=0.94)

    ax0 = fig.add_subplot(gs[0], projection='mollweide')
    pcm0 = pit.plot_mollweide(ax0, lat_g, lon_g, T_map, plt.get_cmap('inferno'),
                               matplotlib.colors.Normalize(vmin=40, vmax=110))
    pit.add_mollweide_labels(ax0)
    ax0.set_title('(a) Orbit-Averaged Temperature (K)', fontweight='bold')
    cb0 = fig.colorbar(pcm0, ax=ax0, orientation='horizontal', fraction=0.05, pad=0.08)
    cb0.set_label('T (K)')

    ax1 = fig.add_subplot(gs[1], projection='mollweide')
    pcm1 = pit.plot_mollweide(ax1, lat_g, lon_g, logPv, plt.get_cmap('inferno'),
                               matplotlib.colors.Normalize(vmin=-6, vmax=0))
    pit.add_mollweide_labels(ax1)
    ax1.set_title(r'(b) Vapor Pressure $\log_{10}(P_v/P_{eq})$', fontweight='bold')
    cb1 = fig.colorbar(pcm1, ax=ax1, orientation='horizontal', fraction=0.05, pad=0.08)
    cb1.set_label(r'$\log_{10}(P_v/P_{eq})$')

    ax2 = fig.add_subplot(gs[2], projection='mollweide')
    norm2 = matplotlib.colors.TwoSlopeNorm(vcenter=0, vmin=-1.5, vmax=1.5)
    pcm2 = pit.plot_mollweide(ax2, lat_g, lon_g, rate_map, plt.get_cmap('RdBu_r'), norm2)
    pit.add_mollweide_labels(ax2)
    ax2.set_title(r'(c) Net Ice Transport Rate ($\mu$m Myr$^{-1}$)', fontweight='bold')
    cb2 = fig.colorbar(pcm2, ax=ax2, orientation='horizontal', fraction=0.05, pad=0.08)
    cb2.set_label(r'Ice rate ($\mu$m Myr$^{-1}$): red = removal, blue = accretion')

    _save(fig, save_dir, 'sh_transport_mollweide')


# ---------------------------------------------------------------------------
# Figure 7-9: globe_temperature, globe_vapor_pressure, globe_ice_transport
# ---------------------------------------------------------------------------

def _globe_figure(save_dir, name, data_grid, cmap, norm, cbar_label, lat_g, lon_g):
    fig = plt.figure(figsize=(9, 9))
    for i, (title, elev, azim) in enumerate(pit.VIEWS):
        ax = fig.add_subplot(2, 2, i + 1, projection='3d')
        pit.plot_globe(ax, lat_g, lon_g, data_grid, cmap, norm, elev, azim, title=title)
    fig.suptitle(name, fontsize=14, fontweight='bold', y=0.98)
    cbar_ax = fig.add_axes([0.25, 0.06, 0.5, 0.02])
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(cbar_label)
    return fig


def fig_globe_temperature(save_dir):
    lat_g = np.linspace(-89, 89, 179)
    lon_g = np.linspace(0, 360, 361)
    LAT_G, LON_G = np.meshgrid(lat_g, lon_g, indexing='ij')
    T_map = T_field(LAT_G, LON_G)
    norm = matplotlib.colors.Normalize(vmin=50, vmax=110)
    fig = _globe_figure(save_dir, 'Orbit-Averaged Temperature (K)', T_map,
                         plt.get_cmap('plasma'), norm, 'T (K)', lat_g, lon_g)
    _save(fig, save_dir, 'globe_temperature')


def fig_globe_vapor_pressure(save_dir):
    lat_g = np.linspace(-89, 89, 179)
    lon_g = np.linspace(0, 360, 361)
    LAT_G, LON_G = np.meshgrid(lat_g, lon_g, indexing='ij')
    logPv = np.log10(_pv_field(LAT_G, LON_G))
    norm = matplotlib.colors.Normalize(vmin=-20, vmax=-12)
    fig = _globe_figure(save_dir, 'Orbit-Averaged Vapor Pressure', logPv,
                         plt.get_cmap('inferno'), norm, r'$\log_{10} P_v$ (Pa)', lat_g, lon_g)
    _save(fig, save_dir, 'globe_vapor_pressure')


def fig_globe_ice_transport(save_dir, l_max_map=5):
    lat_g = np.linspace(-89, 89, 179)
    lon_g = np.linspace(0, 360, 361)
    coeffs5, _, _ = sh_decompose(_pv_field, l_max=l_max_map, n_theta=91, n_phi=181)
    net_map = D_B * sh_reconstruct(coeffs5, l_max_map, lat_g, lon_g,
                                    weight=lambda l: l * (l + 1))
    net_map *= np.sqrt(M_H2O / (2 * np.pi * K_B * T_EQ))
    rate_map = flux_to_um_myr(net_map)
    norm = matplotlib.colors.TwoSlopeNorm(vcenter=0, vmin=-2.0, vmax=2.0)
    fig = _globe_figure(save_dir, 'Net ice sublimation/deposition rate', rate_map,
                         plt.get_cmap('RdBu_r'), norm,
                         r'Net ice sublimation/deposition rate ($\mu$m Myr$^{-1}$)',
                         lat_g, lon_g)
    _save(fig, save_dir, 'globe_ice_transport')


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def _save(fig, save_dir, name):
    os.makedirs(save_dir, exist_ok=True)
    for ext in ('.png', '.pdf'):
        fig.savefig(os.path.join(save_dir, name + ext))
    plt.close(fig)
    print(f'  Saved {name}.png / .pdf')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--save-dir', default='docs/figures/exosphere')
    args = p.parse_args()

    print('Analytical transport figures (closed-form, no PTM3D run):')
    fig_analytical_transport(args.save_dir, 'combined')
    fig_analytical_transport(args.save_dir, 'equatorial')
    fig_analytical_transport(args.save_dir, 'latitudinal')
    fig_latitudinal_transport_analysis(args.save_dir)
    fig_spherical_harmonic_transport(args.save_dir)
    fig_sh_transport_mollweide(args.save_dir)
    fig_globe_temperature(args.save_dir)
    fig_globe_vapor_pressure(args.save_dir)
    fig_globe_ice_transport(args.save_dir)
    print(f'Done. 9 figures written to {args.save_dir}')


if __name__ == '__main__':
    main()
