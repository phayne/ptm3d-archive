"""Visualization functions for the O&S ice shell model.

Reproduces Figs 1-5 from Ojakangas & Stevenson (1989).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker


def plot_strain_pattern(theta=None, lam=None, eij2=None, save_path=None):
    """Plot time-averaged squared strain rate <e_ij^2> (reproduces Fig 1).

    Parameters
    ----------
    theta, lam, eij2 : ndarray, optional
        Pre-computed grid. If None, computes from defaults.
    save_path : str, optional
        If provided, saves figure to this path.
    """
    if eij2 is None:
        from .tidal_strains import strain_squared_grid
        theta, lam, eij2 = strain_squared_grid()

    fig, ax = plt.subplots(figsize=(10, 6))
    lam_deg = np.degrees(lam)
    theta_deg = np.degrees(theta)

    # Normalize to the Fig. 1 unit 0.5*(n*gamma*e*C_R)^2 (~1.2e-21 s^-2).
    # (Before 2026-10 this used g2 = (45/88) gamma e C_R in place of
    # gamma e C_R, which rescaled the contours by (88/45)^2.)
    from . import constants as C
    norm_factor = 0.5 * (C.MEAN_MOTION * C.gamma() * C.ECCENTRICITY * C.C_R)**2
    eij2_norm = eij2 * C.MEAN_MOTION**2 / norm_factor

    cs = ax.contourf(lam_deg, theta_deg, eij2_norm, levels=20, cmap='hot_r')
    ax.contour(lam_deg, theta_deg, eij2_norm, levels=10, colors='k',
               linewidths=0.5, alpha=0.5)
    cbar = plt.colorbar(cs, ax=ax)
    cbar.set_label(r'$\langle\dot{\epsilon}_{ij}^2\rangle$ (normalized)')
    ax.set_xlabel('Longitude [deg]')
    ax.set_ylabel('Colatitude [deg]')
    ax.set_title(r'Time-averaged squared strain rate $\langle\dot{\epsilon}_{ij}^2\rangle$'
                 '\n(cf. O&S 1989 Fig. 1)')
    ax.invert_yaxis()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig, ax


def plot_shell_thickness_contour(model, title=None, save_path=None,
                                  n_theta=181, n_lam=360):
    """Contour plot of shell thickness b(theta, lam) (reproduces Figs 2-5).

    Parameters
    ----------
    model : IceShellModel
        Model to plot.
    title : str, optional
        Plot title.
    save_path : str, optional
        If provided, saves figure.
    """
    theta, lam, b = model.thickness_grid(n_theta, n_lam)
    b_km = b / 1000.0

    fig, ax = plt.subplots(figsize=(10, 6))
    lam_deg = np.degrees(lam)
    theta_deg = np.degrees(theta)

    # Determine contour levels
    b_min = max(0, np.nanmin(b_km[np.isfinite(b_km)]))
    b_max = min(np.nanmax(b_km[np.isfinite(b_km)]), 100)
    levels = np.linspace(b_min, b_max, 21)

    cs = ax.contourf(lam_deg, theta_deg, b_km, levels=levels, cmap='viridis_r')
    ax.contour(lam_deg, theta_deg, b_km, levels=levels[::2], colors='k',
               linewidths=0.5, alpha=0.5)
    cbar = plt.colorbar(cs, ax=ax)
    cbar.set_label('Shell thickness [km]')
    ax.set_xlabel('Longitude [deg]')
    ax.set_ylabel('Colatitude [deg]')
    ax.invert_yaxis()

    if title is None:
        avg = model.average_thickness() / 1000.0
        title = f'{model.rheology.name}, H={model.H_flux*1e3:.0f} mW/m²\nAvg thickness: {avg:.1f} km'
    ax.set_title(title)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig, ax


def plot_surface_temperature(temp_model=None, save_path=None):
    """Plot surface temperature vs. colatitude.

    Parameters
    ----------
    temp_model : object, optional
        Surface temperature model. Default: OSSurfaceTemperature().
    """
    if temp_model is None:
        from .surface_temperature import OSSurfaceTemperature
        temp_model = OSSurfaceTemperature()

    theta = np.linspace(0.001, np.pi - 0.001, 500)
    T = temp_model.temperature(theta)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.degrees(theta), T, 'b-', linewidth=2)
    ax.set_xlabel('Colatitude [deg]')
    ax.set_ylabel('Surface temperature [K]')
    ax.set_title('Annual-average surface temperature (O&S 1989 Appendix A)')
    ax.axhline(110, color='r', linestyle='--', alpha=0.5, label='T_eq ~ 110 K')
    ax.axhline(52, color='b', linestyle='--', alpha=0.5, label='T_pole ~ 52 K')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 180)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig, ax


def plot_thickness_vs_colatitude(model, lam_values=None, save_path=None):
    """Shell thickness vs. colatitude at selected longitudes.

    Parameters
    ----------
    model : IceShellModel
    lam_values : list of float, optional
        Longitudes [rad] to plot. Default: [0, pi/4, pi/2, 3pi/4].
    """
    if lam_values is None:
        lam_values = [0, np.pi/4, np.pi/2, 3*np.pi/4]

    theta = np.linspace(0.001, np.pi - 0.001, 500)

    fig, ax = plt.subplots(figsize=(8, 5))
    for lam in lam_values:
        b = model.thickness(theta, np.full_like(theta, lam))
        b_km = b / 1000.0
        ax.plot(np.degrees(theta), b_km, linewidth=1.5,
                label=f'lon = {np.degrees(lam):.0f} deg')

    ax.set_xlabel('Colatitude [deg]')
    ax.set_ylabel('Shell thickness [km]')
    avg = model.average_thickness() / 1000.0
    ax.set_title(f'{model.rheology.name} rheology, avg = {avg:.1f} km')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 180)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig, ax


def plot_comparison(model_os, model_ptm, save_path=None):
    """Compare shell thickness from O&S vs PTM3D surface temperatures.

    Parameters
    ----------
    model_os : IceShellModel
        Model using O&S surface temperatures.
    model_ptm : IceShellModel
        Model using PTM3D surface temperatures.
    """
    theta_os, lam_os, b_os = model_os.thickness_grid()
    theta_ptm, lam_ptm, b_ptm = model_ptm.thickness_grid()

    b_os_km = b_os / 1000.0
    b_ptm_km = b_ptm / 1000.0

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    lam_deg = np.degrees(lam_os)
    theta_deg = np.degrees(theta_os)

    b_max = max(np.nanmax(b_os_km[np.isfinite(b_os_km)]),
                np.nanmax(b_ptm_km[np.isfinite(b_ptm_km)]))
    b_max = min(b_max, 100)
    levels = np.linspace(0, b_max, 21)

    # O&S temperatures
    cs0 = axes[0].contourf(lam_deg, theta_deg, b_os_km, levels=levels, cmap='viridis_r')
    plt.colorbar(cs0, ax=axes[0], label='km')
    axes[0].set_title('O&S surface temperature')

    # PTM3D temperatures
    cs1 = axes[1].contourf(lam_deg, theta_deg, b_ptm_km, levels=levels, cmap='viridis_r')
    plt.colorbar(cs1, ax=axes[1], label='km')
    axes[1].set_title('PTM3D surface temperature')

    # Difference
    diff = b_ptm_km - b_os_km
    d_max = max(abs(np.nanmin(diff[np.isfinite(diff)])),
                abs(np.nanmax(diff[np.isfinite(diff)])))
    d_max = min(d_max, 20)
    d_levels = np.linspace(-d_max, d_max, 21)
    cs2 = axes[2].contourf(lam_deg, theta_deg, diff, levels=d_levels, cmap='RdBu_r')
    plt.colorbar(cs2, ax=axes[2], label='km')
    axes[2].set_title('PTM3D - O&S')

    for ax in axes:
        ax.set_xlabel('Longitude [deg]')
        ax.set_ylabel('Colatitude [deg]')
        ax.invert_yaxis()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig, axes


def plot_validation_table(results_dict, save_path=None):
    """Bar chart comparing computed values to O&S Table I.

    Parameters
    ----------
    results_dict : dict
        Keys are case labels, values are dicts with:
        'computed_avg_km', 'paper_avg_km', '4t22_km', 'paper_4t22_km',
        't20_2t22_km', 'paper_t20_2t22_km'.
    """
    cases = list(results_dict.keys())
    n = len(cases)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    x = np.arange(n)
    width = 0.35

    for idx, (key, ylabel) in enumerate([
        ('avg_km', 'Average thickness [km]'),
        ('4t22_km', '4t₂₂ [km]'),
        ('t20_2t22_km', 't₂₀ + 2t₂₂ [km]')
    ]):
        computed = [results_dict[c][f'computed_{key}'] for c in cases]
        paper = [results_dict[c][f'paper_{key}'] for c in cases]

        axes[idx].bar(x - width/2, computed, width, label='Computed', color='steelblue')
        axes[idx].bar(x + width/2, paper, width, label='Paper', color='coral')
        axes[idx].set_ylabel(ylabel)
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(cases, rotation=30, ha='right')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3, axis='y')

    plt.suptitle('Validation against O&S (1989) Table I')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig, axes
