#!/usr/bin/env python3
"""Publication-quality 1D thermal model plots for Europa.

Generates three figure sets using heat1d:
  1. Diurnal temperature vs local time at multiple depths (equator)
  2. Depth profiles: min/max/mean envelopes + instantaneous snapshots
  3. Surface temperature diurnal curves for different thermal inertias

Usage:
    conda run -n europa_viz python scripts/plot_heat1d_europa.py \
        --save-dir plots/heat1d
"""

import os
import argparse
import copy
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Publication style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'mathtext.fontset': 'cm',
})

# Correct Klinger (1981) ice heat capacity: cp = 7.49*T + 90
# The installed planets.Europa has cpCoeff = [90.0, 7.49] (swapped!)
# giving cp = 90*T + 7.49 ~ 9000 J/kg/K at 100 K — 10x too high.
# We override to [7.49, 90.0] => cp = 7.49*T + 90 ~ 839 J/kg/K at 100 K.
_CORRECT_CP_COEFF = [7.49, 90.0]
_CORRECT_CP0 = 839.0  # at ~100 K, for grid/timestep consistency

# Standard uniform-property Europa regolith for all plots.
# Use uniform k, rho with depth (H very large) so that the labeled
# thermal inertia matches the effective thermal inertia.
# Target Gamma = 60 tiu (matching PTM3D Europa default).
# Gamma = sqrt(ks * rhos * cp) => ks = Gamma^2 / (rhos * cp)
# With rhos=500, cp(100K)=839: ks = 60^2 / (500*839) = 0.00858
_UNIFORM_PROPS = dict(
    ks=0.00858, kd=0.00858,    # thermal conductivity [W/m/K]
    rhos=500.0, rhod=500.0,    # density [kg/m³]  (solid porous ice)
    H=1e6,                      # no depth variation
)
# Gamma = sqrt(0.00858 * 500 * 839) = 60 tiu

# Diurnal skin depth for these properties [m]
# kappa = ks/(rhos*cp0) = 0.00858/(500*839) = 2.045e-8
# delta = sqrt(2.045e-8 * 306822 / pi) = 0.045 m
_SKIN_DEPTH_M = 0.045  # ~4.5 cm

# Eclipse parameters (from PTM3D body_positions.csv analysis).
# Eclipse occurs once per orbit when Europa passes behind Jupiter.
# Duration: ~9 timesteps out of 288 ≈ 9600 s ≈ 2.67 hours.
# For the sub-Jovian point (lon=0°), the eclipse is centered on local noon.
# For the anti-Jovian point (lon=180°), the eclipse is at local midnight.
# In heat1d coordinates, t=0 is local noon (hour angle h=0).
# Eclipse half-width in hour angle:
#   9/288 of orbit = 9/288 * 2π ≈ 0.196 rad ≈ 11.25°
_ECLIPSE_HALF_ANGLE = 9.0 / 288 * 2 * np.pi  # radians of hour angle


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def _save(fig, save_path):
    """Save figure in both PNG and PDF."""
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    base = save_path.rsplit('.', 1)[0]
    for ext in ['.png', '.pdf']:
        fig.savefig(base + ext)
    print(f'  Saved {base}.png / .pdf')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Model runner
# ---------------------------------------------------------------------------

def run_europa_model(lat=0, ndays=2, NPERDAY=480, NYEARSEQ=3,
                     ks=None, rhos=None, kd=None, rhod=None, Qb=None,
                     H=None):
    """Run heat1d for Europa and return the Model object.

    Runs *ndays* output days (default 2) so the caller can use the last
    diurnal cycle, which has less residual drift from equilibration.

    Parameters
    ----------
    lat : float
        Latitude in degrees.
    ndays : int
        Number of diurnal cycles to output.
    NPERDAY : int
        Time steps per diurnal cycle (higher = smoother curves).
    NYEARSEQ : int
        Number of orbital years to equilibrate.
    ks, rhos, kd, rhod, Qb, H : float or None
        Override Europa defaults if not None.
    """
    import heat1d
    import planets

    planet = copy.copy(planets.Europa)

    # Fix swapped cp coefficients: Klinger (1981) ice cp = 7.49*T + 90
    planet.cpCoeff = _CORRECT_CP_COEFF
    planet.cp0 = _CORRECT_CP0

    if ks is not None:
        planet.ks = ks
    if rhos is not None:
        planet.rhos = rhos
    if kd is not None:
        planet.kd = kd
    if rhod is not None:
        planet.rhod = rhod
    if Qb is not None:
        planet.Qb = Qb
    if H is not None:
        planet.H = H

    config = heat1d.Configurator(
        NYEARSEQ=NYEARSEQ,
        NPERDAY=NPERDAY,
        m=10, n=4, b=20,
    )
    model = heat1d.Model(planet=planet, lat=np.radians(lat),
                         ndays=ndays, config=config)
    model.run()

    # heat1d convention: t=0 is local noon (hour angle = 0).
    # Convert to standard local solar time where noon = 12h.
    model.lst = (model.lt + 12.0) % 24.0

    return model


def _extract_last_day(model, NPERDAY=480):
    """Extract the last diurnal cycle from a multi-day run.

    Returns (lst, T) arrays for one cycle, sorted by local solar time.
    Using the last day minimizes residual equilibration drift.
    """
    N = NPERDAY
    lst_day = model.lst[-N:]
    T_day = model.T[-N:, :]

    order = np.argsort(lst_day)
    return lst_day[order], T_day[order]


def run_europa_model_eclipse(eclipse_center_h=0.0, **kwargs):
    """Run heat1d with a Jupiter eclipse applied during the output phase.

    The model is first equilibrated *without* the eclipse (the eclipse is
    a brief perturbation that doesn't change the long-term mean), then
    the output days are re-integrated with the eclipse mask applied.

    Parameters
    ----------
    eclipse_center_h : float
        Hour angle of eclipse center in heat1d coordinates (radians).
        0 = local noon (sub-Jovian), π = local midnight (anti-Jovian).
    **kwargs :
        Passed to ``run_europa_model``.

    Returns
    -------
    model : heat1d.Model
        Model with eclipse-affected output temperatures.
    """
    import heat1d
    from heat1d.main import orbits, albedoVar

    # Run the standard model to equilibrate and get the output grid
    model = run_europa_model(**kwargs)

    # Re-run the output phase with eclipse
    # Reset to the equilibrated state by backing up the profile
    # and re-running from the end of equilibration.
    # Since model.run() already overwrites T[], we need to re-run.
    # The equilibrated profile is approximately model.profile.T at the
    # end of the first output day.  For simplicity, re-run the model
    # and intercept surfFlux.

    # Monkey-patch surfFlux to apply the eclipse
    original_surfFlux = model.surfFlux.__func__

    def eclipsed_surfFlux(self):
        original_surfFlux(self)
        # Hour angle of current timestep
        h = orbits.hourAngle(self.t, self.planet.day)
        # Angular distance from eclipse center (wrapping around 2π)
        dh = abs(((h - eclipse_center_h + np.pi) % (2 * np.pi)) - np.pi)
        if dh < _ECLIPSE_HALF_ANGLE:
            self.Qs = 0.0

    import types
    model.surfFlux = types.MethodType(eclipsed_surfFlux, model)

    # Re-equilibrate with eclipse active (run a few extra orbits)
    NPERDAY = kwargs.get('NPERDAY', 480)
    ndays = kwargs.get('ndays', 2)
    model.t = 0.0
    model.equiltime = kwargs.get('NYEARSEQ', 3) * model.planet.year
    model.equiltime -= model.equiltime % model.planet.day
    model.dt = model.dtout

    # Equilibrate with eclipse
    while model.t < model.equiltime:
        model.advance()

    # Output phase
    model.dt = model.dtout
    model.t = 0.0
    for i in range(model.N_steps):
        model.advance()
        model.T[i, :] = model.profile.T
        model.lt[i] = model.t / model.planet.day * 24.0

    model.lst = (model.lt + 12.0) % 24.0
    return model


# ---------------------------------------------------------------------------
# Figure 1: Diurnal T vs local time at multiple depths
# ---------------------------------------------------------------------------

def _select_depth_indices(depths_m, max_depth):
    """Select depth indices and labels for plotting."""
    target_depths = [0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20]
    target_depths = [d for d in target_depths if d <= max_depth]

    depth_indices = []
    depth_labels = []
    for td in target_depths:
        idx = np.argmin(np.abs(depths_m - td))
        if idx not in [i for i, _ in depth_indices]:
            depth_indices.append((idx, depths_m[idx]))
            if depths_m[idx] < 0.001:
                depth_labels.append('Surface')
            elif depths_m[idx] < 0.01:
                depth_labels.append(f'{depths_m[idx]*1000:.1f} mm')
            elif depths_m[idx] < 0.1:
                depth_labels.append(f'{depths_m[idx]*100:.1f} cm')
            else:
                depth_labels.append(f'{depths_m[idx]*100:.0f} cm')
    return depth_indices, depth_labels


def fig_diurnal_depths(save_dir):
    """Diurnal temperature at multiple depths for equatorial Europa.

    Plots both sub-Jovian (solid, with eclipse at local noon) and
    anti-Jovian (dashed, no eclipse effect) hemispheres.
    """
    print("Figure 1: Diurnal temperature at multiple depths...")

    NPERDAY = 480
    run_kw = dict(lat=0, ndays=2, NPERDAY=NPERDAY, NYEARSEQ=3,
                  **_UNIFORM_PROPS)

    # Anti-Jovian: eclipse at midnight (hour angle π) — no daytime effect
    print("  Running anti-Jovian (no eclipse)...")
    model_anti = run_europa_model(**run_kw)
    lst_anti, T_anti = _extract_last_day(model_anti, NPERDAY)

    # Sub-Jovian: eclipse centered at noon (hour angle 0)
    print("  Running sub-Jovian (eclipse at noon)...")
    model_sub = run_europa_model_eclipse(eclipse_center_h=0.0, **run_kw)
    lst_sub, T_sub = _extract_last_day(model_sub, NPERDAY)

    # Depth indices (same grid for both models)
    depths_m = model_anti.profile.z
    max_depth = 5 * _SKIN_DEPTH_M
    depth_indices, depth_labels = _select_depth_indices(depths_m, max_depth)

    # Colors: surface warm -> deep cool
    cmap = plt.cm.coolwarm_r
    n_depths = len(depth_indices)
    depth_colors = [cmap(i / (n_depths - 1)) for i in range(n_depths)]

    # Wrap for continuity
    lst_sub_plot = np.concatenate([lst_sub, [lst_sub[0] + 24]])
    lst_anti_plot = np.concatenate([lst_anti, [lst_anti[0] + 24]])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    panels = [
        (axes[0], lst_sub_plot, T_sub, 'Sub-Jovian (0\u00b0E)'),
        (axes[1], lst_anti_plot, T_anti, 'Anti-Jovian (180\u00b0E)'),
    ]

    for ax, lst_plot, T_data, title in panels:
        for i, ((didx, dval), label) in enumerate(zip(depth_indices, depth_labels)):
            T_col = np.concatenate([T_data[:, didx], [T_data[0, didx]]])
            ax.plot(lst_plot, T_col,
                    color=depth_colors[i], linewidth=1.8, label=label)

        ax.set_xlabel('Local Solar Time (hours)')
        ax.set_xlim(0, 24)
        ax.set_xticks(np.arange(0, 25, 3))
        ax.set_xticklabels(['0', '3', '6', '9', '12\n(noon)',
                             '15', '18', '21', '24'])
        ax.tick_params(direction='in', top=True, right=True)
        ax.set_title(title, fontsize=13)

    axes[0].set_ylabel('Temperature (K)')
    axes[1].legend(fontsize=10, loc='upper right', framealpha=0.9,
                   title='Depth', title_fontsize=11)

    fig.tight_layout()
    _save(fig, os.path.join(save_dir, 'diurnal_depth_equator.png'))


# ---------------------------------------------------------------------------
# Figure 2: Depth profiles — envelope + instantaneous snapshots
# ---------------------------------------------------------------------------

def fig_depth_profiles(save_dir):
    """Depth profile: min/max/mean envelopes and instantaneous T(z) snapshots.

    Shows two cases side-by-side: default heat flow (30 mW/m²) and
    elevated heat flow (100 mW/m²).
    """
    print("Figure 2: Depth profiles with min/max/mean envelope...")

    NPERDAY = 480
    max_depth_cm = 5 * _SKIN_DEPTH_M * 100  # 5 skin depths in cm

    cases = [
        {'Qb': 0.03,  'label': '$Q_b$ = 30 mW m$^{-2}$'},
        {'Qb': 0.10,  'label': '$Q_b$ = 100 mW m$^{-2}$'},
    ]

    fig, axes = plt.subplots(1, 2, figsize=(9, 5.5), sharey=True)

    for ci, case in enumerate(cases):
        ax = axes[ci]
        model = run_europa_model(lat=0, ndays=2, NPERDAY=NPERDAY, NYEARSEQ=3,
                                 Qb=case['Qb'], **_UNIFORM_PROPS)

        # Use last day for statistics
        N = NPERDAY
        T_day = model.T[-N:, :]
        T_min = T_day.min(axis=0)
        T_max = T_day.max(axis=0)
        T_mean = T_day.mean(axis=0)

        # Instantaneous profiles at specific local solar times
        lst_day = model.lst[-N:]
        target_lsts = np.arange(0, 24, 3)  # 0, 3, 6, ..., 21 h LST
        lt_indices = []
        for tlst in target_lsts:
            diffs = np.abs(lst_day - tlst)
            diffs_wrap = np.abs(lst_day - (tlst + 24))
            diffs = np.minimum(diffs, diffs_wrap)
            lt_indices.append(np.argmin(diffs))

        cmap_time = plt.cm.twilight
        lt_colors = [cmap_time(tlst / 24.0) for tlst in target_lsts]

        depths_cm = model.profile.z * 100

        # Min/max envelope
        ax.fill_betweenx(depths_cm, T_min, T_max, alpha=0.12, color='0.5',
                         label='Min\u2013Max range')

        # Mean profile
        ax.plot(T_mean, depths_cm, color='k', linewidth=2.0, linestyle='--',
                label='Diurnal mean', zorder=10)

        # Instantaneous profiles
        for i, (lt_idx, tlst) in enumerate(zip(lt_indices, target_lsts)):
            ax.plot(T_day[lt_idx, :], depths_cm, color=lt_colors[i],
                    linewidth=1.3, label=f'{int(tlst):02d}:00')

        ax.set_xlabel('Temperature (K)')
        ax.set_ylim(max_depth_cm, 0)
        ax.tick_params(direction='in', top=True, right=True)
        ax.set_title(case['label'], fontsize=13)

    axes[0].set_ylabel('Depth (cm)')
    axes[1].legend(fontsize=8, loc='lower left', framealpha=0.9, ncol=1)

    fig.tight_layout()
    _save(fig, os.path.join(save_dir, 'depth_profile_equator.png'))


# ---------------------------------------------------------------------------
# Figure 3: Surface T for different thermal inertias
# ---------------------------------------------------------------------------

def fig_thermal_inertia(save_dir):
    """Surface temperature diurnal curves for different thermal inertia values."""
    print("Figure 3: Surface T for different thermal inertias...")

    # Thermal inertia: Gamma = sqrt(k * rho * cp)
    # With corrected Klinger cp at ~100 K: cp ~ 839 J/kg/K
    # Use uniform properties (kd=ks, rhod=rhos) so that Gamma is constant
    # with depth and the surface thermal inertia fully controls the diurnal
    # wave.  Europa's H=0.07m otherwise causes rho and k to increase rapidly
    # within the skin depth, inflating the effective Gamma.
    cp_ref = _CORRECT_CP0
    rhos_val = 500.0  # kg/m³ (solid ice regolith)

    gamma_targets = [20, 40, 60, 80, 100, 200, 500]

    cmap = plt.cm.viridis
    n_gamma = len(gamma_targets)
    gamma_colors = [cmap(i / (n_gamma - 1)) for i in range(n_gamma)]

    NPERDAY = 480
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for i, gamma in enumerate(gamma_targets):
        ks = gamma**2 / (rhos_val * cp_ref)

        print(f'  Gamma={gamma} tiu: ks={ks:.5f}, rhos={rhos_val:.0f}')

        # Uniform properties with depth: kd=ks, rhod=rhos, H very large
        model = run_europa_model(lat=0, ndays=2, NPERDAY=NPERDAY, NYEARSEQ=3,
                                 ks=ks, kd=ks, rhos=rhos_val, rhod=rhos_val,
                                 H=1e6)
        lst, T = _extract_last_day(model, NPERDAY)

        # Wrap for continuity
        lst_plot = np.concatenate([lst, [lst[0] + 24]])
        T_plot = np.concatenate([T[:, 0], [T[0, 0]]])

        ax.plot(lst_plot, T_plot, color=gamma_colors[i],
                linewidth=1.8,
                label=f'$\\Gamma$ = {gamma} tiu')

    ax.set_xlabel('Local Solar Time (hours)')
    ax.set_ylabel('Surface Temperature (K)')
    ax.set_xlim(0, 24)
    ax.set_xticks(np.arange(0, 25, 3))
    ax.set_xticklabels(['0', '3', '6', '9', '12\n(noon)', '15', '18', '21', '24'])
    ax.tick_params(direction='in', top=True, right=True)
    ax.legend(fontsize=10, loc='upper right', framealpha=0.9)

    fig.tight_layout()
    _save(fig, os.path.join(save_dir, 'thermal_inertia_comparison.png'))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Publication-quality 1D thermal model plots for Europa')
    parser.add_argument('--save-dir', default='plots/heat1d',
                        help='Output directory for figures')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    fig_diurnal_depths(args.save_dir)
    fig_depth_profiles(args.save_dir)
    fig_thermal_inertia(args.save_dir)

    print("\nAll figures generated!")


if __name__ == '__main__':
    main()
