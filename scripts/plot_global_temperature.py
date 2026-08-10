#!/usr/bin/env python3
"""Global Europa surface temperature maps (globe + Mollweide).

Produces publication-quality figures of diurnal min, max, and mean surface
temperature for one or more simulation cases (e.g. uniform albedo vs albedo
map).  Output includes:
  - 1x4 globe views per case per statistic
  - Mollweide projections per case per statistic
  - Combined 3x2 Mollweide panel (rows = T_max, T_mean, T_min; cols = cases)

Usage:
    python scripts/plot_global_temperature.py \\
        --cases output/europa_ico6_uniform-ice output/europa_ico6_albedo-map \\
        --labels "Uniform albedo" "Albedo map" \\
        --save-dir plots/global_temperature
"""

import sys
import os
import struct
import argparse
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm, colors, ticker
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata


# ---------------------------------------------------------------------------
# Publication style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'mathtext.fontset': 'cm',
})


# ---------------------------------------------------------------------------
# Data I/O
# ---------------------------------------------------------------------------

def read_ply_mesh(ply_path):
    """Read binary PLY mesh, return face latitudes and longitudes in degrees.

    Handles both float and double vertex properties by parsing the header.
    """
    with open(ply_path, 'rb') as f:
        n_vertices = 0
        n_faces = 0
        vertex_bytes = 0
        in_vertex = False
        in_face = False
        type_sizes = {'float': 4, 'double': 8, 'int': 4, 'uint': 4,
                      'uchar': 1, 'short': 2, 'ushort': 2, 'int32': 4,
                      'uint32': 4, 'float32': 4, 'float64': 8}

        while True:
            line = f.readline().decode('ascii').strip()
            if line.startswith('element vertex'):
                n_vertices = int(line.split()[-1])
                in_vertex = True
                in_face = False
            elif line.startswith('element face'):
                n_faces = int(line.split()[-1])
                in_vertex = False
                in_face = True
            elif line.startswith('property') and in_vertex:
                # property <type> <name>
                parts = line.split()
                if len(parts) >= 3 and parts[1] != 'list':
                    vertex_bytes += type_sizes.get(parts[1], 4)
            elif line == 'end_header':
                break

        # Skip vertex data
        f.seek(f.tell() + n_vertices * vertex_bytes)

        # Read face data: uchar(1) + 3×int(12) + float lat(4) + float lon(4) = 21 bytes
        lats = np.zeros(n_faces)
        lons = np.zeros(n_faces)
        for i in range(n_faces):
            data = f.read(21)
            lat, lon = struct.unpack('<ff', data[13:21])
            lats[i] = lat
            lons[i] = lon
    return lats, lons, n_faces


def read_binary_field(filepath):
    """Read a binary field file: uint32 count + float32 array."""
    with open(filepath, 'rb') as f:
        count = struct.unpack('<I', f.read(4))[0]
        data = np.frombuffer(f.read(count * 4), dtype='<f4')
    return data.astype(np.float64)


# ---------------------------------------------------------------------------
# Regridding
# ---------------------------------------------------------------------------

def regrid_to_latlon(lats, lons, values, n_lat=181, n_lon=361):
    """Regrid unstructured face data to regular lat/lon grid."""
    lat_grid = np.linspace(-90, 90, n_lat)
    lon_grid = np.linspace(0, 360, n_lon)
    LON_G, LAT_G = np.meshgrid(lon_grid, lat_grid)

    lons_norm = lons % 360
    margin = 10.0
    mask_low = lons_norm < margin
    mask_high = lons_norm > (360 - margin)

    lats_ext = np.concatenate([lats, lats[mask_low], lats[mask_high]])
    lons_ext = np.concatenate([lons_norm, lons_norm[mask_low] + 360,
                               lons_norm[mask_high] - 360])
    vals_ext = np.concatenate([values, values[mask_low], values[mask_high]])

    data_grid = griddata(
        (lons_ext, lats_ext), vals_ext,
        (LON_G, LAT_G), method='linear'
    )
    mask_nan = np.isnan(data_grid)
    if np.any(mask_nan):
        data_nn = griddata(
            (lons_ext, lats_ext), vals_ext,
            (LON_G, LAT_G), method='nearest'
        )
        data_grid[mask_nan] = data_nn[mask_nan]

    return lat_grid, lon_grid, data_grid


# ---------------------------------------------------------------------------
# 3D Globe rendering
# ---------------------------------------------------------------------------

def make_globe_xyz(lat_grid, lon_grid):
    """Create X, Y, Z coordinates for a globe surface."""
    lat_rad = np.deg2rad(lat_grid)
    lon_rad = np.deg2rad(lon_grid)
    LON, LAT = np.meshgrid(lon_rad, lat_rad)
    X = np.cos(LAT) * np.cos(LON)
    Y = np.cos(LAT) * np.sin(LON)
    Z = np.sin(LAT)
    return X, Y, Z


def draw_graticule(ax, elev, azim, spacing=30, color='0.55', linewidth=0.2,
                   alpha=0.35):
    """Draw latitude/longitude graticule lines on a 3D globe."""
    R = 1.002
    elev_r = np.deg2rad(elev)
    azim_r = np.deg2rad(azim)
    vx = np.cos(elev_r) * np.cos(azim_r)
    vy = np.cos(elev_r) * np.sin(azim_r)
    vz = np.sin(elev_r)

    t = np.linspace(0, 2 * np.pi, 360)

    for lat_deg in range(-90 + spacing, 90, spacing):
        lat_r = np.deg2rad(lat_deg)
        x = R * np.cos(lat_r) * np.cos(t)
        y = R * np.cos(lat_r) * np.sin(t)
        z = R * np.sin(lat_r) * np.ones_like(t)
        dot = x * vx + y * vy + z * vz
        x_vis = np.where(dot > 0, x, np.nan)
        y_vis = np.where(dot > 0, y, np.nan)
        z_vis = np.where(dot > 0, z, np.nan)
        ax.plot(x_vis, y_vis, z_vis, color=color, linewidth=linewidth,
                alpha=alpha, zorder=10)

    lat_t = np.linspace(-np.pi / 2, np.pi / 2, 180)
    for lon_deg in range(0, 360, spacing):
        lon_r = np.deg2rad(lon_deg)
        x = R * np.cos(lat_t) * np.cos(lon_r)
        y = R * np.cos(lat_t) * np.sin(lon_r)
        z = R * np.sin(lat_t)
        dot = x * vx + y * vy + z * vz
        x_vis = np.where(dot > 0, x, np.nan)
        y_vis = np.where(dot > 0, y, np.nan)
        z_vis = np.where(dot > 0, z, np.nan)
        ax.plot(x_vis, y_vis, z_vis, color=color, linewidth=linewidth,
                alpha=alpha, zorder=10)


def plot_globe(ax, lat_grid, lon_grid, data_grid, cmap, norm, elev, azim,
               title=None, graticule=True):
    """Plot a single globe view on a 3D axis."""
    X, Y, Z = make_globe_xyz(lat_grid, lon_grid)
    facecolors = cmap(norm(data_grid))

    ax.plot_surface(X, Y, Z, facecolors=facecolors,
                    rstride=1, cstride=1, shade=False,
                    antialiased=False, rasterized=True)

    if graticule:
        draw_graticule(ax, elev, azim)

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_zlim(-1.1, 1.1)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_box_aspect([1, 1, 1])
    if title:
        ax.set_title(title, fontsize=9, pad=-6)


# ---------------------------------------------------------------------------
# Mollweide projection
# ---------------------------------------------------------------------------

def plot_mollweide(ax, lat_grid, lon_grid, data_grid, cmap, norm,
                   lon_center=0):
    """Plot data on a Mollweide projection axis."""
    lon_shifted = lon_grid - lon_center
    lon_shifted = (lon_shifted + 180) % 360 - 180
    order = np.argsort(lon_shifted)
    lon_sorted = lon_shifted[order]
    data_sorted = data_grid[:, order]

    lat_rad = np.deg2rad(lat_grid)
    lon_rad = np.deg2rad(lon_sorted)
    LON, LAT = np.meshgrid(lon_rad, lat_rad)

    pcm = ax.pcolormesh(LON, LAT, data_sorted, cmap=cmap, norm=norm,
                        shading='auto', rasterized=True)

    ax.grid(True, color='0.3', linewidth=0.3, alpha=0.4)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    return pcm


def add_mollweide_labels(ax, lon_center=0):
    """Add longitude/latitude tick labels around a Mollweide plot."""
    for lat_deg in [-60, -30, 0, 30, 60]:
        ax.text(np.deg2rad(185 - lon_center), np.deg2rad(lat_deg),
                f'{abs(lat_deg)}\u00b0{"N" if lat_deg > 0 else "S" if lat_deg < 0 else ""}',
                fontsize=5.5, color='0.4', ha='left', va='center',
                transform=ax.transData)

    for lon_deg in [0, 90, 180, 270]:
        lon_plot = lon_deg - lon_center
        if lon_plot > 180:
            lon_plot -= 360
        elif lon_plot < -180:
            lon_plot += 360
        if abs(lon_plot) > 170:
            continue
        ax.text(np.deg2rad(lon_plot), np.deg2rad(-8),
                f'{lon_deg}\u00b0E',
                fontsize=5.5, color='0.4', ha='center', va='top',
                transform=ax.transData)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VIEWS = [
    ('Sub-Jovian (0\u00b0)',    15,   0),
    ('Trailing (90\u00b0E)',    15,  90),
    ('Anti-Jovian (180\u00b0)', 15, 180),
    ('Leading (270\u00b0E)',    15, 270),
]

# Default colormap: truncated inferno (skip very dark bottom 10%)
_inferno_full = plt.get_cmap('inferno')
CMAP_DEFAULT = colors.LinearSegmentedColormap.from_list(
    'inferno_trunc', _inferno_full(np.linspace(0.08, 1.0, 256)))
CMAP = CMAP_DEFAULT  # overridden by --cmap at runtime
CBAR_LABEL = 'Temperature (K)'

def make_temp_norm(vmin, vmax):
    """Create a Normalize spanning [vmin, vmax].

    Uses linear scaling so that the full colormap dynamic range is allocated
    evenly.  Pair with a judiciously chosen vmin (see _stat_norm) to focus the
    color stretch on the temperature range of interest.
    """
    return colors.Normalize(vmin=vmin, vmax=vmax)


# Statistics to load
STATS = [
    ('tmax',  'temperature_max.bin',  r'$T_\mathrm{max}$'),
    ('tmean', 'temperature_avg.bin',  r'$T_\mathrm{mean}$'),
    ('tmin',  'temperature_min.bin',  r'$T_\mathrm{min}$'),
]

# Latitude cutoff for computing vmin: only temperatures at |lat| < this value
# are considered.  Polar regions (which are much colder) saturate at the dark
# end of the colormap, letting the full dynamic range highlight equatorial and
# mid-latitude variations (e.g. eclipse cooling).
_LAT_CUTOFF = 45  # degrees


def _stat_norm(cases, stat_key):
    """Compute a shared Normalize for *stat_key* across all *cases*.

    For T_max the vmin is derived from mid-latitude data only (|lat| < 45°),
    so the colormap concentrates on the equatorial temperature range where
    eclipse cooling and albedo signals are visible.  For T_mean and T_min the
    full data range (with small percentile clips) is used.
    """
    all_raw = np.concatenate([c[stat_key]['raw'] for c in cases])

    if stat_key == 'tmax':
        # Use only mid-latitude faces for vmin
        mid_vals = []
        for c in cases:
            mask = np.abs(c['lats']) < _LAT_CUTOFF
            mid_vals.append(c[stat_key]['raw'][mask])
        mid_raw = np.concatenate(mid_vals)
        vmin = np.nanpercentile(mid_raw, 0.5)
        vmax = np.nanpercentile(all_raw, 99.5)
    else:
        vmin = np.nanpercentile(all_raw, 0.5)
        vmax = np.nanpercentile(all_raw, 99.5)

    return make_temp_norm(vmin, vmax)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_case(data_dir, label):
    """Load mesh + temperature statistics for a case. Returns dict."""
    ply_path = os.path.join(data_dir, 'mesh.ply')
    lats, lons, nf = read_ply_mesh(ply_path)
    print(f"  {label}: {nf} faces")

    case = {'lats': lats, 'lons': lons, 'n_faces': nf, 'label': label}

    for stat_key, stat_file, stat_label in STATS:
        fpath = os.path.join(data_dir, 'statistics', stat_file)
        raw = read_binary_field(fpath)
        lat_g, lon_g, data_g = regrid_to_latlon(lats, lons, raw)
        case[stat_key] = {
            'raw': raw,
            'lat_g': lat_g,
            'lon_g': lon_g,
            'data_g': data_g,
            'label': stat_label,
        }
        print(f"    {stat_label}: [{raw.min():.1f}, {raw.max():.1f}] K, "
              f"mean={raw.mean():.1f} K")

    return case


# ---------------------------------------------------------------------------
# Figure: single-case globes (1x4 row)
# ---------------------------------------------------------------------------

def fig_single_globes(case, stat_key, save_path, norm=None):
    """1x4 row of globe views for a single case + statistic."""
    stat = case[stat_key]
    lat_g = stat['lat_g']
    lon_g = stat['lon_g']
    data_g = stat['data_g']

    if norm is None:
        norm = make_temp_norm(stat['raw'].min(), stat['raw'].max())

    fig = plt.figure(figsize=(7.2, 2.4))
    gs = GridSpec(1, 4, figure=fig, wspace=0.01,
                  left=0.01, right=0.88, top=0.92, bottom=0.12)

    for idx, (view_name, elev, azim) in enumerate(VIEWS):
        ax = fig.add_subplot(gs[0, idx], projection='3d',
                             computed_zorder=False)
        plot_globe(ax, lat_g, lon_g, data_g, CMAP, norm, elev, azim,
                   title=view_name)

    # Colorbar
    sm = cm.ScalarMappable(norm=norm, cmap=CMAP)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.70])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
    cbar.set_label(CBAR_LABEL, fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    title = f"{case['label']} \u2014 {stat['label']}"
    fig.suptitle(title, fontsize=11, y=0.99)

    _save(fig, save_path)


# ---------------------------------------------------------------------------
# Figure: single-case Mollweide
# ---------------------------------------------------------------------------

def fig_single_mollweide(case, stat_key, save_path, norm=None):
    """Single Mollweide projection for one case + statistic."""
    stat = case[stat_key]

    if norm is None:
        norm = make_temp_norm(stat['raw'].min(), stat['raw'].max())

    fig = plt.figure(figsize=(5.5, 3.0))
    ax = fig.add_subplot(111, projection='mollweide')

    pcm = plot_mollweide(ax, stat['lat_g'], stat['lon_g'],
                         stat['data_g'], CMAP, norm)
    add_mollweide_labels(ax)

    cbar = fig.colorbar(pcm, ax=ax, orientation='horizontal',
                        fraction=0.06, pad=0.04, aspect=35)
    cbar.set_label(CBAR_LABEL, fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    title = f"{case['label']} \u2014 {stat['label']}"
    ax.set_title(title, fontsize=10, pad=10)

    _save(fig, save_path)


# ---------------------------------------------------------------------------
# Figure: combined 3x2 Mollweide panel
# ---------------------------------------------------------------------------

def fig_combined_mollweide(cases, save_path):
    """Combined panel: rows = T_max, T_mean, T_min; columns = cases.

    Shared colorbar per row so the two cases are directly comparable.
    """
    n_cases = len(cases)
    n_stats = len(STATS)

    fig = plt.figure(figsize=(5.5 * n_cases + 0.8, 2.6 * n_stats + 0.8))
    gs = GridSpec(n_stats, n_cases, figure=fig, hspace=0.30, wspace=0.08,
                  left=0.05, right=0.88, top=0.93, bottom=0.05)

    panel_labels = 'abcdefghij'
    panel_idx = 0

    for row, (stat_key, stat_file, stat_label) in enumerate(STATS):
        norm = _stat_norm(cases, stat_key)

        pcm = None
        for col, case in enumerate(cases):
            stat = case[stat_key]
            ax = fig.add_subplot(gs[row, col], projection='mollweide')
            pcm = plot_mollweide(ax, stat['lat_g'], stat['lon_g'],
                                 stat['data_g'], CMAP, norm)
            add_mollweide_labels(ax)

            plabel = f"({panel_labels[panel_idx]})"
            if row == 0:
                ax.set_title(f"{plabel}  {case['label']}",
                             fontsize=10, loc='left', pad=8)
            else:
                ax.set_title(f"{plabel}", fontsize=10, loc='left', pad=8)
            panel_idx += 1

        # Row colorbar on the right
        # Position: right of the grid, vertically centered on this row
        row_pos = gs[row, -1].get_position(fig)
        cbar_ax = fig.add_axes([0.90, row_pos.y0 + 0.02,
                                0.015, row_pos.height - 0.04])
        cbar = fig.colorbar(pcm, cax=cbar_ax, orientation='vertical')
        cbar.set_label(f'{stat_label} (K)', fontsize=9)
        cbar.ax.tick_params(labelsize=8)

    _save(fig, save_path)


# ---------------------------------------------------------------------------
# Figure: comparison globes (2 columns x 4 rows) for one statistic
# ---------------------------------------------------------------------------

def fig_comparison_globes(cases, stat_key, save_path):
    """Side-by-side globe comparison, 4 views x N cases, one statistic."""
    n = len(cases)
    stat_label = dict((s[0], s[2]) for s in STATS)[stat_key]

    norm = _stat_norm(cases, stat_key)

    fig = plt.figure(figsize=(3.2 * n + 0.8, 10.5))
    gs = GridSpec(4, n, figure=fig, hspace=0.08, wspace=-0.05,
                  left=0.01, right=0.84, top=0.95, bottom=0.02)

    for col, case in enumerate(cases):
        stat = case[stat_key]
        for row, (view_name, elev, azim) in enumerate(VIEWS):
            ax = fig.add_subplot(gs[row, col], projection='3d',
                                 computed_zorder=False)
            plot_globe(ax, stat['lat_g'], stat['lon_g'],
                       stat['data_g'], CMAP, norm, elev, azim)

        # Column header
        x_center = 0.01 + (col + 0.5) * 0.83 / n
        fig.text(x_center, 0.97, case['label'],
                 fontsize=10, ha='center', va='center')

    # View labels
    x_label = 0.01 + 0.83 / 2.0
    for row, (view_name, _, _) in enumerate(VIEWS):
        y_top = gs[row, 0].get_position(fig).y1
        y_bot = gs[row, 0].get_position(fig).y0
        y_mid = (y_top + y_bot) / 2.0
        fig.text(x_label, y_mid, view_name,
                 fontsize=8, ha='center', va='center', color='0.3')

    # Colorbar
    sm = cm.ScalarMappable(norm=norm, cmap=CMAP)
    cbar_ax = fig.add_axes([0.87, 0.10, 0.018, 0.78])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
    cbar.set_label(f'{stat_label} (K)', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    _save(fig, save_path)


# ---------------------------------------------------------------------------
# Figure: temperature profiles vs latitude and longitude
# ---------------------------------------------------------------------------

# Line styles/colors for the three statistics
_STAT_STYLES = {
    'tmax':  dict(color='C3', label=r'$T_\mathrm{max}$'),
    'tmean': dict(color='C0', label=r'$T_\mathrm{mean}$'),
    'tmin':  dict(color='C4', label=r'$T_\mathrm{min}$'),
}

# Case line styles (solid for first, dashed for second, etc.)
_CASE_LINESTYLES = ['-', '--', '-.', ':']


def _zonal_profile(lat_g, data_g):
    """Average data_g over longitude → T(lat)."""
    return np.nanmean(data_g, axis=1), lat_g


def _meridional_profile(lon_g, data_g, lat_g, lat_band=30):
    """Average data_g over low latitudes (|lat| < lat_band) → T(lon)."""
    mask = np.abs(lat_g) <= lat_band
    return np.nanmean(data_g[mask, :], axis=0), lon_g


def fig_profiles_vs_latitude(cases, save_path):
    """T_min, T_mean, T_max vs latitude for all cases (zonal average)."""
    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    for ci, case in enumerate(cases):
        ls = _CASE_LINESTYLES[ci % len(_CASE_LINESTYLES)]
        for stat_key, _, _ in STATS:
            stat = case[stat_key]
            profile, lats = _zonal_profile(stat['lat_g'], stat['data_g'])
            sty = _STAT_STYLES[stat_key]
            lbl = f"{sty['label']} — {case['label']}" if ci == 0 or True else None
            ax.plot(lats, profile, color=sty['color'], ls=ls,
                    linewidth=1.2, label=lbl)

    ax.set_xlabel('Latitude (\u00b0N)')
    ax.set_ylabel('Temperature (K)')
    ax.set_xlim(-90, 90)
    ax.tick_params(direction='in', top=True, right=True)

    # Build legend: stat colors + case linestyles
    from matplotlib.lines import Line2D
    handles = []
    for stat_key, _, _ in STATS:
        sty = _STAT_STYLES[stat_key]
        handles.append(Line2D([], [], color=sty['color'], ls='-',
                              linewidth=1.2, label=sty['label']))
    for ci, case in enumerate(cases):
        ls = _CASE_LINESTYLES[ci % len(_CASE_LINESTYLES)]
        handles.append(Line2D([], [], color='0.4', ls=ls,
                              linewidth=1.2, label=case['label']))
    ax.legend(handles=handles, fontsize=7, loc='lower center', ncol=2,
              framealpha=0.8)

    fig.tight_layout()
    _save(fig, save_path)


def fig_profiles_vs_longitude(cases, save_path, lat_band=30):
    """T_min, T_mean, T_max vs longitude for all cases (averaged |lat| < lat_band)."""
    fig, axes = plt.subplots(len(STATS), 1, figsize=(5.0, 6.0), sharex=True)

    for row, (stat_key, _, _) in enumerate(STATS):
        ax = axes[row]
        sty = _STAT_STYLES[stat_key]

        for ci, case in enumerate(cases):
            stat = case[stat_key]
            profile, lons = _meridional_profile(
                stat['lon_g'], stat['data_g'], stat['lat_g'], lat_band)
            ls = _CASE_LINESTYLES[ci % len(_CASE_LINESTYLES)]
            ax.plot(lons, profile, color=sty['color'], ls=ls,
                    linewidth=1.2, label=case['label'])

        ax.set_ylabel(f"{sty['label']} (K)")
        ax.tick_params(direction='in', top=True, right=True)
        ax.set_xlim(0, 360)
        ax.set_xticks(np.arange(0, 361, 45))

        # Mark hemispheres
        for lon, name in [(0, 'Sub-J'), (90, 'Trail'),
                          (180, 'Anti-J'), (270, 'Lead')]:
            ax.axvline(lon, color='0.8', linewidth=0.5, zorder=0)
            if row == 0:
                ax.text(lon, ax.get_ylim()[1], name, fontsize=6,
                        ha='center', va='bottom', color='0.5')

        if row == 0:
            ax.legend(fontsize=7, loc='best', framealpha=0.8)

    axes[-1].set_xlabel('Longitude (\u00b0E)')
    fig.suptitle(f'Equatorial temperatures ($|\\phi| < {lat_band}\\degree$)',
                 fontsize=10, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, save_path)


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
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Global Europa surface temperature maps')
    parser.add_argument('--cases', nargs='+', required=True,
                        help='Simulation output directory(s)')
    parser.add_argument('--labels', nargs='+', default=None,
                        help='Labels for each case')
    parser.add_argument('--save-dir', default='plots/global_temperature',
                        help='Output directory for figures')
    parser.add_argument('--cmap', default=None,
                        help='Matplotlib colormap name (default: truncated inferno)')
    args = parser.parse_args()

    # Override the global colormap if requested
    global CMAP
    if args.cmap is not None:
        CMAP = plt.get_cmap(args.cmap)
        print(f"Using colormap: {args.cmap}")

    os.makedirs(args.save_dir, exist_ok=True)
    labels = args.labels or [os.path.basename(d) for d in args.cases]

    # Load all cases
    print("Loading data...")
    cases = []
    for data_dir, label in zip(args.cases, labels):
        cases.append(_load_case(data_dir, label))

    # --- Per-case individual figures ---
    for case in cases:
        tag = case['label'].lower().replace(' ', '_').replace('+', '_')
        for stat_key, _, stat_label in STATS:
            norm = _stat_norm(cases, stat_key)

            print(f"\nGlobe: {case['label']} / {stat_label}")
            fig_single_globes(
                case, stat_key,
                save_path=os.path.join(args.save_dir,
                                       f'globe_{stat_key}_{tag}.png'),
                norm=norm,
            )

            print(f"Mollweide: {case['label']} / {stat_label}")
            fig_single_mollweide(
                case, stat_key,
                save_path=os.path.join(args.save_dir,
                                       f'mollweide_{stat_key}_{tag}.png'),
                norm=norm,
            )

    # --- Comparison globe figures (one per statistic) ---
    if len(cases) > 1:
        for stat_key, _, stat_label in STATS:
            print(f"\nComparison globes: {stat_label}")
            fig_comparison_globes(
                cases, stat_key,
                save_path=os.path.join(args.save_dir,
                                       f'globe_{stat_key}_comparison.png'),
            )

    # --- Combined Mollweide panel ---
    if len(cases) > 1:
        print("\nCombined Mollweide panel...")
        fig_combined_mollweide(
            cases,
            save_path=os.path.join(args.save_dir,
                                   'mollweide_combined.png'),
        )

    # --- Temperature profiles ---
    print("\nTemperature vs latitude (zonal mean)...")
    fig_profiles_vs_latitude(
        cases,
        save_path=os.path.join(args.save_dir, 'profile_vs_latitude.png'),
    )

    print("Temperature vs longitude (equatorial band)...")
    fig_profiles_vs_longitude(
        cases,
        save_path=os.path.join(args.save_dir, 'profile_vs_longitude.png'),
    )

    print(f"\nDone! Figures saved to {args.save_dir}")


if __name__ == '__main__':
    main()
