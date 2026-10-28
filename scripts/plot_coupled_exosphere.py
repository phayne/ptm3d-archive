#!/usr/bin/env python3
"""Publication-quality globe plots for coupled exosphere simulation.

Reads binary exosphere output (PLY mesh + per-face binary fields) and produces
3D globe views using plot_surface, in the project globe style.

Usage:
    python scripts/plot_coupled_exosphere.py \
        --data-dir output/europa_ico6_uniform-ice/exosphere/coupled_5pct_v2 \
        --save-dir docs/figures \
        [--dpi 300]
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
})

PANEL_LABELS = 'abcdefghijklmnop'


# ---------------------------------------------------------------------------
# Data I/O
# ---------------------------------------------------------------------------

def read_ply_mesh(ply_path):
    """Read binary PLY mesh, return face latitudes and longitudes in degrees."""
    with open(ply_path, 'rb') as f:
        # Parse header
        n_vertices = 0
        n_faces = 0
        header_lines = []
        while True:
            line = f.readline().decode('ascii').strip()
            header_lines.append(line)
            if line.startswith('element vertex'):
                n_vertices = int(line.split()[-1])
            elif line.startswith('element face'):
                n_faces = int(line.split()[-1])
            elif line == 'end_header':
                break

        # Skip vertex data (double x, y, z = 24 bytes each)
        f.seek(f.tell() + n_vertices * 24)

        # Read face data: uchar(1) + 3*int32(12) + float lat(4) + float lon(4) = 21 bytes
        lats = np.zeros(n_faces)
        lons = np.zeros(n_faces)
        for i in range(n_faces):
            data = f.read(21)
            count = struct.unpack('<B', data[0:1])[0]
            # Skip vertex indices (3 * int32)
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


def read_time_steps(data_dir):
    """Read time_steps.txt, return times in years."""
    ts_file = os.path.join(data_dir, 'time_steps.txt')
    times = []
    with open(ts_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            times.append(float(line))
    return np.array(times) / (365.25 * 24 * 3600)  # seconds -> years


def discover_epochs(data_dir, field='nonice_concentration'):
    """Discover available epoch numbers from binary files."""
    field_dir = os.path.join(data_dir, field)
    if not os.path.isdir(field_dir):
        return []
    epochs = []
    for fname in os.listdir(field_dir):
        if fname.endswith('.bin'):
            # e.g., nonice_concentration_00500.bin
            parts = fname.replace('.bin', '').split('_')
            try:
                epochs.append(int(parts[-1]))
            except ValueError:
                continue
    return sorted(epochs)


def build_epoch_to_time(data_dir, epochs):
    """Build a mapping from epoch number to time in years.

    time_steps.txt has one entry per exported epoch, in order.
    epochs is the sorted list of exported epoch numbers.
    Returns dict: epoch_number -> time_yr.
    """
    time_steps = read_time_steps(data_dir)
    epoch_to_time = {}
    for i, ep in enumerate(epochs):
        if i < len(time_steps):
            epoch_to_time[ep] = time_steps[i]
        else:
            # Extrapolate from last known
            dt = time_steps[-1] / max(1, len(time_steps) - 1) if len(time_steps) > 1 else 1.0
            epoch_to_time[ep] = time_steps[-1] + (i - len(time_steps) + 1) * dt
    return epoch_to_time


def load_field_at_epoch(data_dir, field, epoch):
    """Load a per-face field at a given epoch number."""
    # Try different zero-padding widths
    for width in [5, 4, 6]:
        fname = f"{field}_{epoch:0{width}d}.bin"
        fpath = os.path.join(data_dir, field, fname)
        if os.path.exists(fpath):
            return read_binary_field(fpath)
    raise FileNotFoundError(f"No file found for {field} epoch {epoch}")


# ---------------------------------------------------------------------------
# Regridding: unstructured mesh -> regular lat/lon grid
# ---------------------------------------------------------------------------

def regrid_to_latlon(lats, lons, values, n_lat=91, n_lon=181):
    """Regrid unstructured face data to a regular lat/lon grid.

    Uses linear interpolation via scipy.interpolate.griddata.
    Returns lat_grid (1D), lon_grid (1D), data_grid (2D).
    """
    # Create regular grid
    lat_grid = np.linspace(-90, 90, n_lat)
    lon_grid = np.linspace(0, 360, n_lon)
    LON_G, LAT_G = np.meshgrid(lon_grid, lat_grid)

    # Normalize longitudes to [0, 360]
    lons_norm = lons % 360

    # Add wrap-around points for longitude continuity
    # Duplicate points near 0 at 360 and vice versa
    margin = 10.0  # degrees
    mask_low = lons_norm < margin
    mask_high = lons_norm > (360 - margin)

    lats_ext = np.concatenate([lats, lats[mask_low], lats[mask_high]])
    lons_ext = np.concatenate([lons_norm, lons_norm[mask_low] + 360,
                               lons_norm[mask_high] - 360])
    vals_ext = np.concatenate([values, values[mask_low], values[mask_high]])

    # Interpolate
    data_grid = griddata(
        (lons_ext, lats_ext), vals_ext,
        (LON_G, LAT_G), method='linear'
    )

    # Fill any remaining NaN with nearest-neighbor
    mask_nan = np.isnan(data_grid)
    if np.any(mask_nan):
        data_nn = griddata(
            (lons_ext, lats_ext), vals_ext,
            (LON_G, LAT_G), method='nearest'
        )
        data_grid[mask_nan] = data_nn[mask_nan]

    return lat_grid, lon_grid, data_grid


# ---------------------------------------------------------------------------
# 3D globe rendering
# ---------------------------------------------------------------------------

def make_globe_xyz(lat_grid, lon_grid, data_grid=None):
    """Create X, Y, Z coordinates for a globe surface.

    lat_grid, lon_grid in degrees. Returns X, Y, Z arrays.
    """
    lat_rad = np.deg2rad(lat_grid)
    lon_rad = np.deg2rad(lon_grid)
    LON, LAT = np.meshgrid(lon_rad, lat_rad)

    R = 1.0
    X = R * np.cos(LAT) * np.cos(LON)
    Y = R * np.cos(LAT) * np.sin(LON)
    Z = R * np.sin(LAT)
    return X, Y, Z


def plot_globe_row(axes, lat_grid, lon_grid, data_grid, cmap, norm,
                   views=None, title=None):
    """Plot a row of globe views on given axes."""
    if views is None:
        views = [
            ('Sub-Jovian (0\u00b0)',    0,   0),
            ('Trailing (90\u00b0E)',    0,  90),
            ('Anti-Jovian (180\u00b0)', 0, 180),
            ('Leading (270\u00b0E)',    0, 270),
        ]

    X, Y, Z = make_globe_xyz(lat_grid, lon_grid)
    facecolors = plt.get_cmap(cmap)(norm(data_grid))

    for idx, ax in enumerate(axes):
        view_name, elev, azim = views[idx]
        ax.plot_surface(X, Y, Z, facecolors=facecolors,
                        rstride=1, cstride=1, shade=False,
                        antialiased=False, rasterized=True)
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_zlim(-1.1, 1.1)
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        ax.set_box_aspect([1, 1, 1])
        ax.set_title(view_name, fontsize=9, pad=-6)


# ---------------------------------------------------------------------------
# Figure: Multi-epoch globe evolution
# ---------------------------------------------------------------------------

def fig_globe_evolution(lats, lons, data_dir, field, epoch_indices,
                        cmap, label, unit, vmin=None, vmax=None,
                        scale=1.0, suptitle=None, save_path=None,
                        diverging=False, epoch_to_time=None):
    """4-epoch x 4-view globe evolution figure.

    epoch_indices: list of 4 epoch numbers to show
    epoch_to_time: dict mapping epoch number -> time in years
    """
    # Load data for all epochs
    all_data = []
    epoch_times = []

    for ep in epoch_indices:
        raw = load_field_at_epoch(data_dir, field, ep) * scale
        all_data.append(raw)
        if epoch_to_time and ep in epoch_to_time:
            epoch_times.append(epoch_to_time[ep])
        else:
            epoch_times.append(0)

    # Compute global vmin/vmax from all epochs
    if vmin is None:
        vmin = min(np.nanpercentile(d, 1) for d in all_data)
    if vmax is None:
        vmax = max(np.nanpercentile(d, 99) for d in all_data)

    if diverging:
        vabs = max(abs(vmin), abs(vmax))
        vmin, vmax = -vabs, vabs
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = colors.Normalize(vmin=vmin, vmax=vmax)

    # Regrid all epochs
    grids = []
    for raw in all_data:
        lat_g, lon_g, data_g = regrid_to_latlon(lats, lons, raw)
        grids.append(data_g)

    views = [
        ('Sub-Jovian (0\u00b0)',    0,   0),
        ('Trailing (90\u00b0E)',    0,  90),
        ('Anti-Jovian (180\u00b0)', 0, 180),
        ('Leading (270\u00b0E)',    0, 270),
    ]

    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(4, 4, figure=fig, hspace=0.02, wspace=0.02,
                  left=0.02, right=0.88, top=0.93, bottom=0.06)

    for row_idx, (ep, t_yr, data_g) in enumerate(zip(epoch_indices,
                                                       epoch_times, grids)):
        axes = []
        for col_idx in range(4):
            ax = fig.add_subplot(gs[row_idx, col_idx], projection='3d',
                                 computed_zorder=False)
            axes.append(ax)

        plot_globe_row(axes, lat_g, lon_g, data_g, cmap, norm, views)

        # Row label (time)
        if t_yr < 1:
            time_label = f"t = 0 yr"
        elif t_yr < 1000:
            time_label = f"t = {t_yr:.0f} yr"
        else:
            time_label = f"t = {t_yr/1000:.1f} kyr"

        fig.text(0.90, 0.93 - (row_idx + 0.5) * (0.87 / 4),
                 time_label, fontsize=11, fontweight='bold',
                 ha='left', va='center')

    # Colorbar
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar_ax = fig.add_axes([0.15, 0.03, 0.55, 0.015])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(f'{label} [{unit}]')

    if suptitle:
        fig.suptitle(suptitle, fontsize=14, fontweight='bold', y=0.97)

    if save_path:
        for ext in ['.png', '.pdf']:
            fig.savefig(save_path.replace('.png', ext).replace('.pdf', ext)
                        if not save_path.endswith(ext)
                        else save_path.rsplit('.', 1)[0] + ext)
        print(f'  Saved {save_path}')
    return fig


# ---------------------------------------------------------------------------
# Figure: Single-epoch globe (4 views)
# ---------------------------------------------------------------------------

def fig_globe_single(lats, lons, data, cmap, label, unit,
                     vmin=None, vmax=None, suptitle=None,
                     save_path=None, diverging=False):
    """Single-epoch, 4-view globe figure (2x2 layout like analytical plots)."""
    if vmin is None:
        vmin = np.nanpercentile(data, 1)
    if vmax is None:
        vmax = np.nanpercentile(data, 99)

    if diverging:
        vabs = max(abs(vmin), abs(vmax))
        vmin, vmax = -vabs, vabs

    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    lat_g, lon_g, data_g = regrid_to_latlon(lats, lons, data)

    views = [
        ('Sub-Jovian (0\u00b0)',    0,   0),
        ('Trailing (90\u00b0E)',    0,  90),
        ('Anti-Jovian (180\u00b0)', 0, 180),
        ('Leading (270\u00b0E)',    0, 270),
    ]

    fig = plt.figure(figsize=(7.2, 7.2))
    gs = GridSpec(2, 2, figure=fig, hspace=0.02, wspace=0.02)

    X, Y, Z = make_globe_xyz(lat_g, lon_g)
    facecolors = plt.get_cmap(cmap)(norm(data_g))

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

    # Colorbar
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.018])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(f'{label} [{unit}]')

    if suptitle:
        fig.suptitle(suptitle, fontsize=13, fontweight='bold', y=0.97)

    if save_path:
        for ext in ['.png', '.pdf']:
            base = save_path.rsplit('.', 1)[0]
            fig.savefig(base + ext)
        print(f'  Saved {save_path}')
    return fig


# ---------------------------------------------------------------------------
# Figure: Time series (2x2 panel)
# ---------------------------------------------------------------------------

def has_sputtering_data(data_dir, epochs):
    """Check if sputtering flux data is available."""
    try:
        load_field_at_epoch(data_dir, 'sputtering_flux_kg_m2_s', epochs[0])
        return True
    except FileNotFoundError:
        return False


def fig_timeseries(data_dir, lats, lons, epochs, save_path=None,
                   epoch_to_time=None):
    """Time series: vapor pressure, albedo, concentration, temperature,
    and optionally sputtering vs sublimation comparison."""
    # Check if sputtering data exists
    sputtering = has_sputtering_data(data_dir, epochs)

    # Collect time series data
    times_kyr = []
    mean_vapor = []
    mean_albedo = []
    mean_conc = []
    max_conc = []
    mean_temp = []
    mean_sublim = []
    mean_sput = []

    for ep in epochs:
        t_yr = epoch_to_time[ep] if (epoch_to_time and ep in epoch_to_time) else 0
        times_kyr.append(t_yr / 1000)

        # Load fields
        try:
            conc = load_field_at_epoch(data_dir, 'nonice_concentration', ep)
            mean_conc.append(np.mean(conc) * 100)
            max_conc.append(np.max(conc) * 100)
        except FileNotFoundError:
            mean_conc.append(np.nan)
            max_conc.append(np.nan)

        try:
            albedo = load_field_at_epoch(data_dir, 'bond_albedo', ep)
            mean_albedo.append(np.mean(albedo))
        except FileNotFoundError:
            mean_albedo.append(np.nan)

        try:
            temp = load_field_at_epoch(data_dir, 'temperature_mean_K', ep)
            mean_temp.append(np.mean(temp))
        except FileNotFoundError:
            mean_temp.append(np.nan)

        try:
            vp = load_field_at_epoch(data_dir, 'vapor_pressure_Pa', ep)
            mean_vapor.append(np.mean(vp))
        except FileNotFoundError:
            mean_vapor.append(np.nan)

        if sputtering:
            try:
                sublim = load_field_at_epoch(data_dir, 'emission_flux_kg_m2_s', ep)
                mean_sublim.append(np.mean(sublim))
            except FileNotFoundError:
                mean_sublim.append(np.nan)
            try:
                sput = load_field_at_epoch(data_dir, 'sputtering_flux_kg_m2_s', ep)
                mean_sput.append(np.mean(sput))
            except FileNotFoundError:
                mean_sput.append(np.nan)

    if sputtering:
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # (a) Vapor pressure
    ax = axes[0, 0]
    ax.semilogy(times_kyr, mean_vapor, 'b-', linewidth=1.5)
    ax.set_ylabel('Mean surface vapor pressure (Pa)')
    ax.set_title('(a) Global mean vapor pressure')
    ax.grid(True, alpha=0.3)

    # (b) Albedo
    ax = axes[0, 1]
    ax.plot(times_kyr, mean_albedo, 'g-', linewidth=1.5)
    ax.set_ylabel('Mean Bond albedo')
    ax.set_title('(b) Global mean albedo')
    ax.grid(True, alpha=0.3)

    # (c) Concentration
    ax = axes[1, 0]
    ax.plot(times_kyr, mean_conc, 'r-', linewidth=1.5, label='Mean')
    ax.plot(times_kyr, max_conc, 'r--', linewidth=1, label='Max')
    ax.set_xlabel('Time (kyr)')
    ax.set_ylabel('Non-ice concentration (%)')
    ax.set_title('(c) Non-ice concentration')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (d) Temperature
    ax = axes[1, 1]
    ax.plot(times_kyr, mean_temp, 'k-', linewidth=1.5)
    ax.set_xlabel('Time (kyr)')
    ax.set_ylabel('Mean surface temperature (K)')
    ax.set_title('(d) Global mean temperature')
    ax.grid(True, alpha=0.3)

    if sputtering:
        # (e) Sublimation vs sputtering flux comparison
        ax = axes[0, 2]
        ax.semilogy(times_kyr, mean_sublim, 'C0-', linewidth=1.5, label='Sublimation')
        ax.semilogy(times_kyr, mean_sput, 'C3-', linewidth=1.5, label='Sputtering')
        ax.set_ylabel('Mean flux (kg m$^{-2}$ s$^{-1}$)')
        ax.set_title('(e) Sublimation vs sputtering')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # (f) Sputtering / sublimation ratio
        ax = axes[1, 2]
        ratio = np.array(mean_sput) / np.array(mean_sublim)
        ax.plot(times_kyr, ratio, 'C5-', linewidth=1.5)
        ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.set_xlabel('Time (kyr)')
        ax.set_ylabel('Sputtering / sublimation ratio')
        ax.set_title('(f) Flux ratio')
        ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path:
        for ext in ['.png', '.pdf']:
            base = save_path.rsplit('.', 1)[0]
            fig.savefig(base + ext)
        print(f'  Saved {save_path}')
    return fig


# ---------------------------------------------------------------------------
# Figure: Latitude profiles
# ---------------------------------------------------------------------------

def fig_latitude_profiles(data_dir, lats, lons, epoch_indices, save_path=None,
                          epoch_to_time=None):
    """3-panel latitude profiles at multiple epochs."""
    # Latitude bins
    lat_bins = np.linspace(-90, 90, 37)  # 5-degree bins
    lat_centers = 0.5 * (lat_bins[:-1] + lat_bins[1:])

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    cmap_epochs = plt.get_cmap('viridis')
    epoch_colors = [cmap_epochs(i / max(1, len(epoch_indices) - 1))
                    for i in range(len(epoch_indices))]

    for ep_idx, ep in enumerate(epoch_indices):
        t_yr = epoch_to_time[ep] if (epoch_to_time and ep in epoch_to_time) else 0
        if t_yr < 1000:
            label = f"{t_yr:.0f} yr"
        else:
            label = f"{t_yr/1000:.1f} kyr"

        color = epoch_colors[ep_idx]

        # Load fields
        try:
            conc = load_field_at_epoch(data_dir, 'nonice_concentration', ep) * 100
        except FileNotFoundError:
            conc = np.full(len(lats), np.nan)
        try:
            albedo = load_field_at_epoch(data_dir, 'bond_albedo', ep)
        except FileNotFoundError:
            albedo = np.full(len(lats), np.nan)
        try:
            temp = load_field_at_epoch(data_dir, 'temperature_mean_K', ep)
        except FileNotFoundError:
            temp = np.full(len(lats), np.nan)

        # Bin by latitude
        conc_binned = []
        albedo_binned = []
        temp_binned = []
        for j in range(len(lat_bins) - 1):
            mask = (lats >= lat_bins[j]) & (lats < lat_bins[j+1])
            conc_binned.append(np.mean(conc[mask]) if np.any(mask) else np.nan)
            albedo_binned.append(np.mean(albedo[mask]) if np.any(mask) else np.nan)
            temp_binned.append(np.mean(temp[mask]) if np.any(mask) else np.nan)

        axes[0].plot(lat_centers, conc_binned, color=color, label=label, linewidth=1.5)
        axes[1].plot(lat_centers, albedo_binned, color=color, label=label, linewidth=1.5)
        axes[2].plot(lat_centers, temp_binned, color=color, label=label, linewidth=1.5)

    axes[0].set_xlabel('Latitude (\u00b0)')
    axes[0].set_ylabel('Non-ice concentration (%)')
    axes[0].set_title('(a) Non-ice concentration')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Latitude (\u00b0)')
    axes[1].set_ylabel('Bond albedo')
    axes[1].set_title('(b) Bond albedo')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].set_xlabel('Latitude (\u00b0)')
    axes[2].set_ylabel('Mean temperature (K)')
    axes[2].set_title('(c) Mean temperature')
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path:
        for ext in ['.png', '.pdf']:
            base = save_path.rsplit('.', 1)[0]
            fig.savefig(base + ext)
        print(f'  Saved {save_path}')
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Plot coupled exosphere results')
    parser.add_argument('--data-dir', required=True,
                        help='Exosphere output directory')
    parser.add_argument('--save-dir', default='docs/figures',
                        help='Output figure directory')
    parser.add_argument('--dpi', type=int, default=300)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    plt.rcParams['savefig.dpi'] = args.dpi

    # Load mesh
    mesh_path = os.path.join(args.data_dir, 'mesh.ply')
    print(f"Loading mesh: {mesh_path}")
    lats, lons, n_faces = read_ply_mesh(mesh_path)
    print(f"  {n_faces} faces, lat [{lats.min():.1f}, {lats.max():.1f}], "
          f"lon [{lons.min():.1f}, {lons.max():.1f}]")

    # Discover epochs
    epochs = discover_epochs(args.data_dir)
    print(f"  {len(epochs)} exported epochs: {epochs[0]}..{epochs[-1]}")

    # Pick 4 representative epochs for evolution plots
    n_ep = len(epochs)
    if n_ep >= 4:
        idx_picks = [0, n_ep // 3, 2 * n_ep // 3, n_ep - 1]
        epoch_picks = [epochs[i] for i in idx_picks]
    else:
        epoch_picks = epochs

    print(f"  Selected epochs for evolution: {epoch_picks}")

    # Build epoch-to-time mapping
    epoch_to_time = build_epoch_to_time(args.data_dir, epochs)
    for ep in epoch_picks:
        t = epoch_to_time.get(ep, 0)
        print(f"    Epoch {ep} -> {t:.1f} yr ({t/1000:.2f} kyr)")

    # ---- Figure 1: Non-ice concentration evolution ----
    print("\nFigure 1: Non-ice concentration evolution...")
    fig_globe_evolution(
        lats, lons, args.data_dir,
        field='nonice_concentration',
        epoch_indices=epoch_picks,
        cmap='inferno',
        label='Non-ice concentration',
        unit='%',
        scale=100.0,
        vmin=0,
        suptitle='Non-Ice Concentration Evolution (Coupled Exosphere)',
        save_path=os.path.join(args.save_dir, 'globe_concentration_evolution.png'),
        epoch_to_time=epoch_to_time,
    )
    plt.close()

    # ---- Figure 2: Albedo evolution ----
    print("Figure 2: Albedo evolution...")
    try:
        load_field_at_epoch(args.data_dir, 'bond_albedo', epochs[0])
        fig_globe_evolution(
            lats, lons, args.data_dir,
            field='bond_albedo',
            epoch_indices=epoch_picks,
            cmap='cividis',
            label='Bond albedo',
            unit='',
            suptitle='Bond Albedo Evolution (Coupled Exosphere)',
            save_path=os.path.join(args.save_dir, 'globe_albedo_evolution.png'),
            epoch_to_time=epoch_to_time,
        )
        plt.close()
    except FileNotFoundError:
        print("  Skipping (bond_albedo not found — albedo coupling disabled)")

    # ---- Figure 3: Net ice flux (final epoch) ----
    print("Figure 3: Net ice flux at final epoch...")
    try:
        net_flux = load_field_at_epoch(args.data_dir, 'net_flux_kg_m2_s',
                                        epochs[-1])
        # Convert to um/kyr for readability
        rho_ice = 917.0
        sec_per_kyr = 1000 * 365.25 * 24 * 3600
        net_flux_um_kyr = net_flux / rho_ice * 1e6 * sec_per_kyr

        fig_globe_single(
            lats, lons, net_flux_um_kyr,
            cmap='RdBu',
            label='Net ice flux',
            unit='\u03bcm kyr\u207b\u00b9',
            suptitle=f'Net Ice Flux (epoch {epochs[-1]})',
            save_path=os.path.join(args.save_dir, 'globe_net_flux_coupled.png'),
            diverging=True,
        )
        plt.close()
    except FileNotFoundError:
        print("  Skipping (net_flux not found)")

    # ---- Figure 4: Temperature at final epoch ----
    print("Figure 4: Temperature at final epoch...")
    try:
        temp = load_field_at_epoch(args.data_dir, 'temperature_mean_K',
                                    epochs[-1])
        fig_globe_single(
            lats, lons, temp,
            cmap='inferno',
            label='Orbit-averaged temperature',
            unit='K',
            suptitle=f'Surface Temperature (epoch {epochs[-1]})',
            save_path=os.path.join(args.save_dir, 'globe_temperature_coupled.png'),
        )
        plt.close()
    except FileNotFoundError:
        print("  Skipping (temperature not found)")

    # ---- Figure 5: Time series ----
    print("Figure 5: Time series...")
    fig_timeseries(
        args.data_dir, lats, lons, epochs,
        save_path=os.path.join(args.save_dir, 'coupled_exo_timeseries.png'),
        epoch_to_time=epoch_to_time,
    )
    plt.close()

    # ---- Figure 6: Latitude profiles ----
    print("Figure 6: Latitude profiles...")
    fig_latitude_profiles(
        args.data_dir, lats, lons, epoch_picks,
        save_path=os.path.join(args.save_dir, 'coupled_exo_latitude_profiles.png'),
        epoch_to_time=epoch_to_time,
    )
    plt.close()

    # ---- Sputtering-specific figures (only if data exists) ----
    if has_sputtering_data(args.data_dir, epochs):
        rho_ice = 917.0
        sec_per_kyr = 1000 * 365.25 * 24 * 3600

        # ---- Figure 7: Sputtering flux globe (final epoch) ----
        print("Figure 7: Sputtering flux at final epoch...")
        try:
            sput_flux = load_field_at_epoch(args.data_dir,
                                             'sputtering_flux_kg_m2_s', epochs[-1])
            sput_um_kyr = sput_flux / rho_ice * 1e6 * sec_per_kyr

            fig_globe_single(
                lats, lons, sput_um_kyr,
                cmap='plasma',
                label='Sputtering flux',
                unit='\u03bcm kyr\u207b\u00b9',
                suptitle=f'Sputtering Flux (epoch {epochs[-1]})',
                save_path=os.path.join(args.save_dir, 'globe_sputtering_flux.png'),
            )
            plt.close()
        except FileNotFoundError:
            print("  Skipping (sputtering_flux not found)")

        # ---- Figure 8: Sputtering spatial weight (epoch 0) ----
        print("Figure 8: Ion flux distribution (spatial weight)...")
        try:
            weight = load_field_at_epoch(args.data_dir,
                                          'sputtering_spatial_weight', epochs[0])
            fig_globe_single(
                lats, lons, weight,
                cmap='YlOrRd',
                label='Ion flux weight',
                unit='',
                suptitle='Ion Flux Distribution (spatial weight)',
                save_path=os.path.join(args.save_dir, 'globe_sputtering_weight.png'),
            )
            plt.close()
        except FileNotFoundError:
            print("  Skipping (sputtering_spatial_weight not found)")

        # ---- Figure 9: Sputtering flux evolution (4 epochs x 4 views) ----
        print("Figure 9: Sputtering flux evolution...")
        try:
            fig_globe_evolution(
                lats, lons, args.data_dir,
                field='sputtering_flux_kg_m2_s',
                epoch_indices=epoch_picks,
                cmap='plasma',
                label='Sputtering flux',
                unit='\u03bcm kyr\u207b\u00b9',
                scale=sec_per_kyr / rho_ice * 1e6,
                vmin=0,
                suptitle='Sputtering Flux Evolution',
                save_path=os.path.join(args.save_dir,
                                        'globe_sputtering_evolution.png'),
                epoch_to_time=epoch_to_time,
            )
            plt.close()
        except FileNotFoundError:
            print("  Skipping (sputtering_flux not found at all epochs)")

    print("\nDone! All figures saved to:", args.save_dir)


if __name__ == '__main__':
    main()
