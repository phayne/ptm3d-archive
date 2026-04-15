#!/usr/bin/env python3
"""Net ice transport globe and Mollweide figures.

Produces publication-quality figures of net ice sublimation/deposition rate:
  - 3D globe views (2x2) with graticule and asymmetric diverging colorscale
  - Mollweide equal-area projections

Uses TwoSlopeNorm for asymmetric color scales so that deposition (positive)
and removal (negative) features are both visible even when ranges differ
by an order of magnitude.

Usage:
    python scripts/plot_ice_transport.py \
        --data-dir output/.../sublimation_no-albedo output/.../sputtering_no-albedo \
        --labels "Sublimation only" "Sublimation + sputtering" \
        --save-dir docs/figures/transport_averaged
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
    """Read binary PLY mesh, return face latitudes and longitudes in degrees."""
    with open(ply_path, 'rb') as f:
        n_vertices = 0
        n_faces = 0
        while True:
            line = f.readline().decode('ascii').strip()
            if line.startswith('element vertex'):
                n_vertices = int(line.split()[-1])
            elif line.startswith('element face'):
                n_faces = int(line.split()[-1])
            elif line == 'end_header':
                break
        f.seek(f.tell() + n_vertices * 24)
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


def discover_epochs(data_dir, field='nonice_concentration'):
    """Discover available epoch numbers from binary files."""
    field_dir = os.path.join(data_dir, field)
    if not os.path.isdir(field_dir):
        return []
    epochs = []
    for fname in os.listdir(field_dir):
        if fname.endswith('.bin'):
            parts = fname.replace('.bin', '').split('_')
            try:
                epochs.append(int(parts[-1]))
            except ValueError:
                continue
    return sorted(epochs)


def load_field_at_epoch(data_dir, field, epoch):
    """Load a per-face field at a given epoch number."""
    for width in [5, 4, 6]:
        fname = f"{field}_{epoch:0{width}d}.bin"
        fpath = os.path.join(data_dir, field, fname)
        if os.path.exists(fpath):
            return read_binary_field(fpath)
    raise FileNotFoundError(f"No file found for {field} epoch {epoch}")


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
# Color scale helpers
# ---------------------------------------------------------------------------

def _nice_ceil(x, base):
    """Round x up to nearest multiple of base."""
    return base * np.ceil(x / base)


def _nice_floor(x, base):
    """Round x down to nearest multiple of base."""
    return base * np.floor(x / base)


def _choose_tick_step(vrange):
    """Choose a nice tick step for a given value range."""
    raw = vrange / 5  # aim for ~5 ticks
    mag = 10 ** np.floor(np.log10(max(abs(raw), 1e-10)))
    candidates = [1, 2, 5, 10]
    for c in candidates:
        step = c * mag
        if vrange / step <= 8:
            return step
    return mag * 10


def make_asymmetric_norm(data_um_kyr):
    """Create a TwoSlopeNorm centered at 0 that fits the data range.

    Pads the range by 5% for visual breathing room.
    """
    vmin = np.nanpercentile(data_um_kyr, 0.5)
    vmax = np.nanpercentile(data_um_kyr, 99.5)

    # Ensure zero is visible
    if vmin >= 0:
        vmin = -0.05 * vmax if vmax > 0 else -1.0
    if vmax <= 0:
        vmax = -0.05 * vmin if vmin < 0 else 1.0

    # Pad by 5%
    span = vmax - vmin
    vmin -= 0.05 * span
    vmax += 0.05 * span

    return colors.TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax)


def make_common_norm(datasets):
    """Create a common TwoSlopeNorm that spans all datasets."""
    vmin = min(np.nanpercentile(d, 0.5) for d in datasets)
    vmax = max(np.nanpercentile(d, 99.5) for d in datasets)

    if vmin >= 0:
        vmin = -0.05 * vmax if vmax > 0 else -1.0
    if vmax <= 0:
        vmax = -0.05 * vmin if vmin < 0 else 1.0

    span = vmax - vmin
    vmin -= 0.05 * span
    vmax += 0.05 * span

    return colors.TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax)


def make_cbar_ticks(norm):
    """Generate explicit tick locations spanning both sides of a TwoSlopeNorm.

    Ensures ticks appear in both the negative and positive ranges,
    which matplotlib's default locator can miss when the ranges are
    very asymmetric.
    """
    vmin, vmax = norm.vmin, norm.vmax

    ticks = [0.0]

    # Negative side
    if vmin < 0:
        step_neg = _choose_tick_step(abs(vmin))
        t = -step_neg
        while t >= vmin:
            ticks.append(t)
            t -= step_neg

    # Positive side
    if vmax > 0:
        step_pos = _choose_tick_step(vmax)
        t = step_pos
        while t <= vmax:
            ticks.append(t)
            t += step_pos

    return sorted(ticks)


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
    """Plot data on a Mollweide projection axis.

    ax must have been created with projection='mollweide'.
    lon_grid is assumed 0..360; Mollweide needs -pi..pi.
    lon_center: east longitude at the center of the projection (default 0°E,
                sub-Jovian point).  Trailing hemisphere (90°E) appears to
                the right.
    """
    # Build coordinate array centered on lon_center
    lon_shifted = lon_grid - lon_center
    # Wrap to [-180, 180]
    lon_shifted = (lon_shifted + 180) % 360 - 180
    # Sort so coordinates are monotonically increasing for pcolormesh
    order = np.argsort(lon_shifted)
    lon_sorted = lon_shifted[order]
    data_sorted = data_grid[:, order]

    lat_rad = np.deg2rad(lat_grid)
    lon_rad = np.deg2rad(lon_sorted)
    LON, LAT = np.meshgrid(lon_rad, lat_rad)

    pcm = ax.pcolormesh(LON, LAT, data_sorted, cmap=cmap, norm=norm,
                        shading='auto', rasterized=True)

    # Graticule
    ax.grid(True, color='0.3', linewidth=0.3, alpha=0.4)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    return pcm


def add_mollweide_labels(ax, lon_center=0):
    """Add longitude/latitude tick labels around a Mollweide plot."""
    # Latitude labels on right edge (outside the ellipse)
    for lat_deg in [-60, -30, 0, 30, 60]:
        ax.text(np.deg2rad(185 - lon_center), np.deg2rad(lat_deg),
                f'{abs(lat_deg)}\u00b0{"N" if lat_deg > 0 else "S" if lat_deg < 0 else ""}',
                fontsize=5.5, color='0.4', ha='left', va='center',
                transform=ax.transData)

    # Longitude labels below the equator (east longitude)
    for lon_deg in [0, 90, 180, 270]:
        lon_plot = lon_deg - lon_center
        # Wrap to [-180, 180]
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

RHO_ICE = 917.0
SEC_PER_KYR = 1000 * 365.25 * 24 * 3600

CMAP = plt.get_cmap('RdBu')
CBAR_LABEL = r'Net ice flux ($\mu$m kyr$^{-1}$)'


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_net_flux(data_dir, epoch=None):
    """Load net flux and return in um/kyr. Returns (data, epoch_used)."""
    epochs = discover_epochs(data_dir, 'net_flux_kg_m2_s')
    if not epochs:
        return None, None
    if epoch is None:
        epoch = epochs[-1]
    net_flux = load_field_at_epoch(data_dir, 'net_flux_kg_m2_s', epoch)
    net_flux_um_kyr = net_flux / RHO_ICE * 1e6 * SEC_PER_KYR
    return net_flux_um_kyr, epoch


def _load_case(data_dir, epoch=None):
    """Load a full case: mesh + net flux.  Returns dict."""
    lats, lons, nf = read_ply_mesh(os.path.join(data_dir, 'mesh.ply'))
    net_flux_um_kyr, ep = _load_net_flux(data_dir, epoch)
    lat_g, lon_g, data_g = regrid_to_latlon(lats, lons, net_flux_um_kyr)
    frac_pos = (net_flux_um_kyr > 0).mean() * 100
    print(f"    range [{net_flux_um_kyr.min():.1f}, {net_flux_um_kyr.max():.1f}] "
          f"\u03bcm/kyr, {frac_pos:.0f}% positive, epoch {ep}")
    return dict(lats=lats, lons=lons, raw=net_flux_um_kyr, epoch=ep,
                lat_g=lat_g, lon_g=lon_g, data_g=data_g)


# ---------------------------------------------------------------------------
# Figure: single-case globes (1x4 row)
# ---------------------------------------------------------------------------

def fig_single_globes(case_data, save_path, label=None, norm=None):
    """1x4 row of globe views for a single case."""
    lat_g = case_data['lat_g']
    lon_g = case_data['lon_g']
    data_g = case_data['data_g']

    if norm is None:
        norm = make_asymmetric_norm(case_data['raw'])

    fig = plt.figure(figsize=(7.2, 2.4))
    gs = GridSpec(1, 4, figure=fig, wspace=0.01,
                  left=0.01, right=0.88, top=0.92, bottom=0.12)

    for idx, (view_name, elev, azim) in enumerate(VIEWS):
        ax = fig.add_subplot(gs[0, idx], projection='3d',
                             computed_zorder=False)
        plot_globe(ax, lat_g, lon_g, data_g, CMAP, norm, elev, azim,
                   title=view_name)

    # Colorbar
    ticks = make_cbar_ticks(norm)
    sm = cm.ScalarMappable(norm=norm, cmap=CMAP)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.70])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical', ticks=ticks)
    cbar.ax.set_yticklabels([f'{v:g}' for v in ticks])
    cbar.set_label(CBAR_LABEL, fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    if label:
        fig.suptitle(label, fontsize=11, y=0.99)

    _save(fig, save_path)


# ---------------------------------------------------------------------------
# Figure: comparison globes (2 columns x 4 rows)
# ---------------------------------------------------------------------------

def fig_comparison_globes(cases, labels, save_path, norm=None):
    """Side-by-side globe comparison, 4 views x N cases."""
    n = len(cases)

    if norm is None:
        norm = make_common_norm([c['raw'] for c in cases])

    fig = plt.figure(figsize=(3.2 * n + 0.8, 10.5))
    gs = GridSpec(4, n, figure=fig, hspace=0.08, wspace=-0.05,
                  left=0.01, right=0.84, top=0.95, bottom=0.02)

    row_y_positions = []  # store row bottom positions for labels
    for col, (case_data, label) in enumerate(zip(cases, labels)):
        for row, (view_name, elev, azim) in enumerate(VIEWS):
            ax = fig.add_subplot(gs[row, col], projection='3d',
                                 computed_zorder=False)
            plot_globe(ax, case_data['lat_g'], case_data['lon_g'],
                       case_data['data_g'], CMAP, norm, elev, azim)
            # Record row positions from first column
            if col == 0:
                bbox = gs[row, col].get_position(fig)
                row_y_positions.append(bbox.y0)

        # Column header
        x_center = 0.01 + (col + 0.5) * 0.83 / n
        fig.text(x_center, 0.97, label,
                 fontsize=10, ha='center', va='center')

    # View labels centered between the two columns, midway down each row
    x_label = 0.01 + 0.83 / 2.0  # center across both columns
    for row, (view_name, _, _) in enumerate(VIEWS):
        # Midpoint between this row's bottom and the next row's top (or figure bottom)
        y_top = gs[row, 0].get_position(fig).y1
        y_bot = gs[row, 0].get_position(fig).y0
        y_mid = (y_top + y_bot) / 2.0
        fig.text(x_label, y_mid, view_name,
                 fontsize=8, ha='center', va='center', color='0.3')

    # Colorbar
    ticks = make_cbar_ticks(norm)
    sm = cm.ScalarMappable(norm=norm, cmap=CMAP)
    cbar_ax = fig.add_axes([0.87, 0.10, 0.018, 0.78])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical', ticks=ticks)
    cbar.ax.set_yticklabels([f'{v:g}' for v in ticks])
    cbar.set_label(CBAR_LABEL, fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    _save(fig, save_path)


# ---------------------------------------------------------------------------
# Figure: single-case Mollweide
# ---------------------------------------------------------------------------

def fig_single_mollweide(case_data, save_path, label=None, norm=None):
    """Single Mollweide projection for one case."""
    if norm is None:
        norm = make_asymmetric_norm(case_data['raw'])

    fig = plt.figure(figsize=(5.5, 3.0))
    ax = fig.add_subplot(111, projection='mollweide')

    pcm = plot_mollweide(ax, case_data['lat_g'], case_data['lon_g'],
                         case_data['data_g'], CMAP, norm)
    add_mollweide_labels(ax)

    ticks = make_cbar_ticks(norm)
    cbar = fig.colorbar(pcm, ax=ax, orientation='horizontal',
                        fraction=0.06, pad=0.04, aspect=35, ticks=ticks)
    cbar.ax.set_xticklabels([f'{v:g}' for v in ticks])
    cbar.set_label(CBAR_LABEL, fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    if label:
        ax.set_title(label, fontsize=10, pad=10)

    _save(fig, save_path)


# ---------------------------------------------------------------------------
# Figure: comparison Mollweide (stacked or side-by-side)
# ---------------------------------------------------------------------------

def fig_comparison_mollweide(cases, labels, save_path, norm=None):
    """Stacked Mollweide projections with shared colorbar."""
    n = len(cases)

    if norm is None:
        norm = make_common_norm([c['raw'] for c in cases])

    fig = plt.figure(figsize=(6.0, 2.8 * n + 0.6))
    gs = GridSpec(n, 1, figure=fig, hspace=0.25,
                  left=0.05, right=0.95, top=0.94, bottom=0.10)

    panel_labels = 'abcdefgh'
    pcm = None
    for i, (case_data, label) in enumerate(zip(cases, labels)):
        ax = fig.add_subplot(gs[i, 0], projection='mollweide')
        pcm = plot_mollweide(ax, case_data['lat_g'], case_data['lon_g'],
                             case_data['data_g'], CMAP, norm)
        add_mollweide_labels(ax)
        ax.set_title(f'({panel_labels[i]})  {label}', fontsize=10,
                     loc='left', pad=8)

    # Shared colorbar
    ticks = make_cbar_ticks(norm)
    cbar_ax = fig.add_axes([0.20, 0.04, 0.60, 0.020])
    cbar = fig.colorbar(pcm, cax=cbar_ax, orientation='horizontal', ticks=ticks)
    cbar.ax.set_xticklabels([f'{v:g}' for v in ticks])
    cbar.set_label(CBAR_LABEL, fontsize=9)
    cbar.ax.tick_params(labelsize=8)

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
        description='Net ice transport globe and Mollweide figures')
    parser.add_argument('--data-dir', nargs='+', required=True,
                        help='Exosphere output directory(s)')
    parser.add_argument('--labels', nargs='+', default=None,
                        help='Labels for each case')
    parser.add_argument('--save-dir', default='docs/figures/transport_averaged',
                        help='Output directory for figures')
    parser.add_argument('--epoch', type=int, default=None,
                        help='Epoch to plot (default: last available)')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    labels = args.labels or [os.path.basename(d) for d in args.data_dir]

    # Load all cases
    print("Loading data...")
    cases = []
    for data_dir, label in zip(args.data_dir, labels):
        print(f"  {label}:")
        cases.append(_load_case(data_dir, epoch=args.epoch))

    # Per-case norms (for individual figures)
    norms = [make_asymmetric_norm(c['raw']) for c in cases]
    # Common norm (for comparison figures)
    common_norm = make_common_norm([c['raw'] for c in cases])

    # --- Individual globe figures ---
    for case_data, label, norm in zip(cases, labels, norms):
        tag = label.lower().replace(' ', '_').replace('+', '_')
        print(f"\nGlobe: {label}")
        fig_single_globes(
            case_data,
            save_path=os.path.join(args.save_dir,
                                   f'globe_{tag}.png'),
            label=label,
            norm=norm,
        )

    # --- Comparison globe figure ---
    if len(cases) > 1:
        print("\nComparison globes...")
        fig_comparison_globes(
            cases, labels,
            save_path=os.path.join(args.save_dir,
                                   'globe_comparison.png'),
            norm=common_norm,
        )

    # --- Individual Mollweide figures ---
    for case_data, label, norm in zip(cases, labels, norms):
        tag = label.lower().replace(' ', '_').replace('+', '_')
        print(f"\nMollweide: {label}")
        fig_single_mollweide(
            case_data,
            save_path=os.path.join(args.save_dir,
                                   f'mollweide_{tag}.png'),
            label=label,
            norm=norm,
        )

    # --- Comparison Mollweide figure ---
    if len(cases) > 1:
        print("\nComparison Mollweide...")
        fig_comparison_mollweide(
            cases, labels,
            save_path=os.path.join(args.save_dir,
                                   'mollweide_comparison.png'),
            norm=common_norm,
        )

    print(f"\nDone! Figures saved to {args.save_dir}")


if __name__ == '__main__':
    main()
