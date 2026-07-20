#!/usr/bin/env python3
"""Combined equatorial and latitudinal transport profile figures.

Produces a 3×2 panel figure showing:
  (a) Equatorial band temperature vs longitude
  (b) Latitudinal profile (zonal mean) temperature vs latitude
  (c) Equatorial vapor pressure ratio P_v/P̄ vs longitude
  (d) Latitudinal P_v/P̄ vs latitude
  (e) Equatorial net transfer rate vs longitude
  (f) Latitudinal net ice rate vs latitude

Optionally overlays multiple cases (sublimation-only, sputtering, etc.)
for direct comparison.

Usage:
    python scripts/plot_transport_profiles.py \
      --data-dir output/.../coupled_5pct_v2 \
      --save-dir docs/figures/transport

    # Multi-case overlay
    python scripts/plot_transport_profiles.py \
      --data-dir output/.../coupled_5pct_v2 output/.../sputtering_no-albedo \
      --labels "Sublimation only" "Sublimation + sputtering" \
      --save-dir docs/figures/transport
"""

import os
import sys
import struct
import argparse
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import ticker

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
    'legend.fontsize': 8,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
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
# Binning helpers
# ---------------------------------------------------------------------------

def bin_equatorial(lons, values, lat_band=5.0, lats=None, n_bins=72):
    """Bin equatorial-band values by longitude.

    Returns bin_centers (deg), bin_means, bin_mins, bin_maxs.
    lat_band: half-width in degrees (|lat| < lat_band).
    """
    mask = np.abs(lats) < lat_band
    lon_eq = lons[mask] % 360
    val_eq = values[mask]

    # Shift to [-180, 180] centered on sub-Jovian
    lon_eq = np.where(lon_eq > 180, lon_eq - 360, lon_eq)

    bin_edges = np.linspace(-180, 180, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    means = np.full(n_bins, np.nan)
    mins = np.full(n_bins, np.nan)
    maxs = np.full(n_bins, np.nan)

    for i in range(n_bins):
        in_bin = (lon_eq >= bin_edges[i]) & (lon_eq < bin_edges[i + 1])
        if np.any(in_bin):
            means[i] = np.mean(val_eq[in_bin])
            mins[i] = np.min(val_eq[in_bin])
            maxs[i] = np.max(val_eq[in_bin])

    return bin_centers, means, mins, maxs


def bin_latitudinal(lats, values, n_bins=90):
    """Bin values by latitude (zonal mean).

    Returns bin_centers (deg), bin_means, bin_mins, bin_maxs.
    """
    bin_edges = np.linspace(-90, 90, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    means = np.full(n_bins, np.nan)
    mins = np.full(n_bins, np.nan)
    maxs = np.full(n_bins, np.nan)

    for i in range(n_bins):
        in_bin = (lats >= bin_edges[i]) & (lats < bin_edges[i + 1])
        if np.any(in_bin):
            means[i] = np.mean(values[in_bin])
            mins[i] = np.min(values[in_bin])
            maxs[i] = np.max(values[in_bin])

    return bin_centers, means, mins, maxs


# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------

RHO_ICE = 917.0
SEC_PER_KYR = 1000 * 365.25 * 24 * 3600


# ---------------------------------------------------------------------------
# Load case data
# ---------------------------------------------------------------------------

def load_case(data_dir, epoch=None):
    """Load all fields needed for the combined figure."""
    lats, lons, n_faces = read_ply_mesh(os.path.join(data_dir, 'mesh.ply'))

    epochs = discover_epochs(data_dir, 'net_flux_kg_m2_s')
    if epoch is None:
        epoch = epochs[-1] if epochs else 0

    fields = {}

    for field in ['temperature_mean_K', 'temperature_min_K',
                   'temperature_max_K', 'vapor_pressure_Pa',
                   'emission_flux_kg_m2_s', 'deposition_flux_kg_m2_s',
                   'net_flux_kg_m2_s', 'sputtering_flux_kg_m2_s']:
        try:
            fields[field] = load_field_at_epoch(data_dir, field, epoch)
        except FileNotFoundError:
            pass

    return lats, lons, fields, epoch


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def fig_transport_combined(cases, save_path, epoch=None):
    """3×2 panel combined transport figure.

    cases: list of dicts with 'data_dir', 'label', optionally 'color'.
    """
    COLORS = ['C0', 'C3', 'C2', 'C4', 'C5']

    fig, axes = plt.subplots(3, 2, figsize=(10, 10.5))
    fig.subplots_adjust(hspace=0.38, wspace=0.35)

    for ci, case in enumerate(cases):
        color = case.get('color', COLORS[ci % len(COLORS)])
        label = case['label']
        lats, lons, fields, ep = load_case(case['data_dir'], epoch)

        has_sputtering = 'sputtering_flux_kg_m2_s' in fields

        # --- Temperature ---
        T_mean = fields.get('temperature_mean_K')
        T_min = fields.get('temperature_min_K')
        T_max = fields.get('temperature_max_K')

        if T_mean is not None and ci == 0:
            # Temperature is the same for both cases (no albedo feedback),
            # so only plot once.
            # (a) Equatorial band temperature
            ax = axes[0, 0]
            lc, tm, tmin, tmax = bin_equatorial(lons, T_mean, lats=lats)
            ax.plot(lc, tm, color='0.15', linewidth=1.5, label='Annual avg.')
            if T_min is not None and T_max is not None:
                _, tmin_b, _, _ = bin_equatorial(lons, T_min, lats=lats)
                _, tmax_b, _, _ = bin_equatorial(lons, T_max, lats=lats)
                ax.fill_between(lc, tmin_b, tmax_b, alpha=0.2, color='C0',
                                label='Min\u2013Max range')

            # (b) Latitudinal temperature
            ax = axes[0, 1]
            lc_lat, tm_lat, _, _ = bin_latitudinal(lats, T_mean)
            ax.plot(lc_lat, tm_lat, color='0.15', linewidth=1.5,
                    label='Annual avg.')
            if T_min is not None and T_max is not None:
                _, tmin_lat, _, _ = bin_latitudinal(lats, T_min)
                _, tmax_lat, _, _ = bin_latitudinal(lats, T_max)
                ax.fill_between(lc_lat, tmin_lat, tmax_lat, alpha=0.2,
                                color='C0', label='Min\u2013Max range')

        # --- Vapor pressure ratio ---
        Pv = fields.get('vapor_pressure_Pa')
        if Pv is not None:
            P_bar = np.mean(Pv)

            # (c) Equatorial P_v/P̄
            ax = axes[1, 0]
            lc, pm, _, _ = bin_equatorial(lons, Pv / P_bar, lats=lats)
            ax.plot(lc, pm, color=color, linewidth=1.5)
            if ci == 0:
                ax.axhline(1.0, color='0.6', linestyle='--', linewidth=0.8)

            # (d) Latitudinal P_v/P̄
            ax = axes[1, 1]
            lc_lat, pm_lat, _, _ = bin_latitudinal(lats, Pv / P_bar)
            ax.plot(lc_lat, pm_lat, color=color, linewidth=1.5)
            if ci == 0:
                ax.axhline(1.0, color='0.6', linestyle='--', linewidth=0.8)

        # --- Net transfer rate ---
        net_flux = fields.get('net_flux_kg_m2_s')
        sput_flux = fields.get('sputtering_flux_kg_m2_s')

        if net_flux is not None:
            # Convert to µm/kyr
            net_um_kyr = net_flux / RHO_ICE * 1e6 * SEC_PER_KYR

            # (e) Equatorial net flux vs longitude
            ax = axes[2, 0]
            lc, nm, _, _ = bin_equatorial(lons, net_um_kyr, lats=lats)
            ax.plot(lc, nm, color=color, linewidth=1.5)

            # Fill positive (accretion) and negative (removal)
            if len(cases) <= 2:
                ax.fill_between(lc, 0, nm, where=(nm >= 0),
                                alpha=0.15 if ci > 0 else 0.25,
                                color='steelblue' if ci == 0 else color)
                ax.fill_between(lc, 0, nm, where=(nm < 0),
                                alpha=0.15 if ci > 0 else 0.25,
                                color='indianred' if ci == 0 else color)

            # (f) Latitudinal net flux
            ax = axes[2, 1]
            lc_lat, nm_lat, _, _ = bin_latitudinal(lats, net_um_kyr)
            ax.plot(lc_lat, nm_lat, color=color, linewidth=1.5)

            if len(cases) <= 2:
                ax.fill_between(lc_lat, 0, nm_lat, where=(nm_lat >= 0),
                                alpha=0.15 if ci > 0 else 0.25,
                                color='steelblue' if ci == 0 else color)
                ax.fill_between(lc_lat, 0, nm_lat, where=(nm_lat < 0),
                                alpha=0.15 if ci > 0 else 0.25,
                                color='indianred' if ci == 0 else color)

    # --- Formatting ---
    panel_labels = 'abcdef'
    legend_kw = dict(fontsize=7, framealpha=0.85, edgecolor='none',
                     handlelength=1.5, handletextpad=0.5)

    # (a) Equatorial temperature
    ax = axes[0, 0]
    ax.set_title(r'Equatorial Band ($|\varphi| < 5\degree$)', fontsize=11)
    ax.set_ylabel('Temperature (K)')
    ax.set_xlim(-180, 180)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(60))
    ax.legend(loc='lower left', **legend_kw)

    # (b) Latitudinal temperature
    ax = axes[0, 1]
    ax.set_title('Latitudinal Profile (zonal mean)', fontsize=11)
    ax.set_ylabel('Temperature (K)')
    ax.set_xlim(-90, 90)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(30))
    ax.legend(loc='upper right', **legend_kw)

    # (c) Equatorial P_v/P̄
    ax = axes[1, 0]
    ax.set_ylabel(r'$P_v / \bar{P}$')
    ax.set_xlim(-180, 180)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(60))

    # (d) Latitudinal P_v/P̄
    ax = axes[1, 1]
    ax.set_ylabel(r'$P_v / \bar{P}$')
    ax.set_xlim(-90, 90)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(30))

    # (e) Equatorial net flux
    ax = axes[2, 0]
    ax.set_xlabel('East longitude from sub-Jovian point (\u00b0)')
    ax.set_ylabel(r'Net transfer rate ($\mu$m kyr$^{-1}$)')
    ax.set_xlim(-180, 180)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(60))
    ax.axhline(0, color='0.4', linewidth=0.5)

    # (f) Latitudinal net flux
    ax = axes[2, 1]
    ax.set_xlabel('Latitude (\u00b0)')
    ax.set_ylabel(r'Net transfer rate ($\mu$m kyr$^{-1}$)')
    ax.set_xlim(-90, 90)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(30))
    ax.axhline(0, color='0.4', linewidth=0.5)

    # Panel labels
    for i, ax in enumerate(axes.flat):
        ax.text(-0.12, 1.05, f'({panel_labels[i]})', transform=ax.transAxes,
                fontsize=12, fontweight='bold', va='top')

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    for ext in ['.png', '.pdf']:
        fig.savefig(save_path.rsplit('.', 1)[0] + ext)
    print(f'  Saved {save_path}')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Combined equatorial + latitudinal transport profiles')
    parser.add_argument('--data-dir', nargs='+', required=True,
                        help='Exosphere output directory(s)')
    parser.add_argument('--labels', nargs='+', default=None,
                        help='Labels for each case')
    parser.add_argument('--save-dir', default='docs/figures/transport',
                        help='Output directory')
    parser.add_argument('--epoch', type=int, default=None,
                        help='Epoch to plot (default: last available)')
    parser.add_argument('--filename', default='transport_profiles_combined',
                        help='Output filename (without extension)')
    args = parser.parse_args()

    labels = args.labels or [os.path.basename(d) for d in args.data_dir]
    if len(labels) != len(args.data_dir):
        print("Error: number of labels must match number of data dirs")
        sys.exit(1)

    cases = [{'data_dir': d, 'label': l}
             for d, l in zip(args.data_dir, labels)]

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, args.filename + '.png')

    print(f"Generating combined transport profiles ({len(cases)} case(s))...")
    fig_transport_combined(cases, save_path, epoch=args.epoch)

    print(f"\nDone! Figures saved to {args.save_dir}")


if __name__ == '__main__':
    main()
