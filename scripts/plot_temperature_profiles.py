#!/usr/bin/env python3
"""Temperature profiles vs latitude and longitude for Europa simulations.

Produces publication-quality line plots of diurnal min, max, and mean surface
temperature as a function of latitude (zonal average) and longitude (equatorial
band average) for one or more simulation cases.

Usage:
    python scripts/plot_temperature_profiles.py \
        --cases output/europa_ico6_uniform-ice output/europa_ico6_albedo-map \
        --labels "Uniform albedo" "Albedo map" \
        --save-dir plots/global_temperature
"""

import os
import struct
import argparse
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
                parts = line.split()
                if len(parts) >= 3 and parts[1] != 'list':
                    vertex_bytes += type_sizes.get(parts[1], 4)
            elif line == 'end_header':
                break

        f.seek(f.tell() + n_vertices * vertex_bytes)

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
# Constants
# ---------------------------------------------------------------------------

STATS = [
    ('tmax',  'temperature_max.bin',  r'$T_\mathrm{max}$'),
    ('tmean', 'temperature_avg.bin',  r'$T_\mathrm{mean}$'),
    ('tmin',  'temperature_min.bin',  r'$T_\mathrm{min}$'),
]

_STAT_STYLES = {
    'tmax':  dict(color='C3', label=r'$T_\mathrm{max}$'),
    'tmean': dict(color='C0', label=r'$T_\mathrm{mean}$'),
    'tmin':  dict(color='C4', label=r'$T_\mathrm{min}$'),
}

_CASE_LINESTYLES = ['-', '--', '-.', ':']


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_case(data_dir, label):
    """Load mesh + temperature statistics for a case."""
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
# Profile helpers
# ---------------------------------------------------------------------------

def _zonal_profile(lat_g, data_g):
    """Average data_g over longitude -> T(lat)."""
    return np.nanmean(data_g, axis=1), lat_g


def _meridional_profile(lon_g, data_g, lat_g, lat_band=30):
    """Average data_g over low latitudes (|lat| < lat_band) -> T(lon)."""
    mask = np.abs(lat_g) <= lat_band
    return np.nanmean(data_g[mask, :], axis=0), lon_g


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_profiles_vs_latitude(cases, save_path):
    """T_min, T_mean, T_max vs latitude for all cases (zonal average)."""
    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    for ci, case in enumerate(cases):
        ls = _CASE_LINESTYLES[ci % len(_CASE_LINESTYLES)]
        for stat_key, _, _ in STATS:
            stat = case[stat_key]
            profile, lats = _zonal_profile(stat['lat_g'], stat['data_g'])
            sty = _STAT_STYLES[stat_key]
            ax.plot(lats, profile, color=sty['color'], ls=ls,
                    linewidth=1.2)

    ax.set_xlabel('Latitude (\u00b0N)')
    ax.set_ylabel('Temperature (K)')
    ax.set_xlim(-90, 90)
    ax.tick_params(direction='in', top=True, right=True)

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
        description='Temperature profiles vs latitude and longitude')
    parser.add_argument('--cases', nargs='+', required=True,
                        help='Simulation output directory(s)')
    parser.add_argument('--labels', nargs='+', default=None,
                        help='Labels for each case')
    parser.add_argument('--save-dir', default='plots/global_temperature',
                        help='Output directory for figures')
    parser.add_argument('--lat-band', type=float, default=30,
                        help='Latitude band for longitude profiles (default: 30)')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    labels = args.labels or [os.path.basename(d) for d in args.cases]

    print("Loading data...")
    cases = []
    for data_dir, label in zip(args.cases, labels):
        cases.append(_load_case(data_dir, label))

    print("\nTemperature vs latitude (zonal mean)...")
    fig_profiles_vs_latitude(
        cases,
        save_path=os.path.join(args.save_dir, 'profile_vs_latitude.png'),
    )

    print("Temperature vs longitude (equatorial band)...")
    fig_profiles_vs_longitude(
        cases,
        save_path=os.path.join(args.save_dir, 'profile_vs_longitude.png'),
        lat_band=args.lat_band,
    )

    print(f"\nDone! Figures saved to {args.save_dir}")


if __name__ == '__main__':
    main()
