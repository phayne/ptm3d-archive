#!/usr/bin/env python3
"""Diurnal surface temperature curves for Europa.

Plots surface temperature vs local time for the sub-Jovian and anti-Jovian
hemispheres at several latitudes.  If two season directories are provided,
the per-face temperatures are averaged to remove the seasonal N/S asymmetry.

Usage:
    python scripts/plot_diurnal_temperature.py \
        --season1 output/europa_ico6_uniform-ice \
        --season2 output/europa_ico6_uniform-ice_opp-season \
        --save-dir plots/global_temperature
"""

import os
import struct
import argparse
import numpy as np

from scipy.signal import savgol_filter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def _frame_file(directory, name, t):
    """Frame ``t`` of ``name`` under either naming (BUGS.md PTM-664).

    Current engines pad the index to >= 6 digits; runs written before used 4.
    """
    for w in range(6, 13):
        p = os.path.join(str(directory), "%s_%0*d.bin" % (name, w, t))
        if os.path.exists(p):
            return p
    return os.path.join(str(directory), "%s_%04d.bin" % (name, t))



# ---------------------------------------------------------------------------
# Publication style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
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
        type_sizes = {'float': 4, 'double': 8, 'int': 4, 'uint': 4,
                      'uchar': 1, 'short': 2, 'ushort': 2, 'int32': 4,
                      'uint32': 4, 'float32': 4, 'float64': 8}
        while True:
            line = f.readline().decode('ascii').strip()
            if line.startswith('element vertex'):
                n_vertices = int(line.split()[-1])
                in_vertex = True
            elif line.startswith('element face'):
                n_faces = int(line.split()[-1])
                in_vertex = False
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
    """Read uint32 count + float32 array."""
    with open(filepath, 'rb') as f:
        count = struct.unpack('<I', f.read(4))[0]
        data = np.frombuffer(f.read(count * 4), dtype='<f4')
    return data.astype(np.float64)


def load_temperature_cube(data_dir, n_steps=288):
    """Load all timestep temperature files into a (n_faces, n_steps) array."""
    temps = []
    for i in range(n_steps):
        fpath = _frame_file(os.path.join(data_dir, 'temperature'), 'temperature', i)
        temps.append(read_binary_field(fpath))
    return np.column_stack(temps)


def read_time_steps(data_dir):
    """Read time_steps.txt and return elapsed seconds array."""
    fpath = os.path.join(data_dir, 'time_steps.txt')
    times = []
    with open(fpath) as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            times.append(float(parts[0]))
    return np.array(times)


def compute_subsolar_longitude(data_dir, n_steps=288):
    """Compute sub-solar longitude at each timestep from body_positions.csv.

    For Europa (tidally locked), the sub-solar longitude in the body-fixed
    frame determines local time.  We compute it from the sun position
    relative to Europa.
    """
    import csv
    fpath = os.path.join(data_dir, 'body_positions.csv')
    sun_lons = []
    with open(fpath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            sx = float(row['sun_x'])
            sy = float(row['sun_y'])
            lon = np.degrees(np.arctan2(sy, sx)) % 360
            sun_lons.append(lon)
    return np.array(sun_lons[:n_steps])


# ---------------------------------------------------------------------------
# Face selection
# ---------------------------------------------------------------------------

def find_nearest_face(lats, lons, target_lat, target_lon, lat_tol=3, lon_tol=10):
    """Find face indices near (target_lat, target_lon) and return the closest."""
    dlat = np.abs(lats - target_lat)
    dlon_raw = np.abs(lons % 360 - target_lon % 360)
    dlon = np.minimum(dlon_raw, 360 - dlon_raw)
    mask = (dlat < lat_tol) & (dlon < lon_tol)
    if not np.any(mask):
        # Broaden search
        dist = np.sqrt(dlat**2 + dlon**2)
        return np.argmin(dist)
    candidates = np.where(mask)[0]
    dist = np.sqrt(dlat[candidates]**2 + dlon[candidates]**2)
    return candidates[np.argmin(dist)]


def find_faces_in_lat_band(lats, target_lat, lat_tol=3):
    """Find all faces within a latitude band (all longitudes)."""
    return np.where(np.abs(lats - target_lat) < lat_tol)[0]


def find_faces_in_band(lats, lons, target_lat, lon_center, lat_tol=3, lon_tol=30):
    """Find all faces within a latitude band and longitude range.

    Returns indices for averaging to get a smoother diurnal curve.
    """
    dlat = np.abs(lats - target_lat)
    dlon_raw = np.abs(lons % 360 - lon_center % 360)
    dlon = np.minimum(dlon_raw, 360 - dlon_raw)
    mask = (dlat < lat_tol) & (dlon < lon_tol)
    if np.sum(mask) < 3:
        # Broaden
        mask = (dlat < lat_tol * 2) & (dlon < lon_tol * 2)
    return np.where(mask)[0]


# ---------------------------------------------------------------------------
# Local time computation
# ---------------------------------------------------------------------------

def compute_local_time(face_lon, subsolar_lons):
    """Convert face longitude + sub-solar longitude to local solar time (hours).

    Local noon = 12h when face_lon == subsolar_lon.
    """
    # Hour angle: difference in longitude
    hour_angle = (face_lon - subsolar_lons) % 360  # 0-360
    # Convert to local time: 0° difference = noon (12h)
    local_time = (hour_angle / 360 * 24 + 12) % 24
    return local_time


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

LATITUDES = [0, 15, 30, 45, 60, 75]

# Colors: warm-to-cool gradient for equator-to-pole
_LAT_COLORS = ['#d62728', '#ff7f0e', '#bcbd22', '#2ca02c', '#1f77b4', '#9467bd']


def _bin_diurnal(temp_cube, face_lons, subsolar_lons, idxs, n_bins=288):
    """Compute a clean diurnal curve by binning per-face local times.

    For each face in *idxs* and each timestep, compute the face's local
    solar time, then bin-average all (lt, T) pairs into *n_bins* uniform
    local-time bins.  This avoids smearing from assigning a single
    center-face local time to temperature averaged over a longitude band.

    Returns (lt_centers, T_binned) arrays of length n_bins.
    """
    n_steps = temp_cube.shape[1]
    # Per-face local times: shape (n_faces_sel, n_steps)
    face_lts = np.array([compute_local_time(face_lons[fi], subsolar_lons)
                         for fi in idxs])  # (n_sel, n_steps)

    # Bin edges: 0 to 24 hours
    bin_edges = np.linspace(0, 24, n_bins + 1)
    lt_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    T_sum = np.zeros(n_bins)
    T_count = np.zeros(n_bins, dtype=int)

    for j, fi in enumerate(idxs):
        lts = face_lts[j]       # (n_steps,)
        temps = temp_cube[fi]   # (n_steps,)
        bin_idx = np.digitize(lts, bin_edges) - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
        for b in range(n_bins):
            mask = bin_idx == b
            T_sum[b] += temps[mask].sum()
            T_count[b] += mask.sum()

    valid = T_count > 0
    T_binned = np.full(n_bins, np.nan)
    T_binned[valid] = T_sum[valid] / T_count[valid]

    # Smooth out mesh-geometry noise (~0.3 K wiggles from icosahedral
    # face distribution).  Window of 25 bins (~2 hours) with mode='wrap'
    # handles the 0/24h boundary.
    if np.all(valid):
        T_binned = savgol_filter(T_binned, 25, 3, mode='wrap')

    return lt_centers, T_binned


def _plot_diurnal_on_ax(ax, temp_cube, lats, lons, subsolar_lons,
                        title=None, hemispheres=True):
    """Plot diurnal curves for all latitudes on a given axes.

    Parameters
    ----------
    hemispheres : bool
        If True, plot sub-Jovian (solid) and anti-Jovian (dashed) separately.
        If False, use all longitudes for a single curve per latitude.
    """
    for li, target_lat in enumerate(LATITUDES):
        color = _LAT_COLORS[li]

        if hemispheres:
            for hem_lon, ls in [(0, '-'), (180, '--')]:
                idxs = find_faces_in_band(lats, lons, target_lat, hem_lon,
                                          lat_tol=3, lon_tol=25)
                if len(idxs) == 0:
                    continue
                lt_sorted, t_sorted = _bin_diurnal(
                    temp_cube, lons, subsolar_lons, idxs)
                lt_plot = np.concatenate([lt_sorted, [lt_sorted[0] + 24]])
                t_plot = np.concatenate([t_sorted, [t_sorted[0]]])
                ax.plot(lt_plot, t_plot, color=color, ls=ls, linewidth=1.3)
        else:
            idxs = find_faces_in_lat_band(lats, target_lat, lat_tol=3)
            if len(idxs) == 0:
                continue
            lt_sorted, t_sorted = _bin_diurnal(
                temp_cube, lons, subsolar_lons, idxs)
            lt_plot = np.concatenate([lt_sorted, [lt_sorted[0] + 24]])
            t_plot = np.concatenate([t_sorted, [t_sorted[0]]])
            ax.plot(lt_plot, t_plot, color=color, ls='-', linewidth=1.5)

    ax.set_xlabel('Local Solar Time (hours)')
    ax.set_ylabel('Surface Temperature (K)')
    ax.set_xlim(0, 24)
    ax.set_xticks(np.arange(0, 25, 3))
    ax.set_xticklabels(['0', '3', '6', '9', '12\n(noon)', '15', '18', '21', '24'])
    ax.tick_params(direction='in', top=True, right=True)
    if title:
        ax.set_title(title, fontsize=13)


def _diurnal_legend_handles(hemispheres=True):
    """Return standard legend handles for diurnal curves."""
    handles = []
    for li, target_lat in enumerate(LATITUDES):
        handles.append(Line2D([], [], color=_LAT_COLORS[li], ls='-',
                              linewidth=1.5,
                              label=f'{target_lat}\u00b0N'))
    if hemispheres:
        handles.append(Line2D([], [], color='0.4', ls='-', linewidth=1.5,
                              label='Sub-Jovian (0\u00b0E)'))
        handles.append(Line2D([], [], color='0.4', ls='--', linewidth=1.5,
                              label='Anti-Jovian (180\u00b0E)'))
    return handles


def fig_diurnal_curves(temp_cube, lats, lons, subsolar_lons, save_path,
                       title=None, hemispheres=True):
    """Plot T vs local time for sub-Jovian and anti-Jovian hemispheres."""
    fig, ax = plt.subplots(figsize=(7.5, 4.0))

    _plot_diurnal_on_ax(ax, temp_cube, lats, lons, subsolar_lons,
                        title=title, hemispheres=hemispheres)

    ax.legend(handles=_diurnal_legend_handles(hemispheres), fontsize=9,
              bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0,
              framealpha=0.9, columnspacing=1.0)

    fig.tight_layout()
    _save(fig, save_path)


def fig_diurnal_comparison(cases, save_path):
    """Two-panel comparison of diurnal curves for different albedo models.

    The first case uses all-longitude averaging (no hemisphere separation);
    subsequent cases show sub-Jovian vs anti-Jovian.

    Parameters
    ----------
    cases : list of dict
        Each dict has keys: 'temp_cube', 'lats', 'lons', 'subsolar_lons',
        'label', and optionally 'hemispheres' (bool, default True).
    """
    n = len(cases)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4.0), sharey=True)
    if n == 1:
        axes = [axes]

    for ci, case in enumerate(cases):
        hem = case.get('hemispheres', True)
        _plot_diurnal_on_ax(axes[ci], case['temp_cube'], case['lats'],
                            case['lons'], case['subsolar_lons'],
                            title=case['label'], hemispheres=hem)
        if ci > 0:
            axes[ci].set_ylabel('')

    # Use the broadest legend (from the hemisphere panel)
    any_hem = any(c.get('hemispheres', True) for c in cases)
    axes[-1].legend(handles=_diurnal_legend_handles(any_hem), fontsize=9,
                    bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0,
                    framealpha=0.9, columnspacing=1.0)

    fig.tight_layout()
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

def load_case(season1, season2=None):
    """Load mesh, temperature cubes, and sub-solar longitudes for one case.

    When two seasons are provided, both cubes and sub-solar arrays are
    concatenated along the time axis so that ``_bin_diurnal`` can place
    each sample at the correct local solar time.  (The previous approach
    of averaging by timestep index was wrong because the two seasons
    have different sub-solar longitude sequences.)

    Returns dict with keys: lats, lons, nf, temp_cube, subsolar_lons, n_steps.
    """
    ply_path = os.path.join(season1, 'mesh.ply')
    lats, lons, nf = read_ply_mesh(ply_path)
    print(f"  Mesh: {nf} faces")

    temp_dir = os.path.join(season1, 'temperature')
    n_steps = len([f for f in os.listdir(temp_dir) if f.endswith('.bin')])
    print(f"  Time steps: {n_steps}")

    print("  Loading season 1 temperatures...")
    cube1 = load_temperature_cube(season1, n_steps)
    print(f"    Shape: {cube1.shape}, range: [{cube1.min():.1f}, {cube1.max():.1f}] K")
    ss1 = compute_subsolar_longitude(season1, n_steps)

    if season2:
        temp_dir2 = os.path.join(season2, 'temperature')
        n_steps2 = len([f for f in os.listdir(temp_dir2) if f.endswith('.bin')])
        print("  Loading season 2 temperatures...")
        cube2 = load_temperature_cube(season2, n_steps2)
        print(f"    Shape: {cube2.shape}, range: [{cube2.min():.1f}, {cube2.max():.1f}] K")
        ss2 = compute_subsolar_longitude(season2, n_steps2)

        # Concatenate along time axis — _bin_diurnal will place each
        # sample at the correct local time using the matching subsolar lon.
        temp_cube = np.concatenate([cube1, cube2], axis=1)
        subsolar_lons = np.concatenate([ss1, ss2])
        n_steps_total = n_steps + n_steps2
        print(f"    Combined: {temp_cube.shape[1]} timesteps, "
              f"range: [{temp_cube.min():.1f}, {temp_cube.max():.1f}] K")
    else:
        temp_cube = cube1
        subsolar_lons = ss1
        n_steps_total = n_steps

    print(f"  Sub-solar longitude range: [{subsolar_lons.min():.1f}, "
          f"{subsolar_lons.max():.1f}]\u00b0")

    return dict(lats=lats, lons=lons, nf=nf, temp_cube=temp_cube,
                subsolar_lons=subsolar_lons, n_steps=n_steps_total)


def main():
    parser = argparse.ArgumentParser(
        description='Diurnal temperature curves for Europa')
    parser.add_argument('--season1', required=True,
                        help='Season-1 output directory')
    parser.add_argument('--season2', default=None,
                        help='Season-2 directory for N/S averaging')
    parser.add_argument('--save-dir', default='plots/global_temperature',
                        help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    print("Loading data...")
    data = load_case(args.season1, args.season2)

    print("\nGenerating diurnal temperature curves...")
    fig_diurnal_curves(
        data['temp_cube'], data['lats'], data['lons'],
        data['subsolar_lons'],
        save_path=os.path.join(args.save_dir, 'diurnal_temperature.png'),
        hemispheres=True,
    )

    print("\nDone!")


if __name__ == '__main__':
    main()
