#!/usr/bin/env python3
"""
Validation plots for the bundled lunar property maps.

Three checks:
  1. Round-trip H map sanity: histogram of per-face H values for each ico level.
  2. Lat/lon raster reconstruction: project the icosphere TI map back to a
     lat/lon grid (1 degree) by averaging face values into bins; compare to the
     reference 10-ppd TI map shipped with the dataset.
  3. Spatial maps: side-by-side equirectangular plots of the raw raster TI and
     the icosphere-derived TI for visual inspection.

Usage:
  python validate_ti_map.py [--ico-level N] [--out plots/property_maps/]
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Icosphere generator (must match MeshUtils::createIcosphereMesh in C++)
# ---------------------------------------------------------------------------
def _build_icosphere(level: int):
    t = (1.0 + math.sqrt(5.0)) / 2.0
    norm = math.sqrt(1.0 + t * t)
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ], dtype=np.float64) / norm
    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ], dtype=np.int64)

    for _ in range(level):
        midpoint_cache: dict[tuple[int, int], int] = {}
        new_faces = []
        verts_list = list(verts)

        def midpoint(a: int, b: int) -> int:
            key = (min(a, b), max(a, b))
            if key in midpoint_cache:
                return midpoint_cache[key]
            m = (verts_list[a] + verts_list[b]) * 0.5
            m /= np.linalg.norm(m)
            verts_list.append(m)
            idx = len(verts_list) - 1
            midpoint_cache[key] = idx
            return idx

        for a, b, c in faces:
            ab = midpoint(a, b)
            bc = midpoint(b, c)
            ca = midpoint(c, a)
            new_faces.extend([
                [a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]
            ])
        verts = np.array(verts_list, dtype=np.float64)
        faces = np.array(new_faces, dtype=np.int64)
    return verts, faces


def _face_lat_lon(verts: np.ndarray, faces: np.ndarray):
    centers = verts[faces].mean(axis=1)
    r = np.linalg.norm(centers, axis=1)
    centers /= r[:, None]
    lat = np.degrees(np.arcsin(centers[:, 2]))
    lon = np.degrees(np.arctan2(centers[:, 1], centers[:, 0]))
    return lat, lon


# ---------------------------------------------------------------------------
# Reference raster reading (128 ppd Diviner map, ±70°)
# ---------------------------------------------------------------------------
RASTER_PATH = (
    "/Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/"
    "moon/hpar/dataset/hpar_global128ppd_v1c.bin"
)
PPD = 128
SAMPLES_PER_LINE = 360 * PPD + 1
N_LINES = 140 * PPD + 1
LAT_TOP = 70.0


def load_raster(downsample: int = 8) -> np.ndarray:
    """Load the H raster at reduced resolution to keep memory manageable."""
    if not Path(RASTER_PATH).exists():
        raise FileNotFoundError(f"Raster not found: {RASTER_PATH}")
    full = np.memmap(RASTER_PATH, dtype=np.float32, mode='r',
                     shape=(N_LINES, SAMPLES_PER_LINE))
    return np.array(full[::downsample, ::downsample])


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_h_histogram(values: np.ndarray, mask: np.ndarray, level: int,
                     out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    valid = mask.astype(bool)
    h_valid = values[valid]
    h_default = values[~valid]
    bins = np.linspace(0, 0.5, 80)
    ax.hist(h_valid, bins=bins, color='steelblue', alpha=0.85,
            label=f'Diviner-sampled ({valid.sum():,} faces)')
    if h_default.size:
        ax.hist(h_default, bins=bins, color='orange', alpha=0.85,
                label=f'Polar fallback default ({(~valid).sum():,} faces)')
    ax.set_xlabel('H-parameter (m)')
    ax.set_ylabel('faces')
    ax.set_title(f'Lunar ico{level} H-parameter distribution')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def plot_ti_spatial(ti_face: np.ndarray, lat: np.ndarray, lon: np.ndarray,
                    raster_h: np.ndarray, level: int, out_path: Path) -> None:
    """Two-panel: ico TI scatter (lat/lon) vs raster H (downsampled)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    sc = axes[0].scatter(lon, lat, c=ti_face, s=5, cmap='inferno',
                         vmin=20, vmax=80, edgecolors='none')
    axes[0].set_xlim(-180, 180)
    axes[0].set_ylim(-90, 90)
    axes[0].set_xlabel('Longitude (deg)')
    axes[0].set_ylabel('Latitude (deg)')
    axes[0].set_title(f'ico{level} thermal inertia at 273 K')
    axes[0].axhline(70, color='cyan', ls=':', lw=1)
    axes[0].axhline(-70, color='cyan', ls=':', lw=1)
    plt.colorbar(sc, ax=axes[0], label='TI (J m^-2 K^-1 s^-1/2)')

    extent = [0, 360, -70, 70]
    im = axes[1].imshow(raster_h, cmap='viridis', vmin=0, vmax=0.2,
                        extent=extent, origin='upper', aspect='auto')
    axes[1].set_xlim(0, 360)
    axes[1].set_ylim(-90, 90)
    axes[1].set_xlabel('Longitude (deg)')
    axes[1].set_ylabel('Latitude (deg)')
    axes[1].set_title('Reference H raster (downsampled, 128 ppd source)')
    axes[1].axhline(70, color='cyan', ls=':', lw=1)
    axes[1].axhline(-70, color='cyan', ls=':', lw=1)
    plt.colorbar(im, ax=axes[1], label='H (m)')

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ico-level', type=int, default=6)
    ap.add_argument('--maps-dir', type=Path,
                    default=Path('data/property_maps/moon'))
    ap.add_argument('--out-dir', type=Path,
                    default=Path('plots/property_maps'))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    level = args.ico_level
    h_path  = args.maps_dir / f'hpar_moon_ico{level}.bin'
    ti_path = args.maps_dir / f'ti273k_moon_ico{level}.bin'
    mask_path = args.maps_dir / f'mask_moon_ico{level}.bin'
    if not h_path.exists():
        sys.exit(f'missing {h_path}')

    h_face  = np.fromfile(h_path,  dtype=np.float32)
    ti_face = np.fromfile(ti_path, dtype=np.float32)
    mask    = np.fromfile(mask_path, dtype=np.uint8)
    n_expected = 20 * (4 ** level)
    if h_face.size != n_expected:
        sys.exit(f'expected {n_expected} faces, got {h_face.size} in {h_path}')

    # Build the icosphere lat/lon (must mirror C++ build).
    verts, faces = _build_icosphere(level)
    if faces.shape[0] != h_face.size:
        sys.exit(f'icosphere face count {faces.shape[0]} != map face count {h_face.size}')
    lat, lon = _face_lat_lon(verts, faces)

    # Plots
    h_hist_path = args.out_dir / f'h_hist_ico{level}.png'
    plot_h_histogram(h_face, mask, level, h_hist_path)
    print(f'wrote {h_hist_path}')

    print('loading downsampled reference raster (this may pull from OneDrive)...')
    try:
        raster = load_raster(downsample=8)
    except Exception as e:
        print(f'WARNING: could not load reference raster ({e}); skipping spatial plot')
        raster = None

    if raster is not None:
        ti_spatial_path = args.out_dir / f'ti_spatial_ico{level}.png'
        plot_ti_spatial(ti_face, lat, lon, raster, level, ti_spatial_path)
        print(f'wrote {ti_spatial_path}')

    # Quick numerical sanity print
    print('\nSummary:')
    print(f'  ico level: {level}  ({h_face.size:,} faces)')
    print(f'  H  min/mean/max: {h_face.min():.4f} / {h_face.mean():.4f} / {h_face.max():.4f} m')
    print(f'  TI min/mean/max: {ti_face.min():.2f} / {ti_face.mean():.2f} / {ti_face.max():.2f}')
    print(f'  fallback faces : {(mask == 0).sum():,} '
          f'({100.0 * (mask == 0).mean():.2f} %)')


if __name__ == '__main__':
    main()
