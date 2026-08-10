"""Cylindrical projection of the bundled per-face H map for visual
comparison with Hayne et al. (2017), Figure 7 (global H-parameter map).

For each ico level we have, render the per-face H to a 720x360 (0.5 deg/px)
lat/lon grid using nearest-neighbor assignment from face centers reconstructed
exactly as MeshUtils::createIcosphereMesh produces them.

The Hayne paper's color scale is H ∈ [0.02, 0.09] m. We match that here so
the maps are directly comparable. Notable features that should appear:
  - Tycho ejecta (~43°S, -11°E)         — low H (bright in TI / dark blue in H)
  - Tycho antipode (~45°N, +169°E)       — low H spot
  - Aristarchus plateau (~25°N, +310°E)  — high H (yellow in our map)
  - Giordano Bruno (~36°N, +103°E)        — low H rayed crater
  - South Pole–Aitken basin (south side)  — slightly higher H
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree


N_F_BY_RES = {3:1280, 4:5120, 5:20480, 6:81920, 7:327680, 8:1310720, 9:5242880}


def build_icosphere(level):
    """Mirror MeshUtils::createIcosphereMesh exactly. Returns face lat, lon."""
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
        cache = {}
        vlist = list(verts)
        nfaces = []

        def midpoint(a, b):
            k = (min(a, b), max(a, b))
            if k in cache:
                return cache[k]
            m = (vlist[a] + vlist[b]) * 0.5
            m /= np.linalg.norm(m)
            vlist.append(m)
            idx = len(vlist) - 1
            cache[k] = idx
            return idx

        for a, b, c in faces:
            ab = midpoint(a, b); bc = midpoint(b, c); ca = midpoint(c, a)
            nfaces.extend([[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]])
        verts = np.array(vlist, dtype=np.float64)
        faces = np.array(nfaces, dtype=np.int64)

    centers = verts[faces].mean(axis=1)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    lat = np.degrees(np.arcsin(centers[:, 2]))
    lon = np.degrees(np.arctan2(centers[:, 1], centers[:, 0]))
    return lat, lon, centers


def render_cylindrical(level, h_face, out_path: Path,
                       grid_dpd: int = 4, vmin=0.02, vmax=0.09):
    """Nearest-neighbor rasterization of per-face H onto an equirectangular
    grid at `grid_dpd` pixels per degree."""
    lat, lon, centers = build_icosphere(level)

    # Build kd-tree on the unit sphere using xyz to avoid lat/lon wrap issues.
    tree = cKDTree(centers)

    # Equirectangular grid
    n_lon = 360 * grid_dpd + 1
    n_lat = 180 * grid_dpd + 1
    lon_grid = np.linspace(-180.0, 180.0, n_lon)
    lat_grid = np.linspace(-90.0, 90.0, n_lat)
    LON, LAT = np.meshgrid(lon_grid, lat_grid)

    # Convert grid lat/lon to xyz
    rad_lat = np.radians(LAT)
    rad_lon = np.radians(LON)
    xyz = np.stack([
        np.cos(rad_lat) * np.cos(rad_lon),
        np.cos(rad_lat) * np.sin(rad_lon),
        np.sin(rad_lat),
    ], axis=-1).reshape(-1, 3)

    print(f'  querying nearest face for {xyz.shape[0]} grid pixels…')
    _, face_idx = tree.query(xyz, k=1)
    H_grid = h_face[face_idx].reshape(LAT.shape)

    # Plot
    fig, ax = plt.subplots(figsize=(15, 6.5))
    # Hayne uses a colormap with H reversed (blue/dark = low H = high TI;
    # yellow/light = high H = low TI). 'viridis' approximates this when shown
    # without inversion.
    im = ax.imshow(H_grid, origin='lower',
                   extent=[-180, 180, -90, 90],
                   cmap='viridis', vmin=vmin, vmax=vmax,
                   aspect='auto', interpolation='nearest')
    cbar = plt.colorbar(im, ax=ax, label='H (m)', shrink=0.85, pad=0.02)
    # Annotate key features
    annotations = [
        ('Tycho ejecta',         -43,  -11, 'cyan'),    # low H
        ('Tycho antipode',        45,  169, 'cyan'),    # low H
        ('Aristarchus plateau',   25, -50,  'orange'),  # high H (-50 = 310°E)
        ('Giordano Bruno',        36,  103, 'cyan'),    # low H
        ('Copernicus',            10,  -20, 'cyan'),    # low H
        ('Jackson',               22, -163, 'cyan'),    # low H
    ]
    for name, plat, plon, col in annotations:
        ax.scatter([plon], [plat], s=140, facecolors='none',
                   edgecolors=col, linewidths=1.4, zorder=10)
        ax.annotate(name, (plon, plat), xytext=(6, 6),
                    textcoords='offset points', color=col,
                    fontsize=9, fontweight='bold', zorder=11)
    ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
    ax.set_xlabel('East Longitude (deg)')
    ax.set_ylabel('Latitude (deg)')
    ax.set_title(f'Bundled per-face H map  (ico{level}, {h_face.size:,} faces) '
                 f'— nearest-neighbor to {grid_dpd}°⁻¹ grid')
    ax.grid(True, color='white', alpha=0.15, linewidth=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {out_path}')

    # Also print the H value at each of those known features (sampled
    # nearest-face) so we can sanity-check numerically.
    print(f'\n  Sampled H at known features (using nearest face center):')
    for name, plat, plon, _ in annotations:
        # Convert lat/lon to xyz, find nearest face
        rl = math.radians(plat); rln = math.radians(plon)
        p = np.array([math.cos(rl)*math.cos(rln), math.cos(rl)*math.sin(rln), math.sin(rl)])
        _, idx = tree.query(p, k=1)
        print(f'    {name:<22} ({plat:+4.0f}°, {plon:+4.0f}°): H = {h_face[idx]:.4f} m')


def main():
    out_dir = Path('plots/property_maps')
    out_dir.mkdir(parents=True, exist_ok=True)

    for level in (7, 9):
        path = Path('data/property_maps/moon') / f'hpar_moon_ico{level}.bin'
        if not path.exists():
            print(f'skip {path} (missing)')
            continue
        h = np.fromfile(path, dtype=np.float32)
        if h.size != N_F_BY_RES[level]:
            print(f'  expected {N_F_BY_RES[level]} faces, got {h.size}; skip')
            continue
        out_path = out_dir / f'hmap_cylindrical_ico{level}.png'
        print(f'rendering ico{level} ({h.size:,} faces) …')
        render_cylindrical(level, h, out_path)


if __name__ == '__main__':
    main()
