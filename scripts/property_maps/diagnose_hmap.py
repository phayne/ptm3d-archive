"""
Diagnose whether the bundled per-face H map shows coherent spatial structure
or noise. Generates two cross-checks:

  1. Reconstruct the icosphere face lat/lon ourselves (mirroring
     MeshUtils::createIcosphereMesh) and scatter-plot the bundled H values.
     Compare to the original 128 ppd LOLA raster of H sampled at the same
     points — if these disagree, the icosphere face ordering used at
     generation time differs from what we expect.

  2. Compare bundled H to fresh raster-sampled H, face-by-face. Reports
     correlation, RMSE, and worst-discrepancy face indices.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


N_V_BY_RES = {3:642, 4:2562, 5:10242, 6:40962, 7:163842, 8:655362, 9:2621442}
N_F_BY_RES = {3:1280, 4:5120, 5:20480, 6:81920, 7:327680, 8:1310720, 9:5242880}


def build_icosphere(level: int):
    """Mirror MeshUtils::createIcosphereMesh face ordering exactly."""
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
        verts_list = list(verts)
        new_faces = []

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


def face_lat_lon(verts, faces):
    centers = verts[faces].mean(axis=1)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    lat = np.degrees(np.arcsin(centers[:, 2]))
    lon = np.degrees(np.arctan2(centers[:, 1], centers[:, 0]))
    return lat, lon


# ---------------------------------------------------------------------------
# Sample the LOLA H-par raster at given (lat, lon) using the same projection
# convention used by lunar_property_map_generator.cpp.
# ---------------------------------------------------------------------------
RASTER = ("/Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/"
          "moon/hpar/dataset/hpar_global128ppd_v1c.bin")
PPD = 128
SAMPLES_PER_LINE = 360 * PPD + 1   # 46081
N_LINES          = 140 * PPD + 1   # 17921
LAT0             = 70.0
DL               = 1.0 / PPD


def sample_raster(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear-sample the 128 ppd raster, mirroring HRaster::sample in C++.

    Returns (h_values, valid_mask) where valid_mask is True for points in
    the ±70° band.
    """
    raster = np.memmap(RASTER, dtype=np.float32, mode='r',
                       shape=(N_LINES, SAMPLES_PER_LINE))

    n = lat.size
    out  = np.full(n, np.nan, dtype=np.float64)
    valid = np.zeros(n, dtype=bool)

    in_band = (lat >= -70.0) & (lat <= 70.0)
    lon_n = np.where(lon < 0.0, lon + 360.0, lon)

    row_f = (LAT0 - lat) / DL
    col_f = lon_n / DL

    r0 = np.floor(row_f).astype(np.int32)
    c0 = np.floor(col_f).astype(np.int32)
    rt = row_f - r0
    ct = col_f - c0

    # Clamp r0
    r0 = np.clip(r0, 0, N_LINES - 2)
    rt = np.where(r0 == 0, np.maximum(rt, 0.0), rt)

    c1 = c0 + 1
    c0 = np.where(c0 < 0, 0, c0)
    c1 = np.where(c1 >= SAMPLES_PER_LINE, 0, c1)

    # Sample bilinearly (only for in-band points)
    idx = np.where(in_band)[0]
    for i in idx:
        v00 = raster[r0[i],   c0[i]]
        v01 = raster[r0[i],   c1[i]]
        v10 = raster[r0[i]+1, c0[i]]
        v11 = raster[r0[i]+1, c1[i]]
        w00 = (1 - rt[i]) * (1 - ct[i])
        w01 = (1 - rt[i]) * ct[i]
        w10 = rt[i]       * (1 - ct[i])
        w11 = rt[i]       * ct[i]
        ws = []
        vs = []
        for v, w in [(v00, w00), (v01, w01), (v10, w10), (v11, w11)]:
            if np.isfinite(v) and v > 0:
                ws.append(w)
                vs.append(v)
        if sum(ws) >= 0.5:
            out[i]  = np.dot(vs, ws) / sum(ws)
            valid[i] = True

    return out, valid


def main():
    level = 7
    n_v, n_f = N_V_BY_RES[level], N_F_BY_RES[level]
    bundled_path = Path('data/property_maps/moon') / f'hpar_moon_ico{level}.bin'
    h_bundled = np.fromfile(bundled_path, dtype=np.float32)
    if h_bundled.size != n_f:
        sys.exit(f'expected {n_f} values, got {h_bundled.size}')

    print(f'Building reference icosphere at level {level}...')
    verts, faces = build_icosphere(level)
    lat, lon = face_lat_lon(verts, faces)
    print(f'  {n_f} faces; lat range {lat.min():.1f}..{lat.max():.1f}  '
          f'lon range {lon.min():.1f}..{lon.max():.1f}')

    print(f'Sampling LOLA raster at the {n_f} reference face locations...')
    h_ref, in_band = sample_raster(lat, lon)
    print(f'  in_band: {in_band.sum()} of {n_f}')

    # Diagnostic stats
    finite = in_band & np.isfinite(h_ref) & np.isfinite(h_bundled)
    print(f'\n--- Per-face H (bundled) vs Per-face H (raster-sampled at our recomputed lat/lon) ---')
    print(f'  faces compared: {finite.sum()}')
    print(f'  bundled  range: {h_bundled[finite].min():.4f}..{h_bundled[finite].max():.4f}  m')
    print(f'  ref      range: {h_ref[finite].min():.4f}..{h_ref[finite].max():.4f}  m')
    diff = h_bundled[finite] - h_ref[finite]
    print(f'  diff     |mean|={np.abs(diff).mean():.4f}  rms={np.sqrt((diff**2).mean()):.4f}  '
          f'max={np.abs(diff).max():.4f} m')
    rho = np.corrcoef(h_bundled[finite], h_ref[finite])[0, 1]
    print(f'  Pearson rho   : {rho:.6f}  (1.0 = identical, 0 = noise)')

    # Plots: 4-panel comparison
    out_dir = Path('plots/property_maps')
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # Panel 1: bundled H (lat/lon scatter)
    sc = axes[0, 0].scatter(lon[finite], lat[finite], c=h_bundled[finite],
                             s=2, vmin=0, vmax=0.2, cmap='viridis')
    axes[0, 0].set_title(f'Bundled H (ico{level}, file)')
    axes[0, 0].set_xlabel('Longitude (deg)')
    axes[0, 0].set_ylabel('Latitude (deg)')
    axes[0, 0].set_xlim(-180, 180); axes[0, 0].set_ylim(-90, 90)
    plt.colorbar(sc, ax=axes[0, 0], label='H (m)')

    # Panel 2: reference H from raster at OUR lat/lon
    sc = axes[0, 1].scatter(lon[finite], lat[finite], c=h_ref[finite],
                             s=2, vmin=0, vmax=0.2, cmap='viridis')
    axes[0, 1].set_title('Raster-sampled H at recomputed face lat/lon')
    axes[0, 1].set_xlabel('Longitude (deg)')
    axes[0, 1].set_ylabel('Latitude (deg)')
    axes[0, 1].set_xlim(-180, 180); axes[0, 1].set_ylim(-90, 90)
    plt.colorbar(sc, ax=axes[0, 1], label='H (m)')

    # Panel 3: difference
    sc = axes[1, 0].scatter(lon[finite], lat[finite],
                             c=(h_bundled[finite] - h_ref[finite]),
                             s=2, vmin=-0.05, vmax=0.05, cmap='RdBu_r')
    axes[1, 0].set_title('Bundled - Raster (Δ H)')
    axes[1, 0].set_xlabel('Longitude (deg)')
    axes[1, 0].set_ylabel('Latitude (deg)')
    axes[1, 0].set_xlim(-180, 180); axes[1, 0].set_ylim(-90, 90)
    plt.colorbar(sc, ax=axes[1, 0], label='ΔH (m)')

    # Panel 4: scatter ref vs bundled
    axes[1, 1].scatter(h_ref[finite], h_bundled[finite], s=1, alpha=0.3)
    lim = max(0.4, h_ref[finite].max(), h_bundled[finite].max())
    axes[1, 1].plot([0, lim], [0, lim], 'r-', lw=0.7)
    axes[1, 1].set_xlabel('H (raster at our lat/lon)')
    axes[1, 1].set_ylabel('H (bundled file)')
    axes[1, 1].set_title(f'1:1 scatter, rho={rho:.4f}')
    axes[1, 1].grid(alpha=0.3)

    fig.tight_layout()
    out_path = out_dir / f'diagnose_hmap_ico{level}.png'
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    print(f'\nWrote diagnostic plot: {out_path}')


if __name__ == '__main__':
    main()
