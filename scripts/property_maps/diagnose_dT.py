"""Visualize the per-face temperature difference (with-map - no-map) on the
icosphere face lat/lon and check whether it correlates with the underlying H
field. Coherent patterns (low-H ↔ low-Tnight or high-Tday) imply correct
application; noise implies face-index scrambling somewhere."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def build_icosphere(level):
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
        cache, vlist, nf = {}, list(verts), []
        def mid(a, b):
            k = (min(a,b), max(a,b))
            if k in cache: return cache[k]
            m = (vlist[a] + vlist[b]) * 0.5
            m /= np.linalg.norm(m)
            vlist.append(m); cache[k] = len(vlist)-1
            return cache[k]
        for a, b, c in faces:
            ab, bc, ca = mid(a,b), mid(b,c), mid(c,a)
            nf.extend([[a,ab,ca],[b,bc,ab],[c,ca,bc],[ab,bc,ca]])
        verts = np.array(vlist); faces = np.array(nf, dtype=np.int64)
    centers = verts[faces].mean(axis=1)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    return np.degrees(np.arcsin(centers[:, 2])), np.degrees(np.arctan2(centers[:, 1], centers[:, 0]))


def load_T(d):
    files = sorted(Path(d).glob('temperature/temperature_*.bin'), key=lambda p: int(str(p).rsplit("_", 1)[1][:-4]))
    return np.stack([np.fromfile(f, dtype=np.float32) for f in files])


lat, lon = build_icosphere(7)
n_f = 327680
h = np.fromfile('data/property_maps/moon/hpar_moon_ico7.bin', dtype=np.float32)

t_with = load_T('output/moon_ico7_ti-test')
t_no   = load_T('/tmp/ico7_no_map_out')

# Trim potential trailing pad face (the bin files seem to have n_f+1 entries
# possibly due to a header byte; the first run reported 327681).
t_with = t_with[:, :n_f]
t_no   = t_no[:,   :n_f]
print(f'shapes: with={t_with.shape}  no={t_no.shape}  h={h.shape}')

dT_max  = t_with.max(axis=0).astype(np.float64) - t_no.max(axis=0).astype(np.float64)
dT_min  = t_with.min(axis=0).astype(np.float64) - t_no.min(axis=0).astype(np.float64)
dT_mean = t_with.mean(axis=0).astype(np.float64) - t_no.mean(axis=0).astype(np.float64)

# Anomaly relative to body default 0.06 m
hdiff = h.astype(np.float64) - 0.06

# Correlation: does dT_max anti-correlate with hdiff (low H = thinner regolith
# = lower thermal inertia = bigger Tmax in daylight)?
finite = np.isfinite(dT_max) & np.isfinite(hdiff)
rho_dTmax_hdiff  = np.corrcoef(dT_max[finite],  hdiff[finite])[0, 1]
rho_dTmin_hdiff  = np.corrcoef(dT_min[finite],  hdiff[finite])[0, 1]
rho_dTmean_hdiff = np.corrcoef(dT_mean[finite], hdiff[finite])[0, 1]
print(f'\nCorrelations (per face) between H-anomaly and ΔT:')
print(f'  rho(ΔTmax,  H-0.06) = {rho_dTmax_hdiff:+.4f}    (expect strongly NEGATIVE: low H ⇒ ΔTmax > 0)')
print(f'  rho(ΔTmin,  H-0.06) = {rho_dTmin_hdiff:+.4f}    (expect strongly POSITIVE: low H ⇒ ΔTmin < 0)')
print(f'  rho(ΔTmean, H-0.06) = {rho_dTmean_hdiff:+.4f}')

# 4-panel plot
fig, axes = plt.subplots(2, 2, figsize=(15, 8))

sc = axes[0, 0].scatter(lon, lat, c=h, s=2, cmap='viridis', vmin=0, vmax=0.2)
axes[0, 0].set_title('Bundled H map (input)')
axes[0, 0].set_xlim(-180, 180); axes[0, 0].set_ylim(-90, 90)
plt.colorbar(sc, ax=axes[0, 0], label='H (m)')

sc = axes[0, 1].scatter(lon, lat, c=dT_max, s=2, cmap='RdBu_r', vmin=-5, vmax=5)
axes[0, 1].set_title('ΔTmax (with-map - no-map)')
axes[0, 1].set_xlim(-180, 180); axes[0, 1].set_ylim(-90, 90)
plt.colorbar(sc, ax=axes[0, 1], label='ΔTmax (K)')

sc = axes[1, 0].scatter(lon, lat, c=dT_min, s=2, cmap='RdBu_r', vmin=-5, vmax=5)
axes[1, 0].set_title('ΔTmin (with-map - no-map)')
axes[1, 0].set_xlim(-180, 180); axes[1, 0].set_ylim(-90, 90)
plt.colorbar(sc, ax=axes[1, 0], label='ΔTmin (K)')

axes[1, 1].scatter(hdiff, dT_max, s=0.6, alpha=0.2, label='ΔTmax')
axes[1, 1].scatter(hdiff, dT_min, s=0.6, alpha=0.2, color='C1', label='ΔTmin')
axes[1, 1].axhline(0, color='k', lw=0.4); axes[1, 1].axvline(0, color='k', lw=0.4)
axes[1, 1].set_xlabel('H - 0.06 m  (anomaly from body default)')
axes[1, 1].set_ylabel('ΔT (K)')
axes[1, 1].legend()
axes[1, 1].set_title(f'Correlation: ρ(ΔTmax, ΔH)={rho_dTmax_hdiff:+.3f},  ρ(ΔTmin)={rho_dTmin_hdiff:+.3f}')
axes[1, 1].grid(alpha=0.3)

fig.tight_layout()
out = Path('plots/property_maps/diagnose_dT_ico7.png')
fig.savefig(out, dpi=140, bbox_inches='tight')
print(f'\nWrote {out}')
