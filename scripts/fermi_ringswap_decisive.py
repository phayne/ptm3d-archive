#!/usr/bin/env python3
"""Three-way ring-swap decisive test (2026-07-22, Sagan's mandate, Fermi-specified).

Extends /tmp/fermi_decisive.py's matched-cell methodology (same window, same
250 m cells, same "populated by BOTH meshes" mask, same sunlit/deep/PSR
Diviner-summer-max strata) to compare, AT THE SAME PHYSICAL INSTANT
(2026-10-28T04:55:52, the timestep of peak area-weighted direct_solar over the
hybrid mesh's own matched-sunlit population across its full 365-step run):

  (a) hybrid ORIGINAL ring (sunlit_ring_50_200.ply)   -- single-instant frame
  (b) hybrid RING-SWAPPED to r32c's ring (haworth_ring_32_200.ply) -- single-instant frame
  (c) r32c ORIGINAL                                    -- statistics/direct_solar_max.bin
                                                           (temporal max, NOT the same
                                                           instant -- kept for continuity
                                                           with Fermi's original number,
                                                           explicitly flagged as such)

(a) vs (b) is the clean, apples-to-apples half of this test (identical core mesh,
identical instant, ONLY the ring differs). (c) is included for continuity with
Fermi's original ratio but is NOT on the same footing (r32c has no per-step
frames saved, so only its full-run temporal max is available).
"""
import sys; sys.path.insert(0, "/Users/paha3326/Projects/ptm/src/python")
import numpy as np
from pathlib import Path
import diviner_seasonal_compare as D

la, lo = -88.181, 6.041
x0, y0 = D.TO_XY.transform(lo, la)
ex = np.arange(x0-D.WINDOW_KM*1e3, x0+D.WINDOW_KM*1e3+D.CELL_M, D.CELL_M)
ey = np.arange(y0-D.WINDOW_KM*1e3, y0+D.WINDOW_KM*1e3+D.CELL_M, D.CELL_M)
nx, ny = len(ex)-1, len(ey)-1


def cellize_mesh(run):
    """Per-cell area-weighted facet slope + population mask (mesh geometry only)."""
    v, f, flat, flon = D.read_ply(str(run / "mesh.ply"))
    area, flat, flon = D.face_geometry(v, f, flat, flon)
    p0, p1, p2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    n = np.cross(p1 - p0, p2 - p0); n /= np.linalg.norm(n, axis=1, keepdims=True)
    cen = (p0 + p1 + p2) / 3.0; rad = cen / np.linalg.norm(cen, axis=1, keepdims=True)
    slope = np.degrees(np.arccos(np.clip(np.abs((n * rad).sum(1)), 0, 1)))
    fx, fy = D.TO_XY.transform(flon, flat); rkm = np.hypot(fx - x0, fy - y0) / 1e3
    ix = np.floor((fx - ex[0]) / D.CELL_M).astype(int)
    iy = np.floor((fy - ey[0]) / D.CELL_M).astype(int)
    keep = (rkm <= D.CORE_KM) & (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    lin = iy[keep] * nx + ix[keep]
    A = area[keep]
    Asum = np.bincount(lin, weights=A, minlength=nx * ny)
    slp = np.bincount(lin, weights=A * slope[keep], minlength=nx * ny)
    with np.errstate(invalid="ignore", divide="ignore"):
        S = np.where(Asum > 0, slp / Asum, np.nan)
    return dict(keep=keep, lin=lin, A=A, Asum=Asum, pop=Asum > 0, slope=S)


def cell_field(geo, values_on_kept_faces):
    num = np.bincount(geo["lin"], weights=geo["A"] * values_on_kept_faces.astype(np.float64),
                       minlength=nx * ny)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(geo["Asum"] > 0, num / geo["Asum"], np.nan)


runc = Path("/Users/paha3326/projects/moon3d/outputs/haworth_production/coupled")
runh = Path("/Users/paha3326/projects/moon3d/outputs/fig8_hybrid_production/coupled")
runs = Path("/Users/paha3326/projects/moon3d/outputs/fig8_hybrid_production/diag_ringswap_r32c")

geo_c = cellize_mesh(runc)
geo_h = cellize_mesh(runh)   # same core mesh as geo_s (ring-swap uses the SAME core)

# (c) r32c: statistics/direct_solar_max.bin (temporal max, whole run)
ds_c = np.fromfile(str(runc / "statistics" / "direct_solar_max.bin"), dtype=np.float32)[1:][geo_c["keep"]]
Q_c = cell_field(geo_c, ds_c)

# (a) hybrid original: SINGLE-INSTANT frame at step 300 (2026-10-28T04:55:52.865)
ds_h_orig = np.fromfile(str(runh / "direct_solar" / "direct_solar_0300.bin"), dtype=np.float32)[1:][geo_h["keep"]]
Q_h_orig = cell_field(geo_h, ds_h_orig)

# (b) hybrid ring-swapped: SINGLE-INSTANT frame at step 0 (2026-10-28T04:55:52.280)
ds_h_swap = np.fromfile(str(runs / "direct_solar" / "direct_solar_0000.bin"), dtype=np.float32)[1:][geo_h["keep"]]
Q_h_swap = cell_field(geo_h, ds_h_swap)

ds_summer = D.diviner_seasonal(D.SEASDIR / "polar_south_80_summer_max_polar.npy",
                                dict(ex=ex, ey=ey, nx=nx, ny=ny)).ravel()
both = geo_c["pop"] & geo_h["pop"] & np.isfinite(ds_summer)
print(f"cells populated by BOTH meshes: {both.sum()}")

for lbl, msk0 in [("sunlit Wsum>=200", ds_summer >= 200),
                  ("deep Wsum<110", ds_summer < 110),
                  ("all PSR<200", ds_summer < 200)]:
    m = both & msk0 & np.isfinite(Q_c) & np.isfinite(Q_h_orig) & np.isfinite(Q_h_swap)
    print(f"\n {lbl}  (N={m.sum()} matched cells)")
    qc, qo, qs = Q_c[m].mean(), Q_h_orig[m].mean(), Q_h_swap[m].mean()
    print(f"   r32c (temporal-max, full run)      : {qc:7.1f} W/m2")
    print(f"   hybrid ORIGINAL ring (at instant)   : {qo:7.1f} W/m2   ratio vs r32c={qo/max(qc,1e-9):.2f}")
    print(f"   hybrid R32C-RING-SWAP (same instant): {qs:7.1f} W/m2   ratio vs r32c={qs/max(qc,1e-9):.2f}")
    print(f"   swap/original ratio (clean, same core+instant): {qs/max(qo,1e-9):.2f}")
    print(f"   mean facet slope: r32c={geo_c['slope'][m].mean():.2f} hybrid={geo_h['slope'][m].mean():.2f} deg")
