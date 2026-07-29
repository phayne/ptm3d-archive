#!/usr/bin/env python3
"""Verify a scale-corrected occluder ring for the fig8_hybrid Haworth SE-rim site
(PTM-21). Fits each candidate ring's stored elevation vs the authoritative LOLA
truth TIF (slope must be ~1.000, was 2.000 for the defective ring) and overlays
the apparent-horizon-vs-azimuth curves, which must collapse onto the OLD ring /
TIF-truth curves. Reuses the horizon geometry from audit_ring_horizon.py.

Candidates checked:
  - DEFECT   sunlit_ring_50_200_DEFECT2X.ply (or the passed defective ring)
  - BASE     rebuild from _lola_base_spstereo.tif via the FIXED build_lola_ring.py
  - HALVED   the defective ring with el/2 (provably == fixed-reader-from-IMG output,
             since sampling/triangulation are elevation-independent) -- cross-check
  - OLD      haworth_ring_32_200.ply (known-correct reference)
  - TIF      independent LOLA-truth horizon

CPU only, no sim.
"""
from __future__ import annotations
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from osgeo import gdal, osr
gdal.UseExceptions()

R = 1737400.0
SITE_LAT, SITE_LON = -88.1, 6.3
CORE = "/Users/paha3326/projects/moon3d/meshes/fig8_hybrid/sunlit_radcore_r50.ply"
OLD_RING = ("/Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/europa/heat3d/"
            "ptm/data/dem/validation/meshes/haworth_ring_32_200.ply")
TIF = "/Users/paha3326/dev/heat3d/lola_local/_lola_20m_spstereo_pm120km.tif"
OUTDIR = "/Users/paha3326/projects/moon3d/figures_ring_audit"

DEFECT = sys.argv[1] if len(sys.argv) > 1 else \
    "/Users/paha3326/projects/moon3d/meshes/fig8_hybrid/sunlit_ring_50_200_DEFECT2X.ply"
BASE = sys.argv[2] if len(sys.argv) > 2 else "/tmp/ring_base.ply"


def read_ply(path):
    with open(path, "rb") as f:
        hdr = b""
        while not hdr.endswith(b"end_header\n"):
            b = f.read(1)
            if not b:
                break
            hdr += b
        txt = hdr.decode("latin1")
        nv = int([l for l in txt.splitlines() if l.startswith("element vertex")][0].split()[-1])
        vdt = np.dtype([("x", "<f8"), ("y", "<f8"), ("z", "<f8"),
                        ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
                        ("lat", "<f8"), ("lon", "<f8"), ("el", "<f8")])
        v = np.fromfile(f, dtype=vdt, count=nv)
    return v


def latlon_to_xyz(lat_deg, lon_deg, el_m):
    la, lo = np.radians(lat_deg), np.radians(lon_deg)
    rr = R + el_m
    return np.column_stack([rr * np.cos(la) * np.cos(lo),
                            rr * np.cos(la) * np.sin(lo),
                            rr * np.sin(la)])


def geod_km(lat, lon):
    slat, slon = np.radians(SITE_LAT), np.radians(SITE_LON)
    la, lo = np.radians(lat), np.radians(lon)
    ca = np.sin(slat)*np.sin(la) + np.cos(slat)*np.cos(la)*np.cos(lo-slon)
    return R*np.arccos(np.clip(ca, -1, 1))/1e3


core = read_ply(CORE)
dcore = geod_km(core["lat"], core["lon"])
si = int(np.argmin(dcore))
S = np.array([core["x"][si], core["y"][si], core["z"][si]])
up = S/np.linalg.norm(S)
east = np.cross([0, 0, 1.0], up); east /= np.linalg.norm(east)
north = np.cross(up, east); north /= np.linalg.norm(north)


def horizon(P, dmin_km=0.0, nbins=180):
    D = P - S
    upc = D @ up; e = D @ east; n = D @ north
    horiz = np.hypot(e, n)
    elev = np.degrees(np.arctan2(upc, horiz))
    az = np.degrees(np.arctan2(e, n)) % 360.0
    if dmin_km > 0:
        k = horiz/1e3 >= dmin_km
        az, elev = az[k], elev[k]
    edges = np.linspace(0, 360, nbins+1)
    idx = np.clip(np.digitize(az, edges)-1, 0, nbins-1)
    h = np.full(nbins, np.nan)
    for b in range(nbins):
        m = idx == b
        if m.any():
            h[b] = elev[m].max()
    return 0.5*(edges[:-1]+edges[1:]), h


# TIF truth setup
ds = gdal.Open(TIF); gt = ds.GetGeoTransform(); band = ds.GetRasterBand(1)
arr = band.ReadAsArray().astype(np.float64); nd = band.GetNoDataValue()
XS, YS = ds.RasterXSize, ds.RasterYSize
srs = osr.SpatialReference(); srs.ImportFromWkt(ds.GetProjection()); geo = srs.CloneGeogCS()
try:
    srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    geo.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
except Exception:
    pass
fwd = osr.CoordinateTransformation(geo, srs)
inv = osr.CoordinateTransformation(srs, geo)


def fit_vs_truth(v, name):
    pts = np.asarray(fwd.TransformPoints(np.column_stack([v["lon"], v["lat"]]).tolist()))
    col = (pts[:, 0]-gt[0])/gt[1]; row = (pts[:, 1]-gt[3])/gt[5]
    inb = (col >= 0) & (col < XS-1) & (row >= 0) & (row < YS-1)
    ci = np.clip(col.astype(int), 0, XS-1); ri = np.clip(row.astype(int), 0, YS-1)
    te = arr[ri, ci]
    good = inb & np.isfinite(te) & (te > -1e30) & (te != (nd if nd is not None else -9999))
    tt, ee = te[good], v["el"][good]
    A = np.vstack([tt, np.ones_like(tt)]).T
    a, b = np.linalg.lstsq(A, ee, rcond=None)[0]
    print(f"  {name:34s} slope {a:.4f}  intercept {b:7.1f} m   (N={good.sum()}, "
          f"resid std {np.std(ee-tt):.1f} m)")
    return a


# candidates
defect = read_ply(DEFECT)
base = read_ply(BASE)
halved = defect.copy(); halved["el"] = defect["el"]/2.0   # cross-check (fixed-reader-equivalent)
old = read_ply(OLD_RING)

print("=== elevation-scale fit vs LOLA truth (slope target: corrected ~1.000) ===")
sd = fit_vs_truth(defect, "DEFECT (as-built)")
sb = fit_vs_truth(base, "BASE rebuild (_lola_base 80m)")
sh = fit_vs_truth(halved, "HALVED el/2 (fixed-reader eqv)")
so = fit_vs_truth(old, "OLD haworth_ring_32_200")

# horizons
def P(v): return np.column_stack([latlon_to_xyz(v["lat"], v["lon"], v["el"])])
az, h_def = horizon(P(defect))
_, h_base = horizon(P(base))
_, h_half = horizon(P(halved))
Pold = np.column_stack([old["x"], old["y"], old["z"]])
_, h_old = horizon(Pold)
# range-matched truth: the corrected ring is a ~46-200 km annulus, so compare its
# horizon to OLD ring and TIF truth restricted to the SAME radial band (beyond 46 km).
_, h_old_far = horizon(Pold, dmin_km=46.0)
# TIF-truth horizon beyond 15 km
step = 3
rr, cc = np.mgrid[0:YS:step, 0:XS:step]
el = arr[rr, cc].ravel()
Xp = gt[0]+(cc.ravel()+0.5)*gt[1]; Yp = gt[3]+(rr.ravel()+0.5)*gt[5]
gmask = np.isfinite(el) & (el > -1e30) & (el != (nd if nd is not None else -9999))
ll = np.asarray(inv.TransformPoints(np.column_stack([Xp[gmask], Yp[gmask]]).tolist()))
Ptif_all = latlon_to_xyz(ll[:, 1], ll[:, 0], el[gmask])
_, h_tif = horizon(Ptif_all, dmin_km=15.0)
_, h_tif_far = horizon(Ptif_all, dmin_km=46.0)   # range-matched to the ring band

# range-matched collapse check: corrected ring vs truth over the SAME 46+ km band
def agree(hc, ht, tag):
    both = np.isfinite(hc) & np.isfinite(ht)
    d = (hc - ht)[both]
    print(f"  {tag:40s} mean {d.mean():+.3f}  median {np.median(d):+.3f}  "
          f"mean|d| {np.abs(d).mean():.3f}  max|d| {np.abs(d).max():.3f} deg (N={both.sum()})")

os.makedirs(OUTDIR, exist_ok=True)
fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 8.5), sharex=True)
a1.plot(az, h_def, color="#c0392b", lw=1.6, label=f"DEFECT ring (slope {sd:.3f}x truth)")
a1.plot(az, h_base, color="#16a085", lw=1.5, label=f"CORRECTED rebuild _lola_base 80m (slope {sb:.3f})")
a1.plot(az, h_half, color="#e67e22", lw=1.1, ls="--", label=f"CORRECTED el/2 (fixed-reader eqv, slope {sh:.3f})")
a1.plot(az, h_old, color="#2471a3", lw=1.2, label="OLD haworth_ring_32_200 (= truth)")
a1.plot(az, h_tif, color="#27ae60", lw=0.9, ls=":", label="LOLA truth TIF (beyond 15 km)")
a1.axhline(0, color="k", lw=0.5); a1.set_ylabel("apparent horizon elevation (deg)")
a1.set_title("Corrected ring verification -- horizon collapses onto OLD ring / TIF truth")
a1.legend(fontsize=8, loc="upper right"); a1.grid(alpha=0.3)

a2.plot(az, h_def - h_half, color="#c0392b", lw=1.2, label="DEFECT - corrected(el/2)")
a2.plot(az, h_base - h_half, color="#16a085", lw=1.2, label="BASE-rebuild - corrected(el/2)")
a2.axhline(0, color="k", lw=0.5); a2.set_ylabel("horizon diff (deg)")
a2.set_xlabel("azimuth (deg)"); a2.set_xlim(0, 360); a2.grid(alpha=0.3)
a2.legend(fontsize=8, loc="upper right")
resid = (h_base - h_half); resid = resid[np.isfinite(resid)]
a2.set_title(f"DEFECT excess removed; BASE-rebuild vs el/2 agree to "
             f"mean|d| {np.abs(resid).mean():.3f} deg (resampling/res confound)")
fig.tight_layout()
out = os.path.join(OUTDIR, "ring_corrected_verification.png")
fig.savefig(out, dpi=150); print("\nwrote", out)

def summ(h):
    v = h[np.isfinite(h)]; return v.mean(), np.median(v), v.max()
print("\n=== horizon summary (deg) ===")
for nm, h in [("DEFECT", h_def), ("CORRECTED base-rebuild", h_base),
              ("CORRECTED el/2", h_half), ("OLD ring (truth)", h_old),
              ("TIF truth", h_tif)]:
    m, md, mx = summ(h); print(f"  {nm:26s} mean {m:6.3f}  median {md:6.3f}  max {mx:6.3f}")
d = h_def - h_half; d = d[np.isfinite(d)]
print(f"\nDEFECT over-occlusion removed: mean {d.mean():+.3f}  median {np.median(d):+.3f}  max {d.max():+.3f} deg")

print("\n=== range-matched COLLAPSE check (corrected ring vs truth, both beyond 46 km) ===")
agree(h_base, h_old_far, "BASE-rebuild vs OLD ring (46+ km)")
agree(h_base, h_tif_far, "BASE-rebuild vs TIF truth (46-120 km)")
agree(h_half, h_old_far, "el/2       vs OLD ring (46+ km)")
print("  (defect for contrast:)")
agree(h_def, h_old_far, "DEFECT     vs OLD ring (46+ km)")
