#!/usr/bin/env python3
"""Ring/core horizon-registration audit for the fig8_hybrid Haworth SE-rim site.

Motivation: a 2-timestep ring-swap test showed the NEW ring
(sunlit_ring_50_200.ply, built from LDEM_80S_20M.IMG via build_lola_ring.py)
over-occludes grazing sun by ~2x vs the OLD ring (haworth_ring_32_200.ply). This
script characterizes the apparent-horizon elevation angle vs azimuth contributed
by each occluder as seen from the site center, plus an independent LOLA-truth
horizon from the projected Float32 GeoTIFF, and reports the ring-vs-truth
elevation ratio that drives the horizon difference.

Verdict (see the printed table): the NEW ring stores elevations exactly 2.0x the
LOLA truth -- an unapplied PDS SCALING_FACTOR=0.5 on LDEM_80S_20M.IMG. All
elevations enter build_lola_ring.py at line ~59 (band.ReadAsArray, raw DN) and
line ~92 (rr = R + el) with no GetScale()/GetOffset() applied.

CPU only, no sim. Outputs figure + prints a numbers table.
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from osgeo import gdal, osr
gdal.UseExceptions()

R = 1737400.0  # Moon 2015 sphere radius (m); matches build_lola_ring.py + LOLA CRS
SITE_LAT, SITE_LON = -88.1, 6.3  # fig8_hybrid mesh build center

CORE = "/Users/paha3326/projects/moon3d/meshes/fig8_hybrid/sunlit_radcore_r50.ply"
NEW_RING = "/Users/paha3326/projects/moon3d/meshes/fig8_hybrid/sunlit_ring_50_200.ply"
OLD_RING = ("/Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/europa/heat3d/"
            "ptm/data/dem/validation/meshes/haworth_ring_32_200.ply")
TIF = "/Users/paha3326/dev/heat3d/lola_local/_lola_20m_spstereo_pm120km.tif"
OUTDIR = "/Users/paha3326/projects/moon3d/figures_ring_audit"


def read_ply_xyz(path):
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


def geodesic_km(lat_deg, lon_deg):
    slat, slon = np.radians(SITE_LAT), np.radians(SITE_LON)
    la, lo = np.radians(lat_deg), np.radians(lon_deg)
    ca = np.sin(slat) * np.sin(la) + np.cos(slat) * np.cos(la) * np.cos(lo - slon)
    return R * np.arccos(np.clip(ca, -1.0, 1.0)) / 1e3


# --- observer position: core-mesh vertex nearest the build center ---
core = read_ply_xyz(CORE)
dcore = geodesic_km(core["lat"], core["lon"])
si = int(np.argmin(dcore))
S = np.array([core["x"][si], core["y"][si], core["z"][si]])
up = S / np.linalg.norm(S)
east = np.cross(np.array([0.0, 0.0, 1.0]), up); east /= np.linalg.norm(east)
north = np.cross(up, east); north /= np.linalg.norm(north)
print(f"Observer: core vertex #{si} at lat {core['lat'][si]:.4f} lon {core['lon'][si]:.4f} "
      f"el {core['el'][si]:.1f} m  ({dcore[si]*1e3:.0f} m from build center)")


def horizon(P, dmin_km=0.0, nbins=180):
    """Max elevation angle (deg) per azimuth bin, from S, for points P (Nx3)."""
    D = P - S
    upc = D @ up
    e = D @ east
    n = D @ north
    horiz = np.hypot(e, n)
    elev = np.degrees(np.arctan2(upc, horiz))
    az = (np.degrees(np.arctan2(e, n))) % 360.0
    if dmin_km > 0:
        keep = (horiz / 1e3) >= dmin_km
        az, elev = az[keep], elev[keep]
    edges = np.linspace(0, 360, nbins + 1)
    idx = np.clip(np.digitize(az, edges) - 1, 0, nbins - 1)
    hor = np.full(nbins, np.nan)
    for b in range(nbins):
        m = idx == b
        if m.any():
            hor[b] = elev[m].max()
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, hor


# --- rings ---
newr = read_ply_xyz(NEW_RING)
oldr = read_ply_xyz(OLD_RING)
Pnew = np.column_stack([newr["x"], newr["y"], newr["z"]])
Pold = np.column_stack([oldr["x"], oldr["y"], oldr["z"]])
# NEW ring with the PDS scale bug CORRECTED (el/2), same vertices/density/radius:
# isolates the pure 2x-scaling effect from density/inner-radius confounds.
Pnew_fix = latlon_to_xyz(newr["lat"], newr["lon"], newr["el"] / 2.0)
Pcore_far = np.column_stack([core["x"], core["y"], core["z"]])[dcore > 15.0]

az_c, h_new = horizon(Pnew)
_,    h_newfix = horizon(Pnew_fix)
_,    h_old = horizon(Pold)
_,    h_core = horizon(Pcore_far)

# --- independent LOLA-truth horizon from the projected Float32 TIF ---
ds = gdal.Open(TIF)
gt = ds.GetGeoTransform()
band = ds.GetRasterBand(1)
nd = band.GetNoDataValue()
arr = band.ReadAsArray().astype(np.float64)
XS, YS = ds.RasterXSize, ds.RasterYSize
srs = osr.SpatialReference(); srs.ImportFromWkt(ds.GetProjection())
geo = srs.CloneGeogCS()
try:
    srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    geo.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
except Exception:
    pass
inv = osr.CoordinateTransformation(srs, geo)  # (x,y)->(lon,lat)
step = 3  # subsample the 12000x12000 grid for the horizon
rr, cc = np.mgrid[0:YS:step, 0:XS:step]
el = arr[rr, cc].ravel()
Xp = gt[0] + (cc.ravel() + 0.5) * gt[1]
Yp = gt[3] + (rr.ravel() + 0.5) * gt[5]
good = np.isfinite(el) & (el > -1e30) & (el != (nd if nd is not None else -9999))
Xp, Yp, el = Xp[good], Yp[good], el[good]
ll = np.asarray(inv.TransformPoints(np.column_stack([Xp, Yp]).tolist()))
lon_t, lat_t = ll[:, 0], ll[:, 1]
Ptif = latlon_to_xyz(lat_t, lon_t, el)
_, h_tif = horizon(Ptif, dmin_km=15.0)

# --- figure ---
import os
os.makedirs(OUTDIR, exist_ok=True)
fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 8.5), sharex=True)
a1.plot(az_c, h_new, color="#c0392b", lw=1.6, label="NEW ring RAW (el=2x truth, as built)")
a1.plot(az_c, h_newfix, color="#e67e22", lw=1.3, ls="--", label="NEW ring CORRECTED (el/2, same mesh)")
a1.plot(az_c, h_old, color="#2471a3", lw=1.4, label="OLD ring haworth_ring_32_200 (= TIF truth)")
a1.plot(az_c, h_core, color="#7f8c8d", lw=0.9, ls="--", label="core mesh, beyond 15 km")
a1.plot(az_c, h_tif, color="#27ae60", lw=0.9, ls=":", label="LOLA truth (pm120km TIF, beyond 15 km)")
a1.axhline(0, color="k", lw=0.5)
a1.set_ylabel("apparent horizon elevation (deg)")
a1.set_title(f"Occluder horizon from Haworth SE-rim site ({SITE_LAT}, {SITE_LON}) — "
             f"NEW ring elevations are 2.0x LOLA truth (PDS SCALING_FACTOR=0.5 unapplied)")
a1.legend(fontsize=8, loc="upper right")
a1.grid(alpha=0.3)

# pure scaling effect on the SAME mesh (density/radius held fixed):
diff = h_new - h_newfix
a2.plot(az_c, diff, color="#8e44ad", lw=1.4)
a2.axhline(0, color="k", lw=0.5)
a2.set_ylabel("RAW - CORRECTED horizon (deg)")
a2.set_xlabel("azimuth (deg, local ENU from site)")
pos = diff[np.isfinite(diff) & (h_new > 0)]
a2.set_title(f"Pure 2x-scaling over-occlusion (same NEW mesh, el/2): "
             f"median-where-lit {np.median(pos):+.3f} deg, max {np.nanmax(diff):+.3f} deg")
a2.grid(alpha=0.3)
a2.set_xlim(0, 360)
fig.tight_layout()
out = os.path.join(OUTDIR, "ring_horizon_overlay.png")
fig.savefig(out, dpi=150)
print("wrote", out)

# --- numbers table ---
def summ(h):
    v = h[np.isfinite(h)]
    return v.mean(), np.median(v), v.max()
print("\n=== HORIZON SUMMARY (elevation angle deg, over azimuth bins) ===")
print(f"{'source':42s} {'mean':>8s} {'median':>8s} {'max':>8s}")
for nm, h in [("NEW ring RAW (el=2x truth)", h_new),
              ("NEW ring CORRECTED (el/2, same mesh)", h_newfix),
              ("OLD ring (haworth_ring_32_200 = truth)", h_old),
              ("core mesh beyond 15 km", h_core),
              ("LOLA truth TIF beyond 15 km", h_tif)]:
    m, md, mx = summ(h)
    print(f"{nm:42s} {m:8.3f} {md:8.3f} {mx:8.3f}")
# where the RAW new ring shadows (horizon>0), how much did the 2x add?
mask = np.isfinite(diff) & (h_new > 0)
print(f"\nPure 2x-scaling over-occlusion (RAW - CORRECTED, same NEW mesh):")
print(f"  over lit-relevant bins (RAW horizon>0, N={mask.sum()}): "
      f"mean {diff[mask].mean():+.3f}  median {np.median(diff[mask]):+.3f}  max {diff[mask].max():+.3f} deg")
# agreement of corrected new ring with old ring / truth (confound check)
both = np.isfinite(h_newfix) & np.isfinite(h_old)
print(f"  CORRECTED-new vs OLD ring horizon: mean|diff| {np.abs(h_newfix-h_old)[both].mean():.3f} deg "
      f"(RAW-new vs OLD: {np.abs(h_new-h_old)[both].mean():.3f} deg)")
