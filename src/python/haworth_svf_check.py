#!/usr/bin/env python3
"""Model-independent geometric ground truth for the Haworth floor->wall view
factor, computed directly from the LOLA DEM via a horizon scan.

For a floor point, in each azimuth march outward over the DEM and find the
horizon elevation angle (highest terrain). Terrain fills elevation [0, horizon]
in that azimuth, so the cos-weighted view factor to terrain is
   VF_terrain = (1/2pi) integral sin^2(horizon(az)) d az
This is the TRUE geometric Sum_j F_ij (the radiosity row sum) for that point,
independent of the model's H-matrix. Compare to the model's dumped row sums.
"""
from __future__ import annotations
import numpy as np
import rasterio
import pyproj

DEM = "data/dem/validation/_lola_20m_spstereo_pm120km.tif"
LAT, LON = -87.45, -2.1          # Haworth center
BODY_R = 1737400.0


def main():
    src = rasterio.open(DEM)
    elev = src.read(1).astype(np.float64) * 0.5   # DN*0.5 = height above sphere (m)
    T = src.transform
    px = T[0]                                       # pixel size (m), polar stereo ~ true near pole
    # project site to DEM CRS
    to_dem = pyproj.Transformer.from_crs(
        pyproj.CRS.from_proj4(f"+proj=longlat +R={BODY_R} +no_defs"), src.crs, always_xy=True)
    x0, y0 = to_dem.transform(LON, LAT)
    inv = ~T
    col0, row0 = inv * (x0, y0)
    H, W = elev.shape
    print(f"DEM {W}x{H}, pixel {px:.1f} m; Haworth center at px ({col0:.0f},{row0:.0f})")

    n_az = 180
    azs = np.linspace(0, 2*np.pi, n_az, endpoint=False)
    max_km = 30.0
    n_r = int(max_km*1000/px)
    radii = (np.arange(1, n_r) * px)               # m

    # sample a set of floor-center points within ~2 km of center
    rng_off = np.linspace(-1500, 1500, 7)
    pts = [(x0+dx, y0+dy) for dx in rng_off for dy in rng_off]
    svf_terrain = []
    horizons = []
    for (xp, yp) in pts:
        cc, rr = inv * (xp, yp)
        cc, rr = int(round(cc)), int(round(rr))
        if not (0 <= rr < H and 0 <= cc < W):
            continue
        z0 = elev[rr, cc]
        hor = np.zeros(n_az)
        for ai, az in enumerate(azs):
            dx, dy = np.sin(az), np.cos(az)         # az from north
            cs = cc + (radii/px)*dx
            rs = rr - (radii/px)*dy
            valid = (cs >= 0) & (cs < W) & (rs >= 0) & (rs < H)
            cs, rs, dd = cs[valid].astype(int), rs[valid].astype(int), radii[:valid.sum()]
            if len(dd) == 0:
                continue
            zz = elev[rs, cs]
            ang = np.arctan2(zz - z0, dd)            # elevation angle to each cell
            hor[ai] = max(0.0, ang.max())
        svf_terrain.append(np.mean(np.sin(hor)**2))  # cos-weighted VF to terrain
        horizons.append(np.degrees(np.mean(hor)))
    svf_terrain = np.array(svf_terrain)
    print(f"\nHaworth floor-center ({len(svf_terrain)} pts within +/-1.5 km):")
    print(f"  mean horizon elevation = {np.mean(horizons):.1f} deg")
    print(f"  GEOMETRIC terrain view factor (DEM horizon) = {svf_terrain.mean():.4f} "
          f"(range {svf_terrain.min():.4f}-{svf_terrain.max():.4f})")
    print(f"  -> compare to model H-matrix floor-center row-sum 0.0132")
    print(f"  crude sin^2(rim) analytic was 0.036 (assumed filled band to rim crest)")


if __name__ == "__main__":
    main()
