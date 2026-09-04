#!/usr/bin/env python3
"""Measure whether a flux-cube lattice can sample a GENUINE PSR at a site.

A permanently shadowed region is dark in ALL seasons. A cube that is dark for
one evaluated window is "window-dark", which is a weaker claim and a different
question -- see docs/flux-cubes.md section 10. Window-darkness is a property of
the product; PSR-ness is a property of the SITE, and it is decided by the
site's psr_mask, not by any single window.

This script answers the site question directly, so the distinction is measured
rather than asserted:

  * how many PSR pixels fall inside the cube grid, and how near the site;
  * how they are grouped into patches, and how big the largest one is;
  * how many of the production lattice points actually land on one;
  * where a targeted dense array should go, and how many of its cubes would
    land on PSR at various spacings.

TWO THINGS THAT WILL GIVE WRONG ANSWERS IF SKIPPED
--------------------------------------------------
1. FRAME ROTATION. In a south-polar stereographic projection with lon_0 = 0,
   the site's local NORTH is the radial-outward direction, not projected +Y,
   and local EAST is perpendicular to it. The rotation equals the site
   longitude (30 deg at a polar site). Treating projected axes as ENU moves the
   patch centroid by tens of meters and changes which lattice points look like
   they are on PSR -- it changed the production-lattice hit count from 1 to 0
   when this was corrected.
2. MASK SEMANTICS. The COG psr_mask is 0/1 at ~99.87% of valid pixels, but the
   remaining ~0.13% carry intermediate values from the COG's own linear
   interpolation at patch boundaries. Counting only `== 1` undercounts;
   counting `>= 0.5` includes partial boundary pixels. Both are reported, so
   the number quoted downstream is bracketed rather than picked.

Usage:
  python3 scripts/flux_cube_psr_probe.py --mask <site>/psr_mask.tif \\
      --lat -85.0 --lon 30.0 --radius 500 --spacing 25
"""

import argparse
import math
import sys

try:
    import numpy as np
    import rasterio
    from scipy import ndimage
    from pyproj import CRS, Transformer
except ImportError as e:
    print("needs numpy, rasterio, scipy and pyproj (available in the europa_viz env): %s" % e)
    sys.exit(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask", required=True, help="psr_mask.tif from the site's COG set")
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--radius", type=float, default=500.0)
    ap.add_argument("--spacing", type=float, default=25.0)
    a = ap.parse_args()

    with rasterio.open(a.mask) as ds:
        tr = ds.transform
        nod = ds.nodata
        arr = ds.read(1)
        crs = CRS.from_wkt(ds.crs.to_wkt())
        fwd = Transformer.from_crs(crs.geodetic_crs, crs, always_xy=True)
        inv = Transformer.from_crs(crs, crs.geodetic_crs, always_xy=True)
        sx, sy = fwd.transform(a.lon, a.lat)

        valid = arr[arr != nod]
        n_one = int((valid == 1).sum())
        n_zero = int((valid == 0).sum())
        n_mid = int(((valid != 0) & (valid != 1)).sum())
        print("mask: %s" % a.mask)
        print("  valid px %d;  ==1: %d;  ==0: %d;  intermediate: %d (%.2f%% -- COG interpolation "
              "at patch boundaries)" % (valid.size, n_one, n_zero, n_mid, 100.0 * n_mid / valid.size))
        print("  pixel size %.2f x %.2f m" % (abs(tr.a), abs(tr.e)))
        print("  site (%.4f, %.4f) -> projected (%.2f, %.2f)" % (a.lat, a.lon, sx, sy))

        ys, xs = np.mgrid[0:ds.height, 0:ds.width]
        X = tr.c + tr.a * (xs + 0.5)
        Y = tr.f + tr.e * (ys + 0.5)
        R = np.hypot(X - sx, Y - sy)

        # Local ENU in projected axes: north = radial outward, angle = longitude.
        lam = math.radians(a.lon)
        north = np.array([math.sin(lam), math.cos(lam)])
        east = np.array([math.cos(lam), -math.sin(lam)])
        print("  local north in projected axes (%.4f, %.4f); rotation = site longitude %.2f deg"
              % (north[0], north[1], a.lon))

        px_area = abs(tr.a * tr.e)
        for label, m in (("strict  (== 1)", arr == 1),
                         ("inclusive (>= 0.5)", (arr >= 0.5) & (arr != nod))):
            sel = m & (R <= a.radius)
            near = R[sel].min() if sel.any() else float("nan")
            print("\n%s: %d px inside %.0f m (%.0f m^2), nearest %.1f m"
                  % (label, int(sel.sum()), a.radius, sel.sum() * px_area, near))

        m = (arr >= 0.5) & (arr != nod)
        sel = m & (R <= a.radius)
        if not sel.any():
            print("\nNo PSR inside the grid: a targeted array is not possible at this radius.")
            return 0
        lab, n = ndimage.label(sel)
        sizes = ndimage.sum(sel, lab, range(1, n + 1))
        order = np.argsort(sizes)[::-1]
        grid_area = math.pi * a.radius ** 2
        print("\n%d distinct PSR patches inside the grid; areal fraction %.4f%% of %.3f km^2"
              % (n, 100.0 * sel.sum() * px_area / grid_area, grid_area / 1e6))
        print("%4s %9s %11s %11s %8s %13s %13s" %
              ("rank", "area_m2", "site_E_m", "site_N_m", "r_m", "lat_deg", "lon_deg"))
        for rank, k0 in enumerate(order[:6], 1):
            k = k0 + 1
            yy, xx = np.where(lab == k)
            pxc, pyc = X[yy, xx].mean(), Y[yy, xx].mean()
            d = np.array([pxc - sx, pyc - sy])
            e, nn = float(d @ east), float(d @ north)
            lo, la = inv.transform(pxc, pyc)
            print("%4d %9d %11.1f %11.1f %8.1f %13.6f %13.6f"
                  % (rank, int(sizes[k0]), e, nn, math.hypot(e, nn), la, lo))

        def on_psr(e, nn):
            q = np.array([sx, sy]) + east * e + north * nn
            ci = int((q[0] - tr.c) / tr.a)
            cj = int((q[1] - tr.f) / tr.e)
            return 0 <= cj < ds.height and 0 <= ci < ds.width and bool(m[cj, ci])

        nl = int(a.radius / a.spacing)
        pts = [(i * a.spacing, j * a.spacing) for i in range(-nl, nl + 1) for j in range(-nl, nl + 1)
               if (i * a.spacing) ** 2 + (j * a.spacing) ** 2 <= a.radius ** 2]
        hits = sum(1 for (e, nn) in pts if on_psr(e, nn))
        expect = len(pts) * sel.sum() * px_area / grid_area
        print("\nproduction lattice (%.0f m spacing): %d cubes, %d on PSR (%.2f expected by area)"
              % (a.spacing, len(pts), hits, expect))
        if hits == 0:
            print("  => this lattice CANNOT sample a genuine PSR here. The patches are far smaller")
            print("     than the spacing; a finer GLOBAL grid is the wrong fix, a targeted array")
            print("     on a named patch is the right one.")

        k = order[0] + 1
        yy, xx = np.where(lab == k)
        pxc, pyc = X[yy, xx].mean(), Y[yy, xx].mean()
        d = np.array([pxc - sx, pyc - sy])
        pe, pn = float(d @ east), float(d @ north)
        lo, la = inv.transform(pxc, pyc)
        print("\ntargeted array on the largest patch (lat %.6f, lon %.6f):" % (la, lo))
        print("%9s %8s %8s %10s" % ("spacing", "radius", "cubes", "on PSR"))
        for sp, rad in ((5.0, 20.0), (2.0, 12.0), (2.0, 20.0), (1.0, 10.0)):
            nl2 = int(rad / sp)
            p2 = [(i * sp, j * sp) for i in range(-nl2, nl2 + 1) for j in range(-nl2, nl2 + 1)
                  if (i * sp) ** 2 + (j * sp) ** 2 <= rad ** 2]
            on = sum(1 for (e, nn) in p2 if on_psr(pe + e, pn + nn))
            print("%9.1f %8.0f %8d %6d (%2.0f%%)" % (sp, rad, len(p2), on, 100.0 * on / max(1, len(p2))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
