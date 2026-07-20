#!/usr/bin/env python3
"""Diviner max bolometric T (left) + co-registered LOLA topography (right) over
a LARGE ROI around each validation site, to verify the model tiles sit on the
intended features and that the Diviner<->LOLA mapping is correct.

Both panels are sampled on the SAME lat/lon grid (built in the Diviner
azimuthal-equidistant frame), so a feature at the same pixel in both panels is
the same physical place. The +/-5 km comparison window is outlined.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
import rasterio
import pyproj
from scipy.ndimage import map_coordinates

sys.path.insert(0, str(Path(__file__).parent))
from ptm_pubstyle import apply_style
from diviner_data import load_diviner, latlon_to_diviner_xy, diviner_xy_to_latlon, KM_PER_DEG

SITES = [("Haworth", -87.45, -2.1), ("Faustini", -87.3, 84.0),
         ("Shackleton", -89.66, 129.2), ("Haworth SE rim", -88.181, 6.041)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--diviner", required=True)
    ap.add_argument("--lola", default="data/dem/validation/_lola_20m_spstereo_pm120km.tif")
    ap.add_argument("--roi-km", type=float, default=30.0)
    ap.add_argument("--win-km", type=float, default=5.0, help="comparison-window radius to outline")
    ap.add_argument("--out", default="docs/figures/diviner_validation/site_overview.png")
    a = ap.parse_args()

    g = load_diviner(a.diviner)
    src = rasterio.open(a.lola)
    to_lola = pyproj.Transformer.from_crs(
        pyproj.CRS.from_proj4("+proj=longlat +R=1737400 +no_defs"), src.crs, always_xy=True)
    inv_t = ~src.transform
    elev = src.read(1)
    ls = LightSource(azdeg=315, altdeg=30)

    W = a.roi_km / KM_PER_DEG               # half-width in degrees
    step = 0.2 / KM_PER_DEG                 # ~200 m grid
    apply_style()
    fig, axes = plt.subplots(len(SITES), 2, figsize=(7.0, 3.1 * len(SITES)))

    for row, (name, lat, lon) in enumerate(SITES):
        xd0, yd0 = latlon_to_diviner_xy(lat, lon)
        gx = np.arange(xd0 - W, xd0 + W, step)
        gy = np.arange(yd0 - W, yd0 + W, step)
        GX, GY = np.meshgrid(gx, gy)
        # Diviner Tmax: nearest pixel in its own frame
        ix = np.round((GX - g.x_deg[0]) / (g.x_deg[1] - g.x_deg[0])).astype(int)
        iy = np.round((GY - g.y_deg[0]) / (g.y_deg[1] - g.y_deg[0])).astype(int)
        okp = (ix >= 0) & (ix < len(g.x_deg)) & (iy >= 0) & (iy < len(g.y_deg))
        DT = np.full(GX.shape, np.nan)
        DT[okp] = g.t_max[iy[okp], ix[okp]]
        # LOLA elev at the SAME lat/lon as each grid point
        glat, glon = diviner_xy_to_latlon(GX, GY)
        lx, ly = to_lola.transform(glon, glat)
        col, rrow = inv_t * (lx, ly)
        EL = map_coordinates(elev, [rrow.ravel(), col.ravel()], order=1,
                             mode="constant", cval=np.nan).reshape(GX.shape)
        ext = [-a.roi_km, a.roi_km, -a.roi_km, a.roi_km]   # km relative to site

        axL = axes[row, 0]
        # per-site robust scale (2-98th pct) to enhance contrast
        dvals = DT[np.isfinite(DT)]
        tlo, thi = (np.percentile(dvals, [2, 98]) if dvals.size else (25, 300))
        if not (thi > tlo):
            tlo, thi = 25, 300
        im = axL.imshow(DT, origin="lower", extent=ext, cmap="inferno", vmin=tlo, vmax=thi)
        axL.set_title(f"{name} — Diviner $T_{{bol,max}}$  ({lat:.2f}, {lon:.1f})", fontsize=10)
        cbL = fig.colorbar(im, ax=axL, shrink=0.85)
        cbL.set_label("$T$ (K)", fontsize=9)

        axR = axes[row, 1]
        EL_m = EL * 0.5                       # LOLA DN -> height in meters
        elv0, elv1 = np.nanpercentile(EL_m, [1, 99])
        hs = ls.hillshade(np.nan_to_num(EL_m), vert_exag=2.0, dx=200, dy=200)
        axR.imshow(hs, origin="lower", extent=ext, cmap="gray")
        axR.imshow(EL_m, origin="lower", extent=ext, cmap="terrain", alpha=0.5,
                   vmin=elv0, vmax=elv1)
        sm = ScalarMappable(norm=Normalize(elv0, elv1), cmap="terrain")
        cbR = fig.colorbar(sm, ax=axR, shrink=0.85)
        cbR.set_label("Elevation (m)", fontsize=9)
        axR.set_title(f"{name} — LOLA topography", fontsize=10)

        for ax in (axL, axR):
            th = np.linspace(0, 2 * np.pi, 100)
            ax.plot(a.win_km * np.cos(th), a.win_km * np.sin(th), "c-", lw=1.2)
            ax.plot(0, 0, "c+", ms=9, mew=1.6)
            ax.set_xlabel("x (km)")
        axL.set_ylabel("y (km)")

    fig.suptitle("Diviner max bolometric T vs LOLA topography — site overviews "
                 f"(±{a.roi_km:.0f} km; cyan = ±{a.win_km:.0f} km comparison window)",
                 fontsize=12, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    for ext_fmt, dpi in ((".png", 300), (".pdf", None)):
        fig.savefig(out.with_suffix(ext_fmt), dpi=dpi)
    print(f"wrote {out.with_suffix('.png')} and {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
