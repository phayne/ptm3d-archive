#!/usr/bin/env python3
"""Paper figures for the Haworth SE rim validation site.

(1) seasonal_se_rim_combined.{png,pdf}: a single figure combining the site
    overview (LOLA topography + Diviner summer-max context over a wide ROI, with
    the comparison window outlined) with the PTM3D-vs-Diviner seasonal Tbol
    comparison (summer/winter x max/mean; Diviner | PTM3D | model-Diviner).

(2) hist_se_rim.{png,pdf}: a clean histogram of the footprint-Tbol distributions
    (PTM3D vs Diviner) for the four season/stat panels at this site.

Reuses diviner_seasonal_compare's pipeline (binning, registration, season split,
per-site color scales) so the comparison is identical to the per-site figure.
"""
from __future__ import annotations
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm, LightSource
from matplotlib.cm import ScalarMappable
import matplotlib.gridspec as gridspec
import numpy as np
import rasterio
import spiceypy as sp

sys.path.insert(0, str(Path(__file__).parent))
import diviner_seasonal_compare as C
from ptm_pubstyle import apply_style

NAME, KEY, LAT, LON = "Haworth SE rim", "sunlit", -88.181, 6.041
ROI_KM = 30.0          # wide context half-width
LOLA = C.REPO / "data/dem/validation/_lola_20m_spstereo_pm120km.tif"
COMBOS = [("summer", "max"), ("summer", "mean"),
          ("winter", "max"), ("winter", "mean")]
LABELS = {("summer", "max"): "summer max", ("summer", "mean"): "summer mean",
          ("winter", "max"): "winter max", ("winter", "mean"): "winter mean"}


def context_grids():
    """LOLA elevation + Diviner summer-max binned on a +/-ROI_KM stereo grid."""
    x0, y0 = C.TO_XY.transform(LON, LAT)
    n = 300
    gx = np.linspace(x0 - ROI_KM * 1e3, x0 + ROI_KM * 1e3, n)
    gy = np.linspace(y0 - ROI_KM * 1e3, y0 + ROI_KM * 1e3, n)
    GX, GY = np.meshgrid(gx, gy)
    # LOLA elevation (DEM is already south-polar stereographic, R=1737400)
    src = rasterio.open(LOLA)
    inv = ~src.transform
    elev = src.read(1).astype(float)
    col, row = inv * (GX.ravel(), GY.ravel())
    from scipy.ndimage import map_coordinates
    EL = map_coordinates(elev, [row, col], order=1, mode="constant",
                         cval=np.nan).reshape(GX.shape) * 0.5  # DN->m
    # Diviner summer-max sampled at each grid point via nearest neighbor
    # (the level-4 product is a sparse point cloud; binning it onto a fine grid
    # leaves white gaps wherever a cell catches no point, so query the nearest
    # Diviner observation at every grid pixel instead).
    d = np.load(C.SEASDIR / "polar_south_80_summer_max_polar.npy")
    good = np.isfinite(d[:, 2]) & (d[:, 2] > 0)
    dx, dy = C.TO_XY.transform(d[good, 1], d[good, 0])
    from scipy.spatial import cKDTree
    tree = cKDTree(np.column_stack([dx, dy]))
    dist, idx = tree.query(np.column_stack([GX.ravel(), GY.ravel()]), k=1)
    DT = d[good, 2][idx].reshape(GX.shape).astype(float)
    DT[dist.reshape(GX.shape) > 1500.0] = np.nan   # mask if nearest point >1.5 km away
    ext = [-ROI_KM, ROI_KM, -ROI_KM, ROI_KM]
    return EL, DT, ext


def main():
    apply_style()
    sp.furnsh(C.METAKERNEL)
    run = C.resolve_run(KEY)
    smask, wmask, _ = C.season_masks(run)
    mods = {"summer": C.model_seasonal(run, LAT, LON, smask),
            "winter": C.model_seasonal(run, LAT, LON, wmask)}
    rows = []
    for season, stat in COMBOS:
        g = mods[season]
        dv = C.diviner_seasonal(
            C.SEASDIR / f"polar_south_80_{season}_{'max' if stat=='max' else 'avg'}_polar.npy", g)
        model = np.where(g["core"], g[stat], np.nan)
        dvm = np.where(g["core"], dv, np.nan)
        st = C.stats(model, dvm, g["core"])
        rows.append((season, stat, model, dvm, st, g))

    EL, DT, cext = context_grids()
    scales = C.SCALES_BY_SITE[KEY]

    # ---- (1) combined figure ----
    fig = plt.figure(figsize=(8.6, 14.0))
    outer = gridspec.GridSpec(2, 1, height_ratios=[1.35, 4.0], hspace=0.18)
    # top: two LARGE context panels with generous spacing
    topgs = outer[0].subgridspec(1, 2, wspace=0.45)
    axT = fig.add_subplot(topgs[0, 0])
    ls = LightSource(azdeg=315, altdeg=30)
    hs = ls.hillshade(np.nan_to_num(EL, nan=np.nanmean(EL)), vert_exag=2.0, dx=200, dy=200)
    e0, e1 = np.nanpercentile(EL, [1, 99])
    axT.imshow(hs, origin="lower", extent=cext, cmap="gray")
    axT.imshow(EL, origin="lower", extent=cext, cmap="terrain", alpha=0.55, vmin=e0, vmax=e1)
    fig.colorbar(ScalarMappable(Normalize(e0, e1), "terrain"), ax=axT, shrink=0.85,
                 pad=0.03, label="Elevation (m)")
    axT.set_title("LOLA topography", fontsize=11)

    axD = fig.add_subplot(topgs[0, 1])
    dl, dh = np.nanpercentile(DT[np.isfinite(DT)], [2, 98])
    imD = axD.imshow(DT, origin="lower", extent=cext, cmap="inferno", vmin=dl, vmax=dh)
    fig.colorbar(imD, ax=axD, shrink=0.85, pad=0.03, label="$T$ (K)")
    axD.set_title("Diviner summer $T_{bol,max}$", fontsize=11)
    for ax in (axT, axD):
        th = np.linspace(0, 2 * np.pi, 100)
        ax.plot(C.CORE_KM * np.cos(th), C.CORE_KM * np.sin(th), "c-", lw=1.4)
        ax.plot(0, 0, "c+", ms=10, mew=1.8)
        ax.set_xlabel("x (km)"); ax.set_ylabel("y (km)")
        ax.set_aspect("equal")

    # bottom: 4x3 comparison grid
    gs = outer[1].subgridspec(4, 3, hspace=0.32, wspace=0.30)
    for r, (season, stat, model, dvm, st, g) in enumerate(rows):
        ext = [(g["ex"][0]-g["x0"])/1e3, (g["ex"][-1]-g["x0"])/1e3,
               (g["ey"][0]-g["y0"])/1e3, (g["ey"][-1]-g["y0"])/1e3]
        vlo, vhi, dlo, dhi = scales[(season, stat)]
        mdisp = np.where(g["core"], C.fill_internal_gaps(model), np.nan)
        panels = [(dvm, "inferno", Normalize(vlo, vhi), "K"),
                  (mdisp, "inferno", Normalize(vlo, vhi), "K"),
                  (mdisp - dvm, "RdBu_r", TwoSlopeNorm(0.0, dlo, dhi), "$\\Delta$K")]
        for c, (arr, cmap, norm, lab) in enumerate(panels):
            ax = fig.add_subplot(gs[r, c])
            im = ax.imshow(arr, origin="lower", extent=ext, cmap=cmap, norm=norm)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=lab)
            if c == 0:
                ax.set_ylabel(f"{LABELS[(season,stat)]}\ny (km)", fontsize=9)
            else:
                ax.tick_params(labelleft=False)
            if r == 0:
                ax.set_title(["Diviner", "PTM3D", "model $-$ Diviner"][c], fontsize=10)
            if r == 3:
                ax.set_xlabel("x (km)")
            if c == 2:
                ax.text(0.03, 0.96, f"bias {st['bias']:+.1f}\nRMSE {st['rmse']:.1f}",
                        transform=ax.transAxes, va="top", fontsize=7.5,
                        bbox=dict(fc="white", ec="0.7", alpha=0.85))
    outdir = C.REPO / "docs/figures/diviner_validation"
    for fmt, dpi in ((".png", 300), (".pdf", None)):
        fig.savefig(outdir / ("seasonal_se_rim_combined" + fmt), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {outdir/'seasonal_se_rim_combined.png'}")

    # ---- (2) clean histogram ----
    figh, axes = plt.subplots(2, 2, figsize=(8.4, 6.2))
    for ax, (season, stat) in zip(axes.ravel(), COMBOS):
        g = mods[season]
        dv = C.diviner_seasonal(
            C.SEASDIR / f"polar_south_80_{season}_{'max' if stat=='max' else 'avg'}_polar.npy", g)
        mv = g[stat][g["core"] & np.isfinite(g[stat])]
        dvv = dv[g["core"] & np.isfinite(dv)]
        both = np.concatenate([mv, dvv])
        lo, hi = np.nanpercentile(both, [0.5, 99.5])
        bins = np.linspace(lo, hi, 45)
        ax.hist(dvv, bins=bins, color="tab:orange", alpha=0.6, density=True, label="Diviner")
        ax.hist(mv, bins=bins, color="tab:blue", alpha=0.6, density=True, label="PTM3D")
        ax.axvline(np.median(dvv), color="tab:orange", ls="--", lw=1.2)
        ax.axvline(np.median(mv), color="tab:blue", ls="--", lw=1.2)
        ax.set_title(LABELS[(season, stat)], fontsize=11)
        ax.set_xlabel("footprint $T_{bol}$ (K)"); ax.set_ylabel("density")
        ax.text(0.97, 0.95, f"med: PTM3D {np.median(mv):.0f} K\nDiviner {np.median(dvv):.0f} K",
                transform=ax.transAxes, va="top", ha="right", fontsize=8,
                bbox=dict(fc="white", ec="0.7", alpha=0.85))
        if (season, stat) == ("summer", "max"):
            ax.legend(fontsize=9, loc="upper left")
    figh.suptitle("Haworth SE rim — footprint $T_{bol}$ distributions (PTM3D vs Diviner)",
                  fontsize=12, fontweight="bold")
    figh.tight_layout(rect=(0, 0, 1, 0.96))
    for fmt, dpi in ((".png", 300), (".pdf", None)):
        figh.savefig(outdir / ("hist_se_rim" + fmt), dpi=dpi)
    plt.close(figh)
    print(f"wrote {outdir/'hist_se_rim.png'}")
    sp.kclear()


if __name__ == "__main__":
    main()
