#!/usr/bin/env python3
"""Combined temperature-map figure: Diviner vs PTM3D max bolometric T for each
validation site, arranged like the site-overview figure (rows = sites).

Columns: Diviner T_bol,max | PTM3D T_bol,max (core-ring) | model - Diviner.
Both temperature panels share a per-row color scale; the difference panel is
diverging. Per-site bias/RMSE annotated. Uses the corrected Diviner projection
and the rigorous max-over-time footprint bolometric T (eps^1/4 * <T^4>^1/4).
"""
from __future__ import annotations
import argparse
import re
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from ptm_pubstyle import apply_style
from diviner_compare import read_ply, face_geometry, load_step
from diviner_data import load_diviner, latlon_to_diviner_xy, KM_PER_DEG

SITES = [("Haworth", "haworth", -87.45, -2.1),
         ("Faustini", "faustini", -87.3, 84.0),
         ("Shackleton", "shackleton", -89.66, 129.2),
         ("Haworth SE rim", "sunlit", -88.181, 6.041)]


def compute(run, lat, lon, g, window_km=15.0, cell_m=250.0, eps=0.95):
    eps4 = eps ** 0.25
    xd0, yd0 = latlon_to_diviner_xy(lat, lon)
    cd = (cell_m / 1000.0) / KM_PER_DEG
    Wd = window_km / KM_PER_DEG
    ex = np.arange(xd0 - Wd, xd0 + Wd + cd, cd)
    ey = np.arange(yd0 - Wd, yd0 + Wd + cd, cd)
    nx, ny = len(ex) - 1, len(ey) - 1
    CX, CY = np.meshgrid(0.5 * (ex[:-1] + ex[1:]), 0.5 * (ey[:-1] + ey[1:]))
    core = (np.hypot(CX - xd0, CY - yd0) * KM_PER_DEG) <= window_km
    # model
    v, f, flat, flon = read_ply(str(run / "mesh.ply"))
    area, flat, flon = face_geometry(v, f, flat, flon)
    fxd, fyd = latlon_to_diviner_xy(flat, flon)
    rkm = np.hypot(fxd - xd0, fyd - yd0) * KM_PER_DEG
    ix = np.floor((fxd - ex[0]) / cd).astype(int); iy = np.floor((fyd - ey[0]) / cd).astype(int)
    keep = (rkm <= window_km) & (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    lin = iy[keep] * nx + ix[keep]; A = area[keep]
    Asum = np.bincount(lin, weights=A, minlength=nx * ny)
    steps = sorted((run / "temperature").glob("temperature_*.bin"),
                   key=lambda p: int(re.search(r"(\d+)", p.name).group(1)))
    best = np.zeros(nx * ny)
    for sp in steps:
        T = load_step(sp)[keep]
        num = np.bincount(lin, weights=A * (T.astype(np.float64) ** 4), minlength=nx * ny)
        with np.errstate(invalid="ignore", divide="ignore"):
            best = np.maximum(best, eps4 * np.where(Asum > 0, num / Asum, 0.0) ** 0.25)
    model = np.where(Asum > 0, best, np.nan).reshape(ny, nx)
    # diviner on the same cells
    DX, DY = np.meshgrid(g.x_deg, g.y_deg)
    sxa = (g.x_deg >= ex[0]) & (g.x_deg <= ex[-1]); sya = (g.y_deg >= ey[0]) & (g.y_deg <= ey[-1])
    dx = DX[np.ix_(sya, sxa)]; dy = DY[np.ix_(sya, sxa)]; dt = g.t_max[np.ix_(sya, sxa)]
    o = np.isfinite(dt)
    dix = np.floor((dx[o] - ex[0]) / cd).astype(int); diy = np.floor((dy[o] - ey[0]) / cd).astype(int)
    dd = (dix >= 0) & (dix < nx) & (diy >= 0) & (diy < ny); dl = diy[dd] * nx + dix[dd]
    ds = np.bincount(dl, weights=dt[o][dd].astype(float), minlength=nx * ny)
    dc = np.bincount(dl, minlength=nx * ny)
    div = np.where(dc > 0, ds / np.maximum(dc, 1), np.nan).reshape(ny, nx)
    model = np.where(core, model, np.nan); div = np.where(core, div, np.nan)
    ext = [-window_km, window_km, -window_km, window_km]
    return model, div, ext


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diviner", required=True)
    ap.add_argument("--outroot", default="/Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/europa/heat3d/output")
    ap.add_argument("--suffix", default="_cr")
    ap.add_argument("--window-km", type=float, default=15.0)
    ap.add_argument("--out", default="docs/figures/diviner_validation/temp_maps.png")
    a = ap.parse_args()
    g = load_diviner(a.diviner)
    apply_style()
    rows = [(disp, k, lat, lon) for disp, k, lat, lon in SITES
            if (Path(a.outroot) / f"divval_{k}{a.suffix}" / "statistics" / "temperature_max.bin").exists()]
    fig, axes = plt.subplots(len(rows), 3, figsize=(8.2, 2.7 * len(rows)))
    if len(rows) == 1: axes = axes[None, :]
    for r, (disp, k, lat, lon) in enumerate(rows):
        run = Path(a.outroot) / f"divval_{k}{a.suffix}"
        model, div, ext = compute(run, lat, lon, g, window_km=a.window_km)
        valid = np.isfinite(model) & np.isfinite(div)
        res = (model - div)[valid]; bias = res.mean(); rmse = np.sqrt((res ** 2).mean())
        both = np.concatenate([div[np.isfinite(div)], model[np.isfinite(model)]])
        vmin = float(np.percentile(both, 1)) if both.size else 40.0
        vmax = float(np.percentile(both, 99)) if both.size else 380.0
        if not (vmax > vmin):
            vmin, vmax = 40.0, 380.0
        dmv = np.percentile(np.abs(res), 98) if res.size else 1.0
        dm = max(float(dmv) if np.isfinite(dmv) else 1.0, 1.0)
        for c, (arr, ttl, cmap, norm) in enumerate([
                (div, "Diviner", "inferno", Normalize(vmin, vmax)),
                (model, "PTM3D (core-ring)", "inferno", Normalize(vmin, vmax)),
                (model - div, "model - Diviner", "RdBu_r",
                 TwoSlopeNorm(vcenter=0.0, vmin=-dm, vmax=dm))]):
            ax = axes[r, c]
            im = ax.imshow(arr, origin="lower", extent=ext, cmap=cmap, norm=norm)
            fig.colorbar(im, ax=ax, shrink=0.8, label="K")
            if r == 0: ax.set_title(ttl, fontsize=11)
            if c == 0:
                ax.set_ylabel(f"{disp}\ny (km)", fontsize=10)
            ax.set_xlabel("x (km)")
        axes[r, 2].text(0.03, 0.96, f"bias {bias:+.1f}\nRMSE {rmse:.1f}",
                        transform=axes[r, 2].transAxes, va="top", fontsize=8,
                        bbox=dict(fc="white", ec="0.7", alpha=0.85))
    fig.suptitle("PTM3D vs Diviner max bolometric T — core+ring runs (±%.0f km windows)"
                 % a.window_km, fontsize=12, fontweight="bold", y=0.997)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
