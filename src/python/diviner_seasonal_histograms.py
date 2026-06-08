#!/usr/bin/env python3
"""Histograms of footprint-Tbol distributions (PTM3D vs Diviner) per site,
season, and statistic (max/mean) over the comparison window's core cells.

Used to choose appropriate per-site color scales for the seasonal comparison
figures (diviner_seasonal_compare.py). Reuses that module's pipeline so the
binning, registration, and season split are identical.

Output: docs/figures/diviner_validation/seasonal/seasonal_histograms.{png,pdf}
and a printed table of suggested (1-99 percentile) scale ranges per panel.
"""
from __future__ import annotations
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import spiceypy as sp

sys.path.insert(0, str(Path(__file__).parent))
import diviner_seasonal_compare as C
from ptm_pubstyle import apply_style


def main():
    apply_style()
    sp.furnsh(C.METAKERNEL)
    combos = [("summer", "max"), ("summer", "mean"),
              ("winter", "max"), ("winter", "mean")]
    sites = [(disp, key, lat, lon) for (disp, key, lat, lon) in C.SITES]

    fig, axes = plt.subplots(len(sites), len(combos),
                             figsize=(15.0, 3.2 * len(sites)))
    suggested = {}
    for r, (disp, key, lat0, lon0) in enumerate(sites):
        run = C.resolve_run(key)
        if not (run / "mesh.ply").exists():
            print(f"skip {key}: no run"); continue
        smask, wmask, _ = C.season_masks(run)
        mods = {"summer": C.model_seasonal(run, lat0, lon0, smask),
                "winter": C.model_seasonal(run, lat0, lon0, wmask)}
        print(f"[{key}] run={run.name}")
        for c, (season, stat) in enumerate(combos):
            g = mods[season]
            dv = C.diviner_seasonal(
                C.SEASDIR / f"polar_south_80_{season}_"
                f"{'max' if stat=='max' else 'avg'}_polar.npy", g)
            mv = g[stat][g["core"] & np.isfinite(g[stat])]
            dvv = dv[g["core"] & np.isfinite(dv)]
            ax = axes[r, c]
            both = np.concatenate([mv, dvv]) if mv.size and dvv.size else mv
            lo, hi = (np.nanpercentile(both, [0.5, 99.5])
                      if both.size else (0, 1))
            bins = np.linspace(lo, hi, 50)
            ax.hist(dvv, bins=bins, color="tab:orange", alpha=0.55,
                    density=True, label="Diviner")
            ax.hist(mv, bins=bins, color="tab:blue", alpha=0.55,
                    density=True, label="PTM3D")
            # suggested T scale = combined 1-99 percentile (rounded to 5 K)
            p1, p99 = np.nanpercentile(both, [1, 99]) if both.size else (lo, hi)
            s_lo, s_hi = 5 * np.floor(p1 / 5), 5 * np.ceil(p99 / 5)
            # suggested dT scale = +/- 95th pct of |model-Diviner| (rounded to 5)
            valid = g["core"] & np.isfinite(g[stat]) & np.isfinite(dv)
            diff = (g[stat] - dv)[valid]
            d95 = np.nanpercentile(np.abs(diff), 95) if diff.size else 5.0
            s_d = max(5.0, 5 * np.ceil(d95 / 5))
            suggested[(key, season, stat)] = (float(s_lo), float(s_hi), float(s_d))
            ax.axvline(s_lo, color="k", ls=":", lw=0.7)
            ax.axvline(s_hi, color="k", ls=":", lw=0.7)
            ax.set_title(f"{disp}\n{season} {stat}", fontsize=8.5)
            ax.set_xlabel("$T_{bol}$ (K)", fontsize=8)
            ax.tick_params(labelsize=7)
            if r == 0 and c == 0:
                ax.legend(fontsize=7)
            ax.text(0.02, 0.96,
                    f"PTM3D {np.median(mv):.0f}\nDiv {np.median(dvv):.0f}\n"
                    f"-> {s_lo:.0f}-{s_hi:.0f}",
                    transform=ax.transAxes, va="top", fontsize=6.5,
                    bbox=dict(fc="white", ec="0.7", alpha=0.8))
    fig.suptitle("Footprint $T_{bol}$ distributions: PTM3D vs Diviner "
                 "(dotted = suggested 1-99% scale)", fontsize=12,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    outdir = C.REPO / "docs/figures/diviner_validation/seasonal"
    for fmt, dpi in ((".png", 200), (".pdf", None)):
        fig.savefig(outdir / ("seasonal_histograms" + fmt), dpi=dpi)
    plt.close(fig)
    print(f"\nwrote {outdir/'seasonal_histograms.png'}")

    print("\n=== SUGGESTED per-site scales: T_lo..T_hi (dT +/-) ===")
    print('SCALES_BY_SITE = {')
    for (disp, key, lat, lon) in sites:
        print(f'    "{key}": {{')
        for (season, stat) in combos:
            k = (key, season, stat)
            if k in suggested:
                lo_, hi_, d_ = suggested[k]
                print(f'        ("{season}", "{stat}"):'
                      f' ({lo_:.0f}, {hi_:.0f}, {-d_:.0f}, {d_:.0f}),')
        print('    },')
    print('}')
    sp.kclear()


if __name__ == "__main__":
    main()
