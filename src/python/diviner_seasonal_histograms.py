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
import argparse
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", action="append", default=[], metavar="SITE=PATH",
                    help="Explicit absolute run-directory override for a site key "
                         "(repeatable). Does NOT change the built-in defaults.")
    ap.add_argument("--site", action="append", default=[], metavar="KEY",
                    help="Restrict to these site key(s) (repeatable). Default: all.")
    ap.add_argument("--outdir", default=None,
                    help="Output directory (default: docs/figures/diviner_validation/seasonal).")
    ap.add_argument("--figname", default="seasonal_histograms",
                    help="Output figure basename (no extension).")
    ap.add_argument("--paper2x2", action="store_true",
                    help="Clean publication layout (single site only): the four "
                         "season x stat panels arranged 2x2, no suggested-scale "
                         "dotted verticals, no figure title (Fig. 9).")
    args = ap.parse_args(argv)
    for spec in args.run_dir:
        if "=" not in spec:
            ap.error(f"--run-dir expects SITE=PATH, got {spec!r}")
        k, p = spec.split("=", 1)
        C.RUN_DIR_ABS[k] = p
    site_filter = set(args.site) or None

    apply_style()
    sp.furnsh(C.METAKERNEL)
    combos = [("summer", "max"), ("summer", "mean"),
              ("winter", "max"), ("winter", "mean")]
    sites = [(disp, key, lat, lon) for (disp, key, lat, lon) in C.SITES
             if site_filter is None or key in site_filter]

    paper = args.paper2x2
    if paper and len(sites) != 1:
        ap.error("--paper2x2 is only valid with exactly one --site")
    if paper:
        # Fig. 9: clean 2x2 publication layout for the single site's four
        # season x stat panels (no suggested-scale dotted verticals, no title).
        fig, axg = plt.subplots(2, 2, figsize=(8.4, 6.2))
        axsel = axg.ravel()
    else:
        fig, axes = plt.subplots(len(sites), len(combos), squeeze=False,
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
            ax = axsel[c] if paper else axes[r, c]
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
            # The dotted verticals annotate the suggested color scale (a utility
            # marker). Omit them (and the figure title) in the clean paper layout.
            if not paper:
                ax.axvline(s_lo, color="k", ls=":", lw=0.7)
                ax.axvline(s_hi, color="k", ls=":", lw=0.7)
            ax.set_title(f"{season} {stat}" if paper else f"{disp}\n{season} {stat}",
                         fontsize=11 if paper else 8.5)
            ax.set_xlabel("footprint $T_{bol}$ (K)" if paper else "$T_{bol}$ (K)",
                          fontsize=9 if paper else 8)
            if paper:
                ax.set_ylabel("density", fontsize=9)
            ax.tick_params(labelsize=8 if paper else 7)
            if (paper and c == 0) or (not paper and r == 0 and c == 0):
                ax.legend(fontsize=8 if paper else 7)
            ax.text(0.02, 0.96,
                    (f"PTM3D {np.median(mv):.0f} K\nDiviner {np.median(dvv):.0f} K"
                     if paper else
                     f"PTM3D {np.median(mv):.0f}\nDiv {np.median(dvv):.0f}\n"
                     f"-> {s_lo:.0f}-{s_hi:.0f}"),
                    transform=ax.transAxes, va="top", fontsize=8 if paper else 6.5,
                    bbox=dict(fc="white", ec="0.7", alpha=0.8))
    if not paper:
        fig.suptitle("Footprint $T_{bol}$ distributions: PTM3D vs Diviner "
                     "(dotted = suggested 1-99% scale)", fontsize=12,
                     fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97) if not paper else (0, 0, 1, 1))
    outdir = (Path(args.outdir) if args.outdir
              else C.REPO / "docs/figures/diviner_validation/seasonal")
    outdir.mkdir(parents=True, exist_ok=True)
    for fmt, dpi in ((".png", 200), (".pdf", None)):
        fig.savefig(outdir / (args.figname + fmt), dpi=dpi)
    plt.close(fig)
    print(f"\nwrote {outdir/(args.figname + '.png')}")

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
