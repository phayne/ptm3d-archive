#!/usr/bin/env python3
"""PTM3D-vs-Diviner scatter + density-contour figures, ringfix comparison
(2026-07-22, Paul's ask after reviewing the _ringfix bias table).

Reuses diviner_seasonal_compare.py's own model_seasonal/diviner_seasonal/
season_masks functions (same matched 250 m cells as the _ringfix stratified
comparison, same paper coordinates -88.181,6.041) -- no re-derivation of the
registration/footprint logic. 2x2 panels: summer max/mean, winter max/mean.
PTM3D on y, Diviner on x, 1:1 line, equal aspect, common axis range per panel
(reusing SCALES_BY_SITE's existing (vlo,vhi) per panel for consistency with
the color-mapped comparison figure). Density via a 2D histogram + filled
contours (single blue hue, light->dark -- crew dataviz convention: sequential
= one hue, never rainbow), thinned/low-alpha scatter underneath so the
contours read as the primary signal. NO on-figure data tables/annotations
(bias/RMSE/N) -- those live in the companion .caption.md, per crew convention
(`~/.claude/ptm-crew-conventions.md` SS3).

Per Fermi's finding, winter-max stratification on the winter Diviner reference
is noise-selecting -- the scatter sidesteps this entirely by not stratifying
at all: every panel plots ALL valid core cells, no PSR/sunlit split.
"""
from __future__ import annotations
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import spiceypy as sp

sys.path.insert(0, str(Path(__file__).parent.parent / "src/python"))
import diviner_seasonal_compare as D
from ptm_pubstyle import apply_style

LA, LO = -88.181, 6.041
OUTDIR = Path("/Users/paha3326/Projects/ptm/docs/figures/diviner_validation/seasonal/fig8_hybrid")

# Same (vlo, vhi) axis ranges as the color-mapped _ringfix figure, for the
# "sunlit" site key -- SCALES_BY_SITE[key][(season, stat)] = (vlo, vhi, dlo, dhi)
PANELS = [
    ("summer", "max", "Summer max"),
    ("summer", "mean", "Summer mean"),
    ("winter", "max", "Winter max"),
    ("winter", "mean", "Winter mean"),
]

# Sequential single-hue (blue), light->dark -- matches the crew dataviz
# convention (never rainbow/jet for a magnitude/density encoding).
BLUES = LinearSegmentedColormap.from_list(
    "ptm_seq_blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#184f95", "#0d366b"])


def load_run(run_dir: Path):
    smask, wmask, _ = D.season_masks(run_dir)
    g_sum = D.model_seasonal(run_dir, LA, LO, smask)
    g_win = D.model_seasonal(run_dir, LA, LO, wmask)
    return {"summer": g_sum, "winter": g_win}


def panel_data(grids, season, stat):
    g = grids[season]
    div_field = "max" if stat == "max" else "avg"
    dv = D.diviner_seasonal(D.SEASDIR / f"polar_south_80_{season}_{div_field}_polar.npy", g)
    model = np.where(g["core"], g[stat], np.nan)
    dvm = np.where(g["core"], dv, np.nan)
    valid = np.isfinite(model) & np.isfinite(dvm)  # ALL valid cells, no PSR/sunlit selection
    return dvm[valid], model[valid]


def make_figure(run_dir: Path, label: str, out_stem: str):
    grids = load_run(run_dir)
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 9.0))
    for ax, (season, stat, title) in zip(axes.ravel(), PANELS):
        x, y = panel_data(grids, season, stat)
        vlo, vhi, _, _ = D.SCALES_BY_SITE["sunlit"][(season, stat)]

        # thinned/low-alpha scatter (context layer, not the primary signal)
        ax.scatter(x, y, s=4, c="#898781", alpha=0.12, linewidths=0, rasterized=True)

        # 2D-histogram density contours, single blue hue light->dark
        bins = 60
        H, xe, ye = np.histogram2d(x, y, bins=bins, range=[[vlo, vhi], [vlo, vhi]])
        cx = 0.5 * (xe[:-1] + xe[1:]); cy = 0.5 * (ye[:-1] + ye[1:])
        Hs = H.T
        if Hs.max() > 0:
            levels = np.linspace(Hs.max() * 0.05, Hs.max(), 7)
            ax.contourf(cx, cy, Hs, levels=levels, cmap=BLUES, alpha=0.85, extend="max")
            ax.contour(cx, cy, Hs, levels=levels, colors="#184f95", linewidths=0.6, alpha=0.5)

        # 1:1 reference line -- thin, solid, muted (not dashed; matches the
        # crew's "no dashed gridlines/axis rules" convention)
        ax.plot([vlo, vhi], [vlo, vhi], color="#898781", lw=1.2, zorder=5)

        ax.set_xlim(vlo, vhi); ax.set_ylim(vlo, vhi)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel("Diviner $T_\\mathrm{bol}$ (K)")
        ax.set_ylabel("PTM3D $T_\\mathrm{bol}$ (K)")

    fig.suptitle(f"PTM3D vs Diviner $T_\\mathrm{{bol}}$ — Haworth SE rim, corrected ring ({label})",
                 fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = OUTDIR / out_stem
    for fmt, dpi in ((".png", 300), (".pdf", None)):
        fig.savefig(out.with_suffix(fmt), dpi=dpi)
    plt.close(fig)
    print(f"wrote {out.with_suffix('.png')}")
    print(f"wrote {out.with_suffix('.pdf')}")


def main():
    apply_style()
    sp.furnsh(D.METAKERNEL)
    make_figure(Path("/Users/paha3326/projects/moon3d/outputs/fig8_hybrid_production_ringfix/coupled"),
                "coupled", "fig8_hybrid_coupled_rigorous_ringfix_scatter")
    make_figure(Path("/Users/paha3326/projects/moon3d/outputs/fig8_hybrid_production_ringfix/radoff"),
                "radoff", "fig8_hybrid_radoff_rigorous_ringfix_scatter")
    sp.kclear()


if __name__ == "__main__":
    main()
