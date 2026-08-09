#!/usr/bin/env python3
"""Step 4 — histograms of per-face min/mean/max temperature for the Moon3D
500k mesh, radiative coupling ON vs OFF.

Uses the PTM3D pretty-plots style (ptm_pubstyle / pretty-plots).

    python3 moon3d_temp_histograms.py \
        --coupled  output/moon3d_500k_clean/statistics \
        --baseline output/moon3d_500k_clean_radoff/statistics \
        --out docs/figures/moon3d_500k/temp_histograms.png
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ptm.pubstyle import apply_style, COLORS, DOUBLE_W, panel_label, load_face_mask


def load_stat(stat_dir: Path, name: str) -> np.ndarray:
    """Read a PTM3D statistics binary (leading int32 face-count header)."""
    raw = np.fromfile(stat_dir / f"temperature_{name}.bin", dtype=np.float32)
    return raw[1:]  # drop the count word


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coupled", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--out", default="docs/figures/moon3d_500k/temp_histograms.png")
    ap.add_argument("--bins", type=int, default=120)
    ap.add_argument("--mask-tmax", type=float, default=0.0,
                    help="exclude faces whose coupled max-T exceeds this (K) as "
                         "view-factor over-count artifacts; 0 = keep all")
    ap.add_argument("--reclassify-mask", default="",
                    help="boolean face-mask (make_psr_mask.py): spurious-lit deep-PSR faces "
                         "reclassified to the PSR-floor temperature")
    a = ap.parse_args()

    cdir, bdir = Path(a.coupled), Path(a.baseline)
    fields = [("min", "Minimum"), ("avg", "Mean"), ("max", "Maximum")]
    cpl = {k: load_stat(cdir, k) for k, _ in fields}
    bas = {k: load_stat(bdir, k) for k, _ in fields}

    rc = load_face_mask(a.reclassify_mask) if a.reclassify_mask else np.zeros(len(cpl["max"]), bool)
    if rc.any():
        dmax = np.fromfile(cdir / "direct_solar_max.bin", dtype=np.float32)[1:]
        gp = (dmax <= 1e-6) & ~rc                          # genuine PSR -> floor value
        for k, _ in fields:
            cpl[k][rc] = np.median(cpl[k][gp]); bas[k][rc] = np.median(bas[k][gp])
        print(f"  reclassified {int(rc.sum())} spurious-lit deep-PSR faces to PSR floor")

    if a.mask_tmax > 0:
        keep = (cpl["max"] <= a.mask_tmax) | rc           # keep reclassified faces
        cpl = {k: v[keep] for k, v in cpl.items()}
        bas = {k: v[keep] for k, v in bas.items()}
        print(f"  masked {int(np.count_nonzero(~keep))} hot faces "
              f"(coupled max-T > {a.mask_tmax:.0f} K) as VF-overcount artifacts")

    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_W, 0.42 * DOUBLE_W))

    for ax, (key, title), lab in zip(axes, fields, "abc"):
        c, b = cpl[key], bas[key]
        lo = min(c.min(), b.min())
        hi = max(c.max(), b.max())          # full range so the warm tail is never clipped
        bins = np.linspace(lo, hi, a.bins)
        ax.hist(b, bins=bins, color=COLORS["ref"], alpha=0.55,
                label="rad OFF", histtype="stepfilled", lw=0)
        ax.hist(c, bins=bins, color=COLORS["ptm"], alpha=0.55,
                label="rad ON", histtype="stepfilled", lw=0)
        # outline for readability
        ax.hist(b, bins=bins, color=COLORS["ref"], histtype="step", lw=1.3)
        ax.hist(c, bins=bins, color=COLORS["ptm"], histtype="step", lw=1.3)
        ax.set_title(f"{title} temperature")
        ax.set_xlabel("T (K)")
        ax.set_yscale("log")
        ax.set_ylim(bottom=0.8)
        if key in ("avg", "max"):
            # left margin clears the (b)/(c) label from the tall floor peak;
            # right margin keeps the warm tail from sitting flush against the edge
            ax.set_xlim(lo - 0.16 * (hi - lo), hi + 0.06 * (hi - lo))
        panel_label(ax, f"({lab})")
        # medians as guide lines
        ax.axvline(np.median(b), color=COLORS["ref"], ls=":", lw=1.1)
        ax.axvline(np.median(c), color=COLORS["ptm"], ls=":", lw=1.1)
        if key == "min":
            # empty white space here is the LEFT side (no faces below the ~68 K floor)
            ax.legend(loc="upper left", bbox_to_anchor=(0.04, 0.88),
                      fontsize=7.5, framealpha=0.9, borderpad=0.4, handlelength=1.4)
    axes[0].set_ylabel("face count")

    fig.tight_layout()
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    if out.suffix == ".png":
        fig.savefig(out.with_suffix(".pdf"))
    print(f"wrote {out}" + (" (+pdf)" if out.suffix == ".png" else ""))

    # quick numeric summary
    for key, title in fields:
        c, b = cpl[key], bas[key]
        print(f"  {title:8s}: rad ON med={np.median(c):6.1f}K  OFF med={np.median(b):6.1f}K  "
              f"Δmed={np.median(c)-np.median(b):+5.1f}K  max|Δface|={np.abs(c-b).max():.1f}K")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
