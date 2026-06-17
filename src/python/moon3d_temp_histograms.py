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

sys.path.insert(0, str(Path(__file__).parent))
from ptm_pubstyle import apply_style, COLORS, DOUBLE_W, panel_label


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
    a = ap.parse_args()

    cdir, bdir = Path(a.coupled), Path(a.baseline)
    fields = [("min", "Minimum"), ("avg", "Mean"), ("max", "Maximum")]
    cpl = {k: load_stat(cdir, k) for k, _ in fields}
    bas = {k: load_stat(bdir, k) for k, _ in fields}

    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_W, 0.42 * DOUBLE_W))

    for ax, (key, title), lab in zip(axes, fields, "abc"):
        c, b = cpl[key], bas[key]
        lo = min(c.min(), b.min())
        hi = max(np.percentile(c, 99.9), np.percentile(b, 99.9))
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
        panel_label(ax, f"({lab})")
        # medians as guide lines
        ax.axvline(np.median(b), color=COLORS["ref"], ls=":", lw=1.1)
        ax.axvline(np.median(c), color=COLORS["ptm"], ls=":", lw=1.1)
        if key == "min":
            ax.legend(loc="upper right")
    axes[0].set_ylabel("face count")

    fig.suptitle("Moon3D 509k-face mesh — per-face temperature distributions",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    print(f"wrote {out}")

    # quick numeric summary
    for key, title in fields:
        c, b = cpl[key], bas[key]
        print(f"  {title:8s}: rad ON med={np.median(c):6.1f}K  OFF med={np.median(b):6.1f}K  "
              f"Δmed={np.median(c)-np.median(b):+5.1f}K  max|Δface|={np.abs(c-b).max():.1f}K")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
