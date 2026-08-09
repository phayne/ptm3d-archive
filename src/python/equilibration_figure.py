#!/usr/bin/env python3
"""
F4 — Thermal equilibration over successive lunations.

Reads the per-location depth CSVs written during a multi-lunation lunar run
(equilibration phase + output window) and plots, for a few representative
latitudes:

  (a) Diurnal-mean surface temperature for each whole lunation
  (b) The same trace expressed as |T_n - T_final|, on a log-y axis, which
      makes the geometric convergence to the periodic steady state explicit.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ptm.pubstyle import (
    apply_style, COLORS, DOUBLE_W, LATITUDE_COLORS, panel_label,
)


LUNAR_DAY_S = 29.530588 * 86400.0


def load_depth_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows, times = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("time"):
                continue
            parts = line.split(",")
            try:
                dt = datetime.fromisoformat(parts[0])
            except ValueError:
                dt = datetime.fromisoformat(parts[0].replace("Z", "+00:00"))
            times.append(dt)
            rows.append([float(x) for x in parts[1:]])
    t0 = times[0]
    sec = np.array([(t - t0).total_seconds() for t in times])
    T = np.array(rows, dtype=np.float64)
    return sec, T


def lunation_means(times_s: np.ndarray, T_surf: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Return (lunation_index, diurnal-mean surface T) for each whole lunation
    contained in the time series."""
    n_full = int(np.floor(times_s[-1] / LUNAR_DAY_S))
    lun_idx = np.arange(1, n_full + 1)
    means = np.empty_like(lun_idx, dtype=np.float64)
    for i, k in enumerate(lun_idx):
        t0 = (k - 1) * LUNAR_DAY_S
        t1 = k * LUNAR_DAY_S
        m = (times_s >= t0) & (times_s < t1)
        # Time-weighted mean (trapezoidal) — sample cadence is uniform here,
        # so simple mean is fine.
        means[i] = T_surf[m].mean() if m.any() else np.nan
    return lun_idx, means


def make_figure(run_dir: Path, out_path: Path) -> None:
    apply_style()

    locations = [
        ("Equator",  0,  LATITUDE_COLORS[0]),
        ("Lat_30N",  30, LATITUDE_COLORS[30]),
        ("Lat_60N",  60, LATITUDE_COLORS[60]),
    ]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(DOUBLE_W * 1.4, 0.55 * DOUBLE_W * 1.4))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.93, bottom=0.14, wspace=0.22)

    print("  diurnal-mean surface T per lunation:")
    series = []  # collected for the convergence panel
    for name, lat, color in locations:
        csv = run_dir / "locations" / f"{name}_depths.csv"
        if not csv.exists():
            print(f"  warning: {csv} missing, skipping")
            continue
        times_s, T = load_depth_csv(csv)
        lun, means = lunation_means(times_s, T[:, 0])
        ax1.plot(lun, means, "o-", color=color, lw=2.2, ms=9, mfc="white",
                 mew=1.6, label=fr"lat $= {lat}^\circ$")
        series.append((lat, color, lun, means))
        drift = float(np.abs(means[-1] - means[-2])) if means.size >= 2 else float("nan")
        print(f"    lat {lat:>3d}°: T = "
              f"{[round(m, 2) for m in means.tolist()]}  |  final |ΔT| = "
              f"{drift:.3f} K")

    ax1.set_xlabel("Lunation number")
    ax1.set_ylabel(r"Diurnal-mean surface $T_0$ (K)")
    if series:
        n_max = max(s[2].max() for s in series)
        ax1.set_xticks(np.arange(1, n_max + 1))
    ax1.set_ylim(170, 230)
    ax1.legend(loc="center right")
    panel_label(ax1, "(a)", x=0.04, y=0.95)

    # ---- (b) convergence to final lunation, log scale --------------------
    for lat, color, lun, means in series:
        delta = np.abs(means[:-1] - means[-1])
        ax2.semilogy(lun[:-1], delta, "o-", color=color, lw=2.2, ms=9,
                     mfc="white", mew=1.6, label=fr"lat $= {lat}^\circ$")
    ax2.axhline(0.1, color=COLORS["neutral"], lw=1.0, ls="--")
    ax2.text(1.05, 0.115, "0.1 K", fontsize=11, color=COLORS["neutral"],
             va="bottom")
    ax2.set_xlabel("Lunation number")
    ax2.set_ylabel(r"$|\langle T_0\rangle_n - \langle T_0\rangle_\mathrm{final}|$ (K)")
    if series:
        ax2.set_xticks(np.arange(1, n_max))
    ax2.set_ylim(0.05, 30)
    ax2.legend(loc="upper right")
    panel_label(ax2, "(b)", x=0.04, y=0.95)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"  wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", nargs="?", default="output/lunar_validation_b15")
    p.add_argument("--out",
                   default="docs/figures/v2/equilibration_convergence.png")
    args = p.parse_args()
    make_figure(Path(args.run_dir), Path(args.out))


if __name__ == "__main__":
    main()
