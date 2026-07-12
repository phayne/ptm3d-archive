#!/usr/bin/env python3
"""
F5 — Bowl-crater radiation-exchange validation (two-panel).

  (a) Ingersoll, Svitek & Murray (1992) analytic floor equilibrium temperature
      vs solar elevation, for d/D ∈ {0.1, 0.2, 0.3}.
  (b) PTM3D mesh sky view factor F_sky on the floor, compared to the analytic
      sin²α value, across four latitudes for d/D = 0.2.

Inputs are parsed from a captured stdout log of `bin/test_bowl_crater`.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from ingersoll_crater import (
    critical_elevation, sky_view_factor, floor_temperature_curve,
)
from ptm_pubstyle import apply_style, COLORS, DOUBLE_W, panel_label


def parse_log(path: Path) -> list[dict]:
    text = path.read_text()
    sections = text.split("Bowl Crater Test")
    runs = []
    for sec in sections[1:]:
        run = {}
        m = re.search(r"Latitude:\s+([\-\d.]+)", sec)
        run["latitude_deg"] = float(m.group(1)) if m else None
        m = re.search(r"Depth/Diameter:\s+([\-\d.]+)", sec)
        run["depth_to_diameter"] = float(m.group(1)) if m else None
        m = re.search(r"F_sky \(computed\)\s+=\s+([\d.eE+\-]+)", sec)
        run["fsky_floor_ptm3d"] = float(m.group(1)) if m else None
        m = re.search(r"F_sky \(analytical\)\s+=\s+([\d.eE+\-]+)", sec)
        run["fsky_floor_analytic"] = float(m.group(1)) if m else None
        m = re.search(r"Sky View Factor Statistics ---\s*"
                      r"\s*Min:\s+([\d.eE+\-]+)\s*"
                      r"\s*Max:\s+([\d.eE+\-]+)\s*"
                      r"\s*Avg:\s+([\d.eE+\-]+)", sec)
        if m:
            run["fsky_min"] = float(m.group(1))
            run["fsky_max"] = float(m.group(2))
            run["fsky_avg"] = float(m.group(3))
        runs.append(run)
    return runs


def make_figure(log_path: Path, out_path: Path) -> None:
    apply_style()
    runs = parse_log(log_path)
    print(f"  parsed {len(runs)} test runs from {log_path}")

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(DOUBLE_W * 1.4, 0.55 * DOUBLE_W * 1.4))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.93, bottom=0.14, wspace=0.24)

    # ---- (a) Ingersoll T_floor vs solar elevation ----
    cmap = {0.1: COLORS["lat0"], 0.2: COLORS["lat45"], 0.3: COLORS["bad"]}
    for dd in (0.1, 0.2, 0.3):
        beta_deg, T = floor_temperature_curve(dd)
        ax1.plot(beta_deg, T, color=cmap[dd], lw=2.4,
                 label=fr"$d/D = {dd}$")
        bcrit = float(np.rad2deg(critical_elevation(dd)))
        ax1.axvline(bcrit, color=cmap[dd], lw=0.8, ls=":")
    ax1.set_xlabel(r"Solar elevation $\beta$ (°)")
    ax1.set_ylabel(r"Floor equilibrium $T$ (K)")
    ax1.set_xlim(0, 90)
    ax1.set_ylim(50, 410)
    ax1.legend(loc="lower right", title="Ingersoll (1992)")
    panel_label(ax1, "(a)", x=0.04, y=0.95)

    # ---- (b) F_sky PTM3D vs analytic ----
    dd_runs = [r for r in runs if r.get("depth_to_diameter") == 0.2
               and r.get("fsky_avg") is not None]
    analytic_fsky = sky_view_factor(0.2)
    if dd_runs:
        labels = [fr"{int(r['latitude_deg'])}$^\circ$" for r in dd_runs]
        mins = np.array([r["fsky_min"] for r in dd_runs])
        maxs = np.array([r["fsky_max"] for r in dd_runs])
        avgs = np.array([r["fsky_avg"] for r in dd_runs])
        x = np.arange(len(dd_runs))

        ax2.errorbar(x, avgs, yerr=[avgs - mins, maxs - avgs],
                     fmt="o", color=COLORS["ptm"], mfc="white",
                     mec=COLORS["ptm"], capsize=5, lw=1.6, ms=10, mew=2.0,
                     label="PTM3D mesh (range)")
        ax2.axhline(analytic_fsky, color=COLORS["ref"], lw=1.6, ls="--",
                    label=fr"Analytic $\sin^2\alpha$")
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels)
        ax2.set_xlabel("Latitude")
        ax2.set_ylabel(r"Floor sky view factor $F_\mathrm{sky}$")
        ax2.set_ylim(0.855, 0.870)
        ax2.legend(loc="upper right")

        max_dev_pct = 100 * np.max(np.abs([mins - analytic_fsky,
                                            maxs - analytic_fsky])) / analytic_fsky
        print(f"  PTM3D F_sky max deviation from analytic: {max_dev_pct:.3f} %")
    panel_label(ax2, "(b)", x=0.04, y=0.95)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"  wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    # Default to a TRACKED artifact so Fig 7 is reproducible (was /tmp, which is
    # cleared on reboot and lost the published numbers). Generate the log with:
    #   build/bin/test_bowl_crater > output/bowl_conv/bowl_crater_full.log
    p.add_argument("log", nargs="?",
                   default="output/bowl_conv/bowl_crater_full.log")
    p.add_argument("--out", default="docs/figures/v2/bowl_crater_ingersoll.png")
    args = p.parse_args()
    make_figure(Path(args.log), Path(args.out))


if __name__ == "__main__":
    main()
