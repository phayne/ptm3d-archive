#!/usr/bin/env python3
"""Before/after figure for the PTM-36 planetary-shadow (eclipse) units fix.

Small multiples: one column per body, one row per measure (never a dual axis).
Series colors are the validated categorical slots 1 (blue) and 2 (orange) of the
crew's default palette, assigned in fixed order: slot 1 = current main (BEFORE),
slot 2 = PTM-36 fix (AFTER).

    python scripts/plot_eclipse_before_after.py --out output/eclipse_diag/eclipse_before_after.png
"""

import argparse
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ptm_eclipse_diag import (  # noqa: E402
    read_body_positions, umbra_geometry, read_ply_face_latlon, read_series,
    PRIMARY_RADIUS_KM)

BEFORE_C = "#2a78d6"   # categorical slot 1
AFTER_C = "#eb6834"    # categorical slot 2
MUTED = "#898781"
GRID = "#e1e0d9"
INK = "#0b0b0b"
INK2 = "#52514e"

PANELS = [
    dict(title="Europa — sub-Jovian point (Jupiter's shadow)",
         before="output/europa_global_res07",
         after="output/eclipse_diag/europa_res7_AFTER",
         primary="JUPITER", lat=0.0, lon=0.0),
    dict(title="Moon — sub-Earth point (Earth's shadow, 2026-03-03)",
         before="output/eclipse_diag/moon_res5_BEFORE",
         after="output/eclipse_diag/moon_res5_AFTER",
         primary="EARTH", lat=0.0, lon=0.0),
]


def elapsed_hours(out_dir, n):
    path = os.path.join(out_dir, "time_steps.txt")
    if os.path.exists(path):
        secs = []
        for line in open(path):
            if line.startswith("#"):
                continue
            parts = line.split()
            if parts:
                try:
                    secs.append(float(parts[0]))
                except ValueError:
                    pass
        if len(secs) == n:
            h = np.array(secs) / 3600.0
            return h - h[0]   # rebase: column 1 counts from the equilibration start
    return np.arange(n, dtype=float)


def load(panel, repo):
    after = os.path.join(repo, panel["after"])
    before = os.path.join(repo, panel["before"])
    utc, sun, pri = read_body_positions(os.path.join(after, "body_positions.csv"))
    n = len(utc)
    b, proj = umbra_geometry(sun, pri)
    rp = PRIMARY_RADIUS_KM[panel["primary"]]
    umbra = (b < rp) & (proj > 0.0)
    lat, lon = read_ply_face_latlon(os.path.join(after, "mesh.ply"))
    dlon = np.abs(((lon - panel["lon"] + 180.0) % 360.0) - 180.0)
    face = int(np.argmin((lat - panel["lat"]) ** 2 + dlon ** 2))
    out = dict(n=n, hours=elapsed_hours(after, n), umbra=umbra, face=face,
               lat=lat[face], lon=lon[face])
    for tag, d in (("before", before), ("after", after)):
        out[tag + "_ds"] = read_series(d, "direct_solar", face, n)
        out[tag + "_T"] = read_series(d, "temperature", face, n)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/Users/paha3326/Projects/ptm")
    ap.add_argument("--out", default="output/eclipse_diag/eclipse_before_after.png")
    args = ap.parse_args()

    data = [load(p, args.repo) for p in PANELS]

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 6.4))
    fig.patch.set_facecolor("#fcfcfb")

    for col, (panel, d) in enumerate(zip(PANELS, data)):
        idx = np.flatnonzero(d["umbra"])
        h = d["hours"]
        if len(idx):
            dh = np.median(np.diff(h)) if len(h) > 1 else 1.0
            span = (h[idx[0]] - 0.5 * dh, h[idx[-1]] + 0.5 * dh)
        else:
            span = None

        for row, (key, label) in enumerate((("ds", "Direct solar [W m$^{-2}$]"),
                                            ("T", "Surface temperature [K]"))):
            ax = axes[row][col]
            ax.set_facecolor("#fcfcfb")
            if span:
                ax.axvspan(span[0], span[1], color=GRID, alpha=0.85, lw=0, zorder=0)
            ax.plot(h, d["before_" + key], color=BEFORE_C, lw=2.0,
                    label="current main (no eclipse)", zorder=3)
            ax.plot(h, d["after_" + key], color=AFTER_C, lw=2.0,
                    label="PTM-36 fix", zorder=4)
            ax.set_ylabel(label, color=INK2, fontsize=9)
            ax.grid(True, color=GRID, lw=0.8, alpha=0.9)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color("#c3c2b7")
            ax.tick_params(colors=MUTED, labelsize=8)
            if row == 0:
                ax.set_title(panel["title"], color=INK, fontsize=10.5, pad=10)
            else:
                ax.set_xlabel("hours from start of output window", color=INK2,
                              fontsize=9)
                if span:
                    ax.annotate("geometric umbra",
                                xy=(0.5 * (span[0] + span[1]), ax.get_ylim()[0]),
                                xytext=(0, 8), textcoords="offset points",
                                ha="center", va="bottom", color=INK2, fontsize=8)

    axes[0][0].legend(loc="lower center", fontsize=8.5, frameon=False,
                      labelcolor=INK2, ncol=2)
    fig.suptitle("PTM-36 — the planetary shadow was 1000x too small "
                 "(km radius vs meter positions)",
                 color=INK, fontsize=12, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(args.repo, args.out) if not os.path.isabs(args.out) else args.out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    print("wrote", out)
    for panel, d in zip(PANELS, data):
        print("%-8s face %d (lat %.2f lon %.2f)  umbra steps %d  "
              "min direct_solar before %.2f -> after %.2f W/m2  "
              "min T before %.2f -> after %.2f K"
              % (panel["primary"], d["face"], d["lat"], d["lon"],
                 int(d["umbra"].sum()),
                 float(np.nanmin(d["before_ds"][d["umbra"]])),
                 float(np.nanmin(d["after_ds"][d["umbra"]])),
                 float(np.nanmin(d["before_T"][d["umbra"]])),
                 float(np.nanmin(d["after_T"][d["umbra"]]))))


if __name__ == "__main__":
    main()
