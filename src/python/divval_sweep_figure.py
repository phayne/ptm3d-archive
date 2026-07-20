#!/usr/bin/env python3
"""Phase C attribution figure for the PSR cold-bias sweep.

Reads docs/figures/diviner_validation/sweep_haworth.json (written by
scripts/divval_sweep.py) and produces:
  (1) a printed table of Haworth PSR bias/RMSE vs each swept knob, and
  (2) a two-panel response figure: PSR bias vs albedo (indirect-magnitude axis)
      and PSR bias vs thermal inertia (k-scaling relative factor + constant-TI
      anchors). A horizontal line marks zero bias (perfect agreement) and the
      baseline is annotated.

The sign and slope of each curve attribute the cold bias: a flat/wrong-signed
TI curve rules TI out; an albedo curve crossing zero identifies the indirect
channel and the corrected value.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from ptm_pubstyle import apply_style

# Baseline (committed run + rebuilt-binary no-op): albedo 0.12, depth-dependent
# Hayne TI (k-scale factor 1.0).
BASELINE_ALBEDO = 0.12

# tag -> (axis, x-value, x-label-note)
ALBEDO = {"alb008": 0.08, "alb018": 0.18, "alb025": 0.25}
KSCALE = {"ks0p5": 0.5, "ks2": 2.0, "ks4": 4.0}       # TI ~ sqrt(k); factor on TI
CONST_TI = {"ti40": 40.0, "ti100": 100.0}             # absolute SI
EQORB = {"base_repro": 2.0, "eq4": 4.0, "eq6": 6.0, "eq8": 8.0}  # equilibration orbits


def getbias(r, tag):
    v = r.get(tag, {})
    b = v.get("PSR_bias")
    rm = v.get("PSR_rmse")
    return (float(b) if b not in (None, "?") else np.nan,
            float(rm) if rm not in (None, "?") else np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="docs/figures/diviner_validation/sweep_haworth.json")
    ap.add_argument("--out", default="docs/figures/diviner_validation/sweep_attribution.png")
    a = ap.parse_args()
    res = json.loads(Path(a.json).read_text())

    base_b, base_rm = getbias(res, "base_repro")
    if np.isnan(base_b):
        base_b, base_rm = -8.7, 12.3  # committed baseline fallback

    # ---- table ----
    print(f"{'tag':12s} {'knob':32s} {'PSR bias':>9s} {'RMSE':>7s} {'wall':>7s}")
    order = ["base_repro", "eq4", "alb008", "alb018", "alb025",
             "ti40", "ti100", "ks0p5", "ks2", "ks4", "emis090"]
    for tag in order:
        if tag not in res:
            continue
        r = res[tag]
        b, rm = getbias(res, tag)
        print(f"{tag:12s} {r.get('desc','?'):32s} {b:+9.1f} {rm:7.1f} "
              f"{str(r.get('wall_s','?')):>7s}")

    apply_style()
    fig, (axA, axT, axE) = plt.subplots(1, 3, figsize=(12.0, 3.7))

    # ---- albedo panel ----
    xs = [BASELINE_ALBEDO] + [ALBEDO[t] for t in ALBEDO if t in res]
    ys = [base_b] + [getbias(res, t)[0] for t in ALBEDO if t in res]
    es = [base_rm] + [getbias(res, t)[1] for t in ALBEDO if t in res]
    o = np.argsort(xs)
    xs, ys, es = np.array(xs)[o], np.array(ys)[o], np.array(es)[o]
    axA.axhline(0, color="0.5", lw=1, ls="--")
    axA.errorbar(xs, ys, yerr=es, marker="o", capsize=3, color="C0")
    axA.scatter([BASELINE_ALBEDO], [base_b], color="k", zorder=5, label="baseline (0.12)")
    axA.set_xlabel("Bond albedo $A_0$")
    axA.set_ylabel("Haworth PSR bias (model - Diviner), K")
    axA.set_title("Indirect-magnitude axis (albedo)")
    axA.legend(fontsize=8)

    # ---- thermal-inertia panel ----
    axT.axhline(0, color="0.5", lw=1, ls="--")
    # k-scale family (relative TI factor; baseline=1.0)
    kx = [1.0] + [KSCALE[t] for t in KSCALE if t in res]
    ky = [base_b] + [getbias(res, t)[0] for t in KSCALE if t in res]
    o = np.argsort(kx)
    kx, ky = np.array(kx)[o], np.array(ky)[o]
    axT.plot(kx, ky, marker="s", color="C1", label="k-scaling (rel. TI)")
    axT.scatter([1.0], [base_b], color="k", zorder=5)
    # constant-TI anchors (absolute SI) on a twin top axis for context
    if any(t in res for t in CONST_TI):
        axT2 = axT.twiny()
        cx = [CONST_TI[t] for t in CONST_TI if t in res]
        cy = [getbias(res, t)[0] for t in CONST_TI if t in res]
        axT2.plot(cx, cy, marker="^", ls=":", color="C2", label="constant TI (SI)")
        axT2.set_xlabel("constant TI, SI", color="C2", fontsize=9)
        axT2.tick_params(axis="x", labelcolor="C2")
        lines = axT.get_lines()[1:] + axT2.get_lines()
        axT.legend(lines, [l.get_label() for l in lines], fontsize=8)
    axT.set_xlabel("relative thermal inertia (k-scaling, baseline=1)")
    axT.set_title("Thermal-inertia axis")

    # ---- equilibration panel (the smoking gun) ----
    ex = sorted(EQORB.values())
    tag_by_orb = {v: k for k, v in EQORB.items()}
    ey = [getbias(res, tag_by_orb[o])[0] for o in ex]
    axE.axhline(0, color="0.5", lw=1, ls="--")
    axE.plot(ex, ey, marker="D", color="C3")
    axE.scatter([2.0], [base_b], color="k", zorder=5, label="production default (2)")
    axE.set_xlabel("equilibration orbits (no indirect flux)")
    axE.set_title("Equilibration axis (root cause)")
    axE.legend(fontsize=8)
    axE.annotate("deep column driven colder\nas it settles w/o wall heating",
                 xy=(ex[-1], ey[-1]), xytext=(0.4, 0.25), textcoords="axes fraction",
                 fontsize=8, ha="left",
                 arrowprops=dict(arrowstyle="->", color="C3", lw=1))

    fig.suptitle("PSR cold-bias attribution at Haworth — properties flat, "
                 "equilibration (indirect-excluded) is the cause", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
