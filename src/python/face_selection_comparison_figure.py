#!/usr/bin/env python3
"""
F10 — Physics-based vs legacy face-selection comparison for the radiation
coupling loop.

Reads two ``ptm_thermal_sim --rad-verbose`` logs (one with
``--rad-coupling-select-mode physics`` and one with ``--rad-coupling-select-mode
legacy``) and plots:

  (a) selected-face %% per Picard iter for both selectors;
  (b) wall-clock per iter (radiosity + re-thermal), bar-grouped.

Companion to §6.4 of the PTM3D methods paper.  The default input pair is
the Amundsen-summer rad-coupling validation runs:

  tmp/amundsen_face_sel_physics.log   (--rad-coupling-select-mode physics)
  tmp/amundsen_face_sel_legacy.log    (--rad-coupling-select-mode legacy)
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from ptm_pubstyle import apply_style, COLORS, DOUBLE_W, panel_label


_RE_ITER       = re.compile(r"^--- Radiation Iteration\s+(\d+)\s*---")
_RE_SELECT_PHY = re.compile(
    r"Faces above predicted .T threshold\s*\([\d.e+\-]+\s*K\):\s*"
    r"(\d+)\s*/\s*(\d+)\s*\(frozen:\s*(\d+)\)"
)
_RE_SELECT_LEG = re.compile(
    # Legacy selector emits "Faces with significant indirect flux: N / N"
    r"Faces with significant indirect flux:\s*(\d+)\s*/\s*(\d+)"
)
_RE_RE_THERMAL = re.compile(r"Re-running thermal model for\s+\d+\s+faces")
_RE_DT_MAX     = re.compile(r"Max temperature change:\s*([\d.e+\-]+)\s*K")
_RE_FMOVING    = re.compile(r"Faces still changing:\s*([\d.e+\-]+)%")
_RE_CONVERGE   = re.compile(r"Converged on")
_RE_INDIR_DONE = re.compile(r"Computing indirect flux \[==+\] 100%\s*\|\s*elapsed\s+([\dms\s]+)")
_RE_BATCH_DONE = re.compile(r"\[GPU Thermal Pool\] (\d+)/(\d+) re-thermal batches done")
_RE_SELMODE    = re.compile(r"Face-selection mode:\s*(\w+)")


def _parse_elapsed(s: str) -> float:
    """Parse '2m 35s' or '85s' style strings to seconds."""
    s = s.strip()
    parts = re.findall(r"(\d+)\s*([smh])", s)
    secs = 0.0
    for v, u in parts:
        v = float(v)
        if u == "h": secs += 3600 * v
        elif u == "m": secs += 60 * v
        else: secs += v
    return secs


def parse_log(path: Path) -> dict:
    """Extract per-iter trace from a ptm_thermal_sim log."""
    text = path.read_text(errors="replace")
    iters: list[dict] = []
    current: dict | None = None
    sel_mode = None
    for line in text.splitlines():
        if m := _RE_SELMODE.search(line):
            sel_mode = m.group(1)
        if m := _RE_ITER.match(line):
            if current is not None:
                iters.append(current)
            current = {"k": int(m.group(1)), "converged": False}
            continue
        if current is None:
            continue
        if m := _RE_SELECT_PHY.search(line):
            current["selected"] = int(m.group(1))
            current["total"]    = int(m.group(2))
            current["frozen"]   = int(m.group(3))
        elif m := _RE_SELECT_LEG.search(line):
            current["selected"] = int(m.group(1))
            current["total"]    = int(m.group(2))
            current["legacy_mode"] = True
        elif m := _RE_DT_MAX.search(line):
            current["dT_max"] = float(m.group(1))
        elif m := _RE_FMOVING.search(line):
            current["f_moving_pct"] = float(m.group(1))
        elif m := _RE_INDIR_DONE.search(line):
            current["radiosity_s"] = _parse_elapsed(m.group(1))
        elif _RE_CONVERGE.search(line):
            current["converged"] = True
    if current is not None:
        iters.append(current)
    return {"select_mode": sel_mode, "iters": iters}


def _bar_at(ax, x, h, w, color, **kw):
    return ax.bar(x, h, width=w, color=color, edgecolor="black",
                  linewidth=0.7, **kw)


def make_figure(physics_log: Path, legacy_log: Path, out_path: Path,
                site_label: str = "Amundsen crater (PSR, 4 M faces)") -> None:
    apply_style()
    info_phy = parse_log(physics_log)
    info_leg = parse_log(legacy_log)
    iters_phy = info_phy["iters"]
    iters_leg = info_leg["iters"]
    if not iters_phy or not iters_leg:
        raise SystemExit(
            f"Failed to parse iterations: physics={len(iters_phy)}, legacy={len(iters_leg)}"
        )

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_W * 1.05, 0.55 * DOUBLE_W))
    fig.subplots_adjust(left=0.11, right=0.97, top=0.84,
                        bottom=0.15, wspace=0.30)

    # ----- (a) Selected fraction per iter -----------------------------------
    ax = axes[0]
    k_phy = np.array([it["k"] for it in iters_phy if "selected" in it])
    k_leg = np.array([it["k"] for it in iters_leg if "selected" in it])
    sel_phy = 100.0 * np.array(
        [it["selected"] / it["total"] for it in iters_phy if "selected" in it])
    sel_leg = 100.0 * np.array(
        [it["selected"] / it["total"] for it in iters_leg if "selected" in it])
    w = 0.38
    _bar_at(ax, k_leg - 0.5 * w, sel_leg, w, "#bbbbbb",
            label=f"legacy (flux threshold)")
    _bar_at(ax, k_phy + 0.5 * w, sel_phy, w, COLORS["ptm"],
            label=f"physics (predicted ΔT)")
    for kx, sv in zip(k_leg, sel_leg):
        ax.text(kx - 0.5 * w, max(sv * 1.10, 0.02),
                f"{sv:.1f}%", ha="center", va="bottom", fontsize=9,
                color="#444")
    for kx, sv in zip(k_phy, sel_phy):
        ax.text(kx + 0.5 * w, max(sv * 1.10, 0.02),
                f"{sv:.2f}%", ha="center", va="bottom", fontsize=9,
                color=COLORS["ptm"])
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 600.0)
    ax.set_xticks(sorted(set(np.concatenate([k_phy, k_leg]))))
    ax.set_xlabel(r"Picard iteration $k$")
    ax.set_ylabel("Faces selected for re-thermal (% of mesh)")
    ax.legend(loc="lower left", fontsize=9, framealpha=0.0)
    panel_label(ax, "(a)", x=0.04, y=0.96)

    # ----- (b) Convergence ΔT per iter --------------------------------------
    ax = axes[1]
    k_phy_dt = np.array([it["k"] for it in iters_phy if "dT_max" in it])
    k_leg_dt = np.array([it["k"] for it in iters_leg if "dT_max" in it])
    dt_phy = np.array([it["dT_max"] for it in iters_phy if "dT_max" in it])
    dt_leg = np.array([it["dT_max"] for it in iters_leg if "dT_max" in it])
    if len(k_leg_dt):
        ax.plot(k_leg_dt, dt_leg, "o--", color="#888888", lw=1.8,
                mfc="white", mew=1.8, ms=8, label="legacy")
    if len(k_phy_dt):
        ax.plot(k_phy_dt, dt_phy, "s-", color=COLORS["ptm"], lw=2.2,
                mfc="white", mew=2.0, ms=9, label="physics")
        for kx, dv in zip(k_phy_dt, dt_phy):
            ax.annotate(f"{dv:.0f} K", (kx, dv), xytext=(9, 4),
                        textcoords="offset points", fontsize=10,
                        color=COLORS["ptm"])
    # τ_T = 0.5 K threshold
    ax.axhline(0.5, color=COLORS["bad"], lw=1.4, ls="--",
               label=r"$\tau_T = 0.5\,$K")
    ax.set_yscale("log")
    all_k = sorted(set(np.concatenate([k_phy, k_leg])))
    ax.set_xticks(all_k)
    ax.set_xlim(min(all_k) - 0.5, max(all_k) + 0.5)
    ax.set_ylim(0.1, max(800.0, max([dt_phy.max() if len(dt_phy) else 1.0,
                                      dt_leg.max() if len(dt_leg) else 1.0]) * 2.5))
    ax.set_xlabel(r"Picard iteration $k$")
    ax.set_ylabel(r"max $|\Delta T^{(k)}|$ (K)")
    ax.legend(loc="lower left", fontsize=9, framealpha=0.0)
    panel_label(ax, "(b)", x=0.04, y=0.96)

    fig.suptitle(f"Face-selection: physics vs legacy — {site_label}",
                 y=0.95, fontsize=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)

    print(f"\nPhysics:  {len(iters_phy)} iters, mode={info_phy['select_mode']}")
    for it in iters_phy:
        sel_str = (f"{100.0 * it['selected'] / max(1, it['total']):>6.2f}%"
                   if "selected" in it else "  ----")
        dt_str  = f"{it.get('dT_max', float('nan')):>6.2f} K"
        fl = " conv" if it.get("converged") else ""
        print(f"  k={it['k']:>2}  sel={sel_str}  dT={dt_str}{fl}")
    print(f"\nLegacy:   {len(iters_leg)} iters, mode={info_leg['select_mode']}")
    for it in iters_leg:
        sel_str = (f"{100.0 * it['selected'] / max(1, it['total']):>6.2f}%"
                   if "selected" in it else "  ----")
        dt_str  = f"{it.get('dT_max', float('nan')):>6.2f} K"
        fl = " conv" if it.get("converged") else ""
        print(f"  k={it['k']:>2}  sel={sel_str}  dT={dt_str}{fl}")
    print(f"\nWrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--physics-log", type=Path,
                   default=Path("tmp/amundsen_face_sel_physics.log"))
    p.add_argument("--legacy-log", type=Path,
                   default=Path("tmp/amundsen_face_sel_legacy.log"))
    p.add_argument("--out", type=Path,
                   default=Path("docs/figures/v2/face_selection_comparison.png"))
    p.add_argument("--site-label", default="Amundsen crater (PSR, 4 M faces)")
    args = p.parse_args()
    make_figure(args.physics_log, args.legacy_log, args.out, args.site_label)


if __name__ == "__main__":
    main()
