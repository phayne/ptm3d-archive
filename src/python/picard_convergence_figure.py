#!/usr/bin/env python3
"""
F9 — Picard-iteration convergence diagram for the thermal/radiation coupling.

Reads the per-iter trace from a production ``ptm_thermal_sim`` log
(e.g. ``tmp/corering_val_noring.log``) and plots:

  (a) selected-face %% per Picard iter k (linearised-predictor +
      observed-ΔT union), with the f_min termination floor drawn as a
      horizontal reference;
  (b) max face-wise ΔT per iter on a log-y axis, with the τ_T temperature
      tolerance drawn as a horizontal reference.

Companion to Table 1 in §6.3 of the PTM3D methods paper. The default
input/output reproduces the A03MM022 lunar production-window data cited
in the paper.
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
from ptm_pubstyle import apply_style, COLORS, COLUMN_W, DOUBLE_W, panel_label


# ---------------------------------------------------------------------------
# Log parsing

_RE_ITER     = re.compile(r"^--- Radiation Iteration\s+(\d+)\s*---")
_RE_SELECT   = re.compile(
    r"Faces above predicted .T threshold\s*\(([\d.e+\-]+)\s*K\):\s*"
    r"(\d+)\s*/\s*(\d+)\s*\(frozen:\s*(\d+)\)"
)
_RE_DT_MAX   = re.compile(r"Max temperature change:\s*([\d.e+\-]+)\s*K")
_RE_FMOVING  = re.compile(r"Faces still changing:\s*([\d.e+\-]+)%")
_RE_CONVERGE = re.compile(r"Converged on")
_RE_THRESH   = re.compile(r"Temperature tolerance:\s*([\d.e+\-]+)\s*K")
_RE_FFLOOR   = re.compile(r"face[- ]fraction criterion \(([\d.e+\-]+)%\s*<\s*([\d.e+\-]+)%\)")
_RE_SELMODE  = re.compile(r"Face-selection mode:\s*(\w+)")


def parse_log(path: Path) -> dict:
    """Extract per-iter (selected, ΔT_max, f_moving, converged) from a log."""
    text = path.read_text(errors="replace")

    iters: list[dict] = []
    current: dict | None = None
    tau_T = 0.5
    f_min_pct = 1.0
    sel_mode = "physics"

    for line in text.splitlines():
        if m := _RE_THRESH.search(line):
            tau_T = float(m.group(1))
        if m := _RE_SELMODE.search(line):
            sel_mode = m.group(1)
        if m := _RE_ITER.match(line):
            if current is not None:
                iters.append(current)
            current = {"k": int(m.group(1)), "converged": False}
            continue
        if current is None:
            continue
        if m := _RE_SELECT.search(line):
            current.setdefault("select_thresh_K", float(m.group(1)))
            current["selected"] = int(m.group(2))
            current["total"]    = int(m.group(3))
            current["frozen"]   = int(m.group(4))
        elif m := _RE_DT_MAX.search(line):
            current["dT_max"] = float(m.group(1))
        elif m := _RE_FMOVING.search(line):
            current["f_moving_pct"] = float(m.group(1))
        elif _RE_CONVERGE.search(line):
            current["converged"] = True
            if m := _RE_FFLOOR.search(line):
                f_min_pct = float(m.group(2))

    if current is not None:
        iters.append(current)

    return {
        "tau_T": tau_T,
        "f_min_pct": f_min_pct,
        "select_mode": sel_mode,
        "iters": iters,
    }


# ---------------------------------------------------------------------------
# Plot

def make_figure(log_path: Path, out_path: Path,
                site_label: str = "A03MM022 production window") -> None:
    info = parse_log(log_path)
    iters = info["iters"]
    if not iters:
        raise SystemExit(f"No Picard iterations parsed from {log_path}")

    apply_style()

    ks       = np.array([it["k"]                        for it in iters])
    sel_pct  = np.array([100.0 * it["selected"] / it["total"]
                         for it in iters if "selected" in it])
    sel_ks   = np.array([it["k"] for it in iters if "selected" in it])
    dT       = np.array([it.get("dT_max", np.nan)       for it in iters], dtype=float)
    fmov_pct = np.array([it.get("f_moving_pct", np.nan) for it in iters], dtype=float)
    converged_k = next((it["k"] for it in iters if it.get("converged")), None)

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_W, 0.5 * DOUBLE_W))
    fig.subplots_adjust(left=0.10, right=0.97, top=0.85,
                        bottom=0.16, wspace=0.32)

    # ----- (a) Selected fraction per iter -----------------------------------
    ax = axes[0]
    width = 0.34
    pos_sel  = sel_ks - 0.5 * width
    pos_mov  = ks    + 0.5 * width
    fmov_plot = np.where(np.isfinite(fmov_pct), fmov_pct, 0.0)

    ax.bar(pos_sel, sel_pct, width=width, color=COLORS["ptm"],
           edgecolor="black", linewidth=0.8,
           label="Selected for re-thermal")
    ax.bar(pos_mov, fmov_plot, width=width, color="#bbbbbb",
           edgecolor="black", linewidth=0.8,
           label=r"$f_\mathrm{moving}$ (observed)")
    for k, fp in zip(ks, fmov_pct):
        if not np.isfinite(fp):
            ax.text(k + 0.5 * width, 0.03, "(none)",
                    ha="center", va="bottom",
                    color=COLORS["neutral"], fontsize=9)

    ax.axhline(info["f_min_pct"], color=COLORS["bad"],
               lw=1.4, ls="--",
               label=fr"$f_{{\min}} = {info['f_min_pct']:.2f}\%$")

    ax.set_yscale("log")
    ax.set_ylim(1e-3, 200.0)
    ax.set_xticks(ks)
    ax.set_xlabel(r"Picard iteration $k$")
    ax.set_ylabel("Faces (% of mesh)")
    ax.legend(loc="lower left", fontsize=9, framealpha=0.0)

    for k, sp in zip(sel_ks, sel_pct):
        ax.text(k - 0.5 * width, sp * 1.25,
                f"{sp:.2f}%", ha="center", va="bottom", fontsize=9,
                color=COLORS["ptm"])

    # ----- (b) ΔT_max per iter ----------------------------------------------
    ax = axes[1]
    valid = np.isfinite(dT)
    ax.plot(ks[valid], dT[valid], "o-",
            color=COLORS["ptm"], lw=2.2, mfc="white",
            mew=2.0, ms=9, label=r"$\Delta T_\mathrm{max}^{(k)}$")
    for k, dtv in zip(ks[valid], dT[valid]):
        ax.annotate(f"{dtv:.1f} K", (k, dtv), xytext=(9, 4),
                    textcoords="offset points", fontsize=10,
                    color=COLORS["ptm"])

    ax.axhline(info["tau_T"], color=COLORS["bad"],
               lw=1.4, ls="--",
               label=fr"$\tau_T = {info['tau_T']:.1f}\,$K")
    ax.set_yscale("log")
    ax.set_xticks(ks)
    ax.set_xlim(ks.min() - 0.5, ks.max() + 0.5)
    ax.set_ylim(0.05, max(500.0, np.nanmax(dT) * 2.5))
    ax.set_xlabel(r"Picard iteration $k$")
    ax.set_ylabel(r"max $|\Delta T^{(k)}|$ (K)")
    ax.legend(loc="lower left", fontsize=10, framealpha=0.0)

    if converged_k is not None:
        for a in axes:
            a.axvline(converged_k, color=COLORS["good"], lw=1.2, ls=":")
            # place text just above the x-axis so it doesn't collide with data
            ymin, ymax = a.get_ylim()
            y_text = np.exp(np.log(ymin) + 0.06 * (np.log(ymax) - np.log(ymin)))
            a.text(converged_k - 0.03, y_text,
                   "converged",
                   color=COLORS["good"], rotation=90,
                   ha="right", va="bottom", fontsize=9)

    panel_label(axes[0], "(a)", x=0.04, y=0.96)
    panel_label(axes[1], "(b)", x=0.04, y=0.96)

    fig.suptitle(
        f"Picard convergence — {site_label} "
        f"(selector: {info['select_mode']})",
        y=0.97, fontsize=12,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"  parsed {len(iters)} iters from {log_path.name}")
    for it in iters:
        sel_str = f"{100.0 * it.get('selected', 0) / max(1, it.get('total', 1)):>6.2f}%" \
            if "selected" in it else "  ----"
        dt_str = f"{it.get('dT_max', float('nan')):>6.2f} K" \
            if "dT_max" in it else "    --"
        fm_str = f"{it.get('f_moving_pct', float('nan')):>6.3f}%" \
            if "f_moving_pct" in it else "    --"
        flag = " converged" if it.get("converged") else ""
        print(f"    k={it['k']:>2}  selected={sel_str}  dT={dt_str}"
              f"  f_moving={fm_str}{flag}")
    print(f"  wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", type=Path,
                   default=Path("tmp/corering_val_noring.log"))
    p.add_argument("--out", type=Path,
                   default=Path("docs/figures/v2/picard_convergence.png"))
    p.add_argument("--site-label", default="A03MM022 production window")
    args = p.parse_args()
    make_figure(args.log, args.out, args.site_label)


if __name__ == "__main__":
    main()
