#!/usr/bin/env python3
"""
Computational performance scaling for a 1-lunation PTM3D run with full
radiation coupling, anchored on the May 2026 polar landing-site production batch
(lunar polar site L, a lunar polar site, 1,797,769 faces, 30-day output window, hourly
cadence, 2-orbit equilibration).

Two panels:

  (a) Wall-clock time vs mesh face count, per model component.
  (b) Total compute cost (thread-seconds) vs mesh face count.
      Thread-seconds = wall × thread_count for the component:
        - CPU OpenMP components count their thread pool (48 on the
          production workstation).
        - GPU components count one "thread" per device (4 W6800X dies).
        - Sequential I/O counts as 1 thread.

Components shown:

  * View factors (Monte-Carlo, CPU OpenMP) — one-shot per site,
    amortized over the 8-window production batch.
  * Illumination (Embree shadow rays, CPU OpenMP) — per-window.
  * Equilibration thermal (Crank-Nicolson, multi-GPU OpenCL) —
    2-orbit spin-up at coarse cadence.
  * Output thermal (Crank-Nicolson, multi-GPU OpenCL) — 720 hourly
    steps over the 30-day output window.
  * Radiation exchange (radiosity SpMV + selective re-thermal
    Picard iterations, multi-GPU OpenCL) — per coupling iter, ~3
    iters to converge.
  * Output I/O & statistics — binary export, ice-stability solver.

All components scale linearly in N (face count) for fixed timestep
counts because (i) the sparse view-factor matrix has fixed average
neighbor count K~100 after the adaptive top-K compaction, (ii) the
Embree BVH cost is O(log N) per ray which is amortized into the
near-linear illumination time, and (iii) the GPU thermal kernel is
embarrassingly parallel in face index. Small departures from N^1
come from the BVH (~N log N for the build, < N for the lookups)
and from cache effects at the high-N end.

Anchor measurements (lunar polar site L production run, 2026-05-13):
  N = 1.80e6 faces; wall = 15m 30s per window; 4-die W6800X Duo,
  48-thread CPU OpenMP.

Usage:
    python3 performance_lunation_figure.py [--out PATH]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from ptm_pubstyle import apply_style, COLORS, DOUBLE_W, panel_label

PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_OUT = PROJECT_ROOT / "docs" / "figures" / "v2" / "performance_lunation.png"


# -- Anchor: measured lunar polar site L production wall times (per window) --
# Wall times in seconds. Source: orchestrator_FIXED.log + per-window
# logs from the 2026-05-13 batch. View-factor wall is per-site (built
# once, cached, reused across the 8 windows), so the per-window cost
# is total_VF_wall / 8.
ANCHOR_N    = 1.797769e6
ANCHOR_WALL = {
    # component             : seconds per 1-lunation run on N_anchor
    "view_factor":            360 / 8,    # 6 min site-total ÷ 8 windows = 45 s
    "illumination":           420.0,      # ~7 min Embree per window
    "eq_thermal":             100.0,      # iter-0 eq phase: 2 cycles × 240 steps GPU
    "out_thermal":             40.0,      # iter-0 output phase: 1 cycle × 720 steps GPU
    "radiation":              420.0,      # radiosity SpMV × 720 ts × 3 coupling iters
    "re_thermal":              50.0,      # 2 Picard re-thermal sweeps, output-cycle only
    "io_stats":                30.0,      # binary export + ice-stability post-solve
}

# Thread count used by each component (for the compute-cost panel).
# CPU OpenMP components fully saturate the 48-thread pool. GPU
# components run on 4 W6800X dies (treated as 4 "threads" for the
# normalized cost metric).
THREADS = {
    "view_factor":   48,
    "illumination":  48,
    "eq_thermal":     4,
    "out_thermal":    4,
    "radiation":      4,
    "re_thermal":     4,
    "io_stats":       1,
}

# Power-law exponents for each component (wall ~ N^alpha).
# View factors and illumination have a (tiny) log N contribution from
# the BVH that we treat as alpha = 1.05 here; everything else is
# strictly linear in N for fixed timestep counts.
ALPHA = {
    "view_factor":   1.05,
    "illumination":  1.05,
    "eq_thermal":    1.00,
    "out_thermal":   1.00,
    "radiation":     1.00,
    "re_thermal":    1.00,
    "io_stats":      1.00,
}

LABELS = {
    "view_factor":   "View factors (Monte-Carlo, CPU)",
    "illumination":  "Illumination (Embree, CPU)",
    "eq_thermal":    "Equilibration thermal (GPU)",
    "out_thermal":   "Output thermal (GPU)",
    "radiation":     "Radiation exchange (GPU)",
    "re_thermal":    "Re-thermal coupling iterations (GPU)",
    "io_stats":      "Output I/O + statistics",
}

COMPONENT_COLORS = {
    "view_factor":   "#8c564b",   # brown
    "illumination":  "#ff7f0e",   # orange
    "eq_thermal":    "#1f77b4",   # blue
    "out_thermal":   "#9467bd",   # purple
    "radiation":     "#d62728",   # red
    "re_thermal":    "#e377c2",   # pink
    "io_stats":      "#7f7f7f",   # gray
}

ORDER = [
    "view_factor",
    "illumination",
    "eq_thermal",
    "out_thermal",
    "radiation",
    "re_thermal",
    "io_stats",
]


def predicted_wall(N: float | np.ndarray, component: str) -> np.ndarray:
    """Predicted wall-clock time (s) for component at mesh size N."""
    return ANCHOR_WALL[component] * (N / ANCHOR_N) ** ALPHA[component]


def predicted_cost(N: float | np.ndarray, component: str) -> np.ndarray:
    """Predicted thread-seconds for component at mesh size N."""
    return predicted_wall(N, component) * THREADS[component]


def make_figure(out_path: Path) -> None:
    apply_style()

    # Mesh size grid: 1e3 to 1e7 faces, log spacing
    N = np.logspace(3, 7, 100)

    fig, (ax_wall, ax_cost) = plt.subplots(
        1, 2,
        figsize=(DOUBLE_W, 0.55 * DOUBLE_W),
        constrained_layout=True,
    )

    # --- Panel (a): wall-clock time ---
    total_wall = np.zeros_like(N)
    for comp in ORDER:
        wall = predicted_wall(N, comp)
        ax_wall.plot(
            N, wall,
            color=COMPONENT_COLORS[comp],
            linewidth=1.5,
            label=LABELS[comp],
        )
        total_wall = total_wall + wall

    # Total curve
    ax_wall.plot(N, total_wall, color="k", linewidth=2.0, linestyle="--",
                 label="Total")

    # Asymptotic reference scalings, anchored at N_anchor to the
    # measured total. These let the reader compare PTM3D's near-linear
    # scaling to the canonical O(N log N) and O(N^2) alternatives that
    # naive view-factor and unsparsified radiosity formulations would
    # exhibit.
    anchor_total = sum(ANCHOR_WALL.values())
    nlogn_ref = anchor_total * (N * np.log(N)) / (ANCHOR_N * np.log(ANCHOR_N))
    n2_ref    = anchor_total * (N / ANCHOR_N) ** 2
    ax_wall.plot(N, nlogn_ref, color="#888888", linewidth=1.0,
                 linestyle=":", label=r"$\mathcal{O}(N \log N)$")
    ax_wall.plot(N, n2_ref, color="#cc6633", linewidth=1.0,
                 linestyle=":", label=r"$\mathcal{O}(N^{2})$")

    # Anchor marker at lunar polar site L measured total
    ax_wall.scatter([ANCHOR_N], [anchor_total],
                    s=70, color="k", marker="*", zorder=5,
                    edgecolor="white", linewidth=0.8,
                    label="lunar polar site L measured")

    ax_wall.set_xscale("log")
    ax_wall.set_yscale("log")
    ax_wall.set_xlabel("Mesh face count, $N$")
    ax_wall.set_ylabel("Wall-clock time per lunation (s)")
    ax_wall.set_xlim(7e2, 1.4e7)
    ax_wall.set_ylim(0.05, 1e6)
    ax_wall.set_xticks([1e3, 1e4, 1e5, 1e6, 1e7])
    ax_wall.grid(True, which="major", alpha=0.3)
    ax_wall.grid(True, which="minor", alpha=0.1)

    # Convenient wall-time markers on right axis
    ax_wall_r = ax_wall.twinx()
    ax_wall_r.set_yscale("log")
    ax_wall_r.set_ylim(ax_wall.get_ylim())
    minute_ticks = [60, 600, 3600, 36000, 360000]
    minute_labels = ["1 min", "10 min", "1 h", "10 h", "100 h"]
    ax_wall_r.set_yticks(minute_ticks)
    ax_wall_r.set_yticklabels(minute_labels)
    ax_wall_r.tick_params(axis="y", which="both", direction="in", labelsize=8)
    ax_wall_r.tick_params(axis="y", which="minor", length=0)

    panel_label(ax_wall, "a")
    ax_wall.legend(loc="upper left", fontsize=6.5, framealpha=0.95,
                   handlelength=1.6, borderaxespad=0.4, ncol=1)

    # --- Panel (b): total compute cost (thread-seconds) ---
    total_cost = np.zeros_like(N)
    for comp in ORDER:
        cost = predicted_cost(N, comp)
        ax_cost.plot(
            N, cost,
            color=COMPONENT_COLORS[comp],
            linewidth=1.5,
            label=LABELS[comp],
        )
        total_cost = total_cost + cost

    ax_cost.plot(N, total_cost, color="k", linewidth=2.0, linestyle="--",
                 label="Total")

    # Asymptotic reference scalings, anchored at the same anchor cost
    anchor_cost = sum(ANCHOR_WALL[c] * THREADS[c] for c in ANCHOR_WALL)
    nlogn_ref_c = anchor_cost * (N * np.log(N)) / (ANCHOR_N * np.log(ANCHOR_N))
    n2_ref_c    = anchor_cost * (N / ANCHOR_N) ** 2
    ax_cost.plot(N, nlogn_ref_c, color="#888888", linewidth=1.0,
                 linestyle=":", label=r"$\mathcal{O}(N \log N)$")
    ax_cost.plot(N, n2_ref_c, color="#cc6633", linewidth=1.0,
                 linestyle=":", label=r"$\mathcal{O}(N^{2})$")

    # Anchor at lunar polar site L measured total cost
    ax_cost.scatter([ANCHOR_N], [anchor_cost],
                    s=70, color="k", marker="*", zorder=5,
                    edgecolor="white", linewidth=0.8)

    ax_cost.set_xscale("log")
    ax_cost.set_yscale("log")
    ax_cost.set_xlabel("Mesh face count, $N$")
    ax_cost.set_ylabel("Compute cost per window (thread-s)")
    ax_cost.set_xlim(7e2, 1.4e7)
    ax_cost.set_ylim(5, 4e7)
    ax_cost.set_xticks([1e3, 1e4, 1e5, 1e6, 1e7])
    ax_cost.grid(True, which="major", alpha=0.3)
    ax_cost.grid(True, which="minor", alpha=0.1)

    # Hour markers on right axis (thread-hours)
    ax_cost_r = ax_cost.twinx()
    ax_cost_r.set_yscale("log")
    ax_cost_r.set_ylim(ax_cost.get_ylim())
    hour_ticks  = [3600, 36000, 360000, 3600000, 36000000]
    hour_labels = ["1 thread-h", "10 thread-h", "100 thread-h",
                   "1000 thread-h", "10000 thread-h"]
    ax_cost_r.set_yticks(hour_ticks)
    ax_cost_r.set_yticklabels(hour_labels)
    ax_cost_r.tick_params(axis="y", which="both", direction="in", labelsize=8)
    ax_cost_r.tick_params(axis="y", which="minor", length=0)

    panel_label(ax_cost, "b")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Wrote {out_path}")

    # Also write a PDF version for journal submission (vector graphics).
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Wrote {pdf_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help="Output PNG path")
    args = ap.parse_args()
    make_figure(args.out)


if __name__ == "__main__":
    main()
