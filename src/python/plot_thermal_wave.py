#!/usr/bin/env python3
"""
F2 — Lunar subsurface thermal-wave (two-panel).

  (a) T(z) snapshots at six local times spanning one diurnal cycle
  (b) Depth-time heatmap of T(z, t)

Depth axis is auto-truncated at the actual solver bottom (the location-export
grid extends to 1 m but the solver only runs to b·δ).
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

sys.path.insert(0, str(Path(__file__).parent))
from ptm_pubstyle import apply_style, COLORS, DOUBLE_W, panel_label


LUNAR_DAY_S = 29.530588 * 86400.0


def load_depths(path: Path) -> np.ndarray:
    return np.loadtxt(path, comments="#")


def load_depth_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    times = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("time"):
                continue
            parts = line.split(",")
            stamp = parts[0]
            try:
                dt = datetime.fromisoformat(stamp)
            except ValueError:
                dt = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
            times.append(dt)
            rows.append([float(x) for x in parts[1:]])
    t0 = times[0]
    sec = np.array([(t - t0).total_seconds() for t in times])
    T = np.array(rows, dtype=np.float64)
    return sec, T


def detect_solver_bottom(depths: np.ndarray, amp: np.ndarray,
                         tol_K: float = 0.05) -> int:
    """Return the deepest non-extrapolated depth index (where amp != flat-line)."""
    n = len(amp)
    cut = n - 1
    for i in range(n - 2, 0, -1):
        if abs(amp[i] - amp[i + 1]) > tol_K:
            cut = i + 1
            break
    return cut


def make_figure(times_s: np.ndarray,
                T: np.ndarray,
                depths: np.ndarray,
                out_path: Path) -> None:
    apply_style()

    # Last lunation, equilibrated
    cycle_start = times_s[-1] - LUNAR_DAY_S
    mask = times_s >= cycle_start
    t_cycle = times_s[mask] - cycle_start
    T_cycle = T[mask]

    # Local hour: noon = peak surface T
    surf = T_cycle[:, 0]
    peak_idx = int(np.argmax(surf))
    t_noon = t_cycle[peak_idx]
    hours = ((t_cycle - t_noon) / LUNAR_DAY_S * 24.0 + 12.0) % 24.0
    order = np.argsort(hours)
    hours_s = hours[order]
    T_s = T_cycle[order]

    # Truncate to solver bottom
    cycle_amp = T_s.max(axis=0) - T_s.min(axis=0)
    bottom_idx = detect_solver_bottom(depths, cycle_amp)
    z_bottom = depths[bottom_idx]
    print(f"  detected solver bottom at z ≈ {z_bottom * 100:.1f} cm")
    depths = depths[: bottom_idx + 1]
    T_s = T_s[:, : bottom_idx + 1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_W * 1.4, 0.6 * DOUBLE_W * 1.4))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.14, wspace=0.28)

    # ---- (a) T(z) snapshots ----
    snap_hours = [0, 4, 8, 12, 16, 20]
    cmap = plt.cm.plasma
    for i, h in enumerate(snap_hours):
        idx = int(np.argmin(np.abs(hours_s - h)))
        ax1.plot(T_s[idx, :], depths * 100,  # cm
                 color=cmap(i / (len(snap_hours) - 1)),
                 lw=2.0, label=f"{h:02d}:00")
    ax1.invert_yaxis()
    ax1.set_yscale("log")
    ax1.set_ylim(z_bottom * 100 * 1.1, max(depths[1] * 100 * 0.5, 0.05))
    ax1.set_xlim(80, 410)
    ax1.set_xlabel(r"Temperature $T$ (K)")
    ax1.set_ylabel(r"Depth $z$ (cm)")
    ax1.legend(loc="lower left", title="Local hour", ncol=2)
    panel_label(ax1, "(a)", x=0.95, y=0.95)

    # ---- (b) heatmap ----
    H_uni = np.linspace(0, 24, 200)
    T_resamp = np.empty((H_uni.size, T_s.shape[1]))
    for k in range(T_s.shape[1]):
        T_resamp[:, k] = np.interp(H_uni, hours_s, T_s[:, k])
    pcm = ax2.pcolormesh(
        H_uni, depths * 100, T_resamp.T,
        shading="auto", cmap="inferno",
        vmin=80, vmax=400,
    )
    ax2.invert_yaxis()
    ax2.set_yscale("log")
    ax2.set_ylim(z_bottom * 100 * 1.1, max(depths[1] * 100 * 0.5, 0.05))
    ax2.set_xlim(0, 24)
    ax2.set_xticks(range(0, 25, 6))
    ax2.set_xlabel("Local solar time (h)")
    ax2.set_ylabel(r"Depth $z$ (cm)")
    cbar = fig.colorbar(pcm, ax=ax2, pad=0.03, fraction=0.06)
    cbar.set_label(r"$T$ (K)")
    panel_label(ax2, "(b)", x=0.04, y=0.95, color="white")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"  wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", nargs="?", default="output/lunar_validation")
    p.add_argument("--location", default="Equator")
    p.add_argument("--depths-file", default="data/locations/depths.txt")
    p.add_argument("--out", default="docs/figures/v2/thermal_wave_profiles.png")
    args = p.parse_args()

    run = Path(args.run_dir)
    csv = run / "locations" / f"{args.location}_depths.csv"
    print(f"Reading {csv}")
    times_s, T = load_depth_csv(csv)
    depths = load_depths(Path(args.depths_file))
    if depths.size != T.shape[1]:
        raise SystemExit(
            f"depth count mismatch: {depths.size} vs {T.shape[1]}")
    print(f"  {T.shape[0]} samples × {T.shape[1]} depths")
    make_figure(times_s, T, depths, Path(args.out))


if __name__ == "__main__":
    main()
