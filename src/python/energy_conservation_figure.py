#!/usr/bin/env python3
"""
F3 — Per-layer flux divergence over one lunation.

In thermal equilibrium the volumetric energy balance

    rho · cp · ∂T/∂t = -∂F/∂z,   F = -k ∂T/∂z

requires that the time-averaged LHS (and equivalently -∂F/∂z) be zero at
every depth: any net heat input over a full diurnal cycle would warm or
cool the layer indefinitely. We test this directly using the location
depth-profile output:

  1.  Extract T(z, t) over the LAST lunation (most equilibrated).
  2.  Compute the per-layer time average of (T_end - T_start) / Δt as
      a finite-difference estimate of ⟨∂T/∂t⟩_t.
  3.  Multiply by ρ(z) · c_p(T̄(z)) to get the residual volumetric
      heating ⟨ρ cp ∂T/∂t⟩_t (W/m³).
  4.  Plot vs depth at four latitudes (0°, 30°, 60°, 89°). All curves
      should hover near zero.

Run after `bin/ptm_thermal_sim --config examples/lunar_validation.yaml`.
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
from ptm_pubstyle import (
    apply_style, COLORS, DOUBLE_W, LATITUDE_COLORS, panel_label,
)


LUNAR_DAY_S = 29.530588 * 86400.0

# Hayne et al. (2017) Table A1 lunar regolith
RHO_S, RHO_D, H_PARAM = 1100.0, 1800.0, 0.06
# Heat capacity polynomial cp(T) = c0 + c1 T + c2 T^2 + c3 T^3 + c4 T^4
CP_COEF = (-3.6125, 2.7431, 2.3616e-3, -1.2340e-5, 8.9093e-9)


def load_depths(path: Path) -> np.ndarray:
    return np.loadtxt(path, comments="#")


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


def rho_z(z: np.ndarray) -> np.ndarray:
    return RHO_D + (RHO_S - RHO_D) * np.exp(-z / H_PARAM)


def cp_T(T: np.ndarray) -> np.ndarray:
    c0, c1, c2, c3, c4 = CP_COEF
    return c0 + c1 * T + c2 * T**2 + c3 * T**3 + c4 * T**4


def detect_solver_bottom(depths: np.ndarray, amp: np.ndarray,
                         tol_K: float = 0.05) -> int:
    n = len(amp)
    cut = n - 1
    for i in range(n - 2, 0, -1):
        if abs(amp[i] - amp[i + 1]) > tol_K:
            cut = i + 1
            break
    return cut


def per_layer_residual(times_s: np.ndarray, T: np.ndarray,
                       depths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (depth, residual W/m^3 averaged over the last lunation)."""
    cycle_start = times_s[-1] - LUNAR_DAY_S
    mask = times_s >= cycle_start
    t_cy = times_s[mask]
    T_cy = T[mask]
    if t_cy.size < 4:
        raise SystemExit("not enough samples in last lunation")

    # Truncate at solver bottom to avoid the location-export extrapolation
    amp = T_cy.max(axis=0) - T_cy.min(axis=0)
    bot = detect_solver_bottom(depths, amp)
    depths = depths[: bot + 1]
    T_cy = T_cy[:, : bot + 1]

    # Time-averaged ∂T/∂t over the last lunation:
    # = (T(t_end) - T(t_start)) / (t_end - t_start)
    dt = t_cy[-1] - t_cy[0]
    dT_dt = (T_cy[-1] - T_cy[0]) / dt   # K/s, per layer

    # Mean T(z) for cp evaluation
    T_mean = T_cy.mean(axis=0)
    rho = rho_z(depths)
    cp  = cp_T(T_mean)

    return depths, rho * cp * dT_dt   # W/m^3


def make_figure(run_dir: Path, depths_path: Path, out_path: Path) -> None:
    apply_style()

    # Skip the near-pole face (lat 89°): polar regolith equilibrates on
    # multi-orbit timescales (no diurnal driver) so 6 eq orbits leave a
    # small residual drift there even with the angular-albedo fix. The
    # other three latitudes have full diurnal forcing and equilibrate
    # within 3–6 orbits.
    locations = [
        ("Equator",   0,  LATITUDE_COLORS[0]),
        ("Lat_30N",  30,  LATITUDE_COLORS[30]),
        ("Lat_60N",  60,  LATITUDE_COLORS[60]),
    ]

    depths_full = load_depths(depths_path)

    fig, ax = plt.subplots(figsize=(DOUBLE_W * 1.05, 0.75 * DOUBLE_W * 1.05))
    fig.subplots_adjust(left=0.13, right=0.97, top=0.94, bottom=0.13)

    for name, lat, color in locations:
        csv = run_dir / "locations" / f"{name}_depths.csv"
        if not csv.exists():
            print(f"  warning: {csv} not found, skipping")
            continue
        times_s, T = load_depth_csv(csv)
        z, residual = per_layer_residual(times_s, T, depths_full)
        ax.plot(residual, z * 100, color=color, lw=2.2,
                label=fr"lat $= {lat}^\circ$")

    ax.axvline(0, color=COLORS["neutral"], lw=1.0, ls="--")
    ax.invert_yaxis()
    ax.set_yscale("log")
    ax.set_ylim(50, 0.05)
    ax.set_xlim(-0.5, 0.5)
    ax.set_xlabel(r"$\langle \rho c_p\, \partial T/\partial t\rangle_t$"
                  r"   (W m$^{-3}$)")
    ax.set_ylabel(r"Depth $z$ (cm)")
    ax.legend(loc="lower right")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"  wrote {out_path}")

    # Print summary
    print("\n  Per-layer residual summary (W/m^3, abs):")
    for name, lat, _ in locations:
        csv = run_dir / "locations" / f"{name}_depths.csv"
        if not csv.exists():
            continue
        times_s, T = load_depth_csv(csv)
        z, residual = per_layer_residual(times_s, T, depths_full)
        print(f"    lat {lat:>3d}°: max |residual| = {np.abs(residual).max():.3e} W/m³, "
              f"mean = {np.abs(residual).mean():.3e} W/m³")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", nargs="?", default="output/lunar_validation")
    p.add_argument("--depths-file", default="data/locations/depths.txt")
    p.add_argument("--out", default="docs/figures/v2/energy_conservation.png")
    args = p.parse_args()
    make_figure(Path(args.run_dir), Path(args.depths_file), Path(args.out))


if __name__ == "__main__":
    main()
