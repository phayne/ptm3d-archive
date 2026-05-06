#!/usr/bin/env python3
"""Mass-balance vs JWST observation overlay.

Tests the abstract's hypothesis directly:
  - Bright-cold regions: high deposition → freshly amorphized → LOW Fresnel peak
  - Dark-warm regions (Tara/Powys): high sublimation+sputtering → buried
    crystalline ice exposed → HIGH Fresnel peak

We regrid the per-face PTM3D net loss rate onto each observed Cartwright
spaxel and compute Spearman / Pearson correlations between:
  1. Net loss rate vs observed 3.1 μm Fresnel peak strength
  2. Bond albedo vs observed Fresnel peak strength
  3. Net loss rate vs observed 4.25 μm CO2 lobe depth
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import mass_balance as mb_mod  # noqa: E402


def latlon_to_xyz(lat, lon):
    rl = np.deg2rad(lat); ro = np.deg2rad(lon)
    return np.stack([np.cos(rl) * np.cos(ro),
                     np.cos(rl) * np.sin(ro),
                     np.sin(rl)], axis=-1)


def regrid(face_lat, face_lon, vals, obs_lat, obs_lon):
    tree = cKDTree(latlon_to_xyz(face_lat, face_lon))
    _, idx = tree.query(latlon_to_xyz(obs_lat, obs_lon))
    return vals[idx]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--obs-npz", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    mb = mb_mod.load_mass_balance(args.run)
    obs = np.load(args.obs_npz, allow_pickle=True)

    fres = dict(lat=obs["fresnel_31um__lat"],
                lon=obs["fresnel_31um__lon"],
                value=obs["fresnel_31um__value"])
    co2  = dict(lat=obs["co2_band_depth__lat"],
                lon=obs["co2_band_depth__lon"],
                value=obs["co2_band_depth__value"])

    nl = mb["net_loss_rate"] * 1.0e6 * 1.0e3   # g/m²/Myr
    sub_minus_dep = (mb["sublimation_rate"] - mb["deposition_rate"]) * 1e6 * 1e3
    alb = mb["bond_albedo"]

    pred_nl_fres = regrid(mb["lat"], mb["lon"], nl, fres["lat"], fres["lon"])
    pred_alb_fres = regrid(mb["lat"], mb["lon"], alb, fres["lat"], fres["lon"])
    pred_subdep_fres = regrid(mb["lat"], mb["lon"], sub_minus_dep,
                              fres["lat"], fres["lon"])
    pred_nl_co2 = regrid(mb["lat"], mb["lon"], nl, co2["lat"], co2["lon"])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    def panel(ax, x, y, xlabel, ylabel, title, x_ticks_log=False):
        valid = np.isfinite(x) & np.isfinite(y)
        x, y = x[valid], y[valid]
        if x.size < 3:
            ax.text(0.5, 0.5, "n<3 → skipped", transform=ax.transAxes,
                    ha="center")
            return
        rho_s, _ = spearmanr(x, y)
        r_p, _ = pearsonr(x, y)
        ax.scatter(x, y, s=6, alpha=0.55, color="tab:blue", edgecolors="none")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title}\nSpearman ρ = {rho_s:+.3f}, "
                     f"Pearson r = {r_p:+.3f}, N = {x.size}")
        ax.grid(alpha=0.25)

    panel(axes[0], pred_nl_fres, fres["value"],
          "PTM3D net mass loss (g m⁻² Myr⁻¹)",
          "Cartwright 3.1 μm Fresnel peak (%)",
          "(A) Crystalline-ice exposure ∝ ablation?")

    panel(axes[1], pred_alb_fres, fres["value"],
          "Bond albedo",
          "Cartwright 3.1 μm Fresnel peak (%)",
          "(B) Crystalline-ice exposure ∝ darkness?")

    panel(axes[2], pred_nl_co2, co2["value"],
          "PTM3D net mass loss (g m⁻² Myr⁻¹)",
          "Cartwright 4.25 μm CO₂ lobe depth (%)",
          "(C) CO₂ exposure ∝ ablation?")

    fig.suptitle("Mass-balance vs JWST observed band parameters "
                 "(per-spaxel correlation tests)", fontsize=12, y=1.02)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=170, bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
