#!/usr/bin/env python3
"""Amorphization rate map: shows where each amorphization pathway dominates.

Three pathways for amorphous-ice production (per Eq. D29 + Eq. D39 + frost):
  1. Radiation: k_amorph = D_dot / E_amorph                   [s^-1]
  2. Depositional dilution: dep_flux / (rho_ice * skin_depth)  [s^-1]
  3. (Implicit) thermal recrystallization rate (the OPPOSING process):
     k_cryst from Mitchell 2017 Avrami at T_max

We plot the LOG ratio  log10(k_dep_dilution / k_rad)  per face, so that
positive values flag faces where deposition (cold/bright) dominates the
amorphization budget — as the abstract hypothesises — and negative values
flag radiation-dominated (warmer / dose-controlled) faces.

We also plot the recrystallization-vs-amorphization competition map.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import mass_balance as mb_mod  # noqa: E402
import equirect  # noqa: E402
import crystallization as cr  # noqa: E402

SECONDS_PER_YEAR = 365.25 * 86400.0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    mb = mb_mod.load_mass_balance(args.run)
    yr = mb["total_years"]

    rho_ice = 917.0
    skin_depth = 0.05
    reservoir = rho_ice * skin_depth   # kg/m²

    k_rad = mb["dose_rate"] / cr.E_AMORPH                          # s^-1
    dep_per_s = mb["deposition_rate"] / SECONDS_PER_YEAR           # kg/m²/s
    k_dep = dep_per_s / reservoir                                  # s^-1
    k_cryst_eq = cr.k_cryst(mb["T_max"], cr.porosity_from_latitude(mb["lat"]))

    # Where deposition ≥ radiation = bright-cold-amorphous regions
    log_dep_over_rad = np.log10(np.maximum(k_dep, 1e-30)
                                / np.maximum(k_rad, 1e-30))
    # Where crystallization exceeds amorphization = thermally-driven crystalline
    log_cryst_over_total = np.log10(np.maximum(k_cryst_eq, 1e-30)
                                    / np.maximum(k_rad + k_dep, 1e-30))

    fig, axes = plt.subplots(2, 1, figsize=(11.5, 8.5),
                             gridspec_kw={"hspace": 0.32})

    # Panel A: where deposition vs radiation dominates amorphization
    lon_g, lat_g, M = equirect.project_to_grid(mb["lat"], mb["lon"],
                                               log_dep_over_rad, 2)
    ax = axes[0]
    im = ax.imshow(M, origin="lower", extent=[0, 360, -90, 90],
                   cmap="RdBu_r", vmin=-3, vmax=3,
                   aspect="auto", interpolation="nearest")
    plt.colorbar(im, ax=ax, shrink=0.85, pad=0.015,
                 label=r"log$_{10}$ (k$_{\rm dep}$ / k$_{\rm rad}$)")
    ax.set_title("(A) Amorphization budget: deposition (cold/bright) vs "
                 "radiation (warm/dark)")
    ax.set_xlabel("East longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    equirect.annotate_landmarks(ax, kinds=("chaos", "ref"),
                                color="white", fontsize=7)

    # Panel B: thermal recrystallization vs combined amorphization
    _, _, M2 = equirect.project_to_grid(mb["lat"], mb["lon"],
                                        log_cryst_over_total, 2)
    ax = axes[1]
    im = ax.imshow(M2, origin="lower", extent=[0, 360, -90, 90],
                   cmap="seismic", vmin=-6, vmax=6,
                   aspect="auto", interpolation="nearest")
    cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.015,
                        label=r"log$_{10}$ (k$_{\rm cryst}$(T$_{\max}$) / "
                              r"(k$_{\rm rad}$+k$_{\rm dep}$))")
    cbar.ax.tick_params(labelsize=8)
    ax.set_title("(B) Thermal recrystallization vs total amorphization "
                 "(positive = crystalline ice survives)")
    ax.set_xlabel("East longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    equirect.annotate_landmarks(ax, kinds=("chaos", "ref"),
                                color="black", fontsize=7)

    fig.suptitle(f"Amorphization vs recrystallization budget (t = {yr:.0f} yr)",
                 fontsize=12, y=0.995)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=170, bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
