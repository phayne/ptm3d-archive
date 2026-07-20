#!/usr/bin/env python3
"""
Headline map figure: equirectangular plot of integrated φ_C(lat, lon) and T_max.

Shows the spatial distribution of predicted crystalline ice with Europa
landmarks overlaid (Tara Regio, Powys Regio, Pwyll, leading/trailing apices).

Usage:
  python plot_phi_C_map.py --phi-npz data/jwst/cartwright2025/phi_eq_albedo_map.npz \
                            --out    docs/figures/jwst/phi_C_map.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import equirect  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phi-npz", type=Path,
                   default=Path("data/jwst/cartwright2025/phi_eq_albedo_map.npz"))
    p.add_argument("--out", type=Path,
                   default=Path("docs/figures/jwst/phi_C_map.png"))
    p.add_argument("--use-tmax", action="store_true",
                   help="Plot phi_eq(T_max) instead of integrated phi_C")
    p.add_argument("--grid-dpd", type=int, default=2)
    args = p.parse_args()

    npz = np.load(args.phi_npz)
    lat, lon = npz["lat"], npz["lon"]
    T_max = npz["T_max"]

    if args.use_tmax or "phi_C_integrated" not in npz.files:
        phi = npz["phi_eq_tmax"]
        phi_label = r"$\varphi_{\mathrm{eq}}(T_{\max})$"
        phi_kind = "phi_eq_tmax"
    else:
        phi = npz["phi_C_integrated"]
        if "cpp_epoch" in npz.files:
            phi_label = (r"$\varphi_C$ (PTM3D C++ multi-timescale, epoch "
                         f"{int(npz['cpp_epoch']):d})")
        elif "n_orbits_integrated" in npz.files:
            phi_label = (r"$\varphi_C$ (Python integrator, "
                         f"{int(npz['n_orbits_integrated']):d} orbits)")
        else:
            phi_label = r"$\varphi_C$ (integrated)"
        phi_kind = "phi_C_integrated"

    print(f"[plot_phi_C_map] Plotting {phi_kind}: range [{phi.min():.3f}, {phi.max():.3f}]")

    lon_g, lat_g, T_grid = equirect.project_to_grid(lat, lon, T_max, args.grid_dpd)
    _, _, P_grid = equirect.project_to_grid(lat, lon, phi, args.grid_dpd)
    extent = [0, 360, -90, 90]

    fig, axes = plt.subplots(2, 1, figsize=(11.0, 8.4),
                             gridspec_kw={"hspace": 0.28})

    ax = axes[0]
    im = ax.imshow(T_grid, origin="lower", extent=extent, cmap="inferno",
                   vmin=80, vmax=160, aspect="auto", interpolation="nearest")
    cbar = plt.colorbar(im, ax=ax, shrink=0.82, pad=0.015)
    cbar.set_label(r"$T_{\max}$  (K)", fontsize=11)
    ax.set_xlabel("East longitude (deg, IAU)", fontsize=11)
    ax.set_ylabel("Latitude (deg)", fontsize=11)
    ax.set_title("PTM3D peak diurnal temperature (ico-6, Galileo albedo map)",
                 fontsize=11)
    equirect.annotate_landmarks(ax, kinds=("chaos", "ref"), color="white")

    ax = axes[1]
    im = ax.imshow(P_grid, origin="lower", extent=extent, cmap="cividis",
                   vmin=0.0, vmax=1.0, aspect="auto", interpolation="nearest")
    cbar = plt.colorbar(im, ax=ax, shrink=0.82, pad=0.015)
    cbar.set_label(phi_label, fontsize=11)
    ax.set_xlabel("East longitude (deg, IAU)", fontsize=11)
    ax.set_ylabel("Latitude (deg)", fontsize=11)
    ax.set_title("PTM3D predicted crystalline-ice fraction "
                 "(Mitchell + Nordheim physics)", fontsize=11)
    equirect.annotate_landmarks(ax, kinds=("chaos", "ref"), color="white")

    # Suptitle
    fig.suptitle("Europa crystallization map — PTM3D prediction\n"
                 "(trailing apex at 90°E, leading apex at 270°E in IAU east-positive)",
                 fontsize=12, y=0.995)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    pdf = args.out.with_suffix(".pdf")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"[plot_phi_C_map] Wrote {args.out} and {pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
