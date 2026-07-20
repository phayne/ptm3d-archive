#!/usr/bin/env python3
"""Albedo vs ablation scatter — direct test of the bright-cold→amorphous /
dark-warm→ablated hypothesis. Each face is one point; Tara, Powys, and
the northern low-latitude zone are highlighted.
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


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    mb = mb_mod.load_mass_balance(args.run)
    yr = mb["total_years"]

    alb = mb["bond_albedo"]
    net = mb["net_loss_rate"] * 1.0e6 * 1.0e3   # g/m²/Myr
    sub_minus_dep = (mb["sublimation_rate"] - mb["deposition_rate"]) * 1.0e6 * 1.0e3
    sput = mb["sputtering_rate"] * 1.0e6 * 1.0e3

    valid = np.isfinite(alb) & np.isfinite(net)
    alb = alb[valid]; net = net[valid]
    sub_minus_dep = sub_minus_dep[valid]; sput = sput[valid]
    lat = mb["lat"][valid]; lon = mb["lon"][valid]

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.4))

    # Panel A: total net loss vs albedo
    ax = axes[0]
    sc = ax.scatter(alb, net, c=lat, cmap="coolwarm", s=4, alpha=0.5,
                    vmin=-90, vmax=90)
    plt.colorbar(sc, ax=ax, shrink=0.8, label="latitude (°)", pad=0.02)
    ax.axhline(0, color="gray", lw=0.7, ls="--")
    ax.set_xlabel("Bond albedo")
    ax.set_ylabel("Net mass loss rate (g m⁻² Myr⁻¹)")
    ax.set_title("(A) Net mass loss vs Bond albedo (every face)")
    ax.grid(alpha=0.3)

    # Mark Tara, Powys, N.lat
    landmarks = [("Tara", -10, 285, "red"),
                 ("Powys",  0, 215, "darkorange"),
                 ("N.lat", 20, 255, "green")]
    for name, plat, plon, col in landmarks:
        d = np.hypot(lat - plat, ((lon - plon + 180) % 360) - 180)
        i = int(np.argmin(d))
        ax.scatter([alb[i]], [net[i]], s=170, facecolors="none",
                   edgecolors=col, linewidths=2.5, marker="s", zorder=10)
        ax.annotate(name, (alb[i], net[i]), xytext=(8, 6),
                    textcoords="offset points", color=col, fontweight="bold",
                    fontsize=9, zorder=11)

    # Panel B: sublimation−deposition vs albedo, with sputtering on a twin axis
    ax = axes[1]
    sc = ax.scatter(alb, sub_minus_dep, c=sput, cmap="viridis", s=5, alpha=0.55,
                    vmin=np.nanpercentile(sput, 5),
                    vmax=np.nanpercentile(sput, 95))
    plt.colorbar(sc, ax=ax, shrink=0.8,
                 label="Sputtering rate (g m⁻² Myr⁻¹)", pad=0.02)
    ax.axhline(0, color="gray", lw=0.7, ls="--")
    ax.set_xlabel("Bond albedo")
    ax.set_ylabel("Net sublimation − deposition (g m⁻² Myr⁻¹)")
    ax.set_title("(B) Sub−dep vs albedo (Spencer 1987 thermal segregation)")
    ax.grid(alpha=0.3)
    for name, plat, plon, col in landmarks:
        d = np.hypot(lat - plat, ((lon - plon + 180) % 360) - 180)
        i = int(np.argmin(d))
        ax.scatter([alb[i]], [sub_minus_dep[i]], s=170, facecolors="none",
                   edgecolors=col, linewidths=2.5, marker="s", zorder=10)
        ax.annotate(name, (alb[i], sub_minus_dep[i]), xytext=(8, 6),
                    textcoords="offset points", color=col, fontweight="bold",
                    fontsize=9, zorder=11)

    fig.suptitle(f"PTM3D Europa: surface H₂O mass balance vs Bond albedo "
                 f"(t = {yr:.0f} yr)", fontsize=12, y=1.005)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=170, bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
