#!/usr/bin/env python3
"""
Sigmoid theory figure (talk slide 2): equilibrium crystalline fraction φ_eq(T)
on Europa for trailing-equator and leading-apex dose rates, with the PTM3D
T_max histogram overlaid.

Anchors the headline argument: the crystallization-amorphization knife-edge
sits at T*≈125 K (trailing) / ~120 K (leading), squarely inside Europa's
diurnal T_max range. Hemispheric T* offset (~5 K) is testable against
JWST band-strength contrast.

Usage:
  python plot_sigmoid_theory.py --phi-npz data/jwst/cartwright2025/phi_eq_albedo_map.npz \
                                 --out docs/figures/jwst/sigmoid_theory.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import crystallization as cr  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phi-npz", type=Path,
                   default=Path("data/jwst/cartwright2025/phi_eq_albedo_map.npz"),
                   help="NPZ from predict_phi_eq.py (used for T_max histogram)")
    p.add_argument("--out", type=Path,
                   default=Path("docs/figures/jwst/sigmoid_theory.png"))
    p.add_argument("--porosity", type=float, default=0.0,
                   help="Porosity for the theoretical sigmoid (default 0 = compact)")
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Theoretical sigmoid for two reference dose rates
    T = np.linspace(80, 160, 1000)

    # Trailing equator (Ḋ = D0)
    kc_compact = cr.k_cryst(T, args.porosity)
    ka_trail = cr.k_amorph(cr.D0)  # scalar
    phi_trail = kc_compact / (kc_compact + ka_trail + 1e-300)

    # Leading apex: f_lon=1/C_D, f_lat=1
    D_lead = cr.D0 / cr.C_D
    ka_lead = cr.k_amorph(D_lead)
    phi_lead = kc_compact / (kc_compact + ka_lead + 1e-300)

    # Polar enhancement (high-lat trailing pole, ~maximum dose)
    f_lat_pole = 1.0 + (cr.E_D - 1.0) * 1.0  # |lat|=90
    D_polar = cr.D0 * 1.0 * f_lat_pole  # f_lon=1 at trailing
    ka_polar = cr.k_amorph(D_polar)
    phi_polar = kc_compact / (kc_compact + ka_polar + 1e-300)

    # Transition temperatures
    T_star_trail = cr.transition_temperature(cr.D0, args.porosity)
    T_star_lead = cr.transition_temperature(D_lead, args.porosity)
    T_star_pol = cr.transition_temperature(D_polar, args.porosity)

    # PTM3D T_max histogram (if available)
    T_max_hist = None
    if args.phi_npz.exists():
        npz = np.load(args.phi_npz)
        T_max_hist = npz["T_max"]
        # Filter out persistent-shadow PSR faces (T_max < 80 K is below crystallization floor)
        T_max_hist = T_max_hist[(T_max_hist > 80) & np.isfinite(T_max_hist)]

    # ---- Plot ----
    fig, ax_main = plt.subplots(figsize=(8.0, 5.4))

    # Sigmoids
    ax_main.plot(T, phi_lead, color="#1f77b4", lw=2.4,
                 label=f"Leading apex   (Ḋ = Ḋ₀/{cr.C_D:.0f})  T*={T_star_lead:.1f} K")
    ax_main.plot(T, phi_trail, color="#d62728", lw=2.4,
                 label=f"Trailing equator (Ḋ = Ḋ₀)        T*={T_star_trail:.1f} K")
    ax_main.plot(T, phi_polar, color="#9467bd", lw=2.0, ls="--",
                 label=f"Trailing pole   (Ḋ = {f_lat_pole:.1f}·Ḋ₀)   T*={T_star_pol:.1f} K")

    # Threshold markers
    for Ts, c in [(T_star_lead, "#1f77b4"), (T_star_trail, "#d62728"),
                  (T_star_pol, "#9467bd")]:
        ax_main.axvline(Ts, color=c, lw=0.8, alpha=0.4, ls=":")

    ax_main.axhline(0.5, color="gray", lw=0.6, ls=":")

    # T_max histogram on twin axis
    if T_max_hist is not None and T_max_hist.size > 0:
        ax_hist = ax_main.twinx()
        ax_hist.hist(T_max_hist, bins=80, range=(80, 160), color="gray",
                     alpha=0.18, edgecolor="none", zorder=-1,
                     label="PTM3D T_max (ico-6)")
        ax_hist.set_ylabel("PTM3D T_max histogram (faces)", color="dimgray", fontsize=10)
        ax_hist.tick_params(axis="y", colors="dimgray", labelsize=9)
        ax_hist.spines["right"].set_color("dimgray")
        # keep histogram in the background
        for line in ax_main.get_lines():
            line.set_zorder(2)

    ax_main.set_xlabel("Surface temperature T  (K)", fontsize=11)
    ax_main.set_ylabel(r"Equilibrium crystalline fraction  $\varphi_{\mathrm{eq}}$",
                       fontsize=11)
    ax_main.set_xlim(80, 160)
    ax_main.set_ylim(-0.02, 1.02)
    ax_main.grid(alpha=0.25)
    ax_main.legend(loc="lower right", fontsize=9, framealpha=0.95)

    title = ("Crystallization-amorphization knife-edge on Europa\n"
             f"(Mitchell 2017 ⨯ Nordheim 2022, calibrated to Cartwright 2025; "
             f"compact ice n={cr.N_LOW:.0f})")
    ax_main.set_title(title, fontsize=11)

    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    pdf = args.out.with_suffix(".pdf")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"Wrote {args.out} and {pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
