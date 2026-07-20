#!/usr/bin/env python3
"""Latitude and longitude profiles of Europa surface H2O mass balance.

Reproduces the abstract's claims:
  - Sublimation dominates ablation at low/mid latitudes; net polar accumulation.
  - Sputtering ~few percent of net ablation.
  - Equatorial ice depletion vs mid-lat / polar enrichment.
  - Longitudinal sub-Jovian eclipse signature: lower noontime temperatures
    suppress sublimation+sputtering near 0°E.

Usage:
  python plot_mass_balance_profiles.py \
      --run output/europa_ico6_albedo-map_cryst \
      --out docs/figures/jwst/mass_balance_profiles.png
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
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--epoch", type=int, default=-1)
    args = p.parse_args()

    mb = mb_mod.load_mass_balance(args.run, epoch=args.epoch)
    yr = mb["total_years"]
    print(f"Loaded mass balance from {args.run} at epoch {mb['epoch']} "
          f"(t = {yr:.1f} yr)")

    fig, axes = plt.subplots(3, 1, figsize=(10.5, 11.0),
                             gridspec_kw={"hspace": 0.36})

    g_per_m2_per_Myr = lambda x: x * 1.0e6 * 1.0e3   # kg/m²/yr × 1e6 × 1e3

    lat_bins = np.linspace(-87.5, 87.5, 36)
    centers, sub_lat  = mb_mod.zonal_mean(mb["lat"], mb["sublimation_rate"], lat_bins)
    _,       sput_lat = mb_mod.zonal_mean(mb["lat"], mb["sputtering_rate"], lat_bins)
    _,       dep_lat  = mb_mod.zonal_mean(mb["lat"], mb["deposition_rate"], lat_bins)
    _,       net_lat  = mb_mod.zonal_mean(mb["lat"], mb["net_loss_rate"],  lat_bins)

    # ------------------------------------------------------------------
    # Panel A: Sublimation vs deposition (gross fluxes — both peak at eq)
    # ------------------------------------------------------------------
    ax = axes[0]
    ax.plot(centers, g_per_m2_per_Myr(sub_lat),  "-",  color="tab:red",
            lw=2.2, label="Sublimation (gross emission)")
    ax.plot(centers, g_per_m2_per_Myr(dep_lat),  "-",  color="tab:blue",
            lw=2.2, label="Deposition (ballistic re-arrival)")
    ax.set_xlabel("Latitude (deg)")
    ax.set_ylabel("Gross flux (g m⁻² Myr⁻¹)")
    ax.set_title("(A) Gross sublimation & deposition — diurnal water cycle")
    ax.set_xlim(-90, 90); ax.set_xticks(np.arange(-90, 91, 30))
    ax.grid(alpha=0.3); ax.legend(fontsize=9, loc="best", framealpha=0.95)

    # ------------------------------------------------------------------
    # Panel B: NET sublimation−deposition (the redistribution signal)
    #          + sputtering on the same axis, stacked from zero
    # ------------------------------------------------------------------
    ax = axes[1]
    sub_minus_dep = sub_lat - dep_lat
    ax.plot(centers, g_per_m2_per_Myr(sub_minus_dep), "-",
            color="tab:red", lw=2.4,
            label="Net sublimation − deposition (positive = net loss)")
    ax.plot(centers, g_per_m2_per_Myr(sput_lat), "-",
            color="tab:purple", lw=2.2,
            label="Sputtering (always net loss)")
    ax.plot(centers, g_per_m2_per_Myr(net_lat), "k-", lw=2.6,
            label="Total net mass loss = (sub−dep) + sputtering")
    ax.axhline(0, color="gray", lw=0.7, ls="--")
    ax.fill_between(centers, 0, g_per_m2_per_Myr(net_lat),
                    where=(g_per_m2_per_Myr(net_lat) > 0),
                    color="tab:red", alpha=0.10, label="net ablation zone")
    ax.fill_between(centers, 0, g_per_m2_per_Myr(net_lat),
                    where=(g_per_m2_per_Myr(net_lat) < 0),
                    color="tab:blue", alpha=0.10, label="net accumulation zone")
    ax.set_xlabel("Latitude (deg)")
    ax.set_ylabel("Net rate (g m⁻² Myr⁻¹)")
    ax.set_title("(B) Net redistribution: equatorial ablation vs polar "
                 "accumulation; sputtering ~ few % of total")
    ax.set_xlim(-90, 90); ax.set_xticks(np.arange(-90, 91, 30))
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="best", framealpha=0.95)

    # Annotation
    eq = (np.abs(centers) < 15)
    pol = (np.abs(centers) > 70)
    if eq.any() and pol.any():
        net_eq = float(np.nanmean(net_lat[eq]))
        net_pol = float(np.nanmean(net_lat[pol]))
        sub_dep_eq = float(np.nanmean(sub_minus_dep[eq]))
        sub_dep_pol = float(np.nanmean(sub_minus_dep[pol]))
        sp_eq = float(np.nanmean(sput_lat[eq]))
        sp_pol = float(np.nanmean(sput_lat[pol]))
        ax.text(0.02, 0.97,
                f"Equator (|lat|<15°):  sub−dep = {g_per_m2_per_Myr(sub_dep_eq):+.1f}, "
                f"sput = {g_per_m2_per_Myr(sp_eq):.1f}  g·m⁻²·Myr⁻¹\n"
                f"Pole   (|lat|>70°):  sub−dep = {g_per_m2_per_Myr(sub_dep_pol):+.1f}, "
                f"sput = {g_per_m2_per_Myr(sp_pol):.1f}  g·m⁻²·Myr⁻¹",
                transform=ax.transAxes, va="top", ha="left", fontsize=8.0,
                family="monospace",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    # ------------------------------------------------------------------
    # Panel C: Longitudinal profile (equatorial band ±30°)
    # ------------------------------------------------------------------
    ax = axes[2]
    lon_bins = np.linspace(0.0, 360.0, 73)  # 5° bins

    band = (-30, 30)
    lcenters, sub_lon  = mb_mod.longitudinal_mean(mb["lon"], mb["lat"],
                                                  mb["sublimation_rate"],
                                                  band, lon_bins)
    _,        sput_lon = mb_mod.longitudinal_mean(mb["lon"], mb["lat"],
                                                  mb["sputtering_rate"],
                                                  band, lon_bins)
    _,        dep_lon  = mb_mod.longitudinal_mean(mb["lon"], mb["lat"],
                                                  mb["deposition_rate"],
                                                  band, lon_bins)
    _,        net_lon  = mb_mod.longitudinal_mean(mb["lon"], mb["lat"],
                                                  mb["net_loss_rate"],
                                                  band, lon_bins)

    ax.plot(lcenters, g_per_m2_per_Myr(sub_lon),  "-",  color="tab:red",
            lw=2.2, label="Sublimation")
    ax.plot(lcenters, g_per_m2_per_Myr(dep_lon),  "-",  color="tab:blue",
            lw=2.2, label="Deposition")
    ax.plot(lcenters, g_per_m2_per_Myr(sput_lon), "-",  color="tab:purple",
            lw=2.2, label="Sputtering")
    ax.plot(lcenters, g_per_m2_per_Myr(net_lon),  "k-", lw=2.5,
            label="Net loss")
    ax.axhline(0, color="gray", lw=0.6, ls="--")

    # Reference longitudes
    ax.axvline(0,   color="gold",      lw=0.8, alpha=0.6); ax.text(2,   ax.get_ylim()[1]*0.92, "sub-Jov.",  fontsize=8, color="darkgoldenrod")
    ax.axvline(90,  color="firebrick", lw=0.8, alpha=0.6); ax.text(92,  ax.get_ylim()[1]*0.92, "trailing",  fontsize=8, color="firebrick")
    ax.axvline(180, color="dimgray",   lw=0.8, alpha=0.6); ax.text(182, ax.get_ylim()[1]*0.92, "anti-Jov.", fontsize=8, color="dimgray")
    ax.axvline(270, color="navy",      lw=0.8, alpha=0.6); ax.text(272, ax.get_ylim()[1]*0.92, "leading",   fontsize=8, color="navy")

    ax.set_xlabel("East longitude (deg, IAU)")
    ax.set_ylabel("Rate (g m⁻² Myr⁻¹)")
    ax.set_title(f"(C) Equatorial-band ({band[0]}° to +{band[1]}°) "
                 "longitudinal profile — sub-Jovian eclipse signature")
    ax.set_xlim(0, 360); ax.set_xticks(np.arange(0, 361, 60))
    ax.grid(alpha=0.3); ax.legend(fontsize=9, loc="lower right", framealpha=0.95)

    # Subtitle / context
    fig.suptitle(
        f"PTM3D Europa surface H₂O mass balance "
        f"(ico-6 albedo-map run, t = {yr:.0f} yr)",
        y=0.995, fontsize=12)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=170, bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Wrote {args.out} and {args.out.with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
