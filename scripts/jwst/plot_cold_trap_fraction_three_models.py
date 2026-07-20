#!/usr/bin/env python3
"""Cold-trap fraction vs latitude for H₂O and CO₂ — comparison of three
model conditions on the same plot:

  · dotted line  — flat surface, uniform Bond albedo A = 0.5
                   (no topographic shadowing, no albedo distribution).
  · dashed line  — topography only: the per-face T_max distribution from
                   the regional PTM3D runs, still with uniform A = 0.5.
  · solid line   — topography + Bond-albedo distribution: each face's
                   T_max rescaled by N draws from the per-latitude
                   Bond-albedo distribution via T_new = T (1-A_new)^(1/4) /
                   (1-A_ref)^(1/4).

The "flat surface" T_max at each latitude is empirically estimated as the
mean of the top 10 % of the regional T_max distribution — the well-
illuminated, near-flat facets within the topographic mesh approximate the
flat-surface case.

Output: docs/figures/jwst/cold_trap_fraction_three_models.{png,pdf}
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# IDL family from pretty_plots — match the rest of the talk deck
from pretty_plots import idl  # noqa: E402
idl.use()

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import cold_trap_volatile_stability as cts   # noqa: E402
import plot_per_latitude_histograms as pph    # noqa: E402

A_REF = pph.A_REF

# Timescale definitions: (label, loss_rate in mm/Gyr).
# The convention used by `cts.stability_temperature` is that the loss rate
# is in mm-of-ice-lost per Gyr, so "stable for 1 Gyr" → loss_rate = 1.0,
# "stable for 1 yr" → loss_rate = 1e9, etc.
TIMESCALES = [
    ("1 day",   365.25 * 1.0e9),
    ("1 yr",    1.0e9),
    ("1 kyr",   1.0e6),
    ("1 Myr",   1.0e3),
    ("10 Myr",  1.0e2),
    ("100 Myr", 1.0e1),
    ("1 Gyr",   1.0),
]


def slug_for_timescale(label):
    return label.replace(" ", "")

# Pole-to-pole; negative lats use the mirror-symmetric N runs (Europa
# axial tilt 0.1° → diurnal stats are mirror-symmetric).
LATITUDES_NORTH = [0, 15, 30, 45, 60, 75, 80, 85, 87]
LATITUDES = (sorted([-l for l in LATITUDES_NORTH if l > 0])
             + LATITUDES_NORTH)
TERRAINS = cts.TERRAIN_KEYS

# Two species. Colors: cool steel-blue for water (ice association),
# warm terracotta for CO2 (dry-ice / desert association). Mid-saturation
# so they read well on white backgrounds and remain distinguishable
# in greyscale (luminances differ by ~15%).
SPECIES_TO_PLOT = ["H2O", "CO2"]
SPECIES_COLOR = {"H2O": "#2C7DA0", "CO2": "#BC5A3C"}

LINESTYLE = {
    "no_topo":     ":",
    "topo":        "--",
    "topo_albedo": "-",
}
LINESTYLE_LABEL = {
    "no_topo":     "no topography, no albedo distribution",
    "topo":        "with topography",
    "topo_albedo": "with topography + albedo distribution",
}


def flat_surface_T_max(T_arr, top_pct=10.0):
    """Empirical proxy for a flat-surface T_max at the same latitude:
    mean of the top `top_pct` % of the regional T_max distribution
    (well-illuminated, near-flat facets)."""
    if T_arr.size == 0:
        return float("nan")
    threshold = np.percentile(T_arr, 100.0 - top_pct)
    return float(np.mean(T_arr[T_arr >= threshold]))


def fractions_three_models(species, A_2d, lat_centers,
                           loss_rate_mm_per_Gyr=1.0,
                           n_samples_per_face=10, lat_half_width=5.0):
    """For each species, return three latitude-indexed dicts:
       {lat: (mean, std)} where mean/std are across the 5 terrains.
       The three dicts correspond to the no_topo, topo, and topo_albedo
       conditions. `loss_rate_mm_per_Gyr` controls the timescale: 1.0 →
       1 Gyr stability, 1e9 → 1 yr, etc."""
    T_sub = {sp: cts.stability_temperature(
                  sp, loss_rate_m_per_Gyr=loss_rate_mm_per_Gyr)
             for sp in species}

    # raw[(model, sp, lat)] = list across terrains
    raw = {(m, sp, lat): [] for m in LINESTYLE
           for sp in species for lat in LATITUDES}

    for lat in LATITUDES:
        A_band = pph.albedos_in_band(A_2d, lat_centers, lat,
                                      half_width_deg=lat_half_width)
        for terrain in TERRAINS:
            data = cts.load_run_temperatures(abs(lat), terrain)
            if data is None:
                continue
            T_max = data[0]

            T_flat = flat_surface_T_max(T_max)
            T_topo = T_max
            if A_band.size and n_samples_per_face > 0:
                T_topo_albedo = pph.broaden_with_albedo(
                    T_max, A_band, A_ref=A_REF,
                    n_samples_per_face=n_samples_per_face,
                    rng_seed=hash((lat, terrain)) & 0xffff)
            else:
                T_topo_albedo = T_max

            for sp in species:
                Ts = T_sub[sp]
                raw[("no_topo",     sp, lat)].append(
                    100.0 if (np.isfinite(T_flat) and T_flat < Ts) else 0.0)
                raw[("topo",        sp, lat)].append(
                    100.0 * float(np.mean(T_topo < Ts)))
                raw[("topo_albedo", sp, lat)].append(
                    100.0 * float(np.mean(T_topo_albedo < Ts)))

    summary = {(m, sp): {} for m in LINESTYLE for sp in species}
    for (m, sp, lat), vals in raw.items():
        if vals:
            summary[(m, sp)][lat] = (float(np.mean(vals)),
                                     float(np.std(vals)))
        else:
            summary[(m, sp)][lat] = (float("nan"), float("nan"))
    return summary, T_sub


def plot_comparison(summary, T_sub, out_path, timescale_label="1 Gyr"):
    """Single-axes figure: three line styles × two species = 6 curves."""
    with plt.rc_context({
        "font.size": 14, "axes.labelsize": 17, "axes.titlesize": 18,
        "legend.fontsize": 13, "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    }):
        fig, ax = plt.subplots(figsize=(9.6, 6.4))

        for sp in SPECIES_TO_PLOT:
            color = SPECIES_COLOR[sp]
            for model in ("no_topo", "topo", "topo_albedo"):
                means = np.array([summary[(model, sp)][lat][0]
                                  for lat in LATITUDES])
                ax.plot(LATITUDES, means,
                        linestyle=LINESTYLE[model], linewidth=2.4,
                        color=color, marker="o", markersize=4.5,
                        markeredgecolor="white", markeredgewidth=0.6,
                        zorder=4)

        ax.set_xlabel("Latitude (°)")
        ax.set_ylabel("Cold trap fraction (%)")
        ax.set_xlim(-92, 92)
        ax.set_ylim(-2, 105)
        ax.set_xticks([-87, -75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75, 87])
        ax.tick_params(direction="in", top=True, right=True)
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.minorticks_on()
        ax.tick_params(which="minor", direction="in", top=True, right=True,
                       length=3)

        # Both family labels on the LEFT side of the plot, each placed
        # near the latitude band where its species' curves rise most
        # prominently — H2O at lat=-35 (the equatorial side of the H2O
        # rising edge) and CO2 at lat=-72 (the polar side, near where the
        # orange CO2 curves climb to saturation). White bbox keeps them
        # readable wherever curves cross.
        bbox_kw = dict(boxstyle="round,pad=0.25", facecolor="white",
                       edgecolor="none", alpha=0.88)
        ax.annotate(f"{cts.SPECIES_LABELS['H2O']}",
                    xy=(-35, 52), color=SPECIES_COLOR["H2O"],
                    fontsize=22, fontweight="bold", ha="center", va="center",
                    bbox=bbox_kw, zorder=10)
        ax.annotate(f"{cts.SPECIES_LABELS['CO2']}",
                    xy=(-72, 52), color=SPECIES_COLOR["CO2"],
                    fontsize=22, fontweight="bold", ha="center", va="center",
                    bbox=bbox_kw, zorder=10)
        ax.annotate(f"$T_\\mathrm{{sub}} = {T_sub['H2O']:.0f}$ K",
                    xy=(-35, 41), color=SPECIES_COLOR["H2O"],
                    fontsize=12, ha="center", va="center",
                    bbox=bbox_kw, zorder=10)
        ax.annotate(f"$T_\\mathrm{{sub}} = {T_sub['CO2']:.0f}$ K",
                    xy=(-72, 41), color=SPECIES_COLOR["CO2"],
                    fontsize=12, ha="center", va="center",
                    bbox=bbox_kw, zorder=10)

        # Linestyle key — placed in the equatorial mid-axis region where
        # the species labels are far away (at |lat|=50).
        from matplotlib.lines import Line2D
        style_handles = [
            Line2D([], [], color="0.2", lw=2.0, ls=LINESTYLE["topo_albedo"],
                   label="topography + albedo"),
            Line2D([], [], color="0.2", lw=2.0, ls=LINESTYLE["topo"],
                   label="topography only"),
            Line2D([], [], color="0.2", lw=2.0, ls=LINESTYLE["no_topo"],
                   label="no topography or albedo"),
        ]
        ax.legend(handles=style_handles, loc="lower center",
                  bbox_to_anchor=(0.5, 0.62), frameon=True,
                  framealpha=0.95, edgecolor="0.8", fancybox=False,
                  ncol=1, fontsize=12)

        ax.set_title(f"Cold-trap fractions ({timescale_label}): "
                     "topography and albedo contributions",
                     pad=10)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {out_path}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--albedo-fits", type=Path,
                   default=Path("data/albedo_maps/Europa-1deg-BondAlbedo.fits"))
    p.add_argument("--out-dir", type=Path,
                   default=Path("docs/figures/jwst"))
    p.add_argument("--n-samples", type=int, default=10)
    p.add_argument("--lat-half-width", type=float, default=5.0)
    args = p.parse_args()

    print(f"Loading Bond-albedo map from {args.albedo_fits}")
    lat_centers, _, A_2d = pph.load_albedo_map(args.albedo_fits)

    for ts_label, loss_rate in TIMESCALES:
        slug = slug_for_timescale(ts_label)
        print(f"\n=== Timescale: {ts_label}  (loss_rate = {loss_rate:.3g} mm/Gyr) ===")
        summary, T_sub = fractions_three_models(
            SPECIES_TO_PLOT, A_2d, lat_centers,
            loss_rate_mm_per_Gyr=loss_rate,
            n_samples_per_face=args.n_samples,
            lat_half_width=args.lat_half_width)
        print(f"  T_sub: H2O = {T_sub['H2O']:.1f} K, CO2 = {T_sub['CO2']:.1f} K")
        if ts_label == "1 Gyr":
            # Default un-suffixed filename for the canonical (1 Gyr) version
            out = args.out_dir / "cold_trap_fraction_three_models.png"
        else:
            out = args.out_dir / f"cold_trap_fraction_three_models_{slug}.png"
        plot_comparison(summary, T_sub, out_path=out, timescale_label=ts_label)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
