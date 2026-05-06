#!/usr/bin/env python3
"""Cold-trap fraction vs latitude — combining topographic shadowing
(per-face PTM3D output) and the Bond-albedo distribution at each latitude.

This is the broadened analogue of `plots/polar_cold_traps/composition_combined.pdf`,
which used the per-face T_max distribution under a uniform Bond albedo
A = 0.5. Here, each face's T_max is rescaled by N independent draws from
the Bond-albedo distribution within ±5° of the same latitude:

    T_new = T_face × ((1 − A_new) / (1 − A_ref))^(1/4)

The cold-trap fraction at each (latitude, terrain) is then the fraction of
(face × albedo-sample) pairs with T_new < T_stab(species, 1 Gyr).

Outputs (PNG + PDF):
  docs/figures/jwst/cold_trap_fraction_albedo_broadened          (4 species: H₂O CO₂ H₂S CH₄)
  docs/figures/jwst/cold_trap_fraction_albedo_broadened_extended (7 species: + CH₃OH NH₃ SO₂)

Usage:
  python scripts/jwst/plot_cold_trap_fraction_with_albedo.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Apply the IDL family from pretty_plots (Helvetica-family sans-serif,
# inward ticks all four sides, rounded line caps, TrueType-embedded PDFs).
from pretty_plots import idl  # noqa: E402
idl.use()

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import cold_trap_volatile_stability as cts  # noqa: E402
import plot_per_latitude_histograms as pph  # noqa: E402

A_REF = pph.A_REF                 # 0.5
TIMESCALE_INDEX_GYR = 2           # cts.TIMESCALE_LOSS_RATE_MMGYR[2] = 1.0  (1 Gyr)

# Pole-to-pole latitude axis. Negative lats reuse the mirror-symmetric
# northern thermal runs (Europa axial tilt ~0.1° → diurnal stats are
# mirror-symmetric to far better than the talk's resolution). The Bond-
# albedo distribution is sampled at the *true* hemisphere.
LATITUDES_NORTH = [0, 15, 30, 45, 60, 75, 80, 85, 87]
LATITUDES = (sorted([-l for l in LATITUDES_NORTH if l > 0])
             + LATITUDES_NORTH)
TERRAINS = cts.TERRAIN_KEYS

# Wong-palette colors and markers, matching the source figure
SPECIES_COLOR = {
    "H2O":   "#0072B2",   # blue
    "CO2":   "#E69F00",   # orange
    "H2S":   "#D55E00",   # vermillion
    "CH4":   "#009E73",   # bluish green
    "CH3OH": "#882255",   # purple
    "NH3":   "#117733",   # green
    "SO2":   "#56B4E9",   # sky blue
}
SPECIES_MARKER = {
    "H2O": "o", "CO2": "s", "H2S": "D", "CH4": "^",
    "CH3OH": "v", "NH3": "P", "SO2": "X",
}

SPECIES_FOUR    = ["H2O", "CO2", "H2S", "CH4"]
SPECIES_SEVEN   = ["H2O", "CH3OH", "CO2", "NH3", "SO2", "H2S", "CH4"]


def _frac_below(T_arr, threshold):
    if not np.isfinite(threshold) or T_arr.size == 0:
        return 0.0
    return float(np.mean(T_arr < threshold))


def compute_fractions(species_names, A_2d, lat_centers,
                      n_samples_per_face=10, lat_half_width=5.0):
    """Build a {species: {lat: (mean, std)}} dict where mean/std are
    computed across the 5 terrains' broadened-T cold-trap fractions.

    Returns also the underlying per-(species,lat,terrain) array so the
    caller can write CSV summaries if desired.
    """
    # Stability T per species at 1 Gyr
    T_stab = {}
    for sp in species_names:
        T_stab[sp] = cts.stability_temperature(
            sp, loss_rate_m_per_Gyr=cts.TIMESCALE_LOSS_RATE_MMGYR[TIMESCALE_INDEX_GYR])

    fracs_table = {sp: {lat: [] for lat in LATITUDES} for sp in species_names}

    for lat in LATITUDES:
        # Bond-albedo distribution at the *true* latitude (north or south)
        A_band = pph.albedos_in_band(A_2d, lat_centers, lat,
                                      half_width_deg=lat_half_width)
        for terrain in TERRAINS:
            data = cts.load_run_temperatures(abs(lat), terrain)
            if data is None:
                continue
            T_max = data[0]
            if A_band.size == 0 or n_samples_per_face <= 0:
                T_eff = T_max
            else:
                T_eff = pph.broaden_with_albedo(
                    T_max, A_band, A_ref=A_REF,
                    n_samples_per_face=n_samples_per_face,
                    rng_seed=hash((lat, terrain)) & 0xffff)
            for sp in species_names:
                fracs_table[sp][lat].append(100.0 * _frac_below(T_eff, T_stab[sp]))

    # Aggregate to mean/std across terrains
    summary = {sp: {} for sp in species_names}
    for sp in species_names:
        for lat in LATITUDES:
            vals = fracs_table[sp][lat]
            if vals:
                summary[sp][lat] = (float(np.mean(vals)), float(np.std(vals)))
            else:
                summary[sp][lat] = (float("nan"), float("nan"))
    return summary, T_stab, fracs_table


def plot_one(species_names, summary, T_stab, out_path,
             title=None):
    """Render the cold-trap-fraction-vs-latitude figure for the given
    species set, with shaded ± std band across the 5 terrains."""
    with plt.rc_context({
        "font.size": 14, "axes.labelsize": 16, "axes.titlesize": 17,
        "legend.fontsize": 13, "xtick.labelsize": 13, "ytick.labelsize": 13,
    }):
        fig, ax = plt.subplots(figsize=(9.4, 5.6))

        for sp in species_names:
            color = SPECIES_COLOR[sp]
            marker = SPECIES_MARKER[sp]
            means = np.array([summary[sp][lat][0] for lat in LATITUDES])
            stds  = np.array([summary[sp][lat][1] for lat in LATITUDES])
            label = f"{cts.SPECIES_LABELS[sp]} ({T_stab[sp]:.0f} K)"
            ax.plot(LATITUDES, means, marker=marker, linestyle="-",
                    color=color, linewidth=2.0, markersize=7,
                    markeredgecolor="white", markeredgewidth=0.7,
                    zorder=5, label=label)
            ax.fill_between(LATITUDES,
                            np.clip(means - stds, 0.0, 100.0),
                            np.clip(means + stds, 0.0, 100.0),
                            color=color, alpha=0.15, linewidth=0,
                            zorder=2)

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

        ax.legend(loc="upper center", frameon=True, framealpha=0.95,
                  edgecolor="0.8", fancybox=False,
                  ncol=min(len(species_names), 4))

        if title:
            ax.set_title(title, pad=10)

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
    p.add_argument("--n-samples", type=int, default=10,
                   help="Albedo draws per face for broadening (default 10)")
    p.add_argument("--lat-half-width", type=float, default=5.0)
    args = p.parse_args()

    print(f"Loading Bond-albedo map from {args.albedo_fits}")
    lat_centers, _, A_2d = pph.load_albedo_map(args.albedo_fits)

    species_union = sorted(set(SPECIES_FOUR) | set(SPECIES_SEVEN),
                           key=SPECIES_SEVEN.index)

    print(f"Computing broadened cold-trap fractions for {len(species_union)} "
          f"species across {len(LATITUDES)} latitudes × {len(TERRAINS)} terrains "
          f"(n_samples_per_face = {args.n_samples})")
    summary, T_stab, _ = compute_fractions(
        species_union, A_2d, lat_centers,
        n_samples_per_face=args.n_samples,
        lat_half_width=args.lat_half_width)

    # Print sanity table
    print("\nSpecies stability temperatures (1 Gyr):")
    for sp in species_union:
        print(f"  {sp:6s}  T_stab = {T_stab[sp]:.1f} K")

    print("\nBroadened cold-trap fraction (% of mesh area, terrain mean):")
    header = "lat   " + "  ".join(f"{sp:>6s}" for sp in species_union)
    print(header)
    for lat in LATITUDES:
        row = f"{lat:+4d}  " + "  ".join(
            f"{summary[sp][lat][0]:6.1f}" for sp in species_union)
        print(row)

    print("\nFigures:")
    plot_one(SPECIES_FOUR, summary, T_stab,
             out_path=args.out_dir / "cold_trap_fraction_albedo_broadened.png",
             title=("Cold-trap fraction vs latitude  "
                    "(topography + Bond-albedo broadening)"))
    plot_one(SPECIES_SEVEN, summary, T_stab,
             out_path=args.out_dir / "cold_trap_fraction_albedo_broadened_extended.png",
             title=("Cold-trap fraction vs latitude  "
                    "(topography + Bond-albedo broadening; extended species set)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
