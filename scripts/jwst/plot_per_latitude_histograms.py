#!/usr/bin/env python3
"""Per-latitude temperature histograms (T_max, T_min, T_mean) at five
representative latitudes: 0°, 30°, 60°, 75°, 85°.

Two figures are produced:

  (a) cold_trap_T_histograms.png — model output as-is. Each face contributes
      one count to the histogram for its run latitude / temperature
      statistic. All 5 named terrains at each latitude are pooled.

  (b) cold_trap_T_histograms_albedo_broadened.png — same panels but each
      face contributes N samples obtained by drawing N albedos from the
      Mergny et al. Bond-albedo distribution within ±5° of that latitude
      and rescaling temperatures via radiative equilibrium:

          T_new = T_model × ((1 − A_new) / (1 − A_ref))^(1/4)

      where A_ref = 0.5 is the uniform Bond albedo the regional thermal
      runs assumed.

Vertical lines mark each volatile's 1-Gyr stability temperature.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

# Apply the IDL family from pretty_plots
# (https://github.com/phayne/pretty-plots). Base IDL — Helvetica-family
# sans-serif, inward ticks on all four sides, rounded line caps, embedded
# TrueType fonts for PDF. Talk-friendly font sizes are layered on top via
# per-figure rc_context overrides.
from pretty_plots import idl  # noqa: E402  (must precede other matplotlib calls)
idl.use()

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import cold_trap_volatile_stability as cts  # noqa: E402

A_REF = 0.5
# Northern latitudes for which regional thermal runs exist on disk.
LATS_NORTH = [0, 30, 60, 75, 85]
# Southern latitudes are produced by mirroring the corresponding northern
# runs under the (excellent) approximation that Europa's 0.1° axial tilt
# makes the diurnal-cycle statistics N–S symmetric. The mirror simply
# relabels lat → −lat without rerunning the thermal model.
LATS_SOUTH = [-85, -75, -60, -30]
# Plot order: Northern latitudes on top, Southern on the bottom (geographic
# convention with north up).
LATS = sorted(LATS_NORTH + LATS_SOUTH, reverse=True)
# |lat| bins used for the N–S overlay figure.
ABS_LATS_OVERLAY = [0, 30, 60, 75, 85]
STATS = [("Tmax",  "Peak temperature"),
         ("Tmin",  "Minimum temperature"),
         ("Tmean", "Mean temperature")]
TERRAINS = cts.TERRAIN_KEYS

# Subset of species to overlay (keeping the cold-trap relevant ones)
OVERLAY_SPECIES = ["H2O", "CH3OH", "CO2", "NH3", "SO2", "H2S", "CH4"]


def load_albedo_map(path: Path):
    """Return (lat_centers_deg, lon_centers_deg, A_2d) with sentinel fill removed."""
    with fits.open(path) as f:
        data = np.asarray(f[0].data, dtype=float)
    H, W = data.shape  # (180, 360) for 1-deg map
    lat_c = np.linspace(-90 + 0.5, 90 - 0.5, H)
    lon_c = np.linspace(0.5, 360 - 0.5, W)
    # Filter sentinel values: valid Bond albedos for Europa span ~0.05-0.9
    A = np.where((data > 0.02) & (data < 0.98), data, np.nan)
    return lat_c, lon_c, A


def collect_lat_temps(lat_target):
    """Pool T_max/T_min/T_mean across all 5 terrains at one latitude.

    Negative latitudes are populated by mirroring the corresponding
    northern run (no separate southern thermal sims exist). This is
    justified by Europa's ~0.1° obliquity, which makes the diurnal
    statistics very nearly N–S symmetric.
    """
    lookup_lat = abs(lat_target)
    out = {"Tmax": [], "Tmin": [], "Tmean": []}
    for terrain in TERRAINS:
        data = cts.load_run_temperatures(lookup_lat, terrain)
        if data is None:
            continue
        Tmax, Tmin, Tmean, _ = data
        out["Tmax"].append(Tmax)
        out["Tmin"].append(Tmin)
        out["Tmean"].append(Tmean)
    return {k: np.concatenate(v) if v else np.array([]) for k, v in out.items()}


def albedos_in_band(A_2d, lat_centers, lat_target, half_width_deg=5.0):
    """Valid Bond albedos within ±half_width of target latitude.

    Unlike the temperature loaders this respects the actual sign of the
    target latitude — we sample the Mergny map at the true latitude (north
    or south), not at its mirror. The two hemispheres have different
    geological signatures (different chaos terrains, plumes, etc.).
    """
    band = np.abs(lat_centers - lat_target) <= half_width_deg
    vals = A_2d[band, :].ravel()
    return vals[np.isfinite(vals)]


def broaden_with_albedo(T_arr, A_band, A_ref=A_REF, n_samples_per_face=8,
                        rng_seed=0):
    """Rescale each face's T by the radiative-equilibrium albedo factor for
    n_samples_per_face randomly drawn albedos from `A_band`."""
    rng = np.random.default_rng(rng_seed)
    if A_band.size == 0 or T_arr.size == 0:
        return T_arr.copy()
    A_draws = rng.choice(A_band, size=(T_arr.size, n_samples_per_face),
                         replace=True)
    factor = ((1.0 - A_draws) / (1.0 - A_ref)) ** 0.25
    T_broad = (T_arr[:, None] * factor).ravel()
    return T_broad


def plot_grid(lat_temps_by_lat, stability_table,
              T_edges, title_suffix, out_path):
    # Talk-friendly font sizes (substantially larger than icarus journal defaults).
    with plt.rc_context({
        "font.size": 16, "axes.labelsize": 17, "axes.titlesize": 19,
        "legend.fontsize": 15, "xtick.labelsize": 15,
        "ytick.labelsize": 15, "figure.titlesize": 21,
    }):
        return _plot_grid_inner(lat_temps_by_lat, stability_table,
                                T_edges, title_suffix, out_path)


def _plot_grid_inner(lat_temps_by_lat, stability_table,
                     T_edges, title_suffix, out_path):
    n_rows = len(LATS)
    fig, axes = plt.subplots(n_rows, 3,
                             figsize=(13.5, 1.55 * n_rows + 1.0),
                             sharex=True, sharey="row",
                             gridspec_kw={"hspace": 0.32, "wspace": 0.10})

    def lat_label(lat_deg):
        if lat_deg == 0:
            return "0°"
        return f"{abs(lat_deg)}°{'N' if lat_deg > 0 else 'S'}"

    for row, lat in enumerate(LATS):
        per_stat = lat_temps_by_lat[lat]
        for col, (key, label) in enumerate(STATS):
            ax = axes[row, col]
            arr = per_stat[key]
            if arr.size:
                ax.hist(arr, bins=T_edges, density=True,
                        color="#446699", alpha=0.85, edgecolor="white",
                        linewidth=0.3)
            else:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                        ha="center", va="center", color="gray")

            # Stability lines
            for sp_name in OVERLAY_SPECIES:
                T_stab = stability_table[(sp_name, 2)]  # 1 Gyr
                if not np.isfinite(T_stab):
                    continue
                if T_stab < T_edges[0] or T_stab > T_edges[-1]:
                    continue
                color = cts.SPECIES[sp_name]["color"]
                ax.axvline(T_stab, color=color, lw=1.6, alpha=0.85, zorder=4)

            ax.set_xlim(T_edges[0], T_edges[-1])
            if row == 0:
                ax.set_title(label)
            if col == 0:
                ax.set_ylabel(f"{lat_label(lat)}\n\nDensity", fontsize=12)
            if row == n_rows - 1:
                ax.set_xlabel("Temperature (K)")
            ax.grid(alpha=0.25, axis="x")

    # Build a combined legend below
    from matplotlib.lines import Line2D
    handles = []
    for sp_name in OVERLAY_SPECIES:
        T = stability_table[(sp_name, 2)]
        lab = cts.SPECIES_LABELS[sp_name]
        if np.isfinite(T):
            lab = f"{lab} ({T:.0f} K)"
        handles.append(Line2D([], [], color=cts.SPECIES[sp_name]["color"],
                              lw=2.4, label=lab))
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.005), ncol=len(OVERLAY_SPECIES),
               frameon=False, title="1-Gyr stability temperature",
               title_fontsize=12)

    fig.suptitle(f"Surface temperature distributions by latitude{title_suffix}",
                 fontsize=15, y=0.995)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"  wrote {out_path}")


def plot_NS_overlay(lat_temps_by_lat, stability_table, T_edges,
                    title_suffix, out_path):
    """One panel per |latitude|; overlays Northern (solid blue fill) and
    Southern (orange hatch, no fill) peak-temperature histograms.
    Distinguishable in colour and in greyscale: only the southern
    distribution is hatched, the northern is a translucent solid fill.
    """
    with plt.rc_context({
        "font.size": 16, "axes.labelsize": 17, "axes.titlesize": 19,
        "legend.fontsize": 15, "xtick.labelsize": 15, "ytick.labelsize": 15,
        "hatch.linewidth": 1.1,
    }):
        return _plot_NS_overlay_inner(lat_temps_by_lat, stability_table,
                                       T_edges, title_suffix, out_path)


def _plot_NS_overlay_inner(lat_temps_by_lat, stability_table, T_edges,
                            title_suffix, out_path):
    n_panels = len(ABS_LATS_OVERLAY)
    fig, axes = plt.subplots(1, n_panels,
                             figsize=(4.0 * n_panels + 1.0, 6.6),
                             sharey=True,
                             gridspec_kw={"wspace": 0.10,
                                          "left": 0.06, "right": 0.99,
                                          "top": 0.88, "bottom": 0.32})

    # Asymmetric styling — only the southern histogram is hatched, the
    # northern is a translucent solid fill. This is much clearer than two
    # competing hatch directions, especially when N ≈ S.
    style_N = dict(facecolor="#1f6db8", edgecolor="#0a3a6e",
                   alpha=0.55, hatch="", linewidth=1.6)
    style_S = dict(facecolor="none",   edgecolor="#c45a3e",
                   alpha=1.00, hatch="////", linewidth=1.6)

    for i, abs_lat in enumerate(ABS_LATS_OVERLAY):
        ax = axes[i]
        n_arr = lat_temps_by_lat.get( abs_lat, {}).get("Tmax", np.array([]))
        # At the equator there is no separate "south" run, but we still
        # plot the same data with the southern style so both "North" and
        # "South" appear in the legend (and the N/S styles overlap exactly
        # at the equator, which is geographically correct).
        s_arr = (lat_temps_by_lat.get(-abs_lat, {}).get("Tmax", np.array([]))
                 if abs_lat != 0 else n_arr)

        if n_arr.size:
            ax.hist(n_arr, bins=T_edges, density=True,
                    label="North", **style_N)
        if s_arr is not None and s_arr.size:
            ax.hist(s_arr, bins=T_edges, density=True,
                    label="South", **style_S)

        # Stability lines for headline species
        for sp_name in OVERLAY_SPECIES:
            T_stab = stability_table[(sp_name, 2)]
            if not np.isfinite(T_stab):
                continue
            if T_stab < T_edges[0] or T_stab > T_edges[-1]:
                continue
            ax.axvline(T_stab, color=cts.SPECIES[sp_name]["color"],
                       lw=1.4, alpha=0.85, zorder=4)

        ax.set_xlim(T_edges[0], T_edges[-1])
        ax.set_xlabel("Peak temperature (K)")
        if i == 0:
            ax.set_ylabel("Density")
            ax.legend(loc="upper left", framealpha=0.95)
        ax.set_title(f"|lat| = {abs_lat}°" if abs_lat else "Equator")
        ax.grid(alpha=0.25, axis="x")

    # Species legend on a second row below
    from matplotlib.lines import Line2D
    handles = []
    for sp_name in OVERLAY_SPECIES:
        T = stability_table[(sp_name, 2)]
        lab = cts.SPECIES_LABELS[sp_name]
        if np.isfinite(T):
            lab = f"{lab} ({T:.0f} K)"
        handles.append(Line2D([], [], color=cts.SPECIES[sp_name]["color"],
                              lw=2.4, label=lab))
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.04), ncol=len(OVERLAY_SPECIES),
               frameon=False, title="1-Gyr stability temperature",
               title_fontsize=11)

    fig.suptitle(f"Peak surface temperature: north vs south{title_suffix}",
                 fontsize=14, y=0.97)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    fig.savefig(out_path.with_suffix(".pdf"))
    print(f"  wrote {out_path}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--albedo-fits", type=Path,
                   default=Path("data/albedo_maps/Europa-1deg-BondAlbedo.fits"))
    p.add_argument("--out-dir", type=Path,
                   default=Path("docs/figures/jwst"))
    p.add_argument("--n-broadening-samples", type=int, default=10,
                   help="Number of albedo draws per face for the broadened "
                        "histogram (default 10)")
    p.add_argument("--lat-half-width", type=float, default=5.0,
                   help="Half-width of latitude band used to extract Mergny "
                        "albedo distribution at each target latitude (deg)")
    args = p.parse_args()

    # Build species stability table once (1 Gyr, 1 Myr, 1 yr — but we only
    # display the 1 Gyr line here for clarity).
    stability_table = {}
    for sp_name in cts.SPECIES_ORDER:
        for ti, lr in enumerate(cts.TIMESCALE_LOSS_RATE_MMGYR):
            stability_table[(sp_name, ti)] = cts.stability_temperature(
                sp_name, loss_rate_m_per_Gyr=lr)

    # Load model temperatures pooled per latitude
    print("Loading regional runs at the 5 target latitudes...")
    lat_temps = {}
    for lat in LATS:
        lat_temps[lat] = collect_lat_temps(lat)
        n = lat_temps[lat]["Tmax"].size
        if n == 0:
            print(f"  lat={lat}°: NO DATA")
            continue
        print(f"  lat={lat}°: {n} faces total across all terrains; "
              f"T_max [{lat_temps[lat]['Tmax'].min():.1f}, "
              f"{lat_temps[lat]['Tmax'].max():.1f}] K")

    # Load Mergny albedo
    print(f"\nLoading Mergny Bond-albedo map from {args.albedo_fits}")
    lat_c, lon_c, A_map = load_albedo_map(args.albedo_fits)
    band_albedos = {}
    for lat in LATS:
        A_band = albedos_in_band(A_map, lat_c, lat,
                                 half_width_deg=args.lat_half_width)
        band_albedos[lat] = A_band
        if A_band.size:
            print(f"  lat~{lat:>3d}°: A range [{A_band.min():.3f}, "
                  f"{A_band.max():.3f}], mean={A_band.mean():.3f}, "
                  f"std={A_band.std():.3f}, n={A_band.size}")

    # ------- Figure A: model-as-is histograms -------
    print("\nFigure A: model-as-is histograms")
    T_edges = np.linspace(20.0, 165.0, 60)
    plot_grid(lat_temps, stability_table, T_edges,
              title_suffix=f"  (uniform A = {A_REF})",
              out_path=args.out_dir / "cold_trap_T_histograms.png")

    # ------- Figure B: albedo-broadened histograms -------
    print("\nFigure B: Mergny-albedo-broadened histograms")
    lat_temps_broad = {}
    for lat in LATS:
        per_stat = lat_temps[lat]
        A_band = band_albedos[lat]
        broad = {}
        for k in ("Tmax", "Tmin", "Tmean"):
            broad[k] = broaden_with_albedo(per_stat[k], A_band,
                                           A_ref=A_REF,
                                           n_samples_per_face=args.n_broadening_samples,
                                           rng_seed=hash(f"{lat}{k}") & 0xffff)
        lat_temps_broad[lat] = broad

    plot_grid(lat_temps_broad, stability_table, T_edges,
              title_suffix=f"  — broadened by the Bond-albedo "
                           f"distribution (T scales as (1−A)$^{{1/4}}$)",
              out_path=args.out_dir / "cold_trap_T_histograms_albedo_broadened.png")

    # ------- Figures C, D: north–south overlay (peak T only) -------
    print("\nFigure C: N–S peak-T overlay (uniform A)")
    plot_NS_overlay(lat_temps, stability_table, T_edges,
                    title_suffix=f"  (uniform A = {A_REF})",
                    out_path=args.out_dir / "cold_trap_T_NS_overlay.png")
    print("\nFigure D: N–S peak-T overlay (albedo-broadened)")
    plot_NS_overlay(lat_temps_broad, stability_table, T_edges,
                    title_suffix="  — Bond-albedo broadened",
                    out_path=args.out_dir / "cold_trap_T_NS_overlay_albedo_broadened.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
