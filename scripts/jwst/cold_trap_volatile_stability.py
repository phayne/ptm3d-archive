#!/usr/bin/env python3
"""Cold-trap volatile stability across Europa from regional thermal models.

Reads the 45 PTM3D regional terrain runs (9 latitudes × 5 named DEMs) and
produces:

  1. A 3-panel headline figure (T_max, T_min, T_mean) showing the per-face
     temperature distributions vs. latitude with stability lines marked for
     nine volatile species at three timescales (1 yr / 1 Myr / 1 Gyr).

  2. A 3×3 grid of small panels — one per species — showing the fraction
     of mesh area where each species is stable, vs. latitude, with three
     curves per species for the three timescales.

  3. Two CSV summaries: per-(latitude, terrain) and area-weighted across
     all 5 terrains per latitude.

Reuses the validated vapor-pressure tables, sublimation-flux helper, and
Brent's-method stability solver from `scripts/plot_polar_cold_traps.py` and
adds three more species (SO₂, NH₃, CH₃OH).

Usage:
    python scripts/jwst/cold_trap_volatile_stability.py \
        --plot-dir docs/figures/jwst \
        --csv-dir  data/jwst
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

# Apply the IDL family from pretty_plots (Helvetica-family sans-serif,
# inward ticks all four sides, rounded line caps, TrueType-embedded PDFs)
# before any other matplotlib commands so the rcParams take effect.
from pretty_plots import idl  # noqa: E402
idl.use()

# Bring in the validated infrastructure from the canonical cold-trap script.
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "scripts"))
import plot_polar_cold_traps as ppc  # noqa: E402

# ------------------------------------------------------------------
# Extend the species table with three more volatiles.
# Coefficients for log10(P/Pa) = A − B/T are taken from Fray & Schmitt
# (2009) where available, with NH₃ and CH₃OH using the canonical Antoine-
# form fits validated against NIST WebBook / Sandford & Allamandola (1993)
# in the ranges relevant for Europan cold traps (T < 130 K).
# ------------------------------------------------------------------
SEC_PER_GYR = ppc.SEC_PER_GYR
AMU         = ppc.AMU
k_B         = ppc.k_B


def pvap_so2(T):
    """SO₂ ice [Pa]. Fray & Schmitt (2009), valid 60–197 K."""
    T = np.asarray(T, dtype=float)
    return np.where(T > 30, 10.0 ** (14.4 - 2308.5 / T), 0.0)


def pvap_nh3(T):
    """NH₃ ice [Pa]. Fray & Schmitt (2009), valid below 152.7 K."""
    T = np.asarray(T, dtype=float)
    return np.where(T > 30, 10.0 ** (13.69 - 1908.2 / T), 0.0)


def pvap_ch3oh(T):
    """CH₃OH ice [Pa]. NIST WebBook / Lucas et al. (2005), 90–150 K range."""
    T = np.asarray(T, dtype=float)
    return np.where(T > 50, 10.0 ** (14.85 - 2474.0 / T), 0.0)


# Build the full species table (existing six + three new) and labels.
SPECIES = dict(ppc.SPECIES)  # shallow copy
SPECIES["SO2"]   = {"pvap": pvap_so2,   "mass": 64.07  * AMU, "rho": 2050.0,
                    "color": "#56B4E9", "Trange": (40, 200)}
SPECIES["NH3"]   = {"pvap": pvap_nh3,   "mass": 17.03  * AMU, "rho": 817.0,
                    "color": "#117733", "Trange": (40, 160)}
SPECIES["CH3OH"] = {"pvap": pvap_ch3oh, "mass": 32.04  * AMU, "rho": 1010.0,
                    "color": "#882255", "Trange": (50, 180)}

SPECIES_LABELS = dict(ppc.SPECIES_LABELS)
SPECIES_LABELS.update({
    "SO2":   r"$\mathrm{SO_2}$",
    "NH3":   r"$\mathrm{NH_3}$",
    "CH3OH": r"$\mathrm{CH_3OH}$",
})

# Species presented in the talk in the order most-stable to least-stable.
# (H₂O, CH₃OH, CO₂, NH₃, SO₂, H₂S, CH₄, O₂, H₂)
SPECIES_ORDER = ["H2O", "CH3OH", "CO2", "NH3", "SO2", "H2S", "CH4", "O2", "H2"]


def stability_temperature(species_name, loss_rate_m_per_Gyr=1.0):
    """Find T such that the sublimation depth-loss rate equals
    `loss_rate_m_per_Gyr` (in mm/Gyr if 1 m/Gyr → 1000 mm/Gyr scale).

    Same physics as ppc.stability_temperature but reads from the extended
    SPECIES table.
    """
    sp = SPECIES[species_name]
    target = sp["rho"] * (loss_rate_m_per_Gyr * 1e-3) / SEC_PER_GYR
    Tlo, Thi = sp["Trange"]

    def residual(T):
        P = sp["pvap"](T)
        flux = P * np.sqrt(sp["mass"] / (2 * np.pi * k_B * T))
        return flux - target

    try:
        return float(brentq(residual, Tlo, Thi, xtol=0.01))
    except ValueError:
        return float("nan")


# Three timescales of interest (mm of ice lost per Gyr equivalent).
# At 1 mm/Gyr a face holds ice for ~1 Gyr (canonical "Gyr stability").
# At 1000 mm/Gyr it holds ice for ~1 Myr; at 1e9 mm/Gyr it holds ice for ~1 yr.
TIMESCALE_LABELS = ["1 yr", "1 Myr", "1 Gyr"]
TIMESCALE_LOSS_RATE_MMGYR = [1.0e9, 1.0e3, 1.0]
TIMESCALE_LINESTYLES = [":", "--", "-"]


# ------------------------------------------------------------------
# Run loader
# ------------------------------------------------------------------
LATITUDES = [0, 15, 30, 45, 60, 75, 80, 85, 87]
TERRAIN_KEYS = ["rhad", "ridges", "agenor", "cilix", "pwyll"]

BASE_DIR = ppc.BASE_DIR


def run_dir_for(lat, terrain):
    if lat == 87:
        return Path(BASE_DIR) / f"output/europa_polar_{terrain}"
    return Path(BASE_DIR) / f"output/europa_lat{lat:02d}_{terrain}"


def load_run_temperatures(lat, terrain):
    """Return (T_max, T_min, T_mean, n_faces) for one regional run, or None
    if the run is not present."""
    run = run_dir_for(lat, terrain)
    stat = run / "statistics"
    if not stat.is_dir():
        return None
    mesh_path = run / "mesh.ply"
    if not mesh_path.exists():
        # fall back to base mesh from DEM_BASES
        mesh_path = Path(ppc.DEM_BASES[terrain]["mesh"])
    if not mesh_path.exists():
        return None
    vertices, faces = ppc.load_ply_mesh(str(mesh_path))
    n = len(faces)
    Tmax  = ppc.load_binary_stat(str(stat / "temperature_max.bin"), n)
    Tmin  = ppc.load_binary_stat(str(stat / "temperature_min.bin"), n)
    Tmean = ppc.load_binary_stat(str(stat / "temperature_avg.bin"), n)
    return Tmax, Tmin, Tmean, n


# ------------------------------------------------------------------
# Stability fractions
# ------------------------------------------------------------------
def fraction_below(T_arr, threshold):
    """Fraction of faces with temperature below threshold (face-equal-weight).

    Each face in a regional run represents a roughly equal area patch on
    the DEM, so unweighted face fraction is a reasonable area proxy.
    """
    if not np.isfinite(threshold):
        return 0.0
    return float(np.mean(T_arr < threshold))


# ------------------------------------------------------------------
# Aggregation across all runs
# ------------------------------------------------------------------
def aggregate_all_runs(verbose=True):
    """Walk every (lat, terrain) regional run; return dict keyed by
    (lat, terrain) → {Tmax, Tmin, Tmean}, plus a per-latitude aggregate
    that pools all 5 terrains."""
    by_run = {}
    by_lat = {lat: {"Tmax": [], "Tmin": [], "Tmean": []} for lat in LATITUDES}
    for lat in LATITUDES:
        for terrain in TERRAIN_KEYS:
            data = load_run_temperatures(lat, terrain)
            if data is None:
                if verbose:
                    print(f"  [skip] lat={lat}° {terrain}: no data")
                continue
            Tmax, Tmin, Tmean, n = data
            if verbose:
                print(f"  lat={lat:>2d}° {terrain:>6s}: n={n:>5d}  "
                      f"T_max [{Tmax.min():5.1f}, {Tmax.max():5.1f}]  "
                      f"T_min [{Tmin.min():5.1f}, {Tmin.max():5.1f}]")
            by_run[(lat, terrain)] = {"Tmax": Tmax, "Tmin": Tmin, "Tmean": Tmean}
            by_lat[lat]["Tmax"].append(Tmax)
            by_lat[lat]["Tmin"].append(Tmin)
            by_lat[lat]["Tmean"].append(Tmean)
    # Concatenate per-latitude pools
    for lat in LATITUDES:
        for k in ("Tmax", "Tmin", "Tmean"):
            arr_list = by_lat[lat][k]
            by_lat[lat][k] = np.concatenate(arr_list) if arr_list else np.array([])
    return by_run, by_lat


# ------------------------------------------------------------------
# Figure 1: 3-panel T vs lat density
# ------------------------------------------------------------------
def plot_T_vs_lat_summary(by_lat, stability_table, out_path):
    """3-panel figure: T_max / T_min / T_mean density vs latitude.

    Layout: 3 panels (one per stat) stacked horizontally.
    Below the panels: a wide legend zone with species color swatches +
    timescale linestyle key.
    """
    from matplotlib.colors import PowerNorm
    from matplotlib.lines import Line2D
    from scipy.interpolate import RegularGridInterpolator

    plt.rcParams.update({
        "font.size": 14, "axes.labelsize": 16, "axes.titlesize": 17,
        "legend.fontsize": 12, "xtick.labelsize": 13, "ytick.labelsize": 13,
    })

    fig = plt.figure(figsize=(17.0, 9.4))
    # Three rows: title (implicit via suptitle), panels, legend zone.
    # Use generous hspace so the x-axis labels never collide with the legend.
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.42], hspace=0.40,
                          wspace=0.08,
                          left=0.06, right=0.985, top=0.92, bottom=0.04)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    legend_ax = fig.add_subplot(gs[1, :]); legend_ax.axis("off")

    panels = [("Tmax", "Peak temperature"),
              ("Tmin", "Minimum temperature"),
              ("Tmean", "Mean temperature")]
    T_edges = np.linspace(20.0, 165.0, 121)
    T_grid  = 0.5 * (T_edges[:-1] + T_edges[1:])
    lat_centers = np.array(LATITUDES, dtype=float)
    lat_fine = np.linspace(0, 87, 175)

    for panel_idx, (key, label) in enumerate(panels):
        ax = axes[panel_idx]
        density = np.zeros((len(LATITUDES), len(T_grid)))
        for li, lat in enumerate(LATITUDES):
            arr = by_lat[lat][key]
            if arr.size == 0:
                continue
            h, _ = np.histogram(arr, bins=T_edges)
            density[li] = h / max(h.sum(), 1)

        # Interpolate latitudinally for a smooth display
        try:
            interp = RegularGridInterpolator(
                (lat_centers, T_grid), density,
                bounds_error=False, fill_value=0.0)
            LL, TT = np.meshgrid(lat_fine, T_grid, indexing="ij")
            D = interp((LL, TT))
        except Exception:
            D = density
            lat_fine = lat_centers

        # PowerNorm with gamma<1 stretches the low end so faint tails are visible
        vmax = max(np.percentile(D, 99.5), 1e-6)
        im = ax.imshow(D.T, origin="lower",
                       extent=[lat_fine.min(), lat_fine.max(),
                               T_edges.min(), T_edges.max()],
                       aspect="auto", cmap="magma",
                       norm=PowerNorm(gamma=0.45, vmin=0.0, vmax=vmax),
                       interpolation="bilinear")

        # Stability lines for each species at three timescales
        for sp_name in SPECIES_ORDER:
            color = SPECIES[sp_name]["color"]
            for ti in range(len(TIMESCALE_LOSS_RATE_MMGYR)):
                T_stab = stability_table[(sp_name, ti)]
                if not np.isfinite(T_stab):
                    continue
                if T_stab < T_edges[0] or T_stab > T_edges[-1]:
                    continue
                ls = TIMESCALE_LINESTYLES[ti]
                ax.axhline(T_stab, color=color, lw=1.6, ls=ls, alpha=0.95,
                           zorder=5)

        ax.set_xlim(0, 87)
        ax.set_xticks([0, 15, 30, 45, 60, 75, 87])
        ax.set_xlabel("Latitude (°N)")
        if panel_idx == 0:
            ax.set_ylabel("Temperature (K)")
        else:
            ax.set_yticklabels([])
        ax.set_title(label, fontsize=16)
        ax.set_facecolor("#1a0e1f")  # dark background so faint tails read
        ax.tick_params(colors="black")
        for spine in ax.spines.values():
            spine.set_edgecolor("black")

    # ---- Legend zone (in legend_ax) ----
    species_handles = []
    for sp_name in SPECIES_ORDER:
        T_stab_gyr = stability_table[(sp_name, 2)]
        lab = SPECIES_LABELS[sp_name]
        if np.isfinite(T_stab_gyr):
            lab = f"{lab}  ({T_stab_gyr:.0f} K)"
        species_handles.append(
            Line2D([], [], color=SPECIES[sp_name]["color"],
                   lw=3.0, ls="-", label=lab))
    style_handles = [
        Line2D([], [], color="0.15", lw=2.5, ls="-",  label="stable for 1 Gyr"),
        Line2D([], [], color="0.15", lw=2.5, ls="--", label="stable for 1 Myr"),
        Line2D([], [], color="0.15", lw=2.5, ls=":",  label="stable for 1 yr"),
    ]
    leg1 = legend_ax.legend(handles=species_handles, loc="upper left",
                            bbox_to_anchor=(0.0, 1.0), ncol=3,
                            frameon=False, handlelength=2.6,
                            title="Volatile cold-trap stability "
                                  "(temperature shown is for 1-Gyr loss)",
                            title_fontsize=13)
    legend_ax.add_artist(leg1)
    legend_ax.legend(handles=style_handles, loc="upper right",
                     bbox_to_anchor=(1.0, 1.0), ncol=1,
                     frameon=False, handlelength=2.6,
                     title="Timescale (line style)",
                     title_fontsize=13)

    fig.suptitle("Surface temperature distributions and "
                 "volatile cold-trap thresholds across Europa",
                 fontsize=18, y=0.985)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"  wrote {out_path}")


# ------------------------------------------------------------------
# Figure 2: 3x3 species panels — fraction-stable vs latitude
# ------------------------------------------------------------------
def plot_stability_grid(by_run, stability_table, out_path):
    """3×3 grid: one panel per species, each shows fraction of faces stable
    vs. latitude with three curves for three timescales."""
    plt.rcParams.update({
        "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 14,
        "legend.fontsize": 9, "xtick.labelsize": 11, "ytick.labelsize": 11,
    })
    fig, axes = plt.subplots(3, 3, figsize=(13.6, 11.5),
                             gridspec_kw={"hspace": 0.36, "wspace": 0.28})
    axes = axes.ravel()

    for sp_idx, sp_name in enumerate(SPECIES_ORDER):
        ax = axes[sp_idx]
        for ti, tau_label in enumerate(TIMESCALE_LABELS):
            T_stab = stability_table[(sp_name, ti)]
            ls = TIMESCALE_LINESTYLES[ti]
            if not np.isfinite(T_stab):
                continue
            # Fraction stable vs latitude — averaged across all 5 terrains
            fracs = []
            for lat in LATITUDES:
                vals = []
                for terrain in TERRAIN_KEYS:
                    if (lat, terrain) not in by_run:
                        continue
                    Tmax = by_run[(lat, terrain)]["Tmax"]
                    vals.append(fraction_below(Tmax, T_stab))
                fracs.append(np.mean(vals) if vals else np.nan)
            fracs = np.array(fracs)
            ax.plot(LATITUDES, 100.0 * fracs,
                    ls=ls, lw=2.4, color=SPECIES[sp_name]["color"],
                    marker="o", markersize=5,
                    label=tau_label)
        ax.set_xlim(-2, 90)
        ax.set_xticks([0, 30, 45, 60, 75, 87])
        ax.set_ylim(-3, 105)
        ax.set_xlabel("Latitude (°N)")
        ax.set_ylabel("Stable fraction (%)")
        T_stab_gyr = stability_table[(sp_name, 2)]
        title = SPECIES_LABELS[sp_name]
        if np.isfinite(T_stab_gyr):
            title += f"   T_stab(1 Gyr) = {T_stab_gyr:.0f} K"
        ax.set_title(title)
        ax.grid(alpha=0.25)
        if sp_idx == 0:
            ax.legend(loc="upper left", framealpha=0.95, title="Timescale")

    fig.suptitle("Fraction of mesh area where each volatile is "
                 "thermally stable, vs. latitude (averaged across "
                 "all 5 terrains)", fontsize=15, y=0.995)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"  wrote {out_path}")


# ------------------------------------------------------------------
# CSV summaries
# ------------------------------------------------------------------
def write_csv_per_run(by_run, stability_table, out_csv):
    """Per-(lat, terrain) fractional-stability table."""
    cols = ["lat_deg", "terrain", "n_faces"]
    species_cols = []
    for sp_name in SPECIES_ORDER:
        for ti, tau_label in enumerate(TIMESCALE_LABELS):
            cn = f"{sp_name}_{tau_label.replace(' ', '')}"
            cols.append(cn)
            species_cols.append((sp_name, ti, cn))

    rows = []
    for lat in LATITUDES:
        for terrain in TERRAIN_KEYS:
            if (lat, terrain) not in by_run:
                continue
            Tmax = by_run[(lat, terrain)]["Tmax"]
            row = {"lat_deg": lat, "terrain": terrain, "n_faces": Tmax.size}
            for sp_name, ti, cn in species_cols:
                T_stab = stability_table[(sp_name, ti)]
                row[cn] = round(100.0 * fraction_below(Tmax, T_stab), 4)
            rows.append(row)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {out_csv}  ({len(rows)} rows)")


def write_csv_terrain_avg(by_run, stability_table, out_csv):
    """Per-latitude average across all 5 terrains."""
    cols = ["lat_deg"]
    species_cols = []
    for sp_name in SPECIES_ORDER:
        for ti, tau_label in enumerate(TIMESCALE_LABELS):
            cn = f"{sp_name}_{tau_label.replace(' ', '')}"
            cols.append(cn)
            species_cols.append((sp_name, ti, cn))

    rows = []
    for lat in LATITUDES:
        row = {"lat_deg": lat}
        for sp_name, ti, cn in species_cols:
            T_stab = stability_table[(sp_name, ti)]
            vals = []
            for terrain in TERRAIN_KEYS:
                if (lat, terrain) not in by_run:
                    continue
                Tmax = by_run[(lat, terrain)]["Tmax"]
                vals.append(fraction_below(Tmax, T_stab))
            row[cn] = round(100.0 * np.mean(vals), 4) if vals else float("nan")
        rows.append(row)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {out_csv}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plot-dir", type=Path,
                   default=Path("docs/figures/jwst"))
    p.add_argument("--csv-dir",  type=Path,
                   default=Path("data/jwst"))
    args = p.parse_args()

    # Build the stability-temperature table (species × timescale)
    print("Computing stability temperatures:")
    print("  species  T(1 yr)  T(1 Myr)  T(1 Gyr)  K")
    stability_table = {}
    for sp_name in SPECIES_ORDER:
        Ts = []
        for ti, lr in enumerate(TIMESCALE_LOSS_RATE_MMGYR):
            T = stability_temperature(sp_name, loss_rate_m_per_Gyr=lr)
            stability_table[(sp_name, ti)] = T
            Ts.append(T)
        print(f"  {sp_name:6s}  {Ts[0]:6.1f}   {Ts[1]:6.1f}    {Ts[2]:6.1f}")

    print("\nLoading regional runs:")
    by_run, by_lat = aggregate_all_runs(verbose=True)
    print(f"  loaded {len(by_run)} (lat, terrain) runs")

    print("\nFigures:")
    plot_T_vs_lat_summary(
        by_lat, stability_table,
        args.plot_dir / "cold_trap_T_vs_lat.png",
    )
    plot_stability_grid(
        by_run, stability_table,
        args.plot_dir / "cold_trap_stability_vs_lat.png",
    )

    print("\nCSV summaries:")
    write_csv_per_run(
        by_run, stability_table,
        args.csv_dir / "cold_trap_fractions_summary.csv",
    )
    write_csv_terrain_avg(
        by_run, stability_table,
        args.csv_dir / "cold_trap_fractions_terrain_avg.csv",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
