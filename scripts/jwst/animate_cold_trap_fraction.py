#!/usr/bin/env python3
"""Animated cold-trap fraction: log-time sweep from 1 yr → 1 Gyr.

Replays the same H₂O / CO₂ × {no-topography, topography, topography+albedo}
six-curve comparison shown in `plot_cold_trap_fraction_three_models.py`,
but as a single MP4/GIF where the timescale advances logarithmically in
each frame. Curves are PCHIP-interpolated to 5° latitude resolution for
smooth animated display.

Outputs (in `docs/figures/jwst/`):
  cold_trap_fraction_animation.mp4   (h.264, primary)
  cold_trap_fraction_animation.gif   (Pillow, companion)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.interpolate import PchipInterpolator

# IDL family from pretty_plots — match the rest of the talk deck
from pretty_plots import idl  # noqa: E402
idl.use()

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import cold_trap_volatile_stability as cts            # noqa: E402
import plot_per_latitude_histograms as pph             # noqa: E402
import plot_cold_trap_fraction_three_models as ctfm    # noqa: E402

A_REF = pph.A_REF
SPECIES_TO_PLOT = ctfm.SPECIES_TO_PLOT       # ["H2O", "CO2"]
SPECIES_COLOR = ctfm.SPECIES_COLOR
LINESTYLE = ctfm.LINESTYLE                    # {no_topo: ":", topo: "--", topo_albedo: "-"}
LATITUDES = ctfm.LATITUDES                    # 17 latitudes pole-to-pole
TERRAINS = ctfm.TERRAINS                      # 5 named DEMs

# 5° interpolation grid for the animated curves.
LAT_GRID = np.linspace(-87.0, 87.0, 35)


# --------------------------------------------------------------------------
# Pre-compute the per-(lat, terrain) temperature arrays once.
# --------------------------------------------------------------------------
def precompute(A_2d, lat_centers, n_albedo_samples=10, lat_half_width=5.0):
    """Load every regional T_max, build the broadened-T array, and cache the
    flat-surface T_max proxy. Returns
        cache[(lat, terrain)] = {"T_max": ..., "T_albedo": ..., "T_flat": ...}
    """
    cache = {}
    for lat in LATITUDES:
        A_band = pph.albedos_in_band(A_2d, lat_centers, lat,
                                     half_width_deg=lat_half_width)
        for terrain in TERRAINS:
            data = cts.load_run_temperatures(abs(lat), terrain)
            if data is None:
                continue
            T_max = data[0]
            T_flat = ctfm.flat_surface_T_max(T_max)
            if A_band.size and n_albedo_samples > 0:
                T_albedo = pph.broaden_with_albedo(
                    T_max, A_band, A_ref=A_REF,
                    n_samples_per_face=n_albedo_samples,
                    rng_seed=hash((lat, terrain)) & 0xffff)
            else:
                T_albedo = T_max.copy()
            cache[(lat, terrain)] = {"T_max": T_max, "T_albedo": T_albedo,
                                     "T_flat": T_flat}
    return cache


# --------------------------------------------------------------------------
# Per-frame fraction computation
# --------------------------------------------------------------------------
def fractions_for_timescale(cache, loss_rate_mm_per_Gyr):
    """Return (T_sub_dict, frac[(model, sp)] = ndarray over LATITUDES)."""
    T_sub = {sp: cts.stability_temperature(sp, loss_rate_m_per_Gyr=loss_rate_mm_per_Gyr)
             for sp in SPECIES_TO_PLOT}
    frac = {("no_topo", sp): np.full(len(LATITUDES), np.nan) for sp in SPECIES_TO_PLOT}
    frac.update({("topo", sp): np.full(len(LATITUDES), np.nan) for sp in SPECIES_TO_PLOT})
    frac.update({("topo_albedo", sp): np.full(len(LATITUDES), np.nan)
                 for sp in SPECIES_TO_PLOT})

    for li, lat in enumerate(LATITUDES):
        per_terrain = {("no_topo", sp): [] for sp in SPECIES_TO_PLOT}
        per_terrain.update({("topo", sp): [] for sp in SPECIES_TO_PLOT})
        per_terrain.update({("topo_albedo", sp): [] for sp in SPECIES_TO_PLOT})
        for terrain in TERRAINS:
            entry = cache.get((lat, terrain))
            if entry is None:
                continue
            T_max = entry["T_max"]
            T_albedo = entry["T_albedo"]
            T_flat = entry["T_flat"]
            for sp in SPECIES_TO_PLOT:
                Ts = T_sub[sp]
                per_terrain[("no_topo", sp)].append(
                    100.0 if (np.isfinite(T_flat) and T_flat < Ts) else 0.0)
                per_terrain[("topo", sp)].append(
                    100.0 * float(np.mean(T_max < Ts)))
                per_terrain[("topo_albedo", sp)].append(
                    100.0 * float(np.mean(T_albedo < Ts)))
        for key, vals in per_terrain.items():
            if vals:
                frac[key][li] = float(np.mean(vals))
    return T_sub, frac


# --------------------------------------------------------------------------
# PCHIP interpolation onto the 5° latitude grid.
# --------------------------------------------------------------------------
def interpolate_to_grid(frac_at_data_lats):
    """Apply PCHIP to a (model, species) curve and evaluate on LAT_GRID,
    clipping to [0, 100] for safety (PCHIP shouldn't overshoot, but the
    boundary points at ±87 fall slightly inside the grid edges)."""
    interp = PchipInterpolator(np.asarray(LATITUDES, dtype=float),
                                frac_at_data_lats, extrapolate=True)
    return np.clip(interp(LAT_GRID), 0.0, 100.0)


# --------------------------------------------------------------------------
# Timescale formatting
# --------------------------------------------------------------------------
def loss_rate_to_years(loss_rate_mm_per_Gyr):
    return 1.0e9 / loss_rate_mm_per_Gyr


def format_timescale(years: float) -> str:
    """Return a compact human-friendly timescale label. Picks the unit
    so the leading number stays in 1–999."""
    if years < 1.0:
        return f"{years*365.25:.1f} day"
    if years < 1.0e3:
        return f"{years:.1f} yr" if years < 10 else f"{years:.0f} yr"
    if years < 1.0e6:
        v = years / 1.0e3
        return f"{v:.1f} kyr" if v < 10 else f"{v:.0f} kyr"
    if years < 1.0e9:
        v = years / 1.0e6
        return f"{v:.1f} Myr" if v < 10 else f"{v:.0f} Myr"
    v = years / 1.0e9
    return f"{v:.2f} Gyr" if v < 10 else f"{v:.1f} Gyr"


# --------------------------------------------------------------------------
# Build the figure scaffolding (axes, fixed annotations, line artists).
# --------------------------------------------------------------------------
def build_figure():
    plt.rcParams.update({
        "font.size": 14, "axes.labelsize": 17, "axes.titlesize": 18,
        "legend.fontsize": 13, "xtick.labelsize": 14, "ytick.labelsize": 14,
    })
    fig, ax = plt.subplots(figsize=(9.6, 6.4))
    ax.set_xlim(-92, 92)
    ax.set_ylim(-2, 105)
    ax.set_xticks([-87, -75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75, 87])
    ax.set_xlabel("Latitude (°)")
    ax.set_ylabel("Cold trap fraction (%)")
    ax.tick_params(direction="in", top=True, right=True)
    ax.minorticks_on()
    ax.tick_params(which="minor", direction="in", top=True, right=True, length=3)
    ax.grid(True, alpha=0.25, linewidth=0.5)

    # Six line artists, smooth (no markers — interpolation supplies the look).
    lines = {}
    for sp in SPECIES_TO_PLOT:
        for model in ("no_topo", "topo", "topo_albedo"):
            ln, = ax.plot([], [], color=SPECIES_COLOR[sp],
                          linestyle=LINESTYLE[model], linewidth=2.4,
                          solid_capstyle="round")
            lines[(model, sp)] = ln

    # Static linestyle key (upper center)
    style_handles = [
        Line2D([], [], color="0.2", lw=2.0, ls=LINESTYLE["topo_albedo"],
               label="topography + albedo"),
        Line2D([], [], color="0.2", lw=2.0, ls=LINESTYLE["topo"],
               label="topography only"),
        Line2D([], [], color="0.2", lw=2.0, ls=LINESTYLE["no_topo"],
               label="no topography or albedo"),
    ]
    ax.legend(handles=style_handles, loc="upper center",
              framealpha=0.95, edgecolor="0.8", fancybox=False,
              ncol=1, fontsize=12)

    # Family labels (positions matched to the static figure)
    bbox_kw = dict(boxstyle="round,pad=0.25", facecolor="white",
                   edgecolor="none", alpha=0.88)
    ax.annotate("$\\mathrm{H_2O}$",
                xy=(-35, 52), color=SPECIES_COLOR["H2O"],
                fontsize=22, fontweight="bold", ha="center", va="center",
                bbox=bbox_kw, zorder=10)
    ax.annotate("$\\mathrm{CO_2}$",
                xy=(-72, 52), color=SPECIES_COLOR["CO2"],
                fontsize=22, fontweight="bold", ha="center", va="center",
                bbox=bbox_kw, zorder=10)

    # Updatable annotations: T_sub captions and the timescale tag.
    txt_h2o = ax.annotate("$T_\\mathrm{sub}$ = ?",
                          xy=(-35, 41), color=SPECIES_COLOR["H2O"],
                          fontsize=12, ha="center", va="center",
                          bbox=bbox_kw, zorder=10)
    txt_co2 = ax.annotate("$T_\\mathrm{sub}$ = ?",
                          xy=(-72, 41), color=SPECIES_COLOR["CO2"],
                          fontsize=12, ha="center", va="center",
                          bbox=bbox_kw, zorder=10)
    txt_tau = ax.annotate("τ = 1 yr",
                          xy=(0.97, 0.97), xycoords="axes fraction",
                          fontsize=22, fontweight="bold", color="0.15",
                          ha="right", va="top",
                          bbox=dict(boxstyle="round,pad=0.4",
                                    facecolor="white",
                                    edgecolor="0.7", linewidth=1.0),
                          zorder=11)

    fig.tight_layout()
    return fig, ax, lines, txt_h2o, txt_co2, txt_tau


# --------------------------------------------------------------------------
# Main animation driver.
# --------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--albedo-fits", type=Path,
                   default=Path("data/albedo_maps/Europa-1deg-BondAlbedo.fits"))
    p.add_argument("--out-dir", type=Path,
                   default=Path("docs/figures/jwst"))
    p.add_argument("--n-frames", type=int, default=60,
                   help="Total frames in the animation (default 60)")
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--n-albedo-samples", type=int, default=10)
    p.add_argument("--gif", action="store_true",
                   help="Also save a GIF copy (slower but universally portable)")
    args = p.parse_args()

    print(f"Loading Bond-albedo map from {args.albedo_fits}")
    lat_centers, _, A_2d = pph.load_albedo_map(args.albedo_fits)

    print(f"Pre-computing T_max arrays for {len(LATITUDES)} latitudes "
          f"× {len(TERRAINS)} terrains (this takes ~10–20 s)…")
    t0 = time.time()
    cache = precompute(A_2d, lat_centers,
                        n_albedo_samples=args.n_albedo_samples)
    print(f"  done in {time.time()-t0:.1f} s; cached {len(cache)} runs")

    # Per-frame timescales: log-uniform from 1 yr (loss_rate=1e9) to 1 Gyr
    # (loss_rate=1.0). loss_rate is mm/Gyr; smaller values = longer τ.
    log_tau_yr = np.linspace(0.0, 9.0, args.n_frames)        # log10 years
    loss_rates = 1.0e9 / np.power(10.0, log_tau_yr)
    timescales_yr = np.power(10.0, log_tau_yr)

    # Build figure once
    fig, ax, lines, txt_h2o, txt_co2, txt_tau = build_figure()

    # Pre-compute every frame's curves (fast; the heavy data load is done)
    print(f"Computing {args.n_frames} frames…")
    t0 = time.time()
    per_frame_curves = []
    per_frame_T_sub = []
    for f_idx, lr in enumerate(loss_rates):
        T_sub, frac = fractions_for_timescale(cache, lr)
        smoothed = {}
        for key, raw in frac.items():
            smoothed[key] = interpolate_to_grid(raw)
        per_frame_curves.append(smoothed)
        per_frame_T_sub.append(T_sub)
    print(f"  done in {time.time()-t0:.1f} s")

    def init():
        for ln in lines.values():
            ln.set_data([], [])
        return list(lines.values()) + [txt_h2o, txt_co2, txt_tau]

    def update(i):
        smoothed = per_frame_curves[i]
        T_sub = per_frame_T_sub[i]
        for key, ln in lines.items():
            ln.set_data(LAT_GRID, smoothed[key])
        txt_h2o.set_text(f"$T_\\mathrm{{sub}}$ = {T_sub['H2O']:.0f} K")
        txt_co2.set_text(f"$T_\\mathrm{{sub}}$ = {T_sub['CO2']:.0f} K")
        txt_tau.set_text(f"τ = {format_timescale(timescales_yr[i])}")
        return list(lines.values()) + [txt_h2o, txt_co2, txt_tau]

    anim = animation.FuncAnimation(
        fig, update, frames=args.n_frames, init_func=init,
        interval=1000 / args.fps, blit=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    mp4_path = args.out_dir / "cold_trap_fraction_animation.mp4"
    print(f"Writing {mp4_path}…")
    t0 = time.time()
    writer_mp4 = animation.FFMpegWriter(
        fps=args.fps, codec="libx264",
        extra_args=["-pix_fmt", "yuv420p", "-crf", "22"])
    anim.save(str(mp4_path), writer=writer_mp4, dpi=180)
    print(f"  wrote {mp4_path} in {time.time()-t0:.1f} s")

    if args.gif:
        gif_path = args.out_dir / "cold_trap_fraction_animation.gif"
        print(f"Writing {gif_path}…")
        t0 = time.time()
        writer_gif = animation.PillowWriter(fps=args.fps)
        anim.save(str(gif_path), writer=writer_gif, dpi=110)
        print(f"  wrote {gif_path} in {time.time()-t0:.1f} s")

    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
