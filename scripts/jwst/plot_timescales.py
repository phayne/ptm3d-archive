#!/usr/bin/env python3
"""Three timescales that govern Europa's surface ice phase state.

  τ_cryst   = 1 / k_cryst(T_max)            [Mitchell 2017 Avrami]
  τ_amorph  = E_amorph / Ḋ                   [Famá 2010, Nordheim 2022]
  τ_accum   = (h_optical · ρ_ice) / Φ_dep    [time to deposit an optically
                                             thick ice layer of thickness
                                             h_optical = 10 μm — the
                                             3.1 μm Fresnel-peak probe depth]

The shortest timescale wins:
  τ_cryst < min(τ_amorph, τ_accum)  → crystalline ice exposed
  τ_amorph < min(τ_cryst, τ_accum)  → radiation-amorphized
  τ_accum  < min(τ_cryst, τ_amorph) → freshly-deposited amorphous frost
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import mass_balance as mb_mod  # noqa: E402
import crystallization as cr  # noqa: E402
import gardening as gard  # noqa: E402
import equirect  # noqa: E402

SECONDS_PER_YEAR = 365.25 * 86400.0
RHO_ICE = 917.0      # kg/m³
H_OPTICAL = 10.0e-6  # m  — 3.1 μm Fresnel-peak probe depth (Cartwright 2025)


def compute_timescales(mb: dict,
                       gard_depth_m: float = gard.Z_OPTICAL_M,
                       gard_lambda: float = gard.LAMBDA_INTERMED,
                       gard_case: str = "B") -> dict:
    """Return per-face τ_cryst, τ_amorph, τ_accum, τ_gard [years].

    Gardening is essentially globally uniform in the Costello et al. (2021)
    model (impact flux is isotropic over Europa), so τ_gard is a scalar
    broadcast onto every face. The defaults match the τ_accum reference
    depth (10 µm = 3.1 µm Fresnel-peak optical depth) so the four
    timescales race head-to-head on the same depth scale.
    """
    porosity = cr.porosity_from_latitude(mb["lat"])

    k_cr = cr.k_cryst(mb["T_max"], porosity)              # s^-1
    k_am = cr.k_amorph(mb["dose_rate"])                    # s^-1
    dep_rate_kg_m2_s = mb["deposition_rate"] / SECONDS_PER_YEAR

    # Optical-thickness mass [kg/m²]
    M_opt = H_OPTICAL * RHO_ICE   # 9.17e-3 kg/m²

    big = 1e15  # cap years for plotting
    tau_cryst = np.where(k_cr > 1e-30, 1.0 / (k_cr * SECONDS_PER_YEAR), big)
    tau_amorph = np.where(k_am > 1e-30, 1.0 / (k_am * SECONDS_PER_YEAR), big)
    tau_accum = np.where(dep_rate_kg_m2_s > 1e-30,
                         M_opt / (dep_rate_kg_m2_s * SECONDS_PER_YEAR), big)

    # Costello et al. (2021) gardening — scalar, broadcast to all faces.
    tau_gard_scalar = float(gard.gardening_timescale(
        gard_depth_m, lam=gard_lambda, case=gard_case))
    tau_gard = np.full_like(tau_cryst, tau_gard_scalar)

    return dict(tau_cryst=tau_cryst, tau_amorph=tau_amorph,
                tau_accum=tau_accum, tau_gard=tau_gard,
                tau_gard_scalar=tau_gard_scalar,
                gard_depth_m=gard_depth_m, gard_lambda=gard_lambda,
                gard_case=gard_case)


def regime_label(t: dict) -> np.ndarray:
    """Per-face categorical: 0 crystalline, 1 amorph-radiation,
    2 amorph-frost, 3 gardening-overturn dominant."""
    nf = t["tau_cryst"].size
    taus = np.stack([t["tau_cryst"], t["tau_amorph"],
                     t["tau_accum"], t["tau_gard"]], axis=1)
    return np.argmin(taus, axis=1).astype(np.int8)


def latitude_profile(ax, lat, vals, n_bins=36, **kw):
    bins = np.linspace(-87.5, 87.5, n_bins)
    centers = 0.5 * (bins[:-1] + bins[1:])
    geom_mean = np.full(centers.size, np.nan)
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        m = (lat >= lo) & (lat < hi) & np.isfinite(vals) & (vals > 0)
        if m.any():
            w = np.cos(np.deg2rad(lat[m]))
            geom_mean[i] = float(np.exp(np.average(np.log(vals[m]), weights=w)))
    ax.plot(centers, geom_mean, **kw)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path,
                   help="Output base path (will write _profile.png and _map.png)")
    p.add_argument("--gard-depth-um", type=float, default=10.0,
                   help="Reference depth for τ_gard curve [µm]. "
                        "Default 10 µm (matches τ_accum's 3.1 µm Fresnel-peak depth).")
    p.add_argument("--gard-case", type=str, default="B", choices=["A", "B"],
                   help="Costello (2021) crater-flux scenario. Default B.")
    p.add_argument("--gard-lambda", type=float, default=gard.LAMBDA_INTERMED,
                   help="Cumulative-Poisson probability bin. "
                        "0.02=2%%, 0.693=50%% (default), 4.605=99%%.")
    args = p.parse_args()

    mb = mb_mod.load_mass_balance(args.run)
    yr = mb["total_years"]
    t = compute_timescales(mb,
                            gard_depth_m=args.gard_depth_um * 1e-6,
                            gard_lambda=args.gard_lambda,
                            gard_case=args.gard_case)

    # ------------------------------------------------------------------
    # Figure 1: latitude profile (one clean panel)
    # ------------------------------------------------------------------
    plt.rcParams.update({"font.size": 16, "axes.labelsize": 18,
                         "axes.titlesize": 20, "legend.fontsize": 16,
                         "xtick.labelsize": 15, "ytick.labelsize": 15})
    fig, ax = plt.subplots(figsize=(9.0, 6.2))
    latitude_profile(ax, mb["lat"], t["tau_cryst"],
                     color="#c0392b", lw=3.2, ls="-",
                     label="Crystallization")
    latitude_profile(ax, mb["lat"], t["tau_amorph"],
                     color="#8e44ad", lw=3.2, ls="--",
                     label="Amorphization (radiation)")
    latitude_profile(ax, mb["lat"], t["tau_accum"],
                     color="#1f6db8", lw=3.2, ls=":",
                     label="Frost burial (10 μm)")

    # Gardening overturn (Costello et al. 2021). Latitude-independent in the
    # paper's model (uniform global flux), so a horizontal line. Shaded band
    # spans z = 1 mm to z = 30 cm at the same probability/case to convey
    # that gardening's timescale depends strongly on the reference depth.
    tau_g = t["tau_gard_scalar"]
    z_um = t["gard_depth_m"] * 1e6
    if z_um >= 1000:
        z_lbl = f"{z_um/1000:g} mm"
    else:
        z_lbl = f"{z_um:g} µm"
    ax.axhline(tau_g, color="#2a9d3f", lw=3.2, ls="-",
               label=f"Gardening overturn ({z_lbl})", zorder=3)
    tau_band_lo = float(gard.gardening_timescale(
        1e-3, lam=t["gard_lambda"], case=t["gard_case"]))   # 1 mm
    tau_band_hi = float(gard.gardening_timescale(
        gard.Z_RESERVOIR_M, lam=t["gard_lambda"], case=t["gard_case"]))  # 30 cm
    ax.axhspan(tau_band_lo, tau_band_hi,
               color="#2a9d3f", alpha=0.10, zorder=1,
               label="Gardening, 1 mm–30 cm")

    # Reference horizontal lines (just three: day, year, Myr)
    for hr_yr, lbl in [(1.0/365.25, "1 day"),
                       (1.0,        "1 year"),
                       (1.0e6,      "1 Myr")]:
        ax.axhline(hr_yr, color="gray", lw=0.6, alpha=0.55, ls=(0, (1, 3)))
        ax.text(-89, hr_yr, lbl, color="gray", fontsize=12,
                ha="left", va="bottom")

    ax.set_yscale("log")
    ax.set_xlim(-90, 90); ax.set_xticks(np.arange(-90, 91, 30))
    ax.set_ylim(1e-4, 1e15)
    ax.set_xlabel("Latitude (°)")
    ax.set_ylabel("Timescale (years)")
    ax.set_title("Surface ice phase timescales on Europa")
    ax.grid(alpha=0.18, which="major")

    # Place the legend outside the plot to keep curves uncovered.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16),
              ncol=3, frameon=False, handlelength=3.0, fontsize=13)

    fig.tight_layout()

    out_profile = args.out.with_name(args.out.stem + "_profile" + args.out.suffix)
    out_profile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_profile, dpi=180, bbox_inches="tight")
    fig.savefig(out_profile.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Wrote {out_profile}")

    # ------------------------------------------------------------------
    # Figure 2: regime map (continuous, with regime-boundary contour)
    # ------------------------------------------------------------------
    log_ratio = np.log10(np.maximum(t["tau_cryst"], 1e-30)
                         / np.maximum(t["tau_amorph"], 1e-30))
    log_ratio = np.clip(log_ratio, -10, 10)
    _, _, M = equirect.project_to_grid(mb["lat"], mb["lon"], log_ratio, 4)

    # Spatial smoothing to reduce per-spaxel noise in the map
    try:
        from scipy.ndimage import gaussian_filter
        M = gaussian_filter(M, sigma=1.5)
    except ImportError:
        pass

    fig2, ax = plt.subplots(figsize=(11.5, 5.8))
    extent = [0, 360, -90, 90]
    im = ax.imshow(M, origin="lower", extent=extent,
                   cmap="RdBu_r", vmin=-6, vmax=6,
                   aspect="auto", interpolation="bilinear")
    cbar = plt.colorbar(im, ax=ax, shrink=0.92, pad=0.02,
                        ticks=[-6, -3, 0, 3, 6])
    cbar.ax.set_yticklabels([r"$10^{-6}$" + "\nCrystalline", r"$10^{-3}$",
                             "1\n(boundary)", r"$10^{3}$",
                             r"$10^{6}$" + "\nAmorphous"], fontsize=13)
    cbar.ax.set_ylabel("Crystallization / amorphization timescale ratio",
                       fontsize=15)

    # Regime boundary contour
    lons_grid = np.linspace(0, 360, M.shape[1])
    lats_grid = np.linspace(-90, 90, M.shape[0])
    ax.contour(lons_grid, lats_grid, M, levels=[0], colors="black",
               linewidths=1.6)

    # Just two annotations: Tara, Powys. White-on-map with a thin black
    # stroke for legibility on both blue (crystalline) and red (amorphous)
    # cells.
    import matplotlib.patheffects as pe
    halo = [pe.withStroke(linewidth=2.0, foreground="black")]
    for name, plat, plon in [("Tara",  -10.0, 285.0),
                             ("Powys",   0.0, 215.0)]:
        sc = ax.scatter([plon], [plat], s=120, marker="s",
                        facecolors="none", edgecolors="white", linewidths=1.8,
                        zorder=5)
        sc.set_path_effects(halo)
        ax.annotate(name, (plon, plat), xytext=(10, 10),
                    textcoords="offset points",
                    color="white", fontsize=15, fontweight="bold",
                    path_effects=halo, zorder=6)

    ax.set_xlabel("Longitude (°E)", fontsize=18)
    ax.set_ylabel("Latitude (°)", fontsize=18)
    ax.set_title("Surface phase regime", fontsize=20, pad=12)
    ax.set_xticks(np.arange(0, 361, 60))
    ax.set_yticks(np.arange(-90, 91, 30))
    ax.tick_params(labelsize=15)
    fig2.tight_layout()
    out_map = args.out.with_name(args.out.stem + "_map" + args.out.suffix)
    fig2.savefig(out_map, dpi=200, bbox_inches="tight")
    fig2.savefig(out_map.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Wrote {out_map}")

    # ------------------------------------------------------------------
    # Summary numbers
    # ------------------------------------------------------------------
    regime = regime_label(t)
    w = np.cos(np.deg2rad(mb["lat"]))
    print(f"\nGardening reference: depth = {t['gard_depth_m']*1e6:g} µm, "
          f"λ = {t['gard_lambda']:.3f}, case {t['gard_case']}, "
          f"τ_gard = {t['tau_gard_scalar']:.3g} yr")
    print("\nGlobal area-weighted regime fractions:")
    for i, name in enumerate(["Crystalline", "Amorph-radiation",
                              "Amorph-frost", "Gardening-overturn"]):
        f = float(np.average((regime == i).astype(np.float64), weights=w))
        print(f"  {name:20s}: {f:.1%}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
