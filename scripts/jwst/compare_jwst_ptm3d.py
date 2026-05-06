#!/usr/bin/env python3
"""
End-to-end comparison: PTM3D predictions vs Cartwright et al. (2025) JWST maps.

Reads:
  - data/jwst/cartwright2025/phi_eq_albedo_map.npz   (PTM3D φ_C, T_max, etc.)
  - data/jwst/cartwright2025/band_center_2um.npz     (PTM3D 2 μm band center)
  - data/jwst/cartwright2025/co2_coldtrap.npz        (PTM3D CO2 stability)
  - data/jwst/cartwright2025/observations.npz        (digitized JWST data)

Computes:
  H1: Spearman rank correlation between observed Fresnel peak and predicted φ_C
  H2: RMS residual + Pearson r for predicted vs observed 2.0 μm band center
  H3: Pattern-mismatch metric for CO₂ — overlap of cold-trap mask with observed
      high-CO₂ regions

Produces a single summary text file + a 4-panel figure (PTM3D φ_C, observed
Fresnel, predicted vs observed 2 μm band centers, CO2 thermal vs observed).

Usage:
  python compare_jwst_ptm3d.py \
      --phi-npz   data/jwst/cartwright2025/phi_eq_albedo_map.npz \
      --bc-npz    data/jwst/cartwright2025/band_center_2um.npz \
      --co2-npz   data/jwst/cartwright2025/co2_coldtrap.npz \
      --obs-npz   data/jwst/cartwright2025/observations.npz \
      --out-dir   docs/figures/jwst \
      --summary   data/jwst/cartwright2025/comparison_summary.txt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import equirect  # noqa: E402


def latlon_to_xyz(lat, lon):
    rad_lat = np.deg2rad(lat); rad_lon = np.deg2rad(lon)
    return np.stack([
        np.cos(rad_lat) * np.cos(rad_lon),
        np.cos(rad_lat) * np.sin(rad_lon),
        np.sin(rad_lat),
    ], axis=-1)


def regrid_face_values_to_obs(face_lat, face_lon, face_vals,
                               obs_lat, obs_lon):
    """Nearest-face value at each observation lat/lon. Uses xyz kd-tree."""
    xyz_face = latlon_to_xyz(face_lat, face_lon)
    xyz_obs = latlon_to_xyz(obs_lat, obs_lon % 360.0)
    tree = cKDTree(xyz_face)
    _, idx = tree.query(xyz_obs, k=1)
    return face_vals[idx]


def load_obs(npz_path):
    z = np.load(npz_path, allow_pickle=True)
    out = {}
    for k in z.files:
        if "__" in k:
            band, sub = k.split("__", 1)
            out.setdefault(band, {})[sub] = z[k]
        else:
            out[k] = z[k]
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phi-npz", default=Path("data/jwst/cartwright2025/phi_eq_albedo_map.npz"),
                   type=Path)
    p.add_argument("--bc-npz",  default=Path("data/jwst/cartwright2025/band_center_2um.npz"),
                   type=Path)
    p.add_argument("--co2-npz", default=Path("data/jwst/cartwright2025/co2_coldtrap.npz"),
                   type=Path)
    p.add_argument("--obs-npz", default=Path("data/jwst/cartwright2025/observations.npz"),
                   type=Path)
    p.add_argument("--out-dir", default=Path("docs/figures/jwst"), type=Path)
    p.add_argument("--summary", default=Path("data/jwst/cartwright2025/comparison_summary.txt"),
                   type=Path)
    args = p.parse_args()

    phi_data = np.load(args.phi_npz)
    bc_data = np.load(args.bc_npz)
    co2_data = np.load(args.co2_npz)
    obs = load_obs(args.obs_npz)
    is_placeholder = bool(obs.get("_placeholder", False))
    if is_placeholder:
        print("[compare] *** Observations are SYNTHETIC PLACEHOLDER ***")

    face_lat = phi_data["lat"]; face_lon = phi_data["lon"]
    phi_C = phi_data["phi_C_integrated"] if "phi_C_integrated" in phi_data.files \
            else phi_data["phi_eq_tmax"]
    pred_lambda_2um = bc_data["lambda_2um"]
    co2_stable = co2_data["co2_stable_10yr"].astype(bool)
    co2_log_loss = co2_data["co2_log10_loss_kg_m2_per_yr"]

    summary_lines = []
    def L(s):
        print(s)
        summary_lines.append(s)

    L("=" * 78)
    L(" PTM3D vs JWST (Cartwright 2025) Comparison Summary")
    L("=" * 78)
    if is_placeholder:
        L("WARNING: Observations are SYNTHETIC PLACEHOLDERS, not real Cartwright data.")
        L("         To use real data, populate digitized CSVs in")
        L("         data/jwst/cartwright2025/digitized/ and re-run ingest_cartwright2025.py.")
    L(f"PTM3D: {face_lat.size} faces, integrated phi_C n_orbits = "
      f"{int(phi_data['n_orbits_integrated']) if 'n_orbits_integrated' in phi_data.files else 'NA'}")
    L("")

    # ---- H1: Fresnel peak (crystalline) vs predicted phi_C ----
    fres = obs.get("fresnel_31um")
    if fres is not None:
        pred_phi = regrid_face_values_to_obs(face_lat, face_lon, phi_C,
                                             fres["lat"], fres["lon"])
        # Also pull T_max at each obs point for the talk-ready table
        pred_Tmax = regrid_face_values_to_obs(face_lat, face_lon,
                                              phi_data["T_max"],
                                              fres["lat"], fres["lon"])
        L("H1: Crystallinity (3.1 μm Fresnel peak) vs PTM3D phi_C")
        if fres["value"].size >= 2:
            rho_s, _ = spearmanr(pred_phi, fres["value"])
            rho_p, _ = pearsonr(pred_phi, fres["value"]) \
                       if fres["value"].size >= 3 else (np.nan, np.nan)
            L(f"   N_obs = {fres['value'].size}    "
              f"Spearman rho = {rho_s:+.3f}    "
              f"Pearson r = {('%.3f' % rho_p) if np.isfinite(rho_p) else 'n/a'}")
        if fres["value"].size <= 8:
            L("   per-region (lat, lon → predicted T_max, phi_C vs observed Fresnel %):")
            for i in range(fres["value"].size):
                L(f"     ({fres['lat'][i]:+5.1f}°, {fres['lon'][i]:5.1f}°E): "
                  f"T_max={pred_Tmax[i]:5.1f}K  phi_C={pred_phi[i]:.3f}  "
                  f"obs Fresnel={fres['value'][i]:5.2f}%")
        L("   ↑ Positive rho means warm regions in the model coincide with "
          "crystalline regions in the observations.")
        L("   ↓ Negative rho indicates pure thermal recrystallization is "
          "FALSIFIED at these regions — observed crystallinity needs a non-thermal driver.")
        L("")

    # ---- H2: 2 um band center thermometer ----
    bc_obs = obs.get("band_center_2um")
    if bc_obs is not None:
        pred = regrid_face_values_to_obs(face_lat, face_lon, pred_lambda_2um,
                                         bc_obs["lat"], bc_obs["lon"])
        # Restrict to physically valid range (anchors are 130-150 K)
        valid = np.isfinite(pred) & np.isfinite(bc_obs["value"]) \
                & (pred > 1.980) & (pred < 1.998)
        if valid.sum() >= 2:
            resid = pred[valid] - bc_obs["value"][valid]
            rms = float(np.sqrt(np.mean(resid ** 2)))
            mean_bias = float(np.mean(resid))
            r_str = "n/a"
            if valid.sum() >= 3:
                r, _ = pearsonr(pred[valid], bc_obs["value"][valid])
                r_str = f"{r:+.3f}"
            L("H2: 2.0 μm band center predicted vs observed")
            L(f"   N_valid = {valid.sum()}/{bc_obs['value'].size}")
            L(f"   RMS residual = {rms*1e3:.3f} nm    "
              f"mean bias = {mean_bias*1e3:+.3f} nm    Pearson r = {r_str}")
            # Per-spaxel residuals (helpful when N is small)
            if valid.sum() <= 8:
                L("   per-spaxel residuals (predicted - observed):")
                for i in np.where(valid)[0]:
                    L(f"     lat={bc_obs['lat'][i]:5.1f} lon={bc_obs['lon'][i]:5.1f}: "
                      f"pred={pred[i]:.4f}μm  obs={bc_obs['value'][i]:.4f}μm  "
                      f"Δ={1e3*(pred[i]-bc_obs['value'][i]):+5.1f}nm")
            L("")
        else:
            L("H2: Too few valid points after filtering.")

    # ---- H3: CO2 cold-trap pattern ----
    co2_obs = obs.get("co2_band_depth")
    if co2_obs is not None:
        # Pull predicted log-loss at observed lat/lon
        pred_log = regrid_face_values_to_obs(face_lat, face_lon, co2_log_loss,
                                             co2_obs["lat"], co2_obs["lon"])
        pred_stable = regrid_face_values_to_obs(face_lat, face_lon,
                                                co2_stable.astype(np.float64),
                                                co2_obs["lat"], co2_obs["lon"]) > 0.5
        # Strong CO2 obs threshold: top 25% of band depths
        thresh = float(np.quantile(co2_obs["value"], 0.75))
        strong_co2 = co2_obs["value"] >= thresh
        # Confusion: observed strong vs predicted-stable
        n11 = int(np.sum(strong_co2 & pred_stable))
        n10 = int(np.sum(strong_co2 & ~pred_stable))
        n01 = int(np.sum(~strong_co2 & pred_stable))
        n00 = int(np.sum(~strong_co2 & ~pred_stable))
        L("H3: CO₂ thermal cold-trap vs observed band depth")
        L(f"   Observed-strong threshold (top 25%): band depth >= {thresh:.2f}")
        L(f"   Confusion (rows = thermal-stable, cols = obs-strong):")
        L(f"     stable & strong   : {n11:5d}      stable & weak   : {n01:5d}")
        L(f"     unstable & strong : {n10:5d}      unstable & weak : {n00:5d}")
        if n11 + n10 > 0:
            recall = n11 / (n11 + n10)
            L(f"   Recall (fraction of strong-CO₂ regions that are thermally stable): "
              f"{recall:.1%}")
            L(f"   ↳ Recall << 1 means observed CO₂ is in regions that are NOT "
              f"thermally stable, demanding a non-thermal source.")
        # Spearman: anti-correlation expected since cold-trap and warm chaos are opposite
        rho, _ = spearmanr(pred_log, co2_obs["value"])
        L(f"   Spearman ρ(log-sublimation-loss, observed CO₂ depth) = {rho:+.3f}")
        L("   ↳ Positive ρ would mean observed CO₂ tracks WARM regions (endogenic);")
        L("     negative ρ would mean it tracks COLD regions (cold-trap).")
        L("")

    L("=" * 78)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text("\n".join(summary_lines) + "\n")
    print(f"[compare] Wrote {args.summary}")

    # ---- 4-panel figure ----
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 8.6),
                             gridspec_kw={"hspace": 0.34, "wspace": 0.20})
    extent = [0, 360, -90, 90]

    # Panel A: predicted phi_C
    lon_g, lat_g, P = equirect.project_to_grid(face_lat, face_lon, phi_C, 2)
    ax = axes[0, 0]
    im = ax.imshow(P, origin="lower", extent=extent, cmap="cividis",
                   vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02,
                 label=r"$\varphi_C$ (predicted)")
    ax.set_title("(A) PTM3D predicted crystalline fraction")
    equirect.annotate_landmarks(ax, kinds=("chaos",), color="white", fontsize=7)

    # Panel B: observed Fresnel (or placeholder)
    ax = axes[0, 1]
    if fres is not None:
        # Auto-scale: the units differ between placeholder ([0,1]) and Cartwright
        # extracted spaxels (units of % spectral contrast), so let the data set
        # the range with a small percentile clip to suppress outliers.
        v = fres["value"]
        vmin = float(np.nanpercentile(v, 1))
        vmax = float(np.nanpercentile(v, 99))
        sc = ax.scatter(fres["lon"], fres["lat"], c=v, cmap="cividis",
                        s=22, edgecolors="k", linewidths=0.3, vmin=vmin, vmax=vmax)
        plt.colorbar(sc, ax=ax, shrink=0.8, pad=0.02,
                     label="Fresnel peak strength" + (" [PLACEHOLDER]" if is_placeholder else " (%)"))
    ax.set_xlim(0, 360); ax.set_ylim(-90, 90)
    ax.set_title("(B) Observed 3.1 μm Fresnel peak"
                 + (" — PLACEHOLDER" if is_placeholder else " (Cartwright 2025)"))
    equirect.annotate_landmarks(ax, kinds=("chaos",), color="black", fontsize=7)

    # Panel C: 2 μm band center scatter (predicted vs observed)
    ax = axes[1, 0]
    if bc_obs is not None:
        pred = regrid_face_values_to_obs(face_lat, face_lon, pred_lambda_2um,
                                         bc_obs["lat"], bc_obs["lon"])
        valid = np.isfinite(pred) & np.isfinite(bc_obs["value"]) \
                & (pred > 1.980) & (pred < 1.998)
        ax.scatter(pred[valid] * 1e3, bc_obs["value"][valid] * 1e3, s=20,
                   alpha=0.7, edgecolors="k", linewidths=0.3)
        lo, hi = 1.980 * 1e3, 1.998 * 1e3
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5)
        ax.set_xlabel("Predicted λ₂.₀ (nm)")
        ax.set_ylabel("Observed λ₂.₀ (nm)")
        ax.set_title("(C) 2.0 μm band-center thermometer")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.grid(alpha=0.3)
        if is_placeholder:
            ax.text(0.05, 0.95, "PLACEHOLDER", transform=ax.transAxes,
                    color="red", fontweight="bold", va="top")

    # Panel D: CO2 stability + observed depth overlay
    ax = axes[1, 1]
    _, _, S = equirect.project_to_grid(face_lat, face_lon,
                                       co2_stable.astype(np.float64), 2)
    im = ax.imshow(S, origin="lower", extent=extent, cmap="Blues",
                   vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    if co2_obs is not None:
        sc = ax.scatter(co2_obs["lon"], co2_obs["lat"], c=co2_obs["value"],
                        s=24, edgecolors="white", linewidths=0.4, cmap="autumn",
                        vmin=co2_obs["value"].min(), vmax=co2_obs["value"].max())
        plt.colorbar(sc, ax=ax, shrink=0.8, pad=0.02,
                     label="Observed CO₂ band depth" + (" [PLACEHOLDER]" if is_placeholder else ""))
    ax.set_title("(D) CO₂: thermal cold-trap mask vs observed band depth")
    ax.set_xlim(0, 360); ax.set_ylim(-90, 90)
    equirect.annotate_landmarks(ax, kinds=("chaos",), color="white", fontsize=7)

    for ax in [axes[0, 0], axes[0, 1], axes[1, 1]]:
        ax.set_xlabel("East longitude (deg)")
        ax.set_ylabel("Latitude (deg)")

    # Source-aware suffix: distinguish C++-runtime vs Python-integrator outputs
    src_tag = ""
    if "source" in phi_data.files and str(phi_data["source"]) == "cpp_runtime":
        src_tag = "_cpp"
        suptitle = "PTM3D (C++ run-time) × JWST/NIRSpec (Cartwright 2025) — comparison"
    else:
        suptitle = "PTM3D (Python integrator) × JWST/NIRSpec (Cartwright 2025) — comparison"
    if is_placeholder:
        suptitle += "  [SYNTHETIC PLACEHOLDER OBSERVATIONS]"
    fig.suptitle(suptitle, fontsize=12, y=0.995)
    out_png = args.out_dir / f"comparison_4panel{src_tag}.png"
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    print(f"[compare] Wrote {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
