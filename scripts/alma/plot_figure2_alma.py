#!/usr/bin/env python3
"""
Generate Figure 2 for the Europa paper: PTM3D vs. ALMA comparison.

Two-panel figure:
  (a) Predicted vs. observed Tb scatter plot with 1:1 line
  (b) Residual vs. sub-observer longitude, colored by frequency

Publication quality for JGR: Planets.

Usage:
  python plot_figure2_alma.py results.json          # From saved comparison
  python plot_figure2_alma.py --demo                # Demo with synthetic data
  python plot_figure2_alma.py --demo --save fig2.pdf
"""

import argparse
import json
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def setup_jgr_matplotlib():
    """Configure matplotlib for JGR: Planets style."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
    })
    return plt


def band_color(band):
    """Return color for an ALMA band."""
    colors = {3: "#2166ac", 6: "#4dac26", 7: "#d73027"}
    return colors.get(band, "gray")


def band_label(band):
    """Return label for an ALMA band."""
    labels = {3: "Band 3 (3.05 mm)", 6: "Band 6 (1.25 mm)",
              7: "Band 7 (0.88 mm)"}
    return labels.get(band, f"Band {band}")


def reference_marker(ref):
    """Return marker style for a reference."""
    if "Trumbo" in ref:
        return "o"
    return "s"


def plot_figure2(results, stats, output_path=None, draft=True):
    """Create the two-panel Figure 2.

    Parameters
    ----------
    results : list of dict
        Comparison results with tb_predicted, tb_published, etc.
    stats : dict
        Statistics from compute_statistics().
    output_path : str or Path, optional
        Output file path. If None, shows on screen.
    draft : bool
        If True, use 150 dpi instead of 300.
    """
    plt = setup_jgr_matplotlib()
    if draft:
        plt.rcParams["figure.dpi"] = 150
        plt.rcParams["savefig.dpi"] = 150

    # Full-width figure: ~190 mm wide for JGR
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.2))

    obs_arr = np.array([r["tb_published"] for r in results])
    pred_arr = np.array([r["tb_predicted"] for r in results])
    err_arr = np.array([r["tb_error"] for r in results])
    bands = [r["band"] for r in results]
    lons = [r["sub_obs_lon"] for r in results]
    refs = [r["reference"] for r in results]

    # ── Panel (a): Predicted vs. Observed ──
    # 1:1 line
    tb_min = min(obs_arr.min(), pred_arr.min()) - 5
    tb_max = max(obs_arr.max(), pred_arr.max()) + 5
    ax1.plot([tb_min, tb_max], [tb_min, tb_max], "k--", lw=0.8, alpha=0.5,
             zorder=0)

    # Plot each point
    plotted_bands = set()
    plotted_refs = set()
    for i, r in enumerate(results):
        color = band_color(r["band"])
        marker = reference_marker(r["reference"])
        # Band legend
        label_band = band_label(r["band"]) if r["band"] not in plotted_bands else None
        plotted_bands.add(r["band"])

        ax1.errorbar(r["tb_published"], r["tb_predicted"],
                     xerr=r["tb_error"],
                     fmt=marker, color=color, markersize=6,
                     capsize=2, capthick=0.8, elinewidth=0.8,
                     label=label_band, zorder=5)

    ax1.set_xlabel("Observed $T_b$ (K)")
    ax1.set_ylabel("Predicted $T_b$ (K)")
    ax1.set_xlim(tb_min, tb_max)
    ax1.set_ylim(tb_min, tb_max)
    ax1.set_aspect("equal")
    ax1.legend(loc="upper left", frameon=True, fancybox=False,
               edgecolor="0.7", framealpha=0.9)

    # Annotation with statistics
    text = (f"$\\chi^2_r$ = {stats['chi2_r']:.2f}\n"
            f"RMS = {stats['rms']:.1f} K\n"
            f"bias = {stats['bias']:+.1f} K")
    ax1.text(0.97, 0.03, text, transform=ax1.transAxes,
             ha="right", va="bottom", fontsize=7.5,
             bbox=dict(boxstyle="round,pad=0.3", fc="white",
                       ec="0.7", alpha=0.9))
    ax1.text(0.02, 0.98, "(a)", transform=ax1.transAxes,
             ha="left", va="top", fontsize=11, fontweight="bold")

    # ── Panel (b): Residual vs. longitude ──
    residuals = pred_arr - obs_arr

    # Zero line
    ax2.axhline(0, color="k", ls="--", lw=0.8, alpha=0.5, zorder=0)

    # Shaded uncertainty band (mean published error)
    mean_err = np.mean(err_arr)
    ax2.axhspan(-mean_err, mean_err, color="0.9", alpha=0.5, zorder=0)

    for i, r in enumerate(results):
        color = band_color(r["band"])
        marker = reference_marker(r["reference"])
        resid = r["tb_predicted"] - r["tb_published"]
        ax2.plot(r["sub_obs_lon"], resid, marker, color=color,
                 markersize=6, zorder=5)

    ax2.set_xlabel("Sub-observer longitude ($\\degree$W)")
    ax2.set_ylabel("$\\Delta T_b$ (predicted $-$ observed) (K)")
    ax2.set_xlim(-10, 370)
    ax2.set_xticks([0, 90, 180, 270, 360])
    ax2.set_ylim(-10, 10)

    # Hemisphere labels (placed at top of fixed y-range)
    for lon_c, label in [(0, "anti-Jovian"), (90, "leading"),
                          (180, "sub-Jovian"), (270, "trailing")]:
        ax2.text(lon_c, 10, label,
                 ha="center", va="top", fontsize=6.5, color="0.4",
                 style="italic")

    ax2.text(0.02, 0.98, "(b)", transform=ax2.transAxes,
             ha="left", va="top", fontsize=11, fontweight="bold")

    fig.tight_layout(w_pad=2.5)

    if output_path:
        fig.savefig(str(output_path))
        print(f"  Saved: {output_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_tb_vs_wavelength(results, output_path=None, draft=True):
    """Supplementary figure: Tb vs. wavelength for leading and trailing.

    Shows the depth-probing capability of multi-frequency ALMA data.
    """
    plt = setup_jgr_matplotlib()
    if draft:
        plt.rcParams["figure.dpi"] = 150

    fig, ax = plt.subplots(figsize=(4, 3.2))

    # Wavelength mapping
    freq_to_wl = {97.5: 3.05, 233.0: 1.25, 343.5: 0.88}

    for hem, marker, color in [("leading", "o", "#2166ac"),
                                 ("trailing", "s", "#d73027")]:
        wl_obs, tb_obs, tb_err = [], [], []
        wl_pred, tb_pred_vals = [], []

        for r in results:
            if r["hemisphere"] != hem:
                continue
            wl = freq_to_wl.get(r["freq_ghz"], r["freq_ghz"])
            wl_obs.append(wl)
            tb_obs.append(r["tb_published"])
            tb_err.append(r["tb_error"])
            wl_pred.append(wl)
            tb_pred_vals.append(r["tb_predicted"])

        if wl_obs:
            ax.errorbar(wl_obs, tb_obs, yerr=tb_err, fmt=marker,
                        color=color, capsize=3, capthick=0.8,
                        label=f"{hem} (observed)", markersize=6,
                        zorder=5)
            ax.plot(wl_pred, tb_pred_vals, marker, color=color,
                    mfc="none", markersize=8, mew=1.5,
                    label=f"{hem} (PTM3D)", zorder=4)

    ax.set_xlabel("Wavelength (mm)")
    ax.set_ylabel("Disk-averaged $T_b$ (K)")
    ax.legend(fontsize=7, frameon=True, fancybox=False, edgecolor="0.7")
    ax.invert_xaxis()  # Short wavelength (shallow) on right
    ax.set_title("Brightness temperature vs. ALMA wavelength")

    fig.tight_layout()
    if output_path:
        fig.savefig(str(output_path))
        print(f"  Saved: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Figure 2: PTM3D vs. ALMA comparison"
    )
    parser.add_argument("input", nargs="?",
                        help="JSON file with comparison results")
    parser.add_argument("--demo", action="store_true",
                        help="Run demo with synthetic T(z) profiles")
    parser.add_argument("--save", default=None,
                        help="Output file path (default: data/alma/diagnostics/)")
    parser.add_argument("--porosity", type=float, default=0.5)
    parser.add_argument("--emissivity", type=float, default=0.75)
    parser.add_argument("--thermal-inertia", type=float, default=95.0)
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / "data" / "alma" / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.input:
        with open(args.input) as f:
            data = json.load(f)
        results = data["results"]
        stats = data["stats"]
    elif args.demo:
        from compare_published_tb import (get_published_tb,
                                           compute_predicted_tb_synthetic,
                                           compute_statistics)
        observations = get_published_tb()
        print(f"  Computing synthetic predictions for {len(observations)} obs...")
        results = []
        for obs in observations:
            tb_pred = compute_predicted_tb_synthetic(
                obs, porosity=args.porosity, emissivity=args.emissivity,
                thermal_inertia=args.thermal_inertia
            )
            results.append({
                "label": obs["label"],
                "band": obs["band"],
                "hemisphere": obs["hemisphere"],
                "sub_obs_lon": obs["sub_obs_lon"],
                "tb_predicted": tb_pred,
                "tb_published": obs["tb_published"],
                "tb_error": obs["tb_error"],
                "freq_ghz": obs["freq_ghz"],
                "reference": obs["reference"],
            })
        stats = compute_statistics(results)

        from compare_published_tb import print_comparison_table
        print_comparison_table(results, stats)
    else:
        parser.print_help()
        return

    # Generate figures
    save_path = args.save or str(out_dir / "figure2_alma_comparison.pdf")
    plot_figure2(results, stats, save_path)

    # Supplementary: Tb vs wavelength
    wl_path = str(out_dir / "figure_tb_vs_wavelength.pdf")
    plot_tb_vs_wavelength(results, wl_path)

    # Save results as JSON for reproducibility
    json_path = out_dir / "alma_comparison_results.json"
    with open(json_path, "w") as f:
        json.dump({"results": results, "stats": stats}, f, indent=2)
    print(f"  Saved results: {json_path}")


if __name__ == "__main__":
    main()
