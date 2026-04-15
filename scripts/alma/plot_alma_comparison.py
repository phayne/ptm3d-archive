#!/usr/bin/env python3
"""
Generate publication figures for ALMA vs. PTM3D comparison (Paper Figure 2).

Creates side-by-side panels showing:
  - ALMA observed Tb
  - PTM3D predicted Tb (beam-convolved)
  - Residual (observed - model)

Supports both individual epoch panels and multi-panel composite figures.

Usage:
  python plot_alma_comparison.py --comparison-dir data/alma/comparison/
  python plot_alma_comparison.py --single T1     # Single epoch figure
  python plot_alma_comparison.py --composite      # Multi-panel composite
"""

import argparse
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Import local modules
import sys
sys.path.insert(0, str(Path(__file__).parent))
from setup_ptm3d_runs import TRUMBO_OBS, THELEN_OBS, get_all_observations


def setup_matplotlib():
    """Configure matplotlib for publication-quality figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 10,
        "font.family": "serif",
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })
    return plt


def plot_disk_image(ax, data_2d, lat_2d, lon_2d, on_disk,
                    vmin=None, vmax=None, cmap="inferno",
                    title="", colorbar_label="Tb (K)"):
    """Plot a Europa disk image with proper projection.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on.
    data_2d : ndarray
        Data values on image grid.
    lat_2d, lon_2d : ndarray
        Latitude and longitude grids.
    on_disk : ndarray
        Boolean mask for on-disk pixels.
    vmin, vmax : float
        Color scale limits.
    cmap : str
        Colormap name.
    title : str
        Plot title.
    colorbar_label : str
        Label for colorbar.

    Returns
    -------
    matplotlib.image.AxesImage
        The image object.
    """
    # Mask off-disk pixels
    plot_data = np.where(on_disk, data_2d, np.nan)

    im = ax.imshow(plot_data, origin="lower", cmap=cmap,
                   vmin=vmin, vmax=vmax, interpolation="nearest")

    # Draw disk outline
    ny, nx = data_2d.shape
    theta = np.linspace(0, 2 * np.pi, 100)
    # Find disk center and radius from the on_disk mask
    ys, xs = np.where(on_disk)
    if len(xs) > 0:
        cx = (xs.min() + xs.max()) / 2
        cy = (ys.min() + ys.max()) / 2
        rx = (xs.max() - xs.min()) / 2
        ry = (ys.max() - ys.min()) / 2
        ax.plot(cx + rx * np.cos(theta), cy + ry * np.sin(theta),
                "w-", linewidth=0.8, alpha=0.5)

    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    return im


def plot_single_comparison(comp_data, obs_info, output_path=None):
    """Plot a single epoch comparison: observed | model | residual.

    Parameters
    ----------
    comp_data : dict
        Comparison data from compare_alma_ptm3d.py.
    obs_info : dict
        Observation metadata.
    output_path : str or Path, optional
        Path to save figure.
    """
    plt = setup_matplotlib()

    obs_tb = comp_data["obs_tb"]
    model_tb = comp_data["model_tb_regrid"]
    residual = comp_data["residual"]
    obs_lat = comp_data["obs_lat"]
    obs_lon = comp_data["obs_lon"]
    emang = comp_data.get("emission_angle", None)

    on_disk = np.isfinite(obs_lat) & np.isfinite(obs_tb)

    # Determine color scale
    valid_obs = obs_tb[on_disk & np.isfinite(obs_tb)]
    if len(valid_obs) == 0:
        print(f"  No valid data for {obs_info['label']}, skipping plot")
        return

    tb_min = np.nanpercentile(valid_obs, 2)
    tb_max = np.nanpercentile(valid_obs, 98)
    res_max = max(abs(np.nanpercentile(residual[on_disk], 5)),
                  abs(np.nanpercentile(residual[on_disk], 95)))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    label = obs_info["label"]
    band = obs_info["band"]
    freq = obs_info["freq_ghz"]
    hem = obs_info["hemisphere"]

    # Panel 1: ALMA observed
    im1 = plot_disk_image(axes[0], obs_tb, obs_lat, obs_lon, on_disk,
                          vmin=tb_min, vmax=tb_max, cmap="inferno",
                          title=f"ALMA Observed ({label})")
    plt.colorbar(im1, ax=axes[0], shrink=0.8, label="Tb (K)")

    # Panel 2: PTM3D predicted
    im2 = plot_disk_image(axes[1], model_tb, obs_lat, obs_lon, on_disk,
                          vmin=tb_min, vmax=tb_max, cmap="inferno",
                          title="PTM3D Predicted")
    plt.colorbar(im2, ax=axes[1], shrink=0.8, label="Tb (K)")

    # Panel 3: Residual
    im3 = plot_disk_image(axes[2], residual, obs_lat, obs_lon, on_disk,
                          vmin=-res_max, vmax=res_max, cmap="coolwarm",
                          title="Residual (Obs - Model)")
    plt.colorbar(im3, ax=axes[2], shrink=0.8, label="ΔTb (K)")

    # Add statistics text
    stats_text = (f"Band {band} ({freq:.0f} GHz), {hem}\n"
                  f"RMS = {comp_data.get('rms', np.nan):.1f} K, "
                  f"Bias = {comp_data.get('bias', np.nan):+.1f} K")
    fig.text(0.5, 0.02, stats_text, ha="center", fontsize=9, style="italic")

    fig.suptitle(f"{obs_info['reference']} — {obs_info['epoch'][:10]}",
                 fontsize=13, y=1.02)
    fig.tight_layout()

    if output_path:
        fig.savefig(str(output_path))
        print(f"  Saved: {output_path}")
    plt.close(fig)


def plot_trumbo_composite(comp_dir, output_path=None):
    """Plot 4-row composite for Trumbo et al. (2018) observations.

    Each row: observed | model | residual for one epoch.
    This is the core of Paper Figure 2.

    Parameters
    ----------
    comp_dir : str or Path
        Directory containing comparison NPZ files.
    output_path : str or Path, optional
        Path to save figure.
    """
    plt = setup_matplotlib()
    comp_dir = Path(comp_dir)

    n_obs = len(TRUMBO_OBS)
    fig, axes = plt.subplots(n_obs, 3, figsize=(14, 4 * n_obs))

    # Global color scale across all panels
    global_tb_min = np.inf
    global_tb_max = -np.inf

    # Load all data first to determine global color scale
    all_data = []
    for obs_info in TRUMBO_OBS:
        label = obs_info["label"]
        comp_file = comp_dir / f"comparison_{label}.npz"
        if not comp_file.exists():
            print(f"  Missing: {comp_file}")
            all_data.append(None)
            continue
        data = dict(np.load(str(comp_file), allow_pickle=True))
        all_data.append(data)
        on_disk = np.isfinite(data["obs_lat"]) & np.isfinite(data["obs_tb"])
        valid = data["obs_tb"][on_disk]
        if len(valid) > 0:
            global_tb_min = min(global_tb_min, np.nanpercentile(valid, 2))
            global_tb_max = max(global_tb_max, np.nanpercentile(valid, 98))

    if global_tb_min == np.inf:
        print("  No valid data found for composite plot")
        return

    global_res_max = 0
    for data in all_data:
        if data is None:
            continue
        on_disk = np.isfinite(data["obs_lat"]) & np.isfinite(data["obs_tb"])
        res_valid = data["residual"][on_disk]
        if len(res_valid) > 0:
            global_res_max = max(global_res_max,
                                 np.nanpercentile(np.abs(res_valid), 95))

    # Plot each row
    for i, (obs_info, data) in enumerate(zip(TRUMBO_OBS, all_data)):
        if data is None:
            for j in range(3):
                axes[i, j].text(0.5, 0.5, "No data", ha="center", va="center",
                                transform=axes[i, j].transAxes)
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])
            continue

        on_disk = np.isfinite(data["obs_lat"]) & np.isfinite(data["obs_tb"])
        label = obs_info["label"]
        lon = obs_info["sub_obs_lon"]

        # Row label
        axes[i, 0].set_ylabel(f"{label}\nlon={lon:.0f}°W\n{obs_info['hemisphere']}",
                               fontsize=10, rotation=0, labelpad=60, va="center")

        # Observed
        im1 = plot_disk_image(axes[i, 0], data["obs_tb"], data["obs_lat"],
                              data["obs_lon"], on_disk,
                              vmin=global_tb_min, vmax=global_tb_max,
                              cmap="inferno",
                              title="ALMA Observed" if i == 0 else "")

        # Model
        im2 = plot_disk_image(axes[i, 1], data["model_tb_regrid"], data["obs_lat"],
                              data["obs_lon"], on_disk,
                              vmin=global_tb_min, vmax=global_tb_max,
                              cmap="inferno",
                              title="PTM3D Predicted" if i == 0 else "")

        # Residual
        im3 = plot_disk_image(axes[i, 2], data["residual"], data["obs_lat"],
                              data["obs_lon"], on_disk,
                              vmin=-global_res_max, vmax=global_res_max,
                              cmap="coolwarm",
                              title="Residual" if i == 0 else "")

        # RMS annotation
        rms = float(data.get("rms", np.nan))
        bias = float(data.get("bias", np.nan))
        axes[i, 2].text(0.98, 0.02, f"RMS={rms:.1f}K\nbias={bias:+.1f}K",
                        ha="right", va="bottom", fontsize=8,
                        transform=axes[i, 2].transAxes,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

    # Colorbars
    fig.subplots_adjust(right=0.92)
    cbar_ax1 = fig.add_axes([0.93, 0.35, 0.015, 0.55])
    fig.colorbar(im1, cax=cbar_ax1, label="Tb (K)")
    cbar_ax2 = fig.add_axes([0.93, 0.05, 0.015, 0.25])
    fig.colorbar(im3, cax=cbar_ax2, label="ΔTb (K)")

    fig.suptitle("ALMA Band 6 (233 GHz) vs. PTM3D — Trumbo et al. (2018)",
                 fontsize=14, y=0.98)

    if output_path:
        fig.savefig(str(output_path))
        print(f"  Saved: {output_path}")
    plt.close(fig)


def plot_thelen_multiband(comp_dir, output_path=None):
    """Plot multi-frequency comparison for Thelen et al. (2024).

    Organized by hemisphere (leading/trailing) and band.

    Parameters
    ----------
    comp_dir : str or Path
        Directory containing comparison NPZ files.
    output_path : str or Path, optional
        Path to save figure.
    """
    plt = setup_matplotlib()
    comp_dir = Path(comp_dir)

    # Organize by band and hemisphere
    bands = [7, 6, 3]
    hemispheres = ["leading", "trailing"]

    fig, axes = plt.subplots(len(bands), 2 * 2, figsize=(16, 3.5 * len(bands)))
    # Layout: each row = one band
    # Columns: leading obs | leading residual | trailing obs | trailing residual

    for b, band in enumerate(bands):
        for h, hem in enumerate(hemispheres):
            obs_for_cell = [o for o in THELEN_OBS
                            if o["band"] == band and o["hemisphere"] == hem]
            if not obs_for_cell:
                for k in range(2):
                    col = h * 2 + k
                    axes[b, col].text(0.5, 0.5, "No data", ha="center",
                                      va="center", transform=axes[b, col].transAxes)
                    axes[b, col].set_xticks([])
                    axes[b, col].set_yticks([])
                continue

            # Use first matching observation
            obs_info = obs_for_cell[0]
            label = obs_info["label"]
            comp_file = comp_dir / f"comparison_{label}.npz"

            if not comp_file.exists():
                for k in range(2):
                    col = h * 2 + k
                    axes[b, col].text(0.5, 0.5, f"{label}: missing",
                                      ha="center", va="center",
                                      transform=axes[b, col].transAxes)
                    axes[b, col].set_xticks([])
                    axes[b, col].set_yticks([])
                continue

            data = dict(np.load(str(comp_file), allow_pickle=True))
            on_disk = np.isfinite(data["obs_lat"]) & np.isfinite(data["obs_tb"])

            valid = data["obs_tb"][on_disk]
            if len(valid) == 0:
                continue

            tb_min = np.nanpercentile(valid, 2)
            tb_max = np.nanpercentile(valid, 98)
            res_max = np.nanpercentile(np.abs(data["residual"][on_disk]), 95)

            # Observed Tb
            col_obs = h * 2
            im = plot_disk_image(
                axes[b, col_obs], data["obs_tb"], data["obs_lat"],
                data["obs_lon"], on_disk,
                vmin=tb_min, vmax=tb_max, cmap="inferno",
                title=f"{hem.title()} ({label})" if b == 0 else ""
            )
            plt.colorbar(im, ax=axes[b, col_obs], shrink=0.8)

            # Residual
            col_res = h * 2 + 1
            im_r = plot_disk_image(
                axes[b, col_res], data["residual"], data["obs_lat"],
                data["obs_lon"], on_disk,
                vmin=-res_max, vmax=res_max, cmap="coolwarm",
                title="Residual" if b == 0 else ""
            )
            plt.colorbar(im_r, ax=axes[b, col_res], shrink=0.8)

        # Row label
        freq = {3: "97.5 GHz", 6: "233 GHz", 7: "343.5 GHz"}[band]
        axes[b, 0].set_ylabel(f"Band {band}\n({freq})",
                               fontsize=10, rotation=0, labelpad=50, va="center")

    fig.suptitle("ALMA Multi-Band vs. PTM3D — Thelen et al. (2024)",
                 fontsize=14, y=1.01)
    fig.tight_layout()

    if output_path:
        fig.savefig(str(output_path))
        print(f"  Saved: {output_path}")
    plt.close(fig)


def plot_sensitivity_heatmap(sensitivity_data, obs_info, output_path=None):
    """Plot sensitivity analysis heatmap (RMS vs. emissivity and porosity).

    Parameters
    ----------
    sensitivity_data : dict
        Output from compare_alma_ptm3d.sensitivity_analysis.
    obs_info : dict
        Observation metadata.
    output_path : str or Path, optional
        Path to save figure.
    """
    plt = setup_matplotlib()

    emiss = sensitivity_data["emissivities"]
    poros = sensitivity_data["porosities"]
    rms = sensitivity_data["rms_grid"]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.pcolormesh(poros * 100, emiss, rms, cmap="viridis", shading="auto")
    plt.colorbar(im, ax=ax, label="RMS (K)")

    # Mark best-fit point
    ax.plot(sensitivity_data["best_porosity"] * 100,
            sensitivity_data["best_emissivity"],
            "r*", markersize=15, markeredgecolor="white", markeredgewidth=1)
    ax.annotate(
        f"Best: ε={sensitivity_data['best_emissivity']:.2f}, "
        f"φ={sensitivity_data['best_porosity']*100:.0f}%\n"
        f"RMS={sensitivity_data['best_rms']:.1f} K",
        xy=(sensitivity_data["best_porosity"] * 100,
            sensitivity_data["best_emissivity"]),
        xytext=(10, -20), textcoords="offset points",
        fontsize=9, color="red",
        arrowprops=dict(arrowstyle="->", color="red"),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
    )

    # Mark Trumbo et al. best-fit
    ax.axhline(0.75, color="white", linestyle="--", alpha=0.5, linewidth=1)
    ax.text(poros[-1] * 100, 0.75, " Trumbo ε=0.75", color="white",
            fontsize=8, va="bottom")

    ax.set_xlabel("Porosity (%)", fontsize=12)
    ax.set_ylabel("Emissivity", fontsize=12)
    ax.set_title(f"Sensitivity Analysis — {obs_info['label']} "
                 f"Band {obs_info['band']} ({obs_info['freq_ghz']:.0f} GHz)",
                 fontsize=13)

    fig.tight_layout()
    if output_path:
        fig.savefig(str(output_path))
        print(f"  Saved: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Generate ALMA vs. PTM3D comparison figures"
    )
    parser.add_argument("--comparison-dir",
                        default=str(PROJECT_ROOT / "data" / "alma" / "comparison"),
                        help="Directory with comparison NPZ files")
    parser.add_argument("-o", "--output-dir",
                        default=str(PROJECT_ROOT / "data" / "alma" / "figures"),
                        help="Output directory for figures")
    parser.add_argument("--single", help="Plot single epoch (e.g., T1, H3)")
    parser.add_argument("--composite", action="store_true",
                        help="Generate composite figure for all Trumbo epochs")
    parser.add_argument("--multiband", action="store_true",
                        help="Generate multi-band figure for Thelen epochs")
    parser.add_argument("--all", action="store_true",
                        help="Generate all figures")
    args = parser.parse_args()

    comp_dir = Path(args.comparison_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.single:
        all_obs = get_all_observations()
        match = [o for o in all_obs if o["label"] == args.single]
        if not match:
            print(f"  Unknown observation label: {args.single}")
            print(f"  Available: {[o['label'] for o in all_obs]}")
            return

        obs_info = match[0]
        comp_file = comp_dir / f"comparison_{args.single}.npz"
        if not comp_file.exists():
            print(f"  Comparison file not found: {comp_file}")
            return

        data = dict(np.load(str(comp_file), allow_pickle=True))
        plot_single_comparison(data, obs_info,
                               out_dir / f"alma_comparison_{args.single}.pdf")

    if args.composite or args.all:
        plot_trumbo_composite(comp_dir,
                              out_dir / "alma_trumbo_composite.pdf")

    if args.multiband or args.all:
        plot_thelen_multiband(comp_dir,
                              out_dir / "alma_thelen_multiband.pdf")

    if not any([args.single, args.composite, args.multiband, args.all]):
        parser.print_help()
        print("\n  Use --all to generate all available figures.")


if __name__ == "__main__":
    main()
