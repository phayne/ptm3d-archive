#!/usr/bin/env python3
"""Synthetic compositional map of Europa based on cold trap stability fractions.

For each pixel on a lat/lon grid, randomly assigns surface composition based on
the cold trap fraction at that latitude (interpolated from simulation results).
Each species is either present or absent; the most volatile species present is
shown on top.

Produces:
  - Mollweide projection
  - Single oblique globe view (pole visible)

Usage:
    python scripts/plot_compositional_map.py [--save-dir plots/polar_cold_traps]
"""

import os
import sys
import argparse
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d

# ---------------------------------------------------------------------------
# Publication style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'mathtext.fontset': 'cm',
})


# ---------------------------------------------------------------------------
# Species config (most volatile last = drawn on top)
# ---------------------------------------------------------------------------
# Order: least volatile -> most volatile (by stability temperature)
SPECIES_ORDER = ["H2O", "H2S", "CO2", "CH4"]

# Stability temperatures at 1 mm/Gyr (from simulations)
STABILITY_TEMPS = {
    "H2O": 101.1,
    "CO2": 63.2,
    "H2S": 68.8,
    "CH4": 44.2,
}

# LaTeX labels with subscripts
SPECIES_LABELS = {
    "H2O": r"$\mathrm{H_2O}$",
    "CO2": r"$\mathrm{CO_2}$",
    "H2S": r"$\mathrm{H_2S}$",
    "CH4": r"$\mathrm{CH_4}$",
}

# Colorblind-safe, perceptually uniform colors (Wong 2011 palette)
SPECIES_COLORS = {
    "H2O": "#0072B2",   # blue
    "CO2": "#E69F00",   # orange
    "H2S": "#D55E00",   # vermillion
    "CH4": "#009E73",   # bluish green
}

# Background color for "bare ice / no cold trap"
BG_COLOR = (0.85, 0.85, 0.85)  # light gray


def _save(fig, save_path):
    """Save figure as both PNG and PDF."""
    base, _ = os.path.splitext(save_path)
    fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {base}.png")
    print(f"  Saved: {base}.pdf")


# ---------------------------------------------------------------------------
# Load cold trap fractions from text files
# ---------------------------------------------------------------------------
def _load_cold_trap_table(lat_dir):
    """Load cold trap fractions from a previously saved text file."""
    table_path = os.path.join(lat_dir, "cold_trap_fractions.txt")
    if not os.path.exists(table_path):
        return []
    rows = []
    with open(table_path) as f:
        header_cols = None
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            if line.startswith("DEM"):
                parts = line.split()
                header_cols = [p.split("(")[0] for p in parts[2:]]
                continue
            if line.startswith("-") or header_cols is None:
                continue
            parts = line.split()
            n_species = len(header_cols)
            val_parts = parts[-n_species:]
            faces_part = parts[-(n_species + 1)]
            dem_name = " ".join(parts[:-(n_species + 1)])
            row = {"DEM": dem_name, "N_faces": int(faces_part)}
            for sp, val in zip(header_cols, val_parts):
                row[sp] = float(val.replace("%", ""))
            rows.append(row)
    return rows


def load_all_fractions(plot_dir):
    """Load cold trap fractions at all latitudes from saved text files.

    Returns dict {species: interp1d(latitude -> fraction/100)}.
    Assumes north-south symmetry.
    """
    all_latitudes = [0, 15, 30, 45, 60, 75, 80, 85, 87]
    species_data = {sp: [] for sp in SPECIES_ORDER}

    for lat in all_latitudes:
        if lat == 87:
            lat_dir = os.path.join(plot_dir, "lat87")
        else:
            lat_dir = os.path.join(plot_dir, f"lat{int(lat):02d}")
        table_path = os.path.join(lat_dir, "cold_trap_fractions.txt")
        if not os.path.exists(table_path):
            print(f"  WARNING: {table_path} not found, skipping lat={lat}")
            continue

        header_cols = None
        fracs_per_species = {sp: [] for sp in SPECIES_ORDER}
        with open(table_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or not line:
                    continue
                if line.startswith("DEM"):
                    parts = line.split()
                    header_cols = [p.split("(")[0] for p in parts[2:]]
                    continue
                if line.startswith("-") or header_cols is None:
                    continue
                parts = line.split()
                n_sp = len(header_cols)
                val_parts = parts[-n_sp:]
                for sp_name, val in zip(header_cols, val_parts):
                    if sp_name in fracs_per_species:
                        fracs_per_species[sp_name].append(
                            float(val.replace("%", "")))

        for sp in SPECIES_ORDER:
            if fracs_per_species[sp]:
                mean_frac = np.mean(fracs_per_species[sp])
                species_data[sp].append((lat, mean_frac / 100.0))

    interpolators = {}
    for sp in SPECIES_ORDER:
        if not species_data[sp]:
            interpolators[sp] = lambda lat: 0.0
            continue
        lats = np.array([x[0] for x in species_data[sp]])
        fracs = np.array([x[1] for x in species_data[sp]])
        if 90 not in lats:
            lats = np.append(lats, 90.0)
            fracs = np.append(fracs, fracs[-1])
        interpolators[sp] = interp1d(lats, fracs, kind='linear',
                                     bounds_error=False, fill_value=0.0)

    return interpolators


# ---------------------------------------------------------------------------
# Generate synthetic composition grid
# ---------------------------------------------------------------------------
def generate_composition_map(interpolators, n_lat=361, n_lon=721, seed=42):
    """Generate a synthetic composition map on a regular lat/lon grid.

    Returns:
        lat_grid, lon_grid, comp_rgb (n_lat, n_lon, 3), comp_id (n_lat, n_lon)
    """
    rng = np.random.default_rng(seed)

    lat_grid = np.linspace(-90, 90, n_lat)
    lon_grid = np.linspace(0, 360, n_lon)

    comp_id = np.zeros((n_lat, n_lon), dtype=int)
    comp_rgb = np.full((n_lat, n_lon, 3), BG_COLOR, dtype=float)

    for i, lat in enumerate(lat_grid):
        abs_lat = abs(lat)
        for sp_idx, sp in enumerate(SPECIES_ORDER, start=1):
            frac = float(interpolators[sp](abs_lat))
            if frac <= 0:
                continue
            mask = rng.random(n_lon) < frac
            comp_id[i, mask] = sp_idx
            comp_rgb[i, mask] = colors.to_rgb(SPECIES_COLORS[sp])

    return lat_grid, lon_grid, comp_rgb, comp_id


# ---------------------------------------------------------------------------
# 3D Globe rendering
# ---------------------------------------------------------------------------
def make_globe_xyz(lat_grid, lon_grid):
    """Create X, Y, Z coordinates for a globe surface."""
    lat_rad = np.deg2rad(lat_grid)
    lon_rad = np.deg2rad(lon_grid)
    LON, LAT = np.meshgrid(lon_rad, lat_rad)
    X = np.cos(LAT) * np.cos(LON)
    Y = np.cos(LAT) * np.sin(LON)
    Z = np.sin(LAT)
    return X, Y, Z


def draw_graticule(ax, elev, azim, spacing=30, color='0.45', linewidth=0.3,
                   alpha=0.4):
    """Draw latitude/longitude graticule lines on a 3D globe."""
    R = 1.002
    elev_r = np.deg2rad(elev)
    azim_r = np.deg2rad(azim)
    vx = np.cos(elev_r) * np.cos(azim_r)
    vy = np.cos(elev_r) * np.sin(azim_r)
    vz = np.sin(elev_r)

    t = np.linspace(0, 2 * np.pi, 360)

    for lat_deg in range(-90 + spacing, 90, spacing):
        lat_r = np.deg2rad(lat_deg)
        x = R * np.cos(lat_r) * np.cos(t)
        y = R * np.cos(lat_r) * np.sin(t)
        z = R * np.sin(lat_r) * np.ones_like(t)
        dot = x * vx + y * vy + z * vz
        x_vis = np.where(dot > 0, x, np.nan)
        y_vis = np.where(dot > 0, y, np.nan)
        z_vis = np.where(dot > 0, z, np.nan)
        ax.plot(x_vis, y_vis, z_vis, color=color, linewidth=linewidth,
                alpha=alpha, zorder=10)

    lat_t = np.linspace(-np.pi / 2, np.pi / 2, 180)
    for lon_deg in range(0, 360, spacing):
        lon_r = np.deg2rad(lon_deg)
        x = R * np.cos(lat_t) * np.cos(lon_r)
        y = R * np.cos(lat_t) * np.sin(lon_r)
        z = R * np.sin(lat_t)
        dot = x * vx + y * vy + z * vz
        x_vis = np.where(dot > 0, x, np.nan)
        y_vis = np.where(dot > 0, y, np.nan)
        z_vis = np.where(dot > 0, z, np.nan)
        ax.plot(x_vis, y_vis, z_vis, color=color, linewidth=linewidth,
                alpha=alpha, zorder=10)


def plot_globe(ax, lat_grid, lon_grid, comp_rgb, elev, azim):
    """Plot a single globe view with per-pixel RGB colors."""
    X, Y, Z = make_globe_xyz(lat_grid, lon_grid)
    fc = np.zeros((len(lat_grid) - 1, len(lon_grid) - 1, 4))
    fc[:, :, :3] = comp_rgb[:-1, :-1, :]
    fc[:, :, 3] = 1.0

    ax.plot_surface(X, Y, Z, facecolors=fc,
                    rstride=1, cstride=1, shade=False,
                    antialiased=False, rasterized=True)

    draw_graticule(ax, elev, azim)

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_zlim(-1.1, 1.1)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_box_aspect([1, 1, 1])


# ---------------------------------------------------------------------------
# Mollweide projection
# ---------------------------------------------------------------------------
def plot_mollweide(ax, lat_grid, lon_grid, comp_id, lon_center=0):
    """Plot composition on Mollweide projection using categorical colormap."""
    cmap_colors = [BG_COLOR]
    for sp in SPECIES_ORDER:
        cmap_colors.append(colors.to_rgb(SPECIES_COLORS[sp]))
    cmap = colors.ListedColormap(cmap_colors)
    norm = colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)

    lon_shifted = lon_grid - lon_center
    lon_shifted = (lon_shifted + 180) % 360 - 180
    order = np.argsort(lon_shifted)
    lon_sorted = lon_shifted[order]
    id_sorted = comp_id[:, order]

    lat_rad = np.deg2rad(lat_grid)
    lon_rad = np.deg2rad(lon_sorted)
    LON, LAT = np.meshgrid(lon_rad, lat_rad)

    pcm = ax.pcolormesh(LON, LAT, id_sorted, cmap=cmap, norm=norm,
                        shading='auto', rasterized=True)

    ax.grid(True, color='0.3', linewidth=0.3, alpha=0.4)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    return pcm


def add_mollweide_labels(ax, lon_center=0):
    """Add longitude/latitude tick labels around a Mollweide plot."""
    for lat_deg in [-60, -30, 0, 30, 60]:
        ax.text(np.deg2rad(185 - lon_center), np.deg2rad(lat_deg),
                f'{abs(lat_deg)}\u00b0{"N" if lat_deg > 0 else "S" if lat_deg < 0 else ""}',
                fontsize=5.5, color='0.4', ha='left', va='center',
                transform=ax.transData)

    for lon_deg in [0, 90, 180, 270]:
        lon_plot = lon_deg - lon_center
        if lon_plot > 180:
            lon_plot -= 360
        elif lon_plot < -180:
            lon_plot += 360
        if abs(lon_plot) > 170:
            continue
        ax.text(np.deg2rad(lon_plot), np.deg2rad(-8),
                f'{lon_deg}\u00b0E',
                fontsize=5.5, color='0.4', ha='center', va='top',
                transform=ax.transData)


# ---------------------------------------------------------------------------
# Species legend
# ---------------------------------------------------------------------------
def make_legend_handles():
    """Create legend handles for species + background, with subscripts."""
    handles = []
    handles.append(Patch(facecolor=BG_COLOR, edgecolor='0.5', linewidth=0.5,
                         label='Sublimation-dominated'))
    for sp in SPECIES_ORDER:
        label = f'{SPECIES_LABELS[sp]} ({STABILITY_TEMPS[sp]:.0f} K)'
        handles.append(Patch(facecolor=SPECIES_COLORS[sp], edgecolor='0.5',
                             linewidth=0.5, label=label))
    return handles


# ---------------------------------------------------------------------------
# Figure: Single oblique globe
# ---------------------------------------------------------------------------
def fig_globe(lat_grid, lon_grid, comp_rgb, save_path):
    """Single oblique globe view with pole visible."""
    fig = plt.figure(figsize=(3.5, 4.0))
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)

    # elev=45 gives a nice oblique view with pole clearly visible
    plot_globe(ax, lat_grid, lon_grid, comp_rgb, elev=45, azim=0)

    handles = make_legend_handles()
    fig.legend(handles=handles, loc='lower center', ncol=1,
               fontsize=8, frameon=True, framealpha=0.95, edgecolor='0.8',
               handlelength=1.2, handleheight=0.9,
               bbox_to_anchor=(0.5, -0.01))

    _save(fig, save_path)


# ---------------------------------------------------------------------------
# Figure: Mollweide
# ---------------------------------------------------------------------------
def fig_mollweide(lat_grid, lon_grid, comp_id, save_path):
    """Mollweide projection of compositional map."""
    fig = plt.figure(figsize=(5.5, 3.5))
    ax = fig.add_subplot(111, projection='mollweide')

    plot_mollweide(ax, lat_grid, lon_grid, comp_id, lon_center=0)
    add_mollweide_labels(ax, lon_center=0)

    handles = make_legend_handles()
    fig.legend(handles=handles, loc='lower center', ncol=len(handles),
               fontsize=7.5, frameon=True, framealpha=0.95, edgecolor='0.8',
               handlelength=1.2, handleheight=0.9,
               bbox_to_anchor=(0.5, 0.01))

    _save(fig, save_path)


# ---------------------------------------------------------------------------
# Figure: North polar view
# ---------------------------------------------------------------------------
def fig_polar(lat_grid, lon_grid, comp_rgb, save_path):
    """North polar globe view."""
    fig = plt.figure(figsize=(3.5, 4.2))
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)

    plot_globe(ax, lat_grid, lon_grid, comp_rgb, elev=90, azim=0)

    handles = make_legend_handles()
    fig.legend(handles=handles, loc='lower center', ncol=1,
               fontsize=8, frameon=True, framealpha=0.95, edgecolor='0.8',
               handlelength=1.2, handleheight=0.9,
               bbox_to_anchor=(0.5, -0.01))

    _save(fig, save_path)


# ---------------------------------------------------------------------------
# Helper: render globe to RGBA image array
# ---------------------------------------------------------------------------
def _render_globe_to_image(lat_grid, lon_grid, comp_rgb, elev, azim,
                           figsize=(3.0, 3.0), dpi=200):
    """Render a globe view to an RGBA numpy array for inset use."""
    fig = plt.figure(figsize=figsize, dpi=dpi)
    fig.patch.set_alpha(0.0)
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
    ax.set_facecolor((1, 1, 1, 0))
    ax.patch.set_alpha(0.0)
    plot_globe(ax, lat_grid, lon_grid, comp_rgb, elev, azim)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf).copy()
    plt.close(fig)
    # Make white background pixels transparent
    white_mask = (img[:, :, 0] > 240) & (img[:, :, 1] > 240) & (img[:, :, 2] > 240)
    img[white_mask, 3] = 0
    # Crop transparent borders
    alpha = img[:, :, 3]
    rows = np.any(alpha > 10, axis=1)
    cols = np.any(alpha > 10, axis=0)
    if rows.any() and cols.any():
        r0, r1 = np.where(rows)[0][[0, -1]]
        c0, c1 = np.where(cols)[0][[0, -1]]
        pad = 5
        r0, r1 = max(r0 - pad, 0), min(r1 + pad, img.shape[0] - 1)
        c0, c1 = max(c0 - pad, 0), min(c1 + pad, img.shape[1] - 1)
        img = img[r0:r1+1, c0:c1+1]
    return img


# ---------------------------------------------------------------------------
# Figure: Combined (stability vs latitude + globe insets)
# ---------------------------------------------------------------------------
def fig_combined(lat_grid, lon_grid, comp_rgb, all_results, stability_temps,
                 save_path):
    """Combined figure: stability vs latitude plot with globe insets.

    Two globe insets (oblique + polar) placed in the white space of the
    lower-left portion of the main plot (where all species are at 0%).
    """
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    # --- Render globe images ---
    print("    Rendering oblique globe...")
    img_oblique = _render_globe_to_image(
        lat_grid, lon_grid, comp_rgb, elev=45, azim=0,
        figsize=(3.5, 3.5), dpi=250)
    print("    Rendering polar globe...")
    img_polar = _render_globe_to_image(
        lat_grid, lon_grid, comp_rgb, elev=90, azim=0,
        figsize=(3.5, 3.5), dpi=250)

    # --- Main stability plot ---
    plot_species = ["H2O", "CO2", "H2S", "CH4"]
    species_markers = {"H2O": "o", "CO2": "s", "H2S": "D", "CH4": "^"}

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for species in plot_species:
        T_stab = stability_temps.get(species, float("nan"))
        if np.isnan(T_stab):
            continue
        means, stds = [], []
        latitudes = sorted(all_results.keys())
        for lat in latitudes:
            fracs = [row[species] for row in all_results[lat]
                     if species in row]
            means.append(np.mean(fracs))
            stds.append(np.std(fracs))
        means = np.array(means)
        stds = np.array(stds)

        color = SPECIES_COLORS[species]
        marker = species_markers[species]
        ax.plot(latitudes, means, marker=marker, linestyle="-",
                color=color, linewidth=2.0, markersize=7,
                markeredgecolor="white", markeredgewidth=0.8, zorder=5,
                label=f"{SPECIES_LABELS[species]} ({T_stab:.0f} K)")
        ax.fill_between(latitudes, np.clip(means - stds, 0, 100),
                         np.clip(means + stds, 0, 100),
                         alpha=0.15, color=color, linewidth=0)

    ax.set_xlabel("Latitude (\u00b0N)", fontsize=13)
    ax.set_ylabel("Cold Trap Fraction (%)", fontsize=13)
    ax.legend(loc="center left", frameon=True, framealpha=0.95,
              edgecolor="0.8", fancybox=False, fontsize=11,
              bbox_to_anchor=(0.37, 0.45))
    ax.set_xlim(-2, 90)
    ax.set_ylim(-2, 105)
    ax.set_xticks([0, 15, 30, 45, 60, 75, 87])
    ax.tick_params(direction="in", top=True, right=True, labelsize=11)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.minorticks_on()
    ax.tick_params(which="minor", direction="in", top=True, right=True,
                   length=3)

    # --- Globe insets stacked vertically in upper-left ---
    # Oblique globe (top)
    ax_obl = ax.inset_axes([0.01, 0.50, 0.28, 0.48])
    ax_obl.imshow(img_oblique)
    ax_obl.axis("off")
    ax_obl.patch.set_alpha(0.0)

    # Polar globe (below oblique)
    ax_pol = ax.inset_axes([0.01, 0.05, 0.28, 0.48])
    ax_pol.imshow(img_polar)
    ax_pol.axis("off")
    ax_pol.patch.set_alpha(0.0)

    fig.tight_layout()
    _save(fig, save_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Synthetic compositional map of Europa from cold trap data")
    parser.add_argument("--data-dir", default="plots/polar_cold_traps",
                        help="Directory with lat{XX}/cold_trap_fractions.txt")
    parser.add_argument("--save-dir", default="plots/polar_cold_traps",
                        help="Output directory for figures")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--resolution", type=int, default=361,
                        help="Number of latitude points (default: 361)")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    print("Loading cold trap fractions from simulation data...")
    interpolators = load_all_fractions(args.data_dir)

    print("\nInterpolated cold trap fractions:")
    print(f"  {'Lat':>5s}", end="")
    for sp in SPECIES_ORDER:
        print(f"  {sp:>6s}", end="")
    print()
    for lat in [0, 30, 45, 60, 75, 80, 85, 87, 90]:
        print(f"  {lat:5.0f}\u00b0", end="")
        for sp in SPECIES_ORDER:
            print(f"  {interpolators[sp](lat)*100:5.1f}%", end="")
        print()

    n_lat = args.resolution
    n_lon = 2 * n_lat - 1
    print(f"\nGenerating composition map ({n_lat}\u00d7{n_lon})...")
    lat_grid, lon_grid, comp_rgb, comp_id = generate_composition_map(
        interpolators, n_lat=n_lat, n_lon=n_lon, seed=args.seed)

    n_total = comp_id.size
    print(f"\nComposition statistics:")
    print(f"  {'Bare ice':<12s}: {np.sum(comp_id == 0) / n_total * 100:5.1f}%")
    for i, sp in enumerate(SPECIES_ORDER, start=1):
        print(f"  {sp:<12s}: {np.sum(comp_id == i) / n_total * 100:5.1f}%")

    print("\nGenerating globe view...")
    fig_globe(lat_grid, lon_grid, comp_rgb,
              os.path.join(args.save_dir, "composition_globe.png"))

    print("Generating Mollweide projection...")
    fig_mollweide(lat_grid, lon_grid, comp_id,
                  os.path.join(args.save_dir, "composition_mollweide.png"))

    print("Generating polar view...")
    fig_polar(lat_grid, lon_grid, comp_rgb,
              os.path.join(args.save_dir, "composition_polar.png"))

    # Combined figure: stability vs latitude + globe insets
    print("Loading cold trap tables for combined figure...")
    all_latitudes = [0, 15, 30, 45, 60, 75, 80, 85, 87]
    all_results = {}
    for lat in all_latitudes:
        if lat == 87:
            lat_dir = os.path.join(args.data_dir, "lat87")
        else:
            lat_dir = os.path.join(args.data_dir, f"lat{int(lat):02d}")
        table = _load_cold_trap_table(lat_dir)
        if table:
            all_results[lat] = table

    if all_results:
        print("Generating combined figure...")
        fig_combined(lat_grid, lon_grid, comp_rgb, all_results,
                     STABILITY_TEMPS,
                     os.path.join(args.save_dir, "composition_combined.png"))

    print("\nDone!")


if __name__ == "__main__":
    main()
