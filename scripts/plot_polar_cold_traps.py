#!/usr/bin/env python3
"""Plot polar cold trap analysis for Europa DEMs at 87° latitude.

Generates for each DEM:
  1. 3D PyVista T_max and T_min maps
  2. Temperature histograms (T_max, T_min, T_mean) with volatile stability lines

Also generates:
  3. Combined histogram comparing all 5 DEMs
  4. Cold trap fraction summary table

Usage:
    python scripts/plot_polar_cold_traps.py [--plot-dir plots/polar_cold_traps]
"""

import argparse
import struct
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from scipy.optimize import brentq

# ──────────────────────────────────────────────────────────────────────
# Physical constants
# ──────────────────────────────────────────────────────────────────────
k_B = 1.380649e-23      # Boltzmann constant [J/K]
AMU = 1.66054e-27        # atomic mass unit [kg]
SEC_PER_GYR = 1e9 * 365.25 * 86400  # seconds per Gyr

# ──────────────────────────────────────────────────────────────────────
# Vapor pressure functions (Pa) — all for solid phase
# ──────────────────────────────────────────────────────────────────────
def pvap_h2o(T):
    """H2O ice vapor pressure [Pa]. Marti & Mauersberger (1993)."""
    T = np.asarray(T, dtype=float)
    return np.where(T > 0, 10.0**(-2663.5 / T + 12.537), 0.0)

def pvap_co2(T):
    """CO2 ice vapor pressure [Pa]. Fray & Schmitt (2009), T < 194.7 K."""
    T = np.asarray(T, dtype=float)
    # log10(P/Pa) = 6.81228 - 1301.679/T - 0.001206*T  (Giauque & Egan 1937 / Fray & Schmitt fit)
    return np.where(T > 10, 10.0**(6.81228 - 1301.679 / T - 0.001206 * T), 0.0)

def pvap_h2s(T):
    """H2S ice vapor pressure [Pa]. Fray & Schmitt (2009)."""
    T = np.asarray(T, dtype=float)
    return np.where(T > 10, 10.0**(-1768.69 / T + 11.7859), 0.0)

def pvap_ch4(T):
    """CH4 ice vapor pressure [Pa]. Fray & Schmitt (2009), T < 90.7 K."""
    T = np.asarray(T, dtype=float)
    return np.where(T > 5, 10.0**(-1159.4 / T + 12.855 - 0.01985 * T), 0.0)

def pvap_o2(T):
    """O2 ice vapor pressure [Pa]. NIST / Honig & Hook (1960)."""
    T = np.asarray(T, dtype=float)
    return np.where(T > 5, 10.0**(-468.0 / T + 7.714), 0.0)

def pvap_h2(T):
    """H2 ice vapor pressure [Pa]. NIST / Leachman et al."""
    T = np.asarray(T, dtype=float)
    return np.where(T > 1, 10.0**(-47.203 / T + 5.836), 0.0)


# Species database — colorblind-safe Wong (2011) palette
SPECIES = {
    "H2O":  {"pvap": pvap_h2o,  "mass": 18.015 * AMU, "rho": 917.0,
             "color": "#0072B2", "Trange": (50, 200)},
    "CO2":  {"pvap": pvap_co2,  "mass": 44.01  * AMU, "rho": 1562.0,
             "color": "#E69F00", "Trange": (30, 150)},
    "H2S":  {"pvap": pvap_h2s,  "mass": 34.08  * AMU, "rho": 1120.0,
             "color": "#D55E00", "Trange": (30, 150)},
    "CH4":  {"pvap": pvap_ch4,  "mass": 16.04  * AMU, "rho": 494.0,
             "color": "#009E73", "Trange": (10, 80)},
    "O2":   {"pvap": pvap_o2,   "mass": 32.00  * AMU, "rho": 1141.0,
             "color": "#CC79A7", "Trange": (10, 50)},
    "H2":   {"pvap": pvap_h2,   "mass": 2.016  * AMU, "rho": 86.0,
             "color": "#F0E442", "Trange": (2, 20)},
}

# LaTeX labels with subscripts
SPECIES_LABELS = {
    "H2O": r"$\mathrm{H_2O}$",
    "CO2": r"$\mathrm{CO_2}$",
    "H2S": r"$\mathrm{H_2S}$",
    "CH4": r"$\mathrm{CH_4}$",
    "O2":  r"$\mathrm{O_2}$",
    "H2":  r"$\mathrm{H_2}$",
}


def sublimation_flux(T, pvap_func, mol_mass):
    """Hertz-Knudsen sublimation mass flux [kg/m^2/s]."""
    P = pvap_func(T)
    return P * np.sqrt(mol_mass / (2 * np.pi * k_B * T))


def stability_temperature(species_name, loss_rate_m_per_Gyr=1.0):
    """Find T where sublimation depth loss rate = loss_rate [mm/Gyr].

    Returns T_stability in Kelvin.
    """
    sp = SPECIES[species_name]
    # Target flux in kg/m^2/s
    target = sp["rho"] * (loss_rate_m_per_Gyr * 1e-3) / SEC_PER_GYR

    Tlo, Thi = sp["Trange"]

    def residual(T):
        return sublimation_flux(T, sp["pvap"], sp["mass"]) - target

    try:
        return brentq(residual, Tlo, Thi, xtol=0.01)
    except ValueError:
        # If flux never reaches target in range, return NaN
        return float("nan")


# ──────────────────────────────────────────────────────────────────────
# I/O helpers (adapted from plot_ridge_heat_flow.py)
# ──────────────────────────────────────────────────────────────────────
# (label, population size, excluded count) triples already warned about --
# _frac_below() runs once per (DEM x species) and would otherwise emit the same
# line hundreds of times for one bad mesh.
_WARNED = set()


def _warn_unresolved(label, resolved):
    """Warn loudly when facets leave the population; return how many did."""
    n_unresolved = int(resolved.size) - int(np.count_nonzero(resolved))
    key = (label, int(resolved.size), n_unresolved)
    if n_unresolved and key not in _WARNED:
        _WARNED.add(key)
        print(f"  WARNING: {label}: {n_unresolved} of {resolved.size} faces have "
              "no value (non-finite) and leave BOTH the numerator and the "
              "denominator of every fraction below.", file=sys.stderr)
    return n_unresolved


def _frac_below(T_arr, threshold):
    """Fraction of faces below `threshold`, over the RESOLVED faces only.

    Paul's ruling, 2026-09-15: "If a facet has no value, then it should leave
    the denominator and the numerator." A facet whose value is not finite --
    NaN, +inf or -inf -- is unresolved and leaves the population entirely. A
    bare ``np.sum(T_arr < threshold) / len(T_arr)`` instead counts a NaN face
    as not-qualifying and a -inf face AS qualifying, both silently.
    """
    T_arr = np.asarray(T_arr)
    resolved = np.isfinite(T_arr)
    n_resolved = int(np.count_nonzero(resolved))
    _warn_unresolved("cold-trap fraction", resolved)
    if not np.isfinite(threshold) or T_arr.size == 0:
        return 0.0
    if n_resolved == 0:
        return float("nan")
    return float(np.count_nonzero(resolved & (T_arr < threshold)) / n_resolved)


def load_ply_mesh(filepath):
    """Load binary little-endian PLY mesh."""
    with open(filepath, "rb") as f:
        header_lines = []
        while True:
            line = f.readline().decode("ascii").strip()
            header_lines.append(line)
            if line == "end_header":
                break

        n_verts = n_faces = 0
        for line in header_lines:
            if line.startswith("element vertex"):
                n_verts = int(line.split()[-1])
            elif line.startswith("element face"):
                n_faces = int(line.split()[-1])

        vert_data = np.frombuffer(f.read(n_verts * 6 * 8), dtype="<f8").reshape(n_verts, 6)
        vertices = vert_data[:, :3]

        face_record_size = 1 + 3 * 4 + 2 * 4
        face_data = f.read(n_faces * face_record_size)
        faces = np.empty((n_faces, 3), dtype=np.int32)
        for i in range(n_faces):
            offset = i * face_record_size
            v0, v1, v2 = struct.unpack_from("<iii", face_data, offset + 1)
            faces[i] = [v0, v1, v2]

    return vertices, faces


def load_binary_stat(filepath, n_faces):
    """Load a binary statistics file (int32 count + float32 per face)."""
    data = np.fromfile(filepath, dtype=np.float32)
    if len(data) == n_faces + 1:
        data = data[1:]
    elif len(data) != n_faces:
        raise ValueError(f"Expected {n_faces}(+1) values, got {len(data)}")
    return data


def make_pv_mesh(vertices, faces):
    """Create a PyVista PolyData mesh."""
    n_faces_count = len(faces)
    pv_faces = np.column_stack([
        np.full(n_faces_count, 3, dtype=np.int64),
        faces.astype(np.int64)
    ]).ravel()
    return pv.PolyData(vertices, pv_faces)


def _autocrop(path, pad=20):
    """Crop transparent borders from a screenshot."""
    from PIL import Image
    img = Image.open(path).convert("RGBA")
    alpha = np.array(img.split()[3])
    rows = np.any(alpha > 0, axis=1)
    cols = np.any(alpha > 0, axis=0)
    if not rows.any():
        return
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    r0 = max(r0 - pad, 0)
    r1 = min(r1 + pad, img.height - 1)
    c0 = max(c0 - pad, 0)
    c1 = min(c1 + pad, img.width - 1)
    img.crop((c0, r0, c1 + 1, r1 + 1)).save(path)


# ──────────────────────────────────────────────────────────────────────
# Plotting functions
# ──────────────────────────────────────────────────────────────────────
def plot_3d_pyvista(pv_mesh, scalars, vmin, vmax, cmap, save_path,
                    scalar_label="Temperature (K)"):
    """Render a 3D mesh with PyVista (off-screen) for publication."""
    plotter = pv.Plotter(off_screen=True, window_size=[2400, 1600])
    plotter.set_background("white")

    pv_mesh[scalar_label] = scalars
    plotter.add_mesh(
        pv_mesh, scalars=scalar_label, cmap=cmap,
        clim=[vmin, vmax], show_edges=False,
        lighting=True, smooth_shading=True,
        scalar_bar_args={
            "title": scalar_label,
            "color": "black",
            "title_font_size": 40,
            "label_font_size": 36,
            "vertical": True,
            "position_x": 0.87, "position_y": 0.08,
            "width": 0.06, "height": 0.78,
            "n_labels": 5, "fmt": "%.1f",
        },
    )

    # Camera at 50° emission angle
    pts = pv_mesh.points
    center = pts.mean(axis=0)
    normal = center / np.linalg.norm(center)
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(normal, ref)) > 0.99:
        ref = np.array([0.0, 1.0, 0.0])
    tangent = np.cross(normal, ref)
    tangent /= np.linalg.norm(tangent)

    emission_angle = np.radians(50.0)
    cam_dir = np.cos(emission_angle) * normal + np.sin(emission_angle) * tangent
    cam_dir /= np.linalg.norm(cam_dir)
    mesh_extent = np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))
    cam_dist = mesh_extent * 1.1
    cam_pos = center + cam_dist * cam_dir
    view_up = normal - np.dot(normal, cam_dir) * cam_dir
    view_up /= np.linalg.norm(view_up)
    plotter.camera_position = [tuple(cam_pos), tuple(center), tuple(view_up)]
    plotter.camera.zoom(0.81)

    plotter.remove_all_lights()
    plotter.enable_lightkit()

    plotter.screenshot(save_path, transparent_background=True)
    plotter.close()
    _autocrop(save_path)
    print(f"  Saved: {save_path}")


def plot_cold_trap_histogram(T_max, T_min, T_mean, dem_label, stability_temps,
                             save_path):
    """Temperature histogram with volatile stability lines."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    datasets = [
        (T_max, r"$T_{\mathrm{max}}$ (K)", "Maximum"),
        (T_min, r"$T_{\mathrm{min}}$ (K)", "Minimum"),
        (T_mean, r"$T_{\mathrm{mean}}$ (K)", "Mean"),
    ]

    for ax, (data, xlabel, stat_name) in zip(axes, datasets):
        ax.hist(data, bins=100, alpha=0.7, color="steelblue",
                edgecolor="navy", linewidth=0.3, density=True)

        for species, T_stab in stability_temps.items():
            if np.isnan(T_stab):
                continue
            sp = SPECIES[species]
            if T_stab < data.max() * 1.3 and T_stab > data.min() * 0.5:
                frac = _frac_below(data, T_stab) * 100
                sp_label = SPECIES_LABELS.get(species, species)
                label = f"{sp_label}: {T_stab:.0f} K ({frac:.1f}%)"
                ax.axvline(T_stab, color=sp["color"], linestyle="--",
                           linewidth=1.5, label=label)

        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("Probability Density", fontsize=11)
        ax.set_title(f"{dem_label} — {stat_name}", fontsize=12)
        ax.legend(fontsize=8, loc="upper right")
        ax.tick_params(labelsize=10)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_combined_histogram(all_T_max, dem_labels, stability_temps, save_path):
    """Combined T_max histogram with log10 y-axis and filled stair step.

    Each DEM normalized by its face count, then averaged.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Common bin edges across all DEMs
    global_min = min(t.min() for t in all_T_max.values())
    global_max = max(t.max() for t in all_T_max.values())
    bins = np.linspace(global_min, global_max, 81)
    bin_width = bins[1] - bins[0]

    # Accumulate normalized histograms (each DEM integrates to 1)
    combined = np.zeros(len(bins) - 1)
    n_dems = 0
    for key, T_max in all_T_max.items():
        counts, _ = np.histogram(T_max, bins=bins)
        combined += counts / (len(T_max) * bin_width)
        n_dems += 1
    combined /= n_dems

    # Filled stair step plot
    ax.stairs(combined, bins, fill=True, alpha=0.7,
              color="steelblue", edgecolor="navy", linewidth=0.5)

    for species, T_stab in stability_temps.items():
        if np.isnan(T_stab):
            continue
        sp = SPECIES[species]
        if T_stab > 10 and T_stab < global_max * 1.3:
            sp_label = SPECIES_LABELS.get(species, species)
            ax.axvline(T_stab, color=sp["color"], linestyle="--",
                       linewidth=2.0, label=f"{sp_label}: {T_stab:.0f} K")

    ax.set_yscale("log")
    ax.set_xlabel(r"$T_{\mathrm{max}}$ (K)", fontsize=13)
    ax.set_ylabel("Probability Density", fontsize=13)
    ax.set_title("Maximum Diurnal Temperature at 87°N — All DEMs Combined",
                 fontsize=14)
    ax.legend(fontsize=9, ncol=2, loc="upper right")
    ax.tick_params(labelsize=11)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def _render_3d_no_cbar(pv_mesh, scalars, vmin, vmax, cmap):
    """Render a 3D mesh to an RGBA numpy array (no scalar bar)."""
    plotter = pv.Plotter(off_screen=True, window_size=[1200, 1000])
    plotter.set_background("white")

    pv_mesh["T"] = scalars
    plotter.add_mesh(
        pv_mesh, scalars="T", cmap=cmap,
        clim=[vmin, vmax], show_edges=False,
        lighting=True, smooth_shading=True,
        show_scalar_bar=False,
    )

    pts = pv_mesh.points
    center = pts.mean(axis=0)
    normal = center / np.linalg.norm(center)
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(normal, ref)) > 0.99:
        ref = np.array([0.0, 1.0, 0.0])
    tangent = np.cross(normal, ref)
    tangent /= np.linalg.norm(tangent)

    emission_angle = np.radians(50.0)
    cam_dir = np.cos(emission_angle) * normal + np.sin(emission_angle) * tangent
    cam_dir /= np.linalg.norm(cam_dir)
    mesh_extent = np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))
    cam_dist = mesh_extent * 1.1
    cam_pos = center + cam_dist * cam_dir
    view_up = normal - np.dot(normal, cam_dir) * cam_dir
    view_up /= np.linalg.norm(view_up)
    plotter.camera_position = [tuple(cam_pos), tuple(center), tuple(view_up)]
    plotter.camera.zoom(0.81)

    plotter.remove_all_lights()
    plotter.enable_lightkit()

    img = plotter.screenshot(transparent_background=False, return_img=True)
    plotter.close()
    return img


def _crop_content(img_array, bg_color=(255, 255, 255), pad=10):
    """Crop white borders from an RGB image array, return cropped array."""
    mask = np.any(img_array != bg_color, axis=2)
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return img_array
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    r0 = max(r0 - pad, 0)
    r1 = min(r1 + pad, img_array.shape[0] - 1)
    c0 = max(c0 - pad, 0)
    c1 = min(c1 + pad, img_array.shape[1] - 1)
    return img_array[r0:r1+1, c0:c1+1]


def _classify_species_colors(T_max, stability_temps):
    """Assign RGB color per face based on most volatile stable species.

    Paints from least volatile (H2O) to most volatile (H2), so smaller
    (more volatile) regions overwrite on top.

    Returns (n_faces, 3) uint8 RGB array and ordered list of
    (species_name, color_rgb, fraction) for the legend.
    """
    import matplotlib.colors as mcolors

    n_faces = len(T_max)
    rgb = np.full((n_faces, 3), 200, dtype=np.uint8)  # light gray = unstable

    # Sort by stability temperature descending (least volatile first)
    species_order = sorted(
        [(s, T) for s, T in stability_temps.items() if not np.isnan(T)],
        key=lambda x: x[1], reverse=True,
    )

    # Paul's ruling, 2026-09-15: an unresolved face leaves BOTH the numerator
    # and the denominator, so every legend percentage below is over the faces
    # that actually have a temperature -- and an unresolved face is never
    # painted with a species color either.
    resolved = np.isfinite(T_max)
    n_resolved = int(np.count_nonzero(resolved))
    _warn_unresolved("species map", resolved)
    legend_entries = []
    # "Sublimation-dominated" entry
    legend_entries.append(("Sublimation-dominated", (0.784, 0.784, 0.784),
                           np.count_nonzero(resolved & (T_max >= species_order[0][1]))
                           / n_resolved * 100
                           if species_order and n_resolved else 100.0))

    for species, T_stab in species_order:
        mask = resolved & (T_max < T_stab)
        color_rgb = mcolors.to_rgb(SPECIES[species]["color"])
        color_uint8 = (np.array(color_rgb) * 255).astype(np.uint8)
        rgb[mask] = color_uint8
        frac = np.count_nonzero(mask) / n_resolved * 100 if n_resolved else float("nan")
        label = SPECIES_LABELS.get(species, species)
        legend_entries.append((label, color_rgb, frac))

    # A face with no value must not READ as a classification. The default fill
    # is the same light gray as the "Sublimation-dominated" swatch, so leaving
    # it would render an absence as a confirmed sublimation-dominated face --
    # a wrong figure, not a missing caveat. Paint it white.
    #
    # No legend entry and no other annotation: Paul, 2026-09-15, "Do not
    # restyle figures or present disclosures/caveats on figures -- these can be
    # recorded in notes outside of the paper that I can incorporate as needed."
    # The count is warned to stderr by _warn_unresolved() above and written to
    # the cold_trap_fractions.txt header, which is where it belongs.
    rgb[~resolved] = np.array([255, 255, 255], dtype=np.uint8)

    return rgb, legend_entries


def _render_3d_species(pv_mesh, rgb_colors):
    """Render a 3D mesh with per-face RGB colors to a numpy image array."""
    plotter = pv.Plotter(off_screen=True, window_size=[1200, 1000])
    plotter.set_background("white")

    pv_mesh.cell_data["RGB"] = rgb_colors
    plotter.add_mesh(
        pv_mesh, scalars="RGB", rgb=True, show_edges=False,
        lighting=True, smooth_shading=True,
        show_scalar_bar=False,
    )

    pts = pv_mesh.points
    center = pts.mean(axis=0)
    normal = center / np.linalg.norm(center)
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(normal, ref)) > 0.99:
        ref = np.array([0.0, 1.0, 0.0])
    tangent = np.cross(normal, ref)
    tangent /= np.linalg.norm(tangent)

    emission_angle = np.radians(50.0)
    cam_dir = np.cos(emission_angle) * normal + np.sin(emission_angle) * tangent
    cam_dir /= np.linalg.norm(cam_dir)
    mesh_extent = np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))
    cam_dist = mesh_extent * 1.1
    cam_pos = center + cam_dist * cam_dir
    view_up = normal - np.dot(normal, cam_dir) * cam_dir
    view_up /= np.linalg.norm(view_up)
    plotter.camera_position = [tuple(cam_pos), tuple(center), tuple(view_up)]
    plotter.camera.zoom(0.81)

    plotter.remove_all_lights()
    plotter.enable_lightkit()

    img = plotter.screenshot(transparent_background=False, return_img=True)
    plotter.close()
    return img


def plot_combined_panel(all_meshes, all_T_max, dem_labels, dem_keys,
                        stability_temps, save_path, latitude=87):
    """Combined figure: 5 species-stability 3D panels + legend + log histogram.

    Layout: top row = 5 3D panels (a-e) colored by most volatile stable species,
    with species legend on right; bottom row = log-scale stair histogram.
    """
    from matplotlib.patches import Patch
    from PIL import Image

    # Global T_max range across all DEMs
    global_vmin = min(t.min() for t in all_T_max.values())
    global_vmax = max(t.max() for t in all_T_max.values())

    # Render each 3D mesh with species stability coloring
    panel_imgs = []
    panel_labels = []
    legend_entries = None
    letters = "abcde"
    for i, key in enumerate(dem_keys):
        if key not in all_meshes:
            continue
        pv_mesh = all_meshes[key]
        T_max = all_T_max[key]
        rgb, entries = _classify_species_colors(T_max, stability_temps)
        if legend_entries is None:
            legend_entries = entries
        img = _render_3d_species(pv_mesh, rgb)
        img = _crop_content(img)
        panel_imgs.append(img)
        panel_labels.append(f"({letters[i]}) {dem_labels[key]}")

    if not panel_imgs:
        return

    n_panels = len(panel_imgs)

    # Resize all panels to same height
    target_h = max(img.shape[0] for img in panel_imgs)
    resized = []
    for img in panel_imgs:
        if img.shape[0] != target_h:
            pil_img = Image.fromarray(img)
            scale = target_h / img.shape[0]
            new_w = int(img.shape[1] * scale)
            pil_img = pil_img.resize((new_w, target_h), Image.LANCZOS)
            resized.append(np.array(pil_img))
        else:
            resized.append(img)
    panel_imgs = resized

    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import FancyArrowPatch

    total_panel_w = sum(img.shape[1] for img in panel_imgs)
    panel_h = panel_imgs[0].shape[0]
    avg_panel_w = total_panel_w / n_panels

    fig_w = 18
    panel_row_h = fig_w * (panel_h / total_panel_w) * 0.85
    hist_h = 3.5
    fig_h = panel_row_h + hist_h + 0.8

    # Columns: [north arrow, panel1..panel5, legend]
    # Use pixel-proportional ratios so legend gets real space
    arrow_w = avg_panel_w * 0.08
    legend_w = avg_panel_w * 0.55
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = GridSpec(2, n_panels + 2, figure=fig,
                  width_ratios=([arrow_w]
                                + [img.shape[1] for img in panel_imgs]
                                + [legend_w]),
                  height_ratios=[panel_row_h, hist_h],
                  hspace=0.15, wspace=0.02,
                  left=0.01, right=0.99, top=0.97, bottom=0.06)

    # North arrow in leftmost column
    ax_arrow = fig.add_subplot(gs[0, 0])
    ax_arrow.axis("off")
    ax_arrow.set_xlim(0, 1)
    ax_arrow.set_ylim(0, 1)
    # Arrow pointing up (toward pole)
    ax_arrow.annotate("", xy=(0.5, 0.72), xytext=(0.5, 0.35),
                      arrowprops=dict(arrowstyle="-|>", color="black",
                                      lw=2.0, mutation_scale=18))
    ax_arrow.text(0.5, 0.78, "N", ha="center", va="bottom",
                  fontsize=14, fontweight="bold")
    ax_arrow.text(0.5, 0.28, "To\npole", ha="center", va="top",
                  fontsize=8, color="gray")

    # Top row: 3D panels (shifted by 1 column for arrow)
    for i, (img, label) in enumerate(zip(panel_imgs, panel_labels)):
        ax = fig.add_subplot(gs[0, i + 1])
        ax.imshow(img)
        ax.set_title(label, fontsize=11, fontweight="bold", pad=4)
        ax.axis("off")

    # Species legend in rightmost column
    ax_leg = fig.add_subplot(gs[0, n_panels + 1])
    ax_leg.axis("off")
    if legend_entries:
        handles = []
        for name, color_rgb, frac in legend_entries:
            handles.append(Patch(facecolor=color_rgb, edgecolor="black",
                                 linewidth=0.5,
                                 label=f"{name} ({frac:.1f}%)"))
        ax_leg.legend(handles=handles, loc="center", fontsize=10,
                      frameon=True, framealpha=0.9, edgecolor="gray",
                      title="Stable species", title_fontsize=11,
                      handlelength=1.5, handleheight=1.2)

    # Bottom row: combined log-scale stair histogram
    ax_hist = fig.add_subplot(gs[1, :])

    bins = np.linspace(global_vmin, global_vmax, 81)
    bin_width = bins[1] - bins[0]

    combined = np.zeros(len(bins) - 1)
    n_dems = 0
    for key in dem_keys:
        if key not in all_T_max:
            continue
        T_max = all_T_max[key]
        counts, _ = np.histogram(T_max, bins=bins)
        combined += counts / (len(T_max) * bin_width)
        n_dems += 1
    combined /= n_dems

    ax_hist.stairs(combined, bins, fill=True, alpha=0.7,
                   color="steelblue", edgecolor="navy", linewidth=0.5)

    for species, T_stab in stability_temps.items():
        if np.isnan(T_stab):
            continue
        sp = SPECIES[species]
        if T_stab > 10 and T_stab < global_vmax * 1.3:
            sp_label = SPECIES_LABELS.get(species, species)
            ax_hist.axvline(T_stab, color=sp["color"], linestyle="--",
                            linewidth=2.0, label=f"{sp_label}: {T_stab:.0f} K")

    ax_hist.set_yscale("log")
    ax_hist.set_xlabel(r"$T_{\mathrm{max}}$ (K)", fontsize=12)
    ax_hist.set_ylabel("Probability Density", fontsize=12)
    ax_hist.set_title(f"(f) Combined $T_{{\\mathrm{{max}}}}$ Distribution at {latitude}°N",
                      fontsize=12, fontweight="bold")
    ax_hist.legend(fontsize=8, ncol=3, loc="upper right")
    ax_hist.tick_params(labelsize=10)

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ──────────────────────────────────────────────────────────────────────
# DEM configurations
# ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEM_DIR = "/Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/europa/DEM/sutton_WorkMORE"

DEM_BASES = {
    "rhad":   {"mesh": os.path.join(DEM_DIR, "Rhadamanthys_DEM_450m_SOCET.ply"),
               "label": "Rhadamanthys"},
    "ridges": {"mesh": os.path.join(DEM_DIR, "Ridges_DEM_60m_SOCET.ply"),
               "label": "Ridges"},
    "agenor": {"mesh": os.path.join(DEM_DIR, "Agenor_DEM_750m_SOCET.ply"),
               "label": "Agenor Linea"},
    "cilix":  {"mesh": os.path.join(DEM_DIR, "Cilix_DEM_350m_SOCET.ply"),
               "label": "Cilix"},
    "pwyll":  {"mesh": os.path.join(DEM_DIR, "Pwyll_DEM_750m_SOCET.ply"),
               "label": "Pwyll"},
}


def get_dem_configs(latitude):
    """Build DEM_CONFIGS dict for a given latitude."""
    configs = {}
    for key, base in DEM_BASES.items():
        if latitude == 87:
            output_dir = os.path.join(BASE_DIR, f"output/europa_polar_{key}")
        else:
            output_dir = os.path.join(BASE_DIR,
                                       f"output/europa_lat{int(latitude):02d}_{key}")
        configs[key] = {
            "mesh": base["mesh"],
            "output": output_dir,
            "label": base["label"],
        }
    return configs


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def process_latitude(latitude, plot_dir, loss_rate, dem_keys=None):
    """Process all DEMs at a given latitude: load data, make plots, return fractions."""
    DEM_CONFIGS = get_dem_configs(latitude)
    if dem_keys is None:
        dem_keys = list(DEM_CONFIGS.keys())

    os.makedirs(plot_dir, exist_ok=True)

    # Compute stability temperatures
    stability_temps = {}
    for species in SPECIES:
        stability_temps[species] = stability_temperature(species, loss_rate)

    all_T_max = {}
    all_meshes = {}
    dem_labels = {}
    cold_trap_table = []

    for key in dem_keys:
        cfg = DEM_CONFIGS[key]
        dem_label = cfg["label"]
        output_dir = cfg["output"]
        dem_labels[key] = dem_label

        print(f"Processing {dem_label} at {latitude}°N...")

        stat_dir = os.path.join(output_dir, "statistics")
        if not os.path.isdir(stat_dir):
            print(f"  WARNING: {stat_dir} not found, skipping.")
            continue

        mesh_path = os.path.join(output_dir, "mesh.ply")
        if not os.path.exists(mesh_path):
            mesh_path = cfg["mesh"]
        vertices, faces = load_ply_mesh(mesh_path)
        n_faces = len(faces)
        print(f"  Mesh: {len(vertices)} vertices, {n_faces} faces")

        T_max = load_binary_stat(os.path.join(stat_dir, "temperature_max.bin"),
                                 n_faces)
        T_min = load_binary_stat(os.path.join(stat_dir, "temperature_min.bin"),
                                 n_faces)
        T_mean = load_binary_stat(os.path.join(stat_dir, "temperature_avg.bin"),
                                  n_faces)

        print(f"  T_max: {T_max.min():.1f} - {T_max.max():.1f} K "
              f"(mean {T_max.mean():.1f} K)")

        all_T_max[key] = T_max

        row = {"DEM": dem_label, "N_faces": n_faces,
               "N_resolved_faces": int(np.count_nonzero(np.isfinite(T_max)))}
        for species, T_stab in stability_temps.items():
            if not np.isnan(T_stab):
                frac = _frac_below(T_max, T_stab) * 100
                row[species] = frac
            else:
                row[species] = 0.0
        cold_trap_table.append(row)

        pv_mesh = make_pv_mesh(vertices, faces)
        all_meshes[key] = pv_mesh

        plot_3d_pyvista(
            pv_mesh, T_max,
            vmin=T_max.min(), vmax=T_max.max(), cmap="inferno",
            save_path=os.path.join(plot_dir, f"3d_Tmax_{key}.png"),
            scalar_label=r"T_max (K)",
        )
        plot_3d_pyvista(
            pv_mesh, T_min,
            vmin=T_min.min(), vmax=T_min.max(), cmap="inferno",
            save_path=os.path.join(plot_dir, f"3d_Tmin_{key}.png"),
            scalar_label=r"T_min (K)",
        )
        plot_cold_trap_histogram(
            T_max, T_min, T_mean, dem_label, stability_temps,
            save_path=os.path.join(plot_dir, f"histogram_{key}.png"),
        )
        print()

    if len(all_T_max) > 1:
        plot_combined_histogram(
            all_T_max, dem_labels, stability_temps,
            save_path=os.path.join(plot_dir, "combined_Tmax_histogram.png"),
        )
        plot_combined_panel(
            all_meshes, all_T_max, dem_labels, dem_keys, stability_temps,
            save_path=os.path.join(plot_dir, "combined_panel_Tmax.png"),
            latitude=latitude,
        )

    # Print and save table
    if cold_trap_table:
        species_names = list(SPECIES.keys())
        header = f"{'DEM':<25s} {'Faces':>8s}"
        for sp in species_names:
            T_s = stability_temps[sp]
            header += f" {sp+f'({T_s:.0f}K)':>14s}"
        print(header)
        print("-" * len(header))
        for row in cold_trap_table:
            line = f"{row['DEM']:<25s} {row['N_faces']:>8d}"
            for sp in species_names:
                line += f" {row[sp]:>13.1f}%"
            print(line)
        print()

        table_path = os.path.join(plot_dir, "cold_trap_fractions.txt")
        with open(table_path, "w") as f:
            f.write(f"# Cold Trap Fractions at {latitude}°N latitude\n")
            f.write(f"# Sublimation threshold: {loss_rate:.1f} mm/Gyr\n")
            f.write(f"# Stability temperatures: ")
            f.write(", ".join(f"{sp}={stability_temps[sp]:.1f}K"
                              for sp in species_names))
            f.write("\n#\n")
            # Population disclosure (Paul's ruling, 2026-09-15). Written as
            # comment lines, which _load_cold_trap_table() already skips, so
            # the fixed-width table round-trips exactly as before.
            f.write("# Fractions are over the RESOLVED faces only -- a face with "
                    "no value (non-finite)\n# leaves BOTH the numerator and the "
                    "denominator:\n")
            for row in cold_trap_table:
                f.write(f"#   {row['DEM']}: {row['N_resolved_faces']} of "
                        f"{row['N_faces']} faces resolved\n")
            f.write("#\n")
            f.write(header + "\n")
            for row in cold_trap_table:
                line = f"{row['DEM']:<25s} {row['N_faces']:>8d}"
                for sp in species_names:
                    line += f" {row[sp]:>13.1f}%"
                f.write(line + "\n")

    return cold_trap_table, stability_temps


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
                # Parse species names from header
                parts = line.split()
                header_cols = []
                for p in parts[2:]:  # skip DEM and Faces
                    sp = p.split("(")[0]
                    header_cols.append(sp)
                continue
            if line.startswith("-"):
                continue
            if header_cols is None:
                continue
            # Parse data row: "DEM_name   N_faces   pct%  pct%  ..."
            # DEM name may have spaces, so parse from the right
            parts = line.split()
            # Find the numeric columns from the end
            n_species = len(header_cols)
            val_parts = parts[-(n_species):]
            faces_part = parts[-(n_species + 1)]
            dem_name = " ".join(parts[:-(n_species + 1)])
            row = {"DEM": dem_name, "N_faces": int(faces_part)}
            for sp, val in zip(header_cols, val_parts):
                row[sp] = float(val.replace("%", ""))
            rows.append(row)
    return rows


def plot_stability_vs_latitude(all_results, stability_temps, save_path):
    """Plot fractional cold trap area vs. latitude for each species.

    all_results: dict  {latitude: [cold_trap_table_rows]}
    Publication-quality figure; excludes O2 and H2 (zero everywhere).
    Saves both PDF and PNG.
    """
    # Publication style
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'mathtext.fontset': 'cm',
    })

    # Species to plot (exclude O2 and H2 — zero at all latitudes)
    plot_species = ["H2O", "CO2", "H2S", "CH4"]

    # Colorblind-safe, perceptually uniform colors (Wong 2011 palette)
    species_colors = {
        "H2O": "#0072B2",   # blue
        "CO2": "#E69F00",   # orange
        "H2S": "#D55E00",   # vermillion
        "CH4": "#009E73",   # bluish green
    }
    species_markers = {
        "H2O": "o", "CO2": "s", "H2S": "D", "CH4": "^",
    }
    species_labels = {
        "H2O": r"$\mathrm{H_2O}$",
        "CO2": r"$\mathrm{CO_2}$",
        "H2S": r"$\mathrm{H_2S}$",
        "CH4": r"$\mathrm{CH_4}$",
    }

    latitudes = sorted(all_results.keys())

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for species in plot_species:
        T_stab = stability_temps.get(species, float("nan"))
        if np.isnan(T_stab):
            continue
        means = []
        stds = []
        for lat in latitudes:
            fracs = [row[species] for row in all_results[lat]
                     if species in row]
            means.append(np.mean(fracs))
            stds.append(np.std(fracs))
        means = np.array(means)
        stds = np.array(stds)

        color = species_colors[species]
        marker = species_markers[species]
        ax.plot(latitudes, means, marker=marker, linestyle="-",
                color=color, linewidth=2.0, markersize=7, markeredgecolor="white",
                markeredgewidth=0.8, zorder=5,
                label=f"{species_labels[species]} ({T_stab:.0f} K)")
        ax.fill_between(latitudes, np.clip(means - stds, 0, 100),
                         np.clip(means + stds, 0, 100),
                         alpha=0.15, color=color, linewidth=0)

    ax.set_xlabel("Latitude (°N)")
    ax.set_ylabel("Cold Trap Fraction (%)")
    ax.legend(loc="upper left", frameon=True, framealpha=0.95,
              edgecolor="0.8", fancybox=False)
    ax.set_xlim(-2, 90)
    ax.set_ylim(-2, 105)
    ax.set_xticks([0, 15, 30, 45, 60, 75, 87])
    ax.tick_params(direction="in", top=True, right=True)
    ax.grid(True, alpha=0.25, linewidth=0.5)

    # Minor ticks
    ax.minorticks_on()
    ax.tick_params(which="minor", direction="in", top=True, right=True,
                   length=3)

    fig.tight_layout()

    # Save PNG and PDF
    base, ext = os.path.splitext(save_path)
    png_path = base + ".png"
    pdf_path = base + ".pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {png_path}")
    print(f"  Saved: {pdf_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot polar cold trap analysis")
    parser.add_argument("--plot-dir", default="plots/polar_cold_traps",
                        help="Output directory for plots")
    parser.add_argument("--loss-rate", type=float, default=1.0,
                        help="Sublimation loss rate threshold [mm/Gyr]")
    parser.add_argument("--dems", nargs="*", default=None,
                        help="Subset of DEMs to plot (default: all)")
    parser.add_argument("--latitude", type=float, default=87.0,
                        help="Latitude in degrees (default: 87)")
    parser.add_argument("--synthesis", action="store_true",
                        help="Run synthesis across all latitudes")
    parser.add_argument("--synthesis-only", action="store_true",
                        help="Only regenerate synthesis plot from existing data")
    args = parser.parse_args()

    # Compute stability temperatures once
    print("Volatile stability temperatures (sublimation = "
          f"{args.loss_rate:.1f} mm/Gyr):")
    stability_temps = {}
    for species in SPECIES:
        T_stab = stability_temperature(species, args.loss_rate)
        stability_temps[species] = T_stab
        print(f"  {species:>4s}: {T_stab:6.1f} K")
    print()

    if args.synthesis_only:
        # Just regenerate synthesis plot from existing cold_trap_fractions.txt
        all_latitudes = [0, 15, 30, 45, 60, 75, 80, 85, 87]
        all_results = {}
        for lat in all_latitudes:
            if lat == 87:
                lat_dir = os.path.join(args.plot_dir, "lat87")
            else:
                lat_dir = os.path.join(args.plot_dir, f"lat{int(lat):02d}")
            table = _load_cold_trap_table(lat_dir)
            if table:
                all_results[lat] = table
                print(f"  Loaded {lat}°N: {len(table)} DEMs")
        if all_results:
            os.makedirs(args.plot_dir, exist_ok=True)
            plot_stability_vs_latitude(
                all_results, stability_temps,
                save_path=os.path.join(args.plot_dir,
                                        "stability_vs_latitude.png"),
            )
        print(f"\nSynthesis plot saved to: {args.plot_dir}")

    elif args.synthesis:
        # Process all latitudes and produce synthesis plot
        all_latitudes = [0, 15, 30, 45, 60, 75, 80, 85, 87]
        all_results = {}

        for lat in all_latitudes:
            if lat == 87:
                lat_dir = os.path.join(args.plot_dir, "lat87")
            else:
                lat_dir = os.path.join(args.plot_dir, f"lat{int(lat):02d}")
            print(f"\n{'='*60}")
            print(f"  Latitude {lat}°N")
            print(f"{'='*60}")
            table, _ = process_latitude(lat, lat_dir, args.loss_rate,
                                         args.dems)
            if table:
                all_results[lat] = table

        # Synthesis plot
        if all_results:
            os.makedirs(args.plot_dir, exist_ok=True)
            plot_stability_vs_latitude(
                all_results, stability_temps,
                save_path=os.path.join(args.plot_dir,
                                        "stability_vs_latitude.png"),
            )
        print(f"\nAll plots saved under: {args.plot_dir}")
    else:
        # Single latitude mode
        table, _ = process_latitude(args.latitude, args.plot_dir,
                                     args.loss_rate, args.dems)
        print(f"\nAll plots saved to: {args.plot_dir}")


if __name__ == "__main__":
    main()
