#!/usr/bin/env python3
"""Plot comparison of baseline vs heated-trough ridge simulations.

Generates:
1. 3D perspective views of daytime and nighttime temperatures (baseline + heated)
2. Diurnal temperature curves for trough floor vs background terrain
3. Temperature difference maps (heated - baseline)
"""

import struct
import sys
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pyvista as pv


def _frame_file(directory, name, t):
    """Frame ``t`` of ``name`` under either naming (BUGS.md PTM-664).

    Current engines pad the index to >= 6 digits; runs written before used 4.
    """
    for w in range(6, 13):
        p = os.path.join(str(directory), "%s_%0*d.bin" % (name, w, t))
        if os.path.exists(p):
            return p
    return os.path.join(str(directory), "%s_%04d.bin" % (name, t))



# ──────────────────────────────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────────────────────────────
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
        face_lat = np.empty(n_faces, dtype=np.float32)
        face_lon = np.empty(n_faces, dtype=np.float32)

        for i in range(n_faces):
            offset = i * face_record_size
            v0, v1, v2 = struct.unpack_from("<iii", face_data, offset + 1)
            lat, lon = struct.unpack_from("<ff", face_data, offset + 13)
            faces[i] = [v0, v1, v2]
            face_lat[i] = lat
            face_lon[i] = lon

    return vertices, faces, face_lat, face_lon


def load_binary_timestep(filepath, n_faces):
    """Load a single binary timestep file (int32 header + float32 per face)."""
    data = np.fromfile(filepath, dtype=np.float32)
    # First 4 bytes are int32 face count; skip the first float32 slot
    if len(data) == n_faces + 1:
        data = data[1:]
    elif len(data) != n_faces:
        raise ValueError(f"Expected {n_faces}(+1) values, got {len(data)}")
    return data


def load_all_temperatures(output_dir, n_faces, n_timesteps):
    """Load all temperature timesteps. Returns (n_faces, n_timesteps) array."""
    temps = np.empty((n_faces, n_timesteps), dtype=np.float32)
    for t in range(n_timesteps):
        fpath = _frame_file(os.path.join(output_dir, "temperature"), "temperature", t)
        temps[:, t] = load_binary_timestep(fpath, n_faces)
    return temps


def load_trough_indices(filepath):
    """Load trough face indices (and optional Qb values) from text file.

    Supports both formats:
        face_index
        face_index qb_value
    """
    indices = []
    qb_values = {}
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            idx = int(parts[0])
            indices.append(idx)
            if len(parts) >= 2:
                qb_values[idx] = float(parts[1])
    return np.array(indices, dtype=int), qb_values


def load_time_steps(filepath):
    """Load time steps from time_steps.txt (elapsed_seconds, utc_time)."""
    times = []
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                try:
                    times.append(float(parts[0]))
                except ValueError:
                    continue
    return np.array(times)


# ──────────────────────────────────────────────────────────────────────
# Plotting functions
# ──────────────────────────────────────────────────────────────────────
def compute_face_elevations(vertices, faces):
    """Per-face elevation relative to median radius."""
    r = np.linalg.norm(vertices, axis=1)
    r_ref = np.median(r)
    vertex_elev = r - r_ref
    face_elev = (vertex_elev[faces[:, 0]] +
                 vertex_elev[faces[:, 1]] +
                 vertex_elev[faces[:, 2]]) / 3.0
    return face_elev


def make_pv_mesh(vertices, faces):
    """Create a PyVista PolyData mesh from vertices and face indices."""
    n_faces = len(faces)
    pv_faces = np.column_stack([
        np.full(n_faces, 3, dtype=np.int64),
        faces.astype(np.int64)
    ]).ravel()
    return pv.PolyData(vertices, pv_faces)


def _autocrop(path, pad=20):
    """Crop whitespace from a screenshot, keeping *pad* px margin."""
    from PIL import Image
    img = Image.open(path).convert("RGBA")
    # Use alpha channel: non-transparent pixels are content
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


def plot_3d_pyvista(pv_mesh, scalars, trough_mask, title, vmin, vmax,
                     cmap, save_path, scalar_label="Temperature (K)"):
    """Render a 3D mesh with PyVista (off-screen) for publication.

    Camera is placed at 45° emission angle from the surface normal at
    the mesh center.  Transparent background, black text, compact layout.
    Auto-crops whitespace after rendering.
    """
    plotter = pv.Plotter(off_screen=True, window_size=[2400, 1600])
    plotter.set_background("white")

    pv_mesh[scalar_label] = scalars
    plotter.add_mesh(
        pv_mesh, scalars=scalar_label, cmap=cmap, clim=[vmin, vmax],
        show_edges=False, lighting=True, smooth_shading=True,
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

    # --- Camera at 45° emission angle from surface normal at mesh center ---
    pts = np.array(pv_mesh.points)
    center = pts.mean(axis=0)
    normal = center / np.linalg.norm(center)
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(normal, ref)) > 0.9:
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

    # --- Lighting ---
    plotter.remove_all_lights()
    plotter.enable_lightkit()

    plotter.screenshot(save_path, transparent_background=True)
    plotter.close()
    _autocrop(save_path)
    print(f"  Saved: {save_path}")


def plot_2d_temperature_map(face_lat, face_lon, face_temps, trough_mask,
                            title, vmin, vmax, cmap, save_path):
    """2D lat/lon scatter map of face temperatures."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    sc = ax.scatter(face_lon, face_lat, c=face_temps, cmap=cmap,
                    s=2, vmin=vmin, vmax=vmax, edgecolors="none", rasterized=True)
    ax.scatter(face_lon[trough_mask], face_lat[trough_mask],
               s=4, facecolors="none", edgecolors="white", linewidths=0.3, alpha=0.5)

    ax.set_xlabel("Longitude (deg)", fontsize=11)
    ax.set_ylabel("Latitude (deg)", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.set_aspect("equal")
    cbar = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Temperature (K)", fontsize=11)

    fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_difference_map(face_lat, face_lon, delta_T, trough_mask,
                        title, save_path):
    """2D lat/lon map of temperature difference (relative to flat surface).

    Uses a diverging colormap centered on zero so positive (warmer than flat)
    and negative (cooler than flat) deviations are both visible.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))

    vlim = max(abs(np.percentile(delta_T, 0.5)),
               abs(np.percentile(delta_T, 99.5)), 1.0)

    sc = ax.scatter(face_lon, face_lat, c=delta_T, cmap="RdBu_r",
                    s=2, vmin=-vlim, vmax=vlim, edgecolors="none",
                    rasterized=True)

    ax.set_xlabel("Longitude (deg)", fontsize=11)
    ax.set_ylabel("Latitude (deg)", fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal")
    cbar = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("$\\Delta T$ (K)", fontsize=11)

    fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_diurnal_curves(times_hours, temps_baseline, temps_heated,
                        trough_mask, save_path, qb_label="10"):
    """Diurnal temperature curves: trough floor vs background."""
    bg_mask = ~trough_mask
    n_trough = trough_mask.sum()
    n_bg = bg_mask.sum()

    # Compute mean, min, max for each group
    trough_mean_b = temps_baseline[trough_mask].mean(axis=0)
    trough_mean_h = temps_heated[trough_mask].mean(axis=0)
    bg_mean_b = temps_baseline[bg_mask].mean(axis=0)
    bg_mean_h = temps_heated[bg_mask].mean(axis=0)

    trough_min_h = temps_heated[trough_mask].min(axis=0)
    trough_max_h = temps_heated[trough_mask].max(axis=0)

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    # Panel 1: Absolute temperatures
    ax = axes[0]
    ax.plot(times_hours, bg_mean_b, "b-", linewidth=1.5, label=f"Background ({n_bg} faces) — baseline")
    ax.plot(times_hours, bg_mean_h, "b--", linewidth=1.2, label="Background — heated sim", alpha=0.7)
    ax.plot(times_hours, trough_mean_b, "r-", linewidth=1.5, label=f"Trough floor ({n_trough} faces) — baseline")
    ax.plot(times_hours, trough_mean_h, "r--", linewidth=1.5, label=f"Trough floor — heated (Qb={qb_label} W/m²)")
    ax.fill_between(times_hours, trough_min_h, trough_max_h,
                     color="red", alpha=0.15, label="Trough range (heated)")
    ax.set_ylabel("Temperature (K)", fontsize=11)
    ax.set_title("Diurnal Temperature: Trough Floor vs Background", fontsize=12)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    # Panel 2: Temperature difference (heated - baseline)
    ax = axes[1]
    delta_trough = trough_mean_h - trough_mean_b
    delta_bg = bg_mean_h - bg_mean_b
    ax.plot(times_hours, delta_trough, "r-", linewidth=1.5,
            label="Trough floor $\\Delta T$")
    ax.plot(times_hours, delta_bg, "b-", linewidth=1.2,
            label="Background $\\Delta T$", alpha=0.7)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("Time (hours from start)", fontsize=11)
    ax.set_ylabel("$\\Delta T$ (heated $-$ baseline) (K)", fontsize=11)
    ax.set_title("Temperature Enhancement from Elevated Heat Flow", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout(h_pad=1.0)
    fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_temperature_histogram(temps_b_day, temps_h_day, temps_b_night, temps_h_night,
                               trough_mask, save_path):
    """Histogram comparison of temperature distributions."""
    bg = ~trough_mask

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    ax = axes[0]
    ax.hist(temps_b_day[bg], bins=80, alpha=0.5, color="blue", label="Background — baseline", density=True)
    ax.hist(temps_h_day[trough_mask], bins=40, alpha=0.7, color="red", label="Trough — heated", density=True)
    ax.hist(temps_b_day[trough_mask], bins=40, alpha=0.4, color="orange", label="Trough — baseline", density=True)
    ax.set_xlabel("Temperature (K)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("Daytime", fontsize=11)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.hist(temps_b_night[bg], bins=80, alpha=0.5, color="blue", label="Background — baseline", density=True)
    ax.hist(temps_h_night[trough_mask], bins=40, alpha=0.7, color="red", label="Trough — heated", density=True)
    ax.hist(temps_b_night[trough_mask], bins=40, alpha=0.4, color="orange", label="Trough — baseline", density=True)
    ax.set_xlabel("Temperature (K)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("Nighttime", fontsize=11)
    ax.legend(fontsize=8)

    fig.suptitle("Temperature Distributions: Baseline vs Heated Trough", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Plot ridge heat flow comparison")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--mesh", default=os.path.join(base_dir, "output/europa_work-more_rhad/mesh.ply"))
    parser.add_argument("--trough", default=os.path.join(base_dir, "output/europa_work-more_rhad/trough_faces.txt"))
    parser.add_argument("--baseline", default=os.path.join(base_dir, "output/rhad_baseline"))
    parser.add_argument("--heated", default=os.path.join(base_dir, "output/rhad_heated"))
    parser.add_argument("--plot-dir", default=os.path.join(base_dir, "plots/rhad_heat_flow_10W"))
    parser.add_argument("--n-timesteps", type=int, default=144)
    args = parser.parse_args()

    mesh_path = args.mesh
    trough_path = args.trough
    baseline_dir = args.baseline
    heated_dir = args.heated
    plot_dir = args.plot_dir
    os.makedirs(plot_dir, exist_ok=True)

    n_timesteps = args.n_timesteps

    # Load mesh
    print("Loading mesh...")
    vertices, faces, face_lat, face_lon = load_ply_mesh(mesh_path)
    n_faces = len(faces)
    face_elev = compute_face_elevations(vertices, faces)

    # Load trough mask and determine peak Qb for labels
    trough_indices, trough_qb = load_trough_indices(trough_path)
    trough_mask = np.zeros(n_faces, dtype=bool)
    trough_mask[trough_indices] = True
    if trough_qb:
        qb_vals = list(trough_qb.values())
        qb_peak = max(qb_vals)
        print(f"  Trough faces: {trough_mask.sum()} "
              f"(Qb range: {min(qb_vals):.3f} - {qb_peak:.3f} W/m^2)")
    else:
        qb_peak = 10.0  # fallback for old-format files
        print(f"  Trough faces: {trough_mask.sum()}")
    # Compact label for peak Qb
    if qb_peak >= 1.0:
        qb_label = f"{qb_peak:.0f}" if qb_peak == int(qb_peak) else f"{qb_peak:.1f}"
    else:
        qb_label = f"{qb_peak:g}"

    # Load temperatures
    print("Loading baseline temperatures...")
    temps_b = load_all_temperatures(baseline_dir, n_faces, n_timesteps)
    print("Loading heated temperatures...")
    temps_h = load_all_temperatures(heated_dir, n_faces, n_timesteps)

    # Load times
    times_path = os.path.join(baseline_dir, "time_steps.txt")
    times = load_time_steps(times_path)
    times_hours = (times - times[0]) / 3600.0

    # Identify daytime and nighttime timesteps
    # Daytime: when mean illumination is high; use direct_solar to find
    # Simple: find timestep with max and min mean temperature
    mean_T = temps_b.mean(axis=0)
    t_day = np.argmax(mean_T)    # Peak daytime
    t_night = np.argmin(mean_T)  # Deepest night
    print(f"  Daytime timestep: {t_day} (mean T = {mean_T[t_day]:.1f} K)")
    print(f"  Nighttime timestep: {t_night} (mean T = {mean_T[t_night]:.1f} K)")

    # Temperature arrays for key timesteps
    T_b_day = temps_b[:, t_day]
    T_h_day = temps_h[:, t_day]
    T_b_night = temps_b[:, t_night]
    T_h_night = temps_h[:, t_night]

    # Compute differences relative to topographic baseline
    dT_day = T_h_day - T_b_day
    dT_night = T_h_night - T_b_night

    # Flat-surface reference: spatial mean of the baseline at each timestep.
    # Deviation from flat shows topographic temperature variations alongside
    # the heat flow anomaly.
    T_flat_day = T_b_day.mean()
    T_flat_night = T_b_night.mean()
    dT_flat_day = T_h_day - T_flat_day
    dT_flat_night = T_h_night - T_flat_night

    # Print summary statistics
    print(f"\n--- Temperature Summary ---")
    print(f"Daytime (timestep {t_day}):")
    print(f"  Baseline  — trough: {T_b_day[trough_mask].mean():.1f} K, bg: {T_b_day[~trough_mask].mean():.1f} K")
    print(f"  Heated    — trough: {T_h_day[trough_mask].mean():.1f} K, bg: {T_h_day[~trough_mask].mean():.1f} K")
    print(f"  Delta T   — trough: {dT_day[trough_mask].mean():.1f} K, bg: {dT_day[~trough_mask].mean():.2f} K")
    print(f"  Flat ref  — T_flat = {T_flat_day:.1f} K")
    print(f"  Topo spread (baseline): {T_b_day.std():.2f} K (1-sigma)")
    print(f"Nighttime (timestep {t_night}):")
    print(f"  Baseline  — trough: {T_b_night[trough_mask].mean():.1f} K, bg: {T_b_night[~trough_mask].mean():.1f} K")
    print(f"  Heated    — trough: {T_h_night[trough_mask].mean():.1f} K, bg: {T_h_night[~trough_mask].mean():.1f} K")
    print(f"  Delta T   — trough: {dT_night[trough_mask].mean():.1f} K, bg: {dT_night[~trough_mask].mean():.2f} K")
    print(f"  Flat ref  — T_flat = {T_flat_night:.1f} K")
    print(f"  Topo spread (baseline): {T_b_night.std():.2f} K (1-sigma)")

    # ────────────── Generate plots ──────────────
    print("\n--- Generating plots ---")

    # 1. 2D temperature maps (daytime)
    T_all_min = min(T_b_day.min(), T_h_day.min())
    T_all_max = max(T_b_day.max(), T_h_day.max())
    plot_2d_temperature_map(face_lat, face_lon, T_b_day, trough_mask,
                            f"Baseline Daytime (t={t_day}, no enhanced Qb)",
                            T_all_min, T_all_max, "inferno",
                            os.path.join(plot_dir, "temp_map_day_baseline.png"))
    plot_2d_temperature_map(face_lat, face_lon, T_h_day, trough_mask,
                            f"Heated Trough Daytime (t={t_day}, Qb={qb_label} W/m²)",
                            T_all_min, T_all_max, "inferno",
                            os.path.join(plot_dir, "temp_map_day_heated.png"))

    # 2. 2D temperature maps (nighttime)
    T_night_min = min(T_b_night.min(), T_h_night.min())
    T_night_max = max(T_b_night.max(), T_h_night.max())
    plot_2d_temperature_map(face_lat, face_lon, T_b_night, trough_mask,
                            f"Baseline Nighttime (t={t_night})",
                            T_night_min, T_night_max, "inferno",
                            os.path.join(plot_dir, "temp_map_night_baseline.png"))
    plot_2d_temperature_map(face_lat, face_lon, T_h_night, trough_mask,
                            f"Heated Trough Nighttime (t={t_night}, Qb={qb_label} W/m²)",
                            T_night_min, T_night_max, "inferno",
                            os.path.join(plot_dir, "temp_map_night_heated.png"))

    # 3. Difference maps (relative to flat-surface reference)
    plot_difference_map(face_lat, face_lon, dT_flat_day, trough_mask,
                        f"$\\Delta T$ Daytime (heated $-$ flat surface, "
                        f"$T_{{flat}}$ = {T_flat_day:.1f} K)",
                        os.path.join(plot_dir, "diff_map_day.png"))
    plot_difference_map(face_lat, face_lon, dT_flat_night, trough_mask,
                        f"$\\Delta T$ Nighttime (heated $-$ flat surface, "
                        f"$T_{{flat}}$ = {T_flat_night:.1f} K)",
                        os.path.join(plot_dir, "diff_map_night.png"))

    # 4. Diurnal curves
    plot_diurnal_curves(times_hours, temps_b, temps_h, trough_mask,
                        os.path.join(plot_dir, "diurnal_curves.png"),
                        qb_label=qb_label)

    # 5. Temperature histograms
    plot_temperature_histogram(T_b_day, T_h_day, T_b_night, T_h_night,
                               trough_mask,
                               os.path.join(plot_dir, "temp_histograms.png"))

    # 6. 3D views with PyVista
    print("  Generating PyVista 3D views...")
    pv_mesh = make_pv_mesh(vertices, faces)

    plot_3d_pyvista(pv_mesh, T_h_day, trough_mask,
                     f"Heated Trough - Daytime (Qb={qb_label} W/m^2)",
                     T_all_min, T_all_max, "inferno",
                     os.path.join(plot_dir, "3d_day_heated.png"))
    plot_3d_pyvista(pv_mesh, T_h_night, trough_mask,
                     f"Heated Trough - Nighttime (Qb={qb_label} W/m^2)",
                     T_night_min, T_night_max, "inferno",
                     os.path.join(plot_dir, "3d_night_heated.png"))
    # Symmetric range centered on zero (same convention as 2D diff maps)
    diff_vlim = max(abs(np.percentile(dT_flat_night, 0.5)),
                    abs(np.percentile(dT_flat_night, 99.5)), 1.0)
    plot_3d_pyvista(pv_mesh, dT_flat_night, trough_mask,
                     "Delta T Nighttime (heated - flat surface)",
                     -diff_vlim, diff_vlim, "RdBu_r",
                     os.path.join(plot_dir, "3d_diff_night.png"),
                     scalar_label="Delta T (K)")

    print(f"\nAll plots saved to: {plot_dir}")


if __name__ == "__main__":
    main()
