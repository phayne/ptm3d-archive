#!/usr/bin/env python3
"""Assemble Clipper-EGA FITS frames into GIF + MP4 animations.

Three variants are produced:
  - visible     : reflected+Sun+Earth radiance with percentile + gamma stretch, gray
  - thermal     : surface temperature K, fixed global [t_min, t_max] range, magma
  - composite   : visible | thermal side-by-side

Frame file convention matches ``ptm_camera_sim`` output: ``frame_NNNN.fits`` with
HDU[0]=RADIANCE, HDU[2]=face_index (Sun=-2, Earth=-3), HDU[6]=TEMP_K, HDU[7]=distance.

The stretch/colormap/earthshine logic is lifted from
``gui/gui_panels/camera_panel.py::_export_animation`` so the look matches the
Artemis II pipeline the user is already familiar with.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import shutil
import subprocess

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits as pyfits


def find_frames(frames_dir: Path):
    frames = sorted(frames_dir.glob("frame_*.fits"))
    if not frames:
        sys.exit(f"No frames found in {frames_dir}")
    return frames


def compute_global_vis_max(frames, pctile: float) -> float:
    """First-pass max reflected radiance across all frames."""
    max_ref = 0.0
    for f in frames:
        with pyfits.open(f) as hdul:
            rad = hdul[0].data
            face = hdul[2].data
            mask = (face >= 0) & (rad > 0.1)
            if mask.any():
                mr = np.nanpercentile(rad[mask], pctile)
                if mr > max_ref:
                    max_ref = mr
    return max_ref if max_ref >= 1e-6 else 1.0


def render_visible_axis(ax, rad, face, max_ref, title_extras=True):
    mask = face >= 0
    sun_mask = face == -2
    earth_mask = face == -3
    ny, nx = rad.shape
    vis = np.zeros((ny, nx))
    vis[mask] = rad[mask]
    vis_gamma = np.power(np.clip(vis / max_ref, 0, 1), 1 / 2.2)
    if sun_mask.any():
        vis_gamma[sun_mask] = 1.0
    if earth_mask.any():
        vis_gamma[earth_mask] = np.power(
            np.clip(rad[earth_mask] / max_ref, 0, 1), 1 / 2.2
        )
    ax.imshow(vis_gamma, cmap="gray", vmin=0, vmax=1, origin="upper", aspect="equal")
    title = "Visible"
    if title_extras:
        extras = []
        if sun_mask.any():
            extras.append("Sun")
        if earth_mask.any():
            extras.append("Earth")
        if extras:
            title += "  (" + ", ".join(extras) + ")"
    ax.set_title(title, fontsize=9, color="white")
    ax.axis("off")


def render_thermal_axis(ax, temp, face, t_min, t_max):
    mask = face >= 0
    ny, nx = temp.shape
    t_img = np.full((ny, nx), np.nan)
    t_img[mask] = temp[mask]
    im = ax.imshow(
        t_img, cmap="magma", vmin=t_min, vmax=t_max, origin="upper", aspect="equal"
    )
    ax.set_title(f"Surface Temperature  [{t_min:.0f}-{t_max:.0f} K]", fontsize=9, color="white")
    ax.axis("off")
    return im


R_MOON_KM = 1737.4


def moon_angular_diameter_deg(dist, mask):
    """Moon angular diameter in degrees from the per-pixel surface distances.

    The minimum DIST_KM hit is the nearest surface point, i.e. center_distance - R,
    so center_distance = min_dist + R. For D >> R this is essentially 2R/D radians.
    """
    if dist is None or not mask.any():
        return 0.0
    d_surface_min = float(np.nanmin(dist[mask]))
    d_center = d_surface_min + R_MOON_KM
    if d_center <= R_MOON_KM:
        return 180.0
    return 2.0 * math.degrees(math.asin(R_MOON_KM / d_center))


def add_geometry_label(fig, fov_x, fov_y, moon_diam_deg, earth_diam_deg=None):
    """Lower-left overlay: fixed FOV + time-varying body angular diameters."""
    lines = [
        f"FOV    : {fov_x:.3f}° × {fov_y:.3f}°",
        f"Moon Ø : {moon_diam_deg:.4f}°",
    ]
    if earth_diam_deg is not None:
        lines.append(f"Earth Ø: {earth_diam_deg:.4f}°")
    fig.text(
        0.01, 0.02, "\n".join(lines),
        color="white", fontsize=8, ha="left", va="bottom",
        family="monospace",
    )


def make_frame_figure(variant, rad, face, temp, dist, utc, max_ref, t_min, t_max,
                      dpi, fov_x, fov_y, earth_diam_deg=None):
    if variant == "visible":
        fig, ax = plt.subplots(figsize=(7, 4.66), dpi=dpi)
        render_visible_axis(ax, rad, face, max_ref)
    elif variant == "thermal":
        fig, ax = plt.subplots(figsize=(7, 4.66), dpi=dpi)
        render_thermal_axis(ax, temp, face, t_min, t_max)
    elif variant == "composite":
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.66), dpi=dpi)
        render_visible_axis(ax1, rad, face, max_ref)
        render_thermal_axis(ax2, temp, face, t_min, t_max)
        fig.subplots_adjust(wspace=0.04)
    else:
        raise ValueError(variant)

    mask = face >= 0
    min_dist = float(np.nanmin(dist[mask])) if (dist is not None and mask.any()) else 0.0
    moon_diam = moon_angular_diameter_deg(dist, mask)
    fig.suptitle(f"{utc}  ·  {min_dist:,.0f} km", fontsize=9, color="white", y=0.97)
    add_geometry_label(fig, fov_x, fov_y, moon_diam, earth_diam_deg)
    fig.patch.set_facecolor("black")
    return fig


def load_geometry_json(path: Path):
    """Returns dict mapping utc_prefix (YYYY-MM-DDTHH:MM:SS, no .ms) -> frame entry."""
    data = json.loads(path.read_text())
    out = {}
    for f in data["frames"]:
        key = f["utc"].split(".")[0]
        out[key] = f
    return out


def fig_to_rgb(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    rgba = np.asarray(fig.canvas.buffer_rgba()).reshape((h, w, 4))
    return rgba[:, :, :3].copy()


def write_mp4_via_ffmpeg(images, mp4_path: Path, fps: int, ffmpeg: str):
    h, w, _ = images[0].shape
    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}", "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "18", "-preset", "medium",
        "-movflags", "+faststart",
        str(mp4_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for img in images:
            proc.stdin.write(np.ascontiguousarray(img).tobytes())
        proc.stdin.close()
        rc = proc.wait()
    except BrokenPipeError:
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {rc}")


def write_outputs(images, out_stem: Path, fps: int):
    gif_path = out_stem.with_suffix(".gif")
    mp4_path = out_stem.with_suffix(".mp4")
    print(f"  -> {gif_path}")
    imageio.mimsave(str(gif_path), images, duration=1.0 / fps, loop=0)
    print(f"  -> {mp4_path}")
    ffmpeg = shutil.which("ffmpeg") or "/usr/local/bin/ffmpeg"
    write_mp4_via_ffmpeg(images, mp4_path, fps, ffmpeg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--prefix", default="clipper_ega")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--dpi", type=int, default=100)
    ap.add_argument("--t-min", type=float, default=90.0)
    ap.add_argument("--t-max", type=float, default=390.0)
    ap.add_argument("--vis-pctile", type=float, default=99.5)
    ap.add_argument(
        "--variants",
        nargs="+",
        default=["visible", "thermal", "composite"],
        choices=["visible", "thermal", "composite"],
    )
    ap.add_argument(
        "--geometry-json", type=Path, default=None,
        help="Optional per-frame geometry JSON (from clipper_ega_wideview_fov.py) "
             "that provides Earth angular diameter for the lower-left label.",
    )
    args = ap.parse_args()

    frames = find_frames(args.frames_dir)
    print(f"Found {len(frames)} frames in {args.frames_dir}")

    print(f"Computing global visible max (pctile={args.vis_pctile}) ...")
    max_ref = compute_global_vis_max(frames, args.vis_pctile)
    print(f"  max reflected radiance = {max_ref:.4g}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    geom_by_utc = load_geometry_json(args.geometry_json) if args.geometry_json else {}

    # Collect per-variant image buffers in a single pass over FITS files.
    per_variant: dict[str, list[np.ndarray]] = {v: [] for v in args.variants}
    n = len(frames)
    for i, f in enumerate(frames):
        with pyfits.open(f) as hdul:
            rad = hdul[0].data
            face = hdul[2].data
            temp = hdul[6].data if len(hdul) > 6 else None
            dist = hdul[7].data if len(hdul) > 7 else None
            hdr = hdul[0].header
            utc = hdr.get("UTC_TIME", "").split(".")[0]
            fov_x = float(hdr.get("FOV_X", 0.0))
            fov_y = float(hdr.get("FOV_Y", 0.0))

        earth_diam = None
        if geom_by_utc:
            entry = geom_by_utc.get(utc)
            if entry is not None:
                earth_diam = entry.get("earth_ang_deg")

        for variant in args.variants:
            fig = make_frame_figure(
                variant, rad, face, temp, dist, utc, max_ref,
                args.t_min, args.t_max, args.dpi, fov_x, fov_y,
                earth_diam_deg=earth_diam,
            )
            per_variant[variant].append(fig_to_rgb(fig))
            plt.close(fig)

        if (i + 1) % 20 == 0 or i == n - 1:
            print(f"  Rendered frame {i + 1}/{n}")

    for variant, images in per_variant.items():
        out_stem = args.out_dir / f"{args.prefix}_{variant}"
        print(f"Writing {variant} animation:")
        write_outputs(images, out_stem, args.fps)

    print("Done.")


if __name__ == "__main__":
    main()
