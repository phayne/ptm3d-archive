#!/usr/bin/env python3
"""Step 4: render a 13-frame 180-deg panorama with ptm_camera_sim.

Camera: 640x480, 15.6 x 12.5 deg (anisotropic pixel pitch, focal 50 mm),
1.0 m above the patch core, boresight pitched down VFOV/2 so the horizon
sits at the top of the frame. 13 frames, North-centered, ~10% overlap.

The SAME camera poses are rendered against both the FINE and the COARSE
PTM3D outputs (`--ptm-roots`). The fine render is the "truth" image; the
coarse render supplies, per pixel, where that ray lands on a coarse DEM.
A JSON sidecar per frame records the exact pose/intrinsics for Step 5.

Example:
  python3 src/python/vise_run_panorama.py \
    --bin build/bin/ptm_camera_sim \
    --scene tmp/vise/scene_params.json --epoch tmp/vise/epoch.json \
    --ptm-roots fine=output/visecam_flux_fine coarse=output/visecam_flux_coarse \
    --out-root output
"""

import argparse
import json
import os
import subprocess

import numpy as np

FOCAL_MM = 50.0
PIX_NX, PIX_NY = 640, 480
HFOV_DEG, VFOV_DEG = 15.6, 12.5
N_FRAMES = 13
OVERLAP = 0.10
# Scale-invariant setup. The DEM-resolution error is purely angular/parallax,
# so absolute size is irrelevant as long as the camera distance scales with
# the mesh. The mesh is kept at the numerically safe ~1 m-facet size (cm-scale
# meshes hit an Embree float32 self-shadow cliff -> zero flux). The original
# "1 cm facet / 1 m camera height / 10 m look" intent is reproduced at 100x:
# 1 m facet / 100 m height / 1000 m look. Identical angular footprint
# (facet ~ 1 px), identical projection geometry.
CAM_HEIGHT_M = 100.0
LOOK_DIST_M = 1000.0


def intrinsics():
    px = 2 * FOCAL_MM * np.tan(np.radians(HFOV_DEG / 2)) / PIX_NX
    py = 2 * FOCAL_MM * np.tan(np.radians(VFOV_DEG / 2)) / PIX_NY
    return px, py


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bin", default="build/bin/ptm_camera_sim")
    ap.add_argument("--scene", default="tmp/vise/scene_params.json")
    ap.add_argument("--epoch", default="tmp/vise/epoch.json")
    ap.add_argument("--ptm-roots", nargs="+", required=True,
                    help="label=dir pairs, e.g. fine=output/visecam_flux_fine")
    ap.add_argument("--out-root", default="output")
    ap.add_argument("--albedo", type=float, default=0.12)
    ap.add_argument("--photometric", default="lommel_seeliger")
    ap.add_argument("--render-mode", default="both",
                    choices=["reflected", "thermal", "both"],
                    help="'both' populates THERMAL/TEMP_K/EMI_ANG too.")
    ap.add_argument("--cam-height-m", type=float, default=CAM_HEIGHT_M)
    ap.add_argument("--look-dist-m", type=float, default=LOOK_DIST_M)
    ap.add_argument("--pitch-down-deg", type=float, default=VFOV_DEG / 2.0,
                    help="Boresight depression below horizontal. The "
                         "default VFOV/2 puts a horizontal horizon at the "
                         "FOV top; on this 100 m-high station the real "
                         "skyline is ~-5.4 deg, so 11.68 puts the actual "
                         "skyline at the FOV top at panorama center.")
    args = ap.parse_args()

    scene = json.load(open(args.scene))
    epoch = json.load(open(args.epoch))
    utc = epoch["utc"]

    g = np.array(scene["ground_point_bf_m"])          # meters, body-fixed
    U = np.array(scene["site_U"])
    E = np.array(scene["site_E"])
    N = np.array(scene["site_N"])

    cam_m = g + args.cam_height_m * U
    px, py = intrinsics()
    pitch_down = np.radians(args.pitch_down_deg)       # skyline -> FOV top

    step = HFOV_DEG * (1.0 - OVERLAP)
    az = (np.arange(N_FRAMES) - (N_FRAMES - 1) / 2.0) * step  # N-centered

    roots = dict(p.split("=", 1) for p in args.ptm_roots)
    manifest = dict(utc=utc, cam_pos_m=cam_m.tolist(),
                    up=U.tolist(), focal_mm=FOCAL_MM,
                    pixel_pitch_mm=[px, py], pixels=[PIX_NX, PIX_NY],
                    hfov_deg=HFOV_DEG, vfov_deg=VFOV_DEG,
                    site_U=U.tolist(), site_E=E.tolist(), site_N=N.tolist(),
                    ground_point_bf_m=g.tolist(), frames=[])

    for label, ptm_dir in roots.items():
        out_dir = os.path.join(args.out_root, f"visecam_pano_{label}")
        os.makedirs(out_dir, exist_ok=True)
        for k, a in enumerate(az):
            th = np.radians(a)
            horiz = np.cos(th) * N + np.sin(th) * E
            d = np.cos(pitch_down) * horiz - np.sin(pitch_down) * U
            d /= np.linalg.norm(d)
            look_m = cam_m + args.look_dist_m * d

            fdir = os.path.join(out_dir, f"frame_{k:02d}")
            os.makedirs(fdir, exist_ok=True)
            cmd = [
                args.bin,
                "--ptm-output", ptm_dir,
                "--time", utc,
                "--cam-pos", *[f"{c/1000.0:.9f}" for c in cam_m],
                "--look-at", *[f"{c/1000.0:.9f}" for c in look_m],
                "--up", *[f"{c:.9f}" for c in U],
                "--focal-mm", f"{FOCAL_MM}",
                "--pixel-pitch", f"{px:.9f}", f"{py:.9f}",
                "--pixels", str(PIX_NX), str(PIX_NY),
                "--render-mode", args.render_mode,
                "--photometric", args.photometric,
                "--albedo", f"{args.albedo}",
                "--output-dir", fdir,
            ]
            print(f"[{label}] frame {k:02d} az={a:+.2f} deg")
            r = subprocess.run(cmd, capture_output=True, text=True)
            fits = os.path.join(fdir, "frame_0000.fits")
            if r.returncode != 0 or not os.path.exists(fits):
                print(r.stdout[-1500:]); print(r.stderr[-1500:])
                raise SystemExit(f"camera sim failed: {label} frame {k}")

            if label == list(roots)[0]:
                manifest["frames"].append(dict(
                    index=k, azimuth_deg=float(a),
                    pitch_down_deg=float(args.pitch_down_deg),
                    boresight=d.tolist(),
                    look_at_m=look_m.tolist()))

    mpath = os.path.join(args.out_root, "vise_pano_manifest.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nwrote {mpath}  ({N_FRAMES} frames x {len(roots)} meshes)")


if __name__ == "__main__":
    main()
