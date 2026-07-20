#!/usr/bin/env python3
"""Analyze Clipper geometry vs Earth + Moon to pick a wide-view FOV.

Samples the PTM3D time_steps.txt epochs and computes per-frame:
  - Clipper->Moon distance, Moon angular diameter
  - Clipper->Earth distance, Earth angular diameter
  - Earth-Moon angular separation from Clipper (with Moon look_at pointing, this
    tells us how far Earth drifts off-center)

Writes JSON with the full per-frame geometry AND prints summary stats used to
pick `fov_x_deg` so both bodies fit in-frame for as many frames as possible.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import spiceypy as spice

R_MOON_KM = 1737.4
R_EARTH_KM = 6378.14


def load_kernels(kernels_dir: Path, clipper_spk: Path):
    spice.furnsh(str(kernels_dir / "lsk" / "naif0012.tls"))
    spice.furnsh(str(kernels_dir / "pck" / "pck00010.tpc"))
    spice.furnsh(str(kernels_dir / "spk" / "de440s.bsp"))
    spice.furnsh(str(clipper_spk))


def read_utc_list(time_steps: Path):
    utcs = []
    with open(time_steps) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            utcs.append(line.split()[1])
    return utcs


def ang_deg(v1, v2):
    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
    c = np.dot(v1, v2) / (n1 * n2)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernels-dir", default="/Users/paha3326/software/cspice/data")
    ap.add_argument("--spk", default="data/spice/clipper_ega_moon.bsp")
    ap.add_argument("--clipper-id", default="-159")
    ap.add_argument("--time-steps", default="gui/output/moon_clipper-EGA_ico9_30m/time_steps.txt")
    ap.add_argument("--output", default="build/clipper_ega_wideview_geometry.json")
    ap.add_argument(
        "--margin",
        type=float,
        default=1.10,
        help="FOV = max(separation + Earth_radius + Moon_radius) × margin",
    )
    args = ap.parse_args()

    load_kernels(Path(args.kernels_dir), Path(args.spk))
    utcs = read_utc_list(Path(args.time_steps))
    print(f"Analyzing {len(utcs)} frames from {args.time_steps}")

    frames = []
    for utc in utcs:
        et = spice.str2et(utc)
        # Clipper position in IAU_MOON (Moon body-fixed). Everything we compute
        # here is a direction/magnitude — frame doesn't matter as long as we
        # use the same frame for Earth too.
        p_clipper, _ = spice.spkpos(args.clipper_id, et, "IAU_MOON", "NONE", "MOON")
        p_earth,   _ = spice.spkpos("399",           et, "IAU_MOON", "NONE", "MOON")

        p_clipper = np.asarray(p_clipper, dtype=float)
        p_earth   = np.asarray(p_earth,   dtype=float)

        # Vectors from Clipper to each body
        v_moon  = -p_clipper                   # Moon is at origin in this frame
        v_earth = p_earth - p_clipper

        d_moon  = float(np.linalg.norm(v_moon))
        d_earth = float(np.linalg.norm(v_earth))

        ang_moon_diam_deg = 2.0 * math.degrees(math.asin(min(1.0, R_MOON_KM / d_moon)))
        ang_earth_diam_deg = 2.0 * math.degrees(math.asin(min(1.0, R_EARTH_KM / d_earth)))
        sep_deg = ang_deg(v_moon, v_earth)

        frames.append({
            "utc": utc,
            "et": float(et),
            "clipper_moon_km":  d_moon,
            "clipper_earth_km": d_earth,
            "moon_ang_deg":     ang_moon_diam_deg,
            "earth_ang_deg":    ang_earth_diam_deg,
            "sep_deg":          sep_deg,
        })

    # Summary
    seps = np.array([f["sep_deg"] for f in frames])
    moons = np.array([f["moon_ang_deg"] for f in frames])
    earths = np.array([f["earth_ang_deg"] for f in frames])
    d_e = np.array([f["clipper_earth_km"] for f in frames])

    # FOV requirement: if pointing at Moon center, the Earth needs to fall
    # inside the half-FOV. Required half-FOV = separation + earth_radius.
    # So required full FOV >= 2 * (sep + earth_radius).
    req_fov_deg = 2.0 * (seps + earths / 2.0)
    i_worst = int(np.argmax(req_fov_deg))
    i_min_earth = int(np.argmin(d_e))

    print("")
    print(f"Moon ang diam:        {moons.min():.3f} – {moons.max():.3f} deg")
    print(f"Earth ang diam:       {earths.min():.3f} – {earths.max():.3f} deg "
          f"(Earth-CA at frame {i_min_earth}: {frames[i_min_earth]['utc']}, "
          f"d={d_e[i_min_earth]:,.0f} km, {earths[i_min_earth]:.2f} deg)")
    print(f"Earth-Moon angular sep: {seps.min():.2f} – {seps.max():.2f} deg")
    print(f"Required FOV (point at Moon):  {req_fov_deg.min():.2f} – {req_fov_deg.max():.2f} deg")
    print(f"Worst frame: #{i_worst} {frames[i_worst]['utc']} "
          f"req_fov={req_fov_deg[i_worst]:.1f} deg (sep={seps[i_worst]:.1f}, "
          f"earth_diam={earths[i_worst]:.2f})")

    # For a reasonable animation, we cap FOV at what fits a comfortable margin
    # around the MEDIAN required FOV. Near Earth-CA the Earth will clip.
    sorted_req = np.sort(req_fov_deg)
    p90 = float(sorted_req[int(0.90 * len(sorted_req))])
    suggested_fov = min(90.0, args.margin * p90)
    print(f"\n90th-percentile req FOV: {p90:.2f} deg")
    print(f"Suggested fov_x_deg:     {suggested_fov:.2f} "
          f"(capped at 90; frames above this will have Earth clipped)")

    clipped = int(np.sum(req_fov_deg > suggested_fov))
    print(f"Frames where Earth will be clipped/off-frame: {clipped} / {len(frames)}")

    out = {
        "summary": {
            "n_frames": len(frames),
            "suggested_fov_x_deg": suggested_fov,
            "fov_margin": args.margin,
            "moon_ang_deg_range":   [float(moons.min()),  float(moons.max())],
            "earth_ang_deg_range":  [float(earths.min()), float(earths.max())],
            "sep_deg_range":        [float(seps.min()),   float(seps.max())],
            "req_fov_deg_p90":      p90,
            "frames_earth_clipped": clipped,
        },
        "frames": frames,
    }
    out_path = Path(args.output); out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWritten: {out_path}")

    spice.kclear()


if __name__ == "__main__":
    main()
