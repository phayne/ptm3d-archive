#!/usr/bin/env python3
"""Compute Clipper closest-approach-to-Moon time/distance and derive camera FOV.

Reads the Clipper SPK (Moon-centered) that was fetched from HORIZONS, samples
range(t) over the full thermal-sim window, locates the minimum, and emits the
FOV that makes the Moon fill ~`--fill` of the image's short axis at that instant.

Writes ``build/clipper_ega_geometry.json`` and prints a summary to stdout.
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import spiceypy as spice

R_MOON_KM = 1737.4


def load_kernels(kernels_dir: Path, clipper_spk: Path):
    spice.furnsh(str(kernels_dir / "lsk" / "naif0012.tls"))
    spice.furnsh(str(kernels_dir / "pck" / "pck00010.tpc"))
    spice.furnsh(str(kernels_dir / "spk" / "de440s.bsp"))
    spice.furnsh(str(clipper_spk))


def sample_range(et0, et1, n_samples, clipper_id):
    ets = np.linspace(et0, et1, n_samples)
    ranges = np.empty_like(ets)
    for i, et in enumerate(ets):
        pos, _ = spice.spkpos(str(clipper_id), et, "J2000", "NONE", "MOON")
        ranges[i] = math.sqrt(pos[0] ** 2 + pos[1] ** 2 + pos[2] ** 2)
    return ets, ranges


def refine_ca(et_coarse, clipper_id, half_window_s=120.0, n_fine=481):
    """Two-pass refinement around the coarse CA to pin it down to seconds."""
    et0 = et_coarse - half_window_s
    et1 = et_coarse + half_window_s
    ets, ranges = sample_range(et0, et1, n_fine, clipper_id)
    idx = int(np.argmin(ranges))
    return ets[idx], ranges[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernels-dir", default=os.environ.get("CSPICE_DATA", str(Path(__file__).resolve().parents[1] / "data" / "kernels")))
    ap.add_argument("--spk", default="data/spice/clipper_ega_moon.bsp")
    ap.add_argument("--clipper-id", default="-159")
    ap.add_argument("--utc-start", default="2026-12-01T23:59:59.803")
    ap.add_argument("--utc-end", default="2026-12-06T23:29:59.803")
    ap.add_argument(
        "--fill",
        type=float,
        default=0.91,
        help="Fraction of short image axis the Moon should occupy at CA",
    )
    ap.add_argument("--image-aspect", type=float, default=1392.0 / 928.0)
    ap.add_argument("--output", default="build/clipper_ega_geometry.json")
    args = ap.parse_args()

    load_kernels(Path(args.kernels_dir), Path(args.spk))

    et_start = spice.str2et(args.utc_start)
    et_end = spice.str2et(args.utc_end)

    # Coarse 1-min sweep across the window, then refine around the minimum.
    n_coarse = int((et_end - et_start) / 60.0) + 1
    ets, ranges = sample_range(et_start, et_end, n_coarse, args.clipper_id)
    imin = int(np.argmin(ranges))
    et_ca, r_ca = refine_ca(ets[imin], args.clipper_id)

    utc_ca = spice.et2utc(et_ca, "ISOC", 3)
    alt_ca = r_ca - R_MOON_KM

    # Moon angular diameter at CA
    theta_moon_deg = 2.0 * math.degrees(math.atan(R_MOON_KM / r_ca))

    # Camera fov_x is the horizontal (long-axis) FOV. With `fill` meaning the
    # fraction of the SHORT axis (y) that the Moon should cover at CA, the
    # horizontal FOV follows from the aspect ratio.
    fov_y_deg = theta_moon_deg / args.fill
    fov_x_deg = fov_y_deg * args.image_aspect

    r_min = float(np.min(ranges))
    r_max = float(np.max(ranges))

    geom = {
        "clipper_naif_id": int(args.clipper_id),
        "spk_path": str(Path(args.spk).resolve()),
        "sim_window_utc": [args.utc_start, args.utc_end],
        "et_ca": float(et_ca),
        "utc_ca": utc_ca,
        "range_ca_km": float(r_ca),
        "altitude_ca_km": float(alt_ca),
        "range_min_km": r_min,
        "range_max_km": r_max,
        "moon_angular_diameter_ca_deg": theta_moon_deg,
        "fill_short_axis": args.fill,
        "image_aspect_wh": args.image_aspect,
        "fov_x_deg": fov_x_deg,
        "fov_y_deg": fov_y_deg,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(geom, f, indent=2)

    print("=== Clipper EGA Camera Geometry ===")
    print(f"  CA UTC:             {utc_ca}")
    print(f"  CA range:           {r_ca:,.1f} km  (altitude {alt_ca:,.1f} km)")
    print(f"  Range over window:  {r_min:,.1f} – {r_max:,.1f} km")
    print(f"  Moon angular diam:  {theta_moon_deg:.4f} deg at CA")
    print(f"  Target fill:        {args.fill*100:.0f}% of short axis")
    print(f"  fov_x_deg:          {fov_x_deg:.4f}")
    print(f"  fov_y_deg:          {fov_y_deg:.4f}")
    print(f"  Written:            {args.output}")

    spice.kclear()


if __name__ == "__main__":
    main()
