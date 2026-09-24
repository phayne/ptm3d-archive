#!/usr/bin/env python3
"""Average per-step temperature files between two seasonal thermal runs.

Extends scripts/average_seasons.py (which only averages statistics) by
also averaging the per-step `temperature/temperature_NNNN.bin` files so the
exosphere stage can run on a season-averaged thermal trajectory.

Usage:
    python average_thermal_per_step.py \
        --season1 output/europa_ico6_albedo-map \
        --season2 output/europa_ico6_albedo-map_opp-season \
        --output  output/europa_ico6_albedo-map_avg
"""
from __future__ import annotations

import argparse
import os
import shutil
import struct
from pathlib import Path

import numpy as np


def read_bin(path):
    with open(path, "rb") as f:
        n = struct.unpack("<I", f.read(4))[0]
        return np.frombuffer(f.read(n * 4), dtype="<f4").copy()


def write_bin(path, arr):
    with open(path, "wb") as f:
        f.write(struct.pack("<I", arr.size))
        f.write(arr.astype(np.float32).tobytes())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--season1", required=True, type=Path)
    p.add_argument("--season2", required=True, type=Path)
    p.add_argument("--output",  required=True, type=Path)
    args = p.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "temperature").mkdir(exist_ok=True)
    (args.output / "statistics").mkdir(exist_ok=True)

    # Symlink mesh.ply (identical between runs)
    mesh_src = (args.season1 / "mesh.ply").resolve()
    mesh_dst = args.output / "mesh.ply"
    if mesh_dst.exists() or mesh_dst.is_symlink():
        mesh_dst.unlink()
    mesh_dst.symlink_to(mesh_src)

    # Average statistics (uses the same logic as average_seasons.py)
    stat_files = ["temperature_min.bin", "temperature_max.bin",
                  "temperature_avg.bin"]
    print("Averaging statistics:")
    for sf in stat_files:
        f1 = args.season1 / "statistics" / sf
        f2 = args.season2 / "statistics" / sf
        if not (f1.exists() and f2.exists()):
            print(f"  skip {sf} (missing in one run)")
            continue
        d1 = read_bin(f1); d2 = read_bin(f2)
        avg = 0.5 * (d1 + d2)
        write_bin(args.output / "statistics" / sf, avg)
        print(f"  {sf}: s1=[{d1.min():.1f},{d1.max():.1f}] "
              f"s2=[{d2.min():.1f},{d2.max():.1f}] avg=[{avg.min():.1f},{avg.max():.1f}]")

    # Average per-step temperature files. We assume the two runs have the
    # same number of steps (typically 288 per orbit at the same cadence);
    # the result is a single 288-step "annual-mean diurnal cycle."
    s1_steps = sorted((args.season1 / "temperature").glob("temperature_*.bin"), key=lambda p: int(str(p).rsplit("_", 1)[1][:-4]))
    s2_steps = sorted((args.season2 / "temperature").glob("temperature_*.bin"), key=lambda p: int(str(p).rsplit("_", 1)[1][:-4]))
    n = min(len(s1_steps), len(s2_steps))
    if n == 0:
        print("No per-step files; skipping per-step averaging")
    else:
        print(f"Averaging {n} per-step temperature files...")
        for i in range(n):
            d1 = read_bin(s1_steps[i])
            d2 = read_bin(s2_steps[i])
            avg = 0.5 * (d1 + d2)
            write_bin(args.output / "temperature" / s1_steps[i].name, avg)
        print(f"  wrote {n} averaged per-step files")

    # Time steps and metadata
    for fname in ("time_steps.txt", "metadata.txt"):
        src = args.season1 / fname
        if src.exists():
            shutil.copy(src, args.output / fname)
    print(f"\nAveraged thermal output written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
