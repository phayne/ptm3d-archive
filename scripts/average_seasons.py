#!/usr/bin/env python3
"""Average temperature statistics from two opposite-season simulations.

Reads binary statistics files (temperature_min/max/avg.bin) from two
simulation output directories, computes element-wise averages, and writes
the results to a new output directory.  Also symlinks the mesh.ply from
the first season (the mesh is identical).

Usage:
    python scripts/average_seasons.py \
        --season1 output/europa_ico6_uniform-ice \
        --season2 output/europa_ico6_uniform-ice_opp-season \
        --output  output/europa_ico6_uniform-ice_avg
"""

import os
import struct
import argparse
import shutil
import numpy as np


STAT_FILES = [
    'temperature_min.bin',
    'temperature_max.bin',
    'temperature_avg.bin',
]


def read_binary_field(filepath):
    """Read uint32 count + float32 array."""
    with open(filepath, 'rb') as f:
        count = struct.unpack('<I', f.read(4))[0]
        data = np.frombuffer(f.read(count * 4), dtype='<f4').copy()
    return data


def write_binary_field(filepath, data):
    """Write uint32 count + float32 array."""
    arr = data.astype(np.float32)
    with open(filepath, 'wb') as f:
        f.write(struct.pack('<I', len(arr)))
        f.write(arr.tobytes())


def main():
    parser = argparse.ArgumentParser(
        description='Average temperature statistics from two seasons')
    parser.add_argument('--season1', required=True,
                        help='First season output directory')
    parser.add_argument('--season2', required=True,
                        help='Second (opposite) season output directory')
    parser.add_argument('--output', required=True,
                        help='Output directory for averaged statistics')
    args = parser.parse_args()

    stats_dir = os.path.join(args.output, 'statistics')
    os.makedirs(stats_dir, exist_ok=True)

    # Symlink mesh.ply from season1 (identical mesh)
    mesh_src = os.path.join(args.season1, 'mesh.ply')
    mesh_dst = os.path.join(args.output, 'mesh.ply')
    if os.path.exists(mesh_dst):
        os.remove(mesh_dst)
    if os.path.isabs(mesh_src):
        os.symlink(mesh_src, mesh_dst)
    else:
        os.symlink(os.path.abspath(mesh_src), mesh_dst)
    print(f"Linked mesh: {mesh_dst} -> {mesh_src}")

    # Average each statistic
    for stat_file in STAT_FILES:
        f1 = os.path.join(args.season1, 'statistics', stat_file)
        f2 = os.path.join(args.season2, 'statistics', stat_file)

        d1 = read_binary_field(f1)
        d2 = read_binary_field(f2)

        if len(d1) != len(d2):
            print(f"WARNING: size mismatch for {stat_file}: "
                  f"{len(d1)} vs {len(d2)}")
            continue

        avg = (d1 + d2) / 2.0
        out_path = os.path.join(stats_dir, stat_file)
        write_binary_field(out_path, avg)

        print(f"  {stat_file}: season1=[{d1.min():.1f}, {d1.max():.1f}] K, "
              f"season2=[{d2.min():.1f}, {d2.max():.1f}] K, "
              f"avg=[{avg.min():.1f}, {avg.max():.1f}] K")

    print(f"\nAveraged statistics written to {stats_dir}")


if __name__ == '__main__':
    main()
