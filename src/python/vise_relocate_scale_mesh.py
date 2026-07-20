#!/usr/bin/env python3
"""Step 1 of the Lunar-VISE camera DEM-resolution study.

Read the high-res cp22 SfS DTM, scale it so the smallest facets are ~1 cm,
and relocate it to the planned landing site so SPICE computes the correct
Sun geometry there. Writes a binary_little_endian PLY the C++ pipeline
loads natively.

Example:
  python3 src/python/vise_relocate_scale_mesh.py \
    --input /Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/europa/heat3d/dtm/cp22_1m_SFS_2M.ply \
    --output tmp/vise/cp22_fine_site.ply \
    --target-lat 36.5 --target-lon -40.7 --scale 0.01
"""

import argparse
import time

import numpy as np

import vise_mesh_transform as vmt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--target-lat", type=float, default=36.5)
    ap.add_argument("--target-lon", type=float, default=-40.7)
    ap.add_argument("--scale", type=float, default=0.01,
                    help="Uniform scale (0.01 -> ~1 cm min facet for cp22).")
    ap.add_argument("--body-radius-m", type=float, default=1737400.0,
                    help="Lunar reference radius (m).")
    args = ap.parse_args()

    t0 = time.time()
    print(f"[1/4] Reading {args.input}")
    V, faces = vmt.read_ascii_ply(args.input)
    print(f"      {V.shape[0]:,} vertices, {faces.shape[0]:,} faces "
          f"({time.time()-t0:.1f}s)")
    mn, mx, ext = vmt.bbox_report(V)
    e0 = vmt.min_edge_sample(V, faces)
    print(f"      source extent (m): {ext}")
    print(f"      source min/median/max edge (m): "
          f"{e0[0]:.4f} / {e0[1]:.4f} / {e0[2]:.4f}")

    print(f"[2/4] Relocate to ({args.target_lat}, {args.target_lon}) "
          f"deg, scale x{args.scale}")
    Vn, lat, lon, elev = vmt.relocate_scale(
        V, args.target_lat, args.target_lon, args.scale, args.body_radius_m)

    mn, mx, ext = vmt.bbox_report(Vn)
    e1 = vmt.min_edge_sample(Vn, faces)
    print(f"      new extent (m): {ext}")
    print(f"      new min/median/max edge (m): "
          f"{e1[0]:.5f} / {e1[1]:.5f} / {e1[2]:.5f}")
    print(f"      new lat range:  {lat.min():.5f} .. {lat.max():.5f}")
    print(f"      new lon range:  {lon.min():.5f} .. {lon.max():.5f}")
    print(f"      elevation (m):  {elev.min():.3f} .. {elev.max():.3f}")

    print(f"[3/4] Writing binary PLY {args.output}")
    vmt.write_binary_ply(args.output, Vn, lat, lon, elev, faces)

    print(f"[4/4] Done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
