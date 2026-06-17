#!/usr/bin/env python3
"""Step 2 of the Lunar-VISE camera DEM-resolution study.

Build a *coarse* mesh emulating the orbital DEM a real mission would have
(~5 m LROC-NAC-class) instead of the 1 m SfS truth.

The cp22 source is an *irregular adaptive TIN*. We decimate it by grid
vertex clustering in the source local ENU frame (snap to a `--cell-m`
lattice, collapse, rebuild valid faces), then apply the SAME relocate+scale
transform as the fine mesh so the two are co-registered at the site.
5 m clustering at scale 0.01 -> ~5 cm facets (~5x coarser than fine).

Example:
  python3 src/python/vise_coarsen_mesh.py \
    --input /Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/europa/heat3d/dtm/cp22_1m_SFS_2M.ply \
    --output tmp/vise/cp22_coarse5m_site.ply \
    --cell-m 5.0 --target-lat 36.5 --target-lon -40.7 --scale 0.01
"""

import argparse
import time

import vise_mesh_transform as vmt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--cell-m", type=float, default=5.0,
                    help="Vertex-clustering cell in source meters "
                         "(emulated orbital-DEM feature size).")
    ap.add_argument("--target-lat", type=float, default=36.5)
    ap.add_argument("--target-lon", type=float, default=-40.7)
    ap.add_argument("--scale", type=float, default=0.01)
    ap.add_argument("--body-radius-m", type=float, default=1737400.0)
    args = ap.parse_args()

    t0 = time.time()
    print(f"[1/4] Reading {args.input}")
    V, faces = vmt.read_ascii_ply(args.input)
    print(f"      {V.shape[0]:,} verts, {faces.shape[0]:,} faces "
          f"({time.time()-t0:.1f}s)")

    print(f"[2/4] Vertex-cluster decimate at {args.cell_m} m "
          f"(source local ENU)")
    e, n, u, *_ = vmt.source_local_coords(V)
    re, rn, ru, cfaces = vmt.vertex_cluster_decimate(
        e, n, u, faces, args.cell_m)
    print(f"      coarse: {re.size:,} verts, {cfaces.shape[0]:,} faces "
          f"({time.time()-t0:.1f}s)")

    print(f"[3/4] Embed at ({args.target_lat}, {args.target_lon}) "
          f"deg, scale x{args.scale}")
    Vn, lat, lon, elev = vmt.embed_local(
        re, rn, ru, args.target_lat, args.target_lon,
        args.scale, args.body_radius_m)

    mn, mx, ext = vmt.bbox_report(Vn)
    em = vmt.min_edge_sample(Vn, cfaces, n=cfaces.shape[0])
    print(f"      new extent (m): {ext}")
    print(f"      min/median/max edge (m): "
          f"{em[0]:.5f} / {em[1]:.5f} / {em[2]:.5f}")
    print(f"      elevation (m):  {elev.min():.3f} .. {elev.max():.3f}")

    print(f"[4/4] Writing binary PLY {args.output}")
    vmt.write_binary_ply(args.output, Vn, lat, lon, elev, cfaces)
    print(f"      Done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
