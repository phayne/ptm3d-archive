#!/usr/bin/env python3
"""Build a FULL-RESOLUTION mesh whose topography is the COARSE DEM.

Keeps the fine mesh's ~1 m triangulation (so the camera render is smooth,
no coarse-facet artifacts) but replaces every vertex elevation with the
value linearly interpolated from the coarse (100x vertex-clustered)
surface. This represents "what the scene looks like if the terrain truly
had only coarse-scale topography" rendered at full image resolution.

Interpolation is barycentric-linear on the coarse mesh's own
triangulation (matplotlib.tri); fine vertices outside the coarse hull are
dropped (and faces touching them), so the result covers the coarse
footprint at fine resolution.

Example:
  python3 src/python/vise_smooth_mesh.py \
    --input .../cp22_1m_SFS_2M.ply --output tmp/vise/cp22_smooth100x_site.ply \
    --cell-m 100 --target-lat 36.5 --target-lon -40.7 --scale 1.0
"""

import argparse
import time

import numpy as np
import matplotlib.tri as mtri

import vise_mesh_transform as vmt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--cell-m", type=float, default=100.0,
                    help="Coarse vertex-clustering cell (source m). 100 = "
                         "the 1 m-equiv DEM used in dem-100x.")
    ap.add_argument("--target-lat", type=float, default=36.5)
    ap.add_argument("--target-lon", type=float, default=-40.7)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--body-radius-m", type=float, default=1737400.0)
    args = ap.parse_args()

    t0 = time.time()
    print(f"[1/5] Reading {args.input}")
    V, faces = vmt.read_ascii_ply(args.input)
    e, n, u, *_ = vmt.source_local_coords(V)
    print(f"      fine: {V.shape[0]:,} verts, {faces.shape[0]:,} faces "
          f"({time.time()-t0:.1f}s)")

    print(f"[2/5] Coarse topo model: vertex-cluster @ {args.cell_m} m")
    re, rn, ru, cfaces = vmt.vertex_cluster_decimate(
        e, n, u, faces, args.cell_m)
    print(f"      coarse: {re.size:,} nodes, {cfaces.shape[0]:,} tris")

    print("[3/5] Linear-interpolate coarse elevation onto fine vertices")
    triang = mtri.Triangulation(re, rn, triangles=cfaces)
    interp = mtri.LinearTriInterpolator(triang, ru)
    u_s = interp(e, n)                       # masked outside coarse hull
    valid_v = ~np.ma.getmaskarray(u_s)
    u_s = np.asarray(u_s.filled(0.0), dtype=np.float64)
    keep_f = valid_v[faces].all(axis=1)
    nf = faces[keep_f]
    used = np.unique(nf)
    remap = np.full(V.shape[0], -1, np.int64)
    remap[used] = np.arange(used.size)
    nf = remap[nf].astype(np.int32)
    print(f"      kept {used.size:,} verts, {nf.shape[0]:,} faces "
          f"(inside coarse hull); dropped "
          f"{faces.shape[0]-nf.shape[0]:,} faces")

    print(f"[4/5] Embed at ({args.target_lat}, {args.target_lon}) "
          f"scale x{args.scale}")
    Vn, lat, lon, elev = vmt.embed_local(
        e[used], n[used], u_s[used],
        args.target_lat, args.target_lon, args.scale, args.body_radius_m)
    mn, mx, ext = vmt.bbox_report(Vn)
    em = vmt.min_edge_sample(Vn, nf, n=nf.shape[0])
    print(f"      extent (m): {ext}")
    print(f"      min/median/max edge (m): "
          f"{em[0]:.4f} / {em[1]:.4f} / {em[2]:.4f}")
    print(f"      elevation (m): {elev.min():.3f} .. {elev.max():.3f} "
          f"(coarse-smoothed; fine relief removed)")

    print(f"[5/5] Writing binary PLY {args.output}")
    vmt.write_binary_ply(args.output, Vn, lat, lon, elev, nf)
    print(f"      Done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
