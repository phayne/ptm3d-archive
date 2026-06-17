#!/usr/bin/env python3
"""Characterize the crater-floor terrain view factor (VF row sum) from a saved
H-matrix dump, and A/B two dumps (e.g. SUBDIVIDE off vs on).

Floor faces are identified geometrically from the mesh: the deep PSR floor =
faces within a horizontal radius of the crater center AND in the lowest
elevation band (the crater depression). The "up" axis is auto-detected from the
vertex bounding box (selenocentric radial if |mean| is planetary-scale, else z).

Usage:
  analyze_floor_vf.py --mesh core.ply --rowsum a.npy [--rowsum-b b.npy]
                      [--floor-radius-m 2500] [--floor-frac 0.25]
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np


def face_centroids(mesh: Path):
    from plyfile import PlyData
    pd = PlyData.read(str(mesh))
    v = np.stack([np.asarray(pd["vertex"]["x"], np.float64),
                  np.asarray(pd["vertex"]["y"], np.float64),
                  np.asarray(pd["vertex"]["z"], np.float64)], axis=1)
    fd = pd["face"].data
    fld = next(n for n in ("vertex_indices", "vertex_index") if n in fd.dtype.names)
    raw = fd[fld]
    faces = np.stack([np.asarray(r, np.int64) for r in raw])
    return v[faces].mean(axis=1)  # (nf,3)


def detect_up(cent: np.ndarray) -> int:
    """Up axis = the one whose mean magnitude is planetary-scale (selenocentric),
    else the axis with the largest absolute mean; falls back to z (2)."""
    means = np.abs(cent.mean(axis=0))
    if means.max() > 1e5:        # planetary-scale coordinate -> radial = up
        return int(np.argmax(means))
    return 2                      # local frame -> z up


def floor_mask(cent: np.ndarray, radius_m: float, frac: float):
    """Faces within `radius_m` of the horizontal center AND in the lowest `frac`
    elevation quantile within that radius (the deep crater floor)."""
    up = detect_up(cent)
    horiz = [a for a in range(3) if a != up]
    c2 = cent[:, horiz]
    center = np.median(c2, axis=0)
    r = np.linalg.norm(c2 - center, axis=1)
    near = r <= radius_m
    if near.sum() == 0:
        return near, up
    elev = cent[:, up]
    thresh = np.quantile(elev[near], frac)
    return near & (elev <= thresh), up


def stats(name: str, rs: np.ndarray, floor: np.ndarray):
    fr = rs[floor]
    zero_all = float(np.mean(rs < 1e-4))
    zero_fl = float(np.mean(fr < 1e-4))
    pct = np.percentile(fr, [10, 25, 50, 75, 90])
    print(f"\n[{name}]")
    print(f"  whole mesh: N={rs.size}  mean={rs.mean():.4f}  "
          f"frac(rowsum<1e-4)={zero_all*100:.1f}%  min={rs.min():.4f} max={rs.max():.4f}")
    print(f"  FLOOR ({floor.sum()} faces): mean F_terrain={fr.mean():.4f}  "
          f"median={pct[2]:.4f}")
    print(f"    floor pctiles 10/25/50/75/90 = "
          f"{pct[0]:.4f}/{pct[1]:.4f}/{pct[2]:.4f}/{pct[3]:.4f}/{pct[4]:.4f}")
    print(f"    frac of FLOOR faces with rowsum<1e-4 (decoupled) = {zero_fl*100:.1f}%")
    print(f"    (geometric DEM ground truth ~0.0055; 100k H-matrix floor ~0.013)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True, type=Path)
    ap.add_argument("--rowsum", required=True, type=Path)
    ap.add_argument("--rowsum-b", type=Path, default=None)
    ap.add_argument("--floor-radius-m", type=float, default=2500.0)
    ap.add_argument("--floor-frac", type=float, default=0.25)
    args = ap.parse_args()

    cent = face_centroids(args.mesh)
    floor, up = floor_mask(cent, args.floor_radius_m, args.floor_frac)
    print(f"mesh {args.mesh.name}: {cent.shape[0]} faces; up-axis={up}; "
          f"bbox={cent.min(0).round(1)}..{cent.max(0).round(1)}")
    print(f"floor: {floor.sum()} faces within {args.floor_radius_m:.0f} m of center, "
          f"lowest {args.floor_frac*100:.0f}% elevation")

    rs_a = np.load(args.rowsum)
    if rs_a.size != cent.shape[0]:
        raise SystemExit(f"rowsum N={rs_a.size} != mesh faces {cent.shape[0]}")
    stats(args.rowsum.stem, rs_a, floor)

    if args.rowsum_b is not None:
        rs_b = np.load(args.rowsum_b)
        if rs_b.size != cent.shape[0]:
            raise SystemExit(f"rowsum-b N={rs_b.size} != mesh faces {cent.shape[0]}")
        stats(args.rowsum_b.stem, rs_b, floor)
        # A/B delta: faces that GAINED coupling
        d = rs_b - rs_a
        gained = d > 1e-3
        print(f"\n[A/B delta  B - A]")
        print(f"  faces gaining coupling (delta>1e-3): {gained.sum()} "
              f"({gained.mean()*100:.2f}% of mesh)")
        if gained.sum():
            gc = cent[gained]
            print(f"  gained-face delta: mean={d[gained].mean():.4f} max={d[gained].max():.4f}")
            print(f"  gained-face centroid bbox: {gc.min(0).round(1)}..{gc.max(0).round(1)}")
        print(f"  FLOOR mean F_terrain: A={rs_a[floor].mean():.4f} -> B={rs_b[floor].mean():.4f}  "
              f"(x{(rs_b[floor].mean()/max(rs_a[floor].mean(),1e-9)):.1f})")
        print(f"  FLOOR frac decoupled (<1e-4): A={np.mean(rs_a[floor]<1e-4)*100:.1f}% "
              f"-> B={np.mean(rs_b[floor]<1e-4)*100:.1f}%")


if __name__ == "__main__":
    main()
