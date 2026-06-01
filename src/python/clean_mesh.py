#!/usr/bin/env python3
"""Robust, body-agnostic surface-mesh cleaner for PTM3D.

PTM3D is a physics-based framework: view factors must satisfy the energy-
conservation bound Sum_j F_ij <= 1 for every face. Pathological mesh geometry
(degenerate triangles, duplicate faces, and decimation "fold-over" sandwiches
-- pairs of near-coincident triangles, often with antiparallel normals)
drives the centroid view-factor estimator r -> 0, which makes F_ij overshoot
and pushes row sums above 1. These artifacts are sub-resolution: the offending
triangles are typically 100-1000x smaller than the median face, so removing
them changes the resolved surface negligibly while restoring energy
conservation.

This tool removes that pathology without any clamps or arbitrary caps -- it
fixes the geometry so the physics-based view factors are well-posed. All
thresholds are *relative* to the mesh's own median face size, so the defaults
work across bodies (Moon, Europa, ...), units (m, km, cm), and resolutions.

Pipeline (each step reports before/after counts):
  1. merge coincident vertices (within --merge-tol, relative to median edge)
  2. drop topologically degenerate faces (a repeated vertex index)
  3. drop zero / sub-resolution faces (area < --rel-area-min * median area)
  4. drop exact duplicate faces (same vertex set)
  5. break near-coincident pairs (centroid sep < --coincident-frac * size):
     greedily remove the smaller-area face of each pair so no overshoot pair
     survives -- this kills fold-over sandwiches
  6. (optional) drop slivers with shape quality < --sliver-qual
  7. (optional) keep only the largest connected region (--largest-region)
  8. recompute consistently oriented vertex/face normals

Usage:
  clean_mesh.py --in mesh.vtk --out mesh_clean.vtk [options] [--report-only]
"""
import argparse, sys
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree


def tri_geom(v, f):
    p0, p1, p2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    cross = np.cross(p1 - p0, p2 - p0)
    twoA = np.linalg.norm(cross, axis=1)
    area = 0.5 * twoA
    with np.errstate(invalid="ignore", divide="ignore"):
        nrm = cross / np.where(twoA[:, None] > 0, twoA[:, None], 1.0)
    cen = (p0 + p1 + p2) / 3.0
    L = np.stack([np.linalg.norm(p1 - p0, axis=1),
                  np.linalg.norm(p2 - p1, axis=1),
                  np.linalg.norm(p0 - p2, axis=1)], axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        qual = 4.0 * np.sqrt(3.0) * area / np.sum(L ** 2, axis=1)
    return p0, p1, p2, area, nrm, cen, qual, L


def faces_of(mesh):
    return mesh.faces.reshape(-1, 4)[:, 1:].copy()


def rebuild(mesh, keep_mask):
    f = faces_of(mesh)[keep_mask]
    out = pv.PolyData(mesh.points.copy(),
                      np.hstack([np.full((len(f), 1), 3, np.int64), f]).ravel())
    return out.clean(point_merging=False)  # drop orphaned points only


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", default=None)
    ap.add_argument("--rel-area-min", type=float, default=1e-2,
                    help="drop faces with area < this * median area (default 1e-2)")
    ap.add_argument("--coincident-frac", type=float, default=0.25,
                    help="pair is near-coincident if centroid sep < this * min(sqrt(area)) (default 0.25)")
    ap.add_argument("--sliver-qual", type=float, default=0.0,
                    help="drop faces with shape quality < this (0=disabled; 1=equilateral)")
    ap.add_argument("--merge-tol", type=float, default=1e-3,
                    help="merge vertices closer than this * median edge length (0=disabled, default 1e-3)")
    ap.add_argument("--largest-region", action="store_true",
                    help="keep only the largest connected component")
    ap.add_argument("--report-only", action="store_true",
                    help="diagnose and report, do not write output")
    a = ap.parse_args()

    m = pv.read(a.inp).extract_surface().triangulate()
    n0, v0 = m.n_cells, m.n_points
    print(f"[clean] input: {n0} faces, {v0} verts  ({a.inp})")

    # --- step 1: merge coincident vertices -------------------------------
    if a.merge_tol > 0 and not a.report_only:
        _, _, _, _, _, _, _, L = tri_geom(m.points, faces_of(m))
        med_edge = float(np.median(L))
        tol = a.merge_tol * med_edge
        before = m.n_points
        m = m.clean(tolerance=tol, absolute=True,
                    lines_to_points=False, polys_to_lines=False).triangulate()
        merged = before - m.n_points
        if merged:
            print(f"[clean] merged coincident vertices (tol={tol:.3e}): {merged} removed")

    f = faces_of(m)
    p0, p1, p2, area, nrm, cen, qual, L = tri_geom(m.points, f)
    med_area = float(np.median(area))
    nF = len(f)
    keep = np.ones(nF, bool)

    # --- step 2: degenerate connectivity (repeated vertex) ---------------
    degconn = (f[:, 0] == f[:, 1]) | (f[:, 1] == f[:, 2]) | (f[:, 0] == f[:, 2])
    keep &= ~degconn
    print(f"[clean] degenerate-connectivity faces: {int(degconn.sum())}")

    # --- step 3: zero / sub-resolution area ------------------------------
    tiny = area < a.rel_area_min * med_area
    keep &= ~tiny
    print(f"[clean] sub-resolution faces (area < {a.rel_area_min:g}*median={a.rel_area_min*med_area:.3e}): "
          f"{int(tiny.sum())}")

    # --- step 4: exact duplicate faces (same vertex set) -----------------
    fsort = np.sort(f, axis=1)
    order = np.lexsort(fsort.T[::-1])
    dup = np.zeros(nF, bool)
    so = fsort[order]
    same = np.all(so[1:] == so[:-1], axis=1)
    dup_in_order = np.concatenate([[False], same])  # mark 2nd+ of each group
    dup[order] = dup_in_order
    keep &= ~dup
    print(f"[clean] exact duplicate faces: {int(dup.sum())}")

    # --- step 5: break near-coincident pairs (fold-over sandwiches) ------
    rad = np.sqrt(np.maximum(area, 0.0))
    med_rad = float(np.median(rad))
    tree = cKDTree(cen)
    pairs = tree.query_pairs(r=a.coincident_frac * med_rad, output_type="ndarray")
    removed_pair = 0
    antiparallel = 0
    if len(pairs):
        i, j = pairs[:, 0], pairs[:, 1]
        d = np.linalg.norm(cen[i] - cen[j], axis=1)
        rmin = np.minimum(rad[i], rad[j])
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where(rmin > 0, d / rmin, 0.0)
        bad = ratio < a.coincident_frac
        bi, bj, br = i[bad], j[bad], ratio[bad]
        dot = np.sum(nrm[bi] * nrm[bj], axis=1)
        antiparallel = int((dot < -0.5).sum())
        # greedy: closest first; remove the smaller-area face of each surviving pair
        for k in np.argsort(br):
            ii, jj = bi[k], bj[k]
            if not (keep[ii] and keep[jj]):
                continue
            drop = ii if area[ii] <= area[jj] else jj
            keep[drop] = False
            removed_pair += 1
    print(f"[clean] near-coincident pairs (sep/size<{a.coincident_frac:g}): "
          f"broke {removed_pair} ({antiparallel} antiparallel sandwiches)")

    # --- step 6: optional sliver removal ---------------------------------
    if a.sliver_qual > 0:
        sliver = (qual < a.sliver_qual) & keep
        keep &= ~sliver
        print(f"[clean] slivers (quality < {a.sliver_qual:g}): {int(sliver.sum())}")

    n_removed = int((~keep).sum())
    print(f"[clean] total faces removed: {n_removed} "
          f"({100.0*n_removed/nF:.3f}%), removed area fraction: "
          f"{area[~keep].sum()/area.sum():.3e}")

    if a.report_only:
        print("[clean] report-only: no output written")
        return 0

    m = rebuild(m, keep)
    if a.largest_region:
        bodies = m.connectivity("largest")
        print(f"[clean] kept largest connected region: {m.n_cells} -> {bodies.n_cells} faces")
        m = bodies.extract_surface().triangulate()

    # consistent outward-ish normals (auto-orient connected components)
    m = m.compute_normals(cell_normals=True, point_normals=True,
                          auto_orient_normals=True, consistent_normals=True,
                          non_manifold_traversal=False)

    # final energy-conservation-relevant report
    f2 = faces_of(m)
    _, _, _, area2, _, _, qual2, _ = tri_geom(m.points, f2)
    print(f"[clean] output: {m.n_cells} faces, {m.n_points} verts  "
          f"(area med={np.median(area2):.3e}, qual min={np.nanmin(qual2):.3e})")
    out = a.out or a.inp.rsplit(".", 1)[0] + "_clean.vtk"
    m.save(out)
    print(f"[clean] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
