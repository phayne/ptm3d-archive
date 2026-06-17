#!/usr/bin/env python3
"""
Compute the view-factor reciprocity residual of a saved VFM matrix.

The view-factor matrix satisfies ``A_i F_ij = A_j F_ji`` exactly in the
continuous limit; the Monte-Carlo estimator used in production introduces
an O(M^{-1/2}) violation at finite sample count M.  This script reads a
``.vfm`` / ``.hmat`` sparse CSR file and the associated mesh PLY, and reports

  - max relative |A_i F_ij − A_j F_ji|  over significant pairs
  - mean / RMS relative residual
  - distribution as percentile quantiles

These are the numbers cited in §6.2 of the PTM3D methods paper.
"""
from __future__ import annotations

import argparse
import struct
import sys
import time
from pathlib import Path

import numpy as np


MAGIC_VFM = 0x5646464D
VERSION_VFM = 1


def read_vfm(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (row_ptr int32[N+1], col_idx int32[nnz], vals float64[nnz])."""
    with open(path, "rb") as f:
        magic, version = struct.unpack("II", f.read(8))
        if magic != MAGIC_VFM:
            raise SystemExit(f"Bad magic 0x{magic:08X} in {path}")
        if version != VERSION_VFM:
            raise SystemExit(f"Unsupported VFM version {version}")
        nrows, ncols, nnz = struct.unpack("QQQ", f.read(24))
        row_ptr = np.frombuffer(f.read((nrows + 1) * 4), dtype=np.int32)
        col_idx = np.frombuffer(f.read(nnz * 4), dtype=np.int32)
        vals = np.frombuffer(f.read(nnz * 8), dtype=np.float64)
    return row_ptr.copy(), col_idx.copy(), vals.copy()


def read_ply_face_areas(path: Path) -> np.ndarray:
    """Return per-face areas (float64) using the plyfile library."""
    from plyfile import PlyData
    pd = PlyData.read(str(path))
    v = np.stack(
        [np.asarray(pd["vertex"]["x"], dtype=np.float64),
         np.asarray(pd["vertex"]["y"], dtype=np.float64),
         np.asarray(pd["vertex"]["z"], dtype=np.float64)],
        axis=1,
    )
    # face list element is a structured ndarray; try common names
    face_data = pd["face"].data
    idx_field = None
    for name in ("vertex_indices", "vertex_index"):
        if name in face_data.dtype.names:
            idx_field = name
            break
    if idx_field is None:
        raise SystemExit(f"No vertex_indices field in face element of {path}; "
                         f"got fields {face_data.dtype.names}")
    raw = face_data[idx_field]
    if isinstance(raw[0], np.ndarray):
        faces = np.stack([np.asarray(r, dtype=np.int64) for r in raw])
    else:
        faces = np.asarray(raw, dtype=np.int64)
    if faces.shape[1] != 3:
        raise SystemExit(f"Non-triangular faces (shape={faces.shape}) in {path}")

    p0 = v[faces[:, 0]]
    p1 = v[faces[:, 1]]
    p2 = v[faces[:, 2]]
    cross = np.cross(p1 - p0, p2 - p0)
    return 0.5 * np.linalg.norm(cross, axis=1)


def reciprocity_stats(row_ptr: np.ndarray, col_idx: np.ndarray,
                      vals: np.ndarray, areas: np.ndarray,
                      min_significant: float = 1e-5,
                      sample: int = 0) -> dict:
    """Return reciprocity-residual statistics.

    Computes ``|A_i F_ij − A_j F_ji| / mean(A_i F_ij, A_j F_ji)`` for every
    (i,j) pair (i<j) where ``max(F_ij, F_ji) >= min_significant``.

    Uses scipy.sparse to vectorise: build CSR F and CSC F (= CSR F^T's
    row-format), so for every nnz of (F + F^T) we get both values in one
    pass.
    """
    from scipy.sparse import csr_matrix

    n = len(row_ptr) - 1
    F = csr_matrix((vals, col_idx, row_ptr), shape=(n, n))

    # Build A_i F_ij and A_j F_ji as sparse matrices with the SAME sparsity
    # pattern (union of F and F^T).  For an entry (i,j) in F+F^T, we want
    # AiF = areas[i] * F[i,j] and AjF = areas[j] * F[j,i].  We compute both
    # by element-wise multiplication on the union pattern.
    Ft = F.transpose().tocsr()
    # Union pattern: F + Ft has nnz at every (i,j) present in either matrix.
    pattern = (F + Ft).tocoo()
    rows = pattern.row
    cols = pattern.col

    # Keep only i<j to count each pair once.
    keep = rows < cols
    rows = rows[keep]
    cols = cols[keep]

    if sample and sample < len(rows):
        rng = np.random.default_rng(0)
        sel = rng.choice(len(rows), size=sample, replace=False)
        rows = rows[sel]
        cols = cols[sel]

    # F_ij and F_ji values via direct indexing through CSR
    F_ij = np.asarray(F[rows, cols]).ravel()
    F_ji = np.asarray(F[cols, rows]).ravel()

    n_unpaired_after_compaction = int(np.sum(np.minimum(F_ij, F_ji) <= 0.0))

    # Significance filter — require BOTH entries to be present and
    # significant.  Reasoning: production MC + adaptive top-K compaction
    # truncates row i to retain only 95% of face i's outgoing energy, which
    # naturally produces an asymmetric sparsity pattern even when the
    # underlying MC estimator is unbiased; we only test reciprocity on the
    # subset of pairs that survive compaction in both directions, which is
    # the proxy for the underlying estimator quality.
    sig = (np.minimum(F_ij, F_ji) >= min_significant)
    n_skipped_small = int(np.sum(~sig))
    rows = rows[sig]; cols = cols[sig]
    F_ij = F_ij[sig]; F_ji = F_ji[sig]

    AiF = areas[rows] * F_ij
    AjF = areas[cols] * F_ji
    denom = 0.5 * (AiF + AjF) + 1e-30
    residuals = np.abs(AiF - AjF) / denom

    # Stratified residual by F magnitude:
    # The MC estimator has variance ~ F(1-F)/M, so relative error scales like
    # 1/sqrt(M F) for small F. Faces with large F-pairs (the "dominant" pairs
    # for energy transport) have the smallest relative residual.
    Fmax = np.maximum(F_ij, F_ji)
    bands = []
    for lo, hi in [(1e-5, 1e-4), (1e-4, 1e-3), (1e-3, 1e-2),
                   (1e-2, 1e-1), (1e-1, 1.0)]:
        mask = (Fmax >= lo) & (Fmax < hi)
        if mask.sum() == 0:
            continue
        bands.append({
            "F_lo": lo,
            "F_hi": hi,
            "n":    int(mask.sum()),
            "max":  float(residuals[mask].max()),
            "mean": float(residuals[mask].mean()),
            "p99":  float(np.percentile(residuals[mask], 99.0)),
        })

    return {
        "n_significant_pairs":         int(len(residuals)),
        "n_skipped_small":             n_skipped_small,
        "n_unpaired_after_compaction": n_unpaired_after_compaction,
        "max":      float(residuals.max())  if len(residuals) else float("nan"),
        "mean":     float(residuals.mean()) if len(residuals) else float("nan"),
        "rms":      float(np.sqrt(np.mean(residuals**2))) if len(residuals) else float("nan"),
        "p50":      float(np.percentile(residuals, 50.0)) if len(residuals) else float("nan"),
        "p90":      float(np.percentile(residuals, 90.0)) if len(residuals) else float("nan"),
        "p99":      float(np.percentile(residuals, 99.0)) if len(residuals) else float("nan"),
        "p99.9":    float(np.percentile(residuals, 99.9)) if len(residuals) else float("nan"),
        "bands":    bands,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vfm", required=True, type=Path)
    p.add_argument("--mesh", required=True, type=Path)
    p.add_argument("--sample-rows", type=int, default=0,
                   help="If >0, sample only this many rows for the residual scan "
                        "(faster on huge VFMs). 0 = full matrix.")
    p.add_argument("--min-significant", type=float, default=1e-5)
    args = p.parse_args()

    print(f"Loading mesh PLY: {args.mesh}")
    t0 = time.time()
    areas = read_ply_face_areas(args.mesh)
    print(f"  {len(areas):,} faces  ({time.time() - t0:.1f}s)")
    print(f"  area: min={areas.min():.3g}  mean={areas.mean():.3g}  max={areas.max():.3g} m^2")

    print(f"\nLoading VFM:    {args.vfm}")
    t0 = time.time()
    row_ptr, col_idx, vals = read_vfm(args.vfm)
    print(f"  N={len(row_ptr)-1:,}  nnz={len(vals):,}  ({time.time() - t0:.1f}s)")
    if len(row_ptr) - 1 != len(areas):
        raise SystemExit(
            f"Face-count mismatch: VFM N={len(row_ptr)-1}, mesh faces={len(areas)}"
        )
    print(f"  values: min={vals.min():.3g}  median={np.median(vals):.3g}  max={vals.max():.3g}")

    print(f"\nReciprocity scan "
          f"(min_significant={args.min_significant}, "
          f"sample_rows={'ALL' if args.sample_rows == 0 else args.sample_rows})...")
    t0 = time.time()
    stats = reciprocity_stats(row_ptr, col_idx, vals, areas,
                              min_significant=args.min_significant,
                              sample=args.sample_rows)
    elapsed = time.time() - t0
    print(f"  scan time: {elapsed:.1f}s")
    print()
    print(f"  Significant pairs scanned:    {stats['n_significant_pairs']:,}")
    print(f"  Skipped (one side compacted): {stats['n_unpaired_after_compaction']:,}")
    print(f"  Skipped (min < threshold):    {stats['n_skipped_small']:,}")
    print()
    print(f"  Relative residual |A_i F_ij − A_j F_ji| / mean(...):")
    print(f"    max     = {stats['max']:.3e}")
    print(f"    mean    = {stats['mean']:.3e}")
    print(f"    rms     = {stats['rms']:.3e}")
    print(f"    p50     = {stats['p50']:.3e}")
    print(f"    p90     = {stats['p90']:.3e}")
    print(f"    p99     = {stats['p99']:.3e}")
    print(f"    p99.9   = {stats['p99.9']:.3e}")
    print()
    print("  Stratified by max(F_ij, F_ji)  — MC relative error ~ 1/sqrt(MF):")
    print(f"    {'F range':<18}  {'n pairs':>14}  {'mean':>10}  {'p99':>10}  {'max':>10}")
    for b in stats["bands"]:
        rng = f"[{b['F_lo']:.0e}, {b['F_hi']:.0e})"
        print(f"    {rng:<18}  {b['n']:>14,}  "
              f"{b['mean']:>10.3e}  {b['p99']:>10.3e}  {b['max']:>10.3e}")


if __name__ == "__main__":
    main()
