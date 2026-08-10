#!/usr/bin/env python3
"""
Top-K compaction η-sensitivity study.

Builds a spherical-cap bowl-crater mesh (same geometry as §8.1 and
``bowl_crater_3d_figure.py``), computes the per-face dense view-factor row,
and reports the post-compaction neighbor count K(η) for each face as a
function of the captured-energy target η ∈ [0.90, 0.99].

This validates the manuscript's qualitative claim (§6.2.1):

    "Reducing η to 99% tightens this loss but increases K by roughly 3×
    and the storage and matrix-vector cost in proportion."

Output: a small table (printed to stdout, optionally written to
``tmp/topk_eta_table.txt``) summarizing:

  η  |  ⟨K⟩  |  Kmax  |  total nnz (K-pairs)  |  ⟨1 − Σ F_ij^kept⟩

The last column is the average "energy budget leaked to the sky aperture",
which is the floor of the lossy compaction.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from bowl_crater_3d_figure import (
    build_crater_mesh, face_geometry, compute_view_factors,
)


def topk_stats(F: np.ndarray, eta: float) -> dict:
    """Per-row top-K compaction at energy fraction η; returns aggregated stats."""
    N = F.shape[0]
    Ks = np.empty(N, dtype=np.int64)
    leaked = np.empty(N, dtype=np.float64)
    for i in range(N):
        row = F[i]
        nz_mask = row > 0
        nz = row[nz_mask]
        if len(nz) == 0:
            Ks[i] = 0
            leaked[i] = 0.0
            continue
        s = np.sort(nz)[::-1]
        cum = np.cumsum(s)
        target = eta * cum[-1]
        k = int(np.searchsorted(cum, target, side="left") + 1)
        k = min(k, len(s))
        Ks[i] = k
        # Energy leaked to sky aperture: (1−η)·Σ F_terrain on this face
        leaked[i] = cum[-1] - cum[k - 1]
    return {
        "eta": eta,
        "K_mean": float(Ks.mean()),
        "K_max":  int(Ks.max()),
        "K_p99":  int(np.percentile(Ks, 99)),
        "total_nnz": int(Ks.sum()),
        "leaked_mean_pct": float(100.0 * leaked.mean()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dd", type=float, default=0.1,
                   help="Crater depth/diameter ratio")
    p.add_argument("--n-rings", type=int, default=60)
    p.add_argument("--n-az", type=int, default=120)
    p.add_argument("--etas", nargs="+", type=float,
                   default=[0.50, 0.80, 0.90, 0.95, 0.99, 0.999])
    p.add_argument("--out-txt", default="tmp/topk_eta_table.txt")
    args = p.parse_args()

    verts, faces, _, _ = build_crater_mesh(
        depth_to_diameter=args.dd, diameter=1.0,
        n_rings=args.n_rings, n_az=args.n_az,
    )
    cent, n, area = face_geometry(verts, faces)
    print(f"Built crater: {len(faces)} faces, d/D={args.dd}")
    print("Computing dense view-factor matrix...", flush=True)
    F = compute_view_factors(cent, n, area)
    avg_terrain = float((F.sum(axis=1)).mean())
    print(f"  mean Σ_j F_ij (terrain) = {avg_terrain:.4f}")
    print(f"  mean F_sky (mesh)       = {1.0 - avg_terrain:.4f}\n")

    rows = []
    for eta in args.etas:
        s = topk_stats(F, eta)
        rows.append(s)

    header = f"  {'η':>6}  {'⟨K⟩':>8}  {'p99 K':>8}  {'K_max':>8}  " \
             f"{'total nnz':>12}  {'⟨leaked⟩ (% of Σ_terrain)':>26}"
    sep = "  " + "-" * (len(header) - 2)
    print(header)
    print(sep)
    lines = [header, sep]
    for r in rows:
        line = (f"  {r['eta']:>6.3f}  {r['K_mean']:>8.2f}  "
                f"{r['K_p99']:>8d}  {r['K_max']:>8d}  {r['total_nnz']:>12,}  "
                f"{r['leaked_mean_pct']:>26.3f}")
        print(line)
        lines.append(line)

    # K scaling: K(η=0.99) / K(η=0.95)
    K95 = next((r["K_mean"] for r in rows if abs(r["eta"] - 0.95) < 1e-6), None)
    K99 = next((r["K_mean"] for r in rows if abs(r["eta"] - 0.99) < 1e-6), None)
    if K95 and K99:
        print(f"\n  K(η=0.99) / K(η=0.95) = {K99/K95:.2f}×")

    out_path = Path(args.out_txt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
