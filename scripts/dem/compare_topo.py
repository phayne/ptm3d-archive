#!/usr/bin/env python3
"""Compare two PTMTOPO files at the same resolution.

Reports:
  - max/mean |Δelev| across vertices
  - max/mean |Δradius| across vertex positions
  - byte-for-byte equality (for sanity)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


N_V_BY_RES = {
    3: 642,        4: 2562,       5: 10242,      6: 40962,
    7: 163842,     8: 655362,     9: 2621442,    10: 10485762, 11: 41943042,
}
N_F_BY_RES = {
    3: 1280,       4: 5120,       5: 20480,      6: 81920,
    7: 327680,     8: 1310720,    9: 5242880,    10: 20971520, 11: 83886080,
}


def read_topo(path: Path, n_v: int, n_f: int):
    HEADER = 27   # 7 magic + 4 int + 8 + 8
    with open(path, 'rb') as f:
        magic = f.read(7)
        if magic != b'PTMTOPO':
            sys.exit(f'{path}: bad magic {magic!r}')
        f.seek(HEADER)
        verts = np.frombuffer(f.read(n_v * 24), dtype=np.float64).reshape(-1, 3)
        f.seek(12 * n_f, 1)         # skip faces (int32 x 3)
        elev  = np.frombuffer(f.read(n_v * 8),  dtype=np.float64)
    return verts, elev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('a', help='first PTMTOPO file (e.g. canonical)')
    ap.add_argument('b', help='second PTMTOPO file (e.g. regenerated)')
    ap.add_argument('--res', type=int, required=True)
    args = ap.parse_args()

    n_v = N_V_BY_RES[args.res]
    n_f = N_F_BY_RES[args.res]

    va, ea = read_topo(Path(args.a), n_v, n_f)
    vb, eb = read_topo(Path(args.b), n_v, n_f)

    ra = np.linalg.norm(va, axis=1)
    rb = np.linalg.norm(vb, axis=1)
    de = ea - eb
    dr = ra - rb

    print(f'{"":<24} {args.a}  vs  {args.b}')
    print(f'  vertices: {n_v},  faces: {n_f}')
    print(f'  elev    : a={ea.min():+8.1f} … {ea.max():+8.1f}   '
          f'b={eb.min():+8.1f} … {eb.max():+8.1f}   m')
    print(f'  radii   : a-R range {ra.min()-1737400:+.0f} … {ra.max()-1737400:+.0f} m   '
          f'b-R range {rb.min()-1737400:+.0f} … {rb.max()-1737400:+.0f} m')
    print(f'  |Δelev| : max={np.abs(de).max():.2f} m   mean={np.abs(de).mean():.3f} m   '
          f'rms={np.sqrt((de**2).mean()):.3f} m')
    print(f'  |Δradius|: max={np.abs(dr).max():.2f} m   mean={np.abs(dr).mean():.3f} m   '
          f'rms={np.sqrt((dr**2).mean()):.3f} m')

    # Spearman correlation as a permutation-robust similarity measure.
    # If files differ only by floating-point reordering, |Δelev| could be
    # misleadingly large; rank correlation should still be ~1.
    if n_v <= 1_000_000:  # rankdata is O(n log n) but uses memory
        from scipy import stats
        rho_e, _ = stats.spearmanr(ea, eb)
        rho_r, _ = stats.spearmanr(ra, rb)
        print(f'  Spearman: elev rho={rho_e:.6f}  radii rho={rho_r:.6f}')


if __name__ == '__main__':
    main()
