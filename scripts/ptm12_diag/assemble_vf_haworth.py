#!/usr/bin/env python3
"""Assemble the per-partner referee sweep into a symmetric (reciprocal) sparse VF for a
Haworth-family mesh (generalizes assemble_vf2.py, which is hardcoded to the moon3d
509,337-face mesh/anchor files and is not directly reusable here).

Same algorithm as assemble_vf2.py: MIN-HIT filter (drop <MIN_HITS-ray survivors) +
symmetric ΣF<=1 cap (reciprocity-preserving congruence) + geometry-hash header. The
MIN-HIT filter here is applied on the RAW hit-fraction already written by mc_ref_pp
(which itself now applies an ABSOLUTE min-hit keep-rule, 2026-07-21 fix) -- for the
Haworth build this script is invoked with MIN_HITS matching what the sweep used (1),
so this second filter is a no-op safety net, not a re-introduction of the old bug: it
uses the SAME NRAYS the sweep actually ran at (passed explicitly, no silent default).

Usage: assemble_vf_haworth.py <mesh.ply> <sweep_chunks_glob> <nrays> <min_hits> <out.csr>
  [--floor-idx floor_idx.npy]   # for the before/after floor-far-wall-aggregate report
"""
import numpy as np, glob, time, sys, argparse
from pathlib import Path

sys.path.insert(0, "/Users/paha3326/projects/ptm/src/python")
import pyvista as pv
from moon3d_temp_3d import tri_faces


def face_areas(mesh_path):
    mesh = pv.read(mesh_path)
    f = tri_faces(mesh)
    v = np.asarray(mesh.points, dtype=np.float64)
    p0, p1, p2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
    ctr = (p0 + p1 + p2) / 3.0
    return area, ctr, f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh")
    ap.add_argument("chunks_glob")
    ap.add_argument("nrays", type=int)
    ap.add_argument("min_hits", type=float)
    ap.add_argument("out_csr")
    ap.add_argument("--floor-idx", default=None)
    ap.add_argument("--old-thr-frac", type=float, default=1e-5,
                    help="the OLD (buggy) fractional threshold, for the before/after report")
    a = ap.parse_args()

    t0 = time.time()
    area, ctr, faces = face_areas(a.mesh)
    N = len(area)
    print(f"[assemble] mesh={a.mesh}  N={N} faces  total_area={area.sum():.3f} m^2")
    GEOMHASH = float(area.sum()) + float((area * np.arange(1, N + 1)).sum()) * 1e-9

    FTHR = a.min_hits / a.nrays
    files = sorted(glob.glob(a.chunks_glob))
    assert files, f"no chunk files matched {a.chunks_glob}"
    print(f"[assemble] {len(files)} chunk file(s), NRAYS={a.nrays}, MIN_HITS={a.min_hits} (FTHR={FTHR:.3g})")

    Is, Js, Vs = [], [], []
    Is_old, Js_old, Vs_old = [], [], []  # also track what the OLD fractional rule would have kept
    for p in files:
        raw = np.frombuffer(open(p, "rb").read(), dtype=np.uint8)
        b = memoryview(raw); q = 0; L = len(raw)
        while q + 8 <= L:
            src = int(np.frombuffer(b[q:q+4], "<i4")[0]); nnz = int(np.frombuffer(b[q+4:q+8], "<i4")[0]); q += 8
            if nnz > 0:
                arr = np.frombuffer(b[q:q+nnz*8], dtype=np.dtype([("j", "<i4"), ("f", "<f4")]))
                keep = arr["f"] >= FTHR
                if keep.any():
                    Is.append(np.full(int(keep.sum()), src, dtype=np.int32))
                    Js.append(arr["j"][keep].copy()); Vs.append(arr["f"][keep].copy())
                keep_old = arr["f"] >= a.old_thr_frac
                if keep_old.any():
                    Is_old.append(np.full(int(keep_old.sum()), src, dtype=np.int32))
                    Js_old.append(arr["j"][keep_old].copy()); Vs_old.append(arr["f"][keep_old].copy())
            q += nnz * 8
    I = np.concatenate(Is); J = np.concatenate(Js); V = np.concatenate(Vs)
    print(f"[assemble] NEW rule directed nnz={len(I)} ({len(I)/1e6:.2f}M)  {time.time()-t0:.0f}s")

    # ---- before/after floor-far-wall-aggregate report (Sagan's requested checkpoint number) ----
    if a.floor_idx:
        floor_idx = set(np.load(a.floor_idx).tolist())
        is_floor = np.isin(I, list(floor_idx))
        dist = np.linalg.norm(ctr[I] - ctr[J], axis=1)
        far = is_floor & (dist >= 15000)
        new_far_sum = V[far].sum()
        I_old = np.concatenate(Is_old); J_old = np.concatenate(Js_old); V_old = np.concatenate(Vs_old)
        is_floor_old = np.isin(I_old, list(floor_idx))
        dist_old = np.linalg.norm(ctr[I_old] - ctr[J_old], axis=1)
        far_old = is_floor_old & (dist_old >= 15000)
        old_far_sum = V_old[far_old].sum()
        print(f"[floor-far-wall BEFORE/AFTER] OLD-rule (F>={a.old_thr_frac:.1g}) kept aggregate ΣF(far)={old_far_sum:.5f}  "
              f"NEW-rule (min_hits={a.min_hits}) kept aggregate ΣF(far)={new_far_sum:.5f}  "
              f"ratio new/old={new_far_sum/max(old_far_sum,1e-12):.2f}x")

    # ---- symmetrize (mean of available directions) ----
    S = area[I] * V.astype(np.float64)
    lo = np.minimum(I, J).astype(np.int64); hi = np.maximum(I, J).astype(np.int64)
    key = lo * np.int64(N) + hi
    o = np.argsort(key, kind="stable"); key = key[o]; S = S[o]; lo = lo[o]; hi = hi[o]
    bnd = np.concatenate(([0], np.nonzero(np.diff(key))[0] + 1, [len(key)]))
    gs = bnd[:-1]; ge = bnd[1:]; csum = np.concatenate(([0.0], np.cumsum(S)))
    Ssym = (csum[ge] - csum[gs]) / (ge - gs).astype(np.float64)
    a_idx = lo[gs]; b_idx = hi[gs]
    print(f"[assemble] unique undirected pairs={len(a_idx)} ({len(a_idx)/1e6:.2f}M)  {time.time()-t0:.0f}s")

    rows = np.concatenate([a_idx, b_idx]); cols = np.concatenate([b_idx, a_idx]); Sd = np.concatenate([Ssym, Ssym])
    Aend = np.concatenate([area[a_idx], area[b_idx]])
    F = Sd / Aend

    # ---- symmetric ΣF<=1 cap (reciprocity-preserving congruence) ----
    for it in range(40):
        rs = np.zeros(N); np.add.at(rs, rows, F)
        if rs.max() <= 1.0 + 1e-9:
            break
        c = np.ones(N); over = rs > 1.0; c[over] = 1.0 / rs[over]
        F = F * np.sqrt(c[rows] * c[cols])
    rs = np.zeros(N); np.add.at(rs, rows, F)
    print(f"[assemble] symmetric-cap sweeps={it+1} -> SigmaF max={rs.max():.4f} (#>1={(rs>1.0+1e-6).sum()}) "
          f"mean={rs.mean():.4f}  {time.time()-t0:.0f}s")

    o2 = np.argsort(rows, kind="stable"); rows = rows[o2]; cols = cols[o2]; F = F[o2]
    indptr = np.zeros(N + 1, dtype=np.int64); np.add.at(indptr, rows + 1, 1); indptr = np.cumsum(indptr)
    out = a.out_csr
    with open(out, "wb") as fh:
        np.array([N], dtype=np.int32).tofile(fh)
        np.array([GEOMHASH], dtype=np.float64).tofile(fh)
        indptr.astype(np.int64).tofile(fh)
        np.array([len(F)], dtype=np.int64).tofile(fh)
        cols.astype(np.int32).tofile(fh)
        F.astype(np.float32).tofile(fh)
    import os
    print(f"[assemble] wrote {out}  nnz={len(F)}  {os.path.getsize(out)/1e9:.3f} GB  geomhash={GEOMHASH:.6f}  {time.time()-t0:.0f}s")
    np.save(str(Path(out).with_suffix(".rowsum.npy")), rs)


if __name__ == "__main__":
    main()
