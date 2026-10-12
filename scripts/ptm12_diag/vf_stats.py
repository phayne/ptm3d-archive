#!/usr/bin/env python3
"""View-factor matrix statistics (Moon3D R1, change C4).

Reviewer 1 asked how sparse the radiation operator is, how many partners each
facet keeps, and what "total view factor" means when not every pair is
evaluated. Nothing on disk answered that: the CSR header carries only N and a
mesh hash, and the provenance sidecar only N, nnz and the row-sum mean/max.
This module computes the numbers once and writes them next to the matrix:

    <csr>.vf_stats.json

Called by assemble_vf_haworth.py after it writes a CSR (stats only: the CSR
bytes are unchanged), and usable standalone on any existing CSR:

    vf_stats.py <csr> [--nrays NR] [--sweep-logs 'dir/log_*.txt'] [--spectral]

Definitions (all stated in the JSON so a reader never has to guess):
  sum_f[i]      = sum_j F_ij over the STORED entries of row i (terrain view factor)
  f_sky[i]      = 1 - sum_f[i]. Space is at 0 K; there is no separate "total
                  view factor". With mc_ref_pp (MIN_HITS=1, divisor 2) the row
                  sum is the terrain-hit fraction of NR cosine-weighted rays, so
                  it is unbiased however many partners are kept.
  saturation    = (nnz/N) / (2 NR <sum_f>): stored partners per row over the
                  most a row could hold if every terrain ray found a new partner
                  in both sweep directions. Near 0 = every visible partner is
                  already detected many times (nnz is set by geometry and grows
                  with the visible set); near 1 = rows are capped by the ray
                  budget. A HINT, not a classification: the thresholds below are
                  engineering choices pending Fermi.
  directed_saturation = partners_kept / terrain_hits from the sweep STATS lines
                  (per launch ray, how often a hit found a partner not seen
                  before), when sweep logs are given.

numpy only; scipy is used for the optional spectral radius if present.
"""
import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np

STATS_RE = re.compile(r"^# STATS (\{.*\})\s*$")
SCHEMA = "ptm.vf_stats/1"
EPS_DEFAULT = 0.95
SPECTRAL_NNZ_CAP = 50_000_000   # in-assembler default; standalone --spectral ignores it


def read_csr(path):
    """CSR as written by assemble_vf_haworth.py / read by PTM_HMATRIX_VF_REPLACE:
    int32 n, float64 geomhash, int64 indptr[n+1], int64 nnz, int32 cols[nnz],
    float32 data[nnz]. Fails loudly on a size mismatch."""
    with open(path, "rb") as fh:
        n = int(np.fromfile(fh, "<i4", 1)[0])
        geomhash = float(np.fromfile(fh, "<f8", 1)[0])
        indptr = np.fromfile(fh, "<i8", n + 1)
        nnz = int(np.fromfile(fh, "<i8", 1)[0])
        cols = np.fromfile(fh, "<i4", nnz)
        data = np.fromfile(fh, "<f4", nnz)
    expect = 4 + 8 + 8 * (n + 1) + 8 + 8 * nnz
    actual = os.path.getsize(path)
    if len(cols) != nnz or len(data) != nnz or expect != actual or int(indptr[-1]) != nnz:
        raise SystemExit(f"FATAL: {path} is not a well-formed CSR (expected {expect} B, "
                         f"file {actual} B, indptr[-1]={int(indptr[-1])}, nnz={nnz})")
    return n, geomhash, indptr, cols, data


def parse_sweep_stats(pattern):
    """Sum the '# STATS {...}' lines mc_ref_pp prints (one per chunk process).
    Returns (totals dict, n_lines, n_files). Files without a STATS line (an older
    referee, or a killed chunk) are COUNTED and reported, never skipped silently."""
    files = sorted(glob.glob(pattern))
    tot, n_lines, missing = {}, 0, []
    max_kept, wall_max = 0, 0.0
    for p in files:
        found = False
        with open(p, errors="replace") as fh:
            for ln in fh:
                m = STATS_RE.match(ln.strip())
                if not m:
                    continue
                d = json.loads(m.group(1))
                found = True
                n_lines += 1
                for k, v in d.items():
                    if k in ("n_faces_mesh", "nrays", "min_hits"):
                        tot.setdefault(k, set()).add(v)
                    elif k == "max_partners_kept":
                        max_kept = max(max_kept, v)
                    elif k == "wall_s":
                        tot["wall_s_sum"] = tot.get("wall_s_sum", 0.0) + v
                        wall_max = max(wall_max, v)
                    else:
                        tot[k] = tot.get(k, 0) + v
        if not found:
            missing.append(os.path.basename(p))
    for k in ("n_faces_mesh", "nrays", "min_hits"):
        if k in tot:
            vals = sorted(tot[k])
            tot[k] = vals[0] if len(vals) == 1 else vals   # a list here means MIXED sweeps
    tot["max_partners_kept"] = max_kept
    tot["wall_s_max_chunk"] = wall_max
    return tot, n_lines, len(files), missing


def _pct(x, qs=(5, 50, 95)):
    if len(x) == 0:
        return {}
    out = {f"p{q}": float(np.percentile(x, q)) for q in qs}
    out.update(min=float(np.min(x)), max=float(np.max(x)), mean=float(np.mean(x)))
    return out


def spectral_radius(indptr, cols, data, n, iters=200, rtol=1e-6):
    """Power iteration for rho(F). F = A^-1 S with S symmetric (reciprocity), so F
    is similar to a symmetric matrix and its eigenvalues are real; the dominant one
    is the Perron root for a nonnegative F. Advisory, as the engine's own estimate
    (radiation_exchange.h SpectralRadiusEstimate) is."""
    try:
        import scipy.sparse as sp
        M = sp.csr_matrix((data.astype(np.float64), cols, indptr), shape=(n, n))
        mv = M.dot
    except ImportError:   # numpy fallback: row-wise reduce
        rows = np.repeat(np.arange(n), np.diff(indptr))
        d64 = data.astype(np.float64)

        def mv(x):
            y = np.zeros(n)
            np.add.at(y, rows, d64 * x[cols])
            return y
    x = np.ones(n) / np.sqrt(n)
    lam_prev, lam = 0.0, 0.0
    for k in range(iters):
        y = mv(x)
        lam = float(np.linalg.norm(y))
        if lam == 0.0:
            return {"ok": False, "reason": "F x = 0"}
        x = y / lam
        if k > 5 and abs(lam - lam_prev) <= rtol * lam:
            break
        lam_prev = lam
    return {"ok": True, "rho": lam, "iterations": k + 1, "rtol": rtol}


def compute(indptr, cols, data, n, area=None, nrays=None, min_hits=None,
            cap_sweeps=None, rows_scaled=None, sweep_logs=None, csr_path=None,
            spectral="auto", eps=EPS_DEFAULT):
    t0 = time.time()
    nnz = int(len(data))
    per_row = np.diff(indptr).astype(np.int64)
    # Row sums by a float64 prefix sum: one 8 B/nnz temporary and no row-index
    # array, because this runs inside the assembler right after its memory peak.
    # Absolute error ~1e-16 x (total mass), i.e. ~1e-11 at N = 5e5.
    cs = np.zeros(nnz + 1)
    np.cumsum(data, dtype=np.float64, out=cs[1:])
    rs = cs[indptr[1:]] - cs[indptr[:-1]]
    del cs
    fsky = 1.0 - rs
    s = {
        "schema": SCHEMA,
        "definitions": {
            "sum_f": "sum_j F_ij over stored entries (terrain view factor of face i)",
            "f_sky": "1 - sum_f (space at 0 K; no separate total view factor exists)",
            "saturation": "(nnz/N) / (2 NR <sum_f>); ~0 detect-all (nnz set by geometry), "
                          "~1 ray-budget-capped. Hint only.",
            "directed_saturation": "sweep partners_kept / terrain_hits",
        },
        "csr": str(csr_path) if csr_path else None,
        "n_faces": int(n),
        "nnz": nnz,
        "nnz_per_face_mean": nnz / n if n else 0.0,
        "density_nnz_over_n2": nnz / float(n) ** 2 if n else 0.0,
        "zeros_fraction": 1.0 - nnz / float(n) ** 2 if n else 0.0,
        "nnz_per_face": _pct(per_row),
        "empty_rows": int((per_row == 0).sum()),
        "sum_f": _pct(rs),
        "f_sky": _pct(fsky),
        "rows_sum_f_gt_1": int((rs > 1.0 + 1e-6).sum()),
        "min_entry": float(data.min()) if nnz else 0.0,
        "max_entry": float(data.max()) if nnz else 0.0,
        "nrays": nrays,
        "min_hits": min_hits,
        "symmetric_cap_sweeps": cap_sweeps,
        "symmetric_cap_rows_scaled": rows_scaled,
        "csr_bytes": os.path.getsize(csr_path) if csr_path and os.path.exists(csr_path) else None,
        # GPU radiosity holds int32 cols + float32 values + int64 row pointers per
        # device, replicated on every die of the pool (radiosity_gpu_pool.h).
        "gpu_replica_bytes_est": 8 * nnz + 8 * (n + 1),
    }
    if area is not None:
        area = np.asarray(area, dtype=np.float64)
        s["sum_f"]["area_weighted_mean"] = float((area * rs).sum() / area.sum())
        s["f_sky"]["area_weighted_mean"] = float((area * fsky).sum() / area.sum())
    if nrays:
        mean_sf = float(rs.mean())
        sat = (nnz / n) / (2.0 * nrays * mean_sf) if mean_sf > 0 else None
        s["saturation"] = sat
        s["saturation_hint"] = (None if sat is None else
                                "detect-all" if sat < 0.1 else
                                "ray-budget-capped" if sat > 0.5 else "transitional")
    if sweep_logs:
        tot, n_lines, n_files, missing = parse_sweep_stats(sweep_logs)
        sw = {"glob": sweep_logs, "files": n_files, "stats_lines": n_lines,
              "files_without_stats": missing, **tot}
        rt = tot.get("rays_total", 0)
        if rt:
            for k in ("terrain_hits", "ring_blocked", "backface_blocked", "sky"):
                sw[k + "_fraction"] = tot.get(k, 0) / rt
        if tot.get("terrain_hits"):
            s["directed_saturation"] = tot.get("partners_kept", 0) / tot["terrain_hits"]
        if missing:
            print(f"[vf_stats] WARNING: {len(missing)} of {n_files} sweep log(s) carry no STATS "
                  f"line (older referee or killed chunk): {missing[:3]}", file=sys.stderr)
        s["sweep"] = sw
    do_spec = (spectral is True) or (spectral == "auto" and nnz <= SPECTRAL_NNZ_CAP)
    if do_spec and nnz:
        sr = spectral_radius(indptr, cols, data, n)
        if sr.get("ok"):
            rho = sr["rho"]
            sr["eps"] = eps
            sr["picard_rate_bound"] = eps * rho / (1.0 - (1.0 - eps) * rho)
        s["spectral"] = sr
    else:
        s["spectral"] = {"ok": False, "reason": (
            f"skipped: nnz {nnz} > in-assembler cap {SPECTRAL_NNZ_CAP}; run "
            "vf_stats.py <csr> --spectral offline" if nnz else "empty matrix")}
    s["compute_seconds"] = time.time() - t0
    return s


def banner(s):
    """One human line for logs, matching the engine's [VF-STATS] banner fields."""
    sf, fk, pr = s["sum_f"], s["f_sky"], s["nnz_per_face"]
    sat = s.get("saturation")
    return (f"[VF-STATS] N={s['n_faces']} nnz={s['nnz']} nnz/face={s['nnz_per_face_mean']:.1f} "
            f"(p5/p50/p95/max {pr.get('p5', 0):.0f}/{pr.get('p50', 0):.0f}/{pr.get('p95', 0):.0f}/"
            f"{pr.get('max', 0):.0f}) nnz/N^2={s['density_nnz_over_n2']:.3e} "
            f"sumF p5/p50/p95/max {sf.get('p5', 0):.4f}/{sf.get('p50', 0):.4f}/"
            f"{sf.get('p95', 0):.4f}/{sf.get('max', 0):.4f} "
            f"Fsky p5/p50/p95 {fk.get('p5', 0):.4f}/{fk.get('p50', 0):.4f}/{fk.get('p95', 0):.4f}"
            + (f" saturation={sat:.3g} ({s['saturation_hint']})" if sat is not None else ""))


def write_sidecar(csr_path, stats):
    out = str(csr_path) + ".vf_stats.json"
    with open(out, "w") as fh:
        json.dump(stats, fh, indent=2, sort_keys=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csr")
    ap.add_argument("--nrays", type=int, default=None,
                    help="sweep ray budget (default: from <csr>.provenance.json if present)")
    ap.add_argument("--sweep-logs", default=None,
                    help="glob of mc_ref_pp chunk logs (stderr) carrying '# STATS' lines")
    ap.add_argument("--mesh", default=None, help="PLY, for area-weighted means")
    ap.add_argument("--spectral", action="store_true", help="power-iterate rho(F) at any size")
    ap.add_argument("--out", default=None, help="JSON path (default <csr>.vf_stats.json)")
    a = ap.parse_args()
    n, _, indptr, cols, data = read_csr(a.csr)
    nrays, min_hits = a.nrays, None
    prov = a.csr + ".provenance.json"
    if os.path.exists(prov):
        p = json.load(open(prov))
        nrays = nrays or p.get("nrays")
        min_hits = p.get("min_hits")
    elif nrays is None:
        print("[vf_stats] WARNING: no --nrays and no provenance sidecar; saturation not computed",
              file=sys.stderr)
    area = None
    if a.mesh:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from assemble_vf_haworth import face_areas
        area = face_areas(a.mesh)[0]
        if len(area) != n:
            raise SystemExit(f"FATAL: mesh has {len(area)} faces, CSR {n}")
    s = compute(indptr, cols, data, n, area=area, nrays=nrays, min_hits=min_hits,
                sweep_logs=a.sweep_logs, csr_path=a.csr,
                spectral=True if a.spectral else "auto")
    out = a.out or write_sidecar(a.csr, s)
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(s, fh, indent=2, sort_keys=True)
    print(banner(s))
    print(f"[vf_stats] wrote {out}")


if __name__ == "__main__":
    main()
