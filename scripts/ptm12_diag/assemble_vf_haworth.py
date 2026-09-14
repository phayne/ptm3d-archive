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

Symmetrization divides each undirected pair by the 2 ray directions that were TRIED, not
by the number that happened to hit (FC-08, 2026-08-19) -- see the comment at the
symmetrize block and scripts/ptm12_diag/test_vf_assembler_gate.py.

Usage: assemble_vf_haworth.py <mesh.ply> <sweep_chunks_glob> <nrays> <min_hits> <out.csr>
  [--floor-idx floor_idx.npy]   # for the before/after floor-far-wall-aggregate report
"""
import numpy as np, glob, time, sys, argparse, os, json
from pathlib import Path

SIGMA = 5.670374419e-8  # Stefan-Boltzmann


def load_radiance_weight(path, N, albedo):
    """Per-face radiance R_j (W/m^2) for radiance-weighted pruning. Two documented
    input paths:
      - a stats DIR from a prior run: R = sigma*T_max^4 + albedo*direct_solar_max
        (reads temperature_max.bin + direct_solar_max.bin);
      - a .npy or raw float32/float64 .bin of per-face R (e.g. a cheap
        illumination-proxy: R ~= sigma*T_eq^4 + albedo*direct_solar_max).
    Returns (R, T_face_or_None); T_face (the receiving-face temperature, for the
    dT-bound) is available only from the stats-dir path."""
    p = Path(path)

    def _rd(fp):
        d = np.fromfile(fp, dtype=np.float32)
        return (d[1:] if len(d) == N + 1 else d).astype(np.float64)

    if p.is_dir():
        Tm = _rd(p / "temperature_max.bin")
        Ds = _rd(p / "direct_solar_max.bin")
        if len(Tm) != N or len(Ds) != N:
            raise SystemExit(f"radiance-weight stats dir face count {len(Tm)}/{len(Ds)} != mesh {N}")
        return SIGMA * Tm ** 4 + albedo * Ds, Tm
    if p.suffix == ".npy":
        R = np.load(p).astype(np.float64)
    else:
        R = _rd(p)
    if len(R) != N:
        raise SystemExit(f"radiance-weight length {len(R)} != mesh face count {N}")
    return R, None


def radiance_prune(rows, cols, F, N, eps, R, Tface, area, out_csr):
    """Keep entry (i,j) iff F_ij * R_j >= eps (W/m^2). Returns (keep_mask, receipt).
    Writes the receipt sidecar json AND prints it -- the user sees the trade before
    the run. Discarded contribution and dT bound are per RECEIVING face i (=rows)."""
    contrib = F * R[cols]                       # W/m^2 each partner contributes to i
    keep = contrib >= eps
    dropped = ~keep
    disc = np.zeros(N)                           # per-receiving-face discarded flux (W/m^2)
    np.add.at(disc, rows[dropped], contrib[dropped])
    Tb = np.maximum(Tface if Tface is not None else np.full(N, 40.0), 20.0)
    dT = disc / (4.0 * SIGMA * Tb ** 3)          # a-priori (loose) dT bound, K
    nnz0 = int(len(F)); nnz1 = int(keep.sum())
    receipt = {
        "radiance_prune_eps_Wm2": float(eps),
        "nnz_before": nnz0, "nnz_after": nnz1, "nnz_dropped": nnz0 - nnz1,
        "kept_fraction": round(nnz1 / max(nnz0, 1), 5),
        "predicted_speedup_nnz": round(nnz0 / max(nnz1, 1), 2),
        "discarded_contribution_Wm2": {
            "max": float(disc.max()), "p99": float(np.percentile(disc, 99)),
            "mean": float(disc.mean())},
        "dT_bound_K": {
            "max": float(dT.max()), "p99": float(np.percentile(dT, 99)),
            "mean": float(dT.mean()),
            "T_source": "face T_max (stats dir)" if Tface is not None else "reference 40 K floor",
            "caveat": "loose UPPER bound (partner MAX radiance, timing-decoupled)."},
        "MEASURED_AGT_counter_example": (
            "site A radiance eps=1e-3 (3.2x nnz) MEASURED FAIL: PSR-floor Tmax mean 3.3 K / "
            "max 22.6 K cold (rims pass, <0.6 K). Diffuse polar terrain does NOT prune "
            "safely; isolated-crater / warm-body terrain is where it pays. "
            "See ~/projects/a3gt/prune_test/verdict.json."),
    }
    with open(str(out_csr) + ".prune_receipt.json", "w") as fh:
        json.dump(receipt, fh, indent=2)
    print("[radiance-prune] RECEIPT (also written to " + str(out_csr) + ".prune_receipt.json):")
    print("  eps=%.4g W/m^2  nnz %d -> %d (keep %.2f%%, drop %d)  predicted speedup %.2fx"
          % (eps, nnz0, nnz1, 100 * nnz1 / max(nnz0, 1), nnz0 - nnz1, nnz0 / max(nnz1, 1)))
    print("  discarded contribution/face (W/m^2): max=%.3f p99=%.4f mean=%.5f"
          % (disc.max(), np.percentile(disc, 99), disc.mean()))
    print("  a-priori dT bound (K, %s): max=%.2f p99=%.3f mean=%.4f  [LOOSE upper bound]"
          % (receipt["dT_bound_K"]["T_source"], dT.max(), np.percentile(dT, 99), dT.mean()))
    print("  CAUTION: measured AGT site A 3.2x -> PSR floor mean 3.3 K / max 22.6 K (verdict.json).")
    return keep, receipt

_PLY_NP = {"float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
           "int": "<i4", "int32": "<i4", "uint": "<u4", "uint32": "<u4",
           "char": "i1", "uchar": "u1", "int8": "i1", "uint8": "u1",
           "short": "<i2", "ushort": "<u2", "int16": "<i2", "uint16": "<u2"}


def narrow_perm(p, name):
    """Store an argsort permutation as int32 when its own length fits, else keep int64.

    A permutation's values are indices into itself, so the bound is len(p), not the face
    count -- and len(p) here is a NONZERO count, which is not bounded by 2**31 (the
    worst-case A3GT site, 1e7 faces, reaches ~2.4e10 nonzeros). int32 halves the array;
    int64 is kept, with a warning on stderr, whenever it would not be exact. Indexing
    with the narrowed permutation is value-for-value identical to indexing with the
    original, so this never changes a computed number -- only the footprint."""
    if len(p) < 2 ** 31:
        return p.astype(np.int32, copy=False)
    print(f"[assemble] *** WARNING: permutation '{name}' has len={len(p)} >= 2**31, so its "
          f"indices do not fit int32; keeping int64 ({p.nbytes / 2**30:.2f} GiB). This is the "
          "documented fallback, not a failure -- the matrix is unaffected. ***", file=sys.stderr)
    return p


def read_ply_geometry(path):
    """Header-aware PLY vertex+face reader (PLY face order), reading vertices in
    their NATIVE precision widened to float64.

    PTM-23 fix: the old path used pyvista's ``mesh.points``, which VTK downcasts
    to float32; the resulting float32 areas made the VF_REPLACE geomhash
    (Sum_i area_i + (i+1)*area_i*1e-9) differ from the C++ sim's float64
    ``areas_`` by ~1e-4 (lunar polar site A: 1.37e-4 of the 1e-3 gate) -- fine here but
    liable to trip the gate on a larger/higher-area mesh. Reading float64 in the
    same PLY face order the C++ loader uses makes the two hashes agree to
    round-off. Also drops the pyvista/VTK dependency (runs in the base python)."""
    raw = open(path, "rb").read()
    he = raw.find(b"end_header")
    he = raw.index(b"\n", he) + 1
    nv = nf = 0
    cur = None
    vprops = []          # (name, np-type) in header order
    face_trailing = []   # scalar props after the vertex_indices list
    fcnt_t, fidx_t = "uchar", "int"
    for ln in raw[:he].split(b"\n"):
        p = ln.decode("latin1").split()
        if not p:
            continue
        if p[0] == "element":
            cur = p[1]
            if p[1] == "vertex":
                nv = int(p[2])
            elif p[1] == "face":
                nf = int(p[2])
        elif p[0] == "property":
            if p[1] == "list":
                if cur == "face":
                    fcnt_t, fidx_t = p[2], p[3]
            elif cur == "vertex":
                vprops.append((p[2], _PLY_NP[p[1]]))
            elif cur == "face":
                face_trailing.append(_PLY_NP[p[1]])
    vdt = np.dtype(vprops)
    V = np.frombuffer(raw, dtype=vdt, count=nv, offset=he)
    xyz = np.stack([V["x"].astype(np.float64),
                    V["y"].astype(np.float64),
                    V["z"].astype(np.float64)], axis=1)
    ffields = [("c", _PLY_NP[fcnt_t]), ("i", _PLY_NP[fidx_t], 3)]
    ffields += [("t%d" % k, t) for k, t in enumerate(face_trailing)]
    fdt = np.dtype(ffields)
    F = np.frombuffer(raw, dtype=fdt, count=nf, offset=he + nv * vdt.itemsize)
    assert (F["c"] == 3).all(), "non-triangle faces; assembler expects a triangulated mesh"
    return xyz, F["i"].astype(np.int64)


def face_areas(mesh_path):
    v, f = read_ply_geometry(mesh_path)
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
    ap.add_argument("--radiance-prune-eps", type=float,
                    default=float(os.environ.get("PTM_RADIANCE_PRUNE_EPS", "0.0")),
                    help="Radiance-weighted VF pruning threshold in W/m^2 (Paul's adjustable knob). "
                         "Keep partner j for face i iff F_ij * R_j >= eps (R_j = sigma*T_max,j^4 + "
                         "albedo*direct_solar_max,j). 0.0 (DEFAULT) = NO PRUNING, byte-identical. "
                         "Env PTM_RADIANCE_PRUNE_EPS sets the default (require-env compatible).")
    ap.add_argument("--radiance-weight", default=None,
                    help="Per-face radiance source when pruning: a prior-run stats DIR "
                         "(temperature_max.bin + direct_solar_max.bin) OR a .npy/.bin of per-face R "
                         "(W/m^2, e.g. an illumination-proxy). Required when --radiance-prune-eps>0.")
    ap.add_argument("--radiance-albedo", type=float, default=0.12,
                    help="A0 for the visible-band radiance weight (reflected solar). Default 0.12.")
    ap.add_argument("--symmetrize-divisor", choices=("tried", "hits"), default="tried",
                    help="Divisor used when averaging the two ray directions of an undirected "
                         "pair. 'tried' (DEFAULT, correct): divide by the 2 directions that were "
                         "TRIED, counting a direction that registered no hit as the measured zero "
                         "it is -- unbiased at every ray budget. 'hits' reproduces the "
                         "pre-2026-08-19 DEFECTIVE estimator (divide by the number of directions "
                         "that happened to hit), which inflates far-field entries by up to 2x. "
                         "'hits' exists ONLY so the regression gate can prove the defect; it must "
                         "never be used to build a view-factor matrix for science.")
    ap.add_argument("--allow-incomplete-sweep", action="store_true",
                    help="Assemble even when the sweep does not cover every face exactly once. "
                         "The result is BIASED on the affected faces (divisor 2 assumes both ray "
                         "directions were tried). Warns loudly; never silent.")
    a = ap.parse_args()

    t0 = time.time()
    area, ctr, faces = face_areas(a.mesh)
    N = len(area)
    # Face-bounded index arrays (lo/hi/a_idx/b_idx/rows/cols) are held as int32: each
    # element is a face index in [0, N), so int32 is exact for any N < 2**31 and halves
    # the footprint of the six largest index arrays. Fail loud rather than truncate.
    if N >= 2 ** 31:
        raise SystemExit(
            f"FATAL: mesh has N={N} faces; the int32 face-index arrays this assembler uses "
            "hold indices only up to 2**31-1. Refusing rather than truncating silently.")
    print(f"[assemble] mesh={a.mesh}  N={N} faces  total_area={area.sum():.3f} m^2")
    GEOMHASH = float(area.sum()) + float((area * np.arange(1, N + 1)).sum()) * 1e-9

    FTHR = a.min_hits / a.nrays
    files = sorted(glob.glob(a.chunks_glob))
    assert files, f"no chunk files matched {a.chunks_glob}"
    print(f"[assemble] {len(files)} chunk file(s), NRAYS={a.nrays}, MIN_HITS={a.min_hits} (FTHR={FTHR:.3g})")

    # Divisor 2 is unbiased only because BOTH ray directions of every pair were tried.
    # An assembler-side MIN-HIT filter with min_hits > 1 breaks that: it discards a
    # direction that DID fire, and the divisor then counts it as a measured zero. The
    # reverse direction of an unequal-area pair fires ~A_i/A_j times less often, so it is
    # the one systematically filtered -- an up-to-2x UNDER-estimate, the mirror image of
    # the FC-08 bias. The sweep must also have run at min_hits=1 (mc_ref_pp applies its
    # own absolute keep-rule); this script cannot verify that, so it is stated here.
    if a.min_hits > 1 and a.symmetrize_divisor == "tried":
        raise SystemExit(
            f"FATAL: min_hits={a.min_hits} with --symmetrize-divisor tried. The filter drops "
            "ray directions that DID register hits, which the divisor then counts as measured "
            "zeros -- an up-to-2x UNDER-estimate on unequal-area pairs. Pass min_hits=1 (the "
            "referee's own absolute keep-rule already applies at sweep time). See FC-08.")

    seen = np.zeros(N, dtype=np.int32)   # sweep coverage: records per source face
    Is, Js, Vs = [], [], []
    Is_old, Js_old, Vs_old = [], [], []  # also track what the OLD fractional rule would have kept
    want_old = bool(a.floor_idx)         # the OLD-rule arrays feed the --floor-idx report ONLY
    truncated = []
    for p in files:
        raw = np.frombuffer(open(p, "rb").read(), dtype=np.uint8)
        b = memoryview(raw); q = 0; L = len(raw)
        while q + 8 <= L:
            src = int(np.frombuffer(b[q:q+4], "<i4")[0]); nnz = int(np.frombuffer(b[q+4:q+8], "<i4")[0]); q += 8
            if nnz < 0 or q + nnz * 8 > L:
                truncated.append(Path(p).name)   # a killed sweep leaves a parseable short file
                break
            if 0 <= src < N:
                seen[src] += 1
            if nnz > 0:
                arr = np.frombuffer(b[q:q+nnz*8], dtype=np.dtype([("j", "<i4"), ("f", "<f4")]))
                keep = arr["f"] >= FTHR
                if keep.any():
                    Is.append(np.full(int(keep.sum()), src, dtype=np.int32))
                    Js.append(arr["j"][keep].copy()); Vs.append(arr["f"][keep].copy())
                if want_old:
                    keep_old = arr["f"] >= a.old_thr_frac
                    if keep_old.any():
                        Is_old.append(np.full(int(keep_old.sum()), src, dtype=np.int32))
                        Js_old.append(arr["j"][keep_old].copy()); Vs_old.append(arr["f"][keep_old].copy())
            q += nnz * 8

    # ---- sweep-coverage gate ----
    # Divisor 2 makes complete coverage a HARD precondition: a face missing from the sweep
    # leaves every pair that touches it with one direction instead of two, and the result is
    # silently HALVED. A face appearing twice (stale chunk files from an earlier sweep with a
    # different chunk count) DOUBLES it. Under the old divisor both self-corrected, so this
    # gate did not exist; it does now.
    n_missing = int((seen == 0).sum()); n_dup = int((seen > 1).sum())
    print(f"[assemble] sweep coverage: {int((seen == 1).sum())}/{N} faces swept exactly once "
          f"(missing {n_missing}, duplicated {n_dup}, truncated chunks {len(truncated)})")
    if n_missing or n_dup or truncated:
        msg = (f"sweep coverage is INCOMPLETE: {n_missing} face(s) never swept, {n_dup} swept "
               f"more than once, {len(truncated)} truncated chunk file(s) {truncated[:3]}. "
               "With --symmetrize-divisor tried this silently halves (or doubles) every pair "
               "touching those faces.")
        if not a.allow_incomplete_sweep:
            raise SystemExit("FATAL: " + msg + " Re-run the missing bins, or pass "
                             "--allow-incomplete-sweep if you accept the bias.")
        print("[assemble] *** WARNING: " + msg + " --allow-incomplete-sweep was given, so the "
              "matrix below IS BIASED on those faces. ***", file=sys.stderr)

    I = np.concatenate(Is); J = np.concatenate(Js); V = np.concatenate(Vs)
    del Is, Js, Vs                       # the per-chunk lists duplicate I/J/V; drop them now
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

    # ---- symmetrize: mean over the TWO ray directions that were TRIED ----
    # S_ij = A_i F_ij carries two independent unbiased Monte-Carlo estimates, one per
    # ray direction. Their plain mean -- divisor 2, with a direction that registered no
    # hit counted as the measured ZERO it is -- is unbiased at every ray budget. The
    # sweep visits every face in both roles (the run's coverage line proves it), so a
    # missing direction is a measurement, not missing data.
    #
    # FC-08 (2026-08-19): this line used to divide by the number of directions that
    # HAPPENED TO REGISTER A HIT. That conditions on detection and inflates the result
    # by E[assembled]/S_true = 2 - (p1+p2)/2, where p_k = P(direction k hits at least
    # once): 1 where both directions are well sampled, and exactly 2 in the far field
    # where neither fires reliably. Closed-form arbitration on a coaxial disk
    # (F = a^2/(a^2+H^2) = 0.250000): raw MC 0.249965, old divisor 0.437889 (+75%),
    # divisor 2 -> 0.251458. The old estimator is also ray-budget dependent; this one
    # is not. See test_vf_assembler_gate.py and
    # SESSION-NOTES-2026-08-19-fc08-far-field-vf-rootcause.md.
    S = area[I] * V.astype(np.float64)
    lo = np.minimum(I, J).astype(np.int32); hi = np.maximum(I, J).astype(np.int32)
    key = np.multiply(lo, np.int64(N), dtype=np.int64)   # int64 accumulator: range ~N**2
    key += hi
    o = narrow_perm(np.argsort(key, kind="stable"), "o")
    key = key[o]; S = S[o]; lo = lo[o]; hi = hi[o]
    bnd = np.concatenate(([0], np.nonzero(np.diff(key))[0] + 1, [len(key)]))
    gs = bnd[:-1]; ge = bnd[1:]; csum = np.concatenate(([0.0], np.cumsum(S)))
    if a.symmetrize_divisor == "hits":
        print("[assemble] *** WARNING: --symmetrize-divisor hits reproduces the pre-2026-08-19 "
              "DEFECTIVE FC-08 estimator: far-field entries are inflated by up to 2x and the "
              "result depends on the ray budget. This CSR is for gate use only -- DO NOT USE "
              "IT FOR SCIENCE. ***", file=sys.stderr)
        ndir = (ge - gs).astype(np.float64)
    else:
        ndir = 2.0
    Ssym = (csum[ge] - csum[gs]) / ndir
    print(f"[assemble] symmetrization divisor = {a.symmetrize_divisor} "
          f"({'2 (directions tried; unbiased)' if a.symmetrize_divisor == 'tried' else 'directions HIT -- DEFECTIVE, gate use only'})")
    a_idx = lo[gs]; b_idx = hi[gs]
    del S, lo, hi, key, o, csum, bnd, gs, ge   # grouping scratch; a_idx/b_idx/Ssym carry it now
    print(f"[assemble] unique undirected pairs={len(a_idx)} ({len(a_idx)/1e6:.2f}M)  {time.time()-t0:.0f}s")

    rows = np.concatenate([a_idx, b_idx]); cols = np.concatenate([b_idx, a_idx]); Sd = np.concatenate([Ssym, Ssym])
    Aend = np.concatenate([area[a_idx], area[b_idx]])
    F = Sd / Aend
    del Sd, Aend                         # both were consumed by F

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

    # ---- optional radiance-weighted pruning (Paul's adjustable knob; DEFAULT OFF) ----
    # Applied to the FINAL symmetric, SigmaF<=1-capped operator so F values are the true
    # view factors the sim uses. eps=0 -> untouched (byte-identical). Keeping true F (no
    # renormalization); discarded mass is bounded and REPORTED in the receipt.
    if a.radiance_prune_eps > 0.0:
        if not a.radiance_weight:
            raise SystemExit("--radiance-prune-eps > 0 requires --radiance-weight (stats dir or R file)")
        R, Tface = load_radiance_weight(a.radiance_weight, N, a.radiance_albedo)
        keep, _ = radiance_prune(rows, cols, F, N, a.radiance_prune_eps, R, Tface, area, a.out_csr)
        rows = rows[keep]; cols = cols[keep]; F = F[keep]
        rs = np.zeros(N); np.add.at(rs, rows, F)
        print(f"[radiance-prune] applied: kept nnz={len(F)}  post-prune SigmaF max={rs.max():.4f} "
              f"mean={rs.mean():.4f}  {time.time()-t0:.0f}s")

    o2 = narrow_perm(np.argsort(rows, kind="stable"), "o2")
    rows = rows[o2]; cols = cols[o2]; F = F[o2]
    indptr = np.zeros(N + 1, dtype=np.int64); np.add.at(indptr, rows + 1, 1); indptr = np.cumsum(indptr)
    out = a.out_csr
    with open(out, "wb") as fh:
        np.array([N], dtype=np.int32).tofile(fh)
        np.array([GEOMHASH], dtype=np.float64).tofile(fh)
        indptr.astype(np.int64).tofile(fh)
        np.array([len(F)], dtype=np.int64).tofile(fh)
        cols.astype(np.int32).tofile(fh)
        F.astype(np.float32).tofile(fh)
    print(f"[assemble] wrote {out}  nnz={len(F)}  {os.path.getsize(out)/1e9:.3f} GB  geomhash={GEOMHASH:.6f}  {time.time()-t0:.0f}s")
    np.save(str(Path(out).with_suffix(".rowsum.npy")), rs)

    # ---- provenance sidecar ----
    # The CSR header carries only N and the mesh-identity geomhash, so nothing in the file
    # itself distinguishes a matrix built with the corrected divisor from one built with the
    # pre-2026-08-19 estimator. Every .csr already on disk predates the fix and loads
    # silently. This sidecar makes the estimator, the ray budget and the sweep coverage
    # readable after the fact; a .csr WITHOUT one was built before FC-08 was fixed.
    prov = {
        "estimator": "FC-08 corrected" if a.symmetrize_divisor == "tried" else "PRE-FC-08 DEFECTIVE",
        "symmetrize_divisor": a.symmetrize_divisor,
        "symmetrize_divisor_meaning": ("mean over the 2 ray directions TRIED (unbiased)"
                                       if a.symmetrize_divisor == "tried"
                                       else "mean over the directions that HIT (inflates far field up to 2x)"),
        "nrays": int(a.nrays), "min_hits": float(a.min_hits),
        "mesh": str(a.mesh), "n_faces": int(N), "nnz": int(len(F)),
        "geomhash": float(GEOMHASH), "total_area_m2": float(area.sum()),
        "sweep_glob": str(a.chunks_glob), "sweep_chunk_files": len(files),
        "sweep_coverage_exactly_once": int((seen == 1).sum()),
        "sweep_coverage_missing": n_missing, "sweep_coverage_duplicated": n_dup,
        "sweep_truncated_chunks": len(truncated),
        "allow_incomplete_sweep": bool(a.allow_incomplete_sweep),
        "sigma_f_max": float(rs.max()), "sigma_f_mean": float(rs.mean()),
        "radiance_prune_eps": float(a.radiance_prune_eps),
        "assembler": str(Path(__file__).resolve()),
        "assembled_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(str(out) + ".provenance.json", "w") as fh:
        json.dump(prov, fh, indent=2)
    print(f"[assemble] provenance -> {out}.provenance.json  (estimator: {prov['estimator']})")


if __name__ == "__main__":
    main()
