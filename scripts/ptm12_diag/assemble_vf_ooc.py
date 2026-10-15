#!/usr/bin/env python3
"""Out-of-core twin of assemble_vf_haworth.py (Ada, 2026-10-15): same inputs, same CLI, and a
BYTE-IDENTICAL output contract (<csr>, <csr>.rowsum.npy, <csr>.provenance.json keys), with peak
RAM set by --ram-gib instead of by nnz. Built for the sky-side Moon3D 509k matrix (PTM-732:
~5.5e9 directed entries, ~9.5e9 nnz, ~76 GB CSR), which the in-core assembler would need
~570 GB of RAM to build (60 B/nnz measured).

How byte-identity is kept (every floating-point operation is the in-core one, in the in-core
order; only the data movement changes):
  A  count pass: per-lo directed-entry counts and sweep coverage (the in-core gate, verbatim).
     lo-range BUCKETS are cut so each fits the RAM budget.
  B  spill pass: each chunk file is parsed exactly as in-core (keep = f >= FTHR, S = area[I]*V),
     lo/hi/S are appended to their bucket's spill files IN FILE-READ ORDER, so a stable sort
     inside a bucket equals the in-core global stable sort restricted to that key range.
  C  symmetrize per bucket in key order. The in-core prefix sum csum is GLOBAL; it is carried
     across buckets exactly (carry added to the bucket's first element, then a sequential
     cumsum -- the same additions in the same order). Pairs (a, b, Ssym) go to pair spill files.
  D  cap: the in-core row sums are np.add.at(rs, concat(a-halves, b-halves), F) -- every a-half
     contribution in global pair order, THEN every b-half one. Reproduced with np.add.at over the
     a-halves of all buckets in order, then the b-halves (bincount would change the summation
     order and is NOT used). F = Ssym / area[...] exactly as in-core; scaled F is spilled only if
     the cap actually scales.
  E  write in row blocks. In-core, a stable argsort of rows over concat(a-half, b-half) puts, in
     each row r, its a-half entries (cols > r ascending) before its b-half ones (cols < r
     ascending). Here the b-half is transposed into row-block spill files in global pair order
     and merged after the a-half with a stable sort per block: the same sequence. cols / float32
     values are written at their final offsets, indptr from the integer per-row counts.
Scratch disk (default next to the output): ~16 B per directed entry, then ~16 B per pair plus a
12 B per pair transpose, each released as the next stage consumes it (see the plan banner).
Not supported here (FATAL, use the in-core assembler): --floor-idx, --radiance-prune-eps > 0.
"""
import argparse, glob, json, os, shutil, struct, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import assemble_vf_haworth as IC   # face_areas, sweep_orientation (shared, not copied)

GiB = 1024 ** 3
B_DIRECTED = 80     # peak RAM bytes per directed entry inside one bucket (symmetrize stage)
B_ROW = 40          # peak RAM bytes per output entry inside one row block (write stage)


def log(msg):
    print(f"[assemble-ooc] {msg}", flush=True)


def parse_file(p, N, FTHR, seen, truncated):
    """One chunk file, parsed exactly as the in-core loop does. Returns I, J, V (kept entries,
    file order). Updates seen / truncated in place."""
    raw = np.frombuffer(open(p, "rb").read(), dtype=np.uint8)
    b = memoryview(raw); q = 0; L = len(raw)
    Is, Js, Vs = [], [], []
    while q + 8 <= L:
        src = int(np.frombuffer(b[q:q+4], "<i4")[0]); nnz = int(np.frombuffer(b[q+4:q+8], "<i4")[0]); q += 8
        if nnz < 0 or q + nnz * 8 > L:
            truncated.append(Path(p).name)
            break
        if 0 <= src < N:
            seen[src] += 1
        if nnz > 0:
            arr = np.frombuffer(b[q:q+nnz*8], dtype=np.dtype([("j", "<i4"), ("f", "<f4")]))
            keep = arr["f"] >= FTHR
            if keep.any():
                Is.append(np.full(int(keep.sum()), src, dtype=np.int32))
                Js.append(arr["j"][keep].copy()); Vs.append(arr["f"][keep].copy())
        q += nnz * 8
    if not Is:
        e = np.zeros(0, np.int32)
        return e, e.copy(), np.zeros(0, np.float32)
    return np.concatenate(Is), np.concatenate(Js), np.concatenate(Vs)


def cut_blocks(counts, budget, align=None):
    """Contiguous index ranges [B_k, B_k+1) whose summed counts stay <= budget (a single index
    above budget gets its own range). If align is given, ranges never straddle its boundaries."""
    N = len(counts)
    cuts = [0]; acc = 0
    hard = set(int(x) for x in (align if align is not None else []))
    for i in range(N):
        c = int(counts[i])
        if (i in hard and i != cuts[-1]) or (acc + c > budget and i != cuts[-1]):
            cuts.append(i); acc = 0
        acc += c
    if cuts[-1] != N:
        cuts.append(N)
    return np.array(cuts, dtype=np.int64)


class Spill:
    """Append-only typed spill files; one set per bucket."""
    def __init__(self, root, name, fields):
        self.root = Path(root); self.name = name; self.fields = fields
    def path(self, k, f):
        return self.root / f"{self.name}_{k:04d}.{f}"
    def append(self, k, **arrs):
        for f, dt in self.fields:
            with open(self.path(k, f), "ab") as fh:
                fh.write(np.ascontiguousarray(arrs[f], dtype=dt).tobytes())
    def load(self, k):
        out = {}
        for f, dt in self.fields:
            p = self.path(k, f)
            out[f] = np.fromfile(p, dtype=dt) if p.exists() else np.zeros(0, dt)
        return out
    def drop(self, k):
        for f, _ in self.fields:
            p = self.path(k, f)
            if p.exists():
                p.unlink()
    def bytes(self):
        return sum(p.stat().st_size for p in self.root.glob(f"{self.name}_*"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mesh"); ap.add_argument("chunks_glob"); ap.add_argument("nrays", type=int)
    ap.add_argument("min_hits", type=float); ap.add_argument("out_csr")
    ap.add_argument("--symmetrize-divisor", choices=("tried", "hits", "pooled"), default="tried")
    ap.add_argument("--sweep-logs", default=None)
    ap.add_argument("--allow-incomplete-sweep", action="store_true")
    ap.add_argument("--floor-idx", default=None)
    ap.add_argument("--radiance-prune-eps", type=float, default=float(os.environ.get("PTM_RADIANCE_PRUNE_EPS", "0.0")))
    ap.add_argument("--ram-gib", type=float, default=96.0, help="peak RAM budget for one bucket / row block")
    ap.add_argument("--scratch", default=None, help="spill directory (default: <out_csr>.ooc_scratch)")
    ap.add_argument("--keep-scratch", action="store_true")
    ap.add_argument("--vf-stats", choices=("full", "nospectral", "off"), default="nospectral",
                    help="C4 stats sidecar on the written CSR (memory-mapped). 'full' adds the spectral "
                         "radius, which needs ~24 B/nnz of RAM.")
    a = ap.parse_args()
    if a.floor_idx:
        raise SystemExit("FATAL: --floor-idx is not supported out of core; use assemble_vf_haworth.py")
    if a.symmetrize_divisor == "pooled":
        raise SystemExit("FATAL: --symmetrize-divisor pooled is not implemented out of core; use assemble_vf_haworth.py")
    if a.radiance_prune_eps > 0.0:
        raise SystemExit("FATAL: radiance pruning is not supported out of core; use assemble_vf_haworth.py")
    if a.min_hits > 1 and a.symmetrize_divisor == "tried":
        raise SystemExit(f"FATAL: min_hits={a.min_hits} with --symmetrize-divisor tried (FC-08; see the in-core assembler)")
    t0 = time.time()
    out = Path(a.out_csr)
    scratch = Path(a.scratch) if a.scratch else Path(str(out) + ".ooc_scratch")
    if scratch.exists() and any(scratch.iterdir()):
        raise SystemExit(f"FATAL: scratch {scratch} is not empty (a previous run?); remove it first")
    scratch.mkdir(parents=True, exist_ok=True)

    area, _, _ = IC.face_areas(a.mesh)
    N = len(area)
    if N >= 2 ** 31:
        raise SystemExit(f"FATAL: N={N} >= 2**31")
    log(f"mesh={a.mesh}  N={N} faces  total_area={area.sum():.3f} m^2")
    GEOMHASH = float(area.sum()) + float((area * np.arange(1, N + 1)).sum()) * 1e-9
    ORIENT = IC.sweep_orientation(a.sweep_logs, N)
    REFEREE = IC.sweep_referee_params(a.sweep_logs)
    FTHR = a.min_hits / a.nrays
    files = sorted(glob.glob(a.chunks_glob))
    assert files, f"no chunk files matched {a.chunks_glob}"
    budget = int(a.ram_gib * GiB)

    # ---- A: count pass (coverage gate + per-lo counts) ----
    seen = np.zeros(N, dtype=np.int32); truncated = []; cnt_lo = np.zeros(N, dtype=np.int64)
    n_dir = 0
    for p in files:
        I, J, V = parse_file(p, N, FTHR, seen, truncated)
        if len(I):
            cnt_lo += np.bincount(np.minimum(I, J), minlength=N)
        n_dir += len(I)
    n_missing = int((seen == 0).sum()); n_dup = int((seen > 1).sum())
    log(f"sweep coverage: {int((seen == 1).sum())}/{N} faces swept exactly once "
        f"(missing {n_missing}, duplicated {n_dup}, truncated chunks {len(truncated)})")
    if n_missing or n_dup or truncated:
        msg = (f"sweep coverage is INCOMPLETE: {n_missing} never swept, {n_dup} more than once, "
               f"{len(truncated)} truncated {truncated[:3]}")
        if not a.allow_incomplete_sweep:
            raise SystemExit("FATAL: " + msg)
        print(f"[assemble-ooc] *** WARNING: {msg}; --allow-incomplete-sweep given, the matrix IS BIASED there ***", file=sys.stderr)
    LB = cut_blocks(cnt_lo, max(1, budget // B_DIRECTED))
    K = len(LB) - 1
    log(f"directed nnz={n_dir} ({n_dir/1e6:.2f}M) in {len(files)} files -> {K} lo-bucket(s) "
        f"(<= {budget // B_DIRECTED:,} entries each at --ram-gib {a.ram_gib:g}); scratch {scratch}  {time.time()-t0:.0f}s")
    log(f"plan: directed spill ~{16*n_dir/1e9:.1f} GB -> pairs ~{16*n_dir/1e9:.1f} GB max -> "
        f"transpose ~{12*n_dir/1e9:.1f} GB max -> CSR ~{8*n_dir*2/1e9:.1f} GB max (upper bounds)")

    # ---- B: spill pass (file-read order preserved inside each bucket) ----
    D = Spill(scratch, "dir", [("lo", "<i4"), ("hi", "<i4"), ("S", "<f8")])
    for p in files:
        I, J, V = parse_file(p, N, FTHR, np.zeros(N, np.int32), [])
        if not len(I):
            continue
        S = area[I] * V.astype(np.float64)
        lo = np.minimum(I, J).astype(np.int32); hi = np.maximum(I, J).astype(np.int32)
        bid = np.searchsorted(LB[1:], lo, side="right")
        o = np.argsort(bid, kind="stable")
        bs = np.bincount(bid, minlength=K); pos = np.concatenate(([0], np.cumsum(bs)))
        for k in np.nonzero(bs)[0]:
            sl = o[pos[k]:pos[k+1]]
            D.append(int(k), lo=lo[sl], hi=hi[sl], S=S[sl])
    log(f"spilled {D.bytes()/1e9:.2f} GB  {time.time()-t0:.0f}s")

    # ---- C: symmetrize per bucket, carrying the GLOBAL prefix sum ----
    P = Spill(scratch, "pair", [("a", "<i4"), ("b", "<i4"), ("Ssym", "<f8")])
    carry = 0.0; n_pairs = 0
    cnt_a = np.zeros(N, np.int64); cnt_b = np.zeros(N, np.int64)
    for k in range(K):
        d = D.load(k)
        lo, hi, S = d["lo"], d["hi"], d["S"]
        if len(lo):
            key = np.multiply(lo, np.int64(N), dtype=np.int64); key += hi
            o = IC.narrow_perm(np.argsort(key, kind="stable"), "o")
            key = key[o]; S = S[o]; lo = lo[o]; hi = hi[o]; del o
            bnd = np.concatenate(([0], np.nonzero(np.diff(key))[0] + 1, [len(key)]))
            gs = bnd[:-1]; ge = bnd[1:]
            S0 = S.copy(); S0[0] = carry + S0[0]
            csum = np.concatenate(([carry], np.cumsum(S0)))
            carry = float(csum[-1])
            ndir = (ge - gs).astype(np.float64) if a.symmetrize_divisor == "hits" else 2.0
            Ssym = (csum[ge] - csum[gs]) / ndir
            aa = lo[gs]; bb = hi[gs]
            P.append(k, a=aa, b=bb, Ssym=Ssym)
            cnt_a += np.bincount(aa, minlength=N); cnt_b += np.bincount(bb, minlength=N)
            n_pairs += len(aa)
            del key, S, S0, lo, hi, csum, bnd, gs, ge, Ssym, aa, bb
        if not a.keep_scratch:
            D.drop(k)
    log(f"symmetrization divisor = {a.symmetrize_divisor}; unique undirected pairs={n_pairs} "
        f"({n_pairs/1e6:.2f}M)  {time.time()-t0:.0f}s")

    # ---- D: symmetric SigmaF<=1 cap, in-core summation order ----
    FS = Spill(scratch, "F", [("Fa", "<f8"), ("Fb", "<f8")])
    scaled = False

    def F_of(k):
        if scaled:
            f = FS.load(k); return f["Fa"], f["Fb"], P.load(k)
        pr = P.load(k)
        return pr["Ssym"] / area[pr["a"]], pr["Ssym"] / area[pr["b"]], pr

    def rowsums():
        rs = np.zeros(N)
        for k in range(K):
            Fa, _, pr = F_of(k); np.add.at(rs, pr["a"], Fa)
        for k in range(K):
            _, Fb, pr = F_of(k); np.add.at(rs, pr["b"], Fb)
        return rs

    cap_rows_scaled = 0; it = 0; broke = False
    for it in range(40):
        rs = rowsums()
        if rs.max() <= 1.0 + 1e-9:
            broke = True
            break
        c = np.ones(N); over = rs > 1.0; c[over] = 1.0 / rs[over]
        if it == 0:
            cap_rows_scaled = int(over.sum())
        for k in range(K):
            Fa, Fb, pr = F_of(k)
            Fa = Fa * np.sqrt(c[pr["a"]] * c[pr["b"]])
            Fb = Fb * np.sqrt(c[pr["b"]] * c[pr["a"]])
            for f in ("Fa", "Fb"):
                pth = FS.path(k, f)
                if pth.exists(): pth.unlink()
            FS.append(k, Fa=Fa, Fb=Fb)
        scaled = True
    if not broke:
        rs = rowsums()
    log(f"symmetric-cap sweeps={it+1} -> SigmaF max={rs.max():.4f} (#>1={(rs>1.0+1e-6).sum()}) "
        f"mean={rs.mean():.4f}  {time.time()-t0:.0f}s")

    # ---- E: write in row blocks ----
    counts = cnt_a + cnt_b
    nnz = int(counts.sum()); assert nnz == 2 * n_pairs
    indptr = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
    RB = cut_blocks(counts, max(1, budget // B_ROW), align=LB)
    nRB = len(RB) - 1
    off_c = 4 + 8 + 8 * (N + 1) + 8
    off_d = off_c + 4 * nnz
    tmp = Path(str(out) + ".ooc_partial")
    with open(tmp, "wb") as fh:
        fh.write(np.array([N], dtype=np.int32).tobytes())
        fh.write(np.array([GEOMHASH], dtype=np.float64).tobytes())
        fh.write(indptr.tobytes())
        fh.write(np.array([nnz], dtype=np.int64).tobytes())
        fh.truncate(off_d + 4 * nnz)
    TR = Spill(scratch, "tr", [("r", "<i4"), ("c", "<i4"), ("v", "<f4")])
    rb_of = lambda x: np.searchsorted(RB[1:], x, side="right")
    with open(tmp, "r+b") as fh:
        for k in range(K):
            Fa, Fb, pr = F_of(k)
            aa, bb = pr["a"], pr["b"]
            # transpose this bucket's b-half (rows = b) into row-block files, global pair order
            rbid = rb_of(bb); o = np.argsort(rbid, kind="stable")
            bs = np.bincount(rbid, minlength=nRB); pos = np.concatenate(([0], np.cumsum(bs)))
            for j in np.nonzero(bs)[0]:
                sl = o[pos[j]:pos[j+1]]
                TR.append(int(j), r=bb[sl], c=aa[sl], v=Fb[sl].astype(np.float32))
            del o, rbid
            # every row block inside [LB[k], LB[k+1]) is now complete: write it
            for j in range(int(rb_of(np.array([LB[k]]))[0]), nRB):
                r0, r1 = int(RB[j]), int(RB[j+1])
                if r0 >= int(LB[k+1]):
                    break
                i0, i1 = np.searchsorted(aa, [r0, r1])    # a-half rows of this block (aa ascending)
                t = TR.load(j)
                rows = np.concatenate((aa[i0:i1], t["r"]))
                cols = np.concatenate((bb[i0:i1], t["c"]))
                vals = np.concatenate((Fa[i0:i1].astype(np.float32), t["v"]))
                o2 = np.argsort(rows, kind="stable")
                cols = cols[o2]; vals = vals[o2]
                e0 = int(indptr[r0]); assert len(cols) == int(indptr[r1]) - e0, (j, len(cols))
                fh.seek(off_c + 4 * e0); fh.write(cols.astype(np.int32).tobytes())
                fh.seek(off_d + 4 * e0); fh.write(vals.astype(np.float32).tobytes())
                if not a.keep_scratch:
                    TR.drop(j)
            if not a.keep_scratch:
                P.drop(k); FS.drop(k)
    os.replace(tmp, out)
    log(f"wrote {out}  nnz={nnz}  {os.path.getsize(out)/1e9:.3f} GB  geomhash={GEOMHASH:.6f}  {time.time()-t0:.0f}s")
    np.save(str(out.with_suffix(".rowsum.npy")), rs)
    if not a.keep_scratch:
        shutil.rmtree(scratch, ignore_errors=True)

    prov = {
        "estimator": {"tried": "FC-08 corrected", "hits": "PRE-FC-08 DEFECTIVE"}[a.symmetrize_divisor],
        "symmetrize_divisor": a.symmetrize_divisor,
        "symmetrize_divisor_meaning": ("mean over the 2 ray directions TRIED (unbiased)"
                                       if a.symmetrize_divisor == "tried"
                                       else "mean over the directions that HIT (inflates far field up to 2x)"),
        "nrays": int(a.nrays), "min_hits": float(a.min_hits),
        "mesh": str(a.mesh), "n_faces": int(N), "nnz": int(nnz),
        "geomhash": float(GEOMHASH), "total_area_m2": float(area.sum()),
        "sweep_glob": str(a.chunks_glob), "sweep_chunk_files": len(files),
        "sweep_coverage_exactly_once": int((seen == 1).sum()),
        "sweep_coverage_missing": n_missing, "sweep_coverage_duplicated": n_dup,
        "sweep_truncated_chunks": len(truncated),
        "allow_incomplete_sweep": bool(a.allow_incomplete_sweep),
        "sigma_f_max": float(rs.max()), "sigma_f_mean": float(rs.mean()),
        "radiance_prune_eps": float(a.radiance_prune_eps),
        **ORIENT,
        **REFEREE,
        "assembler": str(Path(__file__).resolve()),
        "assembler_mode": f"out_of_core (byte-identical contract with assemble_vf_haworth.py; {K} lo-buckets, {nRB} row blocks, --ram-gib {a.ram_gib:g})",
        "assembled_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(str(out) + ".provenance.json", "w") as fh:
        json.dump(prov, fh, indent=2)
    log(f"provenance -> {out}.provenance.json")

    if a.vf_stats != "off":
        import vf_stats
        ip = np.memmap(out, dtype="<i8", mode="r", offset=12, shape=(N + 1,))
        cm = np.memmap(out, dtype="<i4", mode="r", offset=off_c, shape=(nnz,))
        dm = np.memmap(out, dtype="<f4", mode="r", offset=off_d, shape=(nnz,))
        st = vf_stats.compute(ip, cm, dm, N, area=area, nrays=int(a.nrays), min_hits=float(a.min_hits),
                              cap_sweeps=it + 1, rows_scaled=cap_rows_scaled, sweep_logs=a.sweep_logs,
                              csr_path=out, spectral=("auto" if a.vf_stats == "full" else False))
        print(vf_stats.banner(st))
        log(f"vf stats -> {vf_stats.write_sidecar(out, st)}")
    log(f"done  {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
