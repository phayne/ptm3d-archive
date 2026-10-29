#!/usr/bin/env python3
"""Byte-identity gate: assemble_vf_ooc.py vs assemble_vf_haworth.py.

Synthetic, no external data. A small radially placed grid mesh and synthetic referee chunk
files whose F values are large enough that many rows exceed 1, so the symmetric cap actually
SCALES (the path real meshes have never taken). Run for both divisors and at a RAM budget tiny
enough to force many lo-buckets and row blocks (every carry and merge path is exercised).
PASS = the .csr and .rowsum.npy are byte-identical and the provenance agrees on every shared key.
  usage: test_vf_assembler_ooc.py [--real <mesh> <chunks_glob> <nrays> <min_hits> <logs_glob> <ram_gib>]
"""
import json, os, struct, subprocess, sys, tempfile
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
PY = sys.executable
SKIP_KEYS = {"assembler", "assembler_mode", "assembled_utc"}


def write_mesh(path, n=24):
    xs = np.linspace(0, 1.0, n)
    X, Y = np.meshgrid(xs, xs)
    Z = 0.05 * np.sin(3 * X) * np.cos(2 * Y)
    up = np.array([0.0, 0.0, 1.0])
    P = np.c_[X.ravel(), Y.ravel(), 1737400.0 + Z.ravel()]
    F = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b, c, d = i * n + j, i * n + j + 1, (i + 1) * n + j, (i + 1) * n + j + 1
            F += [(a, b, d), (a, d, c)]
    hdr = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(P)}\nproperty double x\n"
           f"property double y\nproperty double z\nelement face {len(F)}\n"
           "property list uchar int vertex_indices\nend_header\n")
    with open(path, "wb") as fh:
        fh.write(hdr.encode()); fh.write(P.astype("<f8").tobytes())
        for f in F:
            fh.write(struct.pack("<B3i", 3, *f))
    return len(F)


def write_chunks(d, nf, nfiles=5, seed=7):
    rng = np.random.default_rng(seed)
    order = rng.permutation(nf)                     # sources in scrambled order across files
    for k, chunk in enumerate(np.array_split(order, nfiles)):
        with open(d / f"pp_{k:02d}.bin", "wb") as fh:
            for s in chunk:
                m = int(rng.integers(0, 60))
                j = rng.choice(nf, m, replace=False).astype("<i4")
                j = j[j != s]
                f = (rng.random(len(j)) * 0.06).astype("<f4")   # row sums ~0-1.8: many rows > 1
                f[rng.random(len(j)) < 0.1] = 1e-6              # some below the keep threshold
                fh.write(struct.pack("<ii", int(s), len(j)))
                fh.write(np.rec.fromarrays([j, f], names="j,f").tobytes())


def run_pair(work, mesh, glob_, nrays, min_hits, logs, div, ram):
    outs = {}
    for tag, script, extra in (("incore", "assemble_vf_haworth.py", []),
                               ("ooc", "assemble_vf_ooc.py", ["--ram-gib", str(ram), "--vf-stats", "off"])):
        out = work / f"{tag}_{div}.csr"
        cmd = [PY, str(HERE / script), str(mesh), glob_, str(nrays), str(min_hits), str(out),
               "--symmetrize-divisor", div] + (["--sweep-logs", logs] if logs else []) + extra
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 and not (tag == "incore" and out.exists()):
            raise SystemExit(f"FAIL: {tag} rc={r.returncode}\n{r.stderr[-3000:]}")
        outs[tag] = (out, r.stdout + r.stderr)
    a, b = outs["incore"][0], outs["ooc"][0]
    same_csr = a.read_bytes() == b.read_bytes()
    same_rs = a.with_suffix(".rowsum.npy").read_bytes() == b.with_suffix(".rowsum.npy").read_bytes()
    pa = json.load(open(str(a) + ".provenance.json")); pb = json.load(open(str(b) + ".provenance.json"))
    keys = (set(pa) | set(pb)) - SKIP_KEYS
    pdiff = sorted(k for k in keys if pa.get(k) != pb.get(k) and k != "sweep_glob")
    cap = [l for l in outs["ooc"][1].splitlines() if "symmetric-cap" in l or "lo-bucket" in l]
    return same_csr, same_rs, pdiff, cap


def main():
    fails = []
    with tempfile.TemporaryDirectory(prefix="ooc_gate_") as w:
        work = Path(w)
        if len(sys.argv) > 1 and sys.argv[1] == "--real":
            mesh, glob_, nrays, mh, logs, ram = sys.argv[2:8]
            for div in ("tried",):
                c, r, p, cap = run_pair(work, mesh, glob_, nrays, mh, logs, div, ram)
                print(f"real {div}: csr_identical={c} rowsum_identical={r} provenance_diffs={p}\n  " + "\n  ".join(cap))
                if not (c and r and not p): fails.append(div)
        else:
            mesh = work / "grid.ply"; nf = write_mesh(mesh)
            (work / "sw").mkdir(); write_chunks(work / "sw", nf)
            for div in ("tried", "hits"):
                for ram in ("0.00002", "64"):                  # ~20 KB budget -> dozens of buckets; and one bucket
                    c, r, p, cap = run_pair(work, mesh, str(work / "sw" / "pp_*.bin"), 1000, 1, None, div, ram)
                    print(f"{div} ram={ram}: csr_identical={c} rowsum_identical={r} provenance_diffs={p}\n  " + "\n  ".join(cap))
                    if not (c and r and not p): fails.append(f"{div}/{ram}")
    if fails:
        print("FAIL:", fails); return 1
    print("PASS: out-of-core assembler byte-identical to the in-core assembler"); return 0


if __name__ == "__main__":
    sys.exit(main())
