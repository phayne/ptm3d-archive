#!/usr/bin/env python3
"""Gate for assemble_vf_haworth.py --symmetrize-divisor pooled (Fermi 2026-10-15). No external data.

(a) EXACT synthetic: small faces (area a) paired with large faces (area 7a), true exchange G per pair.
    Directed hits are drawn binomially, h_ij ~ Bin(NR, G/A_i), h_ji ~ Bin(NR, beta * G/A_j), and fed
    through the real assembler as chunk files. beta = 1 (unbiased reverse) and beta = 0.914 (Fermi's
    measured 509k reverse deficit). Checks:
      - reciprocity exact for both estimators (A_i F_ij == A_j F_ji to float32);
      - beta = 1: both unbiased on the small-face row sums (|mean bias| < 4 SE), pooled variance lower;
      - beta = 0.914: 'tried' row-sum bias ~ (beta-1)/2 = -4.3%, 'pooled' ~ (beta-1) * A_i/(A_i+A_j)
        = -1.1% (the large face's weight in the pooled estimate).
(b) GEOMETRIC: a small square plate facing a 7x larger parallel plate (relief-free, so the keps
    rise filter only acts through its edge scale), swept by the current referee at NR 2e4 and keps
    0.2 / 0.01; reference = the small plate's own forward sum at NR 2e6. Reports the reverse/forward
    ratio and the tried / pooled small-row error. Informational (prints), plus a sanity bar.
"""
import json, os, struct, subprocess, sys, tempfile
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
PY = sys.executable
sys.path.insert(0, str(HERE))


def write_ply(path, V, F):
    hdr = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(V)}\nproperty double x\n"
           f"property double y\nproperty double z\nelement face {len(F)}\n"
           "property list uchar int vertex_indices\nend_header\n")
    with open(path, "wb") as fh:
        fh.write(hdr.encode()); fh.write(np.asarray(V, "<f8").tobytes())
        for f in F:
            fh.write(struct.pack("<B3i", 3, *map(int, f)))


def read_csr(p):
    b = open(p, "rb").read(); n = struct.unpack_from("<i", b, 0)[0]
    ip = np.frombuffer(b, "<i8", n + 1, 12); nnz = struct.unpack_from("<q", b, 12 + 8 * (n + 1))[0]
    c0 = 12 + 8 * (n + 1) + 8
    return n, ip, np.frombuffer(b, "<i4", nnz, c0), np.frombuffer(b, "<f4", nnz, c0 + 4 * nnz)


def assemble(work, mesh, glob_, nr, div, tag):
    out = work / f"{tag}.csr"
    r = subprocess.run([PY, str(HERE / "assemble_vf_haworth.py"), str(mesh), glob_, str(nr), "1", str(out),
                        "--symmetrize-divisor", div], capture_output=True, text=True)
    if not out.exists():
        raise SystemExit(f"FAIL: assembler {div}: {r.stderr[-2000:]}")
    return read_csr(out)


def synthetic(work, beta, seed):
    """P pairs: face 2k (area a), face 2k+1 (area 7a). Geometry is irrelevant to the assembler
    except through the areas, so each face is its own isolated triangle of the right area."""
    P, NR, a = 3000, 20000, 1e-4
    rng = np.random.default_rng(seed)
    V, F, area = [], [], []
    for k in range(2 * P):
        A = a if k % 2 == 0 else 7 * a
        s = np.sqrt(2 * A); x0 = 10.0 * k
        V += [(x0, 0, 1737400.0), (x0 + s, 0, 1737400.0), (x0, s, 1737400.0)]
        F.append((3 * k, 3 * k + 1, 3 * k + 2)); area.append(A)
    area = np.array(area)
    mesh = work / "syn.ply"; write_ply(mesh, V, F)
    area = 0.5 * np.array([np.linalg.norm(np.cross(np.subtract(V[f[1]], V[f[0]]), np.subtract(V[f[2]], V[f[0]]))) for f in F])
    G = rng.uniform(0.02, 0.12, P) * a          # true exchange A_i F_ij of each pair
    h_ij = rng.binomial(NR, G / area[0::2]); h_ji = rng.binomial(NR, beta * G / area[1::2])
    with open(work / "pp_00.bin", "wb") as fh:
        for k in range(P):
            i, j = 2 * k, 2 * k + 1
            for src, dst, h in ((i, j, h_ij[k]), (j, i, h_ji[k])):
                if h > 0:
                    fh.write(struct.pack("<ii", src, 1)); fh.write(struct.pack("<if", dst, h / NR))
                else:
                    fh.write(struct.pack("<ii", src, 0))
    res = {}
    for div in ("tried", "pooled"):
        n, ip, c, d = assemble(work, mesh, str(work / "pp_*.bin"), NR, div, f"syn_{div}_{beta}_{seed}")
        rows = np.repeat(np.arange(n), np.diff(ip))
        Fij = {(int(r), int(cc)): float(v) for r, cc, v in zip(rows, c, d)}
        small = np.array([Fij.get((2 * k, 2 * k + 1), 0.0) * area[2 * k] for k in range(P)])
        rec = max(abs(Fij[(2 * k, 2 * k + 1)] * area[2 * k] - Fij[(2 * k + 1, 2 * k)] * area[2 * k + 1]) /
                  (Fij[(2 * k, 2 * k + 1)] * area[2 * k]) for k in range(P) if (2 * k, 2 * k + 1) in Fij)
        rel = small / G - 1
        res[div] = {"bias": float(rel.mean()), "se": float(rel.std(ddof=1) / np.sqrt(P)),
                    "rms": float(np.sqrt((rel ** 2).mean())), "recip_maxrel": float(rec)}
    return res


def plate(n, side, z, up, R0=1737400.0):
    """n x n grid of 2n^2 triangles, centered, at height z, wound to face +z (up) or -z."""
    xs = np.linspace(-side / 2, side / 2, n + 1)
    V = [(x, y, R0 + z) for y in xs for x in xs]
    F = []
    for r in range(n):
        for c in range(n):
            a, b, d, e = r * (n + 1) + c, r * (n + 1) + c + 1, (r + 1) * (n + 1) + c, (r + 1) * (n + 1) + c + 1
            F += [(a, b, e), (a, e, d)] if up else [(a, e, b), (a, d, e)]
    return V, F


def geometric(work, keps, n=8):
    """Small plate (faces of area a) facing a 7x-area-per-face large plate, both resolved by n x n
    grids: the regime of small cold-floor faces partnered by larger wall faces. Reference: the small
    faces' own forward sums at NR 2e6 (centroid launch is accurate for the small faces)."""
    from ensure_mc_ref_pp import resolve_current
    mc = resolve_current()
    s, h = 0.1, 0.15; L = s * np.sqrt(7)
    V1, F1 = plate(n, s, 0.0, True); V2, F2 = plate(n, L, h, False)
    off = len(V1); V = V1 + V2; F = F1 + [(a + off, b + off, c + off) for a, b, c in F2]
    m = len(F1); nf = len(F)
    mesh = work / f"plates_{keps}.ply"; write_ply(mesh, V, F)
    Vn = np.array(V); Fn = np.array(F)
    A = 0.5 * np.linalg.norm(np.cross(Vn[Fn[:, 1]] - Vn[Fn[:, 0]], Vn[Fn[:, 2]] - Vn[Fn[:, 0]]), axis=1)
    tm = work / "t.bin"; tm.write_bytes(struct.pack("<i", nf) + np.zeros(nf, "<f4").tobytes())
    fl = work / "f.txt"; fl.write_text("\n".join(map(str, range(nf))) + "\n")
    env = dict(os.environ, PTM_PP_ORIENT="raw")  # two facing plates: the raw winding IS the facing side
    out = {}
    for nr in (20000, 2000000):
        b = work / f"pp_{keps}_{nr}.bin"
        subprocess.run([mc, str(mesh), str(tm), str(fl), str(nr), "1e-4", str(keps), "1", str(b)], env=env,
                       capture_output=True, check=True)
        M = np.zeros((nf, nf)); raw = b.read_bytes(); q = 0
        while q + 8 <= len(raw):
            src, k = struct.unpack_from("<ii", raw, q); q += 8
            rec = np.frombuffer(raw, dtype=[("j", "<i4"), ("f", "<f4")], count=k, offset=q); q += 8 * k
            M[src, rec["j"]] = rec["f"]
        out[nr] = M
    S, Lg = slice(0, m), slice(m, nf)
    ref = (A[S, None] * out[2000000][S, Lg]).sum()                 # forward, small faces, 2e6
    fwd = (A[S, None] * out[20000][S, Lg]).sum()
    rev = (A[Lg, None] * out[20000][Lg, S]).sum()
    tried = 0.5 * (fwd + rev)
    G = (out[20000][S, Lg] + out[20000][Lg, S].T) / (1 / A[S, None] + 1 / A[None, Lg])
    pooled = G.sum()
    return {"keps": keps, "faces": [m, nf - m], "area_ratio": float(A[Lg].mean() / A[S].mean()),
            "fwd_err": fwd / ref - 1, "rev_over_fwd": rev / fwd, "tried_err": tried / ref - 1, "pooled_err": pooled / ref - 1}


def main():
    fails = []
    with tempfile.TemporaryDirectory(prefix="pooled_") as w:
        work = Path(w)
        r1 = synthetic(work, 1.0, 1); print("beta=1.000", json.dumps(r1))
        for div in ("tried", "pooled"):
            if abs(r1[div]["bias"]) > 4 * r1[div]["se"]: fails.append(f"{div} biased at beta=1: {r1[div]}")
            if r1[div]["recip_maxrel"] > 2e-6: fails.append(f"{div} reciprocity {r1[div]['recip_maxrel']}")
        if not r1["pooled"]["rms"] < r1["tried"]["rms"]: fails.append("pooled not lower variance")
        for f in work.glob("pp_*.bin"): f.unlink()
        r2 = synthetic(work, 0.914, 2); print("beta=0.914", json.dumps(r2))
        if abs(r2["tried"]["bias"] - (-0.043)) > 0.006: fails.append(f"tried bias {r2['tried']['bias']:.4f} != -0.043")
        if abs(r2["pooled"]["bias"] - (-0.086 / 8)) > 0.004: fails.append(f"pooled bias {r2['pooled']['bias']:.4f} != -0.0108")
        for keps in (0.2, 0.01):
            g = geometric(work, keps); print("geometric", json.dumps(g))
            if not abs(g["pooled_err"]) < abs(g["tried_err"]): fails.append(f"geometric: pooled not closer than tried at keps {keps}: {g}")
            if abs(g["pooled_err"]) > 0.02: fails.append(f"geometric pooled error {g['pooled_err']:.4f} at keps {keps}")
    if fails:
        print("FAIL:\n  " + "\n  ".join(fails)); return 1
    print("PASS: pooled estimator (reciprocal, unbiased, lower variance; down-weights a biased large-face reverse)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
