#!/usr/bin/env python3
"""FC-08 regression gate for the offline view-factor assembler.

Two arms. Both drive the ACTUAL referee pipeline end to end -- mc_ref_pp sweep ->
assemble_vf_haworth.py -> CSR -- so a defect anywhere between the ray sampler and the
written matrix is in scope.

  ARM A  CLOSED FORM.  A 0.01 m^2 horizontal element under a coaxial parallel disk
         (a = 57.735 m at H = 100 m) has the exact answer F = a^2/(a^2 + H^2) =
         0.250000.  The disk is genuinely flat, so tessellating it introduces no
         geometric error; the only approximation is the finite extent of the element,
         O((e0/H)^2) ~ 2e-6.  The disk is cut into ~99,800 facets so every pair sits at
         lambda = NRAYS * F_ij of order 1 -- the production regime.  The assembled row 0
         must reproduce 0.250000 to within --closed-form-tol-pct.

  ARM B  RAY-BUDGET INVARIANCE.  Assemble one and the same geometry (the repo's
         lunar_crater_80N mesh) from two sweeps that differ only in ray budget.  An
         unbiased estimator returns the same matrix at both budgets.  An estimator whose
         weight depends on whether a ray direction happened to register a hit does not:
         it is consistent but biased, and production runs at the biased end.

Both arms also run the pre-2026-08-19 DEFECTIVE estimator (--symmetrize-divisor hits)
and REQUIRE it to fail.  A gate that has quietly lost its teeth then reports itself
instead of passing.

DO NOT add a reciprocity or a SigmaF <= 1 assertion here.  Both were measured against
the defective matrices and both PASS on them: the assembler symmetrizes and then caps by
a symmetric congruence, so reciprocity holds to float32 round-off (measured median
2.7e-8) while the entries are up to 2x too large, and SigmaF peaks at 0.18 so the cap
never fires.  Those invariants constrain the shape of a row, never its overall scale.
The Ingersoll bowl-crater test is blind here too: it builds its view factors with the
in-code calculator and never invokes this pipeline.

Requires numpy and Embree (the same Embree mc_ref_pp is built against).
Runtime on the 48-core Mac Pro: a few minutes.

Usage:
  test_vf_assembler_gate.py [--work DIR] [--jobs N] [--mc-ref-pp PATH]
                            [--arm A|B|both] [--keep]
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ASSEMBLER = HERE / "assemble_vf_haworth.py"
MC_SRC = HERE / "mc_ref_pp.cpp"
REPO = HERE.parent.parent
CRATER_PLY = REPO / "data" / "meshes" / "lunar_crater_80N.ply"

# Closed form for the arm-A geometry.
DISK_H = 100.0
DISK_A = 57.735026918962575          # a^2/(a^2+H^2) == 0.25 exactly
DISK_A0 = 0.01                       # m^2, the receiving element
DISK_NTH = 200                       # azimuthal divisions
DISK_K = 250                         # radial rings  -> 200*250*2 - 200 = 99,800 facets
CLOSED_FORM = DISK_A ** 2 / (DISK_A ** 2 + DISK_H ** 2)


# --------------------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------------------
def write_binary_ply(path, V, F):
    """Binary-little-endian PLY, float64 vertices, uchar+3*int32 faces (a layout both
    mc_ref_pp's loader and the assembler's reader prove by file-size identity)."""
    V = np.asarray(V, float)
    F = np.asarray(F, np.int32)
    with open(path, "wb") as f:
        f.write(b"ply\nformat binary_little_endian 1.0\n")
        f.write(b"element vertex %d\n" % len(V))
        f.write(b"property double x\nproperty double y\nproperty double z\n")
        f.write(b"element face %d\n" % len(F))
        f.write(b"property list uchar int vertex_indices\nend_header\n")
        V.astype("<f8").tofile(f)
        rec = np.zeros(len(F), dtype=[("c", "u1"), ("i", "<i4", 3)])
        rec["c"] = 3
        rec["i"] = F
        rec.tofile(f)


def make_disk_mesh(path):
    """Face 0: an equilateral triangle of area A0 at the origin, normal +z.
    Faces 1..M: a flat disk of radius a at height H, normal -z, tessellated."""
    e0 = np.sqrt(DISK_A0 / (np.sqrt(3) / 4))
    R0 = e0 / np.sqrt(3.0)
    V = [[R0 * np.cos(2 * np.pi * k / 3), R0 * np.sin(2 * np.pi * k / 3), 0.0] for k in range(3)]
    F = [[0, 1, 2]]

    rk = DISK_A * np.sqrt(np.arange(DISK_K + 1) / DISK_K)
    ringstart = []
    for k in range(DISK_K + 1):
        ringstart.append(len(V))
        if k == 0:
            V.append([0.0, 0.0, DISK_H])
        else:
            for t in range(DISK_NTH):
                th = 2 * np.pi * t / DISK_NTH
                V.append([rk[k] * np.cos(th), rk[k] * np.sin(th), DISK_H])
    c = ringstart[0]
    for t in range(DISK_NTH):                       # innermost fan, wound for a -z normal
        F.append([c, ringstart[1] + (t + 1) % DISK_NTH, ringstart[1] + t])
    for k in range(1, DISK_K):
        A, B = ringstart[k], ringstart[k + 1]
        for t in range(DISK_NTH):
            t2 = (t + 1) % DISK_NTH
            F.append([A + t, A + t2, B + t2])
            F.append([A + t, B + t2, B + t])

    V = np.asarray(V, float)
    F = np.asarray(F, np.int32)
    p0, p1, p2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    n = np.cross(p1 - p0, p2 - p0)
    ar = 0.5 * np.linalg.norm(n, axis=1)
    nz = n[:, 2] / np.linalg.norm(n, axis=1)
    if not nz[0] > 0.999:
        raise SystemExit(f"FATAL: element normal must be +z, got {nz[0]:.4f}")
    if not (nz[1:] < -0.999).all():
        raise SystemExit("FATAL: disk normals must all be -z")
    write_binary_ply(path, V, F)
    print(f"  disk mesh: {len(F)} faces, element area {ar[0]:.6f} m^2 (target {DISK_A0}), "
          f"disk area {ar[1:].sum():.3f} m^2 (exact {np.pi*DISK_A*DISK_A:.3f})")
    print(f"  CLOSED FORM  F_0->disk = a^2/(a^2+H^2) = {CLOSED_FORM:.10f}")
    return len(F)


def convert_ascii_ply(src, dst):
    """The repo's crater meshes ship as ASCII PLY; mc_ref_pp wants binary."""
    lines = open(src).read().split("\n")
    he = lines.index("end_header") + 1
    nv = nf = 0
    for ln in lines[:he]:
        p = ln.split()
        if len(p) >= 3 and p[0] == "element" and p[1] == "vertex":
            nv = int(p[2])
        elif len(p) >= 3 and p[0] == "element" and p[1] == "face":
            nf = int(p[2])
    if nv == 0 or nf == 0:
        raise SystemExit(f"FATAL: could not read vertex/face counts from {src}")
    V = np.array([[float(x) for x in lines[he + i].split()[:3]] for i in range(nv)])
    rows = [lines[he + nv + i].split() for i in range(nf)]
    if not all(int(r[0]) == 3 for r in rows):
        raise SystemExit(f"FATAL: {src} is not fully triangulated")
    F = np.array([[int(x) for x in r[1:4]] for r in rows], dtype=np.int32)
    write_binary_ply(dst, V, F)
    print(f"  crater mesh: {nf} faces from {src.name}")
    return nf


# --------------------------------------------------------------------------------------
# referee pipeline
# --------------------------------------------------------------------------------------
def find_or_build_mc_ref_pp(explicit, work):
    for cand in (explicit, os.environ.get("PTM_MC_REF_PP"), HERE / "mc_ref_pp"):
        if cand and Path(cand).is_file() and os.access(str(cand), os.X_OK):
            print(f"  mc_ref_pp: using existing {cand}")
            return str(cand)
    out = work / "mc_ref_pp"
    sdk = subprocess.run(["/usr/bin/xcrun", "--show-sdk-path"],
                         capture_output=True, text=True).stdout.strip()
    cmd = ["/Library/Developer/CommandLineTools/usr/bin/clang++", "-O3", "-std=c++17"]
    if sdk:
        cmd += ["-isysroot", sdk]
    cmd += ["-I/usr/local/include", str(MC_SRC), "-L/usr/local/lib", "-lembree4", "-o", str(out)]
    print("  mc_ref_pp: building --", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not out.is_file():
        sys.stderr.write(r.stderr)
        raise SystemExit("FATAL: could not build mc_ref_pp. Install Embree 4 (headers in "
                         "/usr/local/include/embree4) or pass --mc-ref-pp /path/to/mc_ref_pp.")
    return str(out)


def write_tmax(path, n, value=100.0):
    """mc_ref_pp's per-face temperature input: int32 count followed by float32 values.
    Only the F_warm384/Qir diagnostic columns use it; the view factors do not."""
    with open(path, "wb") as f:
        np.array([n], dtype="<i4").tofile(f)
        np.full(n, value, dtype="<f4").tofile(f)


def sweep(binary, mesh, tmax, nfaces, nrays, outdir, jobs):
    """Run the per-partner referee over every face, in `jobs` parallel chunks."""
    outdir.mkdir(parents=True, exist_ok=True)
    chunks = np.array_split(np.arange(nfaces), jobs)
    env = dict(os.environ)
    env.pop("PTM_PP_RING_PLY", None)          # no occluder mesh in either arm
    procs = []
    t0 = time.time()
    for k, ch in enumerate(chunks):
        fl = outdir / f"fl_{k:02d}.txt"
        fl.write_text("\n".join(str(int(i)) for i in ch) + "\n")
        log = open(outdir / f"log_{k:02d}.txt", "w")
        procs.append((subprocess.Popen(
            [binary, str(mesh), str(tmax), str(fl), str(nrays), "1e-4", "0.2", "1",
             str(outdir / f"pp_{k:02d}.bin")], stdout=log, stderr=subprocess.STDOUT, env=env),
            log, k))
    bad = []
    for p, log, k in procs:
        rc = p.wait()
        log.close()
        if rc != 0:
            bad.append((k, rc))
    if bad:
        for k, rc in bad[:3]:
            sys.stderr.write((outdir / f"log_{k:02d}.txt").read_text()[-2000:])
        raise SystemExit(f"FATAL: mc_ref_pp failed on {len(bad)} chunk(s): {bad}")
    got = sorted(glob.glob(str(outdir / "pp_*.bin")))
    if len(got) != jobs:
        raise SystemExit(f"FATAL: expected {jobs} sweep chunks, found {len(got)}")
    print(f"  sweep NRAYS={nrays}: {jobs} chunks, {nfaces} faces, {time.time()-t0:.0f}s")
    return str(outdir / "pp_*.bin")


def assemble(mesh, chunks_glob, nrays, out_csr, divisor):
    cmd = [sys.executable, str(ASSEMBLER), str(mesh), chunks_glob, str(nrays), "1",
           str(out_csr), "--symmetrize-divisor", divisor]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout + r.stderr)
        raise SystemExit(f"FATAL: assembler failed (divisor={divisor})")
    if divisor == "hits" and "WARNING" not in r.stderr:
        raise SystemExit("FATAL: the assembler ran the defective estimator WITHOUT warning. "
                         "The disclosure has regressed; fix that before trusting this gate.")
    return Path(out_csr)


def read_csr(path):
    with open(path, "rb") as f:
        n = int(np.frombuffer(f.read(4), "<i4")[0])
        f.read(8)                                            # geomhash
        indptr = np.frombuffer(f.read(8 * (n + 1)), "<i8")
        nnz = int(np.frombuffer(f.read(8), "<i8")[0])
        cols = np.frombuffer(f.read(4 * nnz), "<i4")
        data = np.frombuffer(f.read(4 * nnz), "<f4")
    return n, indptr, cols, data


def row_sum(path, row=0):
    _, indptr, _, data = read_csr(path)
    return float(data[indptr[row]:indptr[row + 1]].sum())


def mean_sigma_f(csr_path):
    """Mesh-wide mean row sum, taken from the assembler's own sidecar."""
    rs = np.load(str(Path(csr_path).with_suffix(".rowsum.npy")))
    return float(rs.mean())


# --------------------------------------------------------------------------------------
# arms
# --------------------------------------------------------------------------------------
def arm_a(work, binary, jobs, nrays, tol_pct, defect_min_pct):
    print("\n=== ARM A: closed-form coaxial disk, through the real referee pipeline ===")
    mesh = work / "disk.ply"
    nfaces = make_disk_mesh(mesh)
    tmax = work / "disk_tmax.bin"
    write_tmax(tmax, nfaces)
    g = sweep(binary, mesh, tmax, nfaces, nrays, work / f"disk_nr{nrays}", jobs)

    res = {}
    for div in ("tried", "hits"):
        csr = assemble(mesh, g, nrays, work / f"disk_{div}.csr", div)
        res[div] = row_sum(csr, 0)

    print(f"\n  {'method':<44}{'Sum_j F_0j':>12}{'vs closed form':>17}")
    print(f"  {'CLOSED FORM  a^2/(a^2+H^2)':<44}{CLOSED_FORM:>12.6f}{'--':>17}")
    for div, label in (("tried", "divisor = directions TRIED (shipping)"),
                       ("hits", "divisor = directions HIT  (pre-2026-08-19)")):
        e = 100 * (res[div] / CLOSED_FORM - 1)
        print(f"  {label:<44}{res[div]:>12.6f}{e:>+16.3f}%")

    err_ok = abs(100 * (res["tried"] / CLOSED_FORM - 1))
    err_bad = abs(100 * (res["hits"] / CLOSED_FORM - 1))
    ok = err_ok <= tol_pct
    teeth = err_bad >= defect_min_pct
    print(f"\n  [A1] shipping estimator within {tol_pct}%   : "
          f"{'PASS' if ok else 'FAIL'} ({err_ok:.3f}%)")
    print(f"  [A2] defective estimator rejected >= {defect_min_pct}% : "
          f"{'PASS' if teeth else 'FAIL'} ({err_bad:.3f}%)")
    return ok and teeth


def arm_b(work, binary, jobs, nr_lo, nr_hi, tol_pct, defect_min_pct):
    print("\n=== ARM B: ray-budget invariance on the lunar_crater_80N mesh ===")
    if not CRATER_PLY.is_file():
        raise SystemExit(f"FATAL: {CRATER_PLY} not found")
    mesh = work / "crater80N_tri.ply"
    nfaces = convert_ascii_ply(CRATER_PLY, mesh)
    tmax = work / "crater_tmax.bin"
    write_tmax(tmax, nfaces)

    res = {}
    for nr in (nr_lo, nr_hi):
        g = sweep(binary, mesh, tmax, nfaces, nr, work / f"crater_nr{nr}", jobs)
        for div in ("tried", "hits"):
            csr = assemble(mesh, g, nr, work / f"crater_{div}_nr{nr}.csr", div)
            res[(div, nr)] = mean_sigma_f(csr)

    print(f"\n  {'estimator':<44}{'mean SigmaF @%d' % nr_lo:>20}"
          f"{'mean SigmaF @%d' % nr_hi:>22}{'spread':>10}")
    spread = {}
    for div, label in (("tried", "divisor = directions TRIED (shipping)"),
                       ("hits", "divisor = directions HIT  (pre-2026-08-19)")):
        lo, hi = res[(div, nr_lo)], res[(div, nr_hi)]
        spread[div] = 100 * abs(lo / hi - 1)
        print(f"  {label:<44}{lo:>20.5f}{hi:>22.5f}{spread[div]:>9.2f}%")

    ok = spread["tried"] <= tol_pct
    teeth = spread["hits"] >= defect_min_pct
    print(f"\n  [B1] shipping estimator invariant to <= {tol_pct}%  : "
          f"{'PASS' if ok else 'FAIL'} ({spread['tried']:.3f}%)")
    print(f"  [B2] defective estimator drifts >= {defect_min_pct}%   : "
          f"{'PASS' if teeth else 'FAIL'} ({spread['hits']:.2f}%)")
    return ok and teeth


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default=None, help="scratch directory (default: a temp dir)")
    ap.add_argument("--jobs", type=int, default=min(20, os.cpu_count() or 4))
    ap.add_argument("--mc-ref-pp", default=None, help="prebuilt referee binary")
    ap.add_argument("--arm", choices=("A", "B", "both"), default="both")
    ap.add_argument("--nrays-closed-form", type=int, default=200000)
    ap.add_argument("--nrays-lo", type=int, default=4000)
    ap.add_argument("--nrays-hi", type=int, default=200000)
    ap.add_argument("--closed-form-tol-pct", type=float, default=1.0)
    ap.add_argument("--closed-form-defect-min-pct", type=float, default=10.0)
    ap.add_argument("--invariance-tol-pct", type=float, default=0.1)
    ap.add_argument("--invariance-defect-min-pct", type=float, default=25.0)
    ap.add_argument("--keep", action="store_true", help="keep the scratch directory")
    a = ap.parse_args()

    if not ASSEMBLER.is_file():
        raise SystemExit(f"FATAL: {ASSEMBLER} not found")
    work = Path(a.work) if a.work else Path(
        os.environ.get("TMPDIR", "/tmp")) / f"vfgate_{os.getpid()}"
    work.mkdir(parents=True, exist_ok=True)
    print(f"[vf-gate] work dir {work}  jobs={a.jobs}")
    binary = find_or_build_mc_ref_pp(a.mc_ref_pp, work)

    t0 = time.time()
    results = {}
    if a.arm in ("A", "both"):
        results["A closed form"] = arm_a(work, binary, a.jobs, a.nrays_closed_form,
                                         a.closed_form_tol_pct, a.closed_form_defect_min_pct)
    if a.arm in ("B", "both"):
        results["B ray-budget invariance"] = arm_b(work, binary, a.jobs, a.nrays_lo, a.nrays_hi,
                                                   a.invariance_tol_pct,
                                                   a.invariance_defect_min_pct)

    print(f"\n=== SUMMARY ({time.time()-t0:.0f}s) ===")
    for k, v in results.items():
        print(f"  {k:<32} {'PASS' if v else 'FAIL'}")
    if not a.keep:
        shutil.rmtree(work, ignore_errors=True)
    else:
        print(f"  artifacts kept in {work}")
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
