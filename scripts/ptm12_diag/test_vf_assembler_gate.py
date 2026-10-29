#!/usr/bin/env python3
"""View-factor divisor regression gate for the offline view-factor referee and assembler.

Three arms. A and B drive the ACTUAL referee pipeline end to end -- mc_ref_pp sweep ->
assemble_vf_haworth.py -> CSR -- so a defect anywhere between the ray sampler and the
written matrix is in scope.

  ARM A  CLOSED FORM.  A small horizontal element under a coaxial parallel disk
         (a = 57.735 m at H = 100 m) has the exact answer F = a^2/(a^2 + H^2) =
         0.250000.  The disk is genuinely flat, so tessellating it introduces no
         geometric error; the only approximation is the finite extent of the element,
         O((e0/H)^2) -- 2e-6 at the historical 0.01 m^2, 9e-5 at the CI default of
         2 m^2, both far below any tolerance here.  The disk is cut into enough facets
         that every pair sits at lambda = NRAYS * F_ij of order 1 -- the production
         regime, and the ONLY regime where the view-factor divisor defect is visible.  The assembled
         row 0 must reproduce 0.250000 to within --closed-form-tol-pct.

         SIZING (measured 2026-08-22, and the reason --disk-a0 exists).  Arm A's
         residual is dominated by the REVERSE ray direction, facet -> element, whose
         expected hit count is NRAYS * nfaces * 0.25 * A0 / A_disk.  Shrinking the disk
         for speed without raising A0 makes that count collapse and the residual grows:
         at 19,711 facets and A0 = 0.01 m^2 the shipping estimator read +0.963% / -0.249%
         / -1.700% at NRAYS = 40k / 80k / 120k -- non-monotone, and over the 1% line.
         Raising A0 to 2 m^2 at the same cost pins it at +0.074%.  Never reduce the
         geometry without re-measuring BOTH polarities.

  ARM C  BACK-FACE CULL (Paul's ruling, 2026-08-19).  A ray striking the BACK of a
         one-sided facet has reached the underside of opaque terrain: it is blocked, but
         the facet cannot exchange radiation with the source and is not a partner.  On
         geometry where no such hit is possible the cull must be provably inert; on an
         open heightfield it may only ever REMOVE partner weight.

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

Runtime, measured 2026-08-22 on the Mac Pro with --jobs 2 (a CI-sized machine):
  * shipped defaults, arm A alone .................. 1,032 s   NOT CI-affordable
  * the CI settings below, all three arms .............. 197 s
The CI settings are the ones in .github/workflows/ci.yml, job vf-assembler-gate:
  --disk-nth 90 --disk-k 110 --disk-a0 2.0 --nrays-closed-form 40000
  --nrays-lo 4000 --nrays-hi 40000 --nrays-backface 20000
Measured there: A1 +0.074% (tol 1%), A2 +28.955% (floor 10%), B1 0.044%
(tol 0.1%), B2 71.65% (floor 25%), C1/C2 pass.  The shipped defaults give
A1 +0.583% / A2 +75.155%, reproduced exactly by the 1,032 s run above.

Usage:
  test_vf_assembler_gate.py [--work DIR] [--jobs N] [--mc-ref-pp PATH]
                            [--arm A|B|C|all] [--keep]
                            [--disk-nth N] [--disk-k N] [--disk-a0 M2]
"""
import argparse
import glob
import hashlib
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# find_or_build_mc_ref_pp lives in ensure_mc_ref_pp.py, the single source of
# truth for building/verifying the referee (shared with run_pp_sweep.sh via
# its CLI, and with test_vf_cross_agreement.py). See that module's docstring.
from ensure_mc_ref_pp import find_or_build_mc_ref_pp  # noqa: E402

ASSEMBLER = HERE / "assemble_vf_haworth.py"
REPO = HERE.parent.parent
CRATER_PLY = REPO / "data" / "meshes" / "lunar_crater_80N.ply"

# Closed form for the arm-A geometry.
DISK_H = 100.0
DISK_A = 57.735026918962575          # a^2/(a^2+H^2) == 0.25 exactly
DISK_A0 = 0.01                       # m^2, the receiving element
DISK_NTH = 200                       # azimuthal divisions   (--disk-nth)
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
def write_tmax(path, n, value=100.0):
    """mc_ref_pp's per-face temperature input: int32 count followed by float32 values.
    Only the F_warm384/Qir diagnostic columns use it; the view factors do not."""
    with open(path, "wb") as f:
        np.array([n], dtype="<i4").tofile(f)
        np.full(n, value, dtype="<f4").tofile(f)


def backface_counts(outdir, jobs):
    """Parse mc_ref_pp's BACKFACE summary out of the per-chunk logs. Returns
    (detected, blocked) summed over chunks. Fails loud if the banner is missing --
    a silently absent counter would make arm C vacuous."""
    det = blk = 0
    seen = 0
    for k in range(jobs):
        for ln in (outdir / f"log_{k:02d}.txt").read_text(errors="replace").splitlines():
            if "BACKFACE summary" in ln:
                m = re.search(r"(\d+) back-side hits DETECTED .*?; (\d+) BLOCKED", ln)
                if not m:
                    raise SystemExit(f"FATAL: unparsable BACKFACE banner: {ln}")
                det += int(m.group(1)); blk += int(m.group(2)); seen += 1
    if seen != jobs:
        raise SystemExit(f"FATAL: BACKFACE banner found in {seen}/{jobs} chunk logs. The "
                         "counter is missing, so arm C would be vacuous.")
    return det, blk


def sweep(binary, mesh, tmax, nfaces, nrays, outdir, jobs, no_backface_cull=False,
          keps=0.2):
    """Run the per-partner referee over every face, in `jobs` parallel chunks.

    `keps` is mc_ref_pp's rise-march self-relief threshold, as a fraction of the
    launch face's edge length: a hit whose out-of-plane rise from the launch
    face is below it is dismissed as the face's own local relief and the march
    continues past it. 0.2 is what every caller in this repository passes, and
    it is defensible on a DEM heightfield. It is NOT free on a smoothly curved
    surface, where adjacent facets are near-coplanar and there is no self-relief
    to reject: measured 2026-08-23 on a 648-face spherical cap, the mean terrain
    view factor reads -5.798 percent against the Ingersoll closed form at
    keps = 0.2, -0.832 at 0.05, +0.626 at 0.01 and +0.874 at 0.001, and the
    deficit is INDEPENDENT of ray budget, so it is a filter bias and not noise.
    Parameterized so test_vf_cross_agreement.py can both avoid it and measure
    it; the default is unchanged, so every existing caller is unaffected."""
    outdir.mkdir(parents=True, exist_ok=True)
    chunks = np.array_split(np.arange(nfaces), jobs)
    env = dict(os.environ)
    env.pop("PTM_PP_RING_PLY", None)          # no occluder mesh in either arm
    env.pop("PTM_PP_NO_BACKFACE_CULL", None)
    env["PTM_PP_ORIENT"] = "raw"  # the closed-form test meshes are local-frame and wound raw by design
    if no_backface_cull:
        env["PTM_PP_NO_BACKFACE_CULL"] = "1"
    procs = []
    t0 = time.time()
    for k, ch in enumerate(chunks):
        fl = outdir / f"fl_{k:02d}.txt"
        fl.write_text("\n".join(str(int(i)) for i in ch) + "\n")
        log = open(outdir / f"log_{k:02d}.txt", "w")
        procs.append((subprocess.Popen(
            [binary, str(mesh), str(tmax), str(fl), str(nrays), "1e-4", str(keps), "1",
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
    print(f"  sweep NRAYS={nrays} keps={keps}: {jobs} chunks, {nfaces} faces, "
          f"{time.time()-t0:.0f}s")
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


def arm_c(work, binary, jobs, nrays):
    """Back-face cull (Paul's ruling, 2026-08-19). A ray striking the BACK of a one-sided
    facet has reached the underside of opaque terrain: it is blocked, and the facet is not a
    radiative partner. On geometry where no such hit is possible the cull must be inert."""
    print("\n=== ARM C: back-face cull -- inert where impossible, counted where real ===")
    ok = True

    # C1: the coaxial disk. Element and disk face each other, so no ray can reach a back
    # side. Detected must be 0 AND the raw sweep must be byte-identical with the cull off,
    # which also proves the cull cannot perturb arm A's closed-form number.
    mesh = work / "disk.ply"
    if not mesh.is_file():
        make_disk_mesh(mesh)
    tmax = work / "disk_tmax.bin"
    if not tmax.is_file():
        write_tmax(tmax, 3 + DISK_NTH * (2 * DISK_K - 1))
    nfaces = 3 + DISK_NTH * (2 * DISK_K - 1) - 2   # 1 element + disk facets
    nfaces = 1 + DISK_NTH * (2 * DISK_K - 1)
    digests = {}
    det = {}
    for tag, off in (("cull", False), ("nocull", True)):
        d = work / f"bf_disk_{tag}"
        sweep(binary, mesh, tmax, nfaces, nrays, d, jobs, no_backface_cull=off)
        det[tag] = backface_counts(d, jobs)[0]
        h = hashlib.md5()
        for k in range(jobs):
            h.update((d / f"pp_{k:02d}.bin").read_bytes())
        digests[tag] = h.hexdigest()
    same = digests["cull"] == digests["nocull"]
    print(f"  disk ({nfaces} faces): back-side hits detected = {det['cull']} "
          f"(cull off: {det['nocull']}); sweep digest "
          f"{'IDENTICAL' if same else 'DIFFERS'} ({digests['cull'][:12]})")
    c1 = (det["cull"] == 0 and det["nocull"] == 0 and same)
    print(f"  [C1] cull inert on front-to-front geometry : {'PASS' if c1 else 'FAIL'}")
    ok &= c1

    # C2: the open crater heightfield. Back hits are real here. The cull may only ever
    # REMOVE partner weight, never add it -- an increase would mean the test is inverted.
    cmesh = work / "crater80N_tri.ply"
    if not cmesh.is_file():
        convert_ascii_ply(CRATER_PLY, cmesh)
    ctmax = work / "crater_tmax.bin"
    if not ctmax.is_file():
        write_tmax(ctmax, 12800)
    sig = {}
    cdet = {}
    for tag, off in (("cull", False), ("nocull", True)):
        d = work / f"bf_crater_{tag}"
        g = sweep(binary, cmesh, ctmax, 12800, nrays, d, jobs, no_backface_cull=off)
        cdet[tag] = backface_counts(d, jobs)
        csr = assemble(cmesh, g, nrays, work / f"bf_crater_{tag}.csr", "tried")
        sig[tag] = mean_sigma_f(csr)
    print(f"  crater80N (12800 faces, open): back-side hits detected = {cdet['cull'][0]} "
          f"(blocked {cdet['cull'][1]}); cull off detects {cdet['nocull'][0]}, blocks "
          f"{cdet['nocull'][1]}")
    print(f"  mean SigmaF: cull ON {sig['cull']:.6f}  vs OFF {sig['nocull']:.6f}  "
          f"({100*(sig['cull']/sig['nocull']-1):+.3f}%)")
    c2 = (cdet["cull"][0] == cdet["nocull"][0]          # detection is cull-independent
          and cdet["cull"][1] == cdet["cull"][0]        # ON blocks everything it detects
          and cdet["nocull"][1] == 0                    # OFF blocks nothing
          and sig["cull"] <= sig["nocull"])             # cull can only remove weight
    print(f"  [C2] cull removes only, and OFF admits what it keeps : "
          f"{'PASS' if c2 else 'FAIL'}")
    return ok and c2


def main():
    global DISK_NTH, DISK_K, DISK_A0
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default=None, help="scratch directory (default: a temp dir)")
    ap.add_argument("--jobs", type=int, default=min(20, os.cpu_count() or 4))
    ap.add_argument("--mc-ref-pp", default=None, help="prebuilt referee binary")
    ap.add_argument("--arm", choices=("A", "B", "C", "all"), default="all")
    # Arm-A tessellation. The controlling variable for the view-factor divisor defect is
    # lambda = NRAYS * F_ij, NOT the ray budget alone: a coarse disk with a big
    # budget puts every pair in the well-sampled regime and hides the bug
    # completely. So --disk-nth/--disk-k and --nrays-closed-form must be scaled
    # TOGETHER (roughly, NRAYS proportional to the facet count), and any reduced
    # setting has to be justified by the DEFECTIVE arm still reading >= the
    # --closed-form-defect-min-pct floor. The gate enforces that automatically:
    # arm A2 fails if the teeth have gone.
    ap.add_argument("--disk-nth", type=int, default=DISK_NTH,
                    help="azimuthal divisions of the arm-A disk (default %(default)s)")
    ap.add_argument("--disk-k", type=int, default=DISK_K,
                    help="radial rings of the arm-A disk (default %(default)s)")
    # Element area. The closed form F = a^2/(a^2+H^2) is the DIFFERENTIAL-element
    # result, so this must stay small against H^2 = 1e4 m^2: the error is
    # O((e0/H)^2), i.e. 2.3e-6 at 0.01 m^2 and 2.3e-4 at 1 m^2, both far under any
    # tolerance here. It is a knob because it sets how often the REVERSE ray
    # direction (facet -> element) registers a hit, and that sampling term is what
    # dominates arm A's residual: expected reverse hits ~ NRAYS * nfaces * 0.25 *
    # A0 / A_disk, and the residual falls as 1/sqrt of it. Shrinking the disk
    # without raising A0 makes the residual worse, not better.
    ap.add_argument("--disk-a0", type=float, default=DISK_A0,
                    help="area [m^2] of the arm-A receiving element (default %(default)s)")
    ap.add_argument("--nrays-closed-form", type=int, default=200000)
    ap.add_argument("--nrays-lo", type=int, default=4000)
    ap.add_argument("--nrays-backface", type=int, default=20000,
                    help="Ray budget for arm C. Back-face incidence is a GEOMETRIC\nproperty, not a convergence one, so this need not match the other arms.")
    ap.add_argument("--nrays-hi", type=int, default=200000)
    # Arm A's residual is NOT a random draw: mc_ref_pp uses deterministic Halton QMC, so a
    # given binary/Embree-ISA pair reproduces it bit for bit. Measured 2026-08-19 with
    # Embree 4 (Homebrew, the build production used): +0.583%, against a 1.0% tolerance.
    # The residual is dominated by the REVERSE direction (partner -> element), where a
    # single stray hit carries A_j/A_0 ~ 10.5 times the element's own weight; the forward
    # direction alone lands near 0.3%. If this arm ever fails at 1.0-1.5%, suspect the
    # Embree ISA before suspecting the estimator, and raise --nrays-closed-form to confirm:
    # a true bias does not shrink with ray budget, this noise term does.
    ap.add_argument("--closed-form-tol-pct", type=float, default=1.0)
    ap.add_argument("--closed-form-defect-min-pct", type=float, default=10.0)
    ap.add_argument("--invariance-tol-pct", type=float, default=0.1)
    ap.add_argument("--invariance-defect-min-pct", type=float, default=25.0)
    ap.add_argument("--keep", action="store_true", help="keep the scratch directory")
    a = ap.parse_args()

    DISK_NTH, DISK_K, DISK_A0 = a.disk_nth, a.disk_k, a.disk_a0

    if not ASSEMBLER.is_file():
        raise SystemExit(f"FATAL: {ASSEMBLER} not found")
    work = Path(a.work) if a.work else Path(
        os.environ.get("TMPDIR", "/tmp")) / f"vfgate_{os.getpid()}"
    # Only ever delete a directory this run created -- never a pre-existing --work dir,
    # which may hold results that have nothing to do with the gate.
    we_made_it = not work.exists()
    work.mkdir(parents=True, exist_ok=True)
    print(f"[vf-gate] work dir {work}  jobs={a.jobs}")
    binary = find_or_build_mc_ref_pp(a.mc_ref_pp, work)

    t0 = time.time()
    results = {}
    if a.arm in ("A", "all"):
        results["A closed form"] = arm_a(work, binary, a.jobs, a.nrays_closed_form,
                                         a.closed_form_tol_pct, a.closed_form_defect_min_pct)
    if a.arm in ("B", "all"):
        results["B ray-budget invariance"] = arm_b(work, binary, a.jobs, a.nrays_lo, a.nrays_hi,
                                                   a.invariance_tol_pct,
                                                   a.invariance_defect_min_pct)
    if a.arm in ("C", "all"):
        results["C back-face cull"] = arm_c(work, binary, a.jobs, a.nrays_backface)

    print(f"\n=== SUMMARY ({time.time()-t0:.0f}s) ===")
    for k, v in results.items():
        print(f"  {k:<32} {'PASS' if v else 'FAIL'}")
    # One greppable verdict line, so a CI step can check the log as well as the
    # exit status: an edit that drops the assertions would exit 0 silently.
    # Deliberately worded with no "fail" substring on the passing branch --
    # ptm_r2_acceptance.py once read its own "0 failed" as a failure.
    if all(results.values()):
        print("[vf-assembler-gate] PASS: every arm met its threshold in BOTH polarities.")
    else:
        print("[vf-assembler-gate] at least one arm did NOT meet its threshold (see above).")
    if a.keep or not we_made_it:
        print(f"  artifacts kept in {work}"
              + ("" if we_made_it else " (pre-existing directory; never auto-removed)"))
    else:
        shutil.rmtree(work, ignore_errors=True)
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
