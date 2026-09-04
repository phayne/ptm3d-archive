#!/usr/bin/env python3
"""Cross-agreement gate: the IN-ENGINE view-factor builder against the OFFLINE referee.

WHY THIS EXISTS
---------------
PTM3D computes view factors by two entirely separate implementations and, until
2026-08-23, never compared their answers:

  * IN-ENGINE   src/cpp/radiation/view_factor.cpp -- analytic pair integrals,
                Embree occlusion, M1 centroid far field + M2 double-area near
                field. Gated against closed forms by test_vf_production.
  * OFFLINE     scripts/ptm12_diag/mc_ref_pp.cpp + assemble_vf_haworth.py --
                cosine-weighted Halton ray gather, then a symmetrized assembly.
                Gated against closed forms by test_vf_assembler_gate.py.

Each gate is necessary and neither is sufficient, and the project has now paid
for that twice in one week:

  FC-08 (2026-08-19)  the offline assembler inflated the far field by up to
      exactly 2x. Reciprocity passed. Sum_j F_ij <= 1 passed. The Ingersoll
      bowl gate passed -- it never calls that pipeline at all.
  M2    (2026-08-23)  the in-engine near-field estimator read -21.7 percent on
      edge/vertex-touching pairs. Reciprocity failed, but only after somebody
      thought to look; no gate scored the MAGNITUDE of a near-field entry.

Either defect would have been loud the moment the two matrices were subtracted.
This gate subtracts them.

THE TOLERANCE IS MEASURED, NOT CHOSEN
-------------------------------------
The offline side is a quadrature over a finite ray budget, so it disagrees with
ITSELF at two budgets. That self-disagreement is the floor below which no
cross-comparison can mean anything, and it is what sets the tolerance here: the
gate assembles the SAME sweep geometry at NRAYS and 2 x NRAYS, measures the
spread, and requires the engine-vs-referee difference to sit within
--noise-factor times it. Nothing in this file is a hand-picked epsilon.

mc_ref_pp draws from a DETERMINISTIC Halton sequence with no seed, identical for
every face, so a repeat-seed or split-half spread does not exist to be measured
-- the estimator has no run-to-run randomness at all. Its error is QMC
discrepancy, which is a function of the ray budget, and the budget doubling is
therefore the right and only handle on it. (This also makes the noise arm a
free re-run of the assembler gate's ray-budget-invariance check: an estimator
whose weight depends on whether a direction happened to register a hit -- FC-08
-- is not budget-invariant, and shows up here as a large "noise" that then
swallows the tolerance. So the gate also prints the noise, and a noise figure
that has grown is itself a finding.)

Per-ENTRY comparison is restricted to entries the referee actually resolved.
The control variable is lambda = NRAYS * F_ij, the expected hit count: at
lambda = 100 the referee's own relative standard error is 10 percent, and below
a few tens an entry is a coin flip, not a measurement. Comparing those would
report on the ray budget, not on either estimator. The cut is stated, applied
to BOTH sides symmetrically, and the number of entries it removes is printed.

  [X0]  the per-entry population is not empty (a vacuous arm is not a PASS)
  [X1]  mean row-sum (terrain view factor) difference
  [X2]  worst per-face row-sum difference
  [X3]  p99 per-entry relative difference over lambda-resolved entries
  [X4]  BOTH sides against the Ingersoll closed form -- an absolute anchor, so
        the gate cannot be satisfied by two implementations agreeing on the
        same wrong answer
  [X5]  the referee's own self-relief filter, scored rather than assumed away
  [X6]  the same cross-comparison at a LOW ray budget, where FC-08 lives

FC-08 IS INVISIBLE IN THE WELL-SAMPLED REGIME, SO THE GATE GOES LOOKING FOR IT
------------------------------------------------------------------------------
The FC-08 assembler defect divides the symmetrized pair by the number of ray
directions that HIT rather than by 2, so its inflation factor is
2 - (p1 + p2)/2, p_k = P(direction k registers at least one hit). The control
variable is lambda = NRAYS * F_ij. At lambda >> 1 both directions always hit,
the factor is exactly 1, and THE DEFECT DOES NOT EXIST. Arms [X1] to [X4] run
at NRAYS = 400,000 on a cap whose largest F_ij is 8.7e-4, i.e. lambda ~ 80
everywhere -- measured 2026-08-23, reinstating the defect
(--referee-divisor hits) moves the mean row sum by 4e-8 and every one of those
arms still PASSES. That is the same blindness the Ingersoll bowl gate has, and
shipping it unremarked would be the FC-08 story a second time.

[X6] therefore repeats the row-sum comparison at --nrays-fc08 (default 2,000),
where lambda ~ 0.4 and the defect is at its full strength. The correct
estimator is unbiased at EVERY budget -- that is what makes it correct -- so the
engine and the referee still agree there; the defective one inflates by ~2x and
[X6] turns red. It is the only arm in this file whose tolerance is loose, and
deliberately: it is separating 1x from 2x, not measuring an accuracy.

TWO SCALES, AND THEY ARE NOT INTERCHANGEABLE
--------------------------------------------
Ray-budget noise is the right scale for a PER-ENTRY comparison and the wrong
one for a ROW SUM. The row sums carry a systematic, convergent difference that
has nothing to do with the ray budget: the referee launches its rays from the
face CENTROID, while the in-engine builder integrates over the launch face's
AREA, and on a curved faceted surface those are different quantities at first
order in the facet size. Measured on this cap at NRAYS = 400,000, keps = 0.01:

    faces   engine     referee    gap (1e-3)   engine vs f   referee vs f
      342   0.141417   0.139674      1.743       +2.527%       +1.264%
      648   0.139915   0.138817      1.098       +1.438%       +0.642%
     1404   0.138797   0.138197      0.600       +0.628%       +0.193%

The gap falls like the mesh refines and both sides converge on the closed form,
so it is discretization, not a scale error -- an FC-08-shaped defect would hold
its size (or grow with lambda) instead. So [X1] and [X2] are scored against the
DISCRETIZATION scale, self-calibrated per mesh as
    --disc-factor  x  max(|engine - f|, |referee - f|)
i.e. the two implementations may not disagree with EACH OTHER by more than they
each disagree with the truth. The measured ratio is 0.50 / 0.55 / 0.69 across a
4x range of mesh resolution, so 1.5 is roughly 2x headroom and shrinks nothing.
That tolerance cannot run away, because [X4] independently caps each side's
closed-form error at 2 percent.

Geometry: a spherical cap (Ingersoll & Svitek 1992), whose terrain view factor
f = d^2/(d^2 + r^2) is uniform over the whole cap and known exactly. Every face
exchanges with every other, so the comparison is over a full matrix rather than
one privileged row.

Usage:
  test_vf_cross_agreement.py [--work DIR] [--jobs N] [--nrays N]
                             [--cap-rings R] [--cap-az A] [--quad-order M]
                             [--noise-factor F] [--mc-ref-pp PATH]
                             [--vf-dump-sparse PATH] [--keep]
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Reuse the referee driver wholesale. A second copy of "how to run mc_ref_pp"
# is a second thing to keep in step with mc_ref_pp.
from test_vf_assembler_gate import (           # noqa: E402
    assemble, read_csr, sweep, write_binary_ply, write_tmax,
)
# find_or_build_mc_ref_pp itself lives in ensure_mc_ref_pp.py -- the single
# source of truth shared with run_pp_sweep.sh's staleness guard.
from ensure_mc_ref_pp import find_or_build_mc_ref_pp  # noqa: E402

REPO = HERE.parent.parent

# Cap geometry. d/D and the diameter match arm [I] of test_vf_production, so the
# two gates score the same closed form on the same shape.
CAP_DD = 0.2
CAP_DIAMETER = 100.0


def make_cap_mesh(path, n_rings, n_az):
    """A pure spherical cap of depth d = CAP_DD * D and rim radius D/2, opening
    upward, normals pointing INTO the bowl. No apron: the apron in
    bowl_crater_ptm.cpp is needed for a thermal sweep and destroys the
    uniform-f property near the rim, and this gate scores every face."""
    d = CAP_DD * CAP_DIAMETER
    r_rim = 0.5 * CAP_DIAMETER
    R = (r_rim * r_rim + d * d) / (2.0 * d)
    z_c = R - d                                    # sphere center on +z
    theta_max = np.arcsin(r_rim / R)

    # Rings uniform in POLAR ANGLE, and the apex at z = z_c - R = -d. Both are
    # copied from buildSphericalCap() in test_vf_production.cpp on purpose, so
    # the two gates score the same closed form on the same shape rather than on
    # two shapes that are each "a cap". (The apex is not a free choice: putting
    # it at z = 0 while ring 1 sits at -d builds a bowl with a d-tall SPIKE in
    # the middle. Both pipelines then measure the spike, agree that the mean
    # terrain view factor is 0.1785 instead of 0.1379, and arm [X4] correctly
    # turns red -- which is how this comment came to be written.)
    V = [[0.0, 0.0, z_c - R]]
    ring_start = [0]
    for k in range(1, n_rings + 1):
        th = theta_max * k / n_rings
        rho = R * np.sin(th)
        z = z_c - R * np.cos(th)
        ring_start.append(len(V))
        for t in range(n_az):
            phi = 2.0 * np.pi * t / n_az
            V.append([rho * np.cos(phi), rho * np.sin(phi), z])

    F = []
    for t in range(n_az):                          # innermost fan
        F.append([0, ring_start[1] + t, ring_start[1] + (t + 1) % n_az])
    for k in range(1, n_rings):
        A, B = ring_start[k], ring_start[k + 1]
        for t in range(n_az):
            t2 = (t + 1) % n_az
            F.append([A + t, B + t, B + t2])
            F.append([A + t, B + t2, A + t2])

    V = np.asarray(V, float)
    F = np.asarray(F, np.int32)
    p0, p1, p2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    n = np.cross(p1 - p0, p2 - p0)
    ln = np.linalg.norm(n, axis=1)
    if not (ln > 0).all():
        raise SystemExit("FATAL: cap mesh has a degenerate face")
    n = n / ln[:, None]
    c = (p0 + p1 + p2) / 3.0
    # Every face must look at the sphere center, or the two pipelines are being
    # handed different surfaces and nothing below means anything.
    to_center = np.array([0.0, 0.0, z_c])[None, :] - c
    bad = int((np.einsum("ij,ij->i", n, to_center) <= 0).sum())
    if bad:
        raise SystemExit(f"FATAL: {bad} cap normals point out of the bowl; fix the winding, "
                         "do not flip them here -- vf_dump_sparse and mc_ref_pp both take "
                         "the winding normal and a silent flip would desynchronize them")
    write_binary_ply(path, V, F)
    f_closed = d * d / (d * d + r_rim * r_rim)
    print(f"  cap mesh: {len(F)} faces, {len(V)} vertices, d/D = {CAP_DD}, "
          f"area {0.5 * ln.sum():.3f} m^2")
    print(f"  CLOSED FORM  terrain VF f = d^2/(d^2+r^2) = {f_closed:.10f}  "
          f"(F_sky = {1.0 - f_closed:.10f})")
    return len(F), f_closed


def dense_from_csr(path, n_expect):
    n, indptr, cols, data = read_csr(path)
    if n != n_expect:
        raise SystemExit(f"FATAL: {path} has N={n}, expected {n_expect}")
    d = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        s, e = indptr[i], indptr[i + 1]
        d[i, cols[s:e]] = data[s:e]
    return d


def compare(a, b, lam_floor, nrays):
    """Row-sum and per-entry differences between two dense F matrices.

    `lam_floor` is a cut on lambda = NRAYS * F, the referee's expected hit count
    for the entry. It is applied to the MAXIMUM of the two sides, so neither
    implementation can hide an entry by reporting it as zero."""
    rs_a, rs_b = a.sum(1), b.sum(1)
    row_abs = np.abs(rs_a - rs_b)
    m = np.maximum(a, b)
    sig = m * nrays >= lam_floor
    rel = np.zeros_like(m)
    rel[sig] = np.abs(a[sig] - b[sig]) / m[sig]
    return {
        "row_mean_abs": float(row_abs.mean()),
        "row_worst_abs": float(row_abs.max()),
        "row_mean_rel": float(row_abs.mean() / max(rs_b.mean(), 1e-300)),
        "entry_p99_rel": float(np.percentile(rel[sig], 99)) if sig.any() else 0.0,
        "entry_max_rel": float(rel[sig].max()) if sig.any() else 0.0,
        "n_significant": int(sig.sum()),
        "n_nonzero": int((m > 0).sum()),
    }


def verdict(tag, what, measured, limit, results, unit=""):
    ok = measured <= limit
    print(f"  [{tag}] {what:<52} measured {measured:.6g}{unit}  limit {limit:.6g}{unit}   "
          f"{'PASS' if ok else 'FAILED'}")
    results[tag] = ok


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work", default="")
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--nrays", type=int, default=200000,
                   help="referee ray budget for the LOW arm; the noise arm doubles it. "
                        "Sized from the measured lambda distribution, not chosen: on the "
                        "default cap the largest F_ij is 8.7e-4, so at 40k rays NOT ONE "
                        "entry reaches lambda = 50 and the per-entry arm would be vacuous")
    p.add_argument("--cap-rings", type=int, default=14)
    p.add_argument("--cap-az", type=int, default=24)
    p.add_argument("--keps", type=float, default=0.01,
                   help="mc_ref_pp rise-march self-relief threshold for the COMPARISON "
                        "sweeps. Not the 0.2 every other caller passes -- see arm [X5]")
    p.add_argument("--keps-production", type=float, default=0.2,
                   help="the value production sweeps use; scored by arm [X5]")
    p.add_argument("--keps-deficit-max-pct", type=float, default=8.0)
    p.add_argument("--min-significant", type=int, default=1000,
                   help="fewest lambda-resolved entries the per-entry arm will accept "
                        "before declaring itself vacuous and FAILING")
    p.add_argument("--quad-order", type=int, default=0,
                   help="override radiation.nearfield_quadrature_order in the ENGINE "
                        "side only. DIAGNOSTIC, NOT A NEGATIVE CONTROL -- see the NOT "
                        "COVERED banner: this gate cannot referee the near field and "
                        "--quad-order 1 is expected to stay GREEN")
    p.add_argument("--noise-factor", type=float, default=3.0,
                   help="PER-ENTRY engine-vs-referee difference may be this many times "
                        "the measured referee self-disagreement")
    p.add_argument("--disc-factor", type=float, default=1.5,
                   help="ROW-SUM engine-vs-referee difference may be this many times "
                        "each side's own distance from the closed form")
    p.add_argument("--lambda-floor", type=float, default=50.0,
                   help="minimum NRAYS*F for an entry to enter the per-entry statistic")
    p.add_argument("--closed-form-tol-pct", type=float, default=2.0)
    p.add_argument("--nrays-fc08", type=int, default=2000,
                   help="ray budget for arm [X6]. Small ON PURPOSE: lambda = NRAYS*F "
                        "must be of order 1 for the FC-08 defect to exist at all")
    p.add_argument("--fc08-tol-pct", type=float, default=15.0,
                   help="arm [X6] row-sum agreement. Loose by design: it separates a "
                        "1x from a 2x, and 2,000 rays is genuinely noisy")
    p.add_argument("--referee-divisor", choices=("tried", "hits"), default="tried",
                   help="NEGATIVE CONTROL ONLY. 'hits' restores the pre-2026-08-19 "
                        "FC-08 assembler defect (divide the symmetrized pair by the "
                        "number of ray directions that HIT, not by 2), which inflates "
                        "the far field by up to 2x. The gate must go RED.")
    p.add_argument("--mc-ref-pp", default="")
    p.add_argument("--vf-dump-sparse", default="")
    p.add_argument("--keep", action="store_true")
    a = p.parse_args()

    t_start = time.time()
    work = Path(a.work) if a.work else Path(os.environ.get("TMPDIR", "/tmp")) / "vf_xagree"
    work.mkdir(parents=True, exist_ok=True)

    # An EXPLICITLY named binary that cannot be used is fatal, for the same
    # reason it is in the assembler gate: naming one is how you gate a specific
    # build, and quietly substituting another returns a PASS about the wrong
    # thing.
    dump = a.vf_dump_sparse or os.environ.get("PTM_VF_DUMP_SPARSE") or ""
    if dump and not (Path(dump).is_file() and os.access(dump, os.X_OK)):
        raise SystemExit(f"FATAL: --vf-dump-sparse {dump} is missing or not executable.")
    if not dump:
        for cand in (REPO / "build" / "bin" / "vf_dump_sparse",
                     REPO / "build-gpu" / "bin" / "vf_dump_sparse"):
            if cand.is_file() and os.access(str(cand), os.X_OK):
                dump = str(cand)
                break
    if not dump:
        raise SystemExit("FATAL: vf_dump_sparse not found. Build it:\n"
                         "  cmake --build build --target vf_dump_sparse\n"
                         "or pass --vf-dump-sparse PATH.")
    print(f"  vf_dump_sparse: {dump}")

    print("\n=== GEOMETRY ===")
    mesh = work / "cap.ply"
    nfaces, f_closed = make_cap_mesh(mesh, a.cap_rings, a.cap_az)
    tmax = work / "tmax.bin"
    write_tmax(tmax, nfaces)

    binary = find_or_build_mc_ref_pp(a.mc_ref_pp, work)

    print("\n=== ENGINE: the in-engine sparse builder ===")
    engine_csr = work / "engine.csr"
    cmd = [dump, str(mesh), str(engine_csr)]
    if a.quad_order:
        cmd += ["--quad-order", str(a.quad_order)]
        print(f"  DIAGNOSTIC: engine quadrature order forced to {a.quad_order}. This is "
              f"NOT a negative control for this gate -- it cannot referee the near "
              f"field. See the NOT COVERED banner before reading anything into the "
              f"verdicts below.")
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write("  " + r.stdout.replace("\n", "\n  ").rstrip() + "\n")
    if r.returncode != 0:
        sys.stderr.write(r.stderr)
        raise SystemExit("FATAL: vf_dump_sparse failed")
    print(f"  engine build: {time.time() - t0:.1f}s")

    print("\n=== REFEREE: mc_ref_pp sweep + assemble_vf_haworth.py, two ray budgets ===")
    if a.referee_divisor != "tried":
        print(f"  *** NEGATIVE CONTROL: assembler symmetrization divisor = "
              f"'{a.referee_divisor}'. This is the FC-08 defect, reinstated on purpose. "
              f"A PASS below would mean the gate cannot see a 2x far-field inflation. ***")
    nr_lo, nr_hi = a.nrays, 2 * a.nrays
    ref = {}
    for nr, tag, keps in ((nr_lo, "lo", a.keps), (nr_hi, "hi", a.keps),
                          (nr_hi, "prod_keps", a.keps_production),
                          (a.nrays_fc08, "fc08", a.keps)):
        glob_pat = sweep(binary, mesh, tmax, nfaces, nr, work / f"sweep_{tag}", a.jobs,
                         keps=keps)
        ref[tag] = assemble(mesh, glob_pat, nr, str(work / f"ref_{tag}.csr"),
                            a.referee_divisor)

    print("\n=== MATRICES ===")
    E = dense_from_csr(engine_csr, nfaces)
    Rlo = dense_from_csr(ref["lo"], nfaces)
    Rhi = dense_from_csr(ref["hi"], nfaces)
    Rpk = dense_from_csr(ref["prod_keps"], nfaces)
    for name, M in (("engine", E),
                    (f"referee {nr_lo} keps={a.keps}", Rlo),
                    (f"referee {nr_hi} keps={a.keps}", Rhi),
                    (f"referee {nr_hi} keps={a.keps_production}", Rpk)):
        rs = M.sum(1)
        print(f"  {name:<24} nnz {int((M > 0).sum()):>9}  mean row sum {rs.mean():.6f}  "
              f"(closed form {f_closed:.6f}, {100 * (rs.mean() / f_closed - 1):+.3f}%)")

    print("\n=== NOISE: the referee against ITSELF at two ray budgets ===")
    print("  This is the floor. Nothing below it is a statement about either estimator.")
    noise = compare(Rlo, Rhi, a.lambda_floor, nr_lo)
    print(f"  entries with lambda = NRAYS*F >= {a.lambda_floor:g}: "
          f"{noise['n_significant']} of {noise['n_nonzero']} nonzero "
          f"({100.0 * noise['n_significant'] / max(noise['n_nonzero'], 1):.1f}%)")
    print(f"  row-sum mean |d| {noise['row_mean_abs']:.3e}  worst {noise['row_worst_abs']:.3e}  "
          f"per-entry p99 rel {noise['entry_p99_rel']:.3e}  max {noise['entry_max_rel']:.3e}")

    print("\n=== CROSS: the in-engine builder against the referee (NRAYS = "
          f"{nr_hi}) ===")
    cross = compare(E, Rhi, a.lambda_floor, nr_hi)
    print(f"  entries with lambda >= {a.lambda_floor:g}: {cross['n_significant']} of "
          f"{cross['n_nonzero']} nonzero")
    print(f"  row-sum mean |d| {cross['row_mean_abs']:.3e}  worst {cross['row_worst_abs']:.3e}  "
          f"per-entry p99 rel {cross['entry_p99_rel']:.3e}  max {cross['entry_max_rel']:.3e}")

    print("\n=== VERDICTS ===")
    results = {}
    k = a.noise_factor
    e_err = abs(100.0 * (E.sum(1).mean() / f_closed - 1.0))
    r_err = abs(100.0 * (Rhi.sum(1).mean() / f_closed - 1.0))
    print(f"  tolerances are {k:g} x the MEASURED referee self-disagreement above")

    # A per-entry arm with nothing in its population is not a PASS, it is a
    # measurement that did not happen -- and "0 <= 0 PASS" is exactly the silent
    # green this project keeps paying for. It is checked FIRST and it is fatal.
    if cross["n_significant"] < a.min_significant:
        print(f"  [X0] per-entry population                              measured "
              f"{cross['n_significant']} entries  minimum {a.min_significant}   FAILED")
        print(f"       lambda = NRAYS * F_ij never reaches {a.lambda_floor:g} on enough "
              f"entries at NRAYS = {nr_hi}. The per-entry arm below would be VACUOUS.\n"
              f"       Raise --nrays or coarsen the cap; do NOT lower --lambda-floor to "
              f"make it pass, that only compares noise to noise.")
        results["X0"] = False
    else:
        print(f"  [X0] per-entry population                              measured "
              f"{cross['n_significant']} entries  minimum {a.min_significant}   PASS")
        results["X0"] = True

    # Row sums: the discretization scale (see the module docstring). Printed as
    # a ratio too, because that ratio is the quantity the docstring's
    # convergence table tracks and the thing a future reader should re-measure.
    e_abs = abs(E.sum(1).mean() - f_closed)
    r_abs = abs(Rhi.sum(1).mean() - f_closed)
    disc = max(e_abs, r_abs)
    print(f"  row-sum tolerances are {a.disc_factor:g} x the discretization scale "
          f"max(|engine - f|, |referee - f|) = {disc:.3e}")
    print(f"    measured gap / discretization scale = "
          f"{cross['row_mean_abs'] / max(disc, 1e-300):.3f} "
          f"(0.50 to 0.69 across 342 to 1404 faces, 2026-08-23)")
    verdict("X1", "row-sum mean |difference|", cross["row_mean_abs"],
            a.disc_factor * disc, results)
    verdict("X2", "row-sum worst |difference|", cross["row_worst_abs"],
            2.0 * a.disc_factor * disc, results)
    verdict("X3", "per-entry p99 relative difference", cross["entry_p99_rel"],
            k * noise["entry_p99_rel"], results)

    # X4 is the anchor: two implementations that agree with each other and
    # disagree with the closed form are two implementations that are both wrong.
    verdict("X4a", "engine mean terrain VF vs Ingersoll closed form", e_err,
            a.closed_form_tol_pct, results, "%")
    verdict("X4b", "referee mean terrain VF vs Ingersoll closed form", r_err,
            a.closed_form_tol_pct, results, "%")

    # [X5] The referee's rise-march self-relief filter, scored rather than
    # avoided. mc_ref_pp dismisses a hit whose out-of-plane rise from the launch
    # face is below keps * edge as the face's own local relief. On a DEM that is
    # what it is for. On a smoothly curved surface adjacent facets are
    # near-coplanar and there IS no self-relief, so the filter deletes real
    # near-field partners: measured on this cap, mean terrain VF -5.798% at
    # keps = 0.2 (what run_pp_sweep.sh and test_vf_assembler_gate.py both pass),
    # -0.832% at 0.05, +0.626% at 0.01, +0.874% at 0.001 -- monotone, and
    # INDEPENDENT of ray budget, so it is a filter bias and not noise. This arm
    # bounds it so it cannot grow unnoticed. It does NOT license reading the
    # number off onto a production DEM run: severity is a property of how
    # coplanar the neighbors are, and a real heightfield is the opposite end of
    # that scale from an analytic sphere.
    pk_err = 100.0 * (Rpk.sum(1).mean() / Rhi.sum(1).mean() - 1.0)
    print(f"  referee at production keps = {a.keps_production:g} reads "
          f"{pk_err:+.3f}% against the same sweep at keps = {a.keps:g}")
    verdict("X5", "referee self-relief filter deficit at production keps",
            abs(pk_err), a.keps_deficit_max_pct, results, "%")

    Rfc = dense_from_csr(ref["fc08"], nfaces)
    lam_typ = a.nrays_fc08 * float(np.median(Rhi[Rhi > 0]))
    fc_err = 100.0 * (Rfc.sum(1).mean() / E.sum(1).mean() - 1.0)
    print(f"  low-budget arm: NRAYS = {a.nrays_fc08}, median lambda = NRAYS * F_ij = "
          f"{lam_typ:.2f}  (the FC-08-sensitive regime)")
    print(f"  referee mean row sum {Rfc.sum(1).mean():.6f} against engine "
          f"{E.sum(1).mean():.6f}: {fc_err:+.3f}%")
    verdict("X6", "engine vs referee row sums at low ray budget", abs(fc_err),
            a.fc08_tol_pct, results, "%")

    print("\n=== NOT COVERED BY THIS GATE ===")
    print("  * THE NEAR-FIELD ESTIMATOR. Not an oversight and not fixable here: the\n"
          "    referee launches its rays from the face CENTROID, so on a near pair it\n"
          "    estimates the POINT view factor there, not the area average -- the same\n"
          "    one-sided approximation the 2026-08-23 M2 fix removed from the engine.\n"
          "    It therefore AGREES BETTER with the estimator that shares its bias.\n"
          "    Measured on this cap (near pairs, Ns <= 5, lambda >= 50, 2026-08-23):\n"
          "        engine quadrature order 1 (= M1 centroid)  mean rel |d| 0.0349\n"
          "        engine quadrature order 3 (production M2)  mean rel |d| 0.0474\n"
          "    -- order 1 is the WRONG estimator (it reads +51.7% on the touching-pair\n"
          "    closed form) and this comparison prefers it. Running this gate with\n"
          "    --quad-order 1 is GREEN, and that green means nothing. The near-field\n"
          "    estimator is gated by test_vf_production arms [T], [W] and [H4], where\n"
          "    order 1 is correctly red.")
    print("  * the H-matrix and Monte-Carlo in-engine builders. This dumps the SPARSE\n"
          "    one; test_vf_production arm [H] scores sparse against H-matrix.")
    print("  * neighbor truncation. vf_dump_sparse runs with adaptive_neighbors OFF, so\n"
          "    the production selector's deliberate 95%-of-energy capture is NOT in\n"
          "    scope here; test_vf_production arm [D2] scores that.")
    print("  * occlusion on a concave-but-unshadowed cap. Every pair on this geometry is\n"
          "    mutually visible, so the two pipelines' DIFFERENT shadow-ray offsets are\n"
          "    not exercised. That is a real remaining gap, not an oversight.")

    ok = all(results.values())
    print(f"\n  wall time {time.time() - t_start:.0f}s "
          f"({nfaces} faces, {a.jobs} jobs, NRAYS {nr_lo} and {nr_hi})")
    if not a.keep:
        shutil.rmtree(work, ignore_errors=True)
    if ok:
        print("[test_vf_cross_agreement] PASS: the in-engine builder and the offline\n"
              "                          referee agree within the referee's own noise.")
        return 0
    print("[test_vf_cross_agreement] at least one verdict FAILED. The two independent\n"
          "                          view-factor implementations DISAGREE beyond the\n"
          "                          referee's measured noise. Do not tune a tolerance\n"
          "                          here; one of them is wrong.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
