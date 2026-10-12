#!/usr/bin/env python3
"""Accuracy metrics for the Ingersoll bowl (Moon3D R1, C3).

Three metrics, kept separate on purpose (the manuscript's "0.5%" was the PEAK
per-face spread, not the mean bias; revisions queue item 5):

  1. Floor sky view factor vs sin^2(alpha), alpha = atan(D / 2d):
       mean_bias_pct  area-weighted floor F_sky bias
       peak_dev_pct   worst single floor face
  2. Shadowed-floor temperature / emitted flux vs Ingersoll (1992) Eq. 8
       (bowl_crater_ptm PTM_BOWL_RADIOSITY=1 -> radiosity_eq8.csv)
  3. Per-face indirect flux Q_ind vs a REFERENCE AT THE SAME N (the exact dense
     matrix, or a much larger ray budget). This separates estimator error from
     discretization error: metric 1 and 2 mix both.

Inputs are bowl_crater_ptm output directories (results.csv, radiosity_eq8.csv,
and the PTM_BOWL_DUMP prefix), or mc_ref_pp stdout for floor-only sweeps.

  accuracy.py bowl <run_dir> --dump <prefix> [--ref-dump <prefix>] [--json out.json]
  accuracy.py floor <mc_ref_pp_stdout...> --dd 0.2 [--mesh bowl.ply] [--json out.json]
"""
import argparse
import csv
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np


def fsky_analytic(dd):
    a = math.atan(0.5 / dd)
    return math.sin(a) ** 2


def last_row(path):
    rows = list(csv.DictReader(open(path)))
    if not rows:
        raise SystemExit(f"FATAL: {path} has no data rows")
    if len(rows) > 1:
        print(f"[accuracy] NOTE: {path} has {len(rows)} rows; using the LAST (one run per dir "
              "is the harness convention)", file=sys.stderr)
    return rows[-1]


def per_face(dump, ref):
    """Per-face Q_ind / T differences vs a reference dump at the SAME mesh."""
    d = {k: np.load(f"{dump}_{k}.npy") for k in ("T", "qind", "mask", "areas")}
    r = {k: np.load(f"{ref}_{k}.npy") for k in ("T", "qind", "mask", "areas")}
    if len(d["T"]) != len(r["T"]) or not np.array_equal(d["mask"], r["mask"]) \
            or not np.allclose(d["areas"], r["areas"], rtol=1e-12, atol=0):
        raise SystemExit("FATAL: reference dump is a different mesh (face count / mask / areas)")
    out = {}
    for name, sel in (("floor", d["mask"] == 1), ("all", np.ones(len(d["T"]), bool))):
        w = d["areas"][sel]
        dq = d["qind"][sel] - r["qind"][sel]
        dT = d["T"][sel] - r["T"][sel]
        rms_ref = math.sqrt(float((w * r["qind"][sel] ** 2).sum() / w.sum()))
        out[name] = {
            "n_faces": int(sel.sum()),
            "qind_rms_diff_Wm2": math.sqrt(float((w * dq ** 2).sum() / w.sum())),
            "qind_rel_rms": (math.sqrt(float((w * dq ** 2).sum() / w.sum())) / rms_ref
                             if rms_ref > 0 else None),
            "qind_max_abs_diff_Wm2": float(np.abs(dq).max()),
            "qind_mean_bias_Wm2": float((w * dq).sum() / w.sum()),
            "T_rms_diff_K": math.sqrt(float((w * dT ** 2).sum() / w.sum())),
            "T_max_abs_diff_K": float(np.abs(dT).max()),
        }
    return out


def bowl(a):
    run = Path(a.run_dir)
    res = last_row(run / "results.csv")
    m = {"kind": "bowl", "run_dir": str(run), "kernel": res["kernel"],
         "dd": float(res["dd"]), "n_faces": int(res["nf"]), "n_floor": int(res["n_floor"]),
         "fsky_analytic": float(res["analytic"]), "fsky_floor_area_weighted": float(res["mean_aw"]),
         "fsky_mean_bias_pct": float(res["mean_bias_pct"]),
         "fsky_peak_dev_pct": float(res["peak_dev_pct"]),
         "vf_seconds": float(res["vf_seconds"])}
    eq = run / "radiosity_eq8.csv"
    if eq.exists():
        e = last_row(eq)
        m.update(beta_deg=float(e["beta_deg"]), floor_lit_max=float(e["floor_lit_max"]),
                 Tf_ptm_K=float(e["Tf_ptm"]), Tf_eq8_K=float(e["Tf_eq8"]),
                 Tf_dT_pct=float(e["dT_pct"]), flux_dflux_pct=float(e["dflux_pct"]),
                 picard_iters=int(e["fp_iters"]), picard_seconds=float(e["picard_s"]))
        if m["floor_lit_max"] > 0:
            print("[accuracy] WARNING: floor not fully shadowed (floor_lit_max > 0); the Eq. 8 "
                  "comparison is contaminated by direct sun", file=sys.stderr)
    if a.dump and a.ref_dump:
        m["vs_reference"] = {"reference": a.ref_dump, **per_face(a.dump, a.ref_dump)}
    return m


def floor(a):
    """mc_ref_pp stdout rows '<face> SumF F_sky F_warm Qir' for the floor faces.
    F_sky here is the referee's own (rays to space); 1 - SumF also counts
    back-face/ring-blocked rays as sky. Both are reported."""
    fa = fsky_analytic(a.dd)
    sumf, fsky = [], []
    for p in a.stdout:
        for path in glob.glob(p):
            for ln in open(path):
                if ln.startswith("#") or not ln.strip():
                    continue
                t = ln.split()
                sumf.append(float(t[1]))
                fsky.append(float(t[2]))
    if not sumf:
        raise SystemExit("FATAL: no mc_ref_pp rows found")
    s = 1.0 - np.array(sumf)
    out = {"kind": "floor_sweep", "dd": a.dd, "fsky_analytic": fa, "n_floor": len(sumf)}
    w = None
    if a.mesh:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ptm12_diag"))
        from assemble_vf_haworth import face_areas
        order = [int(ln.split()[0]) for p in a.stdout for path in glob.glob(p)
                 for ln in open(path) if ln.strip() and not ln.startswith("#")]
        w = face_areas(a.mesh)[0][np.array(order)]
    for name, v in (("one_minus_sumf", s), ("referee_fsky", np.array(fsky))):
        mean = float((w * v).sum() / w.sum()) if w is not None else float(v.mean())
        out[name] = {"mean": mean, "area_weighted": w is not None,
                     "mean_bias_pct": 100 * (mean - fa) / fa,
                     "peak_dev_pct": 100 * float(np.abs(v - fa).max()) / fa}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("bowl")
    b.add_argument("run_dir")
    b.add_argument("--dump", default=None)
    b.add_argument("--ref-dump", default=None)
    b.add_argument("--json", default=None)
    f = sp.add_parser("floor")
    f.add_argument("stdout", nargs="+")
    f.add_argument("--dd", type=float, required=True)
    f.add_argument("--mesh", default=None, help="the swept PLY, for area weights")
    f.add_argument("--json", default=None)
    a = ap.parse_args()
    m = bowl(a) if a.cmd == "bowl" else floor(a)
    txt = json.dumps(m, indent=2, sort_keys=True)
    if a.json:
        Path(a.json).write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
