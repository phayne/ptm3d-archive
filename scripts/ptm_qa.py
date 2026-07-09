#!/usr/bin/env python3
"""ptm_qa.py — machine-readable correctness gates for PTM3D outputs.

Why this exists
---------------
The crew's correctness rules were written down, repeatedly, and skimmed past. On
2026-07-09 a runner agent was asked three times, with sharpened wording and the checks
hoisted to the top of its brief, what it must verify before reporting a "noisy IR"
product. All three times it missed the two decisive checks. A boolean fact must be
**computed and cited**, never remembered.

So: every check here emits JSON and a non-zero exit status on failure. Downstream skills
(`ptm-analyze`, `ptm-validate`) refuse to report a product that has no passing artifact
from this tool. The rule binds because the violation is detectable by someone other than
the agent that broke it.

Thresholds are DERIVED FROM REAL FIXTURES, not invented — see `--selftest`.

Subcommands
-----------
  thermal   <run_dir>                 statistics/*.bin: coherence, inversions, ranges
  derived   --parent P --derived D    a derived product must differ from its parent
  signal    <fits|dir>                on-body field is non-zero; degeneracy metrics
  manifest  <out_dir> --binary B      record what produced an output directory
  selftest                            run every gate against the archived fixtures

Exit status: 0 = all checks PASS (warnings allowed), 1 = at least one FAIL, 2 = usage.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np

# --- thresholds, each traceable to a measurement on an archived fixture -------------
# never-lit faces receive zero direct solar; their Tmin must be nearly uniform.
# Measured on 623 Chimaera (18,304 faces):
#     broken binary 5bfbdbe -> std = 18.00 K      (scrambled shadow-face seeds)
#     fixed  binary 79f091e -> std =  0.96 K
# 3.0 K sits ~3x above the good run and ~6x below the broken one.
COHERENCE_MAX_K = 3.0
CMB_K = 2.7          # below this a temperature is not physical; treat as uncomputed
TMAX_K = 500.0
INVALID_FRAC_FAIL = 1e-3   # 1 invalid face in 18,304 is a warning, not a failure
INVERSION_TOL_K = 0.5


class Report:
    def __init__(self, subject: str, kind: str):
        self.d = {"tool": "ptm_qa", "kind": kind, "subject": subject,
                  "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  "checks": []}

    def add(self, name, status, measured=None, threshold=None, note=""):
        self.d["checks"].append({"check": name, "status": status, "measured": measured,
                                 "threshold": threshold, "note": note})

    @property
    def failed(self):
        return [c for c in self.d["checks"] if c["status"] == "FAIL"]

    def finish(self, out_path):
        self.d["verdict"] = "FAIL" if self.failed else "PASS"
        self.d["n_fail"] = len(self.failed)
        if out_path:
            with open(out_path, "w") as fh:
                json.dump(self.d, fh, indent=2)
            self.d["written_to"] = out_path
        for c in self.d["checks"]:
            mark = {"PASS": "  ok  ", "FAIL": " FAIL ", "WARN": " warn "}[c["status"]]
            m = "" if c["measured"] is None else f"  measured={c['measured']}"
            t = "" if c["threshold"] is None else f"  threshold={c['threshold']}"
            print(f"[{mark}] {c['check']}{m}{t}" + (f"\n         {c['note']}" if c["note"] else ""))
        print(f"\nverdict: {self.d['verdict']}" + (f"   -> {out_path}" if out_path else ""))
        return 1 if self.failed else 0


# ------------------------------------------------------------------ helpers
def mesh_face_count(run_dir):
    p = os.path.join(run_dir, "mesh.ply")
    if not os.path.exists(p):
        return None
    with open(p, "rb") as fh:
        for line in fh:
            m = re.match(rb"element face (\d+)", line)
            if m:
                return int(m.group(1))
            if line.strip() == b"end_header":
                break
    return None


def stat(run_dir, name):
    """Read a statistics binary.

    FORMAT (undocumented until 2026-07-09, see BUGS.md PTM-06):
        [int32 n_faces][float32 x n_faces]
    The naive `np.fromfile(p, dtype="<f4")` recipe -- which is what `ptm-analyze`'s
    SKILL.md prescribed -- reads that 4-byte header as a *temperature* and shifts every
    face by one index. Confirmed on 5/5 runs across both the generic-body and planetary
    paths (Chimaera, Justitia, Ousha, A3GT, Moon3D). The header value equals the mesh
    face count exactly, which is how the format was identified.
    """
    p = os.path.join(run_dir, "statistics", name)
    if not os.path.exists(p):
        return None
    n = int(np.fromfile(p, dtype="<i4", count=1)[0])
    body = np.fromfile(p, dtype="<f4")[1:]
    if n != len(body):
        raise ValueError(f"{p}: header says {n} faces but body holds {len(body)} floats")
    return body


def stat_header(run_dir, name):
    p = os.path.join(run_dir, "statistics", name)
    return int(np.fromfile(p, dtype="<i4", count=1)[0]) if os.path.exists(p) else None


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _fits_frames(path):
    if os.path.isdir(path):
        return sorted(glob.glob(os.path.join(path, "*.fits")))
    return [path]


def _read(fp, ext):
    from astropy.io import fits
    with fits.open(fp) as h:
        names = [x.name for x in h]
        if ext not in names:
            raise KeyError(f"{fp}: no HDU {ext!r} (have {names})")
        data = np.asarray(h[ext].data, dtype=float)
        face = np.asarray(h["FACE_IDX"].data) if "FACE_IDX" in names else None
    return data, face


# ------------------------------------------------------------------ thermal
def cmd_thermal(args):
    run = args.run_dir
    r = Report(run, "thermal-statistics")

    n_mesh = mesh_face_count(run)
    tmin, ds = stat(run, "temperature_min.bin"), stat(run, "direct_solar_max.bin")
    if tmin is None or ds is None:
        r.add("statistics present", "FAIL", note="temperature_min.bin / direct_solar_max.bin missing")
        return r.finish(args.out or os.path.join(run, "qa_thermal.json"))

    n_hdr = stat_header(run, "temperature_min.bin")
    r.add("statistics header face-count matches mesh.ply",
          "PASS" if (n_mesh is None or n_hdr == n_mesh == len(tmin)) else "FAIL",
          measured=f"header={n_hdr}, body={len(tmin)}", threshold=n_mesh,
          note="" if n_mesh else "no mesh.ply; count unverified")

    nan = int((~np.isfinite(tmin)).sum())
    r.add("no NaN/inf in temperature_min", "PASS" if nan == 0 else "FAIL", measured=nan, threshold=0)

    invalid = ~np.isfinite(tmin) | (tmin <= CMB_K)
    frac = float(invalid.sum()) / len(tmin)
    r.add("uncomputed faces (Tmin <= 2.7 K)",
          "PASS" if frac == 0 else ("WARN" if frac < INVALID_FRAC_FAIL else "FAIL"),
          measured=f"{int(invalid.sum())} ({frac:.2e})", threshold=f"< {INVALID_FRAC_FAIL:.0e}",
          note="excluded from the checks below" if invalid.any() else "")

    hot = int((tmin[~invalid] > TMAX_K).sum())
    r.add("temperatures below the 500 K clamp", "PASS" if hot == 0 else "FAIL", measured=hot, threshold=0)

    v = ~invalid
    tv, dv = tmin[v], ds[v]
    never, lit = dv <= 0, dv > 0
    if never.sum() < 8:
        r.add("never-lit coherence", "WARN", note=f"only {int(never.sum())} never-lit faces; not evaluated")
    else:
        std = float(tv[never].std())
        # THE regression detector: a scrambled shadow-face initial condition freezes
        # never-lit faces at unrelated temperatures. Broken=18.00 K, fixed=0.96 K.
        r.add("never-lit face coherence (std of Tmin)",
              "PASS" if std <= args.coherence_max_k else "FAIL",
              measured=f"{std:.2f} K", threshold=f"<= {args.coherence_max_k} K",
              note="high scatter means faces that never see the Sun disagree -- "
                   "an initial-condition or forcing defect, not physics")

        floor = float(tv[never].min())
        viol = int((tv[lit] < floor - INVERSION_TOL_K).sum())
        # PRECONDITION, discovered by running this gate on Moon3D: `direct_solar_max == 0`
        # means "not lit during THIS WINDOW", not "permanently shadowed". On a short window
        # (Moon3D radoff: 67% of the globe is nightside) those faces are still warm from
        # earlier insolation -- 64.8 K -- while genuinely-lit polar slopes reach 24 K. Five
        # "inversions" there are correct physics, not a defect. The invariant only holds when
        # the window covers >= 1 full rotation, so it is a WARN unless you assert that.
        strict = getattr(args, "strict_inversion", False)
        status = "PASS" if viol == 0 else ("FAIL" if strict else "WARN")
        r.add("no forbidden inversion (lit face colder than the shadow floor)",
              status, measured=viol, threshold=0,
              note=f"shadow floor = {floor:.2f} K. "
                   + ("" if viol == 0 else
                      "Only a real violation if the window spans >= 1 full rotation, so that "
                      "'never lit' means permanently shadowed rather than merely nightside. "
                      "Pass --strict-inversion to make this a hard failure."))

    return r.finish(args.out or os.path.join(run, "qa_thermal.json"))


# ------------------------------------------------------------------ derived
def cmd_derived(args):
    """A derived product MUST differ from its parent. max|delta| == 0 is a shipped no-op."""
    r = Report(f"{args.derived} vs {args.parent}", "derived-product")
    pf, df = _fits_frames(args.parent), _fits_frames(args.derived)
    if not pf or not df:
        r.add("frames present", "FAIL", note=f"parent={len(pf)} derived={len(df)}")
        return r.finish(args.out)

    r.add("frame count matches parent", "PASS" if len(pf) == len(df) else "FAIL",
          measured=len(df), threshold=len(pf))

    n = min(len(pf), len(df), args.max_frames)
    deltas, identical = [], 0
    for a, b in zip(pf[:n], df[:n]):
        pa, face = _read(a, args.ext)
        da, _ = _read(b, args.ext)
        on = (face >= 0) if face is not None else np.ones_like(pa, bool)
        d = float(np.abs(da[on] - pa[on]).max()) if on.any() else 0.0
        deltas.append(d)
        identical += (d == 0.0)

    worst, med = float(np.max(deltas)), float(np.median(deltas))
    r.add(f"derived {args.ext} differs from parent",
          "PASS" if identical == 0 else "FAIL",
          measured=f"{identical}/{n} frames identical, median max|delta|={med:.4g}",
          threshold="0 identical frames",
          note="max|delta| == 0 means the transform was a NO-OP. ema_camera_noise.py "
               "perturbed the wrong HDU and shipped clean frames as 'noisy' for months.")
    r.d["max_abs_delta"] = worst
    return r.finish(args.out)


# ------------------------------------------------------------------ signal
def cmd_signal(args):
    r = Report(args.path, "signal")
    frames = _fits_frames(args.path)[: args.max_frames]
    if not frames:
        r.add("frames present", "FAIL", note="no .fits found")
        return r.finish(args.out)

    tot_on, tot_nz, distinct, vmax = 0, 0, 0, 0.0
    for fp in frames:
        d, face = _read(fp, args.ext)
        on = (face >= 0) if face is not None else np.ones_like(d, bool)
        tot_on += int(on.sum())
        tot_nz += int((d[on] != 0).sum())
        distinct = max(distinct, int(np.unique(d[on]).size))
        vmax = max(vmax, float(np.abs(d[on]).max()) if on.any() else 0.0)

    r.add(f"on-body {args.ext} is not all-zero",
          "PASS" if tot_nz > 0 else "FAIL",
          measured=f"{tot_nz}/{tot_on} non-zero px, max={vmax:.4g}", threshold="> 0 non-zero",
          note="an all-zero on-body field means this product was never rendered. Four EMA "
               "asteroids shipped 'noisy IR' that was Gaussian noise on nothing.")

    r.add(f"{args.ext} is not degenerate (distinct values)",
          "PASS" if distinct >= args.min_distinct else "WARN",
          measured=distinct, threshold=f">= {args.min_distinct}",
          note="a near-constant field is usually a fill value, not data")

    if args.degeneracy_ext:
        d, face = _read(frames[0], args.degeneracy_ext)
        if face is None:
            r.add("per-facet degeneracy", "WARN", note="no FACE_IDX")
        else:
            ids, cnt = np.unique(face[face >= 0], return_counts=True)
            big = ids[cnt >= max(8, int(np.percentile(cnt, 95)))]
            ratios = []
            for b in big[:200]:
                m = face == b
                ratios.append(np.unique(d[m]).size / m.sum())
            med = float(np.median(ratios)) if ratios else 1.0
            r.add(f"{args.degeneracy_ext} varies per pixel, not per facet",
                  "PASS" if med >= args.quantization_ratio else "WARN",
                  measured=f"median distinct/px = {med:.3f} over {len(ratios)} large facets",
                  threshold=f">= {args.quantization_ratio}",
                  note="a low ratio means the field is FACET-QUANTIZED, not per-pixel. "
                       "Camera LATITUDE/LONGITUDE were facet-centroid values; six rounds of "
                       "renderer rework chased that before anyone histogrammed the input.")
    return r.finish(args.out)


# ------------------------------------------------------------------ manifest
def cmd_manifest(args):
    out = os.path.join(args.out_dir, "manifest.json")
    git = "/Library/Developer/CommandLineTools/usr/bin/git"
    repo = os.environ.get("PTM", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    def sh(cmd):
        try:
            return subprocess.run(cmd, cwd=repo, capture_output=True, text=True,
                                  timeout=120).stdout.strip() or None
        except Exception:
            return None

    man = {"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "commit": sh([git, "rev-parse", "HEAD"]),
           "branch": sh([git, "rev-parse", "--abbrev-ref", "HEAD"]),
           "dirty": bool(sh([git, "status", "--porcelain"])),
           "cmdline": args.cmdline,
           "mesh_face_count": mesh_face_count(args.out_dir)}

    if args.binary and os.path.exists(args.binary):
        man["binary"] = {"path": args.binary, "md5": md5(args.binary)}
        # A thermal-only fix moves libthermal.dylib while the exe md5 is UNCHANGED, because the
        # exe links it dynamically. Reporting only the exe md5 makes a "no change" claim false.
        # `otool -L` would be authoritative, but it is broken on this machine (the Xcode shim,
        # same class as the /usr/bin/git shim), so walk the build tree instead.
        build = os.path.dirname(os.path.abspath(args.binary))
        while build != "/" and not os.path.basename(build).startswith("build"):
            build = os.path.dirname(build)
        if os.path.basename(build).startswith("build"):
            man["build_dir"] = build
            libs = sorted(glob.glob(os.path.join(build, "**", "*.dylib"), recursive=True))[:24]
            man["libs"] = {os.path.relpath(l, build): md5(l) for l in libs}
    for k in ("expected_cells", "expected_timesteps", "expected_frames"):
        v = getattr(args, k)
        if v is not None:
            man[k] = v

    os.makedirs(args.out_dir, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(man, fh, indent=2)
    print(json.dumps(man, indent=2))
    print(f"\nwrote {out}")
    if man["dirty"]:
        print("WARNING: the working tree is dirty -- this output is not reproducible "
              "from `commit` alone.", file=sys.stderr)
    return 0


# ------------------------------------------------------------------ selftest
def cmd_selftest(args):
    E = "/Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/EMA/science"
    C = f"{E}/thermal/623chimaera"
    cases = [
        ("thermal: archived BROKEN field (5bfbdbe) must FAIL",
         ["thermal", f"{C}/flyby_preTerminatorFix_20260704_142231", "--out", "/tmp/_qa_bad.json"], 1),
        ("thermal: current GOOD field (79f091e) must PASS",
         ["thermal", f"{C}/flyby", "--out", "/tmp/_qa_good.json"], 0),
        ("signal: Rockox clean IR (all-zero on-body) must FAIL",
         ["signal", f"{E}/camera/13294rockox/flyby/visible/anim", "--ext", "TEMP_K",
          "--max-frames", "2", "--out", "/tmp/_qa_null.json"], 1),
        ("signal: Chimaera IR (real) must PASS",
         ["signal", f"{E}/camera/623chimaera/flyby/anim", "--ext", "TEMP_K",
          "--max-frames", "2", "--out", "/tmp/_qa_real.json"], 0),
        # WHY BOTH GATES EXIST: Rockox's 'noisy' IR *does* differ from its clean parent
        # (noise on a zero field, Planck-inverted to ~133 K), so the no-op gate passes.
        # Only the signal gate catches that there was never anything to noise.
        ("derived: Rockox noisy-vs-clean PASSES the no-op gate -- which is why `signal` exists",
         ["derived", "--parent", f"{E}/camera/13294rockox/flyby/visible/anim",
          "--derived", f"{E}/camera/13294rockox/flyby/visible/anim/noisy",
          "--ext", "TEMP_K", "--max-frames", "2", "--out", "/tmp/_qa_rocknoise.json"], 0),
        ("derived: Chimaera noisy vs clean must PASS (genuinely noised)",
         ["derived", "--parent", f"{E}/camera/623chimaera/flyby/anim",
          "--derived", f"{E}/camera/623chimaera/flyby/anim/noisy",
          "--ext", "TEMP_K", "--max-frames", "3", "--out", "/tmp/_qa_noise.json"], 0),
        # Cross-body generality: the coherence gate must NOT cry wolf on healthy runs from
        # other bodies, meshes, and code paths. Moon3D also proves the inversion precondition.
        ("cross-body: A3GT lunar 1.8M-face production run must PASS",
         ["thermal", os.path.expanduser("~/Research/moon/artemis/a3gt/"
                                        "sites_v2_3km_corering_postfix/A03MP026"),
          "--out", "/tmp/_qa_a3gt.json"], 0),
        ("cross-body: Moon3D global 509k must PASS (5 'inversions' are a short-window artifact)",
         ["thermal", os.path.expanduser("~/dev/heat3d/moon3d_verify/radoff"),
          "--out", "/tmp/_qa_moon3d.json"], 0),
        ("derived: clean vs ITSELF must FAIL (the no-op we shipped for months)",
         ["derived", "--parent", f"{E}/camera/623chimaera/flyby/anim",
          "--derived", f"{E}/camera/623chimaera/flyby/anim",
          "--ext", "TEMP_K", "--max-frames", "3", "--out", "/tmp/_qa_noop.json"], 1),
    ]
    bad = 0
    for title, argv, want in cases:
        print("\n" + "=" * 78 + f"\n{title}\n  expect exit {want}\n" + "-" * 78)
        rc = main(argv)
        ok = rc == want
        bad += not ok
        print(f"  --> exit {rc}  [{'OK' if ok else 'SELFTEST FAILURE'}]")
    print("\n" + "=" * 78)
    print(f"selftest: {len(cases)-bad}/{len(cases)} gates behave as specified")
    return 1 if bad else 0


# ------------------------------------------------------------------ cli
def main(argv=None):
    p = argparse.ArgumentParser(prog="ptm_qa.py", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("thermal"); t.add_argument("run_dir")
    t.add_argument("--coherence-max-k", type=float, default=COHERENCE_MAX_K)
    t.add_argument("--strict-inversion", action="store_true",
                   help="treat a forbidden inversion as FAIL (only valid when the "
                        "output window spans >= 1 full rotation)")
    t.add_argument("--out"); t.set_defaults(fn=cmd_thermal)

    d = sub.add_parser("derived")
    d.add_argument("--parent", required=True); d.add_argument("--derived", required=True)
    d.add_argument("--ext", default="TEMP_K"); d.add_argument("--max-frames", type=int, default=8)
    d.add_argument("--out"); d.set_defaults(fn=cmd_derived)

    s = sub.add_parser("signal"); s.add_argument("path")
    s.add_argument("--ext", default="TEMP_K"); s.add_argument("--max-frames", type=int, default=4)
    s.add_argument("--min-distinct", type=int, default=16)
    s.add_argument("--degeneracy-ext", default=None)
    s.add_argument("--quantization-ratio", type=float, default=0.5)
    s.add_argument("--out"); s.set_defaults(fn=cmd_signal)

    m = sub.add_parser("manifest"); m.add_argument("out_dir")
    m.add_argument("--binary"); m.add_argument("--cmdline")
    m.add_argument("--expected-cells", type=int); m.add_argument("--expected-timesteps", type=int)
    m.add_argument("--expected-frames", type=int); m.set_defaults(fn=cmd_manifest)

    st = sub.add_parser("selftest"); st.set_defaults(fn=cmd_selftest)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
