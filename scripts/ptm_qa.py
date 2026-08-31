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

try:
    import numpy as np
except ModuleNotFoundError:  # actionable instead of a bare traceback
    sys.stderr.write(
        "ptm_qa.py needs NumPy, which is not installed in this interpreter\n"
        f"  ({sys.executable}).\n"
        "Install the repository's Python environment:\n"
        "    python3 -m pip install -r requirements.txt\n"
    )
    raise SystemExit(2)

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

    FORMAT (see docs/KNOWN_LIMITATIONS.md, PTM-06):
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


# ------------------------------------------------------------------ liveness (P7)
def cmd_liveness(args):
    """A job is stalled ONLY if its own heartbeat has not advanced.

    Never infer a stall from `ps` + output-directory mtime. On 2026-07-06 a render writing
    frames to a temp dir looked byte-identical to a hang -- no worker process at the instant
    of the snapshot, no new files under the output dir -- and was declared stalled. It was
    fine. If a job writes no heartbeat, the honest answer is UNKNOWN, not a guess.
    """
    p = os.path.join(args.dir, "progress.json")
    if not os.path.exists(p):
        print(json.dumps({"state": "UNKNOWN", "reason": "no progress.json; this job writes no "
                                                        "heartbeat. Liveness cannot be determined -- "
                                                        "say so, do not guess from `ps`."}, indent=2))
        return 3
    hb = json.load(open(p))
    now = datetime.now(timezone.utc)
    last = datetime.fromisoformat(hb["last_advance_utc"].replace("Z", "+00:00"))
    age_min = (now - last).total_seconds() / 60.0

    alive = None
    if hb.get("pid"):
        try:
            os.kill(int(hb["pid"]), 0)
            alive = True
        except ProcessLookupError:
            alive = False
        except PermissionError:
            alive = True

    # A cleanly finished job has no pid and no advancing heartbeat -- exactly like a dead one.
    # Only the job itself can tell them apart, so it records `finished` on clean exit.
    if hb.get("failed"):
        state, rc = "FAILED", 1
    elif hb.get("finished"):
        state, rc = "DONE", 0
    elif alive is False:
        state, rc = "SIM_DOWN", 1
    elif age_min > args.stall_min:
        state, rc = "WEDGE_SUSPECT", 1
    else:
        state, rc = "RUNNING", 0

    out = {"state": state, "unit": hb.get("unit"), "unit_total": hb.get("unit_total"),
           "unit_label": hb.get("unit_label"), "pid": hb.get("pid"), "pid_alive": alive,
           "minutes_since_advance": round(age_min, 2), "stall_min": args.stall_min}
    if hb.get("unit") and hb.get("unit_total"):
        out["percent"] = round(100.0 * hb["unit"] / hb["unit_total"], 1)
    print(json.dumps(out, indent=2))
    return rc


# ------------------------------------------------------------------ figure provenance (P11)
def _sha256(path, limit=None):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def cmd_figure(args):
    """Record (or verify) what produced a figure.

    `fit_curves.png` was reviewed and approved, then became unregenerable: its generator
    imported a /tmp copy of ema_ti_fit.py that no longer exists. A caption says what a
    figure means; provenance says how to make it again.
    """
    prov = args.figure + ".provenance.json"
    if args.check:
        if not os.path.exists(prov):
            print(f"FAIL: {args.figure} has no provenance. A reported figure must record its "
                  f"generating script, that script's hash, and its input hashes.")
            return 1
        d = json.load(open(prov))
        stale = []
        s = d.get("script")
        if s and os.path.exists(s["path"]) and md5(s["path"]) != s["md5"]:
            stale.append(f"script changed: {s['path']}")
        for i in d.get("inputs", []):
            if os.path.exists(i["path"]) and md5(i["path"]) != i["md5"]:
                stale.append(f"input changed: {i['path']}")
        for m in stale:
            print(f"STALE: {m}")
        print("PASS: provenance present and current" if not stale
              else "FAIL: figure is stale w.r.t. its recorded provenance")
        return 1 if stale else 0

    if not os.path.exists(args.figure):
        print(f"FAIL: no such figure {args.figure}", file=sys.stderr)
        return 1
    git = "/Library/Developer/CommandLineTools/usr/bin/git"
    repo = os.environ.get("PTM", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    def sh(c):
        try:
            return subprocess.run(c, cwd=repo, capture_output=True, text=True,
                                  timeout=120).stdout.strip() or None
        except Exception:
            return None

    d = {"figure": os.path.abspath(args.figure), "sha256": _sha256(args.figure),
         "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
         "commit": sh([git, "rev-parse", "HEAD"]),
         "dirty": bool(sh([git, "status", "--porcelain"]))}
    if args.script:
        d["script"] = {"path": os.path.abspath(args.script), "md5": md5(args.script),
                       "tracked": bool(sh([git, "ls-files", args.script]))}
        if not d["script"]["tracked"]:
            print("WARNING: the generating script is UNTRACKED. It is one `rm` from gone, and "
                  "this figure is then unregenerable.", file=sys.stderr)
    d["inputs"] = [{"path": os.path.abspath(i), "md5": md5(i), "bytes": os.path.getsize(i)}
                   for i in (args.input or []) if os.path.exists(i)]
    with open(prov, "w") as fh:
        json.dump(d, fh, indent=2)
    print(json.dumps(d, indent=2))
    print(f"\nwrote {prov}")
    return 0


# ------------------------------------------------------------------ precommit (P11)
GIT = "/Library/Developer/CommandLineTools/usr/bin/git"


def _git_env():
    """The environment every git call here runs in, with `GIT_*` removed.

    Measured: with `GIT_DIR`/`GIT_WORK_TREE` exported -- which is the ordinary
    state inside a hook, `git rebase -x` or `git bisect run` --
    `rev-parse --show-toplevel` returns the INHERITED repository no matter what
    cwd it is given. The repo cross-check below then compares that answer with
    itself, agrees, and diffs a tree nobody asked about. Stripping the
    variables makes cwd mean what this tool assumes it means.
    """
    env = dict(os.environ)
    for k in [k for k in env if k.startswith("GIT_")]:
        del env[k]
    return env


def _git(argv, cwd):
    """Run git and RAISE on failure, instead of returning an empty answer.

    `_hunks` used to read `.stdout` and ignore the return code, so every way git
    could fail -- not a repository, no HEAD, a bad ref -- came back as "no
    hunks", which `--verify` then reported as a PASS. A tool that cannot tell
    "nothing changed" from "I could not look" is a tool that cannot fail.
    """
    out = subprocess.run([GIT] + argv, cwd=cwd, capture_output=True, text=True,
                         env=_git_env(), timeout=600)
    if out.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(argv)} failed in {cwd} (exit {out.returncode}): "
            + (out.stderr.strip() or "<no stderr>"))
    return out.stdout


def _hunks(argv, repo):
    """Return {file: [hunk_md5, ...]} from a `git diff` invocation, at -U0 for stability."""
    out = _git(argv, repo)
    files, cur, buf = {}, None, []

    def flush():
        if cur and buf:
            files.setdefault(cur, []).append(hashlib.md5("\n".join(buf).encode()).hexdigest())

    for line in out.split("\n"):
        if line.startswith("diff --git "):
            flush(); buf = []
            cur = line.split(" b/")[-1]
        elif line.startswith("@@"):
            flush(); buf = []
        # NOT `line[:1] in "+-"`: for an empty line that is `"" in "+-"`, which is True in
        # Python, so blank lines got folded into the hunk body. The same hunk then hashed
        # differently depending on whether a trailing blank line followed it -- which is
        # exactly the difference between `git diff -- <one file>` and a full-repo diff.
        elif line[:1] in ("+", "-") and not line.startswith(("+++", "---")):
            buf.append(line)
    flush()
    return files


def _toplevel(path):
    """`git rev-parse --show-toplevel` for `path`, or None if it is not a repo.

    Two measured facts about the answer, both of which decide how it is used:

    * git replies in the FILESYSTEM's own spelling. Asked about
      `/Users/paha3326/projects/ptm-x` on this case-insensitive volume it says
      `/Users/paha3326/Projects/ptm-x`, and it resolves symlinks (`/tmp` ->
      `/private/tmp`) on the way. `os.path.abspath` and `os.path.realpath` do
      NEITHER on macOS -- they hand back the case they were given -- so the raw
      `$PTM` string and this answer routinely differ while naming one directory.
    * because BOTH sides of the comparison in `_resolve_repo` come from here,
      they arrive already canonical, and `==` would agree with `samefile` on
      every case built so far. `samefile` is kept because it is the CORRECT
      question ("is this the same directory") and cannot be wrong where `==`
      can; the selftest's case arm is therefore a do-no-harm arm, not a proof
      that `samefile` is load-bearing, and it says so.
    """
    if not path or not os.path.isdir(path):
        return None
    try:
        out = subprocess.run([GIT, "rev-parse", "--show-toplevel"], cwd=path,
                             capture_output=True, text=True,
                             env=_git_env(), timeout=60)
    except Exception:                                        # noqa: BLE001
        return None
    return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else None


def _resolve_repo():
    """`(repo, note)` -- the checkout precommit will diff, or `(None, refusal)`.

    PTM-287: the repo was taken from `$PTM`, or from the script's own parent
    directory, and NEVER cross-checked against where the command was actually
    run. From a worktree that silently snapshotted and diffed a DIFFERENT
    checkout and printed a PASS about a tree it never looked at -- three
    independent hits in one week. The fix is not to change which source wins;
    it is to REFUSE when the two disagree, and to say which is which.
    """
    env = os.environ.get("PTM")
    asked = env or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    source = "$PTM" if env else "the script's own parent directory"
    asked_top = _toplevel(asked)
    if asked_top is None:
        return None, (f"REFUSED: {asked!r} (from {source}) is not a git "
                      f"checkout, so there is nothing to verify. Point $PTM at "
                      f"the working tree you mean.")
    here_top = _toplevel(os.getcwd())
    if here_top is None:
        # Not run from inside any checkout. Nothing to cross-check against, so
        # this is disclosed rather than refused -- but it is never silent.
        return asked_top, (f"NOTE: the current directory {os.getcwd()!r} is not "
                           f"inside a git checkout, so the repository below "
                           f"could not be cross-checked. Verifying {asked_top} "
                           f"(from {source}).")
    try:
        same = os.path.samefile(asked_top, here_top)
    except OSError as exc:                                   # pragma: no cover
        return None, (f"REFUSED: {asked_top} and {here_top} could not be "
                      f"compared ({exc}). Not proceeding on a repository this "
                      f"tool cannot identify.")
    if not same:
        return None, (
            f"REFUSED: this command would verify a DIFFERENT checkout than the "
            f"one you are in.\n"
            f"  {source} resolves to : {asked_top}\n"
            f"  the current directory: {here_top}\n"
            f"A diff of the first tree says nothing about the second, and that "
            f"silence is what shipped false PASSes (PTM-287). Re-run with "
            f"PTM={here_top} , or cd into the tree you meant.")
    return asked_top, ""


def _git_path(repo, rel):
    """Resolve a path inside the REAL git dir, worktree-safe.

    In a linked worktree, `<repo>/.git` is a gitfile (a text file), not a directory,
    so `os.path.join(repo, ".git", rel)` writes into a "Not a directory" error. Ask git
    for the actual git dir instead: `git rev-parse --git-path <rel>` returns the correct
    location for both a normal checkout (`.git/<rel>`) and a worktree
    (`<main>/.git/worktrees/<name>/<rel>`). Falls back to the naive join only if git is
    unavailable (not a repo at all), preserving the prior behavior there.
    """
    try:
        # `env=_git_env()` for the same reason as everywhere else here: under an
        # exported GIT_DIR this would site the snapshot inside the INHERITED
        # repository's git dir rather than this checkout's.
        out = subprocess.run([GIT, "rev-parse", "--git-path", rel], cwd=repo,
                             capture_output=True, text=True,
                             env=_git_env(), timeout=60)
        p = out.stdout.strip()
        if out.returncode == 0 and p:
            return p if os.path.isabs(p) else os.path.join(repo, p)
    except Exception:
        pass
    return os.path.join(repo, ".git", rel)


def cmd_precommit(args):
    """Catch pre-existing edits riding along in your commit.

    On 2026-07-09 a commit swept in three uncommitted working-tree edits (use_gpu True->False,
    chi 0.0->2.7 x2) under a message asserting "no behavior change to any simulation". The
    edits were the user's, deliberately held back. `git add <explicit paths>` is not enough:
    it stages everything already dirty in those paths.

    Usage:  precommit --snapshot     (at the START of a session, before you edit anything)
            precommit --verify       (after `git add`, before `git commit`)

    Two ways this could not fail, both closed here and both exercised by
    `precommit-selftest`:

    * **An empty index passed.** With nothing staged, `git diff --cached`
      returns nothing, no hunk could be foreign, and the tool printed
      `"staged_files": 0` and PASS. That is a verdict about a commit that does
      not exist -- and it is indistinguishable, in the output, from a real one.
    * **The wrong checkout passed.** See `_resolve_repo`.

    Exit status: 0 PASS, 1 the gate ran and the answer is no (foreign hunks, no
    snapshot, an empty index), 2 the gate could not run at all (the repository
    could not be identified, or git failed).
    """
    repo, note = _resolve_repo()
    if repo is None:
        print(note, file=sys.stderr)
        return 2
    if note:
        print(note, file=sys.stderr)
    snap = args.snapshot_file or _git_path(repo, "ptm_precommit_snapshot.json")

    def verdict(**kw):
        """One JSON shape on EVERY path, so no consumer branches on a KeyError."""
        d = {"repo": repo, "snapshot": snap, "staged_files": 0,
             "staged_paths": [], "staged_path_count": 0,
             "your_hunks": 0, "pre_existing_hunks_staged": {}}
        d.update(kw)
        print(json.dumps(d, indent=2))

    try:
        if args.snapshot:
            pre = _hunks(["diff", "HEAD", "-U0"], repo)
            with open(snap, "w") as fh:
                json.dump({"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                           "repo": repo, "pre_existing": pre}, fh, indent=2)
            n = sum(len(v) for v in pre.values())
            print(f"snapshot: {n} pre-existing hunks across {len(pre)} files "
                  f"in {repo} -> {snap}")
            return 0

        if not os.path.exists(snap):
            print("FAIL: no snapshot. Run `precommit --snapshot` before you start editing, so the "
                  "tool can tell your hunks from ones that were already there.", file=sys.stderr)
            return 1
        with open(snap) as fh:
            recorded = json.load(fh)
        pre = recorded["pre_existing"]
        # A snapshot belongs to ONE checkout. Without this, `--snapshot-file`
        # -- the one flag that moves the file out of the per-repo git dir --
        # let a baseline taken in tree A score tree B's staged hunks, which is
        # the wrong-repo failure the resolver above exists to stop, one level
        # down. Older snapshots carry no `repo` key and are accepted as before.
        snap_repo = recorded.get("repo")
        if snap_repo and not (os.path.isdir(snap_repo)
                              and os.path.samefile(snap_repo, repo)):
            print(f"REFUSED: the snapshot {snap} was taken in {snap_repo}, not "
                  f"{repo}. Scoring one tree's staged hunks against another "
                  f"tree's baseline says nothing about either. Take a snapshot "
                  f"in this checkout.", file=sys.stderr)
            return 2
        # The INDEX first, and by name. `_hunks` sees only content hunks, so a
        # stage that is purely a rename or a mode change is invisible to it and
        # would read as an empty one.
        names = [n for n in _git(["diff", "--cached", "--name-only"], repo).split("\n") if n]
        staged = _hunks(["diff", "--cached", "-U0"], repo)
    except (RuntimeError, subprocess.TimeoutExpired) as exc:
        print(f"REFUSED: git could not be consulted: {exc}", file=sys.stderr)
        return 2
    except (ValueError, KeyError, OSError) as exc:
        # ValueError covers json.JSONDecodeError. A snapshot that cannot be
        # read is "the gate could not look", not "your diff is clean".
        print(f"REFUSED: the snapshot {snap} could not be read "
              f"({type(exc).__name__}: {exc}). Take a fresh one.", file=sys.stderr)
        return 2

    if not names:
        verdict()
        print("\nREFUSED: nothing staged -- nothing verified.\n"
              "         `git diff --cached` is empty in this checkout, so there is no\n"
              "         commit here for this gate to have an opinion about. Stage the\n"
              "         files you mean to commit and run this again.", file=sys.stderr)
        return 1

    bad, mine = {}, 0
    for f, hs in staged.items():
        already = set(pre.get(f, []))
        carried = [h for h in hs if h in already]
        mine += len(hs) - len(carried)
        if carried:
            bad[f] = len(carried)

    # `staged_files` keeps its original meaning -- files carrying content hunks
    # -- so a number quoted in an older report is still comparable. The index's
    # own count is the NEW key beside it.
    verdict(staged_files=len(staged), staged_paths=names,
            staged_path_count=len(names), your_hunks=mine,
            pre_existing_hunks_staged=bad)
    if bad:
        print("\nFAIL: the staged diff contains hunks that existed BEFORE this session.\n"
              "      They are not yours. Unstage them (`git restore --staged <file>`), or\n"
              "      disclose them explicitly in the commit message.", file=sys.stderr)
        return 1
    if not staged:
        # Staged, but nothing `git diff -U0` can hash: a pure rename, a mode
        # change, or an empty file. Said out loud rather than counted as a
        # verified diff.
        print(f"\nPASS with a caveat: {len(names)} path(s) are staged but none carries a\n"
              f"      content hunk (a rename, a mode change or an empty file). Nothing\n"
              f"      pre-existing rode along, and nothing was line-by-line verified.")
        return 0
    print("\nPASS: every staged hunk was authored after the snapshot.")
    return 0


# ------------------------------------------------- precommit selftest (hermetic)
def _pc_run(tool, argv, cwd, ptm=None):
    """One `precommit` invocation against `tool`, in a throwaway checkout."""
    env = _git_env()          # GIT_DIR/GIT_WORK_TREE would point every child
    env.pop("PTM", None)      # at the caller's real repository
    if ptm is not None:
        env["PTM"] = ptm
    out = subprocess.run([sys.executable, tool, "precommit"] + argv, cwd=cwd,
                         env=env, capture_output=True, text=True, timeout=600)
    return out.returncode, (out.stdout + out.stderr)


def _pc_git(argv, cwd):
    """git inside a throwaway repo, insulated from the developer's setup.

    Without `env=_git_env()` an exported `GIT_DIR`/`GIT_WORK_TREE` -- the
    ordinary state inside a hook or `git rebase -x` -- makes these `add` and
    `commit` calls operate on the USER'S REAL repository. The two `-c` flags
    remove the global settings that make a scratch commit fail on other
    machines (a signing key that is not present, a hooks path that rejects).
    """
    subprocess.run([GIT, "-c", "commit.gpgsign=false",
                    "-c", "core.hooksPath=/dev/null"] + argv,
                   cwd=cwd, env=_git_env(), check=True,
                   capture_output=True, text=True)


def _pc_repo(root, name):
    """A one-commit git repository at `root/name`."""
    path = os.path.join(root, name)
    os.makedirs(path)
    _pc_git(["init", "-q"], path)
    _pc_git(["config", "user.email", "selftest@example.invalid"], path)
    _pc_git(["config", "user.name", "ptm_qa selftest"], path)
    with open(os.path.join(path, "tracked.txt"), "w") as fh:
        fh.write("original\n")
    _pc_git(["add", "tracked.txt"], path)
    _pc_git(["commit", "-qm", "base"], path)
    return path


def cmd_precommit_selftest(args):
    """Exercise `precommit` against throwaway repositories.

    Hermetic on purpose: no fixture, no network, no PTM3D output. It answers
    the only question that matters about a gate -- can it fail? Point `--tool`
    at another checkout's `ptm_qa.py` to measure that one instead, which is how
    the pre-fix behavior in these cases was recorded rather than asserted.

    Every case asserts an exit code AND a substring of the message. Exit codes
    alone are ambiguous here: 1 is both "foreign hunks" and "no snapshot", so a
    `--snapshot` that quietly failed would satisfy case 1 for the wrong reason.
    The `--snapshot` calls are checked too, for the same reason.
    """
    import shutil
    import tempfile
    tool = os.path.abspath(args.tool or __file__)
    root = tempfile.mkdtemp(prefix="ptm_qa_pc_")
    cases, bad = [], 0

    def snapshot(path):
        rc, out = _pc_run(tool, ["--snapshot"], path, ptm=path)
        if rc != 0:
            raise RuntimeError(f"--snapshot failed in {path} (exit {rc}): {out}")

    try:
        def stage_a_new_hunk(path, text):
            with open(os.path.join(path, "tracked.txt"), "w") as fh:
                fh.write("original\n" + text + "\n")
            _pc_git(["add", "tracked.txt"], path)

        # -- 1. nothing staged: a verdict about a commit that does not exist ---
        # The snapshot is taken in the SAME repo the verify resolves, so that a
        # "no snapshot" short circuit cannot stand in for the answer -- and the
        # expected message is asserted, so it cannot stand in silently either.
        a = _pc_repo(root, "empty_index")
        snapshot(a)
        rc, out = _pc_run(tool, ["--verify"], a, ptm=a)
        cases.append(("empty index must be REFUSED, not passed", rc, 1,
                      "nothing staged", out))

        # -- 2. the wrong checkout: a PASS about a tree it never looked at ----
        # BOTH trees get a snapshot -- which is the real situation, since the
        # main checkout named by $PTM carries one from an earlier session.
        b = _pc_repo(root, "the_tree_you_are_in")
        c = _pc_repo(root, "the_other_tree")
        snapshot(b)
        snapshot(c)
        stage_a_new_hunk(b, "a hunk in the tree you are actually in")
        rc, out = _pc_run(tool, ["--verify"], b, ptm=c)
        cases.append(("a repo that disagrees with the cwd must be REFUSED",
                      rc, 2, "DIFFERENT checkout", out))

        # -- 3. DO-NO-HARM: the same tree, spelled in another case, still passes
        # NOT a proof that `samefile` is required. `git rev-parse
        # --show-toplevel` canonicalizes through the child's own getcwd, so both
        # sides of the comparison arrive in the filesystem's spelling and `==`
        # would agree here too (measured). This arm exists so that the refusal
        # added above cannot start firing on a legitimate run.
        cs = _pc_repo(root, "Case_Spelling")
        other_case = os.path.join(root, "case_spelling")
        if os.path.isdir(other_case) and os.path.samefile(other_case, cs):
            snapshot(cs)
            stage_a_new_hunk(cs, "mine, written after the snapshot")
            rc, out = _pc_run(tool, ["--verify"], cs, ptm=other_case)
            cases.append(("do-no-harm: the same tree spelled in another case "
                          "must PASS", rc, 0, "PASS: every staged hunk", out))
        else:
            print("  [skip] this filesystem is case-SENSITIVE; the case arm "
                  "cannot be built here")

        # -- 4. a genuine verify still passes ---------------------------------
        d = _pc_repo(root, "genuine")
        snapshot(d)
        stage_a_new_hunk(d, "mine, written after the snapshot")
        rc, out = _pc_run(tool, ["--verify"], d, ptm=d)
        cases.append(("a genuine staged edit must PASS", rc, 0,
                      "PASS: every staged hunk", out))

        # -- 5. the tool's actual job: a pre-existing hunk is caught ----------
        e = _pc_repo(root, "foreign_hunk")
        with open(os.path.join(e, "tracked.txt"), "w") as fh:
            fh.write("someone else's held edit\n")
        snapshot(e)                                 # snapshot AFTER their edit
        _pc_git(["add", "tracked.txt"], e)
        rc, out = _pc_run(tool, ["--verify"], e, ptm=e)
        cases.append(("a hunk that predates the snapshot must FAIL", rc, 1,
                      "existed BEFORE this session", out))
    except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
        # A scenario that could not be BUILT is reported as a failed selftest,
        # not as a traceback that says nothing about the five cases.
        print(f"[ FAIL ] the selftest could not build its fixtures: "
              f"{type(exc).__name__}: {exc}")
        bad += 1
    finally:
        shutil.rmtree(root, ignore_errors=True)

    for title, rc, want, needle, out in cases:
        ok = (rc == want) and (needle in out)
        bad += not ok
        print(f"[{'  ok  ' if ok else ' FAIL '}] {title}\n"
              f"         exit {rc} (expected {want}), message "
              f"{'carries' if needle in out else 'MISSING'} {needle!r}")
        if not ok:
            print("         " + "\n         ".join(out.strip().split("\n")[-6:]))
    print(f"\nprecommit-selftest: {len(cases)-bad}/{len(cases)} cases behave as "
          f"specified  [tool: {tool}]")
    return 1 if bad else 0


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

    lv = sub.add_parser("liveness"); lv.add_argument("dir")
    lv.add_argument("--stall-min", type=float, default=60.0); lv.set_defaults(fn=cmd_liveness)

    fg = sub.add_parser("figure"); fg.add_argument("figure")
    fg.add_argument("--script"); fg.add_argument("--input", action="append")
    fg.add_argument("--check", action="store_true"); fg.set_defaults(fn=cmd_figure)

    pc = sub.add_parser("precommit")
    # Exactly one, and REQUIRED. A bare `precommit` used to fall through to
    # verify, so a mistyped flag silently ran the other mode.
    mode = pc.add_mutually_exclusive_group(required=True)
    mode.add_argument("--snapshot", action="store_true")
    mode.add_argument("--verify", action="store_true")
    pc.add_argument("--snapshot-file"); pc.set_defaults(fn=cmd_precommit)

    ps = sub.add_parser("precommit-selftest")
    ps.add_argument("--tool", help="the ptm_qa.py to measure (default: this one)")
    ps.set_defaults(fn=cmd_precommit_selftest)

    st = sub.add_parser("selftest"); st.set_defaults(fn=cmd_selftest)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
