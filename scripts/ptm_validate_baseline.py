#!/usr/bin/env python3
"""Bank and verify PTM3D byte-identity validation baselines.

This is the tooling half of `docs/validation-baselines.md`. The doctrine half
explains *why*; this file is the part that runs.

Two subcommands:

  bank    Walk a set of finished run directories, digest every output file the
          suite compares, and write a single `baseline_manifest.json` that
          records the digests together with the provenance that makes them
          meaningful: engine commit, engine binary md5, every dynamic library
          the engine links (Embree above all), and the md5 of each config that
          produced an arm.

  verify  Re-digest the same run directories and compare against a banked
          manifest. Any digest mismatch is a regression. Any provenance
          mismatch (different config bytes, different Embree, different
          engine) is reported FIRST and loudly, because comparing bytes across
          a toolchain change answers a different question than the one the
          suite is asking.

Neither subcommand runs the simulator. Producing the run directories is the
job of the suite driver (`scripts/run_validation_baseline.sh`), so that this
file stays safe to run on a machine with no GPU and no science data.

Exit codes: 0 = match, 1 = mismatch or missing input, 2 = usage error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

# Output files the suite compares. Everything under `statistics/` (including
# the `_raw/` subdirectory written by output.write_raw_stats) is in scope; the
# skip list holds files whose bytes legitimately vary run to run.
SKIP_NAMES = {
    "provenance.txt",   # records wall-clock time and the invoking command
    "manifest.json",    # generated post hoc by ptm_qa.py
    ".DS_Store",
}


def md5_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def digest_run(run_dir: str, subdir: str = "statistics") -> dict:
    """Digest every comparable file in one run directory.

    Returns {relative_path: {"md5": ..., "bytes": ...}}. Walks recursively so
    `statistics/_raw/` is covered; a suite that silently ignored a whole
    subdirectory would pass while a regression sat inside it.
    """
    root = os.path.join(run_dir, subdir)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"no '{subdir}/' directory under {run_dir}")
    out = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            if name in SKIP_NAMES:
                continue
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, run_dir)
            out[rel] = {"md5": md5_file(full), "bytes": os.path.getsize(full)}
    if not out:
        raise FileNotFoundError(f"{root} contains no comparable files")
    return out


def linked_libraries(binary: str) -> dict:
    """md5 every dynamic library the engine links.

    An engine binary md5 CANNOT cover a dynamic library that moves results.
    Embree is the live example: swapping the SSE2 Homebrew build for a
    multi-ISA one shifts lunar temperatures by ~0.009 K with the engine binary
    completely unchanged. Record the libraries or the provenance is a lie.
    """
    libs = {}
    try:
        out = subprocess.run(["otool", "-L", binary], capture_output=True,
                             text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"_error": f"otool failed: {exc}"}
    for line in out.splitlines()[1:]:
        path = line.strip().split(" (compatibility")[0].strip()
        if not path:
            continue
        real = os.path.realpath(path)
        entry = {"resolved": real}
        if os.path.exists(real):
            entry["md5"] = md5_file(real)
            entry["bytes"] = os.path.getsize(real)
        else:
            # @rpath / framework entries that otool cannot resolve statically.
            entry["md5"] = None
        libs[path] = entry
    return libs


def engine_sha_from_runs(run_dirs) -> str:
    """Read the engine commit from the runs' own `provenance.txt`.

    This is the authoritative source and the repository's HEAD is not: the sha
    is baked in at BUILD time (`PTM_BUILD_GIT_SHA`), so a manifest banked after
    a docs commit would otherwise record the docs commit and claim the arms
    were produced by an engine that never ran them. Disagreement between arms
    is fatal — it means the arms were not all produced by one engine, and a
    manifest that averaged over that would be a fiction.
    """
    seen = {}
    for run_dir in run_dirs:
        prov = os.path.join(run_dir, "provenance.txt")
        if not os.path.exists(prov):
            continue
        for line in open(prov, errors="replace"):
            if line.startswith("engine_git_sha="):
                seen.setdefault(line.split("=", 1)[1].strip(), []).append(run_dir)
    if not seen:
        return ""
    if len(seen) > 1:
        raise ValueError(
            "the arms were not all produced by one engine -- provenance.txt "
            f"reports {', '.join(f'{k} ({len(v)} arm(s))' for k, v in seen.items())}. "
            "Re-run them on a single binary; a mixed manifest blesses nothing.")
    return next(iter(seen))


def run_threads(run_dir: str) -> dict:
    """The thread count a run actually executed at, from its own provenance.

    Statistics are thread-count dependent: several OpenMP reductions in the
    solver and in the statistics accumulators sum in thread-partition order, so
    the same binary on the same config at 8 and at 48 threads writes different
    bytes. A manifest that does not record this compares two runs that were
    never comparable, and the byte verdict below would be read as a physics
    regression.
    """
    prov = os.path.join(run_dir, "provenance.txt")
    out = {}
    if not os.path.exists(prov):
        return out
    for line in open(prov, errors="replace"):
        for key in ("omp_max_threads", "omp_num_threads_env"):
            if line.startswith(key + "="):
                out[key] = line.split("=", 1)[1].strip()
    return out


def git_sha(repo: str) -> str:
    git = "/Library/Developer/CommandLineTools/usr/bin/git"
    git = git if os.path.exists(git) else "git"
    try:
        return subprocess.run([git, "-C", repo, "rev-parse", "HEAD"],
                              capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _binary_records(paths) -> dict:
    """Key each binary by its build directory (`build_gpu`, `build_cpu`, ...).

    Raises on a key collision rather than letting one binary silently overwrite
    another in the manifest — a manifest that quietly recorded only one of two
    builds would make a later provenance check pass for the wrong reason.
    """
    out = {}
    for path in paths:
        key = os.path.basename(os.path.dirname(os.path.dirname(path))) or path
        if key in out:
            raise ValueError(
                f"two --binary arguments both key to {key!r} "
                f"({out[key]['path']} and {path}); rename a build directory or "
                f"the manifest would silently record only one of them")
        out[key] = {"path": path, "md5": md5_file(path),
                    "libraries": linked_libraries(path)}
    return out


def cmd_bank(args: argparse.Namespace) -> int:
    arms = {}
    for spec in args.arm:
        if "=" not in spec:
            print(f"ERROR: --arm expects NAME=RUNDIR, got {spec!r}", file=sys.stderr)
            return 2
        name, run_dir = spec.split("=", 1)
        threads = run_threads(run_dir)
        arms[name] = {"run_dir": os.path.abspath(run_dir),
                      "threads": threads,
                      "files": digest_run(run_dir)}
        if not threads:
            print(f"  WARNING: {name} records no thread count (engine older "
                  f"than the omp_max_threads provenance key). A later verify "
                  f"cannot tell a thread-count change from a physics change.")
        print(f"  banked {name}: {len(arms[name]['files'])} files"
              f"{', ' + threads['omp_max_threads'] + ' threads' if threads.get('omp_max_threads') else ''}")

    configs = {}
    for spec in args.config:
        if "=" not in spec:
            print(f"ERROR: --config expects NAME=PATH, got {spec!r}", file=sys.stderr)
            return 2
        name, path = spec.split("=", 1)
        configs[name] = {"path": path, "md5": md5_file(os.path.join(args.repo, path))}

    # The engine commit comes from the runs, not from HEAD -- HEAD moves with
    # unrelated work (a docs commit) while the binary that produced the arms
    # does not.
    engine_sha = engine_sha_from_runs(a["run_dir"] for a in arms.values())
    sha_source = "provenance.txt"
    if not engine_sha:
        engine_sha = git_sha(args.repo)
        sha_source = "repository HEAD (NO provenance.txt in any arm)"
        print(f"WARNING: no arm carries a provenance.txt, so engine_git_sha "
              f"falls back to HEAD ({engine_sha[:10]}). That is the commit "
              f"checked out now, NOT necessarily the one the engine was built "
              f"from -- verify it before trusting this manifest.")

    manifest = {
        "schema": "ptm3d-validation-baseline/1",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "baseline_tag": args.tag,
        "engine_git_sha": engine_sha,
        "engine_git_sha_source": sha_source,
        "repo_head_at_bank": git_sha(args.repo),
        "repo": os.path.abspath(args.repo),
        "binaries": _binary_records(args.binary),
        "configs": configs,
        "notes": args.note,
        "arms": arms,
    }
    with open(args.out, "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"wrote {args.out}  ({len(arms)} arms, "
          f"{sum(len(a['files']) for a in arms.values())} files)")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    with open(args.manifest) as fh:
        manifest = json.load(fh)

    problems = 0

    # --- Provenance first. A byte comparison across a changed toolchain or a
    # changed config answers a different question than the suite is asking,
    # and the operator must be told before they read the byte verdict.
    if not args.repo:
        print("PROVENANCE SKIPPED: no --repo given, so the reference configs "
              "were NOT checked against the banked md5s. If one of them has "
              "been edited since the baseline was cut, the byte verdict below "
              "is about a different simulation than the baseline blessed.")
    if not args.binary:
        print("PROVENANCE SKIPPED: no --binary given, so linked libraries "
              "(Embree above all) were NOT checked. An Embree ISA swap moves "
              "lunar results ~0.009 K with the engine binary unchanged.")
    if args.repo:
        sha = git_sha(args.repo)
        banked_sha = manifest.get("engine_git_sha") or "unknown"
        # provenance.txt abbreviates the sha (10 chars) while `git rev-parse`
        # returns 40. Comparing them raw makes this warning fire on EVERY run,
        # including a perfect match -- which trains the operator to ignore the
        # one line that is supposed to stop them.
        k = min(len(sha), len(banked_sha))
        if k == 0 or sha[:k] != banked_sha[:k]:
            print(f"PROVENANCE: engine commit differs -- banked "
                  f"{banked_sha[:10]}, now {sha[:10]}. "
                  f"That is expected when validating a change; it means a byte "
                  f"diff below needs a justification, not that the suite is broken.")
        for name, rec in manifest.get("configs", {}).items():
            path = os.path.join(args.repo, rec["path"])
            if not os.path.exists(path):
                print(f"PROVENANCE FAIL: config {rec['path']} for arm '{name}' is GONE")
                problems += 1
            elif md5_file(path) != rec["md5"]:
                print(f"PROVENANCE FAIL: config {rec['path']} CHANGED since the "
                      f"baseline was cut (banked {rec['md5'][:10]}, now "
                      f"{md5_file(path)[:10]}). The arm no longer runs the "
                      f"physics the baseline blessed -- re-cut, do not compare.")
                problems += 1

    banked_bins = manifest.get("binaries", {})
    for spec in args.binary:
        key = os.path.basename(os.path.dirname(os.path.dirname(spec))) or spec
        brec = banked_bins.get(key)
        if brec is None:
            print(f"PROVENANCE SKIPPED: --binary {spec} keys to {key!r}, which is "
                  f"not in the manifest (it has {sorted(banked_bins)}). Its linked "
                  f"libraries were NOT checked against anything.")
            continue
        if md5_file(spec) != brec.get("md5"):
            print(f"PROVENANCE: engine binary {key} differs -- banked "
                  f"{brec.get('md5')}, now {md5_file(spec)}.")
        libs = linked_libraries(spec)
        for lib, was in brec.get("libraries", {}).items():
            now_lib = libs.get(lib)
            if now_lib and was.get("md5") and now_lib.get("md5") and \
                    now_lib["md5"] != was["md5"]:
                print(f"PROVENANCE: linked library CHANGED: {lib}\n"
                      f"    banked {was['md5']}  ({was.get('resolved')})\n"
                      f"    now    {now_lib['md5']}  ({now_lib.get('resolved')})\n"
                      f"    A library swap can move results with the engine "
                      f"binary unchanged (Embree ISA moves lunar T by "
                      f"~0.009 K). Byte-identity is NOT expected across it.")

    # --- Bytes.
    arms = dict(s.split("=", 1) for s in args.arm) if args.arm else \
        {k: v["run_dir"] for k, v in manifest["arms"].items()}

    total = ident = 0
    for name, run_dir in sorted(arms.items()):
        if name not in manifest["arms"]:
            # A typo in --arm must NOT pass silently: that is the exact failure
            # class this tool exists to prevent.
            print(f"FAIL {name}: no such arm in the manifest "
                  f"(it has {sorted(manifest['arms'])})")
            problems += 1
            continue
        banked = manifest["arms"][name]["files"]
        banked_threads = manifest["arms"][name].get("threads") or {}
        now_threads = run_threads(run_dir)
        b_nt = banked_threads.get("omp_max_threads")
        n_nt = now_threads.get("omp_max_threads")
        if b_nt is None or n_nt is None:
            print(f"PROVENANCE SKIPPED for {name}: thread count unknown "
                  f"(banked {b_nt!r}, now {n_nt!r}). Byte-identity across an "
                  f"unknown thread-count change is not evidence about physics.")
        elif b_nt != n_nt:
            print(f"PROVENANCE: {name} ran at {n_nt} threads, banked at {b_nt}. "
                  f"Statistics are thread-count dependent; a byte diff below is "
                  f"expected and says NOTHING about the change under test. "
                  f"Re-run at {b_nt} threads before reading the verdict.")
        try:
            now = digest_run(run_dir)
        except FileNotFoundError as exc:
            print(f"FAIL {name}: {exc}")
            problems += 1
            continue
        missing = sorted(set(banked) - set(now))
        extra = sorted(set(now) - set(banked))
        diffs = sorted(f for f in set(banked) & set(now)
                       if banked[f]["md5"] != now[f]["md5"])
        n = len(banked)
        total += n
        ident += n - len(diffs) - len(missing)
        status = "IDENTICAL" if not (diffs or missing or extra) else "DIFFERS"
        print(f"{status:9s} {name}: {n - len(diffs) - len(missing)}/{n} files match")
        for f in missing:
            print(f"    MISSING {f}")
        for f in extra:
            print(f"    EXTRA   {f}")
        for f in diffs:
            print(f"    DIFF    {f}")
        if diffs or missing or extra:
            problems += 1

    n_banked = len(manifest["arms"])
    scope = f"{len(arms)} arm(s)"
    if len(arms) < n_banked:
        scope += (f" -- a SUBSET: {n_banked - len(arms)} of the {n_banked} banked "
                  f"arms were not verified in this invocation")
    print(f"\n{ident}/{total} files byte-identical across {scope}")
    if problems:
        print("VERDICT: MISMATCH -- investigate, do not widen a tolerance.")
        return 1
    print("VERDICT: BYTE-IDENTICAL")
    return 0


def cmd_equivalence(args: argparse.Namespace) -> int:
    """Compare two finished run directories directly (N=1 vs N=4, CPU vs GPU)."""
    a, b = args.a, args.b
    da, db = digest_run(a), digest_run(b)
    missing = sorted(set(da) ^ set(db))
    diffs = sorted(f for f in set(da) & set(db) if da[f]["md5"] != db[f]["md5"])
    print(f"A = {a}\nB = {b}")
    for f in missing:
        print(f"    ONLY-IN-ONE {f}")
    for f in diffs:
        print(f"    DIFF        {f}")
    n = len(set(da) | set(db))
    ok = n - len(missing) - len(diffs)
    print(f"{ok}/{n} files byte-identical")
    if missing or diffs:
        print("VERDICT: NOT EQUIVALENT")
        return 1
    print("VERDICT: BYTE-IDENTICAL")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("bank", help="digest finished runs into a baseline manifest")
    b.add_argument("--arm", action="append", default=[], metavar="NAME=RUNDIR",
                   required=True)
    b.add_argument("--binary", action="append", default=[], required=True)
    b.add_argument("--config", action="append", default=[], metavar="NAME=REPOPATH")
    b.add_argument("--repo", required=True)
    b.add_argument("--tag", required=True, help="the git tag this manifest blesses")
    b.add_argument("--note", default="")
    b.add_argument("--out", required=True)
    b.set_defaults(fn=cmd_bank)

    v = sub.add_parser("verify", help="compare finished runs against a manifest")
    v.add_argument("--manifest", required=True)
    v.add_argument("--arm", action="append", default=[], metavar="NAME=RUNDIR")
    v.add_argument("--binary", action="append", default=[])
    v.add_argument("--repo", default="")
    v.set_defaults(fn=cmd_verify)

    e = sub.add_parser("equivalence", help="byte-compare two run directories")
    e.add_argument("a")
    e.add_argument("b")
    e.set_defaults(fn=cmd_equivalence)

    args = p.parse_args(argv)
    try:
        return args.fn(args)
    except ValueError as exc:          # usage errors raised by the helpers
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:   # a missing run directory is not a pass
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
