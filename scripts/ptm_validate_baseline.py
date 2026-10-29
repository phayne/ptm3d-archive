#!/usr/bin/env python3
"""Bank and verify PTM3D byte-identity validation baselines.

This is the tooling half of `docs/dev/validation-baselines.md`. The doctrine half
explains *why*; this file is the part that runs.

Four subcommands:

  run     Execute the arm set defined in `scripts/validation_arms.json` (or
          replayed out of a manifest) with a named binary, into a run root you
          choose. Builds the shared view-factor inputs if they are absent, and
          checks each arm's log for the dispatch rung it is supposed to have
          taken -- an arm that silently fell back to the CPU is a tautology,
          not evidence.

  bank    Walk a set of finished run directories, digest every output file the
          suite compares, and write a single `baseline_manifest.json` that
          records the digests together with the provenance that makes them
          meaningful: engine commit, engine binary md5, the build's own CMake
          flags (`PTM_ARCH_NATIVE` above all), every dynamic library the engine
          links (Embree above all), the md5 of each config that produced an
          arm, and -- since schema 3 -- the RECIPE for each arm, so the bank
          can be re-run from itself.

  verify  Run the arms FRESH and compare against a banked manifest, or compare
          run directories you name. Any digest mismatch is a regression. Any
          provenance mismatch (different config bytes, different Embree,
          different engine) is reported FIRST and loudly, because comparing
          bytes across a toolchain change answers a different question than
          the one the suite is asking.

  equivalence  Byte-compare two finished run directories directly.

⛔ WHY `verify` NO LONGER ACCEPTS A BARE `--manifest`
Until 2026-09-05, `verify --manifest M` with no `--arm` fell back to
`{k: v["run_dir"] for k, v in manifest["arms"].items()}` -- the manifest's OWN
recorded run directories. It re-digested the banked runs and compared them with
themselves. It printed `330/330 files byte-identical across 13 arm(s)`, every
arm `IDENTICAL`, `VERDICT: BYTE-IDENTICAL`, from a tree nine merges past the
bank, including GPU arms the accompanying document calls unreproducible. That
was the exact command `docs/dev/validation-baselines.md` section 5 printed and the
`ptm-validate` skill called the fast path. A gate that cannot fail is worse
than no gate, because it is reported as a pass.

The self-comparison is now impossible rather than discouraged: `verify`
requires either `--rerun` (run every arm fresh into a scratch directory and
compare that) or explicit `--arm NAME=RUNDIR` pairs, and it REFUSES any run
directory that resolves to a banked one. The bank is read-only input.

Exit codes: 0 = match, 1 = mismatch or missing input, 2 = usage error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone

# Output files the suite compares. Everything under `statistics/` (including
# the `_raw/` subdirectory written by output.write_raw_stats) is in scope; the
# skip list holds files whose bytes legitimately vary run to run.
SKIP_NAMES = {
    "provenance.txt",   # records wall-clock time and the invoking command
    "manifest.json",    # generated post hoc by ptm_qa.py
    ".DS_Store",
}

#: Manifest schema. /2 added `binaries[*].build_flags`; /3 adds `arms[*].recipe`
#: and top-level `inputs`, which together are what let `verify --rerun` produce
#: a fresh run instead of re-digesting the bank.
MANIFEST_SCHEMA = "ptm3d-validation-baseline/3"
SUITE_SCHEMA = "ptm3d-validation-suite/1"

#: Written into every bank directory this tool creates. Its presence marks a
#: directory as load-bearing, and `run`/`verify` refuse to write inside one.
#: The previous bank (`postmerge-baseline-2026-08-24`) was deleted in a routine
#: disk cleanup because nothing on disk said it was anything.
DO_NOT_DELETE = "DO-NOT-DELETE.md"
DO_NOT_DELETE_MARKER = "PTM3D VALIDATION BASELINE BANK"


class SuiteError(RuntimeError):
    """An arm did not run correctly, or its inputs are absent.

    Distinct from the usage errors raised as ValueError because they mean
    different things to a caller: a usage error is "you asked wrongly" (exit 2)
    and this is "the thing you asked for did not work" (exit 1). Collapsing
    them was how a failed suite could be read as a mistyped command.
    """


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


def build_flags(binary: str) -> dict:
    """The CMake cache variables that change the code the compiler generates.

    An engine binary md5 identifies a build but does not EXPLAIN it, and the
    manifest previously recorded no build configuration at all. That gap cost a
    month: `validate-baseline-2026-08-15` was cut from a `PTM_ARCH_NATIVE=ON`
    build (AVX2 + AVX-512), every later worktree configured `OFF`, and the
    resulting byte differences were read as unexplained physics drift for eight
    days across four branches. Rebuilding `ecf635a` with `ON` reproduces the
    banked lunar radiation-off arm 16/16; with `OFF` it does not, and the actual
    physical difference is one face at 3.2e-20 -- a denormal.

    So the flags are read from the build's own `CMakeCache.txt` rather than
    asked of the operator: a field that must be remembered is a field that will
    be forgotten, and this one was.

    Returns `{"_cmake_cache": <path>, "<VAR>": "<value>", ...}`, or a record
    carrying `_error` when the cache cannot be read. It never returns silently
    empty -- a missing cache is reported, because "no flags recorded" and "flags
    recorded as absent" are different claims.
    """
    # <build>/bin/ptm_thermal_sim -> <build>/CMakeCache.txt
    build_dir = os.path.dirname(os.path.dirname(os.path.abspath(binary)))
    cache = os.path.join(build_dir, "CMakeCache.txt")
    wanted = ("CMAKE_BUILD_TYPE", "PTM_ARCH_NATIVE", "PTM_ENABLE_GPU",
              "GPU_BACKEND", "PTM_USE_VTK_READER", "BUILD_SHARED_LIBS",
              "CMAKE_CXX_COMPILER", "CMAKE_CXX_FLAGS")
    out = {"_cmake_cache": cache}
    if not os.path.exists(cache):
        out["_error"] = ("no CMakeCache.txt beside the binary; build flags are "
                         "UNRECORDED for this arm")
        return out
    try:
        with open(cache, errors="replace") as fh:
            for line in fh:
                name, sep, rest = line.partition(":")
                if not sep or "=" not in rest:
                    continue
                if name in wanted:
                    out[name] = rest.split("=", 1)[1].strip()
    except OSError as exc:
        out["_error"] = f"could not read {cache}: {exc}"
    missing = [k for k in wanted if k not in out]
    if missing:
        out["_absent_from_cache"] = missing
    return out


def rpath_dirs(binary: str) -> list:
    """The LC_RPATH search paths baked into `binary`, expanded.

    Needed because `otool -L` reports the project's OWN libraries as
    `@rpath/libthermal.dylib`, which `os.path.exists` cannot resolve -- so
    before 2026-09-05 the manifest recorded `md5: null` for every one of them.
    With `BUILD_SHARED_LIBS=ON` (the canonical configuration) that is the whole
    engine: a thermal-only change moves `libthermal.dylib` and leaves
    `ptm_thermal_sim`'s md5 untouched, and the provenance block said nothing
    had changed. A provenance record that cannot see the code is not one.
    """
    dirs = []
    try:
        out = subprocess.run(["otool", "-l", binary], capture_output=True,
                             text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return dirs
    lines = out.splitlines()
    for i, line in enumerate(lines):
        if "LC_RPATH" not in line:
            continue
        for j in range(i, min(i + 6, len(lines))):
            m = re.search(r"path (.+?) \(offset", lines[j])
            if not m:
                continue
            p = m.group(1)
            p = p.replace("@loader_path", os.path.dirname(os.path.abspath(binary)))
            p = p.replace("@executable_path", os.path.dirname(os.path.abspath(binary)))
            dirs.append(p)
            break
    return dirs


def text_digest(path: str) -> str:
    """md5 of the library's executable code, ignoring LC_UUID and LC_RPATH.

    Two builds of identical source at different absolute paths produce
    dylibs that differ in exactly 48 bytes -- the LC_UUID and an LC_RPATH
    carrying the build directory -- so a whole-file md5 comparison across two
    worktrees reports a difference that is not a difference. Measured
    2026-09-04 on `libthermal.dylib`.

    Hashing the `__TEXT,__text` section instead gives a digest that is stable
    across the build path and still moves the instant any code does. Report
    both: whole-file md5 for the same-path case (which is the strict one), and
    this for the cross-worktree case, so "identical" and "identical apart from
    where it was built" are distinguishable claims rather than one argument.
    """
    try:
        out = subprocess.run(["otool", "-s", "__TEXT", "__text", path],
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return ""
    h = hashlib.md5()
    n = 0
    for line in out.splitlines():
        parts = line.split()
        # "<address> <byte> <byte> ..." -- drop the address column, which moves
        # with section layout rather than with code.
        if len(parts) < 2 or not re.fullmatch(r"[0-9a-f]{8,16}", parts[0]):
            continue
        body = parts[1:]
        if not all(re.fullmatch(r"[0-9a-f]{2}", p) for p in body):
            continue
        h.update("".join(body).encode())
        n += 1
    # An empty digest must not look like a valid one: md5("") is a real hex
    # string and would compare equal between two libraries this failed to read.
    return h.hexdigest() if n else ""


def linked_libraries(binary: str) -> dict:
    """md5 every dynamic library the engine links.

    An engine binary md5 CANNOT cover a dynamic library that moves results.
    Embree is the live example: swapping the SSE2 Homebrew build for a
    multi-ISA one shifts lunar temperatures by ~0.009 K with the engine binary
    completely unchanged. Record the libraries or the provenance is a lie.

    Under `BUILD_SHARED_LIBS=ON` the engine's own physics also lives in
    dylibs, reported by `otool -L` as `@rpath/...`; those are resolved through
    the binary's LC_RPATH rather than recorded as unknown.
    """
    libs = {}
    try:
        out = subprocess.run(["otool", "-L", binary], capture_output=True,
                             text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"_error": f"otool failed: {exc}"}
    search = rpath_dirs(binary)
    for line in out.splitlines()[1:]:
        path = line.strip().split(" (compatibility")[0].strip()
        if not path:
            continue
        real = os.path.realpath(path)
        if not os.path.exists(real) and path.startswith("@rpath/"):
            leaf = path[len("@rpath/"):]
            for d in search:
                cand = os.path.join(d, leaf)
                if os.path.exists(cand):
                    real = os.path.realpath(cand)
                    break
        entry = {"resolved": real}
        if os.path.exists(real):
            entry["md5"] = md5_file(real)
            entry["bytes"] = os.path.getsize(real)
            entry["text_md5"] = text_digest(real)
        else:
            # Framework entries, and any @rpath the LC_RPATH list cannot reach.
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
        flags = build_flags(path)
        if "_error" in flags:
            print(f"  WARNING: {key}: {flags['_error']}. A later verify cannot "
                  f"tell an ISA/build-configuration change from a physics "
                  f"change; PTM_ARCH_NATIVE alone moves bytes.")
        out[key] = {"path": path, "md5": md5_file(path),
                    "build_flags": flags,
                    "libraries": linked_libraries(path)}
    return out


# ---------------------------------------------------------------------------
# Running the arms. This is the half the file did not have.
#
# The module docstring used to say "Producing the run directories is the job of
# the suite driver (scripts/run_validation_baseline.sh)". THAT FILE HAS NEVER
# EXISTED in this repository. The only executable copy of the arm set lived in
# an untracked shell script under ~/projects, which meant `verify` had nothing
# to compare against except the bank itself -- which is exactly what it did.
# The runner belongs here, next to the comparator, reading a tracked suite.
# ---------------------------------------------------------------------------

def bank_root_of(path: str):
    """The banked directory `path` sits in, or None.

    A bank is marked by its own DO-NOT-DELETE.md; that file is the ONLY thing
    standing between a bank and the next disk cleanup, so it is also what this
    tool keys its write-protection on.
    """
    cur = os.path.realpath(path)
    while True:
        marker = os.path.join(cur, DO_NOT_DELETE)
        if os.path.isfile(marker):
            try:
                if DO_NOT_DELETE_MARKER in open(marker, errors="replace").read():
                    return cur
            except OSError:
                pass
        parent = os.path.dirname(cur)
        if parent == cur:
            return None
        cur = parent


def refuse_if_banked(path: str, what: str) -> None:
    """Fatal if `path` would write inside a bank. Banks are read-only input."""
    root = bank_root_of(path)
    if root:
        raise ValueError(
            f"{what} resolves to {os.path.realpath(path)}, which is inside the "
            f"validation bank at {root}. A bank is READ-ONLY input: the "
            f"documented re-run workflow used to write over the banked runs, "
            f"which destroys the only copy of the reference the moment "
            f"something goes wrong. Point this at a scratch directory instead.")


def load_suite(path: str) -> dict:
    with open(path) as fh:
        suite = json.load(fh)
    if suite.get("schema") != SUITE_SCHEMA:
        raise ValueError(f"{path} is schema {suite.get('schema')!r}, this tool "
                         f"speaks {SUITE_SCHEMA!r}")
    return suite


def resolve_args(argv, inputs_dir: str, self_path: str = "") -> list:
    """Substitute `{input:NAME}` and `{self}` in an arm's argument list.

    Placeholders rather than absolute paths because the view-factor matrices
    are machine-local: a manifest that baked in `~/projects/<somebody>/vf/...`
    would be unreplayable the moment that directory moved, which is how the
    last bank became unusable.
    """
    out = []
    for a in argv:
        m = re.fullmatch(r"\{input:(.+)\}", a)
        if m:
            if not inputs_dir:
                raise ValueError(
                    f"argument {a!r} needs an inputs directory and none was "
                    f"given (--inputs-dir)")
            out.append(os.path.join(inputs_dir, m.group(1)))
        elif a == "{self}":
            out.append(self_path)
        else:
            out.append(a)
    return out


def check_kernels(repo: str, bodies) -> list:
    """Which SPICE kernels the requested arms need and this checkout lacks.

    `data/kernels/{lsk,pck,spk,fk}` is gitignored, so a fresh worktree has the
    metakernels and none of the kernels they name, and every arm dies with
    `FATAL: SPICE initialization failed for body 'Moon'`. Report that BEFORE
    burning a run on it, and name the one command that fixes it.
    """
    need = [("lsk", "naif0012.tls"), ("pck", "pck00011.tpc"),
            ("spk", "de440s.bsp"), ("pck", "moon_pa_de440_200625.bpc"),
            ("fk", "moon_de440_250416.tf")]
    if "europa" in bodies:
        need.append(("spk", "jup365.bsp"))
    if "bennu" in bodies:
        need.append(("spk", "sb-101955-118_long.bsp"))
    root = os.path.join(repo, "data", "kernels")
    return [f"{s}/{n}" for s, n in need
            if not os.path.exists(os.path.join(root, s, n))]


def check_log(log_path: str, arm: dict) -> list:
    """Confirm the arm took the code path it exists to cover.

    A GPU arm that quietly fell back to the CPU still writes a full statistics
    directory and still compares byte-identical to another arm that fell back
    the same way. That verdict is a fallback tautology. The suite records the
    log line each rung prints; this checks for it, and a missing one is a
    FAILURE of the arm, not a note.
    """
    try:
        text = open(log_path, errors="replace").read()
    except OSError as exc:
        return [f"cannot read the log ({exc}), so the dispatch rung is UNVERIFIED"]
    problems = []
    for pat in arm.get("log_must_match", []):
        if not re.search(pat, text):
            problems.append(f"log does NOT contain /{pat}/ — the rung this arm "
                            f"exists to cover did not engage")
    for pat in arm.get("log_must_not_match", []):
        if re.search(pat, text):
            problems.append(f"log DOES contain /{pat}/ — the arm took a "
                            f"different path than intended")
    return problems


def run_one(binary: str, repo: str, config: str, out_dir: str, extra,
            log_path: str, env=None) -> tuple:
    """One engine invocation, from the repository root. Returns (rc, seconds).

    cwd is the repository root deliberately: the shipped metakernels use
    PATH_VALUES relative to it, so SPICE resolves nothing from anywhere else.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    cmd = [os.path.abspath(binary), "--config", config, "--output", out_dir] + list(extra)
    t0 = time.time()
    with open(log_path, "w") as fh:
        fh.write("# " + " ".join(cmd) + "\n")
        fh.flush()
        rc = subprocess.call(cmd, cwd=repo, stdout=fh, stderr=subprocess.STDOUT,
                             env=env)
    return rc, time.time() - t0


def run_suite(suite: dict, repo: str, binary: str, run_root: str,
              inputs_dir: str, names, threads: str = "48",
              build_inputs: bool = True, engine_args=()) -> dict:
    """Run `names` into `run_root/<name>`. Returns {name: run_dir}. Raises on failure.

    Every failure here is fatal and loud. A run that returns non-zero, writes
    no statistics, or whose log does not show the intended dispatch rung is NOT
    banked and NOT compared: half a suite reported as a suite is how a green
    verdict stops meaning anything.
    """
    refuse_if_banked(run_root, "the run root")

    arms = suite["arms"]
    unknown = [n for n in names if n not in arms]
    if unknown:
        raise ValueError(f"no such arm(s) in the suite: {', '.join(unknown)} "
                         f"(it has {sorted(arms)})")
    if not names:
        raise ValueError("no arms selected, so there is nothing to run. An "
                         "empty selection must not be mistaken for a suite "
                         "that passed.")

    # Which shared matrices the selected arms load, resolved BEFORE anything
    # runs. Deferring this meant a missing --inputs-dir let the engine build a
    # multi-minute view-factor matrix into the repository root and only then
    # abort on the argument that was missing all along.
    needed_inputs = set()
    for n in names:
        for a in arms[n].get("args", []):
            m = re.fullmatch(r"\{input:(.+)\}", a)
            if m:
                needed_inputs.add(m.group(1))
    if needed_inputs and not inputs_dir:
        raise ValueError(
            f"these arms load shared view-factor matrices "
            f"({', '.join(sorted(needed_inputs))}) and no --inputs-dir was "
            f"given. Point it at an existing set (the bank's own inputs/ is "
            f"the right answer for a regression run) or at a scratch "
            f"directory to build them into.")
    if inputs_dir and not os.path.isdir(inputs_dir):
        # Only create it when it does not exist, and only outside a bank: the
        # bank is READ-ONLY input, and an input built into it would be adopted
        # by every later run as if it were the banked reference.
        root = bank_root_of(inputs_dir)
        if root:
            raise SuiteError(
                f"{inputs_dir} does not exist, and it is inside the bank at "
                f"{root}. A bank is READ-ONLY input; this bank has no inputs/ "
                f"to load, so it cannot referee an arm that loads one. Repair "
                f"the bank deliberately, or point --inputs-dir at a scratch "
                f"directory and accept that its matrices are rebuilds.")
        os.makedirs(inputs_dir, exist_ok=True)
    os.makedirs(run_root, exist_ok=True)

    bodies = {arms[n].get("kernels") for n in names} - {None}
    missing = check_kernels(repo, bodies)
    if missing:
        raise SuiteError(
            "this checkout has no SPICE kernels: missing " +
            ", ".join(missing) + f" under {repo}/data/kernels.\n"
            "data/kernels/{lsk,pck,spk,fk} is gitignored, so a fresh worktree "
            "never has them and every arm dies with 'FATAL: SPICE "
            "initialization failed'. Link them from a checkout that does:\n"
            f"    python3 {repo}/scripts/fetch_kernels.py "
            f"--link-from <a checkout with kernels>\n"
            "(that verifies every pinned sha256 before it links), or download "
            "them with the same script and no --link-from.")

    env = dict(os.environ)
    if threads:
        env["OMP_NUM_THREADS"] = str(threads)

    # The shared view-factor matrices, built once with THIS binary so that
    # every arm below loads identical input bytes and the builder is not a
    # variable in the comparison.
    for name in sorted(needed_inputs):
        spec = suite.get("inputs", {}).get(name)
        if spec is None:
            raise SuiteError(f"arm arguments reference input {name!r} which "
                             f"the suite does not define")
        dest = os.path.join(inputs_dir, name)
        if os.path.exists(dest):
            print(f"[input] {name}: present ({os.path.getsize(dest)} bytes, "
                  f"md5 {md5_file(dest)[:10]}) — reused, not rebuilt")
            continue
        if not build_inputs:
            raise SuiteError(f"input {name} is absent from {inputs_dir} and "
                             f"--no-build-inputs was given")
        # A REBUILT matrix is not the banked one, and a bank missing an input
        # must be repaired deliberately, not topped up by the next regression
        # run -- which would then compare every future change against a matrix
        # nobody blessed. SuiteError, not the usage-level refusal: the operator
        # asked correctly and the REFERENCE is incomplete.
        root = bank_root_of(dest)
        if root:
            raise SuiteError(
                f"input {name} is missing from the bank at {root}, and a bank "
                f"is READ-ONLY input: building a replacement into it would "
                f"silently promote a rebuilt view-factor matrix to the "
                f"reference, and every later run would 'reuse' it as if it "
                f"had been blessed. Repair the bank deliberately, or point "
                f"--inputs-dir at a scratch directory and accept that the "
                f"matrix is a rebuild.")
        print(f"[input] {name}: BUILDING with this binary")
        scratch = os.path.join(run_root, "_inputbuild_" + name)
        shutil.rmtree(scratch, ignore_errors=True)
        rc, secs = run_one(binary, repo, spec["config"], scratch,
                           resolve_args(spec["args"], inputs_dir, dest),
                           os.path.join(run_root, "logs",
                                        f"_inputbuild_{name}.log"), env)
        if rc != 0 or not os.path.exists(dest):
            raise SuiteError(f"building input {name} failed (rc={rc}); see "
                             f"{run_root}/logs/_inputbuild_{name}.log")
        print(f"[input] {name}: built in {secs:.0f}s, "
              f"md5 {md5_file(dest)[:10]}")

    made = {}
    status = os.path.join(run_root, "STATUS.txt")
    for name in names:
        arm = arms[name]
        out = os.path.join(run_root, name)
        refuse_if_banked(out, f"the output directory for arm {name}")
        shutil.rmtree(out, ignore_errors=True)
        log = os.path.join(run_root, "logs", name + ".log")
        # engine_args: appended to EVERY arm (e.g. the indirect-aware spin-up seed legacy arm,
        # --engine-arg=--seed-indirect --engine-arg=off). Recorded in STATUS.txt.
        rc, secs = run_one(binary, repo, arm["config"], out,
                           resolve_args(arm.get("args", []), inputs_dir) + list(engine_args),
                           log, env)
        try:
            nfiles = len(digest_run(out))
        except FileNotFoundError:
            # No statistics/ at all, or an empty one. Counted as zero and
            # reported below rather than raised here, so the STATUS line for
            # this arm still gets written -- a suite that dies without
            # recording which arm died is harder to resume than one that does.
            nfiles = 0
        problems = ([f"engine returned {rc}"] if rc else []) + \
                   ([] if nfiles else ["wrote NO statistics files"]) + \
                   check_log(log, arm)
        with open(status, "a") as fh:
            fh.write(f"{name}\trc={rc}\t{secs:.0f}s\t{nfiles} files\t"
                     f"{'OK' if not problems else 'FAILED'}\n")
        if problems:
            for p in problems:
                print(f"  FAIL {name}: {p}")
            raise SuiteError(f"arm {name} did not run correctly; see {log}")
        print(f"  ran {name}: {secs:.0f}s, {nfiles} files")
        made[name] = out
    return made


def cmd_run(args: argparse.Namespace) -> int:
    suite = load_suite(args.suite)
    names = args.arm or [n for n, a in suite["arms"].items()
                         if args.group in ("all", a.get("group"))]
    print(f"binary {args.binary}  md5 {md5_file(args.binary)}")
    print(f"repo   {args.repo}")
    print(f"arms   {len(names)}: {', '.join(names)}")
    if args.engine_arg:
        print(f"engine args appended to every arm: {' '.join(args.engine_arg)}")
    run_suite(suite, args.repo, args.binary, args.run_root, args.inputs_dir,
              names, args.threads, not args.no_build_inputs, args.engine_arg)
    print(f"\nAll {len(names)} arm(s) ran. Run root: {args.run_root}")
    return 0


def copy_run_into_bank(run_dir: str, dest: str) -> None:
    """Copy one arm's comparable output into the bank, and prove the copy.

    Banking a digest of a directory that then gets deleted is how
    `postmerge-baseline-2026-08-24` became a manifest pointing at nothing. The
    bank holds the bytes.

    The copy is verified against the source rather than trusted: a truncated
    copy would otherwise be banked as the reference, and every later comparison
    would be against the truncation.

    It is staged and swapped rather than merged in place. Copying straight onto
    an existing `<bank>/runs/<arm>` with `dirs_exist_ok` merges: files the old
    bank had and the new run does not SURVIVE, files are overwritten one at a
    time, and the digest check that would catch it runs afterwards -- so a
    re-bank that failed would have already eaten part of the previous
    reference. Verify first, then swap; a failure leaves the old bank whole.
    """
    staged = dest + ".incoming"
    shutil.rmtree(staged, ignore_errors=True)
    os.makedirs(staged, exist_ok=True)
    shutil.copytree(os.path.join(run_dir, "statistics"),
                    os.path.join(staged, "statistics"))
    for side in ("provenance.txt", "summary.json"):
        src = os.path.join(run_dir, side)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(staged, side))
    if digest_run(staged) != digest_run(run_dir):
        shutil.rmtree(staged, ignore_errors=True)
        raise ValueError(f"the copy of {run_dir} into {dest} does not digest "
                         f"the same as its source; do not bank it")
    if os.path.exists(dest):
        retired = dest + ".superseded"
        shutil.rmtree(retired, ignore_errors=True)
        os.rename(dest, retired)
    os.rename(staged, dest)
    shutil.rmtree(dest + ".superseded", ignore_errors=True)


def write_do_not_delete(bank_dir: str, tag: str, manifests) -> None:
    with open(os.path.join(bank_dir, DO_NOT_DELETE), "w") as fh:
        fh.write(f"""# {DO_NOT_DELETE_MARKER} — DO NOT DELETE

This directory is the **named validation baseline** `{tag}` that
`docs/dev/validation-baselines.md` section 2.1 resolves to. It is the reference a
byte-identity regression compares against; deleting it does not free a cache,
it removes the project's only executable regression gate.

Manifest(s) here: {', '.join(os.path.basename(m) for m in manifests)}

It holds, per arm, the actual banked output bytes under `runs/<arm>/` and the
shared view-factor matrices under `inputs/`, so the bank is self-contained and
a comparison against it needs nothing else that might have moved.

Written by `scripts/ptm_validate_baseline.py bank`, {datetime.now(timezone.utc)
    .isoformat(timespec='seconds')}.

⛔ **Never delete without Paul's explicit word.** The previous bank
(`postmerge-baseline-2026-08-24`, at `<workdir>/postmerge_2026-08-24/`)
was removed in a routine disk cleanup on 2026-09-03 because nothing on disk
said it was load-bearing, and with it went the corrected `crater80N_sparse.vf`
that its GPU arms consumed. This file exists so that cannot happen silently
again: `ptm_validate_baseline.py` also refuses to WRITE anywhere beneath a
directory carrying this marker.
""")


def cmd_bank(args: argparse.Namespace) -> int:
    suite = load_suite(args.suite) if args.suite else None
    bank_dir = os.path.abspath(args.bank_dir) if args.bank_dir else None
    if bank_dir:
        os.makedirs(os.path.join(bank_dir, "runs"), exist_ok=True)

    arms = {}
    for spec in args.arm:
        if "=" not in spec:
            print(f"ERROR: --arm expects NAME=RUNDIR, got {spec!r}", file=sys.stderr)
            return 2
        name, run_dir = spec.split("=", 1)
        threads = run_threads(run_dir)
        source = os.path.abspath(run_dir)
        if bank_dir:
            dest = os.path.join(bank_dir, "runs", name)
            copy_run_into_bank(run_dir, dest)
            banked_dir = dest
        else:
            banked_dir = source
            print(f"  WARNING: {name} is banked by reference to {source}, "
                  f"which nothing protects. Pass --bank-dir to copy the bytes "
                  f"into the bank; a manifest whose run directories get "
                  f"cleaned up is a manifest that blesses nothing.")
        arms[name] = {"run_dir": banked_dir,
                      "source_run_dir": source,
                      "threads": threads,
                      "files": digest_run(banked_dir)}
        if suite is not None:
            if name not in suite["arms"]:
                print(f"ERROR: arm {name!r} is not in {args.suite} "
                      f"(it has {sorted(suite['arms'])}). A banked arm with no "
                      f"recipe cannot be re-run, which is the whole point.",
                      file=sys.stderr)
                return 2
            arms[name]["recipe"] = suite["arms"][name]
        if not threads:
            print(f"  WARNING: {name} records no thread count (engine older "
                  f"than the omp_max_threads provenance key). A later verify "
                  f"cannot tell a thread-count change from a physics change.")
        print(f"  banked {name}: {len(arms[name]['files'])} files"
              f"{', ' + threads['omp_max_threads'] + ' threads' if threads.get('omp_max_threads') else ''}")

    # The shared view-factor matrices. Recorded by md5 and, when a bank
    # directory is given, COPIED into it: an arm that loads a matrix is
    # meaningless without the matrix, and these are gitignored machine-local
    # multi-megabyte files that no clone has.
    inputs = {}
    if args.inputs_dir:
        for name in sorted(os.listdir(args.inputs_dir)):
            src = os.path.join(args.inputs_dir, name)
            if not os.path.isfile(src) or name.startswith("."):
                continue
            if bank_dir:
                os.makedirs(os.path.join(bank_dir, "inputs"), exist_ok=True)
                dst = os.path.join(bank_dir, "inputs", name)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
                if md5_file(dst) != md5_file(src):
                    print(f"ERROR: copy of input {name} into the bank does not "
                          f"match its source", file=sys.stderr)
                    return 1
                src = dst
            inputs[name] = {"path": src, "md5": md5_file(src),
                            "bytes": os.path.getsize(src)}
            print(f"  input {name}: {inputs[name]['bytes']} bytes, "
                  f"md5 {inputs[name]['md5'][:10]}")

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
        # /2 adds binaries[*].build_flags; /3 adds arms[*].recipe and inputs,
        # which is what makes `verify --rerun` possible. `verify` reads older
        # manifests and says what they cannot support rather than assuming.
        "schema": MANIFEST_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "baseline_tag": args.tag,
        "engine_git_sha": engine_sha,
        "engine_git_sha_source": sha_source,
        "repo_head_at_bank": git_sha(args.repo),
        "repo": os.path.abspath(args.repo),
        "bank_dir": bank_dir or "",
        "suite_file": os.path.abspath(args.suite) if args.suite else "",
        "suite_inputs": suite.get("inputs", {}) if suite else {},
        "binaries": _binary_records(args.binary),
        "configs": configs,
        "inputs": inputs,
        "notes": args.note,
        "arms": arms,
    }
    out = args.out
    if bank_dir and not os.path.isabs(out):
        out = os.path.join(bank_dir, out)
    with open(out, "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"wrote {out}  ({len(arms)} arms, "
          f"{sum(len(a['files']) for a in arms.values())} files)")
    if bank_dir:
        existing = sorted(f for f in os.listdir(bank_dir)
                          if f.startswith("baseline_manifest") and f.endswith(".json"))
        write_do_not_delete(bank_dir, args.tag,
                            [os.path.join(bank_dir, f) for f in existing])
        print(f"wrote {os.path.join(bank_dir, DO_NOT_DELETE)}")
    return 0


def resolve_verify_arms(args: argparse.Namespace, manifest: dict,
                        scratch: list) -> dict:
    """Decide WHICH run directories `verify` compares. The load-bearing fix.

    Exactly two ways to get here, and neither can be the bank:

      --rerun            run every banked arm fresh, from the recipe the
                         manifest carries, into a scratch run root, with the
                         binary the operator names. This is the honest default
                         and the one the documentation now prints.
      --arm NAME=RUNDIR  compare run directories the operator produced. Each is
                         checked against the banked path and REFUSED if it is
                         the same directory.

    There is deliberately no third way. The old fallback -- no `--arm`, so
    re-digest `manifest["arms"][*]["run_dir"]` -- compared the bank with
    itself, returned `VERDICT: BYTE-IDENTICAL` on every arm from any tree, and
    was the command both the skill and the document told people to run.
    """
    # Both are directories whose bytes ARE the reference: `run_dir` is the
    # bank's own copy, and `source_run_dir` is the scratch run it was copied
    # from -- which the documented `run --keep` then `bank` workflow leaves
    # lying around, and which the manifest names. Comparing either with the
    # manifest is a self-comparison; only the first was refused at first.
    banked = {}
    for name, rec in manifest["arms"].items():
        banked[name] = {os.path.realpath(p)
                        for p in (rec.get("run_dir"), rec.get("source_run_dir"))
                        if p}
    bank_dir = manifest.get("bank_dir") or ""

    if args.arm and args.rerun:
        raise ValueError("--arm and --rerun both given; they are two different "
                         "ways to answer the same question. Pick one.")

    if args.arm:
        arms = dict(s.split("=", 1) for s in args.arm if "=" in s)
        bad = [s for s in args.arm if "=" not in s]
        if bad:
            raise ValueError(f"--arm expects NAME=RUNDIR, got {bad!r}")
        for name, run_dir in arms.items():
            real = os.path.realpath(run_dir)
            if real in banked.get(name, set()):
                raise ValueError(
                    f"--arm {name}={run_dir} IS the banked run directory (or "
                    f"the scratch run its bytes were copied from). Comparing "
                    f"it with the manifest compares the baseline with itself "
                    f"and passes unconditionally, no matter what the engine "
                    f"now does. Run the arm somewhere else and point this at "
                    f"that. (`verify --rerun` does it for you.)")
            if bank_dir and (real == os.path.realpath(bank_dir) or
                             real.startswith(os.path.realpath(bank_dir) + os.sep)):
                raise ValueError(
                    f"--arm {name}={run_dir} is inside the bank "
                    f"{bank_dir}. The bank is the reference; a run under it is "
                    f"either the reference itself or an overwrite of it.")
        return arms

    if not args.rerun:
        raise ValueError(
            "verify needs to be told WHICH runs to compare against the "
            "baseline, and it will not fall back to the baseline's own run "
            "directories.\n"
            "  Until 2026-09-05 it did exactly that: a bare `verify "
            "--manifest M` re-digested the banked runs and compared them with "
            "themselves, returning a clean verdict for every arm from "
            "any tree, at any commit. That is not a gate.\n"
            "Use one of:\n"
            "  --rerun --binary <exe> --repo <repo> [--run-root DIR]\n"
            "        run every banked arm fresh from the recipe in the "
            "manifest and compare that\n"
            "  --arm NAME=RUNDIR [--arm ...]\n"
            "        compare run directories you produced yourself")

    # --rerun: replay the manifest's own recipes.
    if not args.binary or not args.repo:
        raise ValueError("--rerun needs --binary <exe> and --repo <repo>: it "
                         "runs the engine.")
    if len(args.binary) > 1:
        raise ValueError(
            f"--rerun runs ONE engine, but {len(args.binary)} --binary "
            f"arguments were given ({', '.join(args.binary)}). Arms produced "
            f"by different binaries cannot be banked or compared together.")
    binary = args.binary[0]
    recipes = {n: r.get("recipe") for n, r in manifest["arms"].items()}
    missing = sorted(n for n, r in recipes.items() if not r)
    if missing:
        raise ValueError(
            f"this manifest (schema {manifest.get('schema')}) carries no "
            f"recipe for {len(missing)} of its arms ({', '.join(missing[:4])}"
            f"{'...' if len(missing) > 4 else ''}), so they cannot be re-run "
            f"from it. Manifests banked before schema "
            f"{MANIFEST_SCHEMA} recorded only digests. Re-run those arms "
            f"yourself and pass --arm NAME=RUNDIR, or re-bank with --suite.")
    suite = {"schema": SUITE_SCHEMA,
             "arms": recipes,
             "inputs": _suite_inputs_from_manifest(manifest, args)}
    # A `--only` name that matches nothing must NOT quietly narrow the suite to
    # zero arms: that reaches the byte loop with an empty set and prints
    # `0/0 files byte-identical ... VERDICT: BYTE-IDENTICAL`, exit 0 -- the
    # same vacuous pass this whole subcommand was rewritten to make impossible.
    if args.only:
        bad = [n for n in args.only if n not in recipes]
        if bad:
            raise ValueError(
                f"--only names {', '.join(bad)}, which the manifest does not "
                f"bless (it has {sorted(recipes)}). A filter that matches "
                f"nothing would report a pass over nothing.")
    names = [n for n in sorted(recipes) if not args.only or n in args.only]

    run_root = args.run_root
    if not run_root:
        run_root = tempfile.mkdtemp(prefix="ptm_verify_")
        scratch.append(run_root)
    inputs_dir = args.inputs_dir or os.path.join(
        manifest.get("bank_dir") or "", "inputs")

    # The thread count is part of the baseline -- several OpenMP reductions sum
    # in thread-partition order -- so a rerun defaults to the count the arms
    # were BANKED at, not to this tool's own 48. Defaulting to 48 on a bank cut
    # at any other count guarantees a MISMATCH that says nothing about the code.
    banked_threads = {(rec.get("threads") or {}).get("omp_max_threads")
                      for rec in manifest["arms"].values()} - {None}
    threads = args.threads or "48"
    if args.threads is None:
        if len(banked_threads) == 1:
            threads = banked_threads.pop()
            print(f"  threads  {threads} (the count these arms were banked at)")
        elif banked_threads:
            raise ValueError(
                f"the banked arms do not share one thread count "
                f"({sorted(banked_threads)}), so a rerun cannot pick one for "
                f"you. Pass --threads explicitly and expect a byte diff on "
                f"any arm banked at a different count.")
        else:
            print(f"  threads  {threads} (the manifest records none, so this "
                  f"is this tool's default and a byte diff below may be a "
                  f"thread-count difference rather than a code change)")
    print(f"RE-RUNNING {len(names)} arm(s) into {run_root}")
    print(f"  binary   {binary}  md5 {md5_file(binary)}")
    print(f"  inputs   {inputs_dir}")
    try:
        if args.engine_arg:
            print(f"engine args appended to every re-run arm: {' '.join(args.engine_arg)}")
        return run_suite(suite, args.repo, binary, run_root, inputs_dir,
                         names, threads, build_inputs=True, engine_args=args.engine_arg)
    except Exception:
        # Whatever went wrong, the partial runs and their logs are the only
        # record of it. Say where they are; the caller only prints the
        # exception, and a temporary run root nobody names is a run root
        # nobody finds.
        print(f"the partial run root is at {run_root} (kept)")
        if run_root in scratch:
            scratch.remove(run_root)
        raise


def _suite_inputs_from_manifest(manifest: dict, args) -> dict:
    """Input build recipes for a --rerun, and a hard check on the input bytes.

    The bank's own copies of the view-factor matrices are the correct input:
    rebuilding them would make the comparison a test of the builder rather than
    of the change under test. If they are present, their md5 must match what
    the manifest recorded, or the run is silently loading a different matrix
    than the baseline did -- the failure mode the HMAT format cannot announce
    for itself, because a stale H-matrix cache still reports version 2.
    """
    inputs_dir = args.inputs_dir or os.path.join(
        manifest.get("bank_dir") or "", "inputs")
    for name, rec in manifest.get("inputs", {}).items():
        path = os.path.join(inputs_dir, name)
        if os.path.exists(path) and md5_file(path) != rec["md5"]:
            # SuiteError, not ValueError: the operator asked correctly and the
            # REFERENCE is wrong (exit 1). A wrapper that reads this as exit 2
            # would file a corrupted view-factor matrix as a typo.
            raise SuiteError(
                f"input {name} at {path} has md5 {md5_file(path)[:10]}, the "
                f"manifest banked {rec['md5'][:10]}. Loading it would compare "
                f"against a different view-factor matrix than the baseline "
                f"used; the HMAT format cannot announce that for itself.")
    # A rerun of a bank whose inputs are gone must rebuild them from the
    # suite's own recipes, and say that the matrix is a rebuild.
    recorded = manifest.get("suite_inputs") or {}
    if recorded:
        return recorded
    print("NOTE: this manifest records no input-build recipes, so any missing "
          "view-factor matrix would be rebuilt from this tool's own fallback "
          "table rather than from the bank. A rebuilt matrix is not the "
          "banked one; check the md5 the run prints.")
    return _DEFAULT_INPUT_RECIPES


#: Fallback input recipes, used only when a bank's `inputs/` copies are missing
#: and the matrices must be rebuilt. Kept in step with
#: `scripts/validation_arms.json`; a rebuilt matrix is announced by `run_suite`.
_DEFAULT_INPUT_RECIPES = {
    "crater80N_coupled.vf": {
        "config": "examples/tutorial04_crater80N_coupled.yaml",
        "args": ["--rad-view-factor-file", "{self}"]},
    "crater80N_hmat.vf": {
        "config": "examples/validation/baseline_crater80N_gputhermal.yaml",
        "args": ["--rad-view-factor-file", "{self}", "--gpu-devices", "1"]},
    "crater80N_sparse.vf": {
        "config": "examples/validation/baseline_crater80N_radgpu.yaml",
        "args": ["--rad-view-factor-file", "{self}", "--rad-gpu-devices", "1"]},
}


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
                  f"not in the manifest (it has {sorted(banked_bins)}). Neither its "
                  f"linked libraries NOR its build flags were checked against "
                  f"anything -- and the key is the build DIRECTORY name, so this "
                  f"fires exactly when you compare two differently-configured "
                  f"builds, which is when the flags matter most. This build reports "
                  f"PTM_ARCH_NATIVE="
                  f"{build_flags(spec).get('PTM_ARCH_NATIVE', 'unknown')}.")
            continue
        if md5_file(spec) != brec.get("md5"):
            print(f"PROVENANCE: engine binary {key} differs -- banked "
                  f"{brec.get('md5')}, now {md5_file(spec)}.")

        # Build configuration, before the bytes. `PTM_ARCH_NATIVE` alone moves
        # bytes with no source change at all, so a byte verdict taken across it
        # is answering a different question.
        was_flags = brec.get("build_flags")
        now_flags = build_flags(spec)
        if was_flags is None:
            print(f"PROVENANCE SKIPPED for binary {key}: this manifest predates "
                  f"build-flag recording, so the ISA configuration it was cut at "
                  f"is UNKNOWN. Baselines cut before 2026-08-24 on this machine "
                  f"were built with PTM_ARCH_NATIVE=ON; the current build reports "
                  f"{now_flags.get('PTM_ARCH_NATIVE', 'unknown')}. Treat a byte "
                  f"diff below as unattributable until the baseline is re-cut.")
        else:
            for var in sorted(set(was_flags) | set(now_flags)):
                if var.startswith("_"):
                    continue
                if was_flags.get(var) != now_flags.get(var):
                    print(f"PROVENANCE: build flag {var} differs for {key} -- "
                          f"banked {was_flags.get(var)!r}, now "
                          f"{now_flags.get(var)!r}. Generated code differs, so "
                          f"byte-identity is NOT expected across it.")

        libs = linked_libraries(spec)
        for lib, was in brec.get("libraries", {}).items():
            now_lib = libs.get(lib)
            if now_lib and was.get("md5") and now_lib.get("md5") and \
                    now_lib["md5"] != was["md5"]:
                # Same code at a different absolute build path differs in 48
                # bytes (LC_UUID + an LC_RPATH holding the build directory).
                # Say which of the two it is; "the library changed" and "the
                # library was built somewhere else" are different findings and
                # only one of them can move a result.
                same_code = (was.get("text_md5") and now_lib.get("text_md5")
                             and was["text_md5"] == now_lib["text_md5"])
                verdict = ("SAME CODE, different build path -- __TEXT,__text "
                           "digests match, so this is LC_UUID/LC_RPATH only "
                           "and cannot move a result"
                           if same_code else
                           "the code itself differs" if was.get("text_md5")
                           else "code digest UNRECORDED in this manifest, so "
                                "this cannot be attributed")
                print(f"PROVENANCE: linked library CHANGED: {lib}\n"
                      f"    banked {was['md5']}  ({was.get('resolved')})\n"
                      f"    now    {now_lib['md5']}  ({now_lib.get('resolved')})\n"
                      f"    {verdict}.\n"
                      f"    A library swap can move results with the engine "
                      f"binary unchanged (Embree ISA moves lunar T by "
                      f"~0.009 K). Byte-identity is NOT expected across it.")

    # --- Bytes. `resolve_verify_arms` is where the runs being compared come
    # from, and it cannot return the bank's own directories.
    scratch: list = []
    arms = resolve_verify_arms(args, manifest, scratch)

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
        status = "IDENTICAL" if not (diffs or missing or extra) else "DIFFERS"
        expect_differ = False
        if args.expect_differ_if_log:
            # The arm's own engine log decides whether a CHANGE is expected (the
            # Indirect-aware spin-up seed default arm: an arm whose log shows the new seed acting must
            # differ from the legacy bank; every other arm must stay identical).
            log = os.path.join(os.path.dirname(os.path.abspath(run_dir)), "logs", name + ".log")
            if not os.path.exists(log):
                print(f"FAIL {name}: --expect-differ-if-log given but no log at {log}; "
                      f"cannot tell whether a difference is expected")
                problems += 1
                continue
            expect_differ = re.search(args.expect_differ_if_log,
                                      open(log, errors="replace").read()) is not None
        n_match = n - len(diffs) - len(missing)
        if expect_differ:
            label = "EXPECTED-DIFFER" if status == "DIFFERS" else "FAIL-NO-CHANGE"
            print(f"{label:15s} {name}: {n_match}/{n} files match (log matches "
                  f"{args.expect_differ_if_log!r}: the change under test acts here)")
            if status == "IDENTICAL":
                problems += 1     # it claims to act and changed nothing: that is a bug
            continue              # not part of the byte-identity tally below
        total += n
        ident += n_match
        print(f"{status:9s} {name}: {n_match}/{n} files match")
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
    for name, run_dir in sorted(arms.items()):
        print(f"    {name:32s} <- {run_dir}")

    # A comparison that touched nothing is not a pass. `0/0 files
    # byte-identical ... VERDICT: BYTE-IDENTICAL` reads exactly like a real
    # green run, and is the same shape of lie as the self-comparison this
    # subcommand was rewritten to prevent.
    if total == 0:
        print("\nVERDICT: NOTHING WAS COMPARED -- 0 files across 0 arms. That "
              "is not a pass; it is a suite that did not run.")
        problems += 1

    for d in scratch:
        # Never delete the fresh runs when they disagree with the bank: they
        # are the evidence, and reproducing them costs the whole suite again.
        if args.keep or problems:
            print(f"scratch run root kept at {d}"
                  f"{' (MISMATCH -- kept so you can diff it)' if problems and not args.keep else ''}")
        else:
            shutil.rmtree(d, ignore_errors=True)
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

    default_suite = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "validation_arms.json")

    r = sub.add_parser("run", help="run the validation arms into a run root")
    r.add_argument("--suite", default=default_suite)
    r.add_argument("--repo", required=True, help="repository root; the engine "
                   "runs with this as its working directory, because the "
                   "shipped metakernels use paths relative to it")
    r.add_argument("--binary", required=True)
    r.add_argument("--run-root", required=True)
    r.add_argument("--inputs-dir", default="", help="where the shared "
                   "view-factor matrices live; built here if absent")
    r.add_argument("--arm", action="append", default=[], metavar="NAME")
    r.add_argument("--group", default="all", choices=["all", "cpu", "gpu"])
    r.add_argument("--threads", default="48", help="OMP_NUM_THREADS; "
                   "statistics are thread-count dependent, so this is part of "
                   "the baseline")
    r.add_argument("--no-build-inputs", action="store_true")
    r.add_argument("--engine-arg", action="append", default=[], metavar="ARG",
                   help="append ARG to every arm's engine command line (repeatable; "
                        "write --engine-arg=--seed-indirect --engine-arg=off)")
    r.set_defaults(fn=cmd_run)

    b = sub.add_parser("bank", help="digest finished runs into a baseline manifest")
    b.add_argument("--arm", action="append", default=[], metavar="NAME=RUNDIR",
                   required=True)
    b.add_argument("--binary", action="append", default=[], required=True)
    b.add_argument("--config", action="append", default=[], metavar="NAME=REPOPATH")
    b.add_argument("--repo", required=True)
    b.add_argument("--tag", required=True, help="the git tag this manifest blesses")
    b.add_argument("--note", default="")
    b.add_argument("--out", required=True)
    b.add_argument("--suite", default=default_suite, help="the arm definitions "
                   "to record as each arm's recipe, so `verify --rerun` can "
                   "replay them; pass '' to bank without recipes")
    b.add_argument("--inputs-dir", default="", help="shared view-factor "
                   "matrices to record (and copy into --bank-dir)")
    b.add_argument("--bank-dir", default="", help="copy the arms' output bytes "
                   "and inputs into this directory and mark it DO-NOT-DELETE, "
                   "so the manifest does not merely point at scratch runs that "
                   "a cleanup will remove, which is how the 2026-08-24 "
                   "baseline was lost")
    b.set_defaults(fn=cmd_bank)

    v = sub.add_parser("verify", help="run the arms fresh (or compare runs you "
                                      "name) against a manifest")
    v.add_argument("--manifest", required=True)
    v.add_argument("--arm", action="append", default=[], metavar="NAME=RUNDIR",
                   help="a run directory YOU produced. Refused if it is the "
                        "banked directory itself.")
    v.add_argument("--engine-arg", action="append", default=[], metavar="ARG",
                   help="with --rerun: append ARG to every arm's engine command line")
    v.add_argument("--expect-differ-if-log", default="", metavar="REGEX",
                   help="an arm whose engine log matches REGEX must DIFFER from the bank "
                        "(identical = FAIL); every other arm must stay byte-identical. "
                        "For the seed default arm: 'seed_indirect: series armed'")
    v.add_argument("--rerun", action="store_true",
                   help="run every banked arm fresh from the recipe in the "
                        "manifest and compare that. Needs --binary and --repo.")
    v.add_argument("--only", action="append", default=[], metavar="NAME",
                   help="with --rerun, restrict to these arms")
    v.add_argument("--run-root", default="", help="with --rerun, where the "
                   "fresh runs go (default: a temporary directory)")
    v.add_argument("--inputs-dir", default="", help="with --rerun, where the "
                   "shared view-factor matrices live (default: the bank's own "
                   "inputs/)")
    v.add_argument("--threads", default=None, dest="threads",
                   help="with --rerun, OMP_NUM_THREADS. Defaults to the count "
                        "the manifest banked, because statistics are "
                        "thread-count dependent")
    v.add_argument("--keep", action="store_true",
                   help="with --rerun, keep the scratch run root")
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
    except SuiteError as exc:          # an arm failed, or its input is absent
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:          # usage errors raised by the helpers
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:   # a missing run directory is not a pass
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
