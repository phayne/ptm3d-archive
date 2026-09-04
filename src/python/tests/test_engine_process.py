#!/usr/bin/env python3
"""Behavior of ptm.run.EngineProcess, against a stand-in for the engine.

The engine writes informational and warning banners to *stderr* during
successful runs, so anything that routes by stream reports a good run as
failed. That is the property most worth pinning, and it needs no engine to
check: a small Python child that emits the same shapes of line does it in a
second, on any machine, with no GPU and no SPICE kernels.

Run directly::

    python3 src/python/tests/test_engine_process.py
"""

from __future__ import annotations

import shlex
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "python"))

from ptm.run import (CompletedRun, EngineProcess, binary_provenance,  # noqa: E402
                     build_command, classify_severity, extract_progress,
                     resolve_engine, run_engine)

# Lines the engine really emits, and the severity each must be given. The
# stderr banners are the point: a successful PTM3D run prints all of these.
CLASSIFICATION_CASES = [
    ("[eq-seed] equilibration seed from cold-face analytic", "info"),
    ("[machine-profile] no host profile; using CONSERVATIVE defaults", "warning"),
    ("[GPU Thermal] dispatch self-chunk ceiling 8192 timesteps", "info"),
    ("Calculating illumination: 42.5%", "info"),
    ("Simulation completed in 5.030 seconds (5030 ms)", "info"),
    ("FATAL: Failed to load mesh from crater.ply", "error"),
    # The startup guards tag their fatals. Verbatim from the engine:
    ("FATAL [gpu]: this binary was built WITHOUT GPU support, but the run "
     "requests the GPU thermal solver.", "error"),
    ("FATAL [mesh-units]: mesh units are ambiguous", "error"),
    ("FATAL [coupling]: radiation.coupling_mode unknown", "error"),
    ("FATAL [exchange]: unknown exchange_selection", "error"),
    ("[FATAL] Caught signal 11", "error"),
    ("[FATAL] Backtrace:", "error"),
    # ...and the word inside another word must not fire.
    ("fatality rate of the cold trap", "info"),
    ("0 errors", "info"),
    ("[Statistics] Exported to output/quickstart_moon_diurnal/statistics", "info"),
    ("ERROR: mesh file not found", "error"),
    ("terminate called after throwing an instance of 'std::runtime_error'", "error"),
    ("Segmentation fault: 11", "error"),
    ("WARNING: falling back to the legacy grid", "warning"),
    ("[config] WARNING: 3 key(s) under `thermophysical:` are read by nothing", "warning"),
]

CHILD = textwrap.dedent("""
    import sys, time
    print("Mesh created with 642 vertices and 1280 faces", flush=True)
    sys.stderr.write("[eq-seed] equilibration seed from cold-face analytic\\n")
    sys.stderr.flush()
    print("Calculating illumination: 42.5%", flush=True)
    sys.stderr.write("[machine-profile] using CONSERVATIVE defaults\\n")
    sys.stderr.flush()
    print("Simulation completed in 0.010 seconds", flush=True)
    {tail}
    sys.exit({code})
""")


def _run_child(code=0, tail="pass", log_path=None):
    seen = {"info": [], "warning": [], "error": [], "progress": [], "beat": []}
    proc = EngineProcess(
        [sys.executable, "-c", CHILD.format(code=code, tail=tail)],
        on_output=seen["info"].append,
        on_warning=seen["warning"].append,
        on_error=seen["error"].append,
        on_progress=seen["progress"].append,
        on_heartbeat=lambda m, s: seen["beat"].append((m, s)),
        log_path=log_path,
    )
    return proc.run(), seen


def test_classification_of_real_engine_lines():
    for text, expected in CLASSIFICATION_CASES:
        got = classify_severity(text)
        assert got == expected, f"{text!r} classified {got}, expected {expected}"


def test_progress_extraction():
    assert extract_progress("Calculating illumination: 42.5%") == 42.5
    assert extract_progress("[####      ] 40%") == 40.0
    assert extract_progress("no percentage here") is None


def test_stderr_banners_are_not_errors():
    """A clean run must report zero errors even though it writes to stderr."""
    rc, seen = _run_child(code=0)
    assert rc == 0, f"child exited {rc}"
    assert not seen["error"], f"clean run reported errors: {seen['error']}"
    assert any("CONSERVATIVE" in line for line in seen["warning"]), (
        f"the conservative-defaults banner was not a warning: {seen}")
    assert any("Mesh created" in line for line in seen["info"])
    assert any("equilibration seed" in line for line in seen["info"]), (
        "an informational stderr banner was not routed to info")
    assert seen["progress"] == [42.5], seen["progress"]


def test_failure_is_reported_and_exit_code_propagates():
    rc, seen = _run_child(
        code=3, tail='sys.stderr.write("FATAL: mesh unreadable\\\\n")')
    assert rc == 3, f"exit status not propagated: {rc}"
    assert any("FATAL" in line for line in seen["error"]), seen


def test_stop_terminates_the_process_group():
    """stop() must kill a child that ignores nothing and never return late."""
    child = "import time\nwhile True: time.sleep(0.05)\n"
    proc = EngineProcess([sys.executable, "-c", child])
    import threading
    result = {}
    thread = threading.Thread(target=lambda: result.update(rc=proc.run()))
    thread.start()
    for _ in range(100):                       # wait for the child to exist
        if proc.process is not None:
            break
        time.sleep(0.02)
    assert proc.process is not None, "child never started"
    start = time.monotonic()
    proc.stop()
    thread.join(timeout=20)
    elapsed = time.monotonic() - start
    assert not thread.is_alive(), "run() did not return after stop()"
    assert elapsed < 15, f"stop() took {elapsed:.1f}s; the waits are meant to be bounded"
    assert proc.process.poll() is not None, "child still running after stop()"


def test_output_left_in_the_pipe_at_exit_is_still_split_into_lines():
    """The end of a run must arrive as lines, not welded into one.

    EngineProcess reads the pipes in 4 KB chunks while the child lives and
    reads whatever is left once it has exited. That last read routinely holds
    the whole end of the run, and emitting it unsplit made it a single "line":
    on_progress fired for none of it, and a consumer that decides from the
    first percentage in a line -- run_engine's throttle does -- then decided
    for all of it at once, dropping the completion banner welded in with the
    bar art. That was a real capture truncation, seen in CI.

    The child flushes and _exit()s, so its death is observed before the reader
    can drain: a C++ engine exits that way, and CPython only avoids it by
    spending a few ms in finalization after flushing. It writes more than one
    4 KB read of bar art, so the final read cannot be empty however the two
    processes happen to be scheduled.
    """
    child = textwrap.dedent("""
        import os, sys
        for label in ("Calculating illumination", "SIMD Batch thermal"):
            for pct in range(0, 101):
                sys.stdout.write("\\r%s [==] %d%%" % (label, pct))
            sys.stdout.write("\\n")
        sys.stdout.write("Simulation completed in 0.010 seconds (10 ms)\\n")
        sys.stdout.flush()
        os._exit(0)
    """)
    # One list for all three severities, so the check is about line structure
    # and arrival order and does not depend on how any line is classified.
    lines, progress = [], []
    proc = EngineProcess([sys.executable, "-c", child],
                         on_output=lines.append, on_warning=lines.append,
                         on_error=lines.append, on_progress=progress.append)
    assert proc.run() == 0, "child did not exit cleanly"

    welded = [ln for ln in lines if "\r" in ln or "\n" in ln]
    if welded:
        worst = max(len(w.replace("\r", "\n").split("\n")) for w in welded)
        raise AssertionError(
            f"{len(welded)} callback(s) carried more than one line; the worst "
            f"held {worst} of them welded together")
    assert len(progress) == 202, (
        f"on_progress fired {len(progress)} times, expected 202 -- the "
        f"readouts in the final read did not reach it")
    assert progress[-1] == 100.0, progress[-1]
    assert lines[-1] == "Simulation completed in 0.010 seconds (10 ms)", lines[-1]


def test_build_command_shape():
    base = build_command("/bin/engine", "cfg.yaml", None)
    assert base == ["/bin/engine", "--config", "cfg.yaml"]

    gpu = build_command("/bin/engine", "cfg.yaml", {"thermal": {"use_gpu": True}})
    assert gpu[-1] == "--use-gpu-thermal"
    assert build_command("/bin/engine", "cfg.yaml",
                         {"thermal": {"use_gpu": False}}) == base

    generic = build_command("/bin/engine", "cfg.yaml",
                            {"generic_body": {"config_file": "body.yaml"}})
    assert generic[-2:] == ["--generic-body-config", "body.yaml"]

    # Radiation stays in the YAML: no radiation flag may appear on the CLI.
    rad = build_command("/bin/engine", "cfg.yaml",
                        {"radiation": {"enabled": True, "use_hmatrix": True}})
    assert rad == base, f"radiation leaked onto the command line: {rad}"


# --- run_engine ---------------------------------------------------------------
# A stub that emits what the engine emits: two carriage-return progress bars
# 0..100, a "Progress i/N" step counter, an stderr banner, and a result line.
# Both bars matter -- with one shared throttle the second one is silent until it
# finishes, which is the phase a reader is actually waiting on.

RUN_STUB = '''\
import sys, json
from pathlib import Path
if "--help" in sys.argv:
    print("  --config FILE\\n  --output DIR")
    raise SystemExit(0)
cfg = sys.argv[sys.argv.index("--config") + 1]
out = None
if "--output" in sys.argv:
    out = sys.argv[sys.argv.index("--output") + 1]
else:
    # The engine parses its own YAML in C++. This stub reads the one key it
    # needs out of the fixture config by hand: PyYAML is a declared
    # dependency of the package under test, not of whatever interpreter is
    # running the tests, and importing it here made a missing library look
    # like an engine that exited 1. Absent key is fatal, never a default.
    for line in Path(cfg).read_text().splitlines():
        if line.strip().startswith("directory:"):
            out = line.split(":", 1)[1].strip()
    if out is None:
        sys.stderr.write("FATAL [stub]: no output.directory in %s\\n" % cfg)
        raise SystemExit(2)
Path(out).mkdir(parents=True, exist_ok=True)
(Path(out) / "argv.json").write_text(json.dumps(sys.argv[1:]))
sys.stderr.write("[machine-profile] using CONSERVATIVE defaults\\n")
for label in ("Calculating illumination", "SIMD Batch thermal"):
    for pct in range(0, 101):
        sys.stdout.write("\\r%s [==] %d%%" % (label, pct))
    sys.stdout.write("\\n")
for i in range(1, 9):
    sys.stdout.write("\\r  Progress %d/8" % i)
sys.stdout.write("\\n")
print("Simulation completed in 0.010 seconds (10 ms)")
raise SystemExit(int(sys.argv[sys.argv.index("--rc") + 1]) if "--rc" in sys.argv else 0)
'''


def _require_pyyaml(what):
    """Fail naming the missing package, not the symptom it produces.

    ``run_engine`` reads ``output.directory`` out of the config with PyYAML.
    Without it the run still happens, a warning goes to stderr and
    ``output_dir`` is ``None`` -- so an assertion on ``output_dir`` reports an
    unsatisfied dependency as though the code were wrong.
    """
    try:
        import yaml                                          # noqa: F401
    except ImportError:
        raise AssertionError(
            f"{what} needs PyYAML, which ptm declares as a runtime dependency "
            f"(pyproject.toml) and {sys.executable} does not have. Install the "
            f"package the way CI does -- `python3 -m pip install -e .` from the "
            f"repository root -- or run these tests with an interpreter that "
            f"already has PyYAML.")


def _check_runnable(exe):
    """Prove the stub can be executed at all, and say why when it cannot.

    ``run_engine`` takes a path and execs it, so the exec bit and an
    exec-permitted directory are irreducible here. When either is missing the
    failure otherwise lands much later as an exit status on an unrelated
    assertion, which is unreadable; it is diagnosed at the source instead.
    """
    try:
        proc = subprocess.run([str(exe), "--help"], capture_output=True,
                              text=True, timeout=60)
    except OSError as exc:
        raise AssertionError(
            f"the stub engine {exe} could not be executed ({exc}); its "
            f"directory must permit execution and the file must keep its exec "
            f"bit") from exc
    if proc.returncode != 0 or "--config" not in proc.stdout:
        raise AssertionError(
            f"the stub engine ran but did not behave: exit {proc.returncode}, "
            f"stdout {proc.stdout!r}, stderr {proc.stderr!r}")


def _stub_engine(tmp):
    """A stand-in engine on disk, with no interpreter path in its shebang.

    ``run_engine`` takes one path to one executable file, so the stub must be
    executable -- but nothing requires the *interpreter* to be named on the
    ``#!`` line, and naming it there is what made this test report the machine
    instead of the code. A shebang is parsed by the kernel: it is truncated
    past a fixed length, it does not quote (a space in the path splits it),
    and it may not point at another script. An interpreter path that is long,
    spaced, or itself a wrapper therefore fails as an exec error or as a
    "command not found" exit status, which surfaces here as "the engine exited
    127" -- a green-or-red verdict decided by where Python is installed.

    Passing the interpreter to ``/bin/sh`` as a quoted argument has none of
    those limits: ``/bin/sh`` is POSIX-mandated at a fixed short path, and the
    interpreter is resolved by the shell rather than the kernel.
    """
    payload = tmp / "stub_engine.py"
    payload.write_text(RUN_STUB)
    exe = tmp / "ptm_thermal_sim"
    exe.write_text("#!/bin/sh\nexec {} {} \"$@\"\n".format(
        shlex.quote(sys.executable), shlex.quote(str(payload))))
    exe.chmod(0o755)
    _check_runnable(exe)
    return exe


def _stub_config(tmp, outdir="out/run_a"):
    cfg = tmp / "cfg.yaml"
    cfg.write_text("simulation:\n  body: Moon\n"
                   f"output:\n  directory: {outdir}\n")
    return cfg


def _capture(fn):
    """Run ``fn`` with stdout captured; return (value, printed text)."""
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        value = fn()
    return value, buf.getvalue()


def test_run_engine_quiet_returns_everything_and_prints_nothing():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        done, printed = _capture(lambda: run_engine(
            _stub_config(tmp), exe=_stub_engine(tmp), cwd=tmp))
        assert isinstance(done, CompletedRun) and done.ok, done
        assert printed == "", f"echo=False printed: {printed!r}"
        assert done.returncode == 0 and done.seconds > 0
        assert "Simulation completed in" in done.text
        # Every progress readout is kept in .lines even when nothing is echoed.
        assert sum("%" in ln for ln in done.lines) == 202, len(done.lines)
        # output.directory comes from the config, which run_engine reads with
        # PyYAML; say so rather than asserting into a None.
        _require_pyyaml("resolving output.directory from the config")
        assert done.output_dir == (tmp / "out" / "run_a").resolve()
        assert done.matching("Simulation completed in"), "matching() found nothing"


def test_run_engine_echo_throttles_both_progress_bars():
    import re as _re
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _, printed = _capture(lambda: run_engine(
            _stub_config(tmp), exe=_stub_engine(tmp), cwd=tmp, echo=True,
            progress_every=25.0))
        percents = _re.findall(r"\] (\d+)%", printed)
        assert percents == ["0", "25", "50", "75", "100"] * 2, percents
        steps = _re.findall(r"Progress (\d+)/8", printed)
        assert steps == ["1", "3", "5", "7", "8"], steps
        assert "Simulation completed in" in printed
        assert "CONSERVATIVE" in printed, "stderr banners must still be echoed"


def test_run_engine_show_prints_only_the_asked_for_lines():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _, printed = _capture(lambda: run_engine(
            _stub_config(tmp), exe=_stub_engine(tmp), cwd=tmp,
            show=("Simulation completed in",)))
        assert printed.strip().startswith("Simulation completed in"), printed
        assert "%" not in printed, printed


def test_run_engine_output_override_and_results_handoff():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        done, _ = _capture(lambda: run_engine(
            _stub_config(tmp), exe=_stub_engine(tmp), cwd=tmp,
            extra_args=["--output", "elsewhere/run_b"]))
        assert done.output_dir == (tmp / "elsewhere" / "run_b").resolve(), \
            done.output_dir
        assert done.output_dir.is_dir()
        # The stub writes no mesh.ply/metadata.txt, so Results must refuse --
        # and refuse for the right reason, not with an AttributeError.
        try:
            done.results()
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("results() accepted a directory with no run in it")


def test_run_engine_raises_on_failure_and_check_false_does_not():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        exe, cfg = _stub_engine(tmp), _stub_config(tmp)
        try:
            _capture(lambda: run_engine(cfg, exe=exe, cwd=tmp,
                                        extra_args=["--rc", "4"]))
        except RuntimeError as exc:
            assert "exited with code 4" in str(exc), exc
        else:
            raise AssertionError("a non-zero exit did not raise")
        done, _ = _capture(lambda: run_engine(cfg, exe=exe, cwd=tmp,
                                              extra_args=["--rc", "4"],
                                              check=False))
        assert done.returncode == 4 and not done.ok


def test_binary_provenance_identifies_a_binary_with_no_run():
    """The build-side question the run directory cannot answer.

    Results.provenance() reads what the engine wrote about itself, which is
    complete -- but only after a run exists. This is the one asked first.
    """
    import hashlib
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        exe = _stub_engine(tmp)
        prov = binary_provenance(exe=exe, repo_root=tmp)
        assert prov.md5 == hashlib.md5(exe.read_bytes()).hexdigest()
        assert prov.size_bytes == exe.stat().st_size
        assert prov.mtime_utc.endswith("Z") and prov.mtime_utc[4] == "-"
        assert prov.help_ok is True and prov.n_flags == 2, prov.as_dict()
        text = str(prov)
        assert "md5" in text and prov.md5 in text
        # Not a git checkout: reported, not silently dropped.
        assert prov.git_sha is None
        assert any("git commit" in item for item in prov.unavailable), \
            prov.unavailable


def test_binary_provenance_flags_a_binary_that_cannot_answer_help():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        broken = tmp / "ptm_thermal_sim"
        broken.write_text("#!/bin/sh\nexit 0\n")     # exits 0, prints nothing
        broken.chmod(0o755)
        prov = binary_provenance(exe=broken, repo_root=tmp)
        assert prov.help_ok is False
        assert any("build is broken" in item for item in prov.unavailable), \
            prov.unavailable


def test_resolve_engine_names_the_build_command_when_absent():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        try:
            resolve_engine(cwd=td, name="ptm_thermal_sim")
        except FileNotFoundError as exc:
            assert "cmake --build" in str(exc), f"no build command offered: {exc}"
        else:
            raise AssertionError("a missing engine did not raise")


# ------------------------------------------------------------------ the log
# A GUI-launched run left NO engine stdout or stderr anywhere on disk. On
# 2026-09-02 a lunar polar flux-cube run had to be verified without its
# `[PTM_HMATRIX_VF_REPLACE] ROUTE = ...` banner and without its
# coupling-convergence report, because the only place either had ever existed
# was a console widget that the next launch clears.

def _log_lines(path):
    return [l for l in Path(path).read_text().split("\n") if l.strip()]


def test_the_engine_log_holds_every_line_the_child_wrote():
    """Both streams, in the file, alongside the callbacks."""
    with tempfile.TemporaryDirectory() as tmp:
        log = Path(tmp) / "deeper" / "engine.log"
        rc, seen = _run_child(code=0, log_path=log)
        assert rc == 0, rc
        assert log.exists(), "no engine.log was written (and the directory " \
                             "under it was supposed to be created)"
        text = log.read_text()
        for needle in ("Mesh created with 642 vertices",       # stdout
                       "equilibration seed from cold-face",     # stderr
                       "Calculating illumination: 42.5%",
                       "Simulation completed"):
            assert needle in text, f"{needle!r} is not in the log:\n{text}"
        assert "command:" in text and "cwd:" in text, \
            "the log does not say what was run or where"
        assert "engine exit status 0" in text, \
            "the log does not record how the run ended"
        # The console is unchanged: everything in the file is also in a callback.
        assert any("Mesh created" in l for l in seen["info"])
        assert seen["progress"] == [42.5], seen["progress"]


def test_the_log_survives_a_failing_run_and_appends_on_the_next():
    """THE CASE THE LOG EXISTS FOR, and the one that must not truncate.

    A crash is exactly when the record is wanted. And a chunked or resumed run
    writes into the same output directory more than once -- a second launch
    that erased the first one's diagnosis would recreate the gap.
    """
    with tempfile.TemporaryDirectory() as tmp:
        log = Path(tmp) / "engine.log"
        rc, _ = _run_child(
            code=3,
            tail='sys.stderr.write("FATAL: Failed to load mesh\\n")',
            log_path=log)
        assert rc == 3, rc
        first = log.read_text()
        assert "FATAL: Failed to load mesh" in first, first
        assert "engine exit status 3" in first, first

        rc2, _ = _run_child(code=0, log_path=log)
        assert rc2 == 0
        second = log.read_text()
        assert "FATAL: Failed to load mesh" in second, \
            "the second run TRUNCATED the first run's log"
        assert second.count("===== engine start") == 2, \
            f"expected two start banners, got {second.count('===== engine start')}"


def test_no_log_path_writes_no_file_and_changes_nothing():
    """DO-NO-HARM: every existing caller passes nothing and is unaffected."""
    with tempfile.TemporaryDirectory() as tmp:
        before = sorted(Path(tmp).iterdir())
        rc, seen = _run_child(code=0)
        assert rc == 0
        assert sorted(Path(tmp).iterdir()) == before
        assert not seen["error"], seen["error"]
        assert seen["progress"] == [42.5]


def test_a_log_that_cannot_be_opened_warns_and_the_run_continues():
    """Failing a run over its own diagnostics would be worse than the gap."""
    with tempfile.TemporaryDirectory() as tmp:
        blocker = Path(tmp) / "not_a_directory"
        blocker.write_text("this is a file")
        rc, seen = _run_child(code=0, log_path=blocker / "engine.log")
        assert rc == 0, f"the run must still complete (exit {rc})"
        assert any("no engine log" in w for w in seen["warning"]), (
            f"the run silently produced no log; warnings were "
            f"{seen['warning']}")


TESTS = [
    test_classification_of_real_engine_lines,
    test_progress_extraction,
    test_stderr_banners_are_not_errors,
    test_failure_is_reported_and_exit_code_propagates,
    test_stop_terminates_the_process_group,
    test_output_left_in_the_pipe_at_exit_is_still_split_into_lines,
    test_build_command_shape,
    test_run_engine_quiet_returns_everything_and_prints_nothing,
    test_run_engine_echo_throttles_both_progress_bars,
    test_run_engine_show_prints_only_the_asked_for_lines,
    test_run_engine_output_override_and_results_handoff,
    test_run_engine_raises_on_failure_and_check_false_does_not,
    test_binary_provenance_identifies_a_binary_with_no_run,
    test_binary_provenance_flags_a_binary_that_cannot_answer_help,
    test_resolve_engine_names_the_build_command_when_absent,
    test_the_engine_log_holds_every_line_the_child_wrote,
    test_the_log_survives_a_failing_run_and_appends_on_the_next,
    test_no_log_path_writes_no_file_and_changes_nothing,
    test_a_log_that_cannot_be_opened_warns_and_the_run_continues,
]


def main():
    failures = 0
    for test in TESTS:
        try:
            test()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {test.__name__}\n     {exc}")
        except Exception as exc:                       # noqa: BLE001
            failures += 1
            print(f"ERROR {test.__name__}: {type(exc).__name__}: {exc}")
        else:
            print(f"PASS {test.__name__}")
    print(f"\n[test_engine_process] {len(TESTS) - failures}/{len(TESTS)} passed")
    if failures:
        print("[test_engine_process] FAIL")
        return 1
    print("[test_engine_process] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
