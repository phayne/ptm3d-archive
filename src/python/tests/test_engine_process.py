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

import sys
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


def _run_child(code=0, tail="pass"):
    seen = {"info": [], "warning": [], "error": [], "progress": [], "beat": []}
    proc = EngineProcess(
        [sys.executable, "-c", CHILD.format(code=code, tail=tail)],
        on_output=seen["info"].append,
        on_warning=seen["warning"].append,
        on_error=seen["error"].append,
        on_progress=seen["progress"].append,
        on_heartbeat=lambda m, s: seen["beat"].append((m, s)),
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
    import yaml
    out = yaml.safe_load(Path(cfg).read_text())["output"]["directory"]
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


def _stub_engine(tmp):
    exe = tmp / "ptm_thermal_sim"
    exe.write_text(f"#!{sys.executable}\n" + RUN_STUB)
    exe.chmod(0o755)
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


TESTS = [
    test_classification_of_real_engine_lines,
    test_progress_extraction,
    test_stderr_banners_are_not_errors,
    test_failure_is_reported_and_exit_code_propagates,
    test_stop_terminates_the_process_group,
    test_build_command_shape,
    test_run_engine_quiet_returns_everything_and_prints_nothing,
    test_run_engine_echo_throttles_both_progress_bars,
    test_run_engine_show_prints_only_the_asked_for_lines,
    test_run_engine_output_override_and_results_handoff,
    test_run_engine_raises_on_failure_and_check_false_does_not,
    test_binary_provenance_identifies_a_binary_with_no_run,
    test_binary_provenance_flags_a_binary_that_cannot_answer_help,
    test_resolve_engine_names_the_build_command_when_absent,
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
