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

from ptm.run import (EngineProcess, build_command,      # noqa: E402
                     classify_severity, extract_progress)

# Lines the engine really emits, and the severity each must be given. The
# stderr banners are the point: a successful PTM3D run prints all of these.
CLASSIFICATION_CASES = [
    ("[eq-seed] equilibration seed from cold-face analytic", "info"),
    ("[machine-profile] no host profile; using CONSERVATIVE defaults", "warning"),
    ("[GPU Thermal] dispatch self-chunk ceiling 8192 timesteps", "info"),
    ("Calculating illumination: 42.5%", "info"),
    ("Simulation completed in 5.030 seconds (5030 ms)", "info"),
    ("FATAL: Failed to load mesh from crater.ply", "error"),
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


TESTS = [
    test_classification_of_real_engine_lines,
    test_progress_extraction,
    test_stderr_banners_are_not_errors,
    test_failure_is_reported_and_exit_code_propagates,
    test_stop_terminates_the_process_group,
    test_build_command_shape,
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
