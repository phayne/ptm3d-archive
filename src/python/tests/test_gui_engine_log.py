#!/usr/bin/env python3
"""A GUI-launched run must leave the engine's own output on disk.

Until 2026-09-02 it did not. Verifying the MP026 flux-cube run that day meant
reconstructing which view-factor matrix had been loaded from the product's own
binary header, because the engine's ``[PTM_HMATRIX_VF_REPLACE] ROUTE = ...``
banner and its whole coupling-convergence report had only ever existed in a
console widget -- which the next launch clears. Nothing was on disk anywhere:
not in the run directory, not in the home directory, not in the repository.

Two things are checked, and they are different questions:

* the thread that starts the engine writes ``<output dir>/engine.log`` and
  that file holds what the child wrote (measured, with a stand-in engine);
* EVERY launcher passes a log path. There are two today -- the main window and
  the dashboard's thermal module -- and the 2026-08-31 flux-cube arc cost a
  round precisely because a fix landed in one of them. This arm reads the
  source with `ast` and will fail on a third launcher as well.

Run directly::

    python3 src/python/tests/test_gui_engine_log.py
"""

from __future__ import annotations

import ast
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "python"))
sys.path.insert(0, str(REPO_ROOT / "gui"))

os.environ.setdefault("PTM_GUI_DISABLE_3D", "1")

_SCRATCH = None
_APP = None


class Skip(Exception):
    """A check that cannot run here, as distinct from one that failed."""


def scratch():
    global _SCRATCH
    if _SCRATCH is None:
        _SCRATCH = tempfile.mkdtemp(prefix="ptm_engine_log_")
    return Path(_SCRATCH)


STUB_ENGINE = """#!/usr/bin/env python3
import sys
print("Mesh created with 642 vertices and 1280 faces", flush=True)
sys.stderr.write("[PTM_HMATRIX_VF_REPLACE] ROUTE = GPU (radiation.use_gpu)\\n")
sys.stderr.flush()
print("Calculating illumination: 42.5%", flush=True)
print("Simulation completed in 0.010 seconds (10 ms)", flush=True)
sys.exit(0)
"""

#: The line the MP026 verification could not find anywhere on disk. It is the
#: whole point of the file, so it is what the arm looks for.
ROUTE_LINE = "[PTM_HMATRIX_VF_REPLACE] ROUTE ="


def _qt():
    global _APP
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        import PyQt6
    except ImportError as exc:
        raise Skip(f"PyQt6 is unavailable here ({exc})")
    plug = Path(PyQt6.__file__).parent / "Qt6" / "plugins"
    os.environ.setdefault("QT_PLUGIN_PATH", str(plug))
    os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(plug / "platforms"))
    from PyQt6.QtWidgets import QApplication
    if QApplication.instance() is None:
        _APP = QApplication([])
    return QApplication.instance()


def _stub_engine():
    exe = scratch() / "stub_engine.py"
    exe.write_text(STUB_ENGINE)
    exe.chmod(0o755)
    return exe


def _thread_for(out_dir, log_path=None):
    """A SimulationThread wired to the stand-in engine, run synchronously."""
    _qt()
    try:
        from gui_core import simulation_runner as sr
    except ImportError as exc:
        raise Skip(f"the GUI stack is unavailable here ({exc})")
    exe = _stub_engine()
    cfg = scratch() / "config.yaml"
    cfg.write_text("simulation:\n  name: stub\noutput:\n  directory: %s\n"
                   % out_dir)
    config_dict = {"output": {"directory": str(out_dir)},
                   "simulation": {"name": "stub"}}
    saved = sr.SimulationRunner.find_ptm_executable
    sr.SimulationRunner.find_ptm_executable = staticmethod(lambda: str(exe))
    try:
        thread = sr.SimulationThread(str(cfg), config_dict=config_dict,
                                     log_path=log_path)
        seen = {"info": [], "warning": [], "error": [], "rc": []}
        thread.output_ready.connect(seen["info"].append)
        thread.warning_ready.connect(seen["warning"].append)
        thread.error_ready.connect(seen["error"].append)
        thread.finished.connect(seen["rc"].append)
        thread.run()                 # synchronous: no event loop needed
        return seen
    finally:
        sr.SimulationRunner.find_ptm_executable = saved


def test_a_launched_run_writes_the_engine_log_beside_its_output():
    """THE ARM. `<output.directory>/engine.log`, derived from the config."""
    from gui_core.simulation_runner import ENGINE_LOG_NAME
    out = scratch() / "run_derived"
    seen = _thread_for(out)
    log = out / ENGINE_LOG_NAME
    assert log.exists(), (
        f"no {ENGINE_LOG_NAME} beside the output; the run left nothing on "
        f"disk. Console lines were {seen['info']}")
    text = log.read_text()
    assert ROUTE_LINE in text, (
        f"the one line the MP026 verification went looking for is not in the "
        f"log:\n{text}")
    assert "Mesh created with 642 vertices" in text, text
    assert "engine exit status 0" in text, text
    # And the console is unchanged.
    assert any("Mesh created" in line for line in seen["info"]), seen


def test_an_explicit_log_path_is_used_as_given():
    """What both launchers actually pass, since both already know the dir."""
    out = scratch() / "run_explicit"
    out.mkdir(parents=True, exist_ok=True)
    target = out / "engine.log"
    _thread_for(scratch() / "somewhere_else", log_path=target)
    assert target.exists(), f"the explicit log path {target} was ignored"
    assert ROUTE_LINE in target.read_text()


def test_a_run_that_cannot_be_filed_says_so_instead_of_going_quiet():
    """No output.directory and no explicit path: a warning, not silence."""
    _qt()
    try:
        from gui_core import simulation_runner as sr
    except ImportError as exc:
        raise Skip(f"the GUI stack is unavailable here ({exc})")
    exe = _stub_engine()
    cfg = scratch() / "config_nodir.yaml"
    cfg.write_text("simulation:\n  name: stub\n")
    saved = sr.SimulationRunner.find_ptm_executable
    sr.SimulationRunner.find_ptm_executable = staticmethod(lambda: str(exe))
    try:
        thread = sr.SimulationThread(str(cfg), config_dict={"simulation": {}})
        warnings = []
        thread.warning_ready.connect(warnings.append)
        thread.run()
    finally:
        sr.SimulationRunner.find_ptm_executable = saved
    assert any("engine.log" in w for w in warnings), (
        f"a run with nowhere to file its output went ahead without a word; "
        f"warnings were {warnings}")


def test_every_launcher_passes_a_log_path():
    """⛔ BOTH LAUNCHERS. Not one; the 2026-08-31 arc paid for this lesson.

    Read with `ast` rather than grep so that a call spread over several lines
    -- which both of them are -- is seen as one call with its keywords.
    """
    calls = []
    for path in sorted((REPO_ROOT / "gui").rglob("*.py")):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:                                  # pragma: no cover
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = fn.id if isinstance(fn, ast.Name) else (
                fn.attr if isinstance(fn, ast.Attribute) else None)
            if name == "SimulationThread":
                calls.append((path, node))
    assert len(calls) >= 2, (
        f"expected at least the two known launchers to construct a "
        f"SimulationThread; found {len(calls)}: "
        f"{[str(p.relative_to(REPO_ROOT)) for p, _ in calls]}")
    missing = [str(p.relative_to(REPO_ROOT)) for p, n in calls
               if "log_path" not in {k.arg for k in n.keywords}]
    assert not missing, (
        f"these launchers start the engine without telling it where to file "
        f"its output: {missing}. A run launched from one of them leaves no "
        f"engine.log, which is the gap this change closes.")


TESTS = [
    test_a_launched_run_writes_the_engine_log_beside_its_output,
    test_an_explicit_log_path_is_used_as_given,
    test_a_run_that_cannot_be_filed_says_so_instead_of_going_quiet,
    test_every_launcher_passes_a_log_path,
]


def main():
    failures = skipped = 0
    for test in TESTS:
        try:
            test()
        except Skip as exc:
            skipped += 1
            print(f"SKIP {test.__name__}: {exc}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {test.__name__}\n     {exc}")
        except Exception as exc:                       # noqa: BLE001
            failures += 1
            print(f"ERROR {test.__name__}: {type(exc).__name__}: {exc}")
        else:
            print(f"PASS {test.__name__}")
    total = len(TESTS)
    print(f"\n[test_gui_engine_log] {total - failures - skipped}/{total} "
          f"passed, {skipped} skipped")
    if _SCRATCH:
        shutil.rmtree(_SCRATCH, ignore_errors=True)
    if skipped == total:
        print("[test_gui_engine_log] SKIPPED ENTIRELY -- proved NOTHING here.")
    if failures:
        print("[test_gui_engine_log] FAIL")
        return 1
    print("[test_gui_engine_log] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
