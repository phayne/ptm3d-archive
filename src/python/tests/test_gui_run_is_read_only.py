#!/usr/bin/env python3
"""Pressing Run must not rewrite the config file the user opened.

Until 2026-08-25 ``PTM3DMainWindow.run_simulation`` called ``save_project()``
before launching whenever a project file was open: no prompt, no diff, no
backup. That is ``yaml.dump`` of the merged dictionary over the author's own
file. Measured on the two shipped examples it erased 10 and 19 comments, added
33 and 42 keys, and silently changed ``illumination.albedo_maps_directory`` to
``''``. A config is a scientific record, and a run that quietly rewrites its own
inputs cannot be audited afterward.

Two checks, and neither needs a display:

1. **Source-level.** ``run_simulation`` contains no call that writes over the
   project file. Parsed from the AST rather than grepped, so a rename or a
   wrapper cannot slip past. This is the exact line that caused the defect, so
   it is the exact line worth pinning. ``PTM3DMainWindow`` cannot be built
   headless (it constructs a ``VisualizationPanel``, which segfaults at the
   pyvistaqt/VTK import), so the running object is out of reach here.
2. **Behavioral.** Loading a shipped example and writing the engine-facing run
   YAML elsewhere leaves the example byte-for-byte unchanged. Needs PyQt6;
   skipped without it. Note what this one is and is not: it pins the invariant
   the launch path now relies on, and it passed on the broken tree too, because
   the write came from ``run_simulation`` and not from ``ConfigManager``.
   Check 1 is the discriminator; check 2 is do-no-harm.

Run directly::

    python3 src/python/tests/test_gui_run_is_read_only.py
"""

from __future__ import annotations

import ast
import hashlib
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "gui"))

GUI_MAIN = REPO_ROOT / "gui" / "ptm3d_gui.py"

#: Attribute calls that write over ``current_project_file``. ``save_run_yaml``
#: is deliberately absent: writing the engine-facing YAML beside the output
#: directory is what Run is supposed to do.
FORBIDDEN_IN_RUN = ("save_project", "save_project_to", "save_to_yaml")

#: Shipped configs to round-trip. Chosen because the 2026-08-25 audit measured
#: the damage on exactly these two.
EXAMPLES = ("tutorial04_crater80N_coupled.yaml", "quickstart_moon_diurnal.yaml")


class Skipped(Exception):
    """A check that cannot run here, as distinct from one that failed."""


def _skip(reason):
    try:
        import pytest
    except ImportError:
        raise Skipped(reason) from None
    pytest.skip(reason)


def _run_simulation_node():
    tree = ast.parse(GUI_MAIN.read_text())
    for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
        for fn in cls.body:
            if isinstance(fn, ast.FunctionDef) and fn.name == "run_simulation":
                return cls.name, fn
    raise AssertionError(f"no run_simulation method found in {GUI_MAIN}")


# --------------------------------------------------------------------------

def test_run_simulation_never_writes_the_project_file():
    """The launch path calls nothing that overwrites the opened file."""
    cls_name, fn = _run_simulation_node()
    offenders = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name in FORBIDDEN_IN_RUN:
            offenders.append(f"{name}() at {GUI_MAIN.name}:{node.lineno}")
    assert not offenders, (
        f"{cls_name}.run_simulation writes over the user's project file: "
        + ", ".join(offenders)
        + ". Run writes the engine-facing YAML beside the output directory "
          "(save_run_yaml); the project file changes only on Save.")


def test_run_simulation_writes_a_run_yaml_beside_the_output():
    """It does still write the config the engine reads -- just elsewhere."""
    _cls_name, fn = _run_simulation_node()
    called = {node.func.attr for node in ast.walk(fn)
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    assert "save_run_yaml" in called, (
        "run_simulation no longer writes an engine-facing run YAML at all; "
        "read-only open is not the same as launching with no config")


def test_the_run_yaml_is_written_only_after_the_preflight_gate():
    """A blocked or canceled launch must not leave a config.yaml behind.

    The tutorial tells the user this file "is the exact configuration that ran",
    which is false if it is written before the gate: a dataless-file block or a
    cancel overwrites the previous run's record with a run that never happened.
    """
    _cls, fn = _run_simulation_node()
    save = confirm = None
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.attr if isinstance(node.func, ast.Attribute) \
            else getattr(node.func, "id", "")
        if name == "save_run_yaml":
            save = node.lineno
        elif name == "confirm_preflight":
            confirm = node.lineno
    assert save is not None and confirm is not None, (save, confirm)
    assert save > confirm, (
        f"save_run_yaml runs at {GUI_MAIN.name}:{save}, before the "
        f"confirm_preflight gate at :{confirm}")


def test_the_console_is_not_cleared_after_the_preflight_output():
    """The physics-change banner has to survive to the screen.

    The environment fix is disclosed through a `[preflight] WARNING:` console
    line. A `console_panel.clear()` after the preflight loop erased it, along
    with the cloud-path and view-factor advisories, the instant the run started.
    """
    _cls, fn = _run_simulation_node()
    clears = [n.lineno for n in ast.walk(fn)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
              and n.func.attr == "clear"]
    preflight = [n.lineno for n in ast.walk(fn)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                 and n.func.id == "run_preflight"]
    assert preflight, "run_simulation no longer runs the pre-flight at all"
    late = [c for c in clears if c > preflight[0]]
    assert not late, (
        f"console_panel.clear() at {GUI_MAIN.name}:{late} runs after "
        f"run_preflight at :{preflight[0]}, erasing every [preflight] line "
        f"including the engine-environment banner")


def test_loading_a_shipped_example_does_not_mutate_it():
    """Load -> write the run YAML elsewhere -> the example is untouched.

    Skipped, not failed, where PyQt6 is missing: ``gui_core`` imports Qt at
    package import.
    """
    try:
        from gui_core.config_manager import ConfigManager
    except ImportError as exc:
        _skip(f"the GUI's config manager is unavailable here ({exc}); "
              f"install PyQt6 to run this check")

    for name in EXAMPLES:
        src = REPO_ROOT / "examples" / name
        assert src.exists(), f"missing shipped example {src}"
        before = hashlib.md5(src.read_bytes()).hexdigest()

        manager = ConfigManager()
        manager.load_from_yaml(str(src))
        with tempfile.TemporaryDirectory() as tmp:
            manager.save_run_yaml(str(Path(tmp) / "config.yaml"), REPO_ROOT)
            assert (Path(tmp) / "config.yaml").exists()

        after = hashlib.md5(src.read_bytes()).hexdigest()
        assert before == after, (
            f"{src} changed during a load-and-run round trip "
            f"(md5 {before} -> {after})")


TESTS = [
    test_run_simulation_never_writes_the_project_file,
    test_run_simulation_writes_a_run_yaml_beside_the_output,
    test_the_run_yaml_is_written_only_after_the_preflight_gate,
    test_the_console_is_not_cleared_after_the_preflight_output,
    test_loading_a_shipped_example_does_not_mutate_it,
]


def main():
    failures = skipped = 0
    for test in TESTS:
        try:
            test()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {test.__name__}\n     {exc}")
        except BaseException as exc:                   # noqa: BLE001
            if type(exc).__name__ != "Skipped":
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                failures += 1
                print(f"ERROR {test.__name__}: {type(exc).__name__}: {exc}")
            else:
                skipped += 1
                print(f"SKIP {test.__name__}\n     {exc}")
        else:
            print(f"PASS {test.__name__}")
    total = len(TESTS)
    tally = f"{total - failures - skipped}/{total} passed"
    if skipped:
        tally += f", {skipped} skipped"
    print(f"\n[test_gui_run_is_read_only] {tally}")
    if failures:
        print("[test_gui_run_is_read_only] FAIL")
        return 1
    print("[test_gui_run_is_read_only] PASS"
          + (f" ({skipped} skipped)" if skipped else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
