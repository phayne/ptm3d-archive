#!/usr/bin/env python3
"""Item 2.4 / DP-6: the GUI reports the ENGINE's refusals, in the engine's words.

``ConfigManager.validate`` is a second, weaker copy of the engine's rules, and
the two disagreed within this very session: the first draft of the
``eq_indirect_source`` check invented a valid set (``auto|none|file|internal``)
that would have refused ``converged`` -- the one non-default value the engine
accepts. A second copy of a rule is a second place for it to be wrong.

So `Simulation -> Check Configuration` hands the config to
``ptm_thermal_sim --validate-config`` and shows what comes back. The property
that must never break: an engine that **cannot** be asked is reported as NOT
CHECKED, never as checked-and-fine. Three of the four states this returns are
some flavor of "I could not do that", and every one of them has to be visible.

Run directly::

    python3 src/python/tests/test_gui_engine_validator.py
"""

from __future__ import annotations

import os
import stat
import sys
import tempfile
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "python"))
sys.path.insert(0, str(REPO_ROOT / "gui"))


class Skip(Exception):
    pass


def _preflight():
    try:
        from gui_core import preflight
    except ImportError as exc:                     # pragma: no cover
        raise Skip(f"the GUI's preflight module is unavailable here ({exc})")
    return preflight


def _manager():
    from gui_core.config_manager import ConfigManager
    return ConfigManager()


def _engine():
    from ptm import paths
    exe = paths.find_engine_executable("ptm_thermal_sim",
                                       search_roots=[REPO_ROOT])
    if exe is None:
        raise Skip("no ptm_thermal_sim in build/bin; build it with: "
                   "cmake --build build --target ptm_thermal_sim")
    return Path(exe).resolve()


def _stub(tmp, body):
    """A fake engine, so the unavailable/refused paths are testable with no build."""
    path = Path(tmp) / "fake_engine"
    path.write_text("#!/usr/bin/env python3\n" + textwrap.dedent(body))
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


# --------------------------------------------------------------------------

def test_a_good_config_is_accepted_by_the_real_engine():
    preflight = _preflight()
    exe = _engine()
    manager = _manager()
    manager.set_value("simulation.timesteps", 4)
    manager.set_value("simulation.duration_days", 1.0)
    manager.set_value("mesh.resolution", 2)
    status, text = preflight.engine_validate_config(
        manager.get_config(), exe, REPO_ROOT)
    assert status == "accepted", f"{status}: {text[-1500:]}"
    assert "[validate-config]" in text


def test_the_engines_refusal_reaches_the_user_verbatim():
    preflight = _preflight()
    exe = _engine()
    manager = _manager()
    manager.set_value("simulation.timesteps", 4)
    manager.set_value("mesh.resolution", 2)
    # A pair the ENGINE refuses at startup. Set through the config directly so
    # this measures the engine's rule, not the GUI's copy of it.
    manager.set_value("radiation.beaming", True)
    manager.set_value("radiation.rms_slope_deg", 0.0)
    status, text = preflight.engine_validate_config(
        manager.get_config(), exe, REPO_ROOT)
    assert status == "refused", f"{status}: {text[-1500:]}"
    assert "radiation.beaming = true requires" in text, (
        f"the engine's own wording did not survive: {text[-600:]}")


def test_a_disagreement_between_the_two_checkers_is_visible():
    """The engine is the authority; the GUI must not paper over a mismatch.

    `eq_indirect_source: converged` is accepted by the engine. If the GUI's own
    copy of that rule ever refuses it again, this fails -- which is the point.
    """
    preflight = _preflight()
    exe = _engine()
    manager = _manager()
    manager.set_value("simulation.timesteps", 4)
    manager.set_value("mesh.resolution", 2)
    manager.set_value("thermal.eq_indirect_source", "converged")
    gui_valid, gui_messages = manager.validate()
    status, text = preflight.engine_validate_config(
        manager.get_config(), exe, REPO_ROOT)
    assert status == "accepted", f"the engine refused it: {text[-800:]}"
    assert gui_valid, (
        f"the GUI refuses a value the engine accepts: {gui_messages}. That is "
        f"the exact failure mode item 2.4 exists to remove.")


def test_no_binary_is_reported_as_not_checked():
    preflight = _preflight()
    manager = _manager()
    for missing in (None, "", "/nonexistent/ptm_thermal_sim"):
        status, text = preflight.engine_validate_config(
            manager.get_config(), missing, REPO_ROOT)
        assert status == "unavailable", (
            f"a missing engine ({missing!r}) reported {status!r}; treating "
            f"'could not ask' as 'fine' is the silence this stage removes")
        assert text.strip()


def test_an_engine_without_the_flag_is_reported_as_not_checked():
    """rc 0 from a binary that ignored the flag is NOT an acceptance."""
    preflight = _preflight()
    manager = _manager()
    with tempfile.TemporaryDirectory() as tmp:
        exe = _stub(tmp, """
            import sys
            print("PTM3D 1.0")
            sys.exit(0)
        """)
        status, text = preflight.engine_validate_config(
            manager.get_config(), exe, REPO_ROOT)
        assert status == "unavailable", (
            f"an engine that silently ignored --validate-config reported "
            f"{status!r}")
        assert "--validate-config" in text


def test_a_hung_engine_times_out_rather_than_hanging_the_gui():
    preflight = _preflight()
    manager = _manager()
    original = preflight.VALIDATE_TIMEOUT_S
    preflight.VALIDATE_TIMEOUT_S = 1.0
    try:
        with tempfile.TemporaryDirectory() as tmp:
            exe = _stub(tmp, """
                import time
                time.sleep(120)
            """)
            status, text = preflight.engine_validate_config(
                manager.get_config(), exe, REPO_ROOT)
            assert status == "unavailable", status
            assert "did not answer" in text, text
    finally:
        preflight.VALIDATE_TIMEOUT_S = original


def test_the_check_leaves_no_temporary_file_behind():
    preflight = _preflight()
    exe = _engine()
    manager = _manager()
    manager.set_value("simulation.timesteps", 4)
    manager.set_value("mesh.resolution", 2)
    before = set(Path(tempfile.gettempdir()).glob("ptm3d_validate_*"))
    preflight.engine_validate_config(manager.get_config(), exe, REPO_ROOT)
    after = set(Path(tempfile.gettempdir()).glob("ptm3d_validate_*"))
    assert after <= before, f"left behind: {sorted(after - before)}"


def test_the_check_does_not_write_into_the_output_directory():
    """Checking a finished run's config must not disturb its record."""
    preflight = _preflight()
    exe = _engine()
    with tempfile.TemporaryDirectory() as tmp:
        outdir = Path(tmp) / "finished_run"
        outdir.mkdir()
        (outdir / "heartbeat.txt").write_text("PRIOR RUN MARKER\n")
        manager = _manager()
        manager.set_value("simulation.timesteps", 4)
        manager.set_value("mesh.resolution", 2)
        manager.set_value("output.directory", str(outdir))
        before = sorted(p.name for p in outdir.iterdir())
        preflight.engine_validate_config(manager.get_config(), exe, REPO_ROOT)
        assert sorted(p.name for p in outdir.iterdir()) == before
        assert (outdir / "heartbeat.txt").read_text() == "PRIOR RUN MARKER\n"


def test_the_check_uses_the_same_environment_the_run_will():
    """Validating under a different environment answers a different question."""
    preflight = _preflight()
    manager = _manager()
    with tempfile.TemporaryDirectory() as tmp:
        marker = Path(tmp) / "env.txt"
        exe = _stub(tmp, f"""
            import os, sys
            with open({str(marker)!r}, "w") as fh:
                fh.write(os.environ.get("PTM_ILLUM_RISE_EPS", "(unset)"))
            print("[validate-config] ACCEPTED")
            sys.exit(0)
        """)
        status, _text = preflight.engine_validate_config(
            manager.get_config(), exe, REPO_ROOT)
        assert status == "accepted", status
        assert marker.read_text() == "1", (
            f"the check ran without the engine environment the launch applies "
            f"({marker.read_text()!r}); PTM_ILLUM_RISE_EPS gates the validated "
            f"grazing-leak fix")


TESTS = [
    test_a_good_config_is_accepted_by_the_real_engine,
    test_the_engines_refusal_reaches_the_user_verbatim,
    test_a_disagreement_between_the_two_checkers_is_visible,
    test_no_binary_is_reported_as_not_checked,
    test_an_engine_without_the_flag_is_reported_as_not_checked,
    test_a_hung_engine_times_out_rather_than_hanging_the_gui,
    test_the_check_leaves_no_temporary_file_behind,
    test_the_check_does_not_write_into_the_output_directory,
    test_the_check_uses_the_same_environment_the_run_will,
]


def main():
    failures = 0
    skipped = 0
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
    print(f"\n[test_gui_engine_validator] {total - failures - skipped}/{total} "
          f"passed, {skipped} skipped")
    if skipped == total:
        print("[test_gui_engine_validator] SKIPPED ENTIRELY -- proved NOTHING.")
    if failures:
        print("[test_gui_engine_validator] FAIL")
        return 1
    print("[test_gui_engine_validator] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
