#!/usr/bin/env python3
"""The engine keys Stage 1 promised the GUI can write -- checked at the writer.

A widget on a tab proves nothing on its own: what matters is whether the key
reaches the saved YAML with the value the user chose, and comes back on reload.
``ConfigManager._config_for_yaml`` is the writer and ``load_from_yaml`` is the
reader, so these checks go straight at them. Tab construction is covered
separately by the offscreen panel probe, which needs PyVista and therefore
cannot run in the Python-layer CI job.

The sharpest of these is ``thermophysical.H_override``. The engine has read it
since it was added (``config_parser.cpp:759``) and applies it by setting
``planet.H`` directly (``ptm_thermal_sim.cpp:14778``), so it works on a built-in
body -- but ``ConfigManager.ENGINE_OVERRIDE_KEYS`` was
``("albedo", "emissivity", "ks", "kd")``, and the Thermophysical tab
additionally disabled the H row on every built-in body. There was no write path
at all: a colleague could not change the regolith scale height on the Moon from
this GUI.

Needs PyQt6 (``gui_core`` imports Qt at package import); skipped without it.

Run directly::

    python3 src/python/tests/test_gui_stage1_keys.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "python"))
sys.path.insert(0, str(REPO_ROOT / "gui"))

from ptm.engine_defaults import ENGINE_DEFAULTS                   # noqa: E402


class Skipped(Exception):
    """A check that cannot run here, as distinct from one that failed."""


def _skip(reason):
    try:
        import pytest
    except ImportError:
        raise Skipped(reason) from None
    pytest.skip(reason)


def _manager():
    try:
        from gui_core.config_manager import ConfigManager
    except ImportError as exc:
        _skip(f"the GUI's config manager is unavailable here ({exc}); "
              f"install PyQt6 to run this check")
    return ConfigManager()


def _round_trip(manager):
    """Save what this manager would write, load it back, return the new manager."""
    from gui_core.config_manager import ConfigManager
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "round.yaml"
        manager.save_to_yaml(str(path))
        back = ConfigManager()
        back.load_from_yaml(str(path))
        return back


# --------------------------------------------------------------------------

def test_H_override_is_written_on_a_builtin_body():
    """Uncheck 'use default H' on the Moon and H_override lands in the YAML."""
    manager = _manager()
    manager.set_value("simulation.body", "Moon")
    manager.set_value("thermophysical.use_default_H", False)
    manager.set_value("thermophysical.H", 0.09)

    written = manager._config_for_yaml()["thermophysical"]
    assert "H_override" in written, (
        "a built-in body with a custom H writes no H_override; the engine "
        "reads thermophysical.H_override (config_parser.cpp:759) and applies "
        "it by setting planet.H, so the value is simply being dropped")
    assert written["H_override"] == 0.09, written["H_override"]
    assert "H" not in written, (
        "the plain scalar thermophysical.H is also being written; the engine "
        "ignores it on a built-in body, and writing it makes a dead key look "
        "like a live setting")


def test_H_override_survives_a_round_trip():
    manager = _manager()
    manager.set_value("simulation.body", "Moon")
    manager.set_value("thermophysical.use_default_H", False)
    manager.set_value("thermophysical.H", 0.09)

    back = _round_trip(manager)
    assert back.get_value("thermophysical.H") == 0.09
    assert back.get_value("thermophysical.use_default_H") is False


def test_H_is_in_the_engine_override_set():
    """The tuple the writer and the tab both key off."""
    from gui_core.config_manager import ConfigManager
    assert "H" in ConfigManager.ENGINE_OVERRIDE_KEYS, (
        "ENGINE_OVERRIDE_KEYS is the single source for both the write path and "
        "the tab's enable/disable rule; H missing from it is what made the "
        "field unreachable")
    for key in ConfigManager.ENGINE_OVERRIDE_KEYS:
        assert f"thermophysical.{key}_override" in ENGINE_DEFAULTS, (
            f"ENGINE_OVERRIDE_KEYS claims thermophysical.{key}_override exists "
            f"and the engine does not read it")


def test_conductivity_model_is_written_and_round_trips():
    manager = _manager()
    manager.set_value("simulation.body", "Moon")
    manager.set_value("thermophysical.conductivity_model", "martinez2021")
    written = manager._config_for_yaml()["thermophysical"]
    assert written.get("conductivity_model") == "martinez2021", written
    back = _round_trip(manager)
    assert back.get_value("thermophysical.conductivity_model") == "martinez2021"


def test_the_default_conductivity_model_is_the_engine_s():
    manager = _manager()
    assert manager.get_value("thermophysical.conductivity_model") == \
        ENGINE_DEFAULTS["thermophysical.conductivity_model"]


def test_coupling_mode_is_writable_and_round_trips():
    """PTM-102. The mode was unreachable while the field it gates was visible."""
    manager = _manager()
    assert manager.get_value("radiation.coupling_mode") == \
        ENGINE_DEFAULTS["radiation.coupling_mode"]
    for mode in ("bounded", "fixed", "legacy"):
        manager.set_value("radiation.coupling_mode", mode)
        written = manager._config_for_yaml()["radiation"]
        assert written.get("coupling_mode") == mode, written
        assert _round_trip(manager).get_value("radiation.coupling_mode") == mode


def test_nearfield_quadrature_order_and_mc_builder_are_writable():
    manager = _manager()
    for key, value in (("radiation.nearfield_quadrature_order", 1),
                       ("radiation.use_monte_carlo_vf", True),
                       ("radiation.mc_rays_per_face", 250000)):
        manager.set_value(key, value)
    written = manager._config_for_yaml()["radiation"]
    assert written["nearfield_quadrature_order"] == 1, written
    assert written["use_monte_carlo_vf"] is True, written
    assert written["mc_rays_per_face"] == 250000, written

    back = _round_trip(manager)
    assert back.get_value("radiation.nearfield_quadrature_order") == 1
    assert back.get_value("radiation.use_monte_carlo_vf") is True
    assert back.get_value("radiation.mc_rays_per_face") == 250000


TESTS = [
    test_H_override_is_written_on_a_builtin_body,
    test_H_override_survives_a_round_trip,
    test_H_is_in_the_engine_override_set,
    test_conductivity_model_is_written_and_round_trips,
    test_the_default_conductivity_model_is_the_engine_s,
    test_coupling_mode_is_writable_and_round_trips,
    test_nearfield_quadrature_order_and_mc_builder_are_writable,
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
    print(f"\n[test_gui_stage1_keys] {tally}")
    if failures:
        print("[test_gui_stage1_keys] FAIL")
        return 1
    print("[test_gui_stage1_keys] PASS"
          + (f" ({skipped} skipped)" if skipped else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
