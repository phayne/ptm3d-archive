#!/usr/bin/env python3
"""The radiation-exchange switch: on every card, in every mode, in every run type.

WHAT WAS WRONG. `radiation.enabled` decides whether faces exchange infrared and
reflected sunlight with each other -- whether a shadowed crater floor is warmed
by its walls at all. It is a key the engine reads on every body. It was
reachable on exactly two of the five cards and only in their LOCAL mode:

    Moon    global  -- no control      Moon    local   -- control
    Europa  global  -- no control      Europa  local   -- control
    Mercury global  -- no control
    asteroid/custom -- no control, on any route

...in every run type. So on the integrated-statistics card Paul was testing,
there was no on/off at all for most of the routes the GUI offers, while the
presets on those routes WRITE the key. A key a preset writes and no screen
shows is the same defect as a control the engine ignores, from the other side.

WHY IT IS OFFERED IN GLOBAL MODE TOO. On a smooth procedural icosphere the
switch is a no-op that costs minutes and gigabytes -- measured 2026-08-28, a
1.13M-block H-matrix in which every block was Zero. `gui_core.preflight`
already WARNS about exactly that and does not switch anything off, and the arm
below holds that warning in place. Hiding the control instead would mean an
opened configuration with `enabled: true` on a global mesh showed nothing at
all, which is how it stays on.

THE DEPENDENCY. `radiation.beaming` lives inside the radiation package, so
`radiation.enabled` is its parent. With the parent off, beaming is INERT (it
reaches the solver only through `rad_config`, built inside
`if (enable_radiation_exchange ...)`, `ptm_thermal_sim.cpp:15663` and `:17977`)
and can still STOP the run (the facet-scale check at `:15372` and the
zero-slope check at `:13763` are behind `rad_beaming` alone). A control that is
inert and can still refuse to launch should not be reachable, which is what
`facet_scale.beaming_gate` now says.

One panel per check, deliberately: see `_fresh`.

Run directly::

    python3 src/python/tests/test_gui_radiation_toggle.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "python"))
sys.path.insert(0, str(REPO_ROOT / "gui"))

_APP = None


def _qt():
    global _APP
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        import PyQt6
    except ImportError as exc:                              # pragma: no cover
        pytest.skip(f"PyQt6 is unavailable here ({exc})")
    plug = Path(PyQt6.__file__).parent / "Qt6" / "plugins"
    os.environ.setdefault("QT_PLUGIN_PATH", str(plug))
    os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(plug / "platforms"))
    from PyQt6.QtCore import QSettings
    from PyQt6.QtWidgets import QApplication
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    scratch = tempfile.mkdtemp()
    for scope in (QSettings.Scope.UserScope, QSettings.Scope.SystemScope):
        QSettings.setPath(QSettings.Format.IniFormat, scope, scratch)
    _APP = QApplication.instance() or QApplication([])
    return _APP


def _fresh():
    """A brand-new manager and body-first panel, per check.

    NOT shared. ``test_gui_body_first`` keeps one panel in a module global and
    has a measured order dependency because of it; ``test_gui_run_type_defaults``
    already documents that and uses this pattern instead.
    """
    _qt()
    from gui_core.config_manager import ConfigManager
    from gui_panels.config_panel import ConfigurationPanel

    manager = ConfigManager()
    panel = ConfigurationPanel(manager)
    return manager, panel.tabs["start"]


#: Every route the landing screen offers, as (body, mode).
ROUTES = (("Moon", "global"), ("Moon", "local"),
          ("Europa", "global"), ("Europa", "local"),
          ("Mercury", "global"), ("asteroid", "body"), ("custom", "body"))

RUN_TYPES = ("time_series", "integrated")


# --------------------------------------------------------------------------
# 1. The switch is there, everywhere
# --------------------------------------------------------------------------

def test_every_card_offers_the_radiation_switch_in_every_run_type():
    """The defect this file exists to close, stated as a property."""
    missing = []
    for body, mode in ROUTES:
        for run_type in RUN_TYPES:
            _manager, start = _fresh()
            card = start.open_card(body, mode, run_type, apply_defaults=True)
            widget = card.widgets.get("radiation.enabled")
            if widget is None:
                missing.append(f"{body}/{mode}/{run_type}")
            elif widget.isHidden():
                missing.append(f"{body}/{mode}/{run_type} (hidden)")
    assert not missing, (
        "these routes offer no radiation-exchange control: "
        + ", ".join(missing))
    print(f"    [radiation] {len(ROUTES) * len(RUN_TYPES)} route/run-type "
          f"pairs all offer radiation.enabled")


def test_the_switch_is_bound_to_the_key_and_writes_it():
    for body, mode in ROUTES:
        manager, start = _fresh()
        card = start.open_card(body, mode, "integrated", apply_defaults=True)
        widget = card.widgets["radiation.enabled"]
        widget.setChecked(True)
        assert manager.get_value("radiation.enabled") is True, \
            f"{body}/{mode}: ticking the switch did not write the key"
        widget.setChecked(False)
        assert manager.get_value("radiation.enabled") is False, \
            f"{body}/{mode}: unticking the switch did not write the key"


def test_the_card_reports_the_switch_rather_than_hiding_it():
    """`radiation.enabled` must not appear in the out-of-card listing.

    The listing is what the cards' "nothing hidden is silently dropped"
    contract rests on; a key now SHOWN must leave it, or the card claims to be
    hiding something it displays.
    """
    from gui_core.body_cards import out_of_card_settings
    for body, mode in ROUTES:
        manager, start = _fresh()
        card = start.open_card(body, mode, "integrated", apply_defaults=True)
        manager.set_value("radiation.enabled", True)
        hidden = {key for key, _v, _d in out_of_card_settings(
            manager.get_config(), card.exposed_keys())}
        assert "radiation.enabled" not in hidden, (
            f"{body}/{mode} shows the radiation switch and still lists it as "
            f"a setting the card does not show")


def test_the_view_factor_cache_travels_with_the_switch():
    """Turning exchange on without a cache path rebuilds the matrix each run."""
    for body, mode in ROUTES:
        _manager, start = _fresh()
        card = start.open_card(body, mode, "integrated", apply_defaults=True)
        assert "radiation.view_factor_file" in card.widgets, (
            f"{body}/{mode} offers radiation exchange with no way to point at "
            f"a view-factor cache")


# --------------------------------------------------------------------------
# 2. The parent/child dependency
# --------------------------------------------------------------------------

def test_beaming_is_refused_while_radiation_is_off():
    from gui_core.facet_scale import beaming_gate
    config = {"simulation": {"body": "Moon"},
              "mesh": {"resolution": 4},
              "radiation": {"enabled": False, "beaming": True}}
    policy = beaming_gate(config)
    assert not policy.allowed
    assert "Radiation exchange is off" in policy.reason
    # It says BOTH facts: inert, and still able to stop the run.
    assert "nothing to act on" in policy.reason
    assert "STOP the run" in policy.reason


def test_beaming_is_judged_on_the_scale_once_radiation_is_on():
    """With the parent on, the gate is exactly the facet-scale rule again."""
    from gui_core.facet_scale import beaming_gate, beaming_policy
    config = {"simulation": {"body": "Moon"},
              "mesh": {"resolution": 4},
              "radiation": {"enabled": True, "beaming": False}}
    gate = beaming_gate(config)
    rule = beaming_policy(config)
    assert gate.allowed == rule.allowed
    assert gate.reason == rule.reason


def test_the_string_true_counts_as_on():
    """YAML hands back a str for a quoted boolean; the engine reads it as one."""
    from gui_core.facet_scale import beaming_gate
    config = {"simulation": {"body": "Moon"}, "mesh": {"resolution": 4},
              "radiation": {"enabled": "true", "beaming": False}}
    assert "Radiation exchange is off" not in beaming_gate(config).reason


def test_the_moon_card_disables_beaming_when_radiation_is_off():
    manager, start = _fresh()
    card = start.open_card("Moon", "global", "integrated", apply_defaults=True)
    beaming = card.widgets["radiation.beaming"]
    manager.set_value("radiation.enabled", False)
    card.refresh_radiation()
    assert not beaming.isEnabled(), (
        "the beaming switch is offerable while radiation exchange is off, "
        "where it changes no temperature and can still abort the run")
    assert "Radiation exchange is off" in card.beaming_note.text()

    card.widgets["radiation.enabled"].setChecked(True)
    assert manager.get_value("radiation.enabled") is True
    # ...and with the parent on, the facet-scale rule decides. A lunar ico4 is
    # 1,244 km per facet, so it decides yes.
    assert beaming.isEnabled()


def test_turning_radiation_off_does_not_rewrite_the_users_beaming():
    """Gating a control is not editing the configuration behind it."""
    manager, start = _fresh()
    card = start.open_card("Moon", "local", "integrated", apply_defaults=True)
    manager.set_value("radiation.enabled", True)
    manager.set_value("radiation.beaming", True)
    manager.set_value("radiation.rms_slope_deg", 25.0)
    manager.set_value("radiation.enabled", False)
    card.refresh_radiation()
    assert manager.get_value("radiation.beaming") is True, (
        "the card turned the file's beaming off; a card gates its own "
        "control, it does not edit the user's configuration")
    assert manager.get_value("radiation.rms_slope_deg") == 25.0


def test_only_the_moon_card_carries_a_beaming_switch():
    """`facet_scale.BEAMING_BODIES` is lunar-only, so the others must not."""
    from gui_core.facet_scale import BEAMING_BODIES
    assert BEAMING_BODIES == ("Moon",)
    for body, mode in ROUTES:
        _manager, start = _fresh()
        card = start.open_card(body, mode, "integrated", apply_defaults=True)
        has_beaming = "radiation.beaming" in card.widgets
        assert has_beaming == (body == "Moon"), (
            f"{body}/{mode}: beaming switch present={has_beaming}, but the "
            f"Smith slope distribution is calibrated for {BEAMING_BODIES}")


# --------------------------------------------------------------------------
# 3. The convex-mesh disclosure, which is what makes a global switch honest
# --------------------------------------------------------------------------

def _find_engine():
    """The engine binary, for a pre-flight that stops without one.

    `run_preflight` reports a missing executable as an ERROR and returns
    before it reaches any of the physics checks, so a check about a WARNING
    has to hand it a real one -- otherwise it passes vacuously on a machine
    with no build, which is the silently-skipped gate this project keeps
    paying for.
    """
    from gui_core.simulation_runner import SimulationRunner
    exe = SimulationRunner.find_ptm_executable()
    if not exe:
        pytest.skip("no ptm_thermal_sim; the pre-flight stops before it "
                    "reaches the convex-mesh check. THIS ARM DID NOT RUN.")
    return exe


def test_the_preflight_still_warns_about_a_convex_global_run():
    """The switch is offered globally BECAUSE this warning exists.

    If it ever stopped firing, offering the control on a smooth icosphere
    would be offering minutes and gigabytes of work that computes exactly
    nothing, with nothing said.
    """
    from gui_core.preflight import run_preflight
    config = {
        "simulation": {"body": "Moon", "start_time": "2026-01-01T00:00:00",
                       "duration_days": 29.53, "timesteps": 144},
        "mesh": {"resolution": 4, "use_topographic": False,
                 "import_file": ""},
        "thermal": {"run_thermal_mesh": True},
        "radiation": {"enabled": True},
        "illumination": {"albedo_source": "none"},
        "output": {"directory": tempfile.mkdtemp()},
    }
    report = run_preflight(config, _find_engine, REPO_ROOT)
    convex = [w for w in report.warnings if "convex" in w]
    assert convex, (
        "the convex-mesh warning did not fire; a global radiation switch is "
        "only honest while it does")
    assert "contributes exactly nothing" in convex[0]


def test_the_warning_does_not_fire_on_terrain():
    from gui_core.preflight import run_preflight
    config = {
        "simulation": {"body": "Moon", "start_time": "2026-01-01T00:00:00",
                       "duration_days": 29.53, "timesteps": 144},
        "mesh": {"resolution": 4, "use_topographic": True},
        "thermal": {"run_thermal_mesh": True},
        "radiation": {"enabled": True},
        "illumination": {"albedo_source": "none"},
        "output": {"directory": tempfile.mkdtemp()},
    }
    report = run_preflight(config, _find_engine, REPO_ROOT)
    assert not [w for w in report.warnings if "convex" in w]


if __name__ == "__main__":                                   # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
