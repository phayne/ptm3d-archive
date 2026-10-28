#!/usr/bin/env python3
"""The run window is seeded on a NEW run and preserved on an opened one.

WHAT WENT WRONG. `SimulationTab._on_body_combo_changed` auto-populated
`simulation.duration_days` to one synodic period of the newly selected body,
unconditionally. That handler does not only run when a user picks a body: a
generic (HORIZONS) body is not in the combo, so `load_from_config` ADDS its
name through `add_horizons_body` and selects it, and the selection fires the
handler -- during the load, with the author's own duration already in hand.

Measured on the fourteen shipped Justitia configurations, one fresh panel each:
all fourteen planned `simulation.duration_days` -> 1.3815675119109634, over
authored windows of 3.75 d (the six ice-depth studies), 2.0 d (six proxops),
3.0 d (descent) and 6.0 d (high). Opening a file and pressing Save would have
rewritten the run window of every one of them.

WHAT IS HELD HERE. The seeding rule, from both sides, driven through the REAL
combo box rather than by calling the handler:

* opening a configuration plans no duration edit, for a built-in body and for
  a generic one (the path that actually fired);
* a new run -- `File > New`, i.e. `reset_to_defaults` -- still gets one synodic
  period of whichever body is chosen, per body;
* an opened configuration whose window is unusable (<= 0) is seeded, because
  there is no author's value to preserve;
* and `ConfigManager.has_source_file`, the predicate the rule turns on,
  answers correctly across new / loaded / reset.

`test_gui_comment_roundtrip.test_opening_a_config_in_the_PANEL_plans_no_edits`
is the population-wide version of the first arm. This file is the mechanism:
it would still fail if that sweep were narrowed, and it names the body change
as the cause rather than reporting a rewritten key.

Run directly::

    python3 src/python/tests/test_gui_duration_seeding.py
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "gui"))
sys.path.insert(0, str(REPO_ROOT / "src" / "python"))

#: One Justitia solar day, in Earth days: the value that overwrote all
#: fourteen windows. Recomputed below from the engine-facing table rather than
#: compared against this literal -- it is here to name the number in the story.
JUSTITIA_SYNODIC_DAYS = 1.3815675119109634

#: A generic-body configuration that reproduces the load path: `simulation.body`
#: is a name the combo does not carry, so the panel adds and selects it.
GENERIC_CONFIG = REPO_ROOT / "examples" / "justitia_proxops.yaml"

#: A built-in-body configuration, for the arms that change body afterward. It
#: has to be one WITHOUT a `generic_body:` section: on a generic-body file the
#: synodic day stays the asteroid's whatever the combo says (the engine reads
#: the generic body and ignores `simulation.body`, which `ConfigManager` warns
#: about by name), so "choose Moon, expect a lunar day" would be measuring that
#: identity conflict rather than the seeding rule.
BUILTIN_CONFIG = REPO_ROOT / "examples" / "quickstart_moon_diurnal.yaml"

_APP = None


def _qt():
    """A QApplication, offscreen, with settings confined to a scratch dir."""
    global _APP
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        import PyQt6
    except ImportError as exc:                             # pragma: no cover
        pytest.skip(f"PyQt6 is unavailable here ({exc})")
    plug = Path(PyQt6.__file__).parent / "Qt6" / "plugins"
    os.environ.setdefault("QT_PLUGIN_PATH", str(plug))
    os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(plug / "platforms"))
    from PyQt6.QtWidgets import QApplication
    # Bound to a module global on purpose: `QApplication.instance() or
    # QApplication([])` as a bare expression lets the new instance be collected
    # immediately, and the next QWidget dies with "Must construct a
    # QApplication before a QWidget".
    _APP = QApplication.instance() or QApplication([])
    return _APP


def _panel():
    """A fresh manager + panel. FRESH matters: the combo remembers a body it
    was given, so a second config naming the same body never re-fires the
    handler, and a shared panel hides thirteen of the fourteen cases."""
    _qt()
    from gui_core.config_manager import ConfigManager
    from gui_panels.config_panel import ConfigurationPanel
    manager = ConfigManager()
    return manager, ConfigurationPanel(manager)


def _sim_tab(panel):
    """The SimulationTab, by the panel's own key."""
    tab = panel.tabs["simulation"]
    assert hasattr(tab, "duration_spin"), (
        "the 'simulation' tab no longer owns the duration control; this file "
        "is looking at the wrong widget")
    return tab


def _choose_body(tab, display_name):
    """Pick a body the way a user does: through the combo, signals live.

    Not `_on_body_combo_changed(name)` directly -- calling the slot would prove
    the slot, not the wiring, and the defect lived in WHEN the slot runs.
    """
    combo = tab.widgets["simulation.body"]
    idx = combo.findText(display_name)
    assert idx >= 0, (f"{display_name!r} is not in the body combo; it offers "
                      f"{[combo.itemText(i) for i in range(combo.count())]}")
    combo.setCurrentIndex(idx)


# --------------------------------------------------------------------------
# The predicate the rule turns on
# --------------------------------------------------------------------------

def test_has_source_file_distinguishes_a_new_run_from_an_opened_one():
    from gui_core.config_manager import ConfigManager

    manager = ConfigManager()
    assert not manager.has_source_file(), (
        "a manager that has loaded nothing is a NEW run")

    with tempfile.TemporaryDirectory() as scratch:
        work = Path(scratch) / GENERIC_CONFIG.name
        shutil.copyfile(GENERIC_CONFIG, work)
        manager.load_from_yaml(str(work))
        assert manager.has_source_file(), (
            "after load_from_yaml this is somebody's own file")

        manager.reset_to_defaults()
        assert not manager.has_source_file(), (
            "File > New must read as a new run again, or the seeding rules "
            "stay switched off for the rest of the session")


def test_saving_a_new_run_makes_it_the_users_own_file():
    """The transition the name `opened_from_file` would have hidden.

    `save_to_yaml` re-baselines against what is now on disk, so a run the user
    has SAVED reads as a file from that moment on -- and a body change
    afterward must not silently re-plan the window they just wrote. Held here
    because it is a rule with a surprising edge, not an implementation detail:
    the same click gives a different result before and after a Save, and the
    only defensible direction is the one that does not rewrite saved work.
    """
    from ptm.body_table import BODY_CAPABILITIES

    with tempfile.TemporaryDirectory() as scratch:
        manager, panel = _panel()
        tab = _sim_tab(panel)
        manager.reset_to_defaults()
        panel.load_from_config()
        _choose_body(tab, "Europa")
        seeded = manager.get_value("simulation.duration_days")
        assert abs(seeded - BODY_CAPABILITIES["Europa"]["solar_day_days"]) < 1e-9

        work = Path(scratch) / "saved.yaml"
        manager.save_to_yaml(str(work))
        assert manager.has_source_file(), (
            "a saved run is the user's own file from that moment on")

        _choose_body(tab, "Moon")
        assert manager.get_value("simulation.duration_days") == seeded, (
            f"changing body after a Save rewrote the saved window "
            f"{seeded} -> {manager.get_value('simulation.duration_days')}")
        panel.deleteLater()


# --------------------------------------------------------------------------
# Opening: the author's window stands
# --------------------------------------------------------------------------

def test_opening_a_generic_body_config_plans_no_duration_edit():
    """The exact path that fired: a body the combo does not carry."""
    with tempfile.TemporaryDirectory() as scratch:
        work = Path(scratch) / GENERIC_CONFIG.name
        shutil.copyfile(GENERIC_CONFIG, work)
        authored = yaml.safe_load(work.read_text())["simulation"]["duration_days"]

        manager, panel = _panel()
        manager.load_from_yaml(str(work))
        panel.load_from_config()

        assert manager.get_value("simulation.duration_days") == authored, (
            f"opening {GENERIC_CONFIG.name} changed the run window in memory: "
            f"{authored} -> {manager.get_value('simulation.duration_days')}")
        plan = manager.plan_save_edits()
        if plan is not None:                    # None == ruamel unavailable
            updates, _removals = plan
            assert "simulation.duration_days" not in updates, (
                f"a Save after an edit-free open would rewrite the run window "
                f"to {updates['simulation.duration_days']}")


def test_opening_every_shipped_justitia_config_plans_no_duration_edit():
    """All fourteen, one fresh panel each -- the measured population.

    A shared panel reports one offender out of fourteen (the combo keeps the
    body it was given), which is how this class survived the general sweep.
    """
    targets = sorted((REPO_ROOT / "examples").glob("justitia_*.yaml"))
    assert len(targets) >= 14, (
        f"only {len(targets)} Justitia configs found in examples/; this arm "
        f"is about the fourteen shipped ones and would be nearly vacuous")

    offenders = {}
    with tempfile.TemporaryDirectory() as scratch:
        for src in targets:
            work = Path(scratch) / src.name
            shutil.copyfile(src, work)
            authored = yaml.safe_load(
                work.read_text())["simulation"]["duration_days"]
            manager, panel = _panel()
            manager.load_from_yaml(str(work))
            panel.load_from_config()
            got = manager.get_value("simulation.duration_days")
            if got != authored:
                offenders[src.name] = (authored, got)
            panel.deleteLater()
    assert not offenders, (
        "opening these configurations moved the run window:\n  "
        + "\n  ".join(f"{k}: {v[0]} -> {v[1]}" for k, v in offenders.items()))


def test_a_body_change_on_an_opened_config_leaves_the_window_alone():
    """The interactive half of the same rule: an opened file is not a new run.

    Switching body on somebody's configuration is a body change, not a request
    to re-plan their campaign window. Both starting points, because the load
    path and the interactive path are different bugs: a generic body (the one
    that fired on open) and a built-in one.
    """
    for src, pick in ((GENERIC_CONFIG, "Moon"), (BUILTIN_CONFIG, "Europa")):
        with tempfile.TemporaryDirectory() as scratch:
            work = Path(scratch) / src.name
            shutil.copyfile(src, work)
            authored = yaml.safe_load(
                work.read_text())["simulation"]["duration_days"]

            manager, panel = _panel()
            manager.load_from_yaml(str(work))
            panel.load_from_config()
            tab = _sim_tab(panel)

            _choose_body(tab, pick)
            assert manager.get_value("simulation.duration_days") == authored, (
                f"choosing {pick} on the opened {src.name} rewrote its "
                f"window: {authored} -> "
                f"{manager.get_value('simulation.duration_days')}")
            panel.deleteLater()


def test_an_opened_config_with_no_usable_window_is_seeded():
    """Only-if-absent, the other half of the rule.

    `DEFAULT_CONFIG` always supplies the key, so "absent" means present and
    unusable. There is no author's value to preserve, and leaving 0 would ship
    a configuration the engine rejects.
    """
    from ptm.body_table import BODY_CAPABILITIES

    with tempfile.TemporaryDirectory() as scratch:
        work = Path(scratch) / "no_window.yaml"
        doc = yaml.safe_load(BUILTIN_CONFIG.read_text())
        doc["simulation"]["duration_days"] = 0.0
        work.write_text(yaml.safe_dump(doc, sort_keys=False))

        manager, panel = _panel()
        manager.load_from_yaml(str(work))
        panel.load_from_config()
        tab = _sim_tab(panel)

        _choose_body(tab, "Europa")
        got = manager.get_value("simulation.duration_days")
        expected = BODY_CAPABILITIES["Europa"]["solar_day_days"]
        assert abs(got - expected) < 1e-9, (
            f"a configuration carrying no usable window was not seeded: "
            f"duration_days = {got}, expected {expected}")


# --------------------------------------------------------------------------
# A new run: still one synodic period, per body
# --------------------------------------------------------------------------

def test_a_new_run_seeds_one_synodic_period_on_every_builtin_body():
    """Acceptance (b), on each body the engine implements, not just the Moon."""
    from ptm.body_table import BODY_CAPABILITIES, IMPLEMENTED_BODIES

    manager, panel = _panel()
    tab = _sim_tab(panel)
    assert not manager.has_source_file()

    wrong = {}
    for body in IMPLEMENTED_BODIES:
        manager.reset_to_defaults()
        panel.load_from_config()
        _choose_body(tab, body)
        expected = BODY_CAPABILITIES[body]["solar_day_days"]
        got = manager.get_value("simulation.duration_days")
        if abs(got - expected) > 1e-9:
            wrong[body] = (expected, got)
    assert not wrong, (
        "a new run did not get one synodic period of the chosen body:\n  "
        + "\n  ".join(f"{k}: expected {v[0]}, got {v[1]}" for k, v in wrong.items()))


def test_the_number_that_overwrote_the_justitia_windows_is_still_the_seed():
    """Naming it: the fix did not change WHAT is seeded, only WHEN.

    1.3815675119109634 d is one Justitia solar day. It was the right default
    for a new Justitia run the whole time; it was written on the wrong path.
    """
    from gui_tooltips import get_synodic_period_days

    with tempfile.TemporaryDirectory() as scratch:
        work = Path(scratch) / GENERIC_CONFIG.name
        shutil.copyfile(GENERIC_CONFIG, work)
        manager, _panel_ = _panel()
        manager.load_from_yaml(str(work))
        body = manager.get_value("simulation.body")
        assert abs(get_synodic_period_days(manager, body)
                   - JUSTITIA_SYNODIC_DAYS) < 1e-9, (
            "the Justitia synodic day moved; this file's story cites it")


if __name__ == "__main__":                                   # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
