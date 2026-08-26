#!/usr/bin/env python3
"""Item 2.1 / DP-9: the Basic/Advanced boundary is drawn by FIELD, not by tab.

Before this, "Advanced options" hid whole tabs -- Radiation, Illumination,
Engine Flags and HORIZONS -- which meant **the Basic tier could not express a
crater run at all**, and this model exists mostly to run craters. It also meant
the engine-environment disclosure (which changed the physics of every GUI run
on 2026-08-25) sat behind a checkbox a Basic user had no reason to tick.

What this pins:

* every section is reachable in the Basic tier, except ``body_definition``,
  whose gate is RELEVANCE (a custom body) and not tier;
* the fields a crater run needs -- ``radiation.enabled``, ``beaming``,
  ``rms_slope_deg`` -- are reachable in Basic;
* the expert fields are not (``aca_max_rank``, ``coupling_face_fraction``,
  ``mc_rays_per_face``, ...);
* the toggle is PURE VISIBILITY: the written config is byte-identical in both
  tiers, in both directions.

``isHidden()``, never ``isVisible()``: offscreen, no ancestor is shown, so
``isVisible()`` is vacuously False for every widget and a check written with it
tests nothing. That cost a probe check in Stage 1.

Run directly::

    python3 src/python/tests/test_gui_field_tiers.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "python"))
sys.path.insert(0, str(REPO_ROOT / "gui"))


class Skip(Exception):
    pass


_APP = None
_PANEL = None
_MANAGER = None


def _panel(scratch):
    """One offscreen ConfigurationPanel, reused (building one costs ~0.4 s)."""
    global _APP, _PANEL, _MANAGER
    if _PANEL is not None:
        return _MANAGER, _PANEL
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        import PyQt6
    except ImportError as exc:                     # pragma: no cover
        raise Skip(f"PyQt6 is unavailable here ({exc})")
    plug = Path(PyQt6.__file__).parent / "Qt6" / "plugins"
    os.environ.setdefault("QT_PLUGIN_PATH", str(plug))
    os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(plug / "platforms"))
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import QSettings

    # Redirect QSettings, or building a panel writes Paul's real
    # com.ptm3d.* plists.
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    for scope in (QSettings.Scope.UserScope, QSettings.Scope.SystemScope):
        QSettings.setPath(QSettings.Format.IniFormat, scope, str(scratch))

    from gui_core.config_manager import ConfigManager
    from gui_panels.config_panel import ConfigurationPanel

    _APP = QApplication.instance() or QApplication([])
    _MANAGER = ConfigManager()
    _PANEL = ConfigurationPanel(_MANAGER)
    return _MANAGER, _PANEL


def _hidden_by_tier(widget, tab):
    """True when `widget` sits inside an Advanced section that is hidden.

    Walks ancestors rather than asking the widget: hiding a container does not
    set the hidden flag on its children.
    """
    node = widget
    while node is not None and node is not tab:
        if node.isHidden():
            return True
        node = node.parentWidget()
    return False


#: Fields the BASIC tier must be able to reach. A colleague setting up a polar
#: crater with self-heating needs every one of these.
BASIC_REQUIRED = [
    ("radiation", "radiation.enabled"),
    ("radiation", "radiation.beaming"),
    ("radiation", "radiation.rms_slope_deg"),
    ("radiation", "radiation.use_gpu"),
    ("mesh", "mesh.import_file"),
    ("mesh", "mesh.resolution"),
    ("thermal", "thermal.run_thermal_mesh"),
    ("thermal", "thermal.use_gpu"),
    ("illumination", "illumination.include_primary_reflection"),
]

#: Fields the BASIC tier must NOT show. Each one is a knob whose default is the
#: validated value and whose misuse is silent.
ADVANCED_REQUIRED = [
    ("radiation", "radiation.aca_max_rank"),
    ("radiation", "radiation.coupling_face_fraction"),
    ("radiation", "radiation.mc_rays_per_face"),
    ("radiation", "radiation.nearfield_quadrature_order"),
    ("mesh", "mesh.occluder_file"),
    ("mesh", "mesh.latitude_override"),
    ("thermal", "thermal.spinup_seed"),
    ("thermal", "thermal.initial_state_file"),
    ("illumination", "illumination.positions_file"),
]


# --------------------------------------------------------------------------

def test_no_section_is_hidden_by_tier():
    with tempfile.TemporaryDirectory() as scratch:
        manager, panel = _panel(scratch)
        panel.advanced_toggle.setChecked(False)
        hidden = [panel._nav_keys[r] for r in range(panel.nav_rail.count())
                  if panel.nav_rail.item(r).isHidden()]
        assert hidden == ["body_definition"], (
            f"nav rows hidden in the Basic tier: {hidden}. Only "
            f"body_definition may be hidden, and its gate is relevance (a "
            f"custom body), not tier.")
        assert "radiation" not in hidden and "illumination" not in hidden, (
            "Radiation and/or Illumination are still hidden wholesale, so the "
            "Basic tier still cannot express a crater run")


def test_basic_reaches_the_fields_a_crater_run_needs():
    with tempfile.TemporaryDirectory() as scratch:
        manager, panel = _panel(scratch)
        panel.advanced_toggle.setChecked(False)
        missing = []
        for tab_name, key in BASIC_REQUIRED:
            tab = panel.tabs[tab_name]
            widget = tab.widgets.get(key)
            if widget is None:
                missing.append(f"{key}: no control at all")
            elif _hidden_by_tier(widget, tab):
                missing.append(f"{key}: hidden in the Basic tier")
        assert not missing, "\n  ".join(["Basic tier gaps:"] + missing)


def test_advanced_fields_are_hidden_until_asked_for():
    with tempfile.TemporaryDirectory() as scratch:
        manager, panel = _panel(scratch)
        panel.advanced_toggle.setChecked(False)
        leaked = []
        for tab_name, key in ADVANCED_REQUIRED:
            tab = panel.tabs[tab_name]
            widget = tab.widgets.get(key)
            if widget is None:
                leaked.append(f"{key}: no control at all")
            elif not _hidden_by_tier(widget, tab):
                leaked.append(f"{key}: visible in the Basic tier")
        assert not leaked, "\n  ".join(["Basic tier is too crowded:"] + leaked)


def test_advanced_reveals_the_sections():
    """Advanced ON puts every expert section on screen -- collapsed."""
    with tempfile.TemporaryDirectory() as scratch:
        manager, panel = _panel(scratch)
        panel.advanced_toggle.setChecked(True)
        hidden_sections = [
            f"{n}/{s.title}" for n, t in panel.tabs.items()
            for s in t.advanced_sections if s.isHidden()]
        assert not hidden_sections, (
            f"sections still hidden with Advanced on: {hidden_sections}")
        # Collapsed is the intended landing state: the header is there, the
        # eighteen radiation knobs are not in the user's face.
        collapsed = [s for t in panel.tabs.values()
                     for s in t.advanced_sections if s._collapsed]
        assert collapsed, (
            "every Advanced section opened itself, which puts the whole expert "
            "surface on screen at once -- the crowding item 2.1 exists to fix")


def test_expanding_an_advanced_section_reaches_every_field():
    with tempfile.TemporaryDirectory() as scratch:
        manager, panel = _panel(scratch)
        panel.advanced_toggle.setChecked(True)
        for tab in panel.tabs.values():
            for section in tab.advanced_sections:
                section.set_collapsed(False)
        try:
            still_hidden = []
            for tab_name, key in BASIC_REQUIRED + ADVANCED_REQUIRED:
                tab = panel.tabs[tab_name]
                widget = tab.widgets.get(key)
                if widget is not None and _hidden_by_tier(widget, tab):
                    still_hidden.append(key)
            assert not still_hidden, (
                f"unreachable even with Advanced on and every section "
                f"expanded: {still_hidden}")
        finally:
            for tab in panel.tabs.values():
                for section in tab.advanced_sections:
                    section.set_collapsed(True)


def test_the_toggle_changes_nothing_that_is_written():
    """Visibility only. If the tier can change the config, it is not a tier."""
    with tempfile.TemporaryDirectory() as scratch:
        manager, panel = _panel(scratch)
        import yaml

        panel.advanced_toggle.setChecked(False)
        basic = yaml.dump(manager._config_for_yaml(), sort_keys=True)
        panel.advanced_toggle.setChecked(True)
        advanced = yaml.dump(manager._config_for_yaml(), sort_keys=True)
        panel.advanced_toggle.setChecked(False)
        back = yaml.dump(manager._config_for_yaml(), sort_keys=True)

        assert basic == advanced, (
            "turning Advanced ON changed the configuration that would be "
            "written; the tier is a visibility filter and nothing else")
        assert basic == back, (
            "turning Advanced OFF again changed the configuration")


def test_every_tab_can_be_asked_about_its_tier():
    """The contract lives on the base class, so no tab can forget it."""
    with tempfile.TemporaryDirectory() as scratch:
        manager, panel = _panel(scratch)
        for name, tab in panel.tabs.items():
            assert hasattr(tab, "advanced_sections"), (
                f"{name} has no advanced_sections list")
            assert hasattr(tab, "set_advanced_visible"), (
                f"{name} cannot apply the tier")
        with_sections = {n for n, t in panel.tabs.items()
                         if t.advanced_sections}
        # The tabs that carry expert machinery. A tab dropping off this set is
        # a regression: its knobs are back in the Basic tier.
        expected = {"mesh", "thermal", "thermophysical", "radiation",
                    "illumination", "engine_flags"}
        assert expected <= with_sections, (
            f"tabs that lost their Advanced section: "
            f"{sorted(expected - with_sections)}")


def test_the_engine_environment_is_disclosed_in_the_basic_tier():
    """The 2026-08-25 physics change must be readable without ticking a box."""
    with tempfile.TemporaryDirectory() as scratch:
        manager, panel = _panel(scratch)
        panel.advanced_toggle.setChecked(False)
        tab = panel.tabs["engine_flags"]
        summary = getattr(tab, "_env_summary", None)
        assert summary is not None, "the Engine Flags tab has no Basic summary"
        assert not _hidden_by_tier(summary, tab), (
            "the engine-environment summary is Advanced-only, which is where "
            "the inert PTM_ILLUM_RISE_HABS hid for a month")
        text = summary.text()
        assert "PTM_ILLUM_RISE_EPS" in text, (
            f"the summary does not name what will be exported: {text!r}")
        assert "PTM_EQ_PERFACE_FLUX" in text, (
            "the summary does not name what is deliberately NOT injected")


TESTS = [
    test_no_section_is_hidden_by_tier,
    test_basic_reaches_the_fields_a_crater_run_needs,
    test_advanced_fields_are_hidden_until_asked_for,
    test_advanced_reveals_the_sections,
    test_expanding_an_advanced_section_reaches_every_field,
    test_the_toggle_changes_nothing_that_is_written,
    test_every_tab_can_be_asked_about_its_tier,
    test_the_engine_environment_is_disclosed_in_the_basic_tier,
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
    print(f"\n[test_gui_field_tiers] {total - failures - skipped}/{total} "
          f"passed, {skipped} skipped")
    if skipped == total:
        print("[test_gui_field_tiers] SKIPPED ENTIRELY -- proved NOTHING here.")
    if failures:
        print("[test_gui_field_tiers] FAIL")
        return 1
    print("[test_gui_field_tiers] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
