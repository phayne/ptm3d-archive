#!/usr/bin/env python3
"""The body-first surface: the flow, the routing, and what each card emits.

Five properties, in the order they would break:

1. **The buttons are the engine's.** The landing screen offers exactly the
   bodies ``planetary_properties.h`` implements, plus the two generic-body
   routes. A body added or removed in C++ changes this screen through the
   generated table, not through somebody remembering.
2. **The flow is a state machine and it terminates.** Every
   body x mode x run-type combination reaches a card, and the card that comes
   up is the one that was asked for.
3. **Open requires a body key.** A file with no ``simulation.body`` is refused
   before anything is loaded, and so is a file that is not YAML. Both negative
   controls are here, because a check that only ever sees valid input has not
   been tested.
4. **Every card emits a configuration the ENGINE accepts.** Each
   body x mode x run type is applied and handed to
   ``ptm_thermal_sim --validate-config``. Skipped, loudly, where no binary is
   available -- never reported as passing.
5. **Nothing the card hides is hidden silently.** Opening a shipped
   configuration lists every key it sets that the card does not show and that
   differs from the engine default.

Run directly::

    python3 src/python/tests/test_gui_body_first.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "python"))
sys.path.insert(0, str(REPO_ROOT / "gui"))


class Skip(Exception):
    """A check that cannot run here, as distinct from one that failed."""


_APP = None
_PANEL = None
_MANAGER = None


def _panel(scratch):
    """One offscreen ConfigurationPanel, reused across the checks."""
    global _APP, _PANEL, _MANAGER
    if _PANEL is not None:
        return _MANAGER, _PANEL
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        import PyQt6
    except ImportError as exc:                              # pragma: no cover
        raise Skip(f"PyQt6 is unavailable here ({exc})")
    plug = Path(PyQt6.__file__).parent / "Qt6" / "plugins"
    os.environ.setdefault("QT_PLUGIN_PATH", str(plug))
    os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(plug / "platforms"))
    from PyQt6.QtCore import QSettings
    from PyQt6.QtWidgets import QApplication
    # Redirect QSettings BEFORE any panel touches it, or this writes the
    # developer's real com.ptm3d.* preferences.
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    for scope in (QSettings.Scope.UserScope, QSettings.Scope.SystemScope):
        QSettings.setPath(QSettings.Format.IniFormat, scope, str(scratch))
    try:
        from gui_core.config_manager import ConfigManager
        from gui_panels.config_panel import ConfigurationPanel
    except ImportError as exc:                              # pragma: no cover
        raise Skip(f"the GUI panel stack is unavailable here ({exc})")
    _APP = QApplication.instance() or QApplication([])
    _MANAGER = ConfigManager()
    _PANEL = ConfigurationPanel(_MANAGER)
    return _MANAGER, _PANEL


def _routes():
    """Every (body, mode, run type) the surface offers."""
    from gui_core.body_cards import body_buttons, modes_for, run_types_for
    out = []
    for button in body_buttons():
        for mode in modes_for(button.key):
            for run_type in run_types_for(button.key, mode.key):
                out.append((button.key, mode.key, run_type.key))
    return out


# --------------------------------------------------------------------------
# 1. The buttons are the engine's
# --------------------------------------------------------------------------

def test_the_landing_screen_offers_exactly_the_implemented_bodies():
    from ptm.body_table import IMPLEMENTED_BODIES
    from gui_core.body_cards import CUSTOM_ROUTES, body_buttons

    buttons = body_buttons()
    builtin = [b.key for b in buttons if b.builtin]
    assert builtin == list(IMPLEMENTED_BODIES), (
        f"the landing screen offers {builtin}; the engine implements "
        f"{list(IMPLEMENTED_BODIES)}. A body it does not implement is refused "
        f"at config parse, so a button for one cannot start a run.")
    generic = [b.key for b in buttons if not b.builtin]
    assert generic == list(CUSTOM_ROUTES), (
        f"the generic-body routes changed: {generic}")
    assert len(buttons) == 5, (
        f"the landing screen has {len(buttons)} buttons, not five "
        f"(three implemented bodies, asteroid, custom body)")
    print(f"    [landing] {len(buttons)} buttons: {[b.key for b in buttons]}")


def test_a_body_button_carries_one_short_functional_line_and_no_paragraph():
    """The landing screen offers a choice; it does not comment on it.

    It used to carry a paragraph of muted prose under each button. One of them
    opened "The honest destination for a world the engine has no built-in
    properties for" -- the register of a session note, in front of somebody who
    has never read one. Two properties replace it:

    * a button has ONE subtitle, one line, short enough to read at a glance;
    * `BodyButton` has no `note` field at all, so the paragraph cannot be put
      back in the table and quietly fail to appear.
    """
    from gui_core.body_cards import body_buttons

    long_lines, multi_line = [], []
    for button in body_buttons():
        assert not hasattr(button, "note"), (
            f"BodyButton grew a `note` field back ({button.key}). The landing "
            f"screen does not render one, so it would be text nobody sees.")
        assert button.subtitle.strip(), (
            f"{button.key} has no subtitle; the button would read as a bare "
            f"word with no idea what it leads to")
        if "\n" in button.subtitle:
            multi_line.append(button.key)
        if len(button.subtitle) > 55:
            long_lines.append(f"{button.key}: {len(button.subtitle)} chars -- "
                              f"{button.subtitle!r}")
    assert not multi_line, (
        f"a body button's subtitle spans more than one line: {multi_line}")
    assert not long_lines, "\n  ".join(
        ["a body-button subtitle is a paragraph, not a functional line:"]
        + long_lines)
    print("    [landing copy] "
          + "; ".join(f"{b.key}: {b.subtitle!r}" for b in body_buttons()))


def test_removing_the_paragraphs_did_not_remove_a_disclosure():
    """The half that makes the deletion above safe rather than merely tidy.

    Mercury's paragraph was the only one carrying a claim a user needs: it
    ships no albedo map, no topography and no worked example, so every Mercury
    run is a uniform-albedo sphere. Deleting the paragraph without this check
    would delete the warning with it. It now lives on the Mercury card -- where
    the controls it constrains are -- plus a two-word badge on the button.

    The custom-body paragraph is the same shape: what that route is for, and
    what happens if you type the body's name under `simulation.body` instead,
    is on the card.
    """
    from gui_core.body_cards import body_buttons, card_preset

    mercury = [b for b in body_buttons() if b.key == "Mercury"]
    assert mercury, "the Mercury button is gone"
    assert mercury[0].maturity.strip(), (
        "the Mercury button lost both its paragraph and its badge, so nothing "
        "on the landing screen says this body is thin on data")

    with tempfile.TemporaryDirectory() as scratch:
        _, panel = _panel(scratch)
        from PyQt6.QtWidgets import QLabel
        card = panel.tabs["card_mercury"]
        text = "\n".join(w.text() for w in card.findChildren(QLabel))
        for phrase in ("no albedo map", "no topography", "uniform-albedo"):
            assert phrase in text, (
                f"the Mercury card no longer says {phrase!r}, and the landing "
                f"paragraph that used to say it has been removed:\n{text[:400]}")

    custom = card_preset("custom", "body")
    assert custom is not None, "the custom-body card lost its preset"
    assert "simulation.body" in custom.summary, (
        f"the custom-body card no longer explains that naming the body under "
        f"simulation.body will not work, and the landing paragraph that used "
        f"to say it has been removed:\n{custom.summary}")


def test_the_landing_screen_is_buttons_not_prose():
    """Measured on the rendered page, not on the tables.

    A paragraph can come back through any widget, not only through
    `BODY_PRESENTATION`. So this counts the standing prose actually on the
    landing page: its heading, its (normally hidden) routing message and the
    baseline entry's (normally hidden) report are the only long labels allowed.
    """
    with tempfile.TemporaryDirectory() as scratch:
        _, panel = _panel(scratch)
        from PyQt6.QtWidgets import QLabel
        start = panel.tabs["start"]
        page = start.stack.widget(start.LANDING)
        prose = []
        for label in page.findChildren(QLabel):
            text = (label.text() or "").strip()
            if len(text) > 90 and label.isVisibleTo(page):
                prose.append(f"{len(text)} chars: {text[:90]}...")
        assert not prose, "\n  ".join(
            ["the landing screen carries standing paragraphs again -- it is a "
             "choice of body, and the detail belongs in the tooltips and on "
             "the cards:"] + prose)
        print("    [landing copy] no standing paragraph on the landing page")


def test_a_body_the_presentation_table_forgets_is_loud():
    """A body implemented in C++ and missing here would have no way in."""
    from gui_core import body_cards
    original = dict(body_cards.BODY_PRESENTATION)
    victim = sorted(original)[0]
    try:
        body_cards.BODY_PRESENTATION.pop(victim)
        raised = False
        try:
            body_cards.body_buttons()
        except KeyError as exc:
            raised = True
            assert victim in str(exc), exc
        assert raised, (
            f"removing {victim} from BODY_PRESENTATION produced no error, so "
            f"a body the engine implements could silently lose its button")
    finally:
        body_cards.BODY_PRESENTATION.clear()
        body_cards.BODY_PRESENTATION.update(original)


def test_a_grid_is_offered_only_where_a_grid_exists():
    """The sweep run type is the one that could most easily be a lie."""
    from gui_core.body_cards import europa_sweep_points, run_types_for
    with_sweep = [(b, m) for b, m, r in _routes() if r == "sweep"]
    assert with_sweep == [("Europa", "local")], (
        f"a grid is offered on {with_sweep}; the only Cartesian product the "
        f"shipped corpus contains is the Europa Sutton-DEM one")
    points = europa_sweep_points(REPO_ROOT)
    assert len(points) == 45, (
        f"the Europa grid derived {len(points)} points, not 45")
    assert "sweep" not in [r.key for r in run_types_for("Moon", "local")], (
        "the Moon site card offers a grid, and no shipped site manifest backs "
        "one")


# --------------------------------------------------------------------------
# 2. The flow terminates, on the card that was asked for
# --------------------------------------------------------------------------

def test_every_route_reaches_its_own_card():
    with tempfile.TemporaryDirectory() as scratch:
        _, panel = _panel(scratch)
        start = panel.tabs["start"]
        checked = 0
        for body, mode, run_type in _routes():
            start.choose_body(body)
            assert start.stack.currentIndex() == start.RUNTYPE, (
                f"choosing {body} did not reach the run-type step")
            start._choose_mode(mode)
            card = start.open_card(body, mode, run_type, apply_defaults=True)
            assert card is not None, f"{body}/{mode}: no card"
            assert start.stack.currentIndex() == start.CARD, (
                f"{body}/{mode}/{run_type} did not reach a card")
            assert start.card_stack.currentWidget() is card, (
                f"{body}/{mode}/{run_type} showed a different card")
            assert card.route == body, (
                f"{body} showed the {card.route} card")
            assert card._active and card._mode == mode, (
                f"{body}/{mode}: the card is not active in that mode")
            for other in start.cards.values():
                if other is not card:
                    assert not other._active, (
                        f"{other.route} stayed active behind {body}")
            checked += 1
        print(f"    [flow] {checked} body x mode x run-type routes reached "
              f"their card")


def test_choosing_a_run_type_writes_and_reports_it():
    from gui_core.body_cards import RUN_TYPES_BY_KEY
    with tempfile.TemporaryDirectory() as scratch:
        manager, panel = _panel(scratch)
        start = panel.tabs["start"]
        for run_type in ("time_series", "integrated"):
            start.choose_body("Moon")
            card = start.open_card("Moon", "global", run_type,
                                   apply_defaults=True)
            for key, value in RUN_TYPES_BY_KEY[run_type].sets.items():
                assert manager.get_value(key) == value, (
                    f"{run_type} did not write {key} = {value!r}")
                assert key in card.report_text(), (
                    f"{run_type} wrote {key} without reporting it:\n"
                    f"{card.report_text()}")


def test_the_mode_choice_changes_which_run_types_are_offered():
    with tempfile.TemporaryDirectory() as scratch:
        _, panel = _panel(scratch)
        start = panel.tabs["start"]
        start.choose_body("Europa")
        start._choose_mode("global")
        titles = [b.text() for b in start._runtype_buttons]
        assert not any("grid" in t.lower() for t in titles), (
            f"whole-Europa offers a grid: {titles}")
        start._choose_mode("local")
        titles = [b.text() for b in start._runtype_buttons]
        assert any("grid" in t.lower() for t in titles), (
            f"a Europa DEM site does not offer the grid: {titles}")


# --------------------------------------------------------------------------
# 3. Open requires a body key -- with its negative controls
# --------------------------------------------------------------------------

def test_open_refuses_a_file_that_is_not_yaml():
    from gui_core.body_cards import open_refusal
    for name in ("mesh.ply", "state.bin", "notes.txt", "project.ptm3d"):
        refusal = open_refusal(name, {"simulation": {"body": "Moon"}})
        assert refusal, f"{name} was accepted as a configuration"
        assert "YAML" in refusal, refusal


def test_open_refuses_yaml_without_a_body():
    from gui_core.body_cards import open_refusal
    for config in ({}, {"simulation": {}}, {"simulation": {"body": ""}},
                   {"mesh": {"resolution": 5}}, {"simulation": None},
                   None, [1, 2, 3]):
        refusal = open_refusal("thing.yaml", config)
        assert refusal, f"{config!r} was accepted without a body"
        assert "simulation.body" in refusal or "mapping" in refusal, refusal


def test_open_accepts_every_shipped_run_configuration():
    """The positive control. A refusal that refuses everything is not a check."""
    import yaml
    from gui_core.body_cards import open_refusal
    accepted = refused = 0
    refusals = []
    for path in sorted((REPO_ROOT / "configs").glob("*.yaml")):
        try:
            config = yaml.safe_load(path.read_text())
        except Exception:                                    # noqa: BLE001
            continue
        if not isinstance(config, dict) or "simulation" not in config:
            continue                     # not a run configuration at all
        refusal = open_refusal(path, config)
        if refusal:
            refused += 1
            refusals.append(f"{path.name}: {refusal}")
        else:
            accepted += 1
    assert accepted > 50, (
        f"only {accepted} shipped configurations were accepted; the check is "
        f"refusing valid files")
    assert not refusals, "\n  ".join(
        ["shipped run configurations were refused:"] + refusals[:10])
    print(f"    [open] {accepted} shipped run configurations accepted, "
          f"{refused} refused")


def test_every_shipped_configuration_routes_to_a_card_or_says_why():
    import yaml
    from gui_core.body_cards import detect_route
    routed, refused = 0, []
    for path in sorted((REPO_ROOT / "configs").glob("*.yaml")):
        try:
            config = yaml.safe_load(path.read_text())
        except Exception:                                    # noqa: BLE001
            continue
        if not isinstance(config, dict) or "simulation" not in config:
            continue
        route = detect_route(config)
        if route.ok:
            routed += 1
        else:
            refused.append(f"{path.name}: {route.refusal}")
    assert routed > 50, f"only {routed} configurations routed to a card"
    assert not refused, "\n  ".join(
        ["shipped configurations with no card:"] + refused[:10])
    print(f"    [route] {routed} shipped configurations land on a body card")


def test_loading_a_config_lands_on_its_own_card():
    with tempfile.TemporaryDirectory() as scratch:
        manager, panel = _panel(scratch)
        cases = [
            ("configs/europa_lat45_ridges.yaml", "Europa", "local"),
            ("examples/quickstart_moon_diurnal.yaml", "Moon", "global"),
        ]
        for relative, body, mode in cases:
            path = REPO_ROOT / relative
            if not path.exists():
                raise Skip(f"{relative} is not in this checkout")
            manager.load_from_yaml(str(path))
            panel.load_from_config()
            route = panel.land_on_loaded_config()
            assert route.ok, f"{relative}: {route.refusal}"
            assert (route.body_key, route.mode_key) == (body, mode), (
                f"{relative} landed on {route.body_key}/{route.mode_key}, "
                f"expected {body}/{mode}")
            start = panel.tabs["start"]
            assert start.stack.currentIndex() == start.CARD
            assert start.card_stack.currentWidget() is start.cards[body]


# --------------------------------------------------------------------------
# 4. What the card hides, it lists
# --------------------------------------------------------------------------

def test_an_opened_file_lists_what_the_card_does_not_show():
    with tempfile.TemporaryDirectory() as scratch:
        manager, panel = _panel(scratch)
        path = REPO_ROOT / "configs" / "europa_lat45_ridges.yaml"
        if not path.exists():
            raise Skip("the Europa DEM configurations are not in this checkout")
        manager.load_from_yaml(str(path))
        panel.load_from_config()
        panel.land_on_loaded_config()
        card = panel.tabs["start"].cards["Europa"]
        report = card.report_text()
        assert report, "the card said nothing about a file with 180 keys in it"
        assert ("does not show" in report
                or "Every setting in this file is shown" in report), report
        if "does not show" in report:
            assert "will be written back as they are" in report, report
        print(f"    [disclosure] {report.splitlines()[0]}")


def test_a_cards_exposed_keys_are_keys_the_engine_reads():
    """A card that shows a control for a key nothing reads is a card that lies."""
    from ptm.engine_config import known_keys
    known = set(known_keys())
    # GUI-synthetic keys that stand for real ones, and are translated at save
    # time by ConfigManager._config_for_yaml.
    synthetic = {"illumination.albedo_source", "mesh.topo_source"}
    with tempfile.TemporaryDirectory() as scratch:
        _, panel = _panel(scratch)
        stray = []
        for name, card in panel.tabs.items():
            if not name.startswith("card_"):
                continue
            for key in sorted(card.widgets):
                if key not in known and key not in synthetic:
                    stray.append(f"{name}: {key}")
        assert not stray, "\n  ".join(
            ["a card is bound to a key the engine does not read:"] + stray)


# --------------------------------------------------------------------------
# 5. Every card emits a configuration the ENGINE accepts
# --------------------------------------------------------------------------

def _engine():
    from gui_core.simulation_runner import SimulationRunner
    exe = SimulationRunner.find_ptm_executable()
    if not exe:
        return None
    try:
        help_text = subprocess.run(
            [str(exe), "--help"], capture_output=True, text=True,
            timeout=60).stdout
    except Exception:                                        # noqa: BLE001
        return None
    # A binary that predates --validate-config has no unknown-argument handler
    # and RUNS THE WHOLE SIMULATION when handed the flag. Measured. So the flag
    # is probed for before it is used, never assumed.
    return str(exe) if "--validate-config" in help_text else None


def test_every_card_emits_a_configuration_the_engine_accepts():
    """Every body x mode x run type, handed to the engine's own checker.

    The configuration is reset first, deliberately. This measures what a CARD
    produces, not what a card produces on top of whatever the previous check in
    this process left behind -- and the difference is real: an earlier check
    opens a Europa DEM configuration, which carries ``radiation.use_gpu: true``,
    and no card clears it because no card shows it. That is the cards behaving
    correctly (they write only what they show, and the out-of-card listing
    discloses the rest), and it would make this arm measure the wrong thing.
    """
    exe = _engine()
    if exe is None:
        raise Skip("no ptm_thermal_sim with --validate-config to ask")
    with tempfile.TemporaryDirectory() as scratch:
        manager, panel = _panel(scratch)
        manager.reset_to_defaults()
        panel.load_from_config()
        start = panel.tabs["start"]
        root = Path(REPO_ROOT)
        failures, checked = [], 0
        for body, mode, run_type in _routes():
            start.choose_body(body)
            start._choose_mode(mode)
            start.open_card(body, mode, run_type, apply_defaults=True)
            # A generic-body route needs a body to exist at all before the
            # engine will read generic_body:, so give it the minimum one.
            if body in ("asteroid", "custom"):
                manager.set_value("generic_body.name", f"probe_{body}")
                manager.set_value("generic_body.shape.type", "spherical")
                manager.set_value("generic_body.shape.radius_km", 10.0)
                manager.set_value("generic_body.orbit.semi_major_axis_au", 2.5)
                manager.set_value("generic_body.orbit.eccentricity", 0.1)
                manager.set_value("generic_body.spin.period_hours", 8.0)
            out = Path(scratch) / f"{body}_{mode}_{run_type}"
            manager.set_value("output.directory", str(out))
            config_path = Path(scratch) / f"{body}_{mode}_{run_type}.yaml"
            manager.save_run_yaml(str(config_path), root)
            result = subprocess.run(
                [exe, "--config", str(config_path), "--validate-config"],
                capture_output=True, text=True, cwd=str(root), timeout=180)
            checked += 1
            if result.returncode != 0:
                tail = (result.stdout + result.stderr).strip().splitlines()
                failures.append(
                    f"{body}/{mode}/{run_type}: rc={result.returncode} :: "
                    + " | ".join(tail[-3:]))
        print(f"    [engine] {checked - len(failures)}/{checked} card "
              f"configurations accepted by {Path(exe).name}")
        assert not failures, "\n  ".join(
            ["the engine refused a configuration a card produced:"] + failures)


# --------------------------------------------------------------------------
# 6. The generic-body round-trip floor (PTM-201)
# --------------------------------------------------------------------------

def test_a_surface_value_too_small_to_survive_is_refused():
    from gui_core.body_cards import (SURFACE_ROUND_TRIP_FLOOR,
                                     tiny_surface_refusals)
    below = SURFACE_ROUND_TRIP_FLOOR / 1.25
    config = {"generic_body": {"surface": {"ks": below, "kd": below,
                                           "rhos": 1500.0}}}
    refusals = tiny_surface_refusals(config)
    assert len(refusals) == 2, refusals
    for line in refusals:
        assert "0.000000" in line and "Europa" in line, line
    # ...and the negative control: ordinary values are not refused.
    ok = {"generic_body": {"surface": {"ks": 0.0096, "kd": 0.05,
                                       "rhos": 500.0, "cp0": 800.0}}}
    assert tiny_surface_refusals(ok) == [], tiny_surface_refusals(ok)
    # Zero means "not set" in this schema and must not be refused.
    zeroed = {"generic_body": {"surface": {"ks": 0.0, "kd": 0.0}}}
    assert tiny_surface_refusals(zeroed) == []


# --------------------------------------------------------------------------
# 7. Opening a file is not an edit to it
# --------------------------------------------------------------------------

def test_showing_a_card_does_not_rewrite_the_configuration():
    """The whole point of landing on a card, held to it.

    Three ways a card could edit a file merely by displaying it, all of which
    a draft of this code did:

    * ``set_bool`` calls ``setChecked`` without blocking signals, so every
      ``toggled`` handler fires on load -- and one of them writes a derived key
      (``radiation.rms_slope_deg`` follows ``radiation.beaming``);
    * the facet-scale gate turned an inherited ``beaming: true`` off;
    * a combo control silently skipped by the mirror would be written back at
      the next unrelated edit.

    The measurement is the whole engine-facing configuration, before and after.
    """
    import copy
    with tempfile.TemporaryDirectory() as scratch:
        manager, panel = _panel(scratch)
        path = Path(scratch) / "beaming.yaml"
        # A lunar crater with beaming ON at a NON-default slope, on a mesh path
        # that does not exist -- so the facet-scale rule cannot measure it and
        # would like to refuse. The value must survive anyway.
        path.write_text(
            "simulation:\n  body: Moon\n  start_time: '2026-01-01T00:00:00'\n"
            "  duration_days: 29.530588\n  timesteps: 144\n"
            "mesh:\n  import_file: /nowhere/crater.ply\n"
            "  latitude_override: -85.0\n  longitude_override: 31.0\n"
            "radiation:\n  enabled: true\n  beaming: true\n"
            "  rms_slope_deg: 25.0\n"
            "illumination:\n  albedo_source: none\n  use_albedo_map: false\n")
        manager.load_from_yaml(str(path))
        panel.load_from_config()
        before = copy.deepcopy(manager._config_for_yaml())
        route = panel.land_on_loaded_config()
        assert route.ok and route.body_key == "Moon" and route.mode_key == "local"
        after = manager._config_for_yaml()
        assert after == before, (
            "landing on a card changed the configuration. Differences:\n  "
            + "\n  ".join(
                f"{k}: {before.get(k)!r} -> {after.get(k)!r}"
                for k in sorted(set(before) | set(after))
                if before.get(k) != after.get(k)))
        assert manager.get_value("radiation.rms_slope_deg") == 25.0, (
            "the file's own sub-facet slope was overwritten by the card's")
        assert manager.get_value("radiation.beaming") is True, (
            "the card turned the file's beaming off; a card gates its own "
            "control, it does not edit the user's configuration on open")
        # ...and the card says which of the three states the rule is in. This
        # mesh does not exist, so the rule cannot MEASURE it -- which is not
        # the same as "too fine", and saying so is the point. The refusal state
        # itself is exercised at module level by
        # `test_gui_beaming_scale_rule.py`, which has meshes to measure.
        card = panel.tabs["start"].cards["Moon"]
        note = card.beaming_note.text()
        assert "COULD NOT BE MEASURED" in note, (
            f"the card reported an unmeasurable mesh as checked: {note!r}")
        assert "nothing here has checked this setting for you" in note, note


def test_a_card_mirrors_its_combo_controls():
    """A combo the mirror skips shows a setting the run does not have."""
    with tempfile.TemporaryDirectory() as scratch:
        manager, panel = _panel(scratch)
        start = panel.tabs["start"]
        path = Path(scratch) / "albedo.yaml"
        for source in ("none", "builtin"):
            path.write_text(
                f"simulation:\n  body: Moon\n"
                f"illumination:\n  albedo_source: {source}\n"
                f"  use_albedo_map: {'false' if source == 'none' else 'true'}\n")
            manager.load_from_yaml(str(path))
            panel.load_from_config()
            start.open_card("Moon", "global", "time_series",
                            apply_defaults=False)
            combo = start.cards["Moon"].widgets["illumination.albedo_source"]
            assert combo.currentText() == source, (
                f"the card shows albedo source {combo.currentText()!r} for a "
                f"configuration that says {source!r}")


def test_the_landing_screen_says_why_a_file_has_no_card():
    """And says it in its OWN message, not in another control's report."""
    with tempfile.TemporaryDirectory() as scratch:
        manager, panel = _panel(scratch)
        path = Path(scratch) / "mercury_mesh.yaml"
        # Mercury on an imported mesh: the engine runs it, the card does not
        # describe it. That must be said, not guessed past.
        path.write_text(
            "simulation:\n  body: Mercury\n"
            "mesh:\n  import_file: /nowhere/mercury_crater.ply\n")
        manager.load_from_yaml(str(path))
        panel.load_from_config()
        route = panel.land_on_loaded_config()
        assert not route.ok, "a Mercury imported-mesh run was routed to a card"
        start = panel.tabs["start"]
        assert start.stack.currentIndex() == start.LANDING
        message = start.landing_message_text()
        assert "Mercury" in message and "All settings" in message, message
        assert start.report_text("reproduce_baseline") == "" or \
            "Mercury" not in start.report_text("reproduce_baseline"), (
            "the routing message landed in the baseline entry's report box, "
            "which answers a different question")


def test_the_shape_types_are_the_engines_own_spellings():
    """A shape type the engine does not branch on runs as a sphere, silently.

    The parser stores ``generic_body.shape.type`` VERBATIM and validates
    nothing (``config_parser.cpp:688``); the solver tests it against two string
    literals and falls through to a sphere for anything else. So a control
    offering "mesh" or "ellipsoid" -- both perfectly natural words, and both in
    the first draft of the asteroid card -- writes a configuration whose shape
    model is never read.
    """
    from gui_core.body_cards import GENERIC_MESH_SHAPE, GENERIC_SHAPE_TYPES

    main = (REPO_ROOT / "src" / "cpp" / "main"
            / "ptm_thermal_sim.cpp").read_text()
    for literal in ("custom_mesh", "ellipsoidal"):
        assert f'shape_type == "{literal}"' in main, (
            f"the engine no longer branches on shape_type == {literal!r}; "
            f"GENERIC_SHAPE_TYPES is describing a simulator that has moved")
    assert GENERIC_MESH_SHAPE == "custom_mesh"
    assert set(GENERIC_SHAPE_TYPES) == {"spherical", "ellipsoidal",
                                        "custom_mesh"}, GENERIC_SHAPE_TYPES

    with tempfile.TemporaryDirectory() as scratch:
        _, panel = _panel(scratch)
        # The card and the Body page must offer the SAME set: two lists is two
        # answers, and the panel-wide one was already right.
        card = panel.tabs["card_asteroid"].widgets["generic_body.shape.type"]
        page = panel.tabs["body_definition"].widgets["generic_body.shape.type"]
        card_items = {card.itemText(i) for i in range(card.count())}
        page_items = {page.itemText(i) for i in range(page.count())}
        assert card_items == page_items == set(GENERIC_SHAPE_TYPES), (
            f"card offers {sorted(card_items)}, the Body page offers "
            f"{sorted(page_items)}")


def test_choosing_a_shape_model_reaches_the_mesh_the_solver_loads():
    """`generic_body.shape.mesh_file` is metadata; `mesh.import_file` is not."""
    with tempfile.TemporaryDirectory() as scratch:
        manager, panel = _panel(scratch)
        manager.reset_to_defaults()
        panel.load_from_config()
        start = panel.tabs["start"]
        card = start.open_card("asteroid", "body", "time_series",
                               apply_defaults=True)
        card.widgets["generic_body.name"].setText("probe")
        card.widgets["generic_body.shape.type"].setCurrentText("custom_mesh")
        card.widgets["generic_body.shape.mesh_file"].setText(
            "data/shapes/probe.obj")
        card.widgets["generic_body.shape.mesh_scale"].setValue(2.0)
        assert manager.get_value("mesh.import_file") == "data/shapes/probe.obj", (
            "the shape model did not reach mesh.import_file, so the run would "
            "use the procedural sphere and never read it")
        assert manager.get_value("mesh.scale_factor") == 2.0

        # ...and the negative control: a spherical body must not clobber a
        # mesh path the user set elsewhere.
        manager.set_value("mesh.import_file", "someone/elses.ply")
        card.widgets["generic_body.shape.type"].setCurrentText("spherical")
        card.widgets["generic_body.shape.mesh_file"].setText("other.obj")
        assert manager.get_value("mesh.import_file") == "someone/elses.ply", (
            "a spherical shape overwrote the top-level mesh path")


def test_a_generic_body_value_below_the_floor_blocks_the_launch():
    """The refusal has to REFUSE, not merely be printed on a card."""
    with tempfile.TemporaryDirectory() as scratch:
        manager, panel = _panel(scratch)
        manager.reset_to_defaults()
        manager.set_value("generic_body.name", "probe")
        manager.set_value("generic_body.surface.ks", 4e-7)
        ok, messages = manager.validate()
        assert not ok, (
            "a configuration whose conductivity cannot reach the solver "
            "passed validation, so the launch would proceed and the run would "
            "silently use a placeholder property")
        assert any("generic_body.surface.ks" in m for m in messages), messages
        manager.set_value("generic_body.surface.ks", 0.0096)
        ok, _ = manager.validate()
        assert ok, "an ordinary conductivity is now refused"


TESTS = [
    test_the_landing_screen_offers_exactly_the_implemented_bodies,
    test_a_body_button_carries_one_short_functional_line_and_no_paragraph,
    test_removing_the_paragraphs_did_not_remove_a_disclosure,
    test_the_landing_screen_is_buttons_not_prose,
    test_a_body_the_presentation_table_forgets_is_loud,
    test_a_grid_is_offered_only_where_a_grid_exists,
    test_every_route_reaches_its_own_card,
    test_choosing_a_run_type_writes_and_reports_it,
    test_the_mode_choice_changes_which_run_types_are_offered,
    test_open_refuses_a_file_that_is_not_yaml,
    test_open_refuses_yaml_without_a_body,
    test_open_accepts_every_shipped_run_configuration,
    test_every_shipped_configuration_routes_to_a_card_or_says_why,
    test_loading_a_config_lands_on_its_own_card,
    test_an_opened_file_lists_what_the_card_does_not_show,
    test_a_cards_exposed_keys_are_keys_the_engine_reads,
    test_every_card_emits_a_configuration_the_engine_accepts,
    test_a_surface_value_too_small_to_survive_is_refused,
    test_showing_a_card_does_not_rewrite_the_configuration,
    test_a_card_mirrors_its_combo_controls,
    test_the_landing_screen_says_why_a_file_has_no_card,
    test_a_generic_body_value_below_the_floor_blocks_the_launch,
    test_the_shape_types_are_the_engines_own_spellings,
    test_choosing_a_shape_model_reaches_the_mesh_the_solver_loads,
]


def main():
    failures = skipped = 0
    for test in TESTS:
        try:
            test()
        except Skip as exc:
            skipped += 1
            print(f"SKIP {test.__name__}\n     {exc}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {test.__name__}\n     {exc}")
        except BaseException as exc:                        # noqa: BLE001
            failures += 1
            print(f"ERROR {test.__name__}: {type(exc).__name__}: {exc}")
        else:
            print(f"PASS {test.__name__}")
    total = len(TESTS)
    print(f"\n[test_gui_body_first] {total - failures - skipped}/{total} "
          f"passed, {skipped} skipped")
    print(f"[test_gui_body_first] {'FAIL' if failures else 'PASS'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
