#!/usr/bin/env python3
"""The body-first surface: the flow, the routing, and what each card emits.

Seven properties, in the order they would break:

1. **The buttons are the engine's.** The landing screen offers exactly the
   bodies ``planetary_properties.h`` implements, plus the two generic-body
   routes. A body added or removed in C++ changes this screen through the
   generated table, not through somebody remembering.
1b. **A header names a thing; it does not ask a question.** Section headers and
   control labels on these screens are technical terms -- "Local mesh", not
   "Which piece of ground". Tooltips are deliberately exempt: explanatory prose
   is where a question is the right shape.
1c. **A control is filed under what it does.** ``thermal.use_gpu`` selects the
   solver's device, so it is in the Thermal section, not beside
   ``output.directory``.
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
import re
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


def test_a_body_button_is_the_body_name_and_nothing_else():
    """The landing screen offers a choice; it does not comment on it.

    Three rounds of copy came off these buttons. First a paragraph of muted
    prose under each -- one opened "The honest destination for a world the
    engine has no built-in properties for", the register of a session note in
    front of somebody who has never read one. Then, on 2026-08-27, the one-line
    subtitle that replaced it. Then, the same day, Mercury's two-word "(limited
    data)" badge, which was the last thing making one button taller than the
    other four. What is left is the body's name, on every one of the five.

    Three properties, and the third is the one that makes the first two hold:

    * the TABLE carries the name and no prose -- `BodyButton` has no `note`,
      `subtitle` or `maturity` field, so any of the three put back into
      `BODY_PRESENTATION` or `CUSTOM_ROUTES` is a TypeError at import rather
      than text nobody sees;
    * the name is the body's OWN capitalization, not shouted;
    * the RENDERED button carries EXACTLY ONE line -- Mercury included, which
      is the arm this round added. Measured on the landing page, because copy
      can reach a button from somewhere other than the table.

    The disclosure the badge used to carry has NOT gone: see
    `test_the_mercury_data_caveat_lives_on_the_mercury_card`, which fails if it
    vanishes from the card.
    """
    from gui_core.body_cards import body_buttons

    expected = {"Moon": "Moon", "Europa": "Europa", "Mercury": "Mercury",
                "asteroid": "Asteroid", "custom": "Custom body"}
    for button in body_buttons():
        for gone in ("note", "subtitle", "maturity"):
            assert not hasattr(button, gone), (
                f"BodyButton grew a `{gone}` field back ({button.key}). The "
                f"landing screen does not render one, so it would be text "
                f"nobody sees.")
        assert button.title == expected[button.key], (
            f"the {button.key} button says {button.title!r}; the landing "
            f"screen shows the body's name as {expected[button.key]!r}")

    # ...and on the rendered page, where a second line could come from any
    # widget rather than from the table.
    with tempfile.TemporaryDirectory() as scratch:
        _, panel = _panel(scratch)
        from PyQt6.QtWidgets import QPushButton
        start = panel.tabs["start"]
        page = start.stack.widget(start.LANDING)
        titles = set(expected.values())
        found, wordy = {}, []
        for widget in page.findChildren(QPushButton):
            lines = [ln for ln in widget.text().split("\n") if ln.strip()]
            if not lines or lines[0] not in titles:
                continue
            found[lines[0]] = lines
            # EXACTLY one line. No badge exception any more: that exception is
            # what let "Mercury\n(limited data)" through, and Mercury is a bare
            # name now like the other four.
            if len(lines) != 1:
                wordy.append(f"{lines[0]}: {widget.text()!r}")
        assert not wordy, "\n  ".join(
            ["a body button carries copy under its name again:"] + wordy)
        assert set(found) == titles, (
            f"the landing page renders buttons for {sorted(found)}; the table "
            f"offers {sorted(titles)}")
        print("    [landing copy] "
              + "; ".join(f"{k}: {v}" for k, v in sorted(found.items())))


#: What the Mercury data caveat has to say, in whatever words it says it. Three
#: claims, because a user who reads only two of them still comes away thinking
#: this body is supported the way the Moon is.
_MERCURY_CAVEAT = ("no albedo map", "no topography", "uniform-albedo")


def _mercury_caveat_missing(card):
    """Which parts of the data caveat this card does not say. [] means all
    three are on it, on a widget that is not hidden."""
    from PyQt6.QtWidgets import QLabel
    # `isHidden()`, NOT `isVisible()`: offscreen, `isVisible()` is False for
    # every widget whose ancestors were never shown, so a visibility check
    # written that way passes vacuously on this machine.
    text = "\n".join(w.text() for w in card.findChildren(QLabel)
                     if not w.isHidden())
    return [phrase for phrase in _MERCURY_CAVEAT if phrase not in text]


def test_the_mercury_data_caveat_lives_on_the_mercury_card():
    """The half that makes the button sweep safe rather than merely tidy.

    Mercury's landing-screen copy was the only one carrying a claim a user
    needs: it ships no albedo map, no topography and no worked example, so
    every Mercury run is a uniform-albedo sphere. That claim has now been
    displaced twice -- out of the paragraph under the button, then out of the
    two-word "(limited data)" badge that replaced the paragraph. The button is
    a bare name, so THE CARD IS THE ONLY PLACE LEFT, and this is the check that
    makes each of those deletions a move rather than a loss.

    The negative control is the point of the exercise: the caveat is blanked on
    the live card and the same predicate must complain. Without it this is an
    assertion that a string is present in a haystack of every label on the
    card, which is exactly the shape that passes for the wrong reason.

    The custom-body paragraph is the same shape: what that route is for, and
    what happens if you type the body's name under `simulation.body` instead,
    is on the card.
    """
    from gui_core.body_cards import body_buttons, card_preset

    mercury = [b for b in body_buttons() if b.key == "Mercury"]
    assert mercury, "the Mercury button is gone"

    with tempfile.TemporaryDirectory() as scratch:
        _, panel = _panel(scratch)
        card = panel.tabs["card_mercury"]
        banner = getattr(card, "data_support_banner", None)
        assert banner is not None, (
            "the Mercury card has no `data_support_banner`. The landing screen "
            "no longer carries a badge or a paragraph for this body, so there "
            "is now nowhere at all that says it ships no data.")
        assert not banner.isHidden(), (
            "the Mercury data-support banner is hidden, which is the same as "
            "not being there")

        missing = _mercury_caveat_missing(card)
        assert not missing, (
            f"the Mercury card no longer says {missing}, and neither the "
            f"landing paragraph nor the badge that used to say it still "
            f"exists:\n{banner.text()[:400]}")

        # ...the negative control. Blank the banner and the predicate must
        # complain about all three claims; then put it back.
        original = banner.text()
        try:
            banner.setText("")
            blanked = _mercury_caveat_missing(card)
            assert blanked == list(_MERCURY_CAVEAT), (
                f"with the banner blanked the check still reports only "
                f"{blanked} missing -- something else on the card is "
                f"satisfying it, so this check does not measure the banner")
        finally:
            banner.setText(original)
        assert _mercury_caveat_missing(card) == [], (
            "the negative control did not restore the banner")
        print(f"    [disclosure] Mercury caveat on card_mercury, "
              f"{len(original)} chars, all of {list(_MERCURY_CAVEAT)} present; "
              f"blanking it fails the check")

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


# --------------------------------------------------------------------------
# 1b. REGISTER -- a header names a thing, it does not ask a question
# --------------------------------------------------------------------------
#
# Paul, click-testing the merged body-first GUI: "we don't need to use informal
# language like 'which piece of ground' (users are planetary scientists) so
# change these to be more technically accurate and concise, e.g., 'Local mesh'."
#
# Every section header on these screens was phrased as a question -- "Which
# sphere", "Which piece of ground", "Where it goes", "How long to settle",
# "What kind of run". A question is the register of a wizard walking somebody
# through a form; a planetary scientist reading a section header wants the name
# of the thing.
#
# SCOPE, deliberately narrow, and each exclusion is a measurement rather than a
# convenience:
#
#   * covered -- QGroupBox titles, checkbox, radio and PUSH-BUTTON labels,
#     field labels and the flow's own headings, on the Start page and the five
#     body cards. Push buttons are in on purpose: the sweep renamed "Prepare
#     the grid..." and both Back controls, and a gate blind to buttons would
#     let "Where should the grid go?" back in green.
#   * NOT covered -- TOOLTIPS. The plain-language what/recommended/when
#     standard stays exactly as it is: explanatory prose is where a question is
#     the right shape ("Whether to model the whole body as a sphere, or one
#     site from a terrain mesh you supply"). This gate would have 46 hits on
#     the tooltip table and every one of them would be a wrong fix.
#   * NOT covered -- the muted notes under controls, which are prose for the
#     same reason, and modal dialog titles, where "Write grid configurations?"
#     is a real question being asked of the user.
#   * NOT covered -- the eleven All settings section pages. Measured on the
#     2026-08-27 corpus: 31 group titles, 0 hits. They were already technical
#     ("Time Parameters", "Output Destination", "Thermal GPU Dispatch"), which
#     is why the complaint was about the cards.

_INTERROGATIVE = re.compile(r"^(Which|What|Where|When|How|Why|Who)\b", re.I)


def _register_corpus(panel):
    """(where, text) for every header and label on the body-first surface."""
    from PyQt6.QtWidgets import (QCheckBox, QGroupBox, QLabel, QPushButton,
                                 QRadioButton)
    out = []
    surfaces = {name: tab for name, tab in panel.tabs.items()
                if name == "start" or name.startswith("card_")}
    assert surfaces, "no body-first surfaces found -- the probe is broken"
    for name, tab in surfaces.items():
        for widget in tab.findChildren(QGroupBox):
            if widget.title().strip():
                out.append((f"{name}/section", widget.title()))
        for widget in tab.findChildren((QCheckBox, QRadioButton)):
            if widget.text().strip():
                out.append((f"{name}/control", widget.text()))
        for widget in tab.findChildren(QPushButton):
            # The FIRST line only. A run-type button carries its summary on the
            # lines below, and a summary is prose, checked nowhere here.
            text = (widget.text() or "").split("\n")[0].strip()
            if text:
                out.append((f"{name}/button", text))
        for widget in tab.findChildren(QLabel):
            text = (widget.text() or "").strip()
            # Field labels only. The muted note paragraphs are prose and are
            # excluded by construction, not by a wordlist.
            if text.endswith(":"):
                out.append((f"{name}/label", text))
    start = panel.tabs["start"]
    for attr in ("landing_heading", "runtype_heading", "card_title"):
        widget = getattr(start, attr, None)
        text = (widget.text() or "").strip() if widget is not None else ""
        if text:
            out.append((f"start/{attr}", text))
    return out


def test_no_header_or_label_on_a_card_is_phrased_as_a_question():
    with tempfile.TemporaryDirectory() as scratch:
        _, panel = _panel(scratch)
        # Drive the flow so the run-type step's own headers are built and the
        # card header carries a title: an unvisited screen would pass empty.
        start = panel.tabs["start"]
        start.choose_body("Moon")
        start._choose_mode("local")
        start.open_card("Moon", "local", "time_series", apply_defaults=False)

        corpus = _register_corpus(panel)
        assert len(corpus) > 60, (
            f"the register corpus is only {len(corpus)} strings -- the "
            f"harvest is broken and this check would pass vacuously")
        hits = [f"{where}: {text!r}" for where, text in corpus
                if _INTERROGATIVE.match(text) or text.rstrip().endswith("?")]
        assert not hits, "\n  ".join(
            [f"a section header or label on the body-first surface is phrased "
             f"as a question ({len(hits)} of {len(corpus)}); name the thing "
             f"instead -- 'Local mesh', not 'Which piece of ground':"] + hits)
        print(f"    [register] {len(corpus)} headers and labels across the "
              f"Start page and five cards, 0 phrased as questions")
        start.stack.setCurrentIndex(start.LANDING)


def test_the_register_check_actually_fires():
    """Its negative controls -- the exact strings this sweep replaced.

    A pattern nobody proved can fire is decoration. Each of these was a real
    section header on the merged body-first GUI on 2026-08-27.
    """
    was_real = ["Which sphere", "Which piece of ground", "Which terrain",
                "Where it goes", "How long to settle", "What kind of run",
                "Which part of it", "Where it is", "How it turns",
                "What shape it is", "What the surface is like", "Which body",
                "When", "When to run it", "What would you like to model?"]
    for text in was_real:
        assert (_INTERROGATIVE.match(text)
                or text.rstrip().endswith("?")), (
            f"the register check does not fire on {text!r}, which was a real "
            f"header before this sweep")
    # ...and the positive control: the replacements pass clean.
    replacements = ["Global mesh", "Local mesh", "Local mesh (DEM)",
                    "Output", "Thermal", "Run type", "Model domain",
                    "Ephemeris", "Spin state", "Shape", "Surface properties",
                    "Body identity", "Time window", "Select a body",
                    "Parameter grid", "Radiation exchange",
                    "Subdivision level:", "Duration (days):",
                    "Output directory:", "Integration window (days):"]
    for text in replacements:
        assert not (_INTERROGATIVE.match(text)
                    or text.rstrip().endswith("?")), (
            f"the register check fires on {text!r}, which is the replacement")


# --------------------------------------------------------------------------
# 1c. A control is filed under what it does, not under what it sits beside
# --------------------------------------------------------------------------

def _owning_section(widget):
    """The QGroupBox title a control is inside, or ""."""
    from PyQt6.QtWidgets import QGroupBox
    parent = widget.parentWidget()
    while parent is not None:
        if isinstance(parent, QGroupBox):
            return parent.title()
        parent = parent.parentWidget()
    return ""


def _misfiled(key, section):
    """A solver setting sitting under a destination or a mesh, or ""."""
    if not key.startswith("thermal."):
        return ""
    if section == "Output":
        return f"{key} is in the Output section"
    # Any mesh section, global OR local. `thermal.integration_days` would land
    # in the LOCAL group if it were re-added to the Europa card, and matching
    # only "Global mesh" would miss it.
    if "mesh" in section.lower():
        return f"{key} is in the {section!r} section"
    return ""


def test_the_misfiled_control_check_actually_fires():
    """Its negative controls. Two of the three arms below cannot fire on the
    current GUI, because the current GUI is correct -- so they are exercised
    against the predicate directly rather than shipped unproven.
    """
    fires = [("thermal.use_gpu", "Output"),                 # the real defect
             ("thermal.integration_days", "Global mesh"),   # where it was
             ("thermal.integration_days", "Local mesh (DEM)"),
             ("thermal.equilibration_orbits", "Local mesh")]
    for key, section in fires:
        assert _misfiled(key, section), (
            f"the misfiled check does not fire on {key} in {section!r}")
    clean = [("thermal.use_gpu", "Thermal"),
             ("thermal.equilibration_orbits", "Thermal"),
             ("thermal.integration_days", "Thermal"),
             ("output.directory", "Output"),
             ("mesh.resolution", "Global mesh"),
             ("mesh.import_file", "Local mesh"),
             ("radiation.enabled", "Radiation exchange")]
    for key, section in clean:
        assert not _misfiled(key, section), (
            f"the misfiled check fires on {key} in {section!r}, which is "
            f"where it belongs")


def test_the_thermal_solver_controls_are_in_the_thermal_section():
    """`thermal.use_gpu` sat in the output section, beside output.directory.

    It selects the device the heat-equation solver runs on. Nothing about it is
    a destination, and a user looking for it under Output found it only by
    accident. Two properties, and the second is the one that would have failed
    before this change:

    * every card has a Thermal section, and both solver settings are in it;
    * NO `thermal.*` control is inside the Output section -- which is where
      `thermal.use_gpu` was, on every one of the five cards.

    Also pinned: `output.directory` stays in Output. Moving the GPU switch out
    must not take the destination with it.
    """
    with tempfile.TemporaryDirectory() as scratch:
        _, panel = _panel(scratch)
        cards = {n: t for n, t in panel.tabs.items() if n.startswith("card_")}
        assert len(cards) == 5, f"expected five cards, found {sorted(cards)}"
        misfiled, missing = [], []
        for name, card in cards.items():
            sections = {key: _owning_section(w)
                        for key, w in card.widgets.items()}
            for key in ("thermal.use_gpu", "thermal.equilibration_orbits"):
                if sections.get(key) != "Thermal":
                    missing.append(f"{name}: {key} is in "
                                   f"{sections.get(key)!r}, not 'Thermal'")
            if sections.get("output.directory") != "Output":
                missing.append(f"{name}: output.directory is in "
                               f"{sections.get('output.directory')!r}, not "
                               f"'Output'")
            for key, section in sections.items():
                complaint = _misfiled(key, section)
                if complaint:
                    misfiled.append(f"{name}: {complaint}")
        assert not missing, "\n  ".join(
            ["a solver or destination control is not in its own section:"]
            + missing)
        assert not misfiled, "\n  ".join(
            ["a thermal-solver control is filed under where the run's files "
             "go, or under the mesh geometry:"] + misfiled)
        # Europa's integration window moved out of the mesh section, and then
        # out of Thermal as well: on the integrated run type it is THE duration
        # control -- the only one that card offers -- so it belongs with the
        # other durations. What must never come back is the mesh section, and
        # `_misfiled` above still holds that line for every `thermal.*` key on
        # every card, so this assertion pins the destination rather than
        # re-checking the prohibition.
        europa = panel.tabs["card_europa"]
        assert _owning_section(
            europa.widgets["thermal.integration_days"]) == "Time window", (
            "thermal.integration_days is not in the Time window section, "
            "where the duration controls live")
        print(f"    [sections] {len(cards)} cards: thermal.use_gpu, "
              f"thermal.equilibration_orbits in Thermal; output.directory in "
              f"Output; integration_days in Time window; 0 thermal keys "
              f"under Output or a mesh section")


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
        f"shipped corpus contains is the Europa DEM one")
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
# 3b. The click-through flow IS the navigation; the rail is a second level
# --------------------------------------------------------------------------
#
# Paul, on the merged body-first GUI: "the tabs ('Simulation', 'Mesh',
# 'Properties', etc) on the lefthand side are somewhat obsolete and confusing.
# The GUI should embrace the click-through structure, and only offer these
# other features as advanced options."
#
# Four properties, in the order they would break:
#
#   1. the rail is not on screen at launch;
#   2. an explicit, clearly-secondary control brings it up, and it exists in
#      more than one place -- the card, the landing screen and the View menu;
#   3. leaving it returns to WHERE THE FLOW WAS, not to the top of it;
#   4. the round trip does not lose or corrupt state in either direction.
#
# (4) is the one worth writing carefully. A value edited on a section page is
# in the configuration but not on the card's controls, and `BaseConfigTab`
# writes a control's value back on the next widget event -- so a card left
# showing a stale number would push it over the edit the moment anything else
# on the card was touched. That is the failure this arm measures directly.

def test_the_section_rail_is_not_the_primary_navigation():
    """It is hidden at launch, and the flow is what is on screen instead."""
    with tempfile.TemporaryDirectory() as scratch:
        _, panel = _panel(scratch)
        panel.show_body_first()
        assert not panel.all_settings_visible(), (
            "the eleven-section rail is on screen at launch; the click-through "
            "flow is supposed to be the only primary navigation")
        assert panel.nav_column.isHidden(), (
            "nav_column is not hidden, so the Simulation/Mesh/Properties list "
            "is still the first thing a user sees")
        assert panel.stacked_widget.currentWidget() is panel.tabs["start"], (
            "the panel does not open on the body-first Start page")
        # Hidden, NOT removed: every section still exists, still loads and
        # still saves, so no written configuration depends on this.
        for key in ("simulation", "mesh", "thermophysical", "radiation",
                    "illumination", "output", "gpu", "engine_flags"):
            assert key in panel.tabs, f"the {key} page was deleted, not hidden"
            assert panel.stacked_widget.indexOf(panel.tabs[key]) >= 0, (
                f"the {key} page is no longer in the stack")
        print(f"    [nav] rail hidden at launch; "
              f"{len(panel._nav_keys)} sections still present and loadable")


def test_all_settings_is_reachable_and_clearly_secondary():
    """From the card, from the landing screen, and from the View menu.

    A single entry point would be one misclick from a user who cannot find
    the settings they know exist -- which is the complaint the rail was hiding
    in the first place, inverted.
    """
    with tempfile.TemporaryDirectory() as scratch:
        _, panel = _panel(scratch)
        from PyQt6.QtWidgets import QPushButton
        start = panel.tabs["start"]

        buttons = [b for b in start.findChildren(QPushButton)
                   if b.objectName() == "AllSettingsButton"]
        assert len(buttons) >= 2, (
            f"only {len(buttons)} 'All settings' control(s) on the Start page; "
            f"the card header and the landing screen must each carry one")
        for button in buttons:
            assert "advanced" in button.text().lower(), (
                f"the control says {button.text()!r}, which does not say it is "
                f"the advanced view")
            assert button.toolTip(), "the All settings control ships bare"

        # It opens the second level, and lands somewhere recognizable.
        panel.show_body_first()
        assert not panel.all_settings_visible()
        panel.show_all_settings()
        assert panel.all_settings_visible(), (
            "'All settings' did not reveal the section rail")
        assert panel.stacked_widget.currentWidget() is panel.tabs["simulation"], (
            "All settings did not land on Simulation")
        # ...and the way back exists, is labeled, and closes it again.
        assert not panel.close_all_settings_button.isHidden(), (
            "there is no visible control to leave All settings")
        assert panel.close_all_settings_button.toolTip(), (
            "the way out of All settings ships bare")
        panel.close_all_settings_button.click()
        assert not panel.all_settings_visible(), (
            "the Back control did not close the section rail")


def test_the_rails_own_start_row_leaves_the_rail():
    """There must not be two navigations on screen at the same time.

    The section list still carries a "Start" row -- it has to, because a row's
    index IS its page's index in the stack and dropping the row would shift
    every section below it. Selecting it used to show the click-through flow
    with the section list still up, which is precisely the "obsolete and
    confusing" arrangement this change removes. It now does what Back does.

    Also the negative control on the recursion guard: `show_body_first` sets
    the rail's row back to Start, which re-enters the same handler.
    """
    with tempfile.TemporaryDirectory() as scratch:
        manager, panel = _panel(scratch)
        manager.reset_to_defaults()
        panel.load_from_config()
        start = panel.tabs["start"]
        start.open_card("Moon", "global", "time_series", apply_defaults=True)
        panel.show_all_settings()
        assert panel.all_settings_visible()
        assert panel.stacked_widget.currentWidget() is panel.tabs["simulation"]

        start_row = panel._nav_keys.index("start")
        panel.nav_rail.setCurrentRow(start_row)
        assert not panel.all_settings_visible(), (
            "selecting 'Start' in the section list showed the flow with the "
            "section list still on screen")
        assert panel.stacked_widget.currentWidget() is panel.tabs["start"]
        assert start.stack.currentIndex() == start.CARD, (
            "leaving through the rail's Start row did not return to the card")

        # ...and every OTHER row still just switches the page.
        panel.show_all_settings()
        for key in ("mesh", "thermophysical", "radiation"):
            row = panel._nav_keys.index(key)
            panel.nav_rail.setCurrentRow(row)
            assert panel.stacked_widget.currentWidget() is panel.tabs[key], (
                f"selecting {key!r} in the section list did not show it")
            assert panel.all_settings_visible(), (
                f"selecting {key!r} closed the section list")
        panel.show_body_first()


def test_a_hidden_section_does_not_eject_the_user_from_all_settings():
    """`_apply_tab_visibility` hides the Body page when no generic body is
    active, and then jumps off it. Jumping to "Start" would close All settings
    out from under a user who only changed a body.
    """
    with tempfile.TemporaryDirectory() as scratch:
        manager, panel = _panel(scratch)
        manager.reset_to_defaults()
        manager.set_value("generic_body.name", "probe")
        panel.load_from_config()
        panel.show_all_settings()
        assert manager.generic_body_active(), "the fixture has no generic body"
        row = panel._nav_keys.index("body_definition")
        panel.nav_rail.setCurrentRow(row)
        assert panel.stacked_widget.currentWidget() is panel.tabs["body_definition"]

        # Take the generic body away: the Body page becomes irrelevant.
        manager.set_value("generic_body.name", "")
        manager.set_value("generic_body.config_file", "")
        manager.set_value("generic_body.orbit.semi_major_axis_au", 0.0)
        panel._apply_tab_visibility()
        assert panel.all_settings_visible(), (
            "hiding the Body page closed All settings; the user asked to "
            "change a body, not to leave the settings view")
        assert panel.stacked_widget.currentWidget() is not panel.tabs["start"], (
            "hiding the Body page fell back to the click-through flow instead "
            "of to another section")
        panel.show_body_first()


def test_leaving_all_settings_returns_to_the_same_card():
    """Not to the top of the flow. Returning to the landing screen would throw
    away the body and run type the user had already chosen."""
    with tempfile.TemporaryDirectory() as scratch:
        manager, panel = _panel(scratch)
        manager.reset_to_defaults()
        panel.load_from_config()
        start = panel.tabs["start"]
        for body, mode, run_type in (("Moon", "local", "time_series"),
                                     ("Europa", "global", "integrated"),
                                     ("asteroid", "body", "time_series")):
            start.choose_body(body)
            start._choose_mode(mode)
            card = start.open_card(body, mode, run_type, apply_defaults=True)
            panel.show_all_settings()
            assert panel.all_settings_visible()
            # The label names where it goes back to, so it is not a mystery.
            assert body in panel.close_all_settings_button.text(), (
                f"the Back control says "
                f"{panel.close_all_settings_button.text()!r} while the flow is "
                f"on {body}")
            panel.show_body_first()
            assert start.stack.currentIndex() == start.CARD, (
                f"returning from All settings left the {body} flow on step "
                f"{start.stack.currentIndex()}, not on its card")
            assert start.card_stack.currentWidget() is card, (
                f"returning from All settings showed a different card than the "
                f"{body} one the user left")
            assert start.route() == (body, mode, run_type), (
                f"the flow's route changed across the round trip: "
                f"{start.route()} != {(body, mode, run_type)}")


def test_an_edit_in_all_settings_reaches_the_card_and_survives_the_return():
    """The state half of the round trip.

    A number typed on the Simulation page must be showing on the card when the
    user comes back. Without `refresh_active_card()` in `show_body_first` this
    arm fails with "the card came back showing 29.530588 for a duration the
    user changed to 12.75" -- measured, by disabling the call.

    THE BLAST RADIUS OF A STALE CARD WAS MEASURED, and it is smaller than it
    first looks. The obvious worry is that touching an unrelated control on a
    stale card writes its stale siblings back over the edit. It does not: with
    the re-mirror disabled, editing `simulation.timesteps` on a card still
    showing 29.530588 left `simulation.duration_days` at 12.75, because Qt
    wires each control to its own key and `on_widget_changed` writes only the
    one that fired. So the third arm below is a NEGATIVE CONTROL on that --
    stated as one, not dressed up as a corruption guard it cannot be. What a
    stale card actually costs is a user reading a wrong number and acting on
    it, which is what the second arm holds.
    """
    with tempfile.TemporaryDirectory() as scratch:
        manager, panel = _panel(scratch)
        manager.reset_to_defaults()
        panel.load_from_config()
        start = panel.tabs["start"]
        card = start.open_card("Moon", "global", "time_series",
                               apply_defaults=True)

        # --- card -> All settings: the section page shows what the card set.
        #
        # Compared to the PAGE'S OWN display precision, deliberately, and this
        # is not a loosened assertion -- it is a measured pre-existing gap that
        # this branch did not introduce and does not fix. The Simulation page's
        # duration box carries four decimals (config_panel.py, setDecimals(4),
        # unchanged since 815c6ad) while the value and the body cards carry
        # six, so a lunation reads 29.5306 on the page and 29.530588 in the
        # configuration. Measured consequence, filed rather than fixed here:
        # stepping that box up and then back down writes 29.5306 to the
        # configuration -- a silent 4.5-second change to the run. Touching an
        # unrelated control does NOT do it, which is the property the arm below
        # measures and which does hold.
        panel.show_all_settings()
        sim = panel.tabs["simulation"]
        shown = sim.widgets["simulation.duration_days"].value()
        stored = manager.get_value("simulation.duration_days")
        places = sim.widgets["simulation.duration_days"].decimals()
        assert round(stored, places) == round(shown, places), (
            f"the Simulation page shows {shown} for a configuration that says "
            f"{stored} -- not a rounding difference at {places} decimals")

        # --- edit in All settings, on a key the card also shows.
        edited = 12.75
        sim.widgets["simulation.duration_days"].setValue(edited)
        assert manager.get_value("simulation.duration_days") == edited

        # --- and back into the flow.
        panel.show_body_first()
        assert start.card_stack.currentWidget() is card
        assert card.widgets["simulation.duration_days"].value() == edited, (
            f"the card came back showing "
            f"{card.widgets['simulation.duration_days'].value()} for a "
            f"duration the user changed to {edited} in All settings")

        # --- negative control: a card writes ONLY the control that fired.
        #     Measured to hold even with the re-mirror disabled, so this is not
        #     evidence for the re-mirror -- it is the guard that stops someone
        #     "fixing" a future mirror by writing every key on every change,
        #     which WOULD make a stale card destructive.
        card.widgets["simulation.timesteps"].setValue(96)
        assert manager.get_value("simulation.duration_days") == edited, (
            f"editing one control on a card wrote a sibling's value too "
            f"({manager.get_value('simulation.duration_days')} != {edited}); "
            f"a card must write only the key whose control the user touched")
        assert manager.get_value("simulation.timesteps") == 96

        # --- and the negative control: the mirror is not itself an edit. A
        #     second round trip with nothing typed changes nothing at all.
        import copy
        before = copy.deepcopy(manager._config_for_yaml())
        panel.show_all_settings()
        panel.show_body_first()
        after = manager._config_for_yaml()
        assert after == before, (
            "an empty card -> All settings -> card round trip changed the "
            "configuration. Differences:\n  " + "\n  ".join(
                f"{k}: {before.get(k)!r} -> {after.get(k)!r}"
                for k in sorted(set(before) | set(after))
                if before.get(k) != after.get(k)))
        print(f"    [round trip] duration {edited} survived "
              f"card -> All settings -> card, and an empty round trip is inert")


def test_opening_a_file_leaves_all_settings_and_lands_on_the_card():
    """Paul's DP-7 ruling, held across the new second level.

    Open lands on the body card. Landing on the card with the section rail
    still up would be landing in the advanced view, which is not the ruling.
    """
    with tempfile.TemporaryDirectory() as scratch:
        manager, panel = _panel(scratch)
        path = REPO_ROOT / "examples" / "quickstart_moon_diurnal.yaml"
        if not path.exists():
            raise Skip("quickstart_moon_diurnal.yaml is not in this checkout")
        panel.show_all_settings()
        assert panel.all_settings_visible(), "the fixture did not open the rail"
        manager.load_from_yaml(str(path))
        panel.load_from_config()
        route = panel.land_on_loaded_config()
        assert route.ok, route.refusal
        assert not panel.all_settings_visible(), (
            "opening a file left the user in the eleven-section view instead "
            "of landing on the body card")
        start = panel.tabs["start"]
        assert start.stack.currentIndex() == start.CARD
        assert start.card_stack.currentWidget() is start.cards["Moon"]


def test_the_main_window_menu_offers_both_levels():
    """Checked from the AST, because the window cannot be built here.

    ``PTM3DMainWindow`` constructs a ``VisualizationPanel``, which segfaults at
    the pyvistaqt/VTK import headless, so no probe can press this menu. The
    same limitation is why ``test_gui_run_is_read_only`` parses the source, and
    this follows it: the wiring is pinned structurally rather than claimed.

    It matters more now than it did. With the section rail hidden by default, a
    user who has never found the card's "All settings (advanced)" button has
    only the menu, and a menu item wired to nothing would be a dead end with no
    error -- the failure mode this whole restructure has to avoid.

    NOT COVERED: that the menu items are enabled, that the shortcuts do not
    collide with another action's, or that anything appears on screen. Those
    need a window.
    """
    import ast
    source = (REPO_ROOT / "gui" / "ptm3d_gui.py").read_text()
    tree = ast.parse(source)

    methods = {node.name for node in ast.walk(tree)
               if isinstance(node, ast.FunctionDef)}
    for name in ("show_all_settings", "show_body_first"):
        assert name in methods, (
            f"PTM3DMainWindow has no {name}(), so the View menu cannot reach "
            f"the two levels")

    # Every QAction built with its label and its connected handler.
    actions = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            fn = node.value.func
            if (getattr(fn, "id", None) or getattr(fn, "attr", None)) != "QAction":
                continue
            args = node.value.args
            if args and isinstance(args[0], ast.Constant):
                target = node.targets[0]
                key = getattr(target, "attr", None) or getattr(target, "id", None)
                actions[key] = args[0].value
    connected = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if getattr(fn, "attr", None) != "connect" or not node.args:
            continue
        handler = getattr(node.args[0], "attr", None)
        owner = getattr(fn.value, "value", None)
        signal = getattr(fn.value, "attr", None)
        if signal == "triggered" and handler:
            connected.add((getattr(owner, "attr", None)
                           or getattr(owner, "id", None), handler))

    wanted = (("all settings", "show_all_settings"),
              ("body cards", "show_body_first"))
    for phrase, handler in wanted:
        named = [key for key, label in actions.items()
                 if phrase in (label or "").replace("&", "").lower()]
        assert named, (
            f"no menu action is labeled with {phrase!r}; the two levels are "
            f"reachable only from the Start page")
        assert any((key, handler) in connected for key in named), (
            f"the {phrase!r} menu action exists but is not connected to "
            f"{handler}(): {sorted(connected)}")

    # ...and the window's own methods delegate rather than reaching into the
    # panel's widgets, so there is one implementation of "which level am I in".
    for name in ("show_all_settings", "show_body_first"):
        body = [n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == name][0]
        calls = {getattr(c.func, "attr", None) for c in ast.walk(body)
                 if isinstance(c, ast.Call)}
        assert name in calls, (
            f"PTM3DMainWindow.{name}() does not call config_panel.{name}(); it "
            f"is a second implementation of the level switch")
    print("    [nav] View menu wires both levels to the panel's own switch")


def test_a_card_that_names_off_card_fields_says_how_to_reach_them():
    """The instruction has to be followable now that the rail starts hidden.

    The crater card's "Still needed" list names mesh.occluder_file and three
    others, none of which are on the card. Before the rail was hidden, "the
    sections holding the fields above have been opened" was enough. With the
    rail hidden it names a place the user cannot see, which is precisely the
    class of quiet dead end this restructure must not introduce.
    """
    with tempfile.TemporaryDirectory() as scratch:
        manager, panel = _panel(scratch)
        manager.reset_to_defaults()
        panel.load_from_config()
        start = panel.tabs["start"]
        start.choose_body("Moon")
        start._choose_mode("local")
        card = start.open_card("Moon", "local", "time_series",
                               apply_defaults=True)
        report = card.report_text()
        assert "Advanced options" in report, (
            f"the crater card no longer says the tier was turned on:\n{report}")
        assert "All settings" in report, (
            f"the card tells the user to fill in fields that are not on it "
            f"without naming the control that gets them there, and the "
            f"section rail is hidden by default:\n{report}")
        assert panel.advanced_toggle.isChecked(), (
            "the crater card did not turn Advanced on")


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
    # GUI-synthetic keys that stand for real ones, and are translated or
    # dropped at save time by ConfigManager._config_for_yaml.
    #
    # The three flux-cube entries are inputs to `ptm_flux_cube_geom`, not to
    # the engine: the tool bakes them into the layout file's header, and the
    # engine reads them back OUT of that header and refuses a layout whose
    # values do not match the run. `config_parser.cpp` accepts them under
    # `flux_cubes:` and reports them as ignored, which is the same statement
    # from the other side. `_flux_cube_section_for_yaml` builds the emitted
    # block from a fixed list of the seven keys the engine reads, so none of
    # these three can reach a written configuration.
    synthetic = {"illumination.albedo_source", "mesh.topo_source",
                 "flux_cubes.mode", "flux_cubes.spacing_m",
                 "flux_cubes.radius_m"}
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

    WHAT THIS DOES NOT CHECK: which body the configuration describes.
    ``--validate-config`` answers "can the engine parse and run this file" --
    it is a SYNTAX and reachability check and has no opinion about identity.
    On 2026-08-27 the Europa Global card emitted ``simulation.body: Europa``
    beside ``generic_body: {name: bennu}`` and ran a 100 km sphere at 1 AU;
    that file is a perfectly valid asteroid run, so this arm passed it and
    would pass it again. Worse, the loop below does not reset between routes,
    so on its own ordering the asteroid route's ``generic_body`` reached the
    ``custom`` route and Mercury's ``simulation.body`` reached the asteroid
    route -- and every arm was green.

    Identity is a separate property and has its own file:
    ``test_gui_card_body_identity.py``, which asserts on the configuration the
    RUN writes and needs no engine at all.

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
# 6. The generic-body round-trip floor
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
        # The precondition is SET here, not inherited from whichever check ran
        # before this one. In-suite the engine arm above happens to leave a
        # generic-body card active, which is how the order-dependent GUI test rewrite was found;
        # run alone, this file used to land on whatever card the panel started
        # on and the bug could not reproduce, so the check passed on the broken
        # code. An asteroid card left open is a thing a user does, and it is the
        # state this property has to hold in.
        panel.tabs["start"].open_card("asteroid", "body", "time_series",
                                      apply_defaults=True)
        manager.set_value("generic_body.name", "probe_asteroid")
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
        # Spelled out, because a bare `assert a and b and c` reported this as
        # an EMPTY failure for as long as it was red: the active asteroid card
        # rewrote simulation.body on the `panel.load_from_config()` two lines
        # up, and the file was then refused -- with nothing on screen saying so.
        assert route.ok and route.body_key == "Moon" \
            and route.mode_key == "local", (
                f"the opened Moon file did not route to its own card: "
                f"ok={route.ok!r} body_key={route.body_key!r} "
                f"mode_key={route.mode_key!r}; simulation.body is now "
                f"{manager.get_value('simulation.body')!r} "
                f"(the file on disk says 'Moon'). "
                f"Refusal: {route.refusal!r}")
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


def test_a_file_naming_two_bodies_keeps_its_generic_body_block():
    """The other side of "opening a file is not an edit to it".

    A file that sets a BUILT-IN ``simulation.body`` and also carries a
    ``generic_body:`` section names two bodies, and the engine resolves that in
    one direction only: ``isConfigured()`` wins and ``simulation.body`` is never
    read (``ptm_thermal_sim.cpp:14626``). The GUI's ``_config_for_yaml``
    resolves it in the OTHER direction -- it DROPS the ``generic_body:`` block
    when the body name is built-in -- so a card that leaves such a file
    untouched emits a Moon run from a screen describing an asteroid.

    This is the boundary of the order-dependent GUI test guard, and it is here because the first
    draft of that guard crossed it: gating the repair on ``_seeding_defaults``
    alone fixed the Moon-file case and broke this one (measured 2026-08-30).
    """
    with tempfile.TemporaryDirectory() as scratch:
        manager, panel = _panel(scratch)
        path = Path(scratch) / "two_bodies.yaml"
        path.write_text(
            "simulation:\n  body: Moon\n  start_time: '2026-01-01T00:00:00'\n"
            "  duration_days: 1.0\n  timesteps: 48\n"
            "generic_body:\n  name: bennu\n"
            "  shape:\n    type: spherical\n    radius_km: 0.25\n"
            "  orbit:\n    semi_major_axis_au: 1.126\n    eccentricity: 0.2\n"
            "  spin:\n    period_hours: 4.3\n")
        manager.load_from_yaml(str(path))
        panel.load_from_config()
        route = panel.land_on_loaded_config()
        assert route.ok and route.body_key == "asteroid", (
            f"a generic-body file did not route to a generic-body card: "
            f"ok={route.ok!r} body_key={route.body_key!r} "
            f"refusal={route.refusal!r}")
        emitted = manager._config_for_yaml()
        assert emitted.get("generic_body"), (
            "the generic_body: block was dropped from the configuration this "
            "file would RUN. simulation.body is "
            f"{(emitted.get('simulation') or {}).get('body')!r}, which the "
            "engine would run as that built-in body -- not as the asteroid "
            "the card is showing.")
        assert not manager.body_identity_conflict(), (
            "the card left the configuration naming two bodies: "
            + manager.body_identity_conflict())


def test_a_second_body_lookup_moves_simulation_body_with_it():
    """`simulation.body` follows the body the card is describing.

    Before 2026-08-30 the sync was a side effect of `_repair_body_name` running
    from `load_card`, and that method only writes over a BUILT-IN name or the
    `Custom` placeholder -- so looking up a second body left the first one in
    `simulation.body`. Measured on `origin/main` 2baf6bb: a cache lookup of
    "23871 Ousha" straight after "623 Chimaera" left `simulation.body:
    623 Chimaera` beside `generic_body.name: 23871 Ousha`. The write is now
    explicit in `_apply_lookup`, and this is what holds it there.
    """
    from gui_core.body_lookup import BodyLookupResult
    with tempfile.TemporaryDirectory() as scratch:
        manager, panel = _panel(scratch)
        manager.reset_to_defaults()
        panel.load_from_config()
        start = panel.tabs["start"]
        start.choose_body("asteroid")
        start._choose_mode("body")
        card = start.open_card("asteroid", "body", "time_series",
                               apply_defaults=True)
        seen = []
        for want in ("623 Chimaera", "23871 Ousha"):
            card._apply_lookup(BodyLookupResult(
                found=True, source="cache", name=want,
                values={"generic_body.name": want},
                message=f"selftest {want}"))
            seen.append(str(manager.get_value("simulation.body", "")))
            assert seen[-1] == want, (
                f"after looking up {want!r} the configuration still says "
                f"simulation.body = {seen[-1]!r}; the run would be recorded "
                f"under the wrong body's name")
        print(f"    [lookup] simulation.body tracked both lookups: {seen}")


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
    test_a_body_button_is_the_body_name_and_nothing_else,
    test_the_mercury_data_caveat_lives_on_the_mercury_card,
    test_the_landing_screen_is_buttons_not_prose,
    test_no_header_or_label_on_a_card_is_phrased_as_a_question,
    test_the_register_check_actually_fires,
    test_the_misfiled_control_check_actually_fires,
    test_the_thermal_solver_controls_are_in_the_thermal_section,
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
    test_the_section_rail_is_not_the_primary_navigation,
    test_all_settings_is_reachable_and_clearly_secondary,
    test_the_rails_own_start_row_leaves_the_rail,
    test_a_hidden_section_does_not_eject_the_user_from_all_settings,
    test_leaving_all_settings_returns_to_the_same_card,
    test_an_edit_in_all_settings_reaches_the_card_and_survives_the_return,
    test_opening_a_file_leaves_all_settings_and_lands_on_the_card,
    test_the_main_window_menu_offers_both_levels,
    test_a_card_that_names_off_card_fields_says_how_to_reach_them,
    test_an_opened_file_lists_what_the_card_does_not_show,
    test_a_cards_exposed_keys_are_keys_the_engine_reads,
    test_every_card_emits_a_configuration_the_engine_accepts,
    test_a_surface_value_too_small_to_survive_is_refused,
    test_showing_a_card_does_not_rewrite_the_configuration,
    test_a_file_naming_two_bodies_keeps_its_generic_body_block,
    test_a_second_body_lookup_moves_simulation_body_with_it,
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
