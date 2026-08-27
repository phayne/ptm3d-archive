#!/usr/bin/env python3
"""Every control in the configuration panel explains itself, and does it once.

The user's complaint was not that controls were missing. It was that names like
"spin-up seed, physical", "seed anchor, lumped" and "refuse two-timescale
factors" mean nothing to someone who has not read the solver. A control whose
label is a term of art and whose tooltip repeats that term of art has told the
user nothing.

So this file is a ratchet on the help text itself, not just on its presence:

  * coverage      -- no control may ship bare, INCLUDING one added tomorrow
  * single source -- `gui_tooltips.py` is the only place the prose lives
  * substance     -- a tooltip may not simply restate its own label
  * plain language-- no ticket numbers and no internal shorthand
  * enums         -- every choice a combo offers is explained by name

Run directly::

    python3 src/python/tests/test_gui_tooltip_coverage.py
"""

from __future__ import annotations

import os
import re
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
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    for scope in (QSettings.Scope.UserScope, QSettings.Scope.SystemScope):
        QSettings.setPath(QSettings.Format.IniFormat, scope, str(scratch))
    from gui_core.config_manager import ConfigManager
    from gui_panels.config_panel import ConfigurationPanel
    _APP = QApplication.instance() or QApplication([])
    _MANAGER = ConfigManager()
    _PANEL = ConfigurationPanel(_MANAGER)
    return _MANAGER, _PANEL


def _controls(panel):
    """Every value-carrying control the user can see, with its key if it has
    one.

    Qt builds a QLineEdit of its own inside every spin box and every editable
    combo. Those are not controls anybody added, they carry no key, and
    counting them would inflate the coverage number by half. They are excluded
    by PARENTAGE rather than by object name, because the name is a Qt
    implementation detail.
    """
    from PyQt6.QtWidgets import (
        QAbstractSpinBox, QCheckBox, QComboBox, QDoubleSpinBox, QLineEdit,
        QRadioButton, QSpinBox,
    )
    inputs = (QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox, QLineEdit,
              QRadioButton)
    found = []
    for tab_name, tab in panel.tabs.items():
        by_widget = {id(w): k for k, w in tab.widgets.items()}
        for widget in tab.findChildren(inputs):
            if isinstance(widget.parentWidget(), (QAbstractSpinBox, QComboBox)):
                continue
            found.append((tab_name, by_widget.get(id(widget)), widget))
    return found


def _describe(tab_name, key, widget):
    from PyQt6.QtWidgets import QCheckBox, QRadioButton
    if key:
        return f"{tab_name}/{key}"
    if isinstance(widget, (QCheckBox, QRadioButton)) and widget.text():
        return f"{tab_name}/({widget.text()})"
    return f"{tab_name}/<{type(widget).__name__} {widget.objectName()!r}>"


# --------------------------------------------------------------------------
# 1. COVERAGE -- the headline property
# --------------------------------------------------------------------------

def test_every_visible_control_has_a_tooltip():
    """N of N, and the N is printed so a regression is visible as a number.

    This is the guard that stops a future key shipping bare: a control added
    without an entry in gui_tooltips.py fails here, on the day it is added,
    rather than reaching a user as an unexplained box.
    """
    with tempfile.TemporaryDirectory() as scratch:
        _, panel = _panel(scratch)
        controls = _controls(panel)
        bare = [_describe(t, k, w) for t, k, w in controls if not w.toolTip()]
        assert controls, "no controls were found at all -- the probe is broken"
        assert not bare, (
            f"{len(bare)} of {len(controls)} controls have no tooltip: "
            f"{sorted(bare)}")
        print(f"    [coverage] {len(controls)}/{len(controls)} controls "
              f"tooltipped ({sum(1 for c in controls if c[1])} bound to a "
              f"configuration key)")


# --------------------------------------------------------------------------
# 2. SINGLE SOURCE
# --------------------------------------------------------------------------

#: Controls whose tooltip is deliberately STATE-DEPENDENT: the panel swaps in
#: a different `ui.*`/`*.forced` entry to disclose that the control is inert or
#: forced in the current configuration. Still sourced from gui_tooltips -- the
#: structural test below proves no literal is authored in the panel -- so these
#: are exempt only from the exact-equality check.
_STATE_DEPENDENT = {
    # Forced on whenever the VRAM floor is non-default.
    "gpu.margin_explicit",
    # On a built-in body the engine has no override key for these, so the panel
    # replaces the property's own help with the reason it is disabled.
    "thermophysical.rhos", "thermophysical.rhod", "thermophysical.cp0",
    "thermophysical.Qb", "thermophysical.chi",
}


def test_the_text_on_screen_is_the_text_in_gui_tooltips():
    """Two copies of a tooltip is two answers to the same question.

    Before this, the prose lived at 131 call sites inside config_panel.py AND
    in gui_tooltips.py, and the two disagreed. The thermal-inertia field is the
    case that proves it mattered: its LABEL said editing the value 'scales ks,
    kd, rhos and rhod by a uniform factor ... using cp at the equilibrium
    temperature', while the FIELD beside it said the value sets chi to 0 and
    makes cp constant. The field was right; the label described behavior the
    GUI does not have.
    """
    from gui_tooltips import TOOLTIPS
    with tempfile.TemporaryDirectory() as scratch:
        _, panel = _panel(scratch)
        wrong = []
        for tab_name, key, widget in _controls(panel):
            if not key or key not in TOOLTIPS or key in _STATE_DEPENDENT:
                continue
            if widget.toolTip() != TOOLTIPS[key]:
                wrong.append(
                    f"{tab_name}/{key}: the widget shows text that is not "
                    f"TOOLTIPS[{key!r}]")
        assert not wrong, "\n  ".join(
            ["a control's help text does not come from gui_tooltips.py:"]
            + wrong)


#: Every source file that builds user-visible controls. The body-first cards
#: are a second such file, and the single-source rule applies to them for the
#: same reason it applies to the first: two copies of a tooltip is two answers
#: to the same question, and which one the user reads is decided by which file
#: someone edited last.
_PANEL_SOURCES = (
    REPO_ROOT / "gui" / "gui_panels" / "config_panel.py",
    REPO_ROOT / "gui" / "gui_panels" / "body_cards_panel.py",
)


def test_the_panel_authors_no_tooltip_prose_of_its_own():
    """The structural half of "single source", and the stronger half.

    The panel files build widgets. Every string they hand to setToolTip must
    come from gui_tooltips -- a `get_tooltip(...)` call, possibly `.format(...)`
    for the two state-dependent ones, or a value carried in from the engine.
    A bare literal here is a second source by definition, whether or not it
    currently agrees with the first.
    """
    import ast
    from gui_tooltips import TOOLTIPS

    literals = []
    asked = set()
    for src in _PANEL_SOURCES:
        if not src.exists():
            raise AssertionError(f"{src} is missing; this check would pass "
                                 f"vacuously for the controls it builds")
        tree = ast.parse(src.read_text())
        for node in ast.walk(tree):
            fn = getattr(node, "func", None)
            if not isinstance(node, ast.Call) \
                    or not isinstance(fn, ast.Attribute) \
                    or fn.attr != "setToolTip":
                continue
            # Collect the string constants that are a get_tooltip() KEY; every
            # other string constant under this call is prose.
            keys = set()
            for sub in ast.walk(node):
                sub_fn = getattr(sub, "func", None)
                name = (getattr(sub_fn, "id", None)
                        or getattr(sub_fn, "attr", None))
                if isinstance(sub, ast.Call) and name == "get_tooltip":
                    for karg in sub.args:
                        if isinstance(karg, ast.Constant):
                            keys.add(id(karg))
            for arg in node.args:
                for sub in ast.walk(arg):
                    if isinstance(sub, ast.Constant) \
                            and isinstance(sub.value, str) \
                            and sub.value.strip() \
                            and id(sub) not in keys:
                        literals.append(
                            f"{src.name}:{node.lineno}: {sub.value[:70]!r}")

        # ...and every key a panel asks for must actually exist, or the control
        # silently ships bare -- which is what the coverage test then catches,
        # but this names the typo directly.
        for node in ast.walk(tree):
            fn = getattr(node, "func", None)
            name = getattr(fn, "id", None) or getattr(fn, "attr", None)
            if isinstance(node, ast.Call) and name == "get_tooltip":
                for karg in node.args:
                    if isinstance(karg, ast.Constant) \
                            and isinstance(karg.value, str):
                        asked.add(karg.value)

    assert not literals, "\n  ".join(
        ["tooltip prose authored inside a panel instead of gui_tooltips.py:"]
        + sorted(set(literals)))

    unknown = sorted(k for k in asked if k not in TOOLTIPS)
    assert not unknown, f"a panel asks for tooltip key(s) that do not exist: {unknown}"
    print(f"    [single source] {len(asked)} tooltip keys asked for across "
          f"{len(_PANEL_SOURCES)} panel files, 0 authored in place")


def test_no_info_key_names_a_tooltip_that_does_not_exist():
    """A stale `info_key=` fails silently -- the label just loses its help.

    Twelve of them were left behind on the Body Definition tab, naming keys
    (`generic_body.orbit.sma`, `generic_body.shape_type`, ...) that were never
    the CONFIGURATION keys the controls are bound to; they only ever worked
    because the old table was keyed to match them. `_apply_tooltip` returns
    early on an empty lookup, so nothing warned.
    """
    import ast
    import re
    from gui_tooltips import TOOLTIPS
    stale = set()
    for path in _PANEL_SOURCES:
        src = path.read_text()
        ast.parse(src)                   # cheap syntax guard on the same file
        stale.update(m.group(1)
                     for m in re.finditer(r'info_key="([^"]+)"', src)
                     if m.group(1) not in TOOLTIPS)
    assert not stale, (
        f"info_key= names {len(stale)} tooltip key(s) that do not exist, so "
        f"those labels silently lose their help: {sorted(stale)}")


def test_a_disabled_control_is_not_styled_as_an_enabled_one():
    """An explicit `color:` overrides Qt's disabled palette.

    The blue tint marks a label that has help behind it. Put on the control
    itself it defeats the graying-out that tells a user a field is inert --
    which is the whole point of gating `radiation.beaming` on facet scale, or
    the H-matrix sub-options on H-matrix being on.
    """
    with tempfile.TemporaryDirectory() as scratch:
        # AFTER _panel, which owns the Skip guard: importing PyQt6 at function
        # scope turns "no PyQt6 here" from a clean skip into a hard error and
        # exits the whole file 1.
        _, panel = _panel(scratch)
        from PyQt6.QtWidgets import QCheckBox
        colored = []
        for tab_name, key, widget in _controls(panel):
            if isinstance(widget, QCheckBox) and "color:" in widget.styleSheet():
                colored.append(f"{tab_name}/{key}")
        assert not colored, (
            f"{len(colored)} checkbox(es) carry an explicit text color, which "
            f"survives being disabled: {sorted(colored)}")


def test_no_call_site_still_carries_a_superseded_tooltip():
    """A dead string that reads like documentation is a trap.

    The field helpers prefer gui_tooltips, so a literal left at a call site for
    a covered key never reaches the screen -- but the next person to improve
    the wording would edit it and see nothing change.
    """
    import ast
    from gui_tooltips import TOOLTIPS

    sigs = {
        "add_string_field": (3, 4), "add_int_field": (3, 6),
        "add_double_field": (3, 7), "add_combo_field": (3, 5),
        "add_bool_field": (3, 4), "add_file_field": (3, 5),
        "add_directory_field": (3, 4), "add_device_ids_field": (3, 4),
        "add_property_with_default": (3, 8),
    }
    dead = []
    for src in _PANEL_SOURCES:
        tree = ast.parse(src.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if not isinstance(fn, ast.Attribute) or fn.attr not in sigs:
                continue
            ki, ti = sigs[fn.attr]
            if len(node.args) <= ki:
                continue
            key_node = node.args[ki]
            if not isinstance(key_node, ast.Constant) \
                    or not isinstance(key_node.value, str) \
                    or key_node.value not in TOOLTIPS:
                continue
            literal = (len(node.args) > ti
                       and isinstance(node.args[ti], ast.Constant)
                       and isinstance(node.args[ti].value, str)
                       and node.args[ti].value)
            for kw in node.keywords:
                if kw.arg == "tooltip" and isinstance(kw.value, ast.Constant) \
                        and kw.value.value:
                    literal = kw.value.value
            if literal:
                dead.append(f"{src.name}:{node.lineno} {key_node.value}")
    assert not dead, "\n  ".join(
        ["these tooltip literals are dead -- gui_tooltips.py supersedes them:"]
        + dead)


# --------------------------------------------------------------------------
# 3. SUBSTANCE
# --------------------------------------------------------------------------

#: Short enough that it cannot answer "what does it do, what should it be, when
#: would I change it". The pre-existing offenders sat at 16-27 characters
#: ("Body shape model", "Orbital eccentricity", "Mean anomaly at epoch").
_MIN_TOOLTIP_CHARS = 80


def test_no_tooltip_merely_restates_its_own_label():
    """"Orbital eccentricity" as the help for a field labeled Eccentricity
    tells a planetary scientist exactly nothing they did not already see."""
    with tempfile.TemporaryDirectory() as scratch:
        _, panel = _panel(scratch)
        thin = []
        for tab_name, key, widget in _controls(panel):
            tip = (widget.toolTip() or "").strip()
            if not tip:
                continue                     # covered by the coverage test
            if len(tip) < _MIN_TOOLTIP_CHARS:
                thin.append(
                    f"{_describe(tab_name, key, widget)}: {len(tip)} chars -- "
                    f"{tip!r}")
        assert not thin, "\n  ".join(
            [f"tooltips shorter than {_MIN_TOOLTIP_CHARS} characters cannot "
             f"answer what/recommended/when:"] + thin)


# --------------------------------------------------------------------------
# 4. PLAIN LANGUAGE
# --------------------------------------------------------------------------

#: Shorthand that is meaningful inside the project and opaque outside it. A
#: user reading a tooltip has no ticket tracker and no session notes.
_JARGON = [
    (re.compile(r"\bPTM-\d+", re.I), "a ticket number"),
    (re.compile(r"\bPR #?\d+", re.I), "a pull-request number"),
    (re.compile(r"\bbyte-identical\b", re.I), "internal shorthand"),
    (re.compile(r"\bprepare-only\b", re.I), "internal shorthand"),
    (re.compile(r"\bdo-no-harm\b", re.I), "internal shorthand"),
    (re.compile(r"\bproven negative\b", re.I), "internal shorthand"),
    (re.compile(r"\bPaul'?s ruling\b", re.I), "a name outside the project"),
    (re.compile(r"\bthe crew\b", re.I), "internal shorthand"),
]


def test_the_help_text_carries_no_internal_shorthand():
    from gui_tooltips import TOOLTIPS
    hits = []
    for key, tip in TOOLTIPS.items():
        for pattern, why in _JARGON:
            match = pattern.search(tip)
            if match:
                hits.append(f"{key}: {match.group(0)!r} is {why}")
    assert not hits, "\n  ".join(
        ["tooltips are read by users, not by the people who wrote the engine:"]
        + hits)


# --------------------------------------------------------------------------
# 5. ENUMS
# --------------------------------------------------------------------------

#: Combos whose choices are POLICY, not data. Each choice has to be explained
#: by name, because picking one is a modeling decision. Deliberately excludes
#: combos whose items are data -- the body list, the property presets -- where
#: naming every entry in the tooltip would be noise.
_POLICY_ENUMS = [
    "thermal.spinup_seed",
    "thermal.seed_anchor",
    "thermal.eq_indirect_source",
    "radiation.coupling_mode",
    "thermophysical.conductivity_model",
    "gpu.placement_policy",
    "gpu.thin_margin_policy",
    "output.format",
    "output.vtk_mode",
    "generic_body.shape.type",
    "generic_body.ephemeris_source",
    "thermal.property_map_mode",
    "thermal.property_map_kind",
]


def test_every_choice_a_policy_combo_offers_is_explained():
    from gui_tooltips import TOOLTIPS
    with tempfile.TemporaryDirectory() as scratch:
        _, panel = _panel(scratch)
        bound = {}
        for tab in panel.tabs.values():
            bound.update(tab.widgets)
        missing = []
        for key in _POLICY_ENUMS:
            widget = bound.get(key)
            assert widget is not None, f"{key} has no control at all"
            tip = TOOLTIPS.get(key, "")
            for i in range(widget.count()):
                choice = widget.itemText(i)
                if choice and choice not in tip:
                    missing.append(f"{key}: choice {choice!r} is not explained")
        assert not missing, "\n  ".join(
            ["a combo offers a choice its tooltip never mentions:"] + missing)


# --------------------------------------------------------------------------
# 6. TABLE HYGIENE
# --------------------------------------------------------------------------

def test_every_relabeled_key_is_a_real_control():
    """A LABELS entry for a key nothing is bound to is a rename that silently
    did nothing -- exactly the class of quiet no-op this pass exists to end."""
    from gui_tooltips import LABELS
    with tempfile.TemporaryDirectory() as scratch:
        _, panel = _panel(scratch)
        bound = set()
        for tab in panel.tabs.values():
            bound.update(tab.widgets)
        orphan = sorted(k for k in LABELS if k not in bound)
        assert not orphan, (
            f"LABELS relabels key(s) with no control: {orphan}")


def test_every_relabeled_control_shows_its_new_label():
    """The relabel has to reach the screen, not just the table."""
    from gui_tooltips import LABELS
    with tempfile.TemporaryDirectory() as scratch:
        _, panel = _panel(scratch)
        from PyQt6.QtWidgets import QCheckBox   # after the Skip guard
        bound = {}
        for tab in panel.tabs.values():
            bound.update(tab.widgets)
        # Only checkboxes carry their own label text; for the rest the label is
        # a sibling QLabel, which is checked through the panel's grid below.
        wrong = []
        for key, label in LABELS.items():
            widget = bound.get(key)
            if isinstance(widget, QCheckBox) and widget.text() != label:
                wrong.append(f"{key}: shows {widget.text()!r}, LABELS says "
                             f"{label!r}")
        assert not wrong, "\n  ".join(["a relabel did not reach the screen:"]
                                      + wrong)


def test_no_tooltip_describes_a_key_the_engine_never_reads():
    """A typo in a TOOLTIPS key is a tooltip nobody ever sees.

    Allows `ui.*` (controls with no configuration key) and the handful of
    panel-synthetic keys that stand for a real one.
    """
    from gui_tooltips import TOOLTIPS
    from ptm.engine_config import known_keys
    known = set(known_keys())
    synthetic = {
        "gpu.margin_explicit",            # stands for "write min_vram_margin_gib"
        "gpu.margin_explicit.forced",     # its forced-on variant
        "thermal.property_map_mode",      # panel-side selector over two keys
        "thermal.property_map_kind",
        "thermophysical.thermal_inertia",  # stands for constant_thermal_inertia
        "generic_body.spin.obliquity",    # derived from the pole pair
        "simulation.end_time",            # derived from start + duration
    }
    with tempfile.TemporaryDirectory() as scratch:
        _, panel = _panel(scratch)
        bound = set()
        for tab in panel.tabs.values():
            bound.update(tab.widgets)
        stray = sorted(
            k for k in TOOLTIPS
            if not k.startswith("ui.")
            and k not in synthetic
            and k not in known
            and k not in bound)
        assert not stray, (
            f"TOOLTIPS describes key(s) the engine does not read and no "
            f"control is bound to: {stray}")


TESTS = [
    test_every_visible_control_has_a_tooltip,
    test_the_text_on_screen_is_the_text_in_gui_tooltips,
    test_the_panel_authors_no_tooltip_prose_of_its_own,
    test_no_call_site_still_carries_a_superseded_tooltip,
    test_no_info_key_names_a_tooltip_that_does_not_exist,
    test_a_disabled_control_is_not_styled_as_an_enabled_one,
    test_no_tooltip_merely_restates_its_own_label,
    test_the_help_text_carries_no_internal_shorthand,
    test_every_choice_a_policy_combo_offers_is_explained,
    test_every_relabeled_key_is_a_real_control,
    test_every_relabeled_control_shows_its_new_label,
    test_no_tooltip_describes_a_key_the_engine_never_reads,
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
    print(f"\n[test_gui_tooltip_coverage] {total - failures - skipped}/{total} "
          f"passed, {skipped} skipped")
    if skipped == total:
        print("[test_gui_tooltip_coverage] SKIPPED ENTIRELY -- proved NOTHING.")
    if failures:
        print("[test_gui_tooltip_coverage] FAIL")
        return 1
    print("[test_gui_tooltip_coverage] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
