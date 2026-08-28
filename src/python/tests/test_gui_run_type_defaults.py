#!/usr/bin/env python3
"""What a card proposes: the window, the spin-up, and the built-in overrides.

Four things Paul asked for after click-testing, and the properties that keep
each of them honest.

**1. The integrated card offers ONE duration.** It used to offer a time window
AND an output-sample count AND, on Europa, an integration window -- three
numbers describing one window, two of which mattered only through their ratio.
For that run type the integration window is now the only duration control, and
``simulation.duration_days`` / ``simulation.timesteps`` are DERIVED from it.
They are still written, because the engine still reads both
(``ptm_thermal_sim.cpp:15958`` and ``computeIntegrationWindow`` at ``:1134``),
so the emission stays a complete configuration -- which is what
``test_gui_body_first``'s ``--validate-config`` arm then checks.

**2. The spin-up is set automatically, on every run type.** The engine has no
``auto``: ``thermal.equilibration_orbits`` is a plain double defaulting to
1.0, which is below the two cycles the equilibration paper calls a minimum. The
rule is the GUI's, it is stated where it is computed, and the card shows it
with its reasoning. See :func:`gui_core.body_cards.auto_equilibration_cycles`.

**3. The built-in bodies' surface properties use the keys the engine READS.**
This is PTM-67, and it is the one that would go wrong silently: on the Moon,
Europa or Mercury the engine reads ``thermophysical.albedo_override`` and
ignores ``thermophysical.albedo``. A card offering the unsuffixed name would
write a value the run reads and then discards. The check below re-derives the
override list FROM THE C++ so a key renamed there fails here.

**4. The duration has a units toggle that round-trips exactly.** Display only:
the configuration key stays days. The toggle re-reads the stored value rather
than converting what is in the box, so a chain of toggles cannot lose a digit
-- which at the old two-decimal hours display it did, showing 3.551181 d as
85.23 h, or 3.551250 d.

Run directly::

    python3 src/python/tests/test_gui_run_type_defaults.py
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
    """A check that cannot run here, as distinct from one that failed."""


_APP = None


def _qt():
    global _APP
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
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    _scratch = tempfile.mkdtemp()
    for scope in (QSettings.Scope.UserScope, QSettings.Scope.SystemScope):
        QSettings.setPath(QSettings.Format.IniFormat, scope, _scratch)
    _APP = QApplication.instance() or QApplication([])
    return _APP


def _fresh():
    """A brand-new manager and body-first panel.

    NOT shared between checks. ``test_gui_body_first`` keeps one panel in a
    module global and has an order dependency because of it -- its
    "showing a card does not rewrite the configuration" check passes alone and
    fails after the ``--validate-config`` arm has run in the same process
    (measured on 9a974d1, before any of this round's changes). One panel per
    check costs a second and cannot develop that fault.
    """
    _qt()
    from gui_core.config_manager import ConfigManager
    from gui_panels.config_panel import ConfigurationPanel

    manager = ConfigManager()
    panel = ConfigurationPanel(manager)
    return manager, panel.tabs["start"]


ROUTES = (("Moon", "global"), ("Moon", "local"),
          ("Europa", "global"), ("Europa", "local"),
          ("Mercury", "global"))


# ---------------------------------------------------------------------- 1 + 4

def test_the_default_window_is_one_synodic_day_from_the_engines_table():
    """One diurnal cycle, per body, read from the generated body table.

    Not typed in. ``ptm.body_table`` is generated from
    ``planetary_properties.h`` and re-derived by ``test_body_capabilities`` on
    every run, so a period retuned in the C++ shows up as a failing test rather
    than as a card proposing a window the engine does not force at.
    """
    from ptm.body_table import BODY_CAPABILITIES
    from gui_core.body_cards import default_duration_days

    for body in ("Moon", "Europa", "Mercury"):
        days, why = default_duration_days(body, {})
        expected = BODY_CAPABILITIES[body]["solar_day_days"]
        assert days == expected, (
            f"{body}: the card proposes {days} d but the engine's own solar "
            f"day is {expected} d")
        assert "diurnal cycle" in why, why
        print(f"    [window] {body}: {days:g} d — {why}")


def test_the_integrated_card_offers_only_the_integration_window():
    """Paul's ruling, as geometry: one duration control, not three."""
    for body, mode in ROUTES:
        manager, start = _fresh()
        card = start.open_card(body, mode, "integrated", apply_defaults=True)
        rows = card._duration_rows
        shown = {
            "duration": not rows["simulation.duration_days"][0].isHidden(),
            "samples": not card._samples_row[0].isHidden(),
            "integration": not rows["thermal.integration_days"][0].isHidden(),
        }
        assert shown == {"duration": False, "samples": False,
                         "integration": True}, (
            f"{body}/{mode} integrated shows {shown}, but the integration "
            f"window must be the only duration control")

        # ...and the time-series card still offers the window and the cadence.
        manager, start = _fresh()
        card = start.open_card(body, mode, "time_series", apply_defaults=True)
        rows = card._duration_rows
        assert not rows["simulation.duration_days"][0].isHidden(), (
            f"{body}/{mode} time series lost its duration control")
        assert not card._samples_row[0].isHidden(), (
            f"{body}/{mode} time series lost its output-sample control")
    print(f"    [rows] {len(ROUTES)} routes: integrated shows the integration "
          f"window alone; time series keeps duration and samples")


def test_the_integrated_window_derives_the_keys_the_engine_reads():
    """The two derived keys follow the one control, on seed AND on edit."""
    from gui_core.body_cards import samples_for_window, synodic_period_days

    manager, start = _fresh()
    card = start.open_card("Moon", "global", "integrated", apply_defaults=True)
    synodic, _ = synodic_period_days("Moon", {})

    seeded = manager.get_value("thermal.integration_days")
    assert seeded == synodic, (
        f"the integrated card seeded a {seeded} d window, not the {synodic} d "
        f"diurnal cycle")
    assert manager.get_value("simulation.duration_days") == seeded, (
        "the output window does not follow the integration window, so the "
        "engine would simulate a span it does not integrate over")

    # An edit to the one control moves both derived keys.
    two = 2.0 * synodic
    card._duration_controls["thermal.integration_days"]["spin"].setValue(two)
    _qt().processEvents()
    assert manager.get_value("thermal.integration_days") == two
    assert manager.get_value("simulation.duration_days") == two, (
        "editing the integration window left the output window behind")
    expected_steps = samples_for_window(two, synodic)
    got = manager.get_value("simulation.timesteps")
    assert got == expected_steps, (
        f"a {two} d window was left at {got} output samples, not the "
        f"{expected_steps} that keep the sampling density fixed")
    assert got > 144, (
        "a window twice as long was sampled no more finely than a one-cycle "
        "window, so the statistics are smeared over the same slots")
    print(f"    [derive] window {two:g} d -> duration {two:g} d, "
          f"{got} samples (from {seeded:g} d / 144)")


def test_the_duration_unit_toggle_round_trips_exactly():
    """Display only. The stored value must survive any chain of toggles."""
    manager, start = _fresh()
    card = start.open_card("Europa", "global", "time_series",
                           apply_defaults=True)
    key = "simulation.duration_days"
    # Six significant decimals -- the value the shipped europa_diurnal preset
    # writes, and the one that lost a digit at the old 4/2-decimal displays.
    manager.set_value(key, 3.551181)
    card.activate("global", "time_series")
    before = manager.get_value(key)

    state = card._duration_controls[key]
    seen = {}
    for unit in ("hours", "synodic periods", "days", "hours", "days"):
        state["combo"].setCurrentText(unit)
        _qt().processEvents()
        seen[unit] = state["spin"].value()
        assert manager.get_value(key) == before, (
            f"toggling the display to {unit!r} changed the stored duration: "
            f"{before!r} -> {manager.get_value(key)!r}. The units control is "
            f"display-only and must never write.")
    after = manager.get_value(key)
    assert after == before, f"{before!r} -> {after!r}"
    # The hours reading must be the real conversion, not a rounded one.
    assert abs(seen["hours"] - before * 24.0) < 1e-9, (
        f"the hours display reads {seen['hours']}, which is "
        f"{seen['hours'] / 24.0} d and not the stored {before} d")
    print(f"    [units] {before} d survives 5 toggles; hours reads "
          f"{seen['hours']} (exactly x24)")


def test_the_integration_sentinel_survives_every_display_unit():
    """-1 means "use the output window", and must reach the screen intact.

    The bounds of a duration control are converted by the same function as its
    value. When they were not, the "synodic periods" floor stayed at 0 and the
    engine's -1 sentinel was clamped to 0 on the way to the display -- a
    control showing a setting the run does not have.
    """
    manager, start = _fresh()
    card = start.open_card("Europa", "global", "time_series",
                           apply_defaults=False)
    key = "thermal.integration_days"
    manager.set_value(key, -1.0)
    card.activate("global", "time_series")
    state = card._duration_controls[key]
    for unit, expect in (("days", -1.0), ("hours", -24.0),
                         ("synodic periods", None), ("days", -1.0)):
        state["combo"].setCurrentText(unit)
        _qt().processEvents()
        shown = state["spin"].value()
        assert manager.get_value(key) == -1.0, (
            f"the sentinel was rewritten while displaying it in {unit!r}: "
            f"{manager.get_value(key)!r}")
        assert shown < 0, (
            f"the -1 sentinel displays as {shown} in {unit!r}, so it has been "
            f"clamped by a bound that was not converted with it")
        if expect is not None:
            assert abs(shown - expect) < 1e-9, (unit, shown, expect)
    print("    [sentinel] integration_days = -1 survives days, hours and "
          "synodic periods intact")


def test_every_duration_control_shows_the_same_precision():
    """GUI-DURATION-DECIMALS-4-VS-6, closed.

    ``simulation.duration_days`` was shown at four decimals by the Simulation
    tab and six by the body cards, so the same configuration read differently
    depending on the screen. Measured 2026-08-28: the stored value SURVIVES --
    the write-back is guarded, so this was never corruption -- but the rounded
    number is what a user edits from.
    """
    from gui_panels.body_cards_panel import BodyCard

    _, start = _fresh()
    card = start.open_card("Moon", "global", "time_series",
                           apply_defaults=True)
    wanted = BodyCard.DURATION_DECIMALS
    for key, state in card._duration_controls.items():
        assert state["spin"].decimals() == wanted, (
            f"{key} is shown at {state['spin'].decimals()} decimals, not the "
            f"{wanted} every duration control uses")
    print(f"    [decimals] every card duration control shows {wanted} "
          f"decimals, in every unit")


# -------------------------------------------------------------------------- 2

def test_the_spin_up_is_set_automatically_on_every_route():
    """Announced, editable, and above the engine's own too-low default."""
    from gui_core.body_cards import EQ_VALIDATED_CYCLES

    for body, mode in ROUTES:
        for run_type in ("time_series", "integrated"):
            manager, start = _fresh()
            card = start.open_card(body, mode, run_type, apply_defaults=True)
            cycles = manager.get_value("thermal.equilibration_orbits")
            assert cycles >= EQ_VALIDATED_CYCLES, (
                f"{body}/{mode}/{run_type}: spin-up {cycles}, below the "
                f"{EQ_VALIDATED_CYCLES} cycles the equilibration paper "
                f"validated against Hayne et al. (2017)")
            assert cycles > 1.0, (
                f"{body}/{mode}/{run_type}: spin-up left at the engine "
                f"default, which is below the paper's own minimum of two")
            note = card.equilibration_note.text()
            assert note.startswith("auto: "), (
                f"{body}/{mode}/{run_type}: the automatic spin-up is not "
                f"announced with its provenance: {note!r}")
            assert "Editable" in note, note
            assert not card.widgets[
                "thermal.equilibration_orbits"].isReadOnly(), (
                "the automatic spin-up must stay editable")
    print(f"    [spin-up] {2 * len(ROUTES)} routes: auto >= "
          f"{EQ_VALIDATED_CYCLES:g} cycles, announced and editable")


def test_the_spin_up_rule_scales_with_the_column_and_the_window():
    """The two terms of the rule, each shown to move the answer.

    Neither is decoration: the depth term is the diffusive relaxation time of
    the deepest layer (quadratic in the column depth measured in skin depths),
    and the window term is the statement that a spin-up shorter than the run
    cannot have settled it.
    """
    from gui_core.body_cards import (EQ_REFERENCE_THERMAL_B,
                                     EQ_VALIDATED_CYCLES,
                                     auto_equilibration_cycles)

    base, _ = auto_equilibration_cycles("Moon", {})
    assert base == EQ_VALIDATED_CYCLES, base

    # Twice the column depth is four times the relaxation time.
    deep, why = auto_equilibration_cycles(
        "Moon", {"thermal": {"thermal_b": 2 * EQ_REFERENCE_THERMAL_B}})
    assert deep == 4 * base, (
        f"doubling thermal_b gave {deep} cycles, not the {4 * base} a "
        f"quadratic relaxation time implies")
    assert "deeper column" in why, why

    # A year-long lunar run needs more spin-up than six cycles.
    seasonal, why = auto_equilibration_cycles(
        "Moon", {"simulation": {"duration_days": 365.25}})
    assert seasonal >= 365.25 / 29.530588, (
        f"a 365.25 d run was given {seasonal} cycles of spin-up, less than "
        f"the run itself")
    assert "the length of the run itself" in why, why

    # A whole number of cycles, always: the engine counts integrated cycles.
    for value in (base, deep, seasonal):
        assert value == int(value), f"{value} is not a whole cycle count"
    print(f"    [rule] base {base:g}, deep column {deep:g}, "
          f"365 d run {seasonal:g} cycles")


# -------------------------------------------------------------------------- 3

#: The `thermophysical:` keys the PARSER reads, scraped from its own read-list.
_KREAD = re.compile(r'static const char\* const kRead\[\]\s*=\s*\{(.*?)\}',
                    re.S)


def _engine_thermophysical_keys():
    source = (REPO_ROOT / "src" / "cpp" / "tools" / "config_parser.cpp")
    if not source.is_file():
        raise Skip(f"{source} is not in this checkout")
    match = _KREAD.search(source.read_text(errors="replace"))
    if not match:
        raise Skip("config_parser.cpp has no kRead[] list to compare against")
    return set(re.findall(r'"([^"]+)"', match.group(1)))


def test_the_card_overrides_are_exactly_the_keys_the_engine_reads():
    """PTM-67, held by re-derivation rather than by memory.

    On a built-in body the engine reads ``*_override`` and ignores the
    unsuffixed names; anything else under ``thermophysical:`` draws a "read by
    nothing and have NO effect" warning at parse time. So the card's list is
    checked against the parser's OWN read-list: a key renamed in the C++ fails
    here instead of becoming a control that quietly does nothing.
    """
    from gui_panels.body_cards_panel import BodyCard

    engine = _engine_thermophysical_keys()
    offered = {key.split(".", 1)[1] for key, *_ in BodyCard.BUILTIN_OVERRIDES}
    unread = offered - engine
    assert not unread, (
        f"the card offers {sorted(unread)} under thermophysical:, which the "
        f"parser's own read-list does not contain -- editing them would "
        f"change no result (PTM-67)")
    # ...and every *_override the engine reads is offered, so a property the
    # engine gained does not sit unreachable.
    missing = {k for k in engine if k.endswith("_override")} - offered
    assert not missing, (
        f"the engine reads {sorted(missing)} but no card control offers them")
    print(f"    [PTM-67] {len(offered)} overrides offered, all in the "
          f"parser's read-list: {sorted(offered)}")


def test_a_builtin_card_shows_the_engines_values_and_writes_nothing():
    """Engine defaults visible, provenance stated, configuration untouched."""
    from ptm.body_table import BODY_CAPABILITIES
    from gui_panels.body_cards_panel import BodyCard

    for body in ("Moon", "Europa", "Mercury"):
        manager, start = _fresh()
        card = start.open_card(body, "global", "time_series",
                               apply_defaults=True)
        engine = BODY_CAPABILITIES[body]
        for key, field, *_ in BodyCard.BUILTIN_OVERRIDES:
            state = card._override_rows[key]
            assert state["tick"].isChecked(), (
                f"{body}: {key} does not start on the engine's own value")
            assert abs(state["spin"].value() - engine[field]) < 1e-9, (
                f"{body}: {key} shows {state['spin'].value()}, not the "
                f"engine's {engine[field]}")
            stored = manager.get_value(key, None)
            assert stored in (None, -1.0), (
                f"{body}: {key} was written as {stored!r} merely by showing "
                f"the card; an untouched override must stay unset")
        assert "engine default" in card.override_note.text(), (
            f"{body}: the provenance line does not say the values are the "
            f"engine's: {card.override_note.text()!r}")
        print(f"    [surface] {body}: 5 overrides show the engine's values, "
              f"none written")


def test_overriding_a_property_writes_the_override_key():
    """Untick, type, and the run uses your number -- through the read key."""
    manager, start = _fresh()
    card = start.open_card("Moon", "global", "time_series",
                           apply_defaults=True)
    key = "thermophysical.albedo_override"
    card._override_rows[key]["tick"].setChecked(False)
    card._override_rows[key]["spin"].setValue(0.25)
    _qt().processEvents()
    assert manager.get_value(key) == 0.25, (
        f"overriding the albedo wrote {manager.get_value(key)!r}")
    assert "albedo set on this card" in card.override_note.text(), (
        card.override_note.text())

    # ...and putting it back clears the override to the engine's sentinel.
    card._override_rows[key]["tick"].setChecked(True)
    _qt().processEvents()
    from gui_panels.body_cards_panel import BodyCard
    assert manager.get_value(key) == BodyCard.OVERRIDE_UNSET, (
        f"un-overriding left {manager.get_value(key)!r}, not the engine's "
        f"{BodyCard.OVERRIDE_UNSET} sentinel")
    assert "engine default" in card.override_note.text()
    print("    [override] albedo 0.25 written and cleared through "
          "thermophysical.albedo_override")


def test_an_opened_file_keeps_its_own_override():
    """A card must display an override, not overwrite it."""
    manager, start = _fresh()
    manager.set_value("simulation.body", "Moon")
    manager.set_value("thermophysical.ks_override", 0.005)
    card = start.open_card("Moon", "global", "time_series",
                           apply_defaults=False)
    state = card._override_rows["thermophysical.ks_override"]
    assert not state["tick"].isChecked(), (
        "a file that sets ks_override was shown as using the engine default")
    assert abs(state["spin"].value() - 0.005) < 1e-9, state["spin"].value()
    assert manager.get_value("thermophysical.ks_override") == 0.005, (
        "opening the file changed the value it set")
    print("    [open] a file's own ks_override is shown, not overwritten")


CHECKS = [
    test_the_default_window_is_one_synodic_day_from_the_engines_table,
    test_the_integrated_card_offers_only_the_integration_window,
    test_the_integrated_window_derives_the_keys_the_engine_reads,
    test_the_duration_unit_toggle_round_trips_exactly,
    test_the_integration_sentinel_survives_every_display_unit,
    test_every_duration_control_shows_the_same_precision,
    test_the_spin_up_is_set_automatically_on_every_route,
    test_the_spin_up_rule_scales_with_the_column_and_the_window,
    test_the_card_overrides_are_exactly_the_keys_the_engine_reads,
    test_a_builtin_card_shows_the_engines_values_and_writes_nothing,
    test_overriding_a_property_writes_the_override_key,
    test_an_opened_file_keeps_its_own_override,
]


def main():
    failed = skipped = 0
    for check in CHECKS:
        name = check.__name__
        try:
            check()
        except Skip as exc:
            skipped += 1
            print(f"SKIP {name}: {exc}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {name}:\n  {exc}")
        except Exception as exc:                             # noqa: BLE001
            failed += 1
            import traceback
            print(f"ERROR {name}: {exc!r}")
            traceback.print_exc()
        else:
            print(f"PASS {name}")
    total = len(CHECKS)
    print(f"\n[test_gui_run_type_defaults] {total - failed - skipped}/{total} "
          f"passed, {skipped} skipped, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
