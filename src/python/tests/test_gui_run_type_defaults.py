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
This is the unread-config-key defect, and it is the one that would go wrong silently: on the Moon,
Europa or Mercury the engine reads ``thermophysical.albedo_override`` and
ignores ``thermophysical.albedo``. A card offering the unsuffixed name would
write a value the run reads and then discards. The check below re-derives the
override list FROM THE C++ so a key renamed there fails here.

**4. The duration has a units toggle that round-trips exactly.** Display only:
the configuration key stays days. The toggle re-reads the stored value rather
than converting what is in the box, so a chain of toggles cannot lose a digit
-- which at the old two-decimal hours display it did, showing 3.551181 d as
85.23 h, or 3.551250 d.

**5. The integrated run's window is a COUNT OF PERIODS, and no sentinel is
reachable.** ``thermal.integration_days`` is the only duration on that run
type, so the engine's -1 ("integrate the output window") had nothing left to
refer to: it showed as the default, and the card offered no way to say how
long the run should be. The control now reads "N diurnal periods" or "N
orbital periods", its floor is a real duration in every unit it offers, and a
file that arrives carrying -1 is resolved to the length -1 already meant, with
the resolution announced.

The orbital period is HELIOCENTRIC on every body, which the engine's own
``props.year`` is not: for Europa the engine sets the year equal to the day,
the 3.55 d orbit around Jupiter. ``ptm.body_capabilities`` carries the
heliocentric values and ``test_body_capabilities`` holds each against Kepler's
third law on the body's own ``rAU``.

**6. Displaying a duration in another unit does not rewrite it.** A real
corruption, measured on the base commit and closed here: see
:func:`test_showing_a_duration_in_another_unit_does_not_rewrite_it`.

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
    from gui_core.body_cards import (WINDOW_UNIT_DIURNAL, samples_for_window,
                                     synodic_period_days)

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

    # ...and the control OPENS in diurnal periods, showing that window as 1.
    state = card._duration_controls["thermal.integration_days"]
    assert state["combo"].currentText() == WINDOW_UNIT_DIURNAL, (
        f"the window control opened in {state['combo'].currentText()!r}, not "
        f"in the diurnal periods its default window is one of")
    assert abs(state["spin"].value() - 1.0) < 1e-9, (
        f"the default window reads {state['spin'].value()} diurnal periods, "
        f"not 1")

    # An edit to the one control moves both derived keys.
    state["spin"].setValue(2.0)                    # two diurnal periods
    _qt().processEvents()
    two = 2.0 * synodic
    assert manager.get_value("thermal.integration_days") == two, (
        f"two diurnal periods stored "
        f"{manager.get_value('thermal.integration_days')!r} d, not {two} d")
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
    print(f"    [derive] 2 diurnal periods -> window {two:g} d, duration "
          f"{two:g} d, {got} samples")


def test_the_window_unit_toggle_round_trips_exactly():
    """Same property, on the window control and its two period units.

    The exactness matters MORE here than on the duration row, because the
    periods are large: a hop through orbital periods on Europa divides by
    4332.589 and multiplies back, and any conversion that went through the
    six-decimal display instead of the stored value would return a different
    number.
    """
    manager, start = _fresh()
    card = start.open_card("Europa", "global", "integrated",
                           apply_defaults=True)
    key = "thermal.integration_days"
    manager.set_value(key, 3.551181)
    card.activate("global", "integrated")
    before = manager.get_value(key)
    assert before == 3.551181, (
        f"activating the card rewrote the window: 3.551181 -> {before!r}")

    state = card._duration_controls[key]
    seen = {}
    for unit in ("orbital periods", "days", "diurnal periods",
                 "orbital periods", "days"):
        state["combo"].setCurrentText(unit)
        _qt().processEvents()
        seen[unit] = state["spin"].value()
        assert manager.get_value(key) == before, (
            f"toggling the display to {unit!r} changed the stored window: "
            f"{before!r} -> {manager.get_value(key)!r}. The units control is "
            f"display-only and must never write.")
    # The DISPLAY is rounded to the control's decimals -- that is what the
    # rounding is for -- so each reading is checked against the true ratio to
    # within half a digit. The STORED value, asserted above, is exact.
    resolution = 0.5 * 10.0 ** -state["spin"].decimals()
    for unit, period in (("diurnal periods", 3.55),
                         ("orbital periods", 4332.589)):
        assert abs(seen[unit] - before / period) <= resolution, (
            f"the {unit} reading is {seen[unit]}, not {before / period} -- "
            f"further off than the {resolution} the display rounds to")
    print(f"    [round trip] {before} d survives 5 window-unit toggles; "
          f"orbital reads {seen['orbital periods']:.6f}")


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
    for unit in ("hours", "diurnal periods", "days", "hours", "days"):
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


#: One window in each unit, per route, with the days it MUST emit -- computed
#: here from the periods rather than from the code under test.
#:
#: The two periods are different questions and the asteroid case is the one
#: that proves it: 3.55 d is Europa's day AND the engine's ``props.year``
#: (``planetary_properties.h:189``, "Same as day for synchronous rotation"),
#: while its orbital period is Jupiter's 4332.589 d. Anything that resolved
#: "orbital periods" from the engine's year would pass every other case here
#: and be wrong by a factor of 1220 on Europa.
_MOON_DAY = 29.530588          # planetary_properties.h, Moon solar day
_EUROPA_DAY = 3.55             # ...synchronous rotation
_MERCURY_DAY = 175.938         # ...3 x the 58.646 d sidereal rotation
_MOON_YEAR = 365.25            # the Moon orbits the Sun with the Earth
_EUROPA_YEAR = 4332.589        # Jupiter's sidereal orbital period
_MERCURY_YEAR = 87.969

#: A generic body at 2.5 AU, computed from Kepler's third law WRITTEN OUT --
#: 2 pi sqrt(a^3 / GM_sun) -- on the engine's own constants
#: (``orbital_elements.h``: gm::SUN, AU_TO_M). Not a call to the function
#: under test.
_ASTEROID_A_AU = 2.5
_ASTEROID_YEAR = (2.0 * 3.141592653589793 * (
    ((2.5 * 1.495978707e11) ** 3 / 1.32712440018e20) ** 0.5)) / 86400.0

WINDOW_CASES = (
    # (body, mode, count, unit, expected days)
    ("Moon", "global", 1.0, "diurnal periods", _MOON_DAY),
    ("Moon", "global", 3.0, "orbital periods", 3.0 * _MOON_YEAR),
    ("Moon", "local", 0.5, "days", 0.5),
    ("Europa", "global", 1.0, "diurnal periods", _EUROPA_DAY),
    ("Europa", "global", 1.0, "orbital periods", _EUROPA_YEAR),
    ("Europa", "local", 2.0, "orbital periods", 2.0 * _EUROPA_YEAR),
    ("Mercury", "global", 2.0, "diurnal periods", 2.0 * _MERCURY_DAY),
    ("Mercury", "global", 1.0, "orbital periods", _MERCURY_YEAR),
)


def test_the_window_units_emit_the_days_they_promise():
    """What the user chose, in days, in the key the engine reads.

    The unit is a DISPLAY choice and ``thermal.integration_days`` is always
    days, so the whole content of the control is this multiplication. Each
    expectation is computed above from the period, not from the code being
    checked.
    """
    for body, mode, count, unit, expected in WINDOW_CASES:
        manager, start = _fresh()
        card = start.open_card(body, mode, "integrated", apply_defaults=True)
        state = card._duration_controls["thermal.integration_days"]
        state["combo"].setCurrentText(unit)
        _qt().processEvents()
        assert state["combo"].currentText() == unit, (
            f"{body}/{mode}: the window control refused the unit {unit!r}, "
            f"which every body this card serves has a period for")
        state["spin"].setValue(count)
        _qt().processEvents()
        got = manager.get_value("thermal.integration_days")
        assert abs(got - expected) <= 1e-9 * expected, (
            f"{body}/{mode}: {count:g} {unit} emitted {got!r} d, not the "
            f"{expected!r} d that many periods is")
        assert manager.get_value("simulation.duration_days") == got, (
            f"{body}/{mode}: the output window did not follow the "
            f"{count:g} {unit} window")

        # ...and the control SAYS what a period is worth on this body, in
        # days, where the choice is made: on the combo and under the row.
        one = f"{expected / count:g}"
        tip, note = state["combo"].toolTip(), card.window_note.text()
        assert one in tip, (
            f"{body}/{mode}: the unit help does not resolve {unit} to "
            f"{one} d for this body: {tip!r}")
        assert one in note, (
            f"{body}/{mode}: the note under the window does not say what "
            f"{unit} is worth here: {note!r}")
    print(f"    [units] {len(WINDOW_CASES)} hand-computed cases emit their "
          f"own days; each is stated in the unit help and under the row")


def test_an_asteroids_orbital_period_comes_from_its_own_semi_major_axis():
    """Kepler's third law, live: edit `a` and the window the card emits moves.

    A generic body has no row in any period table -- it has an orbit. The
    period is computed from ``generic_body.orbit.semi_major_axis_au`` by the
    formula the ENGINE uses (``OrbitalElements::orbitalPeriod``), so there is
    nothing cached to go stale when the semi-major axis changes.
    """
    manager, start = _fresh()
    manager.set_value("generic_body.orbit.semi_major_axis_au", _ASTEROID_A_AU)
    manager.set_value("generic_body.spin.period_hours", 7.0)
    card = start.open_card("asteroid", "body", "integrated",
                           apply_defaults=True)
    state = card._duration_controls["thermal.integration_days"]
    state["combo"].setCurrentText("orbital periods")
    _qt().processEvents()
    state["spin"].setValue(1.0)
    _qt().processEvents()
    got = manager.get_value("thermal.integration_days")
    assert abs(got - _ASTEROID_YEAR) <= 1e-9 * _ASTEROID_YEAR, (
        f"one orbit at a = {_ASTEROID_A_AU} AU emitted {got!r} d, not the "
        f"{_ASTEROID_YEAR!r} d Kepler's third law gives")

    # ...and moving the semi-major axis moves the period, with nothing cached.
    # a -> 4a is P -> 8P exactly, by the three-halves power.
    manager.set_value("generic_body.orbit.semi_major_axis_au",
                      4.0 * _ASTEROID_A_AU)
    state["spin"].setValue(2.0)                # re-emit at the new period
    state["spin"].setValue(1.0)
    _qt().processEvents()
    moved = manager.get_value("thermal.integration_days")
    assert abs(moved - 8.0 * _ASTEROID_YEAR) <= 1e-9 * moved, (
        f"after a -> 4a the same 1 orbital period emitted {moved!r} d; "
        f"Kepler's third law makes it exactly 8x, {8.0 * _ASTEROID_YEAR!r} d")

    # ...and the DISPLAY follows too, with no cached period: a window of the
    # old orbit is one eighth of the new one, and the card says so as soon as
    # it is shown again.
    manager.set_value("thermal.integration_days", _ASTEROID_YEAR)
    card.activate("body", "integrated")
    state["combo"].setCurrentText("orbital periods")
    _qt().processEvents()
    shown = state["spin"].value()
    assert abs(shown - 0.125) < 1e-6, (
        f"the old one-orbit window reads {shown} orbital periods against the "
        f"new 4x semi-major axis, not the 0.125 it now is -- the period the "
        f"card converts with has gone stale")
    print(f"    [kepler] a = {_ASTEROID_A_AU} AU -> {got:.1f} d; "
          f"a = {4 * _ASTEROID_A_AU} AU -> {moved:.1f} d (exactly 8x); the "
          f"same window then reads {shown:g} orbits")


def test_no_sentinel_is_reachable_from_the_integrated_card():
    """Paul's first rule, as geometry: -1 cannot be typed, dialed or shown.

    The floor of the control is a real duration in every unit it offers, and
    the seeded value is positive on every route. There is no combination of
    unit and spin box that produces the engine's "use the output window"
    sentinel, so a user cannot arrive at a window that has no end time.
    """
    from gui_core.body_cards import WINDOW_UNITS

    for body, mode in ROUTES + (("asteroid", "body"),):
        manager, start = _fresh()
        if body == "asteroid":
            manager.set_value("generic_body.spin.period_hours", 7.0)
            manager.set_value("generic_body.orbit.semi_major_axis_au", 2.5)
        card = start.open_card(body, mode, "integrated", apply_defaults=True)
        state = card._duration_controls["thermal.integration_days"]
        assert set(state["units"]) == set(WINDOW_UNITS), (
            f"{body}/{mode}: the window offers {state['units']}")
        seeded = manager.get_value("thermal.integration_days")
        assert seeded > 0, (
            f"{body}/{mode}: the card seeded {seeded!r}, which is the "
            f"sentinel and not a window")
        for unit in state["units"]:
            state["combo"].setCurrentText(unit)
            _qt().processEvents()
            assert state["spin"].minimum() > 0, (
                f"{body}/{mode}: in {unit!r} the window control's floor is "
                f"{state['spin'].minimum()}, so a sentinel is reachable")
            # Try to type one anyway. The spin box clamps, and what is stored
            # must stay a real window.
            state["spin"].setValue(-1.0)
            _qt().processEvents()
            stored = manager.get_value("thermal.integration_days")
            assert stored > 0, (
                f"{body}/{mode}: typing -1 with {unit!r} selected stored "
                f"{stored!r}")
    print(f"    [no sentinel] {len(ROUTES) + 1} routes x "
          f"{len(WINDOW_UNITS)} units: floor positive, -1 unreachable")


def test_an_opened_sentinel_becomes_the_window_it_already_means():
    """A file carrying -1 is resolved, and the card says so.

    -1 is the engine's default and every shipped file that mentions the key
    uses it, so an opened configuration WILL carry it. The card cannot show it
    -- the whole point above -- so it writes the length -1 already stands for:
    ``simulation.duration_days``. The run is unchanged, because
    ``computeIntegrationWindow`` takes ``integration_days > 0 ?
    integration_days : output_days`` (``ptm_thermal_sim.cpp:1158``) and both
    branches now produce the same double.

    The resolution is ANNOUNCED. A silent one would be the substitution this
    project keeps paying for.
    """
    manager, start = _fresh()
    manager.set_value("simulation.body", "Moon")
    manager.set_value("simulation.duration_days", 12.5)
    manager.set_value("thermal.integration_days", -1.0)
    card = start.open_card("Moon", "global", "integrated",
                           apply_defaults=False)
    resolved = manager.get_value("thermal.integration_days")
    assert resolved == 12.5, (
        f"the opened -1 became {resolved!r}, not the 12.5 d output window it "
        f"already meant")
    note = card.window_note.text()
    assert "12.5" in note and "unchanged" in note, (
        f"the card resolved the sentinel without saying so: {note!r}")
    print(f"    [sentinel] -1 over a 12.5 d run resolved to 12.5 d, announced: "
          f"{note[:60]}...")


def test_a_sentinel_arriving_after_the_card_is_open_is_resolved_too():
    """All settings still offers -1, and the card is reloaded, not reopened.

    ``BodyFirstPanel.refresh_active_card`` calls ``load_from_config`` alone,
    so a resolution that ran only from ``activate`` would leave the window
    control displaying its own floor over a configuration that says "integrate
    the whole output window".
    """
    manager, start = _fresh()
    card = start.open_card("Moon", "global", "integrated", apply_defaults=True)
    window = manager.get_value("thermal.integration_days")
    assert window > 0

    manager.set_value("thermal.integration_days", -1.0)   # as All settings does
    start.refresh_active_card()
    _qt().processEvents()
    resolved = manager.get_value("thermal.integration_days")
    assert resolved > 0, (
        f"a -1 set after the card was open survived the reload as {resolved!r}")
    assert resolved == manager.get_value("simulation.duration_days"), (
        f"the resolved window {resolved!r} is not the output window "
        f"{manager.get_value('simulation.duration_days')!r} it stood for")
    shown = card._duration_controls["thermal.integration_days"]["spin"].value()
    assert shown > card._duration_controls[
        "thermal.integration_days"]["spin"].minimum(), (
        f"the control is sitting at its floor ({shown}), which is what it "
        f"shows when it cannot display what the configuration holds")
    print(f"    [reload] a -1 set from All settings resolved to {resolved:g} d "
          f"on the next card reload")


def test_a_period_that_goes_away_takes_its_unit_with_it():
    """The unit is re-resolved, not trusted from when it was chosen.

    A unit outlives the number behind it: choose "diurnal periods" on an
    asteroid, then clear the rotation period. The silent failure is that the
    box goes on reading "diurnal periods" and stores a typed 2 as two DAYS,
    which ``derive_from_duration`` then copies into the run length.
    """
    manager, start = _fresh()
    manager.set_value("generic_body.spin.period_hours", 7.0)
    manager.set_value("generic_body.orbit.semi_major_axis_au", 2.5)
    card = start.open_card("asteroid", "body", "integrated",
                           apply_defaults=True)
    state = card._duration_controls["thermal.integration_days"]
    assert state["combo"].currentText() == "diurnal periods", (
        state["combo"].currentText())

    # The rotation period is cleared -- on the Body page, or by a lookup.
    manager.set_value("generic_body.spin.period_hours", 0.0)

    # THE WRITE PATH, with no reload in between: an edit must be refused, not
    # read as days under a label that says periods.
    state["spin"].setValue(2.0)
    _qt().processEvents()
    assert state["combo"].currentText() == "days", (
        f"the combo still reads {state['combo'].currentText()!r} with no "
        f"period behind it")
    assert manager.get_value("thermal.integration_days") != 2.0, (
        "a 2 typed under 'diurnal periods' was stored as 2 days")
    note = card.window_note.text()
    assert "diurnal periods" in note and "period_hours" in note, (
        f"the refusal does not say what went missing: {note!r}")
    print("    [stale unit] a cleared rotation period refuses its own unit on "
          "the next edit, and says which key is missing")


def test_editing_the_semi_major_axis_moves_what_the_units_are_worth():
    """The "nothing cached" claim, with the hook that makes it true."""
    manager, start = _fresh()
    manager.set_value("generic_body.spin.period_hours", 7.0)
    manager.set_value("generic_body.orbit.semi_major_axis_au", 2.5)
    card = start.open_card("asteroid", "body", "integrated",
                           apply_defaults=True)
    before = card.window_note.text()
    assert f"{_ASTEROID_YEAR:g}" in before, before

    card.on_widget_changed("generic_body.orbit.semi_major_axis_au", 10.0)
    _qt().processEvents()
    after = card.window_note.text()
    assert f"{8.0 * _ASTEROID_YEAR:g}" in after, (
        f"after a -> 4a the card still says an orbit is "
        f"{_ASTEROID_YEAR:g} d: {after!r}")
    print(f"    [live] editing a from 2.5 to 10 AU moved the stated orbit "
          f"from {_ASTEROID_YEAR:.0f} d to {8 * _ASTEROID_YEAR:.0f} d")


def test_a_window_too_long_to_sample_says_so():
    """`samples_for_window` caps at 100000, and the cap is not silent.

    Two Europa orbits is 8665 days; keeping 144 samples per 3.55 d diurnal
    cycle would need about 351,000 of them. The run is still correct at the
    cap -- what would be wrong is the promised density quietly not holding.
    """
    from gui_core.body_cards import SAMPLES_PER_SYNODIC_DAY

    manager, start = _fresh()
    card = start.open_card("Europa", "global", "integrated",
                           apply_defaults=True)
    state = card._duration_controls["thermal.integration_days"]
    state["combo"].setCurrentText("orbital periods")
    _qt().processEvents()
    state["spin"].setValue(2.0)
    _qt().processEvents()
    card.refresh_window_units()
    assert manager.get_value("simulation.timesteps") == 100000, (
        f"expected the sample count to reach its cap, got "
        f"{manager.get_value('simulation.timesteps')!r}")
    note = card.window_note.text()
    assert "capped at 100,000" in note and str(SAMPLES_PER_SYNODIC_DAY) in note, (
        f"the sampling cap was applied without saying so: {note!r}")
    print("    [cap] 2 Europa orbits: samples capped at 100,000 and the card "
          "says the density is no longer 144 per diurnal cycle")


def test_a_file_with_neither_a_window_nor_a_length_is_not_called_unchanged():
    """The one branch where "the run is unchanged" would be a false claim.

    -1 with no positive ``simulation.duration_days`` meant the ENGINE's own
    output window, which the card cannot claim to be preserving. It sets both
    keys and words the note as the change it is.
    """
    manager, start = _fresh()
    manager.set_value("simulation.body", "Moon")
    manager.set_value("simulation.duration_days", 0.0)
    manager.set_value("thermal.integration_days", -1.0)
    card = start.open_card("Moon", "global", "integrated",
                           apply_defaults=False)
    window = manager.get_value("thermal.integration_days")
    assert window == 29.530588, window
    assert manager.get_value("simulation.duration_days") == window, (
        "the run length was left at 0 while the window was set, so the engine "
        "would integrate over a span it does not simulate")
    note = card.window_note.text()
    assert "unchanged" not in note, (
        f"the card claims the run is unchanged when it has just given it a "
        f"length: {note!r}")
    print("    [no length] -1 over a run with no length sets both keys and "
          "does not claim the run is unchanged")


def test_the_window_note_does_not_outlive_the_control_it_describes():
    """A label is state, and state that is only ever appended to goes stale.

    "Window: 29.5306 d in total" is true under a window control and false
    under a Duration control, which is what sits there on the time-series run
    type. The note is rebuilt from state on every refresh rather than left
    alone, so switching run types cannot leave the previous run type's
    sentence on screen.
    """
    manager, start = _fresh()
    card = start.open_card("Moon", "global", "integrated", apply_defaults=True)
    assert "Window:" in card.window_note.text(), card.window_note.text()

    card.activate("global", "time_series")
    _qt().processEvents()
    assert "Window:" not in card.window_note.text(), (
        f"the integrated run type's window description survived onto the time "
        f"series card: {card.window_note.text()!r}")
    assert not card.window_note.isVisible() \
        or not card.window_note.text(), card.window_note.text()
    print("    [stale note] the window description does not survive a switch "
          "to a run type with no window control")


def test_a_unit_with_no_period_is_refused_not_approximated():
    """A body with no orbit cannot be given a window in orbits.

    The wrong answer here is silent: read "2" as two days when the user asked
    for two orbits. The combo goes back to days and the card says why.
    """
    manager, start = _fresh()
    manager.set_value("generic_body.spin.period_hours", 7.0)
    manager.set_value("generic_body.orbit.semi_major_axis_au", 0.0)
    card = start.open_card("asteroid", "body", "integrated",
                           apply_defaults=True)
    state = card._duration_controls["thermal.integration_days"]
    before = manager.get_value("thermal.integration_days")
    state["combo"].setCurrentText("orbital periods")
    _qt().processEvents()
    assert state["combo"].currentText() == "days", (
        f"a body with no semi-major axis accepted the unit "
        f"{state['combo'].currentText()!r}")
    assert manager.get_value("thermal.integration_days") == before, (
        "refusing the unit changed the stored window")
    note = card.window_note.text()
    assert "orbital periods" in note and "semi_major_axis_au" in note, (
        f"the refusal does not say what is missing: {note!r}")
    print(f"    [refusal] no semi-major axis: orbital periods refused, days "
          f"kept, reason shown")


def test_showing_a_duration_in_another_unit_does_not_rewrite_it():
    """Stored duration in display units, closed. This one CORRUPTED files.

    ``BaseConfigTab.set_numeric`` writes the coerced number back to the
    configuration on purpose -- it is how a YAML string like ``5e3`` stops
    being a str that raises on the next comparison. The card's override
    converted the value to the DISPLAY unit first and then let that base call
    write it, so every reload of a card whose unit combo was off days divided
    the stored duration by the period again.

    Measured on the base commit 4e00f8a, Moon, one diurnal cycle::

        after open   29.530588       after reload  1.0
        after unit   29.530588       after 2nd     0.033863192971301484

    Compounding, silent, and it is ``simulation.duration_days`` -- the length
    of the run. Four reloads here, because two were needed to see it.
    """
    manager, start = _fresh()
    card = start.open_card("Moon", "global", "time_series",
                           apply_defaults=True)
    key = "simulation.duration_days"
    before = manager.get_value(key)
    card._duration_controls[key]["combo"].setCurrentText("diurnal periods")
    _qt().processEvents()
    for reload_number in range(1, 5):
        card.activate("global", "time_series")
        _qt().processEvents()
        assert manager.get_value(key) == before, (
            f"reload {reload_number} rewrote the duration: {before!r} -> "
            f"{manager.get_value(key)!r}. The display unit is reaching the "
            f"configuration.")
    print(f"    [no rewrite] {before} d survives 4 card reloads with the "
          f"unit combo off days")


def test_every_duration_control_shows_the_same_precision():
    """Duration shown at four decimals vs six, closed.

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
    """Unread-config-key defect, held by re-derivation rather than by memory.

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
        f"change no result (unread-config-key defect)")
    # ...and every *_override the engine reads is offered, so a property the
    # engine gained does not sit unreachable.
    missing = {k for k in engine if k.endswith("_override")} - offered
    assert not missing, (
        f"the engine reads {sorted(missing)} but no card control offers them")
    print(f"    [unread keys] {len(offered)} overrides offered, all in the "
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
    test_the_window_units_emit_the_days_they_promise,
    test_an_asteroids_orbital_period_comes_from_its_own_semi_major_axis,
    test_no_sentinel_is_reachable_from_the_integrated_card,
    test_an_opened_sentinel_becomes_the_window_it_already_means,
    test_a_sentinel_arriving_after_the_card_is_open_is_resolved_too,
    test_a_file_with_neither_a_window_nor_a_length_is_not_called_unchanged,
    test_a_period_that_goes_away_takes_its_unit_with_it,
    test_editing_the_semi_major_axis_moves_what_the_units_are_worth,
    test_a_window_too_long_to_sample_says_so,
    test_the_window_note_does_not_outlive_the_control_it_describes,
    test_a_unit_with_no_period_is_refused_not_approximated,
    test_the_window_unit_toggle_round_trips_exactly,
    test_the_duration_unit_toggle_round_trips_exactly,
    test_showing_a_duration_in_another_unit_does_not_rewrite_it,
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
