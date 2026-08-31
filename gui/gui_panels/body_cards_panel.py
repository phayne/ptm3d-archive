"""The body-first surface: a landing screen, a run-type step, and one card.

Three screens, in this order (DP-10):

    body  ->  model domain and run type  ->  one card, for that body and that
                                             run type

and, from every card, one control back to the eleven-section panel, which is
unchanged and still the complete view of the configuration.

WHY THIS EXISTS. The panel offers 145 controls. The shipped corpus says a
whole-Moon run is a seventeen-key decision and a Europa DEM site is a two-key
decision, and 115 of the engine's 184 keys never take a second value in any of
the 78 shipped configurations. The gap between those numbers is what these
cards close. ``docs/design/gui-body-first-redesign.md`` is the measurement.

FOUR RULES, each of which a previous stage paid for:

* **A card writes only what it shows, plus body defaults it announces.** Every
  card applies its defaults through ``gui_core.presets.apply_preset``, which
  returns the exact dotted keys it wrote, and the card prints them. That is the
  same contract the task cards had, and the reason it is reused rather than
  reinvented is that it is already pinned by tests.
* **Nothing hidden is silently dropped or silently applied.** Opening a file
  lands on its body's card and lists every key the card does not show that
  differs from the engine default (``body_cards.out_of_card_settings``).
* **An automatic choice is announced and reversible.** Auto-albedo and the
  beaming facet-scale rule both narrate what they did and why.
* **A control that cannot be honored is not offered.** The body buttons come
  from the engine's own implemented-body table; the parameter-grid run type
  appears only where a grid is actually backed by shipped configurations.

REGISTER. Section headers and control labels are technical terms, not
questions: "Local mesh", not "Which piece of ground"; "Output", not "Where it
goes". Explanatory prose belongs in the tooltips, which keep the plain-language
what/recommended/when standard. `test_gui_body_first.py` holds that line.
"""

import logging
from pathlib import Path

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QFontMetrics
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFrame, QGridLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton,
    QRadioButton, QScrollArea, QSpinBox, QStackedWidget, QVBoxLayout, QWidget,
)

from gui_core.body_cards import (GENERIC_MESH_SHAPE, WINDOW_UNIT_DAYS,
                                 WINDOW_UNIT_DIURNAL, WINDOW_UNIT_ORBITAL,
                                 WINDOW_UNITS, window_unit_days)
from gui_core.config_manager import ConfigManager
from gui_panels.config_panel import BaseConfigTab, get_tooltip

# The engine's own generic-body surface defaults. Same two-step import as
# `gui_core.body_cards`: `ptm` is on the path in an installed tree and needs
# the bootstrap in a source checkout.
try:
    from ptm.body_capabilities import (GENERIC_BODY_SURFACE_DEFAULTS,
                                       GENERIC_SURFACE_FIELDS)
except ImportError:                                       # pragma: no cover
    from gui_core import _ptm_bootstrap           # noqa: F401  (adds src/python)
    from ptm.body_capabilities import (GENERIC_BODY_SURFACE_DEFAULTS,
                                       GENERIC_SURFACE_FIELDS)

logger = logging.getLogger(__name__)


#: Rough single-run wall clock, in minutes, used ONLY to size a sweep
#: confirmation. Deliberately crude and labeled as an estimate everywhere it is
#: shown: the real number depends on the mesh, the view factors and whether the
#: GPU pool is available, none of which this screen can know before the run.
_MINUTES_PER_RUN_GUESS = 25.0


def _muted(text, color="gray", size=10):
    label = QLabel(text)
    label.setWordWrap(True)
    # PLAIN TEXT, explicitly. These labels display text that came off the
    # network verbatim -- HORIZONS' refusals, DAMIT's replies, and on an HTTP
    # error the response BODY, which from a proxy or a CDN is HTML. A QLabel
    # defaults to Qt.TextFormat.AutoText, which renders anything
    # `mightBeRichText()` accepts AS rich text: the error the user needs to
    # read gets silently mangled, and Qt's rich text can reference remote
    # resources. Seven other labels in this GUI already pin PlainText for the
    # same reason; these four -- the lookup, coverage, shape and provenance
    # notes -- were the ones that did not, and they are the ones the fetches
    # write into.
    label.setTextFormat(Qt.TextFormat.PlainText)
    label.setStyleSheet(f"color: {color}; font-size: {size}px;")
    return label


# ===========================================================================
# The card base
# ===========================================================================

class BodyCard(BaseConfigTab):
    """One body's options card.

    A `BaseConfigTab` on purpose: the field helpers already carry the tooltip
    contract, the YAML-spelling tolerance that keeps a malformed value from
    aborting the window, and the write-back discipline. Registering each card
    in ``ConfigurationPanel.tabs`` then puts it inside the tooltip-coverage,
    scalar-spelling and tier gates automatically, which is the point.

    A card only mirrors the configuration while it is the ACTIVE route.
    Cards share keys with each other and with the section tabs -- five cards
    all bind ``simulation.start_time`` -- and ``load_from_config`` writes the
    coerced value back, so an inactive card loading a value would let whichever
    card ran last decide the stored precision. Measured hazard, closed by
    construction.
    """

    #: Set by subclasses.
    ROUTE = ""

    def __init__(self, config_manager, route, panel=None):
        self.route = route
        self.panel = panel
        self._active = False
        self._loading = False
        self._mode = ""
        self._run_type = "time_series"
        self._report = None
        self._notes = []
        #: True only while a card is being opened for a NEW run, i.e. under
        #: `open_card(..., apply_defaults=True)`. A card may seed a default it
        #: shows -- see `EuropaCard._ensure_terrain_emitted` -- ONLY while this
        #: is set. Displaying an opened file is not an edit to it, and the same
        #: `activate` runs on both paths.
        self._seeding_defaults = False
        super().__init__(config_manager)

    # -- activation ---------------------------------------------------------

    def activate(self, mode_key, run_type_key, seeding_defaults=False):
        """Make this the live card, for one mode and one run type."""
        self._active = True
        self._mode = mode_key
        self._run_type = run_type_key
        self._seeding_defaults = bool(seeding_defaults)
        #: The two things the window note can say beyond what the units are
        #: worth: a sentinel that was resolved, and a unit that was refused.
        #: Both are shown until this card is opened again, and both are cleared
        #: HERE, at the one point that means "a new opening" -- not inside the
        #: methods that set them, which run more than once per opening.
        self._window_resolution_note = ""
        self._window_refusal_note = ""
        try:
            self.apply_mode()
            # NOT inside `apply_mode`: every card overrides that and none of
            # them calls super(), so a run-type rule put there would apply to
            # whichever cards happened to remember it.
            self.apply_time_mode()
            self.load_from_config()
        finally:
            self._seeding_defaults = False

    def deactivate(self):
        self._active = False

    def apply_mode(self):
        """Show the groups this mode and run type use. Subclasses override."""

    def exposed_keys(self):
        """Every configuration key this card shows, for the disclosure of
        anything it does not."""
        keys = set(self.widgets)
        keys.update(self.extra_exposed_keys())
        return keys

    def extra_exposed_keys(self):
        """Keys a card writes through a control that is not bound one-to-one
        (a terrain picker writing both a mesh and its view-factor cache)."""
        return set()

    # -- reporting ----------------------------------------------------------

    def _make_report(self, layout):
        self._report = QLabel("")
        self._report.setWordWrap(True)
        self._report.setTextFormat(Qt.TextFormat.PlainText)
        self._report.setStyleSheet(
            "font-size: 10px; padding: 4px; border: 1px solid #444; "
            "border-radius: 4px;")
        self._report.setVisible(False)
        layout.addWidget(self._report)

    def show_report(self, text):
        if self._report is None:
            return
        self._report.setText(text)
        self._report.setVisible(bool(text))

    def report_text(self):
        return self._report.text() if self._report is not None else ""

    # -- helpers shared by the cards ----------------------------------------

    def _project_root(self):
        from gui_core.preflight import find_project_root
        from gui_core.simulation_runner import SimulationRunner
        try:
            root = find_project_root(SimulationRunner.find_ptm_executable())
        except Exception:                                    # noqa: BLE001
            root = None
        return root or Path.cwd()

    def _number(self, key, fallback):
        return ConfigManager._as_number(
            self.config_manager.get_value(key, fallback), fallback)

    #: The engine builds an icosphere as `20 * 4**level`. Above about nine that
    #: is tens of millions of faces and no longer a run anybody starts.
    MAX_ICOSPHERE_LEVEL = 9

    def icosphere_note(self, key="mesh.resolution", fallback=5):
        """`ico<N> - <faces> faces`, or a refusal to compute one.

        Bounded on purpose. `mesh.resolution` is read from the configuration,
        not from the spin box -- the box clamps its DISPLAY to the range it
        offers while the configuration deliberately keeps the file's own value
        -- so a config carrying `resolution: 1e5` (which YAML hands back as the
        string "1e5", and which the engine's own parser reads happily) reached
        `20 * 4**100000`. That is a 60,000-digit integer, and formatting it
        raises ValueError under Python's integer-string limit, out of a Qt
        slot, which is an application abort. Found by the card arm of the
        scalar-spelling sweep.
        """
        raw = self._number(key, fallback)
        level = int(raw)
        if level < 0 or level > self.MAX_ICOSPHERE_LEVEL or level != raw:
            return (f"mesh.resolution = {raw:g} is outside the range this "
                    f"control offers (0 to {self.MAX_ICOSPHERE_LEVEL}). The "
                    f"configuration keeps that value and the face count is not "
                    f"shown for it; edit the field to change it.")
        return f"ico{level} — {20 * (4 ** level):,} faces"

    # -- durations, and the units they are shown in --------------------------

    #: Decimals for EVERY duration control, in every unit, on every screen.
    #:
    #: It is one number on purpose. ``simulation.duration_days`` used to be
    #: shown at FOUR decimals by the Simulation tab and at SIX by these cards
    #: (GUI-DURATION-DECIMALS-4-VS-6), so the same configuration read
    #: differently depending on which screen you were on: the shipped Europa
    #: duration 3.551181 d displayed as "3.5512" on one and "3.551181" on the
    #: other. Measured 2026-08-28, the stored value survives -- displaying does
    #: NOT rewrite the configuration, because the write-back is guarded -- so
    #: this is a fidelity defect and not a corruption. It still matters: the
    #: rounded number is what the user EDITS from, and at the old two-decimal
    #: hours display, 3.551181 d showed as 85.23 h, which is 3.551250 d.
    DURATION_DECIMALS = 6

    #: Display units offered beside a plain duration. The CONFIGURATION KEY IS
    #: ALWAYS DAYS; these change the number shown and nothing else.
    #:
    #: "diurnal periods", not "synodic periods": they name the same span -- the
    #: body's solar day -- and the card already says "diurnal" everywhere else
    #: ("one diurnal cycle", "Spin-up (diurnal cycles)"). One word for one
    #: thing. The window control beside it offers the same word, so the two
    #: rows cannot read as two different periods.
    DURATION_UNITS = ("days", "hours", WINDOW_UNIT_DIURNAL)

    #: Hours, which is the one display unit that is not a period of anything
    #: and so needs no body to resolve.
    DURATION_UNIT_HOURS = "hours"

    #: The units that ARE a period of something. They step by half a period
    #: and their worth depends on the body; the rest step by one and do not.
    PERIOD_UNITS = (WINDOW_UNIT_DIURNAL, WINDOW_UNIT_ORBITAL)

    #: The configuration key the window control is bound to. Named once: five
    #: methods below reach for it, and a card that spelled it differently in
    #: one of them would resolve a sentinel it then never displayed.
    WINDOW_KEY = "thermal.integration_days"

    #: Shortest integration window the card will offer, in days. A real
    #: duration, not the engine's -1: about 1.4 minutes, far below any run
    #: anybody starts and far above zero.
    WINDOW_FLOOR_DAYS = 0.001

    #: Every run type a card can be opened on. A duration row that names a
    #: subset is only offered -- and only loaded -- on those.
    ALL_RUN_TYPES = ("time_series", "integrated", "sweep")

    def _duration_state(self):
        state = getattr(self, "_duration_controls", None)
        if state is None:
            state = self._duration_controls = {}
        return state

    def _synodic_days(self):
        """One synodic day for the body this card is showing, in days.

        Through :meth:`_unit_days`, so the card has ONE answer to "how long is
        a day here". It used to ask ``gui_tooltips.get_synodic_period_days``
        while the display conversion asked ``body_cards.synodic_period_days``,
        two lines apart -- and the two took different routes for a generic
        body. 0 when there is none, which every caller already handles.
        """
        days, _why = self._unit_days(WINDOW_UNIT_DIURNAL)
        return float(days) if days else 0.0

    def _unit_days(self, unit):
        """``(days_per_unit, provenance)`` for a display unit on THIS body.

        The single place a unit name becomes a number, for both directions of
        every duration control on every card. ``hours`` is here rather than in
        ``gui_core.body_cards`` because it is not a period of anything -- it is
        a unit of time, and needs no body to resolve.

        ``(None, why)`` when the body has no such period. Callers must refuse
        rather than fall back to days: a control that silently read "2" as two
        days when the user asked for two orbits would be off by a factor of
        1220 on Europa.
        """
        if unit == self.DURATION_UNIT_HOURS:
            return 1.0 / 24.0, self.DURATION_UNIT_HOURS
        return window_unit_days(unit, self.route,
                                self.config_manager.get_config())

    def _per_unit(self, key):
        """Days per display unit for one control, or 1.0 when it cannot say.

        The 1.0 is a LAST RESORT and never a silent reading of a period as
        days: every path that can act on an unresolvable unit -- the edit, the
        unit change, and the activation sweep -- refuses first, so what reaches
        here is a control already on days or one being torn down.
        """
        unit = self._duration_state().get(key, {}).get("unit", WINDOW_UNIT_DAYS)
        per_unit, _why = self._unit_days(unit)
        return per_unit or 1.0

    def _days_to_display(self, key, days):
        return days / self._per_unit(key)

    def _display_to_days(self, key, value):
        return value * self._per_unit(key)

    def duration_row(self, grid, row, label_text, key, minimum=0.001,
                     units=None, unit_tooltip="ui.duration_units",
                     run_types=None):
        """A duration spin box with a display-unit combo beside it.

        Returns ``(label, container)`` so the caller can show or hide the whole
        row: which duration a card offers depends on the run type.

        `units` picks the vocabulary. A plain duration takes
        :data:`DURATION_UNITS`; the integration window takes
        ``body_cards.WINDOW_UNITS``, which drops hours (nobody sets a
        statistics window in hours) and adds the body's orbit.

        **The toggle round-trips exactly, because nothing flows back through
        it.** Changing the unit re-reads the canonical value from the
        configuration and re-displays it; it never converts the number that is
        already in the box and writes the result back. A toggle chain
        days -> hours -> diurnal periods -> days therefore returns the original
        value bit for bit, however coarse the display is, because the display
        was never the source. The alternative -- converting what is shown --
        loses a digit per hop at any finite `decimals`.
        """
        label = QLabel(label_text)
        self._apply_tooltip(label, key)
        label.setToolTip(get_tooltip(key))
        grid.addWidget(label, row, 0)

        container = QWidget()
        box = QHBoxLayout(container)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(4)

        spin = QDoubleSpinBox()
        spin.setDecimals(self.DURATION_DECIMALS)
        spin.setSingleStep(1.0)
        spin.setToolTip(get_tooltip(key))
        box.addWidget(spin, stretch=1)

        offered = tuple(units or self.DURATION_UNITS)
        combo = QComboBox()
        combo.addItems(offered)
        combo.setMaximumWidth(140)
        combo.setToolTip(get_tooltip(unit_tooltip))
        box.addWidget(combo)

        grid.addWidget(container, row, 1)
        self.widgets[key] = spin
        self._duration_state()[key] = {
            "unit": offered[0], "spin": spin, "combo": combo,
            "label": label, "container": container, "minimum": float(minimum),
            "units": offered, "unit_tooltip": unit_tooltip,
            "run_types": tuple(run_types or self.ALL_RUN_TYPES)}
        self._apply_duration_range(key)

        spin.valueChanged.connect(
            lambda value, k=key: self._on_duration_edited(k, value))
        combo.currentTextChanged.connect(
            lambda unit, k=key: self._on_duration_unit_changed(k, unit))
        return label, container

    def _apply_duration_range(self, key):
        """Range the spin box in the unit it is currently displaying.

        The BOUNDS are converted by exactly the same function as the value.
        Converting one and not the other is how a control ends up unable to
        display what it holds -- a floor pinned at 0 for one unit and converted
        for another shows a number the configuration does not carry.
        """
        state = self._duration_state().get(key)
        if state is None:
            return
        spin, unit = state["spin"], state["unit"]
        low = self._days_to_display(key, state["minimum"])
        # ...AND NEVER BELOW ONE DIGIT OF THE DISPLAY. A QDoubleSpinBox rounds
        # its own bounds to `decimals`, so a floor that converts to less than
        # that becomes exactly 0 -- and 0 is a sentinel too: the engine tests
        # `integration_days > 0.0` (ptm_thermal_sim.cpp:1158), so a window
        # dialed to zero would silently mean "the whole output window" again.
        # Europa's 0.001 d floor is 2.3e-7 orbital periods, which is how this
        # was found.
        if state["minimum"] > 0:
            low = max(low, 10.0 ** -spin.decimals())
        blocked = spin.blockSignals(True)
        try:
            spin.setRange(low, self._days_to_display(key, 365000.0))
            spin.setSingleStep(0.5 if unit in self.PERIOD_UNITS else 1.0)
        finally:
            spin.blockSignals(blocked)

    def _on_duration_edited(self, key, display_value):
        """The user moved a duration control. Convert to days and store.

        The unit is re-resolved on every edit, not trusted from when it was
        chosen. A period can go away underneath a selected unit -- clear a
        generic body's rotation period, or its semi-major axis, and the combo
        still reads "diurnal periods" -- and the WRITE path is where that
        turns from a wrong label into a wrong configuration: "2" would be
        stored as 2 days and copied into the run length.
        """
        if self._loading or getattr(self, "_unit_syncing", False):
            # Showing a configuration is not an edit to it -- the same rule
            # `load_from_config` states, applied to the unit conversion so a
            # card that merely DISPLAYS a duration cannot rewrite it.
            return
        state = self._duration_state().get(key)
        if state is not None:
            per_unit, why = self._unit_days(state["unit"])
            if not per_unit:
                self._refuse_window_unit(key, state["unit"], why)
                return
        self.on_widget_changed(key, self._display_to_days(key, display_value))
        self.derive_from_duration(key)

    def _on_duration_unit_changed(self, key, unit):
        """Re-display the stored value in a new unit. Writes nothing.

        A unit this body has no period for is REFUSED, not approximated. The
        combo snaps back to days and the card says why, because the quiet
        alternative -- reading the number in the box as days when the user
        asked for orbits -- is wrong by the ratio of the two periods, which on
        Europa is a factor of 1220.
        """
        state = self._duration_state().get(key)
        if state is None:
            return
        per_unit, why = self._unit_days(unit)
        if not per_unit:
            self._refuse_window_unit(key, unit, why)
            return
        self._window_refusal_note = ""
        state["unit"] = unit
        self._unit_syncing = True
        try:
            self._apply_duration_range(key)
            days = self._number(key, 0.0)
            spin = state["spin"]
            blocked = spin.blockSignals(True)
            try:
                spin.setValue(self._days_to_display(key, days))
            finally:
                spin.blockSignals(blocked)
        finally:
            self._unit_syncing = False
        self.refresh_window_units()

    def _refuse_window_unit(self, key, unit, why):
        """Put the combo back to days and SAY SO. Never a silent fallback.

        The message names the ROW, because the card has two duration controls
        and both can offer a period unit, and it is shown whatever the run
        type -- a refusal written where the current run type does not draw it
        is exactly the silent fallback this method exists to prevent.
        """
        state = self._duration_state().get(key)
        if state is None:
            return
        combo = state["combo"]
        blocked = combo.blockSignals(True)
        try:
            combo.setCurrentText(WINDOW_UNIT_DAYS)
        finally:
            combo.blockSignals(blocked)
        # Days always resolves, so this branch of `_on_duration_unit_changed`
        # cannot recurse back into here; it re-ranges and re-displays, and
        # clears the refusal note, which is why the note is written after it.
        self._on_duration_unit_changed(key, WINDOW_UNIT_DAYS)
        row = state["label"].text().rstrip(":")
        self._window_refusal_note = (
            f"{row}: {unit} cannot be used here — {why}. The value is shown "
            f"in days instead.")
        self.refresh_window_units()

    def set_numeric(self, key, value):
        """Load a number, converting a duration into the unit on display.

        AND PUT THE STORED VALUE BACK IN DAYS. ``BaseConfigTab.set_numeric``
        deliberately writes the coerced number to the configuration -- it is
        how a YAML string like ``5e3`` becomes a float instead of a str that
        raises on the next comparison (``config_panel.py:624``). Combined with
        a display conversion, that wrote the DISPLAY units into the
        configuration key: showing a 29.530588 d window in diurnal periods
        stored 1.0, and the next load showed 1 d.

        Measured on the base commit 4e00f8a, where it needed the unit combo to
        have been moved off days before a reload -- so it was reachable, not
        theoretical, and it silently divided ``simulation.duration_days`` by
        the synodic day. It became unmissable here only because the window
        control now OPENS in diurnal periods. Reported as
        GUI-DURATION-DISPLAY-UNITS-STORED.
        """
        state = self._duration_state().get(key)
        if state is None:
            super().set_numeric(key, value)
            return
        if self._run_type not in state["run_types"]:
            # A control this run type does not offer is not displayed, so
            # there is nothing to convert and nothing to say. Loading it
            # anyway is how the hidden integration window logged "-0.0338 is
            # outside the range" on every time-series card: the number in the
            # message was the display conversion of a -1 the configuration was
            # right to be keeping.
            return
        try:
            canonical = float(value)
        except (TypeError, ValueError):
            # Not a number: let the base implementation warn and leave the box.
            super().set_numeric(key, value)
            return
        super().set_numeric(key, self._days_to_display(key, canonical))
        # The key is days, in every unit. Written explicitly rather than left
        # to the base call, whose whole point is that it must write something.
        self.config_manager.set_value(key, canonical)

    #: Keys a window unit's PERIOD is computed from. Editing any of them
    #: changes what "1 diurnal period" or "1 orbital period" is worth, so the
    #: card re-resolves rather than going on describing the old body.
    PERIOD_KEYS = ("generic_body.orbit.semi_major_axis_au",
                   "generic_body.spin.period_hours",
                   "simulation.body")

    def on_widget_changed(self, key, value):
        """Store an edit, and re-resolve the periods if it moved one."""
        super().on_widget_changed(key, value)
        if key in self.PERIOD_KEYS and not self._loading:
            self._revalidate_duration_units()
            self.refresh_window_units()

    def derive_from_duration(self, key):
        """Write the keys the integrated card no longer asks for.

        On the integrated run type the integration window is the ONLY duration
        the user sets, so the two keys that used to be controls are derived
        from it here:

        * ``simulation.duration_days`` = the window. The engine simulates
          ``max(output_days, offset + integ_days)`` after equilibration
          (`ptm_thermal_sim.cpp:15975`), so making the output window equal to
          the integration window is what keeps the run from integrating over
          a span it did not simulate -- or simulating one it does not use.
        * ``simulation.timesteps`` = the window at the corpus sampling
          cadence. It is not dead on this run type: with no per-sample files
          written it still sets the cadence the statistics window is counted
          in (`computeIntegrationWindow`, `:1134`), so leaving it at whatever
          the previous run type used would sample a 200-day window with the
          144 steps chosen for a 3.55-day one.
        """
        if self._run_type != "integrated" or key != self.WINDOW_KEY:
            return
        from gui_core.body_cards import samples_for_window

        window = self._number(self.WINDOW_KEY, 0.0)
        if not window > 0:
            # Unreachable from the control -- its floor is a real duration --
            # so this can only be a configuration that arrived carrying the
            # engine's -1. `_resolve_integration_window` turns that into the
            # length it already means BEFORE anything is derived from it.
            return
        self.config_manager.set_value("simulation.duration_days", window)
        self.config_manager.set_value(
            "simulation.timesteps",
            samples_for_window(window, self._synodic_days()))
        # The window length is also what a window SERIES is made of, so the
        # series plan and its refusals are re-derived here rather than going
        # on describing the previous window.
        self.refresh_window_series()

    def time_group(self, title="Time window"):
        """The time controls, which depend on WHAT THE RUN PRODUCES.

        A time series is a window plus a sampling density, so it carries a
        duration and an output-sample count. Integrated statistics are not
        sampled for output at all -- every per-sample file is skipped -- so
        offering "duration" and "output samples" beside "integration window"
        asked the user for three numbers to describe one window, two of which
        only mattered through their ratio. Paul's ruling: for that run type the
        INTEGRATION WINDOW is the only duration control.

        The other two keys are still written, because the engine still reads
        them (`ptm_thermal_sim.cpp:15958`, `computeIntegrationWindow` at
        `:1134`): `simulation.duration_days` is the output window that
        `post_eq_days_needed` is taken from, and `simulation.timesteps` sets
        the cadence the statistics window is counted in. They are DERIVED here
        rather than asked for -- see `derive_from_duration` -- so the emission
        stays a valid, complete configuration.

        NO SENTINEL REACHES THIS SCREEN. The integration window's floor is a
        real duration, so the engine's -1 ("integrate the output window")
        cannot be typed, dialed or displayed here. It has no honest reading on
        a card whose only duration control IS the window: -1 means "as long as
        the run", and the run is as long as this control says.

        The window is given as a COUNT and a PERIOD -- 1 diurnal period, 2
        orbital periods -- because that is how a run length is decided.
        `body_cards.WINDOW_UNITS` says why there is no separate "seasonal"
        entry.
        """
        group = QGroupBox(title)
        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        self.add_string_field(grid, 0, "Start (UTC):", "simulation.start_time")
        self._duration_rows = {
            "simulation.duration_days": self.duration_row(
                grid, 1, "Duration:", "simulation.duration_days"),
        }
        self.add_int_field(grid, 2, "Output samples:", "simulation.timesteps",
                           1, 100000)
        self._samples_row = (grid.itemAtPosition(2, 0).widget(),
                             self.widgets["simulation.timesteps"])
        self._duration_rows[self.WINDOW_KEY] = self.duration_row(
            grid, 3, "Integration window:", self.WINDOW_KEY,
            minimum=self.WINDOW_FLOOR_DAYS, units=WINDOW_UNITS,
            unit_tooltip="ui.window_units", run_types=("integrated",))
        #: What the chosen unit resolves to for THIS body, and any refusal.
        #: Under the row rather than only in the tooltip, because the number a
        #: unit stands for is the whole content of the choice.
        self.window_note = _muted("")
        grid.addWidget(self.window_note, 4, 0, 1, 2)
        self._build_window_series_rows(grid, 5)
        group.setLayout(grid)
        return group

    # -- a long run, broken into windows (the A3GT pattern) ------------------

    #: Windows a new series starts with. A3GT's own production year set is 4
    #: years x 2 seasons; four is a set somebody would actually look at.
    DEFAULT_WINDOW_COUNT = 4

    def _build_window_series_rows(self, grid, row):
        """The split-into-windows controls, under the integration window.

        THE SPAN IS NOT A CONFIGURATION KEY, and that is deliberate. The engine
        has no "span" -- a window series is N separate runs, each with its own
        `simulation.start_time` and `simulation.duration_days` -- so binding a
        span control to a dotted key would write a key into the emitted YAML
        that nothing reads, which is the defect PTM-67 names. It lives on the
        card, in days, and is converted for display through the SAME
        `_unit_days` every other duration on this card converts through.
        """
        self.split_check = QCheckBox("Split the run into windows")
        self.split_check.setToolTip(get_tooltip("ui.split_windows"))
        self.split_check.toggled.connect(self._on_split_toggled)
        grid.addWidget(self.split_check, row, 0, 1, 2)

        count_label = QLabel("Number of windows:")
        count_label.setToolTip(get_tooltip("ui.window_count"))
        grid.addWidget(count_label, row + 1, 0)
        self.window_count = QSpinBox()
        self.window_count.setRange(1, 64)
        self.window_count.setValue(self.DEFAULT_WINDOW_COUNT)
        self.window_count.setToolTip(get_tooltip("ui.window_count"))
        self.window_count.valueChanged.connect(
            lambda _v: self.refresh_window_series())
        grid.addWidget(self.window_count, row + 1, 1)

        span_label = QLabel("Total span:")
        span_label.setToolTip(get_tooltip("ui.window_span"))
        grid.addWidget(span_label, row + 2, 0)
        span_box = QWidget()
        span_layout = QHBoxLayout(span_box)
        span_layout.setContentsMargins(0, 0, 0, 0)
        span_layout.setSpacing(4)
        self.span_spin = QDoubleSpinBox()
        self.span_spin.setDecimals(self.DURATION_DECIMALS)
        self.span_spin.setRange(0.001, 365000.0)
        self.span_spin.setToolTip(get_tooltip("ui.window_span"))
        self.span_spin.valueChanged.connect(self._on_span_edited)
        span_layout.addWidget(self.span_spin, stretch=1)
        self.span_unit_combo = QComboBox()
        self.span_unit_combo.addItems(WINDOW_UNITS)
        self.span_unit_combo.setMaximumWidth(140)
        self.span_unit_combo.setToolTip(get_tooltip("ui.window_units"))
        self.span_unit_combo.currentTextChanged.connect(
            self._on_span_unit_changed)
        span_layout.addWidget(self.span_unit_combo)
        grid.addWidget(span_box, row + 2, 1)

        self.series_note = _muted("", color="#64b5f6", size=11)
        grid.addWidget(self.series_note, row + 3, 0, 1, 2)

        self.series_button = QPushButton("Write window configurations…")
        self.series_button.setToolTip(get_tooltip("ui.window_series"))
        self.series_button.clicked.connect(self._on_prepare_window_series)
        grid.addWidget(self.series_button, row + 4, 1)

        #: The span, IN DAYS, which is the canonical unit here as everywhere
        #: else on this card. 0 means "not chosen yet"; the first activation
        #: proposes one.
        self._span_days = 0.0
        self._window_series_rows = (
            self.split_check, count_label, self.window_count, span_label,
            span_box, self.series_note, self.series_button)

    def _span_unit_days(self):
        """Days per span unit, or 1.0 when the body has no such period.

        The 1.0 is a LAST RESORT and never a silent reading of a period as
        days -- the same rule and the same wording as :meth:`_per_unit`. Both
        paths that can act on an unresolvable unit, the edit and the unit
        change, refuse first, so what reaches here is a control already on
        days.
        """
        per_unit, why = self._unit_days(self.span_unit_combo.currentText())
        return (per_unit or 1.0), why

    def _on_span_edited(self, value):
        if self._loading or getattr(self, "_span_syncing", False):
            return
        per_unit, _why = self._span_unit_days()
        self._span_days = float(value) * per_unit
        self.refresh_window_series()

    def _on_span_unit_changed(self, unit):
        """Re-display the span in a new unit. Refuses one this body has not.

        The same rule the duration rows follow, and for the same reason: the
        quiet alternative -- reading the number in the box as days when the
        user asked for orbits -- is wrong by the ratio of the two periods,
        which on Europa is a factor of 1220.
        """
        if getattr(self, "_span_syncing", False):
            return
        per_unit, why = self._unit_days(unit)
        if not per_unit:
            self._span_syncing = True
            try:
                self.span_unit_combo.setCurrentText(WINDOW_UNIT_DAYS)
            finally:
                self._span_syncing = False
            self.series_note.setText(
                f"Total span: {unit} cannot be used here — {why}. The span is "
                f"shown in days instead.")
            per_unit = 1.0
        self._show_span()
        self.refresh_window_series()

    def _show_span(self):
        """Put the stored span in the box without treating that as an edit."""
        per_unit, _why = self._span_unit_days()
        self._span_syncing = True
        try:
            blocked = self.span_spin.blockSignals(True)
            try:
                self.span_spin.setValue(self._span_days / per_unit)
            finally:
                self.span_spin.blockSignals(blocked)
        finally:
            self._span_syncing = False

    def _on_split_toggled(self, _checked):
        self.refresh_window_series()

    def apply_window_series_mode(self):
        """Show the series controls on the integrated run type only.

        A window series IS an integration: each window produces the summary
        maps and no per-sample files, which is what the integrated run type
        means. Offering it on a time series would write N configurations whose
        per-sample output nothing here merges.
        """
        rows = getattr(self, "_window_series_rows", None)
        if not rows:
            return
        integrated = self._run_type == "integrated"
        self.split_check.setVisible(integrated)
        split = integrated and self.split_check.isChecked()
        for widget in rows[1:]:
            widget.setVisible(split)
        if integrated and not self._span_days > 0:
            self._propose_span()
        self._show_span()
        self.refresh_window_series()

    def _propose_span(self):
        """A first span: the windows laid end to end, so nothing overlaps.

        Deliberately the SHORTEST span the chosen count fits in rather than a
        body-specific number. A card that proposed "18.6 years" would be
        proposing the Moon's nodal cycle on Mercury and on every asteroid.
        """
        window = self._number(self.WINDOW_KEY, 0.0)
        if window > 0:
            self._span_days = window * max(1, self.window_count.value())

    def refresh_window_series(self):
        """Plan the series and say what it is, or why there is none.

        Recomputed from the live configuration on every change, so the note
        under the controls always describes the set the button would write.
        """
        if not getattr(self, "_window_series_rows", None):
            return
        if self._run_type != "integrated" or not self.split_check.isChecked():
            note = getattr(self, "series_note", None)
            # `isHidden`, not `isVisible`. A widget in a window that has not
            # been shown yet is not VISIBLE but is not HIDDEN either, and this
            # runs during activation, before the card is on screen -- so
            # `isVisible` would suppress the advice on exactly the path that
            # produces it. The card tests use `isHidden` for the same reason.
            if note is not None and not self.split_check.isHidden():
                from gui_core.window_series import split_advice
                note.setText(split_advice(self._number(self.WINDOW_KEY, 0.0),
                                          self._synodic_days()))
                note.setVisible(bool(note.text()))
            return
        plan = self.window_series_plan()
        hours = len(plan) * _MINUTES_PER_RUN_GUESS / 60.0
        if plan.refusal:
            self.series_note.setText(plan.refusal)
            self.series_button.setEnabled(False)
        else:
            first, last = plan.windows[0], plan.windows[-1]
            self.series_note.setText(
                f"{len(plan)} windows of {plan.window_days:g} days across "
                f"{plan.span_days:g} days, {first.start_iso} to "
                f"{last.end_iso()}. Each is a complete, independent run with "
                f"its own spin-up — they sample the span, they do not continue "
                f"one another. Rough estimate {hours:.0f} h in series on this "
                f"machine. " + " ".join(plan.notes))
            self.series_button.setEnabled(True)
        self.series_note.setVisible(True)

    def window_series_plan(self):
        """The `window_series.WindowPlan` this card's controls describe."""
        from gui_core.window_series import plan_windows
        return plan_windows(
            self.config_manager.get_value("simulation.start_time", ""),
            self._span_days,
            self.window_count.value(),
            self._number(self.WINDOW_KEY, 0.0),
            synodic_days=self._synodic_days())

    def _on_prepare_window_series(self):
        from gui_core.window_series import confirmation_text
        plan = self.window_series_plan()
        if plan.refusal:
            self.show_report(plan.refusal)
            return
        directory = QFileDialog.getExistingDirectory(
            self, "Output directory for the window series", "",
            QFileDialog.Option.ShowDirsOnly)
        if not directory:
            return
        answer = QMessageBox.question(
            self, "Write window configurations?",
            confirmation_text(plan, directory, _MINUTES_PER_RUN_GUESS),
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel)
        if answer != QMessageBox.StandardButton.Ok:
            self.show_report("Window series canceled. Nothing was written.")
            return
        try:
            written, commands = self.write_window_series(directory)
        except Exception as exc:                             # noqa: BLE001
            self.show_report(f"The window series could not be written: {exc}")
            return
        lines = [
            f"Wrote {len(written)} window configurations under {directory},",
            "one per window, plus windows.json recording the schedule and how",
            "its epochs were chosen. Nothing has been launched. Run them",
            "detached, so the series survives closing this window:",
            "",
        ]
        lines.extend("    " + " ".join(command) for command in commands[:3])
        if len(commands) > 3:
            lines.append(f"    ... and {len(commands) - 3} more, one per "
                         f"window; every command is in windows.json.")
        self.show_report("\n".join(lines))

    def write_window_series(self, outdir):
        """Write the series. Returns ``(config paths, runner commands)``.

        The configurations go through ``ConfigManager.save_run_yaml``, which is
        the SAME writer every other emitted run YAML uses -- including the path
        absolutization the engine needs when a configuration is written
        somewhere other than the project directory. A second serializer here is
        how two files that should be identical stop being identical.
        """
        import json

        from gui_core import window_series

        plan = self.window_series_plan()
        if plan.refusal:
            raise ValueError(plan.refusal)
        root = self._project_root()
        base = self.config_manager.get_config()

        def save(config, path):
            """Serialize one window through ConfigManager's own run writer.

            The manager's configuration is SWAPPED for the window's and put
            back, rather than a second serializer being written here.
            `save_run_yaml` is not just a yaml.dump: it absolutizes every path
            field (the engine resolves a relative view-factor path against the
            YAML's own directory, so a configuration that moves must carry
            absolute ones) and it applies `_config_for_yaml`'s albedo-source
            translation. A window written any other way would be a different
            run from the one the card shows.
            """
            manager = self.config_manager
            previous = manager.config
            try:
                manager.config = config
                manager.save_run_yaml(str(path), root)
            finally:
                manager.config = previous

        written = window_series.write_window_series(
            base, plan, outdir, project_root=root, save=save)
        commands = window_series.runner_commands(plan, outdir,
                                                 config_dir=outdir)
        schedule = Path(outdir) / "windows.json"
        document = json.loads(schedule.read_text())
        document["commands"] = [" ".join(command) for command in commands]
        schedule.write_text(json.dumps(document, indent=2) + "\n")
        return written, commands

    def apply_time_mode(self):
        """Show the duration controls this run type actually uses."""
        rows = getattr(self, "_duration_rows", None)
        if not rows:
            return
        integrated = self._run_type == "integrated"
        for widget in rows["simulation.duration_days"]:
            widget.setVisible(not integrated)
        for widget in self._samples_row:
            widget.setVisible(not integrated)
        for widget in rows[self.WINDOW_KEY]:
            widget.setVisible(integrated)
        if integrated:
            self._resolve_integration_window()
            self._start_window_in_its_own_unit()
        self._revalidate_duration_units()
        self.refresh_window_units()
        self.apply_window_series_mode()

    def _revalidate_duration_units(self):
        """Drop any selected unit whose period has gone away, out loud.

        A unit is chosen once and lives as long as the card does, while the
        period behind it comes from the configuration and can be edited or
        cleared -- a generic body's rotation period, its semi-major axis, or
        simply a different body arriving on the same reused card. Checked on
        every activation so a stale unit is caught when the card is shown
        rather than when the user next types into it.
        """
        for key, state in list(self._duration_state().items()):
            if self._run_type not in state["run_types"]:
                continue
            per_unit, why = self._unit_days(state["unit"])
            if not per_unit:
                self._refuse_window_unit(key, state["unit"], why)

    def _resolve_integration_window(self):
        """Replace a sentinel integration window with the length it means.

        A configuration can arrive carrying ``thermal.integration_days: -1``
        -- it is the engine's default and the value every shipped file that
        mentions the key uses. On the integrated run type the card's only
        duration control IS that key, so -1 has to become a number before it
        can be shown at all.

        It is written, not merely displayed, and the SAME RUN comes out:
        ``computeIntegrationWindow`` takes ``integration_days > 0 ?
        integration_days : output_days`` (``ptm_thermal_sim.cpp:1158``) and
        ``requested_integ_days`` does the same at ``:15958``, so replacing -1
        with ``simulation.duration_days`` puts the identical double into both.
        The card SAYS it did it -- a resolution nobody is told about is the
        silent substitution this project keeps paying for.
        """
        window = self._number(self.WINDOW_KEY, 0.0)
        if window > 0:
            # Nothing to resolve. The note is NOT cleared here: this method
            # runs again from `load_from_config`, after the resolution it just
            # made, and clearing on that second pass is how the disclosure
            # disappeared before it reached the screen. `activate` clears it,
            # once per opening.
            return
        from gui_core.body_cards import default_duration_days
        output = self._number("simulation.duration_days", 0.0)
        if output > 0:
            # The equivalent case, and the only one where the run really is
            # untouched. ONE KEY is written, and only the one being resolved:
            # `derive_from_duration` would also rewrite
            # `simulation.timesteps` to this card's sampling cadence --
            # measured at 144 -> 61 on a 12.5 d file -- and that is the
            # author's output cadence, not a restatement of their sentinel.
            self.config_manager.set_value(self.WINDOW_KEY, output)
            self._window_resolution_note = (
                f"This file asked for the whole run window ({window:g}); that "
                f"is {output:g} days here, the length it already sets, and "
                f"the window below now says so. The run is unchanged.")
            return

        # No run length either. -1 then meant "the engine's own output
        # window", which is not a number this card can claim to be preserving,
        # so BOTH keys are set to the default window and the note says the run
        # has been given a length rather than that it is unchanged.
        replacement, why = default_duration_days(
            self.route, self.config_manager.get_config())
        if not replacement or not replacement > 0:
            self._window_resolution_note = (
                f"This file sets no run length and asks for the whole window "
                f"({window:g}), and this body has no known period to propose "
                f"one from — {why}. Type a window below before running.")
            return
        self.config_manager.set_value(self.WINDOW_KEY, replacement)
        self.config_manager.set_value("simulation.duration_days", replacement)
        self._window_resolution_note = (
            f"This file asked for the whole run window ({window:g}) without "
            f"setting a run length. Both are now {replacement:g} days — "
            f"{why}. Change the window below if that is not what you want.")

    def _start_window_in_its_own_unit(self):
        """Open the window control in diurnal periods where there are any."""
        from gui_core.body_cards import default_window_unit
        state = self._duration_state().get(self.WINDOW_KEY)
        if state is None:
            return
        unit = default_window_unit(self.route, self.config_manager.get_config())
        if unit == state["unit"]:
            return
        combo = state["combo"]
        blocked = combo.blockSignals(True)
        try:
            combo.setCurrentText(unit)
        finally:
            combo.blockSignals(blocked)
        self._on_duration_unit_changed(self.WINDOW_KEY, unit)

    def refresh_window_units(self):
        """Say what each window unit is worth on THIS body, and in what.

        On the combo, because that is the control the question is about, and in
        the note under the row, because a tooltip nobody hovers is not a
        disclosure. Both are recomputed from the live configuration on every
        call, and :meth:`on_widget_changed` calls it whenever a key a period
        depends on moves -- so editing a generic body's semi-major axis moves
        the orbital period this card offers, with nothing cached in between.
        """
        state = self._duration_state().get(self.WINDOW_KEY)
        if state is None:
            return
        resolved = []
        for unit in state["units"]:
            if unit == WINDOW_UNIT_DAYS:
                continue
            days, why = self._unit_days(unit)
            resolved.append(f"1 {unit[:-1]} = {days:g} d ({why})"
                            if days else f"{unit}: {why}")
        # Joined BEFORE the call: `setToolTip` may not be handed a string
        # literal, not even a separator, or the single-source gate reads it as
        # tooltip prose authored in a panel.
        detail = "; ".join(resolved)
        tooltip_key = state["unit_tooltip"]
        state["combo"].setToolTip(get_tooltip(tooltip_key).format(detail))

        note = getattr(self, "window_note", None)
        if note is None:
            return
        # ONE place builds this label, from state, every time -- so a line that
        # stopped being true (the window description after a switch to a run
        # type that has no window control) cannot survive by being left alone.
        # Neither note is CONSUMED here: this runs two or three times per
        # activation, once directly and once per unit change, and a note
        # consumed by the first of those would never reach the screen. They are
        # cleared by `activate` and by the next successful unit change.
        lines = [text for text in (getattr(self, "_window_refusal_note", ""),
                                   getattr(self, "_window_resolution_note", ""))
                 if text]
        if self._run_type in state["run_types"]:
            days = self._number(self.WINDOW_KEY, 0.0)
            if days > 0:
                lines.append(f"Window: {days:g} days in total.")
            lines.extend(resolved)
            capped = self._sampling_cap_note()
            if capped:
                lines.append(capped)
        note.setText("  ".join(lines))
        note.setVisible(bool(lines))

    def _sampling_cap_note(self):
        """Say when the derived output-sample count hit its ceiling.

        ``samples_for_window`` keeps the sampling density at
        ``SAMPLES_PER_SYNODIC_DAY`` per diurnal cycle, bounded at 100000 --
        which a window of a few orbital periods reaches easily now that the
        control offers them: two Europa orbits is 8665 days, or 351,000
        samples at the corpus cadence. The bound is right, and the run is
        still correct; what would be wrong is the user not being told the
        density they were promised is no longer the one they are getting.
        """
        from gui_core.body_cards import (SAMPLES_PER_SYNODIC_DAY,
                                         samples_for_window)
        window = self._number(self.WINDOW_KEY, 0.0)
        synodic = self._synodic_days()
        if not (window > 0 and synodic > 0):
            return ""
        wanted = SAMPLES_PER_SYNODIC_DAY * window / synodic
        got = samples_for_window(window, synodic)
        if wanted <= got + 0.5:
            return ""
        return (f"This window is long enough that the output samples are "
                f"capped at {got:,}: {round(wanted):,} would be needed to "
                f"keep {SAMPLES_PER_SYNODIC_DAY} samples per diurnal cycle, "
                f"so the statistics are counted more coarsely than that.")

    def thermal_group(self, title="Thermal"):
        """The thermal-solver controls, in one section.

        THIS SECTION EXISTS BECAUSE ITS CONTROLS WERE FILED BY ACCIDENT.
        ``thermal.use_gpu`` sat in the output section next to
        ``output.directory``, where nothing about it belongs: it selects the
        device the heat-equation solver runs on, not where the run's files
        land. ``thermal.equilibration_orbits`` had a one-control section of its
        own titled "How long to settle". Both are ``thermal.*`` solver
        settings, and the split this draws -- solver here, destination in
        Output -- is the one the All settings view already draws between its
        Thermal and Output pages.

        ``thermal.integration_days`` used to be added here by an
        `integration_window` flag. It moved to :meth:`time_group`, because it
        is a DURATION and on the integrated card it is THE duration.
        """
        group = QGroupBox(title)
        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        spin = self.add_double_field(
            grid, 0, "Spin-up (diurnal cycles):",
            "thermal.equilibration_orbits", 0.0, 100000.0, decimals=2)
        spin.setSingleStep(1.0)
        #: Says where the spin-up number came from and why. Filled in by
        #: `announce_auto_equilibration` when a card seeds its defaults, and
        #: left in place afterwards -- the control stays editable, and the note
        #: describes the value that was PROPOSED, not the one now in the box.
        self.equilibration_note = _muted("")
        grid.addWidget(self.equilibration_note, 1, 0, 1, 2)
        self.add_bool_field(grid, 2, "Run the thermal solver on the GPU",
                            "thermal.use_gpu")
        group.setLayout(grid)
        return group

    def announce_auto_equilibration(self, cycles, reason):
        """Show the automatic spin-up with its provenance."""
        note = getattr(self, "equilibration_note", None)
        if note is not None:
            note.setText(f"auto: {cycles:g} cycles — {reason}. Editable; "
                         f"change the number above to override it.")

    # -- the built-in bodies' surface, via the keys the engine READS ---------

    #: ``(config key, body-table field, label, low, high, decimals)`` for every
    #: surface property a BUILT-IN body can actually override.
    #:
    #: The suffix is the whole point. On a built-in body the engine reads
    #: ``thermophysical.albedo_override`` and IGNORES ``thermophysical.albedo``:
    #: the parser's own read-list is `albedo_hapke_w, albedo_model,
    #: albedo_override, conductivity_model, constant_thermal_inertia,
    #: emissivity_override, H_override, incidence_angle_albedo, kd_override,
    #: ks_override, use_default_albedo` (`config_parser.cpp:170-182`), and
    #: anything else under that block draws a "read by nothing and have NO
    #: effect" warning at parse time. That is PTM-67, and it is why the
    #: Properties tab's `thermophysical.ks` cannot be reused here -- those keys
    #: are honored only under `generic_body.surface:`.
    #:
    #: FIVE properties, not the nine the Properties tab shows. There is no
    #: `rhos_override`, `rhod_override`, `cp0_override` or `Qb_override` in the
    #: engine, so a control for them would be exactly the silently-ignored
    #: field this list exists to avoid.
    BUILTIN_OVERRIDES = (
        ("thermophysical.albedo_override", "albedo",
         "Bond albedo:", 0.0, 1.0, 4),
        ("thermophysical.emissivity_override", "emissivity",
         "Emissivity:", 0.0, 1.0, 4),
        ("thermophysical.ks_override", "ks",
         "Surface conductivity (W/m/K):", 0.0, 100.0, 6),
        ("thermophysical.kd_override", "kd",
         "Deep conductivity (W/m/K):", 0.0, 100.0, 6),
        ("thermophysical.H_override", "H",
         "Density e-folding depth H (m):", 0.0, 10.0, 4),
    )

    #: The value every one of those keys carries when it is NOT overriding.
    #: The engine tests `> 0` (and `>= 0` for albedo) against this sentinel,
    #: `config_parser.h:732-748`.
    OVERRIDE_UNSET = -1.0

    def builtin_surface_group(self, body_key, title="Surface properties"):
        """The five engine-honored overrides, with the engine's own values in.

        Each row is a number and a "Use the engine's value" tick. Ticked -- the
        state a new run starts in -- the key is written as
        :data:`OVERRIDE_UNSET` and the engine uses its built-in property; the
        box still SHOWS that built-in value, from the generated body table, so
        the card discloses what the run will use instead of showing a blank or
        a sentinel. Unticked, the number beside it is written and the run uses
        that.

        The tick is deliberately NOT a configuration key: the engine has no
        `use_default_*_override`, and inventing one would put a key in the
        emitted YAML that nothing reads -- the very defect PTM-67 names.
        """
        from ptm.body_table import BODY_CAPABILITIES

        engine = BODY_CAPABILITIES.get(body_key, {})
        group = QGroupBox(title)
        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        self._override_rows = {}

        for row, (key, field, label_text, low, high, decimals) in enumerate(
                self.BUILTIN_OVERRIDES):
            label = QLabel(label_text)
            self._apply_tooltip(label, key)
            label.setToolTip(get_tooltip(key))
            grid.addWidget(label, row, 0)

            spin = QDoubleSpinBox()
            spin.setRange(low, high)
            spin.setDecimals(decimals)
            spin.setToolTip(get_tooltip(key))
            grid.addWidget(spin, row, 1)

            tick = QCheckBox("Use the engine's value")
            tick.setToolTip(get_tooltip("ui.use_body_default"))
            grid.addWidget(tick, row, 2)

            default = engine.get(field)
            try:
                default = float(default)
            except (TypeError, ValueError):
                default = None
            self.widgets[key] = spin
            self._override_rows[key] = {
                "spin": spin, "tick": tick, "engine": default, "field": field}

            spin.valueChanged.connect(
                lambda value, k=key: self._on_override_edited(k, value))
            tick.toggled.connect(
                lambda checked, k=key: self._on_override_default_toggled(
                    k, checked))

        self.override_note = _muted("")
        grid.addWidget(self.override_note, len(self.BUILTIN_OVERRIDES), 0, 1, 3)
        group.setLayout(grid)
        self._refresh_override_note()
        return group

    def _override_state(self):
        return getattr(self, "_override_rows", {}) or {}

    def _on_override_edited(self, key, value):
        if self._loading or getattr(self, "_override_syncing", False):
            return
        state = self._override_state().get(key)
        if state is not None and state["tick"].isChecked():
            # Editing the number means the user wants their number.
            state["tick"].setChecked(False)
            return                        # the toggle writes it
        self.on_widget_changed(key, value)
        self._refresh_override_note()

    def _on_override_default_toggled(self, key, use_engine):
        state = self._override_state().get(key)
        if state is None:
            return
        state["spin"].setEnabled(not use_engine)
        if self._loading or getattr(self, "_override_syncing", False):
            return
        if use_engine:
            self.on_widget_changed(key, self.OVERRIDE_UNSET)
            if state["engine"] is not None:
                self._show_override(key, state["engine"])
        else:
            self.on_widget_changed(key, state["spin"].value())
        self._refresh_override_note()

    def _show_override(self, key, value):
        """Put a number in an override box without writing anything."""
        state = self._override_state().get(key)
        if state is None:
            return
        self._override_syncing = True
        try:
            spin = state["spin"]
            blocked = spin.blockSignals(True)
            try:
                spin.setValue(float(value))
            finally:
                spin.blockSignals(blocked)
        finally:
            self._override_syncing = False

    def load_overrides(self):
        """Mirror the five override keys, sentinel and all."""
        self._override_syncing = True
        try:
            for key, state in self._override_state().items():
                stored = self._number(key, self.OVERRIDE_UNSET)
                overriding = stored > 0 or (
                    key.endswith("albedo_override") and stored >= 0)
                tick = state["tick"]
                blocked = tick.blockSignals(True)
                try:
                    tick.setChecked(not overriding)
                finally:
                    tick.blockSignals(blocked)
                state["spin"].setEnabled(overriding)
                shown = stored if overriding else state["engine"]
                if shown is not None:
                    self._show_override(key, shown)
        finally:
            self._override_syncing = False
        self._refresh_override_note()

    def _refresh_override_note(self):
        """Say, in one line, which properties the run is taking from where."""
        note = getattr(self, "override_note", None)
        if note is None:
            return
        edited = [key.split(".")[-1].replace("_override", "")
                  for key, state in sorted(self._override_state().items())
                  if not state["tick"].isChecked()]
        if edited:
            note.setText("Provenance: " + ", ".join(edited)
                         + " set on this card; the rest are engine defaults.")
        else:
            note.setText("Provenance: engine default — every value above is "
                         "the engine's own for this body, and none of them is "
                         "written to the configuration.")

    def output_group(self, title="Output"):
        """Where the run's files land, and nothing else."""
        group = QGroupBox(title)
        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        self.add_directory_field(grid, 0, "Output directory:",
                                 "output.directory")
        group.setLayout(grid)
        return group

    # -- radiation exchange, on EVERY card and in every run type -------------

    def radiation_group(self, title="Radiation exchange", beaming=False):
        """The `radiation.enabled` switch, its cache, and optionally beaming.

        ONE SECTION, BUILT ONCE, USED BY EVERY CARD. It was previously written
        out by hand on two of the five cards and only in their LOCAL mode, so
        `radiation.enabled` -- the switch that decides whether a shadowed
        crater floor is warmed by its walls at all -- was unreachable on
        Mercury, on every generic body (which is every asteroid), and on both
        global modes, in every run type. A key the engine reads on every body
        that no screen offers is the same defect as a control the engine
        ignores, arrived at from the other side.

        IT IS OFFERED IN GLOBAL MODE TOO, and that is deliberate. On a smooth
        procedural icosphere the switch is a no-op that costs minutes and
        gigabytes -- measured on 2026-08-28, a 1.13M-block H-matrix in which
        every block was Zero. The answer to that is the pre-flight warning
        already in `gui_core.preflight` (see "Radiation exchange on a convex
        mesh"), which names the cost and the fix and does NOT switch anything
        off. Hiding the control instead would mean an opened configuration
        with `enabled: true` on a global mesh showed nothing at all, which is
        how it stays on.

        `beaming` adds the sub-facet roughness switch. Only the Moon card
        passes it: `facet_scale.BEAMING_BODIES` is lunar-only, and a control
        whose rule refuses it on every configuration a card can produce is a
        control that should not be on that card.
        """
        group = QGroupBox(title)
        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        row = 0
        self.add_bool_field(grid, row, "Facet-to-facet infrared exchange",
                            "radiation.enabled")
        row += 1
        self.radiation_note = _muted("")
        grid.addWidget(self.radiation_note, row, 0, 1, 2)
        row += 1
        if beaming:
            # Label from `gui_tooltips.LABELS`, which wins over any literal.
            self.add_bool_field(grid, row,
                                "Beam infrared from sub-facet roughness",
                                "radiation.beaming")
            row += 1
            self.beaming_note = _muted("")
            grid.addWidget(self.beaming_note, row, 0, 1, 2)
            row += 1
            self.widgets["radiation.beaming"].toggled.connect(
                self._on_beaming_toggled)
        self.add_file_field(grid, row, "View-factor cache:",
                            "radiation.view_factor_file",
                            "View factors (*.vf *.csr *.hmat);;All Files (*)")
        group.setLayout(grid)
        self.widgets["radiation.enabled"].toggled.connect(
            self._on_radiation_toggled)
        return group

    def _on_radiation_toggled(self, _checked):
        """The parent switch moved: re-gate the child and re-describe both."""
        self.refresh_radiation()

    def refresh_radiation(self):
        """Re-run the beaming gate and re-write both notes. Cheap and idempotent.

        Called on every load and on every toggle of either switch, so the two
        lines on screen are always computed from the live configuration rather
        than left over from the last one.
        """
        enabled = bool(self.config_manager.get_value("radiation.enabled",
                                                    False))
        note = getattr(self, "radiation_note", None)
        if note is not None:
            note.setText(
                "Faces exchange infrared and reflected sunlight with each "
                "other. The launch check reports what this will cost, and "
                "says so when the surface is a smooth sphere on which it "
                "computes nothing."
                if enabled else
                "Off: every face radiates only to space. A shadowed crater "
                "floor then has no source of warmth but the geothermal flux.")
        self._update_beaming_gating()

    def _update_beaming_gating(self):
        """Gate the beaming switch -- WITHOUT editing the file.

        The control is disabled and the reason is shown. What it deliberately
        does NOT do is turn an inherited `beaming: true` off: that would be an
        edit to the user's own configuration performed on OPEN, which is the
        class the correctness stage spent fourteen items removing and which the
        Radiation page already refuses to do for the same reason. An opened
        configuration whose beaming the rule would refuse gets a loud line
        saying the rule disagrees with it, and keeps its value.

        The rule is `facet_scale.beaming_gate`, which is the PARENT SWITCH and
        then the facet scale. It used to be `beaming_policy` alone, so beaming
        stayed offerable on a run with radiation off -- where it changes no
        temperature and can still abort the run at startup.
        """
        from gui_core.facet_scale import beaming_gate
        widget = self.widgets.get("radiation.beaming")
        note = getattr(self, "beaming_note", None)
        if widget is None or note is None:
            return
        try:
            policy = beaming_gate(self.config_manager.get_config(),
                                  self._project_root())
        except Exception as exc:                             # noqa: BLE001
            widget.setEnabled(False)
            note.setText(
                f"The facet-scale check could not run ({exc}), so this control "
                f"is disabled. Refusing is the safe direction. Any value "
                f"already in the configuration is left as it is.")
            return
        widget.setEnabled(bool(policy.allowed))
        if not policy.allowed and widget.isChecked():
            note.setText(
                "THIS CONFIGURATION HAS SUB-FACET ROUGHNESS ON, AND THE RULE "
                "DISAGREES WITH IT. " + policy.reason
                + " The value has NOT been changed -- it is your file's. Turn "
                  "it off yourself if you agree, or run it as it is.")
            logger.warning(
                "[card] this configuration enables radiation.beaming and the "
                "beaming gate would refuse it: %s. The setting is left alone; "
                "opening a file is not an edit to it.", policy.reason)
        elif policy.needs_acknowledgment:
            # A THIRD state, and it is not "fine". The rule could not MEASURE
            # this mesh, which is a different statement from "this mesh is too
            # fine" -- and reporting the first as the second would lock beaming
            # out of every format the reader does not parse but the engine
            # loads. The card says which of the two it is, and leaves the
            # judgment where it belongs.
            note.setText(
                "THE FACET SCALE COULD NOT BE MEASURED, so nothing here has "
                "checked this setting for you. " + policy.reason)
        else:
            note.setText(policy.reason)

    def _on_beaming_toggled(self, checked):
        """A USER toggle also sets the slope; a load does not reach here.

        `beaming: true` with `rms_slope_deg: 0` is a startup FATAL, so ticking
        the box without the slope would hand the user a configuration the
        engine refuses.
        """
        if self._loading:
            return
        from gui_core.presets import LUNAR_RMS_SLOPE_DEG
        self.config_manager.set_value(
            "radiation.rms_slope_deg",
            LUNAR_RMS_SLOPE_DEG if checked else 0.0)
        self.value_changed.emit()

    # -- load ---------------------------------------------------------------

    def load_from_config(self):
        if not self._active:
            return
        # `_loading` is not decoration. `set_bool` calls setChecked WITHOUT
        # blocking signals, so merely DISPLAYING a configuration fires every
        # `toggled` handler -- and one of those writes a derived key
        # (radiation.rms_slope_deg follows radiation.beaming). Loading a file
        # with `beaming: true, rms_slope_deg: 25.0` therefore rewrote the slope
        # to 32.5. Opening a file is not an edit to it.
        # Before the load, not only on `activate`. A -1 window can arrive
        # after the card is already open -- the All settings view still offers
        # the sentinel, and `BodyFirstPanel.refresh_active_card` re-loads the
        # card without re-activating it -- and a window control whose
        # configuration says -1 would otherwise display its own floor.
        if self._run_type == "integrated" \
                and self.WINDOW_KEY in self._duration_state():
            self._resolve_integration_window()

        self._loading = True
        try:
            self.load_card()
        finally:
            self._loading = False
        self._revalidate_duration_units()
        self.refresh_window_units()

    #: Combo controls a card owns, and the values each accepts. Without these
    #: the generic mirror below silently skipped every QComboBox -- a card
    #: opened on a configuration with `albedo_source: none` went on showing
    #: "builtin", which is a control displaying a setting the run does not
    #: have. Handled by KEY rather than by widget type because the engine
    #: normalizes some of these and compares others byte-exactly.
    COMBO_VALUES = {
        "illumination.albedo_source": {"builtin": "builtin",
                                       "custom": "custom", "none": "none",
                                       "": "builtin"},
        # The ENGINE's own spellings (ptm_thermal_sim.cpp:14707/:14720). The
        # extra entries are what a hand-written config plausibly says; each
        # maps to the spelling the engine actually branches on rather than
        # being displayed as itself and then ignored at run time.
        "generic_body.shape.type": {"spherical": "spherical",
                                    "sphere": "spherical",
                                    "ellipsoidal": "ellipsoidal",
                                    "ellipsoid": "ellipsoidal",
                                    "custommesh": "custom_mesh",
                                    "custom_mesh": "custom_mesh",
                                    "mesh": "custom_mesh",
                                    "": "spherical"},
    }

    #: Widgets whose loading a subclass handles itself.
    def custom_loaded_keys(self):
        return set()

    @staticmethod
    def _clear_widget(widget):
        """Put a control back to 'nothing set', without writing the config.

        Signals are blocked throughout: these handlers write derived keys, and
        showing a configuration must not edit it.
        """
        blocked = widget.signalsBlocked()
        widget.blockSignals(True)
        try:
            if isinstance(widget, QCheckBox):
                widget.setChecked(False)
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                # The smallest value the control allows, which for every
                # generic-body surface field is 0.0 -- and 0.0 is already how
                # this card reads "not set" (`_number(key, 0.0)`, and
                # `tiny_surface_refusals` skips zero deliberately).
                widget.setValue(widget.minimum())
            elif isinstance(widget, QComboBox):
                if widget.count():
                    widget.setCurrentIndex(0)
            elif isinstance(widget, QLineEdit):
                widget.setText("")
        finally:
            widget.blockSignals(blocked)

    def load_card(self):
        """Mirror the configuration into this card's controls."""
        skip = set(self.custom_loaded_keys())
        # The override rows carry a SENTINEL (-1) that must not reach their
        # spin boxes: the box shows the effective value and the tick carries
        # "not overriding". `load_overrides` does both, so the generic mirror
        # must keep its hands off these five keys.
        skip.update(self._override_state())
        for key, widget in self.widgets.items():
            if key in skip:
                continue
            value = self.config_manager.get_value(key, None)
            if value is None:
                # ABSENT from the configuration, which is not the same as
                # "leave the control alone". Leaving it alone is a control
                # showing a setting the run does not have -- the same defect
                # COMBO_VALUES above exists to close for combo boxes.
                #
                # It became reachable when the body cards started clearing
                # another card's section: `generic_body.surface.*` has no entry
                # in DEFAULT_CONFIG (the keys are created on demand), so after
                # a clear the spin boxes went on showing the previous body's
                # albedo and conductivity while the configuration carried
                # neither. Every other key a card shows is in DEFAULT_CONFIG and
                # so is never None here.
                self._clear_widget(widget)
                continue
            if isinstance(widget, QCheckBox):
                self.set_bool(key, value)
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                self.set_numeric(key, value)
            elif isinstance(widget, QComboBox):
                aliases = self.COMBO_VALUES.get(key)
                if aliases is not None:
                    self.set_enum(key, value, aliases)
            elif isinstance(widget, QLineEdit):
                self.set_text(key, value)
        if self._override_state():
            self.load_overrides()


# ===========================================================================
# Moon
# ===========================================================================

class MoonCard(BodyCard):
    """The best-supported body: albedo map, property map, topography catalog,
    the beaming calibration and every tutorial."""

    def init_ui(self):
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        layout.addWidget(self.time_group())

        # -- global geometry --------------------------------------------------
        self.global_group = QGroupBox("Global mesh")
        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        self.add_int_field(grid, 0, "Subdivision level:", "mesh.resolution",
                           0, 9)
        self.face_count = _muted("")
        grid.addWidget(self.face_count, 1, 0, 1, 2)
        self.add_bool_field(grid, 2, "Apply topographic relief",
                            "mesh.use_topographic")
        self.topo_note = _muted("")
        grid.addWidget(self.topo_note, 3, 0, 1, 2)
        self.global_group.setLayout(grid)
        layout.addWidget(self.global_group)
        self.widgets["mesh.resolution"].valueChanged.connect(
            lambda _v: self._refresh_surface())
        self.widgets["mesh.use_topographic"].toggled.connect(
            lambda _v: self._refresh_surface())

        # -- a site on the Moon ----------------------------------------------
        self.local_group = QGroupBox("Local mesh")
        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        self.add_file_field(grid, 0, "Terrain mesh:", "mesh.import_file",
                            "Meshes (*.ply *.obj *.vtk);;All Files (*)")
        # The label argument is DEAD for this key: `_resolve_text` prefers
        # `gui_tooltips.LABELS`, which spells it "Shadow-only surrounding
        # mesh:". Kept identical to the rendered text so the call site cannot
        # be read as a second, disagreeing answer (BUGS.md
        # GUI-CARD-LABEL-LITERAL-DEAD).
        self.add_file_field(grid, 1, "Shadow-only surrounding mesh:",
                            "mesh.occluder_file",
                            "Meshes (*.ply *.obj *.vtk);;All Files (*)")
        self.add_double_field(grid, 2, "Latitude (deg):",
                              "mesh.latitude_override", -999.0, 90.0,
                              decimals=4)
        self.add_double_field(grid, 3, "Longitude (deg):",
                              "mesh.longitude_override", -999.0, 360.0,
                              decimals=4)
        self.local_group.setLayout(grid)
        layout.addWidget(self.local_group)
        self.widgets["mesh.import_file"].textChanged.connect(
            lambda _t: self._refresh_surface())

        # -- surface ---------------------------------------------------------
        surface = QGroupBox("Surface")
        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        self.add_combo_field(
            grid, 0, "Albedo source:", "illumination.albedo_source",
            ["builtin", "custom", "none"])
        self.albedo_note = _muted("")
        grid.addWidget(self.albedo_note, 1, 0, 1, 2)
        self.add_bool_field(grid, 2,
                            "Directional-hemispheric albedo (Hapke)",
                            "thermophysical.incidence_angle_albedo")
        self.add_bool_field(grid, 3, "Bundled per-face property map",
                            "thermal.use_default_property_map")
        self.properties_note = _muted("")
        grid.addWidget(self.properties_note, 4, 0, 1, 2)
        surface.setLayout(grid)
        layout.addWidget(surface)

        # -- radiation exchange (EVERY mode; see BodyCard.radiation_group) -----
        layout.addWidget(self.radiation_group(beaming=True))

        layout.addWidget(self.builtin_surface_group("Moon"))
        layout.addWidget(self.thermal_group())
        layout.addWidget(self.output_group())
        self._make_report(layout)
        layout.addStretch()
        self.create_scroll_area(content)

    def apply_mode(self):
        local = self._mode == "local"
        self.global_group.setVisible(not local)
        self.local_group.setVisible(local)

    def load_card(self):
        super().load_card()
        self._refresh_surface()

    # -- the auto-albedo and beaming rules, narrated ------------------------

    def _refresh_surface(self):
        from gui_core import datasets
        config = self.config_manager.get_config()
        imported = str(
            (config.get("mesh") or {}).get("import_file", "") or "").strip()
        resolution = int(self._number("mesh.resolution", 5))

        self.face_count.setText("" if imported else self.icosphere_note())

        status = datasets.albedo_status(
            "Moon", resolution, self._project_root(), import_file=imported)
        self.albedo_note.setText(status.detail or str(status.resolved or ""))

        topo = datasets.topo_status("Moon", resolution, self._project_root())
        if imported:
            self.topo_note.setText("")
        elif topo.state == datasets.STATE_OK:
            self.topo_note.setText(f"{topo.resolved}")
        else:
            self.topo_note.setText(
                f"{topo.detail} — the engine stops rather than substituting a "
                f"smooth sphere, so leave this off until the grid is on disk.")

        self.properties_note.setText(
            "The bundled map applies to the procedural icosphere it was built "
            "for; an imported mesh uses the single H value from Properties."
            if imported else
            f"Bundled map for ico{resolution}, matched by face count.")
        self.refresh_radiation()

    def extra_exposed_keys(self):
        """Keys a control on this card WRITES without being bound to it.

        Only those. `thermophysical.albedo_model` was in this set in the first
        draft and nothing on the card writes it -- which would have made the
        card claim to show a live setting it does not show, hiding it from the
        out-of-card listing. That is exactly the defect the listing exists to
        prevent, arrived at from the other direction.
        """
        return {
            # written by the sub-facet roughness switch
            "radiation.rms_slope_deg",
            # written by the albedo-source translation ConfigManager performs
            # at save time (_config_for_yaml)
            "illumination.use_albedo_map", "illumination.albedo_map",
            "illumination.albedo_maps_directory",
        }


# ===========================================================================
# Europa
# ===========================================================================

class EuropaCard(BodyCard):
    """Fifty-five shipped configurations, and the tightest option set in the
    corpus: the forty-five DEM runs differ in exactly two scientific keys."""

    def init_ui(self):
        from gui_core.body_cards import europa_dem_registry

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        self._registry = europa_dem_registry(self._project_root())

        # This card used to keep `thermal.integration_days` visible in EVERY
        # run type, because it is the one card that offered the key before the
        # control became the integrated run type's duration. It no longer does.
        # In a time series the key's own default is -1, "integrate the output
        # window", and the output window is the Duration control two rows up:
        # the only honest thing to put in the box was a sentinel, and a
        # sentinel is what Paul asked to be rid of. A run that genuinely needs
        # a statistics window SHORTER than its output window sets it in All
        # settings, where the sentinel is documented and the output window is
        # a separate control that can disagree with it on purpose.
        layout.addWidget(self.time_group())

        # -- global Europa ----------------------------------------------------
        # `thermal.integration_days` used to sit here, in a mesh-geometry
        # section. It is a solver setting, so it now lives in the Thermal
        # section this card builds below.
        self.global_group = QGroupBox("Global mesh")
        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        self.add_int_field(grid, 0, "Subdivision level:", "mesh.resolution",
                           0, 9)
        self.add_combo_field(grid, 1, "Albedo source:",
                             "illumination.albedo_source",
                             ["builtin", "custom", "none"])
        self.albedo_note = _muted("")
        grid.addWidget(self.albedo_note, 2, 0, 1, 2)
        self.global_group.setLayout(grid)
        layout.addWidget(self.global_group)
        self.widgets["mesh.resolution"].valueChanged.connect(
            lambda _v: self._refresh())

        # -- a Europa DEM site -----------------------------------------------
        self.local_group = QGroupBox("Local mesh (DEM)")
        grid = QGridLayout()
        grid.setColumnStretch(1, 1)

        terrain_label = QLabel("Terrain:")
        terrain_label.setToolTip(get_tooltip("ui.europa_terrain"))
        self._apply_tooltip(terrain_label, "ui.europa_terrain")
        grid.addWidget(terrain_label, 0, 0)
        self.terrain_combo = QComboBox()
        self.terrain_combo.setToolTip(get_tooltip("ui.europa_terrain"))
        for key in sorted(self._registry):
            self.terrain_combo.addItem(self._registry[key].display(), key)
        if not self._registry:
            self.terrain_combo.addItem("(no Europa DEM configurations found)",
                                       "")
            self.terrain_combo.setEnabled(False)
        self.terrain_combo.currentIndexChanged.connect(self._on_terrain)
        grid.addWidget(self.terrain_combo, 0, 1)
        self.terrain_note = _muted("")
        grid.addWidget(self.terrain_note, 1, 0, 1, 2)

        latitude_label = QLabel("Latitude (deg):")
        latitude_label.setToolTip(get_tooltip("mesh.latitude_override"))
        self._apply_tooltip(latitude_label, "mesh.latitude_override")
        grid.addWidget(latitude_label, 2, 0)
        self.latitude_combo = QComboBox()
        self.latitude_combo.setToolTip(get_tooltip("mesh.latitude_override"))
        self.latitude_combo.currentIndexChanged.connect(self._on_latitude)
        grid.addWidget(self.latitude_combo, 2, 1)
        self.widgets["mesh.latitude_override"] = self.latitude_combo

        # `radiation.enabled` used to sit HERE, inside the local-mesh section,
        # so it existed only in the DEM mode of this one card. It is now the
        # shared `BodyCard.radiation_group` below, on every mode and every run
        # type. What stays in this section is the view-factor NOTE, because it
        # describes the cache the TERRAIN picker fills in and belongs with the
        # terrain that decides it.
        self.vf_note = _muted("")
        grid.addWidget(self.vf_note, 3, 0, 1, 2)
        self.local_group.setLayout(grid)
        layout.addWidget(self.local_group)

        # -- radiation exchange (EVERY mode; see BodyCard.radiation_group) -----
        # No beaming switch: `facet_scale.BEAMING_BODIES` is lunar-only and the
        # Smith slope distribution has no measured value for ice.
        layout.addWidget(self.radiation_group())

        # -- the grid (run type "sweep") --------------------------------------
        self.sweep_group = QGroupBox("Parameter grid")
        sweep_layout = QVBoxLayout()
        self.sweep_summary = _muted("", color="#64b5f6", size=11)
        sweep_layout.addWidget(self.sweep_summary)
        row = QHBoxLayout()
        self.sweep_button = QPushButton("Write grid configurations…")
        self.sweep_button.setToolTip(get_tooltip("ui.europa_sweep"))
        self.sweep_button.clicked.connect(self._on_prepare_sweep)
        row.addWidget(self.sweep_button)
        row.addStretch()
        sweep_layout.addLayout(row)
        self.sweep_group.setLayout(sweep_layout)
        layout.addWidget(self.sweep_group)

        # No standing caveat on this card. It used to carry one about archived
        # Europa results predating a 2026-08 solver fix; that is a statement
        # about other people's older output rather than about the run this card
        # is setting up, and the number it quoted holds only for the
        # Jupiter-facing hemisphere. A card describes the run in front of the
        # user.
        layout.addWidget(self.builtin_surface_group("Europa"))
        layout.addWidget(self.thermal_group())
        layout.addWidget(self.output_group())
        self._make_report(layout)
        layout.addStretch()
        self.create_scroll_area(content)

    def apply_mode(self):
        local = self._mode == "local"
        self.global_group.setVisible(not local)
        self.local_group.setVisible(local)
        self.sweep_group.setVisible(local and self._run_type == "sweep")
        if local:
            self._sync_latitudes()
        self._refresh()

    # -- terrain registry ---------------------------------------------------

    def _current_dem(self):
        key = self.terrain_combo.currentData()
        return self._registry.get(key) if key else None

    def _sync_latitudes(self):
        entry = self._current_dem()
        self.latitude_combo.blockSignals(True)
        self.latitude_combo.clear()
        if entry is not None:
            for latitude in entry.latitudes:
                self.latitude_combo.addItem(f"{latitude:g}", latitude)
        self.latitude_combo.blockSignals(False)

    def _on_terrain(self, _index):
        entry = self._current_dem()
        if entry is None:
            return
        self.config_manager.set_value("mesh.import_file", entry.import_file)
        self.config_manager.set_value("radiation.view_factor_file",
                                      entry.view_factor_file)
        self._sync_latitudes()
        self._on_latitude(0)
        self._refresh()
        self.value_changed.emit()

    def _on_latitude(self, _index):
        value = self.latitude_combo.currentData()
        if value is None:
            return
        self.config_manager.set_value("mesh.latitude_override", float(value))
        self._refresh()
        self.value_changed.emit()

    def extra_exposed_keys(self):
        return {"mesh.import_file", "radiation.view_factor_file",
                "illumination.use_albedo_map", "illumination.albedo_map",
                "illumination.albedo_maps_directory"}

    # -- display ------------------------------------------------------------

    def custom_loaded_keys(self):
        # The latitude control is a combo over the shipped values, mirrored by
        # _mirror_terrain, which can also SHOW a latitude the grid does not
        # carry rather than snapping to the nearest one.
        return {"mesh.latitude_override"}

    def load_card(self):
        super().load_card()
        self._mirror_terrain()
        self._ensure_terrain_emitted()
        self._refresh()

    def _ensure_terrain_emitted(self):
        """Put the terrain this card is SHOWING into the configuration.

        WHY. The terrain combo is filled in ``init_ui`` and
        ``currentIndexChanged`` is connected AFTER the fill loop, so populating
        it never runs ``_on_terrain``. Opening this card therefore displayed
        "Agenor -- 750 m posting" and the note
        "data/external/sutton_DEMs/Agenor_DEM_750m_SOCET.ply -- on disk" over a
        configuration whose ``mesh.import_file`` was the empty string. The
        engine takes the icosphere branch when that key is empty
        (``ptm_thermal_sim.cpp:15087``), so pressing Run gave a res-7 GLOBAL
        SPHERE -- measured 2026-08-28: 327,680 faces, median radius 1560.8 km,
        ``mesh_source=icosphere, resolution 7`` in the run's own provenance,
        from a card whose whole subject is a 750 m regional DEM. The latitude
        went the same way: the combo showed 0 while ``mesh.latitude_override``
        stayed at the -999.0 "not placed" sentinel.

        The rule is narrow on purpose. It writes ONLY when
        ``mesh.import_file`` is empty -- a configuration the user opened, or a
        terrain they picked, is never touched -- and it writes the latitude
        only in that same case, because a file that deliberately leaves the DEM
        at its own coordinates (-999.0) must keep them.

        A terrain whose file is not on disk is still written. Emitting the path
        is what turns a silent substitution into the pre-flight's refusal,
        which names the missing file and where to get it; dropping it is the
        defect this method exists to remove.
        """
        if self._mode != "local":
            return
        # ONLY while the card is being opened for a NEW run. The panel calls
        # `load_from_config` on whichever card is ACTIVE, and `deactivate` has
        # not run at that point -- so a Europa terrain card left open while
        # another configuration is displayed is this card writing into somebody
        # else's file. Measured while building this: opening
        # `examples/quickstart_moon_diurnal.yaml` straight after a Europa
        # terrain card put an Agenor DEM into the Moon configuration, which
        # then routed to Moon/LOCAL (caught by
        # `test_gui_body_first.test_loading_a_config_lands_on_its_own_card`).
        # The same would happen to a Europa GLOBAL file, which the body check
        # below cannot catch. Opening a file is not an edit to it.
        if not self._seeding_defaults:
            return
        # ...and only into a EUROPA configuration, for the same reason from the
        # other side: the route defaults have already written simulation.body
        # by the time `activate` runs, so this is the card's own body.
        body = str(self.config_manager.get_value("simulation.body", "") or "")
        if body.strip().lower() != "europa":
            return
        if str(self.config_manager.get_value("mesh.import_file", "") or ""
               ).strip():
            return
        entry = self._current_dem()
        if entry is None or not entry.import_file:
            return
        self.config_manager.set_value("mesh.import_file", entry.import_file)
        self.config_manager.set_value("radiation.view_factor_file",
                                      entry.view_factor_file)
        latitude = self.latitude_combo.currentData()
        if latitude is not None:
            self.config_manager.set_value("mesh.latitude_override",
                                          float(latitude))

    def _mirror_terrain(self):
        current = str(
            self.config_manager.get_value("mesh.import_file", "") or "")
        if not current:
            return
        for index in range(self.terrain_combo.count()):
            key = self.terrain_combo.itemData(index)
            entry = self._registry.get(key) if key else None
            if entry is not None and entry.import_file == current:
                self.terrain_combo.blockSignals(True)
                self.terrain_combo.setCurrentIndex(index)
                self.terrain_combo.blockSignals(False)
                self._sync_latitudes()
                break
        latitude = self._number("mesh.latitude_override", -999.0)
        index = self.latitude_combo.findData(latitude)
        if index >= 0:
            self.latitude_combo.blockSignals(True)
            self.latitude_combo.setCurrentIndex(index)
            self.latitude_combo.blockSignals(False)
        elif latitude > -999.0:
            # A latitude the shipped grid does not use. Shown as its own entry
            # rather than snapped to the nearest one: snapping would move the
            # run without saying so.
            self.latitude_combo.blockSignals(True)
            self.latitude_combo.addItem(f"{latitude:g} (not in the grid)",
                                        latitude)
            self.latitude_combo.setCurrentIndex(self.latitude_combo.count() - 1)
            self.latitude_combo.blockSignals(False)

    def _refresh(self):
        from gui_core import datasets
        from gui_core.body_cards import europa_sweep_points

        root = self._project_root()
        resolution = int(self._number("mesh.resolution", 6))
        imported = str(
            self.config_manager.get_value("mesh.import_file", "") or "")
        status = datasets.albedo_status("Europa", resolution, root,
                                        import_file=imported)
        self.albedo_note.setText(status.detail or str(status.resolved or ""))

        entry = self._current_dem()
        if entry is not None:
            from gui_core.preflight import (is_dataless,
                                            missing_external_data_hint,
                                            resolve_data_path, resolve_path)
            # The SAME resolver the pre-flight and the engine use for
            # mesh.import_file, so the card cannot say "on disk" about a file
            # the run then fails to find. mesh.import_file follows the
            # PTM_DATA_ROOT contract (`preflight._ENGINE_DATA_ROOT_FIELDS`);
            # radiation.view_factor_file deliberately does NOT, so it gets
            # `resolve_path` below. Using one resolver for both would let the
            # card say "cached, will be loaded" about a matrix the run rebuilds
            # whenever PTM_DATA_ROOT is set.
            path = resolve_data_path(entry.import_file, root)
            if not path.exists():
                # Named as a refusal, with the source, rather than as a note.
                # This state used to read "NOT on disk" beside a Run button
                # that launched anyway -- on an icosphere.
                hint = missing_external_data_hint(path)
                self.terrain_note.setText(
                    f"{entry.import_file} — NOT on disk at {path}. This run "
                    f"will be refused at launch. " + hint)
            elif is_dataless(path):
                self.terrain_note.setText(
                    f"{entry.import_file} — present at {path} but DATALESS (a "
                    f"cloud placeholder whose contents are not on this disk). "
                    f"Reading it would hang the run; hydrate it in Finder "
                    f"(right-click > \"Always Keep on This Device\") first. "
                    f"This run will be refused at launch.")
            else:
                self.terrain_note.setText(f"{entry.import_file} — on disk")
            # The view-factor cache gets the same treatment as the DEM: never
            # a bare "not found". A matrix somebody has already built and
            # verified is not interchangeable with one this machine rebuilds,
            # so when it is missing the card states BOTH options and, when the
            # cache is really sitting somewhere on this disk, says where.
            vf = resolve_path(entry.view_factor_file, root)
            if vf.exists() and not is_dataless(vf):
                self.vf_note.setText(
                    f"View factors: {entry.view_factor_file} — cached, will "
                    f"be loaded")
            elif vf.exists():
                self.vf_note.setText(
                    f"View factors: {entry.view_factor_file} — present at "
                    f"{vf} but DATALESS (a cloud placeholder). Hydrate it "
                    f"first, or the run will hang reading it.")
            else:
                vf_hint = missing_external_data_hint(vf)
                self.vf_note.setText(
                    f"View factors: {entry.view_factor_file} — NOT cached at "
                    f"{vf}. Either point Radiation > view_factor_file at a "
                    f"matrix already built for this mesh"
                    + (f" ({vf_hint})" if vf_hint else "")
                    + ", or run as-is and PTM3D will build the matrix and "
                      "save it there. The launch check shows the estimated "
                      "cost before anything starts.")
        points = europa_sweep_points(root)
        terrains = len({p[0] for p in points})
        latitudes = len({p[1] for p in points})
        minutes = len(points) * _MINUTES_PER_RUN_GUESS
        self.sweep_summary.setText(
            f"{terrains} terrains x {latitudes} latitudes = {len(points)} runs."
            f" Rough estimate {minutes / 60.0:.0f} h in series on this machine;"
            f" the real figure depends on the mesh and the view factors.")
        self.refresh_radiation()

    # -- preparing the grid --------------------------------------------------

    def sweep_plan(self, outdir):
        """The `ptm.Sweep` this card would run, built from the CURRENT config.

        The base is what the card shows, written to a file first, because
        ``ptm.Sweep`` takes a path and records its sha256 -- a sweep whose base
        is not a file on disk could not be reproduced from its own record.
        """
        from ptm import Sweep
        from gui_core.body_cards import europa_dem_registry

        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        base_path = outdir / "sweep_base.yaml"
        self.config_manager.save_run_yaml(str(base_path), self._project_root())

        registry = europa_dem_registry(self._project_root())
        dems = sorted(registry)
        latitudes = sorted({lat for entry in registry.values()
                            for lat in entry.latitudes})
        return Sweep(
            base_path,
            grid={"mesh.latitude_override": latitudes, "dem": dems},
            label_axes=("dem",),
            derived={
                "mesh.import_file":
                    lambda p: registry[p["dem"]].import_file,
                "radiation.view_factor_file":
                    lambda p: registry[p["dem"]].view_factor_file,
            },
            name_from=lambda p: (
                f"lat{int(p['mesh.latitude_override']):02d}_{p['dem']}"),
            outdir=outdir, quiet=True)

    def _on_prepare_sweep(self):
        from gui_core.body_cards import europa_sweep_points
        root = self._project_root()
        points = europa_sweep_points(root)
        if not points:
            self.show_report(
                "No Europa DEM configurations were found in configs/, so "
                "there is no grid to derive. The terrain list is read from "
                "those files rather than hard-coded, which is why an empty "
                "list means an empty grid and not a wrong one.")
            return
        directory = QFileDialog.getExistingDirectory(
            self, "Output directory for the grid", "",
            QFileDialog.Option.ShowDirsOnly)
        if not directory:
            return
        minutes = len(points) * _MINUTES_PER_RUN_GUESS
        answer = QMessageBox.question(
            self, "Write grid configurations?",
            f"This writes {len(points)} configurations under\n{directory}\n\n"
            f"Running them all takes roughly {minutes / 60.0:.0f} hours in "
            f"series on this machine — a rough estimate, not a measurement.\n\n"
            f"They are written, not launched. A multi-hour grid belongs in a "
            f"detached process that survives closing this window; the exact "
            f"command is printed when the files are ready.",
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel)
        if answer != QMessageBox.StandardButton.Ok:
            self.show_report("Grid canceled. Nothing was written.")
            return
        try:
            written = self.write_sweep(directory)
        except Exception as exc:                             # noqa: BLE001
            self.show_report(f"The grid could not be written: {exc}")
            return
        self.show_report(
            f"Wrote {len(written)} configurations under {directory}, one per\n"
            f"grid point, alongside the base configuration they came from.\n"
            f"Nothing has been launched. Run them detached, so the grid\n"
            f"survives closing this window -- for example:\n"
            f"    nohup python3 scripts/run_chunked_simulation.py \\\n"
            f"        --config {directory}/<point>/config.yaml &\n"
            f"for each point, or drive the whole set with ptm.Sweep, whose\n"
            f"base configuration is {directory}/sweep_base.yaml.")

    def write_sweep(self, outdir):
        """Write one `config.yaml` per grid point. Returns the paths."""
        sweep = self.sweep_plan(outdir)
        written = []
        for point in sweep:
            point.dir.mkdir(parents=True, exist_ok=True)
            path = point.config_path
            # The sweep's OWN serialization, not a second one: `config_text()`
            # is what `Sweep.run()` and the point's recorded sha256 use, so
            # writing it here means the files on disk are the files the sweep
            # believes it has.
            path.write_text(point.config_text())
            written.append(path)
        return written


# ===========================================================================
# Mercury
# ===========================================================================

class MercuryCard(BodyCard):
    """Runnable, and thin in DATA rather than in physics.

    The two defects that used to make a Mercury button dishonest are fixed:
    the engine now forces at the 175.938 d SOLAR day rather than the 58.646 d
    sidereal rotation, and the 500 K ceiling written for icy bodies is gone
    from every path. What is still missing is a dataset -- no albedo map, no
    topography, no worked example -- and `data_support_banner` is now the ONLY
    place that is said. The landing screen carried a two-word "(limited data)"
    badge until 2026-08-27; the button is the body's bare name now, so a
    Mercury user meets this statement here or nowhere.
    """

    def init_ui(self):
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        # A BANNER, not a caption. It is the single surviving disclosure that
        # this body ships no data, so it is bordered and sits above every
        # control it constrains -- `test_gui_body_first.py` pins both the text
        # and the fact that it is on this card.
        # `setText`, NOT a QLabel(...) constructor argument: the voice gate's
        # source-side harvest reads string constants handed to `setText` and
        # friends, and moving this copy into the constructor would drop the one
        # remaining Mercury disclosure out of that corpus silently.
        self.data_support_banner = QLabel("")
        self.data_support_banner.setText(
            "Limited data: no albedo map, no topography grid and no worked "
            "example ship for Mercury. Every run here is a uniform-albedo "
            "icosphere at the engine's built-in Mercury properties. The "
            "solver, the illumination and the 175.938-day solar day are the "
            "same machinery the Moon uses.")
        self.data_support_banner.setWordWrap(True)
        self.data_support_banner.setTextFormat(Qt.TextFormat.PlainText)
        self.data_support_banner.setStyleSheet(
            "color: #FFCC80; font-size: 11px; padding: 6px; "
            "border: 1px solid #FFCC80; border-radius: 4px;")
        layout.addWidget(self.data_support_banner)

        layout.addWidget(self.time_group())

        sphere = QGroupBox("Global mesh")
        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        self.add_int_field(grid, 0, "Subdivision level:", "mesh.resolution",
                           0, 9)
        self.face_count = _muted("")
        grid.addWidget(self.face_count, 1, 0, 1, 2)
        sphere.setLayout(grid)
        layout.addWidget(sphere)
        self.widgets["mesh.resolution"].valueChanged.connect(
            lambda _v: self._refresh())

        surface = QGroupBox("Surface")
        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        self.add_bool_field(grid, 0,
                            "Directional-hemispheric albedo (Hapke)",
                            "thermophysical.incidence_angle_albedo")
        self.properties_note = _muted("")
        grid.addWidget(self.properties_note, 1, 0, 1, 2)
        surface.setLayout(grid)
        layout.addWidget(surface)

        # -- radiation exchange -----------------------------------------------
        # This card had NO radiation control at all, on any run type, while
        # `radiation.enabled` is a key the engine reads on every body. The
        # Mercury preset writes it (`False`), so a Mercury configuration
        # carried the switch and no screen showed it.
        layout.addWidget(self.radiation_group())

        layout.addWidget(self.builtin_surface_group("Mercury"))
        layout.addWidget(self.thermal_group())
        layout.addWidget(self.output_group())
        self._make_report(layout)
        layout.addStretch()
        self.create_scroll_area(content)

    def load_card(self):
        super().load_card()
        self._refresh()

    def _refresh(self):
        from ptm.body_table import BODY_CAPABILITIES
        self.refresh_radiation()
        row = BODY_CAPABILITIES["Mercury"]
        self.face_count.setText(self.icosphere_note(fallback=4))
        self.properties_note.setText(
            f"Mercury standard — albedo {row['albedo']:g}, emissivity "
            f"{row['emissivity']:g}, ks {row['ks']:g} W/m/K, H {row['H']:g} m, "
            f"solar day {row['solar_day_days']:g} days. These come from the "
            f"engine's own table; change them on the Properties page, where "
            f"only the five keys that have an override path are editable.")


# ===========================================================================
# Generic bodies: asteroid, and anything else
# ===========================================================================

class _LookupWorker(QThread):
    """One HORIZONS request, off the GUI thread.

    Emits exactly once, always. `lookup_body` does not raise -- but if it ever
    did, the exception would cross a thread boundary into nothing and the card
    would sit on "Asking HORIZONS..." forever, so it is caught here too and
    turned into a result the card can show.

    ``route`` is carried through untouched so the card can check, when the
    answer arrives up to ``ONLINE_TIMEOUT_S`` later, that it is still the card
    the answer was asked for.
    """

    done = pyqtSignal(object, object)

    def __init__(self, query, project_root, route):
        super().__init__()
        self._query = query
        self._project_root = project_root
        self.route = route

    def run(self):
        from gui_core.body_lookup import BodyLookupResult, lookup_body
        try:
            result = lookup_body(self._query, self._project_root,
                                 allow_online=True)
        except Exception as exc:                             # noqa: BLE001
            result = BodyLookupResult(
                message=(f"The lookup failed unexpectedly "
                         f"({type(exc).__name__}: {exc}). Nothing has been "
                         f"changed."))
        self.done.emit(result, self.route)


class _FetchWorker(QThread):
    """One network fetch, off the GUI thread, emitting exactly once.

    The same contract as `_LookupWorker` and for the same reasons: a blocking
    request on the GUI thread is a frozen window, and an exception crossing a
    thread boundary into nothing leaves the card on "Fetching..." forever, so
    it is caught here and turned into a result the card can show.

    `route` is carried through untouched so the card can check, when the answer
    arrives up to a minute and a half later, that it is still the card the
    answer was asked for. Without that check a kernel fetched for an asteroid
    would be written into whatever configuration is on screen when it lands.
    """

    done = pyqtSignal(object, object)
    #: One line of a long transfer's own output, forwarded to the GUI thread.
    #: A SIGNAL rather than a callback that touches a widget: a QLabel written
    #: from a worker thread is undefined behavior in Qt, and an 80 MB download
    #: with no sign of life is indistinguishable from a hung window.
    progress = pyqtSignal(str)

    def __init__(self, call, route, kind):
        super().__init__()
        #: `call` takes the progress emitter when it wants one. Passed as an
        #: argument rather than read off the worker so `body_fetch` stays free
        #: of Qt -- the whole fetch layer is measurable without a window.
        self._call = call
        self.route = route
        self.kind = kind

    def run(self):
        from gui_core.body_fetch import FetchResult
        try:
            result = self._call(self.progress.emit)
        except Exception as exc:                             # noqa: BLE001
            result = FetchResult(
                kind=self.kind,
                message=(f"The fetch failed unexpectedly "
                         f"({type(exc).__name__}: {exc}). Nothing has been "
                         f"changed."))
        self.done.emit(result, self.route)


class GenericBodyCard(BodyCard):
    """The `generic_body:` route -- the only way to model a body the engine has
    no built-in properties for.

    The orbit elements stay on the Body page rather than being duplicated here.
    Two controls bound to one configuration key in two places is a real hazard
    -- whichever loads last decides the stored value -- and the seven orbital
    elements are the largest block where that would apply. Everything the card
    DOES show is a decision this screen is meant to make: who the body is, how
    its position is computed, what shape it has, and what its surface is like.
    """

    ASTEROID = False

    #: The two button labels for a body no registry covers. They are CONSTANTS
    #: rather than literals inside `init_ui` because `_refresh_curated` swaps
    #: them for a curated body and has to be able to swap them back -- a card
    #: is reused across bodies, and a button still reading "Install ephemeris
    #: (JPL solution 118)" over 623 Chimaera would be the same cross-card
    #: residue `_repair_body_name` and `_provenance` each remove elsewhere.
    SPK_BUTTON_DEFAULT = "Fetch kernel from HORIZONS"
    SHAPE_BUTTON_DEFAULT = "DAMIT shape model…"

    def init_ui(self):
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        identity = QGroupBox("Body identity")
        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        self.add_string_field(grid, 0, "Name:", "generic_body.name")
        self.add_string_field(grid, 1, "NAIF id:", "generic_body.naif_id")

        # -- the lookup (item 3) ---------------------------------------------
        # A BUTTON, deliberately, not a textChanged handler. The name field
        # already carries one lesson about per-keystroke work: `beaming_policy`
        # ran from `mesh.import_file`'s textChanged and opened a 100 MB mesh on
        # the GUI thread while the user typed. A network request per keystroke
        # would be that, over the internet, and it would fire on every prefix
        # of the name the user is halfway through typing.
        self.lookup_button = QPushButton("Look up")
        self.lookup_button.setToolTip(get_tooltip("ui.body_lookup"))
        self.lookup_button.clicked.connect(self._on_lookup)
        grid.addWidget(self.lookup_button, 0, 2)
        #: The in-flight HORIZONS request, or None. Held so the QThread is not
        #: garbage-collected while it is still running.
        self._lookup_worker = None

        # Physical properties, from SBDB, on their own button. Separate from
        # Look up because it is a separate question with a separate answer:
        # Look up says WHICH body this is, this says what is known about it.
        self.sbdb_button = QPushButton("Fill in properties (SBDB)")
        self.sbdb_button.setToolTip(get_tooltip("ui.fetch_sbdb"))
        self.sbdb_button.clicked.connect(self._on_fetch_sbdb)
        grid.addWidget(self.sbdb_button, 1, 2)

        self.provenance_note = _muted("Parameters entered by hand.", size=10)
        grid.addWidget(self.provenance_note, 2, 0, 1, 3)
        self.lookup_note = _muted("", color="#64b5f6", size=10)
        grid.addWidget(self.lookup_note, 3, 0, 1, 3)

        identity.setLayout(grid)
        layout.addWidget(identity)
        # The name is this card's answer to "which body is this run about", so
        # it owns `simulation.body` as well. Without the mirror the key kept
        # whatever the LAST card wrote -- and every shipped configuration with
        # a `generic_body:` block sets `simulation.body` to that same body's
        # name (25 of 25), so the mirror is the corpus's own convention rather
        # than a new one. See `_mirror_body_name`.
        self.widgets["generic_body.name"].textChanged.connect(
            self._mirror_body_name)
        # The curated banner follows the name, so a user who types "Bennu"
        # sees what this body is served from BEFORE they press a button
        # labeled for a service that will refuse them. A registry lookup and
        # two stat calls; the pinned-kernel table itself is memoized on its
        # own mtime, so this is not a file exec per keystroke.
        self.widgets["generic_body.name"].textChanged.connect(
            lambda _t: self._refresh_curated())

        # -- ephemeris, with the coverage check the project never had ---------
        ephemeris = QGroupBox("Ephemeris")
        egrid = QGridLayout()
        egrid.setColumnStretch(1, 1)
        row = QWidget()
        rlayout = QHBoxLayout(row)
        rlayout.setContentsMargins(0, 0, 0, 0)
        self.kepler_radio = QRadioButton("Keplerian elements")
        self.kepler_radio.setToolTip(get_tooltip("ui.ephemeris_keplerian"))
        self.spice_radio = QRadioButton("SPICE kernel")
        self.spice_radio.setToolTip(get_tooltip("ui.ephemeris_spice"))
        rlayout.addWidget(self.kepler_radio)
        rlayout.addWidget(self.spice_radio)
        rlayout.addStretch()
        source_label = QLabel("Positions from:")
        source_label.setToolTip(get_tooltip("generic_body.ephemeris_source"))
        self._apply_tooltip(source_label, "generic_body.ephemeris_source")
        egrid.addWidget(source_label, 0, 0)
        egrid.addWidget(row, 0, 1)
        self.kepler_radio.toggled.connect(self._on_ephemeris_changed)
        # The kernel can be FETCHED as well as picked. Both stay: a fetch
        # covers the ordinary case and needs a network, a file picker covers a
        # kernel somebody already has and needs none.
        self.spk_fetch_button = QPushButton(self.SPK_BUTTON_DEFAULT)
        self.spk_fetch_button.setToolTip(get_tooltip("ui.fetch_spk"))
        # A zero-argument lambda, NOT `connect(self._on_fetch_spk)`. Qt's
        # `clicked(bool checked)` is delivered to any slot that can take an
        # argument, and `_on_fetch_spk`'s first parameter is now
        # `curated_download` -- so a direct connection binds the checked state
        # to the download consent. It is False today because this button is
        # not checkable, which is exactly the kind of accident that survives
        # until somebody makes it checkable and an 80 MB transfer starts with
        # no dialog.
        self.spk_fetch_button.clicked.connect(lambda: self._on_fetch_spk())
        egrid.addWidget(self.spk_fetch_button, 1, 0, 1, 2)
        self.add_file_field(egrid, 2, "SPICE metakernel:",
                            "generic_body.spice_metakernel",
                            "Metakernels (*.tm);;Kernels (*.bsp);;All Files (*)")
        #: What a body in the curated registry is served from, shown BEFORE
        #: anything is pressed. Blue rather than red: it is a statement of
        #: provenance, not a warning, and it is the sentence that replaces
        #: HORIZONS' (correct, useless) refusal for such a body.
        self.curated_note = _muted("", color="#64b5f6", size=10)
        egrid.addWidget(self.curated_note, 3, 0, 1, 2)
        self.coverage_note = _muted("", color="#EF9A9A", size=10)
        egrid.addWidget(self.coverage_note, 4, 0, 1, 2)
        check = QPushButton("Check SPK coverage")
        check.setToolTip(get_tooltip("ui.spk_coverage"))
        check.clicked.connect(self._check_coverage)
        egrid.addWidget(check, 5, 1)
        orbit_button = QPushButton("Orbital elements on the Body page…")
        orbit_button.setToolTip(get_tooltip("ui.orbit_on_body_page"))
        orbit_button.clicked.connect(self._open_body_page)
        egrid.addWidget(orbit_button, 6, 1)
        ephemeris.setLayout(egrid)
        layout.addWidget(ephemeris)

        # -- spin -------------------------------------------------------------
        spin = QGroupBox("Spin state")
        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        self.add_double_field(grid, 0, "Rotation period (hours):",
                              "generic_body.spin.period_hours", -1.0, 100000.0,
                              decimals=6)
        self.add_double_field(grid, 1, "Spin pole right ascension (deg):",
                              "generic_body.spin.pole_ra_deg", -1.0, 360.0,
                              decimals=3)
        self.add_double_field(grid, 2, "Spin pole declination (deg):",
                              "generic_body.spin.pole_dec_deg", -90.0, 90.0,
                              decimals=3)
        # Per FIELD, not per card. A period from SBDB beside a pole from a
        # lightcurve inversion is the ordinary case, and the two are not the
        # same kind of number.
        self.spin_provenance = _muted("", size=10)
        self.spin_provenance.setToolTip(get_tooltip("ui.spin_provenance"))
        grid.addWidget(self.spin_provenance, 3, 0, 1, 2)
        spin.setLayout(grid)
        layout.addWidget(spin)

        # -- shape -------------------------------------------------------------
        shape = QGroupBox("Shape")
        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        from gui_core.body_cards import GENERIC_SHAPE_TYPES
        shape_combo = self.add_combo_field(
            grid, 0, "Shape:", "generic_body.shape.type",
            list(GENERIC_SHAPE_TYPES))
        shape_combo.currentTextChanged.connect(
            lambda _t: self._propagate_shape_mesh())
        self.add_double_field(grid, 1, "Radius (km):",
                              "generic_body.shape.radius_km", -1.0, 100000.0,
                              decimals=4)
        self.add_file_field(grid, 2, "Shape model:",
                            "generic_body.shape.mesh_file",
                            "Shape models (*.obj *.ply);;All Files (*)")
        self.add_double_field(grid, 3, "Shape model scale:",
                              "generic_body.shape.mesh_scale", 0.0, 100000.0,
                              decimals=6)
        self.damit_button = QPushButton(self.SHAPE_BUTTON_DEFAULT)
        self.damit_button.setToolTip(get_tooltip("ui.fetch_damit_shape"))
        self.damit_button.clicked.connect(self._on_fetch_damit)
        grid.addWidget(self.damit_button, 4, 1)
        self.shape_note = _muted("")
        grid.addWidget(self.shape_note, 5, 0, 1, 2)
        self.shape_provenance = _muted("", size=10)
        self.shape_provenance.setToolTip(get_tooltip("ui.shape_provenance"))
        grid.addWidget(self.shape_provenance, 6, 0, 1, 2)
        #: Non-empty when the configured spin and the shape model's own spin
        #: describe different bodies. Red, because it is not advice.
        self.spin_shape_flag = _muted("", color="#EF9A9A", size=10)
        grid.addWidget(self.spin_shape_flag, 7, 0, 1, 2)
        shape.setLayout(grid)
        layout.addWidget(shape)
        self.widgets["generic_body.shape.mesh_file"].textChanged.connect(
            lambda _t: self._refresh_shape())
        self.widgets["generic_body.shape.mesh_scale"].valueChanged.connect(
            lambda _v: self._propagate_shape_mesh())

        # -- surface ------------------------------------------------------------
        surface = QGroupBox("Surface properties")
        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        # The LABEL is rewritten by `_show_albedo_provenance` when the value
        # came from SBDB, to carry the p_V and G it was computed from. A
        # converted number with no visible conversion is indistinguishable
        # from an unconverted one, and that distinction is the whole point.
        self.albedo_label = self.add_double_field(
            grid, 0, "Bond albedo:", "generic_body.surface.albedo", 0.0, 1.0,
            decimals=4)
        self.albedo_provenance = _muted("", size=10)
        # Row 13: below every control in this group. It used to sit at row 9,
        # which is now the chi field -- two widgets in one grid cell overlap
        # and one of them is unreachable.
        grid.addWidget(self.albedo_provenance, 13, 0, 1, 2)
        #: The grid cell the albedo's own QLabel sits in, so the label can
        #: carry the p_V and G a fetched value was computed from.
        self._albedo_label_cell = (grid, 0)
        self.add_double_field(grid, 1, "Emissivity:",
                              "generic_body.surface.emissivity", 0.0, 1.0,
                              decimals=4)
        self.add_double_field(grid, 2, "Surface conductivity ks (W/m/K):",
                              "generic_body.surface.ks", 0.0, 100.0,
                              decimals=8)
        self.add_double_field(grid, 3, "Deep conductivity kd (W/m/K):",
                              "generic_body.surface.kd", 0.0, 100.0,
                              decimals=8)
        self.add_double_field(grid, 4, "Surface density rhos (kg/m^3):",
                              "generic_body.surface.rhos", 0.0, 20000.0,
                              decimals=3)
        self.add_double_field(grid, 5, "Deep density rhod (kg/m^3):",
                              "generic_body.surface.rhod", 0.0, 20000.0,
                              decimals=3)
        self.add_double_field(grid, 6, "Heat capacity cp0 (J/kg/K):",
                              "generic_body.surface.cp0", 0.0, 10000.0,
                              decimals=3)
        # H, chi and Qb had NO control anywhere in this application, on any
        # page, while `generic_body.surface.{H,chi,Qb}` are keys the engine
        # reads and this card is the only screen that describes a generic
        # body's surface. So the card seeded or showed seven of the ten
        # properties a run uses and the other three were unreachable -- which
        # is worse than blank, because nothing said they existed. Their
        # tooltips were already written; only the controls were missing.
        self.add_double_field(grid, 7, "Regolith H scale (m):",
                              "generic_body.surface.H", 0.0, 100.0,
                              decimals=4)
        self.add_double_field(grid, 8, "Radiative conductivity chi:",
                              "generic_body.surface.chi", 0.0, 100.0,
                              decimals=3)
        self.add_double_field(grid, 9, "Geothermal heat flow Qb (W/m^2):",
                              "generic_body.surface.Qb", 0.0, 10.0,
                              decimals=6)
        self.ti_note = _muted("")
        self.ti_note.setToolTip(get_tooltip("ui.thermal_inertia_manual"))
        grid.addWidget(self.ti_note, 10, 0, 1, 2)
        #: Where the surface numbers came from. Its own line, because "engine
        #: default" and "a value you typed" are different claims and the card
        #: used to make neither.
        self.surface_provenance = _muted("", size=10)
        grid.addWidget(self.surface_provenance, 11, 0, 1, 2)
        self.surface_refusal = _muted("", color="#EF9A9A", size=10)
        grid.addWidget(self.surface_refusal, 12, 0, 1, 2)
        surface.setLayout(grid)
        layout.addWidget(surface)
        for key in ("generic_body.surface.ks", "generic_body.surface.kd",
                    "generic_body.surface.rhos", "generic_body.surface.rhod",
                    "generic_body.surface.cp0"):
            self.widgets[key].valueChanged.connect(
                lambda _v: self._refresh_surface())

        layout.addWidget(self.time_group())

        # -- radiation exchange -----------------------------------------------
        # This card had NO radiation control either, and it is the route EVERY
        # asteroid takes: a concave shape model is exactly the geometry where
        # facet-to-facet exchange matters, and `radiation.enabled` was
        # unreachable for all of them. No beaming switch -- the Smith slope
        # distribution is a lunar regolith calibration and
        # `docs/radiation-exchange.md` section 8.4 declines to carry it to
        # small bodies.
        layout.addWidget(self.radiation_group())

        layout.addWidget(self.thermal_group())
        layout.addWidget(self.output_group())
        self._make_report(layout)
        layout.addStretch()
        self.create_scroll_area(content)

    #: What `simulation.body` says while this card has no name in it yet.
    #: `examples/justitia_generic-test.yaml` ships exactly this pair
    #: (`simulation.body: Custom` beside `generic_body.name: Custom`), and
    #: `ConfigurationPanel._on_body_combo_changed` already writes it for the
    #: custom-body route, so this is the spelling the project uses.
    UNNAMED_BODY = "Custom"

    def _mirror_body_name(self, text=None):
        """Keep `simulation.body` on the body this card is describing.

        Not cosmetic. `simulation.body` is the ONLY thing that says which card
        a configuration belongs to -- `body_cards.detect_route` reads it, the
        pre-flight body-identity check reads it, and `_config_for_yaml` uses it
        to decide whether a `generic_body:` section is this run's body or the
        previous card's leftovers. Left unwritten, a generic-body card
        inherited `Mercury` (or `Moon`, or `Europa`) from whichever card ran
        before it, and the configuration then named two different bodies.

        TWO things it must not do, both reproduced before they were guarded:

        * **Write while the card is only being displayed.** `load_card` runs
          under `_loading`, and `set_text` does not block `textChanged`, so
          this fired on every open. Showing a configuration is not editing it
          (`load_from_config`'s own docstring, and
          `test_showing_a_card_does_not_rewrite_the_configuration`).
        * **Strip the `[H] ` prefix.** That prefix is load-bearing GUI state:
          `ConfigurationPanel._on_body_combo_changed` branches on it,
          `add_horizons_body` inserts the combo entry under the prefixed
          spelling, and `ConfigManager.body_identity_conflict` exempts it. It
          is removed from the engine-facing YAML by `_config_for_yaml`, which
          is the right place for it to go. A HORIZONS body already names
          itself, so there is nothing here to mirror.
        """
        if self._loading:
            return
        self._write_body_name(text)

    def _write_body_name(self, text=None):
        current = str(self.config_manager.get_value(
            "simulation.body", "") or "").strip()
        if current.startswith("[H] "):
            return
        if text is None:
            text = self.config_manager.get_value("generic_body.name", "")
        name = str(text or "").strip()
        wanted = name or self.UNNAMED_BODY
        if current == wanted:
            return
        self.config_manager.set_value("simulation.body", wanted)

    def _repair_body_name(self):
        """Fix `simulation.body` when it names a body this card is not.

        Runs from `load_card`, so it is deliberately narrow: it writes ONLY
        when the stored body is a body the engine has built-in properties for,
        or the unnamed placeholder while a real name exists. Both are
        configurations that would run as something other than what they say --
        a built-in name beside a `generic_body:` section is the contradiction
        `ConfigManager.body_identity_conflict` reports, and letting a card
        display it unchanged would hand the user a run that silently drops
        either the body or its definition.

        Anything else the user or their file chose is left exactly as it is.
        """
        from gui_core.config_manager import BUILTIN_BODY_NAMES
        current = str(self.config_manager.get_value(
            "simulation.body", "") or "").strip()
        name = str(self.config_manager.get_value(
            "generic_body.name", "") or "").strip()
        contradicts = current.lower() in BUILTIN_BODY_NAMES
        placeholder = (current == self.UNNAMED_BODY and bool(name))
        if not (contradicts or placeholder):
            return
        self._write_body_name(name)

    def load_card(self):
        super().load_card()
        self._repair_body_name()
        source = str(self.config_manager.get_value(
            "generic_body.ephemeris_source", "auto") or "auto").lower()
        self.spice_radio.blockSignals(True)
        self.kepler_radio.blockSignals(True)
        self.spice_radio.setChecked(source == "spice")
        self.kepler_radio.setChecked(source != "spice")
        self.spice_radio.blockSignals(False)
        self.kepler_radio.blockSignals(False)
        self._refresh_curated()
        self._refresh_shape()
        self._refresh_surface()
        self._refresh_provenance()
        self.refresh_radiation()

    # -- the curated tier -----------------------------------------------------

    def _curated_entry(self):
        """This card's body in the curated registry, or None. Never raises.

        Reached from `load_card` and from the name field's own handler, so it
        runs on the GUI thread on every open. It is a table lookup and two
        `stat` calls -- no network, and deliberately no hashing of an 80 MB
        kernel, which is what `curated_kernel_status` would cost if it
        verified rather than looked.
        """
        try:
            from gui_core.curated_sources import curated_body
            return curated_body(
                str(self.config_manager.get_value(
                    "generic_body.name", "") or ""),
                str(self.config_manager.get_value(
                    "generic_body.naif_id", "") or ""))
        except Exception:                                    # noqa: BLE001
            logger.warning("[curated] the registry could not be consulted; "
                           "this card falls back to the live services",
                           exc_info=True)
            return None

    def _refresh_curated(self):
        """Say what a curated body is served from, before anything is pressed.

        The whole point of the tier is that the user never meets HORIZONS'
        refusal for such a body, so the replacement statement has to be on the
        card at rest -- not only in the answer to a button they may never
        press because the button's own label sent them elsewhere.

        Wrapped whole. This is connected to the Name field's ``textChanged``,
        and an exception escaping a Qt slot is a ``qFatal`` that takes the
        window down with any unsaved edits. ``curated_kernel_status`` reaches
        ``fetch_kernels_module``, which EXECUTES ``scripts/fetch_kernels.py``:
        a locally broken copy of that script would otherwise kill the GUI on
        the next keystroke rather than degrading to the live services.
        """
        try:
            self._refresh_curated_unguarded()
        except Exception:                                    # noqa: BLE001
            logger.warning("[curated] the banner could not be refreshed; the "
                           "card falls back to the live services",
                           exc_info=True)
            self.curated_note.setText("")
            self.spk_fetch_button.setText(self.SPK_BUTTON_DEFAULT)
            self.damit_button.setText(self.SHAPE_BUTTON_DEFAULT)

    def _refresh_curated_unguarded(self):
        entry = self._curated_entry()
        if entry is None:
            self.curated_note.setText("")
            self.spk_fetch_button.setText(self.SPK_BUTTON_DEFAULT)
            self.damit_button.setText(self.SHAPE_BUTTON_DEFAULT)
            return

        ephemeris = entry.ephemeris
        self.spk_fetch_button.setText(f"Use {ephemeris.short}")
        self.damit_button.setText("Mission shape model (radar)…")

        from gui_core.curated_sources import curated_kernel_status
        status = curated_kernel_status(entry, self._project_root())
        # "Ready" is every kernel the metakernel NAMES, not just the body's
        # own SPK. The first draft reported "already on this disk" over a tree
        # missing three of bennu.tm's four members, and the coverage check
        # then refused -- a banner asserting readiness in front of a refusal.
        state = ("every kernel it loads is on this disk — no download and no "
                 "network needed" if status.ready else status.summary())
        name = (status.metakernel_path.name if status.metakernel_path
                else ephemeris.metakernel)
        self.curated_note.setText(
            f"Ephemeris source: {ephemeris.description}. "
            f"{ephemeris.reference}. {name} → {state}\n"
            f"HORIZONS is not asked for this body and DAMIT is not searched "
            f"for its shape; both would answer correctly that they have "
            f"nothing, which is a dead end rather than an answer.")

    # -- name -> parameters (item 3) ------------------------------------------

    def _on_lookup(self):
        """Resolve the name in the field, cache first, HORIZONS second.

        The cache is on disk and answers instantly, so it runs here. HORIZONS
        is a network request with a timeout, so it runs on a worker thread --
        a blocking request on the GUI thread is a frozen window, and "never a
        hang" is the requirement this control was asked for.

        Nothing is applied until a lookup SUCCEEDS, and every field it writes
        stays editable afterwards. A failure changes nothing at all and says
        which kind of failure it was.
        """
        from gui_core.body_lookup import lookup_body

        query = str(self.config_manager.get_value(
            "generic_body.name", "") or "").strip()
        if not query:
            self.lookup_note.setText(
                "Type a name, a number or a designation in the Name field "
                "first, then press Look up.")
            return

        result = lookup_body(query, self._project_root(), allow_online=False)
        if result.found:
            self._apply_lookup(result)
            return

        if result.ambiguous:
            # More than one cached body matches. Stop here rather than let
            # HORIZONS turn an ambiguous question into one confident answer.
            self.lookup_note.setText(result.message)
            return

        # A request is already in flight. Starting a second one would rebind
        # `self._lookup_worker` and drop the last reference to a QThread that
        # has not finished, which Qt answers with "QThread: Destroyed while
        # thread is still running" and std::terminate -- taking any unsaved
        # edits with it.
        if self._lookup_worker is not None and self._lookup_worker.isRunning():
            self.lookup_note.setText(
                "A lookup is already running. Wait for it to finish.")
            return

        # Cache miss. Say so, then ask HORIZONS off the GUI thread.
        self.lookup_note.setText(f"{result.message} Asking HORIZONS...")
        self.lookup_button.setEnabled(False)
        worker = _LookupWorker(query, self._project_root(), self.route)
        worker.done.connect(self._on_lookup_done)
        # Keep it referenced until Qt itself says the thread has ended, then
        # let Qt delete it. `done` is emitted as the last statement of run(),
        # so the slot can arrive BEFORE run() has returned; `finished` cannot.
        worker.finished.connect(worker.deleteLater)
        worker.finished.connect(self._on_lookup_thread_finished)
        self._lookup_worker = worker
        worker.start()

    def _on_lookup_thread_finished(self):
        self._lookup_worker = None

    def _on_lookup_done(self, result, route):
        self.lookup_button.setEnabled(True)
        if route != self.route or not self._active:
            # The user moved on while HORIZONS was answering. Applying now
            # would write this body's generic_body.* into whatever
            # configuration is on screen -- which, if they went to a built-in
            # body card, is exactly the cross-card residue this branch exists
            # to remove.
            logger.info("[body-lookup] discarding a result for %s; the "
                        "active route is now %s", route, self.route)
            return
        if result.found:
            self._apply_lookup(result)
        else:
            self.lookup_note.setText(result.message)

    def _apply_lookup(self, result):
        """Write what the lookup found, and report every key of it."""
        for key, value in result.values.items():
            self.config_manager.set_value(key, value)
        # One source label per field, so the per-field lines say where these
        # came from too and are not blank until somebody presses a second
        # button. The cache and HORIZONS are different sources and are named
        # differently: a cached body_config.yaml is whatever was written into
        # it, which is not the same claim as "SBDB says so today".
        source = ("the local cache" if result.source == "cache"
                  else "HORIZONS")
        self._provenance().update({key: source for key in result.values})
        # A field the lookup CONVERTED on the way in overrides the blanket
        # label, because "HORIZONS" beside a Bond albedo that HORIZONS
        # published as a geometric one is the label that hid this for months.
        self._provenance().update(dict(result.field_provenance))
        self.provenance_note.setText(result.provenance())
        lines = [result.message]
        lines.extend(result.notes)

        # PTM-145. A kernel that does not span the run is not an error the
        # engine reports: it substitutes Keplerian elements for the epochs
        # outside coverage and logs NOTHING, which is how fourteen of fifteen
        # Justitia configurations "passed". If this body brought a kernel, its
        # coverage is checked HERE rather than waiting for the user to press
        # the separate button.
        if result.kernel:
            self.config_manager.set_value(
                "generic_body.spice_metakernel", result.kernel)
            lines.append(f"Kernel: {result.kernel}")

        self.lookup_note.setText("\n".join(lines))
        self.load_from_config()
        self._refresh_provenance()
        if result.kernel:
            self._check_coverage()
        self.value_changed.emit()

    # -- the three fetches (items 1, 2, 3 and 4) -------------------------------

    #: Where each field's value came from, ``{dotted key: short label}``.
    #: Rebuilt from what a fetch reports, and DROPPED for any key the user
    #: subsequently edits -- a provenance line that survives a hand edit is a
    #: label claiming a source for a number that no longer came from it.
    def _provenance(self):
        if not hasattr(self, "_field_provenance"):
            self._field_provenance = {}
            self._provenance_body = self._current_body_name()
        # Provenance belongs to ONE body. Left alone it would survive the name
        # changing and go on saying "SBDB" over a different asteroid's numbers
        # -- the same residue, one field narrower, that
        # `_repair_body_name` removes from `simulation.body`.
        current = self._current_body_name()
        if current != getattr(self, "_provenance_body", current):
            self._field_provenance = {}
            self._provenance_body = current
        return self._field_provenance

    def _current_body_name(self):
        return str(self.config_manager.get_value(
            "generic_body.name", "") or "").strip().lower()

    def _start_fetch(self, kind, call, button, note):
        """Run `call` on a worker thread, with the one-at-a-time guard.

        `call` takes ONE argument: a function that reports a line of progress.
        Fetches with nothing to report ignore it. It is passed in rather than
        read off the worker so that `gui_core.body_fetch` stays free of Qt --
        the whole fetch layer is measurable without a window, and that is what
        lets the curated tier be tested at all.

        Starting a second fetch of the same kind would rebind the attribute
        holding the running QThread and drop the last reference to it, which
        Qt answers with "QThread: Destroyed while thread is still running" and
        std::terminate -- taking any unsaved edits with it. The same lesson
        `_on_lookup` already carries.
        """
        attribute = f"_fetch_worker_{kind}"
        running = getattr(self, attribute, None)
        if running is not None and running.isRunning():
            note.setText(f"A {kind} fetch is already running. Wait for it to "
                         f"finish.")
            return
        button.setEnabled(False)
        worker = _FetchWorker(call, self.route, kind)
        worker.done.connect(self._on_fetch_done)
        # Queued across the thread boundary by Qt, so the label is only ever
        # written on the GUI thread. Bound to THIS note, so a long transfer
        # reports into the same place its result will land.
        worker.progress.connect(
            lambda line, target=note: target.setText(line))
        worker.finished.connect(worker.deleteLater)
        worker.finished.connect(
            lambda: setattr(self, attribute, None))
        setattr(self, attribute, worker)
        worker.start()

    #: Which button and which note belong to each fetch kind.
    def _fetch_controls(self, kind):
        return {
            "spk": (self.spk_fetch_button, self.coverage_note),
            "damit_shape": (self.damit_button, self.shape_provenance),
            "sbdb": (self.sbdb_button, self.lookup_note),
        }[kind]

    def _on_fetch_done(self, result, route):
        button, note = self._fetch_controls(result.kind)
        button.setEnabled(True)
        if route != self.route or not self._active:
            # The user moved on while the service was answering. Applying now
            # would write this body's generic_body.* into whatever
            # configuration is on screen -- the cross-card residue the
            # body-identity gate exists to catch. The FILES are still on disk
            # and still correct; only the write into a live card is dropped.
            #
            # SAID OUT LOUD, on the card and at WARNING. It used to be an
            # `info` log and nothing else, while `_on_fetch_spk` had already
            # written "Wiring <body> to <ephemeris>..." into this very note --
            # a present-tense claim that the wiring was happening, left
            # standing over a result that was thrown away. The user comes back
            # to a card that says it did the thing, a configuration that never
            # got the key, and a run the engine refuses.
            logger.warning("[body-fetch] a %s result for %s was NOT applied; "
                           "the active route is now %s. Nothing was changed.",
                           result.kind, route, self.route)
            note.setText(
                f"NOT APPLIED. The {result.kind} answer came back after this "
                f"card stopped being the one on screen, so nothing was "
                f"written -- writing it now would put {route}'s properties "
                f"into whatever body is open. Any files it fetched are on "
                f"disk and are fine. Press the button again to apply it.")
            return
        self._apply_fetch(result, note)

    def _apply_fetch(self, result, note):
        """Write what a fetch returned, and report every key of it.

        A refusal writes NOTHING. That is the whole of PTM-145 in one branch:
        the kernel is on disk and is a perfectly good kernel, it simply does
        not cover this run, and the engine would serve the uncovered epochs
        from orbital elements without a word.
        """
        from gui_core.body_fetch import spin_consistency

        if result.ok and result.values:
            if result.kind == "damit_shape":
                self.spin_shape_flag.setText(
                    spin_consistency(self.config_manager.get_config(),
                                     result.values))
            for key, value in result.values.items():
                self.config_manager.set_value(key, value)
            self._provenance().update(result.provenance)
            self.load_from_config()

        note.setText("\n".join(result.lines()))
        self._refresh_provenance()
        self._refresh_curated()
        self._refresh_shape()
        self._refresh_surface()
        if result.ok and result.values:
            self.value_changed.emit()
        # AFTER the message is on screen, so the dialog appears over a card
        # that already says what is being asked and why -- and only for a
        # result that carries the flag, never for one whose message happens to
        # mention a download.
        if getattr(result, "offer", ""):
            self._offer_curated_download(result)

    def _on_fetch_spk(self, curated_download=False):
        """Get this run's ephemeris: the curated registry, or HORIZONS.

        `curated_download` is False on the button press and True only on the
        second pass, after the user has answered the confirmation the first
        pass produced. That two-pass shape is why the offer is a FLAG on the
        result rather than a phrase in its message: the card has to be able to
        tell "this needs your consent" from "this failed" without reading
        English.
        """
        from gui_core.body_fetch import fetch_spk

        name = str(self.config_manager.get_value(
            "generic_body.name", "") or "").strip()
        naif_id = str(self.config_manager.get_value(
            "generic_body.naif_id", "") or "").strip()
        config = self.config_manager.get_config()
        project_root = self._project_root()
        entry = self._curated_entry()
        if entry is not None:
            self.coverage_note.setText(
                f"Wiring {entry.name} to {entry.ephemeris.description}. "
                f"HORIZONS is not being asked. Nothing is changed unless the "
                f"kernel is read back and shown to span this run.")
        else:
            self.coverage_note.setText(
                "Asking HORIZONS for a kernel covering this run, with a year "
                "of margin at each end. Nothing is changed unless what "
                "arrives is read back and shown to span the window.")
        # `emit` is the worker's own progress signal, so a long transfer's
        # output reaches the card LIVE and on the GUI thread. Wiring
        # `progress=None` here instead would leave the whole reporting path --
        # which the tests require, because an 80 MB download with no sign of
        # life is indistinguishable from a hung window -- dead in the one
        # place it matters.
        self._start_fetch(
            "spk",
            lambda emit: fetch_spk(config, project_root, name=name,
                                   naif_id=naif_id,
                                   curated_download=curated_download,
                                   progress=emit),
            self.spk_fetch_button, self.coverage_note)

    def _offer_curated_download(self, result):
        """Ask before spending a download, and run it if the answer is yes.

        Returns True when the question was asked, so the caller stops: the
        answer arrives through a second `_on_fetch_spk`, not through this
        return value.
        """
        if not result.offer:
            return False
        answer = QMessageBox.question(
            self, "Download this body's ephemeris?", result.offer,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if answer != QMessageBox.StandardButton.Yes:
            self.coverage_note.setText(
                "Nothing was downloaded and nothing has been changed. The "
                "same file can be fetched outside this window: see the "
                "message above for the command.")
            return True
        self._on_fetch_spk(curated_download=True)
        return True

    def _on_fetch_damit(self):
        """This body's shape: the curated registry, or DAMIT."""
        from gui_core.body_fetch import fetch_damit_shape

        name = str(self.config_manager.get_value(
            "generic_body.name", "") or "").strip()
        naif_id = str(self.config_manager.get_value(
            "generic_body.naif_id", "") or "").strip()
        radius_km = self._number("generic_body.shape.radius_km", 0.0)
        project_root = self._project_root()
        entry = self._curated_entry()
        self.shape_provenance.setText(
            f"Offering {entry.name}'s mission shape model: "
            f"{entry.shape.description}. DAMIT is not being searched."
            if entry is not None else
            "Asking DAMIT for a shape model. Most asteroids do not have one.")
        self._start_fetch(
            "damit_shape",
            lambda _emit: fetch_damit_shape(project_root, name=name,
                                            naif_id=naif_id,
                                            radius_km=radius_km),
            self.damit_button, self.shape_provenance)

    def _on_fetch_sbdb(self):
        """Fill in the physical properties, converting the two that need it."""
        from gui_core.body_fetch import fetch_sbdb_properties

        query = str(self.config_manager.get_value(
            "generic_body.name", "") or "").strip()
        if not query:
            self.lookup_note.setText(
                "Type a name, a number or a designation in the Name field "
                "first, then press Fill in properties.")
            return
        self.lookup_note.setText(
            f"Asking the Small-Body Database about {query}...")
        self._start_fetch(
            "sbdb",
            lambda _emit: fetch_sbdb_properties(query),
            self.sbdb_button, self.lookup_note)

    def on_widget_changed(self, key, value):
        """A hand edit drops that field's provenance.

        A source line that survives an edit is a label claiming a database
        computed a number the user has since replaced -- and the Bond-albedo
        label is the worst case, because it would go on naming the p_V and G
        the value is no longer derived from. `_loading` is exempt: showing a
        card is not editing it.
        """
        if not self._loading:
            self._provenance().pop(key, None)
            if hasattr(self, "spin_provenance"):
                self._refresh_provenance()
        super().on_widget_changed(key, value)

    def _refresh_provenance(self):
        """Redraw the per-field source lines from what the fetches reported."""
        sources = self._provenance()

        spin_lines = []
        for key, label in (("generic_body.spin.period_hours",
                            "Rotation period"),
                           ("generic_body.spin.pole_ra_deg", "Spin pole")):
            source = sources.get(key)
            spin_lines.append(f"{label}: {source or 'manual'}")
        self.spin_provenance.setText("Source — " + ", ".join(spin_lines) + ".")

        albedo_source = sources.get("generic_body.surface.albedo")
        label_widget = self._grid_label(*self._albedo_label_cell)
        if albedo_source and label_widget is not None:
            label_widget.setText(f"Bond albedo (from {albedo_source}):")
        elif label_widget is not None:
            label_widget.setText("Bond albedo:")
        self.albedo_provenance.setText(
            f"Source — {albedo_source or 'manual'}.")

        # The surface group's own line. A seeded field says so by name, so
        # "engine default" is never mistaken for a measurement of this body
        # and an edited field stops claiming it (`on_widget_changed` drops the
        # label). Named fields rather than a count: "3 fields are defaults" does
        # not tell anyone WHICH conductivity they are about to publish.
        seeded = [field for field in GENERIC_SURFACE_FIELDS
                  if sources.get(f"generic_body.surface.{field}")
                  == self.ENGINE_DEFAULT_LABEL]
        if hasattr(self, "surface_provenance"):
            self.surface_provenance.setText(
                f"Source — {', '.join(seeded)} are the engine's generic-body "
                f"defaults (GenericBody::defaultRockyAsteroid), shown so they "
                f"are visible and editable; a run that omitted them would use "
                f"these same values silently. Everything else is yours."
                if seeded else
                "Source — every surface property here was entered or fetched, "
                "not defaulted.")

    @staticmethod
    def _grid_label(grid, row):
        item = grid.itemAtPosition(row, 0)
        widget = item.widget() if item is not None else None
        return widget if isinstance(widget, QLabel) else None

    def _on_ephemeris_changed(self, _checked):
        """Record the choice, and never leave SPICE half-selected in silence.

        This control writes ONE key. Choosing SPICE with the kernel field empty
        therefore produces `ephemeris_source: spice` beside a
        `spice_metakernel: ""` that `_config_for_yaml` then STRIPS -- so the
        emitted YAML names a SPICE source and no kernel at all, and the engine
        refuses it after the mesh is built (generic_body.h:94). That is
        precisely the configuration a click-through produced on 2026-08-28;
        reproducing it took the lookup, the shape button and this radio, and
        the emitted file matched the failing one on 24 of 24 keys.

        The radio still does what it says -- the choice is the user's -- but a
        selection that cannot run now says so HERE, in the refusal color, next
        to the button that fixes it, instead of surfacing as an engine abort.
        """
        source = "keplerian" if self.kepler_radio.isChecked() else "spice"
        self.config_manager.set_value("generic_body.ephemeris_source", source)
        kernel = str(self.config_manager.get_value(
            "generic_body.spice_metakernel", "") or "").strip()
        if source == "keplerian":
            self.coverage_note.setText(
                "Keplerian elements, chosen deliberately. The run's own record "
                "will say so.")
        elif kernel:
            self.coverage_note.setText(
                "SPICE. Check the kernel's coverage below: when a requested "
                "epoch falls outside it the engine substitutes Keplerian "
                "elements for that epoch and logs nothing at all.")
        else:
            self.coverage_note.setText(
                f"SPICE is selected and NO METAKERNEL IS SET, so this run "
                f"cannot start: the engine refuses a SPICE ephemeris with no "
                f"kernel and does not fall back to anything. Press "
                f"'{self.spk_fetch_button.text()}' to wire one, pick a .tm "
                f"file in the field below, or choose Keplerian elements "
                f"instead so the record says that is what ran.")
        self.value_changed.emit()

    def _check_coverage(self):
        """Compare the kernel's own coverage with the whole run window.

        Three outcomes, kept distinct on purpose: covered, NOT covered, and
        UNKNOWN. Reporting "could not check" as "fine" is how fifteen runs
        completed on the silent per-epoch Keplerian fallback while recording
        SPICE, so an unanswerable question is answered as unanswerable.

        The NAIF ID goes with the kernel path, and it is not optional detail.
        This field usually holds a METAKERNEL, which loads a planetary
        ephemeris (de440s spans 1849 to 2150) beside the target body's own
        short arc; without an ID to report coverage FOR, the honest answer
        would have to be the window common to every body in it. With the ID
        the answer is about this asteroid, which is the question the card is
        asking. The project root goes with it because the committed
        metakernels are written with paths relative to the repository, not to
        wherever the GUI happened to be launched from.
        """
        from gui_core.body_cards import run_window, spk_coverage
        kernel = str(self.config_manager.get_value(
            "generic_body.spice_metakernel", "") or "")
        coverage = spk_coverage(
            kernel,
            naif_id=self.config_manager.get_value("generic_body.naif_id", ""),
            project_root=self._project_root())
        if not coverage.available:
            self.coverage_note.setText(
                f"Coverage UNKNOWN: {coverage.detail}. Unknown is not the same "
                f"as fine — the engine's fallback is silent, so an unchecked "
                f"kernel is an unchecked run.")
            return
        start, end, spin_up, note = run_window(self.config_manager.get_config())
        header = (f"{coverage.detail}: covers {coverage.start_utc} to "
                  f"{coverage.end_utc} ({coverage.days:.1f} days).\n")
        if start is None:
            self.coverage_note.setText(header + "Coverage UNKNOWN for this "
                                       "run: " + note)
            return
        earliest = spin_up or start
        verdict = coverage.covers(earliest, end)
        body = (f"Your run needs {earliest} to {end}"
                + (" (including spin-up)" if spin_up else "")
                + f"; {note}.\n")
        if verdict is None:
            tail = ("Coverage UNKNOWN: the two windows could not be compared "
                    "because one of the timestamps is not in the form the "
                    "kernel reports. Check it yourself before trusting a "
                    "SPICE run.")
        elif verdict:
            tail = "The whole window is inside this kernel."
        else:
            tail = ("PART OR ALL OF THIS RUN IS OUTSIDE THIS KERNEL. Every "
                    "epoch outside it would be served by the Keplerian "
                    "fallback, silently, with nothing in the log. Fetch a "
                    "wider kernel, or choose Keplerian deliberately so the "
                    "choice is recorded.")
        self.coverage_note.setText(header + body + tail)

    def _open_body_page(self):
        parent = self.parent()
        while parent is not None:
            if hasattr(parent, "switch_to_body_definition_tab"):
                parent.switch_to_body_definition_tab()
                return
            parent = parent.parent()

    def _propagate_shape_mesh(self):
        """Mirror the shape model to the TOP-LEVEL mesh path.

        `generic_body.shape.mesh_file` is metadata. The thermal solver loads
        `mesh.import_file` (`ptm_thermal_sim.cpp:6592`), so without this mirror
        choosing a shape model does nothing at all and the run uses the
        procedural sphere from `mesh.resolution` -- with the user's DAMIT model
        sitting unread on disk. Only for `custom_mesh`, so the other shape modes
        cannot clobber a mesh path the user set elsewhere. Same rule, and the
        same reason, as the Body page's own mirror.
        """
        if self._loading:
            return
        shape = self.widgets.get("generic_body.shape.type")
        if shape is None or shape.currentText() != GENERIC_MESH_SHAPE:
            return
        mesh_file = str(self.config_manager.get_value(
            "generic_body.shape.mesh_file", "") or "").strip()
        if not mesh_file:
            return
        scale = self._number("generic_body.shape.mesh_scale", 1.0)
        self.config_manager.set_value("mesh.import_file", mesh_file)
        self.config_manager.set_value("mesh.scale_factor", scale)
        self.value_changed.emit()

    def _refresh_shape(self):
        self._propagate_shape_mesh()
        path = str(self.config_manager.get_value(
            "generic_body.shape.mesh_file", "") or "")
        if not path:
            self.shape_note.setText("")
            return
        notes = []
        resolved = Path(self._project_root()) / path
        candidate = resolved if resolved.exists() else Path(path)
        if candidate.exists():
            notes.append(f"{candidate.name} is on disk")
        else:
            notes.append(f"{path} is NOT on disk")
        if str(path).lower().endswith(".obj"):
            # The .gitignore line is real (line 14, `*.obj`, a rule for C
            # object files) but "therefore untracked" is not always true: a
            # curated body's shape is force-added and SHIPS. Saying it is
            # untracked over a file that is in git would send a user to back
            # up something the repository already carries, and would make the
            # same sentence untrustworthy where it IS true.
            entry = self._curated_entry()
            tracked = (entry is not None and entry.shape.tracked
                       and str(path).replace("\\", "/").endswith(
                           entry.shape.relpath))
            notes.append(
                "shape models in .obj are matched by this project's "
                ".gitignore, but this one is force-added and TRACKED, so it "
                "ships with the repository"
                if tracked else
                "shape models in .obj are excluded by this project's "
                ".gitignore, so this file is untracked and may exist in one "
                "copy only — back it up, or add an exception for it")
        notes.append(
            "asteroid shape models are conventionally in KILOMETERS while "
            "lunar and Europa terrain meshes are in meters; the run prints "
            "the units it inferred")
        self.shape_note.setText(". ".join(notes) + ".")

    #: The label a seeded field carries until the user edits it. Matched by
    #: `_refresh_surface`, so the wording lives in one place.
    ENGINE_DEFAULT_LABEL = "engine default"

    def seed_engine_surface_defaults(self):
        """Fill EMPTY surface properties with the engine's own generic-body
        defaults, and report which ones were filled.

        Returns ``{dotted key: value}`` for the keys it wrote -- empty when
        there was nothing to fill.

        Why this exists. `generic_body.surface.*` has no entry in
        DEFAULT_CONFIG, so a fresh asteroid card showed 0.0 in every box and
        emitted no surface block at all. The run was not wrong -- the engine
        starts from `GenericBody::defaultRockyAsteroid()` and overwrites only
        the positive values a configuration supplies, so an absent block runs
        on exactly the numbers seeded here -- but NOTHING SAID SO. A user
        looking at a conductivity of 0.0 beside a thermal inertia of "0.0
        J m^-2 K^-1 s^-1/2" reasonably concludes the card needs numbers and
        types their own; that is what happened on 2026-08-28, and the guesses
        that went in were not the engine's defaults.

        Two rules make this safe rather than a silent substitution:

        * it writes ONLY keys the configuration does not already have, so a
          fetched albedo, a lookup's spin-derived value or anything the user
          typed is never overwritten; and
        * every key it writes is labeled `engine default` in the provenance
          line under the group, and `on_widget_changed` drops that label the
          moment the field is edited -- so the card never claims the engine
          chose a number the user did.

        It is called from the body-first flow when a card is opened to set up
        a NEW run (`apply_defaults=True`), and NOT when landing on a
        configuration the user opened: a file that omits these keys means "use
        the engine's defaults", and writing them in would add ten keys to the
        author's own file on the next save.
        """
        written = {}
        for field in GENERIC_SURFACE_FIELDS:
            key = f"generic_body.surface.{field}"
            if self.config_manager.get_value(key, None) is not None:
                continue
            value = GENERIC_BODY_SURFACE_DEFAULTS[field]
            self.config_manager.set_value(key, value)
            written[key] = value
        if written:
            self._provenance().update(
                {key: self.ENGINE_DEFAULT_LABEL for key in written})
            self.load_from_config()
            self._refresh_provenance()
        return written

    def _refresh_surface(self):
        from gui_core.body_cards import tiny_surface_refusals
        from gui_tooltips import compute_thermal_inertia
        ks = self._number("generic_body.surface.ks", 0.0)
        rhos = self._number("generic_body.surface.rhos", 0.0)
        cp0 = self._number("generic_body.surface.cp0", 0.0)
        inertia = compute_thermal_inertia(ks, rhos, cp0)
        self.ti_note.setText(
            f"Thermal inertia at the surface = sqrt(ks x rhos x cp0) = "
            f"{inertia:.1f} J m^-2 K^-1 s^-1/2 — the number the observations "
            f"speak in.")
        refusals = tiny_surface_refusals(self.config_manager.get_config())
        self.surface_refusal.setText("\n".join(refusals))



class AsteroidCard(GenericBodyCard):
    ASTEROID = True


class CustomBodyCard(GenericBodyCard):
    ASTEROID = False


# ===========================================================================
# The landing screen and the run-type step
# ===========================================================================

class _BigButton(QPushButton):
    """A large button. `subtitle` is OPTIONAL and empty for a body button.

    A body button carries the body's name and nothing else -- not even the
    two-word maturity badge it briefly had, which is why there is no `badge`
    parameter here any more: a body's caveats belong on that body's card, said
    once. The run-type buttons still carry a summary line, because a run type
    is a decision about what the run produces and the name alone does not
    settle it.

    **The summary is a child label, not a second line of `setText`.** A
    QPushButton does not wrap its own text: it draws each newline-separated
    line on one line and ELIDES what does not fit. Measured on 2026-08-27 at
    this button's own minimum width, in this button's own font, the three
    run-type summaries wanted 962, 1348 and 1122 px against 156 px of room --
    so all three were cut off, and the reader saw a fragment ending in an
    ellipsis. A wrapping QLabel makes the button grow downward instead.

    **And the height must be RE-ASSERTED, because the application stylesheet
    silently throws the constructor's away.** That fix left
    ``setMinimumHeight(96)`` in the constructor and the text went on being cut
    off -- vertically this time -- in the real window. Measured 2026-08-28:
    `ModernStyle.get_stylesheet()` carries ``QPushButton { padding: 8px 16px;
    min-height: 28px; }``, and ``QStyleSheetStyle`` applies a stylesheet box by
    OVERWRITING the widget's minimum size. The button's own minimum height
    became 28 + 8 + 8 + 1 + 1 = 46 px, the layout then had 46 - 24 (margins) -
    4 (spacing) = 18 px for two labels, and each got 9 px where each needed 16:

    ======================  ==========  =========  =======
    what                    given       needed     verdict
    ======================  ==========  =========  =======
    title, every run type      9 px       16 px    clipped
    summary, every run type    9 px       16 px    clipped
    ======================  ==========  =========  =======

    Nothing in the widget noticed, because a QPushButton's own ``sizeHint`` is
    computed from its Qt text and IGNORES a layout set on it -- measured
    ``sizeHint().height() == 29`` on a button whose children need 60. So the
    height can be neither left to the size hint nor set once in the
    constructor. :meth:`required_height` derives it from ``QFontMetrics``, and
    :meth:`_apply_required_height` re-applies it after every event that could
    have re-polished the widget. The alternative of putting the number in the
    button's own stylesheet was measured and rejected: a bare ``min-height``
    there is inherited by the child QLabels, which then became 120 px tall.

    `text()` still returns the TITLE. The Qt text is emptied so the button does
    not paint it underneath the labels, and this override keeps `.text()`
    meaningful for the callers and gates that read it.
    """

    #: Chrome between the button's edge and its text, both axes. Qt's own
    #: default QPushButton content margins are smaller than this on macOS; the
    #: extra is what stops the text touching the frame.
    TEXT_PADDING_PX = 24
    TITLE_POINT_SIZE = 12
    #: Smaller and not bold. The summary is a sentence, not a heading, and at
    #: 12 pt bold it needed six lines of a four-line button.
    SUBTITLE_POINT_SIZE = 10
    #: The height a large button is MEANT to have. It is a floor, not the
    #: answer: a wrapped summary that needs more gets more.
    DESIGN_MIN_HEIGHT = 96
    #: Vertical gap between the title and the summary, matching the layout.
    LABEL_SPACING_PX = 4

    #: CLASS-level defaults, so the height hooks are safe during construction.
    #: `QPushButton.__init__` and `setFont` both deliver a `changeEvent`, which
    #: reaches `_apply_required_height` BEFORE the instance attributes exist --
    #: an AttributeError out of a Qt event handler, which is an abort, not a
    #: traceback. Looking these up on the class instead makes the early call
    #: return the design floor harmlessly.
    subtitle_label = None
    title_label = None
    _subtitle = ""
    _sizing = False

    def __init__(self, title, subtitle="", tooltip=""):
        super().__init__("")
        self._title = title
        self._subtitle = subtitle
        #: Guards the re-entry from `setMinimumHeight` -> resize -> re-apply.
        self._sizing = False
        self.setMinimumHeight(self.DESIGN_MIN_HEIGHT)
        self.setMinimumWidth(180)
        self.setToolTip(tooltip)
        font = QFont()
        font.setPointSize(self.TITLE_POINT_SIZE)
        font.setBold(True)
        self.setFont(font)

        # Both are assigned on BOTH branches, so `button.title_label` is not an
        # AttributeError on a body button while `button.subtitle_label` is a
        # clean None.
        self.title_label = None
        self.subtitle_label = None
        if not subtitle:
            # No summary: let the button draw its own text, exactly as before.
            # A body button is a name in the middle of a large button and
            # nothing about that needed changing.
            self.setText(title)
            return

        half = self.TEXT_PADDING_PX // 2
        box = QVBoxLayout(self)
        box.setContentsMargins(half, half, half, half)
        box.setSpacing(self.LABEL_SPACING_PX)
        self.title_label = QLabel(title)
        self.title_label.setFont(font)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignLeft
                                      | Qt.AlignmentFlag.AlignTop)
        self.subtitle_label = QLabel(subtitle)
        self.subtitle_label.setFont(self.subtitle_font())
        self.subtitle_label.setWordWrap(True)
        self.subtitle_label.setAlignment(Qt.AlignmentFlag.AlignLeft
                                         | Qt.AlignmentFlag.AlignTop)
        for label in (self.title_label, self.subtitle_label):
            # Without this the labels swallow the press and the button never
            # emits clicked() -- the flow stops dead at the run-type screen.
            label.setAttribute(
                Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
            label.setStyleSheet("background: transparent; border: none;")
            box.addWidget(label)
        box.addStretch()

    def subtitle_font(self):
        """The font the summary line is drawn in."""
        font = QFont()
        font.setPointSize(self.SUBTITLE_POINT_SIZE)
        font.setBold(False)
        return font

    # -- height, derived rather than assumed ---------------------------------

    def required_height(self, width=None):
        """The height this button's text needs at `width`, from font metrics.

        The same arithmetic the layout does, done ahead of it: the vertical
        margins, the title's line height, the spacing, and the height the
        WRAPPED summary occupies at the width the summary label will actually
        get. `width` defaults to the button's current width; before the first
        layout that is Qt's 640 px default, so a caller that needs the
        narrowest (and therefore tallest) case passes `minimumWidth()`.

        A button with no summary has no child layout at all -- Qt draws its own
        text -- so it gets the design floor and nothing more.
        """
        from PyQt6.QtCore import QRect

        if self.subtitle_label is None:
            return self.DESIGN_MIN_HEIGHT

        width = self.width() if width is None else int(width)
        inner = max(1, width - self.TEXT_PADDING_PX)
        metrics = QFontMetrics(self.subtitle_font())
        wrapped = metrics.boundingRect(
            QRect(0, 0, inner, 10_000),
            int(Qt.TextFlag.TextWordWrap), self._subtitle).height()
        title = QFontMetrics(self.font()).height()
        needed = (self.TEXT_PADDING_PX + title + self.LABEL_SPACING_PX
                  + wrapped)
        return max(self.DESIGN_MIN_HEIGHT, needed)

    def _apply_required_height(self):
        """Re-assert the derived minimum height.

        Called after anything that could have re-polished the widget. The
        stylesheet's `min-height` is applied by overwriting the widget's
        minimum size, so this must run AFTER that, not once in the constructor
        -- which is the whole defect.
        """
        if self._sizing:
            return
        self._sizing = True
        try:
            needed = self.required_height()
            if self.minimumHeight() != needed:
                self.setMinimumHeight(needed)
        finally:
            self._sizing = False

    def showEvent(self, event):
        super().showEvent(event)
        self._apply_required_height()

    def resizeEvent(self, event):
        # A wrapping label's height depends on its width, so a narrower button
        # is a TALLER one. Recomputing here is what makes the fit hold at every
        # width the window can take rather than only at the one it opened at.
        super().resizeEvent(event)
        self._apply_required_height()

    def changeEvent(self, event):
        from PyQt6.QtCore import QEvent

        super().changeEvent(event)
        if event.type() in (QEvent.Type.StyleChange, QEvent.Type.FontChange):
            self._apply_required_height()

    def text(self):
        """The button's TITLE.

        Qt's own text is empty on a button that has a summary, because the
        child labels do the drawing. Callers -- including
        `test_gui_body_first.py`, which reads this to check which run types are
        offered -- get the title either way.
        """
        return self._title


class BodyFirstPanel(BaseConfigTab):
    """The Start page: landing -> run type -> card.

    It also keeps the five task-card entry points the previous stage built.
    Four of them ARE these body cards now -- the same
    ``gui_core.presets.Preset`` objects supply each card's announced defaults,
    so the "a preset reports exactly what it wrote" contract is one mechanism
    rather than two -- and the fifth, "reproduce a published baseline", is an
    entry mode on the landing screen because it is not a body at all.
    """

    preset_applied = pyqtSignal(str)
    advanced_requested = pyqtSignal(str)
    project_replaced = pyqtSignal(str)
    #: (body_key, mode_key, run_type)
    route_changed = pyqtSignal(str, str, str)
    #: The user asked for the full eleven-section view.
    all_settings_requested = pyqtSignal()
    #: The user asked to open a configuration file from the landing screen.
    open_requested = pyqtSignal()

    LANDING, RUNTYPE, CARD = 0, 1, 2

    def init_ui(self):
        from gui_core.body_cards import body_buttons

        self._reports = {}
        self._route = (None, None, "time_series")
        #: The body of the last card actually opened, or None. Distinct from
        #: `_route[0]`, which `choose_body` moves to the DESTINATION while the
        #: user is still on the landing screen. See `open_card`.
        self._card_body = None
        self.cards = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.stack = QStackedWidget()
        outer.addWidget(self.stack)

        self.stack.addWidget(self._build_landing(body_buttons()))
        self.stack.addWidget(self._build_runtype())

        self.card_host = QWidget()
        host_layout = QVBoxLayout(self.card_host)
        host_layout.setContentsMargins(0, 0, 0, 0)
        host_layout.setSpacing(0)
        host_layout.addWidget(self._card_header())
        self.card_stack = QStackedWidget()
        host_layout.addWidget(self.card_stack, stretch=1)
        self.stack.addWidget(self.card_host)

        self.stack.setCurrentIndex(self.LANDING)

    # -- construction ---------------------------------------------------------

    def _build_landing(self, buttons):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Kept as an attribute so the register gate can read it by name rather
        # than fishing for a bold QLabel among the muted notes.
        self.landing_heading = QLabel("Select a body")
        self.landing_heading.setFont(
            QFont("Helvetica Neue", 15, QFont.Weight.Bold))
        layout.addWidget(self.landing_heading)

        # Its OWN message line. The first draft put "no card describes this
        # file" into the baseline entry's report box, which is a different
        # control answering a different question -- a message in the wrong
        # place is a message nobody connects to what they just did.
        self.landing_message = QLabel("")
        self.landing_message.setWordWrap(True)
        self.landing_message.setTextFormat(Qt.TextFormat.PlainText)
        self.landing_message.setStyleSheet(
            "color: #EF9A9A; font-size: 11px; padding: 4px; "
            "border: 1px solid #444; border-radius: 4px;")
        self.landing_message.setVisible(False)
        layout.addWidget(self.landing_message)

        # The body's NAME, and nothing else. The grid carried a paragraph of
        # prose under each button, then one functional line, then a two-word
        # maturity badge on Mercury; each read as commentary rather than as a
        # choice, and a body with something to disclose discloses it in the
        # button's tooltip and on its own card, where the controls it
        # constrains are. `BodyButton` has no `note`, `subtitle` or `maturity`
        # field any more, so this cannot regress by someone re-adding the text
        # to the table and finding it does not appear.
        grid = QGridLayout()
        grid.setSpacing(10)
        for index, button in enumerate(buttons):
            widget = _BigButton(button.title,
                                tooltip=get_tooltip("ui.body_button"))
            widget.clicked.connect(
                lambda _c=False, k=button.key: self.choose_body(k))
            grid.addWidget(widget, index // 3, index % 3)
        layout.addLayout(grid)

        rule = QFrame()
        rule.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(rule)

        row = QHBoxLayout()
        open_button = QPushButton("Open a configuration file…")
        open_button.setToolTip(get_tooltip("ui.landing_open"))
        open_button.clicked.connect(self.open_requested.emit)
        row.addWidget(open_button)

        baseline_button = QPushButton("Reproduce a published run…")
        baseline_button.setToolTip(get_tooltip("ui.landing_baseline"))
        baseline_button.clicked.connect(self._on_choose_baseline)
        row.addWidget(baseline_button)

        # Clearly secondary, and labeled as what it is. The flow is the way
        # through this application; the eleven-section view is the escape
        # hatch for someone who already knows which key they want.
        settings_button = QPushButton("All settings (advanced)…")
        settings_button.setObjectName("AllSettingsButton")
        settings_button.setToolTip(get_tooltip("ui.all_settings"))
        settings_button.clicked.connect(self.all_settings_requested.emit)
        row.addWidget(settings_button)
        row.addStretch()
        layout.addLayout(row)

        # The baseline entry keeps its own REPORT surface -- what it loaded,
        # which engine build ran it, and whether the environment is knowable --
        # because that is written after the user acts and is the whole point of
        # the entry.
        #
        # What it no longer keeps is the two paragraphs of standing prose that
        # sat here permanently: the preset's five-sentence summary and its
        # beaming line, naming config_used.yaml, provenance.txt and
        # engine_env:. On a landing screen whose job is "choose a body", that
        # is a wall of text in front of a choice. Both are now in the button's
        # own tooltip (`ui.landing_baseline`, which gained the environment
        # caveat verbatim), where a user goes when they want the detail.
        report = QLabel("")
        report.setWordWrap(True)
        report.setTextFormat(Qt.TextFormat.PlainText)
        report.setStyleSheet(
            "font-size: 10px; padding: 4px; border: 1px solid #444; "
            "border-radius: 4px;")
        report.setVisible(False)
        layout.addWidget(report)
        self._reports["reproduce_baseline"] = report

        layout.addStretch()
        return page

    def _build_runtype(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        back = QPushButton("← Bodies")
        back.setToolTip(get_tooltip("ui.back_to_bodies"))
        back.clicked.connect(lambda: self.stack.setCurrentIndex(self.LANDING))
        row = QHBoxLayout()
        row.addWidget(back)
        row.addStretch()
        layout.addLayout(row)

        self.runtype_heading = QLabel("")
        self.runtype_heading.setFont(
            QFont("Helvetica Neue", 14, QFont.Weight.Bold))
        layout.addWidget(self.runtype_heading)

        self.mode_group = QGroupBox("Model domain")
        self.mode_layout = QVBoxLayout()
        self.mode_group.setLayout(self.mode_layout)
        layout.addWidget(self.mode_group)
        self._mode_radios = []

        self.runtype_group = QGroupBox("Run type")
        self.runtype_layout = QVBoxLayout()
        self.runtype_group.setLayout(self.runtype_layout)
        layout.addWidget(self.runtype_group)
        self._runtype_buttons = []
        self._runtype_widgets = []

        layout.addStretch()
        scroll = QScrollArea()
        scroll.setWidget(page)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        return scroll

    def _card_header(self):
        header = QWidget()
        layout = QHBoxLayout(header)
        layout.setContentsMargins(12, 8, 12, 4)
        back = QPushButton("← Run type")
        back.setToolTip(get_tooltip("ui.back_to_runtype"))
        back.clicked.connect(lambda: self.stack.setCurrentIndex(self.RUNTYPE))
        layout.addWidget(back)
        self.card_title = QLabel("")
        self.card_title.setFont(QFont("Helvetica Neue", 13, QFont.Weight.Bold))
        layout.addWidget(self.card_title)
        layout.addStretch()
        settings = QPushButton("All settings (advanced)…")
        settings.setObjectName("AllSettingsButton")
        settings.setToolTip(get_tooltip("ui.all_settings"))
        settings.clicked.connect(self.all_settings_requested.emit)
        layout.addWidget(settings)
        return header

    # -- what the flow is currently on ----------------------------------------

    def route(self):
        """``(body_key, mode_key, run_type)`` -- may carry ``None`` for the
        first two while the user is still on the landing screen."""
        return self._route

    def active_card(self):
        """The card the flow is showing, or None if it is not on a card."""
        if self.stack.currentIndex() != self.CARD:
            return None
        return self.card_stack.currentWidget()

    def refresh_active_card(self):
        """Re-read the configuration into whichever card is live.

        Called when the user returns from All settings. A value edited on a
        section page is in the configuration but not on the card's controls,
        and `BaseConfigTab` writes a control's value back on the next widget
        event -- so a card left showing a stale number would push that stale
        number over the edit the moment anything else on the card was touched.

        Goes through `BodyCard.load_from_config`, which is `_loading`-guarded,
        so re-mirroring is not itself an edit: the same guard is what keeps
        merely DISPLAYING a configuration from firing the `toggled` handler
        that writes `radiation.rms_slope_deg`.
        """
        card = self.active_card()
        if card is None:
            return None
        card.load_from_config()
        return card

    def register_cards(self, cards):
        """Take the card widgets built by the ConfigurationPanel.

        They are built there, not here, so each one is registered in
        ``panel.tabs`` and therefore inside the tooltip-coverage, scalar-
        spelling and tier gates. A card the gates cannot see is a card that can
        ship bare.
        """
        self.cards = dict(cards)
        for card in self.cards.values():
            self.card_stack.addWidget(card)
            self._reports.update(self._card_report_keys(card))

    def _card_report_keys(self, card):
        from gui_core.body_cards import CARD_PRESETS
        out = {}
        for (route, _mode), preset in CARD_PRESETS().items():
            if route == card.route and card._report is not None:
                out[preset.key] = card._report
        return out

    # -- the flow --------------------------------------------------------------

    def choose_body(self, body_key):
        from gui_core.body_cards import modes_for, run_types_for
        self._route = (body_key, None, self._route[2])
        self.runtype_heading.setText(f"{body_key} — run configuration")

        for widget in self._mode_radios:
            widget.setParent(None)
        self._mode_radios = []
        modes = modes_for(body_key)
        for index, mode in enumerate(modes):
            radio = QRadioButton(f"{mode.title} — {mode.summary}")
            radio.setToolTip(get_tooltip("ui.body_mode"))
            radio.setChecked(index == 0)
            radio.toggled.connect(
                lambda checked, m=mode.key: checked and self._choose_mode(m))
            self.mode_layout.addWidget(radio)
            self._mode_radios.append(radio)
        self.mode_group.setVisible(len(modes) > 1)
        self._route = (body_key, modes[0].key, self._route[2])
        self._rebuild_runtypes(run_types_for(body_key, modes[0].key))
        self.stack.setCurrentIndex(self.RUNTYPE)

    def _choose_mode(self, mode_key):
        from gui_core.body_cards import run_types_for
        body_key = self._route[0]
        self._route = (body_key, mode_key, self._route[2])
        self._rebuild_runtypes(run_types_for(body_key, mode_key))

    def _rebuild_runtypes(self, run_types):
        for widget in self._runtype_widgets:
            widget.setParent(None)
        self._runtype_widgets = []
        self._runtype_buttons = []
        for run_type in run_types:
            button = _BigButton(run_type.title, run_type.summary,
                                tooltip=get_tooltip("ui.run_type"))
            button.clicked.connect(
                lambda _c=False, k=run_type.key: self.choose_run_type(k))
            produces = _muted("Produces: " + run_type.produces)
            for widget in (button, produces):
                self.runtype_layout.addWidget(widget)
                self._runtype_widgets.append(widget)
            self._runtype_buttons.append(button)

    def choose_run_type(self, run_type_key):
        body_key, mode_key, _ = self._route
        self.open_card(body_key, mode_key, run_type_key, apply_defaults=True)

    def open_card(self, body_key, mode_key, run_type_key,
                  apply_defaults=False):
        """Show one card, optionally applying and announcing its defaults."""
        from gui_core.body_cards import RUN_TYPES_BY_KEY, card_preset
        card = self.cards.get(body_key)
        if card is None:
            logger.warning("[body-first] no card for %r", body_key)
            return None
        for other in self.cards.values():
            if other is not card:
                other.deactivate()
        # Which body we are coming FROM is the only thing that distinguishes
        # "this card's own terrain" from "another body's terrain".
        #
        # It CANNOT be read from `self._route`: `choose_body` writes the new
        # body into `_route` on the landing screen, before this method is ever
        # called, so by now `_route[0]` is already the destination and every
        # transition would look like a same-body one. `_card_body` is the body
        # of the last card actually OPENED, which is the question being asked.
        previous_body = self._card_body
        self._route = (body_key, mode_key, run_type_key)

        result = None
        if apply_defaults:
            result = self._apply_route_defaults(body_key, mode_key,
                                                run_type_key, previous_body)
        card.activate(mode_key, run_type_key, seeding_defaults=apply_defaults)
        self.card_stack.setCurrentWidget(card)
        run_type = RUN_TYPES_BY_KEY[run_type_key]
        self.card_title.setText(f"{body_key} — {run_type.title}")
        self.stack.setCurrentIndex(self.CARD)
        preset = card_preset(body_key, mode_key)
        if result is not None and preset is not None:
            card.show_report("\n".join(result.summary_lines()))
            self.preset_applied.emit(preset.key)
        self._card_body = body_key
        self.route_changed.emit(body_key, mode_key or "", run_type_key)
        self.value_changed.emit()
        return card

    def _apply_route_defaults(self, body_key, mode_key, run_type_key,
                              previous_body=None):
        from gui_core.body_cards import RUN_TYPES_BY_KEY, card_preset
        from gui_core.presets import (FOREIGN_MESH_CLEARS, apply_preset,
                                      mirror_shape_mesh)
        preset = card_preset(body_key, mode_key)
        if preset is None:
            return None

        # -- another body's terrain, cleared on the body change ---------------
        # BEFORE the preset, so the preset's beaming rule measures what is
        # actually going to be used. Scoped to a CHANGE of body on purpose:
        # the lunar crater card and the Europa DEM card both write
        # mesh.import_file, mesh.latitude_override and
        # radiation.view_factor_file, so going from one to the other carried a
        # crater mesh, a lunar latitude and a view-factor matrix built for a
        # different mesh into the other body's run -- while re-opening the SAME
        # card must leave the mesh the user already chose alone.
        cleared = {}
        if previous_body is not None and previous_body != body_key:
            for path in FOREIGN_MESH_CLEARS:
                cleared.update(self.config_manager.reset_to_default(path))
            # A generic body's own shape model is not "another body's
            # terrain"; put it straight back. Same rule, same function, as the
            # one apply_preset uses.
            for key, value in mirror_shape_mesh(self.config_manager).items():
                cleared.pop(key, None)

        result = apply_preset(self.config_manager, preset, self._project_root())
        result.cleared.update(cleared)
        for key, value in RUN_TYPES_BY_KEY[run_type_key].sets.items():
            self.config_manager.set_value(key, value)
            result.applied[key] = value
        result.applied.update(
            self._apply_time_defaults(body_key, run_type_key, result))

        # A generic body's surface, from the engine's own defaults. HERE and
        # not in the preset: `presets.py`'s first rule is that it must not
        # report a key that would be silently dropped, and `_config_for_yaml`
        # emits `generic_body:` only once a body is actually active -- so the
        # preset cannot honestly claim these. The card can, because it is the
        # screen that shows them. Only on a NEW run (this method runs solely
        # under `apply_defaults=True`); landing on an opened configuration
        # leaves the author's file alone.
        card = self.cards.get(body_key)
        if card is not None and hasattr(card, "seed_engine_surface_defaults"):
            result.applied.update(card.seed_engine_surface_defaults())
        if preset.needs_advanced:
            result.notes.append(
                "The fields listed above are not on this card. 'Advanced "
                "options' has been turned on and the sections holding them "
                "have been opened for you -- press 'All settings (advanced)' "
                "at the top of this card to get to them.")
            self.advanced_requested.emit(preset.key)
        return result

    def _apply_time_defaults(self, body_key, run_type_key, result):
        """The window and the spin-up a NEW run starts from.

        Both are announced, both stay editable, and both are computed rather
        than typed -- see `gui_core.body_cards`, where the rules live so they
        can be measured without a window.

        * **The window is one diurnal cycle**, from
          ``ptm.body_table.solar_day_days``. That table is generated from
          ``planetary_properties.h`` and re-derived by a test on every run, so
          a period retuned in the C++ becomes a failing test rather than a card
          proposing a window the engine does not force at.
        * **The spin-up is the automatic value**, which is the part Paul asked
          for across all run types: the engine's own default is 1.0 cycle,
          below the two the equilibration paper calls a minimum.

        Only on a NEW run: this method is reached solely under
        ``apply_defaults=True``, so landing on an opened configuration leaves
        the author's numbers alone.
        """
        from gui_core.body_cards import (auto_equilibration_cycles,
                                         default_duration_days,
                                         samples_for_window)
        applied = {}

        days, why = default_duration_days(body_key,
                                          self.config_manager.get_config())
        if days is not None:
            samples = samples_for_window(days, days)
            if run_type_key == "integrated":
                # The integrated card's ONE control is the integration window,
                # so that is the key the default lands on; the other two are
                # derived from it exactly as an edit to the control would
                # derive them, and not left at the preset's values.
                self.config_manager.set_value("thermal.integration_days", days)
                applied["thermal.integration_days"] = days
            self.config_manager.set_value("simulation.duration_days", days)
            self.config_manager.set_value("simulation.timesteps", samples)
            applied["simulation.duration_days"] = days
            applied["simulation.timesteps"] = samples
            result.notes.append(f"Run window: {why}.")
        else:
            result.notes.append(
                f"The run window was left as it was: {why}.")

        cycles, reason = auto_equilibration_cycles(
            body_key, self.config_manager.get_config())
        self.config_manager.set_value("thermal.equilibration_orbits", cycles)
        applied["thermal.equilibration_orbits"] = cycles
        result.notes.append(f"Spin-up set automatically to {cycles:g} diurnal "
                            f"cycles: {reason}. It stays editable.")
        card = self.cards.get(body_key)
        if card is not None:
            card.announce_auto_equilibration(cycles, reason)
        return applied

    def _project_root(self):
        from gui_core.preflight import find_project_root
        from gui_core.simulation_runner import SimulationRunner
        try:
            root = find_project_root(SimulationRunner.find_ptm_executable())
        except Exception:                                    # noqa: BLE001
            root = None
        return root or Path.cwd()

    # -- the file the user opened (DP-7) --------------------------------------

    def route_for_config(self):
        """Where the current configuration belongs, per its own keys."""
        from gui_core.body_cards import detect_route
        return detect_route(self.config_manager.get_config())

    def land_on_current_config(self):
        """Land on the card the loaded configuration belongs to.

        Returns the `BodyRoute`. A refusal is REPORTED and the landing screen
        stays put -- the full settings view reads any configuration, and
        guessing a body would show a user another body's defaults for values
        their file never set.
        """
        route = self.route_for_config()
        if not route.ok:
            self.show_landing_message(
                route.refusal + " Everything in the file is loaded and "
                "nothing has been changed; open All settings to see it.")
            self.stack.setCurrentIndex(self.LANDING)
            return route
        self.show_landing_message("")
        card = self.open_card(route.body_key, route.mode_key, route.run_type,
                              apply_defaults=False)
        if card is not None:
            card.show_report(self._out_of_card_report(card))
        return route

    def _out_of_card_report(self, card):
        from gui_core.body_cards import out_of_card_settings
        extra = out_of_card_settings(self.config_manager.get_config(),
                                     card.exposed_keys())
        if not extra:
            return ("Every setting in this file is shown on this card.")
        lines = [f"This file sets {len(extra)} option(s) this card does not "
                 f"show. They are unchanged and will be written back as they "
                 f"are:"]
        for key, value, default in extra[:12]:
            if default is None:
                lines.append(f"    {key} = {value!r}   (not an engine key)")
            else:
                lines.append(f"    {key} = {value!r}   (default {default!r})")
        if len(extra) > 12:
            lines.append(f"    ... and {len(extra) - 12} more")
        lines.append("Open All settings to see and edit them.")
        return "\n".join(lines)

    def show_landing_message(self, text):
        self.landing_message.setText(text)
        self.landing_message.setVisible(bool(text))

    def landing_message_text(self):
        """What the landing screen is currently saying (for probes and tests)."""
        return self.landing_message.text()

    # -- the task-card API the previous stage's gates pin ----------------------

    def apply_preset(self, key):
        """Apply one named preset and show what it wrote.

        Kept because it is the mechanism the body cards use, not for
        compatibility's sake: a card's announced defaults ARE a preset, and
        routing both through one function is why "a preset reports exactly what
        it set" can be one test rather than two.
        """
        from gui_core.body_cards import CARD_PRESETS
        from gui_core.presets import PRESETS_BY_KEY, apply_preset
        preset = PRESETS_BY_KEY.get(key)
        if preset is None:
            for candidate in CARD_PRESETS().values():
                if candidate.key == key:
                    preset = candidate
                    break
        if preset is None:
            raise KeyError(f"no preset or card named {key!r}")
        result = apply_preset(self.config_manager, preset, self._project_root())
        lines = result.summary_lines()
        if preset.needs_advanced:
            lines.append(
                "NOTE: the fields listed above are not on this card. "
                "'Advanced options' has been turned on and the sections "
                "holding them have been opened for you -- press 'All settings "
                "(advanced)' to get to them.")
            self.advanced_requested.emit(key)
        self._show(key, "\n".join(lines))
        self.preset_applied.emit(key)
        self.value_changed.emit()
        return result

    def apply_baseline(self, run_dir):
        """Load a finished run's own config_used.yaml, and say what ran it."""
        from gui_core.preflight import executable_fingerprint
        from gui_core.presets import compare_binary, read_baseline
        from gui_core.simulation_runner import SimulationRunner
        try:
            config, provenance, config_path = read_baseline(run_dir)
        except Exception as exc:                             # noqa: BLE001
            self._show("reproduce_baseline",
                       f"Cannot use this directory: {exc}")
            return None
        try:
            self.config_manager.load_from_yaml(str(config_path))
        except Exception as exc:                             # noqa: BLE001
            self._show("reproduce_baseline",
                       f"{config_path} could not be loaded: {exc}")
            return None
        try:
            exe = SimulationRunner.find_ptm_executable()
            current = executable_fingerprint(exe)[0] if exe else ""
        except Exception:                                    # noqa: BLE001
            current = ""
        lines = [f"Loaded {config_path}", compare_binary(provenance, current)]
        for name in ("engine_git_sha", "run_started_utc", "body", "mesh_faces"):
            if name in provenance:
                lines.append(f"    {name} = {provenance[name]}")
        lines.append(
            "Every key in that file -- spin-up seed, beaming, coupling mode -- "
            "is now loaded as the baseline recorded it.")
        if isinstance(config.get("engine_env"), dict):
            lines.append(
                "Its engine_env: block was carried over exactly as written "
                "(%d variable(s)); the GUI's own set is NOT applied."
                % len(config["engine_env"]))
        else:
            lines.append(
                "WARNING: this baseline's config carries no engine_env: block, "
                "so the environment it ran under is NOT recorded anywhere -- "
                "provenance.txt does not list PTM_* variables either. This run "
                "will use the GUI's current set, which may differ from the "
                "baseline's. Check the Engine Flags page, and the baseline's "
                "own run log if you still have it.")
        lines.append(
            "This is NOT your project file. Use Save As to write it somewhere "
            "of your own; the baseline's own record is left untouched.")
        self._show("reproduce_baseline", "\n".join(lines))
        self.project_replaced.emit(str(config_path))
        self.preset_applied.emit("reproduce_baseline")
        self.value_changed.emit()
        return provenance

    def _on_choose_baseline(self):
        directory = QFileDialog.getExistingDirectory(
            self, "Select the baseline run's output directory", "",
            QFileDialog.Option.ShowDirsOnly)
        if not directory:
            return
        self.apply_baseline(directory)

    def _show(self, key, text):
        report = self._reports.get(key)
        if report is None:
            return
        report.setText(text)
        report.setVisible(True)

    def report_text(self, key):
        report = self._reports.get(key)
        return report.text() if report is not None else ""

    def load_from_config(self):
        # The flow does not mirror the configuration; the ACTIVE card does.
        pass
