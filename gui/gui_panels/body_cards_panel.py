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

from PyQt6.QtCore import Qt, QPoint, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QFont, QFontMetrics
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFrame, QGridLayout,
    QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMessageBox,
    QPushButton, QRadioButton, QScrollArea, QSpinBox, QStackedWidget,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from gui_core.body_cards import (GENERIC_MESH_SHAPE, WINDOW_UNIT_DAYS,
                                 WINDOW_UNIT_DIURNAL, WINDOW_UNIT_ORBITAL,
                                 WINDOW_UNITS, window_unit_days)
from gui_core.config_manager import ConfigManager
from gui_core.curated_sites import (curated_site, curated_sites_for_body,
                                    local_registry_status)
from gui_panels.config_panel import (BaseConfigTab, CollapsibleSection,
                                     get_tooltip)

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


def _flux_mode_aliases():
    """`gui_core.flux_cubes.MODE_ALIASES`, resolved at class-body time.

    A function rather than a module-level import so this file keeps its
    existing import graph: `gui_core.flux_cubes` reaches back into
    `gui_core.preflight` and `gui_core.simulation_runner`, and importing it at
    the top of a panel that `gui_core` itself pulls in is a cycle waiting to
    be discovered by whoever adds the next import.
    """
    from gui_core.flux_cubes import MODE_ALIASES
    return dict(MODE_ALIASES)


def _same_setting(current, wanted):
    """Do these two configuration values mean the same thing?

    Not ``str(a) == str(b)``. YAML reads ``latitude_override: -999`` as an int
    and ``enabled: yes`` as a bool, so the string forms of two IDENTICAL
    settings differ (``'-999'`` against ``'-999.0'``) and a card comparing them
    that way shouts that a configuration disagrees with a set it matches
    exactly. Booleans compare by truth, numbers by value, everything else by
    its string form -- which for the paths this is used on is the right test.
    """
    if isinstance(wanted, bool) or isinstance(current, bool):
        return bool(current) == bool(wanted)
    if isinstance(wanted, (int, float)):
        try:
            return float(current) == float(wanted)
        except (TypeError, ValueError):
            return False
    return str(current) == str(wanted)


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
    #: so the same configuration read
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
        control now OPENS in diurnal periods.
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

    # -- a long run, broken into windows (the polar-site pattern) ------------------

    #: Windows a new series starts with. The polar landing-site campaign's own production year set is 4
    #: years x 2 seasons; four is a set somebody would actually look at.
    DEFAULT_WINDOW_COUNT = 4

    def _build_window_series_rows(self, grid, row):
        """The split-into-windows controls, under the integration window.

        THE SPAN IS NOT A CONFIGURATION KEY, and that is deliberate. The engine
        has no "span" -- a window series is N separate runs, each with its own
        `simulation.start_time` and `simulation.duration_days` -- so binding a
        span control to a dotted key would write a key into the emitted YAML
        that nothing reads, which is the unread-config-key defect. It lives on the
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
    #: effect" warning at parse time. That is the unread-config-key defect, and it is why the
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
        emitted YAML that nothing reads -- the very unread-config-key defect.
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
        row += 1
        row = self._build_vf_discovery_rows(grid, row)
        group.setLayout(grid)
        self.widgets["radiation.enabled"].toggled.connect(
            self._on_radiation_toggled)
        return group

    # -- finding a matrix that is already here -------------------------------

    def _build_vf_discovery_rows(self, grid, row):
        """The "there may already be one of these next to your mesh" row.

        OFFER, NEVER IMPOSE. Nothing here writes a key until the user presses
        Use. The reason is not caution for its own sake: a view-factor matrix
        does not carry the mesh it was built for, so "the one in this folder"
        is a guess, and a wrong guess produces a run that completes. The only
        thing that runs automatically is a COUNT.
        """
        self.vf_find_button = QPushButton("Look for matching view factors")
        self.vf_find_button.setToolTip(get_tooltip("ui.vf_discovery"))
        self.vf_find_button.clicked.connect(self._on_find_view_factors)
        self.vf_candidate_combo = QComboBox()
        self.vf_candidate_combo.setToolTip(get_tooltip("ui.vf_discovery"))
        self.vf_candidate_combo.setEnabled(False)
        self.vf_candidate_combo.currentIndexChanged.connect(
            lambda _i: self._describe_vf_candidate())
        self.vf_use_button = QPushButton("Use this file")
        self.vf_use_button.setEnabled(False)
        self.vf_use_button.clicked.connect(self._on_use_view_factor)

        strip = QWidget()
        strip_layout = QHBoxLayout(strip)
        strip_layout.setContentsMargins(0, 0, 0, 0)
        strip_layout.addWidget(self.vf_find_button)
        strip_layout.addWidget(self.vf_candidate_combo, 1)
        strip_layout.addWidget(self.vf_use_button)
        grid.addWidget(strip, row, 0, 1, 2)
        row += 1
        self.vf_discovery_note = _muted("")
        grid.addWidget(self.vf_discovery_note, row, 0, 1, 2)
        row += 1
        #: The last scan's candidates, in combo order after its "choose one"
        #: row, and the scan's own summary + refusals, which stay on screen
        #: while a candidate is highlighted rather than being replaced by it.
        self._vf_candidates = []
        self._vf_scan_lines = []
        return row

    def _mesh_for_discovery(self):
        return str((self.config_manager.get_config().get("mesh") or {})
                   .get("import_file", "") or "").strip()

    def refresh_vf_discovery(self, scan=False):
        """Re-state what is next to the mesh. Cheap unless ``scan`` is set.

        The automatic pass says only HOW MANY files were found and does not
        rank or offer them, because it runs whenever the mesh path changes --
        including on every keystroke of a half-typed path. The ranking, which
        reads headers, happens when the user asks for it.
        """
        combo = getattr(self, "vf_candidate_combo", None)
        note = getattr(self, "vf_discovery_note", None)
        if combo is None or note is None:
            return
        mesh = self._mesh_for_discovery()
        if not scan:
            self._vf_candidates = []
            self._vf_scan_lines = []
            combo.blockSignals(True)
            combo.clear()
            combo.blockSignals(False)
            combo.setEnabled(False)
            self.vf_use_button.setEnabled(False)
            note.setText(
                "" if not mesh else
                "A view-factor matrix for this mesh may already be on disk "
                "beside it. Nothing is chosen for you: press the button to "
                "list what is there and what each file actually is.")
            return
        if not mesh:
            note.setText("Choose a terrain mesh first — the search looks in "
                         "the mesh's own folder.")
            return
        try:
            from gui_core.vf_discovery import discover_view_factors
            result = discover_view_factors(
                mesh,
                view_factor_file=str(self.config_manager.get_value(
                    "radiation.view_factor_file", "") or ""),
                project_root=self._project_root())
        except Exception as exc:                             # noqa: BLE001
            # This runs from a Qt slot. An exception out of a slot is qFatal
            # and the loss of unsaved edits, and "the search failed" is not a
            # reason to lose a configuration.
            logger.warning("[card] the view-factor search failed: %s", exc,
                           exc_info=True)
            note.setText(f"The search could not run ({type(exc).__name__}: "
                         f"{exc}). Nothing was changed.")
            return
        self._vf_candidates = list(result.matches)
        combo.blockSignals(True)
        combo.clear()
        # NOTHING IS PRE-SELECTED. A combo whose first row is already chosen
        # makes "Use this file" a one-click wiring of whatever happened to sort
        # first, and sort order is not evidence -- measured in the real
        # lunar polar site A directory, where two matrices reach the same tier and the
        # superseded one has the alphabetically earlier name.
        combo.addItem(f"({len(self._vf_candidates)} found — choose one)"
                      if self._vf_candidates else "(none found)", None)
        for candidate in self._vf_candidates:
            combo.addItem(candidate.headline())
        combo.blockSignals(False)
        combo.setEnabled(bool(self._vf_candidates))
        self.vf_use_button.setEnabled(False)
        self._vf_scan_lines = [result.summary()]
        # BOUNDED. Pointing the cache field at a directory of per-site base
        # stubs produces 58 refusals and 12,000 characters, and that is the
        # ordinary case after a curated site is chosen, not a pathological one.
        # The count is always exact; the list is what gets cut.
        shown = result.refusals[:self.VF_REFUSALS_SHOWN]
        for candidate in shown:
            self._vf_scan_lines.append(f"REFUSED — {candidate.path.name}: "
                                       f"{candidate.refusal}")
        hidden = len(result.refusals) - len(shown)
        if hidden > 0:
            self._vf_scan_lines.append(
                f"...and {hidden} more refused, not listed here. The console "
                f"log carries every one.")
            for candidate in result.refusals[len(shown):]:
                logger.info("[card] view-factor candidate refused — %s: %s",
                            candidate.path, candidate.refusal)
        self._describe_vf_candidate()

    #: How many refusals go on the card before the rest go to the log only.
    VF_REFUSALS_SHOWN = 6

    def _on_find_view_factors(self):
        self.refresh_vf_discovery(scan=True)

    def _current_vf_candidate(self):
        """The chosen candidate, or None. Index 0 is the 'choose one' row."""
        index = self.vf_candidate_combo.currentIndex() - 1
        if 0 <= index < len(self._vf_candidates):
            return self._vf_candidates[index]
        return None

    def _describe_vf_candidate(self):
        """Show the scan's own result AND whatever is highlighted, together.

        The two used to be written to the same label in sequence, so the scan
        summary and every refusal were overwritten by the selected candidate's
        notes the moment there was one. That silenced the refusals in the
        common case -- which is the case they exist for.
        """
        note = getattr(self, "vf_discovery_note", None)
        if note is None:
            return
        lines = list(getattr(self, "_vf_scan_lines", []))
        candidate = self._current_vf_candidate()
        self.vf_use_button.setEnabled(candidate is not None)
        if candidate is not None:
            lines.append("")
            lines.append(candidate.headline())
            lines.extend(candidate.notes)
        note.setText("\n".join(lines))

    def _on_use_view_factor(self):
        """Wire the highlighted candidate, and say exactly what was written."""
        candidate = self._current_vf_candidate()
        if candidate is None:
            return
        from gui_core.vf_discovery import FORMAT_CSR, candidate_wiring
        wiring = candidate_wiring(candidate)
        if not wiring:
            self.vf_discovery_note.setText(
                f"{candidate.path.name} was not wired: {candidate.refusal}")
            return
        for key, value in wiring:
            self.config_manager.set_value(key, value)
        written = "; ".join(f"{k} = {v}" for k, v in wiring)
        lines = [f"Wired: {written}"]
        if candidate.fmt == FORMAT_CSR:
            lines.append(
                "A replacement CSR is loaded through the environment, not "
                "through the cache field, and the run ALSO needs a "
                "correctly-dimensioned base file in View-factor cache or the "
                "engine will rebuild the matrix it was just handed. This "
                "wiring did not invent one; set it yourself, or choose a "
                "curated site, which brings its own.")
        if candidate.superseded:
            lines.append("THIS FILE IS SUPERSEDED and you chose it anyway, "
                         "which is a legitimate thing to do — reproducing an "
                         "older run needs the older matrix. " +
                         candidate.superseded)
        lines.extend(candidate.notes)
        # BEFORE the note, not after. `load_from_config` reaches
        # `BodyCard.load_card`, which resets this label to the idle text --
        # measured: the whole confirmation, INCLUDING the missing-base-file
        # warning above, vanished the instant it was written, and nothing else
        # in this project warns about a replacement CSR with no base.
        self.load_from_config()
        self._vf_scan_lines = lines
        self.vf_discovery_note.setText("\n".join(lines))
        self.value_changed.emit()

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

    # -- flux cubes ----------------------------------------------------------

    #: Columns of the named-locations table, in order, as (heading, row key).
    #: Height and cube size are optional per point; blank means "use the
    #: automatic value", which is what a negative number means in the layout
    #: file and what leaving the field out of the configuration means.
    LOCATION_COLUMNS = (("Name", "name"), ("Latitude", "lat_deg"),
                        ("Longitude", "lon_deg"), ("Height (m)", "height_m"),
                        ("Cube size (m)", "edge_m"))

    #: What the group says while the cubes are OFF AND the run could produce
    #: them. The switch is the only control on screen then, so this line has to
    #: say that there IS an options list and name what is in it -- in the same
    #: words the controls use, so someone looking for "grid spacing" finds
    #: "Distance between cubes".
    #:
    #: It names THE CUBES rather than saying "turn this on", and it is not
    #: shown while a blocker stands. `engine_support`'s own text already ends
    #: "Turn it on above" about the RADIATION switch, and the two sentences
    #: side by side put an "it" and a "this" one clause apart pointing at
    #: different controls -- while inviting the user into the exact state the
    #: launch check refuses.
    FLUX_OFF_HINT = ("Turn the cubes on for the placement options: where they "
                     "go (a grid, named points, or both), the distance "
                     "between them, the grid radius, the cube height measured "
                     "from this mesh, and what the run writes.")

    def flux_cube_group(self, title="Flux cubes"):
        """Virtual six-face irradiance probes on the terrain.

        VISIBLE WHEN ENABLED, not hidden behind the Advanced tier, and that is
        a decision with a reason. Everything except the one switch lives inside
        a container that is shown only while the switch is on, so the Basic
        tier gains exactly one checkbox. Putting the whole group behind the
        tier would mean an opened configuration with `flux_cubes.enabled: true`
        showed nothing at all -- which is the same argument `radiation_group`
        makes for keeping the radiation switch on every card, and it is how a
        setting stays on without anybody seeing it.

        THE GROUP IS ON THE CARD IN EVERY MODE. Only a local-mesh run can
        actually produce cubes -- the layout tool takes a mesh file and a
        global run generates its sphere in memory -- but that is said in the
        note, not enforced by taking the group off the screen. Hiding it was
        the second half of Paul's report (2026-09-01, "we're still not getting
        the options menu for the flux cubes"): on a Moon GLOBAL card the whole
        group was `setVisible(False)`, so between Radiation exchange and
        Surface properties there was no checkbox, no note and nothing naming
        the mode as the reason -- measured, and five of the eight
        still-existing files in his own recent-files list open exactly that
        way. `flux_cubes.engine_support` had computed the refusal and the card
        had put it in `flux_note`; the note was then hidden along with its
        group. A feature that is absent without a word is indistinguishable
        from one that was never built.

        THE SWITCH ITSELF IS ALWAYS LIVE, whatever `engine_support` says --
        see `refresh_flux_cubes`. One click always reveals the placement, from
        any state, whether the terrain arrived from the curated-site picker or
        from the file field and in either order. A refusal is a sentence in
        the note and an error in the launch check, never a control that
        discards the click.

        THE PLACEMENT IS THE SPECIFICATION. Nothing in the flow asks for a file
        name: the launch check turns "a 25 m grid out to 500 m" into a
        deterministic path beside the mesh, reuses a layout that answers exactly
        that placement, and otherwise builds one before the run starts. The
        layout-file field is an ADVANCED escape hatch for a layout somebody
        else built, and it is collapsed.
        """
        from gui_core import flux_cubes as fc

        group = QGroupBox(title)
        outer = QVBoxLayout()
        outer.setContentsMargins(8, 8, 8, 8)

        head = QGridLayout()
        head.setColumnStretch(1, 1)
        self.add_bool_field(head, 0, "Record irradiance at virtual "
                                     "measurement cubes", "flux_cubes.enabled")
        self.flux_note = _muted("")
        head.addWidget(self.flux_note, 1, 0, 1, 2)
        outer.addLayout(head)

        # -- everything below is shown only while the switch is on -----------
        self.flux_body = QWidget()
        grid = QGridLayout(self.flux_body)
        grid.setContentsMargins(0, 4, 0, 0)
        grid.setColumnStretch(1, 1)
        row = 0

        self.add_combo_field(grid, row, "Cube placement:",
                             "flux_cubes.mode", list(fc.MODES))
        row += 1

        # Grid placement -------------------------------------------------
        self.flux_lattice = QWidget()
        lattice = QGridLayout(self.flux_lattice)
        lattice.setContentsMargins(0, 0, 0, 0)
        lattice.setColumnStretch(1, 1)
        self.add_double_field(lattice, 0, "Distance between cubes (m):",
                              "flux_cubes.spacing_m", 0.5, 10000.0,
                              decimals=2)
        # RANGES, not decoration. The cube count is counted, not approximated,
        # so the work is proportional to radius/spacing; these two bounds cap
        # that at about a hundred thousand columns, which is milliseconds. They
        # are also the range a site mesh lives in -- a 50 km grid on a 3 km
        # core would put most of its cubes outside the mesh, and the tool
        # aborts on a lattice point with no terrain under it.
        self.add_double_field(lattice, 1,
                              "Grid radius around the mesh center (m):",
                              "flux_cubes.radius_m", 0.0, 50000.0,
                              decimals=2)
        grid.addWidget(self.flux_lattice, row, 0, 1, 2)
        row += 1

        # Named locations -------------------------------------------------
        self.flux_named = QWidget()
        named = QVBoxLayout(self.flux_named)
        named.setContentsMargins(0, 0, 0, 0)
        self.flux_table = QTableWidget(0, len(self.LOCATION_COLUMNS))
        self.flux_table.setHorizontalHeaderLabels(
            [heading for heading, _key in self.LOCATION_COLUMNS])
        self.flux_table.setToolTip(get_tooltip("ui.flux_cubes.locations"))
        self.flux_table.verticalHeader().setVisible(False)
        self.flux_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self.flux_table.setMinimumHeight(120)
        self.flux_table.itemChanged.connect(self._on_location_edited)
        named.addWidget(self.flux_table)
        buttons = QHBoxLayout()
        add_row = QPushButton("Add point")
        add_row.setToolTip(get_tooltip("ui.flux_cubes.locations"))
        add_row.clicked.connect(self._on_add_location)
        remove_row = QPushButton("Remove selected")
        remove_row.setToolTip(get_tooltip("ui.flux_cubes.locations"))
        remove_row.clicked.connect(self._on_remove_location)
        import_csv = QPushButton("Import from file...")
        import_csv.setToolTip(get_tooltip("ui.flux_cubes.import_csv"))
        import_csv.clicked.connect(self._on_import_locations)
        for button in (add_row, remove_row, import_csv):
            buttons.addWidget(button)
        buttons.addStretch()
        named.addLayout(buttons)
        grid.addWidget(self.flux_named, row, 0, 1, 2)
        row += 1

        # Height, read only -----------------------------------------------
        height_label = QLabel("Cube height and size:")
        height_label.setToolTip(get_tooltip("ui.flux_cubes.height"))
        self.flux_height = _muted("")
        self.flux_height.setToolTip(get_tooltip("ui.flux_cubes.height"))
        grid.addWidget(height_label, row, 0)
        grid.addWidget(self.flux_height, row, 1)
        row += 1

        # What will happen to the layout when this run starts --------------
        # THE PLACEMENT ABOVE IS THE SPECIFICATION. The `.fcg` the geometry
        # tool writes is the file it gets stored in, and nothing above this
        # line asks anybody to name one: the launch check works out where the
        # layout belongs, reuses one that answers this exact placement, and
        # otherwise builds it before the run starts. This label says which of
        # those three is about to happen, by name.
        self.flux_status = _muted("")
        self.flux_status.setToolTip(get_tooltip("ui.flux_cubes.status"))
        grid.addWidget(self.flux_status, row, 0, 1, 2)
        row += 1

        # The layout file, for a layout somebody else built ----------------
        # ADVANCED, and collapsed, because naming a file is no longer part of
        # the flow: it is the escape hatch for a pre-built or hand-made layout.
        # It is a COLLAPSIBLE SECTION rather than a tier-gated one, so it is
        # visible on the card whatever the tier, and an opened project that
        # names a layout file is one click from showing which -- the same
        # argument the group's own switch makes for staying out of the tier.
        self.flux_advanced = CollapsibleSection("Advanced: use a layout file "
                                                "somebody else built",
                                                collapsed=True)
        advanced_body = QWidget()
        advanced = QGridLayout(advanced_body)
        advanced.setContentsMargins(0, 0, 0, 0)
        advanced.setColumnStretch(1, 1)
        self.add_file_field(advanced, 0, "Cube layout file:",
                            "flux_cubes.cache_file",
                            "Cube layouts (*.fcg);;All Files (*)")
        self.flux_build = QPushButton("Build a layout now from the placement "
                                      "above")
        self.flux_build.setToolTip(get_tooltip("ui.flux_cubes.build"))
        self.flux_build.clicked.connect(self._on_build_layout)
        advanced.addWidget(self.flux_build, 1, 1)
        self.flux_advanced_note = _muted(
            "Leave this empty and the layout is built for you when the run "
            "starts, from the placement above. Set it and this exact file is "
            "used instead -- the run stops if it does not match the mesh.")
        advanced.addWidget(self.flux_advanced_note, 2, 0, 1, 2)
        self.flux_advanced.content_layout.addWidget(advanced_body)
        grid.addWidget(self.flux_advanced, row, 0, 1, 2)
        row += 1

        # Products ----------------------------------------------------------
        self.add_bool_field(grid, row, "Write the single binary flux file",
                            "flux_cubes.write_binary")
        row += 1
        self.add_bool_field(grid, row, "Write one spreadsheet file per cube",
                            "flux_cubes.write_csv")
        row += 1
        self.add_int_field(grid, row, "Spreadsheet files, at most:",
                           "flux_cubes.write_csv_max_cubes", 0, 1000000)
        row += 1
        # A COMBO, not a spin box. The engine branches on `>= 5` and records
        # `rays >= 5 ? 5 : 1`, so 2, 3 and 4 are silently collapsed to one ray
        # -- a spin box offering them would be three settings that are not.
        # An opened file carrying one of them still shows it, because
        # `set_enum` adds the foreign value and warns rather than displaying
        # something else.
        self.add_combo_field(grid, row, "Sun-visibility test:",
                             "flux_cubes.shadow_probe_rays",
                             [str(n) for n in fc.SHADOW_PROBE_CHOICES])
        row += 1
        self.flux_estimate = _muted("")
        grid.addWidget(self.flux_estimate, row, 0, 1, 2)

        outer.addWidget(self.flux_body)
        group.setLayout(outer)

        # ONE table, so a control added above without a signal is a missing
        # ROW here rather than an omission nobody can see: everything in this
        # group is described from the live configuration, and a control that
        # changes an answer and does not re-ask for it leaves the previous
        # answer on screen.
        # ⛔ `stateChanged`, NOT `toggled`, FOR THE ENABLE SWITCH -- and that
        # is the whole ordering argument, not a style choice.
        #
        # `refresh_flux_cubes` renders `flux_cubes.enabled` OUT of the
        # configuration, and the configuration is written by
        # `add_bool_field`'s slot on `stateChanged`. Listening on `toggled`
        # made "does the write happen first?" a question about the order Qt
        # emits TWO DIFFERENT SIGNALS, which is an implementation detail of
        # `QCheckBox::nextCheckState` and guaranteed by nothing. Listening on
        # the SAME signal, connected after, is a documented Qt guarantee:
        # slots of one signal run in connection order. Measured on a bare
        # `QCheckBox` here: `stateChanged` then `toggled` -- so round 2 was
        # right by luck, and a build where those flipped would render every
        # click one behind, which is exactly "the options appear after
        # checking and then unchecking" (Paul, 2026-09-01).
        for key, signal in (("flux_cubes.enabled", "stateChanged"),
                            ("flux_cubes.mode", "currentTextChanged"),
                            ("flux_cubes.cache_file", "textChanged"),
                            ("flux_cubes.spacing_m", "valueChanged"),
                            ("flux_cubes.radius_m", "valueChanged"),
                            ("flux_cubes.write_binary", "toggled"),
                            ("flux_cubes.write_csv", "toggled"),
                            ("flux_cubes.write_csv_max_cubes", "valueChanged"),
                            ("flux_cubes.shadow_probe_rays",
                             "currentTextChanged")):
            getattr(self.widgets[key], signal).connect(
                lambda _v: self.refresh_flux_cubes())

        # ⛔ AND THE REVEAL HAS TO REACH THE SCREEN. Ticking the box adds
        # ~300 px of controls BELOW the box; on this card that lands outside
        # the scroll viewport and the window does not move.
        #
        # On `stateChanged` as well, and connected THIRD, so the whole chain
        # of one click -- write the configuration, re-describe the group,
        # scroll it into view -- is three slots of ONE signal in connection
        # order. Nothing here depends on how Qt orders `toggled` against
        # `stateChanged`. `stateChanged` carries an int, so the reveal is
        # given a bool.
        self.widgets["flux_cubes.enabled"].stateChanged.connect(
            lambda state: self._reveal_flux_cubes(bool(state)))

        # -- the record of what a click does (gui_core/flux_trace.py) --------
        # Connected LAST on `stateChanged`, so it reports the state after the
        # three slots above have run, and separately on `toggled`, which
        # nothing in this group renders from any more -- a `toggled` line in
        # the log is proof the signal still arrives, and its ORDER against
        # `stateChanged` is visible in the timestamps.
        #
        # Only when the record is being kept -- which, since 2026-09-02, is
        # NOT the default. Without `PTM_GUI_TRACE=1` neither observer is
        # connected at all, so the group costs nothing and
        # `receivers(toggled)` is zero, which is what
        # `test_the_click_chain_is_ordered_by_a_guarantee_not_by_luck` reads
        # on the ordinary path.
        from gui_core import flux_trace
        if flux_trace.enabled():
            switch = self.widgets["flux_cubes.enabled"]
            switch.stateChanged.connect(
                lambda state: self._trace_switch("stateChanged", state))
            switch.toggled.connect(
                lambda on: self._trace_switch("toggled", on))
        return group

    def _trace_switch(self, signal_name, argument):
        """Write down the click, and look again half a second later.

        The delayed look is the part that matters for "the box does not stay
        ticked": it reports the checkbox and every option row again, after
        anything else in the program has had a chance to change them.
        """
        from gui_core import flux_trace
        if not flux_trace.enabled():
            return
        switch = self.widgets.get("flux_cubes.enabled")
        flux_trace.note(
            "checkbox-signal", signal=signal_name, argument=argument,
            checkbox_says=None if switch is None else switch.isChecked(),
            loading=self._loading)
        flux_trace.snapshot(self, "at-click")
        QTimer.singleShot(500, lambda: flux_trace.snapshot(self, "500ms-later"))

    def _reveal_flux_cubes(self, on):
        """Put the placement options on the screen when they are switched on.

        Only on the USER's toggle: `_loading` is set while a file is being
        mirrored into the controls, and scrolling a card the user has not
        touched to a group they did not ask about would move the form under
        them on every open.
        """
        if not on or self._loading:
            return
        # ⛔ IMMEDIATELY, AND AGAIN AFTER THE LAYOUT -- and the FIRST one has
        # to be correct on its own.
        #
        # Round 2 deferred this with `singleShot(0)` because `height()` is
        # still the collapsed value inside the signal. That made the scroll
        # depend on whether Qt's layout pass beats a zero timer, which is an
        # event-loop race: a harness that pumps `processEvents` wins it every
        # time and a real loop need not. `scroll_into_view` now reads
        # `max(height, sizeHint)`, so the immediate call is already right and
        # the deferred one is an idempotent second opinion -- it returns early
        # when the group is whole on screen, which after the first call it is.
        # Belt and braces, with the belt doing the work.
        #
        # The WHOLE group, not just `flux_body`: the note above the placement
        # says whether this run can produce cubes at all, and a reveal that
        # scrolls the refusal off the top answers one question by hiding
        # the other.
        self.scroll_into_view(getattr(self, "flux_group", None))
        QTimer.singleShot(
            0, lambda: self.scroll_into_view(getattr(self, "flux_group", None)))

    def refresh_flux_cubes(self):
        """Re-gate and re-describe the whole group. Cheap and idempotent.

        Called on load and on every edit that can change an answer, so the
        three lines on screen -- what the engine will accept, how big the cube
        is, and what the run will write -- are computed from the live
        configuration rather than left over from the last one.

        It NEVER writes a configuration key. The switch is left exactly as the
        file set it even when the rules refuse it; the group says the rules
        disagree, because opening a file is not an edit to it.
        """
        from gui_core import flux_cubes as fc
        switch = self.widgets.get("flux_cubes.enabled")
        note = getattr(self, "flux_note", None)
        if switch is None or note is None:
            return
        config = self.config_manager.get_config()
        policy = fc.engine_support(config)
        enabled = self._flux_enabled()
        from gui_core import flux_trace
        flux_trace.note_with_caller(
            "describe-group", rendering_value=enabled,
            checkbox_says=switch.isChecked(), loading=self._loading,
            engine_allows=policy.allowed)
        # ⛔ THE SWITCH IS NEVER DISABLED, and that is the fix for Paul's
        # 2026-08-31 report, not a style preference. `setEnabled(policy.allowed)`
        # made a real click a SILENT no-op: on a freshly opened Moon
        # local-mesh card there is no terrain yet, so the switch was dead, and
        # `QCheckBox.click()` on it changes nothing and emits nothing. The user
        # clicks, no option appears, and the only thing that ever said why is a
        # gray line under the box. Measured: the group is 79 px of one checkbox
        # in that state, whichever of the four preconditions is missing.
        #
        # It also produced a setting that was ON and could not be turned OFF --
        # go to another body and back and `presets.FOREIGN_MESH_CLEARS` takes
        # the mesh away while `flux_cubes.enabled` stays true, so the group
        # showed its own live setting behind a dead control.
        #
        # Nothing unsafe is reachable this way: `flux_cubes.preflight_findings`
        # turns every one of these blockers into a launch ERROR, so a run that
        # the engine would refuse after the whole thermal solve is refused
        # before it starts. The card describes; the launch check enforces.
        switch.setEnabled(True)
        body = getattr(self, "flux_body", None)
        if body is not None:
            flux_trace.note_with_caller("show-or-hide", what="options_block",
                                        visible=enabled)
            body.setVisible(enabled)

        if not policy.allowed and enabled:
            # NO PROVENANCE CLAIM. This branch used to end "the setting has NOT
            # been changed -- it is your file's", which was safe only while the
            # switch was dead: the ONLY way to reach it was an opened file. Now
            # the ordinary way to reach it is the user's own click, and telling
            # someone their file did it is a false statement about the thing
            # they just did. It says what is wrong and what fixes it, which is
            # true however the setting got here.
            note.setText(
                "THIS RUN CANNOT PRODUCE FLUX CUBES AS CONFIGURED. "
                + policy.reason
                + self._flux_mode_note()
                + " Nothing has been changed for you: fix what is named above,"
                  " or turn the cubes back off.")
            logger.warning(
                "[card] flux_cubes is on and the engine would refuse the pass: "
                "%s. The setting is not changed here; the launch check refuses "
                "the run.", policy.reason)
        elif not enabled and policy.allowed:
            # OFF AND RUNNABLE is the state Paul was looking at, and the note is
            # the only thing on screen. It has to say that there IS an options
            # list and what is in it -- "there still doesn't seem to be any
            # options list for the flux cubes, such as grid spacing and height"
            # is what a group showing one checkbox and a sentence about the
            # physics reads as.
            note.setText(policy.reason + " " + self.FLUX_OFF_HINT)
        else:
            # Off and blocked: `policy.reason` names the blockers and already
            # ends with what to do about each. Inviting the user to turn the
            # cubes on here would be inviting them into the state the launch
            # check refuses.
            #
            # THIS IS THE ONLY THING ON SCREEN ON A GLOBAL RUN, so it carries
            # the mode sentence too. `engine_support` cannot know the mode --
            # it reads a configuration, and an empty `mesh.import_file` on a
            # local card means "no terrain imported yet", not "global run".
            # The card knows, and says so.
            note.setText(policy.reason + self._flux_mode_note())

        # UNCONDITIONAL, and before the early return. `flux_lattice` and
        # `flux_named` are only ever shown or hidden here, so skipping this
        # while the group was off left their visibility to whatever Qt did when
        # the layout reparented them -- correct by accident, and only for the
        # mode the card happened to open on. Now the placement blocks agree
        # with `flux_cubes.mode` in every state, and the reveal below is the
        # single `flux_body` line rather than an ordering.
        self._refresh_flux_placement()
        # The mode rule reads `flux_cubes.enabled`, which the user can now
        # change WITHOUT a mode switch -- impossible while the control was
        # dead. Re-applied here or an emptied group is left standing on a mode
        # that cannot produce cubes at all.
        self._apply_flux_group_visibility()
        if not enabled:
            return
        self._refresh_flux_height()
        self._refresh_flux_cache()
        self._refresh_flux_estimate()

    def _flux_enabled(self):
        """⛔ ONE SOURCE OF TRUTH: THE CONFIGURATION, read in ONE place.

        The configuration is what gets saved and what the engine runs, and
        several production paths change it without touching the switch (a
        preset, a route default, `reset_to_defaults`), so the card has to
        render the configuration or it would ignore them.

        WHAT MAKES THAT SAFE IS THE SIGNAL THIS GROUP LISTENS ON, not this
        function. The configuration write and the re-describe are two slots of
        the SAME click. `add_bool_field` puts the write on `stateChanged`;
        this group therefore also listens on `stateChanged`, connected
        afterwards, because Qt delivers slots of ONE signal in connection
        order -- a documented guarantee. Listening on `toggled` instead made
        it a CROSS-signal ordering question, which is an implementation detail
        of `QCheckBox::nextCheckState` and no guarantee at all: measured on
        this build a bare `QCheckBox` emits `stateChanged` then `toggled`, so
        round 2 happened to be right, and a build where that flipped would
        render every click one behind -- "the options appear after checking
        and then unchecking". See `flux_cube_group`.
        """
        return bool(self.config_manager.get_value("flux_cubes.enabled", False))

    def _apply_flux_group_visibility(self):
        """Show or hide the whole group. Cards that offer it on one mode only
        override this; a card that offers it always needs nothing here."""

    def _flux_mode_note(self):
        """A card's own sentence about why THIS mode cannot produce cubes.

        `flux_cubes.engine_support` reads a configuration and cannot see which
        mode the card is on, so its mesh blocker has to describe an empty
        `mesh.import_file` in general terms. The card knows the mode and says
        what to do about it. Leading space, or the empty string when the mode
        has nothing to add; it is concatenated onto `policy.reason`.
        """
        return ""

    # -- revealing something means putting it on the screen -------------------

    def scroll_into_view(self, widget):
        """Bring `widget` inside the card's visible rectangle.

        ⛔ `setVisible(True)` IS NOT A REVEAL on a scrolling form, and that is
        Paul's 2026-09-01 report. A card is 1,500-2,100 px of content in a
        viewport of 900-1,100, so a group can be shown, unhidden, and 300 px
        below the bottom edge of the window. Nothing on screen changes when it
        is switched on, and the honest description of that is "we're still not
        getting the options menu".

        Measured on `fb78c9f` at Paul's own saved window width (2559 px, from
        `com.ptm3d.PTM3DSimulator.plist`), Moon local mesh, real `click()`:

            curated lunar polar site A, viewport 1023 px -- the switch is at y=1000 and
            the three placement controls the click reveals land at y=1049,
            1078 and 1110. ALL THREE OUTSIDE THE VIEWPORT. The user ticks the
            box and the screen does not change.

            imported mesh, viewport 883 px -- the switch is at y=784 and the
            grid radius at y=894, outside. Before the tick the group shows the
            note that LISTS the options in prose ("where they go ... the
            distance between them, the grid radius, the cube height ..."); the
            tick replaces that list with one short sentence and puts the real
            controls below the fold. Options before, none after: an inverted
            reveal produced entirely by geometry.

        ⛔ AND THIS IS THE THIRD GENERATION OF ONE BLIND SPOT. `isHidden` was
        blind to a hidden ANCESTOR (2026-08-31a). `setChecked` was blind to a
        DISABLED widget (2026-08-31b). `isVisibleTo` is blind to SCROLL
        POSITION. Forty-six green arms sat in front of this one. An arm about
        a control the user is supposed to see must assert the viewport
        rectangle -- `src/python/tests/test_gui_flux_cubes.py` section 11.

        ⛔ NOT `QScrollArea.ensureWidgetVisible`. It guarantees the widget's
        TOP EDGE and stops: measured on the curated card it scrolled 34 px,
        put the group's first row on screen and left the grid spacing and the
        grid radius outside the viewport -- the same report, 34 px smaller.
        The scroll value is computed here so that the widget's BOTTOM is
        brought in, which is where a reveal puts its new controls, and so the
        arms can assert a rectangle rather than a Qt heuristic.

        No-op when there is no scroll area or the widget is not in it.
        """
        scroll = getattr(self, "_scroll", None)
        if scroll is None or widget is None:
            return False
        content = scroll.widget()
        if content is None:
            return False
        if not content.isAncestorOf(widget):
            # WARNS RATHER THAN GUESSING. `mapTo` across unrelated widgets is
            # undefined, and the quiet alternative -- scrolling somewhere
            # plausible -- is how a reveal ends up pointing at the wrong
            # control. A caller that lands here has passed a widget from
            # another card.
            logger.warning(
                "[card] asked to scroll a widget that is not inside this "
                "card's scroll area; nothing has been moved")
            return False
        bar = scroll.verticalScrollBar()
        view_h = scroll.viewport().height()
        top = widget.mapTo(content, QPoint(0, 0)).y()
        # ⛔ NOT `height()` ALONE. `height()` is the REALIZED geometry, and a
        # reveal runs before Qt has re-laid-out: measured in the real window
        # at the instant `_reveal_flux_cubes` fires, the flux group reports
        # `height() == 305` (still collapsed) while `sizeHint() == 618` (what
        # it is about to become). Scrolling to `top + 305 - view_h` is 313 px
        # short, which puts every control the click just revealed below the
        # fold -- and the SECOND time round the group is already laid out, so
        # it works. Check, nothing; uncheck; check, and there they are. That is
        # Paul's 2026-09-01 report, and `QTimer.singleShot(0)` only hides it:
        # whether the layout pass runs before a zero timer is an event-loop
        # race, not a guarantee, and it comes out differently under a test
        # harness that pumps `processEvents` than under a real loop.
        #
        # The larger of the two is right in both directions: before layout the
        # size hint is the answer, after layout they agree, and neither can
        # under-scroll.
        height = max(widget.height(), widget.sizeHint().height())
        current = bar.value()
        if height >= view_h:
            # Taller than the window: show as much of the END as fits, which
            # is the part that just appeared.
            target = top + height - view_h
        elif top < current:
            target = top                       # above the fold
        elif top + height > current + view_h:
            target = top + height - view_h     # below the fold
        else:
            from gui_core import flux_trace
            flux_trace.note_with_caller(
                "scroll-not-needed", already_at=current, widget_top=top,
                widget_height=height, viewport_height=view_h)
            return True                        # already whole on screen
        clamped = max(bar.minimum(), min(bar.maximum(), target))
        bar.setValue(clamped)
        # The position aimed for, the position asked for after clamping, and
        # the position actually reached. A scrollbar that will not go where it
        # was sent -- because the card has not grown yet, so the maximum is
        # still the old one -- shows up here as three different numbers.
        from gui_core import flux_trace
        flux_trace.note_with_caller(
            "scrolled", from_position=current, aimed_for=target,
            after_clamping=clamped, actually_reached=bar.value(),
            scroll_maximum=bar.maximum(), widget_top=top,
            widget_height_used=height, realized_height=widget.height(),
            preferred_height=widget.sizeHint().height(),
            viewport_height=view_h)
        return True

    def _flux_mode(self):
        from gui_core import flux_cubes as fc
        return fc.normalize_mode(
            self.config_manager.get_value("flux_cubes.mode", fc.MODE_LATTICE))

    def _refresh_flux_placement(self):
        from gui_core import flux_cubes as fc
        from gui_core import flux_trace
        mode = self._flux_mode()
        if getattr(self, "flux_lattice", None) is not None:
            flux_trace.note_with_caller(
                "show-or-hide", what="grid_rows",
                visible=fc.mode_builds_lattice(mode), placement=mode)
            self.flux_lattice.setVisible(fc.mode_builds_lattice(mode))
        if getattr(self, "flux_named", None) is not None:
            flux_trace.note_with_caller(
                "show-or-hide", what="named_points_rows",
                visible=fc.mode_uses_locations(mode), placement=mode)
            self.flux_named.setVisible(fc.mode_uses_locations(mode))

    def _refresh_flux_height(self):
        """The cube height, from the cache when there is one and from a
        measurement of the mesh when there is not -- and it says which."""
        from gui_core import flux_cubes as fc
        label = getattr(self, "flux_height", None)
        if label is None:
            return
        header = self._flux_cache_header()
        if header is not None and header.ok:
            label.setText(
                f"{fc._fmt_length(header.height_m)} above the ground, "
                f"{fc._fmt_length(header.edge_m)} on a side -- measured by the "
                f"layout tool on this mesh (a tenth of its "
                f"{fc._fmt_length(header.local_median_edge_m)} median facet "
                f"edge) and recorded in the layout file.")
            return
        geometry = fc.auto_geometry(self.config_manager.get_config(),
                                    self._project_root())
        if geometry.height_m is None:
            label.setText(
                f"not known yet: {geometry.detail}. The layout tool measures "
                f"the mesh and records what it used.")
            return
        label.setText(
            f"about {fc._fmt_length(geometry.height_m)} above the ground and "
            f"the same on a side -- an estimate for this mesh, taken over the "
            f"whole of it. The layout tool measures the grid area instead, so "
            f"expect a few percent; its number is recorded in the layout file "
            f"and is the one that runs.")

    def _flux_cache_header(self):
        """The current layout file's header, or None when no file is named.

        Cached on the resolved path so that re-describing the group on every
        keystroke does not re-read a 50 MB file. The header is 256 bytes plus
        the name block, so the read itself is small, but the path is resolved
        and stat'd and that is worth doing once.
        """
        from gui_core import flux_cubes as fc
        from gui_core.preflight import resolve_path
        raw = str(self.config_manager.get_value("flux_cubes.cache_file", "")
                  or "").strip()
        if not raw:
            self._flux_header_cache = None
            return None
        try:
            path = str(resolve_path(raw, self._project_root()))
        except Exception:                                    # noqa: BLE001
            path = raw
        cached = getattr(self, "_flux_header_cache", None)
        if cached is not None and cached[0] == path:
            return cached[1]
        header = fc.read_cache_header(path)
        self._flux_header_cache = (path, header)
        return header

    def _flux_layout_plan(self):
        """What the launch check would do about the layout, asked here.

        THE SAME FUNCTION the launch check calls, not a second opinion: the
        card's line and the pre-flight's line are one rule, so a card that says
        "it will be built" cannot sit in front of a launch that refuses.
        """
        from gui_core import flux_cubes as fc
        from gui_core.preflight import _ply_face_count, resolve_data_path
        root = self._project_root()
        return fc.layout_plan(
            self.config_manager.get_config(), root, resolve_data_path,
            _ply_face_count, executable=fc.find_geom_executable(root))

    def _refresh_flux_cache(self):
        label = getattr(self, "flux_status", None)
        if label is None:
            return
        from gui_core import flux_cubes as fc
        header = self._flux_cache_header()
        if header is None:
            # No layout file named, which is the ordinary case now: say what
            # the run will do, and name the file it will do it to. "Set the
            # placement and press Build" was an instruction to perform a step
            # that no longer exists.
            try:
                plan = self._flux_layout_plan()
            except Exception as exc:                         # noqa: BLE001
                label.setText(
                    f"The cube layout could not be planned here ({exc}). The "
                    f"launch check asks the same question again before the "
                    f"run starts.")
                return
            if plan.action == fc.PLAN_REFUSE:
                label.setText(f"No cube layout can be prepared for this run: "
                              f"{plan.reason}.")
            elif plan.action == fc.PLAN_REUSE:
                label.setText(f"Ready: {plan.reason}. It is checked against "
                              f"this placement and this mesh, and reused "
                              f"rather than rebuilt.")
            elif plan.action == fc.PLAN_BUILD:
                label.setText(
                    f"The layout is built for you when the run starts -- "
                    f"{plan.request.describe()}, written to {plan.path}. It "
                    f"takes tens of seconds on a site-scale mesh and can be "
                    f"stopped. Nothing here needs a file name.")
            else:
                label.setText(plan.reason)
            return
        if not header.ok:
            label.setText(header.error)
            return
        text = header.describe()
        # Only the names this run will actually ASK for. In a grid-only mode
        # the table's rows are not emitted, so the run never checks them and a
        # complaint about them here would be about a list nobody sends.
        names = (self._flux_location_rows()
                 if fc.mode_uses_locations(self._flux_mode()) else [])
        if names:
            missing = [row["name"] for row in names
                       if row["name"] not in header.names]
            if missing:
                text += (" This layout does not contain "
                         + ", ".join(sorted(missing))
                         + ". The run checks the names and stops, because "
                           "editing the table does not move a cube -- rebuild "
                           "the layout.")
        label.setText(text)

    def _refresh_flux_estimate(self):
        """Cubes, files and bytes -- the size of what this run will write."""
        from gui_core import flux_cubes as fc
        label = getattr(self, "flux_estimate", None)
        if label is None:
            return
        header = self._flux_cache_header()
        named = self._flux_location_rows()
        if header is not None and header.ok:
            n_cubes = header.n_cubes
            n_named = header.n_named
            source = "the layout file"
        else:
            mode = self._flux_mode()
            n_named = len(named) if fc.mode_uses_locations(mode) else 0
            n_cubes = n_named + (
                fc.lattice_cube_count(
                    self._number("flux_cubes.spacing_m", fc.DEFAULT_SPACING_M),
                    self._number("flux_cubes.radius_m", fc.DEFAULT_RADIUS_M))
                if fc.mode_builds_lattice(mode) else 0)
            source = "the placement above"
        steps = int(self._number("simulation.timesteps", 0))
        parts = [f"{n_cubes:,} cubes from {source}"]
        if steps > 0 and n_cubes > 0:
            if self.config_manager.get_value("flux_cubes.write_binary", True):
                size = fc.product_bytes(n_cubes, steps, has_names=n_named > 0)
                parts.append(f"about {fc.fmt_bytes(size)} of binary flux "
                             f"({n_cubes:,} x 6 faces x 6 components x "
                             f"{steps:,} steps, eight bytes each)")
            if self.config_manager.get_value("flux_cubes.write_csv", True):
                files = fc.csv_files_written(
                    n_cubes, n_named,
                    self._number("flux_cubes.write_csv_max_cubes", 0))
                parts.append(f"{files:,} spreadsheet file(s), roughly "
                             f"{fc.fmt_bytes(fc.csv_bytes(files, steps))}")
        label.setText("Estimated output: " + "; ".join(parts)
                      + ". Sizes are estimates and exclude a short provenance "
                        "record in each file.")

    # -- the named-locations table -------------------------------------------

    def _flux_location_rows(self):
        from gui_core import flux_cubes as fc
        return fc.config_locations(self.config_manager.get_config())

    def load_flux_locations(self):
        """Mirror `flux_cubes.locations` into the table. Never writes back."""
        table = getattr(self, "flux_table", None)
        if table is None:
            return
        rows = self._flux_location_rows()
        blocked = table.signalsBlocked()
        table.blockSignals(True)
        try:
            table.setRowCount(len(rows))
            for r, row in enumerate(rows):
                for c, (_heading, key) in enumerate(self.LOCATION_COLUMNS):
                    value = row.get(key)
                    text = "" if value is None else (
                        value if key == "name" else f"{float(value):g}")
                    table.setItem(r, c, QTableWidgetItem(str(text)))
        finally:
            table.blockSignals(blocked)

    def _table_locations(self):
        """The table's contents as rows, plus the lines that did not parse."""
        from gui_core import flux_cubes as fc
        table = getattr(self, "flux_table", None)
        if table is None:
            return [], []
        lines = []
        for r in range(table.rowCount()):
            fields = []
            for c in range(len(self.LOCATION_COLUMNS)):
                item = table.item(r, c)
                fields.append("" if item is None else item.text().strip())
            if not any(fields):
                continue
            lines.append(",".join(fields))
        # `allow_header=False`: a table has no header row, and the file rule --
        # a first field of exactly `name` is a header -- would silently drop a
        # point somebody typed there.
        return fc.parse_locations_csv("\n".join(lines), allow_header=False)

    def _write_flux_locations(self):
        """Store the table, and say so when a row could not be read.

        A row that does not parse is NOT written and NOT silently dropped: the
        status line names it. A point that vanished on the way into the
        configuration is a cube the user believes is in the product.
        """
        if self._loading:
            return
        rows, errors = self._table_locations()
        self.config_manager.set_value("flux_cubes.locations", rows)
        label = getattr(self, "flux_status", None)
        if errors and label is not None:
            label.setText("These rows were not stored: " + "; ".join(errors))
        else:
            self._refresh_flux_cache()
        self._refresh_flux_estimate()
        self.value_changed.emit()

    def _on_location_edited(self, _item):
        self._write_flux_locations()

    def _on_add_location(self):
        table = getattr(self, "flux_table", None)
        if table is None:
            return
        table.insertRow(table.rowCount())

    def _on_remove_location(self):
        table = getattr(self, "flux_table", None)
        if table is None:
            return
        for index in sorted({i.row() for i in table.selectedIndexes()},
                            reverse=True):
            table.removeRow(index)
        self._write_flux_locations()

    def _on_import_locations(self):
        path, _filter = QFileDialog.getOpenFileName(
            self, "Import cube locations", str(self._project_root()),
            "Comma-separated values (*.csv *.txt);;All Files (*)")
        if not path:
            return
        from gui_core import flux_cubes as fc
        try:
            text = Path(path).read_text()
        except OSError as exc:
            QMessageBox.warning(self, "Import cube locations",
                                f"{path} could not be read: {exc}")
            return
        rows, errors = fc.parse_locations_csv(text)
        if errors:
            QMessageBox.warning(
                self, "Import cube locations",
                f"{len(rows)} point(s) read and "
                f"{len(errors)} line(s) refused:\n\n" + "\n".join(errors)
                + "\n\nNothing was imported. Fix the file and try again.")
            return
        if not rows:
            QMessageBox.warning(self, "Import cube locations",
                                f"{path} contains no points.")
            return
        self.config_manager.set_value("flux_cubes.locations", rows)
        self.load_flux_locations()
        self.refresh_flux_cubes()
        self.value_changed.emit()

    # -- building the layout -------------------------------------------------

    def _on_build_layout(self):
        """Build the layout NOW, instead of when the run starts.

        THE SAME PLAN AND THE SAME WORKER the launch path uses -- this button
        only moves the wait earlier. It deliberately does NOT offer a save
        dialog any more: the layout's name carries a digest of the placement it
        was built for, which is what stops a 25 m grid being reused for a 50 m
        one, and a name chosen by hand carries nothing. To use a layout that
        already exists, put it in the field above instead.

        Placement is the tool's job and only the tool's: it measures the mesh,
        drops each cube onto the terrain and casts the rays that record what
        every cube face can see. Nothing in this window moves a cube, which is
        why editing the fields above changes no result until the layout is
        rebuilt -- and why the run compares the layout against its own mesh and
        stops on a mismatch.
        """
        from gui_core import flux_cubes as fc
        from gui_core.flux_cube_build import build_layout

        rows, errors = self._table_locations()
        if fc.mode_uses_locations(self._flux_mode()) and errors:
            # Refused BEFORE the plan is built, because `config_locations`
            # reads the stored list and a row that never parsed is not in it --
            # so a layout built now would quietly be missing the point the
            # table is showing.
            QMessageBox.warning(
                self, "Build cube layout",
                "The point list has to be readable before a layout can be "
                "built.\n\n" + "\n".join(errors))
            return
        try:
            plan = self._flux_layout_plan()
        except Exception as exc:                             # noqa: BLE001
            QMessageBox.warning(self, "Build cube layout",
                                f"The layout could not be planned: {exc}")
            return
        if plan.action == fc.PLAN_NONE:
            QMessageBox.information(
                self, "Build cube layout",
                "This configuration names its own layout file, so nothing is "
                "built from the placement. Clear the field above to have the "
                "layout built from the placement instead.")
            return
        if plan.action == fc.PLAN_REFUSE:
            QMessageBox.warning(self, "Build cube layout",
                                f"No layout can be built: {plan.reason}.")
            return

        self.flux_build.setEnabled(False)
        try:
            if plan.action == fc.PLAN_REUSE:
                ok, log = True, plan.reason
            else:
                ok, log = build_layout(self, plan, str(self._project_root()))
        finally:
            self.flux_build.setEnabled(True)
        self._on_layout_built(ok, log, plan.path)

    def _on_layout_built(self, ok, log, target):
        # Drop any cached header for this path: the file just changed.
        self._flux_header_cache = None
        if not ok:
            self.flux_status.setText(
                "The layout was not built, and nothing was changed. " + log)
            logger.warning("[card] %s failed: %s", "cube layout build", log)
            return
        self.config_manager.set_value("flux_cubes.cache_file", target)
        widget = self.widgets.get("flux_cubes.cache_file")
        if widget is not None:
            widget.setText(str(target))
        self.refresh_flux_cubes()
        self.value_changed.emit()

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
        # GUI-only, and normalized the same way for the same reason: an older
        # project or a hand-edited file can carry "lattice" or "locations", and
        # a combo left showing "grid" for a configuration that says "locations"
        # would build the wrong layout. The table is the module's own
        # `MODE_ALIASES`, so the combo and every reader agree by construction.
        "flux_cubes.mode": _flux_mode_aliases(),
        # The two the engine distinguishes. A file carrying 2, 3 or 4 is NOT
        # mapped onto either: `set_enum` adds it as a foreign entry and warns,
        # so the card shows what the file says while the tooltip states that
        # the engine reads anything under five as one ray.
        "flux_cubes.shadow_probe_rays": {"1": "1", "5": "5"},
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
        # The offer, reset. A configuration just loaded describes a different
        # mesh from the one the last scan ranked against, and a stale candidate
        # list is a list of matrices for somebody else's terrain.
        self.refresh_vf_discovery()


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

        site_label = QLabel("Curated site:")
        site_label.setToolTip(get_tooltip("ui.curated_site"))
        self._apply_tooltip(site_label, "ui.curated_site")
        grid.addWidget(site_label, 0, 0)
        self.site_combo = QComboBox()
        self.site_combo.setToolTip(get_tooltip("ui.curated_site"))
        self.site_combo.addItem("(none — choose the files yourself)", "")
        for entry in curated_sites_for_body("Moon"):
            self.site_combo.addItem(entry.display(), entry.key)
        self.site_combo.currentIndexChanged.connect(self._on_curated_site)
        grid.addWidget(self.site_combo, 0, 1)
        # ONE LINE, AND A PLACE TO PUT THE REST. Picking a site used to write
        # the whole provenance record here -- about thirty lines, 216 px tall
        # at Paul's window width -- which pushed every group below it, the
        # flux cubes among them, that far down the card and off the bottom of
        # the window. The provenance is worth keeping and worth reading; it is
        # not worth reading every time, and it was standing between the user
        # and the controls underneath. What stays on the line is what the user
        # needs to know without asking: whether the site is usable, and
        # anything about their configuration that disagrees with it.
        self.site_note = _muted("")
        grid.addWidget(self.site_note, 1, 0, 1, 2)
        self.site_details = CollapsibleSection(
            "Details: the files this site uses, and why each one", collapsed=True)
        self.site_details_text = _muted("")
        self.site_details.add_widget(self.site_details_text)
        self.site_details.setVisible(False)
        grid.addWidget(self.site_details, 2, 0, 1, 2)
        # A BROKEN LOCAL OVERLAY MUST NOT LOOK LIKE AN ABSENT ONE. The curated
        # entries live outside the repository (gui_core.curated_sites, "The
        # local overlay"), so an empty list is the normal state for most
        # checkouts -- which is exactly why an overlay that failed to parse
        # has to say so here. Both states end with one item in the combo, and
        # only one of them is a fault.
        overlay = local_registry_status()
        if not overlay.ok:
            self.site_note.setText(overlay.summary())

        self.add_file_field(grid, 3, "Terrain mesh:", "mesh.import_file",
                            "Meshes (*.ply *.obj *.vtk);;All Files (*)")
        # The label argument is DEAD for this key: `_resolve_text` prefers
        # `gui_tooltips.LABELS`, which spells it "Shadow-only surrounding
        # mesh:". Kept identical to the rendered text so the call site cannot
        # be read as a second, disagreeing answer.
        self.add_file_field(grid, 4, "Shadow-only surrounding mesh:",
                            "mesh.occluder_file",
                            "Meshes (*.ply *.obj *.vtk);;All Files (*)")
        self.add_double_field(grid, 5, "Latitude (deg):",
                              "mesh.latitude_override", -999.0, 90.0,
                              decimals=4)
        self.add_double_field(grid, 6, "Longitude (deg):",
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

        # -- flux cubes (LOCAL MESH ONLY; see BodyCard.flux_cube_group) -------
        # Below radiation on purpose: the cubes report what the surrounding
        # terrain sends, which the run only computes when the switch above is
        # on, and the group says so when it is not.
        self.flux_group = self.flux_cube_group()
        layout.addWidget(self.flux_group)
        self.widgets["radiation.enabled"].toggled.connect(
            lambda _c: self.refresh_flux_cubes())

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
        # The gate depends on the mode (a global run has no imported mesh), so
        # the group has to be re-described whenever the mode moves. Without
        # this it kept the previous mode's answer. It re-applies
        # `_apply_flux_group_visibility` on its way through, so the rule below
        # runs for a mode change and for a toggle alike.
        self.refresh_flux_cubes()

    def _apply_flux_group_visibility(self):
        """⛔ THE GROUP IS NEVER HIDDEN, ON EITHER MODE.

        The rule used to be `setVisible(local or enabled)`. It hid the whole
        group -- switch, note and all -- on a Moon GLOBAL card with the cubes
        off, which is the default state of that mode and of five of the eight
        still-existing files in Paul's own recent-files list. He reported it
        twice in the same words: "there still doesn't seem to be any options
        list for the flux cubes" (2026-08-31, the local card, a dead switch)
        and "we're still not getting the options menu for the flux cubes"
        (2026-09-01, on the merged fix). The first report was the switch; this
        is the second, and it is a different defect on a different mode.

        Measured on `fb78c9f` -- Moon / Global / time series, GUI defaults,
        nothing clicked: `flux_group.isHidden()` is True, the switch is
        `isVisibleTo(card) == False`, and `flux_note` has already been given
        the sentence that explains why. The explanation was computed and then
        taken off the screen with the thing it explained. A feature that is
        absent without a word is indistinguishable from one that was never
        built, and the user has nowhere to look.

        Visibility is therefore unconditional and the mode is a SENTENCE
        (`_flux_mode_note`), not an absence. Nothing unsafe becomes reachable:
        `engine_support` still refuses the global mode, the note still says so,
        and `flux_cubes.preflight_findings` still turns it into a launch error.

        ⛔ AND IT LIVES HERE, not inline in `apply_mode`, because the warning
        below reads `flux_cubes.enabled` and that can change without a mode
        change: the switch is always live (see `refresh_flux_cubes`). Both
        callers use this one rule.
        """
        local = self._mode == "local"
        enabled = self._flux_enabled()      # the switch, not the configuration
        from gui_core import flux_trace
        flux_trace.note_with_caller("show-or-hide", what="whole_group",
                                    visible=True, mode=self._mode,
                                    setting=enabled)
        self.flux_group.setVisible(True)
        if not local and enabled:
            logger.warning(
                "[card] this configuration has flux_cubes.enabled on and the "
                "card is on the global mode, which builds its sphere in memory "
                "and has no mesh file to place cubes against. The group is "
                "kept on screen so the setting can be seen and cleared; it is "
                "NOT changed here.")

    def _flux_mode_note(self):
        """Name the mode as the reason, and name the control that changes it.

        Only reached when `engine_support` has already refused, so this never
        appears on a run that can produce cubes.
        """
        if self._mode == "local":
            return ""
        return (" This card is set to Global, which builds its sphere in "
                "memory and writes no mesh file. Choose Local mesh at the top "
                "of the card, and import terrain or pick a curated site, to "
                "record flux cubes.")

    def load_card(self):
        from gui_core import flux_trace
        switch = self.widgets.get("flux_cubes.enabled")
        flux_trace.note_with_caller(
            "card-reloaded",
            checkbox_before=None if switch is None else switch.isChecked(),
            setting_before=self.config_manager.get_value("flux_cubes.enabled",
                                                         None))
        super().load_card()
        self._mirror_curated_site()
        self._refresh_surface()
        # The table is not in `self.widgets`, so the generic mirror does not
        # reach it; loaded here, inside the `_loading` guard the caller holds,
        # so mirroring the list does not write it back.
        self.load_flux_locations()
        self.refresh_flux_cubes()

    # -- the curated lunar polar landing sites ---------------------------------------

    def _current_site(self):
        key = self.site_combo.currentData()
        return curated_site(key, body="Moon") if key else None

    def _mirror_curated_site(self):
        """Show the site whose mesh this configuration already names.

        Matching is on the MESH PATH, not on the occluder or the matrix: the
        core mesh is what identifies a site, and a configuration that has the
        right terrain with a different ring is still that site -- shown, and
        then described by `_describe_curated_site`, which compares every key
        and names the ones that differ rather than quietly resetting them.
        """
        current = str(self.config_manager.get_value("mesh.import_file", "")
                      or "").strip()
        index = 0
        if current:
            for position in range(1, self.site_combo.count()):
                entry = curated_site(self.site_combo.itemData(position),
                                     body="Moon")
                if entry is not None and entry.mesh.path == current:
                    index = position
                    break
        self.site_combo.blockSignals(True)
        self.site_combo.setCurrentIndex(index)
        self.site_combo.blockSignals(False)
        self._describe_curated_site()

    def _describe_curated_site(self, applied=()):
        """Say what this site IS, and where the configuration disagrees.

        ⛔ THE LINE AND THE DETAILS ARE NOT THE SAME THING, and which half a
        sentence belongs in is a judgment about safety, not about length.

        ON THE LINE, always: whether the site is usable, and any way in which
        the open configuration DISAGREES with the curated set. A disagreement
        is a statement about the run somebody is about to launch; putting it
        behind a control they have to find first is how a warning stops being
        a warning. It is on the line even though it is the longest thing here.

        IN THE DETAILS, collapsed: the wiring record and `wiring_provenance` --
        which file was chosen for each role, which near-name files were NOT
        used and why, where the matrix came from and what proved it. True,
        valuable, and about thirty lines. It used to be on the line, 216 px
        tall at Paul's window width, pushing the surface, radiation, flux-cube,
        thermal and output groups that far down the card. That is how the
        flux-cube group ended up below the bottom of his window (2026-09-01).

        Never rewrites anything on its own. A file the user opened that names
        a curated site's terrain with some other ring or matrix keeps what it
        has and is told which keys differ -- opening a file is not an edit to
        it, and this card is not the place that decides somebody's ring was a
        mistake.
        """
        note = getattr(self, "site_note", None)
        if note is None:
            return
        entry = self._current_site()
        if entry is None:
            # ⛔ A BROKEN OVERLAY MUST SURVIVE THIS. The curated entries come
            # from a local file outside the repository, and when that file
            # fails to parse the registry contributes nothing -- so `entry` is
            # None here for a reason that is a FAULT, and this line used to
            # overwrite the fault with the ordinary "no curated site" text on
            # every `load_card`. The user was then told, accurately and
            # uselessly, that they had not picked a site.
            overlay = local_registry_status()
            if not overlay.ok:
                note.setText(overlay.summary())
            else:
                note.setText(
                    "No curated site: the terrain, the surrounding ring and "
                    "the view-factor cache below are yours to set. A curated "
                    "site fills all three from a set that has run to "
                    "completion.")
            self._set_site_details(None)
            return
        from gui_core.curated_sites import (site_status, site_wiring,
                                            wiring_provenance)
        status = site_status(entry)
        headline = [status.summary()]
        # Only worth computing when the card is DESCRIBING a configuration
        # somebody else wrote. Straight after `applied`, every key is by
        # construction the value just written.
        if status.ready and not applied:
            differing = []
            for key, value in site_wiring(entry):
                if key.startswith("engine_env."):
                    current = (self.config_manager.get_value("engine_env", {})
                               or {}).get(key.split(".", 1)[1])
                else:
                    current = self.config_manager.get_value(key)
                if not _same_setting(current, value):
                    differing.append(f"{key} is {current!r}, the curated set "
                                     f"uses {value!r}")
            if differing:
                headline.append(
                    "THIS CONFIGURATION DIFFERS FROM THE CURATED SET "
                    "AND HAS NOT BEEN CHANGED: " + "; ".join(differing))
        detail_lines = []
        if applied:
            headline.append(f"{len(applied)} settings were written from this "
                            f"site; see the details below for each one.")
            detail_lines.append("Wrote: " + "; ".join(f"{k} = {v}"
                                                      for k, v in applied))
        detail_lines.extend(wiring_provenance(entry))
        note.setText(" ".join(headline))
        self._set_site_details(detail_lines)

    def _set_site_details(self, lines):
        """Fill the collapsed details, or take the section off the card.

        ⛔ DOES NOT TOUCH THE COLLAPSED STATE. Re-describing happens on every
        load, every mirror and every site change; forcing it shut would close
        the section under a user who had just opened it, and forcing it open
        would undo the whole point. It opens collapsed once, at construction,
        and after that the state is the user's.
        """
        details = getattr(self, "site_details", None)
        details_text = getattr(self, "site_details_text", None)
        if details is None or details_text is None:
            return
        if not lines:
            details_text.setText("")
            details.setVisible(False)
            return
        details_text.setText("\n".join(lines))
        details.setVisible(True)

    def _on_curated_site(self, _index):
        """Apply a whole set, or refuse it — and say which, every time.

        A curated site is the one place in this card where selecting something
        writes several keys at once, and that is the user's explicit choice.
        What it must never do is write a PARTIAL set: a terrain with no
        matrix, or a matrix whose file is a cloud placeholder, is a run that
        starts and is wrong. `site_status` decides, and a refusal writes
        nothing at all.
        """
        entry = self._current_site()
        if entry is None:
            self._describe_curated_site()
            return
        from gui_core.curated_sites import site_status, site_wiring
        status = site_status(entry)
        if not status.ready:
            logger.warning("[card] the curated site %s was not wired: %s",
                           entry.key, status.summary())
            self.site_combo.blockSignals(True)
            self.site_combo.setCurrentIndex(0)
            self.site_combo.blockSignals(False)
            self.site_note.setText(
                status.summary() + " Nothing was written; the site is left "
                "unselected so the configuration cannot half-describe it.")
            # The details belong to a site that is now unselected. Leaving the
            # previous site's file list under a refusal would read as though
            # the refusal were about those files.
            self._set_site_details(None)
            return
        wiring = site_wiring(entry)
        for key, value in wiring:
            self.config_manager.set_value(key, value)
        self.load_from_config()
        self._describe_curated_site(applied=wiring)
        self.value_changed.emit()

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
        # A ranked candidate list belongs to ONE mesh. Reset it here rather
        # than leaving matrices ranked against the previous terrain on screen
        # beside a new one -- the wrong-mesh case this whole search exists to
        # make visible.
        self.refresh_vf_discovery()
        # The mesh decides the cube height and whether a layout can be built at
        # all, so a mesh change re-measures rather than leaving the previous
        # mesh's number on screen.
        self._flux_header_cache = None
        self.refresh_flux_cubes()

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
            # written by the named-locations table, which is not a single
            # widget and so is not in `self.widgets`
            "flux_cubes.locations",
            # written by the curated-site picker, which sets a whole set at
            # once. The mesh, ring, placement sentinels, radiation switch and
            # cache field are bound controls and are already covered by
            # `self.widgets`; these three are not.
            "simulation.body", "radiation.use_gpu", "engine_env",
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

# `_GeomWorker` used to live here: a second `ptm_flux_cube_geom` runner
# beside `gui_core.flux_cube_build.LayoutBuildWorker`. It is gone, and the
# card's build button goes through the shared worker the launch path uses --
# which, unlike this one, removes a partial layout when a build fails and
# refuses a zero exit code that wrote no file. Two runners meant the button
# and the launch had different guarantees for one job.


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
        over a body the engine has built-in properties for BESIDE an active
        `generic_body:` section, or over the unnamed placeholder while this
        card is being opened for a new run. Both are configurations that would
        run as something other than what they say --
        a built-in name beside a `generic_body:` section is the contradiction
        `ConfigManager.body_identity_conflict` reports, and letting a card
        display it unchanged would hand the user a run that silently drops
        either the body or its definition.

        Anything else the user or their file chose is left exactly as it is.

        WHY EACH BRANCH IS SCOPED THE WAY IT IS. `load_card` runs on
        BOTH paths: ``ConfigurationPanel.load_from_config`` mirrors whichever
        card is ACTIVE, and ``ptm3d_gui.open_project`` calls that
        (``ptm3d_gui.py:597``) BEFORE ``land_on_loaded_config``
        (``ptm3d_gui.py:607``) has had any chance to route the opened file to
        its own card. Unscoped, that made displaying a file an edit to it.

        * The **built-in-name** branch fires only when the configuration also
          carries an ACTIVE ``generic_body:`` section -- i.e. only when the
          file really does name two bodies. Without that condition, an asteroid
          or custom card left open rewrote an ordinary lunar file's
          ``simulation.body`` from ``Moon`` to ``Custom`` on open, and
          ``land_on_loaded_config`` then refused the file it had just edited
          (measured on ``origin/main`` 2baf6bb: ``route.ok`` False,
          ``body_key`` None). With the condition, a plain Moon file is left
          alone and the two-bodies file is still repaired -- which matters,
          because ``_config_for_yaml`` DROPS the ``generic_body:`` block when
          ``simulation.body`` names a built-in body, so leaving that file
          unrepaired emits a Moon run from a card describing an asteroid
          (measured, same day, on the first draft of this guard).
        * The **placeholder** branch is cross-card residue from a NEW run --
          the route defaults write ``Custom`` (``presets.py:315``) and this
          puts the body's real name in its place -- so it is gated on
          ``_seeding_defaults``, the same flag, for the same reason, as
          ``EuropaCard._ensure_terrain_emitted``. It protects nothing on an
          opened file: ``Custom`` is not a built-in name, so the block survives.

        The other caller that legitimately needs the name written -- a HORIZONS
        or cache lookup -- asks for it explicitly in ``_apply_lookup``, because
        a lookup IS an edit.
        """
        from gui_core.config_manager import BUILTIN_BODY_NAMES
        current = str(self.config_manager.get_value(
            "simulation.body", "") or "").strip()
        name = str(self.config_manager.get_value(
            "generic_body.name", "") or "").strip()
        contradicts = (current.lower() in BUILTIN_BODY_NAMES
                       and self.config_manager.generic_body_active())
        placeholder = (self._seeding_defaults
                       and current == self.UNNAMED_BODY and bool(name))
        if not (contradicts or placeholder):
            return
        if contradicts:
            # Never silent: this is the card resolving a contradiction in
            # somebody's file, and which way it went has to be visible.
            logger.warning(
                "[body-card] %s names both simulation.body=%r and an active "
                "generic_body: section (%r); the engine reads the generic body "
                "and ignores simulation.body, so simulation.body is being set "
                "to %r to keep the two in agreement.",
                type(self).__name__, current, name, name or self.UNNAMED_BODY)
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
        # `simulation.body` follows the name the lookup just found. EXPLICIT
        # here rather than as a side effect of the `load_from_config` below,
        # for two measured reasons:
        #   * that door is also the one an OPENED FILE comes through, and it is
        #     now shut (`_repair_body_name`);
        #   * it never covered this case properly anyway. `_repair_body_name`
        #     writes only over a built-in name or the `Custom` placeholder, so
        #     looking up a SECOND body left the FIRST one in `simulation.body`:
        #     measured on `origin/main` 2baf6bb, a cache lookup of "23871
        #     Ousha" straight after "623 Chimaera" left `simulation.body:
        #     623 Chimaera` beside `generic_body.name: 23871 Ousha`.
        # `_write_body_name` keeps its own two guards -- it leaves a `[H] `
        # prefixed name alone and writes nothing when the value already agrees.
        # NOT CLOSED BY THIS, and left as it was found: after
        # `ConfigurationPanel.add_horizons_body` (`config_panel.py:1185`) has
        # written `simulation.body: '[H] <body>'`, a later lookup of a
        # DIFFERENT body still leaves the prefixed name in place, because that
        # early return fires. The prefix is load-bearing GUI state and removing
        # it here is a separate change; it is a ledger row, not a silent tweak.
        self._write_body_name()
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

        # Generic-body metakernel coverage. A kernel that does not span the run is not an error the
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

        A refusal writes NOTHING. That is the whole of the generic-body metakernel fix in one branch:
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
        elif getattr(result, "tracked_destination", ""):
            self._offer_tracked_replacement(result)

    def _on_fetch_spk(self, curated_download=False, on_tracked=None):
        """Get this run's ephemeris: the curated registry, or HORIZONS.

        `curated_download` is False on the button press and True only on the
        second pass, after the user has answered the confirmation the first
        pass produced. That two-pass shape is why the offer is a FLAG on the
        result rather than a phrase in its message: the card has to be able to
        tell "this needs your consent" from "this failed" without reading
        English.

        `on_tracked` is the same shape of answer for the other question this
        fetch can raise -- "the destination is a kernel this repository ships"
        -- and is None on the button press, meaning `fetch_spk`'s own default:
        leave the shipped file alone and save beside it. The user is then
        OFFERED the replacement, which costs a second request and is refused
        anyway if it would lose coverage.
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
        extra = {} if on_tracked is None else {"on_tracked": on_tracked}
        self._start_fetch(
            "spk",
            lambda emit: fetch_spk(config, project_root, name=name,
                                   naif_id=naif_id,
                                   curated_download=curated_download,
                                   progress=emit, **extra),
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

    def _offer_tracked_replacement(self, result):
        """The shipped kernel was left alone. Offer the replacement anyway.

        The fetch has already done the safe thing -- the kernel is beside the
        tracked one and this configuration points at the copy -- so this
        dialog is the way OUT of the guard, not the way past it. Saying yes
        costs a second HORIZONS request and is still refused if the new kernel
        would not contain what the shipped one covers.

        Defaulting the button to No is the whole point: the thing that went
        wrong was a replacement nobody chose.
        """
        from gui_core.body_fetch import ON_TRACKED_REPLACE

        tracked = str(getattr(result, "tracked_destination", "") or "")
        if not tracked:
            return False
        answer = QMessageBox.question(
            self, "Replace the kernel this repository ships?",
            f"{Path(tracked).name} is part of the distribution and has NOT "
            f"been changed. The kernel that was just fetched is beside it, "
            f"and this body now points at that copy.\n\n"
            f"{' '.join(result.lines())}\n\n"
            f"Replace the shipped kernel with the fetched one instead? It "
            f"will be fetched again, and the replacement is refused if it "
            f"would drop coverage the shipped kernel has.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if answer != QMessageBox.StandardButton.Yes:
            # APPENDED. `_apply_fetch` has just put the lines naming both files
            # and both coverages into this label; replacing them with a
            # sentence that says "the message above names both" would delete
            # the message it is talking about.
            self.coverage_note.setText(
                self.coverage_note.text()
                + f"\n{Path(tracked).name} was left as it is; this body uses "
                  f"the kernel saved beside it.")
            return True
        self._on_fetch_spk(on_tracked=ON_TRACKED_REPLACE)
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
