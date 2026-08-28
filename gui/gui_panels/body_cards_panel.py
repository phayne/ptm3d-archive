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
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFrame, QGridLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton,
    QRadioButton, QScrollArea, QSpinBox, QStackedWidget, QVBoxLayout, QWidget,
)

from gui_core.body_cards import GENERIC_MESH_SHAPE
from gui_core.config_manager import ConfigManager
from gui_panels.config_panel import BaseConfigTab, get_tooltip

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
        super().__init__(config_manager)

    # -- activation ---------------------------------------------------------

    def activate(self, mode_key, run_type_key):
        """Make this the live card, for one mode and one run type."""
        self._active = True
        self._mode = mode_key
        self._run_type = run_type_key
        self.apply_mode()
        self.load_from_config()

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

    def time_group(self, title="Time window"):
        """The three time controls every card carries."""
        group = QGroupBox(title)
        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        self.add_string_field(grid, 0, "Start (UTC):", "simulation.start_time")
        spin = self.add_double_field(
            grid, 1, "Duration (days):", "simulation.duration_days",
            0.001, 365000.0, decimals=6)
        spin.setSingleStep(1.0)
        self.add_int_field(grid, 2, "Output samples:", "simulation.timesteps",
                           1, 100000)
        group.setLayout(grid)
        return group

    def thermal_group(self, title="Thermal", integration_window=False):
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

        `integration_window` adds ``thermal.integration_days``, which was
        likewise misfiled: it sat inside the Europa card's mesh-geometry
        section, where it is neither geometry nor Europa-specific.
        """
        group = QGroupBox(title)
        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        row = 0
        if integration_window:
            self.add_double_field(grid, row, "Integration window (days):",
                                  "thermal.integration_days", -1.0, 365000.0,
                                  decimals=4)
            row += 1
        spin = self.add_double_field(
            grid, row, "Spin-up (diurnal cycles):",
            "thermal.equilibration_orbits", 0.0, 100000.0, decimals=2)
        spin.setSingleStep(1.0)
        row += 1
        self.add_bool_field(grid, row, "Run the thermal solver on the GPU",
                            "thermal.use_gpu")
        group.setLayout(grid)
        return group

    def output_group(self, title="Output"):
        """Where the run's files land, and nothing else."""
        group = QGroupBox(title)
        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        self.add_directory_field(grid, 0, "Output directory:",
                                 "output.directory")
        group.setLayout(grid)
        return group

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
        self._loading = True
        try:
            self.load_card()
        finally:
            self._loading = False

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
        skip = self.custom_loaded_keys()
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

        # -- radiation exchange (local only) ----------------------------------
        self.radiation_group = QGroupBox("Radiation exchange")
        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        self.add_bool_field(grid, 0, "Facet-to-facet infrared exchange",
                            "radiation.enabled")
        # Label from `gui_tooltips.LABELS`, which wins over any literal here.
        self.add_bool_field(grid, 1, "Beam infrared from sub-facet roughness",
                            "radiation.beaming")
        self.beaming_note = _muted("")
        grid.addWidget(self.beaming_note, 2, 0, 1, 2)
        self.add_file_field(grid, 3, "View-factor cache:",
                            "radiation.view_factor_file",
                            "View factors (*.vf *.csr *.hmat);;All Files (*)")
        self.radiation_group.setLayout(grid)
        layout.addWidget(self.radiation_group)
        self.widgets["radiation.beaming"].toggled.connect(
            self._on_beaming_toggled)

        layout.addWidget(self.thermal_group())
        layout.addWidget(self.output_group())
        self._make_report(layout)
        layout.addStretch()
        self.create_scroll_area(content)

    def apply_mode(self):
        local = self._mode == "local"
        self.global_group.setVisible(not local)
        self.local_group.setVisible(local)
        self.radiation_group.setVisible(local)

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
        self._update_beaming_gating()

    def _update_beaming_gating(self):
        """Gate the switch on the facet-scale rule -- WITHOUT editing the file.

        The control is disabled and the reason is shown. What it deliberately
        does NOT do is turn an inherited `beaming: true` off: that would be an
        edit to the user's own configuration performed on OPEN, which is the
        class the correctness stage spent fourteen items removing and which the
        Radiation page already refuses to do for the same reason. An opened
        configuration whose beaming the rule would refuse gets a loud line
        saying the rule disagrees with it, and keeps its value.
        """
        from gui_core.facet_scale import beaming_policy
        widget = self.widgets["radiation.beaming"]
        try:
            policy = beaming_policy(self.config_manager.get_config(),
                                    self._project_root())
        except Exception as exc:                             # noqa: BLE001
            widget.setEnabled(False)
            self.beaming_note.setText(
                f"The facet-scale check could not run ({exc}), so this control "
                f"is disabled. Refusing is the safe direction. Any value "
                f"already in the configuration is left as it is.")
            return
        widget.setEnabled(bool(policy.allowed))
        if not policy.allowed and widget.isChecked():
            self.beaming_note.setText(
                "THIS CONFIGURATION HAS SUB-FACET ROUGHNESS ON, AND THE "
                "FACET-SCALE RULE DISAGREES WITH IT. " + policy.reason
                + " The value has NOT been changed -- it is your file's. Turn "
                  "it off yourself if you agree, or run it as it is.")
            logger.warning(
                "[card] this configuration enables radiation.beaming and the "
                "facet-scale rule would refuse it: %s. The setting is left "
                "alone; opening a file is not an edit to it.", policy.reason)
        elif policy.needs_acknowledgment:
            # A THIRD state, and it is not "fine". The rule could not MEASURE
            # this mesh, which is a different statement from "this mesh is too
            # fine" -- and reporting the first as the second would lock beaming
            # out of every format the reader does not parse but the engine
            # loads. The card says which of the two it is, and leaves the
            # judgment where it belongs.
            self.beaming_note.setText(
                "THE FACET SCALE COULD NOT BE MEASURED, so nothing here has "
                "checked this setting for you. " + policy.reason)
        else:
            self.beaming_note.setText(policy.reason)

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

        # -- a Sutton DEM site -----------------------------------------------
        self.local_group = QGroupBox("Local mesh (Sutton DEM)")
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

        self.add_bool_field(grid, 3, "Facet-to-facet infrared exchange",
                            "radiation.enabled")
        self.vf_note = _muted("")
        grid.addWidget(self.vf_note, 4, 0, 1, 2)
        self.local_group.setLayout(grid)
        layout.addWidget(self.local_group)

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
        layout.addWidget(self.thermal_group(integration_window=True))
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
        self._refresh()

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
            path = Path(root) / entry.import_file
            state = ("on disk" if path.exists()
                     else "NOT on disk — the Sutton DEMs are not distributed "
                          "with PTM3D")
            self.terrain_note.setText(f"{entry.import_file} — {state}")
            vf = Path(root) / entry.view_factor_file
            self.vf_note.setText(
                f"View factors: {entry.view_factor_file} — "
                + ("cached, will be loaded" if vf.exists()
                   else "not cached, will be built on the first run"))
        points = europa_sweep_points(root)
        terrains = len({p[0] for p in points})
        latitudes = len({p[1] for p in points})
        minutes = len(points) * _MINUTES_PER_RUN_GUESS
        self.sweep_summary.setText(
            f"{terrains} terrains x {latitudes} latitudes = {len(points)} runs."
            f" Rough estimate {minutes / 60.0:.0f} h in series on this machine;"
            f" the real figure depends on the mesh and the view factors.")

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

    def __init__(self, call, route, kind):
        super().__init__()
        self._call = call
        self.route = route
        self.kind = kind

    def run(self):
        from gui_core.body_fetch import FetchResult
        try:
            result = self._call()
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
        self.spk_fetch_button = QPushButton("Fetch kernel from HORIZONS")
        self.spk_fetch_button.setToolTip(get_tooltip("ui.fetch_spk"))
        self.spk_fetch_button.clicked.connect(self._on_fetch_spk)
        egrid.addWidget(self.spk_fetch_button, 1, 0, 1, 2)
        self.add_file_field(egrid, 2, "SPICE metakernel:",
                            "generic_body.spice_metakernel",
                            "Metakernels (*.tm);;Kernels (*.bsp);;All Files (*)")
        self.coverage_note = _muted("", color="#EF9A9A", size=10)
        egrid.addWidget(self.coverage_note, 3, 0, 1, 2)
        check = QPushButton("Check SPK coverage")
        check.setToolTip(get_tooltip("ui.spk_coverage"))
        check.clicked.connect(self._check_coverage)
        egrid.addWidget(check, 4, 1)
        orbit_button = QPushButton("Orbital elements on the Body page…")
        orbit_button.setToolTip(get_tooltip("ui.orbit_on_body_page"))
        orbit_button.clicked.connect(self._open_body_page)
        egrid.addWidget(orbit_button, 5, 1)
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
        self.damit_button = QPushButton("DAMIT shape model…")
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
        grid.addWidget(self.albedo_provenance, 9, 0, 1, 2)
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
        self.ti_note = _muted("")
        self.ti_note.setToolTip(get_tooltip("ui.thermal_inertia_manual"))
        grid.addWidget(self.ti_note, 7, 0, 1, 2)
        self.surface_refusal = _muted("", color="#EF9A9A", size=10)
        grid.addWidget(self.surface_refusal, 8, 0, 1, 2)
        surface.setLayout(grid)
        layout.addWidget(surface)
        for key in ("generic_body.surface.ks", "generic_body.surface.kd",
                    "generic_body.surface.rhos", "generic_body.surface.rhod",
                    "generic_body.surface.cp0"):
            self.widgets[key].valueChanged.connect(
                lambda _v: self._refresh_surface())

        layout.addWidget(self.time_group())
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
        self._refresh_shape()
        self._refresh_surface()
        self._refresh_provenance()

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
            logger.info("[body-fetch] discarding a %s result for %s; the "
                        "active route is now %s", result.kind, route,
                        self.route)
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
        self._refresh_shape()
        self._refresh_surface()
        if result.ok and result.values:
            self.value_changed.emit()

    def _on_fetch_spk(self):
        """Fetch a kernel spanning this run, with a year of margin each end."""
        from gui_core.body_fetch import fetch_spk

        name = str(self.config_manager.get_value(
            "generic_body.name", "") or "").strip()
        naif_id = str(self.config_manager.get_value(
            "generic_body.naif_id", "") or "").strip()
        config = self.config_manager.get_config()
        project_root = self._project_root()
        self.coverage_note.setText(
            f"Asking HORIZONS for a kernel covering this run, with a year of "
            f"margin at each end. Nothing is changed unless what arrives is "
            f"read back and shown to span the window.")
        self._start_fetch(
            "spk",
            lambda: fetch_spk(config, project_root, name=name,
                              naif_id=naif_id),
            self.spk_fetch_button, self.coverage_note)

    def _on_fetch_damit(self):
        """Fetch this body's DAMIT shape model, and the spin that goes with it."""
        from gui_core.body_fetch import fetch_damit_shape

        name = str(self.config_manager.get_value(
            "generic_body.name", "") or "").strip()
        naif_id = str(self.config_manager.get_value(
            "generic_body.naif_id", "") or "").strip()
        radius_km = self._number("generic_body.shape.radius_km", 0.0)
        project_root = self._project_root()
        self.shape_provenance.setText(
            "Asking DAMIT for a shape model. Most asteroids do not have one.")
        self._start_fetch(
            "damit_shape",
            lambda: fetch_damit_shape(project_root, name=name,
                                      naif_id=naif_id, radius_km=radius_km),
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
            lambda: fetch_sbdb_properties(query),
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

    @staticmethod
    def _grid_label(grid, row):
        item = grid.itemAtPosition(row, 0)
        widget = item.widget() if item is not None else None
        return widget if isinstance(widget, QLabel) else None

    def _on_ephemeris_changed(self, _checked):
        source = "keplerian" if self.kepler_radio.isChecked() else "spice"
        self.config_manager.set_value("generic_body.ephemeris_source", source)
        self.coverage_note.setText(
            "Keplerian elements, chosen deliberately. The run's own record "
            "will say so."
            if source == "keplerian" else
            "SPICE. Check the kernel's coverage below: when a requested epoch "
            "falls outside it the engine substitutes Keplerian elements for "
            "that epoch and logs nothing at all.")
        self.value_changed.emit()

    def _check_coverage(self):
        """Compare the kernel's own coverage with the whole run window.

        Three outcomes, kept distinct on purpose: covered, NOT covered, and
        UNKNOWN. Reporting "could not check" as "fine" is how fifteen runs
        completed on the silent per-epoch Keplerian fallback while recording
        SPICE, so an unanswerable question is answered as unanswerable.
        """
        from gui_core.body_cards import run_window, spk_coverage
        kernel = str(self.config_manager.get_value(
            "generic_body.spice_metakernel", "") or "")
        coverage = spk_coverage(kernel)
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
            notes.append(
                "shape models in .obj are excluded by this project's "
                ".gitignore, so this file is untracked and may exist in one "
                "copy only — back it up, or add an exception for it")
        notes.append(
            "asteroid shape models are conventionally in KILOMETERS while "
            "lunar and Europa terrain meshes are in meters; the run prints "
            "the units it inferred")
        self.shape_note.setText(". ".join(notes) + ".")

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
    ellipsis. A wrapping QLabel makes the button grow downward instead, and
    `sizeHint` then reports the height the text really needs.

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

    def __init__(self, title, subtitle="", tooltip=""):
        super().__init__("")
        self._title = title
        self.setMinimumHeight(96)
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
        box.setSpacing(4)
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
        card.activate(mode_key, run_type_key)
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
        # the lunar crater card and the Europa Sutton-DEM card both write
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
        if preset.needs_advanced:
            result.notes.append(
                "The fields listed above are not on this card. 'Advanced "
                "options' has been turned on and the sections holding them "
                "have been opened for you -- press 'All settings (advanced)' "
                "at the top of this card to get to them.")
            self.advanced_requested.emit(preset.key)
        return result

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
