"""The body-first surface: a landing screen, a run-type step, and one card.

Three screens, in this order (DP-10):

    which body?  ->  which kind of run?  ->  one card, for that body and that
                                             kind of run

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
  from the engine's own implemented-body table; "a grid of runs" appears only
  where a grid is actually backed by shipped configurations.
"""

import logging
from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
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

    def time_group(self, title="When"):
        """The three time controls every card carries."""
        group = QGroupBox(title)
        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        self.add_string_field(grid, 0, "Start (UTC):", "simulation.start_time")
        spin = self.add_double_field(
            grid, 1, "How long (days):", "simulation.duration_days",
            0.001, 365000.0, decimals=6)
        spin.setSingleStep(1.0)
        self.add_int_field(grid, 2, "Output samples:", "simulation.timesteps",
                           1, 100000)
        group.setLayout(grid)
        return group

    def output_group(self, title="Where it goes"):
        group = QGroupBox(title)
        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        self.add_directory_field(grid, 0, "Output folder:", "output.directory")
        self.add_bool_field(grid, 1, "Run the thermal solver on the GPU",
                            "thermal.use_gpu")
        group.setLayout(grid)
        return group

    def settle_group(self):
        group = QGroupBox("How long to settle")
        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        spin = self.add_double_field(
            grid, 0, "Spin-up (day/night cycles):",
            "thermal.equilibration_orbits", 0.0, 100000.0, decimals=2)
        spin.setSingleStep(1.0)
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

    def load_card(self):
        """Mirror the configuration into this card's controls."""
        skip = self.custom_loaded_keys()
        for key, widget in self.widgets.items():
            if key in skip:
                continue
            value = self.config_manager.get_value(key, None)
            if value is None:
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

        # -- whole-Moon geometry --------------------------------------------
        self.global_group = QGroupBox("Which sphere")
        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        self.add_int_field(grid, 0, "Level of detail:", "mesh.resolution", 0, 9)
        self.face_count = _muted("")
        grid.addWidget(self.face_count, 1, 0, 1, 2)
        self.add_bool_field(grid, 2, "Use real topography",
                            "mesh.use_topographic")
        self.topo_note = _muted("")
        grid.addWidget(self.topo_note, 3, 0, 1, 2)
        self.global_group.setLayout(grid)
        layout.addWidget(self.global_group)
        self.widgets["mesh.resolution"].valueChanged.connect(
            lambda _v: self._refresh_surface())
        self.widgets["mesh.use_topographic"].toggled.connect(
            lambda _v: self._refresh_surface())

        # -- a place on the Moon --------------------------------------------
        self.local_group = QGroupBox("Which piece of ground")
        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        self.add_file_field(grid, 0, "Terrain mesh:", "mesh.import_file",
                            "Meshes (*.ply *.obj *.vtk);;All Files (*)")
        self.add_file_field(grid, 1, "Shadow-only ring:", "mesh.occluder_file",
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
        self.add_bool_field(grid, 2, "Vary albedo with sun angle (Hapke)",
                            "thermophysical.incidence_angle_albedo")
        self.add_bool_field(grid, 3, "Use the bundled per-face property map",
                            "thermal.use_default_property_map")
        self.properties_note = _muted("")
        grid.addWidget(self.properties_note, 4, 0, 1, 2)
        surface.setLayout(grid)
        layout.addWidget(surface)

        # -- self-heating (local only) ---------------------------------------
        self.radiation_group = QGroupBox("Self-heating between facets")
        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        self.add_bool_field(grid, 0, "Exchange infrared between facets",
                            "radiation.enabled")
        self.add_bool_field(grid, 1, "Beam infrared from sub-facet roughness",
                            "radiation.beaming")
        self.beaming_note = _muted("")
        grid.addWidget(self.beaming_note, 2, 0, 1, 2)
        self.add_file_field(grid, 3, "Reuse view factors:",
                            "radiation.view_factor_file",
                            "View factors (*.vf *.csr *.hmat);;All Files (*)")
        self.radiation_group.setLayout(grid)
        layout.addWidget(self.radiation_group)
        self.widgets["radiation.beaming"].toggled.connect(
            self._on_beaming_toggled)

        layout.addWidget(self.settle_group())
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

        # -- whole Europa ----------------------------------------------------
        self.global_group = QGroupBox("Which sphere")
        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        self.add_int_field(grid, 0, "Level of detail:", "mesh.resolution", 0, 9)
        self.add_combo_field(grid, 1, "Albedo source:",
                             "illumination.albedo_source",
                             ["builtin", "custom", "none"])
        self.albedo_note = _muted("")
        grid.addWidget(self.albedo_note, 2, 0, 1, 2)
        self.add_double_field(grid, 3, "Integrate over (days):",
                              "thermal.integration_days", -1.0, 365000.0,
                              decimals=4)
        self.global_group.setLayout(grid)
        layout.addWidget(self.global_group)
        self.widgets["mesh.resolution"].valueChanged.connect(
            lambda _v: self._refresh())

        # -- a Sutton DEM site -----------------------------------------------
        self.local_group = QGroupBox("Which terrain")
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

        self.add_bool_field(grid, 3, "Exchange infrared between facets",
                            "radiation.enabled")
        self.vf_note = _muted("")
        grid.addWidget(self.vf_note, 4, 0, 1, 2)
        self.local_group.setLayout(grid)
        layout.addWidget(self.local_group)

        # -- the grid (run type "sweep") --------------------------------------
        self.sweep_group = QGroupBox("The whole grid")
        sweep_layout = QVBoxLayout()
        self.sweep_summary = _muted("", color="#64b5f6", size=11)
        sweep_layout.addWidget(self.sweep_summary)
        row = QHBoxLayout()
        self.sweep_button = QPushButton("Prepare the grid…")
        self.sweep_button.setToolTip(get_tooltip("ui.europa_sweep"))
        self.sweep_button.clicked.connect(self._on_prepare_sweep)
        row.addWidget(self.sweep_button)
        row.addStretch()
        sweep_layout.addLayout(row)
        self.sweep_group.setLayout(sweep_layout)
        layout.addWidget(self.sweep_group)

        # -- the standing caveat ----------------------------------------------
        caveat = _muted(
            "Europa's night side never sits at zero absorbed flux: Jupiter "
            "shines on it. Archived Europa results produced before the "
            "2026-08 surface-solver fix can be wrong by up to 12 K at "
            "mid-latitudes with no out-of-range value to notice, so check the "
            "engine build recorded beside any archived output before "
            "comparing against it.", color="#EF9A9A", size=10)
        layout.addWidget(caveat)

        layout.addWidget(self.settle_group())
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
            self, "Where should the grid's runs go?", "",
            QFileDialog.Option.ShowDirsOnly)
        if not directory:
            return
        minutes = len(points) * _MINUTES_PER_RUN_GUESS
        answer = QMessageBox.question(
            self, "Prepare the whole grid?",
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
    topography, no worked example -- and the card says so rather than implying
    the Moon's level of support.
    """

    def init_ui(self):
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        banner = _muted("", color="#FFCC80", size=11)
        banner.setText(
            "Mercury ships no albedo map, no topography grid and no worked "
            "example, so every run here is a uniform-albedo icosphere at the "
            "engine's built-in Mercury properties. The solver, the "
            "illumination and the 175.938-day solar day are the same "
            "machinery the Moon uses.")
        layout.addWidget(banner)

        layout.addWidget(self.time_group())

        sphere = QGroupBox("Which sphere")
        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        self.add_int_field(grid, 0, "Level of detail:", "mesh.resolution", 0, 9)
        self.face_count = _muted("")
        grid.addWidget(self.face_count, 1, 0, 1, 2)
        sphere.setLayout(grid)
        layout.addWidget(sphere)
        self.widgets["mesh.resolution"].valueChanged.connect(
            lambda _v: self._refresh())

        surface = QGroupBox("Surface")
        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        self.add_bool_field(grid, 0, "Vary albedo with sun angle (Hapke)",
                            "thermophysical.incidence_angle_albedo")
        self.properties_note = _muted("")
        grid.addWidget(self.properties_note, 1, 0, 1, 2)
        surface.setLayout(grid)
        layout.addWidget(surface)

        layout.addWidget(self.settle_group())
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

        identity = QGroupBox("Which body")
        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        self.add_string_field(grid, 0, "Name:", "generic_body.name")
        self.add_string_field(grid, 1, "NAIF id:", "generic_body.naif_id")
        identity.setLayout(grid)
        layout.addWidget(identity)

        # -- ephemeris, with the coverage check the project never had ---------
        ephemeris = QGroupBox("Where it is")
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
        self.add_file_field(egrid, 1, "SPICE metakernel:",
                            "generic_body.spice_metakernel",
                            "Metakernels (*.tm);;Kernels (*.bsp);;All Files (*)")
        self.coverage_note = _muted("", color="#EF9A9A", size=10)
        egrid.addWidget(self.coverage_note, 2, 0, 1, 2)
        check = QPushButton("Check SPK coverage")
        check.setToolTip(get_tooltip("ui.spk_coverage"))
        check.clicked.connect(self._check_coverage)
        egrid.addWidget(check, 3, 1)
        orbit_button = QPushButton("Orbital elements on the Body page…")
        orbit_button.setToolTip(get_tooltip("ui.orbit_on_body_page"))
        orbit_button.clicked.connect(self._open_body_page)
        egrid.addWidget(orbit_button, 4, 1)
        ephemeris.setLayout(egrid)
        layout.addWidget(ephemeris)

        # -- spin -------------------------------------------------------------
        spin = QGroupBox("How it turns")
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
        spin.setLayout(grid)
        layout.addWidget(spin)

        # -- shape -------------------------------------------------------------
        shape = QGroupBox("What shape it is")
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
        self.shape_note = _muted("")
        grid.addWidget(self.shape_note, 4, 0, 1, 2)
        shape.setLayout(grid)
        layout.addWidget(shape)
        self.widgets["generic_body.shape.mesh_file"].textChanged.connect(
            lambda _t: self._refresh_shape())
        self.widgets["generic_body.shape.mesh_scale"].valueChanged.connect(
            lambda _v: self._propagate_shape_mesh())

        # -- surface ------------------------------------------------------------
        surface = QGroupBox("What the surface is like")
        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        self.add_double_field(grid, 0, "Bond albedo:",
                              "generic_body.surface.albedo", 0.0, 1.0,
                              decimals=4)
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

        layout.addWidget(self.time_group("When to run it"))
        layout.addWidget(self.settle_group())
        layout.addWidget(self.output_group())
        self._make_report(layout)
        layout.addStretch()
        self.create_scroll_area(content)

    def load_card(self):
        super().load_card()
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
    def __init__(self, title, subtitle, badge="", tooltip=""):
        super().__init__("")
        self.setMinimumHeight(96)
        self.setMinimumWidth(180)
        text = title if not badge else f"{title}\n({badge})"
        self.setText(f"{text}\n{subtitle}")
        self.setToolTip(tooltip)
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        self.setFont(font)


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

        heading = QLabel("What would you like to model?")
        heading.setFont(QFont("Helvetica Neue", 15, QFont.Weight.Bold))
        layout.addWidget(heading)

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

        grid = QGridLayout()
        grid.setSpacing(10)
        for index, button in enumerate(buttons):
            widget = _BigButton(button.title, button.subtitle,
                                badge=button.maturity,
                                tooltip=get_tooltip("ui.body_button"))
            widget.clicked.connect(
                lambda _c=False, k=button.key: self.choose_body(k))
            grid.addWidget(widget, index // 3, index % 3)
            if button.note:
                note = _muted(button.note)
                grid.addWidget(note, 2 + index // 3, index % 3)
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

        settings_button = QPushButton("All settings…")
        settings_button.setToolTip(get_tooltip("ui.all_settings"))
        settings_button.clicked.connect(self.all_settings_requested.emit)
        row.addWidget(settings_button)
        row.addStretch()
        layout.addLayout(row)

        # The baseline entry keeps the task card's own report surface and its
        # beaming line, because what it promises is unusual and specific: it
        # overrides nothing the baseline recorded, including beaming.
        from gui_core.presets import PRESETS_BY_KEY
        baseline = PRESETS_BY_KEY["reproduce_baseline"]
        layout.addWidget(_muted(baseline.summary))
        layout.addWidget(_muted("Beaming: unchanged. " + baseline.beaming_note))
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

        back = QPushButton("← bodies")
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

        self.mode_group = QGroupBox("Which part of it")
        self.mode_layout = QVBoxLayout()
        self.mode_group.setLayout(self.mode_layout)
        layout.addWidget(self.mode_group)
        self._mode_radios = []

        self.runtype_group = QGroupBox("What kind of run")
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
        back = QPushButton("← run type")
        back.setToolTip(get_tooltip("ui.back_to_runtype"))
        back.clicked.connect(lambda: self.stack.setCurrentIndex(self.RUNTYPE))
        layout.addWidget(back)
        self.card_title = QLabel("")
        self.card_title.setFont(QFont("Helvetica Neue", 13, QFont.Weight.Bold))
        layout.addWidget(self.card_title)
        layout.addStretch()
        settings = QPushButton("All settings…")
        settings.setToolTip(get_tooltip("ui.all_settings"))
        settings.clicked.connect(self.all_settings_requested.emit)
        layout.addWidget(settings)
        return header

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
        self.runtype_heading.setText(f"{body_key} — how should it be run?")

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
        self._route = (body_key, mode_key, run_type_key)

        result = None
        if apply_defaults:
            result = self._apply_route_defaults(body_key, mode_key,
                                                run_type_key)
        card.activate(mode_key, run_type_key)
        self.card_stack.setCurrentWidget(card)
        run_type = RUN_TYPES_BY_KEY[run_type_key]
        self.card_title.setText(f"{body_key} — {run_type.title}")
        self.stack.setCurrentIndex(self.CARD)
        preset = card_preset(body_key, mode_key)
        if result is not None and preset is not None:
            card.show_report("\n".join(result.summary_lines()))
            self.preset_applied.emit(preset.key)
        self.route_changed.emit(body_key, mode_key or "", run_type_key)
        self.value_changed.emit()
        return card

    def _apply_route_defaults(self, body_key, mode_key, run_type_key):
        from gui_core.body_cards import RUN_TYPES_BY_KEY, card_preset
        from gui_core.presets import apply_preset
        preset = card_preset(body_key, mode_key)
        if preset is None:
            return None
        result = apply_preset(self.config_manager, preset, self._project_root())
        for key, value in RUN_TYPES_BY_KEY[run_type_key].sets.items():
            self.config_manager.set_value(key, value)
            result.applied[key] = value
        if preset.needs_advanced:
            result.notes.append(
                "'Advanced options' has been turned on and the sections "
                "holding the fields above have been opened.")
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
                "NOTE: 'Advanced options' has been turned on and the sections "
                "holding the fields above have been opened.")
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
