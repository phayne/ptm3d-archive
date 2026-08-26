"""
Configuration Panel for PTM3D GUI
Tabbed interface for simulation parameter configuration
"""

import logging
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QScrollArea,
    QGroupBox, QGridLayout, QLabel, QLineEdit, QSpinBox, QDoubleSpinBox,
    QComboBox, QCheckBox, QPushButton, QFileDialog, QTextEdit, QSplitter,
    QStackedWidget, QListWidget
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QCursor

from gui_core.config_manager import ConfigManager

logger = logging.getLogger(__name__)

# Import tooltips - use try/except for robustness
try:
    from gui_tooltips import (TOOLTIPS, get_tooltip,
                              BODY_SYNODIC_PERIODS, compute_synodic_day_days,
                              get_synodic_period_days,
                              BODY_ORBITAL_PERIODS, get_orbital_period_days,
                              compute_obliquity_deg, compute_pole_for_obliquity,
                              compute_equilibrium_temperature, compute_cp_at_temperature,
                              compute_thermal_inertia)
except ImportError:
    try:
        # Try parent directory import (when running from gui/gui_panels)
        import sys
        from pathlib import Path
        parent_dir = str(Path(__file__).parent.parent)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        from gui_tooltips import (TOOLTIPS, get_tooltip,
                                  BODY_SYNODIC_PERIODS, compute_synodic_day_days,
                                  get_synodic_period_days,
                                  BODY_ORBITAL_PERIODS, get_orbital_period_days,
                                  compute_obliquity_deg, compute_pole_for_obliquity,
                                  compute_equilibrium_temperature, compute_cp_at_temperature,
                                  compute_thermal_inertia)
    except ImportError:
        TOOLTIPS = {}
        BODY_SYNODIC_PERIODS = {"Europa": 3.55}
        BODY_ORBITAL_PERIODS = {"Europa": 3.55}
        def get_tooltip(key):
            return ""
        def compute_synodic_day_days(config_manager):
            return None
        def get_synodic_period_days(config_manager, body_name=None):
            return 3.55
        def get_orbital_period_days(config_manager, body_name=None):
            return 3.55
        def compute_obliquity_deg(pole_ra, pole_dec, inc, lan):
            return None
        def compute_pole_for_obliquity(obl, ra, dec, inc, lan):
            return (ra, dec)
        def compute_equilibrium_temperature(albedo, emissivity, rAU):
            return 270.0
        def compute_cp_at_temperature(T, cp0, cpCoeff):
            return max(cp0, 800.0)
        def compute_thermal_inertia(ks, rhos, cp):
            import math
            return math.sqrt(max(0.0, ks * rhos * cp))


class CollapsibleSection(QWidget):
    """Collapsible section for grouping related parameters."""

    def __init__(self, title, parent=None, collapsed=False):
        super().__init__(parent)
        self.title = title
        self._collapsed = collapsed

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 2, 0, 2)
        main_layout.setSpacing(0)

        self.toggle_button = QPushButton(
            f"{'▶' if collapsed else '▼'} {title}"
        )
        self.toggle_button.setObjectName("collapsibleHeader")
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(collapsed)
        self.toggle_button.clicked.connect(self.toggle)
        main_layout.addWidget(self.toggle_button)

        self.content_area = QWidget()
        self.content_layout = QVBoxLayout(self.content_area)
        self.content_layout.setContentsMargins(10, 5, 10, 10)
        self.content_layout.setSpacing(8)
        main_layout.addWidget(self.content_area)

        self.content_area.setVisible(not collapsed)

    def toggle(self):
        self._collapsed = not self._collapsed
        symbol = "▶" if self._collapsed else "▼"
        self.toggle_button.setText(f"{symbol} {self.title}")
        self.content_area.setVisible(not self._collapsed)

    def set_collapsed(self, collapsed):
        if self._collapsed != collapsed:
            self.toggle()

    def add_widget(self, widget):
        self.content_layout.addWidget(widget)

    def add_layout(self, layout):
        self.content_layout.addLayout(layout)


# Alias tables for the enum keys the GUI exposes as combo boxes. The engine
# normalizes each string (lowercase, `_`/`-`/space stripped) and accepts
# synonyms, so a plain findText silently displays the WRONG value for a config
# that used one -- e.g. `spinup_seed: historical` showing as "physical" while
# the engine runs the legacy seed. Keys here are the normalized spellings.
#   src/cpp/thermal/eq_seed.cpp parseEqSeedMode / parseEqSeedAnchor
#   src/cpp/thermal/conductivity_model.h parseConductivityModel
#   src/cpp/main/ptm_thermal_sim.cpp:14151 (coupling_mode -- exact, no aliases)
SPINUP_SEED_ALIASES = {
    "": "physical", "physical": "physical", "default": "physical",
    "perface": "physical",
    "legacy": "legacy", "historical": "legacy",
}
SEED_ANCHOR_ALIASES = {
    "": "lumped", "lumped": "lumped", "default": "lumped", "derived": "lumped",
    "warm": "warm", "warmbracket": "warm", "t4": "warm", "legacy": "warm",
}
CONDUCTIVITY_MODEL_ALIASES = {
    "": "hayne2017", "hayne": "hayne2017", "hayne2017": "hayne2017",
    "default": "hayne2017",
    "martinez": "martinez2021", "martinez2021": "martinez2021",
    "martinezsiegler2021": "martinez2021",
}
# The engine compares this one EXACTLY and makes a mismatch a startup FATAL,
# so the GUI must not invent aliases the engine would reject.
COUPLING_MODE_ALIASES = {
    "bounded": "bounded", "fixed": "fixed", "legacy": "legacy",
}


class BaseConfigTab(QWidget):
    """Base class for configuration tabs"""

    value_changed = pyqtSignal()

    def __init__(self, config_manager):
        super().__init__()
        self.config_manager = config_manager
        self.widgets = {}
        #: Sections this tab declares Advanced-tier (item 2.1 / DP-9). The
        #: Basic/Advanced boundary used to be drawn by TAB, which meant the
        #: Basic tier could not express a crater run at all: Radiation and
        #: Illumination were hidden wholesale. It is now drawn by FIELD -- every
        #: tab shows its Basic fields inline and hides these until the user asks
        #: for them. Purely a visibility filter: every widget still exists, and
        #: nothing about the saved YAML depends on it.
        self.advanced_sections = []
        self.init_ui()

    def advanced_section(self, title, collapsed=True):
        """A CollapsibleSection registered as Advanced-tier for this tab."""
        section = CollapsibleSection(title, collapsed=collapsed)
        self.advanced_sections.append(section)
        return section

    def set_advanced_visible(self, visible):
        """Show or hide every Advanced-tier section on this tab."""
        for section in self.advanced_sections:
            section.setVisible(bool(visible))

    def init_ui(self):
        """Initialize UI - implemented by subclasses"""
        raise NotImplementedError

    def _apply_tooltip(self, widget, info_key):
        """Apply tooltip from info_key to widget with visual indicator."""
        if not info_key:
            return
        tooltip_text = get_tooltip(info_key)
        if tooltip_text:
            widget.setToolTip(tooltip_text)
            widget.setCursor(QCursor(Qt.CursorShape.WhatsThisCursor))
            widget.setStyleSheet("color: #64B5F6;")

    def create_scroll_area(self, content_widget):
        """Wrap content in a scroll area"""
        scroll = QScrollArea()
        scroll.setWidget(content_widget)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(scroll)

    def add_string_field(self, layout, row, label_text, key, tooltip="", info_key=None):
        """Add a string input field"""
        label = QLabel(label_text)
        label.setToolTip(tooltip)
        self._apply_tooltip(label, info_key)

        widget = QLineEdit()
        widget.setToolTip(tooltip)
        widget.textChanged.connect(lambda: self.on_widget_changed(key, widget.text()))

        layout.addWidget(label, row, 0)
        layout.addWidget(widget, row, 1)
        self.widgets[key] = widget
        return widget

    def add_int_field(self, layout, row, label_text, key, min_val=0, max_val=10000, tooltip="", info_key=None):
        """Add an integer input field"""
        label = QLabel(label_text)
        label.setToolTip(tooltip)
        self._apply_tooltip(label, info_key)

        widget = QSpinBox()
        widget.setRange(min_val, max_val)
        widget.setToolTip(tooltip)
        widget.valueChanged.connect(lambda: self.on_widget_changed(key, widget.value()))

        layout.addWidget(label, row, 0)
        layout.addWidget(widget, row, 1)
        self.widgets[key] = widget
        return widget

    def add_double_field(self, layout, row, label_text, key, min_val=0.0, max_val=10000.0,
                         decimals=2, tooltip="", info_key=None):
        """Add a double input field"""
        label = QLabel(label_text)
        label.setToolTip(tooltip)
        self._apply_tooltip(label, info_key)

        widget = QDoubleSpinBox()
        widget.setRange(min_val, max_val)
        widget.setDecimals(decimals)
        widget.setToolTip(tooltip)
        widget.valueChanged.connect(lambda: self.on_widget_changed(key, widget.value()))

        layout.addWidget(label, row, 0)
        layout.addWidget(widget, row, 1)
        self.widgets[key] = widget
        return widget

    def add_combo_field(self, layout, row, label_text, key, options, tooltip="", info_key=None):
        """Add a combo box field"""
        label = QLabel(label_text)
        label.setToolTip(tooltip)
        self._apply_tooltip(label, info_key)

        widget = QComboBox()
        widget.addItems(options)
        widget.setToolTip(tooltip)
        widget.currentTextChanged.connect(lambda: self.on_widget_changed(key, widget.currentText()))

        layout.addWidget(label, row, 0)
        layout.addWidget(widget, row, 1)
        self.widgets[key] = widget
        return widget

    def add_bool_field(self, layout, row, label_text, key, tooltip="", info_key=None):
        """Add a checkbox field"""
        widget = QCheckBox(label_text)
        widget.setToolTip(tooltip)
        self._apply_tooltip(widget, info_key)
        widget.stateChanged.connect(lambda: self.on_widget_changed(key, widget.isChecked()))

        layout.addWidget(widget, row, 0, 1, 2)
        self.widgets[key] = widget
        return widget

    def add_file_field(self, layout, row, label_text, key, file_filter="All Files (*)", tooltip="", info_key=None):
        """Add a file selector field"""
        label = QLabel(label_text)
        label.setToolTip(tooltip)
        self._apply_tooltip(label, info_key)

        container = QWidget()
        h_layout = QHBoxLayout(container)
        h_layout.setContentsMargins(0, 0, 0, 0)

        widget = QLineEdit()
        widget.setToolTip(tooltip)
        widget.textChanged.connect(lambda: self.on_widget_changed(key, widget.text()))

        browse_btn = QPushButton("...")
        browse_btn.setMaximumWidth(40)
        browse_btn.clicked.connect(lambda k=key: self.browse_file(widget, file_filter, k))

        h_layout.addWidget(widget)
        h_layout.addWidget(browse_btn)

        layout.addWidget(label, row, 0)
        layout.addWidget(container, row, 1)
        self.widgets[key] = widget
        return widget

    def add_directory_field(self, layout, row, label_text, key, tooltip="", info_key=None):
        """Add a directory selection field"""
        label = QLabel(label_text)
        label.setToolTip(tooltip)
        self._apply_tooltip(label, info_key)

        container = QWidget()
        h_layout = QHBoxLayout(container)
        h_layout.setContentsMargins(0, 0, 0, 0)

        widget = QLineEdit()
        widget.setToolTip(tooltip)
        widget.textChanged.connect(lambda: self.on_widget_changed(key, widget.text()))

        browse_btn = QPushButton("Browse...")
        browse_btn.setMaximumWidth(100)
        browse_btn.clicked.connect(lambda k=key: self.browse_directory(widget, k))

        h_layout.addWidget(widget)
        h_layout.addWidget(browse_btn)

        layout.addWidget(label, row, 0)
        layout.addWidget(container, row, 1)
        self.widgets[key] = widget
        return widget

    def _get_default_browse_dir(self, key):
        """Get a logical default directory based on the config key"""
        # Map keys to default directories relative to project root
        default_dirs = {
            "output.locations_file": "data/locations",
            "output.depths_file": "data/locations",
            "mesh.import_file": "data/meshes",
            "mesh.topo_directory": "data/topo",
            "illumination.albedo_map": "data/albedo_maps",
            "illumination.albedo_maps_directory": "data/albedo_maps",
            "thermal.thermal_inertia_map": "data/thermal_inertia",
            "thermal.initial_state_file": "output",
            "thermal.final_state_file": "output",
            "mesh.occluder_file": "data/meshes",
            "radiation.view_factor_file": "output",
            "output.directory": "output",
        }
        return default_dirs.get(key, ".")

    def browse_file(self, line_edit, file_filter, key=""):
        """Open file browser dialog"""
        start_dir = line_edit.text() if line_edit.text() else self._get_default_browse_dir(key)
        filename, _ = QFileDialog.getOpenFileName(
            self, "Select File",
            start_dir,
            file_filter
        )
        if filename:
            line_edit.setText(filename)

    def browse_directory(self, line_edit, key=""):
        """Open directory selection dialog"""
        start_dir = line_edit.text() if line_edit.text() else self._get_default_browse_dir(key)
        directory = QFileDialog.getExistingDirectory(
            self, "Select Directory",
            start_dir,
            QFileDialog.Option.ShowDirsOnly
        )
        if directory:
            line_edit.setText(directory)

    def on_widget_changed(self, key, value):
        """Handle widget value changes"""
        self.config_manager.set_value(key, value)
        self.value_changed.emit()

    def set_bool(self, key, value):
        """Load a boolean into its checkbox, tolerating YAML's spellings.

        `beaming: 'true'` -- a QUOTED boolean -- is a str after
        yaml.safe_load, and yaml-cpp's `as<bool>` accepts it, so the engine runs
        that config. `setChecked(str)` raises, out of load_from_config, which is
        an application abort. The sibling of set_numeric for the same reason.
        """
        widget = self.widgets.get(key)
        if widget is None:
            return
        if isinstance(value, bool):
            widget.setChecked(value)
            return
        text = str(value).strip().lower()
        # The set yaml-cpp's bool decoder accepts.
        if text in ("true", "yes", "on", "1"):
            widget.setChecked(True)
        elif text in ("false", "no", "off", "0", "", "none"):
            widget.setChecked(False)
        else:
            logging.warning(
                "[config] %s = %r is not a boolean; leaving the control at %r",
                key, value, widget.isChecked())
            return
        self.config_manager.set_value(key, widget.isChecked())

    def set_text(self, key, value):
        """Load a string into its line edit. A non-string is shown, not thrown."""
        widget = self.widgets.get(key)
        if widget is None:
            return
        if value is None:
            widget.setText("")
            return
        if not isinstance(value, str):
            logging.warning(
                "[config] %s = %r is not a string; showing its text form",
                key, value)
        widget.setText(str(value))

    def set_enum(self, key, value, aliases):
        """Select a combo entry, tolerating the engine's own aliases.

        The engine normalizes these strings -- lowercase, `_`/`-`/space
        stripped -- and accepts synonyms: `historical` is `legacy`,
        `perface`/`default` are `physical`, `derived` is `lumped`, `martinez` is
        `martinez2021`. A plain ``findText`` misses every one of them, and
        `if idx >= 0` then leaves the combo showing a DIFFERENT value than the
        one that will run: a config saying `spinup_seed: historical` displayed
        "physical" while the engine used the legacy seed.

        `aliases` maps a normalized spelling to the combo entry it means.
        An unrecognized value is left in the config (the engine will diagnose
        it, and it is not this panel's job to silently rewrite one) and warned
        about, never quietly displayed as something else.
        """
        widget = self.widgets.get(key)
        if widget is None:
            return
        normalized = "".join(
            c for c in str(value).lower() if c not in "_- ")
        canonical = aliases.get(normalized)
        if canonical is None:
            logging.warning(
                "[config] %s = %r is not a value this control offers (%s). The "
                "configuration keeps it and the engine will decide; the control "
                "is showing %r, which is NOT what will run.",
                key, value, "/".join(aliases.values() and
                                     sorted(set(aliases.values()))),
                widget.currentText())
            return
        idx = widget.findText(canonical)
        if idx >= 0:
            widget.setCurrentIndex(idx)
        if str(value) != canonical:
            # Normalize the stored value too, so the saved YAML says what the
            # control shows. Only ever an alias-for-alias substitution.
            self.config_manager.set_value(key, canonical)

    def set_numeric(self, key, value):
        """Load a numeric value into its widget, WARNING if the range clamps it.

        QSpinBox/QDoubleSpinBox.setValue clamps silently and still emits
        valueChanged, so loading a config whose value falls outside a widget's
        range both displayed AND wrote back the clamped number: the run then
        used a value the file never asked for, with nothing said. The one that
        matters most is thermal.geothermal_heat_flow, where an out-of-band value
        is a plausible thing for a sensitivity study to carry.
        """
        widget = self.widgets.get(key)
        if widget is None:
            return
        try:
            number = float(value)
        except (TypeError, ValueError):
            logging.warning("[config] %s = %r is not a number; leaving the "
                            "control at its current value", key, value)
            return
        # NaN and the infinities ARE floats, so float() accepts them -- and
        # `int(nan)` raises ValueError while `int(inf)` raises OverflowError,
        # out of load_from_config, which is a process abort. YAML spells them
        # `.nan`, `.inf`, `-.inf`; the engine's std::stod reads them too.
        if number != number or number in (float("inf"), float("-inf")):
            logging.warning("[config] %s = %r is not a finite number; leaving "
                            "the control at its current value", key, value)
            return
        low, high = widget.minimum(), widget.maximum()
        if number < low or number > high:
            logging.warning(
                "[config] %s = %s is outside the range this control offers "
                "(%s to %s). The control shows the clamped value, and the "
                "configuration KEEPS %s -- edit the field to change it.",
                key, value, low, high, value)
            widget.blockSignals(True)
            widget.setValue(type(widget.value())(min(max(number, low), high)))
            widget.blockSignals(False)
            self.config_manager.set_value(key, value)
            return
        coerced = type(widget.value())(number)
        widget.setValue(coerced)
        # Write the COERCED value back explicitly. Relying on setValue's
        # valueChanged signal leaves a string in the config whenever the number
        # happens to equal the widget's current value (no change, no signal) --
        # and YAML resolves ordinary spellings like `-1e0` and `5e3` as str.
        # Every later `<` or `<=` on that key then raises TypeError, which out
        # of a Qt slot is an application abort.
        self.config_manager.set_value(key, coerced)

    def load_from_config(self):
        """Load values from config manager - implemented by subclasses"""
        raise NotImplementedError


class SimulationTab(BaseConfigTab):
    """Simulation parameters tab"""

    # Separator label used to visually separate built-in from HORIZONS bodies
    SEPARATOR = "──────────"
    CUSTOM_BODY = "Custom Body..."
    BROWSE_HORIZONS = "Browse HORIZONS..."

    # Standard built-in bodies (Moon first: the default and best-validated body)
    BUILTIN_BODIES = ["Moon", "Europa", "Mercury", "Enceladus", "Titan", "Mars", "Ceres"]

    def init_ui(self):
        self._syncing = False
        self._current_synodic_days = 3.55  # Default: Europa

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        # Time Parameters
        time_group = QGroupBox("Time Parameters")
        time_layout = QGridLayout()
        time_layout.setColumnStretch(1, 1)

        # Row 0: Start Time (unchanged)
        self.add_string_field(time_layout, 0, "Start Time:", "simulation.start_time",
                              "ISO 8601 format (e.g., 2025-01-01T00:00:00)",
                              info_key="simulation.start_time")

        # Row 1: Duration with unit selector
        duration_label = QLabel("Duration:")
        self._apply_tooltip(duration_label, "simulation.duration_days")
        time_layout.addWidget(duration_label, 1, 0)

        duration_container = QWidget()
        dur_hlayout = QHBoxLayout(duration_container)
        dur_hlayout.setContentsMargins(0, 0, 0, 0)
        dur_hlayout.setSpacing(4)

        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(0.001, 365000.0)
        self.duration_spin.setDecimals(4)
        self.duration_spin.setSingleStep(1.0)
        self.duration_spin.setToolTip("Simulation duration")
        self.duration_spin.valueChanged.connect(self._on_duration_spin_changed)
        dur_hlayout.addWidget(self.duration_spin, stretch=1)

        self.duration_unit_combo = QComboBox()
        self.duration_unit_combo.addItems(["days", "hours", "Earth years", "synodic periods"])
        self.duration_unit_combo.setMaximumWidth(130)
        self.duration_unit_combo.setToolTip("Duration display unit")
        self.duration_unit_combo.currentTextChanged.connect(self._on_duration_unit_changed)
        dur_hlayout.addWidget(self.duration_unit_combo)

        time_layout.addWidget(duration_container, 1, 1)
        self.widgets["simulation.duration_days"] = self.duration_spin

        # Row 2: End Time (computed, editable)
        end_label = QLabel("End Time (UTC):")
        self._apply_tooltip(end_label, "simulation.end_time")
        time_layout.addWidget(end_label, 2, 0)

        self.end_time_edit = QLineEdit()
        self.end_time_edit.setToolTip("Simulation end time — edit to adjust duration")
        self.end_time_edit.setPlaceholderText("Auto-computed from start + duration")
        self.end_time_edit.editingFinished.connect(self._on_end_time_edited)
        time_layout.addWidget(self.end_time_edit, 2, 1)

        # Row 3: Time Steps
        self.add_int_field(time_layout, 3, "Time Steps:", "simulation.timesteps",
                           1, 100000, "Number of output timesteps",
                           info_key="simulation.timesteps")
        # Connect timesteps widget to sync step size
        self.widgets["simulation.timesteps"].valueChanged.connect(
            self._on_timesteps_changed)

        # Row 4: Time Step Size (alternative to number of steps)
        step_label = QLabel("Step Size:")
        step_label.setToolTip("Output cadence — editing this updates Time Steps, and vice versa")
        time_layout.addWidget(step_label, 4, 0)

        step_container = QWidget()
        step_hlayout = QHBoxLayout(step_container)
        step_hlayout.setContentsMargins(0, 0, 0, 0)
        step_hlayout.setSpacing(4)

        self.step_size_spin = QDoubleSpinBox()
        self.step_size_spin.setRange(0.001, 1e7)
        self.step_size_spin.setDecimals(3)
        self.step_size_spin.setSingleStep(1.0)
        self.step_size_spin.setValue(1.0)
        self.step_size_spin.setToolTip("Output time step size")
        self.step_size_spin.valueChanged.connect(self._on_step_size_changed)
        step_hlayout.addWidget(self.step_size_spin, stretch=1)

        self.step_unit_combo = QComboBox()
        self.step_unit_combo.addItems(["minutes", "seconds", "hours"])
        self.step_unit_combo.setMaximumWidth(100)
        self.step_unit_combo.setToolTip("Step size unit")
        self.step_unit_combo.currentTextChanged.connect(self._on_step_unit_changed)
        step_hlayout.addWidget(self.step_unit_combo)

        time_layout.addWidget(step_container, 4, 1)

        time_group.setLayout(time_layout)
        layout.addWidget(time_group)

        # Planetary Body
        body_group = QGroupBox("Planetary Body")
        body_layout = QGridLayout()
        body_layout.setColumnStretch(1, 1)

        bodies = list(self.BUILTIN_BODIES) + [self.SEPARATOR, self.CUSTOM_BODY, self.BROWSE_HORIZONS]
        self.add_combo_field(body_layout, 0, "Target Body:", "simulation.body", bodies,
                             "Select the planetary body to simulate",
                             info_key="simulation.body")

        # Make separator unselectable
        combo = self.widgets["simulation.body"]
        sep_idx = combo.findText(self.SEPARATOR)
        if sep_idx >= 0:
            combo.model().item(sep_idx).setEnabled(False)

        # Intercept body selection to handle special items
        combo.currentTextChanged.connect(self._on_body_combo_changed)

        # Generic body config path label (shown when HORIZONS body selected)
        self.generic_body_label = QLabel("")
        self.generic_body_label.setStyleSheet("color: #64b5f6; font-size: 10px;")
        self.generic_body_label.setVisible(False)
        body_layout.addWidget(self.generic_body_label, 1, 0, 1, 3)

        body_group.setLayout(body_layout)
        layout.addWidget(body_group)

        layout.addStretch()
        self.create_scroll_area(content)

        # Initialize step size from current duration + timesteps
        self._sync_step_size_from_config()

    def _sync_step_size_from_config(self):
        """Initialize step size display from current config values."""
        # Coerced, because a YAML `5e3` or a quoted number loads as str and a
        # bare `>` here raises out of init_ui -- which in the GUI is a process
        # abort, not a message.
        days = ConfigManager._as_number(
            self.config_manager.get_value("simulation.duration_days", 3.55), 0.0)
        n_steps = ConfigManager._as_number(
            self.config_manager.get_value("simulation.timesteps", 144), 0.0)
        if n_steps > 0 and days > 0:
            step_sec = (days * 86400.0) / n_steps
            self.step_size_spin.blockSignals(True)
            self.step_size_spin.setValue(self._seconds_to_step_display(step_sec))
            self.step_size_spin.blockSignals(False)

    # ── Duration / End-Time synchronization ──────────────────────────

    def on_widget_changed(self, key, value):
        """Override to sync end time when start time changes."""
        super().on_widget_changed(key, value)
        if self._syncing:
            return
        if key == "simulation.start_time":
            self._sync_end_time_from_start_and_duration()

    def _get_synodic_period_days(self, body_name=None):
        """Return the synodic day (solar day) in Earth days for the given body."""
        return get_synodic_period_days(self.config_manager, body_name)

    def _days_to_display(self, days):
        """Convert duration in days to the currently selected display unit."""
        unit = self.duration_unit_combo.currentText()
        if unit == "hours":
            return days * 24.0
        elif unit == "Earth years":
            return days / 365.25
        elif unit == "synodic periods":
            return days / self._current_synodic_days if self._current_synodic_days > 0 else days
        return days  # "days"

    def _display_to_days(self, value):
        """Convert displayed value back to days using the current unit."""
        unit = self.duration_unit_combo.currentText()
        if unit == "hours":
            return value / 24.0
        elif unit == "Earth years":
            return value * 365.25
        elif unit == "synodic periods":
            return value * self._current_synodic_days
        return value  # "days"

    def _update_spin_range_for_unit(self, unit):
        """Adjust spinbox range, decimals, and step size for the selected unit."""
        self.duration_spin.blockSignals(True)
        if unit == "hours":
            self.duration_spin.setRange(0.1, 365000.0 * 24)
            self.duration_spin.setDecimals(2)
            self.duration_spin.setSingleStep(1.0)
        elif unit == "Earth years":
            self.duration_spin.setRange(0.0001, 1000.0)
            self.duration_spin.setDecimals(6)
            self.duration_spin.setSingleStep(0.1)
        elif unit == "synodic periods":
            self.duration_spin.setRange(0.001, 100000.0)
            self.duration_spin.setDecimals(4)
            self.duration_spin.setSingleStep(0.5)
        else:  # days
            self.duration_spin.setRange(0.001, 365000.0)
            self.duration_spin.setDecimals(4)
            self.duration_spin.setSingleStep(1.0)
        self.duration_spin.blockSignals(False)

    def _on_duration_spin_changed(self, display_value):
        """Handle duration spinbox value change."""
        if self._syncing:
            return
        days = self._display_to_days(display_value)
        self.config_manager.set_value("simulation.duration_days", days)
        self.value_changed.emit()
        self._sync_end_time_from_start_and_duration()
        # Also update step size display (timesteps stays fixed, step size adjusts)
        n_steps = self.config_manager.get_value("simulation.timesteps", 144)
        if n_steps > 0 and days > 0:
            step_sec = (days * 86400.0) / n_steps
            self.step_size_spin.blockSignals(True)
            self.step_size_spin.setValue(self._seconds_to_step_display(step_sec))
            self.step_size_spin.blockSignals(False)

    def _on_duration_unit_changed(self, unit_text):
        """Handle unit combo selection change — re-display in new unit."""
        if self._syncing:
            return
        self._syncing = True
        try:
            days = self.config_manager.get_value("simulation.duration_days", 3.55)
            self._update_spin_range_for_unit(unit_text)
            display_val = self._days_to_display(days)
            self.duration_spin.blockSignals(True)
            self.duration_spin.setValue(display_val)
            self.duration_spin.blockSignals(False)
        finally:
            self._syncing = False

    def _on_end_time_edited(self):
        """Handle user editing the end time field — update duration."""
        if self._syncing:
            return
        from datetime import datetime

        end_str = self.end_time_edit.text().strip()
        start_str = self.config_manager.get_value("simulation.start_time", "")

        try:
            start_dt = datetime.fromisoformat(start_str)
            end_dt = datetime.fromisoformat(end_str)
            delta = end_dt - start_dt
            days = delta.total_seconds() / 86400.0

            if days <= 0:
                self._sync_end_time_from_start_and_duration()
                return

            self._syncing = True
            try:
                self.config_manager.set_value("simulation.duration_days", days)
                display_val = self._days_to_display(days)
                self.duration_spin.blockSignals(True)
                self.duration_spin.setValue(display_val)
                self.duration_spin.blockSignals(False)
                self.value_changed.emit()
            finally:
                self._syncing = False
        except ValueError:
            self._sync_end_time_from_start_and_duration()

    def _sync_end_time_from_start_and_duration(self):
        """Compute end time from start + duration and update the field."""
        from datetime import datetime, timedelta

        self._syncing = True
        try:
            start_str = self.config_manager.get_value("simulation.start_time", "")
            days = self.config_manager.get_value("simulation.duration_days", 3.55)
            try:
                start_dt = datetime.fromisoformat(start_str)
                # OverflowError: timedelta(days=inf) and a duration far past
                # datetime.MAX both raise, and this runs from
                # load_from_config -- an abort, not a message.
                end_dt = start_dt + timedelta(days=ConfigManager._as_number(days, 0.0))
                self.end_time_edit.blockSignals(True)
                self.end_time_edit.setText(end_dt.strftime("%Y-%m-%dT%H:%M:%S"))
                self.end_time_edit.blockSignals(False)
            except (ValueError, TypeError, OverflowError):
                self.end_time_edit.blockSignals(True)
                self.end_time_edit.setText("")
                self.end_time_edit.blockSignals(False)
        finally:
            self._syncing = False

    # ── Time Steps ↔ Step Size synchronization ─────────────────────

    def _step_size_to_seconds(self, value=None):
        """Convert the step size spinbox value to seconds using current unit."""
        if value is None:
            value = self.step_size_spin.value()
        unit = self.step_unit_combo.currentText()
        if unit == "seconds":
            return value
        elif unit == "hours":
            return value * 3600.0
        else:  # minutes
            return value * 60.0

    def _seconds_to_step_display(self, seconds):
        """Convert seconds to the current step-size display unit."""
        unit = self.step_unit_combo.currentText()
        if unit == "seconds":
            return seconds
        elif unit == "hours":
            return seconds / 3600.0
        else:  # minutes
            return seconds / 60.0

    def _on_timesteps_changed(self, n_steps):
        """When timesteps changes, update the step size display."""
        if self._syncing:
            return
        days = self.config_manager.get_value("simulation.duration_days", 3.55)
        if n_steps > 0 and days > 0:
            step_sec = (days * 86400.0) / n_steps
            self._syncing = True
            self.step_size_spin.blockSignals(True)
            self.step_size_spin.setValue(self._seconds_to_step_display(step_sec))
            self.step_size_spin.blockSignals(False)
            self._syncing = False

    def _on_step_size_changed(self, display_value):
        """When step size changes, update the timesteps count."""
        if self._syncing:
            return
        step_sec = self._step_size_to_seconds(display_value)
        if step_sec <= 0:
            return
        days = self.config_manager.get_value("simulation.duration_days", 3.55)
        total_sec = days * 86400.0
        n_steps = max(1, round(total_sec / step_sec))

        self._syncing = True
        try:
            ts_widget = self.widgets.get("simulation.timesteps")
            if ts_widget:
                ts_widget.blockSignals(True)
                ts_widget.setValue(n_steps)
                ts_widget.blockSignals(False)
            self.config_manager.set_value("simulation.timesteps", n_steps)
            self.value_changed.emit()
        finally:
            self._syncing = False

    def _on_step_unit_changed(self, unit_text):
        """When the step unit changes, re-display the step size in the new unit."""
        if self._syncing:
            return
        # Recompute from current timesteps + duration
        days = self.config_manager.get_value("simulation.duration_days", 3.55)
        n_steps = self.config_manager.get_value("simulation.timesteps", 144)
        if n_steps > 0 and days > 0:
            step_sec = (days * 86400.0) / n_steps
            self._syncing = True
            self.step_size_spin.blockSignals(True)
            self.step_size_spin.setValue(self._seconds_to_step_display(step_sec))
            self.step_size_spin.blockSignals(False)
            self._syncing = False

    # ── Body selection ───────────────────────────────────────────────

    def _on_body_combo_changed(self, text):
        """Handle body combo box selection."""
        if text == self.SEPARATOR:
            # Revert to previous selection
            combo = self.widgets["simulation.body"]
            config = self.config_manager.get_config()
            prev = config.get("simulation", {}).get("body", "Moon")
            idx = combo.findText(prev)
            if idx >= 0:
                combo.blockSignals(True)
                combo.setCurrentIndex(idx)
                combo.blockSignals(False)
            return

        if text == self.CUSTOM_BODY:
            # Set up custom body mode
            self.config_manager.set_value("simulation.body", "Custom")
            self.config_manager.set_value("generic_body.name", "Custom")
            self.config_manager.set_value("generic_body.config_file", "")
            self.generic_body_label.setVisible(True)
            self.generic_body_label.setText("Custom body \u2014 configure on Body tab")
            self.value_changed.emit()
            # Switch to Body Definition tab
            parent = self.parent()
            while parent is not None:
                if hasattr(parent, 'switch_to_body_definition_tab'):
                    parent.switch_to_body_definition_tab()
                    break
                parent = parent.parent()
            return

        if text == self.BROWSE_HORIZONS:
            # Revert combo (the HORIZONS tab will set the body when selected)
            combo = self.widgets["simulation.body"]
            config = self.config_manager.get_config()
            prev = config.get("simulation", {}).get("body", "Moon")
            idx = combo.findText(prev)
            if idx >= 0:
                combo.blockSignals(True)
                combo.setCurrentIndex(idx)
                combo.blockSignals(False)
            # Signal to switch to HORIZONS tab
            self.value_changed.emit()
            # Find parent ConfigurationPanel and switch to HORIZONS tab
            parent = self.parent()
            while parent is not None:
                if hasattr(parent, 'switch_to_horizons_tab'):
                    parent.switch_to_horizons_tab()
                    break
                parent = parent.parent()
            return

        # Check if this is a HORIZONS body (prefixed with [H])
        text = str(text or "")
        is_horizons = text.startswith("[H] ")
        is_custom = (text == "Custom")
        self.generic_body_label.setVisible(is_horizons or is_custom)

        if is_horizons:
            body_name = text[4:]  # Strip "[H] " prefix
            config_path = self.config_manager.get_value(
                "generic_body.config_file", ""
            )
            self.generic_body_label.setText(f"Config: {config_path}")
        elif is_custom:
            self.generic_body_label.setText("Custom body \u2014 configure on Body tab")

        # Auto-populate duration to 1 synodic period of the selected body
        self._current_synodic_days = self._get_synodic_period_days(text)
        days = self._current_synodic_days

        self._syncing = True
        try:
            self.config_manager.set_value("simulation.duration_days", days)
            display_val = self._days_to_display(days)
            self.duration_spin.blockSignals(True)
            self.duration_spin.setValue(display_val)
            self.duration_spin.blockSignals(False)
            self._sync_end_time_from_start_and_duration()
        finally:
            self._syncing = False
        self.value_changed.emit()

    def add_horizons_body(self, body_name, config_path):
        """Add a HORIZONS body to the combo box and select it."""
        combo = self.widgets["simulation.body"]
        display_name = f"[H] {body_name}"

        # Check if already in combo
        idx = combo.findText(display_name)
        if idx < 0:
            # Insert before separator
            sep_idx = combo.findText(self.SEPARATOR)
            if sep_idx >= 0:
                combo.insertItem(sep_idx, display_name)
            else:
                combo.addItem(display_name)

        # Select it
        idx = combo.findText(display_name)
        if idx >= 0:
            combo.setCurrentIndex(idx)

        # Update config
        self.config_manager.set_value("simulation.body", display_name)
        self.config_manager.set_value("generic_body.config_file", config_path)
        self.config_manager.set_value("generic_body.name", body_name)

        self.generic_body_label.setVisible(True)
        self.generic_body_label.setText(f"Config: {config_path}")
        self.value_changed.emit()

    def refresh_horizons_bodies(self, cached_names):
        """Refresh the list of cached HORIZONS bodies in the combo box."""
        combo = self.widgets["simulation.body"]
        current = combo.currentText()

        # Block signals during the entire refresh to prevent intermediate
        # selections (e.g. the separator) from firing on_widget_changed.
        combo.blockSignals(True)

        # Remove existing [H] entries
        i = 0
        while i < combo.count():
            if combo.itemText(i).startswith("[H] "):
                combo.removeItem(i)
            else:
                i += 1

        # Insert cached bodies before separator
        sep_idx = combo.findText(self.SEPARATOR)
        for name in cached_names:
            display = f"[H] {name}"
            if sep_idx >= 0:
                combo.insertItem(sep_idx, display)
                sep_idx += 1

        # Restore selection
        idx = combo.findText(current)
        if idx >= 0:
            combo.setCurrentIndex(idx)

        combo.blockSignals(False)

    def load_from_config(self):
        config = self.config_manager.get_config()
        sim = config["simulation"]

        self._syncing = True
        try:
            self.set_text("simulation.start_time", sim["start_time"])

            # Update synodic period for the loaded body
            self._current_synodic_days = self._get_synodic_period_days(
                sim.get("body", "Moon")
            )

            # Set duration spinbox in current display units (use saved value,
            # not synodic default). Coerced: YAML resolves `5e3` and a quoted
            # number as str, and setValue(str) raises out of load_from_config,
            # which in the GUI is a process abort rather than a message.
            days = ConfigManager._as_number(sim["duration_days"], 0.0)
            display_val = self._days_to_display(days)
            self.duration_spin.blockSignals(True)
            self.duration_spin.setValue(display_val)
            self.duration_spin.blockSignals(False)

            self.set_numeric("simulation.timesteps", sim["timesteps"])

            # Body combo. str(): findText refuses a non-string argument, and
            # simulation.body is whatever yaml.safe_load made of it.
            body_name = str(sim.get("body", "") or "")
            combo = self.widgets["simulation.body"]
            idx = combo.findText(body_name)
            if idx >= 0:
                combo.blockSignals(True)
                combo.setCurrentIndex(idx)
                combo.blockSignals(False)
            else:
                # Body not found directly — check if it's a HORIZONS body
                config_path = (config.get("generic_body") or {}).get("config_file", "")

                body_name = str(body_name or "")
                if body_name.startswith("[H] "):
                    self.add_horizons_body(body_name[4:], config_path)
                else:
                    h_idx = combo.findText(f"[H] {body_name}")
                    if h_idx >= 0 and config_path:
                        self.add_horizons_body(body_name, config_path)
                    elif config_path:
                        self.add_horizons_body(body_name, config_path)

            # Compute and display end time
            self._sync_end_time_from_start_and_duration()
        finally:
            self._syncing = False


class MeshTab(BaseConfigTab):
    """Mesh configuration tab"""

    def init_ui(self):
        from PyQt6.QtWidgets import QRadioButton, QButtonGroup

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        # Global Mesh (resolution + topographic mesh, coupled in the model)
        global_group = QGroupBox("Global Mesh")
        global_layout = QGridLayout()
        global_layout.setColumnStretch(1, 1)

        self.add_int_field(global_layout, 0, "Resolution:", "mesh.resolution",
                           1, 10, "Icosphere resolution level (higher = more faces)",
                           info_key="mesh.resolution")
        self.add_bool_field(global_layout, 1, "Use Topographic Mesh", "mesh.use_topographic",
                            "Use pre-computed topographic meshes (requires DEM data)")

        # --- Topography source: built-in dataset vs custom path -------------
        src_label = QLabel("Topography source:")
        global_layout.addWidget(src_label, 2, 0)
        src_container = QWidget()
        src_layout = QHBoxLayout(src_container)
        src_layout.setContentsMargins(0, 0, 0, 0)
        self.topo_builtin_radio = QRadioButton("Built-in (recommended)")
        self.topo_builtin_radio.setToolTip(
            "Ship-with-PTM3D topography (data/topo, resolved via "
            "PTM_DATA_ROOT or the repository data directory) -- no "
            "navigation needed. Fetched on demand if missing.")
        self.topo_custom_radio = QRadioButton("Custom path…")
        self.topo_custom_radio.setToolTip(
            "Point at your own directory of pre-computed topography grids")
        self._topo_src_group = QButtonGroup(src_container)
        self._topo_src_group.addButton(self.topo_builtin_radio)
        self._topo_src_group.addButton(self.topo_custom_radio)
        src_layout.addWidget(self.topo_builtin_radio)
        src_layout.addWidget(self.topo_custom_radio)
        src_layout.addStretch()
        global_layout.addWidget(src_container, 2, 1)
        self.topo_builtin_radio.toggled.connect(self._on_topo_source_toggled)

        # Read-only display of what built-in mode will actually load, plus a
        # fetch button when the dataset is not on disk.
        resolved_label = QLabel("Resolves to:")
        global_layout.addWidget(resolved_label, 3, 0)
        resolved_container = QWidget()
        resolved_layout = QHBoxLayout(resolved_container)
        resolved_layout.setContentsMargins(0, 0, 0, 0)
        self.topo_resolved_edit = QLineEdit()
        self.topo_resolved_edit.setReadOnly(True)
        self.topo_resolved_edit.setPlaceholderText(
            "(select Built-in to see the resolved dataset path)")
        resolved_layout.addWidget(self.topo_resolved_edit)
        self.topo_fetch_btn = QPushButton("Fetch…")
        self.topo_fetch_btn.setMaximumWidth(80)
        self.topo_fetch_btn.setVisible(False)
        self.topo_fetch_btn.clicked.connect(self._on_topo_fetch)
        resolved_layout.addWidget(self.topo_fetch_btn)
        global_layout.addWidget(resolved_container, 3, 1)

        self.add_directory_field(global_layout, 4, "Topographic Directory:", "mesh.topo_directory",
                                 "Directory containing pre-computed topographic meshes "
                                 "(enabled in Custom mode)")

        global_group.setLayout(global_layout)
        layout.addWidget(global_group)

        # Custom Mesh
        custom_group = QGroupBox("Custom Mesh Import")
        custom_layout = QGridLayout()
        custom_layout.setColumnStretch(1, 1)

        self.add_file_field(custom_layout, 0, "Import File:", "mesh.import_file",
                            "Mesh Files (*.obj *.ply *.vtk);;All Files (*)",
                            "Import custom triangulated mesh",
                            info_key="mesh.import_file")
        self.add_double_field(custom_layout, 1, "Scale Factor:", "mesh.scale_factor",
                              0.001, 1000.0, 3, "Scaling factor for imported mesh",
                              info_key="mesh.scale_factor")

        custom_group.setLayout(custom_layout)
        layout.addWidget(custom_group)

        # --- Placement and occluder (Advanced) ----------------------------
        # The core-plus-ring architecture and the mesh placement overrides are
        # what a site run needs and a global run never touches, so they sit
        # behind the Advanced disclosure rather than in a hidden tab.
        place_section = self.advanced_section("Mesh Placement and Occluder")
        place_group = QGroupBox("Placement and Occluder")
        place_layout = QGridLayout()
        place_layout.setColumnStretch(1, 1)

        # The core-plus-ring architecture was CLI-only: every A3GT-class site
        # run needs an occluder, and the GUI had no field for it.
        self.add_file_field(place_layout, 0, "Occluder File:", "mesh.occluder_file",
                            "Mesh Files (*.ply *.obj *.vtk);;All Files (*)",
                            "Coarse occluder-only mesh (the ring around a "
                            "high-resolution core). It enters the illumination "
                            "shadow-ray BVH only -- never thermally solved, "
                            "radiosity-coupled or written out. See "
                            "docs/extended-occluder-architecture.md.")
        # Without these an imported DEM sits wherever its own frame put it.
        # LATITUDE IS THE GATE: ptm_thermal_sim.cpp:15317 rotates only when
        # latitude_override > -900, and longitude is read inside that branch.
        # Longitude alone therefore does nothing at all, which is why the
        # longitude field is disabled until a latitude is set rather than left
        # editable with a tooltip nobody reads.
        self.add_double_field(place_layout, 1, "Latitude Override (deg):",
                              "mesh.latitude_override", -999.0, 90.0, 4,
                              "Place the imported mesh at this latitude. "
                              "-999 = leave the mesh in its own frame. This is "
                              "the switch for BOTH overrides: the engine "
                              "rotates only when a latitude is given.")
        self.add_double_field(place_layout, 2, "Longitude Override (deg):",
                              "mesh.longitude_override", -999.0, 360.0, 4,
                              "Place the imported mesh at this longitude. Read "
                              "only when a latitude override is also set -- the "
                              "engine gates the whole rotation on latitude, so "
                              "a longitude on its own is silently ignored.")
        # The engine range-checks NOTHING: its only gate is
        # latitude_override > -900, so -95 reaches rotateMeshToLatLon with
        # cos(lat) < 0 and mirrors the mesh through the pole to +85 at
        # longitude+180. The spin box therefore steps only over the valid band,
        # with -999 as the documented sentinel just below it; a config carrying
        # a value in the dead zone is caught by ConfigManager.validate().
        self.widgets["mesh.latitude_override"].setSingleStep(1.0)
        self.placement_note = QLabel("")
        self.placement_note.setWordWrap(True)
        self.placement_note.setStyleSheet("color: gray; font-size: 10px;")
        place_layout.addWidget(self.placement_note, 3, 0, 1, 2)
        # BOTH, because the note quotes both numbers: connected to latitude
        # only, it went on stating the old longitude for the run about to start.
        # set_numeric() calls setValue during load_from_config, which emits
        # valueChanged, so the signal alone cannot tell a user edit from a load.
        # _loading does.
        self._loading = False
        for _k in ("mesh.latitude_override", "mesh.longitude_override"):
            self.widgets[_k].valueChanged.connect(
                lambda _v: self._update_placement_gating(
                    user_edit=not self._loading))

        place_group.setLayout(place_layout)
        place_section.add_widget(place_group)
        layout.addWidget(place_section)

        # Gray out global mesh options when a custom mesh is selected
        self.widgets["mesh.import_file"].textChanged.connect(self._on_import_file_changed)

        layout.addStretch()
        self.create_scroll_area(content)

        self._update_placement_gating()

    def _update_placement_gating(self, user_edit=False):
        """Longitude is inert without a latitude, so do not offer it alone.

        `user_edit` is what separates "the user just cleared the latitude" from
        "a project was opened". Only the first resets a stale longitude: doing
        it on load DISCARDED a saved longitude_override the moment the file was
        opened, which is a silent edit to the user's configuration and the exact
        thing read-only open exists to prevent. On load the value is kept, the
        field is disabled, and validate() warns that the engine will ignore it.
        """
        lat = self.widgets["mesh.latitude_override"].value()
        placing = lat > -900.0
        lon_widget = self.widgets["mesh.longitude_override"]
        if user_edit and not placing and lon_widget.value() > -900.0:
            # Clearing the latitude leaves a longitude the engine will ignore,
            # in a field about to be disabled -- so the value could not be
            # reached to remove it. Reset it here, and say so rather than
            # dropping a number the user typed without a word.
            logging.warning(
                "[mesh] latitude_override was cleared, so longitude_override "
                "= %s has been reset to -999: the engine gates the rotation on "
                "latitude and would have ignored it.", lon_widget.value())
            lon_widget.setValue(-999.0)
        lon_widget.setEnabled(placing)
        if placing and not (-90.0 <= lat <= 90.0):
            # The engine would mirror the mesh through the pole rather than
            # refuse. Say so here; validate() blocks the launch.
            self.placement_note.setText(
                f"Latitude {lat} is outside [-90, 90]. The engine does not "
                f"range-check it and would mirror the mesh through the pole. "
                f"Use -999 to leave the mesh in its own frame.")
        elif placing:
            lon = self.widgets["mesh.longitude_override"].value()
            lon_text = "the mesh's own longitude" if lon <= -900.0 \
                else f"{lon} deg"
            self.placement_note.setText(
                f"The mesh will be rotated to {lat} deg latitude, {lon_text}.")
        else:
            self.placement_note.setText(
                "No latitude override: the mesh stays in its own frame and the "
                "longitude override is not read (the engine gates the rotation "
                "on latitude).")

    def _on_import_file_changed(self, text):
        """Disable global mesh options when a custom mesh file is specified."""
        has_custom = bool(str(text or "").strip())
        self.widgets["mesh.resolution"].setEnabled(not has_custom)
        self.widgets["mesh.use_topographic"].setEnabled(not has_custom)
        if has_custom:
            self.topo_builtin_radio.setEnabled(False)
            self.topo_custom_radio.setEnabled(False)
            self.widgets["mesh.topo_directory"].setEnabled(False)
        else:
            self.topo_custom_radio.setEnabled(True)
            self._apply_topo_source_enabled()
            # refresh re-applies the body-support gating on the builtin radio
            self.refresh_builtin_topo()

    # -- topography source (built-in vs custom) ------------------------------

    def _current_body(self):
        body = self.config_manager.get_value("simulation.body", "") or ""
        body = str(body or "")
        return body[4:] if body.startswith("[H] ") else body

    def _project_root(self):
        from gui_core.preflight import find_project_root
        from gui_core.simulation_runner import SimulationRunner
        return find_project_root(SimulationRunner.find_ptm_executable())

    def _on_topo_source_toggled(self, _checked):
        source = "builtin" if self.topo_builtin_radio.isChecked() else "custom"
        self.config_manager.set_value("mesh.topo_source", source)
        if source == "builtin":
            # Canonical relative reference; the engine resolves it through
            # its PTM_DATA_ROOT contract (portable saved configs).
            self.set_text("mesh.topo_directory", "data/topo")
        self._apply_topo_source_enabled()
        self.refresh_builtin_topo()
        self.value_changed.emit()

    def _apply_topo_source_enabled(self):
        builtin = self.topo_builtin_radio.isChecked()
        self.widgets["mesh.topo_directory"].setEnabled(not builtin)

    def refresh_builtin_topo(self):
        """Update the read-only resolved-path display (body- and
        resolution-aware). Called on body change, resolution change, and
        mode toggle."""
        from gui_core import datasets
        body = self._current_body()
        res = int(self.widgets["mesh.resolution"].value())
        root = self._project_root()
        status = datasets.topo_status(body, res, root)
        supported = status.state != datasets.STATE_UNSUPPORTED

        self.topo_builtin_radio.setEnabled(
            supported and self.widgets["mesh.use_topographic"].isEnabled())
        self.topo_builtin_radio.setToolTip(
            status.detail if not supported else
            "Ship-with-PTM3D topography (data/topo), fetched on demand")

        if not self.topo_builtin_radio.isChecked():
            self.topo_resolved_edit.setText("")
            self.topo_fetch_btn.setVisible(False)
            self._topo_status = status
            return
        if not supported:
            self.topo_resolved_edit.setText(f"UNAVAILABLE -- {status.detail}")
            self.topo_fetch_btn.setVisible(False)
        elif status.state == datasets.STATE_OK:
            note = f"  ({status.detail})" if status.detail else ""
            self.topo_resolved_edit.setText(f"{status.resolved}{note}")
            self.topo_fetch_btn.setVisible(False)
        else:
            tag = ("NOT ON DISK" if status.state == datasets.STATE_MISSING
                   else "DATALESS (cloud placeholder)")
            self.topo_resolved_edit.setText(
                f"{tag}: {status.resolved} -- use Fetch")
            self.topo_fetch_btn.setVisible(bool(status.dataset))
        self._topo_status = status

    def _on_topo_fetch(self):
        from gui_panels.fetch_dialog import FetchDialog
        status = getattr(self, "_topo_status", None)
        if not status or not status.dataset:
            return
        dlg = FetchDialog(status.dataset, self._project_root(), parent=self)
        dlg.exec()
        self.refresh_builtin_topo()

    def on_body_changed(self, _body_name):
        self.refresh_builtin_topo()

    def load_from_config(self):
        config = self.config_manager.get_config()
        mesh = config["mesh"]

        self._loading = True
        try:
            self.set_numeric("mesh.resolution", mesh["resolution"])
            self.set_bool("mesh.use_topographic", 
                mesh.get("use_topographic", False))
            self.set_text("mesh.topo_directory", mesh.get("topo_directory", ""))
            self.set_text("mesh.import_file", mesh.get("import_file", ""))
            self.set_numeric("mesh.scale_factor", mesh.get("scale_factor", 1.0))
            self.set_text("mesh.occluder_file", mesh.get("occluder_file", ""))
            self.set_numeric("mesh.latitude_override",
                             mesh.get("latitude_override", -999.0))
            self.set_numeric("mesh.longitude_override",
                             mesh.get("longitude_override", -999.0))
        finally:
            self._loading = False
        # Not a user edit: a saved longitude is kept, not discarded on open.
        self._update_placement_gating()

        source = mesh.get("topo_source", "builtin")
        self.topo_builtin_radio.blockSignals(True)
        self.topo_custom_radio.blockSignals(True)
        self.topo_builtin_radio.setChecked(source == "builtin")
        self.topo_custom_radio.setChecked(source != "builtin")
        self.topo_builtin_radio.blockSignals(False)
        self.topo_custom_radio.blockSignals(False)
        self._apply_topo_source_enabled()
        self.refresh_builtin_topo()

        # Set initial enabled/disabled state based on custom mesh
        self._on_import_file_changed(mesh.get("import_file", ""))


class ThermalTab(BaseConfigTab):
    """Thermal model configuration tab"""

    def init_ui(self):
        self._syncing = False
        self._current_synodic_days = get_synodic_period_days(self.config_manager)
        self._current_orbital_days = get_orbital_period_days(self.config_manager)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        # Model Control
        control_group = QGroupBox("Model Control")
        control_layout = QGridLayout()
        control_layout.setColumnStretch(1, 1)

        self.add_bool_field(control_layout, 0, "Run Thermal Model", "thermal.run_thermal_mesh",
                            "Enable thermal calculations for mesh",
                            info_key="thermal.run_thermal_mesh")
        self.add_int_field(control_layout, 1, "CPU Threads:", "thermal.threads",
                           0, 128, "Number of parallel threads (0 = auto)",
                           info_key="thermal.threads")

        # Row 2: Equilibration with unit selector
        eq_label = QLabel("Equilibration:")
        self._apply_tooltip(eq_label, "thermal.equilibration_orbits")
        control_layout.addWidget(eq_label, 2, 0)

        eq_container = QWidget()
        eq_hlayout = QHBoxLayout(eq_container)
        eq_hlayout.setContentsMargins(0, 0, 0, 0)
        eq_hlayout.setSpacing(4)

        self.eq_spin = QDoubleSpinBox()
        self.eq_spin.setRange(0.0, 100000.0)
        self.eq_spin.setDecimals(4)
        self.eq_spin.setSingleStep(0.5)
        self.eq_spin.setToolTip("Equilibration period")
        self.eq_spin.valueChanged.connect(self._on_eq_spin_changed)
        eq_hlayout.addWidget(self.eq_spin, stretch=1)

        self.eq_unit_combo = QComboBox()
        self.eq_unit_combo.addItems(["synodic periods", "orbital periods", "days", "hours", "Earth years"])
        self.eq_unit_combo.setMaximumWidth(130)
        self.eq_unit_combo.setToolTip("Equilibration display unit")
        self.eq_unit_combo.currentTextChanged.connect(self._on_eq_unit_changed)
        eq_hlayout.addWidget(self.eq_unit_combo)

        control_layout.addWidget(eq_container, 2, 1)
        self.widgets["thermal.equilibration_orbits"] = self.eq_spin

        # Compute Statistics stays in Model Control.
        self.add_bool_field(control_layout, 3, "Compute Statistics", "thermal.compute_statistics",
                            "Calculate min/max/mean temperature statistics",
                            info_key="thermal.compute_statistics")

        # Both of these were written into every saved config with no control
        # at all. enable_shadow_flux is a physics switch; geothermal_heat_flow
        # is the per-run bottom boundary condition.
        self.add_bool_field(control_layout, 4, "Enable Shadow Flux",
                            "thermal.enable_shadow_flux",
                            info_key="thermal.enable_shadow_flux")
        self.add_double_field(control_layout, 5, "Geothermal Q_b (W/m²):",
                              # 1000 W/m^2, not 10: the falsifier fixture uses
                              # 5.0 and a sensitivity study can go far higher.
                              # An out-of-range load warns and keeps the config
                              # value (set_numeric), but a range that refuses
                              # plausible science is a range set too tight.
                              "thermal.geothermal_heat_flow", -1.0, 1000.0, 4,
                              "Bottom-boundary heat flux. -1 = the body's own "
                              "default.\n"
                              "Live on every body since PTM-161 was fixed "
                              "(commit 2e06a70): the value is forwarded to the "
                              "solvers as ovr_Qb and the run prints "
                              "'[Thermo override] applied at solver: Qb ...'. "
                              "A positive value also arms the property-override "
                              "tripwire, which aborts the run if the override "
                              "does not reach a solver.")
        # Measured on this tree rather than taken from KNOWN_LIMITATIONS.md
        # section 4, which still says "Status: Open": Moon ico3, 48 steps,
        # Qb -1 vs 0.05 W/m^2, max |dT_min| = 0.8421 K over 1280 faces. It is
        # live, and the docs are stale.
        self.qb_note = QLabel("")
        self.qb_note.setWordWrap(True)
        self.qb_note.setStyleSheet("color: gray; font-size: 10px;")
        control_layout.addWidget(self.qb_note, 6, 0, 1, 2)
        self.widgets["thermal.geothermal_heat_flow"].valueChanged.connect(
            lambda _v: self._update_qb_gating())

        control_group.setLayout(control_layout)
        layout.addWidget(control_group)

        # Equilibration cadence — an expert knob, grouped under "Advanced
        # thermal" (collapsed) below. Decouples equilibration time-step cadence
        # from output cadence; for large meshes with fine output cadence this
        # dramatically reduces illumination memory and compute (up to 50x).
        cad_group = QGroupBox("Equilibration Cadence")
        cad_layout = QGridLayout()
        cad_layout.setColumnStretch(1, 1)

        cad_label = QLabel("Eq. Cadence:")
        cad_label.setToolTip(
            "Time-step cadence during equilibration phase (separate from output cadence).\n"
            "Legacy: match output cadence (slow, byte-identical to old PTM3D).\n"
            "Accuracy tiers set samples per diurnal cycle of the target body.\n"
            "Custom: set explicit steps per Earth-day."
        )
        cad_layout.addWidget(cad_label, 0, 0)

        cad_container = QWidget()
        cad_hlayout = QHBoxLayout(cad_container)
        cad_hlayout.setContentsMargins(0, 0, 0, 0)
        cad_hlayout.setSpacing(4)

        self.eq_cadence_combo = QComboBox()
        # Tier values and accuracy targets were calibrated against an ico3 Moon
        # reference run (see scripts/test_eq_cadence_accuracy.sh for the method).
        # Accuracy = max |ΔT| at the first output timestep vs. the legacy run.
        self.eq_cadence_combo.addItems([
            "Balanced (120 samples/cycle, ~4 K max)",
            "High accuracy (240 samples/cycle, ~3 K max)",
            "Fast (30 samples/cycle, ~7 K max)",
            "Legacy (match output cadence)",
            "Custom (steps/Earth-day)",
        ])
        self.eq_cadence_combo.setToolTip(
            "Preset accuracy tier for equilibration. Affects only equilibration.\n"
            "Output cadence is always set by the Time / Timesteps parameters."
        )
        self.eq_cadence_combo.currentIndexChanged.connect(self._on_eq_cadence_mode_changed)
        cad_hlayout.addWidget(self.eq_cadence_combo, stretch=1)

        # Custom steps/day spinbox (only shown/used in Custom mode)
        self.eq_cadence_spin = QDoubleSpinBox()
        self.eq_cadence_spin.setRange(0.01, 100000.0)
        self.eq_cadence_spin.setDecimals(3)
        self.eq_cadence_spin.setSingleStep(1.0)
        self.eq_cadence_spin.setValue(24.0)
        self.eq_cadence_spin.setMaximumWidth(110)
        self.eq_cadence_spin.setToolTip(
            "Custom cadence in steps per Earth-day. Only used in Custom mode."
        )
        self.eq_cadence_spin.setEnabled(False)
        self.eq_cadence_spin.valueChanged.connect(self._on_eq_cadence_value_changed)
        cad_hlayout.addWidget(self.eq_cadence_spin)

        cad_layout.addWidget(cad_container, 0, 1)

        # Live readout of the effective cadence (body-aware)
        self.eq_cadence_readout = QLabel("")
        self.eq_cadence_readout.setStyleSheet("color: gray; font-size: 10px;")
        self.eq_cadence_readout.setToolTip(
            "Computed effective cadence for the current body.\n"
            "'samples per diurnal cycle' is body-independent; 'steps per Earth-day' is what\n"
            "PTM3D uses internally and depends on the body's synodic day."
        )
        cad_layout.addWidget(self.eq_cadence_readout, 1, 1)

        cad_group.setLayout(cad_layout)

        # GPU Acceleration
        gpu_group = QGroupBox("GPU Acceleration")
        gpu_layout = QGridLayout()
        gpu_layout.setColumnStretch(1, 1)

        self.add_bool_field(gpu_layout, 0, "Use GPU for Thermal Solver", "thermal.use_gpu",
                            "Use OpenCL GPU acceleration for thermal solver",
                            info_key="thermal.use_gpu")

        gpu_group.setLayout(gpu_layout)
        layout.addWidget(gpu_group)


        # --- Initial condition -------------------------------------------
        # spinup_seed selects the initial condition; `legacy` exists to
        # reproduce a pre-2026-08-11 baseline digit-for-digit. The GUI used to
        # defeat that on the GPU path by injecting PTM_EQ_PERFACE_FLUX into
        # every run (PTM-276); it no longer injects it, so `legacy` here means
        # what it says.
        seed_group = QGroupBox("Initial Condition")
        seed_layout = QGridLayout()
        seed_layout.setColumnStretch(1, 1)

        self.add_combo_field(seed_layout, 0, "Spin-up Seed:", "thermal.spinup_seed",
                             ["physical", "legacy"],
                             "physical (default): each face starts from its own "
                             "conductive steady profile, so a permanently "
                             "shadowed column starts on the geothermal floor.\n"
                             "legacy: the pre-2026-08-11 seed plus the analytic "
                             "clear-sky pre-conditioner -- exactly what produced "
                             "results in this repository before that date. Use "
                             "it to reproduce an old baseline.")
        self.add_combo_field(seed_layout, 1, "Seed Anchor:", "thermal.seed_anchor",
                             ["lumped", "warm"],
                             "Which surface anchor the shared seed hangs its "
                             "conductive profile from. Only read under "
                             "spinup_seed: physical.\n"
                             "lumped (default): anchors at the cycle mean.\n"
                             "warm: the fourth-moment anchor, which by Jensen is "
                             ">= the mean -- +45 K on the lit population of the "
                             "shipped lunar 80N crater. Byte-identical to the "
                             "pre-2026-08-12 seed.")
        self.add_double_field(seed_layout, 2, "Analytic Seed Cycles:",
                              "thermal.analytic_seed_cycles", -1.0, 100.0, 1,
                              "Diurnal cycles of clear-sky pre-conditioning "
                              "before the real forcing. It is a FORCING, not a "
                              "seed: no topography, no shadowing, no IR, no "
                              "indirect flux, and the GPU has no analog. "
                              "-1 = follow the spin-up seed; 0 = off.")
        self.add_bool_field(seed_layout, 3, "Refuse Two-Timescale Faces",
                            "thermal.seed_anchor_refuse_two_timescale",
                            "Strict refusal when a face carries diurnal AND "
                            "seasonal power, which one lumped conductance "
                            "cannot cover. Off by default because refusing "
                            "substitutes the warm anchor, and that is the "
                            "estimator with the +45 K bias.")

        self.seed_note = QLabel("")
        self.seed_note.setWordWrap(True)
        self.seed_note.setStyleSheet("color: gray; font-size: 10px;")
        seed_layout.addWidget(self.seed_note, 4, 0, 1, 2)
        self.widgets["thermal.spinup_seed"].currentTextChanged.connect(
            self._update_seed_gating)

        seed_group.setLayout(seed_layout)

        # --- Run state (warm start / chunk handoff) ------------------------
        state_group = QGroupBox("Run State")
        state_layout = QGridLayout()
        state_layout.setColumnStretch(1, 1)

        self.add_file_field(state_layout, 0, "Initial State File:",
                            "thermal.initial_state_file",
                            "State Files (*.bin *.state);;All Files (*)",
                            "Per-face T(z) profile to start from. This is the "
                            "warm start: the run continues an earlier one "
                            "instead of re-equilibrating.")
        self.add_file_field(state_layout, 1, "Final State File:",
                            "thermal.final_state_file",
                            "State Files (*.bin *.state);;All Files (*)",
                            "Where to dump the per-face T(z) profile at the end. "
                            "The state written is the one at the LAST exported "
                            "step, which is what the next chunk must resume from.")

        state_note = QLabel(
            "A run launched from this window is still a single engine process: "
            "it does not go through the chunked runner, so it has no automatic "
            "cool-down, stall abort or resume. These two fields are the manual "
            "handoff.")
        state_note.setWordWrap(True)
        state_note.setStyleSheet("color: gray; font-size: 10px;")
        state_layout.addWidget(state_note, 2, 0, 1, 2)

        state_group.setLayout(state_layout)

        # Initial condition and run state are reproduction/continuation
        # machinery: a first run never touches them, and a baseline
        # reproduction needs both. One Advanced disclosure, not a hidden tab.
        adv_run = self.advanced_section("Initial Condition and Run State")
        adv_run.add_widget(seed_group)
        adv_run.add_widget(state_group)
        layout.addWidget(adv_run)

        # Integration Window (advanced)
        integ_group = QGroupBox("Integration Window")
        integ_layout = QGridLayout()
        integ_layout.setColumnStretch(1, 1)

        self.add_double_field(integ_layout, 0, "Integration Days:", "thermal.integration_days",
                              -1.0, 10000.0, 2, "Integration window in days (-1 = automatic)",
                              info_key="thermal.integration_days")
        self.add_int_field(integ_layout, 1, "Integration Steps:", "thermal.integration_steps",
                           -1, 100000, "Integration steps override (-1 = automatic)",
                           info_key="thermal.integration_steps")

        integ_note = QLabel(
            "Leave at -1 for automatic (integrate over the full simulation "
            "duration / timesteps). Set a positive value only to override.")
        integ_note.setStyleSheet("color: gray; font-size: 10px;")
        integ_note.setWordWrap(True)
        integ_layout.addWidget(integ_note, 2, 0, 1, 2)

        integ_group.setLayout(integ_layout)

        # Property Maps
        maps_group = QGroupBox("Thermal Property Maps")
        maps_layout = QGridLayout()
        maps_layout.setColumnStretch(1, 1)

        # Map mode selector: None / Bundled / Custom
        self.add_combo_field(
            maps_layout, 0, "Map Mode:", "thermal.property_map_mode",
            ["None (uniform body H)",
             "Bundled icosphere default for body",
             "Custom file"],
            "None: every face uses the body-default H. "
            "Bundled: auto-discover the icosphere map under "
            "<property_maps_directory>/<body>/. "
            "Custom: load a per-face binary float32 file (one value per mesh face).",
            info_key="thermal.property_map_mode",
        )

        # Map content type
        self.add_combo_field(
            maps_layout, 1, "Map Content:", "thermal.property_map_kind",
            ["H-parameter (m)", "Thermal inertia (J m^-2 K^-1 s^-1/2)"],
            "H-parameter: per-face e-folding depth feeding rho(z), k(z). "
            "Thermal inertia: per-face TI(273 K); the loader inverts to an "
            "effective H per face so the same downstream model is reused.",
            info_key="thermal.property_map_kind",
        )

        self.add_file_field(
            maps_layout, 2, "Custom Map File:", "thermal.thermal_inertia_map",
            "Binary Files (*.bin);;All Files (*)",
            "Path to per-face binary float32 (matches mesh face count). "
            "Used only when Map Mode = Custom file. The Map Content selector "
            "tells PTM3D how to interpret the file (H or TI).",
            info_key="thermal.thermal_inertia_map",
        )
        self.add_directory_field(
            maps_layout, 3, "Property Maps Dir:", "thermal.property_maps_directory",
            "Root directory for bundled maps. Files are looked up under "
            "<dir>/<body>/{hpar|ti273k}_<body>_ico<level>.bin.",
            info_key="thermal.property_maps_directory",
        )

        maps_group.setLayout(maps_layout)

        # Collect the expert thermal knobs (equilibration cadence, integration
        # window, property maps) under one collapsed section so the Thermal tab
        # leads with the common controls.
        adv_thermal = self.advanced_section("Advanced thermal")
        adv_thermal.add_widget(cad_group)
        adv_thermal.add_widget(integ_group)
        adv_thermal.add_widget(maps_group)
        layout.addWidget(adv_thermal)

        # Wire enable/disable based on mode
        self.widgets["thermal.property_map_mode"].currentIndexChanged.connect(
            self._on_property_map_mode_changed
        )
        self._property_map_dependent_keys = [
            "thermal.thermal_inertia_map",
            "thermal.property_maps_directory",
            "thermal.property_map_kind",
        ]

        layout.addStretch()
        self.create_scroll_area(content)

        self._update_seed_gating()
        self._update_qb_gating()

    # ── Basal heat flow ──────────────────────────────────────────────

    def _update_qb_gating(self):
        """Say what a positive basal heat flow does. It is live on every body.

        This control was briefly disabled on built-in bodies on the strength of
        docs/KNOWN_LIMITATIONS.md section 4, which describes the value being
        parsed, announced and then discarded. That was PTM-161, and PTM-161 was
        fixed by commit 2e06a70 -- an ancestor of this branch. Measured here to
        settle it rather than trusting either document: Moon ico3, 48 steps,
        Qb -1 vs 0.05 W/m^2, max |dT_min| = 0.8421 K, and the run prints
        '[Thermo override] applied at solver: Qb 0.018 -> 0.05 W/m^2'.
        Disabling it made a working knob unreachable on the GUI's own default
        body, and -- worse -- told the user a value the engine WOULD apply was
        going to be ignored.
        """
        # Coerce before comparing. set_numeric can leave a raw scalar in the
        # config (out of range, deliberately), and YAML resolves `-1e0`/`5e3`
        # as str, so a bare `<=` here aborted the application from
        # load_from_config -- the branch's own headline crash class,
        # reintroduced in a function the branch added.
        raw = self.config_manager.get_value("thermal.geothermal_heat_flow", -1.0)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            self.qb_note.setText(
                f"Geothermal Q_b = {raw!r} is not a number. The engine will "
                f"read it as 0 and use the body default.")
            return
        if value <= 0:
            self.qb_note.setText("")
            return
        self.qb_note.setText(
            f"Q_b = {value} W/m² overrides the body default. This is a property "
            f"OVERRIDE: it is forwarded to the solvers as ovr_Qb, announced in "
            f"the log, and arms the tripwire that aborts the run if it fails to "
            f"reach one. It also sets the deep floor, T_geo = "
            f"(Q_b/(eps*sigma))^0.25.")

    # ── Initial-condition handler ────────────────────────────────────

    def _update_seed_gating(self, _mode=None):
        """seed_anchor is only read under spinup_seed: physical."""
        physical = self.widgets["thermal.spinup_seed"].currentText() == "physical"
        for key in ("thermal.seed_anchor",
                    "thermal.seed_anchor_refuse_two_timescale"):
            w = self.widgets.get(key)
            if w is not None:
                w.setEnabled(physical)
        if physical:
            self.seed_note.setText(
                "physical: each face starts from its own conductive steady "
                "profile. Seed Anchor applies.")
        else:
            self.seed_note.setText(
                "legacy: reproduces a pre-2026-08-11 baseline. Seed Anchor is "
                "not read in this mode, and the analytic pre-conditioner runs "
                "unless Analytic Seed Cycles says otherwise.")

    # ── Property-map mode handler ────────────────────────────────────

    def _on_property_map_mode_changed(self, index):
        """Enable/disable property-map dependent fields based on mode.

        index 0 = None (uniform body H)
        index 1 = Bundled icosphere default for body
        index 2 = Custom file
        """
        # Custom file mode: enable the file picker; bundled mode: disable file picker
        # but keep dir picker enabled. None: disable everything except mode itself.
        is_custom = (index == 2)
        is_bundled = (index == 1)
        is_active = (index != 0)

        widgets = self.widgets
        if "thermal.thermal_inertia_map" in widgets:
            widgets["thermal.thermal_inertia_map"].setEnabled(is_custom)
        if "thermal.property_maps_directory" in widgets:
            widgets["thermal.property_maps_directory"].setEnabled(is_bundled or is_custom)
        if "thermal.property_map_kind" in widgets:
            widgets["thermal.property_map_kind"].setEnabled(is_active)

    # ── Equilibration unit conversion ────────────────────────────────

    def _eq_orbits_to_display(self, orbits):
        """Convert equilibration orbits (synodic cycles) to the selected display unit."""
        unit = self.eq_unit_combo.currentText()
        # `orbits` comes straight from the config, where it can be a str.
        orbits = ConfigManager._as_number(orbits, 0.0)
        days = orbits * self._current_synodic_days
        if unit == "days":
            return days
        elif unit == "hours":
            return days * 24.0
        elif unit == "Earth years":
            return days / 365.25
        elif unit == "orbital periods":
            return days / self._current_orbital_days if self._current_orbital_days > 0 else orbits
        return orbits  # "synodic periods"

    def _eq_display_to_orbits(self, value):
        """Convert display value back to synodic cycles."""
        unit = self.eq_unit_combo.currentText()
        if unit == "days":
            return value / self._current_synodic_days if self._current_synodic_days > 0 else value
        elif unit == "hours":
            days = value / 24.0
            return days / self._current_synodic_days if self._current_synodic_days > 0 else days
        elif unit == "Earth years":
            days = value * 365.25
            return days / self._current_synodic_days if self._current_synodic_days > 0 else days
        elif unit == "orbital periods":
            days = value * self._current_orbital_days
            return days / self._current_synodic_days if self._current_synodic_days > 0 else days
        return value  # "synodic periods"

    def _update_eq_spin_range(self, unit):
        """Adjust spinbox range, decimals, and step for the selected unit."""
        self.eq_spin.blockSignals(True)
        if unit == "hours":
            self.eq_spin.setRange(0.0, 365000.0 * 24)
            self.eq_spin.setDecimals(2)
            self.eq_spin.setSingleStep(1.0)
        elif unit == "Earth years":
            self.eq_spin.setRange(0.0, 1000.0)
            self.eq_spin.setDecimals(6)
            self.eq_spin.setSingleStep(0.1)
        elif unit == "days":
            self.eq_spin.setRange(0.0, 365000.0)
            self.eq_spin.setDecimals(4)
            self.eq_spin.setSingleStep(1.0)
        elif unit == "orbital periods":
            self.eq_spin.setRange(0.0, 10000.0)
            self.eq_spin.setDecimals(4)
            self.eq_spin.setSingleStep(0.5)
        else:  # synodic periods
            self.eq_spin.setRange(0.0, 100000.0)
            self.eq_spin.setDecimals(4)
            self.eq_spin.setSingleStep(0.5)
        self.eq_spin.blockSignals(False)

    def _on_eq_spin_changed(self, display_value):
        """Handle equilibration spinbox value change."""
        if self._syncing:
            return
        orbits = self._eq_display_to_orbits(display_value)
        self.config_manager.set_value("thermal.equilibration_orbits", orbits)
        self.value_changed.emit()

    def _on_eq_unit_changed(self, unit_text):
        """Handle unit combo change — re-display in new unit."""
        if self._syncing:
            return
        self._syncing = True
        try:
            orbits = self.config_manager.get_value("thermal.equilibration_orbits", 1.0)
            self._update_eq_spin_range(unit_text)
            display_val = self._eq_orbits_to_display(orbits)
            self.eq_spin.blockSignals(True)
            self.eq_spin.setValue(display_val)
            self.eq_spin.blockSignals(False)
        finally:
            self._syncing = False

    def on_body_changed(self, body_name):
        """Called when the body selection changes on the Simulation tab."""
        self._current_synodic_days = get_synodic_period_days(self.config_manager, body_name)
        self._current_orbital_days = get_orbital_period_days(self.config_manager, body_name)
        # Re-display current value in new body's units
        self._syncing = True
        try:
            orbits = self.config_manager.get_value("thermal.equilibration_orbits", 1.0)
            display_val = self._eq_orbits_to_display(orbits)
            self.eq_spin.blockSignals(True)
            self.eq_spin.setValue(display_val)
            self.eq_spin.blockSignals(False)
        finally:
            self._syncing = False
        # Refresh the cadence readout text for the new body
        self._update_eq_cadence_readout()
        self._update_qb_gating()

    # ── Equilibration cadence handlers ────────────────────────────────

    # Mode indices match the order of items added to eq_cadence_combo
    _EQ_MODE_BALANCED = 0     # 24 samples/diurnal
    _EQ_MODE_HIGH     = 1     # 48 samples/diurnal
    _EQ_MODE_FAST     = 2     # 12 samples/diurnal
    _EQ_MODE_LEGACY   = 3     # match output cadence
    _EQ_MODE_CUSTOM   = 4     # explicit steps/Earth-day

    _EQ_ACCURACY_SAMPLES = {
        _EQ_MODE_BALANCED: 120.0,
        _EQ_MODE_HIGH:     240.0,
        _EQ_MODE_FAST:      30.0,
    }

    def _on_eq_cadence_mode_changed(self, idx):
        """Handle cadence-mode dropdown change."""
        if self._syncing:
            return
        self._syncing = True
        try:
            if idx in self._EQ_ACCURACY_SAMPLES:
                # Accuracy tier: set samples_per_diurnal_cycle, clear explicit steps/day
                samples = self._EQ_ACCURACY_SAMPLES[idx]
                self.config_manager.set_value("thermal.equilibration_samples_per_diurnal_cycle", samples)
                self.config_manager.set_value("thermal.equilibration_steps_per_day", -1.0)
                self.eq_cadence_spin.setEnabled(False)
                # Reflect the chosen samples value in the spinbox for transparency (in
                # Custom mode, the same spinbox means steps/day; here it just shows the
                # sample count for the reference body — advisory only).
                self.eq_cadence_spin.blockSignals(True)
                self.eq_cadence_spin.setValue(samples)
                self.eq_cadence_spin.blockSignals(False)
            elif idx == self._EQ_MODE_LEGACY:
                # Legacy: clear both (backend falls back to matching output cadence)
                self.config_manager.set_value("thermal.equilibration_samples_per_diurnal_cycle", -1.0)
                self.config_manager.set_value("thermal.equilibration_steps_per_day", -1.0)
                self.eq_cadence_spin.setEnabled(False)
            elif idx == self._EQ_MODE_CUSTOM:
                # Custom: set explicit steps/Earth-day from the spinbox, clear samples
                self.config_manager.set_value("thermal.equilibration_samples_per_diurnal_cycle", -1.0)
                self.config_manager.set_value("thermal.equilibration_steps_per_day",
                                              float(self.eq_cadence_spin.value()))
                self.eq_cadence_spin.setEnabled(True)
        finally:
            self._syncing = False
        self._update_eq_cadence_readout()
        self.value_changed.emit()

    def _on_eq_cadence_value_changed(self, value):
        """Handle custom-mode steps/day change."""
        if self._syncing:
            return
        if self.eq_cadence_combo.currentIndex() != self._EQ_MODE_CUSTOM:
            return  # Ignore changes when the spinbox is a read-only display
        self.config_manager.set_value("thermal.equilibration_steps_per_day", float(value))
        self.config_manager.set_value("thermal.equilibration_samples_per_diurnal_cycle", -1.0)
        self._update_eq_cadence_readout()
        self.value_changed.emit()

    def _update_eq_cadence_readout(self):
        """Update the human-readable cadence summary for the current body."""
        idx = self.eq_cadence_combo.currentIndex()
        synodic = self._current_synodic_days if self._current_synodic_days > 0 else 1.0
        if idx in self._EQ_ACCURACY_SAMPLES:
            samples = self._EQ_ACCURACY_SAMPLES[idx]
            spd = samples / synodic
            self.eq_cadence_readout.setText(
                f"= {samples:.0f} samples/diurnal cycle  \u2192  {spd:.3f} steps/Earth-day "
                f"(synodic {synodic:.2f} d)"
            )
        elif idx == self._EQ_MODE_LEGACY:
            self.eq_cadence_readout.setText(
                "= matches output cadence (potentially huge for fine output + long equilibration)"
            )
        elif idx == self._EQ_MODE_CUSTOM:
            spd = float(self.eq_cadence_spin.value())
            samples = spd * synodic
            self.eq_cadence_readout.setText(
                f"= {spd:.3f} steps/Earth-day  \u2192  {samples:.2f} samples/diurnal cycle "
                f"(synodic {synodic:.2f} d)"
            )
        else:
            self.eq_cadence_readout.setText("")

    def _sync_eq_cadence_from_config(self):
        """Set combo + spin to reflect stored values (called from load_from_config)."""
        thermal = self.config_manager.get_config().get("thermal", {})
        # Coerced: yaml.safe_load returns str for `-1e0`, `5e3` and a quoted
        # `'120'` -- and '120' is the GUI's OWN default for the samples key, so
        # a config carrying it quoted aborted the window here.
        explicit = ConfigManager._as_number(
            thermal.get("equilibration_steps_per_day", -1.0), -1.0)
        samples = ConfigManager._as_number(
            thermal.get("equilibration_samples_per_diurnal_cycle", -1.0), -1.0)

        self.eq_cadence_combo.blockSignals(True)
        self.eq_cadence_spin.blockSignals(True)
        try:
            if explicit > 0.0:
                self.eq_cadence_combo.setCurrentIndex(self._EQ_MODE_CUSTOM)
                self.eq_cadence_spin.setValue(float(explicit))
                self.eq_cadence_spin.setEnabled(True)
            elif samples > 0.0:
                # Snap to known tiers; otherwise use Balanced as closest default
                tier_idx = self._EQ_MODE_BALANCED
                for idx_k, s_k in self._EQ_ACCURACY_SAMPLES.items():
                    if abs(float(samples) - s_k) < 1e-6:
                        tier_idx = idx_k
                        break
                self.eq_cadence_combo.setCurrentIndex(tier_idx)
                self.eq_cadence_spin.setValue(self._EQ_ACCURACY_SAMPLES[tier_idx])
                self.eq_cadence_spin.setEnabled(False)
            else:
                # Both unset/legacy
                self.eq_cadence_combo.setCurrentIndex(self._EQ_MODE_LEGACY)
                self.eq_cadence_spin.setEnabled(False)
        finally:
            self.eq_cadence_combo.blockSignals(False)
            self.eq_cadence_spin.blockSignals(False)
        self._update_eq_cadence_readout()

    # ── Config loading ───────────────────────────────────────────────

    def load_from_config(self):
        config = self.config_manager.get_config()
        thermal = config["thermal"]

        self._syncing = True
        try:
            self.set_bool("thermal.run_thermal_mesh", thermal["run_thermal_mesh"])
            self.set_numeric("thermal.threads", thermal["threads"])

            # Update synodic and orbital periods for current body
            self._current_synodic_days = get_synodic_period_days(self.config_manager)
            self._current_orbital_days = get_orbital_period_days(self.config_manager)

            # Set equilibration in current display units
            orbits = thermal.get("equilibration_orbits", 1.0)
            display_val = self._eq_orbits_to_display(orbits)
            self.eq_spin.blockSignals(True)
            self.eq_spin.setValue(display_val)
            self.eq_spin.blockSignals(False)

            self.set_bool("thermal.compute_statistics", thermal["compute_statistics"])
            self.set_bool("thermal.use_gpu", thermal.get("use_gpu", False))
            self.set_bool("thermal.enable_shadow_flux",
                          thermal.get("enable_shadow_flux", False))
            self.set_numeric("thermal.geothermal_heat_flow",
                             thermal.get("geothermal_heat_flow", -1.0))

            self.set_enum("thermal.spinup_seed",
                          thermal.get("spinup_seed", "physical"),
                          SPINUP_SEED_ALIASES)
            self.set_enum("thermal.seed_anchor",
                          thermal.get("seed_anchor", "lumped"),
                          SEED_ANCHOR_ALIASES)
            self.set_numeric("thermal.analytic_seed_cycles",
                             thermal.get("analytic_seed_cycles", -1.0))
            self.set_bool("thermal.seed_anchor_refuse_two_timescale",
                          thermal.get("seed_anchor_refuse_two_timescale", False))
            self.set_text("thermal.initial_state_file",
                          thermal.get("initial_state_file", ""))
            self.set_text("thermal.final_state_file",
                          thermal.get("final_state_file", ""))

            self.set_numeric("thermal.integration_days", thermal["integration_days"])
            self.set_numeric("thermal.integration_steps", thermal["integration_steps"])

            # Property maps (synthetic mode/kind drive the UI; underlying
            # YAML keys are managed by ConfigManager at save/load time).
            # The combo's writeback stores the LABEL TEXT, so a second
            # load_from_config() without an intervening load_from_yaml() saw
            # "Bundled icosphere default for body" and int() raised -- an abort.
            # The dashboard's mesh chain does exactly that sequence
            # (thermal_adapter.set_input -> load_from_config). ConfigManager
            # already defends the same value with _to_index at save time; this
            # call site did not.
            from gui_core.config_manager import ConfigManager as _CM
            mode = _CM._to_index(thermal.get("property_map_mode", 0),
                                 _CM._MODE_LABELS)
            kind = _CM._to_index(thermal.get("property_map_kind", 0),
                                 _CM._KIND_LABELS)
            if "thermal.property_map_mode" in self.widgets:
                self.widgets["thermal.property_map_mode"].setCurrentIndex(mode)
            if "thermal.property_map_kind" in self.widgets:
                self.widgets["thermal.property_map_kind"].setCurrentIndex(kind)
            # The custom-file widget reuses the existing thermal_inertia_map
            # field (file picker). When kind=H-parameter, the same widget is
            # interpreted as an h_parameter_map path at save-time.
            custom_path = thermal.get("thermal_inertia_map", "") or thermal.get("h_parameter_map", "")
            self.set_text("thermal.thermal_inertia_map", custom_path)
            if "thermal.property_maps_directory" in self.widgets:
                self.set_text("thermal.property_maps_directory", 
                    thermal.get("property_maps_directory", "data/property_maps")
                )
            # Sync enable state to mode
            self._on_property_map_mode_changed(mode)
        finally:
            self._syncing = False

        # Sync equilibration-cadence widgets from config (must run OUTSIDE _syncing
        # because the combo/spin have their own block-signals guards).
        self._sync_eq_cadence_from_config()
        self._update_seed_gating()
        self._update_qb_gating()


class RadiationTab(BaseConfigTab):
    """Radiation exchange configuration tab"""

    def init_ui(self):
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        # Radiation Exchange (always visible, includes face selection)
        enable_group = QGroupBox("Radiation Exchange")
        enable_layout = QGridLayout()
        enable_layout.setColumnStretch(1, 1)

        self.add_bool_field(enable_layout, 0, "Enable Radiation Exchange", "radiation.enabled",
                            "Compute thermal radiation exchange between faces",
                            info_key="radiation.enabled")
        self.add_bool_field(enable_layout, 1, "Verbose Progress", "radiation.verbose",
                            "Print detailed progress")
        self.add_bool_field(enable_layout, 2, "Use GPU for Radiosity", "radiation.use_gpu",
                            "Use OpenCL GPU acceleration for radiosity solver",
                            info_key="radiation.use_gpu")
        self.add_double_field(enable_layout, 3, "Min Shadow Fraction:", "radiation.min_shadow_fraction",
                              0.0, 1.0, 2, "Minimum shadow fraction for inclusion in radiation exchange",
                              info_key="radiation.min_shadow_fraction")

        enable_group.setLayout(enable_layout)
        layout.addWidget(enable_group)

        # --- Sub-facet roughness (Smith 1967 beaming) ---------------------
        # The engine enforces only "beaming: true requires rms_slope_deg > 0"
        # and NOTHING about the scale of the mesh it is applied to. So this
        # control pair is the only thing between a user and double-counted
        # roughness; the rule lives in gui_core.facet_scale.
        beam_group = QGroupBox("Sub-Facet Roughness (Beaming)")
        beam_layout = QGridLayout()
        beam_layout.setColumnStretch(1, 1)

        self.add_bool_field(beam_layout, 0, "Enable Beaming", "radiation.beaming",
                            "Smith (1967) anisotropic roughness-beamed emission. "
                            "Only valid when the mesh facets are much LARGER "
                            "than the roughness scale -- otherwise the mesh "
                            "already resolves the roughness and this counts it "
                            "twice.")
        self.add_double_field(beam_layout, 1, "RMS Slope w (deg):",
                              "radiation.rms_slope_deg", 0.0, 60.0, 2,
                              "RMS sub-facet slope in DEGREES. Literature range "
                              "20-37: 32.5 Jhoti et al. (2024) lunar highlands "
                              "(the validated polar value), ~20 Bandfield et al. "
                              "(2015) global regolith fit, ~37 Rubanenko et al. "
                              "(2020). beaming: true with w = 0 is fatal at "
                              "engine startup.")

        self.beaming_note = QLabel("")
        self.beaming_note.setWordWrap(True)
        self.beaming_note.setStyleSheet("font-size: 10px; padding: 4px;")
        beam_layout.addWidget(self.beaming_note, 2, 0, 1, 2)

        self.beaming_ack = QCheckBox(
            "I confirm this mesh does not already resolve sub-facet slopes")
        self.beaming_ack.setVisible(False)
        self.beaming_ack.toggled.connect(self._update_beaming_gating)
        beam_layout.addWidget(self.beaming_ack, 3, 0, 1, 2)

        beam_group.setLayout(beam_layout)
        layout.addWidget(beam_group)

        self.widgets["radiation.beaming"].toggled.connect(
            self._on_beaming_toggled)
        # Beaming is inert with radiation off, so the switch's own gate depends
        # on that checkbox too.
        self.widgets["radiation.enabled"].toggled.connect(
            lambda _v: self._update_beaming_gating())

        # --- Thermal-Radiation Coupling (collapsed) ---
        coupling_section = self.advanced_section("Thermal-Radiation Coupling")
        coupling_widget = QWidget()
        coupling_layout = QGridLayout(coupling_widget)
        coupling_layout.setColumnStretch(1, 1)

        # PTM-102. The mode was unreachable while the field it gates was
        # visible, so a user could set a "Face Fraction (legacy mode only)"
        # that the engine then announced it was ignoring.
        self.add_combo_field(coupling_layout, 0, "Coupling Mode:",
                             "radiation.coupling_mode",
                             ["bounded", "fixed", "legacy"],
                             "How the thermal-radiation coupling loop decides "
                             "it is done (PTM-89).\n"
                             "bounded (default): absolute selection threshold, "
                             "no face-fraction exit.\n"
                             "fixed: as bounded, plus a fixed iteration count.\n"
                             "legacy: the pre-PTM-89 behavior, for reproducing "
                             "published runs. It is the only mode that reads "
                             "Face Fraction.")
        self.add_int_field(coupling_layout, 1, "Max Coupling Iterations:", "radiation.max_coupling_iterations",
                           1, 50, "Maximum thermal-radiation coupling iterations",
                           info_key="radiation.max_coupling_iterations")
        self.add_double_field(coupling_layout, 2, "Temperature Tolerance (K):", "radiation.coupling_temp_threshold",
                              0.1, 10.0, 1, "Temperature convergence threshold",
                              info_key="radiation.coupling_temp_threshold")
        self.add_double_field(coupling_layout, 3, "Flux Tolerance (W/m\u00b2):", "radiation.coupling_flux_threshold",
                              0.01, 10.0, 2, "Flux change threshold",
                              info_key="radiation.coupling_flux_threshold")
        self.add_double_field(coupling_layout, 4, "Face Fraction (legacy mode only):",
                              "radiation.coupling_face_fraction",
                              0.001, 0.1, 3,
                              "Fraction of faces still changing to stop. IGNORED unless "
                              "radiation.coupling_mode is 'legacy': this exit criterion has "
                              "no error bound and made a coupled run's answer depend on how "
                              "the window was chunked (PTM-89). The engine prints a warning "
                              "if you change it without also setting coupling_mode: legacy.")

        # A label that says "legacy mode only" next to an always-editable field
        # is a label that lies. Gate it on the mode it names.
        self.coupling_mode_note = QLabel("")
        self.coupling_mode_note.setWordWrap(True)
        self.coupling_mode_note.setStyleSheet("color: gray; font-size: 10px;")
        coupling_layout.addWidget(self.coupling_mode_note, 5, 0, 1, 2)
        self.widgets["radiation.coupling_mode"].currentTextChanged.connect(
            self._update_coupling_mode_gating)

        coupling_section.add_widget(coupling_widget)
        layout.addWidget(coupling_section)

        # --- View Factor Options (collapsed, merges caching + compression) ---
        vf_section = self.advanced_section("View Factor Options")
        vf_widget = QWidget()
        vf_layout = QGridLayout(vf_widget)
        vf_layout.setColumnStretch(1, 1)

        self.add_file_field(vf_layout, 0, "View Factor File:", "radiation.view_factor_file",
                            "Binary Files (*.bin);;H-Matrix Files (*.hmat);;All Files (*)",
                            "Save/load view factors to/from file",
                            info_key="radiation.view_factor_file")
        self.add_bool_field(vf_layout, 1, "Enable H-Matrix Compression", "radiation.use_hmatrix",
                            "Use H-matrix compression for view factors. Recommended for meshes with 50K-1M faces.",
                            info_key="radiation.use_hmatrix")
        self.add_bool_field(vf_layout, 2, "Use ACA (Adaptive Cross Approximation)", "radiation.use_aca",
                            "Use ACA for faster compression (default: enabled)",
                            info_key="radiation.use_aca")
        self.add_double_field(vf_layout, 3, "ACA Tolerance:", "radiation.aca_tolerance",
                              0.0001, 0.1, 4, "ACA relative tolerance",
                              info_key="radiation.aca_tolerance")
        self.add_int_field(vf_layout, 4, "ACA Max Rank:", "radiation.aca_max_rank",
                           1, 200, "Maximum rank before falling back to sparse")
        self.add_int_field(vf_layout, 5, "H-Matrix Threads:", "radiation.hmatrix_threads",
                           0, 128, "Threads for H-matrix construction (0 = auto)")

        # Which view-factor builder, and how accurately it integrates the near
        # field. Both change the numbers a coupled run produces.
        self.add_int_field(vf_layout, 6, "Near-Field Quadrature Order:",
                           "radiation.nearfield_quadrature_order", 1, 8,
                           "Order of the symmetric double-area quadrature used "
                           "for near-field pairs. 3 is the default and the "
                           "validated production value; order 1 is the centroid "
                           "estimator with no near-field treatment at all and "
                           "reads +51.7% on the touching-pair closed form "
                           "(order 3 reads +18.5%). The engine calls order 1 "
                           "\"a diagnostic setting, not a production one\".")
        self.add_bool_field(vf_layout, 7, "Build View Factors by Monte Carlo",
                            "radiation.use_monte_carlo_vf",
                            "Select the Monte Carlo view-factor builder instead "
                            "of the analytic/H-matrix path. A different builder "
                            "is a different answer, not just a different speed.")
        self.add_int_field(vf_layout, 8, "Monte Carlo Rays per Face:",
                           "radiation.mc_rays_per_face", 1000, 10000000,
                           "Rays cast per face by the Monte Carlo builder. "
                           "Used only when that builder is selected.")

        self.quadrature_note = QLabel("")
        self.quadrature_note.setWordWrap(True)
        self.quadrature_note.setStyleSheet("color: gray; font-size: 10px;")
        vf_layout.addWidget(self.quadrature_note, 9, 0, 1, 2)
        self.widgets["radiation.nearfield_quadrature_order"].valueChanged.connect(
            self._update_vf_builder_gating)
        self.widgets["radiation.use_monte_carlo_vf"].toggled.connect(
            self._update_vf_builder_gating)

        vf_section.add_widget(vf_widget)
        layout.addWidget(vf_section)

        # --- Sparse View Factors (collapsed) ---
        sparse_section = self.advanced_section("Sparse View Factors (>1M Faces)")
        sparse_widget = QWidget()
        sparse_layout = QGridLayout(sparse_widget)
        sparse_layout.setColumnStretch(1, 1)

        self.add_bool_field(sparse_layout, 0, "Adaptive Neighbor Selection", "radiation.adaptive_neighbors",
                            "Enable per-face adaptive selection for sparse view factors",
                            info_key="radiation.adaptive_neighbors")
        self.add_double_field(sparse_layout, 1, "Captured Energy Target:", "radiation.captured_energy_target",
                              0.5, 1.0, 2, "Target fraction of radiative potential to capture (0.95 = 95%)",
                              info_key="radiation.captured_energy_target")
        self.add_int_field(sparse_layout, 2, "Max Neighbors:", "radiation.max_neighbors",
                           10, 5000, "Maximum neighbors per face (hard ceiling)",
                           info_key="radiation.max_neighbors")
        self.add_int_field(sparse_layout, 3, "Min Neighbors:", "radiation.min_neighbors",
                           5, 100, "Minimum neighbors per face",
                           info_key="radiation.min_neighbors")

        sparse_section.add_widget(sparse_widget)
        layout.addWidget(sparse_section)

        # Advisory pre-run cost estimate (disk + solve time), updated live as
        # radiation / mesh / body fields change. Order-of-magnitude only.
        self.vf_estimate_label = QLabel("")
        self.vf_estimate_label.setWordWrap(True)
        self.vf_estimate_label.setObjectName("vfEstimateLabel")
        self.vf_estimate_label.setStyleSheet(
            "color: #64B5F6; font-size: 11px; padding: 6px; "
            "border: 1px solid #444; border-radius: 4px;")
        layout.addWidget(self.vf_estimate_label)

        # The H-matrix compression sub-options (ACA method, ACA tolerance/rank,
        # build threads) only take effect when H-matrix compression is enabled:
        # in the engine ACA is simply the compression method *inside* the
        # H-matrix path (radiation_exchange.cpp: compression_method = use_aca ?
        # ACA : SVD). Gray them out when H-matrix is off so the panel reflects
        # that dependency. This never changes the saved YAML — disabled widgets
        # keep their stored values.
        self.widgets["radiation.use_hmatrix"].toggled.connect(
            self._update_hmatrix_suboptions)

        layout.addStretch()
        self.create_scroll_area(content)

        self._update_hmatrix_suboptions()
        self._update_coupling_mode_gating()
        self._update_vf_builder_gating()
        self._update_beaming_gating()

        # Recompute the estimate whenever a radiation field changes.
        self.value_changed.connect(self.update_vf_estimate)
        self.update_vf_estimate()

    # ── Beaming facet-scale rule (DP-4) ──────────────────────────────

    def _beaming_policy(self):
        from gui_core.facet_scale import beaming_policy
        try:
            return beaming_policy(self.config_manager.get_config(),
                                  self._project_root())
        except Exception as exc:                 # never break the tab
            from gui_core.facet_scale import BeamingPolicy
            return BeamingPolicy(
                allowed=False,
                reason=f"The facet-scale check could not run ({exc}), so "
                       f"beaming is not offered. Refusing is the safe "
                       f"direction: a mesh at the roughness scale would "
                       f"double-count it.")

    def _on_beaming_toggled(self, checked):
        """Turning the switch on with no slope is fatal at engine startup."""
        if checked:
            knob = self.widgets["radiation.rms_slope_deg"]
            if knob.value() <= 0.0:
                # 32.5 deg = Jhoti et al. (2024) lunar highlands, the validated
                # polar value. Announced in the note, never silent.
                knob.setValue(32.5)
        self._update_beaming_gating()

    def _update_beaming_gating(self, _checked=None):
        """Apply the facet-scale and body rules to the beaming control pair.

        The rule governs whether beaming may be turned ON here. It does NOT
        silently turn OFF a config that arrived with it on: rewriting someone's
        physics behind their back is the failure mode this whole stage exists to
        remove, and the rule can decline for reasons that are not "the mesh is
        too fine" -- a topographic mesh, a non-PLY import, an unreadable path.
        An inherited setting the rule disagrees with is kept, said loudly, and
        left switchable so the user can act on it.
        """
        policy = self._beaming_policy()
        switch = self.widgets["radiation.beaming"]
        knob = self.widgets["radiation.rms_slope_deg"]

        # Beaming is INERT with radiation exchange off -- which is the GUI's own
        # default. The engine copies beaming/rms_slope_deg into the radiation
        # config only inside its radiation-enabled blocks
        # (ptm_thermal_sim.cpp:15402, :17717), so the run prints nothing about
        # Smith at all and the operator reads a Lambertian result as a beaming
        # one. That is the failure the engine's own FATAL comment names. Reported
        # here, in the tab, the same way "Face Fraction (legacy mode only)" is
        # gated on coupling_mode.
        if not self.config_manager.get_value("radiation.enabled", False):
            switch.setEnabled(False)
            knob.setEnabled(False)
            self.beaming_ack.setVisible(False)
            self.beaming_note.setStyleSheet(
                "color: #FFB74D; font-size: 10px; padding: 4px;")
            self.beaming_note.setText(
                "Radiation exchange is OFF, so beaming is inert: the engine "
                "reads these two keys only on the radiation path and the run "
                "would say nothing about Smith at all. Enable Radiation "
                "Exchange above to use it."
                + (f"\n{policy.reason}" if switch.isChecked() else ""))
            if switch.isChecked():
                logging.warning(
                    "[beaming] radiation.beaming is ON but radiation.enabled is "
                    "off; the engine will ignore it entirely.")
            return

        # Reset the acknowledgment whenever the thing being acknowledged
        # changes. Hiding it while leaving it CHECKED carried a tick made for
        # one mesh over to the next one, so a second warn-tier mesh could be
        # enabled without the user ever seeing its facet scale -- which is the
        # entire content of that tier.
        scale_id = (policy.scale.source if policy.scale else None,
                    None if not policy.scale or policy.scale.meters is None
                    else round(policy.scale.meters, 6))
        if scale_id != getattr(self, "_beaming_ack_scale", None):
            self.beaming_ack.blockSignals(True)
            self.beaming_ack.setChecked(False)
            self.beaming_ack.blockSignals(False)
            self._beaming_ack_scale = scale_id
        self.beaming_ack.setVisible(policy.needs_acknowledgment)
        ack_ok = (not policy.needs_acknowledgment) or self.beaming_ack.isChecked()
        may_turn_on = policy.allowed and ack_ok
        already_on = switch.isChecked()

        # Enabled when the rule permits it, or when it is already on so the
        # user can still turn it off.
        switch.setEnabled(may_turn_on or already_on)
        knob.setEnabled(already_on)

        text = policy.reason
        if already_on and not policy.allowed:
            # The rule REFUSES this mesh or body, and beaming is on anyway.
            text = ("BEAMING IS ON AND THIS MESH DOES NOT MEET THE RULE. It "
                    "has been left as the configuration set it -- nothing here "
                    "rewrites your physics -- but check it before trusting the "
                    "run.\n" + policy.reason)
            logging.warning("[beaming] radiation.beaming is ON and the "
                            "facet-scale rule declines: %s", policy.reason)
            color = "#E57373"
        elif already_on and policy.needs_acknowledgment and not ack_ok:
            # The rule ALLOWS this mesh; it only wants the number acknowledged.
            # Saying "does not meet the rule" here contradicts the reason text
            # printed immediately below it, and every 5 cm to 1 m production
            # mesh would raise a red alarm on open.
            text = ("Beaming is on and this mesh is in the band where some "
                    "roughness may already be resolved. Confirm below.\n"
                    + policy.reason)
            color = "#FFB74D"
        elif not policy.allowed:
            color = "#E57373"
        elif policy.needs_acknowledgment:
            color = "#FFB74D"
        else:
            color = "#81C784"
        self.beaming_note.setStyleSheet(
            f"color: {color}; font-size: 10px; padding: 4px;")
        self.beaming_note.setText(text)

    def on_mesh_changed(self):
        """The mesh decides the facet scale, so it decides the beaming rule.

        Without this the rule failed OPEN: enable beaming on a lunar icosphere,
        then import a millimeter-scale mesh on the Mesh tab, and `beaming: true`
        went to the engine on exactly the mesh the rule exists to refuse.
        """
        self._update_beaming_gating()
        self.update_vf_estimate()

    def _update_coupling_mode_gating(self, _mode=None):
        """Face Fraction is editable only in the mode whose name it carries."""
        mode = self.widgets["radiation.coupling_mode"].currentText()
        legacy = (mode == "legacy")
        ff = self.widgets.get("radiation.coupling_face_fraction")
        if ff is not None:
            ff.setEnabled(legacy)
        if legacy:
            self.coupling_mode_note.setText(
                "legacy re-introduces PTM-89: the face-fraction exit has no "
                "error bound, so a coupled run's answer depends on how the "
                "window was chunked. Use it to reproduce a published run.")
        elif mode == "fixed":
            self.coupling_mode_note.setText(
                "fixed runs exactly Max Coupling Iterations passes; Face "
                "Fraction is not read.")
        else:
            self.coupling_mode_note.setText(
                "bounded is the PTM-89 fix and the engine default; Face "
                "Fraction is not read.")

    def _update_vf_builder_gating(self, _value=None):
        """Warn on quadrature order 1, and gate the Monte Carlo ray count."""
        order = self.widgets["radiation.nearfield_quadrature_order"].value()
        mc = self.widgets["radiation.use_monte_carlo_vf"].isChecked()
        rays = self.widgets.get("radiation.mc_rays_per_face")
        if rays is not None:
            rays.setEnabled(mc)
        if order == 1:
            # The engine's own wording, so the two do not drift apart.
            self.quadrature_note.setText(
                "Order 1 is the centroid estimator: no near-field treatment at "
                "all. On the touching-pair closed form it reads +51.7% (order 3 "
                "reads +18.5%). This is a diagnostic setting, not a production "
                "one.")
        elif order < 3:
            self.quadrature_note.setText(
                f"Order {order} is below the validated production value of 3.")
        else:
            self.quadrature_note.setText("")

    def _update_hmatrix_suboptions(self, _checked=None):
        """Enable the H-matrix compression sub-fields only when H-matrix is on."""
        on = self.widgets["radiation.use_hmatrix"].isChecked()
        for key in ("radiation.use_aca", "radiation.aca_tolerance",
                    "radiation.aca_max_rank", "radiation.hmatrix_threads"):
            w = self.widgets.get(key)
            if w is not None:
                w.setEnabled(on)

    def _project_root(self):
        from pathlib import Path
        from gui_core.preflight import find_project_root
        from gui_core.simulation_runner import SimulationRunner
        try:
            return find_project_root(SimulationRunner.find_ptm_executable())
        except Exception:
            return Path.cwd()

    def update_vf_estimate(self):
        """Refresh the advisory view-factor cost label from the current config."""
        from gui_core.preflight import estimate_vf_cost
        if not self.config_manager.get_value("radiation.enabled", False):
            self.vf_estimate_label.setText(
                "Enable radiation exchange to see a view-factor cost estimate.")
            return
        try:
            est = estimate_vf_cost(self.config_manager.get_config(),
                                   self._project_root())
        except Exception as e:  # advisory only -- never break the tab
            self.vf_estimate_label.setText(
                f"View-factor cost estimate unavailable: {e}")
            return
        if not est.available:
            self.vf_estimate_label.setText(
                est.lines[0] if est.lines else "View-factor cost: not estimable.")
            return
        text = est.headline()
        for line in est.lines:
            text += f"\n{line}"
        for w in est.warnings:
            text += f"\n⚠ {w}"
        self.vf_estimate_label.setText(text)

    def on_body_changed(self, _body_name=None):
        """Body change affects the mesh face count and the beaming rule."""
        self.update_vf_estimate()
        self._update_beaming_gating()

    def load_from_config(self):
        config = self.config_manager.get_config()
        rad = config.get("radiation", {})

        self.set_bool("radiation.enabled", rad.get("enabled", False))
        self.set_bool("radiation.verbose", rad.get("verbose", False))
        # The fallbacks here track the engine's own defaults (they only fire for
        # a config missing the key entirely). use_gpu was True and
        # max_coupling_iterations 10 while the engine said False and 16.
        self.set_bool("radiation.use_gpu", rad.get("use_gpu", False))
        self.set_numeric("radiation.min_shadow_fraction", rad.get("min_shadow_fraction", 0.3))
        self.set_enum("radiation.coupling_mode",
                      rad.get("coupling_mode", "bounded"),
                      COUPLING_MODE_ALIASES)
        self.set_numeric("radiation.max_coupling_iterations", rad.get("max_coupling_iterations", 16))
        self.set_numeric("radiation.coupling_temp_threshold", rad.get("coupling_temp_threshold", 0.5))
        self.set_numeric("radiation.coupling_flux_threshold", rad.get("coupling_flux_threshold", 0.1))
        self.set_numeric("radiation.coupling_face_fraction", rad.get("coupling_face_fraction", 0.01))
        self.set_text("radiation.view_factor_file", rad.get("view_factor_file", ""))
        self.set_bool("radiation.use_hmatrix", rad.get("use_hmatrix", False))
        self.set_bool("radiation.use_aca", rad.get("use_aca", True))
        self.set_numeric("radiation.aca_tolerance", rad.get("aca_tolerance", 0.001))
        self.set_numeric("radiation.aca_max_rank", rad.get("aca_max_rank", 50))
        self.set_numeric("radiation.hmatrix_threads", rad.get("hmatrix_threads", 0))
        self.set_numeric("radiation.nearfield_quadrature_order",
                         rad.get("nearfield_quadrature_order", 3))
        self.set_bool("radiation.use_monte_carlo_vf",
                      rad.get("use_monte_carlo_vf", False))
        self.set_numeric("radiation.mc_rays_per_face",
                         rad.get("mc_rays_per_face", 100000))
        # Sparse view factors
        self.set_bool("radiation.adaptive_neighbors", rad.get("adaptive_neighbors", True))
        self.set_numeric("radiation.captured_energy_target", rad.get("captured_energy_target", 0.95))
        self.set_numeric("radiation.max_neighbors", rad.get("max_neighbors", 500))
        self.set_numeric("radiation.min_neighbors", rad.get("min_neighbors", 20))
        # Beaming: honor what the file said. The rule then decides whether it
        # may be turned ON from here -- it never turns one OFF; a setting it
        # disagrees with is kept and said loudly.
        self.set_numeric("radiation.rms_slope_deg", rad.get("rms_slope_deg", 0.0))
        self.widgets["radiation.beaming"].blockSignals(True)
        self.set_bool("radiation.beaming", rad.get("beaming", False))
        self.widgets["radiation.beaming"].blockSignals(False)
        self._legacy_beaming_note()
        self._update_hmatrix_suboptions()
        self._update_coupling_mode_gating()
        self._update_vf_builder_gating()
        self._update_beaming_gating()
        self.update_vf_estimate()

    def _legacy_beaming_note(self):
        """A pre-2026-08-25 config had no `beaming` key; the switch was implied.

        config_parser.cpp:631 honors that bridge and prints a NOTE. The GUI
        writes the key explicitly, so such a config loaded here and run would
        carry `beaming: false` and silently demote the run to Lambertian.

        The detection CANNOT be done from `self.config_manager.get_config()`:
        the merge with DEFAULT_CONFIG always supplies `beaming`, so "the file
        did not mention it" is information only ConfigManager.load_from_yaml
        has. It records it on `legacy_implicit_beaming`; this reads that.
        """
        slope = getattr(self.config_manager, "legacy_implicit_beaming", None)
        if slope is None:
            return
        # Consume it. It survived only until now: leaving it set meant every
        # LATER load_from_config() on the same manager re-forced beaming on --
        # and the dashboard's thermal adapter calls that on every chained mesh
        # handoff, so a user who read this warning and deliberately switched
        # beaming off would silently get it back.
        self.config_manager.legacy_implicit_beaming = None
        self.widgets["radiation.beaming"].blockSignals(True)
        self.set_bool("radiation.beaming", True)
        self.widgets["radiation.beaming"].blockSignals(False)
        self.config_manager.set_value("radiation.beaming", True)
        logging.warning(
            "[beaming] this config has rms_slope_deg = %s and no 'beaming' key: "
            "the pre-2026-08-25 implicit switch. Beaming has been turned ON to "
            "match what the engine would do, and will be written explicitly on "
            "the next save.", slope)


class ThermophysicalTab(BaseConfigTab):
    """Thermophysical properties configuration tab

    Engine reality check (config_parser.cpp, "Parse Thermophysical
    section"): for BUILT-IN bodies the engine ignores plain scalar keys in
    the thermophysical YAML section and honors only the *_override keys --
    which exist for albedo, emissivity, ks, kd and H. The remaining properties
    (rhos, rhod, cp0, Qb, chi) can only be customized through
    generic_body.surface, i.e. with a Custom/HORIZONS body. This tab
    therefore DISABLES those five rows when a built-in body is selected
    instead of silently accepting values the engine would never see (the
    old behavior wrote dead placeholders like `cp0: 800.0`).
    """

    # Properties with an engine *_override path for built-in bodies.
    # H moved here on 2026-08-25: the engine has read thermophysical.H_override
    # all along (config_parser.cpp:759, applied at ptm_thermal_sim.cpp:14778,
    # which sets planet.H and therefore works on a built-in body). This tab
    # nevertheless disabled the H row on every built-in body, so the regolith
    # scale height -- arguably THE lunar thermal parameter -- was unreachable.
    BUILTIN_OVERRIDABLE = ("albedo", "emissivity", "ks", "kd", "H")
    # Properties with NO engine path for built-in bodies
    BUILTIN_UNSUPPORTED = ("rhos", "rhod", "cp0", "Qb", "chi")

    # Default properties by body (matches planetary_properties.h)
    BODY_DEFAULTS = {
        "Europa": {
            "albedo": 0.5, "emissivity": 0.9, "Qb": 0.03,
            "ks": 0.0096, "kd": 0.05, "rhos": 500.0, "rhod": 700.0,
            "H": 0.07, "cp0": 0.0, "incidence_angle_albedo": False,
            "chi": 0.0, "rAU": 5.2, "cpCoeff": [7.49],
        },
        "Moon": {
            "albedo": 0.12, "emissivity": 0.95, "Qb": 0.018,
            "ks": 7.4e-4, "kd": 3.4e-3, "rhos": 1100.0, "rhod": 1800.0,
            # Global model -> default Hayne (2017) angular albedo (False); Hapke
            # is opt-in (Paul's ruling). Was True (Hapke).
            "H": 0.06, "cp0": -3.6125, "incidence_angle_albedo": False,
            "chi": 2.7, "rAU": 1.0, "cpCoeff": [2.7431, 2.3616e-3, -1.2340e-5, 8.9093e-9],
        },
        "Mercury": {
            "albedo": 0.07, "emissivity": 0.93, "Qb": 0.02,
            "ks": 0.002, "kd": 2.0, "rhos": 1800.0, "rhod": 2200.0,
            # Mercury defaults to Hapke (True) per Paul's ruling: the Hayne
            # (2017) angular-albedo fit is lunar-calibrated, so it is NOT the
            # right default for Mercury. Moon stays Hayne (False); Mercury is
            # the deliberate exception.
            "H": 0.1, "cp0": 800.0, "incidence_angle_albedo": True,
            "chi": 2.7, "rAU": 0.387, "cpCoeff": [0.25, 0.0],
        },
        "Enceladus": {
            "albedo": 0.81, "emissivity": 0.9, "Qb": 0.025,
            "ks": 0.001, "kd": 0.01, "rhos": 500.0, "rhod": 700.0,
            "H": 0.05, "cp0": 0.0, "incidence_angle_albedo": False,
            "chi": 0.0, "rAU": 9.54, "cpCoeff": [7.49],
        },
        "Titan": {
            "albedo": 0.22, "emissivity": 0.9, "Qb": 0.005,
            "ks": 0.02, "kd": 0.2, "rhos": 500.0, "rhod": 700.0,
            "H": 0.1, "cp0": 0.0, "incidence_angle_albedo": False,
            "chi": 0.0, "rAU": 9.54, "cpCoeff": [7.49],
        },
        "Mars": {
            "albedo": 0.25, "emissivity": 0.95, "Qb": 0.02,
            "ks": 0.02, "kd": 2.0, "rhos": 1500.0, "rhod": 2000.0,
            "H": 0.1, "cp0": 800.0, "incidence_angle_albedo": False,
            "chi": 2.7, "rAU": 1.524, "cpCoeff": [],
        },
        "Ceres": {
            "albedo": 0.09, "emissivity": 0.95, "Qb": 0.01,
            "ks": 0.001, "kd": 0.01, "rhos": 1200.0, "rhod": 1500.0,
            "H": 0.05, "cp0": 800.0, "incidence_angle_albedo": False,
            "chi": 0.0, "rAU": 2.77, "cpCoeff": [],
        },
        # Generic asteroid defaults
        "Asteroid": {
            "albedo": 0.15, "emissivity": 0.95, "Qb": 0.0,
            "ks": 0.01, "kd": 0.5, "rhos": 1500.0, "rhod": 2500.0,
            "H": 0.05, "cp0": 800.0, "incidence_angle_albedo": False,
            "chi": 2.7, "rAU": 2.5, "cpCoeff": [],
        },
        # Asteroid taxonomy presets
        "S-type Asteroid": {
            "albedo": 0.20, "emissivity": 0.95, "Qb": 0.0,
            "ks": 0.01, "kd": 0.5, "rhos": 1800.0, "rhod": 3000.0,
            "H": 0.05, "cp0": 800.0, "incidence_angle_albedo": False,
            "chi": 2.7, "rAU": 2.3, "cpCoeff": [],
        },
        "C-type Asteroid": {
            "albedo": 0.05, "emissivity": 0.95, "Qb": 0.0,
            "ks": 0.005, "kd": 0.2, "rhos": 1300.0, "rhod": 2000.0,
            "H": 0.05, "cp0": 800.0, "incidence_angle_albedo": False,
            "chi": 0.0, "rAU": 2.7, "cpCoeff": [],
        },
        "Bare Rock": {
            "albedo": 0.10, "emissivity": 0.95, "Qb": 0.0,
            "ks": 2.0, "kd": 2.5, "rhos": 3000.0, "rhod": 3500.0,
            "H": 0.5, "cp0": 800.0, "incidence_angle_albedo": False,
            "chi": 2.7, "rAU": 1.5, "cpCoeff": [],
        },
        "Icy Body": {
            "albedo": 0.60, "emissivity": 0.90, "Qb": 0.0,
            "ks": 0.001, "kd": 0.01, "rhos": 500.0, "rhod": 900.0,
            "H": 0.05, "cp0": 800.0, "incidence_angle_albedo": False,
            "chi": 0.0, "rAU": 5.0, "cpCoeff": [7.49],
        },
    }

    def init_ui(self):
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        # Info label (text is kept in sync by _update_builtin_support)
        self.info_label = QLabel(
            "Properties shown are defaults for the selected body. "
            "Uncheck to customize.")
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet(
            "color: #64b5f6; font-size: 10px; padding: 5px;")
        layout.addWidget(self.info_label)

        # Preset dropdown (replaces 9 buttons)
        preset_row = QHBoxLayout()
        preset_label = QLabel("Preset:")
        preset_label.setToolTip("Load thermophysical properties for a known body or material type")
        preset_row.addWidget(preset_label)

        self.preset_combo = QComboBox()
        self.preset_combo.addItems([
            "-- Select Preset --",
            "Europa", "Moon", "Mercury", "Mars", "Ceres",
            "S-type Asteroid", "C-type Asteroid", "Bare Rock", "Icy Body"
        ])
        self.preset_combo.currentIndexChanged.connect(self._on_preset_selected)
        preset_row.addWidget(self.preset_combo, stretch=1)
        layout.addLayout(preset_row)

        # --- Surface Properties (expanded) ---
        surface_section = CollapsibleSection("Surface Properties", collapsed=False)
        surface_widget = QWidget()
        surface_layout = QGridLayout(surface_widget)
        surface_layout.setColumnStretch(1, 1)

        self.add_property_with_default(surface_layout, 0, "Bond Albedo:", "thermophysical.albedo",
                                       "thermophysical.use_default_albedo", 0.0, 1.0, 3,
                                       "Fraction of incident solar radiation reflected (0-1)",
                                       info_key="thermophysical.albedo")
        self.add_property_with_default(surface_layout, 1, "Emissivity:", "thermophysical.emissivity",
                                       "thermophysical.use_default_emissivity", 0.0, 1.0, 3,
                                       "Thermal emissivity for IR radiation (0-1)",
                                       info_key="thermophysical.emissivity")
        # Incidence-angle albedo MODEL selector (Hayne 2017 vs Hapke). The
        # engine key is a single boolean (incidence_angle_albedo): false ->
        # Hayne (2017) angular albedo, true -> Hapke directional model. There
        # is no separate "off" state — false already selects the Hayne angular
        # model, which is the default for global bodies (Paul's ruling).
        ica_label = QLabel("Incidence-Angle Albedo:")
        self._apply_tooltip(ica_label, "thermophysical.incidence_angle_albedo")
        surface_layout.addWidget(ica_label, 2, 0)

        self.incidence_combo = QComboBox()
        self.incidence_combo.addItem("Hayne (2017)", False)
        self.incidence_combo.addItem("Hapke", True)
        self.incidence_combo.setToolTip(
            "Angular albedo model.\n"
            "Hayne (2017): albedo increases toward grazing incidence per the "
            "lunar Hayne et al. (2017) fit — the default for global bodies.\n"
            "Hapke: directional (photometric) albedo model — opt-in.\n"
            "Sets the engine key incidence_angle_albedo (Hayne = false, "
            "Hapke = true).")
        self.incidence_combo.currentIndexChanged.connect(
            self._on_incidence_model_changed)
        surface_layout.addWidget(self.incidence_combo, 2, 1)
        # Registered under the engine key so the existing body-preset / load
        # paths keep finding it; the value it holds is the Hayne/Hapke boolean.
        self.widgets["thermophysical.incidence_angle_albedo"] = self.incidence_combo

        surface_section.add_widget(surface_widget)
        layout.addWidget(surface_section)

        # --- Thermal Conductivity & Density (expanded) ---
        cond_section = CollapsibleSection("Thermal Conductivity & Density", collapsed=False)
        cond_widget = QWidget()
        cond_layout = QGridLayout(cond_widget)
        cond_layout.setColumnStretch(1, 1)

        self.add_property_with_default(cond_layout, 0, "Surface ks (W/m/K):", "thermophysical.ks",
                                       "thermophysical.use_default_ks", 1e-6, 10.0, 6,
                                       "Surface thermal conductivity",
                                       info_key="thermophysical.ks")
        self.add_property_with_default(cond_layout, 1, "Deep kd (W/m/K):", "thermophysical.kd",
                                       "thermophysical.use_default_kd", 1e-6, 100.0, 6,
                                       "Deep thermal conductivity (at depth >> H)",
                                       info_key="thermophysical.kd")
        self.add_property_with_default(cond_layout, 2, "Scale Height H (m):", "thermophysical.H",
                                       "thermophysical.use_default_H", 0.001, 1.0, 4,
                                       "e-folding depth for property variation",
                                       info_key="thermophysical.H")
        self.add_property_with_default(cond_layout, 3, "Radiative \u03c7:", "thermophysical.chi",
                                       "thermophysical.use_default_chi", 0.0, 100.0, 2,
                                       "Radiative conductivity parameter (k = kc \u00d7 [1 + \u03c7(T/350)\u00b3])",
                                       info_key="thermophysical.chi")
        self.add_property_with_default(cond_layout, 4, "Surface \u03c1s (kg/m\u00b3):", "thermophysical.rhos",
                                       "thermophysical.use_default_rhos", 100.0, 5000.0, 1,
                                       "Surface bulk density",
                                       info_key="thermophysical.rhos")
        self.add_property_with_default(cond_layout, 5, "Deep \u03c1d (kg/m\u00b3):", "thermophysical.rhod",
                                       "thermophysical.use_default_rhod", 100.0, 5000.0, 1,
                                       "Deep bulk density (at depth >> H)",
                                       info_key="thermophysical.rhod")

        # Which k(T, rho) law the solver uses. The engine validates the string
        # and names the valid values (config_parser.cpp: hayne2017 |
        # martinez2021), so an unknown one is caught there, not guessed at.
        self.add_combo_field(cond_layout, 6, "Conductivity Model:",
                             "thermophysical.conductivity_model",
                             ["hayne2017", "martinez2021"],
                             "Temperature-dependent conductivity law.\n"
                             "hayne2017: k = kc x [1 + chi (T/350)^3], the "
                             "lunar-calibrated default.\n"
                             "martinez2021: the Martinez et al. (2021) "
                             "formulation, which depends on density as well as "
                             "temperature.")

        cond_section.add_widget(cond_widget)
        layout.addWidget(cond_section)

        # --- Thermal Inertia (expanded, derived) ---
        ti_section = CollapsibleSection("Thermal Inertia", collapsed=False)
        ti_widget = QWidget()
        ti_layout = QGridLayout(ti_widget)
        ti_layout.setColumnStretch(1, 1)

        ti_label = QLabel("TI (SI):")
        self._apply_tooltip(ti_label, "thermophysical.thermal_inertia")
        ti_layout.addWidget(ti_label, 0, 0)

        self.ti_spin = QDoubleSpinBox()
        self.ti_spin.setRange(0.0, 10000.0)
        self.ti_spin.setDecimals(1)
        self.ti_spin.setToolTip(
            "Thermal inertia TI = \u221a(k \u00d7 \u03c1 \u00d7 cp).\n"
            "Setting this value activates constant-TI mode:\n"
            "  \u2022 \u03c7 = 0 (no radiative conductivity, k = ks)\n"
            "  \u2022 cp = TI\u00b2 / (ks \u00d7 \u03c1s), constant\n"
            "  \u2022 No temperature dependence in k or cp"
        )
        self.ti_spin.valueChanged.connect(self._on_ti_changed)
        ti_layout.addWidget(self.ti_spin, 0, 1)

        # Read-only display: cp(T_eq) and T_eq
        cp_label = QLabel("cp(T_eq):")
        cp_label.setToolTip("Heat capacity evaluated at equilibrium temperature")
        ti_layout.addWidget(cp_label, 1, 0)

        self.ti_cp_label = QLabel("---")
        self.ti_cp_label.setStyleSheet("color: #b0bec5;")
        ti_layout.addWidget(self.ti_cp_label, 1, 1)

        teq_label = QLabel("T_eq:")
        teq_label.setToolTip("Equilibrium temperature from albedo, emissivity, distance")
        ti_layout.addWidget(teq_label, 1, 2)

        self.ti_teq_label = QLabel("---")
        self.ti_teq_label.setStyleSheet("color: #b0bec5;")
        ti_layout.addWidget(self.ti_teq_label, 1, 3)

        ti_section.add_widget(ti_widget)
        layout.addWidget(ti_section)

        # --- Heat Capacity & Internal Heat (collapsed) ---
        cp_heat_section = self.advanced_section("Heat Capacity & Internal Heat")
        cp_heat_widget = QWidget()
        cp_heat_layout = QGridLayout(cp_heat_widget)
        cp_heat_layout.setColumnStretch(1, 1)

        self.add_property_with_default(cp_heat_layout, 0, "Reference cp0 (J/kg/K):", "thermophysical.cp0",
                                       "thermophysical.use_default_cp0", -100.0, 2000.0, 4,
                                       "Base heat capacity (may be T-dependent)",
                                       info_key="thermophysical.cp0")
        self.add_property_with_default(cp_heat_layout, 1, "Heat Flow Qb (W/m\u00b2):", "thermophysical.Qb",
                                       "thermophysical.use_default_Qb", 0.0, 1.0, 4,
                                       "Geothermal/internal heat flux at bottom boundary",
                                       info_key="thermophysical.Qb")

        cp_heat_section.add_widget(cp_heat_widget)
        layout.addWidget(cp_heat_section)

        layout.addStretch()
        self.create_scroll_area(content)

    def add_property_with_default(self, layout, row, label_text, value_key, default_key,
                                   min_val, max_val, decimals, tooltip="", info_key=None):
        """Add a property field with a 'Use Default' checkbox"""
        label = QLabel(label_text)
        label.setToolTip(tooltip)
        self._apply_tooltip(label, info_key)
        layout.addWidget(label, row, 0)

        value_widget = QDoubleSpinBox()
        value_widget.setRange(min_val, max_val)
        value_widget.setDecimals(decimals)
        value_widget.setToolTip(tooltip)
        value_widget.valueChanged.connect(lambda v: self.on_widget_changed(value_key, v))
        layout.addWidget(value_widget, row, 1)

        default_check = QCheckBox("Use Default")
        default_check.setToolTip("Use body-specific default value")
        default_check.stateChanged.connect(
            lambda state: self.on_default_toggled(value_key, default_key, state)
        )
        layout.addWidget(default_check, row, 2)

        self.widgets[value_key] = value_widget
        self.widgets[default_key] = default_check

    def on_default_toggled(self, value_key, default_key, state):
        """Handle use default checkbox toggle"""
        use_default = (state == Qt.CheckState.Checked.value)
        self.config_manager.set_value(default_key, use_default)
        self.widgets[value_key].setEnabled(not use_default)

        if use_default:
            self.update_to_body_defaults()

        self.value_changed.emit()

    def _on_preset_selected(self, index):
        """Handle preset combo box selection."""
        if index <= 0:
            return
        self._apply_preset(self.preset_combo.currentText())
        self.preset_combo.blockSignals(True)
        self.preset_combo.setCurrentIndex(0)
        self.preset_combo.blockSignals(False)

    def _apply_preset(self, preset_name):
        """Apply a thermophysical preset to all property fields."""
        defaults = self.BODY_DEFAULTS.get(preset_name, {})
        if not defaults:
            return

        properties = [
            ("thermophysical.albedo", "thermophysical.use_default_albedo", "albedo"),
            ("thermophysical.emissivity", "thermophysical.use_default_emissivity", "emissivity"),
            ("thermophysical.ks", "thermophysical.use_default_ks", "ks"),
            ("thermophysical.kd", "thermophysical.use_default_kd", "kd"),
            ("thermophysical.H", "thermophysical.use_default_H", "H"),
            ("thermophysical.rhos", "thermophysical.use_default_rhos", "rhos"),
            ("thermophysical.rhod", "thermophysical.use_default_rhod", "rhod"),
            ("thermophysical.cp0", "thermophysical.use_default_cp0", "cp0"),
            ("thermophysical.Qb", "thermophysical.use_default_Qb", "Qb"),
            ("thermophysical.chi", "thermophysical.use_default_chi", "chi"),
        ]

        for value_key, default_key, attr in properties:
            if attr in defaults and value_key in self.widgets and default_key in self.widgets:
                # Uncheck "Use Default"
                self.widgets[default_key].blockSignals(True)
                self.widgets[default_key].setChecked(False)
                self.widgets[default_key].blockSignals(False)

                # Set value
                self.widgets[value_key].setEnabled(True)
                self.widgets[value_key].blockSignals(True)
                self.widgets[value_key].setValue(defaults[attr])
                self.widgets[value_key].blockSignals(False)

                # Update config
                self.config_manager.set_value(value_key, defaults[attr])
                self.config_manager.set_value(default_key, False)

        # Incidence-angle albedo model (Hayne=False / Hapke=True)
        ica = defaults.get("incidence_angle_albedo", False)
        self._set_incidence_model(ica)
        self.config_manager.set_value("thermophysical.incidence_angle_albedo", ica)

        self._update_ti_display(defaults)
        # Re-apply engine-support gating: a preset may have unchecked rows
        # that a built-in body cannot actually customize.
        self._update_builtin_support()
        self.value_changed.emit()

    def on_body_changed(self, body_name):
        """Update displayed defaults when body selection changes"""
        self.update_to_body_defaults(body_name)
        self._update_builtin_support()

    def _update_builtin_support(self):
        """Enable/disable rows according to what the ENGINE can honor.

        Built-in body: only the BUILTIN_OVERRIDABLE properties reach the engine
        (via *_override keys written at save time); the BUILTIN_UNSUPPORTED rows
        are forced back to 'Use Default' and disabled so no input dies silently.
        Both tuples are the single source for this rule and for the strings it
        shows, so neither can drift from the other.
        Custom/HORIZONS body: everything is enabled (values are written to
        generic_body.surface, which the engine reads)."""
        generic = getattr(self.config_manager, "generic_body_active",
                          lambda: False)()
        unsupported_tip = (
            "Not customizable for built-in bodies: the engine has no "
            "override key for this property (config_parser.cpp reads "
            + "/".join(self.BUILTIN_OVERRIDABLE) + " overrides only). Use a "
            "Custom or HORIZONS body to set it via generic_body.surface.")
        for attr in self.BUILTIN_UNSUPPORTED:
            vkey = f"thermophysical.{attr}"
            dkey = f"thermophysical.use_default_{attr}"
            cb = self.widgets.get(dkey)
            vw = self.widgets.get(vkey)
            if generic:
                if cb is not None:
                    cb.setEnabled(True)
                    cb.setToolTip("Use body-specific default value")
                if vw is not None and cb is not None:
                    vw.setEnabled(not cb.isChecked())
                if vw is not None:
                    vw.setToolTip("")
            else:
                if cb is not None and not cb.isChecked():
                    # revert a custom value the engine cannot receive
                    cb.setChecked(True)
                if cb is not None:
                    cb.setEnabled(False)
                    cb.setToolTip(unsupported_tip)
                if vw is not None:
                    vw.setEnabled(False)
                    vw.setToolTip(unsupported_tip)
        if hasattr(self, "info_label"):
            if generic:
                self.info_label.setText(
                    "Properties shown are defaults for the selected body. "
                    "Uncheck 'Use Default' to customize (values are written "
                    "to generic_body.surface).")
            else:
                self.info_label.setText(
                    "Properties shown are defaults for the selected body. "
                    "For built-in bodies the engine accepts custom values for "
                    + ", ".join(self.BUILTIN_OVERRIDABLE)
                    + " only (saved as *_override keys); the grayed-out "
                    "properties require a Custom or HORIZONS body.")

    def _on_incidence_model_changed(self, _idx):
        """Selector changed -> write the Hayne/Hapke boolean to config."""
        use_hapke = bool(self.incidence_combo.currentData())
        self.config_manager.set_value(
            "thermophysical.incidence_angle_albedo", use_hapke)
        self.value_changed.emit()

    def _set_incidence_model(self, use_hapke):
        """Reflect a boolean incidence_angle_albedo into the selector without
        emitting a change: Hayne (False) -> index 0, Hapke (True) -> index 1."""
        idx = 1 if use_hapke else 0
        self.incidence_combo.blockSignals(True)
        self.incidence_combo.setCurrentIndex(idx)
        self.incidence_combo.blockSignals(False)

    def update_to_body_defaults(self, body_name=None):
        """Update all fields to show body defaults"""
        if body_name is None:
            config = self.config_manager.get_config()
            body_name = config.get("simulation", {}).get("body", "Moon")

        # str(): simulation.body can be any scalar after yaml.safe_load, and
        # .startswith on a float raises out of load_from_config -- an abort.
        body_name = str(body_name or "")
        # Strip [H] prefix for HORIZONS bodies
        clean_name = body_name[4:] if body_name.startswith("[H] ") else body_name

        # For HORIZONS bodies, try to load defaults from cached config
        if body_name.startswith("[H] "):
            horizons_defaults = self._load_horizons_defaults(clean_name)
            if horizons_defaults:
                defaults = horizons_defaults
            else:
                defaults = self.BODY_DEFAULTS["Asteroid"]
        elif clean_name not in self.BODY_DEFAULTS:
            # Not a built-in body — try HORIZONS cache (handles reloaded
            # configs where the [H] prefix was stripped during save)
            horizons_defaults = self._load_horizons_defaults(clean_name)
            if horizons_defaults:
                defaults = horizons_defaults
            else:
                defaults = self.BODY_DEFAULTS["Asteroid"]
        else:
            defaults = self.BODY_DEFAULTS[clean_name]

        properties = [
            ("thermophysical.albedo", "thermophysical.use_default_albedo", "albedo"),
            ("thermophysical.emissivity", "thermophysical.use_default_emissivity", "emissivity"),
            ("thermophysical.ks", "thermophysical.use_default_ks", "ks"),
            ("thermophysical.kd", "thermophysical.use_default_kd", "kd"),
            ("thermophysical.H", "thermophysical.use_default_H", "H"),
            ("thermophysical.rhos", "thermophysical.use_default_rhos", "rhos"),
            ("thermophysical.rhod", "thermophysical.use_default_rhod", "rhod"),
            ("thermophysical.cp0", "thermophysical.use_default_cp0", "cp0"),
            ("thermophysical.Qb", "thermophysical.use_default_Qb", "Qb"),
            ("thermophysical.chi", "thermophysical.use_default_chi", "chi"),
        ]

        for value_key, default_key, attr in properties:
            if value_key in self.widgets and default_key in self.widgets:
                use_default = self.widgets[default_key].isChecked()
                if use_default:
                    self.widgets[value_key].blockSignals(True)
                    self.widgets[value_key].setValue(defaults.get(attr, 0.0))
                    self.widgets[value_key].blockSignals(False)

        # Update incidence-angle albedo model selector + config (Hayne / Hapke)
        ica = defaults.get("incidence_angle_albedo", False)
        self._set_incidence_model(ica)
        self.config_manager.set_value("thermophysical.incidence_angle_albedo", ica)

        self._update_ti_display(defaults)

    def load_from_config(self):
        """Load values from configuration"""
        config = self.config_manager.get_config()
        thermo = config.get("thermophysical", {})

        # Snapshot the saved incidence-albedo model choice up front. `thermo`
        # is a live reference into the config dict, and BOTH the checkbox loop
        # below (setChecked -> stateChanged -> on_default_toggled ->
        # update_to_body_defaults) and the explicit seed further down call
        # set_value on this key, so it must be read before any of that runs.
        has_saved_ica = "incidence_angle_albedo" in thermo
        saved_ica = bool(thermo.get("incidence_angle_albedo", False))

        properties = [
            ("thermophysical.albedo", "thermophysical.use_default_albedo", "albedo", "use_default_albedo"),
            ("thermophysical.emissivity", "thermophysical.use_default_emissivity", "emissivity", "use_default_emissivity"),
            ("thermophysical.ks", "thermophysical.use_default_ks", "ks", "use_default_ks"),
            ("thermophysical.kd", "thermophysical.use_default_kd", "kd", "use_default_kd"),
            ("thermophysical.H", "thermophysical.use_default_H", "H", "use_default_H"),
            ("thermophysical.rhos", "thermophysical.use_default_rhos", "rhos", "use_default_rhos"),
            ("thermophysical.rhod", "thermophysical.use_default_rhod", "rhod", "use_default_rhod"),
            ("thermophysical.cp0", "thermophysical.use_default_cp0", "cp0", "use_default_cp0"),
            ("thermophysical.Qb", "thermophysical.use_default_Qb", "Qb", "use_default_Qb"),
            ("thermophysical.chi", "thermophysical.use_default_chi", "chi", "use_default_chi"),
        ]

        for value_key, default_key, value_attr, default_attr in properties:
            if value_key in self.widgets:
                # set_numeric, not setValue: a spin box clamps SILENTLY and
                # still emits valueChanged, so an out-of-range config value was
                # written back clamped. H is the one that matters now -- moving
                # it into BUILTIN_OVERRIDABLE made it a live engine override,
                # and its widget caps at 1.0 m.
                self.set_numeric(value_key, thermo.get(value_attr, 0.0))
            if default_key in self.widgets:
                # set_bool, not setChecked: a quoted `use_default_H: 'true'` is
                # a str after yaml.safe_load, the engine's as<bool> accepts it,
                # and setChecked(str) raises out of load_from_config.
                self.set_bool(default_key, thermo.get(default_attr, True))
                use_default = self.widgets[default_key].isChecked()
                self.widgets[value_key].setEnabled(not use_default)

        # Seed body defaults first (this re-applies the body's default
        # incidence-albedo model), THEN let an explicit saved choice win.
        # update_to_body_defaults() unconditionally writes the body's default
        # incidence_angle_albedo (there is no per-property "Use Default"
        # checkbox gating it, unlike the numeric rows above), so a saved
        # per-project Hayne/Hapke selection would be clobbered on reload
        # unless we re-apply it afterward. See PTM-33.
        self.update_to_body_defaults()

        if has_saved_ica:
            self._set_incidence_model(saved_ica)
            self.config_manager.set_value(
                "thermophysical.incidence_angle_albedo", saved_ica)

        # Conductivity model is a plain string key, not a body default: it is
        # not touched by update_to_body_defaults and needs no re-apply dance.
        self.set_enum("thermophysical.conductivity_model",
                      thermo.get("conductivity_model", "hayne2017"),
                      CONDUCTIVITY_MODEL_ALIASES)

        self._update_builtin_support()

    def _load_horizons_defaults(self, body_name):
        """Try to load thermophysical defaults from a cached HORIZONS body config."""
        try:
            import sys
            from pathlib import Path
            _src_python = str(Path(__file__).resolve().parent.parent.parent / "src" / "python")
            if _src_python not in sys.path:
                sys.path.insert(0, _src_python)
            from data_manager import DataManager
            cache_dir = str(Path(__file__).resolve().parent.parent.parent / "data" / "horizons_cache")
            dm = DataManager(cache_dir)
            config_path = dm.get_config_path(body_name)
            if not config_path:
                return None
            import yaml
            with open(config_path) as f:
                body_config = yaml.safe_load(f)
            surface = body_config.get("generic_body", {}).get("surface", {})
            if not surface:
                return None
            # Get rAU from orbit section if available
            orbit = body_config.get("generic_body", {}).get("orbit", {})
            rAU = ConfigManager._as_number(
                orbit.get("semi_major_axis_au", 2.5), 2.5)
            if rAU <= 0:
                rAU = 2.5
            return {
                "albedo": surface.get("albedo", 0.1),
                "emissivity": surface.get("emissivity", 0.95),
                "ks": surface.get("ks", 0.001),
                "kd": surface.get("kd", 0.01),
                "rhos": surface.get("rhos", 1500.0),
                "rhod": surface.get("rhod", 2000.0),
                "H": surface.get("H", 0.05),
                "cp0": surface.get("cp0", 800.0),
                "Qb": surface.get("Qb", 0.0),
                "incidence_angle_albedo": False,
                "chi": surface.get("chi", 2.7),
                "rAU": rAU,
                "cpCoeff": [],  # Constant cp for HORIZONS bodies
            }
        except Exception as e:
            logger.debug("Could not load HORIZONS defaults for %s: %s", body_name, e)
            return None

    # ── Thermal Inertia sync methods ─────────────────────────────────

    def _get_current_body_defaults(self):
        """Return the BODY_DEFAULTS dict for the currently selected body."""
        config = self.config_manager.get_config()
        body_name = str(config.get("simulation", {}).get("body", "Moon") or "")
        clean_name = body_name[4:] if body_name.startswith("[H] ") else body_name
        if body_name.startswith("[H] "):
            hd = self._load_horizons_defaults(clean_name)
            if hd:
                return hd
            return self.BODY_DEFAULTS.get("Asteroid", {})
        if clean_name not in self.BODY_DEFAULTS:
            # Not a built-in body — try HORIZONS cache
            hd = self._load_horizons_defaults(clean_name)
            if hd:
                return hd
            return self.BODY_DEFAULTS.get("Asteroid", {})
        return self.BODY_DEFAULTS[clean_name]

    def _compute_ti_params(self, defaults=None):
        """Compute T_eq, cp(T_eq), and TI from current widget values.

        Returns (T_eq, cp, TI) or (None, None, None) on error.
        """
        if defaults is None:
            defaults = self._get_current_body_defaults()

        # Read current widget values
        ks = self.widgets.get("thermophysical.ks")
        rhos = self.widgets.get("thermophysical.rhos")
        cp0_w = self.widgets.get("thermophysical.cp0")
        albedo_w = self.widgets.get("thermophysical.albedo")
        emiss_w = self.widgets.get("thermophysical.emissivity")

        if not all([ks, rhos, cp0_w, albedo_w, emiss_w]):
            return None, None, None

        ks_val = ks.value()
        rhos_val = rhos.value()
        cp0_val = cp0_w.value()
        albedo_val = albedo_w.value()
        emiss_val = emiss_w.value()

        # Get rAU and cpCoeff from body defaults or config
        rAU = defaults.get("rAU", 1.0)
        cpCoeff = defaults.get("cpCoeff", [])

        # Try to get rAU from generic_body config (overrides body defaults)
        config_rAU = ConfigManager._as_number(
            self.config_manager.get_value(
                "generic_body.orbit.semi_major_axis_au", -1.0), -1.0)
        if config_rAU > 0:
            rAU = config_rAU

        T_eq = compute_equilibrium_temperature(albedo_val, emiss_val, rAU)
        cp = compute_cp_at_temperature(T_eq, cp0_val, cpCoeff)
        ti = compute_thermal_inertia(ks_val, rhos_val, cp)

        return T_eq, cp, ti

    def _update_ti_display(self, defaults=None):
        """Recompute and display TI, cp(T_eq), and T_eq (blocking TI spin signals)."""
        if not hasattr(self, 'ti_spin'):
            return

        T_eq, cp, ti = self._compute_ti_params(defaults)
        if T_eq is None:
            return

        self.ti_spin.blockSignals(True)
        self.ti_spin.setValue(ti)
        self.ti_spin.blockSignals(False)

        self.ti_cp_label.setText(f"{cp:.1f} J/kg/K")
        self.ti_teq_label.setText(f"{T_eq:.1f} K")

    def _on_ti_changed(self, new_ti):
        """Handle TI spinbox change: set constant thermal inertia mode.

        When the user explicitly sets TI, the simulation uses constant
        (temperature-independent) thermophysical properties:
          - chi = 0 (no radiative conductivity, k = ks = const)
          - cp = TI^2 / (ks * rhos) = const (no T-dependent heat capacity)
          - ks and rhos are scaled to match the target TI

        Generic-body awareness: when a generic body is active (asteroid /
        Keplerian run), the C++ solver builds `planet` from
        ``generic_body.thermo`` (see ptm_thermal_sim.cpp:7198) and
        recomputes cp from those ks/rhos at the constant_thermal_inertia
        override (line 7231). To keep the GUI's displayed cp consistent
        with the cp the simulation will actually use, we read ks/rhos from
        ``generic_body.surface`` in that case and write the derived cp
        there too (mirroring to thermophysical.cp0 for display).
        """
        if getattr(self, '_syncing_ti', False):
            return
        self._syncing_ti = True

        try:
            # Detect "generic body active" via either of the canonical
            # config keys the body-definition tab populates. Robust to a
            # YAML that defines only one of name/config_file.
            gb_active = (
                bool(self.config_manager.get_value(
                    "generic_body.name", "")) or
                bool(self.config_manager.get_value(
                    "generic_body.config_file", ""))
            )

            if gb_active:
                # Prefer generic_body.surface values; fall back to top-level
                # widgets only if the generic_body keys aren't populated.
                ks_val = self.config_manager.get_value(
                    "generic_body.surface.ks", -1.0)
                if ks_val is None or ks_val <= 0:
                    ks_val = self.widgets["thermophysical.ks"].value()
                rhos_val = self.config_manager.get_value(
                    "generic_body.surface.rhos", -1.0)
                if rhos_val is None or rhos_val <= 0:
                    rhos_val = self.widgets["thermophysical.rhos"].value()
                source_label = "generic_body.surface"
            else:
                ks_val = self.widgets["thermophysical.ks"].value()
                rhos_val = self.widgets["thermophysical.rhos"].value()
                source_label = "thermophysical"

            if ks_val <= 0:
                ks_val = 0.001
            if rhos_val <= 0:
                rhos_val = 1000.0

            # Compute constant cp from TI = sqrt(ks * rhos * cp)
            # => cp = TI^2 / (ks * rhos)
            cp_const = ((new_ti * new_ti) / (ks_val * rhos_val)
                        if (ks_val * rhos_val) > 0 else 0.0)

            # Set chi = 0 (constant conductivity: k = ks, no radiative term)
            self._set_property(
                "thermophysical.chi", "thermophysical.use_default_chi", 0.0)
            # Set top-level cp0 (mirror, drives the GUI display)
            self._set_property(
                "thermophysical.cp0", "thermophysical.use_default_cp0",
                cp_const)

            if gb_active:
                # Mirror chi=0 and the derived cp into generic_body.surface
                # so the asteroid-path values the C++ actually consumes
                # match what the GUI shows. These keys don't have
                # use_default_* siblings, so set them directly.
                self.config_manager.set_value(
                    "generic_body.surface.chi", 0.0)
                self.config_manager.set_value(
                    "generic_body.surface.cp0", cp_const)

            # Store the constant TI value in config for the C++ solver
            self.config_manager.set_value(
                "thermophysical.constant_thermal_inertia", new_ti)

            # Update display labels
            self.ti_cp_label.setText(
                f"{cp_const:.1f} J/kg/K (const, from {source_label} ks·ρs)")
            T_eq, _, _ = self._compute_ti_params()
            if T_eq is not None:
                self.ti_teq_label.setText(f"{T_eq:.1f} K")

            self.value_changed.emit()
        finally:
            self._syncing_ti = False

    def _set_property(self, value_key, default_key, value):
        """Helper: set a thermophysical property, uncheck 'Use Default'."""
        # Uncheck "Use Default"
        if default_key in self.widgets:
            self.widgets[default_key].blockSignals(True)
            self.widgets[default_key].setChecked(False)
            self.widgets[default_key].blockSignals(False)
            self.config_manager.set_value(default_key, False)

        # Set value
        if value_key in self.widgets:
            self.widgets[value_key].setEnabled(True)
            self.widgets[value_key].blockSignals(True)
            self.widgets[value_key].setValue(value)
            self.widgets[value_key].blockSignals(False)
            self.config_manager.set_value(value_key, value)

    def on_widget_changed(self, key, value):
        """Override to update TI display when relevant properties change.

        Includes generic_body.surface.ks/rhos so that asteroid-mode runs
        re-derive cp from the same values the C++ solver will use when
        constant_thermal_inertia is active.
        """
        super().on_widget_changed(key, value)
        if getattr(self, '_syncing_ti', False):
            return
        if key in ("thermophysical.ks", "thermophysical.kd",
                    "thermophysical.rhos", "thermophysical.rhod",
                    "thermophysical.albedo", "thermophysical.emissivity",
                    "thermophysical.cp0",
                    "generic_body.surface.ks",
                    "generic_body.surface.rhos"):
            self._update_ti_display()


class BodyDefinitionTab(BaseConfigTab):
    """Body definition parameters: shape, orbit, spin, ephemeris source."""

    # Built-in bodies that use SPICE (fields should be informational only)
    BUILTIN_BODIES = {"Europa", "Moon", "Mercury", "Enceladus", "Titan", "Mars", "Ceres"}

    def init_ui(self):
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        # Context-sensitive info banner
        self._info_banner = QLabel(
            "Select a body on the Simulation tab, or choose 'Custom Body...' to define parameters."
        )
        self._info_banner.setStyleSheet("color: #64b5f6; font-size: 10px; padding: 5px;")
        self._info_banner.setWordWrap(True)
        layout.addWidget(self._info_banner)

        # ---- Group 1: Body Shape ----
        shape_group = QGroupBox("Body Shape")
        shape_layout = QGridLayout()
        shape_layout.setColumnStretch(1, 1)

        self.add_combo_field(shape_layout, 0, "Shape Type:", "generic_body.shape.type",
                             ["spherical", "ellipsoidal", "custom_mesh"],
                             "Body shape model", info_key="generic_body.shape_type")

        # Spherical: radius
        self.add_double_field(shape_layout, 1, "Radius (km):", "generic_body.shape.radius_km",
                              0.001, 100000.0, 3, "Mean body radius",
                              info_key="generic_body.shape_radius")

        # Ellipsoidal: three semi-axes
        self.add_double_field(shape_layout, 2, "Semi-axis A (km):", "generic_body.shape.semi_axis_a_km",
                              0.001, 100000.0, 3, "Semi-axis along X")
        self.add_double_field(shape_layout, 3, "Semi-axis B (km):", "generic_body.shape.semi_axis_b_km",
                              0.001, 100000.0, 3, "Semi-axis along Y")
        self.add_double_field(shape_layout, 4, "Semi-axis C (km):", "generic_body.shape.semi_axis_c_km",
                              0.001, 100000.0, 3, "Semi-axis along Z")

        # Custom mesh: file + scale
        self.add_file_field(shape_layout, 5, "Mesh File:", "generic_body.shape.mesh_file",
                            "Mesh Files (*.obj *.ply);;All Files (*)",
                            "Path to custom mesh (OBJ or PLY)")
        self.add_double_field(shape_layout, 6, "Mesh Scale:", "generic_body.shape.mesh_scale",
                              0.001, 1000.0, 3, "Scale factor for mesh coordinates")

        shape_group.setLayout(shape_layout)
        layout.addWidget(shape_group)

        # Store row references for visibility toggling
        self._shape_rows = {
            "radius": 1,
            "axes": [2, 3, 4],
            "mesh": [5, 6],
        }

        # Connect shape type combo to visibility toggler
        self.widgets["generic_body.shape.type"].currentTextChanged.connect(
            self._on_shape_type_changed
        )

        # Propagate generic_body.shape.mesh_file → mesh.import_file when the
        # shape is custom_mesh. The C++ thermal solver loads the mesh from
        # sim_config.mesh.import_file (ptm_thermal_sim.cpp:6592); the
        # generic_body.shape.custom_mesh_file field is set but never read,
        # so without this propagation a "custom_mesh" shape selection in
        # the Body Definition tab would silently fall back to the
        # icosphere from `mesh.resolution`.
        self.widgets["generic_body.shape.mesh_file"].textChanged.connect(
            self._propagate_mesh_to_top_level
        )
        self.widgets["generic_body.shape.mesh_scale"].valueChanged.connect(
            self._propagate_mesh_to_top_level
        )

        # ---- Group 2: Orbital Elements ----
        orbit_group = QGroupBox("Orbital Elements")
        orbit_layout = QGridLayout()
        orbit_layout.setColumnStretch(1, 1)

        self.add_double_field(orbit_layout, 0, "Semi-major Axis (AU):",
                              "generic_body.orbit.semi_major_axis_au",
                              0.01, 1000.0, 6, "Orbital semi-major axis",
                              info_key="generic_body.orbit.sma")
        self.add_double_field(orbit_layout, 1, "Eccentricity:",
                              "generic_body.orbit.eccentricity",
                              0.0, 0.999, 8, "Orbital eccentricity",
                              info_key="generic_body.orbit.ecc")
        self.add_double_field(orbit_layout, 2, "Inclination (deg):",
                              "generic_body.orbit.inclination_deg",
                              0.0, 180.0, 4, "Orbital inclination",
                              info_key="generic_body.orbit.inc")
        self.add_double_field(orbit_layout, 3, "Long. Asc. Node (deg):",
                              "generic_body.orbit.lon_ascending_node_deg",
                              0.0, 360.0, 4, "Longitude of ascending node",
                              info_key="generic_body.orbit.lan")
        self.add_double_field(orbit_layout, 4, "Arg. Perihelion (deg):",
                              "generic_body.orbit.arg_perihelion_deg",
                              0.0, 360.0, 4, "Argument of perihelion",
                              info_key="generic_body.orbit.aop")
        self.add_double_field(orbit_layout, 5, "Mean Anomaly (deg):",
                              "generic_body.orbit.mean_anomaly_deg",
                              0.0, 360.0, 4, "Mean anomaly at epoch",
                              info_key="generic_body.orbit.ma")
        self.add_string_field(orbit_layout, 6, "Epoch:", "generic_body.orbit.epoch",
                              "Reference epoch (ISO 8601, e.g. 2000-01-01T12:00:00)",
                              info_key="generic_body.orbit.epoch")

        orbit_group.setLayout(orbit_layout)
        layout.addWidget(orbit_group)

        # ---- Group 3: Spin State ----
        spin_group = QGroupBox("Spin State")
        spin_layout = QGridLayout()
        spin_layout.setColumnStretch(1, 1)

        self.add_double_field(spin_layout, 0, "Spin Period (hours):",
                              "generic_body.spin.period_hours",
                              0.001, 100000.0, 4, "Sidereal rotation period",
                              info_key="generic_body.spin.period")

        # Row 1: Obliquity (derived, not stored in config)
        self._syncing_obliquity = False
        obl_label = QLabel("Obliquity (deg):")
        self._apply_tooltip(obl_label, "generic_body.spin.obliquity")
        spin_layout.addWidget(obl_label, 1, 0)

        self.obliquity_spin = QDoubleSpinBox()
        self.obliquity_spin.setRange(0.0, 180.0)
        self.obliquity_spin.setDecimals(2)
        self.obliquity_spin.setSingleStep(1.0)
        self.obliquity_spin.setToolTip("Angle between spin axis and orbit normal")
        self.obliquity_spin.valueChanged.connect(self._on_obliquity_changed)
        spin_layout.addWidget(self.obliquity_spin, 1, 1)

        self.add_double_field(spin_layout, 2, "Pole RA (deg, J2000):",
                              "generic_body.spin.pole_ra_deg",
                              0.0, 360.0, 3, "Right ascension of spin pole",
                              info_key="generic_body.spin.pole_ra")
        self.add_double_field(spin_layout, 3, "Pole Dec (deg, J2000):",
                              "generic_body.spin.pole_dec_deg",
                              -90.0, 90.0, 3, "Declination of spin pole",
                              info_key="generic_body.spin.pole_dec")
        self.add_double_field(spin_layout, 4, "Prime Meridian (deg):",
                              "generic_body.spin.prime_meridian_deg",
                              0.0, 360.0, 3, "Prime meridian angle at epoch",
                              info_key="generic_body.spin.pm")

        spin_group.setLayout(spin_layout)
        layout.addWidget(spin_group)

        # ---- Group 4: Ephemeris Source ----
        eph_group = QGroupBox("Ephemeris Source")
        eph_layout = QGridLayout()
        eph_layout.setColumnStretch(1, 1)

        self.add_combo_field(eph_layout, 0, "Source:", "generic_body.ephemeris_source",
                             ["auto", "keplerian", "spice"],
                             "Ephemeris source for position computation",
                             info_key="generic_body.ephemeris_source")
        self.add_file_field(eph_layout, 1, "SPICE Metakernel:", "generic_body.spice_metakernel",
                            "Meta-kernel Files (*.tm);;All Files (*)",
                            "Path to SPICE metakernel file",
                            info_key="generic_body.spice_metakernel")
        self.add_string_field(eph_layout, 2, "NAIF Body ID:", "generic_body.naif_id",
                              "NAIF/SPICE body ID (e.g. 2000269)",
                              info_key="generic_body.naif_id")

        eph_group.setLayout(eph_layout)
        layout.addWidget(eph_group)

        # Store SPICE-related row indices for visibility toggling
        self._spice_rows = [1, 2]
        self.widgets["generic_body.ephemeris_source"].currentTextChanged.connect(
            self._on_ephemeris_source_changed
        )

        layout.addStretch()
        self.create_scroll_area(content)

        # Apply initial visibility
        self._on_shape_type_changed("spherical")
        self._on_ephemeris_source_changed("auto")

    def _on_shape_type_changed(self, shape_type):
        """Toggle field visibility based on shape type selection."""
        is_sphere = (shape_type == "spherical")
        is_ellipsoid = (shape_type == "ellipsoidal")
        is_mesh = (shape_type == "custom_mesh")

        # Radius: visible for spherical and custom_mesh (reference)
        self._set_row_visible("generic_body.shape.radius_km", is_sphere or is_mesh)

        # Semi-axes: visible for ellipsoidal
        for key in ["generic_body.shape.semi_axis_a_km",
                     "generic_body.shape.semi_axis_b_km",
                     "generic_body.shape.semi_axis_c_km"]:
            self._set_row_visible(key, is_ellipsoid)

        # Mesh file + scale: visible for custom_mesh
        self._set_row_visible("generic_body.shape.mesh_file", is_mesh)
        self._set_row_visible("generic_body.shape.mesh_scale", is_mesh)

        # Sync to the top-level mesh.import_file (the actual mesh load path).
        self._propagate_mesh_to_top_level()

    def _propagate_mesh_to_top_level(self, *_args):
        """Mirror generic_body.shape.{mesh_file,mesh_scale} to
        mesh.{import_file,scale_factor} when shape.type == 'custom_mesh'.

        This is the bridge between the asteroid-side body shape config
        and the simulation's actual mesh loader. The C++ thermal sim
        consumes sim_config.mesh.import_file
        (ptm_thermal_sim.cpp:6592); generic_body.shape.custom_mesh_file
        is metadata only. Without this mirror, picking 'custom_mesh' in
        the Body Definition tab silently does nothing — the run still
        uses the icosphere from `mesh.resolution`.

        We only push when shape.type == 'custom_mesh' so other shape
        modes don't clobber a user-set top-level mesh.import_file.
        """
        shape_type_widget = self.widgets.get("generic_body.shape.type")
        if shape_type_widget is None:
            return
        if shape_type_widget.currentText() != "custom_mesh":
            return
        mesh_file = self.widgets["generic_body.shape.mesh_file"].text().strip()
        mesh_scale = float(self.widgets["generic_body.shape.mesh_scale"].value())
        # config_manager is shared across tabs; the Mesh tab will pick up
        # the new value the next time it renders or the YAML is saved.
        if mesh_file:
            self.config_manager.set_value("mesh.import_file", mesh_file)
            self.config_manager.set_value("mesh.scale_factor", mesh_scale)

    def _on_ephemeris_source_changed(self, source):
        """Toggle SPICE-related field visibility."""
        show_spice = (source in ("auto", "spice"))
        self._set_row_visible("generic_body.spice_metakernel", show_spice)
        self._set_row_visible("generic_body.naif_id", show_spice)

    def _set_row_visible(self, widget_key, visible):
        """Show or hide all widgets in the grid row containing the given widget."""
        if widget_key not in self.widgets:
            return
        widget = self.widgets[widget_key]
        parent_layout = widget.parent()
        if parent_layout is None:
            return

        # Walk up to find the QGridLayout containing this widget
        grid = None
        w = widget
        while w is not None:
            p = w.parent()
            if p is not None and p.layout() is not None:
                lay = p.layout()
                if isinstance(lay, QGridLayout):
                    idx = lay.indexOf(widget)
                    if idx >= 0:
                        grid = lay
                        break
                    # Widget might be inside a container (e.g. file field)
                    # Try finding the container widget
                    for i in range(lay.count()):
                        item = lay.itemAt(i)
                        if item and item.widget() is not None:
                            container = item.widget()
                            if container.findChild(type(widget)) == widget:
                                idx = i
                                grid = lay
                                break
                    if grid:
                        break
            w = p

        if grid is None:
            # Fallback: just toggle the widget itself
            widget.setVisible(visible)
            return

        # Find the row of this widget in the grid
        for i in range(grid.count()):
            item = grid.itemAt(i)
            if item is None:
                continue
            item_widget = item.widget()
            if item_widget is None:
                continue
            # Check if this item is or contains our target widget
            if item_widget is widget or item_widget.findChild(type(widget)) == widget:
                row, col, rspan, cspan = grid.getItemPosition(i)
                # Hide/show all widgets in this row
                for j in range(grid.count()):
                    item2 = grid.itemAt(j)
                    if item2 and item2.widget():
                        r2, c2, rs2, cs2 = grid.getItemPosition(j)
                        if r2 == row:
                            item2.widget().setVisible(visible)
                return

    # ── Obliquity sync logic ────────────────────────────────────────

    def on_widget_changed(self, key, value):
        """Handle widget value changes, updating obliquity display when relevant."""
        super().on_widget_changed(key, value)
        if self._syncing_obliquity:
            return
        # When pole RA/Dec or orbital inclination/Ω change, recompute obliquity
        if key in ("generic_body.spin.pole_ra_deg",
                    "generic_body.spin.pole_dec_deg",
                    "generic_body.orbit.inclination_deg",
                    "generic_body.orbit.lon_ascending_node_deg"):
            self._update_obliquity_display()

    def _update_obliquity_display(self):
        """Recompute obliquity from current widget values and update the spinbox."""
        pole_ra = self.widgets["generic_body.spin.pole_ra_deg"].value()
        pole_dec = self.widgets["generic_body.spin.pole_dec_deg"].value()
        inc = self.widgets["generic_body.orbit.inclination_deg"].value()
        lan = self.widgets["generic_body.orbit.lon_ascending_node_deg"].value()

        obl = compute_obliquity_deg(pole_ra, pole_dec, inc, lan)
        if obl is not None:
            self.obliquity_spin.blockSignals(True)
            self.obliquity_spin.setValue(obl)
            self.obliquity_spin.blockSignals(False)

    def _on_obliquity_changed(self, new_obliquity):
        """Handle obliquity spinbox change — compute new pole RA/Dec."""
        if self._syncing_obliquity:
            return
        self._syncing_obliquity = True
        try:
            pole_ra = self.widgets["generic_body.spin.pole_ra_deg"].value()
            pole_dec = self.widgets["generic_body.spin.pole_dec_deg"].value()
            inc = self.widgets["generic_body.orbit.inclination_deg"].value()
            lan = self.widgets["generic_body.orbit.lon_ascending_node_deg"].value()

            new_ra, new_dec = compute_pole_for_obliquity(
                new_obliquity, pole_ra, pole_dec, inc, lan)

            # Update pole RA/Dec widgets (block signals to avoid loop)
            ra_widget = self.widgets["generic_body.spin.pole_ra_deg"]
            dec_widget = self.widgets["generic_body.spin.pole_dec_deg"]
            ra_widget.blockSignals(True)
            dec_widget.blockSignals(True)
            ra_widget.setValue(new_ra)
            dec_widget.setValue(new_dec)
            ra_widget.blockSignals(False)
            dec_widget.blockSignals(False)

            # Push to config
            self.config_manager.set_value("generic_body.spin.pole_ra_deg", new_ra)
            self.config_manager.set_value("generic_body.spin.pole_dec_deg", new_dec)
            self.value_changed.emit()
        finally:
            self._syncing_obliquity = False

    def on_body_changed(self, body_name):
        """Update banner and field state when body selection changes."""
        body_name = str(body_name or "")
        clean_name = body_name[4:] if body_name.startswith("[H] ") else body_name
        is_builtin = clean_name in self.BUILTIN_BODIES
        is_horizons = body_name.startswith("[H] ")
        is_custom = (body_name == "Custom Body..." or body_name == "Custom"
                     or (not is_builtin and not is_horizons))

        if is_builtin:
            self._info_banner.setText(
                f"Built-in body ({clean_name}) uses SPICE ephemeris. "
                "Fields below are for reference only."
            )
            self._info_banner.setStyleSheet(
                "color: #ffa726; font-size: 10px; padding: 5px;"
            )
            self._set_all_fields_enabled(False)
        elif is_horizons:
            self._info_banner.setText(
                f"HORIZONS body ({clean_name}) \u2014 values loaded from cache. "
                "Edit fields below to override."
            )
            self._info_banner.setStyleSheet(
                "color: #64b5f6; font-size: 10px; padding: 5px;"
            )
            self._set_all_fields_enabled(True)
        else:
            self._info_banner.setText(
                "Custom body \u2014 fill in shape, orbit, and spin parameters."
            )
            self._info_banner.setStyleSheet(
                "color: #81c784; font-size: 10px; padding: 5px;"
            )
            self._set_all_fields_enabled(True)

    def _set_all_fields_enabled(self, enabled):
        """Enable or disable all editable widgets."""
        for key, widget in self.widgets.items():
            if isinstance(widget, (QDoubleSpinBox, QSpinBox, QLineEdit, QComboBox)):
                widget.setEnabled(enabled)
        # Obliquity spinbox is not in self.widgets (derived, not config-backed)
        self.obliquity_spin.setEnabled(enabled)

    def load_from_body_config(self, config_path):
        """Load a body_config.yaml and populate all fields."""
        try:
            import yaml as _yaml
            with open(config_path) as f:
                body_config = _yaml.safe_load(f)
        except Exception as e:
            logger.warning("Could not load body config from %s: %s", config_path, e)
            return

        gb = body_config.get("generic_body", {})

        # Ephemeris source
        self._set_widget_value("generic_body.ephemeris_source",
                               gb.get("ephemeris_source", "auto"))
        self._set_widget_value("generic_body.spice_metakernel",
                               gb.get("spice_metakernel", ""))
        self._set_widget_value("generic_body.naif_id",
                               gb.get("naif_id", ""))

        # Shape
        shape = gb.get("shape", {})
        self._set_widget_value("generic_body.shape.type",
                               shape.get("type", "spherical"))
        self._set_widget_value("generic_body.shape.radius_km",
                               shape.get("radius_km", 0.0))
        self._set_widget_value("generic_body.shape.mesh_file",
                               shape.get("mesh_file", ""))
        self._set_widget_value("generic_body.shape.mesh_scale",
                               shape.get("mesh_scale", 1.0))
        # Semi-axes might be in a list or separate keys
        if "semi_axes_km" in shape and isinstance(shape["semi_axes_km"], list):
            axes = shape["semi_axes_km"]
            if len(axes) >= 3:
                self._set_widget_value("generic_body.shape.semi_axis_a_km", axes[0])
                self._set_widget_value("generic_body.shape.semi_axis_b_km", axes[1])
                self._set_widget_value("generic_body.shape.semi_axis_c_km", axes[2])

        # Orbit
        orbit = gb.get("orbit", {})
        self._set_widget_value("generic_body.orbit.semi_major_axis_au",
                               orbit.get("semi_major_axis_au", 0.0))
        self._set_widget_value("generic_body.orbit.eccentricity",
                               orbit.get("eccentricity", 0.0))
        self._set_widget_value("generic_body.orbit.inclination_deg",
                               orbit.get("inclination_deg", 0.0))
        self._set_widget_value("generic_body.orbit.lon_ascending_node_deg",
                               orbit.get("lon_ascending_node_deg", 0.0))
        self._set_widget_value("generic_body.orbit.arg_perihelion_deg",
                               orbit.get("arg_perihelion_deg", 0.0))
        self._set_widget_value("generic_body.orbit.mean_anomaly_deg",
                               orbit.get("mean_anomaly_deg", 0.0))
        self._set_widget_value("generic_body.orbit.epoch",
                               orbit.get("epoch", ""))

        # Spin
        spin = gb.get("spin", {})
        self._set_widget_value("generic_body.spin.period_hours",
                               spin.get("period_hours", 0.0))
        self._set_widget_value("generic_body.spin.pole_ra_deg",
                               spin.get("pole_ra_deg", 0.0))
        self._set_widget_value("generic_body.spin.pole_dec_deg",
                               spin.get("pole_dec_deg", 0.0))
        self._set_widget_value("generic_body.spin.prime_meridian_deg",
                               spin.get("prime_meridian_deg", 0.0))

        # Compute and display obliquity from orbit + spin data
        self._update_obliquity_display()

        # Push all widget values to config_manager
        for key, widget in self.widgets.items():
            if isinstance(widget, QDoubleSpinBox):
                self.config_manager.set_value(key, widget.value())
            elif isinstance(widget, QLineEdit):
                self.config_manager.set_value(key, widget.text())
            elif isinstance(widget, QComboBox):
                self.config_manager.set_value(key, widget.currentText())

    def _set_widget_value(self, key, value):
        """Set a widget's value, blocking signals to avoid cascading updates."""
        if key not in self.widgets:
            return
        widget = self.widgets[key]
        widget.blockSignals(True)
        if isinstance(widget, (QDoubleSpinBox, QSpinBox)):
            # Clamp to widget range. float() is not enough on its own: NaN and
            # the infinities pass it, and int(nan)/int(inf) then raise out of
            # load_from_config, which is a process abort.
            v = ConfigManager._as_number(value, widget.minimum())
            v = max(widget.minimum(), min(widget.maximum(), v))
            widget.setValue(type(widget.value())(v))
        elif isinstance(widget, QLineEdit):
            widget.setText(str(value) if value else "")
        elif isinstance(widget, QComboBox):
            idx = widget.findText(str(value))
            if idx >= 0:
                widget.setCurrentIndex(idx)
        widget.blockSignals(False)

    def load_from_config(self):
        """Load values from the config manager's current state."""
        config = self.config_manager.get_config()
        gb = config.get("generic_body", {})

        # Top-level fields
        self._set_widget_value("generic_body.ephemeris_source",
                               gb.get("ephemeris_source", "auto"))
        self._set_widget_value("generic_body.spice_metakernel",
                               gb.get("spice_metakernel", ""))
        self._set_widget_value("generic_body.naif_id",
                               gb.get("naif_id", ""))

        # Shape
        shape = gb.get("shape", {})
        self._set_widget_value("generic_body.shape.type",
                               shape.get("type", "spherical"))
        self._set_widget_value("generic_body.shape.radius_km",
                               shape.get("radius_km", 0.0))
        self._set_widget_value("generic_body.shape.semi_axis_a_km",
                               shape.get("semi_axis_a_km", 0.0))
        self._set_widget_value("generic_body.shape.semi_axis_b_km",
                               shape.get("semi_axis_b_km", 0.0))
        self._set_widget_value("generic_body.shape.semi_axis_c_km",
                               shape.get("semi_axis_c_km", 0.0))
        self._set_widget_value("generic_body.shape.mesh_file",
                               shape.get("mesh_file", ""))
        self._set_widget_value("generic_body.shape.mesh_scale",
                               shape.get("mesh_scale", 1.0))

        # Orbit
        orbit = gb.get("orbit", {})
        for field in ["semi_major_axis_au", "eccentricity", "inclination_deg",
                       "lon_ascending_node_deg", "arg_perihelion_deg", "mean_anomaly_deg"]:
            # Coerced: a quoted or exponent-spelled number loads as str and
            # the bare `>=` raised out of load_from_config -- an abort.
            val = ConfigManager._as_number(orbit.get(field, -1.0), -1.0)
            self._set_widget_value(f"generic_body.orbit.{field}",
                                   val if val >= 0 else 0.0)
        self._set_widget_value("generic_body.orbit.epoch",
                               orbit.get("epoch", ""))

        # Spin
        spin = gb.get("spin", {})
        for field in ["period_hours", "pole_ra_deg", "pole_dec_deg", "prime_meridian_deg"]:
            val = ConfigManager._as_number(spin.get(field, -1.0), -1.0)
            self._set_widget_value(f"generic_body.spin.{field}",
                                   val if val >= 0 else 0.0)

        # Compute and display obliquity from orbit + spin data
        self._update_obliquity_display()

        # Update banner based on current body
        body = config.get("simulation", {}).get("body", "Moon")
        self.on_body_changed(body)


class IlluminationTab(BaseConfigTab):
    """Illumination parameters tab"""

    def init_ui(self):
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        # Solar Illumination
        illum_group = QGroupBox("Solar Illumination")
        illum_layout = QGridLayout()
        illum_layout.setColumnStretch(1, 1)

        self.add_bool_field(illum_layout, 0, "Include Reflected Sunlight from Primary Body", "illumination.include_primary_reflection",
                            "Include reflected solar radiation from the primary body (e.g., Jupiter for Europa)",
                            info_key="illumination.include_primary_reflection")

        illum_group.setLayout(illum_layout)
        layout.addWidget(illum_group)

        # Albedo
        from PyQt6.QtWidgets import QRadioButton, QButtonGroup

        albedo_group = QGroupBox("Albedo Configuration")
        albedo_layout = QGridLayout()
        albedo_layout.setColumnStretch(1, 1)

        src_label = QLabel("Albedo source:")
        albedo_layout.addWidget(src_label, 0, 0)
        src_container = QWidget()
        src_layout = QHBoxLayout(src_container)
        src_layout.setContentsMargins(0, 0, 0, 0)
        self.albedo_builtin_radio = QRadioButton("Built-in (recommended)")
        self.albedo_builtin_radio.setToolTip(
            "Ship-with-PTM3D Bond-albedo map for the selected body "
            "(data/albedo_maps/<Body>_Albedo_res<N>.bin, resolved via "
            "PTM_DATA_ROOT or the repository data directory). Fetched on "
            "demand if missing.")
        self.albedo_custom_radio = QRadioButton("Custom path…")
        self.albedo_custom_radio.setToolTip(
            "Point at your own albedo map file or directory")
        self.albedo_none_radio = QRadioButton("None (uniform albedo)")
        self.albedo_none_radio.setToolTip(
            "No spatially-varying map: all faces use the constant albedo "
            "from the Properties tab")
        self._albedo_src_group = QButtonGroup(src_container)
        for rb in (self.albedo_builtin_radio, self.albedo_custom_radio,
                   self.albedo_none_radio):
            self._albedo_src_group.addButton(rb)
            src_layout.addWidget(rb)
        src_layout.addStretch()
        albedo_layout.addWidget(src_container, 0, 1)
        self.albedo_builtin_radio.toggled.connect(self._on_albedo_source_toggled)
        self.albedo_none_radio.toggled.connect(self._on_albedo_source_toggled)

        resolved_label = QLabel("Resolves to:")
        albedo_layout.addWidget(resolved_label, 1, 0)
        resolved_container = QWidget()
        resolved_layout = QHBoxLayout(resolved_container)
        resolved_layout.setContentsMargins(0, 0, 0, 0)
        self.albedo_resolved_edit = QLineEdit()
        self.albedo_resolved_edit.setReadOnly(True)
        self.albedo_resolved_edit.setPlaceholderText(
            "(select Built-in to see the resolved dataset path)")
        resolved_layout.addWidget(self.albedo_resolved_edit)
        self.albedo_fetch_btn = QPushButton("Fetch…")
        self.albedo_fetch_btn.setMaximumWidth(80)
        self.albedo_fetch_btn.setVisible(False)
        self.albedo_fetch_btn.clicked.connect(self._on_albedo_fetch)
        resolved_layout.addWidget(self.albedo_fetch_btn)
        albedo_layout.addWidget(resolved_container, 1, 1)

        self.add_file_field(albedo_layout, 2, "Albedo Map:", "illumination.albedo_map",
                            "FITS Files (*.fits *.fit);;All Files (*)",
                            "Path to single albedo map file (Custom mode)",
                            info_key="illumination.albedo_map")
        self.add_directory_field(albedo_layout, 3, "Albedo Maps Directory:", "illumination.albedo_maps_directory",
                                 "Directory containing multiple albedo maps (Custom mode)",
                                 info_key="illumination.albedo_maps_directory")

        albedo_group.setLayout(albedo_layout)
        layout.addWidget(albedo_group)

        self._albedo_map_field_keys = ["illumination.albedo_map", "illumination.albedo_maps_directory"]

        # Advanced illumination (collapsed): batch ray-tracing + precomputed
        # Sun/Primary positions — expert knobs most runs never touch.
        adv_illum = self.advanced_section("Advanced illumination")

        batch_group = QGroupBox("Ray Tracing")
        batch_layout = QGridLayout()
        batch_layout.setColumnStretch(1, 1)
        self.add_bool_field(batch_layout, 0, "Use Batch Processing", "illumination.use_batch_processing",
                            "Use batch processing for ray tracing",
                            info_key="illumination.use_batch_processing")
        batch_group.setLayout(batch_layout)
        adv_illum.add_widget(batch_group)

        pos_group = QGroupBox("Precomputed Positions")
        pos_layout = QGridLayout()
        pos_layout.setColumnStretch(1, 1)
        self.add_file_field(pos_layout, 0, "Positions File:", "illumination.positions_file",
                            "Binary Files (*.bin);;All Files (*)",
                            "Precomputed Sun/Primary positions",
                            info_key="illumination.positions_file")
        pos_group.setLayout(pos_layout)
        adv_illum.add_widget(pos_group)

        layout.addWidget(adv_illum)

        layout.addStretch()
        self.create_scroll_area(content)

    # -- albedo source (built-in / custom / none) ----------------------------

    def _current_body(self):
        body = self.config_manager.get_value("simulation.body", "") or ""
        body = str(body or "")
        return body[4:] if body.startswith("[H] ") else body

    def _project_root(self):
        from gui_core.preflight import find_project_root
        from gui_core.simulation_runner import SimulationRunner
        return find_project_root(SimulationRunner.find_ptm_executable())

    def _current_source(self):
        if self.albedo_builtin_radio.isChecked():
            return "builtin"
        if self.albedo_none_radio.isChecked():
            return "none"
        return "custom"

    def _on_albedo_source_toggled(self, _checked):
        source = self._current_source()
        self.config_manager.set_value("illumination.albedo_source", source)
        self.config_manager.set_value("illumination.use_albedo_map",
                                      source != "none")
        if source in ("builtin", "none"):
            # concrete values are enforced at save time (_config_for_yaml);
            # mirror them in the widgets so the YAML preview matches
            dir_value = "data/albedo_maps" if source == "builtin" else ""
            for key, val in (("illumination.albedo_map", ""),
                             ("illumination.albedo_maps_directory", dir_value)):
                w = self.widgets.get(key)
                if w is not None and w.text() != val:
                    w.setText(val)
        self._apply_albedo_source_enabled()
        self.refresh_builtin_albedo()
        self.value_changed.emit()

    def _apply_albedo_source_enabled(self):
        custom = self.albedo_custom_radio.isChecked()
        for key in self._albedo_map_field_keys:
            widget = self.widgets.get(key)
            if widget is not None:
                widget.setEnabled(custom)

    def refresh_builtin_albedo(self):
        """Update the read-only resolved-path display (body- and
        resolution-aware)."""
        from gui_core import datasets
        body = self._current_body()
        res = int(ConfigManager._as_number(
            self.config_manager.get_value("mesh.resolution", 5), 5))
        root = self._project_root()
        status = datasets.albedo_status(body, res, root)
        supported = status.state != datasets.STATE_UNSUPPORTED

        self.albedo_builtin_radio.setEnabled(supported)
        self.albedo_builtin_radio.setToolTip(
            status.detail if not supported else
            "Ship-with-PTM3D Bond-albedo map for the selected body, "
            "fetched on demand")
        # A body without built-in albedo cannot stay in Built-in mode --
        # fall over to None (uniform albedo) rather than promising a map
        # that does not exist.
        if not supported and self.albedo_builtin_radio.isChecked():
            self.albedo_none_radio.setChecked(True)
            return  # toggled handler re-enters refresh

        if not self.albedo_builtin_radio.isChecked():
            self.albedo_resolved_edit.setText("")
            self.albedo_fetch_btn.setVisible(False)
            self._albedo_status = status
            return
        if status.state == datasets.STATE_OK:
            note = f"  ({status.detail})" if status.detail else ""
            self.albedo_resolved_edit.setText(f"{status.resolved}{note}")
            self.albedo_fetch_btn.setVisible(False)
        else:
            tag = ("NOT ON DISK" if status.state == datasets.STATE_MISSING
                   else "DATALESS (cloud placeholder)")
            self.albedo_resolved_edit.setText(
                f"{tag}: {status.resolved} -- use Fetch")
            self.albedo_fetch_btn.setVisible(bool(status.dataset))
        self._albedo_status = status

    def _on_albedo_fetch(self):
        from gui_panels.fetch_dialog import FetchDialog
        status = getattr(self, "_albedo_status", None)
        if not status or not status.dataset:
            return
        dlg = FetchDialog(status.dataset, self._project_root(), parent=self)
        dlg.exec()
        self.refresh_builtin_albedo()

    def on_body_changed(self, _body_name):
        self.refresh_builtin_albedo()

    def load_from_config(self):
        config = self.config_manager.get_config()
        illum = config["illumination"]

        self.set_bool("illumination.include_primary_reflection", illum["include_primary_reflection"])
        self.set_bool("illumination.use_batch_processing", illum["use_batch_processing"])
        self.set_text("illumination.albedo_map", illum["albedo_map"])
        self.set_text("illumination.albedo_maps_directory", illum["albedo_maps_directory"])
        self.set_text("illumination.positions_file", illum["positions_file"])

        source = illum.get("albedo_source", "builtin")
        for rb in (self.albedo_builtin_radio, self.albedo_custom_radio,
                   self.albedo_none_radio):
            rb.blockSignals(True)
        self.albedo_builtin_radio.setChecked(source == "builtin")
        self.albedo_custom_radio.setChecked(source == "custom")
        self.albedo_none_radio.setChecked(source == "none")
        for rb in (self.albedo_builtin_radio, self.albedo_custom_radio,
                   self.albedo_none_radio):
            rb.blockSignals(False)
        self._apply_albedo_source_enabled()
        self.refresh_builtin_albedo()


class OutputTab(BaseConfigTab):
    """Output configuration tab"""

    def init_ui(self):
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        # Output Destination
        dest_group = QGroupBox("Output Destination")
        dest_layout = QGridLayout()
        dest_layout.setColumnStretch(1, 1)

        self.add_directory_field(dest_layout, 0, "Output Directory:", "output.directory",
                                 "Directory where results are written",
                                 info_key="output.directory")
        self.add_combo_field(dest_layout, 1, "Format:", "output.format",
                             ["binary", "vtk"], "Output format",
                             info_key="output.format")
        self.add_combo_field(dest_layout, 2, "VTK Mode:", "output.vtk_mode",
                             ["thermal", "complete", "both"], "VTK export mode",
                             info_key="output.vtk_mode")
        self.add_bool_field(dest_layout, 3, "Include Primary IR", "output.include_primary_ir",
                            "Include primary IR in VTK export",
                            info_key="output.include_primary_ir")

        dest_group.setLayout(dest_layout)
        layout.addWidget(dest_group)

        # Location Exports
        loc_group = QGroupBox("Location Exports")
        loc_layout = QGridLayout()
        loc_layout.setColumnStretch(1, 1)

        self.add_file_field(loc_layout, 0, "Locations File:", "output.locations_file",
                            "Location Files (*.txt *.csv);;All Files (*)",
                            "Text/CSV with lat,lon locations",
                            info_key="output.locations_file")
        self.add_file_field(loc_layout, 1, "Depths File:", "output.depths_file",
                            "Depth Files (*.txt *.dat);;All Files (*)",
                            "Depths for location output (one per line, meters)",
                            info_key="output.depths_file")
        self.add_string_field(loc_layout, 2, "Series Output:", "output.series_output",
                              "Filename for illumination series CSV")

        loc_group.setLayout(loc_layout)
        layout.addWidget(loc_group)

        layout.addStretch()
        self.create_scroll_area(content)

    def load_from_config(self):
        config = self.config_manager.get_config()
        out = config["output"]

        self.set_text("output.directory", out["directory"])

        # str(): findText refuses a non-string, out of load_from_config.
        idx = self.widgets["output.format"].findText(str(out.get("format", "")))
        if idx >= 0:
            self.widgets["output.format"].setCurrentIndex(idx)

        idx = self.widgets["output.vtk_mode"].findText(
            str(out.get("vtk_mode", "")))
        if idx >= 0:
            self.widgets["output.vtk_mode"].setCurrentIndex(idx)

        self.set_bool("output.include_primary_ir", out.get("include_primary_ir", False))
        self.set_text("output.locations_file", out.get("locations_file", ""))
        self.set_text("output.depths_file", out.get("depths_file", ""))
        self.set_text("output.series_output", out.get("series_output", ""))


class EngineFlagsTab(BaseConfigTab):
    """Advanced: engine environment flags (PTM_*) for ptm_thermal_sim.

    The engine gates a number of physics fixes and diagnostics behind
    environment variables so that old runs stay byte-identical. The GUI
    applies the recommended-ON set by default (gui_core.preflight.
    RECOMMENDED_ENGINE_ENV); this table lets an advanced user override a
    value, add extra flags (e.g. PTM_HMATRIX_THREADS), or clear a value
    (blank = do not set). Stored under config key 'engine_env'; exported by
    build_engine_env() at launch AND read by the engine itself at startup
    (ConfigParser::applyEngineEnv, PTM-42). The two agree by construction --
    the engine leaves an already-exported variable alone -- so a GUI launch
    logs them as "already set (identical)".
    """

    def init_ui(self):
        from PyQt6.QtWidgets import QTableWidget, QTableWidgetItem, QHeaderView
        self._QTableWidgetItem = QTableWidgetItem
        self._loading = False

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        info = QLabel(
            "Environment flags passed to the simulation engine. The rows "
            "below are the recommended physics-fix set (applied by default). "
            "Blank value = flag is not set. Only change these if you know "
            "the engine gate you are targeting."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #64b5f6; font-size: 10px; padding: 5px;")
        layout.addWidget(info)

        # Shown only when the loaded config carries an EMPTY engine_env: block.
        # An empty table would otherwise look like a bug rather than a decision.
        self._declined_note = QLabel(
            "This configuration carries an empty engine_env: block — it "
            "declines every recommended flag, and the launch will also REMOVE "
            "any of them exported by your shell. Use Restore Recommended to "
            "take the default set instead.")
        self._declined_note.setWordWrap(True)
        self._declined_note.setStyleSheet(
            "color: #FFB74D; font-size: 10px; padding: 5px;")
        self._declined_note.setVisible(False)
        layout.addWidget(self._declined_note)

        # Basic tier: WHAT will be applied, read-only. The editable table is
        # Advanced. Before item 2.1 this whole tab was Advanced-only, so a Basic
        # user could not see that the GUI injects five variables into every run
        # -- and one of them changed the physics of every GUI run on 2026-08-25.
        # A disclosure nobody can reach is not a disclosure.
        self._env_summary = QLabel("")
        self._env_summary.setWordWrap(True)
        self._env_summary.setObjectName("engineEnvSummary")
        self._env_summary.setStyleSheet(
            "color: #B0BEC5; font-size: 11px; padding: 6px; "
            "border: 1px solid #444; border-radius: 4px;")
        layout.addWidget(self._env_summary)

        adv_env = self.advanced_section("Edit engine flags", collapsed=True)
        editor = QWidget()
        editor_layout = QVBoxLayout(editor)
        editor_layout.setContentsMargins(0, 0, 0, 0)

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Variable", "Value"])
        self.table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setMinimumHeight(240)
        self.table.cellChanged.connect(self._on_cell_changed)
        editor_layout.addWidget(self.table)

        btn_row = QHBoxLayout()
        add_btn = QPushButton("Add Flag")
        add_btn.clicked.connect(self._add_row)
        btn_row.addWidget(add_btn)
        rm_btn = QPushButton("Remove Selected")
        rm_btn.clicked.connect(self._remove_selected)
        btn_row.addWidget(rm_btn)
        reset_btn = QPushButton("Restore Recommended")
        reset_btn.clicked.connect(self._restore_recommended)
        btn_row.addWidget(reset_btn)
        btn_row.addStretch()
        editor_layout.addLayout(btn_row)

        adv_env.add_widget(editor)
        layout.addWidget(adv_env)

        layout.addStretch()
        self.create_scroll_area(content)

    # -- table <-> config ---------------------------------------------------

    def _set_rows(self, env_dict):
        self._loading = True
        try:
            self.table.setRowCount(0)
            for key in sorted(env_dict):
                r = self.table.rowCount()
                self.table.insertRow(r)
                self.table.setItem(r, 0, self._QTableWidgetItem(str(key)))
                self.table.setItem(r, 1, self._QTableWidgetItem(str(env_dict[key])))
        finally:
            self._loading = False

    def _collect_rows(self):
        env = {}
        for r in range(self.table.rowCount()):
            key_item = self.table.item(r, 0)
            val_item = self.table.item(r, 1)
            key = key_item.text().strip() if key_item else ""
            val = val_item.text().strip() if val_item else ""
            if key:
                env[key] = val
        return env

    def _write_back(self):
        self.config_manager.set_value("engine_env", self._collect_rows())
        self._refresh_declined_note()
        self.value_changed.emit()

    def _on_cell_changed(self, _row, _col):
        if not self._loading:
            self._write_back()

    def _add_row(self):
        r = self.table.rowCount()
        self.table.insertRow(r)
        self.table.setItem(r, 0, self._QTableWidgetItem("PTM_"))
        self.table.setItem(r, 1, self._QTableWidgetItem(""))

    def _remove_selected(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()},
                      reverse=True)
        for r in rows:
            self.table.removeRow(r)
        if rows:
            self._write_back()

    def _restore_recommended(self):
        from gui_core.preflight import RECOMMENDED_ENGINE_ENV
        self._set_rows(dict(RECOMMENDED_ENGINE_ENV))
        self._write_back()
        self._refresh_declined_note()

    def _refresh_summary(self):
        """Say, in the Basic tier, exactly what the launch will export.

        Reads the SAME planner the launch uses (engine_env_plan), so the two
        cannot drift: a summary computed from the raw block would keep claiming
        five variables after the value cells were blanked, and blank-valued
        entries are dropped at launch.
        """
        from gui_core.preflight import (engine_env_plan,
                                        DELIBERATELY_NOT_INJECTED,
                                        ENV_SET_CHANGED_PHYSICS)
        table, source = engine_env_plan(self.config_manager.get_config())
        if table:
            body = ("This run will export %d engine variable(s) (%s):\n    %s"
                    % (len(table), source,
                       "\n    ".join("%s=%s" % kv for kv in sorted(table.items()))))
        else:
            body = ("This run will export NO engine variables, and will REMOVE "
                    "any the shell exported (%s)." % source)
        skipped = ", ".join(sorted(DELIBERATELY_NOT_INJECTED))
        self._env_summary.setText(
            body
            + ("\n\nDeliberately not injected: %s (see the table's tooltip)."
               % skipped if skipped else "")
            + "\n\n" + ENV_SET_CHANGED_PHYSICS)

    def _refresh_declined_note(self):
        """Show the 'declines every flag' note exactly when it is true.

        It used to be set only in load_from_config, so Restore Recommended left
        it on screen -- the button the note tells you to press -- and deleting
        every row wrote an empty block without ever showing it.
        """
        # Key it on the PLAN, not on the raw block. engine_env_plan drops
        # blank-valued entries, so a table whose five value cells were cleared
        # is just as much a decline as an empty one -- and looked populated,
        # with the note hidden and nothing saying the launch would strip those
        # variables out of the inherited environment.
        from gui_core.preflight import engine_env_plan
        table, _source = engine_env_plan(self.config_manager.get_config())
        self._declined_note.setVisible(not table)
        self._refresh_summary()

    def load_from_config(self):
        from gui_core.preflight import RECOMMENDED_ENGINE_ENV
        env = self.config_manager.get_value("engine_env", None)
        # An EMPTY dict is a real state -- "this config declines the flags" --
        # and must be shown as empty. Treating it as "nothing here, show the
        # defaults" displayed all five as if they would be applied (the launch
        # in fact REMOVES them), and the first edit to any cell wrote the whole
        # displayed set back into the config, silently re-injecting exactly what
        # the config had declined.
        if not isinstance(env, dict):
            env = dict(RECOMMENDED_ENGINE_ENV)
        self._set_rows(env)
        self._refresh_declined_note()


class ConfigurationPanel(QWidget):
    """Main configuration panel with tabs and YAML preview"""

    config_changed = pyqtSignal()

    def __init__(self, config_manager):
        super().__init__()
        self.config_manager = config_manager
        self.tabs = {}
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Create splitter for tabs and preview
        splitter = QSplitter(Qt.Orientation.Vertical)

        # Create and add tabs
        self.tabs["simulation"] = SimulationTab(self.config_manager)
        self.tabs["mesh"] = MeshTab(self.config_manager)
        self.tabs["thermal"] = ThermalTab(self.config_manager)
        self.tabs["thermophysical"] = ThermophysicalTab(self.config_manager)
        self.tabs["body_definition"] = BodyDefinitionTab(self.config_manager)
        self.tabs["radiation"] = RadiationTab(self.config_manager)
        self.tabs["illumination"] = IlluminationTab(self.config_manager)
        self.tabs["output"] = OutputTab(self.config_manager)
        self.tabs["engine_flags"] = EngineFlagsTab(self.config_manager)

        # HORIZONS panel (may not be available if imports fail)
        self._horizons_panel = None
        try:
            from gui_panels.horizons_panel import HorizonsSearchPanel
            self._horizons_panel = HorizonsSearchPanel(self.config_manager)
            self._horizons_panel.body_selected.connect(self._on_horizons_body_selected)
        except Exception as e:
            logger.warning("HORIZONS panel not available: %s", e)

        # Navigation rail + stacked content
        self.nav_rail = QListWidget()
        self.nav_rail.setObjectName("NavRail")
        self.nav_rail.setFixedWidth(160)
        self.nav_rail.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self.stacked_widget = QStackedWidget()

        # The nav rail now shows EVERY section (item 2.1 / DP-9). The
        # Basic/Advanced boundary moved from tabs to fields: each tab renders
        # its Basic fields inline and puts the expert groups behind a
        # disclosure. Drawing it by tab meant the Basic tier could not express
        # a crater run at all -- Radiation and Illumination were hidden
        # wholesale, and this model exists mostly to run craters.
        #
        # Still a pure visibility filter: every widget exists in both tiers, so
        # nothing about the saved YAML depends on the toggle.
        tab_order = [
            ("simulation", "Simulation"),
            ("mesh", "Mesh"),
            ("thermophysical", "Properties"),
            ("thermal", "Thermal"),
            ("output", "Output"),
            ("body_definition", "Body"),
            ("illumination", "Illumination"),
            ("radiation", "Radiation"),
            ("engine_flags", "Engine Flags"),
        ]

        # No nav key is Advanced-only any more. `body_definition` still has a
        # RELEVANCE gate -- it describes a custom body, and there is nothing to
        # describe on a built-in one -- which is not a tier and is applied in
        # _apply_tab_visibility.
        self._advanced_nav_keys = set()
        self._nav_keys = []

        for key, label in tab_order:
            self.nav_rail.addItem(label)
            self.stacked_widget.addWidget(self.tabs[key])
            self._nav_keys.append(key)

        if self._horizons_panel:
            self.nav_rail.addItem("HORIZONS")
            self.stacked_widget.addWidget(self._horizons_panel)
            self._nav_keys.append("__horizons__")

        self.nav_rail.currentRowChanged.connect(self.stacked_widget.setCurrentIndex)
        self.nav_rail.setCurrentRow(0)

        # Basic / Advanced disclosure toggle above the nav rail. It now reveals
        # FIELDS, not tabs: every section is always reachable, and each one
        # keeps its expert groups collapsed until this is checked.
        self.advanced_toggle = QCheckBox("Advanced options")
        self.advanced_toggle.setToolTip(
            "Reveal the expert groups on every tab: view-factor and coupling "
            "options, the GPU device controls and guardrails, mesh placement "
            "and occluder, initial condition and run state, the equilibration "
            "cadence, and the engine-flag editor.\n"
            "Every tab is visible either way; this changes what is shown "
            "inside them, never what is saved.")
        self.advanced_toggle.toggled.connect(self._on_advanced_toggled)

        # Container for nav rail + content
        nav_container = QWidget()
        nav_layout = QHBoxLayout(nav_container)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        nav_layout.setSpacing(0)

        nav_column = QWidget()
        nav_col_layout = QVBoxLayout(nav_column)
        nav_col_layout.setContentsMargins(0, 0, 0, 0)
        nav_col_layout.setSpacing(4)
        nav_col_layout.addWidget(self.advanced_toggle)
        nav_col_layout.addWidget(self.nav_rail, stretch=1)

        nav_layout.addWidget(nav_column)
        nav_layout.addWidget(self.stacked_widget, stretch=1)

        # Apply the initial Basic/Advanced visibility (Basic by default).
        self._apply_tab_visibility()

        # Connect tab change signals
        for tab in self.tabs.values():
            tab.value_changed.connect(self.on_value_changed)

        # Connect body selection to thermophysical and body definition tabs
        if "simulation.body" in self.tabs["simulation"].widgets:
            self.tabs["simulation"].widgets["simulation.body"].currentTextChanged.connect(
                self.tabs["thermophysical"].on_body_changed
            )
            self.tabs["simulation"].widgets["simulation.body"].currentTextChanged.connect(
                self.tabs["body_definition"].on_body_changed
            )
            self.tabs["simulation"].widgets["simulation.body"].currentTextChanged.connect(
                self.tabs["thermal"].on_body_changed
            )
            # Built-in dataset displays are body-aware
            self.tabs["simulation"].widgets["simulation.body"].currentTextChanged.connect(
                self.tabs["mesh"].on_body_changed
            )
            self.tabs["simulation"].widgets["simulation.body"].currentTextChanged.connect(
                self.tabs["illumination"].on_body_changed
            )
            self.tabs["simulation"].widgets["simulation.body"].currentTextChanged.connect(
                self.tabs["radiation"].on_body_changed
            )
            # The Body tab appears only for a custom/generic body; a body change
            # (e.g. selecting a HORIZONS body) must re-evaluate nav visibility.
            self.tabs["simulation"].widgets["simulation.body"].currentTextChanged.connect(
                lambda _t: self._apply_tab_visibility()
            )

        # Built-in dataset displays are resolution-aware: switching the mesh
        # resolution live-updates the resolved topo/albedo paths.
        if "mesh.resolution" in self.tabs["mesh"].widgets:
            res_widget = self.tabs["mesh"].widgets["mesh.resolution"]
            res_widget.valueChanged.connect(
                lambda _v: self.tabs["mesh"].refresh_builtin_topo())
            res_widget.valueChanged.connect(
                lambda _v: self.tabs["illumination"].refresh_builtin_albedo())
            # Mesh resolution changes the face count -> refresh the VF estimate
            # AND re-evaluate the beaming facet-scale rule, which is a property
            # of the mesh. Without the second the rule fails open.
            res_widget.valueChanged.connect(
                lambda _v: self.tabs["radiation"].on_mesh_changed())

        # Importing, rescaling or clearing a mesh changes the facet scale.
        #
        # Pick the signal that fires AFTER the config is written, not merely the
        # first one that exists. A QDoubleSpinBox has BOTH textChanged and
        # valueChanged in Qt6 and emits textChanged FIRST, while add_double_field
        # writes the config on valueChanged -- so a `textChanged or valueChanged`
        # lookup re-evaluated the rule against the previous scale factor and
        # nothing ran afterward. Measured: [('textChanged', '2.500'),
        # ('valueChanged', 2.5)].
        for key in ("mesh.import_file", "mesh.scale_factor",
                    "mesh.use_topographic"):
            widget = self.tabs["mesh"].widgets.get(key)
            if widget is None:
                continue
            if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                signal = widget.valueChanged
            elif isinstance(widget, QCheckBox):
                signal = widget.toggled
            else:
                signal = getattr(widget, "textChanged", None)
            if signal is not None:
                signal.connect(lambda _v: self.tabs["radiation"].on_mesh_changed())

        # Populate cached HORIZONS bodies in combo box
        if self._horizons_panel:
            cached = self._horizons_panel.get_cached_body_names()
            if cached:
                self.tabs["simulation"].refresh_horizons_bodies(cached)

        splitter.addWidget(nav_container)

        # YAML Preview
        preview_widget = QWidget()
        preview_layout = QVBoxLayout(preview_widget)
        preview_layout.setContentsMargins(8, 8, 8, 8)

        preview_header = QLabel("YAML Preview — the engine-facing config a run writes")
        preview_header.setFont(QFont("Helvetica Neue", 11, QFont.Weight.Bold))
        # Not what Save writes. Save edits the opened file in place, touching
        # only the keys whose values changed and leaving its comments, layout
        # and unmentioned keys alone (DP-1(b)); this pane is the fully expanded
        # config the engine is handed, which is what lands beside the output
        # directory as config.yaml. Saying "YAML Preview" over an expanded dump
        # while Save writes a minimal edit would be the same class of quiet
        # mismatch this stage exists to remove.
        preview_header.setToolTip(
            "The fully expanded configuration handed to the engine, and "
            "written beside the output directory as config.yaml.\n"
            "Save does NOT write this: it edits your project file in place, "
            "changing only the keys you changed and preserving its comments.")
        preview_layout.addWidget(preview_header)

        self.preview_text = QTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.setFont(QFont("Consolas", 10))
        preview_layout.addWidget(self.preview_text)

        splitter.addWidget(preview_widget)

        # Set splitter sizes (70% tabs, 30% preview)
        splitter.setSizes([400, 200])

        layout.addWidget(splitter)

        # Load initial values
        self.load_from_config()

    def on_value_changed(self):
        """Handle configuration value changes"""
        self.update_preview()
        self.config_changed.emit()

    def _on_advanced_toggled(self, _checked):
        """Basic/Advanced disclosure toggled — re-apply field-level visibility."""
        self._apply_tab_visibility()

    def _apply_tab_visibility(self):
        """Apply the Basic/Advanced tier, at FIELD level (item 2.1 / DP-9).

        Pure visibility: every widget exists in both tiers and still loads and
        saves, so this never changes the configuration or the saved YAML.

        Two independent rules, deliberately not conflated:

        * **tier** -- each tab hides its own Advanced sections while the toggle
          is off. No tab is hidden by tier any more.
        * **relevance** -- the Body tab describes a custom/generic body, and
          there is nothing for it to describe on a built-in one, so it stays
          hidden until a generic body is active. That is true in both tiers.
        """
        if not hasattr(self, "advanced_toggle"):
            return
        advanced = self.advanced_toggle.isChecked()
        generic_active = self.config_manager.generic_body_active()
        for tab in self.tabs.values():
            tab.set_advanced_visible(advanced)
        for row, key in enumerate(self._nav_keys):
            item = self.nav_rail.item(row)
            if item is None:
                continue
            if key == "body_definition":
                hidden = not generic_active
            elif key in self._advanced_nav_keys:
                hidden = not advanced
            else:
                hidden = False
            item.setHidden(hidden)
        # If the current selection just got hidden, jump to the first visible.
        cur = self.nav_rail.currentRow()
        cur_item = self.nav_rail.item(cur) if cur >= 0 else None
        if cur_item is None or cur_item.isHidden():
            for row in range(self.nav_rail.count()):
                it = self.nav_rail.item(row)
                if it is not None and not it.isHidden():
                    self.nav_rail.setCurrentRow(row)
                    break

    def update_preview(self):
        """Update YAML preview"""
        yaml_content = self.config_manager.generate_yaml()
        self.preview_text.setText(yaml_content)

    def switch_to_horizons_tab(self):
        """Switch to the HORIZONS search tab."""
        if self._horizons_panel:
            idx = self.stacked_widget.indexOf(self._horizons_panel)
            if idx >= 0:
                self.stacked_widget.setCurrentIndex(idx)
                self.nav_rail.setCurrentRow(idx)

    def switch_to_body_definition_tab(self):
        """Switch to the Body Definition tab."""
        if "body_definition" in self.tabs:
            idx = self.stacked_widget.indexOf(self.tabs["body_definition"])
            if idx >= 0:
                self.stacked_widget.setCurrentIndex(idx)
                self.nav_rail.setCurrentRow(idx)

    def _on_horizons_body_selected(self, body_name, config_path):
        """Handle body selection from the HORIZONS panel."""
        # Add the body to the simulation tab combo box and select it
        self.tabs["simulation"].add_horizons_body(body_name, config_path)

        # Refresh cached bodies list in combo
        if self._horizons_panel:
            cached = self._horizons_panel.get_cached_body_names()
            self.tabs["simulation"].refresh_horizons_bodies(cached)

        # Populate the Body Definition tab from the cached body_config.yaml
        if "body_definition" in self.tabs:
            self.tabs["body_definition"].load_from_body_config(config_path)

        # --- Auto-adjust settings for heliocentric generic bodies ---
        # Disable primary reflection (no parent body for asteroids/comets)
        self.config_manager.set_value("illumination.include_primary_reflection", False)
        if "illumination.include_primary_reflection" in self.tabs["illumination"].widgets:
            self.tabs["illumination"].widgets[
                "illumination.include_primary_reflection"
            ].setChecked(False)

        # No built-in albedo maps for HORIZONS bodies: switch the albedo
        # source to None (uniform albedo). The radio handler clears the map
        # fields and sets use_albedo_map: false consistently.
        illum_tab = self.tabs["illumination"]
        if hasattr(illum_tab, "albedo_none_radio"):
            illum_tab.albedo_none_radio.setChecked(True)
        else:
            self.config_manager.set_value("illumination.albedo_source", "none")
            self.config_manager.set_value("illumination.albedo_map", "")
            self.config_manager.set_value("illumination.albedo_maps_directory", "")

        # A generic body is now active — refresh nav visibility so the Body
        # tab can appear (when Advanced is on).
        self._apply_tab_visibility()

        # Switch back to simulation tab
        sim_idx = self.stacked_widget.indexOf(self.tabs["simulation"])
        if sim_idx >= 0:
            self.stacked_widget.setCurrentIndex(sim_idx)
            self.nav_rail.setCurrentRow(sim_idx)

        self.on_value_changed()

    def load_from_config(self):
        """Load all tabs from config"""
        for tab in self.tabs.values():
            tab.load_from_config()
        self.update_preview()
        # Re-evaluate Body-tab visibility (a loaded project may carry a
        # generic body).
        self._apply_tab_visibility()
