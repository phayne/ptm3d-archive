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


class BaseConfigTab(QWidget):
    """Base class for configuration tabs"""

    value_changed = pyqtSignal()

    def __init__(self, config_manager):
        super().__init__()
        self.config_manager = config_manager
        self.widgets = {}
        self.init_ui()

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
            "mesh.topo_directory": "data/dem",
            "illumination.albedo_map": "data/albedo_maps",
            "illumination.albedo_maps_directory": "data/albedo_maps",
            "thermal.thermal_inertia_map": "data/thermal_inertia",
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

    def load_from_config(self):
        """Load values from config manager - implemented by subclasses"""
        raise NotImplementedError


class SimulationTab(BaseConfigTab):
    """Simulation parameters tab"""

    # Separator label used to visually separate built-in from HORIZONS bodies
    SEPARATOR = "──────────"
    CUSTOM_BODY = "Custom Body..."
    BROWSE_HORIZONS = "Browse HORIZONS..."

    # Standard built-in bodies
    BUILTIN_BODIES = ["Europa", "Moon", "Mercury", "Enceladus", "Titan", "Mars", "Ceres"]

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
                           1, 10000, "Number of output timesteps",
                           info_key="simulation.timesteps")

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

    # ── Duration / End-Time synchronisation ──────────────────────────

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
                end_dt = start_dt + timedelta(days=days)
                self.end_time_edit.blockSignals(True)
                self.end_time_edit.setText(end_dt.strftime("%Y-%m-%dT%H:%M:%S"))
                self.end_time_edit.blockSignals(False)
            except (ValueError, TypeError):
                self.end_time_edit.blockSignals(True)
                self.end_time_edit.setText("")
                self.end_time_edit.blockSignals(False)
        finally:
            self._syncing = False

    # ── Body selection ───────────────────────────────────────────────

    def _on_body_combo_changed(self, text):
        """Handle body combo box selection."""
        if text == self.SEPARATOR:
            # Revert to previous selection
            combo = self.widgets["simulation.body"]
            config = self.config_manager.get_config()
            prev = config.get("simulation", {}).get("body", "Europa")
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
            prev = config.get("simulation", {}).get("body", "Europa")
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
            self.widgets["simulation.start_time"].setText(sim["start_time"])

            # Update synodic period for the loaded body
            self._current_synodic_days = self._get_synodic_period_days(
                sim.get("body", "Europa")
            )

            # Set duration spinbox in current display units (use saved value, not synodic default)
            days = sim["duration_days"]
            display_val = self._days_to_display(days)
            self.duration_spin.blockSignals(True)
            self.duration_spin.setValue(display_val)
            self.duration_spin.blockSignals(False)

            self.widgets["simulation.timesteps"].setValue(sim["timesteps"])

            # Body combo
            body_name = sim["body"]
            combo = self.widgets["simulation.body"]
            idx = combo.findText(body_name)
            if idx >= 0:
                combo.blockSignals(True)
                combo.setCurrentIndex(idx)
                combo.blockSignals(False)
            else:
                # Body not found directly — check if it's a HORIZONS body
                config_path = config.get("generic_body", {}).get("config_file", "")

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
        self.add_directory_field(global_layout, 2, "Topographic Directory:", "mesh.topo_directory",
                                 "Directory containing pre-computed topographic meshes")

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

        # Gray out global mesh options when a custom mesh is selected
        self.widgets["mesh.import_file"].textChanged.connect(self._on_import_file_changed)

        layout.addStretch()
        self.create_scroll_area(content)

    def _on_import_file_changed(self, text):
        """Disable global mesh options when a custom mesh file is specified."""
        has_custom = bool(text.strip())
        self.widgets["mesh.resolution"].setEnabled(not has_custom)
        self.widgets["mesh.use_topographic"].setEnabled(not has_custom)
        self.widgets["mesh.topo_directory"].setEnabled(not has_custom)

    def load_from_config(self):
        config = self.config_manager.get_config()
        mesh = config["mesh"]

        self.widgets["mesh.resolution"].setValue(mesh["resolution"])
        self.widgets["mesh.use_topographic"].setChecked(mesh.get("use_topographic", False))
        self.widgets["mesh.topo_directory"].setText(mesh.get("topo_directory", ""))
        self.widgets["mesh.import_file"].setText(mesh.get("import_file", ""))
        self.widgets["mesh.scale_factor"].setValue(mesh.get("scale_factor", 1.0))
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

        self.add_bool_field(control_layout, 3, "Compute Statistics", "thermal.compute_statistics",
                            "Calculate min/max/mean temperature statistics",
                            info_key="thermal.compute_statistics")

        control_group.setLayout(control_layout)
        layout.addWidget(control_group)

        # GPU Acceleration
        gpu_group = QGroupBox("GPU Acceleration")
        gpu_layout = QGridLayout()
        gpu_layout.setColumnStretch(1, 1)

        self.add_bool_field(gpu_layout, 0, "Use GPU for Thermal Solver", "thermal.use_gpu",
                            "Use OpenCL GPU acceleration for thermal solver",
                            info_key="thermal.use_gpu")

        gpu_group.setLayout(gpu_layout)
        layout.addWidget(gpu_group)


        # Integration Window
        integ_group = QGroupBox("Integration Window")
        integ_layout = QGridLayout()
        integ_layout.setColumnStretch(1, 1)

        self.add_double_field(integ_layout, 0, "Integration Days:", "thermal.integration_days",
                              -1.0, 10000.0, 2, "Integration window in days (-1 = auto)",
                              info_key="thermal.integration_days")
        self.add_int_field(integ_layout, 1, "Integration Steps:", "thermal.integration_steps",
                           -1, 100000, "Integration steps override (-1 = auto)",
                           info_key="thermal.integration_steps")

        integ_group.setLayout(integ_layout)
        layout.addWidget(integ_group)

        # Property Maps
        maps_group = QGroupBox("Thermal Property Maps")
        maps_layout = QGridLayout()
        maps_layout.setColumnStretch(1, 1)

        self.add_file_field(maps_layout, 0, "Thermal Inertia Map:", "thermal.thermal_inertia_map",
                            "FITS Files (*.fits *.fit);;All Files (*)",
                            "Path to thermal inertia map",
                            info_key="thermal.thermal_inertia_map")

        maps_group.setLayout(maps_layout)
        layout.addWidget(maps_group)

        layout.addStretch()
        self.create_scroll_area(content)

    # ── Equilibration unit conversion ────────────────────────────────

    def _eq_orbits_to_display(self, orbits):
        """Convert equilibration orbits (synodic cycles) to the selected display unit."""
        unit = self.eq_unit_combo.currentText()
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

    # ── Config loading ───────────────────────────────────────────────

    def load_from_config(self):
        config = self.config_manager.get_config()
        thermal = config["thermal"]

        self._syncing = True
        try:
            self.widgets["thermal.run_thermal_mesh"].setChecked(thermal["run_thermal_mesh"])
            self.widgets["thermal.threads"].setValue(thermal["threads"])

            # Update synodic and orbital periods for current body
            self._current_synodic_days = get_synodic_period_days(self.config_manager)
            self._current_orbital_days = get_orbital_period_days(self.config_manager)

            # Set equilibration in current display units
            orbits = thermal.get("equilibration_orbits", 1.0)
            display_val = self._eq_orbits_to_display(orbits)
            self.eq_spin.blockSignals(True)
            self.eq_spin.setValue(display_val)
            self.eq_spin.blockSignals(False)

            self.widgets["thermal.compute_statistics"].setChecked(thermal["compute_statistics"])
            self.widgets["thermal.use_gpu"].setChecked(thermal.get("use_gpu", False))

            self.widgets["thermal.integration_days"].setValue(thermal["integration_days"])
            self.widgets["thermal.integration_steps"].setValue(thermal["integration_steps"])
            self.widgets["thermal.thermal_inertia_map"].setText(thermal["thermal_inertia_map"])
        finally:
            self._syncing = False


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

        # --- Thermal-Radiation Coupling (collapsed) ---
        coupling_section = CollapsibleSection("Thermal-Radiation Coupling", collapsed=True)
        coupling_widget = QWidget()
        coupling_layout = QGridLayout(coupling_widget)
        coupling_layout.setColumnStretch(1, 1)

        self.add_int_field(coupling_layout, 0, "Max Coupling Iterations:", "radiation.max_coupling_iterations",
                           1, 50, "Maximum thermal-radiation coupling iterations",
                           info_key="radiation.max_coupling_iterations")
        self.add_double_field(coupling_layout, 1, "Temperature Tolerance (K):", "radiation.coupling_temp_threshold",
                              0.1, 10.0, 1, "Temperature convergence threshold",
                              info_key="radiation.coupling_temp_threshold")
        self.add_double_field(coupling_layout, 2, "Flux Tolerance (W/m\u00b2):", "radiation.coupling_flux_threshold",
                              0.01, 10.0, 2, "Flux change threshold",
                              info_key="radiation.coupling_flux_threshold")
        self.add_double_field(coupling_layout, 3, "Face Fraction:", "radiation.coupling_face_fraction",
                              0.001, 0.1, 3, "Fraction of faces still changing to stop")

        coupling_section.add_widget(coupling_widget)
        layout.addWidget(coupling_section)

        # --- View Factor Options (collapsed, merges caching + compression) ---
        vf_section = CollapsibleSection("View Factor Options", collapsed=True)
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

        vf_section.add_widget(vf_widget)
        layout.addWidget(vf_section)

        # --- Sparse View Factors (collapsed) ---
        sparse_section = CollapsibleSection("Sparse View Factors (>1M Faces)", collapsed=True)
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

        layout.addStretch()
        self.create_scroll_area(content)

    def load_from_config(self):
        config = self.config_manager.get_config()
        rad = config.get("radiation", {})

        self.widgets["radiation.enabled"].setChecked(rad.get("enabled", False))
        self.widgets["radiation.verbose"].setChecked(rad.get("verbose", False))
        self.widgets["radiation.use_gpu"].setChecked(rad.get("use_gpu", True))
        self.widgets["radiation.min_shadow_fraction"].setValue(rad.get("min_shadow_fraction", 0.3))
        self.widgets["radiation.max_coupling_iterations"].setValue(rad.get("max_coupling_iterations", 10))
        self.widgets["radiation.coupling_temp_threshold"].setValue(rad.get("coupling_temp_threshold", 0.5))
        self.widgets["radiation.coupling_flux_threshold"].setValue(rad.get("coupling_flux_threshold", 0.1))
        self.widgets["radiation.coupling_face_fraction"].setValue(rad.get("coupling_face_fraction", 0.01))
        self.widgets["radiation.view_factor_file"].setText(rad.get("view_factor_file", ""))
        self.widgets["radiation.use_hmatrix"].setChecked(rad.get("use_hmatrix", False))
        self.widgets["radiation.use_aca"].setChecked(rad.get("use_aca", True))
        self.widgets["radiation.aca_tolerance"].setValue(rad.get("aca_tolerance", 0.001))
        self.widgets["radiation.aca_max_rank"].setValue(rad.get("aca_max_rank", 50))
        self.widgets["radiation.hmatrix_threads"].setValue(rad.get("hmatrix_threads", 0))
        # Sparse view factors
        self.widgets["radiation.adaptive_neighbors"].setChecked(rad.get("adaptive_neighbors", True))
        self.widgets["radiation.captured_energy_target"].setValue(rad.get("captured_energy_target", 0.95))
        self.widgets["radiation.max_neighbors"].setValue(rad.get("max_neighbors", 500))
        self.widgets["radiation.min_neighbors"].setValue(rad.get("min_neighbors", 20))


class ThermophysicalTab(BaseConfigTab):
    """Thermophysical properties configuration tab"""

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
            "H": 0.06, "cp0": -3.6125, "incidence_angle_albedo": True,
            "chi": 2.7, "rAU": 1.0, "cpCoeff": [2.7431, 2.3616e-3, -1.2340e-5, 8.9093e-9],
        },
        "Mercury": {
            "albedo": 0.07, "emissivity": 0.93, "Qb": 0.02,
            "ks": 0.002, "kd": 2.0, "rhos": 1800.0, "rhod": 2200.0,
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

        # Info label
        info = QLabel("Properties shown are defaults for the selected body. Uncheck to customize.")
        info.setStyleSheet("color: #64b5f6; font-size: 10px; padding: 5px;")
        layout.addWidget(info)

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
        self.add_bool_field(surface_layout, 2, "Incidence-Angle Dependent Albedo",
                            "thermophysical.incidence_angle_albedo",
                            "Albedo increases at grazing angles (Hayne et al. 2017)",
                            info_key="thermophysical.incidence_angle_albedo")

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
        self.ti_spin.setToolTip("TI = \u221a(ks \u00d7 \u03c1s \u00d7 cp). Editing scales ks, kd, \u03c1s, \u03c1d.")
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
        cp_heat_section = CollapsibleSection("Heat Capacity & Internal Heat", collapsed=True)
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

        # Incidence angle albedo
        if "thermophysical.incidence_angle_albedo" in self.widgets:
            ica = defaults.get("incidence_angle_albedo", False)
            self.widgets["thermophysical.incidence_angle_albedo"].blockSignals(True)
            self.widgets["thermophysical.incidence_angle_albedo"].setChecked(ica)
            self.widgets["thermophysical.incidence_angle_albedo"].blockSignals(False)
            self.config_manager.set_value("thermophysical.incidence_angle_albedo", ica)

        self._update_ti_display(defaults)
        self.value_changed.emit()

    def on_body_changed(self, body_name):
        """Update displayed defaults when body selection changes"""
        self.update_to_body_defaults(body_name)

    def update_to_body_defaults(self, body_name=None):
        """Update all fields to show body defaults"""
        if body_name is None:
            config = self.config_manager.get_config()
            body_name = config.get("simulation", {}).get("body", "Europa")

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

        # Update incidence angle albedo checkbox
        if "thermophysical.incidence_angle_albedo" in self.widgets:
            self.widgets["thermophysical.incidence_angle_albedo"].setChecked(
                defaults.get("incidence_angle_albedo", False)
            )

        self._update_ti_display(defaults)

    def load_from_config(self):
        """Load values from configuration"""
        config = self.config_manager.get_config()
        thermo = config.get("thermophysical", {})

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
                self.widgets[value_key].setValue(thermo.get(value_attr, 0.0))
            if default_key in self.widgets:
                use_default = thermo.get(default_attr, True)
                self.widgets[default_key].setChecked(use_default)
                self.widgets[value_key].setEnabled(not use_default)

        if "thermophysical.incidence_angle_albedo" in self.widgets:
            self.widgets["thermophysical.incidence_angle_albedo"].setChecked(
                thermo.get("incidence_angle_albedo", True)
            )

        self.update_to_body_defaults()

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
            rAU = orbit.get("semi_major_axis_au", 2.5)
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
        body_name = config.get("simulation", {}).get("body", "Europa")
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
        config_rAU = self.config_manager.get_value(
            "generic_body.orbit.semi_major_axis_au", -1.0)
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
        """Handle TI spinbox change: scale ks, kd, rhos, rhod."""
        if getattr(self, '_syncing_ti', False):
            return
        self._syncing_ti = True

        try:
            T_eq, cp, current_ti = self._compute_ti_params()
            if current_ti is None or cp is None:
                return

            if current_ti < 1e-10:
                # Cannot scale from zero — compute absolute values instead.
                # Set rhos to current value, compute ks = TI^2 / (rhos * cp).
                rhos_val = self.widgets["thermophysical.rhos"].value()
                if rhos_val <= 0:
                    rhos_val = 1000.0
                ks_new = (new_ti * new_ti) / (rhos_val * cp) if cp > 0 else 0.0

                for vk, dk in [("thermophysical.ks", "thermophysical.use_default_ks")]:
                    self.widgets[dk].blockSignals(True)
                    self.widgets[dk].setChecked(False)
                    self.widgets[dk].blockSignals(False)
                    self.config_manager.set_value(dk, False)
                    self.widgets[vk].setEnabled(True)
                    self.widgets[vk].blockSignals(True)
                    self.widgets[vk].setValue(ks_new)
                    self.widgets[vk].blockSignals(False)
                    self.config_manager.set_value(vk, ks_new)
            else:
                # Scale factor: TI = sqrt(ks * rhos * cp) → s = (new_ti / current_ti)^2
                # Apply sqrt(s) to each of ks and rhos (and their deep counterparts)
                # Actually, simpler: s = new_ti / current_ti, scale each by s uniformly
                # TI_new = sqrt(ks*s * rhos*s * cp) = s * sqrt(ks*rhos*cp) = s * TI_old ✓
                import math
                s = new_ti / current_ti

                scale_props = [
                    ("thermophysical.ks", "thermophysical.use_default_ks"),
                    ("thermophysical.kd", "thermophysical.use_default_kd"),
                    ("thermophysical.rhos", "thermophysical.use_default_rhos"),
                    ("thermophysical.rhod", "thermophysical.use_default_rhod"),
                ]

                for vk, dk in scale_props:
                    old_val = self.widgets[vk].value()
                    new_val = old_val * s

                    # Uncheck "Use Default"
                    self.widgets[dk].blockSignals(True)
                    self.widgets[dk].setChecked(False)
                    self.widgets[dk].blockSignals(False)
                    self.config_manager.set_value(dk, False)

                    # Set new value
                    self.widgets[vk].setEnabled(True)
                    self.widgets[vk].blockSignals(True)
                    self.widgets[vk].setValue(new_val)
                    self.widgets[vk].blockSignals(False)
                    self.config_manager.set_value(vk, new_val)

            self.value_changed.emit()
        finally:
            self._syncing_ti = False

    def on_widget_changed(self, key, value):
        """Override to update TI display when relevant properties change."""
        super().on_widget_changed(key, value)
        if getattr(self, '_syncing_ti', False):
            return
        if key in ("thermophysical.ks", "thermophysical.kd",
                    "thermophysical.rhos", "thermophysical.rhod",
                    "thermophysical.albedo", "thermophysical.emissivity",
                    "thermophysical.cp0"):
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
        if isinstance(widget, QDoubleSpinBox):
            # Clamp to widget range
            v = float(value) if value is not None else widget.minimum()
            v = max(widget.minimum(), min(widget.maximum(), v))
            widget.setValue(v)
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
            val = orbit.get(field, -1.0)
            self._set_widget_value(f"generic_body.orbit.{field}",
                                   val if val >= 0 else 0.0)
        self._set_widget_value("generic_body.orbit.epoch",
                               orbit.get("epoch", ""))

        # Spin
        spin = gb.get("spin", {})
        for field in ["period_hours", "pole_ra_deg", "pole_dec_deg", "prime_meridian_deg"]:
            val = spin.get(field, -1.0)
            self._set_widget_value(f"generic_body.spin.{field}",
                                   val if val >= 0 else 0.0)

        # Compute and display obliquity from orbit + spin data
        self._update_obliquity_display()

        # Update banner based on current body
        body = config.get("simulation", {}).get("body", "Europa")
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
        self.add_bool_field(illum_layout, 1, "Use Batch Processing", "illumination.use_batch_processing",
                            "Use batch processing for ray tracing",
                            info_key="illumination.use_batch_processing")

        illum_group.setLayout(illum_layout)
        layout.addWidget(illum_group)

        # Albedo
        albedo_group = QGroupBox("Albedo Configuration")
        albedo_layout = QGridLayout()
        albedo_layout.setColumnStretch(1, 1)

        self.add_file_field(albedo_layout, 0, "Albedo Map:", "illumination.albedo_map",
                            "FITS Files (*.fits *.fit);;All Files (*)",
                            "Path to single albedo map file",
                            info_key="illumination.albedo_map")
        self.add_directory_field(albedo_layout, 1, "Albedo Maps Directory:", "illumination.albedo_maps_directory",
                                 "Directory containing multiple albedo maps",
                                 info_key="illumination.albedo_maps_directory")

        albedo_group.setLayout(albedo_layout)
        layout.addWidget(albedo_group)

        # Positions
        pos_group = QGroupBox("Precomputed Positions")
        pos_layout = QGridLayout()
        pos_layout.setColumnStretch(1, 1)

        self.add_file_field(pos_layout, 0, "Positions File:", "illumination.positions_file",
                            "Binary Files (*.bin);;All Files (*)",
                            "Precomputed Sun/Primary positions",
                            info_key="illumination.positions_file")

        pos_group.setLayout(pos_layout)
        layout.addWidget(pos_group)

        layout.addStretch()
        self.create_scroll_area(content)

    def load_from_config(self):
        config = self.config_manager.get_config()
        illum = config["illumination"]

        self.widgets["illumination.include_primary_reflection"].setChecked(illum["include_primary_reflection"])
        self.widgets["illumination.use_batch_processing"].setChecked(illum["use_batch_processing"])
        self.widgets["illumination.albedo_map"].setText(illum["albedo_map"])
        self.widgets["illumination.albedo_maps_directory"].setText(illum["albedo_maps_directory"])
        self.widgets["illumination.positions_file"].setText(illum["positions_file"])


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

        self.widgets["output.directory"].setText(out["directory"])

        idx = self.widgets["output.format"].findText(out["format"])
        if idx >= 0:
            self.widgets["output.format"].setCurrentIndex(idx)

        idx = self.widgets["output.vtk_mode"].findText(out["vtk_mode"])
        if idx >= 0:
            self.widgets["output.vtk_mode"].setCurrentIndex(idx)

        self.widgets["output.include_primary_ir"].setChecked(out.get("include_primary_ir", False))
        self.widgets["output.locations_file"].setText(out.get("locations_file", ""))
        self.widgets["output.depths_file"].setText(out.get("depths_file", ""))
        self.widgets["output.series_output"].setText(out.get("series_output", ""))


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

        tab_order = [
            ("simulation", "Simulation"),
            ("body_definition", "Body"),
            ("mesh", "Mesh"),
            ("thermophysical", "Properties"),
            ("thermal", "Thermal"),
            ("illumination", "Illumination"),
            ("radiation", "Radiation"),
            ("output", "Output"),
        ]

        for key, label in tab_order:
            self.nav_rail.addItem(label)
            self.stacked_widget.addWidget(self.tabs[key])

        if self._horizons_panel:
            self.nav_rail.addItem("HORIZONS")
            self.stacked_widget.addWidget(self._horizons_panel)

        self.nav_rail.currentRowChanged.connect(self.stacked_widget.setCurrentIndex)
        self.nav_rail.setCurrentRow(0)

        # Container for nav rail + content
        nav_container = QWidget()
        nav_layout = QHBoxLayout(nav_container)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        nav_layout.setSpacing(0)
        nav_layout.addWidget(self.nav_rail)
        nav_layout.addWidget(self.stacked_widget, stretch=1)

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

        preview_header = QLabel("YAML Preview")
        preview_header.setFont(QFont("Helvetica Neue", 11, QFont.Weight.Bold))
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

        # Clear body-specific albedo maps (would load Europa maps otherwise)
        self.config_manager.set_value("illumination.albedo_map", "")
        self.config_manager.set_value("illumination.albedo_maps_directory", "")
        if "illumination.albedo_map" in self.tabs["illumination"].widgets:
            self.tabs["illumination"].widgets["illumination.albedo_map"].blockSignals(True)
            self.tabs["illumination"].widgets["illumination.albedo_map"].setText("")
            self.tabs["illumination"].widgets["illumination.albedo_map"].blockSignals(False)
        if "illumination.albedo_maps_directory" in self.tabs["illumination"].widgets:
            self.tabs["illumination"].widgets["illumination.albedo_maps_directory"].blockSignals(True)
            self.tabs["illumination"].widgets["illumination.albedo_maps_directory"].setText("")
            self.tabs["illumination"].widgets["illumination.albedo_maps_directory"].blockSignals(False)

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
