"""
Configuration Panel for Viewshed GUI.
Tabbed interface for viewshed simulation parameter configuration.
"""

import sys
from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QScrollArea,
    QGroupBox, QGridLayout, QLabel, QLineEdit, QSpinBox, QDoubleSpinBox,
    QCheckBox, QPushButton, QFileDialog, QTextEdit, QSplitter,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QCursor

# Import tooltips
try:
    from viewshed_core.viewshed_tooltips import TOOLTIPS, get_tooltip
except ImportError:
    parent_dir = str(Path(__file__).parent.parent)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    try:
        from viewshed_core.viewshed_tooltips import TOOLTIPS, get_tooltip
    except ImportError:
        TOOLTIPS = {}
        def get_tooltip(key):
            return ""


class InfoButton(QPushButton):
    """Small info button that displays tooltip documentation on hover."""

    def __init__(self, tooltip_key, parent=None):
        super().__init__("\u24d8", parent)
        self.setFixedSize(20, 20)
        self.setFlat(True)
        self.setStyleSheet("""
            QPushButton {
                color: #64b5f6;
                font-size: 14px;
                border: none;
                background: transparent;
            }
            QPushButton:hover {
                color: #90caf9;
            }
        """)
        tooltip_text = get_tooltip(tooltip_key) if tooltip_key else ""
        if tooltip_text:
            self.setToolTip(tooltip_text)
        self.setCursor(QCursor(Qt.CursorShape.WhatsThisCursor))


class BaseViewshedConfigTab(QWidget):
    """Base class for viewshed configuration tabs."""

    value_changed = pyqtSignal()

    def __init__(self, config_manager):
        super().__init__()
        self.config_manager = config_manager
        self.widgets = {}
        self.init_ui()

    def init_ui(self):
        raise NotImplementedError

    def create_scroll_area(self, content_widget):
        scroll = QScrollArea()
        scroll.setWidget(content_widget)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(scroll)

    def add_string_field(self, layout, row, label_text, key, tooltip="", info_key=None):
        label = QLabel(label_text)
        label.setToolTip(tooltip)
        widget = QLineEdit()
        widget.setToolTip(tooltip)
        widget.textChanged.connect(lambda: self.on_widget_changed(key, widget.text()))
        layout.addWidget(label, row, 0)
        layout.addWidget(widget, row, 1)
        if info_key:
            layout.addWidget(InfoButton(info_key), row, 2)
        self.widgets[key] = widget
        return widget

    def add_double_field(self, layout, row, label_text, key, min_val=0.0, max_val=1e12,
                         decimals=4, tooltip="", info_key=None):
        label = QLabel(label_text)
        label.setToolTip(tooltip)
        widget = QDoubleSpinBox()
        widget.setRange(min_val, max_val)
        widget.setDecimals(decimals)
        widget.setToolTip(tooltip)
        widget.valueChanged.connect(lambda: self.on_widget_changed(key, widget.value()))
        layout.addWidget(label, row, 0)
        layout.addWidget(widget, row, 1)
        if info_key:
            layout.addWidget(InfoButton(info_key), row, 2)
        self.widgets[key] = widget
        return widget

    def add_bool_field(self, layout, row, label_text, key, tooltip="", info_key=None):
        widget = QCheckBox(label_text)
        widget.setToolTip(tooltip)
        widget.stateChanged.connect(lambda: self.on_widget_changed(key, widget.isChecked()))
        layout.addWidget(widget, row, 0, 1, 2)
        if info_key:
            layout.addWidget(InfoButton(info_key), row, 2)
        self.widgets[key] = widget
        return widget

    def add_directory_field(self, layout, row, label_text, key, tooltip="", info_key=None):
        label = QLabel(label_text)
        label.setToolTip(tooltip)
        container = QWidget()
        h_layout = QHBoxLayout(container)
        h_layout.setContentsMargins(0, 0, 0, 0)
        widget = QLineEdit()
        widget.setToolTip(tooltip)
        widget.textChanged.connect(lambda: self.on_widget_changed(key, widget.text()))
        browse_btn = QPushButton("Browse...")
        browse_btn.setMaximumWidth(100)
        browse_btn.clicked.connect(lambda: self._browse_directory(widget))
        h_layout.addWidget(widget)
        h_layout.addWidget(browse_btn)
        layout.addWidget(label, row, 0)
        layout.addWidget(container, row, 1)
        if info_key:
            layout.addWidget(InfoButton(info_key), row, 2)
        self.widgets[key] = widget
        return widget

    def add_file_field(self, layout, row, label_text, key, file_filter="All Files (*)",
                       tooltip="", info_key=None):
        label = QLabel(label_text)
        label.setToolTip(tooltip)
        container = QWidget()
        h_layout = QHBoxLayout(container)
        h_layout.setContentsMargins(0, 0, 0, 0)
        widget = QLineEdit()
        widget.setToolTip(tooltip)
        widget.textChanged.connect(lambda: self.on_widget_changed(key, widget.text()))
        browse_btn = QPushButton("...")
        browse_btn.setMaximumWidth(40)
        browse_btn.clicked.connect(lambda: self._browse_file(widget, file_filter))
        h_layout.addWidget(widget)
        h_layout.addWidget(browse_btn)
        layout.addWidget(label, row, 0)
        layout.addWidget(container, row, 1)
        if info_key:
            layout.addWidget(InfoButton(info_key), row, 2)
        self.widgets[key] = widget
        return widget

    def _browse_directory(self, line_edit):
        start_dir = line_edit.text() or "."
        directory = QFileDialog.getExistingDirectory(
            self, "Select Directory", start_dir,
            QFileDialog.Option.ShowDirsOnly
        )
        if directory:
            line_edit.setText(directory)

    def _browse_file(self, line_edit, file_filter):
        start_dir = line_edit.text() or "."
        filename, _ = QFileDialog.getOpenFileName(
            self, "Select File", start_dir, file_filter
        )
        if filename:
            line_edit.setText(filename)

    def on_widget_changed(self, key, value):
        self.config_manager.set_value(key, value)
        self.value_changed.emit()

    def load_from_config(self):
        raise NotImplementedError


# =========================================================================
# TAB 1: Input / Output
# =========================================================================

class InputOutputTab(BaseViewshedConfigTab):
    """Input directories and output settings."""

    def init_ui(self):
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setSpacing(15)

        # Input group
        input_group = QGroupBox("Input")
        input_layout = QGridLayout()
        input_layout.setColumnStretch(1, 1)

        self.add_directory_field(input_layout, 0, "PTM3D Output Directory:",
                                 "input.ptm_output_dir",
                                 info_key="input.ptm_output_dir")
        self.add_file_field(input_layout, 1, "Mesh File (optional):",
                            "input.mesh_file",
                            "PLY Files (*.ply);;All Files (*)",
                            info_key="input.mesh_file")
        self.add_file_field(input_layout, 2, "Waypoints CSV:",
                            "input.waypoints_csv",
                            "CSV Files (*.csv);;All Files (*)",
                            info_key="input.waypoints_csv")

        input_group.setLayout(input_layout)
        layout.addWidget(input_group)

        # Output group
        output_group = QGroupBox("Output")
        output_layout = QGridLayout()
        output_layout.setColumnStretch(1, 1)

        self.add_directory_field(output_layout, 0, "Output Directory:",
                                 "output.output_dir",
                                 info_key="output.output_dir")
        self.add_bool_field(output_layout, 1, "Write truth ice map",
                            "output.write_truth_map",
                            info_key="output.write_truth_map")
        self.add_bool_field(output_layout, 2, "Write per-waypoint maps",
                            "output.write_per_waypoint",
                            info_key="output.write_per_waypoint")

        output_group.setLayout(output_layout)
        layout.addWidget(output_group)

        layout.addStretch()
        self.create_scroll_area(content)

    def load_from_config(self):
        inp = self.config_manager.get_config()["input"]
        out = self.config_manager.get_config()["output"]
        self.widgets["input.ptm_output_dir"].setText(inp["ptm_output_dir"])
        self.widgets["input.mesh_file"].setText(inp["mesh_file"])
        self.widgets["input.waypoints_csv"].setText(inp["waypoints_csv"])
        self.widgets["output.output_dir"].setText(out["output_dir"])
        self.widgets["output.write_truth_map"].setChecked(out["write_truth_map"])
        self.widgets["output.write_per_waypoint"].setChecked(out["write_per_waypoint"])


# =========================================================================
# TAB 2: Mission
# =========================================================================

class MissionTab(BaseViewshedConfigTab):
    """Mission timing and body parameters."""

    def init_ui(self):
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setSpacing(15)

        # Timing
        timing_group = QGroupBox("Mission Timing")
        timing_layout = QGridLayout()
        timing_layout.setColumnStretch(1, 1)

        self.add_string_field(timing_layout, 0, "Start UTC (YYYY-MM-DD):",
                              "mission.start_utc",
                              info_key="mission.start_utc")
        self.add_string_field(timing_layout, 1, "End UTC (YYYY-MM-DD):",
                              "mission.end_utc",
                              info_key="mission.end_utc")

        timing_group.setLayout(timing_layout)
        layout.addWidget(timing_group)

        # Body
        body_group = QGroupBox("Planetary Body")
        body_layout = QGridLayout()
        body_layout.setColumnStretch(1, 1)

        self.add_double_field(body_layout, 0, "Body Radius [m]:",
                              "mission.body_radius_m", 1.0, 1e10, 1,
                              info_key="mission.body_radius_m")

        body_group.setLayout(body_layout)
        layout.addWidget(body_group)

        layout.addStretch()
        self.create_scroll_area(content)

    def load_from_config(self):
        mission = self.config_manager.get_config()["mission"]
        self.widgets["mission.start_utc"].setText(mission["start_utc"])
        self.widgets["mission.end_utc"].setText(mission["end_utc"])
        self.widgets["mission.body_radius_m"].setValue(mission["body_radius_m"])


# =========================================================================
# TAB 3: Viewshed
# =========================================================================

class ViewshedParamsTab(BaseViewshedConfigTab):
    """Viewshed geometry parameters."""

    def init_ui(self):
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setSpacing(15)

        params_group = QGroupBox("Viewshed Parameters")
        params_layout = QGridLayout()
        params_layout.setColumnStretch(1, 1)

        self.add_double_field(params_layout, 0, "Max Emission Angle [deg]:",
                              "viewshed.max_emission_angle_deg", 0.0, 90.0, 1,
                              info_key="viewshed.max_emission_angle_deg")
        self.add_double_field(params_layout, 1, "Max Range [km]:",
                              "viewshed.max_range_km", 0.0, 10000.0, 1,
                              info_key="viewshed.max_range_km")

        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        layout.addStretch()
        self.create_scroll_area(content)

    def load_from_config(self):
        vs = self.config_manager.get_config()["viewshed"]
        self.widgets["viewshed.max_emission_angle_deg"].setValue(vs["max_emission_angle_deg"])
        self.widgets["viewshed.max_range_km"].setValue(vs["max_range_km"])


# =========================================================================
# TAB 4: Ice Distribution
# =========================================================================

class IceDistributionTab(BaseViewshedConfigTab):
    """Ice distribution parameters for truth map generation."""

    def init_ui(self):
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setSpacing(15)

        ice_group = QGroupBox("Ice Distribution")
        ice_layout = QGridLayout()
        ice_layout.setColumnStretch(1, 1)

        self.add_double_field(ice_layout, 0, "Temperature Threshold [K]:",
                              "ice.temperature_threshold_K", 0.0, 300.0, 1,
                              info_key="ice.temperature_threshold_K")
        self.add_double_field(ice_layout, 1, "Ice Fraction:",
                              "ice.ice_fraction", 0.0, 1.0, 3,
                              info_key="ice.ice_fraction")

        ice_layout.addWidget(QLabel(""), 2, 0)  # spacer

        self.add_double_field(ice_layout, 3, "Gradient Scale [K]:",
                              "ice.gradient_scale_K", 0.0, 100.0, 1,
                              info_key="ice.gradient_scale_K")

        # Info label
        self.gradient_info = QLabel(
            "Gradient scale = 0: binary ice (on/off at threshold).\n"
            "Gradient scale > 0: smooth sigmoid transition near threshold."
        )
        self.gradient_info.setStyleSheet("color: #888; font-size: 10px; padding: 4px;")
        ice_layout.addWidget(self.gradient_info, 6, 0, 1, 3)

        ice_group.setLayout(ice_layout)
        layout.addWidget(ice_group)

        layout.addStretch()
        self.create_scroll_area(content)

    def load_from_config(self):
        ice = self.config_manager.get_config()["ice"]
        self.widgets["ice.temperature_threshold_K"].setValue(ice["temperature_threshold_K"])
        self.widgets["ice.ice_fraction"].setValue(ice["ice_fraction"])
        self.widgets["ice.gradient_scale_K"].setValue(ice["gradient_scale_K"])


# =========================================================================
# Main Configuration Panel
# =========================================================================

class ViewshedConfigurationPanel(QWidget):
    """Main configuration panel with tabs and YAML preview."""

    config_changed = pyqtSignal()

    def __init__(self, config_manager):
        super().__init__()
        self.config_manager = config_manager
        self.tabs = {}
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Vertical)

        # Tab widget
        self.tab_widget = QTabWidget()

        self.tabs["input_output"] = InputOutputTab(self.config_manager)
        self.tabs["mission"] = MissionTab(self.config_manager)
        self.tabs["viewshed"] = ViewshedParamsTab(self.config_manager)
        self.tabs["ice"] = IceDistributionTab(self.config_manager)

        self.tab_widget.addTab(self.tabs["input_output"], "Input / Output")
        self.tab_widget.addTab(self.tabs["mission"], "Mission")
        self.tab_widget.addTab(self.tabs["viewshed"], "Viewshed")
        self.tab_widget.addTab(self.tabs["ice"], "Ice Distribution")

        for tab in self.tabs.values():
            tab.value_changed.connect(self.on_value_changed)

        splitter.addWidget(self.tab_widget)

        # YAML preview
        preview_widget = QWidget()
        preview_layout = QVBoxLayout(preview_widget)
        preview_layout.setContentsMargins(8, 8, 8, 8)

        preview_header = QLabel("Configuration Preview")
        preview_header.setFont(QFont("Helvetica Neue", 11, QFont.Weight.Bold))
        preview_layout.addWidget(preview_header)

        self.preview_text = QTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.setFont(QFont("Consolas", 10))
        preview_layout.addWidget(self.preview_text)

        splitter.addWidget(preview_widget)
        splitter.setSizes([400, 150])

        layout.addWidget(splitter)

        self.load_from_config()

    def on_value_changed(self):
        self.update_preview()
        self.config_changed.emit()

    def update_preview(self):
        yaml_str = self.config_manager.generate_yaml()
        self.preview_text.setText(yaml_str)

    def load_from_config(self):
        for tab in self.tabs.values():
            tab.load_from_config()
        self.update_preview()
