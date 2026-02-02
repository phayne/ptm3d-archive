"""
Configuration Panel for Exosphere GUI.
Tabbed interface for exosphere simulation parameter configuration.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QScrollArea,
    QGroupBox, QGridLayout, QLabel, QLineEdit, QSpinBox, QDoubleSpinBox,
    QComboBox, QCheckBox, QPushButton, QFileDialog, QTextEdit, QSplitter
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QCursor

import sys
from pathlib import Path

# Import tooltips
try:
    from exo_core.exo_tooltips import TOOLTIPS, get_tooltip
except ImportError:
    parent_dir = str(Path(__file__).parent.parent)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    try:
        from exo_core.exo_tooltips import TOOLTIPS, get_tooltip
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


class BaseExoConfigTab(QWidget):
    """Base class for exosphere configuration tabs."""

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

    def add_int_field(self, layout, row, label_text, key, min_val=0, max_val=100000,
                      tooltip="", info_key=None):
        label = QLabel(label_text)
        label.setToolTip(tooltip)
        widget = QSpinBox()
        widget.setRange(min_val, max_val)
        widget.setToolTip(tooltip)
        widget.valueChanged.connect(lambda: self.on_widget_changed(key, widget.value()))
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
# TAB 1: General
# =========================================================================

class GeneralTab(BaseExoConfigTab):
    """General simulation parameters."""

    def init_ui(self):
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setSpacing(15)

        # Input directories
        input_group = QGroupBox("Input")
        input_layout = QGridLayout()
        input_layout.setColumnStretch(1, 1)

        self.add_directory_field(input_layout, 0, "PTM3D Output Directory:",
                                 "general.ptm_output_dir",
                                 info_key="general.ptm_output_dir")
        self.add_file_field(input_layout, 1, "Mesh File (optional):",
                            "general.mesh_file",
                            "PLY Files (*.ply);;All Files (*)",
                            info_key="general.mesh_file")

        input_group.setLayout(input_layout)
        layout.addWidget(input_group)

        # Output
        output_group = QGroupBox("Output")
        output_layout = QGridLayout()
        output_layout.setColumnStretch(1, 1)

        self.add_directory_field(output_layout, 0, "Output Directory:",
                                 "general.output_dir",
                                 info_key="general.output_dir")

        output_group.setLayout(output_layout)
        layout.addWidget(output_group)

        # Simulation parameters
        sim_group = QGroupBox("Simulation")
        sim_layout = QGridLayout()
        sim_layout.setColumnStretch(1, 1)

        self.add_int_field(sim_layout, 0, "Timesteps:", "general.timesteps",
                           1, 100000, info_key="general.timesteps")

        sim_group.setLayout(sim_layout)
        layout.addWidget(sim_group)

        # Initial ice
        ice_group = QGroupBox("Initial Ice Distribution")
        ice_layout = QGridLayout()
        ice_layout.setColumnStretch(1, 1)

        self.add_double_field(ice_layout, 0, "Equatorial Ice (mm):",
                              "general.equatorial_ice_mm", 0.0, 10000.0, 2,
                              info_key="general.equatorial_ice_mm")
        self.add_double_field(ice_layout, 1, "Polar Ice (mm):",
                              "general.polar_ice_mm", 0.0, 10000.0, 2,
                              info_key="general.polar_ice_mm")

        ice_group.setLayout(ice_layout)
        layout.addWidget(ice_group)

        layout.addStretch()
        self.create_scroll_area(content)

    def load_from_config(self):
        gen = self.config_manager.get_config()["general"]
        self.widgets["general.ptm_output_dir"].setText(gen["ptm_output_dir"])
        self.widgets["general.mesh_file"].setText(gen.get("mesh_file", ""))
        self.widgets["general.output_dir"].setText(gen["output_dir"])
        self.widgets["general.timesteps"].setValue(gen["timesteps"])
        self.widgets["general.equatorial_ice_mm"].setValue(gen["equatorial_ice_mm"])
        self.widgets["general.polar_ice_mm"].setValue(gen["polar_ice_mm"])


# =========================================================================
# TAB 2: Sputtering
# =========================================================================

class SputteringTab(BaseExoConfigTab):
    """Sputtering configuration."""

    def init_ui(self):
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setSpacing(15)

        # Enable
        enable_group = QGroupBox("Magnetospheric Sputtering")
        enable_layout = QGridLayout()
        enable_layout.setColumnStretch(1, 1)

        self.add_bool_field(enable_layout, 0, "Enable Sputtering",
                            "sputtering.enabled",
                            info_key="sputtering.enabled")

        enable_group.setLayout(enable_layout)
        layout.addWidget(enable_group)

        # Parameters
        self.params_group = QGroupBox("Sputtering Parameters")
        params_layout = QGridLayout()
        params_layout.setColumnStretch(1, 1)

        self.add_double_field(params_layout, 0, "Rate Scale Factor:",
                              "sputtering.rate_scale", 0.01, 100.0, 2,
                              info_key="sputtering.rate_scale")
        self.add_double_field(params_layout, 1, "Hemisphere Contrast:",
                              "sputtering.hemisphere_contrast", 0.1, 100.0, 1,
                              info_key="sputtering.hemisphere_contrast")

        self.params_group.setLayout(params_layout)
        layout.addWidget(self.params_group)

        # Connect enable toggle to parameter group visibility
        self.widgets["sputtering.enabled"].stateChanged.connect(
            self._on_enable_toggled
        )

        layout.addStretch()
        self.create_scroll_area(content)

    def _on_enable_toggled(self):
        enabled = self.widgets["sputtering.enabled"].isChecked()
        self.params_group.setEnabled(enabled)

    def load_from_config(self):
        sput = self.config_manager.get_config()["sputtering"]
        self.widgets["sputtering.enabled"].setChecked(sput["enabled"])
        self.widgets["sputtering.rate_scale"].setValue(sput["rate_scale"])
        self.widgets["sputtering.hemisphere_contrast"].setValue(sput["hemisphere_contrast"])
        self._on_enable_toggled()


# =========================================================================
# TAB 3: Ice Layer + Crystallization
# =========================================================================

class IceLayerTab(BaseExoConfigTab):
    """Ice layer model and crystallization configuration."""

    def init_ui(self):
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setSpacing(15)

        # Ice layer enable
        ice_group = QGroupBox("Ice Layer Model")
        ice_layout = QGridLayout()
        ice_layout.setColumnStretch(1, 1)

        self.add_bool_field(ice_layout, 0, "Enable Ice Layer Model",
                            "ice_layer.enabled",
                            info_key="ice_layer.enabled")
        self.add_double_field(ice_layout, 1, "Dust Fraction:",
                              "ice_layer.dust_fraction", 0.0, 1.0, 3,
                              info_key="ice_layer.dust_fraction")

        ice_group.setLayout(ice_layout)
        layout.addWidget(ice_group)

        # Crystallization sub-group
        self.cryst_group = QGroupBox("Ice Crystallization (Mitchell et al. 2017)")
        cryst_layout = QGridLayout()
        cryst_layout.setColumnStretch(1, 1)

        self.add_bool_field(cryst_layout, 0, "Enable Crystallization",
                            "crystallization.enabled",
                            info_key="crystallization.enabled")
        self.add_double_field(cryst_layout, 1, "Initial Crystallinity:",
                              "crystallization.initial_crystallinity", 0.0, 1.0, 3,
                              info_key="crystallization.initial_crystallinity")
        self.add_double_field(cryst_layout, 2, "Activation Energy (J/mol):",
                              "crystallization.activation_energy", 1000.0, 200000.0, 0,
                              info_key="crystallization.activation_energy")
        self.add_double_field(cryst_layout, 3, "Amorphization Dose (eV/mol):",
                              "crystallization.amorphization_dose", 0.1, 100.0, 1,
                              info_key="crystallization.amorphization_dose")

        cryst_layout.addWidget(QLabel(""), 4, 0)  # spacer row

        self.add_double_field(cryst_layout, 5, "Equatorial Porosity:",
                              "crystallization.equatorial_porosity", 0.0, 1.0, 2,
                              info_key="crystallization.equatorial_porosity")
        self.add_double_field(cryst_layout, 6, "Polar Porosity:",
                              "crystallization.polar_porosity", 0.0, 1.0, 2,
                              info_key="crystallization.polar_porosity")

        cryst_layout.addWidget(QLabel(""), 7, 0)  # spacer row

        self.add_double_field(cryst_layout, 8, "Dose Hemisphere Contrast:",
                              "crystallization.dose_hemisphere_contrast", 0.1, 100.0, 1,
                              info_key="crystallization.dose_hemisphere_contrast")
        self.add_double_field(cryst_layout, 9, "Dose Polar Enhancement:",
                              "crystallization.dose_polar_enhancement", 0.1, 100.0, 1,
                              info_key="crystallization.dose_polar_enhancement")
        self.add_double_field(cryst_layout, 10, "Base Dose Rate (eV/(mol*s)):",
                              "crystallization.base_dose_rate", 1e-10, 1e-2, 8,
                              info_key="crystallization.base_dose_rate")

        self.cryst_group.setLayout(cryst_layout)
        layout.addWidget(self.cryst_group)

        # Connect enable toggles
        self.widgets["ice_layer.enabled"].stateChanged.connect(self._on_ice_toggled)
        self.widgets["crystallization.enabled"].stateChanged.connect(self._on_cryst_toggled)

        layout.addStretch()
        self.create_scroll_area(content)

    def _on_ice_toggled(self):
        enabled = self.widgets["ice_layer.enabled"].isChecked()
        self.widgets["ice_layer.dust_fraction"].setEnabled(enabled)
        if not enabled:
            # Disable crystallization if ice layer is off
            self.widgets["crystallization.enabled"].setChecked(False)
        self.cryst_group.setEnabled(enabled)

    def _on_cryst_toggled(self):
        cryst_enabled = self.widgets["crystallization.enabled"].isChecked()
        if cryst_enabled:
            # Auto-enable ice layer
            self.widgets["ice_layer.enabled"].setChecked(True)
            self.config_manager.set_value("ice_layer.enabled", True)
        # Enable/disable crystallization parameter widgets
        cryst_params = [
            "crystallization.initial_crystallinity",
            "crystallization.activation_energy",
            "crystallization.amorphization_dose",
            "crystallization.equatorial_porosity",
            "crystallization.polar_porosity",
            "crystallization.dose_hemisphere_contrast",
            "crystallization.dose_polar_enhancement",
            "crystallization.base_dose_rate",
        ]
        for key in cryst_params:
            if key in self.widgets:
                self.widgets[key].setEnabled(cryst_enabled)

    def load_from_config(self):
        config = self.config_manager.get_config()
        ice = config["ice_layer"]
        cryst = config["crystallization"]

        self.widgets["ice_layer.enabled"].setChecked(ice["enabled"])
        self.widgets["ice_layer.dust_fraction"].setValue(ice["dust_fraction"])
        self.widgets["crystallization.enabled"].setChecked(cryst["enabled"])
        self.widgets["crystallization.initial_crystallinity"].setValue(cryst["initial_crystallinity"])
        self.widgets["crystallization.activation_energy"].setValue(cryst["activation_energy"])
        self.widgets["crystallization.amorphization_dose"].setValue(cryst["amorphization_dose"])
        self.widgets["crystallization.equatorial_porosity"].setValue(cryst["equatorial_porosity"])
        self.widgets["crystallization.polar_porosity"].setValue(cryst["polar_porosity"])
        self.widgets["crystallization.dose_hemisphere_contrast"].setValue(cryst["dose_hemisphere_contrast"])
        self.widgets["crystallization.dose_polar_enhancement"].setValue(cryst["dose_polar_enhancement"])
        self.widgets["crystallization.base_dose_rate"].setValue(cryst["base_dose_rate"])

        self._on_ice_toggled()
        self._on_cryst_toggled()


# =========================================================================
# TAB 4: Multi-Timescale
# =========================================================================

class MultiscaleTab(BaseExoConfigTab):
    """Multi-timescale evolution configuration."""

    def init_ui(self):
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setSpacing(15)

        # Enable
        enable_group = QGroupBox("Geological Evolution")
        enable_layout = QGridLayout()
        enable_layout.setColumnStretch(1, 1)

        self.add_bool_field(enable_layout, 0, "Enable Multi-Timescale Evolution",
                            "multiscale.enabled",
                            info_key="multiscale.enabled")

        enable_group.setLayout(enable_layout)
        layout.addWidget(enable_group)

        # Parameters
        self.params_group = QGroupBox("Evolution Parameters")
        params_layout = QGridLayout()
        params_layout.setColumnStretch(1, 1)

        self.add_int_field(params_layout, 0, "Number of Epochs:",
                           "multiscale.epochs", 1, 1000000,
                           info_key="multiscale.epochs")
        self.add_int_field(params_layout, 1, "Cycles per Epoch:",
                           "multiscale.cycles_per_epoch", 1, 10000,
                           info_key="multiscale.cycles_per_epoch")
        self.add_int_field(params_layout, 2, "Export Interval (epochs):",
                           "multiscale.export_interval", 0, 100000,
                           info_key="multiscale.export_interval")
        self.add_bool_field(params_layout, 3, "Analytical Acceleration",
                            "multiscale.analytical_acceleration",
                            info_key="multiscale.analytical_acceleration")
        self.add_int_field(params_layout, 4, "Cache Refresh Interval:",
                           "multiscale.cache_refresh", 0, 100000,
                           info_key="multiscale.cache_refresh")

        self.params_group.setLayout(params_layout)
        layout.addWidget(self.params_group)

        # Computed info
        self.info_group = QGroupBox("Computed Timescale")
        info_layout = QVBoxLayout()
        self.total_time_label = QLabel("Total simulated time: --")
        self.total_time_label.setStyleSheet("color: #64b5f6; padding: 5px;")
        info_layout.addWidget(self.total_time_label)
        self.info_group.setLayout(info_layout)
        layout.addWidget(self.info_group)

        # Connect signals
        self.widgets["multiscale.enabled"].stateChanged.connect(self._on_enable_toggled)
        self.widgets["multiscale.epochs"].valueChanged.connect(self._update_time_label)
        self.widgets["multiscale.cycles_per_epoch"].valueChanged.connect(self._update_time_label)

        layout.addStretch()
        self.create_scroll_area(content)

    def _on_enable_toggled(self):
        enabled = self.widgets["multiscale.enabled"].isChecked()
        self.params_group.setEnabled(enabled)
        self.info_group.setEnabled(enabled)
        if enabled:
            self._update_time_label()
        else:
            self.total_time_label.setText("Total simulated time: --")

    def _update_time_label(self):
        if not self.widgets["multiscale.enabled"].isChecked():
            return
        time_info = self.config_manager.compute_total_time()
        if time_info:
            years = time_info["total_years"]
            if years < 1:
                self.total_time_label.setText(
                    f"Total: {time_info['total_days']:.0f} days "
                    f"({time_info['epochs']} epochs x {time_info['cycles_per_epoch']} cycles)"
                )
            else:
                self.total_time_label.setText(
                    f"Total: {years:,.0f} years "
                    f"({time_info['epochs']} epochs x {time_info['cycles_per_epoch']} cycles)"
                )

    def load_from_config(self):
        multi = self.config_manager.get_config()["multiscale"]
        self.widgets["multiscale.enabled"].setChecked(multi["enabled"])
        self.widgets["multiscale.epochs"].setValue(multi["epochs"])
        self.widgets["multiscale.cycles_per_epoch"].setValue(multi["cycles_per_epoch"])
        self.widgets["multiscale.export_interval"].setValue(multi["export_interval"])
        self.widgets["multiscale.analytical_acceleration"].setChecked(multi["analytical_acceleration"])
        self.widgets["multiscale.cache_refresh"].setValue(multi["cache_refresh"])
        self._on_enable_toggled()


# =========================================================================
# Main Configuration Panel
# =========================================================================

class ExoConfigurationPanel(QWidget):
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

        self.tabs["general"] = GeneralTab(self.config_manager)
        self.tabs["sputtering"] = SputteringTab(self.config_manager)
        self.tabs["ice_layer"] = IceLayerTab(self.config_manager)
        self.tabs["multiscale"] = MultiscaleTab(self.config_manager)

        self.tab_widget.addTab(self.tabs["general"], "General")
        self.tab_widget.addTab(self.tabs["sputtering"], "Sputtering")
        self.tab_widget.addTab(self.tabs["ice_layer"], "Ice Layer")
        self.tab_widget.addTab(self.tabs["multiscale"], "Multi-Timescale")

        for tab in self.tabs.values():
            tab.value_changed.connect(self.on_value_changed)

        splitter.addWidget(self.tab_widget)

        # YAML Preview
        preview_widget = QWidget()
        preview_layout = QVBoxLayout(preview_widget)
        preview_layout.setContentsMargins(8, 8, 8, 8)

        preview_header = QLabel("CLI Arguments Preview")
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
        args = self.config_manager.build_command_args()
        # Format args nicely: one flag per line
        lines = []
        i = 0
        while i < len(args):
            arg = args[i]
            if arg.startswith("--") and i + 1 < len(args) and not args[i + 1].startswith("--"):
                lines.append(f"  {arg} {args[i + 1]}")
                i += 2
            else:
                lines.append(f"  {arg}")
                i += 1
        self.preview_text.setText("ptm_exosphere_main \\\n" + " \\\n".join(lines))

    def load_from_config(self):
        for tab in self.tabs.values():
            tab.load_from_config()
        self.update_preview()
