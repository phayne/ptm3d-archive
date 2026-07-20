"""
Parameter Tab Widgets for PTM3D Configuration Builder
Each tab handles a specific category of simulation parameters
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
                             QLabel, QLineEdit, QSpinBox, QDoubleSpinBox,
                             QComboBox, QCheckBox, QPushButton, QGroupBox,
                             QScrollArea, QFileDialog, QFrame, QSizePolicy)
from PyQt5.QtCore import Qt, pyqtSignal, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QFont

from config_manager import detect_hardware_capabilities


class CollapsibleSection(QWidget):
    """Art deco styled collapsible section for grouping related parameters"""

    def __init__(self, title, parent=None, collapsed=False):
        super().__init__(parent)
        self.title = title
        self._collapsed = collapsed

        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Header button
        self.toggle_button = QPushButton(f"{'▶' if collapsed else '▼'} {title}")
        self.toggle_button.setObjectName("collapsibleHeader")
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(collapsed)
        self.toggle_button.clicked.connect(self.toggle)
        main_layout.addWidget(self.toggle_button)

        # Content area
        self.content_area = QWidget()
        self.content_layout = QVBoxLayout(self.content_area)
        self.content_layout.setContentsMargins(10, 5, 10, 10)
        self.content_layout.setSpacing(10)
        main_layout.addWidget(self.content_area)

        # Set initial state
        self.content_area.setVisible(not collapsed)

    def toggle(self):
        """Toggle collapsed state"""
        self._collapsed = not self._collapsed
        symbol = "▶" if self._collapsed else "▼"
        self.toggle_button.setText(f"{symbol} {self.title}")
        self.content_area.setVisible(not self._collapsed)

    def set_collapsed(self, collapsed):
        """Set collapsed state explicitly"""
        if self._collapsed != collapsed:
            self.toggle()

    def add_widget(self, widget):
        """Add a widget to the content area"""
        self.content_layout.addWidget(widget)

    def add_layout(self, layout):
        """Add a layout to the content area"""
        self.content_layout.addLayout(layout)


class HardwareStatusWidget(QWidget):
    """Displays GPU and CPU SIMD capabilities with retro styling"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("hardwareStatus")
        self.capabilities = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(5)

        # Title
        title = QLabel("⚡ HARDWARE CAPABILITIES")
        title.setObjectName("hardwareLabel")
        layout.addWidget(title)

        # GPU status
        self.gpu_label = QLabel("GPU: Detecting...")
        layout.addWidget(self.gpu_label)

        # CPU SIMD status
        self.simd_label = QLabel("SIMD: Detecting...")
        layout.addWidget(self.simd_label)

        # Refresh on init
        self.refresh()

    def refresh(self):
        """Refresh hardware detection"""
        self.capabilities = detect_hardware_capabilities()

        # GPU status
        if self.capabilities["opencl_available"]:
            devices = self.capabilities["opencl_devices"]
            gpu_devices = [d for d in devices if "(CPU)" not in d["name"]]
            if gpu_devices:
                dev = gpu_devices[0]
                self.gpu_label.setText(f"GPU: ✓ {dev['name']} ({dev['memory_gb']} GB)")
                self.gpu_label.setObjectName("hardwareAvailable")
            else:
                self.gpu_label.setText("GPU: ⚠ OpenCL CPU only")
                self.gpu_label.setObjectName("hardwareUnavailable")
        else:
            self.gpu_label.setText("GPU: ✗ OpenCL not available")
            self.gpu_label.setObjectName("hardwareUnavailable")

        # SIMD status
        simd_features = []
        if self.capabilities["avx512_available"]:
            simd_features.append("AVX-512")
        if self.capabilities["avx2_available"]:
            simd_features.append("AVX2")

        if simd_features:
            self.simd_label.setText(f"SIMD: ✓ {', '.join(simd_features)}")
            self.simd_label.setObjectName("hardwareAvailable")
        else:
            self.simd_label.setText("SIMD: Basic (no AVX)")
            self.simd_label.setObjectName("hardwareUnavailable")

        # Force style update
        self.gpu_label.style().unpolish(self.gpu_label)
        self.gpu_label.style().polish(self.gpu_label)
        self.simd_label.style().unpolish(self.simd_label)
        self.simd_label.style().polish(self.simd_label)


class BaseTab(QWidget):
    """Base class for all parameter tabs"""
    
    value_changed = pyqtSignal()
    
    def __init__(self, config_manager):
        super().__init__()
        self.config_manager = config_manager
        self.widgets = {}  # Store widget references
        self.init_ui()
        
    def init_ui(self):
        """Initialize UI - to be implemented by subclasses"""
        raise NotImplementedError
        
    def create_scroll_area(self, content_widget):
        """Wrap content in a scroll area"""
        scroll = QScrollArea()
        scroll.setWidget(content_widget)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(scroll)
        
    def create_section_header(self, text):
        """Create a section header label"""
        label = QLabel(text)
        label.setObjectName("sectionHeader")
        return label
        
    def add_string_field(self, layout, row, label_text, key, tooltip=""):
        """Add a string input field"""
        label = QLabel(label_text)
        label.setToolTip(tooltip)
        
        widget = QLineEdit()
        widget.setToolTip(tooltip)
        widget.textChanged.connect(lambda: self.on_widget_changed(key, widget.text()))
        
        layout.addWidget(label, row, 0)
        layout.addWidget(widget, row, 1)
        
        self.widgets[key] = widget
        return widget
        
    def add_int_field(self, layout, row, label_text, key, min_val=0, max_val=10000, tooltip=""):
        """Add an integer input field"""
        label = QLabel(label_text)
        label.setToolTip(tooltip)
        
        widget = QSpinBox()
        widget.setRange(min_val, max_val)
        widget.setToolTip(tooltip)
        widget.valueChanged.connect(lambda: self.on_widget_changed(key, widget.value()))
        
        layout.addWidget(label, row, 0)
        layout.addWidget(widget, row, 1)
        
        self.widgets[key] = widget
        return widget
        
    def add_double_field(self, layout, row, label_text, key, min_val=0.0, max_val=10000.0, 
                         decimals=2, tooltip=""):
        """Add a double input field"""
        label = QLabel(label_text)
        label.setToolTip(tooltip)
        
        widget = QDoubleSpinBox()
        widget.setRange(min_val, max_val)
        widget.setDecimals(decimals)
        widget.setToolTip(tooltip)
        widget.valueChanged.connect(lambda: self.on_widget_changed(key, widget.value()))
        
        layout.addWidget(label, row, 0)
        layout.addWidget(widget, row, 1)
        
        self.widgets[key] = widget
        return widget
        
    def add_combo_field(self, layout, row, label_text, key, options, tooltip=""):
        """Add a combo box field"""
        label = QLabel(label_text)
        label.setToolTip(tooltip)
        
        widget = QComboBox()
        widget.addItems(options)
        widget.setToolTip(tooltip)
        widget.currentTextChanged.connect(lambda: self.on_widget_changed(key, widget.currentText()))
        
        layout.addWidget(label, row, 0)
        layout.addWidget(widget, row, 1)
        
        self.widgets[key] = widget
        return widget
        
    def add_bool_field(self, layout, row, label_text, key, tooltip=""):
        """Add a checkbox field"""
        widget = QCheckBox(label_text)
        widget.setToolTip(tooltip)
        widget.stateChanged.connect(lambda: self.on_widget_changed(key, widget.isChecked()))
        
        layout.addWidget(widget, row, 0, 1, 2)
        
        self.widgets[key] = widget
        return widget
        
    def add_file_field(self, layout, row, label_text, key, file_filter="All Files (*)", tooltip=""):
        """Add a file selector field"""
        label = QLabel(label_text)
        label.setToolTip(tooltip)
        
        # Create a container widget to hold the horizontal layout
        container = QWidget()
        h_layout = QHBoxLayout(container)
        h_layout.setContentsMargins(0, 0, 0, 0)
        
        widget = QLineEdit()
        widget.setToolTip(tooltip)
        widget.textChanged.connect(lambda: self.on_widget_changed(key, widget.text()))
        
        browse_btn = QPushButton("...")
        browse_btn.setMaximumWidth(40)
        browse_btn.clicked.connect(lambda: self.browse_file(widget, file_filter))
        
        h_layout.addWidget(widget)
        h_layout.addWidget(browse_btn)
        
        layout.addWidget(label, row, 0)
        layout.addWidget(container, row, 1)  # Add container widget, not layout
        
        self.widgets[key] = widget
        return widget
        
    def browse_file(self, line_edit, file_filter):
        """Open file browser dialog"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Select File",
            line_edit.text() if line_edit.text() else "",
            file_filter
        )
        if filename:
            line_edit.setText(filename)
    
    def add_directory_field(self, layout, row, label_text, key, tooltip=""):
        """Add a directory selection field (for directories, not files)"""
        label = QLabel(label_text)
        label.setToolTip(tooltip)
        
        # Create a container widget to hold the horizontal layout
        container = QWidget()
        h_layout = QHBoxLayout(container)
        h_layout.setContentsMargins(0, 0, 0, 0)
        
        widget = QLineEdit()
        widget.setToolTip(tooltip)
        widget.textChanged.connect(lambda: self.on_widget_changed(key, widget.text()))
        
        browse_btn = QPushButton("Browse...")
        browse_btn.setMaximumWidth(100)
        browse_btn.clicked.connect(lambda: self.browse_directory(widget))
        
        h_layout.addWidget(widget)
        h_layout.addWidget(browse_btn)
        
        layout.addWidget(label, row, 0)
        layout.addWidget(container, row, 1)  # Add container widget, not layout
        
        self.widgets[key] = widget
        return widget
    
    def browse_directory(self, line_edit):
        """Open directory selection dialog"""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Directory",
            line_edit.text() if line_edit.text() else ".",
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        if directory:
            line_edit.setText(directory)
            
    def on_widget_changed(self, key, value):
        """Handle widget value changes"""
        self.config_manager.set_value(key, value)
        self.value_changed.emit()
        
    def load_from_config(self):
        """Load values from config manager - to be implemented by subclasses"""
        raise NotImplementedError


class SimulationTab(BaseTab):
    """Simulation parameters tab"""
    
    def init_ui(self):
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setAlignment(Qt.AlignTop)
        layout.setSpacing(15)
        
        # Time Parameters Section
        time_group = QGroupBox("TIME PARAMETERS")
        time_layout = QGridLayout()
        time_layout.setColumnStretch(1, 1)
        time_layout.setHorizontalSpacing(15)
        time_layout.setVerticalSpacing(10)
        
        self.add_string_field(
            time_layout, 0, "Start Time:", "simulation.start_time",
            "ISO 8601 format (e.g., 2025-01-01T00:00:00)"
        )
        
        self.add_double_field(
            time_layout, 1, "Duration (days):", "simulation.duration_days",
            0.1, 3650.0, 2,
            "Simulation duration in Earth days"
        )
        
        self.add_int_field(
            time_layout, 2, "Time Steps:", "simulation.timesteps",
            1, 10000,
            "Number of output timesteps"
        )
        
        time_group.setLayout(time_layout)
        layout.addWidget(time_group)
        
        # Body Selection Section
        body_group = QGroupBox("PLANETARY BODY")
        body_layout = QGridLayout()
        body_layout.setColumnStretch(1, 1)
        body_layout.setHorizontalSpacing(15)
        body_layout.setVerticalSpacing(10)
        
        bodies = ["Europa", "Moon", "Mercury", "Enceladus", "Titan", "Mars", "Ceres"]
        self.add_combo_field(
            body_layout, 0, "Target Body:", "simulation.body",
            bodies,
            "Select the planetary body to simulate"
        )
        
        body_group.setLayout(body_layout)
        layout.addWidget(body_group)
        
        # Add info label
        info_label = QLabel(
            "ℹ Configure the fundamental simulation parameters.\n"
            "Start time uses ISO 8601 format. Duration is in Earth days."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #88ccff; padding: 10px; font-size: 10px;")
        layout.addWidget(info_label)
        
        layout.addStretch()
        
        self.create_scroll_area(content)
        
    def load_from_config(self):
        """Load values from configuration"""
        config = self.config_manager.get_config()
        
        self.widgets["simulation.start_time"].setText(config["simulation"]["start_time"])
        self.widgets["simulation.duration_days"].setValue(config["simulation"]["duration_days"])
        self.widgets["simulation.timesteps"].setValue(config["simulation"]["timesteps"])
        
        body_index = self.widgets["simulation.body"].findText(config["simulation"]["body"])
        if body_index >= 0:
            self.widgets["simulation.body"].setCurrentIndex(body_index)


class MeshTab(BaseTab):
    """Mesh configuration tab"""
    
    def init_ui(self):
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setAlignment(Qt.AlignTop)
        layout.setSpacing(15)
        
        # Global Mesh Section
        global_group = QGroupBox("GLOBAL MESH")
        global_layout = QGridLayout()
        global_layout.setColumnStretch(1, 1)
        
        self.add_int_field(
            global_layout, 0, "Resolution:", "mesh.resolution",
            1, 10,
            "Icosphere resolution level (higher = more faces)"
        )
        
        global_group.setLayout(global_layout)
        layout.addWidget(global_group)
        
        # Topographic Mesh Section
        topo_group = QGroupBox("TOPOGRAPHIC MESH")
        topo_layout = QGridLayout()
        topo_layout.setColumnStretch(1, 1)
        
        self.add_bool_field(
            topo_layout, 0, "Use Topographic Mesh", "mesh.use_topographic",
            "Use pre-computed topographic meshes (requires generation)"
        )
        
        self.add_file_field(
            topo_layout, 1, "Topographic Directory:", "mesh.topo_directory",
            "Directories",
            "Directory containing pre-computed topographic meshes"
        )
        
        topo_group.setLayout(topo_layout)
        layout.addWidget(topo_group)
        
        # Custom Mesh Section
        custom_group = QGroupBox("CUSTOM MESH")
        custom_layout = QGridLayout()
        custom_layout.setColumnStretch(1, 1)
        
        self.add_file_field(
            custom_layout, 0, "Import File:", "mesh.import_file",
            "Mesh Files (*.obj *.ply *.vtk);;All Files (*)",
            "Import custom triangulated mesh (OBJ, PLY, VTK)"
        )
        
        self.add_double_field(
            custom_layout, 1, "Scale Factor:", "mesh.scale_factor",
            0.001, 1000.0, 3,
            "Scaling factor for imported mesh"
        )
        
        self.add_bool_field(
            custom_layout, 2, "Export Basic Mesh", "mesh.export_basic",
            "Export basic mesh to VTK format"
        )
        
        custom_group.setLayout(custom_layout)
        layout.addWidget(custom_group)
        
        layout.addStretch()
        self.create_scroll_area(content)
        
    def load_from_config(self):
        """Load values from configuration"""
        config = self.config_manager.get_config()
        mesh = config["mesh"]
        
        self.widgets["mesh.resolution"].setValue(mesh["resolution"])
        self.widgets["mesh.use_topographic"].setChecked(mesh["use_topographic"])
        self.widgets["mesh.topo_directory"].setText(mesh["topo_directory"])
        self.widgets["mesh.import_file"].setText(mesh["import_file"])
        self.widgets["mesh.scale_factor"].setValue(mesh["scale_factor"])
        self.widgets["mesh.export_basic"].setChecked(mesh["export_basic"])


class ThermalTab(BaseTab):
    """Thermal model configuration tab - updated with GPU and equilibration options"""

    def init_ui(self):
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setAlignment(Qt.AlignTop)
        layout.setSpacing(15)

        # Hardware Status at top
        self.hardware_status = HardwareStatusWidget()
        layout.addWidget(self.hardware_status)

        # Model Control Section
        control_group = QGroupBox("MODEL CONTROL")
        control_layout = QGridLayout()
        control_layout.setColumnStretch(1, 1)

        self.add_bool_field(
            control_layout, 0, "Run Thermal Model", "thermal.run_thermal_mesh",
            "Enable thermal calculations for entire mesh"
        )

        self.add_int_field(
            control_layout, 1, "CPU Threads:", "thermal.threads",
            0, 128,
            "Number of parallel threads (0 = auto-detect)"
        )

        self.add_bool_field(
            control_layout, 2, "Compute Statistics", "thermal.compute_statistics",
            "Calculate min/max/mean temperature statistics"
        )

        control_group.setLayout(control_layout)
        layout.addWidget(control_group)

        # GPU Acceleration Section (NEW)
        gpu_group = QGroupBox("GPU ACCELERATION")
        gpu_layout = QGridLayout()
        gpu_layout.setColumnStretch(1, 1)

        self.add_bool_field(
            gpu_layout, 0, "Use GPU for Thermal Solver", "thermal.use_gpu",
            "Use OpenCL GPU acceleration for thermal calculations (experimental)"
        )

        self.add_bool_field(
            gpu_layout, 1, "Use GPU for Illumination Ray Tracing", "thermal.use_gpu_illumination",
            "Use OpenCL GPU acceleration for shadow ray tracing (experimental)"
        )

        gpu_group.setLayout(gpu_layout)
        layout.addWidget(gpu_group)

        # Equilibration Section
        equil_group = QGroupBox("EQUILIBRATION")
        equil_layout = QGridLayout()
        equil_layout.setColumnStretch(1, 1)

        self.add_int_field(
            equil_layout, 0, "Equilibration Cycles:", "thermal.equilibration_orbits",
            1, 100,
            "Number of day/night cycles for thermal equilibration\n"
            "(Default: 2 rotational periods)"
        )

        equil_group.setLayout(equil_layout)
        layout.addWidget(equil_group)

        # Depth Grid Section (NEW)
        grid_group = QGroupBox("DEPTH GRID")
        grid_layout = QGridLayout()
        grid_layout.setColumnStretch(1, 1)

        self.add_combo_field(
            grid_layout, 0, "Grid Quality:", "thermal.grid_quality",
            ["auto", "coarse", "standard", "fine"],
            "Depth discretization quality:\n"
            "• coarse: ~40 layers, fast\n"
            "• standard: ~60 layers\n"
            "• fine: ~100 layers, accurate\n"
            "• auto: adapts to material properties"
        )

        grid_group.setLayout(grid_layout)
        layout.addWidget(grid_group)

        # Integration Window Section
        integ_group = QGroupBox("INTEGRATION WINDOW")
        integ_layout = QGridLayout()
        integ_layout.setColumnStretch(1, 1)

        self.add_double_field(
            integ_layout, 0, "Integration Days:", "thermal.integration_days",
            -1.0, 10000.0, 2,
            "Integration window in days (-1 = use output window)"
        )

        self.add_int_field(
            integ_layout, 1, "Integration Steps:", "thermal.integration_steps",
            -1, 100000,
            "Integration steps override (-1 = auto)"
        )

        self.add_double_field(
            integ_layout, 2, "Offset (days):", "thermal.integration_offset_days",
            0.0, 1000.0, 2,
            "Offset for integration window start"
        )

        self.add_int_field(
            integ_layout, 3, "Offset (steps):", "thermal.integration_offset_steps",
            0, 10000,
            "Offset in timesteps"
        )

        integ_group.setLayout(integ_layout)
        layout.addWidget(integ_group)

        # Shadow/Crater Flux Section
        shadow_group = QGroupBox("SHADOW/CRATER FLUX")
        shadow_layout = QGridLayout()
        shadow_layout.setColumnStretch(1, 1)

        self.add_bool_field(
            shadow_layout, 0, "Enable Shadow Flux", "thermal.enable_shadow_flux",
            "Enable crater flux enhancement calculations"
        )

        self.add_double_field(
            shadow_layout, 1, "Crater D/h Ratio:", "thermal.crater_dd_ratio",
            0.1, 100.0, 1,
            "Crater diameter/depth ratio"
        )

        shadow_group.setLayout(shadow_layout)
        layout.addWidget(shadow_group)

        # Maps Section
        maps_group = QGroupBox("THERMAL PROPERTY MAPS")
        maps_layout = QGridLayout()
        maps_layout.setColumnStretch(1, 1)

        self.add_file_field(
            maps_layout, 0, "Thermal Inertia Map:", "thermal.thermal_inertia_map",
            "FITS Files (*.fits *.fit);;All Files (*)",
            "Path to thermal inertia map file"
        )

        maps_group.setLayout(maps_layout)
        layout.addWidget(maps_group)

        # Info
        info = QLabel(
            "ℹ Thermal model calculates subsurface temperature profiles.\n"
            "Spectral equilibration (FFT) is faster and recommended for most cases.\n"
            "GPU acceleration requires OpenCL support."
        )
        info.setWordWrap(True)
        info.setObjectName("infoBox")
        layout.addWidget(info)

        layout.addStretch()
        self.create_scroll_area(content)

    def load_from_config(self):
        config = self.config_manager.get_config()
        thermal = config["thermal"]

        self.widgets["thermal.run_thermal_mesh"].setChecked(thermal["run_thermal_mesh"])
        self.widgets["thermal.threads"].setValue(thermal["threads"])
        self.widgets["thermal.compute_statistics"].setChecked(thermal["compute_statistics"])
        self.widgets["thermal.use_gpu"].setChecked(thermal.get("use_gpu", False))
        self.widgets["thermal.use_gpu_illumination"].setChecked(thermal.get("use_gpu_illumination", False))
        self.widgets["thermal.equilibration_orbits"].setValue(thermal.get("equilibration_orbits", 2))

        # Grid quality combo
        grid_idx = self.widgets["thermal.grid_quality"].findText(thermal.get("grid_quality", "auto"))
        if grid_idx >= 0:
            self.widgets["thermal.grid_quality"].setCurrentIndex(grid_idx)

        self.widgets["thermal.integration_days"].setValue(thermal["integration_days"])
        self.widgets["thermal.integration_steps"].setValue(thermal["integration_steps"])
        self.widgets["thermal.integration_offset_days"].setValue(thermal["integration_offset_days"])
        self.widgets["thermal.integration_offset_steps"].setValue(thermal["integration_offset_steps"])
        self.widgets["thermal.enable_shadow_flux"].setChecked(thermal["enable_shadow_flux"])
        self.widgets["thermal.crater_dd_ratio"].setValue(thermal["crater_dd_ratio"])
        self.widgets["thermal.thermal_inertia_map"].setText(thermal["thermal_inertia_map"])


class ThermophysicalTab(BaseTab):
    """Thermophysical properties configuration tab"""

    # Signal emitted when body changes to update defaults
    body_changed = pyqtSignal(str)

    # Default properties by body (matches planetary_properties.h)
    BODY_DEFAULTS = {
        "Europa": {
            "albedo": 0.5, "emissivity": 0.9, "Qb": 0.03,
            "ks": 0.0096, "kd": 0.05, "rhos": 500.0, "rhod": 700.0,
            "H": 0.07, "cp0": 0.0, "incidence_angle_albedo": False,
        },
        "Moon": {
            "albedo": 0.12, "emissivity": 0.95, "Qb": 0.018,
            "ks": 7.4e-4, "kd": 3.4e-3, "rhos": 1100.0, "rhod": 1800.0,
            "H": 0.06, "cp0": -3.6125, "incidence_angle_albedo": True,
        },
        "Mercury": {
            "albedo": 0.07, "emissivity": 0.93, "Qb": 0.02,
            "ks": 0.002, "kd": 2.0, "rhos": 1800.0, "rhod": 2200.0,
            "H": 0.1, "cp0": 800.0, "incidence_angle_albedo": True,
        },
        "Enceladus": {
            "albedo": 0.81, "emissivity": 0.9, "Qb": 0.025,
            "ks": 0.001, "kd": 0.01, "rhos": 500.0, "rhod": 700.0,
            "H": 0.05, "cp0": 0.0, "incidence_angle_albedo": False,
        },
        "Titan": {
            "albedo": 0.22, "emissivity": 0.9, "Qb": 0.005,
            "ks": 0.02, "kd": 0.2, "rhos": 500.0, "rhod": 700.0,
            "H": 0.1, "cp0": 0.0, "incidence_angle_albedo": False,
        },
        "Mars": {
            "albedo": 0.25, "emissivity": 0.95, "Qb": 0.02,
            "ks": 0.02, "kd": 2.0, "rhos": 1500.0, "rhod": 2000.0,
            "H": 0.1, "cp0": 800.0, "incidence_angle_albedo": False,
        },
        "Ceres": {
            "albedo": 0.09, "emissivity": 0.95, "Qb": 0.01,
            "ks": 0.001, "kd": 0.01, "rhos": 1200.0, "rhod": 1500.0,
            "H": 0.05, "cp0": 800.0, "incidence_angle_albedo": False,
        },
    }

    def init_ui(self):
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setAlignment(Qt.AlignTop)
        layout.setSpacing(15)

        # Current body label
        body_info = QLabel("Properties shown are defaults for the selected body.")
        body_info.setStyleSheet("color: #88ccff; font-size: 10px; padding: 5px;")
        layout.addWidget(body_info)

        # ===== SURFACE PROPERTIES =====
        surface_group = QGroupBox("SURFACE PROPERTIES")
        surface_layout = QGridLayout()
        surface_layout.setColumnStretch(1, 1)
        surface_layout.setHorizontalSpacing(15)
        surface_layout.setVerticalSpacing(8)

        # Albedo with default checkbox
        self.add_property_with_default(
            surface_layout, 0, "Bond Albedo:", "thermophysical.albedo",
            "thermophysical.use_default_albedo", 0.0, 1.0, 3,
            "Fraction of incident solar radiation reflected (0-1)"
        )

        # Emissivity with default checkbox
        self.add_property_with_default(
            surface_layout, 1, "Emissivity:", "thermophysical.emissivity",
            "thermophysical.use_default_emissivity", 0.0, 1.0, 3,
            "Thermal emissivity for IR radiation (0-1)"
        )

        # Incidence-angle dependent albedo checkbox
        self.add_bool_field(
            surface_layout, 2, "Incidence-Angle Dependent Albedo",
            "thermophysical.incidence_angle_albedo",
            "Albedo increases at grazing angles (Hayne et al. 2017)"
        )

        surface_group.setLayout(surface_layout)
        layout.addWidget(surface_group)

        # ===== THERMAL CONDUCTIVITY =====
        cond_group = QGroupBox("THERMAL CONDUCTIVITY")
        cond_layout = QGridLayout()
        cond_layout.setColumnStretch(1, 1)
        cond_layout.setHorizontalSpacing(15)
        cond_layout.setVerticalSpacing(8)

        self.add_property_with_default(
            cond_layout, 0, "Surface ks (W/m/K):", "thermophysical.ks",
            "thermophysical.use_default_ks", 1e-6, 10.0, 6,
            "Surface thermal conductivity"
        )

        self.add_property_with_default(
            cond_layout, 1, "Deep kd (W/m/K):", "thermophysical.kd",
            "thermophysical.use_default_kd", 1e-6, 100.0, 6,
            "Deep thermal conductivity (at depth >> H)"
        )

        self.add_property_with_default(
            cond_layout, 2, "Scale Height H (m):", "thermophysical.H",
            "thermophysical.use_default_H", 0.001, 1.0, 4,
            "e-folding depth for exponential property variation"
        )

        cond_group.setLayout(cond_layout)
        layout.addWidget(cond_group)

        # ===== DENSITY =====
        density_group = QGroupBox("DENSITY")
        density_layout = QGridLayout()
        density_layout.setColumnStretch(1, 1)
        density_layout.setHorizontalSpacing(15)
        density_layout.setVerticalSpacing(8)

        self.add_property_with_default(
            density_layout, 0, "Surface rhos (kg/m³):", "thermophysical.rhos",
            "thermophysical.use_default_rhos", 100.0, 5000.0, 1,
            "Surface bulk density"
        )

        self.add_property_with_default(
            density_layout, 1, "Deep rhod (kg/m³):", "thermophysical.rhod",
            "thermophysical.use_default_rhod", 100.0, 5000.0, 1,
            "Deep bulk density (at depth >> H)"
        )

        density_group.setLayout(density_layout)
        layout.addWidget(density_group)

        # ===== HEAT CAPACITY =====
        cp_group = QGroupBox("HEAT CAPACITY")
        cp_layout = QGridLayout()
        cp_layout.setColumnStretch(1, 1)
        cp_layout.setHorizontalSpacing(15)
        cp_layout.setVerticalSpacing(8)

        self.add_property_with_default(
            cp_layout, 0, "Reference cp0 (J/kg/K):", "thermophysical.cp0",
            "thermophysical.use_default_cp0", -100.0, 2000.0, 4,
            "Base heat capacity (may be T-dependent)"
        )

        cp_group.setLayout(cp_layout)
        layout.addWidget(cp_group)

        # ===== INTERNAL HEAT =====
        heat_group = QGroupBox("INTERNAL HEAT")
        heat_layout = QGridLayout()
        heat_layout.setColumnStretch(1, 1)
        heat_layout.setHorizontalSpacing(15)
        heat_layout.setVerticalSpacing(8)

        self.add_property_with_default(
            heat_layout, 0, "Heat Flow Qb (W/m²):", "thermophysical.Qb",
            "thermophysical.use_default_Qb", 0.0, 1.0, 4,
            "Geothermal/internal heat flux at bottom boundary"
        )

        heat_group.setLayout(heat_layout)
        layout.addWidget(heat_group)

        # Info label
        info = QLabel(
            "ℹ Thermophysical properties control surface and subsurface heat transfer.\n"
            "Check 'Use Default' to apply body-specific values from literature.\n"
            "Uncheck to specify custom values for specialized modeling."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #88ccff; padding: 10px; font-size: 10px;")
        layout.addWidget(info)

        layout.addStretch()
        self.create_scroll_area(content)

    def add_property_with_default(self, layout, row, label_text, value_key, default_key,
                                   min_val, max_val, decimals, tooltip=""):
        """Add a property field with a 'Use Default' checkbox"""
        # Label
        label = QLabel(label_text)
        label.setToolTip(tooltip)
        layout.addWidget(label, row, 0)

        # Value spinbox
        value_widget = QDoubleSpinBox()
        value_widget.setRange(min_val, max_val)
        value_widget.setDecimals(decimals)
        value_widget.setToolTip(tooltip)
        value_widget.valueChanged.connect(
            lambda v: self.on_widget_changed(value_key, v)
        )
        layout.addWidget(value_widget, row, 1)

        # Use default checkbox
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
        use_default = (state == Qt.Checked)
        self.config_manager.set_value(default_key, use_default)

        # Disable/enable the value widget based on checkbox
        self.widgets[value_key].setEnabled(not use_default)

        # If using default, update to show default value
        if use_default:
            self.update_to_body_defaults()

        self.value_changed.emit()

    def on_body_changed(self, body_name):
        """Update displayed defaults when body selection changes"""
        self.update_to_body_defaults(body_name)

    def update_to_body_defaults(self, body_name=None):
        """Update all fields to show body defaults"""
        if body_name is None:
            config = self.config_manager.get_config()
            body_name = config.get("simulation", {}).get("body", "Europa")

        defaults = self.BODY_DEFAULTS.get(body_name, self.BODY_DEFAULTS["Europa"])

        # Property mapping: (value_key, default_key, default_attr)
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
        ]

        for value_key, default_key, attr in properties:
            if value_key in self.widgets and default_key in self.widgets:
                use_default = self.widgets[default_key].isChecked()
                if use_default:
                    # Block signals to avoid triggering config updates
                    self.widgets[value_key].blockSignals(True)
                    self.widgets[value_key].setValue(defaults[attr])
                    self.widgets[value_key].blockSignals(False)

        # Update incidence angle albedo checkbox
        if "thermophysical.incidence_angle_albedo" in self.widgets:
            self.widgets["thermophysical.incidence_angle_albedo"].setChecked(
                defaults.get("incidence_angle_albedo", False)
            )

    def load_from_config(self):
        """Load values from configuration"""
        config = self.config_manager.get_config()
        thermo = config.get("thermophysical", {})

        # Property mapping
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
        ]

        for value_key, default_key, value_attr, default_attr in properties:
            if value_key in self.widgets:
                self.widgets[value_key].setValue(thermo.get(value_attr, 0.0))
            if default_key in self.widgets:
                use_default = thermo.get(default_attr, True)
                self.widgets[default_key].setChecked(use_default)
                self.widgets[value_key].setEnabled(not use_default)

        # Incidence angle albedo
        if "thermophysical.incidence_angle_albedo" in self.widgets:
            self.widgets["thermophysical.incidence_angle_albedo"].setChecked(
                thermo.get("incidence_angle_albedo", True)
            )

        # Update displayed defaults based on current body
        self.update_to_body_defaults()


class IlluminationTab(BaseTab):
    """Illumination parameters tab"""
    
    def init_ui(self):
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setAlignment(Qt.AlignTop)
        layout.setSpacing(15)

        illum_group = QGroupBox("SOLAR ILLUMINATION")
        illum_layout = QGridLayout()
        illum_layout.setColumnStretch(1, 1)
        self.add_bool_field(
            illum_layout, 0, "Include Primary Reflection", "illumination.include_primary_reflection",
            "Include reflected light from primary body (Jupiter, Earth, etc.)"
        )
        self.add_bool_field(
            illum_layout, 1, "Use Batch Processing", "illumination.use_batch_processing",
            "Use batch processing for ray tracing (faster for large meshes)"
        )
        illum_group.setLayout(illum_layout)
        layout.addWidget(illum_group)

        albedo_group = QGroupBox("ALBEDO CONFIGURATION")
        albedo_layout = QGridLayout()
        albedo_layout.setColumnStretch(1, 1)
        self.add_file_field(
            albedo_layout, 0, "Single Albedo Map:", "illumination.albedo_map",
            "FITS Files (*.fits *.fit);;All Files (*)",
            "Path to single albedo map file (optional)"
        )
        # add directory picker
        self.add_directory_field(
            albedo_layout, 1, "Albedo Maps Directory:", "illumination.albedo_maps_directory",
            "Directory containing multiple albedo map files"
        )
        albedo_group.setLayout(albedo_layout)
        layout.addWidget(albedo_group)

        positions_group = QGroupBox("PRECOMPUTED POSITIONS")
        positions_layout = QGridLayout()
        positions_layout.setColumnStretch(1, 1)
        self.add_file_field(
            positions_layout, 0, "Positions File:", "illumination.positions_file",
            "Binary Files (*.bin);;All Files (*)",
            "Precomputed Sun/Primary positions (optional)"
        )
        positions_group.setLayout(positions_layout)
        layout.addWidget(positions_group)

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


class OutputTab(BaseTab):
    """Output configuration tab"""
    
    def init_ui(self):
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setAlignment(Qt.AlignTop)
        layout.setSpacing(15)

        dir_group = QGroupBox("OUTPUT DESTINATION")
        dir_layout = QGridLayout()
        dir_layout.setColumnStretch(1, 1)
        self.add_directory_field(
            dir_layout, 0, "Output Directory:", "output.directory",
            "Directory where simulation results are written"
        )
        self.add_combo_field(
            dir_layout, 1, "Format:", "output.format",
            ["binary", "vtk"], "Output format"
        )
        self.add_combo_field(
            dir_layout, 2, "VTK Mode:", "output.vtk_mode",
            ["thermal", "complete", "both"], "VTK export mode"
        )
        self.add_bool_field(
            dir_layout, 3, "Include Primary IR (VTK)", "output.include_primary_ir",
            "Include primary IR in VTK export"
        )
        dir_group.setLayout(dir_layout)
        layout.addWidget(dir_group)

        # Custom VTK Filenames Section (NEW)
        vtk_group = QGroupBox("CUSTOM VTK FILENAMES (Optional)")
        vtk_layout = QGridLayout()
        vtk_layout.setColumnStretch(1, 1)
        self.add_string_field(
            vtk_layout, 0, "Thermal VTK File:", "output.thermal_vtk_file",
            "Custom filename for thermal VTK output (leave empty for default)"
        )
        self.add_string_field(
            vtk_layout, 1, "Complete VTK File:", "output.complete_vtk_file",
            "Custom filename for complete VTK output (leave empty for default)"
        )
        vtk_group.setLayout(vtk_layout)
        layout.addWidget(vtk_group)

        loc_group = QGroupBox("LOCATION EXPORTS")
        loc_layout = QGridLayout()
        loc_layout.setColumnStretch(1, 1)
        self.add_file_field(
            loc_layout, 0, "Locations File:", "output.locations_file",
            "CSV Files (*.csv);;Text Files (*.txt);;All Files (*)",
            "CSV/TXT with rows: name,lat,lon or lat,lon (degrees)"
        )
        self.add_file_field(
            loc_layout, 1, "Depths File:", "output.depths_file",
            "Text Files (*.txt *.dat *.csv);;All Files (*)",
            "One depth per line (meters). Enables per-location depth outputs."
        )
        self.add_file_field(
            loc_layout, 2, "Series Output Filename:", "output.series_output",
            "CSV Files (*.csv);;All Files (*)",
            "Filename for combined illumination series (default: illumination_series.csv)"
        )
        self.add_file_field(
            loc_layout, 3, "Location Data:", "output.location_data",
            "Text Files (*.txt *.dat);;All Files (*)",
            "Custom location data (optional)"
        )
        loc_group.setLayout(loc_layout)
        layout.addWidget(loc_group)

        opt_group = QGroupBox("OPTIONS")
        opt_layout = QGridLayout()
        opt_layout.setColumnStretch(1, 1)
        self.add_bool_field(
            opt_layout, 0, "Skip Map Generation", "output.no_maps",
            "Skip generating map files (faster)"
        )
        opt_group.setLayout(opt_layout)
        layout.addWidget(opt_group)

        layout.addStretch()
        self.create_scroll_area(content)

    def load_from_config(self):
        cfg = self.config_manager.get_config()
        out = cfg["output"]
        self.widgets["output.directory"].setText(out["directory"])
        i = self.widgets["output.format"].findText(out["format"])
        if i >= 0:
            self.widgets["output.format"].setCurrentIndex(i)
        j = self.widgets["output.vtk_mode"].findText(out["vtk_mode"])
        if j >= 0:
            self.widgets["output.vtk_mode"].setCurrentIndex(j)
        self.widgets["output.include_primary_ir"].setChecked(out["include_primary_ir"])
        self.widgets["output.thermal_vtk_file"].setText(out.get("thermal_vtk_file", ""))
        self.widgets["output.complete_vtk_file"].setText(out.get("complete_vtk_file", ""))
        self.widgets["output.locations_file"].setText(out["locations_file"])
        self.widgets["output.depths_file"].setText(out.get("depths_file", ""))
        self.widgets["output.series_output"].setText(out["series_output"])
        self.widgets["output.location_data"].setText(out["location_data"])
        self.widgets["output.no_maps"].setChecked(out["no_maps"])


class VisualizationTab(BaseTab):
    """Visualization configuration tab"""
    
    def init_ui(self):
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setAlignment(Qt.AlignTop)
        layout.setSpacing(15)
        
        # Enable Section
        enable_group = QGroupBox("VISUALIZATION CONTROL")
        enable_layout = QGridLayout()
        enable_layout.setColumnStretch(1, 1)
        
        self.add_bool_field(
            enable_layout, 0, "Create Visualization", "visualization.create",
            "Generate visualization/animation of results"
        )
        
        self.add_bool_field(
            enable_layout, 1, "Save Individual Frames", "visualization.save_frames",
            "Save each frame as separate image file"
        )
        
        enable_group.setLayout(enable_layout)
        layout.addWidget(enable_group)
        
        # Dimensions Section
        dim_group = QGroupBox("DIMENSIONS")
        dim_layout = QGridLayout()
        dim_layout.setColumnStretch(1, 1)
        
        self.add_int_field(
            dim_layout, 0, "Width (pixels):", "visualization.width",
            320, 7680,
            "Animation width in pixels"
        )
        
        self.add_int_field(
            dim_layout, 1, "Height (pixels):", "visualization.height",
            240, 4320,
            "Animation height in pixels"
        )
        
        self.add_int_field(
            dim_layout, 2, "FPS:", "visualization.fps",
            1, 120,
            "Frames per second"
        )
        
        dim_group.setLayout(dim_layout)
        layout.addWidget(dim_group)
        
        # Camera Section
        cam_group = QGroupBox("CAMERA POSITION")
        cam_layout = QGridLayout()
        cam_layout.setColumnStretch(1, 1)
        
        self.add_double_field(
            cam_layout, 0, "Distance:", "visualization.camera_distance",
            0.1, 1000.0, 2,
            "Camera distance from body center (radii)"
        )
        
        self.add_double_field(
            cam_layout, 1, "Latitude (°):", "visualization.camera_latitude",
            -90.0, 90.0, 2,
            "Camera latitude in degrees"
        )
        
        self.add_double_field(
            cam_layout, 2, "Longitude (°):", "visualization.camera_longitude",
            -180.0, 360.0, 2,
            "Camera longitude in degrees"
        )
        
        cam_group.setLayout(cam_layout)
        layout.addWidget(cam_group)
        
        # Alternative: Emission Angle
        emiss_group = QGroupBox("ALTERNATIVE: EMISSION ANGLE")
        emiss_layout = QGridLayout()
        emiss_layout.setColumnStretch(1, 1)
        
        self.add_double_field(
            emiss_layout, 0, "Emission Angle (°):", "visualization.emission_angle",
            -1.0, 90.0, 2,
            "Emission angle (-1 = use camera position)"
        )
        
        self.add_double_field(
            emiss_layout, 1, "Azimuth (°):", "visualization.azimuth",
            -1.0, 360.0, 2,
            "Azimuth angle (-1 = use camera position)"
        )
        
        emiss_group.setLayout(emiss_layout)
        layout.addWidget(emiss_group)
        
        # Info
        info = QLabel(
            "ℹ Use either Camera Position OR Emission Angle, not both.\n"
            "Common resolutions: 1024x768 (SD), 1920x1080 (HD), 3840x2160 (4K)"
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #88ccff; padding: 10px; font-size: 10px;")
        layout.addWidget(info)
        
        layout.addStretch()
        self.create_scroll_area(content)
        
    def load_from_config(self):
        config = self.config_manager.get_config()
        vis = config["visualization"]
        
        self.widgets["visualization.create"].setChecked(vis["create"])
        self.widgets["visualization.save_frames"].setChecked(vis["save_frames"])
        self.widgets["visualization.width"].setValue(vis["width"])
        self.widgets["visualization.height"].setValue(vis["height"])
        self.widgets["visualization.fps"].setValue(vis["fps"])
        self.widgets["visualization.camera_distance"].setValue(vis["camera_distance"])
        self.widgets["visualization.camera_latitude"].setValue(vis["camera_latitude"])
        self.widgets["visualization.camera_longitude"].setValue(vis["camera_longitude"])
        self.widgets["visualization.emission_angle"].setValue(vis["emission_angle"])
        self.widgets["visualization.azimuth"].setValue(vis["azimuth"])


class RadiationTab(BaseTab):
    """Radiation exchange configuration tab - NEW"""

    def init_ui(self):
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setAlignment(Qt.AlignTop)
        layout.setSpacing(15)

        # Enable/Disable Section
        enable_group = QGroupBox("RADIATION EXCHANGE")
        enable_layout = QGridLayout()
        enable_layout.setColumnStretch(1, 1)

        self.add_bool_field(
            enable_layout, 0, "Enable Radiation Exchange Between Faces", "radiation.enabled",
            "Compute thermal radiation exchange between mesh faces\n"
            "(Recommended for crater/PSR studies)"
        )

        self.add_bool_field(
            enable_layout, 1, "Verbose Progress Output", "radiation.verbose",
            "Print detailed progress during radiation exchange computation"
        )

        self.add_bool_field(
            enable_layout, 2, "Use GPU for Radiosity Solver", "radiation.use_gpu",
            "Use OpenCL GPU acceleration for radiosity calculations\n"
            "(Recommended when GPU is available)"
        )

        enable_group.setLayout(enable_layout)
        layout.addWidget(enable_group)

        # Face Selection Section
        face_group = QGroupBox("FACE SELECTION CRITERIA")
        face_layout = QGridLayout()
        face_layout.setColumnStretch(1, 1)

        self.add_double_field(
            face_layout, 0, "Min Shadow Fraction:", "radiation.min_shadow_fraction",
            0.0, 1.0, 2,
            "Minimum shadow fraction for face to be included in exchange\n"
            "(0.3 = faces with ≥30% shadow time)"
        )

        self.add_double_field(
            face_layout, 1, "Min Terrain View Angle:", "radiation.min_terrain_view_angle",
            0.0, 90.0, 1,
            "Minimum terrain view angle in degrees\n"
            "(Faces looking at sky more than this threshold are excluded)"
        )

        face_group.setLayout(face_layout)
        layout.addWidget(face_group)

        # Info about face selection
        face_info = QLabel(
            "ℹ Faces that are mostly sunlit or have unobstructed sky view\n"
            "   are excluded from radiation exchange for performance."
        )
        face_info.setObjectName("infoBox")
        face_info.setWordWrap(True)
        layout.addWidget(face_info)

        # Neighbor Search Section
        neighbor_group = QGroupBox("NEIGHBOR SEARCH")
        neighbor_layout = QGridLayout()
        neighbor_layout.setColumnStretch(1, 1)

        self.add_int_field(
            neighbor_layout, 0, "Max Neighbors per Face:", "radiation.max_neighbors",
            10, 10000,
            "Maximum number of neighboring faces to consider\n"
            "(2000 is good for most craters)"
        )

        self.add_double_field(
            neighbor_layout, 1, "Search Radius Factor:", "radiation.search_radius_factor",
            1.0, 100.0, 1,
            "Search radius as multiple of √(face area)"
        )

        neighbor_group.setLayout(neighbor_layout)
        layout.addWidget(neighbor_group)

        # Radiosity Solver Section
        solver_group = QGroupBox("RADIOSITY SOLVER")
        solver_layout = QGridLayout()
        solver_layout.setColumnStretch(1, 1)

        self.add_int_field(
            solver_layout, 0, "Max Iterations:", "radiation.max_iterations",
            1, 1000,
            "Maximum iterations for radiosity solver convergence"
        )

        self.add_double_field(
            solver_layout, 1, "Convergence Tolerance:", "radiation.convergence_tolerance",
            0.0001, 1.0, 4,
            "Radiosity solver convergence threshold"
        )

        solver_group.setLayout(solver_layout)
        layout.addWidget(solver_group)

        # View Factor Caching Section
        cache_group = QGroupBox("VIEW FACTOR CACHING")
        cache_layout = QGridLayout()
        cache_layout.setColumnStretch(1, 1)

        self.add_file_field(
            cache_layout, 0, "View Factor File:", "radiation.view_factor_file",
            "Binary Files (*.bin);;H-Matrix Files (*.hmat);;All Files (*)",
            "Save/load view factors to/from file\n"
            "(Avoids recomputation on subsequent runs)"
        )

        cache_group.setLayout(cache_layout)
        layout.addWidget(cache_group)

        # === COLLAPSIBLE: Thermal-Radiation Coupling (collapsed by default) ===
        self.coupling_section = CollapsibleSection("THERMAL-RADIATION COUPLING", collapsed=True)

        coupling_widget = QWidget()
        coupling_layout = QGridLayout(coupling_widget)
        coupling_layout.setColumnStretch(1, 1)

        # Add coupling parameters using BaseTab helpers but to this widget
        row = 0
        label = QLabel("Max Coupling Iterations:")
        label.setToolTip("Maximum thermal-radiation coupling iterations")
        coupling_layout.addWidget(label, row, 0)
        w = QSpinBox()
        w.setRange(1, 100)
        w.setToolTip("Maximum thermal-radiation coupling iterations\n"
                     "(Adaptive: stops early when converged)")
        w.valueChanged.connect(lambda: self.on_widget_changed("radiation.max_coupling_iterations", w.value()))
        coupling_layout.addWidget(w, row, 1)
        self.widgets["radiation.max_coupling_iterations"] = w

        row += 1
        label = QLabel("Temperature Tolerance (K):")
        label.setToolTip("Temperature change threshold for convergence")
        coupling_layout.addWidget(label, row, 0)
        w = QDoubleSpinBox()
        w.setRange(0.01, 10.0)
        w.setDecimals(2)
        w.setToolTip("Stop when max temperature change < this value")
        w.valueChanged.connect(lambda: self.on_widget_changed("radiation.coupling_temp_threshold", w.value()))
        coupling_layout.addWidget(w, row, 1)
        self.widgets["radiation.coupling_temp_threshold"] = w

        row += 1
        label = QLabel("Flux Change Tolerance (W/m²):")
        label.setToolTip("Flux change threshold for selective re-run")
        coupling_layout.addWidget(label, row, 0)
        w = QDoubleSpinBox()
        w.setRange(0.001, 10.0)
        w.setDecimals(3)
        w.setToolTip("Only re-run thermal model for faces where\n"
                     "indirect flux changed by more than this amount")
        w.valueChanged.connect(lambda: self.on_widget_changed("radiation.coupling_flux_threshold", w.value()))
        coupling_layout.addWidget(w, row, 1)
        self.widgets["radiation.coupling_flux_threshold"] = w

        row += 1
        label = QLabel("Face Fraction to Stop:")
        label.setToolTip("Fraction of faces still changing to trigger stop")
        coupling_layout.addWidget(label, row, 0)
        w = QDoubleSpinBox()
        w.setRange(0.001, 1.0)
        w.setDecimals(3)
        w.setToolTip("Stop when < this fraction of faces show changes\n"
                     "(0.01 = stop when < 1% of faces still changing)")
        w.valueChanged.connect(lambda: self.on_widget_changed("radiation.coupling_face_fraction", w.value()))
        coupling_layout.addWidget(w, row, 1)
        self.widgets["radiation.coupling_face_fraction"] = w

        self.coupling_section.add_widget(coupling_widget)
        layout.addWidget(self.coupling_section)

        # === COLLAPSIBLE: H-Matrix Compression (collapsed by default) ===
        self.hmatrix_section = CollapsibleSection("H-MATRIX COMPRESSION (Large Meshes)", collapsed=True)

        hmatrix_widget = QWidget()
        hmatrix_layout = QGridLayout(hmatrix_widget)
        hmatrix_layout.setColumnStretch(1, 1)

        row = 0
        w = QCheckBox("Enable H-Matrix Compression")
        w.setToolTip("Use hierarchical matrix compression for view factors\n"
                     "(Reduces memory 2-4× for meshes > 100K faces)")
        w.stateChanged.connect(lambda: self.on_widget_changed("radiation.use_hmatrix", w.isChecked()))
        hmatrix_layout.addWidget(w, row, 0, 1, 2)
        self.widgets["radiation.use_hmatrix"] = w

        row += 1
        label = QLabel("SVD Tolerance:")
        label.setToolTip("Truncation tolerance for low-rank approximation")
        hmatrix_layout.addWidget(label, row, 0)
        w = QDoubleSpinBox()
        w.setRange(0.0001, 0.5)
        w.setDecimals(4)
        w.setToolTip("SVD truncation tolerance (0.01 = ~1% error)")
        w.valueChanged.connect(lambda: self.on_widget_changed("radiation.hmatrix_tolerance", w.value()))
        hmatrix_layout.addWidget(w, row, 1)
        self.widgets["radiation.hmatrix_tolerance"] = w

        row += 1
        label = QLabel("Max Quadtree Depth:")
        label.setToolTip("Maximum depth of hierarchical decomposition")
        hmatrix_layout.addWidget(label, row, 0)
        w = QSpinBox()
        w.setRange(1, 20)
        w.setToolTip("Maximum quadtree depth for spatial decomposition")
        w.valueChanged.connect(lambda: self.on_widget_changed("radiation.hmatrix_max_depth", w.value()))
        hmatrix_layout.addWidget(w, row, 1)
        self.widgets["radiation.hmatrix_max_depth"] = w

        row += 1
        label = QLabel("Min Block Size:")
        label.setToolTip("Minimum block size before treating as leaf")
        hmatrix_layout.addWidget(label, row, 0)
        w = QSpinBox()
        w.setRange(1, 1000)
        w.setToolTip("Minimum number of faces per block")
        w.valueChanged.connect(lambda: self.on_widget_changed("radiation.hmatrix_min_block_size", w.value()))
        hmatrix_layout.addWidget(w, row, 1)
        self.widgets["radiation.hmatrix_min_block_size"] = w

        row += 1
        label = QLabel("Max Low-Rank:")
        label.setToolTip("Maximum rank for low-rank approximations")
        hmatrix_layout.addWidget(label, row, 0)
        w = QSpinBox()
        w.setRange(1, 500)
        w.setToolTip("Maximum rank (larger = more accurate, more memory)")
        w.valueChanged.connect(lambda: self.on_widget_changed("radiation.hmatrix_max_rank", w.value()))
        hmatrix_layout.addWidget(w, row, 1)
        self.widgets["radiation.hmatrix_max_rank"] = w

        # Warning about H-matrix
        hmatrix_warn = QLabel(
            "⚠ H-matrix requires recomputation of view factors.\n"
            "   Delete existing view factor file to use new format."
        )
        hmatrix_warn.setObjectName("radiationWarning")
        hmatrix_warn.setWordWrap(True)
        hmatrix_layout.addWidget(hmatrix_warn, row + 1, 0, 1, 2)

        self.hmatrix_section.add_widget(hmatrix_widget)
        layout.addWidget(self.hmatrix_section)

        layout.addStretch()
        self.create_scroll_area(content)

    def load_from_config(self):
        config = self.config_manager.get_config()
        rad = config.get("radiation", {})

        self.widgets["radiation.enabled"].setChecked(rad.get("enabled", False))
        self.widgets["radiation.verbose"].setChecked(rad.get("verbose", False))
        self.widgets["radiation.use_gpu"].setChecked(rad.get("use_gpu", True))
        self.widgets["radiation.min_shadow_fraction"].setValue(rad.get("min_shadow_fraction", 0.3))
        self.widgets["radiation.min_terrain_view_angle"].setValue(rad.get("min_terrain_view_angle", 5.0))
        self.widgets["radiation.max_neighbors"].setValue(rad.get("max_neighbors", 2000))
        self.widgets["radiation.search_radius_factor"].setValue(rad.get("search_radius_factor", 10.0))
        self.widgets["radiation.max_iterations"].setValue(rad.get("max_iterations", 100))
        self.widgets["radiation.convergence_tolerance"].setValue(rad.get("convergence_tolerance", 0.01))
        self.widgets["radiation.view_factor_file"].setText(rad.get("view_factor_file", ""))

        # Coupling parameters
        self.widgets["radiation.max_coupling_iterations"].setValue(rad.get("max_coupling_iterations", 10))
        self.widgets["radiation.coupling_temp_threshold"].setValue(rad.get("coupling_temp_threshold", 0.5))
        self.widgets["radiation.coupling_flux_threshold"].setValue(rad.get("coupling_flux_threshold", 0.1))
        self.widgets["radiation.coupling_face_fraction"].setValue(rad.get("coupling_face_fraction", 0.01))

        # H-matrix parameters
        self.widgets["radiation.use_hmatrix"].setChecked(rad.get("use_hmatrix", False))
        self.widgets["radiation.hmatrix_tolerance"].setValue(rad.get("hmatrix_tolerance", 0.01))
        self.widgets["radiation.hmatrix_max_depth"].setValue(rad.get("hmatrix_max_depth", 10))
        self.widgets["radiation.hmatrix_min_block_size"].setValue(rad.get("hmatrix_min_block_size", 64))
        self.widgets["radiation.hmatrix_max_rank"].setValue(rad.get("hmatrix_max_rank", 50))


class AdvancedTab(BaseTab):
    """Advanced options tab"""

    def init_ui(self):
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setAlignment(Qt.AlignTop)
        layout.setSpacing(15)

        # Debug Section
        debug_group = QGroupBox("DEBUG OPTIONS")
        debug_layout = QGridLayout()
        debug_layout.setColumnStretch(1, 1)

        self.add_bool_field(
            debug_layout, 0, "Verbose Output", "advanced.verbose",
            "Enable detailed console output for debugging"
        )

        debug_group.setLayout(debug_layout)
        layout.addWidget(debug_group)

        # Overrides Section
        override_group = QGroupBox("PARAMETER OVERRIDES")
        override_layout = QGridLayout()
        override_layout.setColumnStretch(1, 1)

        self.add_double_field(
            override_layout, 0, "Default Visible Albedo:", "advanced.default_visible_albedo",
            -1.0, 1.0, 3,
            "Override default visible albedo (-1 = use planet default)"
        )

        override_group.setLayout(override_layout)
        layout.addWidget(override_group)

        # Warning
        warning = QLabel(
            "⚠ Advanced options - modify with caution.\n"
            "Incorrect values may cause unexpected behavior or crashes."
        )
        warning.setWordWrap(True)
        warning.setStyleSheet("color: #ffaa00; padding: 10px; font-size: 10px; font-weight: bold;")
        layout.addWidget(warning)

        layout.addStretch()
        self.create_scroll_area(content)

    def load_from_config(self):
        config = self.config_manager.get_config()
        adv = config["advanced"]

        self.widgets["advanced.verbose"].setChecked(adv["verbose"])
        self.widgets["advanced.default_visible_albedo"].setValue(adv["default_visible_albedo"])

