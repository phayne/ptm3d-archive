"""
Configuration Panel for PTM3D GUI
Tabbed interface for simulation parameter configuration
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QScrollArea,
    QGroupBox, QGridLayout, QLabel, QLineEdit, QSpinBox, QDoubleSpinBox,
    QComboBox, QCheckBox, QPushButton, QFileDialog, QTextEdit, QSplitter
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont


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

    def create_scroll_area(self, content_widget):
        """Wrap content in a scroll area"""
        scroll = QScrollArea()
        scroll.setWidget(content_widget)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(scroll)

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
        layout.addWidget(container, row, 1)
        self.widgets[key] = widget
        return widget

    def add_directory_field(self, layout, row, label_text, key, tooltip=""):
        """Add a directory selection field"""
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
        browse_btn.clicked.connect(lambda: self.browse_directory(widget))

        h_layout.addWidget(widget)
        h_layout.addWidget(browse_btn)

        layout.addWidget(label, row, 0)
        layout.addWidget(container, row, 1)
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

    def browse_directory(self, line_edit):
        """Open directory selection dialog"""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Directory",
            line_edit.text() if line_edit.text() else ".",
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

    def init_ui(self):
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setSpacing(15)

        # Time Parameters
        time_group = QGroupBox("Time Parameters")
        time_layout = QGridLayout()
        time_layout.setColumnStretch(1, 1)

        self.add_string_field(time_layout, 0, "Start Time:", "simulation.start_time",
                              "ISO 8601 format (e.g., 2025-01-01T00:00:00)")
        self.add_double_field(time_layout, 1, "Duration (days):", "simulation.duration_days",
                              0.1, 3650.0, 2, "Simulation duration in Earth days")
        self.add_int_field(time_layout, 2, "Time Steps:", "simulation.timesteps",
                           1, 10000, "Number of output timesteps")

        time_group.setLayout(time_layout)
        layout.addWidget(time_group)

        # Planetary Body
        body_group = QGroupBox("Planetary Body")
        body_layout = QGridLayout()
        body_layout.setColumnStretch(1, 1)

        bodies = ["Europa", "Moon", "Mercury"]
        self.add_combo_field(body_layout, 0, "Target Body:", "simulation.body", bodies,
                             "Select the planetary body to simulate")

        body_group.setLayout(body_layout)
        layout.addWidget(body_group)

        layout.addStretch()
        self.create_scroll_area(content)

    def load_from_config(self):
        config = self.config_manager.get_config()
        sim = config["simulation"]

        self.widgets["simulation.start_time"].setText(sim["start_time"])
        self.widgets["simulation.duration_days"].setValue(sim["duration_days"])
        self.widgets["simulation.timesteps"].setValue(sim["timesteps"])

        idx = self.widgets["simulation.body"].findText(sim["body"])
        if idx >= 0:
            self.widgets["simulation.body"].setCurrentIndex(idx)


class MeshTab(BaseConfigTab):
    """Mesh configuration tab"""

    def init_ui(self):
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setSpacing(15)

        # Global Mesh
        global_group = QGroupBox("Global Mesh")
        global_layout = QGridLayout()
        global_layout.setColumnStretch(1, 1)

        self.add_int_field(global_layout, 0, "Resolution:", "mesh.resolution",
                           1, 10, "Icosphere resolution level (higher = more faces)")

        global_group.setLayout(global_layout)
        layout.addWidget(global_group)

        # Custom Mesh
        custom_group = QGroupBox("Custom Mesh Import")
        custom_layout = QGridLayout()
        custom_layout.setColumnStretch(1, 1)

        self.add_file_field(custom_layout, 0, "Import File:", "mesh.import_file",
                            "Mesh Files (*.obj *.ply *.vtk);;All Files (*)",
                            "Import custom triangulated mesh")
        self.add_double_field(custom_layout, 1, "Scale Factor:", "mesh.scale_factor",
                              0.001, 1000.0, 3, "Scaling factor for imported mesh")
        self.add_bool_field(custom_layout, 2, "Export Basic Mesh", "mesh.export_basic",
                            "Export mesh to VTK format")

        custom_group.setLayout(custom_layout)
        layout.addWidget(custom_group)

        # Topographic Mesh
        topo_group = QGroupBox("Topographic Mesh")
        topo_layout = QGridLayout()
        topo_layout.setColumnStretch(1, 1)

        self.add_bool_field(topo_layout, 0, "Use Topographic Mesh", "mesh.use_topographic",
                            "Use pre-computed topographic meshes")
        self.add_directory_field(topo_layout, 1, "Topo Directory:", "mesh.topo_directory",
                                 "Directory with topographic meshes")

        topo_group.setLayout(topo_layout)
        layout.addWidget(topo_group)

        layout.addStretch()
        self.create_scroll_area(content)

    def load_from_config(self):
        config = self.config_manager.get_config()
        mesh = config["mesh"]

        self.widgets["mesh.resolution"].setValue(mesh["resolution"])
        self.widgets["mesh.import_file"].setText(mesh["import_file"])
        self.widgets["mesh.scale_factor"].setValue(mesh["scale_factor"])
        self.widgets["mesh.export_basic"].setChecked(mesh["export_basic"])
        self.widgets["mesh.use_topographic"].setChecked(mesh["use_topographic"])
        self.widgets["mesh.topo_directory"].setText(mesh["topo_directory"])


class ThermalTab(BaseConfigTab):
    """Thermal model configuration tab"""

    def init_ui(self):
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setSpacing(15)

        # Model Control
        control_group = QGroupBox("Model Control")
        control_layout = QGridLayout()
        control_layout.setColumnStretch(1, 1)

        self.add_bool_field(control_layout, 0, "Run Thermal Model", "thermal.run_thermal_mesh",
                            "Enable thermal calculations for mesh")
        self.add_int_field(control_layout, 1, "CPU Threads:", "thermal.threads",
                           0, 128, "Number of parallel threads (0 = auto)")
        self.add_int_field(control_layout, 2, "Equilibration Orbits:", "thermal.equilibration_orbits",
                           1, 100, "Number of day/night cycles for equilibration")
        self.add_bool_field(control_layout, 3, "Compute Statistics", "thermal.compute_statistics",
                            "Calculate min/max/mean temperature statistics")

        control_group.setLayout(control_layout)
        layout.addWidget(control_group)

        # GPU Acceleration
        gpu_group = QGroupBox("GPU Acceleration")
        gpu_layout = QGridLayout()
        gpu_layout.setColumnStretch(1, 1)

        self.add_bool_field(gpu_layout, 0, "Use GPU for Thermal Solver", "thermal.use_gpu",
                            "Use OpenCL GPU acceleration for thermal solver")

        gpu_group.setLayout(gpu_layout)
        layout.addWidget(gpu_group)


        # Integration Window
        integ_group = QGroupBox("Integration Window")
        integ_layout = QGridLayout()
        integ_layout.setColumnStretch(1, 1)

        self.add_double_field(integ_layout, 0, "Integration Days:", "thermal.integration_days",
                              -1.0, 10000.0, 2, "Integration window in days (-1 = auto)")
        self.add_int_field(integ_layout, 1, "Integration Steps:", "thermal.integration_steps",
                           -1, 100000, "Integration steps override (-1 = auto)")

        integ_group.setLayout(integ_layout)
        layout.addWidget(integ_group)

        # Property Maps
        maps_group = QGroupBox("Thermal Property Maps")
        maps_layout = QGridLayout()
        maps_layout.setColumnStretch(1, 1)

        self.add_file_field(maps_layout, 0, "Thermal Inertia Map:", "thermal.thermal_inertia_map",
                            "FITS Files (*.fits *.fit);;All Files (*)",
                            "Path to thermal inertia map")

        maps_group.setLayout(maps_layout)
        layout.addWidget(maps_group)

        layout.addStretch()
        self.create_scroll_area(content)

    def load_from_config(self):
        config = self.config_manager.get_config()
        thermal = config["thermal"]

        self.widgets["thermal.run_thermal_mesh"].setChecked(thermal["run_thermal_mesh"])
        self.widgets["thermal.threads"].setValue(thermal["threads"])
        self.widgets["thermal.equilibration_orbits"].setValue(thermal.get("equilibration_orbits", 10))
        self.widgets["thermal.compute_statistics"].setChecked(thermal["compute_statistics"])
        self.widgets["thermal.use_gpu"].setChecked(thermal.get("use_gpu", False))

        self.widgets["thermal.integration_days"].setValue(thermal["integration_days"])
        self.widgets["thermal.integration_steps"].setValue(thermal["integration_steps"])
        self.widgets["thermal.thermal_inertia_map"].setText(thermal["thermal_inertia_map"])


class RadiationTab(BaseConfigTab):
    """Radiation exchange configuration tab"""

    def init_ui(self):
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setSpacing(15)

        # Enable/Disable
        enable_group = QGroupBox("Radiation Exchange")
        enable_layout = QGridLayout()
        enable_layout.setColumnStretch(1, 1)

        self.add_bool_field(enable_layout, 0, "Enable Radiation Exchange", "radiation.enabled",
                            "Compute thermal radiation exchange between faces")
        self.add_bool_field(enable_layout, 1, "Verbose Progress", "radiation.verbose",
                            "Print detailed progress")
        self.add_bool_field(enable_layout, 2, "Use GPU for Radiosity", "radiation.use_gpu",
                            "Use OpenCL GPU acceleration for radiosity solver")

        enable_group.setLayout(enable_layout)
        layout.addWidget(enable_group)

        # Face Selection
        face_group = QGroupBox("Face Selection")
        face_layout = QGridLayout()
        face_layout.setColumnStretch(1, 1)

        self.add_double_field(face_layout, 0, "Min Shadow Fraction:", "radiation.min_shadow_fraction",
                              0.0, 1.0, 2, "Minimum shadow fraction for inclusion in radiation exchange")

        face_group.setLayout(face_layout)
        layout.addWidget(face_group)

        # Coupling Parameters
        coupling_group = QGroupBox("Thermal-Radiation Coupling")
        coupling_layout = QGridLayout()
        coupling_layout.setColumnStretch(1, 1)

        self.add_int_field(coupling_layout, 0, "Max Coupling Iterations:", "radiation.max_coupling_iterations",
                           1, 50, "Maximum thermal-radiation coupling iterations")
        self.add_double_field(coupling_layout, 1, "Temperature Tolerance (K):", "radiation.coupling_temp_threshold",
                              0.1, 10.0, 1, "Temperature convergence threshold")
        self.add_double_field(coupling_layout, 2, "Flux Tolerance (W/m²):", "radiation.coupling_flux_threshold",
                              0.01, 10.0, 2, "Flux change threshold")
        self.add_double_field(coupling_layout, 3, "Face Fraction:", "radiation.coupling_face_fraction",
                              0.001, 0.1, 3, "Fraction of faces still changing to stop")

        coupling_group.setLayout(coupling_layout)
        layout.addWidget(coupling_group)

        # Caching
        cache_group = QGroupBox("View Factor Caching")
        cache_layout = QGridLayout()
        cache_layout.setColumnStretch(1, 1)

        self.add_file_field(cache_layout, 0, "View Factor File:", "radiation.view_factor_file",
                            "Binary Files (*.bin);;H-Matrix Files (*.hmat);;All Files (*)",
                            "Save/load view factors to/from file")

        cache_group.setLayout(cache_layout)
        layout.addWidget(cache_group)

        # H-Matrix / ACA Compression
        compression_group = QGroupBox("View Factor Compression (Large Meshes)")
        compression_layout = QGridLayout()
        compression_layout.setColumnStretch(1, 1)

        self.add_bool_field(compression_layout, 0, "Enable H-Matrix Compression", "radiation.use_hmatrix",
                            "Use hierarchical matrix compression (recommended for >50K faces)")
        self.add_bool_field(compression_layout, 1, "Use ACA (Adaptive Cross Approximation)", "radiation.use_aca",
                            "Use ACA for faster compression (default: enabled)")
        self.add_double_field(compression_layout, 2, "ACA Tolerance:", "radiation.aca_tolerance",
                              0.0001, 0.1, 4, "ACA relative tolerance")
        self.add_int_field(compression_layout, 3, "ACA Max Rank:", "radiation.aca_max_rank",
                           1, 200, "Maximum rank before falling back to sparse")

        compression_group.setLayout(compression_layout)
        layout.addWidget(compression_group)

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
        self.widgets["radiation.coupling_temp_threshold"].setValue(rad.get("coupling_temp_threshold", 2.0))
        self.widgets["radiation.coupling_flux_threshold"].setValue(rad.get("coupling_flux_threshold", 0.1))
        self.widgets["radiation.coupling_face_fraction"].setValue(rad.get("coupling_face_fraction", 0.01))
        self.widgets["radiation.view_factor_file"].setText(rad.get("view_factor_file", ""))
        self.widgets["radiation.use_hmatrix"].setChecked(rad.get("use_hmatrix", False))
        self.widgets["radiation.use_aca"].setChecked(rad.get("use_aca", True))
        self.widgets["radiation.aca_tolerance"].setValue(rad.get("aca_tolerance", 0.001))
        self.widgets["radiation.aca_max_rank"].setValue(rad.get("aca_max_rank", 50))


class IlluminationTab(BaseConfigTab):
    """Illumination parameters tab"""

    def init_ui(self):
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setSpacing(15)

        # Solar Illumination
        illum_group = QGroupBox("Solar Illumination")
        illum_layout = QGridLayout()
        illum_layout.setColumnStretch(1, 1)

        self.add_bool_field(illum_layout, 0, "Include Primary Reflection", "illumination.include_primary_reflection",
                            "Include reflected light from primary body")
        self.add_bool_field(illum_layout, 1, "Use Batch Processing", "illumination.use_batch_processing",
                            "Use batch processing for ray tracing")

        illum_group.setLayout(illum_layout)
        layout.addWidget(illum_group)

        # Albedo
        albedo_group = QGroupBox("Albedo Configuration")
        albedo_layout = QGridLayout()
        albedo_layout.setColumnStretch(1, 1)

        self.add_file_field(albedo_layout, 0, "Albedo Map:", "illumination.albedo_map",
                            "FITS Files (*.fits *.fit);;All Files (*)",
                            "Path to single albedo map file")
        self.add_directory_field(albedo_layout, 1, "Albedo Maps Directory:", "illumination.albedo_maps_directory",
                                 "Directory containing multiple albedo maps")

        albedo_group.setLayout(albedo_layout)
        layout.addWidget(albedo_group)

        # Positions
        pos_group = QGroupBox("Precomputed Positions")
        pos_layout = QGridLayout()
        pos_layout.setColumnStretch(1, 1)

        self.add_file_field(pos_layout, 0, "Positions File:", "illumination.positions_file",
                            "Binary Files (*.bin);;All Files (*)",
                            "Precomputed Sun/Primary positions")

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
        layout.setSpacing(15)

        # Output Destination
        dest_group = QGroupBox("Output Destination")
        dest_layout = QGridLayout()
        dest_layout.setColumnStretch(1, 1)

        self.add_directory_field(dest_layout, 0, "Output Directory:", "output.directory",
                                 "Directory where results are written")
        self.add_combo_field(dest_layout, 1, "Format:", "output.format",
                             ["binary", "vtk"], "Output format")
        self.add_combo_field(dest_layout, 2, "VTK Mode:", "output.vtk_mode",
                             ["thermal", "complete", "both"], "VTK export mode")
        self.add_bool_field(dest_layout, 3, "Include Primary IR", "output.include_primary_ir",
                            "Include primary IR in VTK export")

        dest_group.setLayout(dest_layout)
        layout.addWidget(dest_group)

        # Location Exports
        loc_group = QGroupBox("Location Exports")
        loc_layout = QGridLayout()
        loc_layout.setColumnStretch(1, 1)

        self.add_file_field(loc_layout, 0, "Locations File:", "output.locations_file",
                            "CSV Files (*.csv);;Text Files (*.txt);;All Files (*)",
                            "CSV/TXT with lat,lon locations")
        self.add_file_field(loc_layout, 1, "Depths File:", "output.depths_file",
                            "Text Files (*.txt *.dat);;All Files (*)",
                            "Depths for location output (one per line, meters)")
        self.add_string_field(loc_layout, 2, "Series Output:", "output.series_output",
                              "Filename for illumination series CSV")

        loc_group.setLayout(loc_layout)
        layout.addWidget(loc_group)

        # Options
        opt_group = QGroupBox("Options")
        opt_layout = QGridLayout()
        opt_layout.setColumnStretch(1, 1)

        self.add_bool_field(opt_layout, 0, "Skip Map Generation", "output.no_maps",
                            "Skip generating map files")

        opt_group.setLayout(opt_layout)
        layout.addWidget(opt_group)

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

        self.widgets["output.include_primary_ir"].setChecked(out["include_primary_ir"])
        self.widgets["output.locations_file"].setText(out["locations_file"])
        self.widgets["output.depths_file"].setText(out.get("depths_file", ""))
        self.widgets["output.series_output"].setText(out["series_output"])
        self.widgets["output.no_maps"].setChecked(out["no_maps"])


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

        # Tab widget
        self.tab_widget = QTabWidget()

        # Create and add tabs
        self.tabs["simulation"] = SimulationTab(self.config_manager)
        self.tabs["mesh"] = MeshTab(self.config_manager)
        self.tabs["thermal"] = ThermalTab(self.config_manager)
        self.tabs["radiation"] = RadiationTab(self.config_manager)
        self.tabs["illumination"] = IlluminationTab(self.config_manager)
        self.tabs["output"] = OutputTab(self.config_manager)

        self.tab_widget.addTab(self.tabs["simulation"], "Simulation")
        self.tab_widget.addTab(self.tabs["mesh"], "Mesh")
        self.tab_widget.addTab(self.tabs["thermal"], "Thermal")
        self.tab_widget.addTab(self.tabs["radiation"], "Radiation")
        self.tab_widget.addTab(self.tabs["illumination"], "Illumination")
        self.tab_widget.addTab(self.tabs["output"], "Output")

        # Connect tab change signals
        for tab in self.tabs.values():
            tab.value_changed.connect(self.on_value_changed)

        splitter.addWidget(self.tab_widget)

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

    def load_from_config(self):
        """Load all tabs from config"""
        for tab in self.tabs.values():
            tab.load_from_config()
        self.update_preview()
