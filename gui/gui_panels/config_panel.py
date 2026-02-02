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
from PyQt6.QtGui import QFont, QCursor

# Import tooltips - use try/except for robustness
try:
    from gui_tooltips import TOOLTIPS, get_tooltip
except ImportError:
    try:
        # Try parent directory import (when running from gui/gui_panels)
        import sys
        from pathlib import Path
        parent_dir = str(Path(__file__).parent.parent)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        from gui_tooltips import TOOLTIPS, get_tooltip
    except ImportError:
        TOOLTIPS = {}
        def get_tooltip(key):
            return ""


class InfoButton(QPushButton):
    """Small info button that displays tooltip documentation on hover"""

    def __init__(self, tooltip_key, parent=None):
        super().__init__("ⓘ", parent)
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

    def add_string_field(self, layout, row, label_text, key, tooltip="", info_key=None):
        """Add a string input field"""
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

    def add_int_field(self, layout, row, label_text, key, min_val=0, max_val=10000, tooltip="", info_key=None):
        """Add an integer input field"""
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

    def add_double_field(self, layout, row, label_text, key, min_val=0.0, max_val=10000.0,
                         decimals=2, tooltip="", info_key=None):
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
        if info_key:
            layout.addWidget(InfoButton(info_key), row, 2)
        self.widgets[key] = widget
        return widget

    def add_combo_field(self, layout, row, label_text, key, options, tooltip="", info_key=None):
        """Add a combo box field"""
        label = QLabel(label_text)
        label.setToolTip(tooltip)

        widget = QComboBox()
        widget.addItems(options)
        widget.setToolTip(tooltip)
        widget.currentTextChanged.connect(lambda: self.on_widget_changed(key, widget.currentText()))

        layout.addWidget(label, row, 0)
        layout.addWidget(widget, row, 1)
        if info_key:
            layout.addWidget(InfoButton(info_key), row, 2)
        self.widgets[key] = widget
        return widget

    def add_bool_field(self, layout, row, label_text, key, tooltip="", info_key=None):
        """Add a checkbox field"""
        widget = QCheckBox(label_text)
        widget.setToolTip(tooltip)
        widget.stateChanged.connect(lambda: self.on_widget_changed(key, widget.isChecked()))

        layout.addWidget(widget, row, 0, 1, 2)
        if info_key:
            layout.addWidget(InfoButton(info_key), row, 2)
        self.widgets[key] = widget
        return widget

    def add_file_field(self, layout, row, label_text, key, file_filter="All Files (*)", tooltip="", info_key=None):
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
        browse_btn.clicked.connect(lambda k=key: self.browse_file(widget, file_filter, k))

        h_layout.addWidget(widget)
        h_layout.addWidget(browse_btn)

        layout.addWidget(label, row, 0)
        layout.addWidget(container, row, 1)
        if info_key:
            layout.addWidget(InfoButton(info_key), row, 2)
        self.widgets[key] = widget
        return widget

    def add_directory_field(self, layout, row, label_text, key, tooltip="", info_key=None):
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
        browse_btn.clicked.connect(lambda k=key: self.browse_directory(widget, k))

        h_layout.addWidget(widget)
        h_layout.addWidget(browse_btn)

        layout.addWidget(label, row, 0)
        layout.addWidget(container, row, 1)
        if info_key:
            layout.addWidget(InfoButton(info_key), row, 2)
        self.widgets[key] = widget
        return widget

    def _get_default_browse_dir(self, key):
        """Get a logical default directory based on the config key"""
        # Map keys to default directories relative to project root
        default_dirs = {
            "output.locations_file": "data/locations",
            "output.depths_file": "data/locations",
            "mesh.import_file": "data/meshes",
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
                              "ISO 8601 format (e.g., 2025-01-01T00:00:00)",
                              info_key="simulation.start_time")
        self.add_double_field(time_layout, 1, "Duration (days):", "simulation.duration_days",
                              0.1, 3650.0, 2, "Simulation duration in Earth days",
                              info_key="simulation.duration_days")
        self.add_int_field(time_layout, 2, "Time Steps:", "simulation.timesteps",
                           1, 10000, "Number of output timesteps",
                           info_key="simulation.timesteps")

        time_group.setLayout(time_layout)
        layout.addWidget(time_group)

        # Planetary Body
        body_group = QGroupBox("Planetary Body")
        body_layout = QGridLayout()
        body_layout.setColumnStretch(1, 1)

        bodies = ["Europa", "Moon", "Mercury", "Enceladus", "Titan", "Mars", "Ceres"]
        self.add_combo_field(body_layout, 0, "Target Body:", "simulation.body", bodies,
                             "Select the planetary body to simulate",
                             info_key="simulation.body")

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
                           1, 10, "Icosphere resolution level (higher = more faces)",
                           info_key="mesh.resolution")

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
        self.add_bool_field(custom_layout, 2, "Export Basic Mesh", "mesh.export_basic",
                            "Export mesh to VTK format")

        custom_group.setLayout(custom_layout)
        layout.addWidget(custom_group)

        layout.addStretch()
        self.create_scroll_area(content)

    def load_from_config(self):
        config = self.config_manager.get_config()
        mesh = config["mesh"]

        self.widgets["mesh.resolution"].setValue(mesh["resolution"])
        self.widgets["mesh.import_file"].setText(mesh.get("import_file", ""))
        self.widgets["mesh.scale_factor"].setValue(mesh.get("scale_factor", 1.0))
        self.widgets["mesh.export_basic"].setChecked(mesh.get("export_basic", False))


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
                            "Enable thermal calculations for mesh",
                            info_key="thermal.run_thermal_mesh")
        self.add_int_field(control_layout, 1, "CPU Threads:", "thermal.threads",
                           0, 128, "Number of parallel threads (0 = auto)",
                           info_key="thermal.threads")
        self.add_int_field(control_layout, 2, "Equilibration Cycles:", "thermal.equilibration_orbits",
                           1, 100, "Number of day/night cycles for equilibration",
                           info_key="thermal.equilibration_orbits")
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

    def load_from_config(self):
        config = self.config_manager.get_config()
        thermal = config["thermal"]

        self.widgets["thermal.run_thermal_mesh"].setChecked(thermal["run_thermal_mesh"])
        self.widgets["thermal.threads"].setValue(thermal["threads"])
        self.widgets["thermal.equilibration_orbits"].setValue(thermal.get("equilibration_orbits", 1))
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
                            "Compute thermal radiation exchange between faces",
                            info_key="radiation.enabled")
        self.add_bool_field(enable_layout, 1, "Verbose Progress", "radiation.verbose",
                            "Print detailed progress")
        self.add_bool_field(enable_layout, 2, "Use GPU for Radiosity", "radiation.use_gpu",
                            "Use OpenCL GPU acceleration for radiosity solver",
                            info_key="radiation.use_gpu")

        enable_group.setLayout(enable_layout)
        layout.addWidget(enable_group)

        # Face Selection
        face_group = QGroupBox("Face Selection")
        face_layout = QGridLayout()
        face_layout.setColumnStretch(1, 1)

        self.add_double_field(face_layout, 0, "Min Shadow Fraction:", "radiation.min_shadow_fraction",
                              0.0, 1.0, 2, "Minimum shadow fraction for inclusion in radiation exchange",
                              info_key="radiation.min_shadow_fraction")

        face_group.setLayout(face_layout)
        layout.addWidget(face_group)

        # Coupling Parameters
        coupling_group = QGroupBox("Thermal-Radiation Coupling")
        coupling_layout = QGridLayout()
        coupling_layout.setColumnStretch(1, 1)

        self.add_int_field(coupling_layout, 0, "Max Coupling Iterations:", "radiation.max_coupling_iterations",
                           1, 50, "Maximum thermal-radiation coupling iterations",
                           info_key="radiation.max_coupling_iterations")
        self.add_double_field(coupling_layout, 1, "Temperature Tolerance (K):", "radiation.coupling_temp_threshold",
                              0.1, 10.0, 1, "Temperature convergence threshold",
                              info_key="radiation.coupling_temp_threshold")
        self.add_double_field(coupling_layout, 2, "Flux Tolerance (W/m²):", "radiation.coupling_flux_threshold",
                              0.01, 10.0, 2, "Flux change threshold",
                              info_key="radiation.coupling_flux_threshold")
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
                            "Save/load view factors to/from file",
                            info_key="radiation.view_factor_file")

        cache_group.setLayout(cache_layout)
        layout.addWidget(cache_group)

        # H-Matrix / ACA Compression
        compression_group = QGroupBox("View Factor Compression (Large Meshes)")
        compression_layout = QGridLayout()
        compression_layout.setColumnStretch(1, 1)

        self.add_bool_field(compression_layout, 0, "Enable H-Matrix Compression", "radiation.use_hmatrix",
                            "Use H-matrix compression for view factors. Recommended for meshes with 50K-1M faces.",
                            info_key="radiation.use_hmatrix")
        self.add_bool_field(compression_layout, 1, "Use ACA (Adaptive Cross Approximation)", "radiation.use_aca",
                            "Use ACA for faster compression (default: enabled)",
                            info_key="radiation.use_aca")
        self.add_double_field(compression_layout, 2, "ACA Tolerance:", "radiation.aca_tolerance",
                              0.0001, 0.1, 4, "ACA relative tolerance",
                              info_key="radiation.aca_tolerance")
        self.add_int_field(compression_layout, 3, "ACA Max Rank:", "radiation.aca_max_rank",
                           1, 200, "Maximum rank before falling back to sparse")
        self.add_int_field(compression_layout, 4, "H-Matrix Threads:", "radiation.hmatrix_threads",
                           0, 128, "Threads for H-matrix construction (0 = auto)")

        compression_group.setLayout(compression_layout)
        layout.addWidget(compression_group)

        # Sparse View Factors (Adaptive Neighbor Selection)
        sparse_group = QGroupBox("Sparse View Factors (Large Meshes >1M)")
        sparse_layout = QGridLayout()
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

        sparse_group.setLayout(sparse_layout)
        layout.addWidget(sparse_group)

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
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setSpacing(15)

        # Info label
        info = QLabel("Properties shown are defaults for the selected body. Uncheck to customize.")
        info.setStyleSheet("color: #64b5f6; font-size: 10px; padding: 5px;")
        layout.addWidget(info)

        # Surface Properties
        surface_group = QGroupBox("Surface Properties")
        surface_layout = QGridLayout()
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

        surface_group.setLayout(surface_layout)
        layout.addWidget(surface_group)

        # Thermal Conductivity
        cond_group = QGroupBox("Thermal Conductivity")
        cond_layout = QGridLayout()
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

        cond_group.setLayout(cond_layout)
        layout.addWidget(cond_group)

        # Density
        density_group = QGroupBox("Density")
        density_layout = QGridLayout()
        density_layout.setColumnStretch(1, 1)

        self.add_property_with_default(density_layout, 0, "Surface rhos (kg/m³):", "thermophysical.rhos",
                                       "thermophysical.use_default_rhos", 100.0, 5000.0, 1,
                                       "Surface bulk density",
                                       info_key="thermophysical.rhos")
        self.add_property_with_default(density_layout, 1, "Deep rhod (kg/m³):", "thermophysical.rhod",
                                       "thermophysical.use_default_rhod", 100.0, 5000.0, 1,
                                       "Deep bulk density (at depth >> H)",
                                       info_key="thermophysical.rhod")

        density_group.setLayout(density_layout)
        layout.addWidget(density_group)

        # Heat Capacity
        cp_group = QGroupBox("Heat Capacity")
        cp_layout = QGridLayout()
        cp_layout.setColumnStretch(1, 1)

        self.add_property_with_default(cp_layout, 0, "Reference cp0 (J/kg/K):", "thermophysical.cp0",
                                       "thermophysical.use_default_cp0", -100.0, 2000.0, 4,
                                       "Base heat capacity (may be T-dependent)",
                                       info_key="thermophysical.cp0")

        cp_group.setLayout(cp_layout)
        layout.addWidget(cp_group)

        # Internal Heat
        heat_group = QGroupBox("Internal Heat")
        heat_layout = QGridLayout()
        heat_layout.setColumnStretch(1, 1)

        self.add_property_with_default(heat_layout, 0, "Heat Flow Qb (W/m²):", "thermophysical.Qb",
                                       "thermophysical.use_default_Qb", 0.0, 1.0, 4,
                                       "Geothermal/internal heat flux at bottom boundary",
                                       info_key="thermophysical.Qb")

        heat_group.setLayout(heat_layout)
        layout.addWidget(heat_group)

        layout.addStretch()
        self.create_scroll_area(content)

    def add_property_with_default(self, layout, row, label_text, value_key, default_key,
                                   min_val, max_val, decimals, tooltip="", info_key=None):
        """Add a property field with a 'Use Default' checkbox"""
        label = QLabel(label_text)
        label.setToolTip(tooltip)
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

        if info_key:
            layout.addWidget(InfoButton(info_key), row, 3)

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

    def on_body_changed(self, body_name):
        """Update displayed defaults when body selection changes"""
        self.update_to_body_defaults(body_name)

    def update_to_body_defaults(self, body_name=None):
        """Update all fields to show body defaults"""
        if body_name is None:
            config = self.config_manager.get_config()
            body_name = config.get("simulation", {}).get("body", "Europa")

        defaults = self.BODY_DEFAULTS.get(body_name, self.BODY_DEFAULTS["Europa"])

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

        if "thermophysical.incidence_angle_albedo" in self.widgets:
            self.widgets["thermophysical.incidence_angle_albedo"].setChecked(
                thermo.get("incidence_angle_albedo", True)
            )

        self.update_to_body_defaults()


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
        layout.setSpacing(15)

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

        # Tab widget
        self.tab_widget = QTabWidget()

        # Create and add tabs
        self.tabs["simulation"] = SimulationTab(self.config_manager)
        self.tabs["mesh"] = MeshTab(self.config_manager)
        self.tabs["thermal"] = ThermalTab(self.config_manager)
        self.tabs["thermophysical"] = ThermophysicalTab(self.config_manager)
        self.tabs["radiation"] = RadiationTab(self.config_manager)
        self.tabs["illumination"] = IlluminationTab(self.config_manager)
        self.tabs["output"] = OutputTab(self.config_manager)

        self.tab_widget.addTab(self.tabs["simulation"], "Simulation")
        self.tab_widget.addTab(self.tabs["mesh"], "Mesh")
        self.tab_widget.addTab(self.tabs["thermal"], "Thermal")
        self.tab_widget.addTab(self.tabs["thermophysical"], "Properties")
        self.tab_widget.addTab(self.tabs["radiation"], "Radiation")
        self.tab_widget.addTab(self.tabs["illumination"], "Illumination")
        self.tab_widget.addTab(self.tabs["output"], "Output")

        # Connect tab change signals
        for tab in self.tabs.values():
            tab.value_changed.connect(self.on_value_changed)

        # Connect body selection to thermophysical tab
        if "simulation.body" in self.tabs["simulation"].widgets:
            self.tabs["simulation"].widgets["simulation.body"].currentTextChanged.connect(
                self.tabs["thermophysical"].on_body_changed
            )

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
