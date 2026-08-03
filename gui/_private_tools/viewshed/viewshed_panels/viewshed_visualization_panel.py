"""
Visualization Panel for Viewshed GUI.
3D mesh visualization using PyVista with scalar field colouring.
"""

import re
import struct
import sys
from pathlib import Path

import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QFileDialog, QDoubleSpinBox, QCheckBox,
    QDialog, QGroupBox, QRadioButton, QButtonGroup, QSpinBox,
    QListWidget, QListWidgetItem, QGridLayout, QLineEdit,
    QMenu, QSlider, QStackedWidget, QTabWidget, QScrollArea,
    QApplication, QProgressDialog,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QColor, QAction

# Matplotlib for 2D plots (optional)
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from viewshed_panels.viewshed_results_panel import _categorise_field, FIELD_CATEGORIES
except ImportError:
    try:
        from .viewshed_results_panel import _categorise_field, FIELD_CATEGORIES
    except ImportError:
        # Fallback definitions if import fails
        FIELD_CATEGORIES = {
            "truth": "Truth Maps",
            "per_waypoint": "Per-Waypoint",
            "aggregation": "Aggregation",
            "mask": "Custom Masks",
            "other": "Other",
        }

        def _categorise_field(name):
            """Assign a category to a field name."""
            if name.startswith("truth_"):
                return "truth"
            if name.startswith("mask_"):
                return "mask"
            if name in ("visibility_count", "illumination_hours", "temperature_mean_traverse",
                        "earth_visibility_fraction", "estimated_ice_fraction_best", "snr_best"):
                return "aggregation"
            if name.startswith("earth_visible_"):
                return "per_waypoint"
            per_wp_prefixes = (
                "visibility_", "distance_km_", "emission_angle_", "elevation_angle_",
                "temperature_", "estimated_ice_fraction_", "spatial_resolution_", "snr_",
                "solar_flux_", "direct_solar_", "illumination_",
            )
            for pfx in per_wp_prefixes:
                if name.startswith(pfx) and name != pfx.rstrip("_"):
                    return "per_waypoint"
            return "other"

# PyVista (optional)
try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False


def _read_ptm_binary(filepath):
    """Read a PTM3D binary file (uint32 header + float32 array)."""
    with open(filepath, "rb") as f:
        (n,) = struct.unpack("<I", f.read(4))
        data = np.frombuffer(f.read(n * 4), dtype=np.float32)
    return data


class _AnimationDialog(QDialog):
    """Dialog for configuring animation export (timestep sequence)."""

    def __init__(self, field_name, n_steps, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export Animation")
        self.setMinimumWidth(350)
        self.field_name = field_name
        self.n_steps = n_steps
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # Info about what will be animated
        info_label = QLabel(
            f"Animate: {self.field_name.replace('_', ' ').title()}\n"
            f"Steps: {self.n_steps}"
        )
        info_label.setStyleSheet("font-weight: bold; padding: 10px;")
        layout.addWidget(info_label)

        # Format and FPS
        settings_group = QGroupBox("Settings")
        settings_layout = QGridLayout()

        settings_layout.addWidget(QLabel("Format:"), 0, 0)
        self.format_combo = QComboBox()
        self.format_combo.addItems(["GIF (.gif)", "MP4 Video (.mp4)"])
        settings_layout.addWidget(self.format_combo, 0, 1)

        settings_layout.addWidget(QLabel("FPS:"), 1, 0)
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 30)
        self.fps_spin.setValue(5)
        settings_layout.addWidget(self.fps_spin, 1, 1)

        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)

        # Output file
        file_group = QGroupBox("Output File")
        file_layout = QHBoxLayout()
        self.file_input = QLineEdit()
        self.file_input.setPlaceholderText("Select output file...")
        browse_btn = QPushButton("Browse...")
        browse_btn.setMaximumWidth(100)
        browse_btn.clicked.connect(self._browse_file)
        file_layout.addWidget(self.file_input)
        file_layout.addWidget(browse_btn)
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        export_btn = QPushButton("Export")
        export_btn.clicked.connect(self.accept)
        export_btn.setDefault(True)
        btn_layout.addWidget(cancel_btn)
        btn_layout.addWidget(export_btn)
        layout.addLayout(btn_layout)

    def _browse_file(self):
        fmt = self.format_combo.currentText()
        ext = "gif" if "GIF" in fmt else "mp4"
        filter_str = f"{'GIF' if ext == 'gif' else 'MP4'} Files (*.{ext});;All Files (*)"
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Animation", f"viewshed_animation.{ext}", filter_str
        )
        if filename:
            self.file_input.setText(filename)

    def get_format(self):
        return "gif" if "GIF" in self.format_combo.currentText() else "mp4"

    def get_fps(self):
        return self.fps_spin.value()

    def get_filename(self):
        return self.file_input.text()


def _extract_timestep_base(field_name):
    """Extract base name and timestep from field like 'earth_visible_0018'.

    Returns (base_name, timestep_int) or (field_name, None) if no timestep.
    """
    match = re.match(r"^(.+?)_(\d{4})$", field_name)
    if match:
        return match.group(1), int(match.group(2))
    return field_name, None


class ViewshedVisualizationPanel(QWidget):
    """3D mesh visualization with scalar field colouring."""

    face_selected = pyqtSignal(int, dict)  # face_index, info dict

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pv_mesh = None
        self.mesh_data = None
        self.current_field_name = ""
        self.current_field_data = None
        self.output_dir = ""
        self.available_fields = {}  # full_name -> path

        # Unified field structure: base_name -> {"steps": [...], "path": str, "category": str}
        # For static fields: steps = []
        # For timestep fields: steps = sorted list of (timestep_int, full_field_name)
        self.base_fields = {}
        self.current_base_field = ""  # current base field name
        self.current_ts_index = 0  # current slider position

        # Fixed color scale (locked by default)
        self.lock_scale = True
        self.locked_clim = None  # (vmin, vmax) when locked

        # Waypoint metadata for UTC display
        self._waypoint_metadata = []  # list of {label, ptm_idx, mission_time_utc}

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if not PYVISTA_AVAILABLE:
            msg = QLabel(
                "PyVista not available.\n\n"
                "Install with:\n  pip install pyvista pyvistaqt"
            )
            msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
            msg.setStyleSheet("color: #888; font-size: 14px; padding: 20px;")
            layout.addWidget(msg)
            return

        # View toggle buttons at the top
        view_toggle_bar = QHBoxLayout()
        view_toggle_bar.setContentsMargins(5, 5, 5, 0)
        self.btn_3d_view = QPushButton("3D View")
        self.btn_2d_plots = QPushButton("2D Plots")
        self.btn_3d_view.setCheckable(True)
        self.btn_2d_plots.setCheckable(True)
        self.btn_3d_view.setChecked(True)
        self.btn_3d_view.setMaximumWidth(100)
        self.btn_2d_plots.setMaximumWidth(100)
        self.btn_3d_view.clicked.connect(lambda: self._switch_view("3d"))
        self.btn_2d_plots.clicked.connect(lambda: self._switch_view("2d"))
        view_toggle_bar.addWidget(self.btn_3d_view)
        view_toggle_bar.addWidget(self.btn_2d_plots)
        view_toggle_bar.addStretch()
        layout.addLayout(view_toggle_bar)

        # Stacked widget to hold 3D view and 2D plots
        self.view_stack = QStackedWidget()

        # Container for 3D view (PyVista plotter)
        self.plotter_container = QWidget()
        plotter_layout = QVBoxLayout(self.plotter_container)
        plotter_layout.setContentsMargins(0, 0, 0, 0)

        # PyVista plotter
        self.plotter = QtInteractor(self.plotter_container)
        self.plotter.set_background("#1e1e1e")
        self.plotter.add_axes()
        plotter_layout.addWidget(self.plotter.interactor, 1)

        # Controls bar (for 3D view)
        controls = QHBoxLayout()
        controls.setContentsMargins(5, 5, 5, 5)

        # Field selector: button that opens a categorised menu
        controls.addWidget(QLabel("Field:"))
        self.field_btn = QPushButton("(no fields)")
        self.field_btn.setMinimumWidth(250)
        self.field_btn.setStyleSheet(
            "text-align: left; padding: 4px 8px;"
        )
        self.field_menu = QMenu(self)
        self.field_btn.setMenu(self.field_menu)
        controls.addWidget(self.field_btn)

        # Colormap
        controls.addWidget(QLabel("  Colormap:"))
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems([
            "inferno", "viridis", "plasma", "magma", "coolwarm",
            "hot", "jet", "turbo", "RdYlBu_r", "bone",
        ])
        self.cmap_combo.setMaximumWidth(120)
        self.cmap_combo.currentTextChanged.connect(self._refresh_display)
        controls.addWidget(self.cmap_combo)

        # Clamp range
        controls.addWidget(QLabel("  Min:"))
        self.clamp_min = QDoubleSpinBox()
        self.clamp_min.setRange(-1e12, 1e12)
        self.clamp_min.setDecimals(2)
        self.clamp_min.setMaximumWidth(90)
        self.clamp_min.setSpecialValueText("auto")
        self.clamp_min.setValue(-1e12)
        controls.addWidget(self.clamp_min)

        controls.addWidget(QLabel("Max:"))
        self.clamp_max = QDoubleSpinBox()
        self.clamp_max.setRange(-1e12, 1e12)
        self.clamp_max.setDecimals(2)
        self.clamp_max.setMaximumWidth(90)
        self.clamp_max.setSpecialValueText("auto")
        self.clamp_max.setValue(1e12)
        controls.addWidget(self.clamp_max)

        apply_clamp = QPushButton("Apply")
        apply_clamp.setMaximumWidth(60)
        apply_clamp.clicked.connect(self._refresh_display)
        controls.addWidget(apply_clamp)

        auto_range = QPushButton("Auto")
        auto_range.setMaximumWidth(50)
        auto_range.clicked.connect(self._auto_range)
        controls.addWidget(auto_range)

        # Lock scale checkbox
        self.lock_scale_cb = QCheckBox("Lock")
        self.lock_scale_cb.setToolTip("Lock color scale when changing timesteps")
        self.lock_scale_cb.setChecked(True)  # Locked by default
        self.lock_scale_cb.stateChanged.connect(self._on_lock_scale_changed)
        controls.addWidget(self.lock_scale_cb)

        controls.addStretch()

        # Lighting toggle for topographic shading
        self.lighting_cb = QCheckBox("Lighting")
        self.lighting_cb.setToolTip("Add directional light for topographic shading")
        self.lighting_cb.setChecked(True)  # Lighting ON by default
        self.lighting_cb.stateChanged.connect(self._refresh_display)
        controls.addWidget(self.lighting_cb)

        # Mask color selector for faces not visible from any waypoint
        self.mask_color_btn = QPushButton("Mask")
        self.mask_color_btn.setMaximumWidth(60)
        self.mask_color_btn.setToolTip("Color for masked faces (not visible from any waypoint)")
        self.mask_color_btn.clicked.connect(self._choose_mask_color)
        self._mask_color = "#404040"  # Default dark gray
        controls.addWidget(self.mask_color_btn)

        # Show waypoints toggle
        self.show_waypoints_cb = QCheckBox("Waypoints")
        self.show_waypoints_cb.setChecked(False)
        self.show_waypoints_cb.stateChanged.connect(self._toggle_waypoints)
        controls.addWidget(self.show_waypoints_cb)

        # Waypoint sphere size control
        controls.addWidget(QLabel("Size:"))
        self.sphere_size_spin = QDoubleSpinBox()
        self.sphere_size_spin.setRange(0.1, 10.0)
        self.sphere_size_spin.setValue(1.0)
        self.sphere_size_spin.setSingleStep(0.1)
        self.sphere_size_spin.setDecimals(1)
        self.sphere_size_spin.setMaximumWidth(60)
        self.sphere_size_spin.setToolTip("Waypoint sphere size multiplier")
        self.sphere_size_spin.valueChanged.connect(self._on_sphere_size_changed)
        controls.addWidget(self.sphere_size_spin)

        # Light intensity control
        controls.addWidget(QLabel("Light:"))
        self.light_intensity_spin = QDoubleSpinBox()
        self.light_intensity_spin.setRange(0.1, 3.0)
        self.light_intensity_spin.setValue(1.0)
        self.light_intensity_spin.setSingleStep(0.1)
        self.light_intensity_spin.setDecimals(1)
        self.light_intensity_spin.setMaximumWidth(60)
        self.light_intensity_spin.setToolTip("Light intensity multiplier")
        self.light_intensity_spin.valueChanged.connect(self._refresh_display)
        controls.addWidget(self.light_intensity_spin)

        # Light angle (elevation) control - lower = more grazing, emphasizes topography
        controls.addWidget(QLabel("Angle:"))
        self.light_angle_spin = QDoubleSpinBox()
        self.light_angle_spin.setRange(5.0, 90.0)
        self.light_angle_spin.setValue(15.0)  # Low grazing angle by default
        self.light_angle_spin.setSingleStep(5.0)
        self.light_angle_spin.setDecimals(0)
        self.light_angle_spin.setMaximumWidth(50)
        self.light_angle_spin.setToolTip("Light elevation angle (deg) - lower values emphasize topography")
        self.light_angle_spin.valueChanged.connect(self._refresh_display)
        controls.addWidget(self.light_angle_spin)

        # Camera reset
        reset_btn = QPushButton("Reset Camera")
        reset_btn.setMaximumWidth(100)
        reset_btn.clicked.connect(self.reset_camera)
        controls.addWidget(reset_btn)

        plotter_layout.addLayout(controls)

        # Export bar
        export_bar = QHBoxLayout()
        export_bar.setContentsMargins(5, 0, 5, 5)

        export_img_btn = QPushButton("Export Image...")
        export_img_btn.setMaximumWidth(120)
        export_img_btn.clicked.connect(self.export_image)
        export_bar.addWidget(export_img_btn)

        export_anim_btn = QPushButton("Export Animation...")
        export_anim_btn.setMaximumWidth(140)
        export_anim_btn.clicked.connect(lambda: self.export_animation())
        export_bar.addWidget(export_anim_btn)

        export_bar.addStretch()
        plotter_layout.addLayout(export_bar)

        # Timestep slider bar (shown only for time-varying fields)
        self.ts_bar = QWidget()
        ts_layout = QHBoxLayout(self.ts_bar)
        ts_layout.setContentsMargins(5, 2, 5, 2)

        ts_layout.addWidget(QLabel("Step:"))
        self.ts_slider = QSlider(Qt.Orientation.Horizontal)
        self.ts_slider.setMinimum(0)
        self.ts_slider.setMaximum(0)
        self.ts_slider.valueChanged.connect(self._on_ts_slider_changed)
        ts_layout.addWidget(self.ts_slider, 1)

        self.ts_label = QLabel("0 / 0")
        self.ts_label.setMinimumWidth(80)
        ts_layout.addWidget(self.ts_label)

        # Play/Pause button for animation
        self.ts_play_btn = QPushButton("▶")
        self.ts_play_btn.setMaximumWidth(30)
        self.ts_play_btn.setToolTip("Play/Pause animation")
        self.ts_play_btn.clicked.connect(self._toggle_ts_animation)
        ts_layout.addWidget(self.ts_play_btn)

        self.ts_bar.setVisible(False)  # Hidden until a timestep field is selected
        plotter_layout.addWidget(self.ts_bar)

        # Animation timer
        from PyQt6.QtCore import QTimer
        self.ts_timer = QTimer(self)
        self.ts_timer.timeout.connect(self._ts_animation_step)
        self.ts_playing = False

        # Enable face picking
        self.plotter.enable_point_picking(
            callback=self._on_face_picked,
            show_message=False,
            tolerance=0.01,
        )

        # Add plotter container to stacked widget
        self.view_stack.addWidget(self.plotter_container)

        # Create 2D plot container
        self.plot_container = QWidget()
        self._init_2d_plots()
        self.view_stack.addWidget(self.plot_container)

        # Add stacked widget to main layout
        layout.addWidget(self.view_stack, 1)

        # Face info label (shared between views)
        self.face_info = QLabel("")
        self.face_info.setStyleSheet(
            "color: #aaa; font-size: 11px; padding: 2px 5px; "
            "background: #252526; border-top: 1px solid #404040;"
        )
        layout.addWidget(self.face_info)

    def load_mesh(self, mesh_path):
        """Load a PLY mesh file and display it."""
        if not PYVISTA_AVAILABLE:
            return False

        try:
            mesh_path = str(mesh_path)
            self.pv_mesh = pv.read(mesh_path)

            # Store mesh metadata
            self.mesh_data = {
                "path": mesh_path,
                "num_faces": self.pv_mesh.n_cells,
                "num_vertices": self.pv_mesh.n_points,
            }

            # Extract face centres for info display
            self._face_centers = self.pv_mesh.cell_centers().points

            # Compute mesh centroid to determine hemisphere
            centroid = self.pv_mesh.center
            self._mesh_centroid = centroid
            self._is_south_polar = centroid[2] < 0

            self.plotter.clear()
            self.plotter.add_mesh(
                self.pv_mesh,
                color="#555555",
                show_edges=False,
                name="surface",
            )

            # Set up camera based on mesh hemisphere
            if self._is_south_polar:
                # South polar mesh: Z+ points down toward pole, +X points "up" in frame
                # Camera below mesh (at lower Z) looking toward +Z direction
                cam_dist = np.linalg.norm(centroid) * 0.3
                self.plotter.camera_position = [
                    (centroid[0], centroid[1], centroid[2] - cam_dist),  # Camera below mesh
                    centroid,  # Look at centroid
                    (1, 0, 0),  # +X points up in screen
                ]
            else:
                # North polar or equatorial: default camera
                self.plotter.reset_camera()

            return True
        except Exception as e:
            self.face_info.setText(f"Error loading mesh: {e}")
            return False

    def _init_2d_plots(self):
        """Initialize the 2D plotting area with histogram and time series tabs."""
        if not MATPLOTLIB_AVAILABLE:
            fallback_layout = QVBoxLayout(self.plot_container)
            fallback_msg = QLabel(
                "Matplotlib not available.\n\n"
                "Install with:\n  pip install matplotlib"
            )
            fallback_msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
            fallback_msg.setStyleSheet("color: #888; font-size: 14px; padding: 20px;")
            fallback_layout.addWidget(fallback_msg)
            return

        plot_layout = QVBoxLayout(self.plot_container)
        plot_layout.setContentsMargins(5, 5, 5, 5)

        # Tab widget for Histogram vs Time Series
        self.plot_tabs = QTabWidget()

        # Histogram tab
        self.histogram_widget = QWidget()
        self._init_histogram_tab()
        self.plot_tabs.addTab(self.histogram_widget, "Histogram")

        # Time series tab
        self.timeseries_widget = QWidget()
        self._init_timeseries_tab()
        self.plot_tabs.addTab(self.timeseries_widget, "Time Series")

        plot_layout.addWidget(self.plot_tabs)

    def _init_histogram_tab(self):
        """Initialize the histogram tab controls and canvas."""
        layout = QVBoxLayout(self.histogram_widget)

        # Controls bar
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Field:"))
        self.hist_field_combo = QComboBox()
        self.hist_field_combo.setMinimumWidth(200)
        self.hist_field_combo.currentTextChanged.connect(self._on_hist_field_changed)
        controls.addWidget(self.hist_field_combo)

        controls.addWidget(QLabel("  Waypoint:"))
        self.hist_waypoint_spin = QSpinBox()
        self.hist_waypoint_spin.setMinimum(0)
        self.hist_waypoint_spin.setMaximum(0)
        self.hist_waypoint_spin.valueChanged.connect(self._update_histogram)
        controls.addWidget(self.hist_waypoint_spin)

        self.hist_waypoint_label = QLabel("")
        self.hist_waypoint_label.setMinimumWidth(100)
        controls.addWidget(self.hist_waypoint_label)

        controls.addWidget(QLabel("  Bins:"))
        self.hist_bins_spin = QSpinBox()
        self.hist_bins_spin.setRange(10, 200)
        self.hist_bins_spin.setValue(50)
        self.hist_bins_spin.valueChanged.connect(self._update_histogram)
        controls.addWidget(self.hist_bins_spin)

        controls.addStretch()
        layout.addLayout(controls)

        # Matplotlib figure
        self.hist_figure = Figure(figsize=(8, 5))
        self.hist_canvas = FigureCanvasQTAgg(self.hist_figure)
        layout.addWidget(self.hist_canvas, 1)

        # Stats label
        self.hist_stats_label = QLabel("")
        self.hist_stats_label.setStyleSheet("font-family: monospace; padding: 5px;")
        layout.addWidget(self.hist_stats_label)

        # Save buttons
        save_bar = QHBoxLayout()
        save_bar.addStretch()
        self.hist_save_png_btn = QPushButton("Save PNG")
        self.hist_save_png_btn.setMaximumWidth(100)
        self.hist_save_png_btn.clicked.connect(self._save_histogram_png)
        self.hist_save_data_btn = QPushButton("Save Data")
        self.hist_save_data_btn.setMaximumWidth(100)
        self.hist_save_data_btn.clicked.connect(self._save_histogram_data)
        save_bar.addWidget(self.hist_save_png_btn)
        save_bar.addWidget(self.hist_save_data_btn)
        layout.addLayout(save_bar)

        # Storage for histogram data (for export)
        self._hist_data = None

    def _init_timeseries_tab(self):
        """Initialize the time series tab controls and canvas."""
        layout = QVBoxLayout(self.timeseries_widget)

        # Top controls: position selection
        pos_bar = QHBoxLayout()
        pos_bar.addWidget(QLabel("Position:"))
        self.ts_pos_group = QButtonGroup(self)
        self.ts_rover_rb = QRadioButton("Rover")
        self.ts_lander_rb = QRadioButton("Lander")
        self.ts_both_rb = QRadioButton("Both")
        self.ts_both_rb.setChecked(True)
        self.ts_pos_group.addButton(self.ts_rover_rb)
        self.ts_pos_group.addButton(self.ts_lander_rb)
        self.ts_pos_group.addButton(self.ts_both_rb)
        for rb in [self.ts_rover_rb, self.ts_lander_rb, self.ts_both_rb]:
            rb.toggled.connect(self._update_timeseries)
            pos_bar.addWidget(rb)
        pos_bar.addStretch()
        layout.addLayout(pos_bar)

        # Field checkboxes (scrollable horizontal area)
        field_scroll = QScrollArea()
        field_scroll.setWidgetResizable(True)
        field_scroll.setMaximumHeight(80)
        field_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        field_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.ts_field_checkboxes_widget = QWidget()
        self.ts_field_checkboxes_layout = QHBoxLayout(self.ts_field_checkboxes_widget)
        self.ts_field_checkboxes_layout.setContentsMargins(5, 5, 5, 5)
        self.ts_field_checkboxes = {}

        field_scroll.setWidget(self.ts_field_checkboxes_widget)
        layout.addWidget(field_scroll)

        # Matplotlib figure
        self.ts_figure = Figure(figsize=(10, 5))
        self.ts_canvas = FigureCanvasQTAgg(self.ts_figure)
        layout.addWidget(self.ts_canvas, 1)

        # Save buttons
        save_bar = QHBoxLayout()
        save_bar.addStretch()
        self.ts_save_png_btn = QPushButton("Save PNG")
        self.ts_save_png_btn.setMaximumWidth(100)
        self.ts_save_png_btn.clicked.connect(self._save_timeseries_png)
        self.ts_save_data_btn = QPushButton("Save Data")
        self.ts_save_data_btn.setMaximumWidth(100)
        self.ts_save_data_btn.clicked.connect(self._save_timeseries_data)
        save_bar.addWidget(self.ts_save_png_btn)
        save_bar.addWidget(self.ts_save_data_btn)
        layout.addLayout(save_bar)

        # Storage for time series data (for export)
        self._ts_export_data = {'x': None, 'fields': {}}

        # Rover and lander face indices (set when output dir is loaded)
        self._rover_face_idx = None
        self._lander_face_idx = None

    def _switch_view(self, view_type):
        """Switch between 3D and 2D views."""
        if view_type == "3d":
            self.view_stack.setCurrentWidget(self.plotter_container)
            self.btn_3d_view.setChecked(True)
            self.btn_2d_plots.setChecked(False)
        else:
            self.view_stack.setCurrentWidget(self.plot_container)
            self.btn_3d_view.setChecked(False)
            self.btn_2d_plots.setChecked(True)
            # Refresh plots when switching to 2D
            self._update_histogram()
            self._update_timeseries()

    def _on_hist_field_changed(self, text):
        """Handle histogram field selection change."""
        if not text or not hasattr(self, 'base_fields'):
            return

        # Get the base name from combo data
        idx = self.hist_field_combo.currentIndex()
        base_name = self.hist_field_combo.itemData(idx)
        if base_name is None:
            base_name = text

        field_info = self.base_fields.get(base_name, {})
        steps = field_info.get("steps", [])

        # Update waypoint spinner range
        if steps:
            self.hist_waypoint_spin.setMaximum(len(steps) - 1)
            self.hist_waypoint_spin.setEnabled(True)
            # Show waypoint name if available
            waypoints = field_info.get("waypoints")
            if waypoints and len(waypoints) > 0:
                self.hist_waypoint_label.setText(waypoints[0])
            else:
                self.hist_waypoint_label.setText("")
        else:
            self.hist_waypoint_spin.setMaximum(0)
            self.hist_waypoint_spin.setEnabled(False)
            self.hist_waypoint_label.setText("")

        self._update_histogram()

    def _load_field_data_raw(self, field_name):
        """Load raw field data without NaN masking (for 2D plots)."""
        file_path = self.available_fields.get(field_name)
        if file_path is None or not Path(file_path).is_file():
            return None
        try:
            return _read_ptm_binary(file_path)
        except Exception:
            return None

    def _update_histogram(self):
        """Update histogram with current field and waypoint."""
        if not MATPLOTLIB_AVAILABLE:
            return
        if not hasattr(self, 'base_fields') or not self.base_fields:
            return
        if not hasattr(self, 'hist_field_combo'):
            return

        # Get the base name from combo data
        idx = self.hist_field_combo.currentIndex()
        if idx < 0:
            return
        base_name = self.hist_field_combo.itemData(idx)
        if not base_name:
            base_name = self.hist_field_combo.currentText()
        if not base_name:
            return

        field_info = self.base_fields.get(base_name)
        if not field_info:
            return

        # Get data for selected waypoint
        if field_info["steps"]:
            wp_idx = self.hist_waypoint_spin.value()
            if wp_idx >= len(field_info["steps"]):
                return
            _, full_name = field_info["steps"][wp_idx]
            data = self._load_field_data_raw(full_name)

            # Update waypoint label
            waypoints = field_info.get("waypoints")
            if waypoints and wp_idx < len(waypoints):
                self.hist_waypoint_label.setText(waypoints[wp_idx])
        else:
            full_name = base_name
            data = self._load_field_data_raw(base_name)

        if data is None:
            return

        # Filter to visible faces only (if mask available)
        if hasattr(self, '_visibility_mask') and self._visibility_mask is not None:
            data = data[self._visibility_mask]

        # Remove NaN and invalid values
        data = data[~np.isnan(data)]
        data = data[np.isfinite(data)]

        if len(data) == 0:
            self.hist_figure.clear()
            ax = self.hist_figure.add_subplot(111)
            ax.text(0.5, 0.5, "No valid data", ha='center', va='center', transform=ax.transAxes)
            self.hist_canvas.draw()
            self.hist_stats_label.setText("No valid data")
            self._hist_data = None
            return

        # Plot histogram
        self.hist_figure.clear()
        ax = self.hist_figure.add_subplot(111)
        counts, bins, _ = ax.hist(data, bins=self.hist_bins_spin.value(),
                                   edgecolor='black', alpha=0.7, color='steelblue')
        ax.set_xlabel(base_name.replace("_", " ").title())
        ax.set_ylabel("Face Count")
        ax.set_title(f"Distribution of {base_name.replace('_', ' ').title()}")
        ax.grid(True, alpha=0.3)
        self.hist_figure.tight_layout()
        self.hist_canvas.draw()

        # Store data for export
        self._hist_data = (counts, bins, data)

        # Update stats
        stats = (f"N = {len(data):,}  |  "
                f"Mean = {np.mean(data):.4g}  |  "
                f"Median = {np.median(data):.4g}  |  "
                f"Std = {np.std(data):.4g}  |  "
                f"Min = {np.min(data):.4g}  |  "
                f"Max = {np.max(data):.4g}")
        self.hist_stats_label.setText(stats)

    def _update_timeseries(self):
        """Update time series plot for selected fields at rover/lander."""
        if not MATPLOTLIB_AVAILABLE:
            return
        if not hasattr(self, 'base_fields') or not self.base_fields:
            return
        if not hasattr(self, 'ts_field_checkboxes'):
            return

        # Get selected fields
        selected_fields = [name for name, cb in self.ts_field_checkboxes.items()
                          if cb.isChecked()]
        if not selected_fields:
            self.ts_figure.clear()
            ax = self.ts_figure.add_subplot(111)
            ax.text(0.5, 0.5, "Select fields to plot", ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            self.ts_canvas.draw()
            return

        # Get rover/lander face indices
        rover_idx = getattr(self, '_rover_face_idx', None)
        lander_idx = getattr(self, '_lander_face_idx', None)

        show_rover = self.ts_rover_rb.isChecked() or self.ts_both_rb.isChecked()
        show_lander = self.ts_lander_rb.isChecked() or self.ts_both_rb.isChecked()

        if rover_idx is None and show_rover:
            show_rover = False
        if lander_idx is None and show_lander:
            show_lander = False

        if not show_rover and not show_lander:
            self.ts_figure.clear()
            ax = self.ts_figure.add_subplot(111)
            ax.text(0.5, 0.5, "Rover/Lander face indices not available",
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            self.ts_canvas.draw()
            return

        self.ts_figure.clear()
        ax = self.ts_figure.add_subplot(111)
        ax2 = None  # Secondary y-axis for binary/fractional fields

        # Initialize export data structure
        self._ts_export_data = {'x': None, 'utc_times': [], 'fields': {}}

        # Publication-quality color palette
        # Primary axis colors (blues/oranges for temperature/solar)
        primary_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        # Secondary axis colors (purples/greens for visibility)
        secondary_colors = ['#9467bd', '#8c564b', '#e377c2', '#17becf']

        # Fields that should use secondary axis (binary/fractional, 0-1 range)
        secondary_axis_fields = ("earth_visible", "visibility", "earth_visibility")

        # Build UTC time labels and PTM timestep mapping from waypoint metadata
        utc_labels = []
        waypoint_meta = getattr(self, '_waypoint_metadata', [])

        # Build mapping from waypoint label to (ptm_idx, utc_str)
        wp_to_ptm = {}
        for wp_meta in waypoint_meta:
            label = wp_meta.get("label", "")
            ptm_idx = wp_meta.get("ptm_idx")
            utc_str = wp_meta.get("mission_time_utc", "")
            if label and ptm_idx is not None:
                wp_to_ptm[label] = (ptm_idx, utc_str)

        # Number of waypoints determines x-axis length
        n_waypoints = len(waypoint_meta)
        x = np.arange(n_waypoints) if n_waypoints > 0 else None

        # Separate fields into primary and secondary axis groups
        primary_fields = []
        secondary_fields = []
        for field_name in selected_fields:
            if any(field_name.startswith(prefix) for prefix in secondary_axis_fields):
                secondary_fields.append(field_name)
            else:
                primary_fields.append(field_name)

        # Create secondary axis if needed
        if secondary_fields:
            ax2 = ax.twinx()

        primary_color_idx = 0
        secondary_color_idx = 0

        for field_name in selected_fields:
            field_info = self.base_fields.get(field_name)
            if not field_info or not field_info["steps"]:
                continue

            # Determine which axis to use
            use_secondary = any(field_name.startswith(prefix) for prefix in secondary_axis_fields)
            current_ax = ax2 if use_secondary and ax2 is not None else ax

            if use_secondary:
                color = secondary_colors[secondary_color_idx % len(secondary_colors)]
                secondary_color_idx += 1
            else:
                color = primary_colors[primary_color_idx % len(primary_colors)]
                primary_color_idx += 1

            rover_vals = []
            lander_vals = []
            waypoints = field_info.get("waypoints", [])
            is_per_waypoint = waypoints is not None and len(waypoints) > 0

            # Build UTC labels on first field
            if not utc_labels:
                if is_per_waypoint:
                    # Per-waypoint field: use waypoint labels to get UTC times
                    for wp_label in waypoints:
                        utc_str = ""
                        for wp_meta in waypoint_meta:
                            if wp_meta.get("label") == wp_label:
                                utc_str = wp_meta.get("mission_time_utc", "")
                                break
                        if utc_str:
                            try:
                                from datetime import datetime
                                dt = datetime.strptime(utc_str, "%Y-%m-%dT%H:%M:%S")
                                utc_labels.append(dt.strftime("%b %d\n%H:%M"))
                            except ValueError:
                                utc_labels.append(wp_label)
                        else:
                            utc_labels.append(wp_label)
                else:
                    # Timestep field: use waypoint metadata order
                    for wp_meta in waypoint_meta:
                        utc_str = wp_meta.get("mission_time_utc", "")
                        if utc_str:
                            try:
                                from datetime import datetime
                                dt = datetime.strptime(utc_str, "%Y-%m-%dT%H:%M:%S")
                                utc_labels.append(dt.strftime("%b %d\n%H:%M"))
                            except ValueError:
                                utc_labels.append(wp_meta.get("label", ""))
                        else:
                            utc_labels.append(wp_meta.get("label", ""))

            if is_per_waypoint:
                # Per-waypoint field: iterate through waypoint files
                for i, (_, full_name) in enumerate(field_info["steps"]):
                    data = self._load_field_data_raw(full_name)
                    if data is not None:
                        if rover_idx is not None and rover_idx < len(data):
                            rover_vals.append(float(data[rover_idx]))
                        else:
                            rover_vals.append(np.nan)
                        if lander_idx is not None and lander_idx < len(data):
                            lander_vals.append(float(data[lander_idx]))
                        else:
                            lander_vals.append(np.nan)
                    else:
                        rover_vals.append(np.nan)
                        lander_vals.append(np.nan)
                x = np.arange(len(waypoints))
            else:
                # Timestep field: find files matching each waypoint's PTM timestep
                steps_dict = {ts: fname for ts, fname in field_info["steps"]}
                for wp_meta in waypoint_meta:
                    ptm_idx = wp_meta.get("ptm_idx")
                    if ptm_idx is not None and ptm_idx in steps_dict:
                        data = self._load_field_data_raw(steps_dict[ptm_idx])
                        if data is not None:
                            if rover_idx is not None and rover_idx < len(data):
                                rover_vals.append(float(data[rover_idx]))
                            else:
                                rover_vals.append(np.nan)
                            if lander_idx is not None and lander_idx < len(data):
                                lander_vals.append(float(data[lander_idx]))
                            else:
                                lander_vals.append(np.nan)
                        else:
                            rover_vals.append(np.nan)
                            lander_vals.append(np.nan)
                    else:
                        rover_vals.append(np.nan)
                        lander_vals.append(np.nan)
                x = np.arange(n_waypoints)

            if x is None or len(rover_vals) == 0:
                continue

            self._ts_export_data['x'] = x
            self._ts_export_data['utc_times'] = utc_labels

            # Get display label with units
            display_label = self._get_colorbar_title(field_name + "_")
            if not display_label or display_label == field_name + "_":
                display_label = field_info.get("label", field_name.replace("_", " ").title())

            field_export = {}
            if show_rover and len(rover_vals) == len(x):
                current_ax.plot(x, rover_vals, label=f"{display_label} (Rover)",
                       linestyle='-', color=color, marker='o', markersize=5, linewidth=2)
                field_export['rover'] = rover_vals
            if show_lander and len(lander_vals) == len(x):
                current_ax.plot(x, lander_vals, label=f"{display_label} (Lander)",
                       linestyle='--', color=color, marker='s', markersize=5, linewidth=2)
                field_export['lander'] = lander_vals
            self._ts_export_data['fields'][field_name] = field_export

        # Set UTC time labels on x-axis with automatic spacing
        if x is not None and utc_labels and len(utc_labels) == len(x):
            n_points = len(x)
            # Target ~8-12 labels for readability; compute step size
            max_labels = 10
            step = max(1, (n_points + max_labels - 1) // max_labels)

            # Select tick positions and labels at regular intervals
            tick_positions = x[::step]
            tick_labels = utc_labels[::step]

            # Always include first and last point if not already included
            if len(x) > 1:
                if x[-1] not in tick_positions:
                    tick_positions = np.append(tick_positions, x[-1])
                    tick_labels = tick_labels + [utc_labels[-1]]

            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=11)
            ax.set_xlabel("UTC Time", fontsize=14, fontweight='bold')
        else:
            ax.set_xlabel("Waypoint Index", fontsize=14, fontweight='bold')

        # Publication-quality formatting for primary axis
        if primary_fields:
            # Set y-axis label based on selected primary fields
            if len(primary_fields) == 1:
                label = self._get_colorbar_title(primary_fields[0] + "_") or "Value"
                ax.set_ylabel(label, fontsize=14, fontweight='bold', color=primary_colors[0])
            else:
                ax.set_ylabel("Temperature (K) / Solar Flux (W/m²)", fontsize=14, fontweight='bold')
            ax.tick_params(axis='y', labelcolor='black', labelsize=11)
        else:
            ax.set_ylabel("Value", fontsize=14, fontweight='bold')

        # Format secondary axis if present
        if ax2 is not None and secondary_fields:
            ax2.set_ylabel("Earth Visible (0/1)", fontsize=14, fontweight='bold',
                          color=secondary_colors[0])
            ax2.set_ylim(-0.1, 1.1)  # Slightly beyond 0-1 for visibility
            ax2.tick_params(axis='y', labelcolor=secondary_colors[0], labelsize=11)

        ax.set_title("Time Series at Rover/Lander Positions", fontsize=16, fontweight='bold')

        # Combine legends from both axes if secondary axis exists
        # Use 'best' location to automatically find optimal placement
        if ax2 is not None:
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            # Create legend with automatic best placement
            leg = ax.legend(lines1 + lines2, labels1 + labels2, loc='best',
                           fontsize=11, framealpha=0.95, edgecolor='gray')
            leg.set_zorder(100)  # Ensure legend is on top
        else:
            ax.legend(loc='best', fontsize=11, framealpha=0.95, edgecolor='gray')

        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.tick_params(axis='both', which='major', labelsize=11)

        # Tight layout with extra bottom margin for rotated labels
        self.ts_figure.tight_layout()
        self.ts_figure.subplots_adjust(bottom=0.22, right=0.86 if ax2 else 0.95)
        self.ts_canvas.draw()

    def _save_histogram_png(self):
        """Save histogram figure to PNG or PDF file (publication quality)."""
        if not MATPLOTLIB_AVAILABLE or not hasattr(self, 'hist_figure'):
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Histogram", "histogram.png",
            "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg);;All Files (*)"
        )
        if path:
            # High DPI for publication quality
            self.hist_figure.savefig(path, dpi=300, bbox_inches='tight',
                                     facecolor='white', edgecolor='none')
            self.face_info.setText(f"Saved: {path}")

    def _save_histogram_data(self):
        """Save histogram data to text file."""
        if not hasattr(self, '_hist_data') or self._hist_data is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Histogram Data", "histogram_data.txt",
            "Text Files (*.txt);;CSV Files (*.csv);;All Files (*)"
        )
        if path:
            counts, bins, _ = self._hist_data
            idx = self.hist_field_combo.currentIndex()
            base_name = self.hist_field_combo.itemData(idx) or self.hist_field_combo.currentText()
            with open(path, 'w') as f:
                f.write(f"# Histogram: {base_name}\n")
                f.write(f"# Waypoint index: {self.hist_waypoint_spin.value()}\n")
                f.write(f"# {self.hist_stats_label.text()}\n")
                f.write("#\n")
                f.write("# bin_left, bin_right, count\n")
                for i in range(len(counts)):
                    f.write(f"{bins[i]:.6f}, {bins[i+1]:.6f}, {int(counts[i])}\n")
            self.face_info.setText(f"Saved: {path}")

    def _save_timeseries_png(self):
        """Save time series figure to PNG or PDF file (publication quality)."""
        if not MATPLOTLIB_AVAILABLE or not hasattr(self, 'ts_figure'):
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Time Series", "timeseries.png",
            "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg);;All Files (*)"
        )
        if path:
            # High DPI for publication quality
            self.ts_figure.savefig(path, dpi=300, bbox_inches='tight',
                                   facecolor='white', edgecolor='none')
            self.face_info.setText(f"Saved: {path}")

    def _save_timeseries_data(self):
        """Save time series data to CSV file."""
        if not hasattr(self, '_ts_export_data') or self._ts_export_data['x'] is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Time Series Data", "timeseries_data.csv",
            "CSV Files (*.csv);;Text Files (*.txt);;All Files (*)"
        )
        if path:
            with open(path, 'w') as f:
                # Header - include UTC time column if available
                utc_times = self._ts_export_data.get('utc_times', [])
                cols = ["waypoint_index"]
                if utc_times:
                    cols.append("utc_time")
                for field_name, data in self._ts_export_data.get('fields', {}).items():
                    if 'rover' in data:
                        cols.append(f"{field_name}_rover")
                    if 'lander' in data:
                        cols.append(f"{field_name}_lander")
                f.write(",".join(cols) + "\n")

                # Data rows
                x = self._ts_export_data['x']
                for i, wp_idx in enumerate(x):
                    row = [str(int(wp_idx))]
                    # Add UTC time (remove newlines from display format)
                    if utc_times and i < len(utc_times):
                        utc_str = utc_times[i].replace("\n", " ")
                        row.append(f'"{utc_str}"')
                    for field_name, data in self._ts_export_data.get('fields', {}).items():
                        if 'rover' in data and i < len(data['rover']):
                            val = data['rover'][i]
                            row.append(f"{val:.6f}" if not np.isnan(val) else "")
                        if 'lander' in data and i < len(data['lander']):
                            val = data['lander'][i]
                            row.append(f"{val:.6f}" if not np.isnan(val) else "")
                    f.write(",".join(row) + "\n")
            self.face_info.setText(f"Saved: {path}")

    def _find_face_by_coordinate(self, coord):
        """Find mesh face index closest to given coordinate."""
        if self.pv_mesh is None or not hasattr(self, '_face_centers'):
            return None
        distances = np.linalg.norm(self._face_centers - np.array(coord), axis=1)
        return int(np.argmin(distances))

    def set_output_directory(self, output_dir):
        """Scan output directory for .bin fields and build unified base_fields structure."""
        self.output_dir = output_dir
        self.available_fields = {}
        self.base_fields = {}

        output_path = Path(output_dir)
        if not output_path.is_dir():
            return

        # Per-waypoint field type prefixes for grouping
        PER_WP_PREFIXES = [
            ("visibility_", "Visibility"),
            ("distance_km_", "Distance (km)"),
            ("emission_angle_", "Emission Angle"),
            ("elevation_angle_", "Elevation Angle"),
            ("temperature_", "Temperature"),
            ("estimated_ice_fraction_", "Est. Ice Fraction"),
            ("spatial_resolution_", "Spatial Resolution"),
            ("snr_", "SNR"),
            ("earth_visible_", "Earth Visible"),
            ("direct_solar_", "Direct Solar (W/m²)"),
            ("solar_flux_", "Solar Flux (W/m²)"),
            ("illumination_", "Illumination"),
        ]

        # First pass: collect all fields
        timestep_groups = {}  # base -> [(ts, full_name, path), ...]
        static_fields = {}    # name -> path

        for p in sorted(output_path.glob("*.bin")):
            name = p.stem
            self.available_fields[name] = str(p)

            base, ts = _extract_timestep_base(name)
            if ts is not None:
                timestep_groups.setdefault(base, []).append((ts, name, str(p)))
            else:
                static_fields[name] = str(p)

        # Second pass: group static per-waypoint fields by prefix
        per_wp_groups = {}  # prefix -> [(waypoint_name, full_field_name), ...]
        remaining_static = {}

        # Suffixes that indicate aggregation fields, not per-waypoint fields
        AGGREGATION_SUFFIXES = (
            "_mean_traverse", "_best", "_fraction", "_hours", "_count",
            "_max", "_min", "_std", "_avg",
        )

        for name, path in static_fields.items():
            matched = False
            # Skip aggregation fields (they have special suffixes)
            is_aggregation = any(name.endswith(suffix) for suffix in AGGREGATION_SUFFIXES)
            if not is_aggregation:
                for prefix, label in PER_WP_PREFIXES:
                    if name.startswith(prefix) and name != prefix.rstrip("_"):
                        waypoint = name[len(prefix):]
                        # Additional check: waypoint names should look like Lander, Rover01, etc.
                        # not like "mean_traverse" or other aggregation-style names
                        if waypoint and not waypoint.startswith(("mean", "max", "min", "std", "avg")):
                            per_wp_groups.setdefault(prefix, []).append((waypoint, name))
                            matched = True
                            break
            if not matched:
                remaining_static[name] = path

        # Build base_fields structure
        # Truly static fields (aggregation, truth, etc.)
        for name, path in remaining_static.items():
            cat = _categorise_field(name)
            self.base_fields[name] = {
                "steps": [],
                "path": path,
                "category": cat,
                "label": name.replace("_", " ").title(),
            }

        # Per-waypoint static fields grouped by prefix (treated like timesteps)
        for prefix, waypoints in per_wp_groups.items():
            # Sort waypoints naturally (Lander first, then Rover01, Rover02, etc.)
            waypoints.sort(key=lambda x: (not x[0].lower().startswith("lander"), x[0]))
            base_name = prefix.rstrip("_")
            label = next((lbl for pfx, lbl in PER_WP_PREFIXES if pfx == prefix), base_name)
            self.base_fields[base_name] = {
                "steps": [(i, full_name) for i, (wp, full_name) in enumerate(waypoints)],
                "waypoints": [wp for wp, _ in waypoints],  # Store waypoint names for slider label
                "path": None,
                "category": "per_waypoint",
                "label": label,
            }

        # Timestep fields (earth_visible, etc.)
        for base, steps in timestep_groups.items():
            steps.sort(key=lambda x: x[0])
            cat = _categorise_field(base + "_0000")
            label = next((lbl for pfx, lbl in PER_WP_PREFIXES if base.startswith(pfx.rstrip("_"))),
                        base.replace("_", " ").title())
            self.base_fields[base] = {
                "steps": [(ts, full_name) for ts, full_name, _ in steps],
                "waypoints": None,  # No waypoint names, just step numbers
                "path": None,
                "category": cat,
                "label": label,
            }

        # Load visibility mask (faces visible from at least one waypoint)
        # Used to distinguish truly zero values from "not visible" faces
        vis_count_path = output_path / "visibility_count.bin"
        if vis_count_path.exists():
            vis_count = _read_ptm_binary(str(vis_count_path))
            self._visibility_mask = vis_count > 0
        else:
            self._visibility_mask = None

        # Load metadata (waypoint info, rover/lander face indices)
        import json
        metadata_path = output_path / "viewshed_metadata.json"
        self._waypoint_metadata = []  # list of {label, ptm_idx, mission_time_utc, face_idx}
        self._rover_face_idx = None
        self._lander_face_idx = None

        if metadata_path.exists():
            try:
                with open(metadata_path) as f:
                    meta = json.load(f)
                    # Load waypoint metadata for UTC display and face indices
                    self._waypoint_metadata = meta.get("waypoints", [])

                    # Extract rover/lander face indices from waypoint metadata
                    # Look for "Lander" waypoint and first "Rover" waypoint
                    for wp in self._waypoint_metadata:
                        label = wp.get("label", "")
                        face_idx = wp.get("face_idx")
                        if face_idx is not None:
                            if "Lander" in label and self._lander_face_idx is None:
                                self._lander_face_idx = face_idx
                            elif "Rover" in label and self._rover_face_idx is None:
                                self._rover_face_idx = face_idx

                    # Fallback: if no explicit Lander/Rover labels, use first and last waypoints
                    if self._waypoint_metadata and self._rover_face_idx is None:
                        # Use first waypoint as "rover" position
                        first_wp = self._waypoint_metadata[0]
                        self._rover_face_idx = first_wp.get("face_idx")
                    if self._waypoint_metadata and self._lander_face_idx is None:
                        # Use last waypoint as "lander" position
                        last_wp = self._waypoint_metadata[-1]
                        self._lander_face_idx = last_wp.get("face_idx")

            except Exception:
                self._waypoint_metadata = []

        # Fallback: infer from visibility files if metadata not available
        if self._rover_face_idx is None or self._lander_face_idx is None:
            for name, path in self.available_fields.items():
                if name.startswith("visibility_Lander") and self._lander_face_idx is None:
                    try:
                        data = _read_ptm_binary(path)
                        if np.any(data > 0):
                            self._lander_face_idx = int(np.argmax(data))
                    except Exception:
                        pass
                elif name.startswith("visibility_Rover") and self._rover_face_idx is None:
                    try:
                        data = _read_ptm_binary(path)
                        if np.any(data > 0):
                            self._rover_face_idx = int(np.argmax(data))
                    except Exception:
                        pass

        self._rebuild_field_menu()
        self._populate_2d_plot_controls()

    def _rebuild_field_menu(self):
        """Build the categorised field selection menu showing base field names."""
        self.field_menu.clear()

        if not self.base_fields:
            self.field_btn.setText("(no fields)")
            return

        # Group base fields by category
        categories = {}
        for base_name, info in self.base_fields.items():
            cat = info["category"]
            categories.setdefault(cat, []).append(base_name)

        for cat_key in FIELD_CATEGORIES:
            if cat_key not in categories:
                continue
            cat_label = FIELD_CATEGORIES[cat_key]
            fields_in_cat = categories[cat_key]
            submenu = self.field_menu.addMenu(f"{cat_label}")

            for base_name in sorted(fields_in_cat):
                info = self.base_fields[base_name]
                display = info.get("label", base_name.replace("_", " ").title())

                # Show step/waypoint count for slider-enabled fields
                if info["steps"]:
                    if info.get("waypoints"):
                        display += f" ({len(info['steps'])} waypoints)"
                    else:
                        display += f" ({len(info['steps'])} steps)"

                action = submenu.addAction(display)
                action.setData(base_name)
                action.triggered.connect(
                    lambda checked, n=base_name: self.load_base_field(n)
                )

    def _populate_2d_plot_controls(self):
        """Populate the 2D plot controls with available fields."""
        if not MATPLOTLIB_AVAILABLE:
            return

        # Populate histogram field combo
        if hasattr(self, 'hist_field_combo'):
            self.hist_field_combo.blockSignals(True)
            self.hist_field_combo.clear()
            for base_name in sorted(self.base_fields.keys()):
                info = self.base_fields[base_name]
                display = info.get("label", base_name.replace("_", " ").title())
                self.hist_field_combo.addItem(display, base_name)
            self.hist_field_combo.blockSignals(False)

        # Populate time series checkboxes (only per-waypoint fields with steps)
        if hasattr(self, 'ts_field_checkboxes'):
            # Clear existing checkboxes
            for cb in self.ts_field_checkboxes.values():
                cb.deleteLater()
            self.ts_field_checkboxes.clear()

            # Add new checkboxes for per-waypoint fields
            # Only include fields meaningful for time series at rover/lander position:
            # - temperature (varies with time)
            # - solar_flux/illumination (varies with time)
            # - earth_visible (varies with time - calculated from PTM3D primary IR)
            # Exclude: visibility, distance, angles (constant for the rover/lander's own face)
            # Note: base_name has trailing underscore stripped (e.g., "temperature" not "temperature_")
            allowed_bases = (
                "temperature", "solar_flux", "direct_solar", "illumination",
                "earth_visible",
            )
            for base_name, info in sorted(self.base_fields.items()):
                if info["steps"]:  # Only fields with timesteps/waypoints
                    # Only include fields that vary meaningfully over time
                    if base_name not in allowed_bases:
                        continue
                    display = info.get("label", base_name.replace("_", " ").title())
                    cb = QCheckBox(display)
                    cb.setProperty("field_name", base_name)
                    cb.stateChanged.connect(self._update_timeseries)
                    self.ts_field_checkboxes[base_name] = cb
                    self.ts_field_checkboxes_layout.addWidget(cb)

            # Add stretch at the end
            self.ts_field_checkboxes_layout.addStretch()

    def load_base_field(self, base_name):
        """Load a base field, setting up timestep slider if needed."""
        if not PYVISTA_AVAILABLE or self.pv_mesh is None:
            return

        if base_name not in self.base_fields:
            self.face_info.setText(f"Unknown field: {base_name}")
            return

        info = self.base_fields[base_name]
        self.current_base_field = base_name

        if info["steps"]:
            # Timestep field: enable slider, load first step
            self._setup_timestep_slider(base_name, info["steps"])
            # Apply default clim and lock it
            self._apply_default_clim(base_name)
            # Load first timestep
            _, first_field = info["steps"][0]
            self._load_field_data(first_field)
        else:
            # Static field: hide slider, load directly
            self.ts_bar.setVisible(False)
            self._apply_default_clim(base_name)
            self._load_field_data(base_name)

    def _setup_timestep_slider(self, base_name, steps):
        """Configure the timestep slider for a time-varying field."""
        n_steps = len(steps)
        self.ts_slider.blockSignals(True)
        self.ts_slider.setMinimum(0)
        self.ts_slider.setMaximum(n_steps - 1)
        self.ts_slider.setValue(0)
        self.ts_slider.blockSignals(False)

        self.current_ts_index = 0

        # Show waypoint name if available, otherwise step number
        info = self.base_fields.get(base_name, {})
        waypoints = info.get("waypoints")
        if waypoints and len(waypoints) > 0:
            self.ts_label.setText(f"{waypoints[0]} (1/{n_steps})")
        else:
            self.ts_label.setText(f"1 / {n_steps}")

        self.ts_bar.setVisible(True)

    def _apply_default_clim(self, base_name):
        """Set default color range based on field type and lock it."""
        clim = None

        # Determine default range by field type
        if base_name.startswith("visibility_") or base_name == "earth_visible":
            clim = (0, 1)
        elif base_name.startswith("distance_km_"):
            clim = (0, 50)
        elif base_name.startswith("emission_angle_"):
            clim = (0, 90)
        elif base_name.startswith("elevation_angle_"):
            clim = (-90, 90)
        elif base_name.startswith("temperature_") or base_name == "temperature_mean_traverse":
            clim = (25, 300)
        elif "fraction" in base_name or base_name.startswith("truth_") or base_name.startswith("mask_"):
            clim = (0, 1)
        elif base_name == "visibility_count":
            clim = None  # Will use auto-range
        elif base_name == "illumination_hours":
            clim = None  # Will use auto-range
        elif base_name.startswith("snr_"):
            clim = None  # Will use auto-range
        elif base_name.startswith("solar_flux_") or base_name.startswith("direct_solar_") or base_name.startswith("illumination_"):
            clim = None  # Will use auto-range (varies widely with solar angle)

        if clim:
            self.clamp_min.blockSignals(True)
            self.clamp_max.blockSignals(True)
            self.clamp_min.setValue(clim[0])
            self.clamp_max.setValue(clim[1])
            self.clamp_min.blockSignals(False)
            self.clamp_max.blockSignals(False)
            self.locked_clim = clim
            self.lock_scale = True
            self.lock_scale_cb.setChecked(True)
        else:
            # Auto-range will be applied after loading data
            self.lock_scale = False
            self.lock_scale_cb.setChecked(False)
            self.locked_clim = None

    def _load_field_data(self, field_name):
        """Load and display a specific field file on the mesh."""
        if not PYVISTA_AVAILABLE or self.pv_mesh is None:
            return

        file_path = self.available_fields.get(field_name)
        if file_path is None or not Path(file_path).is_file():
            self.face_info.setText(f"File not found for field: {field_name}")
            return

        try:
            data = _read_ptm_binary(file_path)
            if len(data) != self.pv_mesh.n_cells:
                self.face_info.setText(
                    f"Size mismatch: {len(data)} values vs {self.pv_mesh.n_cells} cells"
                )
                return

            # Apply visibility mask: faces not visible from any waypoint → NaN
            # This makes them visually distinct from faces with real zero values
            if hasattr(self, '_visibility_mask') and self._visibility_mask is not None:
                data = data.astype(np.float64)
                data[~self._visibility_mask] = np.nan

            self.current_field_name = field_name
            self.current_field_data = data
            self.pv_mesh.cell_data[field_name] = data

            # Update button label with base field name
            display = self.current_base_field.replace("_", " ").title()
            info = self.base_fields.get(self.current_base_field, {})
            if info.get("steps"):
                display += f" ({len(info['steps'])} steps)"
            self.field_btn.setText(display)

            # Apply auto-range if not locked, then refresh display
            if not self.lock_scale:
                self._auto_range()
            else:
                self._refresh_display()

        except Exception as e:
            self.face_info.setText(f"Error loading field: {e}")

    def load_field(self, field_name, file_path=None):
        """Legacy method for loading a specific field by full name."""
        # Determine base name
        base, ts = _extract_timestep_base(field_name)
        if base in self.base_fields:
            self.load_base_field(base)
            # If this was a specific timestep, navigate to it
            if ts is not None:
                info = self.base_fields[base]
                for i, (step_ts, _) in enumerate(info.get("steps", [])):
                    if step_ts == ts:
                        self.ts_slider.setValue(i)
                        break
        else:
            # Fallback: try to load directly
            self.current_base_field = field_name
            self._load_field_data(field_name)

    def _refresh_display(self):
        """Re-render the mesh with current field and colormap."""
        if not PYVISTA_AVAILABLE or self.pv_mesh is None:
            return
        if self.current_field_data is None:
            return

        cmap = self.cmap_combo.currentText()

        # Determine clamp range (use locked range if lock is enabled)
        if self.lock_scale and self.locked_clim is not None:
            clim = list(self.locked_clim)
        else:
            vmin = self.clamp_min.value()
            vmax = self.clamp_max.value()
            if vmin <= -1e11:
                vmin = None
            if vmax >= 1e11:
                vmax = None
            clim = [vmin, vmax] if vmin is not None and vmax is not None else None

        self.plotter.clear()

        # Check if lighting is enabled for topographic shading
        use_lighting = self.lighting_cb.isChecked()

        # Get mask color for NaN faces (not visible from any waypoint)
        mask_color = getattr(self, '_mask_color', '#404040')

        # Compute normals and add lights if lighting is enabled
        mesh_to_render = self.pv_mesh
        if use_lighting:
            # Compute surface normals (required for lighting calculations)
            mesh_to_render = self.pv_mesh.compute_normals(
                cell_normals=False, point_normals=True, inplace=False
            )
            # Copy the scalar data to the mesh with normals
            mesh_to_render.cell_data[self.current_field_name] = self.current_field_data

            # Get light parameters from UI controls
            light_intensity = getattr(self, 'light_intensity_spin', None)
            intensity = light_intensity.value() if light_intensity else 1.0
            light_angle = getattr(self, 'light_angle_spin', None)
            elevation_deg = light_angle.value() if light_angle else 15.0

            # Add custom directional light at specified elevation angle
            # Light comes from the side (azimuth ~45 deg) at low elevation for grazing effect
            self._add_custom_lighting(intensity, elevation_deg)

        # Get proper colorbar title with units based on base field name
        colorbar_title = self._get_colorbar_title(self.current_base_field)

        kwargs = {
            "scalars": self.current_field_name,
            "cmap": cmap,
            "nan_color": mask_color,  # Color for masked faces
            "show_edges": False,
            "name": "surface",
            "show_scalar_bar": True,
            "scalar_bar_args": {
                "title": colorbar_title,
                "color": "#e0e0e0",
                "label_font_size": 10,
                "title_font_size": 12,
            },
            "lighting": use_lighting,
            "smooth_shading": use_lighting,
        }
        if clim is not None:
            kwargs["clim"] = clim

        self.plotter.add_mesh(mesh_to_render, **kwargs)

        # Add UTC timestamp label in upper-right corner for per-waypoint fields
        self._add_timestamp_label()

    def _add_timestamp_label(self):
        """Add UTC time label in upper-right corner if waypoint metadata is available."""
        if not hasattr(self, '_waypoint_metadata') or not self._waypoint_metadata:
            return

        # Check if current field is a per-waypoint field with timesteps
        info = self.base_fields.get(self.current_base_field, {})
        steps = info.get("steps", [])
        waypoints = info.get("waypoints", [])

        if not steps or not waypoints:
            return

        # Get current waypoint index
        wp_idx = self.current_ts_index
        if wp_idx >= len(waypoints):
            return

        # Find matching metadata by waypoint label
        wp_label = waypoints[wp_idx]
        utc_str = None
        for wp_meta in self._waypoint_metadata:
            if wp_meta.get("label") == wp_label:
                utc_str = wp_meta.get("mission_time_utc", "")
                break

        if utc_str:
            # Format UTC time nicely (e.g., "2030-09-06 12:00 UTC")
            try:
                from datetime import datetime
                dt = datetime.strptime(utc_str, "%Y-%m-%dT%H:%M:%S")
                display_str = f"{dt.strftime('%Y-%m-%d %H:%M')} UTC\n{wp_label}"
            except ValueError:
                display_str = f"{utc_str} UTC\n{wp_label}"

            self.plotter.add_text(
                display_str,
                position='upper_right',
                font_size=10,
                color='white',
                name='timestamp_label'
            )

    def _auto_range(self):
        """Set clamp range to data min/max."""
        if self.current_field_data is None:
            return
        nonzero = self.current_field_data[self.current_field_data != 0]
        if len(nonzero) == 0:
            nonzero = self.current_field_data
        vmin = float(np.min(nonzero))
        vmax = float(np.max(nonzero))

        self.clamp_min.blockSignals(True)
        self.clamp_max.blockSignals(True)
        self.clamp_min.setValue(vmin)
        self.clamp_max.setValue(vmax)
        self.clamp_min.blockSignals(False)
        self.clamp_max.blockSignals(False)

        self._refresh_display()

    def _on_lock_scale_changed(self, state):
        """Handle lock scale checkbox state change."""
        self.lock_scale = (state == Qt.CheckState.Checked.value)
        if self.lock_scale:
            # Store current range as locked range
            vmin = self.clamp_min.value()
            vmax = self.clamp_max.value()
            if vmin > -1e11 and vmax < 1e11:
                self.locked_clim = (vmin, vmax)
            elif self.current_field_data is not None:
                # Use data range if spinboxes are at auto values
                self.locked_clim = (
                    float(np.min(self.current_field_data)),
                    float(np.max(self.current_field_data))
                )
        else:
            self.locked_clim = None

    def _choose_mask_color(self):
        """Open color picker for mask color."""
        from PyQt6.QtWidgets import QColorDialog
        color = QColorDialog.getColor()
        if color.isValid():
            self._mask_color = color.name()
            self._refresh_display()

    def _add_custom_lighting(self, intensity, elevation_deg):
        """Add custom directional lighting for topographic shading.

        Args:
            intensity: Light intensity multiplier (1.0 = normal)
            elevation_deg: Light elevation angle in degrees (lower = more grazing)
        """
        import math

        # Remove all existing lights
        self.plotter.renderer.RemoveAllLights()

        # Convert elevation to radians
        elev_rad = math.radians(elevation_deg)

        # Light direction: comes from upper-left at specified elevation
        # For a south-polar view (camera looking at -Z), we want light from
        # roughly the upper-left of the view
        azimuth_rad = math.radians(45)  # 45 degrees from +X toward +Y

        # Compute light direction vector (pointing toward the scene)
        # Light source is at: (cos(az)*cos(el), sin(az)*cos(el), sin(el))
        # relative to mesh centroid, so light direction is the negative
        lx = -math.cos(azimuth_rad) * math.cos(elev_rad)
        ly = -math.sin(azimuth_rad) * math.cos(elev_rad)
        lz = -math.sin(elev_rad)

        # Add main directional light
        light = pv.Light(
            position=(0, 0, 0),  # Not used for directional lights
            focal_point=(lx, ly, lz),  # Direction the light points
            color='white',
            intensity=intensity,
            light_type='headlight',  # Follows camera but we override direction
        )
        # Convert to directional light
        light.positional = False
        self.plotter.add_light(light)

        # Add subtle fill light from opposite side to prevent pure black shadows
        fill_light = pv.Light(
            position=(0, 0, 0),
            focal_point=(-lx, -ly, -lz * 0.5),  # Opposite direction, lower
            color='white',
            intensity=intensity * 0.2,  # Much dimmer
        )
        fill_light.positional = False
        self.plotter.add_light(fill_light)

        # Add subtle ambient contribution
        ambient_light = pv.Light(light_type='headlight')
        ambient_light.intensity = intensity * 0.1
        self.plotter.add_light(ambient_light)

    def _get_colorbar_title(self, base_field_name):
        """Get colorbar title with units based on field name.

        Returns a consistent label like 'Temperature (K)' instead of
        'Temperature Rover39' which changes with each waypoint.
        """
        if not base_field_name:
            return ""

        # Map field prefixes to display names with units
        field_labels = {
            "temperature_": "Temperature (K)",
            "temperature_mean_traverse": "Mean Temperature (K)",
            "distance_km_": "Distance (km)",
            "emission_angle_": "Emission Angle (deg)",
            "elevation_angle_": "Elevation Angle (deg)",
            "visibility_": "Visibility",
            "visibility_count": "Visibility Count",
            "illumination_hours": "Illumination (hours)",
            "illumination_": "Illumination (W/m²)",
            "solar_flux_": "Solar Flux (W/m²)",
            "direct_solar_": "Direct Solar (W/m²)",
            "earth_visibility_fraction": "Earth Visibility Fraction",
            "earth_visible_": "Earth Visible",
            "estimated_ice_fraction_": "Estimated Ice Fraction",
            "spatial_resolution_": "Spatial Resolution (m/px)",
            "snr_": "SNR",
            "snr_best": "SNR (Best)",
            "truth_": "Truth Map",
            "mask_": "Mask",
        }

        # Check exact matches first
        if base_field_name in field_labels:
            return field_labels[base_field_name]

        # Check prefix matches
        for prefix, label in field_labels.items():
            if prefix.endswith("_") and base_field_name.startswith(prefix):
                return label

        # Fallback: clean up field name
        return base_field_name.replace("_", " ").title()

    def _on_sphere_size_changed(self, value):
        """Handle sphere size change - refresh waypoint marker if visible."""
        if self.show_waypoints_cb.isChecked():
            self._toggle_waypoints(Qt.CheckState.Checked.value)

    def _on_ts_slider_changed(self, value):
        """Handle timestep slider change - keeps color scale fixed."""
        if not self.current_base_field:
            return

        info = self.base_fields.get(self.current_base_field, {})
        steps = info.get("steps", [])
        if not steps or value >= len(steps):
            return

        self.current_ts_index = value
        n_steps = len(steps)

        # Show waypoint name if available, otherwise step number
        waypoints = info.get("waypoints")
        if waypoints and value < len(waypoints):
            self.ts_label.setText(f"{waypoints[value]} ({value + 1}/{n_steps})")
        else:
            self.ts_label.setText(f"{value + 1} / {n_steps}")

        # Load the field for this timestep (color scale remains locked)
        _, field_name = steps[value]
        self._load_field_data(field_name)

        # Update waypoint marker if checkbox is checked
        if self.show_waypoints_cb.isChecked():
            self._toggle_waypoints(Qt.CheckState.Checked.value)

    def _toggle_ts_animation(self):
        """Toggle timestep animation playback."""
        if self.ts_playing:
            self.ts_timer.stop()
            self.ts_playing = False
            self.ts_play_btn.setText("▶")
        else:
            if not self.current_base_field:
                return
            info = self.base_fields.get(self.current_base_field, {})
            if not info.get("steps"):
                return
            self.ts_playing = True
            self.ts_play_btn.setText("⏸")
            self.ts_timer.start(500)  # 500ms per frame = 2 fps

    def _ts_animation_step(self):
        """Advance to next timestep in animation."""
        if not self.current_base_field:
            self._toggle_ts_animation()
            return

        info = self.base_fields.get(self.current_base_field, {})
        steps = info.get("steps", [])
        if not steps:
            self._toggle_ts_animation()
            return

        next_idx = (self.current_ts_index + 1) % len(steps)
        self.ts_slider.setValue(next_idx)

    def _toggle_waypoints(self, state):
        """Show/hide waypoint marker for current position on the mesh.

        Displays a semi-transparent sphere above the mesh surface at the
        current rover/lander position (based on the current slider position).
        """
        if not PYVISTA_AVAILABLE or self.plotter is None or self.pv_mesh is None:
            return

        # Remove existing waypoint marker if any
        try:
            self.plotter.remove_actor('waypoint_marker')
        except Exception:
            pass

        if state != Qt.CheckState.Checked.value:
            self.plotter.render()
            return

        # Get current waypoint position from slider
        position = self._get_current_waypoint_position()
        if position is None:
            self.face_info.setText("No waypoint position data available")
            return

        # Add sphere marker at waypoint position
        self._add_waypoint_sphere(position)
        self.plotter.render()

    def _get_current_waypoint_position(self):
        """Get 3D position for current waypoint based on slider index.

        Returns the face center position for the current waypoint, raised
        slightly above the surface.
        """
        if not hasattr(self, '_face_centers') or self._face_centers is None:
            return None

        # Check if current field has waypoints
        info = self.base_fields.get(self.current_base_field, {})
        waypoints = info.get("waypoints", [])
        steps = info.get("steps", [])

        if not waypoints or not steps:
            return None

        wp_idx = self.current_ts_index
        if wp_idx >= len(waypoints):
            return None

        wp_label = waypoints[wp_idx]

        # Try to find face index from waypoint metadata (preferred method)
        face_idx = None

        # Check metadata for face index (includes pre-computed face_idx)
        for wp_meta in self._waypoint_metadata:
            if wp_meta.get("label") == wp_label:
                face_idx = wp_meta.get("face_idx")
                if face_idx is not None:
                    break

        # If face_idx not in metadata, try using lat/lon from metadata
        if face_idx is None:
            for wp_meta in self._waypoint_metadata:
                if wp_meta.get("label") == wp_label:
                    lat = wp_meta.get("latitude")
                    lon = wp_meta.get("longitude")
                    if lat is not None and lon is not None:
                        face_idx = self._find_face_by_latlon(lat, lon)
                    break

        # Fallback: try to find from visibility data (least accurate)
        if face_idx is None:
            vis_field_name = f"visibility_{wp_label}"
            vis_path = self.available_fields.get(vis_field_name)
            if vis_path and Path(vis_path).exists():
                try:
                    vis_data = _read_ptm_binary(vis_path)
                    if np.any(vis_data > 0):
                        # Get the face with minimum distance to observer
                        # (typically the face the observer is standing on)
                        face_idx = int(np.argmax(vis_data))
                except Exception:
                    pass

        if face_idx is None or face_idx >= len(self._face_centers):
            return None

        # Get face center and compute position above surface
        face_center = self._face_centers[face_idx]

        # Compute sphere size to determine appropriate offset
        min_face_size = self._compute_min_face_size()
        half_wp_spacing = self._compute_half_waypoint_spacing()
        sphere_diameter = max(min_face_size, half_wp_spacing)

        # Ensure minimum offset
        mesh_bounds = self.pv_mesh.bounds
        mesh_diag = np.sqrt(
            (mesh_bounds[1] - mesh_bounds[0])**2 +
            (mesh_bounds[3] - mesh_bounds[2])**2 +
            (mesh_bounds[5] - mesh_bounds[4])**2
        )
        min_offset = 0.001 * mesh_diag

        # Offset above surface by sphere radius (so sphere sits on surface)
        # Direction is from mesh centroid toward face center (outward)
        centroid = np.array(self._mesh_centroid)
        outward = face_center - centroid
        outward_norm = outward / (np.linalg.norm(outward) + 1e-10)

        # Apply user size multiplier to both sphere radius and offset
        size_multiplier = getattr(self, 'sphere_size_spin', None)
        scale = size_multiplier.value() if size_multiplier else 1.0
        offset = max((sphere_diameter / 2.0) * scale, min_offset)
        position = face_center + offset * outward_norm

        return position

    def _find_face_by_latlon(self, lat, lon):
        """Find the mesh face closest to the given latitude/longitude."""
        # Try to load face latitudes/longitudes from available fields
        lat_path = self.available_fields.get("latitude")
        lon_path = self.available_fields.get("longitude")

        if lat_path and lon_path:
            try:
                face_lats = _read_ptm_binary(lat_path)
                face_lons = _read_ptm_binary(lon_path)

                # Simple angular distance
                dlat = face_lats - lat
                dlon = face_lons - lon
                # Wrap longitude difference to [-180, 180]
                dlon = np.where(dlon > 180, dlon - 360, dlon)
                dlon = np.where(dlon < -180, dlon + 360, dlon)
                dist_sq = dlat**2 + dlon**2
                return int(np.argmin(dist_sq))
            except Exception:
                pass

        return None

    def _add_waypoint_sphere(self, position):
        """Add a semi-transparent sphere marker at the given position.

        Sphere diameter is max(half waypoint spacing, smallest face size),
        scaled by the user-controlled size multiplier.
        """
        # Compute smallest mesh face size (approximate from cell areas if available)
        min_face_size = self._compute_min_face_size()

        # Compute half the distance between consecutive waypoints
        half_wp_spacing = self._compute_half_waypoint_spacing()

        # Use the larger of the two as diameter
        diameter = max(min_face_size, half_wp_spacing)
        radius = diameter / 2.0

        # Ensure minimum visible size (0.1% of mesh diagonal as absolute minimum)
        mesh_bounds = self.pv_mesh.bounds
        mesh_diag = np.sqrt(
            (mesh_bounds[1] - mesh_bounds[0])**2 +
            (mesh_bounds[3] - mesh_bounds[2])**2 +
            (mesh_bounds[5] - mesh_bounds[4])**2
        )
        min_radius = 0.001 * mesh_diag
        radius = max(radius, min_radius)

        # Apply user size multiplier
        size_multiplier = getattr(self, 'sphere_size_spin', None)
        if size_multiplier is not None:
            radius *= size_multiplier.value()

        # Create sphere
        sphere = pv.Sphere(radius=radius, center=position)

        # Add to plotter with semi-transparency
        self.plotter.add_mesh(
            sphere,
            color='yellow',
            opacity=0.7,
            name='waypoint_marker',
            pickable=False
        )

    def _compute_min_face_size(self):
        """Compute approximate size of smallest mesh face."""
        if self.pv_mesh is None:
            return 0.0

        # Try to get cell sizes from PyVista
        try:
            # Compute cell sizes (areas for surface meshes)
            sized = self.pv_mesh.compute_cell_sizes(length=False, area=True, volume=False)
            areas = sized.cell_data.get('Area')
            if areas is not None and len(areas) > 0:
                # Use sqrt of minimum area as characteristic face size
                return float(np.sqrt(np.min(areas[areas > 0])))
        except Exception:
            pass

        # Fallback: estimate from mesh bounds and face count
        mesh_bounds = self.pv_mesh.bounds
        mesh_area = (
            (mesh_bounds[1] - mesh_bounds[0]) *
            (mesh_bounds[3] - mesh_bounds[2])
        )
        n_faces = self.pv_mesh.n_cells
        if n_faces > 0:
            avg_face_area = mesh_area / n_faces
            return float(np.sqrt(avg_face_area))

        return 0.0

    def _compute_half_waypoint_spacing(self):
        """Compute half the average distance between consecutive waypoints."""
        if not self._waypoint_metadata or len(self._waypoint_metadata) < 2:
            return 0.0

        if not hasattr(self, '_face_centers') or self._face_centers is None:
            return 0.0

        # Get face positions for all waypoints
        positions = []
        for wp_meta in self._waypoint_metadata:
            face_idx = wp_meta.get("face_idx")
            if face_idx is not None and face_idx < len(self._face_centers):
                positions.append(self._face_centers[face_idx])

        if len(positions) < 2:
            return 0.0

        # Compute distances between consecutive waypoints
        distances = []
        for i in range(len(positions) - 1):
            dist = np.linalg.norm(positions[i + 1] - positions[i])
            distances.append(dist)

        if not distances:
            return 0.0

        # Return half the minimum distance (to avoid overlap)
        return float(np.min(distances) / 2.0)

    def _on_face_picked(self, point):
        """Handle face picking on the mesh."""
        if self.pv_mesh is None or not hasattr(self, "_face_centers"):
            return

        # Find nearest face to picked point
        distances = np.linalg.norm(self._face_centers - point, axis=1)
        face_id = int(np.argmin(distances))

        # Build info
        center = self._face_centers[face_id]
        info = {
            "x": center[0], "y": center[1], "z": center[2],
        }
        if self.current_field_data is not None and face_id < len(self.current_field_data):
            info["value"] = float(self.current_field_data[face_id])
            info["field"] = self.current_field_name

        # Display
        val_str = f" | {self.current_field_name} = {info.get('value', 'N/A'):.4g}" \
            if "value" in info else ""
        self.face_info.setText(
            f"Face {face_id}: ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}){val_str}"
        )

        self.face_selected.emit(face_id, info)

    def reset_camera(self):
        """Reset the camera to default view."""
        if PYVISTA_AVAILABLE and self.plotter:
            self.plotter.reset_camera()

    def export_image(self):
        """Export the current view as an image."""
        if not PYVISTA_AVAILABLE or self.plotter is None:
            return
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Image", "viewshed_visualization.png",
            "PNG Files (*.png);;JPEG Files (*.jpg);;TIFF Files (*.tiff);;All Files (*)"
        )
        if filename:
            self.plotter.screenshot(filename)

    def export_animation(self, parent=None):
        """Show animation export dialog and export timestep sequence animation.

        Uses the current field's timesteps from the slider. The camera view
        remains locked throughout the animation. Shows a modal progress dialog
        with cancel support (like PTM3D GUI).
        """
        if not PYVISTA_AVAILABLE or self.plotter is None or self.pv_mesh is None:
            self.face_info.setText("No mesh loaded for animation export")
            return

        # Check if current field has timesteps
        if not self.current_base_field:
            self.face_info.setText("Select a field first to export animation")
            return

        info = self.base_fields.get(self.current_base_field, {})
        steps = info.get("steps", [])
        if not steps:
            self.face_info.setText("Current field has no timesteps to animate")
            return

        dialog = _AnimationDialog(
            self.current_base_field,
            len(steps),
            parent=parent or self
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        fmt = dialog.get_format()
        fps = dialog.get_fps()
        filename = dialog.get_filename()
        if not filename:
            return

        n_steps = len(steps)

        # Create modal progress dialog (like PTM3D)
        progress = QProgressDialog(
            "Exporting animation...", "Cancel", 0, n_steps, parent or self
        )
        progress.setWindowTitle("Exporting Animation")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.show()

        was_cancelled = False

        def update_progress(current, total):
            nonlocal was_cancelled
            progress.setValue(current)
            progress.setLabelText(f"Rendering frame {current}/{total}...")
            QApplication.processEvents()  # CRITICAL: Allow VTK to complete render
            if progress.wasCanceled():
                was_cancelled = True

        try:
            self._export_timestep_animation(
                filename, fmt, fps, steps, progress_callback=update_progress
            )
            if was_cancelled:
                self.face_info.setText("Animation export cancelled")
            else:
                self.face_info.setText(f"Animation saved: {filename}")
        except Exception as e:
            self.face_info.setText(f"Animation export error: {e}")
            import traceback
            traceback.print_exc()
            # Try to recover the plotter (like PTM3D)
            try:
                if hasattr(self.plotter, 'mwriter') and self.plotter.mwriter:
                    try:
                        self.plotter.mwriter.close()
                    except Exception:
                        pass
                self._reinit_plotter_after_animation()
            except Exception:
                pass
        finally:
            progress.close()

    def _export_timestep_animation(self, filename, fmt, fps, steps, progress_callback=None):
        """Export animation cycling through timesteps with locked camera view.

        Uses the interactive plotter (like PTM3D) for reliable frame capture.
        Each frame updates the mesh data and re-renders in the visible window.

        Args:
            filename: Output file path
            fmt: Format ("gif" or "mp4")
            fps: Frames per second
            steps: List of (index, field_name) tuples for each frame
            progress_callback: Optional callback(current, total) for progress updates.
                               This callback should call QApplication.processEvents()
                               to allow VTK rendering to complete before frame capture.
        """
        # Store current state
        original_ts_index = self.current_ts_index
        camera_pos = self.plotter.camera_position

        # Store camera position for restoration after animation
        # (used by _reinit_plotter_after_animation)
        self._saved_camera_position = camera_pos

        # Get colormap and clim settings
        cmap = self.cmap_combo.currentText()
        if self.lock_scale and self.locked_clim is not None:
            clim = list(self.locked_clim)
        else:
            vmin = self.clamp_min.value()
            vmax = self.clamp_max.value()
            if vmin <= -1e11:
                vmin = None
            if vmax >= 1e11:
                vmax = None
            clim = [vmin, vmax] if vmin is not None and vmax is not None else None

        # Get mask color
        mask_color = getattr(self, '_mask_color', '#404040')

        n_steps = len(steps)

        # Open animation file on the INTERACTIVE plotter (like PTM3D)
        if fmt == "gif":
            self.plotter.open_gif(filename)
        else:
            self.plotter.open_movie(filename, framerate=fps)

        # Process each frame
        for i, (_, field_name) in enumerate(steps):
            # Load and display data for this frame (like PTM3D's load_timestep_data)
            # Pass waypoint_idx=i to re-add the waypoint sphere for each frame
            self._load_and_display_for_animation(field_name, cmap, clim, mask_color, waypoint_idx=i)

            # Restore camera position to ensure consistent framing
            self.plotter.camera_position = camera_pos

            # Ensure complete render before capturing frame
            self.plotter.render()

            # Write frame
            self.plotter.write_frame()

            # Update progress (callback processes Qt events to allow VTK to complete)
            self.current_ts_index = i
            if progress_callback:
                progress_callback(i + 1, n_steps)

        # Final render to stabilize state before closing
        self.plotter.render()

        # Close the animation writer directly (like PTM3D, not plotter.close())
        self.plotter.mwriter.close()

        # Reinitialize plotter after animation export (like PTM3D's _reinit_plotter)
        self._reinit_plotter_after_animation()

        # Restore original timestep
        self.ts_slider.setValue(original_ts_index)

    def _load_and_display_for_animation(self, field_name, cmap, clim, mask_color, waypoint_idx=None):
        """Load field data and update display for animation frame.

        Matches PTM3D's load_timestep_data + update_display pattern.
        Also re-adds the waypoint sphere marker if waypoint_idx is provided.
        """
        file_path = self.available_fields.get(field_name)
        if file_path is None or not Path(file_path).is_file():
            return

        data = _read_ptm_binary(file_path)
        if data is None or len(data) != self.pv_mesh.n_cells:
            return

        # Apply visibility mask
        data = np.array(data, dtype=np.float64)
        if hasattr(self, '_visibility_mask') and self._visibility_mask is not None:
            data[~self._visibility_mask] = np.nan

        # Update mesh with data (like PTM3D: self.mesh[field] = data)
        self.pv_mesh["anim_field"] = data

        # Clear and re-add mesh (like PTM3D's update_display)
        self.plotter.clear()

        # Get proper colorbar title with units
        colorbar_title = self._get_colorbar_title(self.current_base_field)

        kwargs = {
            "scalars": "anim_field",
            "cmap": cmap,
            "nan_color": mask_color,
            "show_edges": False,
            "name": "anim_surface",
            "show_scalar_bar": True,
            "scalar_bar_args": {
                "title": colorbar_title,
                "color": "#e0e0e0",
                "label_font_size": 10,
                "title_font_size": 12,
            },
            "lighting": False,
        }
        if clim is not None:
            kwargs["clim"] = clim

        self.plotter.add_mesh(self.pv_mesh, **kwargs)

        # Re-add waypoint sphere marker and UTC timestamp for this frame
        if waypoint_idx is not None:
            # Temporarily set current_ts_index to get the correct waypoint position and timestamp
            old_ts_idx = self.current_ts_index
            self.current_ts_index = waypoint_idx
            position = self._get_current_waypoint_position()
            if position is not None:
                self._add_waypoint_sphere(position)
            # Add UTC timestamp label for this frame
            self._add_timestamp_label()
            self.current_ts_index = old_ts_idx

    def _reinit_plotter_after_animation(self):
        """Reinitialize plotter after animation export.

        Completely recreates the QtInteractor widget (like PTM3D's _reinit_plotter).
        This is critical for proper cleanup after using open_gif/open_movie.
        """
        if not PYVISTA_AVAILABLE:
            return

        # Get the layout from plotter container
        plotter_layout = self.plotter_container.layout()
        if plotter_layout is None:
            return

        # Remove old plotter widget from layout
        try:
            plotter_layout.removeWidget(self.plotter.interactor)
        except Exception:
            pass

        # Create new plotter
        self.plotter = QtInteractor(self.plotter_container)
        self.plotter.set_background("#1e1e1e")
        self.plotter.add_axes()

        # Add back to layout (at position 0, before controls)
        plotter_layout.insertWidget(0, self.plotter.interactor, 1)

        # Re-enable face picking
        self.plotter.enable_point_picking(
            callback=self._on_face_picked,
            show_message=False,
            tolerance=0.01,
        )

        # Clean up animation scalar from mesh
        if "anim_field" in self.pv_mesh.cell_data:
            del self.pv_mesh.cell_data["anim_field"]

        # Restore mesh display with current field
        if self.pv_mesh is not None and self.current_field_data is not None:
            self._refresh_display()

        # Restore camera position (saved before animation started)
        if hasattr(self, '_saved_camera_position') and self._saved_camera_position is not None:
            self.plotter.camera_position = self._saved_camera_position
            self._saved_camera_position = None  # Clear after use

    def refresh(self):
        """Refresh the visualization (re-scan output directory)."""
        if self.output_dir:
            self.set_output_directory(self.output_dir)
        if self.current_field_name:
            self.load_field(self.current_field_name)
