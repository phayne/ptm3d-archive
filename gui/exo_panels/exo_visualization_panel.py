"""
Visualization Panel for Exosphere GUI.
3D mesh visualization (PyVista) and 2D plotting (Matplotlib).
"""

import sys
import csv
from pathlib import Path

import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QSplitter,
    QLabel, QSlider, QPushButton, QComboBox, QGroupBox, QGridLayout,
    QDoubleSpinBox, QCheckBox, QFrame, QLineEdit, QFileDialog,
    QMessageBox, QApplication
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QFont

# Import data loader and tooltips
try:
    from exo_core.exo_data_loader import (
        ExoDataCache, load_ply_mesh, discover_fields, detect_mode,
        load_timestep_times, compute_epoch_times, load_atmospheric_mass,
        find_nearest_face, read_binary_field
    )
    from exo_core.exo_tooltips import FIELD_INFO, FIELD_CATEGORIES, get_field_info
except ImportError:
    parent_dir = str(Path(__file__).parent.parent)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from exo_core.exo_data_loader import (
        ExoDataCache, load_ply_mesh, discover_fields, detect_mode,
        load_timestep_times, compute_epoch_times, load_atmospheric_mass,
        find_nearest_face, read_binary_field
    )
    from exo_core.exo_tooltips import FIELD_INFO, FIELD_CATEGORIES, get_field_info


# PyVista
try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False

# Matplotlib
try:
    import matplotlib
    matplotlib.use("QtAgg")
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# =========================================================================
# 3D Visualization Widget
# =========================================================================

class ExoPyVistaWidget(QWidget):
    """3D mesh visualization with scalar coloring and color scale controls."""

    face_selected = pyqtSignal(int, dict)  # face_index, info dict

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pv_mesh = None
        self.mesh_data = None
        self.current_data = None
        self.current_field = "temperature_K"
        self.data_cache = ExoDataCache(max_entries=200)

        self.current_source = "exosphere"  # "ptm3d" or "exosphere"
        self.ptm_dir = ""
        self.exo_dir = ""
        self.fields = {}
        # Mapping from slider position to actual timestep file indices
        self._timestep_indices = []

        self._last_slider_position = 0  # Track last slider position for field-change re-render

        self.selected_face_id = None
        self.picking_enabled = False
        self.picking_active = False
        self.hover_face_id = None
        self.hover_highlight_actor = None

        # Color scale state
        self.fixed_color_scale = True
        self.global_data_range = None
        self.color_range_min = None
        self.color_range_max = None
        self._has_loaded_data = False

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if not PYVISTA_AVAILABLE:
            layout.addWidget(QLabel(
                "PyVista not available. Install with: pip install pyvista pyvistaqt"))
            return

        self.plotter = QtInteractor(self)
        self.plotter.set_background("#1e1e1e")
        self.plotter.add_axes()
        layout.addWidget(self.plotter.interactor, 1)

        # --- Controls bar ---
        controls = QHBoxLayout()
        controls.setContentsMargins(5, 5, 5, 5)

        # Source toggle
        controls.addWidget(QLabel("Source:"))
        self.source_combo = QComboBox()
        self.source_combo.addItems(["Exosphere Output", "PTM3D Input"])
        self.source_combo.setMaximumWidth(160)
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)
        controls.addWidget(self.source_combo)

        # Field selector
        controls.addWidget(QLabel("  Field:"))
        self.field_combo = QComboBox()
        self.field_combo.setMinimumWidth(180)
        self.field_combo.currentIndexChanged.connect(self._on_field_combo_changed)
        controls.addWidget(self.field_combo)

        # Camera presets
        controls.addWidget(QLabel("  Camera:"))
        self.camera_combo = QComboBox()
        self.camera_combo.addItems(["Global", "North Pole", "South Pole",
                                     "Trailing Hemisphere", "Leading Hemisphere"])
        self.camera_combo.setMaximumWidth(160)
        self.camera_combo.currentTextChanged.connect(self._set_camera_preset)
        controls.addWidget(self.camera_combo)

        # Pick mode
        self.pick_btn = QPushButton("Select Face")
        self.pick_btn.setCheckable(True)
        self.pick_btn.setMaximumWidth(110)
        self.pick_btn.setStyleSheet("""
            QPushButton {
                background-color: #404040; color: white;
                border: 1px solid #606060; border-radius: 4px; padding: 4px 8px;
            }
            QPushButton:checked {
                background-color: #ff9800; color: black;
                border: 1px solid #ffb74d;
            }
            QPushButton:hover { background-color: #505050; }
            QPushButton:checked:hover { background-color: #ffa726; }
        """)
        self.pick_btn.toggled.connect(self._toggle_picking)
        controls.addWidget(self.pick_btn)

        self.pick_status_label = QLabel("")
        self.pick_status_label.setStyleSheet("color: #888888; font-size: 11px;")
        controls.addWidget(self.pick_status_label)

        controls.addStretch()

        # --- Color scale controls ---

        # Colormap selector
        controls.addWidget(QLabel("Colormap:"))
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems([
            "plasma", "inferno", "viridis", "magma", "cividis",
            "coolwarm", "RdBu_r", "Reds", "Blues", "turbo",
        ])
        self.cmap_combo.setMaximumWidth(100)
        self.cmap_combo.currentTextChanged.connect(self._on_cmap_changed)
        controls.addWidget(self.cmap_combo)

        # Fixed scale toggle
        self.fixed_scale_check = QCheckBox("Fixed Scale")
        self.fixed_scale_check.setChecked(True)
        self.fixed_scale_check.setStyleSheet("color: #e0e0e0; font-size: 11px;")
        self.fixed_scale_check.stateChanged.connect(self._on_fixed_scale_changed)
        controls.addWidget(self.fixed_scale_check)

        # Min input
        self.scale_min_label = QLabel("Min:")
        self.scale_min_label.setStyleSheet("color: #888; font-size: 11px;")
        self.scale_min_label.setVisible(False)
        controls.addWidget(self.scale_min_label)

        self.scale_min_input = QDoubleSpinBox()
        self.scale_min_input.setRange(-1e12, 1e12)
        self.scale_min_input.setDecimals(4)
        self.scale_min_input.setValue(0)
        self.scale_min_input.setMaximumWidth(90)
        self.scale_min_input.setVisible(False)
        self.scale_min_input.valueChanged.connect(self._on_color_range_changed)
        controls.addWidget(self.scale_min_input)

        # Max input
        self.scale_max_label = QLabel("Max:")
        self.scale_max_label.setStyleSheet("color: #888; font-size: 11px;")
        self.scale_max_label.setVisible(False)
        controls.addWidget(self.scale_max_label)

        self.scale_max_input = QDoubleSpinBox()
        self.scale_max_input.setRange(-1e12, 1e12)
        self.scale_max_input.setDecimals(4)
        self.scale_max_input.setValue(400)
        self.scale_max_input.setMaximumWidth(90)
        self.scale_max_input.setVisible(False)
        self.scale_max_input.valueChanged.connect(self._on_color_range_changed)
        controls.addWidget(self.scale_max_input)

        ctrl_widget = QWidget()
        ctrl_widget.setLayout(controls)
        layout.insertWidget(0, ctrl_widget)

    # -----------------------------------------------------------------
    # Mesh loading
    # -----------------------------------------------------------------

    def load_mesh(self, mesh_path):
        """Load PLY mesh and create PyVista surface."""
        if not PYVISTA_AVAILABLE:
            return False

        self.mesh_data = load_ply_mesh(mesh_path)
        if self.mesh_data is None:
            return False

        vertices = self.mesh_data["vertices"]
        faces = self.mesh_data["faces"]

        n_faces = len(faces)
        pv_faces = np.column_stack([
            np.full(n_faces, 3, dtype=np.int64),
            faces
        ]).ravel()

        self.pv_mesh = pv.PolyData(vertices, pv_faces)

        # Store lat/lon as cell data for face picking info
        self.pv_mesh["latitude"] = self.mesh_data["face_latitudes"]
        self.pv_mesh["longitude"] = self.mesh_data["face_longitudes"]

        self._has_loaded_data = False
        self.global_data_range = None

        # Initial display (no scalars yet)
        self.plotter.clear()
        self.plotter.add_mesh(
            self.pv_mesh,
            show_edges=False,
            color="#4488aa",
            lighting=True,
            smooth_shading=True,
            name="main_mesh",
        )
        self.plotter.reset_camera()
        return True

    # -----------------------------------------------------------------
    # Directory and field management
    # -----------------------------------------------------------------

    def set_directories(self, ptm_dir="", exo_dir=""):
        """Set data directories and discover fields."""
        if ptm_dir:
            self.ptm_dir = ptm_dir
        if exo_dir:
            self.exo_dir = exo_dir
        self._refresh_fields()

    def _refresh_fields(self):
        """Refresh available fields for current source."""
        source_dir = self.exo_dir if self.current_source == "exosphere" else self.ptm_dir
        if not source_dir:
            return
        self.fields = discover_fields(source_dir)

        # Build timestep index list from first available field
        self._timestep_indices = []
        if self.fields:
            first_field_ts = next(iter(self.fields.values()))
            self._timestep_indices = [ts_idx for ts_idx, _ in first_field_ts]

        self.global_data_range = None

        self.field_combo.blockSignals(True)
        self.field_combo.clear()
        for name in sorted(self.fields.keys()):
            info = get_field_info(name)
            label = info["label"]
            units = info["units"]
            display = f"{label} ({units})" if units else label
            self.field_combo.addItem(display, name)
        self.field_combo.blockSignals(False)

        # Select a default field and set matching colormap.
        # NOTE: We must explicitly set current_field here because Qt's addItem()
        # auto-selects the first item (index 0), so setCurrentIndex(0) is a no-op
        # and currentIndexChanged does NOT fire.
        if self.field_combo.count() > 0:
            self.field_combo.setCurrentIndex(0)
            self.current_field = self.field_combo.itemData(0)
            self._sync_colormap_to_field()

    def _on_source_changed(self, index):
        self.current_source = "exosphere" if index == 0 else "ptm3d"
        self._refresh_fields()
        # Re-render with new source at current slider position
        if self._has_loaded_data:
            self.update_display(self._last_slider_position)

    def _on_field_combo_changed(self, index):
        if index >= 0:
            field_name = self.field_combo.itemData(index)
            if field_name:
                self.current_field = field_name
                self.global_data_range = None  # reset range for new field
                self._sync_colormap_to_field()
                # Re-render with new field at current slider position
                if self._has_loaded_data:
                    self.update_display(self._last_slider_position)

    def _sync_colormap_to_field(self):
        """Set colormap combo to match the field's default colormap."""
        info = get_field_info(self.current_field)
        cmap = info.get("colormap", "plasma")
        idx = self.cmap_combo.findText(cmap)
        if idx >= 0:
            self.cmap_combo.blockSignals(True)
            self.cmap_combo.setCurrentIndex(idx)
            self.cmap_combo.blockSignals(False)

    def get_timestep_indices(self):
        """Return the list of actual timestep indices (for slider mapping)."""
        return self._timestep_indices

    # -----------------------------------------------------------------
    # Color scale controls
    # -----------------------------------------------------------------

    def _on_cmap_changed(self, text):
        if self._has_loaded_data:
            self._update_mesh_display()

    def _on_fixed_scale_changed(self, state):
        self.fixed_color_scale = (state == Qt.CheckState.Checked.value)
        show_manual = not self.fixed_color_scale
        self.scale_min_label.setVisible(show_manual)
        self.scale_min_input.setVisible(show_manual)
        self.scale_max_label.setVisible(show_manual)
        self.scale_max_input.setVisible(show_manual)

        if show_manual and self.current_data is not None:
            # Populate with current data range
            self.scale_min_input.blockSignals(True)
            self.scale_max_input.blockSignals(True)
            self.scale_min_input.setValue(float(np.nanmin(self.current_data)))
            self.scale_max_input.setValue(float(np.nanmax(self.current_data)))
            self.scale_min_input.blockSignals(False)
            self.scale_max_input.blockSignals(False)

        if self._has_loaded_data:
            self._update_mesh_display()

    def _on_color_range_changed(self):
        if not self.fixed_color_scale:
            self.color_range_min = self.scale_min_input.value()
            self.color_range_max = self.scale_max_input.value()
            if self._has_loaded_data:
                self._update_mesh_display()

    def _compute_global_range(self):
        """Compute min/max across sampled timesteps for current field."""
        source_dir = self.exo_dir if self.current_source == "exosphere" else self.ptm_dir
        if not source_dir or self.current_field not in self.fields:
            return None

        timesteps = self.fields[self.current_field]
        global_min = float("inf")
        global_max = float("-inf")

        # Sample every Nth file for performance
        step = max(1, len(timesteps) // 10)
        sampled = timesteps[::step]

        for _, filepath in sampled:
            data = self.data_cache.get_field(filepath)
            if data is not None and len(data) > 0:
                global_min = min(global_min, float(np.nanmin(data)))
                global_max = max(global_max, float(np.nanmax(data)))

        if global_min < float("inf") and global_max > float("-inf"):
            return (global_min, global_max)
        return None

    # -----------------------------------------------------------------
    # Display update
    # -----------------------------------------------------------------

    def update_display(self, slider_position):
        """Update 3D display for a given slider position.

        Maps slider_position (0, 1, 2, ...) to actual timestep file index
        using _timestep_indices.
        """
        if not PYVISTA_AVAILABLE or self.pv_mesh is None:
            return

        self._last_slider_position = slider_position

        source_dir = self.exo_dir if self.current_source == "exosphere" else self.ptm_dir
        if not source_dir or self.current_field not in self.fields:
            return

        # Map slider position to actual file timestep index
        timesteps = self.fields[self.current_field]
        if slider_position < 0 or slider_position >= len(timesteps):
            return
        actual_ts_idx, filepath = timesteps[slider_position]

        data = self.data_cache.get_field(filepath)
        if data is None:
            return

        self.current_data = data

        # Add scalar to mesh
        if len(data) == self.pv_mesh.n_cells:
            self.pv_mesh.cell_data[self.current_field] = data
        elif len(data) == self.pv_mesh.n_points:
            self.pv_mesh.point_data[self.current_field] = data
        else:
            return

        self._has_loaded_data = True
        self._update_mesh_display()

    def _update_mesh_display(self):
        """Re-render the mesh with current scalars, preserving camera."""
        if not PYVISTA_AVAILABLE or self.pv_mesh is None:
            return

        # Save camera position
        camera_position = self.plotter.camera_position

        self.plotter.clear()

        # Determine clim
        clim = None
        if self.fixed_color_scale:
            if self.global_data_range is None:
                self.global_data_range = self._compute_global_range()
            clim = self.global_data_range
        else:
            if self.color_range_min is not None and self.color_range_max is not None:
                clim = (self.color_range_min, self.color_range_max)

        # If no clim, use current data range
        if clim is None and self.current_data is not None:
            clim = (float(np.nanmin(self.current_data)),
                    float(np.nanmax(self.current_data)))

        # Determine colormap
        cmap = self.cmap_combo.currentText() if hasattr(self, "cmap_combo") else "plasma"

        # Field label for colorbar
        info = get_field_info(self.current_field)
        title = f"{info['label']} ({info['units']})" if info["units"] else info["label"]

        # Determine whether we have scalars to show
        scalars = None
        if self.current_field in self.pv_mesh.cell_data:
            scalars = self.current_field
        elif self.current_field in self.pv_mesh.point_data:
            scalars = self.current_field
        mesh_kwargs = {
            "scalars": scalars,
            "cmap": cmap,
            "show_edges": False,
            "lighting": True,
            "smooth_shading": True,
            "pickable": True,
            "name": "main_mesh",
            "show_scalar_bar": scalars is not None,
            "scalar_bar_args": {
                "title": title,
                "color": "white",
                "title_font_size": 12,
                "label_font_size": 10,
                "vertical": False,
                "position_x": 0.2,
                "position_y": 0.02,
                "width": 0.6,
                "height": 0.06,
                "n_labels": 5,
                "fmt": "%.4g",
            },
        }
        if clim is not None and scalars is not None:
            mesh_kwargs["clim"] = clim

        self.plotter.add_mesh(self.pv_mesh, **mesh_kwargs)

        # Re-add selection highlight
        if self.selected_face_id is not None:
            self._update_selection_highlight(self.selected_face_id)

        # Restore camera
        if self._has_loaded_data and camera_position is not None:
            self.plotter.camera_position = camera_position
        else:
            self.plotter.reset_camera()

    # -----------------------------------------------------------------
    # Camera presets
    # -----------------------------------------------------------------

    def _set_camera_preset(self, preset_name):
        if not PYVISTA_AVAILABLE or self.pv_mesh is None:
            return

        bounds = self.pv_mesh.bounds
        center = self.pv_mesh.center
        max_dim = max(bounds[1] - bounds[0], bounds[3] - bounds[2],
                      bounds[5] - bounds[4])
        dist = max_dim * 2.0

        if preset_name == "Global":
            self.plotter.reset_camera()
            self.plotter.view_isometric()
        elif preset_name == "North Pole":
            self.plotter.camera_position = [
                (center[0], center[1], center[2] + dist),
                center, (0, 1, 0)]
        elif preset_name == "South Pole":
            self.plotter.camera_position = [
                (center[0], center[1], center[2] - dist),
                center, (0, 1, 0)]
        elif preset_name == "Trailing Hemisphere":
            self.plotter.camera_position = [
                (center[0], center[1] + dist, center[2]),
                center, (0, 0, 1)]
        elif preset_name == "Leading Hemisphere":
            self.plotter.camera_position = [
                (center[0], center[1] - dist, center[2]),
                center, (0, 0, 1)]

    # -----------------------------------------------------------------
    # Face picking
    # -----------------------------------------------------------------

    def _toggle_picking(self, enabled):
        if not PYVISTA_AVAILABLE or self.pv_mesh is None:
            self.pick_btn.setChecked(False)
            return

        self.picking_enabled = enabled
        if enabled:
            self.picking_active = True
            self.plotter.enable_cell_picking(
                callback=self._on_face_picked,
                show_message=False,
                color="yellow",
                style="wireframe",
            )
            self.pick_status_label.setText("Click a face to select")
        else:
            self.picking_active = False
            try:
                self.plotter.disable_picking()
            except Exception:
                pass
            self.pick_status_label.setText("")

    def _on_face_picked(self, picked):
        if picked is None:
            return
        cell_id = -1
        if hasattr(picked, "cell_data") and "vtkOriginalCellIds" in picked.cell_data:
            ids = picked.cell_data["vtkOriginalCellIds"]
            if len(ids) > 0:
                cell_id = int(ids[0])
        elif isinstance(picked, dict):
            cell_id = picked.get("id", -1)

        if cell_id < 0 or self.mesh_data is None:
            return

        self.selected_face_id = cell_id
        info = {
            "face_id": cell_id,
            "latitude": float(self.mesh_data["face_latitudes"][cell_id]),
            "longitude": float(self.mesh_data["face_longitudes"][cell_id]),
        }
        if self.current_data is not None and cell_id < len(self.current_data):
            info["value"] = float(self.current_data[cell_id])
            info["field"] = self.current_field

        lat = info["latitude"]
        lon = info["longitude"]
        val_str = f", {info.get('value', 0):.4g}" if "value" in info else ""
        self.pick_status_label.setText(
            f"Face {cell_id} ({lat:.2f}\u00b0, {lon:.2f}\u00b0{val_str})")

        self._update_selection_highlight(cell_id)
        self.face_selected.emit(cell_id, info)

    def _update_selection_highlight(self, face_id):
        """Show yellow highlight for selected face."""
        if not PYVISTA_AVAILABLE or self.pv_mesh is None or face_id is None:
            return
        try:
            try:
                self.plotter.remove_actor("selected_face_marker", render=False)
            except Exception:
                pass

            cell = self.pv_mesh.extract_cells([face_id])
            if cell.n_cells > 0:
                self.plotter.add_mesh(
                    cell, color="yellow", style="surface", opacity=0.7,
                    render=False, name="selected_face_marker", pickable=False)
                self.plotter.render()
        except Exception:
            pass

    def clear_cache(self):
        self.data_cache.clear()

    def reset_camera(self):
        if PYVISTA_AVAILABLE and self.plotter:
            self.plotter.reset_camera()


# =========================================================================
# 2D Plot Widget
# =========================================================================

class ExoPlotWidget(QWidget):
    """2D plotting widget with multiple plot types."""

    PLOT_TYPES = [
        "Field Time Series",
        "Multi-Field Time Series",
        "Mass Balance Evolution",
        "Latitude Distribution",
        "Field Histogram",
        "Crystalline Fraction Map",
        "Sputtering Distribution",
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.data_cache = ExoDataCache(max_entries=100)
        self.mesh_data = None
        self.exo_dir = ""
        self.ptm_dir = ""
        self.selected_face = None
        self.current_timestep = 0
        self.current_field = "temperature_K"
        self.fields = {}
        # Mapping from slider position to actual timestep index
        self._timestep_indices = []
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if not MATPLOTLIB_AVAILABLE:
            layout.addWidget(QLabel(
                "Matplotlib not available. Install with: pip install matplotlib"))
            return

        # Controls
        controls = QHBoxLayout()
        controls.setContentsMargins(5, 5, 5, 5)

        controls.addWidget(QLabel("Plot:"))
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems(self.PLOT_TYPES)
        self.plot_type_combo.setMinimumWidth(180)
        self.plot_type_combo.currentTextChanged.connect(self._on_plot_type_changed)
        controls.addWidget(self.plot_type_combo)

        controls.addWidget(QLabel("  Lat:"))
        self.lat_input = QDoubleSpinBox()
        self.lat_input.setRange(-90, 90)
        self.lat_input.setDecimals(1)
        self.lat_input.setValue(0.0)
        self.lat_input.setMaximumWidth(80)
        controls.addWidget(self.lat_input)

        controls.addWidget(QLabel("Lon:"))
        self.lon_input = QDoubleSpinBox()
        self.lon_input.setRange(0, 360)
        self.lon_input.setDecimals(1)
        self.lon_input.setValue(0.0)
        self.lon_input.setMaximumWidth(80)
        controls.addWidget(self.lon_input)

        self.find_face_btn = QPushButton("Find Face")
        self.find_face_btn.setMaximumWidth(80)
        self.find_face_btn.clicked.connect(self._find_nearest_face)
        controls.addWidget(self.find_face_btn)

        self.face_label = QLabel("Face: --")
        self.face_label.setStyleSheet("color: #64b5f6;")
        controls.addWidget(self.face_label)

        # Time axis toggle
        controls.addWidget(QLabel("  X-axis:"))
        self.time_unit_combo = QComboBox()
        self.time_unit_combo.addItems(["Timestep", "Days", "Years"])
        self.time_unit_combo.setMaximumWidth(80)
        controls.addWidget(self.time_unit_combo)

        # Export
        self.export_btn = QPushButton("Export CSV")
        self.export_btn.setMaximumWidth(90)
        self.export_btn.clicked.connect(self._export_data)
        controls.addWidget(self.export_btn)

        controls.addStretch()

        ctrl_widget = QWidget()
        ctrl_widget.setLayout(controls)
        layout.addWidget(ctrl_widget)

        # Matplotlib canvas
        self.figure = Figure(figsize=(8, 5), dpi=100, facecolor="#1e1e1e")
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 1)

        # Last plotted data (for CSV export)
        self._last_plot_data = None

    def set_directories(self, ptm_dir="", exo_dir=""):
        if ptm_dir:
            self.ptm_dir = ptm_dir
        if exo_dir:
            self.exo_dir = exo_dir
            self.fields = discover_fields(exo_dir)
            if self.fields:
                first_field_ts = next(iter(self.fields.values()))
                self._timestep_indices = [ts_idx for ts_idx, _ in first_field_ts]

    def set_mesh_data(self, mesh_data):
        self.mesh_data = mesh_data

    def set_selected_face(self, face_id):
        self.selected_face = face_id
        if self.mesh_data:
            lat = self.mesh_data["face_latitudes"][face_id]
            lon = self.mesh_data["face_longitudes"][face_id]
            self.face_label.setText(f"Face: {face_id} ({lat:.1f}, {lon:.1f})")
            self.lat_input.setValue(lat)
            self.lon_input.setValue(lon)

    def set_current_field(self, field_name):
        self.current_field = field_name

    def set_current_timestep(self, slider_position):
        self.current_timestep = slider_position

    def refresh_plot(self):
        """Refresh the current plot."""
        plot_type = self.plot_type_combo.currentText()
        self._draw_plot(plot_type)

    def _on_plot_type_changed(self, text):
        self._draw_plot(text)

    def _find_nearest_face(self):
        if self.mesh_data is None:
            return
        lat = self.lat_input.value()
        lon = self.lon_input.value()
        face_idx, dist = find_nearest_face(
            self.mesh_data["face_centers"], lat, lon
        )
        self.selected_face = face_idx
        self.face_label.setText(f"Face: {face_idx} ({lat:.1f}, {lon:.1f})")
        self.refresh_plot()

    def _draw_plot(self, plot_type):
        if not MATPLOTLIB_AVAILABLE:
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_facecolor("#2d2d2d")
        self.figure.patch.set_facecolor("#1e1e1e")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_color("#555")

        try:
            if plot_type == "Field Time Series":
                self._plot_field_timeseries(ax)
            elif plot_type == "Multi-Field Time Series":
                self._plot_multi_field_timeseries(ax)
            elif plot_type == "Mass Balance Evolution":
                self._plot_mass_balance(ax)
            elif plot_type == "Latitude Distribution":
                self._plot_latitude_distribution(ax)
            elif plot_type == "Field Histogram":
                self._plot_field_histogram(ax)
            elif plot_type == "Crystalline Fraction Map":
                self._plot_crystalline_map(ax)
            elif plot_type == "Sputtering Distribution":
                self._plot_sputtering_distribution(ax)
            else:
                ax.text(0.5, 0.5, f"Unknown plot type: {plot_type}",
                        ha="center", va="center", color="white",
                        transform=ax.transAxes)
        except Exception as e:
            ax.text(0.5, 0.5, f"Plot error: {e}",
                    ha="center", va="center", color="#ff6666",
                    transform=ax.transAxes, fontsize=10, wrap=True)

        self.figure.tight_layout()
        self.canvas.draw()

    # ----- Plot implementations -----

    def _get_time_axis(self, n_points):
        """Get time axis values based on selected unit."""
        unit = self.time_unit_combo.currentText()
        times = load_timestep_times(self.exo_dir)
        if times is not None and len(times) >= n_points:
            t = times[:n_points]
            if unit == "Days":
                return t / 86400.0, "Time (days)"
            elif unit == "Years":
                return t / (365.25 * 86400.0), "Time (years)"
            else:
                return np.arange(n_points), "Timestep"

        mode = detect_mode(self.exo_dir)
        if mode == "multiscale" and unit != "Timestep":
            epoch_times = compute_epoch_times(n_points)
            if unit == "Days":
                return epoch_times["days"], "Time (days)"
            elif unit == "Years":
                return epoch_times["years"], "Time (years)"

        return np.arange(n_points), "Timestep"

    def _plot_field_timeseries(self, ax):
        """Single field at selected face vs time."""
        if self.selected_face is None:
            ax.text(0.5, 0.5, "Select a face to plot time series",
                    ha="center", va="center", color="white", transform=ax.transAxes)
            return

        data = self.data_cache.load_timeseries(
            self.exo_dir, self.current_field, [self.selected_face]
        )
        if data is None or len(data) == 0:
            ax.text(0.5, 0.5, f"No data for {self.current_field}",
                    ha="center", va="center", color="white", transform=ax.transAxes)
            return

        values = data[:, 0]
        x, xlabel = self._get_time_axis(len(values))
        info = get_field_info(self.current_field)

        ax.plot(x, values, color="#64b5f6", linewidth=1.5)
        ax.set_xlabel(xlabel)
        ylabel = f"{info['label']} ({info['units']})" if info["units"] else info["label"]
        ax.set_ylabel(ylabel)
        lat = self.mesh_data["face_latitudes"][self.selected_face] if self.mesh_data else 0
        lon = self.mesh_data["face_longitudes"][self.selected_face] if self.mesh_data else 0
        ax.set_title(f"{info['label']} - Face {self.selected_face} ({lat:.1f}, {lon:.1f})")

        self._last_plot_data = {"x": x, "y": values, "xlabel": xlabel, "ylabel": ylabel}

    def _plot_multi_field_timeseries(self, ax):
        """Multiple flux fields at selected face, overlaid."""
        if self.selected_face is None:
            ax.text(0.5, 0.5, "Select a face to plot",
                    ha="center", va="center", color="white", transform=ax.transAxes)
            return

        flux_fields = [
            ("emission_flux_kg_m2_s", "#ff6666"),
            ("deposition_flux_kg_m2_s", "#66ff66"),
            ("net_flux_kg_m2_s", "#6666ff"),
            ("escape_flux_kg_m2_s", "#ffff66"),
            ("sputtering_flux_kg_m2_s", "#ff66ff"),
        ]

        plotted = False
        for field_name, color in flux_fields:
            data = self.data_cache.load_timeseries(
                self.exo_dir, field_name, [self.selected_face]
            )
            if data is not None and len(data) > 0:
                values = data[:, 0]
                x, xlabel = self._get_time_axis(len(values))
                info = get_field_info(field_name)
                ax.plot(x, values, color=color, linewidth=1.2, label=info["label"])
                plotted = True

        if plotted:
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Flux (kg/m\u00b2/s)")
            ax.set_title(f"Mass Fluxes - Face {self.selected_face}")
            ax.legend(fontsize=8, facecolor="#2d2d2d", edgecolor="#555",
                      labelcolor="white")
        else:
            ax.text(0.5, 0.5, "No flux data available",
                    ha="center", va="center", color="white", transform=ax.transAxes)

    def _plot_mass_balance(self, ax):
        """Total emission/deposition/escape vs time."""
        atm_data = load_atmospheric_mass(self.exo_dir)
        if atm_data is None:
            ax.text(0.5, 0.5, "No atmospheric_mass.csv found",
                    ha="center", va="center", color="white", transform=ax.transAxes)
            return

        ts = atm_data.get("timestep", np.arange(len(next(iter(atm_data.values())))))
        x, xlabel = self._get_time_axis(len(ts))

        for key, color, label in [
            ("total_emission", "#ff6666", "Total Emission"),
            ("total_deposition", "#66ff66", "Total Deposition"),
            ("total_escape", "#ffff66", "Total Escape"),
            ("atmospheric_mass", "#64b5f6", "Atmospheric Mass"),
        ]:
            if key in atm_data:
                ax.plot(x[:len(atm_data[key])], atm_data[key], color=color,
                        linewidth=1.2, label=label)

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Mass (kg)")
        ax.set_title("Global Mass Balance")
        ax.legend(fontsize=8, facecolor="#2d2d2d", edgecolor="#555", labelcolor="white")

    def _plot_latitude_distribution(self, ax):
        """Field values binned by latitude at current timestep."""
        if self.mesh_data is None:
            ax.text(0.5, 0.5, "No mesh loaded",
                    ha="center", va="center", color="white", transform=ax.transAxes)
            return

        # Load data using actual file path, not sequential index
        data = self._load_field_at_slider_position(
            self.current_field, self.current_timestep)
        if data is None:
            ax.text(0.5, 0.5, f"No data for {self.current_field} at step {self.current_timestep}",
                    ha="center", va="center", color="white", transform=ax.transAxes)
            return

        lats = self.mesh_data["face_latitudes"]
        bins = np.linspace(-90, 90, 37)  # 5-degree bins
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        bin_means = np.full(len(bin_centers), np.nan)

        for i in range(len(bin_centers)):
            mask = (lats >= bins[i]) & (lats < bins[i + 1])
            if np.any(mask):
                bin_means[i] = np.mean(data[mask])

        valid = ~np.isnan(bin_means)
        info = get_field_info(self.current_field)
        ax.plot(bin_centers[valid], bin_means[valid], "o-", color="#64b5f6",
                markersize=4, linewidth=1.5)
        ax.set_xlabel("Latitude (deg)")
        ylabel = f"{info['label']} ({info['units']})" if info["units"] else info["label"]
        ax.set_ylabel(ylabel)
        ax.set_title(f"Latitude Distribution - {info['label']} (step {self.current_timestep})")

        self._last_plot_data = {"x": bin_centers[valid], "y": bin_means[valid],
                                "xlabel": "Latitude", "ylabel": ylabel}

    def _plot_field_histogram(self, ax):
        """Histogram of field values at current timestep."""
        data = self._load_field_at_slider_position(
            self.current_field, self.current_timestep)
        if data is None:
            ax.text(0.5, 0.5, f"No data for {self.current_field}",
                    ha="center", va="center", color="white", transform=ax.transAxes)
            return

        info = get_field_info(self.current_field)
        valid = data[data != 0] if np.any(data != 0) else data
        ax.hist(valid, bins=50, color="#64b5f6", alpha=0.8, edgecolor="#1e1e1e")
        xlabel = f"{info['label']} ({info['units']})" if info["units"] else info["label"]
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.set_title(f"Distribution - {info['label']} (step {self.current_timestep})")

    def _plot_crystalline_map(self, ax):
        """Crystalline fraction vs latitude."""
        if self.mesh_data is None:
            ax.text(0.5, 0.5, "No mesh loaded",
                    ha="center", va="center", color="white", transform=ax.transAxes)
            return

        data = self._load_field_at_slider_position(
            "crystalline_fraction", self.current_timestep)
        if data is None:
            ax.text(0.5, 0.5, "No crystalline_fraction data",
                    ha="center", va="center", color="white", transform=ax.transAxes)
            return

        lats = self.mesh_data["face_latitudes"]
        lons = self.mesh_data["face_longitudes"]
        scatter = ax.scatter(lons, lats, c=data, cmap="coolwarm", s=1,
                             vmin=0, vmax=1, alpha=0.8)
        self.figure.colorbar(scatter, ax=ax, label="Crystalline Fraction")
        ax.set_xlabel("Longitude (deg)")
        ax.set_ylabel("Latitude (deg)")
        ax.set_title(f"Crystalline Fraction Map (step {self.current_timestep})")

    def _plot_sputtering_distribution(self, ax):
        """Sputtering flux vs latitude/longitude."""
        if self.mesh_data is None:
            ax.text(0.5, 0.5, "No mesh loaded",
                    ha="center", va="center", color="white", transform=ax.transAxes)
            return

        data = self._load_field_at_slider_position(
            "sputtering_flux_kg_m2_s", self.current_timestep)
        if data is None:
            ax.text(0.5, 0.5, "No sputtering data",
                    ha="center", va="center", color="white", transform=ax.transAxes)
            return

        lats = self.mesh_data["face_latitudes"]
        lons = self.mesh_data["face_longitudes"]
        scatter = ax.scatter(lons, lats, c=data, cmap="Reds", s=1, alpha=0.8)
        self.figure.colorbar(scatter, ax=ax, label="Sputtering Flux (kg/m\u00b2/s)")
        ax.set_xlabel("Longitude (deg)")
        ax.set_ylabel("Latitude (deg)")
        ax.set_title(f"Sputtering Distribution (step {self.current_timestep})")

    def _load_field_at_slider_position(self, field_name, slider_position):
        """Load field data using slider position mapped to actual file.

        Unlike the 3D widget which gets data from the cache,
        the 2D widget needs to resolve slider position -> file path.
        """
        fields = discover_fields(self.exo_dir)
        if field_name not in fields:
            return None
        timesteps = fields[field_name]
        if slider_position < 0 or slider_position >= len(timesteps):
            return None
        _, filepath = timesteps[slider_position]
        return self.data_cache.get_field(filepath)

    def _export_data(self):
        """Export last plotted data to CSV."""
        if self._last_plot_data is None:
            QMessageBox.information(self, "Export", "No data to export. Generate a plot first.")
            return
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Data", "", "CSV Files (*.csv);;All Files (*)"
        )
        if not filepath:
            return
        try:
            with open(filepath, "w", newline="") as f:
                writer = csv.writer(f)
                xlabel = self._last_plot_data.get("xlabel", "x")
                ylabel = self._last_plot_data.get("ylabel", "y")
                writer.writerow([xlabel, ylabel])
                for x, y in zip(self._last_plot_data["x"], self._last_plot_data["y"]):
                    writer.writerow([x, y])
            QMessageBox.information(self, "Export", f"Data exported to {filepath}")
        except Exception as e:
            QMessageBox.warning(self, "Export Error", str(e))

    def clear_cache(self):
        self.data_cache.clear()


# =========================================================================
# Combined Visualization Panel
# =========================================================================

class ExoVisualizationPanel(QWidget):
    """Combined 3D + 2D visualization panel with timestep controls."""

    face_selected = pyqtSignal(int, dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.mesh_data = None
        self.fields = {}
        self.n_timesteps = 0
        self.current_timestep = 0
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Tab widget for 3D / 2D
        self.tab_widget = QTabWidget()

        # 3D tab
        self.pyvista_widget = ExoPyVistaWidget()
        self.pyvista_widget.face_selected.connect(self._on_face_selected)
        self.tab_widget.addTab(self.pyvista_widget, "3D View")

        # 2D tab
        self.plot_widget = ExoPlotWidget()
        self.tab_widget.addTab(self.plot_widget, "2D Plots")

        layout.addWidget(self.tab_widget, 1)

        # Timestep controls
        ts_frame = QFrame()
        ts_frame.setFrameShape(QFrame.Shape.StyledPanel)
        ts_layout = QHBoxLayout(ts_frame)
        ts_layout.setContentsMargins(10, 5, 10, 5)

        ts_layout.addWidget(QLabel("Timestep:"))
        self.ts_slider = QSlider(Qt.Orientation.Horizontal)
        self.ts_slider.setMinimum(0)
        self.ts_slider.setMaximum(0)
        self.ts_slider.valueChanged.connect(self._on_timestep_changed)
        ts_layout.addWidget(self.ts_slider, 1)

        self.ts_label = QLabel("0 / 0")
        self.ts_label.setMinimumWidth(80)
        ts_layout.addWidget(self.ts_label)

        # Step buttons
        self.prev_btn = QPushButton("<")
        self.prev_btn.setMaximumWidth(30)
        self.prev_btn.clicked.connect(lambda: self._step(-1))
        ts_layout.addWidget(self.prev_btn)

        self.next_btn = QPushButton(">")
        self.next_btn.setMaximumWidth(30)
        self.next_btn.clicked.connect(lambda: self._step(1))
        ts_layout.addWidget(self.next_btn)

        # Export image
        self.export_img_btn = QPushButton("Export Image")
        self.export_img_btn.setMaximumWidth(100)
        self.export_img_btn.clicked.connect(self._export_image)
        ts_layout.addWidget(self.export_img_btn)

        layout.addWidget(ts_frame)

    def load_mesh(self, mesh_path):
        """Load mesh for both 3D and 2D widgets."""
        success = self.pyvista_widget.load_mesh(mesh_path)
        if success:
            self.mesh_data = self.pyvista_widget.mesh_data
            self.plot_widget.set_mesh_data(self.mesh_data)
        return success

    def set_directories(self, ptm_dir="", exo_dir=""):
        """Set data directories for visualization."""
        self.pyvista_widget.set_directories(ptm_dir, exo_dir)
        self.plot_widget.set_directories(ptm_dir, exo_dir)

        # Update timestep slider using actual timestep count
        source_dir = exo_dir or ptm_dir
        if source_dir:
            fields = discover_fields(source_dir)
            if fields:
                first_field_ts = next(iter(fields.values()))
                self.n_timesteps = len(first_field_ts)
                self.ts_slider.setMaximum(max(0, self.n_timesteps - 1))
                self.ts_label.setText(f"0 / {self.n_timesteps - 1}")

                # Trigger initial display at timestep 0
                QTimer.singleShot(100, lambda: self._force_update(0))

    def _force_update(self, slider_pos):
        """Force an update at the given slider position."""
        self.ts_slider.setValue(slider_pos)
        # If setValue didn't trigger (because it was already 0), call manually
        self.update_timestep(slider_pos)

    def set_field(self, field_name):
        """Set the displayed field."""
        self.pyvista_widget.current_field = field_name
        self.plot_widget.set_current_field(field_name)
        # Update 3D combo to match
        for i in range(self.pyvista_widget.field_combo.count()):
            if self.pyvista_widget.field_combo.itemData(i) == field_name:
                self.pyvista_widget.field_combo.setCurrentIndex(i)
                break

    def update_timestep(self, slider_position):
        """Update both 3D and 2D to a given slider position."""
        self.current_timestep = slider_position
        self.pyvista_widget.update_display(slider_position)
        self.plot_widget.set_current_timestep(slider_position)

    def _on_timestep_changed(self, value):
        self.current_timestep = value
        self.ts_label.setText(f"{value} / {max(0, self.n_timesteps - 1)}")
        self.update_timestep(value)

    def _step(self, delta):
        new_val = self.ts_slider.value() + delta
        new_val = max(0, min(new_val, self.ts_slider.maximum()))
        self.ts_slider.setValue(new_val)

    def _on_face_selected(self, face_id, info):
        self.plot_widget.set_selected_face(face_id)
        self.face_selected.emit(face_id, info)
        # Auto-refresh 2D plot if on time series
        if "Time Series" in self.plot_widget.plot_type_combo.currentText():
            self.plot_widget.refresh_plot()

    def _export_image(self):
        if self.tab_widget.currentIndex() == 0:
            # 3D view
            if PYVISTA_AVAILABLE and self.pyvista_widget.plotter:
                filepath, _ = QFileDialog.getSaveFileName(
                    self, "Export 3D Image", "",
                    "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)"
                )
                if filepath:
                    self.pyvista_widget.plotter.screenshot(filepath)
        else:
            # 2D plot
            if MATPLOTLIB_AVAILABLE:
                filepath, _ = QFileDialog.getSaveFileName(
                    self, "Export Plot", "",
                    "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg);;All Files (*)"
                )
                if filepath:
                    self.plot_widget.figure.savefig(filepath, dpi=150,
                                                     facecolor="#1e1e1e",
                                                     bbox_inches="tight")

    def refresh(self):
        """Refresh current view."""
        self.update_timestep(self.current_timestep)
        if self.tab_widget.currentIndex() == 1:
            self.plot_widget.refresh_plot()

    def clear_cache(self):
        self.pyvista_widget.clear_cache()
        self.plot_widget.clear_cache()
