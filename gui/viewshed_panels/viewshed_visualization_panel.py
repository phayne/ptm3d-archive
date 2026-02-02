"""
Visualization Panel for Viewshed GUI.
3D mesh visualization using PyVista with scalar field colouring.
"""

import struct
import sys
from pathlib import Path

import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QFileDialog, QDoubleSpinBox, QCheckBox,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

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
        self.available_fields = {}  # name -> path

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

        # PyVista plotter
        self.plotter = QtInteractor(self)
        self.plotter.set_background("#1e1e1e")
        self.plotter.add_axes()
        layout.addWidget(self.plotter.interactor, 1)

        # Controls bar
        controls = QHBoxLayout()
        controls.setContentsMargins(5, 5, 5, 5)

        controls.addWidget(QLabel("Field:"))
        self.field_combo = QComboBox()
        self.field_combo.setMinimumWidth(200)
        self.field_combo.currentTextChanged.connect(self._on_field_combo_changed)
        controls.addWidget(self.field_combo)

        # Colormap
        controls.addWidget(QLabel("  Colormap:"))
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems([
            "coolwarm", "viridis", "plasma", "inferno", "magma",
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

        controls.addStretch()

        # Show waypoints toggle
        self.show_waypoints_cb = QCheckBox("Waypoints")
        self.show_waypoints_cb.setChecked(False)
        self.show_waypoints_cb.stateChanged.connect(self._toggle_waypoints)
        controls.addWidget(self.show_waypoints_cb)

        # Camera reset
        reset_btn = QPushButton("Reset Camera")
        reset_btn.setMaximumWidth(100)
        reset_btn.clicked.connect(self.reset_camera)
        controls.addWidget(reset_btn)

        layout.addLayout(controls)

        # Face info label
        self.face_info = QLabel("")
        self.face_info.setStyleSheet(
            "color: #aaa; font-size: 11px; padding: 2px 5px; "
            "background: #252526; border-top: 1px solid #404040;"
        )
        layout.addWidget(self.face_info)

        # Enable face picking
        self.plotter.enable_point_picking(
            callback=self._on_face_picked,
            show_message=False,
            tolerance=0.01,
        )

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

            self.plotter.clear()
            self.plotter.add_mesh(
                self.pv_mesh,
                color="#555555",
                show_edges=False,
                name="surface",
            )
            self.plotter.reset_camera()
            return True
        except Exception as e:
            self.face_info.setText(f"Error loading mesh: {e}")
            return False

    def set_output_directory(self, output_dir):
        """Scan output directory for .bin fields and populate combo box."""
        self.output_dir = output_dir
        self.available_fields = {}

        output_path = Path(output_dir)
        if not output_path.is_dir():
            return

        for p in sorted(output_path.glob("*.bin")):
            self.available_fields[p.stem] = str(p)

        # Update combo box
        current = self.field_combo.currentText()
        self.field_combo.blockSignals(True)
        self.field_combo.clear()
        self.field_combo.addItems(sorted(self.available_fields.keys()))
        # Restore selection
        idx = self.field_combo.findText(current)
        if idx >= 0:
            self.field_combo.setCurrentIndex(idx)
        self.field_combo.blockSignals(False)

    def load_field(self, field_name, file_path=None):
        """Load and display a scalar field on the mesh."""
        if not PYVISTA_AVAILABLE or self.pv_mesh is None:
            return

        if file_path is None:
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

            self.current_field_name = field_name
            self.current_field_data = data
            self.pv_mesh.cell_data[field_name] = data

            # Auto-set combo
            idx = self.field_combo.findText(field_name)
            if idx >= 0:
                self.field_combo.blockSignals(True)
                self.field_combo.setCurrentIndex(idx)
                self.field_combo.blockSignals(False)

            self._refresh_display()
            self._auto_range()

        except Exception as e:
            self.face_info.setText(f"Error loading field: {e}")

    def _on_field_combo_changed(self, field_name):
        """Handle field selection from combo box."""
        if field_name and field_name in self.available_fields:
            self.load_field(field_name)

    def _refresh_display(self):
        """Re-render the mesh with current field and colormap."""
        if not PYVISTA_AVAILABLE or self.pv_mesh is None:
            return
        if self.current_field_data is None:
            return

        cmap = self.cmap_combo.currentText()

        # Determine clamp range
        vmin = self.clamp_min.value()
        vmax = self.clamp_max.value()
        if vmin <= -1e11:
            vmin = None
        if vmax >= 1e11:
            vmax = None

        clim = None
        if vmin is not None and vmax is not None:
            clim = [vmin, vmax]

        self.plotter.clear()
        kwargs = {
            "scalars": self.current_field_name,
            "cmap": cmap,
            "show_edges": False,
            "name": "surface",
            "show_scalar_bar": True,
            "scalar_bar_args": {
                "title": self.current_field_name.replace("_", " "),
                "color": "#e0e0e0",
                "label_font_size": 10,
                "title_font_size": 12,
            },
        }
        if clim is not None:
            kwargs["clim"] = clim

        self.plotter.add_mesh(self.pv_mesh, **kwargs)

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

    def _toggle_waypoints(self, state):
        """Show/hide waypoint markers on the mesh."""
        # Waypoint rendering would require mesh coordinates and waypoint positions.
        # This is a placeholder that will be wired up once the simulation runner
        # provides waypoint lat/lon -> cartesian conversion.
        pass

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
        """Export the current view as a PNG image."""
        if not PYVISTA_AVAILABLE or self.plotter is None:
            return
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Image", "viewshed_visualization.png",
            "PNG Files (*.png);;All Files (*)"
        )
        if filename:
            self.plotter.screenshot(filename)

    def refresh(self):
        """Refresh the visualization (re-scan output directory)."""
        if self.output_dir:
            self.set_output_directory(self.output_dir)
        if self.current_field_name:
            self.load_field(self.current_field_name)
