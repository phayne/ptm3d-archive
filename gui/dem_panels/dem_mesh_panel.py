"""
DEM Mesh Inspection Panel: 3D PyVista mesh preview with statistics.
"""

import numpy as np
from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QFormLayout, QFileDialog, QLineEdit, QSizePolicy,
    QTableWidget, QTableWidgetItem, QHeaderView, QSplitter
)
from PyQt6.QtCore import pyqtSignal, Qt

try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False


class DEMMeshPanel(QWidget):
    """3D mesh inspection with PyVista and statistics table."""

    mesh_loaded = pyqtSignal(str)  # path to loaded mesh

    def __init__(self, parent=None):
        super().__init__(parent)
        self._mesh = None
        self._mesh_path = None
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # File selection
        file_layout = QHBoxLayout()
        self._file_edit = QLineEdit()
        self._file_edit.setPlaceholderText("Load a PLY mesh file...")
        self._file_edit.setReadOnly(True)
        file_layout.addWidget(self._file_edit)

        load_btn = QPushButton("Load PLY")
        load_btn.clicked.connect(self._browse_mesh)
        file_layout.addWidget(load_btn)

        layout.addLayout(file_layout)

        # Splitter: 3D view (top) + stats (bottom)
        self._splitter = QSplitter(Qt.Orientation.Vertical)

        # 3D viewer
        self._viewer_widget = None
        if HAS_PYVISTA:
            self._plotter = QtInteractor(self)
            self._plotter.set_background('#1e1e1e')
            self._plotter.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
            )
            self._viewer_widget = self._plotter
            self._splitter.addWidget(self._plotter)
        else:
            placeholder = QLabel("PyVista not available for 3D preview.\n"
                                 "Install: pip install pyvista pyvistaqt")
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            placeholder.setStyleSheet("color: #a0a0a0; font-size: 14px;")
            self._viewer_widget = placeholder
            self._splitter.addWidget(placeholder)

        # Statistics table
        self._stats_widget = QWidget()
        stats_layout = QVBoxLayout(self._stats_widget)
        stats_layout.setContentsMargins(0, 4, 0, 0)

        stats_label = QLabel("Mesh Statistics")
        stats_label.setStyleSheet("color: #00bcd4; font-weight: bold; font-size: 12px;")
        stats_layout.addWidget(stats_label)

        self._stats_table = QTableWidget(0, 2)
        self._stats_table.setHorizontalHeaderLabels(["Property", "Value"])
        self._stats_table.horizontalHeader().setStretchLastSection(True)
        self._stats_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self._stats_table.verticalHeader().setVisible(False)
        stats_layout.addWidget(self._stats_table)

        self._splitter.addWidget(self._stats_widget)
        self._splitter.setSizes([500, 200])

        layout.addWidget(self._splitter, 1)

    def hide_3d_viewer(self):
        """Hide the 3D viewer, showing only file selector and stats table.

        Used when the 3D view is provided externally (e.g., in the right
        splitter pane of the dashboard).
        """
        if self._viewer_widget is not None:
            self._viewer_widget.hide()
        # Remove max height constraint so stats table fills the space
        self._stats_table.setMaximumHeight(16777215)

    def _browse_mesh(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load Mesh File",
            str(Path.home()),
            "PLY Files (*.ply);;OBJ Files (*.obj);;VTK Files (*.vtk *.vtu *.vtp);;All Files (*)"
        )
        if filepath:
            self.load_mesh(filepath)

    def load_mesh(self, filepath):
        """Load and display a mesh file."""
        if not HAS_PYVISTA:
            self._file_edit.setText(filepath)
            self._mesh_path = filepath
            self.mesh_loaded.emit(filepath)
            return

        try:
            self._mesh = pv.read(filepath)
            self._mesh_path = filepath
            self._file_edit.setText(filepath)

            # Display mesh
            self._plotter.clear()

            # Color by elevation (Z coordinate) if no scalars
            if self._mesh.n_points > 0:
                z_vals = self._mesh.points[:, 2]
                # For spherical meshes, use radial distance as "elevation"
                r_vals = np.linalg.norm(self._mesh.points, axis=1)
                if r_vals.std() / r_vals.mean() > 0.001:
                    # Significant radial variation -> use as elevation
                    elevation = r_vals - r_vals.mean()
                    self._mesh.point_data["Elevation"] = elevation
                    self._plotter.add_mesh(
                        self._mesh, scalars="Elevation",
                        cmap='terrain', show_edges=False,
                        scalar_bar_args={'title': 'Elevation (m)', 'color': '#a0a0a0'}
                    )
                else:
                    self._plotter.add_mesh(
                        self._mesh, color='#4488cc', show_edges=False
                    )

            self._plotter.reset_camera()
            self._plotter.render()

            # Update statistics
            self._update_stats()
            self.mesh_loaded.emit(filepath)

        except Exception as e:
            self._file_edit.setText(f"Error: {e}")

    def _update_stats(self):
        """Compute and display mesh statistics."""
        self._stats_table.setRowCount(0)

        if self._mesh is None:
            return

        stats = []

        # Basic counts
        stats.append(("Vertices", f"{self._mesh.n_points:,}"))
        stats.append(("Faces", f"{self._mesh.n_cells:,}"))

        # File size
        if self._mesh_path:
            size = Path(self._mesh_path).stat().st_size
            if size > 1e9:
                stats.append(("File Size", f"{size / 1e9:.2f} GB"))
            elif size > 1e6:
                stats.append(("File Size", f"{size / 1e6:.1f} MB"))
            else:
                stats.append(("File Size", f"{size / 1e3:.0f} KB"))

        # Bounding box
        if self._mesh.n_points > 0:
            bounds = self._mesh.bounds
            stats.append(("X Range", f"[{bounds[0]:.1f}, {bounds[1]:.1f}]"))
            stats.append(("Y Range", f"[{bounds[2]:.1f}, {bounds[3]:.1f}]"))
            stats.append(("Z Range", f"[{bounds[4]:.1f}, {bounds[5]:.1f}]"))

            # Radial distance (body center to vertices)
            r = np.linalg.norm(self._mesh.points, axis=1)
            stats.append(("Radius Range", f"[{r.min():.1f}, {r.max():.1f}] m"))

            # Convert to lat/lon for geographic bounds
            lat = np.degrees(np.arcsin(np.clip(
                self._mesh.points[:, 2] / np.maximum(r, 1e-10), -1, 1
            )))
            lon = np.degrees(np.arctan2(
                self._mesh.points[:, 1], self._mesh.points[:, 0]
            ))
            stats.append(("Lat Range", f"[{lat.min():.3f}, {lat.max():.3f}] deg"))
            stats.append(("Lon Range", f"[{lon.min():.3f}, {lon.max():.3f}] deg"))

        # Edge lengths
        if self._mesh.n_cells > 0:
            try:
                edges = self._mesh.extract_all_edges()
                if edges.n_cells > 0:
                    lengths = []
                    pts = self._mesh.points
                    # Sample up to 10000 edges for performance
                    cells = edges.lines
                    idx = 0
                    count = 0
                    max_sample = 10000
                    while idx < len(cells) and count < max_sample:
                        n = cells[idx]
                        if n == 2:
                            p1 = pts[cells[idx + 1]]
                            p2 = pts[cells[idx + 2]]
                            lengths.append(np.linalg.norm(p2 - p1))
                        idx += n + 1
                        count += 1

                    if lengths:
                        lengths = np.array(lengths)
                        stats.append(("Edge Length Min", f"{lengths.min():.2f} m"))
                        stats.append(("Edge Length Mean", f"{lengths.mean():.2f} m"))
                        stats.append(("Edge Length Max", f"{lengths.max():.2f} m"))
                        stats.append(("Total Edges", f"{edges.n_cells:,}"))
            except Exception:
                pass

        # Available data arrays
        point_arrays = list(self._mesh.point_data.keys())
        cell_arrays = list(self._mesh.cell_data.keys())
        if point_arrays:
            stats.append(("Point Data", ", ".join(point_arrays)))
        if cell_arrays:
            stats.append(("Cell Data", ", ".join(cell_arrays)))

        # Populate table
        self._stats_table.setRowCount(len(stats))
        for i, (prop, val) in enumerate(stats):
            self._stats_table.setItem(i, 0, QTableWidgetItem(prop))
            self._stats_table.setItem(i, 1, QTableWidgetItem(str(val)))
