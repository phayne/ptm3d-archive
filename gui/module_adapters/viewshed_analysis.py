"""
Viewshed Analysis Panel: visibility heatmaps, illumination-hours,
and mask statistics.
"""

import numpy as np
from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton,
    QGroupBox, QFormLayout, QSizePolicy, QComboBox, QSpinBox
)
from PyQt6.QtCore import pyqtSignal

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False


class ViewshedAnalysisPanel(QWidget):
    """Analysis tools for viewshed simulation output."""

    def __init__(self, config_manager=None, parent=None):
        super().__init__(parent)
        self._cm = config_manager
        self._output_dir = None
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        title = QLabel("Viewshed Analysis")
        title.setStyleSheet("color: #00bcd4; font-weight: bold; font-size: 14px;")
        layout.addWidget(title)

        desc = QLabel(
            "Analyze viewshed output: visibility heatmaps, illumination hours,\n"
            "and custom mask statistics."
        )
        desc.setStyleSheet("color: #a0a0a0;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Visibility analysis
        vis_group = QGroupBox("Visibility Analysis")
        vis_layout = QFormLayout(vis_group)

        self._field_combo = QComboBox()
        self._field_combo.addItems([
            "total_visibility",
            "illumination_hours",
            "shadow_fraction",
        ])
        vis_layout.addRow("Field:", self._field_combo)

        plot_btn = QPushButton("Plot Distribution")
        plot_btn.clicked.connect(self._plot_distribution)
        vis_layout.addRow(plot_btn)

        layout.addWidget(vis_group)

        # Mask statistics
        mask_group = QGroupBox("Mask Statistics")
        mask_layout = QVBoxLayout(mask_group)

        mask_desc = QLabel(
            "Compute statistics for custom boolean masks.\n"
            "Reads truth maps from the viewshed output."
        )
        mask_desc.setStyleSheet("color: #a0a0a0;")
        mask_desc.setWordWrap(True)
        mask_layout.addWidget(mask_desc)

        stats_btn = QPushButton("Compute Mask Statistics")
        stats_btn.clicked.connect(self._compute_mask_stats)
        mask_layout.addWidget(stats_btn)

        self._mask_result = QLabel("")
        self._mask_result.setStyleSheet("color: #81c784;")
        self._mask_result.setWordWrap(True)
        mask_layout.addWidget(self._mask_result)

        layout.addWidget(mask_group)

        # Plot canvas
        if HAS_MATPLOTLIB:
            self._figure = Figure(figsize=(6, 4), dpi=100)
            self._figure.set_facecolor('#1e1e1e')
            self._canvas = FigureCanvasQTAgg(self._figure)
            self._canvas.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
            )
            layout.addWidget(self._canvas, 1)
        else:
            layout.addWidget(QLabel("matplotlib required for analysis plots"), 1)

    def set_output_dir(self, output_dir):
        self._output_dir = output_dir

    def _find_field_files(self):
        """Find viewshed output files."""
        if not self._output_dir:
            return {}
        output_path = Path(self._output_dir)
        files = {}
        for f in output_path.glob("*.bin"):
            files[f.stem] = f
        for f in output_path.glob("*.vtu"):
            files[f.stem] = f
        return files

    def _plot_distribution(self):
        if not HAS_MATPLOTLIB:
            return

        field_name = self._field_combo.currentText()
        files = self._find_field_files()

        # Try to load from VTU first
        data = None
        for name, path in files.items():
            if field_name in name and path.suffix == '.vtu' and HAS_PYVISTA:
                try:
                    mesh = pv.read(str(path))
                    if field_name in mesh.point_data:
                        data = np.array(mesh.point_data[field_name])
                    elif field_name in mesh.cell_data:
                        data = np.array(mesh.cell_data[field_name])
                    break
                except Exception:
                    continue

        if data is None:
            return

        data = data[np.isfinite(data)]

        ax = self._figure.clear()
        ax = self._figure.add_subplot(111)
        ax.set_facecolor('#2d2d2d')

        ax.hist(data, bins=50, color='#81C784', edgecolor='#1e1e1e', alpha=0.8)
        ax.set_xlabel(field_name, color='#a0a0a0')
        ax.set_ylabel("Count", color='#a0a0a0')
        ax.set_title(f"{field_name} distribution", color='#e0e0e0', fontsize=11)
        ax.tick_params(colors='#a0a0a0')
        ax.grid(True, alpha=0.2)

        self._figure.tight_layout()
        self._canvas.draw()

    def _compute_mask_stats(self):
        """Compute and display mask statistics."""
        if not self._output_dir or not HAS_PYVISTA:
            self._mask_result.setText("No output directory set or PyVista unavailable.")
            return

        files = self._find_field_files()
        results = []

        for name, path in files.items():
            if "truth" in name.lower() or "mask" in name.lower():
                try:
                    if path.suffix == '.vtu':
                        mesh = pv.read(str(path))
                        for field in list(mesh.point_data.keys()) + list(mesh.cell_data.keys()):
                            if "truth" in field.lower() or "mask" in field.lower():
                                if field in mesh.point_data:
                                    d = np.array(mesh.point_data[field])
                                else:
                                    d = np.array(mesh.cell_data[field])
                                true_count = np.sum(d > 0.5)
                                total = len(d)
                                results.append(
                                    f"  {field}: {true_count:,}/{total:,} "
                                    f"({true_count/total*100:.1f}%)"
                                )
                except Exception:
                    continue

        if results:
            self._mask_result.setText("Mask Statistics:\n" + "\n".join(results))
        else:
            self._mask_result.setText("No truth/mask fields found in output.")
