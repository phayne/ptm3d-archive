"""
Exosphere Analysis Panel: ice thickness maps, sublimation rate plots,
non-ice concentration, and diurnal temperature variation.
"""

import numpy as np
from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QFormLayout, QSpinBox, QComboBox, QSizePolicy
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


class ExosphereAnalysisPanel(QWidget):
    """Analysis tools for exosphere simulation output."""

    def __init__(self, config_manager=None, parent=None):
        super().__init__(parent)
        self._cm = config_manager
        self._output_dir = None
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        title = QLabel("Exosphere Analysis")
        title.setStyleSheet("color: #00bcd4; font-weight: bold; font-size: 14px;")
        layout.addWidget(title)

        desc = QLabel(
            "Analyze exosphere simulation output: ice thickness, sublimation rates,\n"
            "non-ice concentration, and temporal evolution."
        )
        desc.setStyleSheet("color: #a0a0a0;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Field selector
        field_group = QGroupBox("Field Selection")
        field_layout = QFormLayout(field_group)

        self._field_combo = QComboBox()
        self._field_combo.addItems([
            "ice_thickness",
            "nonice_concentration",
            "sublimation_rate",
            "temperature",
            "bond_albedo",
            "crystallinity",
        ])
        field_layout.addRow("Field:", self._field_combo)

        self._epoch_spin = QSpinBox()
        self._epoch_spin.setRange(0, 100000)
        self._epoch_spin.setValue(0)
        field_layout.addRow("Epoch/Timestep:", self._epoch_spin)

        plot_btn = QPushButton("Plot Field Distribution")
        plot_btn.clicked.connect(self._plot_distribution)
        field_layout.addRow(plot_btn)

        layout.addWidget(field_group)

        # Evolution plot
        evol_group = QGroupBox("Temporal Evolution")
        evol_layout = QVBoxLayout(evol_group)

        evol_desc = QLabel(
            "Plot mean value of selected field vs. epoch/time.\n"
            "Reads all VTU files in the exosphere output directory."
        )
        evol_desc.setStyleSheet("color: #a0a0a0;")
        evol_desc.setWordWrap(True)
        evol_layout.addWidget(evol_desc)

        evol_btn = QPushButton("Plot Evolution")
        evol_btn.clicked.connect(self._plot_evolution)
        evol_layout.addWidget(evol_btn)

        layout.addWidget(evol_group)

        # Matplotlib canvas
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
        """Set the exosphere output directory."""
        self._output_dir = output_dir

    def _find_vtu_files(self):
        """Find all VTU files in the output directory."""
        if not self._output_dir:
            return []
        output_path = Path(self._output_dir)
        # Check for exosphere subdirectory
        exo_dir = output_path / "exosphere"
        if exo_dir.exists():
            output_path = exo_dir

        vtu_files = sorted(output_path.glob("*.vtu"))
        return vtu_files

    def _plot_distribution(self):
        """Plot histogram of selected field at given epoch."""
        if not HAS_MATPLOTLIB or not HAS_PYVISTA:
            return

        vtu_files = self._find_vtu_files()
        if not vtu_files:
            return

        epoch = min(self._epoch_spin.value(), len(vtu_files) - 1)
        field_name = self._field_combo.currentText()

        try:
            mesh = pv.read(str(vtu_files[epoch]))
            if field_name not in mesh.point_data and field_name not in mesh.cell_data:
                return

            if field_name in mesh.point_data:
                data = np.array(mesh.point_data[field_name])
            else:
                data = np.array(mesh.cell_data[field_name])

            data = data[np.isfinite(data)]

            ax = self._figure.clear()
            ax = self._figure.add_subplot(111)
            ax.set_facecolor('#2d2d2d')

            ax.hist(data, bins=50, color='#4FC3F7', edgecolor='#1e1e1e', alpha=0.8)
            ax.set_xlabel(field_name, color='#a0a0a0')
            ax.set_ylabel("Count", color='#a0a0a0')
            ax.set_title(
                f"{field_name} distribution (epoch {epoch})",
                color='#e0e0e0', fontsize=11
            )
            ax.tick_params(colors='#a0a0a0')
            ax.grid(True, alpha=0.2)

            self._figure.tight_layout()
            self._canvas.draw()

        except Exception:
            pass

    def _plot_evolution(self):
        """Plot mean field value vs. epoch."""
        if not HAS_MATPLOTLIB or not HAS_PYVISTA:
            return

        vtu_files = self._find_vtu_files()
        if not vtu_files:
            return

        field_name = self._field_combo.currentText()
        means = []
        epochs = []

        for i, vtu_file in enumerate(vtu_files):
            try:
                mesh = pv.read(str(vtu_file))
                if field_name in mesh.point_data:
                    data = np.array(mesh.point_data[field_name])
                elif field_name in mesh.cell_data:
                    data = np.array(mesh.cell_data[field_name])
                else:
                    continue

                data = data[np.isfinite(data)]
                if len(data) > 0:
                    means.append(np.mean(data))
                    epochs.append(i)
            except Exception:
                continue

        if not means:
            return

        ax = self._figure.clear()
        ax = self._figure.add_subplot(111)
        ax.set_facecolor('#2d2d2d')

        ax.plot(epochs, means, color='#FF8A65', linewidth=1.5, marker='o',
                markersize=3)
        ax.set_xlabel("Epoch", color='#a0a0a0')
        ax.set_ylabel(f"Mean {field_name}", color='#a0a0a0')
        ax.set_title(
            f"{field_name} evolution over time",
            color='#e0e0e0', fontsize=11
        )
        ax.tick_params(colors='#a0a0a0')
        ax.grid(True, alpha=0.2)

        self._figure.tight_layout()
        self._canvas.draw()
