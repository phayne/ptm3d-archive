"""
Thermal Analysis Panel: PSR detection, cold-trap identification,
diurnal temperature curves, and export.
"""

import struct
import numpy as np
from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QFormLayout, QDoubleSpinBox, QSpinBox,
    QFileDialog, QTextEdit, QTabWidget, QSizePolicy, QCheckBox
)
from PyQt6.QtCore import pyqtSignal, Qt

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def _read_binary_field(filepath, expected_count=None):
    """Read a PTM3D binary field file.

    Format: [uint32 count][float32 data[count]]
    """
    path = Path(filepath)
    if not path.exists():
        return None

    with open(path, 'rb') as f:
        raw = f.read(4)
        if len(raw) < 4:
            return None
        count = struct.unpack('<I', raw)[0]

        if expected_count and count != expected_count:
            # Try headerless format
            f.seek(0)
            data = np.frombuffer(f.read(), dtype=np.float32)
            return data

        data = np.frombuffer(f.read(count * 4), dtype=np.float32)
        return data


class ThermalAnalysisPanel(QWidget):
    """Analysis tools for thermal simulation output."""

    faces_highlighted = pyqtSignal(list, str)  # face indices, label

    def __init__(self, config_manager=None, results_panel=None, parent=None):
        super().__init__(parent)
        self._cm = config_manager
        self._results_panel = results_panel
        self._results_dir = None
        self._num_faces = None
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        tabs = QTabWidget()

        # Tab 1: PSR / Cold-Trap Detection
        tabs.addTab(self._create_detection_tab(), "PSR / Cold Traps")

        # Tab 2: Diurnal Curves
        tabs.addTab(self._create_diurnal_tab(), "Diurnal Curves")

        layout.addWidget(tabs)

    def _create_detection_tab(self) -> QWidget:
        """PSR and cold-trap detection tab."""
        page = QWidget()
        layout = QVBoxLayout(page)

        # PSR Detection
        psr_group = QGroupBox("Permanently Shadowed Regions (PSR)")
        psr_form = QFormLayout(psr_group)

        psr_desc = QLabel(
            "Identify faces where average illumination is below a threshold.\n"
            "Uses illumination_avg.bin from statistics output."
        )
        psr_desc.setStyleSheet("color: #a0a0a0;")
        psr_desc.setWordWrap(True)
        psr_form.addRow(psr_desc)

        self._psr_threshold = QDoubleSpinBox()
        self._psr_threshold.setRange(0.0, 100.0)
        self._psr_threshold.setValue(0.1)
        self._psr_threshold.setDecimals(2)
        self._psr_threshold.setSuffix(" W/m^2")
        psr_form.addRow("Max Avg Illumination:", self._psr_threshold)

        psr_btn = QPushButton("Detect PSRs")
        psr_btn.clicked.connect(self._detect_psrs)
        psr_form.addRow(psr_btn)

        self._psr_result = QLabel("")
        self._psr_result.setStyleSheet("color: #81c784;")
        self._psr_result.setWordWrap(True)
        psr_form.addRow(self._psr_result)

        layout.addWidget(psr_group)

        # Cold-Trap Detection
        ct_group = QGroupBox("Cold Traps")
        ct_form = QFormLayout(ct_group)

        ct_desc = QLabel(
            "Identify faces where maximum temperature stays below a threshold.\n"
            "Default 110 K is the H2O stability limit at 1 mm/Gyr loss rate."
        )
        ct_desc.setStyleSheet("color: #a0a0a0;")
        ct_desc.setWordWrap(True)
        ct_form.addRow(ct_desc)

        self._ct_threshold = QDoubleSpinBox()
        self._ct_threshold.setRange(10.0, 500.0)
        self._ct_threshold.setValue(110.0)
        self._ct_threshold.setDecimals(1)
        self._ct_threshold.setSuffix(" K")
        ct_form.addRow("Max Temperature:", self._ct_threshold)

        ct_btn = QPushButton("Detect Cold Traps")
        ct_btn.clicked.connect(self._detect_cold_traps)
        ct_form.addRow(ct_btn)

        self._ct_result = QLabel("")
        self._ct_result.setStyleSheet("color: #64b5f6;")
        self._ct_result.setWordWrap(True)
        ct_form.addRow(self._ct_result)

        layout.addWidget(ct_group)

        # Export
        export_group = QGroupBox("Export")
        export_layout = QHBoxLayout(export_group)
        export_psr_btn = QPushButton("Export PSR faces (CSV)")
        export_psr_btn.clicked.connect(lambda: self._export_faces("psr"))
        export_layout.addWidget(export_psr_btn)

        export_ct_btn = QPushButton("Export cold-trap faces (CSV)")
        export_ct_btn.clicked.connect(lambda: self._export_faces("cold_trap"))
        export_layout.addWidget(export_ct_btn)
        layout.addWidget(export_group)

        layout.addStretch()
        return page

    def _create_diurnal_tab(self) -> QWidget:
        """Diurnal temperature curve tab."""
        page = QWidget()
        layout = QVBoxLayout(page)

        desc = QLabel(
            "Plot temperature vs. time for a selected face.\n"
            "Enter a face index or click on the 3D mesh."
        )
        desc.setStyleSheet("color: #a0a0a0;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        ctrl_layout = QHBoxLayout()
        ctrl_layout.addWidget(QLabel("Face Index:"))
        self._face_spin = QSpinBox()
        self._face_spin.setRange(0, 10000000)
        ctrl_layout.addWidget(self._face_spin)

        plot_btn = QPushButton("Plot Diurnal Curve")
        plot_btn.clicked.connect(self._plot_diurnal)
        ctrl_layout.addWidget(plot_btn)
        ctrl_layout.addStretch()
        layout.addLayout(ctrl_layout)

        if HAS_MATPLOTLIB:
            self._diurnal_figure = Figure(figsize=(6, 4), dpi=100)
            self._diurnal_figure.set_facecolor('#1e1e1e')
            self._diurnal_canvas = FigureCanvasQTAgg(self._diurnal_figure)
            self._diurnal_canvas.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
            )
            layout.addWidget(self._diurnal_canvas, 1)
        else:
            layout.addWidget(QLabel("matplotlib required for plotting"),  1)

        return page

    # -- Results directory -------------------------------------------------

    def set_results_dir(self, results_dir):
        """Set the simulation output directory for analysis."""
        self._results_dir = results_dir
        # Try to determine face count from mesh
        mesh_file = Path(results_dir) / "mesh.ply"
        if mesh_file.exists():
            try:
                with open(mesh_file, 'rb') as f:
                    for line in f:
                        if b"element face" in line:
                            self._num_faces = int(line.decode().split()[-1])
                            break
                        if b"end_header" in line:
                            break
            except Exception:
                pass

    # -- PSR Detection -----------------------------------------------------

    def _detect_psrs(self):
        if not self._results_dir:
            self._psr_result.setText("No results directory set. Run a simulation first.")
            return

        # Try to load illumination_avg.bin from statistics
        stats_dir = Path(self._results_dir) / "statistics"
        illum_file = stats_dir / "illumination_avg.bin"
        if not illum_file.exists():
            self._psr_result.setText(
                f"File not found: {illum_file}\n"
                "Run simulation with compute_statistics enabled."
            )
            return

        data = _read_binary_field(illum_file, self._num_faces)
        if data is None:
            self._psr_result.setText("Failed to read illumination data.")
            return

        threshold = self._psr_threshold.value()
        psr_mask = data <= threshold
        psr_count = np.sum(psr_mask)
        psr_fraction = psr_count / len(data) * 100

        self._psr_faces = np.where(psr_mask)[0].tolist()
        self._psr_result.setText(
            f"PSR faces: {psr_count:,} / {len(data):,} ({psr_fraction:.2f}%)\n"
            f"Threshold: avg illumination <= {threshold} W/m^2"
        )

        self.faces_highlighted.emit(self._psr_faces, "PSR")

    # -- Cold-Trap Detection -----------------------------------------------

    def _detect_cold_traps(self):
        if not self._results_dir:
            self._ct_result.setText("No results directory set. Run a simulation first.")
            return

        stats_dir = Path(self._results_dir) / "statistics"
        tmax_file = stats_dir / "temperature_max.bin"
        if not tmax_file.exists():
            self._ct_result.setText(
                f"File not found: {tmax_file}\n"
                "Run simulation with compute_statistics enabled."
            )
            return

        data = _read_binary_field(tmax_file, self._num_faces)
        if data is None:
            self._ct_result.setText("Failed to read temperature data.")
            return

        threshold = self._ct_threshold.value()
        ct_mask = data <= threshold
        ct_count = np.sum(ct_mask)
        ct_fraction = ct_count / len(data) * 100

        self._ct_faces = np.where(ct_mask)[0].tolist()

        # Temperature statistics for cold traps
        if ct_count > 0:
            ct_temps = data[ct_mask]
            extra = f"\nCold-trap Tmax range: [{ct_temps.min():.1f}, {ct_temps.max():.1f}] K"
        else:
            extra = ""

        self._ct_result.setText(
            f"Cold-trap faces: {ct_count:,} / {len(data):,} ({ct_fraction:.2f}%)\n"
            f"Threshold: Tmax <= {threshold} K{extra}"
        )

        self.faces_highlighted.emit(self._ct_faces, "ColdTrap")

    # -- Diurnal Curve -----------------------------------------------------

    def _plot_diurnal(self):
        if not HAS_MATPLOTLIB:
            return

        if not self._results_dir:
            return

        face_idx = self._face_spin.value()
        temp_dir = Path(self._results_dir) / "temperature"
        if not temp_dir.exists():
            return

        # Find all temperature files
        temp_files = sorted(temp_dir.glob("temperature_*.bin"))
        if not temp_files:
            return

        temperatures = []
        for tf in temp_files:
            data = _read_binary_field(tf, self._num_faces)
            if data is not None and face_idx < len(data):
                temperatures.append(data[face_idx])
            else:
                temperatures.append(np.nan)

        temperatures = np.array(temperatures)
        timesteps = np.arange(len(temperatures))

        # Try to read actual times
        time_file = Path(self._results_dir) / "time_steps.txt"
        times = None
        if time_file.exists():
            try:
                raw = []
                with open(time_file) as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            parts = line.split()
                            raw.append(float(parts[0]))
                if len(raw) == len(temperatures):
                    times = np.array(raw)
                    times = (times - times[0]) / 3600.0  # Convert to hours
            except Exception:
                pass

        # Plot
        ax = self._diurnal_figure.clear()
        ax = self._diurnal_figure.add_subplot(111)
        ax.set_facecolor('#2d2d2d')

        x = times if times is not None else timesteps
        xlabel = "Time (hours)" if times is not None else "Timestep"

        ax.plot(x, temperatures, color='#FF6B6B', linewidth=1.5)
        ax.set_xlabel(xlabel, color='#a0a0a0')
        ax.set_ylabel("Temperature (K)", color='#a0a0a0')
        ax.set_title(f"Face {face_idx} Diurnal Curve", color='#e0e0e0', fontsize=11)
        ax.tick_params(colors='#a0a0a0')
        ax.grid(True, alpha=0.2)

        self._diurnal_figure.tight_layout()
        self._diurnal_canvas.draw()

    # -- Export ------------------------------------------------------------

    def _export_faces(self, face_type):
        """Export detected face indices to CSV."""
        if face_type == "psr":
            faces = getattr(self, '_psr_faces', [])
            label = "PSR"
        else:
            faces = getattr(self, '_ct_faces', [])
            label = "cold_trap"

        if not faces:
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, f"Export {label} Faces",
            str(Path.home() / f"{label}_faces.csv"),
            "CSV Files (*.csv);;All Files (*)"
        )
        if filepath:
            with open(filepath, 'w') as f:
                f.write("face_index\n")
                for idx in faces:
                    f.write(f"{idx}\n")
