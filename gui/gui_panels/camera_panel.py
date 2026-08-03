"""
Camera Simulation Panel for PTM3D GUI.
Provides a complete interface for configuring and launching ptm_camera_sim,
including camera presets, SPICE IK mode, trajectory selection, photometry,
and a thumbnail gallery of rendered frames.
"""

import logging
import math
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QLineEdit,
    QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox, QPushButton,
    QFileDialog, QScrollArea, QProgressBar, QGroupBox, QRadioButton,
    QButtonGroup, QStackedWidget, QSizePolicy, QMessageBox, QSlider
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QPixmap, QImage, QFont

logger = logging.getLogger(__name__)

try:
    from gui_core.camera_presets import CAMERA_PRESETS, PRESET_NAMES, RESOLUTION_SCALES
except ImportError:
    from camera_presets import CAMERA_PRESETS, PRESET_NAMES, RESOLUTION_SCALES


def default_cspice_kernels_dir():
    """Resolve a portable default SPICE kernels directory (GUI-CAM-1).

    Search order: the ``$CSPICE_DATA`` environment variable, then the
    project's ``data/kernels`` directory, else blank. A blank result means
    "let ptm_camera_sim use its own default" -- no ``--kernels-path`` is
    passed (see the launch-command builder). Never hardcode a per-developer
    absolute path here: the old default (``/Users/.../software/cspice/data``)
    was dead on every other machine, with no fallback or warning.
    """
    env = os.environ.get("CSPICE_DATA", "").strip()
    if env and Path(env).is_dir():
        return env
    # gui/gui_panels/camera_panel.py -> project root is parents[2]
    project_kernels = Path(__file__).resolve().parents[2] / "data" / "kernels"
    if project_kernels.is_dir():
        return str(project_kernels)
    return ""


class CameraPanel(QWidget):
    """Camera simulation configuration and launch panel."""

    run_requested = pyqtSignal(list)   # emits the CLI command list
    stop_requested = pyqtSignal()

    def __init__(self, config_manager=None, console_panel=None, parent=None):
        super().__init__(parent)
        self.config_manager = config_manager
        self.console = console_panel
        self._ptm3d_times = []       # parsed (et, utc) from time_steps.txt
        self._ptm3d_cadence_min = 0  # median cadence of PTM3D output
        self._body_radius_km = 1737.4  # default Moon; updated from mesh
        self._suppress_preset_change = False
        self._running = False
        self._rendered_fits = []   # list of FITS paths for re-rendering
        self._selected_fits = None # currently previewed FITS path
        self.init_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def init_ui(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        container = QWidget()
        self.main_layout = QVBoxLayout(container)
        self.main_layout.setSpacing(6)
        self.main_layout.setContentsMargins(8, 8, 8, 8)

        self._build_ptm3d_section()
        self._build_time_section()
        self._build_camera_section()
        self._build_lookat_section()
        self._build_trajectory_section()
        self._build_photometry_section()
        self._build_rendering_section()
        self._build_output_section()
        self._build_run_section()
        self._build_gallery_section()

        self.main_layout.addStretch()
        scroll.setWidget(container)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    # ---------- PTM3D Output ----------

    def _build_ptm3d_section(self):
        grp = QGroupBox("PTM3D Output")
        lay = QGridLayout(grp)

        lay.addWidget(QLabel("Directory:"), 0, 0)
        self.ptm_dir_edit = QLineEdit()
        self.ptm_dir_edit.setPlaceholderText("Auto-detected or browse...")
        lay.addWidget(self.ptm_dir_edit, 0, 1)
        btn = QPushButton("Browse")
        btn.clicked.connect(self._browse_ptm_dir)
        lay.addWidget(btn, 0, 2)

        self.ptm_status_label = QLabel("")
        self.ptm_status_label.setWordWrap(True)
        lay.addWidget(self.ptm_status_label, 1, 0, 1, 3)

        self.main_layout.addWidget(grp)

        # Try auto-detect
        self._auto_detect_ptm_output()

    def _auto_detect_ptm_output(self):
        try:
            from PyQt6.QtCore import QSettings
            settings = QSettings("PTM3D", "PTM3DSimulator")
            last_dir = settings.value("last_output_dir", "")
            if last_dir and Path(last_dir).is_dir():
                self.ptm_dir_edit.setText(last_dir)
                self._load_ptm3d_metadata(last_dir)
        except Exception:
            pass

    def _browse_ptm_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select PTM3D Output Directory")
        if d:
            self.ptm_dir_edit.setText(d)
            self._load_ptm3d_metadata(d)

    def _load_ptm3d_metadata(self, directory):
        p = Path(directory)
        ts_file = p / "time_steps.txt"
        meta_file = p / "metadata.txt"

        if not ts_file.exists():
            self.ptm_status_label.setText("⚠ time_steps.txt not found")
            self.ptm_status_label.setStyleSheet("color: #f0883e;")
            self._ptm3d_times = []
            return

        # Parse time_steps.txt
        times = []
        with open(ts_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        et = float(parts[0])
                        utc = parts[1]
                        times.append((et, utc))
                    except ValueError:
                        continue
        self._ptm3d_times = times

        # Parse metadata for face count
        n_faces = "?"
        if meta_file.exists():
            with open(meta_file) as f:
                for line in f:
                    if line.startswith("faces="):
                        n_faces = line.split("=")[1].strip()

        # Compute cadence
        if len(times) >= 2:
            dts = [(times[i+1][0] - times[i][0]) / 60.0 for i in range(len(times)-1)]
            self._ptm3d_cadence_min = sorted(dts)[len(dts) // 2]  # median
        else:
            self._ptm3d_cadence_min = 0

        if times:
            t0 = times[0][1].split(".")[0]
            t1 = times[-1][1].split(".")[0]
            self.ptm_status_label.setText(
                f"✓ {n_faces} faces, {len(times)} steps, {t0} – {t1}"
            )
            self.ptm_status_label.setStyleSheet("color: #58a6ff;")

            # Pre-fill time range
            if not self.start_edit.text():
                self.start_edit.setText(times[0][1].split(".")[0])
            if not self.stop_edit.text():
                self.stop_edit.setText(times[-1][1].split(".")[0])
            self._update_frame_count()

        # Try to get body radius from mesh
        ply_file = p / "mesh.ply"
        if ply_file.exists():
            self._estimate_body_radius(ply_file)

    def _estimate_body_radius(self, ply_path):
        """Read the first few vertices from mesh.ply to estimate body radius."""
        try:
            import struct
            with open(ply_path, 'rb') as f:
                header = b''
                while True:
                    line = f.readline()
                    header += line
                    if b'end_header' in line:
                        break
                # Read a few vertices (3 doubles each for binary_little_endian)
                radii = []
                for _ in range(min(10, 100)):
                    data = f.read(24)  # 3 * 8 bytes (double)
                    if len(data) < 24:
                        break
                    x, y, z = struct.unpack('<ddd', data)
                    radii.append(math.sqrt(x*x + y*y + z*z))
                if radii:
                    avg_r = sum(radii) / len(radii)
                    # Convert mesh units to km (mesh is typically in meters)
                    if avg_r > 1e6:  # meters
                        self._body_radius_km = avg_r / 1000.0
                    else:  # already km
                        self._body_radius_km = avg_r
        except Exception:
            pass

    # ---------- Time Range ----------

    def _build_time_section(self):
        grp = QGroupBox("Time Range")
        lay = QGridLayout(grp)

        lay.addWidget(QLabel("Start UTC:"), 0, 0)
        self.start_edit = QLineEdit()
        self.start_edit.setPlaceholderText("2026-04-06T22:00:00")
        self.start_edit.textChanged.connect(self._update_frame_count)
        lay.addWidget(self.start_edit, 0, 1)

        lay.addWidget(QLabel("Stop UTC:"), 0, 2)
        self.stop_edit = QLineEdit()
        self.stop_edit.setPlaceholderText("2026-04-07T02:00:00")
        self.stop_edit.textChanged.connect(self._update_frame_count)
        lay.addWidget(self.stop_edit, 0, 3)

        lay.addWidget(QLabel("Cadence (min):"), 1, 0)
        self.cadence_spin = QDoubleSpinBox()
        self.cadence_spin.setRange(0.1, 1440.0)
        self.cadence_spin.setValue(2.0)
        self.cadence_spin.setDecimals(1)
        self.cadence_spin.setSingleStep(0.5)
        self.cadence_spin.valueChanged.connect(self._update_frame_count)
        lay.addWidget(self.cadence_spin, 1, 1)

        lay.addWidget(QLabel("Frames:"), 1, 2)
        self.frame_count_label = QLabel("0")
        self.frame_count_label.setFont(QFont("Consolas", 11, QFont.Weight.Bold))
        lay.addWidget(self.frame_count_label, 1, 3)

        self.time_warning_label = QLabel("")
        self.time_warning_label.setWordWrap(True)
        lay.addWidget(self.time_warning_label, 2, 0, 1, 4)

        self.main_layout.addWidget(grp)

    def _parse_utc(self, text):
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f",
                     "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(text.strip(), fmt)
            except ValueError:
                continue
        return None

    def _update_frame_count(self):
        start = self._parse_utc(self.start_edit.text())
        stop = self._parse_utc(self.stop_edit.text())
        warnings = []

        if start and stop and stop > start:
            cadence_min = self.cadence_spin.value()
            total_min = (stop - start).total_seconds() / 60.0
            n_frames = int(total_min / cadence_min) + 1
            self.frame_count_label.setText(str(n_frames))

            # Check against PTM3D time range
            if self._ptm3d_times:
                ptm_start = self._parse_utc(self._ptm3d_times[0][1])
                ptm_stop = self._parse_utc(self._ptm3d_times[-1][1])
                if ptm_start and ptm_stop:
                    if start < ptm_start or stop > ptm_stop:
                        warnings.append(
                            f"⚠ Time range extends beyond PTM3D data "
                            f"[{self._ptm3d_times[0][1].split('.')[0]} – "
                            f"{self._ptm3d_times[-1][1].split('.')[0]}]"
                        )

            # Cadence warning
            if self._ptm3d_cadence_min > 0 and cadence_min < self._ptm3d_cadence_min:
                # Format both cadences in the most readable unit
                def _fmt_cadence(minutes):
                    if minutes < 1.0:
                        return f"{minutes * 60:.1f} s"
                    elif minutes >= 60.0:
                        return f"{minutes / 60:.2f} h"
                    else:
                        return f"{minutes:.1f} min"
                warnings.append(
                    f"⚠ Camera step ({_fmt_cadence(cadence_min)}) < PTM3D output step "
                    f"({_fmt_cadence(self._ptm3d_cadence_min)}); nearest-neighbor "
                    f"interpolation will be used"
                )
        else:
            self.frame_count_label.setText("—")

        if warnings:
            self.time_warning_label.setText("\n".join(warnings))
            self.time_warning_label.setStyleSheet("color: #f0883e;")
        else:
            self.time_warning_label.setText("")

    # ---------- Camera ----------

    def _build_camera_section(self):
        grp = QGroupBox("Camera")
        main_lay = QVBoxLayout(grp)

        # Preset row
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Preset:"))
        self.preset_combo = QComboBox()
        for name in PRESET_NAMES:
            self.preset_combo.addItem(name)
        self.preset_combo.setCurrentText("Nikon D5 @ 100mm")
        self.preset_combo.currentTextChanged.connect(self._on_preset_changed)
        preset_row.addWidget(self.preset_combo, 1)

        self.load_yaml_btn = QPushButton("Load YAML")
        self.load_yaml_btn.clicked.connect(self._load_camera_yaml)
        preset_row.addWidget(self.load_yaml_btn)
        main_lay.addLayout(preset_row)

        # Resolution scale row
        res_row = QHBoxLayout()
        res_row.addWidget(QLabel("Resolution:"))
        self.resolution_combo = QComboBox()
        for label, _ in RESOLUTION_SCALES:
            self.resolution_combo.addItem(label)
        self.resolution_combo.setCurrentIndex(1)  # 1/4 default
        self.resolution_combo.currentIndexChanged.connect(self._on_resolution_changed)
        res_row.addWidget(self.resolution_combo)
        self.resolution_info_label = QLabel("")
        self.resolution_info_label.setStyleSheet("color: #8b949e;")
        res_row.addWidget(self.resolution_info_label, 1)
        main_lay.addLayout(res_row)

        # Mode radio buttons
        mode_row = QHBoxLayout()
        self.camera_mode_group = QButtonGroup(self)
        self.radio_standard = QRadioButton("Standard Camera")
        self.radio_spice_ik = QRadioButton("SPICE Instrument Kernel")
        self.radio_standard.setChecked(True)
        self.camera_mode_group.addButton(self.radio_standard, 0)
        self.camera_mode_group.addButton(self.radio_spice_ik, 1)
        mode_row.addWidget(self.radio_standard)
        mode_row.addWidget(self.radio_spice_ik)
        main_lay.addLayout(mode_row)

        # Stacked widget for standard vs SPICE IK
        self.camera_stack = QStackedWidget()

        # --- Standard camera page ---
        std_page = QWidget()
        std_lay = QGridLayout(std_page)
        std_lay.setContentsMargins(0, 4, 0, 0)

        std_lay.addWidget(QLabel("Focal length (mm):"), 0, 0)
        self.focal_spin = QDoubleSpinBox()
        self.focal_spin.setRange(1.0, 10000.0)
        self.focal_spin.setValue(100.0)
        self.focal_spin.setDecimals(1)
        self.focal_spin.valueChanged.connect(self._on_intrinsic_changed)
        std_lay.addWidget(self.focal_spin, 0, 1)

        std_lay.addWidget(QLabel("Sensor (mm):"), 0, 2)
        sensor_row = QHBoxLayout()
        self.sensor_w_spin = QDoubleSpinBox()
        self.sensor_w_spin.setRange(1.0, 200.0)
        self.sensor_w_spin.setValue(35.9)
        self.sensor_w_spin.setDecimals(1)
        self.sensor_w_spin.valueChanged.connect(self._on_intrinsic_changed)
        sensor_row.addWidget(self.sensor_w_spin)
        sensor_row.addWidget(QLabel("×"))
        self.sensor_h_spin = QDoubleSpinBox()
        self.sensor_h_spin.setRange(1.0, 200.0)
        self.sensor_h_spin.setValue(23.9)
        self.sensor_h_spin.setDecimals(1)
        self.sensor_h_spin.valueChanged.connect(self._on_intrinsic_changed)
        sensor_row.addWidget(self.sensor_h_spin)
        std_lay.addLayout(sensor_row, 0, 3)

        std_lay.addWidget(QLabel("Pixels:"), 1, 0)
        px_row = QHBoxLayout()
        self.nx_spin = QSpinBox()
        self.nx_spin.setRange(64, 16384)
        self.nx_spin.setValue(1392)
        self.nx_spin.valueChanged.connect(self._on_intrinsic_changed)
        px_row.addWidget(self.nx_spin)
        px_row.addWidget(QLabel("×"))
        self.ny_spin = QSpinBox()
        self.ny_spin.setRange(64, 16384)
        self.ny_spin.setValue(928)
        self.ny_spin.valueChanged.connect(self._on_intrinsic_changed)
        px_row.addWidget(self.ny_spin)
        std_lay.addLayout(px_row, 1, 1)

        std_lay.addWidget(QLabel("FOV:"), 1, 2)
        self.fov_label = QLabel("")
        std_lay.addWidget(self.fov_label, 1, 3)

        std_lay.addWidget(QLabel("Distortion (k1 k2 k3):"), 2, 0)
        dist_row1 = QHBoxLayout()
        self.k1_spin = QDoubleSpinBox()
        self.k2_spin = QDoubleSpinBox()
        self.k3_spin = QDoubleSpinBox()
        for s in (self.k1_spin, self.k2_spin, self.k3_spin):
            s.setRange(-1.0, 1.0)
            s.setDecimals(6)
            s.setSingleStep(0.001)
            s.setValue(0.0)
            dist_row1.addWidget(s)
        std_lay.addLayout(dist_row1, 2, 1, 1, 3)

        std_lay.addWidget(QLabel("Distortion (p1 p2):"), 3, 0)
        dist_row2 = QHBoxLayout()
        self.p1_spin = QDoubleSpinBox()
        self.p2_spin = QDoubleSpinBox()
        for s in (self.p1_spin, self.p2_spin):
            s.setRange(-1.0, 1.0)
            s.setDecimals(6)
            s.setSingleStep(0.001)
            s.setValue(0.0)
            dist_row2.addWidget(s)
        dist_row2.addStretch()
        std_lay.addLayout(dist_row2, 3, 1, 1, 3)

        self.camera_stack.addWidget(std_page)

        # --- SPICE IK page ---
        ik_page = QWidget()
        ik_lay = QGridLayout(ik_page)
        ik_lay.setContentsMargins(0, 4, 0, 0)

        ik_lay.addWidget(QLabel("IK file:"), 0, 0)
        self.ik_file_edit = QLineEdit()
        ik_lay.addWidget(self.ik_file_edit, 0, 1)
        ik_browse = QPushButton("Browse")
        ik_browse.clicked.connect(lambda: self._browse_file(self.ik_file_edit, "SPICE IK (*.ti *.ik);;All (*)"))
        ik_lay.addWidget(ik_browse, 0, 2)

        ik_lay.addWidget(QLabel("CK file:"), 1, 0)
        self.ck_file_edit = QLineEdit()
        ik_lay.addWidget(self.ck_file_edit, 1, 1)
        ck_browse = QPushButton("Browse")
        ck_browse.clicked.connect(lambda: self._browse_file(self.ck_file_edit, "SPICE CK (*.bc *.ck);;All (*)"))
        ik_lay.addWidget(ck_browse, 1, 2)

        ik_lay.addWidget(QLabel("Instrument:"), 2, 0)
        self.instrument_edit = QLineEdit()
        self.instrument_edit.setPlaceholderText("e.g. LRO_LROCNACL")
        ik_lay.addWidget(self.instrument_edit, 2, 1, 1, 2)

        ik_lay.addWidget(QLabel("Pixels:"), 3, 0)
        ik_px_row = QHBoxLayout()
        self.ik_nx_spin = QSpinBox()
        self.ik_nx_spin.setRange(64, 16384)
        self.ik_nx_spin.setValue(1024)
        ik_px_row.addWidget(self.ik_nx_spin)
        ik_px_row.addWidget(QLabel("×"))
        self.ik_ny_spin = QSpinBox()
        self.ik_ny_spin.setRange(64, 16384)
        self.ik_ny_spin.setValue(1024)
        ik_px_row.addWidget(self.ik_ny_spin)
        ik_lay.addLayout(ik_px_row, 3, 1, 1, 2)

        self.camera_stack.addWidget(ik_page)

        self.camera_mode_group.idToggled.connect(
            lambda id, checked: self.camera_stack.setCurrentIndex(id) if checked else None
        )

        main_lay.addWidget(self.camera_stack)
        self.main_layout.addWidget(grp)

        # Apply initial preset
        self._on_preset_changed("Nikon D5 @ 100mm")

    def _on_preset_changed(self, name):
        if self._suppress_preset_change:
            return
        preset = CAMERA_PRESETS.get(name)
        if not preset or name == "Custom":
            return
        self._suppress_preset_change = True
        self.focal_spin.setValue(preset["focal_length_mm"])
        self.sensor_w_spin.setValue(preset["sensor_width_mm"])
        self.sensor_h_spin.setValue(preset["sensor_height_mm"])
        self.k1_spin.setValue(preset["k1"])
        self.k2_spin.setValue(preset["k2"])
        self.k3_spin.setValue(preset["k3"])
        self.p1_spin.setValue(preset["p1"])
        self.p2_spin.setValue(preset["p2"])
        self._suppress_preset_change = False
        self._apply_resolution_scale()
        self._update_fov_label()

    def _on_resolution_changed(self, index):
        if not self._suppress_preset_change:
            self._apply_resolution_scale()

    def _apply_resolution_scale(self):
        """Apply current resolution scale to the active preset's native resolution."""
        preset_name = self.preset_combo.currentText()
        preset = CAMERA_PRESETS.get(preset_name, {})
        native_nx = preset.get("native_nx", self.nx_spin.value())
        native_ny = preset.get("native_ny", self.ny_spin.value())

        idx = self.resolution_combo.currentIndex()
        if idx < len(RESOLUTION_SCALES):
            _, scale = RESOLUTION_SCALES[idx]
        else:
            scale = None

        if scale is not None:
            nx = max(64, int(native_nx * scale))
            ny = max(64, int(native_ny * scale))
            self._suppress_preset_change = True
            self.nx_spin.setValue(nx)
            self.ny_spin.setValue(ny)
            self._suppress_preset_change = False
            self.resolution_info_label.setText(
                f"{nx}×{ny}  (native {native_nx}×{native_ny})")
        else:
            # Custom: leave pixel spinboxes as-is
            self.resolution_info_label.setText(
                f"{self.nx_spin.value()}×{self.ny_spin.value()}  (custom)")

    def _on_intrinsic_changed(self):
        if not self._suppress_preset_change:
            self.preset_combo.setCurrentText("Custom")
            # If user manually edits pixels, switch resolution to Custom too
            custom_idx = next(
                (i for i, (lbl, _) in enumerate(RESOLUTION_SCALES) if lbl == "Custom"),
                len(RESOLUTION_SCALES) - 1)
            self.resolution_combo.setCurrentIndex(custom_idx)
        self._update_fov_label()

    def _update_fov_label(self):
        f = self.focal_spin.value()
        w = self.sensor_w_spin.value()
        h = self.sensor_h_spin.value()
        if f > 0:
            fov_x = 2 * math.degrees(math.atan(w / (2 * f)))
            fov_y = 2 * math.degrees(math.atan(h / (2 * f)))
            self.fov_label.setText(f"{fov_x:.1f}° × {fov_y:.1f}°")
        else:
            self.fov_label.setText("")

    def _load_camera_yaml(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Camera YAML", "", "YAML (*.yaml *.yml);;All (*)")
        if not path:
            return
        try:
            import yaml
            with open(path) as f:
                data = yaml.safe_load(f)
            cam = data.get("camera", data)
            self._suppress_preset_change = True
            if "focal_length_mm" in cam:
                self.focal_spin.setValue(cam["focal_length_mm"])
            if "sensor_mm" in cam and len(cam["sensor_mm"]) == 2:
                self.sensor_w_spin.setValue(cam["sensor_mm"][0])
                self.sensor_h_spin.setValue(cam["sensor_mm"][1])
            if "nx" in cam:
                self.nx_spin.setValue(cam["nx"])
            if "ny" in cam:
                self.ny_spin.setValue(cam["ny"])
            dist = cam.get("distortion", {})
            self.k1_spin.setValue(dist.get("k1", 0.0))
            self.k2_spin.setValue(dist.get("k2", 0.0))
            self.k3_spin.setValue(dist.get("k3", 0.0))
            self.p1_spin.setValue(dist.get("p1", 0.0))
            self.p2_spin.setValue(dist.get("p2", 0.0))
            self.preset_combo.setCurrentText("Custom")
            self._suppress_preset_change = False
            self._update_fov_label()
            if self.console:
                self.console.append_info(f"Loaded camera YAML: {path}")
        except Exception as e:
            QMessageBox.warning(self, "YAML Error", str(e))

    # ---------- Look-At Target ----------

    def _build_lookat_section(self):
        grp = QGroupBox("Look-At Target")
        main_lay = QVBoxLayout(grp)

        # Radio buttons for mode
        mode_row = QHBoxLayout()
        self.lookat_group = QButtonGroup(self)
        self.radio_lookat_fixed = QRadioButton("Fixed XYZ")
        self.radio_lookat_latlon = QRadioButton("Lat/Lon on surface")
        self.radio_lookat_limb_sun = QRadioButton("Track Sun limb")
        self.radio_lookat_limb_earth = QRadioButton("Track primary limb")
        self.radio_lookat_fixed.setChecked(True)
        self.lookat_group.addButton(self.radio_lookat_fixed, 0)
        self.lookat_group.addButton(self.radio_lookat_latlon, 1)
        self.lookat_group.addButton(self.radio_lookat_limb_sun, 2)
        self.lookat_group.addButton(self.radio_lookat_limb_earth, 3)
        mode_row.addWidget(self.radio_lookat_fixed)
        mode_row.addWidget(self.radio_lookat_latlon)
        mode_row.addWidget(self.radio_lookat_limb_sun)
        mode_row.addWidget(self.radio_lookat_limb_earth)
        main_lay.addLayout(mode_row)

        # Stacked widget for mode-specific parameters
        self.lookat_stack = QStackedWidget()

        # --- Fixed XYZ page ---
        fixed_page = QWidget()
        f_lay = QHBoxLayout(fixed_page)
        f_lay.setContentsMargins(0, 4, 0, 0)
        for lbl_text, default in [("X:", 0.0), ("Y:", 0.0), ("Z:", 0.0)]:
            f_lay.addWidget(QLabel(lbl_text))
            spin = QDoubleSpinBox()
            spin.setRange(-1e9, 1e9)
            spin.setDecimals(1)
            spin.setValue(default)
            f_lay.addWidget(spin)
        # Store references: reuse the existing look-at spin boxes from the
        # trajectory section would be messy; create dedicated ones here.
        self.lookat_x = fixed_page.findChildren(QDoubleSpinBox)[0]
        self.lookat_y = fixed_page.findChildren(QDoubleSpinBox)[1]
        self.lookat_z = fixed_page.findChildren(QDoubleSpinBox)[2]
        f_lay.addWidget(QLabel("km"))
        self.lookat_stack.addWidget(fixed_page)

        # --- Lat/Lon page ---
        ll_page = QWidget()
        ll_lay = QHBoxLayout(ll_page)
        ll_lay.setContentsMargins(0, 4, 0, 0)
        ll_lay.addWidget(QLabel("Lat:"))
        self.lookat_lat = QDoubleSpinBox()
        self.lookat_lat.setRange(-90.0, 90.0)
        self.lookat_lat.setDecimals(2)
        self.lookat_lat.setSuffix("°")
        ll_lay.addWidget(self.lookat_lat)
        ll_lay.addWidget(QLabel("Lon:"))
        self.lookat_lon = QDoubleSpinBox()
        self.lookat_lon.setRange(-360.0, 360.0)
        self.lookat_lon.setDecimals(2)
        self.lookat_lon.setSuffix("°")
        ll_lay.addWidget(self.lookat_lon)
        ll_lay.addStretch()
        self.lookat_stack.addWidget(ll_page)

        # --- Limb Sun page (no extra params) ---
        sun_page = QWidget()
        s_lay = QHBoxLayout(sun_page)
        s_lay.setContentsMargins(0, 4, 0, 0)
        s_lay.addWidget(QLabel("Camera tracks the point on the lunar limb closest to the Sun."))
        self.lookat_stack.addWidget(sun_page)

        # --- Limb Earth page (no extra params) ---
        earth_page = QWidget()
        e_lay = QHBoxLayout(earth_page)
        e_lay.setContentsMargins(0, 4, 0, 0)
        e_lay.addWidget(QLabel("Camera tracks the point on the limb closest to the primary body."))
        self.lookat_stack.addWidget(earth_page)

        self.lookat_group.idToggled.connect(
            lambda id, checked: self.lookat_stack.setCurrentIndex(id) if checked else None
        )

        main_lay.addWidget(self.lookat_stack)
        self.main_layout.addWidget(grp)

    # ---------- Trajectory ----------

    def _build_trajectory_section(self):
        grp = QGroupBox("Trajectory")
        main_lay = QVBoxLayout(grp)

        # Radio buttons
        radio_row = QHBoxLayout()
        self.traj_group = QButtonGroup(self)
        self.radio_static = QRadioButton("Static position")
        self.radio_ephemeris = QRadioButton("Ephemeris file")
        self.radio_latlon = QRadioButton("Lat/Lon file")
        self.radio_static.setChecked(True)
        self.traj_group.addButton(self.radio_static, 0)
        self.traj_group.addButton(self.radio_ephemeris, 1)
        self.traj_group.addButton(self.radio_latlon, 2)
        radio_row.addWidget(self.radio_static)
        radio_row.addWidget(self.radio_ephemeris)
        radio_row.addWidget(self.radio_latlon)
        main_lay.addLayout(radio_row)

        self.traj_stack = QStackedWidget()

        # --- Static page ---
        static_page = QWidget()
        s_lay = QGridLayout(static_page)
        s_lay.setContentsMargins(0, 4, 0, 0)

        s_lay.addWidget(QLabel("Position (km):"), 0, 0)
        pos_row = QHBoxLayout()
        self.pos_x = QDoubleSpinBox(); self.pos_x.setRange(-1e9, 1e9); self.pos_x.setDecimals(1); self.pos_x.setValue(0)
        self.pos_y = QDoubleSpinBox(); self.pos_y.setRange(-1e9, 1e9); self.pos_y.setDecimals(1); self.pos_y.setValue(0)
        self.pos_z = QDoubleSpinBox(); self.pos_z.setRange(-1e9, 1e9); self.pos_z.setDecimals(1); self.pos_z.setValue(5000)
        for lbl, s in [("X:", self.pos_x), ("Y:", self.pos_y), ("Z:", self.pos_z)]:
            pos_row.addWidget(QLabel(lbl))
            pos_row.addWidget(s)
        s_lay.addLayout(pos_row, 0, 1, 1, 3)

        self.traj_stack.addWidget(static_page)

        # --- Ephemeris page ---
        eph_page = QWidget()
        e_lay = QGridLayout(eph_page)
        e_lay.setContentsMargins(0, 4, 0, 0)

        e_lay.addWidget(QLabel("File:"), 0, 0)
        self.traj_file_edit = QLineEdit()
        e_lay.addWidget(self.traj_file_edit, 0, 1)
        eph_browse = QPushButton("Browse")
        eph_browse.clicked.connect(lambda: self._browse_file(
            self.traj_file_edit, "SPK (*.bsp);;HORIZONS (*.txt);;All (*)"))
        e_lay.addWidget(eph_browse, 0, 2)

        e_lay.addWidget(QLabel("Format:"), 1, 0)
        self.traj_format_combo = QComboBox()
        self.traj_format_combo.addItems(["spk", "horizons", "propman"])
        e_lay.addWidget(self.traj_format_combo, 1, 1)

        e_lay.addWidget(QLabel("SC ID:"), 1, 2)
        self.sc_id_edit = QLineEdit("-1024")
        e_lay.addWidget(self.sc_id_edit, 1, 3)

        self.traj_stack.addWidget(eph_page)

        # --- Lat/Lon file page ---
        ll_page = QWidget()
        ll_lay = QGridLayout(ll_page)
        ll_lay.setContentsMargins(0, 4, 0, 0)

        ll_lay.addWidget(QLabel("File:"), 0, 0)
        self.latlon_file_edit = QLineEdit()
        self.latlon_file_edit.setPlaceholderText("UTC  lat_deg  lon_deg  alt_km")
        ll_lay.addWidget(self.latlon_file_edit, 0, 1)
        ll_browse = QPushButton("Browse")
        ll_browse.clicked.connect(lambda: self._browse_file(
            self.latlon_file_edit, "Text (*.txt *.csv);;All (*)"))
        ll_lay.addWidget(ll_browse, 0, 2)

        self.traj_stack.addWidget(ll_page)

        self.traj_group.idToggled.connect(
            lambda id, checked: self.traj_stack.setCurrentIndex(id) if checked else None
        )

        main_lay.addWidget(self.traj_stack)
        self.main_layout.addWidget(grp)

    # ---------- Photometry ----------

    def _build_photometry_section(self):
        grp = QGroupBox("Photometry")
        lay = QGridLayout(grp)

        lay.addWidget(QLabel("Model:"), 0, 0)
        self.photo_model_combo = QComboBox()
        self.photo_model_combo.addItems([
            "lommel_seeliger", "lambertian", "minnaert", "hapke"])
        lay.addWidget(self.photo_model_combo, 0, 1)

        lay.addWidget(QLabel("Render:"), 0, 2)
        self.render_mode_combo = QComboBox()
        self.render_mode_combo.addItems(["both", "reflected", "thermal"])
        lay.addWidget(self.render_mode_combo, 0, 3)

        lay.addWidget(QLabel("Albedo:"), 1, 0)
        self.albedo_spin = QDoubleSpinBox()
        self.albedo_spin.setRange(0.0, 1.0)
        self.albedo_spin.setValue(0.12)
        self.albedo_spin.setDecimals(3)
        self.albedo_spin.setSingleStep(0.01)
        lay.addWidget(self.albedo_spin, 1, 1)

        lay.addWidget(QLabel("Emissivity:"), 1, 2)
        self.emissivity_spin = QDoubleSpinBox()
        self.emissivity_spin.setRange(0.0, 1.0)
        self.emissivity_spin.setValue(0.95)
        self.emissivity_spin.setDecimals(3)
        self.emissivity_spin.setSingleStep(0.01)
        lay.addWidget(self.emissivity_spin, 1, 3)

        lay.addWidget(QLabel("Albedo map:"), 2, 0)
        self.albedo_map_edit = QLineEdit()
        lay.addWidget(self.albedo_map_edit, 2, 1, 1, 2)
        am_browse = QPushButton("Browse")
        am_browse.clicked.connect(lambda: self._browse_file(
            self.albedo_map_edit, "Binary (*.bin);;All (*)"))
        lay.addWidget(am_browse, 2, 3)

        self.main_layout.addWidget(grp)

    # ---------- Rendering ----------

    def _build_rendering_section(self):
        grp = QGroupBox("Rendering")
        lay = QGridLayout(grp)

        lay.addWidget(QLabel("Supersample:"), 0, 0)
        self.supersample_combo = QComboBox()
        self.supersample_combo.addItems(["1", "2", "4"])
        self.supersample_combo.setCurrentText("2")
        lay.addWidget(self.supersample_combo, 0, 1)

        self.smooth_check = QCheckBox("Smooth shading")
        self.smooth_check.setChecked(True)
        lay.addWidget(self.smooth_check, 0, 2)

        self.sun_check = QCheckBox("Render Sun")
        self.sun_check.setChecked(True)
        lay.addWidget(self.sun_check, 1, 0)

        self.primary_check = QCheckBox("Render primary body")
        self.primary_check.setChecked(True)
        lay.addWidget(self.primary_check, 1, 1)

        lay.addWidget(QLabel("Target body:"), 2, 0)
        self.target_body_edit = QLineEdit("MOON")
        self.target_body_edit.textChanged.connect(self._on_target_body_changed)
        lay.addWidget(self.target_body_edit, 2, 1)

        lay.addWidget(QLabel("Frame:"), 2, 2)
        self.target_frame_edit = QLineEdit("IAU_MOON")
        lay.addWidget(self.target_frame_edit, 2, 3)

        lay.addWidget(QLabel("SPICE kernels:"), 3, 0)
        self.kernels_edit = QLineEdit(default_cspice_kernels_dir())
        lay.addWidget(self.kernels_edit, 3, 1, 1, 2)
        kern_browse = QPushButton("Browse")
        kern_browse.clicked.connect(lambda: self._browse_dir(self.kernels_edit))
        lay.addWidget(kern_browse, 3, 3)

        self.main_layout.addWidget(grp)

    # ---------- Output ----------

    def _build_output_section(self):
        grp = QGroupBox("Output")
        lay = QGridLayout(grp)

        lay.addWidget(QLabel("Directory:"), 0, 0)
        self.output_dir_edit = QLineEdit("camera_output")
        lay.addWidget(self.output_dir_edit, 0, 1)
        out_browse = QPushButton("Browse")
        out_browse.clicked.connect(lambda: self._browse_dir(self.output_dir_edit))
        lay.addWidget(out_browse, 0, 2)

        self.fits_check = QCheckBox("FITS")
        self.fits_check.setChecked(True)
        lay.addWidget(self.fits_check, 1, 0)

        self.pgm_check = QCheckBox("PGM")
        lay.addWidget(self.pgm_check, 1, 1)

        self.main_layout.addWidget(grp)

    # ---------- Run / Stop ----------

    def _build_run_section(self):
        row = QHBoxLayout()
        self.run_btn = QPushButton("▶  Run Camera Simulation")
        self.run_btn.setMinimumHeight(36)
        self.run_btn.setStyleSheet(
            "QPushButton { background-color: #238636; color: white; font-weight: bold; "
            "border-radius: 6px; padding: 8px 16px; }"
            "QPushButton:hover { background-color: #2ea043; }"
            "QPushButton:disabled { background-color: #21262d; color: #8b949e; }"
        )
        self.run_btn.clicked.connect(self._on_run)
        row.addWidget(self.run_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setMinimumHeight(36)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._on_stop)
        row.addWidget(self.stop_btn)

        self.main_layout.addLayout(row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.main_layout.addWidget(self.progress_bar)

    # ---------- Gallery ----------

    def _build_gallery_section(self):
        grp = QGroupBox("Output Gallery")
        lay = QVBoxLayout(grp)

        # Thumbnail strip
        self.gallery_scroll = QScrollArea()
        self.gallery_scroll.setWidgetResizable(True)
        self.gallery_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.gallery_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.gallery_scroll.setMinimumHeight(80)
        self.gallery_scroll.setMaximumHeight(100)

        self.gallery_widget = QWidget()
        self.gallery_layout = QHBoxLayout(self.gallery_widget)
        self.gallery_layout.setContentsMargins(4, 4, 4, 4)
        self.gallery_layout.setSpacing(4)
        self.gallery_layout.addStretch()
        self.gallery_scroll.setWidget(self.gallery_widget)
        lay.addWidget(self.gallery_scroll)

        self.gallery_status = QLabel("")
        lay.addWidget(self.gallery_status)

        # --- Stretch controls ---
        stretch_row = QGridLayout()
        stretch_row.setColumnStretch(1, 1)
        stretch_row.setColumnStretch(4, 1)

        # Visible: percentile for max normalization
        stretch_row.addWidget(QLabel("Visible %ile:"), 0, 0)
        self.vis_pctile_slider = QSlider(Qt.Orientation.Horizontal)
        self.vis_pctile_slider.setRange(500, 1000)  # 50.0 – 100.0 (×10)
        self.vis_pctile_slider.setValue(995)         # 99.5
        self.vis_pctile_slider.setTickInterval(50)
        self.vis_pctile_slider.valueChanged.connect(self._on_stretch_changed)
        stretch_row.addWidget(self.vis_pctile_slider, 0, 1)
        self.vis_pctile_label = QLabel("99.5")
        self.vis_pctile_label.setMinimumWidth(35)
        stretch_row.addWidget(self.vis_pctile_label, 0, 2)

        # Earthshine: asinh softening (log scale)
        stretch_row.addWidget(QLabel("Earthshine boost:"), 0, 3)
        self.es_boost_slider = QSlider(Qt.Orientation.Horizontal)
        self.es_boost_slider.setRange(0, 100)  # maps to softening 1e-3 → 1e-8
        self.es_boost_slider.setValue(70)       # default ~ 5e-6
        self.es_boost_slider.valueChanged.connect(self._on_stretch_changed)
        stretch_row.addWidget(self.es_boost_slider, 0, 4)
        self.es_boost_label = QLabel("70")
        self.es_boost_label.setMinimumWidth(25)
        stretch_row.addWidget(self.es_boost_label, 0, 5)

        # Temperature range
        stretch_row.addWidget(QLabel("Temp (K):"), 1, 0)
        temp_row = QHBoxLayout()
        self.temp_min_spin = QSpinBox()
        self.temp_min_spin.setRange(0, 500)
        self.temp_min_spin.setValue(80)
        self.temp_min_spin.valueChanged.connect(self._on_stretch_changed)
        temp_row.addWidget(self.temp_min_spin)
        temp_row.addWidget(QLabel("–"))
        self.temp_max_spin = QSpinBox()
        self.temp_max_spin.setRange(50, 1000)
        self.temp_max_spin.setValue(380)
        self.temp_max_spin.valueChanged.connect(self._on_stretch_changed)
        temp_row.addWidget(self.temp_max_spin)
        stretch_row.addLayout(temp_row, 1, 1)

        lay.addLayout(stretch_row)

        # --- 3-panel preview of selected frame ---
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumHeight(200)
        self.preview_label.setStyleSheet("background-color: #0d1117;")
        self.preview_label.setText("Click a thumbnail to preview")
        self.preview_label.setStyleSheet(
            "background-color: #0d1117; color: #8b949e; font-size: 11px;")
        lay.addWidget(self.preview_label)

        # --- Export animation controls ---
        export_row = QHBoxLayout()
        export_row.addWidget(QLabel("Animation DPI:"))
        self.anim_dpi_spin = QSpinBox()
        self.anim_dpi_spin.setRange(50, 300)
        self.anim_dpi_spin.setValue(120)
        self.anim_dpi_spin.setSuffix(" dpi")
        export_row.addWidget(self.anim_dpi_spin)

        export_row.addWidget(QLabel("FPS:"))
        self.anim_fps_spin = QSpinBox()
        self.anim_fps_spin.setRange(5, 30)
        self.anim_fps_spin.setValue(10)
        export_row.addWidget(self.anim_fps_spin)

        self.export_gif_btn = QPushButton("Export GIF")
        self.export_gif_btn.clicked.connect(lambda: self._export_animation("gif"))
        export_row.addWidget(self.export_gif_btn)

        self.export_mp4_btn = QPushButton("Export MP4")
        self.export_mp4_btn.clicked.connect(lambda: self._export_animation("mp4"))
        export_row.addWidget(self.export_mp4_btn)

        export_row.addStretch()
        lay.addLayout(export_row)

        self.main_layout.addWidget(grp)

    def _on_stretch_changed(self):
        # Update labels
        pctile = self.vis_pctile_slider.value() / 10.0
        self.vis_pctile_label.setText(f"{pctile:.1f}")
        self.es_boost_label.setText(str(self.es_boost_slider.value()))
        # Re-render preview if a frame is selected
        if self._selected_fits:
            self._update_preview(self._selected_fits)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    # Primary body auto-detection from target body name
    _PRIMARY_MAP = {
        "MOON": "EARTH", "EUROPA": "JUPITER", "IO": "JUPITER",
        "GANYMEDE": "JUPITER", "CALLISTO": "JUPITER",
        "TITAN": "SATURN", "ENCELADUS": "SATURN", "RHEA": "SATURN",
        "MIMAS": "SATURN", "DIONE": "SATURN", "TETHYS": "SATURN",
        "IAPETUS": "SATURN", "HYPERION": "SATURN",
        "PHOBOS": "MARS", "DEIMOS": "MARS",
        "TRITON": "NEPTUNE", "ARIEL": "URANUS", "MIRANDA": "URANUS",
        "OBERON": "URANUS", "TITANIA": "URANUS", "UMBRIEL": "URANUS",
        "CHARON": "PLUTO",
    }

    def _on_target_body_changed(self, body_name):
        body = body_name.strip().upper()
        # Auto-derive frame
        self.target_frame_edit.setText(f"IAU_{body}")
        # Update primary body label
        primary = self._PRIMARY_MAP.get(body, "")
        if primary:
            self.primary_check.setText(f"Render primary body ({primary})")
            self.radio_lookat_limb_earth.setText(f"Track {primary} limb")
        else:
            self.primary_check.setText("Render primary body")
            self.radio_lookat_limb_earth.setText("Track primary limb")

    def _browse_file(self, line_edit, filter_str):
        path, _ = QFileDialog.getOpenFileName(self, "Select File", "", filter_str)
        if path:
            line_edit.setText(path)

    def _browse_dir(self, line_edit):
        d = QFileDialog.getExistingDirectory(self, "Select Directory")
        if d:
            line_edit.setText(d)

    # ------------------------------------------------------------------
    # Build CLI command
    # ------------------------------------------------------------------

    def _generate_render_times(self):
        start = self._parse_utc(self.start_edit.text())
        stop = self._parse_utc(self.stop_edit.text())
        if not start or not stop or stop <= start:
            return []
        cadence = timedelta(minutes=self.cadence_spin.value())
        times = []
        t = start
        while t <= stop:
            times.append(t.strftime("%Y-%m-%dT%H:%M:%S.000"))
            t += cadence
        return times

    def build_command(self):
        """Build the ptm_camera_sim CLI command from current GUI state."""
        # Find executable
        exe = self._find_camera_executable()
        if not exe:
            return None, "Cannot find ptm_camera_sim executable"

        ptm_dir = self.ptm_dir_edit.text().strip()
        if not ptm_dir:
            return None, "PTM3D output directory not set"

        render_times = self._generate_render_times()
        if not render_times:
            return None, "Invalid time range"

        cmd = [str(exe), "--ptm-output", ptm_dir]

        for t in render_times:
            cmd += ["--time", t]

        # Camera
        if self.radio_standard.isChecked():
            cmd += ["--focal-mm", str(self.focal_spin.value())]
            cmd += ["--sensor-mm", str(self.sensor_w_spin.value()),
                    str(self.sensor_h_spin.value())]
            cmd += ["--pixels", str(self.nx_spin.value()),
                    str(self.ny_spin.value())]
            k1 = self.k1_spin.value()
            k2 = self.k2_spin.value()
            k3 = self.k3_spin.value()
            p1 = self.p1_spin.value()
            p2 = self.p2_spin.value()
            if any(v != 0 for v in [k1, k2, k3, p1, p2]):
                cmd += ["--distortion", str(k1), str(k2), str(k3),
                        str(p1), str(p2)]
        else:  # SPICE IK
            ik = self.ik_file_edit.text().strip()
            if ik:
                cmd += ["--ik-kernel", ik]
            ck = self.ck_file_edit.text().strip()
            if ck:
                cmd += ["--ck-kernel", ck]
            inst = self.instrument_edit.text().strip()
            if inst:
                cmd += ["--instrument", inst]
            cmd += ["--pixels", str(self.ik_nx_spin.value()),
                    str(self.ik_ny_spin.value())]

        # Look-at target
        lookat_id = self.lookat_group.checkedId()
        if lookat_id == 0:  # Fixed XYZ
            cmd += ["--look-at",
                    str(self.lookat_x.value()),
                    str(self.lookat_y.value()),
                    str(self.lookat_z.value())]
        elif lookat_id == 1:  # Lat/Lon on surface
            cmd += ["--look-at-latlon",
                    str(self.lookat_lat.value()),
                    str(self.lookat_lon.value())]
        elif lookat_id == 2:  # Track Sun limb
            cmd += ["--look-at-limb", "sun"]
        elif lookat_id == 3:  # Track primary body limb
            cmd += ["--look-at-limb", "primary"]

        # Trajectory
        traj_id = self.traj_group.checkedId()
        if traj_id == 0:  # Static
            cmd += ["--cam-pos",
                    str(self.pos_x.value()),
                    str(self.pos_y.value()),
                    str(self.pos_z.value())]
        elif traj_id == 1:  # Ephemeris
            traj_file = self.traj_file_edit.text().strip()
            if traj_file:
                cmd += ["--trajectory", traj_file]
                cmd += ["--traj-format", self.traj_format_combo.currentText()]
                cmd += ["--traj-spacecraft", self.sc_id_edit.text().strip()]
        elif traj_id == 2:  # Lat/Lon file
            ll_file = self.latlon_file_edit.text().strip()
            if ll_file:
                tmp = self._convert_latlon_file(ll_file)
                if tmp:
                    cmd += ["--trajectory", tmp, "--traj-format", "horizons"]

        # Photometry
        cmd += ["--photometric", self.photo_model_combo.currentText()]
        cmd += ["--render-mode", self.render_mode_combo.currentText()]
        cmd += ["--albedo", str(self.albedo_spin.value())]
        cmd += ["--emissivity", str(self.emissivity_spin.value())]
        albedo_map = self.albedo_map_edit.text().strip()
        if albedo_map:
            cmd += ["--albedo-map", albedo_map]

        # Rendering
        cmd += ["--supersample", self.supersample_combo.currentText()]
        if not self.smooth_check.isChecked():
            cmd += ["--no-smooth-shading"]
        if self.sun_check.isChecked():
            cmd += ["--render-sun"]
        if self.primary_check.isChecked():
            cmd += ["--render-primary"]
        cmd += ["--target-body", self.target_body_edit.text().strip()]
        cmd += ["--target-frame", self.target_frame_edit.text().strip()]
        kernels = self.kernels_edit.text().strip()
        if kernels:
            cmd += ["--kernels-path", kernels]

        # Output
        cmd += ["--output-dir", self.output_dir_edit.text().strip()]
        if not self.fits_check.isChecked():
            cmd += ["--no-fits"]
        if self.pgm_check.isChecked():
            cmd += ["--pgm"]

        return cmd, None

    def _find_camera_executable(self):
        """Find ptm_camera_sim in build/bin/ relative to the project root."""
        candidates = [
            Path(__file__).parent.parent.parent / "build" / "bin" / "ptm_camera_sim",
            Path(__file__).parent.parent / "build" / "bin" / "ptm_camera_sim",
            Path("build/bin/ptm_camera_sim"),
        ]
        for p in candidates:
            if p.exists():
                return p
        return None

    def _convert_latlon_file(self, filepath):
        """Convert a lat/lon/alt text file to HORIZONS format (temp file)."""
        try:
            lines = []
            with open(filepath) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) < 4:
                        continue
                    utc = parts[0]
                    lat = math.radians(float(parts[1]))
                    lon = math.radians(float(parts[2]))
                    alt = float(parts[3])
                    r = self._body_radius_km + alt
                    x = r * math.cos(lat) * math.cos(lon)
                    y = r * math.cos(lat) * math.sin(lon)
                    z = r * math.sin(lat)
                    lines.append(f"{utc}  {x:.6f}  {y:.6f}  {z:.6f}  0.0  0.0  0.0")

            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False, prefix="cam_traj_")
            for l in lines:
                tmp.write(l + "\n")
            tmp.close()
            if self.console:
                self.console.append_info(
                    f"Converted {len(lines)} lat/lon points → {tmp.name}")
            return tmp.name
        except Exception as e:
            if self.console:
                self.console.append_error(f"Lat/Lon conversion error: {e}")
            return None

    # ------------------------------------------------------------------
    # Run / Stop
    # ------------------------------------------------------------------

    def _on_run(self):
        cmd, err = self.build_command()
        if err:
            QMessageBox.warning(self, "Validation Error", err)
            return

        # Clear gallery
        self._clear_gallery()
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._running = True

        if self.console:
            self.console.append_header("Camera Simulation")
            self.console.append_info("Command: " + " ".join(cmd[:10]) + " ...")

        self.run_requested.emit(cmd)

    def _on_stop(self):
        self._running = False
        self.stop_requested.emit()
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def on_simulation_finished(self, exit_code):
        self._running = False
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        if self.console:
            if exit_code == 0:
                self.console.append_success("Camera simulation completed")
            else:
                self.console.append_error(
                    f"Camera simulation failed (exit code {exit_code})")

    def on_output_line(self, line):
        """Parse stdout lines for progress and frame completion."""
        import re
        # Progress: "=== Frame N/M ==="
        m = re.match(r"=== Frame (\d+)/(\d+) ===", line)
        if m:
            current = int(m.group(1))
            total = int(m.group(2))
            self.progress_bar.setMaximum(total)
            self.progress_bar.setValue(current)
            self.gallery_status.setText(f"{current}/{total} frames")

        # Frame written: "Wrote FITS: path"
        if "Wrote FITS:" in line:
            fits_path = line.split("Wrote FITS:")[-1].strip()
            self._add_thumbnail(fits_path)

    # ------------------------------------------------------------------
    # Gallery
    # ------------------------------------------------------------------

    def _clear_gallery(self):
        while self.gallery_layout.count() > 1:  # keep the stretch
            item = self.gallery_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.gallery_status.setText("")
        self._rendered_fits = []
        self._selected_fits = None
        self.preview_label.setPixmap(QPixmap())
        self.preview_label.setText("Click a thumbnail to preview")

    def _add_thumbnail(self, fits_path):
        """Load a FITS radiance HDU and add a gamma-corrected thumbnail."""
        try:
            import numpy as np
            from astropy.io import fits as pyfits

            self._rendered_fits.append(fits_path)

            with pyfits.open(fits_path) as hdul:
                rad = hdul[0].data
                face = hdul[2].data if len(hdul) > 2 else None

            if rad is None:
                return

            pctile = self.vis_pctile_slider.value() / 10.0
            mask = face >= 0 if face is not None else rad > 0
            vmax = np.nanpercentile(rad[mask], pctile) if mask.any() else 1.0
            if vmax <= 0:
                vmax = 1.0
            img = np.power(np.clip(rad / vmax, 0, 1), 1 / 2.2)
            img = (img * 255).astype(np.uint8)

            h, w = img.shape
            qimg = QImage(img.data, w, h, w, QImage.Format.Format_Grayscale8)
            pixmap = QPixmap.fromImage(qimg)
            pixmap = pixmap.scaled(
                QSize(100, 67), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation)

            label = QLabel()
            label.setPixmap(pixmap)
            label.setToolTip(Path(fits_path).name)
            label.setCursor(Qt.CursorShape.PointingHandCursor)
            # Click to preview
            fpath = fits_path  # capture for lambda
            label.mousePressEvent = lambda ev, p=fpath: self._update_preview(p)

            idx = self.gallery_layout.count() - 1
            self.gallery_layout.insertWidget(idx, label)

            self.gallery_scroll.horizontalScrollBar().setValue(
                self.gallery_scroll.horizontalScrollBar().maximum())

            # Auto-preview the first frame
            if len(self._rendered_fits) == 1:
                self._update_preview(fits_path)

        except Exception as e:
            logger.warning(f"Thumbnail error: {e}")

    def _update_preview(self, fits_path):
        """Render a 3-panel preview of the selected frame using current stretch."""
        self._selected_fits = fits_path
        try:
            import numpy as np
            from astropy.io import fits as pyfits
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import io

            with pyfits.open(fits_path) as hdul:
                rad = hdul[0].data
                face = hdul[2].data if len(hdul) > 2 else None
                temp = hdul[6].data if len(hdul) > 6 else None
                h = hdul[0].header
                utc = h.get('UTC_TIME', '').split('.')[0]

            if rad is None:
                return

            mask = face >= 0 if face is not None else np.ones_like(rad, bool)
            sun_mask = face == -2 if face is not None else np.zeros_like(rad, bool)
            earth_mask = face == -3 if face is not None else np.zeros_like(rad, bool)
            ny, nx = rad.shape

            # Read stretch parameters
            pctile = self.vis_pctile_slider.value() / 10.0
            boost_val = self.es_boost_slider.value()
            # Map 0-100 → softening 1e-3 to 1e-8 (log scale)
            softening = 10.0 ** (-3.0 - 5.0 * boost_val / 100.0)
            t_min = self.temp_min_spin.value()
            t_max = self.temp_max_spin.value()

            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 2.8), dpi=100)

            # --- Visible ---
            vis = np.zeros((ny, nx))
            vis[mask] = rad[mask]
            vmax = np.nanpercentile(rad[mask], pctile) if mask.any() else 1.0
            if vmax <= 0: vmax = 1.0
            vis_gamma = np.power(np.clip(vis / vmax, 0, 1), 1 / 2.2)
            if sun_mask.any(): vis_gamma[sun_mask] = 1.0
            if earth_mask.any():
                vis_gamma[earth_mask] = np.power(
                    np.clip(rad[earth_mask] / vmax, 0, 1), 1 / 2.2)
            ax1.imshow(vis_gamma, cmap='gray', vmin=0, vmax=1,
                       origin='upper', aspect='equal')
            ax1.set_title(f'Visible (p{pctile:.0f})', fontsize=8, color='white')
            ax1.axis('off')

            # --- Earthshine enhanced ---
            es_img = np.zeros((ny, nx))
            es_img[mask] = rad[mask]
            if earth_mask.any(): es_img[earth_mask] = rad[earth_mask]
            if sun_mask.any(): es_img[sun_mask] = vmax
            top = max(vmax, 1.0)
            stretched = np.arcsinh(es_img / softening) / np.arcsinh(top / softening)
            stretched = np.power(np.clip(stretched, 0, 1), 0.5)
            rgb = np.zeros((ny, nx, 3))
            rgb[:, :, 0] = stretched * 0.70
            rgb[:, :, 1] = stretched * 0.80
            rgb[:, :, 2] = stretched * 1.00
            bright = (es_img > 0.5) & mask
            rgb[bright, 0] = np.clip(stretched[bright], 0, 1)
            rgb[bright, 1] = np.clip(stretched[bright] * 0.98, 0, 1)
            rgb[bright, 2] = np.clip(stretched[bright] * 0.95, 0, 1)
            if earth_mask.any():
                e_val = np.clip(rad[earth_mask] / top, 0, 1)
                rgb[earth_mask, 0] = 0.6 * e_val + 0.2
                rgb[earth_mask, 1] = 0.7 * e_val + 0.2
                rgb[earth_mask, 2] = 1.0
            if sun_mask.any(): rgb[sun_mask, :] = 1.0
            rgb = np.clip(rgb, 0, 1)
            ax2.imshow(rgb, origin='upper', aspect='equal')
            ax2.set_title(f'Earthshine (boost={boost_val})',
                          fontsize=8, color='white')
            ax2.axis('off')

            # --- Temperature ---
            if temp is not None:
                t_img = np.full((ny, nx), np.nan)
                t_img[mask] = temp[mask]
                ax3.imshow(t_img, cmap='magma', vmin=t_min, vmax=t_max,
                           origin='upper', aspect='equal')
                ax3.set_title(f'Temperature ({t_min}–{t_max} K)',
                              fontsize=8, color='white')
            else:
                ax3.text(0.5, 0.5, 'No temp data', ha='center', va='center',
                         color='white', transform=ax3.transAxes)
            ax3.axis('off')

            fig.suptitle(utc, fontsize=9, color='white', y=0.98)
            fig.patch.set_facecolor('#0d1117')
            fig.subplots_adjust(wspace=0.04)

            # Render to QPixmap
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight',
                        facecolor='#0d1117', pad_inches=0.05)
            plt.close(fig)
            buf.seek(0)

            pixmap = QPixmap()
            pixmap.loadFromData(buf.read())
            self.preview_label.setPixmap(pixmap.scaled(
                self.preview_label.width(), 280,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))

        except Exception as e:
            logger.warning(f"Preview error: {e}")
            self.preview_label.setText(f"Preview error: {e}")

    def _export_animation(self, fmt):
        """Export rendered FITS frames as a GIF or MP4 animation."""
        if not self._rendered_fits:
            QMessageBox.warning(self, "Export", "No rendered frames to export.")
            return

        out_dir = self.output_dir_edit.text().strip() or "camera_output"
        default_name = f"animation.{fmt}"
        save_path, _ = QFileDialog.getSaveFileName(
            self, f"Save {fmt.upper()}", str(Path(out_dir) / default_name),
            f"{fmt.upper()} (*.{fmt})")
        if not save_path:
            return

        if self.console:
            self.console.append_info(f"Exporting {len(self._rendered_fits)} frames → {save_path}")

        try:
            import numpy as np
            from astropy.io import fits as pyfits
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import imageio.v2 as imageio

            dpi = self.anim_dpi_spin.value()
            fps = self.anim_fps_spin.value()

            # Read stretch settings
            pctile = self.vis_pctile_slider.value() / 10.0
            boost_val = self.es_boost_slider.value()
            softening = 10.0 ** (-3.0 - 5.0 * boost_val / 100.0)
            t_min = self.temp_min_spin.value()
            t_max = self.temp_max_spin.value()

            # First pass: find global visible max
            max_reflected = 0.0
            for f in self._rendered_fits:
                with pyfits.open(f) as hdul:
                    rad = hdul[0].data; face = hdul[2].data
                    mask = (face >= 0) & (rad > 0.1)
                    if mask.any():
                        mr = np.nanpercentile(rad[mask], pctile)
                        if mr > max_reflected: max_reflected = mr
            if max_reflected < 1e-6: max_reflected = 1.0

            # Render frames
            images = []
            total = len(self._rendered_fits)
            for i, f in enumerate(self._rendered_fits):
                with pyfits.open(f) as hdul:
                    rad = hdul[0].data; face = hdul[2].data
                    temp = hdul[6].data if len(hdul) > 6 else None
                    dist_data = hdul[7].data if len(hdul) > 7 else None
                    h = hdul[0].header
                    utc = h.get('UTC_TIME', '').split('.')[0]

                mask = face >= 0
                sun_mask = face == -2; earth_mask = face == -3
                n_sun = np.sum(sun_mask); n_earth = np.sum(earth_mask)
                ny, nx = rad.shape

                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 3.8), dpi=dpi)

                # Visible
                vis = np.zeros((ny, nx))
                vis[mask] = rad[mask]
                vis_gamma = np.power(np.clip(vis / max_reflected, 0, 1), 1/2.2)
                if n_sun > 0: vis_gamma[sun_mask] = 1.0
                if n_earth > 0:
                    vis_gamma[earth_mask] = np.power(np.clip(rad[earth_mask] / max_reflected, 0, 1), 1/2.2)
                ax1.imshow(vis_gamma, cmap='gray', vmin=0, vmax=1, origin='upper', aspect='equal')
                t1 = 'Visible'
                extras = []
                if n_sun > 0: extras.append('Sun')
                if n_earth > 0: extras.append('Earth')
                if extras: t1 += '  (' + ', '.join(extras) + ')'
                ax1.set_title(t1, fontsize=9, color='white'); ax1.axis('off')

                # Earthshine
                es_img = np.zeros((ny, nx))
                es_img[mask] = rad[mask]
                if n_earth > 0: es_img[earth_mask] = rad[earth_mask]
                if n_sun > 0: es_img[sun_mask] = max_reflected
                top = max(max_reflected, 1.0)
                stretched = np.arcsinh(es_img / softening) / np.arcsinh(top / softening)
                stretched = np.power(np.clip(stretched, 0, 1), 0.5)
                rgb = np.zeros((ny, nx, 3))
                rgb[:,:,0] = stretched * 0.70
                rgb[:,:,1] = stretched * 0.80
                rgb[:,:,2] = stretched * 1.00
                bright = (es_img > 0.5) & mask
                rgb[bright, 0] = np.clip(stretched[bright], 0, 1)
                rgb[bright, 1] = np.clip(stretched[bright] * 0.98, 0, 1)
                rgb[bright, 2] = np.clip(stretched[bright] * 0.95, 0, 1)
                if n_earth > 0:
                    e_val = np.clip(rad[earth_mask] / top, 0, 1)
                    rgb[earth_mask, 0] = 0.6 * e_val + 0.2
                    rgb[earth_mask, 1] = 0.7 * e_val + 0.2
                    rgb[earth_mask, 2] = 1.0
                if n_sun > 0: rgb[sun_mask, :] = 1.0
                rgb = np.clip(rgb, 0, 1)
                ax2.imshow(rgb, origin='upper', aspect='equal')
                ax2.set_title('Earthshine (enhanced)', fontsize=9, color='white'); ax2.axis('off')

                # Temperature
                if temp is not None:
                    t_img = np.full((ny, nx), np.nan)
                    t_img[mask] = temp[mask]
                    ax3.imshow(t_img, cmap='magma', vmin=t_min, vmax=t_max, origin='upper', aspect='equal')
                ax3.set_title('Surface Temperature (K)', fontsize=9, color='white'); ax3.axis('off')

                min_dist = np.nanmin(dist_data[mask]) if (dist_data is not None and mask.any()) else 0
                fig.suptitle(f'{utc}  \u00b7  {min_dist:.0f} km',
                             fontsize=9, color='white', y=0.97)
                fig.patch.set_facecolor('black')
                fig.subplots_adjust(wspace=0.04)

                fig.canvas.draw()
                img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                images.append(img_array)
                plt.close(fig)

                if i % 20 == 0 and self.console:
                    self.console.append(f"  Rendering frame {i+1}/{total}...")

            # Write output
            if fmt == "gif":
                imageio.mimsave(save_path, images, duration=1.0/fps, loop=0)
            else:
                writer = imageio.get_writer(save_path, fps=fps, codec='libx264',
                                             quality=8, pixelformat='yuv420p')
                for img in images:
                    writer.append_data(img)
                writer.close()

            size_mb = Path(save_path).stat().st_size / 1024 / 1024
            if self.console:
                self.console.append_success(
                    f"Exported {fmt.upper()}: {save_path} ({size_mb:.1f} MB)")

        except Exception as e:
            if self.console:
                self.console.append_error(f"Export error: {e}")
            else:
                QMessageBox.warning(self, "Export Error", str(e))
