"""
DEM Configuration Panel: 4-page nav rail with Preview, Convert, Inspect, Refine.
"""

from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QStackedWidget, QListWidget,
    QListWidgetItem, QLabel, QGroupBox, QFormLayout, QPushButton,
    QSpinBox, QDoubleSpinBox, QCheckBox, QLineEdit, QFileDialog,
    QProgressBar, QComboBox, QMessageBox
)
from PyQt6.QtCore import pyqtSignal, Qt, QSize

from dem_panels.dem_preview_panel import DEMPreviewPanel
from dem_panels.dem_mesh_panel import DEMMeshPanel
from dem_panels.dem_results_panel import DEMResultsPanel
from dem_core.dem_runner import DEMConvertThread, DEMRefineThread
from dem_core.dem_tooltips import TOOLTIPS


class DEMConfigurationPanel(QWidget):
    """Main DEM configuration panel with nav rail and 4 pages."""

    config_changed = pyqtSignal()
    mesh_converted = pyqtSignal(str)  # path to converted mesh

    def __init__(self, config_manager, parent=None):
        super().__init__(parent)
        self._cm = config_manager
        self._convert_thread = None
        self._refine_thread = None
        self._init_ui()

    def _init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Left nav rail
        self._nav = QListWidget()
        self._nav.setObjectName("NavRail")
        self._nav.setFixedWidth(130)
        self._nav.setIconSize(QSize(24, 24))

        pages = ["Preview", "Convert", "Inspect", "Refine"]
        for name in pages:
            item = QListWidgetItem(name)
            item.setSizeHint(QSize(120, 40))
            self._nav.addItem(item)

        self._nav.setCurrentRow(0)
        self._nav.currentRowChanged.connect(self._on_page_changed)
        layout.addWidget(self._nav)

        # Right stacked widget
        self._stack = QStackedWidget()

        # Page 0: Preview
        self._preview = DEMPreviewPanel(self._cm)
        self._preview.dem_loaded.connect(self._on_dem_loaded)
        self._preview.roi_selected.connect(self._on_roi_selected)
        self._stack.addWidget(self._preview)

        # Page 1: Convert
        self._convert_page = self._create_convert_page()
        self._stack.addWidget(self._convert_page)

        # Page 2: Inspect (stats-only; 3D viewer is in the right splitter pane)
        self._mesh_panel = DEMMeshPanel()
        self._mesh_panel.hide_3d_viewer()
        self._stack.addWidget(self._mesh_panel)

        # Page 3: Refine
        self._refine_page = self._create_refine_page()
        self._stack.addWidget(self._refine_page)

        layout.addWidget(self._stack, 1)

    def _create_convert_page(self) -> QWidget:
        """Create the Convert parameters page (Delaunay triangulation)."""
        from PyQt6.QtWidgets import QScrollArea

        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)

        # Scrollable content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        content = QWidget()
        form_layout = QVBoxLayout(content)
        form_layout.setContentsMargins(12, 12, 12, 12)

        title = QLabel("Delaunay Mesh Conversion")
        title.setStyleSheet("color: #00bcd4; font-weight: bold; font-size: 14px;")
        form_layout.addWidget(title)

        desc = QLabel(
            "Convert a GeoTIFF DEM to a triangulated PLY mesh using\n"
            "Constrained Delaunay Triangulation. Select a DEM in the Preview page first."
        )
        desc.setStyleSheet("color: #a0a0a0;")
        desc.setWordWrap(True)
        form_layout.addWidget(desc)

        # --- Grid & Body ---
        grid_group = QGroupBox("Grid & Body")
        form = QFormLayout(grid_group)

        self._downsample_spin = QSpinBox()
        self._downsample_spin.setRange(1, 200)
        self._downsample_spin.setValue(10)
        self._downsample_spin.setToolTip(TOOLTIPS.get("conversion.downsample", ""))
        self._downsample_spin.valueChanged.connect(
            lambda v: self._cm.set_value("conversion.downsample", v)
        )
        form.addRow("Downsample Factor:", self._downsample_spin)

        self._body_combo = QComboBox()
        from dem_core.dem_config_manager import DEMConfigManager
        for name in DEMConfigManager.BODY_RADII:
            self._body_combo.addItem(name)
        self._body_combo.setCurrentText("Moon")
        self._body_combo.currentTextChanged.connect(self._on_body_changed)
        form.addRow("Body:", self._body_combo)

        self._radius_spin = QDoubleSpinBox()
        self._radius_spin.setRange(1000.0, 100000000.0)
        self._radius_spin.setValue(1737400.0)
        self._radius_spin.setDecimals(0)
        self._radius_spin.setSuffix(" m")
        self._radius_spin.setToolTip(TOOLTIPS.get("input.body_radius_m", ""))
        self._radius_spin.valueChanged.connect(
            lambda v: self._cm.set_value("input.body_radius_m", v)
        )
        form.addRow("Body Radius:", self._radius_spin)

        form_layout.addWidget(grid_group)

        # --- ROI Status ---
        roi_group = QGroupBox("ROI (set in Preview page)")
        roi_form = QFormLayout(roi_group)

        self._roi_label = QLabel("ROI: disabled")
        self._roi_label.setStyleSheet("color: #a0a0a0;")
        roi_form.addRow("Status:", self._roi_label)

        self._core_frac_spin = QDoubleSpinBox()
        self._core_frac_spin.setRange(0.0, 1.0)
        self._core_frac_spin.setValue(0.15)
        self._core_frac_spin.setDecimals(2)
        self._core_frac_spin.setSingleStep(0.05)
        self._core_frac_spin.setToolTip(TOOLTIPS.get("conversion.roi_core_frac", ""))
        self._core_frac_spin.valueChanged.connect(
            lambda v: self._cm.set_value("conversion.roi_core_frac", v)
        )
        roi_form.addRow("Core Fraction:", self._core_frac_spin)

        self._density_exp_spin = QDoubleSpinBox()
        self._density_exp_spin.setRange(0.1, 10.0)
        self._density_exp_spin.setValue(2.0)
        self._density_exp_spin.setDecimals(1)
        self._density_exp_spin.setSingleStep(0.5)
        self._density_exp_spin.setToolTip(TOOLTIPS.get("conversion.density_exp", ""))
        self._density_exp_spin.valueChanged.connect(
            lambda v: self._cm.set_value("conversion.density_exp", v)
        )
        roi_form.addRow("Density Exponent:", self._density_exp_spin)

        form_layout.addWidget(roi_group)

        # --- Triangulation ---
        tri_group = QGroupBox("Triangulation")
        tri_form = QFormLayout(tri_group)

        self._max_edge_combo = QComboBox()
        self._max_edge_combo.addItems(["Auto (from density)", "Disabled (no filtering)", "Custom"])
        self._max_edge_combo.currentIndexChanged.connect(self._on_max_edge_mode_changed)
        tri_form.addRow("Max Edge Mode:", self._max_edge_combo)

        self._max_edge_spin = QDoubleSpinBox()
        self._max_edge_spin.setRange(1.0, 100000.0)
        self._max_edge_spin.setValue(500.0)
        self._max_edge_spin.setDecimals(1)
        self._max_edge_spin.setSuffix(" m")
        self._max_edge_spin.setToolTip(TOOLTIPS.get("conversion.max_edge_m", ""))
        self._max_edge_spin.setEnabled(False)
        self._max_edge_spin.valueChanged.connect(
            lambda v: self._cm.set_value("conversion.max_edge_m", v)
        )
        tri_form.addRow("Max Edge Length:", self._max_edge_spin)

        form_layout.addWidget(tri_group)

        # --- Coordinate Corrections ---
        coord_group = QGroupBox("Coordinate Corrections")
        coord_form = QFormLayout(coord_group)

        self._south_pole_check = QCheckBox("South pole corrections")
        self._south_pole_check.setToolTip(TOOLTIPS.get("conversion.south_pole_coords", ""))
        self._south_pole_check.toggled.connect(
            lambda v: self._cm.set_value("conversion.south_pole_coords", v)
        )
        coord_form.addRow(self._south_pole_check)

        self._swap_axes_check = QCheckBox("Swap X/Y axes")
        self._swap_axes_check.setToolTip(TOOLTIPS.get("conversion.swap_axes", ""))
        self._swap_axes_check.toggled.connect(
            lambda v: self._cm.set_value("conversion.swap_axes", v)
        )
        coord_form.addRow(self._swap_axes_check)

        form_layout.addWidget(coord_group)

        # --- Quality Control ---
        qc_group = QGroupBox("Quality Control")
        qc_form = QFormLayout(qc_group)

        self._qc_enabled_check = QCheckBox("Enable QC sweep")
        self._qc_enabled_check.setChecked(True)
        self._qc_enabled_check.setToolTip(TOOLTIPS.get("quality_control.enabled", ""))
        self._qc_enabled_check.toggled.connect(self._on_qc_toggled)
        qc_form.addRow(self._qc_enabled_check)

        self._sliver_angle_spin = QDoubleSpinBox()
        self._sliver_angle_spin.setRange(0.1, 30.0)
        self._sliver_angle_spin.setValue(5.0)
        self._sliver_angle_spin.setDecimals(1)
        self._sliver_angle_spin.setSuffix(" deg")
        self._sliver_angle_spin.setToolTip(TOOLTIPS.get("quality_control.sliver_angle_deg", ""))
        self._sliver_angle_spin.valueChanged.connect(
            lambda v: self._cm.set_value("quality_control.sliver_angle_deg", v)
        )
        qc_form.addRow("Sliver Angle:", self._sliver_angle_spin)

        self._max_hole_spin = QSpinBox()
        self._max_hole_spin.setRange(0, 1000)
        self._max_hole_spin.setValue(20)
        self._max_hole_spin.setToolTip(TOOLTIPS.get("quality_control.max_hole_edges", ""))
        self._max_hole_spin.valueChanged.connect(
            lambda v: self._cm.set_value("quality_control.max_hole_edges", v)
        )
        qc_form.addRow("Max Hole Edges:", self._max_hole_spin)

        form_layout.addWidget(qc_group)

        # --- Output ---
        output_group = QGroupBox("Output")
        output_layout = QHBoxLayout(output_group)
        self._convert_output = QLineEdit()
        self._convert_output.setPlaceholderText("Output PLY file path...")
        output_layout.addWidget(self._convert_output)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_convert_output)
        output_layout.addWidget(browse_btn)
        form_layout.addWidget(output_group)

        # --- Convert button + progress ---
        btn_layout = QHBoxLayout()
        self._convert_btn = QPushButton("Convert to PLY")
        self._convert_btn.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; "
            "font-weight: bold; padding: 10px 20px; font-size: 13px; }"
            "QPushButton:hover { background-color: #1976D2; }"
        )
        self._convert_btn.clicked.connect(self._run_convert)
        btn_layout.addWidget(self._convert_btn)
        btn_layout.addStretch()
        form_layout.addLayout(btn_layout)

        self._convert_progress = QProgressBar()
        self._convert_progress.setVisible(False)
        form_layout.addWidget(self._convert_progress)

        self._convert_status = QLabel("")
        self._convert_status.setStyleSheet("color: #a0a0a0;")
        self._convert_status.setWordWrap(True)
        form_layout.addWidget(self._convert_status)

        form_layout.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll)
        return page

    def _create_refine_page(self) -> QWidget:
        """Create the Refine parameters page."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(12, 12, 12, 12)

        title = QLabel("Adaptive Mesh Refinement")
        title.setStyleSheet("color: #00bcd4; font-weight: bold; font-size: 14px;")
        layout.addWidget(title)

        desc = QLabel(
            "Subdivide mesh faces near a center point for higher resolution.\n"
            "Requires a DEM file (refinement reads elevations directly)."
        )
        desc.setStyleSheet("color: #a0a0a0;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Refinement parameters
        params_group = QGroupBox("Refinement Parameters")
        form = QFormLayout(params_group)

        self._ref_lat = QDoubleSpinBox()
        self._ref_lat.setRange(-90.0, 90.0)
        self._ref_lat.setDecimals(4)
        self._ref_lat.setSuffix(" deg")
        self._ref_lat.setToolTip(TOOLTIPS.get("refinement.center_lat", ""))
        self._ref_lat.valueChanged.connect(
            lambda v: self._cm.set_value("refinement.center_lat", v)
        )
        form.addRow("Center Latitude:", self._ref_lat)

        self._ref_lon = QDoubleSpinBox()
        self._ref_lon.setRange(-360.0, 360.0)
        self._ref_lon.setDecimals(4)
        self._ref_lon.setSuffix(" deg")
        self._ref_lon.setToolTip(TOOLTIPS.get("refinement.center_lon", ""))
        self._ref_lon.valueChanged.connect(
            lambda v: self._cm.set_value("refinement.center_lon", v)
        )
        form.addRow("Center Longitude:", self._ref_lon)

        self._ref_inner = QDoubleSpinBox()
        self._ref_inner.setRange(0.01, 500.0)
        self._ref_inner.setValue(1.0)
        self._ref_inner.setDecimals(2)
        self._ref_inner.setSuffix(" km")
        self._ref_inner.setToolTip(TOOLTIPS.get("refinement.inner_radius_km", ""))
        self._ref_inner.valueChanged.connect(
            lambda v: self._cm.set_value("refinement.inner_radius_km", v)
        )
        form.addRow("Inner Radius:", self._ref_inner)

        self._ref_outer = QDoubleSpinBox()
        self._ref_outer.setRange(0.1, 1000.0)
        self._ref_outer.setValue(20.0)
        self._ref_outer.setDecimals(2)
        self._ref_outer.setSuffix(" km")
        self._ref_outer.setToolTip(TOOLTIPS.get("refinement.outer_radius_km", ""))
        self._ref_outer.valueChanged.connect(
            lambda v: self._cm.set_value("refinement.outer_radius_km", v)
        )
        form.addRow("Outer Radius:", self._ref_outer)

        self._ref_max_faces = QSpinBox()
        self._ref_max_faces.setRange(100, 10000000)
        self._ref_max_faces.setValue(100000)
        self._ref_max_faces.setSingleStep(10000)
        self._ref_max_faces.setToolTip(TOOLTIPS.get("refinement.max_faces", ""))
        self._ref_max_faces.valueChanged.connect(
            lambda v: self._cm.set_value("refinement.max_faces", v)
        )
        form.addRow("Max Faces:", self._ref_max_faces)

        self._ref_min_edge = QDoubleSpinBox()
        self._ref_min_edge.setRange(0.1, 10000.0)
        self._ref_min_edge.setValue(10.0)
        self._ref_min_edge.setDecimals(1)
        self._ref_min_edge.setSuffix(" m")
        self._ref_min_edge.setToolTip(TOOLTIPS.get("refinement.min_edge_m", ""))
        self._ref_min_edge.valueChanged.connect(
            lambda v: self._cm.set_value("refinement.min_edge_m", v)
        )
        form.addRow("Min Edge Length:", self._ref_min_edge)

        self._ref_max_edge = QDoubleSpinBox()
        self._ref_max_edge.setRange(1.0, 100000.0)
        self._ref_max_edge.setValue(1000.0)
        self._ref_max_edge.setDecimals(1)
        self._ref_max_edge.setSuffix(" m")
        self._ref_max_edge.setToolTip(TOOLTIPS.get("refinement.max_edge_m", ""))
        self._ref_max_edge.valueChanged.connect(
            lambda v: self._cm.set_value("refinement.max_edge_m", v)
        )
        form.addRow("Max Edge Length:", self._ref_max_edge)

        self._ref_init_ds = QSpinBox()
        self._ref_init_ds.setRange(1, 200)
        self._ref_init_ds.setValue(10)
        self._ref_init_ds.setToolTip(TOOLTIPS.get("refinement.initial_downsample", ""))
        self._ref_init_ds.valueChanged.connect(
            lambda v: self._cm.set_value("refinement.initial_downsample", v)
        )
        form.addRow("Initial Downsample:", self._ref_init_ds)

        layout.addWidget(params_group)

        # Output file
        output_group = QGroupBox("Output")
        output_layout = QHBoxLayout(output_group)
        self._refine_output = QLineEdit()
        self._refine_output.setPlaceholderText("Refined PLY output path...")
        output_layout.addWidget(self._refine_output)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_refine_output)
        output_layout.addWidget(browse_btn)
        layout.addWidget(output_group)

        # Refine button + progress
        btn_layout = QHBoxLayout()
        self._refine_btn = QPushButton("Refine Mesh")
        self._refine_btn.setStyleSheet(
            "QPushButton { background-color: #FF9800; color: white; "
            "font-weight: bold; padding: 10px 20px; font-size: 13px; }"
            "QPushButton:hover { background-color: #F57C00; }"
        )
        self._refine_btn.clicked.connect(self._run_refine)
        btn_layout.addWidget(self._refine_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        self._refine_progress = QProgressBar()
        self._refine_progress.setVisible(False)
        layout.addWidget(self._refine_progress)

        self._refine_status = QLabel("")
        self._refine_status.setStyleSheet("color: #a0a0a0;")
        self._refine_status.setWordWrap(True)
        layout.addWidget(self._refine_status)

        layout.addStretch()
        return page

    # -- Navigation --------------------------------------------------------

    def _on_page_changed(self, index):
        self._stack.setCurrentIndex(index)

    # -- Signal handlers ---------------------------------------------------

    def _on_dem_loaded(self, filepath):
        self.config_changed.emit()
        self._convert_status.setText(f"DEM loaded: {Path(filepath).name}")

    def _on_roi_selected(self, lat, lon, radius):
        self._roi_label.setText(f"ROI: ({lat:.3f}, {lon:.3f}), r={radius:.1f} km")
        # Also populate refinement center with ROI center
        self._ref_lat.setValue(lat)
        self._ref_lon.setValue(lon)

    def _on_max_edge_mode_changed(self, index):
        """Handle max edge mode combo change."""
        if index == 0:  # Auto
            self._cm.set_value("conversion.max_edge_m", -1.0)
            self._max_edge_spin.setEnabled(False)
        elif index == 1:  # Disabled
            self._cm.set_value("conversion.max_edge_m", 0.0)
            self._max_edge_spin.setEnabled(False)
        else:  # Custom
            self._max_edge_spin.setEnabled(True)
            self._cm.set_value("conversion.max_edge_m", self._max_edge_spin.value())

    def _on_qc_toggled(self, enabled):
        """Enable/disable QC parameter widgets."""
        self._cm.set_value("quality_control.enabled", enabled)
        self._sliver_angle_spin.setEnabled(enabled)
        self._max_hole_spin.setEnabled(enabled)

    def _on_body_changed(self, body_name):
        self._cm.set_body(body_name)
        radius = self._cm.get_value("input.body_radius_m", 1737400.0)
        self._radius_spin.blockSignals(True)
        self._radius_spin.setValue(radius)
        self._radius_spin.blockSignals(False)

    # -- Browse handlers ---------------------------------------------------

    def _browse_convert_output(self):
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Converted Mesh",
            str(Path.home()),
            "PLY Files (*.ply);;All Files (*)"
        )
        if filepath:
            if not filepath.endswith('.ply'):
                filepath += '.ply'
            self._convert_output.setText(filepath)
            self._cm.set_value("conversion.output_file", filepath)

    def _browse_refine_output(self):
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Refined Mesh",
            str(Path.home()),
            "PLY Files (*.ply);;All Files (*)"
        )
        if filepath:
            if not filepath.endswith('.ply'):
                filepath += '.ply'
            self._refine_output.setText(filepath)
            self._cm.set_value("refinement.output_file", filepath)

    # -- Conversion --------------------------------------------------------

    def _run_convert(self):
        """Start DEM-to-PLY conversion."""
        output = self._convert_output.text()
        if not output:
            QMessageBox.warning(self, "Missing Output",
                                "Please specify an output PLY file path.")
            return

        self._cm.set_value("conversion.output_file", output)

        is_valid, errors = self._cm.validate()
        if not is_valid:
            QMessageBox.warning(self, "Validation Failed",
                                "\n".join(errors))
            return

        self._convert_btn.setEnabled(False)
        self._convert_progress.setVisible(True)
        self._convert_progress.setValue(0)
        self._convert_status.setText("Converting...")

        # Every emitted line overwrites this single status label, so an error
        # that arrives before the process exits (a missing executable, an
        # unsupported platform) is clobbered by the "failed (exit code ...)"
        # line one moment later. Keep the last one so the failure branch can
        # show WHY instead of only the code.
        self._convert_last_error = ""

        self._convert_thread = DEMConvertThread(self._cm)
        self._convert_thread.output_ready.connect(
            lambda t: self._convert_status.setText(t)
        )
        self._convert_thread.error_ready.connect(self._on_convert_error)
        self._convert_thread.progress_update.connect(
            lambda p: self._convert_progress.setValue(int(p))
        )
        self._convert_thread.finished.connect(self._on_convert_finished)
        self._convert_thread.start()

    def _on_convert_error(self, text):
        self._convert_last_error = text
        self._convert_status.setText(f"Error: {text}")

    def _on_convert_finished(self, exit_code):
        self._convert_btn.setEnabled(True)
        self._convert_progress.setVisible(False)

        if exit_code == 0:
            output = self._cm.get_value("conversion.output_file", "")
            self._convert_status.setText(
                f"Conversion complete: {Path(output).name}"
            )
            # Auto-load in Inspect page
            if output and Path(output).exists():
                self._mesh_panel.load_mesh(output)
                self.mesh_converted.emit(output)
        else:
            message = f"Conversion failed (exit code {exit_code})"
            if exit_code == 2:
                # dtm_to_ply_delaunay's "the mesh is empty" refusal.
                message += (
                    " - the tool produced no faces and refused to write a file. "
                    "Check the ROI center against the DEM (a wrong latitude "
                    "sign puts it off the raster), the ROI radius, and "
                    "Max Edge Length."
                )
            detail = getattr(self, "_convert_last_error", "")
            if detail:
                message += f"\n{detail}"
            self._convert_status.setText(message)

        self._convert_thread = None

    # -- Refinement --------------------------------------------------------

    def _run_refine(self):
        """Start adaptive refinement."""
        output = self._refine_output.text()
        if not output:
            QMessageBox.warning(self, "Missing Output",
                                "Please specify a refinement output PLY file path.")
            return

        self._cm.set_value("refinement.output_file", output)
        self._cm.set_value("refinement.enabled", True)

        self._refine_btn.setEnabled(False)
        self._refine_progress.setVisible(True)
        self._refine_progress.setValue(0)
        self._refine_status.setText("Refining mesh...")

        # Same single-label overwrite problem as the convert path above.
        self._refine_last_error = ""

        self._refine_thread = DEMRefineThread(self._cm)
        self._refine_thread.output_ready.connect(
            lambda t: self._refine_status.setText(t)
        )
        self._refine_thread.error_ready.connect(self._on_refine_error)
        self._refine_thread.progress_update.connect(
            lambda p: self._refine_progress.setValue(int(p))
        )
        self._refine_thread.finished.connect(self._on_refine_finished)
        self._refine_thread.start()

    def _on_refine_error(self, text):
        self._refine_last_error = text
        self._refine_status.setText(f"Error: {text}")

    def _on_refine_finished(self, exit_code):
        self._refine_btn.setEnabled(True)
        self._refine_progress.setVisible(False)

        if exit_code == 0:
            output = self._cm.get_value("refinement.output_file", "")
            self._refine_status.setText(
                f"Refinement complete: {Path(output).name}"
            )
            if output and Path(output).exists():
                self._mesh_panel.load_mesh(output)
                self.mesh_converted.emit(output)
        elif exit_code == 2:
            # dtm_advanced_processor's own refusal code: its mesh pipeline is
            # unfinished upstream, so refinement cannot succeed. Name the
            # working alternative here -- this label is the one message that
            # survives, since every output line overwrites the previous one.
            self._refine_status.setText(
                "Adaptive refinement is NOT IMPLEMENTED (exit 2). Use Convert "
                "with ROI enabled instead: it grades point density radially "
                "(Core Fraction / Density Exponent). "
                "See docs/dem-to-mesh-workflow.md."
            )
        else:
            message = f"Refinement failed (exit code {exit_code})"
            detail = getattr(self, "_refine_last_error", "")
            if detail:
                message += f"\n{detail}"
            self._refine_status.setText(message)

        self._refine_thread = None

    # -- Config loading ----------------------------------------------------

    def load_from_config(self):
        """Refresh UI from config manager."""
        self._preview.load_from_config()

        # Grid & Body
        self._downsample_spin.setValue(
            self._cm.get_value("conversion.downsample", 10)
        )
        self._radius_spin.setValue(
            self._cm.get_value("input.body_radius_m", 1737400.0)
        )

        # ROI
        self._core_frac_spin.setValue(
            self._cm.get_value("conversion.roi_core_frac", 0.15)
        )
        self._density_exp_spin.setValue(
            self._cm.get_value("conversion.density_exp", 2.0)
        )

        # Max edge mode
        max_edge = self._cm.get_value("conversion.max_edge_m", -1.0)
        if max_edge < 0:
            self._max_edge_combo.setCurrentIndex(0)  # Auto
        elif max_edge == 0:
            self._max_edge_combo.setCurrentIndex(1)  # Disabled
        else:
            self._max_edge_combo.setCurrentIndex(2)  # Custom
            self._max_edge_spin.setValue(max_edge)

        # Coordinate corrections
        self._south_pole_check.setChecked(
            self._cm.get_value("conversion.south_pole_coords", False)
        )
        self._swap_axes_check.setChecked(
            self._cm.get_value("conversion.swap_axes", False)
        )

        # Quality control
        qc_enabled = self._cm.get_value("quality_control.enabled", True)
        self._qc_enabled_check.setChecked(qc_enabled)
        self._sliver_angle_spin.setValue(
            self._cm.get_value("quality_control.sliver_angle_deg", 5.0)
        )
        self._max_hole_spin.setValue(
            self._cm.get_value("quality_control.max_hole_edges", 20)
        )
        self._on_qc_toggled(qc_enabled)

        # Output
        self._convert_output.setText(
            self._cm.get_value("conversion.output_file", "")
        )

        # Refinement
        self._ref_lat.setValue(self._cm.get_value("refinement.center_lat", 0.0))
        self._ref_lon.setValue(self._cm.get_value("refinement.center_lon", 0.0))
        self._ref_inner.setValue(self._cm.get_value("refinement.inner_radius_km", 1.0))
        self._ref_outer.setValue(self._cm.get_value("refinement.outer_radius_km", 20.0))
        self._ref_max_faces.setValue(self._cm.get_value("refinement.max_faces", 100000))
        self._ref_min_edge.setValue(self._cm.get_value("refinement.min_edge_m", 10.0))
        self._ref_max_edge.setValue(self._cm.get_value("refinement.max_edge_m", 1000.0))
        self._ref_init_ds.setValue(self._cm.get_value("refinement.initial_downsample", 10))
        self._refine_output.setText(
            self._cm.get_value("refinement.output_file", "")
        )

        # ROI status
        if self._cm.get_value("roi.enabled", False):
            lat = self._cm.get_value("roi.center_lat", 0)
            lon = self._cm.get_value("roi.center_lon", 0)
            r = self._cm.get_value("roi.radius_km", 0)
            self._roi_label.setText(f"ROI: ({lat:.3f}, {lon:.3f}), r={r:.1f} km")
        else:
            self._roi_label.setText("ROI: disabled")
