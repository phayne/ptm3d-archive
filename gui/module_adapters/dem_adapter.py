"""
DEM/Mesh module adapter for the unified dashboard.
Wraps the DEM preview, conversion and inspection panels.
Uses a splitter layout: config pages on the left, 3D mesh viewer on the right.
"""

from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QSplitter, QLabel
)
from PyQt6.QtCore import Qt

from dashboard_core.module_interface import AbstractModule


class DEMModule(AbstractModule):
    """DEM/Mesh manipulation module for the dashboard.

    Provides GeoTIFF preview, mesh conversion (including region-of-interest
    extraction with radial density grading) and 3D inspection. The 3D mesh
    viewer lives in the right panel (like the other modules) while
    config/preview pages are on the left.
    """

    def __init__(self, console_panel):
        super().__init__()
        self._console = console_panel
        self._output_mesh = None

        # Try to load full DEM panels; fall back to placeholder
        try:
            from dem_panels.dem_config_panel import DEMConfigurationPanel
            from dem_panels.dem_mesh_panel import DEMMeshPanel
            from dem_core.dem_config_manager import DEMConfigManager

            self._cm = DEMConfigManager()
            self._config_panel = DEMConfigurationPanel(self._cm)
            self._config_panel.mesh_converted.connect(self._on_mesh_converted)

            # Separate mesh viewer for the right side
            self._mesh_viewer = DEMMeshPanel()

            # When the config panel's internal mesh panel loads a mesh,
            # also load it in the right-side viewer
            internal_mesh_panel = self._config_panel._mesh_panel
            internal_mesh_panel.mesh_loaded.connect(
                lambda path: self._mesh_viewer.load_mesh(path)
            )

            self._widget = self._build_widget()
        except Exception as e:
            self._cm = None
            self._config_panel = None
            self._mesh_viewer = None
            self._widget = self._create_placeholder(str(e))
            self._console.append_warning(
                f"[DEM] Full DEM panels not available: {e}"
            )

    # -- Identity ----------------------------------------------------------

    def name(self) -> str:
        return "dem"

    def display_name(self) -> str:
        return "DEM/Mesh Tools"

    def icon_text(self) -> str:
        return "DEM"

    # -- Widget ------------------------------------------------------------

    def widget(self) -> QWidget:
        return self._widget

    def _build_widget(self) -> QWidget:
        """Build splitter layout: config on left, 3D viewer on right."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._config_panel)
        splitter.addWidget(self._mesh_viewer)
        splitter.setSizes([780, 620])

        layout.addWidget(splitter)
        return container

    def _create_placeholder(self, error_msg: str) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        label = QLabel(
            "DEM/Mesh Tools\n\n"
            "This module provides:\n"
            "  - GeoTIFF DEM preview and ROI selection\n"
            "  - Mesh conversion (DEM to PLY), with ROI extraction and\n"
            "    radial point-density grading\n"
            "  - 3D mesh inspection and quality analysis\n\n"
            f"(Loading full panels: {error_msg})"
        )
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("color: #a0a0a0; font-size: 14px;")
        label.setWordWrap(True)
        layout.addWidget(label)
        return w

    # -- Config ------------------------------------------------------------

    def config_manager(self):
        return self._cm

    def set_input(self, input_type: str, value: str):
        if input_type == "dem_file" and self._cm:
            self._cm.set_value("input.dem_file", value)
            if hasattr(self._config_panel, 'load_from_config'):
                self._config_panel.load_from_config()

    def get_outputs(self) -> dict:
        if self._output_mesh:
            return {"mesh_file": self._output_mesh}
        return {}

    # -- Simulation (conversion) -------------------------------------------

    def run_simulation(self):
        self._console.append_info("[DEM] Use the Convert button in the panel")

    def _on_mesh_converted(self, mesh_path: str):
        """Called after a successful conversion."""
        self._output_mesh = mesh_path

        # Load in the right-side viewer
        if self._mesh_viewer:
            self._mesh_viewer.load_mesh(mesh_path)

        self.output_available.emit("mesh_file", {"path": mesh_path})
        self._console.append_success(
            f"[DEM] Mesh ready: {Path(mesh_path).name}"
        )
