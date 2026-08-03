"""
Viewshed module adapter for the unified dashboard.
Composes the same classes used by the standalone ViewshedMainWindow.
"""

from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QSplitter, QTabWidget
)
from PyQt6.QtCore import Qt

from dashboard_core.module_interface import AbstractModule
from viewshed_core.viewshed_config_manager import ViewshedConfigManager
from viewshed_core.viewshed_simulation_runner import ViewshedSimulationThread
from viewshed_panels.viewshed_config_panel import ViewshedConfigurationPanel
from viewshed_panels.viewshed_visualization_panel import ViewshedVisualizationPanel
from viewshed_panels.viewshed_results_panel import ViewshedResultsPanel

try:
    from module_adapters.viewshed_analysis import ViewshedAnalysisPanel
    HAS_VIS_ANALYSIS = True
except ImportError:
    HAS_VIS_ANALYSIS = False


class ViewshedModule(AbstractModule):
    """Viewshed analysis module for the dashboard."""

    def __init__(self, console_panel):
        super().__init__()
        self._console = console_panel
        self._sim_thread = None

        # Create components (same classes as ViewshedMainWindow)
        self._cm = ViewshedConfigManager()
        self._config_panel = ViewshedConfigurationPanel(self._cm)
        self._viz_panel = ViewshedVisualizationPanel()
        self._results_panel = ViewshedResultsPanel()

        # Build composite widget
        self._widget = self._build_widget()

    # -- Identity ----------------------------------------------------------

    def name(self) -> str:
        return "viewshed"

    def display_name(self) -> str:
        return "Viewshed Analysis"

    def icon_text(self) -> str:
        return "VIS"

    # -- Widget ------------------------------------------------------------

    def widget(self) -> QWidget:
        return self._widget

    def _build_widget(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        left_tabs = QTabWidget()
        left_tabs.addTab(self._config_panel, "Configuration")
        left_tabs.addTab(self._results_panel, "Results")

        if HAS_VIS_ANALYSIS:
            self._analysis_panel = ViewshedAnalysisPanel(self._cm)
            left_tabs.addTab(self._analysis_panel, "Analysis")
        else:
            self._analysis_panel = None

        self._left_tabs = left_tabs

        splitter.addWidget(left_tabs)
        splitter.addWidget(self._viz_panel)
        splitter.setSizes([780, 620])

        layout.addWidget(splitter)
        return container

    # -- Config ------------------------------------------------------------

    def config_manager(self):
        return self._cm

    def set_input(self, input_type: str, value: str):
        if input_type == "ptm_output_dir":
            self._cm.set_value("input.ptm_output_dir", value)
            self._config_panel.load_from_config()
            self._console.append_info(
                f"[VIS] PTM output dir set: {Path(value).name}"
            )
        elif input_type == "mesh_file":
            self._cm.set_value("input.mesh_file", value)
            self._config_panel.load_from_config()
            self._console.append_info(
                f"[VIS] Mesh file set: {Path(value).name}"
            )

    def get_outputs(self) -> dict:
        config = self._cm.get_config()
        output_dir = config.get("output", {}).get("output_dir", "")
        if output_dir and Path(output_dir).exists():
            return {"output_dir": output_dir}
        return {}

    # -- Simulation --------------------------------------------------------

    def run_simulation(self):
        is_valid, errors = self._cm.validate()
        if not is_valid:
            self._console.append_error(
                "[VIS] Validation failed:\n" +
                "\n".join(f"  - {e}" for e in errors)
            )
            return

        self._sim_thread = ViewshedSimulationThread(self._cm)
        self._sim_thread.output_ready.connect(
            lambda t: self._console.append_output(f"[VIS] {t}")
        )
        self._sim_thread.error_ready.connect(
            lambda t: self._console.append_error(f"[VIS] {t}")
        )
        self._sim_thread.progress_update.connect(self._on_progress)

        # ViewshedSimulationThread uses finished_signal instead of finished
        if hasattr(self._sim_thread, 'finished_signal'):
            self._sim_thread.finished_signal.connect(self._on_finished)
        else:
            self._sim_thread.finished.connect(self._on_finished)

        self._set_status("running")
        self._console.append_header("Viewshed Analysis Starting")
        self._sim_thread.start()

    def stop_simulation(self):
        if self._sim_thread:
            self._console.append_warning("[VIS] Stopping analysis...")
            self._sim_thread.stop()

    # -- Internal slots ----------------------------------------------------

    def _on_progress(self, percent):
        pass

    def _on_finished(self, exit_code):
        if exit_code == 0:
            self._console.append_success("[VIS] Analysis completed")
            self._set_status("completed")

            config = self._cm.get_config()
            output_dir = config.get("output", {}).get("output_dir", "")
            if output_dir and Path(output_dir).exists():
                self.output_available.emit("output_dir", {"path": output_dir})
                if self._analysis_panel:
                    self._analysis_panel.set_output_dir(output_dir)
        else:
            self._console.append_error(
                f"[VIS] Analysis failed (exit code: {exit_code})"
            )
            self._set_status("failed")

        self._sim_thread = None
