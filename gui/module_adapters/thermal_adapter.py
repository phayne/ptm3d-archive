"""
Thermal module adapter for the unified dashboard.
Composes the same classes used by the standalone PTM3DMainWindow.
"""

from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QSplitter, QTabWidget, QMessageBox
)
from PyQt6.QtCore import Qt

from dashboard_core.module_interface import AbstractModule
from gui_core.config_manager import ConfigManager
from gui_core.simulation_runner import SimulationRunner, SimulationThread
from gui_core.preflight import (
    run_preflight, confirm_preflight, resolve_path
)
from gui_panels.config_panel import ConfigurationPanel
from gui_panels.visualization_panel import VisualizationPanel
from gui_panels.results_panel import ResultsPanel

try:
    from module_adapters.thermal_analysis import ThermalAnalysisPanel
    HAS_THERMAL_ANALYSIS = True
except ImportError:
    HAS_THERMAL_ANALYSIS = False


class ThermalModule(AbstractModule):
    """Thermal simulation module for the dashboard."""

    def __init__(self, console_panel):
        super().__init__()
        self._console = console_panel
        self._sim_thread = None
        self._run_config_file = None

        # Create components (same classes as PTM3DMainWindow)
        self._cm = ConfigManager()
        self._config_panel = ConfigurationPanel(self._cm)
        self._viz_panel = VisualizationPanel()
        self._results_panel = ResultsPanel()

        # Build composite widget
        self._widget = self._build_widget()

        # Point the results browser to the config's output directory
        output_dir = self._cm.get_value("output.directory", "")
        if output_dir:
            self._results_panel.set_default_browse_dir(output_dir)

        # Auto-load last results if the directory still exists
        if self._results_panel.current_results_dir:
            try:
                self._results_panel.load_results(
                    self._results_panel.current_results_dir
                )
            except Exception:
                pass

        # Connect internal signals
        self._config_panel.config_changed.connect(self._on_config_changed)
        self._results_panel.results_selected.connect(self._on_results_selected)
        self._results_panel.timestep_selected.connect(self._on_timestep_selected)

    # -- Identity ----------------------------------------------------------

    def name(self) -> str:
        return "thermal"

    def display_name(self) -> str:
        return "Thermal Simulation"

    def icon_text(self) -> str:
        return "THM"

    # -- Widget ------------------------------------------------------------

    def widget(self) -> QWidget:
        return self._widget

    def _build_widget(self) -> QWidget:
        """Replicate the PTM3DMainWindow layout as a plain QWidget."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: config + results + analysis tabs
        left_tabs = QTabWidget()
        left_tabs.addTab(self._config_panel, "Configuration")
        left_tabs.addTab(self._results_panel, "Results")

        if HAS_THERMAL_ANALYSIS:
            self._analysis_panel = ThermalAnalysisPanel(self._cm, self._results_panel)
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
        if input_type == "mesh_file":
            self._cm.set_value("mesh.import_file", value)
            self._config_panel.load_from_config()
            self._console.append_info(
                f"[THERMAL] Mesh file set: {Path(value).name}"
            )

    def get_outputs(self) -> dict:
        config = self._cm.get_config()
        output_dir = config.get("output", {}).get("directory", "")
        outputs = {}
        if output_dir and Path(output_dir).exists():
            outputs["output_dir"] = output_dir
        mesh_file = config.get("mesh", {}).get("import_file", "")
        if mesh_file:
            outputs["mesh_file"] = mesh_file
        return outputs

    # -- Simulation --------------------------------------------------------

    def run_simulation(self):
        is_valid, errors = self._cm.validate()
        if not is_valid:
            self._console.append_error(
                "[THERMAL] Validation failed:\n" +
                "\n".join(f"  - {e}" for e in errors)
            )
            return

        # Pre-flight (paths, dataless cloud files, disk, VF guard, exe)
        config_dict = self._cm.get_config()
        report = run_preflight(config_dict, SimulationRunner.find_ptm_executable)
        for line in report.summary_lines():
            self._console.append_info(f"[THERMAL][preflight] {line}")
        if not confirm_preflight(self._widget, report):
            self._console.append_warning(
                "[THERMAL] Launch canceled by pre-flight check")
            return

        # Write the run config BESIDE the output directory, never a temp
        # file: the engine resolves a relative radiation.view_factor_file
        # against the YAML's own directory, so a config in /var/folders/...
        # silently missed the VF cache and triggered an (hours-long, and per
        # PTM-19 potentially wrong) rebuild. save_run_yaml absolutizes all
        # path fields against the project root for the same reason.
        try:
            out_dir = resolve_path(
                self._cm.get_value("output.directory", "output"),
                report.project_root)
            out_dir.mkdir(parents=True, exist_ok=True)
            run_config = out_dir / "run_config.yaml"
            self._cm.save_run_yaml(run_config, report.project_root)
            self._run_config_file = str(run_config)
            self._console.append_info(f"[THERMAL] Run config: {run_config}")
        except Exception as e:
            self._console.append_error(f"[THERMAL] Failed to save config: {e}")
            return

        self._sim_thread = SimulationThread(
            self._run_config_file, config_dict=config_dict
        )
        self._sim_thread.output_ready.connect(
            lambda t: self._console.append_output(f"[THERMAL] {t}")
        )
        self._sim_thread.error_ready.connect(
            lambda t: self._console.append_error(f"[THERMAL] {t}")
        )
        self._sim_thread.progress_update.connect(self._on_progress)
        self._sim_thread.heartbeat.connect(
            lambda msg, stalled: (self._console.append_warning(
                f"[THERMAL][watchdog] {msg}") if stalled else
                self._console.append_info(f"[THERMAL][watchdog] {msg}"))
        )
        self._sim_thread.finished.connect(self._on_finished)

        self._set_status("running")
        self._console.append_header("Thermal Simulation Starting")
        self._sim_thread.start()

    def stop_simulation(self):
        if self._sim_thread:
            self._console.append_warning("[THERMAL] Stopping simulation...")
            self._sim_thread.stop()

    # -- Internal slots ----------------------------------------------------

    def _on_config_changed(self):
        # Keep results browser default dir in sync with config output dir
        output_dir = self._cm.get_value("output.directory", "")
        if output_dir:
            self._results_panel.set_default_browse_dir(output_dir)

    def _on_results_selected(self, results_dir):
        self._viz_panel.load_results(results_dir)
        if self._analysis_panel:
            self._analysis_panel.set_results_dir(results_dir)

    def _on_timestep_selected(self, index, time_val):
        self._viz_panel.set_timestep(index)

    def _on_progress(self, percent):
        pass  # Dashboard handles progress bar via status_changed

    def _on_finished(self, exit_code):
        if exit_code == 0:
            self._console.append_success("[THERMAL] Simulation completed")
            self._set_status("completed")

            # Auto-load results
            config = self._cm.get_config()
            output_dir = config.get("output", {}).get("directory", "")
            if output_dir and Path(output_dir).exists():
                self._results_panel.load_results(output_dir)
                self._left_tabs.setCurrentWidget(self._results_panel)
                if self._analysis_panel:
                    self._analysis_panel.set_results_dir(output_dir)

                # Emit output for chaining
                self.output_available.emit("output_dir", {"path": output_dir})

            mesh_file = config.get("mesh", {}).get("import_file", "")
            if mesh_file:
                self.output_available.emit("mesh_file", {"path": mesh_file})
        else:
            self._console.append_error(
                f"[THERMAL] Simulation failed (exit code: {exit_code})"
            )
            self._set_status("failed")

        self._sim_thread = None
        # The run config lives beside the output data on purpose (it
        # documents exactly what produced the run) -- no cleanup.
