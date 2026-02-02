#!/usr/bin/env python3
"""
Viewshed GUI - PyQt6 Application for PEGASUS Viewshed Model
Configures, runs, and visualises viewshed simulations with custom mask builder.
"""

import os
import sys
from pathlib import Path

# Fix Qt plugin discovery on macOS (conda environments often lack QT_PLUGIN_PATH)
if "QT_PLUGIN_PATH" not in os.environ:
    try:
        import PyQt6 as _pyqt6
        _plugin_path = str(Path(_pyqt6.__file__).parent / "Qt6" / "plugins")
        if Path(_plugin_path).is_dir():
            os.environ["QT_PLUGIN_PATH"] = _plugin_path
        del _pyqt6, _plugin_path
    except ImportError:
        pass

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QSplitter, QStatusBar, QToolBar,
    QMessageBox, QFileDialog, QLabel, QProgressBar, QDockWidget,
)
from PyQt6.QtCore import Qt, QSettings
from PyQt6.QtGui import QAction, QFont, QKeySequence

# Ensure gui/ is in path
gui_dir = str(Path(__file__).parent)
if gui_dir not in sys.path:
    sys.path.insert(0, gui_dir)

# Reuse existing styles and console panel
from gui_core.styles import ModernStyle
from gui_panels.console_panel import ConsolePanel

# Viewshed-specific modules
from viewshed_core.viewshed_config_manager import ViewshedConfigManager
from viewshed_core.viewshed_simulation_runner import ViewshedSimulationThread

from viewshed_panels.viewshed_config_panel import ViewshedConfigurationPanel
from viewshed_panels.viewshed_results_panel import ViewshedResultsPanel


class ViewshedMainWindow(QMainWindow):
    """Main application window for Viewshed GUI."""

    def __init__(self):
        super().__init__()

        self.config_manager = ViewshedConfigManager()
        self.current_project_file = None
        self.unsaved_changes = False
        self.simulation_thread = None

        self.settings = QSettings("PTM3D", "ViewshedGUI")

        self.init_ui()
        self.apply_style()
        self.restore_window_state()

    def init_ui(self):
        self.setWindowTitle("PEGASUS Viewshed - Traverse Visibility Analysis")
        self.setMinimumSize(1400, 900)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Main horizontal splitter
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel: Config + Masks + Results
        self.left_panel = self._create_left_panel()
        self.main_splitter.addWidget(self.left_panel)

        # Right panel: Visualization (placeholder until viz panel is implemented)
        self.right_panel = self._create_right_panel()
        self.main_splitter.addWidget(self.right_panel)

        self.main_splitter.setSizes([560, 840])
        main_layout.addWidget(self.main_splitter)

        # Console dock
        self._create_console_dock()

        # Menu, toolbar, status bar
        self._create_menu_bar()
        self._create_toolbar()
        self._create_status_bar()

        # Signal routing
        self._connect_signals()

    # --- Panel creation ---

    def _create_left_panel(self):
        left_widget = QWidget()
        layout = QVBoxLayout(left_widget)
        layout.setContentsMargins(0, 0, 0, 0)

        self.left_tabs = QTabWidget()
        self.left_tabs.setTabPosition(QTabWidget.TabPosition.North)

        self.config_panel = ViewshedConfigurationPanel(self.config_manager)
        self.left_tabs.addTab(self.config_panel, "Configuration")

        # Masks panel (loaded lazily to avoid circular imports)
        self.masks_panel = self._create_masks_panel()
        if self.masks_panel is not None:
            self.left_tabs.addTab(self.masks_panel, "Masks")

        self.results_panel = ViewshedResultsPanel()
        self.left_tabs.addTab(self.results_panel, "Results")

        layout.addWidget(self.left_tabs)
        return left_widget

    def _create_masks_panel(self):
        """Try to load the masks panel; return None if not yet available."""
        try:
            from viewshed_panels.viewshed_masks_panel import ViewshedMasksPanel
            return ViewshedMasksPanel(self.config_manager)
        except ImportError:
            return None

    def _create_right_panel(self):
        """Create the right panel with visualization."""
        right_widget = QWidget()
        layout = QVBoxLayout(right_widget)
        layout.setContentsMargins(0, 0, 0, 0)

        # Try to load PyVista-based visualization; fall back to placeholder
        try:
            from viewshed_panels.viewshed_visualization_panel import ViewshedVisualizationPanel
            self.viz_panel = ViewshedVisualizationPanel()
        except ImportError:
            self.viz_panel = None

        if self.viz_panel is not None:
            layout.addWidget(self.viz_panel)
        else:
            placeholder = QLabel(
                "3D Visualization\n\n"
                "Run a simulation to generate output,\n"
                "then load results via the Results tab.\n\n"
                "(PyVista required for mesh rendering)"
            )
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            placeholder.setStyleSheet(
                "color: #888; font-size: 14px; border: 1px solid #404040; "
                "border-radius: 4px; padding: 40px;"
            )
            layout.addWidget(placeholder)

        return right_widget

    def _create_console_dock(self):
        self.console_dock = QDockWidget("Console Output", self)
        self.console_dock.setObjectName("consoleDock")
        self.console_dock.setAllowedAreas(
            Qt.DockWidgetArea.BottomDockWidgetArea | Qt.DockWidgetArea.TopDockWidgetArea
        )

        self.console_panel = ConsolePanel()
        self.console_dock.setWidget(self.console_panel)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.console_dock)
        self.console_dock.setMinimumHeight(150)

    # --- Menu bar ---

    def _create_menu_bar(self):
        menubar = self.menuBar()

        # File
        file_menu = menubar.addMenu("&File")

        new_action = QAction("&New Project", self)
        new_action.setShortcut(QKeySequence.StandardKey.New)
        new_action.triggered.connect(self.new_project)
        file_menu.addAction(new_action)

        open_action = QAction("&Open Project...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self.open_project)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        save_action = QAction("&Save Project", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self.save_project)
        file_menu.addAction(save_action)

        save_as_action = QAction("Save Project &As...", self)
        save_as_action.setShortcut(QKeySequence("Ctrl+Shift+S"))
        save_as_action.triggered.connect(self.save_project_as)
        file_menu.addAction(save_as_action)

        file_menu.addSeparator()

        load_results_action = QAction("Load &Results...", self)
        load_results_action.setShortcut(QKeySequence("Ctrl+L"))
        load_results_action.triggered.connect(self.load_results)
        file_menu.addAction(load_results_action)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Simulation
        sim_menu = menubar.addMenu("&Simulation")

        run_action = QAction("&Run Simulation", self)
        run_action.setShortcut(QKeySequence("F5"))
        run_action.triggered.connect(self.run_simulation)
        sim_menu.addAction(run_action)

        self.stop_action = QAction("&Stop Simulation", self)
        self.stop_action.setShortcut(QKeySequence("Shift+F5"))
        self.stop_action.triggered.connect(self.stop_simulation)
        self.stop_action.setEnabled(False)
        sim_menu.addAction(self.stop_action)

        sim_menu.addSeparator()

        validate_action = QAction("&Validate Configuration", self)
        validate_action.setShortcut(QKeySequence("Ctrl+T"))
        validate_action.triggered.connect(self.validate_config)
        sim_menu.addAction(validate_action)

        # View
        view_menu = menubar.addMenu("&View")

        toggle_console = QAction("Toggle &Console", self)
        toggle_console.setShortcut(QKeySequence("Ctrl+`"))
        toggle_console.triggered.connect(self.toggle_console)
        view_menu.addAction(toggle_console)

        view_menu.addSeparator()

        reset_layout = QAction("&Reset Layout", self)
        reset_layout.triggered.connect(self.reset_layout)
        view_menu.addAction(reset_layout)

        # Visualization
        viz_menu = menubar.addMenu("V&isualization")

        refresh_action = QAction("&Refresh", self)
        refresh_action.setShortcut(QKeySequence("F6"))
        refresh_action.triggered.connect(self.refresh_visualization)
        viz_menu.addAction(refresh_action)

        export_img_action = QAction("Export &Image...", self)
        export_img_action.setShortcut(QKeySequence("Ctrl+E"))
        export_img_action.triggered.connect(self.export_visualization)
        viz_menu.addAction(export_img_action)

        viz_menu.addSeparator()

        reset_cam_action = QAction("Reset &Camera", self)
        reset_cam_action.setShortcut(QKeySequence("Home"))
        reset_cam_action.triggered.connect(self.reset_camera)
        viz_menu.addAction(reset_cam_action)

        # Help
        help_menu = menubar.addMenu("&Help")

        about_action = QAction("&About Viewshed GUI", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    # --- Toolbar ---

    def _create_toolbar(self):
        toolbar = QToolBar("Main Toolbar")
        toolbar.setObjectName("mainToolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        new_btn = QAction("New", self)
        new_btn.setToolTip("New Project (Ctrl+N)")
        new_btn.triggered.connect(self.new_project)
        toolbar.addAction(new_btn)

        open_btn = QAction("Open", self)
        open_btn.setToolTip("Open Project (Ctrl+O)")
        open_btn.triggered.connect(self.open_project)
        toolbar.addAction(open_btn)

        save_btn = QAction("Save", self)
        save_btn.setToolTip("Save Project (Ctrl+S)")
        save_btn.triggered.connect(self.save_project)
        toolbar.addAction(save_btn)

        toolbar.addSeparator()

        self.run_btn = QAction("Run", self)
        self.run_btn.setToolTip("Run Simulation (F5)")
        self.run_btn.triggered.connect(self.run_simulation)
        toolbar.addAction(self.run_btn)

        self.stop_btn = QAction("Stop", self)
        self.stop_btn.setToolTip("Stop Simulation (Shift+F5)")
        self.stop_btn.triggered.connect(self.stop_simulation)
        self.stop_btn.setEnabled(False)
        toolbar.addAction(self.stop_btn)

        toolbar.addSeparator()

        refresh_btn = QAction("Refresh", self)
        refresh_btn.setToolTip("Refresh Visualization (F6)")
        refresh_btn.triggered.connect(self.refresh_visualization)
        toolbar.addAction(refresh_btn)

    # --- Status bar ---

    def _create_status_bar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.status_label = QLabel("Ready")
        self.status_bar.addWidget(self.status_label, 1)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)

        self.sim_status = QLabel()
        self.sim_status.setMinimumWidth(100)
        self._update_sim_indicator("idle")
        self.status_bar.addPermanentWidget(self.sim_status)

    def _update_sim_indicator(self, state):
        colors = {
            "idle": ("#888", "Idle"),
            "running": ("#4CAF50", "Running"),
            "error": ("#F44336", "Error"),
            "complete": ("#2196F3", "Complete"),
        }
        color, text = colors.get(state, ("#888", "Unknown"))
        self.sim_status.setText(f"  {text}  ")
        self.sim_status.setStyleSheet(
            f"background-color: {color}; color: white; border-radius: 4px; "
            f"padding: 2px 8px; font-weight: bold;"
        )

    # --- Signal routing ---

    def _connect_signals(self):
        self.config_panel.config_changed.connect(self._on_config_changed)
        self.results_panel.field_selected.connect(self._on_field_selected)
        self.results_panel.directory_loaded.connect(self._on_results_loaded)

    def _on_config_changed(self):
        self.unsaved_changes = True
        self._update_window_title()

    def _on_results_loaded(self, directory):
        self.console_panel.append_info(f"Results loaded: {directory}")
        self.status_label.setText(f"Results: {directory}")

        # Auto-load mesh into visualization if available
        if self.viz_panel is not None:
            ptm_dir = self.config_manager.get_value("input.ptm_output_dir")
            if ptm_dir:
                mesh_path = self.config_manager.get_value("input.mesh_file")
                if not mesh_path:
                    mesh_path = str(Path(ptm_dir) / "mesh.ply")
                if Path(mesh_path).is_file():
                    self.viz_panel.load_mesh(mesh_path)
                    self.console_panel.append_info(f"Mesh loaded: {mesh_path}")
            self.viz_panel.set_output_directory(directory)

    def _on_field_selected(self, field_name, file_path):
        self.status_label.setText(f"Field: {field_name}")
        if self.viz_panel is not None:
            self.viz_panel.load_field(field_name, file_path)

    # --- File operations ---

    def new_project(self):
        if not self._check_unsaved():
            return
        self.config_manager.reset_to_defaults()
        self.config_panel.load_from_config()
        self.current_project_file = None
        self.unsaved_changes = False
        self._update_window_title()
        self.console_panel.append_info("New project created")

    def open_project(self):
        if not self._check_unsaved():
            return
        gui_dir = Path(__file__).parent.parent
        examples_dir = gui_dir / "examples"
        start_dir = str(examples_dir) if examples_dir.exists() else "."

        filename, _ = QFileDialog.getOpenFileName(
            self, "Open Project", start_dir,
            "Viewshed Projects (*.yaml *.yml);;All Files (*)"
        )
        if filename:
            try:
                self.config_manager.load_from_yaml(filename)
                self.config_panel.load_from_config()
                self.current_project_file = filename
                self.unsaved_changes = False
                self._update_window_title()
                self.console_panel.append_info(f"Project loaded: {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load project:\n{e}")

    def save_project(self):
        if self.current_project_file:
            self._save_to(self.current_project_file)
        else:
            self.save_project_as()

    def save_project_as(self):
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Project", "",
            "YAML Files (*.yaml);;All Files (*)"
        )
        if filename:
            if not filename.endswith((".yaml", ".yml")):
                filename += ".yaml"
            self._save_to(filename)

    def _save_to(self, filename):
        try:
            self.config_manager.save_to_yaml(filename)
            self.current_project_file = filename
            self.unsaved_changes = False
            self._update_window_title()
            self.console_panel.append_info(f"Project saved: {filename}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save project:\n{e}")

    def load_results(self):
        directory = QFileDialog.getExistingDirectory(
            self, "Select Output Directory", "",
            QFileDialog.Option.ShowDirsOnly
        )
        if directory:
            self.results_panel.load_directory(directory)

    def _check_unsaved(self):
        if not self.unsaved_changes:
            return True
        reply = QMessageBox.question(
            self, "Unsaved Changes",
            "You have unsaved changes. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        return reply == QMessageBox.StandardButton.Yes

    def _update_window_title(self):
        title = "PEGASUS Viewshed"
        if self.current_project_file:
            title += f" - {Path(self.current_project_file).name}"
        if self.unsaved_changes:
            title += " *"
        self.setWindowTitle(title)

    # --- Simulation ---

    def run_simulation(self):
        valid, messages = self.config_manager.validate()
        if not valid:
            QMessageBox.warning(
                self, "Configuration Invalid",
                "Fix errors before running:\n\n" + "\n".join(messages)
            )
            return

        if messages:
            self.console_panel.append_warning("Warnings:\n" + "\n".join(messages))

        self.console_panel.append_header("Starting viewshed simulation...")

        self.simulation_thread = ViewshedSimulationThread(self.config_manager)
        self.simulation_thread.output_ready.connect(self.console_panel.append_output)
        self.simulation_thread.error_ready.connect(self.console_panel.append_error)
        self.simulation_thread.progress_update.connect(self._on_progress)
        self.simulation_thread.finished_signal.connect(self._on_simulation_finished)

        self.simulation_thread.start()

        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.stop_action.setEnabled(True)
        self._update_sim_indicator("running")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

    def stop_simulation(self):
        if self.simulation_thread:
            self.simulation_thread.stop()
            self.console_panel.append_warning("Simulation stop requested...")

    def _on_progress(self, percent):
        self.progress_bar.setValue(int(percent))

    def _on_simulation_finished(self, exit_code):
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.stop_action.setEnabled(False)
        self.progress_bar.setVisible(False)

        if exit_code == 0:
            self._update_sim_indicator("complete")
            self.console_panel.append_success("Simulation completed successfully")
            # Auto-load results
            output_dir = self.config_manager.get_value("output.output_dir")
            if output_dir and Path(output_dir).is_dir():
                self.results_panel.load_directory(output_dir)
        elif exit_code == -2:
            self._update_sim_indicator("idle")
            self.console_panel.append_warning("Simulation stopped by user.")
        else:
            self._update_sim_indicator("error")
            self.console_panel.append_error(f"Simulation failed (exit code {exit_code})")

        self.simulation_thread = None

    def validate_config(self):
        valid, messages = self.config_manager.validate()
        if valid and not messages:
            QMessageBox.information(self, "Validation", "Configuration is valid.")
        elif valid:
            QMessageBox.information(
                self, "Validation",
                "Configuration valid with warnings:\n\n" + "\n".join(messages)
            )
        else:
            QMessageBox.warning(
                self, "Validation Failed",
                "Errors found:\n\n" + "\n".join(messages)
            )

    # --- View ---

    def toggle_console(self):
        self.console_dock.setVisible(not self.console_dock.isVisible())

    def reset_layout(self):
        self.main_splitter.setSizes([560, 840])
        self.console_dock.setVisible(True)

    # --- Visualization ---

    def refresh_visualization(self):
        if self.viz_panel is not None:
            self.viz_panel.refresh()

    def export_visualization(self):
        if self.viz_panel is not None:
            self.viz_panel.export_image()

    def reset_camera(self):
        if self.viz_panel is not None:
            self.viz_panel.reset_camera()

    # --- Help ---

    def show_about(self):
        QMessageBox.about(
            self, "About PEGASUS Viewshed GUI",
            "<h3>PEGASUS Viewshed GUI</h3>"
            "<p>Traverse visibility analysis and custom mask builder "
            "for the PEGASUS lunar mission simulation.</p>"
            "<p>Features:</p>"
            "<ul>"
            "<li>Geometric viewshed computation at waypoints</li>"
            "<li>Truth ice distribution from PTM3D temperature</li>"
            "<li>Custom boolean masks (AND/OR conditions)</li>"
            "<li>Multi-waypoint aggregation (visibility count, illumination hours)</li>"
            "<li>3D mesh visualization with field coloring</li>"
            "</ul>"
        )

    # --- Window state persistence ---

    def apply_style(self):
        self.setStyleSheet(ModernStyle.get_stylesheet())

    def restore_window_state(self):
        geometry = self.settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
        state = self.settings.value("windowState")
        if state:
            self.restoreState(state)

    def save_window_state(self):
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("windowState", self.saveState())

    def closeEvent(self, event):
        if self.simulation_thread and self.simulation_thread.isRunning():
            reply = QMessageBox.question(
                self, "Simulation Running",
                "A simulation is running. Stop it and exit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                event.ignore()
                return
            self.simulation_thread.stop()
            self.simulation_thread.wait(5000)

        self.save_window_state()
        event.accept()


def main():
    """Launch the Viewshed GUI application."""
    app = QApplication(sys.argv)
    app.setApplicationName("PEGASUS Viewshed")
    app.setOrganizationName("PTM3D")

    window = ViewshedMainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
