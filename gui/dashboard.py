"""
PTM3D Unified Dashboard
Single-window application with sidebar navigation for the DEM/Mesh and
Thermal modules. (The Exosphere and Viewshed modules are Paul-only tools kept
under gui/_private_tools/ and are not part of the public distribution.)
"""

import sys
import os
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QStackedWidget, QDockWidget, QStatusBar, QMenuBar, QMenu,
    QToolBar, QLabel, QProgressBar, QMessageBox, QFileDialog
)
from PyQt6.QtCore import Qt, QSettings
from PyQt6.QtGui import QAction, QKeySequence

from gui_core.styles import ModernStyle
from gui_panels.console_panel import ConsolePanel

from dashboard_core.sidebar import IconSidebar
from dashboard_core.module_registry import ModuleRegistry
from dashboard_core.chaining_manager import ChainingManager
from dashboard_core.module_interface import AbstractModule


class UnifiedDashboard(QMainWindow):
    """Main dashboard window integrating all PTM3D modules."""

    APP_VERSION = "1.0"

    def __init__(self):
        super().__init__()

        self.settings = QSettings("PTM3D", "Dashboard")
        self._active_module_index = 0

        # Shared console
        self.console_panel = ConsolePanel()

        # Module infrastructure
        self.registry = ModuleRegistry()
        self.chaining = ChainingManager()

        # Build UI
        self._init_ui()
        self._create_menu_bar()
        self._create_toolbar()
        self._create_status_bar()

        # Register modules (lazy — imported here to avoid circular deps)
        self._register_modules()

        # Wire chaining
        self.chaining.connect_modules(self.registry, self.console_panel)

        # Connect sidebar
        self.sidebar.module_selected.connect(self._on_module_selected)

        # Restore window state
        self._restore_state()

        self.console_panel.append_header("PTM3D Dashboard initialized")
        self.console_panel.append_info(
            f"Modules loaded: {', '.join(self.registry.names())}"
        )

    # -- UI Setup ----------------------------------------------------------

    def _init_ui(self):
        self.setWindowTitle("PTM3D Dashboard")
        self.setMinimumSize(1400, 900)

        # Central widget: sidebar + stacked pages
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Sidebar
        self.sidebar = IconSidebar()
        layout.addWidget(self.sidebar)

        # Stacked widget for module pages
        self.module_stack = QStackedWidget()
        layout.addWidget(self.module_stack)

        # Console dock
        self.console_dock = QDockWidget("Console", self)
        self.console_dock.setObjectName("consoleDock")
        self.console_dock.setWidget(self.console_panel)
        self.console_dock.setAllowedAreas(
            Qt.DockWidgetArea.BottomDockWidgetArea
            | Qt.DockWidgetArea.TopDockWidgetArea
        )
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea,
                           self.console_dock)
        self.console_dock.setMinimumHeight(240)

    def _create_menu_bar(self):
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")
        file_menu.addAction(
            QAction("New Project", self, shortcut=QKeySequence.StandardKey.New,
                    triggered=self._on_new_project)
        )
        file_menu.addAction(
            QAction("Open Project...", self,
                    shortcut=QKeySequence.StandardKey.Open,
                    triggered=self._on_open_project)
        )
        file_menu.addAction(
            QAction("Save Project", self,
                    shortcut=QKeySequence.StandardKey.Save,
                    triggered=self._on_save_project)
        )
        file_menu.addSeparator()
        file_menu.addAction(
            QAction("Exit", self, shortcut=QKeySequence("Ctrl+Q"),
                    triggered=self.close)
        )

        # Simulation menu
        sim_menu = menubar.addMenu("Simulation")
        self._run_action = QAction("Run", self, shortcut=QKeySequence("F5"),
                                   triggered=self._on_run)
        self._stop_action = QAction("Stop", self,
                                    shortcut=QKeySequence("Shift+F5"),
                                    triggered=self._on_stop)
        self._stop_action.setEnabled(False)
        sim_menu.addAction(self._run_action)
        sim_menu.addAction(self._stop_action)

        # View menu
        view_menu = menubar.addMenu("View")
        toggle_console = self.console_dock.toggleViewAction()
        toggle_console.setShortcut(QKeySequence("Ctrl+`"))
        view_menu.addAction(toggle_console)
        view_menu.addSeparator()

        # Module shortcuts
        for i, label in enumerate(["DEM/Mesh", "Thermal"]):
            action = QAction(label, self,
                             shortcut=QKeySequence(f"Ctrl+{i + 1}"),
                             triggered=lambda checked, idx=i:
                             self.sidebar.select_module(idx))
            view_menu.addAction(action)

        # Help menu
        help_menu = menubar.addMenu("Help")
        help_menu.addAction(
            QAction("About PTM3D Dashboard", self,
                    triggered=self._on_about)
        )

    def _create_toolbar(self):
        toolbar = self.addToolBar("Main")
        toolbar.setObjectName("mainToolbar")
        toolbar.setMovable(False)

        toolbar.addAction(self._run_action)
        toolbar.addAction(self._stop_action)
        toolbar.addSeparator()

        # Module indicator label
        self._module_label = QLabel("  DEM/Mesh  ")
        self._module_label.setStyleSheet(
            f"color: {ModernStyle.ACCENT}; font-weight: bold; font-size: 13px;"
        )
        toolbar.addWidget(self._module_label)

    def _create_status_bar(self):
        status_bar = self.statusBar()

        self._status_label = QLabel("Ready")
        status_bar.addWidget(self._status_label, 1)

        self._progress_bar = QProgressBar()
        self._progress_bar.setFixedWidth(200)
        self._progress_bar.setVisible(False)
        status_bar.addPermanentWidget(self._progress_bar)

        self._status_indicator = QLabel("Idle")
        self._status_indicator.setStyleSheet(
            f"color: {ModernStyle.TEXT_SECONDARY}; padding: 0 8px;"
        )
        status_bar.addPermanentWidget(self._status_indicator)

    # -- Module Registration -----------------------------------------------

    def _register_modules(self):
        """Import and register all modules."""
        modules_info = [
            ("DEM", "DEM/Mesh Tools"),
            ("THM", "Thermal Simulation"),
        ]

        # Try to import real adapters; fall back to placeholders
        adapters = self._load_adapters()

        for i, (label, tooltip) in enumerate(modules_info):
            self.sidebar.add_module(i, label, tooltip)

            if i < len(adapters) and adapters[i] is not None:
                module = adapters[i]
                self.registry.register(module)
                self.module_stack.addWidget(module.widget())
                # Connect status changes
                module.status_changed.connect(
                    lambda s, idx=i: self._on_module_status(idx, s)
                )
            else:
                # Placeholder page
                placeholder = self._create_placeholder(modules_info[i][1])
                self.module_stack.addWidget(placeholder)

    def _load_adapters(self):
        """Try to import module adapters. Returns list of modules or None."""
        import traceback
        adapters = [None, None]

        loader_info = [
            (0, "DEM", "module_adapters.dem_adapter", "DEMModule"),
            (1, "Thermal", "module_adapters.thermal_adapter", "ThermalModule"),
        ]

        for idx, label, module_path, class_name in loader_info:
            try:
                mod = __import__(module_path, fromlist=[class_name])
                cls = getattr(mod, class_name)
                adapters[idx] = cls(self.console_panel)
            except Exception as e:
                tb = traceback.format_exc()
                self.console_panel.append_warning(
                    f"{label} module not available: {e}"
                )
                # Log full traceback for debugging
                for line in tb.strip().splitlines():
                    self.console_panel.append_output(f"  {line}")

        return adapters

    def _create_placeholder(self, title: str) -> QWidget:
        """Create a placeholder widget for modules not yet loaded."""
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label = QLabel(f"{title}\n\n(Module not yet available)")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet(
            f"color: {ModernStyle.TEXT_SECONDARY}; font-size: 16px;"
        )
        layout.addWidget(label)
        return w

    # -- Slots -------------------------------------------------------------

    def _on_module_selected(self, index: int):
        """Switch to the selected module page."""
        # Deactivate previous module
        prev = self._active_module_index
        modules = self.registry.modules()
        if prev < len(modules):
            modules[prev].on_deactivated()

        self._active_module_index = index
        self.module_stack.setCurrentIndex(index)

        # Activate new module
        if index < len(modules):
            modules[index].on_activated()
            self._module_label.setText(f"  {modules[index].display_name()}  ")
            # Update run/stop button state
            running = modules[index].is_running()
            self._run_action.setEnabled(not running)
            self._stop_action.setEnabled(running)
        else:
            names = ["DEM/Mesh", "Thermal"]
            self._module_label.setText(
                f"  {names[index] if index < len(names) else '?'}  "
            )

    def _on_module_status(self, module_index: int, status: str):
        """Handle status change from a module."""
        if module_index != self._active_module_index:
            return

        self._status_indicator.setText(status.capitalize())

        color_map = {
            "idle": ModernStyle.TEXT_SECONDARY,
            "running": ModernStyle.WARNING,
            "completed": ModernStyle.SUCCESS,
            "failed": ModernStyle.ERROR,
        }
        color = color_map.get(status, ModernStyle.TEXT_SECONDARY)
        self._status_indicator.setStyleSheet(f"color: {color}; padding: 0 8px;")

        if status == "running":
            self._run_action.setEnabled(False)
            self._stop_action.setEnabled(True)
            self._progress_bar.setVisible(True)
            self._progress_bar.setValue(0)
        else:
            self._run_action.setEnabled(True)
            self._stop_action.setEnabled(False)
            self._progress_bar.setVisible(False)

    def _on_run(self):
        """Run the active module's simulation/action."""
        modules = self.registry.modules()
        idx = self._active_module_index
        if idx < len(modules):
            modules[idx].run_simulation()

    def _on_stop(self):
        """Stop the active module's simulation."""
        modules = self.registry.modules()
        idx = self._active_module_index
        if idx < len(modules):
            modules[idx].stop_simulation()

    def _on_new_project(self):
        modules = self.registry.modules()
        idx = self._active_module_index
        if idx < len(modules):
            cm = modules[idx].config_manager()
            if cm and hasattr(cm, 'reset_to_defaults'):
                cm.reset_to_defaults()
                self.console_panel.append_info("Configuration reset to defaults")

    def _on_open_project(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open Configuration",
            str(Path.home()),
            "YAML Files (*.yaml *.yml);;All Files (*)"
        )
        if filepath:
            modules = self.registry.modules()
            idx = self._active_module_index
            if idx < len(modules):
                cm = modules[idx].config_manager()
                if cm and hasattr(cm, 'load_from_yaml'):
                    cm.load_from_yaml(filepath)
                    self.console_panel.append_info(f"Loaded: {filepath}")

    def _on_save_project(self):
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Configuration",
            str(Path.home()),
            "YAML Files (*.yaml *.yml);;All Files (*)"
        )
        if filepath:
            modules = self.registry.modules()
            idx = self._active_module_index
            if idx < len(modules):
                cm = modules[idx].config_manager()
                if cm and hasattr(cm, 'save_to_yaml'):
                    cm.save_to_yaml(filepath)
                    self.console_panel.append_success(f"Saved: {filepath}")

    def _on_about(self):
        QMessageBox.about(
            self,
            "About PTM3D Dashboard",
            f"PTM3D Dashboard v{self.APP_VERSION}\n\n"
            "Planetary Thermophysical Model 3D\n"
            "Unified simulation and analysis suite\n\n"
            "Modules: DEM/Mesh, Thermal\n\n"
            "University of Colorado Boulder"
        )

    # -- State Persistence -------------------------------------------------

    def _restore_state(self):
        geometry = self.settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
        state = self.settings.value("windowState")
        if state:
            self.restoreState(state)

    def closeEvent(self, event):
        # Check for running simulations
        for module in self.registry.modules():
            if module.is_running():
                reply = QMessageBox.question(
                    self,
                    "Simulation Running",
                    f"Module '{module.display_name()}' is still running.\n"
                    "Quit anyway?",
                    QMessageBox.StandardButton.Yes
                    | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if reply == QMessageBox.StandardButton.No:
                    event.ignore()
                    return
                module.stop_simulation()

        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("windowState", self.saveState())
        event.accept()
