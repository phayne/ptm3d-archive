#!/usr/bin/env python3
"""
PTM3D GUI - Modern PyQt6 + PyVista Application
Planetary Thermophysical Model Simulation and Visualization Suite

Two primary capabilities:
1. Configure and run simulations
2. Visualize results in 3D (mesh) and 2D (plots/graphs)
"""

import sys
import os
from pathlib import Path


def _setup_qt_plugin_path():
    """Point Qt at the PyQt6 wheel's own plugins before any Qt import.

    In a conda environment that also ships its own Qt, a qt.conf beside the
    python binary redirects the plugin search away from the PyQt6 wheel and a
    direct launch of this module dies with: Could not find the Qt platform
    plugin "cocoa" in "". gui/run_gui.py has always compensated; launching
    this module directly must work the same way. setdefault so an
    intentionally set environment (run_gui.py, offscreen test harnesses)
    is never overridden.
    """
    try:
        import PyQt6
        plugin_path = os.path.join(os.path.dirname(PyQt6.__file__), 'Qt6', 'plugins')
        if os.path.exists(plugin_path):
            os.environ.setdefault('QT_PLUGIN_PATH', plugin_path)
            platforms_path = os.path.join(plugin_path, 'platforms')
            if os.path.exists(platforms_path):
                os.environ.setdefault('QT_QPA_PLATFORM_PLUGIN_PATH', platforms_path)
    except ImportError:
        pass


_setup_qt_plugin_path()

# PyQt6 imports
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QSplitter, QStatusBar, QMenuBar, QMenu, QToolBar,
    QMessageBox, QFileDialog, QLabel, QProgressBar, QDockWidget,
    QTreeWidget, QTreeWidgetItem, QStackedWidget, QComboBox
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QSettings
from PyQt6.QtGui import QAction, QIcon, QFont, QKeySequence, QPalette, QColor

# Local modules
from gui_core.config_manager import ConfigManager
from gui_core.styles import ModernStyle
from gui_core.simulation_runner import SimulationRunner, SimulationThread, CameraSimThread
from gui_core.preflight import (run_preflight, confirm_preflight,
                                engine_validate_config)

# Panels
from gui_panels.config_panel import ConfigurationPanel
from gui_panels.visualization_panel import VisualizationPanel
from gui_panels.results_panel import ResultsPanel
from gui_panels.console_panel import ConsolePanel
from gui_panels.camera_panel import CameraPanel


class PTM3DMainWindow(QMainWindow):
    """Main application window for PTM3D GUI"""

    def __init__(self):
        super().__init__()

        # Application state
        self.config_manager = ConfigManager()
        self.current_project_file = None
        self.unsaved_changes = False
        self.simulation_thread = None
        self.current_results_dir = None

        # Settings persistence
        self.settings = QSettings("PTM3D", "PTM3DSimulator")

        # Initialize UI
        self.init_ui()
        self.apply_style()
        self.restore_window_state()

        # The body is chosen on the body-first landing screen, which is the
        # first thing this window shows. A modal asking the same question on
        # first launch is now a second, worse copy of it -- it offered a list
        # rather than the cards, and answering it skipped past the screen that
        # exists to make the choice.

        # Give the docked console a comfortable default height once laid out
        # (first launch only; a saved window state wins).
        QTimer.singleShot(0, self._apply_default_console_height)

    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("PTM3D - Planetary Thermophysical Model")
        self.setMinimumSize(1400, 900)

        # Create central widget with main splitter
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Create main horizontal splitter. A wide, non-collapsible handle makes
        # the config <-> visualization boundary easy to grab and drag (a 2px
        # handle is nearly impossible to find); the visualization side takes the
        # extra space when the window is resized.
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.setHandleWidth(6)
        self.main_splitter.setChildrenCollapsible(False)

        # Left panel: Configuration and Results Browser
        self.left_panel = self.create_left_panel()
        self.main_splitter.addWidget(self.left_panel)

        # Right panel: Visualization
        self.right_panel = self.create_right_panel()
        self.main_splitter.addWidget(self.right_panel)

        # Set initial splitter sizes (40% left, 60% right) and let the viz
        # (right) side absorb resize; the config column stays draggable.
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setSizes([560, 840])

        main_layout.addWidget(self.main_splitter)

        # Create docked console at bottom
        self.create_console_dock()

        # Give camera panel a reference to the console (created after it)
        self.camera_panel.console = self.console_panel

        # Create menu bar
        self.create_menu_bar()

        # Create toolbar
        self.create_toolbar()

        # Create status bar
        self.create_status_bar()

        # Connect signals
        self.connect_signals()

    def create_left_panel(self):
        """Create the left panel with tabs for Configuration and Results"""
        left_widget = QWidget()
        layout = QVBoxLayout(left_widget)
        layout.setContentsMargins(0, 0, 0, 0)

        # Tab widget for Configuration / Results Browser
        self.left_tabs = QTabWidget()
        self.left_tabs.setTabPosition(QTabWidget.TabPosition.North)

        # Configuration Panel
        self.config_panel = ConfigurationPanel(self.config_manager)
        self.left_tabs.addTab(self.config_panel, "Configuration")

        # Results Browser Panel
        self.results_panel = ResultsPanel()
        self.left_tabs.addTab(self.results_panel, "Results Browser")

        # Camera Simulation Panel — an advanced tool, hidden by default. It is
        # created here so its wiring (console, run/stop) is in place, but the
        # tab is only added when the user enables it via View > Show Camera Tool.
        self.camera_panel = CameraPanel(
            config_manager=self.config_manager,
            console_panel=getattr(self, 'console_panel', None))
        self.camera_panel.run_requested.connect(self._launch_camera_sim)
        self.camera_panel.stop_requested.connect(self._stop_camera_sim)
        self.camera_sim_thread = None

        layout.addWidget(self.left_tabs)

        return left_widget

    def set_camera_tab_visible(self, visible):
        """Show or hide the (advanced) Camera tool tab. The panel widget is
        kept alive either way; only its presence in the tab bar changes."""
        idx = self.left_tabs.indexOf(self.camera_panel)
        if visible and idx < 0:
            self.left_tabs.addTab(self.camera_panel, "Camera")
        elif not visible and idx >= 0:
            self.left_tabs.removeTab(idx)

    def create_right_panel(self):
        """Create the right panel with visualization"""
        right_widget = QWidget()
        layout = QVBoxLayout(right_widget)
        layout.setContentsMargins(0, 0, 0, 0)

        # Visualization Panel with 3D view and 2D plots
        self.viz_panel = VisualizationPanel()
        layout.addWidget(self.viz_panel)

        return right_widget

    def create_console_dock(self):
        """Create dockable console panel"""
        self.console_dock = QDockWidget("Console Output", self)
        self.console_dock.setObjectName("consoleDock")
        self.console_dock.setAllowedAreas(
            Qt.DockWidgetArea.BottomDockWidgetArea | Qt.DockWidgetArea.TopDockWidgetArea
        )

        self.console_panel = ConsolePanel()
        self.console_dock.setWidget(self.console_panel)

        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.console_dock)

        # Docked and visible on launch, tall enough to read output comfortably.
        # A readable minimum floor plus a taller default height applied once the
        # window is laid out (see _apply_default_console_height, only when there
        # is no saved window state to restore).
        self.console_dock.setVisible(True)
        self.console_dock.setMinimumHeight(240)

    def create_menu_bar(self):
        """Create application menu bar"""
        menubar = self.menuBar()

        # File Menu
        file_menu = menubar.addMenu("&File")

        new_action = QAction("&New Project", self)
        new_action.setShortcut(QKeySequence.StandardKey.New)
        new_action.triggered.connect(self.new_project)
        file_menu.addAction(new_action)

        open_action = QAction("&Open Project...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self.open_project)
        file_menu.addAction(open_action)

        self.recent_menu = file_menu.addMenu("Open &Recent")
        self.update_recent_menu()

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

        load_results_action = QAction("&Load Results Directory...", self)
        load_results_action.setShortcut(QKeySequence("Ctrl+L"))
        load_results_action.triggered.connect(self.load_results_directory)
        file_menu.addAction(load_results_action)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Simulation Menu
        sim_menu = menubar.addMenu("&Simulation")

        run_action = QAction("&Run Simulation", self)
        run_action.setShortcut(QKeySequence("F5"))
        run_action.triggered.connect(self.run_simulation)
        sim_menu.addAction(run_action)

        stop_action = QAction("&Stop Simulation", self)
        stop_action.setShortcut(QKeySequence("Shift+F5"))
        stop_action.triggered.connect(self.stop_simulation)
        stop_action.setEnabled(False)
        self.stop_action = stop_action
        sim_menu.addAction(stop_action)

        sim_menu.addSeparator()

        validate_action = QAction("&Check Configuration", self)
        validate_action.setShortcut(QKeySequence("Ctrl+T"))
        validate_action.triggered.connect(self.validate_config)
        sim_menu.addAction(validate_action)

        # View Menu
        view_menu = menubar.addMenu("&View")

        toggle_console_action = QAction("Toggle &Console", self)
        toggle_console_action.setShortcut(QKeySequence("Ctrl+`"))
        toggle_console_action.triggered.connect(self.toggle_console)
        view_menu.addAction(toggle_console_action)

        view_menu.addSeparator()

        # The two levels. The click-through flow is the primary navigation and
        # the eleven-section rail is hidden until it is asked for, so the menu
        # carries the same two controls the card and the landing screen do --
        # a user who has hidden the rail must have a way back to it that does
        # not depend on remembering which button revealed it.
        self.all_settings_action = QAction("&All Settings (Advanced)", self)
        self.all_settings_action.setShortcut(QKeySequence("Ctrl+Shift+A"))
        self.all_settings_action.triggered.connect(self.show_all_settings)
        view_menu.addAction(self.all_settings_action)

        self.body_first_action = QAction("&Back to the Body Cards", self)
        self.body_first_action.setShortcut(QKeySequence("Ctrl+Shift+B"))
        self.body_first_action.triggered.connect(self.show_body_first)
        view_menu.addAction(self.body_first_action)

        view_menu.addSeparator()

        # Advanced tools: the Camera simulator tab is hidden by default.
        self.show_camera_action = QAction("Show &Camera Tool", self)
        self.show_camera_action.setCheckable(True)
        self.show_camera_action.setChecked(False)
        self.show_camera_action.toggled.connect(self.set_camera_tab_visible)
        view_menu.addAction(self.show_camera_action)

        view_menu.addSeparator()

        reset_layout_action = QAction("&Reset Layout", self)
        reset_layout_action.triggered.connect(self.reset_layout)
        view_menu.addAction(reset_layout_action)

        # Visualization Menu
        viz_menu = menubar.addMenu("V&isualization")

        refresh_viz_action = QAction("&Refresh 3D View", self)
        refresh_viz_action.setShortcut(QKeySequence("F6"))
        refresh_viz_action.triggered.connect(self.refresh_visualization)
        viz_menu.addAction(refresh_viz_action)

        export_image_action = QAction("Export &Image...", self)
        export_image_action.setShortcut(QKeySequence("Ctrl+E"))
        export_image_action.triggered.connect(self.export_visualization)
        viz_menu.addAction(export_image_action)

        viz_menu.addSeparator()

        reset_camera_action = QAction("Reset &Camera", self)
        reset_camera_action.setShortcut(QKeySequence("Home"))
        reset_camera_action.triggered.connect(self.reset_camera)
        viz_menu.addAction(reset_camera_action)

        # Help Menu
        help_menu = menubar.addMenu("&Help")

        docs_action = QAction("&Documentation", self)
        docs_action.setShortcut(QKeySequence.StandardKey.HelpContents)
        docs_action.triggered.connect(self.show_documentation)
        help_menu.addAction(docs_action)

        help_menu.addSeparator()

        about_action = QAction("&About PTM3D", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def create_toolbar(self):
        """Create main toolbar"""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setObjectName("mainToolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        # New/Open/Save
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

        save_as_btn = QAction("Save As", self)
        save_as_btn.setToolTip("Save Project As... (Ctrl+Shift+S)")
        save_as_btn.triggered.connect(self.save_project_as)
        toolbar.addAction(save_as_btn)

        toolbar.addSeparator()

        # Run/Stop
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

        # Visualization controls
        refresh_btn = QAction("Refresh", self)
        refresh_btn.setToolTip("Refresh Visualization (F6)")
        refresh_btn.triggered.connect(self.refresh_visualization)
        toolbar.addAction(refresh_btn)

        reset_cam_btn = QAction("Reset View", self)
        reset_cam_btn.setToolTip("Reset Camera (Home)")
        reset_cam_btn.triggered.connect(self.reset_camera)
        toolbar.addAction(reset_cam_btn)

        toolbar.addSeparator()

        # Timestep navigation
        toolbar.addWidget(QLabel("Timestep: "))
        self.timestep_combo = QComboBox()
        self.timestep_combo.setMinimumWidth(150)
        self.timestep_combo.setEnabled(False)
        self.timestep_combo.currentIndexChanged.connect(self.on_timestep_changed)
        toolbar.addWidget(self.timestep_combo)

        # Data field selector. Items are populated dynamically once a
        # results directory is loaded — see _refresh_field_combo. Until then
        # we leave a placeholder to indicate no results loaded.
        toolbar.addWidget(QLabel("  Field: "))
        self.field_combo = QComboBox()
        self.field_combo.setMinimumWidth(220)
        self.field_combo.addItem("(no results loaded)", None)
        self.field_combo.setEnabled(False)
        self.field_combo.currentIndexChanged.connect(self.on_field_changed)
        toolbar.addWidget(self.field_combo)

    def create_status_bar(self):
        """Create status bar with progress indicator"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Status message
        self.status_label = QLabel("Ready")
        self.status_bar.addWidget(self.status_label, 1)

        # Progress bar (hidden by default)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)

        # Simulation status indicator
        self.sim_status = QLabel()
        self.sim_status.setMinimumWidth(100)
        self.update_simulation_status("idle")
        self.status_bar.addPermanentWidget(self.sim_status)

    def connect_signals(self):
        """Connect internal signals"""
        # Config panel signals
        self.config_panel.config_changed.connect(self.on_config_changed)
        self.config_panel.project_replaced.connect(self.on_project_replaced)
        # "Open a configuration file…" on the landing screen goes through the
        # window's own Open, so it gets the unsaved-changes check, the recent
        # files list and the body-key requirement, rather than a second path
        # that has none of them.
        self.config_panel.open_requested.connect(self.open_project)

        # Results panel signals
        self.results_panel.results_selected.connect(self.on_results_selected)
        self.results_panel.timestep_selected.connect(self.on_timestep_selected)

        # Visualization panel signals
        self.viz_panel.face_selected.connect(self.on_face_selected)

    def apply_style(self):
        """Apply modern styling"""
        self.setStyleSheet(ModernStyle.get_stylesheet())

    # Comfortable default height (px) for the docked console on first launch.
    DEFAULT_CONSOLE_HEIGHT = 340

    def restore_window_state(self):
        """Restore window geometry and state"""
        geometry = self.settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)

        state = self.settings.value("windowState")
        self._had_saved_window_state = bool(state)
        if state:
            self.restoreState(state)

    def _apply_default_console_height(self):
        """Give the docked console a comfortable default height on first launch.

        Runs once the event loop is up (so the window has a real size). Skipped
        when a saved window state was restored, so a user's own console size is
        never overridden."""
        if getattr(self, "_had_saved_window_state", False):
            return
        try:
            self.resizeDocks([self.console_dock],
                             [self.DEFAULT_CONSOLE_HEIGHT],
                             Qt.Orientation.Vertical)
        except Exception:
            pass

    def save_window_state(self):
        """Save window geometry and state"""
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("windowState", self.saveState())

    # === File Operations ===

    def new_project(self):
        """Create a new project"""
        if not self.check_unsaved_changes():
            return

        self.config_manager.reset_to_defaults()
        self.config_panel.load_from_config()
        self.current_project_file = None
        self.unsaved_changes = False
        self.update_window_title()
        self.status_label.setText("New project created")
        self.console_panel.append_info("New project initialized")

    def open_project(self):
        """Open an existing project file"""
        if not self.check_unsaved_changes():
            return

        # Default to examples/ directory, or current project's directory if one is open
        if self.current_project_file:
            start_dir = str(Path(self.current_project_file).parent)
        else:
            # Try examples/ directory relative to GUI location
            gui_dir = Path(__file__).parent.parent
            examples_dir = gui_dir / "examples"
            start_dir = str(examples_dir) if examples_dir.exists() else str(gui_dir)

        # YAML only. The dialog used to offer "Legacy PTM3D Projects (*.ptm3d)"
        # -- a format nothing in this project reads or writes -- and "All Files
        # (*)", which invited a mesh or a binary into yaml.safe_load.
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open Project",
            start_dir,
            "PTM3D configuration (*.yaml *.yml)"
        )

        if filename:
            self.load_project(filename)

    def load_project(self, filename):
        """Load a project file and land on the body card that describes it.

        Two requirements before anything is loaded (DP-7): the file is YAML,
        and it names ``simulation.body``. Neither is a formality. The body key
        is what decides which card describes the file, and there is no honest
        default -- landing a Europa configuration on the Moon card would show
        lunar defaults for properties the file never set, which is the
        hide-a-live-setting class the correctness stage spent fourteen items
        removing.

        Refusing happens BEFORE the load, so a rejected file leaves the current
        project untouched rather than half-replacing it.
        """
        import yaml

        from gui_core.body_cards import open_refusal

        try:
            with open(filename) as handle:
                probe = yaml.safe_load(handle)
        except Exception as e:                               # noqa: BLE001
            QMessageBox.critical(
                self, "Error", f"Failed to read {Path(filename).name}:\n{e}")
            return
        refusal = open_refusal(filename, probe)
        if refusal:
            QMessageBox.warning(self, "This file cannot be opened", refusal)
            self.console_panel.append_warning(f"Not opened: {refusal}")
            self.status_label.setText("Not opened")
            return

        try:
            self.config_manager.load_from_yaml(filename)
            self.config_panel.load_from_config()
            self.current_project_file = filename
            self.unsaved_changes = False
            self.update_window_title()
            self.add_recent_file(filename)
            self.status_label.setText(f"Loaded: {Path(filename).name}")
            self.console_panel.append_info(f"Project loaded: {filename}")
            # Land on the body card for this file, and say what it carries that
            # the card does not show. Anything not shown is preserved by the
            # comment-preserving writer and editable under All settings.
            route = self.config_panel.land_on_loaded_config()
            if route is not None and not route.ok:
                self.console_panel.append_warning(
                    f"No body card describes this file, so it stayed on the "
                    f"body chooser: {route.refusal} Everything in the file is "
                    f"loaded; use All settings to see and edit it.")
        except Exception as e:                               # noqa: BLE001
            QMessageBox.critical(self, "Error", f"Failed to load project:\n{str(e)}")

    def save_project(self):
        """Save current project"""
        if self.current_project_file:
            self.save_project_to(self.current_project_file)
        else:
            self.save_project_as()

    def save_project_as(self):
        """Save project with new filename"""
        # Default to examples/ directory, or current project's directory if one is open
        if self.current_project_file:
            start_dir = str(Path(self.current_project_file).parent)
        else:
            gui_dir = Path(__file__).parent.parent
            examples_dir = gui_dir / "examples"
            start_dir = str(examples_dir) if examples_dir.exists() else str(gui_dir)

        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Project",
            start_dir,
            "YAML Files (*.yaml);;All Files (*)"
        )

        if filename:
            if not filename.endswith(('.yaml', '.yml')):
                filename += '.yaml'
            self.save_project_to(filename)

    def save_project_to(self, filename):
        """Save project to specified file.

        Whether this save preserved the file's comments and layout is REPORTED,
        not left to be discovered. A lossy save (ruamel.yaml missing) also
        raises a dialog the first time in a session, because the loss is
        silent, permanent and to the user's own file.
        """
        preserved, status = self.config_manager.comment_preservation_status()
        if not preserved and not self.config_manager.comment_loss_disclosed \
                and Path(filename).exists():
            reply = QMessageBox.warning(
                self, "This save cannot preserve comments",
                f"{status}\n\nSave {Path(filename).name} anyway?",
                QMessageBox.StandardButton.Save |
                QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel)
            if reply != QMessageBox.StandardButton.Save:
                self.console_panel.append_warning("Save canceled")
                return
            # Only a save that actually WENT AHEAD consumes the session's one
            # dialog. Setting the flag before the dialog returned meant a user
            # who read the warning and pressed Cancel had spent it, and every
            # later save flattened the file with nothing but a console line.
            self.config_manager.comment_loss_disclosed = True
        try:
            edits = self.config_manager.plan_save_edits()
            mode = self.config_manager.save_to_yaml(filename)
            self.current_project_file = filename
            self.unsaved_changes = False
            self.update_window_title()
            self.add_recent_file(filename)
            self.status_label.setText(f"Saved: {Path(filename).name}")
            self.console_panel.append_info(f"Project saved: {filename}")
            # NOT in replaced_other_file mode: there the target was replaced
            # wholesale by a different file's document, so "Rewrote 2 key(s)"
            # materially understates it and contradicts the warning printed
            # immediately below.
            if edits is not None and mode != "replaced_other_file":
                updates, removals = edits
                touched = sorted(list(updates) + list(removals))
                self.console_panel.append_info(
                    "Rewrote %d key(s): %s"
                    % (len(touched),
                       ", ".join(touched[:8]) +
                       (", ..." if len(touched) > 8 else "")
                       if touched else "none -- the file is unchanged"))
            if mode == "unchanged":
                self.console_panel.append_info(
                    "Nothing to write: the file already matches this project, "
                    "so it was left untouched.")
            elif mode == "in_place":
                self.console_panel.append_info(
                    "Comments and layout preserved.")
            elif mode == "copied_to_new_file":
                self.console_panel.append_info(
                    "Written to a new file, carrying this project's comments "
                    "and layout.")
            elif mode == "replaced_other_file":
                self.console_panel.append_warning(
                    f"{Path(filename).name} already existed and was REPLACED "
                    f"by this project. Its own comments and any keys it "
                    f"carried that this project does not are gone.")
            else:
                self.console_panel.append_warning(status)
            reflowed = getattr(self.config_manager,
                               "last_save_layout_changes", [])
            if reflowed:
                self.console_panel.append_warning(
                    "%d line(s) were reformatted that carry no key you edited "
                    "(a YAML layout the writer could not reproduce exactly): %s"
                    % (len(reflowed), " | ".join(reflowed[:4])))
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save project:\n{str(e)}")

    def load_results_directory(self):
        """Load simulation results from a directory"""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Results Directory",
            str(Path.home()),
            QFileDialog.Option.ShowDirsOnly
        )

        if directory:
            self.results_panel.load_results(directory)
            self.current_results_dir = directory
            self.left_tabs.setCurrentWidget(self.results_panel)
            self.console_panel.append_info(f"Results loaded from: {directory}")

    def check_unsaved_changes(self):
        """Check for unsaved changes and prompt user"""
        if not self.unsaved_changes:
            return True

        reply = QMessageBox.question(
            self, "Unsaved Changes",
            "You have unsaved changes. Do you want to save?",
            QMessageBox.StandardButton.Save |
            QMessageBox.StandardButton.Discard |
            QMessageBox.StandardButton.Cancel
        )

        if reply == QMessageBox.StandardButton.Save:
            self.save_project()
            return not self.unsaved_changes
        elif reply == QMessageBox.StandardButton.Discard:
            return True
        else:
            return False

    def update_window_title(self):
        """Update window title with project name"""
        title = "PTM3D - Planetary Thermophysical Model"
        if self.current_project_file:
            title += f" - {Path(self.current_project_file).name}"
        if self.unsaved_changes:
            title += " *"
        self.setWindowTitle(title)

    def add_recent_file(self, filename):
        """Add file to recent files list"""
        recent = self.settings.value("recentFiles", [])
        if isinstance(recent, str):
            recent = [recent] if recent else []

        if filename in recent:
            recent.remove(filename)
        recent.insert(0, filename)
        recent = recent[:10]  # Keep last 10

        self.settings.setValue("recentFiles", recent)
        self.update_recent_menu()

    def update_recent_menu(self):
        """Update recent files menu"""
        self.recent_menu.clear()
        recent = self.settings.value("recentFiles", [])
        if isinstance(recent, str):
            recent = [recent] if recent else []

        for filepath in recent:
            action = QAction(Path(filepath).name, self)
            action.setData(filepath)
            action.triggered.connect(lambda checked, f=filepath: self.load_project(f))
            self.recent_menu.addAction(action)

        if not recent:
            action = QAction("(No recent files)", self)
            action.setEnabled(False)
            self.recent_menu.addAction(action)

    # === Simulation Operations ===

    def run_simulation(self):
        """Run the PTM simulation"""
        # Validate configuration first
        is_valid, errors = self.config_manager.validate()
        if not is_valid:
            QMessageBox.warning(
                self, "Validation Failed",
                "Configuration validation failed:\n\n" + "\n".join(f"• {e}" for e in errors)
            )
            return

        # A run NEVER writes the file the user opened. Running used to call
        # save_project() first, which is yaml.dump of the merged dictionary
        # over the author's own file: on the two shipped examples that erased
        # 10 and 19 comments, added 33 and 42 keys, and silently changed
        # illumination.albedo_maps_directory to ''. A config is a scientific
        # record; a run that rewrites its own inputs cannot be audited
        # afterward. Opening is now read-only, and the project file changes
        # only when the user presses Save or Save As.
        #
        # What runs instead is an engine-facing YAML written BESIDE the output
        # directory, with every path absolutized against the project root
        # (save_run_yaml) -- absolutized paths resolve regardless of the YAML's
        # location, so nothing is mis-resolved against the output dir. It is
        # not promoted to "the project file".
        config_dict = self.config_manager.get_config()
        from gui_core.preflight import find_project_root, resolve_path
        exe = SimulationRunner.find_ptm_executable()
        root = find_project_root(exe) if exe else Path.cwd()
        output_dir = resolve_path(config_dict["output"]["directory"], root)
        run_config_path = str(output_dir / "config.yaml")
        # Refuse to land on the user's own file even if the output directory
        # happens to hold it.
        if self.current_project_file and Path(run_config_path).resolve() == \
                Path(self.current_project_file).resolve():
            run_config_path = str(output_dir / "config_run.yaml")

        # Clear the console HERE, not after the pre-flight. It used to be
        # cleared just before the "Starting Simulation" header, which erased
        # every [preflight] line the user had just been shown -- including the
        # engine-environment banner that discloses a change in physics, and the
        # cloud-path and view-factor advisories.
        self.console_panel.clear()
        self.console_panel.append_header("Starting Simulation")

        # Pre-flight validation: paths exist and are readable, no dataless
        # (cloud placeholder) files, cloud-path override, disk space, VF
        # face-count guard, executable fingerprint. A config that fails
        # pre-flight would previously hang or crash mid-run with no
        # diagnosis (e.g. an OneDrive topo_directory of evicted files).
        report = run_preflight(config_dict, SimulationRunner.find_ptm_executable)
        for line in report.summary_lines():
            # summary_lines() already tags severity; route it so a warning is
            # colored as one. Everything went through append_info before, which
            # made "WARNING: engine environment changed ..." look like a note.
            if line.startswith(("ERROR:", "DATALESS:")):
                self.console_panel.append_error(f"[preflight] {line}")
            elif line.startswith(("WARNING:", "CLOUD:", "VF-GUARD:",
                                  "BEAM-GUARD:")):
                self.console_panel.append_warning(f"[preflight] {line}")
            else:
                self.console_panel.append_info(f"[preflight] {line}")
        if not confirm_preflight(self, report):
            self.console_panel.append_warning("Launch canceled by pre-flight check")
            return

        # The run YAML is written only once the launch is going ahead. Writing
        # it before the gate left a config.yaml in the output directory after a
        # blocked or canceled launch -- overwriting the previous run's record
        # with a configuration that never ran, while the tutorial tells the user
        # this file "is the exact configuration that ran".
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            self.config_manager.save_run_yaml(run_config_path, root)
        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Failed to write run config:\n{str(e)}")
            return

        # Create simulation thread (cwd is pinned to the project root and the
        # engine env flags are applied inside SimulationThread)
        try:
            self.simulation_thread = SimulationThread(
                run_config_path,
                config_dict=config_dict
            )
            self.simulation_thread.output_ready.connect(self.on_simulation_output)
            self.simulation_thread.warning_ready.connect(self.on_simulation_warning)
            self.simulation_thread.error_ready.connect(self.on_simulation_error)
            self.simulation_thread.progress_update.connect(self.on_simulation_progress)
            self.simulation_thread.heartbeat.connect(self.on_simulation_heartbeat)
            self.simulation_thread.finished.connect(self.on_simulation_finished)

            # Update UI
            self.run_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.stop_action.setEnabled(True)
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.update_simulation_status("running")

            # The engine's config is the generated run YAML, not the project
            # file. Naming the project file here said the run read a file it
            # never opens. (The console was cleared before the pre-flight, so
            # those lines are still on screen above this one.)
            self.console_panel.append_info(f"Config: {run_config_path}")
            if self.current_project_file:
                self.console_panel.append_info(
                    f"Project file NOT modified: {self.current_project_file} "
                    f"(use Save to write your edits back to it)")

            # Start
            self.simulation_thread.start()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start simulation:\n{str(e)}")
            self.reset_simulation_ui()

    def stop_simulation(self):
        """Stop the running simulation"""
        if self.simulation_thread:
            self.console_panel.append_warning("Stopping simulation...")
            self.simulation_thread.stop()

    def on_simulation_output(self, text):
        """Handle an informational simulation line"""
        self.console_panel.append_output(text)

    def on_simulation_warning(self, text):
        """Handle a warning line (e.g. machine-profile conservative defaults)"""
        self.console_panel.append_warning(text)

    def on_simulation_error(self, text):
        """Handle a genuine-error line"""
        self.console_panel.append_error(text)

    def on_simulation_progress(self, percent):
        """Handle simulation progress update"""
        self.progress_bar.setValue(int(percent))

    def on_simulation_heartbeat(self, message, stalled):
        """Watchdog heartbeat: distinguish a hung file read from computation."""
        if stalled:
            self.status_label.setText(f"STALLED -- {message}")
            self.console_panel.append_warning(f"[watchdog] {message}")
        else:
            self.status_label.setText("Simulation running")
            self.console_panel.append_info(f"[watchdog] {message}")

    def on_simulation_finished(self, exit_code):
        """Handle simulation completion"""
        if exit_code == 0:
            self.console_panel.append_success("Simulation completed successfully")
            self.update_simulation_status("completed")

            # Auto-load results. Resolve a relative output directory against
            # the same project root the engine ran in (SimulationThread pins
            # its cwd there), NOT against the GUI's own cwd.
            from gui_core.preflight import find_project_root, resolve_path
            config = self.config_manager.get_config()
            exe = SimulationRunner.find_ptm_executable()
            root = find_project_root(exe) if exe else Path.cwd()
            output_dir = resolve_path(config["output"]["directory"], root)
            if output_dir.exists():
                self.results_panel.load_results(str(output_dir))
                self.current_results_dir = str(output_dir)
                self.left_tabs.setCurrentWidget(self.results_panel)
        else:
            self.console_panel.append_error(f"Simulation failed (exit code: {exit_code})")
            self.update_simulation_status("failed")

        self.reset_simulation_ui()

    def reset_simulation_ui(self):
        """Reset UI after simulation"""
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.stop_action.setEnabled(False)
        self.progress_bar.setVisible(False)
        # Drop our reference to the worker thread ONLY after it has fully
        # terminated. SimulationThread.finished is a custom pyqtSignal
        # emitted from *inside* run() just before it returns, so this slot
        # (on_simulation_finished -> reset_simulation_ui) can execute while
        # the QThread is still winding down. Deallocating the wrapper then
        # triggers "QThread: Destroyed while thread is still running" and
        # SIGABRT -- reliably on fast-exiting runs (e.g. topography enabled
        # but no topography data, where the engine fails loud in <1 s).
        # wait() returns promptly because run() has essentially completed;
        # holding `thread` across the wait keeps the wrapper alive until the
        # OS thread is truly finished, so the later GC is safe.
        thread = self.simulation_thread
        self.simulation_thread = None
        if thread is not None:
            thread.wait()

    # === Camera Simulation ===

    def _launch_camera_sim(self, command):
        """Launch ptm_camera_sim in a background thread."""
        if self.camera_sim_thread and self.camera_sim_thread.isRunning():
            self.console_panel.append_error("Camera simulation already running")
            return

        self.camera_sim_thread = CameraSimThread(command)
        self.camera_sim_thread.output_ready.connect(self._on_camera_output)
        self.camera_sim_thread.error_ready.connect(
            self.console_panel.append_error)
        self.camera_sim_thread.frame_rendered.connect(
            self.camera_panel._add_thumbnail)
        self.camera_sim_thread.finished.connect(
            self.camera_panel.on_simulation_finished)
        self.camera_sim_thread.start()

    def _on_camera_output(self, line):
        """Route camera sim stdout to console and camera panel."""
        self.console_panel.append(line)
        self.camera_panel.on_output_line(line)

    def _stop_camera_sim(self):
        """Stop the running camera simulation."""
        if self.camera_sim_thread:
            self.camera_sim_thread.stop()

    def update_simulation_status(self, status):
        """Update simulation status indicator"""
        status_styles = {
            "idle": ("Idle", "#888888"),
            "running": ("Running", "#ffcc00"),
            "completed": ("Completed", "#00cc00"),
            "failed": ("Failed", "#cc0000")
        }
        text, color = status_styles.get(status, ("Unknown", "#888888"))
        self.sim_status.setText(f"  {text}  ")
        self.sim_status.setStyleSheet(f"background-color: {color}; color: white; border-radius: 3px; padding: 2px 5px;")

    def validate_config(self):
        """Check the configuration -- the GUI's rules AND the engine's own.

        Item 2.4 / DP-6. The GUI kept a second, weaker copy of the engine's
        rules, and the two have already disagreed in this session: a draft
        eq_indirect_source check invented a valid set that would have refused
        the one non-default value the engine accepts. So the engine is asked
        directly (`ptm_thermal_sim --validate-config`) and its words are
        reported verbatim.

        An engine that cannot be asked is reported as NOT CHECKED, never as
        checked-and-fine.
        """
        is_valid, messages = self.config_manager.validate()
        gui_part = (
            ("Errors:\n" if not is_valid else "Warnings:\n")
            + "\n".join(f"  • {m}" for m in messages)
            if messages else "The GUI's own checks found nothing.")

        # The ENGINE-FACING view, not the raw GUI dictionary. They differ
        # materially, and validating the wrong one is worse than not
        # validating: get_config() always carries a full gpu: section, so the
        # temporary file would set min_vram_margin_explicit -- the opposite of
        # what the run does and of what the GPU page tells the user. It also
        # carries GUI-synthetic keys (topo_source, albedo_source,
        # margin_explicit, the use_default_* switches) and none of the
        # translations _config_for_yaml performs, so the "PTM_DATA_ROOT
        # path-resolution contract" the banner claims to have checked would
        # have been checked against paths the run will not have.
        exe = SimulationRunner.find_ptm_executable()
        self.status_label.setText("Checking configuration…")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            status, engine_text = engine_validate_config(
                self.config_manager._config_for_yaml(), exe)
        finally:
            QApplication.restoreOverrideCursor()
            self.status_label.setText("Ready")

        engine_part = {
            "accepted": "The engine ACCEPTED this configuration:",
            "refused": "The engine REFUSED this configuration:",
            "unavailable": "The engine was NOT asked:",
        }[status] + "\n" + (engine_text.strip() or "(no output)")

        # The console keeps the full engine text; the dialog stays readable.
        self.console_panel.append_header("Check Configuration")
        for line in gui_part.splitlines():
            self.console_panel.append_info(f"[check] {line}")
        for line in engine_part.splitlines():
            if status == "refused":
                self.console_panel.append_error(f"[check] {line}")
            elif status == "unavailable":
                self.console_panel.append_warning(f"[check] {line}")
            else:
                self.console_panel.append_info(f"[check] {line}")

        def _clip(text, limit=2400):
            return text if len(text) <= limit else (
                text[:limit] + "\n… (full text in the console)")

        body = _clip(gui_part + "\n\n" + engine_part)
        if not is_valid or status == "refused":
            QMessageBox.warning(self, "Check Configuration", body)
        elif status == "unavailable":
            QMessageBox.warning(self, "Check Configuration — not fully checked",
                                body)
        else:
            QMessageBox.information(self, "Check Configuration", body)

    # === View Operations ===

    def show_all_settings(self):
        """View > All Settings (Advanced). The window's half of the two levels.

        The configuration panel owns the switch; this exists so the menu and
        any future toolbar button go through one path rather than reaching into
        the panel's widgets. It also makes sure the left column is showing the
        configuration and not the Results Browser, which would otherwise leave
        the menu item looking like it did nothing.
        """
        self.left_tabs.setCurrentWidget(self.config_panel)
        self.config_panel.show_all_settings()

    def show_body_first(self):
        """View > Back to the Body Cards. Returns into the click-through flow.

        Nothing is discarded: the panel re-mirrors the live card from the
        configuration on the way back, so the card and the section pages agree.
        """
        self.left_tabs.setCurrentWidget(self.config_panel)
        self.config_panel.show_body_first()

    def toggle_console(self):
        """Toggle console dock visibility"""
        self.console_dock.setVisible(not self.console_dock.isVisible())

    def reset_layout(self):
        """Reset window layout to defaults"""
        self.main_splitter.setSizes([560, 840])
        self.console_dock.setVisible(True)
        self.console_dock.setFloating(False)
        self.resizeDocks([self.console_dock],
                         [self.DEFAULT_CONSOLE_HEIGHT],
                         Qt.Orientation.Vertical)
        self.status_label.setText("Layout reset to defaults")

    # === Visualization Operations ===

    def refresh_visualization(self):
        """Refresh the 3D visualization"""
        if self.current_results_dir:
            self.viz_panel.refresh()

    def export_visualization(self):
        """Export current visualization as image"""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Image",
            str(Path.home() / "ptm3d_visualization.png"),
            "PNG Images (*.png);;JPEG Images (*.jpg);;All Files (*)"
        )

        if filename:
            self.viz_panel.export_image(filename)
            self.status_label.setText(f"Image exported: {Path(filename).name}")

    def reset_camera(self):
        """Reset camera to default view"""
        self.viz_panel.reset_camera()

    def on_timestep_changed(self, index):
        """Handle timestep selection from toolbar"""
        if index >= 0 and self.current_results_dir:
            self.viz_panel.set_timestep(index)
            self.results_panel.select_timestep(index)

    def on_field_changed(self, _index):
        """Handle data field selection. Field key is stored as the combo's
        UserRole data so we don't have to reverse-look-up by label."""
        field = self.field_combo.currentData()
        if field:
            self.viz_panel.set_field(field)

    def _refresh_field_combo(self, results_dir):
        """Repopulate the toolbar field selector based on what's actually
        present in the results directory. Mirrors VisualizationPanel's
        FIELD_INFO + per-step + statistics scan but lives on the toolbar.
        """
        from pathlib import Path
        from gui_panels.visualization_panel import VisualizationPanel
        results_path = Path(results_dir)
        stats_dir = results_path / "statistics"

        self.field_combo.blockSignals(True)
        self.field_combo.clear()

        available = []
        for field_key, (label, _cmap) in VisualizationPanel.FIELD_INFO.items():
            # Use the canonical classifier so any file-naming change (e.g.
            # the new single-file Schorghofer ice_depth output) is picked
            # up automatically.
            if VisualizationPanel.is_stats_field(field_key):
                if (stats_dir / f"{field_key}.bin").is_file():
                    available.append((field_key, label))
            else:
                d = results_path / field_key
                if d.is_dir() and any(d.glob("*.bin")):
                    available.append((field_key, label))

        if not available:
            self.field_combo.addItem("(no data found)", None)
            self.field_combo.setEnabled(False)
        else:
            for field_key, label in available:
                self.field_combo.addItem(label, field_key)
            self.field_combo.setEnabled(True)
            # Default selection: prefer time-series temperature, then
            # temperature_max, then first available.
            preferred_order = ("temperature", "temperature_max")
            default_idx = 0
            for pref in preferred_order:
                for i, (k, _l) in enumerate(available):
                    if k == pref:
                        default_idx = i
                        break
                else:
                    continue
                break
            self.field_combo.setCurrentIndex(default_idx)

        self.field_combo.blockSignals(False)
        # Trigger one synthesize change so the viz panel applies the default
        if self.field_combo.isEnabled():
            self.on_field_changed(self.field_combo.currentIndex())

    # === Signal Handlers ===

    def on_project_replaced(self, source_path):
        """The whole configuration came from another file (the baseline card).

        Detach the project file. The ConfigManager's round-trip document now
        belongs to `source_path`, so leaving current_project_file alone made
        the next Ctrl+S write the BASELINE's document -- comments and all --
        over the user's own project, and report "Comments and layout
        preserved" while doing it. Reproduced before this handler existed.

        The baseline's own record is not adopted as the project file either: it
        is a run's evidence and must not be overwritten by a Save. The user
        gets Save As.
        """
        self.current_project_file = None
        self.unsaved_changes = True
        self.update_window_title()
        self.status_label.setText(f"Loaded baseline: {Path(source_path).name}")
        self.console_panel.append_warning(
            f"Configuration replaced from {source_path}. This is NOT your "
            f"project file -- Save will ask for a new location, and the "
            f"baseline's own record is left untouched.")

    def on_config_changed(self):
        """Handle configuration changes"""
        self.unsaved_changes = True
        self.update_window_title()

    def on_results_selected(self, results_dir):
        """Handle results directory selection"""
        self.current_results_dir = results_dir
        self.viz_panel.load_results(results_dir)

        # Populate timestep combo
        timesteps = self.results_panel.get_timesteps()
        self.timestep_combo.clear()
        self.timestep_combo.addItems([f"t={i}" for i in range(len(timesteps))])
        self.timestep_combo.setEnabled(len(timesteps) > 0)

        # Repopulate the toolbar field selector from FIELD_INFO + the actual
        # contents of the chosen results dir (per-step series + statistics).
        self._refresh_field_combo(results_dir)

    def on_timestep_selected(self, timestep_index, timestep_time):
        """Handle timestep selection from results panel"""
        self.timestep_combo.blockSignals(True)
        self.timestep_combo.setCurrentIndex(timestep_index)
        self.timestep_combo.blockSignals(False)
        self.viz_panel.set_timestep(timestep_index)

    def on_face_selected(self, face_id, data):
        """Handle face selection in 3D view"""
        self.results_panel.show_face_details(face_id, data)

    # === Help ===

    def show_documentation(self):
        """Show documentation"""
        QMessageBox.information(
            self, "Documentation",
            "PTM3D - Planetary Thermophysical Model\n\n"
            "Quick Start:\n"
            "1. Configure simulation parameters\n"
            "2. Click Run to start simulation\n"
            "3. View results in 3D visualization\n"
            "4. Analyze with 2D plots\n\n"
            "Keyboard Shortcuts:\n"
            "• F5 - Run simulation\n"
            "• F6 - Refresh visualization\n"
            "• Home - Reset camera\n"
            "• Ctrl+L - Load results\n"
            "• Ctrl+` - Toggle console"
        )

    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self, "About PTM3D",
            "<h2>PTM3D</h2>"
            "<p>Planetary Thermophysical Model</p>"
            "<p>Version 2.0</p>"
            "<hr>"
            "<p>A simulation and visualization tool for planetary surface "
            "thermal modeling with support for:</p>"
            "<ul>"
            "<li>3D mesh visualization with PyVista</li>"
            "<li>Real-time thermal simulation</li>"
            "<li>Radiation exchange calculations</li>"
            "<li>Multi-format data export</li>"
            "</ul>"
            "<p>Built with PyQt6 and PyVista</p>"
        )

    # === Window Events ===

    def closeEvent(self, event):
        """Handle window close"""
        if not self.check_unsaved_changes():
            event.ignore()
            return

        # Stop any running simulation
        if self.simulation_thread and self.simulation_thread.isRunning():
            reply = QMessageBox.question(
                self, "Simulation Running",
                "A simulation is still running. Stop it and exit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                event.ignore()
                return
            # stop() escalates TERM -> KILL on the child's process group.
            # Bound the thread wait so a child stuck in an uninterruptible
            # cloud-file read cannot hang the GUI on exit (the reader thread
            # is a daemon of this process and dies with it; the process
            # group has already been signaled).
            self.simulation_thread.stop()
            if not self.simulation_thread.wait(7000):
                self.simulation_thread.wait(2000)

        # Clean up visualization
        self.viz_panel.cleanup()

        # Save window state
        self.save_window_state()

        event.accept()


def main():
    """Main entry point"""
    # High DPI support
    os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"

    app = QApplication(sys.argv)
    app.setApplicationName("PTM3D")
    app.setOrganizationName("PTM3D")
    app.setApplicationVersion("2.0")

    # Set application-wide font (use system font on macOS)
    font = QFont("Helvetica Neue", 10)
    app.setFont(font)

    # Create and show main window
    window = PTM3DMainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
