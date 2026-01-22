#!/usr/bin/env python3
"""
PTM3D Configuration Builder - Retro-Futuristic GUI
A visual parameter file generator for the Planetary Thermophysical Model
"""

import sys
import os
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QTabWidget, QPushButton, QLabel,
                             QFileDialog, QMessageBox, QSplitter, QTextEdit,
                             QStatusBar, QMenuBar, QMenu, QAction)
from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QFont, QPalette, QColor, QIcon

from gui_styles import RetroStyle
from gui_tabs import (SimulationTab, MeshTab, ThermalTab, RadiationTab,
                      IlluminationTab, OutputTab, VisualizationTab, AdvancedTab)
from config_manager import ConfigManager
os.environ['QT_LOGGING_RULES'] = 'qt.qpa.fonts=false'

from gui_control_panel import ControlPanel
from gui_console_dock import ConsoleDock


class PTMConfigGUI(QMainWindow):
    """Main application window with retro-futuristic styling"""
    
    def __init__(self):
        super().__init__()
        self.config_manager = ConfigManager()
        self.current_file = None
        self.unsaved_changes = False
        
        self.init_ui()
        self.apply_retro_theme()
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("PTM3D Configuration Builder v1.0")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 1200)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Header
        header = self.create_header()
        main_layout.addWidget(header)
        
        # === NEW: Control Panel ===
        self.control_panel = ControlPanel(self.config_manager, self)
        main_layout.addWidget(self.control_panel)
        
        # Create splitter for main content and preview
        splitter = QSplitter(Qt.Horizontal)
        
        # Left side: Parameter tabs
        tabs_widget = self.create_tabs()
        splitter.addWidget(tabs_widget)
        
        # Right side: Preview
        preview_widget = self.create_preview()
        splitter.addWidget(preview_widget)
        
        # Set initial splitter sizes (70% tabs, 30% preview)
        splitter.setSizes([840, 360])
        
        main_layout.addWidget(splitter, 1)
        
        # === NEW: Console Dock ===
        self.console_dock = ConsoleDock()
        self.console_dock.setMinimumHeight(400)
        self.console_output = self.console_dock.console_output  # Expose for control panel
        main_layout.addWidget(self.console_dock)
        
        # Footer with buttons
        footer = self.create_footer()
        main_layout.addWidget(footer)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.status_light = QLabel("●")
        self.status_light.setStyleSheet("color: #00ff00;")
        self.status_bar.addPermanentWidget(self.status_light)
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready", 3000)
        
        # Create menu bar
        self.create_menu_bar()
        
        # Animate status light
        self.animate_status_light()
        
        # Update the YAML preview to reflect the loaded defaults
        self.update_preview()
        
    def create_header(self):
        """Create the retro-futuristic header"""
        header = QWidget()
        header.setObjectName("header")
        header.setFixedHeight(110)  # Taller to accommodate larger fonts
        layout = QHBoxLayout(header)
        layout.setContentsMargins(20, 10, 20, 10)
        
        # Title
        title = QLabel("PTM3D: Planetary Thermophysical Model")
        title.setObjectName("mainTitle")
        title_font = QFont("Courier New", 32, QFont.Bold)  # kept as fallback
        title.setFont(title_font)
        
        # Subtitle
        subtitle = QLabel("CONFIGURATION BUILDER")
        subtitle.setObjectName("subtitle")
        subtitle_font = QFont("Courier New", 16)  # fallback; CSS overrides to 20px
        subtitle.setFont(subtitle_font)
        
        # Arrange title and subtitle
        title_layout = QVBoxLayout()
        title_layout.addWidget(title)
        title_layout.addWidget(subtitle)
        title_layout.setSpacing(0)
        
        layout.addLayout(title_layout)
        layout.addStretch()
        
        # Status indicator (animated)
        self.status_light = QLabel("●")
        self.status_light.setObjectName("statusLight")
        status_font = QFont("Arial", 24)
        self.status_light.setFont(status_font)
        layout.addWidget(self.status_light)
        
        # Animate status light
        self.animate_status_light()
        
        return header
        
    def create_tabs(self):
        """Create the tabbed parameter interface"""
        self.tab_widget = QTabWidget()
        self.tab_widget.setObjectName("mainTabs")

        # Create tabs
        self.sim_tab = SimulationTab(self.config_manager)
        self.mesh_tab = MeshTab(self.config_manager)
        self.rad_tab = RadiationTab(self.config_manager)  # NEW: Radiation before Thermal
        self.thermal_tab = ThermalTab(self.config_manager)
        self.illum_tab = IlluminationTab(self.config_manager)
        self.output_tab = OutputTab(self.config_manager)
        self.vis_tab = VisualizationTab(self.config_manager)
        self.adv_tab = AdvancedTab(self.config_manager)

        # Add tabs (Radiation comes before Thermal per user preference)
        self.tab_widget.addTab(self.sim_tab, "⚙ SIMULATION")
        self.tab_widget.addTab(self.mesh_tab, "◬ MESH")
        self.tab_widget.addTab(self.rad_tab, "☢ RADIATION")  # NEW
        self.tab_widget.addTab(self.thermal_tab, "♨ THERMAL")
        self.tab_widget.addTab(self.illum_tab, "☀ ILLUMINATION")
        self.tab_widget.addTab(self.output_tab, "💾 OUTPUT")
        self.tab_widget.addTab(self.vis_tab, "📊 VISUALIZATION")
        self.tab_widget.addTab(self.adv_tab, "⚡ ADVANCED")

        # Connect tab changes to preview updates
        self.tab_widget.currentChanged.connect(self.update_preview)

        # Connect value changes to mark as unsaved
        for tab in [self.sim_tab, self.mesh_tab, self.rad_tab, self.thermal_tab,
                    self.illum_tab, self.output_tab, self.vis_tab, self.adv_tab]:
            tab.value_changed.connect(self.on_value_changed)

        # Load default values into all tabs on startup
        self.load_defaults_into_tabs()

        return self.tab_widget

    def load_defaults_into_tabs(self):
        """Load default configuration values into all tab widgets"""
        # Call load_from_config() on each tab to populate widgets
        # with values from config_manager.DEFAULT_CONFIG
        self.sim_tab.load_from_config()
        self.mesh_tab.load_from_config()
        self.rad_tab.load_from_config()  # NEW
        self.thermal_tab.load_from_config()
        self.illum_tab.load_from_config()
        self.output_tab.load_from_config()
        self.vis_tab.load_from_config()
        self.adv_tab.load_from_config()
        
    def create_preview(self):
        """Create the YAML preview panel"""
        preview_widget = QWidget()
        preview_widget.setObjectName("previewPanel")
        
        layout = QVBoxLayout(preview_widget)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Header
        header = QLabel("YAML PREVIEW")
        header.setObjectName("previewHeader")
        header_font = QFont("Courier New", 12, QFont.Bold)
        header.setFont(header_font)
        layout.addWidget(header)
        
        # Preview text area
        self.preview_text = QTextEdit()
        self.preview_text.setObjectName("previewText")
        self.preview_text.setReadOnly(True)
        self.preview_text.setFont(QFont("Courier New", 10))
        layout.addWidget(self.preview_text)
        
        # Update button
        update_btn = QPushButton("↻ REFRESH PREVIEW")
        update_btn.setObjectName("refreshButton")
        update_btn.clicked.connect(self.update_preview)
        layout.addWidget(update_btn)
        
        return preview_widget
        
    def create_footer(self):
        """Create the action buttons footer"""
        footer = QWidget()
        footer.setObjectName("footer")
        footer.setFixedHeight(80)
        
        layout = QHBoxLayout(footer)
        layout.setContentsMargins(20, 10, 20, 10)
        layout.setSpacing(15)
        
        # Action buttons
        new_btn = QPushButton("NEW CONFIG")
        new_btn.setObjectName("actionButton")
        new_btn.clicked.connect(self.new_config)
        
        load_btn = QPushButton("LOAD CONFIG")
        load_btn.setObjectName("actionButton")
        load_btn.clicked.connect(self.load_config)
        
        save_btn = QPushButton("SAVE CONFIG")
        save_btn.setObjectName("actionButton")
        save_btn.clicked.connect(self.save_config)
        
        save_as_btn = QPushButton("SAVE AS...")
        save_as_btn.setObjectName("actionButton")
        save_as_btn.clicked.connect(self.save_config_as)
        
        validate_btn = QPushButton("VALIDATE")
        validate_btn.setObjectName("validateButton")
        validate_btn.clicked.connect(self.validate_config)
        
        layout.addWidget(new_btn)
        layout.addWidget(load_btn)
        layout.addStretch()
        layout.addWidget(validate_btn)
        layout.addStretch()
        layout.addWidget(save_btn)
        layout.addWidget(save_as_btn)
        
        return footer
        
    def create_menu_bar(self):
        """Create the menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        new_action = QAction("New Configuration", self)
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self.new_config)
        file_menu.addAction(new_action)
        
        load_action = QAction("Load Configuration", self)
        load_action.setShortcut("Ctrl+O")
        load_action.triggered.connect(self.load_config)
        file_menu.addAction(load_action)
        
        save_action = QAction("Save Configuration", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_config)
        file_menu.addAction(save_action)
        
        save_as_action = QAction("Save As...", self)
        save_as_action.setShortcut("Ctrl+Shift+S")
        save_as_action.triggered.connect(self.save_config_as)
        file_menu.addAction(save_as_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Presets menu
        presets_menu = menubar.addMenu("Presets")

        europa_quick = QAction("Europa Quick Test", self)
        europa_quick.triggered.connect(lambda: self.load_preset("europa_quick"))
        presets_menu.addAction(europa_quick)

        europa_hr = QAction("Europa High Resolution", self)
        europa_hr.triggered.connect(lambda: self.load_preset("europa_highres"))
        presets_menu.addAction(europa_hr)

        moon_thermal = QAction("Moon Thermal Analysis", self)
        moon_thermal.triggered.connect(lambda: self.load_preset("moon_thermal"))
        presets_menu.addAction(moon_thermal)

        presets_menu.addSeparator()

        # NEW presets for radiation exchange
        crater_radex = QAction("Crater w/ Radiation Exchange", self)
        crater_radex.triggered.connect(lambda: self.load_preset("crater_radex"))
        presets_menu.addAction(crater_radex)

        large_mesh = QAction("Large Mesh (H-Matrix)", self)
        large_mesh.triggered.connect(lambda: self.load_preset("large_mesh_hmatrix"))
        presets_menu.addAction(large_mesh)

        gpu_accel = QAction("GPU Accelerated (Full)", self)
        gpu_accel.triggered.connect(lambda: self.load_preset("gpu_accelerated"))
        presets_menu.addAction(gpu_accel)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
        docs_action = QAction("Documentation", self)
        docs_action.triggered.connect(self.show_docs)
        help_menu.addAction(docs_action)
        
    def apply_retro_theme(self):
        """Apply the retro-futuristic theme"""
        self.setStyleSheet(RetroStyle.get_stylesheet())
        
    def animate_status_light(self):
        """Animate the status indicator light"""
        self.light_state = 0
        self.light_timer = QTimer()
        self.light_timer.timeout.connect(self.pulse_status_light)
        self.light_timer.start(1000)  # Pulse every second
        
    def pulse_status_light(self):
        """Pulse animation for status light"""
        colors = ["#00ff00", "#00cc00", "#009900"]  # Green pulse
        self.light_state = (self.light_state + 1) % len(colors)
        self.status_light.setStyleSheet(f"color: {colors[self.light_state]};")
        
    def update_preview(self):
        """Update the YAML preview"""
        yaml_content = self.config_manager.generate_yaml()
        self.preview_text.setText(yaml_content)
        
    def on_value_changed(self):
        """Mark configuration as having unsaved changes"""
        self.unsaved_changes = True
        self.setWindowTitle("PTM3D Configuration Builder v1.0 *")
        
    def new_config(self):
        """Create a new configuration"""
        if self.unsaved_changes:
            reply = QMessageBox.question(
                self, "Unsaved Changes",
                "You have unsaved changes. Do you want to save before creating a new configuration?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel
            )
            if reply == QMessageBox.Save:
                self.save_config()
            elif reply == QMessageBox.Cancel:
                return
                
        self.config_manager.reset_to_defaults()
        self.refresh_all_tabs()
        self.current_file = None
        self.unsaved_changes = False
        self.setWindowTitle("PTM3D Configuration Builder v1.0")
        self.update_preview()
        self.status_bar.showMessage("New configuration created", 3000)
        
    def load_config(self):
        """Load a configuration file"""
        if self.unsaved_changes:
            reply = QMessageBox.question(
                self, "Unsaved Changes",
                "You have unsaved changes. Do you want to save before loading?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel
            )
            if reply == QMessageBox.Save:
                self.save_config()
            elif reply == QMessageBox.Cancel:
                return
                
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Configuration", "", "YAML Files (*.yaml *.yml);;All Files (*)"
        )
        
        if filename:
            try:
                self.config_manager.load_from_yaml(filename)
                self.refresh_all_tabs()
                self.current_file = filename
                self.unsaved_changes = False
                self.setWindowTitle(f"PTM3D Configuration Builder v1.0 - {Path(filename).name}")
                self.update_preview()
                self.status_bar.showMessage(f"Loaded: {filename}", 5000)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load configuration:\n{str(e)}")
                
    def save_config(self):
        """Save the current configuration"""
        if self.current_file:
            self.save_to_file(self.current_file)
        else:
            self.save_config_as()
            
    def save_config_as(self):
        """Save configuration with a new filename"""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Configuration", "", "YAML Files (*.yaml);;All Files (*)"
        )
        
        if filename:
            if not filename.endswith('.yaml'):
                filename += '.yaml'
            self.save_to_file(filename)
            
    def save_to_file(self, filename):
        """Save configuration to specified file"""
        try:
            self.config_manager.save_to_yaml(filename)
            self.current_file = filename
            self.unsaved_changes = False
            self.setWindowTitle(f"PTM3D Configuration Builder v1.0 - {Path(filename).name}")
            self.status_bar.showMessage(f"Saved: {filename}", 5000)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save configuration:\n{str(e)}")
            
    def validate_config(self):
        """Validate the current configuration"""
        is_valid, errors = self.config_manager.validate()
        
        if is_valid:
            QMessageBox.information(
                self, "Validation Success",
                "✓ Configuration is valid and ready to use!"
            )
            self.status_bar.showMessage("Validation passed", 3000)
        else:
            error_msg = "Configuration validation failed:\n\n" + "\n".join(f"• {e}" for e in errors)
            QMessageBox.warning(self, "Validation Errors", error_msg)
            self.status_bar.showMessage("Validation failed - see errors", 5000)
            
    def load_preset(self, preset_name):
        """Load a preset configuration"""
        try:
            self.config_manager.load_preset(preset_name)
            self.refresh_all_tabs()
            self.unsaved_changes = True
            self.update_preview()
            self.status_bar.showMessage(f"Loaded preset: {preset_name}", 3000)
        except Exception as e:
            QMessageBox.warning(self, "Preset Error", f"Failed to load preset:\n{str(e)}")
            
    def refresh_all_tabs(self):
        """Refresh all tab widgets with current config values"""
        self.sim_tab.load_from_config()
        self.mesh_tab.load_from_config()
        self.rad_tab.load_from_config()  # NEW
        self.thermal_tab.load_from_config()
        self.illum_tab.load_from_config()
        self.output_tab.load_from_config()
        self.vis_tab.load_from_config()
        self.adv_tab.load_from_config()
        
    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self, "About PTM3D Configuration Builder",
            "<h2>PTM3D Configuration Builder v1.0</h2>"
            "<p>A retro-futuristic GUI for creating PTM3D simulation configurations.</p>"
            "<p><b>Features:</b></p>"
            "<ul>"
            "<li>Visual parameter management</li>"
            "<li>Real-time YAML preview</li>"
            "<li>Configuration validation</li>"
            "<li>Preset templates</li>"
            "</ul>"
            "<p>Built with PyQt5 and styled for maximum nostalgia.</p>"
        )
        
    def show_docs(self):
        """Show documentation"""
        QMessageBox.information(
            self, "Documentation",
            "<h3>PTM3D Configuration Builder Help</h3>"
            "<p><b>Getting Started:</b></p>"
            "<ol>"
            "<li>Configure parameters in each tab</li>"
            "<li>Watch the YAML preview update in real-time</li>"
            "<li>Validate your configuration</li>"
            "<li>Save as a .yaml file</li>"
            "<li>Run: <code>ptm_thermal_sim --config your_file.yaml</code></li>"
            "</ol>"
            "<p><b>Tips:</b></p>"
            "<ul>"
            "<li>Use presets as starting points</li>"
            "<li>Hover over parameters for tooltips</li>"
            "<li>Validate before saving</li>"
            "</ul>"
        )
        
    def closeEvent(self, event):
        """Handle window close event"""
        if self.unsaved_changes:
            reply = QMessageBox.question(
                self, "Unsaved Changes",
                "You have unsaved changes. Do you want to save before exiting?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel
            )
            if reply == QMessageBox.Save:
                self.save_config()
                event.accept()
            elif reply == QMessageBox.Discard:
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName("PTM3D Configuration Builder")
    
    # Set application-wide font
    font = QFont("Courier New", 10)
    app.setFont(font)
    
    window = PTMConfigGUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()