"""
Retro-Futuristic Styling for PTM3D Configuration Builder
With dramatic big red RUN button styling
"""

class RetroStyle:
    """Retro-futuristic color scheme and styling"""
    
    # Color palette
    ACCENT_CYAN = "#00ffff"
    ACCENT_GREEN = "#00ff00"
    ACCENT_ORANGE = "#ff8800"
    ACCENT_RED = "#ff0000"
    ACCENT_YELLOW = "#ffff00"
    
    BG_DARK = "#0a0e27"
    BG_PANEL = "#141b3d"
    BG_INPUT = "#1a2347"
    
    TEXT_PRIMARY = "#e0e0e0"
    TEXT_SECONDARY = "#a0a0a0"
    
    BORDER_COLOR = "#00ffff"
    BORDER_DIM = "#004466"
    
    @staticmethod
    def get_stylesheet():
        """Return the complete application stylesheet"""
        return f"""
            /* Global Styles */
            QMainWindow {{
                background-color: {RetroStyle.BG_DARK};
                color: {RetroStyle.TEXT_PRIMARY};
                font-size: 15px;
            }}
            
            QWidget {{
                background-color: {RetroStyle.BG_DARK};
                color: {RetroStyle.TEXT_PRIMARY};
                font-family: "Courier New", monospace;
                font-size: 15px;
            }}
            /* Header title/subtitle overrides */
            QLabel#mainTitle {{
                color: {RetroStyle.ACCENT_GREEN};
                font-size: 36px;
                font-weight: 900;
            }}
            QLabel#subtitle {{
                color: {RetroStyle.ACCENT_CYAN};
                font-size: 20px;
                letter-spacing: 2px;
            }}
            /* Control Panel */
            QWidget#controlPanel {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {RetroStyle.BG_PANEL}, stop:1 {RetroStyle.BG_DARK});
                border-bottom: 3px solid {RetroStyle.ACCENT_RED};
                border-top: 2px solid {RetroStyle.BORDER_COLOR};
            }}
            
            /* BIG RED ARCADE BUTTON */
            QPushButton#bigRunButton {{
                background: qradialgradient(cx:0.5, cy:0.3, radius:0.8,
                    fx:0.5, fy:0.3,
                    stop:0 #ff4444, stop:0.5 #ff0000, stop:1 #aa0000);
                border: 6px solid #660000;
                border-radius: 40px;
                color: #ffffff;
                font-size: 24px;
                font-weight: bold;
                padding: 10px;
                /* Arcade button dome effect */
                qproperty-flat: false;
            }}
            
            QPushButton#bigRunButton:hover {{
                background: qradialgradient(cx:0.5, cy:0.3, radius:0.8,
                    fx:0.5, fy:0.3,
                    stop:0 #ff6666, stop:0.5 #ff2222, stop:1 #cc0000);
                border: 6px solid #880000;
            }}
            
            QPushButton#bigRunButton:pressed {{
                background: qradialgradient(cx:0.5, cy:0.6, radius:0.8,
                    fx:0.5, fy:0.6,
                    stop:0 #cc0000, stop:0.5 #aa0000, stop:1 #880000);
                border: 6px solid #440000;
                padding-top: 12px;
                padding-bottom: 8px;
            }}
            
            QPushButton#bigRunButton:disabled {{
                background: qradialgradient(cx:0.5, cy:0.3, radius:0.8,
                    fx:0.5, fy:0.3,
                    stop:0 #666666, stop:0.5 #444444, stop:1 #222222);
                border: 6px solid #333333;
                color: #777777;
            }}
            
            /* Stop Button */
            QPushButton#stopButton {{
                background-color: {RetroStyle.BG_PANEL};
                border: 2px solid {RetroStyle.ACCENT_RED};
                border-radius: 5px;
                color: {RetroStyle.ACCENT_RED};
                font-weight: bold;
                font-size: 12px;
                padding: 8px;
            }}
            
            QPushButton#stopButton:hover {{
                background-color: {RetroStyle.ACCENT_RED};
                color: #ffffff;
            }}
            
            QPushButton#stopButton:disabled {{
                background-color: {RetroStyle.BG_DARK};
                border-color: {RetroStyle.BORDER_DIM};
                color: {RetroStyle.TEXT_SECONDARY};
            }}
            
            /* Convert Button */
            QPushButton#convertButton {{
                background-color: {RetroStyle.BG_PANEL};
                border: 2px solid {RetroStyle.ACCENT_CYAN};
                border-radius: 5px;
                color: {RetroStyle.ACCENT_CYAN};
                font-weight: bold;
                font-size: 12px;
                padding: 8px;
            }}
            
            QPushButton#convertButton:hover {{
                background-color: {RetroStyle.ACCENT_CYAN};
                color: {RetroStyle.BG_DARK};
            }}
            
            /* Console Dock */
            QWidget#consoleDock {{
                background-color: {RetroStyle.BG_DARK};
                border-top: 2px solid {RetroStyle.BORDER_COLOR};
            }}
            
            QWidget#consoleHeader {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {RetroStyle.BG_PANEL}, stop:1 {RetroStyle.BG_DARK});
                border-bottom: 1px solid {RetroStyle.BORDER_DIM};
            }}
            
            QLabel#consoleTitle {{
                color: {RetroStyle.ACCENT_CYAN};
            }}
            
            QTextEdit#consoleOutput {{
                background-color: #000000;
                border: 2px solid {RetroStyle.BORDER_COLOR};
                border-radius: 5px;
                color: {RetroStyle.ACCENT_GREEN};
                padding: 10px;
                font-family: "Courier New", monospace;
            }}
            
            QPushButton#consoleClearButton {{
                background-color: {RetroStyle.BG_PANEL};
                border: 1px solid {RetroStyle.BORDER_DIM};
                border-radius: 3px;
                color: {RetroStyle.TEXT_PRIMARY};
                padding: 5px 10px;
            }}
            
            QPushButton#consoleClearButton:hover {{
                border-color: {RetroStyle.ACCENT_ORANGE};
                color: {RetroStyle.ACCENT_ORANGE};
            }}
            
            QPushButton#consoleCollapseButton {{
                background-color: {RetroStyle.BG_PANEL};
                border: 1px solid {RetroStyle.BORDER_DIM};
                border-radius: 3px;
                color: {RetroStyle.ACCENT_CYAN};
                padding: 5px;
                font-weight: bold;
            }}
            
            QPushButton#consoleCollapseButton:hover {{
                background-color: {RetroStyle.ACCENT_CYAN};
                color: {RetroStyle.BG_DARK};
            }}
            
            /* Group Boxes */
            QGroupBox {{
                border: 2px solid {RetroStyle.BORDER_COLOR};
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: {RetroStyle.BG_PANEL};
                font-weight: bold;
                color: {RetroStyle.ACCENT_CYAN};
            }}
            
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
                color: {RetroStyle.ACCENT_CYAN};
            }}
            
            /* Labels */
            QLabel {{
                color: {RetroStyle.TEXT_PRIMARY};
                background-color: transparent;
            }}
            
            #sectionHeader {{
                color: {RetroStyle.ACCENT_CYAN};
                font-size: 14px;
                font-weight: bold;
                padding: 5px 0;
            }}
            
            /* Input Fields */
            QLineEdit, QSpinBox, QDoubleSpinBox {{
                background-color: {RetroStyle.BG_INPUT};
                border: 1px solid {RetroStyle.BORDER_DIM};
                border-radius: 3px;
                padding: 5px;
                color: {RetroStyle.ACCENT_GREEN};
                selection-background-color: {RetroStyle.ACCENT_CYAN};
            }}
            
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
                border: 1px solid {RetroStyle.ACCENT_CYAN};
            }}
            
            /* Combo Boxes */
            QComboBox {{
                background-color: {RetroStyle.BG_INPUT};
                border: 1px solid {RetroStyle.BORDER_DIM};
                border-radius: 3px;
                padding: 5px;
                color: {RetroStyle.ACCENT_GREEN};
            }}
            
            QComboBox:focus {{
                border: 1px solid {RetroStyle.ACCENT_CYAN};
            }}
            
            QComboBox::drop-down {{
                border: none;
                width: 20px;
            }}
            
            QComboBox::down-arrow {{
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid {RetroStyle.ACCENT_CYAN};
                margin-right: 5px;
            }}
            
            QComboBox QAbstractItemView {{
                background-color: {RetroStyle.BG_INPUT};
                border: 1px solid {RetroStyle.ACCENT_CYAN};
                selection-background-color: {RetroStyle.ACCENT_CYAN};
                selection-color: {RetroStyle.BG_DARK};
                color: {RetroStyle.ACCENT_GREEN};
            }}
            
            /* Check Boxes */
            QCheckBox {{
                spacing: 5px;
                color: {RetroStyle.TEXT_PRIMARY};
            }}
            
            QCheckBox::indicator {{
                width: 18px;
                height: 18px;
                border: 1px solid {RetroStyle.BORDER_DIM};
                border-radius: 3px;
                background-color: {RetroStyle.BG_INPUT};
            }}
            
            QCheckBox::indicator:checked {{
                background-color: {RetroStyle.ACCENT_CYAN};
                border: 1px solid {RetroStyle.ACCENT_CYAN};
            }}
            
            /* Push Buttons */
            QPushButton {{
                background-color: {RetroStyle.BG_PANEL};
                border: 2px solid {RetroStyle.BORDER_COLOR};
                border-radius: 5px;
                padding: 8px 15px;
                color: {RetroStyle.ACCENT_CYAN};
                font-weight: bold;
                min-height: 30px;
            }}
            
            QPushButton:hover {{
                background-color: {RetroStyle.BG_INPUT};
                border-color: {RetroStyle.ACCENT_GREEN};
                color: {RetroStyle.ACCENT_GREEN};
            }}
            
            QPushButton:pressed {{
                background-color: {RetroStyle.ACCENT_CYAN};
                color: {RetroStyle.BG_DARK};
            }}
            
            QPushButton:disabled {{
                background-color: {RetroStyle.BG_DARK};
                border-color: {RetroStyle.BORDER_DIM};
                color: {RetroStyle.TEXT_SECONDARY};
            }}
            
            /* Action Buttons */
            QPushButton#actionButton {{
                background-color: {RetroStyle.BG_PANEL};
                border: 2px solid {RetroStyle.ACCENT_CYAN};
                color: {RetroStyle.ACCENT_CYAN};
                font-size: 12px;
                min-width: 120px;
            }}
            
            QPushButton#actionButton:hover {{
                border-color: {RetroStyle.ACCENT_GREEN};
                color: {RetroStyle.ACCENT_GREEN};
            }}
            
            /* Validate Button */
            QPushButton#validateButton {{
                background-color: {RetroStyle.BG_PANEL};
                border: 2px solid {RetroStyle.ACCENT_ORANGE};
                color: {RetroStyle.ACCENT_ORANGE};
                font-size: 12px;
                min-width: 120px;
            }}
            
            QPushButton#validateButton:hover {{
                border-color: {RetroStyle.ACCENT_YELLOW};
                color: {RetroStyle.ACCENT_YELLOW};
            }}
            
            /* Refresh Button */
            QPushButton#refreshButton {{
                background-color: {RetroStyle.BG_PANEL};
                border: 2px solid {RetroStyle.ACCENT_CYAN};
                color: {RetroStyle.ACCENT_CYAN};
            }}
            
            /* Text Edit / Preview */
            QTextEdit {{
                background-color: {RetroStyle.BG_INPUT};
                border: 1px solid {RetroStyle.BORDER_DIM};
                border-radius: 3px;
                color: {RetroStyle.TEXT_PRIMARY};
                padding: 5px;
            }}
            
            QTextEdit#previewText {{
                font-family: "Courier New", monospace;
                font-size: 10px;
            }}
            
            /* Scroll Areas */
            QScrollArea {{
                border: none;
                background-color: transparent;
            }}
            
            QScrollBar:vertical {{
                border: none;
                background-color: {RetroStyle.BG_PANEL};
                width: 12px;
                margin: 0;
            }}
            
            QScrollBar::handle:vertical {{
                background-color: {RetroStyle.ACCENT_CYAN};
                border-radius: 6px;
                min-height: 20px;
            }}
            
            QScrollBar::handle:vertical:hover {{
                background-color: {RetroStyle.ACCENT_GREEN};
            }}
            
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                border: none;
                background: none;
            }}
            
            /* Tabs */
            QTabWidget::pane {{
                border: 2px solid {RetroStyle.BORDER_COLOR};
                border-radius: 5px;
                background-color: {RetroStyle.BG_PANEL};
            }}
            
            QTabBar::tab {{
                background-color: {RetroStyle.BG_PANEL};
                border: 2px solid {RetroStyle.BORDER_DIM};
                border-bottom: none;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
                padding: 8px 12px;
                margin-right: 2px;
                color: {RetroStyle.TEXT_SECONDARY};
            }}
            
            QTabBar::tab:selected {{
                background-color: {RetroStyle.BG_INPUT};
                border-color: {RetroStyle.BORDER_COLOR};
                color: {RetroStyle.ACCENT_CYAN};
            }}
            
            QTabBar::tab:hover {{
                color: {RetroStyle.ACCENT_GREEN};
            }}
            
            /* Menu Bar */
            QMenuBar {{
                background-color: {RetroStyle.BG_PANEL};
                color: {RetroStyle.TEXT_PRIMARY};
                border-bottom: 1px solid {RetroStyle.BORDER_COLOR};
            }}
            
            QMenuBar::item:selected {{
                background-color: {RetroStyle.BG_INPUT};
                color: {RetroStyle.ACCENT_CYAN};
            }}
            
            QMenu {{
                background-color: {RetroStyle.BG_PANEL};
                border: 1px solid {RetroStyle.BORDER_COLOR};
                color: {RetroStyle.TEXT_PRIMARY};
            }}
            
            QMenu::item:selected {{
                background-color: {RetroStyle.ACCENT_CYAN};
                color: {RetroStyle.BG_DARK};
            }}
            
            /* Status Bar */
            QStatusBar {{
                background-color: {RetroStyle.BG_PANEL};
                color: {RetroStyle.ACCENT_GREEN};
                border-top: 1px solid {RetroStyle.BORDER_COLOR};
            }}
            
            /* Header and Footer */
            QWidget#header {{
                background-color: {RetroStyle.BG_PANEL};
                border-bottom: 2px solid {RetroStyle.BORDER_COLOR};
            }}
            
            QWidget#footer {{
                background-color: {RetroStyle.BG_PANEL};
                border-top: 2px solid {RetroStyle.BORDER_COLOR};
            }}
            
            /* Preview Panel */
            QWidget#previewPanel {{
                background-color: {RetroStyle.BG_PANEL};
                border-left: 2px solid {RetroStyle.BORDER_COLOR};
            }}
            
            QLabel#previewHeader {{
                color: {RetroStyle.ACCENT_CYAN};
                font-size: 12px;
                font-weight: bold;
                padding: 5px;
            }}
            
            /* Splitter */
            QSplitter::handle {{
                background-color: {RetroStyle.BORDER_COLOR};
                width: 2px;
            }}

            /* Collapsible Section Header */
            QPushButton#collapsibleHeader {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {RetroStyle.BG_INPUT}, stop:0.5 {RetroStyle.BG_PANEL}, stop:1 {RetroStyle.BG_INPUT});
                border: 1px solid {RetroStyle.BORDER_DIM};
                border-radius: 3px;
                color: {RetroStyle.ACCENT_CYAN};
                font-weight: bold;
                text-align: left;
                padding: 8px 15px;
                min-height: 20px;
            }}

            QPushButton#collapsibleHeader:hover {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {RetroStyle.BG_INPUT}, stop:0.5 #1e2854, stop:1 {RetroStyle.BG_INPUT});
                border-color: {RetroStyle.ACCENT_GREEN};
                color: {RetroStyle.ACCENT_GREEN};
            }}

            QPushButton#collapsibleHeader:checked {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {RetroStyle.BG_DARK}, stop:0.5 {RetroStyle.BG_PANEL}, stop:1 {RetroStyle.BG_DARK});
                border-color: {RetroStyle.BORDER_DIM};
            }}

            /* Hardware Status Indicator */
            QWidget#hardwareStatus {{
                background-color: {RetroStyle.BG_PANEL};
                border: 1px solid {RetroStyle.BORDER_DIM};
                border-radius: 5px;
                padding: 8px;
            }}

            QLabel#hardwareLabel {{
                color: {RetroStyle.ACCENT_CYAN};
                font-weight: bold;
            }}

            QLabel#hardwareAvailable {{
                color: {RetroStyle.ACCENT_GREEN};
            }}

            QLabel#hardwareUnavailable {{
                color: {RetroStyle.ACCENT_ORANGE};
            }}

            QLabel#hardwareError {{
                color: {RetroStyle.ACCENT_RED};
            }}

            /* Radiation Tab specific - warning box */
            QLabel#radiationWarning {{
                background-color: rgba(255, 136, 0, 0.1);
                border: 1px solid {RetroStyle.ACCENT_ORANGE};
                border-radius: 3px;
                color: {RetroStyle.ACCENT_ORANGE};
                padding: 8px;
            }}

            /* Info boxes */
            QLabel#infoBox {{
                background-color: rgba(0, 255, 255, 0.05);
                border: 1px solid {RetroStyle.BORDER_DIM};
                border-radius: 3px;
                color: {RetroStyle.TEXT_SECONDARY};
                padding: 8px;
                font-size: 11px;
            }}
        """