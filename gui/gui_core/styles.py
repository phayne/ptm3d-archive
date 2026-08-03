"""
Modern Styling for PTM3D GUI
Clean, professional theme with dark mode support
"""


class ModernStyle:
    """Modern dark theme styling"""

    # Color palette
    PRIMARY = "#2196F3"  # Blue
    PRIMARY_DARK = "#1976D2"
    PRIMARY_LIGHT = "#64B5F6"

    ACCENT = "#00BCD4"  # Cyan
    ACCENT_DARK = "#0097A7"

    SUCCESS = "#4CAF50"  # Green
    WARNING = "#FF9800"  # Orange
    ERROR = "#F44336"  # Red

    BG_DARK = "#1e1e1e"
    BG_MEDIUM = "#252526"
    BG_LIGHT = "#2d2d2d"
    BG_PANEL = "#333333"

    TEXT_PRIMARY = "#e0e0e0"
    TEXT_SECONDARY = "#a0a0a0"
    TEXT_DISABLED = "#606060"

    BORDER_COLOR = "#404040"
    BORDER_FOCUS = "#2196F3"

    @staticmethod
    def get_stylesheet():
        """Return the complete application stylesheet"""
        return f"""
            /* === Global Styles === */
            QMainWindow {{
                background-color: {ModernStyle.BG_DARK};
                color: {ModernStyle.TEXT_PRIMARY};
            }}

            QWidget {{
                background-color: {ModernStyle.BG_DARK};
                color: {ModernStyle.TEXT_PRIMARY};
                font-family: "Helvetica Neue", ".AppleSystemUIFont";
                font-size: 13px;
            }}

            /* === Menu Bar === */
            QMenuBar {{
                background-color: {ModernStyle.BG_MEDIUM};
                color: {ModernStyle.TEXT_PRIMARY};
                border-bottom: 1px solid {ModernStyle.BORDER_COLOR};
                padding: 4px;
            }}

            QMenuBar::item {{
                padding: 6px 12px;
                border-radius: 4px;
            }}

            QMenuBar::item:selected {{
                background-color: {ModernStyle.PRIMARY};
            }}

            QMenu {{
                background-color: {ModernStyle.BG_PANEL};
                border: 1px solid {ModernStyle.BORDER_COLOR};
                border-radius: 4px;
                padding: 4px;
            }}

            QMenu::item {{
                padding: 8px 24px;
                border-radius: 4px;
            }}

            QMenu::item:selected {{
                background-color: {ModernStyle.PRIMARY};
            }}

            QMenu::separator {{
                height: 1px;
                background-color: {ModernStyle.BORDER_COLOR};
                margin: 4px 8px;
            }}

            /* === Toolbar === */
            QToolBar {{
                background-color: {ModernStyle.BG_MEDIUM};
                border-bottom: 1px solid {ModernStyle.BORDER_COLOR};
                spacing: 4px;
                padding: 4px 8px;
            }}

            QToolBar::separator {{
                width: 1px;
                background-color: {ModernStyle.BORDER_COLOR};
                margin: 4px 8px;
            }}

            QToolButton {{
                background-color: transparent;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
                color: {ModernStyle.TEXT_PRIMARY};
            }}

            QToolButton:hover {{
                background-color: {ModernStyle.BG_LIGHT};
            }}

            QToolButton:pressed {{
                background-color: {ModernStyle.PRIMARY_DARK};
            }}

            QToolButton:disabled {{
                color: {ModernStyle.TEXT_DISABLED};
            }}

            /* === Tab Widget === */
            QTabWidget::pane {{
                border: 1px solid {ModernStyle.BORDER_COLOR};
                border-radius: 4px;
                background-color: {ModernStyle.BG_MEDIUM};
            }}

            QTabBar::tab {{
                background-color: {ModernStyle.BG_PANEL};
                color: {ModernStyle.TEXT_SECONDARY};
                border: 1px solid {ModernStyle.BORDER_COLOR};
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                padding: 8px 16px;
                margin-right: 2px;
            }}

            QTabBar::tab:selected {{
                background-color: {ModernStyle.BG_MEDIUM};
                color: {ModernStyle.TEXT_PRIMARY};
                border-bottom: 2px solid {ModernStyle.PRIMARY};
            }}

            QTabBar::tab:hover:!selected {{
                background-color: {ModernStyle.BG_LIGHT};
            }}

            /* === Group Box === */
            QGroupBox {{
                border: none;
                border-top: 1px solid {ModernStyle.BORDER_COLOR};
                border-radius: 0px;
                margin-top: 24px;
                padding: 16px;
                padding-top: 20px;
                background-color: {ModernStyle.BG_MEDIUM};
                font-weight: bold;
            }}

            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 8px;
                color: {ModernStyle.PRIMARY_LIGHT};
                font-size: 14px;
            }}

            /* === Collapsible Section Header === */
            QPushButton#collapsibleHeader {{
                background-color: {ModernStyle.BG_LIGHT};
                border: 1px solid {ModernStyle.BORDER_COLOR};
                border-radius: 4px;
                color: {ModernStyle.PRIMARY_LIGHT};
                font-weight: bold;
                text-align: left;
                padding: 8px 12px;
                font-size: 13px;
            }}

            QPushButton#collapsibleHeader:hover {{
                background-color: {ModernStyle.BG_PANEL};
                border-color: {ModernStyle.PRIMARY};
            }}

            QPushButton#collapsibleHeader:checked {{
                color: {ModernStyle.TEXT_SECONDARY};
            }}

            /* === Navigation Rail === */
            QListWidget#NavRail {{
                background-color: {ModernStyle.BG_PANEL};
                border: none;
                border-right: 1px solid {ModernStyle.BORDER_COLOR};
                outline: none;
                font-size: 13px;
            }}

            QListWidget#NavRail::item {{
                padding: 12px 16px;
                border: none;
                color: {ModernStyle.TEXT_SECONDARY};
            }}

            QListWidget#NavRail::item:selected {{
                background-color: {ModernStyle.PRIMARY_DARK};
                color: white;
                font-weight: bold;
                border-left: 3px solid {ModernStyle.PRIMARY_LIGHT};
            }}

            QListWidget#NavRail::item:hover:!selected {{
                background-color: {ModernStyle.BG_LIGHT};
                color: {ModernStyle.TEXT_PRIMARY};
            }}

            /* === Labels === */
            QLabel {{
                background-color: transparent;
                color: {ModernStyle.TEXT_PRIMARY};
            }}

            QLabel#sectionHeader {{
                color: {ModernStyle.PRIMARY_LIGHT};
                font-size: 14px;
                font-weight: bold;
                padding: 8px 0;
            }}

            /* === Input Fields === */
            QLineEdit, QSpinBox, QDoubleSpinBox {{
                background-color: {ModernStyle.BG_PANEL};
                border: 1px solid {ModernStyle.BORDER_COLOR};
                border-radius: 4px;
                padding: 6px 10px;
                color: {ModernStyle.TEXT_PRIMARY};
                selection-background-color: {ModernStyle.PRIMARY};
            }}

            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
                border-color: {ModernStyle.BORDER_FOCUS};
            }}

            QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{
                background-color: {ModernStyle.BG_DARK};
                color: {ModernStyle.TEXT_DISABLED};
            }}

            /* === Combo Box === */
            QComboBox {{
                background-color: {ModernStyle.BG_PANEL};
                border: 1px solid {ModernStyle.BORDER_COLOR};
                border-radius: 4px;
                padding: 6px 10px;
                color: {ModernStyle.TEXT_PRIMARY};
                min-width: 80px;
            }}

            QComboBox:focus {{
                border-color: {ModernStyle.BORDER_FOCUS};
            }}

            QComboBox::drop-down {{
                border: none;
                width: 24px;
            }}

            QComboBox::down-arrow {{
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 6px solid {ModernStyle.TEXT_SECONDARY};
                margin-right: 8px;
            }}

            QComboBox QAbstractItemView {{
                background-color: {ModernStyle.BG_PANEL};
                border: 1px solid {ModernStyle.BORDER_COLOR};
                selection-background-color: {ModernStyle.PRIMARY};
                color: {ModernStyle.TEXT_PRIMARY};
            }}

            /* === Check Box === */
            QCheckBox {{
                spacing: 8px;
                color: {ModernStyle.TEXT_PRIMARY};
            }}

            QCheckBox::indicator {{
                width: 18px;
                height: 18px;
                border: 2px solid {ModernStyle.BORDER_COLOR};
                border-radius: 4px;
                background-color: {ModernStyle.BG_PANEL};
            }}

            QCheckBox::indicator:checked {{
                background-color: {ModernStyle.PRIMARY};
                border-color: {ModernStyle.PRIMARY};
            }}

            QCheckBox::indicator:hover {{
                border-color: {ModernStyle.PRIMARY_LIGHT};
            }}

            /* === Push Button === */
            QPushButton {{
                background-color: {ModernStyle.BG_PANEL};
                border: 1px solid {ModernStyle.BORDER_COLOR};
                border-radius: 4px;
                padding: 8px 16px;
                color: {ModernStyle.TEXT_PRIMARY};
                font-weight: 500;
                min-height: 28px;
            }}

            QPushButton:hover {{
                background-color: {ModernStyle.BG_LIGHT};
                border-color: {ModernStyle.PRIMARY};
            }}

            QPushButton:pressed {{
                background-color: {ModernStyle.PRIMARY_DARK};
            }}

            QPushButton:disabled {{
                background-color: {ModernStyle.BG_DARK};
                color: {ModernStyle.TEXT_DISABLED};
                border-color: {ModernStyle.BG_PANEL};
            }}

            QPushButton#primaryButton {{
                background-color: {ModernStyle.PRIMARY};
                border-color: {ModernStyle.PRIMARY};
                color: white;
            }}

            QPushButton#primaryButton:hover {{
                background-color: {ModernStyle.PRIMARY_DARK};
            }}

            QPushButton#dangerButton {{
                background-color: {ModernStyle.ERROR};
                border-color: {ModernStyle.ERROR};
                color: white;
            }}

            /* === Text Edit === */
            QTextEdit, QPlainTextEdit {{
                background-color: {ModernStyle.BG_PANEL};
                border: 1px solid {ModernStyle.BORDER_COLOR};
                border-radius: 4px;
                padding: 8px;
                color: {ModernStyle.TEXT_PRIMARY};
                selection-background-color: {ModernStyle.PRIMARY};
            }}

            /* === Scroll Bar === */
            QScrollBar:vertical {{
                border: none;
                background-color: {ModernStyle.BG_DARK};
                width: 12px;
                margin: 0;
            }}

            QScrollBar::handle:vertical {{
                background-color: {ModernStyle.BG_LIGHT};
                border-radius: 6px;
                min-height: 24px;
                margin: 2px;
            }}

            QScrollBar::handle:vertical:hover {{
                background-color: {ModernStyle.TEXT_SECONDARY};
            }}

            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                border: none;
                background: none;
                height: 0;
            }}

            QScrollBar:horizontal {{
                border: none;
                background-color: {ModernStyle.BG_DARK};
                height: 12px;
                margin: 0;
            }}

            QScrollBar::handle:horizontal {{
                background-color: {ModernStyle.BG_LIGHT};
                border-radius: 6px;
                min-width: 24px;
                margin: 2px;
            }}

            QScrollBar::handle:horizontal:hover {{
                background-color: {ModernStyle.TEXT_SECONDARY};
            }}

            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
                border: none;
                background: none;
                width: 0;
            }}

            /* === Scroll Area === */
            QScrollArea {{
                border: none;
                background-color: transparent;
            }}

            /* === Splitter === */
            QSplitter::handle {{
                background-color: {ModernStyle.BORDER_COLOR};
            }}

            QSplitter::handle:horizontal {{
                width: 6px;
            }}

            QSplitter::handle:vertical {{
                height: 6px;
            }}

            QSplitter::handle:hover {{
                background-color: {ModernStyle.PRIMARY};
            }}

            /* === Status Bar === */
            QStatusBar {{
                background-color: {ModernStyle.BG_MEDIUM};
                color: {ModernStyle.TEXT_SECONDARY};
                border-top: 1px solid {ModernStyle.BORDER_COLOR};
            }}

            QStatusBar::item {{
                border: none;
            }}

            /* === Progress Bar === */
            QProgressBar {{
                border: 1px solid {ModernStyle.BORDER_COLOR};
                border-radius: 4px;
                background-color: {ModernStyle.BG_PANEL};
                text-align: center;
                color: {ModernStyle.TEXT_PRIMARY};
            }}

            QProgressBar::chunk {{
                background-color: {ModernStyle.PRIMARY};
                border-radius: 3px;
            }}

            /* === Dock Widget === */
            QDockWidget {{
                color: {ModernStyle.TEXT_PRIMARY};
                titlebar-close-icon: none;
            }}

            QDockWidget::title {{
                background-color: {ModernStyle.BG_MEDIUM};
                padding: 8px;
                border-bottom: 1px solid {ModernStyle.BORDER_COLOR};
            }}

            /* === Tree Widget === */
            QTreeWidget {{
                background-color: {ModernStyle.BG_MEDIUM};
                border: 1px solid {ModernStyle.BORDER_COLOR};
                border-radius: 4px;
                alternate-background-color: {ModernStyle.BG_LIGHT};
            }}

            QTreeWidget::item {{
                padding: 4px;
            }}

            QTreeWidget::item:selected {{
                background-color: {ModernStyle.PRIMARY};
            }}

            QTreeWidget::item:hover {{
                background-color: {ModernStyle.BG_LIGHT};
            }}

            QTreeWidget::branch:has-children:!has-siblings:closed,
            QTreeWidget::branch:closed:has-children:has-siblings {{
                border-image: none;
                image: none;
            }}

            QTreeWidget::branch:open:has-children:!has-siblings,
            QTreeWidget::branch:open:has-children:has-siblings {{
                border-image: none;
                image: none;
            }}

            /* === Header View === */
            QHeaderView::section {{
                background-color: {ModernStyle.BG_PANEL};
                color: {ModernStyle.TEXT_PRIMARY};
                padding: 6px;
                border: none;
                border-right: 1px solid {ModernStyle.BORDER_COLOR};
                border-bottom: 1px solid {ModernStyle.BORDER_COLOR};
            }}

            /* === Slider === */
            QSlider::groove:horizontal {{
                border: none;
                height: 4px;
                background-color: {ModernStyle.BG_LIGHT};
                border-radius: 2px;
            }}

            QSlider::handle:horizontal {{
                background-color: {ModernStyle.PRIMARY};
                border: none;
                width: 16px;
                margin: -6px 0;
                border-radius: 8px;
            }}

            QSlider::handle:horizontal:hover {{
                background-color: {ModernStyle.PRIMARY_LIGHT};
            }}

            /* === Console Specific === */
            QTextEdit#consoleOutput {{
                background-color: #0d1117;
                color: {ModernStyle.SUCCESS};
                font-family: "Consolas", "Courier New", monospace;
                font-size: 12px;
                border: none;
                border-radius: 0;
            }}

            /* === Dashboard Sidebar === */
            QWidget#iconSidebar {{
                background-color: {ModernStyle.BG_MEDIUM};
                border-right: 1px solid {ModernStyle.BORDER_COLOR};
            }}

            QToolButton#sidebarButton {{
                background-color: transparent;
                color: {ModernStyle.TEXT_SECONDARY};
                border: none;
                border-left: 3px solid transparent;
                border-radius: 0;
                padding: 8px 4px;
                font-size: 11px;
                font-weight: bold;
            }}

            QToolButton#sidebarButton:hover {{
                background-color: {ModernStyle.BG_LIGHT};
                color: {ModernStyle.TEXT_PRIMARY};
            }}

            QToolButton#sidebarButton:checked {{
                background-color: {ModernStyle.BG_DARK};
                color: {ModernStyle.PRIMARY};
                border-left: 3px solid {ModernStyle.PRIMARY};
            }}
        """
