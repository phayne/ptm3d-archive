"""
Vertical icon sidebar for module navigation.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QToolButton, QButtonGroup, QSizePolicy, QLabel
)
from PyQt6.QtCore import pyqtSignal, Qt, QSize
from PyQt6.QtGui import QFont


class IconSidebar(QWidget):
    """Vertical sidebar with icon-style buttons for switching modules.

    Each button shows a short label (e.g., "DEM", "THM").
    Emits module_selected(int) when a button is clicked.
    """

    module_selected = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("iconSidebar")
        self.setFixedWidth(72)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 8, 0, 8)
        self._layout.setSpacing(4)

        self._button_group = QButtonGroup(self)
        self._button_group.setExclusive(True)
        self._button_group.idClicked.connect(self._on_button_clicked)

        self._buttons = []

        # Spacer at bottom
        self._layout.addStretch()

    def add_module(self, index: int, label: str, tooltip: str = ""):
        """Add a module button to the sidebar.

        Args:
            index: Module index (maps to QStackedWidget page index).
            label: Short label (2-4 chars), displayed on the button.
            tooltip: Tooltip for the button.
        """
        btn = QToolButton()
        btn.setObjectName("sidebarButton")
        btn.setText(label)
        btn.setToolTip(tooltip)
        btn.setCheckable(True)
        btn.setFixedSize(64, 56)

        font = QFont()
        font.setPointSize(11)
        font.setBold(True)
        btn.setFont(font)

        btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)

        self._button_group.addButton(btn, index)

        # Insert before the stretch
        self._layout.insertWidget(self._layout.count() - 1, btn,
                                  alignment=Qt.AlignmentFlag.AlignHCenter)
        self._buttons.append(btn)

        # Select first button by default
        if index == 0:
            btn.setChecked(True)

    def select_module(self, index: int):
        """Programmatically select a module."""
        btn = self._button_group.button(index)
        if btn:
            btn.setChecked(True)
            self._on_button_clicked(index)

    def _on_button_clicked(self, index: int):
        self.module_selected.emit(index)
