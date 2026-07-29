"""
Console Dock Widget for PTM3D Configuration Builder
Dockable/collapsible console output panel
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTextEdit,
                             QPushButton, QLabel, QCheckBox, QScrollBar)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QTextCursor, QTextOption


class ConsoleDock(QWidget):
    """Dockable console output widget"""
    
    toggled = pyqtSignal(bool)  # Signal when collapsed/expanded
    
    def __init__(self):
        super().__init__()
        self.setObjectName("consoleDock")
        self.is_collapsed = False
        self.last_line_was_progress = False
        self.last_progress_text = ""  # Track last progress line
        self.init_ui()
        
    def init_ui(self):
        """Initialize the console dock UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Header bar
        header = QWidget()
        header.setObjectName("consoleHeader")
        header.setFixedHeight(35)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(15, 5, 15, 5)
        
        # Title
        title = QLabel("▼ CONSOLE OUTPUT")
        title.setObjectName("consoleTitle")
        title.setFont(QFont("Courier New", 14, QFont.Bold))
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        # Auto-scroll checkbox
        self.autoscroll_check = QCheckBox("Auto-scroll")
        self.autoscroll_check.setChecked(True)
        self.autoscroll_check.setStyleSheet("color: #e0e0e0; font-size: 14px;")
        header_layout.addWidget(self.autoscroll_check)
        
        # Clear button
        clear_btn = QPushButton("Clear")
        clear_btn.setObjectName("consoleClearButton")
        clear_btn.setMaximumWidth(70)
        clear_btn.clicked.connect(self.clear_console)
        header_layout.addWidget(clear_btn)
        
        # Collapse/expand button
        self.collapse_btn = QPushButton("▼")
        self.collapse_btn.setObjectName("consoleCollapseButton")
        self.collapse_btn.setMaximumWidth(30)
        self.collapse_btn.clicked.connect(self.toggle_collapse)
        header_layout.addWidget(self.collapse_btn)
        
        layout.addWidget(header)
        
        # Console output area
        self.console_output = QTextEdit()
        self.console_output.setObjectName("consoleOutput")
        self.console_output.setReadOnly(True)
        self.console_output.setFont(QFont("Courier New", 13))
        self.console_output.setMinimumHeight(300)  # Larger console area
        # Disable wrapping so ASCII art and progress bars don’t collapse
        self.console_output.setWordWrapMode(QTextOption.NoWrap)
        self.console_output.setViewportMargins(0, 0, 0, 60)  # left, top, right, bottom
        layout.addWidget(self.console_output)
        # Ensure dock itself keeps a larger minimum height
        self.setMinimumHeight(340)
        
        # Initial message
        self.console_output.setHtml(
            '<span style="color: #00ffff;">═══════════════════════════════════════════════════════════════════</span><br>'
            '<span style="color: #00ff00;">PTM3D SIMULATION CONSOLE</span><br>'
            '<span style="color: #00ffff;">═══════════════════════════════════════════════════════════════════</span><br>'
            '<span style="color: #a0a0a0;">Ready. Press RUN to begin simulation...</span><br>'
        )
    
    def clear_console(self):
        """Clear the console output"""
        self.console_output.clear()
        self.last_line_was_progress = False
        self.last_progress_text = ""
        self.console_output.setHtml(
            '<span style="color: #00ffff;">═══════════════════════════════════════════════════════════════════</span><br>'
            '<span style="color: #00ff00;">CONSOLE CLEARED</span><br>'
            '<span style="color: #00ffff;">═══════════════════════════════════════════════════════════════════</span><br>'
        )
    
    def toggle_collapse(self):
        """Toggle console collapsed/expanded state"""
        self.is_collapsed = not self.is_collapsed
        
        if self.is_collapsed:
            self.console_output.hide()
            self.collapse_btn.setText("▲")
            self.setMaximumHeight(35)
        else:
            self.console_output.show()
            self.collapse_btn.setText("▼")
            self.setMaximumHeight(16777215)  # Remove max height
        
        self.toggled.emit(not self.is_collapsed)
    
    def html_escape(self, s: str) -> str:
        """Escape HTML special characters and preserve spaces"""
        # First escape HTML entities
        s = (s.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;"))
        # Convert spaces to non-breaking spaces to preserve formatting
        # Replace leading/trailing and multiple consecutive spaces
        s = s.replace(" ", "&nbsp;")
        return s

    def append_line(self, text: str, error: bool = False):
        """Append a single line, preserving spaces; overwrite last line for progress bars."""
        is_progress = self._is_progress_line(text)
        color = "#ff0000" if error and not is_progress else "#00ff00"
        html = f'<span style="color:{color}; font-family: Courier New, monospace;">{self.html_escape(text)}</span>'

        cursor = self.console_output.textCursor()
        cursor.movePosition(QTextCursor.End)

        if is_progress and self.last_line_was_progress:
            # Replace the current line (progress refresh)
            # Use line-based selection to avoid affecting previous content
            cursor.movePosition(QTextCursor.StartOfLine, QTextCursor.MoveAnchor)
            cursor.movePosition(QTextCursor.End, QTextCursor.KeepAnchor)
            cursor.removeSelectedText()
            cursor.insertHtml(html)
        else:
            # For non-progress lines or first progress line, add a new line
            if cursor.position() > 0:
                cursor.insertHtml("<br>")
            cursor.insertHtml(html)

        self.last_line_was_progress = bool(is_progress)
        
        # Move cursor to end after insertion
        cursor.movePosition(QTextCursor.End)
        self.console_output.setTextCursor(cursor)

        if self.autoscroll_check.isChecked():
            self.console_output.verticalScrollBar().setValue(
                self.console_output.verticalScrollBar().maximum()
            )

    def append(self, text):
        """Back-compat shim – treat as non-error line."""
        self.append_line(text, error=False)

    def _is_progress_line(self, text):
        """Detect if a line is a progress bar update"""
        tl = text.lower()
        has_bar = ('[' in text and ']' in text and '%' in text)
        keywords = ('calculating', 'thermal simulation', 'progress')
        return has_bar or (any(k in tl for k in keywords) and '%' in text)