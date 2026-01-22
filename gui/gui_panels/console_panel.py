"""
Console Panel for PTM3D GUI
Styled terminal-like output panel for simulation logging
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton,
    QLabel, QCheckBox
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QTextCursor, QTextCharFormat, QColor


class ConsolePanel(QWidget):
    """Console output panel with styled logging"""

    clear_requested = pyqtSignal()

    # Color scheme
    COLORS = {
        'default': '#e0e0e0',
        'info': '#64b5f6',
        'success': '#81c784',
        'warning': '#ffb74d',
        'error': '#e57373',
        'header': '#00bcd4',
        'timestamp': '#888888'
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header bar
        header = QWidget()
        header.setStyleSheet("background-color: #252526; border-bottom: 1px solid #404040;")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(10, 5, 10, 5)

        # Title
        title = QLabel("Console Output")
        title.setFont(QFont("Helvetica Neue", 10, QFont.Weight.Bold))
        title.setStyleSheet("color: #00bcd4;")
        header_layout.addWidget(title)

        header_layout.addStretch()

        # Auto-scroll checkbox
        self.autoscroll_check = QCheckBox("Auto-scroll")
        self.autoscroll_check.setChecked(True)
        self.autoscroll_check.setStyleSheet("color: #e0e0e0;")
        header_layout.addWidget(self.autoscroll_check)

        # Clear button
        clear_btn = QPushButton("Clear")
        clear_btn.setMaximumWidth(60)
        clear_btn.clicked.connect(self.clear)
        header_layout.addWidget(clear_btn)

        layout.addWidget(header)

        # Console output area
        self.output = QTextEdit()
        self.output.setObjectName("consoleOutput")
        self.output.setReadOnly(True)
        self.output.setFont(QFont("Consolas", 11))
        self.output.setStyleSheet("""
            QTextEdit {
                background-color: #0d1117;
                color: #e0e0e0;
                border: none;
                padding: 8px;
            }
        """)

        layout.addWidget(self.output)

        # Initial message
        self.append_header("PTM3D Console Ready")

    def append(self, text, color=None):
        """Append text with optional color"""
        if color is None:
            color = self.COLORS['default']

        cursor = self.output.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)

        # Create format with color
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(color))

        # Insert text
        cursor.insertText(text + "\n", fmt)

        # Auto-scroll
        if self.autoscroll_check.isChecked():
            self.output.verticalScrollBar().setValue(
                self.output.verticalScrollBar().maximum()
            )

    def append_output(self, text):
        """Append standard output"""
        self.append(text, self.COLORS['default'])

    def append_info(self, text):
        """Append info message"""
        self.append(f"[INFO] {text}", self.COLORS['info'])

    def append_success(self, text):
        """Append success message"""
        self.append(f"[OK] {text}", self.COLORS['success'])

    def append_warning(self, text):
        """Append warning message"""
        self.append(f"[WARN] {text}", self.COLORS['warning'])

    def append_error(self, text):
        """Append error message"""
        self.append(f"[ERROR] {text}", self.COLORS['error'])

    def append_header(self, text):
        """Append header/section divider"""
        line = "=" * 60
        self.append(line, self.COLORS['header'])
        self.append(f"  {text}", self.COLORS['header'])
        self.append(line, self.COLORS['header'])

    def append_progress(self, text):
        """Append progress line (overwrites previous if possible)"""
        # For true progress bar behavior, we'd need to track and overwrite
        # For simplicity, just append with special formatting
        self.append(text, self.COLORS['info'])

    def clear(self):
        """Clear console output"""
        self.output.clear()
        self.append_header("Console Cleared")
        self.clear_requested.emit()

    def get_text(self):
        """Get all console text"""
        return self.output.toPlainText()

    def save_log(self, filename):
        """Save console output to file"""
        try:
            with open(filename, 'w') as f:
                f.write(self.get_text())
            return True
        except Exception as e:
            self.append_error(f"Failed to save log: {e}")
            return False
