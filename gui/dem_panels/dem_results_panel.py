"""
DEM Results Panel: browse converted/refined mesh files.
"""

from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QTreeWidget, QTreeWidgetItem,
    QPushButton, QHBoxLayout, QFileDialog
)
from PyQt6.QtCore import pyqtSignal


class DEMResultsPanel(QWidget):
    """Browse DEM conversion and refinement results."""

    mesh_selected = pyqtSignal(str)  # path to selected mesh

    def __init__(self, parent=None):
        super().__init__(parent)
        self._results = []
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        header = QLabel("Converted Meshes")
        header.setStyleSheet("color: #00bcd4; font-weight: bold; font-size: 12px;")
        layout.addWidget(header)

        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["File", "Size", "Faces"])
        self._tree.setColumnCount(3)
        self._tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        layout.addWidget(self._tree, 1)

        # Buttons
        btn_layout = QHBoxLayout()
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh)
        btn_layout.addWidget(refresh_btn)

        load_btn = QPushButton("Load Selected")
        load_btn.clicked.connect(self._load_selected)
        btn_layout.addWidget(load_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

    def add_result(self, filepath):
        """Add a converted mesh to the results list."""
        path = Path(filepath)
        if not path.exists():
            return

        self._results.append(str(path))
        self._update_tree()

    def refresh(self):
        """Refresh the results tree."""
        self._update_tree()

    def _update_tree(self):
        self._tree.clear()
        for filepath in self._results:
            path = Path(filepath)
            if not path.exists():
                continue

            size = path.stat().st_size
            if size > 1e6:
                size_str = f"{size / 1e6:.1f} MB"
            else:
                size_str = f"{size / 1e3:.0f} KB"

            # Try to read face count from PLY header
            face_count = "?"
            try:
                with open(path, 'rb') as f:
                    header = b""
                    while True:
                        line = f.readline()
                        if b"end_header" in line:
                            break
                        header += line
                        if b"element face" in line:
                            face_count = line.decode().split()[-1]
            except Exception:
                pass

            item = QTreeWidgetItem([path.name, size_str, face_count])
            item.setData(0, 256, str(path))  # Store full path in UserRole
            self._tree.addTopLevelItem(item)

    def _on_item_double_clicked(self, item, column):
        filepath = item.data(0, 256)
        if filepath:
            self.mesh_selected.emit(filepath)

    def _load_selected(self):
        item = self._tree.currentItem()
        if item:
            filepath = item.data(0, 256)
            if filepath:
                self.mesh_selected.emit(filepath)
