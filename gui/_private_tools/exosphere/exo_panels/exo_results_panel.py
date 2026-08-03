"""
Results Browser Panel for Exosphere GUI.
Dual-directory browser for PTM3D input and exosphere output.
"""

import sys
from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QFileDialog, QTreeWidget, QTreeWidgetItem,
    QSplitter, QLineEdit
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QColor

# Import data loader
try:
    from exo_core.exo_data_loader import (
        discover_fields, discover_timestep_count, detect_mode, load_ply_mesh
    )
    from exo_core.exo_tooltips import FIELD_INFO, FIELD_CATEGORIES
except ImportError:
    parent_dir = str(Path(__file__).parent.parent)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from exo_core.exo_data_loader import (
        discover_fields, discover_timestep_count, detect_mode, load_ply_mesh
    )
    from exo_core.exo_tooltips import FIELD_INFO, FIELD_CATEGORIES


class DirectoryBrowser(QWidget):
    """Single-directory results browser with field tree and info."""

    directory_selected = pyqtSignal(str)
    field_selected = pyqtSignal(str)
    timestep_selected = pyqtSignal(int)

    def __init__(self, title="Results", parent=None):
        super().__init__(parent)
        self.title = title
        self.directory = ""
        self.fields = {}
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Directory selector
        dir_layout = QHBoxLayout()
        self.dir_input = QLineEdit()
        self.dir_input.setPlaceholderText(f"Select {self.title} directory...")
        self.dir_input.setReadOnly(True)
        browse_btn = QPushButton("Browse...")
        browse_btn.setMaximumWidth(100)
        browse_btn.clicked.connect(self._browse_directory)
        dir_layout.addWidget(self.dir_input)
        dir_layout.addWidget(browse_btn)
        layout.addLayout(dir_layout)

        # Info labels
        self.info_label = QLabel("No directory loaded")
        self.info_label.setStyleSheet("color: #888; font-size: 10px; padding: 2px;")
        layout.addWidget(self.info_label)

        # Field tree
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Field", "Timesteps"])
        self.tree.setColumnWidth(0, 200)
        self.tree.setAlternatingRowColors(True)
        self.tree.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self.tree)

    def _browse_directory(self):
        directory = QFileDialog.getExistingDirectory(
            self, f"Select {self.title} Directory",
            self.dir_input.text() or ".",
            QFileDialog.Option.ShowDirsOnly
        )
        if directory:
            self.load_directory(directory)

    def load_directory(self, directory):
        """Load and display fields from a directory."""
        self.directory = directory
        self.dir_input.setText(directory)

        self.fields = discover_fields(directory)
        mode = detect_mode(directory)
        n_timesteps = discover_timestep_count(directory)

        # Check for mesh
        mesh_path = Path(directory) / "mesh.ply"
        has_mesh = mesh_path.is_file()

        # Update info
        info_parts = [f"Mode: {mode}", f"Fields: {len(self.fields)}"]
        if n_timesteps > 0:
            info_parts.append(f"Timesteps: {n_timesteps}")
        if has_mesh:
            info_parts.append("Mesh: found")
        self.info_label.setText(" | ".join(info_parts))
        self.info_label.setStyleSheet("color: #64b5f6; font-size: 10px; padding: 2px;")

        # Populate tree grouped by category
        self.tree.clear()
        categories = {}
        for field_name, timesteps in self.fields.items():
            info = FIELD_INFO.get(field_name, {})
            cat = info.get("category", "unknown")
            if cat not in categories:
                categories[cat] = []
            categories[cat].append((field_name, timesteps))

        for cat_key in FIELD_CATEGORIES:
            if cat_key not in categories:
                continue
            cat_label = FIELD_CATEGORIES[cat_key]
            cat_item = QTreeWidgetItem([cat_label, ""])
            cat_item.setFont(0, QFont("Helvetica Neue", 10, QFont.Weight.Bold))
            cat_item.setForeground(0, QColor("#90caf9"))
            cat_item.setExpanded(True)
            self.tree.addTopLevelItem(cat_item)

            for field_name, timesteps in sorted(categories[cat_key]):
                info = FIELD_INFO.get(field_name, {})
                label = info.get("label", field_name)
                units = info.get("units", "")
                display = f"{label} ({units})" if units else label
                field_item = QTreeWidgetItem([display, str(len(timesteps))])
                field_item.setData(0, Qt.ItemDataRole.UserRole, field_name)
                cat_item.addChild(field_item)

        # Add "unknown" category for fields not in registry
        if "unknown" in categories:
            cat_item = QTreeWidgetItem(["Other", ""])
            cat_item.setFont(0, QFont("Helvetica Neue", 10, QFont.Weight.Bold))
            cat_item.setExpanded(True)
            self.tree.addTopLevelItem(cat_item)
            for field_name, timesteps in sorted(categories["unknown"]):
                display = field_name.replace("_", " ").title()
                field_item = QTreeWidgetItem([display, str(len(timesteps))])
                field_item.setData(0, Qt.ItemDataRole.UserRole, field_name)
                cat_item.addChild(field_item)

        self.directory_selected.emit(directory)

    def _on_item_clicked(self, item, column):
        field_name = item.data(0, Qt.ItemDataRole.UserRole)
        if field_name:
            self.field_selected.emit(field_name)

    def get_directory(self):
        return self.directory

    def get_fields(self):
        return self.fields


class ExoResultsPanel(QWidget):
    """Dual results browser for PTM3D input and exosphere output."""

    ptm_results_selected = pyqtSignal(str)
    exo_results_selected = pyqtSignal(str)
    field_selected = pyqtSignal(str, str)  # field_name, source ("ptm3d" or "exosphere")
    timestep_selected = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Vertical)

        # PTM3D input browser
        ptm_group = QGroupBox("PTM3D Input")
        ptm_layout = QVBoxLayout()
        self.ptm_browser = DirectoryBrowser("PTM3D Output")
        self.ptm_browser.directory_selected.connect(self.ptm_results_selected.emit)
        self.ptm_browser.field_selected.connect(
            lambda f: self.field_selected.emit(f, "ptm3d")
        )
        ptm_layout.addWidget(self.ptm_browser)
        ptm_group.setLayout(ptm_layout)
        splitter.addWidget(ptm_group)

        # Exosphere output browser
        exo_group = QGroupBox("Exosphere Output")
        exo_layout = QVBoxLayout()
        self.exo_browser = DirectoryBrowser("Exosphere Output")
        self.exo_browser.directory_selected.connect(self.exo_results_selected.emit)
        self.exo_browser.field_selected.connect(
            lambda f: self.field_selected.emit(f, "exosphere")
        )
        exo_layout.addWidget(self.exo_browser)
        exo_group.setLayout(exo_layout)
        splitter.addWidget(exo_group)

        splitter.setSizes([300, 300])
        layout.addWidget(splitter)

    def load_ptm_directory(self, directory):
        self.ptm_browser.load_directory(directory)

    def load_exo_directory(self, directory):
        self.exo_browser.load_directory(directory)

    def get_ptm_directory(self):
        return self.ptm_browser.get_directory()

    def get_exo_directory(self):
        return self.exo_browser.get_directory()
