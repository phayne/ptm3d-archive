"""
Results Browser Panel for Viewshed GUI.
File browser for viewshed output .bin files.
"""

import struct
import sys
from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QFileDialog, QTreeWidget, QTreeWidgetItem,
    QLineEdit,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QColor


# Field categories for organising the tree view
FIELD_CATEGORIES = {
    "truth": "Truth Maps",
    "per_waypoint": "Per-Waypoint",
    "aggregation": "Aggregation",
    "mask": "Custom Masks",
    "other": "Other",
}


def _categorise_field(name):
    """Assign a category to a field name."""
    if name.startswith("truth_"):
        return "truth"
    if name.startswith("mask_"):
        return "mask"
    if name in ("visibility_count", "illumination_hours", "temperature_mean_traverse",
                "earth_visibility_fraction",
                "estimated_ice_fraction_best", "snr_best"):
        return "aggregation"
    if name.startswith("earth_visible_"):
        return "per_waypoint"
    # Per-waypoint fields typically end with a waypoint label
    per_wp_prefixes = (
        "visibility_", "distance_km_", "emission_angle_", "elevation_angle_",
        "temperature_", "estimated_ice_fraction_", "spatial_resolution_", "snr_",
    )
    for pfx in per_wp_prefixes:
        if name.startswith(pfx) and name != pfx.rstrip("_"):
            return "per_waypoint"
    return "other"


class ViewshedResultsPanel(QWidget):
    """Results browser for viewshed output directory."""

    field_selected = pyqtSignal(str, str)  # field_name, file_path
    directory_loaded = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.directory = ""
        self.fields = {}  # name -> path
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Directory selector
        dir_group = QGroupBox("Output Directory")
        dir_layout = QVBoxLayout()

        dir_row = QHBoxLayout()
        self.dir_input = QLineEdit()
        self.dir_input.setPlaceholderText("Select output directory...")
        self.dir_input.setReadOnly(True)
        browse_btn = QPushButton("Browse...")
        browse_btn.setMaximumWidth(100)
        browse_btn.clicked.connect(self._browse_directory)
        dir_row.addWidget(self.dir_input)
        dir_row.addWidget(browse_btn)
        dir_layout.addLayout(dir_row)

        self.info_label = QLabel("No directory loaded")
        self.info_label.setStyleSheet("color: #888; font-size: 10px; padding: 2px;")
        dir_layout.addWidget(self.info_label)

        dir_group.setLayout(dir_layout)
        layout.addWidget(dir_group)

        # Field tree
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Field", "Size"])
        self.tree.setColumnWidth(0, 250)
        self.tree.setAlternatingRowColors(True)
        self.tree.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self.tree)

        # Refresh button
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh)
        layout.addWidget(refresh_btn)

    def _browse_directory(self):
        directory = QFileDialog.getExistingDirectory(
            self, "Select Output Directory",
            self.dir_input.text() or ".",
            QFileDialog.Option.ShowDirsOnly,
        )
        if directory:
            self.load_directory(directory)

    def load_directory(self, directory):
        """Load and display .bin fields from a directory."""
        self.directory = directory
        self.dir_input.setText(directory)

        output_dir = Path(directory)
        bin_files = sorted(output_dir.glob("*.bin"))

        self.fields = {}
        for p in bin_files:
            self.fields[p.stem] = str(p)

        # Count faces from first file
        n_faces_str = ""
        if bin_files:
            try:
                with open(bin_files[0], "rb") as f:
                    (n,) = struct.unpack("<I", f.read(4))
                n_faces_str = f" | Faces: {n:,}"
            except Exception:
                pass

        self.info_label.setText(
            f"Fields: {len(self.fields)}{n_faces_str}"
        )
        self.info_label.setStyleSheet("color: #64b5f6; font-size: 10px; padding: 2px;")

        # Populate tree by category
        self.tree.clear()
        categories = {}
        for name in self.fields:
            cat = _categorise_field(name)
            categories.setdefault(cat, []).append(name)

        for cat_key in FIELD_CATEGORIES:
            if cat_key not in categories:
                continue
            cat_label = FIELD_CATEGORIES[cat_key]
            cat_item = QTreeWidgetItem([cat_label, ""])
            cat_item.setFont(0, QFont("Helvetica Neue", 10, QFont.Weight.Bold))
            cat_item.setForeground(0, QColor("#90caf9"))
            cat_item.setExpanded(True)
            self.tree.addTopLevelItem(cat_item)

            for name in sorted(categories[cat_key]):
                path = self.fields[name]
                size = Path(path).stat().st_size
                size_str = f"{size / 1024:.0f} KB" if size >= 1024 else f"{size} B"
                display = name.replace("_", " ").title()
                field_item = QTreeWidgetItem([display, size_str])
                field_item.setData(0, Qt.ItemDataRole.UserRole, name)
                field_item.setData(1, Qt.ItemDataRole.UserRole, path)
                cat_item.addChild(field_item)

        self.directory_loaded.emit(directory)

    def _refresh(self):
        if self.directory:
            self.load_directory(self.directory)

    def _on_item_clicked(self, item, column):
        name = item.data(0, Qt.ItemDataRole.UserRole)
        path = item.data(1, Qt.ItemDataRole.UserRole)
        if name and path:
            self.field_selected.emit(name, path)

    def get_directory(self):
        return self.directory
