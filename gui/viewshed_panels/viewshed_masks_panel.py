"""
Custom Masks Panel for Viewshed GUI.
Compound boolean mask builder with AND/OR conditions and multi-waypoint aggregation.
"""

import sys
from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QGridLayout,
    QLabel, QLineEdit, QDoubleSpinBox, QComboBox, QPushButton,
    QListWidget, QListWidgetItem, QScrollArea, QCheckBox, QSplitter,
    QMessageBox,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QColor


# Available fields for mask conditions
STATIC_FIELDS = [
    "T_max", "T_min", "T_mean",
    "latitude", "longitude", "elevation", "face_area",
    "ice_fraction_truth",
]

PER_WAYPOINT_FIELDS = [
    "visibility", "distance_km", "emission_angle",
    "elevation_angle", "temperature",
]

AGGREGATION_FIELDS = [
    "visibility_count",
    "earth_visibility_fraction",
]

OPERATORS = ["<", "<=", "==", ">=", ">", "!="]
CONNECTORS = ["AND", "OR"]


class ConditionRow(QWidget):
    """A single condition row: [Field] [Waypoint] [Op] [Value] [Connector] [Remove]."""

    removed = pyqtSignal(object)
    changed = pyqtSignal()

    def __init__(self, waypoint_labels=None, parent=None):
        super().__init__(parent)
        self.waypoint_labels = waypoint_labels or []
        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)

        # Field selector
        self.field_combo = QComboBox()
        self.field_combo.setMinimumWidth(140)
        all_fields = STATIC_FIELDS + PER_WAYPOINT_FIELDS + AGGREGATION_FIELDS
        self.field_combo.addItems(all_fields)
        self.field_combo.currentTextChanged.connect(self._on_field_changed)
        self.field_combo.currentTextChanged.connect(lambda: self.changed.emit())
        layout.addWidget(self.field_combo)

        # Waypoint selector (visible only for per-waypoint fields)
        self.waypoint_combo = QComboBox()
        self.waypoint_combo.setMinimumWidth(80)
        self.waypoint_combo.addItems(self.waypoint_labels)
        self.waypoint_combo.currentTextChanged.connect(lambda: self.changed.emit())
        layout.addWidget(self.waypoint_combo)
        self.waypoint_combo.setVisible(False)

        # Operator
        self.op_combo = QComboBox()
        self.op_combo.setMaximumWidth(60)
        self.op_combo.addItems(OPERATORS)
        self.op_combo.currentTextChanged.connect(lambda: self.changed.emit())
        layout.addWidget(self.op_combo)

        # Value
        self.value_spin = QDoubleSpinBox()
        self.value_spin.setRange(-1e12, 1e12)
        self.value_spin.setDecimals(4)
        self.value_spin.setMaximumWidth(120)
        self.value_spin.valueChanged.connect(lambda: self.changed.emit())
        layout.addWidget(self.value_spin)

        # Connector (AND/OR) — hidden for last row
        self.connector_combo = QComboBox()
        self.connector_combo.setMaximumWidth(70)
        self.connector_combo.addItems(CONNECTORS)
        self.connector_combo.currentTextChanged.connect(lambda: self.changed.emit())
        layout.addWidget(self.connector_combo)

        # Remove button
        remove_btn = QPushButton("x")
        remove_btn.setFixedSize(24, 24)
        remove_btn.setStyleSheet(
            "QPushButton { color: #f44336; border: 1px solid #666; "
            "border-radius: 3px; font-weight: bold; }"
            "QPushButton:hover { background: #f44336; color: white; }"
        )
        remove_btn.clicked.connect(lambda: self.removed.emit(self))
        layout.addWidget(remove_btn)

    def _on_field_changed(self, field):
        is_per_wp = field in PER_WAYPOINT_FIELDS
        self.waypoint_combo.setVisible(is_per_wp)

        # Set sensible default values based on field
        defaults = {
            "T_max": 110.0, "T_min": 50.0, "T_mean": 80.0,
            "visibility": 1.0, "visibility_count": 1.0,
            "distance_km": 10.0, "emission_angle": 85.0,
            "elevation_angle": 5.0, "temperature": 110.0,
            "latitude": -85.0, "longitude": 0.0,
            "ice_fraction_truth": 0.0,
        }
        if field in defaults:
            self.value_spin.setValue(defaults[field])

    def get_condition(self):
        """Return this condition as a dict."""
        cond = {
            "field": self.field_combo.currentText(),
            "op": self.op_combo.currentText(),
            "value": self.value_spin.value(),
            "connector": self.connector_combo.currentText(),
        }
        if self.waypoint_combo.isVisible() and self.waypoint_combo.currentText():
            cond["waypoint"] = self.waypoint_combo.currentText()
        return cond

    def set_condition(self, cond):
        """Populate from a condition dict."""
        idx = self.field_combo.findText(cond.get("field", ""))
        if idx >= 0:
            self.field_combo.setCurrentIndex(idx)
        idx = self.op_combo.findText(cond.get("op", "<"))
        if idx >= 0:
            self.op_combo.setCurrentIndex(idx)
        self.value_spin.setValue(float(cond.get("value", 0)))
        idx = self.connector_combo.findText(cond.get("connector", "AND"))
        if idx >= 0:
            self.connector_combo.setCurrentIndex(idx)
        wp = cond.get("waypoint", "")
        if wp:
            idx = self.waypoint_combo.findText(wp)
            if idx >= 0:
                self.waypoint_combo.setCurrentIndex(idx)

    def set_waypoint_labels(self, labels):
        """Update available waypoint labels."""
        self.waypoint_labels = labels
        current = self.waypoint_combo.currentText()
        self.waypoint_combo.clear()
        self.waypoint_combo.addItems(labels)
        idx = self.waypoint_combo.findText(current)
        if idx >= 0:
            self.waypoint_combo.setCurrentIndex(idx)


class ViewshedMasksPanel(QWidget):
    """Custom output mask builder with compound boolean conditions."""

    masks_changed = pyqtSignal()

    def __init__(self, config_manager, parent=None):
        super().__init__(parent)
        self.config_manager = config_manager
        self.condition_rows = []
        self.waypoint_labels = []
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Orientation.Vertical)

        # --- Section 1: Mask builder ---
        builder_widget = QWidget()
        builder_layout = QVBoxLayout(builder_widget)
        builder_layout.setContentsMargins(0, 0, 0, 0)

        builder_group = QGroupBox("Mask Builder")
        builder_inner = QVBoxLayout()

        # Mask name
        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Mask Name:"))
        self.mask_name_edit = QLineEdit()
        self.mask_name_edit.setPlaceholderText("e.g., cold_visible_faces")
        name_row.addWidget(self.mask_name_edit)
        builder_inner.addLayout(name_row)

        # Conditions container (scrollable)
        self.conditions_container = QWidget()
        self.conditions_layout = QVBoxLayout(self.conditions_container)
        self.conditions_layout.setContentsMargins(0, 0, 0, 0)
        self.conditions_layout.setSpacing(2)

        scroll = QScrollArea()
        scroll.setWidget(self.conditions_container)
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(120)
        scroll.setMaximumHeight(300)
        builder_inner.addWidget(scroll)

        # Add condition button
        add_btn = QPushButton("+ Add Condition")
        add_btn.setStyleSheet(
            "QPushButton { color: #4CAF50; border: 1px dashed #4CAF50; "
            "padding: 4px 12px; border-radius: 3px; }"
            "QPushButton:hover { background: #4CAF50; color: white; }"
        )
        add_btn.clicked.connect(self._add_condition_row)
        builder_inner.addWidget(add_btn)

        # Add to list button
        add_mask_btn = QPushButton("Add Mask to List")
        add_mask_btn.setStyleSheet(
            "QPushButton { background: #2196F3; color: white; padding: 6px; "
            "border-radius: 3px; font-weight: bold; }"
            "QPushButton:hover { background: #1976D2; }"
        )
        add_mask_btn.clicked.connect(self._add_mask_to_list)
        builder_inner.addWidget(add_mask_btn)

        builder_group.setLayout(builder_inner)
        builder_layout.addWidget(builder_group)

        splitter.addWidget(builder_widget)

        # --- Section 2: Aggregation checkboxes ---
        agg_widget = QWidget()
        agg_layout = QVBoxLayout(agg_widget)
        agg_layout.setContentsMargins(0, 0, 0, 0)

        agg_group = QGroupBox("Aggregation Fields (auto-computed)")
        agg_inner = QVBoxLayout()

        self.agg_vis_count = QCheckBox("Visibility count (faces visible from N waypoints)")
        self.agg_vis_count.setChecked(True)
        self.agg_vis_count.setEnabled(False)  # always computed
        agg_inner.addWidget(self.agg_vis_count)

        self.agg_illum = QCheckBox("Illumination hours (direct solar during traverse)")
        self.agg_illum.setChecked(True)
        self.agg_illum.setEnabled(False)  # always computed
        agg_inner.addWidget(self.agg_illum)

        self.agg_mean_temp = QCheckBox("Mean temperature during traverse window")
        self.agg_mean_temp.setChecked(True)
        self.agg_mean_temp.setEnabled(False)  # always computed
        agg_inner.addWidget(self.agg_mean_temp)

        self.agg_earth_vis = QCheckBox(
            "Earth visibility fraction (primary body above horizon)"
        )
        self.agg_earth_vis.setChecked(True)
        self.agg_earth_vis.setEnabled(False)  # always computed
        agg_inner.addWidget(self.agg_earth_vis)

        info = QLabel(
            "These fields are automatically computed for all faces across "
            "all waypoints and timesteps in the traverse window."
        )
        info.setStyleSheet("color: #888; font-size: 10px; padding: 4px;")
        info.setWordWrap(True)
        agg_inner.addWidget(info)

        agg_group.setLayout(agg_inner)
        agg_layout.addWidget(agg_group)

        splitter.addWidget(agg_widget)

        # --- Section 3: Mask list ---
        list_widget = QWidget()
        list_layout = QVBoxLayout(list_widget)
        list_layout.setContentsMargins(0, 0, 0, 0)

        list_group = QGroupBox("Defined Masks")
        list_inner = QVBoxLayout()

        self.mask_list = QListWidget()
        self.mask_list.setAlternatingRowColors(True)
        self.mask_list.itemDoubleClicked.connect(self._edit_mask)
        list_inner.addWidget(self.mask_list)

        # Buttons row
        btn_row = QHBoxLayout()
        edit_btn = QPushButton("Edit")
        edit_btn.clicked.connect(self._edit_selected_mask)
        btn_row.addWidget(edit_btn)

        remove_btn = QPushButton("Remove")
        remove_btn.setStyleSheet(
            "QPushButton { color: #f44336; }"
            "QPushButton:hover { background: #f44336; color: white; }"
        )
        remove_btn.clicked.connect(self._remove_selected_mask)
        btn_row.addWidget(remove_btn)

        btn_row.addStretch()
        list_inner.addLayout(btn_row)

        list_group.setLayout(list_inner)
        list_layout.addWidget(list_group)

        splitter.addWidget(list_widget)

        splitter.setSizes([300, 120, 200])
        layout.addWidget(splitter)

        # Start with one empty condition row
        self._add_condition_row()

        # Load any existing masks from config
        self._load_masks_from_config()

    def set_waypoint_labels(self, labels):
        """Update available waypoint labels for per-waypoint conditions."""
        self.waypoint_labels = labels
        for row in self.condition_rows:
            row.set_waypoint_labels(labels)

    def _add_condition_row(self):
        """Add a new condition row to the builder."""
        row = ConditionRow(waypoint_labels=self.waypoint_labels)
        row.removed.connect(self._remove_condition_row)
        row.changed.connect(self._on_conditions_changed)
        self.condition_rows.append(row)
        self.conditions_layout.addWidget(row)
        self._update_connector_visibility()

    def _remove_condition_row(self, row):
        """Remove a condition row from the builder."""
        if row in self.condition_rows:
            self.condition_rows.remove(row)
            self.conditions_layout.removeWidget(row)
            row.deleteLater()
            self._update_connector_visibility()

    def _update_connector_visibility(self):
        """Hide connector on the last condition row."""
        for i, row in enumerate(self.condition_rows):
            is_last = (i == len(self.condition_rows) - 1)
            row.connector_combo.setVisible(not is_last)

    def _on_conditions_changed(self):
        """Called when any condition row changes."""
        pass  # Could preview mask in visualization

    def _add_mask_to_list(self):
        """Add the current mask definition to the list and config."""
        name = self.mask_name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Missing Name", "Enter a mask name.")
            return
        if not self.condition_rows:
            QMessageBox.warning(self, "No Conditions", "Add at least one condition.")
            return

        # Sanitise name for filename
        safe_name = name.replace(" ", "_").replace("/", "_")

        conditions = [row.get_condition() for row in self.condition_rows]

        mask_def = {
            "name": safe_name,
            "conditions": conditions,
        }

        # Add to config
        masks = self.config_manager.get_value("masks", [])
        # Replace if same name exists
        masks = [m for m in masks if m.get("name") != safe_name]
        masks.append(mask_def)
        self.config_manager.set_value("masks", masks)

        # Add to list widget
        self._refresh_mask_list()

        # Clear builder
        self.mask_name_edit.clear()
        for row in list(self.condition_rows):
            self._remove_condition_row(row)
        self._add_condition_row()

        self.masks_changed.emit()

    def _refresh_mask_list(self):
        """Refresh the mask list widget from config."""
        self.mask_list.clear()
        masks = self.config_manager.get_value("masks", [])
        for mask_def in masks:
            name = mask_def.get("name", "unnamed")
            n_conds = len(mask_def.get("conditions", []))
            summary = self._summarise_mask(mask_def)
            item = QListWidgetItem(f"{name}  ({n_conds} condition{'s' if n_conds != 1 else ''})")
            item.setToolTip(summary)
            item.setData(Qt.ItemDataRole.UserRole, name)
            self.mask_list.addItem(item)

    def _summarise_mask(self, mask_def):
        """Build a human-readable summary of a mask definition."""
        parts = []
        for cond in mask_def.get("conditions", []):
            field = cond.get("field", "?")
            wp = cond.get("waypoint", "")
            if wp:
                field = f"{field}[{wp}]"
            op = cond.get("op", "?")
            val = cond.get("value", "?")
            connector = cond.get("connector", "")
            parts.append(f"{field} {op} {val}")
            if connector:
                parts.append(connector)
        return " ".join(parts)

    def _edit_mask(self, item):
        """Load a mask definition into the builder for editing."""
        name = item.data(Qt.ItemDataRole.UserRole)
        masks = self.config_manager.get_value("masks", [])
        mask_def = next((m for m in masks if m.get("name") == name), None)
        if mask_def is None:
            return

        # Clear current conditions
        for row in list(self.condition_rows):
            self._remove_condition_row(row)

        # Load mask
        self.mask_name_edit.setText(mask_def["name"])
        for cond in mask_def.get("conditions", []):
            self._add_condition_row()
            self.condition_rows[-1].set_condition(cond)
        self._update_connector_visibility()

    def _edit_selected_mask(self):
        """Edit the currently selected mask in the list."""
        item = self.mask_list.currentItem()
        if item:
            self._edit_mask(item)

    def _remove_selected_mask(self):
        """Remove the currently selected mask from the list and config."""
        item = self.mask_list.currentItem()
        if item is None:
            return
        name = item.data(Qt.ItemDataRole.UserRole)
        masks = self.config_manager.get_value("masks", [])
        masks = [m for m in masks if m.get("name") != name]
        self.config_manager.set_value("masks", masks)
        self._refresh_mask_list()
        self.masks_changed.emit()

    def _load_masks_from_config(self):
        """Load existing masks from config on panel creation."""
        self._refresh_mask_list()
