"""
Results Browser Panel for PTM3D GUI
Browse and select simulation results for visualization
"""

import os
from pathlib import Path
from datetime import datetime
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTreeWidget, QTreeWidgetItem,
    QPushButton, QLabel, QGroupBox, QGridLayout, QTextEdit, QSplitter,
    QHeaderView, QAbstractItemView, QFileDialog, QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont


class ResultsPanel(QWidget):
    """Panel for browsing and selecting simulation results"""

    results_selected = pyqtSignal(str)  # Results directory path
    timestep_selected = pyqtSignal(int, float)  # Timestep index, time value

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_results_dir = None
        self.timesteps = []
        # Default to parent directory of GUI (ptm project root)
        self.default_browse_dir = Path(__file__).parent.parent.parent

        # Restore last-used results directory from settings
        from PyQt6.QtCore import QSettings
        self._settings = QSettings("PTM3D", "ResultsPanel")
        saved = self._settings.value("last_results_dir")
        if saved and Path(saved).is_dir():
            self.current_results_dir = saved

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Create splitter for results tree and details
        splitter = QSplitter(Qt.Orientation.Vertical)

        # Results tree
        tree_widget = QWidget()
        tree_layout = QVBoxLayout(tree_widget)
        tree_layout.setContentsMargins(0, 0, 0, 0)

        # Header with load button
        header_layout = QHBoxLayout()
        header_label = QLabel("Simulation Results")
        header_label.setFont(QFont("Helvetica Neue", 11, QFont.Weight.Bold))
        header_layout.addWidget(header_label)
        header_layout.addStretch()

        load_btn = QPushButton("Load Directory...")
        load_btn.setMaximumWidth(120)
        load_btn.clicked.connect(self.browse_results)
        header_layout.addWidget(load_btn)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.setMaximumWidth(80)
        refresh_btn.clicked.connect(self.refresh_results)
        header_layout.addWidget(refresh_btn)

        tree_layout.addLayout(header_layout)

        # Tree widget
        self.results_tree = QTreeWidget()
        self.results_tree.setHeaderLabels(["Name", "Type", "Size"])
        self.results_tree.setAlternatingRowColors(True)
        self.results_tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.results_tree.itemClicked.connect(self.on_item_clicked)
        self.results_tree.itemDoubleClicked.connect(self.on_item_double_clicked)

        header = self.results_tree.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)

        tree_layout.addWidget(self.results_tree)
        splitter.addWidget(tree_widget)

        # Details panel
        details_widget = QWidget()
        details_layout = QVBoxLayout(details_widget)
        details_layout.setContentsMargins(0, 0, 0, 0)

        details_label = QLabel("Details")
        details_label.setFont(QFont("Helvetica Neue", 11, QFont.Weight.Bold))
        details_layout.addWidget(details_label)

        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        self.details_text.setMaximumHeight(200)
        self.details_text.setFont(QFont("Consolas", 10))
        details_layout.addWidget(self.details_text)

        splitter.addWidget(details_widget)

        # Set splitter proportions
        splitter.setSizes([400, 150])

        layout.addWidget(splitter)

        # Timestep list (separate section)
        timestep_group = QGroupBox("Timesteps")
        timestep_layout = QVBoxLayout(timestep_group)

        self.timestep_tree = QTreeWidget()
        self.timestep_tree.setHeaderLabels(["#", "Time", "Label"])
        self.timestep_tree.setAlternatingRowColors(True)
        self.timestep_tree.setMaximumHeight(200)
        self.timestep_tree.itemClicked.connect(self.on_timestep_clicked)

        ts_header = self.timestep_tree.header()
        ts_header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        ts_header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        ts_header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)

        timestep_layout.addWidget(self.timestep_tree)

        layout.addWidget(timestep_group)

    def set_default_browse_dir(self, directory):
        """Set the default starting directory for the browse dialog."""
        if directory and Path(directory).is_dir():
            self.default_browse_dir = Path(directory)

    def browse_results(self):
        """Open dialog to browse for results directory"""
        # Use current results dir if set, otherwise default browse dir
        start_dir = self.current_results_dir or str(self.default_browse_dir)

        directory = QFileDialog.getExistingDirectory(
            self, "Select Results Directory",
            start_dir,
            QFileDialog.Option.ShowDirsOnly
        )

        if directory:
            self.load_results(directory)

    def load_results(self, results_dir):
        """Load simulation results from directory"""
        results_path = Path(results_dir)
        if not results_path.exists():
            QMessageBox.warning(self, "Error", f"Directory not found:\n{results_dir}")
            return

        self.current_results_dir = results_dir
        self._settings.setValue("last_results_dir", results_dir)
        self.results_tree.clear()
        self.timestep_tree.clear()
        self.timesteps = []

        # Check for required files
        mesh_file = results_path / "mesh.ply"
        time_file = results_path / "time_steps.txt"

        if not mesh_file.exists():
            QMessageBox.warning(
                self, "Invalid Results",
                "mesh.ply not found. This doesn't appear to be a valid PTM3D output directory."
            )
            return

        # Populate results tree
        self.populate_results_tree(results_path)

        # Load timesteps
        if time_file.exists():
            self.load_timesteps(time_file)

        # Update details
        self.update_details_summary(results_path)

        # Emit signal
        self.results_selected.emit(results_dir)

    def populate_results_tree(self, results_path):
        """Populate the results tree with directory contents"""
        root = QTreeWidgetItem(self.results_tree)
        root.setText(0, results_path.name)
        root.setText(1, "Results")
        root.setData(0, Qt.ItemDataRole.UserRole, str(results_path))

        # Data directories
        data_dirs = [
            ("temperature", "Temperature Data"),
            ("illumination", "Illumination Data"),
            ("direct_solar", "Direct Solar Data"),
            ("primary_reflected", "Primary Reflected"),
            ("primary_ir", "Primary IR")
        ]

        for dir_name, display_name in data_dirs:
            dir_path = results_path / dir_name
            if dir_path.exists():
                dir_item = QTreeWidgetItem(root)
                dir_item.setText(0, display_name)
                dir_item.setText(1, "Data")

                # Count files
                files = list(dir_path.glob("*.bin"))
                dir_item.setText(2, f"{len(files)} files")
                dir_item.setData(0, Qt.ItemDataRole.UserRole, str(dir_path))

        # Statistics directory (annual aggregates: max/avg/min per face)
        stats_dir = results_path / "statistics"
        if stats_dir.exists():
            stats_item = QTreeWidgetItem(root)
            stats_item.setText(0, "Statistics (annual)")
            stats_item.setText(1, "Aggregate")
            stats_files = sorted(stats_dir.glob("*.bin"))
            stats_item.setText(2, f"{len(stats_files)} files")
            stats_item.setData(0, Qt.ItemDataRole.UserRole, str(stats_dir))
            # Group children by data type for clarity
            tag_to_node = {}
            for f in stats_files:
                # filename: <datatype>_<stat>.bin where stat ∈ {max, avg, min}
                stem = f.stem
                if stem.endswith("_max") or stem.endswith("_avg") or stem.endswith("_min"):
                    tag = stem.rsplit("_", 1)[0]
                    stat = stem.rsplit("_", 1)[1]
                else:
                    tag = stem
                    stat = ""
                if tag not in tag_to_node:
                    parent = QTreeWidgetItem(stats_item)
                    parent.setText(0, tag)
                    parent.setText(1, "Datatype")
                    parent.setData(0, Qt.ItemDataRole.UserRole, str(stats_dir))
                    tag_to_node[tag] = parent
                child = QTreeWidgetItem(tag_to_node[tag])
                child.setText(0, f.name)
                child.setText(1, stat or "stat")
                child.setText(2, self.format_size(f.stat().st_size))
                child.setData(0, Qt.ItemDataRole.UserRole, str(f))

        # Mesh file
        mesh_file = results_path / "mesh.ply"
        if mesh_file.exists():
            mesh_item = QTreeWidgetItem(root)
            mesh_item.setText(0, "mesh.ply")
            mesh_item.setText(1, "Mesh")
            size = mesh_file.stat().st_size
            mesh_item.setText(2, self.format_size(size))
            mesh_item.setData(0, Qt.ItemDataRole.UserRole, str(mesh_file))

        # Time steps file
        time_file = results_path / "time_steps.txt"
        if time_file.exists():
            time_item = QTreeWidgetItem(root)
            time_item.setText(0, "time_steps.txt")
            time_item.setText(1, "Metadata")
            size = time_file.stat().st_size
            time_item.setText(2, self.format_size(size))
            time_item.setData(0, Qt.ItemDataRole.UserRole, str(time_file))

        # Body positions
        pos_file = results_path / "body_positions.csv"
        if pos_file.exists():
            pos_item = QTreeWidgetItem(root)
            pos_item.setText(0, "body_positions.csv")
            pos_item.setText(1, "Metadata")
            size = pos_file.stat().st_size
            pos_item.setText(2, self.format_size(size))
            pos_item.setData(0, Qt.ItemDataRole.UserRole, str(pos_file))

        # VTK output
        vtk_dir = results_path / "vtk"
        if vtk_dir.exists():
            vtk_item = QTreeWidgetItem(root)
            vtk_item.setText(0, "VTK Output")
            vtk_item.setText(1, "Converted")
            files = list(vtk_dir.glob("*.vtu")) + list(vtk_dir.glob("*.pvd"))
            vtk_item.setText(2, f"{len(files)} files")
            vtk_item.setData(0, Qt.ItemDataRole.UserRole, str(vtk_dir))

        root.setExpanded(True)

    def load_timesteps(self, time_file):
        """Load timesteps from file"""
        self.timesteps = []
        self.timestep_tree.clear()

        with open(time_file, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    time_val = float(parts[0])
                    label = parts[1]
                    self.timesteps.append({
                        'time': time_val,
                        'label': label
                    })

        # Populate tree
        for i, ts in enumerate(self.timesteps):
            item = QTreeWidgetItem(self.timestep_tree)
            item.setText(0, str(i))
            item.setText(1, f"{ts['time']:.4f}")
            item.setText(2, ts['label'])
            item.setData(0, Qt.ItemDataRole.UserRole, i)

    def update_details_summary(self, results_path):
        """Update details panel with results summary"""
        lines = []
        lines.append(f"Results Directory: {results_path}")
        lines.append("")

        # Mesh info
        mesh_file = results_path / "mesh.ply"
        if mesh_file.exists():
            size = mesh_file.stat().st_size
            lines.append(f"Mesh: {self.format_size(size)}")

        # Timesteps
        lines.append(f"Timesteps: {len(self.timesteps)}")

        # Data directories
        for dir_name in ["temperature", "illumination", "direct_solar"]:
            dir_path = results_path / dir_name
            if dir_path.exists():
                files = list(dir_path.glob("*.bin"))
                total_size = sum(f.stat().st_size for f in files)
                lines.append(f"{dir_name}: {len(files)} files ({self.format_size(total_size)})")

        # Statistics summary (if present)
        stats_dir = results_path / "statistics"
        if stats_dir.exists():
            stats_files = sorted(stats_dir.glob("*.bin"))
            if stats_files:
                total = sum(f.stat().st_size for f in stats_files)
                lines.append(
                    f"statistics: {len(stats_files)} files ({self.format_size(total)})"
                )

        # Modification time
        lines.append("")
        mtime = datetime.fromtimestamp(results_path.stat().st_mtime)
        lines.append(f"Last modified: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")

        self.details_text.setText("\n".join(lines))

    def refresh_results(self):
        """Refresh current results"""
        if self.current_results_dir:
            self.load_results(self.current_results_dir)

    def on_item_clicked(self, item, column):
        """Handle tree item click"""
        path = item.data(0, Qt.ItemDataRole.UserRole)
        if path:
            path = Path(path)
            if path.is_file():
                self.show_file_details(path)

    def on_item_double_clicked(self, item, column):
        """Handle tree item double-click"""
        path = item.data(0, Qt.ItemDataRole.UserRole)
        if path:
            path = Path(path)
            if path.is_dir():
                # For directories, could expand or show more info
                pass

    def on_timestep_clicked(self, item, column):
        """Handle timestep selection"""
        index = item.data(0, Qt.ItemDataRole.UserRole)
        if index is not None and index < len(self.timesteps):
            time_val = self.timesteps[index]['time']
            self.timestep_selected.emit(index, time_val)

    def show_file_details(self, file_path):
        """Show details for a specific file"""
        lines = []
        lines.append(f"File: {file_path.name}")
        lines.append(f"Path: {file_path}")
        lines.append(f"Size: {self.format_size(file_path.stat().st_size)}")
        mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
        lines.append(f"Modified: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")

        # If this is a per-face binary (PTM3D writeStatisticsBinary or
        # writeBinaryData), read it and summarise. Format: uint32 N + N×float32.
        if file_path.suffix == ".bin" and file_path.stat().st_size > 4:
            try:
                import struct
                import numpy as np
                with open(file_path, "rb") as f:
                    raw_n = f.read(4)
                    n = struct.unpack("<I", raw_n)[0]
                    expected = 4 + n * 4
                    actual = file_path.stat().st_size
                    if actual == expected:
                        arr = np.frombuffer(f.read(n * 4), dtype="<f4")
                        finite = arr[np.isfinite(arr)]
                        lines.append("")
                        lines.append(f"Faces: {n:,}")
                        if finite.size:
                            lines.append(
                                f"Range: {finite.min():.4g} … {finite.max():.4g}"
                            )
                            lines.append(
                                f"Mean : {finite.mean():.4g}   "
                                f"Median: {np.median(finite):.4g}"
                            )
                            n_nan = arr.size - finite.size
                            if n_nan:
                                lines.append(f"NaN  : {n_nan:,} ({100*n_nan/arr.size:.2f}%)")
                        else:
                            lines.append("All values NaN/inf")
            except Exception as exc:
                lines.append(f"(could not parse as PTM3D binary: {exc})")

        self.details_text.setText("\n".join(lines))

    def show_face_details(self, face_id, data):
        """Show details for a selected face"""
        lines = []
        lines.append(f"Selected Face: {face_id}")
        lines.append("")

        if data:
            if 'latitude' in data and data['latitude'] is not None:
                lines.append(f"Latitude: {data['latitude']:.4f}°")
            if 'longitude' in data and data['longitude'] is not None:
                lines.append(f"Longitude: {data['longitude']:.4f}°")

            # Add any other data fields
            for key, value in data.items():
                if key not in ['face_id', 'latitude', 'longitude'] and value is not None:
                    if isinstance(value, float):
                        lines.append(f"{key}: {value:.4f}")
                    else:
                        lines.append(f"{key}: {value}")

        self.details_text.setText("\n".join(lines))

    def get_timesteps(self):
        """Return list of timesteps"""
        return self.timesteps

    def select_timestep(self, index):
        """Select a timestep in the tree"""
        if 0 <= index < self.timestep_tree.topLevelItemCount():
            item = self.timestep_tree.topLevelItem(index)
            self.timestep_tree.setCurrentItem(item)

    @staticmethod
    def format_size(size):
        """Format file size in human-readable form"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"
