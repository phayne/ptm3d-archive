"""
HORIZONS Search and Body Configuration Panel for PTM3D GUI.

Allows searching the JPL HORIZONS database for solar system bodies,
fetching orbital elements and physical parameters, and optionally
downloading DAMIT shape models for asteroids.
"""

import logging
import sys
from pathlib import Path
from functools import partial
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QGridLayout,
    QLabel, QLineEdit, QPushButton, QTableWidget, QTableWidgetItem,
    QHeaderView, QProgressBar, QCheckBox, QMessageBox, QTextEdit,
    QComboBox, QDoubleSpinBox, QSpinBox, QTabWidget, QScrollArea,
    QApplication
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread

logger = logging.getLogger(__name__)

# Add src/python to path for horizons/damit imports
_project_root = Path(__file__).resolve().parent.parent.parent
_src_python = _project_root / "src" / "python"
if str(_src_python) not in sys.path:
    sys.path.insert(0, str(_src_python))

# Lazy imports - these may fail if dependencies are missing
_horizons_available = False
_damit_available = False
_data_manager_available = False

try:
    from horizons.horizons_client import HorizonsClient, HorizonsBody
    _horizons_available = True
except ImportError:
    logger.warning("HORIZONS client not available")

try:
    from damit.damit_client import DAMITClient
    _damit_available = True
except ImportError:
    logger.warning("DAMIT client not available")

try:
    from data_manager import DataManager
    _data_manager_available = True
except ImportError:
    logger.warning("Data manager not available")


class FetchWorker(QThread):
    """Background thread for HORIZONS/DAMIT API calls."""

    result_ready = pyqtSignal(object)
    error_occurred = pyqtSignal(str)
    progress_update = pyqtSignal(str)

    def __init__(self, task, **kwargs):
        super().__init__()
        self.task = task
        self.kwargs = kwargs

    def run(self):
        try:
            if self.task == "search":
                client = HorizonsClient(timeout=15)
                results = client.search_body(self.kwargs["query"])
                self.result_ready.emit(("search", results))

            elif self.task == "fetch_body":
                self.progress_update.emit("Fetching orbital elements...")
                client = HorizonsClient(timeout=30)
                body = client.get_body(self.kwargs["query"])
                self.result_ready.emit(("fetch_body", body))

            elif self.task == "fetch_and_cache":
                from data_manager import fetch_and_cache_body

                self.progress_update.emit("Fetching and caching body data...")

                cache_dir = str(_project_root / "data" / "horizons_cache")
                spk_start = self.kwargs.get("spk_start", "2020-01-01")
                spk_stop = self.kwargs.get("spk_stop", "2030-01-01")
                fetch_damit = self.kwargs.get("fetch_damit", False) and _damit_available
                download_spk = bool(spk_start and spk_stop) and self.kwargs.get("download_spk", False)
                equilibration_orbits = self.kwargs.get("equilibration_orbits", 10)

                config_path = fetch_and_cache_body(
                    query=self.kwargs["query"],
                    cache_dir=cache_dir,
                    spk_start=spk_start,
                    spk_stop=spk_stop,
                    fetch_damit=fetch_damit,
                    ephemeris_source="spice" if download_spk else "keplerian",
                    equilibration_orbits=equilibration_orbits,
                )

                if not config_path:
                    self.error_occurred.emit(
                        f"Failed to fetch/cache body '{self.kwargs['query']}'"
                    )
                    return

                # Also fetch the body info for display
                self.progress_update.emit("Loading body details...")
                client = HorizonsClient(timeout=30)
                body = client.get_body(self.kwargs["query"])

                self.result_ready.emit(("cached", {
                    "config_path": config_path,
                    "body": body,
                }))

        except Exception as e:
            self.error_occurred.emit(str(e))


class HorizonsSearchPanel(QWidget):
    """Panel for searching and selecting HORIZONS bodies."""

    body_selected = pyqtSignal(str, str)  # (body_name, config_path)
    status_message = pyqtSignal(str)

    def __init__(self, config_manager=None):
        super().__init__()
        self.config_manager = config_manager
        self._worker = None
        self._current_body = None
        self._data_manager = None

        if _data_manager_available:
            cache_dir = str(_project_root / "data" / "horizons_cache")
            self._data_manager = DataManager(cache_dir)

        self.init_ui()
        self._refresh_cache_list()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # --- Search Section ---
        search_group = QGroupBox("Search HORIZONS")
        search_layout = QVBoxLayout()

        # Search bar
        search_bar = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText(
            "Body name, number, or designation (e.g. 'Vesta', '4', 'Bennu')"
        )
        self.search_input.returnPressed.connect(self._on_search)
        search_bar.addWidget(self.search_input)

        self.search_btn = QPushButton("Search")
        self.search_btn.clicked.connect(self._on_search)
        self.search_btn.setEnabled(_horizons_available)
        search_bar.addWidget(self.search_btn)
        search_layout.addLayout(search_bar)

        if not _horizons_available:
            warn = QLabel("HORIZONS client not available. Check src/python/ imports.")
            warn.setStyleSheet("color: #ff6b6b; font-size: 10px;")
            search_layout.addWidget(warn)

        # Results table
        self.results_table = QTableWidget(0, 3)
        self.results_table.setHorizontalHeaderLabels(["Name", "NAIF ID", "Type"])
        self.results_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self.results_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self.results_table.setSelectionMode(
            QTableWidget.SelectionMode.SingleSelection
        )
        self.results_table.setMaximumHeight(150)
        self.results_table.itemSelectionChanged.connect(self._on_result_selected)
        search_layout.addWidget(self.results_table)

        search_group.setLayout(search_layout)
        layout.addWidget(search_group)

        # --- Body Details Section ---
        details_group = QGroupBox("Body Details")
        details_layout = QGridLayout()
        details_layout.setColumnStretch(1, 1)
        details_layout.setColumnStretch(3, 1)

        row = 0
        self.detail_labels = {}
        detail_fields = [
            ("Name:", "name", "NAIF ID:", "naif_id"),
            ("Semi-major axis (AU):", "sma", "Eccentricity:", "ecc"),
            ("Inclination (deg):", "inc", "Period (days):", "period"),
            ("Diameter (km):", "diameter", "Albedo:", "albedo"),
            ("Rotation (hours):", "rotation", "Spectral type:", "spectral"),
        ]
        for fields in detail_fields:
            details_layout.addWidget(QLabel(fields[0]), row, 0)
            lbl = QLabel("---")
            lbl.setStyleSheet("font-weight: bold;")
            self.detail_labels[fields[1]] = lbl
            details_layout.addWidget(lbl, row, 1)

            details_layout.addWidget(QLabel(fields[2]), row, 2)
            lbl2 = QLabel("---")
            lbl2.setStyleSheet("font-weight: bold;")
            self.detail_labels[fields[3]] = lbl2
            details_layout.addWidget(lbl2, row, 3)
            row += 1

        details_group.setLayout(details_layout)
        layout.addWidget(details_group)

        # --- Options Section ---
        options_group = QGroupBox("Download Options")
        options_layout = QVBoxLayout()

        # DAMIT shape checkbox
        damit_row = QHBoxLayout()
        self.damit_check = QCheckBox("Fetch DAMIT shape model (asteroids only)")
        self.damit_check.setChecked(True)
        self.damit_check.setEnabled(_damit_available)
        damit_row.addWidget(self.damit_check)
        options_layout.addLayout(damit_row)

        # SPK download option
        spk_row = QHBoxLayout()
        self.spk_check = QCheckBox("Download SPK ephemeris for simulation window")
        self.spk_check.setChecked(False)
        spk_row.addWidget(self.spk_check)
        options_layout.addLayout(spk_row)

        options_group.setLayout(options_layout)
        layout.addWidget(options_group)

        # --- Action buttons ---
        btn_row = QHBoxLayout()

        self.fetch_btn = QPushButton("Fetch and Use Body")
        self.fetch_btn.setEnabled(False)
        self.fetch_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; "
            "padding: 8px 16px; font-weight: bold; border-radius: 4px; }"
            "QPushButton:hover { background-color: #45a049; }"
            "QPushButton:disabled { background-color: #555; color: #888; }"
        )
        self.fetch_btn.clicked.connect(self._on_fetch_and_use)
        btn_row.addWidget(self.fetch_btn)

        btn_row.addStretch()

        self.clear_cache_btn = QPushButton("Clear Cache")
        self.clear_cache_btn.clicked.connect(self._on_clear_cache)
        btn_row.addWidget(self.clear_cache_btn)

        layout.addLayout(btn_row)

        # --- Progress ---
        self.progress_label = QLabel("")
        self.progress_label.setStyleSheet("color: #64b5f6; font-size: 10px;")
        layout.addWidget(self.progress_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(0)  # Indeterminate
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # --- Cached Bodies ---
        cache_group = QGroupBox("Cached Bodies")
        cache_layout = QVBoxLayout()

        self.cache_table = QTableWidget(0, 3)
        self.cache_table.setHorizontalHeaderLabels(["Name", "Type", "Config Path"])
        self.cache_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self.cache_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self.cache_table.setMaximumHeight(120)
        self.cache_table.itemDoubleClicked.connect(self._on_cache_double_click)
        cache_layout.addWidget(self.cache_table)

        use_cached_btn = QPushButton("Use Selected Cached Body")
        use_cached_btn.clicked.connect(self._on_use_cached)
        cache_layout.addWidget(use_cached_btn)

        cache_group.setLayout(cache_layout)
        layout.addWidget(cache_group)

        layout.addStretch()

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _on_search(self):
        query = self.search_input.text().strip()
        if not query:
            return
        if not _horizons_available:
            QMessageBox.warning(self, "Not Available",
                                "HORIZONS client is not available.")
            return

        self.search_btn.setEnabled(False)
        self.progress_label.setText("Searching HORIZONS...")
        self.progress_bar.setVisible(True)
        self.results_table.setRowCount(0)

        self._worker = FetchWorker("search", query=query)
        self._worker.result_ready.connect(self._on_worker_result)
        self._worker.error_occurred.connect(self._on_worker_error)
        self._worker.start()

    def _on_result_selected(self):
        rows = self.results_table.selectionModel().selectedRows()
        self.fetch_btn.setEnabled(len(rows) > 0)

    def _on_fetch_and_use(self):
        rows = self.results_table.selectionModel().selectedRows()
        if not rows:
            return

        row = rows[0].row()
        naif_id = self.results_table.item(row, 1).text()
        name = self.results_table.item(row, 0).text()

        # Build query: use NAIF ID if available, with semicolon for small bodies
        body_type = self.results_table.item(row, 2).text()
        if naif_id:
            query = naif_id
            if body_type == "asteroid":
                query = f"{naif_id};"
        else:
            query = name

        self.fetch_btn.setEnabled(False)
        self.search_btn.setEnabled(False)
        self.progress_label.setText("Fetching body data...")
        self.progress_bar.setVisible(True)

        # Get SPK time range from simulation config if available
        spk_start = ""
        spk_stop = ""
        equilibration_orbits = 10  # default
        if self.spk_check.isChecked() and self.config_manager:
            config = self.config_manager.get_config()
            sim = config.get("simulation", {})
            thermal = config.get("thermal", {})
            spk_start = sim.get("start_time", "2025-01-01")
            duration = sim.get("duration_days", 30)
            equilibration_orbits = thermal.get("equilibration_orbits", 10)
            # Extend SPK range by 10x duration for safety (forward)
            from datetime import datetime, timedelta
            try:
                start_dt = datetime.fromisoformat(spk_start.replace("T", " ").split(".")[0])
                stop_dt = start_dt + timedelta(days=duration * 10)
                spk_stop = stop_dt.strftime("%Y-%m-%d")
            except Exception:
                spk_start = ""
                spk_stop = ""

        self._worker = FetchWorker(
            "fetch_and_cache",
            query=query,
            fetch_damit=self.damit_check.isChecked(),
            download_spk=self.spk_check.isChecked(),
            spk_start=spk_start,
            spk_stop=spk_stop,
            equilibration_orbits=equilibration_orbits,
        )
        self._worker.result_ready.connect(self._on_worker_result)
        self._worker.error_occurred.connect(self._on_worker_error)
        self._worker.progress_update.connect(self._on_progress)
        self._worker.start()

    # ------------------------------------------------------------------
    # Worker callbacks
    # ------------------------------------------------------------------

    def _on_worker_result(self, result):
        self.search_btn.setEnabled(True)
        self.fetch_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

        task, data = result

        if task == "search":
            self.progress_label.setText(f"Found {len(data)} results")
            self.results_table.setRowCount(len(data))
            for i, r in enumerate(data):
                self.results_table.setItem(i, 0, QTableWidgetItem(r.name))
                self.results_table.setItem(i, 1, QTableWidgetItem(r.naif_id))
                self.results_table.setItem(i, 2, QTableWidgetItem(r.body_type))

            # Auto-select first result
            if data:
                self.results_table.selectRow(0)

        elif task == "fetch_body":
            body = data
            self._update_details(body)

        elif task == "cached":
            config_path = data["config_path"]
            body = data["body"]
            self._update_details(body)
            self._refresh_cache_list()

            body_name = body.name or "Unknown"
            self.progress_label.setText(
                f"Cached '{body_name}' - config: {config_path}"
            )
            self.body_selected.emit(body_name, config_path)

    def _on_worker_error(self, error_msg):
        self.search_btn.setEnabled(True)
        self.fetch_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.progress_label.setText(f"Error: {error_msg}")
        logger.error("HORIZONS worker error: %s", error_msg)

    def _on_progress(self, msg):
        self.progress_label.setText(msg)

    # ------------------------------------------------------------------
    # Details display
    # ------------------------------------------------------------------

    def _update_details(self, body):
        self._current_body = body

        def fmt(val, dec=4):
            if val == 0.0 or val == -1.0:
                return "---"
            return f"{val:.{dec}f}"

        self.detail_labels["name"].setText(body.name or "---")
        self.detail_labels["naif_id"].setText(body.naif_id or "---")
        self.detail_labels["sma"].setText(fmt(body.elements.semi_major_axis_au))
        self.detail_labels["ecc"].setText(fmt(body.elements.eccentricity))
        self.detail_labels["inc"].setText(fmt(body.elements.inclination_deg, 2))
        self.detail_labels["period"].setText(fmt(body.elements.orbital_period_days, 1))
        self.detail_labels["diameter"].setText(fmt(body.physical.diameter_km, 1))
        self.detail_labels["albedo"].setText(fmt(body.physical.albedo, 3))
        self.detail_labels["rotation"].setText(fmt(body.physical.rotation_period_hours, 3))
        self.detail_labels["spectral"].setText(
            body.physical.spectral_type or "---"
        )

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def _refresh_cache_list(self):
        self.cache_table.setRowCount(0)
        if not self._data_manager:
            return

        cached = self._data_manager.list_cached()
        self.cache_table.setRowCount(len(cached))
        for i, entry in enumerate(cached):
            self.cache_table.setItem(
                i, 0, QTableWidgetItem(entry.name)
            )
            self.cache_table.setItem(
                i, 1, QTableWidgetItem(entry.body_type)
            )
            self.cache_table.setItem(
                i, 2, QTableWidgetItem(entry.config_path)
            )

    def _on_use_cached(self):
        rows = self.cache_table.selectionModel().selectedRows()
        if not rows:
            return
        row = rows[0].row()
        name = self.cache_table.item(row, 0).text()
        config_path = self.cache_table.item(row, 2).text()
        if name and config_path:
            self.body_selected.emit(name, config_path)

    def _on_cache_double_click(self, item):
        row = item.row()
        name = self.cache_table.item(row, 0).text()
        config_path = self.cache_table.item(row, 2).text()
        if name and config_path:
            self.body_selected.emit(name, config_path)

    def _on_clear_cache(self):
        if not self._data_manager:
            return
        reply = QMessageBox.question(
            self, "Clear Cache",
            "Remove all cached HORIZONS body data?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._data_manager.clear_all()
            self._refresh_cache_list()
            self.progress_label.setText("Cache cleared.")

    def get_cached_body_names(self):
        """Return list of cached body names for the body combo box."""
        if not self._data_manager:
            return []
        return [
            entry.name
            for entry in self._data_manager.list_cached()
            if entry.name
        ]

    def get_cached_config_path(self, body_name):
        """Return the config path for a cached body."""
        if not self._data_manager:
            return ""
        return self._data_manager.get_config_path(body_name) or ""
