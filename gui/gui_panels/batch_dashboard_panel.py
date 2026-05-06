"""Batch Dashboard panel — monitor and control long-running A3GT batch
jobs from the GUI.

This is the first GUI integration point for `scripts/run_a3gt_batch.py`
and `scripts/run_chunked_simulation.py`. It does not own the orchestrator
process; it observes the filesystem (`<output_root>/A3GT_MM*_18yr/`),
the canonical /tmp log files, and the OS process table, and offers a
small set of side-effect controls (pause file toggle, SIGTERM stop).

State semantics:
    queued    site dir does not exist
    running   site dir exists, mesh.ply present (or building),
              statistics/ NOT present, ptm_thermal_sim alive on this site
    paused    pause file present + orchestrator process not alive
    merged    statistics/ present (terminal good state)
    error     site dir exists but neither merged nor making progress

The panel polls every REFRESH_MS milliseconds and updates the table.
The user can pause between sites (touch the pause file), resume (re-run
the same batch command — left as a hint to the user; we don't auto-relaunch
because the original CLI args aren't recoverable from observed state),
or stop (SIGTERM the orchestrator).

The table refreshes are O(n_sites * n_chunks) cheap directory probes per
tick — fine for n_sites ~15 / n_chunks ~24.
"""

import os
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QGroupBox, QTextEdit,
    QFileDialog, QLineEdit, QMessageBox, QAbstractItemView,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor


REFRESH_MS = 5000          # auto-refresh cadence
LOG_TAIL_LINES = 80        # how many lines of the per-site log to show
DEFAULT_PAUSE_FILE = Path("/tmp/a3gt_batch_pause")
DEFAULT_OUTPUT_ROOT = Path("/Users/paha3326/Research/moon/artemis/a3gt")
ORCH_PROCESS_NAME = "run_a3gt_batch.py"
CHUNKED_PROCESS_NAME = "run_chunked_simulation.py"
PTM_THERMAL_BIN = "ptm_thermal_sim"
DTM_BIN = "dtm_to_ply_delaunay"
# Default target chunks per site for the "Chunks" column display. The
# canonical A3GT batch uses 24; if a future run uses --chunks N != 24
# the displayed denominator will be off but the numerator (real
# completed-chunks count) stays correct.
EXPECTED_CHUNKS_DEFAULT = 24

# Status colors mirror ModernStyle accents.
STATUS_COLORS = {
    "queued":  QColor("#7a7a7a"),
    "running": QColor("#3b8eea"),
    "paused":  QColor("#d6a23a"),
    "merged":  QColor("#3eaf7c"),
    "error":   QColor("#d33b3b"),
}


def _read_n_faces_ply(ply: Path) -> int:
    """Lightweight PLY face-count reader; returns 0 on any failure."""
    try:
        with open(ply, "rb") as f:
            for _ in range(200):
                line = f.readline()
                if not line:
                    break
                s = line.decode(errors="replace").strip()
                if s.startswith("element face "):
                    return int(s.split()[2])
                if s == "end_header":
                    break
    except Exception:
        pass
    return 0


def _orch_pid() -> Optional[int]:
    """PID of the active run_a3gt_batch.py, or None."""
    try:
        out = subprocess.check_output(
            ["pgrep", "-f", ORCH_PROCESS_NAME],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        if not out:
            return None
        # pgrep -f matches the script name + this very file's source line
        # if the dashboard's source happens to mention the name. Filter to
        # processes whose argv truly starts a python interpreter.
        for line in out.splitlines():
            try:
                pid = int(line.strip())
            except ValueError:
                continue
            cmd = subprocess.check_output(
                ["ps", "-o", "command=", "-p", str(pid)],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            if "python" in cmd and ORCH_PROCESS_NAME in cmd:
                return pid
        return None
    except subprocess.CalledProcessError:
        return None


def _free_disk_gb(path: Path) -> float:
    """Free space on the volume hosting `path`, in GiB."""
    try:
        st = shutil.disk_usage(path)
        return st.free / (1024 ** 3)
    except Exception:
        return 0.0


def _site_status(out_dir: Path):
    """Inspect one site's output dir; return a dict with status fields.

    Keys: status, n_chunks_complete, n_chunks_total, latest_chunk,
    latest_mtime (epoch s), merged (bool), face_count (int).
    """
    info = {
        "status": "queued",
        "n_chunks_complete": 0,
        "n_chunks_total": 0,
        "latest_chunk": "",
        "latest_mtime": 0.0,
        "merged": False,
        "face_count": 0,
    }
    if not out_dir.is_dir():
        return info

    stats_dir = out_dir / "statistics"
    info["merged"] = (
        (stats_dir / "temperature_avg.bin").is_file()
        and (stats_dir / "ice_depth.bin").is_file()
    )

    chunk_dirs = sorted(out_dir.glob("chunk_*"))
    info["n_chunks_total"] = len(chunk_dirs)
    info["n_chunks_complete"] = sum(
        1 for c in chunk_dirs
        if (c / "statistics" / "_raw" / "temperature_sum.bin").is_file()
    )

    if chunk_dirs:
        latest = max(chunk_dirs, key=lambda p: p.stat().st_mtime)
        info["latest_chunk"] = latest.name
        info["latest_mtime"] = latest.stat().st_mtime

    # mesh.ply at the site root only appears after merge (the chunked
    # orchestrator copies/hardlinks chunk_000/mesh.ply to the parent
    # then). For in-flight runs, fall back to chunk_000/mesh.ply.
    for candidate in (out_dir / "mesh.ply",
                      out_dir / "chunk_000" / "mesh.ply"):
        if candidate.is_file():
            info["face_count"] = _read_n_faces_ply(candidate)
            if info["face_count"]:
                break

    if info["merged"]:
        info["status"] = "merged"
    elif chunk_dirs:
        info["status"] = "running"
    else:
        info["status"] = "queued"
    return info


def _human_duration(seconds: float) -> str:
    if seconds < 0 or seconds != seconds:
        return "—"
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    h = int(seconds // 3600)
    m = int((seconds - h * 3600) // 60)
    return f"{h}h{m}m"


def _human_mtime_age(mtime: float) -> str:
    if mtime <= 0:
        return "—"
    return _human_duration(time.time() - mtime) + " ago"


class BatchDashboardPanel(QWidget):
    """Monitor + control panel for A3GT batch runs.

    Read-only by default; the action buttons (Pause/Resume/Stop) are the
    only side-effecting controls and each prompt for confirmation before
    touching the filesystem or sending signals.
    """

    site_selected = pyqtSignal(str, str)  # (site_tag, output_dir)

    # The set of A3GT MM site tags this dashboard knows about. We read
    # them lazily so the dashboard works even before the batch has been
    # started or before the output_root is populated.
    DEFAULT_SITES = [f"A3GT_MM{i:02d}" for i in range(1, 16)]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.output_root = DEFAULT_OUTPUT_ROOT
        self.pause_file = DEFAULT_PAUSE_FILE
        self.batch_log_path = self._guess_batch_log()
        self.last_seen_site = None
        self._init_ui()
        self._refresh_now()
        self.timer = QTimer(self)
        self.timer.setInterval(REFRESH_MS)
        self.timer.timeout.connect(self._refresh_now)
        self.timer.start()

    # --- UI assembly ---------------------------------------------------

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Top: output-root + log path picker
        path_row = QGridLayout()
        path_row.addWidget(QLabel("Output root:"), 0, 0)
        self.output_root_edit = QLineEdit(str(self.output_root))
        self.output_root_edit.setToolTip(
            "Parent directory containing A3GT_MM*_18yr/ subdirs."
        )
        self.output_root_edit.editingFinished.connect(self._on_output_root_changed)
        path_row.addWidget(self.output_root_edit, 0, 1)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse_output_root)
        path_row.addWidget(browse, 0, 2)

        path_row.addWidget(QLabel("Batch log:"), 1, 0)
        self.batch_log_edit = QLineEdit(str(self.batch_log_path or ""))
        self.batch_log_edit.setToolTip(
            "Path to the run_a3gt_batch.py stdout log "
            "(typically /tmp/a3gt_*.log)."
        )
        self.batch_log_edit.editingFinished.connect(self._on_log_changed)
        path_row.addWidget(self.batch_log_edit, 1, 1)
        log_browse = QPushButton("Browse…")
        log_browse.clicked.connect(self._browse_batch_log)
        path_row.addWidget(log_browse, 1, 2)
        layout.addLayout(path_row)

        # Status banner
        banner = QHBoxLayout()
        self.orch_label = QLabel("orch: —")
        self.disk_label = QLabel("disk: —")
        self.pause_state_label = QLabel("pause file: not present")
        for w in (self.orch_label, self.disk_label, self.pause_state_label):
            w.setStyleSheet("QLabel { padding: 2px 8px; }")
            banner.addWidget(w)
        banner.addStretch()
        layout.addLayout(banner)

        # Action buttons
        actions = QHBoxLayout()
        self.pause_btn = QPushButton("Request Pause")
        self.pause_btn.setToolTip(
            "Touch the pause file. The orchestrator will exit cleanly "
            "after the currently-running site's merge + cleanup. Use "
            "'Clear Pause' to allow a future --resume to continue."
        )
        self.pause_btn.clicked.connect(self._on_pause_clicked)
        actions.addWidget(self.pause_btn)

        self.clear_pause_btn = QPushButton("Clear Pause")
        self.clear_pause_btn.setToolTip(
            "Remove the pause file. Re-run the batch command with "
            "--resume to continue from the last incomplete site."
        )
        self.clear_pause_btn.clicked.connect(self._on_clear_pause_clicked)
        actions.addWidget(self.clear_pause_btn)

        self.stop_btn = QPushButton("Stop Orchestrator")
        self.stop_btn.setToolTip(
            "Send SIGTERM to run_a3gt_batch.py and any active "
            "ptm_thermal_sim child. Existing chunk output stays on disk; "
            "use --resume later to pick up where it left off."
        )
        self.stop_btn.clicked.connect(self._on_stop_clicked)
        actions.addWidget(self.stop_btn)

        actions.addStretch()
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self._refresh_now)
        actions.addWidget(self.refresh_btn)
        layout.addLayout(actions)

        # Sites table
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels([
            "Site", "Status", "Chunks", "Latest", "Last activity",
            "Faces", "Output dir",
        ])
        self.table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.table.verticalHeader().setVisible(False)
        h = self.table.horizontalHeader()
        h.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        h.setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch)
        self.table.itemSelectionChanged.connect(self._on_table_selection)
        layout.addWidget(self.table, 1)

        # Live log tail
        log_group = QGroupBox("Live log (active site)")
        log_layout = QVBoxLayout()
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setStyleSheet(
            "QTextEdit { font-family: Menlo, monospace; font-size: 11px; }"
        )
        self.log_view.setMinimumHeight(160)
        log_layout.addWidget(self.log_view)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

    # --- Helpers -------------------------------------------------------

    def _guess_batch_log(self) -> Optional[Path]:
        """Heuristic: pick the most recently-modified /tmp/a3gt_*.log."""
        try:
            cands = sorted(
                Path("/tmp").glob("a3gt_*.log"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            return cands[0] if cands else None
        except Exception:
            return None

    def _on_output_root_changed(self):
        self.output_root = Path(self.output_root_edit.text().strip())
        self._refresh_now()

    def _browse_output_root(self):
        d = QFileDialog.getExistingDirectory(
            self, "Choose batch output root", str(self.output_root),
        )
        if d:
            self.output_root = Path(d)
            self.output_root_edit.setText(d)
            self._refresh_now()

    def _on_log_changed(self):
        text = self.batch_log_edit.text().strip()
        self.batch_log_path = Path(text) if text else None
        self._refresh_now()

    def _browse_batch_log(self):
        d, _ = QFileDialog.getOpenFileName(
            self, "Choose batch log", "/tmp",
            "Log files (*.log);;All files (*)",
        )
        if d:
            self.batch_log_path = Path(d)
            self.batch_log_edit.setText(d)
            self._refresh_now()

    # --- Actions -------------------------------------------------------

    def _on_pause_clicked(self):
        if self.pause_file.exists():
            QMessageBox.information(
                self, "Pause already requested",
                f"Pause file {self.pause_file} already present.",
            )
            return
        try:
            self.pause_file.parent.mkdir(parents=True, exist_ok=True)
            self.pause_file.touch()
        except Exception as exc:
            QMessageBox.warning(
                self, "Pause failed",
                f"Could not create {self.pause_file}: {exc}",
            )
            return
        QMessageBox.information(
            self, "Pause requested",
            f"Pause file created at {self.pause_file}.\n\n"
            "The orchestrator will exit cleanly after the currently-"
            "running site's merge + cleanup.",
        )
        self._refresh_now()

    def _on_clear_pause_clicked(self):
        if not self.pause_file.exists():
            QMessageBox.information(
                self, "No pause to clear",
                f"Pause file {self.pause_file} is not present.",
            )
            return
        try:
            self.pause_file.unlink()
        except Exception as exc:
            QMessageBox.warning(
                self, "Clear failed",
                f"Could not remove {self.pause_file}: {exc}",
            )
            return
        self._refresh_now()

    def _on_stop_clicked(self):
        pid = _orch_pid()
        if not pid:
            QMessageBox.information(
                self, "Nothing to stop",
                "No run_a3gt_batch.py process is currently running.",
            )
            return
        ok = QMessageBox.question(
            self, "Stop orchestrator",
            f"Send SIGTERM to run_a3gt_batch.py (PID {pid}) and any "
            f"active ptm_thermal_sim children?\n\n"
            f"Existing chunk output stays on disk; use --resume later "
            f"to pick up where it left off.",
        )
        if ok != QMessageBox.StandardButton.Yes:
            return
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except Exception as exc:
            QMessageBox.warning(
                self, "Stop failed", f"SIGTERM failed: {exc}",
            )
            return
        # Children typically terminate when the orchestrator's
        # subprocess.run() chain unwinds, but ptm_thermal_sim is launched
        # by run_chunked_simulation.py (a grandchild), so we kill them
        # directly to avoid orphaned compute.
        for proc_name in (CHUNKED_PROCESS_NAME, PTM_THERMAL_BIN, DTM_BIN):
            subprocess.run(
                ["pkill", "-TERM", "-f", proc_name],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        self._refresh_now()

    # --- Refresh -------------------------------------------------------

    def _refresh_now(self):
        # Banner
        pid = _orch_pid()
        if pid:
            self.orch_label.setText(f"orch: PID {pid} alive")
            self.orch_label.setStyleSheet(
                "QLabel { padding: 2px 8px; color: #3eaf7c; }"
            )
        else:
            self.orch_label.setText("orch: not running")
            self.orch_label.setStyleSheet(
                "QLabel { padding: 2px 8px; color: #888888; }"
            )

        free_gb = _free_disk_gb(self.output_root if self.output_root.exists()
                                else Path.home())
        self.disk_label.setText(f"disk free: {free_gb:.1f} GiB")
        if free_gb < 50:
            self.disk_label.setStyleSheet(
                "QLabel { padding: 2px 8px; color: #d33b3b; }"
            )
        else:
            self.disk_label.setStyleSheet(
                "QLabel { padding: 2px 8px; }"
            )

        if self.pause_file.exists():
            self.pause_state_label.setText(
                f"pause file: PRESENT ({self.pause_file})"
            )
            self.pause_state_label.setStyleSheet(
                "QLabel { padding: 2px 8px; color: #d6a23a; }"
            )
        else:
            self.pause_state_label.setText("pause file: not present")
            self.pause_state_label.setStyleSheet(
                "QLabel { padding: 2px 8px; }"
            )

        # Sites table
        sites = self._discover_sites()
        self.table.setRowCount(len(sites))
        for row, (tag, out_dir) in enumerate(sites):
            info = _site_status(out_dir)
            self._set_row(row, tag, out_dir, info)

        # Active log tail
        active_site = self._active_site_tag(sites)
        if active_site != self.last_seen_site:
            self.last_seen_site = active_site
        self._refresh_log_view(active_site)

    def _discover_sites(self):
        """Return [(tag, out_dir), ...] in canonical site order. Includes
        sites with output dirs even if not in DEFAULT_SITES (so a
        non-A3GT batch could be observed too)."""
        seen = set()
        sites = []
        for tag in self.DEFAULT_SITES:
            out_dir = self.output_root / f"{tag}_18yr"
            sites.append((tag, out_dir))
            seen.add(tag)
        if self.output_root.is_dir():
            for child in sorted(self.output_root.iterdir()):
                if not child.is_dir() or not child.name.endswith("_18yr"):
                    continue
                tag = child.name[: -len("_18yr")]
                if tag not in seen:
                    sites.append((tag, child))
        return sites

    def _set_row(self, row, tag, out_dir, info):
        # Denominator is the typical-target 24 unless we observe more
        # chunk dirs (e.g. a future --chunks N>24 run). Numerator is
        # always the real complete-chunks count.
        denom = max(EXPECTED_CHUNKS_DEFAULT, info["n_chunks_total"])
        if info["status"] == "merged":
            chunks_text = f"{denom} / {denom}"
        elif info["n_chunks_total"]:
            chunks_text = f"{info['n_chunks_complete']} / {denom}"
        else:
            chunks_text = "—"
        items = [
            QTableWidgetItem(tag),
            QTableWidgetItem(info["status"]),
            QTableWidgetItem(chunks_text),
            QTableWidgetItem(info["latest_chunk"] or "—"),
            QTableWidgetItem(_human_mtime_age(info["latest_mtime"])),
            QTableWidgetItem(
                f"{info['face_count']:,}" if info["face_count"] else "—"
            ),
            QTableWidgetItem(str(out_dir)),
        ]
        # Color the status cell.
        color = STATUS_COLORS.get(info["status"])
        if color is not None:
            items[1].setForeground(color)
        for col, it in enumerate(items):
            it.setData(Qt.ItemDataRole.UserRole, str(out_dir))
            self.table.setItem(row, col, it)

    def _active_site_tag(self, sites):
        """Pick the site that's currently being processed: the highest-
        index site whose dir exists and isn't merged. Falls back to the
        most-recently-touched site dir."""
        candidates = [
            (tag, out) for tag, out in sites
            if out.is_dir() and not (out / "statistics" /
                                     "temperature_avg.bin").is_file()
        ]
        if not candidates:
            # Already-merged most-recently-active site
            existing = [(tag, out) for tag, out in sites if out.is_dir()]
            if not existing:
                return None
            existing.sort(key=lambda t: t[1].stat().st_mtime, reverse=True)
            return existing[0][0]
        candidates.sort(key=lambda t: t[1].stat().st_mtime, reverse=True)
        return candidates[0][0]

    def _refresh_log_view(self, active_tag):
        if not active_tag:
            self.log_view.setPlainText("(no active site)")
            return
        log_path = Path(f"/tmp/{active_tag}_batch.log")
        if not log_path.is_file():
            # Fall back to the batch driver log
            log_path = self.batch_log_path
        if not log_path or not Path(log_path).is_file():
            self.log_view.setPlainText(f"(log not found: {log_path})")
            return
        try:
            # Read the tail efficiently for large logs.
            tail = self._tail(log_path, LOG_TAIL_LINES)
        except Exception as exc:
            self.log_view.setPlainText(f"(error reading {log_path}: {exc})")
            return
        # Auto-scroll only if the user is currently at the bottom.
        sb = self.log_view.verticalScrollBar()
        at_bottom = sb.value() >= sb.maximum() - 4
        self.log_view.setPlainText(tail)
        if at_bottom:
            sb.setValue(sb.maximum())

    @staticmethod
    def _tail(path, n_lines):
        """Read approximately the last n_lines lines of a file. Reads at
        most ~512 KB from the tail to bound memory."""
        size_cap = 512 * 1024
        try:
            sz = path.stat().st_size
            with open(path, "rb") as f:
                if sz > size_cap:
                    f.seek(sz - size_cap)
                    f.readline()  # discard partial first line
                buf = f.read()
        except Exception:
            return ""
        lines = buf.decode(errors="replace").splitlines()
        return "\n".join(lines[-n_lines:])

    def _on_table_selection(self):
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return
        row = rows[0].row()
        tag_item = self.table.item(row, 0)
        out_item = self.table.item(row, 6)
        if tag_item and out_item:
            self.site_selected.emit(tag_item.text(), out_item.text())
