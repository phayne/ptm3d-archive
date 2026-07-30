"""
Fetch-on-missing dialog for built-in datasets.

Runs `scripts/fetch_data.py --dataset <name>` as a subprocess (QProcess)
with live output in the dialog, then lets the caller re-validate. Dataset
names come from data/data_manifest.json (see gui_core.datasets); if the
manifest base_url is still the ZENODO placeholder the dialog says so
plainly ("data source not yet published") instead of attempting a broken
download -- a staging base URL (https://... or file:///...) can be entered
to fetch from a pre-publication copy.
"""

import os

from PyQt6.QtCore import QProcess
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit,
    QPushButton
)

from gui_core import datasets


class FetchDialog(QDialog):
    """Modal dialog that fetches one dataset with visible progress."""

    def __init__(self, dataset, project_root, parent=None):
        super().__init__(parent)
        self.dataset = dataset
        self.project_root = project_root
        self.succeeded = False
        self._process = None

        manifest = datasets.read_manifest(project_root)
        ds = (manifest.get("datasets", {}) or {}).get(dataset, {})
        self._placeholder = datasets.base_url_is_placeholder(manifest)

        self.setWindowTitle(f"Fetch dataset: {dataset}")
        self.setMinimumSize(640, 420)
        layout = QVBoxLayout(self)

        desc = ds.get("description", "")
        size = sum(f.get("size", 0) for f in ds.get("files", []) or [])
        head = QLabel(
            f"<b>{dataset}</b>: {desc}<br>"
            f"Download size: {size / 1e6:.0f} MB — unpacked into the data "
            f"directory (PTM_DATA_ROOT or repository data/)."
        )
        head.setWordWrap(True)
        layout.addWidget(head)

        if self._placeholder:
            note = QLabel(
                "The public data deposit is <b>not yet published</b> (the "
                "manifest URL is a placeholder). Enter a staging base URL "
                "below (https://... or file:///...) if you have one; "
                "otherwise check for an updated PTM3D release."
            )
            note.setWordWrap(True)
            note.setStyleSheet("color: #e6a23c;")
            layout.addWidget(note)

        url_row = QHBoxLayout()
        url_row.addWidget(QLabel("Base URL override:"))
        self.base_url_edit = QLineEdit()
        self.base_url_edit.setPlaceholderText(
            "(optional; defaults to the manifest base_url)")
        self.base_url_edit.setText(os.environ.get("PTM_FETCH_BASE_URL", ""))
        url_row.addWidget(self.base_url_edit)
        layout.addLayout(url_row)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("Fetch output appears here…")
        layout.addWidget(self.log, stretch=1)

        btn_row = QHBoxLayout()
        self.status_label = QLabel("")
        btn_row.addWidget(self.status_label, stretch=1)
        self.fetch_btn = QPushButton("Fetch now")
        self.fetch_btn.clicked.connect(self._start_fetch)
        btn_row.addWidget(self.fetch_btn)
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.reject)
        btn_row.addWidget(self.close_btn)
        layout.addLayout(btn_row)

        if self._placeholder and not self.base_url_edit.text().strip():
            self.status_label.setText("Data source not yet published")

    # ------------------------------------------------------------------

    def _append(self, text):
        self.log.appendPlainText(text.rstrip("\n"))

    def _start_fetch(self):
        base_url = self.base_url_edit.text().strip() or None
        if self._placeholder and not base_url:
            self._append(
                "Cannot fetch: the public data deposit is not published yet "
                "(manifest base_url is a placeholder) and no staging base "
                "URL was given.")
            self.status_label.setText("Data source not yet published")
            return

        cmd = datasets.fetch_command(self.dataset, self.project_root,
                                     base_url=base_url)
        self._append("$ " + " ".join(str(c) for c in cmd))
        self.fetch_btn.setEnabled(False)
        self.status_label.setText("Fetching…")

        self._process = QProcess(self)
        self._process.setProcessChannelMode(
            QProcess.ProcessChannelMode.MergedChannels)
        self._process.readyReadStandardOutput.connect(self._on_output)
        self._process.finished.connect(self._on_finished)
        self._process.setWorkingDirectory(str(self.project_root))
        self._process.start(cmd[0], [str(c) for c in cmd[1:]])

    def _on_output(self):
        data = bytes(self._process.readAllStandardOutput())
        text = data.decode("utf-8", errors="replace")
        for line in text.splitlines():
            if line.strip():
                self._append(line)

    def _on_finished(self, exit_code, _exit_status):
        self.fetch_btn.setEnabled(True)
        if exit_code == 0:
            self.succeeded = True
            self.status_label.setText("Fetch complete")
            self._append("Fetch complete.")
            self.close_btn.setText("Done")
        else:
            self.status_label.setText(f"Fetch FAILED (exit {exit_code})")
            self._append(f"Fetch FAILED (exit {exit_code}).")

    def reject(self):
        if self._process and self._process.state() != QProcess.ProcessState.NotRunning:
            self._process.kill()
            self._process.waitForFinished(2000)
        super().reject()
