#!/usr/bin/env python3
"""ptm_progress.py — the heartbeat every long PTM3D job must write.

Why
---
On 2026-07-06 a render was declared stalled because a `ps` snapshot showed no worker and the
output directory's mtime had not moved. It was writing frames to a temp directory and only
lands an MP4 when each render completes. It was healthy the whole time.

A job is stalled ONLY if its own heartbeat has not advanced. `ps` + mtime cannot distinguish
"hung" from "busy on work that produces no visible file yet". So every long job writes
`<out_dir>/progress.json`, atomically, after each unit of work, and the watcher reads THAT.

Read it back with:  `python3 scripts/ptm_qa.py liveness <out_dir> --stall-min 60`

Usage
-----
    from ptm_progress import Heartbeat

    with Heartbeat(out_dir, unit_total=len(cells)) as hb:
        for i, cell in enumerate(cells, 1):
            run(cell)
            hb.beat(i, label=cell.name)     # <- rewrites progress.json atomically

A crash leaves the last beat on disk, so `liveness` reports WEDGE_SUSPECT (pid alive, no
advance) or SIM_DOWN (pid gone) rather than UNKNOWN.
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone


class Heartbeat:
    def __init__(self, out_dir: str, unit_total: int | None = None, label: str = ""):
        os.makedirs(out_dir, exist_ok=True)
        self.path = os.path.join(out_dir, "progress.json")
        self.unit_total = unit_total
        self.label = label
        self.unit = 0

    def _write(self, finished: bool = False, failed: str | None = None):
        payload = {
            "unit": self.unit,
            "unit_total": self.unit_total,
            "unit_label": self.label,
            "last_advance_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "pid": os.getpid(),
            # Without this, a job that finished cleanly is indistinguishable from one that
            # died: in both cases the pid is gone and the heartbeat stops. A watchdog would
            # then "recover" a completed run. Found while testing the liveness states.
            "finished": finished,
            "failed": failed,
        }
        # Atomic: a watcher must never read a half-written heartbeat and call it corrupt.
        d = os.path.dirname(self.path)
        fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as fh:
                json.dump(payload, fh, indent=2)
            os.replace(tmp, self.path)
        except Exception:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    def beat(self, unit: int | None = None, label: str | None = None):
        self.unit = self.unit + 1 if unit is None else unit
        if label is not None:
            self.label = label
        self._write()

    def __enter__(self):
        self._write()
        return self

    def __exit__(self, exc_type, exc, tb):
        # Leave the final beat on disk. Do NOT delete it: a missing heartbeat is
        # indistinguishable from a job that never wrote one, and `liveness` would
        # have to answer UNKNOWN instead of telling you how far it got.
        self._write(finished=exc_type is None,
                    failed=None if exc_type is None else f"{exc_type.__name__}: {exc}")
        return False


if __name__ == "__main__":
    import sys, time
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    with Heartbeat(d, unit_total=3, label="demo") as hb:
        for i in range(1, 4):
            time.sleep(0.2)
            hb.beat(i, label=f"unit {i}")
    print(f"wrote {os.path.join(d, 'progress.json')}")
