#!/usr/bin/env python3
"""Per-step frame names: old (4-digit) and new (>= 6-digit) runs both load.

BUGS.md PTM-664. Engines before the fix zero-padded the frame index to FOUR
digits -- a minimum width, so frame 10000 was written ``10000`` -- and an
alphabetical sort of a run with more than 10,000 frames came out
0000 ... 1000, 10000, 10001, ..., 1001. That scrambled a Justitia TWG Case 4
reduction. Current engines pad to max(6, digits(n_frames - 1)).

Asserted:

1. ``ptm.readers.sorted_frame_files`` returns NUMERIC order on a legacy run
   that crosses 10,000 frames (the exact case that broke), on a new run, and
   ignores same-prefix non-frame files;
2. ``ptm.readers.frame_path`` resolves frame k under either naming and names
   the current-rule path for a missing frame;
3. ``ptm.results.Results.steps`` / ``.field`` read both generations and agree
   frame by frame;
4. a directory holding the same index under two widths is REFUSED, not
   silently deduplicated.

No engine, no SPICE. Run directly (no pytest needed)::

    python3 src/python/tests/test_frame_naming.py
"""

from __future__ import annotations

import shutil
import struct
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                               # noqa: E402

from ptm import readers                                          # noqa: E402


def _write_bin(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(values, dtype="<f4")
    with open(path, "wb") as handle:
        handle.write(struct.pack("<I", values.size))
        handle.write(values.tobytes())


def _legacy_name(k):
    return f"temperature_{k:04d}.bin"          # what pre-PTM-664 engines wrote


def _new_name(k, n):
    return f"temperature_{k:0{readers.frame_index_width(n)}d}.bin"


# Indices straddling the 10,000 rollover. The frame VALUE is its index, so any
# misordering is visible in the data, not only in the names.
INDICES = [0, 1, 999, 1000, 1001, 9998, 9999, 10000, 10001, 10002]


def test_sorted_frame_files_legacy_rollover():
    with tempfile.TemporaryDirectory() as d:
        tdir = Path(d) / "temperature"
        for k in INDICES:
            _write_bin(tdir / _legacy_name(k), [float(k)])
        # A same-prefix companion that is NOT a frame must be ignored.
        _write_bin(tdir / "temperature_z0p05m_max.bin", [-1.0])
        names = [p.name for p in sorted(tdir.glob("temperature_*.bin"))]
        assert names.index("temperature_10000.bin") < names.index("temperature_1001.bin"), \
            "control: the legacy names DO sort wrongly alphabetically"
        got = [readers.frame_index(p) for p in readers.sorted_frame_files(tdir)]
        assert got == INDICES, got


def test_sorted_frame_files_new_naming():
    n = 10003
    with tempfile.TemporaryDirectory() as d:
        tdir = Path(d) / "temperature"
        for k in INDICES:
            _write_bin(tdir / _new_name(k, n), [float(k)])
        files = readers.sorted_frame_files(tdir)
        assert [readers.frame_index(p) for p in files] == INDICES
        # New names need no special reader: alphabetical == numeric.
        assert sorted(p.name for p in files) == [p.name for p in files]
        assert files[0].name == "temperature_000000.bin"


def test_frame_path_both_generations():
    with tempfile.TemporaryDirectory() as d:
        tdir = Path(d) / "temperature"
        _write_bin(tdir / "temperature_0012.bin", [12.0])        # legacy
        _write_bin(tdir / "temperature_10000.bin", [10000.0])    # legacy rollover
        _write_bin(tdir / "temperature_000013.bin", [13.0])      # new
        _write_bin(tdir / "temperature_1234567.bin", [1.0])      # new, 7 digits
        assert readers.frame_path(tdir, "temperature", 12).name == "temperature_0012.bin"
        assert readers.frame_path(tdir, "temperature", 10000).name == "temperature_10000.bin"
        assert readers.frame_path(tdir, "temperature", 13).name == "temperature_000013.bin"
        assert readers.frame_path(tdir, "temperature", 1234567).name == "temperature_1234567.bin"
        missing = readers.frame_path(tdir, "temperature", 14, n_frames=2_000_000)
        assert missing.name == "temperature_0000014.bin" and not missing.exists()


def test_results_api_reads_both_generations():
    from test_results_api import make_run                        # 4-digit writer
    from ptm.results import Results
    tmp = Path(tempfile.mkdtemp(prefix="ptm_frames_"))
    try:
        old = make_run(tmp / "old", n_steps=6)
        new = make_run(tmp / "new", n_steps=6)
        for sub in ("temperature", "direct_solar"):
            for p in list((new / sub).glob(f"{sub}_*.bin")):
                k = readers.frame_index(p)
                p.rename(p.with_name(f"{sub}_{k:06d}.bin"))
        r_old, r_new = Results(old), Results(new)
        assert r_old.steps() == r_new.steps() == list(range(6))
        for k in range(6):
            assert np.array_equal(r_old.field("temperature", k),
                                  r_new.field("temperature", k)), k
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_mixed_width_duplicate_is_refused():
    with tempfile.TemporaryDirectory() as d:
        tdir = Path(d) / "temperature"
        _write_bin(tdir / "temperature_0007.bin", [7.0])
        _write_bin(tdir / "temperature_000007.bin", [7.5])
        try:
            readers.sorted_frame_files(tdir)
        except ValueError as exc:
            assert "more than one" in str(exc)
        else:
            raise AssertionError("two files for frame 7 were accepted silently")


TESTS = [test_sorted_frame_files_legacy_rollover, test_sorted_frame_files_new_naming,
         test_frame_path_both_generations, test_results_api_reads_both_generations,
         test_mixed_width_duplicate_is_refused]


if __name__ == "__main__":
    failed = 0
    for t in TESTS:
        try:
            t()
            print(f"  ok    {t.__name__}")
        except Exception as exc:                                  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {t.__name__}: {exc!r}")
    print(f"{len(TESTS) - failed}/{len(TESTS)} passed")
    sys.exit(1 if failed else 0)
