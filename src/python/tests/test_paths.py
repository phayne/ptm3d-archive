#!/usr/bin/env python3
"""ptm.paths: repository/executable discovery and the run preflight.

The property worth pinning is not "does the file exist" -- it is *which remedy
gets named*. A preflight that says "MISSING jup365.bsp" and stops has moved the
notebook's problem into the library; the value is the exact command, and the
Europa kernel needs a different one from the Moon kernels while both live under
``data/kernels``. Those cases are checked here against a synthetic checkout, so
the test needs no downloads and no engine.

Run directly::

    python3 src/python/tests/test_paths.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "python"))

from ptm import paths                                       # noqa: E402

# The two datasets whose remedies the notebooks quote, exactly as
# data/data_manifest.json declares them.
MANIFEST = {
    "datasets": {
        "albedo-maps": {"files": [{"name": "a.tar.gz",
                                   "unpack_to": "data/albedo_maps/"}]},
        "moon-topo": {"files": [{"name": "t.tar.gz",
                                 "unpack_to": "data/topo/moon/"}]},
    }
}


def _fake_checkout(tmp):
    """A directory that :func:`ptm.paths.find_repo_root` accepts."""
    root = Path(tmp)
    (root / "examples").mkdir(parents=True, exist_ok=True)
    (root / "CMakeLists.txt").write_text("# stand-in\n")
    (root / "data").mkdir(exist_ok=True)
    (root / "data" / "data_manifest.json").write_text(json.dumps(MANIFEST))
    (root / "examples" / "quickstart.yaml").write_text("simulation: {}\n")
    (root / "data" / "kernels" / "lsk").mkdir(parents=True, exist_ok=True)
    (root / "data" / "kernels" / "lsk" / "naif0012.tls").write_text("x")
    return root


def test_remedy_distinguishes_the_three_kernel_sets():
    """Three body sets live under data/kernels, and --bodies moon fetches one.

    Naming the wrong one is worse than naming none: the reader runs the command,
    the file is still missing, and the second failure looks like the first. Bennu
    joined this list on 2026-08-27, when its 80.7 MB ephemeris moved out of git.
    """
    with tempfile.TemporaryDirectory() as td:
        root = _fake_checkout(td)
        moon = paths.remedy_for(root / "data/kernels/pck/pck00011.tpc", root)
        europa = paths.remedy_for(root / "data/kernels/spk/jup365.bsp", root)
        bennu = paths.remedy_for(
            root / "data/kernels/spk/sb-101955-118_long.bsp", root)
        assert moon == "python3 scripts/fetch_kernels.py --bodies moon", moon
        assert europa == "python3 scripts/fetch_kernels.py --bodies europa", europa
        assert bennu == "python3 scripts/fetch_kernels.py --bodies bennu", bennu
        assert len({moon, europa, bennu}) == 3, \
            "each body set needs its own remedy; --bodies moon fetches neither " \
            "the 1.1 GB Galilean set nor Bennu's ephemeris"
        # The metakernels ship, so they are never the missing file -- but if one
        # is ever asked about, it must not be answered with the Moon core set.
        assert paths.remedy_for(root / "data/kernels/bennu.tm", root) == bennu


def test_remedy_comes_from_the_manifest_not_a_hard_coded_table():
    with tempfile.TemporaryDirectory() as td:
        root = _fake_checkout(td)
        assert paths.remedy_for(root / "data/albedo_maps/Europa_res4.bin", root) \
            == "python3 scripts/fetch_data.py --dataset albedo-maps"
        assert paths.remedy_for(root / "data/topo/moon/topo_moon_ico5.bin", root) \
            == "python3 scripts/fetch_data.py --dataset moon-topo"
        # A dataset the manifest does not declare gets no invented command.
        assert paths.remedy_for(root / "data/nothing/here.bin", root) is None


def test_remedy_for_the_engine_names_the_build():
    with tempfile.TemporaryDirectory() as td:
        root = _fake_checkout(td)
        remedy = paths.remedy_for(root / "build/bin/ptm_thermal_sim", root)
        assert "cmake --build" in remedy, remedy


def test_preflight_reports_and_raises_with_the_remedies():
    with tempfile.TemporaryDirectory() as td:
        root = _fake_checkout(td)
        report = paths.preflight(
            ["examples/quickstart.yaml",
             "data/kernels/lsk/naif0012.tls",
             "data/albedo_maps/Europa_res4.bin"],
            root=root, engine="ptm_thermal_sim")
        assert not report.ok
        assert len(report.checks) == 4 and len(report.missing) == 2
        assert report.engine is None, "no engine was built in this fake checkout"
        text = str(report)
        assert "MISSING" in text and "found" in text
        try:
            report.raise_if_missing()
        except FileNotFoundError as exc:
            message = str(exc)
        else:
            raise AssertionError("raise_if_missing did not raise")
        assert "fetch_data.py --dataset albedo-maps" in message, message
        assert "cmake --build" in message, message
        # Only the missing ones are named; a found file must not be nagged about.
        assert "naif0012.tls" not in message, message


def test_preflight_ok_run_is_silent_and_finds_the_engine():
    with tempfile.TemporaryDirectory() as td:
        root = _fake_checkout(td)
        exe = root / "build" / "bin" / "ptm_thermal_sim"
        exe.parent.mkdir(parents=True)
        exe.write_text("#!/bin/sh\n")
        report = paths.preflight(["examples/quickstart.yaml"], root=root,
                                 engine="ptm_thermal_sim")
        assert report.ok, str(report)
        assert report.engine == exe.resolve(), report.engine
        assert report.raise_if_missing() is report


def test_preflight_accepts_an_explicit_name_and_remedy():
    with tempfile.TemporaryDirectory() as td:
        root = _fake_checkout(td)
        report = paths.preflight(
            [("data/custom/thing.bin", "the custom thing", "ask Paul")],
            root=root)
        check = report.checks[0]
        assert check.what == "the custom thing" and check.remedy == "ask Paul"
        assert not check.found


def test_find_repo_root_still_works():
    assert paths.find_repo_root(REPO_ROOT / "src" / "python") == REPO_ROOT


TESTS = [
    test_remedy_distinguishes_the_three_kernel_sets,
    test_remedy_comes_from_the_manifest_not_a_hard_coded_table,
    test_remedy_for_the_engine_names_the_build,
    test_preflight_reports_and_raises_with_the_remedies,
    test_preflight_ok_run_is_silent_and_finds_the_engine,
    test_preflight_accepts_an_explicit_name_and_remedy,
    test_find_repo_root_still_works,
]


def main():
    failures = 0
    for test in TESTS:
        try:
            test()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {test.__name__}\n     {exc}")
        except Exception as exc:                       # noqa: BLE001
            failures += 1
            print(f"ERROR {test.__name__}: {type(exc).__name__}: {exc}")
        else:
            print(f"PASS {test.__name__}")
    print(f"\n[test_paths] {len(TESTS) - failures}/{len(TESTS)} passed")
    if failures:
        print("[test_paths] FAIL")
        return 1
    print("[test_paths] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
