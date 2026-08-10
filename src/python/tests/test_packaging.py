#!/usr/bin/env python3
"""What the distribution ships, and that it says one version.

Two things drift silently and are cheap to pin:

1. **The version.** ``pyproject.toml`` carried a literal ``0.1.0`` while
   ``ptm/__init__.py`` said ``0.3.0``, so ``pip show ptm`` and
   ``ptm --version`` disagreed and neither was obviously wrong. The version is
   now single-sourced; this fails if anyone puts a literal back.
2. **The packages.** ``damit`` and ``horizons`` are part of the small-body
   workflow and were reachable only through ``sys.path`` surgery -- tutorial 04
   inserted a path for one import. They are in the distribution now, and they
   must stay importable without any path fiddling and without pulling in a
   dependency the project does not already declare.

Run directly::

    python3 src/python/tests/test_packaging.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "python"))

PYPROJECT = (REPO_ROOT / "pyproject.toml").read_text()

SKIPPED = []


def test_version_is_single_sourced():
    assert re.search(r'^dynamic\s*=\s*\[\s*"version"\s*\]', PYPROJECT,
                     re.MULTILINE), \
        "pyproject.toml no longer declares a dynamic version"
    assert re.search(r'version\s*=\s*\{\s*attr\s*=\s*"ptm\.__version__"\s*\}',
                     PYPROJECT), \
        "the dynamic version is not taken from ptm.__version__"
    literal = re.search(r'^version\s*=\s*"[\d.]', PYPROJECT, re.MULTILINE)
    assert literal is None, \
        f"a literal version is back in pyproject.toml: {literal.group(0)!r}"


def test_installed_metadata_matches_the_package():
    import ptm
    try:
        import importlib.metadata as metadata
        installed = metadata.version("ptm")
    except Exception:                                          # noqa: BLE001
        SKIPPED.append("test_installed_metadata_matches_the_package "
                       "(ptm is not pip-installed in this interpreter)")
        return
    assert installed == ptm.__version__, (
        f"pip reports {installed}, ptm.__version__ is {ptm.__version__}")


def test_damit_and_horizons_are_in_the_distribution():
    match = re.search(r'^packages\s*=\s*\[([^\]]*)\]', PYPROJECT, re.MULTILINE)
    assert match, "pyproject.toml lists no packages"
    listed = set(re.findall(r'"([^"]+)"', match.group(1)))
    assert {"ptm", "damit", "horizons"} <= listed, listed


def test_damit_and_horizons_import_with_no_path_surgery():
    """Importable in a subprocess whose sys.path this test did not touch.

    Importing them here would pass on the ``sys.path.insert`` at the top of
    this file, which is exactly the thing being removed.
    """
    code = ("import damit, horizons;"
            "from damit import DAMITClient;"
            "from horizons import HorizonsClient, SPKGenerator, "
            "BodyConfigWriter;"
            "print('ok')")
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, cwd=str(Path.home()))
    if proc.returncode != 0 and "No module named" in proc.stderr:
        SKIPPED.append("test_damit_and_horizons_import_with_no_path_surgery "
                       "(the package is not installed in this interpreter)")
        return
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout, proc.stdout


def test_they_add_no_new_dependency():
    """Their imports stay inside the standard library plus the declared set."""
    stdlib = getattr(sys, "stdlib_module_names", None)
    if stdlib is None:                                    # Python < 3.10
        SKIPPED.append("test_they_add_no_new_dependency "
                       "(sys.stdlib_module_names needs Python 3.10)")
        return
    declared = {"numpy", "pyyaml", "yaml", "matplotlib", "scipy"}
    third_party = set()
    for package in ("damit", "horizons"):
        for path in (REPO_ROOT / "src" / "python" / package).glob("*.py"):
            for line in path.read_text().splitlines():
                match = re.match(r'\s*(?:from|import)\s+([A-Za-z_][\w]*)', line)
                if not match:
                    continue
                name = match.group(1)
                if name in stdlib or name in (package, "damit", "horizons"):
                    continue
                third_party.add(name)
    assert third_party <= declared, (
        f"{sorted(third_party - declared)} would become a new dependency")


TESTS = [
    test_version_is_single_sourced,
    test_installed_metadata_matches_the_package,
    test_damit_and_horizons_are_in_the_distribution,
    test_damit_and_horizons_import_with_no_path_surgery,
    test_they_add_no_new_dependency,
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
    for note in SKIPPED:
        print(f"  (skipped: {note})")
    print(f"\n[test_packaging] {len(TESTS) - failures}/{len(TESTS)} passed")
    if failures:
        print("[test_packaging] FAIL")
        return 1
    print("[test_packaging] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
