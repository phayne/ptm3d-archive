#!/usr/bin/env python3
"""Paths to LOCAL run directories the suite uses, kept out of the repository.

Some checks are only meaningful against a real engine output: the run that
provoked a gate, the product a packager was written for. Those directories are
machine-local, and several of them are named after sites this project does not
publish, so the paths cannot live in a tracked test file.

They come from ``~/.ptm3d/test_fixtures.json`` instead -- beside
``machine_profile.json`` and ``curated_sites_local.json``, the same convention
those already use for "facts about THIS computer". Override the location with
``PTM3D_TEST_FIXTURES``.

::

    {
      "schema": "ptm3d.test_fixtures/1",
      "paths": {
        "lunar_radiation_run": "/absolute/path/to/a/finished/run",
        "large_lunar_core_mesh": "/absolute/path/to/a/production/core.ply"
      }
    }

``paths`` entries may be directories or files; :func:`fixture_path` checks for a
directory and :func:`fixture_file` for a file, so a name registered as the wrong
kind is a named skip rather than a confusing read error.

A name that is not in the file, or a path in it that is not a directory, is a
NAMED SKIP -- never a silent pass. The distinction matters here more than
usual: these arms exist because a synthetic fixture could not have caught what
they caught, so an arm that quietly passed with nothing to look at would be
worse than one that never ran.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

SCHEMA = "ptm3d.test_fixtures/1"


def fixtures_path():
    """``~/.ptm3d/test_fixtures.json``, or the env override."""
    override = os.environ.get("PTM3D_TEST_FIXTURES", "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / ".ptm3d" / "test_fixtures.json"


def _load():
    """``(mapping, problem)``. Never raises: a broken file is a named skip."""
    target = fixtures_path()
    if not target.is_file():
        return {}, (f"there is no local test-fixture file at {target}; see "
                    f"src/python/tests/local_fixtures.py for its shape")
    try:
        payload = json.loads(target.read_text())
    except (OSError, ValueError) as exc:
        return {}, f"{target} could not be read as JSON ({exc})"
    if not isinstance(payload, dict):
        return {}, f"{target}'s top level is not an object"
    if payload.get("schema") != SCHEMA:
        return {}, (f"{target} declares schema {payload.get('schema')!r} and "
                    f"this suite reads {SCHEMA!r}")
    paths = payload.get("paths")
    if not isinstance(paths, dict):
        return {}, f"{target} has no 'paths' object"
    return paths, ""


def _resolve(name, skip_exception):
    paths, problem = _load()
    if problem:
        raise skip_exception(problem)
    raw = paths.get(name)
    if not raw:
        raise skip_exception(
            f"no local test fixture is registered as {name!r} in "
            f"{fixtures_path()}")
    return Path(raw).expanduser()


def fixture_path(name, skip_exception):
    """The directory registered under ``name``, or raise ``skip_exception``.

    ``skip_exception`` is passed in rather than imported because every suite in
    this project defines its own ``Skip`` and they are not the same class. A
    helper that raised its own would be caught by nobody and would read as a
    failure.
    """
    path = _resolve(name, skip_exception)
    if not path.is_dir():
        raise skip_exception(
            f"the local test fixture {name!r} points at {path}, which is not "
            f"a directory on this machine")
    return path


def fixture_file(name, skip_exception):
    """The FILE registered under ``name``, or raise ``skip_exception``."""
    path = _resolve(name, skip_exception)
    if not path.is_file():
        raise skip_exception(
            f"the local test fixture {name!r} points at {path}, which is not "
            f"a file on this machine")
    return path


def optional_fixture(name):
    """``Path`` or None, for a caller that is not a test and cannot skip.

    ``scripts/ptm_qa.py``'s selftest uses this: it needs to DROP a case whose
    run directory is not registered here, and dropping it has to be visible in
    its output rather than silent.
    """
    paths, problem = _load()
    if problem:
        return None
    raw = paths.get(name)
    return Path(raw).expanduser() if raw else None
