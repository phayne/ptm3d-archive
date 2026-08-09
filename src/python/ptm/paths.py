"""Locating the repository and the engine executables.

Every tutorial notebook and most scripts in the tree re-derive these two
things. The rules here are the ones already in use, collected in one place:

* a PTM3D checkout is a directory containing ``CMakeLists.txt`` and
  ``examples/`` (the test the tutorial notebooks use);
* the engine binaries live in ``<root>/build/bin/``.

Nothing here guesses silently. ``PTM_BUILD_DIR`` and ``PTM_ROOT`` override the
search, and a value that does not resolve produces a warning on stderr rather
than a quiet fall-through to some other candidate.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

__all__ = [
    "ENGINE_BINARIES",
    "find_repo_root",
    "find_engine_executable",
    "build_dir_candidates",
]

#: The binaries CMake installs into ``<root>/build/bin``.
ENGINE_BINARIES = (
    "ptm_thermal_sim",
    "ptm_camera_sim",
    "ptm_exosphere_main",
    "ptm_merge_chunks",
)

_ROOT_MARKERS = ("CMakeLists.txt", "examples")


def _looks_like_repo(path: Path) -> bool:
    return all((path / m).exists() for m in _ROOT_MARKERS)


def find_repo_root(start=None):
    """Return the PTM3D checkout containing ``start``, or ``None``.

    Walks upward from ``start`` (default: the current directory) looking for a
    directory that holds both ``CMakeLists.txt`` and ``examples/``. ``PTM_ROOT``
    wins if it is set and points at such a directory; if it is set and does
    not, that is reported on stderr and the walk proceeds, because silently
    ignoring an explicit setting is how people lose an afternoon.
    """
    env_root = os.environ.get("PTM_ROOT", "").strip()
    if env_root:
        cand = Path(env_root).expanduser()
        if _looks_like_repo(cand):
            return cand.resolve()
        sys.stderr.write(
            f"[ptm] PTM_ROOT={env_root!r} is not a PTM3D checkout "
            f"(no CMakeLists.txt + examples/); ignoring it and searching upward\n"
        )

    here = Path(start).resolve() if start is not None else Path.cwd().resolve()
    if here.is_file():
        here = here.parent
    for cand in (here, *here.parents):
        if _looks_like_repo(cand):
            return cand
    return None


def build_dir_candidates(search_roots=()):
    """Ordered ``build/bin`` directories to probe, most specific first."""
    out = []
    env_build = os.environ.get("PTM_BUILD_DIR", "").strip()
    if env_build:
        cand = Path(env_build).expanduser()
        if cand.is_dir():
            out.append(cand)
        else:
            sys.stderr.write(
                f"[ptm] PTM_BUILD_DIR={env_build!r} is not a directory; ignoring it\n"
            )
    for root in search_roots:
        if root:
            out.append(Path(root) / "build" / "bin")
    return out


def find_engine_executable(name="ptm_thermal_sim", search_roots=(),
                           extra_candidates=()):
    """Return the absolute path of engine binary ``name``, or ``None``.

    ``search_roots`` are checkout roots whose ``build/bin`` is probed first, in
    order. ``extra_candidates`` are complete paths (absolute, or relative to
    the current directory) probed after those. ``PTM_BUILD_DIR`` precedes
    everything.
    """
    candidates = [d / name for d in build_dir_candidates(search_roots)]
    candidates.extend(Path(c) for c in extra_candidates)
    for path in candidates:
        resolved = Path(path).resolve()
        if resolved.exists():
            return resolved
    return None
