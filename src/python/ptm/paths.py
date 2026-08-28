"""Locating the repository and the engine executables.

Every tutorial notebook and most scripts in the tree re-derive these two
things. The rules here are the ones already in use, collected in one place:

* a PTM3D checkout is a directory containing ``CMakeLists.txt`` and
  ``examples/`` (the test the tutorial notebooks use);
* the engine binaries live in ``<root>/build/bin/``.

Nothing here guesses silently. ``PTM_BUILD_DIR`` and ``PTM_ROOT`` override the
search, and a value that does not resolve produces a warning on stderr rather
than a quiet fall-through to some other candidate.

:func:`preflight` answers the other question every consumer re-derives: is
everything this run needs actually on disk, and if not, what is the command
that gets it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

__all__ = [
    "ENGINE_BINARIES",
    "Check",
    "MissingPrerequisites",
    "Report",
    "build_dir_candidates",
    "find_engine_executable",
    "find_repo_root",
    "preflight",
    "remedy_for",
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


# --- preflight ----------------------------------------------------------------
# "Is the binary there, is the config there, are the kernels there, and here is
# the command that fixes it" is written out longhand in every tutorial notebook
# and most orchestration scripts. The part worth sharing is not the existence
# test -- it is the remedy: which of the several fetch commands gets THIS file,
# spelled exactly as it must be typed.
#
# This is the small, run-anything version. gui/gui_core/preflight.py is the
# large, launch-time one: it takes a whole config and checks dataless cloud
# files, view-factor face counts and free disk. Different question, kept
# separate on purpose.

_BUILD_REMEDY = (
    "build it: cmake -S . -B build -DCMAKE_BUILD_TYPE=Release "
    "-DPTM_ENABLE_GPU=OFF -DPTM_ARCH_NATIVE=OFF -DCSPICE_ROOT=/path/to/cspice "
    "&& cmake --build build --target ptm_thermal_sim -j "
    "(INSTALL.md), or point $PTM_BUILD_DIR at an existing build"
)

#: SPICE kernel subtrees that only the Galilean-system runs need. Everything
#: else under ``data/kernels`` comes with the Moon/Mercury core set, except the
#: Bennu entries below.
_EUROPA_KERNELS = ("spk/jup365.bsp", "europa.tm")

#: The same, for (101955) Bennu. ``--bodies moon`` does NOT fetch these, so a
#: remedy that named it would send the reader back for a second failure on the
#: same missing file.
_BENNU_KERNELS = ("spk/sb-101955-118_long.bsp", "bennu.tm")


_MANIFEST_CACHE = {}


def _dataset_remedies(root):
    """``{repo-relative directory: dataset name}`` from ``data_manifest.json``.

    Read from the manifest rather than hard-coded, so a new dataset gets a
    correct remedy without anyone remembering to update a table here. Cached on
    the manifest's identity, because a preflight of a dozen paths would
    otherwise re-read and re-parse the same file a dozen times.
    """
    manifest = Path(root) / "data" / "data_manifest.json" if root else None
    if manifest is None or not manifest.is_file():
        return {}
    try:
        stat = manifest.stat()
        key = (str(manifest), stat.st_mtime_ns, stat.st_size)
    except OSError:
        return {}
    if key in _MANIFEST_CACHE:
        return _MANIFEST_CACHE[key]

    out = {}
    try:
        import json
        datasets = json.loads(manifest.read_text()).get("datasets", {})
    except (OSError, ValueError):
        return {}
    for name, spec in datasets.items():
        for entry in spec.get("files", []):
            target = (entry.get("unpack_to") or "").strip("/")
            if target:
                out[target] = name
    _MANIFEST_CACHE[key] = out
    return out


def remedy_for(path, root=None):
    """The command that would produce ``path``, or ``None``.

    ``path`` may be absolute or repository-relative. The answer is derived from
    ``data/data_manifest.json`` for the fetchable datasets, from the kernel
    layout for SPICE, and from ``INSTALL.md`` for the engine binaries.
    """
    path = Path(path)
    if root is None:
        root = find_repo_root(path if path.is_absolute() else None)
    try:
        relative = path.resolve().relative_to(Path(root).resolve()).as_posix() \
            if root else path.as_posix()
    except (ValueError, OSError):
        relative = path.as_posix()

    if relative.startswith("build/bin/") or path.name in ENGINE_BINARIES:
        return _BUILD_REMEDY
    if relative.startswith("data/kernels"):
        tail = relative[len("data/kernels/"):]

        def _matches(names):
            return any(tail.startswith(k) or tail == k for k in names)

        if _matches(_EUROPA_KERNELS):
            bodies = "europa"
        elif _matches(_BENNU_KERNELS):
            bodies = "bennu"
        else:
            bodies = "moon"
        return f"python3 scripts/fetch_kernels.py --bodies {bodies}"
    for directory, dataset in sorted(_dataset_remedies(root).items(),
                                     key=lambda kv: -len(kv[0])):
        if relative == directory.rstrip("/") or relative.startswith(directory):
            return f"python3 scripts/fetch_data.py --dataset {dataset}"
    if relative.startswith("examples/") or relative.startswith("configs/"):
        return "this file ships with the repository; the checkout is incomplete"
    return None


class Check:
    """One prerequisite and whether it is there.

    ``path`` is what was looked for, ``what`` how to name it to a human,
    ``remedy`` the command that would produce it (``None`` when nothing in the
    repository knows how).
    """

    __slots__ = ("path", "what", "remedy", "found")

    def __init__(self, path, what, remedy, found):
        self.path = path
        self.what = what
        self.remedy = remedy
        self.found = bool(found)

    def __repr__(self):
        return f"<Check {'found' if self.found else 'MISSING'} {self.what}>"

    def __str__(self):
        return ("found    " if self.found else "MISSING  ") + self.what


class MissingPrerequisites(FileNotFoundError):
    """Raised by :meth:`Report.raise_if_missing`. A ``FileNotFoundError``, so
    existing ``except FileNotFoundError`` handlers keep working."""


class Report:
    """The result of :func:`preflight`.

    ``print(report)`` gives one line per prerequisite;
    :meth:`raise_if_missing` raises with the remedies for the missing ones.
    :attr:`engine` is the resolved engine path when one was requested.
    """

    def __init__(self, checks, root=None, engine=None):
        self.checks = list(checks)
        self.root = root
        self.engine = engine

    def __repr__(self):
        return (f"<Report {len(self.found)}/{len(self.checks)} found"
                f"{'' if self.ok else ', ' + str(len(self.missing)) + ' MISSING'}>")

    @property
    def ok(self):
        return all(c.found for c in self.checks)

    @property
    def missing(self):
        return [c for c in self.checks if not c.found]

    @property
    def found(self):
        return [c for c in self.checks if c.found]

    def __str__(self):
        lines = [str(c) for c in self.checks]
        if self.root is not None:
            lines.append(f"\nRepo root: {self.root}")
        if self.engine is not None:
            lines.append(f"Engine:    {self.engine}")
        return "\n".join(lines)

    def raise_if_missing(self):
        """Raise :class:`MissingPrerequisites` if anything is absent.

        The message names each missing item and the command that produces it,
        which is the whole point: a bare ``FileNotFoundError`` on
        ``jup365.bsp`` sends the reader to a search engine.
        """
        if self.ok:
            return self
        detail = []
        for check in self.missing:
            remedy = check.remedy or "no automated fetch exists for this one"
            detail.append(f"  {check.what}\n      -> {remedy}")
        raise MissingPrerequisites(
            "This run is not ready. Missing:\n" + "\n".join(detail))


def preflight(items=(), root=None, engine=None, start=None):
    """Check that everything a run needs is present, with the fix for what is not.

    ::

        from ptm import paths
        pre = paths.preflight(
            [CONFIG, "data/kernels/lsk/naif0012.tls"],
            engine="ptm_thermal_sim")
        print(pre)
        pre.raise_if_missing()
        binary = pre.engine

    ``items`` are paths -- absolute, or relative to ``root`` -- or
    ``(path, what)`` / ``(path, what, remedy)`` tuples when the default naming
    or the inferred remedy is not what you want. ``engine`` names an engine
    binary to locate through :func:`find_engine_executable`; the result is
    :attr:`Report.engine`.

    ``root`` defaults to the checkout found from ``start`` (or the current
    directory). Nothing is fetched, nothing is built and nothing raises here:
    the report is data, and :meth:`Report.raise_if_missing` is the decision.
    """
    if root is None:
        root = find_repo_root(start)
    root = Path(root).expanduser().resolve() if root is not None else None

    checks = []
    engine_path = None
    if engine:
        engine_path = find_engine_executable(
            engine, search_roots=[root] if root else ())
        checks.append(Check(
            path=engine_path,
            what=(str(engine_path) if engine_path
                  else f"build/bin/{engine} (the PTM3D engine)"),
            remedy=_BUILD_REMEDY,
            found=engine_path is not None))

    for item in items:
        if isinstance(item, (tuple, list)):
            path, what = item[0], (item[1] if len(item) > 1 else None)
            remedy = item[2] if len(item) > 2 else None
        else:
            path, what, remedy = item, None, None
        path = Path(path).expanduser()
        if not path.is_absolute() and root is not None:
            path = root / path
        if what is None:
            try:
                what = str(path.relative_to(root)) if root else str(path)
            except ValueError:
                what = str(path)
        if remedy is None:
            remedy = remedy_for(path, root)
        checks.append(Check(path=path, what=what, remedy=remedy,
                            found=path.exists()))

    return Report(checks, root=root, engine=engine_path)
