#!/usr/bin/env python3
"""The tracked tree must not carry the identifiers on the local deny list.

**What this is for.** Some of what this project works on is site information
that is not published with PTM3D: identifiers, coordinates, and region names
that were removed from the tracked tree and now live only in local files
outside the repository (``gui_core.curated_sites``, "The local overlay").
Removing them once is easy. Keeping them out is the part that needs a test,
because they come back the ordinary way -- somebody pastes a measurement into
a comment, or names a fixture after the run it came from, and every other check
passes.

**The list is not in this file, and that is the whole design.** A test that
carried the strings it forbids would publish them itself, in the file whose job
is keeping them unpublished, and would be the single most-read place to find
them. So the terms live in ``~/.ptm3d/sensitive_identifiers.json`` -- beside
``machine_profile.json`` and ``curated_sites_local.json``, which is where this
project already keeps facts about THIS computer.

With no list this check SKIPS, by name. That is honest rather than safe: on a
machine with no list there is nothing to compare against, and a silent pass
would read like a clean bill of health.

The file::

    {
      "schema": "ptm3d.sensitive_identifiers/1",
      "note": "...",
      "literals":   ["EXACT-STRING", ...],
      "regexes":    ["A03[A-Z]{2}[0-9]{3}", ...],
      "allow_paths": ["some/tracked/file.txt"]
    }

``literals`` are matched case-insensitively as substrings. ``regexes`` are
matched with ``re.IGNORECASE``. ``allow_paths`` are tracked files exempted with
a reason recorded in the list itself -- for a file whose numbers coincide with a
forbidden coordinate by arithmetic accident, which is real: a tracked ephemeris
table of state vectors has components whose digits contain a forbidden
latitude's digits, and a substring search finds them.

**This file is itself scanned.** It was written with a worked example of that
coincidence -- a literal state-vector component -- and the check failed on its
own docstring the moment the file became tracked, which is the correct outcome
and the reason there are no example values here now. Put worked examples in the
local list's ``note`` field, not in this file.

Run directly::

    python3 src/python/tests/test_tracked_tree_hygiene.py
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

#: The one git that works on this machine; ``/usr/bin/git`` is the xcrun shim
#: and exits 1 -- the same code it uses for "untracked", so a caller that only
#: looked at the exit status would read a broken git as a clean tree.
GIT = "/Library/Developer/CommandLineTools/usr/bin/git"

SCHEMA = "ptm3d.sensitive_identifiers/1"

#: Extensions never opened as text. A binary that happens to contain the bytes
#: of an identifier is not the failure this guards against, and decoding a
#: 100 MB mesh to find out costs more than the check is worth.
BINARY_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".pdf", ".bsp", ".bpc", ".tpc", ".tf",
    ".bin", ".fits", ".ply", ".vtk", ".obj", ".h5", ".hdf5", ".npy", ".npz",
    ".tif", ".tiff", ".zip", ".gz", ".ico", ".icns", ".woff", ".woff2",
}


class Skip(unittest.SkipTest):
    """A check that cannot run here, as distinct from one that failed.

    Derived from ``unittest.SkipTest`` so PYTEST reports it as a skip. Most
    suites in this project define a bare ``Exception`` subclass for this and
    pytest reports every one of them as FAILED -- which is how a run with
    nothing wrong reads as dozens of failures. This file does not add to that.
    """


def deny_list_path():
    """``~/.ptm3d/sensitive_identifiers.json``, or the env override."""
    override = os.environ.get("PTM3D_SENSITIVE_IDENTIFIERS", "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / ".ptm3d" / "sensitive_identifiers.json"


def load_deny_list():
    """``(literals, regexes, allow_paths)``, or a named skip."""
    target = deny_list_path()
    if not target.is_file():
        raise Skip(f"there is no local identifier list at {target}, so there "
                   f"is nothing to check the tracked tree against; see this "
                   f"file's docstring for its shape")
    try:
        payload = json.loads(target.read_text())
    except (OSError, ValueError) as exc:
        raise Skip(f"{target} could not be read as JSON ({exc})")
    if payload.get("schema") != SCHEMA:
        raise Skip(f"{target} declares schema {payload.get('schema')!r} and "
                   f"this check reads {SCHEMA!r}")
    literals = [str(x) for x in payload.get("literals", []) if str(x).strip()]
    regexes = [str(x) for x in payload.get("regexes", []) if str(x).strip()]
    allow = {str(x) for x in payload.get("allow_paths", [])}
    if not literals and not regexes:
        raise Skip(f"{target} lists no identifiers")
    return literals, regexes, allow


def tracked_files(root=REPO_ROOT):
    """Every tracked path, from git. Raises Skip when git cannot answer."""
    if not Path(GIT).exists():                            # pragma: no cover
        raise Skip(f"the working git binary is not at {GIT}")
    result = subprocess.run([GIT, "-C", str(root), "ls-files", "-z"],
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise Skip(f"git could not list the tracked files in {root}: "
                   f"{result.stderr.strip()}")
    return [name for name in result.stdout.split("\0") if name]


def scan(root=REPO_ROOT, literals=(), regexes=(), allow=()):
    """``[(path, line number, which term), ...]`` for every hit.

    Reports the TERM that matched and the file and line, and deliberately not
    the matching text: a failure message that quoted the line would put the
    identifier into the log, the terminal scrollback and whatever captured
    them, which is the leak this exists to prevent, arriving by a different
    road.
    """
    folded = [(term, term.lower()) for term in literals]
    compiled = []
    for pattern in regexes:
        try:
            compiled.append((pattern, re.compile(pattern, re.IGNORECASE)))
        except re.error as exc:
            # The list is hand-maintained. A pattern that does not compile has
            # to say so: dropping it silently would leave the check passing
            # while one of its terms was no longer being looked for.
            raise Skip(f"{deny_list_path()}: the pattern {pattern!r} does not "
                       f"compile ({exc}); this check cannot run until it does")
    hits = []
    for name in tracked_files(root):
        if name in allow:
            continue
        path = root / name
        if path.suffix.lower() in BINARY_SUFFIXES:
            continue
        try:
            text = path.read_text()
        except (OSError, UnicodeDecodeError):
            continue
        lowered = text.lower()
        if not (any(needle in lowered for _term, needle in folded)
                or any(rx.search(text) for _pattern, rx in compiled)):
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            low = line.lower()
            for term, needle in folded:
                if needle in low:
                    hits.append((name, number, term))
            for pattern, rx in compiled:
                if rx.search(line):
                    hits.append((name, number, pattern))
    return hits


def test_no_sensitive_identifier_is_in_the_tracked_tree():
    """The boundary. One hit fails, and the message says where, not what."""
    literals, regexes, allow = load_deny_list()
    hits = scan(REPO_ROOT, literals, regexes, allow)
    if hits:
        shown = "\n".join(f"    {name}:{number}  (matched: {term!r})"
                          for name, number, term in hits[:40])
        more = ("\n    ... and %d more" % (len(hits) - 40)) if len(hits) > 40 else ""
        raise AssertionError(
            f"{len(hits)} sensitive identifier(s) are in the TRACKED tree.\n"
            f"These terms are kept out of this repository on purpose; see "
            f"{deny_list_path()} for the list and the reason.\n"
            f"{shown}{more}\n"
            f"Remove them, or -- if a hit is arithmetic coincidence in a data "
            f"file -- add that path to 'allow_paths' in the list, with a note "
            f"saying why.")


def test_the_check_can_actually_fail():
    """A hygiene test that cannot fail is a hygiene test nobody should trust.

    Plants one of the forbidden terms in a file inside a throwaway git
    repository and confirms the scanner finds it. Without this arm, a scanner
    broken in any of the ordinary ways -- a git invocation that returns
    nothing, a suffix filter that swallows everything, a fold that never
    matches -- reports a clean tree forever and reads exactly like success.
    """
    import tempfile
    literals, regexes, _allow = load_deny_list()
    term = literals[0] if literals else None
    if term is None:
        raise Skip("the list has no literal term to plant")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        setup = subprocess.run([GIT, "init", "-q", str(tmp)],
                               capture_output=True, text=True)
        if setup.returncode != 0:                         # pragma: no cover
            raise Skip(f"git init failed: {setup.stderr.strip()}")
        planted = tmp / "note.md"
        planted.write_text(f"a comment mentioning {term} in passing\n")
        (tmp / "clean.md").write_text("nothing to see here\n")
        subprocess.run([GIT, "-C", str(tmp), "add", "-A"],
                       capture_output=True, text=True, check=True)

        hits = scan(tmp, literals, regexes, set())
        assert any(name == "note.md" for name, _n, _t in hits), (
            "the scanner did not find a term planted in a tracked file; it "
            "cannot fail, so its passing says nothing")
        assert not any(name == "clean.md" for name, _n, _t in hits), (
            "the scanner flagged a file with none of the terms in it")

        # And the allow list has to actually exempt.
        exempted = scan(tmp, literals, regexes, {"note.md"})
        assert not any(name == "note.md" for name, _n, _t in exempted), (
            "allow_paths did not exempt the file it names")


def test_the_untracked_working_tree_is_not_scanned():
    """Scope, pinned: this is about what git carries, not what is on disk.

    A local overlay, a run directory or a scratch file sitting inside the
    checkout is exactly where this information is SUPPOSED to be able to live.
    A check that failed on those would push people to keep them somewhere less
    visible, which is worse than the problem.
    """
    import tempfile
    literals, regexes, _allow = load_deny_list()
    term = literals[0] if literals else None
    if term is None:
        raise Skip("the list has no literal term to plant")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        setup = subprocess.run([GIT, "init", "-q", str(tmp)],
                               capture_output=True, text=True)
        if setup.returncode != 0:                         # pragma: no cover
            raise Skip(f"git init failed: {setup.stderr.strip()}")
        (tmp / "tracked.md").write_text("clean\n")
        subprocess.run([GIT, "-C", str(tmp), "add", "-A"],
                       capture_output=True, text=True, check=True)
        (tmp / "untracked.md").write_text(f"{term}\n")
        assert not scan(tmp, literals, regexes, set()), (
            "an UNTRACKED file was reported; the boundary is the tracked tree")


CHECKS = [
    test_no_sensitive_identifier_is_in_the_tracked_tree,
    test_the_check_can_actually_fail,
    test_the_untracked_working_tree_is_not_scanned,
]


def main():
    failed = skipped = 0
    for check in CHECKS:
        try:
            check()
        except Skip as exc:
            skipped += 1
            print(f"SKIP {check.__name__}: {exc}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {check.__name__}:\n  {exc}")
        else:
            print(f"PASS {check.__name__}")
    total = len(CHECKS)
    print(f"\n[test_tracked_tree_hygiene] {total - failed - skipped}/{total} "
          f"passed, {skipped} skipped, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
