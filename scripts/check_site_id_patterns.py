#!/usr/bin/env python3
"""Refuse the SHAPE of a landing-site identifier in the tracked tree.

PTM3D works on lunar landing-site candidates whose identifiers, coordinates and
region names are not published with the model. They are kept out of the tracked
tree, and ``src/python/tests/test_tracked_tree_hygiene.py`` is the check that
keeps them out -- against a deny list of the actual terms, which deliberately
lives OUTSIDE the repository (``~/.ptm3d/sensitive_identifiers.json``).

That design has one hole, and it is not hypothetical. **A machine with no local
list cannot run that check**, so the check SKIPS -- honestly, by name, but it
skips. Continuous-integration runners have no local list and never will, so the
one gate that would have caught the leak does not run in the place where merges
are decided. A pull request reintroduced 61 site-identifier hits across three
files in September 2026 and merged green for exactly this reason.

This file is the half of the check that CAN run anywhere, because it carries no
identifiers at all -- only their *shape*:

* the catalog identifier form: the three-character catalog prefix, one to three
  letters of region code, two to four digits;
* the site-filename form these meshes and matrices are named with: an
  underscore, a high-southern-latitude degree value to three decimals, ``S``,
  an underscore.

Neither pattern is or contains a real identifier, a real coordinate or a real
place name, so both are safe in a tracked file -- which is the entire reason
this check exists as a separate file rather than as more entries on the local
list. It is strictly WEAKER than the deny-list check: it cannot see a bare
region code used as a Python identifier, an unprefixed site number, or a
northern-hemisphere coordinate. It is not a replacement. It is the part that
runs in CI, and it is sized to catch the way these identifiers actually come
back -- pasted in whole, as a path or a fixture name.

Run the local deny-list check too, whenever the machine has the list::

    python3 src/python/tests/test_tracked_tree_hygiene.py

Usage::

    python3 scripts/check_site_id_patterns.py            # whole tracked tree
    python3 scripts/check_site_id_patterns.py PATH ...   # specific paths

Exits 0 clean, 1 on a hit, and 2 on a usage error OR when the tracked file list
cannot be obtained. That last one matters: a checker that reports "clean"
because git failed is a checker that has learned to say yes, and this one
replaced a gate that was already failing by not running.

``scripts/test_check_site_id_patterns.py`` pins all of that, including the exit
codes and the fact that this file's own patterns do not match this file. Run it
after touching this one.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

# --- the patterns -----------------------------------------------------------
# Shapes only. Nothing here names a site, a coordinate or a region: read this
# table aloud and it tells you the FORM of the identifiers this project keeps
# private, not one of them. Anything that would name one belongs on the local
# deny list instead, which is not in this repository.
#
# Kept deliberately narrow. A pattern that also matched ordinary text would be
# switched off by the first person it inconvenienced, and a gate that gets
# switched off is worse than no gate, because the switching-off is quiet.
PATTERNS = [
    (
        r"A03[A-Z]{1,3}[0-9]{2,4}",
        "a landing-site catalog identifier (prefix + region code + number)",
    ),
    (
        r"_8[0-9]\.[0-9]{3}S_",
        "a site-mesh filename's high-southern-latitude coordinate field",
    ),
]

#: Files exempt by path, each with the reason it is exempt. It is EMPTY, and
#: keeping it empty is the goal: every entry is a hole in the gate.
#:
#: In particular this gate does not exempt itself, and does not need to. A
#: regex for the shape does not have the shape -- ``A03`` followed by ``[`` is
#: not ``A03`` followed by a letter -- so the pattern table above does not match
#: the pattern table above, and the self-test pins that rather than leaving it
#: to luck. The self-test likewise plants its shaped strings by assembling them
#: at run time, so it carries no literal either. Both files are scanned like
#: any other, which is what makes this check trustworthy on itself.
ALLOW_PATHS: dict[str, str] = {}

#: Never opened as text. A binary that happens to contain these bytes is not
#: the failure this guards against.
#:
#: This affects CONTENTS only -- the PATH of a binary is still checked, and has
#: to be, because a mesh or matrix named after its site is the single most
#: likely way one of these identifiers gets committed. ``.ply`` is in this set
#: and a site-named ``.ply`` is exactly the case.
#:
#: ``.tf`` and ``.tpc`` are deliberately ABSENT although the equivalent table in
#: ``src/python/tests/test_tracked_tree_hygiene.py`` has them: SPICE text
#: kernels are text (only ``.bsp``/``.bpc`` are binary), and a frame kernel
#: defining a site-fixed frame is a plausible place for one of these to land.
BINARY_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".pdf", ".bsp", ".bpc",
    ".bin", ".fits", ".ply", ".vtk", ".obj", ".h5", ".hdf5", ".npy", ".npz",
    ".tif", ".tiff", ".zip", ".gz", ".ico", ".icns", ".woff", ".woff2",
}

#: End-of-line escape hatch, for the case this cannot anticipate: a genuine
#: coincidence in text that must stay. It has to be visible on the line, so a
#: reviewer sees the claim being made. There is deliberately no equivalent for
#: a PATH: a file whose NAME carries one of these shapes has to be renamed.
INLINE_ALLOW = "site-id-shape: allow"

#: ``re.IGNORECASE`` because the deny list this stands in for matches
#: case-insensitively (``test_tracked_tree_hygiene.py``), and the two halves
#: disagreeing on case would leave the gap open on exactly the axis where this
#: half is the only one that runs: a lowercased filename, a URL slug or a
#: ``str.lower()``-ed fixture key would ship past CI and be caught only on a
#: machine that has the local list.
COMPILED = [(re.compile(p, re.IGNORECASE), why) for p, why in PATTERNS]


class GitUnavailable(RuntimeError):
    """The tracked file list could not be obtained. Never a pass."""


def tracked_files(root="."):
    """Every tracked path, via whichever git works here.

    The CommandLineTools path first because on this project's Mac Pro
    ``/usr/bin/git`` is the xcrun shim, which exits 1 -- the same code it uses
    for "not a repository" -- so a caller that trusted the first git on PATH
    would read a broken git as an empty tree. On a CI runner that path does not
    exist and plain ``git`` is correct.

    Raises rather than returning an empty list: "no files" and "could not ask"
    are the same value and opposite meanings.
    """
    override = os.environ.get("PTM3D_GIT", "").strip()
    candidates = [override] if override else [
        "/Library/Developer/CommandLineTools/usr/bin/git", "git"]
    problems = []
    for git in candidates:
        try:
            out = subprocess.run([git, "-C", root, "ls-files", "-z"],
                                 capture_output=True, check=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            problems.append(f"{git}: {exc}")
            continue
        names = [n for n in out.stdout.decode("utf-8", "replace").split("\0")
                 if n]
        if not names:
            problems.append(f"{git}: listed no tracked files")
            continue
        return names
    raise GitUnavailable(
        "could not list the tracked files, so this check has nothing to scan "
        "and will NOT report clean: " + "; ".join(problems))


#: ``check()`` uses line number 0 for a hit in the file's own PATH, which has
#: no line. Named rather than written as a bare 0 at three call sites.
PATH_LINE = 0


class Unreadable(RuntimeError):
    """A path was named but could not be read. Never a pass."""


def check(path, root="."):
    """``[(line number, why), ...]`` for one file. Line 0 means the PATH.

    Reports the line and what SHAPE matched -- never the matching text. A gate
    against publishing an identifier must not print the identifier into a CI
    log, which is public to everyone who can read the build.

    **The path is checked first, and always.** A file NAMED after a site is the
    likeliest way one of these identifiers is committed -- a mesh, a matrix, a
    per-site config -- and the contents of such a file are usually innocent
    numbers. A first draft of this checker scanned contents only: a tracked
    mesh whose filename carried both a catalog identifier and a coordinate
    field reported ``clean``, doubly so because ``.ply`` skips the read. The
    path has no line to point at, so it is reported as line ``PATH_LINE`` (0).

    Raises ``Unreadable`` rather than returning ``[]`` when a named path cannot
    be opened. "I found nothing" and "I could not look" are the same value and
    opposite meanings, and this file exists because a check that could not run
    was mistaken for one that passed.
    """
    hits = []
    normalized = path.replace(os.sep, "/")
    for rx, why in COMPILED:
        if rx.search(normalized):
            hits.append((PATH_LINE, f"{why}, in the file's own NAME"))
            break

    full = os.path.join(root, path) if root != "." else path
    if os.path.splitext(path)[1].lower() in BINARY_SUFFIXES:
        return hits
    try:
        with open(full, "r", encoding="utf-8") as handle:
            text = handle.read()
    except UnicodeDecodeError:
        # Genuinely not text after all. The path was still checked above.
        return hits
    except OSError as exc:
        raise Unreadable(f"{path} could not be read ({exc})") from exc
    for number, line in enumerate(text.splitlines(), start=1):
        if INLINE_ALLOW in line:
            continue
        for rx, why in COMPILED:
            if rx.search(line):
                # One report per line. A site-shaped path on a line matches
                # BOTH patterns, and counting it twice made the summary say
                # "2 lines in 1 file" about a single line.
                hits.append((number, why))
                break
    return hits


def scan(paths, root="."):
    """``[(path, line number, why), ...]`` over the given tracked paths."""
    hits = []
    for path in sorted(set(paths)):
        if path in ALLOW_PATHS:
            continue
        for number, why in check(path, root):
            hits.append((path, number, why))
    return hits


def _parser():
    p = argparse.ArgumentParser(
        prog="check_site_id_patterns.py",
        description="Refuse landing-site identifier SHAPES in the tracked tree.")
    p.add_argument("paths", nargs="*", metavar="PATH",
                   help="specific tracked paths; default is the whole tracked tree")
    p.add_argument("--root", default=".",
                   help="repository root to scan (default: the current directory)")
    return p


def main(argv):
    # STRICT, for the same reason check_american_english.py is: a hand-rolled
    # loop that files an unrecognized flag under "a path" reports clean for
    # arguments it never understood.
    args = _parser().parse_intermixed_args(argv[1:])
    try:
        paths = args.paths or tracked_files(args.root)
    except GitUnavailable as exc:
        print(f"check_site_id_patterns: {exc}", file=sys.stderr)
        return 2
    try:
        hits = scan(paths, args.root)
    except Unreadable as exc:
        print(f"check_site_id_patterns: {exc}; refusing to report clean on a "
              f"file this check could not open", file=sys.stderr)
        return 2
    for path, number, why in hits:
        where = f"{path}:{number}" if number != PATH_LINE else f"{path} (name)"
        print(f"{where}: matches {why}")
    if hits:
        files = len({p for p, _n, _w in hits})
        print(f"\n{len(hits)} line(s) in {files} file(s) carry the SHAPE of a "
              f"landing-site identifier.\n"
              f"This project does not publish those. Remove them, or -- for a "
              f"genuine coincidence on a LINE -- put '{INLINE_ALLOW}' on it. A "
              f"hit in a file's NAME has no escape hatch: rename the file.\n"
              f"For test fixtures, invent a tag that does not match these "
              f"shapes; for real local paths, register them in "
              f"~/.ptm3d/test_fixtures.json and look them up through "
              f"src/python/tests/local_fixtures.py.\n"
              f"This is the shape-only half of the gate. On a machine with the "
              f"local identifier list, also run "
              f"src/python/tests/test_tracked_tree_hygiene.py.")
        return 1
    print(f"check_site_id_patterns: clean ({len(paths)} tracked files)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
