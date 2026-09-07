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
local list's ``note`` field, not in this file. (It happened a second time on
2026-09-06, to the docstrings added for the three checks below, and the gate
caught it before the commit. Every number written in this file is invented.)

**Three things a substring search over file contents cannot see**, each closed
below and each with its own arm in both directions:

1. **The file NAME.** A mesh or a view-factor matrix named after the site it
   came from publishes that site in a directory listing, and its bytes are
   innocent. Names are checked for EVERY tracked file, including the ones whose
   contents are never opened.
2. **The same coordinate written shorter.** A listed six-decimal latitude is not
   a substring of the same latitude written to three decimals -- which is how
   one survived a scrub for two days. The short forms are DERIVED from the list
   (:func:`short_form_patterns`), never hand-maintained beside it, and guarded
   at both ends against digits so a longer number that merely starts the same
   way is not a hit.
3. **A "binary" file that is text.** ``.tf`` and ``.tpc`` are SPICE *text*
   kernels -- frames and planetary constants, plain ASCII, exactly the files a
   body or site name is written into. They were on the never-open list; the
   binary siblings ``.bsp``/``.bpc`` still are.

Run directly::

    python3 src/python/tests/test_tracked_tree_hygiene.py
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
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
#:
#: ⛔ ``.tf`` and ``.tpc`` are NOT here, and were, which was wrong in the
#: direction that matters. In SPICE those are the FRAMES kernel and the
#: planetary-constants kernel, and both are **text** -- a `.tpc` is the file a
#: body's pole and prime meridian are written into, in plain ASCII, and a `.tf`
#: names frames after whatever the author called them. The binary siblings are
#: ``.bpc`` and ``.bsp`` (the leading `b`), which stay. Three `.tpc` files are
#: sitting untracked in this checkout right now; the moment one is committed the
#: check would have skipped exactly the file most likely to carry a site name.
#: The suffix list is a COST control, not a safety one: everything on it is
#: skipped, so a wrong entry is a silent hole.
BINARY_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".pdf", ".bsp", ".bpc",
    ".bin", ".fits", ".ply", ".vtk", ".obj", ".h5", ".hdf5", ".npy", ".npz",
    ".tif", ".tiff", ".zip", ".gz", ".ico", ".icns", ".woff", ".woff2",
}

#: Fewest decimals a derived SHORT form may carry. A listed coordinate is
#: normally six decimals; the same place written to three is the same place, and
#: a substring search for the six-decimal spelling cannot see it. Two decimals
#: would start colliding with ordinary numbers, which is not a hypothesis:
#: ``docs/figures/architecture.svg`` already carries a text-anchor coordinate
#: whose first three decimals match a listed latitude.
SHORT_FORM_MIN_DECIMALS = 3


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


_DECIMAL = re.compile(r"-?(\d+)\.(\d+)$")


def short_form_patterns(literals, min_decimals=SHORT_FORM_MIN_DECIMALS):
    """``[(reported term, compiled regex)]`` -- the SHORT forms of a coordinate.

    DERIVED from the list, never hand-written beside it. The hole this closes
    was found on 2026-09-05: a scrub genericized a site's name in a docstring
    filename and left its latitude at three decimals, and the deny list carried
    the six-decimal spelling, of which the three-decimal one is NOT a substring.
    The entry was then added by hand -- and hand-adding is the part that does
    not scale, because it has to be remembered once per coordinate and its
    absence is invisible.

    ⛔ The guard against arithmetic coincidence is a WORD BOUNDARY ON DIGITS,
    not an allow list. A matched short form must not be preceded by a digit or
    a point, and must not be FOLLOWED by a digit -- so it has to be the whole
    number, not the front of a longer one. That is what keeps
    ``docs/figures/architecture.svg`` clean: its two coincidental hits are the
    first six characters of a longer text-anchor coordinate, and a coordinate
    written short in a file name is not.

    ⛔ TRUNCATED **and** ROUNDED, at every length from three decimals up to one
    short of the literal's own. Truncation alone is the minority spelling:
    ``"%.3f" % value`` is how a number normally gets shortened, and it ROUNDS,
    so an invented ``11.222888`` printed to three decimals is ``11.223`` and a
    truncation-only rule looks straight past it. Both were measured against this
    tracked tree before being switched on: the rounded forms add 52 patterns and
    **zero** hits, so this costs no allow-list entries.

    The literal's OWN full spelling is deliberately not generated -- it is
    already matched as a literal, and re-deriving it would report every real hit
    twice.

    Signs are dropped and the magnitude is matched, so a listed ``-11.222333``
    is found written ``lat-11.222`` and written ``11.222S`` alike. A leading
    digit or point still blocks it, so ``911.222`` is not a hit. (Every number
    in this docstring is invented; see the module docstring for why there are no
    real ones anywhere in this file.)
    """
    out = []
    for term in literals:
        match = _DECIMAL.fullmatch(str(term).strip())
        if not match:
            continue
        whole, frac = match.group(1), match.group(2)
        if len(frac) <= min_decimals:
            continue                       # nothing shorter left to derive
        full = whole + "." + frac
        # `range` stops one short of the literal's own length, which is what
        # keeps the full spelling out: every entry here has fewer decimals than
        # `full`, so none of them can equal it. Rounding may carry into the
        # integer part (an invented 11.9996 to three decimals is 12.000); that
        # is still the same place written a different way, so it is kept.
        spellings = set()
        for places in range(min_decimals, len(frac)):
            spellings.add(whole + "." + frac[:places])            # truncated
            spellings.add(("%." + str(places) + "f") % float(full))  # rounded
        # sorted() for a DETERMINISTIC pattern, not for match order: `(?!\d)`
        # already makes the alternation unambiguous, but a set iterates in
        # hash order, and a failure message that named a different spelling on
        # each run would be untraceable.
        alternation = "|".join(re.escape(s) for s in
                               sorted(spellings, key=lambda s: (-len(s), s)))
        out.append((f"{term} (short form)",
                    re.compile(r"(?<![\d.])(?:" + alternation + r")(?!\d)")))
    return out


def scan(root=REPO_ROOT, literals=(), regexes=(), allow=()):
    """``[(path, line number, which term), ...]`` for every hit.

    Line number 0 means the hit is in the tracked file's NAME, not its contents.
    Names are checked first and for every tracked file, including the ones whose
    contents are never opened: a mesh, a raster or a view-factor matrix named
    after the site it came from publishes that site in the directory listing,
    and no amount of reading the bytes would have found it. It is also the way
    these come back most often -- somebody names a fixture after the run.

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
    compiled += short_form_patterns(literals)

    hits = []
    for name in tracked_files(root):
        if name in allow:
            continue
        low_name = name.lower()
        for term, needle in folded:
            if needle in low_name:
                hits.append((name, 0, term))
        for pattern, rx in compiled:
            if rx.search(name):
                hits.append((name, 0, pattern))

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


def _throwaway_repo(tmp):
    """An initialized git repository at ``tmp``, or a named skip."""
    setup = subprocess.run([GIT, "init", "-q", str(tmp)],
                           capture_output=True, text=True)
    if setup.returncode != 0:                             # pragma: no cover
        raise Skip(f"git init failed: {setup.stderr.strip()}")


def _add_all(tmp):
    """Stage everything in the throwaway repository, so `scan` can see it."""
    subprocess.run([GIT, "-C", str(tmp), "add", "-A"],
                   capture_output=True, text=True, check=True)


def test_no_sensitive_identifier_is_in_the_tracked_tree():
    """The boundary. One hit fails, and the message says where, not what."""
    literals, regexes, allow = load_deny_list()
    hits = scan(REPO_ROOT, literals, regexes, allow)
    if hits:
        shown = "\n".join(
            f"    {name}  (in the FILE NAME, matched: {term!r})" if number == 0
            else f"    {name}:{number}  (matched: {term!r})"
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
    literals, regexes, _allow = load_deny_list()
    term = literals[0] if literals else None
    if term is None:
        raise Skip("the list has no literal term to plant")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _throwaway_repo(tmp)
        planted = tmp / "note.md"
        planted.write_text(f"a comment mentioning {term} in passing\n")
        (tmp / "clean.md").write_text("nothing to see here\n")
        _add_all(tmp)

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


def test_a_file_NAME_carrying_a_term_is_found():
    """A mesh named after its site publishes the site in the file listing.

    Nothing about that hit is in the file's CONTENTS, and the file is a `.ply`,
    which this check never opens -- so a scanner that only reads bytes is blind
    to it by construction, not by accident. It is also the commonest way these
    come back: somebody names an artifact after the run it came from.
    """
    literals, regexes, _allow = load_deny_list()
    # ⛔ The term has to go into the file name UNCHANGED. Sanitizing it and then
    # asserting on the original is a fixture that fails on somebody else's list
    # -- the list is machine-local, and one containing a space, a slash or a
    # parenthesis would make this arm go red pointing at the production code
    # rather than at itself. Pick a term that is already a legal file-name
    # fragment, and say so by name when none is.
    usable = [t for t in literals
              if t and re.fullmatch(r"[A-Za-z0-9._-]+", str(t))]
    if not usable:
        raise Skip("no term in the list is a legal file-name fragment on its "
                   "own, so none can be planted in a name without altering it")
    term = usable[0]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _throwaway_repo(tmp)
        named = f"mesh_{term}_R3km.ply"
        (tmp / named).write_bytes(b"ply\nformat ascii 1.0\nend_header\n")
        (tmp / "mesh_generic_R3km.ply").write_bytes(b"ply\n")
        _add_all(tmp)

        hits = scan(tmp, literals, regexes, set())
        assert any(name == named and number == 0
                   for name, number, _t in hits), (
            "a tracked file NAMED after a forbidden term was not reported. Its "
            "contents are clean and its suffix is one this check never opens, "
            "so nothing else in this suite can see it")
        assert not any(name == "mesh_generic_R3km.ply" for name, _n, _t in hits), (
            "a generically named file of the same kind was reported; the name "
            "check is matching something other than the term")


def test_a_coordinate_written_SHORT_is_found_and_a_longer_number_is_not():
    """Three decimals of a six-decimal coordinate is the same place.

    ⛔ This is a MEASURED hole, not a hypothesis. On 2026-09-05 a scrub
    genericized a site's name in a docstring filename and left its latitude at
    three decimals; the deny list held the six-decimal spelling, of which the
    short one is not a substring, so the gate passed for two days over a
    coordinate that identifies the site to about a hundred meters.

    The other half is the reason this cannot simply be "match fewer digits": a
    longer number that merely STARTS with the same three decimals is a different
    number, and this repository already contains two of them (a text-anchor
    coordinate in ``docs/figures/architecture.svg``). Both directions are
    asserted here, from the list itself, so no coordinate is written into this
    file.
    """
    literals, regexes, _allow = load_deny_list()
    listed = {str(t).strip() for t in literals}

    def _shortenable(t):
        """A coordinate this arm can actually discriminate on.

        ⛔ It is not enough that the literal has more than three decimals: if
        the three-decimal spelling is ALSO in the list by hand -- as one is,
        added on 2026-09-05 -- the plain substring matcher already finds it and
        this arm would pass with the derived patterns removed. Pick one whose
        short form nothing else covers, so the arm can only pass on the new
        code.
        """
        match = _DECIMAL.fullmatch(str(t).strip())
        if not match or len(match.group(2)) <= SHORT_FORM_MIN_DECIMALS:
            return False
        short = match.group(1) + "." + match.group(2)[:SHORT_FORM_MIN_DECIMALS]
        return short not in listed and ("-" + short) not in listed

    sources = [t for t in literals if _shortenable(t)]
    if not sources:
        raise Skip("every multi-decimal coordinate in the list already carries "
                   "its own short form as a literal, so this arm could not "
                   "tell the derived patterns from the plain ones")
    term = sources[0]
    match = _DECIMAL.fullmatch(str(term).strip())
    whole, frac = match.group(1), match.group(2)
    short = whole + "." + frac[:SHORT_FORM_MIN_DECIMALS]
    # The same coordinate the way `"%.3f" %` would print it -- ROUNDED, which is
    # how a number actually gets shortened and is a different string from the
    # truncation whenever the fourth decimal is 5 or more.
    rounded = ("%." + str(SHORT_FORM_MIN_DECIMALS) + "f") % float(whole + "." + frac)
    # A DIFFERENT number that starts with the same three decimals -- the
    # arithmetic-coincidence shape. Digits chosen so it cannot be the literal.
    tail = "".join(str((int(c) + 5) % 10) for c in frac[SHORT_FORM_MIN_DECIMALS:])
    coincidence = short + (tail or "5")
    assert coincidence != str(term).strip().lstrip("-"), (
        "the negative control was built equal to the term itself")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _throwaway_repo(tmp)
        (tmp / "short.md").write_text(f"the site sits near {short}S in the map\n")
        (tmp / "rounded.md").write_text(f"printed with %.3f it is {rounded}\n")
        (tmp / "coincidence.svg").write_text(f'<text x="{coincidence}"/>\n')
        (tmp / ("site_" + short + "S_R3km.txt")).write_text("nothing here\n")
        _add_all(tmp)
        hits = scan(tmp, literals, regexes, set())

    assert any(name == "short.md" for name, _n, _t in hits), (
        f"a coordinate TRUNCATED to {SHORT_FORM_MIN_DECIMALS} decimals was not "
        f"found; the list carries the long spelling and the short one is not a "
        f"substring of it")
    assert any(name == "rounded.md" for name, _n, _t in hits), (
        f"a coordinate ROUNDED to {SHORT_FORM_MIN_DECIMALS} decimals was not "
        f"found. That is the spelling `%.3f` produces, so it is the commoner "
        f"one; a truncation-only rule looks straight past it")
    assert any(name.startswith("site_") and number == 0
               for name, number, _t in hits), (
        "the short form was not matched in a file NAME, which is where it "
        "actually appeared in the case that prompted this")
    # ⛔ Scoped to the SHORT-FORM terms on purpose. The plain-literal matcher is
    # an unbounded substring search and always has been, so a list that also
    # carries the three-decimal spelling as a literal of its own will match this
    # file through THAT path -- which is pre-existing behavior, not something
    # this change introduced, and not something to quietly relax here. What is
    # asserted is the new matcher's own boundary guard.
    assert not any(name == "coincidence.svg" and "short form" in str(term)
                   for name, _n, term in hits), (
        "a longer number that merely begins with the same three decimals was "
        "reported BY A DERIVED SHORT FORM. That is arithmetic coincidence, it "
        "is already present in this repository twice, and flagging it would "
        "make the check unusable")


def test_the_two_benign_svg_coincidences_still_pass():
    """The named regression: architecture.svg must stay clean.

    Its two hits are the real, measured false positives this guard was designed
    around -- the reason the short forms were NOT bulk-added to the list by hand
    on 2026-09-05. If they ever start failing, the digit-boundary guard has been
    weakened, and the answer is to fix the guard, not to allow-list the file:
    an allow entry would exempt that file from every OTHER term as well.
    """
    literals, regexes, allow = load_deny_list()
    target = "docs/figures/architecture.svg"
    if not (REPO_ROOT / target).is_file():
        raise Skip(f"{target} is not in this checkout")
    hits = [h for h in scan(REPO_ROOT, literals, regexes, allow)
            if h[0] == target]
    assert not hits, (
        f"{target} now reports {len(hits)} hit(s): "
        + ", ".join(f"line {n} ({t!r})" for _p, n, t in hits[:5])
        + ". Its numbers are longer coordinates that happen to share three "
          "decimals with a listed one; fix the boundary guard in "
          "short_form_patterns rather than allow-listing the file")


def test_a_spice_text_kernel_is_read():
    """``.tf`` and ``.tpc`` are TEXT. They were on the never-open list.

    A planetary-constants kernel is where a body's pole and prime meridian are
    written, in ASCII, and a frames kernel names frames after whatever the
    author called them -- both are exactly the kind of file a site name arrives
    in. The binary siblings are ``.bpc`` and ``.bsp``, and those stay skipped;
    the negative arm below is what keeps this from being a blanket "read
    everything" that would decode a 100 MB mesh.
    """
    literals, regexes, _allow = load_deny_list()
    term = literals[0] if literals else None
    if term is None:
        raise Skip("the list has no literal term to plant")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _throwaway_repo(tmp)
        for name in ("frames.tf", "constants.tpc"):
            (tmp / name).write_text(f"\\begindata\nCOMMENT = '{term}'\n")
        (tmp / "ephemeris.bsp").write_bytes(term.encode() + b"\x00\xff")
        (tmp / "orientation.bpc").write_bytes(term.encode() + b"\x00\xff")
        _add_all(tmp)
        hits = scan(tmp, literals, regexes, set())

    found = {name for name, number, _t in hits if number > 0}
    assert "frames.tf" in found and "constants.tpc" in found, (
        f"a term inside a SPICE text kernel was not read; found {sorted(found)}")
    assert "ephemeris.bsp" not in found and "orientation.bpc" not in found, (
        "a binary SPICE kernel was opened as text; the suffix list is a cost "
        "control and the binary siblings must stay on it")


def test_the_untracked_working_tree_is_not_scanned():
    """Scope, pinned: this is about what git carries, not what is on disk.

    A local overlay, a run directory or a scratch file sitting inside the
    checkout is exactly where this information is SUPPOSED to be able to live.
    A check that failed on those would push people to keep them somewhere less
    visible, which is worse than the problem.
    """
    literals, regexes, _allow = load_deny_list()
    term = literals[0] if literals else None
    if term is None:
        raise Skip("the list has no literal term to plant")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _throwaway_repo(tmp)
        (tmp / "tracked.md").write_text("clean\n")
        _add_all(tmp)
        (tmp / "untracked.md").write_text(f"{term}\n")
        assert not scan(tmp, literals, regexes, set()), (
            "an UNTRACKED file was reported; the boundary is the tracked tree")


CHECKS = [
    test_no_sensitive_identifier_is_in_the_tracked_tree,
    test_the_check_can_actually_fail,
    test_a_file_NAME_carrying_a_term_is_found,
    test_a_coordinate_written_SHORT_is_found_and_a_longer_number_is_not,
    test_the_two_benign_svg_coincidences_still_pass,
    test_a_spice_text_kernel_is_read,
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
