"""Fetching an asteroid's ephemeris, shape and physical properties on demand.

Three fetches, one shape. Each of them:

* **is cache-first.** The local ``data/horizons_cache`` is authoritative and
  works offline; the network is asked only when the cache misses, or when the
  caller explicitly says to refetch.
* **asks the curated registry before anything else.** For a body in
  :mod:`gui_core.curated_sources` -- today (101955) Bennu -- neither HORIZONS
  nor DAMIT can serve what is wanted, and both say so correctly. The order is::

      curated registry (itself cache-first) -> body cache -> live service

  and for a curated body neither of the last two is reached. The registry
  comes BEFORE the body cache deliberately, not as a shortcut: Bennu's body
  cache holds ``ephemeris.spk``, JPL solution 76, which covers 2015-2023 and
  was superseded on 2026-08-27. Inside that span it would pass an ordinary
  coverage check and be wired silently. The registry's own first step is
  still a cache check -- "is the pinned kernel already fetched into
  ``data/kernels``?" -- so a machine that has it reaches a wired card with no
  network call at all. See :func:`fetch_spk` and :func:`fetch_damit_shape`.
* **never runs by itself.** Every entry point here is called from a button.
  Nothing on this card reaches the network because a field changed.
* **never hangs.** Every client is constructed with a timeout, and the timeout
  is short enough that a dead network is a message rather than a frozen window.
* **says which kind of failure it was.** "Not found", "offline", "timed out",
  "the service answered but has nothing for this body" and "the file arrived
  but does not cover your run" are five different answers, and reporting any
  of them as another is how a run gets trusted that should not be.
* **returns what it wrote AND where each value came from.** Provenance is
  per field, so the card can label "SBDB" beside a diameter and "DAMIT model
  9110" beside a pole and the user can see which numbers are somebody's fit.

No Qt, so the whole layer is measurable without a window; the card runs these
on a worker thread.

Two things this module refuses to do quietly
--------------------------------------------

**An SPK that does not span the run is not accepted.** ``fetch_spk`` reads the
coverage back out of the file it just downloaded and compares it with the run
window INCLUDING the equilibration reach-back. If it does not span it, the
result is a refusal and ``generic_body.ephemeris_source`` is NOT set to
``spice``. That is the committed Justitia SPK covered 30 days
of 2034 while every Justitia configuration targeted 2035, and all fifteen ran
100 percent on the engine's per-epoch Keplerian fallback -- which logs nothing
at all (``illumination_model.cpp:2793-2818``), so every one of them "passed".
The engine still does not report it. Until it does, this is the guard.

**A kernel the repository SHIPS is not overwritten.** ``data/horizons_cache``
is not scratch space: four ``ephemeris.spk`` and three ``metakernel.tm`` files
in it are tracked, and the fourteen shipped Justitia configurations resolve
their ephemeris through two of them. A fetch used to install its download over
whichever file was at the destination and regenerate the metakernel beside it,
with no check of either -- and on 2026-08-31 that replaced Justitia's committed
899-day kernel with a 762-day one spanning NONE of the campaign windows, and
rewrote two committed metakernels to machine-absolute ``PATH_VALUES`` that each
file's own header forbids. Worse, the
metakernel half needed no download at all: a cache HIT rewrote them too.
:func:`tracked_by_git` now decides, and the default is to save the new kernel
BESIDE the tracked one and point the configuration at the copy. See
:func:`fetch_spk`'s ``on_tracked`` and :func:`_ensure_metakernel`.

**A replacement that loses coverage is refused even when it is asked for.**
Consent to replace a tracked kernel is not consent to install a shorter one:
the old and new coverages are read and compared BEFORE anything is written, and
a new kernel that does not contain the old one's span is refused with both
spans named. That comparison is what the 899 -> 762 day clobber would have
failed.

**A shape model is not fetched without saying where it will live.** ``.obj``
is matched by this project's ``.gitignore`` line 14 -- a rule written for C
object files that swallows every asteroid shape model in the repository
(it is why a Justitia shape went missing). A fetched shape is
therefore written with a ``shape_provenance.json`` sidecar carrying its md5,
its DAMIT model id, its source URL and its vertex and face counts. The sidecar
is NOT ignored, so the shape's identity survives even when the shape does not,
and the card says so in as many words.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = [
    "DAMIT_TIMEOUT_S",
    "HORIZONS_SPK_TIMEOUT_S",
    "ON_TRACKED_CANCEL",
    "ON_TRACKED_CHOICES",
    "ON_TRACKED_REPLACE",
    "ON_TRACKED_SAVE_BESIDE",
    "SPK_MARGIN_DAYS",
    "FetchResult",
    "body_cache_dir",
    "fetch_curated_shape",
    "fetch_damit_shape",
    "fetch_spk",
    "fetch_sbdb_properties",
    "shape_provenance",
    "spin_consistency",
    "tracked_by_git",
]

#: An SPK download builds a kernel server-side and can take appreciably longer
#: than a parameter lookup, so it gets its own, longer budget -- but it is
#: still a budget.
HORIZONS_SPK_TIMEOUT_S = 90

#: DAMIT is a university-hosted service and is scraped rather than queried
#: through a stable API, so it is given room and still bounded.
DAMIT_TIMEOUT_S = 45

#: How far beyond the run window a fetched kernel is asked to reach, at each
#: end. Generous ON PURPOSE. An SPK is a few tens of kilobytes whether it spans
#: a month or a decade, and the failure it prevents is silent: the committed
#: Justitia kernel covered 2034-10-12 to 2034-11-11 while the configurations
#: targeted 2035-08-24 onward, which no output distinguishes from a good run.
#: Paying a kilobyte to remove that class of failure is not a trade-off.
SPK_MARGIN_DAYS = 365.0


class FetchResult:
    """What a fetch did, what it wrote, and where each value came from.

    ``ok`` is the only thing a caller should branch on, and it is False for a
    refusal as well as for a failure -- a kernel that arrived but does not
    cover the run is not a success with a caveat.
    """

    def __init__(self, ok=False, kind="", values=None, provenance=None,
                 message="", warnings=(), paths=None, refusal="",
                 from_cache=False, offer="", offer_bytes=0,
                 tracked_destination=""):
        self.ok = bool(ok)
        #: "spk", "damit_shape" or "sbdb".
        self.kind = kind
        #: ``{dotted config key: value}``, for the caller to apply one at a
        #: time. Deliberately not applied here: the caller reports each one,
        #: and every field stays the user's to edit.
        self.values = dict(values or {})
        #: ``{dotted config key: short source label}``.
        self.provenance = dict(provenance or {})
        #: Always non-empty.
        self.message = message
        self.warnings = tuple(warnings)
        #: ``{role: absolute path}`` for anything written to disk.
        self.paths = dict(paths or {})
        #: Non-empty when the fetch SUCCEEDED technically but the result must
        #: not be used -- today, only out-of-coverage.
        self.refusal = refusal
        #: True when nothing was fetched because the cache already had it.
        self.from_cache = bool(from_cache)
        #: Non-empty when this is not a failure but a QUESTION: the fetch can
        #: be completed, and it costs a download the user has not agreed to.
        #: A FLAG rather than a phrase in ``message``, because the caller has
        #: to branch on it and a caller that branched by searching the message
        #: for words would break the first time the wording improved.
        self.offer = offer
        #: What agreeing to ``offer`` would transfer, in bytes. Disclosed
        #: before the question, not after it.
        self.offer_bytes = int(offer_bytes)
        #: The tracked file this fetch declined to overwrite, or ``""``. A
        #: FLAG, for the same reason ``offer`` is one: the card offers the user
        #: the Replace it did not take, and a card that decided by searching
        #: ``message`` for words would break the first time the wording
        #: improved.
        self.tracked_destination = str(tracked_destination or "")

    def lines(self):
        """The block the card shows: message, refusal, then warnings."""
        out = [self.message]
        if self.refusal:
            out.append(self.refusal)
        out.extend(self.warnings)
        return [line for line in out if line]

    def __repr__(self):                                   # pragma: no cover
        return (f"<FetchResult {self.kind} ok={self.ok}"
                f"{' REFUSED' if self.refusal else ''}>")


# ---------------------------------------------------------------------------
# Where things go
# ---------------------------------------------------------------------------

def body_cache_dir(project_root, name):
    """This body's directory under ``data/horizons_cache/bodies``.

    The same rule ``DataManager._body_dir`` uses -- space and slash to
    underscore -- so a body fetched from the card and the same body fetched
    from the HORIZONS page land in ONE directory rather than two spellings of
    one. It is not created here; whoever writes creates it.

    That rule is not enough on its own. The body name is free text: it is
    whatever the user typed, or whatever HORIZONS returned, and it is used to
    build a path that files are WRITTEN to. ``..`` survives the two
    replacements intact, so a name containing it would place a cache directory
    outside the cache. Rather than quietly rewriting the name into something
    else -- which would put the body's files somewhere the user was never told
    about -- the result is checked against the cache root and refused.
    """
    from gui_core.body_lookup import CACHE_RELPATH
    safe = str(name or "").strip().replace(" ", "_").replace("/", "_")
    root = (Path(project_root) / CACHE_RELPATH / "bodies").resolve()
    candidate = (root / safe).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(
            f"{name!r} does not name a directory inside the body cache "
            f"({candidate} is outside {root}). Rename the body without path "
            f"separators or '..' before fetching anything for it.")
    return root / safe


# ---------------------------------------------------------------------------
# Files the repository ships
# ---------------------------------------------------------------------------

#: What a fetch does when its destination is a file the repository TRACKS.
#: These are the three answers the card offers, and the string a caller passes
#: back as ``on_tracked``; anything else is refused by name rather than
#: silently treated as the default.
ON_TRACKED_SAVE_BESIDE = "save_beside"     #: keep the tracked file, install a copy beside it
ON_TRACKED_REPLACE = "replace"             #: overwrite it -- still coverage-checked
ON_TRACKED_CANCEL = "cancel"               #: change nothing at all
ON_TRACKED_CHOICES = (ON_TRACKED_SAVE_BESIDE, ON_TRACKED_REPLACE,
                      ON_TRACKED_CANCEL)

#: Where to look for git. ``shutil.which`` first, so a colleague's ordinary
#: install is used; the Command Line Tools path is the fallback for a GUI
#: launched from a desktop bundle, whose PATH is the login shell's and often
#: does not carry Xcode's shims.
_GIT_FALLBACKS = ("/Library/Developer/CommandLineTools/usr/bin/git",
                  "/usr/bin/git")


def _git_program():
    """The git to ask, or ``""`` when there is none."""
    import shutil
    found = shutil.which("git")
    if found:
        return found
    for candidate in _GIT_FALLBACKS:
        if Path(candidate).exists():
            return candidate
    return ""


def tracked_by_git(path):
    """``(tracked, detail)`` -- is this path a file the repository tracks?

    ``tracked`` is **True**, **False**, or **None** for "the question could not
    be asked", and the third is not folded into the second on purpose. This is
    the predicate a guard turns on: answering "not tracked" when the truth is
    "I could not look" is how the guard would go quiet exactly on the machines
    where it cannot check, and quietly-not-guarding is the defect, not the
    remedy. Callers treat None as protected and say why.

    A checkout with no ``.git`` -- a release tarball -- is a definite **False**:
    git ran, and answered that this is not a repository. That is what keeps the
    guard from turning a plain install into one that cannot fetch.

    Cheap by construction: one ``git ls-files --error-unmatch`` on one path,
    bounded by a timeout, with the ``GIT_*`` variables stripped for the reason
    ``scripts/ptm_qa.py`` strips them (under an exported ``GIT_DIR`` the answer
    describes the INHERITED repository, whatever cwd it is given).
    """
    import os
    import subprocess

    target = Path(str(path))
    program = _git_program()
    if not program:
        return None, ("git could not be found on this machine, so whether "
                      "this file is part of the distribution could not be "
                      "checked")
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    # The nearest ancestor that EXISTS, not `Path.cwd()`: a body fetched for
    # the first time has no directory yet, and asking git from wherever the GUI
    # happens to have been launched would answer about a different repository
    # -- or about none, reported as "not tracked", which is the wrong answer in
    # the one direction that matters here.
    directory = target.parent
    while not directory.is_dir() and directory != directory.parent:
        directory = directory.parent
    try:
        out = subprocess.run(
            [program, "ls-files", "--error-unmatch", "--", str(target)],
            cwd=str(directory), capture_output=True, text=True, env=env,
            timeout=30)
    except Exception as exc:                                 # noqa: BLE001
        return None, (f"git could not be run to check whether this file is "
                      f"part of the distribution ({exc})")
    if out.returncode == 0:
        return True, ""
    stderr = (out.stderr or "").lower()
    if "not a git repository" in stderr or "not a working tree" in stderr:
        return False, ""
    # Exit 1 is read as "not tracked" ONLY with git's own words for it. Any
    # other exit-1 failure is a question that could not be asked, and mapping
    # it to False would switch this guard off exactly where it cannot check.
    # The concrete case on macOS: `shutil.which("git")` finds the /usr/bin/git
    # xcrun shim, and with the Command Line Tools missing or moved that shim
    # prints "xcrun: error: invalid active developer path" and exits 1 -- on
    # which the committed kernel used to be overwritten in place, which is the
    # incident this function exists to prevent.
    if out.returncode == 1 and "did not match" in stderr:
        return False, ""
    return None, (f"git answered {out.returncode} when asked whether this file "
                  f"is part of the distribution "
                  f"({out.stderr.strip() or '<no stderr>'})")


def _beside_path(path, label="fetched", same_as=None):
    """A distinct, non-colliding name next to ``path``.

    Dated rather than numbered so two fetches on different days are
    distinguishable in a directory listing, and suffixed until it is free so a
    second fetch on one day never overwrites the first -- this whole guard
    exists because something overwrote something.

    ``same_as`` is the file about to be installed. When a sibling from an
    earlier fetch already holds exactly those bytes it is REUSED rather than
    duplicated: the card's Replace offer re-runs the whole fetch, so without
    this a user who answers Yes and is then refused on coverage collects a
    fresh copy of the identical kernel on every attempt.
    """
    path = Path(str(path))
    stamp = datetime.now().strftime("%Y%m%d")
    if same_as is not None:
        try:
            wanted = _md5(same_as)
            for sibling in sorted(path.parent.glob(
                    f"{path.stem}_{label}_*{path.suffix}")):
                if _md5(sibling) == wanted:
                    return sibling
        except OSError:                                      # pragma: no cover
            pass
    candidate = path.with_name(f"{path.stem}_{label}_{stamp}{path.suffix}")
    index = 2
    while candidate.exists():
        candidate = path.with_name(
            f"{path.stem}_{label}_{stamp}_{index}{path.suffix}")
        index += 1
    return candidate


def _install_beside(scratch, path):
    """Put ``scratch`` next to ``path`` without overwriting it, and say where.

    Both branches that decline to replace a tracked kernel end here, and they
    have to agree: a second copy of a kernel already on disk is exactly what
    the card's Replace offer would accumulate, one per attempt.
    """
    beside = _beside_path(path, same_as=scratch)
    if beside.exists():
        scratch.unlink()               # an identical copy is already there
    else:
        scratch.replace(beside)
    return beside


def _coverage_loss(old, new):
    """``""`` when ``new`` keeps everything ``old`` covered, else why not.

    Asked BEFORE any replacement, and asked as containment rather than as a
    day count: a kernel can be longer in days and still not span what the old
    one spanned, which is precisely the 899 -> 762 day clobber (the new arc did
    not merely shrink, it sat somewhere else).

    ``old is None`` means there is no file at the destination, so there is
    nothing to lose. An old file whose coverage cannot be READ is the opposite:
    it is a shipped kernel about which this cannot answer, and returning ""
    there would let an unreadable-coverage machine -- no spiceypy, or no
    leapseconds kernel, which is the state of any fresh checkout -- replace
    the 899-day kernel and only afterward report COVERAGE UNKNOWN. That is the
    incident, reproduced through the guard meant to stop it.
    """
    if old is None:
        return ""
    if not old.available:
        return (f"the coverage of the file on disk could not be read "
                f"({old.detail}), so the fetched kernel cannot be shown to "
                f"keep what it spans. A replacement that cannot be checked is "
                f"not a replacement this will make")
    if not new.available:
        return (f"the file on disk covers {old.start_utc} to {old.end_utc} "
                f"({old.days:.1f} days) and the fetched kernel's own coverage "
                f"could not be read back ({new.detail}), so it cannot be shown "
                f"to keep it")
    verdict = new.covers(old.start_utc, old.end_utc)
    if verdict:
        return ""
    if verdict is None:
        return (f"the file on disk covers {old.start_utc} to {old.end_utc} "
                f"and the two coverages could not be compared, so the fetched "
                f"kernel cannot be shown to keep it")
    return (f"the file on disk covers {old.start_utc} to {old.end_utc} "
            f"({old.days:.1f} days) and the fetched kernel covers "
            f"{new.start_utc} to {new.end_utc} ({new.days:.1f} days), which "
            f"does not contain it")


def _md5(path):
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iso(moment):
    return moment.strftime("%Y-%m-%dT%H:%M:%S")


def _parse_iso(text):
    try:
        return datetime.fromisoformat(str(text or "").replace("Z", ""))
    except (TypeError, ValueError):
        return None


def _update_index(project_root, name, fields):
    """Merge ``fields`` into this body's ``index.json`` entry.

    MERGED, not replaced. ``DataManager.store_body`` overwrites the whole
    entry, which is correct when it has just written every file; a card that
    fetches only a shape must not erase the kernel paths a previous fetch
    recorded. Never raises: a cache index that cannot be updated is a
    degraded cache, not a failed fetch, and the files themselves are on disk
    and are what the lookup actually reads.
    """
    from gui_core.body_lookup import CACHE_RELPATH
    index_path = Path(project_root) / CACHE_RELPATH / "index.json"
    try:
        index = json.loads(index_path.read_text()) if index_path.exists() else {}
        if not isinstance(index, dict):
            index = {}
    except Exception as exc:                                 # noqa: BLE001
        logger.warning("[body-fetch] %s could not be read (%s); it is being "
                       "rewritten from what is on disk.", index_path, exc)
        index = {}
    entry = index.get(name)
    if not isinstance(entry, dict):
        entry = {}
    entry.update(fields)
    entry.setdefault("cache_date", datetime.now().isoformat())
    index[name] = entry
    try:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text(json.dumps(index, indent=2))
    except Exception as exc:                                 # noqa: BLE001
        logger.warning("[body-fetch] %s could not be written (%s); the files "
                       "are on disk but the index does not list them.",
                       index_path, exc)


# ---------------------------------------------------------------------------
# 1. The ephemeris: a HORIZONS SPK spanning the run, with margin
# ---------------------------------------------------------------------------

def _needed_window(config):
    """``(earliest, latest, note)`` the run actually needs, or ``(None, ...)``.

    The earliest is the SPIN-UP start, not ``simulation.start_time``.
    Equilibration integrates BEFORE the start time, so a kernel that covers
    the output window and nothing earlier still leaves every spin-up epoch to
    the silent fallback -- and the spin-up is what sets the temperatures the
    output window begins from.
    """
    from gui_core.body_cards import run_window
    start, end, spin_up, note = run_window(config)
    if start is None:
        return None, None, note
    return (spin_up or start), end, note


def _margin_window(earliest, latest, margin_days):
    """The dates asked of HORIZONS: the run window pushed out at both ends."""
    low = _parse_iso(earliest) - timedelta(days=margin_days)
    high = _parse_iso(latest) + timedelta(days=margin_days)
    # HORIZONS' SPK generator takes plain dates; a date is a whole day, so
    # truncating start DOWN and stop UP keeps the margin from being eroded by
    # the truncation itself.
    return low.strftime("%Y-%m-%d"), (high + timedelta(days=1)).strftime("%Y-%m-%d")


def _coverage_verdict(spk_path, earliest, latest):
    """``(verdict, coverage)`` -- verdict True / False / None for unknown."""
    from gui_core.body_cards import spk_coverage
    coverage = spk_coverage(spk_path)
    if not coverage.available:
        return None, coverage
    return coverage.covers(earliest, latest), coverage


def _accepts_body_type(client):
    """Does this client's ``download_spk`` take a ``body_type``?"""
    import inspect
    try:
        signature = inspect.signature(client.download_spk)
    except (TypeError, ValueError):                          # pragma: no cover
        return False
    parameters = signature.parameters
    if "body_type" in parameters:
        return True
    return any(p.kind is inspect.Parameter.VAR_KEYWORD
               for p in parameters.values())


def _spk_request_failed(scratch, command, name, exc, timeout,
                        body_type="asteroid"):
    """The message for a request that did not produce a kernel.

    FIVE answers, not one. Until 2026-08-27 every one of these was reported as
    "HORIZONS could not be reached, or refused this request, or did not answer
    within 90 s" -- a sentence that names three unrelated causes and commits to
    none, so the first live use of the button sent its user to check a network
    that was up, while HORIZONS' own one-line explanation of what it objected
    to sat in the exception text after the words "could not be reached".

    "Could not be reached" now means the network, and nothing else does.
    """
    try:
        scratch.unlink()
    except OSError:
        pass

    tail = " No kernel was written and nothing has been changed."
    try:
        from horizons.horizons_client import (
            HorizonsHTTPError, HorizonsNetworkError, HorizonsNotFoundError,
            HorizonsParseError, HorizonsRefusalError)
    except Exception:                                        # noqa: BLE001
        HorizonsHTTPError = HorizonsNetworkError = ()        # noqa: N806
        HorizonsNotFoundError = HorizonsParseError = ()      # noqa: N806
        HorizonsRefusalError = ()                            # noqa: N806

    if HorizonsRefusalError and isinstance(exc, HorizonsRefusalError):
        precomputed = "pre-computed objects in the major body index" in str(exc)
        small = str(body_type or "").strip().lower() in ("asteroid", "comet")
        if precomputed and small:
            advice = (
                " HORIZONS builds SPK files only for small bodies, and not "
                "even for all of those: it serves THIS object's ephemeris from "
                "a pre-computed kernel of its own and will not generate a new "
                "one, by any designation form -- measured for (101955) Bennu "
                "on 2026-08-27 against four of them, and against the POST file "
                "endpoint. This is not a fault at your end and a retry will "
                "not change it. Get a kernel from the mission's SPICE archive "
                "(NAIF distributes them), point the SPICE metakernel field at "
                "it, and press Check coverage; or choose Keplerian elements "
                "deliberately, so the record says which was used.")
        elif precomputed:
            advice = (
                f" HORIZONS builds SPK files for small bodies only, and this "
                f"one was asked for as a {body_type or 'major body'}. Use "
                f"Keplerian elements, or a kernel you supply -- for a planet "
                f"or a satellite a planetary ephemeris such as de440s covers "
                f"it already.")
        else:
            advice = (
                " HORIZONS does not build SPK files for every object. If this "
                "is a major planet or a satellite, use Keplerian elements or a "
                "kernel you supply.")
        return FetchResult(kind="spk", message=(
            f"HORIZONS was reached and REFUSED to build a kernel for {name}."
            f"{tail} {exc}{advice}"))

    if HorizonsNotFoundError and isinstance(exc, HorizonsNotFoundError):
        return FetchResult(kind="spk", message=(
            f"HORIZONS was reached and has NO SMALL-BODY RECORD matching "
            f"{command!r}.{tail} Check the number or the designation in the "
            f"Name and NAIF id fields -- press Look up to fill them from "
            f"HORIZONS itself. Details: {exc}"))

    if HorizonsHTTPError and isinstance(exc, HorizonsHTTPError):
        return FetchResult(kind="spk", message=(
            f"HORIZONS was reached and answered with an HTTP error "
            f"{getattr(exc, 'status', '?')} for {command!r}.{tail} That is a "
            f"fault at the service, not in this request; try again later. "
            f"Details: {exc}"))

    if HorizonsParseError and isinstance(exc, HorizonsParseError):
        return FetchResult(kind="spk", message=(
            f"HORIZONS was reached and answered for {command!r}, but this "
            f"client did not understand the answer.{tail} That is a fault in "
            f"PTM3D, not in HORIZONS and not in your request; the answer is "
            f"quoted here so it can be reported. Details: {exc}"))

    if HorizonsNetworkError and isinstance(exc, HorizonsNetworkError):
        return FetchResult(kind="spk", message=(
            f"HORIZONS could not be reached, or did not answer within "
            f"{timeout} s ({exc}).{tail} Check the network and try again."))

    return FetchResult(kind="spk", message=(
        f"Asking HORIZONS for a kernel for {command!r} failed with an error "
        f"this code does not classify ({type(exc).__name__}: {exc}).{tail}"))


def _superseded_cache_warning(project_root, entry):
    """Say when the body cache holds an ephemeris the curated one replaces.

    Two kernels for one body on one disk, and only one of them used, is
    exactly the situation that needs a sentence rather than a silence. Bennu's
    cached ``ephemeris.spk`` is JPL solution 76 (2015-2023); the curated one
    is solution 118 (1900-2135). A card that quietly wired the second while
    the first sat in the body's own directory would leave the user with no way
    to tell which had run.

    Returns "" when there is nothing in the cache to mention.

    The cache directory is resolved from ``entry.name``, NOT from what the
    user typed. The registry accepts ``Bennu``, ``101955``, ``1999 RQ36`` and
    ``(101955) Bennu``; the cache directory is
    ``bodies/101955_Bennu_(1999_RQ36)``. Using the typed spelling made this
    whole disclosure vanish for three of the four -- the superseded kernel
    still on disk, and nothing said about it, for exactly the users who took
    the shortest route to the card.
    """
    try:
        cached = body_cache_dir(project_root, entry.name) / "ephemeris.spk"
    except ValueError:
        return ""
    if not cached.exists():
        return ""
    from gui_core.body_cards import spk_coverage
    coverage = spk_coverage(str(cached), naif_id=entry.ephemeris.naif_id,
                            project_root=project_root)
    span = (f"covering {coverage.start_utc} to {coverage.end_utc}"
            if coverage.available else "whose coverage could not be read here")
    return (f"NOT USED: {cached} is also on this disk, {span}. It is the "
            f"body cache's own kernel and it is NOT what this card wired -- "
            f"the curated entry above is. It is kept for provenance; delete "
            f"nothing on account of this line, but do not assume a run used "
            f"it.")


def _curated_spk(entry, project_root, name, naif_id, earliest, latest, note,
                 allow_online, curated_download, progress, downloader=None):
    """The curated ephemeris tier: on-disk check, an offer, then the wiring.

    Never asks HORIZONS. Never asks the network at all unless
    ``curated_download`` is True AND something is genuinely missing.
    """
    from gui_core.body_cards import spk_coverage
    from gui_core.curated_sources import (curated_kernel_status,
                                          download_curated_kernels,
                                          human_bytes)

    ephemeris = entry.ephemeris
    source_line = (f"Ephemeris source: {ephemeris.description}. "
                   f"{ephemeris.reference}. Coverage as published: "
                   f"{ephemeris.coverage_text()}.")
    status = curated_kernel_status(entry, project_root)

    download_lines = []
    if not status.ready:
        if not allow_online:
            return FetchResult(kind="spk", message=(
                f"{status.summary()} Online fetching is turned off, so "
                f"nothing was downloaded and nothing has been changed. "
                f"{source_line}"))
        command = (f"python3 scripts/fetch_kernels.py --bodies "
                   f"{ephemeris.fetch_bodies}")
        if not status.can_download:
            # Not ready, and a download would not fix it: the metakernel is
            # absent, or a member is a cloud placeholder (which `download`
            # would HYDRATE, since it finds a full-size file and hashes it),
            # or a member has no pinned source. Offering a download for any of
            # these spends bandwidth and changes nothing, and the "0 B" dialog
            # the first draft produced said so to nobody.
            return FetchResult(kind="spk", message=(
                f"{status.summary()} Nothing was downloaded and nothing has "
                f"been changed -- a download would not fix this. "
                f"{source_line}"))
        if not curated_download:
            size = status.download_summary()
            names = ", ".join(k.name for k in status.fetchable)
            return FetchResult(
                kind="spk",
                offer=(f"Download {entry.name}'s ephemeris now? {names} "
                       f"({size}), from {ephemeris.description}. These are "
                       f"pinned files with recorded sha256 values, not a "
                       f"HORIZONS request. A slow or flaky network is "
                       f"retried up to five times with ten seconds between "
                       f"attempts and cannot be canceled from here, so it "
                       f"may take several minutes; the same download runs "
                       f"outside this window as `{command}`."),
                offer_bytes=status.download_bytes,
                message=(
                f"{status.summary()} {entry.name}'s ephemeris is a pinned "
                f"download, not a HORIZONS request -- {entry.why_curated} "
                f"Confirm the download to fetch it here, or run `{command}` "
                f"yourself. Nothing has been changed. {source_line}"))
        ok, download_lines = download_curated_kernels(
            entry, project_root, progress=progress, downloader=downloader)
        status = curated_kernel_status(entry, project_root)
        if not ok or not status.ready:
            return FetchResult(kind="spk", warnings=tuple(download_lines),
                               message=(
                f"{entry.name}'s ephemeris was NOT installed: "
                f"{status.summary()} Nothing about this card has been "
                f"changed. {source_line}"))

    # -- read the coverage back, exactly as the HORIZONS path does ----------
    coverage = spk_coverage(ephemeris.metakernel,
                            naif_id=ephemeris.naif_id,
                            project_root=project_root)
    warnings = list(download_lines)
    warnings.extend(entry.notes)
    superseded = _superseded_cache_warning(project_root, entry)
    if superseded:
        warnings.append(superseded)

    if not coverage.available:
        return FetchResult(
            kind="spk", ok=False,
            paths={"metakernel": str(status.metakernel_path)},
            warnings=tuple(warnings),
            message=(f"{entry.name}'s pinned kernels are on this disk and "
                     f"{ephemeris.metakernel} names them. {source_line}"),
            refusal=(f"COVERAGE UNKNOWN: {coverage.detail}. The ephemeris "
                     f"source has been LEFT ALONE rather than set to SPICE. "
                     f"The engine substitutes Keplerian elements for any "
                     f"epoch outside a kernel and logs nothing, so an "
                     f"unverified kernel is an unverified run -- and a "
                     f"curated kernel is not exempt from being checked."))

    verdict = coverage.covers(earliest, latest)
    if verdict is None or not verdict:
        return FetchResult(
            kind="spk", ok=False,
            paths={"metakernel": str(status.metakernel_path)},
            warnings=tuple(warnings),
            message=(f"{ephemeris.metakernel} covers {coverage.start_utc} to "
                     f"{coverage.end_utc} ({coverage.days:.1f} days) for body "
                     f"{ephemeris.naif_id}. {source_line}"),
            refusal=(f"REFUSED: this run needs {earliest} to {latest} "
                     f"({note}), which that kernel does NOT span. The "
                     f"ephemeris source has been LEFT ALONE rather than set "
                     f"to SPICE, because the engine would serve the uncovered "
                     f"epochs from Keplerian elements silently. Move the run "
                     f"inside the covered span, or choose Keplerian "
                     f"deliberately so the record says so."
                     if verdict is not None else
                     f"COVERAGE UNKNOWN: the run window {earliest} to "
                     f"{latest} and the kernel's own could not be compared, "
                     f"so the ephemeris source has been LEFT ALONE."))

    values = {
        "generic_body.spice_metakernel": ephemeris.metakernel,
        "generic_body.ephemeris_source": "spice",
        "generic_body.naif_id": ephemeris.naif_id,
    }
    label = f"curated: {ephemeris.description}"
    provenance = {key: label for key in values}
    if naif_id and str(naif_id).rstrip("; ").strip() != ephemeris.naif_id:
        warnings.append(
            f"The NAIF id has been set to {ephemeris.naif_id}, which is the "
            f"id this kernel declares; the card had {naif_id}. Small-body SPK "
            f"ids exist in a 2000000+N and a 20000000+N convention and only "
            f"the kernel's own works with it.")

    # The cache index is written ONLY when this fetch actually put something
    # on disk. The HORIZONS path always has -- it just downloaded a kernel --
    # but the curated path's normal case installs nothing, and
    # `data/horizons_cache/index.json` is a TRACKED file: recording "I looked
    # and it was already here" in it would leave every user's checkout dirty
    # after pressing a button that changed nothing. Wiring a kernel that is
    # already present is a read, and a read does not write.
    if download_lines:
        _update_index(project_root, name, {
            "naif_id": ephemeris.naif_id,
            "body_type": "asteroid",
            "curated_source": ephemeris.description,
            "metakernel_path": str(status.metakernel_path),
            "spk_path": ", ".join(str(f.path) for f in status.files),
            "spk_start": coverage.start_utc,
            "spk_stop": coverage.end_utc,
        })

    return FetchResult(
        ok=True, kind="spk", values=values, provenance=provenance,
        from_cache=not download_lines,
        paths={"metakernel": str(status.metakernel_path)},
        warnings=tuple(warnings),
        message=(f"{source_line} {ephemeris.metakernel} covers "
                 f"{coverage.start_utc} to {coverage.end_utc} "
                 f"({coverage.days:.1f} days) for body {ephemeris.naif_id}; "
                 f"this run needs {earliest} to {latest} ({note}), which is "
                 f"inside it -- verified by reading the kernel back, not by "
                 f"trusting the table. HORIZONS was not asked: "
                 f"{entry.why_curated}"))


def fetch_spk(config, project_root, name="", naif_id="", body_type="asteroid",
              margin_days=SPK_MARGIN_DAYS, allow_online=True, refetch=False,
              timeout=HORIZONS_SPK_TIMEOUT_S, client_factory=None,
              curated=True, curated_download=False, progress=None,
              downloader=None, on_tracked=ON_TRACKED_SAVE_BESIDE):
    """Get an SPK that spans this run: curated registry, cache, or HORIZONS.

    Returns a :class:`FetchResult`. ``values`` sets the metakernel path and
    ``generic_body.ephemeris_source`` -- and sets the SOURCE only when the
    kernel has been READ BACK and shown to cover the whole run including
    spin-up. An unverifiable or insufficient kernel leaves the ephemeris
    source alone, so a card cannot end up saying "spice" over a kernel that
    would be silently ignored for most of the run.

    THE CURATED TIER COMES FIRST, and for a body in it HORIZONS is never
    asked. That is not an optimization. HORIZONS' refusal for Bennu is correct
    and permanent -- it serves that body from a pre-computed kernel and will
    not generate another, by any designation form -- so asking is a slower
    route to a message the user can do nothing with. What the curated tier
    offers instead is the mission product the refusal points at.

    For a curated body the tier's own first step is still a cache check: is
    the pinned kernel already fetched into ``data/kernels``? If it is, this
    function makes NO network call of any kind, and a machine with the kernel
    on disk reaches a wired, coverage-verified card offline.

    ``curated_download`` is the consent. With it False (the default) a curated
    body whose kernel is missing gets an OFFER naming the size and the command,
    and nothing is downloaded; the card asks, and calls again with it True.
    Downloading 80 MB because somebody pressed a button labeled "fetch
    kernel" would be a surprise measured in minutes.

    ``on_tracked`` decides what happens when the destination is a file the
    repository SHIPS -- four ``ephemeris.spk`` are tracked, and the fourteen
    Justitia configurations read one of them. The default,
    ``ON_TRACKED_SAVE_BESIDE``, keeps the tracked file exactly as it is,
    installs the download beside it under a dated name and points the
    configuration at the copy; the answer says which file was written, which
    was left alone, and what each covers. ``ON_TRACKED_REPLACE`` is the
    explicit consent to overwrite -- and is STILL refused if the new kernel
    does not contain the old one's coverage, because agreeing to replace a
    kernel is not agreeing to lose an arc. ``ON_TRACKED_CANCEL`` discards the
    download. An untracked destination is unaffected by any of this.
    """
    name = str(name or "").strip()
    if not name:
        return FetchResult(kind="spk", message=(
            "This body has no name yet, so there is nowhere to cache a kernel. "
            "Fill in the Name field (Look up will do it) and try again."))

    if on_tracked not in ON_TRACKED_CHOICES:
        # By name, not by falling back to the default: a caller that passed a
        # misspelled choice would otherwise get the default's behavior under
        # its own word for something else.
        return FetchResult(kind="spk", message=(
            f"on_tracked={on_tracked!r} is not one of "
            f"{', '.join(ON_TRACKED_CHOICES)}. Nothing has been fetched."))

    earliest, latest, note = _needed_window(config)
    if earliest is None:
        return FetchResult(kind="spk", message=(
            f"The run window could not be worked out, so there is no window to "
            f"fetch a kernel for: {note}"))

    if curated:
        from gui_core.curated_sources import curated_body
        entry = curated_body(name, naif_id)
        if entry is not None:
            return _curated_spk(entry, project_root, name, naif_id,
                                earliest, latest, note, allow_online,
                                curated_download, progress, downloader)

    try:
        body_dir = body_cache_dir(project_root, name)
    except ValueError as exc:
        return FetchResult(kind="spk", message=str(exc))
    spk_path = body_dir / "ephemeris.spk"
    metakernel_path = body_dir / "metakernel.tm"

    # -- cache first -------------------------------------------------------
    if spk_path.exists() and not refetch:
        verdict, coverage = _coverage_verdict(str(spk_path), earliest, latest)
        if verdict:
            return _spk_success(
                project_root, name, naif_id, spk_path, metakernel_path,
                coverage, earliest, latest, note, from_cache=True,
                requested=("", ""))
        if verdict is None:
            hint = (f"The cached kernel {spk_path.name} is on disk but its "
                    f"coverage could not be read ({coverage.detail}).")
        else:
            hint = (f"The cached kernel covers {coverage.start_utc} to "
                    f"{coverage.end_utc}, and this run needs {earliest} to "
                    f"{latest}.")
        if not allow_online:
            return FetchResult(kind="spk", message=(
                f"{hint} Online fetching is turned off, so nothing was "
                f"downloaded and nothing has been changed."))
        logger.info("[body-fetch] refetching an SPK for %s: %s", name, hint)

    if not allow_online:
        return FetchResult(kind="spk", message=(
            f"There is no cached kernel for {name} and online fetching is "
            f"turned off, so nothing has been changed."))

    # -- ask HORIZONS ------------------------------------------------------
    start_text, stop_text = _margin_window(earliest, latest, margin_days)
    # The identifier goes to the client AS IT IS. Building the HORIZONS
    # COMMAND string is the client's job (`small_body_spk_command`) and used to
    # be done here, wrongly: this appended a semicolon to whatever was in the
    # NAIF id field, and that field holds an SPK ID -- 2101955 for Bennu,
    # 20000623 for Chimaera. HORIZONS reads `<number>;` as a small-body RECORD
    # number, which stop at 895910, so every numbered asteroid came back
    # "DXREAD: requested IOBJ= ... is out of bounds" and the button had never
    # worked for one. Two layers each doing half of a syntax decision is what
    # made that invisible.
    command = str(naif_id or name).strip()

    if client_factory is None:
        try:
            from horizons.horizons_client import HorizonsClient
        except Exception as exc:                             # noqa: BLE001
            return FetchResult(kind="spk", message=(
                f"The HORIZONS client is not available here ({exc}), so no "
                f"kernel could be fetched. Nothing has been changed."))

        def client_factory():
            return HorizonsClient(timeout=timeout)

    try:
        body_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return FetchResult(kind="spk", message=(
            f"The cache directory {body_dir} could not be created ({exc}), so "
            f"there is nowhere to put a kernel. Nothing has been changed."))

    # Download to a scratch name. Writing straight over a working kernel would
    # replace a kernel that covers the run with a truncated one if the request
    # failed halfway, and the file would still look like an SPK.
    scratch = body_dir / "ephemeris.spk.part"
    # Constructing the client, inspecting it and calling it are all INSIDE the
    # try. This function is called from a Qt slot, where an escaping exception
    # is a qFatal that takes the window down with any unsaved edits (measured:
    # exit 134); "returns a message, never raises" is the contract, and
    # `client_factory()` and `_accepts_body_type` can both raise -- a factory
    # that fails to build a client, or an object with no `download_spk` at all.
    try:
        client = client_factory()
        extra = {"body_type": body_type}
        if not _accepts_body_type(client):
            # A client older than the command-form fix, or a stub. Say so:
            # such a client is also the one that builds the COMMAND string the
            # old, broken way, so a fetch that fails here fails for a reason
            # this message is the only place that names.
            logger.warning(
                "[body-fetch] %s.download_spk does not take body_type, so this "
                "body is being fetched with whatever command form that client "
                "builds. A 'DXREAD ... out of bounds' answer means the client "
                "predates the small-body command fix.", type(client).__name__)
            extra = {}
        client.download_spk(command, start_text, stop_text, str(scratch),
                            **extra)
    except Exception as exc:                                 # noqa: BLE001
        return _spk_request_failed(scratch, command, name, exc, timeout,
                                   body_type)

    if not scratch.exists() or scratch.stat().st_size == 0:
        try:
            scratch.unlink()
        except OSError:
            pass
        return FetchResult(kind="spk", message=(
            f"HORIZONS answered but produced an empty kernel for {command!r}. "
            f"Nothing has been changed."))

    # -- verify BEFORE installing -----------------------------------------
    verdict, coverage = _coverage_verdict(str(scratch), earliest, latest)
    (spk_path, metakernel_path, tracked_warnings, refused,
     protected) = _place_download(scratch, spk_path, metakernel_path,
                                  coverage, earliest, latest, on_tracked)
    if refused is not None:
        return refused

    if verdict is None or not verdict:
        # BOTH refusals keep the file and index it -- it is a real kernel and
        # the user may want it -- and BOTH return no values, so the card
        # cannot set the ephemeris source over it. The two share this block on
        # purpose: they are one decision ("do not claim SPICE") with two
        # reasons, and writing it twice is how the two would drift apart.
        metakernel_path, mk_usable, mk_warnings = _ensure_metakernel(
            spk_path, metakernel_path, name, project_root)
        tracked_warnings = list(tracked_warnings) + mk_warnings
        _update_index(project_root, name, {
            "naif_id": str(naif_id or ""), "body_type": body_type,
            "spk_path": str(spk_path), "spk_start": start_text,
            "spk_stop": stop_text,
            "metakernel_path": str(metakernel_path) if mk_usable else "",
        })

    if verdict is None:
        return FetchResult(
            kind="spk", ok=False, paths={"spk": str(spk_path)},
            warnings=tracked_warnings, tracked_destination=protected,
            message=(f"A kernel was downloaded to {spk_path} for "
                     f"{start_text} to {stop_text}."),
            refusal=(f"COVERAGE UNKNOWN: the kernel's own coverage could not "
                     f"be read back ({coverage.detail}), so it cannot be shown "
                     f"to span this run. The ephemeris source has been LEFT "
                     f"ALONE rather than set to SPICE: the engine substitutes "
                     f"Keplerian elements for any epoch outside a kernel and "
                     f"logs nothing, so an unverified kernel is an unverified "
                     f"run. Install spiceypy, or check the coverage yourself "
                     f"and set the source by hand."))

    if not verdict:
        return FetchResult(
            kind="spk", ok=False, paths={"spk": str(spk_path)},
            warnings=tracked_warnings, tracked_destination=protected,
            message=(f"A kernel was downloaded to {spk_path}; it covers "
                     f"{coverage.start_utc} to {coverage.end_utc} "
                     f"({coverage.days:.1f} days)."),
            refusal=(f"REFUSED: this run needs {earliest} to {latest} "
                     f"({note}), which that kernel does NOT span, even though "
                     f"{start_text} to {stop_text} was requested -- HORIZONS "
                     f"returns what it has, which for some objects is a "
                     f"shorter arc than you asked for. The ephemeris source "
                     f"has been LEFT ALONE rather than set to SPICE, because "
                     f"the engine would serve the uncovered epochs from "
                     f"Keplerian elements silently. Shorten the run, move its "
                     f"start time inside the covered span, or choose Keplerian "
                     f"deliberately so the record says so."))

    return _spk_success(project_root, name, naif_id, spk_path, metakernel_path,
                        coverage, earliest, latest, note, from_cache=False,
                        requested=(start_text, stop_text), body_type=body_type,
                        extra_warnings=tracked_warnings,
                        tracked_destination=protected)


def _place_download(scratch, spk_path, metakernel_path, coverage, earliest,
                    latest, on_tracked):
    """Install a downloaded kernel without overwriting one the repository ships.

    ``(spk_path, metakernel_path, warnings, refusal, protected)``. The first
    two are where the kernel and its metakernel ACTUALLY are afterward -- which
    for a tracked destination is beside the shipped file, not over it -- so
    everything downstream, including the value written into
    ``generic_body.spice_metakernel``, points at what was really written.
    ``refusal`` is a :class:`FetchResult` to return instead, or None, and
    ``protected`` is the tracked path that was NOT overwritten (``""`` when the
    destination was ordinary), which is the flag the card offers Replace on.

    An untracked destination takes the original path: ``scratch.replace``,
    nothing else changed. That is the ordinary case and it is deliberately
    untouched -- ``data/horizons_cache`` is a cache for every body nobody has
    committed, and a guard that made those slower or noisier would be paid for
    by everyone to protect four files.
    """
    tracked, why = tracked_by_git(spk_path)
    if tracked is False:
        scratch.replace(spk_path)
        return spk_path, metakernel_path, [], None, ""

    if tracked is None and not spk_path.exists():
        # Unknown, but there is nothing at the destination to protect. Install
        # normally, and SAY that the check could not be made -- guarding an
        # absent file would make every first fetch on such a machine land under
        # a dated name the cache-first check never looks for, so the next press
        # downloads the whole kernel again and parks another copy.
        scratch.replace(spk_path)
        return spk_path, metakernel_path, [
            f"Whether {spk_path.name} is part of the distribution could not be "
            f"checked, because {why}. Nothing was at that path, so the fetched "
            f"kernel was installed there as usual."], None, ""

    # tracked is True, or None with a file already there -- both protected.
    # Treating "could not ask" as "not tracked" would switch this guard off
    # exactly where it cannot check, which is the failure mode it exists to
    # remove.
    def unchecked(what_happened):
        """The 'could not ask git' line, per branch.

        Built where the branch is, not once up front: the first draft appended
        one sentence saying the file had been left alone and saved beside, and
        the REPLACE and CANCEL branches then did neither -- a result carrying
        both "left alone" and "REPLACED".
        """
        if tracked is not None:
            return []
        return [f"{spk_path.name} was treated as part of the distribution "
                f"because {why}. {what_happened} If this file is yours, move "
                f"or delete it and fetch again."]

    old_coverage = _coverage_verdict(str(spk_path), earliest, latest)[1] \
        if spk_path.exists() else None
    spans = _both_spans(old_coverage, coverage)

    if on_tracked == ON_TRACKED_CANCEL:
        try:
            scratch.unlink()
        except OSError:
            pass
        warnings = unchecked("The download was discarded, as asked.")
        return spk_path, metakernel_path, warnings, FetchResult(
            kind="spk", ok=False, warnings=warnings,
            message=(f"Canceled: {spk_path.name} is part of the distribution "
                     f"and has NOT been touched. The downloaded kernel was "
                     f"discarded. {spans}")), ""

    if on_tracked == ON_TRACKED_REPLACE:
        loss = _coverage_loss(old_coverage, coverage)
        if not loss:
            scratch.replace(spk_path)
            warnings = unchecked("It has been REPLACED, as asked.")
            warnings.append(
                f"{spk_path.name} is part of the distribution and has been "
                f"REPLACED, as asked. {spans} Restore the shipped kernel with "
                f"`git checkout -- {spk_path}` if this was not what you meant.")
            return spk_path, metakernel_path, warnings, None, ""
        # Asked to replace, and refusing -- but the download is real and is
        # kept, beside, rather than thrown away. Which file is which is stated;
        # this is not a quiet substitution.
        beside = _install_beside(scratch, spk_path)
        warnings = unchecked(f"It has been left alone and the new kernel is at "
                             f"{beside.name}.")
        return spk_path, metakernel_path, warnings, FetchResult(
            kind="spk", ok=False, paths={"spk": str(beside)},
            warnings=warnings,
            message=(f"The kernel was saved as {beside.name}. "
                     f"{spk_path.name} has NOT been changed."),
            refusal=(f"REFUSED to replace {spk_path.name}: {loss}. Replacing "
                     f"it would drop coverage this repository ships and other "
                     f"configurations read -- the exact failure that put a "
                     f"762-day kernel over Justitia's 899-day one, spanning "
                     f"none of the campaign windows. Point the configuration "
                     f"at {beside.name} if the new arc is the one you want, or "
                     f"widen the run window and fetch again.")), ""

    # ON_TRACKED_SAVE_BESIDE -- the default, and the only one that writes
    # something useful without asking anybody to agree to anything.
    beside = _install_beside(scratch, spk_path)
    # The metakernel goes with THIS kernel, under a name that pairs with it by
    # sight: a `.tm` that names a different `.spk` than the one beside it is
    # how a run ends up on a kernel nobody chose.
    warnings = unchecked(f"It has been left alone and the new kernel is at "
                         f"{beside.name}.")
    warnings.append(
        f"{spk_path.name} is part of the distribution, so it has been left "
        f"exactly as it is and the fetched kernel saved beside it as "
        f"{beside.name}; this configuration now points at the copy. {spans} "
        f"To replace the shipped kernel instead, fetch again with Replace -- "
        f"it is refused if it would lose coverage.")
    return beside, beside.with_suffix(".tm"), warnings, None, str(spk_path)


def _both_spans(old_coverage, new_coverage):
    """One sentence naming what each kernel covers, for the offer."""
    def span(label, coverage):
        if coverage is None:
            return f"{label}: not on disk"
        if not coverage.available:
            return f"{label}: coverage could not be read ({coverage.detail})"
        return (f"{label}: {coverage.start_utc} to {coverage.end_utc} "
                f"({coverage.days:.1f} days)")
    return (f"Coverage -- {span('the file on disk', old_coverage)}; "
            f"{span('the fetched kernel', new_coverage)}.")


def _ensure_metakernel(spk_path, metakernel_path, name, project_root=None):
    """``(path, wrote, warnings)`` -- a usable metakernel, without clobbering one.

    The write itself reuses ``SPKGenerator``, which carries the generic-body metakernel fixes:
    ``PATH_VALUES`` points at the shared ``data/kernels`` tree rather than at
    the body's own directory, and long paths are split with SPICE's '+'
    continuation so ``furnsh`` does not silently truncate them. It also emits
    ABSOLUTE paths, by design, for that same truncation workaround -- which is
    exactly why it must not be pointed at a committed metakernel:
    ``data/horizons_cache/bodies/269_Justitia_(A887_SA)/metakernel.tm`` says so
    in its own header ("Hand-written relative, NOT regenerated verbatim from
    SPKGenerator.create_metakernel() ... baking those in here would make this
    committed file resolve only on the machine that generated it"), and a fetch
    regenerated it anyway. On a CACHE HIT, with nothing downloaded at all.

    So: a tracked metakernel that still resolves is LEFT ALONE and returned as
    it is. One that is tracked and does NOT resolve gets a machine-local
    variant written beside it, named, and pointed at -- the committed file
    stays the committed file either way.

    The middle value is USABLE, not "was written": "leave it alone" is the
    right answer AND a usable metakernel, while "it does not resolve and the
    local copy could not be written either" is neither. Reporting the second
    as usable would wire ``generic_body.spice_metakernel`` to the very file
    :func:`_metakernel_resolves` had just rejected -- the run served from the
    ephemeris that file names, which is the same silence in the other
    direction.
    """
    metakernel_path = Path(str(metakernel_path))
    tracked, why = tracked_by_git(metakernel_path)
    if metakernel_path.exists() and tracked is not False:
        resolved, detail = _metakernel_resolves(metakernel_path, spk_path,
                                                project_root)
        protected = ("part of the distribution" if tracked
                     else f"treated as part of the distribution because {why}")
        if resolved:
            logger.info("[body-fetch] %s is %s and resolves on this machine; "
                        "leaving it as it is.", metakernel_path.name, protected)
            return metakernel_path, True, []
        local = _local_metakernel_path(metakernel_path)
        wrote = _write_metakernel(spk_path, local, name)
        if not wrote:
            return metakernel_path, False, [
                f"{metakernel_path.name} is {protected} and {detail}, and a "
                f"machine-local replacement could not be written either. The "
                f"kernel is on disk and the SPICE metakernel field points at "
                f"the .spk directly rather than at a .tm this fetch has "
                f"already found unusable."]
        return local, True, [
            f"{metakernel_path.name} is {protected}, so it has NOT been "
            f"rewritten -- {detail}. A machine-local {local.name} was written "
            f"beside it and this configuration points at that instead. The "
            f"committed file is unchanged."]

    wrote = _write_metakernel(spk_path, metakernel_path, name)
    return metakernel_path, wrote, []


def _local_metakernel_path(metakernel_path):
    """``<stem>_local.tm`` beside a committed metakernel, or a dated name.

    One stable name, because this is a machine-local artifact that a second
    fetch should overwrite rather than accumulate -- unless that name is itself
    tracked somewhere, in which case the dated fallback applies and the guard
    does not have to trust its own naming convention.
    """
    local = metakernel_path.with_name(
        f"{metakernel_path.stem}_local{metakernel_path.suffix}")
    if tracked_by_git(local)[0] is False:
        return local
    return _beside_path(metakernel_path, label="local")


def _metakernel_resolves(metakernel_path, spk_path, project_root):
    """``(resolves, detail)`` -- would this metakernel load, and load THIS spk?

    Both halves matter. A committed metakernel that resolves is the right file
    to keep; one that resolves but names a DIFFERENT kernel than the one just
    installed would quietly serve the run from the old ephemeris, which is the
    same silence in the other direction.
    """
    try:
        from gui_core.body_cards import metakernel_members
        members = metakernel_members(str(metakernel_path), project_root)
    except Exception as exc:                                 # noqa: BLE001
        return False, f"it could not be read on this machine ({exc})"
    if members.error:
        return False, f"it does not resolve here ({members.error})"
    absent = [token for token, _p in list(members.missing) + list(members.dataless)]
    absent += list(members.unresolved)
    if absent:
        return False, (f"it names {len(absent)} kernel(s) that are not usable "
                       f"on this machine ({', '.join(absent[:3])})")
    try:
        target = Path(str(spk_path)).resolve()
    except OSError:                                          # pragma: no cover
        target = Path(str(spk_path))
    if not any(path.resolve() == target for _token, path in members.kernels):
        return False, (f"it does not name {Path(str(spk_path)).name}, so it "
                       f"would serve the run from a different ephemeris")
    return True, ""


def _write_metakernel(spk_path, metakernel_path, name):
    """Write the .tm at ``metakernel_path``. Never raises.

    The unguarded write. Reached only through :func:`_ensure_metakernel`, which
    decides WHETHER this path may be written -- do not call it directly, or the
    committed-metakernel guard is back to being a comment.
    """
    try:
        from horizons.spk_generator import SPKGenerator
        SPKGenerator().create_metakernel(str(metakernel_path), str(spk_path),
                                         name)
        return True
    except Exception as exc:                                 # noqa: BLE001
        logger.warning("[body-fetch] no metakernel for %s (%s); the kernel "
                       "itself is on disk.", name, exc)
        return False


def _spk_success(project_root, name, naif_id, spk_path, metakernel_path,
                 coverage, earliest, latest, note, from_cache, requested,
                 body_type="asteroid", extra_warnings=(),
                 tracked_destination=""):
    warnings = list(extra_warnings)
    # `_ensure_metakernel`, NOT `_write_metakernel`. This line is reached on
    # the CACHE-HIT path too -- from_cache=True, nothing downloaded -- and that
    # is how two committed metakernels were regenerated to machine-absolute
    # PATH_VALUES by a button press that fetched nothing at all.
    metakernel_path, have_metakernel, mk_warnings = _ensure_metakernel(
        spk_path, metakernel_path, name, project_root)
    warnings.extend(mk_warnings)
    if not have_metakernel:
        warnings.append(
            "The kernel is on disk but its metakernel could not be written, "
            "so the SPICE metakernel field points at the .spk directly. The "
            "engine also needs leap seconds and a planetary ephemeris; if the "
            "run reports a SPICE error, write a metakernel by hand.")

    kernel_for_config = str(metakernel_path if have_metakernel else spk_path)
    values = {
        "generic_body.spice_metakernel": kernel_for_config,
        "generic_body.ephemeris_source": "spice",
    }
    provenance = {
        "generic_body.spice_metakernel": "HORIZONS SPK",
        "generic_body.ephemeris_source": "HORIZONS SPK",
    }

    # The body id INSIDE the kernel, not one computed from the asteroid
    # number: HORIZONS migrated small-body SPK ids from 2000000+N to
    # 20000000+N, and querying the wrong one fails with SPKINSUFFDATA.
    spk_body_id = ""
    try:
        from data_manager import read_spk_body_id
        spk_body_id = read_spk_body_id(str(spk_path),
                                       fallback_naif_id=str(naif_id or "")) or ""
    except Exception as exc:                                 # noqa: BLE001
        logger.info("[body-fetch] could not read the body id from %s (%s)",
                    spk_path, exc)
    if spk_body_id:
        values["generic_body.naif_id"] = str(spk_body_id)
        provenance["generic_body.naif_id"] = "read from the SPK"
        if naif_id and str(spk_body_id) != str(naif_id).rstrip("; ").strip():
            warnings.append(
                f"The NAIF id has been set to {spk_body_id}, which is the id "
                f"the kernel actually declares; the lookup had "
                f"{naif_id}. HORIZONS uses both a 2000000+N and a 20000000+N "
                f"convention for small bodies and only the kernel's own id "
                f"works with it.")
    else:
        warnings.append(
            "The body id inside the kernel could not be read, so the NAIF id "
            "field was left as it is. If the run reports SPKINSUFFDATA, that "
            "is the field to check.")

    _update_index(project_root, name, {
        "naif_id": str(spk_body_id or naif_id or ""),
        "body_type": body_type,
        "spk_path": str(spk_path),
        "spk_start": requested[0] or coverage.start_utc,
        "spk_stop": requested[1] or coverage.end_utc,
        "metakernel_path": str(metakernel_path) if have_metakernel else "",
    })

    where = ("was already cached" if from_cache
             else f"was fetched from HORIZONS ({requested[0]} to {requested[1]} "
                  f"requested, one year of margin at each end)")
    return FetchResult(
        ok=True, kind="spk", values=values, provenance=provenance,
        from_cache=from_cache, tracked_destination=tracked_destination,
        paths={"spk": str(spk_path),
               "metakernel": str(metakernel_path) if have_metakernel else ""},
        warnings=warnings,
        message=(f"The kernel {where}: {spk_path.name} covers "
                 f"{coverage.start_utc} to {coverage.end_utc} "
                 f"({coverage.days:.1f} days). This run needs {earliest} to "
                 f"{latest} ({note}), which is inside it -- verified by "
                 f"reading the kernel back, not by trusting the request."))


# ---------------------------------------------------------------------------
# 2. The shape: a DAMIT model, when one exists
# ---------------------------------------------------------------------------

def _asteroid_number(naif_id, name):
    """The IAU number, from a NAIF id or a leading number in the name."""
    text = str(naif_id or "").rstrip("; ").strip()
    if text.isdigit():
        value = int(text)
        # 20000000 + N is the current small-body SPK convention, 2000000 + N
        # the older one. Both appear in this project's own cache.
        for base in (20000000, 2000000):
            if value > base:
                return value - base
        return value
    head = str(name or "").strip().split()
    if head and head[0].isdigit():
        return int(head[0])
    return None


def shape_provenance(path):
    """The sidecar next to a fetched shape, as a dict, or ``{}``."""
    sidecar = Path(str(path)).with_name(
        Path(str(path)).stem + "_provenance.json")
    try:
        data = json.loads(sidecar.read_text())
    except Exception:                                        # noqa: BLE001
        return {}
    return data if isinstance(data, dict) else {}


def fetch_curated_shape(entry, project_root):
    """Offer a curated body's shipped mission shape model.

    No network, ever: the file is in the repository. What this returns is the
    same shape of answer a DAMIT fetch returns -- values, provenance, warnings
    -- so the card applies it through one code path and cannot end up with two
    ways of writing a shape into a configuration.

    The kind stays ``"damit_shape"``. It is the routing key the card's
    ``_fetch_controls`` maps to the shape button and the shape note, and
    inventing a second kind for the same slot would mean a second row in that
    table and a second chance for the two to disagree about which widget a
    result belongs to.
    """
    shape = entry.shape
    from gui_core.curated_sources import curated_shape_path
    path, present = curated_shape_path(entry, project_root)
    if not present:
        return FetchResult(kind="damit_shape", message=(
            f"{entry.name}'s shape model should be at {path} and is not on "
            f"this disk. It is a tracked file in this repository, so a "
            f"checkout missing it is a checkout missing a tracked file -- "
            f"restore it with `git checkout -- {shape.relpath}` rather than "
            f"looking for it online. Nothing has been changed."))

    values = {
        "generic_body.shape.type": "custom_mesh",
        "generic_body.shape.mesh_file": shape.relpath,
        "generic_body.shape.mesh_scale": shape.mesh_scale,
        "generic_body.shape.radius_km": shape.radius_km,
    }
    label = f"curated: {shape.description}"
    provenance = {key: label for key in values}

    # THE SPIN GOES WITH THE SHAPE, for the same reason the DAMIT path sends
    # it: the mesh is defined in the body-fixed frame its own pole and period
    # establish, so the mesh with somebody else's pole is the body pointing
    # the wrong way, and no output shows it. Offering the mesh alone also
    # made `spin_consistency` compare nothing and return "", which the card
    # renders as an affirmative "these agree".
    spin_label = f"curated: {shape.spin_reference}"
    if shape.spin_period_hours > 0:
        values["generic_body.spin.period_hours"] = shape.spin_period_hours
        provenance["generic_body.spin.period_hours"] = spin_label
    values["generic_body.spin.pole_ra_deg"] = shape.spin_pole_ra_deg
    values["generic_body.spin.pole_dec_deg"] = shape.spin_pole_dec_deg
    provenance["generic_body.spin.pole_ra_deg"] = spin_label
    provenance["generic_body.spin.pole_dec_deg"] = spin_label

    warnings = [
        f"The spin came with the model and is offered with it: period "
        f"{shape.spin_period_hours:g} h, pole RA/Dec "
        f"({shape.spin_pole_ra_deg:g}, {shape.spin_pole_dec_deg:g}). "
        f"{shape.spin_reference}. A shape model is defined in the body-fixed "
        f"frame its own pole and period establish, so keeping the shape and a "
        f"different spin orients the body wrongly and nothing downstream "
        f"re-checks it.",
        f"DAMIT was NOT searched for {entry.name}, and this is not a fallback: "
        f"DAMIT publishes lightcurve inversions and this body's shape is not "
        f"one. {entry.why_curated}",
        f"Archive: {shape.archive}. {shape.vertices} vertices, {shape.faces} "
        f"facets, already in {shape.units} -- so the shape-model scale is "
        f"{shape.mesh_scale:g} and nothing is rescaled on the way in. Asteroid "
        f"shape models are conventionally in kilometers here while lunar and "
        f"Europa terrain meshes are in meters; the run prints the units it "
        f"inferred.",
    ]
    if shape.provenance_relpath:
        warnings.append(
            f"Full provenance, including what this file supersedes and by how "
            f"much: {shape.provenance_relpath}.")
    if shape.tracked:
        warnings.append(
            f"This file is TRACKED in git and ships with the repository, even "
            f"though *.obj is matched by .gitignore line 14 (a rule written "
            f"for C object files). It was force-added; a re-fetched "
            f"copy would need `git add -f` to go back in.")
    else:
        warnings.append(
            f"This file is NOT tracked in git: *.obj is matched by .gitignore "
            f"line 14, so it exists in one copy on this machine. "
            f"Back it up, or add it with `git add -f {shape.relpath}`.")

    return FetchResult(
        ok=True, kind="damit_shape", values=values, provenance=provenance,
        from_cache=True, paths={"shape": str(path)}, warnings=tuple(warnings),
        message=(f"Mission shape model: {shape.description}. {path.name}, "
                 f"{shape.faces} facets, mean radius {shape.radius_km:g} km."))


def fetch_damit_shape(project_root, name="", naif_id="", radius_km=0.0,
                      allow_online=True, refetch=False, timeout=DAMIT_TIMEOUT_S,
                      client_factory=None, converter=None, curated=True):
    """This body's shape model: the curated registry, the cache, or DAMIT.

    Returns a :class:`FetchResult` whose ``values`` carry the shape AND the
    spin solution that came with it. Those two travel together on purpose: a
    DAMIT shape is defined in the body-fixed frame that its own pole and period
    establish, so taking the shape and keeping a different pole orients the
    model wrongly. The card offers both and flags a divergence rather than
    choosing; see :func:`spin_consistency`.

    A body in the curated registry is served from it and DAMIT is NOT
    searched. DAMIT's "no shape model for asteroid 101955" is a true answer to
    the wrong question -- Bennu has a shape model, it is a radar product, and
    DAMIT does not hold radar products -- so reporting that answer would send
    the user looking for something that was never going to be there.
    """
    name = str(name or "").strip()
    if not name:
        return FetchResult(kind="damit_shape", message=(
            "This body has no name yet, so DAMIT cannot be searched and there "
            "is nowhere to cache a shape. Fill in the Name field first."))

    if curated:
        from gui_core.curated_sources import curated_body
        entry = curated_body(name, naif_id)
        if entry is not None:
            return fetch_curated_shape(entry, project_root)

    number = _asteroid_number(naif_id, name)
    if number is None:
        return FetchResult(kind="damit_shape", message=(
            f"DAMIT is indexed by asteroid NUMBER, and neither the NAIF id "
            f"({naif_id or 'empty'}) nor the name {name!r} contains one. DAMIT "
            f"was not searched. Look the body up first, or type its number at "
            f"the start of the name."))

    try:
        body_dir = body_cache_dir(project_root, name)
    except ValueError as exc:
        return FetchResult(kind="damit_shape", message=str(exc))

    # -- cache first -------------------------------------------------------
    if not refetch:
        cached = sorted(body_dir.glob("shape_damit_*.obj"))
        if cached:
            path = cached[0]
            return _damit_shape_result(project_root, name, path,
                                       shape_provenance(path), from_cache=True)

    if not allow_online:
        return FetchResult(kind="damit_shape", message=(
            f"No DAMIT shape is cached for {name} and online fetching is "
            f"turned off, so nothing has been changed."))

    if client_factory is None:
        try:
            from damit.damit_client import DAMITClient
        except Exception as exc:                             # noqa: BLE001
            return FetchResult(kind="damit_shape", message=(
                f"The DAMIT client is not available here ({exc}), so no shape "
                f"could be fetched. Nothing has been changed."))

        def client_factory():
            # DAMIT's certificate chain has failed verification from this
            # machine before, and the alternative to not verifying is not
            # fetching at all. The content is checksummed into the sidecar
            # either way, which is the check that matters for a shape model.
            return DAMITClient(timeout=timeout, verify_ssl=False)

    try:
        client = client_factory()
        matches = client.search_asteroid(str(number))
    except Exception as exc:                                 # noqa: BLE001
        return FetchResult(kind="damit_shape", message=(
            f"DAMIT could not be reached or did not answer within {timeout} s "
            f"({type(exc).__name__}: {exc}). Nothing has been changed."))

    exact = [row for row in (matches or []) if row.get("number") == number]
    if not exact:
        if matches:
            got = ", ".join(f"({row.get('number')}) {row.get('name', '')}"
                            for row in matches[:4])
            return FetchResult(kind="damit_shape", message=(
                f"DAMIT answered, but none of its matches is asteroid "
                f"{number}: it returned {got}. The shape was NOT taken -- a "
                f"near-miss on an asteroid number is a different body, not a "
                f"close one."))
        return FetchResult(kind="damit_shape", message=(
            f"DAMIT has no shape model for asteroid {number}. Most asteroids "
            f"do not have one; use a sphere or an ellipsoid, or supply your "
            f"own shape file."))

    entry = exact[0]
    # Coerced to an int because it becomes part of a FILENAME. DAMIT's own
    # parser already produces an int, but this module accepts an injected
    # client, and a shape file named from unvalidated scraped text is a path
    # built out of somebody else's HTML.
    try:
        damit_id = int(entry.get("damit_id"))
    except (TypeError, ValueError):
        return FetchResult(kind="damit_shape", message=(
            f"DAMIT returned a model for asteroid {number} with an "
            f"unusable id ({entry.get('damit_id')!r}). Nothing has been "
            f"changed."))
    if len(exact) > 1:
        logger.info("[body-fetch] DAMIT lists %d models for %s; using %s",
                    len(exact), number, damit_id)

    try:
        model = client.get_model(damit_id)
    except Exception as exc:                                 # noqa: BLE001
        return FetchResult(kind="damit_shape", message=(
            f"DAMIT model {damit_id} could not be downloaded "
            f"({type(exc).__name__}: {exc}). Nothing has been changed."))

    if not getattr(model, "vertices", None) or not getattr(model, "faces", None):
        return FetchResult(kind="damit_shape", message=(
            f"DAMIT model {damit_id} for asteroid {number} carries no shape "
            f"data that could be read (the entry exists but its shape file "
            f"did not parse). Nothing has been changed."))

    if converter is None:
        try:
            from damit.shape_converter import ShapeConverter
            converter = ShapeConverter.damit_to_obj
        except Exception as exc:                             # noqa: BLE001
            return FetchResult(kind="damit_shape", message=(
                f"The DAMIT shape converter is not available here ({exc}). "
                f"Nothing has been changed."))

    shape_path = body_dir / f"shape_damit_{damit_id}.obj"
    try:
        radius_km = max(0.0, float(radius_km or 0.0))
    except (TypeError, ValueError):
        radius_km = 0.0
    try:
        body_dir.mkdir(parents=True, exist_ok=True)
        # A DAMIT shape model fixes the body's SHAPE, not its size: the
        # coordinates are normalized, not kilometers. Given a radius, the
        # converter scales the model to it and the mesh is in km, which is the
        # convention for asteroid .obj in this project. Given none, it is
        # written unscaled and the card says the scale must be set by hand --
        # a dimensionless mesh run as if it were km is a body the wrong size
        # by whatever the normalization happened to be.
        converter(model.vertices, model.faces, str(shape_path),
                  radius_km=radius_km, body_name=name)
    except Exception as exc:                                 # noqa: BLE001
        return FetchResult(kind="damit_shape", message=(
            f"DAMIT model {damit_id} downloaded but could not be written to "
            f"{shape_path} ({type(exc).__name__}: {exc}). Nothing has been "
            f"changed."))

    spin = getattr(model, "spin", None)
    record = {
        "source": "DAMIT",
        "damit_id": damit_id,
        "damit_asteroid_number": number,
        "damit_asteroid_name": entry.get("name", ""),
        "model_url": (f"https://astro.troja.mff.cuni.cz/projects/damit/"
                      f"asteroids/view/{damit_id}"),
        "fetched": datetime.now().isoformat(timespec="seconds"),
        "md5": _md5(shape_path),
        "bytes": shape_path.stat().st_size,
        "vertices": int(getattr(model, "num_vertices", 0) or 0),
        "faces": int(getattr(model, "num_faces", 0) or 0),
        "spin_pole_ecliptic_lon_deg": float(
            getattr(spin, "lambda_ecl_deg", 0.0) or 0.0),
        "spin_pole_ecliptic_lat_deg": float(
            getattr(spin, "beta_ecl_deg", 0.0) or 0.0),
        "spin_period_hours": float(getattr(spin, "period_hours", 0.0) or 0.0),
        "scaled_to_radius_km": radius_km,
        "gitignore_note": (
            "*.obj is ignored by this repository (.gitignore line 14, a rule "
            "for C object files). This shape is therefore UNTRACKED and exists "
            "in one copy. This sidecar is not ignored, so the model's identity "
            "survives: refetch from model_url and check md5."),
    }
    sidecar = shape_path.with_name(shape_path.stem + "_provenance.json")
    try:
        sidecar.write_text(json.dumps(record, indent=2))
    except OSError as exc:
        logger.warning("[body-fetch] shape provenance not written (%s)", exc)

    _update_index(project_root, name, {
        "has_damit_shape": True,
        "damit_id": damit_id,
        "shape_path": str(shape_path),
        "shape_md5": record["md5"],
    })
    return _damit_shape_result(project_root, name, shape_path, record,
                               from_cache=False)


def _damit_shape_result(project_root, name, shape_path, record, from_cache):
    values = {
        "generic_body.shape.type": "custom_mesh",
        "generic_body.shape.mesh_file": str(shape_path),
    }
    damit_id = record.get("damit_id", "?")
    label = f"DAMIT model {damit_id}"
    provenance = {key: label for key in values}

    warnings = [
        f"This shape is UNTRACKED: {shape_path.name} matches the *.obj line in "
        f"this project's .gitignore, which was written for C object files and "
        f"swallows every asteroid shape model here. It exists in ONE copy. Its "
        f"md5 ({record.get('md5', 'unknown')}), DAMIT id and source URL are "
        f"recorded in {shape_path.stem}_provenance.json, which is NOT ignored, "
        f"so it can be refetched and checked. To keep the file itself in the "
        f"repository: git add -f {shape_path}",
    ]

    scaled = float(record.get("scaled_to_radius_km") or 0.0)
    if scaled > 0:
        values["generic_body.shape.mesh_scale"] = 1.0
        provenance["generic_body.shape.mesh_scale"] = label
        warnings.append(
            f"A DAMIT shape model is dimensionless -- it fixes the body's "
            f"SHAPE, not its size -- so it was scaled to a mean radius of "
            f"{scaled:g} km on the way in and is now in KILOMETERS, the "
            f"convention for asteroid shape files here. The shape-model scale "
            f"is therefore 1. Change the radius and the model must be "
            f"refetched, or the scale set by hand.")
    else:
        warnings.append(
            "A DAMIT shape model is dimensionless -- it fixes the body's "
            "SHAPE, not its size -- and this one was written UNSCALED because "
            "no radius was known when it was fetched. Set the shape-model "
            "scale so the model's mean radius comes out in kilometers, or the "
            "body will be whatever size the normalization happened to give "
            "and every flux on it will be wrong.")

    period = float(record.get("spin_period_hours") or 0.0)
    lon = record.get("spin_pole_ecliptic_lon_deg")
    lat = record.get("spin_pole_ecliptic_lat_deg")
    if period > 0:
        values["generic_body.spin.period_hours"] = period
        provenance["generic_body.spin.period_hours"] = label
    if lon is not None and lat is not None and (lon or lat):
        from gui_core.body_properties import ecliptic_pole_to_equatorial
        ra, dec = ecliptic_pole_to_equatorial(lon, lat)
        values["generic_body.spin.pole_ra_deg"] = round(ra, 3)
        values["generic_body.spin.pole_dec_deg"] = round(dec, 3)
        provenance["generic_body.spin.pole_ra_deg"] = label
        provenance["generic_body.spin.pole_dec_deg"] = label
        warnings.append(
            f"The pole came with the model and is offered with it: DAMIT "
            f"publishes it as ecliptic (lambda, beta) = ({lon:g}, {lat:g}), "
            f"converted here to equatorial RA/Dec = ({ra:.3f}, {dec:.3f}), "
            f"which is what generic_body.spin takes. A DAMIT shape is defined "
            f"in the frame its own pole and period establish, so keeping the "
            f"shape and a different pole orients the body wrongly.")

    where = "was already cached" if from_cache else "was downloaded"
    return FetchResult(
        ok=True, kind="damit_shape", values=values, provenance=provenance,
        from_cache=from_cache, warnings=warnings,
        paths={"shape": str(shape_path),
               "provenance": str(shape_path.with_name(
                   shape_path.stem + "_provenance.json"))},
        message=(f"DAMIT model {damit_id} {where}: {shape_path.name}, "
                 f"{record.get('vertices', '?')} vertices and "
                 f"{record.get('faces', '?')} faces."))


# ---------------------------------------------------------------------------
# 3. Physical properties, from SBDB
# ---------------------------------------------------------------------------

def fetch_sbdb_properties(query, allow_online=True, timeout=20,
                          client_factory=None):
    """Diameter, p_V, H, G, rotation period and pole, from JPL's SBDB.

    The ONE conversion that matters is here rather than in the writer: SBDB's
    ``albedo`` is the visual GEOMETRIC albedo p_V, and
    ``generic_body.surface.albedo`` is a BOND albedo. See
    :mod:`gui_core.body_properties` for the arithmetic and for what the error
    costs; the short version is that a bright body would absorb a quarter less
    sunlight than it does, and nothing downstream would say so.

    Thermal inertia is deliberately absent. No database publishes it the way
    SBDB publishes a diameter, and a value copied from one thermophysical fit
    into a different model is not a measurement.
    """
    from gui_core.body_properties import bond_albedo

    query = str(query or "").strip()
    if not query:
        return FetchResult(kind="sbdb", message=(
            "Type a name, a number or a designation first."))
    if not allow_online:
        return FetchResult(kind="sbdb", message=(
            "Online fetching is turned off, so SBDB was not asked and nothing "
            "has been changed."))

    if client_factory is None:
        try:
            from horizons.horizons_client import HorizonsClient
        except Exception as exc:                             # noqa: BLE001
            return FetchResult(kind="sbdb", message=(
                f"The SBDB client is not available here ({exc}). Nothing has "
                f"been changed."))

        def client_factory():
            return HorizonsClient(timeout=timeout)

    try:
        physical = client_factory().get_physical_params(query)
    except Exception as exc:                                 # noqa: BLE001
        return FetchResult(kind="sbdb", message=(
            f"SBDB could not be reached or did not answer within {timeout} s "
            f"({type(exc).__name__}: {exc}). Nothing has been changed."))
    if physical is None:
        return FetchResult(kind="sbdb", message=(
            f"SBDB has no record matching {query!r}. Nothing has been "
            f"changed."))

    values, provenance, warnings, filled, missing = {}, {}, [], [], []

    diameter = float(getattr(physical, "diameter_km", 0.0) or 0.0)
    if diameter > 0:
        # SBDB publishes a DIAMETER and the engine takes a RADIUS. A factor of
        # two in every flux on the body, and nothing downstream flags it.
        values["generic_body.shape.radius_km"] = diameter / 2.0
        provenance["generic_body.shape.radius_km"] = "SBDB"
        filled.append(f"radius {diameter / 2.0:g} km (SBDB diameter "
                      f"{diameter:g} km / 2)")
    else:
        missing.append("diameter")

    period = float(getattr(physical, "rotation_period_hours", 0.0) or 0.0)
    if period > 0:
        values["generic_body.spin.period_hours"] = period
        provenance["generic_body.spin.period_hours"] = "SBDB"
        filled.append(f"rotation period {period:g} h")
    else:
        missing.append("rotation period")

    ra = float(getattr(physical, "pole_ra_deg", -1.0) or -1.0)
    dec = float(getattr(physical, "pole_dec_deg", -1.0) or -1.0)
    if ra >= 0.0 and -90.0 <= dec <= 90.0 and dec != -1.0:
        values["generic_body.spin.pole_ra_deg"] = round(ra, 3)
        values["generic_body.spin.pole_dec_deg"] = round(dec, 3)
        provenance["generic_body.spin.pole_ra_deg"] = "SBDB"
        provenance["generic_body.spin.pole_dec_deg"] = "SBDB"
        filled.append(f"spin pole RA/Dec {ra:.3f}/{dec:.3f} deg")
        warnings.append(
            f"The spin pole RA/Dec ({ra:.3f}, {dec:.3f}) is SBDB's published "
            f"pair, used as published: SBDB gives the pole in EQUATORIAL "
            f"J2000, the same frame generic_body.spin takes. (A DAMIT shape "
            f"model's pole is ECLIPTIC and is converted on the way in; the two "
            f"sources are not in the same frame, so a pole from one cannot be "
            f"compared with a pole from the other without converting.)")
    else:
        missing.append("spin pole")

    p_v = float(getattr(physical, "albedo", -1.0) or -1.0)
    slope = getattr(physical, "slope_parameter_g", None)
    conversion = bond_albedo(p_v, slope)
    if conversion is not None:
        values["generic_body.surface.albedo"] = round(conversion.bond, 6)
        provenance["generic_body.surface.albedo"] = (
            f"SBDB p_V={conversion.geometric_albedo:g}, "
            f"{'G assumed' if conversion.slope_assumed else 'G'}="
            f"{conversion.slope_parameter:g}")
        filled.append(f"Bond albedo {conversion.bond:.6g}")
        warnings.append(conversion.detail())
    else:
        missing.append("geometric albedo")

    magnitude = float(getattr(physical, "absolute_magnitude_h", 0.0) or 0.0)

    if not values:
        return FetchResult(kind="sbdb", message=(
            f"SBDB answered for {query!r} but publishes none of the values "
            f"this card uses (diameter, geometric albedo, rotation period, "
            f"pole). Nothing has been changed; fill the fields in by hand."))

    note = ""
    if missing:
        note = (" SBDB has no " + ", ".join(missing) + " for this body; "
                "those field(s) are unchanged and still need a value.")
    extra = ""
    if magnitude:
        extra = f" (H = {magnitude:g})"

    warnings.append(
        "Thermal inertia is NOT filled in from any database, because none "
        "publishes it the way SBDB publishes a diameter -- the values that "
        "exist are individual thermophysical-model fits with their own "
        "roughness and shape assumptions. Set ks, rhos and cp0 yourself; the "
        "inertia they imply is shown under those fields.")

    return FetchResult(
        ok=True, kind="sbdb", values=values, provenance=provenance,
        warnings=warnings,
        message=(f"SBDB{extra}: filled in {', '.join(filled)}.{note}"))


# ---------------------------------------------------------------------------
# 4. Keeping the spin and the shape describing the same body
# ---------------------------------------------------------------------------

#: How far a period may differ before it is called a divergence, as a
#: FRACTION. Different techniques on the same object routinely disagree in the
#: last digit or two of a period; 1 percent is far outside that and is the
#: scale at which two solutions are two solutions.
PERIOD_TOLERANCE = 0.01

#: How far a pole may differ before it is called a divergence, in degrees of
#: arc between the two directions. Pole solutions are quoted to a few degrees
#: at best, so 10 degrees is comfortably outside agreement and comfortably
#: inside "these are different poles".
POLE_TOLERANCE_DEG = 10.0


def _angle_between(ra1, dec1, ra2, dec2):
    import math
    a1, d1 = math.radians(ra1), math.radians(dec1)
    a2, d2 = math.radians(ra2), math.radians(dec2)
    cosine = (math.sin(d1) * math.sin(d2)
              + math.cos(d1) * math.cos(d2) * math.cos(a1 - a2))
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def spin_consistency(config, shape_values):
    """Does the configured spin agree with the one the shape model came with?

    ``shape_values`` is a :class:`FetchResult`'s ``values`` from a DAMIT fetch.
    Returns a message, or "" when they agree or when there is nothing to
    compare.

    This is a FLAG, not a correction. Taking a DAMIT shape while keeping a pole
    from somewhere else is a legitimate thing to do deliberately -- and a
    silent disaster to do accidentally, because the shape is defined in the
    frame its own pole establishes and nothing downstream re-checks it.
    """
    spin = ((config or {}).get("generic_body") or {}).get("spin") or {}

    def number(key, fallback=None):
        try:
            value = float(spin.get(key))
        except (TypeError, ValueError):
            return fallback
        return fallback if value != value else value

    lines = []
    model_period = shape_values.get("generic_body.spin.period_hours")
    current_period = number("period_hours")
    if (model_period and current_period and current_period > 0
            and abs(current_period - model_period)
            > PERIOD_TOLERANCE * abs(model_period)):
        lines.append(
            f"the rotation period on this card is {current_period:g} h and the "
            f"shape model's own is {model_period:g} h "
            f"({100 * abs(current_period - model_period) / abs(model_period):.1f}"
            f" percent apart)")

    model_ra = shape_values.get("generic_body.spin.pole_ra_deg")
    model_dec = shape_values.get("generic_body.spin.pole_dec_deg")
    current_ra = number("pole_ra_deg")
    current_dec = number("pole_dec_deg")
    if (model_ra is not None and model_dec is not None
            and current_ra is not None and current_dec is not None
            and current_ra >= 0.0 and abs(current_dec) <= 90.0):
        separation = _angle_between(current_ra, current_dec,
                                    model_ra, model_dec)
        if separation > POLE_TOLERANCE_DEG:
            lines.append(
                f"the spin pole on this card is RA/Dec "
                f"{current_ra:g}/{current_dec:g} and the shape model's own is "
                f"{model_ra:g}/{model_dec:g}, {separation:.1f} degrees apart")

    if not lines:
        return ""
    return ("SPIN AND SHAPE DISAGREE: " + "; ".join(lines) + ". A DAMIT shape "
            "model is defined in the body-fixed frame that its own pole and "
            "period establish, so using the shape with a different spin state "
            "points the model in the wrong direction -- and no output would "
            "show it. Take the model's spin with its shape, or keep your own "
            "spin deliberately and use a shape that goes with it.")
