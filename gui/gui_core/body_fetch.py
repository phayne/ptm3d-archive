"""Fetching an asteroid's ephemeris, shape and physical properties on demand.

Three fetches, one shape. Each of them:

* **is cache-first.** The local ``data/horizons_cache`` is authoritative and
  works offline; the network is asked only when the cache misses, or when the
  caller explicitly says to refetch.
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
``spice``. That is PTM-145/PTM-300: the committed Justitia SPK covered 30 days
of 2034 while every Justitia configuration targeted 2035, and all fifteen ran
100 percent on the engine's per-epoch Keplerian fallback -- which logs nothing
at all (``illumination_model.cpp:2793-2818``), so every one of them "passed".
The engine still does not report it. Until it does, this is the guard.

**A shape model is not fetched without saying where it will live.** ``.obj``
is matched by this project's ``.gitignore`` line 14 -- a rule written for C
object files that swallows every asteroid shape model in the repository
(PTM-303: it is why a Justitia shape went missing). A fetched shape is
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
    "SPK_MARGIN_DAYS",
    "FetchResult",
    "body_cache_dir",
    "fetch_damit_shape",
    "fetch_spk",
    "fetch_sbdb_properties",
    "shape_provenance",
    "spin_consistency",
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
#: a month or a decade, and the failure it prevents is silent: PTM-300's
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
                 from_cache=False):
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


def fetch_spk(config, project_root, name="", naif_id="", body_type="asteroid",
              margin_days=SPK_MARGIN_DAYS, allow_online=True, refetch=False,
              timeout=HORIZONS_SPK_TIMEOUT_S, client_factory=None):
    """Get an SPK that spans this run, from the cache or from HORIZONS.

    Returns a :class:`FetchResult`. ``values`` sets the metakernel path and
    ``generic_body.ephemeris_source`` -- and sets the SOURCE only when the
    kernel has been READ BACK and shown to cover the whole run including
    spin-up. An unverifiable or insufficient kernel leaves the ephemeris
    source alone, so a card cannot end up saying "spice" over a kernel that
    would be silently ignored for most of the run.
    """
    name = str(name or "").strip()
    if not name:
        return FetchResult(kind="spk", message=(
            "This body has no name yet, so there is nowhere to cache a kernel. "
            "Fill in the Name field (Look up will do it) and try again."))

    earliest, latest, note = _needed_window(config)
    if earliest is None:
        return FetchResult(kind="spk", message=(
            f"The run window could not be worked out, so there is no window to "
            f"fetch a kernel for: {note}"))

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
    command = str(naif_id or name).strip()
    if body_type in ("asteroid", "comet"):
        # A bare number is a MAJOR-body NAIF id to HORIZONS; the trailing
        # semicolon is what makes it a small-body record number. Record 4 is
        # Vesta and NAIF 4 is the Mars barycenter, so this is not cosmetic.
        command = command.rstrip("; ") + ";"

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
    try:
        client_factory().download_spk(command, start_text, stop_text,
                                      str(scratch))
    except Exception as exc:                                 # noqa: BLE001
        try:
            scratch.unlink()
        except OSError:
            pass
        return FetchResult(kind="spk", message=(
            f"HORIZONS could not be reached, or refused this request, or did "
            f"not answer within {timeout} s ({type(exc).__name__}: {exc}). No "
            f"kernel was written and nothing has been changed. HORIZONS does "
            f"not build SPK files for every object; if this body is a major "
            f"planet or a satellite, use Keplerian elements or a kernel you "
            f"supply."))

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
    scratch.replace(spk_path)

    if verdict is None or not verdict:
        # BOTH refusals keep the file and index it -- it is a real kernel and
        # the user may want it -- and BOTH return no values, so the card
        # cannot set the ephemeris source over it. The two share this block on
        # purpose: they are one decision ("do not claim SPICE") with two
        # reasons, and writing it twice is how the two would drift apart.
        _write_metakernel(spk_path, metakernel_path, name)
        _update_index(project_root, name, {
            "naif_id": str(naif_id or ""), "body_type": body_type,
            "spk_path": str(spk_path), "spk_start": start_text,
            "spk_stop": stop_text,
            "metakernel_path": str(metakernel_path)
                               if metakernel_path.exists() else "",
        })

    if verdict is None:
        return FetchResult(
            kind="spk", ok=False, paths={"spk": str(spk_path)},
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
                        requested=(start_text, stop_text), body_type=body_type)


def _write_metakernel(spk_path, metakernel_path, name):
    """Write the .tm beside the kernel. Never raises.

    Reuses ``SPKGenerator``, which carries the PTM-145 fixes: ``PATH_VALUES``
    points at the shared ``data/kernels`` tree rather than at the body's own
    directory, and long paths are split with SPICE's '+' continuation so
    ``furnsh`` does not silently truncate them.
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
                 body_type="asteroid"):
    warnings = []
    have_metakernel = _write_metakernel(spk_path, metakernel_path, name)
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
        from_cache=from_cache,
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


def fetch_damit_shape(project_root, name="", naif_id="", radius_km=0.0,
                      allow_online=True, refetch=False, timeout=DAMIT_TIMEOUT_S,
                      client_factory=None, converter=None):
    """Download this body's DAMIT shape model into the cache, if one exists.

    Returns a :class:`FetchResult` whose ``values`` carry the shape AND the
    spin solution that came with it. Those two travel together on purpose: a
    DAMIT shape is defined in the body-fixed frame that its own pole and period
    establish, so taking the shape and keeping a different pole orients the
    model wrongly. The card offers both and flags a divergence rather than
    choosing; see :func:`spin_consistency`.
    """
    name = str(name or "").strip()
    if not name:
        return FetchResult(kind="damit_shape", message=(
            "This body has no name yet, so DAMIT cannot be searched and there "
            "is nowhere to cache a shape. Fill in the Name field first."))

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
