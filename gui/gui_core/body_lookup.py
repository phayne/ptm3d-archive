"""Resolve a small body's name or designation to parameters a run can use.

The Asteroid card asks for a name, a NAIF id, a radius, a spin period, a pole
and six orbital elements. Every one of those is a number somebody has to find
and retype, and a retyped orbital element is a silent error -- there is no
range check that can tell 2.46093257 AU from 2.46903257 AU.

This module answers "what is this body" from two sources, in this order:

1. **The local cache**, ``data/horizons_cache/bodies/*/body_config.yaml``. It
   is authoritative and it works with no network.
2. **JPL HORIZONS**, through the client this repository already has
   (``src/python/horizons/horizons_client.py``), only if the cache misses and
   only when the caller allows it.

Three rules, and each of them is here because of something that has already
gone wrong in this project:

* **The cache is read from DISK, not from its index.**
  ``data/horizons_cache/index.json`` lists 3 bodies while ``bodies/`` holds 7
  with a ``body_config.yaml``, and the paths it does carry are absolute paths
  into two OTHER checkouts (``~/projects/ptm-justitia-horizons-2026-08-15``
  and the OneDrive tree). ``DataManager.list_cached()`` reads only the index,
  so it reports fewer than half the bodies actually cached here and can hand
  back a path that does not exist on this machine. This module globs the
  directory and treats the index as supplementary metadata.

* **Not found, offline and timed out are three different answers**, and none
  of them is "nothing happened". Each returns a message naming which one it
  was and what to do next.

* **An SPK is only useful over the window it covers.** When a lookup finds a
  kernel, the result carries the coverage so the card can say whether it
  spans the run -- the engine substitutes Keplerian elements for any epoch
  outside the kernel and logs NOTHING (PTM-145: fourteen of fifteen Justitia
  configurations "passed" that way).

No Qt here, so it can be measured without a window.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = [
    "CACHE_RELPATH",
    "BodyLookupResult",
    "cache_entries",
    "lookup_body",
]

#: Where the cache lives, relative to the project root.
CACHE_RELPATH = Path("data") / "horizons_cache"

#: How long a HORIZONS request may take before this gives up and says so.
#: A card that hangs on a dead network is worse than a card that says the
#: network is dead, so there is a timeout and it is short.
ONLINE_TIMEOUT_S = 20

#: `generic_body:` sub-keys copied verbatim out of a cached body_config.yaml.
#: Everything the card and the Body page show, and nothing else -- a cached
#: file may carry keys this schema has no control for, and copying those in
#: would put values in the configuration that nothing displays.
_COPIED = (
    "name", "naif_id", "ephemeris_source", "spice_metakernel",
    "shape.type", "shape.radius_km", "shape.mesh_file", "shape.mesh_scale",
    "shape.semi_axis_a_km", "shape.semi_axis_b_km", "shape.semi_axis_c_km",
    "orbit.semi_major_axis_au", "orbit.eccentricity", "orbit.inclination_deg",
    "orbit.lon_ascending_node_deg", "orbit.arg_perihelion_deg",
    "orbit.mean_anomaly_deg", "orbit.epoch",
    "spin.period_hours", "spin.pole_ra_deg", "spin.pole_dec_deg",
    "spin.prime_meridian_deg",
    "surface.albedo", "surface.emissivity", "surface.ks", "surface.kd",
    "surface.rhos", "surface.rhod", "surface.H", "surface.cp0", "surface.Qb",
)


@dataclass
class BodyLookupResult:
    """What a lookup found, and where it came from.

    ``found`` is the only thing a caller should branch on. ``values`` is
    ``{dotted config key: value}``, ready to hand to
    ``ConfigManager.set_value`` one at a time -- deliberately not applied here,
    so the caller can report each one and so the fields stay the user's to
    edit afterwards.
    """

    found: bool = False
    #: True when the query named MORE THAN ONE body. Distinct from "not found",
    #: and it STOPS the lookup rather than escalating: falling through to
    #: HORIZONS would answer an ambiguous question with one confident body,
    #: which is the "type a number, silently get a different asteroid" failure
    #: this module exists to remove. "1998" matches both 13294 Rockox
    #: (1998 QO105) and 23871 Ousha (1998 RC76) in this checkout.
    ambiguous: bool = False
    #: "cache", "horizons", or "" when nothing was found.
    source: str = ""
    name: str = ""
    naif_id: str = ""
    values: dict = field(default_factory=dict)
    #: Absolute path to a kernel this body ships, or "".
    kernel: str = ""
    #: What to show the user. Always non-empty.
    message: str = ""

    def provenance(self):
        """The one line the card puts under the name field."""
        if not self.found:
            return "Parameters entered by hand."
        if self.source == "cache":
            return (f"Parameters from the local cache: {self.name}"
                    + (f", NAIF {self.naif_id}" if self.naif_id else "")
                    + ". Edit any of them; nothing here is locked.")
        return (f"Parameters from HORIZONS: {self.name}"
                + (f", NAIF {self.naif_id}" if self.naif_id else "")
                + ". Edit any of them; nothing here is locked.")


# ---------------------------------------------------------------------------
# The local cache
# ---------------------------------------------------------------------------

def _index(cache_dir):
    """``index.json`` as a dict, or {} -- never raising over a broken file."""
    path = Path(cache_dir) / "index.json"
    try:
        data = json.loads(path.read_text())
    except Exception as exc:                                 # noqa: BLE001
        if path.exists():
            logger.warning("[body-lookup] %s could not be read (%s); the "
                           "cache directory is used on its own.", path, exc)
        return {}
    return data if isinstance(data, dict) else {}


def cache_entries(project_root):
    """Every body cached on THIS machine, read from the directory.

    Returns a list of ``(display name, body_config path, generic_body dict)``,
    sorted by name. The index is consulted only for a NAIF id the config file
    does not carry itself.
    """
    import yaml

    cache_dir = Path(project_root) / CACHE_RELPATH
    bodies = cache_dir / "bodies"
    if not bodies.is_dir():
        return []
    index = _index(cache_dir)
    by_name = {str(k).strip().lower(): v for k, v in index.items()
               if isinstance(v, dict)}

    out = []
    for config_path in sorted(bodies.glob("*/body_config.yaml")):
        try:
            data = yaml.safe_load(config_path.read_text()) or {}
        except Exception as exc:                             # noqa: BLE001
            logger.warning("[body-lookup] %s is not readable YAML (%s); "
                           "skipped.", config_path, exc)
            continue
        generic = data.get("generic_body")
        if not isinstance(generic, dict):
            logger.warning("[body-lookup] %s carries no generic_body: "
                           "section; skipped.", config_path)
            continue
        name = str(generic.get("name") or config_path.parent.name).strip()
        generic = dict(generic)
        if not generic.get("naif_id"):
            extra = by_name.get(name.lower(), {})
            if extra.get("naif_id"):
                generic["naif_id"] = extra["naif_id"]
        out.append((name, config_path, generic))
    return sorted(out, key=lambda row: row[0].lower())


def _tokens(text):
    """Lower-case word/number tokens of a body name or designation."""
    keep = []
    word = []
    for char in str(text or ""):
        if char.isalnum():
            word.append(char.lower())
        elif word:
            keep.append("".join(word))
            word = []
    if word:
        keep.append("".join(word))
    return keep


def _matches(query, name, naif_id):
    """Does `query` name this body?

    Token-based rather than substring, so "623" matches
    "623 Chimaera (A907 BC)" and does NOT match "13294 Rockox" or "88055 Ghaf"
    -- a substring test on "623" hits neither of those but does hit "1623",
    and a user typing an asteroid number and silently getting a different
    asteroid is the whole failure mode this module is meant to remove.
    """
    wanted = _tokens(query)
    if not wanted:
        return False
    have = set(_tokens(name)) | set(_tokens(naif_id))
    # The NAIF id of a numbered asteroid is 20000000 + number, so a user who
    # types the asteroid number should match a config that only records the
    # eight-digit form.
    for token in list(have):
        if token.isdigit() and len(token) == 8 and token.startswith("2000"):
            have.add(str(int(token) - 20000000))
    return all(token in have for token in wanted)


def _from_cached_generic(generic):
    """`generic_body:` mapping -> ``{dotted config key: value}``."""
    values = {}
    for suffix in _COPIED:
        node = generic
        for part in suffix.split("."):
            if not isinstance(node, dict) or part not in node:
                node = None
                break
            node = node[part]
        if node is None or node == "":
            continue
        values[f"generic_body.{suffix}"] = node
    # semi_axes_km is the on-disk spelling; the GUI holds three scalars.
    axes = ((generic.get("shape") or {}).get("semi_axes_km")
            if isinstance(generic.get("shape"), dict) else None)
    if isinstance(axes, (list, tuple)) and len(axes) == 3:
        for key, value in zip(("a", "b", "c"), axes):
            values[f"generic_body.shape.semi_axis_{key}_km"] = value
    return values


def _resolve_kernel(generic, config_path, project_root):
    """Absolute path to this body's metakernel or SPK, or ""."""
    for key in ("spice_metakernel", "spk_file"):
        raw = str(generic.get(key) or "").strip()
        if not raw:
            continue
        for base in (config_path.parent, Path(project_root)):
            candidate = (base / raw) if not Path(raw).is_absolute() else Path(raw)
            if candidate.exists():
                return str(candidate)
    return ""


def lookup_from_cache(query, project_root):
    """Resolve `query` against the local cache only. Never raises."""
    try:
        entries = cache_entries(project_root)
    except Exception as exc:                                 # noqa: BLE001
        logger.warning("[body-lookup] the cache could not be read (%s)", exc)
        return BodyLookupResult(
            message=f"The local cache could not be read ({exc}).")
    hits = [row for row in entries
            if _matches(query, row[0], str(row[2].get("naif_id") or ""))]
    if not hits:
        return BodyLookupResult(
            message=(f"{query!r} is not in the local cache "
                     f"({len(entries)} bodies cached)."))
    if len(hits) > 1:
        names = ", ".join(row[0] for row in hits[:6])
        return BodyLookupResult(
            ambiguous=True,
            message=(f"{query!r} matches {len(hits)} cached bodies ({names}). "
                     f"Type more of the name or its number."))
    name, config_path, generic = hits[0]
    values = _from_cached_generic(generic)
    return BodyLookupResult(
        found=True, source="cache", name=name,
        naif_id=str(generic.get("naif_id") or ""), values=values,
        kernel=_resolve_kernel(generic, config_path, project_root),
        message=(f"Found {name} in the local cache "
                 f"({config_path.parent.name}); filled in "
                 f"{len(values)} field(s)."))


# ---------------------------------------------------------------------------
# HORIZONS
# ---------------------------------------------------------------------------

def lookup_from_horizons(query, timeout=ONLINE_TIMEOUT_S, client_factory=None):
    """Resolve `query` against JPL HORIZONS. Never raises, never hangs.

    `client_factory` exists so this can be measured without a network: the
    tests pass a stub, and the check that the mapping is right is then a real
    check rather than one that is skipped whenever the machine is offline.
    """
    if client_factory is None:
        try:
            import sys
            here = Path(__file__).resolve().parents[2] / "src" / "python"
            if str(here) not in sys.path:
                sys.path.insert(0, str(here))
            from horizons.horizons_client import HorizonsClient
        except Exception as exc:                             # noqa: BLE001
            return BodyLookupResult(
                message=(f"The HORIZONS client is not available here ({exc}), "
                         f"so only the local cache could be searched."))

        def client_factory():
            return HorizonsClient(timeout=timeout)

    try:
        body = client_factory().get_body(query)
    except Exception as exc:                                 # noqa: BLE001
        # Requests raises a dozen different classes for "no network"; they are
        # all the same answer to the user, and the text of the exception is
        # the only thing that distinguishes them, so it is quoted.
        return BodyLookupResult(
            message=(f"HORIZONS could not be reached or did not answer within "
                     f"{timeout} s ({type(exc).__name__}: {exc}). Nothing has "
                     f"been changed. Fill the fields in by hand, or use the "
                     f"HORIZONS page to fetch and cache this body."))
    if body is None:
        return BodyLookupResult(
            message=(f"HORIZONS has no record matching {query!r}. Check the "
                     f"spelling, or try the asteroid's number."))

    values = {}
    name = str(getattr(body, "name", "") or query).strip()
    naif_id = str(getattr(body, "naif_id", "") or "").strip()
    values["generic_body.name"] = name
    if naif_id:
        values["generic_body.naif_id"] = naif_id

    # -1.0 and "" are this schema's "not set" sentinels. ZERO IS NOT: a
    # circular orbit really does have eccentricity 0.0, and an equatorial one
    # inclination 0.0. Discarding those left the field at -1.0 -- the sentinel
    # -- so a body HORIZONS had described exactly came out undescribed.
    # `semi_major_axis_au` is the exception: it is also the key
    # `generic_body_active()` tests with `> 0`, and a zero semi-major axis is
    # not a body.
    elements = getattr(body, "elements", None)
    for attr, key, zero_ok in (
            ("semi_major_axis_au", "orbit.semi_major_axis_au", False),
            ("eccentricity", "orbit.eccentricity", True),
            ("inclination_deg", "orbit.inclination_deg", True),
            ("lon_ascending_node_deg", "orbit.lon_ascending_node_deg", True),
            ("arg_perihelion_deg", "orbit.arg_perihelion_deg", True),
            ("mean_anomaly_deg", "orbit.mean_anomaly_deg", True),
            ("epoch", "orbit.epoch", False)):
        value = getattr(elements, attr, None)
        if value is None or value == "" or value == -1.0:
            continue
        if value == 0.0 and not zero_ok:
            continue
        values[f"generic_body.{key}"] = value

    physical = getattr(body, "physical", None)
    rotation = getattr(physical, "rotation_period_hours", None)
    if rotation not in (None, 0.0, -1.0):
        values["generic_body.spin.period_hours"] = rotation
    for attr, key in (("pole_ra_deg", "spin.pole_ra_deg"),
                      ("pole_dec_deg", "spin.pole_dec_deg")):
        value = getattr(physical, attr, None)
        if value not in (None, "", -1.0):
            values[f"generic_body.{key}"] = value
    albedo = getattr(physical, "albedo", None)
    if albedo not in (None, 0.0, -1.0):
        values["generic_body.surface.albedo"] = albedo
    diameter = getattr(physical, "diameter_km", None)
    if diameter not in (None, 0.0, -1.0):
        # HORIZONS reports a DIAMETER; the engine wants a radius. Getting this
        # wrong is a factor of two in every flux on the body and nothing
        # downstream would flag it.
        values["generic_body.shape.radius_km"] = float(diameter) / 2.0

    missing = [label for label, key in (
        ("spin period", "generic_body.spin.period_hours"),
        ("pole", "generic_body.spin.pole_ra_deg"),
        ("radius", "generic_body.shape.radius_km"),
        ("semi-major axis", "generic_body.orbit.semi_major_axis_au"),
        ("eccentricity", "generic_body.orbit.eccentricity"),
    ) if key not in values]
    note = ""
    if missing:
        note = (" HORIZONS has no " + ", ".join(missing)
                + " for this body; those field(s) are unchanged and still "
                  "need a value.")
    where = f"Found {name} on HORIZONS"
    if naif_id:
        where += f" (NAIF {naif_id})"
    return BodyLookupResult(
        found=True, source="horizons", name=name, naif_id=naif_id,
        values=values,
        message=f"{where}; filled in {len(values)} field(s).{note}")


# ---------------------------------------------------------------------------
# The one entry point
# ---------------------------------------------------------------------------

def lookup_body(query, project_root, allow_online=True,
                timeout=ONLINE_TIMEOUT_S, client_factory=None):
    """Resolve `query`: local cache first, then HORIZONS if allowed.

    Always returns a :class:`BodyLookupResult` with a non-empty ``message``,
    and never raises -- this is called from a Qt slot, where an escaping
    exception is a ``qFatal`` and takes the window down with any unsaved
    edits (measured: exit 134, SIGABRT).
    """
    query = str(query or "").strip()
    if not query:
        return BodyLookupResult(
            message="Type a name, a number or a designation to look up.")

    cached = lookup_from_cache(query, project_root)
    if cached.found:
        return cached
    if cached.ambiguous:
        # STOP. See BodyLookupResult.ambiguous: asking HORIZONS would turn
        # "which of these two did you mean" into one confident answer.
        return cached

    if not allow_online:
        return BodyLookupResult(
            message=(cached.message + " Online lookup is turned off, so "
                     "nothing else was tried."))

    online = lookup_from_horizons(query, timeout=timeout,
                                  client_factory=client_factory)
    if online.found:
        return online
    return BodyLookupResult(
        message=f"{cached.message} {online.message}")
