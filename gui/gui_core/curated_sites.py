"""Lunar sites whose mesh, occluder and view-factor matrix are a PROVEN set.

This is the second tier of the curated registry. ``curated_sources.py`` covers
bodies whose EPHEMERIS and SHAPE a live service correctly refuses to serve
(Bennu). This module covers the other half of the same problem on the Moon: a
site whose terrain mesh, shadow-only occluder ring and view-factor matrix only
mean anything TOGETHER, and where picking two of the three by hand produces a
run that completes and is wrong.

**The shipped table is empty, and that is deliberate.** The lunar polar sites
this registry was built from are landing-site information that is not published
with PTM3D, so the entries live OUTSIDE the repository in a local overlay --
see "The local overlay" below. Everything else in this module ships: the
classes, the disk checks, the wiring, and the reasons. A machine with no
overlay gets an empty table and a GUI that offers no curated lunar sites, which
is the correct behavior for a checkout that has no site artifacts anyway.

Why a registry rather than three file pickers
---------------------------------------------

The lunar polar campaign this module came from is the sharpest case in this
project's history. Every one of the following is a real, measured, silent
failure from that campaign, and a file picker offers all of them:

* **The occluder ring is not one ring.** A site can have three
  (a 20 km, a 27 km and an 82 km outer radius). Its production runs used the
  27 km one as ``mesh.occluder_file``; its view-factor SWEEP used the 82 km one
  as a ray blocker. Those are different jobs -- run-time shadow BVH versus
  offline Monte-Carlo occlusion -- and the bigger ring is NOT the better choice
  for the first. Nothing on disk says which is which.
* **A matrix can be EMPTY and load clean.** One site's June
  ``view_factors.hmat`` has ``nnz=0``: dimensions match, the loader is happy,
  radiation exchange silently computes nothing (project map, "Trap
  directories").
* **A matrix can be for the wrong mesh and still be the right SIZE.** One core
  DEM was regenerated after its own matrix: 2,397,890 faces against the
  matrix's 1,796,406. That one is caught, by the count. Two matrices that agree
  on the count are not.
* **A matrix can be a pruned experiment whose own receipt records failure.**
  Three sit in the same directory as one good one, and one of them says in its
  receipt: "MEASURED FAIL: PSR-floor Tmax mean 3.3 K / max 22.6 K cold ...
  Diffuse polar terrain does NOT prune safely."
* **Rebuilding, instead of loading, is the wedge.** A multi-gigabyte matrix
  rebuilt on a full core+ring lunar mesh hard-wedged the graphics cards on
  2026-07-03 and needed a reboot. That is what PTM-19 exists to prevent, and a
  wrong or missing cache path is how it happens by accident.

So an entry here is not a convenience. Each is a set that RAN, to completion,
in a campaign that was checked afterwards.

The three rules, the same as the Bennu tier's
---------------------------------------------

**Nothing is invented.** Every face count, nonzero count and geometry hash in
an entry is read off the file it describes, and
``src/python/tests/test_gui_curated_sites.py`` re-reads them. A number that
drifts from its file fails the suite rather than being believed. That check now
runs over whatever the overlay supplies, so an overlay that has gone stale
against its own disk fails in exactly the same way a hard-coded table did.

**Being curated never means being quiet.** A curated site says which matrix
generation it wires, which SUPERSEDED matrices sit in the same directories, and
what it refuses to point at and why.

**Local-only entries degrade, they do not pretend.** The artifacts are hundreds
of gigabytes of DEM-derived meshes and matrices on one machine, and are not
distributed with PTM3D. Every entry is marked ``local_only`` and its status is
computed from the disk in front of the user, so a checkout somewhere else shows
the entry, says the artifacts are not here, and refuses to wire it. It does not
write paths that do not exist and let the launch check find out.

The local overlay
-----------------

``~/.ptm3d/curated_sites_local.json`` -- beside ``machine_profile.json``, which
is the convention this project already had for "facts about THIS computer that
are not the repository's business". Override the path with
``PTM3D_CURATED_SITES_LOCAL``; point it at ``/dev/null``-ish (a nonexistent
path) to run as a fresh checkout would.

JSON rather than an importable Python file on purpose: the overlay is read at
GUI start from a location outside the repository, and a data file cannot become
an execution path. The schema is one version string plus the same fields the
classes below take, so a reader can see what it is without running it.

An overlay that is present but unreadable, or malformed, or written to a schema
this build does not know, is a LOUD failure and not an empty table:
:func:`local_registry_status` records it, the loader logs a warning, and the
Moon card shows the message. The one thing this must never do is look exactly
like "you have no overlay".

No Qt, so the whole tier is measurable without a window.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = [
    "BUILTIN_CURATED_SITES",
    "CURATED_SITES",
    "CuratedSite",
    "CuratedSiteMesh",
    "CuratedSiteOccluder",
    "CuratedSiteViewFactors",
    "DEFECTIVE_VIEW_FACTORS",
    "LOCAL_REGISTRY_ENV",
    "LOCAL_REGISTRY_SCHEMA",
    "LocalRegistry",
    "NOT_CURATED",
    "SiteStatus",
    "SupersededArtifact",
    "curated_site",
    "curated_sites_for_body",
    "default_local_registry_path",
    "load_local_registry",
    "local_registry_status",
    "reload_registry",
    "site_from_mapping",
    "site_status",
    "site_wiring",
    "wiring_provenance",
]

#: The placement sentinel. ``mesh.latitude_override > -900.0`` makes the engine
#: ROTATE the loaded mesh so its vertex-mean direction lands on the given
#: latitude and longitude (``ptm_thermal_sim.cpp:15634`` ->
#: ``rotateMeshToLatLon`` at :1799). A DEM mesh is already built in the lunar
#: body frame at its own coordinates, so it must not be rotated -- see
#: :class:`CuratedSiteMesh` for what happens if it is.
NOT_PLACED = -999.0


class CuratedSiteMesh:
    """The high-resolution core terrain, and where it already sits.

    ``latitude_deg``/``longitude_deg`` are the site's coordinates FOR DISPLAY
    and for naming the run. They are deliberately NOT written into
    ``mesh.latitude_override``/``longitude_override``, and that is the least
    obvious decision in this module:

    A polar core mesh is built from a south-polar-stereographic DEM into lunar
    body-frame Cartesian coordinates, so it is already at its own latitude and
    longitude. Measured from the vertex mean of one production PLY, the mesh
    lands 0.0004 degrees from the coordinates in its own file name. Writing the
    rounded name back as an override therefore asks the engine to rotate the
    mesh by that much -- past the 1e-10 radian no-op guard, so it rotates --
    which at 1737 km is roughly a 12 m displacement.

    And only the CORE moves. ``rotateMeshToLatLon`` takes the core mesh alone;
    the occluder ring is loaded separately and is not rotated with it, so the
    two would come apart. The face AREAS are unchanged by a rotation, so the
    replacement CSR's geometry-hash gate stays green through all of it. It is
    silent by construction.

    Every production configuration for these sites leaves both keys unset for
    exactly this reason. So does this registry: it writes the ``-999.0``
    sentinel and shows the coordinates as text.
    """

    def __init__(self, path, faces, latitude_deg, longitude_deg,
                 core_radius_km, source, measured_latitude_deg=None,
                 measured_longitude_deg=None):
        self.path = str(path)
        self.faces = int(faces)
        self.latitude_deg = float(latitude_deg)
        self.longitude_deg = float(longitude_deg)
        self.core_radius_km = float(core_radius_km)
        #: How the terrain was derived. A LOLA-only fallback and an SfS product
        #: are not the same resolution class, and sites in one campaign can
        #: differ by 200x because of it.
        self.source = source
        #: The vertex-mean direction actually measured from the PLY, kept so
        #: the test can re-measure it and so the card can show the difference
        #: rather than the file name's rounding.
        self.measured_latitude_deg = (float(measured_latitude_deg)
                                      if measured_latitude_deg is not None
                                      else None)
        self.measured_longitude_deg = (float(measured_longitude_deg)
                                       if measured_longitude_deg is not None
                                       else None)

    def coordinates(self):
        ns = "S" if self.latitude_deg < 0 else "N"
        return f"{abs(self.latitude_deg):.3f}{ns}, {self.longitude_deg:.3f}E"

    def __repr__(self):                                   # pragma: no cover
        return f"<CuratedSiteMesh {Path(self.path).name}>"


class CuratedSiteOccluder:
    """The shadow-only ring, which is an ANNULUS and not a second terrain.

    A ``<SITE>_ring_Rcore<N>_Rext<M>km.ply`` has zero vertices inside about
    ``R_core`` of the site center: it pairs with the core through
    ``mesh.occluder_file`` and the two are never merged. Loading one alone
    renders a doughnut with a hole where the site is.

    ``r_ext_km`` is the OUTER radius, and the entry records which ring the
    production runs used rather than the largest one available -- see the
    module docstring.
    """

    def __init__(self, path, faces, r_ext_km, role):
        self.path = str(path)
        self.faces = int(faces)
        self.r_ext_km = float(r_ext_km)
        self.role = role

    def __repr__(self):                                   # pragma: no cover
        return f"<CuratedSiteOccluder {Path(self.path).name}>"


class CuratedSiteViewFactors:
    """The matrix pair: a replacement CSR and the base stub it needs.

    THE STUB IS NOT OPTIONAL AND IS NOT COSMETIC. The engine has no reader for
    the replacement-CSR layout in its normal cache path; the CSR arrives
    through ``PTM_HMATRIX_VF_REPLACE`` (``radiation_exchange.cpp:280``) and the
    normal loader still runs. With ``radiation.view_factor_file`` empty or
    missing, that loader finds no cache and REBUILDS -- multi-gigabyte, on a
    full core+ring lunar mesh, which is the documented compute wedge. The stub
    is a correctly-dimensioned, current-format (``HMAT``/v2), ``nnz=0`` file
    that loads clean in milliseconds so the rebuild never starts.

    It must also be an H-matrix or empty stub rather than a real ``.vf``. The
    engine warns loudly if it is not: a real loaded matrix leaves TWO matrices
    live at once, and the Gauss-Seidel and GPU-decline paths read the
    uncorrected one while the banner names the corrected one
    (``radiation_exchange.cpp:370``).

    ``generation`` names the physics, not the file format. Two matrices on one
    mesh from different generations are different physics and the outputs do
    not say so.
    """

    def __init__(self, replace_csr, base_stub, n_faces, nnz, geomhash,
                 generation, provenance, campaign, gpu_fit,
                 stub_rows=None, stub_nnz=0):
        self.replace_csr = str(replace_csr)
        self.base_stub = str(base_stub)
        self.n_faces = int(n_faces)
        self.nnz = int(nnz)
        self.geomhash = float(geomhash)
        self.generation = generation
        #: The ``.provenance.json`` sidecar's path, which is what makes this
        #: matrix's mesh identity checkable at all.
        self.provenance = str(provenance)
        #: The run that proves the set, named so a claim can be re-checked.
        self.campaign = campaign
        #: Whether this matrix fit one graphics die in production, and what
        #: happened when it did not. Measured, not predicted.
        self.gpu_fit = gpu_fit
        self.stub_rows = int(stub_rows) if stub_rows is not None else int(n_faces)
        self.stub_nnz = int(stub_nnz)

    def disk_bytes(self):
        """The CSR's size implied by its own declared layout.

        ``12 + 8*(n+1) + 8 + 8*nnz``: the header, the index array, the nonzero
        count, and an int32 column plus a float32 value per entry. Computed
        rather than stored so it cannot disagree with ``n_faces``/``nnz``.
        """
        return 12 + 8 * (self.n_faces + 1) + 8 + 8 * self.nnz

    def __repr__(self):                                   # pragma: no cover
        return f"<CuratedSiteViewFactors {Path(self.replace_csr).name}>"


class SupersededArtifact:
    """A file in the same directory that this entry deliberately does NOT wire.

    Named because the alternative is a user finding it and reasonably assuming
    the bigger/newer/nearer file is the one meant. Every one of these is real
    and sits beside the curated matrix.
    """

    def __init__(self, path, why, exists_note=""):
        self.path = str(path)
        self.why = why
        #: For a file this project DELETED on purpose, so its absence is not
        #: read as "the disk is incomplete".
        self.exists_note = exists_note

    def __repr__(self):                                   # pragma: no cover
        return f"<SupersededArtifact {Path(self.path).name}>"


class CuratedSite:
    """One row of the lunar-site table."""

    def __init__(self, key, name, body, region, mesh, occluder, view_factors,
                 why_curated, local_only=True, local_root="", aliases=(),
                 notes=(), superseded=(), use_gpu=True):
        self.key = key
        self.name = name
        self.body = body
        self.region = region
        self.mesh = mesh
        self.occluder = occluder
        self.view_factors = view_factors
        self.why_curated = why_curated
        #: True when the artifacts live outside the distribution. Every entry
        #: today is; the flag exists so a future shipped site is not forced to
        #: claim it is machine-local.
        self.local_only = bool(local_only)
        #: The directory whose absence means "this is a different machine",
        #: as distinct from "one file went missing".
        self.local_root = str(local_root)
        self.aliases = tuple(aliases)
        self.notes = tuple(notes)
        self.superseded = tuple(superseded)
        #: The route the proven campaign ran. `radiation.use_gpu: true`
        #: promotes the replacement CSR to the fused two-band GPU radiosity
        #: pool; when the matrix does not fit, the engine's own
        #: pool -> single-GPU -> CPU cascade handles it and says so in the log.
        self.use_gpu = bool(use_gpu)

    def display(self):
        return f"{self.key} — {self.region} ({self.mesh.coordinates()})"

    def __repr__(self):                                   # pragma: no cover
        return f"<CuratedSite {self.key}>"


# ---------------------------------------------------------------------------
# The shipped table, and the local overlay that fills it
# ---------------------------------------------------------------------------

#: Curated sites distributed WITH PTM3D. Empty, and not by oversight.
#:
#: The sites this registry was built from are lunar landing-site candidates,
#: which is information this project does not publish, so they are supplied by
#: the local overlay instead (module docstring, "The local overlay"). Adding an
#: entry here means asserting it can ship: its coordinates, its identifier and
#: its terrain are all publishable, and its artifacts are ones a user could
#: obtain. Nothing today meets that.
BUILTIN_CURATED_SITES = ()

#: The environment variable that relocates the overlay, for tests and for a
#: second machine profile.
LOCAL_REGISTRY_ENV = "PTM3D_CURATED_SITES_LOCAL"

#: The only schema this build reads. An overlay written to a later one is
#: refused loudly rather than parsed on a guess.
LOCAL_REGISTRY_SCHEMA = "ptm3d.curated_sites_local/1"


def default_local_registry_path():
    """``~/.ptm3d/curated_sites_local.json``, or the env override.

    Beside ``machine_profile.json``: this project already put "facts about
    THIS computer" there, and a second convention would be a second place to
    look when the table is unexpectedly empty.
    """
    override = os.environ.get(LOCAL_REGISTRY_ENV, "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / ".ptm3d" / "curated_sites_local.json"


def _require(mapping, field, where):
    if field not in mapping:
        raise ValueError(f"{where} has no {field!r}")
    return mapping[field]


def site_from_mapping(data):
    """Build a :class:`CuratedSite` from the overlay's (or a test's) mapping.

    Public because the suite builds SYNTHETIC fixture sites through it. A test
    that constructed the classes directly would exercise a different path from
    the one the overlay takes, and the overlay's parsing is the part that can
    break in the field.

    Missing required fields raise :class:`ValueError` naming the entry, rather
    than producing a half-built site: an entry that is silently short a matrix
    path is the failure mode this whole module exists to prevent.
    """
    key = str(_require(data, "key", "a curated-site entry"))
    where = f"curated site {key!r}"
    mesh = dict(_require(data, "mesh", where))
    occluder = dict(_require(data, "occluder", where))
    view_factors = dict(_require(data, "view_factors", where))
    return CuratedSite(
        key=key,
        name=str(data.get("name", key)),
        body=str(_require(data, "body", where)),
        region=str(data.get("region", "")),
        why_curated=str(data.get("why_curated", "")),
        local_only=bool(data.get("local_only", True)),
        local_root=str(data.get("local_root", "")),
        aliases=tuple(data.get("aliases", ())),
        notes=tuple(data.get("notes", ())),
        use_gpu=bool(data.get("use_gpu", True)),
        mesh=CuratedSiteMesh(
            path=_require(mesh, "path", f"{where} mesh"),
            faces=_require(mesh, "faces", f"{where} mesh"),
            latitude_deg=_require(mesh, "latitude_deg", f"{where} mesh"),
            longitude_deg=_require(mesh, "longitude_deg", f"{where} mesh"),
            core_radius_km=mesh.get("core_radius_km", 0.0),
            source=mesh.get("source", ""),
            measured_latitude_deg=mesh.get("measured_latitude_deg"),
            measured_longitude_deg=mesh.get("measured_longitude_deg"),
        ),
        occluder=CuratedSiteOccluder(
            path=_require(occluder, "path", f"{where} occluder"),
            faces=_require(occluder, "faces", f"{where} occluder"),
            r_ext_km=occluder.get("r_ext_km", 0.0),
            role=occluder.get("role", ""),
        ),
        view_factors=CuratedSiteViewFactors(
            replace_csr=_require(view_factors, "replace_csr", f"{where} matrix"),
            base_stub=_require(view_factors, "base_stub", f"{where} matrix"),
            n_faces=_require(view_factors, "n_faces", f"{where} matrix"),
            nnz=_require(view_factors, "nnz", f"{where} matrix"),
            geomhash=_require(view_factors, "geomhash", f"{where} matrix"),
            generation=view_factors.get("generation", ""),
            provenance=view_factors.get("provenance", ""),
            campaign=view_factors.get("campaign", ""),
            gpu_fit=view_factors.get("gpu_fit", ""),
            stub_rows=view_factors.get("stub_rows"),
            stub_nnz=view_factors.get("stub_nnz", 0),
        ),
        superseded=tuple(
            SupersededArtifact(path=_require(item, "path", f"{where} superseded"),
                               why=item.get("why", ""),
                               exists_note=item.get("exists_note", ""))
            for item in data.get("superseded", ())),
    )


class LocalRegistry:
    """What the overlay contributed, and what went wrong if anything did.

    ``error`` non-empty is the state this class exists for. An overlay that is
    present but unreadable must not look like an absent one: the table would be
    empty either way and only one of the two is a fault.
    """

    def __init__(self, path, present, sites=(), not_curated=None,
                 defective=None, error=""):
        self.path = str(path)
        self.present = bool(present)
        self.sites = tuple(sites)
        self.not_curated = dict(not_curated or {})
        self.defective_view_factors = dict(defective or {})
        self.error = error

    @property
    def ok(self):
        return not self.error

    def summary(self):
        if self.error:
            return (f"The local lunar-site overlay {self.path} could not be "
                    f"used: {self.error} No curated sites were loaded from it. "
                    f"This is a fault, not an empty overlay.")
        if not self.present:
            return (f"No local lunar-site overlay at {self.path}, so the "
                    f"curated lunar-site table is empty. That is the normal "
                    f"state for a checkout without local site artifacts.")
        return (f"{len(self.sites)} curated lunar site"
                f"{'' if len(self.sites) == 1 else 's'} loaded from "
                f"{self.path}.")

    def __repr__(self):                                   # pragma: no cover
        return f"<LocalRegistry n={len(self.sites)} ok={self.ok}>"


def load_local_registry(path=None):
    """Read the overlay. Never raises; a fault is reported, not thrown.

    Returns a :class:`LocalRegistry`. The GUI must start with a broken overlay
    -- refusing to open the window over a stale JSON file would be a worse
    failure than the one it is reporting -- but the fault is recorded, logged
    at WARNING, and shown on the card, so it cannot pass for "no overlay".
    """
    target = Path(path).expanduser() if path else default_local_registry_path()
    if not target.is_file():
        return LocalRegistry(target, present=False)
    try:
        payload = json.loads(target.read_text())
    except (OSError, ValueError) as exc:
        message = f"it could not be read as JSON ({exc})."
        logger.warning("[curated-sites] %s %s", target, message)
        return LocalRegistry(target, present=True, error=message)
    if not isinstance(payload, dict):
        message = "its top level is not an object."
        logger.warning("[curated-sites] %s %s", target, message)
        return LocalRegistry(target, present=True, error=message)
    schema = payload.get("schema", "")
    if schema != LOCAL_REGISTRY_SCHEMA:
        message = (f"it declares schema {schema!r} and this build reads "
                   f"{LOCAL_REGISTRY_SCHEMA!r}.")
        logger.warning("[curated-sites] %s %s", target, message)
        return LocalRegistry(target, present=True, error=message)
    try:
        sites = tuple(site_from_mapping(item)
                      for item in payload.get("sites", ()))
    except (ValueError, TypeError, KeyError) as exc:
        message = f"one of its entries is malformed ({exc})."
        logger.warning("[curated-sites] %s %s", target, message)
        return LocalRegistry(target, present=True, error=message)
    keys = [entry.key for entry in sites]
    if len(set(keys)) != len(keys):
        message = "it defines the same site key twice."
        logger.warning("[curated-sites] %s %s", target, message)
        return LocalRegistry(target, present=True, error=message)
    # Checked rather than coerced. `dict(x)` on a LIST of strings raises, and
    # this function is called at MODULE IMPORT -- so a hand-edited overlay with
    # `"not_curated": ["A", "B"]` would take the whole GUI down at startup
    # instead of reporting a fault, which is the exact opposite of what the
    # docstring above promises.
    for field in ("not_curated", "defective_view_factors"):
        value = payload.get(field, {})
        if not isinstance(value, dict):
            message = (f"its {field!r} is a {type(value).__name__} and has to "
                       f"be an object of name -> reason.")
            logger.warning("[curated-sites] %s %s", target, message)
            return LocalRegistry(target, present=True, error=message)
    registry = LocalRegistry(
        target, present=True, sites=sites,
        not_curated=payload.get("not_curated", {}),
        defective=payload.get("defective_view_factors", {}))
    logger.info("[curated-sites] %s", registry.summary())
    return registry


_LOCAL = load_local_registry()

#: The curated lunar-site table: what ships, plus whatever the overlay adds.
CURATED_SITES = BUILTIN_CURATED_SITES + _LOCAL.sites

#: Sites whose meshes exist and which are DELIBERATELY not curated, mapped to
#: the reason. Supplied by the overlay; empty without one.
#:
#: This is documentation with a test behind it, not a feature. Picking one of
#: these meshes by hand does nothing special -- there is no site entry, and
#: whatever matrices are found next to it are ranked by
#: ``gui_core.vf_discovery`` like anybody else's.
NOT_CURATED = dict(_LOCAL.not_curated)

#: Matrix FILES this project knows are defective, keyed by file name.
#:
#: The one case where a view-factor file is loadable, correctly dimensioned,
#: and known to be wrong -- so the only thing that can catch it is a record
#: like this one. ``gui_core.vf_discovery`` reads it and refuses these files
#: outright.
#:
#: Keyed by BASENAME rather than by absolute path deliberately. These matrices
#: are machine-local, so an absolute path here would be a path that exists on
#: one computer; the names are unique to their catalog and travel with the file
#: if somebody copies it. A denylist that only recognized the original location
#: would go quiet exactly when the file moved.
DEFECTIVE_VIEW_FACTORS = dict(_LOCAL.defective_view_factors)


def local_registry_status():
    """The overlay's load result, for the card and the suite."""
    return _LOCAL


def reload_registry(path=None):
    """Re-read the overlay and rebind the module tables. Returns the result.

    For tests and for a user who has just installed an overlay: the alternative
    is telling them to restart the GUI, and a registry that can only be loaded
    once is a registry nobody can check against a fixture.
    """
    global _LOCAL, CURATED_SITES, NOT_CURATED, DEFECTIVE_VIEW_FACTORS
    _LOCAL = load_local_registry(path)
    CURATED_SITES = BUILTIN_CURATED_SITES + _LOCAL.sites
    NOT_CURATED = dict(_LOCAL.not_curated)
    DEFECTIVE_VIEW_FACTORS = dict(_LOCAL.defective_view_factors)
    return _LOCAL


def _fold(text):
    return "".join(ch for ch in str(text or "").lower() if ch.isalnum())


def curated_site(name="", body=""):
    """The site entry for this name, or None.

    Matched on the site key and its aliases, case-folded and
    punctuation-insensitive, and on nothing looser. A site key is an exact
    catalog identifier; a fuzzy match here would wire one polar site's terrain
    under another's name, and two such sites are tens of kilometers apart with
    different illumination.
    """
    folded = _fold(name)
    if not folded:
        return None
    wanted_body = _fold(body)
    for entry in CURATED_SITES:
        if wanted_body and _fold(entry.body) != wanted_body:
            continue
        candidates = {_fold(entry.key), _fold(entry.name)}
        candidates.update(_fold(alias) for alias in entry.aliases)
        candidates.discard("")
        if folded in candidates:
            return entry
    return None


def curated_sites_for_body(body):
    """Every curated site on one body, in catalog order."""
    wanted = _fold(body)
    return tuple(e for e in CURATED_SITES if _fold(e.body) == wanted)


# ---------------------------------------------------------------------------
# Is this site's set actually on THIS disk?
# ---------------------------------------------------------------------------

class SiteArtifact:
    """One file of a curated set, and what the disk says about it."""

    def __init__(self, role, path, present, dataless, size_bytes,
                 expected_bytes=None, mismatch=""):
        self.role = role
        self.path = str(path)
        self.present = bool(present)
        self.dataless = bool(dataless)
        self.size_bytes = int(size_bytes)
        self.expected_bytes = expected_bytes
        #: What the file says about itself that disagrees with the registry.
        #: Non-empty means the entry and the disk have come apart, which is a
        #: harder failure than a missing file: one of them is a lie.
        self.mismatch = mismatch

    @property
    def ok(self):
        return self.present and not self.dataless and not self.mismatch

    def __repr__(self):                                   # pragma: no cover
        return f"<SiteArtifact {self.role} ok={self.ok}>"


class SiteStatus:
    """Whether a curated site can be wired on the machine in front of the user.

    ``ready`` is the only thing to branch on, and it is False for a missing
    file, for a cloud placeholder, and for a file whose own header disagrees
    with the registry. The last one matters most: a curated entry that has
    drifted from its files is worse than no entry, because it looks checked.
    """

    def __init__(self, entry, artifacts, machine_has_root):
        self.entry = entry
        self.artifacts = list(artifacts)
        #: False means "this is a different machine", which is a different
        #: message from "one of your files is missing".
        self.machine_has_root = bool(machine_has_root)
        self.missing = [a for a in self.artifacts
                        if not a.present and not a.dataless]
        self.dataless = [a for a in self.artifacts if a.dataless]
        self.mismatched = [a for a in self.artifacts if a.mismatch]

    @property
    def ready(self):
        """Everything present, readable, and consistent -- AND on this machine.

        ``machine_has_root`` is part of it rather than only a nicer message.
        A local-only entry whose working tree has been moved or renamed while
        individual files survive somewhere is a set nobody is maintaining: the
        matrices in that tree are rolled off disk on purpose as they are
        superseded, so a stale copy outliving its directory is exactly the
        state where the registry's numbers stop being a description of
        anything. Refusing is the safe direction and the message says which
        case it is.
        """
        return (bool(self.artifacts) and all(a.ok for a in self.artifacts)
                and (self.machine_has_root or not self.entry.local_only))

    def summary(self):
        """One line for the card, whichever state this is in."""
        entry = self.entry
        if not self.machine_has_root and entry.local_only:
            return (f"{entry.key}'s files are not on this computer. This site "
                    f"is curated from a local working tree "
                    f"({entry.local_root}) that PTM3D does not distribute -- "
                    f"hundreds of gigabytes of DEM-derived meshes and "
                    f"view-factor matrices. Nothing has been written to your "
                    f"configuration.")
        if self.mismatched:
            first = self.mismatched[0]
            return (f"{Path(first.path).name} disagrees with what this "
                    f"registry records about it: {first.mismatch} The entry "
                    f"and the file have come apart; the site was NOT wired.")
        if self.dataless:
            names = ", ".join(Path(a.path).name for a in self.dataless)
            return (f"{names} is a cloud placeholder — the name is on this "
                    f"disk, the bytes are not. Reading it would block the run "
                    f"while it downloads. Make it available offline first "
                    f"(Finder, right-click, 'Always Keep on This Device'). "
                    f"Nothing has been written to your configuration.")
        if self.missing:
            names = ", ".join(Path(a.path).name for a in self.missing)
            return (f"{entry.key} needs {names}, which "
                    f"{'is' if len(self.missing) == 1 else 'are'} not on this "
                    f"disk. Nothing has been written to your configuration.")
        total = sum(a.size_bytes for a in self.artifacts)
        # `generation` reads "FC-08 corrected (symmetrization divisor = ...,
        # plus back-face cull), 200,000 rays/face, assembled ...", so splitting
        # on a comma cuts INSIDE the parenthetical and leaves the line with an
        # unclosed bracket. Split before the parenthesis instead.
        generation = entry.view_factors.generation.split(" (")[0].split(",")[0]
        return (f"{entry.key}: mesh, occluder ring, base stub and the "
                f"{generation} matrix are all on this disk "
                f"({_human_bytes(total)} in total).")

    def __repr__(self):                                   # pragma: no cover
        return f"<SiteStatus {self.entry.key} ready={self.ready}>"


def _human_bytes(nbytes):
    """The other curated tier's formatter, so both read the same.

    ``curated_sources.human_bytes`` stops at GB, which is right for an 80 MB
    kernel and wrong here -- these matrices run to tens of gigabytes and one
    site's set is over 100 GB. It is extended rather than duplicated: a second
    size formatter is a second place for two cards to disagree about what
    "19.9 GB" means.
    """
    value = float(nbytes)
    if value < 1024.0 ** 3:
        from gui_core.curated_sources import human_bytes
        return human_bytes(nbytes)
    value /= 1024.0 ** 3
    for unit in ("GB", "TB", "PB"):
        if value < 1024.0 or unit == "PB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} PB"                              # pragma: no cover


def _probe(role, path, expected_bytes=None, exact=False):
    """stat only. No file here is small enough to open casually."""
    try:
        stat_result = os.stat(path)
    except OSError:
        return SiteArtifact(role, path, present=False, dataless=False,
                            size_bytes=0, expected_bytes=expected_bytes)
    dataless = stat_result.st_blocks == 0 and stat_result.st_size > 0
    mismatch = ""
    if (exact and expected_bytes is not None
            and stat_result.st_size != expected_bytes):
        mismatch = (f"the registry's face and nonzero counts imply "
                    f"{expected_bytes:,} bytes and the file is "
                    f"{stat_result.st_size:,}.")
    return SiteArtifact(role, path, present=True, dataless=dataless,
                        size_bytes=stat_result.st_size,
                        expected_bytes=expected_bytes, mismatch=mismatch)


def site_status(entry):
    """Is this curated site's whole set on this disk? Local, stat-only.

    Four artifacts, and all four are required. A set missing any one of them
    is not a partially usable set: without the occluder the site is unshadowed
    by its surroundings, without the stub the run rebuilds the matrix, and
    without the matrix there is no facet-to-facet exchange at all.
    """
    root_present = os.path.isdir(entry.local_root) if entry.local_root else True
    vf = entry.view_factors
    artifacts = [
        _probe("core mesh", entry.mesh.path),
        _probe("occluder ring", entry.occluder.path),
        _probe("base stub", vf.base_stub),
        # The CSR is the one file whose size is fully determined by two numbers
        # this registry records, so an exact check costs nothing and catches a
        # truncated copy, a partial download and a stale generation at once.
        _probe("view-factor matrix", vf.replace_csr,
               expected_bytes=vf.disk_bytes(), exact=True),
    ]
    return SiteStatus(entry, artifacts, root_present)


# ---------------------------------------------------------------------------
# Turning a selection into configuration keys
# ---------------------------------------------------------------------------

def site_wiring(entry):
    """``[(dotted key, value), ...]`` for selecting this site. Nothing else.

    Returned rather than applied so the caller can show it, and so the whole
    of what a click does is one testable list.

    The two placement keys are set to the ``-999.0`` "not placed" sentinel ON
    PURPOSE, and clearing them is part of the wiring rather than an omission:
    a leftover latitude from a previously opened configuration would rotate
    this already-placed DEM, silently, and take the core away from its ring.
    See :class:`CuratedSiteMesh`.
    """
    vf = entry.view_factors
    return [
        ("simulation.body", entry.body),
        ("mesh.import_file", entry.mesh.path),
        ("mesh.occluder_file", entry.occluder.path),
        ("mesh.latitude_override", NOT_PLACED),
        ("mesh.longitude_override", NOT_PLACED),
        ("radiation.enabled", True),
        ("radiation.view_factor_file", vf.base_stub),
        ("radiation.use_gpu", bool(entry.use_gpu)),
        ("engine_env.PTM_HMATRIX_VF_REPLACE", vf.replace_csr),
    ]


def wiring_provenance(entry):
    """The lines a card shows so a selection is never a silent one.

    Everything a user would have had to know to assemble this set by hand,
    including what was NOT chosen.
    """
    vf = entry.view_factors
    lines = [
        f"{entry.name} — {entry.mesh.coordinates()}, {entry.mesh.core_radius_km:g} km core.",
        f"Terrain: {Path(entry.mesh.path).name} — {entry.mesh.faces:,} faces "
        f"({entry.mesh.source}).",
        f"Occluder: {Path(entry.occluder.path).name} — {entry.occluder.faces:,} "
        f"faces, outer radius {entry.occluder.r_ext_km:g} km, "
        f"{entry.occluder.role}.",
        f"View factors: {Path(vf.replace_csr).name} — {vf.nnz:,} nonzeros, "
        f"{vf.generation}. Loaded through PTM_HMATRIX_VF_REPLACE; the engine "
        f"recomputes this mesh's geometry hash at load and aborts if it "
        f"disagrees.",
        f"Base stub: {Path(vf.base_stub).name} — an empty {vf.stub_rows:,}-row "
        f"H-matrix so the normal loader finds a cache and never starts a "
        f"multi-gigabyte rebuild.",
        f"Route: {vf.gpu_fit}.",
        f"Proven by: {vf.campaign}.",
        "Placement: mesh.latitude_override and mesh.longitude_override are "
        "set to -999.0 (not placed). This DEM is already in the lunar body "
        "frame at its own coordinates; an override would rotate it -- and "
        "would rotate the core WITHOUT the ring.",
    ]
    for note in entry.notes:
        lines.append(note)
    for item in entry.superseded:
        suffix = f" ({item.exists_note})" if item.exists_note else ""
        lines.append(f"NOT used: {Path(item.path).name}{suffix} — {item.why}")
    if entry.local_only:
        lines.append(
            f"Local only: these files live in {entry.local_root}, which PTM3D "
            f"does not distribute. On another machine this entry is shown and "
            f"refuses rather than writing paths that are not there.")
    return lines
