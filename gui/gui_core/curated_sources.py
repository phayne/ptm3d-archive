"""Bodies whose ephemeris and shape come from a mission, not from a service.

HORIZONS builds an SPK on demand for most small bodies, and DAMIT publishes a
lightcurve-inversion shape for a few thousand asteroids. For a handful of
bodies neither is true, and the reason is the same in both cases: the body has
been visited, so its ephemeris and its shape are MISSION PRODUCTS that live in
an archive rather than answers a service will generate.

(101955) Bennu is the case that forced this module. Asked for an SPK, HORIZONS
answers::

    SPK creation is not available for pre-computed objects in the major body
    index

-- correctly, and unhelpfully: Bennu's HORIZONS record is ``*SPECIAL*``, served
from ``ORX_merged``. Asked for a shape, DAMIT has nothing, because Bennu's
shape is not a lightcurve inversion; it was Nolan et al.'s radar model, and
since the encounter it is OLA's. Both refusals are true, both are dead ends,
and a user who does not already know where the OSIRIS-REx archive is has
nowhere to go from either.

What this module adds is a THIRD TIER, consulted first for the bodies it
covers::

    curated registry (itself cache-first) -> body cache -> live service

For a body in the table, neither of the last two is reached. That is the
point: HORIZONS' refusal for Bennu is correct and permanent, so retrying it is
not resilience, it is a slower way to reach the same message.

**The registry comes before the body cache, and that ordering is load-bearing.**
Bennu's ``data/horizons_cache/bodies/101955_Bennu_(1999_RQ36)/ephemeris.spk``
is JPL solution 76, covering 2015-2023, superseded on 2026-08-27 and kept only
for provenance. Inside that span it would pass an ordinary coverage check, so
a plain cache-first rule would wire it -- silently, and for a 2026 run it would
then fall off the end of its coverage into the engine's unlogged Keplerian
fallback, measured at 0.092 AU and up to 23.9 K wrong. The tier's OWN first
step is a cache check ("is the pinned kernel already fetched into
``data/kernels``?"), so nothing is refetched and a machine that has it reaches
a wired, coverage-verified card with no network call at all.

Three rules the table follows
-----------------------------

**Nothing is invented.** Every field below is read off an artifact that already
ships in this repository -- ``data/kernels/bennu.tm``,
``examples/quickstart_asteroid_bennu.yaml``, and the shape's own
``PROVENANCE.md``. A curated entry for a body nobody has verified would be the
same unbacked button the body-first screen exists to remove; the test suite
asserts each entry's files and citations resolve.

**The pin lives in ONE place.** The kernel URL and its sha256 are
``scripts/fetch_kernels.py``'s (``BODY_KERNELS``), imported here rather than
retyped. A pinned hash with two copies is a hash that goes stale in one of
them, silently, which is exactly what a pin is for.

**Being curated never means being quiet.** A curated body still reports which
kernel it wired, over what coverage, and from what -- and when the body cache
holds a DIFFERENT kernel that the curated one supersedes, it says so rather
than leaving two ephemerides on disk and no statement about which ran.

The other tier
--------------

``gui_core.curated_sites`` is the same idea one level down: curated SITES on a
body, whose terrain mesh, occluder ring and view-factor matrix are a set that
only means anything together. Same three rules, same local-first status, same
"say what was superseded and not used". The two tables are independent -- a
body's ephemeris and a site's geometry have nothing to do with each other --
and they are kept in separate modules for that reason.

Adding a body
-------------

See ``docs/dev/gui-curated-body-sources.md``. In short: add the kernel set to
``BODY_KERNELS`` in ``scripts/fetch_kernels.py`` with its pinned sha256, ship
the metakernel in ``data/kernels/``, add a ``CuratedBody`` below, and let
``src/python/tests/test_gui_curated_sources.py`` check that every path,
citation and coverage string in it resolves. Do not add a body whose ephemeris
and shape have not each been verified against their archive.

No Qt, so the whole tier is measurable without a window.
"""

from __future__ import annotations

import importlib.util
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = [
    "CURATED_BODIES",
    "CuratedBody",
    "CuratedEphemeris",
    "CuratedShape",
    "KernelFile",
    "KernelStatus",
    "curated_body",
    "curated_kernel_files",
    "curated_kernel_status",
    "curated_shape_path",
    "fetch_kernels_module",
    "download_curated_kernels",
    "human_bytes",
]


# ---------------------------------------------------------------------------
# 1. The table
# ---------------------------------------------------------------------------

class CuratedEphemeris:
    """Where a curated body's positions come from, and over what window.

    ``fetch_bodies`` is the ``scripts/fetch_kernels.py --bodies`` selector that
    downloads it, and it is also the KEY into that script's ``BODY_KERNELS``.
    One string doing both jobs is deliberate: the command a card tells the user
    to run and the table it reads the URL from cannot drift apart.

    ``metakernel`` is what goes into ``generic_body.spice_metakernel``. A bare
    file name, resolved under the kernel directory -- see
    ``docs/configuration.md``, "How generic_body.spice_metakernel is resolved".

    ``naif_id`` is the id INSIDE the kernel, not one computed from the
    asteroid number. HORIZONS migrated small-body SPK ids from 2000000+N to
    20000000+N and the wrong one fails with SPKINSUFFDATA; Bennu's kernel
    declares 2101955, which is the older convention, and computing 20101955
    from 101955 would miss it.
    """

    def __init__(self, fetch_bodies, metakernel, spk_filename, naif_id,
                 coverage_start, coverage_end, description, short, reference,
                 approx_bytes):
        self.fetch_bodies = fetch_bodies
        #: The body's OWN ephemeris file among the metakernel's members. The
        #: metakernel also loads shared kernels (leap seconds, planetary
        #: constants, de440s), and ``approx_bytes`` is this file's size alone
        #: -- so the size disclosure has to know which member it describes.
        self.spk_filename = spk_filename
        self.metakernel = metakernel
        self.naif_id = str(naif_id)
        self.coverage_start = coverage_start
        self.coverage_end = coverage_end
        self.description = description
        #: The button's version. A button is 180 px at its narrowest and
        #: `test_gui_button_text_fits.py` measures against that, so the full
        #: description ("JPL solution 118, OSIRIS-REx reconstruction,
        #: 1900-2135") belongs on the card's note line, not inside a control.
        self.short = short
        self.reference = reference
        #: For DISCLOSURE ONLY -- what the card says before downloading. The
        #: authority on the bytes is the pinned sha256 in
        #: ``scripts/fetch_kernels.py``, checked by the downloader itself; this
        #: number is never used to decide whether a file is good.
        self.approx_bytes = int(approx_bytes)

    def coverage_text(self):
        return f"{self.coverage_start} to {self.coverage_end}"

    def __repr__(self):                                   # pragma: no cover
        return f"<CuratedEphemeris {self.metakernel}>"


class CuratedShape:
    """A shape model that ships with the repository, and where it came from.

    ``relpath`` is repo-relative because that is what the configuration holds
    (``examples/quickstart_asteroid_bennu.yaml`` writes exactly this string
    into ``generic_body.shape.mesh_file``), and because the engine resolves it
    through the same ``PTM_DATA_ROOT`` contract as ``mesh.import_file``.

    ``tracked`` records whether the file is in git. It matters and it is not
    obvious: ``.gitignore`` carries ``*.obj``, a rule written for C object
    files that matches every asteroid shape model here (PTM-303). The curated
    body cache is excepted from it by name
    (``!data/horizons_cache/bodies/**/*.obj``), so Bennu's shape ships by RULE
    rather than by the ``git add -f`` it originally needed -- a force-add is
    invisible to the next person, and a shape that quietly stopped being
    tracked would leave this card offering a file a fresh checkout does not
    contain. A future entry whose shape is not tracked would exist in one copy
    on one machine, and the card has to be able to say which case it is.
    """

    def __init__(self, relpath, description, archive, radius_km, mesh_scale,
                 vertices, faces, units, tracked, spin_period_hours,
                 spin_pole_ra_deg, spin_pole_dec_deg, spin_reference,
                 provenance_relpath=""):
        self.relpath = relpath
        self.description = description
        self.archive = archive
        self.radius_km = float(radius_km)
        self.mesh_scale = float(mesh_scale)
        self.vertices = int(vertices)
        self.faces = int(faces)
        self.units = units
        self.tracked = bool(tracked)
        #: THE SPIN TRAVELS WITH THE SHAPE, and it is not decoration. A shape
        #: model is defined in the body-fixed frame that its own pole and
        #: period establish, so taking the mesh and keeping a pole from
        #: somewhere else points the body in the wrong direction and nothing
        #: downstream re-checks it. ``fetch_damit_shape`` has always offered
        #: the two together for that reason; the first draft of the curated
        #: path offered the mesh alone, which ALSO made
        #: ``body_fetch.spin_consistency`` return "" -- so the card
        #: affirmatively showed no disagreement while there was one.
        self.spin_period_hours = float(spin_period_hours)
        self.spin_pole_ra_deg = float(spin_pole_ra_deg)
        self.spin_pole_dec_deg = float(spin_pole_dec_deg)
        self.spin_reference = spin_reference
        self.provenance_relpath = provenance_relpath

    def label(self):
        return f"{self.description} ({self.archive})"

    def __repr__(self):                                   # pragma: no cover
        return f"<CuratedShape {self.relpath}>"


class CuratedBody:
    """One row of the table.

    ``aliases`` exist because the same body is typed four ways -- ``Bennu``,
    ``101955``, ``101955 Bennu (1999 RQ36)``, ``1999 RQ36`` -- and a registry
    that matched only the first would send three of the four to HORIZONS, to
    be refused. Matching is case-folded and punctuation-insensitive; see
    :func:`curated_body`.
    """

    def __init__(self, key, name, number, ephemeris, shape, why_curated,
                 notes=(), aliases=()):
        self.key = key
        #: The name the shipped configuration uses, so a curated card and a
        #: shipped example name the body identically.
        self.name = name
        #: The IAU number, or None.
        self.number = number
        self.ephemeris = ephemeris
        self.shape = shape
        #: Why a live service cannot serve this body. Shown to the user in
        #: place of the service's refusal, so the card explains rather than
        #: reports a dead end.
        self.why_curated = why_curated
        self.notes = tuple(notes)
        self.aliases = tuple(aliases)

    def __repr__(self):                                   # pragma: no cover
        return f"<CuratedBody {self.key}>"


#: The curated table. ONE entry today, and the structure is what makes the
#: second one a data change rather than a code change.
#:
#: Do NOT add a body on the strength of "a mission went there". Each field
#: below is a measurement or a citation from an artifact in this repository,
#: and `test_gui_curated_sources.py` checks them.
CURATED_BODIES = (
    CuratedBody(
        key="bennu",
        name="101955 Bennu (1999 RQ36)",
        number=101955,
        aliases=("bennu", "101955", "1999 RQ36", "101955 Bennu",
                 "(101955) Bennu"),
        why_curated=(
            "HORIZONS will not build an SPK for Bennu -- its record is "
            "*SPECIAL*, served from the pre-computed ORX_merged kernel, and "
            "every designation form is answered 'SPK creation is not "
            "available for pre-computed objects in the major body index' "
            "(measured 2026-08-27 against four designation forms and the POST "
            "file endpoint). DAMIT has no Bennu either: DAMIT publishes "
            "lightcurve inversions, and Bennu's shape is not one -- it is a "
            "spacecraft product, measured by the OSIRIS-REx laser altimeter "
            "and archived at the PDS Small Bodies Node. Both refusals are "
            "correct and permanent, so this body is served from the mission "
            "products instead of from either service."),
        ephemeris=CuratedEphemeris(
            fetch_bodies="bennu",
            metakernel="bennu.tm",
            spk_filename="sb-101955-118_long.bsp",
            naif_id="2101955",
            coverage_start="1900-01-01",
            coverage_end="2135-10-01",
            description=("JPL solution 118, OSIRIS-REx reconstruction, "
                         "1900-2135"),
            short="JPL solution 118",
            reference=("Farnocchia et al. (2021), "
                       "doi:10.1016/j.icarus.2021.114594"),
            approx_bytes=84648960,
        ),
        shape=CuratedShape(
            relpath=("data/horizons_cache/bodies/101955_Bennu_(1999_RQ36)/"
                     "g_06370mm_alt_obj_0000n00000_v021.obj"),
            description=("OSIRIS-REx OLA v21 global shape model, 6.37 m "
                         "ground sample distance"),
            archive="PDS Small Bodies Node, urn:nasa:pds:orex.altimetry:"
                    "data_derived_altimetry_lidar_global_models:"
                    "g_06370mm_alt_obj_0000n00000_v021.obj",
            # Mean |vertex| over the 24,578 REFERENCED vertices, 0.243722 km.
            # Not decoration: `BodyShape::meanRadiusKm` returns this verbatim
            # for a CUSTOM_MESH (`src/cpp/orbital/body_shape.h:41`), so it is
            # the R_body the engine's PTM-37 unit guard cross-checks the mesh
            # against and the radius `setBodyRadius` receives.
            radius_km=0.2437,
            mesh_scale=1.0,
            # The file's counts as the PDS4 label states them. 772 of the
            # vertices appear in no facet -- duplicates the producer flags in
            # the model's own header -- so V - E + F is 774 rather than 2 while
            # the surface is still closed (0 boundary edges, measured with
            # `scripts/validate_shape_model.py`). The archive product ships
            # BYTE FOR BYTE, orphans included, because a checksum that matches
            # the archive is worth more than a tidier vertex list.
            vertices=25350,
            faces=49152,
            units="km",
            tracked=True,
            # Hergenrother et al. (2013), the same PDS archive as the shape;
            # published as ecliptic (lambda, beta) = (45, -88) and carried
            # here in the equatorial RA/Dec that `generic_body.spin` takes,
            # exactly as examples/quickstart_asteroid_bennu.yaml records it.
            # An obliquity of ~178 deg: Bennu is a retrograde rotator.
            spin_period_hours=4.29746,
            spin_pole_ra_deg=86.639,
            spin_pole_dec_deg=-65.109,
            spin_reference=("Hergenrother et al. (2013); pole published as "
                            "ecliptic (lambda, beta) = (45, -88) and "
                            "converted to equatorial RA/Dec"),
            provenance_relpath=("data/horizons_cache/bodies/"
                                "101955_Bennu_(1999_RQ36)/PROVENANCE.md"),
        ),
        notes=(
            "The body cache also holds an older ephemeris.spk for Bennu -- "
            "JPL solution 76 (2013), covering 2015-01-01 to 2023-05-31 only. "
            "It is kept for provenance and is NOT what a run should use: at "
            "2026-01-01 it has no data, and the engine's silent per-epoch "
            "Keplerian fallback was then wrong by 0.092 AU in heliocentric "
            "distance and up to 23.9 K in surface temperature (measured, see "
            "that directory's PROVENANCE.md).",
        ),
    ),
)


def _fold(text):
    """A body name reduced to what a match should ignore.

    Case, spaces, parentheses and underscores. ``(101955) Bennu`` and
    ``101955_Bennu`` and ``bennu`` all fold to something a lookup can compare;
    a designation's own characters (``1999 RQ36`` -> ``1999rq36``) survive.
    """
    return "".join(ch for ch in str(text or "").lower()
                   if ch.isalnum())


def curated_body(name="", naif_id=""):
    """The registry entry for this body, or None.

    Matched on the NAME the user typed and on the NAIF id, in that order, and
    on nothing else. In particular NOT on a leading number alone unless that
    number is the body's own: an asteroid number is the one identifier that
    cannot be confused, and a fuzzy match here would silently serve one
    asteroid's kernel for another's card -- which is the DAMIT near-miss
    failure (``fetch_damit_shape`` refuses those) with a bigger blast radius.
    """
    folded = _fold(name)
    identifier = str(naif_id or "").strip().rstrip("; ").strip()
    for entry in CURATED_BODIES:
        candidates = {_fold(entry.name), _fold(entry.key)}
        candidates.update(_fold(alias) for alias in entry.aliases)
        if entry.number is not None:
            candidates.add(_fold(str(entry.number)))
        candidates.discard("")
        if folded and folded in candidates:
            return entry
        if identifier and identifier == entry.ephemeris.naif_id:
            return entry
        # The other small-body id convention for the same body, so a card
        # carrying 20101955 from a lookup still resolves. Both forms name one
        # asteroid; only one of them is in the kernel.
        if identifier and entry.number is not None and identifier in (
                str(2000000 + entry.number), str(20000000 + entry.number)):
            return entry
    return None


# ---------------------------------------------------------------------------
# 2. The pinned kernel set, read from scripts/fetch_kernels.py
# ---------------------------------------------------------------------------

def _default_project_root():
    """``gui/gui_core/curated_sources.py`` -> the repository root."""
    return Path(__file__).resolve().parents[2]


#: ``{path: (mtime, size, module)}``. The card consults the registry on every
#: keystroke in the Name field, and executing an 8 kB script each time is work
#: for nothing. Keyed on mtime AND size so an edited script is re-read rather
#: than served stale -- a cache that could hand back a superseded pinned hash
#: would be the second copy this whole arrangement exists to avoid.
_MODULE_CACHE = {}


def fetch_kernels_module(project_root=None):
    """Import THIS INSTALLATION's ``scripts/fetch_kernels.py``, or raise.

    ``scripts/`` is not a package and is not on ``sys.path``, so this is a
    file-location import rather than ``import fetch_kernels``. The alternative
    -- copying that script's URL and sha256 table into this module -- is the
    one thing this tier must not do: the hash is pinned precisely so a
    provider swapping a file under the same name is caught, and a pin with two
    copies catches it in whichever copy was updated.

    **The path is derived from this module's own location, NOT from
    ``project_root``, and that is a security boundary rather than a
    convenience.** This function EXECUTES the file it loads. ``project_root``
    reaches here from ``BodyCard._project_root``, which falls back to
    ``Path.cwd()`` when the engine executable cannot be located, and
    ``find_project_root`` returns ``exe.parent`` for an executable outside a
    ``build/bin`` layout -- so honoring it would mean running
    ``<whatever directory the GUI was launched from>/scripts/fetch_kernels.py``
    as code. Everywhere else in this module ``project_root`` selects DATA;
    code comes from the installation, and only from the installation.

    ``project_root`` is still accepted and is still meaningful: a value that
    resolves to a DIFFERENT script is reported, because a caller passing one
    is asking a question about a tree whose pinned table may not be the one
    that answered.

    Raises rather than returning None. A curated body whose kernel table
    cannot be read is not a curated body, and a caller that got a silent None
    would fall through to the live service and reach HORIZONS' refusal -- the
    exact message this tier exists to replace.
    """
    path = _default_project_root() / "scripts" / "fetch_kernels.py"
    if project_root is not None:
        asked = Path(project_root) / "scripts" / "fetch_kernels.py"
        try:
            different = asked.resolve() != path.resolve()
        except OSError:                                      # pragma: no cover
            different = True
        if different:
            logger.info(
                "[curated] the pinned kernel table was read from %s, not from "
                "%s: this module executes that file, so it is always taken "
                "from the installation rather than from a project root, which "
                "can be the working directory.", path, asked)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is not on disk, so the pinned kernel table cannot be "
            f"read. The curated ephemeris tier needs it: it is where every "
            f"download URL and sha256 in this project lives.")
    stat = path.stat()
    stamp = (stat.st_mtime_ns, stat.st_size)
    cached = _MODULE_CACHE.get(str(path))
    if cached is not None and cached[0] == stamp:
        return cached[1]
    spec = importlib.util.spec_from_file_location(
        "ptm_fetch_kernels_for_gui", str(path))
    if spec is None or spec.loader is None:                # pragma: no cover
        raise ImportError(f"{path} could not be loaded as a module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "BODY_KERNELS"):
        raise AttributeError(
            f"{path} defines no BODY_KERNELS table. The curated registry "
            f"reads the download URL and the pinned sha256 from there so "
            f"they exist in one place; if that table was renamed, this "
            f"module must follow it rather than keep a second copy.")
    _MODULE_CACHE[str(path)] = (stamp, module)
    return module


class KernelFile:
    """One pinned kernel of a curated body's set, and whether it is here."""

    def __init__(self, subdir, name, url, sha256, path, present, size_bytes,
                 dataless=False):
        self.subdir = subdir
        self.name = name
        self.url = url
        self.sha256 = sha256
        self.path = path
        self.present = bool(present)
        self.size_bytes = int(size_bytes)
        #: A cloud placeholder: a name on this disk with no bytes behind it.
        #: Distinct from present, and distinct from missing -- reading one
        #: blocks while OneDrive hydrates it, which on an 80 MB kernel is a
        #: frozen window rather than an error.
        self.dataless = bool(dataless)

    def __repr__(self):                                   # pragma: no cover
        return f"<KernelFile {self.name} present={self.present}>"


def _kernel_dir(project_root, env=None):
    """The kernel directory ``scripts/fetch_kernels.py`` writes into.

    Its own default (``<repo>/data/kernels``), and the two environment rungs
    the engine honors, so a card looking for a fetched kernel looks where the
    run will look. ``docs/configuration.md``, "How
    generic_body.spice_metakernel is resolved".

    Delegates to ``preflight.spice_kernel_dir``, which is the one definition.
    Two copies of this ladder would let the card report "every kernel it loads
    is on this disk" in front of a pre-flight that refuses the same file.
    """
    from gui_core.preflight import spice_kernel_dir
    return spice_kernel_dir(project_root, env)


def _is_dataless(path):
    try:
        from gui_core.preflight import is_dataless
    except ImportError:                                   # pragma: no cover
        return False
    try:
        return bool(path.is_file()) and bool(is_dataless(path))
    except Exception:                                     # noqa: BLE001
        return False


def _pinned_by_filename(entry, project_root):
    """``{filename: (subdir, url, sha256)}`` for everything a fetch could get.

    The CORE set as well as the body's own. ``bennu.tm`` names four kernels
    -- leap seconds, planetary constants, ``de440s.bsp`` and the body SPK --
    and only the last is in ``BODY_KERNELS["bennu"]``; the other three are
    ``KERNELS_CORE``. Keying the whole downloadable universe by FILENAME is
    what lets the status below start from "what does this metakernel name?"
    rather than from "what is in this body's row?", which is the question
    that has the right answer.
    """
    module = fetch_kernels_module(project_root)
    table = module.BODY_KERNELS
    selector = entry.ephemeris.fetch_bodies
    if selector not in table:
        raise KeyError(
            f"{selector!r} is not a --bodies selector in "
            f"scripts/fetch_kernels.py (it has "
            f"{', '.join(sorted(table)) or 'none'}). The curated entry for "
            f"{entry.name} names a kernel set that script does not fetch, so "
            f"the card would offer a download that cannot happen.")
    pinned = {}
    for subdir, name, url, sha in list(module.KERNELS_CORE) + table[selector]:
        pinned[name] = (subdir, url, sha)
    return pinned


def curated_kernel_files(entry, project_root=None, env=None):
    """``[KernelFile, ...]`` -- what ``<body>.tm`` NAMES, on this disk.

    Derived from the METAKERNEL, not from the body's row in the pinned table.
    That distinction is the whole of a measured defect: ``bennu.tm`` loads
    four kernels and ``BODY_KERNELS["bennu"]`` holds one, so a status built
    from the row alone reported "bennu.tm and its 1 pinned kernel are on this
    disk" -- and the card printed "already on this disk, no download needed"
    -- on a tree where three of the four were missing and the coverage check
    then refused. Reproduced 2026-08-28 in an isolated project root.

    Each file's PATH is the one the metakernel itself resolves to, so a
    download lands exactly where ``furnsh`` will look for it. ``url`` and
    ``sha256`` are empty for a member this project has no pin for -- a real
    state (a hand-edited metakernel naming somebody's own kernel) that must
    be reported rather than silently offered as a download.

    PURELY LOCAL. Nothing here touches the network -- not even a HEAD request.
    That is load-bearing rather than tidy: ``fetch_kernels.download`` asks the
    server for a Content-Length BEFORE it looks at the local file, so calling
    the downloader to find out whether a kernel is already here would put a
    network round trip in front of a body that needs none. A machine with the
    kernels fetched has to reach "ready to run" offline.
    """
    from gui_core.body_cards import metakernel_members

    root = Path(project_root) if project_root else _default_project_root()
    pinned = _pinned_by_filename(entry, root)
    metakernel = _kernel_dir(root, env) / entry.ephemeris.metakernel
    if not metakernel.is_file():
        return []

    members = metakernel_members(metakernel, root)
    if members.error:
        raise ValueError(members.error)

    out = []
    for token, path, present, dataless in (
            [(t, p, True, False) for t, p in members.kernels]
            + [(t, p, False, False) for t, p in members.missing]
            + [(t, p, False, True) for t, p in members.dataless]):
        subdir, url, sha = pinned.get(path.name, (path.parent.name, "", ""))
        try:
            size = path.stat().st_size if path.is_file() else 0
        except OSError:                                   # pragma: no cover
            size = 0
        out.append(KernelFile(subdir, path.name, url, sha, path,
                              present=present and not dataless,
                              size_bytes=size, dataless=dataless))
        del token
    for token in members.unresolved:
        out.append(KernelFile("", token, "", "", Path(token), present=False,
                              size_bytes=0))
    return out


class KernelStatus:
    """Whether this body's curated ephemeris is on this disk.

    ``ready`` is the only thing a caller should branch on. It is False for a
    missing kernel AND for a cloud placeholder, because neither can be read --
    and it now covers every kernel the metakernel NAMES, not only the body's
    own SPK. ``furnsh`` stops at the first member it cannot open, so a
    metakernel missing its leapseconds kernel gives the run no SPICE at all.
    """

    def __init__(self, entry, files, metakernel_path, error=""):
        self.entry = entry
        self.files = list(files)
        self.metakernel_path = metakernel_path
        self.error = error
        self.missing = [f for f in self.files if not f.present]
        self.dataless = [f for f in self.files if f.dataless]
        #: Missing AND pinned: what pressing "download" would actually fetch.
        self.fetchable = [f for f in self.missing
                          if f.url and f.sha256 and not f.dataless]
        #: Missing and NOT pinned. A download cannot fix these, so offering
        #: one would be a button that spends bandwidth and changes nothing.
        self.unfetchable = [f for f in self.missing
                            if f not in self.fetchable and not f.dataless]

    @property
    def ready(self):
        return (not self.error and bool(self.files) and not self.missing
                and self.metakernel_path is not None
                and Path(self.metakernel_path).is_file())

    @property
    def can_download(self):
        """Is there something a download would actually fix?

        A dataless member is deliberately excluded. ``fetch_kernels.download``
        short-circuits on a file that already exists at full size and calls
        ``verify``, which HASHES it -- so "downloading" a OneDrive placeholder
        would hydrate 80 MB in the background, which is precisely what the
        message about placeholders tells the user to avoid.
        """
        return bool(self.fetchable) and not self.dataless

    @property
    def download_bytes(self):
        """What a fetch would transfer, in bytes, as far as it is known.

        Only sizes this project has actually recorded. The body's own
        ephemeris has a pinned ``approx_bytes``; the shared core kernels do
        not, and inventing one for them -- or charging the body's figure to
        each of them, as the first draft did -- would put a number in a
        consent dialog that nothing measured. See :attr:`unsized`.
        """
        total = 0
        for kernel in self.fetchable:
            if kernel.name == self.entry.ephemeris.spk_filename:
                total += self.entry.ephemeris.approx_bytes
        return total

    @property
    def unsized(self):
        """Fetchable files whose size this project has not recorded."""
        return [f for f in self.fetchable
                if f.name != self.entry.ephemeris.spk_filename]

    def download_summary(self):
        """"80.7 MB, plus 3 shared SPICE kernels" -- never a made-up total."""
        known = human_bytes(self.download_bytes) if self.download_bytes else ""
        extra = len(self.unsized)
        if known and extra:
            return (f"{known} for the body ephemeris, plus {extra} shared "
                    f"SPICE kernel{'' if extra == 1 else 's'} whose size is "
                    f"not pinned here")
        if known:
            return known
        return (f"{extra} shared SPICE kernel{'' if extra == 1 else 's'} "
                f"whose size is not pinned here")

    def summary(self):
        """One line for the card, whichever state this is in."""
        if self.error:
            return self.error
        if self.metakernel_path is None or not Path(
                self.metakernel_path).is_file():
            return (f"{self.entry.ephemeris.metakernel} is not in the kernel "
                    f"directory. It ships with the repository, so a checkout "
                    f"missing it is a checkout missing a tracked file.")
        if self.dataless:
            names = ", ".join(str(f.path) for f in self.dataless)
            return (f"{names} is a cloud placeholder, not a file on this "
                    f"disk. Reading it would block while it hydrates, and a "
                    f"download would hydrate it too. Open its folder in "
                    f"Finder and choose 'Always Keep on This Device', then "
                    f"check again.")
        if self.unfetchable:
            names = ", ".join(str(f.path) for f in self.unfetchable)
            return (f"{self.entry.ephemeris.metakernel} names "
                    f"{names}, which is not on this disk and is not a file "
                    f"this project has a pinned download for. Fetch it "
                    f"yourself, or point the metakernel at one you have.")
        if self.missing:
            names = ", ".join(f.name for f in self.missing)
            return (f"{self.entry.ephemeris.metakernel} needs {names}, which "
                    f"{'is' if len(self.missing) == 1 else 'are'} not fetched "
                    f"yet ({self.download_summary()}).")
        return (f"{self.entry.ephemeris.metakernel} and the "
                f"{len(self.files)} kernel"
                f"{'' if len(self.files) == 1 else 's'} it loads are on this "
                f"disk.")


def human_bytes(nbytes):
    value = float(nbytes)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024.0 or unit == "GB":
            return (f"{int(value)} B" if unit == "B"
                    else f"{value:.1f} {unit}")
        value /= 1024.0
    return f"{value:.1f} GB"                              # pragma: no cover


def curated_kernel_status(entry, project_root=None, env=None):
    """Is this curated body's ephemeris fetched? Local check, no network."""
    root = Path(project_root) if project_root else _default_project_root()
    metakernel = _kernel_dir(root, env) / entry.ephemeris.metakernel
    try:
        files = curated_kernel_files(entry, root, env)
    except (FileNotFoundError, ImportError, AttributeError, KeyError,
            ValueError, OSError, SyntaxError) as exc:
        # SyntaxError and OSError are in this list because this runs from a
        # Qt slot on every keystroke in the Name field: `fetch_kernels_module`
        # EXECUTES scripts/fetch_kernels.py, so a locally broken copy of that
        # script would otherwise raise out of a slot, which PyQt answers with
        # qFatal and the loss of any unsaved edits.
        return KernelStatus(entry, [], None, error=(
            f"It could not be worked out whether {entry.name}'s ephemeris is "
            f"on this disk ({type(exc).__name__}: {exc}). Treat this as "
            f"unknown, not as ready."))
    return KernelStatus(entry, files,
                        metakernel if metakernel.is_file() else None)


# ---------------------------------------------------------------------------
# 3. Running the fetch
# ---------------------------------------------------------------------------

class _LineWriter:
    """A file-like that turns printed output into progress callbacks.

    ``fetch_kernels.download`` reports progress by printing ``\\r  12.0 MB /
    80.7 MB ( 14.9%)`` with no newline. Capturing that is how this reuses the
    real downloader -- with its resume, its bounded retry and its pinned-hash
    check -- instead of writing a second one for the GUI. A second downloader
    is a second place for the hash check to be forgotten.
    """

    def __init__(self, callback, echo=None):
        self._callback = callback
        self._buffer = ""
        #: Where the captured text ALSO goes. ``redirect_stdout`` replaces
        #: process-global ``sys.stdout``, and this runs on a worker thread
        #: while other fetches and any ``StreamHandler`` on stdout are live,
        #: so without an echo the redirect would swallow unrelated output for
        #: the whole of an 80 MB transfer -- and discard it entirely when no
        #: progress callback is supplied.
        self._echo = echo

    def write(self, text):
        if self._echo is not None:
            try:
                self._echo.write(str(text))
            except Exception:                             # noqa: BLE001
                pass
        self._buffer += str(text)
        while True:
            # The FIRST delimiter, not the furthest. `max` of the two `find`
            # results looks equivalent and is not: with a "\n" at 3 and a "\r"
            # at 10 it cuts at 10 and the line ending at 3 is swallowed into
            # the same chunk -- so a burst that arrived as one write would
            # report its last line and drop the ones before it, which on a
            # progress stream is exactly the output the user is watching.
            breaks = [i for i in (self._buffer.find("\n"),
                                  self._buffer.find("\r")) if i >= 0]
            if not breaks:
                break
            index = min(breaks)
            line, self._buffer = self._buffer[:index], self._buffer[index + 1:]
            self._emit(line)
        return len(text)

    def flush(self):
        if self._echo is not None:
            try:
                self._echo.flush()
            except Exception:                             # noqa: BLE001
                pass
        if self._buffer:
            self._emit(self._buffer)
            self._buffer = ""

    def _emit(self, line):
        line = line.strip()
        if not line or self._callback is None:
            return
        try:
            self._callback(line)
        except Exception:                                 # noqa: BLE001
            logger.debug("[curated] a progress callback raised; ignoring it",
                         exc_info=True)


def download_curated_kernels(entry, project_root=None, env=None,
                             progress=None, downloader=None):
    """Fetch the members ``<body>.tm`` names that are missing AND pinned.

    Returns ``(ok, lines)``. The download, the resume, the bounded retry and
    the sha256 check are all ``scripts/fetch_kernels.py``'s -- this function
    chooses WHICH files and reports what happened, and does not reimplement
    any of it.

    Each file is written to THE PATH THE METAKERNEL RESOLVES TO, so a fetch
    lands exactly where ``furnsh`` will look. A missing member with no pinned
    download is reported and skipped rather than guessed at.

    ``downloader`` is for tests: anything with ``download(url, dest, sha)``.
    """
    root = Path(project_root) if project_root else _default_project_root()
    lines = []
    try:
        status = curated_kernel_status(entry, root, env)
    except Exception as exc:                              # noqa: BLE001
        return False, [f"Nothing was downloaded: {exc}"]
    if status.error:
        return False, [f"Nothing was downloaded: {status.error}"]

    if downloader is None:
        try:
            downloader = fetch_kernels_module(root)
        except Exception as exc:                          # noqa: BLE001
            return False, [f"Nothing was downloaded: {exc}"]

    import contextlib
    import sys

    ok = True
    if not status.missing:
        return True, [f"Every kernel {entry.ephemeris.metakernel} loads is "
                      f"already on this disk; nothing was downloaded."]
    for kernel in status.unfetchable:
        lines.append(f"{kernel.path} is missing and this project has no "
                     f"pinned download for it, so it was NOT fetched. "
                     f"{entry.ephemeris.metakernel} will not load without it.")
        ok = False
    for kernel in status.dataless:
        lines.append(f"{kernel.path} is a cloud placeholder and was NOT "
                     f"fetched: a download would find a file of the right "
                     f"size already there and only hash it, hydrating it in "
                     f"the background. Make it available offline first.")
        ok = False

    # Where a download is allowed to land. `KernelFile.path` comes from
    # PARSING a metakernel, and `metakernel_members` deliberately resolves a
    # member anywhere on disk -- which was safe while the only things done
    # with it were `stat` and `getfat`, and is not once a file is WRITTEN
    # there. `PTM_SPICE_KERNELS` can point the .tm lookup at a directory the
    # user chose, so the .tm being parsed is not necessarily the tracked one.
    # A member outside both the kernel directory and the project tree is
    # reported, not created.
    allowed = []
    for candidate in (_kernel_dir(root, env), root):
        try:
            allowed.append(candidate.resolve())
        except OSError:                                   # pragma: no cover
            continue

    def _inside_allowed(path):
        try:
            resolved = path.resolve()
        except OSError:                                   # pragma: no cover
            return False
        return any(base == resolved or base in resolved.parents
                   for base in allowed)

    for kernel in status.fetchable:
        if not _inside_allowed(kernel.path):
            lines.append(
                f"{kernel.path} is outside both the kernel directory and the "
                f"project, so nothing was written there. A metakernel member "
                f"resolving outside the tree is fine to READ and is not "
                f"somewhere this will CREATE a file. Fetch it yourself if "
                f"that path is what you meant.")
            ok = False
            continue
        try:
            kernel.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            lines.append(f"{kernel.path.parent} could not be created ({exc}); "
                         f"{kernel.name} was not fetched.")
            ok = False
            continue
        size = (human_bytes(entry.ephemeris.approx_bytes)
                if kernel.name == entry.ephemeris.spk_filename
                else "size not pinned here")
        lines.append(f"Fetching {kernel.name} ({size}) from {kernel.url} "
                     f"-> {kernel.path}")
        # The echo keeps the redirect from swallowing anything else printed on
        # this process while the transfer runs.
        writer = _LineWriter(progress, echo=sys.stdout)
        try:
            with contextlib.redirect_stdout(writer):
                result = downloader.download(kernel.url, str(kernel.path),
                                             kernel.sha256)
            writer.flush()
        except Exception as exc:                          # noqa: BLE001
            writer.flush()
            lines.append(f"{kernel.name} could not be fetched "
                         f"({type(exc).__name__}: {exc}). Nothing about this "
                         f"card has been changed.")
            ok = False
            continue
        if result:
            lines.append(f"{kernel.name} is on disk and its sha256 matches "
                         f"the pinned value.")
        else:
            # `download` prints the reason -- a checksum mismatch is not
            # retried and says so -- and returns False. Do not restate it as
            # something else; the printed reason is the one that is true.
            lines.append(f"{kernel.name} FAILED to download or verify. The "
                         f"downloader's own reason is above.")
            ok = False
    return ok, lines


# ---------------------------------------------------------------------------
# 4. The shape
# ---------------------------------------------------------------------------

def curated_shape_path(entry, project_root=None):
    """``(path, present)`` for this body's shipped shape model."""
    root = Path(project_root) if project_root else _default_project_root()
    path = root / entry.shape.relpath
    return path, path.is_file()
