"""Flux cubes: the rules the GUI needs, with no Qt and no display.

A flux cube is a PASSIVE virtual measurement fixture standing on the terrain.
It reports the irradiance arriving on six oriented faces -- zenith, nadir,
north, south, east, west -- split into direct sunlight, terrain-scattered
visible, terrain infrared, primary-body reflected and primary-body infrared, at
every output step. It does not occlude, emit, scatter, or enter the view-factor
matrix, and cubes never see each other, so turning cubes on changes no
temperature anywhere on the mesh. `docs/flux-cubes.md` is the reference.

WHAT THIS MODULE IS FOR. The card that offers flux cubes has to answer four
questions before it writes anything, and none of them can be answered by
looking at the configuration alone:

1. **Will the engine accept this run at all?** The post-solve pass has hard
   refusals, and every one of them fires AFTER the thermal solve has finished
   -- so a run that fails them burns the whole solve and then aborts. They are
   transcribed in `engine_support` from `flux_cube_pass.cpp` and
   `ptm_thermal_sim.cpp`, with the file each one comes from, so a card can
   refuse before launch instead of after.
2. **How tall is the cube?** `ptm_flux_cube_geom` measures a length scale off
   the mesh and multiplies it. The card cannot invent a number, and it must not
   quote the tool's number as if it had measured it. `auto_geometry` returns an
   ESTIMATE and labels it one; `read_cache_header` returns the ACHIEVED value
   from a cache that exists, which is the real answer.
3. **How many cubes, and how big is the file?** Both are exactly computable,
   and `product_bytes` was checked against two delivered products rather than
   derived and trusted -- see its docstring.
4. **Where does the geometry come from?** Not from this configuration. The
   cache is built by `ptm_flux_cube_geom` and the engine REFUSES a cache that
   does not match the run's mesh, so `geom_command` builds the argument list
   and nothing here ever writes geometry.

⛔ THE GEOMETRY KEYS ARE NOT ENGINE KEYS. `spacing_m`, `radius_m`,
`height_factor` and the rest are inputs to `ptm_flux_cube_geom` and live in the
cache header; the engine reads them back OUT of the cache and refuses a
mismatch. `config_parser.cpp` accepts them under `flux_cubes:` and reports them
as ignored. The GUI therefore keeps them as its own state and does not emit
them: `test_config_parity.test_gui_written_yaml_uses_only_engine_keys` would
fail on them, correctly, because they would read as settings and are not.
"""

import hashlib
import math
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Defaults. The two lattice numbers are the ones the delivered products used
# (docs/flux-cubes.md section 1: 25 m spacing, 500 m radius, 1,257 cubes).
# ---------------------------------------------------------------------------

#: Lattice spacing, meters.
DEFAULT_SPACING_M = 25.0
#: Grid radius, meters. Every lattice point inside this radius gets a cube.
DEFAULT_RADIUS_M = 500.0
#: Cube center height and cube edge, both as a fraction of the local median
#: facet edge. `ptm_flux_cube_geom` defaults (`--height-factor`,
#: `--edge-factor`); the card shows the height and does not offer the factor,
#: because 0.1 is what every validated product used.
HEIGHT_FACTOR = 0.1
EDGE_FACTOR = 0.1

#: The two placement modes the card offers. `grid` builds a regular lattice
#: around the mesh center; `named locations` places cubes at named
#: latitude/longitude points and builds no lattice (the tool's `--radius -1`,
#: which is NOT `--radius 0` -- that one still places the single origin cube).
#:
#: These strings are the combo's own entries, which is what the field helper
#: stores. They are GUI state and never reach the engine.
MODE_LATTICE = "grid"
MODE_LOCATIONS = "named locations"
#: BOTH, and it is not decoration. The tool appends named locations after the
#: lattice in ONE cache, so a layout can hold both -- and a card with only the
#: first two modes turns that into a trap: open a project whose cache carries a
#: grid AND three pads, and the card reads the pads, shows "named locations",
#: hides the grid fields, and a rebuild writes `--radius -1` and throws the
#: 1,257 lattice cubes away. Found in review before it shipped.
MODE_BOTH = "grid and named locations"
MODES = (MODE_LATTICE, MODE_LOCATIONS, MODE_BOTH)

#: Spellings a stored mode may arrive in -- an older project, a hand-edited
#: file, or the internal name used while this was being written. Keyed the way
#: `BaseConfigTab.set_enum` normalizes (lowercase, no spaces, hyphens or
#: underscores) so one table serves both the combo and every reader.
MODE_ALIASES = {
    "grid": MODE_LATTICE,
    "lattice": MODE_LATTICE,
    "": MODE_LATTICE,
    "namedlocations": MODE_LOCATIONS,
    "locations": MODE_LOCATIONS,
    "named": MODE_LOCATIONS,
    "gridandnamedlocations": MODE_BOTH,
    "both": MODE_BOTH,
}


def normalize_mode(value):
    """A stored mode as one of `MODES`, defaulting to the grid.

    Never raises and never returns something outside `MODES`: this is read on
    every card load, and a mode nobody recognizes would otherwise decide
    whether a lattice is built by falling off the end of an `if`.
    """
    key = "".join(c for c in str(value or "").lower() if c not in "_- ")
    return MODE_ALIASES.get(key, MODE_LATTICE)


def mode_builds_lattice(mode):
    """True when this mode asks the tool for a grid."""
    return normalize_mode(mode) in (MODE_LATTICE, MODE_BOTH)


def mode_uses_locations(mode):
    """True when this mode asks the tool for named points.

    Also decides whether `flux_cubes.locations` is EMITTED: a grid-only run is
    not asking for named points, and writing a table's leftover rows into it
    hands the engine a list of names its layout does not contain -- which it
    refuses, after the whole thermal solve.
    """
    return normalize_mode(mode) in (MODE_LOCATIONS, MODE_BOTH)

#: Six faces, six components, from `flux_cube_format.h`. Fixed order there and
#: fixed here; the sizing arithmetic below depends on both counts.
N_CUBE_FACES = 6
N_COMPONENTS = 6

#: Longest name a location may carry, including the terminating byte
#: (`kNameLen`, flux_cube_format.h). A longer one is truncated in the cache and
#: two locations could then collide, so it is refused here instead.
MAX_LOCATION_NAME = 31

#: Executable that builds the geometry cache.
GEOM_EXECUTABLE = "ptm_flux_cube_geom"

#: The only two values `flux_cubes.shadow_probe_rays` can mean. The engine
#: branches on `shadow_probe_rays >= 5` and records `(rays >= 5 ? 5 : 1)` in
#: the product's provenance, so 2, 3 and 4 are the single center ray with a
#: different number written beside it.
SHADOW_PROBE_CHOICES = (1, 5)


# ---------------------------------------------------------------------------
# 1. Will the engine accept this?
# ---------------------------------------------------------------------------

@dataclass
class SupportPolicy:
    """Whether a configuration can carry flux cubes, and why not if it cannot.

    `blockers` are conditions the engine itself refuses on. `notes` are true
    statements about a run that will succeed.
    """
    allowed: bool = True
    blockers: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    @property
    def reason(self):
        return " ".join(self.blockers) if self.blockers else " ".join(self.notes)


def _section(config, name):
    node = (config or {}).get(name)
    return node if isinstance(node, dict) else {}


def _truthy(value, default=False):
    """A YAML scalar read as a boolean, tolerating the spellings a hand-edited
    file uses. Deliberately the same tolerance the rest of the GUI applies: a
    malformed value must not abort a window."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in ("true", "yes", "on", "1")


def _as_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def engine_support(config):
    """The engine's own preconditions for the flux-cube pass, transcribed.

    Every blocker below is a hard refusal in the engine, and every one of them
    fires at the END of the run -- the pass is post-solve, immediately before
    the location export -- so the cost of learning about it there is the whole
    thermal solve. That is the reason this function exists.

    * **Radiation exchange off.** The band-separated series the pass reads are
      allocated inside the radiation block (`ptm_thermal_sim.cpp`, "Two-band
      radiosity active"), so with radiation off they are empty and the pass
      refuses with "a required field array is missing".
    * **Single mixed radiosity** (`radiation.band_separated: false`). One
      radiosity cannot be split into a visible and an infrared column, and the
      pass will not emit zeros or a merged column in place of the split.
    * **No imported terrain mesh.** `ptm_flux_cube_geom` takes a core mesh file
      and there is no file for a procedural icosphere, so the cache cannot be
      built for a global run at all. This is why the group is offered on the
      local-mesh card only.
    * **Checkpoint look-ahead.** The pass asserts
      `equilibration steps + output steps == total steps`, and a look-ahead adds
      integrated steps past the export window, breaking it. This is also the
      chunked-run marker: under the chunked runner each chunk would write its
      own product and there is no rule for merging them, so a window that needs
      cubes is run in one piece.

    Not a blocker, and deliberately: `output.skip_time_series`. Cubes read the
    band series in memory rather than the per-step frames on disk, and the two
    settings were measured to give a byte-identical product.
    """
    policy = SupportPolicy()
    radiation = _section(config, "radiation")
    thermal = _section(config, "thermal")
    mesh = _section(config, "mesh")

    if not _truthy(radiation.get("enabled"), False):
        policy.blockers.append(
            "Facet-to-facet infrared exchange is off. A flux cube reports the "
            "infrared and scattered sunlight arriving from the surrounding "
            "terrain, and the run only computes those quantities when that "
            "exchange is on, so the engine stops at the end of the run rather "
            "than writing a file of zeros. Turn it on above.")

    if not _truthy(radiation.get("band_separated"), True):
        policy.blockers.append(
            "This configuration asks for one combined radiosity rather than "
            "separate visible and infrared bands. A flux cube reports the two "
            "separately, and they cannot be recovered from the combined one, "
            "so the engine stops. Set radiation.band_separated to true.")

    if not str(mesh.get("import_file", "") or "").strip():
        # SAYS WHAT IS MISSING, NOT WHICH MODE. This used to end "and a global
        # run generates its sphere in memory with no file to build against",
        # which is a statement about a run this function cannot see: an empty
        # `mesh.import_file` on a LOCAL-mesh card means the user has not
        # imported terrain yet, and telling them their run is global is simply
        # untrue. The card knows the mode and appends its own sentence
        # (`BodyCard._flux_mode_note`); this one stays true of every caller,
        # including the launch check, which has no card at all.
        policy.blockers.append(
            "Flux cubes need an imported terrain mesh. The cube grid is built "
            "against a mesh file, and this run has none -- either no terrain "
            "has been imported, or the run generates its sphere in memory.")

    lookahead = _as_float(thermal.get("checkpoint_lookahead_steps", 0), 0.0)
    if lookahead > 0:
        policy.blockers.append(
            "This run integrates extra steps past its output window for a "
            "handoff to the next chunk, which leaves the cube rows unable to "
            "line up with the run's own time axis. Run a window that needs "
            "flux cubes in one piece, with thermal.checkpoint_lookahead_steps "
            "at 0.")

    policy.allowed = not policy.blockers
    if policy.allowed:
        policy.notes.append(
            "Cubes are passive: they read the finished solution and write "
            "their own files, so every temperature and every flux on the mesh "
            "is the same whether this is on or off.")
    return policy


# ---------------------------------------------------------------------------
# 2. How tall is the cube?
# ---------------------------------------------------------------------------

@dataclass
class CubeGeometry:
    """The height and edge `ptm_flux_cube_geom` is expected to resolve.

    `measured` is False for everything this class returns: it is an estimate
    from the whole mesh, and the tool takes its median over the faces inside
    the grid disk. `read_cache_header` is where a measured number comes from.
    """
    height_m: float = None
    edge_m: float = None
    median_edge_m: float = None
    n_faces: int = None
    source: str = "unknown"
    detail: str = ""
    measured: bool = False

    def describe(self):
        if self.height_m is None:
            return (f"Cube height cannot be estimated here: {self.detail}. "
                    f"The geometry tool measures it on the mesh and records it "
                    f"in the cache.")
        return (f"about {_fmt_length(self.height_m)} "
                f"({HEIGHT_FACTOR:g} x the {_fmt_length(self.median_edge_m)} "
                f"median facet edge of this mesh)")


def _fmt_length(m):
    """Same ladder `facet_scale` uses, so two readings of one mesh on one
    screen are not formatted two ways."""
    from gui_core.facet_scale import fmt_length
    return fmt_length(m)


def auto_geometry(config, project_root=None):
    """Estimate the cube height and edge for the mesh this config selects.

    `height = 0.1 x local median facet edge`, and the edge the same, which is
    `docs/flux-cubes.md` section 1 and the tool's own defaults. The length
    scale is the median EQUIVALENT EDGE `sqrt(4A/sqrt(3))`, not `sqrt(mean
    area)`: on the shipped 80N crater those are 26.86 m and 17.95 m, so the
    wrong one puts the cube a third of the way off.

    ⛔ AN ESTIMATE, AND IT SAYS SO. The tool takes its median over the core
    faces inside the grid disk (floored at a 50 m neighborhood), this takes it
    over the whole mesh, and above 200,000 faces the reader samples. On the 80N
    crater the two are 26.86 m and 28.74 m -- 6.5% apart, which is the size of
    the disagreement to expect, not a bound on it. Whenever a cache exists,
    show its header instead.
    """
    from gui_core.facet_scale import facet_scale
    # ⛔ REFUSE FIRST on a run with no imported mesh, before asking
    # `facet_scale`. That function falls back to the ICOSPHERE branch, which is
    # a real answer to its own question and a wrong one to this one: on a
    # default lunar ico5 it returns a 21 km facet, and a tenth of that is a
    # cube "6.54 km above the ground" -- a number with no meaning, offered for
    # a run where the layout tool has no mesh file to build against at all.
    # Caught by the card's own mesh-change arm.
    if not str(_section(config, "mesh").get("import_file", "") or "").strip():
        return CubeGeometry(
            detail="this run has no imported terrain mesh, and the cube "
                   "height is measured on one")
    scale = facet_scale(config, project_root)
    if scale.median_edge_m is None or scale.median_edge_m <= 0:
        return CubeGeometry(
            n_faces=scale.n_faces, source=scale.source,
            detail=scale.detail or "the mesh could not be measured here")
    edge = float(scale.median_edge_m)
    return CubeGeometry(
        height_m=HEIGHT_FACTOR * edge, edge_m=EDGE_FACTOR * edge,
        median_edge_m=edge, n_faces=scale.n_faces, source=scale.source,
        detail=scale.detail, measured=False)


# ---------------------------------------------------------------------------
# 3. How many cubes, and how big is the file?
# ---------------------------------------------------------------------------

def lattice_cube_count(spacing_m, radius_m):
    """Cubes on the lattice, by the tool's own rule.

    Transcribed from `flux_cube_geom.cpp`::

        n_lat = floor(radius / spacing)
        for i in -n_lat .. n_lat, j in -n_lat .. n_lat:
            keep (i*spacing)^2 + (j*spacing)^2 <= radius^2

    Not an approximation of the disk area: the count is what the file will
    contain, and `pi r^2 / s^2` is 1,256.6 against the true 1,257 at the
    defaults -- close here and not close at a small radius, where the lattice
    is a handful of points and rounding decides them.

    A negative radius means "no lattice, named locations only" and returns 0,
    which is the tool's `--radius <0`.

    ⛔ COLUMN BY COLUMN, not point by point. The obvious double loop is O(n^2)
    in `radius/spacing`, and this runs from the `valueChanged` of two spin
    boxes that accept 0.1 m and 1,000,000 m -- measured before it was replaced:
    radius 5,000 m at 1 m spacing took 10.6 s and radius 20,000 m took 171 s,
    per keystroke, on the GUI thread. Counting each column's points in closed
    form is O(n) and gives the same integer, which is what
    `test_the_cube_count_is_the_tools_own_rule` checks against the double loop
    over a range of grids rather than only against the one documented answer.
    """
    spacing = _as_float(spacing_m, 0.0)
    radius = _as_float(radius_m, -1.0)
    if spacing <= 0 or radius < 0 or not math.isfinite(spacing) \
            or not math.isfinite(radius):
        return 0
    n_lat = int(math.floor(radius / spacing))
    r2 = radius * radius
    count = 0
    for i in range(-n_lat, n_lat + 1):
        x = i * spacing
        rest = r2 - x * x
        if rest < 0:
            continue
        # Largest |j| with x*x + (j*spacing)^2 <= r*r -- written as the tool
        # writes it, so the two agree bit for bit on a boundary point -- then
        # the 2k+1 points -k .. +k. `int(sqrt(rest)/spacing)` can land one
        # short or one over on a float boundary, so both neighbors are checked
        # explicitly; the count has to be the tool's exactly, not nearly.
        k = min(int(math.sqrt(rest) / spacing), n_lat)
        while k < n_lat and x * x + ((k + 1) * spacing) ** 2 <= r2:
            k += 1
        while k > 0 and x * x + (k * spacing) ** 2 > r2:
            k -= 1
        if x * x + (k * spacing) ** 2 > r2:
            continue
        count += 2 * k + 1
    return count


#: Sizes taken from the packed structures in `flux_cube_format.h`, not guessed.
#: Everything there is inside `#pragma pack(push, 1)`, so a field is where the
#: sum of the ones before it says it is and there is no padding to allow for.
#: `test_gui_flux_cubes.py` reads a real cache and checks these against it,
#: because a wrong offset here reads plausible garbage rather than raising.
_FCG_HEADER_BYTES = 256      # FcgHeader
_FCB_HEADER_BYTES = 176      # FcbHeader
_CUBE_RECORD_BYTES = 80      # CubeRecord
_PATCH_RECORD_BYTES = 40     # PatchRecord: 8 + 4 + 4 and SIX floats
_UTC_BYTES = 24              # epoch_utc[n_steps][24]
_NAME_BYTES = 32             # kNameLen


def product_bytes(n_cubes, n_steps, has_names=False):
    """Bytes in `flux_cubes/flux_cubes.fcb`, from the documented layout.

    CHECKED, NOT DERIVED. Against the two delivered A03MP026 products:

        1,257 cubes x  29 steps ->  11,018,709 here, file  11,019,828
        1,257 cubes x 720 steps -> 269,011,160 here, file 269,012,280

    Both short by the provenance JSON blob, which was 1,119 and 1,120 bytes.
    The blob is a few hundred characters of paths and hashes and is not
    modeled; the number this returns is therefore a floor about one kilobyte
    below the file, which is the right direction for a size estimate and is
    stated wherever it is shown.
    """
    n_cubes = max(0, int(n_cubes))
    n_steps = max(0, int(n_steps))
    total = _FCB_HEADER_BYTES
    total += n_cubes * N_CUBE_FACES * N_COMPONENTS * n_steps * 8   # flux
    total += n_cubes * n_steps * 8                                 # T_face
    total += n_cubes * n_steps * 1                                 # penumbral
    total += 3 * n_cubes * N_CUBE_FACES * 4                        # f_sky/ring/max
    total += n_cubes * _CUBE_RECORD_BYTES                          # cubes
    total += n_steps * 8                                           # epoch_et
    total += n_steps * _UTC_BYTES                                  # epoch_utc
    if has_names:
        total += n_cubes * _NAME_BYTES
    return total


#: Bytes per CSV row and of the comment header, SOLVED from two real files
#: rather than estimated: the same cube written at 24 and at 118 output steps
#: came to 16,807 and 76,403 bytes, so 94 rows cost 59,596 bytes and the row is
#: 634.0 bytes with a 1,591-byte header. It is that constant because the engine
#: writes the flux columns in scientific notation at fixed precision, so every
#: row is the same width whatever the numbers are. A first draft guessed 520
#: and 1,400 and would have under-reported a full-grid year by 85 MB, in the
#: one line whose purpose is a disk-space warning.
_CSV_BYTES_PER_ROW = 634
_CSV_HEADER_BYTES = 1591


def csv_bytes(n_cubes_written, n_steps):
    """Approximate total bytes of the per-cube CSV fan-out."""
    n_cubes_written = max(0, int(n_cubes_written))
    n_steps = max(0, int(n_steps))
    return n_cubes_written * (_CSV_HEADER_BYTES
                              + n_steps * _CSV_BYTES_PER_ROW)


def csv_files_written(n_cubes, n_named, write_csv_max_cubes):
    """How many per-cube CSV files the engine will write.

    The cap applies to the lattice fan-out only. `flux_cube_pass.cpp` raises
    the count back to every cube as soon as ANY named location is present --
    a named location was asked for by name and is never silently dropped -- so
    a cap plus a named location writes every file, and a card that promised
    three files would be wrong.
    """
    n_cubes = max(0, int(n_cubes))
    cap = max(0, int(write_csv_max_cubes or 0))
    if n_named > 0 or cap <= 0:
        return n_cubes
    return min(n_cubes, cap)


def fmt_bytes(n):
    """A size in the largest unit that keeps it above 1."""
    step = 1024.0
    value = float(n)
    for unit in ("bytes", "kB", "MB", "GB", "TB"):
        if value < step or unit == "TB":
            return (f"{value:.0f} {unit}" if unit == "bytes"
                    else f"{value:.1f} {unit}")
        value /= step
    return f"{value:.1f} TB"


# ---------------------------------------------------------------------------
# 4. Where does the geometry come from?
# ---------------------------------------------------------------------------

_FCG_MAGIC = b"FCG1"
# The FIXED prefix of FcgHeader up to and including `truncate_rel`, then the
# three resolution fields. Little-endian, packed -- `#pragma pack(push, 1)`.
_FCG_HEAD_FMT = "<4sIIIQdQQddddddIIQd"
_FCG_TAIL_FMT = "<ddd"


@dataclass
class CacheHeader:
    """What a `.fcg` says about itself. Every number here was MEASURED by
    `ptm_flux_cube_geom` on the mesh, which is what makes it worth reading back
    rather than re-deriving."""
    ok: bool = False
    error: str = ""
    path: str = ""
    n_cubes: int = 0
    n_named: int = 0
    n_faces_core: int = 0
    geomhash_core: float = 0.0
    center_lat_deg: float = 0.0
    center_lon_deg: float = 0.0
    spacing_m: float = 0.0
    radius_m: float = 0.0
    height_m: float = 0.0
    edge_m: float = 0.0
    local_median_edge_m: float = 0.0
    height_factor: float = 0.0
    n_rays: int = 0
    ring_hash: int = 0
    names: tuple = ()

    def describe(self):
        if not self.ok:
            return self.error
        lattice = (f"{self.n_cubes - self.n_named:,} on a "
                   f"{self.spacing_m:g} m grid out to {self.radius_m:g} m"
                   if self.radius_m >= 0 else "no grid")
        named = (f" plus {self.n_named:,} named location(s)"
                 if self.n_named else "")
        return (f"{self.n_cubes:,} cubes ({lattice}{named}), "
                f"{_fmt_length(self.height_m)} above the ground, "
                f"{_fmt_length(self.edge_m)} on a side, built for a "
                f"{self.n_faces_core:,}-face mesh.")


def read_cache_header(path):
    """Read a `.fcg` header, and the trailing names block if it has one.

    Reads the header and, when `has_names` is set, seeks past the cube,
    patch, view-factor and JSON blocks to the names at the end.

    ⛔ THE OFFSETS ARE CHECKED AGAINST THE FILE'S OWN LENGTH before anything
    after the header is read, and a disagreement is a refusal. A wrong record
    size does not raise here -- it lands inside a neighboring block and returns
    text. The first version of this reader had `PatchRecord` at 36 bytes
    instead of 40 (it has SIX trailing floats, not five) and reported three
    location names that were slices of the provenance JSON: 'south","east"',
    'ite-local ENU', '}pad_alpha'. Every one of them was printable, so a
    printable-character check passed it. The length check is what catches it,
    because it is the one test a wrong offset cannot survive.
    """
    header = CacheHeader(path=str(path))
    p = Path(path)
    if not p.exists():
        header.error = (f"{p.name} does not exist. Build it with the button "
                        f"above, or point at an existing cache.")
        return header
    try:
        with open(p, "rb") as fh:
            head = struct.calcsize(_FCG_HEAD_FMT)
            tail = struct.calcsize(_FCG_TAIL_FMT)
            raw = fh.read(head)
            if len(raw) < head:
                header.error = f"{p.name} is too short to be a flux-cube cache."
                return header
            (magic, version, n_cubes, n_faces_per_cube, total_nnz,
             geomhash, n_faces_core, ring_hash, lat, lon, spacing, radius,
             height, edge, n_rays, sampler_id, seed,
             truncate_rel) = struct.unpack(_FCG_HEAD_FMT, raw)
            if magic != _FCG_MAGIC:
                header.error = (f"{p.name} is not a flux-cube geometry cache "
                                f"(its first four bytes are {magic!r}).")
                return header
            if version != 1:
                header.error = (f"{p.name} is version {version}; this "
                                f"application reads version 1.")
                return header
            (height_factor, edge_factor,
             local_median_edge) = struct.unpack(_FCG_TAIL_FMT, fh.read(tail))
            # frame_origin/east/north/up (12 doubles) + max tilt, then the
            # two trailing uint32s.
            fh.read(8 * 13)
            json_len, has_names = struct.unpack("<II", fh.read(8))

            header.ok = True
            header.n_cubes = int(n_cubes)
            header.n_faces_core = int(n_faces_core)
            header.geomhash_core = float(geomhash)
            header.center_lat_deg = float(lat)
            header.center_lon_deg = float(lon)
            header.spacing_m = float(spacing)
            header.radius_m = float(radius)
            header.height_m = float(height)
            header.edge_m = float(edge)
            header.local_median_edge_m = float(local_median_edge)
            header.height_factor = float(height_factor)
            header.n_rays = int(n_rays)
            header.ring_hash = int(ring_hash)

            # Layout: header, cubes, patches, row_cols, row_vals, json, and
            # then the names when there are any. Add it up and compare with
            # the file, so a record size that has drifted from the C++ struct
            # is caught here rather than read through.
            names_offset = (_FCG_HEADER_BYTES
                            + n_cubes * _CUBE_RECORD_BYTES
                            + n_cubes * n_faces_per_cube * _PATCH_RECORD_BYTES
                            + total_nnz * 4 + total_nnz * 4
                            + json_len)
            expected = names_offset + (n_cubes * _NAME_BYTES if has_names
                                       else 0)
            actual = p.stat().st_size
            if actual != expected:
                header.ok = False
                header.error = (
                    f"{p.name} is {actual:,} bytes and its own header "
                    f"describes {expected:,}. It is either truncated or "
                    f"written by a different version of the geometry tool "
                    f"than this application reads. Rebuild the cache.")
                return header

            if has_names:
                fh.seek(names_offset)
                blob = fh.read(n_cubes * _NAME_BYTES)
                names = []
                for k in range(n_cubes):
                    chunk = blob[k * _NAME_BYTES:(k + 1) * _NAME_BYTES]
                    name = chunk.split(b"\x00", 1)[0].decode(
                        "ascii", errors="replace").strip()
                    names.append(name)
                header.names = tuple(n for n in names if n)
                header.n_named = len(header.names)
    except (OSError, struct.error) as exc:
        header.ok = False
        header.error = f"could not read {p.name}: {exc}"
    return header


def geom_command(executable, core_mesh, out_cache, mode=MODE_LATTICE,
                 spacing_m=DEFAULT_SPACING_M, radius_m=DEFAULT_RADIUS_M,
                 occluder_mesh="", locations_csv="", center_lat=None,
                 center_lon=None, threads=8):
    """The `ptm_flux_cube_geom` argument list, as a list -- never a shell string.

    In named-locations mode `--radius -1` disables the lattice. That is the
    tool's own spelling, and it is NOT the same as `--radius 0`, which still
    places the single cube at the grid origin. The combined mode passes both a
    grid and a location list, which is how the tool appends named cubes after
    the lattice in one layout.
    """
    argv = [str(executable), "--core", str(core_mesh), "--out", str(out_cache)]
    if occluder_mesh:
        argv += ["--ring", str(occluder_mesh)]
    if mode_builds_lattice(mode):
        argv += ["--spacing", repr(float(spacing_m)),
                 "--radius", repr(float(radius_m))]
    else:
        argv += ["--radius", "-1"]
    if locations_csv:
        argv += ["--locations", str(locations_csv)]
    if center_lat is not None and center_lon is not None:
        argv += ["--center-lat", repr(float(center_lat)),
                 "--center-lon", repr(float(center_lon))]
    argv += ["--threads", str(int(threads))]
    return argv


# ---------------------------------------------------------------------------
# 5. The layout is an implementation detail: the placement IS the request
# ---------------------------------------------------------------------------
#
# A user asks for "a 25 m grid out to 500 m". That is the whole specification,
# and the `.fcg` is the file the tool happens to write it into. Everything
# below turns the specification into a deterministic path and decides whether
# a file already on disk answers it -- so nobody has to name a file, and so a
# file that answers a DIFFERENT question is never silently reused.
#
# ⛔ TWO INDEPENDENT GUARDS, deliberately, because either alone is thin:
#
#  1. The request is hashed into the FILE NAME. Two different specifications
#     therefore cannot land on one path, so "change the spacing and the old
#     layout is picked up" cannot happen by name collision.
#  2. The candidate's own HEADER is checked against the request before it is
#     reused (`cache_satisfies`). The name is only a name -- a file can be
#     copied, renamed, or written by a different version of the tool -- and the
#     header carries what the tool MEASURED.
#
# Guard 1 without guard 2 trusts a file name. Guard 2 without guard 1 makes
# every spec fight over one path and rebuild constantly. The stale-layout
# negative control in `test_gui_flux_cubes.py` exercises both.

#: Bumped when anything that changes the CONTENT of a layout for a fixed
#: request changes -- the argument list `geom_command` builds, or a default
#: above. Old caches then miss on the name and are rebuilt rather than reused
#: under a rule that no longer describes them.
LAYOUT_KEY_VERSION = 1

#: Characters of the digest in the file name. 12 hex characters is 48 bits;
#: these are per-mesh-directory, so a collision needs two specifications for
#: ONE mesh, and the header check would still refuse the wrong one.
LAYOUT_KEY_CHARS = 12


@dataclass(frozen=True)
class LayoutRequest:
    """A cube placement, as the user specified it. No file name in here.

    `mesh` and `occluder` are RESOLVED absolute paths -- the same mesh named
    two ways is one request -- and `n_faces_core` is the mesh's face count when
    it could be read, which is what the layout's header records and the engine
    compares against.
    """
    mesh: str = ""
    occluder: str = ""
    mode: str = MODE_LATTICE
    spacing_m: float = DEFAULT_SPACING_M
    radius_m: float = DEFAULT_RADIUS_M
    height_factor: float = HEIGHT_FACTOR
    edge_factor: float = EDGE_FACTOR
    n_faces_core: int = 0
    #: Named points, as a tuple of tuples so the request stays hashable and its
    #: key does not depend on dict ordering.
    locations: tuple = ()

    @property
    def named_names(self):
        return tuple(row[0] for row in self.locations)

    def describe(self):
        parts = []
        if mode_builds_lattice(self.mode):
            parts.append(f"a {self.spacing_m:g} m grid out to "
                         f"{self.radius_m:g} m "
                         f"({lattice_cube_count(self.spacing_m, self.radius_m):,}"
                         f" cubes)")
        if mode_uses_locations(self.mode) and self.locations:
            parts.append(f"{len(self.locations)} named location(s)")
        return " plus ".join(parts) if parts else "no cubes"


def request_from_config(config, project_root=None, resolve=None,
                        face_count=None):
    """The `LayoutRequest` a configuration plus the card's state describes.

    `resolve` and `face_count` are handed in for the same reason the pre-flight
    helpers are: this module is imported BY `gui_core.preflight`, so importing
    it back at module level would be a cycle. Both default to no-ops so the
    request can be built without touching the disk, which is what the key tests
    do.
    """
    flux = _section(config, "flux_cubes")
    mesh_cfg = _section(config, "mesh")
    resolve = resolve or (lambda value, root: value)
    raw_mesh = str(mesh_cfg.get("import_file", "") or "").strip()
    raw_ring = str(mesh_cfg.get("occluder_file", "") or "").strip()
    mesh = str(resolve(raw_mesh, project_root)) if raw_mesh else ""
    ring = str(resolve(raw_ring, project_root)) if raw_ring else ""
    mode = normalize_mode(flux.get("mode"))
    rows = config_locations(config) if mode_uses_locations(mode) else []
    faces = 0
    if mesh and face_count is not None:
        try:
            faces = int(face_count(mesh) or 0)
        except Exception:                                    # noqa: BLE001
            faces = 0
    return LayoutRequest(
        mesh=mesh, occluder=ring, mode=mode,
        spacing_m=_as_float(flux.get("spacing_m"), DEFAULT_SPACING_M),
        radius_m=_as_float(flux.get("radius_m"), DEFAULT_RADIUS_M),
        n_faces_core=faces,
        locations=tuple(
            (str(r.get("name", "")), _as_float(r.get("lat_deg"), 0.0),
             _as_float(r.get("lon_deg"), 0.0),
             _as_float(r.get("height_m"), -1.0),
             _as_float(r.get("edge_m"), -1.0))
            for r in rows))


def request_key(request):
    """A short, stable digest of everything that changes the layout's content.

    `repr(float(...))` is the same round-trippable spelling `geom_command`
    passes on the command line, so the key is keyed on the number the TOOL
    receives and 25 and 25.0 are one request rather than two.

    The mesh's face count is in the key as well as its path: an A3GT core mesh
    has been regenerated under its own name after the artifacts built against
    it (project map, "A3GT DEM meshes can be silently regenerated"), and a key
    made of the path alone would hand that run the old layout. The engine also
    refuses on the geometry hash, but by then the whole solve has run.
    """
    fields = [
        f"v{LAYOUT_KEY_VERSION}",
        f"mesh={os.path.abspath(request.mesh) if request.mesh else ''}",
        f"faces={int(request.n_faces_core)}",
        f"ring={os.path.abspath(request.occluder) if request.occluder else ''}",
        f"mode={normalize_mode(request.mode)}",
        f"hf={repr(float(request.height_factor))}",
        f"ef={repr(float(request.edge_factor))}",
    ]
    if mode_builds_lattice(request.mode):
        fields.append(f"spacing={repr(float(request.spacing_m))}")
        fields.append(f"radius={repr(float(request.radius_m))}")
    else:
        # What `geom_command` actually sends. A named-locations layout does not
        # depend on the grid numbers, so two runs that differ only in a spin
        # box the tool never sees must share one layout rather than rebuild.
        fields.append("radius=-1")
    for name, lat, lon, height, edge in request.locations:
        fields.append(f"loc={name}|{repr(float(lat))}|{repr(float(lon))}|"
                      f"{repr(float(height))}|{repr(float(edge))}")
    blob = "\n".join(fields).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:LAYOUT_KEY_CHARS]


def auto_cache_path(request, fallback_dir=None):
    """Where a layout for this request lives: beside the mesh, keyed by request.

    BESIDE THE MESH, for `default_cache_path`'s reason -- a layout belongs to
    one mesh and the engine refuses it against any other, while an output
    directory belongs to one run and the next run overwrites it.

    The key is in the STEM, so changing the spacing changes the file and the
    old one is neither overwritten nor picked up. `fallback_dir` is used only
    when the mesh's own directory cannot be written to; the caller says so out
    loud, because a layout that quietly moved is a layout somebody will look
    for where it is not.
    """
    if not request.mesh:
        base = Path(fallback_dir or ".")
        return str(base / f"flux_cubes_{request_key(request)}.fcg")
    mesh = Path(request.mesh)
    name = f"{mesh.stem}_flux_cubes_{request_key(request)}.fcg"
    directory = mesh.parent
    if not os.access(directory, os.W_OK) and fallback_dir:
        directory = Path(fallback_dir)
    return str(directory / name)


def _close(a, b, tol=1e-9):
    a, b = float(a), float(b)
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def cache_satisfies(header, request):
    """Does this layout answer this request? Returns (ok, reason).

    THE HEADER, not the file name. `auto_cache_path` already keeps two
    specifications on two paths, but a file can be copied, renamed or written
    by a different build of the tool, and the header carries what the tool
    MEASURED. A layout that disagrees is refused here rather than handed to a
    run that would discover it after the solve -- or, for the grid numbers,
    would not discover it at all, because the engine reads spacing and radius
    OUT of the cache and has nothing to compare them with.
    """
    if header is None or not header.ok:
        return False, (header.error if header is not None
                       else "there is no layout to check")
    if request.n_faces_core and header.n_faces_core != request.n_faces_core:
        return False, (f"it was built for a {header.n_faces_core:,}-face mesh "
                       f"and this run uses a {request.n_faces_core:,}-face one")
    if mode_builds_lattice(request.mode):
        if header.radius_m < 0:
            return False, ("it carries no grid and this run asks for one")
        if not _close(header.spacing_m, request.spacing_m):
            return False, (f"its grid spacing is {header.spacing_m:g} m and "
                           f"this run asks for {float(request.spacing_m):g} m")
        if not _close(header.radius_m, request.radius_m):
            return False, (f"its grid radius is {header.radius_m:g} m and this "
                           f"run asks for {float(request.radius_m):g} m")
    elif header.radius_m >= 0:
        return False, ("it carries a grid and this run asks for named "
                       "locations only")
    if not _close(header.height_factor, request.height_factor):
        return False, (f"its cubes stand at {header.height_factor:g} x the "
                       f"facet edge and this run asks for "
                       f"{float(request.height_factor):g} x")
    wanted = set(request.named_names)
    missing = sorted(wanted - set(header.names))
    if missing:
        return False, (f"it does not contain {len(missing)} named location(s): "
                       f"{', '.join(missing)}")
    if mode_uses_locations(request.mode) and header.n_named != len(wanted):
        return False, (f"it carries {header.n_named} named location(s) and "
                       f"this run asks for {len(wanted)}")
    if not mode_uses_locations(request.mode) and header.n_named:
        return False, (f"it carries {header.n_named} named location(s) and "
                       f"this run asks for a grid only")
    return True, "it matches this placement"


#: What `layout_plan` decided to do. One of these, never a bare string.
PLAN_NONE = "none"          # cubes are off, or the configuration names a file
PLAN_REUSE = "reuse"        # a layout for exactly this request is on disk
PLAN_BUILD = "build"        # one has to be built before the run starts
PLAN_REFUSE = "refuse"      # it cannot be built here, and why


@dataclass
class LayoutPlan:
    """How this run gets its layout, decided before anything is launched."""
    action: str = PLAN_NONE
    path: str = ""
    reason: str = ""
    argv: list = field(default_factory=list)
    locations_csv: str = ""     # written beside the layout before the build
    n_cubes: int = 0
    request: LayoutRequest = None
    #: Set when the layout landed somewhere other than beside the mesh.
    relocated: str = ""

    @property
    def builds(self):
        return self.action == PLAN_BUILD


def layout_plan(config, project_root, resolve_data_path, face_count,
                executable=None, fallback_dir=None, threads=8):
    """Decide where this run's layout comes from, without building anything.

    Called by the launch check, so every refusal here is one the user reads
    BEFORE the solve rather than after it. The caller does the building; this
    function only decides, which is what makes it testable without a subprocess
    and without Qt.
    """
    flux = _section(config, "flux_cubes")
    if not _truthy(flux.get("enabled"), False):
        return LayoutPlan(reason="flux cubes are off")
    if str(flux.get("cache_file", "") or "").strip():
        return LayoutPlan(
            reason="this configuration names its own layout file")

    request = request_from_config(config, project_root,
                                  resolve=resolve_data_path,
                                  face_count=face_count)
    plan = LayoutPlan(request=request)
    if not request.mesh:
        plan.action = PLAN_REFUSE
        plan.reason = ("there is no terrain mesh to build a cube layout "
                       "against")
        return plan
    mesh = Path(request.mesh)
    if not mesh.exists():
        plan.action = PLAN_REFUSE
        plan.reason = (f"the terrain mesh {mesh} is not on disk, and the cube "
                       f"layout is built against it")
        return plan

    plan.n_cubes = (lattice_cube_count(request.spacing_m, request.radius_m)
                    if mode_builds_lattice(request.mode) else 0)
    if mode_uses_locations(request.mode):
        plan.n_cubes += len(request.locations)
    if plan.n_cubes <= 0:
        plan.action = PLAN_REFUSE
        plan.reason = ("this placement contains no cubes at all -- widen the "
                       "grid radius, or add a named location")
        return plan

    plan.path = auto_cache_path(request, fallback_dir)
    if fallback_dir and Path(plan.path).parent != mesh.parent:
        plan.relocated = (f"{mesh.parent} cannot be written to, so the layout "
                          f"goes to {Path(plan.path).parent} instead")

    existing = Path(plan.path)
    if existing.exists():
        ok, why = cache_satisfies(read_cache_header(existing), request)
        if ok:
            plan.action = PLAN_REUSE
            plan.reason = (f"a cube layout for exactly this placement is "
                           f"already on disk ({existing.name})")
            return plan
        # A file sitting on the request's own path that does NOT answer it is
        # never quietly reused and never quietly overwritten in silence: the
        # rebuild is announced with the reason, so "why did that take a
        # minute" has an answer.
        plan.reason = (f"{existing.name} is on this request's path and does "
                       f"not answer it ({why}), so it is rebuilt")

    if executable is None:
        plan.action = PLAN_REFUSE
        plan.reason = (f"{GEOM_EXECUTABLE} was not found beside the simulation "
                       f"program, so the cube layout cannot be built. Build it "
                       f"with: cmake --build build --target {GEOM_EXECUTABLE}")
        return plan

    plan.action = PLAN_BUILD
    if not plan.reason:
        plan.reason = (f"no cube layout for this placement exists yet, so one "
                       f"is built before the run starts")
    if mode_uses_locations(request.mode):
        plan.locations_csv = locations_to_csv(
            [{"name": n, "lat_deg": lat, "lon_deg": lon,
              **({"height_m": h} if h >= 0 else {}),
              **({"edge_m": e} if e >= 0 else {})}
             for n, lat, lon, h, e in request.locations])
    plan.argv = geom_command(
        executable, request.mesh, plan.path, mode=request.mode,
        spacing_m=request.spacing_m, radius_m=request.radius_m,
        occluder_mesh=request.occluder,
        locations_csv=(str(Path(plan.path).with_suffix(".locations.csv"))
                       if plan.locations_csv else ""),
        threads=threads)
    return plan


@dataclass
class PreflightFindings:
    """What a launch check has to say about the flux-cube block.

    Returned rather than appended to a `PreflightReport` directly, so the whole
    rule set lives in this module beside `engine_support` and is testable
    without building a report -- and so `run_preflight`, which is already long,
    gains one call instead of seventy lines.
    """
    errors: list = field(default_factory=list)
    info: list = field(default_factory=list)
    dataless: list = field(default_factory=list)     # (field, path)
    cloud_paths: list = field(default_factory=list)  # (field, path)
    line: str = ""                                   # the size headline
    #: The `LayoutPlan` the launcher must carry out before it starts the run,
    #: or None. Only ever a PLAN_BUILD plan: a reuse needs no action beyond
    #: writing the path into the run's YAML, which the launcher also does.
    build_plan: object = None

    @property
    def ok(self):
        return not self.errors and not self.dataless


#: Dotted key of the layout file, for the report's `(field, path)` pairs.
CACHE_KEY = "flux_cubes.cache_file"


def preflight_findings(config, project_root, resolve_path, resolve_data_path,
                       is_dataless, is_cloud_path, ply_face_count,
                       geom_executable=None, fallback_dir=None):
    """Everything a launch check can settle about flux cubes, before the run.

    EVERY refusal the flux-cube pass makes fires AFTER the thermal solve --
    it is a post-solve pass, immediately before the location export -- so the
    price of learning about one from the engine is the whole run. All of them
    are decidable here, from the configuration and the layout file.

    The path helpers are passed in rather than imported: this module is
    imported BY `gui_core.preflight` at call time, and reaching back into it at
    import time would be a cycle. They are that module's own functions, so
    there is one resolver, not a second copy of the resolution rules.

    ⛔ A MISSING LAYOUT IS NOT AN ERROR ANY MORE. The placement -- a grid at a
    spacing, out to a radius, plus any named points -- IS the specification, and
    the `.fcg` is the file it gets written into. When the configuration does not
    name one, `layout_plan` works out where it belongs, whether one already on
    disk answers this exact placement, and otherwise hands the launcher the
    command that builds it. What stays an error is a placement that CANNOT be
    built here: no mesh, a mesh that is not on disk, a mesh that is a cloud
    placeholder, no geometry tool, or a placement with no cubes in it.

    Returns `PreflightFindings`; the caller merges it into its report.
    """
    findings = PreflightFindings()
    flux = config.get("flux_cubes") if isinstance(config, dict) else None
    if not isinstance(flux, dict) or not flux.get("enabled"):
        return findings

    for blocker in engine_support(config).blockers:
        findings.errors.append("flux cubes: " + blocker)

    cache = str(flux.get("cache_file", "") or "").strip()
    if not cache:
        # No file named: the placement is the request, and the layout is built
        # (or reused) before the run starts.
        plan = layout_plan(config, project_root, resolve_data_path,
                           ply_face_count, executable=geom_executable,
                           fallback_dir=fallback_dir)
        if plan.action == PLAN_REFUSE:
            findings.errors.append(
                f"flux cubes: the cube layout cannot be prepared for this run "
                f"-- {plan.reason}.")
            return findings
        if plan.action == PLAN_NONE:
            return findings
        # The mesh is read by the layout tool, so it gets the same cloud
        # treatment every other input path gets -- a placeholder would hang the
        # BUILD rather than the run, which is the same freeze in a different
        # place.
        mesh_path = Path(plan.request.mesh)
        if is_dataless(mesh_path):
            findings.dataless.append(("mesh.import_file", str(mesh_path)))
            return findings
        if is_cloud_path(mesh_path):
            findings.cloud_paths.append(("mesh.import_file", str(mesh_path)))
        if plan.relocated:
            findings.info.append("flux cubes: " + plan.relocated + ".")
        if plan.action == PLAN_BUILD:
            findings.build_plan = plan
            findings.info.append(
                f"flux cubes: {plan.reason}. "
                f"{plan.request.describe().capitalize()}, written to "
                f"{plan.path}. The run starts when it is finished.")
            # The size headline is worth having before the build, not after:
            # it is the one number that decides whether this run wants a
            # 500 m grid at all.
            findings.line = _size_line(flux, plan.n_cubes,
                                       len(plan.request.locations), config)
            findings.info.append("flux cubes: " + findings.line)
            findings.info.append(_PASSIVE_NOTE)
            return findings
        # PLAN_REUSE: nothing to build, but the run still has to be told which
        # file it used, and the size line still applies.
        findings.info.append(f"flux cubes: {plan.reason}, so it is reused "
                             f"rather than rebuilt.")
        findings.build_plan = plan
        cache = plan.path

    # The same three checks every other path field gets. They are applied here
    # rather than by adding the key to the pre-flight's own file list because
    # this one is only a path when the cubes are ON: that list is walked
    # against the LIVE configuration, which still carries a layout path from
    # the last time they were, and a launch would be blocked over a file this
    # run never opens.
    cache_path = resolve_path(cache, project_root)
    if not cache_path.exists():
        findings.errors.append(
            f"flux cubes: the cube layout file this configuration names was "
            f"not found: {cache_path}. The engine will not build one during a "
            f"run. Clear the layout file on the card's advanced options and "
            f"the placement above it is built for you, or point at a layout "
            f"that exists.")
        return findings
    if is_dataless(cache_path):
        findings.dataless.append((CACHE_KEY, str(cache_path)))
        return findings
    if is_cloud_path(cache_path):
        findings.cloud_paths.append((CACHE_KEY, str(cache_path)))

    header = read_cache_header(cache_path)
    if not header.ok:
        findings.errors.append("flux cubes: " + header.error)
        return findings

    mesh = str(_section(config, "mesh").get("import_file", "") or "").strip()
    expected = ply_face_count(resolve_data_path(mesh, project_root)) \
        if mesh else None
    if expected and header.n_faces_core != expected:
        findings.errors.append(
            f"flux cubes: the layout was built for a "
            f"{header.n_faces_core:,}-face mesh and this run uses a "
            f"{expected:,}-face one. The engine compares the two and stops. "
            f"Rebuild the layout against this mesh.")

    names = [row["name"] for row in config_locations(config)]
    missing = sorted(n for n in names if n not in header.names)
    if missing:
        findings.errors.append(
            f"flux cubes: the layout does not contain {len(missing)} named "
            f"location(s): {', '.join(missing)}. Editing the list does not "
            f"move a cube -- rebuild the layout, or remove the entries.")

    findings.line = _size_line(flux, header.n_cubes, header.n_named, config)
    findings.info.append("flux cubes: " + findings.line)
    findings.info.append(_PASSIVE_NOTE)
    return findings


#: Said once, in one place, so the launch check and the card cannot drift into
#: two wordings of the one property that matters about this feature.
_PASSIVE_NOTE = (
    "flux cubes: passive -- the probes read the finished solution, so every "
    "mesh temperature and flux is the same as the identical run without them.")


def _size_line(flux, n_cubes, n_named, config):
    """The cubes x faces x components x steps headline, and the bytes.

    Shared by the layout-file branch (where the counts are the layout's own,
    measured) and the about-to-build branch (where they are the placement's,
    computed by `lattice_cube_count`, which is the tool's own rule rather than
    an approximation of it). One function, so the number a user sees before the
    build and the number they see after it are formatted the same way.
    """
    # int(float(...)): a configuration may spell its step count `1e2`, which
    # yaml.safe_load hands back as a STRING and int() refuses.
    try:
        steps = int(float(_section(config, "simulation").get("timesteps", 0)
                          or 0))
    except (TypeError, ValueError):
        steps = 0
    line = (f"{n_cubes:,} cubes x {N_CUBE_FACES} faces x "
            f"{N_COMPONENTS} components x {steps:,} steps")
    if steps > 0:
        line += (f" = about "
                 f"{fmt_bytes(product_bytes(n_cubes, steps, n_named > 0))}"
                 f" of binary flux")
        if flux.get("write_csv", True):
            files = csv_files_written(n_cubes, n_named,
                                      flux.get("write_csv_max_cubes", 0))
            line += (f", plus {files:,} spreadsheet file(s) at roughly "
                     f"{fmt_bytes(csv_bytes(files, steps))}")
    line += (". Estimated from the placement and the step count; each product "
             "also carries a short provenance record not counted here.")
    return line


def default_cache_path(config, project_root=None):
    """Where to propose writing a new layout: beside the terrain mesh.

    Beside the MESH, not in the output directory. A layout belongs to one mesh
    and the engine refuses it against any other, while an output directory
    belongs to one run and is overwritten by the next -- so a layout left there
    would be rebuilt every time for the mesh it already matched.
    """
    mesh = str(_section(config, "mesh").get("import_file", "") or "").strip()
    if not mesh:
        return str(Path(project_root or ".") / "flux_cubes.fcg")
    try:
        from gui_core.preflight import resolve_data_path
        path = Path(resolve_data_path(mesh, project_root))
    except Exception:                                        # noqa: BLE001
        path = Path(mesh)
    return str(path.with_suffix("").with_name(path.stem + "_flux_cubes.fcg"))


def find_geom_executable(project_root=None):
    """The geometry tool beside the engine, or None.

    Looked for beside `ptm_thermal_sim` first, because that is the binary the
    rest of the GUI has already resolved and a cache built by a different
    build than the one that will read it is exactly the mismatch the engine
    refuses on.
    """
    candidates = []
    try:
        from gui_core.simulation_runner import SimulationRunner
        exe = SimulationRunner.find_ptm_executable()
        if exe:
            candidates.append(Path(exe).parent / GEOM_EXECUTABLE)
    except Exception:                                        # noqa: BLE001
        pass
    if project_root:
        candidates.append(Path(project_root) / "build" / "bin"
                          / GEOM_EXECUTABLE)
    for candidate in candidates:
        if candidate.exists() and os.access(candidate, os.X_OK):
            return candidate
    return None


# ---------------------------------------------------------------------------
# Named locations
# ---------------------------------------------------------------------------

def parse_locations_csv(text, allow_header=True):
    """`name,lat_deg,lon_deg[,height_m[,edge_m]]` -> (rows, errors).

    Same file the geometry tool reads (`--locations`), and deliberately the
    same tolerances, transcribed from its own loader: blank lines and `#`
    comments are skipped, three to five fields, a name under 32 characters, a
    latitude in range, and no repeated name. It is parsed HERE as well as there
    so a typo is reported in the window rather than by a tool that has already
    started a ray sweep.

    Malformed lines are collected, never skipped: a location silently dropped
    on import is a cube the user believes is in the product and is not.

    ⛔ THE HEADER RULE IS THE TOOL'S, EXACTLY: a row whose first field is the
    literal `name` is a header, wherever it appears in the file. That means a
    point cannot be called `name` -- both readers would skip it -- and matching
    the tool is the lesser evil, because a row this window kept and the tool
    dropped would change the cube count between the layout and the list the run
    verifies against it. An earlier draft here also accepted `site` and `label`
    as headers, which the tool reads as DATA; that disagreement is gone.

    `allow_header=False` for content that cannot have a header row -- the
    card's own table -- so a row there is never skipped for its name.
    """
    rows, errors = [], []
    seen = set()
    for number, raw in enumerate(str(text or "").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if allow_header and parts and parts[0] == "name":
            continue                                  # a header row
        if len(parts) < 3:
            errors.append(f"line {number}: needs at least name, latitude and "
                          f"longitude -- found {len(parts)} field(s)")
            continue
        if len(parts) > 5:
            errors.append(f"line {number}: expected name, latitude, longitude "
                          f"and optionally height and cube size -- found "
                          f"{len(parts)} field(s)")
            continue
        name = parts[0]
        if not name:
            errors.append(f"line {number}: the name is empty")
            continue
        if "," in name or '"' in name:
            # The card writes this list back out as the comma-separated file
            # the layout tool reads, so a comma in a name would split into two
            # fields there. Refused by name rather than left to surface later
            # as "latitude is not a number" from a different field.
            errors.append(f"line {number}: the name {name!r} contains a comma "
                          f"or a quote; the layout tool reads this list as a "
                          f"comma-separated file and a name cannot contain "
                          f"either")
            continue
        if len(name) > MAX_LOCATION_NAME:
            errors.append(f"line {number}: the name {name!r} is "
                          f"{len(name)} characters; the cache stores at most "
                          f"{MAX_LOCATION_NAME}, and two longer names could "
                          f"be cut down to the same one")
            continue
        if name in seen:
            errors.append(f"line {number}: {name!r} is named twice; outputs "
                          f"are keyed by name, so the second would overwrite "
                          f"the first")
            continue
        try:
            lat = float(parts[1])
            lon = float(parts[2])
        except ValueError:
            errors.append(f"line {number}: latitude {parts[1]!r} and "
                          f"longitude {parts[2]!r} must both be numbers")
            continue
        if not -90.0 <= lat <= 90.0:
            errors.append(f"line {number}: latitude {lat} is outside "
                          f"-90 to 90")
            continue
        row = {"name": name, "lat_deg": lat, "lon_deg": lon}
        for index, key in ((3, "height_m"), (4, "edge_m")):
            if len(parts) > index and parts[index]:
                try:
                    row[key] = float(parts[index])
                except ValueError:
                    errors.append(f"line {number}: {key} {parts[index]!r} is "
                                  f"not a number")
                    row = None
                    break
        if row is None:
            continue
        seen.add(name)
        rows.append(row)
    return rows, errors


def locations_to_csv(rows):
    """The inverse, for writing the file the geometry tool reads."""
    lines = ["name,lat_deg,lon_deg,height_m,edge_m"]
    for row in rows or ():
        height = row.get("height_m")
        edge = row.get("edge_m")
        fields = [str(row.get("name", "")),
                  repr(float(row.get("lat_deg", 0.0))),
                  repr(float(row.get("lon_deg", 0.0)))]
        if height is not None or edge is not None:
            fields.append("" if height is None else repr(float(height)))
        if edge is not None:
            fields.append(repr(float(edge)))
        lines.append(",".join(fields))
    return "\n".join(lines) + "\n"


def config_locations(config):
    """`flux_cubes.locations` from a configuration, as clean rows.

    Tolerant of what an opened file can contain -- a scalar where a list
    belongs, an entry that is not a mapping -- because a malformed value must
    not abort the window.
    """
    raw = _section(config, "flux_cubes").get("locations")
    if not isinstance(raw, (list, tuple)):
        return []
    rows = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name", "") or "").strip()
        if not name:
            continue
        row = {"name": name,
               "lat_deg": _as_float(entry.get("lat_deg"), 0.0),
               "lon_deg": _as_float(entry.get("lon_deg"), 0.0)}
        for key in ("height_m", "edge_m"):
            if entry.get(key) is not None:
                value = _as_float(entry.get(key), -1.0)
                if value >= 0:
                    row[key] = value
        rows.append(row)
    return rows
