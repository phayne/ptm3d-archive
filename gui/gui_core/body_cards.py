"""The body-first surface, as data: which bodies, which run types, what a card
shows, and what an opened file carries that a card does not.

No Qt here. Everything the body-first flow decides is decided in this module so
it can be measured without a window -- ``VisualizationPanel`` still crashes
headless, so anything that can only be checked by launching the application
cannot be checked at all on this machine.

Three rules govern the tables below, and each of them was paid for:

1. **A body gets a button only if the engine implements it.** The list comes
   from :data:`ptm.body_table.IMPLEMENTED_BODIES`, which is generated from
   ``planetary_properties.h``. It is not retyped here. The GUI used to offer
   seven built-in bodies against the engine's three, and the four extras ran to
   completion with Europa's radius and zero absorbed flux on every face
   (``BUGS.md`` ENG-BODY-UNBACKED-RUN). A body added or removed in C++ now
   changes this surface through a generated table and a parity test, not
   through somebody remembering.

2. **A choice a card offers must be backed by something on disk or in the
   corpus.** The Europa terrain list is read out of the shipped
   configurations, not typed in: the DEM stems are not uniform
   (``Pwyll_DEM_750m_SOCET`` beside ``Ridges_DEM_60m_SOCET``) and a plausible
   guess produces a card that looks right and runs a different DEM. For the
   same reason each terrain's latitudes come from the files' own
   ``mesh.latitude_override`` -- the five ``europa_polar_*`` configurations are
   at **87 degrees**, not at the pole, and a card that assumed 90 from the file
   name would silently move every polar run three degrees.

3. **Nothing a card hides is silently dropped or silently applied.**
   :func:`out_of_card_settings` is what makes that checkable: it reports every
   key an opened file sets that the card does not show and that differs from
   the engine default.
"""

from __future__ import annotations

import re
from pathlib import Path

try:                                              # the normal case
    from ptm.body_table import IMPLEMENTED_BODIES
    from ptm.engine_defaults import ENGINE_DEFAULTS
except ImportError:                               # pragma: no cover
    from gui_core import _ptm_bootstrap           # noqa: F401  (adds src/python)
    from ptm.body_table import IMPLEMENTED_BODIES
    from ptm.engine_defaults import ENGINE_DEFAULTS


__all__ = [
    "BODY_PRESENTATION",
    "CARD_PRESETS",
    "card_preset",
    "CUSTOM_ROUTES",
    "GENERIC_MESH_SHAPE",
    "GENERIC_SHAPE_TYPES",
    "RUN_TYPES",
    "RUN_TYPES_BY_KEY",
    "SURFACE_ROUND_TRIP_FLOOR",
    "BodyRoute",
    "DemEntry",
    "SpkCoverage",
    "body_buttons",
    "detect_route",
    "europa_dem_registry",
    "europa_sweep_points",
    "modes_for",
    "out_of_card_settings",
    "run_type_of",
    "run_types_for",
    "run_window",
    "spk_coverage",
    "tiny_surface_refusals",
]


# ---------------------------------------------------------------------------
# 1. The buttons
# ---------------------------------------------------------------------------

class _Mode:
    """One model domain on one body -- 'Global' or 'Local mesh'.

    The split is the ENGINE's, and it is the mesh file alone::

        custom_properties["is_custom_mesh"] = mesh_file.empty() ? "0" : "1"
                                              ptm_thermal_sim.cpp:17036

    ``mesh.use_topographic`` deliberately does not enter it. A topographic run
    is still the procedural icosphere with relief on its vertices, so it keeps
    the ``20 * 4^level`` face count and the per-face albedo map still matches.
    """

    def __init__(self, key, title, summary):
        self.key = key
        self.title = title
        self.summary = summary


GLOBAL = _Mode("global", "Global",
               "Procedural icosphere at a chosen subdivision level.")
LOCAL = _Mode("local", "Local mesh",
              "Imported terrain mesh, placed at a latitude and longitude.")


#: How each ENGINE-IMPLEMENTED body is presented. Keys must be exactly
#: `IMPLEMENTED_BODIES`; `src/python/tests/test_gui_body_first.py` asserts that,
#: so a body added to `planetary_properties.h` fails a test here instead of
#: quietly missing its button.
#:
#: A button carries the BODY NAME and NOTHING ELSE. There is no `subtitle`, no
#: `note` and no `maturity` badge: all three were tried, and any second line
#: under each of five large buttons reads as commentary rather than as a
#: choice. What a button leads to is in its tooltip; anything a body genuinely
#: has to disclose is on that body's card, in context and next to the controls
#: it affects -- `test_gui_body_first.py` holds that line for the one body that
#: has such a disclosure, so removing the button copy cannot quietly remove the
#: warning.
BODY_PRESENTATION = {
    "Moon": dict(
        modes=("global", "local"),
    ),
    "Europa": dict(
        modes=("global", "local"),
    ),
    "Mercury": dict(
        modes=("global",),
        # The two defects that held this button back -- a solar day recorded as
        # the 58.646 d SIDEREAL rotation, and a 500 K ceiling written for icy
        # bodies -- were fixed in the engine on 2026-08-26 (BUGS.md
        # ENG-MERCURY-SOLAR-DAY, ENG-MERCURY-TEMP-CEILING). What remains is
        # thin DATA, and the Mercury CARD says so in a banner at the top,
        # beside the controls it constrains. It used to say so twice, once as a
        # two-word badge on the button; the badge is gone and the banner is the
        # single place that statement lives.
    ),
}

#: The two generic-body routes. Neither is a `simulation.body` name: both write
#: a `generic_body:` block, which is the only way the engine models a body it
#: has no built-in properties for. They are separate buttons because the
#: workflows differ -- an asteroid is defined from an orbit, a spin pole and a
#: shape model, and "some other world" is defined from its surface properties.
CUSTOM_ROUTES = {
    "asteroid": dict(
        title="Asteroid",
        modes=("body",),
    ),
    "custom": dict(
        title="Custom body",
        modes=("body",),
        # What this route is FOR, and what happens if you try to reach it by
        # typing a body name instead, is on the card itself -- see
        # `_card_presets()`'s `custom` summary, which a user reads at the moment
        # the choice is in front of them.
    ),
}


class BodyButton:
    """One large button on the landing screen: the body's name, and only that.

    There is deliberately NO `note`, NO `subtitle` and NO `maturity` field. The
    landing screen carried a paragraph under each button, then one functional
    line, then a two-word badge on the one body that needed a caveat -- each
    read as commentary in front of a choice that is five words wide. A body
    that has something to say says it in the button's tooltip and on its own
    card. Keeping any of the three here unrendered would be worse than either
    choice: somebody would fill it in and it would do nothing. Because
    `body_buttons()` splats the presentation table straight into this
    constructor, any of them put back into either table is a TypeError at
    import, not a silent drop.
    """

    def __init__(self, key, title, modes, builtin=True):
        self.key = key
        self.title = title
        self.modes = tuple(modes)
        self.builtin = builtin

    def __repr__(self):                                   # pragma: no cover
        return f"<BodyButton {self.key}>"


def body_buttons():
    """The landing screen's buttons, in order.

    DERIVED from the engine's own implemented-body list plus the two
    generic-body routes. Nothing here is a second copy of "which bodies exist".
    """
    out = []
    for name in IMPLEMENTED_BODIES:
        presentation = BODY_PRESENTATION.get(name)
        if presentation is None:
            # Loud, not silent. A body the engine implements and this table
            # does not describe would otherwise simply have no way in.
            raise KeyError(
                f"{name} is implemented by the engine but BODY_PRESENTATION "
                f"has no entry for it, so the body-first screen would offer no "
                f"way to reach it. Add one in gui_core/body_cards.py.")
        # The body's own name, in its own capitalization -- not upper-cased.
        # Five shouted words is the register the sweep of 2026-08-27 removed.
        out.append(BodyButton(name, name, builtin=True, **presentation))
    for key, presentation in CUSTOM_ROUTES.items():
        fields = dict(presentation)
        out.append(BodyButton(key, fields.pop("title"), builtin=False,
                              **fields))
    return out


def modes_for(body_key):
    """The model domains a body offers, as `_Mode` objects."""
    lookup = {"global": GLOBAL, "local": LOCAL,
              "body": _Mode("body", "Body definition",
                            "Properties supplied in the generic_body: "
                            "section.")}
    for button in body_buttons():
        if button.key == body_key:
            return tuple(lookup[m] for m in button.modes)
    raise KeyError(f"{body_key!r} is not a body-first route")


# ---------------------------------------------------------------------------
# 2. The run types (DP-10)
# ---------------------------------------------------------------------------

class RunType:
    """A shape of run, chosen after the body and before the options card.

    `sets` are written when the run type is chosen, and reported. `needs` names
    what the run type requires of the body to be offerable at all -- a run type
    with nothing behind it is exactly the class of unbacked button that the
    body-first screen exists to remove.
    """

    def __init__(self, key, title, summary, sets, produces):
        self.key = key
        self.title = title
        self.summary = summary
        self.sets = dict(sets)
        self.produces = produces


RUN_TYPES = (
    RunType(
        key="time_series",
        title="Time series",
        summary=("One temperature field per output sample, resolving the "
                 "diurnal or seasonal cycle and letting any face be plotted "
                 "against time."),
        sets={"output.skip_time_series": False,
              "thermal.compute_statistics": True},
        produces=("a per-sample binary file per field, plus the summary maps"),
    ),
    RunType(
        key="integrated",
        title="Integrated statistics",
        summary=("The same simulation, but only the minimum, mean and maximum "
                 "over the whole window are written. Every per-sample file is "
                 "skipped, which is most of the disk a long run uses."),
        sets={"output.skip_time_series": True,
              "thermal.compute_statistics": True},
        produces="the summary maps in statistics/, and no per-sample files",
    ),
    RunType(
        key="sweep",
        title="Parameter grid",
        summary=("The same configuration run once per point of a grid -- for "
                 "Europa, every Sutton terrain crossed with every latitude the "
                 "shipped configurations use."),
        sets={"output.skip_time_series": False,
              "thermal.compute_statistics": True},
        produces="one output directory per grid point, under a common root",
    ),
)

RUN_TYPES_BY_KEY = {r.key: r for r in RUN_TYPES}


def run_types_for(body_key, mode_key):
    """Which run types this body and mode may offer.

    `sweep` is offered ONLY where a grid is actually backed by something: today
    that is the Europa Sutton-DEM product, the one Cartesian product the
    shipped corpus contains (5 terrains x 9 latitudes). Offering a parameter
    grid on a body with no grid behind it would be a button the project cannot
    honor, which is the defect this whole screen exists to close.
    """
    keys = ["time_series", "integrated"]
    if body_key == "Europa" and mode_key == "local":
        keys.append("sweep")
    return tuple(RUN_TYPES_BY_KEY[k] for k in keys)


def run_type_of(config):
    """Which run type an opened configuration is, by its own keys."""
    output = (config or {}).get("output") or {}
    skip = output.get("skip_time_series", False)
    if isinstance(skip, str):
        skip = skip.strip().lower() in ("true", "yes", "on", "1")
    return "integrated" if skip else "time_series"


# ---------------------------------------------------------------------------
# 3. The Europa terrain registry, read out of the shipped configurations
# ---------------------------------------------------------------------------

class DemEntry:
    """One Sutton DEM, as the shipped configurations describe it."""

    def __init__(self, key, label, posting_m, import_file, view_factor_file,
                 latitudes):
        self.key = key
        self.label = label
        self.posting_m = posting_m
        self.import_file = import_file
        self.view_factor_file = view_factor_file
        self.latitudes = tuple(latitudes)

    def display(self):
        if self.posting_m:
            return f"{self.label} — {self.posting_m:g} m posting"
        return self.label

    def __repr__(self):                                   # pragma: no cover
        return f"<DemEntry {self.key} {self.posting_m}m>"


_DEM_STEM = re.compile(r"^(?P<name>[A-Za-z]+)_DEM_(?P<posting>\d+)m_", re.I)
_EUROPA_CONFIG = re.compile(r"^europa_(lat\d+|polar)_(?P<dem>[a-z]+)\.yaml$")


def europa_dem_registry(repo_root):
    """``{dem_key: DemEntry}`` derived from ``configs/europa_*_<dem>.yaml``.

    Every field is read from the files. The latitude in particular is taken
    from each file's own ``mesh.latitude_override`` and NOT from its name: the
    five ``europa_polar_*`` configurations sit at 87 degrees, and a registry
    that read "polar" as 90 would move every one of those runs.

    Returns an empty mapping when the configurations are not present, so a
    checkout without them shows an empty terrain list rather than a wrong one.
    """
    import yaml

    config_dir = Path(repo_root) / "configs"
    if not config_dir.is_dir():
        return {}

    found = {}
    for path in sorted(config_dir.glob("europa_*.yaml")):
        match = _EUROPA_CONFIG.match(path.name)
        if not match:
            continue
        try:
            doc = yaml.safe_load(path.read_text()) or {}
        except (OSError, ValueError):
            continue
        mesh = doc.get("mesh") or {}
        import_file = str(mesh.get("import_file", "") or "")
        if not import_file:
            continue
        latitude = mesh.get("latitude_override")
        try:
            latitude = float(latitude)
        except (TypeError, ValueError):
            continue
        radiation = doc.get("radiation") or {}
        dem_key = match.group("dem")
        entry = found.setdefault(dem_key, {
            "import_file": import_file,
            "view_factor_file": str(radiation.get("view_factor_file", "") or ""),
            "latitudes": set(),
        })
        entry["latitudes"].add(latitude)

    registry = {}
    for dem_key, fields in found.items():
        stem = Path(fields["import_file"]).stem
        stem_match = _DEM_STEM.match(stem)
        if stem_match:
            label = stem_match.group("name")
            posting = float(stem_match.group("posting"))
        else:
            # Named honestly rather than guessed at. A DEM whose file name does
            # not carry a posting still appears; it just does not claim one.
            label, posting = stem, None
        registry[dem_key] = DemEntry(
            key=dem_key, label=label, posting_m=posting,
            import_file=fields["import_file"],
            view_factor_file=fields["view_factor_file"],
            latitudes=sorted(fields["latitudes"]))
    return registry


def europa_sweep_points(repo_root):
    """``[(dem_key, latitude), ...]`` -- the whole shipped Europa grid.

    Ordered terrain-major, latitude ascending, so the confirmation dialog and
    the generated directory names are stable between runs.
    """
    registry = europa_dem_registry(repo_root)
    points = []
    for dem_key in sorted(registry):
        for latitude in registry[dem_key].latitudes:
            points.append((dem_key, latitude))
    return points


# ---------------------------------------------------------------------------
# 4. Where an opened file lands (DP-7)
# ---------------------------------------------------------------------------

class BodyRoute:
    """Which card an opened configuration belongs on, or why it does not."""

    def __init__(self, body_key=None, mode_key=None, run_type="time_series",
                 refusal=""):
        self.body_key = body_key
        self.mode_key = mode_key
        self.run_type = run_type
        self.refusal = refusal

    @property
    def ok(self):
        return not self.refusal

    def __repr__(self):                                   # pragma: no cover
        return (f"<BodyRoute {self.body_key}/{self.mode_key}/{self.run_type}"
                f"{' REFUSED' if self.refusal else ''}>")


def _mesh_is_imported(config):
    mesh = (config or {}).get("mesh") or {}
    return bool(str(mesh.get("import_file", "") or "").strip())


def _generic_body_active(config):
    generic = (config or {}).get("generic_body") or {}
    if not isinstance(generic, dict):
        return False
    if str(generic.get("config_file", "") or "").strip():
        return True
    if str(generic.get("name", "") or "").strip():
        return True
    orbit = generic.get("orbit") or {}
    if isinstance(orbit, dict):
        try:
            if float(orbit.get("semi_major_axis_au", -1.0)) > 0:
                return True
        except (TypeError, ValueError):
            return False
    return False


#: The only extensions File -> Open accepts. The dialog used to offer "Legacy
#: PTM3D Projects (*.ptm3d)" and "All Files (*)" as well; nothing in this
#: project reads a `.ptm3d` file, and "All Files" invited a `.ply` or a `.bin`
#: into `yaml.safe_load`.
CONFIG_SUFFIXES = (".yaml", ".yml")


def open_refusal(path, config):
    """Why this file cannot be opened as a project, or "".

    Two requirements, and no more (Paul's ruling, DP-7): it is YAML, and it
    names a body. The body key is what decides which card describes the file,
    and there is no honest default -- landing a Europa configuration on the
    Moon card would show lunar values for properties the file never set.

    Everything else about the file is opened as it is. A configuration that
    sets keys no card shows is not refused; it is listed.
    """
    suffix = Path(str(path)).suffix.lower()
    if suffix not in CONFIG_SUFFIXES:
        return (f"{Path(str(path)).name} is not a YAML configuration "
                f"(its extension is {suffix or 'missing'}). PTM3D projects are "
                f"{' or '.join(CONFIG_SUFFIXES)} files.")
    if not isinstance(config, dict):
        return (f"{Path(str(path)).name} does not parse as a configuration: "
                f"its top level is not a mapping of sections.")
    simulation = config.get("simulation")
    body = ""
    if isinstance(simulation, dict):
        body = str(simulation.get("body", "") or "").strip()
    if not body:
        return (f"{Path(str(path)).name} does not set simulation.body, so "
                f"there is no way to tell which body it models. Every PTM3D "
                f"configuration names one; add it to the file, or open the "
                f"file in a text editor to see what it is.")
    return ""


def detect_route(config):
    """Which body card an opened configuration lands on.

    ``simulation.body`` is REQUIRED. A configuration without it is refused
    rather than guessed at, because guessing is how a Europa file would land on
    the Moon card and be shown lunar defaults for values it never set. The
    refusal names the key and offers the full settings view, which reads any
    configuration at all.
    """
    if not isinstance(config, dict):
        return BodyRoute(refusal=(
            "This file does not parse as a configuration (its top level is not "
            "a mapping of sections)."))

    simulation = config.get("simulation")
    if not isinstance(simulation, dict):
        return BodyRoute(refusal=(
            "This file has no simulation: section, so there is no way to tell "
            "which body it models."))

    body = str(simulation.get("body", "") or "").strip()
    if not body:
        return BodyRoute(refusal=(
            "This file does not set simulation.body, so there is no way to "
            "tell which body it models. Add a body to the file, or open it in "
            "All settings, which reads any configuration."))

    if body.startswith("[H] ") or _generic_body_active(config):
        # A generic body. Which of the two generic routes it is cannot be read
        # off the file -- the engine draws no such distinction -- so it lands
        # on the asteroid card, which is the superset: it shows the orbit, the
        # spin and the shape as well as the surface.
        return BodyRoute(body_key="asteroid", mode_key="body",
                         run_type=run_type_of(config))

    for name in IMPLEMENTED_BODIES:
        if body.lower() == name.lower():
            mode = "local" if _mesh_is_imported(config) else "global"
            if mode not in BODY_PRESENTATION[name]["modes"]:
                # Mercury with an imported mesh, say: the engine runs it, and
                # the card does not describe it. Say so instead of showing a
                # global card over a config with a mesh in it.
                return BodyRoute(refusal=(
                    f"This file is a {name} run on an imported mesh, and the "
                    f"{name} card only covers the global domain. Everything in "
                    f"the file is loaded; open All settings to see it."))
            return BodyRoute(body_key=name, mode_key=mode,
                             run_type=run_type_of(config))

    return BodyRoute(refusal=(
        f"simulation.body = {body!r} is not a body this engine implements "
        f"({', '.join(IMPLEMENTED_BODIES)}), and the file carries no "
        f"generic_body: block to describe it. The engine refuses this "
        f"configuration too, rather than substituting another body's "
        f"properties for it."))


# ---------------------------------------------------------------------------
# 5. What the card does not show
# ---------------------------------------------------------------------------

def _flatten(config, prefix=""):
    out = {}
    for key, value in (config or {}).items():
        dotted = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(_flatten(value, dotted + "."))
        else:
            out[dotted] = value
    return out


def out_of_card_settings(config, exposed_keys):
    """``[(key, value, default), ...]`` the card does not show and that differ
    from the engine's own default.

    This is the mechanism behind "nothing is hidden that is set to a
    non-default value". A key the card shows is not reported; a key it does not
    show but that carries the engine default is not reported either, because
    writing a default and omitting it are the same run.
    """
    exposed = set(exposed_keys or ())
    out = []
    for key, value in sorted(_flatten(config).items()):
        if key in exposed:
            continue
        if key not in ENGINE_DEFAULTS:
            # Not an engine key at all (a GUI-synthetic mode, or a section the
            # parser does not read). Reported, because a key nothing reads is
            # worth seeing -- two shipped Europa configurations carry a whole
            # `visualization:` block that no parser touches.
            out.append((key, value, None))
            continue
        default = ENGINE_DEFAULTS[key]
        if value != default:
            out.append((key, value, default))
    return out


# ---------------------------------------------------------------------------
# 6. The defaults each card announces
# ---------------------------------------------------------------------------

def _card_presets():
    """``{(route, mode): Preset}`` -- what choosing a card writes.

    FOUR of the seven are the task-card presets the previous stage shipped,
    used as they are rather than copied. That is the point of DP-5: the task
    cards WERE body cards under other names, and reusing the objects keeps one
    mechanism ("a preset reports exactly the keys it wrote"), one beaming rule,
    and one set of tests. The three new ones fill the routes the task cards
    never covered.
    """
    from gui_core.presets import PRESETS_BY_KEY, Preset

    europa_site = Preset(
        key="europa_dem_site",
        title="Europa Sutton DEM site",
        summary=("One of the five Sutton terrains, placed at one of the "
                 "latitudes the shipped configurations use, with radiation "
                 "exchange between facets on."),
        sets={
            "simulation.body": "Europa",
            "simulation.duration_days": 3.55,
            "simulation.timesteps": 144,
            "thermal.run_thermal_mesh": True,
            "thermal.equilibration_orbits": 20.0,
            "radiation.enabled": True,
            "illumination.albedo_source": "none",
        },
        beaming="off",
        needs=("mesh.import_file -- pick a terrain, which also fills in its "
               "view-factor cache path",),
        beaming_note=(
            "The Smith roughness parameterization is calibrated on LUNAR "
            "regolith and has no measured value for ice, so the control "
            "refuses this body regardless of facet scale."),
    )
    mercury = Preset(
        key="mercury_global",
        title="Mercury global",
        summary=("Mercury over one solar day. The engine forces at the "
                 "175.938-day solar day, which is three sidereal rotations "
                 "per two orbits, and no temperature ceiling is applied."),
        sets={
            "simulation.body": "Mercury",
            "simulation.duration_days": 175.938,
            "simulation.timesteps": 144,
            "mesh.resolution": 4,
            "mesh.import_file": "",
            "mesh.occluder_file": "",
            "mesh.use_topographic": False,
            "mesh.latitude_override": -999.0,
            "mesh.longitude_override": -999.0,
            "thermal.run_thermal_mesh": True,
            "radiation.enabled": False,
            "illumination.albedo_source": "none",
        },
        beaming="n/a",
        beaming_note=(
            "Radiation exchange is off, and the sub-facet roughness slope is "
            "a lunar regolith calibration with no measured Mercury value."),
    )
    custom = Preset(
        key="custom_body",
        title="Generic body",
        summary=("Any world the engine has no built-in properties for. "
                 "Everything the run needs is described here and written to "
                 "the configuration's generic_body: section. This is the only "
                 "way to model such a body: typing its name under "
                 "simulation.body instead will not start a run, because the "
                 "engine has no radius, albedo or orbit to go with the name."),
        sets={
            "thermal.run_thermal_mesh": True,
            "radiation.enabled": False,
            "illumination.albedo_source": "none",
            "illumination.include_primary_reflection": False,
        },
        beaming="off",
        needs=("generic_body.name -- what the body is called",
               "generic_body.shape -- a radius, semi-axes, or a shape mesh",
               "generic_body.surface -- albedo, emissivity, conductivity and "
               "density, which decide the thermal inertia"),
        beaming_note=(
            "Beaming is offered for lunar bodies only: the slope distribution "
            "is a lunar regolith calibration and there is no measured value "
            "for another body here."),
    )
    return {
        ("Moon", "global"): PRESETS_BY_KEY["lunar_global_diurnal"],
        ("Moon", "local"): PRESETS_BY_KEY["lunar_polar_crater"],
        ("Europa", "global"): PRESETS_BY_KEY["europa_diurnal"],
        ("Europa", "local"): europa_site,
        ("Mercury", "global"): mercury,
        ("asteroid", "body"): PRESETS_BY_KEY["generic_body"],
        ("custom", "body"): custom,
    }


_CARD_PRESET_CACHE = {}


def CARD_PRESETS():
    """``{(route, mode): Preset}``, built on first use.

    A FUNCTION, not a module-level dict. ``gui_core.presets`` is imported at
    call time so this module stays importable without the GUI package on the
    path, and the first draft achieved that with a lazily-populated dict
    subclass -- which answered ``in`` and ``pop`` from the empty dict, because
    those were not among the methods it overrode. A function cannot be half
    right in that way.
    """
    if not _CARD_PRESET_CACHE:
        _CARD_PRESET_CACHE.update(_card_presets())
    return _CARD_PRESET_CACHE


def card_preset(route, mode):
    """The `Preset` a route's card applies, or None."""
    return CARD_PRESETS().get((route, mode))


# ---------------------------------------------------------------------------
# 7. The generic-body round-trip floor (PTM-201)
# ---------------------------------------------------------------------------

#: Below this, a `generic_body.surface` number does not survive the trip into
#: the solver. The engine forwards it through `Config::setProperty(key, double)`,
#: which is `std::to_string` -- six decimal places -- so anything under 5e-7
#: arrives as the string "0.000000", fails the receiving `> 0.0` guard, and the
#: solver silently keeps the EUROPA placeholder property, because a generic
#: body's name resolves to Europa by design. Measured: ks = kd = 4e-7 and
#: ks = 0.0096, kd = 0.05 produce byte-identical statistics -- a 24000-fold
#: conductivity error, with nothing said (BUGS.md PTM-201).
SURFACE_ROUND_TRIP_FLOOR = 5e-7

#: The shape types the ENGINE branches on, in its own spelling.
#:
#: `ptm_thermal_sim.cpp:14707` tests `cfg.shape_type == "custom_mesh"` and
#: `:14720` tests `== "ellipsoidal"`; anything else falls through to a sphere.
#: `config_parser.h:675` documents the same three. The parser stores the string
#: VERBATIM and validates nothing, so a control offering "mesh" or "ellipsoid"
#: writes a value the engine does not recognize and the run quietly uses a
#: sphere -- with the user's shape model sitting unread on disk. The first
#: draft of the asteroid card offered exactly those two spellings.
#: `test_gui_body_first.py` asserts each of these literals is still in the C++.
GENERIC_SHAPE_TYPES = ("spherical", "ellipsoidal", "custom_mesh")

#: The shape type for which `generic_body.shape.mesh_file` is read at all, and
#: for which it must be mirrored to `mesh.import_file` -- the thermal solver
#: consumes the TOP-LEVEL mesh path, and the generic-body one is metadata.
GENERIC_MESH_SHAPE = "custom_mesh"

#: The generic-body surface keys that go through that path.
_ROUND_TRIP_KEYS = (
    "generic_body.surface.ks", "generic_body.surface.kd",
    "generic_body.surface.rhos", "generic_body.surface.rhod",
    "generic_body.surface.cp0", "generic_body.surface.Qb",
)


def tiny_surface_refusals(config):
    """Lines naming every surface value too small to survive the round trip.

    An empty list means nothing is at risk. This is the GUI holding a line the
    engine does not yet hold: `ks` below 5e-7 W/m/K is not exotic, it is the
    porous end of the cometary and trans-Neptunian regolith literature, which
    is exactly where someone modeling an insulating lag would go.
    """
    flat = _flatten(config)
    out = []
    for key in _ROUND_TRIP_KEYS:
        raw = flat.get(key)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if value != value:                       # NaN
            continue
        if 0.0 < value < SURFACE_ROUND_TRIP_FLOOR:
            out.append(
                f"REFUSED {key} = {value:g}: values below "
                f"{SURFACE_ROUND_TRIP_FLOOR:g} are written to the solver with "
                f"six decimal places, so this arrives as 0.000000, fails the "
                f"engine's positive-value check, and the run silently uses "
                f"Europa's property instead. Scale your model, or use the "
                f"nearest value at or above the floor.")
    return out


# ---------------------------------------------------------------------------
# 8. SPK coverage (the asteroid card)
# ---------------------------------------------------------------------------

class SpkCoverage:
    """What an SPK actually covers, and whether a window falls inside it.

    ``available`` is False when the question could not be ASKED -- no
    ``spiceypy``, no file, an unreadable kernel. That is deliberately distinct
    from "the window is outside": the engine's Keplerian fallback engages
    per-epoch with no logging at all (``illumination_model.cpp:2793-2818``),
    which is how fifteen Justitia configurations ran 100 percent Keplerian
    while recording ``ephemeris_source: spice``. A card that reported "could
    not check" as "fine" would reproduce exactly that failure.
    """

    def __init__(self, available, start_utc="", end_utc="", days=0.0,
                 detail="", bodies=()):
        self.available = available
        self.start_utc = start_utc
        self.end_utc = end_utc
        self.days = days
        self.detail = detail
        self.bodies = tuple(bodies)

    #: The shape `spkcov` reports through `et2utc(..., "ISOC")`, and the shape
    #: `simulation.start_time` is written in. A lexicographic comparison of two
    #: ISO-8601 timestamps is correct ONLY when both are this shape, so the
    #: shape is checked rather than assumed -- a config carrying
    #: `2026-1-1 00:00` would otherwise compare as later than `2034-10-12...`
    #: and the card would report "inside the covered window" for a run that is
    #: nowhere near it.
    _ISO = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")

    def covers(self, start_utc, end_utc):
        """True/False/None -- None meaning the question could not be asked."""
        if not self.available:
            return None
        for value in (start_utc, end_utc, self.start_utc, self.end_utc):
            if not self._ISO.match(str(value or "")):
                return None
        start_utc, end_utc = str(start_utc)[:19], str(end_utc)[:19]
        return bool(self.start_utc[:19] <= start_utc
                    and end_utc <= self.end_utc[:19])


def run_window(config):
    """``(start, end, spin_up_start, note)`` in ISO-8601, or ``(None, ...)``.

    The SPIN-UP reach-back is the part that matters and the part nothing
    reports today: equilibration runs BEFORE ``simulation.start_time``, so a
    kernel that covers the output window can still leave every spin-up epoch to
    the silent Keplerian fallback -- which is how a run's temperatures can be
    set by orbital elements while its record says SPICE.

    It is only computed when the rotation period is known, and it is labeled
    approximate: the engine's equilibration counts SOLAR days, and a generic
    body's configuration carries the sidereal rotation period. The two differ,
    and reporting a reach-back to the second would be inventing precision.
    """
    from datetime import datetime, timedelta

    simulation = (config or {}).get("simulation") or {}
    raw = str(simulation.get("start_time", "") or "").strip()
    try:
        start = datetime.fromisoformat(raw.replace("Z", ""))
    except (TypeError, ValueError):
        return (None, None, None,
                f"simulation.start_time = {raw!r} is not an ISO-8601 "
                f"timestamp, so the run window cannot be compared with any "
                f"kernel's coverage.")

    def number(node, key, fallback):
        try:
            value = float((node or {}).get(key, fallback))
        except (TypeError, ValueError):
            return fallback
        return fallback if value != value else value

    duration = number(simulation, "duration_days", 0.0)
    end = start + timedelta(days=max(0.0, duration))

    thermal = (config or {}).get("thermal") or {}
    cycles = number(thermal, "equilibration_orbits", 0.0)
    spin = ((config or {}).get("generic_body") or {}).get("spin") or {}
    hours = number(spin, "period_hours", -1.0)
    if cycles > 0 and hours > 0:
        spin_up = start - timedelta(days=cycles * hours / 24.0)
        note = (f"spin-up reaches back about {cycles * hours / 24.0:.1f} days, "
                f"using the {hours:g} h rotation period as an approximation "
                f"for the solar day")
    else:
        spin_up, note = None, (
            "the spin-up reach-back is not shown: it needs both a positive "
            "equilibration count and a rotation period")
    iso = "%Y-%m-%dT%H:%M:%S"
    return (start.strftime(iso), end.strftime(iso),
            spin_up.strftime(iso) if spin_up else None, note)


def spk_coverage(spk_path, metakernel=None):
    """Read an SPK's own coverage window with ``spkcov``.

    Nothing in this repository has ever checked this -- ``grep -rn
    'spkcov|spkobj'`` over ``src/cpp``, ``src/python``, ``gui`` and ``scripts``
    returned nothing before today. The cost is on the record: the committed
    Justitia SPK covers 2034-10-12 to 2034-11-11, every SPICE-mode Justitia
    configuration targets 2035-08-24 onward, and all fifteen ran entirely on
    the silent Keplerian fallback.
    """
    path = Path(str(spk_path or "")).expanduser()
    if not str(spk_path or "").strip():
        return SpkCoverage(False, detail="no kernel selected")
    if not path.exists():
        return SpkCoverage(False, detail=f"{path} is not on disk")
    try:
        import spiceypy
    except ImportError as exc:
        return SpkCoverage(False, detail=(
            f"spiceypy is not available here ({exc}), so this kernel's "
            f"coverage cannot be read. Check it yourself before trusting a "
            f"SPICE run: the engine falls back to Keplerian elements per "
            f"epoch and says nothing."))
    try:
        if metakernel and Path(metakernel).exists():
            spiceypy.furnsh(str(metakernel))
        ids = spiceypy.spkobj(str(path))
        body_ids = [int(ids[i]) for i in range(len(ids))]
        if not body_ids:
            return SpkCoverage(False, detail=f"{path.name} declares no bodies")
        starts, ends = [], []
        for body_id in body_ids:
            window = spiceypy.stypes.SPICEDOUBLE_CELL(2000)
            spiceypy.spkcov(str(path), body_id, window)
            for i in range(spiceypy.wncard(window)):
                low, high = spiceypy.wnfetd(window, i)
                starts.append(low)
                ends.append(high)
        if not starts:
            return SpkCoverage(False, detail=(
                f"{path.name} declares bodies but no coverage intervals"))
        low, high = min(starts), max(ends)
        start_utc = spiceypy.et2utc(low, "ISOC", 0)
        end_utc = spiceypy.et2utc(high, "ISOC", 0)
        return SpkCoverage(True, start_utc, end_utc,
                           (high - low) / 86400.0,
                           detail=f"{path.name}", bodies=body_ids)
    except Exception as exc:                              # noqa: BLE001
        return SpkCoverage(False, detail=(
            f"{path.name} could not be read by SPICE ({exc}). Treat this as "
            f"UNKNOWN coverage, not as coverage."))
