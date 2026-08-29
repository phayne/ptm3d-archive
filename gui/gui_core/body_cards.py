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

import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)

try:                                              # the normal case
    from ptm.body_capabilities import (HELIOCENTRIC_PERIOD_DAYS,
                                       kepler_period_days)
    from ptm.body_table import BODY_CAPABILITIES, IMPLEMENTED_BODIES
    from ptm.engine_defaults import ENGINE_DEFAULTS
except ImportError:                               # pragma: no cover
    from gui_core import _ptm_bootstrap           # noqa: F401  (adds src/python)
    from ptm.body_capabilities import (HELIOCENTRIC_PERIOD_DAYS,
                                       kepler_period_days)
    from ptm.body_table import BODY_CAPABILITIES, IMPLEMENTED_BODIES
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
    "SAMPLES_PER_SYNODIC_DAY",
    "EQ_VALIDATED_CYCLES",
    "EQ_REFERENCE_THERMAL_B",
    "WINDOW_UNITS",
    "WINDOW_UNIT_DAYS",
    "WINDOW_UNIT_DIURNAL",
    "WINDOW_UNIT_ORBITAL",
    "BodyRoute",
    "default_window_unit",
    "orbital_period_days",
    "window_unit_days",
    "DemEntry",
    "SpkCoverage",
    "auto_equilibration_cycles",
    "body_buttons",
    "default_duration_days",
    "detect_route",
    "samples_for_window",
    "synodic_period_days",
    "europa_dem_registry",
    "europa_sweep_points",
    "kernel_architecture",
    "metakernel_members",
    "modes_for",
    "out_of_card_settings",
    "run_type_of",
    "run_types_for",
    "run_window",
    "spk_coverage",
    "tiny_surface_refusals",
    "MetakernelMembers",
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
               "Icosphere at a chosen subdivision level.")
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


#: A run type's `summary` is drawn INSIDE its button, which is 180 px wide at
#: its narrowest. Keep these short: the fuller wording belongs in the button's
#: tooltip and in the "Produces:" line under it, both of which have room.
#: `test_gui_button_text_fits.py` measures each one against the button's own
#: font metrics at the button's minimum size and fails if it no longer fits.
RUN_TYPES = (
    RunType(
        key="time_series",
        title="Time series",
        summary="One temperature field per output sample.",
        sets={"output.skip_time_series": False,
              "thermal.compute_statistics": True},
        produces=("a per-sample binary file per field, plus the summary maps. "
                  "Resolves the diurnal or seasonal cycle, so any face can be "
                  "plotted against time"),
    ),
    RunType(
        key="integrated",
        title="Integrated statistics",
        summary="Minimum, mean and maximum only.",
        sets={"output.skip_time_series": True,
              "thermal.compute_statistics": True},
        produces=("the summary maps in statistics/, and no per-sample files. "
                  "The same simulation as a time series, without the files "
                  "that are most of the disk a long run uses"),
    ),
    RunType(
        key="sweep",
        title="Parameter grid",
        summary="One run per point of a grid.",
        sets={"output.skip_time_series": False,
              "thermal.compute_statistics": True},
        produces=("one output directory per grid point, under a common root. "
                  "For Europa the grid is every DEM terrain crossed with "
                  "every latitude the shipped configurations use"),
    ),
)

RUN_TYPES_BY_KEY = {r.key: r for r in RUN_TYPES}


def run_types_for(body_key, mode_key):
    """Which run types this body and mode may offer.

    `sweep` is offered ONLY where a grid is actually backed by something: today
    that is the Europa DEM product, the one Cartesian product the shipped
    corpus contains (5 terrains x 9 latitudes). Offering a parameter grid on a
    body with no grid behind it would be a button the project cannot honor,
    which is the defect this whole screen exists to close.
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
# 2b. The durations a card proposes, and the spin-up it proposes with them
# ---------------------------------------------------------------------------
#
# All of this is HERE, not in the Qt panel, for the reason at the top of the
# file: it can then be measured without a window.

#: Output samples per synodic day, for a duration a card proposes itself.
#:
#: The number is the shipped corpus's own: both `lunar_global_diurnal` and
#: `europa_diurnal` pair one synodic day with `simulation.timesteps: 144`
#: (`gui_core/presets.py`). It is NOT the engine default (72), which no shipped
#: preset uses.
#:
#: For reference and not as the target: `docs/equilibration-paper.md` §6
#: measures the worst-case per-layer error against samples per diurnal cycle --
#: 11.9 K at 24, 0.99 K at 192, 0.21 K at 768, scaling as O(dt^2). 144 sits
#: just under the 192 the paper calls the production cadence.
SAMPLES_PER_SYNODIC_DAY = 144

#: Spin-up cycles the paper's own cross-validation used, and the top of the
#: band it quotes for production. See :func:`auto_equilibration_cycles`.
EQ_VALIDATED_CYCLES = 6.0

#: Skin depths to the bottom of the column at the engine's DEFAULT depth grid
#: -- `thermal.thermal_b`, which is `config.thermal.grid_b`
#: (`config_parser.cpp:404`, default 15 at `config_parser.h:216`).
#: `eq_certificate.cpp:381` confirms the reading: the grid's skin depth is
#: `z_max / grid_b`, so `grid_b` is the column depth measured in skin depths.
#:
#: Read only. Retuning `thermal_b` is a known trap and nothing here writes it.
EQ_REFERENCE_THERMAL_B = 15.0


def synodic_period_days(body_key, config=None):
    """``(days, provenance)`` -- one solar day on this body, or ``(None, why)``.

    From ``ptm.body_table``, which is GENERATED from the engine's own
    ``planetary_properties.h`` and re-derived by
    ``src/python/tests/test_body_capabilities.py`` on every run. That is the
    point of reading it from there rather than typing 29.530588 into the GUI:
    a period retuned in the C++ becomes a test failure and a diff, not a card
    that quietly proposes a window the engine does not force at.

    ``solar_day_days`` is the SOLAR (synodic) day and not the sidereal rotation
    period -- the table says so itself, and the two differ by three sidereal
    rotations per two orbits on Mercury.

    For a GENERIC body there is no table row, so it is computed the way
    ``GenericBody::updateDerivedProperties`` computes it
    (``generic_body.h:39-49``): from the SIDEREAL rotation period
    ``generic_body.spin.period_hours`` and the orbital period Kepler's third
    law gives for ``generic_body.orbit.semi_major_axis_au``,

        1/P_solar = |1/P_sidereal - 1/P_orbital|

    with the synchronous case (the two within a second of each other) falling
    back to the sidereal period, exactly as the engine does. That is the period
    the engine FORCES at, so it is the one a window should be counted in.

    Without a semi-major axis the sidereal period is returned and LABELED as
    the approximation it is, because the card would otherwise present a
    sidereal period as a solar day. The two differ by the ratio of the rotation
    period to the year -- 0.09% on a 33.8 h rotator at 2.62 AU -- so this is a
    labeling question, not a large number.
    """
    row = BODY_CAPABILITIES.get(body_key)
    if row is not None:
        try:
            days = float(row["solar_day_days"])
        except (KeyError, TypeError, ValueError):
            days = 0.0
        if days > 0:
            return days, f"{body_key} solar day, from the engine's body table"

    generic = (config or {}).get("generic_body") or {}
    spin = generic.get("spin") or {}
    try:
        hours = float(spin.get("period_hours", 0.0) or 0.0)
    except (TypeError, ValueError):
        hours = 0.0
    if hours > 0:
        sidereal = hours / 24.0
        orbital = kepler_period_days(
            (generic.get("orbit") or {}).get("semi_major_axis_au", 0.0))
        if orbital and abs(sidereal - orbital) > 1.0 / 86400.0:
            solar = 1.0 / abs(1.0 / sidereal - 1.0 / orbital)
            return solar, ("solar day from generic_body.spin.period_hours and "
                           "the orbit, by the same formula the engine uses")
        if orbital:
            return sidereal, ("rotation period from "
                              "generic_body.spin.period_hours; this body is "
                              "synchronous, so that is its solar day too")
        return (sidereal,
                "rotation period from generic_body.spin.period_hours, which is "
                "the SIDEREAL period and so an approximation to the solar day "
                "until generic_body.orbit.semi_major_axis_au is set")
    return None, (f"no synodic period is known for {body_key or 'this body'}: "
                  f"it is not a built-in body and generic_body.spin."
                  f"period_hours is not set")


def orbital_period_days(body_key, config=None):
    """``(days, provenance)`` -- one HELIOCENTRIC orbit, or ``(None, why)``.

    Around the SUN, on every body, including the moons. That is the distinction
    that matters and the one the engine does not make: ``props.year`` is the
    period the equilibration machinery counts against, and for Europa the
    engine sets it equal to the day -- the 3.55 d orbit around Jupiter
    (``planetary_properties.h:189``). Offering that as "one orbital period"
    would give Europa two controls with the same number in them, and would call
    3.55 d a seasonal cycle.

    ``ptm.body_capabilities.HELIOCENTRIC_PERIOD_DAYS`` carries the built-in
    bodies' values and ``test_body_capabilities.py`` holds each of them against
    Kepler's third law on that body's own engine ``rAU``.

    For a GENERIC body it is computed from
    ``generic_body.orbit.semi_major_axis_au`` by the same formula the engine
    uses (``OrbitalElements::orbitalPeriod``), so editing the semi-major axis
    moves the period the card offers -- there is nothing cached to go stale.
    """
    period = HELIOCENTRIC_PERIOD_DAYS.get(body_key)
    if period:
        return float(period), (f"{body_key}'s orbit around the Sun, which is "
                               f"also its seasonal cycle")

    orbit = ((config or {}).get("generic_body") or {}).get("orbit") or {}
    a_au = orbit.get("semi_major_axis_au", 0.0)
    days = kepler_period_days(a_au)
    if days:
        return days, (f"Kepler's third law on the semi-major axis this body "
                      f"is given, {float(a_au):g} AU — the same formula the "
                      f"engine uses")
    return None, (f"no orbital period is known for {body_key or 'this body'}: "
                  f"it is not a built-in body and "
                  f"generic_body.orbit.semi_major_axis_au is not set")


#: The units a run WINDOW may be given in, in the order the combo offers them.
#:
#: SEASONAL IS NOT AMONG THEM, and that is a measurement rather than an
#: omission. The seasonal cycle is the period of the subsolar latitude, and on
#: every body this project ships it IS the heliocentric orbit: Europa's
#: seasons are Jupiter's trip around the Sun, Mercury's are its own 87.969 d
#: year, and a generic body's are one orbit of whatever semi-major axis it was
#: given. The Moon is the only body with a formally distinct number -- its
#: subsolar latitude cycles on the draconic year, 346.62 d rather than 365.25,
#: because the lunar node regresses in 18.6 years -- and that is a 5%
#: difference on a +-1.54 degree excursion, which does not earn a fourth entry
#: in a combo. A "seasonal periods" unit would therefore be a second name for
#: "orbital periods" on every body but one, and on that one it would be a
#: number the engine's own geometry does not key on.
WINDOW_UNIT_DIURNAL = "diurnal periods"
WINDOW_UNIT_ORBITAL = "orbital periods"
WINDOW_UNIT_DAYS = "days"
WINDOW_UNITS = (WINDOW_UNIT_DIURNAL, WINDOW_UNIT_ORBITAL, WINDOW_UNIT_DAYS)


def window_unit_days(unit, body_key, config=None):
    """``(days_per_unit, provenance)`` for a window unit, or ``(None, why)``.

    The ONE place a unit name becomes a number. A control converts through
    this and nothing else, so "1 orbital period" means the same span wherever
    it is written and a body with no such period cannot be given a window in
    it -- ``None`` is returned and the caller must refuse, not fall back to
    days and let the user believe they asked for an orbit.
    """
    if unit == WINDOW_UNIT_DAYS:
        return 1.0, "days"
    if unit == WINDOW_UNIT_DIURNAL:
        days, why = synodic_period_days(body_key, config)
        return (days, why) if days else (None, why)
    if unit == WINDOW_UNIT_ORBITAL:
        days, why = orbital_period_days(body_key, config)
        return (days, why) if days else (None, why)
    return None, (f"{unit!r} is not one of the window units this card offers "
                  f"({', '.join(WINDOW_UNITS)})")


def default_window_unit(body_key, config=None):
    """The unit a window control starts in: diurnal periods where there are any.

    ONE DIURNAL PERIOD is the default window (:func:`default_duration_days`),
    so the control opens in the unit that shows it as the number 1. A body
    whose diurnal period is unknown -- a generic body with no rotation period
    yet -- opens in days instead, because a control that opened in a unit it
    cannot convert would be showing a window nobody chose.
    """
    days, _why = window_unit_days(WINDOW_UNIT_DIURNAL, body_key, config)
    return WINDOW_UNIT_DIURNAL if days else WINDOW_UNIT_DAYS


def default_duration_days(body_key, config=None):
    """``(days, provenance)`` for a card's proposed run length, or ``(None, why)``.

    ONE synodic period. A diurnal cycle is the shortest window in which every
    face of a body is both lit and unlit, so it is the shortest window whose
    statistics mean anything; it is also what both shipped diurnal presets
    already use.
    """
    days, why = synodic_period_days(body_key, config)
    if days is None:
        return None, why
    return days, f"one diurnal cycle ({days:g} d) — {why}"


def samples_for_window(window_days, synodic_days,
                       per_day=SAMPLES_PER_SYNODIC_DAY):
    """``simulation.timesteps`` for a window, at the corpus sampling cadence.

    Keeps the sampling density fixed at :data:`SAMPLES_PER_SYNODIC_DAY` per
    synodic day however long the window is, so a two-day window is sampled
    twice as often as a one-day window rather than being smeared over the same
    144 slots. At least 2 -- one sample is not a window -- and bounded by the
    range the control offers.
    """
    try:
        window = float(window_days)
        synodic = float(synodic_days)
    except (TypeError, ValueError):
        return int(per_day)
    if not (window > 0) or not (synodic > 0):
        return int(per_day)
    return max(2, min(100000, int(round(per_day * window / synodic))))


def auto_equilibration_cycles(body_key, config=None):
    """``(cycles, reason)`` -- the spin-up length a card proposes.

    **This is the GUI's rule, not the engine's.** The engine has no ``auto``:
    ``thermal.equilibration_orbits`` is a plain double
    (``config_parser.cpp:405``) whose default is 1.0, and there is no pre-run
    estimator in the C++ to ask -- ``ptm_eq_certify`` certifies a run that has
    already happened. So the number is computed here, shown with its reasoning,
    and left editable. Moving it into the engine is a separate change.

    THE RULE, and where each part of it comes from:

    * **A floor of 6 cycles.** ``docs/equilibration-paper.md`` §6 validates the
      full equilibration-plus-output pipeline against Hayne et al. (2017) using
      **six** equilibration orbits, and §7 quotes 2-6 as the production band
      for large meshes. The paper's own sufficiency claim is two cycles "to
      reach the periodic attractor to sub-Kelvin accuracy at all latitudes",
      but two is stated for near-isothermal regolith at the diurnal period and
      a card cannot know the user is in that regime. Six is the number the
      project actually validated against data, so six is the floor.

      It is deliberately well above the engine default of 1.0, which is below
      even the paper's minimum.

    * **Scaled by the depth of the column, as its square.** The spin-up has to
      relax the deepest layer that participates. For a column of depth L the
      slowest diffusive mode relaxes on tau = 4L^2/(pi^2 kappa); with
      L = b * skin depth and skin depth = sqrt(kappa P / pi), that is
      tau/P = 4 b^2 / pi^3 -- quadratic in b, and independent of the
      conductivity, which cancels. The paper's 2-6 band is quoted at the
      engine's default grid, so the floor is scaled by ``(b / 15)^2`` and a
      deeper column asks for proportionally more spin-up. At the default
      ``thermal_b`` this factor is exactly 1 and the answer is 6.

    * **At least the length of the run itself.** A window longer than one
      diurnal cycle carries forcing periods longer than a diurnal cycle, and a
      spin-up shorter than the window cannot have settled them.

    The thermal inertia does NOT enter. That is the point of working in skin
    depths: the diffusivity cancels out of tau/P, so a low-inertia and a
    high-inertia surface need the same NUMBER of cycles, differing only in how
    much wall clock each cycle costs.

    A number is ALWAYS returned: the floor and the depth scaling are both
    period-independent, so a body with no known synodic period still gets the
    six-cycle floor -- only the "at least the length of the run" term needs the
    period, and it is skipped when there is none. Rounded UP to a whole cycle,
    because the engine counts integrated diurnal cycles.
    """
    synodic, _why = synodic_period_days(body_key, config)
    config = config or {}
    thermal = config.get("thermal") or {}
    simulation = config.get("simulation") or {}

    def number(node, key, fallback):
        try:
            value = float(node.get(key, fallback))
        except (TypeError, ValueError):
            return fallback
        return fallback if value != value else value

    thermal_b = number(thermal, "thermal_b", EQ_REFERENCE_THERMAL_B)
    if not (thermal_b > 0):
        thermal_b = EQ_REFERENCE_THERMAL_B
    depth_factor = (thermal_b / EQ_REFERENCE_THERMAL_B) ** 2
    floor = EQ_VALIDATED_CYCLES * depth_factor

    cycles = floor
    reason = (f"{EQ_VALIDATED_CYCLES:g} cycles, the number the equilibration "
              f"paper validated against Hayne et al. (2017)")
    if abs(depth_factor - 1.0) > 1e-9:
        reason += (f", scaled by (thermal_b/{EQ_REFERENCE_THERMAL_B:g})^2 = "
                   f"{depth_factor:.2f} for the deeper column")

    duration = number(simulation, "duration_days", 0.0)
    if synodic and synodic > 0 and duration > 0:
        window_cycles = duration / synodic
        if window_cycles > cycles:
            cycles = window_cycles
            reason = (f"{window_cycles:.2f} cycles, the length of the run "
                      f"itself ({duration:g} d over a {synodic:g} d diurnal "
                      f"cycle) — a spin-up shorter than the window cannot "
                      f"have settled it")
    import math
    cycles = float(math.ceil(cycles - 1e-9))
    return cycles, reason


# ---------------------------------------------------------------------------
# 3. The Europa terrain registry, read out of the shipped configurations
# ---------------------------------------------------------------------------

class DemEntry:
    """One Europa DEM, as the shipped configurations describe it."""

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
    from gui_core.presets import (BUILTIN_BODY_CLEARS, GENERIC_BODY_CLEARS,
                                  PRESETS_BY_KEY, Preset)

    europa_site = Preset(
        key="europa_dem_site",
        title="Europa DEM site",
        summary=("One of the five Europa DEM terrains, placed at one of the "
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
        # The mesh keys are NOT cleared here; see the lunar crater preset. A
        # DIFFERENT body's terrain is cleared on the body change, by
        # BodyFirstPanel._apply_route_defaults.
        clears=BUILTIN_BODY_CLEARS,
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
        clears=BUILTIN_BODY_CLEARS,
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
            # See the `generic_body` preset in gui_core/presets.py: a
            # placeholder the card's own mirror of generic_body.name replaces,
            # here so the route defaults never leave the previous card's body
            # name under a generic_body: section.
            "simulation.body": "Custom",
            "thermal.run_thermal_mesh": True,
            "radiation.enabled": False,
            "illumination.albedo_source": "none",
            "illumination.include_primary_reflection": False,
        },
        clears=GENERIC_BODY_CLEARS,
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
                 detail="", bodies=(), intervals=()):
        self.available = available
        self.start_utc = start_utc
        self.end_utc = end_utc
        self.days = days
        self.detail = detail
        self.bodies = tuple(bodies)
        #: Every covered interval, as ``(start_utc, end_utc)`` pairs. An SPK
        #: may hold two arcs with a hole between them, and `start_utc` /
        #: `end_utc` are only the OUTER ENVELOPE of those arcs. A run window
        #: sitting in the hole is inside the envelope and outside the kernel,
        #: which is where the silent per-epoch fallback lives -- so `covers`
        #: asks about the intervals when they are known. Empty means "one
        #: interval, or not recorded", and the envelope is then the answer.
        self.intervals = tuple(intervals)

    #: The shape `spkcov` reports through `et2utc(..., "ISOC")`, and the shape
    #: `simulation.start_time` is written in. A lexicographic comparison of two
    #: ISO-8601 timestamps is correct ONLY when both are this shape, so the
    #: shape is checked rather than assumed -- a config carrying
    #: `2026-1-1 00:00` would otherwise compare as later than `2034-10-12...`
    #: and the card would report "inside the covered window" for a run that is
    #: nowhere near it.
    _ISO = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")

    def covers(self, start_utc, end_utc):
        """True/False/None -- None meaning the question could not be asked.

        With more than one interval the window must fit inside ONE of them.
        Reporting a run that falls in a gap as covered would be the same lie
        the envelope tells, with a "WITH GAPS" note printed above it.
        """
        if not self.available:
            return None
        for value in (start_utc, end_utc, self.start_utc, self.end_utc):
            if not self._ISO.match(str(value or "")):
                return None
        start_utc, end_utc = str(start_utc)[:19], str(end_utc)[:19]
        windows = [(low, high) for low, high in self.intervals
                   if self._ISO.match(str(low or ""))
                   and self._ISO.match(str(high or ""))]
        if len(windows) != len(self.intervals):
            return None
        if not windows:
            windows = [(self.start_utc, self.end_utc)]
        return any(str(low)[:19] <= start_utc and end_utc <= str(high)[:19]
                   for low, high in windows)


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


def _default_project_root():
    """The repository root, from this file's own location.

    ``gui/gui_core/body_cards.py`` -> ``parents[2]``. The same derivation
    :func:`_ensure_leapseconds` has always used to find ``data/kernels``, kept
    in one place now that kernel paths are resolved here as well.
    """
    return Path(__file__).resolve().parents[2]


def _spice_kernel_dirs(project_root, env=None):
    """The kernel directories the engine searches, in the engine's order.

    ``docs/configuration.md`` "How ``generic_body.spice_metakernel`` is
    resolved": ``$PTM_SPICE_KERNELS``, then ``$PTM_DATA_ROOT/kernels``, then
    the repository's ``data/kernels``. Each rung REPLACES the one below it for
    the engine; here they are all offered, because a card that could not find
    a kernel the run will find is worse than one that looks in an extra place.
    """
    env = env if env is not None else os.environ
    dirs = []
    explicit = str(env.get("PTM_SPICE_KERNELS", "") or "").strip()
    if explicit:
        dirs.append(Path(explicit).expanduser())
    data_root = str(env.get("PTM_DATA_ROOT", "") or "").strip()
    if data_root:
        dirs.append(Path(data_root).expanduser() / "kernels")
    dirs.append(Path(project_root) / "data" / "kernels")
    return dirs


def _resolve_kernel_path(value, project_root=None, env=None):
    """``(path, candidates)`` for a configured kernel value.

    Three forms, and the engine treats them differently
    (``illumination_model.cpp:2678-2696``, tabulated in
    ``docs/configuration.md``). The card has to agree with the run about which
    file is meant, or it reports on a file the run will not open:

    * ABSOLUTE -- used exactly as given. Prepending the kernel directory to an
      absolute path is PTM-145, and it is why every EMA asteroid configuration
      missed its metakernel.
    * relative ``data/...`` -- the ``$PTM_DATA_ROOT`` contract shared with
      ``mesh.import_file``. This is what the seven cached ``body_config.yaml``
      files write, and resolving it against the WORKING DIRECTORY would answer
      a question about where the GUI was launched from.
    * any other relative value, e.g. the shipped ``bennu.tm`` -- the kernel
      directory first, then the working directory.

    ``candidates`` is every path tried, so a value that resolves nowhere can be
    reported the way the engine reports it: by naming them all.
    """
    raw = str(value or "").strip()
    if not raw:
        return None, []
    root = Path(project_root) if project_root else _default_project_root()
    expanded = Path(raw).expanduser()
    if expanded.is_absolute():
        return expanded, [expanded]

    normalized = raw.replace("\\", "/")
    if normalized == "data" or normalized.startswith("data/"):
        try:
            from gui_core.preflight import resolve_data_path
            resolved = Path(resolve_data_path(raw, root, env)).expanduser()
        except Exception:                                # noqa: BLE001
            resolved = root / expanded
        return resolved, [resolved]

    candidates = [directory / expanded
                  for directory in _spice_kernel_dirs(root, env)]
    cwd = Path.cwd() / expanded
    if cwd not in candidates:
        candidates.append(cwd)
    for candidate in candidates:
        if candidate.exists():
            return candidate, candidates
    return candidates[0], candidates


def kernel_architecture(path):
    """``(architecture, type)`` for a file, e.g. ``("DAF", "SPK")``.

    SPICE's own ``getfat`` answers this, which is the point: the alternative
    is guessing from the extension, and the committed metakernels are named
    ``.tm`` while a hand-supplied one may be named anything. ``("?", "?")``
    means the file is not a SPICE kernel at all; ``("", "")`` means the
    question could not be asked.
    """
    try:
        import spiceypy
    except ImportError:                                  # pragma: no cover
        return "", ""
    try:
        architecture, kind = spiceypy.getfat(str(path))
    except Exception as exc:                             # noqa: BLE001
        logger.info("[spk-coverage] getfat(%s) failed (%s)", path, exc)
        return "", ""
    return str(architecture), str(kind)


#: A quoted kernel-pool string. SPICE writes an embedded apostrophe as two
#: apostrophes, so a token body is "anything that is not a quote, or a doubled
#: quote". `data/horizons_cache/bodies/269_Justitia_(A887_SA)` has parentheses
#: in it and would survive a cruder pattern; a body with an apostrophe in its
#: designation would not.
#:
#: Written in the unrolled form `[^']*(?:''[^']*)*` rather than the direct
#: `(?:[^']|'')*`. The two accept exactly the same strings, but the direct form
#: backtracks through an unterminated string: measured on 1 MB with a single
#: opening quote, 0.201 s against 0.062 s, growing quadratically. See
#: `_METAKERNEL_MAX_BYTES` for the bound that keeps either from being a hang.
_KERNEL_POOL_STRING = re.compile(r"'([^']*(?:''[^']*)*)'")

#: A metakernel is a list of file names. The committed ones are 446 to 1,309
#: bytes and the generated ones are about 1 KB, so 4 MiB is four thousand times
#: the real thing -- and it bounds both `read_text` and the quadratic term
#: above (1 MB of pathological input parses in 0.201 s, measured 2026-08-27).
#: Anything larger is not a metakernel and is refused rather than parsed.
_METAKERNEL_MAX_BYTES = 4 * 1024 * 1024

#: The start of a kernel-pool assignment -- `NAME = ...` or `NAME += ...`.
#: The operator is captured because the two are not the same: `=` replaces.
_KERNEL_POOL_ASSIGNMENT = re.compile(
    r"^[ \t]*([A-Za-z_][A-Za-z0-9_]*)[ \t]*(\+?=)", re.MULTILINE)

#: SPICE's string-continuation marker. A pool string ending in `+` is joined
#: to the one after it, with the marker dropped. `SPKGenerator` emits every
#: long path this way (`spk_generator.py:_continued_string_lines`) because a
#: single pool string longer than about 99 characters is TRUNCATED rather than
#: refused -- which is PTM-145, and which is why a parser that ignored this
#: would resolve a deep OneDrive path to a shorter one that does not exist.
_POOL_CONTINUATION = "+"


def _join_pool_continuations(tokens):
    """Fold SPICE `+` continuations in a list of pool strings."""
    joined, pending = [], None
    for token in tokens:
        head = pending or ""
        if token.endswith(_POOL_CONTINUATION):
            pending = head + token[:-1]
        else:
            joined.append(head + token)
            pending = None
    if pending is not None:                  # a dangling '+': malformed, but
        joined.append(pending)               # reported as a missing file, not
    return joined                            # silently dropped


def _kernel_pool_assignments(text):
    """``{NAME: [string, ...]}`` for the ``\\begindata`` sections of a KPL file.

    Only the data sections: everything a metakernel says between
    ``\\begintext`` and the next ``\\begindata`` is prose, and the committed
    metakernels' prose mentions the very file names being parsed for.
    """
    data, in_data = [], False
    for line in str(text or "").splitlines():
        marker = line.strip().lower()
        if marker.startswith("\\begindata"):
            in_data = True
            continue
        if marker.startswith("\\begintext"):
            in_data = False
            continue
        if in_data:
            data.append(line)
    body = "\n".join(data)
    out = {}
    starts = list(_KERNEL_POOL_ASSIGNMENT.finditer(body))
    for index, match in enumerate(starts):
        stop = starts[index + 1].start() if index + 1 < len(starts) else len(body)
        chunk = body[match.end():stop]
        tokens = _join_pool_continuations(
            [m.group(1).replace("''", "'")
             for m in _KERNEL_POOL_STRING.finditer(chunk)])
        name = match.group(1).upper()
        # SPICE REPLACES on `=` and only appends on `+=`. Treating both as an
        # append is not a harmless simplification: a hand-edited metakernel
        # that re-declares PATH_VALUES to override a path would end up with
        # ['old', 'new'], and `zip` would then pair the FIRST symbol with the
        # stale value -- every member resolved under the wrong directory, and
        # reported to the user as "not on disk".
        if match.group(2) == "+=" and name in out:
            out[name].extend(tokens)
        else:
            out[name] = tokens
    return out


class MetakernelMembers:
    """What a text metakernel would load, worked out by READING it.

    Parsed, never furnished. ``furnsh`` is global, permanent for the life of
    the process, and would put a planetary ephemeris (de440s is 32 MB) into a
    long-lived GUI's kernel pool every time somebody presses Check coverage --
    and it would also need the working directory moved to the repository root,
    because the committed metakernels' ``PATH_VALUES`` are relative to it. A
    GUI process left ``chdir``-ed somewhere else is a file dialog that opens in
    the wrong place and a relative output path that lands in the wrong tree,
    so the parse route avoids both.

    The four lists are kept apart because each is a DIFFERENT thing to tell
    the user and a different thing to do about it -- and because any of them
    being non-empty is fatal to the run, since ``furnsh`` signals an error on
    the first member it cannot open rather than skipping it
    (``illumination_model.cpp:5473``).
    """

    def __init__(self, kernels=(), missing=(), dataless=(), unresolved=(),
                 error=""):
        self.kernels = list(kernels)          # [(token, resolved Path)]
        self.missing = list(missing)          # [(token, Path)]
        self.dataless = list(dataless)        # [(token, Path)]
        self.unresolved = list(unresolved)    # [token]
        self.error = error
        self._kinds = {}                      # str(path) -> (arch, type)

    def _kind_of(self, path):
        """``getfat``, memoized. Each answer costs a file open, and this is
        asked once per member per query -- three queries on the normal path,
        which on a metakernel naming thousands of kernels would be thousands
        of opens on the GUI thread for one press of Check coverage."""
        key = str(path)
        if key not in self._kinds:
            self._kinds[key] = kernel_architecture(path)
        return self._kinds[key]

    def of_type(self, architecture, kind):
        """The resolved members whose ``getfat`` answer is this pair."""
        return [path for _token, path in self.kernels
                if self._kind_of(path) == (architecture, kind)]

    def described(self):
        """``[(name, "DAF/SPK"), ...]`` for every member that exists."""
        out = []
        for _token, path in self.kernels:
            architecture, kind = self._kind_of(path)
            out.append((path.name, f"{architecture}/{kind}"
                        if architecture else "unreadable"))
        return out


def metakernel_members(mk_path, project_root=None):
    """Resolve a KPL metakernel's ``KERNELS_TO_LOAD`` to paths on this disk.

    ``$SYMBOL`` prefixes are substituted from ``PATH_SYMBOLS``/``PATH_VALUES``
    by LONGEST match, which is what SPICE does and which matters as soon as
    one symbol is a prefix of another. A member that is still relative after
    substitution is resolved against the project root -- the convention the
    committed metakernels state in their own prose ("Paths below are RELATIVE
    to the repository root") -- rather than against the working directory,
    which is what SPICE itself would use and which is exactly the thing a GUI
    cannot rely on.

    Never raises: it is reached from a Qt slot, where an escaping exception is
    a ``qFatal`` and takes the window down with any unsaved edits.

    A member naming a file OUTSIDE the project is resolved and looked at, the
    same as SPICE would -- an absolute path is what every generated metakernel
    uses, and the shared ``data/kernels`` tree is legitimately outside a body's
    directory, so there is no boundary here to enforce. What bounds it instead
    is what is done with the file: ``stat`` for existence, and ``getfat`` for
    the first bytes of its header. Nothing else is opened, no contents are
    read, and only a file that SPICE itself calls ``DAF/SPK`` is ever passed
    to ``spkcov``.
    """
    path = Path(mk_path)
    root = Path(project_root) if project_root else _default_project_root()
    try:
        size = path.stat().st_size
        if size > _METAKERNEL_MAX_BYTES:
            return MetakernelMembers(error=(
                f"{path.name} is {size / (1024 * 1024):.1f} MB. A metakernel "
                f"is a short list of file names -- the ones this project "
                f"writes are about a kilobyte -- so this is not one, and it "
                f"is not being parsed as one"))
        text = path.read_text(errors="replace")
    except OSError as exc:
        return MetakernelMembers(error=f"{path.name} could not be read ({exc})")
    pool = _kernel_pool_assignments(text)
    values = pool.get("PATH_VALUES", [])
    symbols = pool.get("PATH_SYMBOLS", [])

    tokens = pool.get("KERNELS_TO_LOAD", [])
    if not tokens:
        return MetakernelMembers(error=(
            f"{path.name} has no KERNELS_TO_LOAD entry, so it does not name "
            f"any kernel to load"))
    if len(symbols) != len(values):
        # NOT a warning, and not "the surplus is ignored, as SPICE would":
        # SPICE signals SPICE(PATHMISMATCH) -- "Number of path symbols is N;
        # number of path values is M; counts must match" -- and loads NOTHING.
        # A `zip` here would truncate silently and report a green window for a
        # metakernel the run cannot open at all, which is the exact green-label
        # failure this check exists to prevent. Measured with spiceypy.furnsh
        # on a 3-value / 2-symbol fixture, 2026-08-27.
        return MetakernelMembers(error=(
            f"{path.name} declares {len(symbols)} PATH_SYMBOLS against "
            f"{len(values)} PATH_VALUES. SPICE refuses a metakernel whose "
            f"counts do not match (SPICE(PATHMISMATCH)) and loads none of its "
            f"kernels, so this file would give the run no SPICE at all"))
    table = {name: value for name, value in zip(symbols, values)}
    # Longest first: with symbols BODY and BODYSHAPE defined, a token starting
    # '$BODYSHAPE/...' must not be resolved as '$BODY' plus 'SHAPE/...'.
    order = sorted(table, key=len, reverse=True)

    kernels, missing, dataless, unresolved = [], [], [], []
    for token in tokens:
        resolved = token
        if token.startswith("$"):
            for name in order:
                if token[1:].startswith(name):
                    resolved = table[name] + token[1 + len(name):]
                    break
            else:
                unresolved.append(token)
                continue
        if not str(resolved).strip():
            # An empty pool string, or a dangling '+' continuation with
            # nothing after it. `Path("")` is `.`, which joined to the project
            # root is a DIRECTORY THAT EXISTS -- so without this guard a
            # malformed metakernel is reported as a perfectly good one.
            missing.append((token or "''", root / "<empty entry>"))
            continue
        try:
            candidate = Path(resolved).expanduser()
            if not candidate.is_absolute():
                candidate = root / candidate
            present = candidate.is_file()
            placeholder = present and _is_cloud_placeholder(candidate)
        except (OSError, ValueError) as exc:
            # A pool string can hold anything -- an embedded NUL raises
            # ValueError out of `Path`, and a path longer than the file
            # system's limit raises OSError out of `stat`. Neither may escape
            # into a Qt slot.
            logger.warning("[spk-coverage] %s: member %r could not be "
                           "resolved (%s)", path.name, resolved[:80], exc)
            missing.append((token, root / "<unusable path>"))
            continue
        if placeholder:
            dataless.append((token, candidate))
        elif present:
            kernels.append((token, candidate))
        else:
            missing.append((token, candidate))
    return MetakernelMembers(kernels, missing, dataless, unresolved)


def _is_cloud_placeholder(path):
    """True for an evicted OneDrive file. ``stat`` only -- never hydrates.

    ``is_dataless`` asks "nonzero logical size, zero blocks", which a
    DIRECTORY on APFS also answers yes to. Restricting the question to regular
    files keeps a folder from being reported to the user as an evicted file --
    measured on
    ``13294_Rockox_(1998_QO105)/_corrupted_2026-07-03_20260703_174404``, which
    is a directory.
    """
    try:
        from gui_core.preflight import is_dataless
    except ImportError:                                  # pragma: no cover
        return False
    try:
        return bool(path.is_file()) and bool(is_dataless(path))
    except Exception:                                    # noqa: BLE001
        return False


def _ensure_leapseconds(spiceypy, candidates=()):
    """Put a leapseconds kernel in the pool if there is not one. Never raises.

    ``spkcov`` reads the coverage as ephemeris seconds and needs NO kernel at
    all; ``et2utc`` -- which turns those seconds into the ISO strings the run
    window is compared against -- needs an LSK, and raises SPICE(MISSINGTIMEINFO)
    without one.

    That made this whole check inert on a fresh process. Measured 2026-08-27 on
    a Chimaera kernel HORIZONS had just returned: with no LSK in the pool,
    ``spk_coverage`` reported ``available=False`` ("could not be read by
    SPICE"), which ``fetch_spk`` correctly turns into COVERAGE UNKNOWN and
    which correctly declines to set ``ephemeris_source: spice`` -- so a
    perfectly good kernel could never be accepted, and the reason showed up as
    a SPICE error about the kernel rather than about the pool. The only reason
    the failure was not obvious earlier is that the SPK download itself always
    failed first.

    ``candidates`` are tried before the shipped one, so a metakernel that
    names its own leapseconds kernel is preferred over a guess at where
    ``data/kernels`` is. It only has an effect on the FIRST call that finds
    the pool empty: once any LSK is loaded, ``gdpool`` succeeds and this
    returns immediately for the life of the process. That is deliberate --
    re-furnishing per body would grow the pool on every press -- and it is
    harmless, because two leapseconds kernels disagree by seconds where the
    windows being compared are days wide.

    Returns the path furnished, "" if the pool already had time data, or None
    if no leapseconds kernel could be found -- in which case the caller reports
    UNKNOWN coverage, which is the honest answer and not a silent pass.
    """
    try:
        spiceypy.gdpool("DELTET/DELTA_T_A", 0, 1)
        return ""                                    # already usable
    except Exception:                                # noqa: BLE001
        pass
    shipped = (_default_project_root() / "data" / "kernels" / "lsk"
               / "naif0012.tls",)
    for candidate in tuple(candidates) + shipped:
        candidate = Path(candidate)
        if candidate.exists():
            try:
                spiceypy.furnsh(str(candidate))
            except Exception as exc:                 # noqa: BLE001
                logger.warning("[spk-coverage] %s could not be furnished (%s)",
                               candidate, exc)
                continue
            logger.info("[spk-coverage] furnished %s so coverage can be "
                        "reported as UTC", candidate)
            return str(candidate)
    return None


def _merge_intervals(intervals):
    """Sorted, overlap-merged ``[(low, high), ...]``.

    Two SPKs each holding one arc for the same body are ONE window when the
    arcs touch, and two when they do not -- and the difference is the whole
    point, because the engine's fallback engages epoch by epoch and would
    serve a gap silently.
    """
    merged = []
    for low, high in sorted(intervals):
        if merged and low <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], high))
        else:
            merged.append((low, high))
    return merged


def _intersect_intervals(sets):
    """The intervals covered by EVERY one of ``sets``.

    Not ``max(low)..min(high)``: that is the intersection of the ENVELOPES,
    and it reports a hole in one body's coverage as covered because another
    body spans it. Each list is already sorted and merged.
    """
    if not sets:
        return []
    current = list(sets[0])
    for other in sets[1:]:
        common, i, j = [], 0, 0
        while i < len(current) and j < len(other):
            low = max(current[i][0], other[j][0])
            high = min(current[i][1], other[j][1])
            if low < high:
                common.append((low, high))
            if current[i][1] < other[j][1]:
                i += 1
            else:
                j += 1
        current = common
        if not current:
            break
    return current


def _spk_windows(spk_paths, spiceypy):
    """``{body_id: [(low, high), ...]}`` across every SPK, in ET seconds."""
    windows = {}
    for spk in spk_paths:
        ids = spiceypy.spkobj(str(spk))
        for index in range(len(ids)):
            body_id = int(ids[index])
            cell = spiceypy.stypes.SPICEDOUBLE_CELL(2000)
            spiceypy.spkcov(str(spk), body_id, cell)
            for i in range(spiceypy.wncard(cell)):
                low, high = spiceypy.wnfetd(cell, i)
                windows.setdefault(body_id, []).append((low, high))
    return {body_id: _merge_intervals(rows)
            for body_id, rows in windows.items()}


def _normalized_naif_id(naif_id):
    """``(id, error)``. A blank id is not an error; an unparsable one is.

    The trailing ``;`` is stripped, as ``body_config_writer.py:101`` and
    ``body_fetch.py:597`` already do: HORIZONS returns small-body designations
    that way, and ``body_lookup.py:438`` writes the field with a plain
    ``strip()``, so one can reach here.
    """
    raw = str(naif_id if naif_id is not None else "").strip().rstrip("; ")
    if not raw:
        return None, ""
    try:
        return int(raw), ""
    except ValueError:
        return None, (f"generic_body.naif_id = {raw!r} is not an integer NAIF "
                      f"ID, so coverage cannot be reported for one particular "
                      f"body")


def _coverage_from_windows(windows, naif_id, spiceypy, label):
    """Turn ``{body: intervals}`` into a :class:`SpkCoverage`.

    With a NAIF ID, the answer is that body's own window. WITHOUT one, the
    answer is the window common to EVERY body present -- deliberately, and
    deliberately conservative. A metakernel loads a planetary ephemeris
    beside the target's arc, and ``de440s.bsp`` spans 1849 to 2150: taking the
    min-to-max across all bodies, which is what the single-SPK path does and
    what is right for a single-SPK file, would report a Justitia metakernel as
    covering three centuries and pass every run window ever asked about. That
    is the silent-fallback failure this check exists to prevent, wearing a
    green label.
    """
    if not windows:
        return SpkCoverage(False, detail=(
            f"{label} declares bodies but no coverage intervals"))
    if naif_id is not None:
        rows = windows.get(naif_id)
        if not rows:
            present = ", ".join(str(b) for b in sorted(windows)[:12])
            more = "" if len(windows) <= 12 else f", and {len(windows) - 12} more"
            return SpkCoverage(False, detail=(
                f"{label} carries no ephemeris for NAIF ID {naif_id}. The "
                f"bodies it does carry are {present}{more}. Small-body IDs "
                f"are 2000000 plus the asteroid number (Justitia, number 269, "
                f"is 20000269 and not 2000269), so check "
                f"generic_body.naif_id against the list above before "
                f"trusting a SPICE run."), bodies=sorted(windows))
        bodies = [naif_id]
        detail = f"{label}, body {naif_id}"
    else:
        bodies = sorted(windows)
        rows = _intersect_intervals([windows[body] for body in bodies])
        detail = (f"{label}, {len(bodies)} bodies"
                  if len(bodies) > 1 else f"{label}, body {bodies[0]}")
        if len(bodies) > 1:
            detail += " (window common to all)"
        if not rows:
            return SpkCoverage(False, detail=(
                f"{label} has NO window covered by all {len(bodies)} of the "
                f"bodies it carries, so no run window can be shown to be "
                f"inside it. Set generic_body.naif_id so coverage is reported "
                f"for the body this run is about."), bodies=bodies)
    low, high = rows[0][0], rows[-1][1]
    if len(rows) > 1:
        detail += (f" -- WITH GAPS: the coverage is in {len(rows)} pieces, and "
                   f"the span reported below is their outer envelope, so an "
                   f"epoch inside it may still be uncovered")
    return SpkCoverage(True, spiceypy.et2utc(low, "ISOC", 0),
                       spiceypy.et2utc(high, "ISOC", 0),
                       (high - low) / 86400.0,
                       detail=detail, bodies=bodies,
                       intervals=[(spiceypy.et2utc(a, "ISOC", 0),
                                   spiceypy.et2utc(b, "ISOC", 0))
                                  for a, b in rows])


def _first_few(items, limit=3):
    """``("a, b, c", ", and 4 more")`` -- a bounded list and its remainder.

    A metakernel can name any number of kernels, and a message that pasted
    all of them into a one-line note under a text field would be unreadable.
    """
    items = list(items)
    tail = "" if len(items) <= limit else f", and {len(items) - limit} more"
    return ", ".join(items[:limit]), tail


def _lists_kernels_that(path, count):
    """"<name> lists N kernel(s) that is/are", agreeing in number."""
    return (f"{path.name} lists {count} "
            f"kernel{'' if count == 1 else 's'} that "
            f"{'is' if count == 1 else 'are'}")


def _metakernel_coverage(path, naif_id, project_root, spiceypy):
    """Coverage for a KPL metakernel, from the SPKs it would load.

    Handing this file to ``spkobj`` is what used to happen, and SPICE answered
    with ``SPICE(INVALIDARCHTYPE) -- ... has architecture KPL. The file must be
    a binary SPK file to be readable by this routine``, six lines of banner and
    a routine traceback, presented to the user as the reason their kernel was
    no good. The kernel was fine. The reader was wrong about what it had.
    """
    members = metakernel_members(path, project_root)
    if members.error:
        return SpkCoverage(False, detail=(
            f"{members.error}, so there is nothing to report coverage for. "
            f"This is UNKNOWN coverage, not bad coverage."))
    if members.dataless:
        names, more = _first_few(str(p) for _t, p in members.dataless)
        return SpkCoverage(False, detail=(
            f"{_lists_kernels_that(path, len(members.dataless))} a cloud "
            f"placeholder (dataless) rather than a real file on this disk: "
            f"{names}{more}. Reading one here would hang the window while "
            f"OneDrive hydrated it, and a run would stall the same way. Open "
            f"the folder in Finder and choose 'Always Keep on This Device', "
            f"then check again."))
    if members.missing:
        names, more = _first_few(f"{p} (from {t})" for t, p in members.missing)
        return SpkCoverage(False, detail=(
            f"{_lists_kernels_that(path, len(members.missing))} not on disk: "
            f"{names}{more}. furnsh stops at the first member it cannot open, "
            f"so this metakernel would give the run NO SPICE at all -- and "
            f"the engine then serves every epoch from Keplerian elements "
            f"without logging it. Fetch the missing kernel, or fix the path "
            f"in the metakernel."))
    if members.unresolved:
        names, _more = _first_few(members.unresolved)
        return SpkCoverage(False, detail=(
            f"{path.name} refers to {names} but defines no such PATH_SYMBOLS "
            f"entry, so that path cannot be resolved -- by this check or by "
            f"SPICE. Add the symbol, or write the path out in full."))

    spks = members.of_type("DAF", "SPK")
    if not spks:
        described = ", ".join(f"{name} [{kind}]"
                              for name, kind in members.described()) or "none"
        return SpkCoverage(False, detail=(
            f"{path.name} is a SPICE metakernel and it loads no SPK at all. "
            f"Its members are: {described}. There is no ephemeris here, so "
            f"every epoch of the run would come from Keplerian elements, "
            f"silently. Point this field at a metakernel that includes the "
            f"body's .bsp/.spk, or at the kernel itself."))

    normalized, id_error = _normalized_naif_id(naif_id)
    if id_error:
        return SpkCoverage(False, detail=(
            f"{id_error}. {path.name} loads "
            f"{', '.join(p.name for p in spks)}; correct the ID and check "
            f"again."))
    lsk = members.of_type("KPL", "LSK")
    if _ensure_leapseconds(spiceypy, lsk) is None:
        return SpkCoverage(False, detail=_no_leapseconds(path))
    label = f"{path.name} -> {', '.join(p.name for p in spks)}"
    return _coverage_from_windows(_spk_windows(spks, spiceypy),
                                  normalized, spiceypy, label)


#: What each KPL text-kernel type actually holds, for the message a user gets
#: when they point the kernel field at one.
_KPL_ROLE = {
    "LSK": "leap seconds",
    "PCK": "body orientation and shape constants",
    "IK": "instrument parameters",
    "FK": "reference-frame definitions",
    "SCLK": "spacecraft clock coefficients",
}


def _no_leapseconds(path):
    return (f"no leapseconds kernel is loaded and none was found at "
            f"data/kernels/lsk/naif0012.tls, so {path.name}'s coverage "
            f"cannot be converted to UTC and compared with the run "
            f"window. The kernel itself may be perfectly good; this is "
            f"UNKNOWN coverage, not bad coverage.")


def spk_coverage(spk_path, metakernel=None, naif_id=None, project_root=None):
    """Read a kernel's own coverage window -- binary SPK **or** text metakernel.

    Nothing in this repository has ever checked this -- ``grep -rn
    'spkcov|spkobj'`` over ``src/cpp``, ``src/python``, ``gui`` and ``scripts``
    returned nothing before 2026-08-27. The cost is on the record: the
    committed Justitia SPK covered 2034-10-12 to 2034-11-11, every SPICE-mode
    Justitia configuration targets 2035-08-24 onward, and all fifteen ran
    entirely on the silent Keplerian fallback.

    TWO file classes reach this function, because ``generic_body.
    spice_metakernel`` is the field the card writes and the engine reads, and
    what it holds is usually a ``.tm`` -- ``_spk_success`` writes the
    metakernel path there whenever one could be written and the ``.spk`` only
    as a fallback. Both are handled, and which one a file is is decided by
    SPICE's own ``getfat`` rather than by its extension.

    ``naif_id`` is what makes the metakernel answer meaningful: see
    :func:`_coverage_from_windows` for why the union across a metakernel's
    bodies would report three centuries of coverage for every asteroid.

    ``metakernel`` no longer FURNISHES anything. It is now read for a
    leapseconds kernel only -- see :class:`MetakernelMembers` for why a GUI
    process must not accumulate a global kernel pool. No caller in this
    repository passes it.

    Needs a leapseconds kernel to report the window as UTC; see
    :func:`_ensure_leapseconds` for why that was not obvious.
    """
    if not str(spk_path or "").strip():
        return SpkCoverage(False, detail="no kernel selected")
    path, candidates = _resolve_kernel_path(spk_path, project_root)
    if _is_cloud_placeholder(path):
        return SpkCoverage(False, detail=(
            f"{path} is a cloud placeholder (dataless), not a file on this "
            f"disk. Reading it here would hang the window while OneDrive "
            f"hydrated it. Open its folder in Finder and choose 'Always Keep "
            f"on This Device', then check again."))
    if not path.exists():
        if len(candidates) > 1:
            tried = "; ".join(str(c) for c in candidates)
            return SpkCoverage(False, detail=(
                f"{spk_path} is not on disk. It is a bare relative name, so "
                f"it was looked for exactly where the engine looks: {tried}"))
        return SpkCoverage(False, detail=f"{path} is not on disk")
    if path.is_dir():
        return SpkCoverage(False, detail=(
            f"{path} is a directory, not a kernel. Point this field at the "
            f"body's .bsp/.spk, or at the metakernel that loads it."))
    try:
        import spiceypy
    except ImportError as exc:
        return SpkCoverage(False, detail=(
            f"spiceypy is not available here ({exc}), so this kernel's "
            f"coverage cannot be read. Check it yourself before trusting a "
            f"SPICE run: the engine falls back to Keplerian elements per "
            f"epoch and says nothing."))
    try:
        architecture, kind = kernel_architecture(path)
        if (architecture, kind) == ("KPL", "MK"):
            return _metakernel_coverage(path, naif_id, project_root, spiceypy)
        if architecture == "KPL":
            return SpkCoverage(False, detail=(
                f"{path.name} is a SPICE text kernel of type {kind}, not an "
                f"SPK and not a metakernel, so it carries no ephemeris and no "
                f"coverage to report. {kind} kernels hold "
                f"{_KPL_ROLE.get(kind, 'other kernel-pool data')}. Point this "
                f"field at the body's .bsp/.spk, or at a metakernel that "
                f"loads it."))
        if architecture == "DAF" and kind != "SPK":
            return SpkCoverage(False, detail=(
                f"{path.name} is a binary SPICE kernel of type {kind}, not an "
                f"SPK, so it carries no ephemeris and no coverage to report."))
        if architecture == "?":
            return SpkCoverage(False, detail=(
                f"{path.name} is not a SPICE kernel -- SPICE does not "
                f"recognize its architecture. Point this field at the body's "
                f".bsp/.spk, or at a metakernel that loads it."))
        if not architecture:
            # spiceypy is importable but getfat could not answer. Fall through
            # to the pre-2026-08-27 behavior -- try it as a binary SPK and let
            # SPICE say why not -- rather than refusing on our own guess.
            logger.info("[spk-coverage] %s: getfat gave no answer; trying it "
                        "as a binary SPK", path)

        # A binary SPK, read directly. `metakernel` is a leapseconds hint only.
        hint = []
        if metakernel:
            hinted, _tried = _resolve_kernel_path(metakernel, project_root)
            if hinted and hinted.exists():
                hint = metakernel_members(hinted, project_root).of_type(
                    "KPL", "LSK")
        if _ensure_leapseconds(spiceypy, hint) is None:
            return SpkCoverage(False, detail=_no_leapseconds(path))
        ids = spiceypy.spkobj(str(path))
        if not len(ids):
            return SpkCoverage(False, detail=f"{path.name} declares no bodies")
        windows = _spk_windows([path], spiceypy)
        if not windows:
            return SpkCoverage(False, detail=(
                f"{path.name} declares bodies but no coverage intervals"))
        normalized, id_error = _normalized_naif_id(naif_id)
        if id_error:
            return SpkCoverage(False, detail=f"{id_error}. Correct it and "
                                             f"check again.")
        if normalized is not None:
            return _coverage_from_windows(windows, normalized, spiceypy,
                                          path.name)
        # No ID asked for: the historical answer, min-to-max over every body
        # in this ONE file. A HORIZONS small-body SPK holds exactly one body,
        # so for the path `fetch_spk` walks this is the same number it always
        # was -- verified against a direct `spkcov` read on all eight cached
        # kernels by `test_gui_spk_coverage.py`.
        union = _merge_intervals(
            [row for rows in windows.values() for row in rows])
        low, high = union[0][0], union[-1][1]
        return SpkCoverage(True, spiceypy.et2utc(low, "ISOC", 0),
                           spiceypy.et2utc(high, "ISOC", 0),
                           (high - low) / 86400.0,
                           detail=f"{path.name}", bodies=sorted(windows),
                           intervals=[(spiceypy.et2utc(a, "ISOC", 0),
                                       spiceypy.et2utc(b, "ISOC", 0))
                                      for a, b in union])
    except Exception as exc:                              # noqa: BLE001
        return SpkCoverage(False, detail=(
            f"{path.name} could not be read by SPICE ({exc}). Treat this as "
            f"UNKNOWN coverage, not as coverage."))
