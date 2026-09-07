"""Subsurface temperature maps: the rules the GUI needs, with no Qt.

A subsurface temperature map is the same per-facet product as
``statistics/temperature_{avg,max,min}.bin`` -- one number per facet, reduced
over the run's output window -- except that it is sampled at a chosen depth
below the surface instead of at the surface. The engine writes one file per
depth per statistic::

    statistics/temperature_z0p050m_avg.bin      0.050 m, average
    statistics/temperature_z0p050m_max.bin      0.050 m, maximum
    statistics/temperature_z0p050m_min.bin      0.050 m, minimum
    statistics/depth_map_z.json                 which depth each file holds

and it is switched on by listing depths in ``output.map_depths_m``. An empty
list is off, and a run with an empty list writes exactly the files a run
without the key writes -- so the control can ship without changing any
existing configuration.

WHAT THIS MODULE IS FOR. Three questions have to be answered before a run
starts, and none of them can be answered by reading the configuration alone.

1. **Will the engine accept this run?** Several conditions make it stop rather
   than write a map it cannot make correct: a run that does not solve the
   thermal model, a run with statistics switched off, and facet-to-facet
   infrared exchange on the processor solver. All are decidable from the
   configuration, so they are transcribed in :func:`engine_support` and
   answered before launch.

2. **How large is the product?** The arithmetic is the engine's own
   (``src/cpp/thermal/depth_map_plan.h``, ``estimateBytes``): a 4-byte facet
   count plus one 4-byte value per facet, per depth, per statistic, plus the
   merge-friendly side files when those are requested. The engine will not
   write more than 8 GiB in one run, and that ceiling is mirrored here so the
   number arrives beside the control that sets it.

3. **What does an off-node depth cost?** The model solves the temperature on
   its own set of depths. A requested depth between two of them is reached by
   straight-line interpolation through a curved profile, and at a
   lunar-equatorial surface swing that costs about 0.195 K at any one instant
   -- more than the 0.15 K this project treats as its floor, and MORE than the
   error in the maximum or minimum at the same depth, because pointwise errors
   partly cancel at a peak. The two products therefore carry different error
   budgets, about 2.3x apart and in opposite directions from the naive
   expectation, so they are reported separately and never as one figure.
   Using the nearest computed depth avoids both, which is why it is the
   default.

WHAT THIS MODULE DELIBERATELY DOES NOT DO. The set of depths the model solves
on is built by the solver from the body's properties and the depth-grid
settings, and it does not exist until the run's geometry phase. Deciding here
whether a requested depth falls inside the solved column would mean writing a
second copy of the solver's grid builder; a second copy that disagreed with
the first would be worse than no check at all, and reusing one grid's weights
on another is the failure this product's own refusals exist to prevent. That
clause stays in the engine, where there is exactly one grid.
:func:`preflight_findings` says so in as many words, so a depth accepted here
and refused by the run is explained in advance rather than discovered.

Naming, from the engine: ``z``, the depth in meters to three decimals with the
point written as ``p``, then ``m``. Three decimals is one millimeter -- finer
than the top cell of any shipped grid -- and reversible.
"""

import math
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# 1. Keys
# ---------------------------------------------------------------------------

#: The three keys the engine reads, spelled out rather than assembled, because
#: a key built by string surgery is a key no search finds.
DEPTHS_KEY = "output.map_depths_m"
SNAP_KEY = "output.map_depth_snap_to_node"
STATS_KEY = "output.map_depth_stats"

#: Their leaf names, for reading an ``output:`` section directly.
DEPTHS_LEAF = "map_depths_m"
SNAP_LEAF = "map_depth_snap_to_node"
STATS_LEAF = "map_depth_stats"

#: A panel-side key, NOT an engine key, and stripped before a configuration is
#: written. It records which of the two products the user asked for, so the
#: launch check can answer for the one they picked. Keeping it out of the YAML
#: is the point: a key the engine does not read, sitting in a saved file,
#: reads exactly like a setting that is being honored.
PRODUCT_KEY = "output.map_depth_product"
PRODUCT_LEAF = "map_depth_product"

#: The combo's own entries. Stored as the text, the way this application
#: stores every other choice of this shape.
PRODUCT_STATISTICS = "summary over the run"
PRODUCT_TIME_SERIES = "value at every output step"

#: Normalized spelling -> the entry the combo shows, for ``set_enum``.
PRODUCT_ALIASES = {
    "": PRODUCT_STATISTICS,
    "summaryovertherun": PRODUCT_STATISTICS,
    "summary": PRODUCT_STATISTICS,
    "statistics": PRODUCT_STATISTICS,
    "valueateveryoutputstep": PRODUCT_TIME_SERIES,
    "timeseries": PRODUCT_TIME_SERIES,
    "series": PRODUCT_TIME_SERIES,
}

#: The three statistics the engine writes, in the order it lists them. A closed
#: set: they are the three the surface products already carry, so a depth file
#: is the same quantity as its surface counterpart, measured deeper.
STAT_NAMES = ("avg", "max", "min")

STAT_LABELS = {"avg": "Average", "max": "Maximum", "min": "Minimum"}

#: The engine's own ceiling on this product (``depth_map_plan.h``,
#: ``Request::max_bytes``). Mirrored rather than re-chosen: a different number
#: here would either refuse runs the engine accepts or promise runs it refuses.
MAX_PRODUCT_BYTES = 8 << 30

#: Interpolation error at an off-node depth, per kelvin of surface swing.
#: ``INSTANTANEOUS`` is the error at any one moment and bounds a series;
#: ``PEAK`` is the smaller error in the maximum or the minimum. Both measured
#: on the real lunar grid by the engine's own analytic check, and both quoted
#: from ``depth_map_plan.h``: 0.195 K and 0.085 K at a surface swing of 160 K.
#:
#: ⛔ ``PEAK`` is 0.085/160, NOT the header's 4.8e-4. Those are two different
#: quantities and the second is the wrong one for this control: 4.8e-4 is what
#: accumulating the maximum IN the solver costs, which is what the run does
#: when a depth lands on a solved node. The number the panel needs is the error
#: in the maximum at an OFF-node depth, which is the 0.085 K figure. Using
#: 4.8e-4 understated it by about 10%.
INSTANTANEOUS_ERROR_PER_K = 1.217e-3
PEAK_ERROR_PER_K = 0.085 / 160.0

#: A representative lunar-equatorial surface swing, used only to turn those two
#: coefficients into kelvin a reader can weigh against the 0.15 K floor. Named
#: rather than buried, so it is clear which body the number describes.
REFERENCE_SWING_K = 160.0

#: The accuracy this project treats as its floor, quoted in the same sentence
#: as the interpolation error so the comparison is on screen and not implied.
ACCURACY_FLOOR_K = 0.15


def _section(config, name):
    node = (config or {}).get(name)
    return node if isinstance(node, dict) else {}


def _truthy(value, default=False):
    """A YAML scalar read as a boolean, tolerating the spellings a hand-edited
    file uses. The same tolerance the rest of this package applies, so a
    malformed value never aborts a window."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in ("true", "yes", "on", "1")


def _commas(n):
    return "{:,}".format(int(n))


# ---------------------------------------------------------------------------
# 2. The depth list, as text and as numbers
# ---------------------------------------------------------------------------

def parse_depth_list(text):
    """``(depths, error)`` from what the user typed.

    Commas, spaces and newlines all separate, because all three are what
    people type.

    ⛔ ORDER AND REPETITION ARE PRESERVED. A repeated depth is refused, not
    merged, and dropping it here would hide the mistake from the check whose
    whole job is to name it -- the engine refuses it for the same reason,
    because merging two entries silently changes which files a run writes.

    ⛔ AN ENTRY THAT IS NOT A NUMBER IS KEPT, as the text that was typed, and
    `error` names it. Returning an empty list instead would be a silent
    substitution twice over: `[]` is how this feature is switched OFF, so a
    typo in one of four depths would quietly turn the whole product off, and
    the offending text would be gone from everything downstream. Keeping it
    means :func:`validate_request` names it, the configuration check refuses
    it, and the launch check refuses it -- the same way it treats a bad value
    in a hand-edited file.
    """
    if text is None:
        return [], ""
    tokens = str(text).replace(",", " ").split()
    depths = []
    error = ""
    for token in tokens:
        try:
            value = float(token)
        except ValueError:
            depths.append(token)
            error = error or (
                "%r is not a number. List the depths in meters, separated by "
                "commas or spaces -- for example 0, 0.05, 0.25" % token)
            continue
        if not math.isfinite(value):
            depths.append(token)
            error = error or (
                "%r is not a finite number. List the depths in meters below "
                "the surface." % token)
            continue
        depths.append(value)
    return depths, error


def format_depth_list(depths):
    """The text form of a depth list, for putting back into the field.

    ``%g``, so a value comes back the way it was typed and opening a
    configuration and saving it again does not rewrite the line.
    """
    out = []
    for value in depths or []:
        try:
            out.append("%g" % float(value))
        except (TypeError, ValueError):
            out.append(str(value))
    return ", ".join(out)


def normalize_stats(value):
    """A statistics list from a configuration, as engine names.

    ``None`` is the EMPTY list, not the default set. That is not a nicety: the
    engine reaches this key under ``if (output["map_depth_stats"])``, CLEARS
    its default, and then iterates -- so a file carrying a bare
    ``map_depth_stats:`` line ends up with none and the run stops. Measured
    (2026-09-06): "output.map_depth_stats is empty", exit 1. Reading it as the
    default here would show three ticked boxes for a configuration the engine
    refuses.

    An unrecognized NAME is passed through rather than dropped, so
    :func:`validate_request` can call it out. Dropping it would make a typo in
    a hand-edited file look like a deliberate two-statistic run.
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip().lower() for item in value]
    return []          # a scalar: see `sequence_problems`, which names it


#: Distinguishes "the file does not mention this key" from "the file sets it to
#: nothing". The engine treats those differently and so must this module.
_ABSENT = object()


def sequence_problems(output):
    """Refusals for a key the engine reads as a LIST that does not hold one.

    ⛔ MEASURED on the engine, not reasoned about (2026-09-06, a Moon run on
    the shipped depth-map binary), and then FIXED in the engine the same day.
    Both columns are kept because the first one is why this function exists and
    because a run started from an older binary still behaves that way:

    ==============================  ==================  =====================
    written in the file             before the fix      on today's engine
    ==============================  ==================  =====================
    ``map_depths_m: 0.05``          COMPLETES, exit 0,  STOPS at config load,
                                    no depth maps, no   exit 1, naming the
                                    message at all      list form
    ``map_depths_m: "0.05, 0.25"``  the same silence    the same refusal
    ``map_depth_stats: avg``        STOPS, exit 1,      STOPS, exit 1, "must
                                    "map_depth_stats    be a YAML LIST ...
                                    is empty"           a single value (avg)"
    ``map_depth_stats:`` (nothing)  STOPS, exit 1,      unchanged -- a null is
                                    "is empty"          an empty list
    ==============================  ==================  =====================

    The first two were the dangerous pair: a forgotten pair of brackets was a
    complete, successful, subsurface-free run that nobody was told about. The
    engine refuses them now, and this still refuses them HERE, before a launch,
    because a refusal beside the control that caused it costs nothing and a
    refusal after a queued run costs the queue.

    Refusing also has to be the ONLY outcome -- never a coercion. Splitting the
    string ``"0.05, 0.25"`` into depths would accept, on the user's behalf,
    something the run will not do; splitting it into characters (which is what
    ``list()`` on a string gives) would invent ten depths out of one line.
    """
    problems = []
    depths = output.get(DEPTHS_LEAF, _ABSENT)
    if depths is not _ABSENT and depths is not None \
            and not isinstance(depths, (list, tuple)):
        problems.append(
            "output.map_depths_m holds a single value, not a list. The model "
            "reads this key as a list and stops the run when it is not one. "
            "Write it as a list -- for example [0.05] -- or remove it.")
    stats = output.get(STATS_LEAF, _ABSENT)
    if stats is not _ABSENT and stats is not None \
            and not isinstance(stats, (list, tuple)):
        problems.append(
            "output.map_depth_stats holds a single value, not a list. The "
            "model reads this key as a list and stops the run when it is not "
            "one. Write it as a list -- for example [avg, max, min].")
    return problems


def request_from_config(config):
    """``(depths, stats, snap_to_node, problems)`` as the engine would read them.

    One reader, so the panel, the configuration check and the launch check
    cannot end up with three opinions about what a file is asking for -- and so
    a caller never has to reach into this module's private helpers to find out
    whether a YAML scalar meant true.

    ⛔ NEVER RAISES. It is reached from a Qt slot, where an escaping exception
    is not an error message: PyQt6 calls ``qFatal``, the application exits 134,
    and any unsaved edits go with it. A hand-edited ``map_depths_m: 0.05``
    reached ``list(0.05)`` and did exactly that before this returned a fourth
    value instead.
    """
    output = _section(config, "output")
    problems = sequence_problems(output)

    raw_depths = output.get(DEPTHS_LEAF)
    depths = list(raw_depths) if isinstance(raw_depths, (list, tuple)) else []

    raw_stats = output.get(STATS_LEAF, _ABSENT)
    if raw_stats is _ABSENT:
        stats = list(STAT_NAMES)          # the engine's own default
    else:
        stats = normalize_stats(raw_stats)

    snap = _truthy(output.get(SNAP_LEAF), True)
    return depths, stats, snap, problems


def normalize_product(value):
    """The product choice, as one of the two entries the combo offers.

    An unrecognized value comes back UNCHANGED rather than as the default: the
    combo shows a foreign value as its own entry so the user can see it, and
    quietly reading it as "summary" here would contradict what is on screen.
    """
    text = str(value or "")
    if not text.strip():
        return PRODUCT_STATISTICS
    key = "".join(c for c in text.lower() if c not in "_- ")
    return PRODUCT_ALIASES.get(key, text)


# ---------------------------------------------------------------------------
# 3. Naming -- the engine's rule, transcribed
# ---------------------------------------------------------------------------

def depth_tag(z_m):
    """``z0p050m`` for 0.05 m. The engine's ``depthTag``."""
    return "z" + ("%.3f" % float(z_m)).replace(".", "p") + "m"


def field_name(z_m):
    """``temperature_z0p050m`` -- the base name of one depth's files."""
    return "temperature_" + depth_tag(z_m)


def file_names(depths, stats):
    """Every statistics file a request would write, in the engine's order.

    Built from the REQUESTED depths, which is right for a preview and would be
    wrong for a finished run: with the nearest-computed-depth setting on, the
    name carries the depth actually sampled, and that is not known until the
    run resolves its grid. Callers that show this say which of the two it is.
    """
    out = []
    for z in depths or []:
        try:
            base = field_name(z)
        except (TypeError, ValueError):
            # An entry that is not a number has no file name. It is not
            # skipped quietly -- `validate_request` refuses the request that
            # contains it, so nothing built from this list is ever run.
            continue
        for stat in stats or []:
            out.append("%s_%s.bin" % (base, stat))
    return out


# ---------------------------------------------------------------------------
# 4. Size
# ---------------------------------------------------------------------------

def statistics_bytes(n_depths, n_stats, n_faces, with_raw=False):
    """Bytes the statistics product occupies -- the engine's ``estimateBytes``.

    A 4-byte facet count plus one 4-byte value per facet, per depth per
    statistic; plus, when the merge-friendly side files are on
    (``output.write_raw_stats``), an 8-byte running sum and a 4-byte count per
    facet per depth.
    """
    n_depths = max(0, int(n_depths))
    n_stats = max(0, int(n_stats))
    faces = max(0, int(n_faces))
    total = n_depths * n_stats * (4 + 4 * faces)
    if with_raw:
        total += n_depths * ((4 + 8 * faces) + (4 + 4 * faces))
    return total


def time_series_bytes(n_depths, n_frames, n_faces):
    """Bytes a per-output-step product at depth would occupy.

    One file per depth per frame, in the same per-facet format. Reported so the
    size class is on screen: five depths over a month of hourly frames on a
    2.4-million-facet mesh is 64 GiB, which is a disk event rather than a
    product, and is why this model writes the summary and not the series.
    """
    n_depths = max(0, int(n_depths))
    n_frames = max(0, int(n_frames))
    faces = max(0, int(n_faces))
    return n_depths * n_frames * (4 + 4 * faces)


def fmt_bytes(n):
    """A size in the units a person reads it in."""
    n = float(n)
    for unit, scale in (("GiB", 1 << 30), ("MiB", 1 << 20), ("KiB", 1 << 10)):
        if n >= scale:
            return "%.1f %s" % (n / scale, unit)
    return "%d bytes" % int(n)


# ---------------------------------------------------------------------------
# 5. The refusals that do not need the solved depth grid
# ---------------------------------------------------------------------------

def validate_request(depths, stats, snap_to_node=True):
    """Refusal messages for a request, or ``[]``.

    The clauses the engine checks before it starts anything expensive
    (``depth_map_plan.h``, ``validateRequest``), transcribed so the panel and
    the launch check refuse the same requests it does.

    Plus ONE clause from the engine's grid-bound stage that is decidable here
    in exactly one case. A file name carries the depth to three decimals, so
    two depths closer together than a millimeter resolve to the same name and
    one map would overwrite the other; the engine refuses that. It compares the
    depths it RESOLVED, which needs the grid -- except when depths are not
    being moved to the nearest solved depth, in which case the resolved depth
    IS the requested one and the comparison is available here. `snap_to_node`
    is what selects that case, and with it on the clause is left to the engine
    rather than guessed at.

    Not here, and not decidable here: a depth below the deepest solved node,
    and two depths that land on one node once moved to the nearest. Both need
    the solved grid; see the module docstring. The first of those is refused by
    the engine naming ``thermal.thermal_b`` -- the YAML key, which is spelled
    differently from the model's own field ``grid_b``; if that sentence is ever
    mirrored here, mirror the KEY.
    """
    problems = []
    if not depths:
        return problems

    if not stats:
        problems.append(
            "No statistic is selected, so a depth map would write nothing. "
            "Choose at least one of average, maximum or minimum, or clear the "
            "depth list.")
    seen_stats = []
    for name in stats:
        if name not in STAT_NAMES:
            problems.append(
                "%r is not a statistic this model writes. The choices are "
                "average, maximum and minimum -- the same three the surface "
                "products carry." % name)
        elif name in seen_stats:
            problems.append(
                "The %s is listed twice." % STAT_LABELS[name].lower())
        else:
            seen_stats.append(name)

    for value in depths:
        number = None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            number = float(value)
        if number is None or not math.isfinite(number):
            problems.append("%r is not a depth in meters." % (value,))
        elif number < 0.0:
            problems.append(
                "%g m is above the surface. Depth is measured downward from "
                "the surface, and 0 is the surface itself." % number)

    for i, value in enumerate(depths):
        for j in range(i + 1, len(depths)):
            if depths[j] == value:
                problems.append(
                    "%s m is listed twice (entries %d and %d). Remove the "
                    "repeat -- the model does not merge them, because that "
                    "would change which files a run writes without saying so."
                    % (format_depth_list([value]), i + 1, j + 1))
                break

    if not snap_to_node:
        for i, value in enumerate(depths):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            for j in range(i + 1, len(depths)):
                other = depths[j]
                if not isinstance(other, (int, float)) \
                        or isinstance(other, bool) or other == value:
                    continue
                if field_name(value) == field_name(other):
                    problems.append(
                        "%s m and %s m are less than a millimeter apart, so "
                        "both would be written to %s_*.bin and one would "
                        "overwrite the other. Depth file names carry three "
                        "decimals. Drop one, or move them further apart."
                        % (format_depth_list([value]),
                           format_depth_list([other]), field_name(value)))
                    break
    return problems


# ---------------------------------------------------------------------------
# 6. What the engine will accept
# ---------------------------------------------------------------------------

@dataclass
class SupportPolicy:
    """Whether a configuration can carry depth maps, and why not if it cannot.

    ``blockers`` are conditions the engine itself stops on; ``notes`` are true
    statements about a run that will succeed.

    Deliberately its own class rather than the flux-cube module's identically
    shaped one: this module is loaded by the launch check on every run, and
    importing that one to borrow a six-line container would pull its layout
    reader and cache machinery in with it, for a feature that has nothing to do
    with cubes.
    """
    blockers: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    @property
    def allowed(self):
        return not self.blockers

    @property
    def reason(self):
        return " ".join(self.blockers) if self.blockers else " ".join(self.notes)


def engine_support(config):
    """The engine's own preconditions for a depth map, transcribed.

    Each is a hard stop in ``ptm_thermal_sim.cpp`` immediately after the
    configuration is read, so learning about one from the engine is cheap --
    unlike the flux-cube refusals, which fire after the whole solve. They are
    still answered here, because a launch that stops in its first second still
    interrupts the person who started it, and the reason is easier to act on
    beside the control that caused it.

    * **The thermal model is not solved.** With ``thermal.run_thermal_mesh``
      off there is no temperature column to sample at any depth.
    * **Statistics are switched off.** The depth maps ARE statistics and are
      written with them, so with ``thermal.compute_statistics`` off the run
      would compute them and write none.
    * **Facet-to-facet infrared exchange on the processor solver.** The
      exchange loop re-solves selected facets through a path that does not
      update the depth maps, so those facets would report depths from before
      the exchange while their surface values came from after it -- one map
      holding two different calculations in different places. Moving the
      thermal model to the graphics cards (``thermal.use_gpu``) is enough:
      that path does carry it.

    NOT checked, deliberately: the legacy scalar processor solver, which gives
    every facet its own depth grid and which the engine also refuses. It is
    selected by a command-line flag with no configuration key, and this
    application never passes that flag, so a check for it here could not fire
    and would only look like coverage.
    """
    policy = SupportPolicy()
    thermal = _section(config, "thermal")
    radiation = _section(config, "radiation")

    if not _truthy(thermal.get("run_thermal_mesh"), False):
        policy.blockers.append(
            "This run does not solve the thermal model, so there is no "
            "temperature at depth to report. Turn the thermal solution on, or "
            "clear the depth list.")

    if not _truthy(thermal.get("compute_statistics"), True):
        policy.blockers.append(
            "Statistics are switched off for this run. The depth maps are "
            "statistics and are written with them, so the run would compute "
            "them and write nothing.")

    if _truthy(radiation.get("enabled"), False) \
            and not _truthy(thermal.get("use_gpu"), False):
        policy.blockers.append(
            "Facet-to-facet infrared exchange is on and the thermal model is "
            "set to run on the processor. The exchange loop re-solves selected "
            "facets through a path that does not update the depth maps, so "
            "those facets would report their depths from before the exchange "
            "while their surface values came from after it. Run the thermal "
            "model on the graphics cards, turn the exchange off, or clear the "
            "depth list.")

    if policy.allowed:
        policy.notes.append(
            "The depth maps are read out of the temperature columns the run "
            "already solves, so switching them on changes no temperature "
            "anywhere -- it only adds files.")
    return policy


# ---------------------------------------------------------------------------
# 7. The read-outs
# ---------------------------------------------------------------------------

def snap_note(snap_to_node):
    """One sentence about which depth a file will actually hold."""
    if snap_to_node:
        return (
            "Each depth moves to the nearest depth the model solves on, and "
            "the run reports the depth it used: the file name carries that "
            "depth, and statistics/depth_map_z.json lists it beside the depth "
            "you asked for. Nothing is interpolated, so no interpolation "
            "error is spent.")
    return (
        "Each depth is sampled exactly where you ask, by interpolating "
        "between the two model depths on either side. The file name carries "
        "the depth you asked for, and statistics/depth_map_z.json carries the "
        "interpolation error for each one.")


def error_lines(snap_to_node):
    """The two error budgets, kept apart because they are not one number.

    The instantaneous figure is the larger and bounds a series at that depth;
    the figure for the maximum is smaller, because errors either side of a peak
    partly cancel. Quoting either for the other is wrong by about a factor of
    two, in the direction that flatters the product.
    """
    if snap_to_node:
        return ["Interpolation error: none -- every depth lands on a depth the "
                "model solves on."]
    return [
        "Interpolation error at any one instant: about %.3f K at a "
        "lunar-equatorial surface swing of %.0f K, which is over the %.2f K "
        "this project treats as its floor."
        % (INSTANTANEOUS_ERROR_PER_K * REFERENCE_SWING_K, REFERENCE_SWING_K,
           ACCURACY_FLOOR_K),
        "Interpolation error in the maximum or the minimum: about %.3f K at "
        "the same swing -- smaller, because errors either side of a peak "
        "partly cancel. These are two different numbers; neither stands in "
        "for the other."
        % (PEAK_ERROR_PER_K * REFERENCE_SWING_K),
    ]


def size_refusal(config, n_faces):
    """The message when the product is over the engine's ceiling, or "".

    ⛔ A REFUSAL, not a note. The engine stops on this (``buildPlan``'s size
    guard, 8 GiB), so a launch check that merely mentioned it in a size line
    would start a run that exits in its first seconds -- which is exactly what
    the per-step-series refusal beside it exists to prevent. It read as covered
    because the size line DID carry the sentence; nothing put it where
    ``PreflightReport.ok`` could see it.

    Needs the facet count, so it lives with the launch check rather than with
    the configuration check: working the count out means reading a mesh header,
    and the configuration check runs from a Qt slot on every edit.
    """
    depths, stats, _snap, _problems = request_from_config(config)
    if not depths or not n_faces:
        return ""
    kept = [s for s in stats if s in STAT_NAMES]
    with_raw = _truthy(_section(config, "output").get("write_raw_stats"), False)
    total = statistics_bytes(len(depths), len(kept), n_faces, with_raw)
    if total <= MAX_PRODUCT_BYTES:
        return ""
    return (
        "the depth maps would be %s -- %d depth(s) x %d statistic(s) x %s "
        "facets -- and this model will not write more than %s in one run, so "
        "it would stop before integrating anything. Ask for fewer depths or "
        "fewer statistics."
        % (fmt_bytes(total), len(depths), len(kept), _commas(n_faces),
           fmt_bytes(MAX_PRODUCT_BYTES)))


def size_lines(config, n_faces, n_frames=None):
    """``(summary_lines, series_lines)`` for the depth request in `config`.

    Two separate read-outs, because the two products differ in size AND in the
    error they carry, and a single combined figure would be wrong for whichever
    one the reader had in mind.
    """
    output = _section(config, "output")
    depths, all_stats, _snap, _problems = request_from_config(config)
    if not depths:
        return [], []

    stats = [s for s in all_stats if s in STAT_NAMES]
    with_raw = _truthy(output.get("write_raw_stats"), False)
    plural = "" if len(depths) == 1 else "s"

    if not n_faces:
        # No facet count, so neither product can be sized. The SECOND line
        # still says which of the two this model writes -- that fact does not
        # depend on the mesh, and dropping it would leave the reader with a
        # blank where the answer to their question was.
        return (["Product size: not available yet -- the number of facets "
                 "cannot be read from this configuration. Choose the mesh, and "
                 "this line fills in."],
                ["A value at every output step would be one file per depth per "
                 "output step. This model writes the summary only."])

    total = statistics_bytes(len(depths), len(stats), n_faces, with_raw)
    per_file = statistics_bytes(1, 1, n_faces, False)
    summary = [
        "Product size: %s -- %d depth%s x %d statistic%s x %s facets, %s per "
        "file%s."
        % (fmt_bytes(total), len(depths), plural, len(stats),
           "" if len(stats) == 1 else "s", _commas(n_faces),
           fmt_bytes(per_file),
           ", plus the side files that let a run split into chunks be merged"
           if with_raw else "")]
    over = size_refusal(config, n_faces)
    if over:
        summary.append("That is over the %s this model will write in one run, "
                       "so the run would stop before it started. Ask for fewer "
                       "depths or fewer statistics."
                       % fmt_bytes(MAX_PRODUCT_BYTES))

    if n_frames:
        series = [
            "A value at every output step would be %s instead -- %d depth%s x "
            "%s output steps x %s facets. This model writes the summary only."
            % (fmt_bytes(time_series_bytes(len(depths), n_frames, n_faces)),
               len(depths), plural, _commas(n_frames), _commas(n_faces))]
    else:
        series = [
            "A value at every output step would be one file per depth per "
            "output step. This model writes the summary only."]
    return summary, series


# ---------------------------------------------------------------------------
# 8. The launch check
# ---------------------------------------------------------------------------

@dataclass
class PreflightFindings:
    """What a launch check has to say about the depth-map block.

    Returned rather than written into a report, so the whole rule set stays in
    this module beside :func:`engine_support` and can be checked without
    building one.
    """
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    info: list = field(default_factory=list)
    line: str = ""
    series_line: str = ""

    @property
    def ok(self):
        return not self.errors


def preflight_findings(config, n_faces=None, n_frames=None):
    """Everything a launch check can settle about depth maps, before the run.

    `n_faces` and `n_frames` come from the caller: the launch check already
    works both out, and a second copy of "how many facets will this mesh have"
    would be a second answer to that question.
    """
    findings = PreflightFindings()
    depths, stats, snap, shape_problems = request_from_config(config)
    for problem in shape_problems:
        findings.errors.append("depth maps: " + problem)
    if not depths:
        # A malformed key still has to be reported: the run would complete and
        # write nothing, which is the outcome the shape check exists for.
        return findings
    output = _section(config, "output")

    for problem in validate_request(depths, stats, snap):
        findings.errors.append("depth maps: " + problem)
    for blocker in engine_support(config).blockers:
        findings.errors.append("depth maps: " + blocker)
    over = size_refusal(config, n_faces)
    if over:
        findings.errors.append("depth maps: " + over)

    # The product the user picked. Asking for the per-step series is not a
    # mistake and is not quietly downgraded to the summary: it is refused here,
    # with the size that is the reason, so the choice is answered before a run
    # starts instead of after one finishes carrying the wrong files.
    if normalize_product(output.get(PRODUCT_LEAF)) == PRODUCT_TIME_SERIES:
        size = ""
        if n_faces and n_frames:
            size = (" That would be %s here: %d depth%s x %s output steps x "
                    "%s facets." % (
                        fmt_bytes(time_series_bytes(len(depths), n_frames,
                                                    n_faces)),
                        len(depths), "" if len(depths) == 1 else "s",
                        _commas(n_frames), _commas(n_faces)))
        findings.errors.append(
            "depth maps: this run asks for the temperature at depth at every "
            "output step. This model writes the summary over the run -- "
            "average, maximum and minimum -- and not the per-step series."
            + size
            + " Choose the summary, or export a per-step series at a handful "
              "of points with the locations and depths files.")

    summary, series = size_lines(config, n_faces, n_frames)
    findings.line = " ".join(summary)
    findings.series_line = " ".join(series)

    kept = [s for s in stats if s in STAT_NAMES]
    findings.info.append(
        "depth maps: %d depth(s), %s; %d file(s) under statistics/, plus "
        "statistics/depth_map_z.json recording the depth each file holds."
        % (len(depths),
           "each moved to the nearest depth the model solves on" if snap
           else "each interpolated between the model's own depths",
           len(file_names(depths, kept))))
    if not snap:
        findings.warnings.append(
            "depth maps: the depths are sampled between the model's own "
            "depths. At a lunar-equatorial surface swing that interpolation "
            "alone is about %.3f K at any one instant, more than the %.2f K "
            "this project treats as its floor. "
            "statistics/depth_map_z.json carries the figure for each depth."
            % (INSTANTANEOUS_ERROR_PER_K * REFERENCE_SWING_K,
               ACCURACY_FLOOR_K))
    return findings
