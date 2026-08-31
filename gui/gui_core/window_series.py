"""A long run, broken into windows. NO Qt: this is what the card decides.

WHAT PAUL ASKED FOR, in his words: "add an option to break the integration up
into windows, as we've done previously for the Moon (the AGT runs)."

WHAT THE AGT RUNS ACTUALLY DO, because this had to be reused rather than
reinvented. The A3GT campaign does NOT integrate one 18.6-year run. It picks a
handful of SHORT windows at chosen epochs across the long span and runs each one
as a complete, independent simulation with its own spin-up:

* ``scripts/a3gt_v2_subsolar.py`` writes a schedule JSON -- a ``start_time``, a
  ``total_days``, and a list of ``windows``, each with a ``tag``, a
  ``start_iso`` and a ``duration_days``. Its epochs come from SPICE: the day of
  the annual sub-solar-latitude maximum and minimum in each sampled year of the
  18.6-year nodal cycle.
* ``scripts/a3gt_v2_orchestrator.py`` renders ONE YAML per window
  (``render_yaml``: ``start_time``, ``duration_days``, ``timesteps``,
  ``output_dir``) and hands each to
  ``scripts/run_chunked_simulation.py --config <window>.yaml --total-days <d>
  --output-dir <dir> --resume`` (``run_window``, :401).

So the machinery already exists at both ends, and what this module supplies is
the middle: the schedule, the per-window configurations, and the exact commands.
**Nothing here is a new runner.** The commands it emits are the ones the A3GT
orchestrator itself issues.

WHERE THIS DIFFERS FROM A3GT, stated because it is the one thing a user could
otherwise assume wrongly: **the epochs here are EVENLY SPACED, not chosen.**
A3GT's are sub-solar-latitude extrema found with SPICE for the Moon; this card
has no body-specific seasonal solver behind it, so it places the windows at
equal intervals and says so in the schedule it writes
(``epoch_rule: "even"``). An evenly spaced set samples the seasonal cycle; it
does not land on its extremes. A campaign that needs the extremes runs
``a3gt_v2_subsolar.py`` and feeds its schedule to the orchestrator, which is
still the right tool for that and is not replaced here.

WHAT EACH WINDOW IS. A complete run: its own equilibration (inherited from the
base configuration -- ``thermal.equilibration_orbits`` is not touched here) and
its own output. That is what makes the set cheaper than one continuous run and
it is also what it means: **the windows are independent samples of the long
span, not a continuous integration of it.** There is no state handoff between
them, and the module says so where a user will read it
(:func:`confirmation_text`). ``thermal.initial_state_file`` chaining is a
different design -- it is what ``run_chunked_simulation.py`` does WITHIN a
window, and it is why each window's own chunking stays mathematically
equivalent to running that window in one go.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "MAX_WINDOWS",
    "MINUTES_PER_WINDOW_GUESS",
    "SPLIT_ADVICE_DIURNAL_PERIODS",
    "Window",
    "WindowPlan",
    "confirmation_text",
    "plan_windows",
    "runner_commands",
    "schedule_document",
    "window_configs",
    "write_window_series",
]

#: More windows than this is a campaign, and a campaign belongs to the
#: orchestrator (``scripts/a3gt_v2_orchestrator.py``), which has the resume,
#: the per-window stall watchdog and the per-site merge this card does not.
#: A3GT's own production year set is 4 years x 2 seasons = 8 windows.
MAX_WINDOWS = 64

#: Rough single-window wall clock, minutes. The SAME crude number the Europa
#: grid card uses for the same purpose and with the same disclaimer: it sizes a
#: confirmation, it is not a measurement, and the real figure depends on the
#: mesh, the view factors and whether the GPU pool is available.
MINUTES_PER_WINDOW_GUESS = 25.0

#: Above this many diurnal periods in one integration window, the card
#: SUGGESTS splitting. Advice only -- nothing is switched on for the user.
#: 100 diurnal cycles is about eight Earth years of lunar forcing in a single
#: continuous dispatch; ``scripts/run_yearlong_wedge_safe.py`` documents the
#: macOS+AMD command-buffer watchdog that a long single dispatch runs into.
SPLIT_ADVICE_DIURNAL_PERIODS = 100.0

#: Significant figures a duration is written with on the runner command line.
#: Enough for a double's worth of days: see :func:`runner_commands`.
DAYS_DIGITS = 15

_ISO = "%Y-%m-%dT%H:%M:%S"


@dataclass
class Window:
    """One window of a series: when it starts, how long, and what it is called."""

    index: int
    tag: str
    start_iso: str
    duration_days: float
    timesteps: int

    def end_iso(self):
        start = dt.datetime.strptime(self.start_iso, _ISO)
        return (start + dt.timedelta(days=self.duration_days)).strftime(_ISO)


@dataclass
class WindowPlan:
    """A whole series, or a refusal saying why there is none."""

    windows: list = field(default_factory=list)
    start_iso: str = ""
    span_days: float = 0.0
    window_days: float = 0.0
    #: "even" -- see the module docstring. Recorded in the schedule so a reader
    #: never has to guess how the epochs were chosen.
    epoch_rule: str = "even"
    #: Non-empty when no series could be planned. Everything else is then empty.
    refusal: str = ""
    #: Things the user must be told: a span shorter than the windows asked for,
    #: a sampling cadence that hit its ceiling. Never silent.
    notes: list = field(default_factory=list)

    def __len__(self):
        return len(self.windows)


def plan_windows(start_iso, span_days, n_windows, window_days,
                 synodic_days=None, max_windows=MAX_WINDOWS):
    """``WindowPlan`` -- ``n_windows`` windows of ``window_days`` across a span.

    THE PLACEMENT RULE. The first window starts at ``start_iso`` and the last
    one ENDS at the end of the span, with the rest evenly spaced between::

        start_i = start + i * (span_days - window_days) / (n - 1)

    so the series covers the whole span the user asked about rather than
    stopping ``window_days`` short of it. With one window the series is that
    window, at the start.

    REFUSALS, all of them explicit:

    * a start time that is not ISO-8601 -- the engine reads
      ``simulation.start_time`` as one, and inventing a format here would emit
      configurations that fail at launch;
    * fewer than one window, or more than ``max_windows``;
    * a window length that is not positive;
    * ``n * window_days > span_days``, i.e. the windows do not fit without
      overlapping. Overlapping windows would double-count the span, and
      silently shortening them would run something other than what was asked.

    ``synodic_days`` sets the output cadence through
    ``body_cards.samples_for_window``, which keeps the sampling density fixed
    per diurnal cycle. Without it every window gets that function's own
    fallback and the plan says so.
    """
    plan = WindowPlan(span_days=float(span_days or 0.0),
                      window_days=float(window_days or 0.0))
    raw = str(start_iso or "").strip()
    try:
        start = dt.datetime.fromisoformat(raw.replace("Z", ""))
    except (TypeError, ValueError):
        plan.refusal = (
            f"simulation.start_time = {raw!r} is not an ISO-8601 timestamp, so "
            f"the window epochs cannot be computed. The engine reads that key "
            f"as ISO-8601 too, so this would not have run either.")
        return plan
    plan.start_iso = start.strftime(_ISO)

    try:
        count = int(n_windows)
    except (TypeError, ValueError):
        count = 0
    if count < 1:
        plan.refusal = (f"{n_windows} windows is not a series. Ask for at "
                        f"least one.")
        return plan
    if count > max_windows:
        plan.refusal = (
            f"{count} windows is more than this screen writes ({max_windows}). "
            f"A set that size is a campaign, and a campaign belongs to "
            f"scripts/a3gt_v2_orchestrator.py, which carries the resume, the "
            f"per-window stall watchdog and the per-site merge that this "
            f"screen does not.")
        return plan

    window = float(window_days or 0.0)
    if not window > 0:
        plan.refusal = ("A window has to have a positive length. Set the "
                        "integration window before splitting the run.")
        return plan

    span = float(span_days or 0.0)
    if count * window > span + 1e-9:
        plan.refusal = (
            f"{count} windows of {window:g} days need {count * window:g} days "
            f"and the span is {span:g}. They would have to overlap, which "
            f"would count part of the span twice. Lengthen the span, shorten "
            f"the windows, or ask for fewer.")
        return plan

    from gui_core.body_cards import samples_for_window
    stride = ((span - window) / (count - 1)) if count > 1 else 0.0
    synodic = float(synodic_days) if synodic_days else 0.0
    if synodic <= 0:
        plan.notes.append(
            "This body has no known diurnal period here, so each window's "
            "output sample count is the card's default cadence rather than a "
            "fixed number of samples per diurnal cycle.")
    for index in range(count):
        offset = index * stride
        epoch = start + dt.timedelta(days=offset)
        plan.windows.append(Window(
            index=index,
            tag=f"W{index:02d}",
            start_iso=epoch.strftime(_ISO),
            duration_days=window,
            timesteps=samples_for_window(window, synodic) if synodic > 0
            else samples_for_window(window, window),
        ))
    if count > 1:
        plan.notes.append(
            f"The windows are evenly spaced: one every {stride:g} days, the "
            f"first at the start of the span and the last ending at its end. "
            f"They are NOT placed on seasonal extremes -- the A3GT campaign "
            f"chooses its epochs from a SPICE sub-solar-latitude schedule "
            f"(scripts/a3gt_v2_subsolar.py), and that is still the tool for "
            f"that job.")
    return plan


def schedule_document(plan, body="", output_root=""):
    """The schedule JSON, in the shape ``a3gt_v2_subsolar.py`` writes.

    Deliberately the same keys (``start_time``, ``total_days``, ``windows`` of
    ``{tag, start_iso, duration_days}``), so a schedule written here can be
    read by anything that already reads that file -- and so that a reader who
    knows the A3GT one is not learning a second format. ``epoch_rule`` and
    ``generated_by`` are ADDED, not substituted: a file that looked exactly
    like a SPICE-derived schedule while carrying evenly spaced epochs would be
    the quiet substitution this project keeps paying for.
    """
    return {
        "generated_by": "PTM3D GUI, gui_core.window_series",
        "epoch_rule": plan.epoch_rule,
        "epoch_rule_detail": (
            "Evenly spaced across the span. NOT the A3GT rule: "
            "scripts/a3gt_v2_subsolar.py places its windows on the annual "
            "sub-solar-latitude maximum and minimum of each sampled year, "
            "found with SPICE."),
        "body": str(body or ""),
        "start_time": plan.start_iso,
        "total_days": plan.span_days,
        "window_days": plan.window_days,
        "output_root": str(output_root or ""),
        "windows": [
            {"tag": w.tag, "index": w.index, "start_iso": w.start_iso,
             "end_iso": w.end_iso(), "duration_days": w.duration_days,
             "timesteps": w.timesteps}
            for w in plan.windows
        ],
    }


def window_configs(base_config, plan, output_root):
    """``[(subdirectory, config), ...]`` -- one complete configuration each.

    FIVE keys are written per window and nothing else is touched:

    * ``simulation.start_time`` -- the window's epoch;
    * ``simulation.duration_days`` -- the window length. The engine simulates
      ``max(output_days, offset + integ_days)`` after equilibration
      (``ptm_thermal_sim.cpp:15975``), so output window and integration window
      have to agree or the run integrates over a span it did not simulate;
    * ``thermal.integration_days`` -- the same length, for the same reason;
    * ``simulation.timesteps`` -- the window at the card's sampling cadence;
    * ``output.directory`` -- ``<root>/<tag>``.

    Everything else -- the mesh, the view factors, the spin-up length, the
    radiation settings -- is the base configuration's, unchanged. Each window
    therefore runs the SAME physics at a different epoch, which is the only
    thing that makes the set comparable across the span.
    """
    import copy

    out = []
    for window in plan.windows:
        config = copy.deepcopy(base_config or {})
        _set(config, "simulation", "start_time", window.start_iso)
        _set(config, "simulation", "duration_days", float(window.duration_days))
        _set(config, "simulation", "timesteps", int(window.timesteps))
        _set(config, "thermal", "integration_days", float(window.duration_days))
        _set(config, "output", "directory",
             str(Path(output_root) / window.tag))
        out.append((window.tag, config))
    return out


def _set(config, section, key, value):
    node = config.get(section)
    if not isinstance(node, dict):
        node = {}
        config[section] = node
    node[key] = value


def runner_commands(plan, output_root, config_dir=None, python="python3",
                    runner="scripts/run_chunked_simulation.py"):
    """The exact commands to run the series, one per window.

    The SAME invocation ``a3gt_v2_orchestrator.run_window`` builds
    (``scripts/a3gt_v2_orchestrator.py:414-421``): the chunked runner, the
    window's own total days, its own output directory, and ``--resume`` so a
    re-run skips what is already done. Nothing new is being introduced for the
    GUI to drive.

    ``--total-days`` is formatted at :data:`DAYS_DIGITS` significant figures
    and NOT at the orchestrator's ``%g``. ``%g`` is six, and the A3GT windows
    are whole days (30), so six is exact there -- but a diurnal window is
    29.530588 d, which ``%g`` rounds to 29.5306. The chunked runner divides
    ``--total-days`` into chunks while the configuration keeps its own
    ``duration_days``, so the two would disagree by about a second per window.
    Small, and there is no reason to introduce it.
    """
    root = Path(output_root)
    config_dir = Path(config_dir) if config_dir else root
    return [
        [python, runner,
         "--config", str(config_dir / window.tag / "config.yaml"),
         "--total-days", f"{window.duration_days:.{DAYS_DIGITS}g}",
         "--output-dir", str(root / window.tag),
         "--resume"]
        for window in plan.windows
    ]


def wall_clock_hours(plan, minutes_per_window=MINUTES_PER_WINDOW_GUESS):
    """A crude serial wall-clock estimate, in hours. Labeled as one everywhere."""
    return len(plan.windows) * float(minutes_per_window) / 60.0


def confirmation_text(plan, output_root,
                      minutes_per_window=MINUTES_PER_WINDOW_GUESS):
    """What the confirmation dialog says before anything is written.

    Names the run count and the wall-clock estimate, the way the Europa grid
    card's confirmation does -- and states the one thing about a window series
    that is not obvious: the windows do not continue one another.
    """
    hours = wall_clock_hours(plan, minutes_per_window)
    lines = [
        f"This writes {len(plan.windows)} window configurations under",
        f"{output_root}",
        "",
        f"Each window is {plan.window_days:g} days long, and together they "
        f"sample a {plan.span_days:g}-day span starting {plan.start_iso}.",
        "",
        "Each window is a COMPLETE, INDEPENDENT run with its own spin-up. "
        "They are samples of the long span, not a continuous integration of "
        "it: no temperature state is carried from one window to the next.",
        "",
        f"Running them all takes roughly {hours:.0f} hours in series on this "
        f"machine -- a rough estimate, not a measurement.",
        "",
        "They are written, not launched. The exact commands are printed when "
        "the files are ready.",
    ]
    return "\n".join(lines)


def write_window_series(base_config, plan, output_root, project_root=None,
                        save=None):
    """Write ``<root>/<tag>/config.yaml`` per window plus ``windows.json``.

    ``save`` is a callable ``(config, path) -> None``. The caller supplies
    ``ConfigManager``'s own writer, so the files on disk are serialized by the
    same code that writes a run YAML anywhere else -- including the path
    absolutization the engine needs when a configuration moves
    (``ConfigManager.save_run_yaml``). A second serializer here is exactly how
    two files that should be identical stop being identical.

    Returns the list of written config paths.
    """
    import json

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    written = []
    for tag, config in window_configs(base_config, plan, root):
        directory = root / tag
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "config.yaml"
        if save is None:
            import yaml
            with open(path, "w") as handle:
                yaml.dump(config, handle, default_flow_style=False,
                          sort_keys=False)
        else:
            save(config, path)
        written.append(path)
    body = str(((base_config or {}).get("simulation") or {}).get("body", "")
               or "")
    (root / "windows.json").write_text(
        json.dumps(schedule_document(plan, body=body, output_root=root),
                   indent=2) + "\n")
    return written


def split_advice(window_days, synodic_days,
                 threshold=SPLIT_ADVICE_DIURNAL_PERIODS):
    """A sentence advising a split, or "". ADVICE -- nothing is switched on.

    The threshold is in diurnal periods because that is what the length of a
    thermal run means: 100 diurnal cycles on the Moon is eight Earth years of
    forcing in one continuous dispatch, and
    ``scripts/run_yearlong_wedge_safe.py`` documents in detail what a long
    single dispatch runs into on this machine.
    """
    try:
        window = float(window_days)
        synodic = float(synodic_days)
    except (TypeError, ValueError):
        return ""
    if not (window > 0 and synodic > 0):
        return ""
    periods = window / synodic
    if periods <= threshold:
        return ""
    return (f"This window is {periods:,.0f} diurnal cycles long. A run that "
            f"long is usually done as a series of shorter windows across the "
            f"span rather than one continuous integration -- that is what the "
            f"lunar A3GT campaign does, and it is what keeps a single GPU "
            f"dispatch inside the envelope this machine is known to tolerate. "
            f"Tick \"Split into windows\" below to write one.")
