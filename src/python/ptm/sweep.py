"""Parameter sweeps: one base config, a grid, N runs, one collection of Results.

The friction audit counted **19 independent sweep implementations in this
repository sharing zero lines of code**, driving the engine through **9
different config-mutation mechanisms**, for about **1,610 lines** of bespoke
machinery -- and **51% of all shipped YAML configs are near-duplicates
differing in three lines or fewer**, including a latitude x DEM product frozen
into 40 files. This module is the shared thing those 19 sites each re-typed.

    from ptm import Sweep

    sweep = Sweep("examples/quickstart_asteroid_diurnal.yaml",
                  grid={"thermal_inertia": [50, 100, 200, 400]})
    sweep.run()                                  # serial; skips finished points

    ti = sweep.values("thermal_inertia")
    tmax = sweep.collect(lambda r: r.stat("temperature", "max").max())

What it does, and why each piece is here rather than in the caller
------------------------------------------------------------------

**Serial by default.** There is one GPU on this machine, and 18 of the 19
existing sweeps are serial *by choice* for that reason. ``jobs > 1`` is
available, is CPU-only-safe, and refuses to start when the generated configs
ask for the GPU.

**Resumable.** Long sweeps get interrupted. Each point writes
``sweep_point.json`` before it starts and rewrites it when it finishes, so a
re-run skips what is genuinely done and re-runs what is not -- including a
point that was killed mid-flight, which is the case a "does the output
directory exist?" check gets wrong. Every skip and every re-run says why.

**The swept values travel with the result.** Of the ten-odd sites that needed
this, one did it; the consequence is a figure script in this repo that
hard-codes ``{"alb008": 0.08, ...}`` because the run directories do not record
what was swept. Here the parameters are in ``sweep_point.json`` next to the
outputs, and :meth:`Sweep.values` reads them back.

**Two ways to reach the engine, one meaning.** ``thermal_inertia`` and
``albedo`` are engine CLI flags on newer binaries (``--thermal-inertia``,
``--albedo``). On a binary that does not have them the same parameter is
delivered through the config keys the engine already interprets the same way --
``thermophysical.constant_thermal_inertia`` and
``thermophysical.albedo_override``. Which route was taken is detected from
``--help`` (never assumed), announced, and recorded per point. Any other
parameter is a dotted config key and always goes through the config.

No physics is computed here. The two named parameters map onto knobs the
*engine* interprets -- the engine derives ``cp = TI^2/(ks*rhos)`` for a
constant thermal inertia, not this module -- and anything else is a config key
set verbatim.
"""

from __future__ import annotations

import concurrent.futures
import copy
import datetime as _dt
import hashlib
import itertools
import json
import os
import sys
import time
from collections import namedtuple
from pathlib import Path

from . import paths, run as run_mod
from .engine_config import get_dotted, set_dotted

__all__ = [
    "Curve",
    "NAMED_PARAMETERS",
    "Sweep",
    "SweepPoint",
    "get_dotted",
    "load_sweep",
    "set_dotted",
]

#: A model curve from one grid point: ``label`` for a legend, ``params`` for
#: provenance, ``x``/``y`` as arrays.
Curve = namedtuple("Curve", "label params x y")

_MANIFEST = "sweep.json"
_POINT_MARKER = "sweep_point.json"


def _utc_now():
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _md5(path):
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# --- config plumbing ---------------------------------------------------------
# get_dotted / set_dotted live in ptm.engine_config -- they are config plumbing,
# not sweep plumbing, and the next module that needs them should find them
# there. Re-exported above so `from ptm.sweep import set_dotted` also works.


# --- the two named parameters ------------------------------------------------
# Both config keys below were checked against the engine on a GENERIC body as
# well as a built-in one, because `generic_body.surface.*` is the only place a
# generic body's properties are normally read (PTM-67) and a body-class-
# dependent mapping would have been the obvious guess. Measured on
# quickstart_asteroid_diurnal, binary f0f66a09:
#   * thermophysical.albedo_override 0.30 and generic_body.surface.albedo 0.30
#     produce byte-identical temperature statistics (max|diff| = 0.0 K);
#   * thermophysical.constant_thermal_inertia 200 reproduces the engine's own
#     derivation cp = TI^2/(ks*rhos) = 3200 exactly (max|diff| = 8.4e-05 K
#     against a run that sets cp0 = 3200 directly).
# One mechanism therefore covers both body classes, and this module needs no
# body-class branch.

class _NamedParameter:
    """One physical parameter, and the two interchangeable ways to set it.

    ``flag`` is what a newer engine accepts on the command line. ``config_key``
    is the YAML key an older one reads for the *same* quantity with the *same*
    meaning -- not an approximation of it. ``check`` reports config states in
    which the setting would be silently weakened, because a sweep whose axis
    quietly does nothing is the failure this module exists to prevent.
    """

    def __init__(self, name, flag, config_key, units, note, check):
        self.name = name
        self.flag = flag
        self.config_key = config_key
        self.units = units
        self.note = note
        self._check = check

    def overrides(self, value):
        """``{dotted_key: value}`` for the config route."""
        return {self.config_key: float(value)}

    def warnings(self, config):
        return self._check(config)


def _check_thermal_inertia(config):
    out = []
    thermo = config.get("thermophysical") or {}
    if str(thermo.get("conductivity_model", "")).strip() == "martinez2021":
        out.append(
            "thermophysical.conductivity_model is martinez2021, which ignores "
            "ks entirely; the engine calls this combination FATAL "
            "(ptm_thermal_sim.cpp composition guard 1). Every point will fail.")
    out.append(
        "the engine delivers a constant thermal inertia by solving for heat "
        "capacity, cp = TI^2/(ks*rhos), and zeroing chi: k and rho are left "
        "alone, so the thermal diffusivity varies along this axis. Setting the "
        "same TI by scaling ks instead is a different column "
        "(measured 2.9 K in equatorial diurnal amplitude on "
        "quickstart_asteroid_diurnal at TI=200).")
    return out


def _check_albedo(config):
    out = []
    illum = config.get("illumination") or {}
    # illumination.use_albedo_map defaults to TRUE in the engine, so an absent
    # key is the dangerous case, not the safe one.
    uses_map = illum.get("use_albedo_map", True)
    if uses_map or str(illum.get("albedo_map", "")).strip():
        out.append(
            "illumination.use_albedo_map is in force, and a loaded per-face "
            "albedo map out-ranks the scalar albedo: the sweep axis would only "
            "move the equilibration seed and uncovered faces. Set "
            "illumination.use_albedo_map: false to sweep albedo directly.")
    return out


#: The parameters that have an engine flag. Everything else is a dotted config
#: key, applied verbatim.
NAMED_PARAMETERS = {
    "thermal_inertia": _NamedParameter(
        "thermal_inertia", "--thermal-inertia",
        "thermophysical.constant_thermal_inertia",
        "SI (J m^-2 K^-1 s^-1/2)",
        "constant in temperature and depth",
        _check_thermal_inertia),
    "albedo": _NamedParameter(
        "albedo", "--albedo", "thermophysical.albedo_override",
        "", "bolometric Bond albedo, uniform over the body",
        _check_albedo),
}


# --- one point ----------------------------------------------------------------

def _format_value(value):
    """A filesystem-safe token for a parameter value."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        text = f"{value:g}"
    else:
        text = str(value)
    return (text.replace(".", "p").replace("-", "m").replace("+", "")
                .replace("/", "-").replace(" ", ""))


def default_point_name(params):
    """``{'thermal_inertia': 100.0}`` -> ``'thermal_inertia-100'``."""
    return "__".join(f"{key.split('.')[-1]}-{_format_value(value)}"
                     for key, value in params.items())


class SweepPoint:
    """One grid point: its parameters, its run directory, its Results."""

    def __init__(self, sweep, index, coords, params):
        self.sweep = sweep
        self.index = int(index)
        self.coords = tuple(coords)
        self.params = dict(params)
        self.name = sweep._name_for(self.params)
        self.dir = sweep.outdir / self.name
        self._results = None
        self._config_text = None
        self._config_sha = None

    def __repr__(self):
        return (f"<SweepPoint {self.index} {self.name} "
                f"status={self.status()}>")

    # -- files ---------------------------------------------------------------

    @property
    def marker_path(self):
        return self.dir / _POINT_MARKER

    @property
    def config_path(self):
        return self.dir / "config.yaml"

    @property
    def log_path(self):
        return self.dir / "engine.log"

    def marker(self):
        """The recorded ``sweep_point.json``, or ``{}``."""
        try:
            return json.loads(self.marker_path.read_text())
        except (OSError, ValueError):
            return {}

    def status(self):
        return self.marker().get("status", "pending")

    # -- the config this point runs -------------------------------------------

    def config(self):
        """The full config dict for this point (base + overrides)."""
        return self.sweep._config_for(self.params, self.dir)

    def config_text(self):
        """The YAML this point runs, serialized once and kept.

        :meth:`completeness` hashes this, and it is called for every point by
        ``run()``, ``collect()``, ``table()`` and ``curves()`` alike -- four
        deep copies and four YAML dumps per point per figure without the
        cache. A Sweep's base config and overrides do not change over its
        lifetime, so one serialization per point is all there is.
        """
        if self._config_text is None:
            import yaml
            self._config_text = yaml.safe_dump(
                self.config(), sort_keys=False, default_flow_style=False)
        return self._config_text

    def config_sha256(self):
        if self._config_sha is None:
            self._config_sha = hashlib.sha256(
                self.config_text().encode("utf-8")).hexdigest()
        return self._config_sha

    def command(self):
        """The engine argv for this point."""
        cmd = [str(self.sweep.exe), "--config", str(self.config_path)]
        cmd.extend(self.sweep._flag_args(self.params))
        cmd.extend(str(a) for a in self.sweep.engine_args)
        return cmd

    # -- completeness ----------------------------------------------------------

    def outputs_present(self):
        """Whether the run directory holds a run the readers can open."""
        if not (self.dir / "metadata.txt").is_file():
            return False, "metadata.txt is missing"
        stats = self.dir / "statistics"
        if not stats.is_dir() or not any(stats.glob("*.bin")):
            return False, "statistics/ holds no .bin files"
        return True, ""

    def completeness(self):
        """``(is_complete, reason)`` -- the resume decision, with its reason."""
        marker = self.marker()
        if not marker:
            if self.dir.is_dir() and any(self.dir.iterdir()):
                return False, ("no sweep_point.json but the directory is not "
                               "empty; it will be overwritten")
            return False, "not run yet"
        status = marker.get("status")
        if status != "complete":
            return False, f"last attempt ended {status!r}"
        recorded = marker.get("config_sha256")
        current = self.config_sha256()
        if recorded != current:
            return False, ("the config changed since this point ran "
                           f"({str(recorded)[:12]} -> {current[:12]})")
        ok, why = self.outputs_present()
        if not ok:
            return False, f"marked complete but {why}"
        return True, "complete"

    def is_complete(self):
        return self.completeness()[0]

    # -- results ----------------------------------------------------------------

    @property
    def results(self):
        """A :class:`ptm.results.Results` for this point's output directory."""
        if self._results is None:
            ok, why = self.completeness()
            if not ok:
                raise RuntimeError(
                    f"point {self.name} has no usable results: {why}")
            from .results import Results
            self._results = Results(self.dir, **self.sweep.results_kwargs)
        return self._results

    # -- execution ---------------------------------------------------------------

    def _write_marker(self, **fields):
        record = self.marker()
        record.update(fields)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.marker_path.write_text(json.dumps(record, indent=2, default=str))

    def execute(self, echo=True):
        """Write the config, run the engine, record the outcome. Returns status."""
        self.dir.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(self.config_text())
        command = self.command()

        self._write_marker(
            index=self.index, coords=list(self.coords), name=self.name,
            params=self.params,
            config_overrides=self.sweep._overrides_for(self.params, self.dir),
            engine_flags=self.sweep._flag_args(self.params),
            parameter_route=self.sweep.route,
            command=command,
            engine=str(self.sweep.exe), engine_md5=self.sweep.exe_md5,
            base_config=str(self.sweep.base_config_path),
            base_config_sha256=self.sweep.base_sha256,
            config_sha256=self.config_sha256(),
            working_dir=str(self.sweep.workdir) if self.sweep.workdir else None,
            status="running", started_utc=_utc_now(),
            finished_utc=None, exit_code=None, wall_seconds=None, reason=None)

        started = time.monotonic()
        with open(self.log_path, "w") as log:
            # No lock: EngineProcess pumps both pipes from the calling thread,
            # so every callback for this point arrives on one thread, and each
            # point writes its own log file even under jobs > 1.
            def write(prefix, text):
                log.write(f"{prefix}{text}\n")
                log.flush()

            process = run_mod.EngineProcess(
                command, working_dir=self.sweep.workdir, env=self.sweep.env,
                on_output=lambda t: write("", t),
                on_warning=lambda t: write("WARNING ", t),
                on_error=lambda t: write("ERROR   ", t),
                on_heartbeat=lambda m, s: write("[heartbeat] ", m),
                timeout=self.sweep.timeout)
            exit_code = process.run()
        wall = time.monotonic() - started

        if process.timed_out:
            status, reason = "failed", (
                f"stopped at the {self.sweep.timeout:g} s timeout")
        elif exit_code != 0:
            status, reason = "failed", f"engine exited {exit_code}"
        else:
            ok, why = self.outputs_present()
            status = "complete" if ok else "failed"
            reason = "" if ok else f"engine exited 0 but {why}"

        self._write_marker(status=status, exit_code=exit_code,
                           wall_seconds=round(wall, 3),
                           finished_utc=_utc_now(), reason=reason or None)
        if echo:
            mark = "ok  " if status == "complete" else "FAIL"
            extra = f" -- {reason}" if reason else ""
            print(f"[sweep] {mark} {self.name} ({wall:.1f} s){extra}",
                  file=sys.stderr)
            if status == "failed":
                print(f"[sweep]      log: {self.log_path}", file=sys.stderr)
        return status


# --- the sweep -----------------------------------------------------------------

class Sweep:
    """A base config, a parameter grid, and the runs they define.

    Parameters
    ----------
    base_config
        Path to the YAML the sweep varies. It is never modified.
    grid
        ``{name: [values...]}``. Iterated as a cartesian product in the order
        given, first axis slowest. A name in :data:`NAMED_PARAMETERS` reaches
        the engine by flag or by its config key; any other name is treated as a
        dotted config key (``'thermal.equilibration_orbits'``).
    outdir
        Where the points go, one directory each. Defaults to the base config's
        ``output.directory`` with ``_sweep`` appended.
    overrides
        ``{dotted_key: value}`` applied to every point, before the grid. The
        knobs that are the same in all N runs but different from the base
        config -- an output cadence, a locations file, radiation off.
    derived
        ``{dotted_key: callable(params) -> value}``, applied after the grid.
        This is where a caller puts a relation between parameters -- four of
        the existing sweeps need one -- and it stays the caller's relation.
    label_axes
        Axis names that are *labels*, not config keys: they name the point and
        feed ``derived``, and are never written into the YAML. Declaring them
        is required, because the engine accepts unknown keys in silence and a
        stray ``dem: agenor`` at the top of a generated config would sit there
        being read by nothing.
    exe, workdir, env, engine_args
        The engine to run, where to run it (default: the repository root, which
        is where the shipped metakernels resolve), its environment, and extra
        argv appended to every point.
    use_flags
        ``'auto'`` (default) detects ``--thermal-inertia`` / ``--albedo`` from
        ``exe --help``; ``True`` requires them; ``False`` always writes config
        keys instead.
    timeout
        Wall-clock ceiling per point, in seconds.
    results_kwargs
        Forwarded to :class:`ptm.results.Results` (``units_style``,
        ``mesh_units``, ...).
    """

    def __init__(self, base_config, grid, outdir=None, overrides=None,
                 derived=None, label_axes=(), exe=None, workdir=None, env=None,
                 engine_args=(), use_flags="auto", timeout=None,
                 name_from=None, results_kwargs=None, quiet=False):
        import yaml

        self.base_config_path = Path(base_config).expanduser().resolve()
        if not self.base_config_path.is_file():
            raise FileNotFoundError(f"base config not found: {base_config}")
        raw = self.base_config_path.read_bytes()
        self.base_sha256 = hashlib.sha256(raw).hexdigest()
        self.base_config = yaml.safe_load(raw.decode("utf-8")) or {}

        if not grid:
            raise ValueError("grid is empty: a sweep needs at least one axis")
        self.axes = {}
        for name, values in grid.items():
            values = list(values)
            if not values:
                raise ValueError(f"grid axis {name!r} has no values")
            self.axes[str(name)] = values
        self.overrides = dict(overrides or {})
        clash = sorted(set(self.overrides) & set(self.axes))
        if clash:
            raise ValueError(
                f"{clash} appear in both overrides= and grid=; a key cannot be "
                f"both fixed and swept")
        self.derived = dict(derived or {})
        self.label_axes = frozenset(label_axes)
        for name in self.axes:
            if (name in NAMED_PARAMETERS or "." in name
                    or name in self.label_axes):
                continue
            raise ValueError(
                f"axis {name!r} is neither a named parameter "
                f"({', '.join(sorted(NAMED_PARAMETERS))}), nor a dotted config "
                f"key such as 'thermal.equilibration_orbits'. If it is a label "
                f"that only feeds derived=, pass label_axes=({name!r},); "
                f"writing it into the YAML would add a key the engine reads "
                f"and ignores without complaint.")
        self._name_from = name_from
        self.results_kwargs = dict(results_kwargs or {})
        self.timeout = None if timeout is None else float(timeout)
        self.env = env
        self.engine_args = tuple(engine_args)
        self.quiet = bool(quiet)

        self.root = paths.find_repo_root(self.base_config_path.parent) \
            or paths.find_repo_root()
        self.workdir = Path(workdir).expanduser().resolve() if workdir else (
            self.root if self.root else None)

        resolved = (Path(exe).expanduser().resolve() if exe
                    else paths.find_engine_executable(
                        search_roots=[self.root] if self.root else []))
        if resolved is None or not Path(resolved).exists():
            raise FileNotFoundError(
                "ptm_thermal_sim not found. Build it, set PTM_BUILD_DIR, or "
                "pass exe=. See INSTALL.md.")
        self.exe = Path(resolved)
        self.exe_md5 = _md5(self.exe)

        base_out = get_dotted(self.base_config, "output.directory") or "output"
        self.outdir = Path(outdir).expanduser().resolve() if outdir else (
            (self.workdir or Path.cwd()) / f"{base_out}_sweep").resolve()

        self._said_vf = False
        self.route = self._decide_route(use_flags)
        self._report_axes()
        self.points = self._build_points()
        self._check_names_unique()

    def __repr__(self):
        shape = "x".join(str(n) for n in self.shape)
        return (f"<Sweep {self.base_config_path.name} {shape} = "
                f"{len(self.points)} points -> {self.outdir}>")

    def __len__(self):
        return len(self.points)

    def __iter__(self):
        return iter(self.points)

    def __getitem__(self, index):
        return self.points[index]

    @property
    def shape(self):
        """The grid shape, one entry per axis."""
        return tuple(len(v) for v in self.axes.values())

    def axis(self, name):
        """The values of one axis, in grid order."""
        return tuple(self.axes[name])

    # -- how each parameter reaches the engine ---------------------------------

    def _decide_route(self, use_flags):
        """``{parameter: 'flag'|'config'}``, decided once and announced."""
        route = {}
        available = None
        for name in self.axes:
            param = NAMED_PARAMETERS.get(name)
            if param is None:
                route[name] = "config"
                continue
            if use_flags is False:
                route[name] = "config"
                continue
            if available is None:
                available = run_mod.engine_flags(self.exe)
            if param.flag in available:
                route[name] = "flag"
            elif use_flags is True:
                raise RuntimeError(
                    f"use_flags=True but {self.exe} does not advertise "
                    f"{param.flag}. Build a newer engine, or pass "
                    f"use_flags='auto' to fall back to "
                    f"{param.config_key}.")
            else:
                route[name] = "config"
        return route

    def _report_axes(self):
        """State the route and every way an axis could silently do nothing."""
        if self.quiet:
            return

        def say(text):
            print(f"[sweep] {text}", file=sys.stderr)

        say(f"engine {self.exe} (md5 {self.exe_md5})")
        for name, values in self.axes.items():
            param = NAMED_PARAMETERS.get(name)
            if param is None:
                say(f"axis {name}: {len(values)} values -> config key {name}")
                continue
            if self.route[name] == "flag":
                say(f"axis {name}: {len(values)} values -> {param.flag} "
                    f"(engine advertises it)")
            else:
                say(f"axis {name}: {len(values)} values -> config key "
                    f"{param.config_key} ({param.flag} not advertised by this "
                    f"engine)")
            for warning in param.warnings(self.base_config):
                say(f"  note ({name}): {warning}")
        for key in self.derived:
            say(f"derived: {key} (from the caller's own relation)")

    def _flag_args(self, params):
        """Engine argv contributed by the parameters routed to flags."""
        out = []
        for name, value in params.items():
            if self.route.get(name) == "flag":
                out.extend([NAMED_PARAMETERS[name].flag, f"{float(value):.17g}"])
        return out

    def _overrides_for(self, params, point_dir):
        """The dotted config keys this point sets, and to what."""
        overrides = dict(self.overrides)
        for name, value in params.items():
            if self.route.get(name) == "flag" or name in self.label_axes:
                continue
            param = NAMED_PARAMETERS.get(name)
            if param is None:
                overrides[name] = value
            else:
                overrides.update(param.overrides(value))
        for key, function in self.derived.items():
            overrides[key] = function(dict(params))
        overrides["output.directory"] = str(point_dir)
        return overrides

    _VF_KEY = "radiation.view_factor_file"

    def _config_for(self, params, point_dir):
        config = copy.deepcopy(self.base_config)
        overrides = self._overrides_for(params, point_dir)
        for key, value in overrides.items():
            set_dotted(config, key, value)
        self._anchor_view_factor(config, self._VF_KEY in overrides)
        return config

    def _anchor_view_factor(self, config, from_caller):
        """Make a relative ``radiation.view_factor_file`` absolute.

        This is the one key the engine resolves against the **YAML file's own
        directory** rather than the working directory
        (``config_parser.cpp:507-517``), and the generated config sits beside
        its outputs rather than beside the base config -- so a relative value
        left alone would silently name a different file, and the engine would
        rebuild a view-factor matrix instead of loading the cached one.

        Which directory it is anchored to depends on where the value came
        from, because the two mean different things:

        * from the **base config**: the base config's own directory, which is
          what the engine would have resolved it against had that file been
          run directly. The generated config then means exactly what the base
          config meant, right or wrong.
        * from **overrides= or derived=**: the working directory, because a
          caller writing ``data/x.vf`` in Python means it relative to where
          they run, not relative to a YAML they never see.

        Either way it is announced once, since a rewritten path is a
        substitution.
        """
        vf = get_dotted(config, self._VF_KEY)
        if not vf or Path(vf).is_absolute():
            return
        anchor = (self.workdir or Path.cwd()) if from_caller \
            else self.base_config_path.parent
        # os.path.normpath, not Path.resolve: the engine composes
        # `absolute(yaml).parent_path() / vf` and calls lexically_normal(),
        # which does not follow symlinks. Resolving them here would hand the
        # engine a different path than the one it would have built, and this
        # tree does use symlinked view-factor files.
        absolute = os.path.normpath(os.path.join(str(anchor), vf))
        set_dotted(config, self._VF_KEY, absolute)
        if not self.quiet and not self._said_vf:
            self._said_vf = True
            origin = ("overrides=/derived=, anchored to the working directory"
                      if from_caller else
                      "the base config, anchored to its own directory (which "
                      "is what the engine would resolve it against)")
            print(f"[sweep] {self._VF_KEY} {vf!r} came from {origin}: "
                  f"{absolute}", file=sys.stderr)

    # -- points -----------------------------------------------------------------

    def _name_for(self, params):
        if self._name_from is not None:
            return str(self._name_from(dict(params)))
        return default_point_name(params)

    def _build_points(self):
        names = list(self.axes)
        points = []
        for index, coords in enumerate(
                itertools.product(*(range(len(self.axes[n])) for n in names))):
            params = {n: self.axes[n][c] for n, c in zip(names, coords)}
            points.append(SweepPoint(self, index, coords, params))
        return points

    def _check_names_unique(self):
        seen = {}
        for point in self.points:
            if point.name in seen:
                raise ValueError(
                    f"two grid points get the same directory name "
                    f"{point.name!r}: {seen[point.name]} and {point.params}. "
                    f"Pass name_from= to disambiguate.")
            seen[point.name] = point.params

    # -- running -----------------------------------------------------------------

    def _guard_parallel(self, jobs):
        if jobs <= 1:
            return
        gpu_points = [p.name for p in self.points
                      if get_dotted(p.config(), "thermal.use_gpu")
                      or get_dotted(p.config(), "radiation.use_gpu")]
        if gpu_points:
            raise RuntimeError(
                f"jobs={jobs} with {len(gpu_points)} point(s) configured for "
                f"the GPU (first: {gpu_points[0]}). There is one GPU; "
                f"concurrent engine processes on it wedge the device. Run "
                f"serially, or set thermal.use_gpu/radiation.use_gpu false.")
        print(f"[sweep] jobs={jobs}: {jobs} engine processes at once. This is "
              f"CPU-only-safe and nothing more -- each point still uses "
              f"thermal.threads threads, so oversubscription is the caller's "
              f"to manage.", file=sys.stderr)

    def run(self, resume=True, jobs=1, dry_run=False, stop_on_failure=False):
        """Run every point that is not already complete. Returns a summary dict.

        ``resume=True`` (the default) skips points whose ``sweep_point.json``
        says complete, whose config still hashes the same, and whose outputs
        are on disk; anything else is re-run, and the reason is printed.
        """
        jobs = max(1, int(jobs))
        self.outdir.mkdir(parents=True, exist_ok=True)
        self._write_manifest()

        todo, skipped = [], []
        for point in self.points:
            complete, why = point.completeness()
            if resume and complete:
                skipped.append(point)
                if not self.quiet:
                    print(f"[sweep] skip {point.name} (complete)",
                          file=sys.stderr)
            else:
                todo.append(point)
                if not self.quiet and why not in ("not run yet", "complete"):
                    print(f"[sweep] rerun {point.name}: {why}", file=sys.stderr)

        if dry_run:
            for point in todo:
                print(" ".join(point.command()))
            return {"planned": len(todo), "skipped": len(skipped),
                    "complete": 0, "failed": 0, "dry_run": True}

        if todo:
            self._guard_parallel(jobs)

        started = time.monotonic()
        statuses = {}
        if jobs == 1:
            for point in todo:
                statuses[point.name] = point.execute(echo=not self.quiet)
                if stop_on_failure and statuses[point.name] != "complete":
                    print(f"[sweep] stopping: {point.name} failed",
                          file=sys.stderr)
                    break
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
                futures = {pool.submit(p.execute, not self.quiet): p
                           for p in todo}
                for future in concurrent.futures.as_completed(futures):
                    statuses[futures[future].name] = future.result()

        summary = {
            "planned": len(todo),
            "skipped": len(skipped),
            "complete": sum(1 for s in statuses.values() if s == "complete"),
            "failed": sum(1 for s in statuses.values() if s != "complete"),
            "not_attempted": len(todo) - len(statuses),
            "wall_seconds": round(time.monotonic() - started, 1),
            "dry_run": False,
        }
        self._write_manifest()
        if not self.quiet:
            print(f"[sweep] {summary['complete']} complete, "
                  f"{summary['failed']} failed, {summary['skipped']} skipped, "
                  f"{summary['wall_seconds']} s", file=sys.stderr)
        return summary

    # -- manifest -------------------------------------------------------------

    def _write_manifest(self):
        record = {
            "created_utc": _utc_now(),
            "base_config": str(self.base_config_path),
            "base_config_sha256": self.base_sha256,
            "outdir": str(self.outdir),
            "axes": {k: list(v) for k, v in self.axes.items()},
            "shape": list(self.shape),
            "overrides": dict(self.overrides),
            "label_axes": sorted(self.label_axes),
            "derived_keys": sorted(self.derived),
            "route": dict(self.route),
            "engine": str(self.exe),
            "engine_md5": self.exe_md5,
            "engine_args": list(self.engine_args),
            "working_dir": str(self.workdir) if self.workdir else None,
            "timeout_seconds": self.timeout,
            "points": [{"index": p.index, "name": p.name, "coords": list(p.coords),
                        "params": p.params, "dir": str(p.dir),
                        "status": p.status()} for p in self.points],
        }
        self.outdir.mkdir(parents=True, exist_ok=True)
        (self.outdir / _MANIFEST).write_text(
            json.dumps(record, indent=2, default=str))

    # -- reading the collection --------------------------------------------------

    def completed(self):
        """The points with usable results, in grid order."""
        return [p for p in self.points if p.is_complete()]

    def results(self):
        """A :class:`ptm.results.Results` per completed point, in grid order."""
        return [p.results for p in self.completed()]

    def values(self, name, points=None):
        """One parameter's value at each point: an array, in grid order.

        This is the accessor whose absence made a figure script in this repo
        hard-code ``{"alb008": 0.08, ...}`` -- the run directories did not
        record what was swept, so the axis was re-typed by hand at the figure
        stage.
        """
        import numpy as np
        points = self.points if points is None else points
        if name not in self.axes and name not in self.derived:
            raise KeyError(f"{name!r} is not an axis of this sweep; "
                           f"have {sorted(self.axes)}")
        return np.array([p.params[name] for p in points])

    def collect(self, function, missing="raise"):
        """``function(Results)`` at every point: an array, in grid order.

        ``missing`` decides what an incomplete point contributes:
        ``'raise'`` (default), ``'skip'``, or ``'nan'``.
        """
        import numpy as np
        if missing not in ("raise", "skip", "nan"):
            raise ValueError("missing must be 'raise', 'skip' or 'nan'")
        out = []
        for point in self.points:
            complete, why = point.completeness()
            if complete:
                out.append(function(point.results))
            elif missing == "raise":
                raise RuntimeError(
                    f"point {point.name} is not complete ({why}); pass "
                    f"missing='skip' or missing='nan' to tolerate it")
            elif missing == "nan":
                out.append(np.nan)
        return np.array(out)

    def stat(self, field="temperature", moment="max", missing="raise"):
        """One per-face statistic across the sweep: ``(n_points, n_faces)``.

        "Give me field X across the sweep axis" -- the shape every downstream
        plot and lookup-table fit starts from.
        """
        return self.collect(lambda r: r.stat(field, moment), missing=missing)

    def grid_of(self, function, missing="nan"):
        """:meth:`collect`, reshaped to :attr:`shape`."""
        return self.collect(function, missing=missing).reshape(self.shape)

    def table(self, missing="skip", **columns):
        """Tidy rows: one dict per point, parameters plus derived columns.

        ``sweep.table(tmax=lambda r: r.stat('temperature', 'max').max())``
        gives ``[{'thermal_inertia': 50.0, 'status': 'complete', 'tmax': ...},
        ...]`` -- a list of dicts, so nothing here needs pandas, and
        ``pandas.DataFrame(sweep.table(...))`` works if the caller has it.
        """
        rows = []
        for point in self.points:
            complete, why = point.completeness()
            if not complete and missing == "skip":
                continue
            row = {"index": point.index, "name": point.name}
            row.update(point.params)
            row["status"] = point.status()
            row["run_dir"] = str(point.dir)
            marker = point.marker()
            if marker.get("wall_seconds") is not None:
                row["wall_seconds"] = marker["wall_seconds"]
            for label, function in columns.items():
                if complete:
                    row[label] = function(point.results)
                elif missing == "nan":
                    row[label] = float("nan")
                else:
                    raise RuntimeError(
                        f"point {point.name} is not complete ({why})")
            rows.append(row)
        return rows

    def write_table(self, path, missing="skip", **columns):
        """:meth:`table` as CSV. Returns the rows it wrote."""
        import csv
        rows = self.table(missing=missing, **columns)
        if not rows:
            raise RuntimeError("no completed points to tabulate")
        fields = list(rows[0])
        with open(path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return rows

    def curves(self, of=None, field="temperature", lat=None, lon=None,
               label=None, missing="skip", **kwargs):
        """One :class:`Curve` per completed point.

        ``of`` is a callable ``Results -> (x, y)``. Without one, the curve is
        the noon-referenced local-solar-time curve of ``field`` when no face is
        named, or that face's time series when ``lat``/``lon`` are given.
        """
        if of is None:
            if lat is None and lon is None:
                said = []

                def of(results):
                    # The local-solar-time reduction folds whatever steps it is
                    # given onto ONE 24-hour axis, so handing it a two-rotation
                    # window draws two cycles. Take the last rotation, and say
                    # so once -- picking a subset of the data silently is not
                    # an option.
                    steps = kwargs.pop("steps", None) if "steps" in kwargs \
                        else results.rotation_steps(field=field)
                    total = len(results.steps(field))
                    if not said and len(steps) < total:
                        said.append(True)
                        print(f"[sweep] curves: using the last of "
                              f"{results.n_rotations()} rotations "
                              f"({len(steps)} of {total} output steps) for the "
                              f"local-solar-time curve", file=sys.stderr)
                    x, mean, _, _, _ = results.local_solar_time_curve(
                        field=field, steps=steps, **kwargs)
                    return x, mean
            else:
                def of(results):
                    return results.series(field=field, lat=lat, lon=lon,
                                          **kwargs)
        out = []
        for point in self.points:
            complete, why = point.completeness()
            if not complete:
                if missing == "skip":
                    continue
                raise RuntimeError(f"point {point.name} is not complete ({why})")
            x, y = of(point.results)
            text = (label(point.params) if callable(label)
                    else self._default_label(point.params))
            out.append(Curve(text, dict(point.params), x, y))
        return out

    @staticmethod
    def _default_label(params):
        parts = []
        for key, value in params.items():
            short = key.split(".")[-1].replace("_", " ")
            parts.append(f"{short} = {value:g}" if isinstance(value, float)
                         else f"{short} = {value}")
        return ", ".join(parts)

    # -- plots (thin delegations, so importing ptm.sweep never imports pyplot) --

    def plot_curves(self, **kwargs):
        """Family of curves, one per grid point. See
        :func:`ptm.plots.plot_sweep_curves`."""
        from . import plots
        return plots.plot_sweep_curves(self, **kwargs)

    def plot_residuals(self, obs_x, obs_y, **kwargs):
        """Curves against observations, with residuals. See
        :func:`ptm.plots.plot_sweep_residuals`."""
        from . import plots
        return plots.plot_sweep_residuals(self, obs_x, obs_y, **kwargs)


# --- reopening a finished sweep -------------------------------------------------

def load_sweep(outdir, **overrides):
    """Rebuild a :class:`Sweep` from the ``sweep.json`` in ``outdir``.

    Reading a finished sweep should not require retyping the grid. The engine
    is still resolved (a sweep is a runnable object), and every field in the
    manifest can be overridden by keyword -- ``exe=`` in particular, when the
    binary that produced the sweep has since been rebuilt or moved.
    """
    outdir = Path(outdir).expanduser().resolve()
    manifest_path = outdir / _MANIFEST
    if not manifest_path.is_file():
        raise FileNotFoundError(f"{manifest_path} not found; not a sweep directory")
    manifest = json.loads(manifest_path.read_text())

    kwargs = dict(
        base_config=manifest["base_config"],
        grid=manifest["axes"],
        outdir=outdir,
        overrides=manifest.get("overrides") or {},
        label_axes=manifest.get("label_axes") or (),
        exe=manifest.get("engine"),
        workdir=manifest.get("working_dir"),
        engine_args=manifest.get("engine_args", ()),
        timeout=manifest.get("timeout_seconds"),
        # The route is a property of the engine, and the manifest records what
        # it was. Re-detecting is right when the engine is the same binary and
        # wrong when it is not, so the recorded route is the default and a
        # caller who rebuilt passes use_flags='auto' explicitly.
        use_flags=any(v == "flag" for v in manifest.get("route", {}).values()),
        quiet=True,
    )
    kwargs.update(overrides)
    sweep = Sweep(**kwargs)
    recorded_md5 = manifest.get("engine_md5")
    if recorded_md5 and recorded_md5 != sweep.exe_md5:
        print(f"[sweep] the engine at {sweep.exe} (md5 {sweep.exe_md5}) is not "
              f"the one that produced this sweep (md5 {recorded_md5})",
              file=sys.stderr)
    return sweep
