"""The ``ptm`` command: look at a finished run without writing any Python.

Four subcommands::

    ptm info  <run_dir>                  what the run is, and what it does not record
    ptm plot  <run_dir> <what> [--save]  diurnal | map | globes | hist
    ptm run   <config.yaml>              start the engine on a config
    ptm sweep <config.yaml> --vary K=V,V run a grid of them

``ptm info`` and ``ptm plot`` close the gap the install audit measured: the
documented quickstart ends at a stdlib snippet printing three floats, and every
route from there to a picture needed a second environment (the GUI's twelve
unpinned dependencies, Jupyter -- which is named in no requirements file -- or
ParaView).

``ptm run`` is a single serial run and nothing more. ``ptm sweep`` is the same
engine driver over a grid, with the resume, provenance and collection that the
nineteen bespoke sweep scripts in this tree each re-implemented.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

__all__ = ["main"]

_PLOT_KINDS = ("diurnal", "map", "globes", "hist")


def _add_common(parser):
    parser.add_argument("run_dir", help="a finished PTM3D output directory")


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="ptm",
        description="Inspect and plot PTM3D runs, and start the engine.")
    parser.add_argument("--version", action="store_true",
                        help="print the ptm package version and exit")
    sub = parser.add_subparsers(dest="command")

    info = sub.add_parser(
        "info", help="print what a run directory records (and what it does not)")
    _add_common(info)
    info.add_argument("--json", action="store_true",
                      help="emit provenance() as JSON instead of a report")

    plot = sub.add_parser("plot", help="draw one of the canned figures")
    _add_common(plot)
    plot.add_argument("what", choices=_PLOT_KINDS,
                      help="diurnal (one face over time), map (regridded "
                           "lat/lon), globes (orthographic set), hist "
                           "(distribution)")
    plot.add_argument("--field", default="temperature",
                      help="field name (default: temperature)")
    plot.add_argument("--moment", default="max", choices=("avg", "max", "min"),
                      help="which statistic to draw (default: max); ignored "
                           "with --step")
    plot.add_argument("--step", type=int, default=None,
                      help="draw one output step instead of a statistic")
    plot.add_argument("--lat", type=float, default=None,
                      help="latitude for 'diurnal' (degrees)")
    plot.add_argument("--lon", type=float, default=None,
                      help="longitude for 'diurnal' (degrees)")
    plot.add_argument("--face", type=int, default=None,
                      help="face index for 'diurnal', instead of --lat/--lon")
    plot.add_argument("--resolution", type=float, default=None,
                      help="map grid resolution in degrees (default: estimated "
                           "from the mesh)")
    plot.add_argument("--units", default="ascii", choices=("ascii", "unicode"),
                      help="unit spelling in labels: W/m^2 or W/m2 with a "
                           "superscript (default: ascii)")
    plot.add_argument("--mesh-units", default=None, choices=("m", "km"),
                      help="override the mesh length unit if it cannot be "
                           "inferred")
    plot.add_argument("--save", default=None, metavar="FILE",
                      help="write the figure here (extension picks the format)")
    plot.add_argument("--show", action="store_true",
                      help="open an interactive window")
    plot.add_argument("--dpi", type=int, default=200, help="raster DPI (200)")

    run = sub.add_parser("run", help="run the engine on one config")
    run.add_argument("config", help="a PTM3D YAML config")
    run.add_argument("--exe", default=None,
                     help="engine binary (default: found via PTM_BUILD_DIR, "
                          "then <repo>/build/bin)")
    run.add_argument("--workdir", default=None,
                     help="working directory (default: the repository root, "
                          "which is where the shipped metakernels resolve)")

    sweep = sub.add_parser(
        "sweep", help="run a grid of simulations from one base config",
        description="Run a parameter grid. Serial by default: there is one "
                    "GPU. Resumable: re-running skips points that finished. "
                    "Reading a finished sweep needs no engine.",
        epilog="thermal inertia is TWO parameters, and they are different physics:\n"
               "  thermal_inertia            ks = TI^2/(rhos*cp(273 K)), kd scaled\n"
               "                             with it; depth contrast preserved.\n"
               "                             Needs an engine with --thermal-inertia.\n"
               "  constant_thermal_inertia   cp = TI^2/(ks*rhos), chi=0, kd=ks,\n"
               "                             rhod=rhos; TI exactly constant in\n"
               "                             temperature AND depth. Any engine.\n"
               "  They give measurably different diurnal amplitudes on the same\n"
               "  config. Pick one deliberately; you cannot sweep both at once,\n"
               "  and neither is a default.\n"
               "\n"
               "examples:\n"
               "  ptm sweep examples/quickstart_asteroid_diurnal.yaml \\\n"
               "      --vary thermal_inertia=50,100,200,400\n"
               "  ptm sweep output/asteroid_sweep --no-run --table sweep.csv "
               "--plot curves --save family.png\n",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sweep.add_argument(
        "target",
        help="a base YAML config, or an existing sweep directory to reopen")
    sweep.add_argument(
        "--vary", action="append", default=[], metavar="NAME=V1,V2,...",
        help="one grid axis. NAME is 'thermal_inertia', "
             "'constant_thermal_inertia', 'albedo', or any dotted config key "
             "(thermal.equilibration_orbits). Repeat for a cartesian product. "
             "See the note below on the two thermal-inertia conventions.")
    sweep.add_argument("--outdir", default=None,
                       help="where the points go (default: the base config's "
                            "output.directory with _sweep appended)")
    sweep.add_argument("--exe", default=None, help="engine binary")
    sweep.add_argument("--workdir", default=None,
                       help="working directory for the engine (default: the "
                            "repository root)")
    sweep.add_argument("--jobs", type=int, default=1,
                       help="concurrent engine processes (default 1). CPU-only: "
                            "refuses to start if any point asks for the GPU.")
    sweep.add_argument("--timeout", type=float, default=None,
                       help="wall-clock ceiling per point, in seconds")
    sweep.add_argument("--no-resume", action="store_true",
                       help="re-run every point, including finished ones")
    sweep.add_argument("--no-run", action="store_true",
                       help="do not launch anything; only --table / --plot the "
                            "points already on disk")
    sweep.add_argument("--dry-run", action="store_true",
                       help="print the engine command for each point, run none")
    sweep.add_argument("--list", action="store_true",
                       help="print the grid and each point's status, run none")
    sweep.add_argument("--use-flags", default="auto",
                       choices=("auto", "yes", "no"),
                       help="deliver the named parameters through the engine "
                            "CLI flags (auto: use them if this binary has "
                            "them). 'no' cannot serve thermal_inertia, which "
                            "has no config-file equivalent.")
    sweep.add_argument("--table", default=None, metavar="FILE.csv",
                       help="write a tidy table of the finished points")
    sweep.add_argument("--plot", default=None, choices=("curves",),
                       help="draw the family-of-curves figure")
    sweep.add_argument("--field", default="temperature",
                       help="field for --table and --plot (default temperature)")
    sweep.add_argument("--save", default=None, metavar="FILE",
                       help="write the figure here")
    sweep.add_argument("--dpi", type=int, default=200, help="raster DPI (200)")
    return parser


# --- info -------------------------------------------------------------------

def _print_info(run):
    prov = run.provenance()
    print(f"run          {prov['run_dir']}")
    print(f"faces        {prov['n_faces']}")

    time = prov["time"]
    if time:
        cadence = ("" if time["cadence_hours"] is None
                   else f", {time['cadence_hours']:.3g} h cadence")
        print(f"steps        {time['n_steps']} over "
              f"{time['elapsed_hours']:.4g} h "
              f"({time['elapsed_hours'] / 24.0:.4g} d){cadence}")
        print(f"window       {time['first_utc']} .. {time['last_utc']}")
    meta = prov["metadata"]
    if meta.get("failed_timesteps"):
        print(f"WARNING      {meta['failed_timesteps']} timestep(s) failed to "
              f"export")

    unit = prov["mesh_units"]
    corroborated = prov["mesh_units_corroborated_by"]
    how = (f"consistent with the {corroborated} radius under this unit"
           if corroborated
           else "from face-center magnitude alone -- NOT corroborated")
    print(f"mesh units   {unit} ({how})")
    print(f"mesh radius  {prov['mesh_median_radius_m'] / 1e3:.4g} km (median "
          f"|face center|)")

    print(f"per-step     {', '.join(prov['step_fields']) or '(none)'}")
    stats = prov["statistics"]
    if stats:
        grouped = {}
        for field, moment in stats:
            grouped.setdefault(field, []).append(moment or "(scalar)")
        print("statistics   " + "; ".join(
            f"{f} [{', '.join(m)}]" for f, m in sorted(grouped.items())))
    else:
        print("statistics   (none)")

    print(f"units style  {prov['units_style']} "
          f"(flux spelled {run.units_of('illumination')})")
    if prov["config_file"]:
        print(f"config       {prov['config_file']}")
        print(f"  sha256     {prov['config_sha256']}")
    engine = prov["engine"]
    if engine:
        # The point of provenance.txt is that these are answerable at all; a
        # run that has them should not make the reader open the file.
        print(f"engine       {prov['engine_version']} @ "
              f"{prov['engine_git_sha']} "
              f"({engine.get('engine_build_type', '?')}, GPU "
              f"{'on' if engine.get('engine_gpu_enabled') == 'true' else 'off'})")
        print(f"  binary     {engine.get('engine_binary', '?')}")
        print(f"  md5        {prov['engine_binary_md5']}")
        if prov["command_line"]:
            print(f"  invoked    {prov['command_line']}")
    if prov["started_utc"]:
        source = "provenance.txt" if engine.get("run_started_utc") else "heartbeat.txt"
        print(f"started      {prov['started_utc']} ({source})")

    print("\nnot recorded by the run:")
    for item in prov["not_recorded"]:
        print(f"  - {item}")


# --- plot -------------------------------------------------------------------

def _do_plot(args, run):
    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    from . import plots

    # save= is deliberately not forwarded: the writing happens once, below, so
    # that --dpi applies to the file that is actually kept.
    common = dict(show=args.show)
    if args.what == "diurnal":
        if args.face is None and (args.lat is None or args.lon is None):
            raise SystemExit(
                "ptm plot diurnal needs --face, or both --lat and --lon")
        fig = plots.plot_diurnal(run, field=args.field, face=args.face,
                                 lat=args.lat, lon=args.lon, **common)
    elif args.what == "map":
        fig = plots.plot_map(run, args.field, moment=args.moment,
                             step=args.step, resolution=args.resolution,
                             **common)
    elif args.what == "globes":
        fig = plots.plot_globes(run, args.field, moment=args.moment,
                                step=args.step,
                                mark_subsolar=args.step is not None, **common)
    else:
        fig = plots.plot_histogram(run, args.field, moment=args.moment,
                                   step=args.step, **common)

    if args.save:
        fig.savefig(args.save, dpi=args.dpi)
        print(f"wrote {args.save}")
    elif not args.show:
        print("nothing written: pass --save FILE or --show", file=sys.stderr)
        return 1
    return 0


# --- run --------------------------------------------------------------------

def _do_run(args):
    from . import paths, run as run_mod

    config = Path(args.config).expanduser().resolve()
    if not config.is_file():
        raise SystemExit(f"config not found: {config}")

    root = paths.find_repo_root(config.parent) or paths.find_repo_root()
    executable = (Path(args.exe).expanduser() if args.exe
                  else paths.find_engine_executable(
                      search_roots=[root] if root else []))
    if executable is None or not Path(executable).exists():
        raise SystemExit(
            "ptm_thermal_sim not found. Build it, or set PTM_BUILD_DIR, or "
            "pass --exe. See INSTALL.md.")

    workdir = args.workdir or (str(root) if root else None)
    config_dict = {}
    try:
        import yaml
        config_dict = yaml.safe_load(config.read_text()) or {}
    except Exception as exc:                                   # noqa: BLE001
        print(f"[ptm] could not parse {config} for CLI flags ({exc}); "
              f"passing it to the engine unread", file=sys.stderr)

    command = run_mod.build_command(executable, config, config_dict)
    print(f"[ptm] {' '.join(command)}")
    if workdir:
        print(f"[ptm] working directory: {workdir}")
    for msg in run_mod.long_run_advisory(config_dict):
        print(msg, file=sys.stderr)

    process = run_mod.EngineProcess(
        command, working_dir=workdir,
        on_output=lambda line: print(line),
        on_warning=lambda line: print(f"WARNING {line}", file=sys.stderr),
        on_error=lambda line: print(f"ERROR   {line}", file=sys.stderr),
        on_heartbeat=lambda msg, stalled: print(f"[ptm] {msg}", file=sys.stderr),
    )
    status = process.run()
    print(f"[ptm] engine exited {status}")
    return status


# --- sweep --------------------------------------------------------------------

def _parse_scalar(text):
    """``'50'`` -> 50, ``'0.05'`` -> 0.05, ``'true'`` -> True, else the string."""
    lowered = text.strip().lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def _parse_vary(specs):
    """``['ti=1,2', ...]`` -> an ordered ``{name: [values]}``."""
    grid = {}
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(
                f"--vary needs NAME=V1,V2,...; got {spec!r}")
        name, _, values = spec.partition("=")
        name = name.strip()
        items = [v for v in values.split(",") if v.strip() != ""]
        if not items:
            raise SystemExit(f"--vary {name}: no values given")
        if name in grid:
            raise SystemExit(f"--vary {name} given twice")
        grid[name] = [_parse_scalar(v) for v in items]
    return grid


def _print_sweep_list(sweep):
    print(f"base   {sweep.base_config_path}")
    print(f"outdir {sweep.outdir}")
    print(f"engine {sweep.exe} (md5 {sweep.exe_md5})")
    for name, values in sweep.axes.items():
        route = sweep.route.get(name, "config")
        print(f"axis   {name} = {values}  [{route}]")
    print(f"points {len(sweep)} ({'x'.join(str(n) for n in sweep.shape)})")
    for point in sweep:
        complete, why = point.completeness()
        mark = "done" if complete else "todo"
        print(f"  {mark}  {point.name:<40s} {why}")


def _do_sweep(args):
    from . import sweep as sweep_mod

    target = Path(args.target).expanduser()
    if target.is_dir():
        if args.vary:
            raise SystemExit(
                f"{target} is an existing sweep directory; its grid comes from "
                f"sweep.json, so --vary cannot be combined with it")
        sweep = sweep_mod.load_sweep(target, exe=args.exe) if args.exe \
            else sweep_mod.load_sweep(target)
    else:
        if not target.is_file():
            raise SystemExit(f"not a config file or sweep directory: {target}")
        if not args.vary:
            raise SystemExit("ptm sweep needs at least one --vary NAME=V1,V2")
        use_flags = {"auto": "auto", "yes": True, "no": False}[args.use_flags]
        try:
            sweep = sweep_mod.Sweep(
                target, grid=_parse_vary(args.vary), outdir=args.outdir,
                exe=args.exe, workdir=args.workdir, use_flags=use_flags,
                timeout=args.timeout)
        except ValueError as exc:
            raise SystemExit(f"ptm sweep: {exc}")

    if args.list:
        try:
            _print_sweep_list(sweep)
        except (FileNotFoundError, RuntimeError) as exc:
            raise SystemExit(f"ptm sweep: {exc}")
        return 0

    status = 0
    if not args.no_run:
        # A missing engine and an axis this engine cannot deliver are both
        # ordinary operator errors with a paragraph of explanation attached;
        # they belong on stderr as a message, not as a traceback.
        try:
            summary = sweep.run(resume=not args.no_resume, jobs=args.jobs,
                                dry_run=args.dry_run)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            raise SystemExit(f"ptm sweep: {exc}")
        if summary["failed"]:
            status = 1
        if args.dry_run:
            return status

    if args.table:
        import numpy as np
        field = args.field
        rows = sweep.write_table(
            args.table,
            **{f"{field}_max": lambda r: float(np.max(r.stat(field, "max"))),
               f"{field}_min": lambda r: float(np.min(r.stat(field, "min"))),
               f"{field}_mean": lambda r: float(np.mean(r.stat(field, "avg")))})
        print(f"wrote {args.table} ({len(rows)} rows)")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        fig = sweep.plot_curves(field=args.field)
        if not args.save:
            raise SystemExit("--plot needs --save FILE")
        fig.savefig(args.save, dpi=args.dpi)
        print(f"wrote {args.save}")
    return status


# --- entry point ------------------------------------------------------------

def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.version:
        from . import __version__
        print(__version__)
        return 0
    if args.command is None:
        parser.print_help()
        return 1

    if args.command == "run":
        return _do_run(args)
    if args.command == "sweep":
        return _do_sweep(args)

    from .results import Results
    units = getattr(args, "units", "ascii")
    mesh_units = getattr(args, "mesh_units", None)
    try:
        run = Results(args.run_dir, units_style=units, mesh_units=mesh_units)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"ptm: {exc}")

    if args.command == "info":
        if args.json:
            import json
            print(json.dumps(run.provenance(), indent=2, default=str))
            return 0
        _print_info(run)
        return 0
    return _do_plot(args, run)


if __name__ == "__main__":
    raise SystemExit(main())
