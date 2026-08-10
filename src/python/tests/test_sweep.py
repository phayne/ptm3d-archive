#!/usr/bin/env python3
"""The Sweep layer, driven end to end against a stub engine.

No C++ engine, no SPICE, no GPU, no network. A 40-line Python script stands in
for ``ptm_thermal_sim``: it answers ``--help`` with a flag list this test
controls, and on ``--config`` it writes a run directory in the exact byte
layouts the engine writes (reusing ``test_results_api.make_run``). That makes
the interesting cases -- what happens when a point is killed halfway, what
happens when the engine grows a flag -- testable rather than argued about.

What is asserted, in the order it would break:

1. **Flag detection is whole-token.** Today's engine advertises
   ``--albedo-map`` and no ``--albedo``; a substring test says yes to both, and
   the sweep would then hand the engine a flag it does not have.
2. **The flag route and the config route are interchangeable**: the parameter
   is delivered exactly once either way, never twice and never zero times.
3. **Resume is decided on evidence.** A point killed mid-run leaves a directory
   full of files and no results; a point whose base config changed is stale; a
   point marked complete whose outputs were deleted is not complete.
4. The generated config is written beside its own outputs, with the one
   YAML-relative engine path anchored so it still resolves.
5. The collection accessors return the grid in grid order.
6. ``jobs > 1`` refuses to start on a GPU config.
7. The depth-series reader trims equilibration by matching the time axis, and
   never guesses when it cannot.
8. Both sweep figures draw.

Run directly (no pytest needed)::

    python3 src/python/tests/test_sweep.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                              # noqa: E402
import yaml                                                     # noqa: E402

from ptm import readers, run as run_mod                         # noqa: E402
from ptm.results import Results                                 # noqa: E402
from ptm.sweep import (NAMED_PARAMETERS, Sweep, get_dotted,     # noqa: E402
                       load_sweep, set_dotted)
from test_results_api import make_run                           # noqa: E402


# --- a stub engine -------------------------------------------------------------

STUB = '''\
#!/usr/bin/env python3
"""Stands in for ptm_thermal_sim: answers --help, and writes a run directory."""
import sys
from pathlib import Path

HELP = """PTM3D thermal simulator
  --config FILE              Configuration file
  --albedo-map <file>        Custom albedo FITS map file
  --thermal-inertia-map FILE Path to thermal inertia map file
EXTRA_FLAGS"""

if "--help" in sys.argv:
    print(HELP)
    raise SystemExit(0)

sys.path.insert(0, r"TESTS_DIR")
sys.path.insert(0, r"PKG_DIR")
import yaml
from test_results_api import make_run

config_path = sys.argv[sys.argv.index("--config") + 1]
config = yaml.safe_load(Path(config_path).read_text())
outdir = Path(config["output"]["directory"])
make_run(outdir, n_steps=8)
# Record the argv so a test can see which route delivered the parameter.
(outdir / "argv.json").write_text(repr(sys.argv[1:]))
print("Output written to " + str(outdir))
'''


def write_stub(path, extra_flags=()):
    """A stub engine advertising ``extra_flags`` on top of the real ones."""
    lines = "\n".join(f"  {flag} VALUE           override" for flag in extra_flags)
    text = (STUB.replace("EXTRA_FLAGS", lines)
                .replace("TESTS_DIR", str(Path(__file__).resolve().parent))
                .replace("PKG_DIR", str(REPO_ROOT / "src" / "python")))
    path.write_text(f"#!{sys.executable}\n" + text.split("\n", 1)[1])
    path.chmod(0o755)
    return path


def write_base_config(path, outdir, gpu=False, view_factor=None):
    config = {
        "simulation": {"body": "Moon", "duration_days": 1.0, "timesteps": 8},
        "mesh": {"resolution": 3},
        "thermal": {"run_thermal_mesh": True, "use_gpu": gpu,
                    "equilibration_orbits": 2.0},
        "illumination": {"use_albedo_map": False},
        "output": {"directory": str(outdir)},
    }
    if view_factor:
        config["radiation"] = {"enabled": True, "view_factor_file": view_factor}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


# --- 1. flag detection ----------------------------------------------------------

def test_flag_detection_is_whole_token(tmp):
    exe = write_stub(tmp / "engine_old.py")
    assert run_mod.engine_supports_flag(exe, "--albedo-map"), (
        "the stub advertises --albedo-map")
    assert not run_mod.engine_supports_flag(exe, "--albedo"), (
        "--albedo is NOT advertised; a substring test against the help text "
        "would say it is, because --albedo-map contains it. That mistake "
        "hands the engine a flag it does not have.")
    assert not run_mod.engine_supports_flag(exe, "--thermal-inertia")
    assert run_mod.engine_supports_flag(exe, "--thermal-inertia-map")

    new = write_stub(tmp / "engine_new.py",
                     extra_flags=("--thermal-inertia", "--albedo"))
    assert run_mod.engine_supports_flag(new, "--albedo")
    assert run_mod.engine_supports_flag(new, "--thermal-inertia")


def test_missing_engine_reports_rather_than_claims_support(tmp):
    flags = run_mod.engine_flags(tmp / "does_not_exist")
    assert flags == frozenset(), "a missing binary advertises nothing"


# --- 2. the two routes are interchangeable ---------------------------------------

def _one_point_config_and_argv(sweep):
    point = sweep.points[0]
    return point.config(), point.command()


NEW_FLAGS = ("--thermal-inertia", "--constant-thermal-inertia", "--albedo")


def test_config_route_when_the_flag_is_absent(tmp):
    base = write_base_config(tmp / "cfg" / "base.yaml", tmp / "out")
    sweep = Sweep(base, grid={"constant_thermal_inertia": [50.0, 200.0]},
                  outdir=tmp / "sweep_cfg", exe=write_stub(tmp / "old.py"),
                  workdir=tmp, quiet=True)
    assert sweep.route["constant_thermal_inertia"] == "config"
    config, argv = _one_point_config_and_argv(sweep)
    key = NAMED_PARAMETERS["constant_thermal_inertia"].config_key
    assert get_dotted(config, key) == 50.0, (
        f"the config route must set {key}")
    assert "--constant-thermal-inertia" not in argv


def test_flag_route_when_the_engine_has_it(tmp):
    base = write_base_config(tmp / "cfg" / "base.yaml", tmp / "out")
    sweep = Sweep(base, grid={"constant_thermal_inertia": [50.0, 200.0]},
                  outdir=tmp / "sweep_flag",
                  exe=write_stub(tmp / "new.py", extra_flags=NEW_FLAGS),
                  workdir=tmp, quiet=True)
    assert sweep.route["constant_thermal_inertia"] == "flag"
    config, argv = _one_point_config_and_argv(sweep)
    key = NAMED_PARAMETERS["constant_thermal_inertia"].config_key
    assert get_dotted(config, key) is None, (
        "with the flag in use the config key must NOT also be written: two "
        "sources for one quantity is how they drift apart")
    assert "--constant-thermal-inertia" in argv
    assert argv[argv.index("--constant-thermal-inertia") + 1].startswith("50")


def test_the_parameter_is_delivered_exactly_once(tmp):
    """Whichever route is taken, the value appears once. Never both, never none."""
    base = write_base_config(tmp / "cfg" / "base.yaml", tmp / "out")
    for name, extra in (("old", ()), ("new", NEW_FLAGS)):
        sweep = Sweep(base,
                      grid={"constant_thermal_inertia": [123.0],
                            "albedo": [0.2]},
                      outdir=tmp / f"sweep_once_{name}",
                      exe=write_stub(tmp / f"engine_{name}.py",
                                     extra_flags=extra),
                      workdir=tmp, quiet=True)
        config, argv = _one_point_config_and_argv(sweep)
        for param, value in (("constant_thermal_inertia", 123.0),
                             ("albedo", 0.2)):
            spec = NAMED_PARAMETERS[param]
            in_config = get_dotted(config, spec.config_key) is not None
            in_argv = spec.flag in argv
            assert in_config != in_argv, (
                f"{param} on the {name} engine: in_config={in_config}, "
                f"in_argv={in_argv} -- it must be exactly one")

    # thermal_inertia has no config key at all, so on the new engine it is the
    # flag and nothing else.
    sweep = Sweep(base, grid={"thermal_inertia": [123.0]},
                  outdir=tmp / "sweep_once_ti",
                  exe=write_stub(tmp / "engine_ti.py", extra_flags=NEW_FLAGS),
                  workdir=tmp, quiet=True)
    config, argv = _one_point_config_and_argv(sweep)
    assert "--thermal-inertia" in argv
    assert get_dotted(config, "thermophysical.constant_thermal_inertia") is None
    assert get_dotted(config, "thermophysical.ks_override") is None, (
        "the wrapper must not compute ks itself -- the engine does that")


def test_thermal_inertia_refuses_rather_than_switching_convention(tmp):
    """The 3a contract: --thermal-inertia is ks-scaling, constant_thermal_inertia
    is cp-solving, and they differ by ~3 K. An engine without the flag must get
    an error, never the other convention wearing the same name."""
    base = write_base_config(tmp / "cfg" / "base.yaml", tmp / "out")
    old = write_stub(tmp / "old_ti.py")
    for kwargs, why in (({}, "an engine without the flag"),
                        ({"use_flags": False}, "use_flags=False")):
        sweep = Sweep(base, grid={"thermal_inertia": [50.0]},
                      outdir=tmp / "sweep_ti_refuse", exe=old, workdir=tmp,
                      quiet=True, **kwargs)
        try:
            sweep.route
        except (RuntimeError, FileNotFoundError) as exc:
            text = str(exc)
            assert "constant_thermal_inertia" in text, (
                f"{why}: the error must name the alternative")
            assert "NOT THE SAME PHYSICS" in text, (
                f"{why}: the error must say the two conventions differ")
        else:
            raise AssertionError(
                f"{why}: thermal_inertia must not silently fall back to "
                f"thermophysical.constant_thermal_inertia -- that is a "
                f"different column (39.27 K vs 42.19 K at TI=200)")


def test_the_two_thermal_inertia_conventions_cannot_both_be_swept(tmp):
    base = write_base_config(tmp / "cfg" / "base.yaml", tmp / "out")
    try:
        Sweep(base, grid={"thermal_inertia": [50.0],
                          "constant_thermal_inertia": [50.0]},
              outdir=tmp / "sweep_both",
              exe=write_stub(tmp / "both.py", extra_flags=NEW_FLAGS),
              workdir=tmp, quiet=True)
    except ValueError as exc:
        assert "two conventions" in str(exc)
    else:
        raise AssertionError("the engine refuses both; so must this")


def test_requiring_an_absent_flag_is_an_error(tmp):
    base = write_base_config(tmp / "cfg" / "base.yaml", tmp / "out")
    sweep = Sweep(base, grid={"albedo": [0.1]}, outdir=tmp / "sweep_req",
                  exe=write_stub(tmp / "old2.py"), workdir=tmp,
                  use_flags=True, quiet=True)
    try:
        sweep.route
    except RuntimeError as exc:
        assert "--albedo" in str(exc)
    else:
        raise AssertionError("use_flags=True must not silently fall back")


# --- 3. the generated config ------------------------------------------------------

def test_generated_config_lands_beside_its_outputs(tmp):
    base = write_base_config(tmp / "cfg" / "base.yaml", tmp / "out")
    sweep = Sweep(base, grid={"thermal.equilibration_orbits": [1.0, 5.0]},
                  outdir=tmp / "sweep_cfgdir", exe=write_stub(tmp / "e.py"),
                  workdir=tmp, quiet=True)
    point = sweep.points[1]
    config = point.config()
    assert get_dotted(config, "thermal.equilibration_orbits") == 5.0
    assert get_dotted(config, "output.directory") == str(point.dir)
    assert point.config_path.parent == point.dir, (
        "generated configs belong beside their outputs, not in the tracked "
        "examples/ directory, where 35 of them are sitting today")


def test_view_factor_path_is_anchored_to_the_base_config(tmp):
    """The one key the engine resolves against the YAML's own directory."""
    base = write_base_config(tmp / "cfg" / "base.yaml", tmp / "out",
                             view_factor="vf/moon.hmat")
    sweep = Sweep(base, grid={"albedo": [0.1]}, outdir=tmp / "sweep_vf",
                  exe=write_stub(tmp / "e2.py"), workdir=tmp, quiet=True)
    resolved = get_dotted(sweep.points[0].config(), "radiation.view_factor_file")
    assert Path(resolved).is_absolute()
    # Compared against the sweep's own record of where the base config is:
    # Path.resolve() rewrites /var -> /private/var on macOS, and the point of
    # the check is the directory the path is anchored to, not that spelling.
    assert Path(resolved) == sweep.base_config_path.parent / "vf" / "moon.hmat", (
        "moving the YAML must not move what its relative VF path means "
        "(config_parser.cpp:507-517)")
    assert ".." not in resolved, "the composed path must be normalized"


def test_fixed_overrides_apply_to_every_point(tmp):
    base = write_base_config(tmp / "cfg" / "base.yaml", tmp / "out")
    sweep = Sweep(base, grid={"albedo": [0.1, 0.2]}, outdir=tmp / "sweep_ovr",
                  overrides={"thermal.threads": 4}, exe=write_stub(tmp / "e3.py"),
                  workdir=tmp, quiet=True)
    for point in sweep:
        assert get_dotted(point.config(), "thermal.threads") == 4


def test_derived_keys_come_from_the_callers_own_relation(tmp):
    base = write_base_config(tmp / "cfg" / "base.yaml", tmp / "out")
    sweep = Sweep(base, grid={"albedo": [0.1, 0.4]}, outdir=tmp / "sweep_der",
                  derived={"thermophysical.emissivity_override":
                           lambda p: 1.0 - p["albedo"]},
                  exe=write_stub(tmp / "e4.py"), workdir=tmp, quiet=True)
    values = [get_dotted(p.config(), "thermophysical.emissivity_override")
              for p in sweep]
    assert values == [0.9, 0.6]


# --- 4. the grid ------------------------------------------------------------------

def test_an_undeclared_bare_axis_name_is_refused(tmp):
    """A label axis must be declared, or it lands in the YAML as a dead key."""
    base = write_base_config(tmp / "cfg" / "base.yaml", tmp / "out")
    try:
        Sweep(base, grid={"dem": ["agenor", "cilix"]},
              outdir=tmp / "sweep_bare", exe=write_stub(tmp / "bare.py"),
              workdir=tmp, quiet=True)
    except ValueError as exc:
        assert "label_axes" in str(exc)
    else:
        raise AssertionError(
            "nothing in the engine rejects an unknown top-level key, so a "
            "stray 'dem: agenor' would sit in every generated config read by "
            "nothing")

    ok = Sweep(base, grid={"dem": ["agenor", "cilix"]},
               label_axes=("dem",), outdir=tmp / "sweep_label",
               derived={"mesh.import_file": lambda p: f"{p['dem']}.ply"},
               exe=write_stub(tmp / "label.py"), workdir=tmp, quiet=True)
    config = ok.points[0].config()
    assert "dem" not in config, "a label axis is never written to the config"
    assert get_dotted(config, "mesh.import_file") == "agenor.ply"


def test_grid_is_a_cartesian_product_in_order(tmp):
    base = write_base_config(tmp / "cfg" / "base.yaml", tmp / "out")
    sweep = Sweep(base, grid={"albedo": [0.1, 0.2, 0.3],
                              "thermal_inertia": [50.0, 100.0]},
                  outdir=tmp / "sweep_grid", exe=write_stub(tmp / "e5.py"),
                  workdir=tmp, quiet=True)
    assert sweep.shape == (3, 2) and len(sweep) == 6
    assert [p.params["albedo"] for p in sweep] == [0.1, 0.1, 0.2, 0.2, 0.3, 0.3]
    assert sweep.points[0].name == "albedo-0p1__thermal_inertia-50"


def test_colliding_point_names_are_refused(tmp):
    base = write_base_config(tmp / "cfg" / "base.yaml", tmp / "out")
    try:
        Sweep(base, grid={"albedo": [0.1, 0.2]}, outdir=tmp / "sweep_col",
              exe=write_stub(tmp / "e6.py"), workdir=tmp, quiet=True,
              name_from=lambda params: "same")
    except ValueError as exc:
        assert "same" in str(exc)
    else:
        raise AssertionError("two points sharing a directory must be refused")


# --- 5. running, and resuming -------------------------------------------------------

def _run_sweep(tmp, name, grid=None, **kwargs):
    base = kwargs.pop("base", None) or write_base_config(
        tmp / "cfg" / f"{name}.yaml", tmp / "out")
    sweep = Sweep(base, grid=grid or {"albedo": [0.1, 0.2, 0.3]},
                  outdir=tmp / name, exe=write_stub(tmp / f"{name}_engine.py"),
                  workdir=tmp, quiet=True, **kwargs)
    return sweep, sweep.run()


def test_a_sweep_runs_and_collects(tmp):
    sweep, summary = _run_sweep(tmp, "sweep_run")
    assert summary["complete"] == 3 and summary["failed"] == 0
    assert [p.status() for p in sweep] == ["complete"] * 3
    assert np.allclose(sweep.values("albedo"), [0.1, 0.2, 0.3])
    tmax = sweep.collect(lambda r: float(r.stat("temperature", "max").max()))
    assert tmax.shape == (3,) and np.all(np.isfinite(tmax))
    assert sweep.stat("temperature", "max").shape == (3, 320)
    rows = sweep.table(tmax=lambda r: float(r.stat("temperature", "max").max()))
    assert len(rows) == 3 and set(rows[0]) >= {"albedo", "status", "tmax"}


def test_resume_skips_only_what_is_genuinely_done(tmp):
    sweep, _ = _run_sweep(tmp, "sweep_resume")
    again = sweep.run()
    assert again["skipped"] == 3 and again["planned"] == 0, (
        "a completed sweep must re-run nothing")


def test_an_interrupted_point_is_not_complete(tmp):
    sweep, _ = _run_sweep(tmp, "sweep_killed")
    point = sweep.points[1]
    marker = json.loads(point.marker_path.read_text())
    marker["status"] = "running"          # what a kill mid-flight leaves behind
    point.marker_path.write_text(json.dumps(marker))
    complete, why = point.completeness()
    assert not complete and "running" in why, (
        "the directory is full of files and the run never finished; a "
        "does-the-directory-exist check calls this done")
    assert sweep.run()["planned"] == 1


def test_a_changed_base_config_invalidates_a_point(tmp):
    base = write_base_config(tmp / "cfg" / "chg.yaml", tmp / "out")
    sweep, _ = _run_sweep(tmp, "sweep_changed", base=base)
    config = yaml.safe_load(base.read_text())
    set_dotted(config, "thermal.equilibration_orbits", 99.0)
    base.write_text(yaml.safe_dump(config, sort_keys=False))
    fresh = Sweep(base, grid={"albedo": [0.1, 0.2, 0.3]},
                  outdir=tmp / "sweep_changed",
                  exe=write_stub(tmp / "sweep_changed_engine.py"),
                  workdir=tmp, quiet=True)
    complete, why = fresh.points[0].completeness()
    assert not complete and "config changed" in why


def test_deleted_outputs_invalidate_a_complete_marker(tmp):
    sweep, _ = _run_sweep(tmp, "sweep_deleted")
    shutil.rmtree(sweep.points[0].dir / "statistics")
    complete, why = sweep.points[0].completeness()
    assert not complete and "statistics" in why, (
        "'complete' in a marker file is a claim; the outputs are the evidence")


def test_a_failing_engine_is_recorded_as_failed(tmp):
    base = write_base_config(tmp / "cfg" / "fail.yaml", tmp / "out")
    broken = tmp / "broken.py"
    broken.write_text(f"#!{sys.executable}\n"
                      "import sys\n"
                      "print('  --config FILE  x') if '--help' in sys.argv "
                      "else None\n"
                      "raise SystemExit(0 if '--help' in sys.argv else 3)\n")
    broken.chmod(0o755)
    sweep = Sweep(base, grid={"albedo": [0.1]}, outdir=tmp / "sweep_fail",
                  exe=broken, workdir=tmp, quiet=True)
    summary = sweep.run()
    assert summary["failed"] == 1 and summary["complete"] == 0
    assert "exited 3" in sweep.points[0].marker()["reason"]


def test_exit_zero_without_outputs_is_still_a_failure(tmp):
    base = write_base_config(tmp / "cfg" / "empty.yaml", tmp / "out")
    quiet_liar = tmp / "liar.py"
    quiet_liar.write_text(f"#!{sys.executable}\n"
                          "import sys\n"
                          "print('  --config FILE  x')\n")
    quiet_liar.chmod(0o755)
    sweep = Sweep(base, grid={"albedo": [0.1]}, outdir=tmp / "sweep_liar",
                  exe=quiet_liar, workdir=tmp, quiet=True)
    assert sweep.run()["failed"] == 1, "rc=0 is not success"
    assert "metadata.txt is missing" in sweep.points[0].marker()["reason"]


def test_dry_run_launches_nothing(tmp):
    base = write_base_config(tmp / "cfg" / "dry.yaml", tmp / "out")
    sweep = Sweep(base, grid={"albedo": [0.1, 0.2]}, outdir=tmp / "sweep_dry",
                  exe=write_stub(tmp / "dry_engine.py"), workdir=tmp, quiet=True)
    summary = sweep.run(dry_run=True)
    assert summary["dry_run"] and summary["planned"] == 2
    assert not (sweep.points[0].dir / "metadata.txt").exists()


def test_the_manifest_round_trips(tmp):
    sweep, _ = _run_sweep(tmp, "sweep_manifest")
    reopened = load_sweep(sweep.outdir)
    assert reopened.axes == sweep.axes
    assert [p.name for p in reopened] == [p.name for p in sweep]
    assert all(p.is_complete() for p in reopened)


def test_provenance_is_recorded_with_the_point(tmp):
    sweep, _ = _run_sweep(tmp, "sweep_prov")
    marker = sweep.points[0].marker()
    for key in ("params", "config_overrides", "engine", "engine_md5",
                "base_config_sha256", "config_sha256", "command",
                "wall_seconds", "started_utc"):
        assert key in marker, f"sweep_point.json must record {key}"
    assert marker["params"] == {"albedo": 0.1}


# --- 5b. no engine present ------------------------------------------------------------
# The python-layer CI job never builds the engine, and neither does a colleague
# who was handed a results directory. Everything that is analysis must work
# without one; only launching may demand it.

class _NoEngine:
    """Make ptm.paths.find_engine_executable find nothing, as on a fresh clone."""

    def __enter__(self):
        from ptm import paths
        self._paths, self._real = paths, paths.find_engine_executable
        paths.find_engine_executable = lambda *a, **k: None
        return self

    def __exit__(self, *exc):
        self._paths.find_engine_executable = self._real
        return False


def test_a_sweep_is_usable_with_no_engine_at_all(tmp):
    base = write_base_config(tmp / "cfg" / "base.yaml", tmp / "out")
    with _NoEngine():
        sweep = Sweep(base,
                      grid={"thermal.equilibration_orbits": [1.0, 5.0, 9.0]},
                      outdir=tmp / "sweep_noengine", workdir=tmp, quiet=True)
        assert sweep.exe_or_none is None
        assert len(sweep) == 3

        # --list: the status of every point, with its reason.
        assert [p.completeness()[0] for p in sweep] == [False] * 3

        # the generated configs, and the route
        assert sweep.route == {"thermal.equilibration_orbits": "config"}
        assert get_dotted(sweep.points[1].config(),
                          "thermal.equilibration_orbits") == 5.0

        # --dry-run
        summary = sweep.run(dry_run=True)
        assert summary["planned"] == 3 and summary["dry_run"]
        assert sweep.points[0].command()[0] == "ptm_thermal_sim"

        # and launching is the one thing that fails, with a message that says
        # reading does not need one.
        try:
            sweep.run()
        except FileNotFoundError as exc:
            assert "PTM_BUILD_DIR" in str(exc) and "load_sweep" in str(exc)
        else:
            raise AssertionError("run() must demand an engine")


def test_a_finished_sweep_is_readable_with_no_engine(tmp):
    """Analysis of results that already exist must never need a binary."""
    sweep, _ = _run_sweep(tmp, "sweep_read")
    (tmp / "sweep_read_engine.py").unlink()      # the binary is gone
    with _NoEngine():
        reopened = load_sweep(sweep.outdir)
        assert reopened.exe_or_none is None
        assert len(reopened.completed()) == 3
        assert np.allclose(reopened.values("albedo"), [0.1, 0.2, 0.3])
        rows = reopened.table(
            tmax=lambda r: float(r.stat("temperature", "max").max()))
        assert len(rows) == 3
        assert reopened.grid_of(
            lambda r: float(r.stat("temperature", "max").max())).shape == (3,)
        reopened.write_table(tmp / "read.csv")
        assert (tmp / "read.csv").exists()

        import matplotlib
        matplotlib.use("Agg")
        reopened.plot_curves(
            of=lambda r: r.series("temperature", lat=0.0, lon=0.0),
            save=tmp / "read.png")
        assert (tmp / "read.png").exists()


def test_a_finished_flag_route_sweep_reopens_without_asking_the_engine(tmp):
    """The manifest records the route, so reopening never re-detects."""
    base = write_base_config(tmp / "cfg" / "flagged.yaml", tmp / "out")
    sweep = Sweep(base, grid={"thermal_inertia": [50.0, 200.0]},
                  outdir=tmp / "sweep_flagged",
                  exe=write_stub(tmp / "flagged_engine.py",
                                 extra_flags=NEW_FLAGS),
                  workdir=tmp, quiet=True)
    sweep.run()
    (tmp / "flagged_engine.py").unlink()
    with _NoEngine():
        reopened = load_sweep(sweep.outdir)
        # thermal_inertia is flag-only: without the recorded route this would
        # raise, because there is no engine to ask and no config fallback.
        assert reopened.route == {"thermal_inertia": "flag"}
        assert len(reopened.completed()) == 2


# --- 6. parallelism is opt-in and CPU-only ------------------------------------------

def test_parallel_refuses_a_gpu_config(tmp):
    base = write_base_config(tmp / "cfg" / "gpu.yaml", tmp / "out", gpu=True)
    sweep = Sweep(base, grid={"albedo": [0.1, 0.2]}, outdir=tmp / "sweep_gpu",
                  exe=write_stub(tmp / "gpu_engine.py"), workdir=tmp, quiet=True)
    try:
        sweep.run(jobs=2)
    except RuntimeError as exc:
        assert "one GPU" in str(exc)
    else:
        raise AssertionError("concurrent engine processes on one GPU wedge it")


def test_parallel_runs_a_cpu_sweep(tmp):
    base = write_base_config(tmp / "cfg" / "par.yaml", tmp / "out")
    sweep = Sweep(base, grid={"albedo": [0.1, 0.2, 0.3, 0.4]},
                  outdir=tmp / "sweep_par",
                  exe=write_stub(tmp / "par_engine.py"), workdir=tmp, quiet=True)
    assert sweep.run(jobs=2)["complete"] == 4


# --- 7. the depth-series product -----------------------------------------------------

def _write_depth_csv(path, utc, depths, temperature):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "time," + ",".join(f"T(z={d:.3f} m)" for d in depths)
    lines = ["# Location: equator", header]
    for label, row in zip(utc, temperature):
        lines.append(label + "," + ",".join(f"{v:.6f}" for v in row))
    path.write_text("\n".join(lines) + "\n")


def test_depth_series_round_trips(tmp):
    depths = np.array([0.0, 0.01, 0.05, 0.2])
    utc = [f"2026-04-{i + 1:02d}T00:00:00.000" for i in range(4)]
    temperature = np.arange(16, dtype=float).reshape(4, 4) + 100.0
    path = tmp / "depths" / "equator_depths.csv"
    _write_depth_csv(path, utc, depths, temperature)
    series = readers.read_depth_series(path)
    assert series.location == "equator"
    assert np.allclose(series.depths_m, depths)
    assert np.allclose(series.temperature, temperature)


def test_depth_series_trims_equilibration_by_matching_the_time_axis(tmp):
    root = make_run(tmp / "depth_run", n_steps=8)
    run = Results(root)
    output_utc = list(run.times().utc)
    # The engine writes the equilibration period into this file and into no
    # other product: 8 output rows preceded here by 16 equilibration rows.
    utc = [f"2026-03-{i + 1:02d}T00:00:00.000" for i in range(16)] + output_utc
    depths = np.array([0.0, 0.02, 0.1])
    temperature = np.arange(len(utc) * 3, dtype=float).reshape(len(utc), 3)
    _write_depth_csv(root / "locations" / "equator_depths.csv", utc, depths,
                     temperature)

    trimmed = run.depth_series()
    assert trimmed.n_equilibration == 16
    assert len(trimmed.utc) == 8 and list(trimmed.utc) == output_utc
    assert np.allclose(trimmed.temperature, temperature[-8:])

    whole = run.depth_series(window="all")
    assert len(whole.utc) == 24 and whole.n_equilibration == 0


def test_depth_series_refuses_to_guess_the_window(tmp):
    root = make_run(tmp / "depth_bad", n_steps=8)
    run = Results(root)
    utc = [f"2099-01-{i + 1:02d}T00:00:00.000" for i in range(12)]
    depths = np.array([0.0, 0.02])
    _write_depth_csv(root / "locations" / "equator_depths.csv", utc, depths,
                     np.zeros((12, 2)))
    try:
        run.depth_series()
    except ValueError as exc:
        assert "cannot be identified" in str(exc)
    else:
        raise AssertionError("a window that cannot be matched must not be guessed")


def test_collapsed_depth_headings_warn(tmp):
    # The file prints depths to three decimals: 0.0021 and 0.0024 both become
    # 'T(z=0.002 m)', and the second column is then unattributable.
    path = tmp / "dup" / "equator_depths.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# Location: equator\n"
                    "time,T(z=0.002 m),T(z=0.002 m)\n"
                    "2026-04-01T00:00:00.000,100.0,101.0\n")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        readers.read_depth_series(path)
    assert any("three decimals" in str(w.message) for w in caught)


# --- 8. rotation counting and the figures ----------------------------------------------

def test_multi_rotation_window_says_so(tmp):
    run = Results(make_run(tmp / "rot_run", n_steps=8))
    assert run.n_rotations() == 1
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run.local_solar_time_curve(n_bins=8, lat_max=30.0)
    assert not caught, "a one-rotation window must not warn"
    assert len(run.rotation_steps()) == 8


def test_both_sweep_figures_draw(tmp):
    import matplotlib
    matplotlib.use("Agg")
    sweep, _ = _run_sweep(tmp, "sweep_figs")

    def curve(results):
        return results.series("temperature", lat=0.0, lon=0.0)

    fig = sweep.plot_curves(of=curve, xlabel="Elapsed time (h)",
                            ylabel="Temperature (K)",
                            legend_title="Albedo",
                            save=tmp / "curves.png")
    assert (tmp / "curves.png").exists()
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert labels == ["0.1", "0.2", "0.3"], (
        "with the axis named in the legend title the entries are bare values")

    obs_x, obs_y = curve(sweep.points[1].results)
    sweep.plot_residuals(obs_x, obs_y + 1.0, obs_sigma=0.5, of=curve,
                         xlabel="Elapsed time (h)", ylabel="Temperature (K)",
                         save=tmp / "residuals.png")
    assert (tmp / "residuals.png").exists()


def test_a_flat_axis_is_visible_in_the_collection(tmp):
    """The collection must make 'the axis did nothing' checkable."""
    sweep, _ = _run_sweep(tmp, "sweep_flat")
    values = sweep.collect(lambda r: float(r.stat("temperature", "max").mean()))
    # The stub writes the same field for every point, so the spread is zero --
    # which is exactly what a sweep whose parameter never reached the solver
    # looks like, and the accessor shows it rather than hiding it.
    assert float(np.ptp(values)) == 0.0


# --- 9. the CLI's grid parsing -----------------------------------------------------------

def test_vary_parsing():
    from ptm.cli import _parse_vary
    grid = _parse_vary(["thermal_inertia=50,100,200",
                        "thermal.equilibration_orbits=1.5,3"])
    assert grid == {"thermal_inertia": [50, 100, 200],
                    "thermal.equilibration_orbits": [1.5, 3]}
    assert list(grid) == ["thermal_inertia", "thermal.equilibration_orbits"], (
        "axis order is the order given, because it sets the grid order")


TESTS = [
    test_flag_detection_is_whole_token,
    test_missing_engine_reports_rather_than_claims_support,
    test_config_route_when_the_flag_is_absent,
    test_flag_route_when_the_engine_has_it,
    test_the_parameter_is_delivered_exactly_once,
    test_thermal_inertia_refuses_rather_than_switching_convention,
    test_the_two_thermal_inertia_conventions_cannot_both_be_swept,
    test_requiring_an_absent_flag_is_an_error,
    test_generated_config_lands_beside_its_outputs,
    test_view_factor_path_is_anchored_to_the_base_config,
    test_fixed_overrides_apply_to_every_point,
    test_derived_keys_come_from_the_callers_own_relation,
    test_an_undeclared_bare_axis_name_is_refused,
    test_grid_is_a_cartesian_product_in_order,
    test_colliding_point_names_are_refused,
    test_a_sweep_runs_and_collects,
    test_resume_skips_only_what_is_genuinely_done,
    test_an_interrupted_point_is_not_complete,
    test_a_changed_base_config_invalidates_a_point,
    test_deleted_outputs_invalidate_a_complete_marker,
    test_a_failing_engine_is_recorded_as_failed,
    test_exit_zero_without_outputs_is_still_a_failure,
    test_dry_run_launches_nothing,
    test_the_manifest_round_trips,
    test_provenance_is_recorded_with_the_point,
    test_a_sweep_is_usable_with_no_engine_at_all,
    test_a_finished_sweep_is_readable_with_no_engine,
    test_a_finished_flag_route_sweep_reopens_without_asking_the_engine,
    test_parallel_refuses_a_gpu_config,
    test_parallel_runs_a_cpu_sweep,
    test_depth_series_round_trips,
    test_depth_series_trims_equilibration_by_matching_the_time_axis,
    test_depth_series_refuses_to_guess_the_window,
    test_collapsed_depth_headings_warn,
    test_multi_rotation_window_says_so,
    test_both_sweep_figures_draw,
    test_a_flat_axis_is_visible_in_the_collection,
    test_vary_parsing,
]


def main():
    tmp = Path(tempfile.mkdtemp(prefix="ptm_sweep_"))
    failures = 0
    try:
        for test in TESTS:
            needs_tmp = test.__code__.co_argcount == 1
            case = tmp / test.__name__
            case.mkdir(parents=True, exist_ok=True)
            try:
                test(case) if needs_tmp else test()
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {test.__name__}\n     {exc}")
            except Exception as exc:                   # noqa: BLE001
                failures += 1
                print(f"ERROR {test.__name__}: {type(exc).__name__}: {exc}")
            else:
                print(f"PASS {test.__name__}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    total = len(TESTS)
    print(f"\n[test_sweep] {total - failures}/{total} passed")
    if failures:
        print("[test_sweep] FAIL")
        return 1
    print("[test_sweep] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
