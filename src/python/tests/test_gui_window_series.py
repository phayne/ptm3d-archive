#!/usr/bin/env python3
"""Breaking a long run into windows, the way the polar landing-site campaign does.

WHAT IS BEING HELD. `gui_core.window_series` plans N short windows across a
long span and emits one complete configuration per window plus the schedule and
the commands to run them. The commands are
``scripts/run_chunked_simulation.py``'s -- the SAME invocation
``a3gt_v2_orchestrator.run_window`` builds -- and the schedule is the shape
``a3gt_v2_subsolar.py`` writes. Nothing here is a new runner, and the arms below
are what keep that true: ``test_the_commands_are_the_orchestrators_own`` reads
the orchestrator's own source and compares flags.

THE ONE DIFFERENCE, held explicitly by
``test_the_schedule_says_its_epochs_are_evenly_spaced``: the polar landing-site campaign's window epochs
are subsolar-latitude extrema found with SPICE; these are evenly spaced. A
schedule that looked like the SPICE one while carrying even spacing would be
the silent substitution this project keeps paying for, so the difference is
written into the file.

``test_every_emitted_window_is_accepted_by_the_engine`` is the arm that matters
most and the one that can SKIP: it hands every emitted configuration to
``ptm_thermal_sim --validate-config``. It skips loudly when there is no such
binary, because an emission gate that silently does not run is not a gate.

Run directly::

    python3 src/python/tests/test_gui_window_series.py
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "gui"))
sys.path.insert(0, str(REPO_ROOT / "src" / "python"))

from gui_core.window_series import (                             # noqa: E402
    MAX_WINDOWS, confirmation_text, plan_windows, runner_commands,
    schedule_document, split_advice, window_configs, write_window_series)

#: A lunar diurnal cycle, the period every count below is in.
SYNODIC = 29.530588


def _base_config(scratch):
    """A minimal integrated lunar run: exactly what the card would hand over."""
    return {
        "simulation": {"body": "Moon", "start_time": "2034-10-30T00:00:00",
                       "duration_days": SYNODIC, "timesteps": 144},
        "mesh": {"resolution": 4},
        "thermal": {"run_thermal_mesh": True, "equilibration_orbits": 6.0,
                    "integration_days": SYNODIC},
        "radiation": {"enabled": False},
        "illumination": {"albedo_source": "none", "use_albedo_map": False},
        "output": {"directory": str(Path(scratch) / "out"),
                   "skip_time_series": True},
    }


# --------------------------------------------------------------------------
# The plan
# --------------------------------------------------------------------------

def test_the_windows_span_the_whole_span():
    """First window at the start, last window ENDING at the end."""
    plan = plan_windows("2034-10-30T00:00:00", 6798.5, 4, 30.0,
                        synodic_days=SYNODIC)
    assert not plan.refusal, plan.refusal
    assert len(plan) == 4
    first = dt.datetime.strptime(plan.windows[0].start_iso, "%Y-%m-%dT%H:%M:%S")
    last_end = dt.datetime.strptime(plan.windows[-1].end_iso(),
                                    "%Y-%m-%dT%H:%M:%S")
    assert first == dt.datetime(2034, 10, 30)
    assert abs((last_end - first).total_seconds() / 86400.0 - 6798.5) < 1e-6
    # Equal intervals, and no overlap.
    starts = [dt.datetime.strptime(w.start_iso, "%Y-%m-%dT%H:%M:%S")
              for w in plan.windows]
    gaps = {round((b - a).total_seconds() / 86400.0, 6)
            for a, b in zip(starts, starts[1:])}
    assert len(gaps) == 1
    assert gaps.pop() > 30.0


def test_one_window_is_the_window_itself():
    plan = plan_windows("2034-10-30T00:00:00", 6798.5, 1, 30.0,
                        synodic_days=SYNODIC)
    assert len(plan) == 1
    assert plan.windows[0].start_iso == "2034-10-30T00:00:00"
    assert plan.windows[0].duration_days == 30.0


def test_windows_that_do_not_fit_are_refused_not_shortened():
    """Overlapping windows would count part of the span twice."""
    plan = plan_windows("2034-10-30T00:00:00", 100.0, 8, 30.0,
                        synodic_days=SYNODIC)
    assert plan.refusal
    assert "would have to overlap" in plan.refusal
    assert plan.windows == []


def test_a_bad_start_time_is_refused_by_name():
    plan = plan_windows("last tuesday", 100.0, 2, 10.0, synodic_days=SYNODIC)
    assert "is not an ISO-8601 timestamp" in plan.refusal


def test_too_many_windows_points_at_the_orchestrator():
    plan = plan_windows("2034-10-30T00:00:00", 1e6, MAX_WINDOWS + 1, 1.0,
                        synodic_days=SYNODIC)
    assert "a3gt_v2_orchestrator.py" in plan.refusal


def test_a_zero_length_window_is_refused():
    plan = plan_windows("2034-10-30T00:00:00", 100.0, 2, 0.0,
                        synodic_days=SYNODIC)
    assert "positive length" in plan.refusal


def test_the_sampling_cadence_follows_the_diurnal_period():
    """144 samples per synodic day, the corpus cadence, per window."""
    plan = plan_windows("2034-10-30T00:00:00", 6798.5, 4, SYNODIC,
                        synodic_days=SYNODIC)
    assert {w.timesteps for w in plan.windows} == {144}
    half = plan_windows("2034-10-30T00:00:00", 6798.5, 4, SYNODIC / 2.0,
                        synodic_days=SYNODIC)
    assert {w.timesteps for w in half.windows} == {72}


def test_a_body_with_no_period_is_told_so_rather_than_guessed_at():
    plan = plan_windows("2034-10-30T00:00:00", 1000.0, 2, 10.0,
                        synodic_days=0.0)
    assert not plan.refusal
    assert any("no known diurnal period" in note for note in plan.notes)


# --------------------------------------------------------------------------
# What it emits
# --------------------------------------------------------------------------

def test_each_window_config_is_the_base_with_five_keys_moved():
    with tempfile.TemporaryDirectory() as scratch:
        base = _base_config(scratch)
        plan = plan_windows("2034-10-30T00:00:00", 6798.5, 4, 30.0,
                            synodic_days=SYNODIC)
        emitted = window_configs(base, plan, Path(scratch) / "series")
    assert len(emitted) == 4
    for (tag, config), window in zip(emitted, plan.windows):
        assert config["simulation"]["start_time"] == window.start_iso
        assert config["simulation"]["duration_days"] == 30.0
        assert config["thermal"]["integration_days"] == 30.0
        assert config["simulation"]["timesteps"] == window.timesteps
        assert config["output"]["directory"].endswith(f"/{tag}")
        # Everything else is the base's, untouched. Same physics at a
        # different epoch is the whole point of the set.
        assert config["thermal"]["equilibration_orbits"] == 6.0
        assert config["mesh"] == base["mesh"]
        assert config["radiation"] == base["radiation"]
        assert config["output"]["skip_time_series"] is True


def test_the_base_configuration_is_not_mutated():
    """A card hands over its LIVE configuration. Emitting must not edit it."""
    import copy
    with tempfile.TemporaryDirectory() as scratch:
        base = _base_config(scratch)
        before = copy.deepcopy(base)
        plan = plan_windows("2034-10-30T00:00:00", 6798.5, 4, 30.0,
                            synodic_days=SYNODIC)
        window_configs(base, plan, Path(scratch) / "series")
    assert base == before


def test_the_output_window_and_the_integration_window_agree():
    """The engine simulates max(output, offset + integration).

    They have to be equal, or the run integrates over a span it did not
    simulate -- or simulates one it does not use.
    """
    with tempfile.TemporaryDirectory() as scratch:
        plan = plan_windows("2034-10-30T00:00:00", 500.0, 3, 30.0,
                            synodic_days=SYNODIC)
        for _tag, config in window_configs(plan and _base_config(scratch),
                                           plan, scratch):
            assert (config["simulation"]["duration_days"]
                    == config["thermal"]["integration_days"])


def test_the_schedule_says_its_epochs_are_evenly_spaced():
    plan = plan_windows("2034-10-30T00:00:00", 6798.5, 4, 30.0,
                        synodic_days=SYNODIC)
    document = schedule_document(plan, body="Moon", output_root="/tmp/x")
    # The polar-site schedule's own keys, so anything reading that file reads this.
    for key in ("start_time", "total_days", "windows"):
        assert key in document
    for window in document["windows"]:
        for key in ("tag", "start_iso", "duration_days"):
            assert key in window
    # ...and the difference from it, in the file rather than only in a docstring.
    assert document["epoch_rule"] == "even"
    assert "a3gt_v2_subsolar.py" in document["epoch_rule_detail"]
    assert "NOT the sub-solar-latitude rule" in document["epoch_rule_detail"]


def test_the_commands_are_the_orchestrators_own():
    """Not a new runner: the same flags a3gt_v2_orchestrator passes.

    Read out of the orchestrator's source rather than retyped, so a flag that
    changes there shows up here as a failure instead of as a GUI that emits a
    command the runner no longer accepts.
    """
    orchestrator = (REPO_ROOT / "scripts" / "a3gt_v2_orchestrator.py").read_text()
    plan = plan_windows("2034-10-30T00:00:00", 6798.5, 2, 30.0,
                        synodic_days=SYNODIC)
    commands = runner_commands(plan, "/tmp/series")
    assert len(commands) == 2
    for command in commands:
        assert command[1].endswith("run_chunked_simulation.py")
        for flag in ("--config", "--total-days", "--output-dir", "--resume"):
            assert flag in command, flag
            assert flag in orchestrator, (
                f"{flag} is not in the orchestrator's own invocation any more")
    assert "run_chunked_simulation.py" in orchestrator


def test_the_total_days_flag_carries_the_windows_full_precision():
    """The orchestrator's own ``%g`` is six figures; a diurnal window is not.

    polar-site windows are whole days, so ``%g`` is exact there. 29.530588 d is not:
    ``%g`` gives 29.5306, and the chunked runner divides ``--total-days`` into
    chunks while the configuration keeps its own ``duration_days``. The two
    would then describe runs a second apart.
    """
    plan = plan_windows("2034-10-30T00:00:00", 6798.5, 2, SYNODIC,
                        synodic_days=SYNODIC)
    command = runner_commands(plan, "/tmp/series")[0]
    total = float(command[command.index("--total-days") + 1])
    assert total == SYNODIC, f"{total!r} is not the window length {SYNODIC!r}"


def test_the_confirmation_names_the_run_count_and_the_wall_clock():
    plan = plan_windows("2034-10-30T00:00:00", 6798.5, 8, 30.0,
                        synodic_days=SYNODIC)
    text = confirmation_text(plan, "/tmp/series", minutes_per_window=25.0)
    assert "8 window configurations" in text
    assert "roughly 3 hours" in text
    # And the one thing about a window series that is not obvious.
    assert "not a continuous integration" in text
    assert "written, not launched" in text


def test_writing_a_series_leaves_configs_and_a_schedule():
    with tempfile.TemporaryDirectory() as scratch:
        base = _base_config(scratch)
        plan = plan_windows("2034-10-30T00:00:00", 6798.5, 3, 30.0,
                            synodic_days=SYNODIC)
        root = Path(scratch) / "series"
        written = write_window_series(base, plan, root)
        assert len(written) == 3
        for path in written:
            assert path.exists()
            document = yaml.safe_load(path.read_text())
            assert document["simulation"]["body"] == "Moon"
        schedule = json.loads((root / "windows.json").read_text())
        assert len(schedule["windows"]) == 3
        assert schedule["body"] == "Moon"


def test_the_split_advice_fires_only_on_a_long_window():
    assert split_advice(SYNODIC, SYNODIC) == ""
    assert split_advice(SYNODIC * 50, SYNODIC) == ""
    advice = split_advice(SYNODIC * 200, SYNODIC)
    assert "200 diurnal cycles" in advice
    assert "polar-site campaign" in advice
    # No period, no advice -- and no crash.
    assert split_advice(SYNODIC * 200, 0.0) == ""
    assert split_advice("nonsense", SYNODIC) == ""


# --------------------------------------------------------------------------
# The gate: the engine's own parser reads every window
# --------------------------------------------------------------------------

def _engine():
    from gui_core.simulation_runner import SimulationRunner
    exe = SimulationRunner.find_ptm_executable()
    if not exe:
        return None
    try:
        help_text = subprocess.run([str(exe), "--help"], capture_output=True,
                                   text=True, timeout=60).stdout
    except Exception:                                        # noqa: BLE001
        return None
    # A binary predating the flag has no unknown-argument handler and RUNS THE
    # SIMULATION when handed it. Probe, never assume.
    return str(exe) if "--validate-config" in help_text else None


def test_every_emitted_window_is_accepted_by_the_engine():
    exe = _engine()
    if exe is None:
        pytest.skip("no ptm_thermal_sim with --validate-config to ask. THE "
                    "EMISSION GATE DID NOT RUN; build it with: cmake --build "
                    "build --target ptm_thermal_sim")
    with tempfile.TemporaryDirectory() as scratch:
        base = _base_config(scratch)
        plan = plan_windows("2034-10-30T00:00:00", 6798.5, 4, 30.0,
                            synodic_days=SYNODIC)
        root = Path(scratch) / "series"
        written = write_window_series(base, plan, root)
        failures = []
        for path in written:
            result = subprocess.run(
                [exe, "--config", str(path), "--validate-config"],
                capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=180)
            if result.returncode != 0:
                tail = (result.stdout + result.stderr).strip().splitlines()
                failures.append(f"{path.parent.name}: rc="
                                f"{result.returncode} :: "
                                + " | ".join(tail[-3:]))
        print(f"    [engine] {len(written) - len(failures)}/{len(written)} "
              f"window configurations accepted by {Path(exe).name}")
        assert not failures, "\n  ".join(
            ["the engine refused a window configuration:"] + failures)


if __name__ == "__main__":                                   # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
