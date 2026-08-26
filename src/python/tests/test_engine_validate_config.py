#!/usr/bin/env python3
"""``ptm_thermal_sim --validate-config``: parse and check, then exit.

The GUI's "Check configuration" hands a config to the engine and reports the
ENGINE's own text, so it stops maintaining a second, weaker copy of rules that
then disagrees with the engine (docs/design/gui-update-plan.md item 2.4, DP-6).
That is only worth anything if the mode really is what it claims:

* it accepts a configuration the engine would accept,
* it refuses -- with the engine's own message and rc 1 -- every configuration
  class the engine refuses at startup,
* it does **not** simulate, and does not touch the output directory the config
  names, and
* it says what it did **not** check, because rc 0 here is not "this run will
  complete".

Needs a built ``ptm_thermal_sim``; SKIPS (rc 0, loudly) when there is none,
because a missing binary is a build state, not a defect in this behavior.

Run directly::

    python3 src/python/tests/test_engine_validate_config.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "python"))

from ptm import paths  # noqa: E402


class Skip(Exception):
    pass


def _engine():
    exe = paths.find_engine_executable("ptm_thermal_sim", search_roots=[REPO_ROOT])
    if exe is None:
        raise Skip(
            "no ptm_thermal_sim in build/bin (nor $PTM_BUILD_DIR); build it "
            "with: cmake --build build --target ptm_thermal_sim")
    return Path(exe).resolve()


BASE = """\
simulation:
  start_time: '2025-01-01T00:00:00'
  duration_days: 1.0
  timesteps: 4
  body: Moon
mesh:
  resolution: 2
thermal:
  run_thermal_mesh: true
  use_gpu: false
output:
  directory: '{outdir}'
  format: binary
"""


def _write(tmp, name, extra=""):
    outdir = Path(tmp) / ("out_" + name)
    path = Path(tmp) / (name + ".yaml")
    path.write_text(BASE.format(outdir=outdir) + extra)
    return path, outdir


def _run(exe, config, timeout=120):
    env = dict(os.environ)
    env.setdefault("PTM_DATA_ROOT", str(REPO_ROOT / "data"))
    t0 = time.monotonic()
    proc = subprocess.run(
        [str(exe), "--config", str(config), "--validate-config"],
        capture_output=True, text=True, timeout=timeout, cwd=str(REPO_ROOT),
        env=env)
    return proc, time.monotonic() - t0


# --------------------------------------------------------------------------

def test_a_valid_config_is_accepted():
    exe = _engine()
    with tempfile.TemporaryDirectory() as tmp:
        config, _ = _write(tmp, "good")
        proc, _ = _run(exe, config)
        assert proc.returncode == 0, (
            f"rc={proc.returncode} on a config with no defects\n"
            f"{proc.stdout[-1500:]}{proc.stderr[-1500:]}")
        assert "[validate-config] ACCEPTED" in proc.stdout, (
            "the acceptance banner is the machine-readable half of the "
            "contract; without it a caller can only guess from rc")


def test_it_discloses_what_it_did_not_check():
    """rc 0 means well formed, NOT that the run will start. Say so."""
    exe = _engine()
    with tempfile.TemporaryDirectory() as tmp:
        config, _ = _write(tmp, "good")
        proc, _ = _run(exe, config)
        assert "NOT checked" in proc.stdout, (
            "the scope note is missing. A caller that reads rc=0 as 'this run "
            "will start' would be wrong -- mesh, SPICE and generic-body "
            "construction all happen after this point.")
        for phrase in ("mesh", "SPICE", "martinez2021"):
            assert phrase in proc.stdout, (
                f"the scope note no longer names {phrase!r}")
        # These two need nothing but the config and are still NOT checked --
        # they live below this exit inside a block that also reads their
        # environment overrides, and hoisting them would mean a second copy of
        # the rule. So the banner has to say so, or the block's own claim to
        # check "everything knowable from the configuration alone" is false.
        # Verified against the binary: thermal.eq_indirect_passes: 0 is
        # ACCEPTED here and FATAL in a real run.
        for phrase in ("eq_indirect_source", "eq_indirect_passes"):
            assert phrase in proc.stdout, (
                f"the scope note does not admit that {phrase} is unchecked; "
                f"a caller would read ACCEPTED for a config the engine "
                f"refuses at startup")


def test_the_two_config_only_checks_below_the_exit_are_named_not_hidden():
    """The honest half of the scope boundary.

    `thermal.eq_indirect_passes: 0` is a startup FATAL that needs nothing but
    the config, and this mode accepts it -- because the check lives below the
    exit, inside a block that also reads PTM_EQ_INDIRECT_PASSES. Measured both
    ways. The mode is allowed not to check it; it is not allowed to be quiet
    about not checking it, and the GUI carries its own copy of both rules.
    """
    exe = _engine()
    with tempfile.TemporaryDirectory() as tmp:
        config, _ = _write(tmp, "eqpasses", "thermal:\n  eq_indirect_passes: 0\n")
        proc, _ = _run(exe, config)
        assert proc.returncode == 0, (
            "this asserts the CURRENT boundary; if the exit moved below the "
            "check, update the banner and this test together")
        assert "eq_indirect_passes" in proc.stdout, (
            "accepted a config the engine refuses at startup WITHOUT naming "
            "the key in the not-checked list")


def test_engine_refusals_are_reproduced_verbatim():
    """Each class the engine refuses at startup must refuse here too."""
    exe = _engine()
    cases = [
        ("beam_no_slope",
         "radiation:\n  beaming: true\n  rms_slope_deg: 0.0\n",
         "radiation.beaming = true requires"),
        ("bad_conductivity",
         "thermophysical:\n  conductivity_model: hayne2018\n",
         "hayne2018"),
        ("bad_coupling_mode",
         "radiation:\n  enabled: true\n  coupling_mode: bounded_v2\n",
         "bounded_v2"),
        ("bad_gpu_policy",
         "thermal:\n  use_gpu: true\ngpu:\n  placement_policy: sometimes\n",
         "placement_policy"),
        ("bad_quadrature_order",
         "radiation:\n  enabled: true\n  nearfield_quadrature_order: 0\n",
         "nearfield_quadrature_order"),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        for name, extra, phrase in cases:
            config, _ = _write(tmp, name, extra)
            proc, _ = _run(exe, config)
            assert proc.returncode == 1, (
                f"{name}: rc={proc.returncode}, expected 1. The engine refuses "
                f"this at startup; a validator that accepts it is worse than "
                f"no validator.\n{proc.stdout[-800:]}{proc.stderr[-800:]}")
            assert phrase in (proc.stdout + proc.stderr), (
                f"{name}: the refusal no longer contains {phrase!r}, so the "
                f"caller cannot show the engine's own words")


def test_malformed_yaml_is_refused_not_defaulted():
    """PTM-66: a rejected config must never fall back to built-in defaults."""
    exe = _engine()
    with tempfile.TemporaryDirectory() as tmp:
        config = Path(tmp) / "broken.yaml"
        config.write_text("simulation:\n  body: Moon\n   timesteps: [\n")
        proc, _ = _run(exe, config)
        assert proc.returncode == 1, (
            f"rc={proc.returncode}: malformed YAML validated as acceptable")


def test_it_needs_a_config_to_validate():
    exe = _engine()
    proc = subprocess.run([str(exe), "--validate-config"], capture_output=True,
                          text=True, cwd=str(REPO_ROOT), timeout=60)
    assert proc.returncode == 1, (
        "--validate-config with no --config reported success about a file it "
        "never read")


def test_nothing_is_simulated_and_nothing_is_written():
    exe = _engine()
    with tempfile.TemporaryDirectory() as tmp:
        # (a) a fresh output directory is not created
        config, outdir = _write(tmp, "noside")
        if outdir.exists():
            shutil.rmtree(outdir)
        proc, elapsed = _run(exe, config)
        assert proc.returncode == 0
        assert not outdir.exists(), (
            f"--validate-config created {outdir}; checking a config must not "
            f"leave a directory behind")
        assert elapsed < 30.0, (
            f"took {elapsed:.1f} s -- long enough that it may be doing real "
            f"work rather than parsing")

        # (b) an EXISTING output directory is not disturbed. The heartbeat is
        # the specific hazard: it is written from the config's output.directory
        # at startup, so validating a finished run's config would erase the
        # record of how that run ended.
        config, outdir = _write(tmp, "existing")
        outdir.mkdir(parents=True, exist_ok=True)
        marker = outdir / "heartbeat.txt"
        marker.write_text("PRIOR RUN MARKER\n")
        before = sorted(p.name for p in outdir.iterdir())
        _run(exe, config)
        assert sorted(p.name for p in outdir.iterdir()) == before, (
            "validating added or removed files in an existing output directory")
        assert marker.read_text() == "PRIOR RUN MARKER\n", (
            "validating overwrote heartbeat.txt in an existing output directory")


TESTS = [
    test_a_valid_config_is_accepted,
    test_it_discloses_what_it_did_not_check,
    test_the_two_config_only_checks_below_the_exit_are_named_not_hidden,
    test_engine_refusals_are_reproduced_verbatim,
    test_malformed_yaml_is_refused_not_defaulted,
    test_it_needs_a_config_to_validate,
    test_nothing_is_simulated_and_nothing_is_written,
]


def main():
    failures = 0
    skipped = 0
    for test in TESTS:
        try:
            test()
        except Skip as exc:
            skipped += 1
            print(f"SKIP {test.__name__}: {exc}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {test.__name__}\n     {exc}")
        except Exception as exc:                       # noqa: BLE001
            failures += 1
            print(f"ERROR {test.__name__}: {type(exc).__name__}: {exc}")
        else:
            print(f"PASS {test.__name__}")
    total = len(TESTS)
    print(f"\n[test_engine_validate_config] {total - failures - skipped}/{total} "
          f"passed, {skipped} skipped")
    if skipped == total:
        print("[test_engine_validate_config] SKIPPED -- no engine binary; "
              "this gate proved NOTHING on this machine.")
    if failures:
        print("[test_engine_validate_config] FAIL")
        return 1
    print("[test_engine_validate_config] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
