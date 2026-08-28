#!/usr/bin/env python3
"""The curated Bennu flow has to produce a configuration that RUNS.

**Why this file exists, and why the file next to it did not catch this.**

``test_gui_curated_sources.py`` proved the curated tier end to end at the
``gui_core.body_fetch`` layer: ``fetch_spk(...)`` returns
``generic_body.spice_metakernel: 'bennu.tm'`` and ``ephemeris_source: 'spice'``,
never reaches HORIZONS, and opens no socket. Every one of those checks passed.
Its two Qt arms press nothing -- they set a name and assert on the *note text*
and the *button labels*. So the suite covered:

    fetch_spk() -> values dict

and covered NOTHING of:

    button click -> card state -> ConfigManager -> emitted YAML -> engine

A click-through on 2026-08-28 failed in that uncovered span, with::

    Error: Invalid generic body configuration:
    SPICE ephemeris source requires a metakernel for body
    '101955 Bennu (1999 RQ36)'

and the emitted ``output/bennu_test/config.yaml`` carried
``ephemeris_source: spice`` with **no** ``spice_metakernel`` key at all. The
same day's passing acceptance run was a HAND-WRITTEN configuration
(``examples/quickstart_asteroid_bennu.yaml``, ``generic_body.name: Bennu``, no
``orbit:`` block) handed straight to the engine -- so it exercised the engine
and the kernel, and not one line of the GUI's emission path.

Three separate defects live in that span, all measured, and this file pins all
three plus the fix for blank surface properties:

1. **A SPICE selection that carries no kernel.** The card's ephemeris radio
   writes ``generic_body.ephemeris_source`` and nothing else;
   ``_strip_sentinels`` then drops the empty ``spice_metakernel`` on the way to
   YAML, so the emitted file names a SPICE source with no kernel and the engine
   refuses it (``generic_body.h:94``) AFTER building the mesh.
   ``--validate-config`` does not reach that check -- its own banner lists
   "generic-body construction" under NOT checked -- so nothing caught it.
   Driving the lookup's values, the curated SHAPE button and this radio
   reproduced the failing file on 24 of 24 keys.
2. **A bare kernel name absolutized into nothing.**
   ``absolutize_config_paths`` treated ``generic_body.spice_metakernel`` as an
   ordinary data path and rewrote ``bennu.tm`` to ``<project_root>/bennu.tm``,
   a file that exists nowhere. The engine resolves a BARE name against the
   SPICE kernel directory (``illumination_model.cpp:2684``), so this broke a
   correctly wired card: pre-flight reported the fabricated path missing and
   the engine died with "SPICE meta-kernel not found". Measured -- and it would
   have bitten the moment defect 1 was fixed on its own.
3. **A fetch result dropped in silence.** ``_on_fetch_done`` discards a result
   that lands after the card stops being active, at ``logger.info`` and with no
   mark on screen, while ``_on_fetch_spk`` has already written "Wiring <body>
   to <ephemeris>..." into that same note -- a present-tense claim standing
   over a result that was thrown away.

And the fourth item, which is why a user was editing the card by hand at all:
``generic_body.surface.*`` had no defaults anywhere in the GUI, so every box
read 0.0 while the engine would have used
``GenericBody::defaultRockyAsteroid()``. Three of the ten properties (H, chi,
Qb) had no control on any page.

Run directly::

    python3 src/python/tests/test_gui_curated_run_emission.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "python"))
sys.path.insert(0, str(REPO_ROOT / "gui"))

BENNU = "101955 Bennu (1999 RQ36)"
#: The form the curated registry writes and `docs/configuration.md` documents:
#: a BARE name, resolved by the engine against the SPICE kernel directory.
BARE_METAKERNEL = "bennu.tm"


class Skip(Exception):
    """A check that cannot run here, as distinct from one that failed."""


# ---------------------------------------------------------------------------
# Qt harness
# ---------------------------------------------------------------------------

def _app():
    """An offscreen QApplication, or a Skip that names what is missing.

    The plugin path is set explicitly when the environment blanks it. A shell
    exporting an EMPTY ``QT_QPA_PLATFORM_PLUGIN_PATH`` defeats PyQt6's own
    discovery, and Qt answers that with ``qFatal`` -- the process aborts with
    134 before a single assertion runs, which reads as a failing suite rather
    than as a machine that cannot host one.
    """
    try:
        import PyQt6
        from PyQt6.QtWidgets import QApplication
    except ImportError as exc:
        raise Skip(f"PyQt6 is not available here ({exc})")
    if not os.environ.get("QT_QPA_PLATFORM"):
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
    if not os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH"):
        plugins = (Path(PyQt6.__file__).parent / "Qt6" / "plugins"
                   / "platforms")
        if plugins.is_dir():
            os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(plugins)
    return QApplication.instance() or QApplication([])


def _pump(app, seconds, until=None):
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()
        if until is not None and until():
            return True
        time.sleep(0.02)
    return until() if until is not None else False


def _open_asteroid_card(app):
    """The body-first flow, driven the way a user drives it.

    Landing screen -> Asteroid -> run type, through the panel's own methods,
    so the route defaults (and the surface seeding that hangs off them) run
    exactly as they do on a click.
    """
    from gui_core.config_manager import ConfigManager
    from gui_panels.config_panel import ConfigurationPanel

    manager = ConfigManager()
    panel = ConfigurationPanel(manager)
    start = panel.tabs["start"]
    start._project_root = lambda: REPO_ROOT
    start.choose_body("asteroid")
    start.choose_run_type("time_series")
    card = start.cards["asteroid"]
    card._project_root = lambda: REPO_ROOT
    app.processEvents()
    return manager, panel, start, card


def _name_the_body(app, manager, card):
    """Type the name, as the Name field receives it."""
    card.widgets["generic_body.name"].setText(BENNU)
    app.processEvents()


def _press_curated_ephemeris(app, card):
    """Press the real button and wait for the worker to land.

    Not ``fetch_spk(...)``. The gap this file exists to close is exactly the
    distance between that call and this click.
    """
    from PyQt6.QtWidgets import QMessageBox
    QMessageBox.question = staticmethod(
        lambda *a, **k: QMessageBox.StandardButton.Yes)
    label = card.spk_fetch_button.text()
    if "solution 118" not in label:
        raise AssertionError(
            f"the ephemeris button does not offer the curated source for "
            f"Bennu; it reads {label!r}")
    card.spk_fetch_button.click()
    landed = _pump(app, 180.0, until=lambda: (
        getattr(card, "_fetch_worker_spk", None) is None
        and card.coverage_note.text() != ""))
    if not landed:
        raise Skip("the curated ephemeris fetch did not settle in 180 s")
    return label


def _kernels_present():
    from gui_core.curated_sources import CURATED_BODIES, curated_kernel_status
    status = curated_kernel_status(CURATED_BODIES[0], REPO_ROOT)
    return status.ready, status.summary()


def _engine():
    """The engine binary, or a Skip. ``PTM_ENGINE`` overrides."""
    explicit = os.environ.get("PTM_ENGINE", "").strip()
    candidates = [Path(explicit)] if explicit else []
    candidates.append(REPO_ROOT / "build" / "bin" / "ptm_thermal_sim")
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    raise Skip("no ptm_thermal_sim; build it or set PTM_ENGINE")


# ---------------------------------------------------------------------------
# 1. The click path emits a runnable configuration
# ---------------------------------------------------------------------------

def test_the_curated_button_writes_the_kernel_into_the_emitted_config():
    """Press the button; the RUN YAML must carry both keys.

    Both, together, is the whole point: ``ephemeris_source: spice`` on its own
    is the configuration the engine refuses, and it is what a click-through
    produced.
    """
    app = _app()
    ready, summary = _kernels_present()
    if not ready:
        raise Skip(f"Bennu's kernels are not on this disk: {summary}")

    import yaml

    manager, _panel, _start, card = _open_asteroid_card(app)
    _name_the_body(app, manager, card)
    label = _press_curated_ephemeris(app, card)

    gb = manager.get_config().get("generic_body") or {}
    assert gb.get("ephemeris_source") == "spice", (
        f"pressing {label!r} left generic_body.ephemeris_source = "
        f"{gb.get('ephemeris_source')!r}")
    assert gb.get("spice_metakernel") == BARE_METAKERNEL, (
        f"pressing {label!r} did not write the metakernel into card state; "
        f"generic_body.spice_metakernel = {gb.get('spice_metakernel')!r}. "
        f"This is the defect: the card SHOWED the ephemeris in its note and "
        f"never wrote the key.")

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "config.yaml"
        manager.save_run_yaml(str(out), REPO_ROOT)
        emitted = (yaml.safe_load(out.read_text()) or {}).get(
            "generic_body") or {}
    assert emitted.get("ephemeris_source") == "spice", emitted
    assert "spice_metakernel" in emitted, (
        "the emitted run YAML has NO spice_metakernel key beside "
        "ephemeris_source: spice -- an empty string is stripped on the way "
        "out, which is exactly how the engine came to be handed a SPICE "
        "source with no kernel")
    written = emitted["spice_metakernel"]
    assert Path(written).name == BARE_METAKERNEL, written
    print(f"    [emit] pressing {label!r} emits ephemeris_source=spice with "
          f"spice_metakernel={written!r}")


def test_a_bare_kernel_name_survives_the_run_yaml_intact():
    """`bennu.tm` must NOT become `<project_root>/bennu.tm`.

    The engine's own resolution for this key is: absolute is used as given, a
    bare name is looked for under the SPICE kernel directory and then the
    working directory (illumination_model.cpp:2681-2696). Absolutizing it
    against the project root produces a path that satisfies neither branch,
    and the engine then prints "SPICE meta-kernel not found" listing that one
    invented path. Measured on the click path 2026-08-28.
    """
    from gui_core.preflight import (absolutize_config_paths,
                                    metakernel_candidates,
                                    resolve_metakernel_path)

    config = {"generic_body": {"spice_metakernel": BARE_METAKERNEL}}
    with tempfile.TemporaryDirectory() as tmp:
        # A root with no kernel directory: nothing resolves, so nothing may be
        # rewritten -- the engine's own search and its own message must be what
        # the user meets, not a path this layer made up.
        out = absolutize_config_paths(config, tmp)
        assert out["generic_body"]["spice_metakernel"] == BARE_METAKERNEL, (
            f"an unresolvable bare kernel name was rewritten to "
            f"{out['generic_body']['spice_metakernel']!r}")
        assert resolve_metakernel_path(BARE_METAKERNEL, tmp) is None
        tried = metakernel_candidates(BARE_METAKERNEL, tmp)
        assert len(tried) == 2, tried
        assert str(tried[0]).endswith("data/kernels/" + BARE_METAKERNEL), tried

    # And where it DOES resolve, the emitted path is the real file.
    ready, _summary = _kernels_present()
    if ready:
        out = absolutize_config_paths(config, REPO_ROOT)
        written = Path(out["generic_body"]["spice_metakernel"])
        assert written.is_absolute() and written.exists(), written
    print(f"    [paths] a bare {BARE_METAKERNEL!r} survives absolutization "
          f"verbatim when it cannot be resolved, and resolves to the kernel "
          f"directory when it can")


def test_the_emitted_config_is_accepted_and_the_engine_runs_it():
    """The two gates the brief asks for, in order: --validate-config, then a
    real short run whose log names the kernel.

    Both are needed and neither substitutes for the other:
    ``--validate-config`` prints its own list of what it does NOT check, and
    "generic-body construction, SPICE kernels" is on it -- so the exact defect
    this file is about passes that mode with rc=0. Only a run reaches
    ``GenericBody::validate``.
    """
    app = _app()
    ready, summary = _kernels_present()
    if not ready:
        raise Skip(f"Bennu's kernels are not on this disk: {summary}")
    engine = _engine()

    import yaml

    manager, _panel, _start, card = _open_asteroid_card(app)
    _name_the_body(app, manager, card)
    _press_curated_ephemeris(app, card)
    # A shape and an orbit, so the body is complete. The curated shape button
    # is the card's own control for it.
    card.damit_button.click()
    _pump(app, 120.0, until=lambda: (
        getattr(card, "_fetch_worker_damit_shape", None) is None
        and card.shape_provenance.text() != ""))
    manager.set_value("generic_body.orbit.semi_major_axis_au", 1.1289220453)
    manager.set_value("generic_body.orbit.eccentricity", 0.2046513239)

    ok, errors = manager.validate()
    assert ok, f"the GUI refused its own configuration: {errors}"

    with tempfile.TemporaryDirectory() as tmp:
        manager.set_value("output.directory", str(Path(tmp) / "out"))
        manager.set_value("simulation.timesteps", 2)
        manager.set_value("simulation.duration_days", 0.2)
        manager.set_value("thermal.equilibration_orbits", 0.0)
        config_path = Path(tmp) / "config.yaml"
        manager.save_run_yaml(str(config_path), REPO_ROOT)

        check = subprocess.run(
            [str(engine), "--validate-config", "--config", str(config_path)],
            capture_output=True, text=True, timeout=120)
        assert check.returncode == 0, (
            f"--validate-config rc={check.returncode}\n{check.stdout}\n"
            f"{check.stderr}")

        run = subprocess.run(
            [str(engine), "--config", str(config_path)],
            capture_output=True, text=True, timeout=900, cwd=str(REPO_ROOT))
        log = run.stdout + run.stderr
        assert "Invalid generic body configuration" not in log, (
            "the engine refused the generic body:\n"
            + "\n".join(line for line in log.splitlines()
                        if "Invalid generic body" in line))
        assert run.returncode == 0, (
            f"the engine exited {run.returncode}\n{log[-4000:]}")
        kernel_lines = [line for line in log.splitlines()
                        if "Metakernel:" in line]
        assert kernel_lines, (
            "the run log never named a metakernel; grep 'Metakernel:'\n"
            + log[-3000:])
        assert any(BARE_METAKERNEL in line for line in kernel_lines), (
            f"the run used a different kernel: {kernel_lines}")
    print(f"    [engine] --validate-config rc=0 and a 2-step run rc=0; "
          f"log says {kernel_lines[0].strip()!r}")


# ---------------------------------------------------------------------------
# 2. The half-selected SPICE source, which is what actually bit
# ---------------------------------------------------------------------------

def test_selecting_spice_without_a_kernel_is_refused_before_the_engine():
    """The exact click-path failure, caught by the GUI instead of the solver.

    The ephemeris radio writes ONE key. Everything else about the card can be
    right -- the name, the NAIF id, the orbit, the curated shape and its spin,
    a hand-typed surface -- and this one control still produces the
    configuration the engine throws out after building the mesh. Reproduced
    against ``output/bennu_test/config.yaml``, 24 of 24 keys identical.
    """
    app = _app()
    manager, _panel, _start, card = _open_asteroid_card(app)
    _name_the_body(app, manager, card)
    manager.set_value("generic_body.orbit.semi_major_axis_au", 1.1289220453)

    card.spice_radio.click()
    app.processEvents()
    assert manager.get_value("generic_body.ephemeris_source") == "spice"
    assert not manager.get_value("generic_body.spice_metakernel"), (
        "this arm is only meaningful while the kernel field is empty")

    note = card.coverage_note.text()
    assert "NO METAKERNEL" in note, (
        f"choosing SPICE with no kernel said nothing about it: {note!r}")
    assert card.spk_fetch_button.text() in note, (
        "the refusal does not name the control that fixes it")

    ok, errors = manager.validate()
    assert not ok, (
        "the GUI accepted a SPICE ephemeris with no metakernel -- the engine "
        "refuses this outright, so accepting it here only moves the failure "
        "to after the mesh is built")
    assert any("metakernel" in e.lower() for e in errors), errors
    print("    [refusal] SPICE with no kernel is refused by the card AND by "
          "validate(), naming the button that fixes it")


def test_a_fetch_that_lands_on_a_departed_card_says_so():
    """A discarded result must not leave "Wiring ..." standing over nothing.

    The card's note is the only place a user learns what a button did. Before
    this, a result that arrived after they navigated away was dropped at
    ``logger.info`` with the in-progress message left on screen -- so the card
    claimed the wiring while the configuration never got the key.
    """
    app = _app()
    ready, summary = _kernels_present()
    if not ready:
        raise Skip(f"Bennu's kernels are not on this disk: {summary}")

    manager, _panel, _start, card = _open_asteroid_card(app)
    _name_the_body(app, manager, card)
    card.spk_fetch_button.click()
    card.deactivate()                     # the user moves on mid-fetch
    _pump(app, 180.0,
          until=lambda: getattr(card, "_fetch_worker_spk", None) is None)

    assert not manager.get_value("generic_body.spice_metakernel"), (
        "a result was applied to a card that is no longer on screen")
    note = card.coverage_note.text()
    assert "NOT APPLIED" in note, (
        f"a discarded fetch left the card claiming it acted: {note!r}")
    print("    [discard] a result that lands on a departed card says NOT "
          "APPLIED instead of leaving the in-progress message up")


# ---------------------------------------------------------------------------
# 3. Surface properties: the engine's own defaults, visible and editable
# ---------------------------------------------------------------------------

def test_the_card_seeds_every_surface_property_from_the_engine():
    """Opening the card for a new run fills all ten, with the engine's values.

    Ten, not seven: H, chi and Qb had no control on any page of this
    application while being keys the engine reads.
    """
    app = _app()
    from ptm.body_capabilities import (GENERIC_BODY_SURFACE_DEFAULTS,
                                       GENERIC_SURFACE_FIELDS)

    manager, _panel, _start, card = _open_asteroid_card(app)
    for field in GENERIC_SURFACE_FIELDS:
        key = f"generic_body.surface.{field}"
        assert key in card.widgets, (
            f"{key} has no control on the card, so a user cannot see or "
            f"change a property the engine reads")
        got = manager.get_value(key, None)
        assert got is not None, f"{key} was left unset when the card opened"
        assert got == GENERIC_BODY_SURFACE_DEFAULTS[field], (
            f"{key} seeded {got!r}, engine default is "
            f"{GENERIC_BODY_SURFACE_DEFAULTS[field]!r}")

    line = card.surface_provenance.text()
    assert "engine" in line and "default" in line, line
    for field in ("ks", "rhos", "cp0"):
        assert field in line, (
            f"the provenance line does not name {field}: {line!r}")
    print(f"    [defaults] all {len(GENERIC_SURFACE_FIELDS)} surface "
          f"properties seeded from GenericBody::defaultRockyAsteroid and "
          f"labeled 'engine default'")


def test_a_seeded_field_stops_claiming_the_engine_once_it_is_edited():
    """The label is a claim about where a number came from."""
    app = _app()
    manager, _panel, _start, card = _open_asteroid_card(app)
    assert "ks" in card.surface_provenance.text()
    card.widgets["generic_body.surface.ks"].setValue(0.0075)
    app.processEvents()
    line = card.surface_provenance.text()
    assert "ks" not in line.split("are the engine")[0], (
        f"an edited conductivity is still labeled an engine default: {line!r}")
    assert manager.get_value("generic_body.surface.ks") == 0.0075
    print("    [provenance] editing ks drops its 'engine default' label and "
          "keeps the typed value")


def test_the_seeded_defaults_are_re_derived_from_the_cpp_sources():
    """Drift between the engine's defaults and the table is a test failure."""
    from ptm.body_capabilities import (GENERIC_BODY_SURFACE_DEFAULTS,
                                       parse_generic_body_defaults)
    derived = parse_generic_body_defaults(REPO_ROOT)
    assert derived == GENERIC_BODY_SURFACE_DEFAULTS, (
        f"ptm/body_table.py no longer matches the C++ sources.\n"
        f"  committed: {GENERIC_BODY_SURFACE_DEFAULTS}\n"
        f"  derived  : {derived}\n"
        f"Rerun: python3 -m ptm.body_capabilities --regenerate")
    print(f"    [parity] {len(derived)} generic-body surface defaults "
          f"re-derived from generic_body.h and ptm_thermal_sim.cpp and equal "
          f"to the committed table")


def test_a_defaults_only_card_emits_a_configuration_the_engine_runs():
    """No hand-entered properties at all: the run must still start.

    "The card fills the boxes in" is only worth anything if what it fills them
    with is a configuration the solver accepts.
    """
    app = _app()
    engine = _engine()

    manager, _panel, _start, card = _open_asteroid_card(app)
    _name_the_body(app, manager, card)
    # Keplerian, so this arm tests the surface block and nothing else.
    card.kepler_radio.click()
    manager.set_value("generic_body.orbit.semi_major_axis_au", 1.1289220453)
    manager.set_value("generic_body.orbit.eccentricity", 0.2046513239)
    manager.set_value("generic_body.spin.period_hours", 4.29746)
    app.processEvents()

    ok, errors = manager.validate()
    assert ok, f"the GUI refused a card carrying only its own defaults: {errors}"

    with tempfile.TemporaryDirectory() as tmp:
        manager.set_value("output.directory", str(Path(tmp) / "out"))
        manager.set_value("simulation.timesteps", 2)
        manager.set_value("simulation.duration_days", 0.2)
        manager.set_value("thermal.equilibration_orbits", 0.0)
        config_path = Path(tmp) / "config.yaml"
        manager.save_run_yaml(str(config_path), REPO_ROOT)
        run = subprocess.run(
            [str(engine), "--config", str(config_path)],
            capture_output=True, text=True, timeout=900, cwd=str(REPO_ROOT))
        log = run.stdout + run.stderr
        assert "Invalid generic body configuration" not in log, log[-3000:]
        assert run.returncode == 0, f"rc={run.returncode}\n{log[-4000:]}"
    print("    [defaults-run] a card carrying only the engine's own surface "
          "defaults emits a configuration the engine runs (rc=0)")


# ---------------------------------------------------------------------------

def main():
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    failures, skips = [], []
    for test in tests:
        print(f"  {test.__name__}")
        try:
            test()
        except Skip as exc:
            skips.append((test.__name__, str(exc)))
            print(f"    SKIPPED: {exc}")
        except AssertionError as exc:
            failures.append((test.__name__, str(exc)))
            print(f"    FAILED: {exc}")
        except Exception as exc:                          # noqa: BLE001
            failures.append((test.__name__, f"{type(exc).__name__}: {exc}"))
            print(f"    ERROR: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - len(failures) - len(skips)} passed, "
          f"{len(failures)} failed, {len(skips)} skipped")
    for name, why in skips:
        print(f"  SKIP {name}: {why}")
    for name, why in failures:
        print(f"  FAIL {name}: {why}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
