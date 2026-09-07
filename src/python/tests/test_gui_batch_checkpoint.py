#!/usr/bin/env python3
"""Banking finished facet groups, as the application presents it.

The model can write each group of facets to disk the moment that group is
finished for the whole run, so an interruption costs only the groups that were
never reached. This file pins the application's side of that: the controls that
switch it on, the size figure beside them, and what the launch check says about
work already banked in a directory.

WHAT IS PINNED, AND WHY EACH ARM EXISTS.

* **The size arithmetic is the model's own, and it is checked against the
  model's own printed figure.** Two real runs on the shipped crater mesh were
  used to fix it: one banking only the surface series, one banking the surface
  series, a finished column and two subsurface depths. The model printed
  ``192 B/face`` and ``336 B/face``; the numbers appear below as constants and
  the Python arithmetic has to reproduce both. A mirror of an engine rule that
  nothing compares against the engine is the failure this project has already
  paid for twice, in this exact shape.

* **The estimate is a LOWER BOUND and says so.** What is banked covers the
  spin-up steps as well as the reported ones, and how many spin-up steps a run
  takes is settled by the solver. Guessing here would be a second copy of that
  rule. The measured pair above is the proof: the reported window was 12 steps
  and the model banked 24, so an estimate over the reported window alone is
  exactly half. The arm pins the inequality, not the ratio.

* **The refusals are the model's, measured on the model.** Each blocker below
  was run: the graphics-card solver, facet-to-facet infrared exchange, a
  spin-up ending on its own convergence test, and a finished state file inside
  the directory holding part-finished work. All four stop the run with status 1.

* ⛔ **One case is NOT a refusal, and that is the point of its arm.** With
  ``thermal.run_thermal_mesh`` off and the switch on, the run finishes with
  status 0 and never mentions the switch. Nothing is banked and nothing says
  so. The application must warn -- and must NOT refuse, because refusing would
  block a launch the model accepts.

* **The controls are asserted with ``click()`` and ``isVisibleTo``.** A
  ``setChecked`` call succeeds on a control the panel has disabled, and
  ``isHidden`` is blind to a hidden ancestor. Both blind spots have let arms
  in this application pass on a control that did nothing.

Needs PyQt6 for the control arms; the arithmetic and the directory reading run
without it. Run directly::

    python3 src/python/tests/test_gui_batch_checkpoint.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "python"))
sys.path.insert(0, str(REPO_ROOT / "gui"))

# ⛔ BEFORE ANY GUI IMPORT. `gui_panels.visualization_panel` reads this at
# module import time; setting it later has no effect.
os.environ.setdefault("PTM_GUI_DISABLE_3D", "1")

CHECKPOINT_HEADER = REPO_ROOT / "src" / "cpp" / "thermal" / "batch_checkpoint.h"

# ---------------------------------------------------------------------------
# The two measurements this file is anchored to.
#
# Both come from real runs of the shipped 12,800-facet crater mesh over a
# 12-step window with one period of spin-up, on the build that introduced the
# feature. The model prints the per-facet figure at startup and repeats the
# shape in the index it writes beside the banked files.
# ---------------------------------------------------------------------------

#: Surface series only: 24 steps x 8 bytes.
MEASURED_SURFACE_ONLY = dict(
    n_time_steps=24, n_layers=23, n_depths=0, sections=1,
    bytes_per_face=192)

#: Surface series + finished column + two subsurface depths:
#: 24*8 + 23*4 + 2*24 + 4.
MEASURED_ALL_SECTIONS = dict(
    n_time_steps=24, n_layers=23, n_depths=2, sections=7,
    bytes_per_face=336)

#: What the configuration asked for in both runs, against what was banked. The
#: gap is the spin-up, and it is why the application's figure is a bound.
MEASURED_REPORTED_STEPS = 12
MEASURED_FACES = 12800


class Skip(Exception):
    """A check that cannot run here, as distinct from one that failed."""


_APP = None
_PANEL = None
_MANAGER = None
_SCRATCH = None


def _panel():
    """One offscreen ConfigurationPanel, reused across the control arms."""
    global _APP, _PANEL, _MANAGER, _SCRATCH
    if _PANEL is not None:
        return _MANAGER, _PANEL
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        import PyQt6
    except ImportError as exc:                     # pragma: no cover
        raise Skip(f"PyQt6 is unavailable here ({exc})")
    plug = Path(PyQt6.__file__).parent / "Qt6" / "plugins"
    os.environ.setdefault("QT_PLUGIN_PATH", str(plug))
    os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(plug / "platforms"))
    from PyQt6.QtCore import QSettings
    from PyQt6.QtWidgets import QApplication

    # Redirect QSettings before any panel touches it, or this writes the real
    # preferences of an installed copy.
    _SCRATCH = tempfile.mkdtemp(prefix="ptm_batchckpt_test_")
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    for scope in (QSettings.Scope.UserScope, QSettings.Scope.SystemScope):
        QSettings.setPath(QSettings.Format.IniFormat, scope, _SCRATCH)
    if QApplication.instance() is None:
        _APP = QApplication([])
    try:
        from gui_core.config_manager import ConfigManager
        from gui_panels.config_panel import ConfigurationPanel
    except ImportError as exc:                     # pragma: no cover
        raise Skip(f"the GUI panel stack is unavailable here ({exc})")
    _MANAGER = ConfigManager()
    _PANEL = ConfigurationPanel(_MANAGER)
    return _MANAGER, _PANEL


def _tab():
    """The Thermal tab, on a configuration reset to this application's defaults."""
    manager, panel = _panel()
    manager.reset_to_defaults()
    tab = panel.tabs["thermal"]
    tab.load_from_config()
    return manager, tab


def _pump():
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance()
    for _ in range(4):
        app.processEvents()


def _module():
    from gui_core import batch_checkpoint as bck
    return bck


def _config(**thermal):
    """A configuration with the feature on and nothing in the way."""
    base = {"batch_checkpoint": True, "run_thermal_mesh": True,
            "use_gpu": False}
    base.update(thermal)
    return {"thermal": base, "radiation": {"enabled": False},
            "output": {"directory": "/tmp/does-not-exist-run"},
            "simulation": {"timesteps": MEASURED_REPORTED_STEPS}}


def _bank(tmp, **manifest):
    """A directory shaped like one the model leaves behind."""
    os.makedirs(tmp, exist_ok=True)
    for i in range(int(manifest.pop("_files", 1))):
        with open(os.path.join(tmp, "part_g0001_s%06d.ptmb" % i), "wb") as fh:
            fh.write(b"PTMB")
    body = dict(schema="ptm3d-batch-partial-manifest/1",
                n_faces_total=MEASURED_FACES, batch_size=8, n_layers=23,
                n_time_steps=24, n_depths=0, sections=1, run_identity=1,
                batches_banked=1600, faces_banked=MEASURED_FACES,
                complete=True,
                files=["part_g0001_s%06d.ptmb" % i
                       for i in range(int(manifest.get("_listed", 1)))])
    manifest.pop("_listed", None)
    body.update(manifest)
    with open(os.path.join(tmp, "manifest.json"), "w") as fh:
        json.dump(body, fh)
    return tmp


# ---------------------------------------------------------------------------
# 1. The size arithmetic, against the model's own printed figure
# ---------------------------------------------------------------------------

def test_the_per_facet_size_reproduces_the_models_own_figure():
    """The one arm that makes this a mirror and not a second opinion.

    Both measurements are from real runs. The first exercises the surface
    series alone; the second adds the finished column and the subsurface
    accumulators, so every term in the model's formula is covered by one of
    them and a term dropped from the Python copy fails at least one.
    """
    bck = _module()
    for name, m in (("surface only", MEASURED_SURFACE_ONLY),
                    ("all sections", MEASURED_ALL_SECTIONS)):
        got = bck.bytes_per_face(
            m["n_time_steps"],
            n_layers=m["n_layers"],
            n_depths=m["n_depths"],
            with_column=bool(m["sections"] & bck.SECTION_FINAL_COLUMN),
            with_depths=bool(m["sections"] & bck.SECTION_DEPTH_ACCUM))
        assert got == m["bytes_per_face"], (
            f"{name}: this module says {got} bytes per facet, the model "
            f"printed {m['bytes_per_face']}")


def test_dropping_a_term_from_the_size_would_be_caught():
    """The mirror arm can fail. Without this, an arm that agreed with itself
    would be indistinguishable from one that agreed with the model."""
    bck = _module()
    m = MEASURED_ALL_SECTIONS
    wrong = bck.bytes_per_face(m["n_time_steps"], n_layers=m["n_layers"],
                               n_depths=m["n_depths"],
                               with_column=False, with_depths=True)
    assert wrong != m["bytes_per_face"], (
        "leaving the finished column out of the sum still matched the "
        "model's figure, so the arm above proves nothing")


def test_the_size_the_application_shows_is_a_lower_bound_and_says_so():
    """It counts the reported window; the model also banks the spin-up.

    Measured: the configuration asked for 12 steps and the model banked 24.
    The arm pins the inequality and the flag, not the factor -- the factor is
    a property of that one configuration's spin-up.
    """
    bck = _module()
    est = bck.size_estimate(_config(), MEASURED_FACES, MEASURED_REPORTED_STEPS)
    actual = MEASURED_SURFACE_ONLY["bytes_per_face"] * MEASURED_FACES
    assert est.bytes_at_least <= actual, (
        f"the application's figure {est.bytes_at_least} exceeds what the "
        f"model actually banked, {actual}; a bound that is not a bound is "
        f"worse than no figure")
    assert not est.exact, "the figure is presented as exact, and it is not"
    assert any("spin-up" in term for term in est.unknown_terms), (
        "the spin-up steps are left out of the figure and not named as left "
        "out")


def test_asking_for_subsurface_depths_grows_the_figure_by_the_models_term():
    """The subsurface accumulators are banked too, so a run that asks for them
    banks more per facet -- by the model's own per-depth term, which the
    ``336 B/face`` measurement above pins independently."""
    bck = _module()
    plain = bck.size_estimate(_config(), MEASURED_FACES,
                              MEASURED_REPORTED_STEPS)
    with_depths = _config()
    with_depths["output"]["map_depths_m"] = [0.0, 0.05]
    deeper = bck.size_estimate(with_depths, MEASURED_FACES,
                               MEASURED_REPORTED_STEPS)
    grew = deeper.bytes_at_least - plain.bytes_at_least
    expected = MEASURED_FACES * (2 * bck.DEPTH_BYTES_PER_DEPTH
                                 + bck.DEPTH_BYTES_FIXED)
    assert grew == expected, (
        f"asking for two subsurface depths grew the figure by {grew} bytes; "
        f"the model banks {expected} more")


def test_the_read_out_names_the_group_size_and_what_it_does_not_bound():
    bck = _module()
    text = " ".join(bck.size_lines(_config(), MEASURED_FACES,
                                   MEASURED_REPORTED_STEPS))
    assert str(bck.BATCH_FACES) in text, (
        "the read-out does not say how many facets are in one group, which is "
        "the size of the loss an interruption still costs")
    assert "shorter runs" in text, (
        "the read-out does not point at the thing that bounds an interruption "
        "more tightly")


def test_a_size_over_the_ceiling_is_an_error_and_not_a_line_of_prose():
    """⛔ Assert the BRANCH, not the string. A message that reaches only a
    summary line while the launch proceeds is not a refusal, and that exact
    shape shipped once in this application already."""
    bck = _module()
    cfg = _config(batch_checkpoint_max_gib=0.000001)
    findings = bck.preflight_findings(cfg, MEASURED_FACES,
                                      MEASURED_REPORTED_STEPS)
    assert not findings.ok, (
        "a request over the ceiling did not block the launch check")
    assert any("limit" in e for e in findings.errors), findings.errors

    generous = bck.preflight_findings(_config(), MEASURED_FACES,
                                      MEASURED_REPORTED_STEPS)
    assert generous.ok, generous.errors


def test_the_ceiling_can_be_switched_off_the_way_the_model_reads_it():
    """Zero means no ceiling, because that is what the model does with it."""
    bck = _module()
    est = bck.size_estimate(_config(batch_checkpoint_max_gib=0.0),
                            MEASURED_FACES, MEASURED_REPORTED_STEPS)
    assert est.ceiling_bytes == 0 and not est.over_ceiling


# ---------------------------------------------------------------------------
# 2. What the model refuses, and the one thing it does not
# ---------------------------------------------------------------------------

def test_the_refusals_are_the_ones_the_model_makes():
    """Each of these was run and stopped the model with status 1."""
    bck = _module()
    cases = {
        "graphics cards": _config(use_gpu=True),
        "infrared exchange": {"thermal": {"batch_checkpoint": True,
                                          "run_thermal_mesh": True,
                                          "use_gpu": False},
                              "radiation": {"enabled": True}},
        "convergence test": _config(equilibration_stop_on_certificate=True),
    }
    for name, cfg in cases.items():
        blockers = bck.engine_support(cfg).blockers
        assert blockers, f"{name}: the model stops on this and nothing said so"
    assert not bck.engine_support(_config()).blockers


def test_a_run_that_solves_no_thermal_model_warns_and_does_not_block():
    """⛔ MEASURED: this combination completes with status 0 and never
    mentions the switch. It is the one unsupported case the model is silent
    about, so it is the one the application has to speak up about -- and it
    must not refuse a launch the model accepts."""
    bck = _module()
    cfg = _config(run_thermal_mesh=False)
    assert not bck.engine_support(cfg).blockers, (
        "refusing here would block a launch the model completes")
    assert bck.inert_warning(cfg), (
        "the switch will do nothing and nothing says so, which is the "
        "silence this warning exists to break")
    findings = bck.preflight_findings(cfg, MEASURED_FACES,
                                      MEASURED_REPORTED_STEPS)
    assert findings.ok, "a launch the model accepts was blocked"
    assert findings.warnings, "and it was not warned about either"


def test_a_finished_state_inside_the_bank_is_refused():
    """Measured: the model stops with status 1 and names both paths."""
    bck = _module()
    cfg = _config()
    cfg["thermal"]["final_state_file"] = \
        bck.partial_dir(cfg) + "/final.bin"
    assert bck.final_state_conflict(cfg), (
        "the model refuses this and the launch check did not")
    cfg["thermal"]["final_state_file"] = "/tmp/elsewhere/final.bin"
    assert not bck.final_state_conflict(cfg)


def test_the_switch_being_off_makes_the_launch_check_silent():
    bck = _module()
    cfg = _config(batch_checkpoint=False, use_gpu=True)
    findings = bck.preflight_findings(cfg, MEASURED_FACES,
                                      MEASURED_REPORTED_STEPS)
    assert not findings.errors and not findings.warnings and not findings.info
    assert findings.line == ""


# ---------------------------------------------------------------------------
# 3. Where the banked work goes, and what is already there
# ---------------------------------------------------------------------------

def test_the_directory_is_the_one_the_model_resolves():
    """⛔ The expected path is SPELLED OUT, not built from the module's own
    constant. Written the other way this arm compared the constant with
    itself: changing ``DEFAULT_SUBDIR`` to the wrong folder moved both sides
    of the comparison and the arm went on passing. Found by injecting exactly
    that defect."""
    bck = _module()
    cfg = _config()
    cfg["output"]["directory"] = "/tmp/run"
    assert bck.partial_dir(cfg) == "/tmp/run/_thermal_state/partial", \
        bck.partial_dir(cfg)
    cfg["thermal"]["batch_checkpoint_dir"] = "/elsewhere/bank"
    assert bck.partial_dir(cfg) == "/elsewhere/bank"


def test_the_default_directory_is_the_string_the_model_builds():
    """And the same spelling appears in the model's own source, so the two
    cannot drift apart in silence. This is the whole point of a mirror: an
    arm that only checks the copy against itself proves nothing."""
    engine = (REPO_ROOT / "src" / "cpp" / "main" /
              "ptm_thermal_sim.cpp").read_text()
    assert '"/_thermal_state/partial"' in engine, (
        "the model no longer builds its default directory from that string; "
        "the application's copy is now guessing")
    assert _module().DEFAULT_SUBDIR == "_thermal_state/partial"


def test_an_empty_directory_is_reported_and_not_passed_over():
    """The mistyped-directory case. A resume that quietly found nothing and
    integrated the whole mesh again is indistinguishable, in the output, from
    one that found its work -- which is why the model says this out loud too."""
    bck = _module()
    with tempfile.TemporaryDirectory() as tmp:
        lines = bck.resume_lines(bck.scan_partials(tmp))
        assert lines and "wrong directory" in " ".join(lines), lines


def test_banked_work_is_read_from_the_files_and_the_index_separately():
    bck = _module()
    with tempfile.TemporaryDirectory() as tmp:
        _bank(tmp, _files=3, _listed=2)
        banked = bck.scan_partials(tmp)
        assert banked.files_on_disk == 3
        assert banked.index_files == 2
        assert banked.index_behind, (
            "three files on disk against two in the index is the ordinary "
            "shape of an interruption and was not noticed")
        text = " ".join(bck.resume_lines(banked))
        assert "3" in text and "2" in text


def test_an_unreadable_index_is_not_treated_as_an_empty_one():
    bck = _module()
    with tempfile.TemporaryDirectory() as tmp:
        _bank(tmp, _files=1)
        with open(os.path.join(tmp, "manifest.json"), "w") as fh:
            fh.write("{ this is not json")
        banked = bck.scan_partials(tmp)
        assert banked.any_work, "the banked file was lost with the index"
        assert banked.manifest_present and not banked.manifest_readable
        assert "rebuilds it" in " ".join(bck.resume_lines(banked))


def test_a_completed_bank_says_the_next_run_would_integrate_nothing():
    """Measured: a run that finishes leaves its banked work behind."""
    bck = _module()
    with tempfile.TemporaryDirectory() as tmp:
        _bank(tmp, complete=True)
        text = " ".join(bck.resume_lines(bck.scan_partials(tmp)))
        assert "integrate nothing" in text, text


def test_work_banked_for_a_different_mesh_is_flagged_before_the_launch():
    bck = _module()
    with tempfile.TemporaryDirectory() as tmp:
        _bank(tmp, n_faces_total=999)
        problems = bck.resume_conflicts(bck.scan_partials(tmp), _config(),
                                        MEASURED_FACES)
        assert problems and "999" in problems[0], problems
        agree = bck.resume_conflicts(bck.scan_partials(tmp), _config(), 999)
        assert not agree, agree


def test_a_changed_product_shape_is_flagged_before_the_launch():
    """The model compares which parts each banked group carries, so turning a
    finished state file on -- or asking for subsurface depths -- after some
    work is banked makes it refuse the whole run."""
    bck = _module()
    with tempfile.TemporaryDirectory() as tmp:
        _bank(tmp, sections=1, n_depths=0)
        cfg = _config()
        cfg["thermal"]["final_state_file"] = "/tmp/elsewhere/final.bin"
        problems = bck.resume_conflicts(bck.scan_partials(tmp), cfg,
                                        MEASURED_FACES)
        assert any("column" in p for p in problems), problems

        cfg2 = _config()
        cfg2["output"]["map_depths_m"] = [0.0, 0.05]
        problems2 = bck.resume_conflicts(bck.scan_partials(tmp), cfg2,
                                         MEASURED_FACES)
        assert any("depth" in p for p in problems2), problems2


def test_a_conflict_is_a_warning_and_never_blocks_the_launch():
    """The model does not trust the index either: it re-reads every banked
    file. A check written here against the index is a signal, not a verdict,
    and must not stop a launch on the strength of a file the model treats as
    advisory."""
    bck = _module()
    with tempfile.TemporaryDirectory() as tmp:
        _bank(tmp, n_faces_total=999)
        cfg = _config()
        cfg["thermal"]["batch_checkpoint_dir"] = tmp
        findings = bck.preflight_findings(cfg, MEASURED_FACES,
                                          MEASURED_REPORTED_STEPS)
        assert findings.warnings, "the disagreement was not reported at all"
        assert findings.ok, "a launch was blocked on the strength of an index"


def test_nothing_here_reads_the_banked_files_themselves():
    """A second reader of the banked format is the failure this project
    already has a ledger entry for. The application reads the directory
    listing and the index, and nothing else."""
    source = (REPO_ROOT / "gui" / "gui_core" / "batch_checkpoint.py").read_text()
    for forbidden in ("struct.unpack", '"rb"', "'rb'", "PTMB"):
        assert forbidden not in source, (
            f"{forbidden!r} appears in the module, which suggests it has "
            f"started reading the banked format itself")


# ---------------------------------------------------------------------------
# 4. The controls
# ---------------------------------------------------------------------------

def test_every_control_exists_and_is_reachable():
    """⛔ `isVisibleTo`, not `isHidden`: the second is blind to a hidden
    ancestor, and 36 merged arms in this application once passed on a control
    whose whole group was hidden."""
    bck = _module()
    _manager, tab = _tab()
    for key in (bck.ENABLE_KEY, bck.DIR_KEY, bck.MAX_GIB_KEY):
        widget = tab.widgets.get(key)
        assert widget is not None, f"{key} has no control"
        assert widget.isEnabled(), (
            f"{key} is disabled, and a click on a disabled control does "
            f"nothing while setChecked on it still succeeds")
        assert widget.isVisibleTo(tab.crash_group), (
            f"{key} is not visible inside its own group")


def test_the_switch_is_clicked_and_the_configuration_follows():
    """⛔ `click()`, not `setChecked()`. setChecked succeeds on a control the
    panel has disabled, so an arm written with it cannot tell a live switch
    from a dead one."""
    bck = _module()
    manager, tab = _tab()
    box = tab.widgets[bck.ENABLE_KEY]
    assert not box.isChecked(), "the switch is not off by default"
    assert manager.get_config()["thermal"]["batch_checkpoint"] is False
    box.click()
    _pump()
    assert manager.get_config()["thermal"]["batch_checkpoint"] is True, (
        "clicking the switch did not reach the configuration")
    box.click()
    _pump()
    assert manager.get_config()["thermal"]["batch_checkpoint"] is False


def test_the_read_out_is_not_a_dead_label():
    bck = _module()
    _manager, tab = _tab()
    off = tab.crash_summary.text()
    assert off, "the read-out is empty when the feature is off"
    tab.widgets[bck.ENABLE_KEY].click()
    _pump()
    on = tab.crash_summary.text()
    assert on and on != off, (
        "the read-out did not change when the switch was clicked, so it is "
        "not reporting anything about the setting")
    assert bck.DEFAULT_SUBDIR in on, (
        "the read-out does not say where the work will be written, which is "
        "what makes a mistyped directory visible")


def test_the_read_out_carries_the_refusal_the_model_would_make():
    bck = _module()
    manager, tab = _tab()
    tab.widgets[bck.ENABLE_KEY].click()
    _pump()
    manager.set_value("thermal.use_gpu", True)
    tab.load_from_config()
    tab._refresh_crash_readout()
    _pump()
    assert "graphics cards" in tab.crash_summary.text(), (
        tab.crash_summary.text())


def test_the_three_keys_are_written_at_the_models_own_defaults():
    """And no fourth. A key with no control, stamped into every file that is
    saved, reads exactly like a setting that is being honored."""
    from ptm.engine_defaults import ENGINE_DEFAULTS
    bck = _module()
    manager, _tab_ = _tab()
    thermal = manager.get_config()["thermal"]
    for key, leaf in ((bck.ENABLE_KEY, bck.ENABLE_LEAF),
                      (bck.DIR_KEY, bck.DIR_LEAF),
                      (bck.MAX_GIB_KEY, bck.MAX_GIB_LEAF)):
        assert leaf in thermal, f"{leaf} is not written"
        assert thermal[leaf] == ENGINE_DEFAULTS[key], (
            f"{leaf} is {thermal[leaf]!r}, the model's own default is "
            f"{ENGINE_DEFAULTS[key]!r}")
    assert bck.FLUSH_MIB_LEAF not in thermal, (
        "the flush size is written into every saved file and has no control, "
        "so it reads as a setting and is not one")


def test_the_configuration_check_refuses_what_the_model_refuses():
    manager, _tab_ = _tab()
    manager.set_value("thermal.batch_checkpoint", True)
    manager.set_value("thermal.use_gpu", True)
    result = manager.validate()
    errors = result[1] if isinstance(result, tuple) else result
    assert any("Crash resilience" in e for e in errors), errors


def test_the_configuration_check_says_nothing_when_the_banking_is_off():
    """⛔ The arm that was missing, and it cost four shipped configurations.

    Every fixture in this file switches the banking ON, so nothing here varied
    the one input that decides whether these conditions are conditions at all.
    The first draft of the configuration check called ``engine_support``
    unconditionally, and every run on the graphics cards -- and every run with
    facet-to-facet infrared exchange -- was reported as invalid. The full test
    suite found it; this file did not.
    """
    bck = _module()
    off = _config(batch_checkpoint=False, use_gpu=True)
    off["radiation"]["enabled"] = True
    assert bck.engine_support(off).blockers, (
        "engine_support is expected to describe the conditions whatever the "
        "switch says; it is the CALLER that must gate on the switch")

    manager, _tab_ = _tab()
    manager.set_value("thermal.batch_checkpoint", False)
    manager.set_value("thermal.use_gpu", True)
    result = manager.validate()
    errors = result[1] if isinstance(result, tuple) else result
    assert not any("Crash resilience" in e for e in errors), errors


def test_the_launch_report_carries_the_size_line():
    from gui_core.preflight import PreflightReport
    report = PreflightReport()
    assert report.batch_checkpoint_line == ""
    report.batch_checkpoint_line = "a figure"
    assert any(line.startswith("BATCH-CHECKPOINT:")
               for line in report.summary_lines())


# ---------------------------------------------------------------------------

TESTS = [
    test_the_per_facet_size_reproduces_the_models_own_figure,
    test_dropping_a_term_from_the_size_would_be_caught,
    test_the_size_the_application_shows_is_a_lower_bound_and_says_so,
    test_asking_for_subsurface_depths_grows_the_figure_by_the_models_term,
    test_the_read_out_names_the_group_size_and_what_it_does_not_bound,
    test_a_size_over_the_ceiling_is_an_error_and_not_a_line_of_prose,
    test_the_ceiling_can_be_switched_off_the_way_the_model_reads_it,
    test_the_refusals_are_the_ones_the_model_makes,
    test_a_run_that_solves_no_thermal_model_warns_and_does_not_block,
    test_a_finished_state_inside_the_bank_is_refused,
    test_the_switch_being_off_makes_the_launch_check_silent,
    test_the_directory_is_the_one_the_model_resolves,
    test_the_default_directory_is_the_string_the_model_builds,
    test_an_empty_directory_is_reported_and_not_passed_over,
    test_banked_work_is_read_from_the_files_and_the_index_separately,
    test_an_unreadable_index_is_not_treated_as_an_empty_one,
    test_a_completed_bank_says_the_next_run_would_integrate_nothing,
    test_work_banked_for_a_different_mesh_is_flagged_before_the_launch,
    test_a_changed_product_shape_is_flagged_before_the_launch,
    test_a_conflict_is_a_warning_and_never_blocks_the_launch,
    test_nothing_here_reads_the_banked_files_themselves,
    test_every_control_exists_and_is_reachable,
    test_the_switch_is_clicked_and_the_configuration_follows,
    test_the_read_out_is_not_a_dead_label,
    test_the_read_out_carries_the_refusal_the_model_would_make,
    test_the_three_keys_are_written_at_the_models_own_defaults,
    test_the_configuration_check_refuses_what_the_model_refuses,
    test_the_configuration_check_says_nothing_when_the_banking_is_off,
    test_the_launch_report_carries_the_size_line,
]


def main():
    failures = skipped = 0
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
    print(f"\n[test_gui_batch_checkpoint] {total - failures - skipped}/{total} "
          f"passed"
          + (f", {skipped} skipped" if skipped else "")
          + (f", {failures} failed" if failures else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
