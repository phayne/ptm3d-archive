#!/usr/bin/env python3
"""DP-1(b): opening a config and saving it must not rewrite the whole file.

The Stage-1 fix stopped a *run* from overwriting the user's file. It did not
make Save itself lossless: ``save_to_yaml`` was ``yaml.dump`` of the merged
dictionary, which erases every comment (measured: 10 and 19 to 0 on two shipped
examples) and writes every key the GUI knows about whether the file mentioned
it or not.

The property this pins, and the one every Stage-2 item leans on:

  **open -> save with no edits is byte-identical**, and
  **open -> edit ONE key -> save changes only that key's line.**

The second half is what makes ``gpu.min_vram_margin_gib`` safe to expose:
``config_parser.h`` reads its mere PRESENCE as "enforce the VRAM floor on every
platform, not only the wedge-prone one", so a writer that always writes
everything cannot expose that section at all.

Run directly::

    python3 src/python/tests/test_gui_comment_roundtrip.py
"""

from __future__ import annotations

import difflib
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "python"))
sys.path.insert(0, str(REPO_ROOT / "gui"))


class Skip(Exception):
    pass


_APP = None


def _manager():
    try:
        from gui_core.config_manager import ConfigManager
    except ImportError as exc:                     # pragma: no cover
        raise Skip(f"the GUI's config manager is unavailable here ({exc})")
    return ConfigManager()


def _require_ruamel():
    from gui_core import yaml_roundtrip
    if not yaml_roundtrip.available():
        raise Skip(
            "ruamel.yaml is not installed. It is a REQUIREMENT of "
            "gui/requirements.txt, not an optional extra: without it the GUI "
            "cannot preserve comments and cannot express an unset key. "
            "Install with: pip install -r gui/requirements.txt")


#: Shipped examples with real comments. These are the files a colleague opens.
COMMENTED_EXAMPLES = [
    "examples/tutorial04_crater80N_coupled.yaml",
    "examples/quickstart_moon_diurnal.yaml",
]


#: Configs where opening them in the panel legitimately CHANGES the
#: configuration, with the reason. Every entry is a substitution the GUI makes
#: on the user's behalf and now announces in the console; listing it here keeps
#: it a decision on the record instead of a silent drift the general check
#: would otherwise have to be weakened for.
#:
#: justitia_generic-test.yaml is a generic body with no built-in albedo map, so
#: IlluminationTab.refresh_builtin_albedo falls the source back from Built-in
#: to None rather than promising a map that does not exist. Pre-existing on
#: main (faa5bb8); invisible before the comment-preserving save, because Save
#: rewrote everything anyway. Owed as GUI-ALBEDO-FALLBACK-EDIT.
ANNOUNCED_ON_OPEN = {
    "justitia_generic-test.yaml": {
        "illumination.use_albedo_map",
        "illumination.albedo_maps_directory",
        "illumination.albedo_map",
    },
}


def _comment_lines(text):
    return [ln for ln in text.splitlines() if ln.lstrip().startswith("#")]


def _changed_lines(before, after):
    """Lines that differ, ignoring pure additions of a trailing newline."""
    diff = list(difflib.unified_diff(
        before.splitlines(), after.splitlines(), lineterm="", n=0))
    return [ln for ln in diff
            if ln[:1] in "+-" and not ln.startswith(("+++", "---"))]


# --------------------------------------------------------------------------

def test_ruamel_is_installed():
    """The requirement is a requirement. Say so loudly if it is missing."""
    _require_ruamel()


def test_open_then_save_with_no_edits_is_byte_identical():
    _require_ruamel()
    with tempfile.TemporaryDirectory() as tmp:
        for rel in COMMENTED_EXAMPLES:
            src = REPO_ROOT / rel
            work = Path(tmp) / Path(rel).name
            shutil.copyfile(src, work)
            before = work.read_text()
            assert _comment_lines(before), (
                f"{rel} carries no comments, so it cannot test comment "
                f"preservation; pick a different fixture")

            manager = _manager()
            manager.load_from_yaml(str(work))
            manager.save_to_yaml(str(work))
            after = work.read_text()

            assert after == before, (
                f"{rel}: open-then-save changed the file with no edits.\n"
                + "\n".join(_changed_lines(before, after)[:40]))


def test_comments_survive_an_edit():
    _require_ruamel()
    with tempfile.TemporaryDirectory() as tmp:
        for rel in COMMENTED_EXAMPLES:
            src = REPO_ROOT / rel
            work = Path(tmp) / Path(rel).name
            shutil.copyfile(src, work)
            before = work.read_text()
            n_before = len(_comment_lines(before))

            manager = _manager()
            manager.load_from_yaml(str(work))
            manager.set_value("simulation.timesteps", 999)
            manager.save_to_yaml(str(work))
            after = work.read_text()

            n_after = len(_comment_lines(after))
            assert n_after == n_before, (
                f"{rel}: comment lines {n_before} -> {n_after} after editing "
                f"one key. This is the defect (GUI-COMMENT-LOSS) the round "
                f"trip exists to fix.")
            assert "timesteps: 999" in after, (
                f"{rel}: the edit did not reach the file")


def test_the_byte_diff_is_confined_to_the_edited_key():
    _require_ruamel()
    with tempfile.TemporaryDirectory() as tmp:
        rel = COMMENTED_EXAMPLES[0]
        work = Path(tmp) / Path(rel).name
        shutil.copyfile(REPO_ROOT / rel, work)
        before = work.read_text()

        manager = _manager()
        manager.load_from_yaml(str(work))
        manager.set_value("simulation.timesteps", 999)
        manager.save_to_yaml(str(work))
        after = work.read_text()

        changed = _changed_lines(before, after)
        # One line removed, one added -- and nothing else.
        assert len(changed) == 2, (
            f"editing one key rewrote {len(changed)} lines:\n"
            + "\n".join(changed[:40]))
        assert all("timesteps" in ln for ln in changed), (
            "the byte difference reaches lines other than the edited key:\n"
            + "\n".join(changed))


def test_an_untouched_key_the_file_omits_is_not_written():
    """The property that makes gpu.min_vram_margin_gib safe to expose.

    ``config_parser.h``: writing the key AT ALL promotes a macOS+AMD guardrail
    into a universal one. The GUI must therefore be able to leave it out, and
    'leave it out' has to survive an unrelated edit.
    """
    _require_ruamel()
    with tempfile.TemporaryDirectory() as tmp:
        rel = COMMENTED_EXAMPLES[0]
        work = Path(tmp) / Path(rel).name
        shutil.copyfile(REPO_ROOT / rel, work)
        assert "min_vram_margin_gib" not in work.read_text()

        manager = _manager()
        manager.load_from_yaml(str(work))
        manager.set_value("simulation.timesteps", 999)
        manager.save_to_yaml(str(work))
        after = work.read_text()

        assert "min_vram_margin_gib" not in after, (
            "saving wrote gpu.min_vram_margin_gib into a config that never "
            "mentioned it. The engine reads its presence as 'enforce the VRAM "
            "floor on EVERY platform'; this save changed the run's GPU policy "
            "without anyone asking.")
        # A top-level `gpu:` line, not the substring -- `use_gpu:` contains it.
        assert not any(ln.rstrip() == "gpu:" for ln in after.splitlines()), (
            "saving added a top-level gpu: section to a config that had none")


def test_editing_a_gpu_key_does_write_it():
    """The other half: an unset key that the user SETS must appear."""
    _require_ruamel()
    with tempfile.TemporaryDirectory() as tmp:
        rel = COMMENTED_EXAMPLES[0]
        work = Path(tmp) / Path(rel).name
        shutil.copyfile(REPO_ROOT / rel, work)

        manager = _manager()
        manager.load_from_yaml(str(work))
        manager.set_value("gpu.min_vram_margin_gib", 3.5)
        manager.save_to_yaml(str(work))
        after = work.read_text()

        assert "min_vram_margin_gib: 3.5" in after, (
            "an explicitly edited gpu.min_vram_margin_gib did not reach the "
            "file, so the control would be inert")

        # ...and it round-trips back.
        again = _manager()
        again.load_from_yaml(str(work))
        assert again.get_value("gpu.min_vram_margin_gib") == 3.5


def test_removing_an_override_removes_its_line():
    _require_ruamel()
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "with_override.yaml"
        work.write_text(
            "# a comment that must survive\n"
            "simulation:\n"
            "  body: Moon\n"
            "  timesteps: 24\n"
            "thermophysical:\n"
            "  albedo_override: 0.3   # trailing comment\n")
        manager = _manager()
        manager.load_from_yaml(str(work))
        assert manager.get_value("thermophysical.use_default_albedo") is False
        manager.set_value("thermophysical.use_default_albedo", True)
        manager.save_to_yaml(str(work))
        after = work.read_text()

        assert "albedo_override" not in after, (
            "unticking the albedo override left its key in the file, so the "
            "run would still apply it")
        assert "# a comment that must survive" in after


def test_a_new_project_still_saves_and_then_round_trips():
    """No source file to preserve -> full dump, and the NEXT save is minimal."""
    _require_ruamel()
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "fresh.yaml"
        manager = _manager()
        preserved, message = manager.comment_preservation_status()
        assert preserved is False and message, (
            "a project with no source file claimed to be preserving a layout "
            "it does not have")
        manager.save_to_yaml(str(work))
        assert work.exists() and work.read_text().strip()

        # The file just written becomes the baseline.
        preserved, _ = manager.comment_preservation_status()
        assert preserved is True, (
            "after saving, the written file should be the round-trip baseline")
        before = work.read_text()
        manager.save_to_yaml(str(work))
        assert work.read_text() == before, (
            "a second save with no edits rewrote the file")


def test_status_is_reported_not_inferred():
    """Whichever path ran, the manager can say which. Silence is the defect."""
    _require_ruamel()
    manager = _manager()
    preserved, message = manager.comment_preservation_status()
    assert isinstance(preserved, bool) and message.strip(), (
        "comment_preservation_status() must always carry an explanation the "
        "GUI can show")


def test_opening_a_config_in_the_PANEL_plans_no_edits():
    """The whole-window version of the invariant, and the one that catches the
    type-coercion class.

    ``load_from_yaml`` snapshots the engine-facing view BEFORE any panel runs.
    Every tab then writes its widgets' values back through ``set_numeric``,
    which stores ``type(widget.value())(number)``. A control bound as a double
    for a key the engine reads as an int therefore turns 15 into 15.0, and
    ``plan_edits``' type clause fires: opening a config and pressing Save adds
    keys the file never mentioned, for controls the user may not even be able
    to edit. Measured on ``tutorial04_crater80N_coupled.yaml``: the read-only
    depth grid injected ``thermal_b: 15.0``, ``thermal_m: 10.0`` and
    ``thermal_n: 15.0``.

    Fixing the three instances is not enough -- this checks the property, over
    every shipped example, so the next mis-bound control fails here.
    """
    _require_ruamel()
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        import PyQt6
    except ImportError as exc:                     # pragma: no cover
        raise Skip(f"PyQt6 is unavailable here ({exc})")
    plug = Path(PyQt6.__file__).parent / "Qt6" / "plugins"
    os.environ.setdefault("QT_PLUGIN_PATH", str(plug))
    os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(plug / "platforms"))
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import QSettings

    with tempfile.TemporaryDirectory() as scratch:
        QSettings.setDefaultFormat(QSettings.Format.IniFormat)
        for scope in (QSettings.Scope.UserScope, QSettings.Scope.SystemScope):
            QSettings.setPath(QSettings.Format.IniFormat, scope, scratch)
        from gui_core.config_manager import ConfigManager
        from gui_panels.config_panel import ConfigurationPanel
        # Bind it: `QApplication.instance() or QApplication([])` as a bare
        # expression lets the new instance be collected immediately, and the
        # next QWidget dies with "Must construct a QApplication before a
        # QWidget".
        global _APP
        _APP = QApplication.instance() or QApplication([])
        manager = ConfigManager()
        panel = ConfigurationPanel(manager)

        offenders = {}
        # configs/ TOO, not just examples/. Sweeping only examples/ missed the
        # one file that reproduced the sharpest instance of this class:
        # configs/survey_template.yaml, where an edit-free open+Save flipped
        # thermophysical.incidence_angle_albedo true -> false -- Hapke silently
        # switched off in the user's own file, under a console line saying
        # "Comments and layout preserved".
        examples = (sorted((REPO_ROOT / "examples").glob("*.yaml"))
                    + sorted((REPO_ROOT / "configs").rglob("*.yaml")))
        assert len(examples) >= 50, (
            f"only {len(examples)} configs found; this check would be "
            f"nearly vacuous")
        for src in examples:
            work = Path(scratch) / src.name
            shutil.copyfile(src, work)
            manager.load_from_yaml(str(work))
            panel.load_from_config()
            updates, removals = manager.plan_save_edits()
            allowed = ANNOUNCED_ON_OPEN.get(src.name, set())
            unexpected = ({k: v for k, v in updates.items() if k not in allowed},
                          [k for k in removals if k not in allowed])
            if unexpected[0] or unexpected[1]:
                offenders[src.name] = unexpected
        assert not offenders, (
            "opening these configs in the panel and pressing Save would "
            "rewrite keys nobody edited:\n  "
            + "\n  ".join(f"{k}: {v}" for k, v in offenders.items()))


def test_no_edit_save_is_byte_identical_on_EVERY_shipped_config():
    """Not "most files". Every one, examples/ and configs/ alike.

    Two failure modes were reproduced across the 77 shipped configs, both on a
    save whose edit plan was EMPTY: ruamel re-emitted `configs/survey_template.
    yaml`'s block sequence at its own indentation (6 rewritten lines), and it
    normalized `examples/quick_test.yaml`'s missing trailing newline. Both were
    invisible to a test that only looked at examples/ with comments.

    Two fixes, and this checks the property rather than either fix: the writer
    copies the file's own block-sequence shape, and a save with nothing to
    write does not touch the file at all.
    """
    _require_ruamel()
    targets = (sorted((REPO_ROOT / "configs").rglob("*.yaml"))
               + sorted((REPO_ROOT / "examples").glob("*.yaml")))
    assert len(targets) >= 50, (
        f"only {len(targets)} configs found; this sweep would be near-vacuous")
    offenders = {}
    with tempfile.TemporaryDirectory() as tmp:
        for src in targets:
            work = Path(tmp) / src.name
            shutil.copyfile(src, work)
            before = work.read_text()
            manager = _manager()
            manager.load_from_yaml(str(work))
            mode = manager.save_to_yaml(str(work))
            if work.read_text() != before:
                offenders[src.name] = (
                    mode, _changed_lines(before, work.read_text())[:6])
    assert not offenders, (
        f"{len(offenders)} of {len(targets)} configs changed on a save with "
        f"no edits:\n  "
        + "\n  ".join(f"{k}: {v}" for k, v in offenders.items()))


def test_one_edit_touches_only_that_key_across_every_config():
    """The stronger half, over the same 77 files."""
    _require_ruamel()
    targets = (sorted((REPO_ROOT / "configs").rglob("*.yaml"))
               + sorted((REPO_ROOT / "examples").glob("*.yaml")))
    offenders = {}
    with tempfile.TemporaryDirectory() as tmp:
        for src in targets:
            work = Path(tmp) / src.name
            shutil.copyfile(src, work)
            before = work.read_text()
            manager = _manager()
            manager.load_from_yaml(str(work))
            manager.set_value("simulation.timesteps", 999)
            manager.save_to_yaml(str(work))
            changed = _changed_lines(before, work.read_text())
            # A file with no `simulation:` section legitimately grows the
            # section header along with the key.
            stray = [ln for ln in changed
                     if "timesteps" not in ln and ln.strip() != "+simulation:"]
            if stray:
                offenders[src.name] = stray[:6]
    assert not offenders, (
        "editing one key reached unrelated lines:\n  "
        + "\n  ".join(f"{k}: {v}" for k, v in offenders.items()))


def test_a_reflow_the_writer_cannot_avoid_is_REPORTED():
    """detect_sequence_indent copies the FIRST sequence's shape, which is all
    ruamel offers -- there is no per-node indentation. A document that MIXES
    shapes will have the non-matching ones reflowed on a save with edits.

    That is a real limitation, so it is measured and reported rather than
    assumed away: any changed line mentioning none of the edited keys' path
    segments is a layout change, and the console says so.
    """
    _require_ruamel()
    with tempfile.TemporaryDirectory() as tmp:
        # Two block sequences at DIFFERENT indentations.
        mixed = Path(tmp) / "mixed.yaml"
        mixed.write_text(
            "a_list:\n  - one\n  - two\n"
            "simulation:\n  timesteps: 5\n"
            "b_list:\n- three\n- four\n")
        manager = _manager()
        manager.load_from_yaml(str(mixed))
        manager.set_value("simulation.timesteps", 7)
        manager.save_to_yaml(str(mixed))
        assert manager.last_save_layout_changes, (
            "a sequence at a shape the writer cannot reproduce was reflowed "
            "with nothing said")
        assert any("three" in ln for ln in manager.last_save_layout_changes), (
            manager.last_save_layout_changes)

    # ...and it does NOT cry wolf on the section header an edit really created.
    with tempfile.TemporaryDirectory() as tmp:
        plain = Path(tmp) / "plain.yaml"
        plain.write_text("mesh:\n  resolution: 4\n")
        manager = _manager()
        manager.load_from_yaml(str(plain))
        manager.set_value("simulation.timesteps", 7)
        manager.save_to_yaml(str(plain))
        assert manager.last_save_layout_changes == [], (
            f"the `simulation:` header the edit created was reported as a "
            f"reflow: {manager.last_save_layout_changes}")


def test_layout_detection_matches_a_KEY_not_a_substring():
    """`leaf in line` let one edit absorb another key's reflow.

    Editing output.directory gave leaves {"output", "directory"}, and an
    unrelated reflow of `topo_directory:` contains "directory" -- so the guard
    rail went quiet exactly where it was needed. Segments like `thermal`,
    `mesh` and `body` made that common.
    """
    _require_ruamel()
    from gui_core.yaml_roundtrip import layout_only_changes
    before = "output:\n  directory: out\nmesh:\n  topo_directory: data/topo\n"
    after = "output:\n  directory: NEW\nmesh:\n topo_directory: data/topo\n"
    reported = layout_only_changes(before, after, {"output.directory"})
    assert any("topo_directory" in ln for ln in reported), (
        f"a reflow of topo_directory was absorbed by an edit to "
        f"output.directory: {reported}")
    assert not any(ln.strip().endswith("directory: NEW") for ln in reported), (
        f"the edited line itself was reported as a reflow: {reported}")

    # A section header the edit really created is still not a reflow.
    assert layout_only_changes(
        "mesh:\n  resolution: 4\n",
        "mesh:\n  resolution: 4\nsimulation:\n  timesteps: 7\n",
        {"simulation.timesteps"}) == []


def test_comments_deleted_with_their_key_are_not_called_a_reflow():
    """Clearing a custom body deletes generic_body.orbit.* and the comments
    above them. Reporting those as "reformatted, carrying no key you edited"
    is the guard rail crying wolf on a correct deletion. Reproduced on
    examples/quickstart_asteroid_diurnal.yaml: 5 lines, four of them comments.
    """
    _require_ruamel()
    from gui_core.yaml_roundtrip import layout_only_changes
    before = ("thermal:\n  threads: 4\n"
              "generic_body:\n"
              "  # the orbit, from HORIZONS\n"
              "  orbit:\n    eccentricity: 0.2\n")
    after = "thermal:\n  threads: 8\n"
    reported = layout_only_changes(
        before, after, {"thermal.threads", "generic_body.orbit.eccentricity"})
    assert not any(ln.lstrip("-+").strip().startswith("#")
                   for ln in reported), (
        f"a comment deleted with its key was reported as a reflow: {reported}")
    # An ADDED comment is a real surprise and still reported.
    assert layout_only_changes(
        "a: 1\n", "a: 1\n# where did this come from\n", {"a"})


def test_a_second_save_does_not_repeat_the_first_saves_warning():
    _require_ruamel()
    with tempfile.TemporaryDirectory() as tmp:
        mixed = Path(tmp) / "mixed.yaml"
        mixed.write_text(
            "a_list:\n  - one\n"
            "simulation:\n  timesteps: 5\n"
            "b_list:\n- three\n")
        manager = _manager()
        manager.load_from_yaml(str(mixed))
        manager.set_value("simulation.timesteps", 7)
        manager.save_to_yaml(str(mixed))
        assert manager.last_save_layout_changes, "fixture did not reflow"
        assert manager.save_to_yaml(str(mixed)) == "unchanged"
        assert manager.last_save_layout_changes == [], (
            "a save that wrote nothing re-reported the previous save's reflow")


def test_no_shipped_config_is_reflowed_by_an_edit():
    """The corpus answer to the limitation above: 0 of 77."""
    _require_ruamel()
    targets = (sorted((REPO_ROOT / "configs").rglob("*.yaml"))
               + sorted((REPO_ROOT / "examples").glob("*.yaml")))
    offenders = {}
    with tempfile.TemporaryDirectory() as tmp:
        for src in targets:
            work = Path(tmp) / src.name
            shutil.copyfile(src, work)
            manager = _manager()
            manager.load_from_yaml(str(work))
            manager.set_value("simulation.timesteps", 999)
            manager.save_to_yaml(str(work))
            if manager.last_save_layout_changes:
                offenders[src.name] = manager.last_save_layout_changes[:4]
    assert not offenders, offenders


def test_a_recognized_alias_keeps_the_files_own_spelling():
    """Normalizing the STORED value made an edit-free save rewrite the file.

    `spinup_seed: historical` is `legacy`, `conductivity_model: martinez` is
    `martinez2021` -- the engine normalizes them the same way the combo does,
    so both spellings run identically. set_enum used to write the canonical
    form back into the config, and since _loaded_engine_view is snapshotted
    before the tabs run, plan_edits saw a user edit. No shipped config uses an
    alias, which is why the corpus sweep could not catch it.
    """
    _require_ruamel()
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        import PyQt6
    except ImportError as exc:                     # pragma: no cover
        raise Skip(f"PyQt6 is unavailable here ({exc})")
    plug = Path(PyQt6.__file__).parent / "Qt6" / "plugins"
    os.environ.setdefault("QT_PLUGIN_PATH", str(plug))
    os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(plug / "platforms"))
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import QSettings

    with tempfile.TemporaryDirectory() as scratch:
        QSettings.setDefaultFormat(QSettings.Format.IniFormat)
        for scope in (QSettings.Scope.UserScope, QSettings.Scope.SystemScope):
            QSettings.setPath(QSettings.Format.IniFormat, scope, scratch)
        from gui_core.config_manager import ConfigManager
        from gui_panels.config_panel import ConfigurationPanel
        global _APP
        _APP = QApplication.instance() or QApplication([])
        manager = ConfigManager()
        panel = ConfigurationPanel(manager)

        work = Path(scratch) / "aliases.yaml"
        work.write_text(
            "# keep me\nsimulation:\n  body: Moon\n"
            "thermal:\n"
            "  spinup_seed: historical\n"
            "  seed_anchor: derived\n"
            "thermophysical:\n"
            "  conductivity_model: martinez\n")
        before = work.read_text()
        manager.load_from_yaml(str(work))
        panel.load_from_config()
        assert manager.plan_save_edits() == ({}, []), (
            f"an alias spelling plans an edit: {manager.plan_save_edits()}")
        manager.save_to_yaml(str(work))
        assert work.read_text() == before, (
            "opening a config that uses documented alias spellings and saving "
            "with no edits rewrote it:\n"
            + "\n".join(_changed_lines(before, work.read_text())))
        # ...and the CONTROLS still show the canonical spelling.
        thermal = panel.tabs["thermal"]
        assert thermal.widgets["thermal.spinup_seed"].currentText() == "legacy"
        assert thermal.widgets["thermal.seed_anchor"].currentText() == "lumped"


def test_an_int_spelled_gpu_key_is_not_DELETED_on_an_edit_free_save():
    """The gpu: writer had its own equality rule, and it removed a line.

    A file spelling `placement_margin_gib: 4` (int) put 4 into the loaded view;
    the panel's set_numeric coerced it to 4.0; that now equals the engine's 4.0
    default, so the selective gpu writer dropped it -- and plan_edits
    classified the disappearance as a REMOVAL. Reproduced end to end through a
    real ConfigurationPanel: an open-and-Save with zero edits deleted the line
    from the user's file. No shipped examples/ config carries a gpu: section,
    which is why the corpus sweep did not see it.
    """
    _require_ruamel()
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        import PyQt6
    except ImportError as exc:                     # pragma: no cover
        raise Skip(f"PyQt6 is unavailable here ({exc})")
    plug = Path(PyQt6.__file__).parent / "Qt6" / "plugins"
    os.environ.setdefault("QT_PLUGIN_PATH", str(plug))
    os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(plug / "platforms"))
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import QSettings

    with tempfile.TemporaryDirectory() as scratch:
        QSettings.setDefaultFormat(QSettings.Format.IniFormat)
        for scope in (QSettings.Scope.UserScope, QSettings.Scope.SystemScope):
            QSettings.setPath(QSettings.Format.IniFormat, scope, scratch)
        from gui_core.config_manager import ConfigManager
        from gui_panels.config_panel import ConfigurationPanel
        global _APP
        _APP = QApplication.instance() or QApplication([])
        manager = ConfigManager()
        panel = ConfigurationPanel(manager)

        work = Path(scratch) / "gpu_int.yaml"
        work.write_text(
            "# keep me\nsimulation:\n  body: Moon\n"
            "gpu:\n"
            "  placement_margin_gib: 4       # int, numerically the default\n"
            "  min_vram_margin_gib: 2\n")
        before = work.read_text()
        manager.load_from_yaml(str(work))
        panel.load_from_config()
        assert manager.plan_save_edits() == ({}, []), (
            f"an edit-free open plans {manager.plan_save_edits()}")
        manager.save_to_yaml(str(work))
        after = work.read_text()
        assert after == before, (
            "an edit-free open+Save changed a gpu: line:\n"
            + "\n".join(_changed_lines(before, after)))
        assert "placement_margin_gib" in after, (
            "the line was deleted from the user's file")
        # ...and the explicit floor the file asked for is still explicit.
        assert manager.get_value("gpu.margin_explicit") is True


def test_no_shipped_config_is_BLOCKED_by_the_GUIs_own_validate():
    """A GUI rule the engine does not have is a launch blocked for nothing.

    validate() refused output.format / output.vtk_mode values the engine
    accepts -- five shipped configs use `none`, and the engine's own
    ConfigParser::validate() is DEAD CODE on this path
    (ptm_thermal_sim.cpp:17063 says so), so nothing refuses them. The dialog
    printed "Errors: VTK mode must be one of ..." directly above "The engine
    ACCEPTED this configuration".
    """
    targets = (sorted((REPO_ROOT / "examples").glob("*.yaml"))
               + sorted((REPO_ROOT / "configs").rglob("*.yaml")))
    blocked = {}
    for src in targets:
        manager = _manager()
        try:
            manager.load_from_yaml(str(src))
        except Exception as exc:                       # noqa: BLE001
            blocked[src.name] = f"load raised: {exc}"
            continue
        valid, messages = manager.validate()
        if not valid:
            blocked[src.name] = messages[:2]
    assert not blocked, (
        "the GUI refuses to launch these shipped configs:\n  "
        + "\n  ".join(f"{k}: {v}" for k, v in blocked.items()))


def test_the_word_off_survives_a_save_and_reopen():
    """YAML 1.1 vs 1.2, and it breaks the engine's own documented value.

    ruamel's round-trip emitter is YAML 1.2, where `off` is a string. PyYAML --
    which load_from_yaml and every other reader in this repo use -- is YAML
    1.1, where `off` is the boolean False. So:

    * a config spelling `thermal.eq_indirect_source: off` (which the engine's
      yaml-cpp reads as the string and ACCEPTS) loaded as False, validate()
      refused it, and Run was blocked on a config the engine would run;
    * picking `off` in the combo and saving emitted it bare, so reopening gave
      False and the run YAML carried `false`, which the engine FATALs on. The
      setting could not survive one save-and-reopen.

    The same split turned `illumination.bvh_cache_file: off` -- the DISABLE
    sentinel -- into a cache filename.
    """
    _require_ruamel()
    from gui_core import preflight
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "off.yaml"
        work.write_text(
            "simulation:\n  body: Moon\n"
            "thermal:\n  eq_indirect_source: off\n"
            "illumination:\n  bvh_cache_file: off\n")

        manager = _manager()
        manager.load_from_yaml(str(work))
        assert manager.get_value("thermal.eq_indirect_source") == "off", (
            f"read back as "
            f"{manager.get_value('thermal.eq_indirect_source')!r}")
        assert manager.get_value("illumination.bvh_cache_file") == "off"
        assert manager.validate()[0], manager.validate()[1]
        assert preflight.is_path_sentinel(
            "illumination.bvh_cache_file",
            manager.get_value("illumination.bvh_cache_file"))

        # The RUN yaml -- the one the engine reads -- must carry a string.
        import yaml as pyyaml
        run = pyyaml.safe_load(pyyaml.dump(manager._config_for_yaml()))
        assert run["thermal"]["eq_indirect_source"] == "off", run["thermal"]
        assert run["illumination"]["bvh_cache_file"] == "off"

        # And a user EDIT to `off` must come back as `off`, not False.
        work2 = Path(tmp) / "edit.yaml"
        work2.write_text(
            "simulation:\n  body: Moon\n"
            "thermal:\n  eq_indirect_source: auto\n")
        manager = _manager()
        manager.load_from_yaml(str(work2))
        manager.set_value("thermal.eq_indirect_source", "off")
        manager.save_to_yaml(str(work2))
        again = _manager()
        again.load_from_yaml(str(work2))
        assert again.get_value("thermal.eq_indirect_source") == "off", (
            f"one save-and-reopen turned it into "
            f"{again.get_value('thermal.eq_indirect_source')!r}")
        assert again.validate()[0], again.validate()[1]


def test_the_writer_quotes_anything_yaml_1_1_would_re_read_differently():
    """General, not a word list."""
    _require_ruamel()
    from gui_core.yaml_roundtrip import quote_if_ambiguous, RoundTripDoc
    import yaml as pyyaml
    with tempfile.TemporaryDirectory() as tmp:
        for text in ("off", "on", "yes", "no", "true", "False", "null", "~",
                     "1", "1e5", "3.5", "", "0o17"):
            # Written by the round-trip writer, read back by PyYAML -- which
            # is the actual path: ruamel emits, load_from_yaml reads.
            work = Path(tmp) / "probe.yaml"
            work.write_text("section:\n  k: placeholder\n")
            doc = RoundTripDoc.load(work)
            doc.apply({"section.k": text})
            rendered = doc.dump()
            assert pyyaml.safe_load(rendered)["section"]["k"] == text, (
                f"{text!r} did not survive ruamel-write / PyYAML-read as a "
                f"string: {rendered!r}")
    for text in ("auto", "converged", "data/site.bvh", "Moon"):
        assert quote_if_ambiguous(text) == text
        assert type(quote_if_ambiguous(text)) is str, (
            f"{text!r} was needlessly wrapped, which would requote a line "
            f"nobody asked to change")

    # The quoted wrapper must never escape into the config dict: PyYAML's
    # dumper cannot represent it, and save_run_yaml -- the RUN's config -- is
    # PyYAML.
    manager = _manager()
    manager.set_value("thermal.eq_indirect_source", "off")
    pyyaml.dump(manager._config_for_yaml())


def test_an_int_spelled_double_is_not_rewritten():
    """`scale_factor: 1` must stay `1`, not become `1.0`.

    set_numeric coerces a loaded value to the widget's type, so a file that
    spells a double-valued key as an integer came back as a float and a strict
    type test rewrote the line on a save with NO EDITS. Reproduced: opening a
    config with `scale_factor: 1` and `geothermal_heat_flow: 0` planned
    {'mesh.scale_factor': 1.0, 'thermal.geothermal_heat_flow': 0.0}.
    """
    _require_ruamel()
    from gui_core.yaml_roundtrip import values_equal
    assert values_equal(1, 1.0) and values_equal(1.0, 1)
    # ...but a switch spelled `0` that becomes True IS a change, even though
    # Python says 0 == False.
    assert not values_equal(0, False)
    assert not values_equal(1, True)
    assert values_equal(False, False) and not values_equal(False, True)
    assert not values_equal("20", 20.0)

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "intspell.yaml"
        work.write_text(
            "# keep me\nsimulation:\n  body: Moon\n"
            "mesh:\n  scale_factor: 1\n"
            "thermal:\n  geothermal_heat_flow: 0\n")
        before = work.read_text()
        manager = _manager()
        manager.load_from_yaml(str(work))
        # Emulate the panel's load-time coercion without building one.
        manager.set_value("mesh.scale_factor", 1.0)
        manager.set_value("thermal.geothermal_heat_flow", 0.0)
        assert manager.plan_save_edits() == ({}, []), (
            f"an int-spelled double plans an edit: "
            f"{manager.plan_save_edits()}")
        manager.save_to_yaml(str(work))
        assert work.read_text() == before


def test_save_says_which_file_it_actually_rewrote():
    """Save As over a DIFFERENT existing file is not "comments preserved".

    The target is replaced wholesale by this project's document -- which is
    what Save As means, but the target's own comments are gone, and the console
    said "Comments and layout preserved". Reproduced.
    """
    _require_ruamel()
    with tempfile.TemporaryDirectory() as tmp:
        site_a = Path(tmp) / "siteA.yaml"
        site_b = Path(tmp) / "siteB.yaml"
        site_c = Path(tmp) / "siteC.yaml"
        site_a.write_text("# SITE A\nsimulation:\n  body: Moon\n")
        site_b.write_text("# SITE B\nsimulation:\n  body: Europa\n")

        manager = _manager()
        manager.load_from_yaml(str(site_a))
        manager.set_value("simulation.timesteps", 42)
        assert manager.save_to_yaml(str(site_a)) == "in_place"
        assert manager.save_to_yaml(str(site_c)) == "copied_to_new_file"

        manager = _manager()
        manager.load_from_yaml(str(site_a))
        assert manager.save_to_yaml(str(site_b)) == "replaced_other_file", (
            "overwriting a different existing file is reported as an in-place "
            "edit, so the console claims its comments were preserved")
        assert "SITE B" not in site_b.read_text(), (
            "the fixture did not actually exercise the overwrite")

        assert _manager().save_to_yaml(str(Path(tmp) / "fresh.yaml")) == \
            "expanded"


def test_plan_save_edits_reports_exactly_what_would_change():
    _require_ruamel()
    with tempfile.TemporaryDirectory() as tmp:
        rel = COMMENTED_EXAMPLES[0]
        work = Path(tmp) / Path(rel).name
        shutil.copyfile(REPO_ROOT / rel, work)

        manager = _manager()
        manager.load_from_yaml(str(work))
        assert manager.plan_save_edits() == ({}, []), (
            "a freshly loaded config already plans edits, so 'save with no "
            "changes is a no-op' is not actually true")

        manager.set_value("simulation.timesteps", 999)
        updates, removals = manager.plan_save_edits()
        assert updates == {"simulation.timesteps": 999}, updates
        assert removals == [], removals


TESTS = [
    test_ruamel_is_installed,
    test_open_then_save_with_no_edits_is_byte_identical,
    test_comments_survive_an_edit,
    test_the_byte_diff_is_confined_to_the_edited_key,
    test_an_untouched_key_the_file_omits_is_not_written,
    test_editing_a_gpu_key_does_write_it,
    test_removing_an_override_removes_its_line,
    test_a_new_project_still_saves_and_then_round_trips,
    test_status_is_reported_not_inferred,
    test_opening_a_config_in_the_PANEL_plans_no_edits,
    test_no_edit_save_is_byte_identical_on_EVERY_shipped_config,
    test_one_edit_touches_only_that_key_across_every_config,
    test_a_reflow_the_writer_cannot_avoid_is_REPORTED,
    test_layout_detection_matches_a_KEY_not_a_substring,
    test_comments_deleted_with_their_key_are_not_called_a_reflow,
    test_a_second_save_does_not_repeat_the_first_saves_warning,
    test_no_shipped_config_is_reflowed_by_an_edit,
    test_a_recognized_alias_keeps_the_files_own_spelling,
    test_an_int_spelled_gpu_key_is_not_DELETED_on_an_edit_free_save,
    test_no_shipped_config_is_BLOCKED_by_the_GUIs_own_validate,
    test_the_word_off_survives_a_save_and_reopen,
    test_the_writer_quotes_anything_yaml_1_1_would_re_read_differently,
    test_an_int_spelled_double_is_not_rewritten,
    test_save_says_which_file_it_actually_rewrote,
    test_plan_save_edits_reports_exactly_what_would_change,
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
    print(f"\n[test_gui_comment_roundtrip] {total - failures - skipped}/{total} "
          f"passed, {skipped} skipped")
    if skipped == total:
        print("[test_gui_comment_roundtrip] SKIPPED ENTIRELY -- this gate "
              "proved NOTHING on this machine.")
    if failures:
        print("[test_gui_comment_roundtrip] FAIL")
        return 1
    print("[test_gui_comment_roundtrip] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
