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
