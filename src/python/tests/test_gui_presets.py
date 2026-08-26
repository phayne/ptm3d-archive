#!/usr/bin/env python3
"""Items 2.2 and 2.3 / DP-5: the task cards, and the beaming rule they carry.

The old ``ConfigManager.PRESETS`` was unreachable code for months while
``gui/README.md`` advertised the feature (GUI-PRESET-1). Stage 1 deleted the
table; this pins the replacement.

The property that matters most is not "the preset sets the keys". It is that
**a preset never turns beaming on past the facet-scale rule.** The engine
enforces only that ``beaming: true`` carries a positive slope and nothing at
all about the scale of the mesh it is applied to
(``docs/radiation-exchange.md`` §8.4), so a preset that ticked the switch
would be the only thing standing between a user and double-counted roughness,
and it would not be standing.

Run directly::

    python3 src/python/tests/test_gui_presets.py
"""

from __future__ import annotations

import os
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


_APP = None
_PANEL = None
_MANAGER = None


def _panel(scratch):
    global _APP, _PANEL, _MANAGER
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
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import QSettings
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    for scope in (QSettings.Scope.UserScope, QSettings.Scope.SystemScope):
        QSettings.setPath(QSettings.Format.IniFormat, scope, str(scratch))
    from gui_core.config_manager import ConfigManager
    from gui_panels.config_panel import ConfigurationPanel
    _APP = QApplication.instance() or QApplication([])
    _MANAGER = ConfigManager()
    _PANEL = ConfigurationPanel(_MANAGER)
    return _MANAGER, _PANEL


EXPECTED_KEYS = ("lunar_global_diurnal", "lunar_polar_crater",
                 "europa_diurnal", "generic_body", "reproduce_baseline")


# --------------------------------------------------------------------------

def test_the_five_presets_exist_and_are_reachable():
    from gui_core.presets import PRESETS, PRESETS_BY_KEY
    assert tuple(p.key for p in PRESETS) == EXPECTED_KEYS, (
        f"the preset list changed: {[p.key for p in PRESETS]}")
    with tempfile.TemporaryDirectory() as scratch:
        manager, panel = _panel(scratch)
        assert "start" in panel.tabs, (
            "there is no Start page, so the presets are unreachable again -- "
            "which is exactly GUI-PRESET-1")
        assert panel._nav_keys[0] == "start", (
            f"the Start page is not first in the nav rail: {panel._nav_keys}")
        for key in PRESETS_BY_KEY:
            assert key in panel.tabs["start"]._reports, (
                f"{key} has no card")


def test_every_preset_writes_only_engine_keys():
    """A preset that sets a key the engine ignores is a preset that lies."""
    from ptm.engine_config import known_keys
    from gui_core.presets import PRESETS
    known = known_keys()
    # GUI-synthetic keys the writer translates; not engine keys by name.
    synthetic = {"illumination.albedo_source", "mesh.topo_source"}
    bad = []
    for preset in PRESETS:
        for key in preset.sets:
            if key in synthetic or key in known:
                continue
            if any(k.startswith(key + ".") for k in known):
                continue
            bad.append(f"{preset.key}: {key}")
    assert not bad, f"presets set key(s) the engine never reads: {bad}"


def test_a_preset_reports_exactly_what_it_set():
    from gui_core.presets import PRESETS_BY_KEY, apply_preset
    manager = _manager()
    preset = PRESETS_BY_KEY["lunar_global_diurnal"]
    result = apply_preset(manager, preset)
    for key, value in preset.sets.items():
        assert result.applied.get(key) == value, (
            f"{key} not reported as applied: {result.applied}")
        assert manager.get_value(key) == value, (
            f"{key} reported but not actually set")
    # beaming is reported even though it is not in `sets`: leaving a stale
    # slope behind hands the user a config the engine refuses at startup.
    assert result.applied["radiation.beaming"] is False
    assert result.applied["radiation.rms_slope_deg"] == 0.0
    assert result.summary_lines()[0].startswith("Set ")


def test_the_crater_preset_refuses_beaming_on_a_millimeter_mesh():
    """The single most important line in this file.

    A mesh whose facets ARE the roughness scale already resolves it. Turning
    Smith beaming on there counts it twice, and no engine check would catch it.
    """
    from gui_core.presets import PRESETS_BY_KEY, apply_preset
    with tempfile.TemporaryDirectory() as tmp:
        # A tiny ASCII PLY with millimeter-scale triangles.
        ply = Path(tmp) / "mm.ply"
        ply.write_text(
            "ply\nformat ascii 1.0\nelement vertex 3\n"
            "property float x\nproperty float y\nproperty float z\n"
            "element face 1\nproperty list uchar int vertex_index\n"
            "end_header\n"
            "0 0 0\n0.001 0 0\n0 0.001 0\n"
            "3 0 1 2\n")
        manager = _manager()
        manager.set_value("mesh.import_file", str(ply))
        result = apply_preset(manager, PRESETS_BY_KEY["lunar_polar_crater"],
                              project_root=Path(tmp))
        assert manager.get_value("radiation.beaming") is False, (
            "the preset turned beaming ON for a mesh at the roughness scale")
        assert any(k == "radiation.beaming" for k, _ in result.refused), (
            f"the refusal was not reported: {result.refused}")
        reason = dict(result.refused)["radiation.beaming"]
        assert "mm" in reason or "facet scale" in reason, (
            f"the refusal does not carry the measured facet scale: {reason}")
        # ...and it still set up the rest of the crater run.
        assert manager.get_value("radiation.enabled") is True
        assert manager.get_value("radiation.coupling_mode") == "bounded"


def test_the_crater_preset_turns_beaming_on_when_the_rule_passes():
    from gui_core.presets import (PRESETS_BY_KEY, apply_preset,
                                  LUNAR_RMS_SLOPE_DEG)
    with tempfile.TemporaryDirectory() as tmp:
        # 100 m triangles: far above the millimeter-to-centimeter roughness
        # band, so the rule allows without an acknowledgment.
        ply = Path(tmp) / "coarse.ply"
        ply.write_text(
            "ply\nformat ascii 1.0\nelement vertex 3\n"
            "property float x\nproperty float y\nproperty float z\n"
            "element face 1\nproperty list uchar int vertex_index\n"
            "end_header\n"
            "0 0 0\n100 0 0\n0 100 0\n"
            "3 0 1 2\n")
        manager = _manager()
        manager.set_value("mesh.import_file", str(ply))
        result = apply_preset(manager, PRESETS_BY_KEY["lunar_polar_crater"],
                              project_root=Path(tmp))
        assert manager.get_value("radiation.beaming") is True, (
            f"beaming stayed off on a 100 m mesh: {result.refused}")
        assert manager.get_value("radiation.rms_slope_deg") == \
            LUNAR_RMS_SLOPE_DEG
        assert result.applied["radiation.rms_slope_deg"] == LUNAR_RMS_SLOPE_DEG


def test_europa_and_generic_body_presets_never_ask_for_beaming():
    """w = 32.5 deg is a LUNAR regolith calibration."""
    from gui_core.presets import PRESETS_BY_KEY, apply_preset
    for key in ("europa_diurnal", "generic_body"):
        manager = _manager()
        result = apply_preset(manager, PRESETS_BY_KEY[key])
        assert manager.get_value("radiation.beaming") is False, key
        assert manager.get_value("radiation.rms_slope_deg") == 0.0, key
        assert any("lunar" in n.lower() or "Lunar" in n for n in result.notes), (
            f"{key} does not say why beaming is off: {result.notes}")


def test_every_preset_leaves_a_config_the_gui_accepts():
    """A preset that produces an invalid config is worse than no preset."""
    from gui_core.presets import PRESETS, apply_preset
    for preset in PRESETS:
        if preset.key == "reproduce_baseline":
            continue                       # it sets nothing; it reads a file
        manager = _manager()
        apply_preset(manager, preset)
        valid, messages = manager.validate()
        assert valid, f"{preset.key} produced an invalid config: {messages}"


def test_a_preset_config_is_accepted_by_test_config_parity_rules():
    """Every key a preset leaves behind must be one the engine reads."""
    from ptm.engine_config import unknown_keys
    from gui_core.presets import PRESETS, apply_preset
    for preset in PRESETS:
        manager = _manager()
        apply_preset(manager, preset)
        dead = unknown_keys(manager._config_for_yaml())
        assert not dead, f"{preset.key} left dead keys: {dead}"


def test_reproduce_a_baseline_reads_the_runs_own_record():
    """No table of remembered baselines: read what the run actually used."""
    from gui_core.presets import read_baseline, compare_binary
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp) / "baseline_run"
        run.mkdir()
        (run / "config_used.yaml").write_text(
            "simulation:\n  body: Moon\n  timesteps: 77\n"
            "thermal:\n  spinup_seed: legacy\n"
            "engine_env: {}\n")
        (run / "provenance.txt").write_text(
            "engine_git_sha=deadbeef01\n"
            "engine_binary_md5=abc123\n"
            "body=Moon\n")
        config, provenance, path = read_baseline(run)
        assert config["simulation"]["timesteps"] == 77
        assert provenance["engine_binary_md5"] == "abc123"
        assert path.name == "config_used.yaml"

        same = compare_binary(provenance, "abc123")
        assert "IS the baseline" in same, same
        diff = compare_binary(provenance, "999999")
        assert "DIFFERENT BINARY" in diff, diff
        unknown = compare_binary(provenance, "")
        assert "cannot be compared" in unknown, unknown

    # A directory with no record must SAY so, not quietly do nothing.
    with tempfile.TemporaryDirectory() as tmp:
        try:
            read_baseline(tmp)
        except FileNotFoundError as exc:
            assert "config_used.yaml" in str(exc)
        else:
            raise AssertionError(
                "read_baseline accepted a directory with no run record")


def test_the_baseline_card_honors_the_files_engine_env():
    """PTM-276 could not have happened through this path.

    The baseline's own `engine_env: {}` means "no fix flags", and Stage 1's
    load rule takes a present block at its word. If the card re-injected the
    GUI's recommended set, the reproduction would run different illumination
    physics than the baseline did.
    """
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp) / "run"
        run.mkdir()
        (run / "config_used.yaml").write_text(
            "simulation:\n  body: Moon\n"
            "thermal:\n  spinup_seed: legacy\n"
            "engine_env: {}\n")
        manager = _manager()
        manager.load_from_yaml(str(run / "config_used.yaml"))
        assert manager.get_value("engine_env") == {}, (
            f"the baseline's empty engine_env was re-populated: "
            f"{manager.get_value('engine_env')}")
        assert manager.get_value("thermal.spinup_seed") == "legacy"


def test_loading_a_baseline_detaches_the_users_project_file():
    """Otherwise the next Ctrl+S writes the BASELINE over the user's own file.

    `apply_baseline` calls `load_from_yaml` on the run's `config_used.yaml`,
    which repoints the round-trip document. Nothing else told the main window,
    so `save_to_yaml(current_project_file)` dumped the baseline's document --
    comments and all -- over the user's project, while the console reported
    "Comments and layout preserved". Reproduced before `project_replaced`
    existed; the probe still runs that arm as the negative control.
    """
    with tempfile.TemporaryDirectory() as scratch:
        manager, panel = _panel(scratch)
        run = Path(scratch) / "baseline_run"
        run.mkdir(exist_ok=True)
        (run / "config_used.yaml").write_text(
            "# BASELINE\nsimulation:\n  body: Moon\n  timesteps: 77\n")
        (run / "provenance.txt").write_text("engine_binary_md5=abc\n")

        seen = []
        panel.project_replaced.connect(seen.append)
        panel.tabs["start"].apply_baseline(str(run))
        assert seen, (
            "project_replaced was not emitted, so the main window cannot know "
            "its save target no longer matches the loaded document")
        assert seen[0].endswith("config_used.yaml"), seen

        # ...and a directory with no record must not emit it at all.
        seen.clear()
        empty = Path(scratch) / "not_a_run"
        empty.mkdir(exist_ok=True)
        panel.tabs["start"].apply_baseline(str(empty))
        assert not seen, (
            "a directory with no config_used.yaml detached the project file "
            "anyway")
        assert "Cannot use this directory" in \
            panel.tabs["start"].report_text("reproduce_baseline")
        panel.tabs["start"].apply_preset("lunar_global_diurnal")


def test_the_card_applies_through_the_panel_and_the_tabs_follow():
    """A card writes a GROUP of keys; every tab must re-read afterward."""
    with tempfile.TemporaryDirectory() as scratch:
        manager, panel = _panel(scratch)
        start = panel.tabs["start"]
        result = start.apply_preset("europa_diurnal")
        assert result.applied["simulation.body"] == "Europa"
        combo = panel.tabs["simulation"].widgets.get("simulation.body")
        assert combo is not None and combo.currentText() == "Europa", (
            f"the Simulation tab still shows "
            f"{combo.currentText() if combo else None!r} after the card set "
            f"Europa; the next widget event would write the stale value back")
        assert "Set " in start.report_text("europa_diurnal"), (
            "the card did not report what it set")
        # Put it back so a later test in this process starts from the Moon.
        start.apply_preset("lunar_global_diurnal")


TESTS = [
    test_the_five_presets_exist_and_are_reachable,
    test_every_preset_writes_only_engine_keys,
    test_a_preset_reports_exactly_what_it_set,
    test_the_crater_preset_refuses_beaming_on_a_millimeter_mesh,
    test_the_crater_preset_turns_beaming_on_when_the_rule_passes,
    test_europa_and_generic_body_presets_never_ask_for_beaming,
    test_every_preset_leaves_a_config_the_gui_accepts,
    test_a_preset_config_is_accepted_by_test_config_parity_rules,
    test_reproduce_a_baseline_reads_the_runs_own_record,
    test_the_baseline_card_honors_the_files_engine_env,
    test_loading_a_baseline_detaches_the_users_project_file,
    test_the_card_applies_through_the_panel_and_the_tabs_follow,
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
    print(f"\n[test_gui_presets] {total - failures - skipped}/{total} passed, "
          f"{skipped} skipped")
    if skipped == total:
        print("[test_gui_presets] SKIPPED ENTIRELY -- proved NOTHING here.")
    if failures:
        print("[test_gui_presets] FAIL")
        return 1
    print("[test_gui_presets] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
