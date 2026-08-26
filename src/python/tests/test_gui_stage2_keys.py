#!/usr/bin/env python3
"""Stage-2 items 2.5-2.9: the keys they promise, and the one that must NOT be written.

Items 2.5 (`gpu:` guardrails), 2.6 (ice stability), 2.7 (equilibration
convergence), 2.8 (GPU device and batch controls) and 2.9 (depth grid,
read-only). A key with a control that does not reach the YAML is worse than no
control: it shows a number the run will not use.

The sharpest case, and the reason DP-1(b) had to land first:
``gpu.min_vram_margin_gib`` is the only key in the engine's 184-key surface
whose semantics depend on whether it was WRITTEN. The GUI must be able to leave
the whole ``gpu:`` section out -- on the round-trip path AND on the full dump
that a RUN writes -- and must write the key when, and only when, the user asks.

Run directly::

    python3 src/python/tests/test_gui_stage2_keys.py
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


def _manager():
    from gui_core.config_manager import ConfigManager
    return ConfigManager()


#: Every engine key Stage 2 promises a control for.
STAGE2_KEYS = [
    # 2.7 equilibration convergence
    "thermal.equilibration_target_K",
    "thermal.equilibration_stop_on_certificate",
    "thermal.equilibration_cycles",
    "thermal.eq_indirect_passes",
    "thermal.eq_indirect_source",
    "thermal.eq_indirect_flux_file",
    # 2.9 depth grid (read-only, but still a control)
    "thermal.thermal_b", "thermal.thermal_m", "thermal.thermal_n",
    # 2.6 ice stability
    "thermal.compute_ice_stability",
    "thermal.ice_stability.flux_threshold_kgm2s",
    "thermal.ice_stability.geothermal_gradient_K_per_m",
    "thermal.ice_stability.grain_hop_length_um",
    "thermal.ice_stability.n_depth_subsamples",
    "thermal.ice_stability.sublimation_enthalpy_kJmol",
    # 2.8 GPU device and batch controls
    "thermal.gpu_devices", "thermal.gpu_device_ids", "thermal.gpu_batch_size",
    "thermal.gpu_inter_batch_pause_ms", "thermal.auto_calibrate_gpu",
    "radiation.gpu_devices", "radiation.gpu_device_ids",
    "radiation.gpu_batch_size", "radiation.gpu_inter_batch_pause_ms",
    "illumination.use_gpu", "illumination.gpu_devices",
    "illumination.gpu_device_ids", "illumination.gpu_inter_batch_pause_ms",
    "illumination.bvh_cache_file",
    # 2.5 gpu guardrails
    "gpu.min_vram_margin_gib", "gpu.placement_policy",
    "gpu.placement_margin_gib", "gpu.thin_margin_policy",
    "gpu.hard_min_vram_margin_gib",
]


# --------------------------------------------------------------------------

def test_every_stage2_key_has_a_control():
    with tempfile.TemporaryDirectory() as scratch:
        manager, panel = _panel(scratch)
        bound = set()
        for tab in panel.tabs.values():
            bound.update(tab.widgets)
        missing = [k for k in STAGE2_KEYS if k not in bound]
        assert not missing, (
            f"{len(missing)} Stage-2 key(s) have no control: {missing}")


def test_every_stage2_key_is_one_the_engine_reads():
    from ptm.engine_config import known_keys
    known = known_keys()
    unknown = [k for k in STAGE2_KEYS if k not in known]
    assert not unknown, (
        f"controls for key(s) the engine never reads: {unknown}")


def test_the_gpu_section_is_absent_from_a_default_config():
    """DP-8's whole safety property, on BOTH writers."""
    manager = _manager()
    written = manager._config_for_yaml()
    assert "gpu" not in written, (
        f"a default config writes a gpu: section ({written.get('gpu')}). "
        f"gpu.min_vram_margin_gib is enforcing on every platform WHENEVER IT "
        f"IS PRESENT, so this would change GPU policy for every config the "
        f"GUI touches, on every machine.")

    # ...and on the full dump the RUN writes, which is the one the engine
    # actually reads. This is the path a round-trip test would not cover.
    with tempfile.TemporaryDirectory() as tmp:
        run_yaml = Path(tmp) / "config.yaml"
        manager.save_run_yaml(str(run_yaml), Path(tmp))
        import yaml
        data = yaml.safe_load(run_yaml.read_text()) or {}
        assert "gpu" not in data, (
            f"the RUN yaml carries a gpu: section: {data.get('gpu')}")


def test_the_vram_floor_is_written_when_and_only_when_asked():
    manager = _manager()
    engine_default = 2.0

    # (a) same value, not asked for -> absent
    manager.set_value("gpu.min_vram_margin_gib", engine_default)
    assert "gpu" not in manager._config_for_yaml()

    # (b) same value, explicitly asked for -> written. This is the case a
    #     value-based rule alone gets WRONG: writing 2.0 is meaningful.
    manager.set_value("gpu.margin_explicit", True)
    written = manager._config_for_yaml().get("gpu", {})
    assert written.get("min_vram_margin_gib") == engine_default, (
        f"ticking 'enforce on every platform' did not write the floor: "
        f"{written}")
    assert "margin_explicit" not in written, (
        "the GUI-synthetic margin_explicit key leaked into the YAML")

    # (c) a different value -> written whether or not it was asked for
    manager.set_value("gpu.margin_explicit", False)
    manager.set_value("gpu.min_vram_margin_gib", 3.5)
    written = manager._config_for_yaml().get("gpu", {})
    assert written.get("min_vram_margin_gib") == 3.5, written

    # (d) a non-default policy key alone -> only that key
    manager = _manager()
    manager.set_value("gpu.placement_policy", "disjoint")
    written = manager._config_for_yaml().get("gpu", {})
    assert written == {"placement_policy": "disjoint"}, (
        f"writing one guardrail dragged others along: {written}")


def test_loading_a_config_that_wrote_the_floor_keeps_it_written():
    """Otherwise the first save silently demotes the run to advisory."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "explicit.yaml"
        path.write_text(
            "simulation:\n  body: Moon\n"
            "gpu:\n  min_vram_margin_gib: 2.0\n")
        manager = _manager()
        manager.load_from_yaml(str(path))
        assert manager.get_value("gpu.margin_explicit") is True, (
            "a config that WROTE the floor was not recorded as explicit")
        written = manager._config_for_yaml().get("gpu", {})
        assert written.get("min_vram_margin_gib") == 2.0, (
            f"saving would drop the floor this config asked for: {written}")


def test_device_ids_round_trip_as_a_sequence():
    """The engine reads this key only when it IS a sequence."""
    with tempfile.TemporaryDirectory() as tmp:
        manager = _manager()
        manager.set_value("thermal.gpu_device_ids", [0, 1])
        written = manager._config_for_yaml()
        assert written["thermal"]["gpu_device_ids"] == [0, 1]

        path = Path(tmp) / "ids.yaml"
        manager.save_to_yaml(str(path))
        again = _manager()
        again.load_from_yaml(str(path))
        assert again.get_value("thermal.gpu_device_ids") == [0, 1]


def test_an_empty_device_list_is_not_written():
    manager = _manager()
    written = manager._config_for_yaml()
    for section in ("thermal", "radiation", "illumination"):
        assert "gpu_device_ids" not in written[section], (
            f"{section}.gpu_device_ids: [] is written into every config; it "
            f"looks like a setting and is not")


def test_a_malformed_device_list_is_refused_not_ignored():
    manager = _manager()
    manager.set_value("radiation.gpu_device_ids", "zero, one")
    valid, messages = manager.validate()
    assert not valid, (
        "a malformed gpu_device_ids validated as fine. The engine reads the "
        "key only as a sequence, so it would be IGNORED and the run would "
        "silently use a different device set than the panel shows.")
    assert any("gpu_device_ids" in m for m in messages), messages


def test_the_gpu_policy_enums_are_the_engines_own():
    """A GUI enum the engine refuses is a startup FATAL the user cannot see."""
    manager = _manager()
    manager.set_value("gpu.placement_policy", "sometimes")
    valid, messages = manager.validate()
    assert not valid and any("placement_policy" in m for m in messages), messages

    manager = _manager()
    manager.set_value("gpu.thin_margin_policy", "maybe")
    valid, messages = manager.validate()
    assert not valid and any("thin_margin_policy" in m for m in messages), messages

    # The separate band must be non-empty, which the engine also refuses.
    manager = _manager()
    manager.set_value("gpu.hard_min_vram_margin_gib", 4.0)
    valid, messages = manager.validate()
    assert not valid and any("hard_min_vram_margin_gib" in m for m in messages), \
        messages


def test_the_hard_floor_rule_is_the_engines_rule():
    """Transcribed from ptm_thermal_sim.cpp:14365, not recalled.

    The first spelling here was `hard >= floor > 0`, wrong in BOTH directions
    and reproduced against the binary:
      * hard == floor == 2.0 -- the engine ACCEPTS it, the GUI raised a hard
        error that blocked the launch while asserting "the engine refuses this
        at startup". A false statement about a valid config.
      * floor = 0 (the documented pre-2026-08-22 behavior, which the tooltip
        invites) with hard = 0.25 -- the engine REFUSES it, the GUI passed it,
        which is the "saves a launch" case the check exists for.
    """
    # equal is legal
    manager = _manager()
    manager.set_value("gpu.min_vram_margin_gib", 2.0)
    manager.set_value("gpu.hard_min_vram_margin_gib", 2.0)
    valid, messages = manager.validate()
    assert valid, f"hard == floor is accepted by the engine: {messages}"

    # hard above floor is not
    manager.set_value("gpu.hard_min_vram_margin_gib", 2.5)
    valid, messages = manager.validate()
    assert not valid and any("hard_min_vram_margin_gib" in m for m in messages), \
        messages

    # floor = 0 with a positive hard floor is refused by the engine
    manager = _manager()
    manager.set_value("gpu.min_vram_margin_gib", 0.0)
    manager.set_value("gpu.hard_min_vram_margin_gib", 0.25)
    valid, messages = manager.validate()
    assert not valid, (
        "floor = 0 with hard = 0.25 validated fine; the engine refuses it at "
        "startup")

    # negatives are refused, as the engine does
    for key in ("gpu.min_vram_margin_gib", "gpu.hard_min_vram_margin_gib",
                "gpu.placement_margin_gib"):
        manager = _manager()
        manager.set_value(key, -1.0)
        valid, messages = manager.validate()
        assert not valid and any(key.split(".")[1] in m for m in messages), (
            f"{key} = -1 validated fine: {messages}")


def test_eq_indirect_source_matches_the_engines_valid_set():
    """auto | converged | off -- taken from the engine's own refusal text."""
    manager = _manager()
    for good in ("auto", "converged", "off"):
        manager.set_value("thermal.eq_indirect_source", good)
        valid, messages = manager.validate()
        assert valid, f"{good} rejected: {messages}"
    manager.set_value("thermal.eq_indirect_source", "internal")
    valid, messages = manager.validate()
    assert not valid, (
        "an eq_indirect_source the engine refuses at startup validated fine")


def test_ice_parameters_are_disabled_until_the_product_is_on():
    with tempfile.TemporaryDirectory() as scratch:
        manager, panel = _panel(scratch)
        tab = panel.tabs["thermal"]
        tab.widgets["thermal.compute_ice_stability"].setChecked(False)
        for key in tab._ice_dependent_keys:
            assert not tab.widgets[key].isEnabled(), (
                f"{key} is editable while ice stability is off, and the run "
                f"will not read it")
        tab.widgets["thermal.compute_ice_stability"].setChecked(True)
        for key in tab._ice_dependent_keys:
            assert tab.widgets[key].isEnabled(), f"{key} stayed disabled"
        tab.widgets["thermal.compute_ice_stability"].setChecked(False)


def test_the_depth_grid_is_display_only():
    """Standing instruction: do not retune thermal_b."""
    with tempfile.TemporaryDirectory() as scratch:
        manager, panel = _panel(scratch)
        tab = panel.tabs["thermal"]
        for key in ("thermal.thermal_b", "thermal.thermal_m",
                    "thermal.thermal_n"):
            widget = tab.widgets[key]
            assert widget.isReadOnly(), (
                f"{key} is editable; the standing instruction is to display "
                f"the depth grid, not offer it")


def test_a_configs_own_depth_grid_is_displayed():
    """Read-only must not mean 'always shows the default'."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "deep.yaml"
        path.write_text(
            "simulation:\n  body: Moon\nthermal:\n  thermal_b: 30\n")
        with tempfile.TemporaryDirectory() as scratch:
            manager, panel = _panel(scratch)
            manager.load_from_yaml(str(path))
            panel.load_from_config()
            shown = panel.tabs["thermal"].widgets["thermal.thermal_b"].value()
            assert shown == 30, (
                f"a config with thermal_b: 30 displays {shown}; the read-only "
                f"grid shows the default instead of the run's own value")


def test_every_stage2_key_survives_a_save_load_round_trip():
    """A control whose value does not reach the YAML shows a lie."""
    import yaml
    probe_values = {
        "thermal.equilibration_target_K": 0.25,
        "thermal.equilibration_stop_on_certificate": True,
        "thermal.equilibration_cycles": 40,
        "thermal.eq_indirect_passes": 3,
        "thermal.eq_indirect_source": "converged",
        "thermal.compute_ice_stability": True,
        "thermal.ice_stability.grain_hop_length_um": 50.0,
        "thermal.ice_stability.n_depth_subsamples": 8,
        "thermal.gpu_devices": 4,
        "thermal.gpu_batch_size": 2048,
        "thermal.gpu_inter_batch_pause_ms": 250,
        "thermal.auto_calibrate_gpu": False,
        "radiation.gpu_devices": 2,
        "radiation.gpu_batch_size": 100000,
        "radiation.gpu_inter_batch_pause_ms": 100,
        "illumination.use_gpu": True,
        "illumination.gpu_devices": 2,
        "illumination.gpu_inter_batch_pause_ms": 50,
        "gpu.placement_policy": "disjoint",
        "gpu.thin_margin_policy": "refuse",
        "gpu.placement_margin_gib": 6.0,
        "gpu.hard_min_vram_margin_gib": 0.5,
    }
    with tempfile.TemporaryDirectory() as tmp:
        manager = _manager()
        for key, value in probe_values.items():
            manager.set_value(key, value)
        path = Path(tmp) / "probe.yaml"
        manager.save_to_yaml(str(path))
        raw = yaml.safe_load(path.read_text()) or {}

        again = _manager()
        again.load_from_yaml(str(path))
        wrong = []
        for key, value in probe_values.items():
            got = again.get_value(key)
            if got != value:
                wrong.append(f"{key}: wrote {value!r}, read back {got!r}")
        assert not wrong, "\n  ".join(["round trip lost values:"] + wrong)
        assert raw.get("gpu", {}).get("placement_policy") == "disjoint", raw


def test_ice_stability_on_the_cpu_path_is_disclosed():
    """The product only exists on the GPU solver. Say so BEFORE the launch.

    Measured 2026-08-25 (Moon ico2, 4 steps): the ice-stability accumulator is
    handed only to runThermalModelsForAllFaces_GPU_Persistent
    (ptm_thermal_sim.cpp:17320). On the CPU paths the run completes, prints
    "[Ice stability] WARNING: requested but accumulator not ready", and writes
    NO ice_depth.bin. Same request with thermal.use_gpu: true wrote
    statistics/ice_depth.bin (320 finite, 0.0..0.5 m).

    A warning, not an error: the run is valid, it just will not make the thing
    that was asked for -- which is exactly the silence worth breaking early.
    """
    manager = _manager()
    manager.set_value("thermal.compute_ice_stability", True)
    manager.set_value("thermal.use_gpu", False)
    valid, messages = manager.validate()
    assert valid, f"this must warn, not block: {messages}"
    assert any("ice_depth" in m and "use_gpu" in m for m in messages), (
        f"no warning that the CPU path writes no ice product: {messages}")

    manager.set_value("thermal.use_gpu", True)
    _, messages = manager.validate()
    assert not any("ice_depth" in m for m in messages), (
        f"the warning fires on the GPU path, where the product IS produced: "
        f"{messages}")

    # ...and the control itself says it, not only the launch dialog.
    with tempfile.TemporaryDirectory() as scratch:
        panel_manager, panel = _panel(scratch)
        tab = panel.tabs["thermal"]
        tab.widgets["thermal.use_gpu"].setChecked(False)
        tab.widgets["thermal.compute_ice_stability"].setChecked(True)
        assert "GPU thermal solver" in tab.ice_note.text(), tab.ice_note.text()
        tab.widgets["thermal.use_gpu"].setChecked(True)
        assert "ice_depth.bin" in tab.ice_note.text(), tab.ice_note.text()
        tab.widgets["thermal.compute_ice_stability"].setChecked(False)
        tab.widgets["thermal.use_gpu"].setChecked(False)


def test_the_illumination_gpu_control_carries_its_measurement():
    """Exposed WITH the reason it is off, not as a bare checkbox."""
    with tempfile.TemporaryDirectory() as scratch:
        manager, panel = _panel(scratch)
        widget = panel.tabs["illumination"].widgets["illumination.use_gpu"]
        tip = widget.toolTip()
        assert "faster" in tip and "correct" in tip, (
            f"the illumination GPU control does not say why it is off by "
            f"default: {tip!r}")


TESTS = [
    test_every_stage2_key_has_a_control,
    test_every_stage2_key_is_one_the_engine_reads,
    test_the_gpu_section_is_absent_from_a_default_config,
    test_the_vram_floor_is_written_when_and_only_when_asked,
    test_loading_a_config_that_wrote_the_floor_keeps_it_written,
    test_device_ids_round_trip_as_a_sequence,
    test_an_empty_device_list_is_not_written,
    test_a_malformed_device_list_is_refused_not_ignored,
    test_the_gpu_policy_enums_are_the_engines_own,
    test_the_hard_floor_rule_is_the_engines_rule,
    test_eq_indirect_source_matches_the_engines_valid_set,
    test_ice_parameters_are_disabled_until_the_product_is_on,
    test_the_depth_grid_is_display_only,
    test_a_configs_own_depth_grid_is_displayed,
    test_every_stage2_key_survives_a_save_load_round_trip,
    test_ice_stability_on_the_cpu_path_is_disclosed,
    test_the_illumination_gpu_control_carries_its_measurement,
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
    print(f"\n[test_gui_stage2_keys] {total - failures - skipped}/{total} "
          f"passed, {skipped} skipped")
    if skipped == total:
        print("[test_gui_stage2_keys] SKIPPED ENTIRELY -- proved NOTHING here.")
    if failures:
        print("[test_gui_stage2_keys] FAIL")
        return 1
    print("[test_gui_stage2_keys] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
