#!/usr/bin/env python3
"""The engine keys Stage 1 promised the GUI can write -- checked at the writer.

A widget on a tab proves nothing on its own: what matters is whether the key
reaches the saved YAML with the value the user chose, and comes back on reload.
``ConfigManager._config_for_yaml`` is the writer and ``load_from_yaml`` is the
reader, so these checks go straight at them. Tab construction is covered
separately by the offscreen panel probe, which needs PyVista and therefore
cannot run in the Python-layer CI job.

The sharpest of these is ``thermophysical.H_override``. The engine has read it
since it was added (``config_parser.cpp:759``) and applies it by setting
``planet.H`` directly (``ptm_thermal_sim.cpp:14778``), so it works on a built-in
body -- but ``ConfigManager.ENGINE_OVERRIDE_KEYS`` was
``("albedo", "emissivity", "ks", "kd")``, and the Thermophysical tab
additionally disabled the H row on every built-in body. There was no write path
at all: a colleague could not change the regolith scale height on the Moon from
this GUI.

Needs PyQt6 (``gui_core`` imports Qt at package import); skipped without it.

Run directly::

    python3 src/python/tests/test_gui_stage1_keys.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "python"))
sys.path.insert(0, str(REPO_ROOT / "gui"))

from ptm.engine_defaults import ENGINE_DEFAULTS                   # noqa: E402


class Skipped(Exception):
    """A check that cannot run here, as distinct from one that failed."""


def _skip(reason):
    try:
        import pytest
    except ImportError:
        raise Skipped(reason) from None
    pytest.skip(reason)


def _manager():
    try:
        from gui_core.config_manager import ConfigManager
    except ImportError as exc:
        _skip(f"the GUI's config manager is unavailable here ({exc}); "
              f"install PyQt6 to run this check")
    return ConfigManager()


def _round_trip(manager):
    """Save what this manager would write, load it back, return the new manager."""
    from gui_core.config_manager import ConfigManager
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "round.yaml"
        manager.save_to_yaml(str(path))
        back = ConfigManager()
        back.load_from_yaml(str(path))
        return back


# --------------------------------------------------------------------------

def test_H_override_is_written_on_a_builtin_body():
    """Uncheck 'use default H' on the Moon and H_override lands in the YAML."""
    manager = _manager()
    manager.set_value("simulation.body", "Moon")
    manager.set_value("thermophysical.use_default_H", False)
    manager.set_value("thermophysical.H", 0.09)

    written = manager._config_for_yaml()["thermophysical"]
    assert "H_override" in written, (
        "a built-in body with a custom H writes no H_override; the engine "
        "reads thermophysical.H_override (config_parser.cpp:759) and applies "
        "it by setting planet.H, so the value is simply being dropped")
    assert written["H_override"] == 0.09, written["H_override"]
    assert "H" not in written, (
        "the plain scalar thermophysical.H is also being written; the engine "
        "ignores it on a built-in body, and writing it makes a dead key look "
        "like a live setting")


def test_H_override_survives_a_round_trip():
    manager = _manager()
    manager.set_value("simulation.body", "Moon")
    manager.set_value("thermophysical.use_default_H", False)
    manager.set_value("thermophysical.H", 0.09)

    back = _round_trip(manager)
    assert back.get_value("thermophysical.H") == 0.09
    assert back.get_value("thermophysical.use_default_H") is False


def test_H_is_in_the_engine_override_set():
    """The tuple the writer and the tab both key off."""
    from gui_core.config_manager import ConfigManager
    assert "H" in ConfigManager.ENGINE_OVERRIDE_KEYS, (
        "ENGINE_OVERRIDE_KEYS is the single source for both the write path and "
        "the tab's enable/disable rule; H missing from it is what made the "
        "field unreachable")
    for key in ConfigManager.ENGINE_OVERRIDE_KEYS:
        assert f"thermophysical.{key}_override" in ENGINE_DEFAULTS, (
            f"ENGINE_OVERRIDE_KEYS claims thermophysical.{key}_override exists "
            f"and the engine does not read it")


def test_conductivity_model_is_written_and_round_trips():
    manager = _manager()
    manager.set_value("simulation.body", "Moon")
    manager.set_value("thermophysical.conductivity_model", "martinez2021")
    written = manager._config_for_yaml()["thermophysical"]
    assert written.get("conductivity_model") == "martinez2021", written
    back = _round_trip(manager)
    assert back.get_value("thermophysical.conductivity_model") == "martinez2021"


def test_the_default_conductivity_model_is_the_engine_s():
    manager = _manager()
    assert manager.get_value("thermophysical.conductivity_model") == \
        ENGINE_DEFAULTS["thermophysical.conductivity_model"]


def test_coupling_mode_is_writable_and_round_trips():
    """PTM-102. The mode was unreachable while the field it gates was visible."""
    manager = _manager()
    assert manager.get_value("radiation.coupling_mode") == \
        ENGINE_DEFAULTS["radiation.coupling_mode"]
    for mode in ("bounded", "fixed", "legacy"):
        manager.set_value("radiation.coupling_mode", mode)
        written = manager._config_for_yaml()["radiation"]
        assert written.get("coupling_mode") == mode, written
        assert _round_trip(manager).get_value("radiation.coupling_mode") == mode


def test_nearfield_quadrature_order_and_mc_builder_are_writable():
    manager = _manager()
    for key, value in (("radiation.nearfield_quadrature_order", 1),
                       ("radiation.use_monte_carlo_vf", True),
                       ("radiation.mc_rays_per_face", 250000)):
        manager.set_value(key, value)
    written = manager._config_for_yaml()["radiation"]
    assert written["nearfield_quadrature_order"] == 1, written
    assert written["use_monte_carlo_vf"] is True, written
    assert written["mc_rays_per_face"] == 250000, written

    back = _round_trip(manager)
    assert back.get_value("radiation.nearfield_quadrature_order") == 1
    assert back.get_value("radiation.use_monte_carlo_vf") is True
    assert back.get_value("radiation.mc_rays_per_face") == 250000


def test_the_initial_condition_group_is_writable():
    """spinup_seed and its two companions reach the YAML and come back."""
    manager = _manager()
    for key in ("thermal.spinup_seed", "thermal.seed_anchor",
                "thermal.analytic_seed_cycles",
                "thermal.seed_anchor_refuse_two_timescale"):
        assert manager.get_value(key) == ENGINE_DEFAULTS[key], key

    manager.set_value("thermal.spinup_seed", "legacy")
    manager.set_value("thermal.seed_anchor", "warm")
    manager.set_value("thermal.analytic_seed_cycles", 24.0)
    manager.set_value("thermal.seed_anchor_refuse_two_timescale", True)

    written = manager._config_for_yaml()["thermal"]
    assert written["spinup_seed"] == "legacy", written
    assert written["seed_anchor"] == "warm", written
    assert written["analytic_seed_cycles"] == 24.0, written
    assert written["seed_anchor_refuse_two_timescale"] is True, written

    back = _round_trip(manager)
    assert back.get_value("thermal.spinup_seed") == "legacy"
    assert back.get_value("thermal.seed_anchor") == "warm"
    assert back.get_value("thermal.analytic_seed_cycles") == 24.0


def test_spinup_seed_legacy_is_no_longer_defeated_by_the_injected_environment():
    """The pair that makes `legacy` mean what it says (PTM-276).

    `spinup_seed: legacy` exists to reproduce a pre-fix baseline
    digit-for-digit. On the GPU path with that seed, an injected
    PTM_EQ_PERFACE_FLUX=1 switches the seed from the first-10-step instantaneous
    flux to the per-face time mean -- the GUI was the one caller that could
    defeat the setting without the user having asked for anything. Exposing the
    control and dropping the injection have to hold together.
    """
    from gui_core.preflight import build_engine_env

    manager = _manager()
    manager.set_value("thermal.spinup_seed", "legacy")
    config = manager.get_config()
    env, _applied = build_engine_env(config, base_env={})
    assert "PTM_EQ_PERFACE_FLUX" not in env, (
        "the GUI still injects PTM_EQ_PERFACE_FLUX into a run that asked for "
        "spinup_seed: legacy, which is exactly what that seed exists to avoid")


def test_the_state_files_are_writable():
    """Warm start and chunk handoff -- without these a GUI run cannot resume."""
    manager = _manager()
    manager.set_value("thermal.initial_state_file", "output/prev/final_state.bin")
    manager.set_value("thermal.final_state_file", "output/here/final_state.bin")
    written = manager._config_for_yaml()["thermal"]
    assert written["initial_state_file"] == "output/prev/final_state.bin", written
    assert written["final_state_file"] == "output/here/final_state.bin", written

    back = _round_trip(manager)
    assert back.get_value("thermal.initial_state_file") == \
        "output/prev/final_state.bin"
    assert back.get_value("thermal.final_state_file") == \
        "output/here/final_state.bin"


def test_both_state_paths_are_absolutized_for_the_run_yaml():
    """The run YAML is written elsewhere, so relative paths must be resolved.

    initial_state_file is checked for existence too (it is in _FILE_FIELDS);
    final_state_file is an output path, so it is absolutized but not checked.
    """
    from gui_core.preflight import absolutize_config_paths

    manager = _manager()
    manager.set_value("thermal.initial_state_file", "output/prev/final_state.bin")
    manager.set_value("thermal.final_state_file", "output/here/final_state.bin")
    resolved = absolutize_config_paths(manager._config_for_yaml(), REPO_ROOT)
    for key in ("initial_state_file", "final_state_file"):
        value = resolved["thermal"][key]
        assert Path(value).is_absolute(), (
            f"thermal.{key} stayed relative ({value!r}); the run YAML lives "
            f"beside the output directory, so the engine would resolve it "
            f"against the wrong place")


def test_the_mesh_placement_keys_are_writable():
    """occluder_file plus the two placement overrides."""
    manager = _manager()
    for key in ("mesh.occluder_file", "mesh.latitude_override",
                "mesh.longitude_override"):
        assert manager.get_value(key) == ENGINE_DEFAULTS[key], key

    manager.set_value("mesh.import_file", "data/meshes/core.ply")
    manager.set_value("mesh.occluder_file", "data/meshes/ring.ply")
    manager.set_value("mesh.latitude_override", -84.223)
    manager.set_value("mesh.longitude_override", 33.502)

    written = manager._config_for_yaml()["mesh"]
    assert written["occluder_file"] == "data/meshes/ring.ply", written
    assert written["latitude_override"] == -84.223, written
    assert written["longitude_override"] == 33.502, written

    back = _round_trip(manager)
    assert back.get_value("mesh.occluder_file") == "data/meshes/ring.ply"
    assert back.get_value("mesh.latitude_override") == -84.223


def test_the_occluder_is_preflight_checked_like_the_core_mesh():
    """It is read at BVH build time, so a cloud placeholder hangs the run."""
    from gui_core import preflight
    assert "mesh.occluder_file" in preflight._FILE_FIELDS, (
        "mesh.occluder_file is not in preflight's file-field list, so it gets "
        "no existence, dataless or cloud-path check -- unlike mesh.import_file, "
        "which the engine reads the same way")


def test_the_occluder_path_is_absolutized_for_the_run_yaml():
    from gui_core.preflight import absolutize_config_paths

    manager = _manager()
    manager.set_value("mesh.occluder_file", "data/meshes/ring.ply")
    resolved = absolutize_config_paths(manager._config_for_yaml(), REPO_ROOT)
    assert Path(resolved["mesh"]["occluder_file"]).is_absolute()


def test_the_beaming_pair_is_writable_and_never_half_set():
    """beaming: true with rms_slope_deg == 0 is FATAL at engine startup."""
    manager = _manager()
    assert manager.get_value("radiation.beaming") is False
    assert manager.get_value("radiation.rms_slope_deg") == 0.0

    manager.set_value("radiation.beaming", True)
    manager.set_value("radiation.rms_slope_deg", 32.5)
    written = manager._config_for_yaml()["radiation"]
    assert written["beaming"] is True, written
    assert written["rms_slope_deg"] == 32.5, written

    back = _round_trip(manager)
    assert back.get_value("radiation.beaming") is True
    assert back.get_value("radiation.rms_slope_deg") == 32.5


def test_the_pre_2026_08_25_implicit_beaming_switch_survives_a_load():
    """A config with a slope and no `beaming` key must not be demoted.

    config_parser.cpp:631 turns beaming ON for such a config and says so. The
    GUI writes `beaming` explicitly, so unless the implicit switch is detected
    at load, the run YAML carries `beaming: false` and silently demotes a
    validated beaming run to Lambertian.

    The detection cannot be done from the merged config -- DEFAULT_CONFIG always
    supplies `beaming` -- which is exactly how the first attempt at this check
    came out dead. It is recorded by load_from_yaml.
    """
    manager_cls = type(_manager())
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "legacy.yaml"
        path.write_text("simulation:\n  body: Moon\n"
                        "radiation:\n  enabled: true\n  rms_slope_deg: 32.5\n")
        manager = manager_cls()
        manager.load_from_yaml(str(path))
        assert manager.legacy_implicit_beaming == 32.5, (
            f"the implicit switch was not detected "
            f"({manager.legacy_implicit_beaming!r})")
        assert manager.get_value("radiation.beaming") is True
        assert manager._config_for_yaml()["radiation"]["beaming"] is True, (
            "the run YAML would hand the engine beaming: false for a config "
            "the engine itself would run with beaming ON")

        # An explicit `beaming: false` next to a slope is NOT the implicit
        # switch, and must be left alone.
        path.write_text("simulation:\n  body: Moon\nradiation:\n"
                        "  enabled: true\n  beaming: false\n  rms_slope_deg: 32.5\n")
        manager = manager_cls()
        manager.load_from_yaml(str(path))
        assert manager.legacy_implicit_beaming is None
        assert manager.get_value("radiation.beaming") is False


def test_a_quoted_rms_slope_does_not_break_the_load():
    """`rms_slope_deg: "20"` is a str after yaml.safe_load.

    The engine's getWithDefault<double> parses it without complaint, so a bare
    comparison in the implicit-beaming detection turned a loadable config into
    "Failed to load project".
    """
    manager_cls = type(_manager())
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "quoted.yaml"
        path.write_text('simulation:\n  body: Moon\nradiation:\n'
                        '  enabled: true\n  rms_slope_deg: "20"\n')
        manager = manager_cls()
        manager.load_from_yaml(str(path))       # must not raise
        assert manager.legacy_implicit_beaming == 20.0
        assert manager.get_value("radiation.beaming") is True

        path.write_text('simulation:\n  body: Moon\nradiation:\n'
                        '  rms_slope_deg: not-a-number\n')
        manager = manager_cls()
        manager.load_from_yaml(str(path))       # must not raise either
        assert manager.legacy_implicit_beaming is None


def test_the_basal_heat_flow_control_is_not_gated_on_the_body():
    """PTM-161 is FIXED in this tree; the key is live on every body.

    It was briefly disabled on built-in bodies on the strength of
    docs/KNOWN_LIMITATIONS.md section 4, which still described the pre-2e06a70
    behavior. Re-measured on the current engine: Moon ico3, 48 steps, Qb -1 vs
    0.05 W/m^2 gives max |dT_min| = 0.8421 K and the run prints
    "[Thermo override] applied at solver: Qb 0.018 -> 0.05 W/m^2".
    """
    manager = _manager()
    manager.set_value("simulation.body", "Moon")
    manager.set_value("thermal.geothermal_heat_flow", 0.05)
    written = manager._config_for_yaml()["thermal"]
    assert written["geothermal_heat_flow"] == 0.05, written

    source = (REPO_ROOT / "src" / "cpp" / "main" / "ptm_thermal_sim.cpp").read_text()
    assert "ovr_Qb" in source, (
        "the engine no longer forwards geothermal_heat_flow as ovr_Qb; if "
        "PTM-161 has regressed, the GUI control and the docs both need to say "
        "so again")

    limits = (REPO_ROOT / "docs" / "KNOWN_LIMITATIONS.md").read_text()
    section = limits.split("## 4.", 1)[1].split("\n## ", 1)[0]
    assert "FIXED" in section, (
        "KNOWN_LIMITATIONS.md section 4 no longer records that PTM-161 is "
        "fixed; a stale limitation is what disabled this control once already")


def test_the_enum_alias_tables_match_the_engine_s_parsers():
    """The engine normalizes and accepts synonyms; a plain findText does not.

    A config saying `spinup_seed: historical` or `Legacy` left the combo showing
    "physical" while the engine ran the legacy seed -- a control that reports a
    value other than the one that runs.
    """
    try:
        from gui_panels.config_panel import (CONDUCTIVITY_MODEL_ALIASES,
                                             COUPLING_MODE_ALIASES,
                                             SEED_ANCHOR_ALIASES,
                                             SPINUP_SEED_ALIASES)
    except ImportError as exc:
        _skip(f"the GUI panels are unavailable here ({exc}); "
              f"install PyQt6 and PyVista to run this check")

    # Spellings taken from eq_seed.cpp and conductivity_model.h.
    for aliases, cases in (
            (SPINUP_SEED_ALIASES,
             {"physical": "physical", "default": "physical",
              "perface": "physical", "legacy": "legacy",
              "historical": "legacy"}),
            (SEED_ANCHOR_ALIASES,
             {"lumped": "lumped", "default": "lumped", "derived": "lumped",
              "warm": "warm", "warmbracket": "warm", "t4": "warm",
              "legacy": "warm"}),
            (CONDUCTIVITY_MODEL_ALIASES,
             {"hayne": "hayne2017", "hayne2017": "hayne2017",
              "default": "hayne2017", "martinez": "martinez2021",
              "martinez2021": "martinez2021",
              "martinezsiegler2021": "martinez2021"})):
        for spelling, canonical in cases.items():
            assert aliases.get(spelling) == canonical, (
                f"{spelling!r} should normalize to {canonical!r}, got "
                f"{aliases.get(spelling)!r}")

    # coupling_mode is compared EXACTLY by the engine and a mismatch is a
    # startup FATAL, so the GUI must not invent aliases it would reject.
    assert set(COUPLING_MODE_ALIASES) == {"bounded", "fixed", "legacy"}
    assert set(COUPLING_MODE_ALIASES.values()) == {"bounded", "fixed", "legacy"}


def test_validate_catches_the_mesh_placement_traps():
    """The engine range-checks neither override and gates both on latitude."""
    manager = _manager()
    manager.set_value("mesh.latitude_override", -95.0)
    ok, messages = manager.validate()
    assert not ok and any("latitude_override" in m for m in messages), messages

    # A longitude with no latitude WARNS but does not block. It used to be a
    # hard error, which locked the launch behind a field the Mesh tab disables
    # as soon as the latitude gate closes -- the user could not reach the
    # control that would clear it. The tab now resets the stale longitude and
    # says so; this is the backstop for a config that arrives that way.
    manager.set_value("mesh.latitude_override", -999.0)
    manager.set_value("mesh.longitude_override", 30.0)
    ok, messages = manager.validate()
    assert ok, f"a longitude with no latitude must not block the launch: {messages}"
    assert any("longitude_override" in m for m in messages), (
        f"...but it must still be said: {messages}")

    manager.set_value("mesh.latitude_override", -84.223)
    ok, messages = manager.validate()
    assert ok, messages


def test_validate_refuses_beaming_without_a_slope():
    """`beaming: true` with `rms_slope_deg: 0` is FATAL at engine startup.

    run_simulation calls validate() first, so without this the GUI cheerfully
    launches a run that dies before doing any work. The tab seeds 32.5 when the
    switch is flipped on, but the slope can be stepped back to zero afterwards
    and a config can arrive with the pair already broken.
    """
    manager = _manager()
    manager.set_value("radiation.enabled", True)
    manager.set_value("radiation.beaming", True)
    manager.set_value("radiation.rms_slope_deg", 0.0)
    ok, messages = manager.validate()
    assert not ok, "validate() accepted the pair the engine refuses at startup"
    assert any("rms_slope_deg" in m for m in messages), messages

    manager.set_value("radiation.rms_slope_deg", 32.5)
    ok, messages = manager.validate()
    assert ok, messages


TESTS = [
    test_H_override_is_written_on_a_builtin_body,
    test_H_override_survives_a_round_trip,
    test_H_is_in_the_engine_override_set,
    test_conductivity_model_is_written_and_round_trips,
    test_the_default_conductivity_model_is_the_engine_s,
    test_coupling_mode_is_writable_and_round_trips,
    test_nearfield_quadrature_order_and_mc_builder_are_writable,
    test_the_initial_condition_group_is_writable,
    test_spinup_seed_legacy_is_no_longer_defeated_by_the_injected_environment,
    test_the_state_files_are_writable,
    test_both_state_paths_are_absolutized_for_the_run_yaml,
    test_the_mesh_placement_keys_are_writable,
    test_the_occluder_is_preflight_checked_like_the_core_mesh,
    test_the_occluder_path_is_absolutized_for_the_run_yaml,
    test_the_beaming_pair_is_writable_and_never_half_set,
    test_the_pre_2026_08_25_implicit_beaming_switch_survives_a_load,
    test_a_quoted_rms_slope_does_not_break_the_load,
    test_the_basal_heat_flow_control_is_not_gated_on_the_body,
    test_validate_refuses_beaming_without_a_slope,
    test_the_enum_alias_tables_match_the_engine_s_parsers,
    test_validate_catches_the_mesh_placement_traps,
]


def main():
    failures = skipped = 0
    for test in TESTS:
        try:
            test()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {test.__name__}\n     {exc}")
        except BaseException as exc:                   # noqa: BLE001
            if type(exc).__name__ != "Skipped":
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                failures += 1
                print(f"ERROR {test.__name__}: {type(exc).__name__}: {exc}")
            else:
                skipped += 1
                print(f"SKIP {test.__name__}\n     {exc}")
        else:
            print(f"PASS {test.__name__}")
    total = len(TESTS)
    tally = f"{total - failures - skipped}/{total} passed"
    if skipped:
        tally += f", {skipped} skipped"
    print(f"\n[test_gui_stage1_keys] {tally}")
    if failures:
        print("[test_gui_stage1_keys] FAIL")
        return 1
    print("[test_gui_stage1_keys] PASS"
          + (f" ({skipped} skipped)" if skipped else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
