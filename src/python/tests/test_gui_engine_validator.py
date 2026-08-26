#!/usr/bin/env python3
"""Item 2.4 / DP-6: the GUI reports the ENGINE's refusals, in the engine's words.

``ConfigManager.validate`` is a second, weaker copy of the engine's rules, and
the two disagreed within this very session: the first draft of the
``eq_indirect_source`` check invented a valid set (``auto|none|file|internal``)
that would have refused ``converged`` -- the one non-default value the engine
accepts. A second copy of a rule is a second place for it to be wrong.

So `Simulation -> Check Configuration` hands the config to
``ptm_thermal_sim --validate-config`` and shows what comes back. The property
that must never break: an engine that **cannot** be asked is reported as NOT
CHECKED, never as checked-and-fine. Three of the four states this returns are
some flavor of "I could not do that", and every one of them has to be visible.

Run directly::

    python3 src/python/tests/test_gui_engine_validator.py
"""

from __future__ import annotations

import os
import stat
import sys
import tempfile
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "python"))
sys.path.insert(0, str(REPO_ROOT / "gui"))


class Skip(Exception):
    pass


def _preflight():
    try:
        from gui_core import preflight
    except ImportError as exc:                     # pragma: no cover
        raise Skip(f"the GUI's preflight module is unavailable here ({exc})")
    return preflight


def _manager():
    from gui_core.config_manager import ConfigManager
    return ConfigManager()


def _engine():
    from ptm import paths
    exe = paths.find_engine_executable("ptm_thermal_sim",
                                       search_roots=[REPO_ROOT])
    if exe is None:
        raise Skip("no ptm_thermal_sim in build/bin; build it with: "
                   "cmake --build build --target ptm_thermal_sim")
    return Path(exe).resolve()


#: Prepended to every stub that is meant to PASS the capability probe. The
#: probe runs `--help` before the config is handed over (see
#: test_an_engine_without_the_flag_is_never_handed_the_config), so a stub that
#: does not answer it never reaches the branch under test.
_SUPPORTS_FLAG = """
    import sys
    if "--help" in sys.argv:
        print("  --validate-config   Parse and check the config, then exit")
        sys.exit(0)
"""


def _stub(tmp, body, supports_flag=True, name="fake_engine"):
    """A fake engine, so the unavailable/refused paths are testable with no build."""
    path = Path(tmp) / name
    prologue = textwrap.dedent(_SUPPORTS_FLAG) if supports_flag else ""
    # THIS interpreter, not `env python3`: the stub needs the same PyYAML the
    # test has, and /usr/bin/python3 on macOS does not carry it.
    path.write_text(f"#!{sys.executable}\n" + prologue
                    + textwrap.dedent(body))
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


# --------------------------------------------------------------------------

def test_a_good_config_is_accepted_by_the_real_engine():
    preflight = _preflight()
    exe = _engine()
    manager = _manager()
    manager.set_value("simulation.timesteps", 4)
    manager.set_value("simulation.duration_days", 1.0)
    manager.set_value("mesh.resolution", 2)
    status, text = preflight.engine_validate_config(
        manager.get_config(), exe, REPO_ROOT)
    assert status == "accepted", f"{status}: {text[-1500:]}"
    assert "[validate-config]" in text


def test_the_engines_refusal_reaches_the_user_verbatim():
    preflight = _preflight()
    exe = _engine()
    manager = _manager()
    manager.set_value("simulation.timesteps", 4)
    manager.set_value("mesh.resolution", 2)
    # A pair the ENGINE refuses at startup. Set through the config directly so
    # this measures the engine's rule, not the GUI's copy of it.
    manager.set_value("radiation.beaming", True)
    manager.set_value("radiation.rms_slope_deg", 0.0)
    status, text = preflight.engine_validate_config(
        manager.get_config(), exe, REPO_ROOT)
    assert status == "refused", f"{status}: {text[-1500:]}"
    assert "radiation.beaming = true requires" in text, (
        f"the engine's own wording did not survive: {text[-600:]}")


def test_a_disagreement_between_the_two_checkers_is_visible():
    """The engine is the authority; the GUI must not paper over a mismatch.

    `eq_indirect_source: converged` is accepted by the engine. If the GUI's own
    copy of that rule ever refuses it again, this fails -- which is the point.
    """
    preflight = _preflight()
    exe = _engine()
    manager = _manager()
    manager.set_value("simulation.timesteps", 4)
    manager.set_value("mesh.resolution", 2)
    manager.set_value("thermal.eq_indirect_source", "converged")
    gui_valid, gui_messages = manager.validate()
    status, text = preflight.engine_validate_config(
        manager.get_config(), exe, REPO_ROOT)
    assert status == "accepted", f"the engine refused it: {text[-800:]}"
    assert gui_valid, (
        f"the GUI refuses a value the engine accepts: {gui_messages}. That is "
        f"the exact failure mode item 2.4 exists to remove.")


def test_no_binary_is_reported_as_not_checked():
    preflight = _preflight()
    manager = _manager()
    for missing in (None, "", "/nonexistent/ptm_thermal_sim"):
        status, text = preflight.engine_validate_config(
            manager.get_config(), missing, REPO_ROOT)
        assert status == "unavailable", (
            f"a missing engine ({missing!r}) reported {status!r}; treating "
            f"'could not ask' as 'fine' is the silence this stage removes")
        assert text.strip()


def test_an_engine_without_the_flag_is_never_handed_the_config():
    """The sharpest hazard in this feature, and it is not hypothetical.

    The engine's argument loop has no unknown-argument handler, so a binary
    that predates --validate-config IGNORES it and RUNS THE SIMULATION.
    MEASURED against ~/projects/ptm-postmerge-2026-08-24 (md5
    9d657590009ade0c0d8f2d07d2faa7a2): handed --validate-config it ran a
    complete Moon ico2 job, wrote the whole output directory, and exited 0.
    On a production config, pressing Ctrl+T would have started a multi-hour
    GPU run.

    So the config is not passed until a --help probe confirms the flag exists.
    """
    preflight = _preflight()
    manager = _manager()
    with tempfile.TemporaryDirectory() as tmp:
        marker = Path(tmp) / "IT_RAN"
        # An "old engine": --help does not mention the flag, and being handed a
        # config makes it do work.
        exe = _stub(tmp, f"""
            import sys
            if "--help" in sys.argv:
                print("Usage: ptm_thermal_sim [options]")
                print("  --config FILE   Load parameters from YAML")
                sys.exit(0)
            open({str(marker)!r}, "w").write("simulated\\n")
            sys.exit(0)
        """, supports_flag=False)
        status, text = preflight.engine_validate_config(
            manager.get_config(), exe, REPO_ROOT)
        assert status == "unavailable", (
            f"an engine with no --validate-config reported {status!r}")
        assert not marker.exists(), (
            "the config was handed to a binary that does not support the flag; "
            "an old engine would have RUN it")
        assert "--validate-config" in text and "RUNS THE SIMULATION" in text


def test_the_check_never_targets_the_users_output_directory():
    """Belt and braces behind the capability probe."""
    preflight = _preflight()
    manager = _manager()
    with tempfile.TemporaryDirectory() as tmp:
        seen = Path(tmp) / "seen_output_dir.txt"
        exe = _stub(tmp, f"""
            import yaml
            cfg = yaml.safe_load(open(sys.argv[sys.argv.index("--config") + 1]))
            with open({str(seen)!r}, "w") as fh:
                fh.write(cfg["output"]["directory"])
            print("[validate-config] ACCEPTED")
            sys.exit(0)
        """)
        manager.set_value("output.directory", "/tmp/a_real_run_output")
        status, _text = preflight.engine_validate_config(
            manager._config_for_yaml(), exe, REPO_ROOT)
        assert status == "accepted", status
        handed = seen.read_text()
        assert "a_real_run_output" not in handed, (
            f"the check handed the engine the user's own output directory "
            f"({handed}); anything it wrote would land in a real run's "
            f"results")
        assert "ptm3d_validate_" in handed, handed


def test_a_hung_engine_times_out_rather_than_hanging_the_gui():
    preflight = _preflight()
    manager = _manager()
    original = preflight.VALIDATE_TIMEOUT_S
    preflight.VALIDATE_TIMEOUT_S = 1.0
    try:
        with tempfile.TemporaryDirectory() as tmp:
            exe = _stub(tmp, """
                import time
                time.sleep(120)
            """)
            status, text = preflight.engine_validate_config(
                manager.get_config(), exe, REPO_ROOT)
            assert status == "unavailable", status
            assert "did not answer" in text, text
    finally:
        preflight.VALIDATE_TIMEOUT_S = original


def test_the_check_leaves_no_temporary_file_behind():
    preflight = _preflight()
    exe = _engine()
    manager = _manager()
    manager.set_value("simulation.timesteps", 4)
    manager.set_value("mesh.resolution", 2)
    before = set(Path(tempfile.gettempdir()).glob("ptm3d_validate_*"))
    preflight.engine_validate_config(manager.get_config(), exe, REPO_ROOT)
    after = set(Path(tempfile.gettempdir()).glob("ptm3d_validate_*"))
    assert after <= before, f"left behind: {sorted(after - before)}"


def test_the_check_does_not_write_into_the_output_directory():
    """Checking a finished run's config must not disturb its record."""
    preflight = _preflight()
    exe = _engine()
    with tempfile.TemporaryDirectory() as tmp:
        outdir = Path(tmp) / "finished_run"
        outdir.mkdir()
        (outdir / "heartbeat.txt").write_text("PRIOR RUN MARKER\n")
        manager = _manager()
        manager.set_value("simulation.timesteps", 4)
        manager.set_value("mesh.resolution", 2)
        manager.set_value("output.directory", str(outdir))
        before = sorted(p.name for p in outdir.iterdir())
        preflight.engine_validate_config(manager.get_config(), exe, REPO_ROOT)
        assert sorted(p.name for p in outdir.iterdir()) == before
        assert (outdir / "heartbeat.txt").read_text() == "PRIOR RUN MARKER\n"


def test_the_check_uses_the_same_environment_the_run_will():
    """Validating under a different environment answers a different question."""
    preflight = _preflight()
    manager = _manager()
    with tempfile.TemporaryDirectory() as tmp:
        marker = Path(tmp) / "env.txt"
        exe = _stub(tmp, f"""
            import os, sys
            with open({str(marker)!r}, "w") as fh:
                fh.write(os.environ.get("PTM_ILLUM_RISE_EPS", "(unset)"))
            print("[validate-config] ACCEPTED")
            sys.exit(0)
        """)
        status, _text = preflight.engine_validate_config(
            manager.get_config(), exe, REPO_ROOT)
        assert status == "accepted", status
        assert marker.read_text() == "1", (
            f"the check ran without the engine environment the launch applies "
            f"({marker.read_text()!r}); PTM_ILLUM_RISE_EPS gates the validated "
            f"grazing-leak fix")


def test_the_document_checked_is_the_document_that_will_run():
    """Validating the raw GUI view answers a different question.

    `get_config()` always carries a full `gpu:` section, so the temporary file
    would set `min_vram_margin_explicit` -- the OPPOSITE of what the run does,
    and the opposite of what the GPU page tells the user in the same session.
    It also carries GUI-synthetic keys and none of `_config_for_yaml`'s
    translations, so the PTM_DATA_ROOT paths the banner claims to have checked
    would not be the run's paths. Reproduced before the call site changed.
    """
    import re
    source = (REPO_ROOT / "gui" / "ptm3d_gui.py").read_text()
    calls = re.findall(r"engine_validate_config\(\s*\n?\s*([^,]+),", source)
    assert calls, "no engine_validate_config call found in the main window"
    for arg in calls:
        assert "_config_for_yaml()" in arg, (
            f"the check is handed {arg.strip()!r}, not the engine-facing view")

    # ...and the two really are different, so this is not a distinction
    # without a difference.
    manager = _manager()
    assert "gpu" in manager.get_config()
    assert "gpu" not in manager._config_for_yaml()


def test_a_crashing_engine_is_not_a_config_refusal():
    """rc < 0 is a signal, not a verdict. Do not blame the user's YAML."""
    preflight = _preflight()
    manager = _manager()
    with tempfile.TemporaryDirectory() as tmp:
        exe = Path(tmp) / "crashy"
        exe.write_text('#!/bin/sh\n'
                       'case "$1" in --help) echo "  --validate-config"; '
                       'exit 0;; esac\n'
                       'kill -SEGV $$\n')
        exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
        status, text = preflight.engine_validate_config(
            manager.get_config(), exe, REPO_ROOT)
        assert status == "unavailable", (
            f"a SIGSEGV in the parser reported {status!r}, which tells the "
            f"user their configuration was refused")
        assert "CRASHED" in text and "SIGSEGV" in text, text

    # rc > 1 is also not a verdict.
    with tempfile.TemporaryDirectory() as tmp:
        exe = Path(tmp) / "odd_rc"
        exe.write_text('#!/bin/sh\n'
                       'case "$1" in --help) echo "  --validate-config"; '
                       'exit 0;; esac\n'
                       'exit 42\n')
        exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
        status, text = preflight.engine_validate_config(
            manager.get_config(), exe, REPO_ROOT)
        assert status == "unavailable", status
        assert "42" in text


def test_the_bvh_cache_follows_the_engines_data_root_contract():
    """Silent when wrong: the cache would be written and read in two places."""
    preflight = _preflight()
    import os
    manager = _manager()
    manager.set_value("illumination.bvh_cache_file", "data/cache/site.bvh")
    with tempfile.TemporaryDirectory() as tmp:
        previous = os.environ.get("PTM_DATA_ROOT")
        os.environ["PTM_DATA_ROOT"] = tmp
        try:
            out = preflight.absolutize_config_paths(
                manager._config_for_yaml(), REPO_ROOT)
        finally:
            if previous is None:
                os.environ.pop("PTM_DATA_ROOT", None)
            else:
                os.environ["PTM_DATA_ROOT"] = previous
    resolved = out["illumination"]["bvh_cache_file"]
    assert resolved.startswith(tmp), (
        f"illumination.bvh_cache_file resolved to {resolved}, ignoring "
        f"PTM_DATA_ROOT. The engine resolves this key with resolveDataPath "
        f"(config_parser.cpp:870), so the GUI and the engine would use "
        f"different cache locations -- silently.")


def test_only_the_engines_own_data_root_keys_follow_that_contract():
    """`_ENGINE_DATA_ROOT_FIELDS` is transcribed, not inferred.

    thermal.eq_indirect_flux_file looks exactly like a data path and is not
    one: config_parser.cpp's resolveDataPath block does not list it, so the
    engine opens the raw string relative to the process working directory.
    Resolving it under PTM_DATA_ROOT made the GUI's run YAML point somewhere a
    CLI run of the same config would not look -- and this file always wins over
    eq_indirect_source, so the divergence changes the deep answer.
    """
    import os
    preflight = _preflight()
    manager = _manager()
    manager.set_value("thermal.eq_indirect_flux_file", "data/qind.bin")
    # mesh.import_file, not albedo_map: the writer clears albedo_map in
    # built-in mode, so it would test nothing here.
    manager.set_value("mesh.import_file", "data/meshes/core.ply")
    with tempfile.TemporaryDirectory() as tmp:
        previous = os.environ.get("PTM_DATA_ROOT")
        os.environ["PTM_DATA_ROOT"] = tmp
        try:
            out = preflight.absolutize_config_paths(
                manager._config_for_yaml(), REPO_ROOT)
        finally:
            if previous is None:
                os.environ.pop("PTM_DATA_ROOT", None)
            else:
                os.environ["PTM_DATA_ROOT"] = previous
    assert not out["thermal"]["eq_indirect_flux_file"].startswith(tmp), (
        f"eq_indirect_flux_file resolved under PTM_DATA_ROOT "
        f"({out['thermal']['eq_indirect_flux_file']}); the engine does not")
    assert out["mesh"]["import_file"].startswith(tmp), (
        f"a key the engine DOES resolve that way stopped doing so: "
        f"{out['mesh']['import_file']}")

    # And the list is the engine's, key for key.
    engine_keys = set(preflight._ENGINE_DATA_ROOT_FIELDS)
    source = (REPO_ROOT / "src" / "cpp" / "tools" / "config_parser.cpp")
    if source.exists():
        import re
        text = source.read_text()
        block = text.split("resolution contract", 1)[-1]
        named = set(re.findall(r'resolveDataPath\([^;]*?"([a-z_]+\.[a-z_.]+)"',
                               block, re.S))
        # generic_body.mesh_file is the engine's own key for a field the GUI
        # writes as generic_body.shape.mesh_file; not comparable by name.
        named.discard("generic_body.mesh_file")
        missing = named - engine_keys
        assert not missing, (
            f"the engine resolves these through PTM_DATA_ROOT and the GUI "
            f"does not: {sorted(missing)}")


def test_the_bvh_off_sentinel_survives_absolutization():
    """`bvh_cache_file: "off"` DISABLES the cache; it is not a path.

    config_parser.h:435 and illumination_model.cpp:781/795 document it, and the
    engine's resolveDataPath returns it verbatim. The GUI turned it into
    "<project_root>/off", which would ENABLE caching into a file named `off` --
    the opposite of what was asked, in the run YAML the engine actually reads.
    """
    preflight = _preflight()
    out = preflight.absolutize_config_paths(
        {"illumination": {"bvh_cache_file": "off"}}, "/proj")
    assert out["illumination"]["bvh_cache_file"] == "off", (
        f"the disable sentinel became {out['illumination']['bvh_cache_file']!r}")
    assert preflight.is_path_sentinel("illumination.bvh_cache_file", "off")
    # EXACT, because the engine's test is exact (`bvh_cache_file_ != "off"`).
    # Matching case-insensitively would make the GUI call "OFF" disabled while
    # the engine treated it as a relative path and wrote a cache to <repo>/OFF
    # -- the disclosure and the run disagreeing about the same file. So "OFF"
    # is a PATH here, exactly as it is to the engine.
    for other in ("OFF", " off ", "Off"):
        assert not preflight.is_path_sentinel(
            "illumination.bvh_cache_file", other), other
    assert not preflight.is_path_sentinel("illumination.bvh_cache_file",
                                          "data/site.bvh")
    assert not preflight.is_path_sentinel("mesh.import_file", "off"), (
        "the sentinel leaked to a key that has no such contract")


def test_the_preflight_and_the_run_yaml_resolve_a_path_the_same_way():
    """They disagreed, and the disagreement was silent in both directions.

    With PTM_DATA_ROOT set, the pre-flight stat'd $PTM_DATA_ROOT/... for four
    keys the engine reads relative to the working directory: a launch could be
    blocked for a file that exists, or waved through for one that does not.
    """
    import os
    preflight = _preflight()
    cases = [
        ("thermal.eq_indirect_flux_file", "thermal", "eq_indirect_flux_file"),
        ("thermal.initial_state_file", "thermal", "initial_state_file"),
        ("illumination.positions_file", "illumination", "positions_file"),
        ("mesh.import_file", "mesh", "import_file"),
    ]
    previous = os.environ.get("PTM_DATA_ROOT")
    os.environ["PTM_DATA_ROOT"] = "/DATA"
    try:
        for dotted, section, leaf in cases:
            config = {section: {leaf: "data/thing.bin"}}
            written = preflight.absolutize_config_paths(config, "/proj")
            resolver = (preflight.resolve_data_path
                        if dotted in preflight._ENGINE_DATA_ROOT_FIELDS
                        else preflight.resolve_path)
            checked = str(resolver("data/thing.bin", "/proj"))
            assert written[section][leaf] == checked, (
                f"{dotted}: the run YAML says {written[section][leaf]}, the "
                f"pre-flight would check {checked}")
    finally:
        if previous is None:
            os.environ.pop("PTM_DATA_ROOT", None)
        else:
            os.environ["PTM_DATA_ROOT"] = previous


TESTS = [
    test_the_bvh_off_sentinel_survives_absolutization,
    test_the_preflight_and_the_run_yaml_resolve_a_path_the_same_way,
    test_only_the_engines_own_data_root_keys_follow_that_contract,
    test_the_document_checked_is_the_document_that_will_run,
    test_a_crashing_engine_is_not_a_config_refusal,
    test_the_bvh_cache_follows_the_engines_data_root_contract,
    test_a_good_config_is_accepted_by_the_real_engine,
    test_the_engines_refusal_reaches_the_user_verbatim,
    test_a_disagreement_between_the_two_checkers_is_visible,
    test_no_binary_is_reported_as_not_checked,
    test_an_engine_without_the_flag_is_never_handed_the_config,
    test_the_check_never_targets_the_users_output_directory,
    test_a_hung_engine_times_out_rather_than_hanging_the_gui,
    test_the_check_leaves_no_temporary_file_behind,
    test_the_check_does_not_write_into_the_output_directory,
    test_the_check_uses_the_same_environment_the_run_will,
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
    print(f"\n[test_gui_engine_validator] {total - failures - skipped}/{total} "
          f"passed, {skipped} skipped")
    if skipped == total:
        print("[test_gui_engine_validator] SKIPPED ENTIRELY -- proved NOTHING.")
    if failures:
        print("[test_gui_engine_validator] FAIL")
        return 1
    print("[test_gui_engine_validator] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
