#!/usr/bin/env python3
"""Config parity: what Python believes about the engine's config vs the engine.

The engine has no required config key and rejects no unknown key
(``ConfigParser::loadFromYAML``: 154 keys, every one through ``getWithDefault``,
each section behind ``if (root["section"])``). A key the Python side spells
wrongly, or a default the Python side remembers from an older engine, therefore
produces no error anywhere -- just a different run than the one intended. These
checks turn that silence into a failure.

Four things are asserted, in the order they would break:

1. the extractor still understands ``config_parser.cpp`` (a floor on the key
   count and the exact set of sections, so a parse that silently found nothing
   cannot pass);
2. ``ptm/engine_defaults.py`` still equals what the C++ sources say -- this is
   the drift guard: change a default in C++ and this fails until the table is
   regenerated;
3. every key the GUI's config writer puts into a run YAML is a key the engine
   reads;
4. the dead keys in the shipped ``examples/`` configs are exactly the ones
   already known, so a new one cannot be added unnoticed.

Check 3 needs PyQt6, because it reads the GUI's own config writer. Without it
that check is **skipped**, and the summary distinguishes skipped from failed --
a missing GUI dependency is not a config-parity failure. CI installs PyQt6 and
sees all five run.

Run directly (no pytest needed)::

    python3 src/python/tests/test_config_parity.py

Also collectable by pytest, if it is installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "python"))
sys.path.insert(0, str(REPO_ROOT / "gui"))


class Skipped(Exception):
    """A check that could not run here, as distinct from one that failed.

    Only ``test_gui_written_yaml_uses_only_engine_keys`` raises this, and only
    for a missing PyQt6: the GUI's ConfigManager imports Qt, so without it the
    check is unrunnable, not failing. It used to surface as
    ``ERROR ... ModuleNotFoundError`` and a ``4/5 ... FAIL`` summary on any
    machine without the GUI's dependencies installed -- a red result that said
    nothing about the config surface. CI installs PyQt6, so CI still runs all
    five and a genuine failure is still a failure.
    """


def _skip(reason):
    """Skip the current check. Uses pytest's own mechanism when under pytest."""
    try:
        import pytest
    except ImportError:
        raise Skipped(reason) from None
    pytest.skip(reason)

from ptm import engine_config                                    # noqa: E402
from ptm.engine_config import (ENGINE_DEFAULTS, ENGINE_SECTIONS,  # noqa: E402
                               KEYS_WITHOUT_DEFAULT, unknown_keys)

# A parse that understood nothing would return an empty table and pass every
# comparison below. These floors are set just under the counts measured on
# 2026-08-09 (154 keys / 9 sections); they exist to fail a broken parse, not to
# track the exact number, which assertion 2 pins precisely.
MIN_KEYS = 140
EXPECTED_SECTIONS = (
    # `flux_cubes` is a real new top-level section (the passive flux-cube probe,
    # docs/flux-cubes.md), not a stray. Its `locations` key is the parser's only
    # SEQUENCE OF MAPS; its per-entry fields are deliberately absent from the
    # table because there is no dotted key for them -- see the note at the
    # locations block in config_parser.cpp.
    "advanced", "flux_cubes", "generic_body", "illumination", "mesh", "output",
    "radiation", "simulation", "thermal", "thermophysical",
)

# Keys present in shipped example configs that the engine's parser never reads.
# Every entry here is a config that asks for something and is silently given
# nothing. Do not add to this list to make the test pass -- either the key is
# real and the engine should read it, or the config should drop it.
#
#   visualization.*  an entire 11-key section; the string "visualization"
#                    appears in no C++ and no GUI source
#   output.no_maps   parsed by nothing
#: The GUI's built-in-body write path emits only the keys the engine reads
#: (``_config_for_yaml``). Its generic-body path does not: it writes the raw
#: ``thermophysical:`` block, whose scalars and ``use_default_*`` switches the
#: engine ignores -- ``warnUnreadThermophysicalKeys`` names them at runtime.
#: The live channel for a generic body is ``generic_body.surface``, which the
#: same function fills from these values. They are kept in the YAML because the
#: GUI reads them back to repopulate its own form, so removing them here would
#: break the save/load round trip; that is a change to make deliberately, not
#: as a side effect. Pinned so a NEW dead key still fails.
GUI_GENERIC_BODY_DEAD_KEYS = (
    "thermophysical.H", "thermophysical.Qb", "thermophysical.albedo",
    "thermophysical.chi", "thermophysical.cp0", "thermophysical.emissivity",
    "thermophysical.kd", "thermophysical.ks", "thermophysical.rhod",
    "thermophysical.rhos",
    "thermophysical.use_default_H", "thermophysical.use_default_Qb",
    "thermophysical.use_default_chi", "thermophysical.use_default_cp0",
    "thermophysical.use_default_emissivity", "thermophysical.use_default_kd",
    "thermophysical.use_default_ks", "thermophysical.use_default_rhod",
    "thermophysical.use_default_rhos",
)

KNOWN_DEAD_KEYS_IN_EXAMPLES = {
    "examples/europa_high_res.yaml": (
        "output.no_maps",
        "visualization.azimuth", "visualization.camera_distance",
        "visualization.camera_latitude", "visualization.camera_longitude",
        "visualization.create", "visualization.emission_angle",
        "visualization.fps", "visualization.height",
        "visualization.save_frames", "visualization.width",
    ),
    "examples/quick_test.yaml": (
        "output.no_maps",
        "visualization.azimuth", "visualization.camera_distance",
        "visualization.camera_latitude", "visualization.camera_longitude",
        "visualization.create", "visualization.emission_angle",
        "visualization.fps", "visualization.height",
        "visualization.save_frames", "visualization.width",
    ),
}


def _derive():
    defaults, no_default, unparsed = engine_config.parse_engine_config(REPO_ROOT)
    assert not unparsed, (
        "config_parser defaults that are not plain literals (the extractor "
        f"cannot represent them): {unparsed}")
    return defaults, no_default


# --------------------------------------------------------------------------

def test_extractor_is_not_vacuous():
    """The C++ parse still finds a complete-looking config surface."""
    defaults, _ = _derive()
    assert len(defaults) >= MIN_KEYS, (
        f"only {len(defaults)} config keys found in config_parser.cpp "
        f"(expected at least {MIN_KEYS}); the extractor has most likely lost "
        f"track of the parser's shape, not the parser lost keys")
    sections = tuple(sorted({k.split(".")[0] for k in defaults}))
    assert sections == EXPECTED_SECTIONS, (
        f"top-level sections changed: {sections} != {EXPECTED_SECTIONS}")


def test_committed_table_matches_cxx_sources():
    """ptm/engine_defaults.py still says what config_parser.{h,cpp} say."""
    defaults, no_default = _derive()

    missing = sorted(set(defaults) - set(ENGINE_DEFAULTS))
    extra = sorted(set(ENGINE_DEFAULTS) - set(defaults))
    changed = sorted(
        (k, ENGINE_DEFAULTS[k], defaults[k])
        for k in set(defaults) & set(ENGINE_DEFAULTS)
        if ENGINE_DEFAULTS[k] != defaults[k]
        or type(ENGINE_DEFAULTS[k]) is not type(defaults[k])
    )
    report = []
    if missing:
        report.append(f"keys the engine reads but the table lacks: {missing}")
    if extra:
        report.append(f"keys in the table the engine no longer reads: {extra}")
    if changed:
        report.append("defaults that drifted (key, table, engine): "
                      + ", ".join(repr(c) for c in changed))
    assert not report, (
        "ptm/engine_defaults.py is out of date with the C++ parser.\n  "
        + "\n  ".join(report)
        + "\nRegenerate with: python3 -m ptm.engine_config --regenerate")

    assert tuple(no_default) == tuple(KEYS_WITHOUT_DEFAULT), (
        f"keys read without a literal default changed: {no_default} != "
        f"{list(KEYS_WITHOUT_DEFAULT)}")
    assert tuple(sorted({k.split('.')[0] for k in defaults})) == tuple(ENGINE_SECTIONS)


def test_generated_module_is_byte_current():
    """The committed file equals what --regenerate would write right now."""
    defaults, no_default = _derive()
    rendered = engine_config._render_module(defaults, no_default)
    target = Path(engine_config.__file__).with_name("engine_defaults.py")
    assert target.read_text() == rendered, (
        f"{target} differs from a fresh render; run "
        f"python3 -m ptm.engine_config --regenerate")


def test_gui_written_yaml_uses_only_engine_keys():
    """Every key the GUI writes into a run YAML is one the engine reads.

    The GUI's DEFAULT_CONFIG deliberately carries UI-only state
    (``topo_source``, ``property_map_kind``, the ``thermophysical.use_default_*``
    switches, the three scalar ``semi_axis_?_km`` fields). ``_config_for_yaml``
    is what removes or translates them, so that is what is checked -- checking
    DEFAULT_CONFIG instead would report 26 false positives.

    Skipped, not failed, where PyQt6 is not installed: the check needs the
    GUI's own writer and there is no substitute for it.
    """
    try:
        from gui_core.config_manager import ConfigManager
    except ImportError as exc:
        _skip(f"the GUI's config writer is unavailable here ({exc}); "
              f"install PyQt6 to run this check")

    manager = ConfigManager()
    written = manager._config_for_yaml()
    dead = unknown_keys(written)
    assert not dead, (
        f"the GUI writes {len(dead)} key(s) the engine never reads: {dead}")

    # Again with the generic-body branch populated, which is where the
    # semi_axis_a/b/c -> semi_axes_km translation lives.
    manager = ConfigManager()
    body = manager.config["generic_body"]
    body["name"] = "parity-probe"
    body["orbit"]["semi_major_axis_au"] = 2.5
    body["shape"]["semi_axis_a_km"] = 3.0
    body["shape"]["semi_axis_b_km"] = 2.0
    body["shape"]["semi_axis_c_km"] = 1.0
    written = manager._config_for_yaml()
    assert "generic_body" in written, "generic_body was not written despite carrying data"
    assert written["generic_body"]["shape"]["semi_axes_km"] == [3.0, 2.0, 1.0], (
        "the three scalar semi_axis_?_km fields no longer translate into the "
        "semi_axes_km sequence the engine reads")
    dead = tuple(unknown_keys(written))
    assert dead == GUI_GENERIC_BODY_DEAD_KEYS, (
        "the dead keys the GUI writes on the generic-body path changed.\n"
        f"  now:      {list(dead)}\n"
        f"  expected: {list(GUI_GENERIC_BODY_DEAD_KEYS)}")


def test_shipped_examples_have_no_new_dead_keys():
    """No shipped example config grows a key the engine ignores."""
    import yaml

    found = {}
    for path in sorted((REPO_ROOT / "examples").glob("*.yaml")):
        config = yaml.safe_load(path.read_text()) or {}
        dead = unknown_keys(config)
        if dead:
            found[f"examples/{path.name}"] = tuple(dead)

    new = {f: k for f, k in found.items()
           if tuple(sorted(k)) != tuple(sorted(KNOWN_DEAD_KEYS_IN_EXAMPLES.get(f, ())))}
    fixed = {f: k for f, k in KNOWN_DEAD_KEYS_IN_EXAMPLES.items() if f not in found}
    assert not new, (
        "example configs carry keys the engine does not read: "
        + "; ".join(f"{f}: {list(k)}" for f, k in sorted(new.items())))
    assert not fixed, (
        "these entries in KNOWN_DEAD_KEYS_IN_EXAMPLES are now clean and should "
        f"be deleted from the list: {sorted(fixed)}")


TESTS = [
    test_extractor_is_not_vacuous,
    test_committed_table_matches_cxx_sources,
    test_generated_module_is_byte_current,
    test_gui_written_yaml_uses_only_engine_keys,
    test_shipped_examples_have_no_new_dead_keys,
]


def main():
    failures = 0
    skipped = 0
    for test in TESTS:
        try:
            test()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {test.__name__}\n     {exc}")
        except BaseException as exc:                   # noqa: BLE001
            # pytest.skip raises its own Skipped, which derives from
            # BaseException, so match on the name and let anything else through
            # as the error it is.
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
    passed = total - failures - skipped
    tally = f"{passed}/{total} passed"
    if skipped:
        tally += f", {skipped} skipped"
    if failures:
        tally += f", {failures} failed"
    print(f"\n[test_config_parity] {tally}, "
          f"{len(ENGINE_DEFAULTS)} engine config keys checked")
    if failures:
        print("[test_config_parity] FAIL")
        return 1
    print("[test_config_parity] PASS"
          + (f" ({skipped} skipped)" if skipped else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
