#!/usr/bin/env python3
"""Every numeric config key, in every YAML spelling, without crashing the GUI.

An exception escaping a Qt slot is a **process abort**, not an error message:
PyQt6 calls ``qFatal``, the application exits 134 (SIGABRT), and any unsaved
edits go with it. There is no ``sys.excepthook`` in this GUI. So a ``TypeError``
in ``load_from_config`` is not a cosmetic bug; it is the whole window.

``yaml.safe_load`` returns a **str** for several perfectly ordinary float
spellings -- its float resolver wants a decimal point or a signed exponent, so
``-1e0``, ``5e3`` and ``1e5`` all come back as strings -- and ``None`` for a bare
``key:``. The engine's ``getWithDefault<double>`` parses every one of them
without complaint, so a config that runs fine on the command line could take the
GUI down.

This was fixed three times on the 2026-08-25 branch and came back twice, because
each fix covered the lines in that day's diff rather than the code path. Measured
across the 26 numeric keys and 5 spellings below:

    base cf7d97c   130 of 130 raised
    after Stage 1    0 of 130

Needs PyQt6 **and** the panel stack (PyVista); skipped where either is missing.

Run directly::

    python3 src/python/tests/test_gui_scalar_spellings.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "python"))
sys.path.insert(0, str(REPO_ROOT / "gui"))

#: Spellings yaml.safe_load hands back as something other than the type the
#: key wants -- every one of which the ENGINE parses without complaint.
#: `-1e0`, `5e3` and `1e5` are str (PyYAML's float resolver wants a decimal
#: point or a signed exponent); `~` is None; `.nan`/`.inf`/`-.inf` are floats
#: that int() refuses; `'true'` is a str that yaml-cpp's as<bool> accepts.
SPELLINGS = ("-1e0", "5e3", "1e5", '"20"', "~",
             ".nan", ".inf", "-.inf", "'true'", "'no'", "[]")


def _leaf_keys():
    """Every leaf key of the GUI's own DEFAULT_CONFIG, as (section, dotted-rest).

    DERIVED, not enumerated. A hand-written key list is how this test reported
    "130 of 130 clean" while `thermal.equilibration_samples_per_diurnal_cycle`
    -- whose GUI default is 120 -- still aborted the window when a config
    quoted it. The set of keys to check is exactly the set the GUI reads.
    """
    from gui_core.config_manager import ConfigManager

    out = []

    def walk(node, prefix):
        for key, value in node.items():
            dotted = f"{prefix}{key}"
            if isinstance(value, dict):
                walk(value, dotted + ".")
            else:
                out.append(dotted)

    walk(ConfigManager.DEFAULT_CONFIG, "")
    return out


class Skipped(Exception):
    """A check that cannot run here, as distinct from one that failed."""


def _skip(reason):
    try:
        import pytest
    except ImportError:
        raise Skipped(reason) from None
    pytest.skip(reason)


def _panel_stack(scratch):
    """Build an offscreen QApplication and return the two classes we need."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        import PyQt6
    except ImportError as exc:
        _skip(f"PyQt6 is unavailable here ({exc})")
    plug = Path(PyQt6.__file__).parent / "Qt6" / "plugins"
    os.environ.setdefault("QT_PLUGIN_PATH", str(plug))
    os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(plug / "platforms"))

    from PyQt6.QtCore import QSettings
    from PyQt6.QtWidgets import QApplication

    # Redirect QSettings BEFORE any panel touches it, or this writes the
    # developer's real com.ptm3d.* preferences.
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    for scope in (QSettings.Scope.UserScope, QSettings.Scope.SystemScope):
        QSettings.setPath(QSettings.Format.IniFormat, scope, str(scratch))
    if QApplication.instance() is None:
        _panel_stack.app = QApplication([])

    try:
        from gui_core.config_manager import ConfigManager
        from gui_panels.config_panel import ConfigurationPanel
    except ImportError as exc:
        _skip(f"the GUI panel stack is unavailable here ({exc}); it needs "
              f"PyVista as well as PyQt6")
    return ConfigManager, ConfigurationPanel


# --------------------------------------------------------------------------

def test_no_scalar_spelling_can_abort_the_gui():
    import yaml

    with tempfile.TemporaryDirectory() as scratch:
        ConfigManager, ConfigurationPanel = _panel_stack(scratch)
        path = Path(scratch) / "spelling.yaml"
        out_path = Path(scratch) / "out.yaml"
        failures = []
        checked = 0
        keys = _leaf_keys()
        assert len(keys) > 80, (
            f"only {len(keys)} leaf keys found in DEFAULT_CONFIG; the "
            f"derivation is broken and this test would pass vacuously")

        # ONE manager and ONE panel, reused. Rebuilding a ConfigurationPanel per
        # case costs ~0.4 s and turns this into a nine-minute gate, which is a
        # gate nobody runs. Reuse is also the harsher test: the widgets carry
        # state from the previous case, which is exactly how the property-map
        # combo's label-text writeback aborted a SECOND load_from_config().
        manager = ConfigManager()
        panel = ConfigurationPanel(manager)

        for spelling in SPELLINGS:
            parsed = yaml.safe_load(spelling)
            for dotted in keys:
                doc = {"simulation": {"body": "Moon"},
                       "radiation": {"enabled": True, "beaming": True}}
                node = doc
                parts = dotted.split(".")
                for part in parts[:-1]:
                    node = node.setdefault(part, {})
                node[parts[-1]] = parsed
                path.write_text(yaml.dump(doc, default_flow_style=False))
                checked += 1
                try:
                    manager.load_from_yaml(str(path))
                    for tab in panel.tabs.values():
                        tab.load_from_config()
                    # A SECOND pass: the combo writebacks store label text, so
                    # the second load is a different input than the first.
                    for tab in panel.tabs.values():
                        tab.load_from_config()
                    manager.validate()
                    manager.save_to_yaml(str(out_path))
                except Exception as exc:                   # noqa: BLE001
                    failures.append(
                        f"{dotted} = {spelling}: {type(exc).__name__}: {exc}")

        # Print the denominator even on success: "0 of 1375" is a measurement,
        # "PASS" alone could be a sweep that checked nothing.
        print(f"    swept {checked} key x spelling combinations "
              f"({len(keys)} leaf keys x {len(SPELLINGS)} spellings), "
              f"{len(failures)} raised")
        assert not failures, (
            f"{len(failures)} of {checked} loads raised. Each one is an "
            f"application abort (PyQt6 qFatal, exit 134), not an error "
            f"message:\n  " + "\n  ".join(failures[:15]))


def test_no_scalar_spelling_can_abort_a_body_card():
    """The same sweep, over the body-first cards.

    The cards are a SECOND set of widgets bound to the same configuration keys,
    with their own loaders and their own refresh functions -- the auto-albedo
    narration, the beaming facet-scale gate, the terrain registry mirror, the
    thermal-inertia readout. Every one of those reads a value straight out of
    the configuration, where ``yaml.safe_load`` may have left a string, a None
    or a NaN, and every one runs from a Qt slot where an escaping exception is
    a process abort rather than a message.

    The sweep above cannot reach them: a card mirrors the configuration only
    while it is the ACTIVE route, which is deliberate (five cards bind
    ``simulation.start_time``, and an inactive one loading would let whichever
    ran last decide the stored precision). So this drives each card through
    ``activate()`` for every mode it offers, over its OWN bound keys -- which
    is complete for the cards rather than a sample of them.
    """
    import yaml

    with tempfile.TemporaryDirectory() as scratch:
        ConfigManager, ConfigurationPanel = _panel_stack(scratch)
        from gui_core.body_cards import modes_for

        path = Path(scratch) / "card_spelling.yaml"
        manager = ConfigManager()
        panel = ConfigurationPanel(manager)
        start = panel.tabs["start"]

        routes = []
        for name, card in sorted(panel.tabs.items()):
            if not name.startswith("card_"):
                continue
            for mode in modes_for(card.route):
                routes.append((card, mode.key, sorted(card.widgets)))
        assert routes, "no body cards were found; this sweep checks nothing"
        assert sum(len(keys) for _c, _m, keys in routes) > 40, (
            "the cards bind fewer than forty keys in total; the derivation is "
            "broken and this test would pass vacuously")

        failures = []
        checked = 0
        for card, mode, keys in routes:
            for spelling in SPELLINGS:
                parsed = yaml.safe_load(spelling)
                for dotted in keys:
                    doc = {"simulation": {"body": "Moon"},
                           "radiation": {"enabled": True, "beaming": True}}
                    node = doc
                    parts = dotted.split(".")
                    for part in parts[:-1]:
                        node = node.setdefault(part, {})
                    node[parts[-1]] = parsed
                    path.write_text(yaml.dump(doc, default_flow_style=False))
                    checked += 1
                    try:
                        manager.load_from_yaml(str(path))
                        # activate() is the real entry point: it applies the
                        # mode, shows or hides groups, and mirrors the config.
                        # One run type is enough -- the run type decides which
                        # groups are VISIBLE and nothing about what is read --
                        # and two doubled a gate that already runs for minutes.
                        card.activate(mode, "time_series")
                        start.route_for_config()
                    except Exception as exc:               # noqa: BLE001
                        failures.append(
                            f"{card.route}/{mode}: {dotted} = {spelling}: "
                            f"{type(exc).__name__}: {exc}")

        print(f"    swept {checked} card x key x spelling combinations across "
              f"{len(routes)} card modes, {len(failures)} raised")
        assert not failures, (
            f"{len(failures)} of {checked} card loads raised. Each one is an "
            f"application abort (PyQt6 qFatal, exit 134), not an error "
            f"message:\n  " + "\n  ".join(failures[:15]))


TESTS = [test_no_scalar_spelling_can_abort_the_gui,
         test_no_scalar_spelling_can_abort_a_body_card]


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
    print(f"\n[test_gui_scalar_spellings] {tally}")
    if failures:
        print("[test_gui_scalar_spellings] FAIL")
        return 1
    print("[test_gui_scalar_spellings] PASS"
          + (f" ({skipped} skipped)" if skipped else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
