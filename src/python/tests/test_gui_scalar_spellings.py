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

#: Spellings yaml.safe_load hands back as something other than int/float.
SPELLINGS = ("-1e0", "5e3", "1e5", '"20"', "~")

#: Every numeric key the GUI reads back into a widget or compares.
NUMERIC_KEYS = {
    "simulation": ("duration_days", "timesteps"),
    "mesh": ("resolution", "scale_factor", "latitude_override",
             "longitude_override"),
    "thermal": ("threads", "equilibration_orbits", "geothermal_heat_flow",
                "analytic_seed_cycles", "integration_days", "integration_steps"),
    "radiation": ("min_shadow_fraction", "max_coupling_iterations",
                  "coupling_temp_threshold", "coupling_flux_threshold",
                  "coupling_face_fraction", "aca_tolerance", "aca_max_rank",
                  "hmatrix_threads", "captured_energy_target", "max_neighbors",
                  "min_neighbors", "rms_slope_deg",
                  "nearfield_quadrature_order", "mc_rays_per_face"),
}


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
        failures = []
        checked = 0

        for spelling in SPELLINGS:
            for section, keys in NUMERIC_KEYS.items():
                for key in keys:
                    doc = {"simulation": {"body": "Moon"},
                           "radiation": {"enabled": True, "beaming": True}}
                    doc.setdefault(section, {})
                    doc[section][key] = yaml.safe_load(spelling)
                    path.write_text(yaml.dump(doc, default_flow_style=False))
                    checked += 1
                    try:
                        manager = ConfigManager()
                        manager.load_from_yaml(str(path))
                        panel = ConfigurationPanel(manager)
                        for tab in panel.tabs.values():
                            tab.load_from_config()
                        manager.validate()
                        manager.save_to_yaml(str(Path(scratch) / "out.yaml"))
                    except Exception as exc:               # noqa: BLE001
                        failures.append(
                            f"{section}.{key} = {spelling}: "
                            f"{type(exc).__name__}: {exc}")

        assert not failures, (
            f"{len(failures)} of {checked} loads raised. Each one is an "
            f"application abort (PyQt6 qFatal, exit 134), not an error "
            f"message:\n  " + "\n  ".join(failures[:12]))


TESTS = [test_no_scalar_spelling_can_abort_the_gui]


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
    n = len(SPELLINGS) * sum(len(v) for v in NUMERIC_KEYS.values())
    print(f"\n[test_gui_scalar_spellings] {tally}, {n} config loads")
    if failures:
        print("[test_gui_scalar_spellings] FAIL")
        return 1
    print("[test_gui_scalar_spellings] PASS"
          + (f" ({skipped} skipped)" if skipped else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
