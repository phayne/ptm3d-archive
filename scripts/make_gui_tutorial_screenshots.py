#!/usr/bin/env python3
"""Render the figures for docs/tutorials/04_gui.md — offscreen.

Everything here drives the REAL GUI object model (`gui/gui_panels/*`,
`gui/gui_core/*`) under `QT_QPA_PLATFORM=offscreen` and saves `QWidget.grab()`
output.  No screen scraping, and nothing is ever shown on a display.

Known limit (measured, not assumed): constructing `VisualizationPanel` offscreen
dies with SIGSEGV (exit 139) inside headless PyVista/VTK — and `PTM3DMainWindow`
constructs one, so the composite window and the 3D / 2D result views cannot be
captured here.  Those figures are marked `[SCREENSHOT-PAUL #n]` in the tutorial
and must be shot on a real display.  This is a pre-existing environment limit,
not a defect introduced by the GUI refactor.  Re-test with:

    python scripts/make_gui_tutorial_screenshots.py --probe-mainwindow

Usage (needs a Python with PyQt6 — on this machine, the `europa_viz` env):

    python scripts/make_gui_tutorial_screenshots.py            # panels only
    python scripts/make_gui_tutorial_screenshots.py --run      # + one tiny
                                                              #   CPU demo run

`--run` launches ONE quickstart-class simulation (Moon ico3, 1280 faces, CPU,
radiation off) through the GUI's own launch path — auto-saved absolutized
config, `run_preflight`, `SimulationThread` — so the console and results-browser
figures show real engine output rather than mock text.  It needs
`build/bin/ptm_thermal_sim` and SPICE kernels under `data/kernels/`.

QSettings is redirected to a scratch INI file so this script never touches the
user's real GUI preferences.
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GUI = REPO / "gui"
OUT = REPO / "docs" / "tutorials" / "images" / "gui"


def setup_offscreen():
    """Point Qt at the PyQt6 plugin dir and force the offscreen platform.

    Mirrors run_gui.py:setup_qt_plugin_path — without it the offscreen plugin
    is not found at all ("Could not find the Qt platform plugin 'offscreen'").
    """
    sys.path.insert(0, str(GUI))
    import PyQt6
    plugins = Path(PyQt6.__file__).parent / "Qt6" / "plugins"
    if plugins.exists():
        os.environ["QT_PLUGIN_PATH"] = str(plugins)
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(plugins / "platforms")
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")


def isolate_settings(app_scratch):
    """Redirect QSettings to a scratch INI tree (never touch the real prefs)."""
    from PyQt6.QtCore import QSettings
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat,
                      QSettings.Scope.UserScope, str(app_scratch))


def probe_mainwindow():
    """Try the full window offscreen; report whether it survives.

    Kept as a subcommand so the 'cannot capture the whole window' claim in the
    tutorial stays a measurement anyone can repeat, not folklore.
    """
    setup_offscreen()
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import QTimer
    scratch = tempfile.mkdtemp(prefix="ptm3d-gui-shots-")
    app = QApplication(sys.argv)
    isolate_settings(scratch)
    print("instantiating PTM3DMainWindow offscreen ...", flush=True)
    from ptm3d_gui import PTM3DMainWindow          # noqa: F401 (import side effects)
    w = PTM3DMainWindow()
    w.resize(1400, 900)
    w.show()
    print("SURVIVED — full-window capture is possible in this environment",
          flush=True)
    QTimer.singleShot(500, app.quit)
    app.exec()
    return 0


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def save(widget, name, app, width=None, height=None):
    """Lay out `widget`, grab it, and write docs/tutorials/images/gui/<name>."""
    if width or height:
        widget.resize(width or widget.width(), height or widget.height())
    widget.show()
    app.processEvents()
    app.processEvents()
    pix = widget.grab()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    ok = pix.save(str(path))
    print(f"{'wrote' if ok else 'FAILED'} {path.relative_to(REPO)} "
          f"({pix.width()}x{pix.height()})", flush=True)
    return ok


def section(tab, title):
    """Return the CollapsibleSection with the given title, or None."""
    from gui_panels.config_panel import CollapsibleSection
    for s in tab.findChildren(CollapsibleSection):
        if s.title == title:
            return s
    return None


def show_page(panel, key):
    """Select a nav-rail page by its tab key (panel._nav_keys order)."""
    panel.nav_rail.setCurrentRow(panel._nav_keys.index(key))


# --------------------------------------------------------------------------
# the figures
# --------------------------------------------------------------------------

def capture_panels(app):
    from gui_core.config_manager import ConfigManager
    from gui_core.styles import ModernStyle
    from gui_panels.config_panel import ConfigurationPanel

    cm = ConfigManager()
    panel = ConfigurationPanel(cm)
    panel.setStyleSheet(ModernStyle.get_stylesheet())

    # 01 — Basic mode, Simulation page, stock defaults (Moon, one lunation).
    show_page(panel, "simulation")
    save(panel, "01_basic_simulation_tab.png", app, 640, 760)

    # 02 — Mesh page: built-in vs custom topography source.
    show_page(panel, "mesh")
    save(panel, "02_basic_mesh_tab.png", app, 640, 760)

    # 03 — Properties page on a BUILT-IN body: the six rows the engine cannot
    #      receive are grayed out (config_panel.py:_update_builtin_support).
    show_page(panel, "properties" if "properties" in panel._nav_keys
              else "thermophysical")
    save(panel, "03_basic_properties_tab.png", app, 640, 900)

    # 04 — Thermal page, "Advanced thermal" expanded so the cadence /
    #      integration-window / property-map knobs are visible.
    show_page(panel, "thermal")
    sec = section(panel.tabs["thermal"], "Advanced thermal")
    if sec:
        sec.set_collapsed(False)
    save(panel, "04_basic_thermal_tab.png", app, 640, 980)
    if sec:
        sec.set_collapsed(True)

    # 05 — Output page.
    show_page(panel, "output")
    save(panel, "05_basic_output_tab.png", app, 640, 700)

    # 06 — Advanced options ON: the nav rail grows by Illumination, Radiation,
    #      Engine Flags, HORIZONS (Body stays hidden for a built-in body).
    panel.advanced_toggle.setChecked(True)
    show_page(panel, "simulation")
    save(panel, "06_advanced_nav_rail.png", app, 640, 760)

    # 07 — Radiation page with radiation ON and the coupling section expanded:
    #      the "Face Fraction (legacy mode only)" label and the live
    #      view-factor cost estimate.
    show_page(panel, "radiation")
    rad = panel.tabs["radiation"]
    rad.widgets["radiation.enabled"].setChecked(True)
    sec = section(rad, "Thermal-Radiation Coupling")
    if sec:
        sec.set_collapsed(False)
    rad.update_vf_estimate()
    save(panel, "07_advanced_radiation_coupling.png", app, 640, 940)

    # 08 — The same estimator on a mesh big enough to trip the single-GPU
    #      radiosity wall warning (2^31-byte SpMV limit).
    if sec:
        sec.set_collapsed(True)
    mesh_res = panel.tabs["mesh"].widgets["mesh.resolution"]
    prev_res = mesh_res.value()
    for level in (9, 10):
        mesh_res.setValue(level)
        rad.update_vf_estimate()
        if "exceed" in rad.vf_estimate_label.text():
            break
    save(panel, "08_vf_cost_estimator_warning.png", app, 640, 760)
    mesh_res.setValue(prev_res)
    rad.widgets["radiation.enabled"].setChecked(False)
    rad.update_vf_estimate()

    # 09 — Engine Flags page: the recommended-ON physics-fix env set.
    show_page(panel, "engine_flags")
    save(panel, "09_advanced_engine_flags.png", app, 640, 700)

    panel.advanced_toggle.setChecked(False)
    return panel


def capture_preflight_dialog(app):
    """Capture the real view-factor-rebuild confirmation from confirm_preflight.

    The report is produced by the shipped `run_preflight` on a real config
    (radiation on, no view-factor file), so the warning text AND the cost line
    are the engine-facing ones, not hand-written.  The dialog is parented to a
    ModernStyle-styled widget so it renders in the app's dark theme, as it does
    under the main window.
    """
    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QDialog, QWidget
    from gui_core.config_manager import ConfigManager
    from gui_core.preflight import run_preflight, confirm_preflight
    from gui_core.simulation_runner import SimulationRunner
    from gui_core.styles import ModernStyle

    cm = ConfigManager()
    cm.set_value("radiation.enabled", True)
    cm.set_value("radiation.view_factor_file", "")
    cm.set_value("mesh.resolution", 6)
    report = run_preflight(cm.get_config(),
                           SimulationRunner.find_ptm_executable)
    if not report.vf_rebuild_warning:
        print("SKIP preflight dialog: no VF-rebuild warning produced "
              "(is ptm_thermal_sim present?)", flush=True)
        return

    host = QWidget()
    host.setStyleSheet(ModernStyle.get_stylesheet())

    def grab_modal():
        for w in app.topLevelWidgets():
            if isinstance(w, QDialog) and w.isVisible():
                app.processEvents()
                pix = w.grab()
                OUT.mkdir(parents=True, exist_ok=True)
                path = OUT / "10_preflight_view_factor_dialog.png"
                pix.save(str(path))
                print(f"wrote {path.relative_to(REPO)} "
                      f"({pix.width()}x{pix.height()})", flush=True)
                w.reject()
                return
        print("WARNING: no modal dialog found to grab", flush=True)

    QTimer.singleShot(400, grab_modal)
    confirm_preflight(host, report)


# --------------------------------------------------------------------------
# the demo run
# --------------------------------------------------------------------------

DEMO_OUTPUT = "output/gui_tutorial_moon_quickstart"


def demo_config():
    """The tutorial's demo configuration: quickstart-class Moon run.

    1280-face icosphere, one lunation, 144 output steps, CPU, radiation off.
    """
    from gui_core.config_manager import ConfigManager
    cm = ConfigManager()
    cm.set_value("mesh.resolution", 3)
    cm.set_value("simulation.timesteps", 144)
    cm.set_value("thermal.use_gpu", False)
    cm.set_value("radiation.enabled", False)
    cm.set_value("output.directory", DEMO_OUTPUT)
    return cm


def capture_console_launch(app):
    """Console as it looks the moment Run is pressed — real preflight output.

    Reproduces PTM3DMainWindow.run_simulation's pre-launch console writes
    (header, auto-save note, `[preflight]` summary lines) using the shipped
    `run_preflight`.  No simulation is started here.
    """
    from gui_core.preflight import (find_project_root, resolve_path,
                                    run_preflight)
    from gui_core.simulation_runner import SimulationRunner
    from gui_core.styles import ModernStyle
    from gui_panels.console_panel import ConsolePanel

    cm = demo_config()
    config = cm.get_config()
    exe = SimulationRunner.find_ptm_executable()
    if not exe:
        print("SKIP console-launch figure: ptm_thermal_sim not found",
              flush=True)
        return
    root = find_project_root(exe)
    out_dir = resolve_path(config["output"]["directory"], root)

    console = ConsolePanel()
    console.setStyleSheet(ModernStyle.get_stylesheet())
    console.append_header("Starting Simulation")
    console.append_info(f"Auto-saved run config to: {out_dir / 'config.yaml'}")
    for line in run_preflight(config,
                              SimulationRunner.find_ptm_executable).summary_lines():
        console.append_info(f"[preflight] {line}")
    save(console, "11_console_preflight.png", app, 1000, 400)


def demo_run(app):
    """Run ONE quickstart-class simulation through the GUI's own launch path.

    Mirrors PTM3DMainWindow.run_simulation: validate -> auto-save an
    absolutized run config into the output directory -> run_preflight ->
    SimulationThread with the same signal wiring, into a real ConsolePanel.
    """
    from PyQt6.QtCore import QEventLoop
    from gui_core.preflight import (find_project_root, resolve_path,
                                    run_preflight)
    from gui_core.simulation_runner import SimulationRunner, SimulationThread
    from gui_core.styles import ModernStyle
    from gui_panels.console_panel import ConsolePanel
    from gui_panels.results_panel import ResultsPanel

    cm = demo_config()
    config = cm.get_config()
    exe = SimulationRunner.find_ptm_executable()
    if not exe:
        print("SKIP demo run: ptm_thermal_sim not found "
              "(build it, or symlink build/bin/ptm_thermal_sim)", flush=True)
        return None
    root = find_project_root(exe)
    out_dir = resolve_path(config["output"]["directory"], root)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_config = str(out_dir / "config.yaml")
    cm.save_run_yaml(run_config, root)

    console = ConsolePanel()
    console.setStyleSheet(ModernStyle.get_stylesheet())
    console.append_header("Starting Simulation")
    console.append_info(f"Auto-saved run config to: {run_config}")

    report = run_preflight(config, SimulationRunner.find_ptm_executable)
    for line in report.summary_lines():
        console.append_info(f"[preflight] {line}")
    if not report.ok:
        print("PREFLIGHT BLOCKED:", report.summary_lines(), flush=True)
        save(console, "12_console_completed.png", app, 1000, 460)
        return None

    thread = SimulationThread(run_config, config_dict=config)
    loop = QEventLoop()
    thread.output_ready.connect(console.append_output)
    thread.warning_ready.connect(console.append_warning)
    thread.error_ready.connect(console.append_error)
    thread.heartbeat.connect(
        lambda m, stalled: console.append_warning(f"[watchdog] {m}")
        if stalled else console.append_info(f"[watchdog] {m}"))
    exit_code = {}

    def on_finished(code):
        exit_code["code"] = code
        if code == 0:
            console.append_success("Simulation completed successfully")
        else:
            console.append_error(f"Simulation failed (exit code: {code})")
        loop.quit()

    thread.finished.connect(on_finished)
    thread.start()
    loop.exec()
    thread.wait()

    save(console, "12_console_completed.png", app, 1000, 620)
    print("demo run exit code:", exit_code.get("code"), flush=True)

    if exit_code.get("code") == 0 and out_dir.exists():
        results = ResultsPanel()
        results.setStyleSheet(ModernStyle.get_stylesheet())
        results.load_results(str(out_dir))
        save(results, "13_results_browser.png", app, 640, 760)
    return exit_code.get("code")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="store_true",
                    help="also launch ONE tiny CPU demo simulation and capture "
                         "the console + results browser")
    ap.add_argument("--probe-mainwindow", action="store_true",
                    help="test whether the full window survives offscreen")
    args = ap.parse_args()

    if args.probe_mainwindow:
        return probe_mainwindow()

    setup_offscreen()
    from PyQt6.QtWidgets import QApplication
    scratch = tempfile.mkdtemp(prefix="ptm3d-gui-shots-")
    app = QApplication(sys.argv)
    isolate_settings(scratch)

    capture_panels(app)
    capture_preflight_dialog(app)
    capture_console_launch(app)
    if args.run:
        demo_run(app)
    return 0


if __name__ == "__main__":
    sys.exit(main())
