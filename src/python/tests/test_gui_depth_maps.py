#!/usr/bin/env python3
"""Subsurface temperature maps on the Output tab: the rules, the controls, the file.

A depth map is the surface temperature product measured deeper -- one number
per facet, at a chosen depth, reduced over the run's output window. Asking for
one must change no temperature anywhere, and NOT asking for one must change
nothing at all about the configuration the application writes. So most of what
is checked here is what does not happen.

The five properties that matter, and why each is checked rather than trusted:

* **The arithmetic is the engine's, not a second copy of it.** Section 1 reads
  the constants back out of ``src/cpp/thermal/depth_map_plan.h`` -- the naming
  rule, the size formula, the ceiling, the default -- so an engine-side change
  fails here instead of quietly leaving the panel promising a file name or a
  size the run will not produce. A cross-language mirror that nothing compares
  is a mirror that drifts.

* **Nothing leaks.** With no depths listed the emitted configuration carries no
  ``map_depth*`` key at all. An empty list is the engine's own "off", so
  silence and a list saying nothing are the same run -- and writing the keys
  would change the file every existing project saves, for no behavioral reason.

* **Every refusal the panel can make, the engine also makes.** The clauses that
  need the run's solved depth grid are NOT mirrored, deliberately, and section
  2 pins that boundary so a future "helpful" grid guess in Python is a test
  failure rather than a second opinion about the physics.

* **A typo does not switch the product off.** An entry that is not a number is
  kept and refused by name. Dropping it would leave an empty list, which is how
  this feature is turned OFF -- so one bad character in a list of four depths
  would silently produce a surface-only run.

* **The controls are real.** Asserted with ``click()``, not ``setChecked``, and
  with ``isVisibleTo`` plus the viewport rectangle, because those three
  instruments were each blind to a different defect on this application's cards
  and the lesson is written down in ``test_gui_flux_cubes.py`` section 11.

Needs PyQt6 and the panel stack for the control arms; the arithmetic, the
refusals and the header comparison run without either. Run directly::

    python3 src/python/tests/test_gui_depth_maps.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "python"))
sys.path.insert(0, str(REPO_ROOT / "gui"))

# ⛔ BEFORE ANY GUI IMPORT. `gui_panels.visualization_panel` reads this at
# module import time; setting it later has no effect.
os.environ.setdefault("PTM_GUI_DISABLE_3D", "1")

PLAN_HEADER = REPO_ROOT / "src" / "cpp" / "thermal" / "depth_map_plan.h"


class Skip(Exception):
    """A check that cannot run here, as distinct from one that failed."""


_APP = None
_PANEL = None
_MANAGER = None
_SCRATCH = None


def _panel():
    """One offscreen ConfigurationPanel, reused across the control arms."""
    global _APP, _PANEL, _MANAGER, _SCRATCH
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
    from PyQt6.QtCore import QSettings
    from PyQt6.QtWidgets import QApplication

    # Redirect QSettings before any panel touches it, or this writes the real
    # com.ptm3d.* preferences.
    _SCRATCH = tempfile.mkdtemp(prefix="ptm_depthmap_test_")
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    for scope in (QSettings.Scope.UserScope, QSettings.Scope.SystemScope):
        QSettings.setPath(QSettings.Format.IniFormat, scope, _SCRATCH)
    if QApplication.instance() is None:
        _APP = QApplication([])
    try:
        from gui_core.config_manager import ConfigManager
        from gui_panels.config_panel import ConfigurationPanel
    except ImportError as exc:                     # pragma: no cover
        raise Skip(f"the GUI panel stack is unavailable here ({exc})")
    _MANAGER = ConfigManager()
    _PANEL = ConfigurationPanel(_MANAGER)
    return _MANAGER, _PANEL


def _tab():
    """The Output tab, on a configuration reset to this application's defaults."""
    manager, panel = _panel()
    manager.reset_to_defaults()
    tab = panel.tabs["output"]
    tab.load_from_config()
    return manager, tab


def _pump():
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance()
    for _ in range(4):
        app.processEvents()


def _header_text():
    if not PLAN_HEADER.is_file():
        raise Skip(f"{PLAN_HEADER} is not in this checkout")
    return PLAN_HEADER.read_text()


# ==========================================================================
# 1. The arithmetic, anchored to the engine's own header
# ==========================================================================

def test_the_naming_rule_is_the_engines():
    """``z<meters, '.' -> 'p'>m``, three decimals -- read back out of the C++.

    The GUI shows these file names before a run exists and the raster exporter
    parses them after one, so a drift in either direction produces a name that
    matches nothing. The header is the one place the rule is written down.
    """
    from gui_core import depth_maps as dm
    text = _header_text()
    assert 'std::snprintf(buf, sizeof(buf), "%.3f", z_m)' in text, (
        "depth_map_plan.h no longer formats the depth with %.3f; the GUI and "
        "the raster exporter both spell the name that way and would now be "
        "naming files the engine does not write")
    assert "if (c == '.') c = 'p'" in text, (
        "depth_map_plan.h no longer replaces the decimal point with 'p'")
    assert 'return "z" + s + "m"' in text
    assert 'return "temperature_" + depthTag(z_m)' in text

    for meters, expected in ((0.0, "temperature_z0p000m"),
                             (0.05, "temperature_z0p050m"),
                             (0.3, "temperature_z0p300m"),
                             (1.25, "temperature_z1p250m"),
                             (12.3456, "temperature_z12p346m")):
        assert dm.field_name(meters) == expected, (
            f"field_name({meters}) is {dm.field_name(meters)!r}, "
            f"not {expected!r}")


def test_the_size_formula_is_the_engines():
    """One 4-byte count plus 4 bytes a facet, per depth per statistic.

    Anchored to numbers the plan for this feature published: 9,591,564 bytes
    for one depth and one statistic at 2,397,890 facets, 137 MiB for five
    depths and three statistics, 64.3 GiB for five depths over 1,440 output
    steps. Those are the figures a reader will compare the panel against.
    """
    from gui_core import depth_maps as dm
    text = _header_text()
    assert "const std::uint64_t per_field = 4ull + 4ull * F;" in text, (
        "the engine's per-file size formula changed; the panel's size line is "
        "a mirror of it and would now be wrong")
    assert "const std::uint64_t per_raw = (4ull + 8ull * F) + (4ull + 4ull * F);" \
        in text, "the engine's raw side-file size formula changed"

    faces = 2397890
    assert dm.statistics_bytes(1, 1, faces) == 9591564
    assert dm.statistics_bytes(5, 3, faces) == 143873460
    assert dm.fmt_bytes(dm.statistics_bytes(5, 3, faces)) == "137.2 MiB"
    assert dm.fmt_bytes(dm.time_series_bytes(5, 1440, faces)) == "64.3 GiB"
    # The merge-friendly side files add a float64 sum and a uint32 count.
    assert dm.statistics_bytes(1, 3, faces, with_raw=True) == \
        dm.statistics_bytes(1, 3, faces) + (4 + 8 * faces) + (4 + 4 * faces)


def test_the_ceiling_and_the_default_are_the_engines():
    """8 GiB, and depths move to the nearest solved depth unless told not to.

    A different ceiling here would either refuse runs the engine accepts or
    promise runs it refuses; a different default would put the panel and the
    run on opposite sides of a 0.195 K interpolation error.
    """
    from gui_core import depth_maps as dm
    text = _header_text()
    assert "std::uint64_t max_bytes = 8ull << 30;" in text, (
        "the engine's product ceiling moved; the panel mirrors it")
    assert "bool snap_to_node = true;" in text, (
        "the engine no longer moves depths to the nearest solved depth by "
        "default")
    assert dm.MAX_PRODUCT_BYTES == 8 << 30
    from ptm.engine_defaults import ENGINE_DEFAULTS
    assert ENGINE_DEFAULTS["output.map_depth_snap_to_node"] is True


def test_the_statistics_are_the_three_the_engine_writes():
    from gui_core import depth_maps as dm
    text = _header_text()
    assert "enum class Stat { Avg, Max, Min };" in text, (
        "the engine's statistic set changed; the panel offers three boxes")
    assert dm.STAT_NAMES == ("avg", "max", "min")
    assert set(dm.STAT_LABELS) == set(dm.STAT_NAMES)


def test_the_interpolation_error_figures_are_the_engines():
    """0.195 K and 0.077 K at a 160 K swing, and they are not one number.

    The whole reason the panel carries two read-outs is that these differ by
    about 2.5x and in opposite directions from the naive expectation. A single
    figure would be wrong for one of the two products.
    """
    from gui_core import depth_maps as dm
    text = _header_text()
    assert "1.217e-3 * A = 0.195 K" in text, (
        "the engine's stated instantaneous interpolation error changed")
    assert "0.085 K" in text, (
        "the engine's stated error in the maximum changed")
    assert abs(dm.INSTANTANEOUS_ERROR_PER_K * dm.REFERENCE_SWING_K
               - 0.195) < 0.001
    # ⛔ 0.085 K, NOT the header's 4.8e-4 x A = 0.077 K. Those are different
    # quantities: 4.8e-4 is what accumulating the maximum IN the solver costs,
    # which is what a depth ON a solved node gets. The panel is describing an
    # OFF-node depth, where the figure is 0.085 K. The two assertions used to
    # sit in separate suites -- one pinning "0.085 K" in the header, one
    # pinning "0.077" on screen -- and never met, so the 10% understatement was
    # invisible.
    assert abs(dm.PEAK_ERROR_PER_K * dm.REFERENCE_SWING_K - 0.085) < 0.001, (
        f"the panel quotes {dm.PEAK_ERROR_PER_K * dm.REFERENCE_SWING_K:.4f} K "
        f"for the error in the maximum; the engine's own header says 0.085 K")
    ratio = dm.INSTANTANEOUS_ERROR_PER_K / dm.PEAK_ERROR_PER_K
    assert 2.0 < ratio < 2.6, (
        f"the two budgets are {ratio:.2f}x apart; the header says ~2.3x")
    assert dm.INSTANTANEOUS_ERROR_PER_K > dm.PEAK_ERROR_PER_K, (
        "the instantaneous error must be the LARGER of the two; if these ever "
        "swap, every sentence the panel prints about them is backwards")
    lines = dm.error_lines(snap_to_node=False)
    assert len(lines) == 2, (
        "the two error budgets must be reported separately; one line means "
        "one of the two products is being quoted the other's number")


# ==========================================================================
# 2. The refusals -- every one fires, and the boundary is where it should be
# ==========================================================================

def test_every_grid_free_refusal_fires():
    from gui_core import depth_maps as dm
    cases = {
        "no statistic": ([0.05], []),
        "unknown statistic": ([0.05], ["mean"]),
        "repeated statistic": ([0.05], ["avg", "avg"]),
        "negative depth": ([-0.05], ["avg"]),
        "repeated depth": ([0.05, 0.05], ["avg"]),
        "a word where a depth belongs": (["deep"], ["avg"]),
    }
    silent = [name for name, (depths, stats) in cases.items()
              if not dm.validate_request(depths, stats)]
    assert not silent, f"these requests were accepted in silence: {silent}"


def test_two_depths_a_millimeter_apart_are_refused_when_not_snapping():
    """A file name carries three decimals, so they would be the same file.

    Decidable here ONLY when depths are not being moved to the nearest solved
    depth: then the depth sampled is the depth asked for. With snapping on the
    resolved depths depend on the run's grid, and this check must stay out of
    it rather than guess.
    """
    from gui_core import depth_maps as dm
    problems = dm.validate_request([0.0001, 0.0002], ["avg"],
                                   snap_to_node=False)
    assert problems and "overwrite" in problems[0], (
        f"two depths that resolve to one file name were accepted: {problems}")
    assert not dm.validate_request([0.0001, 0.0002], ["avg"],
                                   snap_to_node=True), (
        "with snapping on, where these two land is decided by the run's own "
        "depth grid; refusing here would be a guess about the physics")


def test_a_valid_request_is_not_refused():
    """The control. A refusal list that fires on everything is not a check."""
    from gui_core import depth_maps as dm
    assert dm.validate_request([0.0, 0.05, 0.25], ["avg", "max", "min"]) == []
    assert dm.validate_request([], []) == [], (
        "an empty depth list is this feature switched OFF, not an error")


def test_the_grid_bound_clauses_are_left_to_the_engine():
    """The boundary, pinned so nobody adds a second depth grid in Python.

    A depth deeper than the model's column is refused BY THE ENGINE, which
    builds the grid. Reproducing that here means reproducing the grid builder,
    and a second copy that disagreed would refuse runs the engine accepts --
    or worse, accept ones it should not.
    """
    from gui_core import depth_maps as dm
    assert dm.validate_request([1000.0], ["avg"]) == [], (
        "a depth of 1000 m was refused here. That clause needs the run's own "
        "solved depth grid, which does not exist until the run builds it; "
        "answering it in Python means keeping a second copy of the grid "
        "builder")
    source = Path(dm.__file__).read_text()
    assert "does not exist until the run" in source, (
        "the module no longer states that the depth grid is not available "
        "here; that sentence is what stops the next reader from adding one")


def test_the_engine_preconditions_all_fire():
    from gui_core import depth_maps as dm
    base = {"thermal": {"run_thermal_mesh": True, "compute_statistics": True,
                        "use_gpu": True},
            "radiation": {"enabled": False}}

    assert dm.engine_support(base).allowed, (
        "a straightforward configuration was blocked")

    no_thermal = {"thermal": dict(base["thermal"], run_thermal_mesh=False),
                  "radiation": {"enabled": False}}
    assert not dm.engine_support(no_thermal).allowed

    no_stats = {"thermal": dict(base["thermal"], compute_statistics=False),
                "radiation": {"enabled": False}}
    assert not dm.engine_support(no_stats).allowed

    cpu_coupled = {"thermal": dict(base["thermal"], use_gpu=False),
                   "radiation": {"enabled": True}}
    assert not dm.engine_support(cpu_coupled).allowed, (
        "facet-to-facet exchange on the processor solver was allowed. The "
        "engine stops on it because the exchange loop re-solves selected "
        "facets through a path that does not update the depth maps")

    gpu_coupled = {"thermal": dict(base["thermal"], use_gpu=True),
                   "radiation": {"enabled": True}}
    assert dm.engine_support(gpu_coupled).allowed, (
        "the graphics-card thermal path DOES carry the depth accumulator "
        "through the exchange loop; blocking it would refuse the production "
        "case")


def test_a_typo_does_not_switch_the_product_off():
    """The sharpest one. An empty list is OFF, so a dropped entry is silence.

    Type four depths, fumble one, and a version that discarded the bad entry
    would leave three -- or, with one depth, none at all, which is a
    surface-only run that nothing announced.
    """
    from gui_core import depth_maps as dm
    depths, error = dm.parse_depth_list("0.05, deep, 0.25")
    assert error, "a word where a depth belongs produced no message"
    assert len(depths) == 3, (
        f"the unparseable entry was dropped: {depths!r}. An empty list is how "
        f"this feature is switched off")
    assert "deep" in depths
    assert dm.validate_request(depths, ["avg"]), (
        "the kept entry is not refused, so it would reach a run")

    only_bad, error = dm.parse_depth_list("deep")
    assert only_bad == ["deep"] and error, (
        "one bad entry became an empty list, which is this feature OFF")


def test_the_depth_text_survives_a_round_trip():
    from gui_core import depth_maps as dm
    for text in ("0, 0.05, 0.25", "0.5", "0 0.1 0.2"):
        depths, error = dm.parse_depth_list(text)
        assert not error
        again, error = dm.parse_depth_list(dm.format_depth_list(depths))
        assert not error and again == depths, (
            f"{text!r} did not survive the trip through the field")


# ==========================================================================
# 3. Nothing leaks into a configuration that did not ask for it
# ==========================================================================

def test_a_run_without_depths_writes_no_depth_key():
    from gui_core.config_manager import ConfigManager
    manager = ConfigManager()
    written = manager._config_for_yaml()["output"]
    stray = sorted(k for k in written if "map_depth" in k)
    assert not stray, (
        f"a configuration that asks for no depth maps carries {stray}. An "
        f"empty depth list is the engine's own 'off', so every project file "
        f"this application has ever saved would change on its next save")


def test_the_panels_own_product_selector_is_never_written():
    from gui_core.config_manager import ConfigManager
    from gui_core import depth_maps as dm
    manager = ConfigManager()
    manager.set_value(dm.DEPTHS_KEY, [0.05])
    manager.set_value(dm.PRODUCT_KEY, dm.PRODUCT_TIME_SERIES)
    written = manager._config_for_yaml()["output"]
    assert dm.PRODUCT_LEAF not in written, (
        "the panel's product selector reached the configuration file. The "
        "engine has no such key, and a key nothing reads cannot be told apart "
        "from a setting being honored")


def test_the_two_companion_settings_are_written_only_when_changed():
    """Opening somebody's file and saving it must not add lines they did not
    write. The engine's own default is all three statistics and snapping on,
    so writing those states a preference the author never expressed."""
    from gui_core.config_manager import ConfigManager
    from gui_core import depth_maps as dm
    manager = ConfigManager()
    manager.set_value(dm.DEPTHS_KEY, [0.0, 0.05])
    written = manager._config_for_yaml()["output"]
    assert written.get(dm.DEPTHS_LEAF) == [0.0, 0.05]
    assert dm.SNAP_LEAF not in written and dm.STATS_LEAF not in written, (
        f"the defaults were written out: {sorted(written)}")

    manager.set_value(dm.SNAP_KEY, False)
    manager.set_value(dm.STATS_KEY, ["avg", "max"])
    written = manager._config_for_yaml()["output"]
    assert written[dm.SNAP_LEAF] is False
    assert written[dm.STATS_LEAF] == ["avg", "max"], (
        "a non-default statistic list did not reach the file, so the run "
        "would write three files per depth when two were asked for")


def test_the_rest_of_the_output_section_is_untouched():
    """The depth keys are selective; nothing else in the section became so."""
    from gui_core.config_manager import ConfigManager
    manager = ConfigManager()
    written = manager._config_for_yaml()["output"]
    for key in ("directory", "format", "vtk_mode", "include_primary_ir",
                "locations_file", "depths_file", "series_output"):
        assert key in written, (
            f"output.{key} stopped being written when the depth keys were "
            f"made selective")


# ==========================================================================
# 4. The comment-preserving round trip
# ==========================================================================

_SAMPLE = """\
# A configuration with the author's own comments.
simulation:
  body: Moon          # the body
  timesteps: 48
thermal:
  run_thermal_mesh: true
  compute_statistics: true
output:
  directory: output   # where the run lands
  map_depths_m: [0.0, 0.05]
"""


def _round_trip(edit=None):
    from gui_core.config_manager import ConfigManager
    manager = ConfigManager()
    with tempfile.TemporaryDirectory() as scratch:
        path = Path(scratch) / "sample.yaml"
        path.write_text(_SAMPLE)
        manager.load_from_yaml(str(path))
        if edit is not None:
            edit(manager)
        plan = manager.plan_save_edits()
        if plan is None:
            raise Skip("ruamel.yaml is unavailable, so the comment-preserving "
                       "save path cannot be exercised here")
        return manager, plan


def test_opening_a_file_with_depths_and_saving_it_changes_nothing():
    """Displaying a configuration is not an edit to it.

    The depth field's own handler parses what it shows, so without blocked
    signals a file spelling a depth `5e-2` would come back `0.05` on a save
    nobody asked for.
    """
    _manager, (updates, removals) = _round_trip()
    depth_updates = {k: v for k, v in updates.items() if "map_depth" in k}
    assert not depth_updates, (
        f"merely opening the file planned these writes: {depth_updates}")
    assert not [k for k in removals if "map_depth" in k], (
        f"opening the file planned to delete a depth key: {removals}")


def test_a_file_that_names_only_the_depth_list_does_not_gain_the_others():
    """The seeding rule, stated as what it prevents.

    ``_SAMPLE`` names ``map_depths_m`` and neither companion. The merged
    defaults supply both, and they are the ENGINE's defaults, so the writer
    leaves them out -- and the author's file does not grow two lines they never
    wrote and would then have to maintain.
    """
    _manager, (updates, _removals) = _round_trip()
    gained = sorted(k for k in updates
                    if k in ("output.map_depth_stats",
                             "output.map_depth_snap_to_node"))
    assert not gained, f"the file gained {gained} it never mentioned"


def test_changing_a_depth_writes_that_line_and_no_other():
    from gui_core import depth_maps as dm

    def edit(manager):
        manager.set_value(dm.DEPTHS_KEY, [0.0, 0.05, 0.25])

    _manager, (updates, removals) = _round_trip(edit)
    assert updates.get("output.map_depths_m") == [0.0, 0.05, 0.25]
    assert set(updates) == {"output.map_depths_m"}, (
        f"changing the depth list also planned to write {sorted(updates)}")
    assert not removals


def test_clearing_the_depth_list_removes_the_line_the_file_carries():
    from gui_core import depth_maps as dm

    def edit(manager):
        manager.set_value(dm.DEPTHS_KEY, [])

    _manager, (updates, removals) = _round_trip(edit)
    assert "output.map_depths_m" in removals, (
        f"clearing the list left the file's own line in place: {removals}")
    assert not [k for k in updates if "map_depth" in k]


def test_a_depth_list_the_file_wrote_inline_stays_inline():
    """A one-line list must not become four lines on an edit to it.

    It is still the key the user changed, so this is not wrong output -- but
    the save's own guard rail recognizes an edited line by matching a YAML key
    at the start of it, and a sequence item has none. So a correct edit was
    ALSO announced as "layout the writer could not reproduce exactly", which
    is the guard rail crying wolf on the one edit the user actually made.
    Reproduced on a depth list before this was fixed.
    """
    try:
        from gui_core import yaml_roundtrip as yr
    except ImportError as exc:                       # pragma: no cover
        raise Skip(f"the round-trip writer is unavailable here ({exc})")
    if not yr.available():
        raise Skip("ruamel.yaml is unavailable here")
    with tempfile.TemporaryDirectory() as scratch:
        path = Path(scratch) / "inline.yaml"
        path.write_text("output:\n  directory: out   # keep me\n"
                        "  map_depths_m: [0.0, 0.05]\n")
        doc = yr.RoundTripDoc.load(path)
        doc.apply({"output.map_depths_m": [0.0, 0.05, 0.25]})
        text = doc.dump()
    assert "map_depths_m: [0.0, 0.05, 0.25]" in text, (
        f"an inline list was rewritten as a block sequence:\n{text}")
    assert "# keep me" in text

    # ...and a list the file wrote as a block stays a block.
    with tempfile.TemporaryDirectory() as scratch:
        path = Path(scratch) / "block.yaml"
        path.write_text("output:\n  map_depths_m:\n    - 0.0\n    - 0.05\n")
        doc = yr.RoundTripDoc.load(path)
        doc.apply({"output.map_depths_m": [0.0, 0.25]})
        text = doc.dump()
    assert "- 0.25" in text and "[0.0, 0.25]" not in text, (
        f"a block sequence was collapsed onto one line:\n{text}")


def test_the_saved_text_keeps_the_authors_comments():
    from gui_core import depth_maps as dm
    from gui_core.config_manager import ConfigManager
    manager = ConfigManager()
    with tempfile.TemporaryDirectory() as scratch:
        path = Path(scratch) / "sample.yaml"
        path.write_text(_SAMPLE)
        manager.load_from_yaml(str(path))
        if manager.plan_save_edits() is None:
            raise Skip("ruamel.yaml is unavailable here")
        manager.set_value(dm.DEPTHS_KEY, [0.0, 0.05, 0.25])
        manager.save_to_yaml(str(path))
        text = path.read_text()
    assert "# the body" in text and "# where the run lands" in text, (
        "a save after a depth edit dropped the author's comments")
    assert "map_depths_m" in text


# ==========================================================================
# 5. The controls
# ==========================================================================

def test_the_group_is_on_the_output_tab_and_visible():
    """⛔ `isVisibleTo`, NOT `isHidden`. `isHidden()` is the widget's own flag
    and is blind to a hidden ANCESTOR -- which is how thirty-six merged arms
    passed in front of a group nobody could see."""
    _manager, tab = _tab()
    from gui_core import depth_maps as dm
    assert tab.depth_group.isVisibleTo(tab)
    for key in (dm.DEPTHS_KEY, dm.SNAP_KEY, dm.PRODUCT_KEY,
                "output.map_depth_stat_avg", "output.map_depth_stat_max",
                "output.map_depth_stat_min"):
        widget = tab.widgets[key]
        assert widget.isVisibleTo(tab), f"{key} is not visible on the tab"
        assert widget.isEnabled(), (
            f"{key} is disabled. ⛔ `setChecked` and `setCurrentIndex` both "
            f"SUCCEED on a disabled widget, so a control left disabled here "
            f"would look live to every arm that does not click it")


def test_a_real_click_on_the_nearest_depth_switch_reaches_the_configuration():
    """⛔ `click()`, not `setChecked(True)`. setChecked succeeds on a DISABLED
    widget, so it cannot tell a live control from a dead one."""
    manager, tab = _tab()
    from gui_core import depth_maps as dm
    box = tab.widgets[dm.SNAP_KEY]
    before = box.isChecked()
    assert before is True, "depths should move to the nearest solved depth by default"
    box.click()
    _pump()
    assert box.isChecked() is not before, "the click did not change the box"
    assert manager.get_value(dm.SNAP_KEY) is False, (
        "the click changed the box and not the configuration")
    box.click()
    _pump()
    assert manager.get_value(dm.SNAP_KEY) is True


def test_a_real_click_on_each_statistic_reaches_the_configuration():
    manager, tab = _tab()
    from gui_core import depth_maps as dm
    tab.depth_edit.setText("0.05")
    _pump()
    assert sorted(manager.get_value(dm.STATS_KEY)) == ["avg", "max", "min"]
    tab.depth_stat_boxes["max"].click()
    _pump()
    assert manager.get_value(dm.STATS_KEY) == ["avg", "min"], (
        f"unticking the maximum gave {manager.get_value(dm.STATS_KEY)!r}; the "
        f"list must stay in the engine's own order")
    tab.depth_stat_boxes["max"].click()
    _pump()
    assert manager.get_value(dm.STATS_KEY) == ["avg", "max", "min"]


def test_typing_depths_reaches_the_configuration_as_numbers():
    manager, tab = _tab()
    from gui_core import depth_maps as dm
    tab.depth_edit.setText("0, 0.05, 0.25")
    _pump()
    stored = manager.get_value(dm.DEPTHS_KEY)
    assert stored == [0.0, 0.05, 0.25], (
        f"the field stored {stored!r}. The engine reads a sequence of numbers "
        f"here; storing the text would hand it a string")
    tab.depth_edit.setText("")
    _pump()
    assert manager.get_value(dm.DEPTHS_KEY) == []


def test_the_read_out_names_the_files_and_the_size():
    manager, tab = _tab()
    tab.depth_edit.setText("0.05, 0.25")
    _pump()
    text = tab.depth_summary.text()
    assert "temperature_z0p050m_avg.bin" in text, (
        f"the read-out does not name the files the run will write:\n{text}")
    assert "Product size" in text and "facets" in text, (
        f"the read-out carries no size:\n{text}")
    assert tab.depth_series_note.text(), (
        "the second read-out is empty; the per-step size is the whole reason "
        "there are two")


def test_the_read_out_says_so_when_no_depths_are_listed():
    _manager, tab = _tab()
    tab.depth_edit.setText("")
    _pump()
    text = tab.depth_summary.text()
    assert "surface products only" in text, (
        f"with no depths the read-out does not say what the run will do:\n"
        f"{text}")


def test_unticking_the_nearest_depth_switch_puts_both_error_budgets_on_screen():
    """The panel must never quote one product's error for the other's."""
    manager, tab = _tab()
    from gui_core import depth_maps as dm
    tab.depth_edit.setText("0.05")
    _pump()
    assert "Interpolation error: none" in tab.depth_summary.text()
    tab.widgets[dm.SNAP_KEY].click()
    _pump()
    text = tab.depth_summary.text()
    assert "at any one instant" in text and "in the maximum" in text, (
        f"only one error budget is shown:\n{text}")
    assert "0.195" in text and "0.085" in text, (
        f"the two figures are not both on screen:\n{text}")
    tab.widgets[dm.SNAP_KEY].click()
    _pump()


def test_the_read_out_carries_the_refusal_the_engine_would_make():
    manager, tab = _tab()
    tab.depth_edit.setText("0.05, 0.05")
    _pump()
    assert "listed twice" in tab.depth_summary.text(), (
        f"a repeated depth is not called out on the tab:\n"
        f"{tab.depth_summary.text()}")
    tab.depth_edit.setText("0.05, deep")
    _pump()
    assert "not a number" in tab.depth_summary.text()
    tab.depth_edit.setText("")
    _pump()


def test_the_size_read_out_follows_a_change_of_mesh():
    """The size depends on the MESH, which is set on a different tab.

    Each tab's `value_changed` is its own and nothing on the Mesh tab reaches
    here, so without a refresh when this tab comes forward the panel kept
    showing a small sphere's size after a large mesh was imported -- including
    "under the ceiling" for a request that had gone over it. The launch check
    was right the whole time; only the number in front of the user was stale.
    """
    manager, tab = _tab()
    manager.set_value("mesh.resolution", 4)
    manager.set_value("mesh.import_file", "")
    tab.depth_edit.setText("0.05")
    _pump()
    small = tab.depth_summary.text()
    assert "5,120 facets" in small, (
        f"the read-out does not carry the small mesh's facet count:\n{small}")

    manager.set_value("mesh.resolution", 7)
    _pump()
    assert tab.depth_summary.text() == small, (
        "this arm needs the read-out to be stale before the refresh, or it "
        "proves nothing")
    tab.showEvent(_show_event())
    _pump()
    grown = tab.depth_summary.text()
    assert "327,680 facets" in grown, (
        f"bringing the tab forward did not pick up the new mesh:\n{grown}")


def _show_event():
    from PyQt6.QtGui import QShowEvent
    return QShowEvent()


def test_choosing_the_per_step_series_says_it_is_not_written():
    manager, tab = _tab()
    from gui_core import depth_maps as dm
    tab.depth_edit.setText("0.05")
    _pump()
    combo = tab.widgets[dm.PRODUCT_KEY]
    index = combo.findText(dm.PRODUCT_TIME_SERIES)
    assert index >= 0, "the per-step choice is not offered at all"
    combo.setCurrentIndex(index)
    _pump()
    text = tab.depth_series_note.text()
    assert "summary" in text and "not the per-step series" in text, (
        f"choosing the per-step product does not say the model will not write "
        f"it:\n{text}")
    combo.setCurrentIndex(combo.findText(dm.PRODUCT_STATISTICS))
    _pump()


# --------------------------------------------------------------------------
# 5b. The viewport rectangle
#
# ⛔ THIS IS THE THIRD INSTRUMENT. `isHidden` was blind to a hidden ANCESTOR;
# `setChecked` was blind to a DISABLED widget; `isVisibleTo` is blind to SCROLL
# POSITION -- a control can be visible, enabled and three hundred pixels below
# the bottom of the window, and the user's honest description of that is "the
# options did not appear". The arms below ask the viewport, not Qt's flags.
#
# ⚠️ MEASURED, and stated because it changes what these arms are worth: the
# Output tab is about 640 px of content and this harness cannot give it a
# viewport shorter than 668 px, so at every real window size its scroll maximum
# is 0 and nothing in this group can be off screen. These arms therefore PIN
# the property rather than catch a live defect, and the ruler arm below shrinks
# the scroll area itself so the ruler is proved able to fail rather than
# assumed to be.
# --------------------------------------------------------------------------

def _scroll_of(tab):
    from PyQt6.QtWidgets import QScrollArea
    areas = tab.findChildren(QScrollArea)
    if not areas:
        raise Skip("this tab has no scroll area to measure against")
    return areas[0]


def _on_screen(tab, widget):
    from PyQt6.QtCore import QPoint
    if not widget.isVisibleTo(tab):
        return False
    viewport = _scroll_of(tab).viewport()
    top = widget.mapTo(viewport, QPoint(0, 0)).y()
    return (top + widget.height() > 0) and (top < viewport.height())


def _shown(width, height):
    manager, panel = _panel()
    manager.reset_to_defaults()
    tab = panel.tabs["output"]
    panel.resize(width, height)
    panel.stacked_widget.setCurrentWidget(tab)
    panel.show()
    _pump()
    tab.load_from_config()
    _scroll_of(tab).verticalScrollBar().setValue(0)
    _pump()
    return manager, tab


def test_the_viewport_ruler_can_fail():
    """⛔ FIRST, PROVE THE RULER IS NOT PAINTED ON.

    `_on_screen` has to report False for something `isVisibleTo` calls visible,
    or every arm below it is vacuous -- which is exactly how two earlier rounds
    of this defect got past a green suite. The scroll area is given a viewport
    shorter than the form so the bottom of it must read as off screen while Qt
    still says it is not hidden.
    """
    _manager, tab = _shown(1400, 900)
    scroll = _scroll_of(tab)
    original = scroll.maximumHeight()
    try:
        scroll.setFixedHeight(300)
        _pump()
        scroll.verticalScrollBar().setValue(0)
        _pump()
        bottom = tab.depth_series_note
        assert bottom.isVisibleTo(tab), (
            "this arm needs a control Qt calls visible")
        assert _on_screen(tab, bottom) is False, (
            "the viewport check says a control well below the window is on "
            "screen; it cannot detect the defect it exists for")
    finally:
        scroll.setMinimumHeight(0)
        scroll.setMaximumHeight(original)
        _pump()


def test_every_depth_control_is_inside_the_window():
    from gui_core import depth_maps as dm
    keys = [("depth list", dm.DEPTHS_KEY),
            ("nearest-depth switch", dm.SNAP_KEY),
            ("average", "output.map_depth_stat_avg"),
            ("maximum", "output.map_depth_stat_max"),
            ("minimum", "output.map_depth_stat_min"),
            ("form", dm.PRODUCT_KEY)]
    for width, height in ((2559, 1381), (2559, 1200), (2559, 1000)):
        _manager, tab = _shown(width, height)
        tab.depth_edit.setText("0, 0.05, 0.25")
        _pump()
        missing = [label for label, key in keys
                   if not _on_screen(tab, tab.widgets[key])]
        assert not missing, (
            f"at {width}x{height} these depth controls are outside the "
            f"window: {', '.join(missing)}")
        for label, widget in (("size read-out", tab.depth_summary),
                              ("per-step read-out", tab.depth_series_note)):
            assert _on_screen(tab, widget), (
                f"at {width}x{height} the {label} is outside the window, so "
                f"the number it carries reaches nobody")


# ==========================================================================
# 6. The launch check
# ==========================================================================

def _findings(**overrides):
    from gui_core import depth_maps as dm
    config = {
        "output": {"map_depths_m": [0.05], "map_depth_stats": ["avg"],
                   "map_depth_snap_to_node": True},
        "thermal": {"run_thermal_mesh": True, "compute_statistics": True,
                    "use_gpu": True},
        "radiation": {"enabled": False},
    }
    for dotted, value in overrides.items():
        section, _, leaf = dotted.partition("__")
        config.setdefault(section, {})[leaf] = value
    return dm.preflight_findings(config, n_faces=2397890, n_frames=1440)


def test_the_launch_check_is_silent_when_no_depths_are_asked_for():
    from gui_core import depth_maps as dm
    findings = dm.preflight_findings({"output": {}}, n_faces=100)
    assert findings.ok and not findings.info and not findings.line, (
        "the launch check spoke about a feature this run does not use")


def test_the_launch_check_reports_the_size_of_both_products():
    findings = _findings()
    assert "9.1 MiB" in findings.line, (
        f"the size line does not carry the product size: {findings.line!r}")
    assert findings.series_line and "GiB" in findings.series_line, (
        f"the per-step size is missing: {findings.series_line!r}")
    assert findings.line != findings.series_line, (
        "one line is standing in for both products")


def test_the_launch_check_refuses_the_per_step_series_with_its_size():
    """The 64 GiB class arrives as a message BEFORE the run, not as a crash."""
    from gui_core import depth_maps as dm
    findings = _findings(output__map_depth_product=dm.PRODUCT_TIME_SERIES)
    assert not findings.ok, "the per-step product was accepted"
    refusal = " ".join(findings.errors)
    assert "not the per-step series" in refusal
    assert "GiB" in refusal, (
        f"the refusal does not say how large the product would be, which is "
        f"the reason for it: {refusal}")


def test_the_launch_check_refuses_a_product_over_the_ceiling():
    """Above 8 GiB the engine stops, so the launch check must too.

    ⛔ `not findings.ok` IS THE ASSERTION. An earlier version of this arm
    checked only that the sentence appeared in the size LINE -- and it did,
    while `errors` stayed empty, `PreflightReport.ok` stayed True and the
    launcher started a run the engine exits in its first seconds. A message in
    a line nobody branches on is not a refusal.
    """
    from gui_core import depth_maps as dm
    many = [round(0.01 * (i + 1), 3) for i in range(300)]
    config = {"output": {"map_depths_m": many,
                         "map_depth_stats": ["avg", "max", "min"]},
              "thermal": {"run_thermal_mesh": True, "compute_statistics": True,
                          "use_gpu": True},
              "radiation": {"enabled": False}}
    findings = dm.preflight_findings(config, n_faces=2397890, n_frames=1440)
    assert not findings.ok, (
        "a product over the engine's own ceiling was accepted for launch")
    refusal = " ".join(findings.errors)
    assert "8.0 GiB" in refusal and "GiB" in refusal, (
        f"the refusal does not say what the ceiling is: {refusal}")
    assert "over the 8.0 GiB" in findings.line, (
        "the size read-out no longer mentions it either")

    # ...and a request under the ceiling is not refused.
    config["output"]["map_depths_m"] = [0.05, 0.25]
    assert dm.preflight_findings(config, n_faces=2397890,
                                 n_frames=1440).ok


def test_a_depth_list_that_is_not_a_list_is_refused_and_does_not_crash():
    """⛔ MEASURED on the engine, and the measurement changed the design.

    A hand-edited `map_depths_m: 0.05` -- a forgotten pair of brackets -- USED
    TO make the engine complete the run, exit 0, and write NO subsurface maps,
    saying nothing at all. The engine refuses it at config load now
    (PTM-DEPTHMAP-SCALAR-SILENT-OFF, config_parser.cpp), and this refusal stays
    here as well: it is answered beside the control that caused it rather than
    after a queued run has been dispatched, and a run started from an older
    binary still behaves the old way.

    The second half matters as much: `list(0.05)` RAISES. This is reached from
    a Qt slot, where an escaping exception is not an error message -- PyQt6
    calls qFatal and the window goes with any unsaved edits.
    """
    from gui_core import depth_maps as dm
    from gui_core.config_manager import ConfigManager

    for value in (0.05, "0.05, 0.25", True):
        config = {"output": {"map_depths_m": value},
                  "thermal": {"run_thermal_mesh": True,
                              "compute_statistics": True, "use_gpu": True},
                  "radiation": {"enabled": False}}
        depths, _stats, _snap, problems = dm.request_from_config(config)
        assert depths == [], (
            f"map_depths_m: {value!r} produced depths {depths!r}; a string "
            f"split into characters, or a scalar accepted, is this "
            f"application answering for the user")
        assert problems, f"map_depths_m: {value!r} was accepted in silence"
        findings = dm.preflight_findings(config, n_faces=1000, n_frames=10)
        assert not findings.ok, (
            f"map_depths_m: {value!r} passed the launch check; the engine "
            f"stops on it at config load, and an older binary finishes with "
            f"no subsurface product and no message")

    # ...and the whole application survives it, from load to save.
    manager = ConfigManager()
    with tempfile.TemporaryDirectory() as scratch:
        path = Path(scratch) / "scalar.yaml"
        path.write_text("output:\n  directory: out\n  map_depths_m: 0.05\n")
        manager.load_from_yaml(str(path))       # used to raise TypeError
        _ok, messages = manager.validate()
        assert any("not a list" in m for m in messages), (
            f"the configuration check said nothing: {messages}")
        written = manager._config_for_yaml()["output"]
    assert written["map_depths_m"] == 0.05, (
        "the user's own value was rewritten rather than left for them to fix")


def test_a_statistic_list_that_is_not_a_list_is_refused():
    """⛔ MEASURED: `map_depth_stats: avg` and a bare `map_depth_stats:` BOTH
    stop the engine, exit 1. Reading the null as "all three" would tick three
    boxes for a run that will not start.

    They stop it for DIFFERENT reasons, and until 2026-09-06 they reported the
    same one: the scalar reached "output.map_depth_stats is empty", which is a
    misdiagnosis of a file that plainly named a statistic
    (PTM-DEPTHMAP-STATS-SCALAR-MISDIAGNOSED). The scalar is now refused at
    config load as a shape error; the null still reaches the empty-list clause,
    where that sentence is true.
    """
    from gui_core import depth_maps as dm
    scalar = {"output": {"map_depths_m": [0.05], "map_depth_stats": "avg"},
              "thermal": {"run_thermal_mesh": True, "compute_statistics": True,
                          "use_gpu": True},
              "radiation": {"enabled": False}}
    _d, stats, _s, problems = dm.request_from_config(scalar)
    assert stats == [] and problems, (
        f"a scalar statistic list gave {stats!r}; ['a','v','g'] is what "
        f"list() of a string produces and is not three statistics")
    assert not dm.preflight_findings(scalar, n_faces=1000).ok

    null = {"output": {"map_depths_m": [0.05], "map_depth_stats": None},
            "thermal": {"run_thermal_mesh": True, "compute_statistics": True,
                        "use_gpu": True},
            "radiation": {"enabled": False}}
    _d, stats, _s, _p = dm.request_from_config(null)
    assert stats == [], (
        "a null statistic list was read as the default set; the engine clears "
        "its default at that key and then refuses the run")
    findings = dm.preflight_findings(null, n_faces=1000)
    assert not findings.ok and any("No statistic" in e for e in findings.errors)

    # An ABSENT key is the engine's default, and is not an error.
    absent = {"output": {"map_depths_m": [0.05]},
              "thermal": {"run_thermal_mesh": True, "compute_statistics": True,
                          "use_gpu": True},
              "radiation": {"enabled": False}}
    _d, stats, _s, problems = dm.request_from_config(absent)
    assert stats == ["avg", "max", "min"] and not problems
    assert dm.preflight_findings(absent, n_faces=1000).ok


def test_the_launch_check_warns_when_depths_are_not_snapped():
    findings = _findings(output__map_depth_snap_to_node=False)
    warning = " ".join(findings.warnings)
    assert "0.195" in warning and "0.15" in warning, (
        f"an interpolated depth was not weighed against the accuracy floor: "
        f"{warning!r}")


def test_the_launch_check_carries_the_engines_own_refusals():
    findings = _findings(thermal__use_gpu=False, radiation__enabled=True)
    assert not findings.ok
    assert any("exchange" in e for e in findings.errors), (
        f"the processor-plus-exchange refusal did not reach the launch "
        f"check: {findings.errors}")


def test_the_report_shows_both_depth_lines():
    """They travel as two lines, so neither can be read as the other."""
    try:
        from gui_core.preflight import PreflightReport
    except ImportError as exc:                       # pragma: no cover
        raise Skip(f"the launch check is unavailable here ({exc})")
    report = PreflightReport()
    report.depth_map_line = "Product size: 27.4 MiB"
    report.depth_map_series_line = "A value at every output step would be 64.3 GiB"
    lines = [line for line in report.summary_lines()
             if line.startswith("DEPTH-MAPS:")]
    assert len(lines) == 2, f"the two read-outs were merged: {lines}"


def test_the_configuration_check_refuses_what_the_engine_refuses():
    from gui_core.config_manager import ConfigManager
    from gui_core import depth_maps as dm
    manager = ConfigManager()
    manager.set_value(dm.DEPTHS_KEY, [0.05, 0.05])
    _ok, messages = manager.validate()
    assert any("listed twice" in m for m in messages), (
        f"the configuration check accepted a repeated depth: {messages}")

    manager = ConfigManager()
    manager.set_value(dm.DEPTHS_KEY, [0.05])
    manager.set_value("radiation.enabled", True)
    manager.set_value("thermal.use_gpu", False)
    _ok, messages = manager.validate()
    assert any("exchange" in m for m in messages), (
        f"the configuration check accepted a run the engine stops: {messages}")


TESTS = [
    test_the_naming_rule_is_the_engines,
    test_the_size_formula_is_the_engines,
    test_the_ceiling_and_the_default_are_the_engines,
    test_the_statistics_are_the_three_the_engine_writes,
    test_the_interpolation_error_figures_are_the_engines,
    test_every_grid_free_refusal_fires,
    test_two_depths_a_millimeter_apart_are_refused_when_not_snapping,
    test_a_valid_request_is_not_refused,
    test_the_grid_bound_clauses_are_left_to_the_engine,
    test_the_engine_preconditions_all_fire,
    test_a_typo_does_not_switch_the_product_off,
    test_the_depth_text_survives_a_round_trip,
    test_a_run_without_depths_writes_no_depth_key,
    test_the_panels_own_product_selector_is_never_written,
    test_the_two_companion_settings_are_written_only_when_changed,
    test_the_rest_of_the_output_section_is_untouched,
    test_opening_a_file_with_depths_and_saving_it_changes_nothing,
    test_a_file_that_names_only_the_depth_list_does_not_gain_the_others,
    test_changing_a_depth_writes_that_line_and_no_other,
    test_clearing_the_depth_list_removes_the_line_the_file_carries,
    test_a_depth_list_the_file_wrote_inline_stays_inline,
    test_the_saved_text_keeps_the_authors_comments,
    test_the_group_is_on_the_output_tab_and_visible,
    test_a_real_click_on_the_nearest_depth_switch_reaches_the_configuration,
    test_a_real_click_on_each_statistic_reaches_the_configuration,
    test_typing_depths_reaches_the_configuration_as_numbers,
    test_the_read_out_names_the_files_and_the_size,
    test_the_read_out_says_so_when_no_depths_are_listed,
    test_unticking_the_nearest_depth_switch_puts_both_error_budgets_on_screen,
    test_the_read_out_carries_the_refusal_the_engine_would_make,
    test_the_size_read_out_follows_a_change_of_mesh,
    test_choosing_the_per_step_series_says_it_is_not_written,
    test_the_viewport_ruler_can_fail,
    test_every_depth_control_is_inside_the_window,
    test_the_launch_check_is_silent_when_no_depths_are_asked_for,
    test_the_launch_check_reports_the_size_of_both_products,
    test_the_launch_check_refuses_the_per_step_series_with_its_size,
    test_the_launch_check_refuses_a_product_over_the_ceiling,
    test_a_depth_list_that_is_not_a_list_is_refused_and_does_not_crash,
    test_a_statistic_list_that_is_not_a_list_is_refused,
    test_the_launch_check_warns_when_depths_are_not_snapped,
    test_the_launch_check_carries_the_engines_own_refusals,
    test_the_report_shows_both_depth_lines,
    test_the_configuration_check_refuses_what_the_engine_refuses,
]


def main():
    failures = skipped = 0
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
    print(f"\n[test_gui_depth_maps] {total - failures - skipped}/{total} passed"
          + (f", {skipped} skipped" if skipped else "")
          + (f", {failures} failed" if failures else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
