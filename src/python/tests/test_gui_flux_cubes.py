#!/usr/bin/env python3
"""Flux cubes on the Moon local-mesh card: the arithmetic, the gate, the file.

A flux cube is a passive six-face irradiance probe standing on the terrain
(``docs/flux-cubes.md``). Turning them on must change no temperature anywhere,
and turning them off must change nothing about the configuration the GUI
writes -- so most of what is checked here is what does NOT happen.

The four properties that matter, and why each is here rather than trusted:

* **Nothing leaks.** With cubes off the emitted YAML carries no ``flux_cubes:``
  section at all. The engine's default is ``enabled: false`` and it reads the
  whole block under ``if (root["flux_cubes"])``, so silence and a block saying
  false are the same run -- and emitting the block would change the file every
  existing project writes, for no behavioral reason.

* **The arithmetic is anchored to delivered files, not derived.** The cube
  count and the product size are checked against numbers measured on real
  products: 1,257 cubes at the documented 25 m / 500 m grid, and two
  ``flux_cubes.fcb`` files whose byte counts are known.

* **The layout reader is checked against a real cache AND against a corrupted
  one.** A wrong record size in that reader does not raise; it lands in a
  neighboring block and returns text. The first version had ``PatchRecord`` at
  36 bytes instead of 40 and reported three location names that were slices of
  the provenance JSON -- all printable, so a printable-character check passed
  them. The length check is the arm that catches it, and
  ``test_a_truncated_layout_is_refused`` is what keeps that arm honest.

* **A run the engine would refuse is refused before it starts.** Every
  flux-cube refusal fires AFTER the thermal solve, so learning about one from
  the engine costs the whole run.

Needs PyQt6 and the panel stack for the card arms; the arithmetic and the file
reader run without either. Run directly::

    python3 src/python/tests/test_gui_flux_cubes.py
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "python"))
sys.path.insert(0, str(REPO_ROOT / "gui"))


class Skip(Exception):
    """A check that cannot run here, as distinct from one that failed."""


_APP = None
_PANEL = None
_MANAGER = None
_SCRATCH = None

#: The mesh every card arm uses. Shipped with the repository, 12,800 faces, and
#: the same one `examples/validation/baseline_crater80N_*.yaml` runs.
CRATER = REPO_ROOT / "data" / "meshes" / "lunar_crater_80N.ply"


def _panel():
    """One offscreen ConfigurationPanel, reused (building one costs ~0.4 s)."""
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
    _SCRATCH = tempfile.mkdtemp(prefix="ptm_fluxcube_test_")
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


def _card():
    """The Moon card, opened on the local-mesh mode with the shipped crater."""
    manager, panel = _panel()
    manager.reset_to_defaults()
    card = panel.tabs["card_moon"]
    manager.set_value("mesh.import_file", str(CRATER))
    manager.set_value("radiation.enabled", True)
    card.activate("local", "time_series")
    return manager, card


def _hidden(widget):
    """`isHidden`, never `isVisible`: offscreen no ancestor is shown, so
    `isVisible()` is vacuously False for every widget and a check written with
    it tests nothing."""
    return widget.isHidden()


# ==========================================================================
# 1. The arithmetic, anchored to delivered products
# ==========================================================================

def test_the_cube_count_is_the_tools_own_rule():
    from gui_core import flux_cubes as fc
    # MEASURED: `ptm_flux_cube_geom --spacing 25 --radius 500` places 1,257
    # cubes, on the A03MP026 site core and on the shipped 80N crater alike --
    # it is a property of the lattice, not of the mesh.
    assert fc.lattice_cube_count(25.0, 500.0) == 1257, (
        f"{fc.lattice_cube_count(25.0, 500.0)} cubes at the documented 25 m / "
        f"500 m grid; the tool places 1,257")
    # A negative radius is the tool's "no lattice, named locations only", and
    # zero still places the single origin cube. Those are different things and
    # the count has to tell them apart.
    assert fc.lattice_cube_count(25.0, -1.0) == 0
    assert fc.lattice_cube_count(25.0, 0.0) == 1
    # Degenerate input returns nothing rather than dividing by zero: this runs
    # from a spin box that a configuration can drive to any value.
    assert fc.lattice_cube_count(0.0, 500.0) == 0
    assert fc.lattice_cube_count("nonsense", "nonsense") == 0
    assert fc.lattice_cube_count(1.0, float("inf")) == 0
    assert fc.lattice_cube_count(float("nan"), 500.0) == 0


def test_the_fast_count_agrees_with_the_tools_double_loop_everywhere():
    """The column-by-column count against the loop it replaced.

    The obvious implementation is the tool's own nested loop, which is O(n^2)
    and was measured at 171 s for a 20 km grid at 1 m spacing -- on the GUI
    thread, per keystroke. Replacing it is only safe if the two agree on every
    grid, including the ones where a point sits exactly on the boundary circle,
    which is where a closed form and a comparison part company.
    """
    import math
    import time
    from gui_core import flux_cubes as fc

    def slow(spacing, radius):
        n_lat = int(math.floor(radius / spacing))
        r2 = radius * radius
        return sum(1
                   for i in range(-n_lat, n_lat + 1)
                   for j in range(-n_lat, n_lat + 1)
                   if (i * spacing) ** 2 + (j * spacing) ** 2 <= r2)

    grids = [(25.0, 500.0), (1.0, 10.0), (1.0, 10.5), (0.5, 7.0),
             (3.0, 3.0), (2.0, 2.0), (7.0, 100.0), (100.0, 300.0),
             (25.0, 25.0), (25.0, 24.999), (10.0, 50.0), (13.0, 91.0),
             (1.0, 100.0), (2.5, 47.5)]
    for spacing, radius in grids:
        assert fc.lattice_cube_count(spacing, radius) == slow(spacing, radius), (
            f"spacing {spacing} radius {radius}: fast count "
            f"{fc.lattice_cube_count(spacing, radius)}, the tool's own loop "
            f"gives {slow(spacing, radius)}")

    # ...and it has to be fast at the extreme the controls allow, because this
    # runs from a spin box's valueChanged.
    start = time.time()
    fc.lattice_cube_count(0.5, 50000.0)
    elapsed = time.time() - start
    assert elapsed < 2.0, (
        f"the largest grid the controls offer took {elapsed:.1f} s to count, "
        f"on the thread that draws the window")


def test_the_product_size_matches_two_delivered_files():
    """The estimate is checked against real files, not against its own algebra.

    Both delivered A03MP026 products, 1,257 cubes at 29 and at 720 output
    steps. The formula is short by the provenance JSON blob each file carries
    (1,119 and 1,120 bytes), which is not modeled -- so the estimate is a floor
    about a kilobyte below the file, and it must stay one.
    """
    from gui_core import flux_cubes as fc
    for n_steps, actual in ((29, 11_019_828), (720, 269_012_280)):
        predicted = fc.product_bytes(1257, n_steps)
        short_by = actual - predicted
        assert 0 < short_by <= 4096, (
            f"{n_steps} steps: predicted {predicted:,}, the delivered file is "
            f"{actual:,}, a difference of {short_by:,} bytes. The estimate "
            f"should be short only by the provenance record.")
    # A third anchor, produced by this branch's own engine run on the shipped
    # crater: 1,257 cubes, 118 output steps, file 44,249,284 bytes.
    short_by = 44_249_284 - fc.product_bytes(1257, 118)
    assert 0 < short_by <= 4096, (
        f"the 118-step crater product is off by {short_by:,} bytes")


def test_a_name_never_loses_its_spreadsheet_to_the_cap():
    """`write_csv_max_cubes` caps the grid fan-out and never a named point.

    Transcribed from the engine: as soon as any named location is present the
    pass raises the count back to every cube. A card that promised three files
    for a capped run with a named point would be wrong.
    """
    from gui_core import flux_cubes as fc
    assert fc.csv_files_written(1257, 0, 3) == 3
    assert fc.csv_files_written(1257, 0, 0) == 1257
    assert fc.csv_files_written(1257, 2, 3) == 1257, (
        "a cap dropped a named location, which the engine never does")


def test_the_height_rule_is_the_tools_own_length_scale():
    """0.1 x the MEDIAN EQUIVALENT EDGE, not 0.1 x sqrt(mean area).

    Measured on the shipped 80N crater: the median equivalent edge over the
    whole mesh is 26.86 m and sqrt(mean area) is 17.95 m -- a factor of 1.5,
    so the wrong statistic puts the cube a third of the way off. The tool
    itself measured 28.74 m inside the 500 m grid disk and resolved a 2.874 m
    height, which is what the estimate has to be close to and is not equal to.
    """
    from gui_core import flux_cubes as fc
    from gui_core.facet_scale import facet_scale
    if not CRATER.exists():
        raise Skip(f"{CRATER} is not in this checkout")
    config = {"mesh": {"import_file": str(CRATER)}, "simulation": {"body": "Moon"}}
    scale = facet_scale(config, REPO_ROOT)
    assert scale.median_edge_m is not None, "the mesh reader returned no edge"
    assert abs(scale.median_edge_m - 26.864) < 0.01, scale.median_edge_m
    assert abs(scale.meters - 17.947) < 0.01, (
        f"sqrt(mean area) moved: {scale.meters}")
    geometry = fc.auto_geometry(config, REPO_ROOT)
    assert abs(geometry.height_m - 2.6864) < 0.01, geometry.height_m
    assert geometry.height_m == geometry.edge_m
    assert not geometry.measured, (
        "auto_geometry reported itself as measured; it is an estimate over "
        "the whole mesh and the tool measures the grid disk")
    # Within 10% of the tool's own 2.8743 m, and NOT equal to it. If these ever
    # coincided exactly, one of the two definitions would have quietly changed.
    assert abs(geometry.height_m - 2.8743) / 2.8743 < 0.10
    assert geometry.height_m != 2.8743


# ==========================================================================
# 2. The layout file
# ==========================================================================

def _build_cache(tmp, extra=()):
    """Build a real `.fcg` with the geometry tool, or skip."""
    from ptm import paths
    from gui_core import flux_cubes as fc
    exe = paths.find_engine_executable(fc.GEOM_EXECUTABLE,
                                       search_roots=[REPO_ROOT])
    if exe is None:
        raise Skip(f"no {fc.GEOM_EXECUTABLE} in build/bin; build it with: "
                   f"cmake --build build --target {fc.GEOM_EXECUTABLE}")
    if not CRATER.exists():
        raise Skip(f"{CRATER} is not in this checkout")
    import subprocess
    out = Path(tmp) / "site.fcg"
    argv = [str(exe), "--core", str(CRATER), "--out", str(out),
            "--n-rays", "1024", "--threads", "4"] + list(extra)
    proc = subprocess.run(argv, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, timeout=900)
    assert proc.returncode == 0, proc.stdout.decode("utf-8", "replace")[-2000:]
    return out


def test_a_real_layout_reads_back_as_the_tool_wrote_it():
    from gui_core import flux_cubes as fc
    with tempfile.TemporaryDirectory() as tmp:
        # A small grid, so the arm is seconds rather than a minute.
        cache = _build_cache(tmp, ("--spacing", "100", "--radius", "300"))
        header = fc.read_cache_header(cache)
        assert header.ok, header.error
        assert header.n_cubes == fc.lattice_cube_count(100.0, 300.0), (
            f"the tool placed {header.n_cubes} cubes and this reader's rule "
            f"predicts {fc.lattice_cube_count(100.0, 300.0)}")
        assert header.n_faces_core == 12800, header.n_faces_core
        assert header.n_named == 0 and header.names == ()
        assert abs(header.height_m - 0.1 * header.local_median_edge_m) < 1e-9
        assert abs(header.height_factor - 0.1) < 1e-12


def test_named_locations_come_back_by_name():
    from gui_core import flux_cubes as fc
    with tempfile.TemporaryDirectory() as tmp:
        pads = Path(tmp) / "pads.csv"
        pads.write_text("name,lat_deg,lon_deg\n"
                        "pad_alpha,80.0020,0.0000\n"
                        "pad_bravo,79.9980,0.0500\n"
                        "pad_charlie,80.0000,-0.0500\n")
        cache = _build_cache(tmp, ("--radius", "-1",
                                   "--locations", str(pads)))
        header = fc.read_cache_header(cache)
        assert header.ok, header.error
        assert header.n_cubes == 3, header.n_cubes
        assert header.names == ("pad_alpha", "pad_bravo", "pad_charlie"), (
            f"the names block read back as {header.names}")
        assert header.radius_m < 0, (
            "a named-locations layout should record no lattice")


def test_a_truncated_layout_is_refused():
    """The negative control for the reader's offsets.

    Named-block offsets are computed from record sizes, and a wrong size does
    not raise -- it reads text out of the provenance JSON. This is the arm that
    catches it: the file's own header describes a length, and a file that does
    not have that length is refused rather than parsed.
    """
    from gui_core import flux_cubes as fc
    with tempfile.TemporaryDirectory() as tmp:
        pads = Path(tmp) / "pads.csv"
        pads.write_text("pad_alpha,80.0020,0.0000\n")
        cache = _build_cache(tmp, ("--radius", "-1", "--locations", str(pads)))
        good = fc.read_cache_header(cache)
        assert good.ok and good.names == ("pad_alpha",), good.error

        clipped = Path(tmp) / "clipped.fcg"
        clipped.write_bytes(cache.read_bytes()[:-8])
        header = fc.read_cache_header(clipped)
        assert not header.ok, (
            "a file eight bytes shorter than its own header describes was "
            "read as a valid layout")
        assert "bytes" in header.error

        # ...and something that is not a layout at all.
        foreign = Path(tmp) / "foreign.fcg"
        foreign.write_bytes(b"PLY\n" + b"\x00" * 4096)
        header = fc.read_cache_header(foreign)
        assert not header.ok and "not a flux-cube" in header.error

        missing = fc.read_cache_header(Path(tmp) / "nope.fcg")
        assert not missing.ok and "does not exist" in missing.error


# ==========================================================================
# 3. The engine's own preconditions
# ==========================================================================

def _runnable_config():
    return {"simulation": {"body": "Moon", "timesteps": 118},
            "mesh": {"import_file": str(CRATER)},
            "radiation": {"enabled": True},
            "thermal": {}}


def test_a_runnable_configuration_is_allowed():
    from gui_core import flux_cubes as fc
    policy = fc.engine_support(_runnable_config())
    assert policy.allowed, policy.blockers
    assert "passive" in policy.reason


def test_every_engine_refusal_is_caught_before_the_run():
    """One arm per hard refusal in the post-solve pass.

    Each of these aborts the engine AFTER the thermal solve, so a gate that
    cannot see one costs a full run to discover it.
    """
    from gui_core import flux_cubes as fc

    cases = {
        "radiation off": ("radiation", "enabled", False, "infrared exchange"),
        "single radiosity": ("radiation", "band_separated", False,
                             "combined radiosity"),
        "no imported mesh": ("mesh", "import_file", "", "imported terrain"),
        "checkpoint look-ahead": ("thermal", "checkpoint_lookahead_steps", 1,
                                  "one piece"),
    }
    for label, (section, key, value, expect) in cases.items():
        config = _runnable_config()
        config.setdefault(section, {})[key] = value
        policy = fc.engine_support(config)
        assert not policy.allowed, f"{label} was allowed"
        assert any(expect in b for b in policy.blockers), (
            f"{label}: the refusal does not explain itself -- "
            f"{policy.blockers}")

    # And the one that is deliberately NOT a refusal: cubes read the band
    # series in memory, not the per-step frames, and the two settings were
    # measured to give a byte-identical product.
    config = _runnable_config()
    config["output"] = {"skip_time_series": True}
    assert fc.engine_support(config).allowed, (
        "skip_time_series was treated as a blocker; it is not one")


def test_the_gate_reads_a_yaml_spelling_the_engine_accepts():
    """`radiation: {enabled: 'true'}` is a string to PyYAML and a bool to the
    engine. A gate that refused it would refuse a configuration that runs."""
    from gui_core import flux_cubes as fc
    for spelling in (True, "true", "True", "on", "yes", 1):
        config = _runnable_config()
        config["radiation"]["enabled"] = spelling
        assert fc.engine_support(config).allowed, (
            f"radiation.enabled spelled {spelling!r} was read as off")
    for spelling in (False, "false", "off", "no", 0, None):
        config = _runnable_config()
        config["radiation"]["enabled"] = spelling
        assert not fc.engine_support(config).allowed, (
            f"radiation.enabled spelled {spelling!r} was read as on")


# ==========================================================================
# 4. Emission
# ==========================================================================

def test_cubes_off_writes_no_section_at_all():
    """The leak gate. Silence and `enabled: false` are the same run, and a
    block of seven defaults in every configuration is not."""
    manager, _panel_ = _panel()
    manager.reset_to_defaults()
    written = manager._config_for_yaml()
    assert "flux_cubes" not in written, (
        f"a default project emits a flux_cubes: section: "
        f"{written['flux_cubes']}")
    # ...and it stays out with placement settings edited but the switch off.
    manager.set_value("flux_cubes.spacing_m", 10.0)
    manager.set_value("flux_cubes.mode", "named locations")
    manager.set_value("flux_cubes.locations",
                      [{"name": "pad", "lat_deg": 80.0, "lon_deg": 0.0}])
    assert "flux_cubes" not in manager._config_for_yaml()


def test_the_grid_mode_emits_the_seven_engine_keys_and_nothing_else():
    manager, _panel_ = _panel()
    manager.reset_to_defaults()
    manager.set_value("flux_cubes.enabled", True)
    manager.set_value("flux_cubes.cache_file", "site.fcg")
    manager.set_value("flux_cubes.spacing_m", 25.0)
    manager.set_value("flux_cubes.radius_m", 500.0)
    block = manager._config_for_yaml().get("flux_cubes")
    assert block is not None, "cubes on and no section written"
    assert set(block) == {"enabled", "cache_file", "write_binary",
                          "write_csv", "write_csv_max_cubes",
                          "shadow_probe_rays"}, sorted(block)
    assert block["enabled"] is True and block["cache_file"] == "site.fcg"
    # The geometry keys live in the layout file, and the engine reads them back
    # out of it. A YAML copy would be a second source of truth for one number,
    # and `config_parser.cpp` reports them as ignored.
    for dead in ("spacing_m", "radius_m", "mode", "height_factor"):
        assert dead not in block, (
            f"{dead} was emitted; the engine ignores it and it would read as "
            f"a setting")


def test_the_named_mode_emits_its_locations():
    manager, _panel_ = _panel()
    manager.reset_to_defaults()
    manager.set_value("flux_cubes.enabled", True)
    manager.set_value("flux_cubes.cache_file", "site.fcg")
    manager.set_value("flux_cubes.mode", "named locations")
    manager.set_value("flux_cubes.locations", [
        {"name": "pad_alpha", "lat_deg": 80.002, "lon_deg": 0.0},
        {"name": "pad_bravo", "lat_deg": 79.998, "lon_deg": 0.05,
         "height_m": 2.0, "edge_m": 1.0},
    ])
    block = manager._config_for_yaml()["flux_cubes"]
    assert [row["name"] for row in block["locations"]] == ["pad_alpha",
                                                           "pad_bravo"]
    assert block["locations"][1]["height_m"] == 2.0
    assert "mode" not in block, "the placement mode is GUI state, not a key"
    # An empty list is what "no named locations" already means; writing it
    # would put a key in the file that says nothing.
    manager.set_value("flux_cubes.locations", [])
    assert "locations" not in manager._config_for_yaml()["flux_cubes"]


def test_a_grid_only_run_does_not_ask_for_points_it_has_no_layout_for():
    """The mode gates whether `locations` is written, and the reason is sharp.

    Add three points, switch back to the grid, build a grid-only layout, save.
    If the leftover rows were still emitted, the engine would check the layout
    for three names it does not contain and refuse -- at the END of the run,
    with the table that caused it hidden by the very mode that made it stale.
    Found in review before it shipped.

    Switching modes must not DISCARD the points either: they are still in the
    configuration and come back with the mode.
    """
    manager, _panel_ = _panel()
    manager.reset_to_defaults()
    manager.set_value("flux_cubes.enabled", True)
    manager.set_value("flux_cubes.cache_file", "site.fcg")
    manager.set_value("flux_cubes.locations",
                      [{"name": "pad_alpha", "lat_deg": 80.0, "lon_deg": 0.0}])

    manager.set_value("flux_cubes.mode", "grid")
    assert "locations" not in manager._config_for_yaml()["flux_cubes"], (
        "a grid-only run asked the engine to verify a named point its layout "
        "does not contain")
    for mode in ("named locations", "grid and named locations"):
        manager.set_value("flux_cubes.mode", mode)
        block = manager._config_for_yaml()["flux_cubes"]
        assert [r["name"] for r in block["locations"]] == ["pad_alpha"], mode
    manager.set_value("flux_cubes.mode", "grid")
    assert [r["name"] for r in manager.get_value("flux_cubes.locations")] \
        == ["pad_alpha"], "switching the mode threw the point list away"


def test_the_probe_setting_is_written_as_a_number():
    """Its control is a combo, and a combo stores the text it shows."""
    manager, _panel_ = _panel()
    manager.reset_to_defaults()
    manager.set_value("flux_cubes.enabled", True)
    manager.set_value("flux_cubes.cache_file", "site.fcg")
    for shown, expected in (("5", 5), ("1", 1), (5, 5), (1.0, 1)):
        manager.set_value("flux_cubes.shadow_probe_rays", shown)
        assert manager._config_for_yaml()["flux_cubes"]["shadow_probe_rays"] \
            == expected, shown
    # Unreadable: left out, so the engine's own default applies, rather than a
    # number invented here.
    manager.set_value("flux_cubes.shadow_probe_rays", "five")
    assert "shadow_probe_rays" not in manager._config_for_yaml()["flux_cubes"]


def test_the_engine_accepts_both_modes():
    """`--validate-config` on the real engine, for a grid and for named points.

    The GUI's own opinion of a configuration is not evidence that the engine
    parses it; this hands the emitted file to the engine's own parser.
    """
    from ptm import paths
    from gui_core import preflight
    exe = paths.find_engine_executable("ptm_thermal_sim",
                                       search_roots=[REPO_ROOT])
    if exe is None:
        raise Skip("no ptm_thermal_sim in build/bin; build it with: "
                   "cmake --build build --target ptm_thermal_sim")
    manager, _panel_ = _panel()
    with tempfile.TemporaryDirectory() as tmp:
        cache = _build_cache(tmp, ("--spacing", "200", "--radius", "300"))
        for mode, locations in (("grid", []),
                                ("named locations",
                                 [{"name": "pad_alpha", "lat_deg": 80.002,
                                   "lon_deg": 0.0}])):
            manager.reset_to_defaults()
            manager.set_value("simulation.timesteps", 4)
            manager.set_value("simulation.duration_days", 1.0)
            manager.set_value("mesh.import_file", str(CRATER))
            manager.set_value("radiation.enabled", True)
            manager.set_value("flux_cubes.enabled", True)
            manager.set_value("flux_cubes.mode", mode)
            manager.set_value("flux_cubes.cache_file", str(cache))
            manager.set_value("flux_cubes.locations", locations)
            # `_config_for_yaml()`, which is what the GUI's own Check
            # Configuration button hands the engine and what a run writes.
            # `get_config()` is the LIVE view, still carrying the GUI-synthetic
            # keys -- validating that would ask the engine about a file nobody
            # will ever hand it, and the engine would correctly report the
            # synthetic keys as dead.
            status, text = preflight.engine_validate_config(
                manager._config_for_yaml(), exe, REPO_ROOT)
            assert status == "accepted", f"{mode}: {status}: {text[-1500:]}"
            # The parser's own dead-key warner must have nothing to say about
            # this block: every key the card writes is one the engine reads.
            assert "key(s) under `flux_cubes:` are read by nothing" not in text, (
                f"{mode}: the engine reported dead keys under flux_cubes:\n"
                f"{text[-1500:]}")


def test_opening_a_project_takes_the_placement_from_the_layout_file():
    """The layout is the record of what was placed, and the file is not.

    Two things this pins, both of which were wrong in the first draft:

    * the path is RESOLVED the way every other path in this application is, so
      a project-relative `cache_file` -- which is what a shipped configuration
      carries -- is actually found. `read_cache_header` returns a refusal
      rather than raising, so an unresolved path failed silently and the
      placement fields kept showing defaults;
    * a layout holding a grid AND named points opens in the combined mode. It
      used to open in "named locations", which hid the grid fields and made a
      rebuild write `--radius -1` and discard 1,257 lattice cubes.
    """
    import yaml
    from gui_core.config_manager import ConfigManager
    from gui_core import flux_cubes as fc
    with tempfile.TemporaryDirectory() as tmp:
        pads = Path(tmp) / "pads.csv"
        pads.write_text("name,lat_deg,lon_deg\npad_alpha,80.002,0.0\n")
        cache = _build_cache(tmp, ("--spacing", "150", "--radius", "400",
                                   "--locations", str(pads)))
        header = fc.read_cache_header(cache)
        assert header.ok and header.n_named == 1 and header.n_cubes > 1, (
            "the probe layout does not hold both a grid and a named point")

        project = Path(tmp) / "project.yaml"
        project.write_text(yaml.dump({
            "simulation": {"body": "Moon", "timesteps": 24},
            "mesh": {"import_file": str(CRATER)},
            "radiation": {"enabled": True},
            # RELATIVE, resolved against the project root -- which is what a
            # shipped configuration carries.
            "flux_cubes": {"enabled": True,
                           "cache_file": os.path.relpath(cache, REPO_ROOT),
                           "locations": [{"name": "pad_alpha",
                                          "lat_deg": 80.002,
                                          "lon_deg": 0.0}]},
        }))
        manager = ConfigManager()
        manager.load_from_yaml(str(project))
        assert manager.get_value("flux_cubes.mode") == fc.MODE_BOTH, (
            f"a layout holding a grid and a named point opened as "
            f"{manager.get_value('flux_cubes.mode')!r}")
        assert abs(manager.get_value("flux_cubes.spacing_m") - 150.0) < 1e-9, (
            f"the grid spacing came from this application's defaults, not "
            f"from the layout: {manager.get_value('flux_cubes.spacing_m')}")
        assert abs(manager.get_value("flux_cubes.radius_m") - 400.0) < 1e-9


# ==========================================================================
# 5. The card
# ==========================================================================

def test_the_group_is_on_the_card_in_both_modes_and_says_which():
    """⛔ REPLACES an arm that asserted the OPPOSITE, and that is the point.

    It used to read `assert _hidden(card.flux_group)` on the global mode with
    the cubes off, encoding a deliberate decision: only a local-mesh run can
    produce cubes, so do not offer the group elsewhere. The decision was wrong
    in the one way that matters -- it removed the explanation along with the
    control. See `MoonCard._apply_flux_group_visibility` for the measurement.

    The engine rule has NOT changed: a global run still cannot produce cubes.
    That is asserted on the FLOW card in section 10 -- `card.activate(...)`
    leaves `mesh.import_file` exactly where it was, deliberately (see
    `test_a_live_setting_is_never_hidden_by_the_mode`), so a card driven this
    way is a global mode holding a local mesh and `engine_support` allows it.
    This arm is about the group being on screen, and nothing else.
    """
    manager, card = _card()
    assert not _hidden(card.flux_group), (
        "the flux-cube group is hidden on the local-mesh card, where the "
        "layout tool has a mesh to work with")
    manager.set_value("flux_cubes.enabled", False)
    card.activate("global", "time_series")
    assert not _hidden(card.flux_group), (
        "the flux-cube group is off the screen on a global run with the cubes "
        "off, so nothing on the card says the mode is the reason")
    assert card.widgets["flux_cubes.enabled"].isVisibleTo(card), (
        "the enable switch is off screen on the global mode")


def test_a_live_setting_is_never_hidden_by_the_mode():
    """PAUL'S DEFECT, 2026-08-31, and the arm that would have caught it.

    Turn the cubes on with the local mesh, then move the card to the global
    mode. The group used to disappear WHOLE -- and nothing cleared
    `flux_cubes.enabled`, and `mesh.import_file` survives the mode switch, so
    `engine_support` still passed and the launch check refused with

        flux cubes: no cube layout file is set. ... Build it from the card

    against a card that was showing no such thing, and no control anywhere in
    the window could set it, build it or turn it off. Reported as "there's no
    option to set the layout file", and it is exactly that.

    ⛔ `isVisibleTo`, NOT `isHidden`. `isHidden()` is the widget's OWN explicit
    flag and was False for every one of these controls the whole time they were
    off screen -- their ANCESTOR was hidden. That is why the 36 arms already in
    this file all passed on the broken build, and it is why `_hidden` is not
    used here.
    """
    manager, card = _card()
    manager.set_value("flux_cubes.enabled", True)
    card.refresh_flux_cubes()
    switch = card.widgets["flux_cubes.enabled"]
    assert switch.isVisibleTo(card)

    card.activate("global", "time_series")
    assert manager.get_value("flux_cubes.enabled") is True, (
        "the mode switch changed the setting; opening a mode is not an edit")
    assert switch.isVisibleTo(card), (
        "flux_cubes.enabled is on and its switch is off screen: the setting "
        "cannot be seen or cleared, and the launch check refuses over it")
    assert card.flux_advanced.isVisibleTo(card), (
        "the layout-file section is off screen while the setting that needs "
        "it is on")
    # And the launch check still refuses this run -- but now over something
    # the window can show, rather than over a control that is not there.
    assert switch.isVisibleTo(card) and not _hidden(card.flux_group)


def test_the_settings_appear_only_once_the_switch_is_on():
    """One extra checkbox until it is asked for -- and everything on screen the
    moment a configuration says the cubes are on."""
    manager, card = _card()
    manager.set_value("flux_cubes.enabled", False)
    card.refresh_flux_cubes()
    assert _hidden(card.flux_body), (
        "the whole group is on screen with the cubes switched off")
    manager.set_value("flux_cubes.enabled", True)
    card.refresh_flux_cubes()
    assert not _hidden(card.flux_body), (
        "a configuration with the cubes on shows nothing to change")


def test_the_mode_shows_the_controls_that_mode_uses():
    from gui_core import flux_cubes as fc
    manager, card = _card()
    manager.set_value("flux_cubes.enabled", True)
    manager.set_value("flux_cubes.mode", fc.MODE_LATTICE)
    card.refresh_flux_cubes()
    assert not _hidden(card.flux_lattice) and _hidden(card.flux_named)
    manager.set_value("flux_cubes.mode", fc.MODE_LOCATIONS)
    card.refresh_flux_cubes()
    assert _hidden(card.flux_lattice) and not _hidden(card.flux_named)
    # The layout tool appends named points after the lattice in ONE file, so
    # the card has to be able to ask for both -- without this mode, opening a
    # project whose layout holds both hid the grid fields and a rebuild threw
    # the lattice away.
    manager.set_value("flux_cubes.mode", fc.MODE_BOTH)
    card.refresh_flux_cubes()
    assert not _hidden(card.flux_lattice) and not _hidden(card.flux_named)
    # An older project, or a hand-edited file, can spell it differently. The
    # combo and the card must not disagree about which layout will be built.
    for spelling in ("lattice", "LATTICE", "grid"):
        manager.set_value("flux_cubes.mode", spelling)
        card.refresh_flux_cubes()
        assert not _hidden(card.flux_lattice), spelling
    for spelling in ("locations", "named_locations", "Named Locations"):
        manager.set_value("flux_cubes.mode", spelling)
        card.refresh_flux_cubes()
        assert not _hidden(card.flux_named), spelling


def test_the_height_is_recomputed_when_the_mesh_changes():
    """It is read off the mesh, so a different mesh is a different number."""
    manager, card = _card()
    manager.set_value("flux_cubes.enabled", True)
    card.refresh_flux_cubes()
    crater80 = card.flux_height.text()
    assert "2.69 m" in crater80, crater80
    assert "estimate" in crater80, (
        "the height is presented as measured; the layout tool measures the "
        "grid area and this is taken over the whole mesh")

    other = REPO_ROOT / "data" / "meshes" / "lunar_crater_85N.ply"
    if not other.exists():
        raise Skip(f"{other} is not in this checkout")
    # Through the WIDGET, which is how a user changes it: the card recomputes
    # from `mesh.import_file`'s own textChanged, and a test that set the
    # configuration directly would pass on a card that never listens.
    card.widgets["mesh.import_file"].setText(str(other))
    assert card.flux_height.text() != crater80, (
        f"the cube height did not move when the mesh changed: {crater80}")

    card.widgets["mesh.import_file"].setText("")
    assert "not known yet" in card.flux_height.text(), (
        f"with no mesh the card still quotes a height: "
        f"{card.flux_height.text()}")


def test_the_card_refuses_a_run_the_engine_would_refuse_and_edits_nothing():
    """Showing a configuration is not editing it -- the beaming gate's rule.

    A file that asks for cubes on a run the engine cannot serve keeps its
    setting, and the card says the rules disagree with it.

    The refusal is SAID, not enforced by a dead control. This arm used to
    assert the switch went insensitive, which is what swallowed Paul's click
    on 2026-08-31; the guarantee that replaces it is the launch check, and it
    is asserted here rather than assumed.
    """
    from gui_core import flux_cubes as fc
    manager, card = _card()
    manager.set_value("flux_cubes.enabled", True)
    manager.set_value("radiation.enabled", False)
    card.refresh_flux_cubes()
    assert manager.get_value("flux_cubes.enabled") is True, (
        "the card turned the cubes off by itself")
    assert card.widgets["flux_cubes.enabled"].isEnabled(), (
        "the switch went dead on a run that cannot produce cubes; a click on "
        "it is then a silent no-op and nothing on the card says so")
    assert "CANNOT PRODUCE" in card.flux_note.text(), card.flux_note.text()
    # What actually protects the user: the launch check turns the same
    # blocker into an ERROR, so the run does not start.
    errors = fc.preflight_findings(
        manager.get_config(), REPO_ROOT, lambda p, r: Path(p),
        lambda p, r: Path(p), lambda p: False, lambda p: False,
        lambda p: 0).errors
    assert any("Facet-to-facet infrared exchange is off" in e
               for e in errors), errors
    manager.set_value("radiation.enabled", True)
    card.refresh_flux_cubes()
    assert card.widgets["flux_cubes.enabled"].isEnabled()
    assert "passive" in card.flux_note.text(), card.flux_note.text()


def test_the_card_estimates_the_size_of_what_the_run_will_write():
    manager, card = _card()
    manager.set_value("flux_cubes.enabled", True)
    manager.set_value("simulation.timesteps", 118)
    manager.set_value("flux_cubes.spacing_m", 25.0)
    manager.set_value("flux_cubes.radius_m", 500.0)
    card.refresh_flux_cubes()
    text = card.flux_estimate.text()
    assert "1,257 cubes" in text, text
    assert "MB" in text and "estimate" in text.lower(), text
    assert "spreadsheet" in text, text


def test_the_card_says_what_the_layout_file_contains():
    from gui_core import flux_cubes as fc
    manager, card = _card()
    manager.set_value("flux_cubes.enabled", True)
    card.refresh_flux_cubes()
    # With no layout file named -- the ordinary case -- the line says what the
    # run will DO, and names the file it will do it to. It does not ask for a
    # step the flow no longer has.
    status = card.flux_status.text()
    assert "built for you when the run starts" in status or \
           "already on disk" in status, status
    assert ".fcg" in status, status
    with tempfile.TemporaryDirectory() as tmp:
        cache = _build_cache(tmp, ("--spacing", "100", "--radius", "300"))
        card.widgets["flux_cubes.cache_file"].setText(str(cache))
        status = card.flux_status.text()
        assert "cubes" in status and "12,800-face mesh" in status, status
        # ...and the height now comes from the file rather than from the
        # window's own estimate of the same mesh.
        assert "measured by the layout tool" in card.flux_height.text(), (
            card.flux_height.text())
        header = fc.read_cache_header(cache)
        assert f"{header.n_cubes:,} cubes" in card.flux_estimate.text()


def test_a_name_the_layout_does_not_have_is_reported_by_the_card():
    manager, card = _card()
    manager.set_value("flux_cubes.enabled", True)
    manager.set_value("flux_cubes.mode", "named locations")
    manager.set_value("flux_cubes.locations",
                      [{"name": "pad_missing", "lat_deg": 80.0,
                        "lon_deg": 0.0}])
    with tempfile.TemporaryDirectory() as tmp:
        cache = _build_cache(tmp, ("--spacing", "100", "--radius", "300"))
        card.widgets["flux_cubes.cache_file"].setText(str(cache))
        card.refresh_flux_cubes()
        assert "pad_missing" in card.flux_status.text(), card.flux_status.text()
        assert "rebuild the layout" in card.flux_status.text().lower()


def test_the_locations_table_mirrors_the_configuration():
    manager, card = _card()
    manager.set_value("flux_cubes.enabled", True)
    manager.set_value("flux_cubes.mode", "named locations")
    manager.set_value("flux_cubes.locations", [
        {"name": "pad_alpha", "lat_deg": 80.002, "lon_deg": 0.0},
        {"name": "pad_bravo", "lat_deg": 79.998, "lon_deg": 0.05,
         "height_m": 2.0},
    ])
    card.load_flux_locations()
    assert card.flux_table.rowCount() == 2
    assert card.flux_table.item(0, 0).text() == "pad_alpha"
    assert card.flux_table.item(1, 3).text() == "2"
    # Editing a cell writes the list back.
    from PyQt6.QtWidgets import QTableWidgetItem
    card.flux_table.setItem(0, 0, QTableWidgetItem("pad_renamed"))
    names = [row["name"]
             for row in manager.get_value("flux_cubes.locations", [])]
    assert names == ["pad_renamed", "pad_bravo"], names


# ==========================================================================
# 6. Refusals on the way in
# ==========================================================================

def test_a_point_list_refuses_rather_than_dropping_a_line():
    """A location silently dropped on import is a cube the user believes is in
    the product and is not."""
    from gui_core import flux_cubes as fc
    rows, errors = fc.parse_locations_csv(
        "name,lat_deg,lon_deg\n"
        "# a comment\n"
        "\n"
        "pad_alpha,80.0,0.0\n"
        "pad_bravo,79.9,0.5,2.0,1.0\n")
    assert not errors, errors
    assert [r["name"] for r in rows] == ["pad_alpha", "pad_bravo"]
    assert rows[1]["height_m"] == 2.0 and rows[1]["edge_m"] == 1.0

    bad = [
        ("pad_alpha,80.0\n", "at least name"),
        ("pad_alpha,north,0.0\n", "must both be numbers"),
        ("pad_alpha,120.0,0.0\n", "outside -90 to 90"),
        (",80.0,0.0\n", "name is empty"),
        ("pad_alpha,80.0,0.0\npad_alpha,79.0,0.0\n", "named twice"),
        ("x" * 40 + ",80.0,0.0\n", "characters"),
        ("pad_alpha,80.0,0.0,tall\n", "not a number"),
        ("pad_alpha,80.0,0.0,1,2,3\n", "found 6 field(s)"),
        ('"pad,alpha",80.0,0.0\n', "comma or a quote"),
    ]
    for text, expect in bad:
        rows, errors = fc.parse_locations_csv(text)
        assert errors, f"{text!r} was accepted"
        assert any(expect in e for e in errors), (
            f"{text!r}: refusal does not say why -- {errors}")


def test_the_header_rule_is_the_layout_tools_header_rule():
    """The two parsers must agree about which rows are data.

    `flux_cube_geom.cpp` treats a row whose FIRST field is the literal `name`
    as a header, wherever it sits in the file, and reads anything else as data.
    A window that also swallowed `site` or `label` -- an earlier draft here did
    -- would show a different point list than the layout the tool builds from
    the same file, and the run verifies one against the other.
    """
    from gui_core import flux_cubes as fc
    rows, errors = fc.parse_locations_csv(
        "name,lat_deg,lon_deg\npad_alpha,80.0,0.0\nname,lat_deg,lon_deg\n"
        "pad_bravo,79.0,1.0\n")
    assert not errors and [r["name"] for r in rows] == ["pad_alpha",
                                                        "pad_bravo"]
    # `site` and `label` are DATA to the tool, so they are data here.
    rows, errors = fc.parse_locations_csv("site,80.0,0.0\nlabel,79.0,1.0\n")
    assert not errors, errors
    assert [r["name"] for r in rows] == ["site", "label"]
    # ...and the card's table has no header row, so nothing there is skipped.
    rows, errors = fc.parse_locations_csv("name,80.0,0.0\n",
                                          allow_header=False)
    assert not errors and [r["name"] for r in rows] == ["name"]


def test_the_spreadsheet_size_constants_are_measured():
    """Two real per-cube files at different step counts solve for both.

    16,807 bytes at 24 steps and 76,403 at 118 give 634.0 bytes a row and a
    1,591-byte header. A first draft guessed 520 and 1,400 and would have
    under-reported a full-grid year by 85 MB, in the one line whose purpose is
    a disk-space warning.
    """
    from gui_core import flux_cubes as fc
    assert fc.csv_bytes(1, 24) == 16807, fc.csv_bytes(1, 24)
    assert fc.csv_bytes(1, 118) == 76403, fc.csv_bytes(1, 118)


def test_a_layout_command_is_a_list_and_says_which_lattice():
    from gui_core import flux_cubes as fc
    grid = fc.geom_command("/bin/tool", "core.ply", "out.fcg",
                           mode=fc.MODE_LATTICE, spacing_m=25.0,
                           radius_m=500.0, occluder_mesh="ring.ply")
    assert isinstance(grid, list) and all(isinstance(a, str) for a in grid)
    assert "--ring" in grid and grid[grid.index("--radius") + 1] == "500.0"
    named = fc.geom_command("/bin/tool", "core.ply", "out.fcg",
                            mode=fc.MODE_LOCATIONS, locations_csv="pads.csv")
    # -1 disables the lattice; 0 would still place the single origin cube, and
    # those are different products.
    assert named[named.index("--radius") + 1] == "-1"
    assert "--spacing" not in named
    assert named[named.index("--locations") + 1] == "pads.csv"
    both = fc.geom_command("/bin/tool", "core.ply", "out.fcg",
                           mode=fc.MODE_BOTH, spacing_m=25.0, radius_m=500.0,
                           locations_csv="pads.csv")
    assert both[both.index("--radius") + 1] == "500.0"
    assert both[both.index("--spacing") + 1] == "25.0"
    assert both[both.index("--locations") + 1] == "pads.csv", (
        "the combined mode did not ask the tool for the named points")


# ==========================================================================
# 7. Pre-flight
# ==========================================================================

def _preflight_report(config, tmp):
    from gui_core import preflight
    from ptm import paths
    exe = paths.find_engine_executable("ptm_thermal_sim",
                                       search_roots=[REPO_ROOT])
    if exe is None:
        raise Skip("no ptm_thermal_sim in build/bin")
    return preflight.run_preflight(config, lambda: exe, REPO_ROOT)


def _launchable_config(tmp):
    """A runnable config whose mesh is a COPY in `tmp`.

    The copy is not tidiness. The layout now lands beside the mesh, so a
    configuration pointed at the checkout's own `data/meshes/` would have the
    tests writing `.fcg` files into the repository.
    """
    manager, _panel_ = _panel()
    manager.reset_to_defaults()
    mesh = Path(tmp) / CRATER.name
    if not mesh.exists():
        shutil.copy(CRATER, mesh)
    manager.set_value("mesh.import_file", str(mesh))
    manager.set_value("radiation.enabled", True)
    manager.set_value("output.directory", str(Path(tmp) / "out"))
    manager.set_value("flux_cubes.enabled", True)
    return manager


def test_preflight_prepares_a_layout_instead_of_blocking():
    """A missing layout is a job to do, not a refusal.

    The placement is the specification. A run with the cubes on and no layout
    file used to be blocked -- "Build it from the card" -- which made a file
    name a step in the flow. It now comes back with a PLAN: where the layout
    belongs, how many cubes it holds, and the command that writes it. What is
    still a refusal is a placement that cannot be built AT ALL.
    """
    from gui_core import flux_cubes as fc
    with tempfile.TemporaryDirectory() as tmp:
        manager = _launchable_config(tmp)
        report = _preflight_report(manager.get_config(), tmp)
        flux_errors = [e for e in report.errors if e.startswith("flux cubes")]
        assert not flux_errors, (
            f"a run whose layout can be built was blocked: {flux_errors}")
        plan = report.flux_cube_plan
        assert plan is not None and plan.action == fc.PLAN_BUILD, plan
        assert plan.n_cubes == fc.lattice_cube_count(
            fc.DEFAULT_SPACING_M, fc.DEFAULT_RADIUS_M), plan.n_cubes
        assert Path(plan.path).parent == Path(tmp), (
            f"the layout was planned for {plan.path}, not beside the mesh")
        assert plan.argv and Path(plan.argv[0]).name == fc.GEOM_EXECUTABLE
        assert report.flux_cube_line, "no size line before the build"

        # A layout file that IS named and is not there stays an error: the
        # engine will not build one during a run, and this configuration asked
        # for that exact file rather than for a placement.
        manager.set_value("flux_cubes.cache_file", str(Path(tmp) / "gone.fcg"))
        report = _preflight_report(manager.get_config(), tmp)
        assert any("was not found" in e for e in report.errors), report.errors

        manager.set_value("radiation.enabled", False)
        report = _preflight_report(manager.get_config(), tmp)
        assert any("infrared exchange is off" in e for e in report.errors), (
            report.errors)


def test_a_placement_that_cannot_be_built_is_still_refused_before_the_run():
    """The refusals that survive, each decidable before the solve."""
    from gui_core import flux_cubes as fc
    from gui_core.preflight import _ply_face_count, resolve_data_path
    with tempfile.TemporaryDirectory() as tmp:
        manager = _launchable_config(tmp)
        exe = _geom_exe()
        base = manager.get_config()

        # A mesh that is not on disk. The layout is built AGAINST it, so this
        # is decidable here rather than by the tool half a minute in.
        gone = dict(base, mesh={"import_file": str(Path(tmp) / "nope.ply")})
        plan = fc.layout_plan(gone, tmp, resolve_data_path, _ply_face_count,
                              executable=exe)
        assert plan.action == fc.PLAN_REFUSE and "not on disk" in plan.reason

        # No geometry tool. Names the target to build, rather than failing at
        # the moment of launch with a FileNotFoundError.
        plan = fc.layout_plan(base, tmp, resolve_data_path, _ply_face_count,
                              executable=None)
        assert plan.action == fc.PLAN_REFUSE
        assert fc.GEOM_EXECUTABLE in plan.reason and "cmake" in plan.reason

        # A placement with nothing in it: the named-locations mode with an
        # empty table. The tool disables the lattice (`--radius -1`) and is
        # then asked for no points at all, so the run would produce a product
        # with no cubes in it. Refused with what to do about it.
        empty = dict(base, flux_cubes=dict(base["flux_cubes"],
                                           mode=fc.MODE_LOCATIONS,
                                           locations=[]))
        plan = fc.layout_plan(empty, tmp, resolve_data_path, _ply_face_count,
                              executable=exe)
        assert plan.action == fc.PLAN_REFUSE and "no cubes" in plan.reason, (
            f"{plan.action}: {plan.reason}")

        # ...but a grid so tight it holds only the origin cube is a real, if
        # unusual, request. The count is the tool's own rule, not an estimate
        # of the disk's area, so this is one cube and not zero.
        single = dict(base, flux_cubes=dict(base["flux_cubes"],
                                            spacing_m=100.0, radius_m=1.0))
        plan = fc.layout_plan(single, tmp, resolve_data_path, _ply_face_count,
                              executable=exe)
        assert plan.action == fc.PLAN_BUILD and plan.n_cubes == 1, (
            f"{plan.action}: {plan.n_cubes} cubes")


def test_preflight_reports_the_product_size_for_a_launchable_run():
    with tempfile.TemporaryDirectory() as tmp:
        cache = _build_cache(tmp, ("--spacing", "100", "--radius", "300"))
        manager = _launchable_config(tmp)
        manager.set_value("flux_cubes.cache_file", str(cache))
        manager.set_value("simulation.timesteps", 118)
        report = _preflight_report(manager.get_config(), tmp)
        flux_errors = [e for e in report.errors if e.startswith("flux cubes")]
        assert not flux_errors, flux_errors
        assert report.flux_cube_line, "no size line was produced"
        assert "6 faces x 6 components" in report.flux_cube_line
        assert any("passive" in i for i in report.info)
        assert any(line.startswith("FLUX-CUBES:")
                   for line in report.summary_lines())


def test_preflight_says_nothing_about_a_layout_a_run_never_opens():
    """The live configuration keeps a layout path after the cubes are switched
    off. Checking it would block a launch over a file this run never reads."""
    with tempfile.TemporaryDirectory() as tmp:
        manager = _launchable_config(tmp)
        manager.set_value("flux_cubes.cache_file", str(Path(tmp) / "gone.fcg"))
        manager.set_value("flux_cubes.enabled", False)
        report = _preflight_report(manager.get_config(), tmp)
        assert not any("gone.fcg" in e for e in report.errors), report.errors
        assert not report.flux_cube_line


# ==========================================================================
# 8. Nothing here can abort the window
# ==========================================================================

def test_no_spelling_of_a_flux_cube_key_aborts_the_card():
    """The crash sweep, aimed at this group.

    An exception escaping a Qt slot is a process abort, not a message. Every
    control in this group is refreshed from the configuration on every edit, so
    a value `yaml.safe_load` hands back as a string or None has to survive the
    whole refresh path -- including the two that read a FILE.
    """
    manager, card = _card()
    spellings = ["-1e0", "5e3", "1e5", "20", None, float("nan"),
                 float("inf"), "true", "no", [], {}, "", "/nonexistent.fcg"]
    keys = ["flux_cubes.enabled", "flux_cubes.mode", "flux_cubes.cache_file",
            "flux_cubes.spacing_m", "flux_cubes.radius_m",
            "flux_cubes.write_binary", "flux_cubes.write_csv",
            "flux_cubes.write_csv_max_cubes", "flux_cubes.shadow_probe_rays",
            "flux_cubes.locations", "simulation.timesteps",
            "mesh.import_file", "radiation.enabled",
            "radiation.band_separated", "thermal.checkpoint_lookahead_steps"]
    checked = 0
    for key in keys:
        original = manager.get_value(key, None)
        for value in spellings:
            manager.set_value(key, value)
            # Every path a keystroke can reach.
            card.refresh_flux_cubes()
            card.load_flux_locations()
            manager._config_for_yaml()
            checked += 1
        manager.set_value(key, original)
    assert checked == len(keys) * len(spellings)
    print(f"    [crash sweep] {checked} key x spelling combinations, "
          f"0 aborts")


def test_every_control_in_the_group_explains_itself():
    """Coverage for this group specifically, sourced from gui_tooltips.

    The panel-wide gate already requires a tooltip on every control; this one
    names the group, so a control added here without help fails with the group
    in the message rather than in a list of every control in the application.
    """
    from gui_tooltips import TOOLTIPS
    manager, card = _card()
    bare = []
    for key, widget in card.widgets.items():
        if not key.startswith("flux_cubes."):
            continue
        tip = widget.toolTip()
        if not tip:
            bare.append(f"{key}: no tooltip")
        elif key in TOOLTIPS and tip != TOOLTIPS[key]:
            bare.append(f"{key}: shows text that is not TOOLTIPS[{key!r}]")
        elif len(tip) < 80:
            bare.append(f"{key}: {len(tip)} characters cannot answer what it "
                        f"does, what to use and when to change it")
    for name in ("flux_table", "flux_build", "flux_height", "flux_status"):
        widget = getattr(card, name, None)
        if widget is None:
            bare.append(f"{name}: the card does not have it")
        elif not widget.toolTip():
            bare.append(f"{name}: no tooltip")
    assert not bare, "\n  ".join(["flux-cube controls without help:"] + bare)


# ==========================================================================
# 7. The layout builds itself: the key, the reuse, and the stale control
# ==========================================================================

def _geom_exe():
    from ptm import paths
    from gui_core import flux_cubes as fc
    exe = paths.find_engine_executable(fc.GEOM_EXECUTABLE,
                                       search_roots=[REPO_ROOT])
    if exe is None:
        raise Skip(f"no {fc.GEOM_EXECUTABLE} in build/bin; build it with: "
                   f"cmake --build build --target {fc.GEOM_EXECUTABLE}")
    return exe


def _request(**over):
    from gui_core import flux_cubes as fc
    base = dict(mesh="/m/site.ply", occluder="", mode=fc.MODE_LATTICE,
                spacing_m=25.0, radius_m=500.0, n_faces_core=12800,
                locations=())
    base.update(over)
    return fc.LayoutRequest(**base)


def test_the_layout_key_moves_with_every_input_that_changes_the_layout():
    """One digest per specification, and a different one for each difference.

    This is the guard that keeps two placements off one path. Everything below
    changes what the tool would write, so every one of them has to change the
    key; anything that does NOT change the file must not change it, or a run
    rebuilds a layout it already has.
    """
    from gui_core import flux_cubes as fc
    base = _request()
    key = fc.request_key(base)
    assert len(key) == fc.LAYOUT_KEY_CHARS and key.isalnum()

    differ = {
        "spacing": _request(spacing_m=12.5),
        "radius": _request(radius_m=250.0),
        "mesh": _request(mesh="/m/other.ply"),
        "face count": _request(n_faces_core=12801),
        "occluder ring": _request(occluder="/m/ring.ply"),
        "mode": _request(mode=fc.MODE_LOCATIONS),
        "height factor": _request(height_factor=0.2),
        "edge factor": _request(edge_factor=0.2),
        "a named point": _request(mode=fc.MODE_BOTH,
                                  locations=(("pad", 80.0, 0.0, -1.0, -1.0),)),
    }
    for what, other in differ.items():
        assert fc.request_key(other) != key, (
            f"changing the {what} left the layout key unchanged, so the old "
            f"layout would be found on the new request's own path")

    # A named point that MOVES is a different layout, not the same one.
    a = _request(mode=fc.MODE_BOTH,
                 locations=(("pad", 80.0, 0.0, -1.0, -1.0),))
    b = _request(mode=fc.MODE_BOTH,
                 locations=(("pad", 80.5, 0.0, -1.0, -1.0),))
    assert fc.request_key(a) != fc.request_key(b)

    # ...and the same request spelled two ways is ONE key: 25 and 25.0 are the
    # same number to the tool, and a rebuild per keystroke is not a feature.
    assert fc.request_key(_request(spacing_m=25)) == key
    assert fc.request_key(_request(mesh="/m/./site.ply")) == key

    # In the named-locations mode the grid numbers are not sent to the tool at
    # all (`--radius -1`), so they must not split one layout into two.
    named = _request(mode=fc.MODE_LOCATIONS,
                     locations=(("pad", 80.0, 0.0, -1.0, -1.0),))
    assert fc.request_key(named) == fc.request_key(
        _request(mode=fc.MODE_LOCATIONS, spacing_m=7.0, radius_m=99.0,
                 locations=(("pad", 80.0, 0.0, -1.0, -1.0),)))


def test_a_changed_placement_never_reuses_the_old_layout():
    """THE STALE-LAYOUT NEGATIVE CONTROL, against a real built layout.

    Build a 50 m grid, then ask for a 25 m one. The 50 m file is still on disk
    and must not be picked up -- checked on BOTH guards independently, because
    either alone is thin: the request key keeps them on two paths, and the
    header check would refuse the old one even if something put it on the new
    path (a copy, a rename, a hand-edit).

    A layout silently reused across a changed spacing is not a crash. It is a
    delivered product whose cubes are somewhere else than the settings say,
    and nothing downstream would ever say so -- the engine reads the spacing
    OUT of the cache and has nothing to compare it with.
    """
    from gui_core import flux_cubes as fc
    from gui_core.preflight import _ply_face_count, resolve_data_path
    exe = _geom_exe()
    with tempfile.TemporaryDirectory() as tmp:
        mesh = Path(tmp) / CRATER.name
        shutil.copy(CRATER, mesh)
        cfg = {"mesh": {"import_file": str(mesh)},
               "radiation": {"enabled": True, "band_separated": True},
               "simulation": {"timesteps": 24},
               "flux_cubes": {"enabled": True, "mode": "grid",
                              "spacing_m": 50.0, "radius_m": 150.0}}

        first = fc.layout_plan(cfg, tmp, resolve_data_path, _ply_face_count,
                               executable=exe)
        assert first.action == fc.PLAN_BUILD, first.reason
        ok, message = _build_through_the_worker(first, tmp)
        assert ok, message
        assert Path(first.path).exists()

        # Same placement -> reused, not rebuilt.
        again = fc.layout_plan(cfg, tmp, resolve_data_path, _ply_face_count,
                               executable=exe)
        assert again.action == fc.PLAN_REUSE, again.reason
        assert again.path == first.path

        # Changed placement -> a different file, and the old one refused.
        cfg["flux_cubes"]["spacing_m"] = 25.0
        changed = fc.layout_plan(cfg, tmp, resolve_data_path, _ply_face_count,
                                 executable=exe)
        assert changed.action == fc.PLAN_BUILD, changed.reason
        assert changed.path != first.path, (
            f"a 25 m grid was planned onto the 50 m grid's own file "
            f"({changed.path}); the old layout would be found and reused")
        old_header = fc.read_cache_header(first.path)
        satisfied, why = fc.cache_satisfies(old_header, changed.request)
        assert not satisfied, (
            f"the 50 m layout was accepted for a 25 m request: {why}")
        assert "50" in why and "25" in why, why
        # The old layout is left alone, not overwritten: it still answers its
        # own request, and another project may be using it.
        assert Path(first.path).exists()
        assert fc.cache_satisfies(old_header, first.request)[0]


def test_a_layout_built_for_another_mesh_is_refused_on_its_header():
    """The second guard on its own, with the file name taken out of it."""
    from gui_core import flux_cubes as fc
    exe = _geom_exe()
    with tempfile.TemporaryDirectory() as tmp:
        cache = _build_cache(tmp, ("--spacing", "100", "--radius", "300"))
        header = fc.read_cache_header(cache)
        assert header.ok, header.error
        request = _request(spacing_m=100.0, radius_m=300.0,
                           n_faces_core=header.n_faces_core)
        assert fc.cache_satisfies(header, request)[0]
        # The A3GT trap: a core mesh regenerated under its own name. Same path,
        # different face count -- and this is caught before the solve rather
        # than by the engine's geometry hash after it.
        stale = _request(spacing_m=100.0, radius_m=300.0,
                         n_faces_core=header.n_faces_core + 1)
        ok, why = fc.cache_satisfies(header, stale)
        assert not ok and "face mesh" in why, why
        assert fc.request_key(stale) != fc.request_key(request)
        assert exe  # the arm needs a real layout, so the tool is required


def _build_through_the_worker(plan, cwd):
    """Run `plan` through the SHIPPED worker, not through subprocess.run.

    The worker is what the launch path uses, and it does three things the raw
    call does not: it streams the tool's output as progress, it removes a
    partial layout on failure, and it refuses a zero exit code that left no
    file. An arm that called the tool directly would test none of them.
    """
    from PyQt6.QtCore import QEventLoop
    _panel()                                   # a QApplication has to exist
    from gui_core.flux_cube_build import LayoutBuildWorker
    loop, outcome, seen = QEventLoop(), {}, []
    worker = LayoutBuildWorker(plan, str(cwd))
    worker.progress.connect(seen.append)
    worker.done.connect(
        lambda ok, msg: (outcome.update(ok=ok, msg=msg, lines=len(seen)),
                         loop.quit()))
    worker.start()
    loop.exec()
    worker.wait(5000)
    return bool(outcome.get("ok")), str(outcome.get("msg", ""))


def test_pauls_sequence_reaches_the_engine_with_a_layout():
    """enable -> grid, 25 m -> Run. No file step anywhere in it.

    The reported flow, end to end: tick the switch, leave the placement on the
    grid at its 25 m default, and launch. What this asserts is that the run
    YAML the engine reads names a layout that EXISTS and answers that
    placement -- which is the only thing the engine will accept.
    """
    from gui_core import flux_cubes as fc
    exe = _geom_exe()
    with tempfile.TemporaryDirectory() as tmp:
        manager = _launchable_config(tmp)
        _manager, card = _panel()[0], None
        manager.set_value("flux_cubes.spacing_m", 25.0)
        # A radius the shipped crater can hold, so the arm is seconds. The
        # 500 m default is the same code path with more cubes in it.
        manager.set_value("flux_cubes.radius_m", 150.0)
        manager.set_value("simulation.timesteps", 24)

        report = _preflight_report(manager.get_config(), tmp)
        assert not [e for e in report.errors if "flux" in e], report.errors
        plan = report.flux_cube_plan
        assert plan is not None and plan.action == fc.PLAN_BUILD

        ok, message = _build_through_the_worker(plan, tmp)
        assert ok, f"the layout was not built: {message}"

        # This is what `_prepare_flux_cube_layout` does next, and it is the
        # step that puts the layout into the run's record. PROVENANCE: the
        # engine-facing YAML has to name the layout that ran, or the file that
        # documents the run documents a different one.
        manager.set_value("flux_cubes.cache_file", plan.path)
        import yaml as _yaml
        run_yaml = Path(tmp) / "out" / "config.yaml"
        run_yaml.parent.mkdir(parents=True, exist_ok=True)
        manager.save_run_yaml(str(run_yaml), Path(tmp))
        written = _yaml.safe_load(run_yaml.read_text())
        assert written["flux_cubes"]["cache_file"] == plan.path, (
            f"the run YAML names {written['flux_cubes'].get('cache_file')!r} "
            f"and the run used {plan.path!r}")

        # ...and the engine's own preconditions are all met now.
        final = _preflight_report(manager.get_config(), tmp)
        assert not [e for e in final.errors if "flux" in e], final.errors
        header = fc.read_cache_header(plan.path)
        assert header.ok and header.n_cubes == plan.n_cubes, header.describe()
        assert header.n_faces_core == 12800, header.n_faces_core
        assert exe


def test_a_build_that_fails_leaves_no_layout_behind():
    """A partial layout on the request's own path is the next run's input.

    The tool writes as it goes, so a build that dies leaves a truncated `.fcg`
    exactly where the next plan will look. `cache_satisfies` would refuse it on
    its length, but relying on a reader to catch a writer's mess is one check
    away from a silent reuse -- the worker removes it instead.
    """
    from gui_core import flux_cubes as fc
    exe = _geom_exe()
    with tempfile.TemporaryDirectory() as tmp:
        mesh = Path(tmp) / CRATER.name
        shutil.copy(CRATER, mesh)
        out = Path(tmp) / "doomed.fcg"
        # A radius far larger than the mesh: the tool REFUSES a lattice point
        # with no terrain under it rather than dropping cubes, which is a real
        # non-zero exit from the real tool rather than a simulated one.
        plan = fc.LayoutPlan(
            action=fc.PLAN_BUILD, path=str(out),
            request=_request(mesh=str(mesh), radius_m=50000.0),
            argv=fc.geom_command(exe, mesh, out, spacing_m=25.0,
                                 radius_m=50000.0))
        ok, message = _build_through_the_worker(plan, tmp)
        assert not ok, "a grid that runs off the mesh was reported as built"
        assert message.strip(), "the refusal came back with no message"
        assert not out.exists(), (
            f"{out} was left on disk after a failed build; the next plan would "
            f"find it on this request's own path")


def test_named_locations_build_themselves_too():
    """The table IS the specification -- no file, and no CSV to hand-write."""
    from gui_core import flux_cubes as fc
    from gui_core.preflight import _ply_face_count, resolve_data_path
    exe = _geom_exe()
    with tempfile.TemporaryDirectory() as tmp:
        mesh = Path(tmp) / CRATER.name
        shutil.copy(CRATER, mesh)
        cfg = {"mesh": {"import_file": str(mesh)},
               "radiation": {"enabled": True, "band_separated": True},
               "simulation": {"timesteps": 24},
               "flux_cubes": {"enabled": True, "mode": fc.MODE_LOCATIONS,
                              "spacing_m": 25.0, "radius_m": 500.0,
                              "locations": [
                                  {"name": "pad_alpha", "lat_deg": 80.0,
                                   "lon_deg": 0.0}]}}
        plan = fc.layout_plan(cfg, tmp, resolve_data_path, _ply_face_count,
                              executable=exe)
        assert plan.action == fc.PLAN_BUILD, plan.reason
        assert plan.n_cubes == 1, plan.n_cubes
        assert "--radius" in plan.argv and "-1" in plan.argv, plan.argv
        assert plan.locations_csv and "pad_alpha" in plan.locations_csv
        # The point list is written by the WORKER, beside the layout, so the
        # two travel together -- nobody types a CSV path.
        assert "--locations" in plan.argv
        ok, message = _build_through_the_worker(plan, tmp)
        assert ok, message
        header = fc.read_cache_header(plan.path)
        assert header.ok and header.names == ("pad_alpha",), header.names
        assert fc.cache_satisfies(header, plan.request)[0]
        # Add a second pad and the old layout is not reused.
        cfg["flux_cubes"]["locations"].append(
            {"name": "pad_bravo", "lat_deg": 80.01, "lon_deg": 0.0})
        second = fc.layout_plan(cfg, tmp, resolve_data_path, _ply_face_count,
                                executable=exe)
        assert second.action == fc.PLAN_BUILD and second.path != plan.path
        assert not fc.cache_satisfies(header, second.request)[0]


def test_every_launcher_carries_out_the_plan():
    """Two launchers, one step -- and a source check, because there is no other.

    `run_preflight` no longer refuses a run with the cubes on and no layout: it
    hands back a PLAN. A launcher that ignores it starts the run anyway, the
    solve completes, and the flux-cube pass refuses at the very end -- which is
    the entire cost this feature exists to avoid, arrived at from the other
    direction. There is no runtime signal for "a launcher forgot", so the
    contract is checked here, at the two places that call `run_preflight` and
    then start the engine.
    """
    launchers = {
        "the main window": REPO_ROOT / "gui" / "ptm3d_gui.py",
        "the dashboard's thermal module":
            REPO_ROOT / "gui" / "module_adapters" / "thermal_adapter.py",
    }
    for what, path in launchers.items():
        source = path.read_text()
        assert "run_preflight(" in source, (
            f"{what} no longer calls run_preflight; this arm is looking at "
            f"the wrong file")
        assert "prepare_from_report" in source, (
            f"{what} calls run_preflight and starts the engine without "
            f"carrying out report.flux_cube_plan. A run with the cubes on and "
            f"no layout would solve to the end and then refuse.")


def test_the_shared_prepare_step_names_the_layout_it_built():
    """`flux_cubes.cache_file` after the step, which is what the run YAML reads."""
    from gui_core import flux_cubes as fc
    from gui_core.flux_cube_build import prepare_from_report
    from gui_core.preflight import _ply_face_count, resolve_data_path
    exe = _geom_exe()
    _panel()                                   # a QApplication has to exist
    with tempfile.TemporaryDirectory() as tmp:
        manager = _launchable_config(tmp)
        manager.set_value("flux_cubes.radius_m", 150.0)
        manager.set_value("flux_cubes.spacing_m", 50.0)

        class _Report:
            flux_cube_plan = None
        report = _Report()
        report.flux_cube_plan = fc.layout_plan(
            manager.get_config(), tmp, resolve_data_path, _ply_face_count,
            executable=exe)
        assert report.flux_cube_plan.action == fc.PLAN_BUILD
        said = []
        assert prepare_from_report(None, report, manager, tmp,
                                   info=said.append, error=said.append)
        assert manager.get_value("flux_cubes.cache_file") == \
            report.flux_cube_plan.path, (
                "the step built a layout and did not name it; the run YAML "
                "would carry the empty field it started from")
        assert Path(report.flux_cube_plan.path).exists()
        assert said, "the step built a layout and said nothing about it"

        # ...and a second pass reuses rather than rebuilding, still naming it.
        manager.set_value("flux_cubes.cache_file", "")
        report.flux_cube_plan = fc.layout_plan(
            manager.get_config(), tmp, resolve_data_path, _ply_face_count,
            executable=exe)
        assert report.flux_cube_plan.action == fc.PLAN_REUSE
        assert prepare_from_report(None, report, manager, tmp,
                                   info=said.append, error=said.append)
        assert manager.get_value("flux_cubes.cache_file") == \
            report.flux_cube_plan.path

        # A report with no plan is not an error: most runs have no cubes.
        report.flux_cube_plan = None
        assert prepare_from_report(None, report, manager, tmp,
                                   info=said.append, error=said.append)


def test_the_card_button_builds_the_same_layout_the_launch_would():
    """The button moves the wait earlier; it must not move the file."""
    from gui_core import flux_cubes as fc
    exe = _geom_exe()
    manager, card = _card()
    with tempfile.TemporaryDirectory() as tmp:
        mesh = Path(tmp) / CRATER.name
        shutil.copy(CRATER, mesh)
        manager.set_value("mesh.import_file", str(mesh))
        manager.set_value("flux_cubes.enabled", True)
        manager.set_value("flux_cubes.spacing_m", 50.0)
        manager.set_value("flux_cubes.radius_m", 150.0)
        card.refresh_flux_cubes()
        planned = card._flux_layout_plan()
        assert planned.action == fc.PLAN_BUILD

        card._on_build_layout()
        assert manager.get_value("flux_cubes.cache_file") == planned.path, (
            f"the button wrote {manager.get_value('flux_cubes.cache_file')!r} "
            f"and the launch check would have used {planned.path!r}; the "
            f"layout's name is what keeps two placements apart")
        assert Path(planned.path).exists()
        # ...and the launch check now finds it and reuses it rather than
        # building it a second time.
        again = card._flux_layout_plan()
        assert again.action == fc.PLAN_NONE, again.reason
        manager.set_value("flux_cubes.cache_file", "")
        assert card._flux_layout_plan().action == fc.PLAN_REUSE
        assert exe


def test_the_layout_file_field_is_advanced_and_the_placement_is_not():
    """What the flow asks for, and what it merely allows.

    The placement controls are in front of the user; the layout file is behind
    a disclosure that starts closed. If this inverts, the flow is a file-naming
    exercise again.
    """
    manager, card = _card()
    manager.set_value("flux_cubes.enabled", True)
    card.refresh_flux_cubes()
    for key in ("flux_cubes.mode", "flux_cubes.spacing_m",
                "flux_cubes.radius_m"):
        assert card.widgets[key].isVisibleTo(card.flux_group), (
            f"{key} is not on screen; it is the specification")
    assert card.flux_advanced.isVisibleTo(card.flux_group), (
        "the advanced section's own header is not reachable")
    assert card.flux_advanced._collapsed, (
        "the layout-file section starts open, which puts a file name back in "
        "front of the placement")
    assert not card.widgets["flux_cubes.cache_file"].isVisibleTo(card), (
        "the layout-file field is in front of the user before it is asked for")
    card.flux_advanced.toggle()
    assert card.widgets["flux_cubes.cache_file"].isVisibleTo(card.flux_group), (
        "opening the advanced section does not reveal the layout-file field")
    assert card.flux_build.isVisibleTo(card.flux_group)
    card.flux_advanced.toggle()


# ==========================================================================
# 9. Paul's curated-site sequence, clicked rather than configured
#
# THE ARMS ABOVE ALL SET `flux_cubes.enabled` THROUGH THE CONFIGURATION AND
# THEN CALL `refresh_flux_cubes()` BY HAND. `_card()` also hands every one of
# them a mesh and radiation already on, so they only ever ran in the state
# where `engine_support` is satisfied. Two things follow, and both are why
# thirty-six green arms sat in front of Paul's 2026-08-31 report:
#
#   * `set_value` + `refresh` cannot fail on a DEAD control. A real click on a
#     disabled QCheckBox changes nothing and emits nothing -- measured on the
#     pre-fix build: `switch.click()` left it unchecked, with a 79 px group of
#     one checkbox and one gray line as the entire feedback.
#   * a card reached by `card.activate(...)` skips the preset, the curated-site
#     picker and the whole body-first flow, so "the mesh arrived from the site
#     combo" was never exercised at all.
#
# Every arm below therefore drives the REAL flow and CLICKS.
# ==========================================================================

#: The essential options, by the name Paul used for each: a control key in
#: `card.widgets`, or an attribute on the card for the blocks and the lines.
#: ONE table, so "is it on screen" and "did it fall behind the Advanced
#: collapse" cannot come to disagree about what "essential" means.
ESSENTIAL = (("placement mode", "flux_cubes.mode"),
             ("grid spacing", "flux_cubes.spacing_m"),
             ("grid radius", "flux_cubes.radius_m"),
             ("the grid spacing box", "flux_lattice"),
             ("the cube height line", "flux_height"),
             ("the layout status line", "flux_status"),
             ("the size estimate", "flux_estimate"))


def _essentials(card):
    """`[(name, widget), ...]` for every essential option."""
    return [(label, card.widgets[name] if "." in name else getattr(card, name))
            for label, name in ESSENTIAL]


def _flow_card(mode="local", run_type="time_series"):
    """The Moon card as the FLOW opens it: body -> mode -> run type.

    Not `card.activate(...)`: that skips `_apply_route_defaults`, so a card
    reached that way has neither the preset's radiation setting nor the
    landing screen's clearing rules, and the curated-site combo is being
    driven against a configuration no user ever has.
    """
    manager, panel = _panel()
    manager.reset_to_defaults()
    start = panel.tabs["start"]
    start.choose_body("Moon")
    start._choose_mode(mode)
    start.choose_run_type(run_type)
    card = start.active_card()
    if card is not panel.tabs["card_moon"]:
        raise Skip("the flow did not open the Moon card")
    return manager, card


def _select_curated(card, key="A03MP026"):
    """Pick a curated site through the COMBO, as a click does, and prove the
    selection landed.

    ONE function, because the steps have to happen together and the review
    found what happens when they do not. The curated artifacts are
    machine-local: `_on_curated_site` REFUSES a site whose files are missing,
    resets the combo and writes nothing -- so an arm that skips the on-disk
    check silently ends up testing an empty `mesh.import_file` under whatever
    name it gave itself. Checking on disk first, and then asserting the mesh
    actually landed, closes both halves. Absent files SKIP; they do not pass
    quietly and they do not fail on somebody else's laptop.
    """
    from gui_core.curated_sites import curated_site, site_status
    entry = curated_site(key, body="Moon")
    if entry is None:
        raise Skip(f"{key} is not in this build's curated-site registry")
    status = site_status(entry)
    if not status.ready:
        raise Skip(f"{key} is not on this disk: {status.summary()}")
    for position in range(card.site_combo.count()):
        if card.site_combo.itemData(position) == key:
            card.site_combo.setCurrentIndex(position)
            break
    else:
        raise Skip(f"{key} is not in this card's curated-site list")
    assert card.config_manager.get_value("mesh.import_file"), (
        f"selecting {key} wrote no terrain mesh")
    return card


def _curated_card(key="A03MP026"):
    """A flow-opened Moon local-mesh card with the curated site selected."""
    manager, card = _flow_card()
    _select_curated(card, key)
    return manager, card


def _assert_options_visible(card, where):
    switch = card.widgets["flux_cubes.enabled"]
    assert switch.isChecked(), (
        f"{where}: the switch is not on after a click -- it was a dead "
        f"control and the click was discarded in silence")
    for label, widget in _essentials(card):
        assert widget.isVisibleTo(card), (
            f"{where}: {label} is not on screen with the cubes turned on")


def test_pauls_curated_site_sequence_shows_the_options():
    """PAUL'S REPORT, 2026-08-31, driven exactly as he described it.

    Moon card -> Local mesh -> the curated site A03MP026 -> tick the box.
    Every essential option must be on screen afterwards. Before the fix this
    passed only because the site had already satisfied `engine_support` by the
    time the box was clicked; the arm below it is the one that failed.
    """
    _manager, card = _curated_card()
    card.widgets["flux_cubes.enabled"].click()
    _assert_options_visible(card, "curated site then tick")


def test_the_reverse_order_shows_the_options_too():
    """Tick FIRST, then pick the site. THIS IS THE ONE THAT FAILED.

    On a freshly opened local-mesh card there is no terrain yet, so
    `engine_support` refuses and the switch used to go insensitive. The click
    was discarded -- no setting changed, no message, no option revealed -- and
    picking the site afterwards left a live-but-unticked box above one gray
    line. That is "there still doesn't seem to be any options list for the
    flux cubes, such as grid spacing and height", verbatim.
    """
    _manager, card = _flow_card()
    switch = card.widgets["flux_cubes.enabled"]
    assert switch.isEnabled(), (
        "the flux-cube switch is dead on a freshly opened local-mesh card; a "
        "click on it is a silent no-op")
    switch.click()
    assert switch.isChecked(), "the click was swallowed"
    _assert_options_visible(card, "tick with no mesh yet")
    # The options are honest about not knowing the mesh yet, rather than absent
    assert "not known yet" in card.flux_height.text(), card.flux_height.text()
    _select_curated(card)
    _assert_options_visible(card, "tick then curated site")
    assert "not known yet" not in card.flux_height.text(), (
        "the height did not re-measure when the curated site's mesh arrived: "
        + card.flux_height.text())


def test_the_switch_is_never_a_dead_control():
    """A click always lands, in every state `engine_support` refuses.

    Each of the four blockers, one at a time, plus the state that has none.
    A refusal is a sentence and a launch error; it is never a control that
    throws the click away.
    """
    from gui_core import flux_cubes as fc
    blockers = (("radiation.enabled", False),
                ("radiation.band_separated", False),
                ("mesh.import_file", ""),
                ("thermal.checkpoint_lookahead_steps", 4))
    for key, value in blockers:
        # `_curated_card` is what keeps three of these four cells from
        # silently collapsing onto the missing-MESH blocker under another
        # name on a machine without the site's files.
        manager, card = _curated_card()
        manager.set_value(key, value)
        card.refresh_flux_cubes()
        assert not fc.engine_support(manager.get_config()).allowed, (
            f"{key}={value!r} was expected to block the pass and does not; "
            f"this arm is no longer testing what it says")
        switch = card.widgets["flux_cubes.enabled"]
        assert switch.isEnabled(), (
            f"the switch is dead while {key}={value!r}; the click a user "
            f"makes there is discarded with no message")
        switch.click()
        assert switch.isChecked(), f"the click was swallowed with {key}={value!r}"
        _assert_options_visible(card, f"{key}={value!r}")
        assert "CANNOT PRODUCE" in card.flux_note.text(), (
            f"the card does not say the run cannot produce cubes with "
            f"{key}={value!r}: {card.flux_note.text()}")


def test_the_group_says_where_the_options_are_while_it_is_off():
    """OFF, the note is the whole group. It has to name the options list.

    A checkbox above one sentence about the physics is what Paul read as "no
    options list at all"; the sentence now says there is one and what is in
    it, in the words the controls themselves use.
    """
    _manager, card = _curated_card()
    assert not card.widgets["flux_cubes.enabled"].isChecked()
    text = card.flux_note.text()
    for phrase in ("Turn the cubes on", "distance between them", "grid radius",
                   "cube height"):
        assert phrase in text, f"the off-state note does not say {phrase!r}: {text}"
    assert not card.flux_body.isVisibleTo(card), (
        "the options are on screen while the cubes are off, which is a "
        "different card from the one this arm describes")


def test_the_advanced_section_holds_only_the_layout_file_machinery():
    """Nothing essential may migrate behind the collapse.

    Checked by WALKING the section rather than by naming what should not be
    there: a control added to `flux_advanced` in a later round is caught here
    even though nobody thought to list it.
    """
    from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox,
                                 QSpinBox, QTableWidget, QWidget)
    _manager, card = _curated_card()
    card.widgets["flux_cubes.enabled"].click()
    # THE OBJECTS, not their `id()`. `findChildren` mints a fresh Python
    # wrapper for any child that has none and frees it at the end of the
    # comprehension, so an id-keyed set can collide with anything constructed
    # afterwards. Only sound today because every widget compared below already
    # existed; an arm added later would get a phantom hit.
    inside = card.flux_advanced.findChildren(QWidget)
    for label, widget in _essentials(card) + [("the named-points table",
                                               card.flux_table)]:
        assert widget not in inside, (
        "the named-locations table is inside the collapsed Advanced section")
    # And what IS in there is only the layout file and its build button.
    controls = card.flux_advanced.findChildren(
        (QCheckBox, QComboBox, QDoubleSpinBox, QSpinBox, QTableWidget))
    assert not controls, (
        "the Advanced section has grown a setting control: "
        + ", ".join(sorted(type(c).__name__ for c in controls)))
    assert card.widgets["flux_cubes.cache_file"] in inside, (
        "the layout-file field left the Advanced section")


def test_the_named_mode_table_is_visible_on_the_curated_site():
    """`table for named mode`, on the site Paul is using."""
    from gui_core import flux_cubes as fc
    manager, card = _curated_card()
    card.widgets["flux_cubes.enabled"].click()
    assert not card.flux_named.isVisibleTo(card), (
        "the named-points table is on screen in the grid mode")
    manager.set_value("flux_cubes.mode", fc.MODE_LOCATIONS)
    card.refresh_flux_cubes()
    assert card.flux_named.isVisibleTo(card), (
        "the named-points table is off screen in the named mode")
    assert card.flux_table.isVisibleTo(card)
    assert not card.flux_lattice.isVisibleTo(card), (
        "the grid boxes are on screen in the named-points-only mode")
    manager.set_value("flux_cubes.mode", fc.MODE_BOTH)
    card.refresh_flux_cubes()
    assert card.flux_named.isVisibleTo(card) \
        and card.flux_lattice.isVisibleTo(card), (
            "the combined mode does not show both placements")


def test_the_toggle_moves_the_setting_and_not_the_group():
    """The switch is re-read on a TOGGLE, not only on a mode change.

    A live switch means `flux_cubes.enabled` can move without `apply_mode`
    ever running -- impossible while the control was dead. What the toggle
    moves is the SETTING and the placement body; the group itself stays put on
    both modes, so unticking on a global run leaves the switch and the
    sentence that says the mode cannot produce cubes, rather than a blank
    where a feature used to be.

    (This arm used to end `assert not card.flux_group.isVisibleTo(card)` here.
    That was the defect, not the rule; see the round-2 arms in section 10.)
    """
    manager, card = _flow_card(mode="global")
    switch = card.widgets["flux_cubes.enabled"]
    switch.click()
    assert switch.isChecked()
    assert card.flux_group.isVisibleTo(card), (
        "the group hid itself on the global mode while its setting is on")
    switch.click()
    assert not switch.isChecked()
    assert manager.get_value("flux_cubes.enabled") is False
    assert card.flux_group.isVisibleTo(card), (
        "unticking on the global mode took the whole group off the screen")
    assert not card.flux_body.isVisibleTo(card), (
        "the placement options are still on screen with the cubes off")
    # ...and the local mode keeps it, on or off.
    _manager, card = _flow_card(mode="local")
    assert card.flux_group.isVisibleTo(card)


def test_a_blocked_card_does_not_invite_you_into_the_refusal():
    """OFF and blocked: name the blocker, do not advertise the options.

    `engine_support`'s own text ends "Turn it on above" about the RADIATION
    switch. Appending "Turn the cubes on ..." beside it put an "it" and a
    "the cubes" one clause apart pointing at two different controls, and
    invited the user into the state the launch check refuses.
    """
    manager, card = _flow_card()
    assert not manager.get_value("mesh.import_file"), (
        "this arm needs the blocked state a freshly opened card has")
    text = card.flux_note.text()
    assert "imported terrain mesh" in text, text
    assert "Turn the cubes on" not in text, (
        "the card advertises the placement options on a run that cannot "
        "produce cubes: " + text)


def test_the_card_claims_no_provenance_for_a_setting_it_cannot_know():
    """A refusal must not tell the user their FILE did what they just did.

    The message used to read "The setting has NOT been changed -- it is your
    file's", which was only ever reachable from an opened file while the
    switch was dead. It is now reachable from a click.
    """
    manager, card = _flow_card()
    card.widgets["flux_cubes.enabled"].click()
    text = card.flux_note.text()
    assert "CANNOT PRODUCE" in text, text
    assert "your file" not in text, (
        "the card blames the file for a setting the user just clicked: " + text)


def test_the_curated_mesh_reads_its_own_facet_scale():
    """The height and the cube count ON A03MP026, against the real PLY.

    ⛔ 0.35 m IS THE WRONG NUMBER for this mesh, and it is wrong in a way
    worth recording: it is `sqrt(mean face area)` = 3.4596 m, and the layout
    tool's length scale is the MEDIAN EQUIVALENT EDGE
    `sqrt(4A/sqrt(3))` (`flux_cube_geom.cpp:873`). Measured exhaustively over
    all 2,397,890 faces of the delivered PLY: median equivalent edge
    3.0453 m, median area 4.0156 m^2, mean area 11.9691 m^2 -- a distribution
    skewed 3:1, which is why the two disagree by 14%. The card's sampled
    estimate lands at 3.0441 m, 0.04% off the exhaustive answer, so the
    ANSWER IS 0.3045 m and the sampler is not the reason.
    """
    from gui_core import flux_cubes as fc
    from gui_core.facet_scale import facet_scale
    manager, card = _curated_card()
    card.widgets["flux_cubes.enabled"].click()

    scale = facet_scale(manager.get_config(), card._project_root())
    assert scale.median_edge_m is not None, scale.detail
    assert abs(scale.median_edge_m - 3.0453) < 0.02, (
        f"the median equivalent edge of the curated MP026 mesh moved: "
        f"{scale.median_edge_m} m, measured exhaustively as 3.0453 m")
    geometry = fc.auto_geometry(manager.get_config(), card._project_root())
    assert abs(geometry.height_m - 0.3045) < 0.002, (
        f"the automatic cube height on MP026 is {geometry.height_m} m; the "
        f"tool's rule gives 0.1 x 3.0453 m = 0.3045 m")
    assert "30.4 cm" in card.flux_height.text(), card.flux_height.text()

    manager.set_value("flux_cubes.spacing_m", 25.0)
    manager.set_value("flux_cubes.radius_m", 500.0)
    card.refresh_flux_cubes()
    assert fc.lattice_cube_count(25.0, 500.0) == 1257
    assert "1,257 cubes" in card.flux_estimate.text(), card.flux_estimate.text()
    assert card.flux_estimate.isVisibleTo(card), (
        "the cube-count estimate is off screen")


# ==========================================================================
# 10. Round 2: the group is not on screen at all on a global run
#
# Paul, on the MERGED fix (`fb78c9f`): "we're still not getting the options
# menu for the flux cubes." The merge was intact -- `fb78c9f^{tree}` is
# byte-identical to the branch tree -- and his own checkout was at `fb78c9f`
# with the GUI launched 16 seconds after the pull. The local-mesh curated
# sequence he reported the first time now works: the 32-cell click sweep is
# 32/32 on `fb78c9f` and 20/32 on its parent.
#
# What still shows nothing is the GLOBAL mode, and that is where his files
# are: five of the eight still-existing entries in his recent-files list
# (`com.ptm3d.PTM3DSimulator.plist`) are `moon_global_*`, and every one of them
# opens on a card with no flux-cube control and no sentence anywhere saying
# why. `flux_note` was given the explanation and then hidden with its group.
#
# Every arm below FAILS on `fb78c9f`.
# ==========================================================================

def _global_flow_card():
    """The Moon card as the flow opens it on the GLOBAL mode, GUI defaults."""
    return _flow_card(mode="global")


def test_a_global_run_still_shows_the_flux_cube_group():
    """PAUL'S SECOND REPORT, and the arm that would have caught it.

    Moon / Global / time series, nothing clicked. Measured on `fb78c9f`:
    `flux_group.isHidden()` True, switch `isVisibleTo(card)` False. There is
    no checkbox between Radiation exchange and Surface properties and no text
    naming the mode -- indistinguishable from a feature that does not exist.
    """
    _manager, card = _global_flow_card()
    assert not _hidden(card.flux_group), (
        "the Flux cubes group is hidden on a Moon global run with the cubes "
        "off -- the state five of Paul's recent files open in")
    assert card.flux_group.isVisibleTo(card), (
        "the Flux cubes group has a hidden ancestor on a global run")
    switch = card.widgets["flux_cubes.enabled"]
    assert switch.isVisibleTo(card), (
        "the enable switch is off screen on a global run, so there is nothing "
        "to click and nothing to read")
    assert not card.flux_body.isVisibleTo(card), (
        "the placement options are on screen with the cubes off; the group "
        "should be one checkbox and a sentence until it is ticked")


def test_the_global_card_says_the_mode_is_the_reason():
    """The refusal has to name the mode AND the control that changes it.

    `engine_support` cannot: it reads a configuration, and an empty
    `mesh.import_file` is "no terrain imported yet" on a local card. The card
    knows, via `BodyCard._flux_mode_note`.
    """
    _manager, card = _global_flow_card()
    text = card.flux_note.text()
    assert text.strip(), "the group is on screen on a global run saying nothing"
    assert "imported terrain mesh" in text, text
    assert "Global" in text and "Local mesh" in text, (
        "the note does not name the mode that blocks the cubes, nor the mode "
        "that allows them: " + text)
    assert "Turn the cubes on" not in text, (
        "a global run advertises the placement options it cannot deliver: "
        + text)


def test_the_local_card_is_not_told_its_run_is_global():
    """⛔ A FALSE STATEMENT ABOUT THE USER'S OWN RUN.

    The mesh blocker used to end "and a global run generates its sphere in
    memory with no file to build against". On a LOCAL-mesh card with no
    terrain imported yet -- the state every local card opens in -- that tells
    the user their run is a global one. It is not.
    """
    from gui_core import flux_cubes as fc
    manager, card = _flow_card(mode="local")
    assert not manager.get_value("mesh.import_file"), (
        "this arm needs the freshly opened local card, with no terrain yet")
    blocker = " ".join(fc.engine_support(manager.get_config()).blockers)
    assert "imported terrain mesh" in blocker, blocker
    assert "a global run generates" not in blocker, (
        "engine_support tells a local-mesh run it is global: " + blocker)
    text = card.flux_note.text()
    assert "This card is set to Global" not in text, (
        "the local card carries the global mode's sentence: " + text)


def test_the_global_switch_is_live_and_reveals_the_options():
    """Live control, not a dead one -- the round-1 rule, on the other mode.

    Clicking on a global run gives the placement options AND the refusal. The
    launch check still refuses the run; nothing unsafe is reachable, and the
    setting can be seen and cleared where it could not be seen at all.
    """
    from gui_core import flux_cubes as fc
    manager, card = _global_flow_card()
    switch = card.widgets["flux_cubes.enabled"]
    fired = []
    switch.toggled.connect(lambda v: fired.append(v))
    switch.click()
    assert fired == [True], (
        "a real click on the global card emitted nothing: the switch is dead")
    assert switch.isChecked()
    for key in ("flux_cubes.mode", "flux_cubes.spacing_m",
                "flux_cubes.radius_m"):
        assert card.widgets[key].isVisibleTo(card), (
            f"{key} is off screen after a click on the global card")
    text = card.flux_note.text()
    assert "CANNOT PRODUCE" in text and "Local mesh" in text, text
    # ...and the run is still refused, before it starts.
    errors = fc.preflight_findings(
        manager.get_config(), REPO_ROOT, lambda p, r: Path(p),
        lambda p, r: Path(p), lambda p: False, lambda p: False,
        lambda p: 0).errors
    assert any("imported terrain mesh" in e for e in errors), (
        "a global run with flux cubes on passes the launch check: "
        + repr(errors))


def test_every_moon_mode_offers_the_group_in_every_run_type():
    """The sweep, reduced to its invariant: on the Moon card the group is
    always there, on both modes and both run types, before any click."""
    for mode in ("local", "global"):
        for run_type in ("time_series", "integrated"):
            _manager, card = _flow_card(mode=mode, run_type=run_type)
            assert card.flux_group.isVisibleTo(card), (
                f"no flux-cube group on Moon / {mode} / {run_type}")
            assert card.widgets["flux_cubes.enabled"].isVisibleTo(card), (
                f"no enable switch on Moon / {mode} / {run_type}")
            assert card.flux_note.text().strip(), (
                f"the group says nothing on Moon / {mode} / {run_type}")


# ==========================================================================
# 11. Round 2b: the reveal has to reach the SCREEN
#
# Paul, on a freshly relaunched GUI whose `git log` shows the merged fix:
#
#   (1) with the curated A03MP026 site, NO flux-cube options appear before OR
#       after clicking the checkbox;
#   (2) with a manually imported mesh, the options are visible BEFORE ticking
#       the box and DISAPPEAR when it is checked.
#
# Neither is a logic bug and neither is a stale install -- his tree is
# byte-identical to `origin/main` (md5 4fb63379cc311151c0802c21ea541807), and
# the signal order is `stateChanged` (writes the configuration) BEFORE
# `toggled` (re-describes the group), so nothing reads a pre-click value.
#
# It is GEOMETRY. The card is 1,558-2,078 px of content in a viewport of
# 730-1,023, and ticking the box adds ~305 px BELOW the box. Measured on his
# tree at his own saved window width (2559 px, from
# `com.ptm3d.PTM3DSimulator.plist`), scroll at the top, real `click()`:
#
#   window   path      after the click
#   2559x1381 curated  placement, spacing, radius OUTSIDE the viewport
#   2559x1200 curated  the switch itself, and all three, OUTSIDE
#   2559x1200 manual   radius OUTSIDE
#   2559x1000 manual   the switch itself, and all three, OUTSIDE
#
# Symptom (2) is that table plus the note: while the cubes are OFF the note
# LISTS the options in prose, and the tick replaces that list with one short
# sentence and puts the real controls below the fold. Options before, none
# after.
#
# ⛔ THE ARMS ASSERT A RECTANGLE. `isVisibleTo` is True for every one of those
# controls in every row above. It is the third instrument in a row to be blind
# to this group: `isHidden` missed a hidden ancestor, `setChecked` missed a
# disabled widget, `isVisibleTo` misses the scroll position. Forty-six green
# arms stood in front of this.
#
# Every arm below FAILS on `fb78c9f`.
# ==========================================================================

#: Paul's own window, from his saved geometry: normalGeometry 2559 x 1381 on a
#: 3008 px screen. The heights bracket it, because the real main window spends
#: another 80-120 px on the menu bar, tool bar and status bar that this panel
#: does not have.
PAUL_W = 2559
PAUL_HEIGHTS = (1381, 1200, 1000)

ON_SCREEN = [("placement", "flux_cubes.mode"),
             ("spacing", "flux_cubes.spacing_m"),
             ("radius", "flux_cubes.radius_m")]


def _pump():
    """Let Qt lay out and run the zero-timers a reveal schedules."""
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance()
    for _ in range(3):
        app.processEvents()


def _scroll_of(card):
    """The card's scroll area, found the way anything can find it.

    ⛔ NOT `card._scroll`. That attribute is part of the fix, so an arm that
    reads it SKIPS on the broken build instead of failing on it -- and a skip
    is not a falsification. Everything section 11 measures is measured from Qt,
    not from the code under test.
    """
    from PyQt6.QtWidgets import QScrollArea
    areas = card.findChildren(QScrollArea)
    if not areas:
        raise Skip("this card has no scroll area to measure against")
    return areas[0]


def _on_screen(card, widget):
    """Is `widget` inside the card's visible rectangle right now?

    The question `isVisibleTo` does not answer, computed here from the
    viewport geometry so the arms cannot inherit the defect's blind spot.
    """
    from PyQt6.QtCore import QPoint
    if not widget.isVisibleTo(card):
        return False
    viewport = _scroll_of(card).viewport()
    top = widget.mapTo(viewport, QPoint(0, 0)).y()
    return (top + widget.height() > 0) and (top < viewport.height())


def _sized_card(width, height, path):
    """A Moon local-mesh card in a window of a given size, scrolled to the top.

    `path` is "curated" (the A03MP026 picker, which writes a long provenance
    note above the group) or "manual" (the shipped crater in the file field).
    """
    manager, panel = _panel()
    panel.resize(width, height)
    manager.reset_to_defaults()
    start = panel.tabs["start"]
    start.choose_body("Moon")
    start._choose_mode("local")
    start.choose_run_type("time_series")
    card = start.active_card()
    if card is not panel.tabs["card_moon"]:
        raise Skip("the flow did not open the Moon card")
    if path == "curated":
        _select_curated(card)
    else:
        if not CRATER.exists():
            raise Skip(f"{CRATER} is not in this checkout")
        manager.set_value("mesh.import_file", str(CRATER))
        manager.set_value("radiation.enabled", True)
        card.load_from_config()
    panel.show()
    _pump()
    _scroll_of(card).verticalScrollBar().setValue(0)
    _pump()
    return manager, card


def _off_screen(card, names=ON_SCREEN):
    return [label for label, key in names
            if not _on_screen(card, card.widgets[key])]


def test_the_instrument_can_fail():
    """⛔ FIRST, PROVE THE RULER IS NOT PAINTED ON.

    `_on_screen` has to report False for something `isVisibleTo` calls
    visible, or every arm below is vacuous -- which is precisely how the last
    two rounds of this defect got past a green suite. Scroll to the top of a
    card taller than its window and the bottom of the form must read as off
    screen while Qt still says it is not hidden.
    """
    _manager, card = _sized_card(PAUL_W, 1000, "curated")
    scroll = _scroll_of(card)
    if scroll.verticalScrollBar().maximum() <= 0:
        raise Skip("this card fits its window; nothing can be off screen")
    bottom = card.widgets["flux_cubes.radius_m"]
    card.widgets["flux_cubes.enabled"].click()
    _pump()
    scroll.verticalScrollBar().setValue(0)
    _pump()
    assert bottom.isVisibleTo(card), (
        "this arm needs a control Qt calls visible")
    assert _on_screen(card, bottom) is False, (
        "the viewport check says a control 300 px below the window is on "
        "screen; it cannot detect the defect it exists for")


def test_the_click_puts_the_options_on_the_screen_curated():
    """PAUL'S SYMPTOM (1): the curated site, nothing before OR after.

    The curated picker writes a provenance note that is 216 px tall at his
    window width, which puts the group at y=971 in a 1,023 px viewport: the
    switch is reachable and everything the click reveals is not.
    """
    for height in PAUL_HEIGHTS:
        _manager, card = _sized_card(PAUL_W, height, "curated")
        card.widgets["flux_cubes.enabled"].click()
        _pump()
        missing = _off_screen(card)
        assert not missing, (
            f"at {PAUL_W}x{height} on the curated site, clicking the box "
            f"leaves these off the screen: {', '.join(missing)} -- the user "
            f"ticks the box and nothing changes in front of them")


def test_the_click_puts_the_options_on_the_screen_imported_mesh():
    """PAUL'S SYMPTOM (2), second half: the reveal lands below the fold."""
    for height in PAUL_HEIGHTS:
        _manager, card = _sized_card(PAUL_W, height, "manual")
        card.widgets["flux_cubes.enabled"].click()
        _pump()
        missing = _off_screen(card)
        assert not missing, (
            f"at {PAUL_W}x{height} with an imported mesh, clicking the box "
            f"leaves these off the screen: {', '.join(missing)}")


def test_the_written_list_is_not_withdrawn_without_showing_the_real_thing():
    """PAUL'S SYMPTOM (2), stated as the mechanism it actually is.

    "The options are visible BEFORE ticking the box and DISAPPEAR when it is
    checked." Nothing literally disappears -- what is on screen before the
    tick is `FLUX_OFF_HINT`, a sentence that LISTS the options in words ("the
    distance between them, the grid radius, the cube height ..."). Ticking the
    box replaces that list with one short line and puts the real controls
    below the fold, so the card takes the written list away and puts nothing
    in its place. Read as "the options disappeared", correctly.

    ⛔ A COUNT OF VISIBLE CONTROLS DOES NOT CATCH THIS, and the first version
    of this arm was that count. Before the click the three placement controls
    are `isVisibleTo == False` and count zero; after it they are visible but
    outside the viewport and still count zero. The count never drops and the
    arm passed on the broken build. The invariant has to name the trade: the
    list may only be withdrawn in exchange for the controls it described.
    """
    for path in ("curated", "manual"):
        for height in PAUL_HEIGHTS:
            _manager, card = _sized_card(PAUL_W, height, path)
            phrase = "the distance between them"
            listed_before = phrase in card.flux_note.text()
            card.widgets["flux_cubes.enabled"].click()
            _pump()
            listed_after = phrase in card.flux_note.text()
            if not (listed_before and not listed_after):
                continue                # the list was never on offer here
            missing = _off_screen(card)
            assert not missing, (
                f"at {PAUL_W}x{height} ({path}) ticking the box withdrew the "
                f"written list of options and left these off the screen: "
                f"{', '.join(missing)} -- the card showed the user a list of "
                f"options, then neither the list nor the options")


def test_the_reveal_does_not_scroll_a_card_nobody_touched():
    """DO-NO-HARM. Passes on `fb78c9f` too, and is meant to: `fb78c9f` never
    scrolls anything. It guards the behavior this branch ADDS, not the defect
    it fixes -- do not read it as a falsifier.

    A file that arrives with the cubes on must not move the form.

    `_reveal_flux_cubes` is guarded on `_loading` for this: `set_bool` calls
    `setChecked` while a configuration is being mirrored into the controls,
    which emits `toggled` exactly as a click does. Without the guard, opening
    any project with `flux_cubes.enabled: true` would scroll the card to the
    middle before the user has looked at it.
    """
    manager, panel = _panel()
    panel.resize(PAUL_W, 1000)
    manager.reset_to_defaults()
    start = panel.tabs["start"]
    start.choose_body("Moon")
    start._choose_mode("local")
    start.choose_run_type("time_series")
    card = start.active_card()
    if not CRATER.exists():
        raise Skip(f"{CRATER} is not in this checkout")
    manager.set_value("mesh.import_file", str(CRATER))
    manager.set_value("radiation.enabled", True)
    manager.set_value("flux_cubes.enabled", True)
    panel.show()
    _pump()
    _scroll_of(card).verticalScrollBar().setValue(0)
    card.load_from_config()          # the opened-file path, `_loading` held
    _pump()
    assert _scroll_of(card).verticalScrollBar().value() == 0, (
        "opening a file with the cubes already on scrolled the card away "
        "from its top; displaying a configuration is not the user asking "
        "for anything")


# ==========================================================================
# 12. The curated-site note, collapsed (Paul's ruling, 2026-09-01)
#
# Round 2 left one contributor unfixed and asked: picking a curated site wrote
# the whole provenance record onto the card -- about thirty lines, MEASURED at
# 216 px tall at Paul's 2559 px window width -- and every group below it, the
# flux cubes among them, started 216 px further down. He ruled: one line, with
# the rest behind an expander.
#
# These arms live here, beside the fold measurements they exist to protect,
# and not in `test_gui_curated_sites.py`, because the reason for the change is
# the fold. That suite owns what the note SAYS; this one owns how much room it
# takes and what that costs the controls underneath.
#
# ⛔ ONE THING MUST NEVER GO IN THE EXPANDER: a statement that the open
# configuration DISAGREES with the curated set. That is a warning about the run
# somebody is about to launch, and a warning behind a control you have to find
# first is not a warning.
# ==========================================================================

def _site_wiring_pairs(card):
    from gui_core.curated_sites import curated_site, site_wiring
    entry = curated_site("A03MP026", body="Moon")
    if entry is None:
        raise Skip("A03MP026 is not in the curated registry")
    return site_wiring(entry)


def test_the_curated_note_is_one_line_at_pauls_window():
    """MEASURED, not eyeballed: the note was 216 px and is now one line.

    A line of `_muted` text renders at 12 px here. The bound is deliberately
    loose -- two lines is still a note, thirty is a wall -- and the number that
    matters is the one it is nowhere near.
    """
    _manager, card = _sized_card(PAUL_W, 1381, "curated")
    note = card.site_note
    assert "\n" not in note.text(), (
        "the curated-site note is still multi-line:\n" + note.text())
    assert note.height() <= 40, (
        f"the curated-site note renders {note.height()} px tall at "
        f"{PAUL_W} px wide; it was 216 px and the ruling was one line")
    assert "A03MP026" in note.text(), note.text()


def test_the_details_expander_starts_closed_and_holds_the_provenance():
    """The record is kept, and kept out of the way until it is asked for."""
    _manager, card = _sized_card(PAUL_W, 1381, "curated")
    details = getattr(card, "site_details", None)
    assert details is not None, "there is no details section on the card"
    assert details.isVisibleTo(card), (
        "a curated site is selected and its details section is not on the card")
    body = card.site_details_text
    assert not body.isVisibleTo(card), (
        "the provenance is on screen without being asked for; the section "
        "should open collapsed")
    details.toggle_button.click()
    _pump()
    assert body.isVisibleTo(card), (
        "opening the details section revealed nothing")
    text = body.text()
    assert "NOT used" in text, (
        "the expander does not carry the which-files-were-rejected record, "
        "which is the part of the provenance that prevents a wrong rerun:\n"
        + text[:400])
    assert text.count("\n") >= 8, (
        f"the expander holds {text.count(chr(10)) + 1} lines; the provenance "
        f"record is longer than that and something has been dropped")
    # ⛔ PUT IT BACK. `_panel()` hands every arm the SAME panel, and the
    # collapsed state is the user's -- the card deliberately does not reset it
    # on a re-describe, so an arm that opens the section and walks away adds
    # ~200 px to the card for every arm after it. Caught the honest way: the
    # fold arm below failed at 2559x1381, which is a real measurement of a
    # state no user is in.
    details.toggle_button.click()
    _pump()
    assert not body.isVisibleTo(card), "the details section did not close again"


def test_no_curated_site_means_no_details_section():
    """Nothing to expand, so nothing offering to expand."""
    manager, panel = _panel()
    panel.resize(PAUL_W, 1381)
    manager.reset_to_defaults()
    start = panel.tabs["start"]
    start.choose_body("Moon")
    start._choose_mode("local")
    start.choose_run_type("time_series")
    card = start.active_card()
    panel.show()
    _pump()
    assert not card.site_details.isVisibleTo(card), (
        "the details section is on the card with no curated site selected")


def test_a_disagreement_stays_on_the_line_and_is_never_collapsed():
    """⛔ THE ONE THING THE EXPANDER MUST NOT SWALLOW.

    Open a file that names this site's terrain with one of its other settings
    changed and the card says so. That sentence is about the run the user is
    about to launch, not about provenance, and it belongs where they cannot
    miss it -- the expander is closed by default, so a warning inside it is a
    warning nobody reads.
    """
    _manager, card = _sized_card(PAUL_W, 1381, "curated")
    manager = _panel()[0]
    changed = None
    for key, value in _site_wiring_pairs(card):
        if key.startswith("engine_env.") or not isinstance(value, bool):
            continue
        manager.set_value(key, not value)
        changed = key
        break
    if changed is None:
        raise Skip("this site wires no boolean this arm can disagree with")
    card._describe_curated_site()          # the DESCRIBING path, no `applied`
    _pump()
    line = card.site_note.text()
    assert "DIFFERS FROM THE CURATED SET" in line, (
        f"changing {changed} away from the curated value is not reported on "
        f"the line: " + line)
    assert changed in line, line
    body = card.site_details_text.text()
    assert "DIFFERS FROM THE CURATED SET" not in body, (
        "the disagreement warning was put inside the collapsed section, "
        "where the user has to open it to be warned")


def test_collapsing_the_note_lifts_the_flux_switch_above_the_fold():
    """The point of the ruling, measured on the control it was blocking.

    Before: the curated card's content was 1,774 px and the flux-cube switch
    sat at y=1000, outside a 883 px viewport -- the user could not reach the
    control at all at 2559x1200. After: 1,605 px of content and the switch at
    y=831, on screen at both of the window heights Paul actually uses, with no
    scrolling at all.
    """
    for height in (1381, 1200):
        _manager, card = _sized_card(PAUL_W, height, "curated")
        # Stated, not assumed: this measurement is only about the collapsed
        # card, and `_panel()` is shared. An earlier arm leaving the section
        # open must say so here rather than show up as a mystery 200 px.
        assert card.site_details._collapsed, (
            "the details section is open before this measurement; an earlier "
            "arm left it that way and this is not the state being measured")
        switch = card.widgets["flux_cubes.enabled"]
        assert _on_screen(card, switch), (
            f"at {PAUL_W}x{height} the flux-cube switch is still below the "
            f"fold on a freshly opened curated card, with nothing scrolled")


TESTS = [
    test_the_cube_count_is_the_tools_own_rule,
    test_the_fast_count_agrees_with_the_tools_double_loop_everywhere,
    test_the_product_size_matches_two_delivered_files,
    test_a_name_never_loses_its_spreadsheet_to_the_cap,
    test_the_spreadsheet_size_constants_are_measured,
    test_the_height_rule_is_the_tools_own_length_scale,
    test_a_real_layout_reads_back_as_the_tool_wrote_it,
    test_named_locations_come_back_by_name,
    test_a_truncated_layout_is_refused,
    test_a_runnable_configuration_is_allowed,
    test_every_engine_refusal_is_caught_before_the_run,
    test_the_gate_reads_a_yaml_spelling_the_engine_accepts,
    test_cubes_off_writes_no_section_at_all,
    test_the_grid_mode_emits_the_seven_engine_keys_and_nothing_else,
    test_the_named_mode_emits_its_locations,
    test_a_grid_only_run_does_not_ask_for_points_it_has_no_layout_for,
    test_the_probe_setting_is_written_as_a_number,
    test_opening_a_project_takes_the_placement_from_the_layout_file,
    test_the_engine_accepts_both_modes,
    test_the_group_is_on_the_card_in_both_modes_and_says_which,
    test_the_settings_appear_only_once_the_switch_is_on,
    test_the_mode_shows_the_controls_that_mode_uses,
    test_the_height_is_recomputed_when_the_mesh_changes,
    test_the_card_refuses_a_run_the_engine_would_refuse_and_edits_nothing,
    test_the_card_estimates_the_size_of_what_the_run_will_write,
    test_the_card_says_what_the_layout_file_contains,
    test_a_name_the_layout_does_not_have_is_reported_by_the_card,
    test_the_locations_table_mirrors_the_configuration,
    test_a_point_list_refuses_rather_than_dropping_a_line,
    test_the_header_rule_is_the_layout_tools_header_rule,
    test_a_layout_command_is_a_list_and_says_which_lattice,
    test_preflight_prepares_a_layout_instead_of_blocking,
    test_a_placement_that_cannot_be_built_is_still_refused_before_the_run,
    test_preflight_reports_the_product_size_for_a_launchable_run,
    test_preflight_says_nothing_about_a_layout_a_run_never_opens,
    test_no_spelling_of_a_flux_cube_key_aborts_the_card,
    test_every_control_in_the_group_explains_itself,
    # -- the layout builds itself -----------------------------------------
    test_a_live_setting_is_never_hidden_by_the_mode,
    test_the_layout_key_moves_with_every_input_that_changes_the_layout,
    test_a_changed_placement_never_reuses_the_old_layout,
    test_a_layout_built_for_another_mesh_is_refused_on_its_header,
    test_pauls_sequence_reaches_the_engine_with_a_layout,
    test_a_build_that_fails_leaves_no_layout_behind,
    test_named_locations_build_themselves_too,
    test_every_launcher_carries_out_the_plan,
    test_the_shared_prepare_step_names_the_layout_it_built,
    test_the_card_button_builds_the_same_layout_the_launch_would,
    test_the_layout_file_field_is_advanced_and_the_placement_is_not,
    # 9. the curated-site sequence, clicked
    test_pauls_curated_site_sequence_shows_the_options,
    test_the_reverse_order_shows_the_options_too,
    test_the_switch_is_never_a_dead_control,
    test_the_group_says_where_the_options_are_while_it_is_off,
    test_the_advanced_section_holds_only_the_layout_file_machinery,
    test_the_named_mode_table_is_visible_on_the_curated_site,
    test_the_toggle_moves_the_setting_and_not_the_group,
    test_a_blocked_card_does_not_invite_you_into_the_refusal,
    test_the_card_claims_no_provenance_for_a_setting_it_cannot_know,
    test_the_curated_mesh_reads_its_own_facet_scale,
    # 10. round 2: the global mode showed nothing at all
    test_a_global_run_still_shows_the_flux_cube_group,
    test_the_global_card_says_the_mode_is_the_reason,
    test_the_local_card_is_not_told_its_run_is_global,
    test_the_global_switch_is_live_and_reveals_the_options,
    test_every_moon_mode_offers_the_group_in_every_run_type,
    # 11. round 2b: the reveal has to reach the screen
    test_the_instrument_can_fail,
    test_the_click_puts_the_options_on_the_screen_curated,
    test_the_click_puts_the_options_on_the_screen_imported_mesh,
    test_the_written_list_is_not_withdrawn_without_showing_the_real_thing,
    test_the_reveal_does_not_scroll_a_card_nobody_touched,
    # 12. the curated-site note, collapsed (Paul's ruling)
    test_the_curated_note_is_one_line_at_pauls_window,
    test_the_details_expander_starts_closed_and_holds_the_provenance,
    test_no_curated_site_means_no_details_section,
    test_a_disagreement_stays_on_the_line_and_is_never_collapsed,
    test_collapsing_the_note_lifts_the_flux_switch_above_the_fold,
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
    print(f"\n[test_gui_flux_cubes] {total - failures - skipped}/{total} "
          f"passed, {skipped} skipped")
    if _SCRATCH:
        shutil.rmtree(_SCRATCH, ignore_errors=True)
    if skipped == total:
        print("[test_gui_flux_cubes] SKIPPED ENTIRELY -- proved NOTHING here.")
    if failures:
        print("[test_gui_flux_cubes] FAIL")
        return 1
    print("[test_gui_flux_cubes] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
