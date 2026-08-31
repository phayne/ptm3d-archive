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

def test_the_group_is_offered_on_the_local_mesh_and_not_on_the_global_one():
    manager, card = _card()
    assert not _hidden(card.flux_group), (
        "the flux-cube group is hidden on the local-mesh card, where the "
        "layout tool has a mesh to work with")
    card.activate("global", "time_series")
    assert _hidden(card.flux_group), (
        "the flux-cube group is offered on a global run, where there is no "
        "mesh file for the layout tool to build against")


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
    """
    manager, card = _card()
    manager.set_value("flux_cubes.enabled", True)
    manager.set_value("radiation.enabled", False)
    card.refresh_flux_cubes()
    assert manager.get_value("flux_cubes.enabled") is True, (
        "the card turned the cubes off by itself")
    assert not card.widgets["flux_cubes.enabled"].isEnabled(), (
        "the switch is still live on a run that cannot produce cubes")
    assert "CANNOT PRODUCE" in card.flux_note.text(), card.flux_note.text()
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
    assert "No layout file yet" in card.flux_status.text()
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
    manager, _panel_ = _panel()
    manager.reset_to_defaults()
    manager.set_value("mesh.import_file", str(CRATER))
    manager.set_value("radiation.enabled", True)
    manager.set_value("output.directory", str(Path(tmp) / "out"))
    manager.set_value("flux_cubes.enabled", True)
    return manager


def test_preflight_blocks_a_run_that_would_stop_after_the_solve():
    with tempfile.TemporaryDirectory() as tmp:
        manager = _launchable_config(tmp)
        report = _preflight_report(manager.get_config(), tmp)
        assert any("no cube layout file" in e for e in report.errors), (
            f"a run with cubes on and no layout was not blocked: "
            f"{report.errors}")

        manager.set_value("flux_cubes.cache_file", str(Path(tmp) / "gone.fcg"))
        report = _preflight_report(manager.get_config(), tmp)
        assert any("was not found" in e for e in report.errors), report.errors

        manager.set_value("radiation.enabled", False)
        report = _preflight_report(manager.get_config(), tmp)
        assert any("infrared exchange is off" in e for e in report.errors), (
            report.errors)


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
    test_the_group_is_offered_on_the_local_mesh_and_not_on_the_global_one,
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
    test_preflight_blocks_a_run_that_would_stop_after_the_solve,
    test_preflight_reports_the_product_size_for_a_launchable_run,
    test_preflight_says_nothing_about_a_layout_a_run_never_opens,
    test_no_spelling_of_a_flux_cube_key_aborts_the_card,
    test_every_control_in_the_group_explains_itself,
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
