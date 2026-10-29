#!/usr/bin/env python3
"""The card you ran from is the body that runs.

**Why this file exists.** On 2026-08-27 the Europa Global card produced a run
whose own ``provenance.txt`` recorded ``body=bennu``, ``generic_body=true`` and
``mesh_median_radius=99.99`` km -- a 100 km sphere at 1 AU on a 6 h spin, where
a 1561 km moon at 5.2 AU on a 3.55 d spin was asked for. The user had entered a
name on the Asteroid card, gone back, and run from the Europa card. The
configuration the run wrote said ``simulation.body: Europa`` on line 5 and
``generic_body: {name: bennu}`` on line 136, and the engine resolves that pair
in exactly one direction::

    ConfigParser::GenericBodyParams::isConfigured()      config_parser.h:717
        return !config_file.empty() || !name.empty()
               || hasOrbitalElements() || hasSpinState();

    if (sim_config.generic_body.isConfigured()) {        ptm_thermal_sim.cpp:14626
        planet = thermal::Planet(generic_body.name,      // simulation.body is
                                 generic_body.thermo);   // NEVER read
    }

``target_body_name`` is assigned only on the built-in branch (``:14127``), so
the albedo-map loader at ``:7660`` also reported "No body name in config" and
fell back to a uniform 0.1 -- on a card whose whole point is Europa's albedo
map. One stale string, three wrong things.

**Why the existing acceptance test did not catch it.**
``test_gui_body_first.test_every_card_emits_a_configuration_the_engine_accepts``
hands each card's configuration to ``ptm_thermal_sim --validate-config``. That
check answers "is this a configuration the engine can parse and run" -- it is a
SYNTAX and reachability check, and it has no opinion about WHICH BODY the
configuration describes. It passed the leaked configuration, and it would pass
it again: the leaked configuration is a perfectly valid asteroid run. It also
walks the routes without resetting between them, so on that test's own
ordering the asteroid route's ``generic_body`` reached the ``custom`` route and
Mercury's ``simulation.body`` reached the asteroid route, and every arm was
green. Identity is a separate property and needs a separate check, which is
this file.

**What this file asserts, and on what.** On the FILE ``save_run_yaml`` writes
-- the ``output/<dir>/config.yaml`` the engine is actually handed -- never on
an intermediate dictionary. The dictionary is not what runs.

Deliberately engine-free: it reads YAML and compares strings, so it holds on
the 2026-08-25 binary that has no ``--validate-config`` at all, and on any
older one. A guard that only works on a new engine is no guard for the user
running last month's build.

Run directly::

    python3 src/python/tests/test_gui_card_body_identity.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "python"))
sys.path.insert(0, str(REPO_ROOT / "gui"))


class Skip(Exception):
    """A check that cannot run here, as distinct from one that failed."""


_APP = None
_PANEL = None
_MANAGER = None


def _panel(scratch):
    """One offscreen ConfigurationPanel, reused across the checks."""
    global _APP, _PANEL, _MANAGER
    if _PANEL is not None:
        return _MANAGER, _PANEL
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        import PyQt6
    except ImportError as exc:                              # pragma: no cover
        raise Skip(f"PyQt6 is unavailable here ({exc})")
    plug = Path(PyQt6.__file__).parent / "Qt6" / "plugins"
    os.environ.setdefault("QT_PLUGIN_PATH", str(plug))
    os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(plug / "platforms"))
    from PyQt6.QtCore import QSettings
    from PyQt6.QtWidgets import QApplication
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    for scope in (QSettings.Scope.UserScope, QSettings.Scope.SystemScope):
        QSettings.setPath(QSettings.Format.IniFormat, scope, str(scratch))
    try:
        from gui_core.config_manager import ConfigManager
        from gui_panels.config_panel import ConfigurationPanel
    except ImportError as exc:                              # pragma: no cover
        raise Skip(f"the GUI panel stack is unavailable here ({exc})")
    _APP = QApplication.instance() or QApplication([])
    _MANAGER = ConfigManager()
    _PANEL = ConfigurationPanel(_MANAGER)
    return _MANAGER, _PANEL


def _routes():
    from gui_core.body_cards import body_buttons, modes_for, run_types_for
    out = []
    for button in body_buttons():
        for mode in modes_for(button.key):
            for run_type in run_types_for(button.key, mode.key):
                out.append((button.key, mode.key, run_type.key))
    return out


def _builtin_names():
    from ptm.body_table import IMPLEMENTED_BODIES
    return {str(b).strip().lower() for b in IMPLEMENTED_BODIES}


def _emit(manager, panel, body, mode, run_type, scratch, tag=""):
    """Open a card and return the configuration the RUN would write.

    Goes through ``save_run_yaml`` to a real path and reads the file back,
    because that is the artifact the engine is handed. Returns
    ``(parsed dict, path)``.
    """
    start = panel.tabs["start"]
    start.choose_body(body)
    start._choose_mode(mode)
    start.open_card(body, mode, run_type, apply_defaults=True)
    stem = f"{body}_{mode}_{run_type}{tag}"
    out = Path(scratch) / stem
    out.mkdir(parents=True, exist_ok=True)
    manager.set_value("output.directory", str(out))
    path = out / "config.yaml"
    manager.save_run_yaml(str(path), REPO_ROOT)
    return yaml.safe_load(path.read_text()) or {}, path


def _poison_asteroid(manager, panel):
    """Paul's own sequence, step one: enter data on the Asteroid card.

    Uses the card's real widgets, so this exercises the same ``textChanged``
    and ``valueChanged`` handlers a keystroke does -- ``set_value`` on the
    manager would bypass exactly the code being tested.
    """
    start = panel.tabs["start"]
    start.choose_body("asteroid")
    start._choose_mode("body")
    card = start.open_card("asteroid", "body", "time_series",
                           apply_defaults=True)
    card.widgets["generic_body.name"].setText("bennu")
    card.widgets["generic_body.naif_id"].setText("2101955")
    card.widgets["generic_body.shape.radius_km"].setValue(0.2447)
    card.widgets["generic_body.surface.albedo"].setValue(0.044)
    start.stack.setCurrentIndex(start.LANDING)          # the "back" press
    return card


def _poison_moon_local(manager, panel):
    """The reverse sequence: a lunar core mesh and its placement, then back."""
    start = panel.tabs["start"]
    start.choose_body("Moon")
    start._choose_mode("local")
    start.open_card("Moon", "local", "time_series", apply_defaults=True)
    manager.set_value("mesh.import_file", "data/dem/lunar_crater_80N.ply")
    manager.set_value("mesh.occluder_file", "data/dem/ring_Rcore3_Rext20km.ply")
    manager.set_value("mesh.latitude_override", 80.0)
    manager.set_value("mesh.longitude_override", 12.5)
    manager.set_value("radiation.view_factor_file", "vf/crater80N.vf")
    start.stack.setCurrentIndex(start.LANDING)          # the "back" press


# --------------------------------------------------------------------------
# 1. Paul's sequence, and its mirror image
# --------------------------------------------------------------------------

def test_asteroid_card_residue_never_reaches_a_builtin_body_run():
    """Asteroid card -> edit -> back -> Europa Global -> Run.

    The measured sequence, asserted on the emitted file. This is the check
    that fails on the pre-fix GUI: it emitted ``generic_body: {name: bennu}``
    under ``simulation.body: Europa``.
    """
    with tempfile.TemporaryDirectory() as scratch:
        manager, panel = _panel(scratch)
        manager.reset_to_defaults()
        panel.load_from_config()
        _poison_asteroid(manager, panel)

        emitted, path = _emit(manager, panel, "Europa", "global",
                              "time_series", scratch, tag="_after_asteroid")

        body = str((emitted.get("simulation") or {}).get("body", ""))
        assert body.strip().lower() == "europa", (
            f"the Europa Global card emitted simulation.body = {body!r}")
        assert "generic_body" not in emitted, (
            f"{path} carries a generic_body: section "
            f"{emitted['generic_body']!r} under simulation.body: {body!r}. "
            f"The engine reads the generic body and never looks at "
            f"simulation.body (ptm_thermal_sim.cpp:14626), so this run would "
            f"be that body, not Europa -- which is exactly the 2026-08-27 "
            f"europa_global_res06 run that recorded body=bennu, a 99.99 km "
            f"radius and a 1 AU orbit.")
        print("    [sequence] asteroid edits -> back -> Europa Global: "
              "emitted config is pure Europa")


def test_builtin_mesh_residue_never_reaches_a_generic_body_run():
    """The mirror image: Moon Local mesh -> back -> Asteroid -> Run.

    Same engine rule from the other side. The solver takes
    ``mesh.import_file`` whenever it is non-empty and reaches the icosphere
    branch only when it is empty (``ptm_thermal_sim.cpp:15087``), so a lunar
    crater core mesh left behind here becomes the ASTEROID's shape -- at a
    lunar latitude and longitude, against a view-factor matrix built for a
    different mesh.
    """
    with tempfile.TemporaryDirectory() as scratch:
        manager, panel = _panel(scratch)
        manager.reset_to_defaults()
        panel.load_from_config()
        _poison_moon_local(manager, panel)

        emitted, path = _emit(manager, panel, "asteroid", "body",
                              "time_series", scratch, tag="_after_moon")
        mesh = emitted.get("mesh") or {}
        rad = emitted.get("radiation") or {}
        leaked = {k: v for k, v in (
            ("mesh.import_file", mesh.get("import_file")),
            ("mesh.occluder_file", mesh.get("occluder_file")),
            ("radiation.view_factor_file", rad.get("view_factor_file")),
        ) if str(v or "").strip()}
        assert not leaked, (
            f"{path} carries the Moon card's mesh: {leaked}. The engine would "
            f"use it as this body's shape (ptm_thermal_sim.cpp:15087).")
        for key in ("latitude_override", "longitude_override"):
            assert mesh.get(key) == -999.0, (
                f"{path} carries the Moon card's {key} = {mesh.get(key)!r}; "
                f"-999.0 is the engine's 'not placed' sentinel.")
        print("    [sequence] Moon Local mesh -> back -> Asteroid: "
              "emitted config carries no lunar mesh")


def test_custom_body_residue_never_reaches_a_builtin_body_run():
    """The Custom-body route leaks the same way the Asteroid route does."""
    with tempfile.TemporaryDirectory() as scratch:
        manager, panel = _panel(scratch)
        manager.reset_to_defaults()
        panel.load_from_config()
        start = panel.tabs["start"]
        start.choose_body("custom")
        start._choose_mode("body")
        card = start.open_card("custom", "body", "time_series",
                               apply_defaults=True)
        card.widgets["generic_body.name"].setText("some other world")
        card.widgets["generic_body.surface.ks"].setValue(0.02)
        start.stack.setCurrentIndex(start.LANDING)

        for body, mode in (("Moon", "global"), ("Mercury", "global")):
            emitted, path = _emit(manager, panel, body, mode, "time_series",
                                  scratch, tag="_after_custom")
            assert "generic_body" not in emitted, (
                f"{path}: the Custom-body card's definition survived into a "
                f"{body} run: {emitted['generic_body']!r}")
            assert str((emitted.get("simulation") or {}).get("body", "")
                       ).strip().lower() == body.lower(), (
                f"{path}: simulation.body is not {body}")
        print("    [sequence] Custom-body edits -> Moon and Mercury: "
              "emitted configs carry no generic body")


# --------------------------------------------------------------------------
# 2. Every route, against every other route's residue
# --------------------------------------------------------------------------

def test_every_route_emits_its_own_body_after_every_other_route():
    """The full Cartesian product, not just the two orderings measured.

    Each route is opened after each OTHER route has been opened and filled in,
    and the emitted file is checked for the body it claims. A pairwise sweep
    rather than a single sequence, because the defect was not specific to
    Europa or to Bennu -- it was that nothing scoped the emitted configuration
    to the active card.
    """
    from gui_core.body_cards import europa_dem_registry
    builtin = _builtin_names()
    routes = _routes()
    # The Europa terrain card's own grid, for the positive claim below. Read
    # from the shipped configurations, like the card itself; an empty registry
    # (a checkout without them) simply skips that claim rather than inventing
    # one.
    _europa_grid = europa_dem_registry(REPO_ROOT)
    _europa_meshes = {Path(e.import_file).name
                      for e in _europa_grid.values()}
    _europa_lats = {lat for e in _europa_grid.values() for lat in e.latitudes}
    with tempfile.TemporaryDirectory() as scratch:
        manager, panel = _panel(scratch)
        failures, checked = [], 0
        for prev_body, prev_mode, _ in routes:
            for body, mode, run_type in routes:
                if (prev_body, prev_mode) == (body, mode):
                    continue
                manager.reset_to_defaults()
                panel.load_from_config()
                # Fill the PREVIOUS route in, then leave it.
                start = panel.tabs["start"]
                start.choose_body(prev_body)
                start._choose_mode(prev_mode)
                prev_card = start.open_card(prev_body, prev_mode,
                                            "time_series", apply_defaults=True)
                if prev_body in ("asteroid", "custom"):
                    prev_card.widgets["generic_body.name"].setText(
                        f"leftover_{prev_body}")
                else:
                    manager.set_value("mesh.import_file",
                                      f"leftover_{prev_body}.ply")
                    if prev_mode == "local":
                        # A local-mesh card also places the mesh and names a
                        # view-factor matrix built FOR that mesh. Poisoning
                        # only import_file left the Moon-crater-to-Europa-DEM
                        # leak invisible to this arm.
                        manager.set_value("mesh.occluder_file",
                                          f"leftover_{prev_body}_ring.ply")
                        manager.set_value("mesh.latitude_override", 80.0)
                        manager.set_value("mesh.longitude_override", 12.5)
                        manager.set_value("radiation.view_factor_file",
                                          f"leftover_{prev_body}.vf")
                start.stack.setCurrentIndex(start.LANDING)

                emitted, path = _emit(
                    manager, panel, body, mode, run_type, scratch,
                    tag=f"_from_{prev_body}_{prev_mode}")
                checked += 1
                name = str((emitted.get("simulation") or {}).get("body", ""))
                where = f"{prev_body}/{prev_mode} -> {body}/{mode}/{run_type}"
                if body in builtin or body.lower() in builtin:
                    if name.strip().lower() != body.lower():
                        failures.append(
                            f"{where}: simulation.body = {name!r}")
                    if "generic_body" in emitted:
                        failures.append(
                            f"{where}: carries generic_body "
                            f"{emitted['generic_body']!r}")
                    # The MESH, on the built-in side too -- when the BODY
                    # changed. Checking only simulation.body here is what let
                    # a Moon Local crater mesh, its lunar latitude and its
                    # view-factor matrix reach a Europa DEM-site run while
                    # every other assertion in this arm stayed green.
                    #
                    # Scoped to a body change deliberately, because that is
                    # what the rule promises: a lunar mesh carried from the
                    # Moon Global card to the Moon LOCAL card is the same
                    # body's terrain arriving at a card that is about exactly
                    # that, and clearing it there would leave the beaming
                    # facet-scale rule measuring nothing on a mesh the user
                    # had already chosen (`test_gui_presets`).
                    if prev_body == body:
                        continue
                    mesh = emitted.get("mesh") or {}
                    rad = emitted.get("radiation") or {}
                    for label, value in (
                            ("mesh.import_file", mesh.get("import_file")),
                            ("mesh.occluder_file", mesh.get("occluder_file")),
                            ("radiation.view_factor_file",
                             rad.get("view_factor_file"))):
                        if f"leftover_{prev_body}" in str(value or ""):
                            failures.append(
                                f"{where}: carries {label} = {value!r}")
                    if prev_mode == "local":
                        # The rule is that the PREVIOUS body's placement does
                        # not survive -- NOT that the key comes out blank.
                        # This arm used to assert the -999.0 "not placed"
                        # sentinel, which forbade the one card whose subject IS
                        # a placed terrain from placing it: the Europa
                        # DEM card must fill the latitude in, and the
                        # positive claim for it is immediately below. Checked
                        # against the poisoned values, which is the question
                        # actually being asked.
                        for key, poison in (("latitude_override", 80.0),
                                            ("longitude_override", 12.5)):
                            if mesh.get(key) == poison:
                                failures.append(
                                    f"{where}: carries the {prev_body} card's "
                                    f"mesh.{key} = {mesh.get(key)!r}")
                    if (body, mode) == ("Europa", "local") and _europa_grid:
                        # ...and the positive half, on the destination card:
                        # arriving from another body's terrain card, the Europa
                        # terrain card still emits ITS OWN terrain at one of ITS
                        # OWN latitudes. Before 2026-08-28 it emitted neither --
                        # `mesh.import_file: ''` and the -999.0 sentinel -- and
                        # every assertion in this arm stayed green while the run
                        # became a global icosphere.
                        got = Path(str(mesh.get("import_file") or "").strip()
                                   ).name
                        if got not in _europa_meshes:
                            failures.append(
                                f"{where}: the Europa terrain card emitted "
                                f"mesh.import_file = {got!r}, which is not one "
                                f"of its terrains")
                        if mesh.get("latitude_override") not in _europa_lats:
                            failures.append(
                                f"{where}: the Europa terrain card emitted "
                                f"mesh.latitude_override = "
                                f"{mesh.get('latitude_override')!r}, which is "
                                f"not one of its shipped latitudes")
                else:
                    if name.strip().lower() in builtin:
                        failures.append(
                            f"{where}: a generic-body card emitted "
                            f"simulation.body = {name!r}, a built-in body")
                    mesh = emitted.get("mesh") or {}
                    if str(mesh.get("import_file") or "").strip():
                        failures.append(
                            f"{where}: carries mesh.import_file "
                            f"{mesh['import_file']!r}")
        assert not failures, "\n  ".join(
            [f"{len(failures)} of {checked} route pairs emitted the wrong "
             f"body:"] + failures[:30])
        print(f"    [pairwise] {checked} route pairs; every emitted "
              f"configuration names its own card's body")


# --------------------------------------------------------------------------
# 3. The dead thermophysical block
# --------------------------------------------------------------------------

#: Keys the engine's config parser does NOT read on a built-in body. It reads
#: only use_default_albedo, incidence_angle_albedo / albedo_model,
#: conductivity_model, constant_thermal_inertia and the `*_override` keys
#: (config_parser.cpp, "Parse Thermophysical section"); everything else in the
#: section produces one `[config] WARNING: N key(s) under thermophysical: are
#: read by nothing` line and is then ignored.
_DEAD_THERMOPHYSICAL = (
    "albedo", "emissivity", "ks", "kd", "H", "rhos", "rhod", "cp0", "Qb",
    "chi", "use_default_emissivity", "use_default_ks", "use_default_kd",
    "use_default_H", "use_default_rhos", "use_default_rhod",
    "use_default_cp0", "use_default_Qb", "use_default_chi",
)


def test_a_builtin_body_card_writes_overrides_not_the_dead_block():
    """No card emits the `thermophysical:` keys the engine ignores.

    Same root cause as the body leak and worth its own arm because it has its
    own symptom: ``_config_for_yaml`` gated the thermophysical CLEANUP on the
    same "is a generic body active" question, so one stale ``generic_body.name``
    also put the whole 19-key dead block back. The 2026-08-27 run printed
    ``[config] WARNING: 19 key(s) under thermophysical: are read by nothing``.

    The positive control is in the same arm: a non-default property must come
    out as an ``*_override`` key, which the engine DOES read.
    """
    with tempfile.TemporaryDirectory() as scratch:
        manager, panel = _panel(scratch)
        failures = []
        for body, mode, run_type in _routes():
            if body in ("asteroid", "custom"):
                continue         # generic bodies read generic_body.surface
            manager.reset_to_defaults()
            panel.load_from_config()
            _poison_asteroid(manager, panel)
            emitted, path = _emit(manager, panel, body, mode, run_type,
                                  scratch, tag="_thermo")
            thermo = emitted.get("thermophysical") or {}
            dead = sorted(k for k in _DEAD_THERMOPHYSICAL if k in thermo)
            if dead:
                failures.append(f"{body}/{mode}/{run_type}: {len(dead)} dead "
                                f"key(s) in {path.name}: {dead}")
        assert not failures, "\n  ".join(
            ["a card emitted thermophysical keys the engine ignores:"]
            + failures)

        # positive control: an override the engine DOES read survives.
        manager.reset_to_defaults()
        panel.load_from_config()
        manager.set_value("thermophysical.use_default_albedo", False)
        manager.set_value("thermophysical.albedo", 0.07)
        emitted, path = _emit(manager, panel, "Moon", "global", "time_series",
                              scratch, tag="_override")
        thermo = emitted.get("thermophysical") or {}
        assert thermo.get("albedo_override") == 0.07, (
            f"{path}: a non-default albedo was not written as "
            f"albedo_override; got {thermo!r}")
        assert "albedo" not in thermo, (
            f"{path}: the dead plain `albedo` key came back beside the "
            f"override")
        print("    [thermophysical] no card emits a dead key; a changed "
              "property comes out as albedo_override")


# --------------------------------------------------------------------------
# 4. Europa Global: the albedo map is wired AND present
# --------------------------------------------------------------------------

def test_europa_global_wires_the_albedo_map_at_the_chosen_resolution():
    """The card's whole point, checked on the emitted file and on disk.

    Two claims, and they are different: the configuration must ASK for the map
    (``illumination.use_albedo_map: true`` with a directory to find it in), and
    the map must BE THERE at the resolution the card chose -- otherwise the
    engine loads nothing and falls back to a uniform albedo, saying so only on
    stdout.

    A missing data file is a SKIP with the fetch command, not a failure: the
    albedo maps are downloaded, not in the repository.
    """
    from gui_core.datasets import STATE_OK, albedo_status
    with tempfile.TemporaryDirectory() as scratch:
        manager, panel = _panel(scratch)
        manager.reset_to_defaults()
        panel.load_from_config()
        _poison_asteroid(manager, panel)
        emitted, path = _emit(manager, panel, "Europa", "global",
                              "time_series", scratch, tag="_albedo")

        illum = emitted.get("illumination") or {}
        assert illum.get("use_albedo_map") is True, (
            f"{path}: Europa Global emitted use_albedo_map = "
            f"{illum.get('use_albedo_map')!r}")
        assert str(illum.get("albedo_maps_directory") or "").strip(), (
            f"{path}: use_albedo_map is on with no albedo_maps_directory, so "
            f"the engine has nowhere to look")

        resolution = (emitted.get("mesh") or {}).get("resolution")
        assert isinstance(resolution, int), (
            f"{path}: mesh.resolution is {resolution!r}")
        status = albedo_status("Europa", resolution, REPO_ROOT,
                               import_file=(emitted.get("mesh")
                                            or {}).get("import_file"))
        if status.state != STATE_OK:
            raise Skip(
                f"the Europa albedo map for resolution {resolution} is "
                f"{status.state} here ({status.detail}); the configuration "
                f"wiring above was checked, the data file was not")
        print(f"    [albedo] Europa Global res{resolution}: "
              f"use_albedo_map on, map present ({status.detail})")


# --------------------------------------------------------------------------
# 4b. Europa Local: the terrain the card SHOWS is the terrain that runs
# --------------------------------------------------------------------------
#
# The same defect as the Bennu one above, arrived at from the mesh side rather
# than the body side, and it reached a real run on 2026-08-28. The Europa
# terrain card displayed "Agenor -- 750 m posting" and the note
# "data/external/sutton_DEMs/Agenor_DEM_750m_SOCET.ply -- on disk", and emitted
# `mesh.import_file: ''`. The engine reaches the icosphere branch exactly when
# that key is empty (`ptm_thermal_sim.cpp:15087`), so the run's own
# provenance.txt recorded `mesh_source=icosphere, resolution 7`,
# `mesh_faces=327680`, `mesh_median_radius=1560.78` km -- a whole moon, from a
# card about a regional DEM -- and radiation exchange dutifully built a
# 1.13M-block H-matrix of which every block was Zero, because a sphere is
# convex.
#
# `test_gui_europa_grid.py` did not catch it and could not: it exercises
# `sweep_plan`, whose `derived=` lambdas write `mesh.import_file` from the
# registry directly. The interactive path -- open the card, press Run -- is
# what nothing covered.

def _europa_terrain_registry():
    from gui_core.body_cards import europa_dem_registry
    registry = europa_dem_registry(REPO_ROOT)
    if not registry:
        raise Skip("no configs/europa_*_<dem>.yaml in this checkout, so there "
                   "is no terrain registry to check")
    return registry


def test_europa_terrain_card_emits_the_terrain_it_shows():
    """Open the terrain card, change nothing, Run: what does the file say?

    Three claims, all on the written run configuration:

    1. ``mesh.import_file`` is NON-EMPTY. An empty one is not "no terrain
       chosen", it is a global icosphere, and the card gives no sign of it.
    2. It is the terrain the combo is DISPLAYING, not some other one.
    3. ``mesh.latitude_override`` is a real latitude, not the -999.0 "not
       placed" sentinel -- the latitude combo had the same defect as the
       terrain combo and showed 0 over a configuration that carried -999.

    Whether that file is ON DISK is checked too, and its absence is a SKIP with
    the source named: the Europa DEMs are third-party data that PTM3D does not
    distribute, so a clean checkout must not show a red gate for not having
    them. The REFUSAL for that case is the next check.
    """
    registry = _europa_terrain_registry()
    with tempfile.TemporaryDirectory() as scratch:
        manager, panel = _panel(scratch)
        manager.reset_to_defaults()
        panel.load_from_config()
        start = panel.tabs["start"]
        start.choose_body("Europa")
        start._choose_mode("local")
        card = start.open_card("Europa", "local", "time_series",
                               apply_defaults=True)
        shown_key = card.terrain_combo.currentData()
        shown = registry.get(shown_key)
        assert shown is not None, (
            f"the terrain combo is showing {card.terrain_combo.currentText()!r} "
            f"({shown_key!r}), which is not in the registry")
        shown_latitude = card.latitude_combo.currentData()

        emitted, path = _emit(manager, panel, "Europa", "local",
                              "time_series", scratch, tag="_terrain")
        mesh = emitted.get("mesh") or {}
        got = str(mesh.get("import_file") or "").strip()
        assert got, (
            f"{path}: the Europa terrain card emitted mesh.import_file = "
            f"{mesh.get('import_file')!r} while showing "
            f"{card.terrain_combo.currentText()!r}. The engine takes the "
            f"icosphere branch when that key is empty "
            f"(ptm_thermal_sim.cpp:15087), so this run would be a global "
            f"res-{mesh.get('resolution')!r} sphere -- which is what the "
            f"2026-08-28 europa_test run was.")
        assert Path(got).name == Path(shown.import_file).name, (
            f"{path}: the card showed {shown.import_file!r} and emitted "
            f"{got!r}")
        latitude = mesh.get("latitude_override")
        assert latitude != -999.0, (
            f"{path}: mesh.latitude_override is the -999.0 'not placed' "
            f"sentinel while the card showed "
            f"{card.latitude_combo.currentText()!r}. The DEM would sit at its "
            f"own coordinates, not at the latitude on screen.")
        assert shown_latitude is None or float(latitude) == float(
            shown_latitude), (
            f"{path}: the card showed latitude {shown_latitude!r} and emitted "
            f"{latitude!r}")

        resolved = Path(got)
        if not resolved.is_absolute():
            resolved = REPO_ROOT / resolved
        if not resolved.exists():
            raise Skip(
                f"{resolved} is not on this disk (the Europa DEMs are "
                f"third-party data, see data/external/README.md). The "
                f"emission was checked; the file was not.")
        print(f"    [terrain] Europa Local emits {Path(got).name} at "
              f"latitude {latitude}, and the file is on disk")


def test_every_terrain_the_card_offers_reaches_the_emitted_file():
    """Not just the one the combo happens to open on.

    A pairwise-style sweep rather than a single case, for the same reason the
    body arm above sweeps: the defect was not about Agenor, it was that nothing
    connected the control to the configuration.
    """
    registry = _europa_terrain_registry()
    with tempfile.TemporaryDirectory() as scratch:
        manager, panel = _panel(scratch)
        failures = []
        for key in sorted(registry):
            manager.reset_to_defaults()
            panel.load_from_config()
            start = panel.tabs["start"]
            start.choose_body("Europa")
            start._choose_mode("local")
            card = start.open_card("Europa", "local", "time_series",
                                   apply_defaults=True)
            index = card.terrain_combo.findData(key)
            if index < 0:
                failures.append(f"{key}: not offered by the terrain combo")
                continue
            card.terrain_combo.setCurrentIndex(index)
            emitted, path = _emit(manager, panel, "Europa", "local",
                                  "time_series", scratch, tag=f"_{key}")
            mesh = emitted.get("mesh") or {}
            got = str(mesh.get("import_file") or "").strip()
            if Path(got).name != Path(registry[key].import_file).name:
                failures.append(
                    f"{key}: emitted {got!r}, expected "
                    f"{registry[key].import_file!r}")
            if mesh.get("latitude_override") == -999.0:
                failures.append(f"{key}: emitted the -999.0 latitude sentinel")
            vf = str((emitted.get("radiation") or {}).get(
                "view_factor_file") or "")
            if Path(vf).name != Path(registry[key].view_factor_file).name:
                failures.append(
                    f"{key}: emitted view_factor_file {vf!r}, expected "
                    f"{registry[key].view_factor_file!r}")
        assert not failures, "\n  ".join(
            ["a terrain the card offers did not reach the emitted file:"]
            + failures)
        print(f"    [terrain] all {len(registry)} terrains reach "
              f"mesh.import_file, their latitude and their view-factor path")


def test_a_terrain_whose_file_is_absent_is_refused_and_says_where_to_get_it():
    """The missing-file case, which is how this started.

    The card writes the path whether or not the file is there -- that is
    deliberate, and it is what turns a silent substitution into a refusal. This
    check is the other half: that the refusal actually happens, names the exact
    path, and says where the file comes from.

    Negative control in the same arm: an ordinary missing file, not under
    ``data/external/``, gets no sourcing hint invented for it.
    """
    from gui_core.preflight import missing_external_data_hint, run_preflight

    registry = _europa_terrain_registry()
    entry = registry[sorted(registry)[0]]
    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch)
        (root / "data" / "external" / "sutton_DEMs").mkdir(parents=True)
        exe = root / "build" / "bin" / "ptm_thermal_sim"
        exe.parent.mkdir(parents=True)
        exe.write_text("#!/bin/sh\nexit 0\n")
        config = {
            "simulation": {"body": "Europa"},
            "mesh": {"import_file": entry.import_file, "resolution": 5},
            "radiation": {"enabled": False},
            "illumination": {"albedo_source": "none"},
            "output": {"directory": str(root / "out")},
        }
        report = run_preflight(config, lambda: str(exe), project_root=root)
        assert not report.ok, (
            "a configuration naming a Europa DEM that is not on disk passed "
            "pre-flight")
        blocking = "\n".join(report.errors)
        assert entry.import_file.split("/")[-1] in blocking, (
            f"the refusal does not name the missing file:\n{blocking}")
        assert "data/external/README.md" in blocking, (
            f"the refusal does not say where the file comes from:\n{blocking}")
        print("    [terrain] a missing DEM is refused by name, with its "
              "source")

    assert missing_external_data_hint("/tmp/somewhere/mesh.ply") == "", (
        "a missing file outside data/external/ was given a collaborator-data "
        "hint it has no business carrying")
    hinted = missing_external_data_hint(
        "/x/data/external/sutton_DEMs/Agenor_DEM_750m_SOCET.ply")
    assert "data/external/README.md" in hinted, hinted
    # A SOURCE PATH IS NAMED ONLY WHEN IT IS REALLY THERE. The table ships to
    # every checkout, and "get it from <path>" that is wrong on the reader's
    # machine is worse than no hint at all -- so the candidates are probed, and
    # this is the check that they are.
    for token in hinted.replace(",", " ").split():
        if token.startswith("/") and token.endswith(".ply"):
            assert Path(token).exists(), (
                f"the hint names {token} as the source, and it is not on this "
                f"disk")
    # An external dataset with no table entry still gets told what it is.
    generic = missing_external_data_hint("/x/data/external/nosuch/thing.ply")
    assert "nosuch" in generic and "README" in generic, generic


def test_a_missing_view_factor_cache_offers_the_choice_not_a_refusal():
    """The SECOND missing artifact, hit the same afternoon as the first.

    The card wires ``radiation.view_factor_file`` to
    ``data/external/sutton_DEMs/<DEM>.vf``, which is not in any checkout --
    PTM3D builds that matrix itself, so it is neither shipped nor obtainable
    with the elevation models beside it. The pre-flight used to make that a
    hard ERROR -- which blocked all 45 shipped ``configs/europa_lat*_*.yaml``
    on any machine that had not built the caches, while the command line ran
    them happily, because the engine's behavior for a named cache that does
    not exist is to BUILD it and SAVE it there
    (``ptm_thermal_sim.cpp:15716-15734``).

    So it is a CHOICE -- point at an existing matrix, or let this run build one
    -- and the message has to say so. It is still gated: the view-factor rebuild hazard override
    checkbox and the cost estimate are what stand between a user and an
    accidental multi-GB rebuild.

    The negative control is the part that must NOT relax: a cache that EXISTS
    with the wrong face count is still a refusal, because that one is not a
    choice, it is the wrong matrix.
    """
    import struct
    from gui_core.preflight import run_preflight

    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch)
        exe = root / "build" / "bin" / "ptm_thermal_sim"
        exe.parent.mkdir(parents=True)
        exe.write_text("#!/bin/sh\nexit 0\n")
        config = {
            "simulation": {"body": "Europa"},
            "mesh": {"resolution": 3, "import_file": "",
                     "use_topographic": False},
            "radiation": {
                "enabled": True,
                "view_factor_file":
                    "data/external/sutton_DEMs/Agenor_DEM_750m_SOCET.vf"},
            "illumination": {"albedo_source": "builtin"},
            "output": {"directory": str(root / "out")},
        }
        report = run_preflight(config, lambda: str(exe), project_root=root)
        assert not [e for e in report.errors
                    if "view_factor_file" in e], (
            "a view-factor cache that does not exist yet was refused; the "
            "engine builds and saves one:\n" + "\n".join(report.errors))
        warning = report.vf_rebuild_warning
        assert warning, "nothing was said about the missing view-factor cache"
        assert "Agenor_DEM_750m_SOCET.vf" in warning, warning
        assert "POINT" in warning and "PROCEED" in warning, (
            f"the message does not offer both choices:\n{warning}")

        # negative control: the WRONG matrix is still a refusal.
        vf = root / "data" / "external" / "sutton_DEMs"
        vf.mkdir(parents=True)
        wrong = vf / "Agenor_DEM_750m_SOCET.vf"
        wrong.write_bytes(struct.pack("<IIQQ", 0x5646464D, 2, 999, 999)
                          + b"\0" * 8)
        report = run_preflight(config, lambda: str(exe), project_root=root)
        assert [e for e in report.errors if "face count" in e], (
            "a view-factor matrix built for a different mesh (999 faces vs "
            "1280) was not refused:\n" + "\n".join(report.errors))
        print("    [vf] a missing cache offers point-at-one or build-one; a "
              "cache with the wrong face count is still refused")


def test_the_terrain_card_writes_its_dem_only_into_a_europa_run():
    """The mirror image of the rule this card is usually on the receiving end of.

    The panel calls ``load_from_config`` on whichever card is ACTIVE, and
    ``deactivate`` has not run at that point -- so a Europa terrain card left
    open while a lunar configuration is loaded is a card writing a Europa DEM
    into a Moon run. Measured while building the fix: the Moon configuration
    then routed to Moon/LOCAL, because it had acquired an imported mesh.
    """
    _europa_terrain_registry()
    moon = REPO_ROOT / "examples" / "quickstart_moon_diurnal.yaml"
    if not moon.exists():
        raise Skip("examples/quickstart_moon_diurnal.yaml is not in this "
                   "checkout")
    with tempfile.TemporaryDirectory() as scratch:
        manager, panel = _panel(scratch)
        manager.reset_to_defaults()
        panel.load_from_config()
        start = panel.tabs["start"]
        start.choose_body("Europa")
        start._choose_mode("local")
        start.open_card("Europa", "local", "time_series", apply_defaults=True)
        assert str(manager.get_value("mesh.import_file", "") or "").strip(), (
            "the terrain card did not write a terrain at all; this check is "
            "measuring nothing")

        # The Europa card is still ACTIVE. Load a lunar configuration.
        manager.load_from_yaml(str(moon))
        panel.load_from_config()
        leaked = str(manager.get_value("mesh.import_file", "") or "").strip()
        assert not leaked, (
            f"loading {moon.name} while the Europa terrain card was open gave "
            f"the lunar configuration mesh.import_file = {leaked!r}")
        route = panel.land_on_loaded_config()
        assert (route.body_key, route.mode_key) == ("Moon", "global"), (
            f"{moon.name} landed on {route.body_key}/{route.mode_key}")

        # The harder half, which a body check cannot catch: a EUROPA file that
        # is a global icosphere run. Same card, same body, and it still must
        # not acquire a regional DEM by being looked at.
        glob = REPO_ROOT / "configs" / "europa_ico6_uniform-ice.yaml"
        if glob.exists():
            manager.reset_to_defaults()
            panel.load_from_config()
            start.choose_body("Europa")
            start._choose_mode("local")
            start.open_card("Europa", "local", "time_series",
                            apply_defaults=True)
            manager.load_from_yaml(str(glob))
            panel.load_from_config()
            leaked = str(manager.get_value("mesh.import_file", "") or "").strip()
            assert not leaked, (
                f"opening the global {glob.name} while the Europa terrain card "
                f"was open gave it mesh.import_file = {leaked!r}, turning a "
                f"whole-moon run into a regional DEM run")
            route = panel.land_on_loaded_config()
            assert (route.body_key, route.mode_key) == ("Europa", "global"), (
                f"{glob.name} landed on {route.body_key}/{route.mode_key}")
        print("    [terrain] an open Europa terrain card does not put a DEM "
              "into a lunar configuration, nor into a Europa global one")


# --------------------------------------------------------------------------
# 4c. One albedo field, not two
# --------------------------------------------------------------------------

def test_the_europa_cards_never_emit_two_albedo_fields():
    """`use_albedo_map: false` now DOES stop radiation coupling loading a map.

    UPDATED 2026-08-30 with the engine fix. Until then the
    radiation-coupling albedo load honored no configuration key at all, so a
    `use_albedo_map: false` run could still take its radiosity source albedo
    off the map -- two albedo fields in one simulation -- and the engine only
    warned about it after the run had started. The 2026-08-28 europa_test run
    printed exactly that. The engine now honors the switch at that call site,
    so the pre-flight rule that warned about it is retired with it, and this
    check is inverted: the detector must be SILENT on the plain switch-off
    case, and must still fire on the one mixed state that survives (an
    explicit `illumination.albedo_map`, which the absorbed-flux composer
    honors ahead of the switch while the coupling has no FITS reader).

    The rule the cards follow is unchanged: a GLOBAL Europa run uses the map,
    and a LOCAL Europa run uses a constant albedo on an IMPORTED mesh, where
    the per-face map cannot load at all because it is matched to an icosphere
    by face count.
    """
    from gui_core.preflight import radiation_coupling_albedo_map
    from gui_core.config_manager import illumination_for_engine

    with tempfile.TemporaryDirectory() as scratch:
        manager, panel = _panel(scratch)

        manager.reset_to_defaults()
        panel.load_from_config()
        emitted, path = _emit(manager, panel, "Europa", "global",
                              "time_series", scratch, tag="_albedo_global")
        assert (emitted.get("illumination") or {}).get(
            "use_albedo_map") is True, (
            f"{path}: the Europa Global card is the albedo-map card and "
            f"emitted use_albedo_map off")

        _europa_terrain_registry()
        manager.reset_to_defaults()
        panel.load_from_config()
        emitted, path = _emit(manager, panel, "Europa", "local",
                              "time_series", scratch, tag="_albedo_local")
        illum = emitted.get("illumination") or {}
        assert illum.get("use_albedo_map") is False, (
            f"{path}: a regional DEM run emitted use_albedo_map on; the "
            f"per-face map is matched to an icosphere by face count and "
            f"cannot apply to an imported mesh")
        found, _n = radiation_coupling_albedo_map(emitted, illum, REPO_ROOT)
        assert found is None, (
            f"{path}: this configuration asks for a uniform albedo and "
            f"radiation coupling would still load {found}, so the run would "
            f"carry two different albedo fields")

    # The measured 2026-08-28 configuration, on which the detector must now be
    # SILENT -- the engine honors the switch, so there is no mixed state left
    # to report -- and the explicit-map configuration, on which it must still
    # fire. Without this the check above proves only that today's cards are
    # clean, not that anything would notice if they stopped being.
    #
    # Driven through a REAL ConfigManager and handed `get_config()` -- the raw
    # dictionary, which is exactly what `ptm3d_gui.run_simulation` passes to
    # `run_preflight`. A hand-built dictionary here passed while the detector
    # was DEAD on every actual GUI run: DEFAULT_CONFIG always carries a
    # `generic_body` scaffold of empty keys, and the first draft read its
    # truthiness as "this is a generic body" and bailed. A check exercised only
    # on a shape of input the product never produces is not a check.
    from gui_core.preflight import run_preflight
    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch)
        exe = root / "build" / "bin" / "ptm_thermal_sim"
        exe.parent.mkdir(parents=True)
        exe.write_text("#!/bin/sh\nexit 0\n")
        maps = REPO_ROOT / "data" / "albedo_maps"
        if not (maps / "Europa_Albedo_res5.bin").exists():
            raise Skip("data/albedo_maps/Europa_Albedo_res5.bin is not on this "
                       "disk, so the mixed-albedo detector has nothing to find")
        manager, panel = _panel(scratch)
        manager.reset_to_defaults()
        panel.load_from_config()
        manager.set_value("simulation.body", "Europa")
        manager.set_value("mesh.resolution", 5)
        manager.set_value("mesh.import_file", "")
        manager.set_value("mesh.use_topographic", False)
        manager.set_value("radiation.enabled", True)
        manager.set_value("radiation.view_factor_file", "")
        manager.set_value("illumination.albedo_source", "none")
        manager.set_value("output.directory", str(root / "out"))
        assert manager.get_config().get("generic_body"), (
            "DEFAULT_CONFIG no longer carries a generic_body scaffold, so this "
            "check no longer covers the case it was written for")

        def albedo_warnings():
            report = run_preflight(manager.get_config(), lambda: str(exe),
                                   project_root=REPO_ROOT,
                                   config_manager=manager)
            return [w for w in report.warnings if w.startswith("albedo:")]

        assert not albedo_warnings(), (
            "the retired mixed-albedo warning still fires on a uniform-albedo "
            "radiation run with a res-matched per-face map on disk. The engine "
            "honors illumination.use_albedo_map at that call site since "
            "the coupling albedo-load fix, so this warning is now false -- and its advice "
            "('turn the albedo map on so both use it') would change the "
            "physics of the run")

        # ...but the load itself is still correctly IDENTIFIED, so a future
        # regression that stops honoring the switch has something to find.
        # Silence must come from the engine's behavior, not from the GUI
        # losing the ability to see the file.
        found, n_found = radiation_coupling_albedo_map(
            manager.get_config(),
            illumination_for_engine(
                (manager.get_config().get("illumination") or {})),
            REPO_ROOT)
        assert found is not None and "Europa_Albedo_res5.bin" in str(found), (
            "the pre-flight can no longer resolve the res-matched per-face "
            "map at all, so this check would pass even if the engine went "
            "back to loading it regardless of the switch")

        # The one mixed state that survives the engine fix: an EXPLICIT
        # illumination.albedo_map with the switch off. The absorbed-flux
        # composer tests that key BEFORE use_albedo_map and honors it; the
        # radiation coupling has no FITS reader and stays uniform.
        #
        # The GUI cannot reach it -- albedo_source "none" clears albedo_map --
        # and that is the first thing asserted, because "the pre-flight would
        # catch it" is a weaker guarantee than "the emitter cannot write it".
        assert illumination_for_engine(
            {"albedo_source": "none", "albedo_map": "x.fits"}
        )["albedo_map"] == "", (
            "the GUI's albedo-source translation no longer clears an explicit "
            "albedo_map when the source is None, so a card could now emit the "
            "one mixed-albedo state the engine fix does not remove")

        # ...but a configuration read back from disk has no albedo_source, so
        # it CAN carry that pair, and then the pre-flight is the only detector.
        # Raw dict on purpose: that is the already-emitted shape.
        explicit_cfg = {
            "simulation": {"body": "Europa"},
            "mesh": {"resolution": 5, "import_file": "", "occluder_file": "",
                     "use_topographic": True},
            "radiation": {"enabled": True, "view_factor_file": ""},
            "illumination": {"use_albedo_map": False,
                             "albedo_map": str(maps / "explicit.fits"),
                             "albedo_maps_directory": ""},
            "output": {"directory": str(root / "out2")},
        }
        explicit_report = run_preflight(explicit_cfg, lambda: str(exe),
                                        project_root=REPO_ROOT)
        explicit = [w for w in explicit_report.warnings if w.startswith("albedo:")]
        assert explicit and "albedo_map" in explicit[0], (
            "an explicit illumination.albedo_map with use_albedo_map off was "
            "not reported; the surface boundary condition would use that map "
            "and the radiosity source albedo would not")

        # negative control 1: the map ON -- both fields are the map.
        manager.set_value("illumination.albedo_source", "builtin")
        assert not albedo_warnings(), (
            "an albedo warning fired on a run that uses the map for "
            "both fields")

        # negative control 2: radiation off -- there is no second field.
        manager.set_value("illumination.albedo_source", "none")
        manager.set_value("radiation.enabled", False)
        assert not albedo_warnings(), (
            "an albedo warning fired with radiation exchange off")

        # negative control 3: a real generic body has no published map, and
        # `target_body_name` never reaches that loader at all.
        manager.set_value("radiation.enabled", True)
        manager.set_value("generic_body.name", "623 Chimaera (A907 BC)")
        assert not albedo_warnings(), (
            "an albedo warning fired on a generic body, whose "
            "radiation-coupling load cannot resolve a map name at all")
        manager.set_value("generic_body.name", "")

        # and the translation the pre-flight now shares with the emitter --
        # including the case that must be left ALONE, an already-emitted
        # section with no albedo_source key.
        assert illumination_for_engine(
            {"albedo_source": "none", "use_albedo_map": True}
        )["use_albedo_map"] is False
        already = {"use_albedo_map": False, "albedo_map": "",
                   "albedo_maps_directory": ""}
        assert illumination_for_engine(already) == already, (
            "translating an already-emitted illumination section changed it; "
            "defaulting the absent albedo_source to 'builtin' turned a "
            "uniform-albedo run back into a map run")
        print("    [albedo] Europa Global uses the map, Europa Local cannot, "
              "the switch is honored, and the surviving explicit-map mixed "
              "state is named when it happens")


# --------------------------------------------------------------------------
# 4d. Radiation exchange on a convex sphere is a no-op that costs GB
# --------------------------------------------------------------------------

def test_radiation_on_a_smooth_global_icosphere_is_reported():
    """It is not wrong, it is empty -- and it took two minutes and 18 GB.

    A smooth icosphere is convex: no facet sees another, every view factor is
    zero. The 2026-08-28 run's H-matrix was 1.13M blocks, all Zero, all row
    sums 0.0, indirect flux 0 everywhere. The pre-flight says so; it does NOT
    turn the switch off, because a global mesh WITH topography is a different
    and perfectly good run.
    """
    from gui_core.preflight import run_preflight
    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch)
        exe = root / "build" / "bin" / "ptm_thermal_sim"
        exe.parent.mkdir(parents=True)
        exe.write_text("#!/bin/sh\nexit 0\n")
        base = {
            "simulation": {"body": "Europa"},
            "mesh": {"resolution": 5, "import_file": "", "occluder_file": "",
                     "use_topographic": False},
            "radiation": {"enabled": True, "view_factor_file": ""},
            "illumination": {"albedo_source": "builtin"},
            "output": {"directory": str(root / "out")},
        }

        def convex_warnings(cfg):
            rep = run_preflight(cfg, lambda: str(exe), project_root=REPO_ROOT)
            return [w for w in rep.warnings if "convex" in w]

        assert convex_warnings(base), (
            "radiation exchange on a smooth global icosphere was not reported "
            "as a no-op")

        # ...and an occluder ring does NOT rescue it. The engine adds that
        # geometry to the shadow-ray BVH only -- "NOT thermally solved /
        # radiosity-coupled / output" (ptm_thermal_sim.cpp:16405-16410) -- so
        # the radiation problem is still convex. Treating the ring as terrain
        # would silence this on exactly the lunar-polar-style configurations
        # most likely to carry one.
        ringed = {**base, "mesh": {**base["mesh"],
                                   "occluder_file": "ring_Rcore3_Rext20km.ply"}}
        assert convex_warnings(ringed), (
            "an occluder ring silenced the convex-mesh warning; the occluder "
            "is shadow-only and is not radiosity-coupled")

        # negative control 1: topography -- real relief, real self-heating.
        topo = {**base, "mesh": {**base["mesh"], "use_topographic": True}}
        assert not convex_warnings(topo), (
            "the convex-mesh warning fired on a global run WITH topography")

        # negative control 2: an imported mesh.
        imported = {**base, "mesh": {**base["mesh"],
                                     "import_file": "some_terrain.ply"}}
        assert not convex_warnings(imported), (
            "the convex-mesh warning fired on an imported mesh")

        # negative control 3: radiation off.
        off = {**base, "radiation": {"enabled": False}}
        assert not convex_warnings(off), (
            "the convex-mesh warning fired with radiation exchange off")
        print("    [convex] radiation exchange on a smooth global icosphere is "
              "reported; topography, an imported mesh and radiation-off are "
              "silent")


# --------------------------------------------------------------------------
# 5. The negative control -- this gate can fail
# --------------------------------------------------------------------------

def test_the_conflict_detector_fires_on_a_configuration_that_has_one():
    """A check that has never been seen to fail has not been tested.

    Builds the 2026-08-27 configuration by hand and asserts the detector names
    it, then asserts the two configurations it must NOT fire on: a legitimate
    generic body, and a built-in body with no generic section at all.
    """
    from gui_core.config_manager import ConfigManager

    manager = ConfigManager()
    manager.set_value("simulation.body", "Europa")
    manager.set_value("generic_body.name", "bennu")
    conflict = manager.body_identity_conflict()
    assert conflict, (
        "a configuration naming Europa with a generic_body: section was not "
        "reported as naming two bodies")
    assert "bennu" in conflict and "Europa" in conflict, conflict
    emitted = manager._config_for_yaml()
    assert "generic_body" not in emitted, emitted.get("generic_body")
    assert manager.last_body_identity_conflict == conflict

    # negative control 1: a real generic body is untouched.
    manager = ConfigManager()
    manager.set_value("simulation.body", "623 Chimaera (A907 BC)")
    manager.set_value("generic_body.name", "623 Chimaera (A907 BC)")
    manager.set_value("generic_body.shape.radius_km", 22.045)
    assert manager.body_identity_conflict() == ""
    emitted = manager._config_for_yaml()
    assert (emitted.get("generic_body") or {}).get("name") \
        == "623 Chimaera (A907 BC)", (
        "a legitimate generic body was dropped from the emitted configuration")

    # negative control 2: a built-in body with no generic section.
    manager = ConfigManager()
    manager.set_value("simulation.body", "Moon")
    assert manager.body_identity_conflict() == ""
    assert "generic_body" not in manager._config_for_yaml()

    # negative control 3: an explicit HORIZONS selection keeps its body.
    manager = ConfigManager()
    manager.set_value("simulation.body", "[H] 269 Justitia (A887 SA)")
    manager.set_value("generic_body.name", "269 Justitia (A887 SA)")
    assert manager.body_identity_conflict() == ""
    assert "generic_body" in manager._config_for_yaml()
    print("    [detector] fires on the measured conflict; silent on a real "
          "generic body, a plain built-in body and a HORIZONS selection")


def test_every_shipped_configuration_names_one_body():
    """The corpus the emission rule was derived from, held to it.

    Twenty-five shipped configurations carry a ``generic_body:`` section and
    every one of them sets ``simulation.body`` to that same body's name. That
    is what makes "a built-in ``simulation.body`` beside a ``generic_body:``
    section is always residue" safe to act on -- and if somebody adds a
    configuration that breaks the pattern, the rule needs revisiting, so this
    fails rather than letting it pass unnoticed.
    """
    builtin = _builtin_names()
    offenders, checked = [], 0
    for folder in ("examples", "configs"):
        for path in sorted((REPO_ROOT / folder).rglob("*.yaml")):
            try:
                data = yaml.safe_load(path.read_text())
            except Exception:                                # noqa: BLE001
                continue
            if not isinstance(data, dict) or not data.get("generic_body"):
                continue
            checked += 1
            body = str((data.get("simulation") or {}).get("body", "")).strip()
            if body.lower() in builtin:
                offenders.append(
                    f"{path.relative_to(REPO_ROOT)}: simulation.body = "
                    f"{body!r} beside generic_body "
                    f"{data['generic_body'].get('name')!r}")
    assert not offenders, "\n  ".join(
        ["a shipped configuration names a built-in body AND a generic body, "
         "so the emission rule would strip its generic_body section:"]
        + offenders)
    print(f"    [corpus] {checked} shipped configurations carry a "
          f"generic_body: section; none names a built-in body as well")


# --------------------------------------------------------------------------
# 6. What clearing another card's residue must NOT take with it
# --------------------------------------------------------------------------

def test_a_shape_model_survives_its_own_cards_mesh_clear():
    """The generic cards clear `mesh.import_file` -- and put their own back.

    `mesh.import_file` is a MIRROR of `generic_body.shape.mesh_file`: the
    solver reads the top-level path (``ptm_thermal_sim.cpp:15087``) and the
    generic-body one is metadata. Clearing it to remove the PREVIOUS body's
    terrain would also delete a `custom_mesh` asteroid's own shape model, and
    the card's `_propagate_shape_mesh` cannot restore it (it returns early
    while a card is loading). The run would then use the procedural sphere
    from `mesh.resolution` with the shape model unread on disk -- silently,
    which is the failure that mirror exists to prevent.
    """
    with tempfile.TemporaryDirectory() as scratch:
        manager, panel = _panel(scratch)
        manager.reset_to_defaults()
        panel.load_from_config()
        start = panel.tabs["start"]
        start.choose_body("asteroid")
        start._choose_mode("body")
        card = start.open_card("asteroid", "body", "time_series",
                               apply_defaults=True)
        card.widgets["generic_body.name"].setText("shapey")
        card.widgets["generic_body.shape.type"].setCurrentText("custom_mesh")
        card.widgets["generic_body.shape.mesh_file"].setText(
            "data/horizons_cache/bodies/shapey/shape.obj")
        card.widgets["generic_body.shape.mesh_scale"].setValue(2.5)
        assert str(manager.get_value("mesh.import_file", "")).endswith(
            "shape.obj"), "the card did not mirror the shape model at all"
        start.stack.setCurrentIndex(start.LANDING)

        # Leave and come back: the clear runs on the way in.
        emitted, path = _emit(manager, panel, "asteroid", "body",
                              "time_series", scratch, tag="_shape")
        mesh = emitted.get("mesh") or {}
        assert str(mesh.get("import_file") or "").endswith("shape.obj"), (
            f"{path}: the asteroid's own shape model was cleared and not "
            f"restored; mesh.import_file = {mesh.get('import_file')!r}. The "
            f"run would use the icosphere from mesh.resolution "
            f"{mesh.get('resolution')!r} with the shape model unread.")
        assert mesh.get("scale_factor") == 2.5, (
            f"{path}: the shape model came back at scale "
            f"{mesh.get('scale_factor')!r}, not 2.5")
        print("    [shape] a custom_mesh asteroid keeps its own shape model "
              "across a card round trip")


def test_a_generic_body_config_without_simulation_body_is_left_alone():
    """The identity rule must not fire on DEFAULT_CONFIG showing through.

    `simulation.body` is optional in a generic-body configuration -- the
    engine ignores the key entirely when a generic body is configured, and all
    seven cached ``body_config.yaml`` files are written without it. The merge
    with DEFAULT_CONFIG then supplies 'Moon'. Reading that as "the user named
    the Moon" would delete the whole ``generic_body:`` section of a perfectly
    good asteroid file and emit a Moon run -- the opposite of the defect the
    rule exists to fix, with the same silence.
    """
    import yaml as _yaml
    from gui_core.config_manager import ConfigManager

    with tempfile.TemporaryDirectory() as scratch:
        path = Path(scratch) / "asteroid_no_body.yaml"
        path.write_text(_yaml.dump({
            "simulation": {"duration_days": 1.0, "timesteps": 48},
            "generic_body": {
                "name": "623 Chimaera (A907 BC)",
                "shape": {"type": "spherical", "radius_km": 22.045},
                "orbit": {"semi_major_axis_au": 2.46093257,
                          "eccentricity": 0.11327891},
                "spin": {"period_hours": 14.635},
            },
        }))
        manager = ConfigManager()
        manager.load_from_yaml(str(path))
        assert manager.body_is_explicit is False, (
            "a file with no simulation.body was recorded as naming one")
        assert manager.body_identity_conflict() == "", (
            f"a generic-body file that never mentions simulation.body was "
            f"reported as naming two bodies:\n"
            f"{manager.body_identity_conflict()}")
        emitted = manager._config_for_yaml()
        assert (emitted.get("generic_body") or {}).get("name") \
            == "623 Chimaera (A907 BC)", (
            f"the body definition was dropped: "
            f"{emitted.get('generic_body')!r}")

        # ...and the positive control, same file, one key added.
        manager2 = ConfigManager()
        manager2.load_from_yaml(str(path))
        manager2.set_value("simulation.body", "Europa")
        assert manager2.body_is_explicit is True
        assert manager2.body_identity_conflict(), (
            "naming Europa over the same generic body was not reported")
        assert "generic_body" not in manager2._config_for_yaml()
    print("    [optional body] a generic-body file with no simulation.body "
          "keeps its body; adding one that contradicts it is reported")


def test_jupiter_shine_is_a_body_decision_not_an_inheritance():
    """`illumination.include_primary_reflection` follows the body, not history.

    The generic-body cards set it False, which is right for a heliocentric
    asteroid and wrong for a moon: Europa's night side is lit by Jupiter.
    Nothing restored it, so an asteroid card visited earlier in the session
    turned Jupiter-shine off for every later Europa run.
    """
    with tempfile.TemporaryDirectory() as scratch:
        manager, panel = _panel(scratch)
        manager.reset_to_defaults()
        panel.load_from_config()
        default = manager.get_value("illumination.include_primary_reflection")
        _poison_asteroid(manager, panel)
        assert manager.get_value(
            "illumination.include_primary_reflection") is False, (
            "the asteroid card no longer turns primary reflection off; this "
            "check is measuring nothing")
        emitted, path = _emit(manager, panel, "Europa", "global",
                              "time_series", scratch, tag="_jupiter")
        got = (emitted.get("illumination") or {}).get(
            "include_primary_reflection")
        assert got == default, (
            f"{path}: Europa Global emitted include_primary_reflection = "
            f"{got!r}, inherited from the asteroid card; the engine default "
            f"is {default!r} and Europa's night side is lit by Jupiter")
        print(f"    [jupiter] Europa Global emits "
              f"include_primary_reflection = {got!r} after an asteroid card")


CHECKS = [
    test_asteroid_card_residue_never_reaches_a_builtin_body_run,
    test_a_shape_model_survives_its_own_cards_mesh_clear,
    test_a_generic_body_config_without_simulation_body_is_left_alone,
    test_jupiter_shine_is_a_body_decision_not_an_inheritance,
    test_builtin_mesh_residue_never_reaches_a_generic_body_run,
    test_custom_body_residue_never_reaches_a_builtin_body_run,
    test_every_route_emits_its_own_body_after_every_other_route,
    test_a_builtin_body_card_writes_overrides_not_the_dead_block,
    test_europa_global_wires_the_albedo_map_at_the_chosen_resolution,
    test_europa_terrain_card_emits_the_terrain_it_shows,
    test_every_terrain_the_card_offers_reaches_the_emitted_file,
    test_a_terrain_whose_file_is_absent_is_refused_and_says_where_to_get_it,
    test_a_missing_view_factor_cache_offers_the_choice_not_a_refusal,
    test_the_terrain_card_writes_its_dem_only_into_a_europa_run,
    test_the_europa_cards_never_emit_two_albedo_fields,
    test_radiation_on_a_smooth_global_icosphere_is_reported,
    test_the_conflict_detector_fires_on_a_configuration_that_has_one,
    test_every_shipped_configuration_names_one_body,
]


def main():
    failed = skipped = 0
    for check in CHECKS:
        name = check.__name__
        try:
            check()
        except Skip as exc:
            skipped += 1
            print(f"SKIP {name}: {exc}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {name}:\n  {exc}")
        except Exception as exc:                             # noqa: BLE001
            failed += 1
            import traceback
            print(f"ERROR {name}: {exc!r}")
            traceback.print_exc()
        else:
            print(f"PASS {name}")
    total = len(CHECKS)
    print(f"\n{total - failed - skipped}/{total} passed, {skipped} skipped, "
          f"{failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
