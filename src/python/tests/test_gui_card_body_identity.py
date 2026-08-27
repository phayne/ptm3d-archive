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
    builtin = _builtin_names()
    routes = _routes()
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
                        for key in ("latitude_override", "longitude_override"):
                            if mesh.get(key) != -999.0:
                                failures.append(
                                    f"{where}: carries mesh.{key} = "
                                    f"{mesh.get(key)!r}")
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
# 3. The dead thermophysical block (PTM-67)
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
