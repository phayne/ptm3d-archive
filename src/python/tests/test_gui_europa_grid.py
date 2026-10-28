#!/usr/bin/env python3
"""The Europa card's grid reproduces the forty-five shipped configurations.

``configs/europa_lat{00,15,30,45,60,75,80,85}_{agenor,cilix,pwyll,rhad,ridges}
.yaml`` plus ``configs/europa_polar_{...}.yaml`` is a latitude-by-terrain
Cartesian product materialized as one file per point: **nine latitudes by five
terrains, forty-five files**, differing in two scientific keys. The card
reproduces it from five controls.

That is a claim about forty-five files, so this file checks it point by point
rather than asserting a count. Three differences are EXPECTED, and each is here
because it was measured, not because the comparison needed it to pass:

1. ``output.directory`` -- by construction. Every point gets its own.

2. ``radiation.view_factor_file`` -- the shipped value is RELATIVE, and the
   engine resolves a relative view-factor path against the YAML file's own
   directory (``config_parser.cpp:507-517``), i.e. under ``configs/``. There is
   no ``configs/data/external/`` in any checkout, so the shipped files would
   rebuild the matrix rather than load the cache. The card writes an absolute
   path to where ``data/external/`` actually is. Same intended file; the card's
   is the one that resolves. The test asserts they name the same file.

3. The plain ``thermophysical:`` values become ``*_override`` values. On a
   built-in body the engine reads only the override keys -- ``thermophysical.H``
   and its nine siblings are read by nothing -- so the
   shipped files state ten properties the engine ignores, and the card states
   the four that have an override path. Every value is Europa's own, so the two
   are the same run. MEASURED, not argued: two ico3 Europa runs, one with the
   shipped block and one with the card's, produce **byte-identical statistics
   in all sixteen binaries**. ``test_the_property_translation_is_inert`` re-runs
   that comparison here whenever an engine is available.

Run directly::

    python3 src/python/tests/test_gui_europa_grid.py
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "python"))
sys.path.insert(0, str(REPO_ROOT / "gui"))

#: (terrain, latitude) is read from each file's own mesh.latitude_override, not
#: from its name: the five `europa_polar_*` configurations sit at 87 degrees,
#: and a test that read "polar" as 90 would agree with a card that had moved
#: every polar run three degrees.
_NAME = re.compile(r"^europa_(?:lat\d+|polar)_(?P<dem>[a-z]+)\.yaml$")

#: Differ by construction; see the module docstring.
_EXPECTED_DIFFERENCES = ("output.directory", "radiation.view_factor_file")

#: The plain thermophysical keys the engine ignores on a built-in body, and the
#: override key each becomes where one exists.
_OVERRIDABLE = {
    "thermophysical.albedo": "thermophysical.albedo_override",
    "thermophysical.emissivity": "thermophysical.emissivity_override",
    "thermophysical.ks": "thermophysical.ks_override",
    "thermophysical.kd": "thermophysical.kd_override",
    "thermophysical.H": "thermophysical.H_override",
}
#: Keys the engine's parser does not read AT ALL. Five properties with no
#: override path, and nine of the ten `use_default_*` flags -- `config_parser.cpp`
#: says so in as many words at line 144: "Every `use_default_<key>` flag except
#: `use_default_albedo` is likewise unread, which is worse than merely inert".
#: Dropping them changes nothing, and carrying them forward would be carrying
#: forward a claim the engine ignores.
_DEAD_ON_A_BUILTIN_BODY = (
    "thermophysical.rhos", "thermophysical.rhod", "thermophysical.cp0",
    "thermophysical.Qb", "thermophysical.chi",
    "thermophysical.use_default_emissivity", "thermophysical.use_default_ks",
    "thermophysical.use_default_kd", "thermophysical.use_default_H",
    "thermophysical.use_default_Qb", "thermophysical.use_default_chi",
    "thermophysical.use_default_cp0", "thermophysical.use_default_rhos",
    "thermophysical.use_default_rhod",
)


class Skip(Exception):
    """A check that cannot run here, as distinct from one that failed."""


def _shipped():
    """``{(terrain, latitude): (path, config)}`` for the whole shipped grid."""
    import yaml
    out = {}
    for path in sorted((REPO_ROOT / "configs").glob("europa_*.yaml")):
        match = _NAME.match(path.name)
        if not match:
            continue
        config = yaml.safe_load(path.read_text()) or {}
        mesh = config.get("mesh") or {}
        if not str(mesh.get("import_file", "") or ""):
            continue
        try:
            latitude = float(mesh.get("latitude_override"))
        except (TypeError, ValueError):
            continue
        out[(match.group("dem"), latitude)] = (path, config)
    return out


_APP = None
_CARD = None
_MANAGER = None
_PANEL = None


def _card(scratch):
    global _APP, _CARD, _MANAGER, _PANEL
    if _CARD is not None:
        return _MANAGER, _CARD
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
    panel = _PANEL = ConfigurationPanel(_MANAGER)
    base = REPO_ROOT / "configs" / "europa_lat00_agenor.yaml"
    if not base.exists():
        raise Skip("the Europa DEM configurations are not in this checkout")
    # The base is what the CARD SHOWS, so it is loaded through the normal open
    # path. A card that swept some other file would reproduce that file, which
    # is a different and much weaker claim.
    _MANAGER.load_from_yaml(str(base))
    panel.load_from_config()
    route = panel.land_on_loaded_config()
    assert route.ok and route.body_key == "Europa" and route.mode_key == "local"
    _CARD = panel.tabs["start"].cards["Europa"]
    _CARD.activate("local", "sweep")
    return _MANAGER, _CARD


# --------------------------------------------------------------------------

def test_the_registry_is_read_from_the_shipped_files():
    from gui_core.body_cards import europa_dem_registry
    registry = europa_dem_registry(REPO_ROOT)
    shipped = _shipped()
    assert len(shipped) == 45, (
        f"{len(shipped)} shipped Europa DEM configurations, not 45")
    assert set(registry) == {dem for dem, _ in shipped}, (
        f"registry terrains {sorted(registry)} != shipped "
        f"{sorted({d for d, _ in shipped})}")
    for dem, entry in sorted(registry.items()):
        want = sorted({lat for d, lat in shipped if d == dem})
        assert list(entry.latitudes) == want, (
            f"{dem}: registry latitudes {entry.latitudes} != shipped {want}")
        path, config = shipped[(dem, entry.latitudes[0])]
        assert entry.import_file == config["mesh"]["import_file"], (
            f"{dem}: the registry's mesh path is not the shipped one")
    assert 87.0 in registry["agenor"].latitudes, (
        "the polar configurations are at 87 degrees and the registry lost "
        "them; reading 'polar' as 90 would move every polar run")
    print(f"    [registry] {len(registry)} terrains x "
          f"{len(registry['agenor'].latitudes)} latitudes, read from "
          f"{len(shipped)} shipped files")


def test_the_grid_is_exactly_the_shipped_points():
    with tempfile.TemporaryDirectory() as scratch:
        _, card = _card(scratch)
        sweep = card.sweep_plan(Path(scratch) / "grid")
        generated = {(p.params["dem"],
                      float(p.params["mesh.latitude_override"]))
                     for p in sweep}
        shipped = set(_shipped())
        assert generated == shipped, (
            f"the grid is not the shipped set.\n"
            f"  only generated: {sorted(generated - shipped)}\n"
            f"  only shipped:   {sorted(shipped - generated)}")
        assert len(generated) == 45
        print(f"    [grid] {len(generated)} points, exactly the shipped set")


def test_every_point_matches_its_shipped_configuration():
    from ptm.engine_config import flatten_config
    with tempfile.TemporaryDirectory() as scratch:
        _, card = _card(scratch)
        sweep = card.sweep_plan(Path(scratch) / "grid")
        shipped = _shipped()
        differences, compared, points = [], 0, 0
        for point in sweep:
            key = (point.params["dem"],
                   float(point.params["mesh.latitude_override"]))
            path, config = shipped[key]
            want_all = flatten_config(config)
            got_all = flatten_config(point.config())
            points += 1
            for dotted, want in sorted(want_all.items()):
                if dotted in _EXPECTED_DIFFERENCES:
                    continue
                if dotted in _DEAD_ON_A_BUILTIN_BODY:
                    assert dotted not in got_all, (
                        f"{path.name}: the card wrote {dotted}, which the "
                        f"engine reads on no built-in body")
                    continue
                if dotted in _OVERRIDABLE:
                    override = _OVERRIDABLE[dotted]
                    got = got_all.get(override, got_all.get(dotted))
                    compared += 1
                    if got != want and not (
                            dotted == "thermophysical.albedo"
                            and got_all.get("thermophysical.use_default_albedo")):
                        differences.append(
                            f"{path.name}: {dotted} = {want!r}, card wrote "
                            f"{override} = {got!r}")
                    continue
                compared += 1
                got = got_all.get(dotted, "<absent>")
                if want != got:
                    differences.append(
                        f"{path.name}: {dotted}: shipped {want!r} != "
                        f"generated {got!r}")
        assert points == 45, points
        assert compared > 2000, (
            f"only {compared} keys compared across 45 points; the comparison "
            f"is not reaching the configurations")
        assert not differences, "\n  ".join(
            [f"{len(differences)} difference(s) between the card's grid and "
             f"the shipped configurations:"] + differences[:15])
        print(f"    [match] {compared} keys compared across {points} points, "
              f"0 unexpected differences")


def test_the_view_factor_path_names_the_same_file():
    """The one expected path difference, held to naming the same cache."""
    with tempfile.TemporaryDirectory() as scratch:
        _, card = _card(scratch)
        sweep = card.sweep_plan(Path(scratch) / "grid")
        shipped = _shipped()
        for point in sweep:
            key = (point.params["dem"],
                   float(point.params["mesh.latitude_override"]))
            path, config = shipped[key]
            want = Path((config.get("radiation") or {})["view_factor_file"])
            got = Path(point.config()["radiation"]["view_factor_file"])
            assert got.name == want.name, (
                f"{path.name}: the card points at {got.name}, the shipped "
                f"configuration at {want.name}")
            assert got.is_absolute(), (
                f"{path.name}: the card's view-factor path is relative "
                f"({got}); the engine resolves a relative one against the "
                f"YAML's own directory, which for a generated point is the "
                f"point directory, not the data root")


def test_the_property_translation_is_inert():
    """Two Europa runs, the shipped block against the card's, byte-compared.

    This is the check that makes difference (3) a fact rather than a reading of
    the source. It uses the procedural icosphere, because the Europa elevation
    models are not distributed with PTM3D -- the property translation is not a
    property of the mesh, so a sphere settles it.
    """
    from gui_core.simulation_runner import SimulationRunner
    exe = SimulationRunner.find_ptm_executable()
    if not exe:
        raise Skip("no ptm_thermal_sim to run")
    kernels = REPO_ROOT / "data" / "kernels"
    if not (kernels / "europa.tm").exists() or not (kernels / "lsk").exists():
        raise Skip("the core SPICE kernel set is not in this checkout")

    shared = ("simulation: {start_time: '2030-01-01T00:00:00', "
              "duration_days: 3.55, timesteps: 8, body: Europa}\n"
              "mesh: {resolution: 3, import_file: '', scale_factor: 1.0, "
              "use_topographic: false}\n"
              "thermal: {run_thermal_mesh: true, threads: 8, "
              "equilibration_orbits: 2.0, use_gpu: false, "
              "compute_statistics: true}\n"
              "radiation: {enabled: false}\n"
              "illumination: {include_primary_reflection: true, "
              "use_batch_processing: true, use_albedo_map: false}\n"
              "advanced: {verbose: false}\n")
    blocks = {
        "shipped": ("thermophysical:\n  use_default_albedo: true\n"
                    "  albedo: 0.5\n  emissivity: 0.9\n  ks: 0.0096\n"
                    "  kd: 0.05\n  rhos: 500.0\n  rhod: 700.0\n  H: 0.07\n"
                    "  cp0: 0.0\n  Qb: 0.03\n  chi: 0.0\n"),
        "card": ("thermophysical:\n  use_default_albedo: true\n"
                 "  incidence_angle_albedo: false\n"
                 "  conductivity_model: hayne2017\n"
                 "  constant_thermal_inertia: -1.0\n"
                 "  emissivity_override: 0.9\n  ks_override: 0.0096\n"
                 "  kd_override: 0.05\n  H_override: 0.07\n"),
    }
    with tempfile.TemporaryDirectory() as scratch:
        outputs = {}
        for arm, block in blocks.items():
            out = Path(scratch) / f"out_{arm}"
            config = Path(scratch) / f"{arm}.yaml"
            config.write_text(
                shared + block
                + f"output: {{directory: {out}, format: binary, "
                  f"vtk_mode: none}}\n")
            result = subprocess.run(
                [str(exe), "--config", str(config)], capture_output=True,
                text=True, cwd=str(REPO_ROOT), timeout=900)
            assert result.returncode == 0, (
                f"the {arm} arm did not run: rc={result.returncode}\n"
                + (result.stdout + result.stderr)[-1200:])
            outputs[arm] = out / "statistics"

        names = sorted(p.name for p in outputs["shipped"].glob("*.bin"))
        assert len(names) >= 12, (
            f"only {len(names)} statistics binaries were written; the arms "
            f"did not produce a comparable output")
        differing = [n for n in names
                     if (outputs["shipped"] / n).read_bytes()
                     != (outputs["card"] / n).read_bytes()]
        assert not differing, (
            f"the property translation is NOT inert: {differing} differ "
            f"between the shipped thermophysical block and the card's "
            f"override block. The card's forty-five configurations would not "
            f"reproduce the shipped ones.")
        print(f"    [translation] {len(names)}/{len(names)} statistics "
              f"binaries byte-identical between the shipped block and the "
              f"card's overrides")


def _fresh_terrain_card(scratch, dem_key, latitude=None):
    """A terrain card opened from the LANDING SCREEN, not from a file.

    ``_card`` above opens ``configs/europa_lat00_agenor.yaml``, which already
    carries a ``mesh.import_file``. That is the right base for the grid
    comparison and the wrong one for this check: the defect measured on
    2026-08-28 lived on the OTHER path -- press Europa, press the terrain card,
    press Run -- where nothing had ever written that key.
    """
    manager, _existing = _card(scratch)        # shares the QApplication
    panel = _PANEL
    manager.reset_to_defaults()
    panel.load_from_config()
    start = panel.tabs["start"]
    start.choose_body("Europa")
    start._choose_mode("local")
    card = start.open_card("Europa", "local", "time_series",
                           apply_defaults=True)
    index = card.terrain_combo.findData(dem_key)
    if index < 0:
        raise Skip(f"the terrain {dem_key!r} is not offered in this checkout")
    card.terrain_combo.setCurrentIndex(index)
    if latitude is not None:
        lat_index = card.latitude_combo.findData(latitude)
        if lat_index < 0:
            raise Skip(f"latitude {latitude} is not in the shipped grid")
        card.latitude_combo.setCurrentIndex(lat_index)
    return manager, card


def test_the_card_emits_a_terrain_run_the_engine_actually_loads():
    """End to end: press the card, and see the DEM in the engine's own log.

    Three steps, because each answers a different question and the first two
    were both green on the broken configuration:

    1. the emitted file names an existing ``mesh.import_file``;
    2. ``--validate-config`` accepts it (a SYNTAX check -- it accepted the
       empty one too, which is why it is not the gate);
    3. a SHORT real run, and the engine's own banner says it imported that PLY
       with that face count. The 2026-08-28 run said
       ``mesh_source=icosphere, resolution 7`` instead, and nothing in the
       configuration, the card, or the pre-flight had said so.

    Radiation exchange is OFF and the solver is on the processor: this arm is
    about which mesh loads, and a view-factor build on a 30k-facet DEM would
    turn a 30-second check into a long one without answering anything extra.
    """
    from gui_core.simulation_runner import SimulationRunner
    exe = SimulationRunner.find_ptm_executable()
    if not exe:
        raise Skip("no ptm_thermal_sim to run")
    kernels = REPO_ROOT / "data" / "kernels"
    if not (kernels / "europa.tm").exists() or not (kernels / "lsk").exists():
        raise Skip("the core SPICE kernel set is not in this checkout")

    with tempfile.TemporaryDirectory() as scratch:
        manager, _card_widget = _fresh_terrain_card(scratch, "pwyll", 0.0)
        out = Path(scratch) / "pwyll_lat00"
        out.mkdir(parents=True, exist_ok=True)
        manager.set_value("output.directory", str(out))
        # A few steps, on the processor, radiation off.
        manager.set_value("simulation.timesteps", 8)
        manager.set_value("thermal.equilibration_orbits", 1.0)
        manager.set_value("thermal.use_gpu", False)
        manager.set_value("radiation.enabled", False)
        manager.set_value("output.vtk_mode", "none")
        config_path = out / "config.yaml"
        manager.save_run_yaml(str(config_path), REPO_ROOT)

        import yaml as _yaml
        emitted = _yaml.safe_load(config_path.read_text()) or {}
        mesh_path = str((emitted.get("mesh") or {}).get("import_file") or "")
        assert mesh_path, (
            f"{config_path}: the terrain card emitted an empty "
            f"mesh.import_file, so this would be a global icosphere")
        resolved = Path(mesh_path)
        if not resolved.is_absolute():
            resolved = REPO_ROOT / resolved
        if not resolved.exists():
            raise Skip(
                f"{resolved} is not on this disk (collaborator data, see "
                f"data/external/README.md); the emission was checked, the run "
                f"was not")

        # The launch gate, on the card's OWN radiation-on configuration.
        # Separate from the run below, which turns radiation off for speed:
        # this is the check that the card produces something the GUI would
        # actually launch, and it is where the missing-DEM and
        # missing-view-factor-cache messages come from.
        from gui_core.preflight import run_preflight
        rad_config = _yaml.safe_load(config_path.read_text()) or {}
        rad_config["radiation"] = dict(rad_config.get("radiation") or {},
                                       enabled=True)
        report = run_preflight(rad_config, lambda: str(exe),
                               project_root=REPO_ROOT)
        assert not report.errors, (
            "the card's radiation-on terrain configuration is refused at "
            "launch:\n  " + "\n  ".join(report.errors))
        vf_line = [i for i in report.info if i.startswith("view factors:")]
        if vf_line:
            assert "matches mesh" in vf_line[0], vf_line[0]
            print(f"    [terrain-preflight] {vf_line[0]}")
        elif report.vf_rebuild_warning:
            print(f"    [terrain-preflight] no view-factor cache on this "
                  f"disk; the launch offers the choice rather than refusing: "
                  f"{report.vf_rebuild_warning.splitlines()[0]}")
        else:
            # The third state: the cache is on disk but DATALESS. `report.ok`
            # is False for it, so the assertion above has already fired -- but
            # indexing the (empty) rebuild warning here would have raised an
            # unrelated IndexError instead, which on this machine's
            # cloud-synced data is a reachable path.
            print(f"    [terrain-preflight] view-factor cache present but "
                  f"dataless: {report.dataless}")

        validated = subprocess.run(
            [str(exe), "--config", str(config_path), "--validate-config"],
            capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=300)
        assert validated.returncode == 0, (
            f"--validate-config rejected the card's terrain configuration: "
            f"rc={validated.returncode}\n"
            + (validated.stdout + validated.stderr)[-1500:])

        result = subprocess.run(
            [str(exe), "--config", str(config_path)],
            capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=1800)
        log = result.stdout + result.stderr
        assert result.returncode == 0, (
            f"the terrain run failed: rc={result.returncode}\n{log[-2000:]}")

        # The engine's own record, not the log's prose: provenance.txt is
        # written by the run and is what a reader would audit afterwards.
        prov = {}
        prov_path = out / "provenance.txt"
        assert prov_path.exists(), f"{prov_path} was not written"
        for line in prov_path.read_text().splitlines():
            if "=" in line:
                key, _, value = line.partition("=")
                prov[key.strip()] = value.strip()
        assert "icosphere" not in prov.get("mesh_source", "").lower(), (
            f"the run recorded mesh_source={prov.get('mesh_source')!r} -- the "
            f"terrain card produced a procedural sphere again")
        assert resolved.name in prov.get("mesh_source", ""), (
            f"the run recorded mesh_source={prov.get('mesh_source')!r}, which "
            f"is not {resolved.name}")
        faces = int(prov.get("mesh_faces", "0") or 0)
        assert faces > 0
        print(f"    [terrain-run] {resolved.name}: mesh_source="
              f"{prov.get('mesh_source')}, mesh_faces={faces}, "
              f"mesh_length_unit={prov.get('mesh_length_unit')}, "
              f"median radius {prov.get('mesh_median_radius')}")


TESTS = [
    test_the_registry_is_read_from_the_shipped_files,
    test_the_grid_is_exactly_the_shipped_points,
    test_every_point_matches_its_shipped_configuration,
    test_the_view_factor_path_names_the_same_file,
    test_the_property_translation_is_inert,
    test_the_card_emits_a_terrain_run_the_engine_actually_loads,
]


def main():
    failures = skipped = 0
    for test in TESTS:
        try:
            test()
        except Skip as exc:
            skipped += 1
            print(f"SKIP {test.__name__}\n     {exc}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {test.__name__}\n     {exc}")
        except BaseException as exc:                        # noqa: BLE001
            failures += 1
            print(f"ERROR {test.__name__}: {type(exc).__name__}: {exc}")
        else:
            print(f"PASS {test.__name__}")
    total = len(TESTS)
    print(f"\n[test_gui_europa_grid] {total - failures - skipped}/{total} "
          f"passed, {skipped} skipped")
    print(f"[test_gui_europa_grid] {'FAIL' if failures else 'PASS'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
