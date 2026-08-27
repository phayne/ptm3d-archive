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
   and its nine siblings are read by nothing (``BUGS.md`` PTM-67) -- so the
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


def _card(scratch):
    global _APP, _CARD, _MANAGER
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
    panel = ConfigurationPanel(_MANAGER)
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
    the source. It uses the procedural icosphere, because the Sutton elevation
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


TESTS = [
    test_the_registry_is_read_from_the_shipped_files,
    test_the_grid_is_exactly_the_shipped_points,
    test_every_point_matches_its_shipped_configuration,
    test_the_view_factor_path_names_the_same_file,
    test_the_property_translation_is_inert,
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
