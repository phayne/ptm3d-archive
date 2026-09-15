#!/usr/bin/env python3
"""The GUI's "Export Data" has to be able to write a Cloud Optimized GeoTIFF,
and the raster it writes has to be georeferenced and honest about its nodata.

⛔⛔ THE DEFECT THIS EXISTS TO STOP -- MEASURED, NOT INFERRED.

A raster has ONE nodata value. PTM3D writes TWO sentinels that do not mean the
same thing:

* ``NaN``  -- this facet was not evaluated / carries no value.
* ``+inf`` -- the Sun was still not visible at the top of the shadow-depth
  search ceiling. That is a MEASUREMENT about the Sun, not an absence.

Both regrid paths in ``ptm.results.MeshRegridder`` destroy the second one, in
opposite directions, and both were measured on a 12,800-facet mesh with 800
``+inf`` facets:

* ``regrid`` (inverse-distance) zeroes the weight of every NON-FINITE neighbor,
  so a ``+inf`` facet is dropped and its pixel is filled from the finite facets
  around it. **All 58 of the pixels whose facets were majority-unresolved came
  back holding a finite 2.0 m depth.** "The Sun is never visible below the
  ceiling" was published as a depth in meters.
* ``regrid_linear`` (barycentric, the default for COGs) multiplies ``+inf`` by a
  barycentric weight, so it SPREADS: **194 pixels went infinite where only 58
  facets were unresolved**, erasing 136 finite measurements.

``ptm_cog.export_field`` therefore separates the two classes before it
interpolates anything, stamps the unresolved region out of the main band, and
writes the ``+inf`` population beside it as its own layer. Section 3 proves the
leak is real on the raw regridder and closed in the raster.

Section 4 reads the written raster back with **rasterio**, whose bundled GDAL is
a different build from the ``osgeo.gdal`` the writer uses, and checks the
georeferencing against a forward polar-stereographic projection written out
longhand in this file -- not against ``ptm.results._forward_polar_stereo``,
which is the code under test.

⛔ Every fixture here is synthetic or the repository's own shipped crater mesh.
No site identifier and no real site coordinate appears in this file, under the
2026-09-03 standing ruling.

Run directly::

    python3 src/python/tests/test_gui_cog_export.py
"""

from __future__ import annotations

import ast
import math
import os
import struct
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "python"))
sys.path.insert(0, str(REPO_ROOT / "gui"))

PANEL_SOURCE = REPO_ROOT / "gui" / "gui_panels" / "visualization_panel.py"
EXPORTER_SOURCE = REPO_ROOT / "gui" / "gui_core" / "data_exporter.py"

#: Lunar radius, in meters. The synthetic fixtures are built on a sphere of
#: this radius so the projection is the south-polar stereographic branch a real
#: polar run takes. It is a body constant, not a site.
BODY_RADIUS_M = 1737400.0

#: Where the synthetic patch sits. INVENTED round numbers, chosen to be south
#: of the 70 deg polar threshold and away from the pole itself so the
#: site-centered meridian is a real number rather than a wrap artifact. Under
#: the 2026-09-03 standing ruling no real landing-site coordinate may appear in
#: the tracked tree, and this file is a coordinate test, so this pair is checked
#: against the local sensitive-identifier list by
#: ``test_tracked_tree_hygiene.py`` like every other number here.
FIXTURE_LAT_DEG = -80.0
FIXTURE_LON_DEG = 20.0

#: The one QApplication, held so it is not collected mid-run.
_QAPP = None


class Skip(Exception):
    """A check that cannot run here, as distinct from one that failed."""


FAILURES: list[str] = []
SKIPS: list[str] = []
PASSES: list[str] = []


def check(name, fn, *args):
    try:
        detail = fn(*args)
    except Skip as exc:
        SKIPS.append(f"{name}: {exc}")
        print(f"SKIP {name} :: {exc}", flush=True)
    except AssertionError as exc:
        FAILURES.append(f"{name}: {exc}")
        print(f"FAIL {name} :: {exc}", flush=True)
    except Exception as exc:                                    # noqa: BLE001
        FAILURES.append(f"{name}: {type(exc).__name__}: {exc}")
        print(f"FAIL {name} :: {type(exc).__name__}: {exc}", flush=True)
    else:
        PASSES.append(name)
        print(f"PASS {name} :: {detail or ''}", flush=True)


# ---------------------------------------------------------------------------
# Section 1 -- the dialog offers it, over the source. Needs nothing installed.
# ---------------------------------------------------------------------------

def _tree(path):
    return ast.parse(path.read_text(encoding="utf-8"))


def _class(path, name):
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"class {name} is gone from {path.name}")


def _source_of(path, node):
    """The source text of one AST node.

    ``ast.get_source_segment``, not ``ast.unparse``: the environment that has
    PyQt6 and rasterio on this machine is Python 3.8, where ``unparse`` does
    not exist. A gate that raises AttributeError is not a gate.
    """
    return ast.get_source_segment(path.read_text(encoding="utf-8"), node) or ""


def _methods(cls):
    return {n.name: n for n in cls.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def check_the_export_dialog_offers_a_cog_button():
    """A format the worker understands but the dialog never shows is not a
    feature. This reads the dialog source, so it runs with no Qt."""
    cls = _class(PANEL_SOURCE, "ExportDataDialog")
    src = _source_of(PANEL_SOURCE, cls)
    assert "cog_radio" in src, (
        "ExportDataDialog has no COG radio button, so 'Export Data' cannot "
        "reach the COG writer")
    assert "FORMAT_ID_COG" in src, (
        "the COG radio button has no named format id; adding a format by bare "
        "integer is how the four existing branches drifted apart")
    start = _methods(cls).get("_start_export")
    assert start is not None, "ExportDataDialog._start_export is gone"
    body = _source_of(PANEL_SOURCE, start)
    assert "'cog'" in body or '"cog"' in body, (
        "_start_export never puts 'cog' in the formats list, so the button "
        "would be inert")
    return "COG radio, named id and formats entry all present"


def check_a_requested_cog_that_cannot_be_written_is_refused_out_loud():
    """The worker must refuse, not quietly write four of the five products.

    Every silent override, fallback or skip in this project has had to be paid
    for later. This one is stated as a rule over the source so it survives a
    rewrite of the branch.
    """
    cls = _class(EXPORTER_SOURCE, "DataExportWorker")
    body = _source_of(EXPORTER_SOURCE, _methods(cls)["_do_export"])
    assert "COG_UNAVAILABLE_REASON" in body, (
        "DataExportWorker._do_export does not name COG_UNAVAILABLE_REASON, so "
        "an unavailable COG writer would be skipped without saying why")
    assert "'cog' in formats and not do_cog" in body.replace('"', "'"), (
        "there is no branch that catches 'COG was asked for and cannot be "
        "produced'")
    return "an unavailable COG writer is reported, not skipped"


def check_the_offered_statistics_come_from_the_shared_registry():
    """No fourth field catalog. The COG export must enumerate exactly what
    ``ptm.results.detect_available_statistics`` reports for the run."""
    src = EXPORTER_SOURCE.read_text(encoding="utf-8")
    assert "_detect_stats_in_dir" in src, (
        "gui_core.data_exporter no longer delegates statistics discovery to "
        "ptm.results")
    panel = PANEL_SOURCE.read_text(encoding="utf-8")
    assert "detect_available_statistics" in panel, (
        "the export dialog no longer asks for the run's statistics; a "
        "hand-written list here would be a fourth field catalog")
    assert "COG_FIELDS" not in src and "COG_FIELDS" not in panel, (
        "a COG-specific field catalog has appeared; there are already three "
        "hand-editable ones and this must not become the fourth")
    return "statistics enumeration still comes from ptm.results"


# ---------------------------------------------------------------------------
# Section 2 -- fixtures
# ---------------------------------------------------------------------------

def _synthetic_run(root, n=60, half=1500.0, unresolved_every=37,
                   nan_every=53):
    """A results directory with two statistics maps, one carrying both
    sentinels.

    A square patch of terrain at lunar radius, so the regridder takes the
    south-polar stereographic branch. Synthetic: no site, no real coordinate.
    Returns ``(mesh_data, root, n_faces)``.
    """
    import numpy as np

    root = Path(root)
    (root / "statistics").mkdir(parents=True, exist_ok=True)
    u = np.linspace(-half, half, n)
    X, Y = np.meshgrid(u, u)
    elev = 120.0 * np.sin(X / 700.0) * np.cos(Y / 900.0)

    # Local east/north/up at the synthetic center, rotated into the
    # body-centered frame -- so the patch is a real piece of a sphere and its
    # latitudes and longitudes are a narrow band, as a site mesh's are.
    lat0 = np.radians(FIXTURE_LAT_DEG)
    lon0 = np.radians(FIXTURE_LON_DEG)
    east = np.array([-np.sin(lon0), np.cos(lon0), 0.0])
    north = np.array([-np.sin(lat0) * np.cos(lon0),
                      -np.sin(lat0) * np.sin(lon0), np.cos(lat0)])
    up = np.array([np.cos(lat0) * np.cos(lon0),
                   np.cos(lat0) * np.sin(lon0), np.sin(lat0)])
    verts = (up * BODY_RADIUS_M
             + X.ravel()[:, None] * east
             + Y.ravel()[:, None] * north
             + elev.ravel()[:, None] * up).astype(np.float64)

    tris = []
    for j in range(n - 1):
        for i in range(n - 1):
            a = j * n + i
            tris.append((a, a + n, a + 1))
            tris.append((a + 1, a + n, a + n + 1))
    tris = np.asarray(tris, dtype=np.int32)

    cen = verts[tris].mean(axis=1)
    rad = np.linalg.norm(cen, axis=1)
    lat = np.degrees(np.arcsin(cen[:, 2] / rad)).astype(np.float32)
    lon = np.degrees(np.arctan2(cen[:, 1], cen[:, 0])).astype(np.float32)
    nf = len(tris)

    temps = (200.0 + 60.0 * np.linspace(0.0, 1.0, nf)).astype("<f4")
    _write_stats(root / "statistics" / "temperature_max.bin", temps)

    shadow = np.full(nf, 2.0, dtype="<f4")
    shadow[::unresolved_every] = np.inf       # umbral above the ceiling
    shadow[1::nan_every] = np.nan             # not evaluated
    _write_stats(root / "statistics" / "shadow_depth_max.bin", shadow)

    mesh_data = {"vertices": verts, "faces": tris,
                 "latitudes": lat, "longitudes": lon}
    return mesh_data, root, nf


def _synthetic_equatorial_run(root, n=30, half=40000.0):
    """The same terrain at the equator, so `choose_projection` returns 'eqc'."""
    import numpy as np

    root = Path(root)
    (root / "statistics").mkdir(parents=True, exist_ok=True)
    u = np.linspace(-half, half, n)
    X, Y = np.meshgrid(u, u)
    lat0 = 0.0
    lon0 = math.radians(FIXTURE_LON_DEG)
    east = np.array([-math.sin(lon0), math.cos(lon0), 0.0])
    north = np.array([0.0, 0.0, 1.0])
    up = np.array([math.cos(lon0), math.sin(lon0), 0.0])
    verts = (up * BODY_RADIUS_M
             + X.ravel()[:, None] * east
             + Y.ravel()[:, None] * north).astype(np.float64)
    tris = []
    for j in range(n - 1):
        for i in range(n - 1):
            a = j * n + i
            tris.append((a, a + n, a + 1))
            tris.append((a + 1, a + n, a + n + 1))
    tris = np.asarray(tris, dtype=np.int32)
    cen = verts[tris].mean(axis=1)
    rad = np.linalg.norm(cen, axis=1)
    lat = np.degrees(np.arcsin(cen[:, 2] / rad)).astype(np.float32)
    lon = np.degrees(np.arctan2(cen[:, 1], cen[:, 0])).astype(np.float32)
    nf = len(tris)
    _write_stats(root / "statistics" / "temperature_max.bin",
                 (200.0 + np.linspace(0.0, 60.0, nf)).astype("<f4"))
    return {"vertices": verts, "faces": tris,
            "latitudes": lat, "longitudes": lon}, root, nf


def _write_stats(path, values):
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(values)))
        f.write(values.astype("<f4").tobytes())


def _need(module):
    try:
        return __import__(module)
    except Exception as exc:                                    # noqa: BLE001
        raise Skip(f"{module} is not importable here ({exc})")


def _qapp():
    """A Qt application, because DataExportWorker is a QThread."""
    global _QAPP
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        import PyQt6
        from PyQt6.QtWidgets import QApplication
    except Exception as exc:                                    # noqa: BLE001
        raise Skip(f"PyQt6 is not importable here ({exc})")
    # ⛔ Without these Qt does not raise when it cannot find its offscreen
    # platform plugin: it calls abort(), which takes the buffered stdout with
    # it and reports NOTHING at all.
    plug = Path(PyQt6.__file__).parent / "Qt6" / "plugins"
    os.environ.setdefault("QT_PLUGIN_PATH", str(plug))
    os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(plug / "platforms"))
    if QApplication.instance() is None:
        _QAPP = QApplication([])
    else:
        _QAPP = QApplication.instance()
    return _QAPP


def _run_cog_export(tmp, center_lon, resolution=6.0e-5):
    """Drive the real ``DataExportWorker`` and return its output directory."""
    _need("numpy")
    _need("scipy")
    _need("rasterio")
    _qapp()
    try:
        from gui_core.data_exporter import DataExportWorker, COG_AVAILABLE, \
            COG_UNAVAILABLE_REASON
    except Exception as exc:                                    # noqa: BLE001
        raise Skip(f"gui_core.data_exporter is not importable ({exc})")
    if not COG_AVAILABLE:
        raise Skip(f"the COG writer is unavailable here "
                   f"({COG_UNAVAILABLE_REASON})")

    mesh_data, root, _nf = _synthetic_run(Path(tmp) / "run")
    out = Path(tmp) / f"out_{'shared' if center_lon == 0.0 else 'site'}"
    worker = DataExportWorker({
        "results_dir": str(root),
        "mesh_data": mesh_data,
        "timesteps": [],
        "fields": [],
        "formats": ["cog"],
        "resolution": resolution,
        "output_dir": str(out),
        "geotiff_per_timestep": True,
        "body_radius": BODY_RADIUS_M,
        "statistics": [("temperature", "max"), ("shadow_depth", "max")],
        "center_lon": center_lon,
    })
    finished = []
    worker.export_finished.connect(lambda ok, msg: finished.append((ok, msg)))
    worker._do_export()
    assert finished and finished[0][0], (
        f"the export did not succeed: {finished[0][1] if finished else 'no signal'}")
    return mesh_data, root, out


# ---------------------------------------------------------------------------
# Section 3 -- the sentinels
# ---------------------------------------------------------------------------

def check_the_raw_regridder_really_does_lose_the_infinite_sentinel():
    """The defect, on the shared regridder, so the guard is not guarding air.

    If this ever starts failing because ``MeshRegridder`` learned to handle the
    sentinel itself, the stamping in ``ptm_cog.export_field`` becomes
    redundant -- but it must be RETIRED deliberately, not left passing by
    accident.
    """
    np = _need("numpy")
    _need("scipy")
    from ptm.results import MeshRegridder

    mesh_data, _root, nf = _synthetic_run(tempfile.mkdtemp(prefix="_leak_"))
    rg = MeshRegridder(mesh_data["latitudes"], mesh_data["longitudes"],
                       resolution=1.5e-4, body_radius=BODY_RADIUS_M)
    values = np.full(nf, 2.0)
    unresolved = np.zeros(nf, bool)
    unresolved[::17] = True
    values[unresolved] = np.inf

    indicator = rg.regrid_linear(unresolved.astype(float))
    hot = np.isfinite(indicator) & (indicator >= 0.5)
    assert hot.sum() > 0, "the fixture produced no majority-unresolved pixel"

    idw = rg.regrid(values.copy())
    leaked = int((hot & np.isfinite(idw)).sum())
    assert leaked > 0, (
        "MeshRegridder.regrid no longer fills an unresolved pixel from its "
        "finite neighbors -- re-aim or retire the stamping in "
        "ptm_cog.export_field rather than leaving this check passing")

    linear = rg.regrid_linear(values.copy())
    smeared = int(np.isposinf(linear).sum())
    assert smeared > int(hot.sum()), (
        "MeshRegridder.regrid_linear no longer spreads +inf beyond the facets "
        "that carry it -- same instruction as above")
    return (f"IDW reports a finite depth at {leaked}/{int(hot.sum())} "
            f"unresolved pixels; barycentric spreads +inf to {smeared}")


def check_an_unresolved_region_never_reads_as_a_finite_depth(tmp):
    """The main band must be nodata wherever the facets were unresolved."""
    np = _need("numpy")
    rasterio = _need("rasterio")
    _mesh, _root, out = _run_cog_export(tmp, center_lon=0.0)

    main = out / "cog" / "statistics" / "shadow_depth_max.tif"
    companion = out / "cog" / "statistics" / "shadow_depth_max_unresolved.tif"
    assert main.is_file(), f"{main} was not written"
    assert companion.is_file(), (
        f"{companion} was not written, so the +inf population was conflated "
        f"with 'no value' and is unrecoverable from the raster")

    with rasterio.open(main) as a, rasterio.open(companion) as b:
        A = a.read(1, masked=True)
        B = b.read(1, masked=True)
        tags = a.tags(1)
        assert a.crs == b.crs and a.transform == b.transform, (
            "the companion layer is not on the same grid as the band it "
            "explains, so it cannot be laid over it")
    hot = B.filled(0.0) >= 0.5
    assert int(hot.sum()) > 0, "no unresolved pixel in the fixture"
    leaked = int((hot & ~A.mask).sum())
    assert leaked == 0, (
        f"{leaked} pixels hold a finite shadow depth where the facets behind "
        f"them were umbral above the search ceiling")
    assert int(tags["PTM_FACES_POS_INF"]) > 0, (
        "the band does not record how many facets carried +inf")
    assert int(tags["PTM_FACES_NAN"]) > 0, (
        "the band does not record how many facets carried NaN")
    assert "umbral" in tags["PTM_SENTINEL_POLICY"], (
        "the band does not state which sentinel rule was applied")
    return (f"{int(hot.sum())} unresolved pixels, 0 finite; "
            f"+inf facets {tags['PTM_FACES_POS_INF']}, "
            f"NaN facets {tags['PTM_FACES_NAN']}")


def check_a_field_with_no_sentinel_says_so_rather_than_nothing(tmp):
    np = _need("numpy")
    rasterio = _need("rasterio")
    _mesh, _root, out = _run_cog_export(tmp, center_lon=0.0)
    with rasterio.open(out / "cog" / "statistics" / "temperature_max.tif") as ds:
        tags = ds.tags(1)
    assert tags["PTM_FACES_POS_INF"] == "0", "temperature carries no +inf"
    assert "no infinite sentinel" in tags["PTM_SENTINEL_POLICY"], (
        "a field without an infinite sentinel must still state the rule that "
        "was applied to it")
    assert not (out / "cog" / "statistics"
                / "temperature_max_unresolved.tif").exists(), (
        "a companion layer was written for a field with no +inf population")
    return "census tags present, no spurious companion layer"


def check_an_all_unresolved_field_is_exported_not_dropped(tmp):
    """A field that is +inf everywhere is fully MEASURED, not empty.

    "The Sun was never visible below the ceiling anywhere on this mesh" is a
    result. Dropping it for having no finite value writes neither the band nor
    the companion, and the measurement cannot be recovered from the export.
    """
    np = _need("numpy")
    rasterio = _need("rasterio")
    _need("scipy")
    _qapp()
    try:
        from gui_core.data_exporter import DataExportWorker, COG_AVAILABLE, \
            COG_UNAVAILABLE_REASON
    except Exception as exc:                                    # noqa: BLE001
        raise Skip(f"gui_core.data_exporter is not importable ({exc})")
    if not COG_AVAILABLE:
        raise Skip(f"the COG writer is unavailable here "
                   f"({COG_UNAVAILABLE_REASON})")

    mesh_data, root, nf = _synthetic_run(Path(tmp) / "allinf")
    _write_stats(root / "statistics" / "shadow_depth_max.bin",
                 np.full(nf, np.inf, dtype="<f4"))
    out = Path(tmp) / "out_allinf"
    worker = DataExportWorker({
        "results_dir": str(root), "mesh_data": mesh_data, "timesteps": [],
        "fields": [], "formats": ["cog"], "resolution": 3.0e-4,
        "output_dir": str(out), "geotiff_per_timestep": True,
        "body_radius": BODY_RADIUS_M,
        "statistics": [("shadow_depth", "max")], "center_lon": 0.0,
    })
    messages = []
    worker.status_updated.connect(messages.append)
    finished = []
    worker.export_finished.connect(lambda ok, msg: finished.append((ok, msg)))
    worker._do_export()
    assert finished and finished[0][0], f"export failed: {finished}"

    companion = out / "cog" / "statistics" / "shadow_depth_max_unresolved.tif"
    assert companion.is_file(), (
        f"a field that is +inf on every facet produced no companion layer. "
        f"Worker said: {messages}")
    with rasterio.open(companion) as ds:
        mask = ds.read(1, masked=True)
    assert float(mask.max()) >= 0.99, (
        "the companion layer does not record the unresolved population")
    return f"all-+inf field exported; companion max {float(mask.max()):.3f}"


def check_negative_infinity_is_not_labeled_a_solar_measurement(tmp):
    """`-inf` has no meaning in any PTM3D product. It must be counted and
    dropped, never written into the layer that says the Sun was hidden."""
    np = _need("numpy")
    rasterio = _need("rasterio")
    _need("scipy")
    from ptm.results import MeshRegridder
    import ptm_cog

    mesh_data, root, nf = _synthetic_run(Path(tmp) / "neginf")
    rg = MeshRegridder(mesh_data["latitudes"], mesh_data["longitudes"],
                       resolution=3.0e-4, body_radius=BODY_RADIUS_M,
                       center_lon=0.0)
    values = np.full(nf, 2.0)
    values[::29] = -np.inf
    out = Path(tmp) / "neginf_out" / "field.tif"
    ptm_cog.export_field(out, values, rg, "shadow_depth_max",
                         interpolation="linear")
    with rasterio.open(out) as ds:
        tags = ds.tags(1)
    assert tags["PTM_FACES_NEG_INF"] == str(int(np.isneginf(values).sum())), (
        "the band does not count the -inf population")
    assert not Path(ptm_cog.unresolved_mask_path(out)).exists(), (
        "-inf was written into the 'umbral above the search ceiling' layer, "
        "which relabels an underflow as a statement about solar visibility")
    assert "-inf" in tags["PTM_SENTINEL_POLICY"], (
        "the band does not say what happened to the -inf population")
    return f"{tags['PTM_FACES_NEG_INF']} -inf facets counted, not relabeled"


def check_the_shared_crs_control_is_inert_where_it_would_lie(tmp):
    """⛔ The control only reaches the raster on the polar stereographic
    branch.

    `MeshRegridder` carries `center_lon` into `_init_stereo`; `_init_eqc` builds
    the grid from the raw longitude bounds and both raster writers recompute the
    equirectangular central meridian from those bounds. A checkbox that changes
    the README and the PROJ string without moving a pixel is worse than no
    checkbox.

    ⛔ Asserted with `click()`, not `setChecked`: `setChecked` is blind to a
    DISABLED widget and would report success on a control the user cannot
    operate.
    """
    _need("numpy")
    _need("scipy")
    _qapp()
    try:
        import gui_panels.visualization_panel as panel
    except Exception as exc:                                    # noqa: BLE001
        raise Skip(f"the visualization panel is not importable ({exc})")

    equatorial, root, _nf = _synthetic_equatorial_run(Path(tmp) / "eqc")
    dialog = panel.ExportDataDialog(str(root), equatorial, [])
    assert dialog.projection == "eqc", (
        f"the equatorial fixture did not take the equirectangular branch "
        f"({dialog.projection}); this check would be vacuous")
    assert not dialog.shared_crs_check.isEnabled(), (
        "the shared-meridian control is offered on an equirectangular export, "
        "where it changes the stated CRS without moving a pixel")
    dialog.shared_crs_check.click()
    assert dialog._center_lon_choice() is None, (
        "clicking the shared-meridian control on an equirectangular export "
        "forced a central meridian the raster writers ignore")

    polar, proot, _n2 = _synthetic_run(Path(tmp) / "polar_dialog")
    pdialog = panel.ExportDataDialog(str(proot), polar, [])
    assert pdialog.projection == "stere", "the polar fixture is not polar"
    assert pdialog.shared_crs_check.isEnabled(), (
        "the shared-meridian control is disabled on a polar export, where it "
        "is the one place it works")
    pdialog.shared_crs_check.click()
    assert pdialog._center_lon_choice() == 0.0, (
        "clicking the control on a polar export did not force meridian 0")
    return "disabled on eqc, live on stere, checked through click()"


# ---------------------------------------------------------------------------
# Section 4 -- the raster, read back by an independent reader
# ---------------------------------------------------------------------------

def _forward_south_polar_stereo(lat_deg, lon_deg, center_lon_deg, radius):
    """Forward south-polar stereographic, written out here on purpose.

    NOT ``ptm.results._forward_polar_stereo``: that is the code the raster was
    built with, and checking a projection against itself proves nothing.
    """
    import math
    rho = 2.0 * radius * math.tan(math.pi / 4.0 + math.radians(lat_deg) / 2.0)
    dlon = math.radians(lon_deg - center_lon_deg)
    return rho * math.sin(dlon), rho * math.cos(dlon)


def check_the_raster_is_georeferenced_where_the_facet_is(tmp):
    """A known facet's value has to land in the pixel its coordinates project
    to, under a projection this file computes independently."""
    np = _need("numpy")
    rasterio = _need("rasterio")
    from rasterio.transform import rowcol

    mesh, root, out = _run_cog_export(tmp, center_lon=0.0)
    with open(root / "statistics" / "temperature_max.bin", "rb") as f:
        n = struct.unpack("<I", f.read(4))[0]
        values = np.frombuffer(f.read(4 * n), dtype="<f4")
    hottest = int(np.argmax(values))
    lat = float(mesh["latitudes"][hottest])
    lon = float(mesh["longitudes"][hottest])
    x, y = _forward_south_polar_stereo(lat, lon, 0.0, BODY_RADIUS_M)

    with rasterio.open(out / "cog" / "statistics" / "temperature_max.tif") as ds:
        row, col = rowcol(ds.transform, x, y)
        assert 0 <= row < ds.height and 0 <= col < ds.width, (
            f"the hottest facet projects to ({row}, {col}), outside the "
            f"{ds.height}x{ds.width} raster")
        band = ds.read(1, masked=True)
        wkt = ds.crs.to_wkt()
        nodata = ds.nodata
    pixel = band[row, col]
    assert not np.ma.is_masked(pixel), (
        "the pixel the hottest facet projects to is nodata")
    assert abs(float(pixel) - float(values[hottest])) < 0.05, (
        f"the pixel at the hottest facet holds {float(pixel):.4f} K, the facet "
        f"holds {float(values[hottest]):.4f} K")
    assert "Polar_Stereographic" in wkt, (
        f"a polar mesh was not written in a polar stereographic CRS: {wkt[:80]}")
    assert f"{int(BODY_RADIUS_M)}" in wkt, (
        f"the CRS does not carry the body radius that was exported with: "
        f"{wkt[:160]}")
    assert nodata is not None and nodata < -1e38, (
        f"nodata did not round-trip as the float32 sentinel: {nodata}")
    return (f"facet {hottest} -> pixel ({row}, {col}) = {float(pixel):.4f} K "
            f"vs {float(values[hottest]):.4f} K; nodata {nodata:.7e}")


def check_the_shared_meridian_choice_reaches_the_crs(tmp):
    """The one projection choice the dialog exposes has to be real."""
    rasterio = _need("rasterio")
    _m, _r, shared = _run_cog_export(tmp, center_lon=0.0)
    _m2, _r2, site = _run_cog_export(tmp, center_lon=None)

    def meridian(out):
        with rasterio.open(out / "cog" / "statistics"
                           / "temperature_max.tif") as ds:
            wkt = ds.crs.to_wkt()
        marker = '"central_meridian",'
        i = wkt.index(marker) + len(marker)
        return float(wkt[i:wkt.index("]", i)])

    shared_cm = meridian(shared)
    site_cm = meridian(site)
    assert abs(shared_cm) < 1e-9, (
        f"'Shared CRS' produced central meridian {shared_cm}, not 0")
    assert abs(site_cm) > 1e-6, (
        f"the site-centered option produced central meridian {site_cm}, which "
        f"is indistinguishable from the shared one -- the checkbox is inert")
    return f"shared {shared_cm:g} deg vs site-centered {site_cm:.4f} deg"


def check_the_export_writes_a_readme_that_states_the_sentinel_rule(tmp):
    """A GeoTIFF outlives the directory it came from, and so does the question
    "what does nodata mean here"."""
    _m, _r, out = _run_cog_export(tmp, center_lon=0.0)
    readme = (out / "README.txt").read_text(encoding="utf-8")
    for needed in ("NoData", "+inf", "unresolved", "PTM_FACES_POS_INF"):
        assert needed in readme, (
            f"the export README never mentions {needed!r}, so a reader of the "
            f"rasters cannot tell an absent measurement from a measured "
            f"absence")
    return "README states the nodata value and the sentinel rule"


# ---------------------------------------------------------------------------

def main():
    tmp = tempfile.mkdtemp(prefix="ptm_cog_gate_")
    check("test_the_export_dialog_offers_a_cog_button",
          check_the_export_dialog_offers_a_cog_button)
    check("test_a_requested_cog_that_cannot_be_written_is_refused_out_loud",
          check_a_requested_cog_that_cannot_be_written_is_refused_out_loud)
    check("test_the_offered_statistics_come_from_the_shared_registry",
          check_the_offered_statistics_come_from_the_shared_registry)
    check("test_the_raw_regridder_really_does_lose_the_infinite_sentinel",
          check_the_raw_regridder_really_does_lose_the_infinite_sentinel)
    check("test_an_unresolved_region_never_reads_as_a_finite_depth",
          check_an_unresolved_region_never_reads_as_a_finite_depth, tmp)
    check("test_a_field_with_no_sentinel_says_so_rather_than_nothing",
          check_a_field_with_no_sentinel_says_so_rather_than_nothing, tmp)
    check("test_an_all_unresolved_field_is_exported_not_dropped",
          check_an_all_unresolved_field_is_exported_not_dropped, tmp)
    check("test_negative_infinity_is_not_labeled_a_solar_measurement",
          check_negative_infinity_is_not_labeled_a_solar_measurement, tmp)
    check("test_the_shared_crs_control_is_inert_where_it_would_lie",
          check_the_shared_crs_control_is_inert_where_it_would_lie, tmp)
    check("test_the_raster_is_georeferenced_where_the_facet_is",
          check_the_raster_is_georeferenced_where_the_facet_is, tmp)
    check("test_the_shared_meridian_choice_reaches_the_crs",
          check_the_shared_meridian_choice_reaches_the_crs, tmp)
    check("test_the_export_writes_a_readme_that_states_the_sentinel_rule",
          check_the_export_writes_a_readme_that_states_the_sentinel_rule, tmp)

    total = len(PASSES) + len(FAILURES) + len(SKIPS)
    print(f"\n[test_gui_cog_export] {len(PASSES)}/{total} passed"
          + (f", {len(SKIPS)} skipped" if SKIPS else "")
          + (f", {len(FAILURES)} failed" if FAILURES else ""), flush=True)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
