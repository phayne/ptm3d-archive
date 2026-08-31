#!/usr/bin/env python3
"""The elevation-colored mesh preview, measured without a window.

``VisualizationPanel`` cannot be constructed headless on this machine -- it
takes the process down with a ``qFatal`` -- so the rendering hook that draws
this data is NOT covered here and cannot be. What IS covered is every decision
the preview makes: which reader, which elevation rule, which datum, which unit,
and when a mesh is too large to draw as a surface. ``gui_core.mesh_preview``
imports no Qt and no VTK for exactly that reason.

THE ANCHOR TEST is ``test_the_geometric_rule_reproduces_a_real_dem``: the DEM
meshes this project runs carry their own per-vertex ``elevation`` property, so
the geometric rule (radial distance minus the body's reference radius) can be
held against a number somebody else computed rather than against itself. On
``sunlit_radcore_r50.ply`` the two agree to 1e-9 m over 70,149 vertices. That
test is SKIPPED where the mesh is not on disk, and it says so loudly, because a
silently skipped anchor is not an anchor.

Run directly::

    python3 src/python/tests/test_gui_mesh_preview.py
"""

from __future__ import annotations

import struct
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "gui"))

import numpy as np                                                # noqa: E402

from gui_core.mesh_preview import (                               # noqa: E402
    BODY_RADII_M, PREVIEW_FACE_CAP, MeshPreview, elevation_field,
    mesh_preview_for_config, preview_mesh_path, read_mesh_preview)


# --------------------------------------------------------------------------
# Fixtures: geometry written here, so the checks run on a machine with no DEMs
# --------------------------------------------------------------------------

#: The real meshes, for the arms that need a file nobody in this test wrote.
#: Absolute because they live outside the repository; each arm skips with a
#: named reason when its file is absent.
REAL_LUNAR_DEM = Path(
    "/Users/paha3326/projects/moon3d/meshes/fig8_hybrid/sunlit_radcore_r50.ply")
REAL_EUROPA_DEM = Path(
    "/Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/europa/DEM/"
    "sutton_WorkMORE/Cilix_DEM_350m_SOCET.ply")
SHIPPED_CRATER = REPO_ROOT / "data" / "meshes" / "lunar_crater_80N.ply"
SHIPPED_SHAPE = (REPO_ROOT / "data" / "horizons_cache" / "bodies"
                 / "101955_Bennu_(1999_RQ36)" / "bennu_radar_shape.obj")


def _write_patch_ply(path, radius=1737400.0, relief=None, n=8,
                     with_elevation=False, spacing=100.0):
    """An ``n x n`` DEM-like patch on a sphere of `radius`, binary PLY.

    Laid out the way ``dtm_to_ply_delaunay`` lays one out: body-centered
    Cartesian, with an optional per-vertex ``elevation`` equal to
    ``|v| - radius`` -- which is what the real tool writes, measured.
    """
    relief = relief if relief is not None else (lambda i, j: 0.0)
    verts, elevs = [], []
    for j in range(n + 1):
        for i in range(n + 1):
            # A small tangent-plane offset around +z, lifted to `radius + h`.
            x = (i - n / 2.0) * spacing
            y = (j - n / 2.0) * spacing
            h = float(relief(i, j))
            r = radius + h
            z = (r ** 2 - x ** 2 - y ** 2) ** 0.5
            verts.append((x, y, z))
            elevs.append((x * x + y * y + z * z) ** 0.5 - radius)
    faces = []
    for j in range(n):
        for i in range(n):
            a = j * (n + 1) + i
            faces.append((a, a + 1, a + (n + 1)))
            faces.append((a + 1, a + n + 2, a + (n + 1)))
    props = ("property double x\nproperty double y\nproperty double z\n"
             + ("property double elevation\n" if with_elevation else ""))
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(verts)}\n{props}"
              f"element face {len(faces)}\n"
              "property list uchar int vertex_indices\n"
              "end_header\n")
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        for v, e in zip(verts, elevs):
            fh.write(struct.pack("<ddd", *v))
            if with_elevation:
                fh.write(struct.pack("<d", e))
        for f in faces:
            fh.write(struct.pack("<Biii", 3, *f))
    return len(verts), len(faces), elevs


def _write_local_patch_ply(path, n=8, spacing=10.0, height=lambda i, j: 0.0):
    """A patch in a LOCAL frame: the origin sits in the middle of the mesh."""
    verts = [((i - n / 2.0) * spacing, (j - n / 2.0) * spacing,
              float(height(i, j)))
             for j in range(n + 1) for i in range(n + 1)]
    faces = []
    for j in range(n):
        for i in range(n):
            a = j * (n + 1) + i
            faces.append((a, a + 1, a + (n + 1)))
            faces.append((a + 1, a + n + 2, a + (n + 1)))
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(verts)}\n"
              "property double x\nproperty double y\nproperty double z\n"
              f"element face {len(faces)}\n"
              "property list uchar int vertex_indices\n"
              "end_header\n")
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        for v in verts:
            fh.write(struct.pack("<ddd", *v))
        for f in faces:
            fh.write(struct.pack("<Biii", 3, *f))
    return len(verts), len(faces)


# --------------------------------------------------------------------------
# The anchor: the rule against a real DEM's own elevation
# --------------------------------------------------------------------------

def test_the_geometric_rule_reproduces_a_real_dem():
    """Radial-minus-body-radius IS what the DEM tool wrote, to 1e-9 m.

    The falsifying test for the whole elevation rule. If the rule were wrong --
    a different datum, a sign slip, the wrong radius -- this arm fails on a
    file the project runs its published lunar figures on.
    """
    if not REAL_LUNAR_DEM.exists():
        pytest.skip(f"the reference lunar DEM is not on this disk: "
                    f"{REAL_LUNAR_DEM}. This arm is the anchor for the "
                    f"elevation rule and it did NOT run.")
    preview = read_mesh_preview(REAL_LUNAR_DEM, body="Moon")
    assert not preview.refusal, preview.refusal
    assert preview.elevation_source == "file"
    assert preview.units == "meters"
    assert preview.n_vertices_total == 70149
    assert preview.n_faces_total == 140091

    radius = np.sqrt((preview.vertices ** 2).sum(axis=1))
    geometric, source, units, _datum = elevation_field(
        preview.vertices, body="Moon")
    assert source == "radius" and units == "meters"
    # The rule, held against a number this code did not compute.
    assert np.abs(geometric - preview.elevation).max() < 1e-9
    assert np.abs(radius - preview.elevation
                  - BODY_RADII_M["Moon"]).max() < 1e-9
    # And the published relief of that site, to the meter.
    low, high = preview.elevation_range()
    assert -4159.0 < low < -4157.0
    assert 2566.0 < high < 2568.0


def test_a_europa_dem_uses_its_own_datum_and_the_preview_says_so():
    """The Sutton Europa DEMs are NOT on the engine's Europa radius.

    Measured: their stored elevation is ``|v| - 1,562,090 m`` exactly (standard
    deviation 1.6e-10 m over every vertex), while the engine builds Europa at
    1560.8 km -- a 1290 m offset. The preview shows the FILE's elevation and
    states the disagreement rather than silently recomputing on a datum the
    file does not use.
    """
    if not REAL_EUROPA_DEM.exists():
        pytest.skip(f"the reference Europa DEM is not on this disk: "
                    f"{REAL_EUROPA_DEM}. This arm did NOT run.")
    preview = read_mesh_preview(REAL_EUROPA_DEM, body="Europa")
    assert not preview.refusal, preview.refusal
    assert preview.elevation_source == "file"
    radius = np.sqrt((preview.vertices ** 2).sum(axis=1))
    offsets = radius - preview.elevation
    assert offsets.std() < 1e-6
    assert abs(float(offsets.mean()) - 1562090.0) < 1.0
    # The disagreement is DISCLOSED, not smoothed over.
    assert "NOT radial distance minus the Europa reference radius" \
        in preview.datum


def test_a_shipped_ascii_ply_reads_and_lands_on_the_body_radius():
    if not SHIPPED_CRATER.exists():
        pytest.skip(f"{SHIPPED_CRATER} is not in this checkout.")
    preview = read_mesh_preview(SHIPPED_CRATER, body="Moon")
    assert not preview.refusal, preview.refusal
    assert preview.reader == "ply-ascii"
    assert preview.n_vertices_total == 6561
    assert preview.n_faces_total == 12800
    assert preview.elevation_source == "radius"
    assert preview.units == "meters"
    low, high = preview.elevation_range()
    # A 200 m bowl cut into a sphere at the Moon's own radius.
    assert -201.0 < low < -199.0
    assert -1.0 < high < 1.0


def test_a_shipped_shape_model_reads_as_kilometers_when_given_a_radius():
    if not SHIPPED_SHAPE.exists():
        pytest.skip(f"{SHIPPED_SHAPE} is not in this checkout.")
    preview = read_mesh_preview(SHIPPED_SHAPE, body="Bennu", radius_m=245.0)
    assert not preview.refusal, preview.refusal
    assert preview.reader == "obj"
    assert preview.n_vertices_total == 1348
    assert preview.n_faces_total == 2692
    assert preview.units == "kilometers"
    low, high = preview.elevation_range()
    # Bennu's radar shape is ~0.25 km mean radius with tens of meters of relief.
    assert -0.1 < low < 0.0 < high < 0.1


# --------------------------------------------------------------------------
# The rule itself, on geometry written here
# --------------------------------------------------------------------------

def test_a_stored_elevation_property_is_used_verbatim():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "patch.ply"
        n_v, n_f, elevs = _write_patch_ply(
            path, with_elevation=True,
            relief=lambda i, j: 37.0 * ((i + j) % 3))
        preview = read_mesh_preview(path, body="Moon")
    assert preview.elevation_source == "file"
    assert preview.n_vertices_total == n_v
    assert preview.n_faces_total == n_f
    assert np.abs(preview.elevation - np.array(elevs)).max() < 1e-6


def test_without_a_stored_property_the_radial_rule_gives_the_same_answer():
    """The two paths agree on identical geometry. If they did not, the
    preview would show a different surface depending on which tool wrote the
    file."""
    with tempfile.TemporaryDirectory() as tmp:
        with_prop = Path(tmp) / "with.ply"
        without = Path(tmp) / "without.ply"
        relief = lambda i, j: 21.0 * ((i * j) % 5)             # noqa: E731
        _write_patch_ply(with_prop, with_elevation=True, relief=relief)
        _write_patch_ply(without, with_elevation=False, relief=relief)
        a = read_mesh_preview(with_prop, body="Moon")
        b = read_mesh_preview(without, body="Moon")
    assert a.elevation_source == "file" and b.elevation_source == "radius"
    assert np.abs(a.elevation - b.elevation).max() < 1e-6


def test_a_local_frame_patch_uses_z_and_says_it_is_an_assumption():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "local.ply"
        _write_local_patch_ply(path, height=lambda i, j: 3.0 * i)
        preview = read_mesh_preview(path, body="Moon")
    assert preview.elevation_source == "vertical"
    low, high = preview.elevation_range()
    assert (low, high) == (0.0, 24.0)
    assert "assumption about the file" in preview.datum


def test_a_body_with_no_radius_reports_relief_about_the_mesh_mean():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "patch.ply"
        _write_patch_ply(path, relief=lambda i, j: 10.0 * i)
        preview = read_mesh_preview(path, body="Vesta")
    assert preview.elevation_source == "radius"
    assert preview.units == "the file's own units"
    assert "relief about the mesh mean" in preview.datum
    # The mean is subtracted, so the field straddles zero.
    low, high = preview.elevation_range()
    assert low < 0.0 < high


def test_a_kilometer_frame_is_detected_and_not_read_as_meters():
    """A frame slip is a factor of 1000 on the color bar. It is MEASURED."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "km.ply"
        _write_patch_ply(path, radius=1737.4, spacing=0.1,
                         relief=lambda i, j: 0.01 * i)
        preview = read_mesh_preview(path, body="Moon")
    assert preview.units == "kilometers"
    low, high = preview.elevation_range()
    assert -0.001 < low < 0.001
    assert 0.07 < high < 0.09


def test_the_body_radii_agree_with_the_facet_scale_table():
    """A third copy of the body radii is only safe if it is checked.

    ``mesh_preview`` needs the table without importing the ``ptm`` package, the
    same constraint ``facet_scale`` documents for its own copy. Two tables that
    quietly drift would put the preview's elevation datum a body radius away
    from the beaming rule's.
    """
    from gui_core.facet_scale import _BODY_RADII_M
    assert BODY_RADII_M == _BODY_RADII_M
    try:
        from ptm.results import BODY_RADII
    except ImportError:                                  # pragma: no cover
        pytest.skip("the ptm package is not importable here; only the two GUI "
                    "tables were compared.")
    for body, radius in BODY_RADII_M.items():
        if body in BODY_RADII:
            assert float(BODY_RADII[body]) == radius, body


def test_an_explicit_radius_wins_over_the_catalog():
    """A generic body has no catalog row; its card has a radius."""
    vertices = np.array([[0.0, 0.0, 1000.0], [0.0, 0.0, 1010.0]])
    elevation, source, _units, datum = elevation_field(
        vertices, body="Whatever", radius_m=1000.0)
    assert source == "radius"
    assert list(elevation) == [0.0, 10.0]
    assert "1,000" in datum


# --------------------------------------------------------------------------
# Size: the cap, and what it says
# --------------------------------------------------------------------------

def test_a_mesh_under_the_cap_is_drawn_as_a_surface():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "small.ply"
        _n_v, n_f, _ = _write_patch_ply(path, n=8)
        preview = read_mesh_preview(path, body="Moon")
    assert preview.mode == "surface"
    assert preview.n_faces_shown == n_f
    assert len(preview.faces) == n_f


def test_a_mesh_over_the_cap_becomes_points_and_discloses_it():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "big.ply"
        _n_v, n_f, _ = _write_patch_ply(path, n=8)
        # The cap is lowered rather than a 500k-face file being written: the
        # decision under test is the comparison, not the file size.
        preview = read_mesh_preview(path, body="Moon", face_cap=4)
    assert preview.mode == "points"
    assert preview.n_faces_shown == 0
    assert len(preview.faces) == 0
    assert preview.n_faces_total == n_f
    assert any("above the 4-face preview limit" in note
               for note in preview.notes)
    assert any("not a decimated mesh" in note for note in preview.notes)


def test_the_point_cap_strides_and_says_which_sample_is_drawn():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "big.ply"
        n_v, _n_f, _ = _write_patch_ply(path, n=8)
        preview = read_mesh_preview(path, body="Moon", face_cap=4,
                                    point_cap=10)
    assert preview.n_vertices_total == n_v
    assert preview.n_vertices_shown <= 10
    assert len(preview.elevation) == preview.n_vertices_shown
    assert any("of 81" in note for note in preview.notes)


def test_the_real_a3gt_core_is_capped_not_refused():
    """A 1.8M-face production core previews as points, and reads correctly."""
    core = Path("/Users/paha3326/projects/a3gt/meshes/"
                "A03CB001_83.561S_-68.169E_R3km_linear.ply")
    if not core.exists():
        pytest.skip(f"{core} is not on this disk. This arm did NOT run.")
    preview = read_mesh_preview(core, body="Moon")
    assert not preview.refusal, preview.refusal
    assert preview.n_faces_total == 1797825
    assert preview.n_faces_total > PREVIEW_FACE_CAP
    assert preview.mode == "points"
    assert preview.elevation_source == "file"


# --------------------------------------------------------------------------
# Refusals: values, never exceptions
# --------------------------------------------------------------------------

def test_a_missing_file_is_a_refusal_and_not_an_exception():
    preview = read_mesh_preview("/no/such/mesh.ply", body="Moon")
    assert isinstance(preview, MeshPreview)
    assert "is not on disk" in preview.refusal
    assert preview.vertices is None


def test_an_unsupported_extension_says_what_the_engine_still_loads():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "mesh.vtk"
        path.write_bytes(b"# vtk DataFile Version 5.1\n")
        preview = read_mesh_preview(path, body="Moon")
    assert "legacy VTK" in preview.refusal
    assert "says nothing about whether the run will load it" in preview.refusal


def test_a_quad_mesh_is_refused_rather_than_re_triangulated():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "quads.ply"
        header = ("ply\nformat binary_little_endian 1.0\n"
                  "element vertex 4\n"
                  "property double x\nproperty double y\nproperty double z\n"
                  "element face 1\n"
                  "property list uchar int vertex_indices\n"
                  "end_header\n")
        with open(path, "wb") as fh:
            fh.write(header.encode("ascii"))
            for v in ((0, 0, 1e6), (1, 0, 1e6), (1, 1, 1e6), (0, 1, 1e6)):
                fh.write(struct.pack("<ddd", *[float(c) for c in v]))
            fh.write(struct.pack("<Biiii", 4, 0, 1, 2, 3))
        preview = read_mesh_preview(path, body="Moon")
    assert "draws triangles only" in preview.refusal


def test_a_truncated_file_is_refused_by_name():
    """Truncated just short of the header's claim, past the size guard.

    The cheap size guard above catches a header that could not fit at ALL. This
    one fits the file's total size (2400 bytes of vertices in a 2450-byte file)
    and is still short once the header is deducted, which is the case only the
    read-length check can see. Both refuse; the point is that neither path
    reaches numpy with a short buffer.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "short.ply"
        header = ("ply\nformat binary_little_endian 1.0\n"
                  "element vertex 100\n"
                  "property double x\nproperty double y\nproperty double z\n"
                  "element face 1\n"
                  "property list uchar int vertex_indices\n"
                  "end_header\n").encode("ascii")
        # 100 * 24 = 2400 bytes claimed; the file is bigger than that in total
        # but has only 2300 bytes after the header.
        path.write_bytes(header + b"\x00" * 2300)
        assert path.stat().st_size > 2400
        preview = read_mesh_preview(path, body="Moon")
    assert "ends early" in preview.refusal


def test_a_face_index_outside_the_vertex_array_is_refused():
    """Not a Python error downstream -- a SEGFAULT inside VTK.

    ``pv.PolyData`` hands connectivity straight to VTK, which reads the point
    array at that offset; the try/except around the render call cannot catch
    that. The check has to be here, before the data reaches Qt at all.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bad_index.ply"
        header = ("ply\nformat binary_little_endian 1.0\n"
                  "element vertex 3\n"
                  "property double x\nproperty double y\nproperty double z\n"
                  "element face 1\n"
                  "property list uchar int vertex_indices\n"
                  "end_header\n")
        with open(path, "wb") as fh:
            fh.write(header.encode("ascii"))
            for v in ((0.0, 0.0, 1e6), (1.0, 0.0, 1e6), (1.0, 1.0, 1e6)):
                fh.write(struct.pack("<ddd", *v))
            fh.write(struct.pack("<Biii", 3, 0, 1, 99))
        preview = read_mesh_preview(path, body="Moon")
    assert "does not address its own geometry" in preview.refusal
    assert "vertex 99" in preview.refusal


def test_a_negative_face_index_is_refused_too():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "neg_index.ply"
        header = ("ply\nformat ascii 1.0\n"
                  "element vertex 3\n"
                  "property float x\nproperty float y\nproperty float z\n"
                  "element face 1\n"
                  "property list uchar int vertex_indices\n"
                  "end_header\n")
        path.write_text(header + "0 0 1000000\n1 0 1000000\n1 1 1000000\n"
                        + "3 0 1 -2\n")
        preview = read_mesh_preview(path, body="Moon")
    assert "does not address its own geometry" in preview.refusal


def test_an_absurd_header_count_is_refused_before_it_is_allocated():
    """A header is text. ``element vertex 999999999999`` asks CPython for a
    24 TB buffer before the short read can discover the file is 200 bytes."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "huge.ply"
        header = ("ply\nformat binary_little_endian 1.0\n"
                  "element vertex 999999999999\n"
                  "property double x\nproperty double y\nproperty double z\n"
                  "element face 0\n"
                  "property list uchar int vertex_indices\n"
                  "end_header\n")
        path.write_bytes(header.encode("ascii") + b"\x00" * 24)
        preview = read_mesh_preview(path, body="Moon")
    assert "cannot fit in a file this size" in preview.refusal
    assert "999,999,999,999" in preview.refusal


def test_a_file_that_is_not_a_ply_is_refused():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "notaply.ply"
        path.write_text("this is not a mesh\n")
        preview = read_mesh_preview(path, body="Moon")
    assert "PLY" in preview.refusal


# --------------------------------------------------------------------------
# What a configuration asks to preview
# --------------------------------------------------------------------------

def test_an_imported_mesh_is_what_gets_previewed():
    path, why = preview_mesh_path(
        {"mesh": {"import_file": "/abs/terrain.ply"}}, REPO_ROOT)
    assert str(path) == "/abs/terrain.ply"
    assert why == "mesh.import_file"


def test_a_shape_model_is_previewed_on_a_generic_body_card():
    path, why = preview_mesh_path(
        {"generic_body": {"shape": {"mesh_file": "/abs/shape.obj"}}},
        REPO_ROOT)
    assert str(path) == "/abs/shape.obj"
    assert why == "generic_body.shape.mesh_file"


def test_an_icosphere_run_is_told_it_has_no_topography():
    path, why = preview_mesh_path({"mesh": {"resolution": 5}}, REPO_ROOT)
    assert path is None
    assert "procedural icosphere" in why
    preview = mesh_preview_for_config({"mesh": {"resolution": 5}}, REPO_ROOT)
    assert "procedural icosphere" in preview.refusal


def test_the_occluder_ring_is_never_previewed():
    """The ring is shadow-only and is not thermally solved. Showing it beside
    the core would draw terrain the run does not solve on."""
    path, why = preview_mesh_path(
        {"mesh": {"occluder_file": "/abs/ring.ply", "resolution": 5}},
        REPO_ROOT)
    assert path is None
    assert "procedural icosphere" in why


def test_a_relative_mesh_path_resolves_against_the_repository_not_the_cwd():
    path, _why = preview_mesh_path(
        {"mesh": {"import_file": "data/meshes/lunar_crater_80N.ply"}},
        REPO_ROOT)
    assert path == REPO_ROOT / "data" / "meshes" / "lunar_crater_80N.ply"


def test_a_generic_body_radius_becomes_the_datum():
    if not SHIPPED_SHAPE.exists():
        pytest.skip(f"{SHIPPED_SHAPE} is not in this checkout.")
    config = {
        "simulation": {"body": "Bennu"},
        "generic_body": {"shape": {"mesh_file": str(SHIPPED_SHAPE),
                                   "radius_km": 0.245}},
    }
    preview = mesh_preview_for_config(config, REPO_ROOT)
    assert not preview.refusal, preview.refusal
    assert preview.units == "kilometers"
    assert "0.2450 km" in preview.datum


if __name__ == "__main__":                                   # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
