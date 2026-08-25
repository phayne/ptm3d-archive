#!/usr/bin/env python3
"""The GUI's facet-scale rule for `radiation.beaming` (DP-4).

The engine enforces exactly one thing about beaming -- ``beaming: true``
requires ``rms_slope_deg > 0`` (``ptm_thermal_sim.cpp:13592``) -- and nothing at
all about the scale of the mesh it is applied to. A mesh whose facets are
already AT the roughness scale resolves the roughness geometrically, so a
sub-facet model on top counts it twice, and no code anywhere refuses that. The
GUI is therefore the only guard, and this file is what keeps it honest.
``BUGS.md`` ENG-BEAM-SCALE-1 records that a CLI user still has none.

Boundaries under test (``gui_core.facet_scale``): refuse below 0.05 m, warn
between 0.05 m and 1 m, allow at or above 1 m, lunar bodies only.

Geometry is generated here as small PLY files rather than read from the real
meshes: the checks have to run on a machine that has none of them. The real
meshes were measured separately and agreed with
``docs/radiation-exchange.md`` section 8.4 to 0.1% (``sunlit_radcore_r50``,
236.6 m against 237 m) and 0.6% (A3GT ``A03MP026`` 3 km core, 3.479 m against
3.46 m, sampled).

No PyQt6 needed: ``gui_core.facet_scale`` imports no Qt.

Run directly::

    python3 src/python/tests/test_gui_beaming_scale_rule.py
"""

from __future__ import annotations

import math
import struct
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "gui"))

from gui_core.facet_scale import (                                # noqa: E402
    BEAMING_BODIES, FACET_SCALE_DISABLE_BELOW_M, FACET_SCALE_WARN_BELOW_M,
    beaming_policy, facet_scale)


def _write_grid_ply(path, edge_m, n=8, binary=True):
    """An n x n grid of right triangles with legs `edge_m`.

    Mean triangle area is edge^2 / 2, so the facet scale is edge / sqrt(2).
    """
    verts = [(i * edge_m, j * edge_m, 0.0)
             for j in range(n + 1) for i in range(n + 1)]
    faces = []
    for j in range(n):
        for i in range(n):
            a = j * (n + 1) + i
            b = a + 1
            c = a + (n + 1)
            d = c + 1
            faces.append((a, b, c))
            faces.append((b, d, c))
    fmt = "binary_little_endian 1.0" if binary else "ascii 1.0"
    header = (f"ply\nformat {fmt}\n"
              f"element vertex {len(verts)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              f"element face {len(faces)}\n"
              "property list uchar int vertex_indices\n"
              "end_header\n")
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        if binary:
            for v in verts:
                fh.write(struct.pack("<fff", *v))
            for f in faces:
                fh.write(struct.pack("<Biii", 3, *f))
        else:
            body = "".join(f"{v[0]} {v[1]} {v[2]}\n" for v in verts)
            body += "".join(f"3 {f[0]} {f[1]} {f[2]}\n" for f in faces)
            fh.write(body.encode("ascii"))
    return edge_m / math.sqrt(2.0)


def _config(path=None, body="Moon", resolution=5):
    mesh = {"resolution": resolution, "scale_factor": 1.0}
    if path is not None:
        mesh["import_file"] = str(path)
    return {"simulation": {"body": body}, "mesh": mesh}


# --------------------------------------------------------------------------

def test_the_measured_scale_matches_the_geometry():
    """Both PLY encodings, against a scale known in closed form."""
    with tempfile.TemporaryDirectory() as tmp:
        for binary in (True, False):
            path = Path(tmp) / f"grid_{'bin' if binary else 'ascii'}.ply"
            expected = _write_grid_ply(path, edge_m=4.0, binary=binary)
            got = facet_scale(_config(path))
            assert got.meters is not None, got.detail
            assert abs(got.meters - expected) / expected < 1e-4, (
                f"{'binary' if binary else 'ascii'} PLY: measured "
                f"{got.meters} against {expected}")
            assert got.n_faces == 128, got.n_faces


def test_the_scale_factor_is_applied():
    """mesh.scale_factor multiplies the geometry, so it multiplies the scale."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "grid.ply"
        expected = _write_grid_ply(path, edge_m=4.0)
        config = _config(path)
        config["mesh"]["scale_factor"] = 0.001
        got = facet_scale(config)
        assert abs(got.meters - expected * 0.001) / (expected * 0.001) < 1e-4


def test_a_millimeter_scale_mesh_is_refused():
    """The moon3d 509k case: 1.96 mm facets, the mesh IS the roughness."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "fine.ply"
        _write_grid_ply(path, edge_m=0.00277)      # -> ~1.96 mm facet scale
        policy = beaming_policy(_config(path))
        assert not policy.allowed, policy.reason
        assert "roughness" in policy.reason
        assert policy.scale.meters < FACET_SCALE_DISABLE_BELOW_M


def test_the_warn_band_requires_an_acknowledgment():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "mid.ply"
        _write_grid_ply(path, edge_m=0.4)          # -> ~0.28 m facet scale
        policy = beaming_policy(_config(path))
        assert policy.allowed, policy.reason
        assert policy.needs_acknowledgment, policy.reason
        assert FACET_SCALE_DISABLE_BELOW_M <= policy.scale.meters \
            < FACET_SCALE_WARN_BELOW_M


def test_a_meter_scale_mesh_passes_cleanly():
    """The A3GT 3 km core case: 3.46 m facets."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "coarse.ply"
        _write_grid_ply(path, edge_m=4.9)          # -> ~3.46 m facet scale
        policy = beaming_policy(_config(path))
        assert policy.allowed and not policy.needs_acknowledgment, policy.reason
        assert policy.scale.meters >= FACET_SCALE_WARN_BELOW_M


def test_the_boundaries_are_where_the_decision_says():
    """Straddle each threshold, so a moved constant fails here."""
    with tempfile.TemporaryDirectory() as tmp:
        for target, want_allowed, want_ack in (
                (FACET_SCALE_DISABLE_BELOW_M * 0.9, False, False),
                (FACET_SCALE_DISABLE_BELOW_M * 1.1, True, True),
                (FACET_SCALE_WARN_BELOW_M * 0.9, True, True),
                (FACET_SCALE_WARN_BELOW_M * 1.1, True, False)):
            path = Path(tmp) / f"b_{target:.4f}.ply"
            _write_grid_ply(path, edge_m=target * math.sqrt(2.0))
            policy = beaming_policy(_config(path))
            assert policy.allowed is want_allowed, (
                f"facet scale {target} m: allowed={policy.allowed}, "
                f"expected {want_allowed}. {policy.reason}")
            assert policy.needs_acknowledgment is want_ack, (
                f"facet scale {target} m: ack={policy.needs_acknowledgment}, "
                f"expected {want_ack}")


def test_non_lunar_bodies_are_refused_whatever_the_scale():
    """w = 32.5 deg is calibrated on lunar regolith and stops there."""
    for body in ("Europa", "Mercury", "Mars", "Ceres", "Enceladus", "Titan"):
        assert body not in BEAMING_BODIES
        policy = beaming_policy(_config(body=body, resolution=5))
        assert not policy.allowed, f"{body}: {policy.reason}"
        assert "lunar regolith" in policy.reason


def test_a_lunar_icosphere_is_allowed_and_costs_no_file_read():
    for res in (3, 5, 7):
        policy = beaming_policy(_config(resolution=res))
        assert policy.allowed and not policy.needs_acknowledgment, policy.reason
        assert policy.scale.source == "icosphere"
        assert policy.scale.n_faces == 20 * (4 ** res)


def _write_colored_ply(path, edge_m):
    """A meshlab-style face element: index list, then per-face RGB."""
    verts = [(0.0, 0.0, 0.0), (edge_m, 0.0, 0.0), (0.0, edge_m, 0.0),
             (edge_m, edge_m, 0.0)]
    faces = [(0, 1, 2), (1, 3, 2)]
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(verts)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              f"element face {len(faces)}\n"
              "property list uchar int vertex_indices\n"
              "property uchar red\nproperty uchar green\n"
              "property uchar blue\nend_header\n")
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        for v in verts:
            fh.write(struct.pack("<fff", *v))
        for f in faces:
            fh.write(struct.pack("<Biii", 3, *f) + bytes([255, 0, 0]))
    return edge_m / math.sqrt(2.0)


def test_a_face_element_with_trailing_colors_is_read_correctly():
    """The engine reads this layout, so refusing it would refuse a valid mesh.

    ``mesh_utils.cpp`` locates the ``vertex_indices`` property by name and skips
    the rest, so a MeshLab color export runs fine. What must not happen is
    reading only the index list and leaving the stream misaligned: a misread
    record that happens to decode ``n == 3`` yields a WRONG mean area silently,
    which could report a mesh as coarser than it is and allow beaming on one the
    rule exists to refuse.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "colored.ply"
        expected = _write_colored_ply(path, edge_m=4.0)
        scale = facet_scale(_config(path))
        assert scale.meters is not None, scale.detail
        assert abs(scale.meters - expected) / expected < 1e-4, (
            f"per-face colors threw the reader off: measured {scale.meters}, "
            f"expected {expected}")
        assert scale.n_faces == 2


def test_a_second_list_on_the_face_element_is_refused():
    """An unknown-length trailing list cannot be skipped; refuse, do not guess."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "twolists.ply"
        verts = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
        header = ("ply\nformat binary_little_endian 1.0\n"
                  f"element vertex {len(verts)}\n"
                  "property float x\nproperty float y\nproperty float z\n"
                  "element face 1\n"
                  "property list uchar int vertex_indices\n"
                  "property list uchar float texcoord\nend_header\n")
        with open(path, "wb") as fh:
            fh.write(header.encode("ascii"))
            for v in verts:
                fh.write(struct.pack("<fff", *v))
            fh.write(struct.pack("<Biii", 3, 0, 1, 2))
            fh.write(struct.pack("<B", 6) + struct.pack("<6f", *([0.0] * 6)))
        scale = facet_scale(_config(path))
        assert scale.meters is None, (
            f"a second per-face list was read as {scale.meters} m; the offsets "
            f"cannot be known without decoding it")
        assert "layout" in scale.detail, scale.detail
        assert not beaming_policy(_config(path)).allowed


def test_an_unreadable_mesh_refuses_rather_than_guesses():
    """Refusing is the safe direction, and it has to say why."""
    for name, payload in (("missing.ply", None),
                          ("nope.vtk", b"# vtk DataFile Version 5.1\n"),
                          ("broken.ply", b"ply\nformat ascii 1.0\n")):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / name
            if payload is not None:
                path.write_bytes(payload)
            policy = beaming_policy(_config(path))
            assert not policy.allowed, f"{name}: {policy.reason}"
            assert policy.reason, f"{name} refused with no reason"


def test_a_topographic_lunar_mesh_is_measurable():
    """The standard lunar production configuration must not be refused.

    A topographic run loads ``data/topo/<body>/topo_<body>_ico<N>.bin``, which
    is an ico-N mesh. MEASURED on this tree with the engine's own
    "topographic mesh with ... faces" line: ico4 -> 5,120 faces and
    ico5 -> 20,480, both exactly ``20*4^N``. Reporting it unmeasurable disabled
    the beaming switch and raised the launch guard on every run of a mesh whose
    facets are tens of kilometers across.
    """
    for res in (4, 5, 6):
        config = _config(resolution=res)
        config["mesh"]["use_topographic"] = True
        scale = facet_scale(config)
        assert scale.meters is not None, scale.detail
        assert scale.n_faces == 20 * (4 ** res), scale.n_faces
        assert scale.source == "icosphere-topographic", scale.source
        # The displacement adds surface area, so this is an upper bound; the
        # detail has to say so rather than present it as exact.
        assert "slightly smaller" in scale.detail, scale.detail
        assert beaming_policy(config).allowed


def test_a_non_lunar_body_is_refused_without_reading_the_mesh():
    """The body check must come first: measuring decodes every PLY vertex.

    beaming_policy runs from mesh.import_file's per-keystroke textChanged, so a
    Europa project with a large imported mesh was paying a multi-million-vertex
    decode on the GUI thread for an answer that is "no" on body grounds anyway.
    """
    policy = beaming_policy(_config(Path("/nonexistent/huge.ply"), body="Europa"))
    assert not policy.allowed
    assert policy.scale is not None and policy.scale.meters is None
    assert "wrong body" in policy.scale.detail, policy.scale.detail


def test_the_launch_guard_fires_on_a_refused_mesh():
    """The tab shows the rule; the LAUNCH is where it has to be asked.

    The tab deliberately never rewrites `radiation.beaming`, so without a
    preflight guard the only disclosure was a label on a tab the user may not
    be looking at.
    """
    from gui_core.preflight import PreflightReport

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "fine.ply"
        _write_grid_ply(path, edge_m=0.00277)
        config = _config(path)
        config["radiation"] = {"enabled": True, "beaming": True,
                               "rms_slope_deg": 32.5}
        policy = beaming_policy(config)
        assert not policy.allowed

    # The report field and its summary tag exist and are routed as a warning.
    report = PreflightReport()
    report.beaming_warning = "test"
    lines = report.summary_lines()
    assert any(ln.startswith("BEAM-GUARD:") for ln in lines), lines


TESTS = [
    test_the_measured_scale_matches_the_geometry,
    test_the_scale_factor_is_applied,
    test_a_millimeter_scale_mesh_is_refused,
    test_the_warn_band_requires_an_acknowledgment,
    test_a_meter_scale_mesh_passes_cleanly,
    test_the_boundaries_are_where_the_decision_says,
    test_non_lunar_bodies_are_refused_whatever_the_scale,
    test_a_lunar_icosphere_is_allowed_and_costs_no_file_read,
    test_a_face_element_with_trailing_colors_is_read_correctly,
    test_a_second_list_on_the_face_element_is_refused,
    test_an_unreadable_mesh_refuses_rather_than_guesses,
    test_a_topographic_lunar_mesh_is_measurable,
    test_a_non_lunar_body_is_refused_without_reading_the_mesh,
    test_the_launch_guard_fires_on_a_refused_mesh,
]


def main():
    failures = 0
    for test in TESTS:
        try:
            test()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {test.__name__}\n     {exc}")
        else:
            print(f"PASS {test.__name__}")
    total = len(TESTS)
    print(f"\n[test_gui_beaming_scale_rule] {total - failures}/{total} passed, "
          f"thresholds {FACET_SCALE_DISABLE_BELOW_M} m / "
          f"{FACET_SCALE_WARN_BELOW_M} m, bodies {list(BEAMING_BODIES)}")
    if failures:
        print("[test_gui_beaming_scale_rule] FAIL")
        return 1
    print("[test_gui_beaming_scale_rule] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
