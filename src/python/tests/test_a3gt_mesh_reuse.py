#!/usr/bin/env python3
"""A cached core mesh is reusable because it is *valid*, not because it is big.

``a3gt_v2_orchestrator.build_mesh`` decided whether to reuse a site's existing
core mesh with::

    if ply.exists() and ply.stat().st_size > 1_000_000 and read_n_faces(ply) > 0:

The 1 MB floor is a proxy for "this file is complete", and it is wrong in the
direction that costs a campaign. A legitimately coarse LOLA-fallback core mesh
is **379,264 bytes / 8,746 faces** -- well under the floor, and exactly the
geometry its own corrected-VF CSR header declares. The floor called such a mesh
incomplete, so the orchestrator rebuilt it rather than reusing it -- and the
rebuild refused (that site's VRT source raster is not beside it), exiting 2 and
hard-stopping the 56-window ring-fix campaign at 48/56 on 2026-09-03.

``mesh_reuse_reason`` replaces the floor with the checks the floor stood in for:

1. the header parses and declares at least one face;
2. the file is long enough to hold the vertices and faces it declares -- a
   *truncated* PLY is the real failure a size floor was ever approximating;
3. the face count equals the one the site's pinned corrected VF was built for,
   when it has one (the same mesh identity ``PTM_HMATRIX_VF_REPLACE`` enforces
   in ``radiation_exchange.cpp``, applied before a window starts rather than
   after the matrix upload).

The load-bearing check is the first one below: it FAILS against the old rule
and PASSES against the new one. Everything else guards the ways a
too-permissive replacement could let a bad mesh through.

**Why the leading case is synthetic.** It used to name the real mesh by its
site identifier and its coordinates, which this project does not publish
(``src/python/tests/test_tracked_tree_hygiene.py``). It is now built here, from
``_write_ply``, at 4,420 vertices and 8,746 faces -- which produces a PLY of
**exactly 379,264 bytes**, the real file's size to the byte, because both carry
the same ``dtm_to_ply_delaunay`` property layout. So the discriminating arm no
longer needs local data at all, and it went from SKIPPING on a fresh clone to
running everywhere. The real artifacts are still checked, by the arms directly
below it, through the local fixture registry (``local_fixtures``) that already
exists for exactly this: paths that name sites, kept in
``~/.ptm3d/test_fixtures.json`` rather than in a tracked file.

Run directly::

    python3 src/python/tests/test_a3gt_mesh_reuse.py
"""

from __future__ import annotations

import struct
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import local_fixtures                          # noqa: E402
from a3gt_v2_orchestrator import (             # noqa: E402
    MeshVFMismatch, build_mesh, mesh_reuse_reason, parse_ply_header,
    read_csr_n_faces, read_n_faces,
)

#: The size class that provoked this change. These are MEASUREMENTS of a mesh
#: kind, not identifiers of a place: five of the campaign's sites carry a
#: LOLA-fallback core mesh with exactly these numbers. ``COARSE_VERTICES`` is
#: what ``_write_ply`` needs to reproduce ``COARSE_BYTES`` exactly, and the
#: first check below asserts that it does rather than trusting the arithmetic.
COARSE_FACES = 8_746
COARSE_VERTICES = 4_420
COARSE_BYTES = 379_264

#: The rule this change removes, quoted verbatim so the test states exactly
#: what it is discriminating against.
OLD_FLOOR_BYTES = 1_000_000

#: An invented tag for fixture filenames. It is deliberately shaped UNLIKE a
#: real site identifier -- it matches neither the deny list in
#: ``~/.ptm3d/sensitive_identifiers.json`` nor the structural patterns in
#: ``scripts/check_site_id_patterns.py``, which is the point: a fixture named
#: after the run it came from is how the identifiers got back into this file.
FIXTURE_TAG = "TESTSITE01"

#: A face count for the "pinned VF is for a different mesh" case. Any number
#: other than ``COARSE_FACES`` works; this is the size of a full-resolution
#: 3 km core, so the fixture mismatch is the realistic one (coarse cached mesh
#: against a matrix built for the fine geometry).
OTHER_VF_FACES = 2_397_890


class Skip(Exception):
    """This check cannot run in this environment (missing local data)."""


def _write_ply(path: Path, n_vertices: int, n_faces: int,
               *, truncate_to: int | None = None) -> Path:
    """A binary PLY with the dtm_to_ply_delaunay property layout.

    3 doubles + 3 floats + 3 doubles per vertex (60 B), one uchar-counted
    triangle of int32 indices per face (13 B).
    """
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment DTM to PLY with Delaunay triangulation for PTM3D\n"
        f"element vertex {n_vertices}\n"
        "property double x\nproperty double y\nproperty double z\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        "property double latitude\nproperty double longitude\n"
        "property double elevation\n"
        f"element face {n_faces}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    ).encode("ascii")
    body = bytearray()
    for i in range(n_vertices):
        body += struct.pack("<3d3f3d", float(i), 0.0, 0.0,
                            0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    for i in range(n_faces):
        body += struct.pack("<B3i", 3, i % max(n_vertices, 1),
                            (i + 1) % max(n_vertices, 1),
                            (i + 2) % max(n_vertices, 1))
    blob = header + bytes(body)
    if truncate_to is not None:
        blob = blob[:truncate_to]
    path.write_bytes(blob)
    return path


def _assert_coarse_mesh_discriminates(ply: Path) -> None:
    """The whole discrimination, asserted in both directions, on one file.

    Shared by the synthetic arm and the real-artifact arm so the two cannot
    drift: whatever the synthetic fixture is asked to prove, the real mesh is
    asked to prove as well.
    """
    size = ply.stat().st_size
    n = read_n_faces(ply)
    assert n == COARSE_FACES, f"expected {COARSE_FACES:,} faces, got {n:,}"
    assert size == COARSE_BYTES, f"expected {COARSE_BYTES:,} bytes, got {size:,}"

    # The old rule, quoted: it rejects this legitimate mesh.
    old_ok = size > OLD_FLOOR_BYTES and n > 0
    assert not old_ok, (
        "the old floor would have ACCEPTED this mesh, so this test no longer "
        "discriminates -- check COARSE_BYTES/OLD_FLOOR_BYTES")

    # The new rule accepts it.
    assert mesh_reuse_reason(ply) is None, mesh_reuse_reason(ply)


# --------------------------------------------------------------------------
# 1. THE DISCRIMINATING CHECK
# --------------------------------------------------------------------------

def check_coarse_valid_mesh_is_accepted_and_old_floor_rejected_it():
    """A coarse but valid mesh: refused by the floor, accepted by the new check.

    This is the check that would have caught the bug, and the one that fails
    if the fix is reverted. It asserts BOTH directions, so it cannot pass
    vacuously against a rule that accepts everything -- and it is synthetic,
    so it runs on a fresh clone with no campaign data anywhere on the machine.
    """
    with tempfile.TemporaryDirectory() as td:
        ply = _write_ply(Path(td) / f"{FIXTURE_TAG}_R3km_linear.ply",
                         COARSE_VERTICES, COARSE_FACES)
        _assert_coarse_mesh_discriminates(ply)


def check_registered_coarse_mesh_is_the_shape_the_synthetic_one_claims():
    """The real artifact, when this machine has it, is that same mesh.

    The synthetic fixture above is only worth anything if it is a faithful
    stand-in, and "faithful" here is checkable to the byte. Registered as
    ``coarse_lola_core_mesh`` in the local fixture file so this tracked test
    does not have to name the site the mesh belongs to.

    A mesh of a DIFFERENT size class registered under that name is a
    misregistration, not a regression, so it is a named skip: the arm compares
    against one specific pair of numbers, and reddening the suite because a
    colleague registered their own coarse mesh would say nothing about the fix.
    """
    mesh = local_fixtures.fixture_file("coarse_lola_core_mesh", Skip)
    size, n = mesh.stat().st_size, read_n_faces(mesh)
    if (size, n) != (COARSE_BYTES, COARSE_FACES):
        raise Skip(
            f"the mesh registered as 'coarse_lola_core_mesh' is {size:,} bytes "
            f"/ {n:,} faces, and this arm compares against the "
            f"{COARSE_BYTES:,}-byte / {COARSE_FACES:,}-face class the synthetic "
            f"fixture reproduces; register one of those, or update both "
            f"constants together")
    _assert_coarse_mesh_discriminates(mesh)


def check_registered_corrected_vf_header_agrees_with_the_mesh():
    """The pinned corrected VF's int32 header is the mesh's face count."""
    csr = local_fixtures.fixture_file("coarse_lola_corrected_vf", Skip)
    n_csr = read_csr_n_faces(csr)
    assert n_csr == COARSE_FACES, f"CSR header says {n_csr}, expected {COARSE_FACES}"
    try:
        mesh = local_fixtures.fixture_file("coarse_lola_core_mesh", Skip)
    except Skip:
        return
    assert read_n_faces(mesh) == n_csr, "mesh and CSR disagree"


# --------------------------------------------------------------------------
# 2. THE REPLACEMENT MUST STILL REFUSE WHAT THE FLOOR WAS AIMED AT
# --------------------------------------------------------------------------

def check_truncated_ply_is_refused():
    """A half-written PLY is refused -- the failure a size floor approximated.

    Truncated to 200,000 bytes: still over any naive "faces * 13 bytes" lower
    bound (113,698 here), so only an honest expected-size computation catches
    it. This is the specific weak replacement this check rules out.
    """
    with tempfile.TemporaryDirectory() as td:
        good = _write_ply(Path(td) / "good.ply", COARSE_VERTICES, COARSE_FACES)
        assert mesh_reuse_reason(good) is None, mesh_reuse_reason(good)
        full = good.stat().st_size

        cut = _write_ply(Path(td) / "cut.ply", COARSE_VERTICES, COARSE_FACES,
                         truncate_to=200_000)
        assert cut.stat().st_size == 200_000
        assert 200_000 > COARSE_FACES * 13, (
            "the truncation point must exceed the naive lower bound, or this "
            "check does not discriminate")
        reason = mesh_reuse_reason(cut)
        assert reason is not None, (
            f"a PLY truncated from {full:,} to 200,000 bytes was ACCEPTED")
        assert "truncated" in reason, reason


def check_zero_face_ply_is_refused():
    """The 0-face PLY a failed rebuild would have written is refused."""
    with tempfile.TemporaryDirectory() as td:
        empty = _write_ply(Path(td) / "empty.ply", 0, 0)
        reason = mesh_reuse_reason(empty)
        assert reason is not None, "a 0-face PLY was accepted"
        assert "no faces" in reason, reason


def check_unparsable_file_is_refused():
    """A file that is not a PLY at all is refused, not crashed on."""
    with tempfile.TemporaryDirectory() as td:
        junk = Path(td) / "junk.ply"
        junk.write_bytes(b"\x00\x01\x02" * 5000)
        reason = mesh_reuse_reason(junk)
        assert reason is not None, "a non-PLY file was accepted"
        assert "parse" in reason, reason


def check_missing_file_is_refused():
    with tempfile.TemporaryDirectory() as td:
        reason = mesh_reuse_reason(Path(td) / "nope.ply")
        assert reason is not None and "exist" in reason, reason


# --------------------------------------------------------------------------
# 3. THE VF-DIMENSION HALF
# --------------------------------------------------------------------------

def check_vf_mismatch_refuses_and_does_NOT_rebuild_over_the_mesh():
    """A mesh/VF face-count mismatch skips the site; it never rebuilds.

    This is the case a size floor can never see: the mesh passes every
    size/parse test and is simply the wrong geometry for the matrix the run
    would load. Rebuilding cannot fix it -- the CSR is the fixed, multi-hour
    artifact -- and would destroy the only copy of the geometry. So build_mesh
    must raise rather than invoke the converter, and the file must be
    byte-identical afterwards.
    """
    with tempfile.TemporaryDirectory() as td:
        ply = Path(td) / f"{FIXTURE_TAG}_00.000S_000.000E_R3km_linear.ply"
        _write_ply(ply, COARSE_VERTICES, COARSE_FACES)
        assert mesh_reuse_reason(ply) is None, "fixture should be valid"
        before = ply.read_bytes()
        site = {"ellipse_id": FIXTURE_TAG, "lat_deg": -87.0, "lon_deg": 127.0}
        try:
            build_mesh(site, dry_run=False, roi_km=3.0, mesh_path=ply,
                       dem_override=Path("/nonexistent/dem.vrt"),
                       expected_faces=OTHER_VF_FACES)
        except MeshVFMismatch as exc:
            msg = str(exc)
            assert f"{COARSE_FACES:,}" in msg and f"{OTHER_VF_FACES:,}" in msg, msg
            assert "Not rebuilding" in msg, msg
        else:
            raise AssertionError(
                "build_mesh accepted a mesh/VF mismatch instead of refusing")
        assert ply.read_bytes() == before, (
            "the mesh was modified while refusing a VF mismatch")


def check_no_vf_means_no_face_count_constraint():
    """With no pinned VF, the mesh is judged on its own validity alone."""
    with tempfile.TemporaryDirectory() as td:
        ply = _write_ply(Path(td) / "m.ply", 100, 200)
        assert mesh_reuse_reason(ply) is None


def check_unreadable_csr_yields_none_not_a_crash():
    with tempfile.TemporaryDirectory() as td:
        assert read_csr_n_faces(Path(td) / "absent.csr") is None
        short = Path(td) / "short.csr"
        short.write_bytes(b"\x01\x02")
        assert read_csr_n_faces(short) is None


def check_an_hmat_vf_is_not_mistaken_for_a_csr():
    """An .hmat/.vf file must not be read as a face count.

    vf_replace_path() accepts `<site>.vf` as a corrected-VF candidate, and a
    `.vf` here is the H-matrix format whose first four bytes are the magic
    ('TAMH' on disk). Believed as an int32 that is ~1.2e9 faces -- which would
    declare EVERY cached mesh mismatched and rebuild all of them, recreating
    the exact mass-rebuild this change exists to prevent.
    """
    with tempfile.TemporaryDirectory() as td:
        stub = Path(td) / f"{FIXTURE_TAG}.vf"
        # The real 80-byte base-VF stub layout: magic, version, rows, cols, ...
        stub.write_bytes(b"TAMH" + struct.pack("<i", 2)
                         + struct.pack("<2q", COARSE_FACES, COARSE_FACES)
                         + b"\x00" * 56)
        assert len(stub.read_bytes()) == 80
        got = read_csr_n_faces(stub)
        assert got is None, (
            f"an H-matrix .vf was read as a face count of {got}")

    # And the general structural guard: a plausible n with a file far too
    # small to be that CSR is refused too.
    with tempfile.TemporaryDirectory() as td:
        fake = Path(td) / "fake.csr"
        fake.write_bytes(struct.pack("<i", OTHER_VF_FACES) + b"\x00" * 64)
        assert read_csr_n_faces(fake) is None, (
            "a too-small file was accepted as a 2.4M-face CSR")


def check_registered_base_vf_stub_on_disk_is_rejected():
    """The same, against a real committed stub rather than a fixture."""
    stub = local_fixtures.fixture_file("coarse_lola_base_vf_stub", Skip)
    assert read_csr_n_faces(stub) is None, (
        "a real base-VF stub on disk was read as a face count")


def check_scalar_face_property_disables_the_size_check():
    """A face property we cannot size must not yield an under-estimate.

    Under-estimating `need` would let a genuinely truncated PLY (from any
    writer that adds face colors, say) pass the truncation test and be reused.
    """
    with tempfile.TemporaryDirectory() as td:
        ply = Path(td) / "colored.ply"
        header = (
            "ply\nformat binary_little_endian 1.0\n"
            "element vertex 3\n"
            "property double x\nproperty double y\nproperty double z\n"
            "element face 2\n"
            "property list uchar int vertex_indices\n"
            "property uchar red\n"
            "end_header\n").encode("ascii")
        ply.write_bytes(header + b"\x00" * (3 * 24 + 2 * 14))
        info = parse_ply_header(ply)
        assert info is not None and info["n_faces"] == 2, info
        assert info["face_record_bytes"] is None, (
            "sized a face record that carries an unaccounted scalar property")
        assert mesh_reuse_reason(ply) is None, "a complete file was refused"


# --------------------------------------------------------------------------
# 4. DO NO HARM: every other campaign site is still reused
# --------------------------------------------------------------------------

def check_other_campaign_meshes_still_reused():
    """Every core mesh in the registered directory is still accepted.

    Scope note, stated as what it actually is rather than as what it used to
    be: this covers every ``*_R3km_linear.ply`` in whatever directory is
    registered as ``campaign_core_mesh_dir``, so on this machine it is the
    campaign's core meshes, and on another it is whatever that person put
    there. It says nothing about ring PLYs, which go through a separate 100 kB
    check this change does not touch.

    A failure here names the mesh by its INDEX in the sorted listing plus its
    size and face count, not by filename: the filenames carry site identifiers
    (``scripts/check_site_id_patterns.py``), and an assertion message ends up
    in logs and scrollback. The index is enough to find it locally.
    """
    mesh_dir = local_fixtures.fixture_path("campaign_core_mesh_dir", Skip)
    others = sorted(p for p in mesh_dir.glob("*_R3km_linear.ply")
                    if "BACKUP" not in p.name)
    if not others:
        raise Skip(f"no core meshes matching *_R3km_linear.ply in {mesh_dir}")
    checked = 0
    for index, ply in enumerate(others):
        n = read_n_faces(ply)
        size = ply.stat().st_size
        reason = mesh_reuse_reason(ply)
        assert reason is None, (
            f"mesh #{index} of {len(others)} in the registered directory "
            f"({size:,} B, {n:,} faces): {reason}")
        if size > OLD_FLOOR_BYTES:
            checked += 1
    assert checked >= 1, "expected at least one above-floor mesh to confirm"
    print(f"    ({len(others)} core meshes accepted, "
          f"{checked} of them also passed the old floor)")


def check_header_parse_reports_the_real_layout():
    """parse_ply_header returns the numbers the size arithmetic depends on."""
    with tempfile.TemporaryDirectory() as td:
        ply = _write_ply(Path(td) / "m.ply", 7, 11)
        info = parse_ply_header(ply)
        assert info is not None
        assert info["n_vertices"] == 7 and info["n_faces"] == 11, info
        assert info["vertex_stride"] == 60, info      # 3d + 3f + 3d
        assert info["face_record_bytes"] == 13, info  # uchar + 3 int32
        expect = info["header_bytes"] + 7 * 60 + 11 * 13
        assert ply.stat().st_size == expect, (ply.stat().st_size, expect)


CHECKS = [
    check_coarse_valid_mesh_is_accepted_and_old_floor_rejected_it,
    check_registered_coarse_mesh_is_the_shape_the_synthetic_one_claims,
    check_registered_corrected_vf_header_agrees_with_the_mesh,
    check_truncated_ply_is_refused,
    check_zero_face_ply_is_refused,
    check_unparsable_file_is_refused,
    check_missing_file_is_refused,
    check_vf_mismatch_refuses_and_does_NOT_rebuild_over_the_mesh,
    check_no_vf_means_no_face_count_constraint,
    check_unreadable_csr_yields_none_not_a_crash,
    check_an_hmat_vf_is_not_mistaken_for_a_csr,
    check_registered_base_vf_stub_on_disk_is_rejected,
    check_scalar_face_property_disables_the_size_check,
    check_other_campaign_meshes_still_reused,
    check_header_parse_reports_the_real_layout,
]


def main() -> int:
    failed = skipped = 0
    for fn in CHECKS:
        name = fn.__name__
        try:
            fn()
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
