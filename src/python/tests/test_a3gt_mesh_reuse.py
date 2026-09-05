#!/usr/bin/env python3
"""A cached core mesh is reusable because it is *valid*, not because it is big.

``a3gt_v2_orchestrator.build_mesh`` decided whether to reuse a site's existing
core mesh with::

    if ply.exists() and ply.stat().st_size > 1_000_000 and read_n_faces(ply) > 0:

The 1 MB floor is a proxy for "this file is complete", and it is wrong in the
direction that costs a campaign. A03SP010's real, correct, corrected-VF-paired
core mesh is a legitimately coarse LOLA-fallback mesh: **379,264 bytes, 8,746
faces**, matching its ``A03SP010_corrected_vf_v3.csr`` header exactly. The
floor called it incomplete, so the orchestrator rebuilt it rather than reusing
it -- and the rebuild refused (the site VRT's source raster is not beside it),
exiting 2 and hard-stopping the 56-window ring-fix campaign at 48/56 on
2026-09-03.

``mesh_reuse_reason`` replaces the floor with the checks the floor stood in for:

1. the header parses and declares at least one face;
2. the file is long enough to hold the vertices and faces it declares -- a
   *truncated* PLY is the real failure a size floor was ever approximating;
3. the face count equals the one the site's pinned corrected VF was built for,
   when it has one (the same mesh identity ``PTM_HMATRIX_VF_REPLACE`` enforces
   in ``radiation_exchange.cpp``, applied before a window starts rather than
   after the matrix upload).

The load-bearing check here is #1 in the list below: it FAILS against the old
rule and PASSES against the new one. Everything else guards the ways a
too-permissive replacement could let a bad mesh through.

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

from a3gt_v2_orchestrator import (            # noqa: E402
    mesh_reuse_reason, parse_ply_header, read_csr_n_faces, read_n_faces,
)

#: The real artifacts from the site that triggered this. Local campaign data,
#: not shipped with the repository, so every check that wants them SKIPs when
#: they are absent rather than reporting red on a fresh clone.
SP010_MESH = Path(
    "/Users/paha3326/projects/a3gt/meshes/"
    "A03SP010_87.130S_127.826E_R3km_linear.ply")
SP010_CSR = Path(
    "/Users/paha3326/projects/a3gt/vf_build/A03SP010/"
    "A03SP010_corrected_vf_v3.csr")
SP010_FACES = 8746
SP010_BYTES = 379_264

#: The rule this change removes, quoted verbatim so the test states exactly
#: what it is discriminating against.
OLD_FLOOR_BYTES = 1_000_000


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


# --------------------------------------------------------------------------
# 1. THE DISCRIMINATING CHECK
# --------------------------------------------------------------------------

def check_sp010_real_mesh_is_accepted_and_old_floor_rejected_it():
    """SP010's real mesh: refused by the floor, accepted by the new check.

    This is the check that would have caught the bug, and the one that fails
    if the fix is reverted. It asserts BOTH directions, so it cannot pass
    vacuously against a rule that accepts everything.
    """
    if not SP010_MESH.is_file():
        raise Skip(f"{SP010_MESH} not present")
    size = SP010_MESH.stat().st_size
    n = read_n_faces(SP010_MESH)
    assert size == SP010_BYTES, f"expected {SP010_BYTES:,} bytes, got {size:,}"
    assert n == SP010_FACES, f"expected {SP010_FACES:,} faces, got {n:,}"

    # The old rule, quoted: it rejects this legitimate mesh.
    old_ok = size > OLD_FLOOR_BYTES and n > 0
    assert not old_ok, (
        "the old floor would have ACCEPTED this mesh, so this test no longer "
        "discriminates -- check SP010_BYTES/OLD_FLOOR_BYTES")

    # The new rule accepts it, including against its own pinned VF.
    assert mesh_reuse_reason(SP010_MESH) is None, mesh_reuse_reason(SP010_MESH)
    r = mesh_reuse_reason(SP010_MESH, expected_faces=SP010_FACES)
    assert r is None, f"real SP010 mesh refused against its own VF: {r}"


def check_sp010_csr_header_agrees_with_the_mesh():
    """The pinned corrected VF's int32 header is the mesh's face count."""
    if not SP010_CSR.is_file():
        raise Skip(f"{SP010_CSR} not present")
    n_csr = read_csr_n_faces(SP010_CSR)
    assert n_csr == SP010_FACES, f"CSR header says {n_csr}, expected {SP010_FACES}"
    if SP010_MESH.is_file():
        assert read_n_faces(SP010_MESH) == n_csr, "mesh and CSR disagree"
        assert mesh_reuse_reason(SP010_MESH, expected_faces=n_csr) is None


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
        good = _write_ply(Path(td) / "good.ply", 4420, SP010_FACES)
        assert mesh_reuse_reason(good) is None, mesh_reuse_reason(good)
        full = good.stat().st_size

        cut = _write_ply(Path(td) / "cut.ply", 4420, SP010_FACES,
                         truncate_to=200_000)
        assert cut.stat().st_size == 200_000
        assert 200_000 > SP010_FACES * 13, (
            "the truncation point must exceed the naive lower bound, or this "
            "check does not discriminate")
        reason = mesh_reuse_reason(cut)
        assert reason is not None, (
            f"a PLY truncated from {full:,} to 200,000 bytes was ACCEPTED")
        assert "truncated" in reason, reason


def check_zero_face_ply_is_refused():
    """The 0-face PLY the failed SP010 rebuild would have written is refused."""
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

def check_wrong_face_count_for_the_pinned_vf_is_refused():
    """A complete, large, valid mesh is still refused if the VF disagrees.

    This is the case a size floor can never see: the mesh passes every
    size/parse test and is simply the wrong geometry for the matrix the run
    would load. The engine aborts on it (radiation_exchange.cpp); the
    orchestrator should not have started the window.
    """
    with tempfile.TemporaryDirectory() as td:
        ply = _write_ply(Path(td) / "m.ply", 4420, SP010_FACES)
        assert mesh_reuse_reason(ply) is None
        reason = mesh_reuse_reason(ply, expected_faces=2_397_890)
        assert reason is not None, "a mesh/VF face-count mismatch was accepted"
        assert "8,746" in reason and "2,397,890" in reason, reason


def check_no_vf_means_no_face_count_constraint():
    """With no pinned VF, the mesh is judged on its own validity alone."""
    with tempfile.TemporaryDirectory() as td:
        ply = _write_ply(Path(td) / "m.ply", 100, 200)
        assert mesh_reuse_reason(ply, expected_faces=None) is None


def check_unreadable_csr_yields_none_not_a_crash():
    with tempfile.TemporaryDirectory() as td:
        assert read_csr_n_faces(Path(td) / "absent.csr") is None
        short = Path(td) / "short.csr"
        short.write_bytes(b"\x01\x02")
        assert read_csr_n_faces(short) is None


# --------------------------------------------------------------------------
# 4. DO NO HARM: every other campaign site is still reused
# --------------------------------------------------------------------------

def check_other_campaign_meshes_still_reused():
    """The 6 sites the floor accepted are still accepted, unchanged.

    Scope note: this covers the ring-fix campaign's own core meshes only. It
    says nothing about any other campaign's meshes or about ring PLYs, which
    go through a separate 100 kB check this change does not touch.
    """
    mesh_dir = Path("/Users/paha3326/projects/a3gt/meshes")
    if not mesh_dir.is_dir():
        raise Skip(f"{mesh_dir} not present")
    others = sorted(p for p in mesh_dir.glob("A03*_R3km_linear.ply")
                    if "BACKUP" not in p.name)
    if not others:
        raise Skip("no campaign core meshes on disk")
    checked = 0
    for ply in others:
        n = read_n_faces(ply)
        size = ply.stat().st_size
        reason = mesh_reuse_reason(ply)
        assert reason is None, f"{ply.name} ({size:,} B, {n:,} faces): {reason}"
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
    check_sp010_real_mesh_is_accepted_and_old_floor_rejected_it,
    check_sp010_csr_header_agrees_with_the_mesh,
    check_truncated_ply_is_refused,
    check_zero_face_ply_is_refused,
    check_unparsable_file_is_refused,
    check_missing_file_is_refused,
    check_wrong_face_count_for_the_pinned_vf_is_refused,
    check_no_vf_means_no_face_count_constraint,
    check_unreadable_csr_yields_none_not_a_crash,
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
