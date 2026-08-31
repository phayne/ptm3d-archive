#!/usr/bin/env python3
"""Finding a view-factor matrix that is already on disk, without guessing.

**Why this file exists.** A view-factor matrix does not record the surface it
was built for. Two lunar sites at the same DEM posting produce meshes of
different face counts, so a count catches most mismatches -- and two matrices
that agree on the count do not agree on the geometry. That is the trap the
whole search has to be built around: the only thing worse than not finding a
matrix is finding the wrong one and wiring it, because the run then finishes.

So this suite is about what is CHECKED, not about what is found:

  * a rows-mismatch is REFUSED, with both numbers, not ranked low
  * a provenance sidecar naming this mesh outranks a bare face-count match
  * an H-matrix says its physics generation is unverifiable, every time
  * a version-1 sparse file says its near-field estimator was retired
  * an empty matrix is not offered as a matrix
  * a cloud placeholder is refused BEFORE anything reads it
  * nothing is selected, applied or written by any of it

Every fixture below is written byte by byte from the format the engine's own
writer produces, so a format change breaks this suite rather than silently
making the search blind.

Run directly::

    python3 src/python/tests/test_gui_vf_discovery.py
"""

from __future__ import annotations

import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "python"))
sys.path.insert(0, str(REPO_ROOT / "gui"))


class Skip(Exception):
    """A check that cannot run here, as distinct from one that failed."""


def _discovery():
    try:
        from gui_core import vf_discovery
    except ImportError as exc:                            # pragma: no cover
        raise Skip(f"the GUI's discovery module is unavailable here ({exc})")
    return vf_discovery


# --------------------------------------------------------------------------
# Fixtures, written the way the engine writes them
# --------------------------------------------------------------------------

def _ply(path, faces, vertices=None):
    """A minimal ASCII PLY whose header declares `faces` faces.

    Only the header is read by anything under test (and by the pre-flight's
    own reader, which this delegates to), so the body is one comment line.
    """
    vertices = faces + 2 if vertices is None else vertices
    path.write_text(
        "ply\nformat ascii 1.0\n"
        f"element vertex {vertices}\nproperty float x\n"
        f"element face {faces}\nproperty list uchar int vertex_indices\n"
        "end_header\n")
    return path


def _vffm(path, rows, nnz, version=2):
    """A sparse VFFM file: header only, which is all any probe reads.

    ``view_factor.cpp:527`` -- magic, version, nrows, ncols, nnz, then the
    arrays. The arrays are omitted deliberately: a probe that needed them
    would be reading gigabytes from a Qt slot.
    """
    path.write_bytes(struct.pack("<IIQQQ", 0x5646464D, version, rows, rows,
                                 nnz))
    return path


def _hmat(path, rows, nnz, version=2, blocks=0):
    """An H-matrix file: the 32-byte header and the six trailing statistics.

    ``hmatrix_view_factor.cpp:5217``. With zero root blocks the two run
    together and the file is exactly 80 bytes -- which is what
    ``make_empty_vf`` writes and what the A3GT base stubs are.
    """
    body = struct.pack("<IIQQQ", 0x484D4154, version, rows, rows, blocks)
    stats = struct.pack("<6Q", 0, 0, 0, 0, 0, nnz)
    path.write_bytes(body + stats)
    return path


def _csr(path, rows, nnz, geomhash=1234.5):
    """A replacement CSR, at full declared length.

    ``radiation_exchange.cpp:276``: ``i32 n, f64 geomhash, i64 indptr[n+1],
    i64 nnz, i32 cols[nnz], f32 data[nnz]``. Written out in full because the
    probe's identification test IS the total length -- this format has no
    magic number, so its own arithmetic is the only thing that says it is one.
    """
    head = struct.pack("<id", rows, geomhash)
    indptr = struct.pack(f"<{rows + 1}q", *range(rows + 1)) if rows else b""
    payload = struct.pack("<q", nnz)
    cols = struct.pack(f"<{nnz}i", *([0] * nnz)) if nnz else b""
    data = struct.pack(f"<{nnz}f", *([0.5] * nnz)) if nnz else b""
    path.write_bytes(head + indptr + payload + cols + data)
    return path


def _sidecar(csr_path, mesh_path, faces, geomhash, **extra):
    record = {"mesh": str(mesh_path), "n_faces": faces, "geomhash": geomhash,
              "nnz": 0, "nrays": 200000, "estimator": "FC-08 corrected",
              "assembled_utc": "2026-08-20T04:59:35Z",
              "assembler": "/somewhere/assemble_vf_haworth.py"}
    record.update(extra)
    Path(str(csr_path) + ".provenance.json").write_text(json.dumps(record))


# --------------------------------------------------------------------------
# The checks
# --------------------------------------------------------------------------

def test_a_rows_mismatch_is_refused_and_names_both_numbers():
    """The silent wrong-mesh trap, closed at the only place it can be.

    A matrix for another site loads, solves, and produces temperatures. The
    count is the cheapest thing that can disagree, so a disagreement must be a
    REFUSAL rather than a low rank -- and it has to print both numbers, because
    "does not match" leaves the user with no way to tell which file is wrong.
    """
    vf = _discovery()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        mesh = _ply(tmp / "site.ply", 1000)
        _vffm(tmp / "other.vf", rows=1001, nnz=5000)
        result = vf.discover_view_factors(mesh, project_root=tmp)
        assert not result.matches, (
            "a 1001-row matrix was OFFERED for a 1000-face mesh. That is the "
            "wrong-mesh-same-size failure with the one check that could have "
            "caught it switched off.")
        assert len(result.refusals) == 1, result.refusals
        why = result.refusals[0].refusal
        assert "1,001" in why and "1,000" in why, (
            "the refusal must carry BOTH counts so the user can tell which "
            f"file is the wrong one; it said: {why}")


def test_a_provenance_sidecar_naming_this_mesh_outranks_a_bare_match():
    """Rank 0 is evidence of IDENTITY, and the only evidence there is.

    Two matrices, both correctly sized. One has a sidecar naming this mesh and
    carrying the geometry hash in the matrix header; the other has nothing.
    Ranked on face count alone they are indistinguishable, which is precisely
    the case that needs an ordering.
    """
    vf = _discovery()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        mesh = _ply(tmp / "site.ply", 64)
        _csr(tmp / "anonymous.csr", rows=64, nnz=8, geomhash=99.5)
        traced = _csr(tmp / "traced.csr", rows=64, nnz=8, geomhash=42.25)
        _sidecar(traced, mesh, faces=64, geomhash=42.25)

        result = vf.discover_view_factors(mesh, project_root=tmp)
        assert result.best is not None and result.best.path.name == "traced.csr", (
            "the matrix whose provenance names this mesh must come first; got "
            f"{[c.path.name for c in result.matches]}")
        assert result.best.rank == vf.RANK_PROVENANCE
        anonymous = [c for c in result.matches if c.path.name == "anonymous.csr"]
        assert anonymous and anonymous[0].rank == vf.RANK_CSR_ROWS
        assert any("face count" in n for n in anonymous[0].notes), (
            "a matrix with no provenance must SAY that only its face count "
            f"was checked; its notes were {anonymous[0].notes}")


def test_a_sidecar_for_a_different_mesh_does_not_promote_it():
    """The sidecar has to agree with the mesh in hand, not merely exist.

    A sidecar naming some other terrain is still useful -- it says what this
    matrix is -- but it is evidence AGAINST wiring it here, and promoting on
    "has a sidecar" would invert that.
    """
    vf = _discovery()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        mesh = _ply(tmp / "site.ply", 64)
        foreign = _csr(tmp / "foreign.csr", rows=64, nnz=8, geomhash=42.25)
        _sidecar(foreign, tmp / "somewhere_else.ply", faces=64, geomhash=42.25)
        result = vf.discover_view_factors(mesh, project_root=tmp)
        assert result.best.rank == vf.RANK_CSR_ROWS, (
            "a sidecar naming a DIFFERENT mesh must not reach the "
            "provenance-verified tier")
        assert any("somewhere_else.ply" in n for n in result.best.notes), (
            f"it has to name the mesh it really belongs to: {result.best.notes}")


def test_a_sidecar_that_disagrees_with_its_own_matrix_is_not_used():
    """A sidecar is only evidence while it matches the file it sits beside."""
    vf = _discovery()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        mesh = _ply(tmp / "site.ply", 64)
        drifted = _csr(tmp / "drifted.csr", rows=64, nnz=8, geomhash=42.25)
        _sidecar(drifted, mesh, faces=64, geomhash=7.5)
        result = vf.discover_view_factors(mesh, project_root=tmp)
        assert result.best.rank == vf.RANK_CSR_ROWS
        assert any("different geometry" in n for n in result.best.notes), (
            f"the disagreement has to be stated: {result.best.notes}")


def test_an_hmatrix_always_declares_its_physics_generation_unverifiable():
    """The owed ledger row, made unmissable at the only place it surfaces.

    The sparse header carries a PHYSICS version (``view_factor.cpp:517-525``).
    The H-matrix header does not: magic, a format version that has been 2 for
    the life of the format, dimensions and block counts. So an H-matrix that
    loads proves compatibility and nothing about which estimator built it, and
    a ranking that did not say so would present the two formats as equally
    known.
    """
    vf = _discovery()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        mesh = _ply(tmp / "site.ply", 128)
        _hmat(tmp / "cache.hmat", rows=128, nnz=4096)
        result = vf.discover_view_factors(mesh, project_root=tmp)
        assert result.best is not None and result.best.rank == vf.RANK_HMAT
        joined = " ".join(result.best.notes)
        assert "unverifiable" in joined and "generation" in joined, (
            "every H-matrix candidate must disclose that its physics "
            f"generation cannot be read from the file; notes were {joined}")


def test_a_version_one_sparse_file_is_offered_and_warned_about():
    """Refusing it would force a rebuild, which is the wedge. So: warn.

    Same bytes, different physics. The engine loads a v1 file and announces
    it; this must not be stricter than the engine, and must not be quieter.
    """
    vf = _discovery()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        mesh = _ply(tmp / "site.ply", 256)
        _vffm(tmp / "old.vf", rows=256, nnz=9000, version=1)
        _vffm(tmp / "new.vf", rows=256, nnz=9000, version=2)
        result = vf.discover_view_factors(mesh, project_root=tmp)
        names = [c.path.name for c in result.matches]
        assert names == ["new.vf", "old.vf"], (
            f"version 2 must rank above version 1; got {names}")
        old = result.matches[1]
        assert old.ok, "a v1 file is offered, not refused -- refusing it " \
                       "forces the multi-gigabyte rebuild PTM-19 forbids"
        joined = " ".join(old.notes)
        assert "-22%" in joined and "2026-08-23" in joined, (
            f"the v1 warning must say what changed and when: {joined}")


def test_an_empty_matrix_is_not_offered_as_a_matrix():
    """nnz=0 loads clean and computes nothing. That is the June trap.

    It still appears, because it is exactly what the replacement-CSR route
    needs as a base -- but last, and saying what it is.
    """
    vf = _discovery()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        mesh = _ply(tmp / "site.ply", 512)
        _hmat(tmp / "base_empty.vf", rows=512, nnz=0)
        _vffm(tmp / "real.vf", rows=512, nnz=120000)
        result = vf.discover_view_factors(mesh, project_root=tmp)
        assert [c.path.name for c in result.matches] == ["real.vf",
                                                         "base_empty.vf"], (
            f"the empty stub must rank last; got "
            f"{[c.path.name for c in result.matches]}")
        stub = result.matches[-1]
        assert stub.rank == vf.RANK_EMPTY and stub.nnz == 0
        joined = " ".join(stub.notes)
        assert "no facet-to-facet exchange" in joined, (
            f"an empty matrix must say it computes nothing: {joined}")


def test_a_cloud_placeholder_is_refused_before_anything_reads_it():
    """A dataless file is refused on the stat, and never opened.

    The test proves the second half by making the file UNREADABLE: if the
    probe opened it, the probe would raise or report a read failure instead of
    the placeholder message.
    """
    vf = _discovery()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        placeholder = tmp / "evicted.vf"
        placeholder.write_bytes(b"\0" * 4096)

        real_stat = vf.os.stat

        class _Evicted:
            """A stat result that says 'name here, bytes elsewhere'."""

            def __init__(self, real):
                self.st_size = real.st_size
                self.st_blocks = 0
                self.st_dev = real.st_dev
                self.st_ino = real.st_ino

        def fake_stat(path, *args, **kwargs):
            result = real_stat(path, *args, **kwargs)
            if str(path).endswith("evicted.vf"):
                return _Evicted(result)
            return result

        vf.os.stat = fake_stat
        try:
            candidate = vf.probe_view_factor(placeholder, mesh_faces=10)
        finally:
            vf.os.stat = real_stat
        assert candidate.dataless and not candidate.ok
        assert "placeholder" in candidate.refusal, candidate.refusal
        assert "Always Keep on This Device" in candidate.refusal, (
            "the refusal must say how to fix it, not only that it is broken")


def test_a_file_that_is_not_a_view_factor_matrix_is_refused_by_name():
    """An unrecognized magic is refused, and the refusal names the cost.

    The consequence of an unloadable cache is not an error the run stops on:
    the loader returns false and the matrix is REBUILT. That is the whole
    reason a refusal here has to be readable.
    """
    vf = _discovery()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        mesh = _ply(tmp / "site.ply", 32)
        (tmp / "notamatrix.vf").write_bytes(b"# vtk DataFile Version 5.1\n"
                                            + b"\0" * 64)
        result = vf.discover_view_factors(mesh, project_root=tmp)
        assert not result.matches
        assert "REBUILT" in result.refusals[0].refusal, (
            "the refusal must name the silent-rebuild consequence: "
            f"{result.refusals[0].refusal}")


def test_the_format_is_read_from_the_magic_number_and_not_the_extension():
    """In this project the two disagree, and it is not a corner case.

    ``~/projects/a3gt/campaign/base_vf/*.hmat`` are SPARSE version-1 files
    with zero nonzeros -- twenty-five of them, banked for the A3GT catalog --
    while the current empty stubs are H-matrix version 2 and are named ``.vf``.
    Both spellings are wrong for what the file contains, and a search that
    trusted the extension would describe every one of them backwards.

    This also pins the ``MFFV`` correction. ``MFFV`` is not a third format: it
    is ``VFFM``'s magic as it sits on disk, little-endian.
    """
    vf = _discovery()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        mesh = _ply(tmp / "site.ply", 1799459)
        # Byte for byte the shape of A03MA001.hmat, read 2026-08-31.
        legacy = _vffm(tmp / "legacy.hmat", rows=1799459, nnz=0, version=1)
        assert legacy.read_bytes()[:4] == b"MFFV", (
            "fixture problem: the on-disk spelling of the sparse magic is "
            "MFFV, which is the point of this check")
        candidate = vf.probe_view_factor(legacy, mesh_faces=1799459)
        assert candidate.fmt == vf.FORMAT_VFFM, (
            f"a .hmat-named sparse file was read as {candidate.fmt!r}")
        assert candidate.version == 1 and candidate.rank == vf.RANK_EMPTY
        joined = " ".join(candidate.notes)
        assert "version 1" in joined and "no facet-to-facet exchange" in joined, (
            f"it is BOTH an old generation and empty; both must be said: "
            f"{joined}")

        modern = _hmat(tmp / "modern.vf", rows=1799459, nnz=0)
        other = vf.probe_view_factor(modern, mesh_faces=1799459)
        assert other.fmt == vf.FORMAT_HMAT, (
            f"a .vf-named H-matrix was read as {other.fmt!r}")


def test_an_identical_mesh_copy_elsewhere_still_counts_as_this_mesh():
    """Otherwise rung 1 refuses every genuine match this project has.

    Every real A3GT provenance sidecar records the mesh under the repository's
    ``data/dem/a3gt_v2/``; every production configuration loads it from
    ``~/projects/a3gt/meshes/``. Measured 2026-08-31, those two copies are
    byte-identical. Matching on device and inode alone would therefore have
    demoted the one case the provenance tier was built for -- and it did,
    which is how this rung came to exist.

    What is promoted is a same-name, same-length copy, and the note says that
    the bytes were not compared. The engine's own geometry-hash check is what
    settles it.
    """
    vf = _discovery()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        here, there = tmp / "here", tmp / "there"
        here.mkdir()
        there.mkdir()
        wired = _ply(here / "site.ply", 64)
        copy = _ply(there / "site.ply", 64)
        assert wired.stat().st_size == copy.stat().st_size
        csr = _csr(here / "m.csr", rows=64, nnz=8, geomhash=42.25)
        _sidecar(csr, copy, faces=64, geomhash=42.25)
        candidate = vf.probe_view_factor(csr, mesh_faces=64,
                                         mesh_path=str(wired))
        assert candidate.rank == vf.RANK_PROVENANCE, (
            "a byte-identical copy under another path must still count as "
            f"this mesh; got {candidate.rank_label}")
        joined = " ".join(candidate.notes)
        assert "bytes themselves were not compared" in joined, (
            f"the weaker rung must say what it did NOT check: {joined}")

        # ...and a different-sized file of the same name must NOT be promoted.
        # 640000 rather than 65: this PLY fixture's length depends only on the
        # DIGIT COUNT of the numbers in its header, so a 65-face file is
        # byte-for-byte as long as a 64-face one and would have proved nothing.
        _ply(there / "site.ply", 640000)
        assert (there / "site.ply").stat().st_size != wired.stat().st_size
        again = vf.probe_view_factor(csr, mesh_faces=64, mesh_path=str(wired))
        assert again.rank == vf.RANK_CSR_ROWS, (
            "same name is not enough on its own")


def test_a_truncated_csr_is_refused_rather_than_offered():
    """This format has no magic number; its arithmetic is its identity.

    A CSR cut short -- an interrupted copy, a partial download -- declares
    dimensions its length cannot hold. The engine aborts on the same
    shortfall, one array later.
    """
    vf = _discovery()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        mesh = _ply(tmp / "site.ply", 64)
        full = _csr(tmp / "cut.csr", rows=64, nnz=8)
        raw = full.read_bytes()
        full.write_bytes(raw[:-4])
        result = vf.discover_view_factors(mesh, project_root=tmp)
        assert not result.matches
        assert "truncated" in result.refusals[0].refusal


def test_the_five_orphaned_matrices_are_refused_by_name():
    """PTM-312: loadable, correctly dimensioned, and known to be wrong.

    Nothing in these files says so -- current layout, right face count -- and
    the sweeps that would let them be rebuilt were in ``/tmp`` and are gone.
    A discovery mechanism that ranked them like anything else would be the
    quiet path back to a defect that is already on the ledger.
    """
    vf = _discovery()
    from gui_core.curated_sites import DEFECTIVE_VIEW_FACTORS, NOT_CURATED
    defective = vf.known_defective_view_factors()
    assert len(defective) == 5, defective
    assert defective == DEFECTIVE_VIEW_FACTORS, (
        "the search must read the registry's table, not keep a second copy: "
        "two lists of which matrices are defective is one list that goes "
        "stale")
    assert len(NOT_CURATED) == len(DEFECTIVE_VIEW_FACTORS), (
        "the excluded sites and the distrusted matrices are the same five "
        "facts and must not drift apart")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        mesh = _ply(tmp / "site.ply", 64)
        _csr(tmp / "A03MP016_corrected_vf.csr", rows=64, nnz=8)
        result = vf.discover_view_factors(mesh, project_root=tmp)
        assert not result.matches, "a PTM-312 matrix was offered"
        assert "PTM-312" in result.refusals[0].refusal


def test_a_pruned_matrix_is_refused_with_its_own_receipt():
    """The receipt beside it records a MEASURED failure. Read it, quote it.

    The fixture uses ``radiance_prune_eps_Wm2``, which is the receipt's real
    key -- the same string ``radiation_exchange.cpp:399`` greps for, and the
    one in ``A03MP026_rad_demo.csr.prune_receipt.json`` on this machine. An
    earlier draft of both the module and this fixture used the PROVENANCE
    sidecar's key, ``radiance_prune_eps``, and agreed with itself: the refusal
    printed "eps=?" against every real receipt and no test could see it.
    """
    vf = _discovery()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        mesh = _ply(tmp / "site.ply", 64)
        pruned = _csr(tmp / "rad_demo.csr", rows=64, nnz=8)
        Path(str(pruned) + ".prune_receipt.json").write_text(json.dumps({
            "radiance_prune_eps_Wm2": 0.001,
            "MEASURED_AGT_counter_example":
                "MEASURED FAIL: PSR-floor Tmax mean 3.3 K / max 22.6 K cold"}))
        result = vf.discover_view_factors(mesh, project_root=tmp)
        assert not result.matches
        why = result.refusals[0].refusal
        assert "MEASURED FAIL" in why
        assert "0.001" in why and "?" not in why, (
            f"the pruning threshold must be reported, not rendered as '?': "
            f"{why}")


def test_a_prune_recorded_only_in_the_provenance_sidecar_is_still_caught():
    """Two schemas record the same thing, and only one is a separate file.

    A matrix assembled through the ordinary pipeline carries
    ``radiance_prune_eps`` in its provenance sidecar -- 0.0 when nothing was
    discarded -- and no receipt at all. Reading only the receipt would offer
    it as an ordinary candidate.
    """
    vf = _discovery()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        mesh = _ply(tmp / "site.ply", 64)
        pruned = _csr(tmp / "pruned.csr", rows=64, nnz=8, geomhash=42.25)
        _sidecar(pruned, mesh, faces=64, geomhash=42.25,
                 radiance_prune_eps=1e-3)
        whole = _csr(tmp / "whole.csr", rows=64, nnz=8, geomhash=42.25)
        _sidecar(whole, mesh, faces=64, geomhash=42.25,
                 radiance_prune_eps=0.0)
        result = vf.discover_view_factors(mesh, project_root=tmp)
        assert [c.path.name for c in result.matches] == ["whole.csr"], (
            f"a pruned matrix was offered: "
            f"{[c.path.name for c in result.matches]}")
        assert "22.6 K" in result.refusals[0].refusal, (
            f"the refusal must carry the measured consequence: "
            f"{result.refusals[0].refusal}")


def test_a_matrix_this_project_calls_superseded_never_sorts_first():
    """Measured, and it is why this rule exists.

    In the real A03MP026 directory ``..._corrected_vf_divisor2.csr`` and
    ``..._corrected_vf_v3.csr`` BOTH reach the provenance-verified tier: their
    sidecars agree on the estimator, the ray count, the geometry hash and the
    mesh, and differ only in a timestamp and a nonzero count that nothing here
    ranks on. Ordering by rank and then by path therefore put the superseded
    one first -- where a default selection picks it up. The site registry
    already records which is which; this consults it.
    """
    from gui_core.curated_sites import CURATED_SITES, site_status
    vf = _discovery()
    known = vf.superseded_view_factors()
    assert known, "the registry's superseded list did not reach the search"

    entry = None
    for candidate in CURATED_SITES:
        if site_status(candidate).ready:
            entry = candidate
            break
    if entry is None:
        raise Skip("no curated site's files are on this machine")

    directory = Path(entry.view_factors.replace_csr).parent
    result = vf.discover_view_factors(
        entry.mesh.path,
        view_factor_file=str(directory / "anything.vf"))
    assert result.best is not None, result.summary()
    assert result.best.path == Path(entry.view_factors.replace_csr), (
        f"the first offer for {entry.key} is {result.best.path.name}; the "
        f"registry wires {Path(entry.view_factors.replace_csr).name}")
    demoted = [c for c in result.matches if c.superseded]
    assert demoted, (
        "the superseded matrices in this directory were not recognized; the "
        "check cannot fail and proves nothing")
    for candidate in demoted:
        assert "SUPERSEDED" in candidate.headline()
        assert result.matches.index(candidate) > result.matches.index(
            result.best)


def test_the_search_also_looks_where_the_cache_field_points():
    """A user who set that key once keeps their matrices together."""
    vf = _discovery()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        meshes, caches = tmp / "meshes", tmp / "caches"
        meshes.mkdir()
        caches.mkdir()
        mesh = _ply(meshes / "site.ply", 64)
        _vffm(caches / "elsewhere.vf", rows=64, nnz=99)
        alone = vf.discover_view_factors(mesh, project_root=tmp)
        assert not alone.matches, "nothing is beside the mesh itself"
        together = vf.discover_view_factors(
            mesh, view_factor_file=str(caches / "current.vf"),
            project_root=tmp)
        assert [c.path.name for c in together.matches] == ["elsewhere.vf"]
        assert len(together.directories) == 2, together.directories


def test_discovery_writes_nothing_and_selects_nothing():
    """The whole point. It returns a list; the user decides.

    Checked two ways: the directory is byte-identical afterwards, and the
    wiring helper -- the only thing that produces keys -- is a pure function
    that has to be CALLED with a chosen candidate.
    """
    vf = _discovery()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        mesh = _ply(tmp / "site.ply", 64)
        _vffm(tmp / "a.vf", rows=64, nnz=10)
        before = {p.name: p.stat().st_mtime_ns for p in tmp.iterdir()}
        result = vf.discover_view_factors(mesh, project_root=tmp)
        after = {p.name: p.stat().st_mtime_ns for p in tmp.iterdir()}
        assert before == after, "the search modified the directory it scanned"
        assert result.matches, "fixture problem: nothing was found"
        assert vf.candidate_wiring(None) == []


def test_wiring_a_csr_never_pretends_it_has_a_base_file():
    """A CSR alone is half a wiring, and the half that is missing REBUILDS.

    ``PTM_HMATRIX_VF_REPLACE`` installs the matrix, and the normal loader
    still runs: with no cache file it finds nothing and starts a multi-gigabyte
    build. So the helper returns the environment line alone and it is the
    caller's job to say the base is missing -- inventing a stub path here
    would be a file this function claimed exists.
    """
    vf = _discovery()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        mesh = _ply(tmp / "site.ply", 64)
        csr = _csr(tmp / "m.csr", rows=64, nnz=8)
        candidate = vf.probe_view_factor(csr, mesh_faces=64, mesh_path=str(mesh))
        alone = vf.candidate_wiring(candidate)
        assert alone == [("engine_env.PTM_HMATRIX_VF_REPLACE", str(csr))], alone
        with_stub = vf.candidate_wiring(candidate, stub_path=str(tmp / "b.vf"))
        assert with_stub[0][0] == "radiation.view_factor_file"
        assert with_stub[1][0] == "engine_env.PTM_HMATRIX_VF_REPLACE"

        vffm = _vffm(tmp / "m.vf", rows=64, nnz=10)
        plain = vf.probe_view_factor(vffm, mesh_faces=64)
        assert vf.candidate_wiring(plain) == [
            ("radiation.view_factor_file", str(vffm))], (
            "a .vf goes into the cache field and nowhere else")


def test_a_mesh_whose_face_count_is_unreadable_ranks_nothing():
    """No count, no judgment -- and it says so rather than offering anyway."""
    vf = _discovery()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        mesh = tmp / "site.vtk"
        mesh.write_bytes(b"# vtk DataFile Version 5.1\n")
        _vffm(tmp / "a.vf", rows=64, nnz=10)
        result = vf.discover_view_factors(mesh, project_root=tmp)
        assert result.mesh_faces is None
        assert "could not be read" in result.summary(), result.summary()
        assert all(c.rank != vf.RANK_REFUSED or c.refusal
                   for c in result.matches + result.refusals)


def test_the_real_a3gt_matrices_rank_the_way_the_registry_claims():
    """Against the real files, not fixtures -- when they are on this machine.

    The synthetic checks above prove the RULES. This proves the rules meet the
    artifacts: each curated site's own matrix reaches the provenance-verified
    tier against its own mesh, and its base stub is recognized as the empty
    stub it is rather than as a matrix.
    """
    from gui_core.curated_sites import CURATED_SITES, site_status
    vf = _discovery()
    checked = 0
    for entry in CURATED_SITES:
        status = site_status(entry)
        if not status.ready:
            continue
        checked += 1
        candidate = vf.probe_view_factor(
            entry.view_factors.replace_csr,
            mesh_faces=entry.mesh.faces,
            mesh_path=entry.mesh.path)
        assert candidate.rank == vf.RANK_PROVENANCE, (
            f"{entry.key}: its own matrix should be provenance-verified "
            f"against its own mesh; got rank {candidate.rank} "
            f"({candidate.refusal or candidate.rank_label})")
        assert candidate.nnz == entry.view_factors.nnz, (
            f"{entry.key}: the registry records {entry.view_factors.nnz:,} "
            f"nonzeros and the file declares {candidate.nnz:,}")
        stub = vf.probe_view_factor(entry.view_factors.base_stub,
                                    mesh_faces=entry.mesh.faces)
        assert stub.rank == vf.RANK_EMPTY, (
            f"{entry.key}: the base stub must be recognized as EMPTY, not "
            f"offered as a matrix (rank {stub.rank})")
    if not checked:
        raise Skip("no curated A3GT site's files are on this machine")


CHECKS = [
    test_a_rows_mismatch_is_refused_and_names_both_numbers,
    test_a_provenance_sidecar_naming_this_mesh_outranks_a_bare_match,
    test_a_sidecar_for_a_different_mesh_does_not_promote_it,
    test_a_sidecar_that_disagrees_with_its_own_matrix_is_not_used,
    test_an_hmatrix_always_declares_its_physics_generation_unverifiable,
    test_a_version_one_sparse_file_is_offered_and_warned_about,
    test_an_empty_matrix_is_not_offered_as_a_matrix,
    test_a_cloud_placeholder_is_refused_before_anything_reads_it,
    test_a_file_that_is_not_a_view_factor_matrix_is_refused_by_name,
    test_the_format_is_read_from_the_magic_number_and_not_the_extension,
    test_an_identical_mesh_copy_elsewhere_still_counts_as_this_mesh,
    test_a_truncated_csr_is_refused_rather_than_offered,
    test_the_five_orphaned_matrices_are_refused_by_name,
    test_a_pruned_matrix_is_refused_with_its_own_receipt,
    test_a_prune_recorded_only_in_the_provenance_sidecar_is_still_caught,
    test_a_matrix_this_project_calls_superseded_never_sorts_first,
    test_the_search_also_looks_where_the_cache_field_points,
    test_discovery_writes_nothing_and_selects_nothing,
    test_wiring_a_csr_never_pretends_it_has_a_base_file,
    test_a_mesh_whose_face_count_is_unreadable_ranks_nothing,
    test_the_real_a3gt_matrices_rank_the_way_the_registry_claims,
]


def _tracked_state():
    git = "/Library/Developer/CommandLineTools/usr/bin/git"
    if not Path(git).exists():                            # pragma: no cover
        return None
    out = subprocess.run([git, "-C", str(REPO_ROOT), "status", "--porcelain"],
                         capture_output=True, text=True)
    return out.stdout


def main():
    failed = skipped = 0
    before = _tracked_state()
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
        except Exception as exc:                          # noqa: BLE001
            failed += 1
            import traceback
            print(f"ERROR {name}: {exc!r}")
            traceback.print_exc()
        else:
            print(f"PASS {name}")
    after = _tracked_state()
    if before is not None and after != before:
        failed += 1
        print("FAIL the suite left the working tree dirty; these checks must "
              f"not write into the repository:\n  before:\n{before}"
              f"  after:\n{after}")
    total = len(CHECKS)
    print(f"\n{total - failed - skipped}/{total} passed, {skipped} skipped, "
          f"{failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
