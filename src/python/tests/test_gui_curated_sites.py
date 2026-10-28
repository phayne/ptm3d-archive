#!/usr/bin/env python3
"""The lunar-site tier: a set that ran, or nothing at all.

**Why this file exists.** A curated site is three files -- a core terrain, a
shadow-only occluder ring, and a view-factor matrix built for that terrain --
and every way of getting the combination wrong produces a run that FINISHES.
The campaign this registry is drawn from produced, at various times: a matrix
with zero stored nonzeros that loaded clean and computed no radiation, a matrix
for a mesh that had since been regenerated, three radiance-pruned experiments
whose own receipts record measured failure, an occluder ring chosen for a
different job, and one accidental rebuild that hard-wedged the graphics cards.

So the checks here are not "does the table parse". They are:

  * every number in the table is re-read off the file it describes
  * every path the table REFUSES is still checked, so a refusal cannot rot
    into a path that quietly started working
  * the placement keys are the not-placed sentinel, because the alternative
    rotates an already-placed mesh and leaves its ring behind
  * a site whose files are not on this computer is shown, refuses, and writes
    nothing
  * what a selection emits parses in the ENGINE, with the environment intact

**Two registries, on purpose.** The shipped table is empty -- the real entries
are landing-site candidates and live in a local overlay outside this repository
(``gui_core.curated_sites``, "The local overlay"). So the checks split:

  * MACHINERY checks run against SYNTHETIC fixture sites (``synthetic_sites``),
    written to a temporary directory and loaded through the overlay's own
    reader. They run everywhere, including on a checkout with no overlay, which
    is more coverage than the hard-coded table gave -- those checks used to
    skip entirely on any machine without the artifacts.
  * DISK-ANCHORED checks ("every recorded number is re-read off its own file")
    run against the real overlay when one is installed, and skip BY NAME when
    it is not, rather than passing vacuously.

Run directly::

    python3 src/python/tests/test_gui_curated_sites.py
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "python"))
sys.path.insert(0, str(REPO_ROOT / "src" / "python" / "tests"))
sys.path.insert(0, str(REPO_ROOT / "gui"))


class Skip(Exception):
    """A check that cannot run here, as distinct from one that failed."""


def _sites():
    try:
        from gui_core import curated_sites
    except ImportError as exc:                            # pragma: no cover
        raise Skip(f"the GUI's site registry is unavailable here ({exc})")
    return curated_sites


@contextlib.contextmanager
def _synthetic():
    """The machinery checks' registry: invented sites, on real temp files.

    Yields ``(module, payload)`` with the synthetic overlay loaded, and puts
    the machine's own overlay back afterwards whatever happens -- these tests
    run in one process and a fixture left installed would make every later
    check read the wrong table.
    """
    import synthetic_sites
    sites = _sites()
    with tempfile.TemporaryDirectory() as tmp:
        payload = synthetic_sites.install(sites, tmp)
        try:
            yield sites, payload
        finally:
            synthetic_sites.restore(sites)


def _real_overlay():
    """The module with the machine's own overlay, or a named skip.

    The subject of the disk-anchored checks is whether the registry has drifted
    from artifacts ON THIS DISK. With no overlay there are no artifacts to have
    drifted from, and a synthetic stand-in would be checking the fixture
    generator against itself.
    """
    sites = _sites()
    sites.reload_registry()
    status = sites.local_registry_status()
    if not status.ok:
        raise Skip(f"the local site overlay is faulty: {status.summary()}")
    if not sites.CURATED_SITES:
        raise Skip(f"no local lunar-site overlay on this machine "
                   f"({status.path}), so there are no real artifacts to "
                   f"re-measure the registry against")
    return sites


def _on_disk(entry):
    """The entry if its whole set is here, else a named skip."""
    sites = _sites()
    status = sites.site_status(entry)
    if not status.ready:
        raise Skip(f"{entry.key} is not on this machine: {status.summary()}")
    return status


#: The readers below deliberately duplicate what ``gui_core.vf_discovery``
#: does. They are INDEPENDENT: a suite that checked the registry through the
#: module the registry feeds would be checking two halves of one implementation
#: against each other and would agree with itself through any shared mistake.
#: Do not "simplify" these into calls to `probe_view_factor`.


def _ply_header(path):
    """``(vertices, faces)`` from a PLY header. Header only."""
    with open(path, "rb") as handle:
        text = handle.read(65536).decode("ascii", errors="replace")
    vertices = faces = None
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) == 3 and parts[0] == "element":
            if parts[1] == "vertex":
                vertices = int(parts[2])
            elif parts[1] == "face":
                faces = int(parts[2])
        if line.strip() == "end_header":
            break
    return vertices, faces


def _csr_header(path):
    """``(n, geomhash, nnz)`` from a replacement CSR. Two small reads.

    ``radiation_exchange.cpp:276``. ``nnz`` sits after the index array, so it
    is one seek rather than a read of nineteen gigabytes.
    """
    with open(path, "rb") as handle:
        head = handle.read(12)
        n, = struct.unpack("<i", head[:4])
        geomhash, = struct.unpack("<d", head[4:12])
        handle.seek(12 + 8 * (n + 1))
        nnz, = struct.unpack("<q", handle.read(8))
    return n, geomhash, nnz


def _vf_stub_header(path):
    """``(magic, version, rows, nnz)`` from an 80-byte H-matrix stub."""
    data = Path(path).read_bytes()
    magic, version = struct.unpack("<II", data[:8])
    rows, _cols = struct.unpack("<QQ", data[8:24])
    nnz, = struct.unpack("<Q", data[-8:])
    return magic, version, rows, nnz


# --------------------------------------------------------------------------
# The table against the files
# --------------------------------------------------------------------------

def test_every_recorded_number_is_re_read_off_its_own_file():
    """The table is a claim about files. This re-measures every claim.

    Face counts from the PLY headers, the matrix's rows/nonzeros/geometry hash
    from the CSR itself, and the stub's format and emptiness from its 80 bytes.
    A registry that has drifted from its files is worse than no registry: it
    reads as checked.
    """
    sites = _real_overlay()
    checked = []
    for entry in sites.CURATED_SITES:
        _on_disk(entry)
        checked.append(entry.key)

        _v, faces = _ply_header(entry.mesh.path)
        assert faces == entry.mesh.faces, (
            f"{entry.key}: the registry says {entry.mesh.faces:,} core faces "
            f"and the PLY header says {faces:,}")
        _v, ring_faces = _ply_header(entry.occluder.path)
        assert ring_faces == entry.occluder.faces, (
            f"{entry.key}: the registry says {entry.occluder.faces:,} ring "
            f"faces and the PLY header says {ring_faces:,}")

        n, geomhash, nnz = _csr_header(entry.view_factors.replace_csr)
        assert n == entry.view_factors.n_faces == entry.mesh.faces, (
            f"{entry.key}: the matrix is {n:,} rows, the registry says "
            f"{entry.view_factors.n_faces:,}, the mesh has "
            f"{entry.mesh.faces:,}. All three have to be one number.")
        assert nnz == entry.view_factors.nnz, (
            f"{entry.key}: the matrix declares {nnz:,} nonzeros and the "
            f"registry records {entry.view_factors.nnz:,}")
        assert geomhash == entry.view_factors.geomhash, (
            f"{entry.key}: the matrix header's geometry hash is {geomhash!r} "
            f"and the registry records {entry.view_factors.geomhash!r}")
        size = os.path.getsize(entry.view_factors.replace_csr)
        assert size == entry.view_factors.disk_bytes(), (
            f"{entry.key}: {n:,} rows and {nnz:,} nonzeros account for "
            f"{entry.view_factors.disk_bytes():,} bytes; the file is "
            f"{size:,}. It is truncated, or the format changed.")

        magic, version, rows, stub_nnz = _vf_stub_header(
            entry.view_factors.base_stub)
        assert magic == 0x484D4154 and version == 2, (
            f"{entry.key}: the base stub must be the CURRENT H-matrix format "
            f"(magic 0x484D4154, version 2); it is 0x{magic:08x} v{version}. "
            f"A stale-format base fails to load and triggers the full rebuild "
            f"this pairing exists to avoid.")
        assert rows == entry.mesh.faces, (
            f"{entry.key}: the stub is dimensioned for {rows:,} faces and the "
            f"mesh has {entry.mesh.faces:,}")
        assert stub_nnz == 0, (
            f"{entry.key}: the base stub must be EMPTY -- a real matrix here "
            f"leaves two matrices live at once and the solver reads the "
            f"uncorrected one (radiation_exchange.cpp:370)")
    if not checked:
        raise Skip("no curated site's files are on this machine")


def test_the_provenance_sidecar_agrees_with_the_matrix_it_describes():
    """The sidecar is the only artifact that says which MESH a matrix is for.

    So it has to be there, it has to name a mesh, and its numbers have to
    match the matrix beside it -- otherwise the strongest identity check this
    project has is a file that agrees with nothing.
    """
    sites = _real_overlay()
    checked = 0
    for entry in sites.CURATED_SITES:
        _on_disk(entry)
        sidecar = Path(entry.view_factors.provenance)
        assert sidecar.is_file(), (
            f"{entry.key}: {sidecar} is missing. A curated matrix without "
            f"provenance cannot be identity-checked against its mesh.")
        record = json.loads(sidecar.read_text())
        checked += 1
        assert record["n_faces"] == entry.mesh.faces
        assert record["nnz"] == entry.view_factors.nnz
        assert record["geomhash"] == entry.view_factors.geomhash
        assert Path(record["mesh"]).name == Path(entry.mesh.path).name, (
            f"{entry.key}: the sidecar was built for "
            f"{Path(record['mesh']).name} and the registry wires "
            f"{Path(entry.mesh.path).name}")
        assert record.get("sweep_coverage_missing") == 0, (
            f"{entry.key}: the sweep that built this matrix did not cover "
            f"every face")
        assert record.get("allow_incomplete_sweep") is False
    if not checked:
        raise Skip("no curated site's files are on this machine")


def test_the_mesh_really_sits_where_the_registry_says_it_does():
    """And therefore must NOT be rotated by a placement override.

    Measured from the PLY's own vertices, the way the engine measures it:
    the vertex mean, then its direction. If this ever stopped agreeing with
    the recorded coordinates, the registry's whole reason for writing the
    not-placed sentinel would have changed and somebody has to look.
    """
    sites = _real_overlay()
    try:
        import numpy
    except ImportError as exc:                            # pragma: no cover
        raise Skip(f"numpy is unavailable here ({exc})")
    dtype = numpy.dtype([("x", "<f8"), ("y", "<f8"), ("z", "<f8"),
                         ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
                         ("lat", "<f8"), ("lon", "<f8"), ("elev", "<f8")])
    checked = 0
    for entry in sites.CURATED_SITES:
        _on_disk(entry)
        with open(entry.mesh.path, "rb") as handle:
            buffer = b""
            while b"end_header" not in buffer:
                chunk = handle.read(4096)
                if not chunk:
                    break
                buffer += chunk
        if b"end_header" not in buffer:                   # pragma: no cover
            raise Skip(f"{entry.key}: the PLY header could not be found")
        offset = buffer.index(b"end_header") + len(b"end_header")
        while buffer[offset:offset + 1] in (b"\r", b"\n"):
            offset += 1
        vertices, _faces = _ply_header(entry.mesh.path)
        if b"binary_little_endian" not in buffer:         # pragma: no cover
            raise Skip(f"{entry.key}: this check reads binary PLY only")
        points = numpy.fromfile(entry.mesh.path, dtype=dtype, count=vertices,
                                offset=offset)
        cx = float(points["x"].mean())
        cy = float(points["y"].mean())
        cz = float(points["z"].mean())
        radius = math.sqrt(cx * cx + cy * cy + cz * cz)
        latitude = math.degrees(math.asin(cz / radius))
        longitude = math.degrees(math.atan2(cy, cx))
        checked += 1
        assert abs(latitude - entry.mesh.latitude_deg) < 0.01, (
            f"{entry.key}: the mesh's own vertex mean is at latitude "
            f"{latitude:.4f} and the registry records "
            f"{entry.mesh.latitude_deg}. This mesh is not where the entry "
            f"says it is.")
        assert abs(longitude - entry.mesh.longitude_deg) < 0.01, (
            f"{entry.key}: the mesh sits at longitude {longitude:.4f}, the "
            f"registry records {entry.mesh.longitude_deg}")
        assert abs(latitude - entry.mesh.measured_latitude_deg) < 1e-3
        assert abs(longitude - entry.mesh.measured_longitude_deg) < 1e-3
    if not checked:
        raise Skip("no curated site's files are on this machine")


def test_selecting_a_site_never_asks_the_engine_to_move_the_mesh():
    """The quietest wrong thing this registry could do, closed by test.

    ``mesh.latitude_override > -900`` makes the engine rotate the CORE mesh so
    its vertex mean lands on the given point -- and the occluder ring is not
    rotated with it. These DEMs are already at their coordinates, so writing
    the rounded name back would rotate by a fraction of a degree, move the
    core about twelve meters relative to its ring, and leave the matrix's
    geometry-hash gate green throughout, because a rotation does not change
    face areas.

    Every production configuration for these sites leaves both keys unset.
    """
    with _synthetic() as (sites, _payload):
        assert sites.CURATED_SITES, "the fixture registry is empty"
        for entry in sites.CURATED_SITES:
            wiring = dict(sites.site_wiring(entry))
            assert wiring["mesh.latitude_override"] == sites.NOT_PLACED, (
                f"{entry.key}: the wiring writes latitude "
                f"{wiring['mesh.latitude_override']!r}, which the engine reads "
                f"as 'rotate this mesh'")
            assert wiring["mesh.longitude_override"] == sites.NOT_PLACED
            assert sites.NOT_PLACED < -900.0, (
                "the sentinel has to be below the engine's own -900 threshold")


def test_the_wiring_is_the_whole_set_or_nothing():
    """A partial set is the failure mode, not a lesser version of success."""
    with _synthetic() as (sites, _payload):
        assert sites.CURATED_SITES, "the fixture registry is empty"
        _assert_whole_wiring(sites)


def _assert_whole_wiring(sites):
    for entry in sites.CURATED_SITES:
        wiring = dict(sites.site_wiring(entry))
        assert wiring["mesh.import_file"] == entry.mesh.path
        assert wiring["mesh.occluder_file"] == entry.occluder.path
        assert wiring["radiation.enabled"] is True
        assert wiring["radiation.view_factor_file"] == \
            entry.view_factors.base_stub, (
            f"{entry.key}: the cache field must hold the EMPTY BASE. With it "
            f"unset the loader finds no cache and rebuilds the matrix the "
            f"environment variable was about to hand it.")
        assert wiring["engine_env.PTM_HMATRIX_VF_REPLACE"] == \
            entry.view_factors.replace_csr
        assert wiring["simulation.body"] == entry.body


def test_the_registry_refuses_the_matrices_it_says_it_refuses():
    """A refusal that rotted into a working path would be invisible.

    Each superseded artifact is named with a reason. This checks the reasons
    are about real files -- and, for the ones that still exist, that NONE of
    them is what any entry actually wires.

    Run against BOTH registries: the synthetic one proves the rule is enforced
    at all, and the machine's own overlay proves this machine's entries obey
    it. The second is the one that catches a refusal that has rotted.
    """
    with _synthetic() as (sites, _payload):
        _assert_refusals_hold(sites, minimum=5)
    try:
        sites = _real_overlay()
    except Skip:
        return
    _assert_refusals_hold(sites, minimum=1)


def _assert_refusals_hold(sites, minimum):
    wired = set()
    for entry in sites.CURATED_SITES:
        wired.add(entry.view_factors.replace_csr)
        wired.add(entry.view_factors.base_stub)
    named = 0
    for entry in sites.CURATED_SITES:
        for item in entry.superseded:
            named += 1
            assert item.path not in wired, (
                f"{entry.key} lists {item.path} as NOT used and something in "
                f"this registry wires it")
            assert item.why.strip(), f"{item.path} is refused with no reason"
            if not os.path.exists(item.path):
                assert item.exists_note, (
                    f"{entry.key}: {item.path} is not on this disk and the "
                    f"entry does not say it was removed on purpose. A "
                    f"refusal for a path that no longer exists is a refusal "
                    f"nobody can check.")
    assert named >= minimum, (
        "the superseded list is the part of this registry that keeps somebody "
        "from picking the bigger file in the same folder; it should not be "
        f"nearly empty (found {named})")


def test_the_pruned_experiments_are_refused_and_their_receipts_say_why():
    """Their own receipts record MEASURED failure, in this project's words."""
    with _synthetic() as (sites, _payload):
        assert _count_failing_receipts(sites) >= 1, (
            "the fixture set has no prune receipt, so this check has no "
            "subject")
    try:
        sites = _real_overlay()
    except Skip:
        return
    _count_failing_receipts(sites)


def _count_failing_receipts(sites):
    checked = 0
    for entry in sites.CURATED_SITES:
        for item in entry.superseded:
            receipt = Path(item.path + ".prune_receipt.json")
            if not receipt.is_file():
                continue
            checked += 1
            body = receipt.read_text()
            assert "FAIL" in body.upper(), (
                f"{receipt} does not record a failure, and the registry says "
                f"it does")
    return checked


def test_the_orphaned_sites_are_not_curated_and_say_why():
    """Defective offline view-factor matrices. The meshes survive; the matrices beside them do not.

    They are not curated, they must not become curated by accident, and the
    reason has to travel with the fact -- otherwise the next person sees sites
    with meshes and matrices on disk and no explanation for their absence from
    a table full of their neighbors.
    """
    with _synthetic() as (sites, payload):
        assert set(sites.NOT_CURATED) == set(payload["not_curated"]), (
            "the overlay's not-curated table did not reach the module intact")
        _assert_orphans_hold(sites)
    try:
        sites = _real_overlay()
    except Skip:
        return
    _assert_orphans_hold(sites)


def _assert_orphans_hold(sites):
    keys = {e.key for e in sites.CURATED_SITES}
    assert not (keys & set(sites.NOT_CURATED)), (
        "a site is both curated and explicitly not curated")
    for site, why in sites.NOT_CURATED.items():
        assert "PTM-312" in why, f"{site}'s exclusion cites no ledger row"
        assert sites.curated_site(site) is None, (
            f"{site} resolves to a curated entry and must not")


def test_a_machine_without_the_files_is_told_so_and_gets_no_wiring():
    """The local-only contract, exercised against an empty tree.

    These artifacts are not distributed. On another computer the entry has to
    be visible, honest, and inert -- not a set of paths handed to the launch
    check to discover.
    """
    with _synthetic() as (sites, _payload):
        _assert_absent_machine_refuses(sites, sites.CURATED_SITES[0])


def _assert_absent_machine_refuses(sites, entry):
    with tempfile.TemporaryDirectory() as tmp:
        moved = sites.CuratedSite(
            key=entry.key, name=entry.name, body=entry.body,
            region=entry.region,
            mesh=sites.CuratedSiteMesh(
                path=os.path.join(tmp, "core.ply"), faces=entry.mesh.faces,
                latitude_deg=entry.mesh.latitude_deg,
                longitude_deg=entry.mesh.longitude_deg,
                core_radius_km=entry.mesh.core_radius_km, source="test"),
            occluder=sites.CuratedSiteOccluder(
                path=os.path.join(tmp, "ring.ply"),
                faces=entry.occluder.faces,
                r_ext_km=entry.occluder.r_ext_km, role="test"),
            view_factors=sites.CuratedSiteViewFactors(
                replace_csr=os.path.join(tmp, "m.csr"),
                base_stub=os.path.join(tmp, "b.vf"),
                n_faces=entry.mesh.faces, nnz=1, geomhash=1.0,
                generation="test", provenance="", campaign="test",
                gpu_fit="test"),
            why_curated="test", local_only=True,
            local_root=os.path.join(tmp, "not_here"))
        status = sites.site_status(moved)
        assert not status.ready
        assert not status.machine_has_root
        summary = status.summary()
        assert "not on this computer" in summary, summary
        assert "does not distribute" in summary, summary
        assert "Nothing has been written" in summary, (
            f"a refusal must say the configuration was left alone: {summary}")
        assert not sites.site_status(moved).ready


def test_a_cloud_placeholder_in_the_set_refuses_the_whole_site():
    """Reading one blocks the run. It is a refusal, not a warning."""
    with _synthetic() as (sites, _payload):
        entry = sites.CURATED_SITES[0]
        real_stat = sites.os.stat

        class _Evicted:
            def __init__(self, real):
                self.st_size = real.st_size
                self.st_blocks = 0

        def fake_stat(path, *args, **kwargs):
            result = real_stat(path, *args, **kwargs)
            if str(path) == entry.mesh.path:
                return _Evicted(result)
            return result

        sites.os.stat = fake_stat
        try:
            status = sites.site_status(entry)
        finally:
            sites.os.stat = real_stat
        assert not status.ready and status.dataless
        assert "cloud placeholder" in status.summary()
        assert "Nothing has been written" in status.summary()


def test_a_truncated_matrix_is_caught_by_the_registrys_own_arithmetic():
    """Rows and nonzeros determine the byte length exactly. Use that.

    A partial copy of a nineteen-gigabyte matrix is a real thing to have on
    disk, and it is indistinguishable from a good one by every other cheap
    check.
    """
    sites = _sites()
    with tempfile.TemporaryDirectory() as tmp:
        stub = Path(tmp) / "b.vf"
        stub.write_bytes(b"\0" * 80)
        short = Path(tmp) / "m.csr"
        short.write_bytes(b"\0" * 128)
        for name in ("core.ply", "ring.ply"):
            (Path(tmp) / name).write_text("ply\n")
        probe = sites.CuratedSite(
            key="TEST", name="test", body="Moon", region="test",
            mesh=sites.CuratedSiteMesh(str(Path(tmp) / "core.ply"), 10, -80.0,
                                       0.0, 3.0, "test"),
            occluder=sites.CuratedSiteOccluder(str(Path(tmp) / "ring.ply"),
                                               10, 20.0, "test"),
            view_factors=sites.CuratedSiteViewFactors(
                replace_csr=str(short), base_stub=str(stub), n_faces=10,
                nnz=1000, geomhash=1.0, generation="test", provenance="",
                campaign="test", gpu_fit="test"),
            why_curated="test", local_only=False, local_root="")
        status = sites.site_status(probe)
        assert not status.ready and status.mismatched
        assert "imply" in status.summary(), status.summary()


def test_the_provenance_text_names_the_ring_that_was_not_chosen():
    """A site can have three rings, and a campaign use two of them, for two
    different jobs. A card that showed only the one it wired would leave the
    user unable to reconcile it with the project's own notes -- which name the
    other one in the same breath as this matrix.
    """
    with _synthetic() as (sites, _payload):
        entry = sites.curated_site("SYN-ALPHA01")
        assert entry is not None
        text = " ".join(sites.wiring_provenance(entry))
        assert "27 km" in text, text
        assert "81 km" in text, (
            "the ring used for the offline sweep has to be named, or the "
            "entry reads as if it disagreed with the project's own record")
        assert "NOT used" in text
        assert "not placed" in text, (
            "the placement decision is the least obvious thing this entry "
            "does and it has to be on the card")


def test_the_coarse_site_declares_its_resolution_gap():
    """A 200x coarser site is a data gap, not an omission, and must say so."""
    with _synthetic() as (sites, _payload):
        entry = sites.curated_site("SYN-GAMMA03")
        assert entry is not None
        text = " ".join(sites.wiring_provenance(entry))
        assert "LOLA" in text and "200x" in text, text
        assert entry.mesh.faces < \
            sites.curated_site("SYN-ALPHA01").mesh.faces / 100


def test_the_lookup_will_not_match_one_site_for_another():
    """A site key is exact. A fuzzy match wires the wrong polar site.

    The near-miss arms are the point: a prefix, a suffix, a sibling key and a
    right key on the wrong body all have to miss. Two polar sites in one
    catalog are tens of kilometers apart with different illumination, and the
    wrong one resolves to a run that completes.
    """
    with _synthetic() as (sites, _payload):
        assert sites.curated_site("SYN-ALPHA01") is not None
        assert sites.curated_site("ALPHA01") is not None, "alias lookup"
        assert sites.curated_site("syn-alpha01") is not None, "case folding"
        assert sites.curated_site("synalpha01") is not None, "punctuation"
        assert sites.curated_site("SYN-ALPHA0") is None, "prefix must miss"
        assert sites.curated_site("SYN-ALPHA011") is None, "suffix must miss"
        assert sites.curated_site("SYN-BETA02") is not None
        assert sites.curated_site("SYN-ORPHAN01") is None, (
            "a not-curated key must not resolve to an entry")
        assert sites.curated_site("SYN-ALPHA01", body="Europa") is None


# --------------------------------------------------------------------------
# What a selection emits, in the engine's own words
# --------------------------------------------------------------------------

def _engine():
    from ptm import paths
    exe = paths.find_engine_executable("ptm_thermal_sim",
                                       search_roots=[REPO_ROOT])
    if exe is None:
        raise Skip("no ptm_thermal_sim in build/bin; build it with: "
                   "cmake --build build --target ptm_thermal_sim")
    return Path(exe).resolve()


def test_a_curated_selection_emits_a_config_the_engine_accepts():
    """The end of the chain: the picker's keys, through the GUI's own writer,
    into the engine's own parser.

    ``--validate-config`` is the engine's answer rather than a second copy of
    its rules. What this proves is narrow and worth stating: the emitted YAML
    PARSES and the engine accepts every key the site wrote. It does not run
    the site, and it does not load the matrix.
    """
    exe = _engine()
    from gui_core.config_manager import ConfigManager

    # The SYNTHETIC set, deliberately. What this proves is that the keys a
    # selection emits parse in the engine -- which is about the key names, not
    # about any one site's terrain. The engine never opens these files here
    # (`--validate-config` does not load a mesh), so a fixture is a faithful
    # subject and the check runs on every machine instead of only this one.
    with _synthetic() as (sites, _payload):
        entry = sites.CURATED_SITES[0]
        _assert_engine_accepts(sites, entry, exe, ConfigManager)


def _assert_engine_accepts(sites, entry, exe, ConfigManager):
    manager = ConfigManager()
    for key, value in sites.site_wiring(entry):
        manager.set_value(key, value)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "curated_site.yaml"
        manager.save_to_yaml(str(out))
        text = out.read_text()
        assert entry.mesh.path in text and entry.occluder.path in text
        assert entry.view_factors.base_stub in text
        assert "PTM_HMATRIX_VF_REPLACE" in text, (
            "the environment block must survive into the emitted file: it is "
            "how the matrix reaches the run")
        assert entry.view_factors.replace_csr in text

        import yaml
        document = yaml.safe_load(text)
        env = document.get("engine_env") or {}
        assert env.get("PTM_HMATRIX_VF_REPLACE") == \
            entry.view_factors.replace_csr
        for name in ("PTM_ILLUM_TRUE_NORMALS", "PTM_ILLUM_RISE_EPS"):
            assert name in env, (
                f"{name} is gone from engine_env. The site picker MERGES into "
                f"that block; replacing it would silently drop the "
                f"recommended illumination fixes from every curated run.")
        assert document["mesh"]["latitude_override"] == sites.NOT_PLACED

        result = subprocess.run([str(exe), "--config", str(out),
                                 "--validate-config"],
                                capture_output=True, text=True, timeout=300)
        assert result.returncode == 0, (
            f"the engine refused a curated site's configuration "
            f"(rc={result.returncode}):\n{result.stdout[-4000:]}\n"
            f"{result.stderr[-4000:]}")


def test_the_emitted_environment_reaches_the_launch_plan_unchanged():
    """The block in the file and the environment at launch are one thing.

    ``engine_env_plan`` is what the runner exports. A config that carries the
    block is taken at its word -- so a picker that REPLACED the block instead
    of adding to it would silently strip the recommended illumination fixes,
    and this is where that would show.
    """
    from gui_core.config_manager import ConfigManager
    from gui_core.preflight import RECOMMENDED_ENGINE_ENV, engine_env_plan

    with _synthetic() as (sites, _payload):
        entry = sites.CURATED_SITES[0]
        manager = ConfigManager()
        for key, value in sites.site_wiring(entry):
            manager.set_value(key, value)
        table, source = engine_env_plan(manager.get_config())
        assert table.get("PTM_HMATRIX_VF_REPLACE") == \
            entry.view_factors.replace_csr, table
        for name, value in RECOMMENDED_ENGINE_ENV.items():
            assert table.get(name) == value, (
                f"{name} was lost between the picker and the launch plan")
        assert "engine_env" in source


def test_the_moon_card_offers_the_sites_and_refuses_an_absent_one():
    """The card, offscreen: it lists them, and a site it cannot back refuses.

    The refusal path is the one worth exercising, because it is the one that
    runs on every machine that is not this one.

    The card is built INSIDE the fixture: its combo is populated once, at
    construction, from whatever the registry holds then.
    """
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        import PyQt6
    except ImportError as exc:                            # pragma: no cover
        raise Skip(f"PyQt6 is unavailable here ({exc})")
    plug = Path(PyQt6.__file__).parent / "Qt6" / "plugins"
    os.environ.setdefault("QT_PLUGIN_PATH", str(plug))
    os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(plug / "platforms"))
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])

    from gui_core.config_manager import ConfigManager
    from gui_panels.body_cards_panel import MoonCard

    with _synthetic() as (sites, _payload):
        manager = ConfigManager()
        card = MoonCard(manager, route=None)
        card._active = True
        card._mode = "local"
        try:
            offered = [card.site_combo.itemData(i)
                       for i in range(card.site_combo.count())]
            assert offered[0] == "", "the first entry must be 'none'"
            assert set(offered[1:]) == {e.key for e in sites.CURATED_SITES}, \
                offered

            # A site whose files are absent: select it and confirm the card
            # wrote nothing and put itself back to none.
            entry = sites.CURATED_SITES[0]
            real_isdir = sites.os.path.isdir

            def fake_isdir(path):
                if str(path) == entry.local_root:
                    return False
                return real_isdir(path)

            sites.os.path.isdir = fake_isdir
            try:
                index = offered.index(entry.key)
                card.site_combo.setCurrentIndex(index)
            finally:
                sites.os.path.isdir = real_isdir
            assert card.site_combo.currentIndex() == 0, (
                "a site that cannot be backed must not stay selected")
            assert not str(manager.get_value("mesh.import_file", "") or ""), (
                "a refused site wrote a mesh path anyway")
            assert "Nothing was written" in card.site_note.text(), \
                card.site_note.text()
        finally:
            card.deleteLater()
    del app


def test_a_checkout_with_no_overlay_offers_no_curated_sites():
    """The shipped state: an empty table, and a card that says nothing extra.

    The registry ships empty because its real entries are landing-site
    candidates kept outside this repository. This pins that: with the overlay
    pointed at a path that does not exist, there are no entries, no
    not-curated rows, no defective-matrix rows, and the Moon card offers only
    "(none)".
    """
    sites = _sites()
    with tempfile.TemporaryDirectory() as tmp:
        absent = Path(tmp) / "no_overlay_here.json"
        status = sites.reload_registry(absent)
        try:
            assert status.ok, "an ABSENT overlay is not a fault"
            assert not status.present
            assert sites.CURATED_SITES == (), sites.CURATED_SITES
            assert sites.NOT_CURATED == {}
            assert sites.DEFECTIVE_VIEW_FACTORS == {}
            assert sites.curated_sites_for_body("Moon") == ()
            assert "normal state" in status.summary(), status.summary()
        finally:
            sites.reload_registry()


def test_a_broken_overlay_is_loud_and_does_not_look_like_an_absent_one():
    """The failure this design could most easily have: a silent empty table.

    An overlay that is present but unparseable, or written to a schema this
    build does not read, ends with the same empty table as no overlay at all.
    Only one of the two is a fault, so they must not read the same.
    """
    sites = _sites()
    with tempfile.TemporaryDirectory() as tmp:
        cases = {
            "not_json.json": "{ this is not json",
            "wrong_schema.json":
                json.dumps({"schema": "ptm3d.curated_sites_local/99",
                            "sites": []}),
            "not_an_object.json": json.dumps([1, 2, 3]),
            "bad_entry.json": json.dumps({
                "schema": sites.LOCAL_REGISTRY_SCHEMA,
                "sites": [{"key": "SYN-BROKEN01", "body": "Moon"}]}),
            "duplicate_key.json": json.dumps({
                "schema": sites.LOCAL_REGISTRY_SCHEMA,
                "sites": [_minimal_entry("SYN-DUP01"),
                          _minimal_entry("SYN-DUP01")]}),
        }
        try:
            for name, text in cases.items():
                path = Path(tmp) / name
                path.write_text(text)
                status = sites.reload_registry(path)
                assert not status.ok, f"{name} was accepted and must not be"
                assert status.present, f"{name} exists and must be reported so"
                assert "could not be used" in status.summary(), status.summary()
                assert "not an empty overlay" in status.summary(), (
                    f"{name}'s message does not distinguish itself from having "
                    f"no overlay: {status.summary()}")
                assert sites.CURATED_SITES == (), (
                    f"{name} contributed entries despite failing to load")
        finally:
            sites.reload_registry()


def _minimal_entry(key):
    """The smallest mapping ``site_from_mapping`` accepts, for the loader arms."""
    return {
        "key": key, "body": "Moon",
        "mesh": {"path": "/nowhere/core.ply", "faces": 4,
                 "latitude_deg": -12.345, "longitude_deg": 210.987},
        "occluder": {"path": "/nowhere/ring.ply", "faces": 2},
        "view_factors": {"replace_csr": "/nowhere/m.csr",
                         "base_stub": "/nowhere/b.vf", "n_faces": 4,
                         "nnz": 1, "geomhash": 1.0},
    }


def test_the_overlay_reaches_the_card_when_it_is_broken():
    """A fault in a file the user cannot see must appear where they are.

    ``local_registry_status`` is only useful if something shows it. The Moon
    card's site note is where an empty combo gets explained, so that is where
    a broken overlay has to land.
    """
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        import PyQt6
    except ImportError as exc:                            # pragma: no cover
        raise Skip(f"PyQt6 is unavailable here ({exc})")
    plug = Path(PyQt6.__file__).parent / "Qt6" / "plugins"
    os.environ.setdefault("QT_PLUGIN_PATH", str(plug))
    os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(plug / "platforms"))
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])

    from gui_core.config_manager import ConfigManager
    from gui_panels.body_cards_panel import MoonCard

    sites = _sites()
    with tempfile.TemporaryDirectory() as tmp:
        broken = Path(tmp) / "broken.json"
        broken.write_text("{ not json at all")
        sites.reload_registry(broken)
        try:
            manager = ConfigManager()
            card = MoonCard(manager, route=None)
            try:
                assert card.site_combo.count() == 1, (
                    "a broken overlay must contribute no sites")
                # THROUGH load_card(). Asserting straight after the constructor
                # passed while the behavior was broken: `load_card` runs on
                # every config load and card activation, and it used to
                # overwrite this note with the ordinary "no curated site" text.
                card.load_card()
                note = card.site_note.text()
                assert "could not be used" in note, note
                assert str(broken) in note, (
                    f"the card does not name the file the user has to fix: "
                    f"{note}")
            finally:
                card.deleteLater()
        finally:
            sites.reload_registry()
    del app


def _offscreen_card():
    """A live MoonCard with no window. Raises Skip when Qt is unavailable."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        import PyQt6
    except ImportError as exc:                            # pragma: no cover
        raise Skip(f"PyQt6 is unavailable here ({exc})")
    plug = Path(PyQt6.__file__).parent / "Qt6" / "plugins"
    os.environ.setdefault("QT_PLUGIN_PATH", str(plug))
    os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(plug / "platforms"))
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])

    from gui_core.config_manager import ConfigManager
    from gui_panels.body_cards_panel import MoonCard
    manager = ConfigManager()
    card = MoonCard(manager, route=None)
    card._active = True
    card._mode = "local"
    return app, manager, card


def test_the_search_result_survives_being_used():
    """The confirmation must not erase itself, and it did.

    ``_on_use_view_factor`` writes the note and then calls
    ``load_from_config``, which reaches ``BodyCard.load_card`` and resets the
    same label to its idle text. Measured offscreen: the whole confirmation
    disappeared the instant it was written -- INCLUDING the only warning
    anywhere in this project that a replacement CSR with no base file makes the
    run rebuild the matrix it was just handed. Nothing in the pre-flight checks
    that pair.
    """
    _app, manager, card = _offscreen_card()
    vf = None
    try:
        from gui_core import vf_discovery as vf
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            mesh = tmp / "site.ply"
            mesh.write_text("ply\nformat ascii 1.0\nelement vertex 6\n"
                            "property float x\nelement face 4\n"
                            "property list uchar int vertex_indices\n"
                            "end_header\n")
            csr = tmp / "m.csr"
            head = struct.pack("<id", 4, 1.0)
            indptr = struct.pack("<5q", 0, 1, 2, 3, 4)
            body = struct.pack("<q", 4) + struct.pack("<4i", 0, 0, 0, 0) \
                + struct.pack("<4f", 0.1, 0.1, 0.1, 0.1)
            csr.write_bytes(head + indptr + body)

            manager.set_value("mesh.import_file", str(mesh))
            card.refresh_vf_discovery(scan=True)
            assert card._vf_candidates, card.vf_discovery_note.text()
            assert card.vf_candidate_combo.currentIndex() == 0
            assert not card.vf_use_button.isEnabled(), (
                "nothing may be pre-selected: sort order is not evidence")

            card.vf_candidate_combo.setCurrentIndex(1)
            assert card.vf_use_button.isEnabled()
            card._on_use_view_factor()

            text = card.vf_discovery_note.text()
            assert "Wired:" in text, text
            assert "rebuild the matrix it was just handed" in text, (
                f"the missing-base-file warning was erased: {text}")
            env = manager.get_value("engine_env", {}) or {}
            assert env.get("PTM_HMATRIX_VF_REPLACE") == str(csr)
    finally:
        card.deleteLater()
    del vf


def test_the_refusals_stay_on_screen_next_to_the_selection():
    """"Refuse with the reason" is not served by a reason nobody reads.

    The scan summary and every refusal used to be written to the same label
    and then immediately overwritten by the highlighted candidate's notes, so
    in the common case -- any scan that found at least one match -- the
    refusals were invisible.
    """
    _app, manager, card = _offscreen_card()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            mesh = tmp / "site.ply"
            mesh.write_text("ply\nformat ascii 1.0\nelement vertex 6\n"
                            "property float x\nelement face 4\n"
                            "property list uchar int vertex_indices\n"
                            "end_header\n")
            # one that fits, one that does not
            (tmp / "good.vf").write_bytes(
                struct.pack("<IIQQQ", 0x5646464D, 2, 4, 4, 12))
            (tmp / "wrong.vf").write_bytes(
                struct.pack("<IIQQQ", 0x5646464D, 2, 5, 5, 12))
            manager.set_value("mesh.import_file", str(mesh))
            card.refresh_vf_discovery(scan=True)
            card.vf_candidate_combo.setCurrentIndex(1)
            text = card.vf_discovery_note.text()
            assert "REFUSED — wrong.vf" in text, (
                f"the refusal was overwritten by the selection: {text}")
            assert "good.vf" in text, text
    finally:
        card.deleteLater()


CHECKS = [
    test_every_recorded_number_is_re_read_off_its_own_file,
    test_the_provenance_sidecar_agrees_with_the_matrix_it_describes,
    test_the_mesh_really_sits_where_the_registry_says_it_does,
    test_selecting_a_site_never_asks_the_engine_to_move_the_mesh,
    test_the_wiring_is_the_whole_set_or_nothing,
    test_the_registry_refuses_the_matrices_it_says_it_refuses,
    test_the_pruned_experiments_are_refused_and_their_receipts_say_why,
    test_the_orphaned_sites_are_not_curated_and_say_why,
    test_a_machine_without_the_files_is_told_so_and_gets_no_wiring,
    test_a_cloud_placeholder_in_the_set_refuses_the_whole_site,
    test_a_truncated_matrix_is_caught_by_the_registrys_own_arithmetic,
    test_the_provenance_text_names_the_ring_that_was_not_chosen,
    test_the_coarse_site_declares_its_resolution_gap,
    test_the_lookup_will_not_match_one_site_for_another,
    test_a_curated_selection_emits_a_config_the_engine_accepts,
    test_the_emitted_environment_reaches_the_launch_plan_unchanged,
    test_the_moon_card_offers_the_sites_and_refuses_an_absent_one,
    test_a_checkout_with_no_overlay_offers_no_curated_sites,
    test_a_broken_overlay_is_loud_and_does_not_look_like_an_absent_one,
    test_the_overlay_reaches_the_card_when_it_is_broken,
    test_the_search_result_survives_being_used,
    test_the_refusals_stay_on_screen_next_to_the_selection,
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
