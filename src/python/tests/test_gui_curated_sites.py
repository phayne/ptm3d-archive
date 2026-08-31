#!/usr/bin/env python3
"""The lunar-site tier: a set that ran, or nothing at all.

**Why this file exists.** An A3GT site is three files -- a core terrain, a
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

A machine without ``~/projects/a3gt`` skips the disk-anchored checks BY NAME
rather than passing them vacuously.

Run directly::

    python3 src/python/tests/test_gui_curated_sites.py
"""

from __future__ import annotations

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
sys.path.insert(0, str(REPO_ROOT / "gui"))


class Skip(Exception):
    """A check that cannot run here, as distinct from one that failed."""


def _sites():
    try:
        from gui_core import curated_sites
    except ImportError as exc:                            # pragma: no cover
        raise Skip(f"the GUI's site registry is unavailable here ({exc})")
    return curated_sites


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
    sites = _sites()
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
    sites = _sites()
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
    sites = _sites()
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
    sites = _sites()
    for entry in sites.CURATED_SITES:
        wiring = dict(sites.site_wiring(entry))
        assert wiring["mesh.latitude_override"] == sites.NOT_PLACED, (
            f"{entry.key}: the wiring writes latitude "
            f"{wiring['mesh.latitude_override']!r}, which the engine reads as "
            f"'rotate this mesh'")
        assert wiring["mesh.longitude_override"] == sites.NOT_PLACED
        assert sites.NOT_PLACED < -900.0, (
            "the sentinel has to be below the engine's own -900 threshold")


def test_the_wiring_is_the_whole_set_or_nothing():
    """A partial set is the failure mode, not a lesser version of success."""
    sites = _sites()
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
    """
    sites = _sites()
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
    assert named >= 5, (
        "the superseded list is the part of this registry that keeps somebody "
        "from picking the bigger file in the same folder; it should not be "
        f"nearly empty (found {named})")


def test_the_pruned_experiments_are_refused_and_their_receipts_say_why():
    """Their own receipts record MEASURED failure, in this project's words."""
    sites = _sites()
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
    if not checked:
        raise Skip("no prune receipt is on this machine")


def test_the_five_orphaned_sites_are_not_curated_and_say_why():
    """PTM-312. The meshes survive; the matrices beside them do not.

    They are not curated, they must not become curated by accident, and the
    reason has to travel with the fact -- otherwise the next person sees five
    A3GT sites with meshes and matrices on disk and no explanation for their
    absence from a table full of A3GT sites.
    """
    sites = _sites()
    assert set(sites.NOT_CURATED) == {"A03MP016", "A03MP037", "A03MP038",
                                      "A03MP039", "A03MP040"}
    keys = {e.key for e in sites.CURATED_SITES}
    assert not (keys & set(sites.NOT_CURATED))
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
    sites = _sites()
    entry = sites.CURATED_SITES[0]
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
    sites = _sites()
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

    try:
        real_stat(entry.mesh.path)
    except OSError:
        raise Skip(f"{entry.key}'s mesh is not on this machine")

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
    entry = sites.CURATED_SITES[0]
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
    """A03MP026 has three rings and the campaign used two of them, for two
    different jobs. A card that showed only the one it wired would leave the
    user unable to reconcile it with the project's own notes -- which name the
    other one in the same breath as this matrix.
    """
    sites = _sites()
    entry = sites.curated_site("A03MP026")
    assert entry is not None
    text = " ".join(sites.wiring_provenance(entry))
    assert "27.3108" in text, text
    assert "81.6712" in text, (
        "the ring used for the offline sweep has to be named, or the entry "
        "reads as if it disagreed with the project's own record")
    assert "NOT used" in text
    assert "not placed" in text, (
        "the placement decision is the least obvious thing this entry does "
        "and it has to be on the card")


def test_the_coarse_site_declares_its_resolution_gap():
    """Slater Plain is 200x coarser and it is a data gap, not an omission."""
    sites = _sites()
    entry = sites.curated_site("A03SP010")
    assert entry is not None
    text = " ".join(sites.wiring_provenance(entry))
    assert "LOLA" in text and "200x" in text, text
    assert entry.mesh.faces < sites.curated_site("A03MP026").mesh.faces / 100


def test_the_lookup_will_not_match_one_site_for_another():
    """An A3GT key is exact. A fuzzy match wires the wrong polar site."""
    sites = _sites()
    assert sites.curated_site("A03MP026") is not None
    assert sites.curated_site("MP026") is not None
    assert sites.curated_site("a03mp026") is not None
    assert sites.curated_site("A03MP02") is None
    assert sites.curated_site("A03MP0261") is None
    assert sites.curated_site("A03MP016") is None
    assert sites.curated_site("A03MP026", body="Europa") is None


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
    sites = _sites()
    exe = _engine()
    from gui_core.config_manager import ConfigManager

    entry = None
    for candidate in sites.CURATED_SITES:
        if sites.site_status(candidate).ready:
            entry = candidate
            break
    if entry is None:
        raise Skip("no curated site's files are on this machine")

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
    sites = _sites()
    from gui_core.config_manager import ConfigManager
    from gui_core.preflight import RECOMMENDED_ENGINE_ENV, engine_env_plan

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
    """
    sites = _sites()
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
    try:
        offered = [card.site_combo.itemData(i)
                   for i in range(card.site_combo.count())]
        assert offered[0] == "", "the first entry must be 'none'"
        assert set(offered[1:]) == {e.key for e in sites.CURATED_SITES}, offered

        # A site whose files are absent: select it and confirm the card wrote
        # nothing and put itself back to none.
        entry = sites.CURATED_SITES[0]
        real_isdir = sites.os.path.isdir

        def fake_isdir(path):
            return False if str(path) == entry.local_root else real_isdir(path)

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
    test_the_five_orphaned_sites_are_not_curated_and_say_why,
    test_a_machine_without_the_files_is_told_so_and_gets_no_wiring,
    test_a_cloud_placeholder_in_the_set_refuses_the_whole_site,
    test_a_truncated_matrix_is_caught_by_the_registrys_own_arithmetic,
    test_the_provenance_text_names_the_ring_that_was_not_chosen,
    test_the_coarse_site_declares_its_resolution_gap,
    test_the_lookup_will_not_match_one_site_for_another,
    test_a_curated_selection_emits_a_config_the_engine_accepts,
    test_the_emitted_environment_reaches_the_launch_plan_unchanged,
    test_the_moon_card_offers_the_sites_and_refuses_an_absent_one,
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
