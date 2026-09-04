#!/usr/bin/env python3
"""SYNTHETIC curated lunar sites, invented for the suite and nothing else.

**Every site, coordinate, file name and number in this module is made up.** The
keys are prefixed ``SYN-`` and the coordinates are equatorial far-side points
chosen so they cannot be confused with anywhere a mission would land. Nothing
here describes a real site, a real terrain product, or a real campaign.

Why the suite needs its own sites
--------------------------------

``gui_core.curated_sites`` ships an EMPTY table: the real entries are lunar
landing-site candidates and live in a local overlay outside the repository
(that module's docstring, "The local overlay"). A suite that only exercised the
shipped table would therefore pass vacuously on every machine -- which is the
one outcome worth avoiding, because the registry's whole job is refusing bad
combinations and a registry with no rows refuses nothing.

So the checks that are about the MACHINERY -- lookup exactness, the
not-placed sentinel, the provenance text, the refusal list, the status probes
-- run against these synthetic entries, on real files this module writes into a
temporary directory. They now run everywhere, including on a checkout with no
overlay and no site artifacts at all, which is strictly more coverage than the
hard-coded table gave.

The checks that are about REAL FILES -- "every recorded number is re-read off
the file it describes" -- still run against the overlay when one is installed,
and skip by name when it is not. A synthetic site cannot stand in for those:
their subject is whether the registry has drifted from artifacts on this disk.

Written through the overlay's own reader
----------------------------------------

:func:`install` writes JSON and calls ``curated_sites.reload_registry`` on it,
rather than constructing the classes directly. The overlay's parsing is the
part that breaks in the field, so the fixtures go through it: a suite that
built the objects by hand would exercise a path no user takes.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

#: The current H-matrix magic and version an empty base stub must carry.
HMAT_MAGIC = 0x484D4154
HMAT_VERSION = 2


def _write_ply(path, vertices, faces):
    """A binary PLY whose header declares these counts. Header only.

    The registry reads headers and never the body, so a header plus a token
    payload is a faithful subject for it and costs nothing.
    """
    header = (f"ply\nformat binary_little_endian 1.0\n"
              f"element vertex {vertices}\n"
              f"property float64 x\nproperty float64 y\nproperty float64 z\n"
              f"element face {faces}\n"
              f"property list uchar int vertex_indices\n"
              f"end_header\n")
    Path(path).write_bytes(header.encode("ascii") + b"\0" * 32)


def _write_csr(path, n_faces, nnz, geomhash):
    """A replacement CSR of exactly the length its own header implies.

    ``12 + 8*(n+1) + 8 + 8*nnz`` -- the same arithmetic
    ``CuratedSiteViewFactors.disk_bytes`` does, written out so the truncation
    check has a file that is genuinely NOT truncated to compare against.
    """
    blob = struct.pack("<id", n_faces, geomhash)
    blob += struct.pack(f"<{n_faces + 1}q", *range(n_faces + 1))
    blob += struct.pack("<q", nnz)
    blob += b"\0" * (8 * nnz)
    Path(path).write_bytes(blob)


def _write_stub(path, rows):
    """An 80-byte empty H-matrix stub in the CURRENT format."""
    blob = struct.pack("<II", HMAT_MAGIC, HMAT_VERSION)
    blob += struct.pack("<QQ", rows, rows)
    blob += b"\0" * (80 - 8 - 16 - 8)
    blob += struct.pack("<Q", 0)
    assert len(blob) == 80
    Path(path).write_bytes(blob)


def _write_receipt(path, verdict):
    Path(path).write_text(json.dumps({
        "radiance_prune_eps_Wm2": 1e-3,
        "verdict": verdict,
    }))


def build(root):
    """Write a synthetic site set under ``root`` and return the overlay payload.

    Three sites, chosen to carry the three properties the machinery has to
    handle and nothing else:

    * ``SYN-ALPHA01`` -- the ordinary case, and the one with THREE rings on
      disk of which the entry wires the middle one. Its superseded list names
      the other two plus a pruned experiment carrying a FAIL receipt and a
      matrix that was deleted on purpose.
    * ``SYN-BETA02`` -- fewer faces, more nonzeros, one ring used for both
      jobs.
    * ``SYN-GAMMA03`` -- the coarse case: a tiny core from a fallback base map,
      declaring the resolution gap in its own notes.
    """
    root = Path(root)
    meshes = root / "meshes"
    vf_build = root / "vf_build"
    base_vf = root / "base_vf"
    for directory in (meshes, vf_build, base_vf):
        directory.mkdir(parents=True, exist_ok=True)

    sites = []
    spec = [
        # key, region, lat, lon, core faces, ring faces, nnz, geomhash, coarse
        ("SYN-ALPHA01", "Synthetic Alpha Plateau", -12.345, 210.987,
         4096, 512, 8192, 1234.5678, False),
        ("SYN-BETA02", "Synthetic Beta Rim", -23.456, 200.111,
         2048, 384, 12288, 2345.6789, False),
        ("SYN-GAMMA03", "Synthetic Gamma Plain", -34.567, 190.222,
         16, 256, 32, 3456.7891, True),
    ]
    for (key, region, lat, lon, core_faces, ring_faces, nnz, geomhash,
         coarse) in spec:
        site_dir = vf_build / key
        site_dir.mkdir(parents=True, exist_ok=True)
        core = meshes / f"{key}_core.ply"
        ring = meshes / f"{key}_ring_Rcore3_Rext27km.ply"
        _write_ply(core, core_faces * 2, core_faces)
        _write_ply(ring, ring_faces * 2, ring_faces)
        csr = site_dir / f"{key}_synthetic_vf_v3.csr"
        _write_csr(csr, core_faces, nnz, geomhash)
        stub = base_vf / f"{key}_base_empty.vf"
        _write_stub(stub, core_faces)
        provenance = Path(str(csr) + ".provenance.json")
        provenance.write_text(json.dumps({
            "mesh": str(core), "n_faces": core_faces, "nnz": nnz,
            "geomhash": geomhash, "sweep_coverage_missing": 0,
            "allow_incomplete_sweep": False,
        }))

        superseded = []
        notes = []
        if key == "SYN-ALPHA01":
            # Three rings, and the entry wires the middle one. This is the
            # shape the provenance text has to explain, because the largest
            # ring is the offline sweep's ray blocker and NOT the run-time
            # occluder, and picking by size gets it wrong.
            small = meshes / f"{key}_ring_Rcore3_Rext20km.ply"
            sweep = meshes / f"{key}_ring_Rcore3_Rext81km.ply"
            _write_ply(small, 128, 64)
            _write_ply(sweep, 4096, 2048)
            pruned = site_dir / f"{key}_pruned_demo.csr"
            _write_csr(pruned, core_faces, 64, geomhash)
            _write_receipt(Path(str(pruned) + ".prune_receipt.json"),
                           "SYNTHETIC FAIL: invented number, invented site")
            superseded = [
                {"path": str(pruned),
                 "why": ("a radiance-PRUNED experiment whose own receipt "
                         "beside it records a measured failure. A pruned "
                         "matrix is not a cheaper version of the full one.")},
                {"path": str(site_dir / f"{key}_synthetic_vf_divisor2.csr"),
                 "why": ("the intermediate step: the divisor fix WITHOUT the "
                         "back-face cull, superseded the same day."),
                 "exists_note": "never built here — a refusal, not a gap"},
                {"path": str(site_dir / f"{key}_synthetic_vf.csr"),
                 "why": "the first build, before either correction.",
                 "exists_note": "deleted on purpose — its absence is not a fault"},
            ]
            notes = [
                (f"This site has THREE occluder rings on disk. The 27 km one "
                 f"is what every production configuration sets as "
                 f"mesh.occluder_file. The 81 km one was the RAY BLOCKER for "
                 f"the offline view-factor sweep. Both are correct for their "
                 f"own job and swapping them changes the shadowing of the "
                 f"run."),
            ]
        elif key == "SYN-BETA02":
            superseded = [
                {"path": str(site_dir / f"{key}_synthetic_vf.csr"),
                 "why": "the first build, before the correction.",
                 "exists_note": "deleted on purpose — its absence is not a fault"},
                {"path": str(site_dir / f"{key}_synthetic_vf_divisor2.csr"),
                 "why": "the divisor fix without the back-face cull.",
                 "exists_note": "never built here — a refusal, not a gap"},
            ]
            notes = [
                ("Fewer faces than SYN-ALPHA01 and more nonzeros. Rim terrain "
                 "sees itself; a plateau does not. Matrix size follows "
                 "concavity, not face count."),
            ]
        else:
            superseded = [
                {"path": str(site_dir / f"{key}_synthetic_vf.csr"),
                 "why": "the first build, before the correction.",
                 "exists_note": "deleted on purpose — its absence is not a fault"},
            ]
            notes = [
                (f"⚠️ {core_faces:,} faces against SYN-ALPHA01's 4,096 — about "
                 f"200x coarser, and it is not an omission. No "
                 f"high-resolution product covers this synthetic site: the "
                 f"generic LOLA base map is what built this. Never present "
                 f"results from this site as being at the same resolution "
                 f"class as the other two."),
            ]

        sites.append({
            "key": key,
            "name": f"{key} — {region}",
            "body": "Moon",
            "region": region,
            "aliases": [key.split("-", 1)[1], region.lower()],
            "local_only": True,
            "local_root": str(root),
            "use_gpu": True,
            "why_curated": ("A synthetic set, invented for the test suite: a "
                            "core terrain, a shadow-only occluder ring, and a "
                            "matrix built for THAT core."),
            "notes": notes,
            "mesh": {
                "path": str(core), "faces": core_faces,
                "latitude_deg": lat, "longitude_deg": lon,
                "measured_latitude_deg": lat, "measured_longitude_deg": lon,
                "core_radius_km": 3.0,
                "source": ("generic fallback base map (80 m/px) — NO "
                           "high-resolution product covers this synthetic "
                           "site")
                if coarse else "synthetic core DEM, 3 km region of interest",
            },
            "occluder": {
                "path": str(ring), "faces": ring_faces, "r_ext_km": 27.0,
                "role": ("run-time shadow occluder (mesh.occluder_file)"
                         if key == "SYN-ALPHA01" else
                         "run-time shadow occluder AND the offline sweep's "
                         "ray blocker — this site has one ring, used for both"),
            },
            "view_factors": {
                "replace_csr": str(csr), "base_stub": str(stub),
                "n_faces": core_faces, "nnz": nnz, "geomhash": geomhash,
                "generation": ("SYNTHETIC corrected (symmetrization divisor = "
                               "directions TRIED, plus back-face cull), "
                               "invented numbers"),
                "provenance": str(provenance),
                "campaign": "SYNTHETIC — no campaign; this set is invented",
                "gpu_fit": "invented; fits anywhere",
                "stub_rows": core_faces, "stub_nnz": 0,
            },
            "superseded": superseded,
        })

    return {
        "schema": "ptm3d.curated_sites_local/1",
        "note": "SYNTHETIC fixtures for the test suite. Nothing here is real.",
        "sites": sites,
        "not_curated": {
            name: ("PTM-312: this synthetic site's matrix came from the "
                   "pipeline BEFORE the correction, so there is no corrected "
                   "matrix to point at. Nothing in the file says so.")
            for name in ("SYN-ORPHAN01", "SYN-ORPHAN02", "SYN-ORPHAN03")
        },
        "defective_view_factors": {
            f"{name}_synthetic_vf.csr":
                ("PTM-312: built before the correction; its far field reads "
                 "up to 2x high, and the file says none of it.")
            for name in ("SYN-ORPHAN01", "SYN-ORPHAN02", "SYN-ORPHAN03")
        },
    }


def install(curated_sites, root):
    """Write the fixtures under ``root``, load them, and return the payload.

    Goes through ``reload_registry`` so the entries arrive by the same route a
    user's overlay does. Call :func:`restore` to put the module back.
    """
    payload = build(root)
    path = Path(root) / "curated_sites_local.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    result = curated_sites.reload_registry(path)
    if not result.ok:                                     # pragma: no cover
        raise AssertionError(f"the synthetic overlay did not load: "
                             f"{result.summary()}")
    return payload


def restore(curated_sites):
    """Reload whatever overlay this machine really has."""
    return curated_sites.reload_registry()
