#!/usr/bin/env python3
"""A3GT v2 — crop an extended-mesh site deliverable back to the core ROI.

The sim runs on the physics-horizon *extended* mesh so distant topography
shadows the core. The delivered product must contain ONLY the requested
core ROI. This module, run at the end of the per-site composite merge,
rewrites the site root so ``mesh.ply`` and every ``statistics/*.bin`` are
core-only and remain face-index aligned (Architecture B from the plan):

  * one boolean ``keep`` mask (geodesic distance of each face centroid from
    the site center ≤ R_core) is applied IDENTICALLY to the PLY face list
    and to every per-face statistics binary — positional alignment is
    preserved by construction;
  * unreferenced vertices are dropped and face vertex indices renumbered;
  * the full extended mesh is retained as ``mesh_extended.ply`` (provenance);
  * ``summary.json`` is recomputed over the core faces.

Face centroid lat/lon use the SAME formula as ``ptm_regridder`` /
``ptm3d_stats_to_cog`` so the kept set is exactly the faces the COG step
would place inside the core. The geodesic mirrors the C++ haversine
(``dtm_to_ply_delaunay`` ``geodesicDistance``, lunar R = 1737.4 km).

mesh.ply may be a hardlink to a chunk copy (the merger uses os.link), so
the original is renamed aside and a brand-new file is written — the crop
never mutates a file in place that could be shared.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path("/Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/europa/heat3d/ptm")
BODY_R_KM = 1737.4

_PLY_T = {
    "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
    "int": "<i4", "int32": "<i4", "uint": "<u4", "uint32": "<u4",
    "uchar": "<u1", "uint8": "<u1", "char": "<i1", "int8": "<i1",
    "short": "<i2", "int16": "<i2", "ushort": "<u2", "uint16": "<u2",
}


def geodesic_km(lat0: float, lon0: float,
                lat: np.ndarray, lon: np.ndarray,
                body_r_km: float = BODY_R_KM) -> np.ndarray:
    """Great-circle distance (km), haversine — matches the C++ mesh tool."""
    p0 = np.radians(lat0)
    p1 = np.radians(lat)
    dlat = np.radians(lat - lat0)
    dlon = np.radians(lon - lon0)
    a = (np.sin(dlat / 2.0) ** 2
         + np.cos(p0) * np.cos(p1) * np.sin(dlon / 2.0) ** 2)
    return body_r_km * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _load_ply(path: Path):
    """Parse a binary_little_endian PLY.

    Returns (header, vprops, vert, faces, fview, face_lat, face_lon).
    ``fview`` is the raw (n_f, stride) uint8 face block — kept so the crop
    can preserve per-face scalar properties (the dtm tool emits
    ``property float latitude/longitude`` per face → 21-byte faces; they
    must survive the crop or the header/body desyncs).
    """
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError(f"not a PLY: {path}")
        header = [b"ply"]
        n_v = n_f = 0
        vprops: list[tuple[str, str]] = []
        face_extra = []
        cur = None
        binary = False
        while True:
            raw = f.readline()
            header.append(raw.rstrip(b"\n"))
            ln = raw.decode("ascii", "replace").strip()
            if ln.startswith("format"):
                binary = "binary_little_endian" in ln
            elif ln.startswith("element vertex"):
                n_v = int(ln.split()[-1]); cur = "v"
            elif ln.startswith("element face"):
                n_f = int(ln.split()[-1]); cur = "f"
            elif ln.startswith("property list"):
                pass
            elif ln.startswith("property") and cur == "v":
                t = ln.split()
                vprops.append((t[2], _PLY_T.get(t[1], "<f4")))
            elif ln.startswith("property") and cur == "f":
                t = ln.split()
                face_extra.append(_PLY_T.get(t[1], "<f4"))
            elif ln == "end_header":
                break
        if not binary:
            raise ValueError("only binary_little_endian PLY supported")

        vdt = np.dtype(vprops)
        vert = np.frombuffer(f.read(n_v * vdt.itemsize), dtype=vdt).copy()

        blob = f.read()
    extra = sum(np.dtype(t).itemsize for t in face_extra)
    stride = 1 + 12 + extra            # uchar count + 3*int32 (+ extras)
    if len(blob) != n_f * stride:
        raise ValueError(
            f"face block {len(blob)} != {n_f}*{stride}; non-triangular or "
            "unexpected face props — crop expects triangles")
    fview = np.frombuffer(blob, dtype=np.uint8).reshape(n_f, stride)
    if not np.all(fview[:, 0] == 3):
        raise ValueError("non-triangle face found; crop expects triangles")
    faces = fview[:, 1:13].copy().view("<i4").reshape(n_f, 3)

    vx = vert[vprops[0][0]].astype(np.float64)
    vy = vert[vprops[1][0]].astype(np.float64)
    vz = vert[vprops[2][0]].astype(np.float64)
    cx = (vx[faces[:, 0]] + vx[faces[:, 1]] + vx[faces[:, 2]]) / 3.0
    cy = (vy[faces[:, 0]] + vy[faces[:, 1]] + vy[faces[:, 2]]) / 3.0
    cz = (vz[faces[:, 0]] + vz[faces[:, 1]] + vz[faces[:, 2]]) / 3.0
    r = np.sqrt(cx * cx + cy * cy + cz * cz)
    face_lat = np.degrees(np.arcsin(np.clip(cz / np.maximum(r, 1e-30), -1, 1)))
    face_lon = np.degrees(np.arctan2(cy, cx))
    return header, vprops, vert, faces, fview, face_lat, face_lon


def _write_ply(path: Path, header: list[bytes], vert: np.ndarray,
               face_rows: np.ndarray, n_faces: int) -> None:
    """Write a cropped binary PLY.

    ``face_rows`` is the (n_keep, stride) uint8 block already carrying the
    original per-face scalar properties (bytes 13:) with the vertex-index
    bytes (1:13) re-mapped — so the header (copied verbatim, counts updated)
    stays consistent with the body. Stripping the per-face props instead
    would desync the header and break read_ply_mesh_minimal."""
    new_hdr = []
    for line in header:
        if line.startswith(b"element vertex"):
            new_hdr.append(b"element vertex %d" % len(vert))
        elif line.startswith(b"element face"):
            new_hdr.append(b"element face %d" % n_faces)
        else:
            new_hdr.append(line)
    with open(path, "wb") as f:
        f.write(b"\n".join(new_hdr) + b"\n")
        f.write(vert.tobytes())
        f.write(np.ascontiguousarray(face_rows).tobytes())


def _crop_bin(path: Path, keep: np.ndarray) -> tuple[int, int]:
    """Crop one PTM3D per-face binary (<u4 count><count f4>) by ``keep``.

    Returns (old_n, new_n). Atomic via temp + os.replace.
    """
    with open(path, "rb") as f:
        head = f.read(4)
        n = struct.unpack("<I", head)[0]
        data = np.frombuffer(f.read(), dtype="<f4")
    if data.size != keep.size or n != keep.size:
        raise ValueError(
            f"{path.name}: count {n}/{data.size} != mask {keep.size} "
            "(face-index misalignment — refusing to crop)")
    kept = data[keep]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(struct.pack("<I", int(kept.size)))
        f.write(kept.tobytes())
    os.replace(tmp, path)
    return n, int(kept.size)


def crop_site(site_dir: Path, site_lat: float, site_lon: float,
              r_core_km: float, *, dry_run: bool = False,
              verbose: bool = True) -> dict[str, Any]:
    """Crop ``<site_dir>/mesh.ply`` + ``statistics/*.bin`` to the core ROI."""
    site_dir = Path(site_dir)
    mesh = site_dir / "mesh.ply"
    if not mesh.is_file():
        raise FileNotFoundError(f"no mesh.ply in {site_dir}")

    header, vprops, vert, faces, fview, flat, flon = _load_ply(mesh)
    d = geodesic_km(site_lat, site_lon, flat, flon)
    keep = d <= r_core_km
    n_f = len(faces)
    n_keep = int(keep.sum())

    res: dict[str, Any] = {
        "site_dir": str(site_dir), "r_core_km": r_core_km,
        "n_faces_extended": n_f, "n_faces_core": n_keep,
        "max_d_km_extended": round(float(d.max()), 3) if n_f else None,
        "cropped_bins": [], "dry_run": dry_run,
    }
    if verbose:
        print(f"  [crop] {site_dir.name}: {n_f:,} -> {n_keep:,} core faces "
              f"(R_core={r_core_km:g} km, ext max={res['max_d_km_extended']} km)",
              file=sys.stderr)

    if n_keep == n_f:
        res["note"] = "mesh already core-only (no extended ring); no-op"
        if verbose:
            print(f"  [crop] {site_dir.name}: nothing to crop",
                  file=sys.stderr)
        return res
    if n_keep == 0:
        raise ValueError(f"{site_dir.name}: 0 faces within R_core "
                         "— wrong site center or radius?")
    if dry_run:
        return res

    # Renumber: drop unreferenced vertices, remap face vertex indices.
    # Keep the original per-face rows (so per-face scalar props survive);
    # only the vertex-index bytes (1:13) are rewritten.
    kept_faces = faces[keep]
    used = np.unique(kept_faces)
    remap = np.full(len(vert), -1, dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    new_faces = np.ascontiguousarray(remap[kept_faces].astype("<i4"))
    new_vert = vert[used]

    kept_rows = fview[keep].copy()                       # (n_keep, stride) u8
    kept_rows[:, 1:13] = new_faces.view(np.uint8).reshape(len(new_faces), 12)
    # byte 0 (count==3) and bytes 13: (lat/lon float props) preserved as-is.

    # mesh.ply may be a hardlink to a chunk copy: move aside, write fresh.
    ext_path = site_dir / "mesh_extended.ply"
    os.replace(mesh, ext_path)
    _write_ply(mesh, header, new_vert, kept_rows, int(keep.sum()))
    res["mesh_extended"] = str(ext_path)

    stats_dir = site_dir / "statistics"
    if stats_dir.is_dir():
        for b in sorted(stats_dir.glob("*.bin")):
            try:
                o, nn = _crop_bin(b, keep)
                res["cropped_bins"].append({"file": b.name,
                                            "old_n": o, "new_n": nn})
            except ValueError as e:
                res.setdefault("errors", []).append(str(e))
                if verbose:
                    print(f"  [crop] WARN {e}", file=sys.stderr)

    # Recompute summary.json over the cropped binaries (reuse the merger's
    # writer so the schema stays identical).
    sm = site_dir / "summary.json"
    if sm.is_file():
        prev = json.loads(sm.read_text())
        try:
            sys.path.insert(0, str(REPO / "scripts"))
            from a3gt_v2_merge_sparse import write_summary
            write_summary(site_dir, prev.get("ellipse_id", site_dir.name),
                          n_windows=prev.get("n_windows_merged", 0),
                          n_chunks=prev.get("n_chunks_merged", 0))
            res["summary_recomputed"] = True
        except Exception as e:                       # noqa: BLE001
            res.setdefault("errors", []).append(f"summary recompute: {e}")

    if res.get("errors") and verbose:
        print(f"  [crop] {site_dir.name}: completed with "
              f"{len(res['errors'])} error(s)", file=sys.stderr)
    return res


def _site_lat_lon_from_manifest(manifest: Path, ellipse_id: str):
    m = json.loads(Path(manifest).read_text())
    for s in m.get("sites", []):
        if s["ellipse_id"] == ellipse_id:
            return float(s["lat_deg"]), float(s["lon_deg"])
    raise KeyError(f"{ellipse_id} not in {manifest}")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--site-dir", type=Path, required=True)
    p.add_argument("--r-core-km", type=float, required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--site-latlon", nargs=2, type=float,
                   metavar=("LAT", "LON"))
    g.add_argument("--manifest", type=Path,
                   help="site_manifest.json; site looked up by site-dir name.")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if args.manifest:
        lat, lon = _site_lat_lon_from_manifest(args.manifest,
                                               args.site_dir.name)
    else:
        lat, lon = args.site_latlon

    res = crop_site(args.site_dir, lat, lon, args.r_core_km,
                    dry_run=args.dry_run)
    print(json.dumps(res, indent=2))
    return 0 if not res.get("errors") else 1


if __name__ == "__main__":
    sys.exit(main())
