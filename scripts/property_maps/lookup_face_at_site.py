#!/usr/bin/env python3
"""
Find the mesh face nearest a given (lat, lon) on a PTM3D PLY mesh and print
the per-face statistics computed by `compute_statistics: true`.

Statistics directory layout (per writeStatisticsBinary):
  <stats-dir>/<datatype>_<avg|max|min>.bin   — uint32 N + N float32 values
  datatypes: temperature, illumination, direct_solar, primary_reflected,
             primary_ir, ice_depth (avg only).

Usage:
  python lookup_face_at_site.py \
      --mesh data/dem/<region>/<site>.ply \
      --stats-dir output/mm_site1_yearly/statistics \
      --lat -84.36705 --lon 39.14019
"""
from __future__ import annotations

import argparse
import math
import struct
import sys
from pathlib import Path

import numpy as np


def parse_ply_header(path: Path):
    """Return (header_size_bytes, n_vertices, n_faces, vertex_dtype, has_face_lat_lon)."""
    with open(path, "rb") as f:
        n_v = n_f = None
        props_v = []
        in_v = in_f = False
        face_props = []
        while True:
            raw = f.readline()
            if not raw:
                break
            line = raw.strip().decode()
            if line.startswith("element vertex"):
                n_v = int(line.split()[2]); in_v = True; in_f = False
            elif line.startswith("element face"):
                n_f = int(line.split()[2]); in_v = False; in_f = True
            elif line.startswith("property") and in_v:
                # property TYPE NAME
                parts = line.split()
                props_v.append((parts[1], parts[2]))
            elif line.startswith("property") and in_f:
                face_props.append(line)
            elif line == "end_header":
                header_size = f.tell()
                break
    type_map = {"double": "<f8", "float": "<f4", "int": "<i4", "uint": "<u4",
                "uchar": "<u1", "char": "<i1", "short": "<i2", "ushort": "<u2"}
    fields = []
    for typ, name in props_v:
        if typ in type_map:
            fields.append((name, type_map[typ]))
    vdtype = np.dtype(fields)
    return header_size, n_v, n_f, vdtype


def read_vertices(path: Path):
    header_size, n_v, n_f, vdtype = parse_ply_header(path)
    with open(path, "rb") as f:
        f.seek(header_size)
        v = np.frombuffer(f.read(n_v * vdtype.itemsize), dtype=vdtype)
    return v, n_v, n_f, header_size + n_v * vdtype.itemsize


def read_faces_indices(path: Path, face_data_offset: int, n_f: int):
    """Read face vertex indices. PLY face format: uchar count + count*int32 indices."""
    with open(path, "rb") as f:
        f.seek(face_data_offset)
        # For triangulated mesh, each face is 1 + 3*4 = 13 bytes
        raw = f.read(n_f * 13)
    arr = np.frombuffer(raw, dtype=np.dtype([('count', '<u1'),
                                              ('a', '<i4'),
                                              ('b', '<i4'),
                                              ('c', '<i4')]))
    return arr


def read_stat(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        n = struct.unpack("<I", f.read(4))[0]
        arr = np.frombuffer(f.read(n * 4), dtype="<f4")
    return arr


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--stats-dir", required=True)
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--body-radius", type=float, default=1737400.0)
    args = ap.parse_args()

    mesh_path = Path(args.mesh)
    stats_dir = Path(args.stats_dir)
    if not mesh_path.exists():
        sys.exit(f"mesh not found: {mesh_path}")
    if not stats_dir.exists():
        sys.exit(f"stats-dir not found: {stats_dir}")

    print(f"Reading mesh: {mesh_path.name}")
    v, n_v, n_f, faces_offset = read_vertices(mesh_path)
    print(f"  vertices: {n_v:,}   faces: {n_f:,}")

    # Compute face centers from vertex indices
    print("  reading faces...")
    faces = read_faces_indices(mesh_path, faces_offset, n_f)
    if not np.all(faces['count'] == 3):
        sys.exit("non-triangular faces in PLY")
    fa, fb, fc = faces['a'], faces['b'], faces['c']
    x, y, z = v['x'], v['y'], v['z']
    cx = (x[fa] + x[fb] + x[fc]) / 3.0
    cy = (y[fa] + y[fb] + y[fc]) / 3.0
    cz = (z[fa] + z[fb] + z[fc]) / 3.0
    cr = np.sqrt(cx*cx + cy*cy + cz*cz)
    cx_n, cy_n, cz_n = cx/cr, cy/cr, cz/cr

    # Target unit vector
    rl, rln = math.radians(args.lat), math.radians(args.lon)
    tx = math.cos(rl) * math.cos(rln)
    ty = math.cos(rl) * math.sin(rln)
    tz = math.sin(rl)

    dot = cx_n * tx + cy_n * ty + cz_n * tz
    fi = int(np.argmax(dot))
    ang_rad = math.acos(max(-1.0, min(1.0, float(dot[fi]))))
    ang_m = ang_rad * args.body_radius
    face_lat = math.degrees(math.asin(cz_n[fi]))
    face_lon = math.degrees(math.atan2(cy_n[fi], cx_n[fi]))
    elev = float(cr[fi]) - args.body_radius
    print(f"\nNearest face to ({args.lat:.5f}°, {args.lon:.5f}°):")
    print(f"  face index : {fi}")
    print(f"  face center: lat={face_lat:.5f}°  lon={face_lon:.5f}°  elev={elev:.1f} m")
    print(f"  great-circle distance: {ang_m:.2f} m")

    # Read each statistic
    print(f"\nStatistics from {stats_dir}:")
    DATATYPES = ["temperature", "illumination", "direct_solar",
                 "primary_reflected", "primary_ir"]
    for dtype in DATATYPES:
        row = []
        for stat in ("max", "avg", "min"):
            p = stats_dir / f"{dtype}_{stat}.bin"
            if not p.exists():
                row.append("missing")
                continue
            arr = read_stat(p)
            if fi < arr.size:
                v_ = arr[fi]
                if math.isnan(v_):
                    row.append("nan")
                else:
                    row.append(f"{v_:.3f}")
            else:
                row.append("oob")
        print(f"  {dtype:<20s}  max={row[0]:>10s}  avg={row[1]:>10s}  min={row[2]:>10s}")

    # Schorghofer & Taylor (2007) ice-stability depth (single file, no
    # avg/max/min split). Falls back to the legacy ice_depth_avg.bin if
    # the new file isn't present.
    new_p = stats_dir / "ice_depth.bin"
    legacy_p = stats_dir / "ice_depth_avg.bin"
    if new_p.exists():
        arr = read_stat(new_p)
        if fi < arr.size:
            v_ = arr[fi]
            v_str = "nan" if math.isnan(v_) else f"{v_:.5f} m  ({v_*100:.2f} cm)"
        else:
            v_str = "oob"
        print(f"  {'ice_depth':<20s}  {v_str}  [Schorghofer/Taylor 2007 depth-resolved]")
    elif legacy_p.exists():
        arr = read_stat(legacy_p)
        if fi < arr.size:
            v_ = arr[fi]
            v_str = "nan" if math.isnan(v_) else f"{v_:.5f} m  ({v_*100:.2f} cm)"
        else:
            v_str = "oob"
        print(f"  {'ice_depth':<20s}  {v_str}  [legacy cubic-T(z) approximation]")
    else:
        print(f"  {'ice_depth':<20s}  missing  (enable thermal.compute_ice_stability)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
