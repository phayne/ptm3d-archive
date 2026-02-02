"""
Data loader for Viewshed GUI.
Wraps pegasus_sim I/O functions and provides helpers for loading
PTM3D output data needed by the viewshed simulation.
"""

import struct
from pathlib import Path

import numpy as np


def load_mesh(mesh_path):
    """Load a PLY mesh file.

    Tries to import the pegasus_sim reader; falls back to a minimal PLY parser.

    Returns:
        dict with keys: n_vertices, n_faces, vertices, faces, face_centers,
        face_normals, face_areas, face_latitudes, face_longitudes.
    """
    try:
        from pegasus_sim.ptm_io import read_ply_mesh
        return read_ply_mesh(mesh_path)
    except ImportError:
        return _read_ply_mesh_minimal(mesh_path)


def load_ptm_binary(filepath):
    """Read a PTM3D binary file (uint32 header + float32 array).

    Returns:
        numpy.ndarray of float32 values.
    """
    filepath = Path(filepath)
    with open(filepath, "rb") as f:
        (n,) = struct.unpack("<I", f.read(4))
        data = np.frombuffer(f.read(n * 4), dtype=np.float32)
    return data


def write_ptm_binary(filepath, data):
    """Write a PTM3D binary file (uint32 header + float32 array)."""
    filepath = Path(filepath)
    arr = np.asarray(data, dtype=np.float32)
    with open(filepath, "wb") as f:
        f.write(struct.pack("<I", len(arr)))
        f.write(arr.tobytes())


def load_temperature_statistics(ptm_dir):
    """Load temperature statistics from PTM3D output.

    Returns:
        dict with keys 'T_max', 'T_min', 'T_mean' (each a float32 array),
        or partial dict if some files are missing.
    """
    ptm_dir = Path(ptm_dir)
    stats = {}
    for name, filename in [
        ("T_max", "temperature_max.bin"),
        ("T_min", "temperature_min.bin"),
        ("T_mean", "temperature_mean.bin"),
    ]:
        path = ptm_dir / "statistics" / filename
        if path.is_file():
            stats[name] = load_ptm_binary(path)
    return stats


def load_temperature_timestep(ptm_dir, idx):
    """Load temperature at a specific timestep.

    Returns:
        numpy.ndarray of float32 values.
    """
    path = Path(ptm_dir) / "temperature" / f"temperature_{idx:04d}.bin"
    return load_ptm_binary(path)


def load_timesteps(ptm_dir):
    """Load the timestep index from time_steps.txt.

    Returns:
        list of (index, utc_string) tuples.
    """
    try:
        from pegasus_sim.ptm_io import read_timesteps
        return read_timesteps(Path(ptm_dir) / "time_steps.txt")
    except ImportError:
        return _read_timesteps_minimal(Path(ptm_dir) / "time_steps.txt")


def load_waypoints(csv_path):
    """Load waypoints from CSV file.

    Returns:
        list of dicts with keys: label, latitude, longitude, height.
    """
    try:
        from pegasus_sim.ptm_io import read_waypoints_csv
        return read_waypoints_csv(csv_path)
    except ImportError:
        return _read_waypoints_minimal(csv_path)


def load_illumination_timestep(ptm_dir, idx):
    """Load illumination/solar flux at a specific timestep.

    Returns:
        numpy.ndarray or None if not available.
    """
    ptm_dir = Path(ptm_dir)
    # Try several possible naming conventions
    for pattern in [
        f"illumination/solar_flux_{idx:04d}.bin",
        f"illumination/illumination_{idx:04d}.bin",
        f"solar_flux/solar_flux_{idx:04d}.bin",
    ]:
        path = ptm_dir / pattern
        if path.is_file():
            return load_ptm_binary(path)
    return None


def load_primary_ir_timestep(ptm_dir, idx):
    """Load primary IR flux at a specific timestep.

    Returns:
        numpy.ndarray or None if not available.
    """
    ptm_dir = Path(ptm_dir)
    path = ptm_dir / f"primary_ir/primary_ir_{idx:04d}.bin"
    if path.is_file():
        return load_ptm_binary(path)
    return None


def discover_output_fields(output_dir):
    """Scan an output directory for .bin files.

    Returns:
        list of dicts with keys: name, path, size_bytes.
    """
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        return []
    fields = []
    for p in sorted(output_dir.glob("*.bin")):
        fields.append({
            "name": p.stem,
            "path": str(p),
            "size_bytes": p.stat().st_size,
        })
    return fields


def discover_timestep_range(ptm_dir, start_utc, end_utc):
    """Find timestep indices that fall within a UTC window.

    Parameters:
        ptm_dir: PTM3D output directory.
        start_utc: Start datetime string (ISO).
        end_utc: End datetime string (ISO).

    Returns:
        list of (index, utc_string) tuples within the window.
    """
    from datetime import datetime, timedelta

    SYNODIC_MONTH_DAYS = 29.53059

    def parse_utc(s):
        for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
            try:
                return datetime.strptime(s, fmt)
            except ValueError:
                continue
        raise ValueError(f"Cannot parse UTC: {s}")

    mission_start = datetime.strptime(start_utc[:10], "%Y-%m-%d")
    mission_end = datetime.strptime(end_utc[:10], "%Y-%m-%d")

    # Shift back by one synodic month to match PTM3D time domain
    shifted_start = mission_start - timedelta(days=SYNODIC_MONTH_DAYS)
    shifted_end = mission_end - timedelta(days=SYNODIC_MONTH_DAYS)

    timesteps = load_timesteps(ptm_dir)
    result = []
    for idx, utc_str in timesteps:
        ts_dt = parse_utc(utc_str)
        if shifted_start <= ts_dt <= shifted_end:
            result.append((idx, utc_str))
    return result


# ---------------------------------------------------------------------------
# Minimal fallback readers (no pegasus_sim dependency)
# ---------------------------------------------------------------------------

def _read_timesteps_minimal(ts_path):
    """Parse time_steps.txt without pegasus_sim."""
    timesteps = []
    with open(ts_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    idx = int(parts[0])
                    utc = parts[1]
                    timesteps.append((idx, utc))
                except ValueError:
                    continue
    return timesteps


def _read_waypoints_minimal(csv_path):
    """Parse a waypoints CSV without pegasus_sim."""
    import csv
    waypoints = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            wp = {
                "label": row.get("label", row.get("name", "")),
                "latitude": float(row.get("latitude", row.get("lat", 0))),
                "longitude": float(row.get("longitude", row.get("lon", 0))),
                "height": float(row.get("height", row.get("elevation", 2.0))),
            }
            waypoints.append(wp)
    return waypoints


def _read_ply_mesh_minimal(mesh_path):
    """Minimal PLY reader for ASCII and binary_little_endian formats.

    Reads vertex positions and triangular faces, computes face centres,
    normals, areas, and approximate lat/lon.
    """
    mesh_path = Path(mesh_path)
    vertices = []
    faces = []

    with open(mesh_path, "rb") as f:
        # Parse header
        is_binary = False
        n_vertices = 0
        n_faces = 0
        header_lines = []
        while True:
            line = f.readline().decode("ascii", errors="replace").strip()
            header_lines.append(line)
            if line.startswith("element vertex"):
                n_vertices = int(line.split()[-1])
            elif line.startswith("element face"):
                n_faces = int(line.split()[-1])
            elif "binary_little_endian" in line:
                is_binary = True
            elif line == "end_header":
                break

        if is_binary:
            # Binary little-endian: assume 3 floats per vertex
            vdata = np.frombuffer(f.read(n_vertices * 12), dtype="<f4")
            vertices = vdata.reshape(n_vertices, 3)
            # Faces: each prefixed by uint8 count (3) then 3 int32 indices
            fdata = f.read(n_faces * (1 + 12))
            faces = np.zeros((n_faces, 3), dtype=np.int32)
            offset = 0
            for i in range(n_faces):
                count = fdata[offset]
                offset += 1
                idx = np.frombuffer(fdata[offset:offset + count * 4], dtype="<i4")
                faces[i] = idx[:3]
                offset += count * 4
        else:
            # ASCII
            for _ in range(n_vertices):
                parts = f.readline().decode().split()
                vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])
            vertices = np.array(vertices, dtype=np.float64)

            for _ in range(n_faces):
                parts = f.readline().decode().split()
                n = int(parts[0])
                faces.append([int(parts[j + 1]) for j in range(min(n, 3))])
            faces = np.array(faces, dtype=np.int32)

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)

    # Compute face properties
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    face_centers = (v0 + v1 + v2) / 3.0
    edge1 = v1 - v0
    edge2 = v2 - v0
    cross = np.cross(edge1, edge2)
    norms = np.linalg.norm(cross, axis=1, keepdims=True)
    face_normals = cross / np.maximum(norms, 1e-30)
    face_areas = 0.5 * norms.ravel()

    # Approximate lat/lon from Cartesian (assuming body-centred coords)
    r = np.linalg.norm(face_centers, axis=1)
    face_latitudes = np.degrees(np.arcsin(
        np.clip(face_centers[:, 2] / np.maximum(r, 1e-30), -1, 1)
    ))
    face_longitudes = np.degrees(np.arctan2(face_centers[:, 1], face_centers[:, 0]))

    return {
        "n_vertices": len(vertices),
        "n_faces": len(faces),
        "vertices": vertices,
        "faces": faces,
        "face_centers": face_centers,
        "face_normals": face_normals,
        "face_areas": face_areas,
        "face_latitudes": face_latitudes,
        "face_longitudes": face_longitudes,
    }
