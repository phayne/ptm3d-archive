"""
Shared I/O utilities for the survey pipeline.
"""
import struct
import json
from pathlib import Path

import numpy as np


def read_stats_binary(filepath):
    """Read binary stats file: uint32 count + float32 array."""
    with open(filepath, 'rb') as f:
        count = struct.unpack('<I', f.read(4))[0]
        return np.frombuffer(f.read(count * 4), dtype=np.float32)


def write_stats_binary(filepath, data):
    """Write uint32 count + float32 array."""
    arr = data.astype(np.float32)
    with open(filepath, 'wb') as f:
        f.write(struct.pack('<I', len(arr)))
        f.write(arr.tobytes())


def parse_ply_header(f):
    """Parse binary PLY header."""
    format_type, num_vertices, num_faces = 'ascii', 0, 0
    vertex_props, face_props = [], []
    current_element = None
    while True:
        line = f.readline().decode('ascii').strip()
        if line == 'end_header':
            break
        parts = line.split()
        if not parts:
            continue
        if parts[0] == 'format':
            format_type = parts[1]
        elif parts[0] == 'element':
            current_element = parts[1]
            if current_element == 'vertex':
                num_vertices = int(parts[2])
            elif current_element == 'face':
                num_faces = int(parts[2])
        elif parts[0] == 'property':
            if current_element == 'vertex' and len(parts) >= 3:
                vertex_props.append((parts[1], parts[2]))
            elif current_element == 'face':
                if parts[1] == 'list':
                    face_props.append(('list',))
                else:
                    face_props.append((parts[1], parts[2]))
    return {'format': format_type, 'num_vertices': num_vertices,
            'num_faces': num_faces, 'vertex_props': vertex_props,
            'face_props': face_props}


def read_ply_mesh(file_path):
    """Read binary PLY mesh -> vertices, faces, face_centers, latitudes, longitudes."""
    with open(file_path, 'rb') as f:
        hdr = parse_ply_header(f)
        nv, nf = hdr['num_vertices'], hdr['num_faces']
        endian = '<' if hdr['format'] == 'binary_little_endian' else '>'
        vfmt, bpv = [], 0
        for pt, pn in hdr['vertex_props']:
            if pt == 'float':
                vfmt.append(('f4', pn)); bpv += 4
            elif pt == 'double':
                vfmt.append(('f8', pn)); bpv += 8
        vdtype = np.dtype([(name, endian + dt) for dt, name in vfmt])
        va = np.frombuffer(f.read(nv * bpv), dtype=vdtype)
        vertices = np.column_stack([va[ax] for ax in 'xyz']).astype(np.float64)
        has_ll = any(len(p) == 2 and p[1] == 'latitude' for p in hdr['face_props'])
        faces = np.zeros((nf, 3), dtype=np.int32)
        lat_f = np.zeros(nf, np.float64) if has_ll else None
        lon_f = np.zeros(nf, np.float64) if has_ll else None
        for i in range(nf):
            cnt = struct.unpack('B', f.read(1))[0]
            idx = struct.unpack(f'{endian}{cnt}I', f.read(cnt * 4))
            faces[i] = idx[:3]
            if has_ll:
                lat_f[i] = struct.unpack(endian + 'f', f.read(4))[0]
                lon_f[i] = struct.unpack(endian + 'f', f.read(4))[0]
    fc = (vertices[faces[:, 0]] + vertices[faces[:, 1]] + vertices[faces[:, 2]]) / 3.0
    if lat_f is not None:
        return vertices, faces, fc, lat_f, lon_f
    x, y, z = fc[:, 0], fc[:, 1], fc[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    return vertices, faces, fc, np.degrees(np.arcsin(z / r)), np.degrees(np.arctan2(y, x))


def read_classification_binary(filepath):
    """Read face classification: uint32 count + uint8 array."""
    with open(filepath, 'rb') as f:
        count = struct.unpack('<I', f.read(4))[0]
        return np.frombuffer(f.read(count), dtype=np.uint8)


def load_state(state_file):
    """Load survey state from JSON."""
    if Path(state_file).exists():
        with open(state_file) as f:
            return json.load(f)
    return {}


def save_state(state_file, state):
    """Save survey state to JSON."""
    Path(state_file).parent.mkdir(parents=True, exist_ok=True)
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)


def haversine_distance(lat1, lon1, lat2, lon2, R_km=1737.4):
    """Great-circle distance between two lat/lon points in km."""
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return 2 * R_km * np.arcsin(np.sqrt(np.minimum(a, 1.0)))


def compute_face_slopes(vertices, faces, face_centers, body_radius=1737400.0):
    """Compute terrain slope (degrees) for each face from face normal vs local radial."""
    # Face normals via cross product
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    e1 = v1 - v0
    e2 = v2 - v0
    normals = np.cross(e1, e2)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / (norms + 1e-10)
    # Ensure outward-pointing
    radial = face_centers / np.linalg.norm(face_centers, axis=1, keepdims=True)
    flip = np.sum(normals * radial, axis=1) < 0
    normals[flip] *= -1
    # Slope = angle between face normal and radial direction
    cos_slope = np.clip(np.sum(normals * radial, axis=1), -1, 1)
    return np.degrees(np.arccos(cos_slope))
