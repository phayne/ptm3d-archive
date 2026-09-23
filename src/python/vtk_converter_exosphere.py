#!/usr/bin/env python3
"""
VTK Converter for PTM Exosphere Data

Converts binary exosphere simulation output to VTU format for ParaView visualization.
Supports both diurnal and multi-timescale (geological evolution) modes.

Usage:
    python vtk_converter_exosphere.py --input-dir <binary_data_dir> --output-dir <output_dir>

Binary format (per timestep file):
    [uint32 face_count] [float32 data[face_count]]

Output:
    - Per-timestep .vtu files (VTK unstructured grid with face scalars)
    - exosphere.pvd collection file for time-series animation in ParaView
"""

import os
import sys
import struct
import argparse
from pathlib import Path
import time
import numpy as np
import gc

try:
    import vtk
    from vtk.util import numpy_support
except ImportError as e:
    print(f"ERROR: Failed to import VTK")
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    print(f"Import error: {e}")
    print(f"\nPlease install VTK in this Python environment:")
    print(f"  {sys.executable} -m pip install vtk")
    sys.exit(1)


# ============================================================================
# Field registry
# ============================================================================

# Maps new field names (with units) to legacy names for backward compatibility.
# If new-name directory exists, use it; otherwise fall back to legacy if available.
FIELD_MAPPING = {
    # Temperature
    'temperature_K': 'temperature',
    'temperature_mean_K': None,
    'temperature_min_K': None,
    'temperature_max_K': None,
    # Vapor and flux
    'vapor_pressure_Pa': 'vapor_pressure',
    'emission_flux_kg_m2_s': 'emission_flux',
    'deposition_flux_kg_m2_s': 'deposition_flux',
    'net_flux_kg_m2_s': 'net_flux',
    'escape_flux_kg_m2_s': 'escape_flux',
    'accumulated_mass_kg_m2': 'accumulated_mass',
    'ice_thickness_m': 'ice_thickness',
    # Sputtering
    'sputtering_flux_kg_m2_s': 'sputtering_flux',
    'sputtering_yield': None,
    'sputtering_mass_loss_kg_m2_s': 'sputtering_mass_loss',
    'sputtering_spatial_weight': None,
    # Ice layer
    'nonice_concentration': None,
    'ice_mass_kg_m2': 'ice_mass',
    'cumulative_sublimated_kg_m2': 'cumulative_sublimated',
    'cumulative_sputtered_kg_m2': 'cumulative_sputtered',
    'cumulative_deposited_kg_m2': 'cumulative_deposited',
    # Crystallization
    'crystalline_fraction': None,
    'porosity': None,
    'radiation_dose_rate_eV_mol_s': None,
    'cumulative_dose_eV_mol': None,
    'crystallization_timescale_yr': None,
}


# ============================================================================
# PLY mesh I/O
# ============================================================================

def parse_ply_header(file):
    """Parse PLY header and return format info."""
    vertex_props = []
    face_props = []
    num_vertices = 0
    num_faces = 0
    format_type = 'ascii'
    current_element = None

    while True:
        line = file.readline()
        if isinstance(line, bytes):
            line = line.decode('ascii')
        line = line.strip()

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
            if current_element == 'vertex':
                if len(parts) >= 3:
                    vertex_props.append((parts[1], parts[2]))
            elif current_element == 'face':
                if parts[1] == 'list':
                    name = parts[4] if len(parts) > 4 else 'vertex_indices'
                    face_props.append(('list', parts[2], parts[3], name))
                else:
                    face_props.append((parts[1], parts[2]))

    return {
        'format': format_type,
        'num_vertices': num_vertices,
        'num_faces': num_faces,
        'vertex_props': vertex_props,
        'face_props': face_props,
    }


def read_binary_mesh(mesh_file):
    """Read binary PLY mesh file.

    Returns (vertices, faces, normals, latitudes, longitudes).
    Uses numpy for vertex data and struct for per-face records (variable-size
    list property prevents pure numpy parsing of the face block).
    """
    print(f"Reading mesh from {mesh_file}...")

    with open(mesh_file, 'rb') as f:
        header = parse_ply_header(f)

        print(f"  Format: {header['format']}")
        print(f"  Vertices: {header['num_vertices']:,}")
        print(f"  Faces: {header['num_faces']:,}")

        if header['format'] not in ('binary_little_endian', 'binary_big_endian'):
            raise ValueError(f"Only binary PLY formats supported, got: {header['format']}")

        endian = '<' if header['format'] == 'binary_little_endian' else '>'

        # --- Vertices (numpy bulk read) ---
        vertex_dtype = []
        for prop_type, prop_name in header['vertex_props']:
            if prop_type == 'float':
                vertex_dtype.append((prop_name, f'{endian}f4'))
            elif prop_type == 'double':
                vertex_dtype.append((prop_name, f'{endian}f8'))
            else:
                raise ValueError(f"Unsupported vertex property type: {prop_type}")

        vertex_dtype = np.dtype(vertex_dtype)
        vertex_buf = f.read(header['num_vertices'] * vertex_dtype.itemsize)
        vertex_arr = np.frombuffer(vertex_buf, dtype=vertex_dtype)

        vertices = np.empty((header['num_vertices'], 3), dtype=np.float32)
        vertices[:, 0] = vertex_arr['x']
        vertices[:, 1] = vertex_arr['y']
        vertices[:, 2] = vertex_arr['z']

        normals = None
        if 'nx' in vertex_arr.dtype.names:
            normals = np.empty((header['num_vertices'], 3), dtype=np.float32)
            normals[:, 0] = vertex_arr['nx']
            normals[:, 1] = vertex_arr['ny']
            normals[:, 2] = vertex_arr['nz']

        # --- Faces ---
        # Determine per-face scalar properties (after the list property)
        face_scalar_props = []
        for prop in header['face_props']:
            if prop[0] != 'list':
                face_scalar_props.append(prop)

        has_lat_lon = any(p[1] == 'latitude' for p in face_scalar_props)

        # Compute bytes per face record: 1 (count) + 12 (3×uint32) + scalars
        scalar_bytes = 0
        for ptype, _ in face_scalar_props:
            scalar_bytes += 8 if ptype == 'double' else 4

        bytes_per_face = 1 + 12 + scalar_bytes
        face_buf = f.read(header['num_faces'] * bytes_per_face)

        faces = np.empty((header['num_faces'], 3), dtype=np.int32)
        latitudes = np.empty(header['num_faces'], dtype=np.float32) if has_lat_lon else None
        longitudes = np.empty(header['num_faces'], dtype=np.float32) if has_lat_lon else None

        offset = 0
        for i in range(header['num_faces']):
            # Skip vertex count byte (always 3 for triangles)
            offset += 1
            faces[i] = struct.unpack_from(f'{endian}III', face_buf, offset)
            offset += 12
            if has_lat_lon:
                latitudes[i] = struct.unpack_from(f'{endian}f', face_buf, offset)[0]
                offset += 4
                longitudes[i] = struct.unpack_from(f'{endian}f', face_buf, offset)[0]
                offset += 4
            else:
                offset += scalar_bytes

        # Compute lat/lon from face centers if not embedded
        if latitudes is None:
            print("  Computing latitude/longitude from face centers...")
            centers = (vertices[faces[:, 0]] + vertices[faces[:, 1]] + vertices[faces[:, 2]]) / 3.0
            r = np.linalg.norm(centers, axis=1)
            r[r == 0] = 1.0
            latitudes = np.degrees(np.arcsin(centers[:, 2] / r)).astype(np.float32)
            longitudes = np.degrees(np.arctan2(centers[:, 1], centers[:, 0])).astype(np.float32)

        # Compute normals if not in file
        if normals is None:
            print("  Computing vertex normals...")
            norms = np.linalg.norm(vertices, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            normals = (vertices / norms).astype(np.float32)

    print(f"  Mesh loaded successfully")
    return vertices, faces, normals, latitudes, longitudes


# ============================================================================
# Binary timestep I/O
# ============================================================================

def read_binary_timestep(file_path, expected_count=None):
    """Read a single binary timestep file: [uint32 count][float32 data[count]]."""
    if not os.path.exists(file_path):
        return None

    try:
        with open(file_path, 'rb') as f:
            count = struct.unpack('<I', f.read(4))[0]
            if expected_count and count != expected_count:
                print(f"  Warning: count mismatch in {file_path}: expected {expected_count}, got {count}")
            data = np.frombuffer(f.read(count * 4), dtype=np.float32)
            return data
    except Exception as e:
        print(f"  Error reading {file_path}: {e}")
        return None


# ============================================================================
# VTK grid construction (numpy_support for performance)
# ============================================================================

def create_vtk_grid(vertices, faces, normals, latitudes, longitudes):
    """Create VTK unstructured grid using numpy_support for fast array transfer."""
    ugrid = vtk.vtkUnstructuredGrid()

    # Points
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_support.numpy_to_vtk(vertices, deep=True))
    ugrid.SetPoints(vtk_points)

    # Cells (triangles) — build connectivity array [3, v0, v1, v2, ...]
    cell_array = np.empty((len(faces), 4), dtype=np.int64)
    cell_array[:, 0] = 3
    cell_array[:, 1:4] = faces
    cells = vtk.vtkCellArray()
    cells.SetCells(len(faces), numpy_support.numpy_to_vtk(
        cell_array.ravel(), deep=True, array_type=vtk.VTK_ID_TYPE))
    ugrid.SetCells(vtk.VTK_TRIANGLE, cells)

    # Cell data: latitude and longitude
    if latitudes is not None:
        lat_arr = numpy_support.numpy_to_vtk(latitudes, deep=True)
        lat_arr.SetName("latitude")
        ugrid.GetCellData().AddArray(lat_arr)

    if longitudes is not None:
        lon_arr = numpy_support.numpy_to_vtk(longitudes, deep=True)
        lon_arr.SetName("longitude")
        ugrid.GetCellData().AddArray(lon_arr)

    return ugrid


def write_vtk_timestep(ugrid, data_dict, output_file):
    """Add field arrays to grid, write .vtu, then remove arrays for reuse."""
    for name, data in data_dict.items():
        if data is not None:
            vtk_arr = numpy_support.numpy_to_vtk(data, deep=False)
            vtk_arr.SetName(name)
            ugrid.GetCellData().AddArray(vtk_arr)

    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(output_file)
    writer.SetInputData(ugrid)
    writer.SetDataModeToAppended()
    writer.SetCompressorTypeToZLib()
    writer.SetCompressionLevel(1)
    writer.SetEncodeAppendedData(0)
    writer.Write()

    for name in data_dict:
        if data_dict[name] is not None:
            ugrid.GetCellData().RemoveArray(name)


# ============================================================================
# Field and timestep discovery
# ============================================================================

def discover_fields(input_dir):
    """Find available data fields, preferring new names over legacy.

    Returns list of (field_name, dir_path) tuples.
    """
    input_dir = Path(input_dir)
    fields = []
    used_legacy = set()

    for new_name, legacy_name in FIELD_MAPPING.items():
        new_dir = input_dir / new_name
        if new_dir.is_dir():
            fields.append((new_name, new_dir))
            if legacy_name:
                used_legacy.add(legacy_name)
        elif legacy_name:
            legacy_dir = input_dir / legacy_name
            if legacy_dir.is_dir() and legacy_name not in used_legacy:
                fields.append((legacy_name, legacy_dir))
                used_legacy.add(legacy_name)

    return fields


def discover_timesteps(input_dir):
    """Determine available timestep indices from a temperature field directory.

    Handles both diurnal (temperature_K) and multiscale (temperature_mean_K) modes.
    Returns (sorted list of timestep ints, field_name used for discovery).
    """
    input_dir = Path(input_dir)
    for temp_name in ('temperature_K', 'temperature_mean_K', 'temperature'):
        temp_dir = input_dir / temp_name
        if temp_dir.is_dir():
            indices = sorted(
                int(f.stem.split('_')[-1])
                for f in temp_dir.glob(f"{temp_name}_*.bin")
            )
            if indices:
                return indices, temp_name
    return [], None


def detect_timestep_width(fields, timesteps):
    """Detect zero-padding width used in filenames (typically 4)."""
    if not fields or not timesteps:
        return 4  # default
    field_name, field_dir = fields[0]
    # Check the last timestep to find the widest number
    t = timesteps[-1]
    for width in (4, 5, 6, 3):
        candidate = field_dir / f"{field_name}_{t:0{width}d}.bin"
        if candidate.exists():
            return width
    return 4


def read_time_steps_txt(input_dir):
    """Read time_steps.txt written by C++ exosphere code.

    The file contains one time value per line (in seconds),
    with optional comment lines starting with #.
    Returns dict mapping timestep_index → time_in_seconds.
    """
    ts_file = Path(input_dir) / "time_steps.txt"
    time_values = {}
    if not ts_file.exists():
        return time_values
    try:
        idx = 0
        with open(ts_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    time_values[idx] = float(line)
                    idx += 1
        print(f"  Read {len(time_values)} time values from time_steps.txt")
    except Exception as e:
        print(f"  Warning: could not read time_steps.txt: {e}")
    return time_values


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Convert PTM exosphere binary data to VTU format for ParaView',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert all timesteps
  python vtk_converter_exosphere.py --input-dir exosphere_output --output-dir vtk_output

  # Convert specific timesteps
  python vtk_converter_exosphere.py --input-dir exosphere_output --output-dir vtk_output --timesteps 0-10,100,200

  # Test with first timestep only
  python vtk_converter_exosphere.py --input-dir exosphere_output --output-dir vtk_output --test-mode
        """)
    parser.add_argument('--input-dir', required=True,
                        help='Directory containing binary exosphere output')
    parser.add_argument('--output-dir', required=True,
                        help='Output directory for VTU files')
    parser.add_argument('--timesteps', type=str, default=None,
                        help='Timesteps to convert (e.g., "0-10,20,100")')
    parser.add_argument('--test-mode', action='store_true',
                        help='Only convert the first timestep')

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Mesh ----
    mesh_file = input_dir / "mesh.ply"
    if not mesh_file.exists():
        print(f"Error: mesh file not found at {mesh_file}")
        return 1

    try:
        vertices, faces, normals, latitudes, longitudes = read_binary_mesh(str(mesh_file))
    except Exception as e:
        print(f"Error reading mesh: {e}")
        return 1

    num_faces = len(faces)

    # ---- Discover fields and timesteps ----
    fields = discover_fields(input_dir)
    if not fields:
        print("Error: no data fields found in input directory")
        return 1

    available_timesteps, temp_field = discover_timesteps(input_dir)
    if not available_timesteps:
        print("Error: no timestep data found (checked temperature_K, temperature_mean_K, temperature)")
        return 1

    # Detect mode
    mode = "multiscale" if temp_field == "temperature_mean_K" else "diurnal"
    pad_width = detect_timestep_width(fields, available_timesteps)

    field_names = [f[0] for f in fields]
    print(f"\nMode: {mode}")
    print(f"Found {len(available_timesteps)} timesteps (padding: {pad_width} digits)")
    print(f"Available fields ({len(fields)}): {', '.join(field_names)}")

    # ---- Select timesteps ----
    ts_set = set(available_timesteps)
    if args.test_mode:
        timesteps_to_process = [available_timesteps[0]]
        print("Test mode: converting only first timestep")
    elif args.timesteps:
        timesteps_to_process = []
        for part in args.timesteps.split(','):
            part = part.strip()
            if '-' in part:
                start, end = map(int, part.split('-'))
                timesteps_to_process.extend(t for t in available_timesteps if start <= t <= end)
            else:
                t = int(part)
                if t in ts_set:
                    timesteps_to_process.append(t)
    else:
        timesteps_to_process = available_timesteps

    print(f"Converting {len(timesteps_to_process)} timesteps...\n")

    # ---- Build base VTK grid ----
    print("Creating VTK grid...")
    base_grid = create_vtk_grid(vertices, faces, normals, latitudes, longitudes)

    # ---- Convert timesteps ----
    print("Processing timesteps:")
    start_time = time.time()

    for idx, t in enumerate(timesteps_to_process):
        pct = 100.0 * (idx + 1) / len(timesteps_to_process)
        print(f"  [{pct:5.1f}%] Timestep {t} ({idx + 1}/{len(timesteps_to_process)})...",
              end='', flush=True)

        data = {}
        for field_name, field_dir in fields:
            file_path = field_dir / f"{field_name}_{t:0{pad_width}d}.bin"
            data[field_name] = read_binary_timestep(str(file_path), num_faces)

        output_file = str(output_dir / f"exosphere_{t:0{pad_width}d}.vtu")
        write_vtk_timestep(base_grid, data, output_file)
        print(" done")

        del data
        if idx % 50 == 0:
            gc.collect()

    # ---- PVD collection file ----
    if len(timesteps_to_process) > 1:
        pvd_file = output_dir / "exosphere.pvd"
        print(f"\nCreating PVD file: {pvd_file}")

        time_values = read_time_steps_txt(input_dir)
        # time_steps.txt lists one time per snapshot, in label order (PTM-636), so
        # the k-th entry belongs to the k-th AVAILABLE label -- not to the label
        # whose value happens to equal k (with labels 0, 2, 4, ... a label lookup
        # gave label 2 the time of the third snapshot).
        by_label = {}
        if len(time_values) == len(available_timesteps):
            by_label = {t: time_values[k] for k, t in enumerate(available_timesteps)}
        elif time_values:
            print(f"  WARNING: time_steps.txt has {len(time_values)} entries for "
                  f"{len(available_timesteps)} snapshots (written by a build before PTM-636, "
                  f"which omitted the final snapshot and held nominal times); PVD times are "
                  f"approximate")

        with open(pvd_file, 'w') as f:
            f.write('<?xml version="1.0"?>\n')
            f.write('<VTKFile type="Collection" version="0.1">\n')
            f.write('  <Collection>\n')
            for idx, t in enumerate(timesteps_to_process):
                if t in by_label:
                    tv = by_label[t]
                elif available_timesteps.index(t) in time_values:
                    tv = time_values[available_timesteps.index(t)]
                else:
                    tv = float(t) * 3600.0  # fallback: timestep × 1 hour
                    print(f"  WARNING: no time for snapshot {t}; PVD uses {tv:.1f} s")
                f.write(f'    <DataSet timestep="{tv}" '
                        f'file="exosphere_{t:0{pad_width}d}.vtu"/>\n')
            f.write('  </Collection>\n')
            f.write('</VTKFile>\n')

    # ---- Summary ----
    elapsed = time.time() - start_time
    print(f"\nConversion completed in {elapsed:.1f} seconds")
    if len(timesteps_to_process) > 0:
        print(f"Average: {elapsed / len(timesteps_to_process):.3f} seconds per timestep")
    print(f"Output files written to: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
