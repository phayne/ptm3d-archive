#!/usr/bin/env python3
"""
Extract time series data from exosphere VTU files for specified lat/lon locations.

Usage:
    # Single location
    python extract_timeseries.py --input-dir output/simulation --lat 45 --lon 180

    # Multiple locations from command line
    python extract_timeseries.py --input-dir output/simulation --locations "0,0;45,180;-30,90"

    # Multiple locations from file (one "lat,lon" per line)
    python extract_timeseries.py --input-dir output/simulation --locations-file locations.txt

    # Specify fields to extract
    python extract_timeseries.py --input-dir output/simulation --lat 0 --lon 180 --fields temperature_mean_K,nonice_concentration

    # Output to CSV
    python extract_timeseries.py --input-dir output/simulation --lat 0 --lon 180 --output timeseries.csv
"""

import argparse
import numpy as np
from pathlib import Path
import struct
import sys


def parse_ply_header(f):
    """Parse PLY header and return metadata."""
    header_lines = []
    format_type = 'ascii'
    num_vertices = 0
    num_faces = 0
    vertex_props = []
    face_props = []
    current_element = None

    while True:
        line = f.readline().decode('ascii').strip()
        header_lines.append(line)
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
                    prop_type = parts[1]
                    prop_name = parts[2]
                    vertex_props.append((prop_type, prop_name))
            elif current_element == 'face':
                if parts[1] == 'list':
                    face_props.append(('list', parts[2], parts[3], parts[4] if len(parts) > 4 else 'vertex_indices'))
                else:
                    face_props.append((parts[1], parts[2]))

    return {
        'format': format_type,
        'num_vertices': num_vertices,
        'num_faces': num_faces,
        'vertex_props': vertex_props,
        'face_props': face_props,
        'header_lines': header_lines
    }


def read_binary_mesh_flexible(file_path):
    """Read PLY mesh file (binary or ASCII) and extract vertices, faces, and compute lat/lon."""
    with open(file_path, 'rb') as f:
        # Parse header
        header = parse_ply_header(f)

        num_vertices = header['num_vertices']
        num_faces = header['num_faces']

        if header['format'] not in ['binary_little_endian', 'binary_big_endian', 'ascii']:
            raise ValueError(f"Unsupported PLY format: {header['format']}")

        is_little_endian = (header['format'] == 'binary_little_endian')
        endian = '<' if is_little_endian else '>'

        # Determine bytes per vertex based on properties
        bytes_per_vertex = 0
        vertex_format = []
        for prop_type, prop_name in header['vertex_props']:
            if prop_type == 'float':
                vertex_format.append(('f4', prop_name))
                bytes_per_vertex += 4
            elif prop_type == 'double':
                vertex_format.append(('f8', prop_name))
                bytes_per_vertex += 8
            else:
                raise ValueError(f"Unsupported vertex property type: {prop_type}")

        # Read vertices
        if 'binary' in header['format']:
            vertex_dtype = np.dtype([(name, endian + dtype) for dtype, name in vertex_format])
            vertex_buffer = f.read(num_vertices * bytes_per_vertex)
            vertex_array = np.frombuffer(vertex_buffer, dtype=vertex_dtype)

            vertices = np.zeros((num_vertices, 3), dtype=np.float64)
            if 'x' in vertex_array.dtype.names:
                vertices[:, 0] = vertex_array['x']
            if 'y' in vertex_array.dtype.names:
                vertices[:, 1] = vertex_array['y']
            if 'z' in vertex_array.dtype.names:
                vertices[:, 2] = vertex_array['z']
        else:
            # ASCII format
            vertices = np.zeros((num_vertices, 3), dtype=np.float64)
            for i in range(num_vertices):
                line = f.readline().decode('ascii').strip()
                parts = line.split()
                vertices[i] = [float(parts[0]), float(parts[1]), float(parts[2])]

        # Check for lat/lon in face properties
        has_lat_lon = False
        for prop in header['face_props']:
            if len(prop) == 2 and prop[1] == 'latitude':
                has_lat_lon = True
                break

        latitudes_from_file = None
        longitudes_from_file = None
        if has_lat_lon:
            latitudes_from_file = np.zeros(num_faces, dtype=np.float64)
            longitudes_from_file = np.zeros(num_faces, dtype=np.float64)

        # Read faces
        faces = np.zeros((num_faces, 3), dtype=np.int32)

        if 'binary' in header['format']:
            for i in range(num_faces):
                # Read vertex count (first byte for list property)
                count = struct.unpack('B', f.read(1))[0]
                # Read vertex indices (unsigned int)
                if count >= 3:
                    indices = struct.unpack(f'{endian}{count}I', f.read(count * 4))
                    faces[i] = indices[:3]
                else:
                    f.read(count * 4)
                    faces[i] = [0, 0, 0]

                # Read additional face properties (lat/lon) if present
                if has_lat_lon:
                    lat = struct.unpack(endian + 'f', f.read(4))[0]
                    lon = struct.unpack(endian + 'f', f.read(4))[0]
                    latitudes_from_file[i] = lat
                    longitudes_from_file[i] = lon
        else:
            # ASCII format
            for i in range(num_faces):
                line = f.readline().decode('ascii').strip()
                parts = line.split()
                faces[i] = [int(parts[1]), int(parts[2]), int(parts[3])]
                if has_lat_lon and len(parts) >= 6:
                    latitudes_from_file[i] = float(parts[4])
                    longitudes_from_file[i] = float(parts[5])

    # Compute face centers
    face_centers = np.zeros((num_faces, 3), dtype=np.float64)
    for i in range(num_faces):
        v0, v1, v2 = faces[i]
        face_centers[i] = (vertices[v0] + vertices[v1] + vertices[v2]) / 3.0

    # Use lat/lon from file if available, otherwise compute from face centers
    if latitudes_from_file is not None:
        latitudes = latitudes_from_file
        longitudes = longitudes_from_file
    else:
        # Convert to lat/lon (assuming spherical body centered at origin)
        x, y, z = face_centers[:, 0], face_centers[:, 1], face_centers[:, 2]
        r = np.sqrt(x**2 + y**2 + z**2)

        latitudes = np.degrees(np.arcsin(z / r))
        longitudes = np.degrees(np.arctan2(y, x))

    # Normalize longitudes to [0, 360)
    longitudes = np.mod(longitudes, 360.0)

    return vertices, faces, face_centers, latitudes, longitudes


def find_nearest_face(target_lat, target_lon, latitudes, longitudes):
    """Find the face index nearest to the target lat/lon."""
    # Normalize target longitude to [0, 360)
    target_lon = target_lon % 360.0

    # Compute angular distance using haversine-like formula
    lat1 = np.radians(target_lat)
    lon1 = np.radians(target_lon)
    lat2 = np.radians(latitudes)
    lon2 = np.radians(longitudes)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    angular_dist = 2 * np.arcsin(np.sqrt(a))

    nearest_idx = np.argmin(angular_dist)
    min_dist_deg = np.degrees(angular_dist[nearest_idx])

    return nearest_idx, min_dist_deg


def read_binary_timestep(file_path, num_faces):
    """Read a binary data file for a single timestep.

    File format (from C++ writeBinaryArray):
    - 4 bytes: uint32 face count header
    - N * 4 bytes: float32 data values
    """
    try:
        with open(file_path, 'rb') as f:
            # Read header (uint32 face count)
            header = f.read(4)
            if len(header) < 4:
                return None
            count = np.frombuffer(header, dtype=np.uint32)[0]

            # Read float32 data
            data = np.frombuffer(f.read(), dtype=np.float32)

            if len(data) == count == num_faces:
                return data.astype(np.float64)  # Return as float64 for consistency
            else:
                return None
    except FileNotFoundError:
        return None


def read_time_steps(input_dir):
    """Read time values from time_steps.txt."""
    time_file = input_dir / "time_steps.txt"
    time_values = []

    if time_file.exists():
        with open(time_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    time_values.append(float(line))

    return time_values


def discover_fields(input_dir):
    """Discover available data fields from directory structure."""
    fields = []
    for item in input_dir.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            # Check if it contains .bin files
            bin_files = list(item.glob("*.bin"))
            if bin_files:
                fields.append(item.name)
    return sorted(fields)


def discover_timesteps(input_dir, fields):
    """Discover available timesteps from the first available field.

    Returns a dict mapping timestep number to the actual filename suffix used.
    This handles both zero-padded (0000) and non-padded (10000) numbering.
    """
    if not fields:
        return {}, []

    field_dir = input_dir / fields[0]
    timestep_to_suffix = {}

    for f in field_dir.glob(f"{fields[0]}_*.bin"):
        try:
            # Extract the suffix after the last underscore (e.g., "0000" or "10000")
            suffix = f.stem.split('_')[-1]
            ts = int(suffix)
            timestep_to_suffix[ts] = suffix
        except ValueError:
            continue

    sorted_timesteps = sorted(timestep_to_suffix.keys())
    return timestep_to_suffix, sorted_timesteps


def parse_locations(args):
    """Parse location arguments into list of (lat, lon, label) tuples."""
    locations = []

    if args.lat is not None and args.lon is not None:
        locations.append((args.lat, args.lon, f"lat{args.lat}_lon{args.lon}"))

    if args.locations:
        for loc_str in args.locations.split(';'):
            parts = loc_str.strip().split(',')
            if len(parts) >= 2:
                lat = float(parts[0])
                lon = float(parts[1])
                label = parts[2] if len(parts) > 2 else f"lat{lat}_lon{lon}"
                locations.append((lat, lon, label))

    if args.locations_file:
        loc_file = Path(args.locations_file)
        if loc_file.exists():
            with open(loc_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        parts = line.split(',')
                        if len(parts) >= 2:
                            lat = float(parts[0])
                            lon = float(parts[1])
                            label = parts[2].strip() if len(parts) > 2 else f"lat{lat}_lon{lon}"
                            locations.append((lat, lon, label))

    return locations


def main():
    parser = argparse.ArgumentParser(
        description='Extract time series data from exosphere VTU files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--input-dir', required=True, help='Directory containing simulation output')
    parser.add_argument('--lat', type=float, help='Latitude of point to extract')
    parser.add_argument('--lon', type=float, help='Longitude of point to extract')
    parser.add_argument('--locations', type=str,
                        help='Semicolon-separated list of lat,lon pairs (e.g., "0,0;45,180")')
    parser.add_argument('--locations-file', type=str,
                        help='File with lat,lon pairs (one per line, optional label as third column)')
    parser.add_argument('--fields', type=str,
                        help='Comma-separated list of fields to extract (default: all)')
    parser.add_argument('--output', '-o', type=str, help='Output CSV file (default: stdout)')
    parser.add_argument('--list-fields', action='store_true', help='List available fields and exit')
    parser.add_argument('--time-unit', choices=['seconds', 'days', 'years'], default='years',
                        help='Time unit for output (default: years)')

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}", file=sys.stderr)
        return 1

    # Discover available fields
    available_fields = discover_fields(input_dir)

    if args.list_fields:
        print("Available fields:")
        for field in available_fields:
            print(f"  {field}")
        return 0

    if not available_fields:
        print("Error: No data fields found in input directory", file=sys.stderr)
        return 1

    # Parse locations
    locations = parse_locations(args)
    if not locations:
        print("Error: No locations specified. Use --lat/--lon, --locations, or --locations-file",
              file=sys.stderr)
        return 1

    # Parse fields to extract
    if args.fields:
        fields = [f.strip() for f in args.fields.split(',')]
        # Validate fields
        for field in fields:
            if field not in available_fields:
                print(f"Warning: Field '{field}' not found. Available: {', '.join(available_fields)}",
                      file=sys.stderr)
        fields = [f for f in fields if f in available_fields]
    else:
        fields = available_fields

    if not fields:
        print("Error: No valid fields to extract", file=sys.stderr)
        return 1

    # Load mesh
    mesh_file = input_dir / "mesh.ply"
    if not mesh_file.exists():
        print(f"Error: Mesh file not found: {mesh_file}", file=sys.stderr)
        return 1

    print(f"Loading mesh from {mesh_file}...", file=sys.stderr)
    vertices, faces, face_centers, latitudes, longitudes = read_binary_mesh_flexible(str(mesh_file))
    num_faces = len(faces)
    print(f"  Loaded {num_faces} faces", file=sys.stderr)

    # Find nearest faces for each location
    face_indices = []
    for lat, lon, label in locations:
        idx, dist = find_nearest_face(lat, lon, latitudes, longitudes)
        actual_lat = latitudes[idx]
        actual_lon = longitudes[idx]
        print(f"  Location '{label}': requested ({lat:.2f}, {lon:.2f}) -> "
              f"face {idx} at ({actual_lat:.2f}, {actual_lon:.2f}), dist={dist:.3f}°",
              file=sys.stderr)
        face_indices.append((idx, label, actual_lat, actual_lon))

    # Discover timesteps
    timestep_to_suffix, timesteps = discover_timesteps(input_dir, fields)
    if not timesteps:
        print("Error: No timesteps found", file=sys.stderr)
        return 1
    print(f"  Found {len(timesteps)} timesteps", file=sys.stderr)

    # Read time values from time_steps.txt or compute from epoch numbers
    time_values_from_file = read_time_steps(input_dir)

    # Try to match time values to timesteps
    # time_steps.txt may not include timestep 0 (initial state), so we need to handle that
    if len(time_values_from_file) == len(timesteps):
        time_values = time_values_from_file
    elif len(time_values_from_file) == len(timesteps) - 1 and timesteps[0] == 0:
        # time_steps.txt doesn't include initial timestep 0
        time_values = [0.0] + time_values_from_file
        print(f"  Note: Added t=0 for initial timestep", file=sys.stderr)
    else:
        # Compute time from epoch numbers
        # Each epoch = cycles_per_epoch * diurnal_period
        # Default: 100 cycles * 3.55 days = 355.12 days ≈ 0.972 years
        SECONDS_PER_YEAR = 365.25 * 24 * 3600
        DIURNAL_PERIOD_SECONDS = 3.5512 * 24 * 3600  # Europa's orbital period
        CYCLES_PER_EPOCH = 100
        EPOCH_DURATION_SECONDS = CYCLES_PER_EPOCH * DIURNAL_PERIOD_SECONDS

        time_values = [ts * EPOCH_DURATION_SECONDS for ts in timesteps]
        print(f"  Note: Computing time from epoch numbers "
              f"(time_steps.txt has {len(time_values_from_file)} entries, "
              f"found {len(timesteps)} timesteps)", file=sys.stderr)

    # Convert time units
    time_scale = 1.0
    time_label = "time_seconds"
    if args.time_unit == 'days':
        time_scale = 1.0 / 86400.0
        time_label = "time_days"
    elif args.time_unit == 'years':
        time_scale = 1.0 / (365.25 * 86400.0)
        time_label = "time_years"

    # Extract data
    print(f"\nExtracting {len(fields)} fields for {len(locations)} locations...", file=sys.stderr)

    # Build header
    header_parts = [time_label, "timestep_idx"]
    for idx, label, actual_lat, actual_lon in face_indices:
        for field in fields:
            header_parts.append(f"{label}_{field}")

    # Open output
    if args.output:
        out_file = open(args.output, 'w')
    else:
        out_file = sys.stdout

    # Write header
    out_file.write(','.join(header_parts) + '\n')

    # Process each timestep
    for ts_idx, ts in enumerate(timesteps):
        row = [f"{time_values[ts_idx] * time_scale:.6e}", str(ts)]

        # Get the actual filename suffix for this timestep
        suffix = timestep_to_suffix[ts]

        for face_idx, label, _, _ in face_indices:
            for field in fields:
                file_path = input_dir / field / f"{field}_{suffix}.bin"
                data = read_binary_timestep(file_path, num_faces)
                if data is not None:
                    row.append(f"{data[face_idx]:.6e}")
                else:
                    row.append("NaN")

        out_file.write(','.join(row) + '\n')

    if args.output:
        out_file.close()
        print(f"\nOutput written to: {args.output}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
