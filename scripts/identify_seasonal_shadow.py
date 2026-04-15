#!/usr/bin/env python3
"""
Phase 1 Post-Processing: Identify seasonally shadowed faces at Amundsen crater.

Reads per-lunation binary illumination statistics from the survey output,
classifies each face as PSR / seasonally shadowed / always illuminated,
clusters contiguous target faces, and writes a locations file for Phase 2.

Usage:
    python3 identify_seasonal_shadow.py \
        --survey-dir output/survey \
        --mesh ../dtm/LDEM_80S_20M__amundsen_2Mpts.ply \
        --num-lunations 25 \
        --output-locations configs/amundsen_target_locations.csv
"""

import argparse
import struct
import sys
from collections import deque
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# PLY mesh reading (adapted from src/python/extract_timeseries.py)
# ---------------------------------------------------------------------------

def parse_ply_header(f):
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
                    vertex_props.append((parts[1], parts[2]))
            elif current_element == 'face':
                if parts[1] == 'list':
                    face_props.append(('list', parts[2], parts[3],
                                       parts[4] if len(parts) > 4 else 'vertex_indices'))
                else:
                    face_props.append((parts[1], parts[2]))
    return {
        'format': format_type,
        'num_vertices': num_vertices,
        'num_faces': num_faces,
        'vertex_props': vertex_props,
        'face_props': face_props,
    }


def read_ply_mesh(file_path):
    """Read PLY mesh, return vertices, faces, face_centers, latitudes, longitudes."""
    with open(file_path, 'rb') as f:
        header = parse_ply_header(f)
        nv = header['num_vertices']
        nf = header['num_faces']
        is_le = header['format'] == 'binary_little_endian'
        endian = '<' if is_le else '>'

        # Build vertex dtype
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

        vertex_dtype = np.dtype([(name, endian + dtype) for dtype, name in vertex_format])
        vertices = np.zeros((nv, 3), dtype=np.float64)
        if 'binary' in header['format']:
            buf = f.read(nv * bytes_per_vertex)
            va = np.frombuffer(buf, dtype=vertex_dtype)
            for i, ax in enumerate('xyz'):
                if ax in va.dtype.names:
                    vertices[:, i] = va[ax]
        else:
            for i in range(nv):
                parts = f.readline().decode('ascii').split()
                vertices[i] = [float(parts[j]) for j in range(3)]

        # Check for face lat/lon
        has_lat_lon = any(len(p) == 2 and p[1] == 'latitude' for p in header['face_props'])

        faces = np.zeros((nf, 3), dtype=np.int32)
        lat_from_file = np.zeros(nf) if has_lat_lon else None
        lon_from_file = np.zeros(nf) if has_lat_lon else None

        if 'binary' in header['format']:
            for i in range(nf):
                count = struct.unpack('B', f.read(1))[0]
                if count >= 3:
                    indices = struct.unpack(f'{endian}{count}I', f.read(count * 4))
                    faces[i] = indices[:3]
                else:
                    f.read(count * 4)
                if has_lat_lon:
                    lat_from_file[i] = struct.unpack(endian + 'f', f.read(4))[0]
                    lon_from_file[i] = struct.unpack(endian + 'f', f.read(4))[0]
        else:
            for i in range(nf):
                parts = f.readline().decode('ascii').split()
                faces[i] = [int(parts[1]), int(parts[2]), int(parts[3])]
                if has_lat_lon and len(parts) >= 6:
                    lat_from_file[i] = float(parts[4])
                    lon_from_file[i] = float(parts[5])

    # Face centers
    fc = (vertices[faces[:, 0]] + vertices[faces[:, 1]] + vertices[faces[:, 2]]) / 3.0

    if lat_from_file is not None:
        return vertices, faces, fc, lat_from_file, lon_from_file
    else:
        x, y, z = fc[:, 0], fc[:, 1], fc[:, 2]
        r = np.sqrt(x**2 + y**2 + z**2)
        lats = np.degrees(np.arcsin(z / r))
        lons = np.degrees(np.arctan2(y, x))
        return vertices, faces, fc, lats, lons


# ---------------------------------------------------------------------------
# Binary statistics reading
# ---------------------------------------------------------------------------

def read_stats_binary(filepath):
    """Read a binary statistics file: uint32 count followed by count float32 values."""
    with open(filepath, 'rb') as f:
        count = struct.unpack('<I', f.read(4))[0]
        data = np.frombuffer(f.read(count * 4), dtype=np.float32)
    return data


# ---------------------------------------------------------------------------
# Face adjacency and clustering
# ---------------------------------------------------------------------------

def build_adjacency(faces, num_faces):
    """Build face adjacency via shared edges."""
    from collections import defaultdict
    edge_to_faces = defaultdict(list)
    for fi in range(num_faces):
        v = tuple(sorted(faces[fi]))
        for e in [(v[0], v[1]), (v[0], v[2]), (v[1], v[2])]:
            edge_to_faces[e].append(fi)

    adj = [[] for _ in range(num_faces)]
    for face_list in edge_to_faces.values():
        for i in range(len(face_list)):
            for j in range(i + 1, len(face_list)):
                adj[face_list[i]].append(face_list[j])
                adj[face_list[j]].append(face_list[i])
    return adj


def cluster_faces(face_indices, adj):
    """BFS clustering of face indices using adjacency."""
    face_set = set(face_indices)
    visited = set()
    clusters = []

    for fi in face_indices:
        if fi in visited:
            continue
        cluster = []
        queue = deque([fi])
        visited.add(fi)
        while queue:
            node = queue.popleft()
            cluster.append(node)
            for nb in adj[node]:
                if nb in face_set and nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        clusters.append(cluster)

    return clusters


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Classify Amundsen crater faces by seasonal shadow pattern')
    parser.add_argument('--survey-dir', required=True, help='Base directory of survey output')
    parser.add_argument('--mesh', required=True, help='Path to PLY mesh file')
    parser.add_argument('--num-lunations', type=int, default=25, help='Number of lunation directories')
    parser.add_argument('--min-dark-lunations', type=int, default=12,
                        help='Minimum dark lunations to qualify as seasonally shadowed')
    parser.add_argument('--output-locations', default='configs/amundsen_target_locations.csv',
                        help='Output locations CSV for Phase 2')
    parser.add_argument('--output-classification', default=None,
                        help='Output per-face classification binary (default: survey-dir/face_classification.bin)')
    parser.add_argument('--min-cluster-size', type=int, default=1,
                        help='Minimum number of faces per cluster to include')
    args = parser.parse_args()

    survey_dir = Path(args.survey_dir)
    if args.output_classification is None:
        args.output_classification = str(survey_dir / 'face_classification.bin')

    # Read mesh
    print(f"Reading mesh: {args.mesh}")
    vertices, faces, face_centers, latitudes, longitudes = read_ply_mesh(args.mesh)
    num_faces = len(faces)
    print(f"  {num_faces} faces, {len(vertices)} vertices")

    # Read per-lunation illumination statistics
    print(f"\nReading illumination statistics from {args.num_lunations} lunations...")
    illuminated = np.zeros((args.num_lunations, num_faces), dtype=bool)
    avg_flux = np.zeros((args.num_lunations, num_faces), dtype=np.float32)

    for i in range(args.num_lunations):
        lunation_dir = survey_dir / f'lunation_{i:02d}'
        stats_file = lunation_dir / 'statistics' / 'direct_solar_avg.bin'
        if not stats_file.exists():
            print(f"  WARNING: {stats_file} not found, skipping lunation {i}")
            continue
        avg = read_stats_binary(stats_file)
        if len(avg) != num_faces:
            print(f"  WARNING: Lunation {i} has {len(avg)} values, expected {num_faces}")
            continue
        avg_flux[i] = avg
        illuminated[i] = avg > 0
        n_lit = np.sum(illuminated[i])
        print(f"  Lunation {i:02d}: {n_lit:,} illuminated faces ({100*n_lit/num_faces:.1f}%)")

    # Classify faces
    n_illuminated = np.sum(illuminated, axis=0)  # per face: how many lunations illuminated
    n_dark = args.num_lunations - n_illuminated

    psr_mask = (n_illuminated == 0)
    seasonal_mask = (n_dark >= args.min_dark_lunations) & (n_illuminated >= 1)
    always_lit_mask = ~psr_mask & ~seasonal_mask

    n_psr = np.sum(psr_mask)
    n_seasonal = np.sum(seasonal_mask)
    n_always = np.sum(always_lit_mask)

    print(f"\n=== Face Classification ===")
    print(f"  PSR (never illuminated):      {n_psr:>10,} ({100*n_psr/num_faces:.1f}%)")
    print(f"  Seasonally shadowed (target):  {n_seasonal:>10,} ({100*n_seasonal/num_faces:.1f}%)")
    print(f"  Always illuminated:            {n_always:>10,} ({100*n_always/num_faces:.1f}%)")

    if n_seasonal == 0:
        print("\nNo seasonally shadowed faces found. Try adjusting --min-dark-lunations.")
        sys.exit(1)

    # Save per-face classification: 0=PSR, 1=seasonal, 2=always_lit
    classification = np.zeros(num_faces, dtype=np.uint8)
    classification[seasonal_mask] = 1
    classification[always_lit_mask] = 2
    with open(args.output_classification, 'wb') as f:
        f.write(struct.pack('<I', num_faces))
        f.write(classification.tobytes())
    print(f"\nFace classification written to: {args.output_classification}")

    # Cluster contiguous seasonal faces
    print("\nBuilding face adjacency graph...")
    adj = build_adjacency(faces, num_faces)

    seasonal_indices = np.where(seasonal_mask)[0].tolist()
    print(f"Clustering {len(seasonal_indices)} seasonally shadowed faces...")
    clusters = cluster_faces(seasonal_indices, adj)
    clusters.sort(key=len, reverse=True)  # Largest first

    # Filter by minimum cluster size
    clusters = [c for c in clusters if len(c) >= args.min_cluster_size]
    print(f"  Found {len(clusters)} clusters (min size = {args.min_cluster_size})")

    # Select representative face per cluster
    # Choose the face with the most illuminated lunations (best observation windows)
    locations = []
    for ci, cluster in enumerate(clusters):
        cluster_arr = np.array(cluster)
        # Rank by number of illuminated lunations (descending)
        cluster_n_lit = n_illuminated[cluster_arr]
        best_local = np.argmax(cluster_n_lit)
        best_face = cluster_arr[best_local]

        lat = latitudes[best_face]
        lon = longitudes[best_face]
        n_lit = n_illuminated[best_face]
        n_faces_in_cluster = len(cluster)

        name = f"Cluster_{ci:03d}_{n_faces_in_cluster}faces"
        locations.append({
            'name': name,
            'lat': lat,
            'lon': lon,
            'face_idx': best_face,
            'n_illuminated': int(n_lit),
            'n_dark': int(args.num_lunations - n_lit),
            'cluster_size': n_faces_in_cluster,
        })

    # Sort by number of illuminated lunations (more = easier to observe)
    locations.sort(key=lambda x: x['n_illuminated'], reverse=True)

    # Write locations CSV
    output_path = Path(args.output_locations)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write("# Amundsen crater seasonally shadowed locations\n")
        f.write("# Generated by identify_seasonal_shadow.py\n")
        f.write(f"# {len(locations)} locations from {len(clusters)} clusters\n")
        f.write("# Columns: name, latitude (deg), longitude (deg East)\n")
        f.write("# Additional info: face_idx, n_illuminated_lunations, n_dark_lunations, cluster_size\n")
        for loc in locations:
            f.write(f"{loc['name']},{loc['lat']:.6f},{loc['lon']:.6f}\n")
    print(f"\nLocations file written: {output_path}")
    print(f"  {len(locations)} locations")

    # Summary table
    print(f"\n{'Name':<35} {'Lat':>9} {'Lon':>9} {'Lit':>5} {'Dark':>5} {'Size':>6}")
    print("-" * 75)
    for loc in locations[:30]:  # Show top 30
        print(f"{loc['name']:<35} {loc['lat']:>9.4f} {loc['lon']:>9.4f} "
              f"{loc['n_illuminated']:>5} {loc['n_dark']:>5} {loc['cluster_size']:>6}")
    if len(locations) > 30:
        print(f"  ... and {len(locations) - 30} more locations")

    print(f"\nNext step: Run Phase 2 thermal simulations:")
    print(f"  ./scripts/amundsen_thermal_series.sh")


if __name__ == '__main__':
    main()
