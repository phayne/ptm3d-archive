#!/usr/bin/env python3
"""Identify trough-bottom faces in a double-ridge PLY mesh.

Reads a binary PLY mesh, computes per-face elevation, builds a face
adjacency graph, and selects connected clusters of low-elevation faces
(trough bottoms). Outputs a text file of face indices suitable for
use with PTM3D's --heat-flow-mask flag.
"""

import argparse
import os
import struct
import sys
from collections import defaultdict, deque

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for save-only mode

import numpy as np


def load_ply_mesh(filepath):
    """Load a binary little-endian PLY mesh.

    Returns
    -------
    vertices : ndarray, shape (n_verts, 3)
        Vertex positions (x, y, z).
    normals : ndarray, shape (n_verts, 3)
        Vertex normals (nx, ny, nz).
    faces : ndarray, shape (n_faces, 3), dtype int32
        Triangle vertex indices.
    face_lat : ndarray, shape (n_faces,)
        Per-face latitude (degrees).
    face_lon : ndarray, shape (n_faces,)
        Per-face longitude (degrees).
    """
    with open(filepath, "rb") as f:
        # Parse ASCII header
        header_lines = []
        while True:
            line = f.readline().decode("ascii").strip()
            header_lines.append(line)
            if line == "end_header":
                break

        n_verts = n_faces = 0
        for line in header_lines:
            if line.startswith("element vertex"):
                n_verts = int(line.split()[-1])
            elif line.startswith("element face"):
                n_faces = int(line.split()[-1])

        print(f"Loading PLY: {n_verts} vertices, {n_faces} faces")

        # Read vertices: 6 doubles (x, y, z, nx, ny, nz) per vertex
        vert_data = np.frombuffer(f.read(n_verts * 6 * 8), dtype="<f8").reshape(n_verts, 6)
        vertices = vert_data[:, :3]
        normals = vert_data[:, 3:6]

        # Read faces: each face = 1 byte (count=3) + 3 int32 + 2 float32
        face_record_size = 1 + 3 * 4 + 2 * 4  # 21 bytes
        face_data = f.read(n_faces * face_record_size)

        faces = np.empty((n_faces, 3), dtype=np.int32)
        face_lat = np.empty(n_faces, dtype=np.float32)
        face_lon = np.empty(n_faces, dtype=np.float32)

        for i in range(n_faces):
            offset = i * face_record_size
            count = struct.unpack_from("<B", face_data, offset)[0]
            assert count == 3, f"Face {i} has {count} vertices (expected 3)"
            v0, v1, v2 = struct.unpack_from("<iii", face_data, offset + 1)
            lat, lon = struct.unpack_from("<ff", face_data, offset + 13)
            faces[i] = [v0, v1, v2]
            face_lat[i] = lat
            face_lon[i] = lon

    return vertices, normals, faces, face_lat, face_lon


def compute_face_elevations(vertices, faces):
    """Compute per-face elevation as mean vertex radial distance minus median."""
    r = np.linalg.norm(vertices, axis=1)
    r_ref = np.median(r)
    vertex_elev = r - r_ref

    # Mean elevation of each face's three vertices
    face_elev = (vertex_elev[faces[:, 0]] +
                 vertex_elev[faces[:, 1]] +
                 vertex_elev[faces[:, 2]]) / 3.0
    return face_elev, r_ref


def compute_face_normals(vertices, faces):
    """Compute outward-pointing face normals."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    edge1 = v1 - v0
    edge2 = v2 - v0
    normals = np.cross(edge1, edge2)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normals /= norms

    # Ensure outward-pointing (dot with centroid direction > 0)
    centroids = (v0 + v1 + v2) / 3.0
    centroid_dirs = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
    flip = np.sum(normals * centroid_dirs, axis=1) < 0
    normals[flip] *= -1
    return normals, centroids


def build_face_adjacency(faces):
    """Build face adjacency via shared edges.

    Returns a dict mapping face_index -> set of neighboring face indices.
    """
    edge_to_faces = defaultdict(list)
    for fi in range(len(faces)):
        v = tuple(sorted(faces[fi]))
        # Three edges per triangle
        for e in [(v[0], v[1]), (v[0], v[2]), (v[1], v[2])]:
            edge_to_faces[e].append(fi)

    adjacency = defaultdict(set)
    for edge, face_list in edge_to_faces.items():
        for i in range(len(face_list)):
            for j in range(i + 1, len(face_list)):
                adjacency[face_list[i]].add(face_list[j])
                adjacency[face_list[j]].add(face_list[i])

    return adjacency


def find_connected_components(face_indices, adjacency):
    """Find connected components among a subset of faces.

    Returns list of sets, each set containing face indices in one component.
    """
    remaining = set(face_indices)
    components = []

    while remaining:
        seed = next(iter(remaining))
        component = set()
        queue = deque([seed])
        while queue:
            fi = queue.popleft()
            if fi in component:
                continue
            component.add(fi)
            remaining.discard(fi)
            for neighbor in adjacency.get(fi, set()):
                if neighbor in remaining and neighbor not in component:
                    queue.append(neighbor)
        components.append(component)

    return components


def select_trough_faces(face_elev, face_normals, centroids, adjacency,
                        percentile=5.0, min_cluster_size=10,
                        max_normal_angle=None):
    """Select trough-bottom faces using elevation and adjacency filtering.

    Parameters
    ----------
    face_elev : ndarray
        Per-face elevation.
    face_normals : ndarray, shape (n, 3)
        Per-face outward normals.
    centroids : ndarray, shape (n, 3)
        Per-face centroids.
    adjacency : dict
        Face adjacency graph.
    percentile : float
        Select faces below this elevation percentile.
    min_cluster_size : int
        Discard connected components smaller than this.
    max_normal_angle : float or None
        If set, exclude faces whose normal deviates more than this angle
        (degrees) from the local radial direction (filters out steep walls).

    Returns
    -------
    selected : sorted list of face indices
    components : list of sets (connected components that passed filtering)
    threshold : float, the elevation threshold used
    """
    threshold = np.percentile(face_elev, percentile)
    candidates = set(np.where(face_elev <= threshold)[0])
    print(f"  Elevation threshold (percentile {percentile}): {threshold:.2f} m")
    print(f"  Initial candidates: {len(candidates)} faces")

    # Optional normal-angle filter
    if max_normal_angle is not None:
        radial = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
        cos_angle = np.sum(face_normals * radial, axis=1)
        cos_threshold = np.cos(np.radians(max_normal_angle))
        normal_ok = set(np.where(cos_angle >= cos_threshold)[0])
        before = len(candidates)
        candidates &= normal_ok
        print(f"  Normal filter (< {max_normal_angle} deg from radial): "
              f"{before} -> {len(candidates)} faces")

    # Connected component filtering
    components = find_connected_components(candidates, adjacency)
    components.sort(key=len, reverse=True)

    print(f"  Found {len(components)} connected components")
    for i, comp in enumerate(components[:5]):
        print(f"    Component {i}: {len(comp)} faces")
    if len(components) > 5:
        print(f"    ... and {len(components) - 5} more")

    # Keep only large enough components
    kept = []
    selected = set()
    for comp in components:
        if len(comp) >= min_cluster_size:
            kept.append(comp)
            selected |= comp

    print(f"  After min_cluster_size={min_cluster_size} filter: "
          f"{len(kept)} components, {len(selected)} faces")

    return sorted(selected), kept, threshold


def write_face_indices(filepath, indices, mesh_path, n_faces, method_desc, threshold):
    """Write face indices to a text file."""
    with open(filepath, "w") as f:
        f.write(f"# Trough faces for elevated geothermal heat flow\n")
        f.write(f"# Mesh: {mesh_path} ({n_faces} faces total)\n")
        f.write(f"# Method: {method_desc}\n")
        f.write(f"# Elevation threshold: {threshold:.4f} m\n")
        f.write(f"# N_selected: {len(indices)}\n")
        for idx in indices:
            f.write(f"{idx}\n")
    print(f"Wrote {len(indices)} face indices to {filepath}")


def visualize_mesh(vertices, faces, face_elev, selected_indices, components,
                   save_path=None):
    """Visualize the mesh with selected faces highlighted."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    selected_set = set(selected_indices)

    fig = plt.figure(figsize=(16, 6))

    # Panel 1: Elevation map
    ax1 = fig.add_subplot(121, projection="3d")
    centroids = (vertices[faces[:, 0]] + vertices[faces[:, 1]] +
                 vertices[faces[:, 2]]) / 3.0

    # Subsample for 3D view if too many faces
    n_faces = len(faces)
    stride = max(1, n_faces // 5000)

    tri_verts = []
    colors = []
    elev_min, elev_max = np.percentile(face_elev, [2, 98])
    cmap = plt.cm.terrain

    for i in range(0, n_faces, stride):
        tri = vertices[faces[i]]
        tri_verts.append(tri)
        if i in selected_set:
            colors.append((1, 0, 0, 0.8))  # Red for selected
        else:
            norm_elev = (face_elev[i] - elev_min) / (elev_max - elev_min + 1e-10)
            colors.append(cmap(np.clip(norm_elev, 0, 1)))

    pc = Poly3DCollection(tri_verts, facecolors=colors, edgecolors="none",
                          linewidths=0.1)
    ax1.add_collection3d(pc)

    # Set axis limits from mesh bounds
    all_pts = vertices
    margin = 0.05 * (all_pts.max(axis=0) - all_pts.min(axis=0))
    ax1.set_xlim(all_pts[:, 0].min() - margin[0], all_pts[:, 0].max() + margin[0])
    ax1.set_ylim(all_pts[:, 1].min() - margin[1], all_pts[:, 1].max() + margin[1])
    ax1.set_zlim(all_pts[:, 2].min() - margin[2], all_pts[:, 2].max() + margin[2])
    ax1.set_title(f"3D Mesh (red = selected, stride={stride})")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")

    # Panel 2: Elevation histogram
    ax2 = fig.add_subplot(122)
    ax2.hist(face_elev, bins=100, color="steelblue", alpha=0.7, label="All faces")
    if selected_indices:
        ax2.hist(face_elev[selected_indices], bins=50, color="red", alpha=0.7,
                 label="Selected (trough)")
    ax2.axvline(np.percentile(face_elev, 5), color="orange", linestyle="--",
                label="5th percentile")
    ax2.set_xlabel("Elevation (m)")
    ax2.set_ylabel("Count")
    ax2.set_title("Face Elevation Distribution")
    ax2.legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Identify trough-bottom faces in a double-ridge PLY mesh.")
    parser.add_argument("--mesh", required=True, help="Path to PLY mesh file")
    parser.add_argument("--output", required=True,
                        help="Output text file for face indices")
    parser.add_argument("--threshold-percentile", type=float, default=5.0,
                        help="Elevation percentile for trough selection (default: 5)")
    parser.add_argument("--min-cluster-size", type=int, default=10,
                        help="Minimum connected component size (default: 10)")
    parser.add_argument("--max-normal-angle", type=float, default=None,
                        help="Max angle (deg) between face normal and radial direction "
                             "(filters steep walls; default: no filter)")
    parser.add_argument("--visualize", action="store_true",
                        help="Show visualization of selected faces")
    parser.add_argument("--save-figure", default=None,
                        help="Save visualization to this path (PNG/PDF)")
    args = parser.parse_args()

    # Load mesh
    print(f"Loading mesh: {args.mesh}")
    vertices, vert_normals, faces, face_lat, face_lon = load_ply_mesh(args.mesh)

    # Compute face properties
    print("Computing face elevations...")
    face_elev, r_ref = compute_face_elevations(vertices, faces)
    print(f"  Reference radius: {r_ref:.2f} m")
    print(f"  Elevation range: {face_elev.min():.2f} to {face_elev.max():.2f} m")

    print("Computing face normals...")
    face_normals, centroids = compute_face_normals(vertices, faces)

    print("Building face adjacency graph...")
    adjacency = build_face_adjacency(faces)

    # Select trough faces
    print(f"\nSelecting trough faces (percentile={args.threshold_percentile}, "
          f"min_cluster={args.min_cluster_size})...")
    selected, components, threshold = select_trough_faces(
        face_elev, face_normals, centroids, adjacency,
        percentile=args.threshold_percentile,
        min_cluster_size=args.min_cluster_size,
        max_normal_angle=args.max_normal_angle)

    if not selected:
        print("WARNING: No faces selected!")
        sys.exit(1)

    # Report statistics on selected faces
    sel_elev = face_elev[selected]
    sel_lat = face_lat[selected]
    sel_lon = face_lon[selected]
    print(f"\nSelected face statistics:")
    print(f"  Count: {len(selected)} / {len(faces)} "
          f"({100*len(selected)/len(faces):.1f}%)")
    print(f"  Elevation: {sel_elev.min():.2f} to {sel_elev.max():.2f} m "
          f"(mean {sel_elev.mean():.2f} m)")
    print(f"  Latitude: {sel_lat.min():.4f} to {sel_lat.max():.4f} deg")
    print(f"  Longitude: {sel_lon.min():.4f} to {sel_lon.max():.4f} deg")

    # Write output
    method = (f"elevation < percentile({args.threshold_percentile}), "
              f"min_cluster={args.min_cluster_size}")
    if args.max_normal_angle is not None:
        method += f", max_normal_angle={args.max_normal_angle}"
    write_face_indices(args.output, selected, args.mesh, len(faces),
                       method, threshold)

    # Visualize
    if args.visualize or args.save_figure:
        visualize_mesh(vertices, faces, face_elev, selected, components,
                       save_path=args.save_figure)


if __name__ == "__main__":
    main()
