#!/usr/bin/env python3
"""Detect double-ridge trough faces using slope + azimuth analysis.

Identifies paired ridge flanks from local slope and aspect azimuth,
scores double-ridge candidates by prominence, and selects the trough
floor between the best-paired flanks.  Outputs a text file of face
indices suitable for PTM3D's --heat-flow-mask flag.
"""

import argparse
import os
import struct
import sys
from collections import defaultdict, deque

import matplotlib
matplotlib.use("Agg")

import numpy as np


# ──────────────────────────────────────────────────────────────────
# Reusable mesh I/O (from identify_trough_faces.py)
# ──────────────────────────────────────────────────────────────────

def load_ply_mesh(filepath):
    """Load a binary little-endian PLY mesh (PTM3D output format).

    Returns vertices (n,3), normals (n,3), faces (m,3), face_lat (m,),
    face_lon (m,).
    """
    with open(filepath, "rb") as f:
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

        vert_data = np.frombuffer(
            f.read(n_verts * 6 * 8), dtype="<f8"
        ).reshape(n_verts, 6)
        vertices = vert_data[:, :3]
        normals = vert_data[:, 3:6]

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
    """Per-face elevation relative to median radius."""
    r = np.linalg.norm(vertices, axis=1)
    r_ref = np.median(r)
    vertex_elev = r - r_ref
    face_elev = (vertex_elev[faces[:, 0]] +
                 vertex_elev[faces[:, 1]] +
                 vertex_elev[faces[:, 2]]) / 3.0
    return face_elev, r_ref


def compute_face_normals(vertices, faces):
    """Compute outward-pointing face normals and centroids."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    edge1 = v1 - v0
    edge2 = v2 - v0
    normals = np.cross(edge1, edge2)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normals /= norms

    centroids = (v0 + v1 + v2) / 3.0
    centroid_dirs = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
    flip = np.sum(normals * centroid_dirs, axis=1) < 0
    normals[flip] *= -1
    return normals, centroids


def build_face_adjacency(faces):
    """Build face adjacency via shared edges."""
    edge_to_faces = defaultdict(list)
    for fi in range(len(faces)):
        v = tuple(sorted(faces[fi]))
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
    """BFS connected components among a subset of faces."""
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


def write_face_indices(filepath, indices, mesh_path, n_faces, method_desc,
                       qb_values=None):
    """Write face indices (and optional per-face Qb values) to a text file.

    Parameters
    ----------
    qb_values : dict or None
        If provided, maps face_index -> Qb value (W/m²).  Output format
        becomes ``face_index qb_value``.
    """
    with open(filepath, "w") as f:
        f.write("# Trough faces for elevated geothermal heat flow\n")
        f.write(f"# Mesh: {mesh_path} ({n_faces} faces total)\n")
        f.write(f"# Method: {method_desc}\n")
        f.write(f"# N_selected: {len(indices)}\n")
        if qb_values is not None:
            qb_vals = [qb_values[i] for i in indices]
            f.write(f"# Qb range: {min(qb_vals):.4f} - {max(qb_vals):.4f} W/m^2\n")
            for idx in indices:
                f.write(f"{idx} {qb_values[idx]:.6f}\n")
        else:
            for idx in indices:
                f.write(f"{idx}\n")
    print(f"Wrote {len(indices)} face indices to {filepath}")


def compute_diffuse_heat_flow(trough_indices, face_elev, centroids,
                               n_core=4, diffusion_yr=100.0,
                               peak_qb=10.0, threshold_frac=0.01,
                               verbose=False):
    """Compute a diffuse heat flow pattern using thermal diffusion in ice.

    Selects the n_core lowest-elevation trough faces as the heat source,
    then applies a 2-D Gaussian decay based on the lateral thermal
    diffusion length scale for zero-porosity crystalline ice.

    Ice properties at ~100 K:
        k  = 567/T  ≈ 5.67 W/(m·K)   [Klinger 1980]
        ρ  = 917 kg/m³
        cp = 7.49·T ≈ 749 J/(kg·K)   [Giauque & Stout 1936]
        κ  = k/(ρ·cp) ≈ 8.25e-6 m²/s

    Parameters
    ----------
    trough_indices : list of int
        Face indices of the detected trough (from extract_trough_faces).
    face_elev : ndarray
        Per-face elevation.
    centroids : ndarray, shape (n_faces, 3)
        Per-face centroid positions.
    n_core : int
        Number of core faces receiving peak heat flow.
    diffusion_yr : float
        Timescale for lateral thermal diffusion (years).
    peak_qb : float
        Heat flow at core faces (W/m²).
    threshold_frac : float
        Minimum Qb fraction (relative to peak) to include a face.
    verbose : bool

    Returns
    -------
    qb_dict : dict  {face_index: qb_value}
        Per-face heat flow values for all faces above threshold.
    core_faces : list of int
        The selected core face indices.
    diffusion_length : float
        The computed diffusion length (m).
    """
    T_ref = 100.0  # K — representative Europa surface temperature
    k_ice = 567.0 / T_ref           # W/(m·K)
    rho_ice = 917.0                  # kg/m³
    cp_ice = 7.49 * T_ref           # J/(kg·K)
    kappa = k_ice / (rho_ice * cp_ice)  # thermal diffusivity, m²/s
    t_sec = diffusion_yr * 365.25 * 86400.0
    L = np.sqrt(kappa * t_sec)      # diffusion length, m

    if verbose:
        print(f"  Ice thermal properties at {T_ref:.0f} K:")
        print(f"    k = {k_ice:.2f} W/(m·K), rho = {rho_ice:.0f} kg/m³, "
              f"cp = {cp_ice:.0f} J/(kg·K)")
        print(f"    kappa = {kappa:.3e} m²/s")
        print(f"  Diffusion timescale: {diffusion_yr:.0f} yr")
        print(f"  Diffusion length L = sqrt(kappa·t) = {L:.1f} m")

    # Select the n_core lowest-elevation trough faces
    ti = np.array(trough_indices)
    elev_order = np.argsort(face_elev[ti])
    core_faces = ti[elev_order[:n_core]].tolist()

    if verbose:
        print(f"  Core faces ({n_core}): {core_faces}")
        print(f"    Elevations: "
              + ", ".join(f"{face_elev[f]:.1f} m" for f in core_faces))

    # Compute distances from every mesh face to nearest core face centroid
    core_centroids = centroids[core_faces]  # (n_core, 3)
    # Use broadcasting: (n_faces, 1, 3) - (1, n_core, 3)
    diffs = centroids[:, np.newaxis, :] - core_centroids[np.newaxis, :, :]
    dists = np.sqrt((diffs ** 2).sum(axis=2))  # (n_faces, n_core)
    min_dist = dists.min(axis=1)               # (n_faces,)

    # Gaussian decay
    qb_all = peak_qb * np.exp(-min_dist ** 2 / (2.0 * L ** 2))

    # Threshold
    qb_thresh = peak_qb * threshold_frac
    mask = qb_all >= qb_thresh
    face_indices = np.where(mask)[0]

    qb_dict = {}
    for fi in face_indices:
        qb_dict[int(fi)] = float(qb_all[fi])

    if verbose:
        vals = np.array(list(qb_dict.values()))
        print(f"  Faces above {threshold_frac*100:.0f}% threshold: {len(qb_dict)}")
        print(f"  Qb range: {vals.min():.4f} - {vals.max():.4f} W/m²")
        # Show how many at various levels
        for frac in [1.0, 0.5, 0.1, 0.01]:
            n = (vals >= peak_qb * frac).sum()
            print(f"    >= {frac*100:5.1f}% ({peak_qb*frac:.2f} W/m²): {n} faces")

    return qb_dict, core_faces, L


# ──────────────────────────────────────────────────────────────────
# Stage 1: Slope and azimuth computation
# ──────────────────────────────────────────────────────────────────

def compute_slope_azimuth(face_normals, centroids):
    """Compute per-face slope angle and aspect azimuth.

    Parameters
    ----------
    face_normals : (n, 3) outward-pointing face normals
    centroids : (n, 3) face centroids in Cartesian

    Returns
    -------
    slopes : (n,) slope angle in degrees (0 = flat)
    azimuths : (n,) aspect azimuth in degrees [0, 360), clockwise from N
    """
    # Local radial (up) direction
    r_hat = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)

    # Slope = angle between normal and radial
    cos_slope = np.sum(face_normals * r_hat, axis=1)
    cos_slope = np.clip(cos_slope, -1.0, 1.0)
    slopes = np.degrees(np.arccos(cos_slope))

    # Azimuth in local ENU frame
    # Compute lat/lon from Cartesian centroids
    cx, cy, cz = centroids[:, 0], centroids[:, 1], centroids[:, 2]
    r = np.sqrt(cx**2 + cy**2 + cz**2)
    lat = np.arcsin(cz / r)  # radians
    lon = np.arctan2(cy, cx)  # radians

    # ENU basis vectors
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    sin_lon, cos_lon = np.sin(lon), np.cos(lon)

    # e_east  = [-sin(lon), cos(lon), 0]
    # e_north = [-sin(lat)*cos(lon), -sin(lat)*sin(lon), cos(lat)]
    e_east = np.column_stack([-sin_lon, cos_lon, np.zeros_like(sin_lon)])
    e_north = np.column_stack([-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat])

    # Horizontal component of normal (projection onto tangent plane)
    n_dot_r = np.sum(face_normals * r_hat, axis=1, keepdims=True)
    n_horiz = face_normals - n_dot_r * r_hat

    # Project onto ENU
    n_east = np.sum(n_horiz * e_east, axis=1)
    n_north = np.sum(n_horiz * e_north, axis=1)

    # Azimuth: direction the face tilts toward (where the normal points
    # horizontally), measured clockwise from North
    azimuths = np.degrees(np.arctan2(n_east, n_north)) % 360.0

    return slopes, azimuths


# ──────────────────────────────────────────────────────────────────
# Stage 2: Ridge orientation detection
# ──────────────────────────────────────────────────────────────────

def circular_distance(a, b):
    """Shortest angular distance between two angles in degrees."""
    d = (a - b) % 360
    return np.minimum(d, 360 - d)


def detect_ridge_orientation(azimuths, slopes, slope_threshold=5.0,
                              verbose=False):
    """Detect dominant ridge strike from the azimuth distribution of steep faces.

    Returns
    -------
    peak_A : float — azimuth of first flank direction (degrees)
    peak_B : float — azimuth of second flank direction (degrees)
    ridge_strike : float — ridge strike direction (degrees, 0=N)
    """
    steep = slopes > slope_threshold
    n_steep = steep.sum()
    if n_steep < 50:
        raise RuntimeError(
            f"Only {n_steep} faces with slope > {slope_threshold} deg; "
            f"need at least 50 for orientation detection"
        )

    az_steep = azimuths[steep]

    # Circular histogram, 5-degree bins
    bin_edges = np.arange(0, 365, 5)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    counts, _ = np.histogram(az_steep, bins=bin_edges)

    # Smooth with Gaussian kernel (sigma = 2 bins = 10 degrees)
    # Circular padding for wrap-around
    n_bins = len(counts)
    pad = 10
    padded = np.concatenate([counts[-pad:], counts, counts[:pad]])
    from scipy.ndimage import gaussian_filter1d
    smoothed_padded = gaussian_filter1d(padded.astype(float), sigma=2)
    smoothed = smoothed_padded[pad:pad + n_bins]

    # Find peaks
    from scipy.signal import find_peaks
    peaks_idx, props = find_peaks(smoothed, distance=5,
                                   prominence=0.1 * smoothed.max())

    if len(peaks_idx) < 2:
        # Fallback: use the two highest bins
        sorted_bins = np.argsort(smoothed)[::-1]
        peaks_idx = sorted_bins[:2]

    # Sort by height
    peak_heights = smoothed[peaks_idx]
    order = np.argsort(peak_heights)[::-1]
    peaks_idx = peaks_idx[order]

    # Find the best antipodal pair among top peaks
    best_pair = None
    best_score = -1
    for i in range(len(peaks_idx)):
        for j in range(i + 1, min(len(peaks_idx), 8)):
            az_i = bin_centers[peaks_idx[i]]
            az_j = bin_centers[peaks_idx[j]]
            separation = circular_distance(az_i, az_j)
            # How close to 180 degrees?
            antipodal_err = abs(separation - 180)
            if antipodal_err < 30:
                score = (smoothed[peaks_idx[i]] + smoothed[peaks_idx[j]]) * \
                        (1 - antipodal_err / 30)
                if score > best_score:
                    best_score = score
                    best_pair = (peaks_idx[i], peaks_idx[j])

    if best_pair is None:
        # No good antipodal pair; take the two tallest peaks
        if verbose:
            print("  WARNING: No antipodal peak pair found; using two tallest")
        best_pair = (peaks_idx[0], peaks_idx[1])

    peak_A = bin_centers[best_pair[0]]
    peak_B = bin_centers[best_pair[1]]

    # Ridge strike is perpendicular to the mean flank azimuth
    # The flanks face ~peak_A and ~peak_B; the ridge runs perpendicular
    ridge_strike = (peak_A + 90) % 360

    if verbose:
        sep = circular_distance(peak_A, peak_B)
        print(f"  Steep faces: {n_steep} (slope > {slope_threshold} deg)")
        print(f"  Azimuth peaks: {peak_A:.0f} deg and {peak_B:.0f} deg "
              f"(separation {sep:.0f} deg)")
        print(f"  Ridge strike: {ridge_strike:.0f} deg from N")

    return peak_A, peak_B, ridge_strike, bin_centers, smoothed


# ──────────────────────────────────────────────────────────────────
# Stage 3: Flank classification
# ──────────────────────────────────────────────────────────────────

FLANK_LEFT = 1
FLANK_RIGHT = 2
STEEP_OTHER = 3

def classify_flanks(slopes, azimuths, peak_A, peak_B,
                    slope_threshold=5.0, sector_half_width=60.0):
    """Label each face as FLANK_LEFT, FLANK_RIGHT, STEEP_OTHER, or 0 (flat).

    Returns labels array (n_faces,), dtype int.
    """
    n = len(slopes)
    labels = np.zeros(n, dtype=int)

    steep = slopes > slope_threshold
    dist_A = circular_distance(azimuths, peak_A)
    dist_B = circular_distance(azimuths, peak_B)

    labels[steep & (dist_A <= sector_half_width)] = FLANK_LEFT
    labels[steep & (dist_B <= sector_half_width)] = FLANK_RIGHT
    # Faces matching both sectors (unlikely with 60 deg half-width): pick closer
    both = (labels == FLANK_LEFT) & (dist_B <= sector_half_width) & steep
    labels[both & (dist_B < dist_A)] = FLANK_RIGHT
    # Steep faces in neither sector
    labels[steep & (labels == 0)] = STEEP_OTHER

    return labels


# ──────────────────────────────────────────────────────────────────
# Stage 4 & 5: Grouping, pairing, scoring
# ──────────────────────────────────────────────────────────────────

def compute_ridge_coordinates(centroids, ridge_strike_deg):
    """Project centroids onto along-ridge and cross-ridge axes.

    Uses the local tangent plane approximation.  The cross-ridge axis
    is perpendicular to the ridge strike; positive = to the right when
    looking along the strike direction.

    Returns along_ridge (n,), cross_ridge (n,) in meters.
    """
    # Convert centroids to local lat/lon
    r = np.linalg.norm(centroids, axis=1)
    lat = np.arcsin(centroids[:, 2] / r)
    lon = np.arctan2(centroids[:, 1], centroids[:, 0])

    # Mean lat/lon as reference
    lat0 = np.mean(lat)
    lon0 = np.mean(lon)
    r0 = np.mean(r)

    # Local tangent plane: approximate x ~ r * cos(lat) * dlon, y ~ r * dlat
    x_local = r0 * np.cos(lat0) * (lon - lon0)  # meters, east
    y_local = r0 * (lat - lat0)                   # meters, north

    # Rotate into ridge coordinate system
    strike_rad = np.radians(ridge_strike_deg)
    # Along-ridge: direction of strike (bearing from N)
    # Cross-ridge: perpendicular, 90 deg clockwise from strike
    along = x_local * np.sin(strike_rad) + y_local * np.cos(strike_rad)
    cross = x_local * np.cos(strike_rad) - y_local * np.sin(strike_rad)

    return along, cross


def group_mean_position(component, along, cross):
    """Mean along-ridge and cross-ridge position of a component."""
    idx = list(component)
    return np.mean(along[idx]), np.mean(cross[idx])


def pair_and_score_ridges(left_groups, right_groups, face_elev,
                           centroids, along, cross, adjacency,
                           max_pair_distance=5000.0, verbose=False):
    """Pair opposing flank groups and score double-ridge candidates.

    Returns list of (score, left_group, right_group, info_dict), sorted
    descending by score.
    """
    n_faces = len(face_elev)
    candidates = []

    for li, lg in enumerate(left_groups):
        lg_along, lg_cross = group_mean_position(lg, along, cross)
        lg_elev = np.mean(face_elev[list(lg)])

        for ri, rg in enumerate(right_groups):
            rg_along, rg_cross = group_mean_position(rg, along, cross)

            # Proximity check (centroid distance in ridge coords)
            dist = np.sqrt((lg_along - rg_along)**2 +
                          (lg_cross - rg_cross)**2)
            if dist > max_pair_distance:
                continue

            # Must be on opposite sides of cross-ridge axis
            # (one has higher cross_ridge, the other lower)
            cross_separation = abs(lg_cross - rg_cross)
            if cross_separation < 100:  # too close in cross-ridge direction
                continue

            rg_elev = np.mean(face_elev[list(rg)])

            # Along-ridge overlap
            lg_idx = list(lg)
            rg_idx = list(rg)
            lg_along_range = (np.min(along[lg_idx]), np.max(along[lg_idx]))
            rg_along_range = (np.min(along[rg_idx]), np.max(along[rg_idx]))
            overlap_start = max(lg_along_range[0], rg_along_range[0])
            overlap_end = min(lg_along_range[1], rg_along_range[1])
            overlap_length = max(0, overlap_end - overlap_start)
            if overlap_length < 500:  # minimum overlap 500 m
                continue

            # Relief: mean flank elevation minus min elevation between them
            cross_min = min(lg_cross, rg_cross)
            cross_max = max(lg_cross, rg_cross)
            along_min = overlap_start
            along_max = overlap_end
            between = ((cross >= cross_min) & (cross <= cross_max) &
                       (along >= along_min) & (along <= along_max))
            between_idx = np.where(between)[0]
            # Exclude the flank faces themselves
            between_idx = np.array(
                [i for i in between_idx if i not in lg and i not in rg]
            )

            if len(between_idx) < 5:
                continue

            mean_flank_elev = 0.5 * (lg_elev + rg_elev)
            min_between_elev = np.min(face_elev[between_idx])
            relief = mean_flank_elev - min_between_elev

            if relief < 20:  # minimum 20 m relief
                continue

            size = len(lg) + len(rg)
            score = relief * size * overlap_length

            info = {
                "relief": relief,
                "size": size,
                "overlap_length": overlap_length,
                "cross_separation": cross_separation,
                "mean_flank_elev": mean_flank_elev,
                "min_between_elev": min_between_elev,
                "between_idx": between_idx,
            }
            candidates.append((score, lg, rg, info))

    candidates.sort(key=lambda x: x[0], reverse=True)

    if verbose:
        print(f"  Ridge candidates: {len(candidates)}")
        for i, (sc, lg, rg, info) in enumerate(candidates[:5]):
            print(f"    #{i}: score={sc:.0f}, L={len(lg)} faces, "
                  f"R={len(rg)} faces, relief={info['relief']:.0f} m, "
                  f"overlap={info['overlap_length']:.0f} m")

    return candidates


# ──────────────────────────────────────────────────────────────────
# Stage 6: Trough extraction
# ──────────────────────────────────────────────────────────────────

def expand_by_adjacency(face_set, adjacency, n_hops=1):
    """Expand a set of face indices by n_hops of adjacency."""
    expanded = set(face_set)
    frontier = set(face_set)
    for _ in range(n_hops):
        new_frontier = set()
        for fi in frontier:
            for nb in adjacency.get(fi, set()):
                if nb not in expanded:
                    new_frontier.add(nb)
                    expanded.add(nb)
        frontier = new_frontier
    return expanded


def extract_trough_faces(best_candidate, face_elev, slopes, along, cross,
                          adjacency, max_trough_slope=8.0, n_hops=3,
                          max_trough_half_width=1000.0, verbose=False):
    """Select trough-floor faces between the best-paired flanks.

    Uses a medial-axis width constraint to ensure the selection is a
    narrow linear feature: for each along-ridge slice, only faces within
    max_trough_half_width of the local elevation minimum are kept.

    Returns sorted list of face indices.
    """
    score, left_flank, right_flank, info = best_candidate
    between_idx = info["between_idx"]
    min_between_elev = info["min_between_elev"]
    mean_flank_elev = info["mean_flank_elev"]
    relief = info["relief"]

    # Elevation threshold: bottom 60% of the relief
    # (generous to capture full trough length; width constraint enforces linearity)
    elev_threshold = min_between_elev + 0.6 * relief

    candidates = []
    for fi in between_idx:
        if slopes[fi] <= max_trough_slope and face_elev[fi] < elev_threshold:
            candidates.append(fi)

    if verbose:
        print(f"  Elevation filter: elev < {elev_threshold:.0f} m "
              f"(bottom 60% of {relief:.0f} m relief)")
        print(f"  Initial candidates (slope < {max_trough_slope} deg): "
              f"{len(candidates)} faces")

    if not candidates:
        return []

    # ── Medial-axis width constraint ──
    # Bin candidates by along-ridge coordinate, find the cross-ridge
    # position of the minimum-elevation face in each bin (the trough axis),
    # then keep only faces within max_trough_half_width of this axis.
    cand_arr = np.array(candidates)
    cand_along = along[cand_arr]
    cand_cross = cross[cand_arr]
    cand_elev = face_elev[cand_arr]

    # Determine bin size from mesh spacing (~500 m for this mesh)
    along_range = cand_along.max() - cand_along.min()
    n_bins = max(5, int(along_range / 400))
    bin_edges = np.linspace(cand_along.min() - 1, cand_along.max() + 1, n_bins + 1)

    # Find the medial axis: cross-ridge position of lowest face in each bin
    axis_along = []
    axis_cross = []
    for b in range(n_bins):
        in_bin = (cand_along >= bin_edges[b]) & (cand_along < bin_edges[b + 1])
        if not in_bin.any():
            continue
        bin_idx = np.where(in_bin)[0]
        min_i = bin_idx[np.argmin(cand_elev[bin_idx])]
        axis_along.append(cand_along[min_i])
        axis_cross.append(cand_cross[min_i])

    if not axis_along:
        return []

    axis_along = np.array(axis_along)
    axis_cross = np.array(axis_cross)

    # For each candidate, find its distance from the interpolated axis
    # Use nearest-axis-point distance
    narrowed = []
    for i, fi in enumerate(candidates):
        # Find nearest axis point (by along-ridge distance)
        da = np.abs(cand_along[i] - axis_along)
        nearest = np.argmin(da)
        cross_dist = abs(cand_cross[i] - axis_cross[nearest])
        if cross_dist <= max_trough_half_width:
            narrowed.append(fi)

    if verbose:
        print(f"  Width constraint (half-width < {max_trough_half_width:.0f} m): "
              f"{len(candidates)} -> {len(narrowed)} faces")

    if not narrowed:
        return []

    # Connected component filtering
    components = find_connected_components(narrowed, adjacency)
    components.sort(key=len, reverse=True)

    if verbose:
        print(f"  Connected components: {len(components)}")
        for i, comp in enumerate(components[:5]):
            print(f"    Component {i}: {len(comp)} faces, "
                  f"mean elev = {np.mean(face_elev[list(comp)]):.1f} m")

    # Keep components near both flanks
    left_expanded = expand_by_adjacency(left_flank, adjacency, n_hops)
    right_expanded = expand_by_adjacency(right_flank, adjacency, n_hops)

    valid_components = []
    for comp in components:
        near_left = bool(comp & left_expanded)
        near_right = bool(comp & right_expanded)
        if near_left and near_right:
            valid_components.append(comp)
        elif len(comp) >= 20 and (near_left or near_right):
            valid_components.append(comp)

    if not valid_components and components:
        if verbose:
            print("  WARNING: No component near both flanks; using largest")
        valid_components = [components[0]]

    selected = set()
    for comp in valid_components:
        selected |= comp

    if verbose:
        if selected:
            si = np.array(list(selected))
            along_span = along[si].max() - along[si].min()
            cross_span = cross[si].max() - cross[si].min()
            print(f"  Selected: {len(selected)} faces, "
                  f"along={along_span:.0f} m, cross={cross_span:.0f} m, "
                  f"aspect={along_span / max(cross_span, 1):.1f}")

    return sorted(selected)


# ──────────────────────────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────────────────────────

def visualize_ridge_detection(face_lat, face_lon, face_elev, slopes,
                               azimuths, labels, bin_centers, smoothed_hist,
                               peak_A, peak_B, ridge_strike,
                               best_candidate, trough_indices,
                               along, cross, save_path=None):
    """6-panel diagnostic figure."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # Panel 1: Elevation map
    ax = axes[0, 0]
    vmin, vmax = np.percentile(face_elev, [2, 98])
    sc = ax.scatter(face_lon, face_lat, c=face_elev, s=1, cmap="RdBu_r",
                    vmin=vmin, vmax=vmax, edgecolors="none", rasterized=True)
    fig.colorbar(sc, ax=ax, label="Elevation (m)")
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.set_title("Elevation")
    ax.set_aspect("equal")

    # Panel 2: Azimuth histogram with detected peaks
    ax = axes[0, 1]
    ax.bar(bin_centers, smoothed_hist, width=4.5, color="steelblue", alpha=0.7)
    ax.axvline(peak_A, color="red", linewidth=2, linestyle="--",
               label=f"Peak A = {peak_A:.0f}°")
    ax.axvline(peak_B, color="blue", linewidth=2, linestyle="--",
               label=f"Peak B = {peak_B:.0f}°")
    ax.axvline(ridge_strike, color="green", linewidth=2,
               label=f"Strike = {ridge_strike:.0f}°")
    ax.set_xlabel("Azimuth (deg from N)")
    ax.set_ylabel("Count (smoothed)")
    ax.set_title("Steep Face Azimuths")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 360)

    # Panel 3: Flank classification map
    ax = axes[0, 2]
    flat = labels == 0
    steep_other = labels == STEEP_OTHER
    left = labels == FLANK_LEFT
    right = labels == FLANK_RIGHT

    ax.scatter(face_lon[flat], face_lat[flat], c="lightgray", s=0.5,
               edgecolors="none", rasterized=True, label="Flat")
    ax.scatter(face_lon[steep_other], face_lat[steep_other], c="gold", s=1,
               edgecolors="none", rasterized=True, label="Steep other")
    ax.scatter(face_lon[left], face_lat[left], c="blue", s=2,
               edgecolors="none", rasterized=True, label="Left flank")
    ax.scatter(face_lon[right], face_lat[right], c="red", s=2,
               edgecolors="none", rasterized=True, label="Right flank")
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.set_title("Flank Classification")
    ax.set_aspect("equal")
    ax.legend(fontsize=8, markerscale=5)

    # Panel 4: Best ridge pair
    ax = axes[1, 0]
    ax.scatter(face_lon, face_lat, c="lightgray", s=0.5,
               edgecolors="none", rasterized=True)
    if best_candidate is not None:
        _, lg, rg, info = best_candidate
        lg_idx = list(lg)
        rg_idx = list(rg)
        ax.scatter(face_lon[lg_idx], face_lat[lg_idx], c="blue", s=3,
                   edgecolors="none", rasterized=True, label="Left flank")
        ax.scatter(face_lon[rg_idx], face_lat[rg_idx], c="red", s=3,
                   edgecolors="none", rasterized=True, label="Right flank")
        ax.set_title(f"Best Ridge Pair (relief={info['relief']:.0f} m)")
    else:
        ax.set_title("No ridge pair found")
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.set_aspect("equal")
    ax.legend(fontsize=9, markerscale=4)

    # Panel 5: Trough selection on elevation
    ax = axes[1, 1]
    ax.scatter(face_lon, face_lat, c=face_elev, s=1, cmap="RdBu_r",
               vmin=vmin, vmax=vmax, edgecolors="none", rasterized=True)
    if trough_indices:
        ti = np.array(trough_indices)
        ax.scatter(face_lon[ti], face_lat[ti], c="lime", s=4,
                   edgecolors="none", rasterized=True, label="Trough")
        ax.legend(fontsize=10, markerscale=3)
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.set_title(f"Trough Selection ({len(trough_indices)} faces)")
    ax.set_aspect("equal")

    # Panel 6: Cross-ridge elevation transect
    ax = axes[1, 2]
    if best_candidate is not None and trough_indices:
        _, lg, rg, info = best_candidate
        # Find along-ridge center of the trough
        ti = np.array(trough_indices)
        along_center = np.mean(along[ti])
        # Select faces within a narrow band of the along-ridge center
        band_width = 0.15 * (np.max(along) - np.min(along))
        band = np.abs(along - along_center) < band_width
        band_idx = np.where(band)[0]

        ax.scatter(cross[band_idx], face_elev[band_idx],
                   c="gray", s=2, alpha=0.3, label="All faces")

        # Highlight flanks and trough in transect
        trough_set = set(trough_indices)
        left_set = lg
        right_set = rg
        for label_name, fset, color in [
            ("Left flank", left_set, "blue"),
            ("Right flank", right_set, "red"),
            ("Trough", trough_set, "lime"),
        ]:
            in_band = [i for i in fset if i in set(band_idx)]
            if in_band:
                ax.scatter(cross[in_band], face_elev[in_band],
                           c=color, s=8, label=label_name, zorder=5)

        ax.set_xlabel("Cross-ridge distance (m)")
        ax.set_ylabel("Elevation (m)")
        ax.set_title("Cross-Ridge Transect")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No ridge detected", ha="center", va="center",
                transform=ax.transAxes)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved diagnostic figure to {save_path}")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Detect double-ridge trough faces using slope+azimuth.")
    parser.add_argument("--mesh", required=True, help="Path to PLY mesh file")
    parser.add_argument("--output", required=True,
                        help="Output text file for face indices")
    parser.add_argument("--slope-threshold", type=float, default=5.0,
                        help="Min slope for flank detection (default: 5 deg)")
    parser.add_argument("--sector-half-width", type=float, default=60.0,
                        help="Azimuth sector half-width (default: 60 deg)")
    parser.add_argument("--min-flank-size", type=int, default=20,
                        help="Min faces per flank group (default: 20)")
    parser.add_argument("--max-pair-distance", type=float, default=5000.0,
                        help="Max centroid distance for pairing (default: 5000 m)")
    parser.add_argument("--max-trough-slope", type=float, default=8.0,
                        help="Max slope for trough floor (default: 8 deg)")
    parser.add_argument("--max-trough-half-width", type=float, default=800.0,
                        help="Max half-width of trough from axis (default: 800 m)")
    parser.add_argument("--n-adjacency-hops", type=int, default=3,
                        help="Adjacency hops for flank-trough link (default: 3)")
    parser.add_argument("--n-core-faces", type=int, default=4,
                        help="Number of core faces at peak heat flow (default: 4)")
    parser.add_argument("--diffusion-timescale", type=float, default=100.0,
                        help="Lateral thermal diffusion timescale in years (default: 100)")
    parser.add_argument("--peak-heat-flow", type=float, default=10.0,
                        help="Peak heat flow at core faces in W/m^2 (default: 10)")
    parser.add_argument("--qb-threshold-frac", type=float, default=0.01,
                        help="Min Qb fraction to include a face (default: 0.01)")
    parser.add_argument("--save-figure", default=None,
                        help="Save diagnostic figure (PNG/PDF)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # ── Load mesh ──
    print(f"Loading mesh: {args.mesh}")
    vertices, vert_normals, faces, face_lat, face_lon = load_ply_mesh(args.mesh)
    n_faces = len(faces)

    # ── Stage 1: Geometry ──
    print("Computing face geometry...")
    face_elev, r_ref = compute_face_elevations(vertices, faces)
    face_normals, centroids = compute_face_normals(vertices, faces)
    slopes, azimuths = compute_slope_azimuth(face_normals, centroids)

    print(f"  Elevation: {face_elev.min():.1f} to {face_elev.max():.1f} m")
    print(f"  Slope: mean={slopes.mean():.1f} deg, max={slopes.max():.1f} deg")
    print(f"  Steep faces (>{args.slope_threshold} deg): "
          f"{(slopes > args.slope_threshold).sum()}")

    # ── Stage 2: Ridge orientation ──
    print("\nDetecting ridge orientation...")
    peak_A, peak_B, ridge_strike, bin_centers, smoothed_hist = \
        detect_ridge_orientation(azimuths, slopes,
                                 slope_threshold=args.slope_threshold,
                                 verbose=True)

    # ── Stage 3: Flank classification ──
    print("\nClassifying flanks...")
    labels = classify_flanks(slopes, azimuths, peak_A, peak_B,
                             slope_threshold=args.slope_threshold,
                             sector_half_width=args.sector_half_width)
    n_left = (labels == FLANK_LEFT).sum()
    n_right = (labels == FLANK_RIGHT).sum()
    n_other = (labels == STEEP_OTHER).sum()
    print(f"  Left flank: {n_left}, Right flank: {n_right}, "
          f"Steep other: {n_other}")

    # ── Stage 4: Spatial grouping ──
    print("\nBuilding adjacency and grouping flanks...")
    adjacency = build_face_adjacency(faces)

    left_faces = set(np.where(labels == FLANK_LEFT)[0])
    right_faces = set(np.where(labels == FLANK_RIGHT)[0])

    left_components = find_connected_components(left_faces, adjacency)
    right_components = find_connected_components(right_faces, adjacency)

    left_groups = [c for c in left_components if len(c) >= args.min_flank_size]
    right_groups = [c for c in right_components if len(c) >= args.min_flank_size]
    left_groups.sort(key=len, reverse=True)
    right_groups.sort(key=len, reverse=True)

    print(f"  Left groups (>= {args.min_flank_size} faces): {len(left_groups)} "
          f"(sizes: {[len(g) for g in left_groups[:5]]})")
    print(f"  Right groups (>= {args.min_flank_size} faces): {len(right_groups)} "
          f"(sizes: {[len(g) for g in right_groups[:5]]})")

    # ── Stage 5: Pairing and scoring ──
    print("\nPairing and scoring ridge candidates...")
    along, cross = compute_ridge_coordinates(centroids, ridge_strike)

    candidates = pair_and_score_ridges(
        left_groups, right_groups, face_elev, centroids, along, cross,
        adjacency, max_pair_distance=args.max_pair_distance, verbose=True)

    if not candidates:
        print("ERROR: No valid ridge candidates found!")
        sys.exit(1)

    best = candidates[0]

    # ── Stage 6: Trough extraction ──
    print("\nExtracting trough faces...")
    trough_indices = extract_trough_faces(
        best, face_elev, slopes, along, cross, adjacency,
        max_trough_slope=args.max_trough_slope,
        n_hops=args.n_adjacency_hops,
        max_trough_half_width=args.max_trough_half_width, verbose=True)

    if not trough_indices:
        print("ERROR: No trough faces selected!")
        sys.exit(1)

    # ── Summary ──
    ti = np.array(trough_indices)
    _, lg, rg, info = best
    print(f"\n{'='*60}")
    print(f"RESULT: Selected {len(trough_indices)} trough faces "
          f"({100*len(trough_indices)/n_faces:.1f}%)")
    print(f"  Ridge orientation: strike {ridge_strike:.0f} deg "
          f"(flanks at {peak_A:.0f}/{peak_B:.0f} deg)")
    print(f"  Left flank: {len(lg)} faces, Right flank: {len(rg)} faces")
    print(f"  Relief: {info['relief']:.0f} m")
    print(f"  Trough elevation: {face_elev[ti].min():.1f} to "
          f"{face_elev[ti].max():.1f} m "
          f"(mean {face_elev[ti].mean():.1f} m)")
    print(f"  Trough lat: {face_lat[ti].min():.4f} to "
          f"{face_lat[ti].max():.4f} deg")
    print(f"  Trough lon: {face_lon[ti].min():.4f} to "
          f"{face_lon[ti].max():.4f} deg")
    print(f"{'='*60}")

    # ── Diffuse heat flow pattern ──
    print("\nComputing diffuse heat flow pattern...")
    qb_dict, core_faces, L_diff = compute_diffuse_heat_flow(
        trough_indices, face_elev, centroids,
        n_core=args.n_core_faces,
        diffusion_yr=args.diffusion_timescale,
        peak_qb=args.peak_heat_flow,
        threshold_frac=args.qb_threshold_frac,
        verbose=True)

    # ── Write output ──
    method = (f"slope+azimuth ridge detection: strike={ridge_strike:.0f} deg, "
              f"slope_thresh={args.slope_threshold}, "
              f"relief={info['relief']:.0f} m, "
              f"flanks L={len(lg)}/R={len(rg)}, "
              f"diffuse: {len(core_faces)} core faces, "
              f"L={L_diff:.0f} m, t={args.diffusion_timescale:.0f} yr")
    diffuse_indices = sorted(qb_dict.keys())
    write_face_indices(args.output, diffuse_indices, args.mesh, n_faces,
                       method, qb_values=qb_dict)

    # ── Visualization ──
    if args.save_figure:
        print("\nGenerating diagnostic figure...")
        visualize_ridge_detection(
            face_lat, face_lon, face_elev, slopes, azimuths, labels,
            bin_centers, smoothed_hist, peak_A, peak_B, ridge_strike,
            best, trough_indices, along, cross,
            save_path=args.save_figure)


if __name__ == "__main__":
    main()
