#!/usr/bin/env python3
"""
Schematic figure illustrating direct illumination and surface-to-surface
radiation exchange in PTM3D.

Renders an oblique 3D view of a ~500 m × 500 m region of a lunar
polar crater interior + rim, with triangle edges shown and
faces colored by peak direct solar flux.  Two example triangles are
highlighted: one in deep shadow (PSR floor) and one on a sunlit wall.
Three labeled arrows show:

  1. Direct solar flux from above the mesh impinging on the sunlit
     face (with explicit incidence-angle treatment).
  2. The radiative power transferred from face j to face i,
     F_{ij} B_j A_i.
  3. The reciprocal exchange from face i to face j,
     F_{ji} B_i A_j.

Output:
  docs/figures/v2/radiation_exchange_schematic.{png,pdf}
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import struct

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from ptm_pubstyle import apply_style, DOUBLE_W

PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_OUT = PROJECT_ROOT / "docs" / "figures" / "v2" / "radiation_exchange_schematic.png"

#: The run this figure is drawn from. It used to be two absolute paths naming
#: one campaign's site; the site is not published with PTM3D, so the run is an
#: ARGUMENT now. `--run-dir` supplies both halves (`mesh.ply` and
#: `statistics/`), or `--mesh`/`--stats` supply them separately. There is no
#: default: a placeholder path would fail at read time with "no such file"
#: instead of saying what it wanted.
MESH_PATH = None
STATS_DIR = None


def load_ply(path: Path):
    """Site mesh.ply: 6 doubles/vertex (x,y,z,nx,ny,nz) + face record
    of (uchar count, 3 int32, 2 float32 lat/lon)."""
    with open(path, "rb") as f:
        n_verts = n_faces = None
        while True:
            line = f.readline().decode("ascii", errors="ignore").strip()
            if line.startswith("element vertex"):
                n_verts = int(line.split()[-1])
            elif line.startswith("element face"):
                n_faces = int(line.split()[-1])
            elif line == "end_header":
                break
        assert n_verts and n_faces
        vert_arr = np.fromfile(f, dtype=np.float64, count=n_verts * 6).reshape(n_verts, 6)
        verts = vert_arr[:, :3]
        face_dt = np.dtype([
            ("count", np.uint8),
            ("idx",   np.int32, 3),
            ("lat",   np.float32),
            ("lon",   np.float32),
        ])
        faces = np.fromfile(f, dtype=face_dt, count=n_faces)
    return verts, faces["idx"]


def load_stats_bin(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    n = struct.unpack("<i", raw[:4])[0]
    return np.frombuffer(raw[4:], dtype=np.float32)


def main():
    global MESH_PATH, STATS_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path,
                        help="A finished run directory: its mesh.ply and "
                             "statistics/ are used.")
    parser.add_argument("--mesh", type=Path,
                        help="The mesh PLY, if not --run-dir/mesh.ply.")
    parser.add_argument("--stats", type=Path,
                        help="The statistics directory, if not "
                             "--run-dir/statistics.")
    args, _rest = parser.parse_known_args()
    MESH_PATH = args.mesh or (args.run_dir / "mesh.ply" if args.run_dir else None)
    STATS_DIR = args.stats or (args.run_dir / "statistics" if args.run_dir
                               else None)
    if MESH_PATH is None or STATS_DIR is None:
        sys.exit("This figure is drawn from a finished run, and there is no "
                 "default. Pass --run-dir <dir>, or --mesh and --stats.")

    apply_style()

    # ---- Load mesh + per-face direct solar flux max ----
    print("Loading mesh…")
    verts, tri_idx = load_ply(MESH_PATH)
    n_faces = len(tri_idx)
    print(f"  {len(verts)} verts, {n_faces} faces")

    direct_max = load_stats_bin(STATS_DIR / "direct_solar_max.bin")
    assert len(direct_max) == n_faces, (len(direct_max), n_faces)

    # Face centroids, areas, and outward normals
    v0 = verts[tri_idx[:, 0]]
    v1 = verts[tri_idx[:, 1]]
    v2 = verts[tri_idx[:, 2]]
    centroids = (v0 + v1 + v2) / 3.0
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    normals = cross / (2.0 * areas[:, None] + 1e-30)
    # On a sphere-fitted mesh the cross product may point either way; align
    # to the outward radial direction (away from the body center).
    radial = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-30)
    flip = (np.einsum("ij,ij->i", normals, radial) < 0)
    normals[flip] *= -1.0

    # ---- Pick a 500 m × 500 m sub-region with PSR floor + sunlit wall ----
    # Strategy: find a window where the direct_max range is large, indicating
    # the kind of shadow boundary that motivates the radiation-exchange
    # discussion.  Search by gridding the centroid xy plane.
    print("Searching for a crater bowl with mutually visible floor/wall…")
    cx, cy, cz = centroids[:, 0], centroids[:, 1], centroids[:, 2]
    half_size = 200.0  # meters — bigger window so a crater fits

    # ---- Direct floor-face search: find faces that are local low spots
    # AND in shadow (low direct_solar_max), then for each one search the
    # 200-m horizontal neighborhood for a wall face above it with strong
    # mutual visibility.
    #
    # Note: the mesh sits near the lunar south pole, so "outward radial"
    # in the absolute Cartesian frame is the unit vector centroid/|centroid|
    # — NOT global +z. We use the radial-alignment cosine as the
    # "near-horizontal" test, where 1.0 = perfectly horizontal terrain.
    radial = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-30)
    n_dot_r = np.einsum("ij,ij->i", normals, radial)
    is_horizontal = n_dot_r > 0.85   # face's outward normal mostly radial
    # Define a "local up" elevation: project centroid onto the average
    # radial direction so we can compare faces by their height along the
    # local vertical, not global z.
    avg_radial = radial.mean(axis=0)
    avg_radial /= np.linalg.norm(avg_radial)
    elev_local = centroids @ avg_radial   # signed elevation along local up
    is_lo = elev_local < np.percentile(elev_local, 15)
    is_dark = direct_max < np.percentile(direct_max, 15)
    is_floor_cand = is_horizontal & is_dark & is_lo & (areas > 1.0)
    floor_pool = np.where(is_floor_cand)[0]
    print(f"  {len(floor_pool)} floor candidates "
          f"(normal·radial > 0.85, low elev, low direct_max)")
    # Pre-compute a KDTree on (cx, cy) for fast neighborhood queries
    from scipy.spatial import cKDTree
    tree = cKDTree(np.column_stack([cx, cy]))

    best = None  # (score, i, j, horiz, dz, cos_i, cos_j)
    # To limit cost, sample at most ~2000 floor candidates by local elevation
    if len(floor_pool) > 2000:
        floor_pool = floor_pool[np.argsort(elev_local[floor_pool])[:2000]]
    for i in floor_pool:
        ci_xy = (cx[i], cy[i])
        neigh_idx = tree.query_ball_point(ci_xy, r=200.0)
        if not neigh_idx:
            continue
        neigh = np.asarray(neigh_idx, dtype=np.int64)
        # Within neighborhood, pick wall candidates: elevated above floor
        # (local-vertical) and sunlit, with normal somewhat tilted off
        # the radial.
        d_elev = elev_local[neigh] - elev_local[i]
        d_n    = direct_max[neigh]
        ndr_n  = n_dot_r[neigh]
        wall_mask_n = ((d_elev > 30.0) & (d_n > 40.0)
                       & (ndr_n > 0.4) & (ndr_n < 0.97))
        for j in neigh[wall_mask_n]:
            r_ij = centroids[j] - centroids[i]
            horiz = np.hypot(r_ij[0], r_ij[1])
            if horiz < 25:
                continue
            r_unit = r_ij / (np.linalg.norm(r_ij) + 1e-30)
            cos_i_v = np.dot(normals[i], r_unit)
            cos_j_v = np.dot(normals[j], -r_unit)
            if cos_i_v < 0.20 or cos_j_v < 0.15:
                continue
            d_elev_pair = elev_local[j] - elev_local[i]
            score_pair = cos_i_v * cos_j_v * (d_elev_pair / horiz)
            if best is None or score_pair > best[0]:
                best = (score_pair, int(i), int(j), horiz,
                        d_elev_pair, cos_i_v, cos_j_v)
    assert best is not None, "no mutually visible floor/wall pair found"
    _, face_i, face_j, horiz, dz, cos_i, cos_j = best
    # Center the window on the midpoint of the two faces
    xc = 0.5 * (cx[face_i] + cx[face_j])
    yc = 0.5 * (cy[face_i] + cy[face_j])
    # Compute window membership for rendering
    mask = (np.abs(cx - xc) < half_size) & (np.abs(cy - yc) < half_size)
    idx = np.where(mask)[0]
    print(f"  example pair: face_i (floor) {face_i}, face_j (wall) {face_j}")
    print(f"    horiz {horiz:.1f} m, vert {dz:.1f} m, "
          f"cos_i {cos_i:.3f}, cos_j {cos_j:.3f}")
    print(f"  window centered at ({xc:.0f}, {yc:.0f}) m, {len(idx)} faces")

    win_size = half_size

    # ---- Convert to a LOCAL East/North/Up frame centered on the window.
    # The mesh sits near the lunar south pole, where global +z is roughly
    # "down" (toward the body center), so rendering with the raw
    # lunar-centered Cartesian frame makes terrain appear inverted and
    # squashed in the matplotlib 3D view. Build an ENU rotation matrix
    # such that:
    #   - the local +Z axis is the outward radial at the window center
    #   - the local +X axis is the in-tangent direction obtained by
    #     projecting global +x onto the tangent plane
    #   - the local +Y axis completes a right-handed frame
    # Apply the rotation to the verts and centroids; render in this
    # rotated frame so the ground is at the bottom and the sky is up.
    centre_window = np.array([xc, yc, centroids[idx, 2].mean()])
    up = centre_window / np.linalg.norm(centre_window)   # outward radial
    east_guess = np.array([1.0, 0.0, 0.0])
    east = east_guess - np.dot(east_guess, up) * up
    if np.linalg.norm(east) < 1e-6:
        east = np.array([0.0, 1.0, 0.0]) - np.dot([0.0, 1.0, 0.0], up) * up
    east /= np.linalg.norm(east)
    north = np.cross(up, east)
    north /= np.linalg.norm(north)
    R = np.column_stack([east, north, up])   # columns are basis vectors
    # Rotate vertices and centroids into the local ENU frame, centered
    # on the window centroid so the local origin sits at the window
    # mid-elevation.
    def to_local(P):
        return (P - centre_window) @ R   # columns of R are the basis vectors → multiply on right
    verts     = to_local(verts)
    centroids = to_local(centroids)
    # Re-anchor z so the lowest in-window vertex sits at z=0
    z_floor_in_win = verts[tri_idx[idx]][..., 2].min()
    verts[:, 2]     -= z_floor_in_win
    centroids[:, 2] -= z_floor_in_win

    # ---- Render ----
    print("Rendering…")

    fig = plt.figure(figsize=(DOUBLE_W, 0.9 * DOUBLE_W))
    ax = fig.add_subplot(111, projection="3d")

    # Build per-face triangle vertices array for Poly3DCollection.
    tris = np.stack([verts[tri_idx[idx, 0]],
                     verts[tri_idx[idx, 1]],
                     verts[tri_idx[idx, 2]]], axis=1)

    # Normalize direct_max for coloring.  Cap at a percentile to avoid a
    # handful of glinty sub-pixel-area triangles dominating the scale.
    d_face = direct_max[idx]
    vmax = max(np.percentile(d_face, 99), 10.0)
    norm = mcolors.Normalize(vmin=0.0, vmax=vmax)
    cmap = plt.get_cmap("inferno")
    face_colors = cmap(norm(d_face))

    poly = Poly3DCollection(
        tris,
        facecolors=face_colors,
        edgecolors=(0.15, 0.15, 0.2, 0.5),
        linewidths=0.12,
    )
    ax.add_collection3d(poly)

    # ---- Highlight the two example triangles. The native faces are
    # small (~m²) at the 400-m window zoom, so we mark each face with a
    # bright screen-space scatter dot floated a few meters above the
    # terrain so matplotlib's 3D depth-sort cannot bury it.
    for face_idx_global, color in [
        (face_i, "#00ddff"),   # cyan against dark PSR floor
        (face_j, "#ff33aa"),   # magenta against bright sunlit terrain
    ]:
        tri = np.stack([verts[tri_idx[face_idx_global, 0]],
                        verts[tri_idx[face_idx_global, 1]],
                        verts[tri_idx[face_idx_global, 2]]])
        # Solid native face for "ground truth" position
        hl_native = Poly3DCollection(
            [tri],
            facecolors=[(*mcolors.to_rgb(color), 1.0)],
            edgecolors=color,
            linewidths=1.5,
        )
        ax.add_collection3d(hl_native)
        # Bright marker above the face
        c = tri.mean(axis=0) + np.array([0, 0, 4.0])
        ax.scatter(c[0], c[1], c[2], s=120, color=color,
                   edgecolors="black", linewidths=1.0, zorder=30)

    # ---- Arrows and labels intentionally omitted; added externally
    # in the paper layout. Only the highlighted face markers and the
    # terrain are rendered here. ----

    # ---- Camera & axes ----
    ax.view_init(elev=22, azim=-55)

    # Zoom in around the window (now in local coords centered on 0,0).
    # Pad in z to keep some sky above and enforce equal x/y/z spatial
    # aspect so topography doesn't look squished.
    z_floor_mesh = tris[..., 2].min()
    z_ceil_mesh  = tris[..., 2].max()
    z_span = z_ceil_mesh - z_floor_mesh
    # Half-span of horizontal view (matches win_size)
    half = win_size
    # Fixed z upper bound for layout consistency with the paper figure
    z_lo = z_floor_mesh - 0.05 * z_span
    z_hi = 200.0
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_zlim(z_lo, z_hi)
    # 3D box with horizontal-equal aspect and a moderate vertical
    # exaggeration of 1.6× so the slope geometry reads visually without
    # distorting it to caricature. This is a schematic figure so a
    # small exaggeration is acceptable; we note it in the caption.
    VEX = 1.6  # vertical exaggeration
    box_x = 2 * half
    box_y = 2 * half
    box_z = (z_hi - z_lo) * VEX
    ax.set_box_aspect((box_x, box_y, box_z))

    # Sky background: pale gradient that suggests an airless sky
    ax.xaxis.set_pane_color((0.94, 0.96, 0.99, 1.0))
    ax.yaxis.set_pane_color((0.94, 0.96, 0.99, 1.0))
    ax.zaxis.set_pane_color((0.99, 0.99, 0.995, 1.0))
    ax.grid(False)

    # Subdued axis labels (LOCAL Cartesian meters around the window
    # centroid; the absolute coordinates are 1.737e6 m in lunar-centered
    # frame).
    ax.set_xlabel("x (m, local)", fontsize=9)
    ax.set_ylabel("y (m, local)", fontsize=9)
    ax.set_zlabel("z (m, local)", fontsize=9)
    ax.tick_params(axis="both", which="major", labelsize=7)

    # Colorbar for direct_solar_max
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.55, pad=0.04, aspect=18)
    cbar.set_label(r"$\max\,Q_{\odot}(t)$  (W m$^{-2}$)", fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    # ---- Save ----
    out = DEFAULT_OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Wrote {out}")
    pdf = out.with_suffix(".pdf")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
