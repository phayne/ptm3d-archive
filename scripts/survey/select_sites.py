#!/usr/bin/env python3
"""
Select observation target sites based on illumination, duration, and temperature criteria.
"""
import sys
from collections import deque, defaultdict
from pathlib import Path

import numpy as np

from survey_config import SurveyConfig, load_config
from utils import read_ply_mesh, read_stats_binary, write_stats_binary, save_state, load_state


def build_adjacency(faces, nf):
    """Build face adjacency graph via shared edges."""
    edge_to_faces = defaultdict(list)
    for fi in range(nf):
        v = tuple(sorted(faces[fi]))
        for e in [(v[0], v[1]), (v[0], v[2]), (v[1], v[2])]:
            edge_to_faces[e].append(fi)
    adj = [[] for _ in range(nf)]
    for fl in edge_to_faces.values():
        for i in range(len(fl)):
            for j in range(i + 1, len(fl)):
                adj[fl[i]].append(fl[j])
                adj[fl[j]].append(fl[i])
    return adj


def cluster_faces(face_indices, adj):
    """BFS clustering of face indices."""
    face_set = set(face_indices)
    visited = set()
    clusters = []
    for fi in face_indices:
        if fi in visited:
            continue
        cluster = []
        q = deque([fi]); visited.add(fi)
        while q:
            node = q.popleft(); cluster.append(node)
            for nb in adj[node]:
                if nb in face_set and nb not in visited:
                    visited.add(nb); q.append(nb)
        clusters.append(cluster)
    return clusters


def select_sites(cfg: SurveyConfig) -> bool:
    """Select target sites based on criteria."""
    print("=== Site Selection ===")

    # Load mesh
    print("  Loading mesh...")
    verts, faces, fc, lats, lons = read_ply_mesh(str(cfg.mesh_path))
    nf = len(faces)

    # Load duration maps
    frac_file = cfg.survey_dir / "illum_total_fraction.bin"
    hours_file = cfg.survey_dir / "illum_max_continuous_hours.bin"
    if not frac_file.exists() or not hours_file.exists():
        print("ERROR: Duration maps not found. Run illumination first.")
        return False

    illum_frac = read_stats_binary(str(frac_file))
    max_cont_hrs = read_stats_binary(str(hours_file))

    # Load summer temperature max (if thermal was run)
    summer_tmax = None
    for lun in cfg.thermal.lunations:
        tmax_file = cfg.thermal_dir / f"lunation_{lun:02d}" / "statistics" / "temperature_max.bin"
        if tmax_file.exists():
            summer_tmax = read_stats_binary(str(tmax_file))
            print(f"  Loaded temperature_max from lunation {lun:02d}")
            break

    # Apply criteria
    sel = cfg.selection
    mask = (
        (illum_frac > sel.min_illum_fraction) &
        (illum_frac < sel.max_illum_fraction) &
        (max_cont_hrs >= sel.min_continuous_hours)
    )
    if summer_tmax is not None:
        mask &= (summer_tmax < sel.max_temperature_K)

    n_candidates = np.sum(mask)
    print(f"  Criteria: illum {sel.min_illum_fraction*100:.2f}–{sel.max_illum_fraction*100:.0f}%, "
          f"cont >= {sel.min_continuous_hours:.0f} hr"
          + (f", T_max < {sel.max_temperature_K:.0f} K" if summer_tmax is not None else ""))
    print(f"  Candidates: {n_candidates:,} faces")

    # Cluster candidates
    print("  Building adjacency...")
    adj = build_adjacency(faces, nf)
    candidate_indices = np.where(mask)[0].tolist()
    clusters = cluster_faces(candidate_indices, adj)
    clusters = [c for c in clusters if len(c) >= sel.min_cluster_size]
    clusters.sort(key=len, reverse=True)
    print(f"  Clusters (>= {sel.min_cluster_size} faces): {len(clusters)}")

    # Select representative face per cluster
    locations = []
    for ci, cluster in enumerate(clusters[:100]):
        arr = np.array(cluster)
        best = arr[np.argmax(max_cont_hrs[arr])]
        locations.append({
            'name': f"Site_{ci:03d}_{len(cluster)}f",
            'lat': float(lats[best]),
            'lon': float(lons[best]),
            'face_idx': int(best),
            'cluster_size': len(cluster),
            'illum_frac': float(illum_frac[best]),
            'max_cont_hrs': float(max_cont_hrs[best]),
            'summer_tmax': float(summer_tmax[best]) if summer_tmax is not None else 0.0,
        })

    # Add named sites
    for ns in cfg.named_sites:
        dlat = lats - ns.lat
        dlon = lons - ns.lon
        fidx = int(np.argmin(dlat**2 + dlon**2))
        locations.append({
            'name': ns.name,
            'lat': ns.lat,
            'lon': ns.lon,
            'face_idx': fidx,
            'cluster_size': 1,
            'illum_frac': float(illum_frac[fidx]),
            'max_cont_hrs': float(max_cont_hrs[fidx]),
            'summer_tmax': float(summer_tmax[fidx]) if summer_tmax is not None else 0.0,
        })

    # Write locations CSV
    cfg.locations_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.locations_csv, 'w') as f:
        f.write(f"# {cfg.name} target locations\n")
        f.write(f"# Criteria: illum {sel.min_illum_fraction*100:.2f}-{sel.max_illum_fraction*100:.0f}%, "
                f"cont >= {sel.min_continuous_hours:.0f} hr, T_max < {sel.max_temperature_K:.0f} K\n")
        for loc in locations:
            f.write(f"{loc['name']},{loc['lat']:.6f},{loc['lon']:.6f}\n")

    print(f"  Selected {len(locations)} sites -> {cfg.locations_csv}")

    state = load_state(cfg.state_file)
    state['selection'] = {
        'status': 'complete',
        'n_candidates': int(n_candidates),
        'n_clusters': len(clusters),
        'n_locations': len(locations),
    }
    save_state(cfg.state_file, state)
    return True


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python select_sites.py <config.yaml>")
        sys.exit(1)
    cfg = load_config(sys.argv[1])
    select_sites(cfg)
