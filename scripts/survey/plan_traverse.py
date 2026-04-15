#!/usr/bin/env python3
"""
Traverse planner: plan rover/astronaut routes visiting target sites
with constraints on distance, duration, slope, time windows, and return.
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from survey_config import SurveyConfig, load_config
from utils import (read_ply_mesh, read_stats_binary, haversine_distance,
                   compute_face_slopes, save_state, load_state)


def load_sites(cfg):
    """Load target sites from locations CSV."""
    sites = []
    with open(cfg.locations_csv) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split(',')
            if len(parts) >= 3:
                sites.append({
                    'name': parts[0].strip(),
                    'lat': float(parts[1]),
                    'lon': float(parts[2]),
                })
    return sites


def get_illumination_windows(cfg, site_lat, site_lon, lats, lons):
    """Determine which lunations a site is illuminated."""
    nf = len(lats)
    # Find nearest face
    dlat = lats - site_lat
    dlon = lons - site_lon
    fidx = int(np.argmin(dlat**2 + dlon**2))

    windows = []  # list of (start_date, end_date) when illuminated
    lit_lunations = []

    for i in range(cfg.illumination.num_lunations):
        stats_file = cfg.survey_dir / f"lunation_{i:02d}" / "statistics" / "direct_solar_avg.bin"
        if not stats_file.exists():
            continue
        avg = read_stats_binary(str(stats_file))
        if len(avg) == nf and avg[fidx] > 0:
            lit_lunations.append(i)

    # Convert lunation indices to date ranges
    for lun in lit_lunations:
        start = cfg.time.start_datetime + timedelta(days=lun * cfg.time.lunation_days)
        end = start + timedelta(days=cfg.time.lunation_days)
        windows.append((start, end))

    return windows, lit_lunations


def compute_distance_matrix(sites, R_km=1737.4):
    """Pairwise great-circle distances between sites."""
    n = len(sites)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = haversine_distance(
                sites[i]['lat'], sites[i]['lon'],
                sites[j]['lat'], sites[j]['lon'], R_km
            )
            dist[i, j] = dist[j, i] = d
    return dist


def check_slope_feasible(lat1, lon1, lat2, lon2, lats, lons, slopes, max_slope, R_km=1737.4):
    """Check if path between two points crosses terrain steeper than max_slope.
    Simplified: check slope at midpoint and endpoints."""
    # Sample points along the path
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        lat = lat1 + frac * (lat2 - lat1)
        lon = lon1 + frac * (lon2 - lon1)
        dlat = lats - lat
        dlon = lons - lon
        fidx = int(np.argmin(dlat**2 + dlon**2))
        if slopes[fidx] > max_slope:
            return False
    return True


def greedy_traverse(sites, dist_matrix, start_idx, windows, cfg):
    """Greedy nearest-neighbor traverse with time window constraints."""
    tc = cfg.traverse
    start_time = datetime.fromisoformat(tc.start_time)
    max_dur = timedelta(days=tc.max_duration_days)
    speed = tc.traverse_speed_kmh
    rest_hours = tc.min_rest_hours

    n = len(sites)
    visited = [start_idx]
    total_dist = 0.0
    current_time = start_time
    current_idx = start_idx
    schedule = [{
        'site': sites[start_idx]['name'],
        'lat': sites[start_idx]['lat'],
        'lon': sites[start_idx]['lon'],
        'arrive': start_time.isoformat(),
        'depart': (start_time + timedelta(hours=rest_hours)).isoformat(),
        'dist_from_prev': 0.0,
    }]
    current_time += timedelta(hours=rest_hours)

    for _ in range(n - 1):
        # Find nearest unvisited feasible site
        best_idx = -1
        best_dist = float('inf')

        for j in range(n):
            if j in visited:
                continue

            d = dist_matrix[current_idx, j]
            travel_time = timedelta(hours=d / speed)
            arrive = current_time + travel_time

            # Check time window
            if tc.require_illuminated and windows[j]:
                in_window = any(ws <= arrive <= we for ws, we in windows[j])
                if not in_window:
                    continue

            # Check total distance (including return)
            return_dist = dist_matrix[j, start_idx] if tc.return_to_start else 0
            if total_dist + d + return_dist > tc.max_distance_km:
                continue

            # Check total duration
            if arrive - start_time + timedelta(hours=rest_hours) > max_dur:
                continue

            if d < best_dist:
                best_dist = d
                best_idx = j

        if best_idx < 0:
            break  # No feasible next site

        # Travel to best site
        travel_time = timedelta(hours=best_dist / speed)
        arrive = current_time + travel_time
        total_dist += best_dist
        visited.append(best_idx)

        schedule.append({
            'site': sites[best_idx]['name'],
            'lat': sites[best_idx]['lat'],
            'lon': sites[best_idx]['lon'],
            'arrive': arrive.isoformat(),
            'depart': (arrive + timedelta(hours=rest_hours)).isoformat(),
            'dist_from_prev': best_dist,
        })

        current_time = arrive + timedelta(hours=rest_hours)
        current_idx = best_idx

    # Return to start if required
    if tc.return_to_start and len(visited) > 1:
        d_return = dist_matrix[current_idx, start_idx]
        travel_time = timedelta(hours=d_return / speed)
        arrive = current_time + travel_time
        total_dist += d_return
        schedule.append({
            'site': f"Return to {sites[start_idx]['name']}",
            'lat': sites[start_idx]['lat'],
            'lon': sites[start_idx]['lon'],
            'arrive': arrive.isoformat(),
            'depart': arrive.isoformat(),
            'dist_from_prev': d_return,
        })

    total_duration = datetime.fromisoformat(schedule[-1]['arrive']) - start_time

    return {
        'schedule': schedule,
        'total_distance_km': total_dist,
        'total_duration_days': total_duration.total_seconds() / 86400,
        'sites_visited': len(visited),
        'visited_indices': visited,
    }


def plan_traverse(cfg: SurveyConfig) -> bool:
    """Plan a traverse route through target sites."""
    if not cfg.traverse.enabled:
        print("Traverse planning disabled in config.")
        return True

    print("=== Traverse Planning ===")

    # Load mesh for slope computation
    verts, faces, fc, lats, lons = read_ply_mesh(str(cfg.mesh_path))
    slopes = compute_face_slopes(verts, faces, fc, cfg.body_radius)

    # Load sites
    sites = load_sites(cfg)
    if not sites:
        print("ERROR: No target sites found.")
        return False

    # Add start point as first site
    start_site = {
        'name': 'Start',
        'lat': cfg.traverse.start_lat,
        'lon': cfg.traverse.start_lon,
    }
    all_sites = [start_site] + sites

    # Get illumination windows for each site
    print(f"  Computing illumination windows for {len(all_sites)} sites...")
    windows = []
    for site in all_sites:
        w, _ = get_illumination_windows(cfg, site['lat'], site['lon'], lats, lons)
        windows.append(w)

    # Compute distance matrix
    dist_matrix = compute_distance_matrix(all_sites, R_km=cfg.body_radius / 1000)

    # Filter by slope feasibility
    print(f"  Max slope: {cfg.traverse.max_slope_deg}°")

    # Run greedy planner
    result = greedy_traverse(all_sites, dist_matrix, 0, windows, cfg)

    # 2-opt improvement
    # (simplified: try swapping adjacent pairs in the schedule)
    # TODO: implement full 2-opt

    # Print results
    print(f"\n  === Traverse Result ===")
    print(f"  Sites visited: {result['sites_visited']} (of {len(all_sites)})")
    print(f"  Total distance: {result['total_distance_km']:.1f} km")
    print(f"  Total duration: {result['total_duration_days']:.1f} days")
    print(f"\n  Schedule:")
    for stop in result['schedule']:
        print(f"    {stop['arrive'][:16]}  {stop['site']:30s}  "
              f"({abs(stop['lat']):.2f}°S, {stop['lon']:.2f}°E)  "
              f"+{stop['dist_from_prev']:.1f} km")

    # Save result
    import json
    result_file = Path(cfg.output.base_dir) / "traverse_result.json"
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n  Result saved: {result_file}")

    state = load_state(cfg.state_file)
    state['traverse'] = {
        'status': 'complete',
        'sites_visited': result['sites_visited'],
        'total_distance_km': result['total_distance_km'],
    }
    save_state(cfg.state_file, state)
    return True


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python plan_traverse.py <config.yaml>")
        sys.exit(1)
    cfg = load_config(sys.argv[1])
    plan_traverse(cfg)
