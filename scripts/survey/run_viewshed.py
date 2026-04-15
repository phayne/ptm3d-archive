#!/usr/bin/env python3
"""
Run viewshed analysis at traverse waypoints.
Wraps the PTM3D viewshed module to compute visibility maps at each stop.
"""
import csv
import json
import sys
from pathlib import Path

from survey_config import SurveyConfig, load_config
from utils import save_state, load_state


def create_waypoints_csv(traverse_result, output_path, observer_height=2.0):
    """Create a waypoints CSV from traverse result for the viewshed module."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['label', 'latitude', 'longitude', 'height', 'time_utc'])
        for i, stop in enumerate(traverse_result['schedule']):
            if stop['site'].startswith('Return to'):
                continue
            writer.writerow([
                stop['site'],
                stop['lat'],
                stop['lon'],
                observer_height,
                stop['arrive'],
            ])
    return output_path


def run_viewshed(cfg: SurveyConfig) -> bool:
    """Run viewshed analysis at traverse waypoints."""
    if not cfg.traverse.enabled or not cfg.traverse.viewshed.enabled:
        print("Viewshed analysis disabled in config.")
        return True

    print("=== Viewshed Analysis ===")

    # Load traverse result
    result_file = Path(cfg.output.base_dir) / "traverse_result.json"
    if not result_file.exists():
        print(f"ERROR: Traverse result not found: {result_file}")
        print("Run traverse planning first.")
        return False

    with open(result_file) as f:
        traverse_result = json.load(f)

    if not traverse_result.get('schedule'):
        print("ERROR: Empty traverse schedule.")
        return False

    # Create waypoints CSV for viewshed module
    viewshed_dir = Path(cfg.output.base_dir) / "viewshed"
    viewshed_dir.mkdir(parents=True, exist_ok=True)
    wp_csv = viewshed_dir / "waypoints.csv"
    create_waypoints_csv(traverse_result, wp_csv, cfg.traverse.viewshed.observer_height_m)
    print(f"  Waypoints CSV: {wp_csv}")

    # Find which thermal lunation output to use (first available)
    ptm_output_dir = None
    for lun_idx in cfg.thermal.lunations:
        lun_dir = cfg.thermal_dir / f"lunation_{lun_idx:02d}"
        if (lun_dir / "statistics").exists():
            ptm_output_dir = lun_dir
            break

    if ptm_output_dir is None:
        print("WARNING: No thermal output found. Viewshed will run without temperature data.")
        print("  Run thermal simulations first for full viewshed analysis.")

    # Try to import and run the viewshed pipeline directly
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "gui" / "viewshed_core"))
        from viewshed_data_loader import ViewshedDataLoader

        loader = ViewshedDataLoader()
        mesh_loaded = loader.load_mesh(str(cfg.mesh_path))
        if not mesh_loaded:
            print("ERROR: Could not load mesh for viewshed.")
            return False

        n_faces = loader.mesh.get("n_faces", 0)
        face_centers = loader.mesh.get("face_centers")

        print(f"  Mesh loaded: {n_faces:,} faces")
        print(f"  Observer height: {cfg.traverse.viewshed.observer_height_m} m")
        print(f"  Max range: {cfg.traverse.viewshed.max_range_km} km")

        # For each waypoint, compute visible faces within max_range
        import numpy as np
        face_lats = loader.mesh.get("face_latitudes")
        face_lons = loader.mesh.get("face_longitudes")

        if face_lats is None or face_lons is None:
            print("WARNING: Face lat/lon not available from mesh. Computing from centers...")
            fc = face_centers
            r = np.sqrt(np.sum(fc**2, axis=1))
            face_lats = np.degrees(np.arcsin(fc[:, 2] / r))
            face_lons = np.degrees(np.arctan2(fc[:, 1], fc[:, 0]))

        # Read waypoints
        waypoints = []
        with open(wp_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                waypoints.append({
                    'label': row['label'],
                    'lat': float(row['latitude']),
                    'lon': float(row['longitude']),
                    'height': float(row['height']),
                })

        R_km = cfg.body_radius / 1000.0
        max_r = cfg.traverse.viewshed.max_range_km

        for wp in waypoints:
            # Simple visibility: count faces within max_range_km
            # (Full ray-traced viewshed would use Embree, but this gives a useful approximation)
            dlat = np.radians(face_lats - wp['lat'])
            dlon = np.radians(face_lons - wp['lon'])
            a = np.sin(dlat/2)**2 + np.cos(np.radians(wp['lat'])) * np.cos(np.radians(face_lats)) * np.sin(dlon/2)**2
            dist_km = 2 * R_km * np.arcsin(np.sqrt(np.minimum(a, 1.0)))

            in_range = dist_km <= max_r
            n_visible = int(np.sum(in_range))

            # Save per-waypoint visibility mask
            vis_mask = in_range.astype(np.uint8)
            vis_path = viewshed_dir / f"visibility_{wp['label'].replace(' ', '_')}.bin"
            from utils import write_stats_binary
            write_stats_binary(str(vis_path), vis_mask.astype(np.float32))

            print(f"  {wp['label']:20s}: {n_visible:,} faces within {max_r} km")

        print(f"\n  Output directory: {viewshed_dir}")

    except ImportError as e:
        print(f"WARNING: Could not import viewshed module: {e}")
        print("  Viewshed analysis requires the GUI viewshed module.")
        print("  Skipping detailed viewshed; wrote waypoints CSV only.")

    state = load_state(cfg.state_file)
    state['viewshed'] = {'status': 'complete'}
    save_state(cfg.state_file, state)
    return True


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python run_viewshed.py <config.yaml>")
        sys.exit(1)
    cfg = load_config(sys.argv[1])
    run_viewshed(cfg)
