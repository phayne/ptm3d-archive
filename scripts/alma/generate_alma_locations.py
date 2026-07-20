#!/usr/bin/env python3
"""
Generate location grids and depth files for PTM3D ALMA comparison runs.

For each ALMA observation epoch, creates a grid of lat/lon points covering
the visible hemisphere so PTM3D can export T(z) depth profiles at those
locations. Also generates a shared depths file.

Usage:
  python generate_alma_locations.py              # Generate all
  python generate_alma_locations.py --list        # Preview locations
  python generate_alma_locations.py --label H1    # Single observation
"""

import argparse
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from setup_ptm3d_runs import get_published_tb, get_all_observations

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs" / "alma"


def emission_angle_cos(lat_deg, lon_deg, sub_obs_lat_deg, sub_obs_lon_deg):
    """Cosine of emission angle for a surface point."""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    lat0 = np.radians(sub_obs_lat_deg)
    lon0 = np.radians(sub_obs_lon_deg)
    return (np.sin(lat0) * np.sin(lat) +
            np.cos(lat0) * np.cos(lat) * np.cos(lon - lon0))


def generate_visible_hemisphere_grid(sub_obs_lon, sub_obs_lat,
                                      n_lat=15, n_lon=24,
                                      max_emission_angle=80.0):
    """Generate lat/lon points visible from the sub-observer point.

    Parameters
    ----------
    sub_obs_lon : float
        Sub-observer longitude (degrees West).
    sub_obs_lat : float
        Sub-observer latitude (degrees).
    n_lat : int
        Number of latitude bins.
    n_lon : int
        Number of longitude bins.
    max_emission_angle : float
        Maximum emission angle to include (degrees).

    Returns
    -------
    list of tuples
        (name, lat, lon, cos_emission, weight) for each valid point.
    """
    lats = np.linspace(-85, 85, n_lat)
    lons = np.linspace(0, 360 - 360 / n_lon, n_lon)

    cos_em_max = np.cos(np.radians(max_emission_angle))

    points = []
    for lat in lats:
        for lon in lons:
            cos_em = emission_angle_cos(lat, lon, sub_obs_lat, sub_obs_lon)
            if cos_em < cos_em_max:
                continue

            # Weight by projected area: cos(emission) × cos(latitude)
            weight = cos_em * np.cos(np.radians(lat))
            name = f"lat_{lat:+06.1f}_lon_{lon:06.1f}"
            name = name.replace("+", "p").replace("-", "m").replace(".", "d")
            points.append((name, lat, lon, cos_em, weight))

    return points


def generate_depths_file(max_depth_m=0.5, n_points=50):
    """Generate a depth grid for ALMA subsurface probing.

    Uses geometric spacing near the surface (where thermal gradients are
    steepest) and extends to at least 3× the Band 3 penetration depth.
    Band 3 at 97.5 GHz with 50% porosity ice has penetration depth ~20 cm.

    Parameters
    ----------
    max_depth_m : float
        Maximum depth in meters.
    n_points : int
        Number of depth points.

    Returns
    -------
    ndarray
        Depth values in meters.
    """
    # Geometric spacing from 0.1 mm to max_depth
    depths = np.geomspace(1e-4, max_depth_m, n_points)
    return depths


def write_locations_csv(points, output_path):
    """Write a PTM3D-compatible locations CSV file.

    Format: name,lat,lon (one point per line, header row).

    Parameters
    ----------
    points : list of tuples
        (name, lat, lon, cos_em, weight) from generate_visible_hemisphere_grid.
    output_path : Path
        Output file path.
    """
    with open(output_path, "w") as f:
        f.write("name,lat,lon\n")
        for name, lat, lon, _, _ in points:
            f.write(f"{name},{lat:.2f},{lon:.2f}\n")


def write_depths_file(depths, output_path):
    """Write a PTM3D-compatible depths file.

    Format: one depth value per line, in meters.
    """
    with open(output_path, "w") as f:
        for d in depths:
            f.write(f"{d:.6f}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate location grids for PTM3D ALMA runs"
    )
    parser.add_argument("--label", help="Generate for single observation")
    parser.add_argument("--list", action="store_true",
                        help="Preview locations without writing")
    parser.add_argument("--n-lat", type=int, default=15,
                        help="Number of latitude bins (default: 15)")
    parser.add_argument("--n-lon", type=int, default=24,
                        help="Number of longitude bins (default: 24)")
    parser.add_argument("--published-only", action="store_true",
                        help="Only generate for observations with published Tb")
    args = parser.parse_args()

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    # Select observations
    if args.published_only:
        observations = get_published_tb()
    else:
        observations = get_all_observations()

    if args.label:
        observations = [o for o in observations if o["label"] == args.label]
        if not observations:
            print(f"  Unknown label: {args.label}")
            return

    # Generate shared depths file
    depths = generate_depths_file()
    depths_path = CONFIG_DIR / "alma_depths.txt"

    if not args.list:
        write_depths_file(depths, depths_path)
        print(f"  Depths file: {depths_path}")
        print(f"    {len(depths)} points, {depths[0]*1000:.1f} mm to "
              f"{depths[-1]*100:.1f} cm")

    # Generate per-epoch location grids
    for obs in observations:
        label = obs["label"]
        points = generate_visible_hemisphere_grid(
            obs["sub_obs_lon"], obs["sub_obs_lat"],
            n_lat=args.n_lat, n_lon=args.n_lon
        )

        if args.list:
            print(f"\n  {label}: {len(points)} locations "
                  f"(sub-obs lon={obs['sub_obs_lon']:.1f}°W)")
            if points:
                weights = [p[4] for p in points]
                print(f"    Lat range: {min(p[1] for p in points):.0f}° to "
                      f"{max(p[1] for p in points):.0f}°")
                print(f"    Lon range: {min(p[2] for p in points):.0f}° to "
                      f"{max(p[2] for p in points):.0f}°")
                print(f"    Total weight: {sum(weights):.2f}")
        else:
            loc_path = CONFIG_DIR / f"alma_{label}_locations.csv"
            write_locations_csv(points, loc_path)
            print(f"  {label}: {len(points)} locations -> {loc_path.name}")

    if not args.list:
        print(f"\n  Generated files for {len(observations)} observations")
        print(f"\n  To use with PTM3D, add to YAML config or command line:")
        print(f"    --locations {CONFIG_DIR}/alma_T1_locations.csv "
              f"--depths {CONFIG_DIR}/alma_depths.txt")


if __name__ == "__main__":
    main()
