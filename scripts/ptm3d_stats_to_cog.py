#!/usr/bin/env python3
"""
Convert PTM3D per-face statistics binaries to Cloud Optimized GeoTIFFs (COGs).

Generic across PTM3D runs: takes a mesh, a statistics/ directory, and an
output directory; writes one COG per available stats field. Site-centered
polar stereographic when |lat| > 70 everywhere; equirectangular otherwise.

Example:

    /Users/paha3326/anaconda3/envs/europa_viz/bin/python \\
        scripts/ptm3d_stats_to_cog.py \\
        --mesh   /Users/paha3326/Research/moon/artemis/a3gt/sites_v2_3km_rad/A03MM022/mesh.ply \\
        --stats  /Users/paha3326/Research/moon/artemis/a3gt/sites_v2_3km_rad/A03MM022/statistics \\
        --output /Users/paha3326/Research/moon/artemis/a3gt/sites_v2_3km_rad/A03MM022/cog

Default resolution is auto-estimated from face spacing. For A3GT v2 1 m/px
meshes pass ``--resolution-m 1`` to force 1 m pixels (matches the colleague's
example COGs).
"""

import argparse
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / 'src' / 'python'))

import numpy as np

try:
    import rasterio  # noqa: F401
except ImportError:
    sys.stderr.write(
        "ERROR: rasterio not importable. Run with the europa_viz Python:\n"
        "       /Users/paha3326/anaconda3/envs/europa_viz/bin/python "
        f"{sys.argv[0]} ...\n"
    )
    sys.exit(2)

from ptm_regridder import (
    STATISTICS_FIELDS, STAT_TYPES, BODY_RADII,
    MeshRegridder, GridResolutionEstimator,
    detect_body_from_mesh, detect_available_statistics,
    read_binary_data, read_ply_mesh_minimal,
)
from ptm_cog import export_field


def _expected_field_files():
    """All (field_name, basename) pairs the converter knows about."""
    out = []
    for base, info in STATISTICS_FIELDS.items():
        if info.get('scalar'):
            out.append((base, f"{base}.bin"))
        else:
            for stat in STAT_TYPES:
                out.append((f"{base}_{stat}", f"{base}_{stat}.bin"))
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Convert PTM3D statistics .bin files to COG GeoTIFFs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--mesh', required=True,
                        help="PTM3D mesh .ply (used for face lat/lon)")
    parser.add_argument('--stats', required=True,
                        help="Directory containing per-face statistics .bin files")
    parser.add_argument('--output', required=True,
                        help="Output directory for .tif COGs")
    parser.add_argument('--fields', nargs='+', default=None,
                        help="Specific field names to export (e.g. "
                             "'temperature_max ice_depth'). Default: all "
                             "available.")
    parser.add_argument('--resolution', type=float, default=None,
                        help="Grid resolution in DEGREES at equator. "
                             "Default: auto from face spacing.")
    parser.add_argument('--resolution-m', type=float, default=None,
                        help="Grid resolution in METERS (converts to degrees "
                             "via body radius). Use --resolution-m 1 for 1 m/px.")
    parser.add_argument('--body-radius', type=float, default=None,
                        help="Body radius in meters. Default: auto-detect from "
                             "mesh; falls back to Moon (1737400 m).")
    parser.add_argument('--blocksize', type=int, default=512,
                        help="COG internal tile size in pixels")
    parser.add_argument('--compress', default='LZW',
                        choices=['LZW', 'DEFLATE', 'ZSTD', 'NONE'],
                        help="COG compression")
    parser.add_argument('--overview-resampling', default='AVERAGE',
                        help="Resampling method for overview levels")
    parser.add_argument('--interpolation', default='linear',
                        choices=['linear', 'idw'],
                        help="Face -> pixel interpolation. 'linear' "
                             "(default) = barycentric over a Delaunay of "
                             "face centroids; fills internal gaps and "
                             "produces NaN only outside the mesh convex "
                             "hull. 'idw' = legacy KDTree k=3 inverse-"
                             "distance weighting with a tight nearest-"
                             "centroid mask (leaves visible internal gaps "
                             "when pixel size < face spacing).")
    parser.add_argument('--fixed-meridian', default='0',
                        help="Central meridian (degrees) for the output "
                             "projection. Default '0' = standard south/north "
                             "polar stereographic, so multiple site COGs "
                             "share the same CRS and overlay correctly in "
                             "any GIS tool. Pass 'auto' (or 'site') to use "
                             "the mesh centroid (site-centered projection - "
                             "matches the older single-site Malapert example "
                             "but doesn't compose well across sites).")
    parser.add_argument('--resume', action='store_true',
                        help="Skip output files that already exist")
    args = parser.parse_args()

    mesh_path = Path(args.mesh).resolve()
    stats_dir = Path(args.stats).resolve()
    output_dir = Path(args.output).resolve()

    if not mesh_path.is_file():
        sys.exit(f"ERROR: mesh file not found: {mesh_path}")
    if not stats_dir.is_dir():
        sys.exit(f"ERROR: stats directory not found: {stats_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load mesh + derive face lat/lon
    t0 = time.time()
    print(f"[1/4] Loading mesh from {mesh_path} ...", flush=True)
    mesh = read_ply_mesh_minimal(mesh_path)
    n_faces = mesh['n_faces']
    print(f"      {n_faces} faces, {mesh['n_vertices']} vertices "
          f"({time.time() - t0:.1f}s)", flush=True)

    # Body radius
    if args.body_radius is not None:
        body_radius = float(args.body_radius)
        body_label = f"user ({body_radius:.0f} m)"
    else:
        body_name = detect_body_from_mesh(mesh)
        if body_name and body_name in BODY_RADII:
            body_radius = BODY_RADII[body_name]
            body_label = f"{body_name} ({body_radius:.0f} m)"
        else:
            body_radius = BODY_RADII['Moon']
            body_label = f"Moon default ({body_radius:.0f} m)"
    print(f"      body radius: {body_label}", flush=True)

    # Resolution
    if args.resolution_m is not None:
        resolution_deg = float(args.resolution_m) / body_radius * (180.0 / np.pi)
        res_label = f"{args.resolution_m} m -> {resolution_deg:.6g} deg"
    elif args.resolution is not None:
        resolution_deg = float(args.resolution)
        res_label = f"{resolution_deg:.6g} deg"
    else:
        resolution_deg = GridResolutionEstimator.estimate_resolution(
            mesh['face_latitudes'], mesh['face_longitudes']
        )
        res_label = f"auto ({resolution_deg:.6g} deg)"
    print(f"      grid resolution: {res_label}", flush=True)

    # Available fields
    available = detect_available_statistics(stats_dir)
    have_names = {f"{base}_{stat}" if stat else base for base, stat in available}
    print(f"      available stats fields: {len(have_names)} "
          f"({', '.join(sorted(have_names))})", flush=True)
    if not have_names:
        sys.exit("ERROR: no stats binaries with non-NaN data found")

    # Pick fields
    if args.fields:
        requested = set(args.fields)
        missing = requested - have_names
        if missing:
            print(f"      WARNING: skipping missing fields: "
                  f"{sorted(missing)}", flush=True)
        targets = sorted(requested & have_names)
    else:
        targets = sorted(have_names)

    # Resolve --fixed-meridian
    fm = args.fixed_meridian
    if isinstance(fm, str) and fm.lower() in ('auto', 'site', 'sitecentered'):
        forced_center_lon = None
        cm_label = "auto (mesh-centered)"
    else:
        try:
            forced_center_lon = float(fm)
            cm_label = f"{forced_center_lon:g} deg (shared CRS)"
        except (TypeError, ValueError):
            sys.exit(f"ERROR: invalid --fixed-meridian {fm!r} (expected a "
                     "number or 'auto')")

    # Build regridder once
    t0 = time.time()
    print(f"[2/4] Building regridder (KDTree over {n_faces} face centroids) ...",
          flush=True)
    print(f"      central meridian: {cm_label}", flush=True)
    regridder = MeshRegridder(
        mesh['face_latitudes'], mesh['face_longitudes'],
        resolution=resolution_deg, body_radius=body_radius,
        center_lon=forced_center_lon,
    )
    print(f"      projection: {regridder.projection} "
          f"({regridder.n_lon} x {regridder.n_lat} grid)", flush=True)
    if regridder.projection == 'stere':
        print(f"      pole_lat={regridder.pole_lat}, "
              f"center_lon={regridder.center_lon:.6f} deg",
              flush=True)
    print(f"      regridder built ({time.time() - t0:.1f}s)", flush=True)
    print(f"      interpolation: {args.interpolation}", flush=True)

    # Export
    print(f"[3/4] Writing COGs to {output_dir} ...", flush=True)
    fields_to_basename = {}
    for base, info in STATISTICS_FIELDS.items():
        if info.get('scalar'):
            fields_to_basename[base] = f"{base}.bin"
        else:
            for stat in STAT_TYPES:
                fields_to_basename[f"{base}_{stat}"] = f"{base}_{stat}.bin"

    written = 0
    skipped = 0
    failed = []
    for field_name in targets:
        bin_name = fields_to_basename[field_name]
        bin_path = stats_dir / bin_name
        out_path = output_dir / f"{field_name}.tif"

        if args.resume and out_path.is_file():
            print(f"      skip  {field_name:30s} (exists)", flush=True)
            skipped += 1
            continue

        face_values = read_binary_data(bin_path, n_faces)
        if face_values is None:
            print(f"      FAIL  {field_name:30s} (could not read {bin_name})",
                  flush=True)
            failed.append(field_name)
            continue

        t0 = time.time()
        try:
            _grid, stats = export_field(
                out_path, face_values, regridder, field_name,
                blocksize=args.blocksize, compress=args.compress,
                overview_resampling=args.overview_resampling,
                interpolation=args.interpolation,
            )
        except Exception as exc:
            print(f"      FAIL  {field_name:30s} ({type(exc).__name__}: {exc})",
                  flush=True)
            failed.append(field_name)
            continue
        dt = time.time() - t0
        size_mb = out_path.stat().st_size / (1 << 20)
        valid_pct = stats.get('STATISTICS_VALID_PERCENT', '?')
        print(f"      ok    {field_name:30s} "
              f"{size_mb:7.1f} MB  valid={valid_pct}%  ({dt:.1f}s)",
              flush=True)
        written += 1

    print(f"[4/4] Done: {written} written, {skipped} skipped, "
          f"{len(failed)} failed", flush=True)
    if failed:
        print(f"      failed fields: {failed}", flush=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
