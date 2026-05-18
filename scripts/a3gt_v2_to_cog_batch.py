#!/usr/bin/env python3
"""
Batch driver: produce COG GeoTIFFs for every A3GT v2 site whose merged
statistics are present.

Layout it walks::

    <root>/<site_id>/mesh.ply
    <root>/<site_id>/statistics/{16 .bin files}
    <root>/<site_id>/cog/{16 .tif COGs}  <- output

Default root: ``/Users/paha3326/Research/moon/artemis/a3gt/sites_v2_3km_rad``.

For each site, invokes ``ptm3d_stats_to_cog.py`` with ``--resolution-m 1`` and
``--resume`` (so it skips fields already converted). Sites whose ``cog/``
already contains all 16 expected COGs are fully skipped.

DO NOT run while a heavy production batch is active on the same workstation;
the regridding step is CPU + memory intensive (see memory note
``feedback_parallel_local_work``).
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_CLI = _HERE / 'ptm3d_stats_to_cog.py'

# Same list as ptm_regridder.STATISTICS_FIELDS minus the scalar/units cruft.
EXPECTED_COG_BASENAMES = [
    'temperature_avg.tif', 'temperature_max.tif', 'temperature_min.tif',
    'illumination_avg.tif', 'illumination_max.tif', 'illumination_min.tif',
    'direct_solar_avg.tif', 'direct_solar_max.tif', 'direct_solar_min.tif',
    'primary_reflected_avg.tif', 'primary_reflected_max.tif',
    'primary_reflected_min.tif',
    'primary_ir_avg.tif', 'primary_ir_max.tif', 'primary_ir_min.tif',
    'ice_depth.tif',
]


def _site_has_complete_cog(cog_dir: Path) -> bool:
    if not cog_dir.is_dir():
        return False
    return all((cog_dir / name).is_file() for name in EXPECTED_COG_BASENAMES)


def main():
    parser = argparse.ArgumentParser(
        description="Batch-convert A3GT v2 per-site stats to COG GeoTIFFs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--root', default='/Users/paha3326/Research/moon/artemis/a3gt/'
                          'sites_v2_3km_rad',
        help="Root of the A3GT v2 production tree (one subdir per site)",
    )
    parser.add_argument(
        '--sites', nargs='+', default=None,
        help="Specific site IDs to convert (default: every subdir of --root "
             "with mesh.ply + statistics/)",
    )
    parser.add_argument(
        '--resolution-m', type=float, default=1.0,
        help="Grid resolution in meters",
    )
    parser.add_argument(
        '--python', default=sys.executable,
        help="Python interpreter to run the CLI with",
    )
    parser.add_argument(
        '--force', action='store_true',
        help="Convert even sites whose cog/ is already complete",
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help="Print what would be done, don't run anything",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.is_dir():
        sys.exit(f"ERROR: root not found: {root}")

    if args.sites:
        site_ids = args.sites
    else:
        site_ids = sorted(
            d.name for d in root.iterdir()
            if d.is_dir() and (d / 'mesh.ply').is_file()
            and (d / 'statistics').is_dir()
        )
    print(f"Found {len(site_ids)} candidate site(s) under {root}", flush=True)
    if not site_ids:
        return

    t_start = time.time()
    done = []
    skipped = []
    failed = []
    for i, site in enumerate(site_ids, start=1):
        site_dir = root / site
        mesh = site_dir / 'mesh.ply'
        stats = site_dir / 'statistics'
        cog = site_dir / 'cog'

        if not mesh.is_file() or not stats.is_dir():
            print(f"[{i}/{len(site_ids)}] SKIP {site}: mesh.ply or "
                  f"statistics/ missing", flush=True)
            skipped.append(site)
            continue

        if not args.force and _site_has_complete_cog(cog):
            print(f"[{i}/{len(site_ids)}] SKIP {site}: cog/ already complete",
                  flush=True)
            skipped.append(site)
            continue

        cog.mkdir(parents=True, exist_ok=True)
        cmd = [
            args.python, str(_CLI),
            '--mesh', str(mesh),
            '--stats', str(stats),
            '--output', str(cog),
            '--resolution-m', str(args.resolution_m),
            '--resume',
        ]
        print(f"[{i}/{len(site_ids)}] {site}  ->  {cog}", flush=True)
        print(f"      $ {' '.join(cmd)}", flush=True)
        if args.dry_run:
            done.append(site)
            continue

        t0 = time.time()
        rc = subprocess.call(cmd)
        dt = time.time() - t0
        if rc == 0:
            print(f"      site done in {dt/60:.1f} min", flush=True)
            done.append(site)
        else:
            print(f"      site FAILED (exit {rc}) after {dt/60:.1f} min",
                  flush=True)
            failed.append(site)

    total = time.time() - t_start
    print(f"\nBatch done in {total/60:.1f} min: {len(done)} converted, "
          f"{len(skipped)} skipped, {len(failed)} failed", flush=True)
    if failed:
        print(f"Failed sites: {failed}", flush=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
