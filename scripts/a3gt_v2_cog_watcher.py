#!/usr/bin/env python3
"""
Watch the A3GT v2 production tree and convert each completed site's merged
statistics to Cloud Optimized GeoTIFFs (COGs) as soon as it's safe to.

Safety rules:
  1. Skip a site whose ``statistics/`` is missing any of the 16 expected
     ``.bin`` files.
  2. Skip a site if any running ``ptm_thermal_sim`` process has the site
     name in its config path (``--ignore-running`` to bypass).
  3. Skip a site that an active ``a3gt_v2_orchestrator`` has in its
     ``--filter`` list at or after the currently-processing site (these
     have stale baseline stats waiting to be overwritten by the new
     batch). Sites in the filter that appear BEFORE the active site are
     considered done by this batch and eligible.
  4. Require the newest ``.bin`` mtime in ``statistics/`` to be older than
     ``--settle-minutes`` (default 5 min) - guards against catching a
     mid-merge state.

Output goes to ``<site>/cog/{16 .tif files}``. Each conversion runs under
``nice -n <N>`` (default 19) so it can't preempt the active production
ray-tracing/regridding work (see memory note ``feedback_parallel_local_work``).

Run modes:
  Default = loop forever, polling every ``--poll-seconds`` (default 60),
  exiting cleanly once every site has a complete ``cog/``.
  ``--once`` = single pass, then exit.

Log: each line prefixed with timestamp; pipe to a log file or use
``tail -f`` on stdout.
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


EXPECTED_STATS_BASENAMES = [
    'temperature_avg.bin', 'temperature_max.bin', 'temperature_min.bin',
    'illumination_avg.bin', 'illumination_max.bin', 'illumination_min.bin',
    'direct_solar_avg.bin', 'direct_solar_max.bin', 'direct_solar_min.bin',
    'primary_reflected_avg.bin', 'primary_reflected_max.bin',
    'primary_reflected_min.bin',
    'primary_ir_avg.bin', 'primary_ir_max.bin', 'primary_ir_min.bin',
    'ice_depth.bin',
]
EXPECTED_COG_BASENAMES = [n[:-4] + '.tif' for n in EXPECTED_STATS_BASENAMES]


def ts():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def site_complete_stats(stats_dir: Path) -> bool:
    return all((stats_dir / name).is_file() for name in EXPECTED_STATS_BASENAMES)


def site_complete_cog(cog_dir: Path) -> bool:
    if not cog_dir.is_dir():
        return False
    return all((cog_dir / name).is_file() for name in EXPECTED_COG_BASENAMES)


def site_stable(stats_dir: Path, settle_minutes: float) -> bool:
    bins = list(stats_dir.glob('*.bin'))
    if not bins:
        return False
    newest = max(b.stat().st_mtime for b in bins)
    return (time.time() - newest) > settle_minutes * 60


def _pgrep(pattern):
    try:
        out = subprocess.check_output(
            ['pgrep', '-fl', pattern], stderr=subprocess.DEVNULL,
        ).decode()
    except subprocess.CalledProcessError:
        return ''
    return out


def running_thermal_site_names():
    """Return a set of site IDs that any running ``ptm_thermal_sim`` is
    actively touching (detected by site-name substring in the process's
    command line)."""
    out = _pgrep('ptm_thermal_sim')
    active = set()
    for line in out.splitlines():
        for token in line.split('/'):
            if token.startswith('A03'):
                active.add(token)
    return active


def running_cog_converter_sites():
    """Return a set of site IDs that another instance of the COG converter
    (ptm3d_stats_to_cog.py) is already processing - catches orphans from a
    previously-killed watcher or a manual run by the user."""
    out = _pgrep('ptm3d_stats_to_cog')
    active = set()
    for line in out.splitlines():
        for token in line.split('/'):
            if token.startswith('A03'):
                active.add(token)
    return active


def orchestrator_pending_sites():
    """Return set of site IDs that an active a3gt_v2_orchestrator is still
    going to process (i.e., at-or-after the currently-active site in its
    ``--filter`` list).

    Returns ``set()`` if no orchestrator is running. If we can detect the
    orchestrator's filter but not which site is currently active, we
    conservatively block every site in the filter.
    """
    out = _pgrep('a3gt_v2_orchestrator')
    if not out:
        return set()

    # Parse --filter ARGS from the orchestrator's command line. The flag is
    # nargs='+' so all tokens between '--filter' and the next flag-style
    # argument are site IDs.
    filter_sites = []
    for line in out.splitlines():
        parts = line.split()
        try:
            i = parts.index('--filter')
        except ValueError:
            continue
        for tok in parts[i + 1:]:
            if tok.startswith('-'):
                break
            filter_sites.append(tok)

    if not filter_sites:
        return set()

    # Find the currently-active site (from running ptm_thermal_sim) and
    # cut filter to the suffix starting at that site.
    active = running_thermal_site_names()
    cut_index = None
    for idx, site in enumerate(filter_sites):
        if site in active:
            cut_index = idx
            break

    if cut_index is None:
        # Orchestrator is running but no ptm_thermal_sim active right now -
        # could be between chunks. Block all filter sites to be safe.
        return set(filter_sites)

    return set(filter_sites[cut_index:])


def classify_sites(root: Path, settle_minutes: float, ignore_running: bool):
    """Return (ready, blocked) lists for the current poll.

    ``ready``: list of Path site_dirs that are safe to convert now.
    ``blocked``: list of (site_name, reason) tuples.
    """
    ready, blocked, complete = [], [], []

    if ignore_running:
        active_sites = set()
        pending_sites = set()
    else:
        active_sites = running_thermal_site_names()
        pending_sites = orchestrator_pending_sites()
    converting_sites = running_cog_converter_sites()

    site_dirs = sorted(
        d for d in root.iterdir()
        if d.is_dir() and (d / 'mesh.ply').is_file()
        and (d / 'statistics').is_dir()
    )
    for site_dir in site_dirs:
        site = site_dir.name
        stats = site_dir / 'statistics'
        cog = site_dir / 'cog'

        if site_complete_cog(cog):
            complete.append(site)
            continue
        if not site_complete_stats(stats):
            blocked.append((site, 'stats incomplete'))
            continue
        if site in active_sites:
            blocked.append((site, 'ptm_thermal_sim active'))
            continue
        if site in pending_sites:
            blocked.append((site, 'orchestrator pending re-merge'))
            continue
        if site in converting_sites:
            blocked.append((site, 'conversion already running'))
            continue
        if not site_stable(stats, settle_minutes):
            blocked.append((site, f'stats mtime < {settle_minutes:g} min'))
            continue
        ready.append(site_dir)

    return ready, blocked, complete


def convert_one(site_dir: Path, *, python_exe: str, cli: Path,
                resolution_m: float, nice_level: int) -> int:
    """Run the COG converter on one site. Returns subprocess exit code."""
    site = site_dir.name
    mesh = site_dir / 'mesh.ply'
    stats = site_dir / 'statistics'
    cog = site_dir / 'cog'
    cog.mkdir(parents=True, exist_ok=True)

    cmd = [
        python_exe, str(cli),
        '--mesh', str(mesh),
        '--stats', str(stats),
        '--output', str(cog),
        '--resolution-m', str(resolution_m),
        '--resume',
    ]
    if nice_level is not None and nice_level > 0:
        cmd = ['nice', '-n', str(nice_level)] + cmd

    print(f"[{ts()}] {site}: starting conversion", flush=True)
    print(f"[{ts()}] {site}: $ {' '.join(cmd)}", flush=True)
    t0 = time.time()
    rc = subprocess.call(cmd)
    dt = time.time() - t0
    status = 'ok' if rc == 0 else f'FAILED rc={rc}'
    print(f"[{ts()}] {site}: {status} in {dt/60:.1f} min", flush=True)
    return rc


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--root',
                        default='/Users/paha3326/Research/moon/artemis/a3gt/'
                                'sites_v2_3km_rad')
    parser.add_argument('--python', default=sys.executable,
                        help='Python interpreter to run the COG CLI with '
                             '(should be europa_viz with rasterio/gdal)')
    parser.add_argument('--resolution-m', type=float, default=1.0,
                        help='Grid resolution in meters (matches example COGs)')
    parser.add_argument('--poll-seconds', type=int, default=60)
    parser.add_argument('--settle-minutes', type=float, default=5.0,
                        help='Min minutes since newest stats .bin mtime '
                             'before a site is eligible')
    parser.add_argument('--nice', type=int, default=19,
                        help='nice level for the conversion subprocess; '
                             '0 disables niceing')
    parser.add_argument('--ignore-running', action='store_true',
                        help='Convert even sites whose ptm_thermal_sim is '
                             'actively running. Default: skip.')
    parser.add_argument('--once', action='store_true',
                        help='Convert at most one site then exit')
    parser.add_argument('--dry-run', action='store_true',
                        help='Classify and report; do not convert')
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.is_dir():
        sys.exit(f"ERROR: root not found: {root}")

    cli = Path(__file__).resolve().parent / 'ptm3d_stats_to_cog.py'
    if not cli.is_file():
        sys.exit(f"ERROR: COG CLI not found: {cli}")

    print(f"[{ts()}] watcher starting", flush=True)
    print(f"[{ts()}]   root={root}", flush=True)
    print(f"[{ts()}]   poll={args.poll_seconds}s settle={args.settle_minutes}min "
          f"nice={args.nice} resolution_m={args.resolution_m}", flush=True)

    last_blocked_report = ''
    while True:
        ready, blocked, complete = classify_sites(
            root, args.settle_minutes, args.ignore_running
        )

        total = len(ready) + len(blocked) + len(complete)
        ready_names = [d.name for d in ready]

        if args.dry_run:
            print(f"[{ts()}] DRY-RUN classification:", flush=True)
            print(f"[{ts()}]   complete ({len(complete)}): {complete}",
                  flush=True)
            print(f"[{ts()}]   ready    ({len(ready)}): {ready_names}",
                  flush=True)
            for s, r in blocked:
                print(f"[{ts()}]   blocked  {s:12s} {r}", flush=True)
            break

        if complete and len(complete) == total and total > 0:
            print(f"[{ts()}] all {total} sites complete; exiting", flush=True)
            break

        if not ready:
            blocked_report = ', '.join(f'{s}({r})' for s, r in blocked)
            line = (f"[{ts()}] done={len(complete)}/{total} "
                    f"blocked={len(blocked)} | {blocked_report}")
            # Only re-print blocked state when it changes (keeps log readable)
            if line != last_blocked_report:
                print(line, flush=True)
                last_blocked_report = line
            if args.once:
                break
            time.sleep(args.poll_seconds)
            continue

        last_blocked_report = ''
        site_dir = ready[0]
        convert_one(site_dir,
                    python_exe=args.python, cli=cli,
                    resolution_m=args.resolution_m,
                    nice_level=args.nice)

        if args.once:
            break
        # Loop immediately (no sleep) to pick up the next ready site


if __name__ == '__main__':
    main()
