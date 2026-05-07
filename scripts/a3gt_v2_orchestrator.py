#!/usr/bin/env python3
"""A3GT v2 — site × window orchestrator for the sparse 18-yr lunar workflow.

Drives the per-site mesh generation and per-window simulation runs for the
A3GT v2 batch. Wraps:
  - dtm_to_ply_delaunay (CPU mesh builder) for Phase 1.
  - scripts/run_chunked_simulation.py for per-window thermal+illumination runs.

Two actions:
  --action generate-meshes
      Builds a 1 m/px-center / 5 km-ROI / linear-density-falloff mesh for
      each site whose status == "in" (or the explicit --filter list).
  --action run
      For each site × window in the production (or cross-check) schedule,
      renders examples/a3gt_v2_template.yaml with the window's parameters
      and invokes run_chunked_simulation.py. Resumable: skips windows that
      already have statistics/_raw/.

Inputs:
  --manifest output/a3gt_v2/site_manifest.json       (from a3gt_v2_site_filter.py)
  --schedule output/a3gt_v2/subsolar_production.json (from a3gt_v2_subsolar.py)

Output layout:
  data/dem/a3gt_v2/{ellipse_id}_{|lat|}S_{lon}E_R5km_linear.ply
  output/a3gt_v2/{ellipse_id}/{window_tag}/   (one ptm_thermal_sim run per window)
  output/a3gt_v2/{ellipse_id}/yaml/{window_tag}.yaml  (rendered configs)
  output/a3gt_v2/{ellipse_id}/logs/{window_tag}.log

The CPU mesh builder is reused from scripts/run_a3gt_batch.py — same
binary, but invoked with --density-exp 1.0 (linear) and the per-site
custom output path.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

REPO = Path("/Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/europa/heat3d/ptm")
DEM = Path(
    "/Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/moon/DEM/a3gt/mons-mouton/mm-sfs-dem.tif")
DTM_TOOL_LOCAL = Path("/tmp/dtm_to_ply_delaunay")
DTM_TOOL_REMOTE = Path(
    "/Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/europa/heat3d/dtm/dtm_to_ply_delaunay")
DTM_TOOL = (DTM_TOOL_LOCAL if DTM_TOOL_LOCAL.is_file()
            and os.access(DTM_TOOL_LOCAL, os.X_OK) else DTM_TOOL_REMOTE)
MESH_DIR = REPO / "data" / "dem" / "a3gt_v2"
TEMPLATE = REPO / "examples" / "a3gt_v2_template.yaml"
RUN_CHUNKED = REPO / "scripts" / "run_chunked_simulation.py"
DEFAULT_OUTPUT_ROOT = REPO / "output" / "a3gt_v2"
# Mutable; set in main() based on --output-root.
OUTPUT_ROOT: Path = DEFAULT_OUTPUT_ROOT

PYTHON = "/Users/paha3326/anaconda3/envs/europa_viz/bin/python3"


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        m = int(seconds / 60); s = int(seconds - m * 60)
        return f"{m}m {s}s"
    h = int(seconds / 3600); m = int((seconds - h * 3600) / 60)
    return f"{h}h {m}m"


def read_n_faces(ply: Path) -> int:
    try:
        with open(ply, "rb") as f:
            for _ in range(200):
                line = f.readline()
                if not line:
                    break
                s = line.decode(errors="replace").strip()
                if s.startswith("element face "):
                    return int(s.split()[2])
                if s == "end_header":
                    break
    except Exception:
        pass
    return 0


def site_mesh_path(site: dict, roi_radius_km: float = 5.0) -> Path:
    """data/dem/a3gt_v2/A03MM022_85.218S_31.850E_R{N}km_linear.ply"""
    rkm = f"{roi_radius_km:g}"
    return MESH_DIR / (
        f"{site['ellipse_id']}_{abs(site['lat_deg']):.3f}S_"
        f"{site['lon_deg']:.3f}E_R{rkm}km_linear.ply"
    )


def window_dir(ellipse_id: str, window_tag: str) -> Path:
    return OUTPUT_ROOT / ellipse_id / window_tag


def window_yaml_path(ellipse_id: str, window_tag: str) -> Path:
    return OUTPUT_ROOT / ellipse_id / "yaml" / f"{window_tag}.yaml"


def window_log_path(ellipse_id: str, window_tag: str) -> Path:
    return OUTPUT_ROOT / ellipse_id / "logs" / f"{window_tag}.log"


def window_is_complete(window_dir_: Path) -> bool:
    """The chunked runner finalizes statistics/ on success. We probe for the
    canonical merged temperature_avg.bin; if present, the window is done."""
    return (window_dir_ / "statistics" / "temperature_avg.bin").is_file()


# --------------------------------------------------------------------------
# Mesh build (Phase 1)
# --------------------------------------------------------------------------

def build_mesh(site: dict, *, dry_run: bool, roi_km: float = 5.0,
               core_frac: float = 0.05, max_points: int = 2_500_000,
               density_exp: float = 1.0) -> Path:
    """Build a 1 m/px-center / N km-ROI / linear-density-falloff PLY for a site.
    Reuses the existing dtm_to_ply_delaunay binary; same call signature as
    run_a3gt_batch.py:build_mesh but with --density-exp 1.0 instead of 2.0."""
    ply = site_mesh_path(site, roi_radius_km=roi_km)
    MESH_DIR.mkdir(parents=True, exist_ok=True)

    if ply.exists() and ply.stat().st_size > 1_000_000 and read_n_faces(ply) > 0:
        n = read_n_faces(ply)
        print(f"  [{site['ellipse_id']}] mesh present ({ply.name}, {n:,} faces) — skip")
        return ply

    cmd = [
        str(DTM_TOOL),
        str(DEM),
        str(ply),
        "--roi-center", f"{site['lat_deg']:.6f}", f"{site['lon_deg']:.6f}",
        "--roi-radius", f"{roi_km:g}",
        "--roi-core-frac", f"{core_frac:g}",
        "--density-exp", f"{density_exp:g}",
        "--roi-max-points", str(int(max_points)),
        "--max-edge", "0",
        "--south-pole-coords",
    ]
    print(f"  [{site['ellipse_id']}] building mesh: {ply.name}")
    print(f"             cmd: {' '.join(cmd)}")
    if dry_run:
        return ply

    log_dir = OUTPUT_ROOT / site["ellipse_id"] / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "mesh_build.log"

    t0 = time.time()
    with open(log_path, "w") as logf:
        ret = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    if ret.returncode != 0:
        sys.exit(f"  [{site['ellipse_id']}] mesh build FAILED (exit {ret.returncode}); "
                 f"see {log_path}")
    n = read_n_faces(ply)
    print(f"  [{site['ellipse_id']}] mesh built in {fmt_duration(dt)}: "
          f"{n:,} faces ({ply.stat().st_size / (1024*1024):.1f} MB)")
    return ply


# --------------------------------------------------------------------------
# YAML render (Phase 2/4)
# --------------------------------------------------------------------------

def render_yaml(template_text: str, *, start_time: str, duration_days: float,
                timesteps: int, mesh_path: Path, output_dir: Path) -> str:
    out = template_text
    repl = {
        "{{ start_time }}":    start_time,
        "{{ duration_days }}": f"{duration_days:g}",
        "{{ timesteps }}":     str(int(timesteps)),
        "{{ mesh_path }}":     str(mesh_path),
        "{{ output_dir }}":    str(output_dir),
    }
    for k, v in repl.items():
        out = out.replace(k, v)
    return out


def write_window_yaml(site: dict, window: dict, mesh: Path,
                       template_text: str) -> Path:
    """Render the per-window YAML to disk and return its path."""
    yaml_path = window_yaml_path(site["ellipse_id"], window["tag"])
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    out_dir = window_dir(site["ellipse_id"], window["tag"])
    out_dir.mkdir(parents=True, exist_ok=True)

    duration = window["duration_days"]
    timesteps = int(round(duration * 24))  # hourly cadence
    rendered = render_yaml(
        template_text,
        start_time=window["start_iso"],
        duration_days=duration,
        timesteps=timesteps,
        mesh_path=mesh,
        output_dir=out_dir,
    )
    yaml_path.write_text(rendered)
    return yaml_path


# --------------------------------------------------------------------------
# Run (Phase 2/4)
# --------------------------------------------------------------------------

def run_window(site: dict, window: dict, yaml_path: Path,
                dry_run: bool) -> bool:
    """Invoke run_chunked_simulation.py for one window. Returns True if window
    completes successfully (or was already complete)."""
    out_dir = window_dir(site["ellipse_id"], window["tag"])
    log_path = window_log_path(site["ellipse_id"], window["tag"])
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if window_is_complete(out_dir):
        print(f"  [{site['ellipse_id']}/{window['tag']}] already complete — skip")
        return True

    cmd = [
        PYTHON, str(RUN_CHUNKED),
        "--config",     str(yaml_path),
        "--total-days", f"{window['duration_days']:g}",
        "--output-dir", str(out_dir),
        "--resume",
    ]
    print(f"  [{site['ellipse_id']}/{window['tag']}] running")
    print(f"             cmd: {' '.join(cmd)}")
    if dry_run:
        return True

    t0 = time.time()
    with open(log_path, "w") as logf:
        ret = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    if ret.returncode != 0:
        print(f"  [{site['ellipse_id']}/{window['tag']}] run FAILED "
              f"(exit {ret.returncode}); see {log_path}", file=sys.stderr)
        return False
    print(f"  [{site['ellipse_id']}/{window['tag']}] done in {fmt_duration(dt)}")
    return True


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def filter_sites(manifest: dict, only: list[str], also_partial: bool) -> list[dict]:
    """Return sites matching --filter (or all 'in' / 'in+partial' if no filter)."""
    sites = manifest["sites"]
    if only:
        keep_ids = set(only)
        return [s for s in sites if s["ellipse_id"] in keep_ids]
    accepted = {"in", "partial"} if also_partial else {"in"}
    return [s for s in sites if s["status"] in accepted]


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, required=True,
                   help="Site manifest JSON (a3gt_v2_site_filter.py output).")
    p.add_argument("--schedule", type=Path,
                   help="Subsolar schedule JSON (required for --action run).")
    p.add_argument("--action", choices=("generate-meshes", "run"), required=True)
    p.add_argument("--filter", nargs="+", default=[],
                   help="Restrict to specific ellipse_ids "
                        "(default: all sites with status='in').")
    p.add_argument("--include-partial", action="store_true",
                   help="Include status='partial' sites (DEM-clipped ROIs).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands without executing.")
    # Mesh knobs
    p.add_argument("--roi-radius-km", type=float, default=5.0)
    p.add_argument("--roi-core-frac", type=float, default=0.05)
    p.add_argument("--roi-max-points", type=int, default=2_500_000)
    p.add_argument("--density-exp", type=float, default=1.0,
                   help="Mesh density falloff exponent (1.0 = linear; A3GT v2 default).")
    p.add_argument("--output-root", type=Path, default=None,
                   help=f"Root directory for per-site simulation outputs "
                        f"(default: {DEFAULT_OUTPUT_ROOT}). "
                        "Use a local-disk path to keep outputs off OneDrive.")
    args = p.parse_args()

    # Honor --output-root by overriding the module-level OUTPUT_ROOT;
    # downstream helpers (window_dir, window_yaml_path, etc.) read this.
    global OUTPUT_ROOT
    if args.output_root is not None:
        OUTPUT_ROOT = args.output_root.expanduser().resolve()
    print(f"Output root: {OUTPUT_ROOT}")

    if not args.manifest.is_file():
        sys.exit(f"manifest not found: {args.manifest}")
    if args.action == "run" and (not args.schedule or not args.schedule.is_file()):
        sys.exit("--action run requires --schedule path to a subsolar schedule JSON")

    manifest = json.loads(args.manifest.read_text())
    sites = filter_sites(manifest, args.filter, args.include_partial)
    if not sites:
        sys.exit("No sites match the filter / status criteria.")

    print(f"Selected {len(sites)} site(s):")
    for s in sites:
        print(f"  {s['ellipse_id']}  {s['region']:<22}  status={s['status']}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    if args.action == "generate-meshes":
        for s in sites:
            build_mesh(
                s,
                dry_run=args.dry_run,
                roi_km=args.roi_radius_km,
                core_frac=args.roi_core_frac,
                max_points=args.roi_max_points,
                density_exp=args.density_exp,
            )
        return 0

    # action == run
    schedule = json.loads(args.schedule.read_text())
    windows = schedule["windows"]
    template_text = TEMPLATE.read_text()
    print(f"\nSchedule: {schedule['year_set']} ({len(windows)} windows)\n")

    n_total = len(sites) * len(windows)
    n_done = n_total
    failures: list[tuple[str, str]] = []
    start_all = time.time()

    for i, s in enumerate(sites):
        # First check / build mesh for this site.
        mesh = build_mesh(
            s,
            dry_run=args.dry_run,
            roi_km=args.roi_radius_km,
            core_frac=args.roi_core_frac,
            max_points=args.roi_max_points,
            density_exp=args.density_exp,
        )
        for j, w in enumerate(windows):
            yaml_path = write_window_yaml(s, w, mesh, template_text)
            ok = run_window(s, w, yaml_path, dry_run=args.dry_run)
            done_idx = i * len(windows) + j + 1
            elapsed = time.time() - start_all
            if done_idx > 0 and not args.dry_run:
                eta = elapsed * (n_total - done_idx) / done_idx
                print(f"  [progress] {done_idx}/{n_total} windows  "
                      f"elapsed={fmt_duration(elapsed)}  ETA={fmt_duration(eta)}")
            if not ok:
                failures.append((s["ellipse_id"], w["tag"]))
                n_done -= 1

    print()
    print(f"Done in {fmt_duration(time.time() - start_all)}: "
          f"{n_done}/{n_total} windows succeeded.")
    if failures:
        print(f"\nFailures ({len(failures)}):")
        for sid, tag in failures:
            print(f"  {sid}/{tag}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
