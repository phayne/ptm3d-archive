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
import atexit
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

# AGT campaign portability (Phase R): REPO now defaults to wherever THIS file
# lives (self-relative) instead of a hardcoded OneDrive path, so a checkout
# such as ~/Projects/ptm runs fully off its own tree (scripts, templates,
# meshes) with zero OneDrive dependency in the hot path -- a OneDrive outage
# mid-campaign no longer stalls a 4-day unattended run. PTM_A3GT_REPO still
# overrides explicitly for a caller that needs the old hardcoded behavior.
# Byte-identical for any existing caller: __file__ resolves to the same path
# the old hardcoded constant pointed at when this file is run from that tree.
REPO = Path(os.environ.get(
    "PTM_A3GT_REPO", str(Path(__file__).resolve().parents[1])))
sys.path.insert(0, str(REPO / "scripts"))
# Legacy single-DEM fallback (used only when the manifest is NOT catalog mode).
DEFAULT_DEM = Path(
    "/Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/moon/DEM/a3gt/mons-mouton/mm-sfs-dem.tif")
DTM_TOOL_LOCAL = Path("/tmp/dtm_to_ply_delaunay")
DTM_TOOL_REMOTE = Path(
    "/Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/europa/heat3d/dtm/dtm_to_ply_delaunay")
DTM_TOOL = (DTM_TOOL_LOCAL if DTM_TOOL_LOCAL.is_file()
            and os.access(DTM_TOOL_LOCAL, os.X_OK) else DTM_TOOL_REMOTE)
# PTM_A3GT_MESH_DIR: point mesh/VRT lookup at a local, pre-verified copy
# (e.g. ~/projects/a3gt/meshes/) instead of REPO/data/dem/a3gt_v2 -- kills the
# OneDrive-eviction hazard for the one artifact class (multi-GB core+ring
# meshes) that's actually big enough to matter. Unset => old REPO-relative
# behavior, unchanged.
MESH_DIR = Path(os.environ.get(
    "PTM_A3GT_MESH_DIR", str(REPO / "data" / "dem" / "a3gt_v2")))
TEMPLATE = REPO / "examples" / "a3gt_v2_template.yaml"
# PTM_A3GT_TEMPLATE_RAD: override the radiation-enabled template (e.g. to
# select a Hayne-albedo variant for baseline-parity comparisons) without
# touching the file every other caller of this script uses by default.
TEMPLATE_RAD = Path(os.environ.get(
    "PTM_A3GT_TEMPLATE_RAD", str(REPO / "examples" / "a3gt_v2_template_radiation.yaml")))
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


def site_mesh_path(site: dict, roi_radius_km: float = 5.0,
                   r_core_km: float | None = None,
                   r_ext_km: float | None = None,
                   density_exp: float | None = None) -> Path:
    """Per-site mesh PLY path.

    Legacy (single-DEM / non-extended): keeps the historical name
    ``A03MM022_85.218S_31.850E_R{N}km_linear.ply`` so existing production
    meshes are still found/skipped.

    Extended (physics-horizon): encodes both radii + the density exponent so
    extended and legacy meshes never collide:
    ``A03MM022_85.218S_31.850E_Rcore3_Rext11.57_p1.5.ply``.
    """
    base = (f"{site['ellipse_id']}_{abs(site['lat_deg']):.3f}S_"
            f"{site['lon_deg']:.3f}E_")
    if r_core_km is not None and r_ext_km is not None:
        return MESH_DIR / (
            f"{base}Rcore{r_core_km:g}_Rext{r_ext_km:g}"
            f"_p{(density_exp if density_exp is not None else 0):g}.ply")
    return MESH_DIR / f"{base}R{roi_radius_km:g}km_linear.ply"


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


def site_is_composited(site_dir_: Path) -> bool:
    """A site is fully done once a3gt_v2_merge_sparse.py has written the
    site-root composite (statistics/temperature_avg.bin + summary.json).

    That merge step ALSO deletes the per-window dirs to reclaim disk, so
    window_is_complete() finds no surviving marker for any window of a
    finished site and run_window() would (wrongly) recompute all of them
    on resume. Guard on the site-root composite to skip the whole site."""
    return ((site_dir_ / "statistics" / "temperature_avg.bin").is_file()
            and (site_dir_ / "summary.json").is_file())


# --------------------------------------------------------------------------
# Mesh build (Phase 1)
# --------------------------------------------------------------------------

def build_mesh(site: dict, *, dry_run: bool, roi_km: float = 5.0,
               core_frac: float = 0.05, max_points: int = 2_500_000,
               density_exp: float = 1.0,
               dem_override: Path | None = None,
               mesh_path: Path | None = None) -> Path:
    """Build a 1 m/px-center / N km-ROI / density-falloff PLY for a site.

    Reuses the existing dtm_to_ply_delaunay binary. ``dem_override`` is the
    per-site VRT mosaic in catalog mode (GDAL opens a .vrt transparently, so
    no C++ change); ``mesh_path`` is the extended-name PLY. Both default to
    the legacy single-DEM behavior when omitted."""
    ply = mesh_path or site_mesh_path(site, roi_radius_km=roi_km)
    dem = dem_override or DEFAULT_DEM
    MESH_DIR.mkdir(parents=True, exist_ok=True)

    if ply.exists() and ply.stat().st_size > 1_000_000 and read_n_faces(ply) > 0:
        n = read_n_faces(ply)
        print(f"  [{site['ellipse_id']}] mesh present ({ply.name}, {n:,} faces) — skip")
        return ply

    cmd = [
        str(DTM_TOOL),
        str(dem),
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
                timesteps: int, mesh_path: Path, output_dir: Path,
                vf_file: Path | None = None,
                occluder_path: Path | None = None) -> str:
    out = template_text
    repl = {
        "{{ start_time }}":    start_time,
        "{{ duration_days }}": f"{duration_days:g}",
        "{{ timesteps }}":     str(int(timesteps)),
        "{{ mesh_path }}":     str(mesh_path),
        "{{ output_dir }}":    str(output_dir),
        "{{ vf_file }}":       str(vf_file) if vf_file is not None else "",
        "{{ occluder_path }}": str(occluder_path) if occluder_path else "",
    }
    for k, v in repl.items():
        out = out.replace(k, v)
    return out


def write_window_yaml(site: dict, window: dict, mesh: Path,
                       template_text: str,
                       vf_file: Path | None = None,
                       occluder: Path | None = None,
                       steps_per_day: float = 24.0) -> Path:
    """Render the per-window YAML to disk and return its path."""
    yaml_path = window_yaml_path(site["ellipse_id"], window["tag"])
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    out_dir = window_dir(site["ellipse_id"], window["tag"])
    out_dir.mkdir(parents=True, exist_ok=True)

    duration = window["duration_days"]
    timesteps = int(round(duration * steps_per_day))  # cadence = samples/Earth-day
    rendered = render_yaml(
        template_text,
        start_time=window["start_iso"],
        duration_days=duration,
        timesteps=timesteps,
        mesh_path=mesh,
        output_dir=out_dir,
        vf_file=vf_file,
        occluder_path=occluder,
    )
    yaml_path.write_text(rendered)
    return yaml_path


# --------------------------------------------------------------------------
# Run (Phase 2/4)
# --------------------------------------------------------------------------
#
# Ported 2026-07-23 from the OneDrive tree's a3gt_v2_orchestrator.py, where
# this heartbeat-based stall watchdog was built and validated but NEVER
# reached ~/Projects/ptm (the tree carrying the corrected-VF / illum / eq-seed
# physics fixes and the pinned campaign binary) -- a real gap: the two trees'
# orchestrator.py had diverged, so the "AGT campaign has the #1 fix" premise
# was false for the tree actually used to launch it. Ports faithfully rather
# than re-deriving, since this exact design already closed the 2026-06-27
# 38-hour silent-stall failure mode (chunk runner SIGKILLed a hung sim but
# then blocked reaping the wedged D-state child forever; a log-heartbeat
# catches that, a wall-clock cap alone would not).

# Popen of the in-flight chunk runner, so a signal/atexit can tear down its
# whole process group (it runs in its own session — see _run_chunked_supervised).
_CURRENT_CHILD: subprocess.Popen | None = None


def _kill_process_group(proc: subprocess.Popen) -> None:
    """SIGTERM then SIGKILL the child's entire process group. The chunk runner
    runs in its own session, so this also reaches its ptm_thermal_sim and
    thermal-monitor grandchildren (a bare proc.kill() would orphan them)."""
    try:
        pgid = os.getpgid(proc.pid)
    except (ProcessLookupError, OSError):
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
        except (ProcessLookupError, OSError):
            return
        try:
            proc.wait(timeout=15)
            return
        except subprocess.TimeoutExpired:
            continue  # escalate to SIGKILL


def _terminate_child(*_args) -> None:
    """Signal/atexit hook: never leave a detached chunk runner + GPU sim
    running after the orchestrator itself is told to stop."""
    if _CURRENT_CHILD is not None and _CURRENT_CHILD.poll() is None:
        _kill_process_group(_CURRENT_CHILD)


def _signal_cleanup(signum, _frame) -> None:
    """On SIGINT/SIGTERM (incl. the watchdog's pkill of this orchestrator),
    tear down the in-flight chunk runner + GPU sim, then exit with the
    conventional 128+signum status."""
    _terminate_child()
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)


def _run_chunked_supervised(cmd: list[str], logf, log_path: Path,
                            stall_timeout_min: float) -> tuple[str, int | None]:
    """Launch run_chunked_simulation.py detached in its OWN process group and
    supervise it on a log-heartbeat. Returns (status, returncode):
      ("complete", 0)   normal success
      ("failed",   rc)  clean non-zero exit (transient / per-window error)
      ("wedged",  None) the window log produced no output for stall_timeout_min
                        => the runner is stuck on an unrecoverable GPU hang;
                        its whole process group has been killed.

    Why heartbeat, not a fixed wall-clock cap: a healthy window keeps writing
    progress to its log, so silence longer than the chunk runner's own
    --chunk-timeout-min is the unambiguous wedge signal. This is the
    2026-06-27 failure mode: the chunk runner's per-chunk timeout SIGKILLed a
    hung sim, but the runner then blocked reaping the wedged (D-state) child
    and never exited, so the orchestrator's blocking wait hung for ~38 h. With
    stall_timeout_min > --chunk-timeout-min this never fires before the runner
    has had its own chance to self-correct."""
    global _CURRENT_CHILD
    # <=0 disables the heartbeat (legacy: block until the runner exits).
    stall_s = stall_timeout_min * 60.0 if stall_timeout_min > 0 else float("inf")
    if 0 < stall_s < 60.0:
        stall_s = 60.0
    poll_s = 30.0
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT,
                            start_new_session=True)
    _CURRENT_CHILD = proc
    try:
        while True:
            try:
                rc = proc.wait(timeout=poll_s)
                return ("complete" if rc == 0 else "failed", rc)
            except subprocess.TimeoutExpired:
                pass
            try:
                quiet_s = time.time() - os.stat(log_path).st_mtime
            except OSError:
                quiet_s = 0.0  # log not created yet — give it time
            if quiet_s >= stall_s:
                _kill_process_group(proc)
                return ("wedged", None)
    finally:
        _CURRENT_CHILD = None


def run_window(site: dict, window: dict, yaml_path: Path,
                dry_run: bool, chunk_timeout_min: float | None = None,
                stall_timeout_min: float = 90.0,
                ptm_bin: Path | None = None, merge_bin: Path | None = None,
                required_env: dict[str, str | None] | None = None) -> str:
    """Invoke run_chunked_simulation.py for one window. Returns one of
    "complete" (success or already done), "failed" (clean non-zero exit,
    INCLUDING an env-completeness preflight failure), or "wedged" (no log
    output for stall_timeout_min => GPU-wedge hang; the runner's process
    group has been killed)."""
    out_dir = window_dir(site["ellipse_id"], window["tag"])
    log_path = window_log_path(site["ellipse_id"], window["tag"])
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if window_is_complete(out_dir):
        print(f"  [{site['ellipse_id']}/{window['tag']}] already complete — skip")
        return "complete"

    # ENV-COMPLETENESS preflight (Phase R, mirrors run_yearlong_wedge_safe.py):
    # refuse to launch a window whose required physics env is missing/wrong
    # rather than discover it post-hoc in the log banners. Radoff-mode
    # windows must NOT pass PTM_HMATRIX_VF_REPLACE in `required_env` (the
    # caller decides that; this function only checks what it's given).
    if required_env:
        import ptm_run_guard as guard
        problems = guard.check_env_completeness(required_env)
        if problems:
            print(f"  [{site['ellipse_id']}/{window['tag']}] ENV-COMPLETENESS "
                  f"gate FAILED -- refusing to launch:", file=sys.stderr)
            for p in problems:
                print(f"    - {p}", file=sys.stderr)
            return "failed"

    cmd = [
        PYTHON, str(RUN_CHUNKED),
        "--config",     str(yaml_path),
        "--total-days", f"{window['duration_days']:g}",
        "--output-dir", str(out_dir),
        "--resume",
    ]
    if chunk_timeout_min is not None:
        cmd.extend(["--chunk-timeout-min", f"{chunk_timeout_min:g}"])
    if ptm_bin is not None:
        cmd.extend(["--ptm-bin", str(ptm_bin)])
    if merge_bin is not None:
        cmd.extend(["--merge-bin", str(merge_bin)])
    print(f"  [{site['ellipse_id']}/{window['tag']}] running")
    print(f"             cmd: {' '.join(cmd)}")
    if dry_run:
        return "complete"

    t0 = time.time()
    with open(log_path, "w") as logf:
        status, rc = _run_chunked_supervised(cmd, logf, log_path,
                                             stall_timeout_min)
    dt = time.time() - t0
    if status == "wedged":
        print(f"  [{site['ellipse_id']}/{window['tag']}] WEDGED — no log output "
              f"for {stall_timeout_min:g} min ({fmt_duration(dt)} elapsed); "
              f"killed the chunk-runner process group. See {log_path}",
              file=sys.stderr)
        return "wedged"
    if status == "failed":
        print(f"  [{site['ellipse_id']}/{window['tag']}] run FAILED "
              f"(exit {rc}); see {log_path}", file=sys.stderr)
        return "failed"
    print(f"  [{site['ellipse_id']}/{window['tag']}] done in {fmt_duration(dt)}")
    return "complete"


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def _catalog_for(manifest: dict) -> dict | None:
    """Load the dem_catalog.json referenced by a catalog-mode manifest."""
    if manifest.get("dem_mode") != "catalog":
        return None
    cp = manifest.get("dem_catalog")
    if cp and Path(cp).is_file():
        return json.loads(Path(cp).read_text())
    sys.exit(f"catalog-mode manifest but dem_catalog not found: {cp}")


def resolve_site_dem(site: dict, manifest: dict, catalog: dict | None,
                     dry_run: bool, clip_radius_km: float, *,
                     include_lola: bool = True, tag: str = "",
                     sfs: bool = True) -> Path:
    """Per-site DEM: a lazily-built ``-te``-clipped VRT (catalog mode) or
    the legacy single DEM.

    ``include_lola=False`` → SfS-only VRT (used for the high-res *core*;
    LOLA must not contaminate the thermally-solved ROI). ``True`` → SfS +
    LOLA underlay (used for the coarse occluder *ring*). ``tag`` keeps the
    core vs ring VRTs distinct on disk.

    MANDATORY ``-te`` clip: an unclipped SfS+LOLA VRT is 608k x 608k @ 1 m
    (the 1 m SfS forces the whole LOLA extent to a 1 m grid). Raises
    ValueError if no canonical DEM is usable (caller → graceful skip)."""
    if catalog is None:
        return DEFAULT_DEM
    from dem_vrt import build_site_vrt
    x, y = float(site["x_m"]), float(site["y_m"])
    pad = max(2000.0, 0.05 * clip_radius_km * 1000.0)   # circle->bbox + proj pad
    r_m = clip_radius_km * 1000.0 + pad
    te = {"x_min": x - r_m, "x_max": x + r_m,
          "y_min": y - r_m, "y_max": y + r_m}
    suffix = f"{('_' + tag) if tag else ''}_R{clip_radius_km:g}km" \
             f"{'_lola' if include_lola else '_sfs'}"
    vrt = MESH_DIR / f"{site['ellipse_id']}{suffix}.vrt"
    if vrt.exists():
        return vrt
    by_id = {d["id"]: d for d in catalog["dems"]}
    # The occluder ring is a COARSE LOLA-resolution annulus (it never enters
    # the radiosity, only the shadow BVH). Including the 1 m SfS forces the
    # DTM tool to read a 1 m grid over the whole R_ext extent — ~10 GB at
    # R_ext=20 km but ~100+ GB at R_ext=82 km (it materialized that and filled
    # the disk). `sfs=False` builds a LOLA-only (~80 m) ring VRT; the ring is
    # sampled at hundreds of meters regardless, so the occluder is unaffected.
    ranked = ([by_id[d["id"]] for d in site.get("dems", [])
               if d.get("role") != "base" and d["id"] in by_id]
              if sfs else [])
    lola = catalog.get("lola_base") if include_lola else None
    if dry_run:
        print(f"  [{site['ellipse_id']}] (dry-run) VRT {vrt.name} "
              f"(-te ±{r_m/1000:g}km) <- {[d['id'] for d in ranked]}"
              f"{' +lola' if lola and lola.get('present') else ''}")
        return vrt
    build_site_vrt(ranked, lola, vrt, target_extent_m=te, verbose=True)
    return vrt


def extended_params_for(site: dict, schedule: dict, args) -> dict:
    """Physics-horizon extended-mesh dtm params for one site."""
    from a3gt_v2_extended_mesh import (extended_mesh_params,
                                       DENSITY_EXP_RING_DEFAULT)
    de = (args.density_exp_ring if args.density_exp_ring is not None
          else DENSITY_EXP_RING_DEFAULT)
    return extended_mesh_params(
        site, schedule, r_core_km=args.r_core_km,
        sun_elev_floor_deg=args.sun_elev_floor_deg,
        r_ext_hard_cap_km=args.r_ext_hard_cap_km,
        max_points=args.ext_max_points, density_exp_ring=de)


def plan_and_build_mesh(site: dict, manifest: dict, catalog: dict | None,
                        schedule: dict | None, args
                        ) -> tuple[Path | None, Path | None]:
    """Resolve the per-site DEM(s) + build the mesh(es).

    Returns ``(core_mesh, occluder_ring)``:
      * legacy / non-extended → ``(mesh, None)`` (unchanged behavior).
      * ``--extended`` → core+ring architecture
        (docs/extended-occluder-architecture.md): the **core** is the
        standard production R3km SfS mesh (thermal/radiation/output, no
        crop); the **ring** is a coarse LOLA annulus passed to the sim as
        ``mesh.occluder_file`` (shadow-ray BVH only).
    ``(None, None)`` if the site must be skipped (no usable DEM)."""
    if not args.extended:
        try:
            dem = resolve_site_dem(site, manifest, catalog, args.dry_run,
                                   args.roi_radius_km)
        except (ValueError, RuntimeError) as e:
            print(f"  [{site['ellipse_id']}] SKIP: DEM unavailable ({e})",
                  file=sys.stderr)
            return None, None
        return (build_mesh(site, dry_run=args.dry_run,
                           roi_km=args.roi_radius_km,
                           core_frac=args.roi_core_frac,
                           max_points=args.roi_max_points,
                           density_exp=args.density_exp,
                           dem_override=dem), None)

    # --- core + occluder-ring architecture ---
    ep = extended_params_for(site, schedule, args)
    r_core, r_ext = ep["r_core_km"], ep["r_ext_km"]
    print(f"  [{site['ellipse_id']}] core+ring: R_core={r_core:g} "
          f"R_ext={r_ext:g} km (raw {ep['r_ext_raw_km']:g}, "
          f"sun_min={ep['sun_elev_min_deg']:g}deg"
          f"{', COVERAGE-CAPPED' if ep['coverage_capped'] else ''})")
    # Core DEM = SfS-only (LOLA must not contaminate the solved ROI). For
    # sites with NO SfS at the center (the LOLA-rescued ones), fall back to
    # a LOLA-resolution core — same as their pre-core/ring behavior; small
    # & GPU-safe, just coarse science.
    try:
        try:
            core_dem = resolve_site_dem(site, manifest, catalog, args.dry_run,
                                        r_core, include_lola=False, tag="core")
        except (ValueError, RuntimeError):
            print(f"  [{site['ellipse_id']}] no SfS at center — LOLA-"
                  f"resolution core fallback", file=sys.stderr)
            core_dem = resolve_site_dem(site, manifest, catalog, args.dry_run,
                                        r_core, include_lola=True,
                                        tag="corelola")
        ring_vrt = resolve_site_dem(site, manifest, catalog, args.dry_run,
                                    r_ext, include_lola=True, tag="ring",
                                    sfs=False)
    except (ValueError, RuntimeError) as e:
        print(f"  [{site['ellipse_id']}] SKIP: DEM unavailable ({e})",
              file=sys.stderr)
        return None, None

    # Core = standard production R3km mesh (proven density; no extended
    # calibration). Legacy site_mesh_path name (roi=r_core).
    core_mesh = build_mesh(site, dry_run=args.dry_run, roi_km=r_core,
                           core_frac=args.roi_core_frac,
                           max_points=args.roi_max_points,
                           density_exp=args.density_exp,
                           dem_override=core_dem)
    # Empty-SfS-core fallback: resolve_site_dem can succeed (a covering SfS
    # DEM exists) yet the dtm tool emits a 0-face PLY when SfS pixels are
    # too sparse / holed inside R_core. Catch that and retry with a LOLA-
    # underlaid core DEM, same fallback the no-SfS-at-center branch uses.
    if (not args.dry_run and core_mesh and Path(core_mesh).exists()
            and read_n_faces(Path(core_mesh)) == 0):
        print(f"  [{site['ellipse_id']}] SfS core produced 0 faces "
              f"(sparse SfS coverage at center) — LOLA-resolution core "
              f"fallback", file=sys.stderr)
        try:
            core_dem = resolve_site_dem(site, manifest, catalog, args.dry_run,
                                        r_core, include_lola=True,
                                        tag="corelola")
        except (ValueError, RuntimeError) as e:
            print(f"  [{site['ellipse_id']}] SKIP: LOLA core DEM unavailable "
                  f"({e})", file=sys.stderr)
            return None, None
        core_mesh = build_mesh(site, dry_run=args.dry_run, roi_km=r_core,
                               core_frac=args.roi_core_frac,
                               max_points=args.roi_max_points,
                               density_exp=args.density_exp,
                               dem_override=core_dem)
        if read_n_faces(Path(core_mesh)) == 0:
            print(f"  [{site['ellipse_id']}] SKIP: LOLA core also produced "
                  f"0 faces", file=sys.stderr)
            return None, None
    ring_ply = (MESH_DIR / f"{site['ellipse_id']}_{abs(site['lat_deg']):.3f}S_"
                f"{site['lon_deg']:.3f}E_ring_Rcore{r_core:g}"
                f"_Rext{r_ext:g}km.ply")
    if ring_ply.exists() and ring_ply.stat().st_size > 100_000:
        print(f"  [{site['ellipse_id']}] ring present ({ring_ply.name}) — skip")
        return core_mesh, ring_ply
    from a3gt_v2_ring_mesh import build_ring
    try:
        build_ring(float(site["lat_deg"]), float(site["lon_deg"]), ring_vrt,
                   ring_ply, r_core_km=r_core, r_ext_km=r_ext,
                   ring_max_points=args.ring_max_points,
                   dry_run=args.dry_run)
    except (RuntimeError, ValueError) as e:
        print(f"  [{site['ellipse_id']}] SKIP: ring build failed ({e})",
              file=sys.stderr)
        return None, None
    return core_mesh, ring_ply


def filter_sites(manifest: dict, only: list[str], also_partial: bool) -> list[dict]:
    """Return sites matching --filter (or all 'in' / 'in+partial' if no filter).

    Catalog-mode sites flagged ``skip`` (no DEM covers the center) are always
    excluded — even when explicitly named in --filter — with the reason
    printed, so an uncovered site never aborts the batch."""
    sites = manifest["sites"]
    if only:
        keep_ids = set(only)
        chosen = [s for s in sites if s["ellipse_id"] in keep_ids]
    else:
        accepted = {"in", "partial"} if also_partial else {"in"}
        chosen = [s for s in sites if s["status"] in accepted]
    out = []
    for s in chosen:
        if s.get("skip"):
            print(f"  [{s['ellipse_id']}] SKIP: "
                  f"{s.get('skip_reason', 'no DEM coverage')}")
            continue
        out.append(s)
    return out


def order_sites(sites: list[dict], order_file: Path | None) -> list[dict]:
    """Reorder (and optionally re-filter) `sites` to match a plain-text,
    one-ellipse_id-per-line priority file. Purely additive: --filter's own
    selection/skip semantics are untouched when order_file is None (the
    existing filter_sites() already iterates in manifest order, unchanged).

    Sites named in the file but absent from `sites` are ignored with a
    warning (already filtered/skipped upstream); sites present in `sites`
    but absent from the file are APPENDED at the end in their existing
    (manifest) order, so an incomplete order file never silently drops a
    site from the run."""
    if order_file is None:
        return sites
    wanted = [l.strip() for l in order_file.read_text().splitlines()
              if l.strip() and not l.strip().startswith("#")]
    by_id = {s["ellipse_id"]: s for s in sites}
    out = []
    seen = set()
    for sid in wanted:
        s = by_id.get(sid)
        if s is None:
            print(f"  [site-order] WARNING: '{sid}' in {order_file} is not "
                  f"in the selected/filtered site set -- skipping", file=sys.stderr)
            continue
        out.append(s)
        seen.add(sid)
    leftover = [s for s in sites if s["ellipse_id"] not in seen]
    if leftover:
        print(f"  [site-order] {len(leftover)} site(s) not listed in "
              f"{order_file} appended at the end (manifest order): "
              f"{', '.join(s['ellipse_id'] for s in leftover)}", file=sys.stderr)
    return out + leftover


# --------------------------------------------------------------------------
# Rolling-VF safety hooks (AGT corrected-VF re-run, 2026-07-23).
#
# Two artifacts per site, staged OUTSIDE the sim's own output tree so a
# post-composite cleanup can delete them without touching statistics/:
#   * a BASE view-factor file, placed at OUTPUT_ROOT/<site>/view_factors.hmat
#     before the site's first window runs. It only needs to (a) exist and
#     (b) declare the SAME face count as the site's current core mesh --
#     PTM_HMATRIX_VF_REPLACE overwrites its actual content unconditionally
#     once the solver initializes (radiation_exchange.cpp ~line 101). This
#     satisfies the "confirm the run LOADS an existing VF, never rebuilds
#     one" rule without needing a physically-correct base VF.
#   * a corrected per-partner CSR, referenced by PTM_HMATRIX_VF_REPLACE.
#
# HARD GUARD: a base VF is only accepted if it is BOTH dimension-matched
# AND small (<= --vf-base-max-mb). 2026-07-23 audit of the 49-site pool
# found 21 sites (Cabeus-B / Nobile Rim 1 / Nobile Rim 2) whose only
# existing view_factors.hmat (from the superseded sites_v2_3km_rad_corering
# generation) is 1.4-4.9 GB -- exactly the multi-GB-VF-on-full-core+ring-mesh
# profile that hard-wedged the GPU on 2026-07-03. Loading one of those as a
# "placeholder" would recreate that wedge risk even though VF_REPLACE would
# eventually discard its content. Such sites are BLOCKED (not started) until
# a properly small, dimension-matched base VF is staged for them.
# --------------------------------------------------------------------------

HMAT_HEADER_STRUCT = "<IIQQQ"  # magic(u32) version(u32) n_rows(u64) n_cols(u64) num_blocks(u64)


def read_hmat_dims(path: Path) -> tuple[int, int] | None:
    """Return (n_rows, n_cols) from a .hmat file's fixed 32-byte header, or
    None if the file is too short / unreadable. Tolerant of the magic/version
    value (older generations used a different magic/version-1 layout but the
    same field ORDER and WIDTH -- confirmed empirically 2026-07-23 against
    46/48 sites' rad_corering-generation VFs, whose n_rows matched their
    current core mesh face count to within a few hundred faces)."""
    dims_blocks = read_hmat_dims_and_blocks(path)
    return None if dims_blocks is None else dims_blocks[:2]


def read_hmat_dims_and_blocks(path: Path) -> tuple[int, int, int] | None:
    """Return (n_rows, n_cols, num_blocks) from a .hmat file's 32-byte
    header. num_blocks==0 is Ada's "truly empty" base VF signature (matches
    the current HMATRIX_MAGIC/VERSION=2 layout -- confirmed via direct hex
    dump of A03MP026_base_empty.vf, 2026-07-23)."""
    import struct
    try:
        with open(path, "rb") as f:
            raw = f.read(32)
        if len(raw) < 32:
            return None
        _magic, _version, n_rows, n_cols, num_blocks = struct.unpack(HMAT_HEADER_STRUCT, raw)
        return int(n_rows), int(n_cols), int(num_blocks)
    except OSError:
        return None


def stage_base_vf(site_id: str, core_mesh: Path, vf_target: Path,
                   vf_base_dir: Path | None, vf_base_max_mb: float) -> tuple[bool, str]:
    """Ensure `vf_target` (OUTPUT_ROOT/<site>/view_factors.hmat) exists and is
    safe to load before this site's first window runs. Returns (ok, reason).

    Looks for a pre-staged candidate at `vf_base_dir/<site_id>.hmat` (or
    `.vf`). Never touches vf_target if it already exists (idempotent resume:
    a site mid-campaign keeps whatever base it started with)."""
    if vf_target.exists():
        return True, f"already present ({vf_target.stat().st_size / 1e6:.1f} MB)"
    if vf_base_dir is None:
        return False, "no --vf-base-dir given and no base VF present"
    n_faces = read_n_faces(core_mesh)
    if n_faces <= 0:
        return False, f"could not read face count from core mesh {core_mesh}"
    candidate = None
    for pattern in (f"{site_id}_base_empty.vf",  # Ada's naming (base_vf_manifest.tsv)
                    f"{site_id}.hmat", f"{site_id}.vf"):
        c = vf_base_dir / pattern
        if c.is_file():
            candidate = c
            break
    if candidate is None:
        return False, (f"no base VF staged at {vf_base_dir}/{site_id}"
                       f"_base_empty.vf (or .hmat/.vf)")
    size_mb = candidate.stat().st_size / 1e6
    if size_mb > vf_base_max_mb:
        return False, (f"base VF {candidate} is {size_mb:.0f} MB > "
                       f"--vf-base-max-mb {vf_base_max_mb:g} -- refusing "
                       f"(multi-GB-VF wedge profile; see 2026-07-03 incident)")
    dims_blocks = read_hmat_dims_and_blocks(candidate)
    if dims_blocks is None:
        return False, f"could not parse .hmat header from {candidate}"
    n_rows, _n_cols, num_blocks = dims_blocks
    if n_rows != n_faces:
        return False, (f"base VF {candidate} n_rows={n_rows} != core mesh "
                       f"face count {n_faces} -- refusing (face-count "
                       f"mismatch is the documented VF-rebuild-fallback "
                       f"wedge trigger)")
    vf_target.parent.mkdir(parents=True, exist_ok=True)
    with open(candidate, "rb") as fi, open(vf_target, "wb") as fo:
        while True:
            b = fi.read(1 << 20)
            if not b:
                break
            fo.write(b)
    if vf_target.stat().st_size != candidate.stat().st_size:
        return False, f"staged copy size mismatch ({candidate} -> {vf_target})"
    return True, (f"staged from {candidate} ({size_mb:.2f} MB, n_faces={n_faces}, "
                  f"num_blocks={num_blocks}{' TRULY EMPTY' if num_blocks == 0 else ''})")


def vf_replace_path(vf_replace_dir: Path | None, site_id: str) -> Path | None:
    if vf_replace_dir is None:
        return None
    # Ada's assemble_and_verify.sh layout: <vf_replace_dir>/<site>/<site>_corrected_vf.csr
    # (a per-site subdirectory, alongside her sweep/ scratch) -- checked first.
    # Flat <vf_replace_dir>/<site>.csr (or .vf) is also accepted for a simpler
    # staging layout if one is ever used instead.
    candidates = [
        vf_replace_dir / site_id / f"{site_id}_corrected_vf.csr",
        vf_replace_dir / f"{site_id}_corrected_vf.csr",
        vf_replace_dir / f"{site_id}.csr",
        vf_replace_dir / f"{site_id}.vf",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


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
    # Core + occluder-ring architecture (catalog mode). Requires --schedule.
    # docs/extended-occluder-architecture.md
    p.add_argument("--extended", action="store_true",
                   help="Core+ring: build the standard R3km SfS core "
                        "(thermal/radiation/output, no crop) plus a coarse "
                        "LOLA occluder ring out to the physics horizon, "
                        "passed to the sim as mesh.occluder_file (shadow "
                        "ray-trace only). Requires --schedule.")
    p.add_argument("--r-core-km", type=float, default=3.0,
                   help="Core ROI radius (the delivered/solved region).")
    p.add_argument("--sun-elev-floor-deg", type=float, default=1.0,
                   help="Floor on min solar elevation (caps R_ext in PSRs).")
    p.add_argument("--r-ext-hard-cap-km", type=float, default=20.0,
                   help="Hard upper cap on the extended radius (20 km: "
                        "near-field occluders + tractable VRT/mesh build).")
    p.add_argument("--ring-max-points", type=int, default=80_000,
                   help="Occluder-ring point budget (default 80k). Raise for "
                        "finer distant-horizon shadows on a large R_ext.")
    p.add_argument("--steps-per-day", type=float, default=24.0,
                   help="Output samples per Earth-day (cadence). 24=hourly "
                        "(default), 48=30-min, 144=10-min.")
    p.add_argument("--density-exp-ring", type=float, default=None,
                   help="Override the calibrated ring density exponent "
                        "(default 1.5, calibrated 2026-05-16).")
    p.add_argument("--ext-max-points", type=int, default=1_400_000,
                   help="Point budget for EXTENDED meshes (default 1.4M, "
                        "re-calibrated at R_ext=20km/LOLA: ~2.8M faces "
                        "(~1.55x production), 2-3km core <=5m (~=production), "
                        "LOLA-saturated ring, far below the GPU-wedge regime. "
                        "Separate from legacy --roi-max-points so non-extended "
                        "runs are unaffected.")
    p.add_argument("--output-root", type=Path, default=None,
                   help=f"Root directory for per-site simulation outputs "
                        f"(default: {DEFAULT_OUTPUT_ROOT}). "
                        "Use a local-disk path to keep outputs off OneDrive.")
    p.add_argument("--radiation", action="store_true",
                   help="Enable surface-to-surface radiation exchange "
                        "(uses examples/a3gt_v2_template_radiation.yaml). "
                        "Per-site view-factor cache is built once and reused "
                        "across the 8 windows.")
    p.add_argument("--chunk-timeout-min", type=float, default=None,
                   help="Per-chunk wall-time cap (minutes) passed through to "
                        "run_chunked_simulation.py. Default = the runner's own "
                        "default (60 min). Use a larger value (e.g. 360) for "
                        "radiation-coupled runs, where each iteration's thermal "
                        "re-run can push window wall-time well past 60 min.")
    p.add_argument("--window-stall-min", type=float, default=None,
                   help="Orchestrator-level watchdog: if a window's log goes "
                        "silent (no output) for this many minutes, treat the "
                        "chunk runner as wedged on an unrecoverable GPU hang, "
                        "kill its process group, and abort the run (resumable). "
                        "Default = (--chunk-timeout-min or 60) + 30, so it never "
                        "fires before the chunk runner's own per-chunk timeout. "
                        "Set <=0 to disable (legacy: block forever on a hang).")
    p.add_argument("--ptm-bin", type=Path, default=None,
                   help="Explicit ptm_thermal_sim binary, forwarded to "
                        "run_chunked_simulation.py --ptm-bin. Pin this "
                        "explicitly for a production campaign; the runner's "
                        "own default is self-relative to ITS OWN tree, which "
                        "silently changes if that tree is rebuilt mid-campaign.")
    p.add_argument("--merge-bin", type=Path, default=None,
                   help="Explicit ptm_merge_chunks binary, forwarded to "
                        "run_chunked_simulation.py --merge-bin.")
    p.add_argument("--site-order-file", type=Path, default=None,
                   help="Plain-text file, one ellipse_id per line (# comments "
                        "OK), giving the exact site processing order. Sites "
                        "selected by --filter/status but absent from this file "
                        "are appended at the end (manifest order), never "
                        "silently dropped. Default: manifest/filter order.")
    p.add_argument("--vf-base-dir", type=Path, default=None,
                   help="Directory of pre-staged per-site base view-factor "
                        "files (<ellipse_id>.hmat or .vf), copied to "
                        "<site>/view_factors.hmat before that site's first "
                        "window (only if not already present). Only accepted "
                        "if <= --vf-base-max-mb AND its declared n_rows "
                        "matches the site's current core-mesh face count -- "
                        "otherwise the site is BLOCKED (skipped, resumable) "
                        "rather than risking a face-count-mismatch VF rebuild "
                        "or loading a multi-GB placeholder (both are the "
                        "documented GPU-wedge trigger classes).")
    p.add_argument("--vf-base-max-mb", type=float, default=50.0,
                   help="Safety ceiling on a --vf-base-dir candidate's size "
                        "(default 50 MB; the known-safe compact VFs measured "
                        "2026-07-23 are ~7-20 MB, the known-dangerous ones "
                        "1.4-4.9 GB -- there is no plausible legitimate file "
                        "in between for this mesh scale).")
    p.add_argument("--vf-replace-dir", type=Path, default=None,
                   help="Directory of per-site corrected-VF CSR files "
                        "(<ellipse_id>.csr), used to set PTM_HMATRIX_VF_REPLACE "
                        "for that site's windows. If --radiation is set and "
                        "this is given but a site's CSR is missing, the site "
                        "is BLOCKED (skipped, resumable) rather than running "
                        "uncorrected radiation silently.")
    p.add_argument("--vf-replace-keep", action="store_true",
                   help="Do NOT delete a site's CSR (--vf-replace-dir entry) "
                        "after its composite merge succeeds. Default: delete "
                        "it (the physics is already baked into the outputs; "
                        "the CSR is scratch, not a deliverable).")
    p.add_argument("--require-env", type=str, default="",
                   help="Comma-separated 'NAME' or 'NAME=value' physics-env "
                        "manifest, checked before EVERY window launch (Phase "
                        "R ENV-COMPLETENESS gate, ptm_run_guard). A missing "
                        "or mismatched var fails that window (not the whole "
                        "campaign) loudly rather than running silently-wrong "
                        "physics. Radoff-mode windows should omit "
                        "PTM_HMATRIX_VF_REPLACE from this manifest.")
    p.add_argument("--min-free-gb", type=float, default=None,
                   help="Disk-floor guard: refuse to START a NEW site (never "
                        "mid-window) if free space on OUTPUT_ROOT's volume "
                        "drops below this many GB. The site in progress is "
                        "allowed to finish; the campaign then stops cleanly "
                        "(resumable) rather than filling the disk.")
    p.add_argument("--window-filter", nargs="+", default=[],
                   help="Restrict which window TAGS from --schedule are "
                        "actually SIMULATED (e.g. Y00_summer), for a fast "
                        "single-window validation run. Deliberately does NOT "
                        "trim the schedule object itself -- plan_and_build_mesh "
                        "still sees the FULL schedule, so R_ext/horizon "
                        "geometry (extended_mesh_params) is computed exactly "
                        "as the full production run would, and matches the "
                        "cached mesh/ring filenames. 2026-07-23 incident: a "
                        "hand-trimmed schedule file changed sun_elev_min_deg "
                        "(4.2127 vs the full schedule's 4.1884 deg for "
                        "A03MP026) and therefore R_ext (27.1522 vs the cached "
                        "27.3108 km) -- a cache-miss that fell through to the "
                        "catalog-based VRT rebuild path (dem_catalog_lola.json "
                        "is an empty stub, dem_root=_empty_demroot -- that "
                        "fallback is non-functional for ANY site, not an "
                        "A03MP026-specific bug) and failed loudly with a "
                        "relative-path GDAL open error. --window-filter is the "
                        "fix: always plan against the untouched schedule.")
    args = p.parse_args()

    # Tear down the in-flight chunk runner + GPU sim if we're interrupted
    # (incl. the watchdog's pkill of this orchestrator) or exit unexpectedly.
    signal.signal(signal.SIGINT, _signal_cleanup)
    signal.signal(signal.SIGTERM, _signal_cleanup)
    atexit.register(_terminate_child)

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
    if args.extended and (not args.schedule or not args.schedule.is_file()):
        sys.exit("--extended requires --schedule (min solar elevation drives "
                 "the physics horizon)")

    manifest = json.loads(args.manifest.read_text())
    catalog = _catalog_for(manifest)
    schedule = (json.loads(args.schedule.read_text())
                if args.schedule and args.schedule.is_file() else None)
    sites = filter_sites(manifest, args.filter, args.include_partial)
    if not sites:
        sys.exit("No sites match the filter / status criteria.")
    sites = order_sites(sites, args.site_order_file)

    print(f"Selected {len(sites)} site(s)"
          f"{' [catalog mode]' if catalog else ''}"
          f"{' [core+occluder-ring]' if args.extended else ''}"
          f"{' [priority order from ' + args.site_order_file.name + ']' if args.site_order_file else ''}:")
    for s in sites:
        print(f"  {s['ellipse_id']}  {s['region']:<22}  status={s['status']}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    if args.action == "generate-meshes":
        for s in sites:
            plan_and_build_mesh(s, manifest, catalog, schedule, args)
        return 0

    # action == run
    schedule = json.loads(args.schedule.read_text())
    windows = schedule["windows"]
    if args.window_filter:
        wanted_tags = set(args.window_filter)
        all_tags = [w["tag"] for w in windows]
        windows = [w for w in windows if w["tag"] in wanted_tags]
        if not windows:
            sys.exit(f"--window-filter {sorted(wanted_tags)} matched no windows "
                     f"in the schedule (available: {all_tags})")
        print(f"--window-filter: simulating {[w['tag'] for w in windows]} of "
              f"{len(all_tags)} scheduled windows ({all_tags}). NOTE: "
              f"plan_and_build_mesh below still uses the FULL, untouched "
              f"`schedule` object for R_ext/horizon geometry -- only this "
              f"local `windows` list (which windows actually get simulated) "
              f"is filtered.")
    template_path = TEMPLATE_RAD if args.radiation else TEMPLATE
    template_text = template_path.read_text()

    # Orchestrator-level stall watchdog (see _run_chunked_supervised). Kept
    # strictly above the chunk runner's own per-chunk timeout so a legitimately
    # long chunk never trips it.
    if args.window_stall_min is not None:
        window_stall_min = args.window_stall_min
    else:
        base = (args.chunk_timeout_min
                if args.chunk_timeout_min and args.chunk_timeout_min > 0
                else 60.0)
        window_stall_min = base + 30.0
    if window_stall_min > 0:
        print(f"Per-window stall watchdog: abort the run if a window log is "
              f"silent for {window_stall_min:g} min (suspected GPU wedge).")
    else:
        print("Per-window stall watchdog: DISABLED (will block on a hang).")

    # Physics-env manifest for the per-window ENV-COMPLETENESS gate. Parsed
    # once; PTM_HMATRIX_VF_REPLACE is added PER-SITE below (its value is
    # site-specific), only when --radiation + --vf-replace-dir are both set.
    base_required_env: dict[str, str | None] = {}
    for tok in (args.require_env or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "=" in tok:
            k, v = tok.split("=", 1)
            base_required_env[k.strip()] = v.strip()
        else:
            base_required_env[tok] = None

    print(f"\nSchedule: {schedule['year_set']} ({len(windows)} windows)")
    print(f"Template: {template_path.name}"
          f"{' [radiation enabled]' if args.radiation else ''}\n")

    n_total = len(sites) * len(windows)
    n_done = n_total
    failures: list[tuple[str, str]] = []
    start_all = time.time()

    skipped: list[str] = []
    blocked: list[tuple[str, str]] = []
    for i, s in enumerate(sites):
        sid = s["ellipse_id"]

        # Disk-floor guard: never START a new site below the free-space
        # floor (a site already in progress is allowed to finish -- this
        # check only runs here, at the top of the per-site loop, between
        # sites). Stops the campaign cleanly (resumable) instead of filling
        # the disk mid-unattended-run.
        #
        # 2026-07-23 finding: a SINGLE A03MP026 CSR build+delete+GPU-window
        # cycle left ~90 GB locked in local Time Machine snapshots (hourly
        # auto-snapshots capturing the churn) even though the live files on
        # disk accounted for only ~13 GB -- `shutil.disk_usage().free`
        # reports the OS-visible free space, which is suppressed by exactly
        # this. Over 4 unattended days of repeated multi-GB CSR churn this
        # could trip --min-free-gb on PURGEABLE, not real, pressure. Thin
        # local snapshots (best-effort; a no-op if tmutil is absent/fails)
        # before trusting the free-space reading.
        if args.min_free_gb is not None:
            try:
                subprocess.run(["tmutil", "thinlocalsnapshots", "/", "50000000000", "4"],
                               capture_output=True, timeout=120)
            except Exception:
                pass  # best-effort; never let snapshot thinning itself block a launch
            free_gb = shutil.disk_usage(OUTPUT_ROOT).free / 1e9
            if free_gb < args.min_free_gb:
                print(f"\nSTOPPING: free space {free_gb:.1f} GB < "
                      f"--min-free-gb {args.min_free_gb:g} before starting "
                      f"site {sid}. Campaign is resumable once space is freed "
                      f"(re-run with the same arguments).", file=sys.stderr)
                # This site and everything after it in priority order were
                # never attempted -- don't let them silently count as
                # "succeeded" in the final n_done/n_total summary.
                n_remaining_sites = len(sites) - i
                n_done -= n_remaining_sites * len(windows)
                blocked.extend((sites[k]["ellipse_id"], "disk floor reached before this site started")
                               for k in range(i, len(sites)))
                break

        # First check / build mesh for this site (catalog VRT + extended
        # params resolved here; None => no usable DEM, skip gracefully).
        core_mesh, occluder = plan_and_build_mesh(
            s, manifest, catalog, schedule, args)
        if core_mesh is None:
            skipped.append(sid)
            n_done -= len(windows)
            continue
        # Per-site view-factor cache: lives at <site_dir>/view_factors.hmat,
        # built by the first window's run, reused by all subsequent windows.
        # (Same geometry across all 8 windows — view factors are identical.)
        vf_file = (OUTPUT_ROOT / sid / "view_factors.hmat"
                    if args.radiation else None)
        if vf_file is not None:
            vf_file.parent.mkdir(parents=True, exist_ok=True)

        # Site-level resume guard: if the composite merge already ran for
        # this site, its per-window dirs were deleted by the merge step,
        # so the per-window check below would recompute the entire site.
        # Skip the whole site instead (idempotent resume).
        if (not args.dry_run
                and site_is_composited(OUTPUT_ROOT / sid)):
            print(f"  [{sid}] site composite present "
                  f"(statistics/ + summary.json) — skipping all "
                  f"{len(windows)} windows")
            continue

        # Rolling-VF pre-step. BLOCK (skip, resumable) rather than let a
        # missing/oversized/mismatched VF fall through to a rebuild or a
        # multi-GB load -- both are the documented GPU-wedge trigger classes.
        site_csr = None
        if args.radiation and not args.dry_run:
            ok, reason = stage_base_vf(sid, core_mesh, vf_file,
                                       args.vf_base_dir, args.vf_base_max_mb)
            if not ok:
                print(f"  [{sid}] BLOCKED (base VF not ready): {reason}",
                      file=sys.stderr)
                blocked.append((sid, f"base VF: {reason}"))
                n_done -= len(windows)
                continue
            print(f"  [{sid}] base VF: {reason}")
            if args.vf_replace_dir is not None:
                site_csr = vf_replace_path(args.vf_replace_dir, sid)
                if site_csr is None:
                    print(f"  [{sid}] BLOCKED (no corrected CSR staged at "
                          f"{args.vf_replace_dir}/{sid}.csr yet)", file=sys.stderr)
                    blocked.append((sid, "corrected CSR not staged"))
                    n_done -= len(windows)
                    continue
                print(f"  [{sid}] corrected VF: {site_csr} "
                      f"({site_csr.stat().st_size / 1e6:.1f} MB)")

        # Per-site env: PTM_HMATRIX_VF_REPLACE is site-specific; the rest
        # (illum fixes, eq-seed, ...) come from --require-env / the caller's
        # own environment and are identical across all 8 windows.
        if site_csr is not None:
            os.environ["PTM_HMATRIX_VF_REPLACE"] = str(site_csr)
        elif "PTM_HMATRIX_VF_REPLACE" in os.environ and args.vf_replace_dir is not None:
            del os.environ["PTM_HMATRIX_VF_REPLACE"]
        required_env = dict(base_required_env)
        if args.radiation and args.vf_replace_dir is not None:
            required_env.setdefault("PTM_HMATRIX_VF_REPLACE", None)

        site_windows_ok = 0
        wedged = False
        for j, w in enumerate(windows):
            yaml_path = write_window_yaml(s, w, core_mesh, template_text,
                                           vf_file=vf_file,
                                           occluder=occluder,
                                           steps_per_day=args.steps_per_day)
            status = run_window(s, w, yaml_path, dry_run=args.dry_run,
                                chunk_timeout_min=args.chunk_timeout_min,
                                stall_timeout_min=window_stall_min,
                                ptm_bin=args.ptm_bin, merge_bin=args.merge_bin,
                                required_env=required_env)
            done_idx = i * len(windows) + j + 1
            elapsed = time.time() - start_all
            if done_idx > 0 and not args.dry_run:
                eta = elapsed * (n_total - done_idx) / done_idx
                print(f"  [progress] {done_idx}/{n_total} windows  "
                      f"elapsed={fmt_duration(elapsed)}  ETA={fmt_duration(eta)}")
            if status == "complete":
                site_windows_ok += 1
            else:
                failures.append((sid, w["tag"]))
                n_done -= 1
                if status == "wedged":
                    # A no-output stall means the chunk runner wedged on an
                    # unrecoverable GPU hang. The driver stays wedged until a
                    # reboot, so every remaining window would hit the same
                    # wall -- abort now instead of hanging or burning the
                    # queue. Resumable: reboot, then re-run with the same args.
                    print(f"\nABORTING run: suspected GPU wedge at "
                          f"{sid}/{w['tag']} (no progress for "
                          f"{window_stall_min:g} min). Reboot, then re-run "
                          f"this orchestrator with the same arguments to "
                          f"resume.", file=sys.stderr)
                    wedged = True
                    break

        if wedged:
            if failures:
                print(f"\nFailures ({len(failures)}):")
                for fsid, tag in failures:
                    print(f"  {fsid}/{tag}")
            return 2

        # After all windows for this site succeed, run the composite merger
        # so the GUI Results Browser can read the site (mesh.ply at site root,
        # statistics/{temperature,illumination,ice_depth}*.bin populated).
        # Skip if any window failed -- the user should debug first.
        #
        # CRITICAL: also skip whenever --window-filter narrowed `windows`
        # below the schedule's real count. `site_windows_ok == len(windows)`
        # only sees the FILTERED list, so a single-window validation run
        # (e.g. V1: --window-filter Y00_summer) would satisfy "all windows
        # complete" after just 1/8 windows, merge+composite the site, and
        # (per a3gt_v2_merge_sparse.py) DELETE the per-window directory --
        # silently corrupting site_is_composited()'s resume state for the
        # REAL full-schedule campaign run later (it would see this site as
        # permanently "done" with only 1 of 8 windows ever having run).
        # 2026-07-23 incident: exactly this happened on A03MP026's first
        # --window-filter run; caught before the real campaign launched.
        if args.window_filter:
            print(f"  [{sid}] --window-filter active ({args.window_filter}) -- "
                  f"this is a partial/validation run; SKIPPING the composite "
                  f"merge (would corrupt resume state for the remaining "
                  f"{len(schedule['windows']) - len(windows)} window(s) not "
                  f"run this pass). Re-run WITHOUT --window-filter to "
                  f"composite this site for real.")
        elif site_windows_ok == len(windows) and not args.dry_run:
            site_dir = OUTPUT_ROOT / sid
            print(f"  [{sid}] all windows complete; running composite merge")
            # core+ring architecture: thermal output is natively the core
            # ROI — NO crop (the ring is occluder-only, never solved/output).
            merge_cmd = [PYTHON, str(REPO / "scripts" / "a3gt_v2_merge_sparse.py"),
                         "--site-dir", str(site_dir)]
            mret = subprocess.run(merge_cmd)
            if mret.returncode != 0:
                print(f"  [{sid}] WARNING: merge exited {mret.returncode}; "
                      f"GUI may not be able to read this site until manually merged",
                      file=sys.stderr)
            else:
                print(f"  [{sid}] merge complete; site is GUI-ready")
                # Post-composite cleanup: the corrected VF is already baked
                # into the outputs -- delete the scratch CSR (never the base
                # view_factors.hmat, which is a real output-tree artifact and
                # small regardless). Keeps the rolling-VF disk footprint to
                # ~1-2 CSRs in flight rather than accumulating all 49.
                if site_csr is not None and not args.vf_replace_keep:
                    try:
                        site_csr.unlink()
                        print(f"  [{sid}] cleaned up corrected-VF scratch CSR {site_csr}")
                    except OSError as e:
                        print(f"  [{sid}] WARNING: could not delete {site_csr}: {e}",
                              file=sys.stderr)

    print()
    print(f"Done in {fmt_duration(time.time() - start_all)}: "
          f"{n_done}/{n_total} windows succeeded.")
    if skipped:
        print(f"\nSkipped {len(skipped)} site(s) (no usable DEM): "
              f"{', '.join(skipped)}")
    if blocked:
        print(f"\nBlocked {len(blocked)} site(s) (rolling-VF not ready -- "
              f"re-run once staged to pick these up):")
        for bsid, reason in blocked:
            print(f"  {bsid}: {reason}")
    if failures:
        print(f"\nFailures ({len(failures)}):")
        for sid, tag in failures:
            print(f"  {sid}/{tag}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
