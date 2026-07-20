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
sys.path.insert(0, str(REPO / "scripts"))
# Legacy single-DEM fallback (used only when the manifest is NOT catalog mode).
DEFAULT_DEM = Path(
    "/Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/moon/DEM/a3gt/mons-mouton/mm-sfs-dem.tif")
DTM_TOOL_LOCAL = Path("/tmp/dtm_to_ply_delaunay")
DTM_TOOL_REMOTE = Path(
    "/Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/europa/heat3d/dtm/dtm_to_ply_delaunay")
DTM_TOOL = (DTM_TOOL_LOCAL if DTM_TOOL_LOCAL.is_file()
            and os.access(DTM_TOOL_LOCAL, os.X_OK) else DTM_TOOL_REMOTE)
MESH_DIR = REPO / "data" / "dem" / "a3gt_v2"
TEMPLATE = REPO / "examples" / "a3gt_v2_template.yaml"
TEMPLATE_RAD = REPO / "examples" / "a3gt_v2_template_radiation.yaml"
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

def run_window(site: dict, window: dict, yaml_path: Path,
                dry_run: bool, chunk_timeout_min: float | None = None) -> bool:
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
    if chunk_timeout_min is not None:
        cmd.extend(["--chunk-timeout-min", f"{chunk_timeout_min:g}"])
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
    vrt = (REPO / "data" / "dem" / "a3gt_v2" /
           f"{site['ellipse_id']}{suffix}.vrt")
    if vrt.exists():
        return vrt
    by_id = {d["id"]: d for d in catalog["dems"]}
    # The occluder ring is a COARSE LOLA-resolution annulus (it never enters
    # the radiosity, only the shadow BVH). Including the 1 m SfS forces the
    # DTM tool to read a 1 m grid over the whole R_ext extent — ~10 GB at
    # R_ext=20 km but ~100+ GB at R_ext=82 km (it materialized that and filled
    # the disk). `sfs=False` builds a LOLA-only (~80 m) ring VRT; the ring is
    # sampled at hundreds of metres regardless, so the occluder is unaffected.
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

    print(f"Selected {len(sites)} site(s)"
          f"{' [catalog mode]' if catalog else ''}"
          f"{' [core+occluder-ring]' if args.extended else ''}:")
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
    template_path = TEMPLATE_RAD if args.radiation else TEMPLATE
    template_text = template_path.read_text()
    print(f"\nSchedule: {schedule['year_set']} ({len(windows)} windows)")
    print(f"Template: {template_path.name}"
          f"{' [radiation enabled]' if args.radiation else ''}\n")

    n_total = len(sites) * len(windows)
    n_done = n_total
    failures: list[tuple[str, str]] = []
    start_all = time.time()

    skipped: list[str] = []
    for i, s in enumerate(sites):
        # First check / build mesh for this site (catalog VRT + extended
        # params resolved here; None => no usable DEM, skip gracefully).
        core_mesh, occluder = plan_and_build_mesh(
            s, manifest, catalog, schedule, args)
        if core_mesh is None:
            skipped.append(s["ellipse_id"])
            n_done -= len(windows)
            continue
        # Per-site view-factor cache: lives at <site_dir>/view_factors.hmat,
        # built by the first window's run, reused by all subsequent windows.
        # (Same geometry across all 8 windows — view factors are identical.)
        vf_file = (OUTPUT_ROOT / s["ellipse_id"] / "view_factors.hmat"
                    if args.radiation else None)
        if vf_file is not None:
            vf_file.parent.mkdir(parents=True, exist_ok=True)

        # Site-level resume guard: if the composite merge already ran for
        # this site, its per-window dirs were deleted by the merge step,
        # so the per-window check below would recompute the entire site.
        # Skip the whole site instead (idempotent resume).
        if (not args.dry_run
                and site_is_composited(OUTPUT_ROOT / s["ellipse_id"])):
            print(f"  [{s['ellipse_id']}] site composite present "
                  f"(statistics/ + summary.json) — skipping all "
                  f"{len(windows)} windows")
            continue

        site_windows_ok = 0
        for j, w in enumerate(windows):
            yaml_path = write_window_yaml(s, w, core_mesh, template_text,
                                           vf_file=vf_file,
                                           occluder=occluder,
                                           steps_per_day=args.steps_per_day)
            ok = run_window(s, w, yaml_path, dry_run=args.dry_run,
                             chunk_timeout_min=args.chunk_timeout_min)
            done_idx = i * len(windows) + j + 1
            elapsed = time.time() - start_all
            if done_idx > 0 and not args.dry_run:
                eta = elapsed * (n_total - done_idx) / done_idx
                print(f"  [progress] {done_idx}/{n_total} windows  "
                      f"elapsed={fmt_duration(elapsed)}  ETA={fmt_duration(eta)}")
            if ok:
                site_windows_ok += 1
            else:
                failures.append((s["ellipse_id"], w["tag"]))
                n_done -= 1

        # After all windows for this site succeed, run the composite merger
        # so the GUI Results Browser can read the site (mesh.ply at site root,
        # statistics/{temperature,illumination,ice_depth}*.bin populated).
        # Skip if any window failed -- the user should debug first.
        if site_windows_ok == len(windows) and not args.dry_run:
            site_dir = OUTPUT_ROOT / s["ellipse_id"]
            print(f"  [{s['ellipse_id']}] all windows complete; running composite merge")
            # core+ring architecture: thermal output is natively the core
            # ROI — NO crop (the ring is occluder-only, never solved/output).
            merge_cmd = [PYTHON, str(REPO / "scripts" / "a3gt_v2_merge_sparse.py"),
                         "--site-dir", str(site_dir)]
            mret = subprocess.run(merge_cmd)
            if mret.returncode != 0:
                print(f"  [{s['ellipse_id']}] WARNING: merge exited {mret.returncode}; "
                      f"GUI may not be able to read this site until manually merged",
                      file=sys.stderr)
            else:
                print(f"  [{s['ellipse_id']}] merge complete; site is GUI-ready")

    print()
    print(f"Done in {fmt_duration(time.time() - start_all)}: "
          f"{n_done}/{n_total} windows succeeded.")
    if skipped:
        print(f"\nSkipped {len(skipped)} site(s) (no usable DEM): "
              f"{', '.join(skipped)}")
    if failures:
        print(f"\nFailures ({len(failures)}):")
        for sid, tag in failures:
            print(f"  {sid}/{tag}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
