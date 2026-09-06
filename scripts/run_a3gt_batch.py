#!/usr/bin/env python3
"""
A3GT batch driver — runs PTM3D 18-year chunked simulations for a list of
candidate sites that all share one regional DEM.

For each site this driver:
  1. Generates a variable-resolution ROI PLY via `dtm_to_ply_delaunay`
     (1 m/px in the central 1-km-diameter core, quadratic falloff to
     10 km, ~5 M faces total).
  2. Renders a per-site YAML config from the 18-year template
     by substituting `simulation.start_time` (kept), `mesh.import_file`,
     and `output.directory`.
  3. Invokes `scripts/run_chunked_simulation.py --resume` to run the
     full 18.6-year (1-day cadence) chunked simulation.

Sites are processed sequentially because GPU + RAM are shared
resources. Each site uses --resume so a single chunk failure does not
lose more than ~10 minutes of progress.

**The site list, the DEM and the template are INPUTS, not constants.** They
used to be a table of coordinates and two absolute paths in this file.
Candidate site coordinates are not published with PTM3D, so the catalog now
comes from a GeoJSON of `[lon, lat]` positions that the operator supplies, the
DEM comes from `--dem`, the template from `--template`, and NONE has a
default: with one missing the driver exits and says what it wants. A silent
fallback to a built-in list is the one behavior worth refusing here — it would
run a whole 18-year batch on the wrong places and look completely normal doing
it.

Usage::

    python3 scripts/run_a3gt_batch.py --sites-geojson sites.geojson \\
        --dem dem.tif --template examples/my_18yr.yaml
    # or limit to a specific subset (tags are <prefix><NN> in file order):
    python3 scripts/run_a3gt_batch.py --sites-geojson sites.geojson \\
        --dem dem.tif --template examples/my_18yr.yaml --sites SITE01,SITE02
    # see what would run, without actually running:
    python3 scripts/run_a3gt_batch.py --sites-geojson sites.geojson \\
        --dem dem.tif --template examples/my_18yr.yaml --dry-run
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

# --------------------------------------------------------------------------
# Site catalog
# --------------------------------------------------------------------------

@dataclasses.dataclass
class Site:
    tag: str
    lat: float
    lon: float


#: Tag prefix for sites read out of the catalog, numbered in file order
#: (1-indexed). Overridable so two catalogs' outputs cannot collide.
DEFAULT_SITE_PREFIX = "SITE"

#: What a site tag may contain. Enforced in `load_sites`, because a tag from
#: the catalog becomes a file name in three places.
SAFE_TAG = re.compile(r"[A-Za-z0-9_][A-Za-z0-9._-]*")


def load_sites(path, prefix=DEFAULT_SITE_PREFIX):
    """``[Site, ...]`` from a GeoJSON of ``[lon, lat]`` positions.

    Accepts a bare ``LineString``/``MultiPoint`` geometry, a ``Feature``
    wrapping one, or a ``FeatureCollection`` of ``Point`` features. A feature's
    ``properties.tag`` or ``properties.name`` is used as the tag when it has
    one, so a catalog can carry its own names rather than positional ones.

    Coordinates are ``[lon, lat]`` -- GeoJSON order, and the reverse of how
    they are written everywhere else in this project. Getting that backwards
    puts a polar site on the equator and the run completes, so the range check
    below is not decoration: it is the only thing that catches a transposed
    file.
    """
    try:
        document = json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        sys.exit(f"Could not read the site catalog {path}: {exc}")

    positions, tags = [], []

    def take(geometry, properties=None):
        kind = (geometry or {}).get("type")
        coords = (geometry or {}).get("coordinates")
        if kind in ("LineString", "MultiPoint"):
            positions.extend(coords or [])
            tags.extend([None] * len(coords or []))
        elif kind == "Point":
            positions.append(coords)
            props = properties or {}
            tags.append(props.get("tag") or props.get("name"))
        else:
            sys.exit(f"{path}: geometry type {kind!r} is not a site list; "
                     f"expected LineString, MultiPoint or Point.")

    kind = document.get("type") if isinstance(document, dict) else None
    if kind == "FeatureCollection":
        for feature in document.get("features", []):
            take(feature.get("geometry"), feature.get("properties"))
    elif kind == "Feature":
        take(document.get("geometry"), document.get("properties"))
    else:
        take(document if isinstance(document, dict) else {})

    if not positions:
        sys.exit(f"{path} contains no site positions.")

    sites = []
    for index, (position, tag) in enumerate(zip(positions, tags), start=1):
        try:
            lon, lat = float(position[0]), float(position[1])
        except (TypeError, ValueError, IndexError):
            sys.exit(f"{path}: position {index} is not a [lon, lat] pair: "
                     f"{position!r}")
        if not -90.0 <= lat <= 90.0:
            sys.exit(f"{path}: position {index} has latitude {lat}, which is "
                     f"out of range. GeoJSON order is [lon, lat] -- this file "
                     f"looks transposed.")
        sites.append(Site(tag or f"{prefix}{index:02d}", lat=lat, lon=lon))

    seen = set()
    for site in sites:
        # A TAG BECOMES THREE FILESYSTEM PATHS -- the ROI mesh, the rendered
        # per-site config, and the output directory (`mesh_path`, `yaml_path`,
        # `out_dir`). It used to be a constant in this file and is now read out
        # of a file the operator points at, so it is validated rather than
        # trusted: a tag of "../../etc/x" would otherwise write outside the
        # trees this driver owns, and one containing a separator would write
        # somewhere nobody looks for it.
        if not SAFE_TAG.fullmatch(site.tag):
            sys.exit(f"{path}: the site tag {site.tag!r} is not usable as a "
                     f"file name. Tags become the mesh, config and output "
                     f"directory names, so they are limited to letters, "
                     f"digits, '.', '_' and '-', and may not begin with '.'.")
        if site.tag in seen:
            sys.exit(f"{path}: two sites share the tag {site.tag!r}. A tag "
                     f"names the mesh, the config and the output directory, so "
                     f"a duplicate would have them overwrite each other.")
        seen.add(site.tag)
    return sites


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

REPO   = Path(__file__).resolve().parent.parent
#: Bound from --dem in main(). There is deliberately no default: see the module
#: docstring. A module-level constant here was an absolute path to one
#: machine's copy of one campaign's DEM.
DEM    = None
# dtm_to_ply_delaunay is now built from this repository
# (src/cpp/tools/dtm_to_ply_delaunay.cpp -> build/bin/). Prefer that copy: it
# is the binary this checkout built, and it is on local disk, so it cannot hit
# OneDrive's on-demand fetch and the `execve` "operation canceled" (errno 89)
# that motivated the /tmp copy. Both older paths remain as fallbacks, and the
# choice is PRINTED rather than made silently. See BUGS.md PTM-04 and
# docs/dem-to-mesh-workflow.md.
DTM_TOOL_REPO   = REPO / "build" / "bin" / "dtm_to_ply_delaunay"
DTM_TOOL_LOCAL  = Path("/tmp/dtm_to_ply_delaunay")
DTM_TOOL_REMOTE = Path("/Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/europa/heat3d/dtm/dtm_to_ply_delaunay")


def _pick_dtm_tool():
    for label, path in (("in-repo build/bin", DTM_TOOL_REPO),
                        ("local /tmp copy", DTM_TOOL_LOCAL)):
        if path.is_file() and os.access(path, os.X_OK):
            print(f"[a3gt-batch] dtm_to_ply_delaunay: {label} -> {path}")
            return path
    print(f"[a3gt-batch] WARNING: no dtm_to_ply_delaunay in {DTM_TOOL_REPO} or "
          f"{DTM_TOOL_LOCAL}; falling back to the OneDrive copy "
          f"{DTM_TOOL_REMOTE}. Build the in-repo tool with "
          f"'cmake --build build --target dtm_to_ply_delaunay -j' (needs GDAL).")
    return DTM_TOOL_REMOTE


DTM_TOOL = _pick_dtm_tool()
PTM_BIN   = REPO / "build" / "bin" / "ptm_thermal_sim"
MERGE_BIN = REPO / "build" / "bin" / "ptm_merge_chunks"
CHUNK_RUN = REPO / "scripts" / "run_chunked_simulation.py"
#: Bound from --template in main(). No default -- see the module docstring.
TEMPLATE  = None

MESH_DIR    = REPO / "data" / "dem" / "a3gt_batch"
EXAMPLES    = REPO / "examples"
OUT_ROOT    = REPO / "output"

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m {s}s"
    h = int(seconds // 3600)
    m = int((seconds - h * 3600) // 60)
    return f"{h}h{m}m"


def _roi_suffix(roi_radius_km: float, roi_core_frac: float,
                roi_max_points: int) -> str:
    """Compact tag to disambiguate meshes built with different ROI args.
    Empty for the default (R=10 km, core_frac=0.05, max_pts=2.5M) so that
    legacy mesh files keep their existing names."""
    if (roi_radius_km == 10.0 and roi_core_frac == 0.05
            and roi_max_points == 2_500_000):
        return ""
    return f"_R{roi_radius_km:g}km_core{roi_core_frac:g}_pts{roi_max_points//1000}k"


def mesh_path(site: Site, roi_radius_km: float = 10.0,
              roi_core_frac: float = 0.05,
              roi_max_points: int = 2_500_000) -> Path:
    """``<mesh dir>/<TAG>_<|lat|>S_<lon>E[<roi suffix>].ply``"""
    suffix = _roi_suffix(roi_radius_km, roi_core_frac, roi_max_points)
    return MESH_DIR / f"{site.tag}_{abs(site.lat):.3f}S_{site.lon:.3f}E{suffix}.ply"


def yaml_path(site: Site, output_root: Path = OUT_ROOT) -> Path:
    """Per-site YAML lives next to the template. Tag includes the parent
    output dir's name when not the default OUT_ROOT, so a custom-output
    run gets its own YAML and doesn't clobber the canonical one."""
    if output_root == OUT_ROOT:
        return EXAMPLES / f"{site.tag}_18yr.yaml"
    return EXAMPLES / f"{site.tag}_18yr_{output_root.name}.yaml"


def out_dir(site: Site, output_root: Path = OUT_ROOT) -> Path:
    return output_root / f"{site.tag}_18yr"


def read_n_faces(ply: Path) -> int:
    """Read 'element face N' from a PLY header. Returns 0 on failure."""
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


# --------------------------------------------------------------------------
# Per-site steps
# --------------------------------------------------------------------------

def build_mesh(site: Site, log_path: Path, dry_run: bool,
               roi_radius_km: float = 10.0,
               roi_core_frac: float = 0.05,
               roi_max_points: int = 2_500_000,
               face_target_band: tuple = (4_000_000, 6_000_000)) -> Path:
    """Run dtm_to_ply_delaunay if the PLY isn't already present and
    well-formed. Returns the PLY path."""
    ply = mesh_path(site, roi_radius_km, roi_core_frac, roi_max_points)
    MESH_DIR.mkdir(parents=True, exist_ok=True)

    if ply.exists():
        # Same rule as a3gt_v2_orchestrator: a mesh is reusable because it is
        # VALID, not because it is big. This driver carried its own copy of the
        # `st_size > 1_000_000` floor, which rejects a legitimately coarse
        # LOLA-fallback core mesh (A03SP010's is 379,264 bytes / 8,746 faces)
        # and forces the rebuild that stopped the 2026-09-03 campaign. Imported
        # rather than re-implemented so the two drivers cannot drift apart
        # again. This driver pins no corrected VF, so validity is the whole
        # check here.
        from a3gt_v2_orchestrator import mesh_reuse_reason
        reason = mesh_reuse_reason(ply)
        if reason is None:
            n_faces = read_n_faces(ply)
            print(f"  [{site.tag}] mesh already present "
                  f"({ply.name}, {n_faces:,} faces) — skipping mesh build")
            return ply
        print(f"  [{site.tag}] WARNING: cached mesh {ply.name} is NOT reusable "
              f"({reason}) — rebuilding it", file=sys.stderr)

    cmd = [
        str(DTM_TOOL),
        str(DEM),
        str(ply),
        "--roi-center",   f"{site.lat:.6f}", f"{site.lon:.6f}",
        "--roi-radius",   f"{roi_radius_km:g}",
        "--roi-core-frac",f"{roi_core_frac:g}",
        "--density-exp",  "2.0",
        "--roi-max-points",str(int(roi_max_points)),
        "--max-edge",     "0",
        "--south-pole-coords",
    ]
    print(f"  [{site.tag}] building mesh: {ply.name}")
    print(f"             cmd: {' '.join(cmd)}")
    if dry_run:
        return ply

    t0 = time.time()
    with open(log_path, "w") as logf:
        ret = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
    dt_s = time.time() - t0
    if ret.returncode != 0:
        sys.exit(f"  [{site.tag}] mesh build FAILED (exit {ret.returncode}); "
                 f"see {log_path}")

    n_faces = read_n_faces(ply)
    print(f"  [{site.tag}] mesh built in {fmt_duration(dt_s)}: "
          f"{n_faces:,} faces "
          f"({ply.stat().st_size / (1024*1024):.1f} MB)")
    lo, hi = face_target_band
    if not (lo <= n_faces <= hi):
        print(f"  [{site.tag}] WARNING: face count {n_faces:,} outside the "
              f"{lo/1e6:g}–{hi/1e6:g} M target band; check the mesh build log.")
    return ply


def render_yaml(site: Site, ply: Path, dry_run: bool,
                output_root: Path = OUT_ROOT) -> Path:
    """Render a per-site YAML config from the 18-year
    template by substituting mesh.import_file and output.directory.

    Doing this with simple string substitution rather than yaml.safe_load
    + dump preserves the template's comments and ordering."""
    out = yaml_path(site, output_root)
    if not TEMPLATE.is_file():
        sys.exit(f"  [{site.tag}] template not found: {TEMPLATE}")
    src = TEMPLATE.read_text()

    # Replace the mesh.import_file: ... line.
    new_lines = []
    for line in src.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("import_file:"):
            indent = line[: len(line) - len(stripped)]
            new_lines.append(f"{indent}import_file: {ply}")
        elif stripped.startswith("directory:"):
            indent = line[: len(line) - len(stripped)]
            new_lines.append(f"{indent}directory: {out_dir(site, output_root)}")
        else:
            new_lines.append(line)
    rendered = "\n".join(new_lines) + "\n"

    print(f"  [{site.tag}] rendering YAML: {out}")
    if dry_run:
        return out
    EXAMPLES.mkdir(parents=True, exist_ok=True)
    out.write_text(rendered)
    return out


def run_chunked_sim(site: Site, cfg: Path, log_path: Path,
                    total_days: float, chunks: int, retries: int,
                    dry_run: bool, output_root: Path = OUT_ROOT) -> float:
    """Invoke the chunked-simulation orchestrator for one site.
    Returns wall time in seconds (0 if dry-run)."""
    cmd = [
        sys.executable, "-u", str(CHUNK_RUN),
        "--config",     str(cfg),
        "--total-days", str(total_days),
        "--output-dir", str(out_dir(site, output_root)),
        "--resume",
        "--retries",    str(retries),
    ]
    if chunks is not None:
        cmd += ["--chunks", str(chunks)]
    print(f"  [{site.tag}] launching chunked simulation; log → {log_path}")
    print(f"             cmd: {' '.join(cmd)}")
    if dry_run:
        return 0.0

    t0 = time.time()
    # Tee both stdout and stderr to the log file so the per-site wall log
    # captures everything from the chunked orchestrator + each chunk's
    # ptm_thermal_sim subprocess. We don't echo to the parent stdout to
    # keep the batch log scannable; the user can `tail -f` the log.
    with open(log_path, "w") as logf:
        ret = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
    dt_s = time.time() - t0
    if ret.returncode != 0:
        sys.exit(f"  [{site.tag}] chunked simulation FAILED "
                 f"(exit {ret.returncode}); see {log_path}")
    return dt_s


def sanity_check_output(site: Site, output_root: Path = OUT_ROOT) -> None:
    """Quick post-run readout: face count + temperature_max range."""
    stats_dir = out_dir(site, output_root) / "statistics"
    p = stats_dir / "temperature_max.bin"
    if not p.is_file():
        print(f"  [{site.tag}] WARNING: {p} missing — output may be incomplete.")
        return
    with open(p, "rb") as f:
        n = struct.unpack("<I", f.read(4))[0]
        # Read a small head sample for a quick range estimate.
        import array
        arr = array.array("f")
        arr.frombytes(f.read(n * 4))
    if not arr:
        return
    print(f"  [{site.tag}] verify: {n:,} faces; "
          f"T_max range {min(arr):.1f}..{max(arr):.1f} K")


def cleanup_chunks(site: Site, output_root: Path = OUT_ROOT) -> None:
    """Remove per-chunk dirs and the thermal_state checkpoint dir for a
    site whose merge has succeeded. Keeps the merged statistics/ tree,
    mesh.ply, time_steps.txt, and any user-created subdirs (e.g. exports/).

    Refuses to clean if the merged statistics aren't intact, so a
    half-failed run can still be resumed."""
    out = out_dir(site, output_root)
    stats = out / "statistics"
    required = ["temperature_avg.bin", "temperature_max.bin",
                "temperature_min.bin", "ice_depth.bin"]
    missing = [r for r in required if not (stats / r).is_file()]
    if missing:
        print(f"  [{site.tag}] post-merge cleanup SKIPPED — "
              f"missing merged stats: {missing}")
        return

    import shutil
    chunk_dirs = sorted(out.glob("chunk_*"))
    state_dir = out / "_thermal_state"
    n_chunks = len(chunk_dirs)
    state_files = list(state_dir.glob("thermal_state_after_chunk_*.bin")) \
                  if state_dir.is_dir() else []
    if not chunk_dirs and not state_dir.is_dir():
        return

    print(f"  [{site.tag}] cleaning {n_chunks} chunk dir(s) + "
          f"{len(state_files)} thermal-state file(s)...")
    for cd in chunk_dirs:
        shutil.rmtree(cd, ignore_errors=True)
    if state_dir.is_dir():
        shutil.rmtree(state_dir, ignore_errors=True)
    print(f"  [{site.tag}] cleanup done.")


# --------------------------------------------------------------------------
# Pre-flight
# --------------------------------------------------------------------------

def preflight() -> None:
    missing = []
    for tag, p in [("DEM", DEM), ("dtm_to_ply_delaunay", DTM_TOOL),
                   ("ptm_thermal_sim", PTM_BIN),
                   ("ptm_merge_chunks", MERGE_BIN),
                   ("18-yr template", TEMPLATE),
                   ("scripts/run_chunked_simulation.py", CHUNK_RUN)]:
        if not p.exists():
            missing.append(f"  {tag}: {p}")
    if missing:
        sys.exit("Pre-flight failed; the following files are missing:\n"
                 + "\n".join(missing))


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    # DEM, TEMPLATE and MESH_DIR are bound from the command line below; they
    # were module constants naming one machine's copy of one campaign's data.
    global DEM, TEMPLATE, MESH_DIR
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    p.add_argument("--sites-geojson", type=Path, required=True,
                   help="REQUIRED. GeoJSON of [lon, lat] site positions "
                        "(LineString, MultiPoint, or a FeatureCollection of "
                        "Points). There is no built-in catalog; see the "
                        "module docstring for why.")
    p.add_argument("--dem", type=Path, required=True,
                   help="REQUIRED. The regional DEM every site is cut from.")
    p.add_argument("--site-prefix", default=DEFAULT_SITE_PREFIX,
                   help=f"Tag prefix for sites the catalog does not name "
                        f"(default: {DEFAULT_SITE_PREFIX}).")
    p.add_argument("--template", type=Path, required=True,
                   help="REQUIRED. The 18-year config template each per-site "
                        "config is rendered from. No default: the one this "
                        "driver used to name is not in the repository, so a "
                        "default here was a path that always failed "
                        "pre-flight.")
    p.add_argument("--mesh-dir", type=Path, default=MESH_DIR,
                   help=f"Where per-site ROI meshes are written "
                        f"(default: {MESH_DIR}).")
    p.add_argument("--sites", default="",
                   help="Comma-separated subset of site tags to run "
                        "(default: every site in --sites-geojson).")
    p.add_argument("--total-days", type=float, default=6798.5,
                   help="Total simulated duration per site in days "
                        "(default: 6798.5 = full 18.61-yr lunar nodal cycle).")
    p.add_argument("--chunks", type=int, default=24,
                   help="Number of chunks per site (default: 24, matches "
                        "the validated mm_site1 batch).")
    p.add_argument("--retries", type=int, default=3,
                   help="Retries on transient GPU hangs per chunk "
                        "(default: 3, with exponential cooldown).")
    p.add_argument("--output-root", type=Path, default=OUT_ROOT,
                   help=f"Parent directory for per-site output dirs "
                        f"(default: {OUT_ROOT}). Use a non-OneDrive path "
                        f"to avoid sync corruption / quota issues.")
    p.add_argument("--roi-radius-km", type=float, default=10.0,
                   help="ROI outer radius in km (default 10).")
    p.add_argument("--roi-core-frac", type=float, default=0.05,
                   help="High-density core radius as fraction of outer "
                        "radius (default 0.05 = 5%%).")
    p.add_argument("--roi-max-points", type=int, default=2_500_000,
                   help="Max points in the variable-density Delaunay "
                        "mesh (default 2.5M ~ 5M faces).")
    p.add_argument("--face-target-min", type=int, default=4_000_000,
                   help="Lower bound on accepted face count for the "
                        "warning band (default 4M).")
    p.add_argument("--face-target-max", type=int, default=6_000_000,
                   help="Upper bound on accepted face count for the "
                        "warning band (default 6M).")
    p.add_argument("--keep-chunks", action="store_true",
                   help="Keep per-chunk dirs and _thermal_state/ after a "
                        "site's merge succeeds. By default they are "
                        "deleted to free disk (the merged statistics/ "
                        "tree, mesh.ply, and time_steps.txt are kept).")
    p.add_argument("--pause-file", type=Path,
                   default=Path("/tmp/a3gt_batch_pause"),
                   help="Path to a sentinel file the driver checks BETWEEN "
                        "sites. If present, the driver exits cleanly after "
                        "the current site's merge + cleanup. Touch the "
                        "file at any time to request a pause; remove it "
                        "before re-running --resume to continue.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the per-site plan and exit without running anything.")
    args = p.parse_args()

    # Bind the three things that used to be constants BEFORE pre-flight, so
    # pre-flight checks what this invocation will actually use.
    DEM = args.dem
    TEMPLATE = args.template
    MESH_DIR = args.mesh_dir

    preflight()

    catalog = load_sites(args.sites_geojson, args.site_prefix)
    requested = [t.strip() for t in args.sites.split(",") if t.strip()]
    if requested:
        sites_to_run = [s for s in catalog if s.tag in requested]
        unknown = set(requested) - {s.tag for s in sites_to_run}
        if unknown:
            sys.exit(f"Unknown site tag(s): {sorted(unknown)}. "
                     f"{args.sites_geojson} defines "
                     f"{[s.tag for s in catalog]}.")
    else:
        sites_to_run = list(catalog)

    args.output_root.mkdir(parents=True, exist_ok=True)

    print(f"\n=== A3GT batch ({len(sites_to_run)} site(s)) ===")
    for i, s in enumerate(sites_to_run):
        print(f"  {i+1:>2}. {s.tag}: lat={s.lat:+.5f}°  lon={s.lon:+.5f}°")
    print(f"  total_days/site: {args.total_days}")
    print(f"  chunks/site:     {args.chunks}")
    print(f"  retries/chunk:   {args.retries}")
    print(f"  output_root:     {args.output_root}")
    print(f"  ROI:             radius={args.roi_radius_km:g} km, "
          f"core_frac={args.roi_core_frac:g}, max_points={args.roi_max_points:,}")

    site_walls = {}
    batch_t0 = time.time()
    paused = False
    for i, site in enumerate(sites_to_run):
        # Honor a manual pause request between sites. The pause-file is
        # only checked here (not mid-site) so any in-flight site finishes
        # its merge + cleanup before we exit.
        if not args.dry_run and args.pause_file and args.pause_file.exists():
            print(f"\n[paused] Found pause file {args.pause_file}; "
                  f"exiting before {site.tag}.")
            print(f"[paused] Remove the file and re-run with --resume to "
                  f"continue from {site.tag}.")
            paused = True
            break
        print(f"\n--- Site {i+1}/{len(sites_to_run)}: {site.tag} ---")
        site_t0 = time.time()
        ply = build_mesh(site,
                         log_path=Path(f"/tmp/{site.tag}_mesh.log"),
                         dry_run=args.dry_run,
                         roi_radius_km=args.roi_radius_km,
                         roi_core_frac=args.roi_core_frac,
                         roi_max_points=args.roi_max_points,
                         face_target_band=(args.face_target_min,
                                           args.face_target_max))
        cfg = render_yaml(site, ply, dry_run=args.dry_run,
                          output_root=args.output_root)
        wall = run_chunked_sim(
            site, cfg,
            log_path=Path(f"/tmp/{site.tag}_batch.log"),
            total_days=args.total_days,
            chunks=args.chunks,
            retries=args.retries,
            dry_run=args.dry_run,
            output_root=args.output_root,
        )
        site_walls[site.tag] = time.time() - site_t0
        if not args.dry_run:
            sanity_check_output(site, args.output_root)
            if not args.keep_chunks:
                cleanup_chunks(site, args.output_root)
        print(f"  [{site.tag}] site total wall time: "
              f"{fmt_duration(site_walls[site.tag])}")

    if args.dry_run:
        print("\n[dry-run] no changes made.")
        return

    if paused:
        print("\n=== A3GT batch PAUSED ===")
    else:
        print("\n=== A3GT batch complete ===")
    print(f"  total wall time: {fmt_duration(time.time() - batch_t0)}")
    for tag, w in site_walls.items():
        print(f"  {tag}: {fmt_duration(w)}")


if __name__ == "__main__":
    main()
