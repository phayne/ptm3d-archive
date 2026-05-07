#!/usr/bin/env python3
"""A3GT v2 — composite stats merger across 8 sparse-sampled lunation windows.

For one site, gathers all per-chunk raw statistics buffers from every
window (production schedule = 4 yrs × summer+winter = 8 windows; each
window may have one or more RAM-fitting chunks under chunk_NNN/) and
invokes the existing bin/ptm_merge_chunks against the union, producing
a per-site statistics/ folder mathematically equivalent to a single
hypothetical "all sampled time" merge.

Why this works:
  - bin/ptm_merge_chunks already does the math we need:
      mean   = sum(sum_buffers) / sum(count_buffers)
      max/min = per-face per-chunk max/min reduction
      ice_depth = Schorghofer J(z) from sum_E / n_samples
  - It accepts an arbitrary list of chunk dirs via --chunks. Each must
    contain statistics/_raw/{datatype}_sum.bin/_count.bin (and the
    ice_depth_sum_E.bin / depth_grid.bin / ice_depth_n_samples.txt
    triplet for ice-stability).
  - Per-window run_chunked_simulation.py keeps each chunk_NNN/ dir
    intact (with the raw buffers) after its own merge, so the cross-
    window merge has everything it needs.

The output layout per site is:
  output/a3gt_v2/{ellipse_id}/statistics/             <- composite final
  output/a3gt_v2/{ellipse_id}/{window_tag}/            <- per-window
  output/a3gt_v2/{ellipse_id}/{window_tag}/chunk_000/  <- per-chunk

A summary.json is also written with key scalars (mean PSR T, deepest
stable ice depth, fraction of faces with finite stable depth, etc.)
useful for the all-sites-summary table downstream.
"""

from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path("/Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/europa/heat3d/ptm")
MERGE_BIN = REPO / "build" / "bin" / "ptm_merge_chunks"
MERGE_BIN_ALT = REPO / "bin" / "ptm_merge_chunks"


def find_merge_bin() -> Path:
    for p in (MERGE_BIN, MERGE_BIN_ALT):
        if p.is_file():
            return p
    sys.exit("ptm_merge_chunks binary not found; build the C++ targets first.")


def discover_chunk_dirs(site_dir: Path) -> list[Path]:
    """Return all chunk_NNN/ directories beneath every window subdir of site_dir.

    Skips windows that look incomplete (missing statistics/temperature_avg.bin).
    Skips chunks that don't have the raw buffers.
    """
    chunk_dirs: list[Path] = []
    for window_dir in sorted(site_dir.iterdir()):
        if not window_dir.is_dir() or window_dir.name in ("yaml", "logs", "statistics"):
            continue
        # Per-window completion sentinel from ptm_merge_chunks.
        if not (window_dir / "statistics" / "temperature_avg.bin").is_file():
            print(f"  warn: skipping incomplete window {window_dir.name}", file=sys.stderr)
            continue
        for chunk_dir in sorted(window_dir.glob("chunk_*")):
            raw = chunk_dir / "statistics" / "_raw"
            if not raw.is_dir():
                continue
            if not any(raw.glob("*_sum.bin")):
                continue
            chunk_dirs.append(chunk_dir)
    return chunk_dirs


def read_float32_array(path: Path) -> np.ndarray:
    """Read uint32 N + N float32 array."""
    with path.open("rb") as f:
        n = struct.unpack("<I", f.read(4))[0]
        a = np.frombuffer(f.read(n * 4), dtype="<f4")
    if a.shape[0] != n:
        raise ValueError(f"{path}: header N={n} but {a.shape[0]} floats read")
    return a


def write_summary(site_dir: Path, ellipse_id: str, n_windows: int,
                   n_chunks: int) -> Path:
    """Write a brief per-site summary.json."""
    stats = site_dir / "statistics"
    summary: dict = {
        "ellipse_id": ellipse_id,
        "site_dir": str(site_dir),
        "n_windows_merged": n_windows,
        "n_chunks_merged": n_chunks,
    }

    def stat_block(name: str) -> dict | None:
        try:
            mn = read_float32_array(stats / f"{name}_min.bin")
            av = read_float32_array(stats / f"{name}_avg.bin")
            mx = read_float32_array(stats / f"{name}_max.bin")
        except FileNotFoundError:
            return None
        finite = np.isfinite(av)
        return {
            "n_faces": int(av.size),
            "n_finite": int(finite.sum()),
            "min_min": float(np.nanmin(mn)) if finite.any() else None,
            "min_avg": float(np.nanmin(av[finite])) if finite.any() else None,
            "mean_avg": float(np.nanmean(av[finite])) if finite.any() else None,
            "max_avg": float(np.nanmax(av[finite])) if finite.any() else None,
            "max_max": float(np.nanmax(mx)) if finite.any() else None,
        }
    for s in ("temperature", "illumination"):
        block = stat_block(s)
        if block is not None:
            summary[s] = block

    # Ice stability scalars.
    ice_path = stats / "ice_depth.bin"
    if ice_path.is_file():
        ice = read_float32_array(ice_path)
        # Faces with FINITE positive stable depth (the rest are 0 = unstable).
        stable = ice[ice > 0]
        summary["ice_depth"] = {
            "n_faces": int(ice.size),
            "n_stable": int(stable.size),
            "fraction_stable": float(stable.size / ice.size) if ice.size else 0.0,
            "max_depth_m": float(stable.max()) if stable.size else None,
            "median_depth_m": float(np.median(stable)) if stable.size else None,
        }

    out = site_dir / "summary.json"
    out.write_text(json.dumps(summary, indent=2))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--site-dir", required=True, type=Path,
                   help="Per-site root, e.g. output/a3gt_v2/A03MM022/")
    p.add_argument("--merge-bin", type=Path, default=None,
                   help="Path to ptm_merge_chunks (default: auto-detect).")
    args = p.parse_args()

    if not args.site_dir.is_dir():
        sys.exit(f"site_dir not a directory: {args.site_dir}")

    merge_bin = args.merge_bin or find_merge_bin()
    chunk_dirs = discover_chunk_dirs(args.site_dir)
    if not chunk_dirs:
        sys.exit(f"No completed chunks found under {args.site_dir}; "
                 "run the orchestrator first.")

    # Each chunk_dir is .../<site>/<window>/<chunk_NNN>; the window dir is .parent.
    n_windows = len({c.parent.name for c in chunk_dirs})
    print(f"Site {args.site_dir.name}: merging "
          f"{len(chunk_dirs)} chunk(s) from {n_windows} window(s) "
          f"into {args.site_dir}/statistics/")

    cmd = [str(merge_bin), "--chunks", *(str(c) for c in chunk_dirs),
           "--output", str(args.site_dir)]
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        sys.exit(f"ptm_merge_chunks exit {ret.returncode}")

    summary_path = write_summary(args.site_dir, args.site_dir.name,
                                  n_windows=n_windows, n_chunks=len(chunk_dirs))
    print(f"\nSummary: {summary_path}")
    summary = json.loads(summary_path.read_text())
    if "temperature" in summary:
        t = summary["temperature"]
        print(f"  Temperature: mean of avg = {t.get('mean_avg'):.2f} K, "
              f"min of min = {t.get('min_min'):.2f} K, "
              f"max of max = {t.get('max_max'):.2f} K")
    if "ice_depth" in summary:
        i = summary["ice_depth"]
        print(f"  Ice depth:   {i['fraction_stable']*100:.2f}% stable, "
              f"median depth (where stable) = {i.get('median_depth_m')!s} m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
