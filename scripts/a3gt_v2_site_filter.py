#!/usr/bin/env python3
"""A3GT v2 — DEM-coverage filter for the candidate landing-sites CSV.

Reads the CUI candidate-sites CSV (ellipse_id, Region, x_m, y_m, lon_deg,
lat_deg) and a GeoTIFF DEM, and partitions sites into:
  - "in":      full ROI of radius `--roi-km` fits inside the DEM bounds.
  - "partial": site center is inside the DEM but the ROI extends past
               the DEM edge by `clip_m` (per affected axis).
  - "out":     site center is outside the DEM entirely.

Writes a JSON manifest the orchestrator and downstream tools consume.
The manifest mirrors the CSV row plus a `status` and per-axis clip info.

DEM bounds are derived from `gdalinfo -json` (no GDAL Python dep required).
The CSV's x_m/y_m are assumed to share the DEM's projection — for the
A3GT batch this is south-polar stereographic on a 1 737.4 km lunar
sphere (verified 2026-05-06 against `mm-sfs-dem.tif`).

Example:
    python3 scripts/a3gt_v2_site_filter.py \\
        --csv .../CUI_2026_a3gt_landing-sites.csv \\
        --dem .../mm-sfs-dem.tif \\
        --roi-km 5 \\
        --output output/a3gt_v2/site_manifest.json
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def gdal_extent(dem_path: Path) -> dict[str, float]:
    """Return DEM bounds in projected coordinates via `gdalinfo -json`."""
    proc = subprocess.run(
        ["gdalinfo", "-json", str(dem_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    info = json.loads(proc.stdout)
    cs = info["cornerCoordinates"]
    xs = [cs["upperLeft"][0], cs["upperRight"][0],
          cs["lowerLeft"][0], cs["lowerRight"][0]]
    ys = [cs["upperLeft"][1], cs["upperRight"][1],
          cs["lowerLeft"][1], cs["lowerRight"][1]]
    return {
        "x_min": min(xs), "x_max": max(xs),
        "y_min": min(ys), "y_max": max(ys),
        "size_px": info["size"],
        "resolution_m": info.get("geoTransform", [None, None])[1],
        "wkt": info.get("coordinateSystem", {}).get("wkt", ""),
    }


def classify_site(site: dict[str, str], extent: dict[str, float],
                  roi_m: float) -> dict[str, Any]:
    """Tag a site as in / partial / out and report per-axis clip in metres."""
    x = float(site["x_m"])
    y = float(site["y_m"])
    out = {
        "ellipse_id": site["ellipse_id"],
        "region": site["Region"],
        "x_m": x, "y_m": y,
        "lon_deg": float(site["lon_deg"]),
        "lat_deg": float(site["lat_deg"]),
        "clip_m": {},
    }

    # Distance from the site center to each DEM edge (positive = inside).
    d_left   = x - extent["x_min"]
    d_right  = extent["x_max"] - x
    d_bottom = y - extent["y_min"]
    d_top    = extent["y_max"] - y

    # If center outside DEM, status = out.
    if any(d < 0 for d in (d_left, d_right, d_bottom, d_top)):
        out["status"] = "out"
        return out

    # ROI clip per axis (negative = ROI extends past the DEM edge).
    clip_left   = roi_m - d_left
    clip_right  = roi_m - d_right
    clip_bottom = roi_m - d_bottom
    clip_top    = roi_m - d_top

    if (clip_left <= 0 and clip_right <= 0
            and clip_bottom <= 0 and clip_top <= 0):
        out["status"] = "in"
    else:
        out["status"] = "partial"
        if clip_left   > 0: out["clip_m"]["west"]  = clip_left
        if clip_right  > 0: out["clip_m"]["east"]  = clip_right
        if clip_bottom > 0: out["clip_m"]["south"] = clip_bottom
        if clip_top    > 0: out["clip_m"]["north"] = clip_top
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", required=True, type=Path,
                   help="CUI candidate-sites CSV.")
    p.add_argument("--dem", required=True, type=Path,
                   help="GeoTIFF DEM that the sites are filtered against.")
    p.add_argument("--roi-km", type=float, default=5.0,
                   help="ROI radius in km (default: 5).")
    p.add_argument("--output", required=True, type=Path,
                   help="Output JSON manifest path.")
    args = p.parse_args()

    if not args.csv.is_file():
        print(f"CSV not found: {args.csv}", file=sys.stderr); return 1
    if not args.dem.is_file():
        print(f"DEM not found: {args.dem}", file=sys.stderr); return 1

    extent = gdal_extent(args.dem)
    roi_m = args.roi_km * 1000.0

    print(f"DEM extent: x=[{extent['x_min']:.0f}, {extent['x_max']:.0f}] m, "
          f"y=[{extent['y_min']:.0f}, {extent['y_max']:.0f}] m, "
          f"res={extent.get('resolution_m')} m/px",
          file=sys.stderr)

    sites: list[dict[str, Any]] = []
    with args.csv.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            sites.append(classify_site(row, extent, roi_m))

    counts = {"in": 0, "partial": 0, "out": 0}
    for s in sites:
        counts[s["status"]] += 1

    manifest = {
        "csv": str(args.csv),
        "dem": str(args.dem),
        "roi_km": args.roi_km,
        "dem_extent": extent,
        "counts": counts,
        "sites": sites,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nWrote {args.output}")
    print(f"  in:      {counts['in']:3d}")
    print(f"  partial: {counts['partial']:3d}")
    print(f"  out:     {counts['out']:3d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
