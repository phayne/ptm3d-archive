#!/usr/bin/env python3
"""A3GT v2 — DEM-coverage filter for the candidate landing-sites CSV.

Reads the candidate-sites CSV (ellipse_id, Region, x_m, y_m, lon_deg,
lat_deg) and a GeoTIFF DEM, and partitions sites into:
  - "in":      full ROI of radius `--roi-km` fits inside the DEM bounds.
  - "partial": site center is inside the DEM but the ROI extends past
               the DEM edge by `clip_m` (per affected axis).
  - "out":     site center is outside the DEM entirely.

Writes a JSON manifest the orchestrator and downstream tools consume.
The manifest mirrors the CSV row plus a `status` and per-axis clip info.

Two modes:
  * Legacy single-DEM (``--dem PATH``): unchanged — one DEM, in/partial/out.
  * Catalog/association (``--catalog dem_catalog.json``): per site, picks the
    best-available DEM(s) (resolution -> ROI valid% -> specificity, plus an
    optional ``--overrides`` CSV), records the mosaic members + per-site VRT
    path, and classifies coverage for both ``--roi-km`` and
    ``--extended-roi-km``. Sites whose center no canonical DEM covers are
    flagged ``skip`` (graceful — the orchestrator excludes them, no abort).
    The manifest stays backward-compatible (all legacy keys retained;
    ``status``/``clip_m`` reflect the mosaic ROI-scope result).

DEM bounds are derived from `gdalinfo -json` (no GDAL Python dep required).
The CSV's x_m/y_m are assumed to share the DEM's projection — for the
A3GT batch this is south-polar stereographic on a 1 737.4 km lunar
sphere (verified 2026-05-06 against `mm-sfs-dem.tif`).

Examples:
    # legacy single-DEM
    python3 scripts/a3gt_v2_site_filter.py \\
        --csv <candidate-sites>.csv \\
        --dem .../mm-sfs-dem.tif --roi-km 5 \\
        --output output/a3gt_v2/site_manifest.json

    # catalog / multi-DEM mosaic + extended-ROI coverage
    python3 scripts/a3gt_v2_site_filter.py \\
        --csv <candidate-sites>.csv \\
        --catalog output/a3gt_v2/dem_catalog.json \\
        --roi-km 3 --extended-roi-km 15 \\
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


def coverage_status(x: float, y: float, extent: dict[str, float],
                    roi_m: float) -> tuple[str, dict[str, float]]:
    """Classify a point against one DEM extent.

    Returns ``(status, clip_m)`` where status is ``out`` (center outside the
    DEM), ``in`` (ROI of radius ``roi_m`` fits entirely inside), or
    ``partial`` (center inside but the ROI clips one or more DEM edges).
    ``clip_m`` maps the clipped axes (west/east/south/north) to the overshoot
    in meters (empty unless status == ``partial``).

    This is the geometry primitive shared by ``classify_site`` (single-DEM
    legacy mode) and the catalog/association mode. Behavior is identical to
    the pre-refactor inline logic — see the /tmp regression gate.
    """
    clip_m: dict[str, float] = {}

    # Distance from the point to each DEM edge (positive = inside).
    d_left   = x - extent["x_min"]
    d_right  = extent["x_max"] - x
    d_bottom = y - extent["y_min"]
    d_top    = extent["y_max"] - y

    # If center outside DEM, status = out.
    if any(d < 0 for d in (d_left, d_right, d_bottom, d_top)):
        return "out", clip_m

    # ROI clip per axis (negative = ROI extends past the DEM edge).
    clip_left   = roi_m - d_left
    clip_right  = roi_m - d_right
    clip_bottom = roi_m - d_bottom
    clip_top    = roi_m - d_top

    if (clip_left <= 0 and clip_right <= 0
            and clip_bottom <= 0 and clip_top <= 0):
        return "in", clip_m

    if clip_left   > 0: clip_m["west"]  = clip_left
    if clip_right  > 0: clip_m["east"]  = clip_right
    if clip_bottom > 0: clip_m["south"] = clip_bottom
    if clip_top    > 0: clip_m["north"] = clip_top
    return "partial", clip_m


def classify_site(site: dict[str, str], extent: dict[str, float],
                  roi_m: float) -> dict[str, Any]:
    """Tag a site as in / partial / out and report per-axis clip in meters."""
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
    status, clip_m = coverage_status(x, y, extent, roi_m)
    out["status"] = status
    out["clip_m"] = clip_m
    return out


# ---------------------------------------------------------------------------
# Catalog / association mode (multi-DEM, best-DEM selection, VRT mosaic)
# ---------------------------------------------------------------------------

REPO = Path("/Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/europa/heat3d/ptm")


def load_catalog(path: Path) -> dict[str, Any]:
    """Load the dem_catalog.json produced by scripts/dem_catalog.py."""
    return json.loads(Path(path).read_text())


def load_overrides(path: Path | None) -> dict[str, str]:
    """Optional per-site override CSV (columns: ellipse_id,dem_id)."""
    if not path:
        return {}
    out: dict[str, str] = {}
    with Path(path).open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row.get("ellipse_id") and row.get("dem_id"):
                out[row["ellipse_id"].strip()] = row["dem_id"].strip()
    return out


def _usable(dem: dict[str, Any]) -> bool:
    """A DEM that can enter a per-site VRT (canonical CRS, not equirect)."""
    return bool(dem.get("crs", {}).get("is_canonical")) and \
        "equirectangular" not in dem.get("flags", [])


def _bbox_intersects(extent: dict[str, float], x: float, y: float,
                     r_m: float) -> bool:
    """Does the axis-aligned ROI bbox [x±r, y±r] overlap the DEM rectangle?

    Used for mosaic membership: a DEM contributes distant-topography even if
    it does not contain the site center (e.g. a massif in an adjacent tile).
    """
    return not (x + r_m < extent["x_min"] or x - r_m > extent["x_max"]
                or y + r_m < extent["y_min"] or y - r_m > extent["y_max"])


def covering_dems(x: float, y: float, catalog: dict[str, Any],
                  r_m: float) -> tuple[list[dict[str, Any]],
                                       list[tuple[str, str]]]:
    """Usable catalog DEMs whose extent intersects the radius-r ROI bbox.

    Returns (members, dropped) where dropped is (id, reason) for DEMs that
    would have covered but are non-canonical / equirectangular.
    """
    members, dropped = [], []
    for d in catalog.get("dems", []):
        if not _bbox_intersects(d["extent_m"], x, y, r_m):
            continue
        if _usable(d):
            members.append(d)
        else:
            crs = d.get("crs", {})
            dropped.append((d["id"],
                            f"non-canonical CRS (proj={crs.get('proj')}, "
                            f"lat_0={crs.get('lat_0')}, lon_0={crs.get('lon_0')})"))
    return members, dropped


def rank_dems(members: list[dict[str, Any]],
              override_id: str | None) -> list[dict[str, Any]]:
    """Order best-first: override -> resolution asc -> valid% desc ->
    specificity asc -> id (stable, deterministic)."""
    def key(d: dict[str, Any]):
        vp = d["stats"].get("valid_percent")
        return (
            0 if (override_id and d["id"] == override_id) else 1,
            d.get("resolution_m") or 9e9,
            -(vp if vp is not None else -1.0),
            d.get("specificity_rank") if d.get("specificity_rank")
            is not None else 9999,
            d["id"],
        )
    return sorted(members, key=key)


def _union_extent(members: list[dict[str, Any]],
                  lola: dict[str, Any] | None) -> dict[str, float]:
    exts = [m["extent_m"] for m in members]
    if lola and lola.get("present") and _usable(lola) and "extent_m" in lola:
        exts.append(lola["extent_m"])
    return {
        "x_min": min(e["x_min"] for e in exts),
        "x_max": max(e["x_max"] for e in exts),
        "y_min": min(e["y_min"] for e in exts),
        "y_max": max(e["y_max"] for e in exts),
    }


def _scope_status(x: float, y: float, members: list[dict[str, Any]],
                  lola: dict[str, Any] | None, r_m: float
                  ) -> tuple[str, dict[str, float], int]:
    """Coverage of a radius-r ROI by the mosaic. 'in' if a single member
    contains it, else evaluated against the union bbox. Returns
    (status, clip_m, n_contributing)."""
    contributing = [m for m in members
                    if _bbox_intersects(m["extent_m"], x, y, r_m)]
    n = len(contributing) + (1 if (lola and lola.get("present")
                                   and _usable(lola)) else 0)
    for m in contributing:
        st, clip = coverage_status(x, y, m["extent_m"], r_m)
        if st == "in":
            return "in", {}, n
    if not contributing and not (lola and lola.get("present")
                                 and _usable(lola)):
        return "out", {}, 0
    st, clip = coverage_status(x, y, _union_extent(contributing, lola), r_m)
    return st, clip, n


def _inscribed_radius_km(x: float, y: float,
                         extent: dict[str, float]) -> float:
    """Largest fully-covered radius (km) from (x,y) to the nearest bbox edge."""
    d = min(x - extent["x_min"], extent["x_max"] - x,
            y - extent["y_min"], extent["y_max"] - y)
    return max(0.0, d) / 1000.0


def site_coverage(site: dict[str, str], catalog: dict[str, Any],
                  roi_km: float, ext_km: float,
                  overrides: dict[str, str],
                  vrt_dir: Path) -> dict[str, Any]:
    """Associate one site with its best DEM(s); emit the augmented record.

    Backward-compatible: keeps ellipse_id/region/x_m/y_m/lon_deg/lat_deg/
    clip_m/status (status/clip_m = mosaic ROI-scope result; "out" when the
    site is skipped) so the orchestrator's filter_sites() is unchanged.
    """
    x, y = float(site["x_m"]), float(site["y_m"])
    eid = site["ellipse_id"]
    roi_m, ext_m = roi_km * 1000.0, ext_km * 1000.0
    override_id = overrides.get(eid)

    # Members chosen against the EXTENDED radius (distant topography matters).
    members, dropped = covering_dems(x, y, catalog, ext_m)
    ranked = rank_dems(members, override_id)
    lola = catalog.get("lola_base")
    lola_usable = bool(lola and lola.get("present") and _usable(lola))

    roi_st, roi_clip, roi_n = _scope_status(x, y, ranked, lola, roi_m)
    ext_st, ext_clip, ext_n = _scope_status(x, y, ranked, lola, ext_m)

    # Skip only when NOTHING covers the site center (no usable DEM, no base).
    center_in = any(coverage_status(x, y, m["extent_m"], 0.0)[0] != "out"
                    for m in ranked) or lola_usable
    skip = not center_in
    skip_reason = None
    if skip:
        skip_reason = (
            f"site center not covered by any canonical DEM"
            + (f" (dropped non-canonical: {[d for d,_ in dropped]})"
               if dropped else "")
            + ("; no LOLA base staged" if not lola_usable else ""))

    out: dict[str, Any] = {
        "ellipse_id": eid,
        "region": site["Region"],
        "x_m": x, "y_m": y,
        "lon_deg": float(site["lon_deg"]),
        "lat_deg": float(site["lat_deg"]),
        "clip_m": {} if skip else roi_clip,
        "status": "out" if skip else roi_st,
    }

    dems_rec: list[dict[str, Any]] = []
    for i, m in enumerate(ranked):
        mst, mclip = coverage_status(x, y, m["extent_m"], roi_m)
        dems_rec.append({
            "id": m["id"], "path": m["path"],
            "role": "primary" if i == 0 else "mosaic",
            "resolution_m": m["resolution_m"],
            "status": mst, "clip_m": mclip,
            "valid_percent": m["stats"].get("valid_percent"),
            "specificity_rank": m.get("specificity_rank"),
        })
    if lola_usable:
        dems_rec.append({"id": lola.get("id", "lola_base"),
                         "path": lola.get("path"), "role": "base",
                         "resolution_m": lola.get("resolution_m")})

    # relief proxy + max covered radius for the extended-mesh sizing (Task 5).
    elev_max = [m["stats"]["elev_max"] for m in ranked
                if m["stats"].get("elev_max") is not None]
    elev_min = [m["stats"]["elev_min"] for m in ranked
                if m["stats"].get("elev_min") is not None]
    relief_max_m = (round(max(elev_max) - min(elev_min), 2)
                    if elev_max and elev_min else None)
    if ranked or lola_usable:
        max_cov_km = _inscribed_radius_km(
            x, y, _union_extent(ranked, lola if lola_usable else None))
    else:
        max_cov_km = 0.0

    vrt_abs = Path(vrt_dir) / f"{eid}_R{ext_km:g}km.vrt"
    try:
        vrt_rel = str(vrt_abs.relative_to(REPO))
    except ValueError:
        vrt_rel = str(vrt_abs)

    out.update({
        "dems": dems_rec,
        "primary_dem": ranked[0]["id"] if ranked else None,
        "dropped_dems": [{"id": i, "reason": r} for i, r in dropped],
        "vrt_path": None if skip else vrt_rel,
        "coverage": {
            "roi": {"radius_km": roi_km, "status": roi_st,
                    "clip_m": roi_clip, "n_dems": roi_n},
            "extended": {"radius_km": ext_km, "status": ext_st,
                         "clip_m": ext_clip, "n_dems": ext_n,
                         "max_covered_radius_km": round(max_cov_km, 3),
                         "relief_max_m": relief_max_m},
        },
        "override_dem": override_id,
        "skip": skip,
        "skip_reason": skip_reason,
    })
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", required=True, type=Path,
                   help="Candidate-sites CSV.")
    p.add_argument("--dem", type=Path,
                   help="Single GeoTIFF DEM (legacy mode; not needed with "
                        "--catalog).")
    p.add_argument("--catalog", type=Path,
                   help="dem_catalog.json -> enables multi-DEM catalog/"
                        "association mode.")
    p.add_argument("--roi-km", type=float, default=5.0,
                   help="ROI radius in km (default: 5).")
    p.add_argument("--extended-roi-km", type=float, default=None,
                   help="Extended (shadow-ring) radius in km for mosaic "
                        "member selection / coverage. Default = --roi-km.")
    p.add_argument("--overrides", type=Path, default=None,
                   help="Optional per-site override CSV (ellipse_id,dem_id).")
    p.add_argument("--vrt-dir", type=Path,
                   default=REPO / "data" / "dem" / "a3gt_v2",
                   help="Directory for per-site .vrt mosaics.")
    p.add_argument("--build-vrt", action="store_true",
                   help="Also materialize each non-skipped site's .vrt now "
                        "(otherwise the orchestrator builds it lazily).")
    p.add_argument("--output", required=True, type=Path,
                   help="Output JSON manifest path.")
    args = p.parse_args()

    if not args.csv.is_file():
        print(f"CSV not found: {args.csv}", file=sys.stderr); return 1

    # ---- Legacy single-DEM mode (unchanged; byte-identical manifest) ----
    if not args.catalog:
        if not args.dem or not args.dem.is_file():
            print("legacy mode needs --dem PATH (or pass --catalog)",
                  file=sys.stderr); return 1
        extent = gdal_extent(args.dem)
        roi_m = args.roi_km * 1000.0
        print(f"DEM extent: x=[{extent['x_min']:.0f}, {extent['x_max']:.0f}] m, "
              f"y=[{extent['y_min']:.0f}, {extent['y_max']:.0f}] m, "
              f"res={extent.get('resolution_m')} m/px", file=sys.stderr)
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
        for k in ("in", "partial", "out"):
            print(f"  {k+':':9s}{counts[k]:3d}")
        return 0

    # ---- Catalog / association mode ----
    catalog = load_catalog(args.catalog)
    overrides = load_overrides(args.overrides)
    ext_km = args.extended_roi_km if args.extended_roi_km else args.roi_km

    sites = []
    with args.csv.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            sites.append(site_coverage(row, catalog, args.roi_km, ext_km,
                                       overrides, args.vrt_dir))

    counts = {"in": 0, "partial": 0, "out": 0}
    for s in sites:
        counts[s["status"]] += 1
    skipped = [s["ellipse_id"] for s in sites if s["skip"]]
    mosaic = [s["ellipse_id"] for s in sites
              if not s["skip"] and len(s.get("dems", [])) > 1]

    manifest = {
        "csv": str(args.csv),
        "dem": "<per-site VRT>",
        "dem_mode": "catalog",
        "dem_catalog": str(args.catalog),
        "roi_km": args.roi_km,
        "extended_roi_km": ext_km,
        "vrt_dir": str(args.vrt_dir),
        "counts": counts,
        "n_skipped": len(skipped),
        "n_mosaic": len(mosaic),
        "sites": sites,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(manifest, f, indent=2)

    if args.build_vrt:
        from dem_vrt import build_site_vrt
        cat_by_id = {d["id"]: d for d in catalog["dems"]}
        for s in sites:
            if s["skip"]:
                continue
            ranked = [cat_by_id[d["id"]] for d in s["dems"]
                      if d["role"] != "base"]
            try:
                build_site_vrt(ranked, catalog.get("lola_base"),
                               REPO / s["vrt_path"], verbose=False)
            except (ValueError, RuntimeError) as e:
                print(f"  VRT {s['ellipse_id']}: {e}", file=sys.stderr)

    print(f"\nWrote {args.output}  (catalog mode, ext_roi={ext_km:g} km)")
    for k in ("in", "partial", "out"):
        print(f"  {k+':':9s}{counts[k]:3d}")
    print(f"  skipped: {len(skipped)}  {skipped if skipped else ''}")
    print(f"  mosaic : {len(mosaic)} site(s) need >1 DEM")
    return 0


if __name__ == "__main__":
    sys.exit(main())
