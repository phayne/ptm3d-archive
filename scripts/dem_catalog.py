#!/usr/bin/env python3
"""Polar landing-site campaign (v2): DEM catalog builder.

Scans a tree of regional lunar DEMs (default the polar-site Shape-from-Shading set
under ``<DEM root>/<region>/``) and emits a machine catalog
(``dem_catalog.json``) plus a human spreadsheet (``dem_catalog.csv``). The
catalog is consumed by ``a3gt_v2_site_filter.py`` (catalog/association mode)
to pick the best DEM(s) per landing site and build per-site VRT mosaics.

One entry per DEM tile (directory-keyed; multi-tile regions such as
``nobile_rim_2/`` contribute every tile). Per entry we record projected
extent, native resolution, CRS (and whether it is the canonical lunar
south-polar-stereographic frame), whole-DEM elevation/valid statistics
(from the GDAL ``.aux.xml`` sidecar when present), an area-based
specificity rank, and QA flags.

Resolution and CRS are always read per-DEM (the set is a *mix* of 1 m/px
and 5 m/px products) — never assumed.

LOLA base prerequisite
----------------------
The catalog can carry one coarse south-polar LOLA "base" DEM used as the
gap-fill / extension-ring underlay in every per-site VRT. It is staged
*manually, once* (recommended source: PGDA products/78 south-pole mosaic,
Barker et al. 2021, or an imbrium.mit.edu LDEM — confirm the exact product
before a production batch). It MUST be normalized into the canonical CRS
once:

    gdalwarp -t_srs '+proj=stere +lat_0=-90 +lon_0=0 +R=1737400 +units=m' \\
        -r bilinear -dstnodata -3.4028235e+38 \\
        -co TILED=YES -co BIGTIFF=IF_SAFER \\
        <staged_lola>.tif \\
        data/dem/a3gt_v2/_lola_base_spstereo.tif

Then pass ``--lola-base data/dem/a3gt_v2/_lola_base_spstereo.tif``. If the
flag is omitted the catalog records ``lola_base.present = false`` and the
extension ring is limited to SfS coverage.

Example
-------
    python3 scripts/dem_catalog.py \\
        --dem-root /path/to/polar_sfs_dems \\
        --lola-base data/dem/a3gt_v2/_lola_base_spstereo.tif \\
        --out-json output/a3gt_v2/dem_catalog.json \\
        --out-csv  output/a3gt_v2/dem_catalog.csv
"""

from __future__ import annotations

import argparse
import csv as _csv
import datetime as _dt
import json
import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

# Reuse the single source of truth for projected-extent extraction.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from a3gt_v2_site_filter import gdal_extent  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DEFAULT_DEM_ROOT = Path(os.environ.get("PTM_DEM_ROOT", str(REPO / "data" / "dem")))

# Canonical lunar south-polar-stereographic frame shared by every polar-site SfS DEM
# and required for the per-site VRT mosaic (the C++ mesh tool converts the ROI
# center via the dataset's own polar-stereo params, so a mixed-CRS VRT breaks).
CANON = {"proj": "stere", "lat_0": -90.0, "lon_0": 0.0, "R": 1737400.0}
CANON_TOL = 1e-6

# DEM file selection: a regional dir's elevation product is "<stem>sfs<stem>dem".
# Exclude the non-elevation companions (hillshade, valid-count, weight, error,
# resolution and the *-map products) that some SfS deliveries ship alongside.
_INCLUDE_RE = re.compile(r"sfs.*dem.*\.tif$", re.I)
_EXCLUDE_RE = re.compile(
    r"(hs|hillshade|count|weight|error|stddev|resolution|-map|maxlit|max-lit)",
    re.I)

_BIG_PIXELS = 1_000_000_000  # size_px[0]*size_px[1] above this -> "large_dem"


def _tool(name: str) -> str:
    """Prefer the Homebrew GDAL >=3.12 (matches the COG pipeline), else PATH."""
    cand = f"/usr/local/bin/{name}"
    return cand if os.path.exists(cand) else (shutil.which(name) or name)


def find_dem_tifs(root: Path) -> list[Path]:
    """Return every SfS elevation .tif under ``root`` (maxdepth 2), sorted.

    Multi-tile regions (e.g. ``nobile_rim_2/`` with nr2 + dm2-box{1,2,3})
    contribute one path per tile.
    """
    out: list[Path] = []
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        for tif in sorted(sub.glob("*.tif")):
            name = tif.name
            if _INCLUDE_RE.search(name) and not _EXCLUDE_RE.search(name):
                out.append(tif)
    return out


def read_pam_stats(tif: Path) -> dict[str, Any]:
    """Parse the GDAL ``<tif>.aux.xml`` PAM sidecar for band-1 statistics.

    Returns ``{}``-valued fields (None) when the sidecar is absent so the
    catalog still lists the DEM (flagged ``aux_xml_missing``).
    """
    keys = {
        "STATISTICS_VALID_PERCENT": "valid_percent",
        "STATISTICS_MINIMUM": "elev_min",
        "STATISTICS_MAXIMUM": "elev_max",
        "STATISTICS_MEAN": "elev_mean",
        "STATISTICS_STDDEV": "elev_stddev",
        "STATISTICS_APPROXIMATE": "approximate",
    }
    res: dict[str, Any] = {v: None for v in keys.values()}
    aux = tif.with_suffix(tif.suffix + ".aux.xml")
    if not aux.is_file():
        res["_aux_present"] = False
        return res
    res["_aux_present"] = True
    try:
        root = ET.parse(aux).getroot()
        for mdi in root.iter("MDI"):
            k = mdi.get("key", "")
            if k in keys:
                val = (mdi.text or "").strip()
                if keys[k] == "approximate":
                    res["approximate"] = (val.upper() == "YES")
                else:
                    try:
                        res[keys[k]] = float(val)
                    except ValueError:
                        res[keys[k]] = None
    except ET.ParseError:
        res["_aux_present"] = False
    return res


def proj_params(tif: Path) -> dict[str, Any]:
    """Extract proj4-derived CRS params via ``gdalsrsinfo -o proj4``.

    Returns ``{proj, lat_0, lon_0, R, kind, is_canonical, proj4}``.
    ``kind`` is ``polar_stereo`` | ``equirectangular`` | ``other``.
    """
    try:
        p = subprocess.run([_tool("gdalsrsinfo"), "-o", "proj4", str(tif)],
                            capture_output=True, text=True, check=True)
        proj4 = p.stdout.strip().strip("'").strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        proj4 = ""

    def _num(flag: str) -> float | None:
        m = re.search(rf"\+{flag}=([-+0-9.eE]+)", proj4)
        return float(m.group(1)) if m else None

    pm = re.search(r"\+proj=(\w+)", proj4)
    proj = pm.group(1) if pm else ""
    lat0, lon0, rad = _num("lat_0"), _num("lon_0"), _num("R")
    kind = ("polar_stereo" if proj == "stere"
            else "equirectangular" if proj in ("eqc", "longlat")
            else "other")
    is_canon = (
        proj == CANON["proj"]
        and lat0 is not None and abs(lat0 - CANON["lat_0"]) < CANON_TOL
        and lon0 is not None and abs(lon0 - CANON["lon_0"]) < CANON_TOL
        and rad is not None and abs(rad - CANON["R"]) < 1.0
    )
    return {"proj": proj, "lat_0": lat0, "lon_0": lon0, "R": rad,
            "kind": kind, "is_canonical": bool(is_canon), "proj4": proj4}


def _entry(tif: Path, root: Path) -> dict[str, Any]:
    ext = gdal_extent(tif)              # x_min/max, y_min/max, size_px, resolution_m, wkt
    crs = proj_params(tif)
    pam = read_pam_stats(tif)

    region = tif.parent.name
    siblings = sorted(tif.parent.glob("*.tif"))
    multi = len([s for s in siblings
                 if _INCLUDE_RE.search(s.name)
                 and not _EXCLUDE_RE.search(s.name)]) > 1
    tile = tif.stem if multi else None

    size_px = ext.get("size_px") or [0, 0]
    res_m = abs(ext.get("resolution_m") or 0.0)
    width_m = ext["x_max"] - ext["x_min"]
    height_m = ext["y_max"] - ext["y_min"]
    area_km2 = round(width_m * height_m / 1e6, 4)

    flags: list[str] = []
    if not crs["is_canonical"]:
        flags.append("crs_noncanonical")
    if crs["kind"] == "equirectangular":
        flags.append("equirectangular")
    if not pam.get("_aux_present", False):
        flags.append("aux_xml_missing")
    if multi:
        flags.append("multi_tile_region")
    if size_px and size_px[0] * size_px[1] > _BIG_PIXELS:
        flags.append("large_dem")

    rel = os.path.relpath(tif, root)
    eid = region if not multi else f"{region}/{tif.stem}"
    return {
        "id": eid,
        "region": region,
        "tile": tile,
        "path": str(tif),
        "rel_path": rel,
        "role": "sfs",
        "resolution_m": res_m,
        "extent_m": {"x_min": ext["x_min"], "x_max": ext["x_max"],
                     "y_min": ext["y_min"], "y_max": ext["y_max"]},
        "size_px": size_px,
        "area_km2": area_km2,
        "crs": {"proj": crs["proj"], "lat_0": crs["lat_0"],
                "lon_0": crs["lon_0"], "R": crs["R"],
                "kind": crs["kind"], "is_canonical": crs["is_canonical"]},
        "stats": {"valid_percent": pam["valid_percent"],
                  "elev_min": pam["elev_min"], "elev_max": pam["elev_max"],
                  "elev_mean": pam["elev_mean"],
                  "elev_stddev": pam["elev_stddev"],
                  "approximate": pam.get("approximate")},
        "aux_xml": str(tif.with_suffix(tif.suffix + ".aux.xml"))
        if pam.get("_aux_present") else None,
        "readme": str(tif.parent / "README.txt")
        if (tif.parent / "README.txt").is_file() else None,
        "specificity_rank": None,   # filled after all entries known
        "flags": flags,
    }


def _lola_entry(lola_path: Path | None) -> dict[str, Any]:
    if lola_path is None:
        return {"id": "lola_base", "role": "base", "present": False,
                "path": None, "resolution_m": None,
                "note": "PGDA products/78 (Barker 2021) / imbrium LDEM; stage "
                        "and gdalwarp to canonical CRS once before a batch."}
    p = lola_path.expanduser()
    if not p.is_file():
        return {"id": "lola_base", "role": "base", "present": False,
                "path": str(p), "resolution_m": None,
                "note": f"--lola-base given but file not found: {p}"}
    ext = gdal_extent(p)
    crs = proj_params(p)
    e = {"id": "lola_base", "role": "base", "present": True, "path": str(p),
         "resolution_m": abs(ext.get("resolution_m") or 0.0),
         "extent_m": {"x_min": ext["x_min"], "x_max": ext["x_max"],
                      "y_min": ext["y_min"], "y_max": ext["y_max"]},
         "size_px": ext.get("size_px"),
         "crs": {"proj": crs["proj"], "lat_0": crs["lat_0"],
                 "lon_0": crs["lon_0"], "R": crs["R"],
                 "kind": crs["kind"], "is_canonical": crs["is_canonical"]},
         "flags": [] if crs["is_canonical"] else ["crs_noncanonical"]}
    if not crs["is_canonical"]:
        e["note"] = ("LOLA base is NOT in the canonical south-polar-stereo "
                     "frame; gdalwarp it (see module docstring) before use.")
    return e


def build_catalog(root: Path, lola_base: Path | None = None,
                  extra_roots: list[Path] | None = None) -> dict[str, Any]:
    tifs = find_dem_tifs(root)
    for er in extra_roots or []:
        if er.is_dir():
            tifs += find_dem_tifs(er)

    dems = [_entry(t, root) for t in tifs]

    # specificity_rank: ascending area (smaller = more localized/specific).
    for rank, e in enumerate(sorted(dems, key=lambda d: d["area_km2"])):
        e["specificity_rank"] = rank

    return {
        "generated_utc": _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "dem_root": str(root),
        "extra_roots": [str(x) for x in (extra_roots or [])],
        "canonical_crs": CANON,
        "lola_base": _lola_entry(lola_base),
        "count": len(dems),
        "dems": dems,
    }


_CSV_COLS = ["id", "region", "tile", "path", "role", "resolution_m",
             "x_min", "x_max", "y_min", "y_max", "area_km2",
             "valid_percent", "elev_min", "elev_max",
             "crs_canonical", "kind", "specificity_rank", "flags"]


def write_json(cat: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cat, indent=2))


def write_csv(cat: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(_CSV_COLS)
        for d in cat["dems"]:
            ex = d["extent_m"]
            w.writerow([
                d["id"], d["region"], d["tile"] or "", d["path"], d["role"],
                d["resolution_m"], ex["x_min"], ex["x_max"],
                ex["y_min"], ex["y_max"], d["area_km2"],
                d["stats"]["valid_percent"], d["stats"]["elev_min"],
                d["stats"]["elev_max"], d["crs"]["is_canonical"],
                d["crs"]["kind"], d["specificity_rank"],
                "|".join(d["flags"]),
            ])


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dem-root", type=Path, default=DEFAULT_DEM_ROOT,
                   help=f"DEM tree to scan (default: {DEFAULT_DEM_ROOT}).")
    p.add_argument("--extra-roots", type=Path, nargs="*", default=[],
                   help="Additional DEM trees (e.g. malapert/, "
                        "compton-belkovitch/). Non-canonical DEMs are "
                        "cataloged but flagged and excluded from VRTs.")
    p.add_argument("--lola-base", type=Path, default=None,
                   help="Optional coarse south-polar LOLA base DEM "
                        "(canonical CRS — see module docstring).")
    p.add_argument("--out-json", type=Path,
                   default=REPO / "output" / "a3gt_v2" / "dem_catalog.json")
    p.add_argument("--out-csv", type=Path,
                   default=REPO / "output" / "a3gt_v2" / "dem_catalog.csv")
    args = p.parse_args()

    if not args.dem_root.is_dir():
        print(f"DEM root not found: {args.dem_root}", file=sys.stderr)
        return 1

    cat = build_catalog(args.dem_root, args.lola_base, args.extra_roots)
    write_json(cat, args.out_json)
    write_csv(cat, args.out_csv)

    n_canon = sum(1 for d in cat["dems"] if d["crs"]["is_canonical"])
    n_flag = sum(1 for d in cat["dems"] if d["flags"])
    lb = cat["lola_base"]
    print(f"Cataloged {cat['count']} DEM(s) from {args.dem_root}")
    print(f"  canonical CRS: {n_canon}/{cat['count']}")
    print(f"  with QA flags: {n_flag}")
    print(f"  LOLA base:     "
          f"{'present ' + str(lb.get('path')) if lb.get('present') else 'absent'}")
    print(f"  -> {args.out_json}")
    print(f"  -> {args.out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
