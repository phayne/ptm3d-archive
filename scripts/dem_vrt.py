#!/usr/bin/env python3
"""A3GT v2 — per-site DEM VRT mosaic builder.

Given a site's resolution-ranked list of covering DEMs (from
``a3gt_v2_site_filter.py`` catalog mode) plus the optional coarse LOLA base,
build a GDAL ``.vrt`` the C++ mesh tool (``dtm_to_ply_delaunay``) can open
transparently — so multi-DEM mosaicking needs **zero** C++ change.

Layering contract
-----------------
``gdalbuildvrt`` is *last-wins on overlap*. To get "finest SfS on top,
coarse LOLA underneath", inputs are passed in **reverse priority order**:

    [ lola_base, coarsest_sfs, ... , finest_sfs ]   (finest is the last arg)

``ranked`` arrives best/finest-first (the rank_dems order); we reverse it.

CRS contract (critical)
-----------------------
The C++ tool converts the ROI center to projected XY using the *dataset's
own* polar-stereo params. A VRT mixing CRSs, or defaulting the CRS, would
corrupt the ROI math. Therefore only DEMs in the canonical
``+proj=stere +lat_0=-90 +lon_0=0 +R=1737400`` frame are admitted; any
``crs_noncanonical`` / ``equirectangular`` member is dropped and logged
(never silently reprojected into the mosaic). A non-canonical DEM can be
recovered with the same one-time ``gdalwarp`` recipe documented in
``dem_catalog.py`` for the LOLA base.

NoData
------
``-srcnodata/-vrtnodata = -3.4028235e+38`` (the SfS float fill) makes
``gdalbuildvrt`` emit ``<ComplexSource><NODATA>`` so a finer tile's holes
fall through to the coarser underlay, and the C++ NoData skip keeps working.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path("/Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/europa/heat3d/ptm")
DEFAULT_VRT_DIR = REPO / "data" / "dem" / "a3gt_v2"
SFS_NODATA = "-3.4028235e+38"


def _tool(name: str) -> str:
    cand = f"/usr/local/bin/{name}"
    return cand if os.path.exists(cand) else (shutil.which(name) or name)


def vrt_path_for(ellipse_id: str, roi_km: float,
                 vrt_dir: Path = DEFAULT_VRT_DIR) -> Path:
    """Stable per-site VRT path: ``<vrt_dir>/<id>_R{roi:g}km.vrt``."""
    return Path(vrt_dir) / f"{ellipse_id}_R{roi_km:g}km.vrt"


def _is_canonical(dem: dict[str, Any]) -> bool:
    crs = dem.get("crs", {})
    return bool(crs.get("is_canonical")) and "equirectangular" not in dem.get(
        "flags", [])


def admissible_members(ranked: list[dict[str, Any]],
                       lola_base: dict[str, Any] | None
                       ) -> tuple[list[dict[str, Any]], dict[str, Any] | None,
                                  list[tuple[str, str]]]:
    """Split ``ranked`` into canonical members + dropped (id, reason).

    Returns ``(members_best_first, lola_or_None, dropped)``.
    """
    members, dropped = [], []
    for d in ranked:
        if _is_canonical(d):
            members.append(d)
        else:
            crs = d.get("crs", {})
            dropped.append((d.get("id", "?"),
                            f"non-canonical CRS proj={crs.get('proj')} "
                            f"lat_0={crs.get('lat_0')} lon_0={crs.get('lon_0')}"
                            " — excluded from VRT (gdalwarp-normalize to use)"))
    lb = None
    if lola_base and lola_base.get("present"):
        if _is_canonical(lola_base):
            lb = lola_base
        else:
            dropped.append((lola_base.get("id", "lola_base"),
                            "LOLA base not in canonical CRS — gdalwarp it "
                            "(see dem_catalog.py) before use"))
    return members, lb, dropped


def build_site_vrt(ranked: list[dict[str, Any]],
                   lola_base: dict[str, Any] | None,
                   out_vrt: Path,
                   target_extent_m: dict[str, float] | None = None,
                   *, dry_run: bool = False,
                   verbose: bool = True) -> dict[str, Any]:
    """Build (or describe, if dry_run) a per-site VRT mosaic.

    ``ranked``     best/finest-first list of catalog DEM entries.
    ``lola_base``  catalog ``lola_base`` dict or None.
    Returns a result dict: ``{vrt, n_members, members, dropped, cmd, built}``.
    Raises ``ValueError`` if no canonical member is available (caller should
    treat the site as a graceful skip).
    """
    members, lb, dropped = admissible_members(ranked, lola_base)
    if verbose:
        for did, why in dropped:
            print(f"  [dem_vrt] drop {did}: {why}", file=sys.stderr)
    if not members and lb is None:
        raise ValueError(
            "no canonical-CRS DEM available for this site "
            f"(dropped: {[d for d, _ in dropped]})")
    if not members and verbose:
        print("  [dem_vrt] LOLA-base-only site (no SfS coverage at center)",
              file=sys.stderr)

    # Reverse-priority order: base first (lowest), finest SfS last (wins).
    inputs: list[str] = []
    if lb is not None:
        inputs.append(lb["path"])
    inputs += [m["path"] for m in reversed(members)]   # coarsest..finest

    out_vrt = Path(out_vrt)
    cmd = [_tool("gdalbuildvrt"), "-overwrite",
           "-resolution", "highest",
           "-r", "bilinear",
           "-srcnodata", SFS_NODATA, "-vrtnodata", SFS_NODATA,
           "-allow_projection_difference"]
    if target_extent_m:
        te = target_extent_m
        cmd += ["-te", f"{te['x_min']}", f"{te['y_min']}",
                f"{te['x_max']}", f"{te['y_max']}"]
    cmd += [str(out_vrt), *inputs]

    result = {
        "vrt": str(out_vrt),
        "n_members": len(members) + (1 if lb else 0),
        "members": [m["id"] for m in members],
        "lola_base": lb["id"] if lb else None,
        "dropped": dropped,
        "cmd": cmd,
        "built": False,
    }
    if verbose:
        finest = (f"finest={members[0]['id']} @ {members[0]['resolution_m']} m"
                  if members else "LOLA-base only")
        print(f"  [dem_vrt] {out_vrt.name}: "
              f"{'lola+' if lb else ''}{len(members)} SfS ({finest})")
    if dry_run:
        print("  [dem_vrt] DRY-RUN cmd: " + " ".join(cmd), file=sys.stderr)
        return result

    out_vrt.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"gdalbuildvrt failed ({proc.returncode}): "
                           f"{proc.stderr.strip()}")
    result["built"] = True
    result["complex_source_ok"] = _verify_complex_source(out_vrt)
    if verbose and not result["complex_source_ok"]:
        print(f"  [dem_vrt] WARN: {out_vrt.name} has no "
              f"<ComplexSource><NODATA> — NoData fall-through may not work",
              file=sys.stderr)
    return result


def _verify_complex_source(vrt: Path) -> bool:
    """Confirm the VRT uses ComplexSource with a NODATA element so a finer
    tile's holes reveal the coarser underlay."""
    try:
        txt = vrt.read_text()
    except OSError:
        return False
    return ("<ComplexSource" in txt) and ("<NODATA>" in txt)


def _load_catalog(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalog", type=Path,
                   default=REPO / "output" / "a3gt_v2" / "dem_catalog.json")
    p.add_argument("--members", nargs="+", required=True,
                   help="Catalog DEM ids, best/finest first.")
    p.add_argument("--out", type=Path, required=True, help="Output .vrt path.")
    p.add_argument("--with-lola", action="store_true",
                   help="Include the catalog's lola_base as underlay.")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    cat = _load_catalog(args.catalog)
    by_id = {d["id"]: d for d in cat["dems"]}
    try:
        ranked = [by_id[m] for m in args.members]
    except KeyError as e:
        print(f"unknown DEM id: {e}", file=sys.stderr)
        return 1
    lola = cat.get("lola_base") if args.with_lola else None
    try:
        res = build_site_vrt(ranked, lola, args.out, dry_run=args.dry_run)
    except (ValueError, RuntimeError) as e:
        print(f"VRT build failed: {e}", file=sys.stderr)
        return 1
    print(json.dumps({k: v for k, v in res.items() if k != "cmd"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
