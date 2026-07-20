#!/usr/bin/env python3
"""A3GT v2 — coarse occluder ring/annulus builder.

Produces the `ring.ply` consumed by ptm_thermal_sim's `mesh.occluder_file`
(see docs/extended-occluder-architecture.md). The ring is **occluder-only**
geometry: it is added to the illumination shadow-ray BVH so distant
topography (out to the physics horizon) shadows the SfS core, but it is
never thermally solved, radiosity-coupled, or output.

Method (no dtm_to_ply_delaunay change needed):
  1. Build a *coarse* full disk of radius R_ext from the site's `-te`-clipped
     VRT (LOLA underlay + any SfS). A low `--roi-max-points` gives ~LOLA-
     native facet spacing (80-160 m) — finer is pointless (LOLA is 80 m/px)
     and only inflates the BVH.
  2. Drop every face whose centroid is within R_core of the site center
     (geodesic) → the annulus [R_core, R_ext]. The thermally-solved core
     mesh covers the inside; the ring must not shadow-coincide with it.
  3. Renumber vertices and write `ring.ply` (binary PLY, per-face props
     preserved — reuses the crop module's reader/writer).

Ring face count is ~2 x `--ring-max-points`; the default keeps a 20 km
ring near LOLA resolution at ~10^5 triangles (trivial BVH cost).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path("/Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/europa/heat3d/ptm")
sys.path.insert(0, str(REPO / "scripts"))
from a3gt_v2_crop_core import _load_ply, _write_ply, geodesic_km  # noqa: E402

DTM_TOOL_DEFAULT = Path(
    "/Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/"
    "europa/heat3d/dtm/dtm_to_ply_delaunay")
RING_MAX_POINTS_DEFAULT = 80_000   # ~1e5 ring tris @ R_ext=20 km ≈ LOLA 80-160 m


def build_ring(site_lat: float, site_lon: float, vrt: Path,
               out_ply: Path, *, r_core_km: float, r_ext_km: float,
               ring_max_points: int = RING_MAX_POINTS_DEFAULT,
               dtm_tool: Path = DTM_TOOL_DEFAULT,
               dry_run: bool = False, verbose: bool = True) -> dict[str, Any]:
    """Build the occluder annulus PLY. Returns a stats dict."""
    out_ply = Path(out_ply)
    res: dict[str, Any] = {
        "out_ply": str(out_ply), "r_core_km": r_core_km,
        "r_ext_km": r_ext_km, "ring_max_points": ring_max_points,
    }
    tmp_disk = Path(tempfile.gettempdir()) / f"_ringdisk_{out_ply.stem}.ply"
    cmd = [
        str(dtm_tool), str(vrt), str(tmp_disk),
        "--roi-center", f"{site_lat:.6f}", f"{site_lon:.6f}",
        "--roi-radius", f"{r_ext_km:g}",
        "--roi-core-frac", "0.02",     # ~uniform: tiny protected core
        "--density-exp", "1.0",        # uniform-ish coarse sampling
        "--roi-max-points", str(int(ring_max_points)),
        "--max-edge", "0", "--south-pole-coords",
    ]
    res["cmd"] = " ".join(cmd)
    if verbose:
        print(f"  [ring] {out_ply.name}: coarse disk R={r_ext_km:g} km "
              f"(<= {ring_max_points} pts) then drop < {r_core_km:g} km core",
              file=sys.stderr)
    if dry_run:
        res["dry_run"] = True
        return res

    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"dtm_to_ply_delaunay (ring disk) failed "
                           f"({p.returncode}): {p.stderr.strip()[-300:]}")

    header, vprops, vert, faces, fview, flat, flon = _load_ply(tmp_disk)
    n_disk = len(faces)
    d = geodesic_km(site_lat, site_lon, flat, flon)
    keep = d >= r_core_km                       # annulus only
    n_ring = int(keep.sum())
    res.update({"n_disk_faces": n_disk, "n_ring_faces": n_ring,
                "max_d_km": round(float(d.max()), 3)})
    if n_ring == 0:
        raise ValueError("ring empty — R_core >= R_ext or bad site center?")

    kept_faces = faces[keep]
    used = np.unique(kept_faces)
    remap = np.full(len(vert), -1, dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    new_faces = np.ascontiguousarray(remap[kept_faces].astype("<i4"))
    new_vert = vert[used]
    kept_rows = fview[keep].copy()
    kept_rows[:, 1:13] = new_faces.view(np.uint8).reshape(len(new_faces), 12)

    out_ply.parent.mkdir(parents=True, exist_ok=True)
    _write_ply(out_ply, header, new_vert, kept_rows, n_ring)
    try:
        tmp_disk.unlink()
    except OSError:
        pass

    # mean ring facet spacing (occlusion-fidelity sanity vs LOLA ~80 m)
    annulus_km2 = np.pi * (r_ext_km**2 - r_core_km**2)
    res["approx_ring_spacing_m"] = (
        round(float(np.sqrt(annulus_km2 * 1e6 / max(n_ring, 1))), 1))
    if verbose:
        print(f"  [ring] {out_ply.name}: {n_disk:,} disk -> {n_ring:,} ring "
              f"tris (~{res['approx_ring_spacing_m']} m, LOLA~80 m)",
              file=sys.stderr)
    return res


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vrt", type=Path, required=True,
                   help="Site's -te-clipped VRT (LOLA underlay + SfS).")
    p.add_argument("--site-latlon", nargs=2, type=float, required=True,
                   metavar=("LAT", "LON"))
    p.add_argument("--out", type=Path, required=True, help="ring.ply path")
    p.add_argument("--r-core-km", type=float, default=3.0)
    p.add_argument("--r-ext-km", type=float, required=True)
    p.add_argument("--ring-max-points", type=int,
                   default=RING_MAX_POINTS_DEFAULT)
    p.add_argument("--dtm-tool", type=Path, default=DTM_TOOL_DEFAULT)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    try:
        res = build_ring(args.site_latlon[0], args.site_latlon[1], args.vrt,
                         args.out, r_core_km=args.r_core_km,
                         r_ext_km=args.r_ext_km,
                         ring_max_points=args.ring_max_points,
                         dtm_tool=args.dtm_tool, dry_run=args.dry_run)
    except (RuntimeError, ValueError) as e:
        print(f"ring build failed: {e}", file=sys.stderr)
        return 1
    import json
    print(json.dumps({k: v for k, v in res.items() if k != "cmd"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
