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
  2. Punch the inner hole by **registering it to the core mesh's actual outer
     boundary polygon** (`--core-ply`): drop every ring face any part of which
     lies planimetrically inside that boundary. See `ring_registration.py`.
  3. Renumber vertices and write `ring.ply` (binary PLY, per-face props
     preserved — reuses the crop module's reader/writer).
  4. Re-measure the emitted ring against the core boundary and REFUSE to
     deliver it if any ring geometry still overlaps the core.

Ring face count is ~2 x `--ring-max-points`; the default keeps a 20 km
ring near LOLA resolution at ~10^5 triangles (trivial BVH cost).

Step 2 used to be "drop every face whose CENTROID is within a nominal R_core
of the site center". That kept triangles whose inward-most vertex protruded
up to ~130 m inside R_core, laying coarse ring surface on top of the core's
own terrain; at grazing polar sun the overlapping lip cast a hard, spurious,
near-permanent shadow on core edge faces (the >50 K discontinuous cold rim
seen at sites A, B and C). The ring never enters a view-factor matrix, so
re-clipping it never requires a VF rebuild.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

# Import the sibling modules from THIS checkout. The previous hardcoded
# OneDrive path meant a working-tree edit to a3gt_v2_crop_core.py silently had
# no effect here, which is the same two-checkout divergence that has bitten
# this project before.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from a3gt_v2_crop_core import _load_ply, _write_ply, geodesic_km  # noqa: E402,F401
import ring_registration as rreg  # noqa: E402

DTM_TOOL_DEFAULT = Path(
    "/Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/"
    "europa/heat3d/dtm/dtm_to_ply_delaunay")
RING_MAX_POINTS_DEFAULT = 80_000   # ~1e5 ring tris @ R_ext=20 km ≈ LOLA 80-160 m


def _vertex_latlon(path: Path, vprops, vert):
    """Per-vertex lat/lon (deg) from a DTM-tool PLY.

    Falls back to deriving them from x,y,z, but says so — a silent frame or
    body-radius substitution here would mis-register the seam by meters.
    """
    names = {p[0] for p in vprops}
    if {"latitude", "longitude"} <= names:
        return (vert["latitude"].astype(np.float64),
                vert["longitude"].astype(np.float64))
    print(f"  [ring] NOTE: {Path(path).name} carries no per-vertex "
          f"latitude/longitude; deriving them from x,y,z on a "
          f"{rreg.BODY_R_M:.0f} m sphere.", file=sys.stderr)
    x = vert[vprops[0][0]].astype(np.float64)
    y = vert[vprops[1][0]].astype(np.float64)
    z = vert[vprops[2][0]].astype(np.float64)
    r = np.sqrt(x * x + y * y + z * z)
    return (np.degrees(np.arcsin(np.clip(z / np.maximum(r, 1e-30), -1, 1))),
            np.degrees(np.arctan2(y, x)))


def build_ring(site_lat: float, site_lon: float, vrt: Path,
               out_ply: Path, *, r_core_km: float, r_ext_km: float,
               ring_max_points: int = RING_MAX_POINTS_DEFAULT,
               dtm_tool: Path = DTM_TOOL_DEFAULT,
               core_ply: Path | None = None,
               seam_margin_m: float = 0.0,
               allow_unregistered: bool = False,
               dry_run: bool = False, verbose: bool = True) -> dict[str, Any]:
    """Build the occluder annulus PLY. Returns a stats dict.

    ``core_ply`` is the site's thermally solved core mesh. Its actual outer
    boundary polygon defines the ring's inner hole. Omitting it falls back to
    the legacy nominal-circle centroid clip, which is a known defect and
    therefore requires ``allow_unregistered=True``.
    """
    out_ply = Path(out_ply)
    res: dict[str, Any] = {
        "out_ply": str(out_ply), "r_core_km": r_core_km,
        "r_ext_km": r_ext_km, "ring_max_points": ring_max_points,
        "core_ply": str(core_ply) if core_ply else None,
        "seam_margin_m": seam_margin_m,
        "registered": core_ply is not None,
    }
    if core_ply is None and not allow_unregistered:
        raise ValueError(
            "no --core-ply given. The ring's inner hole must be registered to "
            "the core mesh's actual outer boundary; the legacy nominal-circle "
            "centroid clip leaves up to ~130 m of ring surface lying on top of "
            "the core and casts a spurious hard shadow on its edge faces. Pass "
            "--core-ply, or --allow-unregistered to reproduce the old (broken) "
            "geometry deliberately.")
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
        how = (f"clip hole to {Path(core_ply).name}'s outer boundary"
               if core_ply else
               f"UNREGISTERED nominal-circle drop < {r_core_km:g} km core")
        print(f"  [ring] {out_ply.name}: coarse disk R={r_ext_km:g} km "
              f"(<= {ring_max_points} pts) then {how}", file=sys.stderr)
    if dry_run:
        res["dry_run"] = True
        return res

    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"dtm_to_ply_delaunay (ring disk) failed "
                           f"({p.returncode}): {p.stderr.strip()[-300:]}")

    header, vprops, vert, faces, fview, flat, flon = _load_ply(tmp_disk)
    n_disk = len(faces)
    core = None
    if core_ply is not None:
        core = rreg.core_boundary_profile(Path(core_ply), site_lat, site_lon)
        vlat, vlon = _vertex_latlon(tmp_disk, vprops, vert)
        keep, _, _ = rreg.clip_ring_to_core(
            vlat, vlon, faces, core, site_lat, site_lon,
            margin_m=seam_margin_m)
        res["core_boundary_r_m"] = [round(float(core.r.min()), 2),
                                    round(float(core.r.max()), 2)]
    else:
        print("  [ring] WARNING: building an UNREGISTERED ring (legacy "
              "nominal-circle centroid clip). Ring geometry will overlap the "
              "core mesh by up to ~130 m and cast spurious hard shadows on its "
              "edge faces. Do not use this for science.", file=sys.stderr)
        d = geodesic_km(site_lat, site_lon, flat, flon)
        keep = d >= r_core_km                   # annulus only (legacy)
        res["max_d_km"] = round(float(d.max()), 3)
    n_ring = int(keep.sum())
    res.update({"n_disk_faces": n_disk, "n_ring_faces": n_ring})
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

    # --- registration gate: re-measure what was actually written ----------
    if core is not None:
        rlat, rlon, _, rtris = rreg.read_ply(out_ply)
        rep = rreg.registration_report(rlat, rlon, rtris, core,
                                       site_lat, site_lon)
        rreg.print_report(rep, name=out_ply.name, stream=sys.stderr)
        res["registration"] = {k: v for k, v in rep.items()
                               if k != "gap_profile_m"}
        if not rreg.gate(rep, name=out_ply.name, stream=sys.stderr):
            # Remove the rejected file. Leaving it behind poisons the
            # orchestrator's "ring present -> skip" cache: the next run takes
            # the cached branch, the cached-ring check fails, and the site is
            # skipped forever until a human deletes it by hand.
            try:
                out_ply.unlink()
                print(f"  [ring] removed the rejected ring {out_ply.name}",
                      file=sys.stderr)
            except OSError as e:
                print(f"  [ring] WARNING: could not remove the rejected ring "
                      f"{out_ply}: {e}. DELETE IT BY HAND -- it will otherwise "
                      "be picked up as a cached ring.", file=sys.stderr)
            raise RuntimeError(
                f"registration gate FAILED for {out_ply.name}: "
                f"{rep['n_overlapping_faces']} ring faces overlap the core by "
                f"up to {rep['max_overlap_m']:.1f} m")
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
    p.add_argument("--core-ply", type=Path, default=None,
                   help="Site's core mesh PLY. Its actual outer boundary "
                        "polygon defines the ring's inner hole. Required "
                        "unless --allow-unregistered.")
    p.add_argument("--seam-margin-m", type=float, default=0.0,
                   help="Extra clearance beyond the core boundary (m).")
    p.add_argument("--allow-unregistered", action="store_true",
                   help="Reproduce the legacy nominal-circle centroid clip. "
                        "Produces a ring that overlaps the core and casts "
                        "spurious hard shadows; for regression fixtures only.")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    try:
        res = build_ring(args.site_latlon[0], args.site_latlon[1], args.vrt,
                         args.out, r_core_km=args.r_core_km,
                         r_ext_km=args.r_ext_km,
                         core_ply=args.core_ply,
                         seam_margin_m=args.seam_margin_m,
                         allow_unregistered=args.allow_unregistered,
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
