#!/usr/bin/env python3
"""A3GT v2 — physics-horizon extended-mesh parameter derivation.

The sim must run on a mesh extended radially past the requested ROI so that
distant massifs / crater rims cast shadows and contribute reflected/IR flux;
the *delivered* product is later cropped back to the core ROI
(scripts/a3gt_v2_crop_core.py).

This module computes, per site, the dtm_to_ply_delaunay parameters for that
extended mesh at a CONSTANT face budget (the ring's faces come from
coarsening, not a bigger budget).

Extension radius — physics horizon
----------------------------------
Terrain of relief Δh at distance d occludes the sun at elevation θ when
Δh/d > tan θ. The farthest terrain that can still shadow the core is

    R_ext = relief_max / tan(sun_elev_min)

clamped below by R_core and above by both a hard cap and the DEM coverage
the site actually has:

    R_ext = clip( relief_max / tan(sun_elev_min) / 1000,
                  R_core,  min(R_EXT_HARD_CAP, max_covered_radius) )   [km]

``sun_elev_min`` is the smallest *peak* solar elevation across the scheduled
windows (peak ≈ 90° − |lat_site − lat_subsolar|), floored at
``sun_elev_floor_deg`` so PSR/grazing geometry does not send R_ext → ∞.

Density model caveat (verified in dtm_to_ply_delaunay.cpp:1124-1145)
-------------------------------------------------------------------
At 1 m/px a 3 km core (~28 M pts) far exceeds the point budget, so the C++
tool sets ``thin_entire_domain`` and ``core_radius_km = 0`` — there is **no
hard protected core** and ``--roi-core-frac`` is inert. Resolution falls off
smoothly from the center by ``1/(1+β·(d/R_ext)^p)²`` with β auto-searched to
the budget. ``core_frac`` is therefore emitted for documentation only; the
operative knob is ``density_exp`` (p), whose production default comes from
the calibration gate (``--calibrate``).
"""

from __future__ import annotations

import argparse
import csv as _csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path("/Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/europa/heat3d/ptm")

# Defaults (production). DENSITY_EXP_RING is set by the calibration gate.
R_CORE_KM_DEFAULT = 3.0
SUN_ELEV_FLOOR_DEG = 1.0
# 20 km cap (chosen 2026-05-16): captures the near-field occluders that
# dominate at these sun elevations while keeping the per-site VRT a
# tractable 40k x 40k @ 1 m (~3 min/mesh). 60 km (true horizon) makes the
# clipped VRT 120k x 120k (~30 min/mesh) and a near-empty ring at any
# feasible budget. The VRT is -te clipped to this radius (mandatory: an
# unclipped SfS+LOLA VRT is 608k x 608k @ 1 m = 3.7e11 px, infeasible).
R_EXT_HARD_CAP_KM = 20.0
# Re-calibrated 2026-05-16 at the REAL geometry (R_ext=20 km hard cap, LOLA
# base staged so the ring is 80 m/px LOLA — dense ring faces are pointless).
# Apples-to-apples vs the 3 km-only/900 K production baseline (radial profile,
# 2-3 km annulus = 4.76 m). The earlier 11.567 km / aggregate-median pick was
# discarded after the radial profile showed a 900 K extended core regresses
# the outer ROI. Budget sweep on lunar polar site L (p=1.5):
#   1.0 M -> 2.00 M faces, 2-3 km = 6.25 m
#   1.1 M -> 2.20 M faces, 2-3 km = 5.82 m
#   1.4 M -> 2.80 M faces, 2-3 km = 4.95 m  <-- chosen (meets the user's
#            hard requirement of <=5 m at 2-3 km; ~=production 4.76 m; inner
#            <=1.5 km core >= production; ~1.55x prod faces, << 4.99 M
#            GPU-wedge regime). The "2.0-2.2 M faces" target was an
#            illustrative estimate that could not meet the <=5 m bar.
MAX_POINTS_DEFAULT = 1_400_000
RELIEF_MAX_FALLBACK_M = 2000.0
DENSITY_EXP_RING_DEFAULT = 1.5


def sun_elev_min_deg(site_lat_deg: float, schedule: dict[str, Any],
                     floor_deg: float = SUN_ELEV_FLOOR_DEG) -> float:
    """Smallest peak solar elevation across the scheduled windows (deg)."""
    peaks = []
    for w in schedule.get("windows", []):
        lat_sun = w["subsolar_lat_at_center_deg"]
        peaks.append(90.0 - abs(site_lat_deg - lat_sun))
    if not peaks:
        return floor_deg
    return max(floor_deg, min(peaks))


def extended_mesh_params(site: dict[str, Any], schedule: dict[str, Any],
                         *,
                         r_core_km: float = R_CORE_KM_DEFAULT,
                         sun_elev_floor_deg: float = SUN_ELEV_FLOOR_DEG,
                         r_ext_hard_cap_km: float = R_EXT_HARD_CAP_KM,
                         max_points: int = MAX_POINTS_DEFAULT,
                         density_exp_ring: float = DENSITY_EXP_RING_DEFAULT,
                         relief_max_fallback_m: float = RELIEF_MAX_FALLBACK_M,
                         dem_coverage: dict[str, Any] | None = None
                         ) -> dict[str, Any]:
    """Derive the extended-mesh dtm_to_ply_delaunay parameters for one site.

    ``site``      a manifest record from a3gt_v2_site_filter catalog mode
                  (carries ``lat_deg`` and ``coverage.extended``).
    ``schedule``  subsolar_production.json (windows w/ subsolar lat).
    ``dem_coverage`` optional override of the ``coverage.extended`` block.
    """
    lat = float(site["lat_deg"])
    cov = dem_coverage or site.get("coverage", {}).get("extended", {})
    relief_max_m = cov.get("relief_max_m")
    if relief_max_m is None or relief_max_m <= 0:
        relief_max_m = relief_max_fallback_m
    max_cov_km = cov.get("max_covered_radius_km")
    if max_cov_km is None or max_cov_km <= 0:
        max_cov_km = r_core_km

    sem = sun_elev_min_deg(lat, schedule, sun_elev_floor_deg)
    r_ext_raw_km = relief_max_m / math.tan(math.radians(sem)) / 1000.0
    upper = min(r_ext_hard_cap_km, max_cov_km)
    r_ext_km = max(r_core_km, min(r_ext_raw_km, upper))

    return {
        "ellipse_id": site.get("ellipse_id"),
        "roi_radius_km": round(r_ext_km, 4),
        "core_frac": round(r_core_km / r_ext_km, 6),   # documentation only
        "density_exp": density_exp_ring,
        "max_points": int(max_points),
        "r_core_km": r_core_km,
        "r_ext_km": round(r_ext_km, 4),
        "r_ext_raw_km": round(r_ext_raw_km, 2),
        "sun_elev_min_deg": round(sem, 4),
        "relief_max_m": round(float(relief_max_m), 2),
        "max_covered_radius_km": round(float(max_cov_km), 3),
        "coverage_capped": r_ext_raw_km > upper,
    }


# ---------------------------------------------------------------------------
# Calibration gate
# ---------------------------------------------------------------------------

def _read_ply_face_centroids(ply: Path):
    """(n_faces, face_lat[], face_lon[]) via the canonical ptm_regridder
    reader — same code path the crop step and COG export use."""
    sys.path.insert(0, str(REPO / "src" / "python"))
    from ptm_regridder import read_ply_mesh_minimal
    m = read_ply_mesh_minimal(str(ply))
    return m["n_faces"], m["face_latitudes"], m["face_longitudes"]


def _median_spacing_m(lat, lon, body_r=1737400.0):
    """Median nearest-neighbor great-circle spacing of a centroid set (m)."""
    import numpy as np
    from scipy.spatial import cKDTree
    la = np.radians(lat); lo = np.radians(lon)
    xyz = np.column_stack([np.cos(la) * np.cos(lo),
                           np.cos(la) * np.sin(lo), np.sin(la)]) * body_r
    if len(xyz) < 2:
        return float("nan")
    d, _ = cKDTree(xyz).query(xyz, k=2)
    return float(np.median(d[:, 1]))


def calibrate(site: dict[str, Any], schedule: dict[str, Any], vrt: Path,
              dtm_tool: Path, exps: list[float], out_dir: Path,
              max_points: int = MAX_POINTS_DEFAULT) -> list[dict[str, Any]]:
    """Build one extended mesh per density_exp and report core/ring spacing.

    Acceptance (per the plan): total faces ≈ production budget; inner-R_core
    median spacing within ~2× of a R_core-only mesh; ring spacing fine
    enough (≲ few hundred m) to resolve horizon relief.
    """
    import numpy as np
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    base = extended_mesh_params(site, schedule, max_points=max_points)
    rc, re = base["r_core_km"], base["r_ext_km"]
    slat, slon = float(site["lat_deg"]), float(site["lon_deg"])

    def gc_km(lat, lon):
        la0, lo0 = math.radians(slat), math.radians(slon)
        la, lo = np.radians(lat), np.radians(lon)
        dla = la - la0; dlo = lo - lo0
        a = np.sin(dla / 2) ** 2 + math.cos(la0) * np.cos(la) * np.sin(dlo / 2) ** 2
        return 1737.4 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

    for p in exps:
        ply = out_dir / f"calib_p{p:g}.ply"
        cmd = [str(dtm_tool), str(vrt), str(ply),
               "--roi-center", f"{slat:.6f}", f"{slon:.6f}",
               "--roi-radius", f"{re:g}",
               "--roi-core-frac", f"{rc/re:.6f}",
               "--density-exp", f"{p:g}",
               "--roi-max-points", str(int(max_points)),
               "--max-edge", "0", "--south-pole-coords"]
        t0 = __import__("time").time()
        r = subprocess.run(cmd, capture_output=True, text=True)
        dt = __import__("time").time() - t0
        if r.returncode != 0:
            rows.append({"density_exp": p, "error": r.stderr.strip()[-300:]})
            continue
        n_f, lat, lon = _read_ply_face_centroids(ply)
        d_km = gc_km(lat, lon)
        core = d_km <= rc
        ring = (d_km > rc) & (d_km <= re)
        rows.append({
            "density_exp": p, "n_faces": int(n_f),
            "core_faces": int(core.sum()), "ring_faces": int(ring.sum()),
            "core_spacing_m": round(_median_spacing_m(lat[core], lon[core]), 2)
            if core.sum() > 1 else None,
            "ring_spacing_m": round(_median_spacing_m(lat[ring], lon[ring]), 2)
            if ring.sum() > 1 else None,
            "build_s": round(dt, 1),
        })
    return rows


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, required=True,
                   help="site_manifest.json (catalog mode).")
    p.add_argument("--schedule", type=Path, required=True)
    p.add_argument("--site", help="ellipse_id (default: all non-skip sites).")
    p.add_argument("--out-json", type=Path,
                   help="Write per-site params JSON.")
    p.add_argument("--calibrate", action="store_true",
                   help="Run the density_exp calibration sweep on --site.")
    p.add_argument("--vrt", type=Path, help="VRT for --calibrate.")
    p.add_argument("--dtm-tool", type=Path, default=Path(
        "/Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/"
        "europa/heat3d/dtm/dtm_to_ply_delaunay"))
    p.add_argument("--exps", default="1.5,2,3,4")
    p.add_argument("--out-dir", type=Path, default=Path("/tmp/calib_ext"))
    p.add_argument("--max-points", type=int, default=MAX_POINTS_DEFAULT)
    args = p.parse_args()

    manifest = json.loads(args.manifest.read_text())
    schedule = json.loads(args.schedule.read_text())
    by_id = {s["ellipse_id"]: s for s in manifest["sites"]}

    if args.calibrate:
        site = by_id[args.site]
        rows = calibrate(site, schedule, args.vrt, args.dtm_tool,
                         [float(x) for x in args.exps.split(",")],
                         args.out_dir, args.max_points)
        print(json.dumps({"site": args.site,
                          "params": extended_mesh_params(
                              site, schedule, max_points=args.max_points),
                          "sweep": rows}, indent=2))
        return 0

    targets = ([by_id[args.site]] if args.site
               else [s for s in manifest["sites"] if not s.get("skip")])
    res = [extended_mesh_params(s, schedule, max_points=args.max_points)
           for s in targets]
    if args.out_json:
        args.out_json.write_text(json.dumps(res, indent=2))
        print(f"Wrote {args.out_json}  ({len(res)} site(s))")
    else:
        print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
