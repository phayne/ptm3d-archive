#!/usr/bin/env python3
"""A3GT v2 — SPICE-driven sub-solar-latitude schedule for the lunar 18.6-yr nodal cycle.

Computes lunar sub-solar latitude vs time across the requested year window
and identifies the day-of-year of the annual maximum ("summer") and minimum
("winter") for each year in the chosen sampling set. Builds a ±15-day
30-day lunation window centered on each extremum.

The lunar nodal cycle does NOT modulate the annual amplitude (verified
2026-05-06: 3.094°–3.149° peak-to-peak across all 18 yrs, < 1.7 % spread).
What it does modulate is the *phase* of the annual sinusoid — the
calendar day on which subsolar-lat hits its annual max drifts ~19.4°/yr
in solar longitude, returning to start after 18.6 yr. The year-set
production / crosscheck / full options sample this phase drift at
different densities.

SPICE kernels used (verified on disk 2026-05-06):
  $CSPICE_DATA/lsk/naif0012.tls
  $CSPICE_DATA/spk/de440.bsp
  $CSPICE_DATA/pck/pck00011.tpc

Output JSON shape:
  {
    "start_time": "2034-10-30T00:00:00",
    "total_days": 6798.5,
    "year_set": "production" | "crosscheck" | "full",
    "year_indices": [0, 5, 9, 14],
    "windows": [
        {"tag": "Y00_summer", "year": 0, "season": "summer",
         "center_et_tdb": ..., "center_iso": "...",
         "start_iso": "...", "duration_days": 30,
         "subsolar_lat_at_center_deg": +1.54},
        ...
    ]
  }
"""

from __future__ import annotations

import os
import argparse
import json
import math
import sys
from pathlib import Path
from typing import Sequence

import numpy as np


_KDIR = Path(os.environ.get("CSPICE_DATA",
                            str(Path(__file__).resolve().parents[1] / "data" / "kernels")))
KERNEL_PATHS = [
    _KDIR / "lsk/naif0012.tls",
    _KDIR / "spk/de440.bsp",
    _KDIR / "pck/pck00011.tpc",
]


YEAR_SETS: dict[str, list[int]] = {
    # Production: 4 years sampling the 4 quadrants of the 18.6-yr nodal cycle.
    "production": [0, 5, 9, 14],
    # Cross-check: complementary 4 years; offset by ~1 nodal-quadrant-quarter.
    # Used by Phase 3 to verify the production set captured the inter-annual phase drift.
    "crosscheck": [1, 7, 12, 17],
    # Full: all 18 yrs, for fallback if both Phase 3 checks fail.
    "full": list(range(18)),
}


def furnsh_kernels() -> "module":  # type: ignore[name-defined]
    """Import spiceypy and load the lunar subsolar-lat kernels. Bails clearly if missing."""
    try:
        import spiceypy as sp  # type: ignore[import-not-found]
    except ImportError:
        print("ERROR: spiceypy not installed. Run on the europa_viz env.", file=sys.stderr)
        raise SystemExit(2)
    for k in KERNEL_PATHS:
        if not k.is_file():
            print(f"ERROR: SPICE kernel missing: {k}", file=sys.stderr)
            raise SystemExit(2)
        sp.furnsh(str(k))
    return sp


def compute_subsolar_lat_series(sp, et0: float, n_days: int) -> np.ndarray:
    """Return subsolar latitude (deg) at 1-day cadence starting at et0."""
    day = 86_400.0
    lats = np.empty(n_days, dtype=np.float64)
    for i in range(n_days):
        et = et0 + i * day
        # Sub-Sun point on Moon in IAU_MOON, light-time + stellar aberration corrected.
        spoint, _trgepc, _srfvec = sp.subpnt(
            "NEAR POINT/ELLIPSOID", "MOON", et, "IAU_MOON", "LT+S", "SUN")
        # Convert sub-Sun rectangular coords to planetographic lat/lon on the lunar sphere.
        _, lat, _ = sp.recpgr("MOON", spoint, 1737.4, 0.0)
        lats[i] = math.degrees(lat)
    return lats


def find_annual_extrema(lats: np.ndarray, year_d: float = 365.25) -> list[dict]:
    """For each integer year in `lats`, return {year, summer_doy, winter_doy, summer_lat, winter_lat}."""
    n = len(lats)
    n_years = int(n // year_d)
    out: list[dict] = []
    for y in range(n_years):
        lo = int(round(y * year_d))
        hi = min(int(round((y + 1) * year_d)), n)
        chunk = lats[lo:hi]
        sdoy = int(np.argmax(chunk))   # day-of-year (0-based) of summer extremum
        wdoy = int(np.argmin(chunk))
        out.append({
            "year": y,
            "summer_doy": sdoy,
            "winter_doy": wdoy,
            "summer_lat_deg": float(chunk[sdoy]),
            "winter_lat_deg": float(chunk[wdoy]),
            "year_start_day": lo,
        })
    return out


def build_windows(extrema: Sequence[dict], year_indices: Sequence[int],
                  et0: float, lunation_days: int, sp) -> list[dict]:
    """For each year in `year_indices`, emit summer + winter lunation windows."""
    day = 86_400.0
    half = lunation_days / 2.0
    wins: list[dict] = []
    for yi in year_indices:
        ex = extrema[yi]
        for season, key_doy, key_lat in (
            ("summer", ex["summer_doy"], ex["summer_lat_deg"]),
            ("winter", ex["winter_doy"], ex["winter_lat_deg"]),
        ):
            center_day = ex["year_start_day"] + key_doy
            center_et  = et0 + center_day * day
            start_et   = center_et - half * day
            wins.append({
                "tag": f"Y{yi:02d}_{season}",
                "year": yi,
                "season": season,
                "center_et_tdb": center_et,
                "center_iso": sp.et2utc(center_et, "ISOC", 0),
                "start_et_tdb": start_et,
                "start_iso": sp.et2utc(start_et, "ISOC", 0),
                "duration_days": lunation_days,
                "subsolar_lat_at_center_deg": key_lat,
            })
    return wins


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", default="2034-10-30T00:00:00",
                   help="Start time of the 18-yr window (ISO; default matches "
                        "the previous A3GT batch).")
    p.add_argument("--total-days", type=float, default=6798.5,
                   help="Total span over which to compute subsolar-lat samples "
                        "(default: 6798.5 = lunar 18.6-yr nodal cycle).")
    p.add_argument("--year-set", choices=list(YEAR_SETS.keys()), default="production",
                   help="Which year set to emit windows for "
                        "(production=0,5,9,14; crosscheck=1,7,12,17; full=0..17).")
    p.add_argument("--lunation-days", type=int, default=30,
                   help="Lunation window length in days (default: 30, ~1 synodic day).")
    p.add_argument("--output", required=True, type=Path,
                   help="Output schedule JSON path.")
    args = p.parse_args()

    sp = furnsh_kernels()
    et0 = sp.str2et(args.start)
    n_days = int(args.total_days)
    print(f"Computing subsolar-lat at 1-day cadence over {n_days} d "
          f"starting {args.start}...", file=sys.stderr)
    lats = compute_subsolar_lat_series(sp, et0, n_days)

    extrema = find_annual_extrema(lats)
    print(f"Found {len(extrema)} integer years; "
          f"per-year amplitudes (max-min): "
          f"min={min(e['summer_lat_deg']-e['winter_lat_deg'] for e in extrema):.4f}°, "
          f"max={max(e['summer_lat_deg']-e['winter_lat_deg'] for e in extrema):.4f}°",
          file=sys.stderr)

    year_indices = YEAR_SETS[args.year_set]
    if max(year_indices) >= len(extrema):
        print(f"ERROR: year_set '{args.year_set}' references year "
              f"{max(year_indices)} but only {len(extrema)} years computed.",
              file=sys.stderr)
        return 1

    windows = build_windows(extrema, year_indices, et0, args.lunation_days, sp)

    schedule = {
        "start_time": args.start,
        "total_days": args.total_days,
        "year_set": args.year_set,
        "year_indices": year_indices,
        "lunation_days": args.lunation_days,
        "extrema_per_year": extrema,
        "windows": windows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(schedule, f, indent=2)

    print(f"\nWrote {args.output}: {len(windows)} windows "
          f"(year_set={args.year_set}, sampled days = {len(windows)*args.lunation_days})")
    for w in windows:
        print(f"  {w['tag']:<14}  {w['start_iso']}  +{w['duration_days']:>3} d  "
              f"subsolar_lat_center = {w['subsolar_lat_at_center_deg']:+.4f}°")
    return 0


if __name__ == "__main__":
    sys.exit(main())
