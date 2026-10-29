#!/usr/bin/env python3
"""Recompute the Moon3D de-tilted latitude-sweep summary statistics that feed
docs/figures/final/deltas.json (manuscript Section 4.3 latitude trend).

For each de-tilted run (dt85..dt60, a 254k-vertex Ames Regolith Testbed crater
patch de-tilted and placed at a clean integer latitude) this reports, from the
per-face statistics binaries:

  * psr_area_frac  : AREA fraction with direct_solar_max < 1.0 W/m^2 (permanent
                     shadow -- never directly sunlit). Geometric, coupled run.
  * psr_count_frac : COUNT fraction with the same PSR criterion.
  * median_tmax_all: median of temperature_max over all faces (coupled).
  * median_tavg_all: median of temperature_avg over all faces (coupled).
  * faces_over_400K: number of coupled faces with temperature_max > 400 K
                     (honest concave-focusing tail -- NO mask; reported for audit).
  * radoff median_tmax_all / median_tavg_all for the no-coupling counterpart.

A facet whose value is not finite (NaN, +inf, -inf) has no value and is
excluded from BOTH the numerator and the denominator of every fraction and
count above (Paul's ruling, 2026-09-15) -- it is never counted as qualifying
and never as not-qualifying. The resolved population is printed under the table
and stored as n_resolved_psr / n_resolved_tmax, because a fraction over a
shrinking denominator is not interpretable without its N.

Each statistics/*.bin carries a leading int32/float32 face-count header, so the
data field is raw[1:] (face f lives at index f+1 in the file, index f after the
drop). PSR geometry is identical between coupled and radoff (same mesh), so the
psr_* fractions are computed on the coupled run only.

    python3 moon3d_deltas.py \
        --root .../output/moon3d_lat_final \
        --json-out /path/dt_rows.json     # optional JSON fragment for deltas.json

With --check it compares psr_count_frac / psr_area_frac / median_tmax_all against
the hand-verified table and flags any deviation beyond --tol.
"""
from __future__ import annotations
import os
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pyvista as pv

sys.path.insert(0, str(Path(__file__).parent))
from moon3d_temp_3d import read_ply_vertices_double, tri_faces
from ptm_population import (area_fraction, count_fraction, population_note,
                            resolved_mask, resolved_values, warn_if_unresolved)

PSR_THRESH = 1.0    # W/m^2  direct-solar max below this = permanent shadow
COLD_THRESH = 110.0  # K
HEAT3D = Path(os.environ.get("PTM_HEAT3D_ROOT", str(Path(__file__).resolve().parents[3])))   # holds output/<run>

# de-tilted sweep tags, latitude in degrees (clean integer placements)
DT = [("dt85", -85.0), ("dt80", -80.0), ("dt75", -75.0),
      ("dt70", -70.0), ("dt65", -65.0), ("dt60", -60.0)]

# the hand-verified table: tag -> (psr_count_frac, psr_area_frac, median_tmax_all)
LUNA = {
    "dt60": (0.1446, 0.0365, 316.07),
    "dt65": (0.2110, 0.0618, 302.57),
    "dt70": (0.3231, 0.1150, 282.33),
    "dt75": (0.4871, 0.2123, 228.72),
    "dt80": (0.6752, 0.3455, 94.04),
    "dt85": (0.8565, 0.4865, 74.21),
}


def load_stat(run: Path, name: str) -> np.ndarray:
    return np.fromfile(run / "statistics" / f"{name}.bin", dtype=np.float32)[1:]


def face_areas(run: Path) -> np.ndarray:
    v = read_ply_vertices_double(str(run / "mesh.ply"))
    f = tri_faces(pv.read(str(run / "mesh.ply")))
    tri = v[f]
    return 0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(HEAT3D / "output/moon3d_lat_final"))
    ap.add_argument("--json-out", default=None, help="write a deltas.json fragment here")
    ap.add_argument("--check", action="store_true", default=True)
    ap.add_argument("--tol", type=float, default=0.01,
                    help="abs tol on fractions; K tol is 100x this on median_tmax")
    a = ap.parse_args()

    root = Path(a.root)
    rows = {}
    populations = {}
    any_flag = False
    hdr = (f"{'tag':5s} {'lat':>6s} {'psr_count':>10s} {'psr_area':>9s} "
           f"{'medTmax':>8s} {'medTavg':>8s} {'>400K':>6s} | "
           f"{'Δcount':>8s} {'Δarea':>7s} {'ΔmedTmax':>9s}")
    print(hdr)
    print("-" * len(hdr))
    for tag, lat in DT:
        cdir = root / f"{tag}_coupled"
        rdir = root / f"{tag}_radoff"
        area = face_areas(cdir)
        dsm = load_stat(cdir, "direct_solar_max")
        tmc = load_stat(cdir, "temperature_max")
        tac = load_stat(cdir, "temperature_avg")
        tmr = load_stat(rdir, "temperature_max")
        tar = load_stat(rdir, "temperature_avg")
        assert len(dsm) == len(tmc) == len(area) == len(tac), \
            f"{tag}: coupled face/stat mismatch"
        assert len(tmr) == len(tar), f"{tag}: radoff face/stat mismatch"

        # Paul's ruling 2026-09-15: a facet with no value leaves BOTH the
        # numerator and the denominator. A bare `dsm < PSR_THRESH` would call a
        # -inf facet a PSR and a NaN facet not-a-PSR; a finiteness test does
        # neither. The area weights join the test because an unresolved facet's
        # area must leave the total too.
        res_psr = resolved_mask(dsm, area)
        warn_if_unresolved(f"{tag} PSR (direct_solar_max)", res_psr)
        psr = (dsm < PSR_THRESH) & res_psr
        psr_area_frac = area_fraction(psr, res_psr, area, f"{tag} psr_area_frac")
        psr_count_frac = count_fraction(psr, res_psr, f"{tag} psr_count_frac")
        # The order statistics leave the unresolved facets behind too: an
        # unguarded np.median() goes NaN on one bad facet, and NaN silently
        # passes the tolerance check below (every NaN comparison is False), so
        # the script would print ALL WITHIN TOL and exit 0 on data whose median
        # is unknown.
        res_tmc = resolved_mask(tmc)
        res_tac = resolved_mask(tac)
        warn_if_unresolved(f"{tag} median_tmax_all (temperature_max)", res_tmc)
        warn_if_unresolved(f"{tag} median_tavg_all (temperature_avg)", res_tac)
        med_tmax = float(np.median(resolved_values(tmc, res_tmc)))
        med_tavg = float(np.median(resolved_values(tac, res_tac)))
        res_400 = resolved_mask(tmc)
        warn_if_unresolved(f"{tag} faces_over_400K (temperature_max)", res_400)
        over400 = int(np.count_nonzero((tmc > 400.0) & res_400))
        populations[tag] = (population_note(res_psr, area),
                            population_note(res_400))

        rows[f"{tag}_coupled"] = {
            "lat": lat, "n": int(len(tmc)),
            "psr_area_frac": round(psr_area_frac, 4),
            "psr_count_frac": round(psr_count_frac, 4),
            "median_tmax_all": round(med_tmax, 2),
            "median_tavg_all": round(med_tavg, 2),
            "faces_over_400K": over400,
            # population the psr_* fractions and faces_over_400K were computed
            # over (Paul's ruling 2026-09-15) -- a fraction over a shrinking
            # denominator is only interpretable next to its N.
            "n_resolved_psr": int(res_psr.sum()),
            "n_resolved_tmax": int(res_400.sum()),
        }
        rows[f"{tag}_radoff"] = {
            "lat": lat, "n": int(len(tmr)),
            "psr_area_frac": round(psr_area_frac, 4),
            "psr_count_frac": round(psr_count_frac, 4),
            "n_resolved_psr": int(res_psr.sum()),
            "median_tmax_all": round(float(np.median(
                resolved_values(tmr, resolved_mask(tmr)))), 2),
            "median_tavg_all": round(float(np.median(
                resolved_values(tar, resolved_mask(tar)))), 2),
        }

        lc, la, lt = LUNA[tag]
        dcount, darea, dtmax = psr_count_frac - lc, psr_area_frac - la, med_tmax - lt
        # A non-finite delta must FLAG, not sail through: `abs(nan) > tol` is
        # False, so an unguarded check would report ALL WITHIN TOL on a value
        # that is not a number.
        flag_c = not np.isfinite(dcount) or abs(dcount) > a.tol
        flag_a = not np.isfinite(darea) or abs(darea) > a.tol
        flag_t = not np.isfinite(dtmax) or abs(dtmax) > a.tol * 100
        any_flag = any_flag or flag_c or flag_a or flag_t
        star = " <== FLAG" if (flag_c or flag_a or flag_t) else ""
        print(f"{tag:5s} {lat:6.1f} {psr_count_frac:10.4f} {psr_area_frac:9.4f} "
              f"{med_tmax:8.2f} {med_tavg:8.2f} {over400:6d} | "
              f"{dcount:+8.4f} {darea:+7.4f} {dtmax:+9.2f}{star}")

    # Population disclosure (Paul's ruling 2026-09-15). The fractions above are
    # "of the facets we could actually evaluate"; without N they are not
    # interpretable. Printed as its own block so the table lines are unchanged.
    print("\npopulation behind each row (unresolved facets leave numerator AND denominator):")
    for tag, _ in DT:
        psr_note, tmax_note = populations[tag]
        print(f"  {tag:5s} psr_*_frac: {psr_note}   faces_over_400K: {tmax_note}")

    # monotonicity report (as computed, ordered dt85..dt60 = lat -85..-60)
    pc = [rows[f"{t}_coupled"]["psr_count_frac"] for t, _ in DT]
    pa = [rows[f"{t}_coupled"]["psr_area_frac"] for t, _ in DT]
    mt = [rows[f"{t}_coupled"]["median_tmax_all"] for t, _ in DT]
    print("\nmonotonicity dt85->dt60 (lat -85 -> -60):")
    print(f"  psr_count_frac  {pc}  strictly-decreasing={all(x>y for x,y in zip(pc,pc[1:]))}")
    print(f"  psr_area_frac   {pa}  strictly-decreasing={all(x>y for x,y in zip(pa,pa[1:]))}")
    print(f"  median_tmax_all {mt}  strictly-increasing={all(x<y for x,y in zip(mt,mt[1:]))}")

    print("\nLuna-table check:", "MISMATCH" if any_flag else "ALL WITHIN TOL")

    if a.json_out:
        Path(a.json_out).write_text(json.dumps(rows, indent=2))
        print(f"wrote fragment {a.json_out}")
    return 1 if any_flag else 0


if __name__ == "__main__":
    raise SystemExit(main())
