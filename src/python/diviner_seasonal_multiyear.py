#!/usr/bin/env python3
"""Seasonal-amplitude convergence vs equilibration: analyze a multi-year PTM3D
run year-by-year against the Diviner seasonal product (Williams et al. 2019).

The ~45 cm thermal column has a multi-year time constant, so a 2-month
equilibration + 1-year output never reaches periodic-steady seasonal state. Here
we run N years (coupled) and, for EACH simulated year, split into summer/winter
(subsolar latitude < / >= 0) and compute the footprint Tbol max & mean, comparing
to the Diviner summer/winter max & avg. This shows whether the modeled seasonal
amplitude (summer - winter) grows toward the observed value as the deep column
equilibrates over the years.

Usage: diviner_seasonal_multiyear.py --run-dir <multiyear run> --center-lat .. --center-lon ..
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

import numpy as np
import spiceypy as sp

sys.path.insert(0, str(Path(__file__).parent))
from diviner_seasonal_compare import (
    season_masks, model_seasonal, diviner_seasonal, stats, METAKERNEL, SEASDIR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--center-lat", type=float, default=-87.45)
    ap.add_argument("--center-lon", type=float, default=-2.1)
    ap.add_argument("--steps-per-year", type=int, default=365)
    a = ap.parse_args()
    run = Path(a.run_dir)

    sp.furnsh(METAKERNEL)
    smask, wmask, lats = season_masks(run)
    n = len(lats)
    nyr = n // a.steps_per_year
    print(f"{run.name}: {n} output steps, ~{nyr} years; "
          f"subsolar lat {lats.min():+.2f}..{lats.max():+.2f}")

    # Diviner seasonal grids (computed once per site grid, reused across years)
    print(f"\n{'year':>4} | {'summer-max':>20} | {'winter-max':>20} | "
          f"{'model amplitude':>15}")
    print(f"{'':>4} | {'modelMx  Div  bias':>20} | {'modelMx  Div  bias':>20} | "
          f"{'sumMx-winMx':>15}")
    div_cache = {}
    for yr in range(nyr):
        lo, hi = yr * a.steps_per_year, min((yr + 1) * a.steps_per_year, n)
        ymask = np.zeros(n, bool); ymask[lo:hi] = True
        out = {}
        for season, smk in (("summer", smask), ("winter", wmask)):
            g = model_seasonal(run, a.center_lat, a.center_lon, ymask & smk)
            for stat_name in ("max", "mean"):
                key = (season, "max" if stat_name == "max" else "avg")
                if key not in div_cache:
                    div_cache[key] = diviner_seasonal(
                        SEASDIR / f"polar_south_80_{season}_{key[1]}_polar.npy", g)
                model = np.where(g["core"], g[stat_name], np.nan)
                dvm = np.where(g["core"], div_cache[key], np.nan)
                st = stats(model, dvm, g["core"])
                out[(season, stat_name)] = (float(np.nanmedian(model)),
                                            float(np.nanmedian(dvm)), st["bias"])
        sM = out[("summer", "max")]; wM = out[("winter", "max")]
        sA = out[("summer", "mean")]; wA = out[("winter", "mean")]
        amp_model = sM[0] - wM[0]
        print(f"{yr+1:4d} | {sM[0]:6.1f} {sM[1]:5.1f} {sM[2]:+5.1f} | "
              f"{wM[0]:6.1f} {wM[1]:5.1f} {wM[2]:+5.1f} | {amp_model:9.1f}")
        # Full 2x2 table (model / Diviner / bias=model-Div) for max AND mean.
        print(f"     | {'-- summer-max --':>20} | {'-- winter-max --':>20} |")
        print(f"     |  model={sM[0]:6.1f} Div={sM[1]:5.1f} bias={sM[2]:+5.1f}"
              f" |  model={wM[0]:6.1f} Div={wM[1]:5.1f} bias={wM[2]:+5.1f} |")
        print(f"     | {'-- summer-mean --':>20} | {'-- winter-mean --':>20} |")
        print(f"     |  model={sA[0]:6.1f} Div={sA[1]:5.1f} bias={sA[2]:+5.1f}"
              f" |  model={wA[0]:6.1f} Div={wA[1]:5.1f} bias={wA[2]:+5.1f} |")
    # Diviner seasonal amplitude reference (summer-max minus winter-max medians)
    div_sumMx = np.nanmedian(div_cache[("summer", "max")])
    div_winMx = np.nanmedian(div_cache[("winter", "max")])
    print(f"\nDiviner: summer-max med {div_sumMx:.1f}, winter-max med {div_winMx:.1f}, "
          f"AMPLITUDE {div_sumMx - div_winMx:.1f} K  (target for the model amplitude column)")
    sp.kclear()


if __name__ == "__main__":
    main()
