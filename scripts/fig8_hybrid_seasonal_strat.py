#!/usr/bin/env python3
"""Stratified bias/RMSE for the fig8_hybrid rigorous seasonal Diviner comparison
(coupled vs radoff), on top of diviner_seasonal_compare.py's own grid-building
functions (model_seasonal / diviner_seasonal / season_masks) -- no re-derivation
of the registration or footprint-Tbol logic, just additional masks over the
SAME arrays diviner_seasonal_compare.py already computes and plots.

Strata (2026-07-22, Sagan's fig8_hybrid resume task):
  overall : all valid core cells (same numbers diviner_seasonal_compare.py prints)
  PSR     : Diviner < 200 K (diviner_compare.py's own --psr-threshold convention)
  sunlit  : Diviner >= 200 K
  deep    : Diviner < 110 K  (a finer PSR subset -- NOT an established repo
            convention; chosen to bracket the ~40-110 K deep-floor range BUGS.md
            documents for genuine PSR floors; flagged, not asserted, as "the"
            deep-floor cut -- confirm with Paul if a different value is wanted)
  margin  : 110 K <= Diviner < 200 K (PSR margin / less-extreme shadow)

Usage:
    python3 scripts/fig8_hybrid_seasonal_strat.py \\
        --run-dir /Users/paha3326/projects/moon3d/outputs/fig8_hybrid_production/coupled \\
        --label coupled --center-lat -88.1 --center-lon 6.3
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import spiceypy as sp

sys.path.insert(0, str(Path(__file__).parent.parent / "src/python"))
import diviner_seasonal_compare as dsc

DEEP_K = 110.0
PSR_K = 200.0


def stratified(model, div, core, lo, hi):
    m = core & np.isfinite(model) & np.isfinite(div) & (div >= lo) & (div < hi)
    r = (model - div)[m]
    if r.size < 3:
        return dict(n=int(r.size), bias=float("nan"), rmse=float("nan"))
    return dict(n=int(r.size), bias=float(r.mean()), rmse=float(np.sqrt((r ** 2).mean())))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--center-lat", type=float, required=True)
    ap.add_argument("--center-lon", type=float, required=True)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    run = Path(args.run_dir)
    sp.furnsh(dsc.METAKERNEL)
    smask, wmask, lats = dsc.season_masks(run)
    print(f"[{args.label}] run={run} summer={int(smask.sum())} winter={int(wmask.sum())} steps")

    mods = {"summer": dsc.model_seasonal(run, args.center_lat, args.center_lon, smask),
            "winter": dsc.model_seasonal(run, args.center_lat, args.center_lon, wmask)}

    result = {"label": args.label, "run_dir": str(run),
              "center": [args.center_lat, args.center_lon],
              "deep_threshold_K": DEEP_K, "psr_threshold_K": PSR_K}
    for season in ("summer", "winter"):
        g = mods[season]
        for stat_name in ("max", "mean"):
            div_field = "max" if stat_name == "max" else "avg"
            dv = dsc.diviner_seasonal(
                dsc.SEASDIR / f"polar_south_80_{season}_{div_field}_polar.npy", g)
            model = np.where(g["core"], g[stat_name], np.nan)
            dvm = np.where(g["core"], dv, np.nan)
            core = g["core"]
            strata = {
                "overall": stratified(model, dvm, core, -np.inf, np.inf),
                "PSR_lt200K": stratified(model, dvm, core, -np.inf, PSR_K),
                "sunlit_ge200K": stratified(model, dvm, core, PSR_K, np.inf),
                "deep_lt110K": stratified(model, dvm, core, -np.inf, DEEP_K),
                "margin_110_200K": stratified(model, dvm, core, DEEP_K, PSR_K),
            }
            key = f"{season}_{stat_name}"
            result[key] = strata
            print(f"  {key:14s}", end="")
            for sk, st in strata.items():
                print(f" | {sk}: N={st['n']:5d} bias={st['bias']:+6.1f} RMSE={st['rmse']:5.1f}",
                      end="")
            print()

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(result, indent=2))
        print(f"wrote {args.out_json}")
    sp.kclear()
    return result


if __name__ == "__main__":
    main()
