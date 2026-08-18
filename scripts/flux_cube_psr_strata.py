#!/usr/bin/env python3
"""Stratified analysis of a flux-cube product, for the Fermi gate.

The sunlit stratum is forgiving: direct solar is ~1000 W/m^2 and swamps
everything, so a sign error, a dropped term or a units slip in any of the small
components is invisible there. The PSR stratum is where those become visible --
direct solar is identically zero, terrain IR falls to ~0.02 W/m^2, and the
primary body's IR becomes COMPARABLE to the entire terrain budget. That is the
stratum this script exists to expose.

Strata are defined on the cube's own window-maximum direct-solar irradiance
across its six faces:

  PSR          max direct == 0            never sees the Sun at any epoch
  grazing      0 < max direct <= 50 W/m^2  terminator / low-Sun
  sunlit       max direct  > 50 W/m^2

Reported per stratum, per face: the window max and median of every component,
the component's share of the total, and the dynamic range. Nothing is
thresholded or hidden -- Fermi reads the numbers, not a verdict.

Usage:
  python3 scripts/flux_cube_psr_strata.py <run>/flux_cubes/flux_cubes.fcb
"""

import argparse
import math
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ptm_flux_cubes_to_hdf5 import read_fcb, idx, FACE_NAMES, COMP_NAMES, SIGMA  # noqa: E402


def percentile(v, p):
    if not v:
        return float("nan")
    s = sorted(v)
    k = min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))
    return s[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fcb")
    ap.add_argument("--grazing-max", type=float, default=50.0)
    a = ap.parse_args()
    d = read_fcb(a.fcb)
    nc, nf, ns = d["n_cubes"], d["n_faces"], d["n_steps"]

    # --- stratify on the cube's own peak direct solar over the window ---
    peak = []
    for cu in range(nc):
        m = 0.0
        for fa in range(nf):
            b = idx(d, cu, fa, 0, 0)
            m = max(m, max(d["flux"][b:b + ns]))
        peak.append(m)
    strata = {"PSR": [], "grazing": [], "sunlit": []}
    for cu in range(nc):
        if peak[cu] <= 0.0:
            strata["PSR"].append(cu)
        elif peak[cu] <= a.grazing_max:
            strata["grazing"].append(cu)
        else:
            strata["sunlit"].append(cu)

    print("=== flux-cube stratified summary ===")
    print("file: %s" % a.fcb)
    print("cubes %d, faces %d, output steps %d, height %.4f m"
          % (nc, nf, ns, d["height_m"]))
    print("\nstrata (by the cube's window-max direct solar across its six faces):")
    for k in ("PSR", "grazing", "sunlit"):
        n = len(strata[k])
        print("  %-8s %5d cubes (%5.1f%%)" % (k, n, 100.0 * n / max(1, nc)))

    # --- T_face by stratum: the independent cross-check on the classification
    print("\n--- underlying face temperature by stratum (K) ---")
    print("%-8s %10s %10s %10s" % ("stratum", "min", "median", "max"))
    for k in ("PSR", "grazing", "sunlit"):
        vals = []
        for cu in strata[k]:
            vals.extend(d["t_face"][cu * ns:(cu + 1) * ns])
        if vals:
            print("%-8s %10.3f %10.3f %10.3f"
                  % (k, min(vals), percentile(vals, 0.5), max(vals)))

    # --- per stratum, per face, per component ---
    for k in ("PSR", "grazing", "sunlit"):
        cubes = strata[k]
        if not cubes:
            continue
        print("\n=== stratum: %s  (%d cubes) ===" % (k, len(cubes)))
        print("%-8s %-13s %12s %12s %12s" % ("face", "component", "median", "max", "%% of total max"))
        for fa in range(nf):
            tot_max = 0.0
            for cu in cubes:
                b = idx(d, cu, fa, 5, 0)
                tot_max = max(tot_max, max(d["flux"][b:b + ns]))
            for co in range(6):
                vals = []
                mx = 0.0
                for cu in cubes:
                    b = idx(d, cu, fa, co, 0)
                    seg = d["flux"][b:b + ns]
                    vals.extend(seg)
                    mx = max(mx, max(seg))
                share = (100.0 * mx / tot_max) if tot_max > 0 else float("nan")
                print("%-8s %-13s %12.5g %12.5g %12.2f"
                      % (FACE_NAMES[fa] if co == 0 else "", COMP_NAMES[co],
                         percentile(vals, 0.5), mx, share))
            print("")

    # --- the PSR headline: is Earthshine comparable to terrain IR there? ---
    if strata["PSR"]:
        print("--- PSR headline: primary-body IR vs terrain IR ---")
        print("%-8s %14s %14s %10s" % ("face", "terrain_ir max", "primary_ir max", "ratio"))
        for fa in range(nf):
            tir = pir = 0.0
            for cu in strata["PSR"]:
                b2 = idx(d, cu, fa, 2, 0)
                b4 = idx(d, cu, fa, 4, 0)
                tir = max(tir, max(d["flux"][b2:b2 + ns]))
                pir = max(pir, max(d["flux"][b4:b4 + ns]))
            r = (pir / tir) if tir > 0 else float("nan")
            print("%-8s %14.6g %14.6g %10.4f" % (FACE_NAMES[fa], tir, pir, r))
        print("\n(The design's expectation is that these are COMPARABLE in a PSR --")
        print(" the stratum where a sign error or a dropped term is visible at all.)")

    # --- dynamic range, which sets the float64 requirement ---
    allv = [x for x in d["flux"] if x > 0]
    if allv:
        print("\ndynamic range of non-zero irradiance: %.4g .. %.4g W/m^2 (%.1f orders)"
              % (min(allv), max(allv), math.log10(max(allv) / min(allv))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
