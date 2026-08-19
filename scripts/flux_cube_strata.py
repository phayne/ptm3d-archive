#!/usr/bin/env python3
"""Stratified analysis of a flux-cube product, for the Fermi gate.

TERMINOLOGY, and it matters
---------------------------
A **PSR** -- permanently shadowed region -- is dark in ALL seasons. A cube that
is dark for one evaluated window is NOT a PSR sample: if it were truly
permanently shadowed it would also have been dark in the summer window. This
script therefore calls that stratum **window-dark**, never "PSR".

  window-dark   max direct solar == 0 over the ENTIRE evaluated window
  grazing       0 < max direct <= 50 W/m^2
  sunlit        max direct  > 50 W/m^2

The window-dark stratum is still the right thing to measure: it exercises the
**indirect-only flux path** -- the physics regime a genuine PSR cube lives in,
where direct solar is identically zero, terrain IR falls to the tens of
mW/m^2, and the primary body's IR stops being negligible. It validates the
dark-flux path. It does not sample a PSR, and this script will not say it does.

Whether a cube sits on a genuine PSR is a separate question answered by the
site's psr_mask, not by any single window. Measured for A03MP026 inside the
500 m production grid: 441 one-metre PSR pixels (346 counting only mask==1
interior), nearest 118.7 m from the site, in 7 patches, the largest 195 m^2 and
only 15 x 15 m across. Against a 0.785 km^2 grid that is ~0.06% areal coverage,
so a 25 m lattice is expected to land on ~0.7 of them -- and in fact lands on
**0 of 1257**. Genuine PSR cubes require a targeted dense array, not a finer
global lattice.

The sunlit stratum is forgiving: direct solar is ~1000 W/m^2 and swamps
everything, so a sign error, a dropped term or a units slip in any small
component is invisible there. The dark strata are where those become visible.

Reported per stratum, per face: the window max and median of every component,
the component's share of the total, and the dynamic range. Nothing is
thresholded or hidden -- Fermi reads the numbers, not a verdict.

Usage:
  python3 scripts/flux_cube_strata.py <run>/flux_cubes/flux_cubes.fcb
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
    strata = {"window_dark": [], "grazing": [], "sunlit": []}
    for cu in range(nc):
        if peak[cu] <= 0.0:
            strata["window_dark"].append(cu)
        elif peak[cu] <= a.grazing_max:
            strata["grazing"].append(cu)
        else:
            strata["sunlit"].append(cu)

    print("=== flux-cube stratified summary ===")
    print("file: %s" % a.fcb)
    print("cubes %d, faces %d, output steps %d, height %.4f m"
          % (nc, nf, ns, d["height_m"]))
    print("\nstrata (by the cube's window-max direct solar across its six faces).")
    print("NOTE: window-dark is NOT a PSR sample -- a PSR is dark in ALL seasons.")
    for k in ("window_dark", "grazing", "sunlit"):
        n = len(strata[k])
        print("  %-12s %5d cubes (%5.1f%%)" % (k, n, 100.0 * n / max(1, nc)))

    # --- T_face by stratum: the independent cross-check on the classification
    print("\n--- underlying face temperature by stratum (K) ---")
    print("%-12s %10s %10s %10s" % ("stratum", "min", "median", "max"))
    for k in ("window_dark", "grazing", "sunlit"):
        vals = []
        for cu in strata[k]:
            vals.extend(d["t_face"][cu * ns:(cu + 1) * ns])
        if vals:
            print("%-12s %10.3f %10.3f %10.3f"
                  % (k, min(vals), percentile(vals, 0.5), max(vals)))

    # --- per stratum, per face, per component ---
    for k in ("window_dark", "grazing", "sunlit"):
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

    # --- COLD stratum, cut on the cube's own minimum T_face over the window ---
    # A site/window can contain NO permanently shadowed cube at all and still
    # have a cold stratum that matters: the epoch-wise minimum is when a
    # radiator sees its coldest sink, and it is where the small components stop
    # being negligible. Cutting on peak direct solar alone cannot find it, so
    # this cut is always reported, PSR cubes or not.
    tmin = [min(d["t_face"][cu * ns:(cu + 1) * ns]) for cu in range(nc)]
    order = sorted(range(nc), key=lambda c: tmin[c])
    ncold = max(1, nc // 10)
    cold, warm = order[:ncold], order[-ncold:]
    print("\n=== COLD stratum: coldest %d cubes by min T_face (%.2f .. %.2f K) ==="
          % (ncold, tmin[cold[0]], tmin[cold[-1]]))
    print("    (warmest %d for contrast: %.2f .. %.2f K)"
          % (ncold, tmin[warm[0]], tmin[warm[-1]]))
    for label, grp in (("cold", cold), ("warm", warm)):
        print("\n  -- %s --" % label)
        print("  %-8s %14s %14s %14s %10s" %
              ("face", "terrain_ir", "primary_ir", "primary_refl", "pIR/tIR"))
        for fa in range(nf):
            # Evaluate at each cube's OWN coldest epoch: that is the moment the
            # question is about, and a window max would hide it.
            tir = pir = prf = 0.0
            for cu in grp:
                seg = d["t_face"][cu * ns:(cu + 1) * ns]
                st = seg.index(min(seg))
                tir += d["flux"][idx(d, cu, fa, 2, st)]
                pir += d["flux"][idx(d, cu, fa, 4, st)]
                prf += d["flux"][idx(d, cu, fa, 3, st)]
            n = len(grp)
            tir, pir, prf = tir / n, pir / n, prf / n
            r = (pir / tir) if tir > 0 else float("nan")
            print("  %-8s %14.6g %14.6g %14.6g %10.4f"
                  % (FACE_NAMES[fa], tir, pir, prf, r))
    print("\n  (means over the stratum, each cube evaluated at its OWN coldest epoch)")

    # --- dark-flux headline: is Earthshine comparable to terrain IR there? ---
    if not strata["window_dark"]:
        print("\n--- dark-flux headline: NOT AVAILABLE for this product ---")
        print("  Every one of the %d cubes receives direct sunlight at some epoch of this" % nc)
        print("  window, so there is no window-dark stratum and the indirect-only flux path")
        print("  is UNEXERCISED here. Use the COLD decile above, or a winter window.")
        print("  (Separately: a window-dark cube would still NOT be a PSR sample.)")
    if strata["window_dark"]:
        print("--- dark-flux headline: primary-body IR vs terrain IR (WINDOW-DARK stratum) ---")
        print("%-8s %14s %14s %10s" % ("face", "terrain_ir max", "primary_ir max", "ratio"))
        for fa in range(nf):
            tir = pir = 0.0
            for cu in strata["window_dark"]:
                b2 = idx(d, cu, fa, 2, 0)
                b4 = idx(d, cu, fa, 4, 0)
                tir = max(tir, max(d["flux"][b2:b2 + ns]))
                pir = max(pir, max(d["flux"][b4:b4 + ns]))
            r = (pir / tir) if tir > 0 else float("nan")
            print("%-8s %14.6g %14.6g %10.4f" % (FACE_NAMES[fa], tir, pir, r))
        print("\n(These are expected to become COMPARABLE where direct solar is zero --")
        print(" the regime where a sign error or a dropped term is visible at all.)")
        print(" ⛔ This is the WINDOW-DARK stratum, which validates the indirect-only flux")
        print(" path. It is NOT a permanently-shadowed sample: a PSR is dark in all seasons,")
        print(" and whether a cube sits on one is decided by the site psr_mask, not by this")
        print(" window. At A03MP026 no 25 m lattice point lands on a PSR pixel (0 of 1257).")

    # --- dynamic range, which sets the float64 requirement ---
    allv = [x for x in d["flux"] if x > 0]
    if allv:
        print("\ndynamic range of non-zero irradiance: %.4g .. %.4g W/m^2 (%.1f orders)"
              % (min(allv), max(allv), math.log10(max(allv) / min(allv))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
