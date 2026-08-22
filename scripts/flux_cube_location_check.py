#!/usr/bin/env python3
"""Discriminator for named-location flux cubes: the named path must reproduce
the lattice path BIT FOR BIT at the same point.

A cube placed by name at the site's own lat/lon and a cube placed by the
lattice at its origin describe the same physical fixture. If the two placement
routines are genuinely one routine, every recorded field except the grid
indices -- which are meant to differ, the named cube carrying a sentinel -- and
every ray-traced view-factor entry must be identical to the last bit.

This is a strong test precisely because it is easy to fail by a hair and still
look right. Getting it to pass required three real fixes, each of which a
tolerance would have hidden:

  1. `d * r_mean - org` subtracts two ~1.7e6 m vectors to get ~0 m --
     catastrophic cancellation, ~1e-11 m residue. Difference the DIRECTIONS
     first, then scale.
  2. The same three trig lines written in two places are not the same
     computation: the optimizer may contract them differently. One shared
     noinline dirFromLatLon() for both callers.
  3. Even then the site frame normalized the direction TWICE (U = unit(C) where
     C was already unit) while the named path normalized once. Same formula,
     different SEQUENCE, one ulp apart. Both paths now do the same thing.

Usage:
  python3 scripts/flux_cube_location_check.py --bin build/bin/ptm_flux_cube_geom \
      --core data/meshes/lunar_crater_80N.ply --lat 80 --lon 0 --workdir /tmp/fcloc
Exit code = number of failures.
"""

import argparse
import os
import struct
import subprocess
import sys

HDR = "<4sIIIQdQQddddddIIQdddd12ddII"
CUBE = "<iidd3dif4fd"
PATCH = "<QII6f"
FIELDS = ["grid_i", "grid_j", "x_m", "y_m", "cx", "cy", "cz", "face_id",
          "edge_m", "slope_deg", "aspect_deg", "area_m2", "elev_m", "height_m"]


def load(path):
    b = open(path, "rb").read()
    hs = struct.calcsize(HDR)
    v = struct.unpack_from(HDR, b, 0)
    nc, nf, nnz = v[2], v[3], v[4]
    cs, ps = struct.calcsize(CUBE), struct.calcsize(PATCH)
    cubes = [struct.unpack_from(CUBE, b, hs + i * cs) for i in range(nc)]
    off = hs + nc * cs
    pat = [struct.unpack_from(PATCH, b, off + k * ps) for k in range(nc * nf)]
    off += nc * nf * ps
    cols = struct.unpack_from("<%di" % nnz, b, off)
    off += 4 * nnz
    vals = struct.unpack_from("<%df" % nnz, b, off)
    return v, cubes, pat, cols, vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--core", required=True)
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--n-rays", type=int, default=16384)
    a = ap.parse_args()
    os.makedirs(a.workdir, exist_ok=True)

    csv = os.path.join(a.workdir, "one.csv")
    with open(csv, "w") as f:
        f.write("name,lat_deg,lon_deg\nsite_center,%r,%r\n" % (a.lat, a.lon))

    common = [a.bin, "--core", a.core, "--center-lat", str(a.lat), "--center-lon", str(a.lon),
              "--spacing", "100", "--height-m", "1.0", "--edge-m", "0.5",
              "--n-rays", str(a.n_rays)]
    A = os.path.join(a.workdir, "A_lattice.fcg")
    B = os.path.join(a.workdir, "B_named.fcg")
    for tag, extra, out in (("A lattice", ["--radius", "0"], A),
                            ("B named", ["--radius", "-1", "--locations", csv], B)):
        r = subprocess.run(common + extra + ["--out", out], capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-2000:]); print(r.stderr[-1500:])
            print("FAIL: arm %s returned %d" % (tag, r.returncode))
            return 1

    va, ca, pa, cola, vala = load(A)
    vb, cb, pb, colb, valb = load(B)
    fa, fb = list(ca[0]), list(cb[0])
    fails = 0
    print("=== named location vs lattice cube at the same point ===")
    print("grid indices  A=(%d,%d)  B=(%d,%d)   <- MUST differ (named sentinel)"
          % (fa[0], fa[1], fb[0], fb[1]))
    if (fb[0], fb[1]) != (-2147483648, -2147483648):
        print("FAIL: named cube did not carry the sentinel grid index"); fails += 1
    diffs = [(FIELDS[i], x, y) for i, (x, y) in enumerate(zip(fa, fb)) if i >= 2 and x != y]
    print("CubeRecord fields beyond the indices that differ: %d %s"
          % (len(diffs), diffs if diffs else "(none)"))
    if diffs:
        fails += 1
    for label, ok in (("all 6 PatchRecords bit-identical", pa == pb),
                      ("VF column indices identical", cola == colb),
                      ("VF values BIT-identical", vala == valb),
                      ("nnz equal (%d vs %d)" % (va[4], vb[4]), va[4] == vb[4])):
        print("%-38s %s" % (label, "PASS" if ok else "FAIL"))
        if not ok:
            fails += 1
    print("\nVERDICT: %s  (%d failure%s)"
          % ("PASS -- the named path reproduces the lattice cube BIT FOR BIT" if fails == 0
             else "FAIL -- the two placement paths disagree", fails, "" if fails == 1 else "s"))
    return fails


if __name__ == "__main__":
    sys.exit(main())
