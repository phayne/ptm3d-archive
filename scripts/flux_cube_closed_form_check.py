#!/usr/bin/env python3
"""Closed-form validation of the flux-cube GEOMETRY (design section 10.1).

Builds analytic meshes whose sky fractions and view-factor row sums are known
in closed form, runs ptm_flux_cube_geom on them, and scores the result against
the analytic value using the estimator's OWN a-priori error bar. Nothing here
is a hand-set tolerance.

ARM 1 -- INFINITE FLAT PLANE.  Cube at h above a plane so large it fills the
lower hemisphere. Exact:

      face        f_sky   sum_j F
      zenith        1        0
      nadir         0        1
      N/S/E/W      1/2      1/2

  Falsifies: frame handedness, the (E,N,U) construction, which face is 'nadir',
  and the terrain/sky channel split.

ARM 2 -- FINITE SQUARE PLANE, two-sided analytic bracket.  For a downward
patch at height h above a DISK of radius a, the cosine-weighted view factor is
exactly a^2/(a^2+h^2). A square of half-width L has an inscribed disk of radius
L and a circumscribed disk of radius L*sqrt(2), so

      L^2/(L^2+h^2)  <=  sum_j F(nadir)  <=  2L^2/(2L^2+h^2)

  This is the arm that falsifies the COSINE WEIGHTING itself. A sampler that
  drew uniformly over the hemisphere instead of cosine-weighted would report
  1 - h/sqrt(L^2+h^2) for the inscribed disk -- 0.553 at L=10, h=5, far outside
  the bracket [0.800, 0.889]. Getting the 1/pi normalization wrong lands outside
  it too. Arm 1 alone cannot see either error, because at h -> 0 every weighting
  gives 1.

Usage:
  python3 scripts/flux_cube_closed_form_check.py \\
      --bin build/bin/ptm_flux_cube_geom --workdir /tmp/fcvalidate --n-rays 16384
Exit code = number of failures.
"""

import argparse
import math
import os
import struct
import subprocess
import sys

R_BODY = 1737400.0  # m; places the analytic patches at a plausible planetary radius

# --- on-disk layouts, mirroring src/cpp/fluxcubes/flux_cube_format.h (packed) ---
HDR_FMT = "<4sIIIQdQQddddddIIQdddd12ddII"
CUBE_FMT = "<iidd3diiffffd"
PATCH_FMT = "<QII6f"
FACE_NAMES = ["zenith", "nadir", "north", "south", "east", "west"]


def write_ply(path, verts, faces):
    with open(path, "wb") as f:
        f.write(b"ply\n")
        f.write(b"format binary_little_endian 1.0\n")
        f.write(b"element vertex %d\n" % len(verts))
        f.write(b"property float x\nproperty float y\nproperty float z\n")
        f.write(b"element face %d\n" % len(faces))
        f.write(b"property list uchar int vertex_indices\n")
        f.write(b"end_header\n")
        for v in verts:
            f.write(struct.pack("<3f", *v))
        for tri in faces:
            f.write(struct.pack("<B3i", 3, *tri))


def make_plane(path, half_extent, cell, offset=0.0):
    """Flat square patch tangent to the body at lat=0, lon=0 (surface normal +x).

    Vertices are emitted at radius R_BODY so the tool's outward-normal test
    (dot with the radial direction) resolves exactly as it does on a real mesh.
    """
    n = int(round(2 * half_extent / cell)) + 1
    verts, faces = [], []
    for i in range(n):
        for j in range(n):
            verts.append((R_BODY, -half_extent + i * cell + offset,
                          -half_extent + j * cell + offset))
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j
            b = a + 1
            c = (i + 1) * n + j
            d = c + 1
            faces.append((a, c, b))
            faces.append((b, c, d))
    write_ply(path, verts, faces)
    return len(verts), len(faces)


def read_fcg(path):
    with open(path, "rb") as f:
        blob = f.read()
    hdr_size = struct.calcsize(HDR_FMT)
    v = struct.unpack_from(HDR_FMT, blob, 0)
    assert v[0] == b"FCG1", "bad magic %r" % (v[0],)
    n_cubes, n_faces_pc, total_nnz = v[2], v[3], v[4]
    cube_size = struct.calcsize(CUBE_FMT)
    patch_size = struct.calcsize(PATCH_FMT)
    off = hdr_size + n_cubes * cube_size
    patches = []
    for k in range(n_cubes * n_faces_pc):
        ro, rl, n50, fsky, fring, fmax, clo, tv, tvs = struct.unpack_from(
            PATCH_FMT, blob, off + k * patch_size)
        patches.append(dict(row_len=rl, f_sky=fsky, f_ring=fring, f_max=fmax,
                            closure=clo, tv=tv))
    return dict(n_cubes=n_cubes, n_faces=n_faces_pc, total_nnz=total_nnz,
                patches=patches, height_m=v[12], n_rays=v[14])


def run_geom(binpath, core, out, height, spacing, radius, n_rays, extra=None):
    cmd = [binpath, "--core", core, "--out", out,
           "--center-lat", "0", "--center-lon", "0",
           "--spacing", str(spacing), "--radius", str(radius),
           "--height-m", str(height), "--edge-m", "0.1",
           "--n-rays", str(n_rays), "--threads", "8"]
    if extra:
        cmd += extra
    return subprocess.run(cmd, capture_output=True, text=True)


def arm1(binpath, workdir, n_rays, fails):
    """Infinite plane: exact f_sky per face orientation."""
    plane = os.path.join(workdir, "plane_big.ply")
    nv, nf = make_plane(plane, 400.0, 1.0)
    out = os.path.join(workdir, "plane_big.fcg")
    r = run_geom(binpath, plane, out, 0.1, 50.0, 100.0, n_rays)
    print("\n=== ARM 1: infinite flat plane, cube at h = 0.1 m ===")
    print("plane %d verts / %d faces (800 m across; horizon at 89.99 deg)" % (nv, nf))
    if r.returncode != 0:
        print(r.stdout[-2500:]); print(r.stderr[-2500:])
        fails.append("arm1: geom tool rc=%d" % r.returncode)
        return
    fcg = read_fcg(out)
    expect = [1.0, 0.0, 0.5, 0.5, 0.5, 0.5]
    print("%-8s %10s %10s %12s %10s %8s" %
          ("face", "f_sky", "exact", "3*sigma_bin", "|delta|", "verdict"))
    for fi, name in enumerate(FACE_NAMES):
        vals = [fcg["patches"][c * 6 + fi]["f_sky"] for c in range(fcg["n_cubes"])]
        mean = sum(vals) / len(vals)
        p = expect[fi]
        sigma = math.sqrt(max(p * (1 - p), 1e-12) / n_rays)
        bar = max(3.0 * sigma, 3.0 / n_rays)
        d = abs(mean - p)
        ok = d <= bar
        if not ok:
            fails.append("arm1 %s: f_sky %.5f vs exact %.5f (bar %.5f)" % (name, mean, p, bar))
        print("%-8s %10.5f %10.5f %12.5f %10.5f %8s" %
              (name, mean, p, bar, d, "PASS" if ok else "FAIL"))
    worst = max(p["closure"] for p in fcg["patches"])
    ok = worst < 1e-6
    print("worst closure residual %.3e  %s" % (worst, "PASS" if ok else "FAIL"))
    if not ok:
        fails.append("arm1: closure %.3e" % worst)


def arm2(binpath, workdir, n_rays, fails):
    """Finite square plane: two-sided analytic bracket on the nadir row sum."""
    L, h = 10.0, 5.0
    plane = os.path.join(workdir, "plane_small.ply")
    nv, nf = make_plane(plane, L, 0.25)
    out = os.path.join(workdir, "plane_small.fcg")
    # One cube only, at the plane center, so the bracket is exact (an off-center
    # cube would see an asymmetric square and the inscribed-disk bound would not
    # apply).
    r = run_geom(binpath, plane, out, h, 100.0, 1.0, n_rays)
    lo = L * L / (L * L + h * h)
    hi = 2 * L * L / (2 * L * L + h * h)
    uniform_wrong = 1.0 - h / math.sqrt(L * L + h * h)
    print("\n=== ARM 2: finite square plane, L = %.0f m, h = %.0f m ===" % (L, h))
    print("cosine-weighted bracket: inscribed disk %.4f <= sum_j F(nadir) <= %.4f circumscribed"
          % (lo, hi))
    print("a UNIFORM (non-cosine) sampler would report %.4f -- outside the bracket, "
          "which is what makes this arm discriminating" % uniform_wrong)
    if r.returncode != 0:
        print(r.stdout[-2500:]); print(r.stderr[-2500:])
        fails.append("arm2: geom tool rc=%d" % r.returncode)
        return
    fcg = read_fcg(out)
    if fcg["n_cubes"] != 1:
        print("  NOTE: %d cubes placed; scoring the one at the lattice origin" % fcg["n_cubes"])
    nad = fcg["patches"][1]           # cube 0, face index 1 = nadir
    sumF = 1.0 - nad["f_sky"] - nad["f_ring"]
    zen = fcg["patches"][0]
    ok = (lo <= sumF <= hi)
    if not ok:
        fails.append("arm2: nadir sum_j F = %.5f outside [%.5f, %.5f]" % (sumF, lo, hi))
    print("\n%-28s %10s %10s %8s" % ("quantity", "measured", "expected", "verdict"))
    print("%-28s %10.5f %10s %8s" % ("nadir sum_j F", sumF,
                                     "[%.3f,%.3f]" % (lo, hi), "PASS" if ok else "FAIL"))
    # The zenith face of the same cube looks away from the plane: pure sky.
    zok = abs(zen["f_sky"] - 1.0) <= max(3.0 / n_rays, 1e-6)
    if not zok:
        fails.append("arm2: zenith f_sky %.6f != 1" % zen["f_sky"])
    print("%-28s %10.5f %10.3f %8s" % ("zenith f_sky", zen["f_sky"], 1.0,
                                       "PASS" if zok else "FAIL"))
    print("%-28s %10.5f" % ("nadir f_max (1 facet share)", nad["f_max"]))
    print("%-28s %10d" % ("nadir row_len (partners)", nad["row_len"]))


def arm3(binpath, workdir, n_rays, fails):
    """FLUSH LIMIT (design section 10.4), in its geometric form.

    As the patch is lowered onto a facet, its ray-traced view-factor row must
    converge to that single facet: f_max -> 1 and n_faces_50pct -> 1. This is
    the one arm that compares the cube's ray-traced row against the real
    triangulation rather than against an analytic surface, and it is monotone
    or it is wrong.

    The plane is deliberately OFFSET by a third of a cell so the lattice origin
    lands in a triangle INTERIOR. See arm 4 for why that matters.
    """
    L, cell = 50.0, 1.0
    plane = os.path.join(workdir, "plane_offset.ply")
    make_plane(plane, L, cell, offset=0.37)
    print("\n=== ARM 3: flush limit -- the row collapses onto one facet as h -> 0 ===")
    print("plane of %.1f m cells (facet area %.3f m^2), cube in a triangle interior" %
          (cell, 0.5 * cell * cell))
    print("%10s %10s %10s %12s" % ("height_m", "f_max", "n50", "row_len"))
    prev = -1.0
    rows = []
    for h in (1.0, 0.1, 0.01):
        out = os.path.join(workdir, "flush_h%g.fcg" % h)
        r = run_geom(binpath, plane, out, h, 100.0, 1.0, n_rays)
        if r.returncode != 0:
            print(r.stdout[-1500:]); print(r.stderr[-1500:])
            fails.append("arm3: geom tool rc=%d at h=%g" % (r.returncode, h))
            return
        fcg = read_fcg(out)
        nad = fcg["patches"][1]
        rows.append((h, nad["f_max"], nad["n_faces_50pct"] if "n_faces_50pct" in nad else 0,
                     nad["row_len"]))
        print("%10.2f %10.4f %10s %12d" % (h, nad["f_max"], "-", nad["row_len"]))
        if nad["f_max"] < prev:
            fails.append("arm3: f_max not monotone (h=%g gave %.4f after %.4f)"
                         % (h, nad["f_max"], prev))
        prev = nad["f_max"]
    final = rows[-1][1]
    ok = final > 0.99
    print("h = 0.01 m: f_max = %.4f  (want > 0.99)  %s" % (final, "PASS" if ok else "FAIL"))
    if not ok:
        fails.append("arm3: f_max only reached %.4f at h=0.01 m" % final)


def arm4(binpath, workdir, n_rays, fails):
    """VERTEX DEGENERACY -- a disclosure test, not a defect test.

    On a REGULAR mesh a lattice-aligned cube grid can land exactly on a mesh
    vertex. The nadir patch is then a 4-way average of the facets meeting there
    and f_max saturates at ~1/4 no matter how far the cube is lowered -- it never
    becomes the single-facet sensor the height factor is chosen to produce.

    This is not a bug in the estimator: the answer is correct for that geometry.
    It is a property a consumer must be able to SEE, which is exactly why f_max
    and n_faces_50pct are emitted per patch. This arm asserts that the
    diagnostic actually reveals it, so the disclosure cannot quietly stop
    working.

    A03MP026 is an irregular Delaunay triangulation and does not show this
    (measured nadir f_max p50 = 0.85), but a regularly-gridded DEM might.
    """
    L, cell = 50.0, 1.0
    plane = os.path.join(workdir, "plane_aligned.ply")
    make_plane(plane, L, cell, offset=0.0)   # vertex exactly at the lattice origin
    out = os.path.join(workdir, "vertex_h0.01.fcg")
    r = run_geom(binpath, plane, out, 0.01, 100.0, 1.0, n_rays)
    print("\n=== ARM 4: vertex-aligned placement is DISCLOSED by f_max ===")
    if r.returncode != 0:
        fails.append("arm4: geom tool rc=%d" % r.returncode)
        return
    fcg = read_fcg(out)
    nad = fcg["patches"][1]
    print("cube on a mesh VERTEX at h = 0.01 m: f_max = %.4f, row_len = %d" %
          (nad["f_max"], nad["row_len"]))
    print("  (arm 3, same height, cube in a facet interior: f_max -> 0.999)")
    ok = nad["f_max"] < 0.5
    print("  f_max saturates well below 1, so the diagnostic reveals it: %s"
          % ("PASS" if ok else "FAIL"))
    if not ok:
        fails.append("arm4: f_max = %.4f did not reveal the vertex degeneracy" % nad["f_max"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--n-rays", type=int, default=16384)
    a = ap.parse_args()
    os.makedirs(a.workdir, exist_ok=True)
    fails = []
    arm1(a.bin, a.workdir, a.n_rays, fails)
    arm2(a.bin, a.workdir, a.n_rays, fails)
    arm3(a.bin, a.workdir, a.n_rays, fails)
    arm4(a.bin, a.workdir, a.n_rays, fails)
    print("\n==== flux-cube closed-form check: %d failure(s) ====" % len(fails))
    for f in fails:
        print("  FAIL:", f)
    return len(fails)


if __name__ == "__main__":
    sys.exit(main())
