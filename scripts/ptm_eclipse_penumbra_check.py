#!/usr/bin/env python3
"""PTM-38 acceptance harness: does the engine's direct-solar track the TRUE
fraction of the solar disk left visible during a partial (penumbral) eclipse?

The engine's eclipse model is a ray/sphere test of the primary's *geometric*
sphere against each sample direction on the finite solar disk, so the physical
truth this harness computes uses the identical assumption -- two circles on the
sky, the primary's disk of angular radius ``asin(R_p/d_p)`` occulting the Sun's
disk of angular radius ``asin(R_sun/d_sun)``.  No umbra-cone convergence, no
atmospheric refraction, no limb darkening: those are separate tickets, and the
point here is to test the *normalization*, not the shadow shape.

Geometry is evaluated FROM THE PROBE FACE (not the body center), exactly as the
illumination model does, using the run's own ``body_positions.csv`` and the face
latitude/longitude carried in ``mesh.ply``.

Usage
-----
    python scripts/ptm_eclipse_penumbra_check.py OUTPUT_DIR \
        [--primary EARTH] [--lat 0 --lon 0] [--steps 80:95] [--tol 0.016]

Prints a per-step table of

    visible_fraction_true | direct_solar/full_flux (engine) | difference

and exits 1 if any step inside the requested window differs by more than
``--tol`` (the sampling quantum).  Exit 0 = the engine tracks the truth.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ptm_eclipse_diag import (PRIMARY_RADIUS_KM, read_body_positions,  # noqa: E402
                              read_ply_face_latlon, read_series)


def read_ply_face_centroids(path):
    """Face centroids [n_face, 3] in the mesh's own units, from a PTM3D
    binary_little_endian mesh.ply (vertex: 6 doubles; face: uchar count +
    count*int32 + float lat + float lon)."""
    import struct
    with open(path, "rb") as fh:
        header = b""
        while b"end_header" not in header:
            chunk = fh.readline()
            if not chunk:
                raise SystemExit("mesh.ply: no end_header")
            header += chunk
        text = header.decode("ascii", "replace")
        if "binary_little_endian" not in text:
            raise SystemExit("mesh.ply: only binary_little_endian is supported")
        n_vert = n_face = 0
        for line in text.splitlines():
            if line.startswith("element vertex"):
                n_vert = int(line.split()[2])
            elif line.startswith("element face"):
                n_face = int(line.split()[2])
        verts = np.frombuffer(fh.read(n_vert * 6 * 8), dtype="<f8").reshape(n_vert, 6)[:, :3]
        blob = fh.read()
    cent = np.empty((n_face, 3), dtype=np.float64)
    off = 0
    for i in range(n_face):
        cnt = blob[off]
        off += 1
        idx = struct.unpack_from("<%di" % cnt, blob, off)
        off += 4 * cnt
        off += 8  # lat, lon floats
        cent[i] = verts[list(idx)].mean(axis=0)
    return cent

SUN_RADIUS_KM = 695700.0          # constants.cpp SUN_RADIUS
SOLAR_CONSTANT = 1361.0           # constants.cpp SOLAR_CONSTANT
KM_PER_AU = 149597870.7           # constants.cpp KM_PER_AU


def circle_overlap_area(r1, r2, d):
    """Area of the lens where two disks of radii r1, r2 whose centers are d
    apart overlap.  Standard closed form; vectorized over d."""
    r1 = np.asarray(r1, dtype=float)
    r2 = np.asarray(r2, dtype=float)
    d = np.asarray(d, dtype=float)
    out = np.zeros(np.broadcast(r1, r2, d).shape, dtype=float)

    separate = d >= (r1 + r2)
    contained = d <= np.abs(r1 - r2)
    partial = ~separate & ~contained

    out[contained] = np.pi * np.minimum(r1, r2)[contained] ** 2
    if np.any(partial):
        a, b, c = r1[partial] if r1.ndim else r1, r2[partial] if r2.ndim else r2, d[partial]
        a = np.broadcast_to(a, c.shape)
        b = np.broadcast_to(b, c.shape)
        t1 = a ** 2 * np.arccos(np.clip((c ** 2 + a ** 2 - b ** 2) / (2 * c * a), -1.0, 1.0))
        t2 = b ** 2 * np.arccos(np.clip((c ** 2 + b ** 2 - a ** 2) / (2 * c * b), -1.0, 1.0))
        t3 = 0.5 * np.sqrt(np.maximum(0.0, (-c + a + b) * (c + a - b) * (c - a + b) * (c + a + b)))
        out[partial] = t1 + t2 - t3
    return out


def visible_fraction(sun_km, primary_km, face_km, primary_radius_km):
    """Fraction of the solar disk still visible from ``face_km`` at each step."""
    sun_rel = sun_km - face_km
    pri_rel = primary_km - face_km
    d_sun = np.linalg.norm(sun_rel, axis=1)
    d_pri = np.linalg.norm(pri_rel, axis=1)

    ang_sun = np.arcsin(np.clip(SUN_RADIUS_KM / d_sun, -1.0, 1.0))
    ang_pri = np.arcsin(np.clip(primary_radius_km / d_pri, -1.0, 1.0))

    cos_sep = np.einsum("ij,ij->i", sun_rel, pri_rel) / (d_sun * d_pri)
    sep = np.arccos(np.clip(cos_sep, -1.0, 1.0))

    covered = circle_overlap_area(ang_pri, ang_sun, sep) / (np.pi * ang_sun ** 2)
    # The primary can only occult if it is nearer than the Sun.
    covered = np.where(d_pri < d_sun, covered, 0.0)
    return np.clip(1.0 - covered, 0.0, 1.0), d_sun, sep, ang_sun, ang_pri


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("output_dir")
    ap.add_argument("--primary", default="EARTH")
    ap.add_argument("--primary-radius-km", type=float, default=None)
    ap.add_argument("--body-radius-km", type=float, default=None,
                    help="target body radius [km]; default = mean |face| from mesh.ply")
    ap.add_argument("--lat", type=float, default=0.0)
    ap.add_argument("--lon", type=float, default=0.0)
    ap.add_argument("--steps", default=None,
                    help="step window 'a:b' to gate on (default: every partial step)")
    ap.add_argument("--tol", type=float, default=0.016,
                    help="max allowed |engine - truth| in units of full flux")
    args = ap.parse_args(argv)

    out = args.output_dir
    rp = args.primary_radius_km
    if rp is None:
        rp = PRIMARY_RADIUS_KM.get(args.primary.upper())
        if rp is None:
            raise SystemExit("unknown primary %r" % args.primary)

    utc, sun, pri = read_body_positions(os.path.join(out, "body_positions.csv"))
    n = len(utc)

    lat, lon = read_ply_face_latlon(os.path.join(out, "mesh.ply"))
    dlon = np.abs(((lon - args.lon + 180.0) % 360.0) - 180.0)
    face = int(np.argmin((lat - args.lat) ** 2 + dlon ** 2))

    # Probe-face position in the body frame, read from the mesh itself (no
    # assumed body radius).  Global icosphere meshes are built in KM, DEM/site
    # meshes in METERS -- the same |c| > 10000 discriminator the engine uses
    # (kmToNativeScale) converts to km here.
    cent = read_ply_face_centroids(os.path.join(out, "mesh.ply"))
    fc = cent[face]
    r_native = float(np.linalg.norm(fc))
    fc_km = fc / 1000.0 if r_native > 10000.0 else fc
    r_body = float(np.linalg.norm(fc_km))
    if args.body_radius_km is not None:      # explicit override for cross-checks
        fc_km = fc_km / r_body * args.body_radius_km
        r_body = args.body_radius_km
    face_dir = fc_km / np.linalg.norm(fc_km)
    face_km = np.broadcast_to(fc_km, (n, 3))

    vis, d_sun, sep, ang_sun, ang_pri = visible_fraction(sun, pri, face_km, rp)

    full = SOLAR_CONSTANT / (d_sun / KM_PER_AU) ** 2   # W/m^2 at normal incidence

    ds = read_series(out, "direct_solar", face, n)
    # cos(incidence) at the probe face, from the run's own geometry.
    sun_rel = sun - face_km
    sun_dir = sun_rel / np.linalg.norm(sun_rel, axis=1)[:, None]
    cos_inc = np.clip(np.einsum("ij,ij->i", sun_dir, np.broadcast_to(face_dir, (n, 3))), 0.0, 1.0)
    denom = full * cos_inc
    eng = np.where(denom > 0.0, ds / np.where(denom > 0.0, denom, 1.0), np.nan)

    print("run          : %s" % out)
    print("probe face   : %d at lat %.3f lon %.3f" % (face, lat[face], lon[face]))
    print("primary      : %s R=%.1f km ; body radius used %.1f km" %
          (args.primary.upper(), rp, r_body))
    print("partial steps: %s" % np.array2string(np.flatnonzero((vis > 1e-9) & (vis < 1.0 - 1e-9))))
    print()
    print("step  utc                    sep/deg  ang_pri  ang_sun  vis_true  engine   diff")
    if args.steps:
        a, b = (int(x) for x in args.steps.split(":"))
        window = range(a, min(b, n))
    else:
        window = np.flatnonzero(vis < 1.0 - 1e-9)

    worst = 0.0
    for i in window:
        d = eng[i] - vis[i]
        if np.isfinite(d) and 0.0 < vis[i] < 1.0:
            worst = max(worst, abs(d))
        print("%4d  %-22s %7.4f  %7.4f  %7.4f  %8.4f  %7.4f  %+7.4f" %
              (i, utc[i], np.degrees(sep[i]), np.degrees(ang_pri[i]),
               np.degrees(ang_sun[i]), vis[i], eng[i], d))

    print()
    print("worst |engine - truth| over PARTIAL steps in window: %.4f (tol %.4f)" %
          (worst, args.tol))
    ok = worst <= args.tol
    print("VERDICT: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
