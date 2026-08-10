#!/usr/bin/env python3
"""PTM-41 / PTM-44 acceptance harness for the primary-body ("Jupiter-shine",
"Earthshine") flux paths.

Two independent checks, both on a finished run's own outputs.

PHASE-LAW GATE (PTM-41)
-----------------------
Compares the engine's ``primary_reflected`` against the closed-form
Lambert-sphere law

    E = p * F * sin^2(alpha) * Phi_L(a) * <cos theta>,
    p = (2/3) A_bond,   Phi_L(a) = [sin a + (pi - a) cos a] / pi

*through the ratio to* ``primary_ir``:

    primary_reflected     p * F * Phi_L(a) * sin^2(alpha)
    ----------------- =  ---------------------------------
       primary_ir            J * Omega/pi

Both terms carry the SAME ``<cos theta>`` and the same terrain visibility, so
the ratio cancels them.  The gate therefore needs no face normals, is immune to
PTM-44, and isolates the photometry alone.  That matters: on a global icosphere
the face normal is only approximately the radial direction, and a gate that had
to assume it would be testing the mesh instead of the physics.

A body orbiting its primary sweeps the full phase range every orbit (Europa:
0 to 180 deg in 1.77 d), so ONE diurnal run exercises the whole phase curve.

QUANTIZATION GATE (PTM-44)
--------------------------
Given two runs of the same config, one from a build BEFORE the fix and one
after, reports the distribution of ``primary_ir_after / primary_ir_before``.
The pre-fix code decided the primary's visibility with a single ray to the disk
center and zeroed the flux at ``cos theta <= 0``, so that ratio can only be 0,
1, or +inf (0 -> nonzero) before the fix and takes intermediate values after it.
The count of intermediate values IS the defect's footprint.

Usage
-----
    python scripts/ptm_primary_flux_check.py phase  RUN_DIR [--primary JUPITER]
    python scripts/ptm_primary_flux_check.py compare BEFORE_DIR AFTER_DIR

Exit codes: 0 pass, 1 fail, 3 refused (not enough usable samples).
"""

import argparse
import csv
import os
import struct
import sys

import numpy as np

# constants.cpp
SOLAR_CONSTANT = 1361.0
KM_PER_AU = 149597870.7
PRIMARY = {
    #            radius km,  bond albedo, IR flux W/m^2
    "JUPITER": (69911.0, 0.343, 13.8),
    "EARTH":   (6371.0,  0.306, 240.0),
    "SATURN":  (58232.0, 0.342, 4.62),
    "MARS":    (3389.5,  0.250, 110.3),
}


def read_bin(path):
    """One PTM3D per-face binary: int32 count then count float32."""
    with open(path, "rb") as fh:
        n = struct.unpack("<i", fh.read(4))[0]
        return np.frombuffer(fh.read(4 * n), dtype="<f4").astype(np.float64)


def read_body_positions(path):
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit("empty body_positions.csv: %s" % path)
    utc = [r["time"] for r in rows]
    sun = np.array([[float(r["sun_x"]), float(r["sun_y"]), float(r["sun_z"])] for r in rows])
    pri = np.array([[float(r["primary_x"]), float(r["primary_y"]), float(r["primary_z"])]
                    for r in rows])
    return utc, sun, pri


def read_mesh(path):
    """Face centroids and OUTWARD geometric face normals, mesh's own units.

    The normal is the vertex cross product, flipped to point away from the body
    center.  On a convex global icosphere that is exactly the normal the engine
    uses; on a DEM patch it is not necessarily so, which is why the horizon
    census below is only offered for global meshes.
    """
    with open(path, "rb") as fh:
        header = b""
        while b"end_header" not in header:
            chunk = fh.readline()
            if not chunk:
                raise SystemExit("mesh.ply: no end_header")
            header += chunk
        text = header.decode("ascii", "replace")
        n_vert = n_face = 0
        for line in text.splitlines():
            if line.startswith("element vertex"):
                n_vert = int(line.split()[2])
            elif line.startswith("element face"):
                n_face = int(line.split()[2])
        verts = np.frombuffer(fh.read(n_vert * 6 * 8), dtype="<f8").reshape(n_vert, 6)[:, :3]
        blob = fh.read()
    cent = np.empty((n_face, 3))
    norm = np.empty((n_face, 3))
    off = 0
    for i in range(n_face):
        cnt = blob[off]
        off += 1
        idx = struct.unpack_from("<%di" % cnt, blob, off)
        off += 4 * cnt + 8
        p3 = verts[list(idx)]
        cent[i] = p3.mean(axis=0)
        nv = np.cross(p3[1] - p3[0], p3[2] - p3[0])
        nl = np.linalg.norm(nv)
        nv = nv / nl if nl > 0 else np.array([0.0, 0.0, 1.0])
        if nv @ cent[i] < 0:
            nv = -nv
        norm[i] = nv
    return cent, norm


def horizon_mean_g(t):
    t = np.clip(t, -1.0, 1.0)
    s = np.sqrt(np.maximum(0.0, 1.0 - t * t))
    return (t * (np.pi / 2) + t * t * s + t * np.arcsin(t) + (2.0 / 3.0) * s ** 3) / np.pi


def disk_mean_cos_above_horizon(c0, alpha):
    """Mirror of ptm::disk::diskMeanCosAboveHorizon."""
    c = np.clip(c0, -1.0, 1.0)
    h = np.arcsin(c)
    out = np.where(h >= alpha, c, np.where(h <= -alpha, 0.0,
                   np.sin(alpha * horizon_mean_g(h / np.maximum(alpha, 1e-300)))))
    return out


def read_centroids(path):
    """Face centroids in the mesh's own units."""
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
    cent = np.empty((n_face, 3))
    off = 0
    for i in range(n_face):
        cnt = blob[off]
        off += 1
        idx = struct.unpack_from("<%di" % cnt, blob, off)
        off += 4 * cnt + 8          # skip the per-face lat/lon floats
        cent[i] = verts[list(idx)].mean(axis=0)
    return cent


def lambert_phase(a):
    return (np.sin(a) + (np.pi - a) * np.cos(a)) / np.pi


def n_steps(run_dir, component):
    d = os.path.join(run_dir, component)
    return len([f for f in os.listdir(d) if f.endswith(".bin")])


def cmd_phase(args):
    run = args.run_dir
    radius, bond, ir_flux = PRIMARY[args.primary]

    utc, sun, pri = read_body_positions(os.path.join(run, "body_positions.csv"))
    cent = read_centroids(os.path.join(run, "mesh.ply"))
    # Mesh unit: km icosphere or meter DEM patch (the engine's own |c| > 10000
    # discriminator, illumination_model.cpp kmToNativeScale).
    if np.linalg.norm(cent[0]) > 10000.0:
        cent = cent / 1000.0

    nt = min(len(utc), n_steps(run, "primary_ir"), n_steps(run, "primary_reflected"))
    rows = []
    worst = 0.0
    used = 0
    for t in range(nt):
        ir = read_bin(os.path.join(run, "primary_ir", "primary_ir_%04d.bin" % t))
        rf = read_bin(os.path.join(run, "primary_reflected",
                                   "primary_reflected_%04d.bin" % t))
        rel = pri[t] - cent                                     # face -> primary
        dist = np.linalg.norm(rel, axis=1)
        pdir = rel / dist[:, None]
        sun_rel = sun[t] - pri[t]
        sdir = sun_rel / np.linalg.norm(sun_rel)
        cos_phase = np.clip(-(pdir @ sdir), -1.0, 1.0)
        a = np.arccos(cos_phase)

        sin_alpha = radius / dist
        alpha = np.arcsin(np.clip(sin_alpha, -1, 1))
        omega_over_pi = 2.0 * (1.0 - np.cos(alpha))
        F = SOLAR_CONSTANT / (np.linalg.norm(sun_rel) / KM_PER_AU) ** 2

        expect = ((2.0 / 3.0) * bond * F * lambert_phase(a) * sin_alpha ** 2
                  / (ir_flux * omega_over_pi))

        # Only faces that actually receive both terms carry information.
        m = (ir > 1e-6 * ir.max(initial=0.0)) & (ir > 0) & (expect > 1e-6)
        if not np.any(m):
            continue
        got = rf[m] / ir[m]
        err = np.abs(got - expect[m]) / np.maximum(expect[m], 1e-12)
        rows.append((t, utc[t], np.degrees(a[m]).mean(), expect[m].mean(),
                     got.mean(), err.max()))
        worst = max(worst, float(err.max()))
        used += 1

    print("  step  UTC                       phase[deg]  expect      engine      "
          "worst rel.err")
    stride = max(1, len(rows) // 24)
    for r in rows[::stride]:
        print("  %4d  %-24s  %8.3f  %10.6f  %10.6f  %10.3e"
              % (r[0], r[1][:24], r[2], r[3], r[4], r[5]))

    print("\n  primary                 : %s (R=%.1f km, A_bond=%.3f, J=%.2f W/m^2)"
          % (args.primary, radius, bond, ir_flux))
    print("  steps with usable faces : %d of %d" % (used, nt))
    print("  phase range covered     : %.1f to %.1f deg"
          % (min(r[2] for r in rows), max(r[2] for r in rows)))
    print("  worst relative error    : %.3e   (tol %.1e)" % (worst, args.tol))
    if used < args.min_steps:
        print("  VERDICT: REFUSED (only %d usable steps, need %d)" % (used, args.min_steps))
        return 3
    ok = worst <= args.tol
    print("  VERDICT: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def cmd_compare(args):
    before, after = args.before_dir, args.after_dir
    nt = min(n_steps(before, "primary_ir"), n_steps(after, "primary_ir"))
    tot = 0
    n_zero_to_pos = 0
    n_pos_to_zero = 0
    n_partial = 0            # 0 < ratio < 1 - eps : the disk is partly hidden
    n_same = 0
    max_gain = 0.0
    max_abs_dw = 0.0
    for t in range(nt):
        b = read_bin(os.path.join(before, "primary_ir", "primary_ir_%04d.bin" % t))
        a = read_bin(os.path.join(after, "primary_ir", "primary_ir_%04d.bin" % t))
        tot += b.size
        zb = b <= 0.0
        za = a <= 0.0
        n_zero_to_pos += int(np.count_nonzero(zb & ~za))
        n_pos_to_zero += int(np.count_nonzero(~zb & za))
        both = ~zb & ~za
        if np.any(both):
            r = a[both] / b[both]
            n_partial += int(np.count_nonzero(np.abs(r - 1.0) > 1e-6))
            n_same += int(np.count_nonzero(np.abs(r - 1.0) <= 1e-6))
            max_gain = max(max_gain, float(r.max()))
        max_abs_dw = max(max_abs_dw, float(np.abs(a - b).max()))

    print("  face-steps compared      : %d over %d steps" % (tot, nt))
    print("  identical (ratio == 1)   : %d (%.3f %%)" % (n_same, 100.0 * n_same / tot))
    print("  changed but both nonzero : %d (%.3f %%)" % (n_partial, 100.0 * n_partial / tot))
    print("  0 -> nonzero             : %d (%.3f %%)" % (n_zero_to_pos,
                                                         100.0 * n_zero_to_pos / tot))
    print("  nonzero -> 0             : %d (%.3f %%)" % (n_pos_to_zero,
                                                         100.0 * n_pos_to_zero / tot))
    print("  max |delta| primary_ir   : %.6g W/m^2" % max_abs_dw)
    print("  max after/before ratio   : %.4f" % max_gain)
    print("\n  Interpretation: the pre-fix path could only ever emit 0 or the full")
    print("  center-ray flux, so every count in the middle two rows is a face-step")
    print("  the all-or-nothing treatment was getting wrong.")
    return 0


def cmd_horizon(args):
    """Fermi F1 acceptance: no face whose analytic horizon clip A > 0 may be
    reported with primary_ir == 0, and the engine's `below horizon` census must
    equal the purely geometric count."""
    run = args.run_dir
    radius, _bond, _ir = PRIMARY[args.primary]
    utc, sun, pri = read_body_positions(os.path.join(run, "body_positions.csv"))
    cent, norm = read_mesh(os.path.join(run, "mesh.ply"))
    if np.linalg.norm(cent[0]) > 10000.0:
        cent = cent / 1000.0

    nt = min(len(utc), n_steps(run, "primary_ir"))
    bad = 0
    apos = 0
    below_geom_step0 = None
    worst_A = 0.0
    for t in range(nt):
        ir = read_bin(os.path.join(run, "primary_ir", "primary_ir_%04d.bin" % t))
        rel = pri[t] - cent
        dist = np.linalg.norm(rel, axis=1)
        pdir = rel / dist[:, None]
        c0 = np.einsum("ij,ij->i", norm, pdir)
        sin_alpha = np.clip(radius / dist, -1.0, 1.0)
        alpha = np.arcsin(sin_alpha)
        A = disk_mean_cos_above_horizon(c0, alpha)
        if t == 0:
            below_geom_step0 = int(np.count_nonzero(A <= 0.0))
        m = A > 0.0
        apos += int(np.count_nonzero(m))
        z = m & (ir <= 0.0)
        nz = int(np.count_nonzero(z))
        bad += nz
        if nz:
            worst_A = max(worst_A, float(A[z].max()))

    print("  steps                        : %d" % nt)
    print("  face-steps with A > 0        : %d" % apos)
    print("  ... of those, primary_ir == 0: %d   <-- Fermi F1 acceptance: must be 0"
          % bad)
    if bad:
        print("  largest analytic A among them: %.6g" % worst_A)
    print("  geometric 'below horizon' at step 0 (A <= 0): %d" % below_geom_step0)
    print("     compare with the engine census printed by PTM_PRIMARY_DISK_STATS=1")
    return 0 if bad == 0 else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("phase", help="PTM-41 Lambert-sphere phase-law gate")
    p.add_argument("run_dir")
    p.add_argument("--primary", default="JUPITER", choices=sorted(PRIMARY))
    p.add_argument("--tol", type=float, default=2e-3,
                   help="worst allowed relative error (default 2e-3; the engine "
                        "and this script use the same closed form, so the "
                        "residual is float32 output rounding plus the per-face "
                        "phase-angle spread)")
    p.add_argument("--min-steps", type=int, default=10)
    p.set_defaults(func=cmd_phase)

    c = sub.add_parser("compare", help="PTM-44 quantization footprint, two runs")
    c.add_argument("before_dir")
    c.add_argument("after_dir")
    c.set_defaults(func=cmd_compare)

    h = sub.add_parser("horizon", help="Fermi F1 acceptance: A>0 faces must not "
                                       "be zeroed, census == geometric count")
    h.add_argument("run_dir")
    h.add_argument("--primary", default="JUPITER", choices=sorted(PRIMARY))
    h.set_defaults(func=cmd_horizon)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
