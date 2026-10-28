#!/usr/bin/env python3
"""Eclipse diagnostic for PTM3D satellite runs (Europa/Jupiter, Moon/Earth, ...).

Answers one question with no rebuild and no GPU: *does the run's direct-solar
field actually go dark when the target passes through its primary's umbra?*

It compares three things over the run's own recorded ephemeris
(``body_positions.csv``, written by every run):

1. TRUTH   -- the geometric umbra predicate, evaluated in consistent km:
              Jupiter (or Earth) lies between the target and the Sun and the
              shadow-axis impact parameter b is smaller than the primary radius.
2. CODE    -- the predicate as the illumination model evaluates it. The engine
              converts the primary POSITION km->m (``kmToMeters``) but leaves
              ``primary_radius_`` in km, so the effective test is b*1000 < R_p.
              Pass ``--code-scale 1.0`` to model the fixed engine.
3. MEASURED -- the direct-solar (and temperature) time series of the face
              closest to a requested lat/lon, read straight out of the run.

Usage
-----
    python scripts/ptm_eclipse_diag.py OUTPUT_DIR [--lat 0 --lon 0]
                                       [--primary-radius-km 69911]
                                       [--plot output/eclipse_diag/europa.png]

Exit code 0 always; the verdict is printed and (optionally) plotted.
"""

import argparse
import csv
import os
import struct
import sys

import numpy as np


def _frame_file(directory, name, t):
    """Frame ``t`` of ``name`` under either naming.

    Current engines pad the index to >= 6 digits; runs written before used 4.
    """
    for w in range(6, 13):
        p = os.path.join(str(directory), "%s_%0*d.bin" % (name, w, t))
        if os.path.exists(p):
            return p
    return os.path.join(str(directory), "%s_%04d.bin" % (name, t))


# Primary radii exactly as the engine defines them (src/cpp/constants/constants.cpp).
PRIMARY_RADIUS_KM = {
    "JUPITER": 69911.0,
    "EARTH": 6371.0,
    "SATURN": 58232.0,
    "MARS": 3389.5,
}


def read_body_positions(path):
    """Return (utc[list], sun_km[N,3], primary_km[N,3])."""
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit("empty body_positions.csv: %s" % path)
    utc = [r["time"] for r in rows]
    sun = np.array([[float(r["sun_x"]), float(r["sun_y"]), float(r["sun_z"])]
                    for r in rows])
    pri = np.array([[float(r["primary_x"]), float(r["primary_y"]),
                     float(r["primary_z"])] for r in rows])
    return utc, sun, pri


def umbra_geometry(sun_km, primary_km):
    """Impact parameter b [km] of the primary's center off the target->Sun line,
    and the sunward projection (positive => primary is between target and Sun)."""
    d = sun_km / np.linalg.norm(sun_km, axis=1)[:, None]
    proj = np.einsum("ij,ij->i", primary_km, d)
    perp = primary_km - proj[:, None] * d
    return np.linalg.norm(perp, axis=1), proj


def read_ply_face_latlon(path):
    """Face latitude/longitude [deg] from a PTM3D binary_little_endian mesh.ply.

    Layout: element vertex (6 doubles) then element face
    (uchar count + count*int32 + float latitude + float longitude).
    """
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
        fh.seek(n_vert * 6 * 8, os.SEEK_CUR)  # skip x,y,z,nx,ny,nz doubles
        lat = np.empty(n_face, dtype=np.float64)
        lon = np.empty(n_face, dtype=np.float64)
        blob = fh.read()
    off = 0
    for i in range(n_face):
        cnt = blob[off]
        off += 1
        off += 4 * cnt
        lat[i], lon[i] = struct.unpack_from("<ff", blob, off)
        off += 8
    return lat, lon


def read_series(out_dir, component, face_idx, n_steps):
    """Per-timestep value of one face for a component directory."""
    vals = np.full(n_steps, np.nan)
    for t in range(n_steps):
        path = _frame_file(os.path.join(out_dir, component), component, t)
        if not os.path.exists(path):
            continue
        with open(path, "rb") as fh:
            n = struct.unpack("<i", fh.read(4))[0]
            if face_idx >= n:
                raise SystemExit("face %d out of range (%d faces)" % (face_idx, n))
            fh.seek(4 * face_idx, os.SEEK_CUR)
            vals[t] = struct.unpack("<f", fh.read(4))[0]
    return vals


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("output_dir")
    ap.add_argument("--primary", default="JUPITER",
                    help="primary body name (JUPITER, EARTH, SATURN, MARS)")
    ap.add_argument("--primary-radius-km", type=float, default=None,
                    help="override the primary radius [km]")
    ap.add_argument("--lat", type=float, default=0.0, help="probe latitude [deg]")
    ap.add_argument("--lon", type=float, default=0.0,
                    help="probe longitude [deg]; 0 = sub-primary point")
    ap.add_argument("--code-scale", type=float, default=1000.0,
                    help="km->m inflation the engine applies to the sphere CENTER "
                         "but not the radius. 1000 = buggy engine, 1 = fixed engine.")
    ap.add_argument("--plot", default=None, help="write a PNG here")
    ap.add_argument("--label", default=None, help="plot title suffix")
    args = ap.parse_args(argv)

    out = args.output_dir
    rp = args.primary_radius_km
    if rp is None:
        rp = PRIMARY_RADIUS_KM.get(args.primary.upper())
        if rp is None:
            raise SystemExit("unknown primary %r; pass --primary-radius-km"
                             % args.primary)

    utc, sun, pri = read_body_positions(os.path.join(out, "body_positions.csv"))
    n = len(utc)
    b, proj = umbra_geometry(sun, pri)

    truth = (b < rp) & (proj > 0.0)
    code = (b * args.code_scale < rp) & (proj > 0.0)

    print("run           : %s" % out)
    print("steps         : %d   (%s -> %s)" % (n, utc[0], utc[-1]))
    print("primary       : %s  R = %.1f km" % (args.primary.upper(), rp))
    print("impact param b: min %.1f km (all steps), min-at-conjunction %.1f km"
          % (b.min(), b[proj > 0].min() if (proj > 0).any() else float("nan")))
    print("TRUTH  umbra  : %d / %d steps eclipsed" % (truth.sum(), n))
    for i in np.flatnonzero(truth):
        print("    step %4d  %s  b=%9.1f km" % (i, utc[i], b[i]))
    print("ENGINE (scale=%g): %d / %d steps eclipsed"
          % (args.code_scale, code.sum(), n))

    face = None
    ply = os.path.join(out, "mesh.ply")
    if os.path.exists(ply):
        lat, lon = read_ply_face_latlon(ply)
        dlon = np.abs(((lon - args.lon + 180.0) % 360.0) - 180.0)
        face = int(np.argmin((lat - args.lat) ** 2 + dlon ** 2))
        print("probe face    : %d at lat %.3f lon %.3f (requested %.1f/%.1f)"
              % (face, lat[face], lon[face], args.lat, args.lon))

    # Verdict uses the DEEP umbra (b < R_p/2) only. The engine traces from each
    # FACE, not the body center, so near the umbra limb a face on the far limb can
    # legitimately still be lit -- for the Moon the body radius is 38% of the umbra
    # radius, so the limb steps are genuinely ambiguous and must not drive the call.
    deep = (b < 0.5 * rp) & (proj > 0.0)

    ds = temp = None
    if face is not None and os.path.isdir(os.path.join(out, "direct_solar")):
        ds = read_series(out, "direct_solar", face, n)
        print("direct_solar  : min %.4f  max %.4f W/m^2" % (np.nanmin(ds), np.nanmax(ds)))
        if truth.any():
            print("   at TRUE-eclipse steps: %s"
                  % np.array2string(ds[truth], precision=3))
            lit = ds[~truth]
            if deep.any():
                print("   VERDICT (deep umbra, b < R/2, %d steps): %s"
                      % (deep.sum(),
                         "ECLIPSE PRESENT (direct solar < 5%% of the lit peak)"
                         if np.nanmax(ds[deep]) < 0.05 * np.nanmax(lit)
                         else "ECLIPSE ABSENT (no dip where the umbra is deepest)"))
            else:
                print("   VERDICT: inconclusive — no deep-umbra sample in this run")
    if face is not None and os.path.isdir(os.path.join(out, "temperature")):
        temp = read_series(out, "temperature", face, n)
        print("temperature   : min %.3f  max %.3f K" % (np.nanmin(temp), np.nanmax(temp)))
        if truth.any():
            print("   at TRUE-eclipse steps: %s"
                  % np.array2string(temp[truth], precision=3))

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(os.path.dirname(os.path.abspath(args.plot)), exist_ok=True)
        hours = np.arange(n, dtype=float)
        # elapsed hours from time_steps.txt if available, else step index
        ts = os.path.join(out, "time_steps.txt")
        if os.path.exists(ts):
            secs = []
            for line in open(ts):
                if line.startswith("#"):
                    continue
                parts = line.split()
                if parts:
                    try:
                        secs.append(float(parts[0]))
                    except ValueError:
                        pass
            if len(secs) == n:
                hours = np.array(secs) / 3600.0

        nax = 2 + (temp is not None)
        fig, axes = plt.subplots(nax, 1, figsize=(9, 2.4 * nax), sharex=True)
        ax = axes[0]
        ax.plot(hours, b, color="#3b6ea5", lw=1.6, label="impact parameter b")
        ax.axhline(rp, color="#b5443a", ls="--", lw=1.2,
                   label="%s radius %.0f km" % (args.primary.title(), rp))
        for i in np.flatnonzero(truth):
            ax.axvspan(hours[i] - 0.3, hours[i] + 0.3, color="#b5443a", alpha=0.15, lw=0)
        ax.set_ylabel("b [km]")
        ax.set_yscale("log")
        ax.legend(loc="upper right", fontsize=8, frameon=False)
        ax.set_title("Eclipse diagnostic%s" % ("" if not args.label else " — " + args.label))

        ax = axes[1]
        if ds is not None:
            ax.plot(hours, ds, color="#2f6f4e", lw=1.6)
        for i in np.flatnonzero(truth):
            ax.axvspan(hours[i] - 0.3, hours[i] + 0.3, color="#b5443a", alpha=0.15, lw=0)
        ax.set_ylabel("direct solar\n[W m$^{-2}$]")

        if temp is not None:
            ax = axes[2]
            ax.plot(hours, temp, color="#8a5a2b", lw=1.6)
            for i in np.flatnonzero(truth):
                ax.axvspan(hours[i] - 0.3, hours[i] + 0.3, color="#b5443a", alpha=0.15, lw=0)
            ax.set_ylabel("T [K]")
        axes[-1].set_xlabel("elapsed time [h]")
        for a in axes:
            a.spines["top"].set_visible(False)
            a.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=140)
        print("plot          : %s" % args.plot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
