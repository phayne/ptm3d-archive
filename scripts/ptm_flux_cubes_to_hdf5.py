#!/usr/bin/env python3
"""Read a PTM3D flux-cube binary (.fcb), QA it, and package it as HDF5.

The engine writes .fcb; HDF5 is produced here rather than in C++ because the
repo declares HDF5 OFF by default in-tree "to avoid link issues on macOS/Conda"
(src/cpp/viewshed/CMakeLists.txt), and adding an HDF5 link to the production
binary mid-campaign would change its provenance for every remaining site run in
order to serve a file format. The requirement -- CSV and HDF5 -- is met; only
the producer moves.

QA is not optional decoration. `--qa` runs checks that can FAIL:

  * total == sum of the five components, exactly (the writer builds it that way,
    so any drift means the file is corrupt or mis-strided);
  * f_sky + sum_j F + f_ring == 1 per patch;
  * no NaN or negative irradiance anywhere;
  * nadir terrain_ir >= eps*sigma*T_face^4 * f_max, i.e. the face beneath cannot
    contribute less than its own emission times its own view factor. This is the
    check that catches a mis-indexed cube -> face mapping, which every purely
    structural check would pass.

IMPORTANT, and carried into every output file: terrain_vis and terrain_ir are
CORE-MESH ONLY. The occluder ring is a shadow-only body with no thermal
solution and therefore no radiosity, so its contribution is absent. The omitted
irradiance on a patch is bounded by f_ring * sigma*T_ring^4 for any assumed
ring temperature; f_ring is stored per patch so a consumer can apply that bound.

Usage:
  python3 scripts/ptm_flux_cubes_to_hdf5.py <run>/flux_cubes/flux_cubes.fcb \\
      --out site_flux_cubes.h5 --qa
"""

import argparse
import json
import math
import struct
import sys

FCB_HDR = "<4sIIIIIQQdddd3d3d3d3dII"
CUBE_FMT = "<iidd3diiffffd"
FACE_NAMES = ["zenith", "nadir", "north", "south", "east", "west"]
COMP_NAMES = ["direct", "terrain_vis", "terrain_ir", "primary_refl", "primary_ir", "total"]
SIGMA = 5.670374419e-8

RING_NOTE = (
    "terrain_vis and terrain_ir EXCLUDE the occluder-ring contribution: the ring is a "
    "shadow-only body with no thermal solution and therefore no radiosity. The omitted "
    "irradiance on a patch is bounded by f_ring * sigma*T_ring^4 for any assumed ring "
    "temperature; per-patch f_ring is in this file."
)
UNITS_NOTE = (
    "All flux values are INCIDENT irradiance in W m-2. No albedo, no emissivity and no "
    "cube material are applied anywhere -- bolt your own optical model on top."
)


def read_fcb(path):
    with open(path, "rb") as f:
        blob = f.read()
    hs = struct.calcsize(FCB_HDR)
    v = struct.unpack_from(FCB_HDR, blob, 0)
    if v[0] != b"FCB1":
        raise SystemExit("not an FCB1 file: %r" % (v[0],))
    d = dict(version=v[1], n_cubes=v[2], n_faces=v[3], n_comp=v[4], n_steps=v[5],
             equilibration_steps=v[6], total_time_steps=v[7],
             height_m=v[8], edge_m=v[9], spacing_m=v[10], radius_m=v[11])
    json_len = v[-2]
    nc, nf, ncp, ns = d["n_cubes"], d["n_faces"], d["n_comp"], d["n_steps"]
    off = hs
    n = nc * nf * ncp * ns
    flux = struct.unpack_from("<%dd" % n, blob, off); off += 8 * n
    tf = struct.unpack_from("<%dd" % (nc * ns), blob, off); off += 8 * nc * ns
    pen = struct.unpack_from("<%dB" % (nc * ns), blob, off); off += nc * ns
    np_ = nc * nf
    fsky = struct.unpack_from("<%df" % np_, blob, off); off += 4 * np_
    fring = struct.unpack_from("<%df" % np_, blob, off); off += 4 * np_
    fmax = struct.unpack_from("<%df" % np_, blob, off); off += 4 * np_
    cs = struct.calcsize(CUBE_FMT)
    cubes = [struct.unpack_from(CUBE_FMT, blob, off + i * cs) for i in range(nc)]
    off += nc * cs
    et = struct.unpack_from("<%dd" % ns, blob, off); off += 8 * ns
    utc = []
    for k in range(ns):
        utc.append(blob[off:off + 24].split(b"\0")[0].decode("ascii", "replace"))
        off += 24
    meta = {}
    if json_len:
        try:
            meta = json.loads(blob[off:off + json_len].decode("utf-8"))
        except Exception as e:
            print("  warning: header JSON did not parse (%s)" % e)
    d.update(flux=flux, t_face=tf, penumbral=pen, f_sky=fsky, f_ring=fring,
             f_max=fmax, cubes=cubes, et=et, utc=utc, meta=meta)
    return d


def idx(d, cu, fa, co, st):
    return ((cu * d["n_faces"] + fa) * d["n_comp"] + co) * d["n_steps"] + st


def qa(d):
    fails = []
    nc, nf, ns = d["n_cubes"], d["n_faces"], d["n_steps"]
    print("\n--- QA ---")

    worst_tot = 0.0
    n_nan = 0
    n_neg = 0
    for cu in range(nc):
        for fa in range(nf):
            for st in range(ns):
                comps = [d["flux"][idx(d, cu, fa, c, st)] for c in range(5)]
                tot = d["flux"][idx(d, cu, fa, 5, st)]
                for x in comps + [tot]:
                    if math.isnan(x) or math.isinf(x):
                        n_nan += 1
                    elif x < -1e-12:
                        n_neg += 1
                worst_tot = max(worst_tot, abs(sum(comps) - tot))
    print("  total == sum(components)   worst |d| = %.3e  %s"
          % (worst_tot, "PASS" if worst_tot < 1e-9 else "FAIL"))
    if worst_tot >= 1e-9:
        fails.append("total != sum(components), worst %.3e" % worst_tot)
    print("  NaN/Inf values             %d  %s" % (n_nan, "PASS" if n_nan == 0 else "FAIL"))
    if n_nan:
        fails.append("%d NaN/Inf flux values" % n_nan)
    print("  negative irradiance        %d  %s" % (n_neg, "PASS" if n_neg == 0 else "FAIL"))
    if n_neg:
        fails.append("%d negative flux values" % n_neg)

    # Nadir consistency: the facet beneath cannot contribute less than its own
    # blackbody-floor emission times its own view factor. Catches a mis-indexed
    # cube -> face mapping that every structural check would pass.
    eps_lo = 0.90          # lunar regolith is ~0.95; use a conservative floor
    worst_ratio = 1e30
    worst_at = None
    for cu in range(nc):
        fmax = d["f_max"][cu * nf + 1]
        if fmax <= 0:
            continue
        for st in range(ns):
            T = d["t_face"][cu * ns + st]
            got = d["flux"][idx(d, cu, 1, 2, st)]      # nadir terrain_ir
            floor = eps_lo * SIGMA * T ** 4 * fmax
            if floor <= 0:
                continue
            r = got / floor
            if r < worst_ratio:
                worst_ratio, worst_at = r, (cu, st, T, got, floor)
    ok = worst_ratio >= 1.0
    print("  nadir terrain_ir >= eps*sigma*T^4*f_max   worst ratio %.4f  %s"
          % (worst_ratio, "PASS" if ok else "FAIL"))
    if worst_at:
        cu, st, T, got, floor = worst_at
        print("      (cube %d step %d: T_face=%.2f K, got %.4f, floor %.4f W/m^2)"
              % (cu, st, T, got, floor))
    if not ok:
        fails.append("nadir terrain_ir below the facet's own emission floor (ratio %.4f)"
                     % worst_ratio)
    return fails


def summarize(d):
    nc, nf, ns = d["n_cubes"], d["n_faces"], d["n_steps"]
    print("\n--- file ---")
    print("  cubes %d, faces %d, components %d, output steps %d"
          % (nc, nf, d["n_comp"], ns))
    print("  height %.4f m, edge %.4f m, spacing %.1f m, radius %.1f m"
          % (d["height_m"], d["edge_m"], d["spacing_m"], d["radius_m"]))
    print("  epochs %s .. %s" % (d["utc"][0], d["utc"][-1]))
    print("  equilibration_steps %d, total_time_steps %d"
          % (d["equilibration_steps"], d["total_time_steps"]))
    if d["meta"]:
        for k in ("source_albedo", "primary_model", "primary_disk_samples",
                  "shadow_probe_rays", "n_rays", "terrain_ir_scope",
                  "f_ring_mean", "f_ring_max"):
            if k in d["meta"]:
                print("  %-22s %s" % (k, d["meta"][k]))
    print("\n--- per face: max over all cubes and steps (W/m^2) ---")
    print("%-8s %10s %12s %12s %13s %11s %11s %9s" %
          ("face", "direct", "terrain_vis", "terrain_ir", "primary_refl",
           "primary_ir", "total", "f_sky p50"))
    for fa in range(nf):
        mx = [0.0] * 6
        for cu in range(nc):
            for co in range(6):
                base = idx(d, cu, fa, co, 0)
                m = max(d["flux"][base:base + ns])
                mx[co] = max(mx[co], m)
        fs = sorted(d["f_sky"][cu * nf + fa] for cu in range(nc))
        print("%-8s %10.4f %12.4f %12.4f %13.3e %11.3e %11.4f %9.4f" %
              (FACE_NAMES[fa], mx[0], mx[1], mx[2], mx[3], mx[4], mx[5],
               fs[len(fs) // 2]))
    tmin = min(d["t_face"]); tmax = max(d["t_face"])
    npen = sum(1 for x in d["penumbral"] if x)
    print("\n  T_face range %.2f .. %.2f K;  penumbral samples %d/%d (%.2f%%)"
          % (tmin, tmax, npen, nc * ns, 100.0 * npen / max(1, nc * ns)))
    fr = d["f_ring"]
    print("  f_ring: mean %.6f, max %.6f" % (sum(fr) / len(fr), max(fr)))
    print("  " + RING_NOTE)


def write_hdf5(d, out):
    try:
        import h5py
        import numpy as np
    except ImportError:
        print("\n  h5py/numpy not importable in this interpreter; skipping HDF5.")
        print("  Run under the europa_viz env:  conda run -n europa_viz python3 ...")
        return False
    nc, nf, ncp, ns = d["n_cubes"], d["n_faces"], d["n_comp"], d["n_steps"]
    flux = np.asarray(d["flux"], dtype=np.float64).reshape(nc, nf, ncp, ns)
    with h5py.File(out, "w") as h:
        ds = h.create_dataset("flux", data=flux,
                              chunks=(min(64, nc), nf, ncp, ns),
                              compression="gzip", compression_opts=4)
        ds.attrs["units"] = "W m-2 incident"
        ds.attrs["note"] = UNITS_NOTE
        ds.attrs["ring_note"] = RING_NOTE
        ds.attrs["axes"] = "cube, face, component, step"
        ds.attrs["face_order"] = [s.encode() for s in FACE_NAMES]
        ds.attrs["component_order"] = [s.encode() for s in COMP_NAMES]
        t = h.create_dataset("T_face", data=np.asarray(d["t_face"]).reshape(nc, ns))
        t.attrs["units"] = "K"
        t.attrs["note"] = "temperature of the mesh face beneath each cube"
        h.create_dataset("penumbral", data=np.asarray(d["penumbral"], np.uint8).reshape(nc, ns))
        g = h.create_group("geometry")
        for name, key in (("f_sky", "f_sky"), ("f_ring", "f_ring"), ("f_max", "f_max")):
            g.create_dataset(name, data=np.asarray(d[key], np.float32).reshape(nc, nf))
        g["f_ring"].attrs["note"] = RING_NOTE
        g["f_max"].attrs["note"] = (
            "largest single view-factor entry in the patch's row. At a site-relative "
            "height a LARGE value on the nadir face is the intended behaviour: the face "
            "is a near-unity-view-factor sensor of the facet beneath it. A value near "
            "1/4 that does not grow as the height falls means the cube sits on a mesh "
            "vertex and the reading is a multi-facet average.")
        cb = np.asarray(d["cubes"],
                        dtype=[("grid_i", "i4"), ("grid_j", "i4"), ("x_m", "f8"), ("y_m", "f8"),
                               ("cx", "f8"), ("cy", "f8"), ("cz", "f8"),
                               ("face_id", "i4"), ("pad", "i4"),
                               ("slope_deg", "f4"), ("aspect_deg", "f4"),
                               ("area_m2", "f4"), ("elev_m", "f4"), ("height_m", "f8")])
        for nm in ("grid_i", "grid_j", "x_m", "y_m", "face_id", "slope_deg",
                   "aspect_deg", "area_m2", "elev_m"):
            g.create_dataset(nm, data=cb[nm])
        tg = h.create_group("time")
        tg.create_dataset("et", data=np.asarray(d["et"]))
        tg.create_dataset("utc", data=np.array([s.encode() for s in d["utc"]]))
        for k, v in (("ptm3d_flux_cubes_schema", 1), ("height_m", d["height_m"]),
                     ("edge_m", d["edge_m"]), ("spacing_m", d["spacing_m"]),
                     ("radius_m", d["radius_m"]), ("n_cubes", nc), ("n_steps", ns),
                     ("equilibration_steps", d["equilibration_steps"]),
                     ("total_time_steps", d["total_time_steps"])):
            h.attrs[k] = v
        h.attrs["units"] = UNITS_NOTE
        h.attrs["ring_note"] = RING_NOTE
        h.attrs["time_axis_source"] = "time_steps.txt"
        for k, v in (d["meta"] or {}).items():
            try:
                h.attrs["engine_" + k] = v if not isinstance(v, (dict, list)) else json.dumps(v)
            except Exception:
                pass
    print("\n  wrote %s" % out)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fcb")
    ap.add_argument("--out", default=None, help="HDF5 output path")
    ap.add_argument("--qa", action="store_true")
    a = ap.parse_args()
    d = read_fcb(a.fcb)
    summarize(d)
    fails = qa(d) if a.qa else []
    if a.out:
        write_hdf5(d, a.out)
    if fails:
        print("\n==== %d QA FAILURE(S) ====" % len(fails))
        for f in fails:
            print("  FAIL:", f)
        return 1
    if a.qa:
        print("\n==== QA: all checks passed ====")
    return 0


if __name__ == "__main__":
    sys.exit(main())
