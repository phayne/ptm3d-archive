#!/usr/bin/env python3
"""Regression test: radiation exchange that is ~zero must change ~nothing.

Convex-shape drift. The coupled CPU run solves the whole mesh on the SIMD-batch
integrator and then re-integrates every face carrying ANY indirect flux (the
default selection threshold is 0 K). Until 2026-09-24 that re-integration used
the SCALAR integrator, whose flux sampling lags by half a substep, so a
selected face moved by the two solvers' difference whatever its indirect flux:
0.26 K on Justitia TWG Case 3, where the exchange is <= 2.4e-5 W/m^2.

The test body is the shipped DAMIT Justitia shape, which is strictly convex:
its view factors are zero except for a few near-coplanar facet pairs, so a
handful of faces ARE selected (the test refuses to pass if none is -- that
would be vacuous) while the physical effect of the exchange is below 1e-4 K.

Both arms pin ``thermal.seed_anchor: warm``. With radiation on, the engine
falls back from the derived seed anchor to the warm one on EVERY face
(``seed_forcing_radiation_coupled``), a spin-up choice that leaves ~1 mK after
the equilibration here and is not the exchange path; pinning it makes the two
arms start from the same state, so the comparison isolates the re-thermal.

Pass: every frame of every face within ``--tol`` (default 1e-4 K) between
radiation on and radiation off, on the default CPU (SIMD-batch) solver.

Usage:  python3 scripts/test_rethermal_zero_exchange.py \\
            --exe build/bin/ptm_thermal_sim --workdir <scratch dir>
Stdlib only. Exit 0 = pass.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shutil
import struct
import subprocess
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MESH = os.path.join(_REPO, "data", "horizons_cache", "bodies", "269_Justitia_(A887_SA)",
                    "shape.obj")
PERIOD_H = 33.1296
STEPS = 144

BASE = {
    "simulation": {"start_time": "2034-10-30T00:00:00", "duration_days": PERIOD_H / 24.0,
                   "timesteps": STEPS, "body": "Rethermal Test Justitia"},
    "mesh": {"resolution": 5, "import_file": MESH, "scale_factor": 1.0},
    "thermal": {"run_thermal_mesh": True, "threads": 8, "use_gpu": False,
                "equilibration_orbits": 20.0, "compute_statistics": False,
                "geothermal_heat_flow": -1.0, "seed_anchor": "warm"},
    "thermophysical": {"use_default_albedo": False, "incidence_angle_albedo": False},
    "radiation": {"enabled": False},
    "illumination": {"include_primary_reflection": False, "use_batch_processing": True,
                     "albedo_maps_directory": ""},
    "output": {"format": "binary", "vtk_mode": "none"},
    "generic_body": {
        "name": "Rethermal Test Justitia", "config_file": "", "ephemeris_source": "keplerian",
        "shape": {"type": "custom_mesh", "radius_km": 25.364, "mesh_scale": 1.0,
                  "mesh_file": MESH},
        "orbit": {"semi_major_axis_au": 2.613, "eccentricity": 0.0, "inclination_deg": 0.0,
                  "lon_ascending_node_deg": 0.0, "arg_perihelion_deg": 0.0,
                  "mean_anomaly_deg": 146.0},
        "spin": {"period_hours": PERIOD_H, "pole_ra_deg": 270.0, "pole_dec_deg": 66.56},
        "surface": {"albedo": 0.03, "emissivity": 0.95, "ks": 0.01, "kd": 0.01,
                    "rhos": 1500.0, "rhod": 1500.0, "H": 0.05, "cp0": 666.67,
                    "Qb": 0.0, "chi": 0.0},
    },
}


def run(exe, cfg, wd, label, extra=()):
    out = os.path.join(wd, label)
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    cfg = copy.deepcopy(cfg)
    cfg["output"]["directory"] = out
    cp = os.path.join(wd, label + ".yaml")
    with open(cp, "w") as fh:
        json.dump(cfg, fh, indent=1)              # JSON is valid YAML
    log = os.path.join(wd, label + ".log")
    with open(log, "w") as lf:
        rc = subprocess.call([exe, "--config", cp] + list(extra), cwd=_REPO, stdout=lf,
                             stderr=subprocess.STDOUT)
    if rc != 0:
        raise SystemExit("%s: engine rc=%d (log %s)" % (label, rc, log))
    return out, open(log).read()


def frames(rundir):
    d = os.path.join(rundir, "temperature")
    names = sorted((n for n in os.listdir(d) if re.fullmatch(r"temperature_\d+\.bin", n)),
                   key=lambda n: int(n[len("temperature_"):-4]))
    out = []
    for n in names:
        raw = open(os.path.join(d, n), "rb").read()
        k = struct.unpack("<i", raw[:4])[0]
        out.append(struct.unpack("<%df" % k, raw[4:]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--exe", default=os.path.join(_REPO, "build", "bin", "ptm_thermal_sim"))
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--tol", type=float, default=1e-4)
    args = ap.parse_args()
    exe = os.path.abspath(args.exe)
    wd = os.path.abspath(args.workdir)
    os.makedirs(wd, exist_ok=True)
    if not os.path.exists(MESH):
        raise SystemExit("test mesh missing: " + MESH)

    off, _ = run(exe, BASE, wd, "radoff")
    on_cfg = copy.deepcopy(BASE)
    on_cfg["radiation"] = {"enabled": True, "use_gpu": False, "use_hmatrix": False,
                           "view_factor_file": os.path.join(wd, "justitia.vf"),
                           "max_coupling_iterations": 10, "coupling_temp_threshold": 0.5}
    on, log = run(exe, on_cfg, wd, "radon")

    sel = [int(m) for m in re.findall(r"Re-running thermal model for (\d+) faces", log)]
    path = re.findall(r"RE-THERMAL \(radiation coupling\): (\S+)", log)
    print("  re-thermal selections %s, path %s" % (sel, sorted(set(path))))
    if not sel or max(sel) == 0:
        print("  FAIL: no face was selected for re-thermal -- the test would be vacuous")
        return 1

    A, B = frames(on), frames(off)
    if len(A) != STEPS or len(B) != STEPS:
        print("  FAIL: frame counts %d / %d, expected %d" % (len(A), len(B), STEPS))
        return 1
    worst, at = 0.0, None
    for t, (a, b) in enumerate(zip(A, B)):
        for f, (x, y) in enumerate(zip(a, b)):
            if abs(x - y) > worst:
                worst, at = abs(x - y), (t, f)
    print("  radiation ON vs OFF: max |dT| = %.3g K over %d frames x %d faces (at frame %s face %s)"
          % (worst, len(A), len(A[0]), at[0] if at else "-", at[1] if at else "-"))
    ok = worst <= args.tol
    print("RESULT:", "PASS" if ok else "FAIL (exceeds %.1e K)" % args.tol)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
