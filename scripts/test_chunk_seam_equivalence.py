#!/usr/bin/env python3
"""Regression test: a chunked run must equal the unchunked run at EVERY frame.

The chunked workflow (``scripts/run_chunked_simulation.py``) hands a thermal
state from one chunk to the next through ``thermal.final_state_file`` /
``thermal.initial_state_file`` with ``thermal.checkpoint_lookahead_steps: 1``,
so chunk k+1 starts at exactly the epoch where chunk k stopped exporting. The
contract is that the concatenated frames are the single-shot frames. Before
the warm-start seam fix the first frame of every restored chunk was not: the solver
filed the restored column's raw surface node under output index 0 instead of
re-solving the massless surface at that frame's own epoch, the way it does for
every other frame (fix/frame-forcing-epoch). Measured 0.31 K on Justitia.

What is compared
----------------
Both arms start from the SAME warm state S0 (a short spin-up run, banked with
lookahead 1), so neither arm involves an equilibration seed and the only
difference between them is the seam:

    straight  : S0 -> N steps
    split     : S0 -> N/2 steps (lookahead 1, banks S1) ; S1 -> N/2 steps

Every temperature frame of the split arm is compared against the straight
arm's frame at the same epoch. The epoch grids themselves (time_steps.txt) are
compared too, so a seam that is off by one step fails loudly instead of
comparing the wrong frames.

The tolerance is set by the checkpoint, not by the physics: a PTMT state is
FLOAT32, so the restored column differs from the live double-precision column
by up to half an ulp of float32 (~1e-5 K at 200 K). Frames are float32 too.

Solver paths exercised: CPU SIMD batch (the default) and CPU scalar
(``--no-simd-batch``). The CPU streamed driver shares the scalar per-face
integrator (runThermalModelWithMatchedTimes -> Model::run) but is selected by a
memory threshold, so it is NOT separately exercised here. The GPU persistent
solver is not exercised either (see the warm-start seam fix for its status).

Usage
-----
    python3 scripts/test_chunk_seam_equivalence.py \\
        --exe build/bin/ptm_thermal_sim --workdir <scratch dir outside /tmp>

Exit status 0 = every compared frame within tolerance on every path; 1 = not.
Stdlib only.
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import re
import shutil
import struct
import subprocess
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src", "python"))

# 2 rotations of a 5 h asteroid at 96 samples/rotation, as in
# examples/quickstart_asteroid_diurnal.yaml, on a 320-face icosphere. chi is
# NONZERO on purpose: with chi = 0 the conductivity is constant and a seam that
# restores the wrong k(T) would be invisible.
STEPS_PER_ROT = 96
PERIOD_H = 5.0
DT_S = PERIOD_H * 3600.0 / STEPS_PER_ROT

BASE = {
    "simulation": {
        "start_time": "2026-01-01T00:00:00",
        "body": "Seam Test Asteroid",
    },
    "mesh": {"resolution": 2, "import_file": "", "scale_factor": 1.0,
             "use_topographic": False},
    "thermal": {
        "run_thermal_mesh": True,
        "threads": 4,
        "use_gpu": False,
        "compute_statistics": False,
        "enable_shadow_flux": False,
        "geothermal_heat_flow": -1.0,
    },
    "thermophysical": {"use_default_albedo": False},
    "radiation": {"enabled": False},
    "illumination": {"include_primary_reflection": False,
                     "use_batch_processing": True, "use_albedo_map": False,
                     "albedo_maps_directory": ""},
    "output": {"format": "binary", "vtk_mode": "none"},
    "generic_body": {
        "name": "Seam Test Asteroid",
        "config_file": "",
        "ephemeris_source": "keplerian",
        "shape": {"type": "spherical", "radius_km": 5.0, "mesh_scale": 1.0},
        "orbit": {"semi_major_axis_au": 2.7, "eccentricity": 0.1,
                  "inclination_deg": 8.0, "lon_ascending_node_deg": 80.0,
                  "arg_perihelion_deg": 50.0, "mean_anomaly_deg": 0.0,
                  "epoch": "2000-01-01T12:00:00"},
        "spin": {"period_hours": PERIOD_H, "pole_ra_deg": 30.0,
                 "pole_dec_deg": 60.0, "prime_meridian_deg": 0.0},
        "surface": {"albedo": 0.05, "emissivity": 0.95, "ks": 0.004,
                    "kd": 0.02, "rhos": 1300.0, "rhod": 1800.0, "H": 0.02,
                    "cp0": 800.0, "Qb": 0.018, "chi": 2.7},
    },
}


def read_frame(path):
    with open(path, "rb") as fh:
        raw = fh.read()
    n = struct.unpack("<i", raw[:4])[0]
    if 4 + 4 * n != len(raw):
        raise SystemExit("%s: header says %d faces but file is %d bytes"
                         % (path, n, len(raw)))
    return struct.unpack("<%df" % n, raw[4:])


def frames(rundir):
    try:
        from ptm.readers import sorted_frame_files
    except ImportError:          # older tree: fall back to a numeric sort here
        sorted_frame_files = None
    if sorted_frame_files is not None:
        files = sorted_frame_files(os.path.join(rundir, "temperature"), "temperature")
    else:
        files = sorted(glob.glob(os.path.join(rundir, "temperature",
                                              "temperature_*.bin")),
                       key=lambda p: int(re.search(r"_(\d+)\.bin$", p).group(1)))
    if not files:
        raise SystemExit("no temperature frames under " + rundir)
    return [read_frame(f) for f in files]


def epochs(rundir):
    """ET of every exported frame, from time_steps.txt (one per line)."""
    path = os.path.join(rundir, "time_steps.txt")
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            out.append(line.split()[0])
    return out


def run(exe, cfg, workdir, label, extra):
    rundir = os.path.join(workdir, label)
    shutil.rmtree(rundir, ignore_errors=True)
    os.makedirs(rundir)
    cfg = copy.deepcopy(cfg)
    cfg["output"]["directory"] = rundir
    cpath = os.path.join(workdir, label + ".yaml")
    with open(cpath, "w") as fh:
        json.dump(cfg, fh, indent=1)        # JSON is valid YAML
    log = os.path.join(workdir, label + ".log")
    with open(log, "w") as lf:
        # cwd = repository root: the default leap-second kernel path is repo-relative.
        rc = subprocess.call([exe, "--config", cpath] + extra, stdout=lf, cwd=_REPO,
                             stderr=subprocess.STDOUT)
    if rc != 0:
        raise SystemExit("%s: engine rc=%d (log %s)" % (label, rc, log))
    return rundir


def window(offset_steps, n, init_state, final_state, lookahead, eq_orbits=0.0):
    c = copy.deepcopy(BASE)
    s = c["simulation"]
    s["start_time_offset_seconds"] = float(offset_steps * DT_S)
    s["duration_days"] = n * DT_S / 86400.0
    s["timesteps"] = int(n)
    th = c["thermal"]
    th["equilibration_orbits"] = float(eq_orbits)
    if init_state:
        th["initial_state_file"] = init_state
    th["final_state_file"] = final_state
    th["checkpoint_lookahead_steps"] = int(lookahead)
    return c


def check_path(exe, workdir, path_label, extra, n, tol):
    wd = os.path.join(workdir, path_label)
    os.makedirs(wd, exist_ok=True)
    s0 = os.path.join(wd, "S0.ptmt")
    n0 = STEPS_PER_ROT // 2                  # spin-up window ends mid-rotation
    run(exe, window(0, n0, None, s0, 1, eq_orbits=20.0), wd, "spinup", extra)

    straight = run(exe, window(n0, n, s0, os.path.join(wd, "straight_end.ptmt"), 1),
                   wd, "straight", extra)
    h = n // 2
    s1 = os.path.join(wd, "S1.ptmt")
    a = run(exe, window(n0, h, s0, s1, 1), wd, "split_a", extra)
    b = run(exe, window(n0 + h, n - h, s1, os.path.join(wd, "split_b_end.ptmt"), 1),
            wd, "split_b", extra)

    ref, ref_t = frames(straight), epochs(straight)
    got, got_t = frames(a) + frames(b), epochs(a) + epochs(b)
    if len(ref) != n or len(got) != n:
        print("  FAIL [%s]: frame counts straight=%d split=%d, expected %d"
              % (path_label, len(ref), len(got), n))
        return 1
    if ref_t != got_t:
        bad = [i for i, (x, y) in enumerate(zip(ref_t, got_t)) if x != y]
        print("  FAIL [%s]: epoch grids differ at %d frame(s), first %d: %s vs %s"
              % (path_label, len(bad), bad[0], ref_t[bad[0]], got_t[bad[0]]))
        return 1
    worst, worst_at, per_frame = 0.0, (0, 0), []
    for i, (fr, fg) in enumerate(zip(ref, got)):
        d = max(abs(x - y) for x, y in zip(fr, fg))
        per_frame.append(d)
        if d > worst:
            worst, worst_at = d, (i, max(range(len(fr)), key=lambda f: abs(fr[f] - fg[f])))
    seam = per_frame[h]
    rest = max(per_frame[:h] + per_frame[h + 1:])
    print("  [%s] %d frames x %d faces: max |dT| = %.3g K (frame %d, face %d); "
          "seam frame %d: %.3g K; all other frames: %.3g K"
          % (path_label, n, len(ref[0]), worst, worst_at[0], worst_at[1], h, seam, rest))
    if worst > tol:
        print("  FAIL [%s]: exceeds %.1e K" % (path_label, tol))
        return 1
    print("  PASS [%s]" % path_label)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--exe", default=os.path.join(_REPO, "build", "bin", "ptm_thermal_sim"))
    ap.add_argument("--workdir", required=True,
                    help="scratch directory for the runs (not /tmp if you intend to keep them)")
    ap.add_argument("--steps", type=int, default=STEPS_PER_ROT,
                    help="frames per run (default: one rotation)")
    ap.add_argument("--tol", type=float, default=1e-4,
                    help="kelvin; float32 checkpoint + frame quantization is ~1e-5 K")
    ap.add_argument("--paths", default="simd,scalar")
    args = ap.parse_args()
    if args.steps < 4 or args.steps % 2:
        raise SystemExit("--steps must be even and >= 4")
    exe = os.path.abspath(args.exe)
    os.makedirs(args.workdir, exist_ok=True)
    print("=== chunk-seam equivalence: %s ===" % exe)
    rc = 0
    for p in args.paths.split(","):
        extra = {"simd": [], "scalar": ["--no-simd-batch"]}[p]
        rc |= check_path(exe, os.path.abspath(args.workdir), p, extra, args.steps, args.tol)
    print("RESULT:", "PASS" if rc == 0 else "FAIL")
    return rc


if __name__ == "__main__":
    sys.exit(main())
