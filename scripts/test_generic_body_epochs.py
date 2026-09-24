#!/usr/bin/env python3
"""Engine test: generic_body spin.prime_meridian_deg, spin.epoch and orbit.epoch
move the Sun exactly as specified (BUGS.md PTM-663).

Until 2026-09-24 ``generic_body.spin.prime_meridian_deg`` and
``generic_body.orbit.epoch`` were parsed and dropped: every generic body rotated
as W(t) = omega * t_J2000 and its Kepler mean anomaly was applied at J2000.
A Justitia TWG run had to rotate its mesh to impose DAMIT's rotational phase.

Each arm runs ``ptm_thermal_sim`` for two output epochs on a small icosphere
with a Keplerian orbit and reads the Sun vector the engine itself recorded in
``body_positions.csv`` (body-fixed frame, km). Nothing here re-implements the
engine's frame; the checks are differences between runs, plus one absolute
Kepler identity:

  A  prime_meridian_deg = 30     -> sub-solar longitude moves by -30 deg
  B  spin.epoch = J2000 + P/4    -> sub-solar longitude moves by +90 deg
                                    (W(t) = omega*(t - epoch))
  C  (spin.epoch E1, W0 = w)  ==  (J2000, W0 = w - 360*(E1 - J2000)/P)
                                 -> same Sun vector (the epoch is linear)
  D  orbit.epoch = the first frame's epoch, mean_anomaly_deg = 0, e = 0.3
                                 -> |r| at frame 0 = a(1 - e) (perihelion)
  E  refusals: a 'UTC'-suffixed epoch, an unknown ephemeris_source and an
     unknown shape.type exit non-zero; an unknown key is named in a WARNING.

Also checks the null case: a config that spells out the J2000 defaults
('2000-01-01T12:00:00', prime_meridian_deg 0) gives the same Sun vectors as one
that omits them.

Usage:  python3 scripts/test_generic_body_epochs.py --exe build/bin/ptm_thermal_sim \\
            --workdir <scratch dir>
Stdlib only. Exit 0 = pass.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import shutil
import subprocess
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PERIOD_H = 5.0
A_AU = 2.7
AU_KM = 1.495978707e8

BASE = {
    "simulation": {"start_time": "2030-06-15T03:00:00", "duration_days": 2 * 0.1 / 24.0,
                   "timesteps": 2, "body": "Epoch Test Asteroid"},
    "mesh": {"resolution": 1, "import_file": "", "scale_factor": 1.0, "use_topographic": False},
    "thermal": {"run_thermal_mesh": True, "threads": 2, "use_gpu": False,
                "equilibration_orbits": 0.0, "compute_statistics": False,
                "geothermal_heat_flow": -1.0},
    "thermophysical": {"use_default_albedo": False},
    "radiation": {"enabled": False},
    "illumination": {"include_primary_reflection": False, "use_batch_processing": True,
                     "use_albedo_map": False, "albedo_maps_directory": ""},
    "output": {"format": "binary", "vtk_mode": "none", "skip_time_series": True},
    "generic_body": {
        "name": "Epoch Test Asteroid", "config_file": "", "ephemeris_source": "keplerian",
        "shape": {"type": "spherical", "radius_km": 5.0, "mesh_scale": 1.0},
        "orbit": {"semi_major_axis_au": A_AU, "eccentricity": 0.1, "inclination_deg": 8.0,
                  "lon_ascending_node_deg": 80.0, "arg_perihelion_deg": 50.0,
                  "mean_anomaly_deg": 20.0},
        "spin": {"period_hours": PERIOD_H, "pole_ra_deg": 30.0, "pole_dec_deg": 60.0},
        "surface": {"albedo": 0.05, "emissivity": 0.95, "ks": 0.004, "kd": 0.004,
                    "rhos": 1500.0, "rhod": 1500.0, "H": 0.02, "cp0": 800.0,
                    "Qb": 0.0, "chi": 0.0},
    },
}


def run(exe, cfg, wd, label, expect_ok=True):
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
        rc = subprocess.call([exe, "--config", cp], cwd=_REPO, stdout=lf,
                             stderr=subprocess.STDOUT)
    text = open(log).read()
    if expect_ok and rc != 0:
        raise SystemExit("%s: engine rc=%d (log %s)" % (label, rc, log))
    return out, rc, text


def sun_vectors(rundir):
    with open(os.path.join(rundir, "body_positions.csv")) as fh:
        rows = list(csv.DictReader(fh))
    return [(float(r["sun_x"]), float(r["sun_y"]), float(r["sun_z"])) for r in rows]


def lon_deg(v):
    return math.degrees(math.atan2(v[1], v[0]))


def wrap(d):
    return (d + 180.0) % 360.0 - 180.0


def first_et(rundir):
    with open(os.path.join(rundir, "time_steps.txt")) as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                return float(line.split()[0])
    raise SystemExit("no epochs in time_steps.txt")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--exe", default=os.path.join(_REPO, "build", "bin", "ptm_thermal_sim"))
    ap.add_argument("--workdir", required=True)
    args = ap.parse_args()
    exe = os.path.abspath(args.exe)
    wd = os.path.abspath(args.workdir)
    os.makedirs(wd, exist_ok=True)
    fails = []

    def check(ok, what):
        print(("  ok    " if ok else "  FAIL  ") + what)
        if not ok:
            fails.append(what)

    ref_dir, _, _ = run(exe, BASE, wd, "ref")
    ref = sun_vectors(ref_dir)

    # Null case: spelling out the J2000 defaults changes nothing.
    c = copy.deepcopy(BASE)
    c["generic_body"]["orbit"]["epoch"] = "2000-01-01T12:00:00"
    c["generic_body"]["spin"]["prime_meridian_deg"] = 0.0
    c["generic_body"]["spin"]["epoch"] = "2000-01-01T12:00:00"
    d0, _, _ = run(exe, c, wd, "explicit_defaults")
    check(sun_vectors(d0) == ref, "explicit J2000 defaults == omitted keys (identical Sun vectors)")

    # A: prime meridian.
    c = copy.deepcopy(BASE)
    c["generic_body"]["spin"]["prime_meridian_deg"] = 30.0
    da, _, _ = run(exe, c, wd, "pm30")
    va = sun_vectors(da)
    dl = [wrap(lon_deg(v) - lon_deg(r)) for v, r in zip(va, ref)]
    dz = [abs(v[2] - r[2]) / math.hypot(*r) for v, r in zip(va, ref)]
    check(all(abs(x + 30.0) < 1e-3 for x in dl),
          "A prime_meridian_deg 30 -> sub-solar longitude %s deg (expect -30)"
          % ["%.5f" % x for x in dl])
    check(max(dz) < 1e-5, "A sub-solar latitude unchanged (|dz|/r = %.2g)" % max(dz))

    # B: spin epoch a quarter period after J2000 -> W shifts by -90 deg ->
    # the Sun's body-fixed longitude by +90 deg.
    c = copy.deepcopy(BASE)
    c["generic_body"]["spin"]["epoch"] = "JD %.12f" % (2451545.0 + 0.25 * PERIOD_H / 24.0)
    db, _, _ = run(exe, c, wd, "spin_epoch_quarter")
    dl = [wrap(lon_deg(v) - lon_deg(r)) for v, r in zip(sun_vectors(db), ref)]
    check(all(abs(x - 90.0) < 1e-3 for x in dl),
          "B spin.epoch = J2000 + P/4 -> sub-solar longitude %s deg (expect +90)"
          % ["%.5f" % x for x in dl])

    # C: epoch/W0 equivalence, with a calendar-form epoch on the TDB scale.
    e1 = "1984-07-01T06:00:00 TDB"
    jd1 = 2445882.75                      # 1984-07-01 06:00 TDB
    w = 17.0
    rot = 360.0 * ((jd1 - 2451545.0) * 24.0 / PERIOD_H)
    c1 = copy.deepcopy(BASE)
    c1["generic_body"]["spin"].update({"epoch": e1, "prime_meridian_deg": w})
    c2 = copy.deepcopy(BASE)
    c2["generic_body"]["spin"]["prime_meridian_deg"] = math.fmod(w - rot, 360.0)
    v1 = sun_vectors(run(exe, c1, wd, "damit_style_epoch")[0])
    v2 = sun_vectors(run(exe, c2, wd, "equivalent_w0_at_j2000")[0])
    dl = [abs(wrap(lon_deg(x) - lon_deg(y))) for x, y in zip(v1, v2)]
    # The two arms differ only in rounding (a 1.3e5-rotation phase folded two
    # ways); 1e-3 deg is ~2e-5 of a rotation.
    check(max(dl) < 1e-3, "C (spin.epoch 1984, W0) == (J2000, W0 - omega*dt): max |dlon| = %.2g deg"
          % max(dl))
    moved = max(abs(wrap(lon_deg(x) - lon_deg(r))) for x, r in zip(v1, ref))
    check(moved > 1.0, "C is not vacuous: those arms differ from the reference by %.3g deg" % moved)

    # D: orbit epoch. M0 = 0 at the first frame's own epoch -> perihelion.
    et0 = first_et(ref_dir)
    c = copy.deepcopy(BASE)
    ecc = 0.3
    c["generic_body"]["orbit"].update({"eccentricity": ecc, "mean_anomaly_deg": 0.0,
                                       "epoch": "JD %.12f" % (2451545.0 + et0 / 86400.0)})
    vd = sun_vectors(run(exe, c, wd, "orbit_epoch_perihelion")[0])
    r0 = math.hypot(*vd[0])
    expect = A_AU * (1.0 - ecc) * AU_KM
    # body_positions.csv carries 6 significant digits.
    check(abs(r0 / expect - 1.0) < 2e-6,
          "D orbit.epoch honored: |r(frame 0)| = %.6g km, a(1-e) = %.6g km (rel %.1e)"
          % (r0, expect, r0 / expect - 1.0))
    c["generic_body"]["orbit"]["epoch"] = "2000-01-01T12:00:00"
    vj = sun_vectors(run(exe, c, wd, "orbit_epoch_j2000")[0])
    check(abs(math.hypot(*vj[0]) / expect - 1.0) > 1e-3,
          "D control: the same elements at a J2000 epoch are NOT at perihelion then")

    # E: refusals and warnings.
    c = copy.deepcopy(BASE)
    c["generic_body"]["orbit"]["epoch"] = "2030-06-15T03:00:00 UTC"
    _, rc, text = run(exe, c, wd, "refuse_utc", expect_ok=False)
    check(rc != 0 and "time scale other than TDB" in text, "E a UTC-suffixed epoch is refused")
    c = copy.deepcopy(BASE)
    c["generic_body"]["ephemeris_source"] = "horizons"
    _, rc, text = run(exe, c, wd, "refuse_ephemeris", expect_ok=False)
    check(rc != 0 and "ephemeris_source" in text, "E an unknown ephemeris_source is refused")
    c = copy.deepcopy(BASE)
    c["generic_body"]["shape"]["type"] = "sphere"
    _, rc, text = run(exe, c, wd, "refuse_shape", expect_ok=False)
    check(rc != 0 and "shape.type" in text, "E an unknown shape.type is refused")
    c = copy.deepcopy(BASE)
    c["generic_body"]["spin"]["pole_ra"] = 30.0          # typo for pole_ra_deg
    _, rc, text = run(exe, c, wd, "warn_typo", expect_ok=True)
    check("generic_body.spin.pole_ra" in text and "read by nothing" in text,
          "E a misspelled key is named in a WARNING")

    print("RESULT:", "PASS" if not fails else "FAIL (%d)" % len(fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
