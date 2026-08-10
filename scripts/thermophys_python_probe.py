#!/usr/bin/env python3
"""Probe the Python API's thermophysical channels for silence.

Question: with ``ptm.Sweep``, which ways of naming a thermophysical property
reach the solver, and which ones fail quietly? Runs the engine for the routes
that are supposed to work, and only constructs (never runs) the ones being
checked for validation behavior.

    python3 scripts/thermophys_python_probe.py --exe build-cpu/bin/ptm_thermal_sim
"""

from __future__ import annotations

import argparse
import hashlib
import struct
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "python"))

import numpy as np  # noqa: E402

from ptm import Sweep  # noqa: E402
from ptm.engine_config import unknown_keys  # noqa: E402


def tstat(rundir, which="max"):
    data = (Path(rundir) / "statistics" / f"temperature_{which}.bin").read_bytes()
    count = struct.unpack("<I", data[:4])[0]
    arr = (np.frombuffer(data[4:], dtype="<f4") if len(data) == 4 + count * 4
           else np.frombuffer(data, dtype="<f4"))
    return arr.astype(np.float64)


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def check_construct(label, **kwargs):
    """Construct a Sweep and report whether it complained."""
    try:
        s = Sweep(**kwargs)
    except Exception as exc:  # noqa: BLE001 - reporting, not handling
        print(f"  {label:52s} REFUSED  {type(exc).__name__}: "
              f"{str(exc).splitlines()[0][:90]}")
        return None
    # does anything downstream name the key?
    cfg = s.points[0].config()
    dead = unknown_keys(cfg)
    verdict = "ACCEPTED"
    if dead:
        verdict += f"  (ptm.engine_config.unknown_keys would say: {dead})"
    print(f"  {label:52s} {verdict}")
    return s


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default="build-cpu/bin/ptm_thermal_sim")
    ap.add_argument("--outdir", default=str(Path.home() / "projects/thermophys-audit/python"))
    ap.add_argument("--no-run", action="store_true")
    args = ap.parse_args(argv)

    exe = Path(args.exe)
    if not exe.is_absolute():
        exe = REPO / exe
    base = REPO / "examples" / "quickstart_moon_diurnal.yaml"
    out = Path(args.outdir)

    print("\n== A. what the Sweep constructor accepts (no engine launched)")
    check_construct("grid={'thermal_inertia': [...]}  (named)",
                    base_config=base, grid={"thermal_inertia": [100.0]},
                    outdir=out / "ctor", exe=exe)
    check_construct("grid={'thermal_inertai': [...]}  (typo, no dot)",
                    base_config=base, grid={"thermal_inertai": [100.0]},
                    outdir=out / "ctor", exe=exe)
    check_construct("grid={'thermophysical.ks_override': [...]}",
                    base_config=base,
                    grid={"thermophysical.ks_override": [0.00296]},
                    outdir=out / "ctor", exe=exe)
    check_construct("grid={'thermophysical.ks_overide': [...]}  (typo, dotted)",
                    base_config=base,
                    grid={"thermophysical.ks_overide": [0.00296]},
                    outdir=out / "ctor", exe=exe)
    check_construct("grid={'thermal.equilibration_orbit': [...]} (typo, dotted)",
                    base_config=base,
                    grid={"thermal.equilibration_orbit": [4.0]},
                    outdir=out / "ctor", exe=exe)
    check_construct("overrides={'thermophysical.ks_overide': ...} (typo)",
                    base_config=base, grid={"thermophysical.albedo_override": [0.2]},
                    overrides={"thermophysical.ks_overide": 0.00296},
                    outdir=out / "ctor", exe=exe)
    check_construct("grid={'thermophysical.ks': [...]}  (dead plain key)",
                    base_config=base, grid={"thermophysical.ks": [0.00296]},
                    outdir=out / "ctor", exe=exe)
    check_construct("both TI conventions on one grid",
                    base_config=base,
                    grid={"thermal_inertia": [100.0],
                          "constant_thermal_inertia": [100.0]},
                    outdir=out / "ctor", exe=exe)

    if args.no_run:
        return 0

    print("\n== B. does the named 'thermal_inertia' axis reach the solver?")
    sweep = Sweep(base, grid={"thermal_inertia": [100.0, 200.0, 400.0]},
                  outdir=out / "ti", exe=exe)
    print(f"  route: {sweep.route}")
    for p in sweep.points:
        print("  cmd:", " ".join(str(c) for c in p.command()[-4:]))
    sweep.run()
    tmax = [tstat(p.dir).max() for p in sweep.points]
    tmin = [tstat(p.dir, "min").min() for p in sweep.points]
    amp = [float(np.median(tstat(p.dir) - tstat(p.dir, "min")))
           for p in sweep.points]
    md5s = [md5(Path(p.dir) / "statistics" / "temperature_max.bin")
            for p in sweep.points]
    for ti, a, b, c, m in zip([100, 200, 400], tmax, tmin, amp, md5s):
        print(f"  TI={ti:4d}  Tmax(field max)={a:8.3f}  Tmin(field min)={b:8.3f} "
              f" median amplitude={c:8.3f}  md5={m[:12]}")
    print(f"  distinct md5s: {len(set(md5s))} of {len(md5s)}")

    print("\n== C. does the dotted config key route reach the solver?")
    sweep2 = Sweep(base, grid={"thermophysical.ks_override": [0.00074, 0.00296]},
                   outdir=out / "ksov", exe=exe)
    sweep2.run()
    md5s2 = [md5(Path(p.dir) / "statistics" / "temperature_max.bin")
             for p in sweep2.points]
    amp2 = [float(np.median(tstat(p.dir) - tstat(p.dir, "min")))
            for p in sweep2.points]
    for ks, a, m in zip([0.00074, 0.00296], amp2, md5s2):
        print(f"  ks={ks:<9g} median amplitude={a:8.3f}  md5={m[:12]}")
    print(f"  distinct md5s: {len(set(md5s2))} of {len(md5s2)}")

    print("\n== D. a dead key on a grid axis: does anything say so?")
    sweep3 = Sweep(base, grid={"thermophysical.ks": [0.00074, 0.00296]},
                   outdir=out / "deadks", exe=exe)
    sweep3.run()
    md5s3 = [md5(Path(p.dir) / "statistics" / "temperature_max.bin")
             for p in sweep3.points]
    print(f"  distinct md5s: {len(set(md5s3))} of {len(md5s3)}  {md5s3[0][:12]}")
    log = (Path(sweep3.points[1].dir) / "engine.log").read_text()
    for line in log.splitlines():
        if "no effect" in line or "WARNING" in line:
            print("  engine log:", line[:150])

    print("\n== E. a typo'd key outside thermophysical: does anything say so?")
    sweep4 = Sweep(base, grid={"thermal.equilibration_orbit": [1.0, 4.0]},
                   outdir=out / "typo", exe=exe)
    sweep4.run()
    md5s4 = [md5(Path(p.dir) / "statistics" / "temperature_max.bin")
             for p in sweep4.points]
    print(f"  distinct md5s: {len(set(md5s4))} of {len(md5s4)}  {md5s4[0][:12]}")
    log = (Path(sweep4.points[1].dir) / "engine.log").read_text()
    hits = [ln for ln in log.splitlines()
            if "equilibration_orbit" in ln or "no effect" in ln]
    print(f"  engine log lines naming the key: {len(hits)}")

    print("\n== F. geothermal_heat_flow through the Python override channel")
    sweep5 = Sweep(base, grid={"thermal.geothermal_heat_flow": [-1.0, 5.0]},
                   outdir=out / "qb", exe=exe)
    sweep5.run()
    md5s5 = [md5(Path(p.dir) / "statistics" / "temperature_max.bin")
             for p in sweep5.points]
    print(f"  distinct md5s: {len(set(md5s5))} of {len(md5s5)}  {md5s5[0][:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
