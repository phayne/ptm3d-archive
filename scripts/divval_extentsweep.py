#!/usr/bin/env python3
"""Extent / resolution sweep for the Haworth PSR floor summer-max.

After the eq-convergence test showed the floor converges cold and the binding
constraint is the weak SUMMER-PEAK wall->floor flux, this decomposes the two
levers using EXISTING cached VFs (Phase A, just runs):

  R25       : r25 core (99,874 f, ~224 m/face), R=25 km  -> baseline
  R25_100m  : r25 core (400,973 f, ~112 m/face), R=25 km -> RESOLUTION (fixed extent)
  R32       : r32c core (99,665 f, ~287 m/face), R=32 km -> EXTENT (vs R25; coarser)

All eq=25 (converged), big occluder, NO two-pass (it is a ~1.5 K no-op for the
summer-max), so the floor-max trend cleanly reflects extent vs resolution.
Phase B (build R40/R50 cores+VFs) only if the Phase-A trend warrants.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from divval_orchestrate import (  # noqa: E402
    PTM, OUTROOT, FIGDIR, REPO, RUN_ENV, log, compare)

LAT, LON = -87.45, -2.1
BASE = REPO / "examples/divval_haworth_eq49_bigocc.yaml"
MESHDIR = REPO / "data/dem/validation/meshes"


def set_path(d, path, val):
    for k in path[:-1]:
        d = d.setdefault(k, {})
    d[path[-1]] = val


def run(tag, mesh, vf, occ):
    cfg = yaml.safe_load(BASE.read_text())
    out = OUTROOT / f"extsweep_haworth_{tag}"
    set_path(cfg, ("output", "directory"), str(out))
    set_path(cfg, ("mesh", "import_file"), str(MESHDIR / mesh))
    set_path(cfg, ("mesh", "occluder_file"), str(MESHDIR / occ))
    set_path(cfg, ("radiation", "view_factor_file"), str(MESHDIR / vf))
    set_path(cfg, ("thermal", "equilibration_orbits"), 25.0)
    set_path(cfg, ("thermal", "eq_indirect_flux_file"), "")   # no two-pass (no-op for max)
    cp = REPO / f"examples/extsweep_haworth_{tag}.yaml"
    cp.write_text(yaml.safe_dump(cfg, sort_keys=False))
    log(f"[{tag}] running -> {out.name}  mesh={mesh}")
    t0 = time.time()
    with open(f"/tmp/extsweep_haworth_{tag}.log", "w") as rl:
        try:
            subprocess.run([str(PTM), "--config", str(cp)], stdout=rl,
                           stderr=subprocess.STDOUT, timeout=4 * 3600,
                           cwd=str(REPO), env=RUN_ENV)
        except subprocess.TimeoutExpired:
            log(f"[{tag}] TIMEOUT")
    stats = out / "statistics" / "temperature_max.bin"
    if not (stats.exists() and stats.stat().st_size > 1000):
        log(f"[{tag}] FAILED - see /tmp/extsweep_haworth_{tag}.log")
        return {"error": "sim"}
    m = compare(f"extsweep_haworth_{tag}", LAT, LON, out)
    m["wall_s"] = round(time.time() - t0, 1)
    td = out / "temperature"
    if td.exists():
        for p in td.glob("*.bin"):
            p.unlink()
        try:
            td.rmdir()
        except OSError:
            pass
    return m


def main():
    rf = FIGDIR / "extsweep_haworth.json"
    res = json.loads(rf.read_text()) if rf.exists() else {}
    #     tag        mesh                              vf                                occluder
    plan = [
        ("R25",      "haworth_radcore_r25.ply",        "haworth_radcore_r25.vf",        "haworth_ring_25_200.ply"),
        ("R32",      "haworth_radcore_r32c.ply",       "haworth_radcore_r32c.vf",       "haworth_ring_32_200.ply"),
        ("R25_100m", "haworth_radcore_r25_100m.ply",   "haworth_radcore_r25_100m.vf",   "haworth_ring_25_200.ply"),
    ]
    for tag, mesh, vf, occ in plan:
        if tag in res and "error" not in res[tag]:
            log(f"[{tag}] already done (PSR bias {res[tag].get('PSR_bias')}); skip")
            continue
        res[tag] = run(tag, mesh, vf, occ)
        rf.write_text(json.dumps(res, indent=2))
        log(f"[{tag}] DONE -> PSR bias={res[tag].get('PSR_bias')} "
            f"RMSE={res[tag].get('PSR_rmse')} ({res[tag].get('wall_s')}s)")
    log("===== EXTENT/RESOLUTION SWEEP (Haworth PSR floor bias, eq25, big occluder) =====")
    for tag, *_ in plan:
        v = res.get(tag, {})
        log(f"  {tag:10s} PSR bias={str(v.get('PSR_bias', '?')):>7s} K  "
            f"RMSE={str(v.get('PSR_rmse', '?')):>5s} K")
    log("  (R25 vs R25_100m = resolution @ fixed extent;  R25 vs R32 = extent, coarser)")


if __name__ == "__main__":
    sys.exit(main())
