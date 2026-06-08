#!/usr/bin/env python3
"""PSR cold-bias sensitivity sweep at Haworth.

Single-knob STAR sweep on the Moon/SPICE path, reusing the cached view factors
(geometry-only -> VF build skipped, ~8-9 min/run). Knobs (all off by default in
the binary; sentinel-gated property overrides added for this study):
  - albedo_override          (reflected-solar + wall-temperature indirect channel)
  - constant_thermal_inertia (uniform-TI sign anchor)
  - ks_override/kd_override  (k-scaling: depth-profile-preserving TI axis, TI ~ sqrt(k))
  - emissivity_override      (bounding)
  - equilibration_orbits     (under-equilibration control)
Each run -> diviner_compare PSR bias/RMSE vs Diviner (rigorous max-over-time Tbol).

Entry 0 is a no-op reproduction: the rebuilt binary with NO override must
reproduce the committed baseline (-8.7 K) before any sweep result is trusted.
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

HAWORTH_LAT, HAWORTH_LON = -87.45, -2.1
BASE_CFG = REPO / "examples/divval_haworth_cr.yaml"
KS, KD = 0.00074, 0.0034  # Moon default surface/deep conductivity (W/m/K)

# (tag, description, [(yaml_path_tuple, value), ...])
SWEEP = [
    ("base_repro", "no-op (rebuilt binary must match -8.7 K)", []),
    ("eq4", "equilibration 4 orbits (control)",
        [(("thermal", "equilibration_orbits"), 4.0)]),
    ("alb008", "albedo 0.08", [(("thermophysical", "albedo_override"), 0.08)]),
    ("alb018", "albedo 0.18", [(("thermophysical", "albedo_override"), 0.18)]),
    ("alb025", "albedo 0.25", [(("thermophysical", "albedo_override"), 0.25)]),
    ("ti40", "constant TI 40 SI",
        [(("thermophysical", "constant_thermal_inertia"), 40.0)]),
    ("ti100", "constant TI 100 SI",
        [(("thermophysical", "constant_thermal_inertia"), 100.0)]),
    ("ks2", "k-scale TIx2 (k x4)",
        [(("thermophysical", "ks_override"), KS * 4), (("thermophysical", "kd_override"), KD * 4)]),
    ("ks4", "k-scale TIx4 (k x16)",
        [(("thermophysical", "ks_override"), KS * 16), (("thermophysical", "kd_override"), KD * 16)]),
    ("ks0p5", "k-scale TIx0.5 (k x0.25)",
        [(("thermophysical", "ks_override"), KS * 0.25), (("thermophysical", "kd_override"), KD * 0.25)]),
    ("emis090", "emissivity 0.90",
        [(("thermophysical", "emissivity_override"), 0.90)]),
    # Equilibration-convergence follow-up: is the PSR floor still cooling?
    ("eq6", "equilibration 6 orbits",
        [(("thermal", "equilibration_orbits"), 6.0)]),
    ("eq8", "equilibration 8 orbits",
        [(("thermal", "equilibration_orbits"), 8.0)]),
]


def set_path(d, path, val):
    for k in path[:-1]:
        d = d.setdefault(k, {})
    d[path[-1]] = val


def run_one(tag, desc, overrides):
    cfg = yaml.safe_load(BASE_CFG.read_text())
    out = OUTROOT / f"sweep_haworth_{tag}"
    set_path(cfg, ("output", "directory"), str(out))
    for path, val in overrides:
        set_path(cfg, path, val)
    cfg_path = REPO / f"examples/sweep_haworth_{tag}.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    stats = out / "statistics" / "temperature_max.bin"
    log(f"[{tag}] {desc} -> {out.name}")
    t0 = time.time()
    with open(f"/tmp/sweep_haworth_{tag}.log", "w") as rl:
        try:
            subprocess.run([str(PTM), "--config", str(cfg_path)], stdout=rl,
                           stderr=subprocess.STDOUT, timeout=3 * 3600,
                           cwd=str(REPO), env=RUN_ENV)
        except subprocess.TimeoutExpired:
            log(f"[{tag}] TIMEOUT")
    if not (stats.exists() and stats.stat().st_size > 1000):
        log(f"[{tag}] FAILED (no stats) - see /tmp/sweep_haworth_{tag}.log")
        return {"tag": tag, "desc": desc, "error": "sim"}
    dt = time.time() - t0
    m = compare(f"sweep_haworth_{tag}", HAWORTH_LAT, HAWORTH_LON, out)
    m.update(tag=tag, desc=desc, wall_s=round(dt, 1))
    # Tidy: per-step temps consumed by the rigorous compare; drop to save disk.
    tdir = out / "temperature"
    if tdir.exists():
        for p in tdir.glob("*.bin"):
            p.unlink()
        try:
            tdir.rmdir()
        except OSError:
            pass
    return m


def main():
    FIGDIR.mkdir(parents=True, exist_ok=True)
    rf = FIGDIR / "sweep_haworth.json"
    results = json.loads(rf.read_text()) if rf.exists() else {}
    for tag, desc, ov in SWEEP:
        if tag in results and "error" not in results[tag]:
            log(f"[{tag}] already done (PSR bias {results[tag].get('PSR_bias')}); skip")
            continue
        results[tag] = run_one(tag, desc, ov)
        rf.write_text(json.dumps(results, indent=2))
        log(f"[{tag}] DONE -> PSR bias={results[tag].get('PSR_bias')} "
            f"RMSE={results[tag].get('PSR_rmse')} ({results[tag].get('wall_s')}s)")
    log("===== SWEEP SUMMARY (Haworth PSR bias vs knob) =====")
    for tag, _, _ in SWEEP:
        r = results.get(tag, {})
        log(f"  {tag:10s} {r.get('desc', '?'):32s} "
            f"PSR bias={str(r.get('PSR_bias', '?')):>6s} K  "
            f"RMSE={str(r.get('PSR_rmse', '?')):>5s} K  ({r.get('wall_s', '?')}s)")


if __name__ == "__main__":
    sys.exit(main())
