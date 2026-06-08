#!/usr/bin/env python3
"""Deep radiation-coupling two-pass validation at Haworth.

Tests the fix for the PSR cold-bias root cause (indirect flux excluded from
equilibration -> deep column too cold). Orchestrated as sequential full runs
that pass the per-face mean indirect flux via a binary file:
  pass0: normal run, writes thermal.indirect_mean_out_file (physically identical
         to the baseline -> PSR bias must reproduce -8.7 K; validates the read
         path is off-by-default and the write path works).
  pass1: feeds pass0's mean indirect into equilibration
         (thermal.eq_indirect_flux_file) -> deep column equilibrates warm ->
         expect the PSR floor max to warm toward Diviner.
  pass2: feeds pass1's mean indirect -> convergence check.
Each run reuses the cached VF (geometry-only), ~5 min/run.
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
BASE = REPO / "examples/divval_haworth_cr.yaml"
QIND0 = OUTROOT / "haworth_qind_p0.bin"
QIND1 = OUTROOT / "haworth_qind_p1.bin"


def set_path(d, path, val):
    for k in path[:-1]:
        d = d.setdefault(k, {})
    d[path[-1]] = val


def run(tag, overrides):
    cfg = yaml.safe_load(BASE.read_text())
    out = OUTROOT / f"sweep_haworth_{tag}"
    set_path(cfg, ("output", "directory"), str(out))
    for p, v in overrides:
        set_path(cfg, p, v)
    cp = REPO / f"examples/sweep_haworth_{tag}.yaml"
    cp.write_text(yaml.safe_dump(cfg, sort_keys=False))
    log(f"[{tag}] running -> {out.name}")
    t0 = time.time()
    with open(f"/tmp/sweep_haworth_{tag}.log", "w") as rl:
        try:
            subprocess.run([str(PTM), "--config", str(cp)], stdout=rl,
                           stderr=subprocess.STDOUT, timeout=3 * 3600,
                           cwd=str(REPO), env=RUN_ENV)
        except subprocess.TimeoutExpired:
            log(f"[{tag}] TIMEOUT")
    stats = out / "statistics" / "temperature_max.bin"
    if not (stats.exists() and stats.stat().st_size > 1000):
        log(f"[{tag}] FAILED - see /tmp/sweep_haworth_{tag}.log")
        return {"error": "sim"}
    m = compare(f"sweep_haworth_{tag}", LAT, LON, out)
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
    rf = FIGDIR / "deepcouple_haworth.json"
    res = json.loads(rf.read_text()) if rf.exists() else {}
    plan = [
        ("dc_pass0", [(("thermal", "indirect_mean_out_file"), str(QIND0))]),
        ("dc_pass1", [(("thermal", "eq_indirect_flux_file"), str(QIND0)),
                      (("thermal", "indirect_mean_out_file"), str(QIND1))]),
        ("dc_pass2", [(("thermal", "eq_indirect_flux_file"), str(QIND1))]),
        # Decisive test: at convergence (8 orbits) the deep column is cold, so
        # eps*sigma*T^4 ~ <q_ind>; the fix should matter most here. Compare to
        # eq8 WITHOUT fix (-22.5 K from the property sweep).
        ("dc_eq8_fix", [(("thermal", "equilibration_orbits"), 8.0),
                        (("thermal", "eq_indirect_flux_file"), str(QIND0))]),
    ]
    for tag, ov in plan:
        if tag in res and "error" not in res[tag]:
            log(f"[{tag}] already done (PSR bias {res[tag].get('PSR_bias')}); skip")
            continue
        res[tag] = run(tag, ov)
        rf.write_text(json.dumps(res, indent=2))
        log(f"[{tag}] DONE -> PSR bias={res[tag].get('PSR_bias')} "
            f"RMSE={res[tag].get('PSR_rmse')} ({res[tag].get('wall_s')}s)")
    log("===== DEEP-COUPLING SUMMARY (Haworth PSR bias) =====")
    for tag, _ in plan:
        v = res.get(tag, {})
        log(f"  {tag:10s} PSR bias={str(v.get('PSR_bias', '?')):>6s} K  "
            f"RMSE={str(v.get('PSR_rmse', '?')):>5s} K")


if __name__ == "__main__":
    sys.exit(main())
