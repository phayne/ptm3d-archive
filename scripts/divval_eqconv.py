#!/usr/bin/env python3
"""H5 equilibration-convergence test for the Haworth PSR summer-max.

Question: is the model's ~50 K floor the CONVERGED answer, or is the two-pass
time-MEAN indirect still over-cooling the deep column as equilibration deepens?

Method (all on the CORRECTED big-occluder setup, cached VF reused):
  p0       : eq=2, write the per-face mean indirect flux (eq_indirect OFF) -> qind.
  eq{8,25,49}_fix : equilibration_orbits = N, eq_indirect_flux_file = qind.
If the PSR-floor summer-max bias is ~flat across eq8/25/49, the two-pass fix has
removed the equilibration cold-bias and ~50 K is converged (-> the residual gap
to Diviner is data-side / knife-edge). If it keeps dropping with eq, the
time-mean two-pass is incomplete and the model still over-cools.

Compare to the WITHOUT-fix trend (from the property sweep / memory):
  eq2 -8.7, eq4 -13.6, eq8 -22.5 K  (deepens -> colder).
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
BASE = REPO / "examples/divval_haworth_eq49_bigocc.yaml"   # big occluder + cached VF
QIND = OUTROOT / "haworth_eqconv_qind.bin"


def set_path(d, path, val):
    for k in path[:-1]:
        d = d.setdefault(k, {})
    d[path[-1]] = val


def run(tag, overrides):
    cfg = yaml.safe_load(BASE.read_text())
    out = OUTROOT / f"eqconv_haworth_{tag}"
    set_path(cfg, ("output", "directory"), str(out))
    for p, v in overrides:
        set_path(cfg, p, v)
    cp = REPO / f"examples/eqconv_haworth_{tag}.yaml"
    cp.write_text(yaml.safe_dump(cfg, sort_keys=False))
    log(f"[{tag}] running -> {out.name}  (eq={cfg['thermal'].get('equilibration_orbits')})")
    t0 = time.time()
    with open(f"/tmp/eqconv_haworth_{tag}.log", "w") as rl:
        try:
            subprocess.run([str(PTM), "--config", str(cp)], stdout=rl,
                           stderr=subprocess.STDOUT, timeout=3 * 3600,
                           cwd=str(REPO), env=RUN_ENV)
        except subprocess.TimeoutExpired:
            log(f"[{tag}] TIMEOUT")
    stats = out / "statistics" / "temperature_max.bin"
    if not (stats.exists() and stats.stat().st_size > 1000):
        log(f"[{tag}] FAILED - see /tmp/eqconv_haworth_{tag}.log")
        return {"error": "sim"}
    m = compare(f"eqconv_haworth_{tag}", LAT, LON, out)
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
    rf = FIGDIR / "eqconv_haworth.json"
    res = json.loads(rf.read_text()) if rf.exists() else {}
    plan = [
        # baseline + write a fresh, self-consistent qind (eq_indirect OFF here)
        ("p0", [(("thermal", "equilibration_orbits"), 2.0),
                (("thermal", "eq_indirect_flux_file"), ""),
                (("thermal", "indirect_mean_out_file"), str(QIND))]),
        ("eq8_fix",  [(("thermal", "equilibration_orbits"), 8.0),
                      (("thermal", "eq_indirect_flux_file"), str(QIND))]),
        ("eq25_fix", [(("thermal", "equilibration_orbits"), 25.0),
                      (("thermal", "eq_indirect_flux_file"), str(QIND))]),
        ("eq49_fix", [(("thermal", "equilibration_orbits"), 49.0),
                      (("thermal", "eq_indirect_flux_file"), str(QIND))]),
    ]
    for tag, ov in plan:
        if tag in res and "error" not in res[tag]:
            log(f"[{tag}] already done (PSR bias {res[tag].get('PSR_bias')}); skip")
            continue
        res[tag] = run(tag, ov)
        rf.write_text(json.dumps(res, indent=2))
        log(f"[{tag}] DONE -> PSR bias={res[tag].get('PSR_bias')} "
            f"RMSE={res[tag].get('PSR_rmse')} ({res[tag].get('wall_s')}s)")
    log("===== EQ-CONVERGENCE SUMMARY (Haworth PSR floor bias, big occluder, two-pass) =====")
    for tag, _ in plan:
        v = res.get(tag, {})
        log(f"  {tag:10s} PSR bias={str(v.get('PSR_bias', '?')):>7s} K  "
            f"RMSE={str(v.get('PSR_rmse', '?')):>5s} K")
    log("  (without-fix reference: eq2 -8.7, eq4 -13.6, eq8 -22.5 K)")


if __name__ == "__main__":
    sys.exit(main())
