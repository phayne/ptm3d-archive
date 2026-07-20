#!/usr/bin/env python3
"""Science payoff for the H-matrix subdivide fix: does resolving the summer
lit-wall fringe (now that fine cores have un-collapsed floor<->wall VFs) change
the Haworth PSR-floor summer-max bias?

Runs the SAME setup (eq25, big occluder, NO two-pass, rad-coupled) on two cores
that differ ONLY in resolution, BOTH built with the fixed binary
(PTM_HMATRIX_SUBDIVIDE=1) so the comparison is clean:
  r25_fixed      : 99,874 f  (~224 m/face)  vf=/tmp/r25_on.vf       (floor F_terr 0.0175)
  r25_100m_fixed : 400,973 f (~112 m/face)  vf=/tmp/r25_100m_on3.vf (floor F_terr 0.0049 ~ geometric)

Recorded legacy baselines (extentsweep, collapsed/coarse VF): R25 -34.2, R32c -29.0.
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
ENV = {**RUN_ENV, "PTM_HMATRIX_SUBDIVIDE": "1"}   # ensure fix is on if any rebuild


def set_path(d, path, val):
    for k in path[:-1]:
        d = d.setdefault(k, {})
    d[path[-1]] = val


def run(tag, mesh, vf, occ):
    cfg = yaml.safe_load(BASE.read_text())
    out = OUTROOT / f"hmfix_haworth_{tag}"
    set_path(cfg, ("output", "directory"), str(out))
    set_path(cfg, ("mesh", "import_file"), mesh)
    set_path(cfg, ("mesh", "occluder_file"), str(MESHDIR / occ))
    set_path(cfg, ("radiation", "view_factor_file"), vf)
    set_path(cfg, ("thermal", "equilibration_orbits"), 25.0)
    set_path(cfg, ("thermal", "eq_indirect_flux_file"), "")
    cp = REPO / f"examples/hmfix_haworth_{tag}.yaml"
    cp.write_text(yaml.safe_dump(cfg, sort_keys=False))
    stats = out / "statistics" / "temperature_max.bin"
    for attempt in range(1, 4):
        log(f"[{tag}] sim attempt {attempt} -> {out.name}")
        t0 = time.time()
        with open(f"/tmp/hmfix_haworth_{tag}.log", "w") as rl:
            try:
                subprocess.run([str(PTM), "--config", str(cp)], stdout=rl,
                               stderr=subprocess.STDOUT, timeout=6 * 3600,
                               cwd=str(REPO), env=ENV)
            except subprocess.TimeoutExpired:
                log(f"[{tag}] TIMEOUT attempt {attempt}")
        if stats.exists() and stats.stat().st_size > 1000:
            m = compare(f"hmfix_haworth_{tag}", LAT, LON, out)
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
        log(f"[{tag}] no stats (GPU hang?), retry"); time.sleep(90)
    return {"error": "sim"}


def main():
    rf = FIGDIR / "hmfix_haworth.json"
    res = json.loads(rf.read_text()) if rf.exists() else {}
    plan = [
        ("r25_fixed",      str(MESHDIR / "haworth_radcore_r25.ply"),      "/tmp/r25_on.vf",        "haworth_ring_25_200.ply"),
        ("r25_100m_fixed", str(MESHDIR / "haworth_radcore_r25_100m.ply"), "/tmp/r25_100m_on3.vf",  "haworth_ring_25_200.ply"),
    ]
    for tag, mesh, vf, occ in plan:
        if tag in res and "error" not in res[tag]:
            log(f"[{tag}] done (PSR bias {res[tag].get('PSR_bias')}); skip"); continue
        if not Path(vf).exists():
            log(f"[{tag}] MISSING VF {vf}; skip"); res[tag] = {"error": "no vf"}; continue
        res[tag] = run(tag, mesh, vf, occ)
        rf.write_text(json.dumps(res, indent=2))
        log(f"[{tag}] DONE -> PSR bias={res[tag].get('PSR_bias')} "
            f"RMSE={res[tag].get('PSR_rmse')} ({res[tag].get('wall_s')}s)")
    log("===== H-MATRIX-FIX RESOLUTION TEST (Haworth PSR floor bias, eq25, big occluder) =====")
    for tag, *_ in plan:
        v = res.get(tag, {})
        log(f"  {tag:16s} PSR bias={str(v.get('PSR_bias','?')):>7s} K  "
            f"RMSE={str(v.get('PSR_rmse','?')):>5s} K  ({v.get('wall_s','?')}s)")
    log("  (legacy baselines: R25 -34.2, R32c -29.0;  fixed-floor F_terr: r25 0.0175, r25_100m 0.0049)")


if __name__ == "__main__":
    sys.exit(main())
