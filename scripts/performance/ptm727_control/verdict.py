#!/usr/bin/env python3
"""PTM-727 decisive control: verdict (revised per Fermi's review, 2026-10-14).

Runs (all dt65 coupled, same lunation, same binary, same CSR, same environment):
  asrun  runs/stage3/dt65_coupled   the paper-lineage run (seed without indirect)
  Ap     ctrlAp_dt65_asrun          warm start: donor + the replay's AS-RUN post-eq columns  (C1)
  B      ctrlB_dt65_spun            warm start: donor + the replay's SPUN-UP columns
  R      fixON_dt65                 thermal.seed_indirect: series (the fix)
  A      ctrlA_dt65_donor           DIAGNOSTIC ONLY: raw donor column = "donor single-phase
                                    eq-<q> artifact" (its 1-step window makes eq <q>
                                    instantaneous, so its columns run cold). Not PTM-726.

Metric: median cold-trap T_avg, on each run's OWN set (T_max < 110 K) and on the FIXED as-run set.
  VERDICT on B - A' (spin-up memory in PTM's own solver; Fermi predicts ~ +2.2 K):
     CONFIRMED  both own and fixed >= 1.0 K
     REFUTED    both own and fixed <  0.5 K
     otherwise  INCONCLUSIVE (including own/fixed disagreeing)             (C2)
  FIX ACCEPTANCE: |R - B| <= 0.5 K on the fixed set (Fermi predicts ~ -0.2 K)  (C3)
Terminal value for an automated consumer: verdict.json "verdict" is exactly CONFIRMED, REFUTED or
INCONCLUSIVE; "fix_accepted" is C3; "rerun_with_fix_ok" = (CONFIRMED and fix_accepted), the only
state under which the pre-approved Stage-3 re-run should use this binary. CONFIRMED with
fix_accepted false means the bias is real but THIS fix does not remove it: stop and report.
Guards (exit 1, the chain writes ABORT_verdict): face counts differ between runs, a set is empty,
or any reported number is not finite. A NaN never becomes INCONCLUSIVE silently.
Reported: R - asrun split by face class (never-lit / cold-trap / lit), and the mean absorbed
indirect IR on the fixed cold-trap set, R vs asrun (exposes the lit-wall cooling effect).
"""
import json
import sys
from pathlib import Path

import numpy as np

C = Path("/Users/paha3326/projects/moon3d_R1/runs/ptm727_control")
RUNS = {"asrun": Path("/Users/paha3326/projects/moon3d_R1/runs/stage3/dt65_coupled"),
        "Ap": C / "ctrlAp_dt65_asrun", "B": C / "ctrlB_dt65_spun", "R": C / "fixON_dt65",
        "A": C / "ctrlA_dt65_donor"}
MESH = "/Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/europa/heat3d/ptm/data/dem/moon3d_lat/moon3d_dt65.ply"
CONFIRM, REFUTE, ACCEPT = 1.0, 0.5, 0.5


def stat(run, name):
    b = open(run / "statistics" / f"{name}.bin", "rb").read()
    n = int(np.frombuffer(b[:4], "<u4")[0])
    v = np.frombuffer(b[4:4 + 4 * n], "<f4").astype(np.float64)
    if v.size != n:
        raise SystemExit(f"FATAL: {run}/statistics/{name}.bin truncated")
    return v


def main():
    sys.path.insert(0, "/Users/paha3326/projects/moon3d_R1/ptm_bench/scripts/ptm12_diag")
    from assemble_vf_haworth import face_areas
    area = face_areas(MESH)[0]
    At = area.sum()
    F = {k: dict(Tmax=stat(p, "temperature_max"), Tavg=stat(p, "temperature_avg"),
                 qir=stat(p, "indirect_ir_avg")) for k, p in RUNS.items()}
    sizes = {k: (v["Tmax"].size, v["Tavg"].size, v["qir"].size) for k, v in F.items()}
    if len({x for t in sizes.values() for x in t} | {area.size}) != 1:
        raise SystemExit(f"FATAL: face counts differ between runs / mesh: {sizes}, mesh {area.size}")
    ref = F["asrun"]
    fixed = ref["Tmax"] < 110.0
    if not fixed.any():
        raise SystemExit("FATAL: the fixed cold-trap set (as-run T_max < 110 K) is empty")
    neverlit = stat(RUNS["asrun"], "direct_solar_max") <= 0.0
    classes = {"cold-trap": fixed, "never-lit (not ct)": neverlit & ~fixed, "lit": ~neverlit & ~fixed}
    m = {}
    for k, v in F.items():
        own = v["Tmax"] < 110.0
        if not own.any():
            raise SystemExit(f"FATAL: run {k}: own cold-trap set is empty")
        m[k] = {"own": float(np.median(v["Tavg"][own])), "fixed": float(np.median(v["Tavg"][fixed])),
                "area110_pct": float(100 * area[own].sum() / At), "n_ct": int(own.sum()),
                "qir_ct_fixed_mean": float(np.mean(v["qir"][fixed]))}
    d = lambda a, b, key: m[a][key] - m[b][key]
    mem_own, mem_fix = d("B", "Ap", "own"), d("B", "Ap", "fixed")
    if mem_own >= CONFIRM and mem_fix >= CONFIRM:
        verdict = "CONFIRMED"
    elif mem_own < REFUTE and mem_fix < REFUTE:
        verdict = "REFUTED"
    else:
        verdict = "INCONCLUSIVE"
    fix_gap = d("R", "B", "fixed")
    fix_ok = abs(fix_gap) <= ACCEPT
    by_class = {}
    for c, mask in classes.items():
        by_class[c] = {"n": int(mask.sum()),
                       "R_minus_asrun_Tavg_median": float(np.median((F["R"]["Tavg"] - ref["Tavg"])[mask])),
                       "B_minus_Ap_Tavg_median": float(np.median((F["B"]["Tavg"] - F["Ap"]["Tavg"])[mask]))}
    nums = [mem_own, mem_fix, fix_gap] + [x for v in m.values() for x in v.values()]
    if not np.all(np.isfinite(nums)):
        raise SystemExit(f"FATAL: non-finite verdict input: B-A' own {mem_own}, fixed {mem_fix}, "
                         f"R-B {fix_gap}, metrics {m}")
    rep = {"runs": {k: str(p) for k, p in RUNS.items()}, "metrics": m,
           "B_minus_Ap_own_K": mem_own, "B_minus_Ap_fixed_K": mem_fix,
           "verdict": verdict,
           "verdict_rule": f"CONFIRMED iff own and fixed >= {CONFIRM} K; REFUTED iff both < {REFUTE} K; "
                           "else INCONCLUSIVE (Fermi C1/C2)",
           "fix_R_minus_B_fixed_K": fix_gap, "fix_accepted": bool(fix_ok),
           "rerun_with_fix_ok": bool(verdict == "CONFIRMED" and fix_ok),
           "fix_rule": f"|R - B| <= {ACCEPT} K on the fixed cold-trap set (Fermi C3)",
           "R_minus_asrun_by_class": by_class,
           "qir_ct_fixed_mean_Wm2": {"asrun": m["asrun"]["qir_ct_fixed_mean"], "R": m["R"]["qir_ct_fixed_mean"]},
           "diagnostic_A": {"label": "donor single-phase eq-<q> artifact (not PTM-726)",
                            "B_minus_A_own_K": d("B", "A", "own"), "Ap_minus_A_own_K": d("Ap", "A", "own")},
           "B_minus_asrun_own_K": d("B", "asrun", "own"), "Ap_minus_asrun_own_K": d("Ap", "asrun", "own")}
    tmp = C / "verdict.json.tmp"
    tmp.write_text(json.dumps(rep, indent=1))
    tmp.replace(C / "verdict.json")
    for k, v in m.items():
        print(f"{k:6s} ct T_avg median own {v['own']:.3f} K  fixed {v['fixed']:.3f} K  area@110 "
              f"{v['area110_pct']:.4f} %  n_ct {v['n_ct']}  <q_ir> ct(fixed) {v['qir_ct_fixed_mean']:.3f} W/m^2")
    print(f"B - A' (spin-up memory, PTM solver): own {mem_own:+.3f} K, fixed {mem_fix:+.3f} K -> {verdict}")
    print(f"fix: R - B fixed {fix_gap:+.3f} K -> {'ACCEPTED' if fix_ok else 'NOT ACCEPTED'} (|R-B| <= {ACCEPT} K)")
    for c, v in by_class.items():
        print(f"  class {c:20s} n {v['n']:7d}  R - asrun {v['R_minus_asrun_Tavg_median']:+.3f} K   "
              f"B - A' {v['B_minus_Ap_Tavg_median']:+.3f} K")
    print(f"diagnostic A (donor artifact): B - A own {rep['diagnostic_A']['B_minus_A_own_K']:+.3f} K, "
          f"A' - A own {rep['diagnostic_A']['Ap_minus_A_own_K']:+.3f} K")


if __name__ == "__main__":
    main()
