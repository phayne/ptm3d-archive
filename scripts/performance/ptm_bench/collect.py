#!/usr/bin/env python3
"""Flatten a benchmark ledger into one CSV (one row per completed run) for the
exponent fits (Fermi) and the measured Fig. 25 replacement.

Columns: run identity, mesh size, matrix size, per-component wall/CPU/RSS from
the engine's PTM_TIMING_JSON, sweep/assembly cost, C4 matrix statistics, and the
C3 accuracy metrics. Rows recorded on a shared machine are KEPT and flagged
(shared_machine=1); a fit must exclude them, and this script says how many there are.

  collect.py <ledger.jsonl> --csv out.csv
"""
import argparse
import csv
import json
import sys
from pathlib import Path


def flat(rec):
    row = rec.get("_row", {})
    res = rec["result"]
    o = {"id": rec["id"], "kind": row.get("kind"), "arm": rec["id"].split("_")[0],
         "n_faces": row.get("n_faces"), "nrays": row.get("nrays"), "keps": row.get("keps"),
         "estimator": row.get("estimator"), "dd": row.get("dd"),
         "shared_machine": int(bool(rec.get("shared_machine"))),
         "predicted_wall_s": row.get("predicted_wall_s"),
         "engine_md5": rec.get("_prov", {}).get("engine", {}).get("md5"),
         "git_sha": rec.get("_prov", {}).get("git", {}).get("sha")}
    t = res.get("timing")
    if t:
        o.update(total_wall_s=t["total"]["wall_s"], total_cpu_s=t["total"]["cpu_s"],
                 peak_rss_bytes=t["total"]["peak_rss_bytes"])
        for k, v in t["intervals"].items():
            o[f"{k}_wall_s"] = v["wall_s"]
            o[f"{k}_cpu_s"] = v["cpu_s"]
            o[f"{k}_calls"] = v["calls"]
        # A record with eq_indirect_restart=1 marks a loop restart: every record before
        # the LAST marker belongs to an abandoned loop. Its cost is real machine time
        # (kept in *_total), but only the final loop counts as the coupling iterations.
        it = t.get("coupling_iterations", [])
        last = max([k for k, r in enumerate(it) if r.get("eq_indirect_restart")], default=-1)
        real = [r for r in it if not r.get("eq_indirect_restart")]
        final = [r for r in it[last + 1:] if not r.get("eq_indirect_restart")]
        o["coupling_iterations"] = len(final)
        o["abandoned_iterations"] = len(real) - len(final)
        o["faces_rethermal_final"] = sum(r.get("faces_rethermal", 0) for r in final)
        o["faces_rethermal_total"] = sum(r.get("faces_rethermal", 0) for r in real)
        o.update({k: v for k, v in t["counters"].items()})
    b = res.get("vf_stats_banner")
    if b:
        import re
        m = re.search(r"N=(\d+) nnz=(\d+)", b)
        if m:
            o["nnz"] = int(m.group(2))
    for k in ("sweep_wall_s", "jobs", "assemble_wall_s", "assemble_peak_rss_bytes", "wall_s"):
        if k in res:
            o[k] = res[k]
    vs = res.get("vf_stats")
    if vs:
        o.update(nnz=vs["nnz"], nnz_per_face=vs["nnz_per_face_mean"],
                 density=vs["density_nnz_over_n2"], sum_f_mean=vs["sum_f"]["mean"],
                 f_sky_p50=vs["f_sky"]["p50"], saturation=vs.get("saturation"),
                 directed_saturation=vs.get("directed_saturation"),
                 rho=vs.get("spectral", {}).get("rho"), csr_bytes=vs.get("csr_bytes"))
    ss = res.get("sweep_stats")
    if ss:
        o.update(partners_kept=ss.get("partners_kept"), rays_total=ss.get("rays_total"),
                 terrain_hits=ss.get("terrain_hits"))
    a = res.get("accuracy")
    if a:
        for k in ("fsky_mean_bias_pct", "fsky_peak_dev_pct", "Tf_dT_pct", "flux_dflux_pct",
                  "picard_iters", "picard_seconds", "vf_seconds"):
            if k in a:
                o[k] = a[k]
        for side in ("floor", "all"):
            v = a.get("vs_reference", {}).get(side)
            if v:
                o[f"qind_rel_rms_{side}"] = v["qind_rel_rms"]
                o[f"T_rms_diff_K_{side}"] = v["T_rms_diff_K"]
        for name in ("one_minus_sumf", "referee_fsky"):
            if name in a:
                o[f"floor_{name}_bias_pct"] = a[name]["mean_bias_pct"]
                o[f"floor_{name}_peak_pct"] = a[name]["peak_dev_pct"]
    return o


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ledger")
    ap.add_argument("--csv", required=True)
    a = ap.parse_args()
    start, done = {}, []
    for ln in Path(a.ledger).read_text().splitlines():
        try:
            r = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if r.get("event") == "start":
            start[r["id"]] = r
        elif r.get("event") == "done":
            s = start.get(r["id"], {})
            r["_row"], r["_prov"] = s.get("row", {}), s.get("provenance", {})
            done.append(r)
    rows = [flat(r) for r in done]
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(a.csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    shared = sum(r["shared_machine"] for r in rows)
    print(f"[collect] {len(rows)} completed runs -> {a.csv}")
    if shared:
        print(f"[collect] WARNING: {shared} run(s) were timed on a SHARED machine "
              "(shared_machine=1); exclude them from every cost fit.", file=sys.stderr)


if __name__ == "__main__":
    main()
