#!/usr/bin/env python3
"""Inventory of timings ALREADY measured for Moon3D R1 (Stage 1, Stage 3, keps test,
v4 sweeps), so the benchmark batch re-measures only what is missing.

Those runs predate PTM_TIMING_JSON, so components are recovered from what the
logs do carry, and each recovered number says how:
  illumination   "Illumination calculation completed in X seconds"        exact
  radiosity      final 'elapsed' of each "Computing indirect flux" bar    1 s resolution
  GPU thermal    "[GPU Thermal] Persistent kernel: ... in X ms" lines,     kernel time only
                 segmented by "--- Radiation Iteration k ---"             (no host pauses)
  total, RSS     /usr/bin/time -l (*.time.txt) or "Simulation completed"  exact
  remainder      total - the above (setup, CSR load, host side, export)   derived
Contamination: every run window is checked against every other R1 run window and
against the windows in which Ada's own CPU jobs ran (MY_WINDOWS, from file mtimes
and harness ledgers). Other agents' interactive work is NOT knowable from disk.

  inventory.py --json out.json        (prints a table; writes the full record)
"""
import argparse
import glob
import json
import re
import statistics as st
from datetime import datetime, timezone
from pathlib import Path

RUNS = Path.home() / "projects/moon3d_R1/runs"
MY_WINDOWS = [("2026-10-12T18:31:00Z", "2026-10-12T19:53:00Z", "Ada: builds -j8, VF gates 8 jobs, smokes"),
              ("2026-10-13T19:47:00Z", "2026-10-13T21:27:00Z", "Ada: disk tests T1-T4, 8 jobs")]
BAR_RE = re.compile(r"Computing indirect flux \[=+\] 100% \| elapsed ((?:\d+h )?(?:\d+m )?\d+s)")
BAR_PT = re.compile(r"Computing indirect flux \[[=> ]+\] (\d+)% \| elapsed ((?:\d+h )?(?:\d+m )?\d+s)")
KER_RE = re.compile(r"\[GPU Thermal\] Persistent kernel: \d+ cycles x (\d+) timesteps for (\d+) faces in ([0-9.]+) ms")


def t(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()


def secs(s):
    h = re.search(r"(\d+)h", s)
    m = re.search(r"(\d+)m", s)
    x = re.search(r"(\d+)s", s)
    return (int(h.group(1)) * 3600 if h else 0) + (int(m.group(1)) * 60 if m else 0) + (int(x.group(1)) if x else 0)


def status_windows(path, key_start=("launch",), key_end=("run_exit", "driver_exit")):
    """{arm: (start, end)} from a CHAIN_STATUS/STATUS file (last launch, its exit)."""
    w = {}
    if not Path(path).exists():
        return w
    for ln in Path(path).read_text().splitlines():
        p = ln.split("\t")
        if len(p) < 3:
            continue
        arm = p[1].replace("arm=", "")
        ph = p[2].replace("phase=", "")
        if ph in key_start:
            w[arm] = [t(p[0]), None]
        elif ph in key_end and arm in w:
            w[arm][1] = t(p[0])
    return {k: tuple(v) for k, v in w.items() if v[1]}


def parse_engine(stdout_log, stderr_log=None):
    s = Path(stdout_log).read_text(errors="replace")
    e = Path(stderr_log).read_text(errors="replace") if stderr_log and Path(stderr_log).exists() else ""
    both = s + "\n" + e
    r = {"log": str(stdout_log)}
    m = re.search(r"Illumination calculation completed in ([0-9.]+) seconds", s)
    r["illumination_s"] = float(m.group(1)) if m else None
    m = re.search(r"Simulation completed in ([0-9.]+) seconds", s)
    r["engine_total_s"] = float(m.group(1)) if m else None
    m = re.search(r"^\s*([0-9.]+) real", e, re.M)
    r["wall_s"] = float(m.group(1)) if m else None
    m = re.search(r"(\d+)\s+maximum resident set size", e)
    r["peak_rss_bytes"] = int(m.group(1)) if m else None
    m = re.search(r"Total time steps \(including equilibration and integration coverage\): (\d+)", s)
    r["total_steps"] = int(m.group(1)) if m else None
    m = re.search(r"REPLACED H-matrix[^\n]*?nnz=(\d+)", s)
    r["nnz"] = int(m.group(1)) if m else None
    m = re.search(r"ROUTE = (\w+)", s)
    r["radiosity_route"] = m.group(1) if m else None
    m = re.search(r"Loaded mesh with \d+ vertices and (\d+) faces|Successfully loaded \d+ vertices and (\d+) faces", s)
    r["n_faces"] = int(m.group(1) or m.group(2)) if m else None
    r["sweeps_s"] = [secs(x) for x in BAR_RE.findall(both)]
    r["faces_rethermal"] = [int(x) for x in re.findall(r"Re-running thermal model for (\d+) faces", s)]
    r["max_dT"] = [float(x) for x in re.findall(r"Max temperature change: ([0-9.]+) K", s)]
    # GPU kernel ms, segmented by iteration markers (segment 0 = before iteration 1)
    segs, cur = [], 0.0
    for ln in s.splitlines():
        if ln.startswith("--- Radiation Iteration"):
            segs.append(cur)
            cur = 0.0
        km = KER_RE.search(ln)
        if km:
            cur += float(km.group(3)) / 1e3
    segs.append(cur)
    r["gpu_kernel_s_by_segment"] = segs
    # within-sweep linearity of the FIRST full radiosity bar: elapsed vs percent
    pts = []
    for pc, el in BAR_PT.findall(both):
        pc = int(pc)
        if pts and pc < pts[-1][0]:
            break
        pts.append((pc, secs(el)))
    if len(pts) > 10 and pts[-1][1] > 20:
        T = pts[-1][1]
        dev = max(abs(el - T * pc / 100.0) for pc, el in pts if pc >= 10)
        r["sweep_bar_max_dev_from_linear_s"] = dev
        r["sweep_bar_total_s"] = T
    return r


def overlaps(win, others):
    a, b = win
    return [name for (x, y, name) in others if x < b and y > a]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    recs = []
    # ---- Stage 3 single-shot (Moon3D 509k, binary afbd2bd GPU build) ------------------
    for d, status in ((RUNS / "stage3", "CHAIN_STATUS.txt"), (RUNS / "stage3_convcheck", "STATUS.txt"),
                      (RUNS / "stage3_eqconv", "STATUS.txt")):
        win = status_windows(d / status)
        for log in sorted(d.glob("*.log")):
            arm = log.stem
            if arm.endswith((".verify", ".driver", ".stitch")) or arm in ("chain_nohup", "nohup") \
                    or arm.startswith("yearlong"):
                continue
            if arm not in win:
                continue
            r = parse_engine(log, d / f"{arm}.time.txt")
            r.update(source=f"{d.name}/{arm}", family="moon3d_509k_paper_mesh", window=win[arm],
                     exe_md5=(d / f"{arm}.exe_md5.txt").read_text().split()[0]
                     if (d / f"{arm}.exe_md5.txt").exists() else None)
            recs.append(r)
    # ---- Stage 1 + keps arm b (Haworth r50 140,091, chunked yearlong driver) ---------
    for d, status in ((RUNS / "stage1_haworth", "CHAIN_STATUS.txt"), (RUNS / "keps_test", "STATUS.txt")):
        win = status_windows(d / status, key_start=("launch",), key_end=("driver_exit",))
        for arm, w in win.items():
            for ch in sorted((d / arm).glob("chunk_00?.log")):
                r = parse_engine(ch, ch)   # chunk logs carry stdout+stderr together
                r.update(source=f"{d.name}/{arm}/{ch.stem}", family="haworth_r50_140k", window=w,
                         note="chunked yearlong driver; window = whole arm")
                recs.append(r)
    windows = [(r["window"][0], r["window"][1], r["source"]) for r in recs]
    windows.append((t("2026-10-13T17:42:19Z"), t("2026-10-13T17:48:23Z"), "keps_test sweep+assembly"))
    mine = [(t(x), t(y), n) for x, y, n in MY_WINDOWS]
    for r in recs:
        r["overlaps_other_r1_runs"] = [n for n in overlaps(r["window"], windows) if n != r["source"]
                                       and not (n.split("/")[:2] == r["source"].split("/")[:2])]
        r["overlaps_my_cpu_jobs"] = overlaps(r["window"], mine)
        r["window"] = [datetime.fromtimestamp(x, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                       for x in r["window"]]
    # ---- sweeps (no engine) ---------------------------------------------------------
    sweeps = []
    ks = RUNS / "keps_test"
    stats = [json.loads(m) for f in sorted(glob.glob(str(ks / "logs/err_*.txt")))
             for m in re.findall(r"# STATS (\{.*\})", Path(f).read_text())]
    sw = {"source": "keps_test sweep (sunlit_radcore_r50, 140,091 f, NR 2e5, keps 0.01, 48 procs, "
                    "mc_ref_pp_brew_sse2 md5 38c25338...)",
          "window": ["2026-10-13T17:42:19Z", "2026-10-13T17:48:12Z"], "wall_s": t("2026-10-13T17:48:12Z") - t("2026-10-13T17:42:19Z"),
          "rays": sum(x["rays_total"] for x in stats), "stats_lines": len(stats),
          "max_chunk_wall_s": max((x["wall_s"] for x in stats), default=None),
          "assembly_wall_s": 377.71, "assembly_peak_rss_bytes": 69129035776,
          "assembly_note": "assembler assemble_vf_haworth_53a6a65.py (pre-memory-fix, ~120 B/nnz)",
          "nnz": 573061316}
    sw["rays_per_s"] = sw["rays"] / sw["wall_s"] if sw["wall_s"] else None
    sw["overlaps_other_r1_runs"] = overlaps((t(sw["window"][0]), t(sw["window"][1])), windows)
    sweeps.append(sw)
    ht = json.loads((Path.home() / "projects/moon3d/vf/resweep_v4_2026-08-24/headline_table.json").read_text())
    m5 = ht["matrices"]["moon3d_509k"]["v4_today"]
    sweeps.append({"source": "v4 resweep 2026-08-24 Moon3D 509k (headline_table.json; mc_ref_pp md5 "
                             "a5556602..., Homebrew SSE2 Embree, older binary)", "wall_s": m5["sweep_wall_s"],
                   "assembly_wall_s": m5["assemble_wall_s"], "rays": 509337 * 200000,
                   "rays_per_s": 509337 * 200000 / m5["sweep_wall_s"], "nnz": m5["nnz"],
                   "contamination": "unknown (August; not reconstructable)"})
    out = {"engine_runs": recs, "sweeps": sweeps, "my_windows": MY_WINDOWS}
    # ---- table ------------------------------------------------------------------------
    print(f"{'source':44s} {'N':>7s} {'wall':>6s} {'illum':>6s} {'sweeps':>14s} {'it':>3s} "
          f"{'kern0':>6s} {'kern/it':>8s}  route  contamination")
    for r in recs:
        sw_ = r["sweeps_s"]
        ker = r["gpu_kernel_s_by_segment"]
        kit = st.mean(ker[1:]) if len(ker) > 1 else None
        cont = ("MINE " if r["overlaps_my_cpu_jobs"] else "") + \
               (("R1:" + ",".join(x.split("/")[-1] for x in r["overlaps_other_r1_runs"])) if r["overlaps_other_r1_runs"] else "")
        print(f"{r['source'][:44]:44s} {r['n_faces'] or 0:7d} {r['wall_s'] or r['engine_total_s'] or 0:6.0f} "
              f"{r['illumination_s'] or 0:6.1f} {len(sw_):3d}x{(st.mean(sw_) if sw_ else 0):6.1f}s{(st.pstdev(sw_) if len(sw_) > 1 else 0):4.1f} "
              f"{len(r['max_dT']):3d} {ker[0]:6.1f} {kit or 0:8.1f}  {r['radiosity_route'] or '-':5s}  {cont or 'none found'}")
    for s_ in sweeps:
        print(f"SWEEP {s_['source'][:90]}: {s_['wall_s']:.0f} s, {s_['rays_per_s']:.2e} rays/s, "
              f"assembly {s_['assembly_wall_s']:.0f} s, overlaps {s_.get('overlaps_other_r1_runs', s_.get('contamination'))}")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))
        print(f"wrote {a.json}")


if __name__ == "__main__":
    main()
