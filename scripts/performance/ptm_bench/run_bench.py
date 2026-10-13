#!/usr/bin/env python3
"""Execute a Moon3D R1 benchmark plan (written by plan.py), resumably.

Run kinds (each plan row has "id" and "kind"):
  sweep   mc_ref_pp over a mesh (all faces, or a facelist), J chunk processes,
          then (unless "assemble": false) assemble_vf_haworth.py with --sweep-logs
          -> <csr> + .provenance.json + .vf_stats.json (C4). Times the sweep and
          the assembly separately; peak RSS of the assembly via /usr/bin/time -l.
  bowl    bowl_crater_ptm in PTM_BOWL_RADIOSITY mode on one d/D, with the exact,
          sparse_production or CSR estimator, then accuracy.py vs Ingersoll and
          vs a same-N reference dump.
  engine  ptm_thermal_sim on the template (or a given base config) + overrides,
          PTM_TIMING_JSON set, PTM_HMATRIX_VF_REPLACE over a make_empty_vf stub
          when a CSR is given (and the log must then say "REPLACED H-matrix").

Every record carries provenance (git SHA, exe md5 and EVERY linked dylib md5,
Embree library), the machine state, and shared_machine=true if --allow-shared
was needed. Nothing is ever timed silently on a busy machine.

  run_bench.py --plan plan.json [--only REGEX] [--dry-run] [--allow-shared]
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (REPO, Ledger, binary_provenance, git_state, log, machine_state,  # noqa: E402
                    md5, require_quiet, run_stamped, which_python_has)

PTM12 = REPO / "scripts/ptm12_diag"
TEMPLATE = Path(__file__).resolve().parent / "templates/moon_lunation.yaml"
# GPU safety (plan 4.5): applied to every row that enables a GPU. Coupled rows N=1.
GPU_SAFE = {"thermal.gpu_batch_size": 1280, "thermal.gpu_inter_batch_pause_ms": 1500}
GPU_ENV = {"PTM_RAD_RERUN_MAX_GPUS": "1"}


def set_dotted(d, key, val):
    cur = d
    parts = key.split(".")
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = val


def get_dotted(d, key, default=None):
    cur = d
    for p in key.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def resolve_mc_ref_pp(embree_root):
    sys.path.insert(0, str(PTM12))
    os.environ["PTM_EMBREE_ROOT"] = str(embree_root)
    import ensure_mc_ref_pp
    return ensure_mc_ref_pp.resolve_current()   # smoke-tested inside (PTM-711)


# ------------------------------------------------------------------ sweep
def run_sweep(row, a, rundir):
    mc = resolve_mc_ref_pp(a.embree_root)
    mesh = Path(row["mesh"])
    nf = int(row["n_faces"])
    faces = (Path(row["facelist"]).read_text().split() if row.get("facelist")
             else [str(i) for i in range(nf)])
    J = min(int(row.get("jobs", a.jobs)), len(faces))
    import numpy as np
    tmax = rundir / "tmax.bin"
    tmax.write_bytes(np.int32(nf).tobytes() + np.full(nf, 250.0, np.float32).tobytes())
    chunks = [faces[k::J] for k in range(J)]   # strided: balances near/far faces
    procs, t0 = [], time.time()
    (rundir / "sweep").mkdir(exist_ok=True)
    env = dict(os.environ)
    if row.get("ring"):
        env["PTM_PP_RING_PLY"] = row["ring"]
    assemble = row.get("assemble", True)
    for k, ch in enumerate(chunks):
        fl = rundir / "sweep" / f"chunk_{k:03d}.txt"
        fl.write_text("\n".join(ch) + "\n")
        out = open(rundir / "sweep" / f"out_{k:03d}.txt", "w")
        err = open(rundir / "sweep" / f"log_{k:03d}.txt", "w")
        procs.append(subprocess.Popen(
            [mc, str(mesh), str(tmax), str(fl), str(row["nrays"]), str(row.get("offset", 1e-4)),
             str(row.get("keps", 0.2)), "1",
             # M1: a row that is never assembled writes no partner bins at all; its nnz
             # comes from the STATS lines (a bracket, see below), its F_sky from stdout.
             str(rundir / "sweep" / f"pp_{k:03d}.bin") if assemble else "/dev/null"],
            stdout=out, stderr=err, env=env))
    rcs = [p.wait() for p in procs]
    sweep_s = time.time() - t0
    if any(rcs):
        raise RuntimeError(f"mc_ref_pp failed in {sum(1 for r in rcs if r)} chunk(s)")
    res = {"sweep_wall_s": sweep_s, "jobs": J, "mc_ref_pp": mc, "mc_ref_pp_md5": md5(mc)}
    if assemble:
        csr = rundir / "vf.csr"
        cmd = [sys.executable, str(PTM12 / "assemble_vf_haworth.py"), str(mesh),
               str(rundir / "sweep/pp_*.bin"), str(row["nrays"]), "1", str(csr),
               "--sweep-logs", str(rundir / "sweep/log_*.txt")]
        rc, wall, rss = run_stamped(cmd, rundir / "assemble.log")
        if rc != 0:
            raise RuntimeError(f"assembly failed rc={rc}")
        res.update(assemble_wall_s=wall, assemble_peak_rss_bytes=rss, csr=str(csr),
                   vf_stats=json.loads((rundir / "vf.csr.vf_stats.json").read_text()))
        # M2: the bins are consumed; the CSR, its provenance and vf_stats sidecar carry
        # everything downstream needs. Record what is deleted, then delete it.
        bins = sorted((rundir / "sweep").glob("pp_*.bin"))
        res["bins_deleted"] = {"files": len(bins), "bytes": sum(b.stat().st_size for b in bins)}
        for b in bins:
            b.unlink()
    else:
        sys.path.insert(0, str(PTM12))
        import vf_stats
        tot, nl, nfiles, missing = vf_stats.parse_sweep_stats(str(rundir / "sweep/log_*.txt"))
        res["sweep_stats"] = {**tot, "stats_lines": nl, "files": nfiles, "missing": missing}
        if not row.get("facelist") and tot.get("partners_kept") is not None:
            D = int(tot["partners_kept"])
            res["nnz_bracket"] = {
                "lo": D, "hi": 2 * D, "directed_partners_D": D,
                "label": "BRACKET, not a count: the symmetric matrix would store between D "
                         "(every pair seen from both ends) and 2D (every pair seen from one "
                         "end) entries. No bins written, no assembly (M1)."}
        if row.get("dd") is not None and row.get("facelist"):
            # floor-only bowl sweep: F_sky straight from the referee rows (no assembly)
            r = subprocess.run([sys.executable, str(Path(__file__).resolve().parent / "accuracy.py"),
                                "floor", str(rundir / "sweep/out_*.txt"), "--dd", str(row["dd"]),
                                "--mesh", str(mesh),
                                "--json", str(rundir / "accuracy.json")],
                               capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError("accuracy.py floor failed: " + r.stderr[-2000:])
            res["accuracy"] = json.loads((rundir / "accuracy.json").read_text())
    return res


# ------------------------------------------------------------------ bowl
def run_bowl(row, a, rundir):
    env = dict(os.environ, OMP_NUM_THREADS=str(row.get("threads", a.threads)),
               PTM_BOWL_NRINGS=str(row["n_rings"]), PTM_BOWL_NAZ=str(row["n_az"]),
               PTM_BOWL_NFLAT=str(row["n_flat"]), PTM_BOWL_RADIOSITY="1",
               PTM_RAD_TWO_BAND="1", PTM_SOLAR_ALBEDO="0.12",
               PTM_BOWL_DUMP=str(rundir / "d"))
    for k in ("PTM_BOWL_VF_CSR", "PTM_BOWL_ESTIMATOR"):
        env.pop(k, None)
    est = row["estimator"]
    if est == "csr":
        env["PTM_BOWL_VF_CSR"] = row["csr"]
    elif est == "sparse_production":
        env["PTM_BOWL_ESTIMATOR"] = "sparse_production"
    elif est != "exact":
        raise SystemExit(f"FATAL: unknown bowl estimator {est}")
    rc, wall, rss = run_stamped([a.bowl_bin, rundir, row["dd"]], rundir / "bowl.log", env=env)
    if rc != 0:
        raise RuntimeError(f"bowl_crater_ptm rc={rc}")
    acc = [sys.executable, str(Path(__file__).resolve().parent / "accuracy.py"), "bowl", str(rundir),
           "--dump", str(rundir / "d"), "--json", str(rundir / "accuracy.json")]
    if row.get("ref_dump"):
        acc += ["--ref-dump", row["ref_dump"]]
    r = subprocess.run(acc, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("accuracy.py failed: " + r.stderr[-2000:])
    return {"wall_s": wall, "peak_rss_bytes": rss,
            "accuracy": json.loads((rundir / "accuracy.json").read_text())}


# ------------------------------------------------------------------ engine
def run_engine(row, a, rundir):
    import yaml
    base = Path(row.get("base_config", TEMPLATE))
    cfg = yaml.safe_load(base.read_text())
    out = rundir / "out"
    set_dotted(cfg, "mesh.import_file", row["mesh"])
    set_dotted(cfg, "output.directory", str(out))
    env = dict(os.environ, PTM_TIMING_JSON=str(rundir / "timing.json"),
               PTM_RAD_COST_REPORT="1", OMP_NUM_THREADS=str(row.get("threads", a.threads)))
    env.pop("PTM_HMATRIX_VF_REPLACE", None)
    for k, v in row.get("overrides", {}).items():
        set_dotted(cfg, k, v)
    gpu = bool(get_dotted(cfg, "thermal.use_gpu")) or bool(get_dotted(cfg, "radiation.use_gpu"))
    if gpu:
        if not a.gpu_ok:
            raise SystemExit(f"FATAL: row {row['id']} enables a GPU; pass --gpu-ok only when "
                             "the GPU is yours (never while Stage 1/3 own it).")
        for k, v in GPU_SAFE.items():
            if get_dotted(cfg, k) is None:
                set_dotted(cfg, k, v)
        env.update(GPU_ENV)
    if get_dotted(cfg, "radiation.enabled"):
        if not row.get("csr"):
            raise SystemExit(f"FATAL: coupled row {row['id']} has no CSR. A silent in-engine "
                             "VF build on a large lunar mesh is the wedge trigger (plan 4.5).")
        stub = rundir / "empty_stub.vf"
        if not stub.exists():
            subprocess.run([str(a.bin_dir / "make_empty_vf"), "--faces", str(row["n_faces"]),
                            str(stub)], check=True, capture_output=True)
        set_dotted(cfg, "radiation.view_factor_file", str(stub))
        env["PTM_HMATRIX_VF_REPLACE"] = row["csr"]
    else:
        set_dotted(cfg, "radiation.view_factor_file", "")
    for k, v in row.get("env", {}).items():
        env[k] = str(v)
    cfg_path = rundir / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    rc, wall, rss = run_stamped([a.bin_dir / "ptm_thermal_sim", "--config", cfg_path],
                                rundir / "engine.log", env=env, cwd=REPO)
    if rc != 0:
        raise RuntimeError(f"ptm_thermal_sim rc={rc}")
    logtxt = (rundir / "engine.log").read_text(errors="replace")
    if get_dotted(cfg, "radiation.enabled") and "REPLACED H-matrix" not in logtxt:
        raise RuntimeError("coupled run did not log 'REPLACED H-matrix': timing would describe "
                           "a different operator")
    t = json.loads((rundir / "timing.json").read_text())
    # M4: per-step frames are bulk (~10 kB/face at 360 steps) and not part of the
    # benchmark product; statistics/, timing.json and the small top-level files stay.
    freed = prune_frames(out)
    m = re.search(r"\[VF-STATS\][^\n]*", logtxt)
    return {"wall_s": wall, "peak_rss_bytes": rss, "timing": t, "frames_deleted": freed,
            "vf_stats_banner": m.group(0) if m else None,
            "rad_cost": (re.findall(r"\[rad-cost\][^\n]*", logtxt) or [None])[-1]}


KINDS = {"sweep": run_sweep, "bowl": run_bowl, "engine": run_engine}

GiB = 1024 ** 3


def dir_bytes(d):
    return sum(f.stat().st_size for f in Path(d).rglob("*") if f.is_file())


def prune_frames(out):
    """M4: delete every subdirectory of an engine output dir except statistics/."""
    out = Path(out)
    if not out.is_dir():
        return {"dirs": [], "bytes": 0}
    gone, nbytes = [], 0
    for d in sorted(out.iterdir()):
        if d.is_dir() and d.name != "statistics":
            nbytes += dir_bytes(d)
            shutil.rmtree(d)
            gone.append(d.name)
    return {"dirs": gone, "bytes": nbytes}


def expanded_ids(row, a):
    """The run ids a plan row expands to (same rule the main loop uses)."""
    reps = row.get("repeats", 1)
    if a.repeat and reps > 1:
        reps = a.repeat
    return [row["id"]] + [f"{row['id']}#r{k}" for k in range(1, reps)]


def csr_lifecycle(plan_rows, root, a):
    """{csr path produced by a sweep row of THIS plan: [every consumer run id]}.
    Built over the WHOLE plan (not the --only subset), so a filtered invocation can
    never delete a matrix a later row still needs. External matrices (the paper's
    v4 CSRs) are not produced here and are never touched."""
    produced = {}
    for r in plan_rows:
        if r["kind"] == "sweep" and r.get("assemble", True):
            produced[str(root / r["id"].replace("#", "_") / "vf.csr")] = []
    for r in plan_rows:
        c = r.get("csr")
        if c in produced:
            produced[c] += expanded_ids(r, a)
    return produced


def collect_garbage(lifecycle, led, root):
    """M3: delete a harness-produced CSR once EVERY consumer run is done (a failed or
    pending consumer keeps it). md5, bytes, nnz and the vf_stats sidecar are written
    to the ledger first; the sidecar, provenance and rowsum files stay on disk."""
    for path, consumers in lifecycle.items():
        p = Path(path)
        if not p.exists() or not all(led.done(c) for c in consumers):
            continue
        assert Path(root).resolve() in p.resolve().parents, p   # never outside the run root
        with open(p, "rb") as fh:
            n = int.from_bytes(fh.read(4), "little")
            fh.seek(4 + 8 + 8 * (n + 1))
            nnz = int.from_bytes(fh.read(8), "little")
        side = Path(str(p) + ".vf_stats.json")
        rec = {"path": str(p), "md5": md5(p), "bytes": p.stat().st_size, "n_faces": n,
               "nnz": nnz, "consumers": consumers,
               "vf_stats": json.loads(side.read_text()) if side.exists() else None}
        led.write(id=str(p), event="csr_deleted", record=rec)
        p.unlink()
        log(f"M3: deleted {p.name} of {p.parent.name} ({rec['bytes'] / GiB:.2f} GiB, md5 "
            f"{rec['md5']}) after its last consumer")


def disk_guard(row, rid, root, a, led):
    """Refuse to START a row unless free >= 2 x its predicted disk peak + a fixed
    reserve. A failure aborts the whole batch cleanly, between rows, never mid-row."""
    peak = row.get("predicted_disk_peak_bytes")
    if peak is None:
        raise SystemExit(f"FATAL: row {row['id']} has no predicted_disk_peak_bytes; this plan "
                         "predates the disk model. Regenerate it with plan.py.")
    free = shutil.disk_usage(root).free
    need = 2.0 * peak + a.disk_reserve_gib * GiB
    if free < need:
        led.write(id=rid, event="abort_disk", free_bytes=free, need_bytes=need,
                  predicted_disk_peak_bytes=peak, reserve_gib=a.disk_reserve_gib)
        log(f"ABORT (disk guard) before {rid}: free {free / GiB:.1f} GiB < 2 x predicted "
            f"peak {peak / GiB:.2f} GiB + reserve {a.disk_reserve_gib:.0f} GiB = "
            f"{need / GiB:.1f} GiB. No row was started; resume after freeing space.")
        sys.exit(3)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--root", default=None, help="run root (default: plan's 'root')")
    ap.add_argument("--only", default=None, help="regex on run ids")
    ap.add_argument("--bin-dir", default=str(REPO / "build/bin"))
    ap.add_argument("--embree-root", default=str(Path.home() / "opt/embree-4.3.3-avx512"))
    ap.add_argument("--jobs", type=int, default=48, help="mc_ref_pp chunk processes")
    ap.add_argument("--threads", type=int, default=48, help="OMP threads (bowl, engine)")
    ap.add_argument("--allow-shared", action="store_true",
                    help="SMOKE TESTS ONLY: run on a busy machine; records are marked")
    ap.add_argument("--gpu-ok", action="store_true", help="permit GPU rows (never during Stage 1/3)")
    ap.add_argument("--repeat", type=int, default=None,
                    help="override the repeat count of rows the plan already repeats (timing rows)")
    ap.add_argument("--quiet-wait-min", type=float, default=20.0,
                    help="max wait for the machine to go quiet before each row")
    ap.add_argument("--disk-reserve-gib", type=float, default=15.0,
                    help="fixed reserve on top of 2 x the row's predicted disk peak")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    which_python_has("numpy", "yaml")
    plan = json.loads(Path(a.plan).read_text())
    root = Path(a.root or plan["root"]).expanduser()
    a.bin_dir = Path(a.bin_dir)
    a.bowl_bin = str(a.bin_dir / "bowl_crater_ptm")
    rows = [r for r in plan["runs"] if not a.only or re.search(a.only, r["id"])]
    led = Ledger(root / "ledger.jsonl")
    if a.dry_run:
        for r in rows:
            print(("DONE " if led.done(r["id"]) else "todo ") + r["id"])
        return
    if not (REPO / "data/kernels").exists():
        raise SystemExit("FATAL: data/kernels missing in the worktree (SPICE is CWD-relative); "
                         "run scripts/fetch_kernels.py --link-from ~/Projects/ptm first")
    prov = {"git": git_state(), "engine": binary_provenance(a.bin_dir / "ptm_thermal_sim"),
            "bowl": binary_provenance(a.bowl_bin), "embree_root": a.embree_root}
    if prov["git"]["dirty_files"]:
        log(f"WARNING: worktree has {prov['git']['dirty_files']} modified tracked files; the "
            "SHA alone does not identify this build")
    lifecycle = csr_lifecycle(plan["runs"], root, a)
    collect_garbage(lifecycle, led, root)   # anything a previous invocation left eligible
    for row in rows:
        reps = row.get("repeats", 1)
        if a.repeat and reps > 1:
            reps = a.repeat
        for rep in range(reps):
            # Repeat 0 keeps the plan id (and directory): dependent rows point at it.
            rid = row["id"] if rep == 0 else f"{row['id']}#r{rep}"
            if led.done(rid):
                continue
            st = require_quiet(allow_shared=a.allow_shared, wait_min=a.quiet_wait_min)
            disk_guard(row, rid, root, a, led)
            rundir = root / rid.replace("#", "_")
            rundir.mkdir(parents=True, exist_ok=True)
            led.write(id=rid, event="start", row=row, machine=st, provenance=prov)
            log(f"run {rid} ({row['kind']})")
            try:
                res = KINDS[row["kind"]](row, a, rundir)
            except Exception as e:   # recorded, then the batch continues
                if row["kind"] == "engine":
                    prune_frames(rundir / "out")   # M4 applies to a failed row as well
                led.write(id=rid, event="fail", error=str(e))
                log(f"FAIL {rid}: {e}")
                continue
            led.write(id=rid, event="done", result=res, machine_after=machine_state(),
                      shared_machine=st["shared_machine"],
                      disk_free_after_bytes=shutil.disk_usage(root).free)
            log(f"done {rid}")
            collect_garbage(lifecycle, led, root)


if __name__ == "__main__":
    main()
