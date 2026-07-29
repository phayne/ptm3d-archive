#!/usr/bin/env python3
"""
R2 self-chunking acceptance suite (Fermi's tests A-D) — TURNKEY for the GPU gap.

Runs ONLY when no ptm_thermal_sim is on the GPU (it refuses otherwise). Fermi's
bar is BIT-IDENTICAL (0 differing bytes), not a tolerance. Tests:

  A) UNCHUNKED byte-regression vs a pre-R2 build. Runs the pre-R2 binary and the
     R2 binary with PTM_MAX_TIMESTEPS_PER_DISPATCH huge (=> single dispatch,
     t_offset==0 path). statistics/*.bin must be byte-for-byte identical. This
     proves the kernel t_offset arg did not perturb codegen.

  B) SPLIT-vs-UNSPLIT: the R2 binary at PTM_MAX_TIMESTEPS_PER_DISPATCH in
     {huge, 240, 50, 7}. Every split run must be byte-identical to the huge
     (unsplit) baseline. Use a config that exercises: ragged last chunk
     (timesteps % 7 != 0), the Schorghofer/ice accumulator (accum boundary
     crossing), and equilibration cycles (multi-cycle).

  C) BOWL-CRATER with chunking forced ON (PTM_MAX_TIMESTEPS_PER_DISPATCH=7): the
     analytic radiation gate must still PASS.

  D) WEDGE-EFFICACY on the production config: run with the profiled ceiling and
     PTM_LOG_SUBDISPATCH=1; assert every sub-dispatch wall stayed under the
     safe ceiling (proves the wedge is actually prevented).

Usage (after the year-long run + radoff finish, GPU free):
    python3 scripts/ptm_r2_acceptance.py \
        --r2-bin      build/bin/ptm_thermal_sim \
        --pre-r2-bin  ~/projects/ptm-bin/ptm_thermal_sim \
        --config      <small test config with eq cycles + ice accumulator, ts%7!=0> \
        --bowl-bin    build/bin/test_bowl_crater \
        --prod-config <production config> \
        --work-dir    /tmp/r2_accept --tests ABCD
"""
from __future__ import annotations
import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

HUGE = 1_000_000_000


def gpu_busy() -> bool:
    try:
        out = subprocess.run(["pgrep", "-f", "ptm_thermal_sim --config"],
                             capture_output=True, text=True)
        return bool(out.stdout.strip())
    except Exception:
        return False


def stats_digest(run_dir: Path) -> str | None:
    """md5 over the sorted statistics/*.bin so two runs can be compared byte-exact."""
    sdir = run_dir / "statistics"
    if not sdir.is_dir():
        return None
    h = hashlib.md5()
    files = sorted(sdir.glob("*.bin"))
    if not files:
        return None
    for f in files:
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def run_sim(bin_path: Path, config: Path, out_dir: Path, ceiling: int | None,
            log_sub: bool = False) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    if ceiling is not None:
        env["PTM_MAX_TIMESTEPS_PER_DISPATCH"] = str(ceiling)
    if log_sub:
        env["PTM_LOG_SUBDISPATCH"] = "1"
    # The config must point output at out_dir; we copy+rewrite output.directory.
    cfg = out_dir / "config.yaml"
    _rewrite_output_dir(config, cfg, out_dir)
    log = out_dir / "run.log"
    with open(log, "w") as lf:
        return subprocess.run([str(bin_path), "--config", str(cfg)],
                              stdout=lf, stderr=subprocess.STDOUT, env=env).returncode


def _rewrite_output_dir(src: Path, dst: Path, out_dir: Path):
    try:
        import yaml
        cfg = yaml.safe_load(src.read_text())
        cfg.setdefault("output", {})["directory"] = str(out_dir)
        dst.write_text(yaml.safe_dump(cfg, sort_keys=False))
    except Exception:
        # No yaml or unparseable -> copy verbatim and rely on the config's own dir.
        shutil.copy2(src, dst)


def test_A(args, work: Path) -> bool:
    print("\n=== TEST A: unchunked byte-regression (R2 huge-ceiling) vs pre-R2 ===")
    pre = work / "A_pre_r2"
    r2 = work / "A_r2_unsplit"
    run_sim(Path(args.pre_r2_bin), Path(args.config), pre, ceiling=None)
    run_sim(Path(args.r2_bin), Path(args.config), r2, ceiling=HUGE)
    d_pre, d_r2 = stats_digest(pre), stats_digest(r2)
    ok = d_pre is not None and d_pre == d_r2
    print(f"  pre-R2  digest: {d_pre}\n  R2/huge digest: {d_r2}\n  A: "
          + ("PASS (byte-identical)" if ok else "FAIL (bytes differ or missing)"))
    return ok


def test_B(args, work: Path) -> bool:
    print("\n=== TEST B: split-vs-unsplit (ceilings huge,240,50,7) ===")
    base = work / "B_huge"
    run_sim(Path(args.r2_bin), Path(args.config), base, ceiling=HUGE)
    d_base = stats_digest(base)
    print(f"  baseline (huge) digest: {d_base}")
    ok = d_base is not None
    for c in (240, 50, 7):
        d = work / f"B_ceil{c}"
        run_sim(Path(args.r2_bin), Path(args.config), d, ceiling=c)
        dig = stats_digest(d)
        this_ok = dig is not None and dig == d_base
        ok = ok and this_ok
        print(f"  ceiling={c:>4} digest: {dig}  -> "
              + ("PASS" if this_ok else "FAIL (must be byte-identical to baseline)"))
    print("  B: " + ("PASS" if ok else "FAIL"))
    return ok


def test_C(args, work: Path) -> bool:
    print("\n=== TEST C: bowl-crater with chunking forced ON (ceiling=7) ===")
    if not args.bowl_bin:
        print("  C: SKIP (no --bowl-bin)"); return True
    env = dict(os.environ); env["PTM_MAX_TIMESTEPS_PER_DISPATCH"] = "7"
    r = subprocess.run([str(args.bowl_bin)], capture_output=True, text=True, env=env)
    passed = ("PASS" in r.stdout) and r.returncode == 0 and "FAIL" not in r.stdout.upper().replace("PASS", "")
    print(r.stdout[-800:])
    print("  C: " + ("PASS" if passed else "FAIL (bowl-crater gate)"))
    return passed


def test_D(args, work: Path) -> bool:
    print("\n=== TEST D: wedge-efficacy on the production config ===")
    if not args.prod_config:
        print("  D: SKIP (no --prod-config)"); return True
    prof_ceiling = args.ceiling or 240
    d = work / "D_prod"
    run_sim(Path(args.r2_bin), Path(args.prod_config), d, ceiling=prof_ceiling,
            log_sub=True)
    log = (d / "run.log").read_text(errors="replace")
    subs = [float(m) for m in re.findall(r"\[subdispatch\].*in ([\d.]+) ms", log)]
    hung = "gpusKillClientExt" in log or "GPU hang" in log
    if not subs:
        print("  D: INCONCLUSIVE (no [subdispatch] lines — did the run reach the "
              "GPU thermal phase? check run.log)")
        return False
    worst = max(subs)
    budget_ms = args.safe_ceiling_s * 1000.0
    ok = (not hung) and worst <= budget_ms
    print(f"  {len(subs)} sub-dispatches; worst {worst:.0f} ms; budget "
          f"{budget_ms:.0f} ms; hung={hung}")
    print("  D: " + ("PASS (all bounded, no wedge)" if ok else "FAIL"))
    return ok


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--r2-bin", required=True)
    p.add_argument("--pre-r2-bin", default="")
    p.add_argument("--config", default="")
    p.add_argument("--bowl-bin", default="")
    p.add_argument("--prod-config", default="")
    p.add_argument("--work-dir", default="/tmp/r2_accept")
    p.add_argument("--tests", default="ABCD", help="subset of tests to run, e.g. AB")
    p.add_argument("--ceiling", type=int, default=0, help="ceiling for test D (0=240)")
    p.add_argument("--safe-ceiling-s", type=float, default=25.0)
    p.add_argument("--force", action="store_true",
                   help="run even if a ptm_thermal_sim is detected (NOT recommended)")
    args = p.parse_args(argv)

    if gpu_busy() and not args.force:
        sys.exit("REFUSING: a ptm_thermal_sim is running on the GPU. R2 acceptance "
                 "tests are GPU tests — run them only in the post-sim gap (or --force).")

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    results = {}
    if "A" in args.tests:
        if not args.pre_r2_bin or not args.config:
            print("A: SKIP (need --pre-r2-bin and --config)"); results["A"] = None
        else:
            results["A"] = test_A(args, work)
    if "B" in args.tests:
        results["B"] = test_B(args, work) if args.config else (print("B: SKIP (need --config)") or None)
    if "C" in args.tests:
        results["C"] = test_C(args, work)
    if "D" in args.tests:
        results["D"] = test_D(args, work)

    print("\n=== SUMMARY ===")
    for k, v in results.items():
        print(f"  {k}: {'PASS' if v else ('SKIP' if v is None else 'FAIL')}")
    hard = [v for v in results.values() if v is not None]
    return 0 if all(hard) else 1


if __name__ == "__main__":
    sys.exit(main())
