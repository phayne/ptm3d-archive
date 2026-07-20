#!/usr/bin/env python3
"""
Time the CPU and GPU thermal solvers with and without a per-face H map.

For each of {CPU, GPU} × {map enabled, map disabled}, runs the simulation
N times, captures wall-clock from the simulator's own "Thermal computation
completed in ..." line, and reports mean ± std. Surface temperatures are
not compared here (see cpu_gpu_regression.py for that).

Default config: examples/moon_ico6_perf.yaml (ico6, 81920 faces).
"""
from __future__ import annotations

import argparse
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import yaml


TIMING_RE = re.compile(
    r"(?:SIMD Batch thermal computation|GPU.*?completed.*?in|Thermal computation completed in)\s*([0-9.]+)\s*(?:s|seconds)",
    re.IGNORECASE,
)
GPU_TIMING_RE = re.compile(
    r"GPU Thermal.*?solve.*?finished in\s*([0-9.]+)\s*ms",
    re.IGNORECASE,
)


def write_variant(src: Path, dst: Path, out_dir: Path,
                  use_gpu: bool, use_map: bool) -> None:
    cfg = yaml.safe_load(src.read_text())
    cfg.setdefault("thermal", {})
    cfg["thermal"]["use_gpu"] = use_gpu
    cfg["thermal"]["use_default_property_map"] = use_map
    cfg.setdefault("output", {})
    cfg["output"]["directory"] = str(out_dir)
    dst.write_text(yaml.dump(cfg, sort_keys=False))


def parse_timing(text: str) -> float | None:
    """Extract simulator-reported solver wall time, in seconds."""
    matches = TIMING_RE.findall(text)
    if matches:
        try:
            return float(matches[-1])
        except ValueError:
            pass
    matches = GPU_TIMING_RE.findall(text)
    if matches:
        try:
            return float(matches[-1]) / 1000.0
        except ValueError:
            pass
    return None


def run_once(repo: Path, yaml_path: Path) -> tuple[float, float | None]:
    cmd = [str(repo / "build" / "bin" / "ptm_thermal_sim"),
           "--config", str(yaml_path)]
    t0 = time.monotonic()
    p = subprocess.run(cmd, cwd=repo, capture_output=True, text=True)
    wall = time.monotonic() - t0
    if p.returncode != 0:
        sys.stderr.write(p.stderr[-2000:])
        raise RuntimeError(f"sim returned {p.returncode}")
    solver = parse_timing(p.stdout + "\n" + p.stderr)
    return wall, solver


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="examples/moon_ico6_perf.yaml")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--repo", default=".")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    src_yaml = repo / args.config

    yaml_dir = Path("/tmp/ptm_propmap_yaml_perf")
    yaml_dir.mkdir(parents=True, exist_ok=True)
    out_root = Path("/tmp/ptm_propmap_perf")

    variants = [
        ("CPU + no map",  False, False),
        ("CPU + map",     False, True),
        ("GPU + no map",  True,  False),
        ("GPU + map",     True,  True),
    ]

    print(f"Running {args.config}, {args.repeats} repeats per variant\n")
    print(f"{'Variant':<18} {'Wall (s)':>22}    {'Solver-only (s)':>22}")
    print("-" * 70)

    summary = {}
    for name, use_gpu, use_map in variants:
        slug = name.replace(" ", "_").replace("+", "")
        yaml_path = yaml_dir / f"{slug}.yaml"
        out_dir = out_root / slug
        if out_dir.exists():
            shutil.rmtree(out_dir)
        write_variant(src_yaml, yaml_path, out_dir, use_gpu, use_map)

        walls, solvers = [], []
        for _ in range(args.repeats):
            if out_dir.exists():
                shutil.rmtree(out_dir)
            wall, solver = run_once(repo, yaml_path)
            walls.append(wall)
            if solver is not None:
                solvers.append(solver)

        wall_mean = statistics.mean(walls)
        wall_std  = statistics.stdev(walls) if len(walls) > 1 else 0.0
        if solvers:
            sol_mean = statistics.mean(solvers)
            sol_std  = statistics.stdev(solvers) if len(solvers) > 1 else 0.0
            sol_str = f"{sol_mean:7.3f} ± {sol_std:.3f}"
        else:
            sol_str = "n/a"

        summary[name] = {"wall": (wall_mean, wall_std),
                         "solver": (sol_mean, sol_std) if solvers else None}
        print(f"{name:<18} {wall_mean:7.3f} ± {wall_std:.3f}        {sol_str}")

    print("\n--- Per-face H overhead (map vs no-map) ---")
    for solver in ("CPU", "GPU"):
        no_map = summary[f"{solver} + no map"]["wall"][0]
        with_map = summary[f"{solver} + map"]["wall"][0]
        delta = with_map - no_map
        ratio = with_map / no_map if no_map > 0 else float("inf")
        print(f"  {solver}: {with_map:.3f} s with map vs {no_map:.3f} s without "
              f"(Δ = {delta:+.3f} s, ratio {ratio:.3f}×)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
