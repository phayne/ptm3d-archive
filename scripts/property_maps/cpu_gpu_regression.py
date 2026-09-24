#!/usr/bin/env python3
"""
Regression: confirm CPU and GPU thermal solvers respond to a per-face H map
the same way (within tolerance), and confirm the map produces a measurable
effect on temperatures.

Runs four simulations:
  cpu_map     CPU + property map enabled
  cpu_no_map  CPU + map disabled (baseline)
  gpu_map     GPU + property map enabled
  gpu_no_map  GPU + map disabled (baseline)

Reports:
  effect_cpu     = |T_cpu_map - T_cpu_no_map|   (per-face H impact on CPU)
  effect_gpu     = |T_gpu_map - T_gpu_no_map|   (per-face H impact on GPU)
  cross         = |effect_cpu - effect_gpu|     (consistency)

Pass criteria:
  1. effect_cpu.max  > 1 K   (the map must measurably change CPU temperatures)
  2. effect_gpu.max  > 1 K   (same for GPU)
  3. cross.mean      < 0.1 K (CPU and GPU agree on the H-driven shift)
  4. cross.max       < 5 K   (worst-face cross-discrepancy bounded by the
                              pre-existing CPU/GPU radiative-BC baseline gap)

The 50 K pre-existing CPU/GPU disagreement on cold polar faces (linearized
radiative BC vs Newton iteration) is a separate issue and is excluded from
the per-face H validation.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml


def write_yaml(src_yaml: Path, dst_yaml: Path,
               output_dir: Path, use_gpu: bool, use_map: bool) -> None:
    cfg = yaml.safe_load(src_yaml.read_text())
    cfg.setdefault("thermal", {})
    cfg["thermal"]["use_gpu"] = use_gpu
    cfg["thermal"]["use_default_property_map"] = use_map
    cfg.setdefault("output", {})
    cfg["output"]["directory"] = str(output_dir)
    dst_yaml.write_text(yaml.dump(cfg, sort_keys=False))


def run(repo: Path, yaml_path: Path) -> int:
    cmd = [str(repo / "build" / "bin" / "ptm_thermal_sim"),
           "--config", str(yaml_path)]
    print(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=repo,
                          stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL).returncode


def load_temps(d: Path) -> np.ndarray:
    files = sorted((d / "temperature").glob("temperature_*.bin"), key=lambda p: int(str(p).rsplit("_", 1)[1][:-4]))
    if not files:
        raise FileNotFoundError(f"no temperature_*.bin in {d}/temperature")
    return np.stack([np.fromfile(f, dtype=np.float32) for f in files])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="examples/moon_ico4_propmap_long.yaml")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--effect-min-K", type=float, default=1.0)
    ap.add_argument("--cross-mean-K", type=float, default=0.1)
    ap.add_argument("--cross-max-K", type=float, default=5.0)
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    src_yaml = repo / args.config

    yaml_dir = Path("/tmp/ptm_propmap_yaml")
    yaml_dir.mkdir(parents=True, exist_ok=True)
    out_root = Path("/tmp/ptm_propmap_regr")

    runs = {
        "cpu_map":    {"yaml": yaml_dir / "cpu_map.yaml",     "out": out_root / "cpu_map",
                       "use_gpu": False, "use_map": True},
        "cpu_no_map": {"yaml": yaml_dir / "cpu_no_map.yaml",  "out": out_root / "cpu_no_map",
                       "use_gpu": False, "use_map": False},
        "gpu_map":    {"yaml": yaml_dir / "gpu_map.yaml",     "out": out_root / "gpu_map",
                       "use_gpu": True,  "use_map": True},
        "gpu_no_map": {"yaml": yaml_dir / "gpu_no_map.yaml",  "out": out_root / "gpu_no_map",
                       "use_gpu": True,  "use_map": False},
    }

    for name, r in runs.items():
        if r["out"].exists():
            shutil.rmtree(r["out"])
        write_yaml(src_yaml, r["yaml"], r["out"], r["use_gpu"], r["use_map"])

    for name, r in runs.items():
        print(f"\n[{name}]")
        rc = run(repo, r["yaml"])
        if rc != 0:
            print(f"  FAILED (return code {rc})")
            return 2

    cpu_map    = load_temps(runs["cpu_map"]["out"])
    cpu_no_map = load_temps(runs["cpu_no_map"]["out"])
    gpu_map    = load_temps(runs["gpu_map"]["out"])
    gpu_no_map = load_temps(runs["gpu_no_map"]["out"])

    n_steps, n_faces = cpu_map.shape
    print(f"\nFrames: {n_steps},  faces: {n_faces}")

    eff_cpu = np.abs(cpu_map.astype(np.float64) - cpu_no_map.astype(np.float64))
    eff_gpu = np.abs(gpu_map.astype(np.float64) - gpu_no_map.astype(np.float64))
    cross   = np.abs(eff_cpu - eff_gpu)

    print(f"\n--- per-face H map effect ---")
    print(f"  CPU |dT|: max={eff_cpu.max():7.3f} K  mean={eff_cpu.mean():7.4f} K  "
          f"faces affected (|dT|>0.1K, any step): {(eff_cpu.max(axis=0) > 0.1).sum()} / {n_faces}")
    print(f"  GPU |dT|: max={eff_gpu.max():7.3f} K  mean={eff_gpu.mean():7.4f} K  "
          f"faces affected (|dT|>0.1K, any step): {(eff_gpu.max(axis=0) > 0.1).sum()} / {n_faces}")

    print(f"\n--- CPU/GPU consistency on the H-driven shift ---")
    print(f"  cross |dT|: max={cross.max():7.3f} K  mean={cross.mean():7.4f} K  "
          f"99th pct={np.percentile(cross, 99):.3f} K")

    checks = [
        ("CPU responds to map (max effect > 1K)",
         eff_cpu.max() > args.effect_min_K),
        ("GPU responds to map (max effect > 1K)",
         eff_gpu.max() > args.effect_min_K),
        (f"cross-mean < {args.cross_mean_K} K",
         cross.mean() < args.cross_mean_K),
        (f"cross-max  < {args.cross_max_K} K",
         cross.max()  < args.cross_max_K),
    ]
    print("\n--- Pass/fail ---")
    all_pass = True
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
        if not ok:
            all_pass = False

    print(f"\n{'PASS' if all_pass else 'FAIL'}: per-face H propagation through CPU and GPU thermal solvers")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
