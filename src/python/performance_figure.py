#!/usr/bin/env python3
"""
F6 — Computational performance scaling figure.

Measures wall-clock time of the PTM3D illumination and thermal modules vs
mesh size for a fixed integration window (1 lunar day, 144 timesteps, no
radiation exchange) on lunar icosphere meshes ico4..ico7. Three thermal
back-ends are compared:

  * CPU (OpenMP, 24 threads)
  * GPU OpenCL (single AMD W6800X die)
  * GPU OpenCL pool (4 AMD W6800X dies via --gpu-devices 4)

The figure overlays the model curves from `plots/performance/runtime_estimates.csv`
as the high-N extrapolation.

Usage
-----
* Generate fresh measurements on this workstation:
    python3 performance_figure.py --bench

* Re-plot from cached CSV (default after first --bench run):
    python3 performance_figure.py
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ptm.pubstyle import apply_style, COLORS, DOUBLE_W, panel_label


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PTM_BIN = PROJECT_ROOT / "build" / "bin" / "ptm_thermal_sim"
SCALING_CSV = PROJECT_ROOT / "plots" / "performance" / "runtime_estimates.csv"
DEFAULT_CACHE = Path("/tmp/ptm_perf_results.csv")

ICO_FACES = {3: 1280, 4: 5120, 5: 20480, 6: 81920, 7: 327680,
             8: 1310720, 9: 5242880, 10: 20971520}

BACKENDS = [
    ("cpu",  {"use_gpu": False,                                    }),
    ("gpu1", {"use_gpu": True,  "gpu_devices": 1                   }),
    ("gpu4", {"use_gpu": True,  "gpu_devices": 4                   }),
]


# ---------------------------------------------------------------------------
# Generate ephemeral YAML config

def make_config(ico: int, backend_args: dict, out_dir: Path) -> Path:
    cfg = f"""
simulation:
  start_time: '2025-01-01T00:00:00'
  duration_days: 1.0
  timesteps: 144
  body: Moon
mesh:
  resolution: {ico}
  use_topographic: false
  import_file: ''
  scale_factor: 1.0
thermal:
  run_thermal_mesh: true
  threads: 0
  equilibration_orbits: 0
  equilibration_steps_per_day: 192
  use_gpu: {str(backend_args["use_gpu"]).lower()}
  gpu_devices: {backend_args.get("gpu_devices", 1)}
  compute_statistics: true
  enable_shadow_flux: false
  crater_dd_ratio: 8.0
  integration_days: -1.0
  integration_steps: -1
  integration_offset_days: 0.0
radiation:
  enabled: false
  use_gpu: false
illumination:
  include_primary_reflection: false
  use_batch_processing: true
  use_albedo_map: false
output:
  directory: {out_dir}
  format: binary
  vtk_mode: thermal
  include_primary_ir: false
  locations_file: ''
  depths_file: ''
advanced:
  verbose: false
"""
    cfg_path = out_dir / "config.yaml"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(cfg)
    return cfg_path


# ---------------------------------------------------------------------------
# Parse timing from console output

TIMING_PATTERNS = {
    "illumination_s": re.compile(
        r"Illumination calculation completed in\s+([\d.]+)\s*seconds"),
    "total_s": re.compile(
        r"Simulation completed in\s+([\d.]+)\s*seconds"),
    "gpu_kernel_ms": re.compile(
        r"\[GPU Thermal\] Persistent kernel:.*?in\s+([\d.]+)\s*ms"),
}


def parse_log(text: str) -> dict:
    out: dict[str, float | None] = {}
    for key, pat in TIMING_PATTERNS.items():
        m = pat.search(text)
        out[key] = float(m.group(1)) if m else None
    # "thermal + I/O" time: end-to-end after illumination. This is what the
    # user experiences as the cost of the post-illumination pipeline (thermal
    # solver, statistics, output) and is comparable across backends.
    if out["total_s"] and out["illumination_s"]:
        out["post_illum_s"] = max(0.0, out["total_s"] - out["illumination_s"])
    else:
        out["post_illum_s"] = None
    # GPU thermal kernel-only time (when present)
    if out.get("gpu_kernel_ms"):
        out["gpu_kernel_s"] = out["gpu_kernel_ms"] / 1000.0
    else:
        out["gpu_kernel_s"] = None
    return out


def run_benchmark(ico: int, backend_name: str, backend_args: dict) -> dict:
    workdir = Path(tempfile.mkdtemp(prefix=f"ptmperf_{backend_name}_ico{ico}_"))
    cfg = make_config(ico, backend_args, workdir / "out")
    print(f"  [{backend_name} ico{ico}]  starting...", flush=True)
    t0 = time.time()
    try:
        proc = subprocess.run(
            [str(PTM_BIN), "--config", str(cfg)],
            capture_output=True, text=True, timeout=900,
            cwd=str(PROJECT_ROOT),
        )
    except subprocess.TimeoutExpired:
        return {"backend": backend_name, "ico": ico,
                "n_faces": ICO_FACES[ico], "wall_s": None,
                "stderr": "timeout"}
    wall = time.time() - t0
    log = proc.stdout + "\n" + proc.stderr
    parsed = parse_log(log)
    rec = {
        "backend": backend_name,
        "ico": ico,
        "n_faces": ICO_FACES[ico],
        "wall_s": wall,
        "rc": proc.returncode,
        **parsed,
    }
    print(f"  [{backend_name} ico{ico}]  wall={wall:.1f}s  "
          f"illum={parsed['illumination_s']}  total={parsed['total_s']}",
          flush=True)
    return rec


def benchmark_all(out_csv: Path, ico_list: list[int]) -> Path:
    rows = []
    for ico in ico_list:
        for name, args in BACKENDS:
            rows.append(run_benchmark(ico, name, args))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  wrote {out_csv}")
    return out_csv


# ---------------------------------------------------------------------------
# Plot

def load_results(path: Path) -> list[dict]:
    out = []
    with open(path) as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            for k in ("wall_s", "illumination_s", "total_s",
                      "post_illum_s", "gpu_kernel_s"):
                if k in r and r[k] not in ("", None):
                    try:
                        r[k] = float(r[k])
                    except ValueError:
                        r[k] = None
                else:
                    r[k] = None
            r["ico"] = int(r["ico"])
            r["n_faces"] = int(r["n_faces"])
            out.append(r)
    return out


def load_scaling_model(path: Path) -> dict:
    """Return parallel arrays (n_faces, t_illum, t_thermal_cpu) from the
    `runtime_estimates.csv` model."""
    n, illum, thermal = [], [], []
    with open(path) as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            try:
                n.append(int(r["N_faces"]))
                illum.append(float(r["Illumination"]))
                thermal.append(float(r["Thermal solver"]))
            except (KeyError, ValueError):
                continue
    order = np.argsort(n)
    return {
        "n_faces":   np.array(n)[order],
        "illum_s":   np.array(illum)[order],
        "thermal_s": np.array(thermal)[order],
    }


def make_figure(results: list[dict], scaling: dict, out_path: Path) -> None:
    apply_style()

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(DOUBLE_W * 1.4, 0.55 * DOUBLE_W * 1.4))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.14, wspace=0.24)

    by_backend: dict[str, list[dict]] = {}
    for r in results:
        by_backend.setdefault(r["backend"], []).append(r)
    for k in by_backend:
        by_backend[k].sort(key=lambda r: r["n_faces"])

    backend_color = {
        "cpu":  COLORS["bad"],
        "gpu1": COLORS["lat45"],
        "gpu4": COLORS["good"],
    }
    backend_label = {
        "cpu":  "CPU thermal",
        "gpu1": r"1$\times$W6800X",
        "gpu4": r"4$\times$W6800X",
    }

    # ---- (a) wall-clock time vs N_faces ----
    illum_runs = [r for r in results if r["backend"] == "cpu"
                  and r.get("illumination_s") is not None]
    if illum_runs:
        nf = [r["n_faces"] for r in illum_runs]
        t = [r["illumination_s"] for r in illum_runs]
        ax1.loglog(nf, t, "s-", color=COLORS["ptm"], mfc="white", mew=1.6,
                   ms=10, lw=2.0, label="Illumination (CPU)")
    for name in ("cpu", "gpu1", "gpu4"):
        rs = by_backend.get(name, [])
        if not rs:
            continue
        nf = [r["n_faces"] for r in rs]
        t = [r.get("post_illum_s") for r in rs]
        if not all(x is not None for x in t):
            continue
        ax1.loglog(nf, t, "o-", color=backend_color[name], mfc="white",
                   mew=1.6, ms=10, lw=2.0,
                   label=f"Thermal {backend_label[name]}")

    ax1.set_xlabel(r"Mesh size $N_\mathrm{faces}$")
    ax1.set_ylabel("Wall-clock time (s)")
    ax1.set_xlim(3e3, 3e5)
    ax1.legend(loc="lower right")
    panel_label(ax1, "(a)", x=0.04, y=0.95)

    # ---- (b) GPU/CPU speedup bar chart ----
    cpu_map = {r["n_faces"]: r.get("post_illum_s")
               for r in by_backend.get("cpu", [])}
    g1_map = {r["n_faces"]: r.get("post_illum_s")
              for r in by_backend.get("gpu1", [])}
    g4_map = {r["n_faces"]: r.get("post_illum_s")
              for r in by_backend.get("gpu4", [])}
    common = sorted(set(cpu_map) & set(g1_map))
    if common:
        x = np.arange(len(common), dtype=float)
        w = 0.38
        speed1 = [cpu_map[n] / g1_map[n] if g1_map[n] else 0 for n in common]
        speed4 = [cpu_map[n] / g4_map.get(n, float("nan"))
                  if g4_map.get(n) else 0 for n in common]
        ax2.bar(x - w / 2, speed1, w,
                color=backend_color["gpu1"], alpha=0.85,
                label=r"1$\times$W6800X / CPU")
        ax2.bar(x + w / 2, speed4, w,
                color=backend_color["gpu4"], alpha=0.85,
                label=r"4$\times$W6800X / CPU")
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"{n:,}" for n in common])
        ax2.set_xlabel(r"Mesh size $N_\mathrm{faces}$")
        ax2.set_ylabel(r"Speedup vs CPU thermal ($\times$)")
        ax2.axhline(1.0, color=COLORS["neutral"], lw=1.0, ls="--")
        ax2.legend(loc="upper left", bbox_to_anchor=(0.10, 0.98))
        ymax = max(max(speed1, default=1.0), max(speed4, default=1.0)) * 1.30
        ax2.set_ylim(0, ymax)
        for xi, s1, s4 in zip(x, speed1, speed4):
            if s1 > 0:
                ax2.text(xi - w / 2, s1 + ymax * 0.02, f"{s1:.1f}",
                         ha="center", fontsize=11)
            if s4 > 0:
                ax2.text(xi + w / 2, s4 + ymax * 0.02, f"{s4:.1f}",
                         ha="center", fontsize=11)
    panel_label(ax2, "(b)", x=0.04, y=0.95)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"  wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bench", action="store_true",
                   help="Run benchmarks now (will take ~10–20 min)")
    p.add_argument("--cache", default=str(DEFAULT_CACHE),
                   help="CSV cache for results")
    p.add_argument("--ico", default="4,5,6",
                   help="Mesh resolutions to bench (comma-separated)")
    p.add_argument("--out", default="docs/figures/v2/performance_scaling.png")
    args = p.parse_args()

    ico_list = [int(x) for x in args.ico.split(",")]
    cache = Path(args.cache)

    if args.bench or not cache.exists():
        benchmark_all(cache, ico_list)
    else:
        print(f"  using cached {cache}")

    results = load_results(cache)
    scaling = load_scaling_model(SCALING_CSV)
    make_figure(results, scaling, Path(args.out))


if __name__ == "__main__":
    main()
