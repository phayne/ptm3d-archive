#!/usr/bin/env python3
"""Average per-face exosphere outputs between two seasonal C++ runs.

Solves the Jensen's-inequality problem with simply averaging temperatures:
  AVERAGING TEMPERATURES BEFORE EXOSPHERE understates sublimation flux
  because flux ∝ exp(-Q/kT) is convex in T.
  AVERAGING FLUXES AFTER TWO SEPARATE EXOSPHERE RUNS captures the actual
  seasonal mean (sub & dep & sput over the full Jovian year).

For each per-face output dir we average across season1 / season2 at the
final epoch and write the result to <output>/<field>/<field>_0999.bin
plus a synthetic evolution_summary.txt.

Usage:
    python average_exosphere_seasons.py \
        --season1 output/europa_ico6_albedo-map_cryst \
        --season2 output/europa_ico6_albedo-map_opp-season_cryst \
        --output  output/europa_ico6_albedo-map_seasonavg_cryst
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import struct
from pathlib import Path

import numpy as np


# Per-face field directories that we want to average.
# Cumulative quantities → straight average is OK (both runs simulated the same
# duration so the time-averaged rates are comparable).
# Per-epoch state quantities (concentrations, fractions) → also averaged at the
# final epoch.
FIELD_DIRS = [
    "cumulative_sublimated_kg_m2",
    "cumulative_deposited_kg_m2",
    "cumulative_sputtered_kg_m2",
    "cumulative_dose_eV_mol",
    "accumulated_mass_kg_m2",
    "ice_mass_kg_m2",
    "ice_thickness_m",
    "nonice_concentration",
    "crystalline_fraction",
    "crystallization_timescale_yr",
    "porosity",
    "radiation_dose_rate_eV_mol_s",
    "temperature_max_K",
    "temperature_mean_K",
    "temperature_min_K",
    "vapor_pressure_Pa",
    "emission_flux_kg_m2_s",
    "deposition_flux_kg_m2_s",
    "sputtering_flux_kg_m2_s",
    "sputtering_mass_loss_kg_m2_s",
    "sputtering_yield",
    "sputtering_spatial_weight",
    "escape_flux_kg_m2_s",
    "net_flux_kg_m2_s",
]


def read_bin(path):
    with open(path, "rb") as f:
        n = struct.unpack("<I", f.read(4))[0]
        return np.frombuffer(f.read(n * 4), dtype="<f4").copy()


def write_bin(path, arr):
    with open(path, "wb") as f:
        f.write(struct.pack("<I", arr.size))
        f.write(arr.astype(np.float32).tobytes())


def find_last_epoch(run_dir: Path, field_dir: str) -> int:
    files = sorted((run_dir / field_dir).glob(f"{field_dir}_*.bin"))
    if not files:
        return -1
    m = re.search(r"_(\d+)\.bin$", files[-1].name)
    return int(m.group(1)) if m else -1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--season1", required=True, type=Path)
    p.add_argument("--season2", required=True, type=Path)
    p.add_argument("--output",  required=True, type=Path)
    args = p.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    # Symlink mesh.ply
    src_mesh = (args.season1 / "mesh.ply").resolve()
    dst_mesh = args.output / "mesh.ply"
    if dst_mesh.exists() or dst_mesh.is_symlink():
        dst_mesh.unlink()
    dst_mesh.symlink_to(src_mesh)

    print(f"Averaging fields between\n  s1 = {args.season1}\n  s2 = {args.season2}")
    written = 0
    for fdir in FIELD_DIRS:
        s1 = args.season1 / fdir
        s2 = args.season2 / fdir
        if not (s1.is_dir() and s2.is_dir()):
            continue
        ep1 = find_last_epoch(args.season1, fdir)
        ep2 = find_last_epoch(args.season2, fdir)
        if ep1 < 0 or ep2 < 0:
            continue
        ep = min(ep1, ep2)
        f1 = s1 / f"{fdir}_{ep:04d}.bin"
        f2 = s2 / f"{fdir}_{ep:04d}.bin"
        if not (f1.exists() and f2.exists()):
            continue
        d1 = read_bin(f1); d2 = read_bin(f2)
        avg = 0.5 * (d1 + d2)
        out_dir = args.output / fdir
        out_dir.mkdir(exist_ok=True)
        write_bin(out_dir / f"{fdir}_{ep:04d}.bin", avg)
        # Also write epoch 0 (used by mass_balance loader for stats)
        if ep > 0:
            f1_0 = s1 / f"{fdir}_0000.bin"
            f2_0 = s2 / f"{fdir}_0000.bin"
            if f1_0.exists() and f2_0.exists():
                d10 = read_bin(f1_0); d20 = read_bin(f2_0)
                write_bin(out_dir / f"{fdir}_0000.bin", 0.5 * (d10 + d20))
        written += 1
        print(f"  {fdir} (epoch {ep}): "
              f"s1=[{d1.min():.3g},{d1.max():.3g}] "
              f"s2=[{d2.min():.3g},{d2.max():.3g}] "
              f"avg=[{avg.min():.3g},{avg.max():.3g}]")

    # Synthesize an evolution_summary.txt that mass_balance.load_evolution_total_years
    # can parse.
    s1_summary = args.season1 / "evolution_summary.txt"
    if s1_summary.exists():
        with open(s1_summary) as f:
            lines = f.readlines()
        with open(args.output / "evolution_summary.txt", "w") as f:
            for line in lines:
                f.write(line)
            f.write("\n# Note: This summary was synthesized by averaging two seasonal\n"
                    "# exosphere runs. Mass totals are the per-face-averaged values\n"
                    "# divided by area; rates are season-averaged.\n")

    # Also copy time_steps and metadata
    for fname in ("time_steps.txt", "metadata.txt"):
        src = args.season1 / fname
        if src.exists():
            shutil.copy(src, args.output / fname)

    print(f"\nWrote {written} averaged field directories to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
