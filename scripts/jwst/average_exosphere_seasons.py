#!/usr/bin/env python3
"""Average per-face exosphere outputs across N seasonal/orbital-phase runs.

Solves the Jensen's-inequality problem with simply averaging temperatures:
  AVERAGING TEMPERATURES BEFORE EXOSPHERE understates sublimation flux
  because flux ~ exp(-Q/kT) is convex in T.
  AVERAGING FLUXES AFTER N SEPARATE EXOSPHERE RUNS captures the actual
  time-weighted mean over the orbit (sub & dep & sput over the full
  Jupiter year, or any other cycle sampled at N phases).

For each per-face output field we compute the weighted mean across all N
runs at each run's final epoch and write the result to
<output>/<field>/<field>_<epoch>.bin.

evolution_summary.txt is SYNTHESIZED from the averaged per-face fields (M8):
mass totals are sum(weighted-averaged cumulative field [kg/m^2] x face area
[m^2]); concentration/crystallinity statistics and the latitude table come
from the weighted-averaged fields. The summary is refused -- not written,
nonzero exit -- if any two runs disagree on duration or final epoch, or if
a run's own summary is inconsistent with its own fields.

Weights: for N phases evenly spaced in mean anomaly (the Jupiter-year
sampling design), the time-weighted orbital average is exactly the uniform
mean 1/N (mean anomaly is linear in time by definition), so the default
is uniform weights. --weights lets you override this for non-uniform
sampling or a sensitivity study (e.g. an N=2 two-point quadrature).

Usage (N=8 Jupiter-year phases, uniform weights):
    python average_exosphere_seasons.py \
        --runs output/.../phase0 output/.../phase1 ... output/.../phase7 \
        --output output/.../eightphase_avg

Usage (legacy two-season case, explicit for clarity):
    python average_exosphere_seasons.py \
        --runs output/europa_ico6_albedo-map_cryst \
               output/europa_ico6_albedo-map_opp-season_cryst \
        --output output/europa_ico6_albedo-map_seasonavg_cryst
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "python"))
import extract_timeseries as _ets  # noqa: E402

EUROPA_RADIUS_M = 1.5608e6  # matches ptm::exosphere::constants::EUROPA_RADIUS


# Per-face field directories that we want to average.
# Cumulative quantities -> weighted average is OK (all runs simulated the
# same duration so the time-averaged rates are comparable).
# Per-epoch state quantities (concentrations, fractions) -> also averaged at
# each run's final epoch.
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
    "bond_albedo",
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


def face_geometry_m(mesh_path: Path):
    """Face areas [m^2] and centroid latitudes [deg] from a mesh in any length
    unit, using the same uniform unit scale as the C++ exosphere (EXO-02)."""
    v, f, _, _, _ = _ets.read_binary_mesh_flexible(str(mesh_path))
    v = np.asarray(v, dtype=float)
    f = np.asarray(f, dtype=np.int64)
    scale = EUROPA_RADIUS_M / np.linalg.norm(v, axis=1).mean()
    e1 = v[f[:, 1]] - v[f[:, 0]]
    e2 = v[f[:, 2]] - v[f[:, 0]]
    areas = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1) * scale * scale
    rel = areas.sum() / (4.0 * np.pi * EUROPA_RADIUS_M ** 2) - 1.0
    # Same gate as the C++ computeExosphereFaceGeometry(): 1e-2 catches unit
    # errors (off by ~1e6) and non-global meshes but admits coarse icospheres.
    if not abs(rel) < 1e-2:
        raise ValueError(f"total face area differs from 4 pi R^2 by {rel:.3e} (limit 1e-2)")
    c = v[f].mean(axis=1)
    lat = np.degrees(np.arcsin(c[:, 2] / np.linalg.norm(c, axis=1)))
    return areas, lat


_SUMMARY_KEYS = {
    "total_years": r"Total time:\s*([\d.eE+-]+)\s*years",
    "elapsed_years": r"Elapsed time:\s*([\d.eE+-]+)\s*years",
    "epochs": r"Epochs:\s*(\d+)",
    "cycles_per_epoch": r"Cycles per epoch:\s*(\d+)",
    "faces_depleted": r"Faces depleted \(no ice\):\s*(\d+)",
    "sublimated": r"Total sublimated:\s*([\d.eE+-]+)\s*kg",
    "deposited": r"Total deposited:\s*([\d.eE+-]+)\s*kg",
    "sputtered": r"Total sputtered:\s*([\d.eE+-]+)\s*kg",
    "escaped": r"Total escaped:\s*([\d.eE+-]+)\s*kg",
}


def parse_summary(path: Path) -> dict:
    text = path.read_text()
    out = {}
    for key, pat in _SUMMARY_KEYS.items():
        m = re.search(pat, text)
        if m is None:
            raise ValueError(f"{path}: no '{key}' entry")
        out[key] = float(m.group(1))
    return out


def synthesize_summary(runs, weights, areas, lat, fields, epochs_used) -> str:
    """Build evolution_summary.txt text from the N weighted-averaged fields.

    `fields` maps field name -> (list of N per-run arrays, weighted average)
    at each run's own final epoch. Raises ValueError (the caller refuses to
    write a summary) on any inconsistency.
    """
    n = len(runs)
    needed = ["cumulative_sublimated_kg_m2", "cumulative_deposited_kg_m2",
              "cumulative_sputtered_kg_m2", "nonice_concentration"]
    missing = [k for k in needed if k not in fields]
    if missing:
        raise ValueError(f"averaged fields missing: {missing}")
    used = {epochs_used[k] for k in needed}
    if len(used) != 1:
        raise ValueError(f"fields come from different final epochs: {epochs_used}")

    sums = [parse_summary(r / "evolution_summary.txt") for r in runs]
    for key in ("total_years", "elapsed_years", "epochs", "cycles_per_epoch"):
        vals = [s[key] for s in sums]
        spread = max(vals) - min(vals)
        if spread > 1e-6 * max(max(abs(v) for v in vals), 1.0):
            raise ValueError(f"runs disagree on {key}: {vals}")

    # Per-run cross-check: each run's own field totals must reproduce its own
    # summary. Ratio 1 = summary in true units; ratio 1e6 = summary written
    # before the EXO-02 fix (km^2 areas). Anything else is refused.
    unit_factor = []
    for idx, (run, summ) in enumerate(zip(runs, sums)):
        total = float(np.sum(fields["cumulative_sublimated_kg_m2"][0][idx] * areas))
        ratio = total / summ["sublimated"] if summ["sublimated"] > 0 else float("nan")
        if abs(ratio - 1.0) < 0.01:
            unit_factor.append(1.0)
        elif abs(ratio / 1e6 - 1.0) < 0.01:
            print(f"WARNING: {run} evolution_summary.txt predates the EXO-02 area fix: its "
                  f"mass totals are 1e6 too small (field total / summary = {ratio:.4e}). "
                  f"The synthesized totals below use true m^2 areas.", file=sys.stderr)
            unit_factor.append(1e6)
        else:
            raise ValueError(f"{run}: field total {total:.4e} kg does not match its summary "
                              f"{summ['sublimated']:.4e} kg (ratio {ratio:.4e}, expected 1 or 1e6)")

    def total(name):
        return float(np.sum(fields[name][1] * areas))

    escaped = sum(w * s["escaped"] * uf for w, s, uf in zip(weights, sums, unit_factor))
    conc = fields["nonice_concentration"][1]
    # Depleted = no ice left (concentration at 1) in EVERY run, i.e. the
    # weighted-averaged concentration is 1 to float32 precision (only
    # possible if every input was already 1, since weights sum to 1 and
    # concentration is bounded in [0,1]).
    depleted_all = int((conc >= 1.0 - 1e-6).sum())
    per_run_depleted = [int(s["faces_depleted"]) for s in sums]
    L = []
    L.append("Europa Ice Evolution Summary")
    L.append("==============================")
    L.append("")
    L.append(f"SYNTHESIZED: weighted average of {n} runs (not a single simulation).")
    for r, w in zip(runs, weights):
        L.append(f"  run (weight {w:.4f}): {r}")
    L.append(f"  Every number below is computed from the weighted-averaged per-face fields at")
    L.append(f"  epoch {used.pop()}, except 'Total escaped' (weighted mean of the {n} runs'")
    L.append(f"  summaries; no per-face escape field exists). For N phases evenly spaced in")
    L.append(f"  mean anomaly with uniform weights, this is the time-weighted orbital mean")
    L.append(f"  exactly (mean anomaly is linear in time); for N=2 or non-uniform weights it")
    L.append(f"  is a quadrature approximation, not a full-orbit integral.")
    if any(u != 1.0 for u in unit_factor):
        L.append("  PRE-FIX INPUTS: at least one run's summary predates the EXO-02 area fix")
        L.append("  (its mass totals were 1e6 too small); the totals below use true m^2 areas.")
        L.append("  The per-face fields of such a run were also computed with the pre-fix K0")
        L.append("  transport kernel (EXO-01) and diurnal-mean transport temperature (M1),")
        L.append("  so they are NOT a corrected product.")
    L.append("")
    L.append("Simulation Parameters:")
    L.append(f"  Epochs: {int(sums[0]['epochs'])}")
    L.append(f"  Cycles per epoch: {int(sums[0]['cycles_per_epoch'])}")
    L.append(f"  Total time: {sums[0]['total_years']:.3e} years")
    L.append("")
    L.append("Final State (weighted-averaged fields):")
    L.append(f"  Elapsed time: {sums[0]['elapsed_years']:.3e} years")
    L.append(f"  Max non-ice concentration: {conc.max() * 100:.2f} %")
    L.append(f"  Mean non-ice concentration: {conc.mean() * 100:.2f} %")
    L.append(f"  Faces depleted (no ice): {depleted_all} / {conc.size}")
    L.append(f"    (depleted in ALL {n} runs; per-run: {per_run_depleted})")
    L.append("  Converged: not applicable (average of N runs)")
    L.append("")
    L.append("Mass Balance (weighted-averaged fields x face area [m^2]):")
    L.append(f"  Total sublimated: {total('cumulative_sublimated_kg_m2'):.3e} kg")
    L.append(f"  Total deposited: {total('cumulative_deposited_kg_m2'):.3e} kg")
    L.append(f"  Total sputtered: {total('cumulative_sputtered_kg_m2'):.3e} kg")
    L.append(f"  Total escaped: {escaped:.3e} kg")
    L.append("")
    L.append("Latitude Distribution (final non-ice concentration, weighted-averaged):")
    bins = np.clip(((lat + 90.0) // 10.0).astype(int), 0, 17)
    for b in range(18):
        sel = bins == b
        v = conc[sel].mean() * 100 if sel.any() else 0.0
        L.append(f"  {-85.0 + 10.0 * b:6.2f}°: {v:.2f} %")
    if "crystalline_fraction" in fields:
        phi = fields["crystalline_fraction"][1]
        L.append("")
        L.append("Crystallization State (weighted-averaged fields):")
        L.append(f"  Mean crystalline fraction: {phi.mean() * 100:.2f} %")
        L.append(f"  Range: {phi.min() * 100:.2f} % - {phi.max() * 100:.2f} %")
        L.append(f"  Mostly crystalline (>90%): {int((phi > 0.9).sum())} faces")
        L.append(f"  Mostly amorphous (<10%): {int((phi < 0.1).sum())} faces")
    L.append("")
    L.append("Units:")
    L.append("  nonice_concentration: dimensionless [0-1], 0 = pure ice, 1 = no ice")
    L.append("  crystalline_fraction: dimensionless [0-1], 0 = amorphous, 1 = crystalline")
    L.append("  Fluxes: kg/m²/s")
    L.append("  Cumulative values: kg/m²")
    return "\n".join(L) + "\n"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", required=True, type=Path, nargs="+",
                   help="N run output directories to average (N >= 2)")
    p.add_argument("--weights", type=float, nargs="+", default=None,
                   help="N weights, one per --runs entry (default: uniform 1/N; "
                        "renormalized to sum to 1 if they do not already)")
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()

    runs = args.runs
    n = len(runs)
    if n < 2:
        print("ERROR: need at least 2 --runs", file=sys.stderr)
        return 1
    if args.weights is None:
        weights = [1.0 / n] * n
    else:
        if len(args.weights) != n:
            print(f"ERROR: --weights has {len(args.weights)} entries, need {n}", file=sys.stderr)
            return 1
        s = sum(args.weights)
        weights = [w / s for w in args.weights]

    src_mesh = (runs[0] / "mesh.ply").resolve()

    print(f"Averaging fields across {n} runs (weights {['%.4f' % w for w in weights]}):")
    for r in runs:
        print(f"  {r}")

    # Read and average everything in memory first; validate and synthesize the
    # summary BEFORE writing anything, so a refusal leaves no half-built product.
    fields = {}        # name -> (list of N per-run arrays, weighted average) at final epoch
    epochs_used = {}
    epoch0 = {}        # name -> weighted-averaged epoch-0 field
    for fdir in FIELD_DIRS:
        dirs = [r / fdir for r in runs]
        if not all(d.is_dir() for d in dirs):
            continue
        eps = [find_last_epoch(r, fdir) for r in runs]
        if any(e < 0 for e in eps):
            continue
        ep = min(eps)
        files = [r / fdir / f"{fdir}_{ep:04d}.bin" for r in runs]
        if not all(f.exists() for f in files):
            continue
        arrs = [read_bin(f) for f in files]
        if len({a.size for a in arrs}) != 1:
            print(f"WARNING: size mismatch for {fdir} across runs: "
                  f"{[a.size for a in arrs]}", file=sys.stderr)
            continue
        avg = sum(w * a for w, a in zip(weights, arrs))
        fields[fdir] = (arrs, avg)
        epochs_used[fdir] = ep
        # Also average epoch 0 (used by mass_balance loader for stats)
        if ep > 0:
            files0 = [r / fdir / f"{fdir}_0000.bin" for r in runs]
            if all(f.exists() for f in files0):
                arrs0 = [read_bin(f) for f in files0]
                epoch0[fdir] = sum(w * a for w, a in zip(weights, arrs0))

    # Synthesize evolution_summary.txt from the averaged fields (M8); it keeps
    # the "Total time: ... years" line that mass_balance.load_evolution_total_years
    # parses. On any inconsistency, refuse: write nothing and exit nonzero.
    summary_path = args.output / "evolution_summary.txt"
    try:
        areas, lat = face_geometry_m(src_mesh)
        text = synthesize_summary(runs, weights, areas, lat, fields, epochs_used)
    except (ValueError, FileNotFoundError) as exc:
        print(f"ERROR: refusing to write {args.output}: {exc}", file=sys.stderr)
        if summary_path.exists():
            summary_path.unlink()
            print(f"ERROR: removed stale {summary_path} from an earlier run", file=sys.stderr)
        return 2

    args.output.mkdir(parents=True, exist_ok=True)
    dst_mesh = args.output / "mesh.ply"
    if dst_mesh.exists() or dst_mesh.is_symlink():
        dst_mesh.unlink()
    dst_mesh.symlink_to(src_mesh)
    written = 0
    for fdir, (arrs, avg) in fields.items():
        ep = epochs_used[fdir]
        out_dir = args.output / fdir
        out_dir.mkdir(exist_ok=True)
        write_bin(out_dir / f"{fdir}_{ep:04d}.bin", avg)
        if fdir in epoch0:
            write_bin(out_dir / f"{fdir}_0000.bin", epoch0[fdir])
        written += 1
        ranges = ", ".join(f"[{a.min():.3g},{a.max():.3g}]" for a in arrs)
        print(f"  {fdir} (epoch {ep}): per-run={ranges} avg=[{avg.min():.3g},{avg.max():.3g}]")
    summary_path.write_text(text)
    print(f"Wrote synthesized {summary_path}")

    # Also copy time_steps and metadata from the first run
    for fname in ("time_steps.txt", "metadata.txt"):
        src = runs[0] / fname
        if src.exists():
            shutil.copy(src, args.output / fname)

    print(f"\nWrote {written} averaged field directories to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
