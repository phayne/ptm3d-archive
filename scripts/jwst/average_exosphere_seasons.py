#!/usr/bin/env python3
"""Average per-face exosphere outputs between two seasonal C++ runs.

Solves the Jensen's-inequality problem with simply averaging temperatures:
  AVERAGING TEMPERATURES BEFORE EXOSPHERE understates sublimation flux
  because flux ∝ exp(-Q/kT) is convex in T.
  AVERAGING FLUXES AFTER TWO SEPARATE EXOSPHERE RUNS captures the actual
  seasonal mean (sub & dep & sput over the full Jovian year).

For each per-face output dir we average across season1 / season2 at the
final epoch and write the result to <output>/<field>/<field>_0999.bin.

evolution_summary.txt is SYNTHESIZED from the averaged per-face fields (M8):
mass totals are sum(averaged cumulative field [kg/m^2] x face area [m^2]),
concentration/crystallinity statistics and the latitude table come from the
averaged fields. (An earlier version copied season 1's summary verbatim and
appended a note claiming it was an average.) The summary is refused -- not
written, nonzero exit -- if the two seasons disagree on duration or final
epoch, or if a season's own summary is inconsistent with its fields.

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
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "python"))
import extract_timeseries as _ets  # noqa: E402

EUROPA_RADIUS_M = 1.5608e6  # matches ptm::exosphere::constants::EUROPA_RADIUS


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
    if not abs(rel) < 1e-3:
        raise ValueError(f"total face area differs from 4 pi R^2 by {rel:.3e} (limit 1e-3)")
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


def synthesize_summary(args, areas, lat, fields, epochs_used) -> str:
    """Build evolution_summary.txt text from the averaged fields.

    `fields` maps field name -> (season1, season2, average) arrays at the final
    epoch. Raises ValueError (the caller refuses to write a summary) on any
    inconsistency.
    """
    needed = ["cumulative_sublimated_kg_m2", "cumulative_deposited_kg_m2",
              "cumulative_sputtered_kg_m2", "nonice_concentration"]
    missing = [k for k in needed if k not in fields]
    if missing:
        raise ValueError(f"averaged fields missing: {missing}")
    used = {epochs_used[k] for k in needed}
    if len(used) != 1:
        raise ValueError(f"fields come from different final epochs: {epochs_used}")

    sums = [parse_summary(args.season1 / "evolution_summary.txt"),
            parse_summary(args.season2 / "evolution_summary.txt")]
    for key in ("total_years", "elapsed_years", "epochs", "cycles_per_epoch"):
        a, b = sums[0][key], sums[1][key]
        if abs(a - b) > 1e-6 * max(abs(a), abs(b), 1.0):
            raise ValueError(f"seasons disagree on {key}: {a} vs {b}")

    # Per-season cross-check: each season's field totals must reproduce its own
    # summary. Ratio 1 = summary in true units; ratio 1e6 = summary written
    # before the EXO-02 fix (km^2 areas). Anything else is refused.
    unit_factor = []
    for idx, (label, summ) in enumerate(zip(("season1", "season2"), sums)):
        total = float(np.sum(fields["cumulative_sublimated_kg_m2"][idx] * areas))
        ratio = total / summ["sublimated"] if summ["sublimated"] > 0 else float("nan")
        if abs(ratio - 1.0) < 0.01:
            unit_factor.append(1.0)
        elif abs(ratio / 1e6 - 1.0) < 0.01:
            print(f"WARNING: {label} evolution_summary.txt predates the EXO-02 area fix: its "
                  f"mass totals are 1e6 too small (field total / summary = {ratio:.4e}). "
                  f"The synthesized totals below use true m^2 areas.", file=sys.stderr)
            unit_factor.append(1e6)
        else:
            raise ValueError(f"{label}: field total {total:.4e} kg does not match its summary "
                             f"{summ['sublimated']:.4e} kg (ratio {ratio:.4e}, expected 1 or 1e6)")

    def total(name):
        return float(np.sum(fields[name][2] * areas))

    escaped = 0.5 * (sums[0]["escaped"] * unit_factor[0] + sums[1]["escaped"] * unit_factor[1])
    conc = fields["nonice_concentration"][2]
    # Depleted = no ice left (concentration at 1) in BOTH seasons, i.e. the
    # averaged concentration is 1 to float32 precision.
    depleted_both = int((conc >= 1.0 - 1e-6).sum())
    L = []
    L.append("Europa Ice Evolution Summary")
    L.append("==============================")
    L.append("")
    L.append("SYNTHESIZED: average of two seasonal runs (not a single simulation).")
    L.append(f"  season1: {args.season1}")
    L.append(f"  season2: {args.season2}")
    L.append(f"  Every number below is computed from the season-averaged per-face fields at")
    L.append(f"  epoch {used.pop()}, except 'Total escaped' (mean of the two seasons' summaries;")
    L.append(f"  no per-face escape field exists). This is a 2-point seasonal quadrature, not")
    L.append(f"  an average over the Jovian orbit.")
    if any(u != 1.0 for u in unit_factor):
        L.append("  PRE-FIX INPUTS: at least one season's summary predates the EXO-02 area fix")
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
    L.append("Final State (season-averaged fields):")
    L.append(f"  Elapsed time: {sums[0]['elapsed_years']:.3e} years")
    L.append(f"  Max non-ice concentration: {conc.max() * 100:.2f} %")
    L.append(f"  Mean non-ice concentration: {conc.mean() * 100:.2f} %")
    L.append(f"  Faces depleted (no ice): {depleted_both} / {conc.size}")
    L.append(f"    (depleted in both seasons; per season: {int(sums[0]['faces_depleted'])} / "
             f"{int(sums[1]['faces_depleted'])})")
    L.append("  Converged: not applicable (average of two runs)")
    L.append("")
    L.append("Mass Balance (season-averaged fields x face area [m^2]):")
    L.append(f"  Total sublimated: {total('cumulative_sublimated_kg_m2'):.3e} kg")
    L.append(f"  Total deposited: {total('cumulative_deposited_kg_m2'):.3e} kg")
    L.append(f"  Total sputtered: {total('cumulative_sputtered_kg_m2'):.3e} kg")
    L.append(f"  Total escaped: {escaped:.3e} kg")
    L.append("")
    L.append("Latitude Distribution (final non-ice concentration, season-averaged):")
    bins = np.clip(((lat + 90.0) // 10.0).astype(int), 0, 17)
    for b in range(18):
        sel = bins == b
        v = conc[sel].mean() * 100 if sel.any() else 0.0
        L.append(f"  {-85.0 + 10.0 * b:6.2f}°: {v:.2f} %")
    if "crystalline_fraction" in fields:
        phi = fields["crystalline_fraction"][2]
        L.append("")
        L.append("Crystallization State (season-averaged fields):")
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
    p.add_argument("--season1", required=True, type=Path)
    p.add_argument("--season2", required=True, type=Path)
    p.add_argument("--output",  required=True, type=Path)
    args = p.parse_args()

    src_mesh = (args.season1 / "mesh.ply").resolve()

    print(f"Averaging fields between\n  s1 = {args.season1}\n  s2 = {args.season2}")
    # Read and average everything in memory first; validate and synthesize the
    # summary BEFORE writing anything, so a refusal leaves no half-built product.
    fields = {}        # name -> (season1, season2, average) at the final epoch
    epochs_used = {}
    epoch0 = {}        # name -> averaged epoch-0 field
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
        fields[fdir] = (d1, d2, 0.5 * (d1 + d2))
        epochs_used[fdir] = ep
        # Also average epoch 0 (used by mass_balance loader for stats)
        if ep > 0:
            f1_0 = s1 / f"{fdir}_0000.bin"
            f2_0 = s2 / f"{fdir}_0000.bin"
            if f1_0.exists() and f2_0.exists():
                epoch0[fdir] = 0.5 * (read_bin(f1_0) + read_bin(f2_0))

    # Synthesize evolution_summary.txt from the averaged fields (M8); it keeps
    # the "Total time: ... years" line that mass_balance.load_evolution_total_years
    # parses. On any inconsistency, refuse: write nothing and exit nonzero.
    summary_path = args.output / "evolution_summary.txt"
    try:
        areas, lat = face_geometry_m(src_mesh)
        text = synthesize_summary(args, areas, lat, fields, epochs_used)
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
    for fdir, (d1, d2, avg) in fields.items():
        ep = epochs_used[fdir]
        out_dir = args.output / fdir
        out_dir.mkdir(exist_ok=True)
        write_bin(out_dir / f"{fdir}_{ep:04d}.bin", avg)
        if fdir in epoch0:
            write_bin(out_dir / f"{fdir}_0000.bin", epoch0[fdir])
        written += 1
        print(f"  {fdir} (epoch {ep}): "
              f"s1=[{d1.min():.3g},{d1.max():.3g}] "
              f"s2=[{d2.min():.3g},{d2.max():.3g}] "
              f"avg=[{avg.min():.3g},{avg.max():.3g}]")
    summary_path.write_text(text)
    print(f"Wrote synthesized {summary_path}")

    # Also copy time_steps and metadata
    for fname in ("time_steps.txt", "metadata.txt"):
        src = args.season1 / fname
        if src.exists():
            shutil.copy(src, args.output / fname)

    print(f"\nWrote {written} averaged field directories to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
