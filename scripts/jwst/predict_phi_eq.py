#!/usr/bin/env python3
"""
Predict equilibrium / time-integrated crystalline fraction φ_C(lat, lon) from
a PTM3D Europa run, using the same physics (Mitchell 2017 + Nordheim 2022 +
Cartwright 2025 calibration) the C++ exosphere module evolves at run time.

Three modes (mutually exclusive):
  (default)      : evaluate φ_eq(T_max) per face. Fast; gives an upper bound
                   on time-integrated φ_C since k_cryst is exponential in T.
  --integrate    : load all per-step temperature files and integrate Eq. D33
                   to steady state. Matches what the C++ run-time evolution
                   would produce if crystallization had been enabled.
  --cpp-output   : load φ_C from a completed C++ exosphere run that already
                   has crystallization enabled. Use this once the C++ binary
                   has been rebuilt with the post-fix dose convention and
                   <k_cryst(T)> averaging.

Output: NPZ with face-aligned arrays {lat, lon, T_max, T_mean, phi_eq_tmax,
phi_C_integrated (if --integrate or --cpp-output), phi_eq_tmean, dose_rate}.

Every crystalline/amorphous face fraction printed below is over the RESOLVED
faces only: a face whose value is not finite has no value and leaves BOTH the
numerator and the denominator (Paul's ruling, 2026-09-15). phi derives from
T_max / T_mean, so a blank temperature propagates straight into it. The
resolved population is printed beside each fraction.

Usage:
  python predict_phi_eq.py --run output/europa_ico6_albedo-map \
                           --out data/jwst/cartwright2025/phi_eq_albedo_map.npz \
                           [--integrate] [--n-orbits 200]
  python predict_phi_eq.py --cpp-output output/europa_ico6_albedo-map_cryst \
                           --out data/jwst/cartwright2025/phi_eq_albedo_map_cpp.npz \
                           [--cpp-epoch -1]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import crystallization as cr  # noqa: E402
import loaders  # noqa: E402


def _find_last_cpp_epoch(run_dir: Path) -> int:
    """Return the highest epoch index among crystalline_fraction_NNNN.bin files."""
    files = sorted((run_dir / "crystalline_fraction").glob("crystalline_fraction_*.bin"), key=lambda p: int(str(p).rsplit("_", 1)[1][:-4]))
    if not files:
        raise FileNotFoundError(
            f"No crystalline_fraction_*.bin files in {run_dir/'crystalline_fraction'}. "
            "Did the C++ run enable --crystallization and --multiscale?")
    return int(files[-1].stem.split("_")[-1])


# (label, population size, excluded count) triples already warned about
_WARNED = set()


def _population(field, label):
    """Faces that have a value, plus how many -- Paul's ruling, 2026-09-15.

    "If a facet has no value, then it should leave the denominator and the
    numerator." phi derives from T_max / T_mean, so a face whose temperature
    was never resolved carries a non-finite phi straight through; it is
    excluded from the population entirely rather than counted as amorphous.
    A finiteness test, not an isnan test: `NaN > 0.5` is False (silently
    "not crystalline") and `-inf < 0.1` is True (silently "amorphous").
    """
    field = np.asarray(field)
    resolved = np.isfinite(field)
    n_resolved = int(np.count_nonzero(resolved))
    n_unresolved = int(resolved.size) - n_resolved
    key = (label, int(resolved.size), n_unresolved)
    if n_unresolved and key not in _WARNED:
        _WARNED.add(key)
        print(f"[predict_phi_eq]  WARNING: {label}: {n_unresolved} of "
              f"{resolved.size} faces have no value (non-finite) and leave BOTH "
              f"the numerator and the denominator; fractions are over "
              f"{n_resolved} faces.", file=sys.stderr)
    return resolved, n_resolved


def _frac(qualifying, resolved, n_resolved):
    """COUNT fraction over the resolved faces (NaN if none are resolved)."""
    if n_resolved == 0:
        return float("nan")
    return float(np.count_nonzero(np.asarray(qualifying) & resolved) / n_resolved)


def _stat(fn, field, resolved, n_resolved):
    """An order statistic over the resolved faces (NaN if none are resolved)."""
    if n_resolved == 0:
        return float("nan")
    return float(fn(np.asarray(field)[resolved]))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--run", type=Path,
                     help="PTM3D thermal run directory (mesh.ply, statistics/, temperature/)")
    src.add_argument("--cpp-output", type=Path,
                     help="Completed C++ exosphere run directory containing "
                          "crystalline_fraction/, temperature_max_K/, mesh.ply")
    p.add_argument("--out", required=True, type=Path, help="Output NPZ path")
    p.add_argument("--integrate", action="store_true",
                   help="(--run mode) Also integrate Eq. D33 over the full diurnal cycle")
    p.add_argument("--n-orbits", type=int, default=200,
                   help="(--run mode) Number of diurnal cycles for time integration (default 200)")
    p.add_argument("--cpp-epoch", type=int, default=-1,
                   help="(--cpp-output mode) Epoch index to load. -1 = last available (default).")
    args = p.parse_args()

    cpp_mode = args.cpp_output is not None
    run = args.cpp_output if cpp_mode else args.run
    if not run.is_dir():
        p.error(f"Run directory {run} does not exist")
    if args.integrate and cpp_mode:
        p.error("--integrate is only valid in --run mode")

    print(f"[predict_phi_eq] Loading mesh from {run}/mesh.ply")
    lat, lon = loaders.load_mesh(run)
    nf = lat.size
    print(f"[predict_phi_eq]   {nf} faces, lat range [{lat.min():.1f}, {lat.max():.1f}],"
          f" lon range [{lon.min():.1f}, {lon.max():.1f}]")

    # Load temperature statistics — schema differs between thermal-run output
    # and C++ exosphere-run output.
    print("[predict_phi_eq] Loading per-face temperature statistics")
    if cpp_mode:
        # C++ exosphere copies T statistics into per-epoch files. Epoch 0 is the
        # original thermal-run statistic; later epochs are identical when albedo
        # coupling is off, so we just pull epoch 0.
        T_max  = loaders.load_per_face_field(run, "temperature_max_K/temperature_max_K_0000.bin",   nf)
        T_mean = loaders.load_per_face_field(run, "temperature_mean_K/temperature_mean_K_0000.bin", nf)
        T_min  = loaders.load_per_face_field(run, "temperature_min_K/temperature_min_K_0000.bin",   nf)
    else:
        T_max  = loaders.load_per_face_field(run, "statistics/temperature_max.bin", nf)
        T_mean = loaders.load_per_face_field(run, "statistics/temperature_avg.bin", nf)
        T_min  = loaders.load_per_face_field(run, "statistics/temperature_min.bin", nf)
    print(f"[predict_phi_eq]   T_max  range [{T_max.min():.1f}, {T_max.max():.1f}] K")
    print(f"[predict_phi_eq]   T_mean range [{T_mean.min():.1f}, {T_mean.max():.1f}] K")

    # Compute dose rate and porosity per face (from spatial parameterization)
    dose = cr.dose_rate(lat, lon)
    porosity = cr.porosity_from_latitude(lat)

    # Equilibrium phi at T_max (relevant for steady-state high-temperature limit)
    phi_eq_tmax = cr.phi_eq(T_max, lat, lon)
    phi_eq_tmean = cr.phi_eq(T_mean, lat, lon)

    print("[predict_phi_eq] Computed phi_eq from T_max:")
    res_eq, n_eq = _population(phi_eq_tmax, "phi_eq(T_max)")
    crystal_frac = _frac(phi_eq_tmax > 0.5, res_eq, n_eq)
    amorphous_frac = _frac(phi_eq_tmax < 0.1, res_eq, n_eq)
    print(f"  faces with phi_eq(T_max) > 0.5: {crystal_frac:.1%}")
    print(f"  faces with phi_eq(T_max) < 0.1: {amorphous_frac:.1%}")
    print(f"  median phi_eq(T_max): {_stat(np.median, phi_eq_tmax, res_eq, n_eq):.3f}")
    print(f"  population: {n_eq} of {phi_eq_tmax.size} faces resolved")

    out = dict(
        lat=lat, lon=lon, T_max=T_max, T_mean=T_mean, T_min=T_min,
        dose_rate=dose, porosity=porosity,
        phi_eq_tmax=phi_eq_tmax,
        phi_eq_tmean=phi_eq_tmean,
    )

    if args.integrate:
        print(f"[predict_phi_eq] Loading per-step temperature time series ({nf} faces)")
        t0 = time.time()
        times_s, Tt = loaders.load_temperature_timeseries(run, nf)
        print(f"[predict_phi_eq]   loaded ({Tt.shape[0]} steps) in {time.time()-t0:.1f}s")
        print(f"[predict_phi_eq] Integrating Eq. D33 for {args.n_orbits} orbits")
        t0 = time.time()
        phi_C = cr.integrate_diurnal(Tt.astype(np.float64), times_s, lat, lon,
                                     n_orbits=args.n_orbits, phi0=0.0)
        print(f"[predict_phi_eq]   integrated in {time.time()-t0:.1f}s")
        res_int, n_int = _population(phi_C, "integrated phi_C")
        crystal_int = _frac(phi_C > 0.5, res_int, n_int)
        amorph_int = _frac(phi_C < 0.1, res_int, n_int)
        print(f"[predict_phi_eq] Integrated phi_C statistics:")
        print(f"  faces with phi_C > 0.5: {crystal_int:.1%}")
        print(f"  faces with phi_C < 0.1: {amorph_int:.1%}")
        print(f"  median phi_C: {_stat(np.median, phi_C, res_int, n_int):.3f}")
        print(f"  population: {n_int} of {phi_C.size} faces resolved")
        out["phi_C_integrated"] = phi_C
        out["n_orbits_integrated"] = args.n_orbits

    if cpp_mode:
        ep = args.cpp_epoch if args.cpp_epoch >= 0 else _find_last_cpp_epoch(run)
        print(f"[predict_phi_eq] Loading C++ phi_C from epoch {ep}")
        rel = f"crystalline_fraction/crystalline_fraction_{ep:04d}.bin"
        phi_C = loaders.load_per_face_field(run, rel, nf)
        res_cpp, n_cpp = _population(phi_C, "C++ crystalline_fraction")
        crystal_cpp = _frac(phi_C > 0.5, res_cpp, n_cpp)
        amorph_cpp  = _frac(phi_C < 0.1, res_cpp, n_cpp)
        print(f"[predict_phi_eq] C++ phi_C statistics:")
        print(f"  faces with phi_C > 0.5: {crystal_cpp:.1%}")
        print(f"  faces with phi_C < 0.1: {amorph_cpp:.1%}")
        print(f"  mean phi_C:   {_stat(np.mean, phi_C, res_cpp, n_cpp):.4f}")
        print(f"  median phi_C: {_stat(np.median, phi_C, res_cpp, n_cpp):.4f}")
        print(f"  population: {n_cpp} of {phi_C.size} faces resolved")
        out["phi_C_integrated"] = phi_C
        out["cpp_epoch"] = ep
        out["source"] = "cpp_runtime"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)
    print(f"[predict_phi_eq] Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
