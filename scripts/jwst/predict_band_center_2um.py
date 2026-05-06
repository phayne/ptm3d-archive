#!/usr/bin/env python3
"""
Predict the 2.0 μm H2O ice band center from PTM3D surface temperature
(H2: independent thermometer cross-check on H1).

Cartwright et al. (2025) report two anchor points (Tara/Powys ~150 K /
1.986 μm; cold N low-lat zone ~130 K / 1.992 μm). Linear fit:
    λ_2.0(T) = 1.992 - slope * (T - 130)    [μm]
    slope ≈ (1.992 - 1.986) / (130 - 150) = -3.0e-4  μm/K (note sign)

Equivalently: λ shifts to longer wavelength as T decreases.
The reference temperature for prediction is T_max (the daily peak), under
the same assumption made for H1: spectroscopic measurements at the JWST
observation epoch are dominated by the warmest local time on each face.

Usage:
  python predict_band_center_2um.py --phi-npz data/jwst/cartwright2025/phi_eq_albedo_map.npz \
                                    --out    data/jwst/cartwright2025/band_center_2um.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Cartwright (2025) anchor points
T_ANCHOR_WARM = 150.0  # K (Tara/Powys)
LAMBDA_WARM = 1.986  # μm
T_ANCHOR_COLD = 130.0  # K (northern low-latitude zone)
LAMBDA_COLD = 1.992  # μm
SLOPE_UM_PER_K = (LAMBDA_WARM - LAMBDA_COLD) / (T_ANCHOR_WARM - T_ANCHOR_COLD)


def lambda_2um(T_K: np.ndarray) -> np.ndarray:
    """λ_2.0(T) [μm] from linear Cartwright-anchored fit."""
    return LAMBDA_COLD + SLOPE_UM_PER_K * (np.asarray(T_K, dtype=np.float64) - T_ANCHOR_COLD)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phi-npz", required=True, type=Path,
                   help="NPZ from predict_phi_eq.py (provides lat, lon, T_max)")
    p.add_argument("--out", required=True, type=Path,
                   help="Output NPZ with predicted band center map")
    p.add_argument("--use-tmean", action="store_true",
                   help="Predict from T_mean instead of T_max")
    args = p.parse_args()

    npz = np.load(args.phi_npz)
    lat, lon = npz["lat"], npz["lon"]
    T = npz["T_mean"] if args.use_tmean else npz["T_max"]

    lam = lambda_2um(T)
    print(f"[predict_band_center_2um] slope = {SLOPE_UM_PER_K*1e3:+.3f} nm/K")
    print(f"[predict_band_center_2um] {'T_mean' if args.use_tmean else 'T_max'} range "
          f"[{T.min():.1f}, {T.max():.1f}] K")
    print(f"[predict_band_center_2um] λ_2.0  range "
          f"[{lam.min():.4f}, {lam.max():.4f}] μm")

    out = dict(lat=lat, lon=lon, T_used=T, lambda_2um=lam,
               anchor_warm_T=T_ANCHOR_WARM, anchor_warm_lambda=LAMBDA_WARM,
               anchor_cold_T=T_ANCHOR_COLD, anchor_cold_lambda=LAMBDA_COLD,
               slope_um_per_K=SLOPE_UM_PER_K,
               source=("T_mean" if args.use_tmean else "T_max"))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)
    print(f"[predict_band_center_2um] Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
