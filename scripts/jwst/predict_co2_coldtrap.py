#!/usr/bin/env python3
"""
H3 forward model: pure-thermal CO₂ cold-trap stability prediction.

CO₂ on Europa sublimates rapidly above ~80–90 K. A face is a thermal
cold trap if its peak diurnal temperature stays below the residence
threshold corresponding to the relevant timescale.

We use the standard Clausius–Clapeyron form for solid CO₂:
    P_sat(T) = exp(A - B / T)   [Pa]
with Brown & Ziegler (1980) coefficients fit to T = 80–125 K range:
    A = 27.7, B = 3115 K
giving sublimation pressures spanning ~1e-12 to ~1e-2 Pa over 80–125 K.

The Hertz-Knudsen flux at vacuum is
    Φ_sub = P_sat(T) · sqrt(M_CO2 / (2π R T))   [kg/(m² s)]

We define a stability temperature T_stab(τ) at which a single
ice-monolayer (~5e-6 kg/m²) sublimates over τ. Comparing PTM3D T_max
to T_stab(τ_obs) flags faces that should retain CO₂ over the observation
timescale (~1–10 yr for Cartwright observations).

Headline test: Cartwright (2025) reports CO₂ band depth peaking at
warm leading-hemisphere chaos terrain (Tara/Powys), NOT at cold poles.
If thermal cold-trapping is the only redistribution mechanism, the CO₂
should be poleward-concentrated. A spatial mismatch implies a non-thermal
(endogenic / cryovolcanic / chaos-overturn) source.

Usage:
  python predict_co2_coldtrap.py --phi-npz data/jwst/cartwright2025/phi_eq_albedo_map.npz \
                                  --out    data/jwst/cartwright2025/co2_coldtrap.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Brown & Ziegler (1980) CO2 vapor pressure (T = 80-125 K range)
# log10(P_torr) = A_BZ - B_BZ / T  →  here in Pa with constants converted
CO2_VP_A = 27.7      # ln(Pa) intercept
CO2_VP_B = 3115.0    # K
M_CO2 = 0.04401      # kg/mol
R_GAS = 8.314        # J/(mol K)
N_A = 6.02214e23
MONOLAYER_KG_PER_M2 = 5.0e-6  # ~1 ML of CO2 ice (rough; mass ≈ 1e-7 kg/m^2 per ML * factor)
TARGET_STABILITY_TIMESCALES_YR = (1.0, 10.0, 100.0, 1.0e6)


def co2_vapor_pressure_pa(T_K: np.ndarray) -> np.ndarray:
    T = np.asarray(T_K, dtype=np.float64)
    return np.exp(CO2_VP_A - CO2_VP_B / np.maximum(T, 1.0))


def co2_sublimation_flux_kg_m2_s(T_K: np.ndarray) -> np.ndarray:
    T = np.asarray(T_K, dtype=np.float64)
    P = co2_vapor_pressure_pa(T)
    # Hertz–Knudsen: Φ = P / sqrt(2π M R T / N_A^2 ?) — using molecular form:
    # Φ = P * sqrt(m / (2π k_B T)) where m = M/N_A
    # Equivalent SI mass form: Φ = P * sqrt(M / (2π R T))
    return P * np.sqrt(M_CO2 / (2.0 * np.pi * R_GAS * np.maximum(T, 1.0)))


def stability_temperature_for_timescale(tau_s: float) -> float:
    """Bisect for T such that one monolayer sublimates over tau seconds:
       Φ_sub(T) = MONOLAYER / tau"""
    target = MONOLAYER_KG_PER_M2 / tau_s
    lo, hi = 30.0, 300.0
    for _ in range(120):
        mid = 0.5 * (lo + hi)
        if co2_sublimation_flux_kg_m2_s(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phi-npz", required=True, type=Path)
    p.add_argument("--out",     required=True, type=Path)
    args = p.parse_args()

    npz = np.load(args.phi_npz)
    lat, lon = npz["lat"], npz["lon"]
    T_max = npz["T_max"]

    # Stability temperature thresholds for several timescales
    thresholds_K = {}
    for tau_yr in TARGET_STABILITY_TIMESCALES_YR:
        T_stab = stability_temperature_for_timescale(tau_yr * 365.25 * 86400.0)
        thresholds_K[f"T_stab_{tau_yr:g}yr"] = T_stab
        n_stable = int(np.sum(T_max < T_stab))
        print(f"[predict_co2_coldtrap]  τ = {tau_yr:6g} yr ⇒ T_stab = {T_stab:5.1f} K  "
              f"({n_stable:6d} faces / {T_max.size}  = {n_stable/T_max.size:.1%} stable)")

    # Per-face cumulative sublimation depth over 1 yr (proxy for "how much CO2 lost")
    flux_at_Tmax = co2_sublimation_flux_kg_m2_s(T_max)
    loss_kg_per_yr = flux_at_Tmax * 365.25 * 86400.0  # kg/m² lost per year of T_max exposure
    log10_loss_yr = np.log10(np.maximum(loss_kg_per_yr, 1e-30))

    # Stability mask: T_max below 10-yr threshold
    T_stab_10 = stability_temperature_for_timescale(10.0 * 365.25 * 86400.0)
    co2_stable_10yr = (T_max < T_stab_10)

    print(f"[predict_co2_coldtrap]  Stable on {co2_stable_10yr.sum()} of {T_max.size} faces "
          f"({co2_stable_10yr.mean():.1%}) at the 10-yr observation timescale")

    out = dict(lat=lat, lon=lon, T_max=T_max,
               co2_flux_at_Tmax_kg_m2_s=flux_at_Tmax,
               co2_log10_loss_kg_m2_per_yr=log10_loss_yr,
               co2_stable_10yr=co2_stable_10yr.astype(np.uint8),
               **thresholds_K)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)
    print(f"[predict_co2_coldtrap] Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
