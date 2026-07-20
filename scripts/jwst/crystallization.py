"""
Python port of the C++ ice crystallization model.

Reproduces docs/appendix-d-exosphere-model-full.md §D.7 (equations D26-D38)
and the C++ implementation at src/cpp/exosphere/ice_crystallization_model.h.

Constants are taken verbatim from ice_crystallization_model.h so the Python
forward model and the C++ run-time evolution share the same physics.
"""

from __future__ import annotations

import numpy as np

# Mitchell et al. (2017) Avrami kinetics
R_GAS = 8.314                 # J/(mol K)
DELTA_HA = 62000.0            # J/mol  (62 kJ/mol)
A0_INV_N = 3.9e19             # s^-1   (pre-exponential; "A_0^(1/n)")
N_LOW = 4.0                   # compact ice
N_HIGH = 1.5                  # porous ice
POROSITY_MAX = 0.7

# Famá et al. (2010) amorphization
E_AMORPH = 3.0                # eV/molecule

# Nordheim et al. (2022) dose-rate spatial pattern, calibrated to Cartwright (2025)
D0 = 2.3e-6                   # eV/(molecule s) at trailing equator (~15-day timescale)
C_D = 5.0                     # trailing/leading dose contrast
E_D = 2.5                     # polar dose enhancement

# Trailing-apex east-longitude. Cassidy et al. (2013) and Nordheim et al. (2022)
# place Europa's trailing apex at 270°W = 90°E in IAU east-positive longitude.
# This is also where the run's bond_albedo map shows the lowest albedo (sputter
# darkening). Note: src/cpp/exosphere/ice_crystallization_model.h uses 270°E,
# which is 180° offset from the actual trailing apex on Europa — flagged as a
# bug to fix in the C++ model. The Python forward model uses the correct
# convention to match the run's mesh.ply lat/lon and the albedo map data.
TRAILING_APEX_LON_E = 90.0

# Porosity (Cartwright 2025 / Thelen 2024 ALMA)
POR_EQ = 0.50
POR_POL = 0.20

# Numerical floor: below this, k_cryst is treated as identically zero
T_FLOOR = 80.0


def avrami_exponent(porosity: np.ndarray | float) -> np.ndarray | float:
    """n(porosity) per Eq. D28 (decreases with porosity)."""
    t = np.clip(np.asarray(porosity) / POROSITY_MAX, 0.0, 1.0)
    return N_LOW + (N_HIGH - N_LOW) * t


def porosity_from_latitude(latitude_deg: np.ndarray) -> np.ndarray:
    """Linear interp from equatorial (POR_EQ) to polar (POR_POL), clamped [0.05, 0.7]."""
    abs_lat = np.abs(np.asarray(latitude_deg))
    t = abs_lat / 90.0
    phi = POR_EQ + (POR_POL - POR_EQ) * t
    return np.clip(phi, 0.05, 0.70)


def dose_rate(latitude_deg: np.ndarray, longitude_deg: np.ndarray) -> np.ndarray:
    """Per-face Ḋ [eV/(molec s)] from Eqs. D30-D32 (Nordheim 2022 parameterization).

    f_lon spans [1/C_D, 1] with the maximum at the trailing apex
    (90°E in IAU east-positive convention; see TRAILING_APEX_LON_E).
    f_lat: super-linear (exponent 1.5) polar enhancement up to E_D.
    """
    lat = np.asarray(latitude_deg, dtype=np.float64)
    lon = np.asarray(longitude_deg, dtype=np.float64)

    cos_trailing = np.cos(np.deg2rad(lon - TRAILING_APEX_LON_E))
    min_factor = 1.0 / C_D
    f_lon = min_factor + (1.0 - min_factor) * np.maximum(0.0, (cos_trailing + 1.0) / 2.0)

    f_lat = 1.0 + (E_D - 1.0) * (np.abs(lat) / 90.0) ** 1.5

    return D0 * f_lon * f_lat


def k_cryst(temperature_K: np.ndarray, porosity: np.ndarray | float) -> np.ndarray:
    """Effective first-order crystallization rate [s^-1] (Eqs. D26-D27).

    k_eff = n * A_0^(1/n) * exp(-ΔH_a / (R*T))

    Returns 0 below T_FLOOR (numerical safeguard; rate is < 1e-21 s^-1 anyway).
    """
    T = np.asarray(temperature_K, dtype=np.float64)
    n = avrami_exponent(porosity)
    n = np.broadcast_to(np.asarray(n, dtype=np.float64), T.shape)

    rate = np.zeros_like(T)
    mask = T >= T_FLOOR
    if np.any(mask):
        rate[mask] = n[mask] * A0_INV_N * np.exp(-DELTA_HA / (R_GAS * T[mask]))
    return rate


def k_amorph(dose_rate_eV_per_molec_s: np.ndarray | float) -> np.ndarray:
    """Amorphization rate [s^-1] (Eq. D29). E_AMORPH > 0 by construction."""
    return np.asarray(dose_rate_eV_per_molec_s, dtype=np.float64) / E_AMORPH


def phi_eq(temperature_K: np.ndarray,
           latitude_deg: np.ndarray,
           longitude_deg: np.ndarray) -> np.ndarray:
    """Equilibrium crystalline fraction per Eq. D35.

    φ_eq = k_cryst / (k_cryst + k_amorph)
    Returns 0 where k_cryst+k_amorph is below floor.
    """
    porosity = porosity_from_latitude(latitude_deg)
    kc = k_cryst(temperature_K, porosity)
    ka = k_amorph(dose_rate(latitude_deg, longitude_deg))
    total = kc + ka
    out = np.zeros_like(np.broadcast_to(np.asarray(temperature_K, dtype=np.float64), kc.shape))
    mask = total > 1e-30
    out[mask] = kc[mask] / total[mask]
    return out


def evolve_one_step(phi_C: np.ndarray,
                    dt: float,
                    temperature_K: np.ndarray,
                    porosity: np.ndarray,
                    k_amorph_face: np.ndarray) -> np.ndarray:
    """Closed-form single-step update of φ_C (Eq. D34) for a flat array of faces.

    φ_C(t+dt) = φ_eq + (φ_C(t) - φ_eq) exp(-(k_cryst + k_amorph) dt)
    """
    kc = k_cryst(temperature_K, porosity)
    total = kc + k_amorph_face
    safe = total > 1e-30
    phi_eq_local = np.where(safe, kc / np.where(safe, total, 1.0), 0.0)
    decay = np.where(safe, np.exp(-total * dt), 1.0)
    out = phi_eq_local + (phi_C - phi_eq_local) * decay
    return np.clip(out, 0.0, 1.0)


def integrate_diurnal(temperatures_t: np.ndarray,
                      times_s: np.ndarray,
                      latitude_deg: np.ndarray,
                      longitude_deg: np.ndarray,
                      n_orbits: int = 200,
                      phi0: float = 0.0) -> np.ndarray:
    """Time-integrate Eq. D33 over n_orbits repeats of the diurnal cycle.

    Parameters
    ----------
    temperatures_t : (Nt, Nfaces) array of T(t) per face.
    times_s        : (Nt,) timestamps in seconds. Must be monotonic.
    latitude_deg, longitude_deg : (Nfaces,) arrays.
    n_orbits       : number of times to replay the diurnal cycle. Default 200
                     is well past steady state for compact-ice T_max>120 K.
    phi0           : initial crystalline fraction (0 = fully amorphous).

    Returns
    -------
    phi_C_final : (Nfaces,) crystalline fraction after n_orbits diurnal cycles.

    The analytical per-step update means this is dirt-cheap even for ico-9 meshes.
    """
    Nt, Nfaces = temperatures_t.shape
    assert times_s.shape == (Nt,)

    # dt array (loop closure: assume diurnal is periodic; close gap with first step)
    dt = np.diff(times_s)
    period = times_s[-1] - times_s[0]
    closing_dt = max(period / Nt, dt.mean()) if Nt > 1 else 0.0
    dt = np.concatenate([dt, [closing_dt]])  # length Nt

    porosity = porosity_from_latitude(latitude_deg)
    ka = k_amorph(dose_rate(latitude_deg, longitude_deg))

    phi = np.full(Nfaces, phi0, dtype=np.float64)
    for _ in range(n_orbits):
        for t_idx in range(Nt):
            phi = evolve_one_step(phi, dt[t_idx], temperatures_t[t_idx],
                                  porosity, ka)
    return phi


def transition_temperature(dose_rate_val: float,
                           porosity_val: float = 0.0) -> float:
    """T* per Eq. D37 where φ_eq = 0.5.

    T* = ΔH_a / (R * ln(n * A_0^(1/n) * E_amorph / Ḋ))
    """
    n = avrami_exponent(porosity_val)
    arg = n * A0_INV_N * E_AMORPH / dose_rate_val
    return DELTA_HA / (R_GAS * np.log(arg))


__all__ = [
    "R_GAS", "DELTA_HA", "A0_INV_N", "N_LOW", "N_HIGH",
    "E_AMORPH", "D0", "C_D", "E_D", "POR_EQ", "POR_POL", "T_FLOOR",
    "avrami_exponent", "porosity_from_latitude", "dose_rate",
    "k_cryst", "k_amorph", "phi_eq",
    "evolve_one_step", "integrate_diurnal",
    "transition_temperature",
]
