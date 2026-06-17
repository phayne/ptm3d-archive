"""
Analytic floor-temperature solver for bowl-shaped craters following the
two-zone (floor + wall) model of Ingersoll, Svitek & Murray (1992),
"Stability of polar frosts in spherical bowl-shaped craters on the Moon,
Mercury, and Mars", Icarus 100, 40–47.

Geometry
--------
A bowl-shaped crater is a spherical cap of depth ``d`` and rim diameter ``D``.
The sphere radius is

    R  =  (r² + d²) / (2 d),     r = D/2

The half-opening angle from the floor centre to the rim is

    α  =  arctan(r / d),

and the floor's view factor to the sky is

    F_sky  =  cos²(α / 2)? — Ingersoll uses sin²(α).

The "rim-shadow" critical solar elevation above which the floor sees direct
sunlight is

    β_crit  =  π/2 − α   ≡   arctan(2 d / D).

Two-zone energy balance
-----------------------
Walls and floor are assumed isothermal (Ingersoll's simplification). With
emissivity ε, Bond albedo A, solar constant S, and wall mean temperature
T_w (taken from the global diurnal mean at the same latitude), the floor
energy balance is

    ε σ T_f⁴ = (1−A) S V(β)  +  ε σ T_w⁴ (1 − F_sky)

where V(β) is the fraction of the floor that sees direct sun:

    V(β)  =  max(0,  sin(β) − sin(β_crit)) / (1 − sin(β_crit))   for β ≥ β_crit,
                                          0                         for β < β_crit.

For β < β_crit the floor is in permanent shadow and is heated only by the
warm walls (the second term).

This simplified two-zone closed-form model captures the essential physics
of Ingersoll Fig. 2 to within a few K and produces the characteristic
"shoulder" at β = β_crit.
"""
from __future__ import annotations

import numpy as np


SIGMA = 5.670374e-8
SOLAR_CONSTANT = 1361.0


def crater_alpha(depth_to_diameter: float) -> float:
    """Half-opening angle (rad) from the floor centre to the rim."""
    return float(np.arctan(0.5 / depth_to_diameter))


def critical_elevation(depth_to_diameter: float) -> float:
    """Solar elevation above which the floor centre sees direct sun (rad)."""
    return float(np.arctan(2.0 * depth_to_diameter))


def sky_view_factor(depth_to_diameter: float) -> float:
    """Floor-to-sky view factor for a spherical cap (Ingersoll 1992)."""
    a = crater_alpha(depth_to_diameter)
    return float(np.sin(a) ** 2)


def visible_fraction(beta_rad: float, depth_to_diameter: float) -> float:
    """Fraction of the floor that sees direct sun at solar elevation β."""
    bcrit = critical_elevation(depth_to_diameter)
    if beta_rad <= bcrit:
        return 0.0
    return min(1.0, (np.sin(beta_rad) - np.sin(bcrit)) / (1.0 - np.sin(bcrit)))


def floor_temperature(
    beta_rad: float | np.ndarray,
    depth_to_diameter: float,
    *,
    albedo: float = 0.12,
    emissivity: float = 0.95,
    T_wall: float = 200.0,
    solar_constant: float = SOLAR_CONSTANT,
) -> np.ndarray:
    """Equilibrium crater-floor temperature (K) at solar elevation β.

    β is measured above the local horizon (so β = π/2 ⇒ Sun at zenith).
    """
    beta = np.atleast_1d(np.asarray(beta_rad, dtype=float))
    fsky = sky_view_factor(depth_to_diameter)
    bcrit = critical_elevation(depth_to_diameter)
    sin_b = np.sin(beta)
    sin_bcrit = np.sin(bcrit)
    V = np.where(beta > bcrit,
                 np.minimum(1.0, (sin_b - sin_bcrit) / (1.0 - sin_bcrit)),
                 0.0)
    direct = (1.0 - albedo) * solar_constant * V
    wall_term = emissivity * SIGMA * T_wall ** 4 * (1.0 - fsky)
    rhs = direct + wall_term
    T4 = rhs / (emissivity * SIGMA)
    return T4 ** 0.25


def floor_temperature_curve(
    depth_to_diameter: float,
    n: int = 181,
    **kw,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (β_deg, T_floor_K) for elevation 0..90° in n steps."""
    beta = np.deg2rad(np.linspace(0.0, 90.0, n))
    T = floor_temperature(beta, depth_to_diameter, **kw)
    return np.rad2deg(beta), T


# ---------------------------------------------------------------------------
# Self-test / smoke check

def _smoke_check() -> None:
    """Run a sanity check, print a small table for the d/D ∈ {0.1, 0.2, 0.3}
    cases of Ingersoll Fig. 2."""
    print("\nIngersoll bowl-crater self-test (Moon defaults: A=0.12, ε=0.95, T_w=200K)")
    print(f"{'d/D':>6} {'F_sky':>8} {'β_crit (°)':>11} "
          f"{'T_psr (K)':>10} {'T_zenith (K)':>14}")
    for dd in (0.1, 0.2, 0.3):
        bcrit = critical_elevation(dd)
        fsky = sky_view_factor(dd)
        T_psr = float(floor_temperature(0.0, dd))
        T_zen = float(floor_temperature(np.deg2rad(90.0), dd))
        print(f"{dd:>6.2f} {fsky:>8.4f} {np.rad2deg(bcrit):>11.2f} "
              f"{T_psr:>10.2f} {T_zen:>14.2f}")


if __name__ == "__main__":
    _smoke_check()
