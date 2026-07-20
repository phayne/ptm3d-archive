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
# Ingersoll, Svitek & Murray (1992) Eq 8 — the SHADOWED-interior closed form.
#
# For a spherical-cap bowl whose interior is fully in shadow (β < β_crit), the
# interior sees no direct Sun; it is heated only by (a) thermal IR from the
# sunlit walls and (b) sunlight the walls scatter down into it. Ingersoll's
# Eq 8 gives the equilibrium emission of that shadowed interior:
#
#     σ T⁴ = S sin(β) · f(1−A)/(1−A f) · [ 1 + A(1−f)/ε ],
#
# with f = 1 − F_sky the interior-to-wall view factor (geometry only). This is
# the reference the production two-band radiosity is validated against on the
# bowl harness (bowl_crater_ptm.cpp). It is DISTINCT from the two-zone
# ``floor_temperature`` above (which assumes an isothermal 200 K wall and adds a
# direct-Sun term for β > β_crit); Eq 8 self-consistently closes the sunlit-wall
# scattering and so is the correct shadow-limit curve.

def shadow_sigmaT4_eq8(
    beta_rad: float | np.ndarray,
    depth_to_diameter: float,
    *,
    albedo: float = 0.12,
    emissivity: float = 0.95,
    solar_constant: float = SOLAR_CONSTANT,
) -> np.ndarray:
    """σ T⁴ (W m⁻²) of the shadowed spherical-cap interior — Ingersoll Eq 8.

    β is the solar elevation above the local horizon; valid where the interior
    is fully shadowed (β < β_crit).
    """
    beta = np.atleast_1d(np.asarray(beta_rad, dtype=float))
    f = 1.0 - sky_view_factor(depth_to_diameter)
    A = albedo
    sigT4 = (solar_constant * np.sin(beta)
             * f * (1.0 - A) / (1.0 - A * f)
             * (1.0 + A * (1.0 - f) / emissivity))
    return sigT4


def shadow_temperature_eq8(
    beta_rad: float | np.ndarray,
    depth_to_diameter: float,
    **kw,
) -> np.ndarray:
    """Equilibrium temperature (K) of the shadowed interior — Ingersoll Eq 8."""
    return (shadow_sigmaT4_eq8(beta_rad, depth_to_diameter, **kw) / SIGMA) ** 0.25


def shadow_flux_eq8(
    beta_rad: float | np.ndarray,
    depth_to_diameter: float,
    *,
    emissivity: float = 0.95,
    **kw,
) -> np.ndarray:
    """Emitted flux ε σ T⁴ (W m⁻²) of the shadowed interior — Ingersoll Eq 8."""
    return emissivity * shadow_sigmaT4_eq8(
        beta_rad, depth_to_diameter, emissivity=emissivity, **kw)


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
