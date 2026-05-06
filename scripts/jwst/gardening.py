"""
Python forward model for impact gardening on Europa.

Reproduces Costello, Phillips, Lucey & Ghent (2021), "Impact gardening on
Europa and repercussions for possible biosignatures," Nature Astronomy 5,
951-956. doi:10.1038/s41550-021-01393-1.

The analytical model (Eq. 1 of the paper):

    Λ(t) = h * (4(v+2) λ / (u v π t c^2))^(1/(v+2))

where Λ is the gardening depth, t is the time interval, h is the
dimensionless excavation-depth fraction of crater diameter, (u, v) are the
crater-production-function parameters (cumulative N = u D^v in km^-2 yr^-1
for D in km), λ is the cumulative-Poisson average events per interval (sets
the probability of an arbitrary point being in an excavated volume), and c
is a dimensionless overlap parameter.

For secondary craters with v = -4, this reduces to a square-root law:

    Λ(t) = h * sqrt(u π t c^2 / (2 λ))

inverted to a closed-form timescale:

    τ(Λ; λ) = (2 λ) / (u π c^2) * (Λ / h)^2

The headline result of the paper is that gardening on Europa, with
secondary craters included, reaches ~30 cm at 50 % probability over 10-200
Myr — vastly faster than earlier estimates that omitted secondaries.

Constants come from the paper's "Modelling secondary impacts" /
"The impact rate at Europa" subsections. Both cratering-rate scenarios are
provided:

  - Case A: Zahnle et al. (2003) "Case A" — conservative low impact rate
  - Case B: Zahnle et al. (2003) "Case B" + Levison et al. (2000) — the
            scenario adopted by Tsuji & Teanby (2016) and consistent with
            recent Jupiter impact-flash observations.
"""

from __future__ import annotations

import numpy as np

# Crater-production-function parameters (cumulative N = u D^v) for the
# secondary-crater flux at Europa. Units: u in [km^-2 yr^-1], D in [km].
# Source: Costello et al. (2021), "The impact rate at Europa" subsection.
CASE_A_SEC = {"u": 2.02e-12, "v": -4}
CASE_B_SEC = {"u": 7.03e-12, "v": -4}

CASES = {"A": CASE_A_SEC, "B": CASE_B_SEC}

# Cumulative-Poisson rate parameters — set the gardening probability bin.
# λ = -ln(1-p), where p is the probability that an arbitrary point at
# depth Λ has been in the excavated volume of at least one crater.
LAMBDA_EQUIL = 0.02       # 2 %  — equilibrium / degradation-zone boundary
LAMBDA_INTERMED = 0.693   # 50 % — intermediate / "in-situ reworking zone"
LAMBDA_SAT = 4.605        # 99 % — saturation / homogenized-reworking-zone boundary

# Geometric defaults. h = excavation-depth fraction of crater diameter
# (Costello et al. 2018 calibrated h_0 ≈ 0.04 against Apollo cores;
# Costello et al. 2021 uses the same value). c = overlap-scaling parameter
# (1.0 for the geometric-saturation Poisson formulation).
H_DEFAULT = 0.04
C_DEFAULT = 1.0

# Reference depth scales used by the talk's timescales plot.
Z_OPTICAL_M = 10e-6     # 10 µm — 3.1 µm Fresnel-peak probe depth
Z_RADIATION_M = 1e-2    # 1 cm  — order of the Nordheim radiation-skin depth
Z_RESERVOIR_M = 0.30    # 30 cm — Costello "homogenized reworking zone" depth


def _resolve_case(case):
    if isinstance(case, str):
        try:
            return CASES[case.upper()]
        except KeyError as exc:
            raise ValueError(f"Unknown gardening case '{case}'. "
                             f"Valid: {sorted(CASES)}") from exc
    if isinstance(case, dict) and "u" in case and "v" in case:
        return case
    raise TypeError("case must be 'A', 'B', or a dict with 'u' and 'v' keys")


def gardening_depth(t_yr,
                    lam=LAMBDA_INTERMED,
                    case="B",
                    h=H_DEFAULT,
                    c=C_DEFAULT):
    """Costello et al. (2021) Eq. 1: gardening depth Λ at time t [years].

    Parameters
    ----------
    t_yr : float or array
        Time interval [years].
    lam : float, optional
        Cumulative-Poisson rate parameter (probability bin).
        0.02 = equilibrium (2 %), 0.693 = intermediate (50 %),
        4.605 = saturation (99 %). Default 0.693.
    case : {"A", "B"} or dict, optional
        Cratering-rate scenario. "A" = Zahnle Case A (conservative),
        "B" = Zahnle Case B + Levison (preferred, default).
    h : float, optional
        Excavation-depth fraction of crater diameter. Default 0.04.
    c : float, optional
        Overlap-scaling parameter. Default 1.0.

    Returns
    -------
    Λ : float or array
        Gardening depth [meters].
    """
    p = _resolve_case(case)
    u, v = p["u"], p["v"]
    t = np.asarray(t_yr, dtype=float)

    # Eq. 1 in km (u is in km^-2 yr^-1 with t in yr); convert to m on output.
    inner = 4.0 * (v + 2) * lam / (u * v * np.pi * t * c * c)
    lam_km = h * np.power(inner, 1.0 / (v + 2))
    return 1000.0 * lam_km


def gardening_timescale(depth_m,
                        lam=LAMBDA_INTERMED,
                        case="B",
                        h=H_DEFAULT,
                        c=C_DEFAULT):
    """Inverse of `gardening_depth`: time t [years] for the gardening
    depth to reach Λ at probability bin λ.

    For v = -4 (secondary craters) this reduces to the closed form

        τ = (2 λ) / (u π c^2) * (Λ / h)^2.

    For other v values, it is computed by solving Eq. 1 for t.

    Parameters
    ----------
    depth_m : float or array
        Target gardening depth [meters].
    lam, case, h, c
        See `gardening_depth`.

    Returns
    -------
    t : float or array
        Time [years].
    """
    p = _resolve_case(case)
    u, v = p["u"], p["v"]
    z_km = np.asarray(depth_m, dtype=float) / 1000.0

    # Λ = h * (4(v+2) λ / (u v π t c²))^{1/(v+2)}
    #  ⇒ (Λ/h)^{v+2} = 4(v+2)λ / (u v π t c²)
    #  ⇒ t = 4(v+2)λ / (u v π c² (Λ/h)^{v+2})
    ratio = z_km / h
    return (4.0 * (v + 2) * lam) / (u * v * np.pi * c * c * np.power(ratio, v + 2))


def gardening_rate(depth_m,
                   lam=LAMBDA_INTERMED,
                   case="B",
                   h=H_DEFAULT,
                   c=C_DEFAULT):
    """Effective overturn rate k_gard ≡ 1/τ(Λ) at the given reference depth.

    Useful for the C++ Phase 7b coupling: a per-face relaxation rate that
    pulls the surface bin's φ_C and non-ice concentration toward the
    column-mean reservoir values.
    """
    return 1.0 / gardening_timescale(depth_m, lam=lam, case=case, h=h, c=c)


__all__ = [
    "CASE_A_SEC", "CASE_B_SEC", "CASES",
    "LAMBDA_EQUIL", "LAMBDA_INTERMED", "LAMBDA_SAT",
    "H_DEFAULT", "C_DEFAULT",
    "Z_OPTICAL_M", "Z_RADIATION_M", "Z_RESERVOIR_M",
    "gardening_depth", "gardening_timescale", "gardening_rate",
]
