"""Physical constants for the Ojakangas & Stevenson (1989) ice shell model.

All values in SI units unless noted. Equation numbers refer to Ojakangas &
Stevenson (1989a), "Thermal state of an ice shell on Europa", Icarus 81,
220-241 ("thermal paper"), and (1989b), "Polar wander of an ice shell on
Europa", Icarus 81, 242-270 ("OSPW").

Two parameter sets
------------------
``PAPER``  -- the values O&S used. This is the VALIDATION set: with it,
              ``python -m ojakangas_stevenson.validate`` reproduces Table I.
``MODERN`` -- current values for SCIENCE runs (project decision, 2026-10-04).
``MODERN_KLINGER`` -- MODERN with the Klinger (1980) conductivity.

The module-level names below (``ECCENTRICITY``, ``EUROPA_RADIUS``, ...) hold
the ACTIVE set. They start as ``PAPER``. Switch with
``use_parameter_set('modern')`` once, at program start, BEFORE building any
model; every function in this package reads the module-level names at call
time. ``parameter_set(name)`` is the same switch as a context manager (used
by the tests). ``active_parameter_set()`` returns the set in force so that
outputs can record it.
"""

import contextlib
from dataclasses import dataclass, asdict

import numpy as np


@dataclass(frozen=True)
class ParameterSet:
    """One self-consistent set of Europa and ice constants."""
    name: str
    eccentricity: float          # e
    radius: float                # a [m]
    density: float               # mean density rho_bar [kg/m^3]
    gravity: float               # surface gravity g [m/s^2]
    a1_conductivity: float       # k(T) = a1/T + a0, a1 [W/m]
    a0_conductivity: float       # a0 [W/(m K)]
    mass: float                  # M [kg]
    sources: tuple               # (quantity, citation, status) triples

    def as_dict(self):
        """Plain-dict form for writing into run metadata."""
        d = asdict(self)
        d['sources'] = [list(s) for s in self.sources]
        return d


G_GRAV = 6.674e-11          # Gravitational constant [m^3/(kg s^2)]

# --- PAPER set (validation) -------------------------------------------------
# a1 = 4.88e7 erg/(cm s) = 488 W/m and a0 = 4.68e4 erg/(cm K s) = 0.468
# W/(m K) are Eq. 2.3 (thermal paper p. 223, Hobbs 1974).
# Neither paper prints e, a, rho_bar or g. The values below are the ones this
# package has used since 2026-03 ("paper-inferred"); they are consistent with
# the two numbers the paper does print: mu' = mu/(rho_bar g a) ~ 0.6 for
# mu = 4e10 dyn/cm^2 (Eq. B.17 text; here 0.640), and the Fig. 1 unit
# (1/2)(n gamma e C_R)^2 ~ 1.2e-21 s^-2 (here 1.27e-21 with C_R = 0.5).
# The mass is derived from a and rho_bar for self-consistency.
_PAPER_RADIUS = 1.569e6
_PAPER_DENSITY = 3040.0
_INFERRED = 'paper-inferred: consistent with mu\' ~ 0.6 and the Fig. 1 unit'
PAPER = ParameterSet(
    name='paper',
    eccentricity=0.01,
    radius=_PAPER_RADIUS,
    density=_PAPER_DENSITY,
    gravity=1.31,
    a1_conductivity=488.0,
    a0_conductivity=0.468,
    mass=(4.0 / 3.0) * np.pi * _PAPER_RADIUS**3 * _PAPER_DENSITY,
    sources=(
        ('e', 'not printed in O&S 1989', _INFERRED),
        ('a', 'not printed in O&S 1989', _INFERRED),
        ('rho', 'not printed in O&S 1989', _INFERRED),
        ('g', 'not printed in O&S 1989', _INFERRED),
        ('k', 'O&S 1989 Eq. 2.3 (Hobbs 1974): a1 = 4.88e7 erg/(cm s)', 'paper'),
    ),
)

# --- MODERN set (science) ---------------------------------------------------
# Verified 2026-10-04 against the SPICE kernels shipped in data/kernels:
#   a   = 1560.8 km: mean of the pck00011.tpc triaxial radii
#         BODY502_RADII = (1562.6, 1560.3, 1559.5) km (IAU WGCCRE 2015,
#         Archinal et al. 2018); the volumetric mean is also 1560.80 km.
#   rho = 3013 kg/m^3 and g = 1.315 m/s^2: from BODY502_GM =
#         3202.7388 km^3/s^2 (gm_de431.tpc) with a = 1560.8 km and G above
#         (rho = 3013.0, g = 1.3147).
#   e   = 0.0094: the mean per-orbit geometric eccentricity
#         (r_max - r_min)/(r_max + r_min) of Europa about the Jupiter
#         barycenter in jup388s.bsp is 0.00936 (411 orbits from 2021-01-01;
#         range 0.0089-0.0098). This is the forced eccentricity commonly quoted
#         as 0.0094 (JPL SSD mean elements).
#   k   = 651/T W/(m K): Petrenko & Whitworth (1999), "Physics of Ice", OUP.
#         Option 567/T: Klinger (1980), J. Glaciol. 25, 179-186.
#         UNVERIFIED: neither coefficient was checked against its source in
#         this session; both are the values adopted on 2026-10-04.
_MODERN_GM = 3.202738774922892e12      # m^3/s^2 (gm_de431.tpc, BODY502_GM)
_MODERN_RADIUS = 1.5608e6
_MODERN_SOURCES = (
    ('e', 'jup388s.bsp mean geometric e = 0.00936 (JPL SSD mean elements: 0.0094)',
     'verified-kernel'),
    ('a', 'pck00011.tpc BODY502_RADII mean (Archinal et al. 2018)', 'verified-kernel'),
    ('rho', 'gm_de431.tpc BODY502_GM with a = 1560.8 km', 'verified-kernel'),
    ('g', 'gm_de431.tpc BODY502_GM / a^2', 'verified-kernel'),
)
MODERN = ParameterSet(
    name='modern',
    eccentricity=0.0094,
    radius=_MODERN_RADIUS,
    density=3013.0,
    gravity=1.315,
    a1_conductivity=651.0,
    a0_conductivity=0.0,
    mass=_MODERN_GM / G_GRAV,
    sources=_MODERN_SOURCES + (
        ('k', 'Petrenko & Whitworth 1999: k = 651/T', 'UNVERIFIED'),),
)
MODERN_KLINGER = ParameterSet(
    name='modern-klinger',
    eccentricity=MODERN.eccentricity,
    radius=MODERN.radius,
    density=MODERN.density,
    gravity=MODERN.gravity,
    a1_conductivity=567.0,
    a0_conductivity=0.0,
    mass=MODERN.mass,
    sources=_MODERN_SOURCES + (
        ('k', 'Klinger 1980: k = 567/T', 'UNVERIFIED'),),
)

PARAMETER_SETS = {s.name: s for s in (PAPER, MODERN, MODERN_KLINGER)}

# --- Constants common to both sets ------------------------------------------
T_MELT = 273.0               # Melting temperature of ice I [K]
SOLAR_FLUX_JUPITER = 50.0    # pi*F_sun at Jupiter orbit [W/m^2] (= S_0 in the paper)
BOND_ALBEDO = 0.5            # Bond albedo of Europa (thermal paper p. 237)
STEFAN_BOLTZMANN = 5.670374419e-8 # Stefan-Boltzmann [W/(m^2 K^4)], SI-2019 exact;
                                  # canonical: src/cpp/constants/stefan_boltzmann.h
OBLIQUITY = np.radians(3.0)  # Effective obliquity i ~ 3 deg [rad]

ORBITAL_PERIOD = 3.5512 * 86400.0  # Europa orbital period [s]
MEAN_MOTION = 2.0 * np.pi / ORBITAL_PERIOD  # n [rad/s] ~ 2.048e-5
JUPITER_MASS = 1.8982e27    # Mass of Jupiter [kg]

ICE_DENSITY = 920.0         # Density of ice [kg/m^3]
WATER_DENSITY = 1000.0      # Density of liquid water [kg/m^3]

# --- Maxwell rheology parameters (Section 3) ---
ICE_RIGIDITY = 4.0e9        # mu [Pa] (= 4e10 dyn/cm^2)
# Paper chooses nominal (n*tau_M)_min = 1, implying eta_0 = mu/n ~ 2e14 Pa s
# (= 2e15 poise). See p. 226: "we have chosen the nominal value of n*tau_M = 1"
MELT_VISCOSITY = ICE_RIGIDITY / (2.0 * np.pi / (3.5512 * 86400.0))  # eta_0 = mu/n ~ 1.95e14 Pa s
VISCOSITY_EXPONENT = 24     # l (power-law exponent in eta ~ T^{-l})

# --- Generalized flow law parameters (Section 4) ---
GLEN_EXPONENT_1 = 3         # n_G
ARRHENIUS_EXPONENT = 61     # n_A (same for both cases)
# A0 = 5.1e-27 s^-1 (dyn/cm^2)^{-3}; 1 dyn/cm^2 = 0.1 Pa
A0_FLOW_1_CGS = 5.1e-27     # [s^-1 (dyn/cm^2)^{-3}]
A0_FLOW_1_SI = 5.1e-27 * (10.0**3)  # [s^-1 Pa^{-3}] = 5.1e-24
GLEN_EXPONENT_2 = 5.2       # n_G
A0_FLOW_2_CGS = 8.0e-41     # [s^-1 (dyn/cm^2)^{-5.2}]
A0_FLOW_2_SI = 8.0e-41 * (10.0**5.2)  # [s^-1 Pa^{-5.2}]

# --- Tidal strain parameters (Appendix B) ---
# C_R: Ross & Schubert (1987) correction to the tidal strain amplitude.
# The paper states C_R = 0.5 (thermal paper p. 239). With the correct
# thickness law (Eq. 2.8, see shell_thickness.py) C_R = 0.5 reproduces ALL
# twelve Table I entries. The former value 0.85 was an accidental fit that
# compensated an incorrect thickness formula (physics review, 2026-10-03).
C_R = 0.5

# --- Core heat flux (Table I) ---
# 10 erg/(cm^2 s) = 10 * 1e-3 W/m^2 = 0.01 W/m^2
H_FLUX_ZERO = 0.0           # [W/m^2]
H_FLUX_10 = 0.01            # [W/m^2] (= 10 erg/(cm^2 s))


# --- Active parameter set ---------------------------------------------------
_ACTIVE = PAPER
ECCENTRICITY = PAPER.eccentricity
EUROPA_RADIUS = PAPER.radius
EUROPA_DENSITY = PAPER.density
EUROPA_GRAVITY = PAPER.gravity
EUROPA_MASS = PAPER.mass
A1_CONDUCTIVITY = PAPER.a1_conductivity
A0_CONDUCTIVITY = PAPER.a0_conductivity


def use_parameter_set(name):
    """Make ``name`` ('paper', 'modern', 'modern-klinger') the active set.

    Rebinds the module-level constants and returns the ParameterSet. Call it
    once, before any model is built. Prints the choice so that no run uses a
    parameter set silently.
    """
    global _ACTIVE, ECCENTRICITY, EUROPA_RADIUS, EUROPA_DENSITY
    global EUROPA_GRAVITY, EUROPA_MASS, A1_CONDUCTIVITY, A0_CONDUCTIVITY
    if isinstance(name, ParameterSet):
        ps = name
    else:
        try:
            ps = PARAMETER_SETS[name]
        except KeyError:
            raise ValueError(
                f"unknown parameter set {name!r}; choose from "
                f"{sorted(PARAMETER_SETS)}") from None
    _ACTIVE = ps
    ECCENTRICITY = ps.eccentricity
    EUROPA_RADIUS = ps.radius
    EUROPA_DENSITY = ps.density
    EUROPA_GRAVITY = ps.gravity
    EUROPA_MASS = ps.mass
    A1_CONDUCTIVITY = ps.a1_conductivity
    A0_CONDUCTIVITY = ps.a0_conductivity
    return ps


def active_parameter_set():
    """The ParameterSet currently bound to the module-level constants."""
    return _ACTIVE


@contextlib.contextmanager
def parameter_set(name):
    """Context manager: activate ``name``, restore the previous set on exit."""
    previous = _ACTIVE
    try:
        yield use_parameter_set(name)
    finally:
        use_parameter_set(previous)


def describe_active():
    """One-line description of the active set, for logs and run metadata."""
    ps = _ACTIVE
    return (f"parameter set '{ps.name}': e={ps.eccentricity}, a={ps.radius:.4e} m, "
            f"rho={ps.density} kg/m^3, g={ps.gravity} m/s^2, "
            f"k={ps.a1_conductivity:g}/T+{ps.a0_conductivity:g} W/(m K), C_R={C_R}")


def gamma():
    """Ratio of the tidal/rotational to the self-gravity potential (Eq B.26).

    gamma = n^2 * a^3 / (G * M_s)
    """
    return MEAN_MOTION**2 * EUROPA_RADIUS**3 / (G_GRAV * EUROPA_MASS)


def mu_prime():
    """Dimensionless rigidity mu' = mu / (rho * g * a) (Eq B.17).

    Uses Europa's mean density (not ice density) since this parameter
    describes the tidal deformation of the whole satellite.
    For Europa with mu = 4e9 Pa: mu' ~ 0.64
    """
    return ICE_RIGIDITY / (EUROPA_DENSITY * EUROPA_GRAVITY * EUROPA_RADIUS)


def g2(c_r=None):
    """Characteristic strain amplitude g2 = (45/88) * C_R * gamma * e (Eq B.25).

    The 45/88 prefactor comes from solving the Peale & Cassen (1978) boundary
    conditions for a thin incompressible elastic shell on a fluid interior
    (Eqs B.13-B.16 in the delta -> 0 limit).

    Parameters
    ----------
    c_r : float, optional
        Tidal correction factor. Default: module-level C_R.
    """
    if c_r is None:
        c_r = C_R
    return (45.0 / 88.0) * c_r * gamma() * ECCENTRICITY


def g3(c_r=None):
    """Strain amplitude g3 = (165/88) * C_R * gamma * e (Eq B.25).

    The 165/88 prefactor comes from the Peale & Cassen (1978) thin-shell
    solution. Note: g3 > g2, and the ratio g3/g2 = 165/45 = 11/3.
    In the thin shell limit, the strains are independent of shell rigidity.

    Parameters
    ----------
    c_r : float, optional
        Tidal correction factor. Default: module-level C_R.
    """
    if c_r is None:
        c_r = C_R
    return (165.0 / 88.0) * c_r * gamma() * ECCENTRICITY


def ntau_min(eta0=MELT_VISCOSITY, mu=ICE_RIGIDITY):
    """Minimum dimensionless Maxwell time at base of ice.

    (n * tau_M)_min = n * eta_0 / mu
    Nominally = 1 for eta_0 = 10^13 Pa s and mu chosen appropriately.
    """
    return MEAN_MOTION * eta0 / mu


def thermal_conductivity(T):
    """Temperature-dependent thermal conductivity of ice (Eq 2.3).

    k(T) = a1/T + a0 [W/(m K)]
    """
    return A1_CONDUCTIVITY / T + A0_CONDUCTIVITY


def thermal_conductivity_approx(T):
    """Simplified thermal conductivity k(T) ~ a1/T (Eq 2.3, a0 neglected)."""
    return A1_CONDUCTIVITY / T
