"""Turning what a database publishes into what the engine actually consumes.

Every number the Asteroid card auto-populates comes out of a database in the
units and the definition that database uses, and TWO of those definitions are
not the engine's. This module is where the translation happens, once, with the
conversion visible in the label rather than buried in a writer:

* **Diameter, not radius.** SBDB and HORIZONS publish a diameter; the engine's
  ``generic_body.shape.radius_km`` is a radius. That one is already handled in
  :mod:`gui_core.body_lookup` and is only named here for completeness.

* **GEOMETRIC albedo, not BOND albedo.** SBDB's ``albedo`` field is the visual
  geometric albedo p_V. The engine's ``generic_body.surface.albedo`` is a Bond
  albedo -- ``thermal_model.cpp:2906`` computes ``1.0 - planet.albedo`` and
  multiplies the solar constant by it, which is the definition of a Bond
  albedo and nothing else. Feeding p_V into that slot makes the body reflect
  far more sunlight than it does and run cold.

  The size of the error is not a rounding detail. For a bright body it is the
  larger part of the absorbed flux::

      Vesta-like:  p_V = 0.42, G = 0.32  ->  A_B = 0.2137
                   1 - A: 0.580 (p_V as Bond) vs 0.786 (correct)
                   absorbed flux LOW by 26.2%, equilibrium temperature LOW
                   by 7.3% -- about 13 K at 180 K

      Chimaera:    p_V = 0.037, G = 0.15 ->  A_B = 0.0145
                   1 - A: 0.963 vs 0.9855
                   absorbed flux LOW by 2.3%, temperature LOW by 0.6%

  A dark asteroid barely notices; a bright one is badly wrong. Neither says
  anything, because both run to completion.

The conversion is the standard one (Bowell et al. 1989, *Asteroids II*,
pp. 524-556): the phase integral of the IAU H, G magnitude system is

    q = 0.290 + 0.684 G

and the Bond albedo follows as A_B = p_V q. This is the same relation the
NEATM thermal-model literature uses to get a Bond albedo out of a published
p_V (Harris 1998, *Icarus* 131, 291-301, eq. 2), so a card that reports it is
reporting what the field would expect.

No Qt and no network here, so the arithmetic can be measured against a
hand-computed case rather than against itself.
"""

from __future__ import annotations

__all__ = [
    "DEFAULT_SLOPE_PARAMETER",
    "SLOPE_PARAMETER_RANGE",
    "THERMAL_INERTIA_GUIDANCE",
    "AlbedoConversion",
    "bond_albedo",
    "ecliptic_pole_to_equatorial",
    "phase_integral",
    "thermal_inertia_tooltip",
]


#: The IAU H, G system's default slope parameter, used when a body has no
#: fitted G. It is not a neutral placeholder -- it is the value the H
#: magnitudes in every catalog were themselves derived under -- but it IS an
#: assumption, so anything that falls back to it says so in the label.
DEFAULT_SLOPE_PARAMETER = 0.15

#: The range over which ``q = 0.290 + 0.684 G`` is a fit rather than an
#: extrapolation. Bowell et al. derived it across the observed population;
#: fitted G values for real objects run from about -0.12 (a few comets and
#: very low-albedo objects) to about 0.60 (the brightest E-types). Outside it
#: the relation still returns a number, and this module still returns that
#: number -- with a warning attached, because silently clamping a value the
#: user can see in SBDB would be a substitution nothing reports.
SLOPE_PARAMETER_RANGE = (-0.12, 0.60)


def phase_integral(slope_parameter=DEFAULT_SLOPE_PARAMETER):
    """The phase integral q of the IAU H, G system.

    ``q = 0.290 + 0.684 G`` (Bowell et al. 1989). At the default G = 0.15 this
    is 0.3926.
    """
    return 0.290 + 0.684 * float(slope_parameter)


class AlbedoConversion:
    """A geometric albedo turned into a Bond albedo, with its arithmetic kept.

    Everything the card needs to LABEL the field is here, not just the answer.
    A converted number with no visible conversion is indistinguishable from an
    unconverted one, and this whole module exists because that distinction was
    invisible.
    """

    def __init__(self, geometric_albedo, slope_parameter, q, bond,
                 slope_assumed, warnings):
        #: SBDB's p_V, exactly as published.
        self.geometric_albedo = geometric_albedo
        #: The G used -- fitted or assumed; ``slope_assumed`` says which.
        self.slope_parameter = slope_parameter
        #: The phase integral.
        self.q = q
        #: What goes into ``generic_body.surface.albedo``.
        self.bond = bond
        #: True when G was not published and the default stood in.
        self.slope_assumed = slope_assumed
        #: Anything the user must know. Empty is the normal case.
        self.warnings = tuple(warnings)

    def label(self):
        """The field label. Carries the inputs, so the value is checkable."""
        source = "G assumed" if self.slope_assumed else "G"
        return (f"Bond albedo (from SBDB p_V={self.geometric_albedo:g}, "
                f"{source}={self.slope_parameter:g})")

    def detail(self):
        """The one line under the field, and the body of its tooltip."""
        assumed = ""
        if self.slope_assumed:
            assumed = (f" SBDB publishes no slope parameter for this body, so "
                       f"the IAU default G = {DEFAULT_SLOPE_PARAMETER:g} was "
                       f"used; if you have a fitted G, type the Bond albedo "
                       f"yourself.")
        return (f"A_B = p_V x q with q = 0.290 + 0.684 G = {self.q:.4f}, so "
                f"A_B = {self.geometric_albedo:g} x {self.q:.4f} = "
                f"{self.bond:.6g}. SBDB publishes the GEOMETRIC albedo p_V; "
                f"the engine's surface albedo is a BOND albedo (it computes "
                f"1 - albedo and multiplies the solar constant by it). Using "
                f"p_V here directly would absorb too little sunlight and run "
                f"the body cold.{assumed}"
                + ("" if not self.warnings else " " + " ".join(self.warnings)))

    def __repr__(self):                                   # pragma: no cover
        return (f"<AlbedoConversion p_V={self.geometric_albedo:g} "
                f"G={self.slope_parameter:g} -> A_B={self.bond:.6g}>")


def bond_albedo(geometric_albedo, slope_parameter=None):
    """``AlbedoConversion`` from a published p_V, or None if p_V is unusable.

    Returns None -- rather than a guess -- when there is no geometric albedo to
    convert. A body SBDB has no albedo for must keep whatever the user typed;
    inventing one would be the same class of silent substitution as the p_V
    confusion this function exists to end.

    ``slope_parameter`` of None means "not published", and the IAU default
    stands in with ``slope_assumed`` set so the label can say so.
    """
    try:
        p_v = float(geometric_albedo)
    except (TypeError, ValueError):
        return None
    # SBDB uses 0.0 and this project uses -1.0 as "not set". A real geometric
    # albedo is positive and at most about 1.7 (Enceladus-class ices exceed 1
    # because p_V is a ratio to a Lambert disk, not a fraction of anything),
    # so the upper bound is generous on purpose and only rejects nonsense.
    if not (p_v > 0.0) or p_v != p_v or p_v > 2.0:
        return None

    warnings = []
    assumed = slope_parameter is None
    if assumed:
        slope = DEFAULT_SLOPE_PARAMETER
    else:
        try:
            slope = float(slope_parameter)
        except (TypeError, ValueError):
            slope = DEFAULT_SLOPE_PARAMETER
            assumed = True
        else:
            if slope != slope:                            # NaN
                slope, assumed = DEFAULT_SLOPE_PARAMETER, True

    low, high = SLOPE_PARAMETER_RANGE
    if not assumed and not (low <= slope <= high):
        # NOT clamped. The number is used and the extrapolation is declared.
        warnings.append(
            f"WARNING: G = {slope:g} is outside the {low:g} to {high:g} range "
            f"the q(G) relation was fitted over, so the phase integral here is "
            f"an extrapolation. The value is used as published, not clamped -- "
            f"check it before trusting the Bond albedo.")

    q = phase_integral(slope)
    bond = p_v * q
    if bond >= 1.0:
        warnings.append(
            f"WARNING: this gives a Bond albedo of {bond:.4g}, at or above 1, "
            f"which would absorb no sunlight at all. Check p_V and G.")
    return AlbedoConversion(p_v, slope, q, bond, assumed, warnings)


# ---------------------------------------------------------------------------
# The spin pole: published in ecliptic coordinates, consumed in equatorial
# ---------------------------------------------------------------------------

def ecliptic_pole_to_equatorial(lambda_deg, beta_deg, obliquity_deg=None):
    """``(pole_ra_deg, pole_dec_deg)`` from an ecliptic pole ``(lambda, beta)``.

    The second unit trap on this card, and a quieter one than the albedo.
    BOTH sources of a spin pole publish it in ECLIPTIC coordinates:

    * SBDB's ``pole`` physical parameter is documented as the pole direction in
      ecliptic longitude/latitude and arrives as the string ``"lambda/beta"``.
    * DAMIT's ``spin.txt`` is ecliptic ``lambda beta period`` by construction.

    ``generic_body.spin.pole_ra_deg`` and ``pole_dec_deg`` are EQUATORIAL. A
    pole written straight across is wrong by up to the obliquity, 23.4 degrees
    -- enough to move the sub-solar latitude, the seasons and the polar
    temperatures, with nothing in the log, because a spin pole has no
    plausibility check that 23 degrees would fail.

    THIS IS A RE-EXPORT, not a second implementation. The arithmetic lives in
    ``horizons.horizons_client.ecliptic_pole_to_equatorial``, because that is
    the module that parses SBDB's ecliptic ``pole`` string and must publish an
    equatorial pair for its own ``PhysicalParams`` contract. Two copies of a
    frame conversion is how the two copies drift, and one of them was already
    subtly narrower: ``damit_client.DAMITSpinParams.pole_ra_deg`` divides
    through by ``cos(beta)``, which is algebraically identical for every real
    pole but raises on an exactly polar ``beta = +-90`` -- a legal ecliptic
    pole (the orbit-normal case), and the one an idealized test would use.
    """
    from horizons.horizons_client import (J2000_OBLIQUITY_DEG,
                                          ecliptic_pole_to_equatorial as _impl)
    if obliquity_deg is None:
        obliquity_deg = J2000_OBLIQUITY_DEG
    return _impl(lambda_deg, beta_deg, obliquity_deg)


# ---------------------------------------------------------------------------
# Thermal inertia: deliberately NOT auto-populated
# ---------------------------------------------------------------------------

#: Why the card leaves thermal inertia to the user, and what the literature
#: says the answer tends to be. No standard database publishes a thermal
#: inertia the way SBDB publishes a diameter: the values that exist come from
#: individual thermophysical-model fits to individual infrared data sets, each
#: with its own roughness and shape assumptions, and they are not
#: interchangeable. Auto-filling one would be the card claiming a provenance
#: it does not have.
THERMAL_INERTIA_GUIDANCE = (
    "No standard database publishes thermal inertia the way SBDB publishes a "
    "diameter, so this is left for you to set. Published fits, in "
    "J m^-2 K^-1 s^-1/2:\n"
    "  fine lunar-like regolith           ~ 30-60\n"
    "  large main-belt asteroids (>100 km) ~ 5-25\n"
    "  kilometer-scale near-Earth asteroids ~ 200-600\n"
    "  (101955) Bennu, rubble pile          ~ 300\n"
    "  (162173) Ryugu, rubble pile          ~ 200-300\n"
    "  bare rock / boulder-dominated        > 1000\n"
    "Thermal inertia falls with body size because gravity retains finer "
    "regolith on larger bodies (Delbo et al. 2007, Icarus 190, 236-249; "
    "Delbo et al. 2015, Asteroids IV, pp. 107-128). Bennu: Rozitis et al. "
    "2020, Sci. Adv. 6, eabc3699. Ryugu: Shimaki et al. 2020, Icarus 348, "
    "113835.\n"
    "This card sets ks, rhos and cp0 rather than the inertia itself; the "
    "inertia it implies is reported under those fields."
)


def thermal_inertia_tooltip():
    """The tooltip body for the surface-property block."""
    return THERMAL_INERTIA_GUIDANCE
