"""Geometric albedo p_V to Bond albedo A_B -- the one implementation.

This lives in ``src/python/horizons`` rather than in the GUI because THREE
things need it and only one of them is a GUI:

* ``horizons.body_config_writer`` writes ``data/horizons_cache/bodies/*/
  body_config.yaml`` and must store a Bond albedo;
* ``scripts/fix_cached_body_albedo.py`` rewrites the files written before it
  did;
* ``gui_core.body_properties`` labels the conversion on the Asteroid card, and
  RE-EXPORTS from here rather than keeping a second copy -- the same
  arrangement ``ecliptic_pole_to_equatorial`` already uses, and for the same
  reason: two copies of a unit conversion is how the two copies drift.

Why it matters
--------------

``generic_body.surface.albedo`` is a BOND albedo: the engine computes
``1 - albedo`` and multiplies the solar constant by it
(``thermal_model.cpp:2906``), which is the definition of a Bond albedo and
nothing else.  SBDB's ``albedo`` field is the visual GEOMETRIC albedo p_V.
Writing p_V into that slot makes a body reflect far more sunlight than it does
and run cold, and nothing downstream reports it -- the run completes.

    Vesta-like:  p_V = 0.42, G = 0.32  ->  A_B = 0.2137
                 1 - A: 0.580 (p_V as Bond) vs 0.786 (correct)
                 absorbed flux LOW by 26.2%, equilibrium temperature LOW by
                 7.3% -- about 13 K at 180 K

    Bennu:       p_V = 0.044, G = -0.031 -> A_B = 0.0118
                 1 - A: 0.956 vs 0.988, absorbed flux LOW by 3.3%

    Chimaera:    p_V = 0.0372, G assumed 0.15 -> A_B = 0.0146

A dark asteroid barely notices; a bright one is badly wrong.  Neither says
anything.

The conversion is the standard one (Bowell et al. 1989, *Asteroids II*,
pp. 524-556): the phase integral of the IAU H, G magnitude system is

    q = 0.290 + 0.684 G

and the Bond albedo follows as A_B = p_V q.  This is the relation the NEATM
thermal-model literature uses to get a Bond albedo from a published p_V
(Harris 1998, *Icarus* 131, 291-301, eq. 2).

No Qt, no network and no yaml here, so the arithmetic can be measured against
a hand-computed case rather than against itself.
"""

from __future__ import annotations

__all__ = [
    "DEFAULT_SLOPE_PARAMETER",
    "SLOPE_PARAMETER_RANGE",
    "bond_albedo_from_pv",
    "phase_integral",
]


#: The IAU H, G system's default slope parameter, used when a body has no
#: fitted G.  It is not a neutral placeholder -- it is the value the H
#: magnitudes in every catalog were themselves derived under -- but it IS an
#: assumption, so anything that falls back to it records that it did.
DEFAULT_SLOPE_PARAMETER = 0.15

#: The range over which ``q = 0.290 + 0.684 G`` is a fit rather than an
#: extrapolation.  Bowell et al. derived it across the observed population;
#: fitted G values for real objects run from about -0.12 (a few comets and very
#: low-albedo objects) to about 0.60 (the brightest E-types).  Outside it the
#: relation still returns a number, and this module still returns that number
#: -- with a warning attached, because silently clamping a value the user can
#: see in SBDB would be a substitution nothing reports.
SLOPE_PARAMETER_RANGE = (-0.12, 0.60)


def phase_integral(slope_parameter=DEFAULT_SLOPE_PARAMETER):
    """The phase integral q of the IAU H, G system.

    ``q = 0.290 + 0.684 G`` (Bowell et al. 1989).  At the default G = 0.15 this
    is 0.3926.
    """
    return 0.290 + 0.684 * float(slope_parameter)


def bond_albedo_from_pv(geometric_albedo, slope_parameter=None):
    """A dict describing the conversion, or ``None`` if p_V is unusable.

    Returns ``None`` -- rather than a guess -- when there is no geometric
    albedo to convert.  A body SBDB has no albedo for must keep whatever the
    user typed; inventing one would be the same class of silent substitution as
    the p_V confusion this function exists to end.

    ``slope_parameter`` of ``None`` means "not published", and the IAU default
    stands in with ``slope_assumed`` set so every caller can say so.

    Keys: ``geometric_albedo``, ``slope_parameter``, ``slope_assumed``, ``q``,
    ``bond``, ``warnings`` (a tuple) and ``provenance`` (one line, safe to
    write into a YAML file).
    """
    try:
        p_v = float(geometric_albedo)
    except (TypeError, ValueError):
        return None
    # SBDB uses 0.0 and this project uses -1.0 as "not set".  A real geometric
    # albedo is positive and at most about 1.7 (Enceladus-class ices exceed 1
    # because p_V is a ratio to a Lambert disk, not a fraction of anything), so
    # the upper bound is generous on purpose and only rejects nonsense.
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
            slope, assumed = DEFAULT_SLOPE_PARAMETER, True
        else:
            if slope != slope:                                # NaN
                slope, assumed = DEFAULT_SLOPE_PARAMETER, True

    low, high = SLOPE_PARAMETER_RANGE
    if not assumed and not (low <= slope <= high):
        # NOT clamped.  The number is used and the extrapolation is declared.
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

    provenance = (
        f"Bond albedo A_B = p_V x q, q = 0.290 + 0.684 G (Bowell et al. 1989); "
        f"p_V = {p_v:g} from SBDB, G = {slope:g} "
        f"({'IAU default, SBDB publishes none' if assumed else 'SBDB'}), "
        f"q = {q:.4f}, A_B = {bond:.6g}.")
    if warnings:
        provenance += " " + " ".join(warnings)

    return {
        "geometric_albedo": p_v,
        "slope_parameter": slope,
        "slope_assumed": assumed,
        "q": q,
        "bond": bond,
        "warnings": tuple(warnings),
        "provenance": provenance,
    }
