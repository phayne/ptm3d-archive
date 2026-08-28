"""Which bodies the engine can actually run, and with what properties.

``src/cpp/thermal/planetary_properties.h`` is the only authority on that.
Until 2026-08-26 nothing on the Python side asked it: the GUI offered seven
built-in bodies, the engine implemented three, and the other four resolved to
Europa's properties and ran to completion with zero absorbed flux and exit 0
(``BUGS.md`` ENG-BODY-UNBACKED-RUN). The engine now refuses those names; this
module is the other half, so the GUI, the tutorials and any script can ask what
is runnable instead of carrying their own copy of the answer.

Same pattern, and the same reasoning, as :mod:`ptm.engine_config`:

* :func:`parse_body_capabilities` reads ``planetary_properties.h`` and returns
  ``{body: {field: value}}`` for every implemented body.
* :data:`BODY_CAPABILITIES` in ``ptm/body_table.py`` is that result, committed,
  so the table is available without the C++ sources at hand.
* ``src/python/tests/test_body_capabilities.py`` re-derives the table from the
  source and fails if it no longer matches the committed one -- and, separately,
  fails if any of the FOUR other Python/GUI copies of these numbers
  (``gui_tooltips.BODY_SYNODIC_PERIODS``, ``gui_tooltips.BODY_DEFAULTS``,
  ``gui_core.facet_scale._BODY_RADII_M``, ``ptm.results.BODY_RADII``) disagrees
  with the engine. A body added, removed, or retuned in C++ therefore breaks a
  test instead of quietly leaving the GUI describing a different simulator.

Regenerate after an intentional engine change::

    python3 -m ptm.body_capabilities --regenerate

Limits worth stating, in the same spirit as ``engine_config``:

* The parse is textual. It understands ``props.<field> = <expression>;`` and
  local ``constexpr double k<Name> = <expression>;`` inside each
  ``get<Body>Properties()`` function, where an expression is numeric literals,
  previously-declared local constants, and ``+ - * / ( )``. That is exactly the
  shape those functions have; if it changes, the parse loses fields, which is
  why the test asserts that every required field is present for every body
  rather than trusting whatever it happens to find.
* Data availability is NOT in the header and is not baked in here. Which
  albedo maps and topography grids exist is a property of the fetched dataset,
  so :func:`albedo_map_status` and :func:`topography_status` answer it by
  looking, using the engine's own file-naming contract
  (``maps/albedo_manager.cpp``, ``mesh/mesh_utils.cpp``), which IS parsed from
  the sources and committed.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

from .body_table import (ALBEDO_BINARY_PATTERN, ALBEDO_FITS_PATTERNS,
                         ALBEDO_MAP_RESOLUTIONS, BODY_CAPABILITIES,
                         CAPABILITY_FIELDS, GENERIC_BODY_SURFACE_DEFAULTS,
                         IMPLEMENTED_BODIES, TOPOGRAPHY_PATTERNS)

__all__ = [
    "ALBEDO_BINARY_PATTERN",
    "ALBEDO_FITS_PATTERNS",
    "ALBEDO_MAP_RESOLUTIONS",
    "BODY_CAPABILITIES",
    "CAPABILITY_FIELDS",
    "GENERIC_BODY_SURFACE_DEFAULTS",
    "GENERIC_SURFACE_FIELDS",
    "IMPLEMENTED_BODIES",
    "TOPOGRAPHY_PATTERNS",
    "albedo_map_status",
    "capability_for",
    "is_implemented",
    "parse_body_capabilities",
    "parse_generic_body_defaults",
    "solar_day_days",
    "topography_status",
]

_HEADER = "src/cpp/thermal/planetary_properties.h"
_ALBEDO = "src/cpp/maps/albedo_manager.cpp"
_MESH = "src/cpp/mesh/mesh_utils.cpp"
_GENERIC = "src/cpp/orbital/generic_body.h"
_SIM = "src/cpp/main/ptm_thermal_sim.cpp"

#: The `generic_body.surface.*` keys, in the order the card shows them.
GENERIC_SURFACE_FIELDS = ("albedo", "emissivity", "ks", "kd", "rhos", "rhod",
                          "H", "cp0", "Qb", "chi")

_SECONDS_PER_DAY = 24.0 * 3600.0

#: Significant digits kept when a period stored in SECONDS is reported in DAYS.
#: The header writes `29.530588 * 24 * 3600`, so the round trip is exact to
#: ~1e-15 and reports 29.530588000000005 -- which then fails an exact comparison
#: against the same literal in a Python table for no physical reason. Twelve
#: figures is far beyond the precision of any number here and far beyond any
#: difference that could matter, so the round trip is the only thing it hides.
_PERIOD_SIG_FIGS = 12


def _round_sig(value, figs=_PERIOD_SIG_FIGS):
    if value == 0.0 or value != value:
        return value
    from math import floor, log10
    return round(value, -int(floor(log10(abs(value)))) + (figs - 1))


# --------------------------------------------------------------------------
# Public helpers
# --------------------------------------------------------------------------

def is_implemented(body):
    """True if ``body`` names a body the engine can run without ``generic_body``.

    Case-insensitive, matching ``thermal::isImplementedBody``. Anything else is
    a FATAL at config parse, not a fallback.
    """
    lowered = str(body or "").strip().lower()
    return any(lowered == name.lower() for name in IMPLEMENTED_BODIES)


def capability_for(body):
    """The engine's capability row for ``body``. ``KeyError`` if not implemented."""
    lowered = str(body or "").strip().lower()
    for name in IMPLEMENTED_BODIES:
        if lowered == name.lower():
            return BODY_CAPABILITIES[name]
    raise KeyError(
        f"{body!r} is not an implemented body; the engine refuses it at config "
        f"parse. Implemented: {', '.join(IMPLEMENTED_BODIES)}. Use a "
        f"generic_body: block for anything else.")


def solar_day_days(body):
    """Solar (synodic) day in Earth days -- the period the engine forces at."""
    return capability_for(body)["solar_day_days"]


def albedo_map_status(body, data_root):
    """Which albedo maps for ``body`` exist under ``data_root/albedo_maps``.

    Returns ``(fits, resolutions)``: the FITS files present, in the order the
    engine tries them, and the icosphere resolutions whose per-face binary is
    present, highest first. Both empty means the engine will fall back to the
    constant body albedo and say so.

    This LOOKS rather than remembers, because whether a map is on disk is a
    property of the fetched dataset and of ``PTM_DATA_ROOT``, not of the engine.
    ``gui_core.datasets.albedo_status`` answers a body+resolution question from
    a hardcoded set instead, which is why it can promise a per-face map the
    engine will reject (``BUGS.md`` GUI-ALBEDO-STATUS-IGNORES-MESH).
    """
    directory = Path(data_root) / "albedo_maps"
    proper = str(body or "").strip().capitalize()
    fits = [name.format(Body=proper) for name in ALBEDO_FITS_PATTERNS
            if (directory / name.format(Body=proper)).exists()]
    resolutions = tuple(
        res for res in ALBEDO_MAP_RESOLUTIONS
        if (directory / ALBEDO_BINARY_PATTERN.format(Body=proper, res=res)).exists())
    return fits, resolutions


def topography_status(body, data_root):
    """Icosphere resolutions with a shipped topography grid for ``body``.

    Same contract as :func:`albedo_map_status`: it looks. The patterns are the
    ones ``MeshUtils::checkForTopographicMesh`` tries.
    """
    root = Path(data_root) / "topo"
    lower = str(body or "").strip().lower()
    found = []
    for res in ALBEDO_MAP_RESOLUTIONS:
        for pattern in TOPOGRAPHY_PATTERNS:
            candidate = root / pattern.format(body=lower,
                                              Body=lower.capitalize(), res=res)
            if candidate.exists():
                found.append(res)
                break
    return tuple(found)


# --------------------------------------------------------------------------
# Deriving the table from the C++ source
# --------------------------------------------------------------------------

def _strip_comments(text):
    text = re.sub(r"//[^\n]*", "", text)
    return re.sub(r"/\*.*?\*/", "", text, flags=re.S)


_ALLOWED_NODES = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub,
                  ast.Mult, ast.Div, ast.USub, ast.UAdd, ast.Constant,
                  ast.Name, ast.Load)


def _eval_cxx_number(expr, symbols):
    """Evaluate a C++ arithmetic expression over numeric literals and locals.

    Deliberately not ``eval``: an AST walk restricted to the four operators and
    to names already declared in this function. Anything else raises, so a
    header that grows a call or a cast fails the parse loudly instead of
    contributing a wrong number to a committed table.
    """
    cleaned = expr.strip().rstrip(";").strip()
    # C++ float suffixes and digit separators, neither of which Python parses.
    cleaned = re.sub(r"(\d)[fF]\b", r"\1", cleaned)
    cleaned = cleaned.replace("'", "")
    tree = ast.parse(cleaned, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(
                f"{_HEADER}: cannot evaluate {expr!r} -- it contains "
                f"{type(node).__name__}, which this parser deliberately "
                f"refuses rather than guess at")
        if isinstance(node, ast.Name) and node.id not in symbols:
            raise ValueError(
                f"{_HEADER}: {expr!r} refers to {node.id!r}, which is not a "
                f"local constexpr declared above it")

    def walk(node):
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant):
            return float(node.value)
        if isinstance(node, ast.Name):
            return symbols[node.id]
        if isinstance(node, ast.UnaryOp):
            value = walk(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        left, right = walk(node.left), walk(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        return left / right

    return walk(tree)


def _function_body(text, name, source=_HEADER):
    """The braced body of ``static ThermophysicalProperties <name>() { ... }``."""
    match = re.search(r"\b" + re.escape(name) + r"\s*\(\s*\)\s*\{", text)
    if match is None:
        raise ValueError(f"{source}: no function {name}()")
    start = match.end() - 1
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:i]
    raise ValueError(f"{source}: {name}() is not brace-balanced")


_SCALAR_FIELDS = ("day", "year", "rAU", "S", "albedo", "emissivity", "Qb",
                  "ks", "kd", "rhos", "rhod", "H", "cp0", "body_radius_km")


def _parse_one_body(body_text):
    """``{field: value}`` for one get<Body>Properties() body."""
    symbols = {}
    for name, expr in re.findall(
            r"constexpr\s+double\s+(\w+)\s*=\s*([^;]+);", body_text):
        symbols[name] = _eval_cxx_number(expr, symbols)

    out = {}
    for field in _SCALAR_FIELDS:
        matches = re.findall(
            r"props\." + field + r"\s*=\s*([^;]+);", body_text)
        if not matches:
            continue
        # Last assignment wins, as it would at runtime.
        out[field] = _eval_cxx_number(matches[-1], symbols)

    for field in ("cpCoeff", "albedoCoef"):
        matches = re.findall(
            r"props\." + field + r"\s*=\s*\{([^}]*)\}\s*;", body_text)
        if matches:
            items = [t.strip() for t in matches[-1].split(",") if t.strip()]
            out[field] = tuple(_eval_cxx_number(t, symbols) for t in items)

    metakernel = re.findall(
        r'props\.spice_metakernel\s*=\s*"([^"]*)"\s*;', body_text)
    if metakernel:
        out["spice_metakernel"] = metakernel[-1]

    # Derived, and named in the units a user thinks in. The engine stores
    # seconds; every Python/GUI table that duplicates these stores days.
    if "day" in out:
        out["solar_day_days"] = _round_sig(out.pop("day") / _SECONDS_PER_DAY)
    if "year" in out:
        out["year_days"] = _round_sig(out.pop("year") / _SECONDS_PER_DAY)
    if "body_radius_km" in out:
        out["radius_km"] = out.pop("body_radius_km")
    return out


def parse_generic_body_defaults(repo_root):
    """The engine's own surface properties for a body it has no table for.

    ``{field: value}`` over :data:`GENERIC_SURFACE_FIELDS`, read from the two
    places the engine keeps them:

    * ``GenericBody::defaultRockyAsteroid()`` (``generic_body.h``) for the nine
      ``thermo`` values. This is not a suggestion the parser applies when the
      YAML is silent -- ``ptm_thermal_sim.cpp:14759`` starts EVERY generic body
      from this struct and then overwrites only the fields the configuration
      gives a positive value for. A generic-body run with no
      ``generic_body.surface:`` block is a run on exactly these numbers.
    * ``chi``, which is not in that struct: the solver takes
      ``(chi_override >= 0) ? chi_override : 2.7`` (``ptm_thermal_sim.cpp``,
      Hayne et al. 2017), so the default is a literal at the forwarding site
      and is read from there rather than copied.

    Why this is worth parsing rather than typing: the asteroid card left every
    one of these blank, so a user either ran on values they never saw or, as on
    2026-08-28, typed their own guesses in to make the card stop complaining.
    Seeding the card from a hand-copied table would have made the GUI's idea of
    "the default" drift away from the engine's the first time either moved.
    """
    root = Path(repo_root)
    generic = _strip_comments((root / _GENERIC).read_text())
    body = _function_body(generic, "defaultRockyAsteroid", source=_GENERIC)

    symbols = {}
    for name, expr in re.findall(
            r"constexpr\s+double\s+(\w+)\s*=\s*([^;]+);", body):
        symbols[name] = _eval_cxx_number(expr, symbols)

    out = {}
    for field in GENERIC_SURFACE_FIELDS:
        matches = re.findall(
            r"body\.thermo\." + field + r"\s*=\s*([^;]+);", body)
        if matches:
            out[field] = _eval_cxx_number(matches[-1], symbols)

    sim = _strip_comments((root / _SIM).read_text())
    chi = re.search(
        r"chi_override\s*>=\s*0\s*\)\s*\?\s*chi_override\s*:\s*([0-9.]+)", sim)
    if chi is None:
        raise ValueError(
            f"{_SIM}: no `(chi_override >= 0) ? chi_override : <default>` "
            f"expression; the generic-body chi default would be invented")
    out["chi"] = float(chi.group(1))

    missing = [f for f in GENERIC_SURFACE_FIELDS if f not in out]
    if missing:
        raise ValueError(
            f"{_GENERIC}: defaultRockyAsteroid() no longer assigns "
            f"{', '.join(missing)}; the card would seed an incomplete surface")
    return out


def parse_body_capabilities(repo_root):
    """Derive the implemented-body table from the C++ sources.

    Returns ``(bodies, capabilities, file_patterns)``:

    * ``bodies``        tuple of canonical names, in the header's own order
    * ``capabilities``  ``{body: {field: value}}``
    * ``file_patterns`` ``(fits_patterns, binary_pattern, resolutions,
      topo_patterns)`` -- the engine's own map-discovery contract
    """
    root = Path(repo_root)
    header = _strip_comments((root / _HEADER).read_text())

    listed = re.search(
        r"kImplementedBodies\s*=\s*\{([^}]*)\}", header)
    if listed is None:
        raise ValueError(
            f"{_HEADER}: no kImplementedBodies initializer; the parse would "
            f"silently report that the engine implements nothing")
    bodies = tuple(name for name in re.findall(r'"([^"]+)"', listed.group(1)))
    if not bodies:
        raise ValueError(f"{_HEADER}: kImplementedBodies is empty")

    capabilities = {}
    for body in bodies:
        capabilities[body] = _parse_one_body(
            _function_body(header, f"get{body}Properties"))

    albedo = _strip_comments((root / _ALBEDO).read_text())
    # Only the fits_patterns vector: the binary loop below it builds its name
    # from the same `dir + proper_name + "..."` shape, so an unanchored search
    # picks up "_Albedo_res" as a fifth FITS pattern.
    fits_block = re.search(
        r"fits_patterns\s*=\s*\{(.*?)\}\s*;", albedo, flags=re.S)
    if fits_block is None:
        raise ValueError(f"{_ALBEDO}: no fits_patterns initializer found")
    fits = tuple(re.findall(r'"([^"]+)"', fits_block.group(1)))
    if not fits:
        raise ValueError(f"{_ALBEDO}: fits_patterns is empty")
    binary = re.search(
        r'proper_name\s*\+\s*"(_Albedo_res)"\s*\+\s*std::to_string\(res\)\s*\+\s*"(\.bin)"',
        albedo)
    if binary is None:
        raise ValueError(f"{_ALBEDO}: no per-face binary map name found")
    loop = re.search(r"for\s*\(int res\s*=\s*(\d+);\s*res\s*>=\s*(\d+);", albedo)
    if loop is None:
        raise ValueError(f"{_ALBEDO}: no binary-map resolution loop found")
    resolutions = tuple(range(int(loop.group(1)), int(loop.group(2)) - 1, -1))

    # Topography. Only the three DIRECT fast-path candidates are represented;
    # checkForTopographicMesh also tries case variants of the legacy name and
    # then falls back to a relaxed directory scan, and neither is expressible
    # as a pattern. Reported honestly rather than approximated: a caller that
    # gets an empty answer from topography_status() should conclude "no grid
    # under the standard names", not "the engine cannot find one".
    mesh = _strip_comments((root / _MESH).read_text())
    ico = re.search(r'ico_suffix\s*=\s*std::string\("(_ico)"\)', mesh)
    legacy = re.search(r'suffix_exact\s*=\s*std::string\("(_Topo_res)"\)', mesh)
    prefix = re.search(r'\("(topo_)"\s*\+\s*bn_lower\s*\+\s*ico_suffix\)', mesh)
    if not (ico and legacy and prefix):
        raise ValueError(
            f"{_MESH}: checkForTopographicMesh no longer builds its candidate "
            f"names from the literals this parser reads; the topography "
            f"patterns would be silently wrong")
    topo = (
        "{body}/" + prefix.group(1) + "{body}" + ico.group(1) + "{res}.bin",
        prefix.group(1) + "{body}" + ico.group(1) + "{res}.bin",
        "{Body}" + legacy.group(1) + "{res}.bin",
    )
    return (bodies, capabilities,
            (tuple("{Body}" + name for name in fits),
             "{Body}" + binary.group(1) + "{res}" + binary.group(2),
             resolutions, topo))


# --------------------------------------------------------------------------
# Regeneration
# --------------------------------------------------------------------------

_GENERATED_HEADER = '''\
"""Implemented planetary bodies and their engine properties -- GENERATED.

Do not edit by hand. Derived from src/cpp/thermal/planetary_properties.h,
src/cpp/maps/albedo_manager.cpp, src/cpp/orbital/generic_body.h and
src/cpp/main/ptm_thermal_sim.cpp by

    python3 -m ptm.body_capabilities --regenerate

src/python/tests/test_body_capabilities.py re-derives this from the C++ sources
on every run and fails if the two disagree, so a body added, removed or retuned
in the engine shows up as a test failure and a diff to this file rather than as
a GUI that quietly describes a different simulator.

{count} implemented bodies.
"""

#: Body names `simulation.body` accepts with no `generic_body:` block. Anything
#: else is a FATAL at config parse (thermal::isImplementedBody). The GUI must
#: not offer a name that is not here.
IMPLEMENTED_BODIES = {bodies}

#: Fields every row carries. `solar_day_days` is the SOLAR (synodic) day, which
#: is what the engine forces at -- not the sidereal rotation period.
CAPABILITY_FIELDS = {fields}

#: {{body: {{field: engine value}}}}. Lengths in km, periods in Earth days,
#: fluxes in W/m^2, conductivities in W/m/K, densities in kg/m^3.
BODY_CAPABILITIES = {capabilities}

#: FITS albedo-map names the engine tries, in order (AlbedoManager::tryLoadBestMap).
ALBEDO_FITS_PATTERNS = {fits}

#: Per-face binary albedo-map name, tried at each of ALBEDO_MAP_RESOLUTIONS.
ALBEDO_BINARY_PATTERN = {binary!r}

#: Icosphere resolutions the engine tries for a per-face binary map, in order.
ALBEDO_MAP_RESOLUTIONS = {resolutions}

#: Topography grid names (MeshUtils::checkForTopographicMesh), relative to the
#: topography directory.
TOPOGRAPHY_PATTERNS = {topo}

#: `generic_body.surface.*` for a body the engine has no table for. NOT a GUI
#: suggestion: ptm_thermal_sim.cpp starts every generic body from
#: GenericBody::defaultRockyAsteroid() and overwrites only the fields the
#: configuration gives a positive value for, so a run with no
#: `generic_body.surface:` block runs on exactly these. chi is the one that is
#: not in that struct; the solver's own `? chi_override : 2.7`.
GENERIC_BODY_SURFACE_DEFAULTS = {generic_defaults}
'''


def _render_tuple(items, indent="    "):
    if not items:
        return "()"
    return "(\n" + "".join(f"{indent}{item!r},\n" for item in items) + ")"


def _render_capabilities(capabilities, fields):
    out = ["{"]
    for body in sorted(capabilities):
        out.append(f"    {body!r}: {{")
        for field in fields:
            if field in capabilities[body]:
                out.append(f"        {field!r}: {capabilities[body][field]!r},")
        out.append("    },")
    out.append("}")
    return "\n".join(out)


def _render_generic_defaults(defaults):
    out = ["{"]
    for field in GENERIC_SURFACE_FIELDS:
        out.append(f"    {field!r}: {defaults[field]!r},")
    out.append("}")
    return "\n".join(out)


def _render_module(bodies, capabilities, patterns, generic_defaults):
    fits, binary, resolutions, topo = patterns
    fields = tuple(sorted({f for row in capabilities.values() for f in row}))
    return _GENERATED_HEADER.format(
        count=len(bodies),
        bodies=_render_tuple(bodies),
        fields=_render_tuple(fields),
        capabilities=_render_capabilities(capabilities, fields),
        fits=_render_tuple(fits),
        binary=binary,
        resolutions=repr(resolutions),
        topo=_render_tuple(topo),
        generic_defaults=_render_generic_defaults(generic_defaults),
    )


def _main(argv=None):
    parser = argparse.ArgumentParser(
        description="Regenerate or check ptm/body_table.py against the C++ sources.")
    parser.add_argument("--repo-root", default=None,
                        help="PTM3D checkout (default: found by walking up from this file)")
    parser.add_argument("--regenerate", action="store_true",
                        help="rewrite ptm/body_table.py from the C++ sources")
    args = parser.parse_args(argv)

    from . import paths
    root = Path(args.repo_root) if args.repo_root else paths.find_repo_root(Path(__file__))
    if root is None:
        parser.error("no PTM3D checkout found; pass --repo-root")

    bodies, capabilities, patterns = parse_body_capabilities(root)
    generic_defaults = parse_generic_body_defaults(root)
    target = Path(__file__).with_name("body_table.py")
    rendered = _render_module(bodies, capabilities, patterns, generic_defaults)
    if args.regenerate:
        target.write_text(rendered)
        print(f"wrote {target} ({len(bodies)} bodies: {', '.join(bodies)})")
        return 0

    current = target.read_text() if target.exists() else ""
    if current == rendered:
        print(f"body_table.py is current ({len(bodies)} bodies)")
        return 0
    print("body_table.py is STALE; rerun with --regenerate", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(_main())
