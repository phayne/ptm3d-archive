#!/usr/bin/env python3
"""Body parity: what Python and the GUI believe about bodies vs the engine.

``src/cpp/thermal/planetary_properties.h`` implements three bodies. The GUI
offered seven, and the four with no properties resolved to Europa's and ran to
completion with zero absorbed flux and exit 0 (``BUGS.md``
ENG-BODY-UNBACKED-RUN). The engine now refuses those names at config parse;
this file is the drift guard that stops the two sides diverging again -- the
same job ``test_config_parity.py`` does for configuration keys.

Six things are asserted, in the order they would break:

1. the extractor still understands ``planetary_properties.h`` (every body has
   every required field, so a parse that silently found nothing cannot pass);
2. ``ptm/body_table.py`` still equals what the C++ sources say -- change a
   radius, a day or a metakernel in C++ and this fails until the table is
   regenerated;
3. the committed file is byte-current with a fresh render;
4. every OTHER copy of these numbers in Python and the GUI agrees with the
   engine -- ``gui_tooltips.BODY_SYNODIC_PERIODS``,
   ``gui_tooltips.BODY_DEFAULTS``, ``gui_core.facet_scale._BODY_RADII_M``
   and ``ptm.results.BODY_RADII``;
5. the GUI's built-in body list is DERIVED from the generated table rather
   than written out, so it cannot grow a body the engine refuses;
6. the engine really does refuse an unbacked name -- asserted against the
   header text, so it holds without a build.

Check 4 needs the GUI sources on ``sys.path``; it does NOT need PyQt6, because
``gui_tooltips`` and ``gui_core.facet_scale`` are deliberately Qt-free. A
genuinely missing module is a failure, not a skip. Check 5 is structural for
the same reason, with an additional runtime assertion that runs only where the
panel stack (PyQt6 and PyVista) imports; when it cannot, it says so on stdout
rather than passing quietly.

Run directly (no pytest needed)::

    python3 src/python/tests/test_body_capabilities.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "python"))
sys.path.insert(0, str(REPO_ROOT / "gui"))

from ptm import body_capabilities                                   # noqa: E402
from ptm.body_capabilities import (BODY_CAPABILITIES,                # noqa: E402
                                   IMPLEMENTED_BODIES, is_implemented)

#: Every implemented body must carry all of these. A parse that lost a field
#: would otherwise pass check 2 by agreeing with an equally incomplete table.
REQUIRED_FIELDS = (
    "radius_km", "solar_day_days", "year_days", "rAU", "S",
    "albedo", "emissivity", "Qb", "ks", "kd", "rhos", "rhod", "H", "cp0",
    "spice_metakernel",
)

#: Bodies the GUI's built-in dropdown offers that the engine does NOT implement.
#: Every one of these is a config the engine refuses at parse
#: (ENG-BODY-UNBACKED-RUN); they are pinned so a NEW unbacked body cannot be
#: added to the dropdown unnoticed, and so that removing them from the GUI
#: (Phase B of the body-first work) fails this test until the pin is updated
#: too. Do not add to this list to make the test pass.
GUI_UNBACKED_BODIES = ()

#: Non-body entries in the same dropdown: separators and actions, not names the
#: engine ever sees.
GUI_DROPDOWN_NON_BODIES = ("─" * 10, "Custom Body...", "Browse HORIZONS...")


def _derive():
    return body_capabilities.parse_body_capabilities(REPO_ROOT)


# --------------------------------------------------------------------------

def test_extractor_is_not_vacuous():
    """The C++ parse still finds a complete-looking body table."""
    bodies, capabilities, patterns = _derive()
    assert bodies, "no implemented bodies parsed out of planetary_properties.h"
    assert set(bodies) == set(capabilities), (
        f"parsed body list {bodies} and property rows {sorted(capabilities)} "
        f"disagree")
    for body in bodies:
        missing = [f for f in REQUIRED_FIELDS if f not in capabilities[body]]
        assert not missing, (
            f"{body}: the parse lost {missing}. The extractor has most likely "
            f"stopped recognizing the shape of get{body}Properties(), not the "
            f"engine lost the properties.")
        assert capabilities[body]["radius_km"] > 0, f"{body}: radius is not positive"
        assert capabilities[body]["solar_day_days"] > 0, f"{body}: day is not positive"
    fits, binary, resolutions, topo = patterns
    assert fits and binary and resolutions and topo, (
        "the map-discovery contract parsed empty; albedo_map_status() would "
        "then report 'nothing available' for every body")


def test_committed_table_matches_cxx_sources():
    """ptm/body_table.py still says what planetary_properties.h says."""
    bodies, capabilities, _patterns = _derive()

    assert tuple(bodies) == tuple(IMPLEMENTED_BODIES), (
        f"implemented bodies changed: engine says {list(bodies)}, the table "
        f"says {list(IMPLEMENTED_BODIES)}.\n"
        f"Regenerate with: python3 -m ptm.body_capabilities --regenerate")

    report = []
    for body in bodies:
        engine, table = capabilities[body], BODY_CAPABILITIES[body]
        for field in sorted(set(engine) | set(table)):
            if field not in table:
                report.append(f"{body}.{field}: engine has it, table does not")
            elif field not in engine:
                report.append(f"{body}.{field}: table has it, engine does not")
            elif engine[field] != table[field]:
                report.append(f"{body}.{field}: table {table[field]!r} != "
                              f"engine {engine[field]!r}")
    assert not report, (
        "ptm/body_table.py is out of date with planetary_properties.h.\n  "
        + "\n  ".join(report)
        + "\nRegenerate with: python3 -m ptm.body_capabilities --regenerate")


def test_generated_module_is_byte_current():
    """The committed file equals what --regenerate would write right now."""
    bodies, capabilities, patterns = _derive()
    rendered = body_capabilities._render_module(bodies, capabilities, patterns)
    target = Path(body_capabilities.__file__).with_name("body_table.py")
    assert target.read_text() == rendered, (
        f"{target} differs from a fresh render; run "
        f"python3 -m ptm.body_capabilities --regenerate")


def test_every_other_copy_of_these_numbers_agrees():
    """Four duplicate tables, one engine. They must not drift apart.

    These are real duplicates, not caches: each was typed from the engine at
    some point and nothing checked them afterwards. That is exactly how
    ``gui_tooltips.BODY_SYNODIC_PERIODS`` came to carry 175.942 d for Mercury
    under a comment claiming it matched ``thermo.day`` -- while the engine
    carried the SIDEREAL 58.646 d. Both numbers were wrong for what they
    claimed to be, and no test could see it.
    """
    from gui_tooltips import BODY_DEFAULTS, BODY_SYNODIC_PERIODS
    from gui_core.facet_scale import _BODY_RADII_M
    from ptm.results import BODY_RADII

    report = []
    for body in IMPLEMENTED_BODIES:
        engine = BODY_CAPABILITIES[body]

        if body not in BODY_SYNODIC_PERIODS:
            report.append(f"gui_tooltips.BODY_SYNODIC_PERIODS has no {body}")
        elif BODY_SYNODIC_PERIODS[body] != engine["solar_day_days"]:
            report.append(
                f"gui_tooltips.BODY_SYNODIC_PERIODS[{body!r}] = "
                f"{BODY_SYNODIC_PERIODS[body]} d, engine solar day = "
                f"{engine['solar_day_days']} d")

        for table_name, table in (("gui_core.facet_scale._BODY_RADII_M",
                                   _BODY_RADII_M),
                                  ("ptm.results.BODY_RADII", BODY_RADII)):
            if body not in table:
                report.append(f"{table_name} has no {body}")
            elif table[body] != engine["radius_km"] * 1000.0:
                report.append(
                    f"{table_name}[{body!r}] = {table[body]} m, engine radius = "
                    f"{engine['radius_km'] * 1000.0} m")

        if body not in BODY_DEFAULTS:
            report.append(f"gui_tooltips.BODY_DEFAULTS has no {body}")
            continue
        for field in ("albedo", "emissivity", "Qb", "ks", "kd", "rhos", "rhod",
                      "H", "cp0", "rAU"):
            gui = BODY_DEFAULTS[body].get(field)
            if gui != engine[field]:
                report.append(
                    f"gui_tooltips.BODY_DEFAULTS[{body!r}][{field!r}] = {gui!r}, "
                    f"engine = {engine[field]!r}")
        gui_cp = tuple(BODY_DEFAULTS[body].get("cpCoeff", ()))
        if gui_cp != tuple(engine.get("cpCoeff", ())):
            report.append(
                f"gui_tooltips.BODY_DEFAULTS[{body!r}]['cpCoeff'] = {list(gui_cp)}, "
                f"engine = {list(engine.get('cpCoeff', ()))}")

    assert not report, (
        "a Python or GUI copy of the engine's body properties has drifted:\n  "
        + "\n  ".join(report))


def test_gui_offers_no_unpinned_unbacked_body():
    """The dropdown may not grow a body the engine refuses, unnoticed.

    It used to be a hand-written list, and this test's job was to PIN the four
    names on it that the engine refuses so none could be added quietly. Phase B
    of the body-first work removed those four and replaced the list with
    ``list(IMPLEMENTED_BODIES)``, which is the generated table -- so the
    property is now structural: there is no literal for anyone to extend.

    That is a stronger guarantee than the pin, and this test now checks THAT,
    plus the runtime value where the GUI stack is importable. A future edit
    that reintroduces a literal list fails here on the day it is made.
    """
    source = (REPO_ROOT / "gui" / "gui_panels" / "config_panel.py").read_text()
    literals = re.findall(r"BUILTIN_BODIES\s*=\s*[\[{]([^\]}]*)[\]}]", source)
    offenders = []
    for block in literals:
        names = [item.strip().strip("\"'") for item in block.split(",")
                 if item.strip()]
        offenders.extend(n for n in names
                         if n not in GUI_DROPDOWN_NON_BODIES
                         and not is_implemented(n))
    assert tuple(offenders) == GUI_UNBACKED_BODIES, (
        "config_panel.py names built-in bodies the engine cannot run.\n"
        f"  now:      {offenders}\n"
        f"  expected: {list(GUI_UNBACKED_BODIES)}\n"
        "Every name here produces a FATAL at config parse. Derive the list "
        "from ptm.body_table.IMPLEMENTED_BODIES instead of writing one.")

    assert "BUILTIN_BODIES = list(_IMPLEMENTED)" in source \
        and "BUILTIN_BODIES = set(_IMPLEMENTED)" in source, (
        "the two BUILTIN_BODIES lists in config_panel.py (the Simulation "
        "dropdown and the Body page's is-this-built-in set) are no longer "
        "derived from the generated implemented-body table, so they can drift "
        "from the engine again")

    try:
        sys.path.insert(0, str(REPO_ROOT / "gui"))
        from gui_panels.config_panel import BodyDefinitionTab, SimulationTab
    except Exception as exc:                                 # noqa: BLE001
        print(f"    [gui] not importable here ({type(exc).__name__}); the "
              f"structural check above still ran")
        return
    for owner in (SimulationTab, BodyDefinitionTab):
        offered = [b for b in owner.BUILTIN_BODIES
                   if b not in GUI_DROPDOWN_NON_BODIES]
        unbacked = [b for b in offered if not is_implemented(b)]
        assert not unbacked, (
            f"{owner.__name__}.BUILTIN_BODIES offers {unbacked}, which the "
            f"engine refuses at config parse")
        for body in IMPLEMENTED_BODIES:
            assert body in offered, (
                f"{owner.__name__} no longer offers {body}, which the engine "
                f"implements")


def test_the_engine_refuses_unbacked_names():
    """The refusal exists in the engine, and names the route that works.

    Asserted against the source rather than a run so it holds in a checkout
    that has never been built; ``test_engine_validate_config.py`` asserts the
    same thing against the binary, with rc and message.
    """
    header = (REPO_ROOT / "src" / "cpp" / "thermal"
              / "planetary_properties.h").read_text()
    for name in ("isImplementedBody", "implementedBodiesList",
                 "kImplementedBodies"):
        assert name in header, f"planetary_properties.h no longer defines {name}"

    main = (REPO_ROOT / "src" / "cpp" / "main" / "ptm_thermal_sim.cpp").read_text()
    assert "thermal::isImplementedBody(sim_config.time.body" in main, (
        "ptm_thermal_sim no longer checks simulation.body against the "
        "implemented-body registry; an unbacked name would run as Europa again")
    assert "generic_body.isConfigured()" in main, (
        "the generic-body exemption is gone; every asteroid config would now "
        "be refused")


TESTS = [
    test_extractor_is_not_vacuous,
    test_committed_table_matches_cxx_sources,
    test_generated_module_is_byte_current,
    test_every_other_copy_of_these_numbers_agrees,
    test_gui_offers_no_unpinned_unbacked_body,
    test_the_engine_refuses_unbacked_names,
]


def main():
    failures = 0
    for test in TESTS:
        try:
            test()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {test.__name__}\n     {exc}")
        except Exception as exc:                            # noqa: BLE001
            failures += 1
            print(f"ERROR {test.__name__}: {type(exc).__name__}: {exc}")
        else:
            print(f"PASS {test.__name__}")
    total = len(TESTS)
    print(f"\n[test_body_capabilities] {total - failures}/{total} passed, "
          f"{len(IMPLEMENTED_BODIES)} implemented bodies checked "
          f"({', '.join(IMPLEMENTED_BODIES)})")
    if failures:
        print("[test_body_capabilities] FAIL")
        return 1
    print("[test_body_capabilities] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
