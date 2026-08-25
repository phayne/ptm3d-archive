#!/usr/bin/env python3
"""The environment the GUI injects vs the one docs/ENVIRONMENT.md validates.

Why this file exists: from 2026-07-29 to 2026-08-25 the GUI injected
``PTM_ILLUM_RISE_HABS=0`` and documented it as "kills 100% of leaks", while
never setting ``PTM_ILLUM_RISE_EPS=1`` -- the gate that makes ``RISE_HABS``
readable at all (``illumination_model.cpp:616`` and ``:1419``). The variable was
inert for a month and nothing said so, because nothing compared the GUI's table
against the documented set. These checks do that comparison.

Three things are asserted:

1. every variable the GUI injects is one ``docs/ENVIRONMENT.md`` lists as
   ``[recommended-ON]``, with the documented value -- a typo or a stale value
   fails here;
2. every documented variable the GUI does NOT inject appears in
   ``preflight.DELIBERATELY_NOT_INJECTED`` with a reason -- so an omission is a
   decision on the record, and a variable ADDED to the documentation later fails
   this check until someone decides about it;
3. ``PTM_ILLUM_RISE_HABS`` and ``PTM_ILLUM_RISE_KEPS`` are never injected
   without their gate ``PTM_ILLUM_RISE_EPS=1``. This is the exact 2026-08-25
   defect, pinned so it cannot recur silently.

Needs no PyQt6: ``gui_core.preflight`` imports Qt lazily.

Run directly::

    python3 src/python/tests/test_gui_engine_env.py

Also collectable by pytest, if it is installed.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "gui"))

from gui_core.preflight import (                                  # noqa: E402
    DELIBERATELY_NOT_INJECTED, ENV_SET_CHANGED_PHYSICS,
    RECOMMENDED_ENGINE_ENV, build_engine_env)

ENVIRONMENT_DOC = REPO_ROOT / "docs" / "ENVIRONMENT.md"

#: Variables that are only read when ``PTM_ILLUM_RISE_EPS`` is on. Setting one
#: of these without the gate is the 2026-08-25 defect: an injected variable the
#: engine never reads, next to a comment claiming it took effect.
RISE_EPS_GATED = ("PTM_ILLUM_RISE_KEPS", "PTM_ILLUM_RISE_HABS")


def documented_recommended_on():
    """Parse the ``[recommended-ON]`` set out of docs/ENVIRONMENT.md.

    The doc states it as one backtick-quoted, whitespace-separated run of
    ``NAME=VALUE`` pairs after "The validated set is:". Parsing it rather than
    copying it is the point -- a copy would be one more table to drift.
    """
    text = ENVIRONMENT_DOC.read_text()
    marker = "**The validated set is:**"
    assert marker in text, (
        f"{ENVIRONMENT_DOC} no longer contains {marker!r}; this test can no "
        f"longer find the recommended-ON set and would silently pass")
    tail = text.split(marker, 1)[1]
    # The set is the run of `PTM_...=...` tokens up to the sentence that ends it.
    body = tail.split("New science runs", 1)[0]
    pairs = re.findall(r"(PTM_[A-Z0-9_]+)=([^\s`]+)", body)
    assert pairs, f"no PTM_NAME=VALUE pairs found after {marker!r} in {ENVIRONMENT_DOC}"
    return dict(pairs)


# --------------------------------------------------------------------------

def test_injected_variables_match_the_documentation():
    """Everything the GUI injects is documented, with the documented value."""
    documented = documented_recommended_on()
    undocumented = sorted(set(RECOMMENDED_ENGINE_ENV) - set(documented))
    assert not undocumented, (
        f"the GUI injects variables docs/ENVIRONMENT.md does not list as "
        f"[recommended-ON]: {undocumented}")
    wrong = sorted(
        (k, RECOMMENDED_ENGINE_ENV[k], documented[k])
        for k in RECOMMENDED_ENGINE_ENV
        if str(RECOMMENDED_ENGINE_ENV[k]) != str(documented[k]))
    assert not wrong, (
        "the GUI injects a value the documentation does not validate "
        "(variable, GUI, doc): " + ", ".join(repr(w) for w in wrong))


def test_every_omission_is_on_the_record():
    """A documented variable the GUI skips must carry a written reason."""
    documented = documented_recommended_on()
    omitted = set(documented) - set(RECOMMENDED_ENGINE_ENV)
    unexplained = sorted(omitted - set(DELIBERATELY_NOT_INJECTED))
    assert not unexplained, (
        "docs/ENVIRONMENT.md marks these [recommended-ON] and the GUI neither "
        f"injects them nor records why: {unexplained}. Add them to "
        "RECOMMENDED_ENGINE_ENV, or to DELIBERATELY_NOT_INJECTED with a reason.")
    stale = sorted(set(DELIBERATELY_NOT_INJECTED) - omitted)
    assert not stale, (
        "these entries in DELIBERATELY_NOT_INJECTED are no longer omitted "
        f"documented variables and should be deleted: {stale}")
    for name, reason in DELIBERATELY_NOT_INJECTED.items():
        assert reason and len(reason) > 20, (
            f"DELIBERATELY_NOT_INJECTED[{name!r}] needs a real reason, not {reason!r}")


def test_rise_eps_gated_variables_are_never_orphaned():
    """No rise-epsilon parameter is injected without its gate.

    The gate is ``PTM_ILLUM_RISE_EPS=1``; ``use_rise_eps_`` is what makes
    ``traceOccludedRiseEps`` -- the only reader of ``rise_habs_`` and
    ``rise_keps_`` -- reachable. Checked on the table AND through
    ``build_engine_env``, because the table is not what the engine sees.
    """
    for probe, table in (("RECOMMENDED_ENGINE_ENV", dict(RECOMMENDED_ENGINE_ENV)),
                         ("build_engine_env(default config)",
                          build_engine_env({}, base_env={})[0])):
        present = [v for v in RISE_EPS_GATED if v in table]
        if not present:
            continue
        assert table.get("PTM_ILLUM_RISE_EPS") == "1", (
            f"{probe} sets {present} but not PTM_ILLUM_RISE_EPS=1. Those "
            f"variables are read only inside traceOccludedRiseEps, which is "
            f"reachable only under the gate -- so they would be INERT, exactly "
            f"the 2026-07-29..2026-08-25 defect.")


def test_the_physics_change_is_announced():
    """The banner names the variables and says temperatures move."""
    for token in ("PTM_ILLUM_RISE_EPS", "PTM_ILLUM_RISE_KEPS",
                  "PTM_EQ_PERFACE_FLUX", "2026-08-25"):
        assert token in ENV_SET_CHANGED_PHYSICS, (
            f"the change banner no longer mentions {token}")


TESTS = [
    test_injected_variables_match_the_documentation,
    test_every_omission_is_on_the_record,
    test_rise_eps_gated_variables_are_never_orphaned,
    test_the_physics_change_is_announced,
]


def main():
    failures = 0
    for test in TESTS:
        try:
            test()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {test.__name__}\n     {exc}")
        else:
            print(f"PASS {test.__name__}")
    total = len(TESTS)
    print(f"\n[test_gui_engine_env] {total - failures}/{total} passed, "
          f"{len(RECOMMENDED_ENGINE_ENV)} variable(s) injected, "
          f"{len(DELIBERATELY_NOT_INJECTED)} deliberately omitted")
    if failures:
        print("[test_gui_engine_env] FAIL")
        return 1
    print("[test_gui_engine_env] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
