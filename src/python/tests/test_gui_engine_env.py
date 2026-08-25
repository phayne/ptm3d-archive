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


class Skipped(Exception):
    """A check that could not run here, as distinct from one that failed.

    Only the ConfigManager check raises it, and only for a missing PyQt6:
    ``gui_core`` pulls Qt in at package import, so without it that one check is
    unrunnable, not failing. Everything else here reaches ``preflight``
    directly, which imports Qt lazily.
    """


def _skip(reason):
    """Skip the current check. Uses pytest's own mechanism when under pytest."""
    try:
        import pytest
    except ImportError:
        raise Skipped(reason) from None
    pytest.skip(reason)


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


def test_a_config_can_decline_the_flags():
    """An ``engine_env:`` block is honored exactly, including an empty one.

    ``load_from_yaml`` used to merge the loaded file INTO a DEFAULT_CONFIG whose
    ``engine_env`` is the recommended set, so ``engine_env: {}`` -- the
    documented way to say "no fix flags" -- came back carrying all of them, and
    a block naming one variable silently gained the rest. There was no way to
    write "this run wants nothing."
    """
    env, applied = build_engine_env({"engine_env": {}}, base_env={})
    leaked = sorted(k for k in env if k.startswith("PTM_"))
    assert not leaked, (
        f"a config carrying an empty engine_env: block still got {leaked}; "
        f"'no fix flags' has to be expressible")
    assert applied == [], f"applied should be empty, got {applied}"

    # Declining has to survive a shell that already exports the variable.
    # Setting-only would leave the child running with it while this function
    # reported an empty environment -- the PTM-276 hole, from the other side.
    inherited = {"PTM_EQ_PERFACE_FLUX": "1", "PTM_ILLUM_RISE_EPS": "1",
                 "PATH": "/usr/bin"}
    env, applied = build_engine_env({"engine_env": {}}, base_env=inherited)
    leaked = sorted(k for k in env if k.startswith("PTM_"))
    assert not leaked, (
        f"an inherited {leaked} survived an engine_env: {{}} decline; the "
        f"launching shell must not be able to re-enable a flag the config "
        f"declined")
    assert env["PATH"] == "/usr/bin", (
        "the environment outside the managed variables must be left alone")
    assert any("removed" in line for line in applied), (
        f"the removal was not disclosed: applied={applied}")

    env, _ = build_engine_env(
        {"engine_env": {"PTM_ILLUM_TRUE_NORMALS": "0"}}, base_env={})
    assert env.get("PTM_ILLUM_TRUE_NORMALS") == "0"
    assert "PTM_ILLUM_RISE_EPS" not in env, (
        "a config that named one variable was silently given the rest of the "
        "recommended set; an explicit block must mean exactly what it says")

    # No block at all is "I did not think about this": apply the set.
    env, _ = build_engine_env({}, base_env={})
    assert env.get("PTM_ILLUM_RISE_EPS") == "1"

    # A blank value inside a block still means "do not set".
    env, _ = build_engine_env(
        {"engine_env": {"PTM_ILLUM_RISE_EPS": "1", "PTM_ILLUM_SEAL_HOLES": ""}},
        base_env={})
    assert env.get("PTM_ILLUM_RISE_EPS") == "1"
    assert "PTM_ILLUM_SEAL_HOLES" not in env


def test_declining_the_flags_is_reported_not_silent():
    """Honoring a reduced block still warns that it is reduced."""
    from gui_core.preflight import check_env_completeness, engine_env_plan

    env, _ = build_engine_env({"engine_env": {}}, base_env={})
    problems = check_env_completeness(env)
    assert problems, (
        "an engine_env: {} run got no completeness warning at all; honoring a "
        "block is not the same as endorsing it")

    _table, source = engine_env_plan({"engine_env": {}})
    assert "engine_env" in source, source
    _table, source = engine_env_plan({})
    assert "recommended" in source, source
    # A truncated `engine_env:` must not be read as a request for nothing.
    table, source = engine_env_plan({"engine_env": None})
    assert table == dict(RECOMMENDED_ENGINE_ENV), table
    assert "not a mapping" in source, source


def test_loading_a_config_does_not_re_add_the_flags():
    """ConfigManager keeps the file's own block instead of merging the default.

    Skipped, not failed, where PyQt6 is missing: ``gui_core`` imports Qt at
    package import even though ``preflight`` itself does not.
    """
    import tempfile

    try:
        from gui_core.config_manager import ConfigManager
    except ImportError as exc:
        _skip(f"the GUI's config manager is unavailable here ({exc}); "
              f"install PyQt6 to run this check")

    def loaded_env(text):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
            fh.write(text)
            path = fh.name
        manager = ConfigManager()
        manager.load_from_yaml(path)
        Path(path).unlink()
        return manager.get_value("engine_env")

    assert loaded_env("simulation:\n  body: Moon\nengine_env: {}\n") == {}, (
        "a config declaring an empty engine_env: block came back carrying the "
        "recommended set; there is then no way to write 'no fix flags'")

    one = loaded_env("simulation:\n  body: Moon\n"
                     "engine_env:\n  PTM_ILLUM_TRUE_NORMALS: '0'\n")
    assert one == {"PTM_ILLUM_TRUE_NORMALS": "0"}, (
        f"a block naming one variable silently gained the others: {one}")

    # No block at all keeps the recommended default.
    none = loaded_env("simulation:\n  body: Moon\n")
    assert none == dict(RECOMMENDED_ENGINE_ENV), none


def test_a_shell_export_cannot_re_enable_the_dropped_variable():
    """PTM-276 from the other side, in the default (no engine_env:) case.

    The GUI stopped injecting PTM_EQ_PERFACE_FLUX. That is worth nothing if a
    user who once exported it in their shell still hands it to every GUI run,
    where on the GPU path it overrides an explicit thermal.spinup_seed: legacy.
    """
    from gui_core.preflight import MANAGED_ENGINE_ENV

    env, applied = build_engine_env(
        {}, base_env={"PTM_EQ_PERFACE_FLUX": "1", "HOME": "/tmp"})
    assert "PTM_EQ_PERFACE_FLUX" not in env, (
        "an exported PTM_EQ_PERFACE_FLUX reached the engine even though the "
        "GUI deliberately does not inject it")
    assert env["HOME"] == "/tmp", "unmanaged variables must be untouched"
    assert any("PTM_EQ_PERFACE_FLUX" in line and "removed" in line
               for line in applied), applied

    for name in DELIBERATELY_NOT_INJECTED:
        assert name in MANAGED_ENGINE_ENV, (
            f"{name} is declared not-injected but is not managed, so an "
            f"inherited copy would still reach the engine")
    for name in RECOMMENDED_ENGINE_ENV:
        assert name in MANAGED_ENGINE_ENV, name


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
    test_a_config_can_decline_the_flags,
    test_declining_the_flags_is_reported_not_silent,
    test_loading_a_config_does_not_re_add_the_flags,
    test_a_shell_export_cannot_re_enable_the_dropped_variable,
    test_the_physics_change_is_announced,
]


def main():
    failures = 0
    skipped = 0
    for test in TESTS:
        try:
            test()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {test.__name__}\n     {exc}")
        except BaseException as exc:                   # noqa: BLE001
            # pytest.skip raises its own Skipped, which derives from
            # BaseException; match on the name and let anything else through.
            if type(exc).__name__ != "Skipped":
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                failures += 1
                print(f"ERROR {test.__name__}: {type(exc).__name__}: {exc}")
            else:
                skipped += 1
                print(f"SKIP {test.__name__}\n     {exc}")
        else:
            print(f"PASS {test.__name__}")
    total = len(TESTS)
    tally = f"{total - failures - skipped}/{total} passed"
    if skipped:
        tally += f", {skipped} skipped"
    print(f"\n[test_gui_engine_env] {tally}, "
          f"{len(RECOMMENDED_ENGINE_ENV)} variable(s) injected, "
          f"{len(DELIBERATELY_NOT_INJECTED)} deliberately omitted")
    if failures:
        print("[test_gui_engine_env] FAIL")
        return 1
    print("[test_gui_engine_env] PASS"
          + (f" ({skipped} skipped)" if skipped else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
