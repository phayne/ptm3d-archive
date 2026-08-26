#!/usr/bin/env python3
"""The GUI's defaults are the engine's defaults, except where it says otherwise.

``ConfigManager.DEFAULT_CONFIG`` was hand-maintained, and it drifted: the engine
raised ``radiation.max_coupling_iterations`` from 10 to 16 when the relative
convergence criterion landed and the GUI went on stamping 10 into every config
it saved. ``radiation.use_gpu`` was ``true`` in the GUI and ``false`` in the
engine, injected into configs whose author never mentioned radiation. Neither
produced an error anywhere -- the engine has no required key and rejects no
unknown one, so a stale default is just a different run than the one intended.

The values now come from ``ptm.engine_defaults``, which is generated from
``config_parser.{h,cpp}`` and pinned by ``test_config_parity.py``. What is left
to check here is the exception list:

1. every key in ``GUI_DEFAULT_OVERRIDES`` is a key the engine actually reads
   (a typo would silently never apply);
2. every one of them really does differ from the engine -- an override that has
   caught up with the engine is dead weight that hides the next real one;
3. every key the GUI and the engine disagree about is on the list. This is the
   drift guard: a new divergence fails here.

Needs PyQt6 (``gui_core`` imports Qt at package import); skipped without it.

Run directly::

    python3 src/python/tests/test_gui_defaults_follow_engine.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "python"))
sys.path.insert(0, str(REPO_ROOT / "gui"))

from ptm.engine_defaults import ENGINE_DEFAULTS                   # noqa: E402


class Skipped(Exception):
    """A check that cannot run here, as distinct from one that failed."""


def _skip(reason):
    try:
        import pytest
    except ImportError:
        raise Skipped(reason) from None
    pytest.skip(reason)


def _manager_class():
    try:
        from gui_core.config_manager import ConfigManager
    except ImportError as exc:
        _skip(f"the GUI's config manager is unavailable here ({exc}); "
              f"install PyQt6 to run this check")
    return ConfigManager


def _flat(d, prefix=""):
    out = {}
    for key, value in (d or {}).items():
        dotted = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(_flat(value, dotted + "."))
        else:
            out[dotted] = value
    return out


# --------------------------------------------------------------------------

def test_every_override_names_a_real_engine_key():
    manager = _manager_class()
    unknown = sorted(k for k in manager.GUI_DEFAULT_OVERRIDES
                     if k not in ENGINE_DEFAULTS)
    assert not unknown, (
        f"GUI_DEFAULT_OVERRIDES names key(s) the engine does not read, so the "
        f"override can never apply: {unknown}")


def test_every_override_carries_a_reason():
    manager = _manager_class()
    thin = sorted(k for k, why in manager.GUI_DEFAULT_OVERRIDES.items()
                  if not why or len(why) < 30)
    assert not thin, (
        f"these overrides have no real justification written down: {thin}. "
        f"A divergence from the engine is a decision; write the decision.")


def test_no_override_has_caught_up_with_the_engine():
    manager = _manager_class()
    gui = _flat(manager.DEFAULT_CONFIG)
    stale = sorted(k for k in manager.GUI_DEFAULT_OVERRIDES
                   if k in gui and gui[k] == ENGINE_DEFAULTS.get(k))
    assert not stale, (
        f"these entries in GUI_DEFAULT_OVERRIDES no longer differ from the "
        f"engine and should be deleted: {stale}")


def test_every_divergence_is_declared():
    """The drift guard: a difference the list does not know about fails."""
    manager = _manager_class()
    gui = _flat(manager.DEFAULT_CONFIG)
    undeclared = sorted(
        (k, gui[k], ENGINE_DEFAULTS[k])
        for k in set(gui) & set(ENGINE_DEFAULTS)
        if k not in manager.GUI_DEFAULT_OVERRIDES
        and ENGINE_DEFAULTS[k] is not None
        and (gui[k] != ENGINE_DEFAULTS[k]
             or type(gui[k]) is not type(ENGINE_DEFAULTS[k])))
    assert not undeclared, (
        "the GUI default differs from the engine's for (key, gui, engine): "
        + ", ".join(repr(u) for u in undeclared)
        + ". Either take the engine's value, or add the key to "
          "GUI_DEFAULT_OVERRIDES with the reason.")


def test_the_two_2026_08_defects_stay_fixed():
    """Named explicitly, because these two are what the mechanism is for."""
    manager = _manager_class()
    gui = _flat(manager.DEFAULT_CONFIG)
    assert gui["radiation.max_coupling_iterations"] == 16, (
        f"radiation.max_coupling_iterations is "
        f"{gui['radiation.max_coupling_iterations']}, not the engine's 16")
    assert gui["radiation.use_gpu"] is False, (
        f"radiation.use_gpu is {gui['radiation.use_gpu']!r}; the GUI should "
        f"not inject GPU radiosity into a config whose author never mentioned "
        f"radiation")


TESTS = [
    test_every_override_names_a_real_engine_key,
    test_every_override_carries_a_reason,
    test_no_override_has_caught_up_with_the_engine,
    test_every_divergence_is_declared,
    test_the_two_2026_08_defects_stay_fixed,
]


def main():
    failures = skipped = 0
    for test in TESTS:
        try:
            test()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {test.__name__}\n     {exc}")
        except BaseException as exc:                   # noqa: BLE001
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
    print(f"\n[test_gui_defaults_follow_engine] {tally}")
    if failures:
        print("[test_gui_defaults_follow_engine] FAIL")
        return 1
    print("[test_gui_defaults_follow_engine] PASS"
          + (f" ({skipped} skipped)" if skipped else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
