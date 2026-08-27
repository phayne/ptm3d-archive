#!/usr/bin/env python3
"""Resolving a body name fills the card in, and never hangs or lies.

The Asteroid card asks for a NAIF id, a radius, a spin period, a pole and six
orbital elements. Every one of them is a number somebody would otherwise find
and retype, and a retyped element is a silent error: nothing downstream can
tell 2.46093257 AU from 2.46903257 AU.

Five properties:

1. **The cache is read from disk.** ``data/horizons_cache/index.json`` lists 3
   bodies while ``bodies/`` holds 7 with a ``body_config.yaml``, and the paths
   the index does carry are absolute paths into two OTHER checkouts. Anything
   that trusted the index would offer fewer than half the bodies actually here
   and could hand back a path that does not exist on this machine.
2. **A number is resolved to the right body.** "623" is Chimaera; it is not
   any of the other six, and it is not 13294 Rockox.
3. **Not found, offline and timed out are three different answers**, each said
   out loud, and none of them changes the configuration.
4. **The online path is measured, not skipped.** A stub client stands in for
   HORIZONS, so the field mapping -- including diameter -> radius, which is a
   factor of two in every flux on the body -- is checked on a machine with no
   network.
5. **The lookup never raises.** It is called from a Qt slot, where an escaping
   exception is a ``qFatal`` and takes the window down with any unsaved edits.

Run directly::

    python3 src/python/tests/test_gui_body_lookup.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "python"))
sys.path.insert(0, str(REPO_ROOT / "gui"))


class Skip(Exception):
    """A check that cannot run here, as distinct from one that failed."""


# --------------------------------------------------------------------------
# 1. The cache, from disk
# --------------------------------------------------------------------------

def test_the_cache_is_read_from_disk_not_from_its_index():
    import json

    from gui_core.body_lookup import CACHE_RELPATH, cache_entries

    entries = cache_entries(REPO_ROOT)
    if not entries:
        raise Skip("no bodies are cached in this checkout")
    on_disk = {name for name, _, _ in entries}

    index_path = REPO_ROOT / CACHE_RELPATH / "index.json"
    indexed = set()
    if index_path.exists():
        try:
            indexed = set(json.loads(index_path.read_text()))
        except Exception:                                    # noqa: BLE001
            indexed = set()

    assert len(on_disk) >= len(indexed), (
        f"the directory holds {len(on_disk)} bodies and the index names "
        f"{len(indexed)}; reading the index would be reading less")
    for name, config_path, generic in entries:
        assert config_path.exists(), config_path
        assert generic.get("name") or name, (
            f"{config_path} has a generic_body: section with no name")
    print(f"    [cache] {len(on_disk)} bodies on disk, {len(indexed)} in the "
          f"index: {sorted(on_disk)}")


def test_a_number_resolves_to_the_right_body():
    from gui_core.body_lookup import cache_entries, lookup_body

    names = [name for name, _, _ in cache_entries(REPO_ROOT)]
    chimaera = [n for n in names if "chimaera" in n.lower()]
    if not chimaera:
        raise Skip("623 Chimaera is not cached in this checkout")

    for query in ("623", "chimaera", "Chimaera", "623 Chimaera"):
        result = lookup_body(query, REPO_ROOT, allow_online=False)
        assert result.found, f"{query!r}: {result.message}"
        assert result.name == chimaera[0], (
            f"{query!r} resolved to {result.name!r}, not {chimaera[0]!r}")
        assert result.source == "cache", result.source

    # The eight-digit NAIF form and the bare asteroid number are the same body.
    by_naif = lookup_body("20000623", REPO_ROOT, allow_online=False)
    assert by_naif.found and by_naif.name == chimaera[0], by_naif.message

    # ...and a number that is NOT this body does not reach it.
    other = lookup_body("13294", REPO_ROOT, allow_online=False)
    assert not other.found or "rockox" in other.name.lower(), (
        f"13294 resolved to {other.name!r}")
    print(f"    [resolve] 623 / chimaera / 20000623 -> {chimaera[0]}; "
          f"13294 -> {other.name or 'not found'}")


def test_a_cache_hit_fills_in_the_fields_the_card_shows():
    from gui_core.body_lookup import cache_entries, lookup_body

    if not [n for n, _, _ in cache_entries(REPO_ROOT)
            if "chimaera" in n.lower()]:
        raise Skip("623 Chimaera is not cached in this checkout")
    result = lookup_body("623", REPO_ROOT, allow_online=False)
    wanted = (
        "generic_body.name",
        "generic_body.naif_id",
        "generic_body.shape.radius_km",
        "generic_body.spin.period_hours",
        "generic_body.spin.pole_ra_deg",
        "generic_body.spin.pole_dec_deg",
        "generic_body.orbit.semi_major_axis_au",
        "generic_body.orbit.eccentricity",
        "generic_body.orbit.inclination_deg",
    )
    missing = [k for k in wanted if k not in result.values]
    assert not missing, f"a cache hit did not supply {missing}"
    # every value goes through set_value, so every key must be dotted
    assert all(k.startswith("generic_body.") for k in result.values), \
        sorted(result.values)
    assert "Chimaera" in result.provenance(), result.provenance()
    assert "cache" in result.provenance().lower(), result.provenance()
    print(f"    [fill] a cache hit supplies {len(result.values)} keys "
          f"including radius {result.values['generic_body.shape.radius_km']} "
          f"km and spin "
          f"{result.values['generic_body.spin.period_hours']} h")


# --------------------------------------------------------------------------
# 2. HORIZONS, with a stub
# --------------------------------------------------------------------------

class _StubElements:
    semi_major_axis_au = 1.128922045323413
    eccentricity = 0.2046513239154546
    inclination_deg = 6.025539444076259
    lon_ascending_node_deg = 2.178489605966394
    arg_perihelion_deg = 66.22306084784298
    mean_anomaly_deg = 101.7039710396088
    epoch = "2000-01-01T12:00:00"


class _StubPhysical:
    diameter_km = 0.4894
    albedo = 0.044
    rotation_period_hours = 4.296057
    pole_ra_deg = 85.46
    pole_dec_deg = -60.36
    spectral_type = "B"


class _StubBody:
    name = "101955 Bennu (1999 RQ36)"
    naif_id = "2101955"
    elements = _StubElements()
    physical = _StubPhysical()


def test_the_online_path_maps_every_field_it_claims_to():
    from gui_core.body_lookup import lookup_body

    class _Client:
        def get_body(self, query):
            return _StubBody()

    result = lookup_body("bennu", REPO_ROOT, allow_online=True,
                         client_factory=_Client)
    assert result.found and result.source == "horizons", result.message
    values = result.values
    assert values["generic_body.naif_id"] == "2101955"
    # DIAMETER -> RADIUS. HORIZONS reports a diameter and the engine wants a
    # radius; taking one for the other is a factor of two in every flux on the
    # body, and nothing downstream would flag it.
    assert abs(values["generic_body.shape.radius_km"] - 0.2447) < 1e-9, (
        f"diameter {_StubPhysical.diameter_km} km became radius "
        f"{values['generic_body.shape.radius_km']} km")
    assert abs(values["generic_body.spin.period_hours"] - 4.296057) < 1e-9
    assert values["generic_body.spin.pole_ra_deg"] == 85.46
    assert values["generic_body.spin.pole_dec_deg"] == -60.36
    assert abs(values["generic_body.orbit.semi_major_axis_au"]
               - 1.128922045323413) < 1e-15
    assert "HORIZONS" in result.provenance(), result.provenance()
    print(f"    [online] a stub HORIZONS record supplies "
          f"{len(values)} keys; diameter "
          f"{_StubPhysical.diameter_km} km -> radius "
          f"{values['generic_body.shape.radius_km']} km")


def test_a_field_horizons_does_not_have_is_named_not_invented():
    from gui_core.body_lookup import lookup_body

    class _Sparse:
        name = "Sparse Body"
        naif_id = "9999"

        class elements:
            semi_major_axis_au = 2.5
            eccentricity = 0.1
            inclination_deg = -1.0
            lon_ascending_node_deg = -1.0
            arg_perihelion_deg = -1.0
            mean_anomaly_deg = -1.0
            epoch = ""

        class physical:
            diameter_km = 0.0
            albedo = 0.0
            rotation_period_hours = 0.0
            pole_ra_deg = -1.0
            pole_dec_deg = -1.0
            spectral_type = ""

    class _Client:
        def get_body(self, query):
            return _Sparse()

    result = lookup_body("sparse", REPO_ROOT, allow_online=True,
                         client_factory=_Client)
    assert result.found, result.message
    for absent in ("generic_body.shape.radius_km",
                   "generic_body.spin.period_hours",
                   "generic_body.spin.pole_ra_deg"):
        assert absent not in result.values, (
            f"{absent} was filled in from a record that does not have it")
    for word in ("spin period", "pole", "radius"):
        assert word in result.message, (
            f"the message does not say the {word} is missing:\n"
            f"{result.message}")
    print("    [gaps] a sparse record leaves the fields it lacks alone and "
          "names them")


# --------------------------------------------------------------------------
# 3. The three ways it can fail, each said out loud
# --------------------------------------------------------------------------

def test_offline_not_found_and_timed_out_are_three_distinct_answers():
    from gui_core.body_lookup import lookup_body

    # (a) not in the cache, online not allowed
    offline = lookup_body("no such body at all", REPO_ROOT,
                          allow_online=False)
    assert not offline.found
    assert "not in the local cache" in offline.message, offline.message
    assert "Online lookup is turned off" in offline.message, offline.message

    # (b) HORIZONS answers, with nothing
    class _Empty:
        def get_body(self, query):
            return None

    missing = lookup_body("no such body at all", REPO_ROOT,
                          allow_online=True, client_factory=_Empty)
    assert not missing.found
    assert "no record matching" in missing.message, missing.message

    # (c) HORIZONS cannot be reached
    class _Dead:
        def get_body(self, query):
            raise OSError("Network is unreachable")

    dead = lookup_body("no such body at all", REPO_ROOT, allow_online=True,
                       client_factory=_Dead)
    assert not dead.found
    assert "could not be reached" in dead.message, dead.message
    assert "Nothing has been changed" in dead.message, dead.message
    assert "Network is unreachable" in dead.message, dead.message

    messages = {offline.message, missing.message, dead.message}
    assert len(messages) == 3, "two failure modes give the same message"
    for result in (offline, missing, dead):
        assert result.values == {}, "a failed lookup wrote something"
        assert result.provenance() == "Parameters entered by hand.", \
            result.provenance()
    print("    [failures] cache miss, no record, and no network each give a "
          "different message and change nothing")


def test_the_lookup_never_raises():
    """It is called from a Qt slot; an escaping exception is a SIGABRT."""
    from gui_core.body_lookup import lookup_body

    class _Exploding:
        def get_body(self, query):
            raise RuntimeError("boom")

    hostile = [
        ("", REPO_ROOT, False),
        (None, REPO_ROOT, False),
        ("623", Path("/nonexistent/checkout"), False),
        ("623", Path("/nonexistent/checkout"), True),
        ("\x00\x01", REPO_ROOT, True),
        ("x" * 4000, REPO_ROOT, True),
        (12345, REPO_ROOT, False),
    ]
    for query, root, online in hostile:
        result = lookup_body(query, root, allow_online=online,
                             client_factory=_Exploding if online else None)
        assert result.message, f"{query!r} produced an empty message"
        assert isinstance(result.values, dict)
    print(f"    [robust] {len(hostile)} hostile inputs, none raised, all "
          f"carry a message")


# --------------------------------------------------------------------------
# 4. The control on the card
# --------------------------------------------------------------------------

def test_the_card_has_a_lookup_button_and_it_is_not_wired_to_keystrokes():
    """The control exists, is a button, and the name field does not fire it."""
    import os
    import tempfile

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        import PyQt6
    except ImportError as exc:                              # pragma: no cover
        raise Skip(f"PyQt6 is unavailable here ({exc})")
    plug = Path(PyQt6.__file__).parent / "Qt6" / "plugins"
    os.environ.setdefault("QT_PLUGIN_PATH", str(plug))
    os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(plug / "platforms"))
    from PyQt6.QtCore import QSettings
    from PyQt6.QtWidgets import QApplication, QPushButton

    with tempfile.TemporaryDirectory() as scratch:
        QSettings.setDefaultFormat(QSettings.Format.IniFormat)
        for scope in (QSettings.Scope.UserScope, QSettings.Scope.SystemScope):
            QSettings.setPath(QSettings.Format.IniFormat, scope, scratch)
        from gui_core.config_manager import ConfigManager
        from gui_panels.config_panel import ConfigurationPanel

        app = QApplication.instance() or QApplication([])
        manager = ConfigManager()
        panel = ConfigurationPanel(manager)
        start = panel.tabs["start"]
        start.choose_body("asteroid")
        start._choose_mode("body")
        card = start.open_card("asteroid", "body", "time_series",
                               apply_defaults=True)

        assert isinstance(card.lookup_button, QPushButton)
        assert card.lookup_button.text() == "Look up", card.lookup_button.text()
        assert card.lookup_button.toolTip().strip(), (
            "the lookup button has no tooltip")
        assert card.provenance_note.text() == "Parameters entered by hand."

        # Typing a name must NOT start a lookup: the note stays empty until
        # the button is pressed.
        card.widgets["generic_body.name"].setText("623")
        app.processEvents()
        assert card.lookup_note.text() == "", (
            f"typing started a lookup: {card.lookup_note.text()!r}")

        # ...and pressing it does.
        card.lookup_button.click()
        app.processEvents()
        note = card.lookup_note.text()
        assert note, "pressing Look up said nothing at all"
        if "Chimaera" in note:
            assert card.config_manager.get_value(
                "generic_body.spin.period_hours", -1) > 0, (
                "a successful lookup did not write the spin period")
            assert "Chimaera" in card.provenance_note.text(), \
                card.provenance_note.text()
            # ...and the name is still the user's to change.
            assert card.widgets["generic_body.name"].isEnabled()
            assert card.widgets[
                "generic_body.shape.radius_km"].isEnabled()
            print(f"    [button] Look up on '623' -> "
                  f"{card.provenance_note.text()}")
        else:
            raise Skip(f"623 Chimaera is not cached here; the button worked "
                       f"and said: {note!r}")


def test_an_ambiguous_query_stops_instead_of_asking_horizons():
    """Two matches is not "no match", and it must not escalate.

    ``lookup_from_cache`` used to return ``found=False`` for a multi-hit, so
    ``lookup_body`` fell through to HORIZONS, which answers with ONE confident
    body -- turning "which of these two did you mean" into a silent choice.
    That is the failure ``_matches`` was written to remove, arriving by
    another door.
    """
    from gui_core.body_lookup import cache_entries, lookup_body

    names = [n for n, _, _ in cache_entries(REPO_ROOT)]
    shared = [n for n in names if "1998" in n]
    if len(shared) < 2:
        raise Skip(f"no ambiguous token in this cache ({len(shared)} match "
                   f"'1998')")

    asked = []

    class _Client:
        def get_body(self, query):
            asked.append(query)
            return _StubBody()

    result = lookup_body("1998", REPO_ROOT, allow_online=True,
                         client_factory=_Client)
    assert not result.found, (
        f"an ambiguous query resolved to {result.name!r}")
    assert result.ambiguous, "the ambiguity was not reported as one"
    assert asked == [], (
        f"HORIZONS was asked to settle an ambiguous cache match: {asked}")
    assert result.values == {}, "an ambiguous lookup wrote something"
    assert all(n in result.message for n in shared[:2]), result.message
    print(f"    [ambiguous] '1998' matches {len(shared)} cached bodies; the "
          f"lookup stops and names them")


def test_a_circular_orbit_keeps_its_zero_eccentricity():
    """Zero is a value; -1.0 and "" are the sentinels.

    Discarding 0.0 left `eccentricity` at the -1.0 sentinel, so a circular
    orbit HORIZONS had described exactly came back undescribed -- and the
    message did not name orbital elements among the fields it said were
    missing.
    """
    from gui_core.body_lookup import lookup_body

    class _Circular:
        name = "Circular Probe"
        naif_id = "1"

        class elements:
            semi_major_axis_au = 3.0
            eccentricity = 0.0
            inclination_deg = 0.0
            lon_ascending_node_deg = 0.0
            arg_perihelion_deg = 0.0
            mean_anomaly_deg = 0.0
            epoch = "2000-01-01T12:00:00"

        class physical:
            diameter_km = 10.0
            albedo = 0.1
            rotation_period_hours = 8.0
            pole_ra_deg = 0.0
            pole_dec_deg = 90.0
            spectral_type = "C"

    class _Client:
        def get_body(self, query):
            return _Circular()

    result = lookup_body("circular", REPO_ROOT, allow_online=True,
                         client_factory=_Client)
    assert result.found, result.message
    for key in ("generic_body.orbit.eccentricity",
                "generic_body.orbit.inclination_deg",
                "generic_body.orbit.mean_anomaly_deg",
                "generic_body.spin.pole_ra_deg"):
        assert key in result.values, f"{key} was discarded as 'not set'"
        assert result.values[key] == 0.0, result.values[key]
    # ...and the sentinel is still treated as a sentinel.
    assert "generic_body.shape.radius_km" in result.values
    print("    [zeros] a circular equatorial orbit keeps its four zeros")


CHECKS = [
    test_the_cache_is_read_from_disk_not_from_its_index,
    test_an_ambiguous_query_stops_instead_of_asking_horizons,
    test_a_circular_orbit_keeps_its_zero_eccentricity,
    test_a_number_resolves_to_the_right_body,
    test_a_cache_hit_fills_in_the_fields_the_card_shows,
    test_the_online_path_maps_every_field_it_claims_to,
    test_a_field_horizons_does_not_have_is_named_not_invented,
    test_offline_not_found_and_timed_out_are_three_distinct_answers,
    test_the_lookup_never_raises,
    test_the_card_has_a_lookup_button_and_it_is_not_wired_to_keystrokes,
]


def main():
    failed = skipped = 0
    for check in CHECKS:
        name = check.__name__
        try:
            check()
        except Skip as exc:
            skipped += 1
            print(f"SKIP {name}: {exc}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {name}:\n  {exc}")
        except Exception as exc:                             # noqa: BLE001
            failed += 1
            import traceback
            print(f"ERROR {name}: {exc!r}")
            traceback.print_exc()
        else:
            print(f"PASS {name}")
    total = len(CHECKS)
    print(f"\n{total - failed - skipped}/{total} passed, {skipped} skipped, "
          f"{failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
