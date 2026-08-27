#!/usr/bin/env python3
"""Auto-populating the Asteroid card converts what it fetches, and refuses.

The card can now fill itself in from three services. Each of them publishes at
least one quantity in a definition that is NOT the engine's, and every one of
those is a silent error -- the run completes, the output looks ordinary, and
the number is wrong:

1. **SBDB's ``albedo`` is the GEOMETRIC albedo p_V.**
   ``generic_body.surface.albedo`` is a BOND albedo: ``thermal_model.cpp:2906``
   computes ``1 - albedo`` and multiplies the solar constant by it. For a
   bright body, feeding p_V straight in absorbs a quarter less sunlight than
   the body does. The conversion ``A_B = p_V (0.290 + 0.684 G)`` is checked
   here against a case computed by hand.

2. **Both published spin poles are ECLIPTIC.** SBDB's ``pole`` parameter and
   DAMIT's ``spin.txt`` give ``(lambda, beta)``; ``generic_body.spin`` takes
   equatorial RA/Dec. Straight across is wrong by up to 23.4 degrees, and a
   spin pole has no plausibility check that would catch it.

3. **An SPK that does not span the run is not an error the engine reports.**
   It substitutes Keplerian elements per epoch and logs nothing (PTM-145: the
   Justitia kernel covered 30 days of 2034 while the configurations targeted
   2035, and all fifteen "passed"). The negative control below is the arm that
   matters: a kernel that arrives and does NOT cover the run must leave the
   ephemeris source alone.

Plus the properties every fetch on this card has to have: cache-first with no
network, distinct failure messages, and no writing into anybody's
configuration -- these functions RETURN values, they do not apply them, which
is what keeps a fetch on one card out of another card's run.

Run directly::

    python3 src/python/tests/test_gui_body_autopopulate.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "python"))
sys.path.insert(0, str(REPO_ROOT / "gui"))


class Skip(Exception):
    """A check that cannot run here, as distinct from one that failed."""


# --------------------------------------------------------------------------
# 1. The conversion, against a case computed by hand
# --------------------------------------------------------------------------

def test_geometric_albedo_becomes_bond_albedo():
    """p_V is not a Bond albedo, and the difference is most of a bright body.

    Hand computation, Vesta-like, p_V = 0.42 and G = 0.32:

        q   = 0.290 + 0.684 x 0.32 = 0.290 + 0.21888 = 0.50888
        A_B = 0.42 x 0.50888                         = 0.2137296

    and the cost of skipping it:

        1 - A: 0.58 (p_V used as Bond) against 0.7862704 (correct)
        absorbed flux low by 1 - 0.58/0.7862704 = 26.24 percent
        temperature   low by 1 - (0.58/0.7862704)^(1/4) = 7.32 percent
    """
    from gui_core.body_properties import bond_albedo, phase_integral

    q = phase_integral(0.32)
    assert abs(q - 0.50888) < 1e-12, f"q(G=0.32) = {q}, hand value 0.50888"

    conversion = bond_albedo(0.42, 0.32)
    assert conversion is not None
    assert abs(conversion.q - 0.50888) < 1e-12
    assert abs(conversion.bond - 0.2137296) < 1e-12, (
        f"A_B = {conversion.bond}, hand value 0.2137296")
    assert not conversion.slope_assumed

    # The error this removes, stated in the units it is felt in.
    wrong, right = 1.0 - 0.42, 1.0 - conversion.bond
    flux_deficit = 1.0 - wrong / right
    temp_deficit = 1.0 - (wrong / right) ** 0.25
    assert abs(flux_deficit - 0.2624) < 1e-3, flux_deficit
    assert abs(temp_deficit - 0.0732) < 1e-3, temp_deficit

    # The label carries BOTH inputs, so a reader can redo the arithmetic.
    label = conversion.label()
    assert "p_V=0.42" in label and "G=0.32" in label, label
    assert "Bond albedo" in label, label
    assert "0.290 + 0.684" in conversion.detail(), conversion.detail()

    print(f"    [albedo] p_V=0.42, G=0.32 -> q={conversion.q:.5f}, "
          f"A_B={conversion.bond:.7f}; using p_V directly would absorb "
          f"{100 * flux_deficit:.1f}% less flux and run "
          f"{100 * temp_deficit:.1f}% cold")


def test_a_missing_slope_parameter_is_assumed_and_labeled():
    """No published G means the IAU default, and the label says 'assumed'."""
    from gui_core.body_properties import DEFAULT_SLOPE_PARAMETER, bond_albedo

    conversion = bond_albedo(0.037, None)
    assert conversion.slope_assumed
    assert conversion.slope_parameter == DEFAULT_SLOPE_PARAMETER
    # q(0.15) = 0.290 + 0.684 x 0.15 = 0.3926; 0.037 x 0.3926 = 0.0145262
    assert abs(conversion.bond - 0.0145262) < 1e-12, conversion.bond
    assert "assumed" in conversion.label(), conversion.label()

    # A FITTED G of zero is a real value and must not read as "absent".
    fitted_zero = bond_albedo(0.2, 0.0)
    assert not fitted_zero.slope_assumed, (
        "a fitted G of 0.0 was treated as 'not published'")
    assert abs(fitted_zero.bond - 0.058) < 1e-12, fitted_zero.bond

    # No p_V at all: nothing is invented.
    for absent in (0.0, -1.0, None, "", "n/a"):
        assert bond_albedo(absent) is None, (
            f"a Bond albedo was invented from p_V = {absent!r}")

    # An out-of-range G is USED, not clamped, and is declared.
    extrapolated = bond_albedo(0.3, 0.9)
    assert abs(extrapolated.slope_parameter - 0.9) < 1e-12, (
        "G was silently clamped into the fitted range")
    assert any("outside" in w for w in extrapolated.warnings), \
        extrapolated.warnings

    print("    [albedo] absent G is assumed and labeled; a fitted G=0 is kept; "
          "an out-of-range G is used and declared, not clamped")


def test_the_published_spin_pole_is_ecliptic_and_gets_converted():
    """Three poles whose equatorial answers are known independently."""
    from gui_core.body_properties import ecliptic_pole_to_equatorial

    # The north ecliptic pole is at RA 18h = 270 deg, Dec +66 deg 34'.
    ra, dec = ecliptic_pole_to_equatorial(0.0, 90.0)
    assert abs(ra - 270.0) < 1e-6, ra
    assert abs(dec - 66.5607) < 1e-3, dec

    # The vernal equinox is the shared origin of both frames.
    ra, dec = ecliptic_pole_to_equatorial(0.0, 0.0)
    assert abs(ra) < 1e-9 and abs(dec) < 1e-9, (ra, dec)

    # Ecliptic longitude 90 deg on the ecliptic is the summer solstice point:
    # RA 6h = 90 deg, Dec = +obliquity.
    ra, dec = ecliptic_pole_to_equatorial(90.0, 0.0)
    assert abs(ra - 90.0) < 1e-6, ra
    assert abs(dec - 23.439279) < 1e-4, dec

    # The straight-across error this prevents, at its worst.
    straight = abs(90.0 - 66.5607)
    assert straight > 23.0
    print(f"    [pole] NEP, equinox and solstice all land where they should; "
          f"copying ecliptic latitude into declination would be "
          f"{straight:.1f} deg wrong at the pole")


# --------------------------------------------------------------------------
# 2. SBDB: the field mapping, measured against a stub
# --------------------------------------------------------------------------

class _StubPhysical:
    """SBDB's ACTUAL record for (4) Vesta, in SBDB's own definitions.

    These are the values the live service returned on 2026-08-27, not invented
    ones. That matters: the first version of this stub encoded the ASSUMPTION
    that SBDB's ``pole`` is ecliptic, so the test agreed with the code and both
    were wrong -- a stub that carries the same belief as the code under test
    proves only that the belief is consistently held. The live check is what
    caught it, and the fix is to make the stub a transcript.
    """
    name = "4 Vesta (A807 FA)"
    diameter_km = 522.77
    albedo = 0.4228               # GEOMETRIC p_V ("geometric albedo", SBDB's
                                  # own title for the field)
    rotation_period_hours = 5.3421276322
    spectral_type = "V"
    gm_km3_s2 = 17.28
    density_g_cm3 = 3.456
    absolute_magnitude_h = 3.25
    slope_parameter_g = 0.32
    # EQUATORIAL J2000, as published. IAU WGCCRE gives 309.031 / 42.235.
    pole_ra_deg = 309.06110
    pole_dec_deg = 42.232386
    pole_ecliptic_lon_deg = -999.0   # SBDB publishes no ecliptic pole
    pole_ecliptic_lat_deg = -999.0
    albedo_is_geometric = True


def test_sbdb_maps_every_field_it_claims_to():
    from gui_core.body_fetch import fetch_sbdb_properties

    stub = _StubPhysical()

    class _Client:
        def get_physical_params(self, query):
            return stub

    result = fetch_sbdb_properties("vesta", client_factory=_Client)
    assert result.ok, result.message
    values = result.values

    # DIAMETER -> RADIUS. A factor of two in every flux on the body.
    assert abs(values["generic_body.shape.radius_km"] - 261.385) < 1e-9, (
        f"diameter {stub.diameter_km} km became radius "
        f"{values['generic_body.shape.radius_km']} km")

    # p_V -> BOND. NOT the published number.
    # 0.4228 x (0.290 + 0.684 x 0.32) = 0.4228 x 0.50888 = 0.21515...
    bond = values["generic_body.surface.albedo"]
    assert abs(bond - 0.215154) < 1e-6, bond
    assert abs(bond - stub.albedo) > 0.2, (
        "SBDB's geometric albedo was written into the Bond albedo field")

    # The pole is EQUATORIAL as published and must NOT be converted. Checked
    # against the IAU WGCCRE pole for Vesta (309.031, 42.235), which is an
    # independent source rather than a restatement of the input.
    assert abs(values["generic_body.spin.pole_ra_deg"] - 309.061) < 0.001, (
        f"the pole RA came out {values['generic_body.spin.pole_ra_deg']}; "
        f"SBDB publishes 309.06110 in equatorial J2000 and the IAU value is "
        f"309.031, so it has been converted when it should not have been")
    assert abs(values["generic_body.spin.pole_dec_deg"] - 42.232) < 0.001, (
        f"the pole declination came out "
        f"{values['generic_body.spin.pole_dec_deg']}; SBDB publishes "
        f"42.232386 and the IAU value is 42.235")
    assert abs(values["generic_body.spin.pole_ra_deg"] - 309.031) < 0.05
    assert abs(values["generic_body.spin.pole_dec_deg"] - 42.235) < 0.05

    assert abs(values["generic_body.spin.period_hours"]
               - 5.3421276322) < 1e-9

    # Provenance is PER FIELD and names the inputs of the conversion.
    assert result.provenance["generic_body.shape.radius_km"] == "SBDB"
    albedo_source = result.provenance["generic_body.surface.albedo"]
    assert "p_V=0.4228" in albedo_source and "G=0.32" in albedo_source, \
        albedo_source

    # Thermal inertia is NOT fetched, and the card says why.
    assert not any(k.startswith("generic_body.surface.ks")
                   or k.startswith("generic_body.surface.rhos")
                   or k.startswith("generic_body.surface.cp0")
                   for k in values), (
        "a thermal-inertia input was auto-populated from a database that does "
        "not publish one")
    assert any("Thermal inertia is NOT filled in" in w
               for w in result.warnings), result.warnings

    print(f"    [sbdb] diameter {stub.diameter_km} -> radius "
          f"{values['generic_body.shape.radius_km']} km; p_V {stub.albedo} -> "
          f"Bond {bond:.6f}; pole kept equatorial at "
          f"({values['generic_body.spin.pole_ra_deg']:.3f}, "
          f"{values['generic_body.spin.pole_dec_deg']:.3f}), IAU says "
          f"(309.031, 42.235)")


def test_sbdb_parses_the_pole_h_and_g_it_used_to_drop():
    """The parser reads the three fields no earlier version read.

    ``pole`` is a string, so the old ``float(value)`` gate discarded it before
    the name was ever tested -- which is why no lookup in this project has ever
    filled in a spin pole, however many bodies have a published one.
    """
    from horizons.horizons_client import HorizonsClient, _parse_sbdb_pole

    ra, dec = _parse_sbdb_pole("309.06110/42.232386")
    assert (ra, dec) == (309.06110, 42.232386)
    # The '+' sign really appears in SBDB records ((21) Lutetia is "51.8/+10.8").
    assert _parse_sbdb_pole("51.8/+10.8") == (51.8, 10.8)
    assert _parse_sbdb_pole("") == (None, None)
    assert _parse_sbdb_pole("garbage") == (None, None)
    assert _parse_sbdb_pole("309.0/99.0") == (None, None), \
        "a declination outside +-90 was accepted"

    client = HorizonsClient()
    client._fetch_json = lambda url, params: {
        "object": {"fullname": "4 Vesta"},
        "phys_par": [
            {"name": "diameter", "value": "522.77"},
            {"name": "albedo", "value": "0.4228"},
            {"name": "rot_per", "value": "5.3421276322"},
            {"name": "H", "value": "3.25"},
            {"name": "G", "value": "0.32"},
            {"name": "pole", "value": "309.06110/42.232386"},
            {"name": "spec_T", "value": "V"},
        ],
    }
    physical = client.get_physical_params("vesta")
    assert physical.absolute_magnitude_h == 3.25
    assert physical.slope_parameter_g == 0.32
    # EQUATORIAL, kept as published. Checked against the IAU WGCCRE pole
    # (309.031, 42.235), which is an independent source; a version of this
    # parser that treated SBDB's pole as ecliptic produced (300.41, 22.83),
    # about 20 degrees away, and agreed with a stub that shared its belief.
    assert abs(physical.pole_ra_deg - 309.06110) < 1e-6, physical.pole_ra_deg
    assert abs(physical.pole_dec_deg - 42.232386) < 1e-6, physical.pole_dec_deg
    assert abs(physical.pole_ra_deg - 309.031) < 0.05, (
        "the SBDB pole does not agree with the IAU value, so it has been "
        "transformed when it should have been used as published")
    assert abs(physical.pole_dec_deg - 42.235) < 0.05
    assert physical.pole_ecliptic_lon_deg == -999.0, (
        "an ecliptic pole was invented from an equatorial one")
    assert physical.albedo_is_geometric is True
    assert physical.spectral_type == "V"

    # A body with no G keeps None, which is DISTINCT from a fitted zero.
    client._fetch_json = lambda url, params: {
        "object": {}, "phys_par": [{"name": "albedo", "value": "0.05"}]}
    sparse = client.get_physical_params("something")
    assert sparse.slope_parameter_g is None, (
        "an absent G became a number, so the label could not say 'assumed'")

    print("    [sbdb] pole, H and G are parsed; an absent G stays None")


# --------------------------------------------------------------------------
# 3. The negative control: a kernel that does not cover the run
# --------------------------------------------------------------------------

def _config(start="2035-08-24T00:00:00", days=10.0, cycles=20.0, period=5.0):
    return {
        "simulation": {"start_time": start, "duration_days": days},
        "thermal": {"equilibration_orbits": cycles},
        "generic_body": {"spin": {"period_hours": period}},
    }


class _FakeCoverage:
    """Stands in for a real ``spkcov`` read of a kernel on disk."""

    def __init__(self, available, start_utc="", end_utc="", days=0.0,
                 detail="stub"):
        self.available = available
        self.start_utc = start_utc
        self.end_utc = end_utc
        self.days = days
        self.detail = detail
        self.bodies = ()

    def covers(self, start_utc, end_utc):
        if not self.available:
            return None
        return bool(self.start_utc <= str(start_utc)[:19]
                    and str(end_utc)[:19] <= self.end_utc)


def _with_stub_coverage(coverage):
    """Patch ``body_cards.spk_coverage``; returns the restore callable."""
    from gui_core import body_cards
    original = body_cards.spk_coverage
    body_cards.spk_coverage = lambda path, metakernel=None: coverage
    return lambda: setattr(body_cards, "spk_coverage", original)


class _SpkClient:
    """A HORIZONS stub that writes a plausible file and records the request."""

    calls = []

    def __init__(self, payload=b"DAF/SPK stub payload"):
        self.payload = payload

    def download_spk(self, body_id, start, stop, output_path):
        type(self).calls.append((body_id, start, stop))
        Path(output_path).write_bytes(self.payload)
        return output_path


def test_a_kernel_that_does_not_cover_the_run_is_refused():
    """PTM-145's negative control, and the arm that matters most here.

    The kernel arrives. It is a real file. It simply does not span the run --
    which is exactly the Justitia case, and exactly what the engine does not
    report. The fetch must NOT set the ephemeris source to SPICE.
    """
    from gui_core.body_fetch import fetch_spk

    _SpkClient.calls = []
    # Covers a month of 2034; the run is in 2035. The real Justitia numbers.
    restore = _with_stub_coverage(_FakeCoverage(
        True, "2034-10-12T00:00:00", "2034-11-11T00:00:00", 30.0,
        "ephemeris.spk"))
    try:
        with tempfile.TemporaryDirectory() as root:
            result = fetch_spk(_config(), root, name="269 Justitia (A887 SA)",
                               naif_id="20000269",
                               client_factory=lambda: _SpkClient())
    finally:
        restore()

    assert not result.ok, "an out-of-coverage kernel was accepted"
    assert result.refusal, "no refusal was reported"
    assert "REFUSED" in result.refusal, result.refusal
    assert "generic_body.ephemeris_source" not in result.values, (
        "the ephemeris source was set to SPICE over a kernel that does not "
        "cover the run -- the engine would serve the uncovered epochs from "
        "Keplerian elements and log nothing")
    assert "generic_body.spice_metakernel" not in result.values
    assert "2034" in result.message and "2035" in result.refusal, (
        result.message + result.refusal)

    print(f"    [refusal] a 2034 kernel over a 2035 run is refused and leaves "
          f"the ephemeris source alone")


def test_an_unreadable_kernel_is_unknown_and_not_fine():
    """'Could not check' must not be reported as 'covered'."""
    from gui_core.body_fetch import fetch_spk

    restore = _with_stub_coverage(_FakeCoverage(False, detail="no spiceypy"))
    try:
        with tempfile.TemporaryDirectory() as root:
            result = fetch_spk(_config(), root, name="623 Chimaera",
                               naif_id="20000623",
                               client_factory=lambda: _SpkClient())
    finally:
        restore()

    assert not result.ok
    assert "UNKNOWN" in result.refusal, result.refusal
    assert "generic_body.ephemeris_source" not in result.values, (
        "an unverifiable kernel was accepted as SPICE")
    print("    [unknown] an unreadable kernel is UNKNOWN, and UNKNOWN is not "
          "'fine'")


def test_the_fetched_window_carries_a_year_of_margin_at_each_end():
    """PTM-300: the request must reach well past the run, at BOTH ends.

    Both ends, because equilibration integrates BEFORE the start time. A margin
    that only extends forward -- which is what the HORIZONS page computes --
    leaves every spin-up epoch to the silent fallback.
    """
    from datetime import datetime

    from gui_core.body_fetch import SPK_MARGIN_DAYS, fetch_spk

    _SpkClient.calls = []
    restore = _with_stub_coverage(_FakeCoverage(
        True, "2030-01-01T00:00:00", "2040-01-01T00:00:00", 3653.0))
    try:
        with tempfile.TemporaryDirectory() as root:
            fetch_spk(_config(start="2035-08-24T00:00:00", days=10.0,
                              cycles=20.0, period=5.0),
                      root, name="269 Justitia", naif_id="20000269",
                      client_factory=lambda: _SpkClient())
    finally:
        restore()

    assert _SpkClient.calls, "no request was made"
    body_id, start_text, stop_text = _SpkClient.calls[-1]
    start = datetime.fromisoformat(start_text)
    stop = datetime.fromisoformat(stop_text)

    run_start = datetime.fromisoformat("2035-08-24T00:00:00")
    spin_up_days = 20.0 * 5.0 / 24.0
    earliest = run_start.timestamp() - spin_up_days * 86400.0

    assert (earliest - start.timestamp()) / 86400.0 >= SPK_MARGIN_DAYS - 1.5, (
        f"requested start {start_text} is not a year before the spin-up start")
    latest = run_start.timestamp() + 10.0 * 86400.0
    assert (stop.timestamp() - latest) / 86400.0 >= SPK_MARGIN_DAYS - 1.5, (
        f"requested stop {stop_text} is not a year past the run end")
    assert body_id.endswith(";"), (
        f"{body_id!r} has no small-body marker; HORIZONS would read a bare "
        f"number as a major-body NAIF id (record 4 is Vesta, NAIF 4 is the "
        f"Mars barycenter)")

    print(f"    [margin] run 2035-08-24 +10 d with {spin_up_days:.1f} d of "
          f"spin-up -> requested {start_text} to {stop_text}")


def test_a_covering_kernel_is_accepted_and_cached_and_reused_offline():
    """Cache round trip: the second call must not touch the network."""
    from gui_core.body_fetch import fetch_spk

    coverage = _FakeCoverage(True, "2030-01-01T00:00:00",
                             "2040-01-01T00:00:00", 3653.0, "ephemeris.spk")
    restore = _with_stub_coverage(coverage)
    try:
        with tempfile.TemporaryDirectory() as root:
            first = fetch_spk(_config(), root, name="623 Chimaera (A907 BC)",
                              naif_id="20000623",
                              client_factory=lambda: _SpkClient())
            assert first.ok, first.message + first.refusal
            assert not first.from_cache
            assert first.values["generic_body.ephemeris_source"] == "spice"
            spk = Path(first.paths["spk"])
            assert spk.exists() and spk.name == "ephemeris.spk", spk
            assert spk.parent.name == "623_Chimaera_(A907_BC)", spk.parent
            assert not list(spk.parent.glob("*.part")), (
                "the scratch download file was left behind")

            index = json.loads(
                (Path(root) / "data" / "horizons_cache"
                 / "index.json").read_text())
            entry = index["623 Chimaera (A907 BC)"]
            assert entry["spk_path"] == str(spk)
            assert entry["spk_start"] and entry["spk_stop"]

            # Second call: a client that would RAISE if it were used at all.
            def _explode():
                raise AssertionError(
                    "the cache was not consulted; the network was")

            second = fetch_spk(_config(), root,
                               name="623 Chimaera (A907 BC)",
                               naif_id="20000623",
                               client_factory=_explode)
            assert second.ok, second.message
            assert second.from_cache, "a cached, covering kernel was refetched"
            assert (second.values["generic_body.ephemeris_source"]
                    == "spice")
    finally:
        restore()

    print("    [cache] a covering kernel is written once, indexed, and reused "
          "with no network on the second call")


def test_the_ways_an_spk_fetch_can_fail_are_distinct_and_change_nothing():
    from gui_core.body_fetch import fetch_spk

    restore = _with_stub_coverage(_FakeCoverage(False, detail="no spiceypy"))
    try:
        with tempfile.TemporaryDirectory() as root:
            # (a) no name
            unnamed = fetch_spk(_config(), root, name="")
            # (b) a run window that cannot be worked out
            windowless = fetch_spk({"simulation": {"start_time": "soon"}},
                                   root, name="623 Chimaera")
            # (c) offline
            offline = fetch_spk(_config(), root, name="623 Chimaera",
                                allow_online=False)

            # (d) the service cannot be reached
            class _Dead:
                def download_spk(self, *a, **k):
                    raise OSError("Network is unreachable")

            dead = fetch_spk(_config(), root, name="623 Chimaera",
                             client_factory=_Dead)

            # (e) the service answers with nothing
            class _Empty:
                def download_spk(self, body_id, start, stop, output_path):
                    Path(output_path).write_bytes(b"")
                    return output_path

            empty = fetch_spk(_config(), root, name="623 Chimaera",
                              client_factory=_Empty)
    finally:
        restore()

    results = [unnamed, windowless, offline, dead, empty]
    for result in results:
        assert not result.ok, result.message
        assert result.values == {}, f"a failed fetch wrote {result.values}"
        assert result.message, "a failure with no message"
    messages = {r.message for r in results}
    assert len(messages) == 5, (
        f"two failure modes give the same message: {sorted(messages)}")
    assert "Network is unreachable" in dead.message, dead.message
    assert "turned off" in offline.message, offline.message
    assert "empty kernel" in empty.message, empty.message
    print("    [failures] five distinct SPK failure messages, none of which "
          "changes a value")


# --------------------------------------------------------------------------
# 4. DAMIT: the shape, its provenance sidecar, and the spin that came with it
# --------------------------------------------------------------------------

class _StubSpin:
    lambda_ecl_deg = 103.0
    beta_ecl_deg = 22.0
    period_hours = 14.635
    epoch_jd = 2451545.0


class _StubModel:
    damit_id = 9110
    asteroid_number = 623
    asteroid_name = "Chimaera"
    vertices = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0], [-1.0, -1.0, -1.0]]
    faces = [[0, 1, 2], [0, 1, 3], [1, 2, 3], [0, 2, 3]]
    spin = _StubSpin()
    num_vertices = 4
    num_faces = 4


class _DamitClient:
    def __init__(self, number=623, damit_id=9110):
        self._number = number
        self._damit_id = damit_id

    def search_asteroid(self, query):
        return [{"damit_id": self._damit_id, "number": self._number,
                 "name": "Chimaera"}]

    def get_model(self, damit_id):
        assert damit_id == self._damit_id
        return _StubModel()


def test_a_damit_shape_is_cached_with_its_md5_and_brings_its_spin():
    from gui_core.body_fetch import fetch_damit_shape, shape_provenance

    with tempfile.TemporaryDirectory() as root:
        result = fetch_damit_shape(root, name="623 Chimaera (A907 BC)",
                                   naif_id="20000623", radius_km=22.045,
                                   client_factory=_DamitClient)
        assert result.ok, result.message
        shape = Path(result.paths["shape"])
        assert shape.exists() and shape.suffix == ".obj"
        assert shape.name == "shape_damit_9110.obj", shape.name

        # PTM-303: the .obj itself is swallowed by .gitignore, so its identity
        # lives in a sidecar that is NOT ignored.
        record = shape_provenance(shape)
        assert record["damit_id"] == 9110
        assert record["source"] == "DAMIT"
        assert len(record["md5"]) == 32, record["md5"]
        import hashlib
        assert record["md5"] == hashlib.md5(shape.read_bytes()).hexdigest()
        assert record["vertices"] == 4 and record["faces"] == 4
        assert "gitignore" in record["gitignore_note"].lower()
        assert any(".gitignore" in w for w in result.warnings), result.warnings
        assert any("git add -f" in w for w in result.warnings), result.warnings

        # The shape is wired to the key the SOLVER reads, through the card's
        # own mirror: the shape type must switch to the one that is read.
        assert result.values["generic_body.shape.type"] == "custom_mesh"
        assert result.values["generic_body.shape.mesh_file"] == str(shape)

        # The spin comes WITH the shape, converted out of DAMIT's ecliptic.
        from gui_core.body_properties import ecliptic_pole_to_equatorial
        ra, dec = ecliptic_pole_to_equatorial(103.0, 22.0)
        assert abs(result.values["generic_body.spin.pole_ra_deg"]
                   - round(ra, 3)) < 1e-9
        assert abs(result.values["generic_body.spin.pole_dec_deg"]
                   - round(dec, 3)) < 1e-9
        assert abs(result.values["generic_body.spin.period_hours"]
                   - 14.635) < 1e-9
        assert result.provenance[
            "generic_body.spin.pole_ra_deg"] == "DAMIT model 9110"

        # Scaled to km, so the scale factor is 1 and the card says so.
        assert result.values["generic_body.shape.mesh_scale"] == 1.0
        assert any("KILOMETERS" in w for w in result.warnings), result.warnings

        # Cache round trip: no client at all the second time.
        def _explode():
            raise AssertionError("DAMIT was asked again for a cached shape")

        again = fetch_damit_shape(root, name="623 Chimaera (A907 BC)",
                                  naif_id="20000623",
                                  client_factory=_explode)
        assert again.ok and again.from_cache, again.message
        assert again.values["generic_body.shape.mesh_file"] == str(shape)

    print("    [damit] shape cached as shape_damit_9110.obj with an md5 "
          "sidecar; its ecliptic pole arrives as RA/Dec; reused from cache")


def test_damit_does_not_substitute_a_near_miss_asteroid():
    """A different number is a different body, not a close one."""
    from gui_core.body_fetch import fetch_damit_shape

    with tempfile.TemporaryDirectory() as root:
        result = fetch_damit_shape(
            root, name="623 Chimaera", naif_id="20000623",
            client_factory=lambda: _DamitClient(number=6231, damit_id=1))
        assert not result.ok
        assert "none of its matches is asteroid 623" in result.message, \
            result.message
        assert result.values == {}
        assert not list(Path(root).rglob("*.obj")), (
            "a shape for the wrong asteroid was written to the cache")

    print("    [damit] a near-miss on the asteroid number is refused, not "
          "used")


def test_spin_and_shape_disagreeing_is_flagged_not_silently_resolved():
    from gui_core.body_fetch import spin_consistency

    shape_values = {
        "generic_body.spin.period_hours": 14.635,
        "generic_body.spin.pole_ra_deg": 143.077,
        "generic_body.spin.pole_dec_deg": 19.873,
    }

    agree = {"generic_body": {"spin": {
        "period_hours": 14.636, "pole_ra_deg": 143.1, "pole_dec_deg": 19.9}}}
    assert spin_consistency(agree, shape_values) == "", \
        spin_consistency(agree, shape_values)

    differ = {"generic_body": {"spin": {
        "period_hours": 7.3, "pole_ra_deg": 20.0, "pole_dec_deg": -40.0}}}
    flag = spin_consistency(differ, shape_values)
    assert flag, "a 2x period and a 90-degree pole difference was not flagged"
    assert "SPIN AND SHAPE DISAGREE" in flag, flag
    assert "7.3" in flag and "14.635" in flag, flag

    # Nothing configured yet is not a disagreement.
    assert spin_consistency({}, shape_values) == ""

    print("    [consistency] a divergent spin is flagged and neither side is "
          "overwritten")


# --------------------------------------------------------------------------
# 5. No fetch writes into anybody's configuration
# --------------------------------------------------------------------------

def test_no_fetch_applies_anything_by_itself():
    """The leak guard, at the layer where a leak would start.

    Every fetch RETURNS ``{key: value}`` for the card to apply under its own
    route check. None of them holds a ConfigManager, so a result that arrives
    after the user has moved to another card cannot write into that card's
    run -- which is the cross-card residue the body-identity gate exists to
    catch, caught here one level earlier.
    """
    import inspect

    from gui_core import body_fetch

    for name in ("fetch_spk", "fetch_damit_shape", "fetch_sbdb_properties"):
        function = getattr(body_fetch, name)
        signature = inspect.signature(function)
        for parameter in signature.parameters:
            assert "config_manager" not in parameter, (
                f"{name} takes a {parameter}; a fetch that can write is a "
                f"fetch that can write into the wrong card")
        source = inspect.getsource(function)
        assert "set_value" not in source, (
            f"{name} calls set_value; fetches return values, they do not "
            f"apply them")

    # And a fetch given a configuration must not mutate it.
    config = _config()
    before = json.dumps(config, sort_keys=True)
    restore = _with_stub_coverage(_FakeCoverage(False, detail="stub"))
    try:
        with tempfile.TemporaryDirectory() as root:
            body_fetch.fetch_spk(config, root, name="623 Chimaera",
                                 allow_online=False)
    finally:
        restore()
    assert json.dumps(config, sort_keys=True) == before, (
        "fetch_spk mutated the configuration it was shown")

    print("    [leak] no fetch takes a config manager, calls set_value, or "
          "mutates the configuration it is shown")


# --------------------------------------------------------------------------
# 6. The card: the buttons exist, a refusal writes nothing, and a late answer
#    for another card is dropped
# --------------------------------------------------------------------------

def _card(scratch):
    """One offscreen Asteroid card. Raises Skip when Qt cannot start here."""
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        import PyQt6
    except ImportError as exc:                              # pragma: no cover
        raise Skip(f"PyQt6 is unavailable here ({exc})")
    plug = Path(PyQt6.__file__).parent / "Qt6" / "plugins"
    os.environ.setdefault("QT_PLUGIN_PATH", str(plug))
    os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH",
                          str(plug / "platforms"))
    from PyQt6.QtCore import QSettings
    from PyQt6.QtWidgets import QApplication

    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    for scope in (QSettings.Scope.UserScope, QSettings.Scope.SystemScope):
        QSettings.setPath(QSettings.Format.IniFormat, scope, scratch)
    from gui_core.config_manager import ConfigManager
    from gui_panels.config_panel import ConfigurationPanel

    app = QApplication.instance() or QApplication([])
    panel = ConfigurationPanel(ConfigManager())
    start = panel.tabs["start"]
    start.choose_body("asteroid")
    start._choose_mode("body")
    card = start.open_card("asteroid", "body", "time_series",
                           apply_defaults=True)
    return app, panel, card


def test_the_card_offers_the_three_fetches_and_none_runs_on_a_keystroke():
    import tempfile

    from PyQt6.QtWidgets import QPushButton

    with tempfile.TemporaryDirectory() as scratch:
        app, _panel, card = _card(scratch)
        buttons = {
            "SPK": card.spk_fetch_button,
            "SBDB": card.sbdb_button,
            "DAMIT": card.damit_button,
        }
        for name, button in buttons.items():
            assert isinstance(button, QPushButton), name
            assert button.text().strip(), f"the {name} button has no text"
            assert button.toolTip().strip(), (
                f"the {name} button has no tooltip")

        # Typing a name must not reach the network. All three notes stay as
        # they were: a request per keystroke would fire on every prefix of a
        # name somebody is halfway through typing.
        before = (card.lookup_note.text(), card.coverage_note.text(),
                  card.shape_provenance.text())
        card.widgets["generic_body.name"].setText("101955 Bennu")
        card.widgets["generic_body.shape.radius_km"].setValue(0.2447)
        app.processEvents()
        after = (card.lookup_note.text(), card.coverage_note.text(),
                 card.shape_provenance.text())
        assert before == after, (
            f"typing started a fetch: {before!r} -> {after!r}")

    print(f"    [card] three fetch buttons, all tooltipped, none wired to a "
          f"keystroke")


def test_a_refused_kernel_writes_nothing_into_the_card():
    """The card half of the PTM-145 negative control."""
    import tempfile

    from gui_core.body_fetch import FetchResult

    with tempfile.TemporaryDirectory() as scratch:
        app, _panel, card = _card(scratch)
        card.config_manager.set_value("generic_body.name", "269 Justitia")
        card.config_manager.set_value("generic_body.ephemeris_source",
                                      "keplerian")
        card.load_from_config()

        # The refusal deliberately CARRIES values. `fetch_spk` does not
        # populate them on a refusal today, so a card that only checked
        # `result.values` would pass this arm vacuously -- and the card's own
        # guard is the last one before a run gets a silent Keplerian fallback,
        # so it is pinned here rather than assumed from the layer below.
        refusal = FetchResult(
            ok=False, kind="spk",
            values={"generic_body.ephemeris_source": "spice",
                    "generic_body.spice_metakernel":
                        "/tmp/nowhere/metakernel.tm"},
            provenance={"generic_body.ephemeris_source": "HORIZONS SPK"},
            message="A kernel was downloaded; it covers 2034-10-12 to "
                    "2034-11-11.",
            refusal="REFUSED: this run needs 2035-08-24 to 2035-09-03.",
            paths={"spk": "/tmp/nowhere/ephemeris.spk"})
        card._on_fetch_done(refusal, card.route)
        app.processEvents()

        assert card.config_manager.get_value(
            "generic_body.ephemeris_source") == "keplerian", (
            "a refused kernel switched the ephemeris source to SPICE")
        assert not str(card.config_manager.get_value(
            "generic_body.spice_metakernel", "") or "").strip(), (
            "a refused kernel was written into the metakernel field")
        note = card.coverage_note.text()
        assert "REFUSED" in note, note
        assert "2034" in note and "2035" in note, note

    print("    [card] a refusal is shown in full and changes no value")


def test_an_answer_that_arrives_for_another_card_is_discarded():
    """The cross-card leak arm, at the point where a leak would happen.

    A fetch can take a minute and a half. If the user goes to the Moon card in
    the meantime, applying the answer would write an asteroid's shape, spin and
    ephemeris into a lunar run -- a configuration that is internally valid and
    completely wrong, which is exactly what the body-identity gate was built
    after.
    """
    import tempfile

    from gui_core.body_fetch import FetchResult

    with tempfile.TemporaryDirectory() as scratch:
        app, _panel, card = _card(scratch)
        before = json.dumps(card.config_manager.get_config(), sort_keys=True,
                            default=str)

        landed = FetchResult(
            ok=True, kind="sbdb",
            values={"generic_body.shape.radius_km": 262.7,
                    "generic_body.surface.albedo": 0.21373},
            provenance={"generic_body.shape.radius_km": "SBDB",
                        "generic_body.surface.albedo": "SBDB p_V=0.42, G=0.32"},
            message="SBDB: filled in radius 262.7 km.")

        # (a) the answer is for a DIFFERENT route
        card._on_fetch_done(landed, "moon/global")
        app.processEvents()
        assert json.dumps(card.config_manager.get_config(), sort_keys=True,
                          default=str) == before, (
            "a result for another route was applied to this card")

        # (b) the answer is for this route, but the card is no longer live
        card.deactivate()
        card._on_fetch_done(landed, card.route)
        app.processEvents()
        assert json.dumps(card.config_manager.get_config(), sort_keys=True,
                          default=str) == before, (
            "a result was applied to a card that is no longer displayed")

        # (c) same answer, same route, card live: NOW it applies.
        card.activate("body", "time_series")
        card._on_fetch_done(landed, card.route)
        app.processEvents()
        assert abs(card.config_manager.get_value(
            "generic_body.shape.radius_km", 0.0) - 262.7) < 1e-9
        assert "SBDB" in card.albedo_provenance.text(), \
            card.albedo_provenance.text()

    print("    [leak] a result for another route, and a result for a card "
          "that has been left, are both dropped; the live case applies")


def test_the_field_label_carries_the_conversion_and_a_hand_edit_drops_it():
    import tempfile

    from gui_core.body_fetch import FetchResult

    with tempfile.TemporaryDirectory() as scratch:
        app, _panel, card = _card(scratch)
        card._on_fetch_done(FetchResult(
            ok=True, kind="sbdb",
            values={"generic_body.surface.albedo": 0.21373,
                    "generic_body.spin.period_hours": 5.342128},
            provenance={"generic_body.surface.albedo":
                        "SBDB p_V=0.42, G=0.32",
                        "generic_body.spin.period_hours": "SBDB"},
            message="SBDB: filled in Bond albedo 0.21373."), card.route)
        app.processEvents()

        label = card._grid_label(*card._albedo_label_cell)
        assert label is not None, "the albedo label could not be found"
        assert "p_V=0.42" in label.text(), label.text()
        assert "G=0.32" in label.text(), label.text()
        assert "SBDB" in card.spin_provenance.text(), \
            card.spin_provenance.text()

        # A hand edit is not SBDB's number any more, and the label must stop
        # saying it is.
        card.widgets["generic_body.surface.albedo"].setValue(0.07)
        app.processEvents()
        assert "p_V" not in label.text(), (
            f"the label still claims a conversion after a hand edit: "
            f"{label.text()!r}")
        assert "manual" in card.albedo_provenance.text(), \
            card.albedo_provenance.text()
        # The period was NOT edited, so its source line is untouched.
        assert "SBDB" in card.spin_provenance.text(), \
            card.spin_provenance.text()

    print("    [provenance] the label carries p_V and G, and a hand edit "
          "drops that field's source and only that field's")


def test_a_body_name_cannot_place_the_cache_outside_the_cache():
    """The body name is free text and it builds a path files are written to.

    ``DataManager._body_dir``'s rule replaces spaces and slashes, which leaves
    ``..`` untouched. Refused rather than rewritten: quietly renaming a body
    would put its kernel and its shape somewhere the user was never told
    about, which is a worse answer than "no".
    """
    from gui_core.body_fetch import body_cache_dir, fetch_damit_shape, fetch_spk

    with tempfile.TemporaryDirectory() as root:
        ok = body_cache_dir(root, "623 Chimaera (A907 BC)")
        assert ok.name == "623_Chimaera_(A907_BC)", ok.name

        # The property is "inside the cache, or refused" -- NOT "always
        # refused". Replacing '/' with '_' already makes most of these into
        # harmless literal directory names (``../..`` becomes ``.._..``), and
        # a test that demanded a refusal for those would be asserting a
        # behavior the code does not need and should not have. The one input
        # that genuinely escapes is a name that is exactly ``..``, because it
        # contains no separator to replace.
        cache_root = (Path(root) / "data" / "horizons_cache"
                      / "bodies").resolve()
        escaped, refused = [], []
        for hostile in ("..", ".", "../..", "../../../etc", "a/../..",
                        "/etc/passwd", "..\\..", "  ..  "):
            try:
                got = body_cache_dir(root, hostile).resolve()
            except ValueError:
                refused.append(hostile)
                continue
            if got != cache_root and cache_root not in got.parents:
                escaped.append(f"{hostile!r} -> {got}")
        assert not escaped, (
            "a body name placed the cache outside the cache: "
            + "; ".join(escaped))
        assert ".." in refused, (
            f"a body name of '..' was not refused; it was {refused}")

        # And the fetches turn it into a message rather than an exception --
        # they are called from a Qt slot, where an escaping exception is a
        # SIGABRT that takes any unsaved edits with it.
        for result in (fetch_spk(_config(), root, name=".."),
                       fetch_damit_shape(root, name="..", naif_id="20000623")):
            assert not result.ok
            assert result.message and "inside the body cache" in result.message

    print("    [paths] a body name that would escape the cache is refused, "
          "not rewritten, and never raises out of a slot")


CHECKS = [
    test_geometric_albedo_becomes_bond_albedo,
    test_a_missing_slope_parameter_is_assumed_and_labeled,
    test_the_published_spin_pole_is_ecliptic_and_gets_converted,
    test_sbdb_maps_every_field_it_claims_to,
    test_sbdb_parses_the_pole_h_and_g_it_used_to_drop,
    test_a_kernel_that_does_not_cover_the_run_is_refused,
    test_an_unreadable_kernel_is_unknown_and_not_fine,
    test_the_fetched_window_carries_a_year_of_margin_at_each_end,
    test_a_covering_kernel_is_accepted_and_cached_and_reused_offline,
    test_the_ways_an_spk_fetch_can_fail_are_distinct_and_change_nothing,
    test_a_damit_shape_is_cached_with_its_md5_and_brings_its_spin,
    test_damit_does_not_substitute_a_near_miss_asteroid,
    test_spin_and_shape_disagreeing_is_flagged_not_silently_resolved,
    test_no_fetch_applies_anything_by_itself,
    test_a_body_name_cannot_place_the_cache_outside_the_cache,
    test_the_card_offers_the_three_fetches_and_none_runs_on_a_keystroke,
    test_a_refused_kernel_writes_nothing_into_the_card,
    test_an_answer_that_arrives_for_another_card_is_discarded,
    test_the_field_label_carries_the_conversion_and_a_hand_edit_drops_it,
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
