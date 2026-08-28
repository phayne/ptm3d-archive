#!/usr/bin/env python3
"""MANUAL LIVE PROBE of the HORIZONS SPK endpoint. NOT a test. NOT automated.

This makes REAL network calls to ssd.jpl.nasa.gov and ssd-api.jpl.nasa.gov.
Nothing in ``src/python/tests`` may import it, and no gate runs it: a check
that needs a network is a check that fails on a train, and a suite that is
skipped whenever the machine is offline is a suite nobody trusts.  The unit
tests use stubs built from the responses this script captured; run this when
you want to know whether HORIZONS still behaves the way those stubs say.

It exists because the GUI's "Fetch kernel from HORIZONS" button failed on its
first live click (2026-08-27) with a message blaming the network, while
HORIZONS was up and had explained itself.  Reproducing that needed the real
service, so the real service is probed here, once, deliberately, by hand.

WHAT IT ESTABLISHED, 2026-08-27
-------------------------------

1.  ``COMMAND='<SPK ID>;'`` is wrong and always was.  A bare number in front of
    the semicolon is a small-body RECORD number, and those stop at 895910, so
    ``'20000623;'`` and ``'2101955;'`` both answer ``DXREAD: requested IOBJ=
    ... is out of bounds``.  This broke the fetch for EVERY numbered asteroid,
    not only for Bennu.
2.  ``COMMAND='623;'`` (IAU number) and ``COMMAND='DES=2000623;'`` (SPK ID as a
    designation) both return a kernel.  JPL's own documentation says the same.
3.  (101955) Bennu cannot be fetched at all.  Its HORIZONS record is marked
    ``*SPECIAL*`` and sourced from ``ORX_merged``; SPK generation answers ``SPK
    creation is not available for pre-computed objects in the major body
    index`` for every designation form tried -- ``101955;``, ``DES=101955;``,
    ``DES=2101955;``, ``DES=1999 RQ36;`` and the POST ``horizons_file.api``
    endpoint -- while VECTORS for the same object are served normally.  That is
    a property of the object, not of the request.

Usage::

    python3 scripts/probe_horizons_spk.py                 # the whole matrix
    python3 scripts/probe_horizons_spk.py --fetch 623     # a real download
    python3 scripts/probe_horizons_spk.py --fetch 623 --verify   # + spkcov

``--verify`` needs spiceypy; without it the coverage check is reported as
skipped rather than passed.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src" / "python") not in sys.path:
    sys.path.insert(0, str(REPO / "src" / "python"))

HORIZONS_API_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"

#: Every COMMAND form worth knowing the answer to, with what was measured on
#: 2026-08-27 so a rerun says "unchanged" or "CHANGED" rather than just
#: printing text again.
FORMS = [
    ("'623;'", "kernel", "IAU record number -- the documented form"),
    ("'DES=2000623;'", "kernel", "SPK ID via DES=, 2000000+N convention"),
    ("'DES=20000623;'", "kernel", "SPK ID via DES=, 20000000+N convention"),
    ("'Chimaera;'", "kernel", "name search"),
    ("'DES=A907 BC;'", "kernel", "provisional designation via DES="),
    ("'2000623;'", "out of bounds", "SPK ID in the RECORD slot -- the bug"),
    ("'20000623;'", "out of bounds", "the exact string fetch_spk used to send"),
    ("'2101955;'", "out of bounds", "Bennu, the same way"),
    ("'101955;'", "pre-computed", "Bennu resolved correctly, still refused"),
    ("'DES=2101955;'", "pre-computed", "Bennu by SPK ID, still refused"),
    ("'DES=1999 RQ36;'", "pre-computed", "Bennu by designation, still refused"),
    ("'DES=101955;'", "no match", "DES= takes a designation, not a number"),
    ("'599'", "pre-computed", "a MAJOR body: Jupiter, refused as one"),
    ("'599;'", "kernel", "the same digits with ';' are asteroid 599 Luisa"),
    ("'ZZZNOTABODY;'", "no match", "garbage designation"),
]


def ask(command, start="2025-01-01", stop="2025-02-01", timeout=60):
    """One live SPK request. Returns the parsed JSON."""
    params = {
        "format": "json", "COMMAND": command, "OBJ_DATA": "NO",
        "MAKE_EPHEM": "YES", "EPHEM_TYPE": "SPK", "CENTER": "'500@0'",
        "START_TIME": f"'{start}'", "STOP_TIME": f"'{stop}'",
    }
    url = HORIZONS_API_URL + "?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(url)
    request.add_header("User-Agent", "PTM3D-HorizonsProbe/1.0")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def classify(payload):
    """The outcome, in the same vocabulary the FORMS table uses."""
    if payload.get("spk"):
        return "kernel"
    text = str(payload.get("error") or payload.get("result") or "")
    if "out of bounds" in text:
        return "out of bounds"
    if "pre-computed objects in the major body index" in text:
        return "pre-computed"
    if "No matches found" in text:
        return "no match"
    return "UNCLASSIFIED"


def run_matrix(args):
    print(f"{'COMMAND':<20} {'measured':<14} {'2026-08-27':<14} note")
    drifted = 0
    for command, expected, note in FORMS:
        payload = ask(command, timeout=args.timeout)
        got = classify(payload)
        flag = "" if got == expected else "   <<< CHANGED"
        drifted += got != expected
        extra = ""
        if got == "kernel":
            extra = f" (spk_file_id {payload.get('spk_file_id')})"
        print(f"{command:<20} {got:<14} {expected:<14} {note}{extra}{flag}")
        if got == "UNCLASSIFIED":
            print("    raw: "
                  + " ".join(str(payload.get("error")
                                 or payload.get("result") or "").split())[:300])
    print()
    if drifted:
        print(f"{drifted} form(s) no longer behave as measured on 2026-08-27. "
              f"The stubs in src/python/tests/test_gui_body_autopopulate.py "
              f"were built from those measurements and need re-checking.")
    else:
        print("All 15 forms behave exactly as measured on 2026-08-27.")
    return 1 if drifted else 0


def run_fetch(args):
    """A REAL download through the production client, then verify it."""
    from horizons.horizons_client import HorizonsClient, small_body_spk_command

    command = small_body_spk_command(args.fetch, "asteroid")
    print(f"identifier {args.fetch!r} -> COMMAND='{command}'")
    print(f"window {args.start} to {args.stop}")

    # The identifier is argv and lands in a filename, so it is reduced to
    # characters that cannot leave the directory. A probe is still a program
    # that writes files.
    safe = "".join(c if (c.isalnum() or c in "-_") else "_"
                   for c in str(args.fetch))
    target = Path(args.out or (REPO / "build" / "probe_spk" / f"{safe}.bsp"))
    target.parent.mkdir(parents=True, exist_ok=True)
    HorizonsClient(timeout=args.timeout).download_spk(
        args.fetch, args.start, args.stop, str(target))
    size = target.stat().st_size
    header = target.read_bytes()[:8]
    print(f"wrote {target} ({size} bytes), header {header!r}")
    assert header == b"DAF/SPK ", "that is not an SPK"

    if not args.verify:
        print("coverage NOT checked (--verify not given)")
        return 0
    try:
        import spiceypy
    except ImportError:
        print("coverage SKIPPED: spiceypy is not installed in this "
              "interpreter")
        return 0

    # A leapseconds kernel is needed to PRINT a calendar date, not to read the
    # coverage. Without one the window is still reported, in TDB seconds and
    # via etcal, rather than the check being silently skipped.
    lsk = REPO / "data" / "kernels" / "lsk" / "naif0012.tls"
    have_lsk = lsk.exists()
    if have_lsk:
        spiceypy.furnsh(str(lsk))
    else:
        print(f"note: {lsk} is absent, so dates below are etcal (TDB) rather "
              f"than UTC")

    def when(seconds):
        if have_lsk:
            return spiceypy.timout(seconds, "YYYY-MM-DDTHR:MN:SC ::UTC") + " UTC"
        return spiceypy.etcal(seconds) + " TDB"

    ids = spiceypy.spkobj(str(target))
    print(f"bodies in the kernel: {[i for i in ids]}")
    spans = []
    for body in ids:
        cell = spiceypy.stypes.SPICEDOUBLE_CELL(2000)
        spiceypy.spkcov(str(target), body, cell)
        for i in range(spiceypy.wncard(cell)):
            begin, end = spiceypy.wnfetd(cell, i)
            spans.append((begin, end))
            print(f"  spkcov body {body}: {when(begin)} to {when(end)}")

    # Loadable, and it actually answers. spkcov says what the file DECLARES;
    # this asks it for a state, which is the thing the run will do.
    #
    # A HORIZONS small-body SPK is centered on the SUN (body 10) whatever
    # CENTER the request asked for -- the file's own header says "Center: 10".
    # Asking for a position relative to the solar system barycenter therefore
    # needs a planetary ephemeris loaded to chain 10 -> 0, and without one
    # SPICE answers SPKINSUFFDATA. The engine's metakernels load de440s for
    # exactly this reason; so does this probe, and it says which observer it
    # ended up using rather than quietly picking the one that works.
    spiceypy.furnsh(str(target))
    if not have_lsk:
        print("spkpos probe SKIPPED: no leapseconds kernel to build an epoch")
        return 0
    planetary = REPO / "data" / "kernels" / "spk" / "de440s.bsp"
    observer = "10"
    if planetary.exists():
        spiceypy.furnsh(str(planetary))
        observer = "0"
    else:
        print(f"note: {planetary} is absent, so the probe asks for a position "
              f"relative to the Sun rather than the barycenter")
    # The MIDDLE of what the kernel actually covers, not a fixed date: a fixed
    # one falls outside a short window and SPICE then answers SPKINSUFFDATA,
    # which reads like a broken kernel and is not one.
    if args.probe_epoch:
        epoch = spiceypy.str2et(args.probe_epoch)
        label = args.probe_epoch
    else:
        epoch = 0.5 * (spans[0][0] + spans[0][1])
        label = when(epoch) + " (the midpoint of the coverage)"
    position, light_time = spiceypy.spkpos(
        str(ids[0]), epoch, "J2000", "NONE", observer)
    where = "the solar system barycenter" if observer == "0" else "the Sun"
    print(f"  spkpos {ids[0]} at {label} relative to {where}: "
          f"[{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}] km, "
          f"one-way light time {light_time:.3f} s")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--fetch", default="",
                        help="download a real kernel for this identifier")
    parser.add_argument("--out", default="", help="where to write it")
    parser.add_argument("--start", default="2026-01-01")
    parser.add_argument("--stop", default="2028-01-01")
    parser.add_argument("--verify", action="store_true",
                        help="read the coverage back with spkcov, furnsh the "
                             "kernel and ask it for a state")
    parser.add_argument("--probe-epoch", default="",
                        help="the epoch the spkpos probe asks for (default: "
                             "the midpoint of the kernel's own coverage)")
    parser.add_argument("--timeout", type=int, default=90)
    args = parser.parse_args(argv)

    print("LIVE NETWORK PROBE -- this contacts JPL. It is not a test.\n")
    return run_fetch(args) if args.fetch else run_matrix(args)


if __name__ == "__main__":
    raise SystemExit(main())
