#!/usr/bin/env python3
"""The coverage check reads a metakernel as well as a binary SPK.

``generic_body.spice_metakernel`` is one field holding two file classes.
``body_fetch._spk_success`` writes the ``.tm`` into it whenever a metakernel
could be written and the ``.spk`` only as a fallback, so the ordinary case --
and every one of the seven cached ``body_config.yaml`` files -- is a TEXT
metakernel. It was handed to ``spkobj``, which reads binary SPKs, and SPICE
answered::

    SPICE(INVALIDARCHTYPE) -- Input file .../metakernel.tm has architecture
    KPL. The file must be a binary SPK file to be readable by this routine.

Six lines of banner and a routine traceback, presented to the user as the
reason their kernel was no good. The kernel was fine; every one of the thirteen
metakernels in this repository's cache reported UNKNOWN coverage, and UNKNOWN
is what makes ``fetch_spk`` decline to set ``ephemeris_source: spice`` and what
makes the card refuse to give a verdict.

Seven properties:

1. **A metakernel gets a real answer**, for the body the run is about.
2. **A text kernel never produces a raw SPICE architecture error.** That string
   is the negative control and it must not appear for any file class.
3. **The failures are distinguished.** No SPK among the members, a member that
   is not on disk, a member that is a cloud placeholder, an unresolved
   ``$SYMBOL`` and a NAIF ID that is in none of the SPKs are five different
   sentences, each naming its own cause.
4. **The NAIF ID is what makes the answer mean anything.** A metakernel loads
   ``de440s.bsp`` beside the body's own arc, and de440s spans 1849 to 2150.
5. **The binary-SPK answer is unchanged**, to the second and to the day.
6. **Nothing global moves.** No metakernel is furnished, and the working
   directory of a long-lived GUI process is exactly where it was.
7. **The card asks the question the way the module needs it asked** -- with
   the ID and the project root -- and shows the verdict.

Run directly::

    python3 src/python/tests/test_gui_spk_coverage.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "python"))
sys.path.insert(0, str(REPO_ROOT / "gui"))

#: The real, committed, hand-written relative metakernel from Paul's Justitia
#: run -- the file that reproduced this. Not a synthetic fixture: it carries
#: the PATH_SYMBOLS indirection, a body directory with parentheses in its name,
#: and paths relative to the repository root.
JUSTITIA_DIR = (REPO_ROOT / "data" / "horizons_cache" / "bodies"
                / "269_Justitia_(A887_SA)")
JUSTITIA_MK = JUSTITIA_DIR / "metakernel.tm"
JUSTITIA_SPK = JUSTITIA_DIR / "ephemeris.spk"
JUSTITIA_ID = "20000269"

#: The camera metakernel beside it, which is the ABSOLUTE-path form the
#: generator emits, complete with a `+` continuation splitting one OneDrive
#: path across two pool strings. A parser that ignored the continuation would
#: resolve `.../europa/heat3d/p` and report the wrong reason for the failure.
JUSTITIA_CAMERA_MK = JUSTITIA_DIR / "metakernel_camera.tm"

#: `data/kernels/{lsk,pck,spk}` are gitignored -- a fresh clone ships the four
#: `*.tm` files and nothing else until `scripts/fetch_kernels.py` runs. Every
#: check that needs a real ephemeris asks for these by name and SKIPS, so a
#: clone without them reports "not fetched" rather than a red bar.
KERNELS_DIR = REPO_ROOT / "data" / "kernels"
LSK = KERNELS_DIR / "lsk" / "naif0012.tls"
DE440S = KERNELS_DIR / "spk" / "de440s.bsp"

#: The string SPICE produces when a text kernel reaches the binary SPK reader.
#: This is the whole bug, so it is the negative control.
SPICE_ARCHITECTURE_ERROR = "must be a binary SPK file"


class Skip(Exception):
    """A check that cannot run here, as distinct from one that failed."""


def _require(*paths):
    for path in paths:
        if not path.exists():
            raise Skip(f"{path} is not in this checkout; run "
                       f"scripts/fetch_kernels.py")
    return paths[0]


def _write_metakernel(directory, name, path_values, symbols, kernels,
                      header="KPL/MK"):
    """Write a metakernel with the given pool contents. Returns its path."""
    values = ",\n                   ".join(f"'{v}'" for v in path_values)
    names = ",\n                   ".join(f"'{s}'" for s in symbols)
    loads = "\n            ".join(f"'{k}'" for k in kernels)
    text = (f"{header}\n\n\\begintext\n\n  A fixture.\n\n\\begindata\n\n"
            f"  PATH_VALUES  = ( {values} )\n"
            f"  PATH_SYMBOLS = ( {names} )\n"
            f"  KERNELS_TO_LOAD = (\n            {loads}\n  )\n\n"
            f"\\begintext\n")
    target = Path(directory) / name
    target.write_text(text)
    return target


# --------------------------------------------------------------------------
# 1. A metakernel gets a real answer
# --------------------------------------------------------------------------

def test_the_committed_justitia_metakernel_reports_its_own_coverage():
    """The reproducer, asserted on the file that reproduced it."""
    from gui_core.body_cards import spk_coverage

    _require(JUSTITIA_MK, JUSTITIA_SPK, LSK, DE440S)
    coverage = spk_coverage(JUSTITIA_MK, naif_id=JUSTITIA_ID,
                            project_root=REPO_ROOT)
    assert coverage.available, coverage.detail
    assert coverage.bodies == (int(JUSTITIA_ID),), coverage.bodies
    assert coverage.days > 1.0, coverage.days
    assert coverage.start_utc.startswith("20"), coverage.start_utc
    assert coverage.end_utc > coverage.start_utc, (coverage.start_utc,
                                                   coverage.end_utc)

    # The metakernel's answer is the SPK member's answer, not de440s's. Read
    # the member directly and compare: the two must agree exactly, which is
    # the whole claim.
    direct = spk_coverage(JUSTITIA_SPK, naif_id=JUSTITIA_ID,
                          project_root=REPO_ROOT)
    assert direct.available, direct.detail
    assert (coverage.start_utc, coverage.end_utc) == (direct.start_utc,
                                                      direct.end_utc), (
        f"the metakernel reports {coverage.start_utc}..{coverage.end_utc} "
        f"while its own SPK reports {direct.start_utc}..{direct.end_utc}")
    print(f"    [justitia] {coverage.start_utc} .. {coverage.end_utc} "
          f"({coverage.days:.1f} d), body {coverage.bodies[0]}")


def test_every_cached_metakernel_is_readable_and_none_blames_spice():
    """The negative control, over every metakernel in the cache.

    Before the fix this loop found thirteen for thirteen unavailable, each
    detail carrying the SPICE architecture banner.
    """
    from gui_core.body_cards import spk_coverage

    _require(LSK, DE440S)
    cache = REPO_ROOT / "data" / "horizons_cache" / "bodies"
    if not cache.is_dir():
        raise Skip("no horizons cache in this checkout")
    metakernels = sorted(cache.glob("*/*.tm"))
    if not metakernels:
        raise Skip("no metakernels in the cache")

    available = 0
    for path in metakernels:
        coverage = spk_coverage(path, project_root=REPO_ROOT)
        assert SPICE_ARCHITECTURE_ERROR not in coverage.detail, (
            f"{path.name} still reports the raw SPICE architecture error:\n"
            f"  {coverage.detail}")
        assert "spkobj_c" not in coverage.detail, coverage.detail
        available += bool(coverage.available)
    assert available, (
        f"none of the {len(metakernels)} cached metakernels could be read; "
        f"at least the committed ones must be")
    print(f"    [cache] {available}/{len(metakernels)} metakernels readable, "
          f"0 raw SPICE architecture errors")


def test_a_continued_path_is_reassembled_not_truncated():
    """SPICE's `+` continuation, on the real file that uses it."""
    from gui_core.body_cards import metakernel_members

    _require(JUSTITIA_CAMERA_MK)
    members = metakernel_members(JUSTITIA_CAMERA_MK, REPO_ROOT)
    every = [str(p) for _t, p in
             (members.kernels + members.missing + members.dataless)]
    joined = [p for p in every if p.endswith("ephemeris.spk")]
    assert joined, f"the body SPK member was not resolved at all: {every}"
    assert "heat3d/ptm/data" in joined[0], (
        f"the continued path was truncated: {joined[0]}")
    print(f"    [continuation] {joined[0]}")


# --------------------------------------------------------------------------
# 2. The failures are distinguished
# --------------------------------------------------------------------------

def test_a_missing_member_is_named_and_not_blamed_on_the_kernel():
    from gui_core.body_cards import spk_coverage

    _require(JUSTITIA_SPK, LSK)
    with tempfile.TemporaryDirectory() as scratch:
        mk = _write_metakernel(
            scratch, "missing_member.tm",
            [str(REPO_ROOT / "data" / "kernels"), scratch],
            ["KERNELS", "BODY"],
            ["$KERNELS/lsk/naif0012.tls", "$BODY/nowhere.bsp"])
        coverage = spk_coverage(mk, naif_id=JUSTITIA_ID,
                                project_root=REPO_ROOT)
    assert not coverage.available, coverage.detail
    assert "nowhere.bsp" in coverage.detail, coverage.detail
    assert "not on disk" in coverage.detail, coverage.detail
    assert SPICE_ARCHITECTURE_ERROR not in coverage.detail, coverage.detail
    print("    [missing] the member is named, not the metakernel")


def test_a_cloud_placeholder_member_is_called_a_cloud_placeholder():
    """The OneDrive-eviction case, said the way the pre-flight says it.

    A dataless file cannot be created here (that is the file system's answer,
    not ours), so `is_dataless` is stood in for -- which is exactly the one
    fact this branch turns on.
    """
    from gui_core import body_cards, preflight

    _require(JUSTITIA_SPK, LSK)
    original = preflight.is_dataless
    try:
        preflight.is_dataless = lambda path: str(path).endswith("evicted.bsp")
        with tempfile.TemporaryDirectory() as scratch:
            (Path(scratch) / "evicted.bsp").write_bytes(b"placeholder")
            mk = _write_metakernel(
                scratch, "evicted.tm",
                [str(REPO_ROOT / "data" / "kernels"), scratch],
                ["KERNELS", "BODY"],
                ["$KERNELS/lsk/naif0012.tls", "$BODY/evicted.bsp"])
            coverage = body_cards.spk_coverage(mk, project_root=REPO_ROOT)
    finally:
        preflight.is_dataless = original
    assert not coverage.available, coverage.detail
    assert "cloud placeholder" in coverage.detail, coverage.detail
    assert "evicted.bsp" in coverage.detail, coverage.detail
    assert "not on disk" not in coverage.detail, (
        "an evicted file was reported as a missing one; they are different "
        "problems with different remedies")
    print("    [evicted] reported as a cloud placeholder, not as missing")


def test_a_metakernel_with_no_spk_says_so():
    from gui_core.body_cards import spk_coverage

    kernels_dir = KERNELS_DIR
    _require(LSK, KERNELS_DIR / "pck" / "pck00011.tpc")
    with tempfile.TemporaryDirectory() as scratch:
        mk = _write_metakernel(
            scratch, "no_spk.tm", [str(kernels_dir)], ["KERNELS"],
            ["$KERNELS/lsk/naif0012.tls", "$KERNELS/pck/pck00011.tpc"])
        coverage = spk_coverage(mk, naif_id=JUSTITIA_ID,
                                project_root=REPO_ROOT)
    assert not coverage.available, coverage.detail
    assert "loads no SPK" in coverage.detail, coverage.detail
    assert "naif0012.tls [KPL/LSK]" in coverage.detail, coverage.detail
    assert "not on disk" not in coverage.detail, coverage.detail
    print("    [no SPK] the members are listed with what each one is")


def test_an_unresolved_path_symbol_says_which_symbol():
    from gui_core.body_cards import spk_coverage

    with tempfile.TemporaryDirectory() as scratch:
        mk = _write_metakernel(
            scratch, "no_symbol.tm", [str(REPO_ROOT / "data" / "kernels")],
            ["KERNELS"], ["$ELSEWHERE/ephemeris.spk"])
        coverage = spk_coverage(mk, project_root=REPO_ROOT)
    assert not coverage.available, coverage.detail
    assert "$ELSEWHERE" in coverage.detail, coverage.detail
    assert "PATH_SYMBOLS" in coverage.detail, coverage.detail
    print("    [symbol] the undefined symbol is named")


def test_a_text_kernel_is_identified_rather_than_called_corrupt():
    from gui_core.body_cards import spk_coverage

    lsk = _require(LSK)
    coverage = spk_coverage(lsk, project_root=REPO_ROOT)
    assert not coverage.available, coverage.detail
    assert "LSK" in coverage.detail, coverage.detail
    assert "leap seconds" in coverage.detail, coverage.detail
    assert SPICE_ARCHITECTURE_ERROR not in coverage.detail, coverage.detail
    assert "corrupted" not in coverage.detail, coverage.detail
    print("    [text kernel] named as an LSK, not reported as corrupt")


def test_the_five_failures_are_five_different_sentences():
    """No two of them may be reported with the same words."""
    from gui_core import body_cards, preflight

    kernels_dir = KERNELS_DIR
    _require(LSK, DE440S, JUSTITIA_MK, JUSTITIA_SPK)
    details = {}
    with tempfile.TemporaryDirectory() as scratch:
        details["no SPK"] = body_cards.spk_coverage(
            _write_metakernel(scratch, "a.tm", [str(kernels_dir)],
                              ["KERNELS"], ["$KERNELS/lsk/naif0012.tls"]),
            project_root=REPO_ROOT).detail
        details["missing"] = body_cards.spk_coverage(
            _write_metakernel(scratch, "b.tm", [str(kernels_dir), scratch],
                              ["KERNELS", "BODY"], ["$BODY/nowhere.bsp"]),
            project_root=REPO_ROOT).detail
        details["symbol"] = body_cards.spk_coverage(
            _write_metakernel(scratch, "c.tm", [str(kernels_dir)],
                              ["KERNELS"], ["$NOPE/x.bsp"]),
            project_root=REPO_ROOT).detail
        (Path(scratch) / "evicted.bsp").write_bytes(b"placeholder")
        original = preflight.is_dataless
        try:
            preflight.is_dataless = lambda p: str(p).endswith("evicted.bsp")
            details["evicted"] = body_cards.spk_coverage(
                _write_metakernel(scratch, "d.tm", [str(kernels_dir), scratch],
                                  ["KERNELS", "BODY"], ["$BODY/evicted.bsp"]),
                project_root=REPO_ROOT).detail
        finally:
            preflight.is_dataless = original
    details["wrong body"] = body_cards.spk_coverage(
        JUSTITIA_MK, naif_id="2000269", project_root=REPO_ROOT).detail

    for label, detail in details.items():
        assert detail, f"{label} said nothing at all"
        assert SPICE_ARCHITECTURE_ERROR not in detail, (label, detail)
    for left in details:
        for right in details:
            if left < right:
                assert details[left] != details[right], (
                    f"{left} and {right} are reported identically")
    print(f"    [taxonomy] {len(details)} causes, "
          f"{len(set(details.values()))} distinct messages")


# --------------------------------------------------------------------------
# 3. The NAIF ID is what makes the answer mean anything
# --------------------------------------------------------------------------

def test_a_naif_id_absent_from_the_kernels_is_reported_as_that():
    """`2000269` and `20000269` are different bodies, and this says so."""
    from gui_core.body_cards import spk_coverage

    _require(JUSTITIA_MK, JUSTITIA_SPK, LSK, DE440S)
    wrong = spk_coverage(JUSTITIA_MK, naif_id="2000269",
                         project_root=REPO_ROOT)
    assert not wrong.available, wrong.detail
    assert "no ephemeris for NAIF ID 2000269" in wrong.detail, wrong.detail
    assert "20000269" in wrong.detail, (
        "the message does not show the ID that IS there")
    # The positive control on the same file: the right ID passes. Without it
    # this check would also pass on a coverage reader that refused everything.
    right = spk_coverage(JUSTITIA_MK, naif_id=JUSTITIA_ID,
                         project_root=REPO_ROOT)
    assert right.available, right.detail
    print("    [wrong id] refused, and the right id on the same file passes")


def test_the_metakernel_answer_is_not_de440s_three_centuries():
    """The failure mode this whole argument turns on.

    `de440s.bsp` spans 1849 to 2150. A metakernel answer taken as the min to
    max across its bodies -- which is the right answer for a single SPK file,
    and which is what the binary path still does -- would report about 110,000
    days of coverage for every asteroid and pass every window ever asked
    about, in green.
    """
    from gui_core.body_cards import spk_coverage

    _require(JUSTITIA_MK, JUSTITIA_SPK, LSK, DE440S)
    for naif_id in (JUSTITIA_ID, ""):
        coverage = spk_coverage(JUSTITIA_MK, naif_id=naif_id,
                                project_root=REPO_ROOT)
        assert coverage.available, coverage.detail
        assert coverage.days < 20000, (
            f"naif_id={naif_id!r} reported {coverage.days:.0f} days, which is "
            f"de440s's span and not this asteroid's arc: {coverage.detail}")
    # A run window de440s covers and the asteroid's arc does not must be
    # reported as NOT covered.
    coverage = spk_coverage(JUSTITIA_MK, naif_id=JUSTITIA_ID,
                            project_root=REPO_ROOT)
    assert coverage.covers("1990-01-01T00:00:00",
                           "1990-02-01T00:00:00") is False, coverage.detail
    print(f"    [de440s] {coverage.days:.1f} days, not a planetary ephemeris'"
          f" three centuries")


def test_an_unparsable_naif_id_is_refused_rather_than_ignored():
    from gui_core.body_cards import spk_coverage

    _require(JUSTITIA_MK, JUSTITIA_SPK, LSK, DE440S)
    coverage = spk_coverage(JUSTITIA_MK, naif_id="Justitia",
                            project_root=REPO_ROOT)
    assert not coverage.available, (
        "a NAIF ID that is not a number was silently ignored, and the answer "
        "then covers whatever the metakernel happens to hold")
    assert "not an integer NAIF ID" in coverage.detail, coverage.detail
    print("    [bad id] refused out loud")


def test_a_bare_kernel_name_resolves_the_way_the_engine_resolves_it():
    """`bennu.tm` is the shipped quickstart's spelling, and it is not a path.

    ``docs/configuration.md``: a relative value that is not a ``data/...``
    path is looked for under the KERNEL DIRECTORY first, then the working
    directory. Resolving it as ``<project root>/bennu.tm`` reports "not on
    disk" for a configuration that runs correctly -- the card telling the user
    their kernel is missing when it is not.
    """
    from gui_core.body_cards import spk_coverage

    bare = _require(KERNELS_DIR / "bennu.tm").name
    coverage = spk_coverage(bare, project_root=REPO_ROOT)
    if not coverage.available and "not on disk" in coverage.detail:
        raise AssertionError(
            f"the bare metakernel name was not resolved under the kernel "
            f"directory: {coverage.detail}")
    assert SPICE_ARCHITECTURE_ERROR not in coverage.detail, coverage.detail
    # ...and a bare name that really is nowhere names every place it looked.
    missing = spk_coverage("nowhere_at_all.tm", project_root=REPO_ROOT)
    assert not missing.available
    assert str(KERNELS_DIR) in missing.detail, missing.detail
    print(f"    [bare name] {bare} -> {coverage.detail[:60]}")


def test_a_path_symbol_count_mismatch_is_refused_not_truncated():
    """SPICE refuses the whole file; `zip` would truncate and report green."""
    from gui_core.body_cards import spk_coverage

    _require(JUSTITIA_SPK, LSK)
    with tempfile.TemporaryDirectory() as scratch:
        mk = _write_metakernel(
            scratch, "mismatch.tm",
            [str(KERNELS_DIR), str(JUSTITIA_DIR), "/somewhere/else"],
            ["KERNELS", "BODY"],
            ["$KERNELS/lsk/naif0012.tls", "$BODY/ephemeris.spk"])
        coverage = spk_coverage(mk, naif_id=JUSTITIA_ID,
                                project_root=REPO_ROOT)
    assert not coverage.available, (
        "a metakernel SPICE would refuse with PATHMISMATCH was reported as "
        f"covering {coverage.start_utc}..{coverage.end_utc}")
    assert "PATHMISMATCH" in coverage.detail, coverage.detail
    print("    [pathmismatch] refused, as furnsh would")


def test_a_redeclared_pool_name_replaces_rather_than_appends():
    """`=` replaces in SPICE; `+=` appends. Getting it backwards misaligns
    PATH_SYMBOLS against PATH_VALUES and resolves every member elsewhere."""
    from gui_core.body_cards import metakernel_members

    _require(JUSTITIA_SPK)
    with tempfile.TemporaryDirectory() as scratch:
        mk = Path(scratch) / "redeclared.tm"
        mk.write_text(
            "KPL/MK\n\\begindata\n"
            "  PATH_VALUES  = ( '/stale/one', '/stale/two' )\n"
            f"  PATH_VALUES  = ( '{KERNELS_DIR}', '{JUSTITIA_DIR}' )\n"
            "  PATH_SYMBOLS = ( 'KERNELS', 'BODY' )\n"
            "  KERNELS_TO_LOAD = ( '$BODY/ephemeris.spk' )\n"
            "\\begintext\n")
        members = metakernel_members(mk, REPO_ROOT)
    resolved = [str(p) for _t, p in members.kernels]
    assert resolved == [str(JUSTITIA_SPK)], (
        f"the re-declared PATH_VALUES was appended, not replaced: "
        f"{resolved or members.missing}")
    print("    [redeclare] the second declaration wins, as SPICE does it")


def test_an_empty_member_entry_is_not_reported_as_a_loaded_kernel():
    """`Path("")` is `.`, and the project root exists."""
    from gui_core.body_cards import metakernel_members

    with tempfile.TemporaryDirectory() as scratch:
        mk = Path(scratch) / "empty_entry.tm"
        mk.write_text(
            "KPL/MK\n\\begindata\n"
            f"  PATH_VALUES  = ( '{KERNELS_DIR}' )\n"
            "  PATH_SYMBOLS = ( 'KERNELS' )\n"
            "  KERNELS_TO_LOAD = ( '' '+' )\n"
            "\\begintext\n")
        members = metakernel_members(mk, REPO_ROOT)
    assert not members.kernels, (
        f"an empty KERNELS_TO_LOAD entry was recorded as a real kernel: "
        f"{[str(p) for _t, p in members.kernels]}")
    assert members.missing, "and it was not reported at all"
    print("    [empty entry] reported, not resolved to the project root")


def test_a_run_window_in_a_coverage_gap_is_not_reported_as_covered():
    """The reason the intervals are carried and not just their envelope."""
    from gui_core.body_cards import SpkCoverage

    gapped = SpkCoverage(
        True, "2034-10-01T00:00:00", "2035-09-01T00:00:00", 335.0,
        detail="two arcs", bodies=[20000269],
        intervals=[("2034-10-01T00:00:00", "2034-11-01T00:00:00"),
                   ("2035-08-01T00:00:00", "2035-09-01T00:00:00")])
    assert gapped.covers("2034-10-05T00:00:00",
                         "2034-10-20T00:00:00") is True
    assert gapped.covers("2034-12-01T00:00:00",
                         "2034-12-05T00:00:00") is False, (
        "a window sitting in the hole between two arcs was reported as inside "
        "the kernel; the engine would serve every one of those epochs from "
        "Keplerian elements, silently")
    assert gapped.covers("2034-10-20T00:00:00",
                         "2035-08-05T00:00:00") is False, (
        "a window spanning the hole was reported as covered")
    # An envelope-only SpkCoverage -- what a caller that never saw the
    # intervals builds -- must still answer the way it always did.
    envelope = SpkCoverage(True, "2034-10-01T00:00:00",
                           "2035-09-01T00:00:00", 335.0)
    assert envelope.covers("2034-12-01T00:00:00",
                           "2034-12-05T00:00:00") is True
    print("    [gaps] a window in the hole is outside the kernel")


def test_a_semicolon_suffixed_naif_id_is_accepted():
    """HORIZONS returns small-body designations with a trailing `;`, which
    two other sites in this codebase already strip."""
    from gui_core.body_cards import spk_coverage

    _require(JUSTITIA_MK, JUSTITIA_SPK, LSK, DE440S)
    coverage = spk_coverage(JUSTITIA_MK, naif_id=f"{JUSTITIA_ID};",
                            project_root=REPO_ROOT)
    assert coverage.available, coverage.detail
    assert coverage.bodies == (int(JUSTITIA_ID),), coverage.bodies
    print("    [designation] a trailing ';' is stripped, as elsewhere")


def test_a_hostile_or_absurd_metakernel_is_refused_quickly_and_quietly():
    """It is a parser reached from a Qt slot, on a file the user picked.

    Three ways a bad file could hurt: an unbounded read, quadratic
    backtracking on an unterminated pool string, and an exception escaping
    into the slot (a PyQt6 qFatal, SIGABRT, taking unsaved edits with it).
    """
    import time

    from gui_core.body_cards import metakernel_members, spk_coverage

    with tempfile.TemporaryDirectory() as scratch:
        # Too large to be a list of file names: refused, not read.
        huge = Path(scratch) / "huge.tm"
        huge.write_text("KPL/MK\n\\begindata\n  K = ( '"
                        + "a" * (5 * 1024 * 1024) + "\n")
        started = time.time()
        members = metakernel_members(huge, REPO_ROOT)
        elapsed = time.time() - started
        assert members.error and "not one" in members.error, members.error
        assert elapsed < 2.0, f"refusing an oversized file took {elapsed:.1f} s"

        # Pathological but within the bound: parsed, and fast.
        awkward = Path(scratch) / "awkward.tm"
        awkward.write_text("KPL/MK\n\\begindata\n  KERNELS_TO_LOAD = ( '"
                           + "a" * (1024 * 1024) + "\n")
        started = time.time()
        metakernel_members(awkward, REPO_ROOT)
        elapsed = time.time() - started
        assert elapsed < 5.0, f"1 MB of unterminated string took {elapsed:.1f} s"

        # Members that cannot become a path at all, and one naming a real
        # file outside the project. Nothing may raise, the unusable ones must
        # be reported, and the readable-but-not-a-kernel one must never be
        # offered as an SPK -- `getfat` is the gate, and it reads a header,
        # never the contents.
        nasty = Path(scratch) / "nasty.tm"
        nasty.write_text(
            "KPL/MK\n\\begindata\n"
            "  PATH_VALUES  = ( '/nowhere' )\n"
            "  PATH_SYMBOLS = ( 'X' )\n"
            "  KERNELS_TO_LOAD = ( '../../../../etc/passwd'\n"
            "                      'a\\x00b.bsp'\n"
            f"                      '{'d' * 4000}.bsp' )\n"
            "\\begintext\n")
        result = metakernel_members(nasty, REPO_ROOT)
        assert result.missing, "the unusable members were not reported"
        assert not result.of_type("DAF", "SPK"), (
            f"a non-kernel was offered as an SPK: {result.of_type('DAF', 'SPK')}")
        coverage = spk_coverage(nasty, project_root=REPO_ROOT)
        assert not coverage.available, coverage.detail
        assert SPICE_ARCHITECTURE_ERROR not in coverage.detail, coverage.detail
        assert "root:" not in coverage.detail, (
            "a member file's CONTENTS reached the message")
    print("    [hostile] oversized refused, 1 MB parsed fast, nothing raised")


# --------------------------------------------------------------------------
# 4. The binary SPK is unchanged
# --------------------------------------------------------------------------

def test_the_binary_spk_answer_matches_a_direct_spice_read():
    """The regression arm: `fetch_spk`'s path, against SPICE itself."""
    from gui_core.body_cards import spk_coverage

    try:
        import spiceypy
    except ImportError as exc:                              # pragma: no cover
        raise Skip(f"spiceypy is unavailable here ({exc})")
    lsk = _require(LSK)
    spiceypy.furnsh(str(lsk))
    checked = 0
    cache = REPO_ROOT / "data" / "horizons_cache" / "bodies"
    for spk in sorted(cache.glob("*/ephemeris.spk")):
        ids = spiceypy.spkobj(str(spk))
        starts, ends = [], []
        for index in range(len(ids)):
            window = spiceypy.stypes.SPICEDOUBLE_CELL(2000)
            spiceypy.spkcov(str(spk), int(ids[index]), window)
            for i in range(spiceypy.wncard(window)):
                low, high = spiceypy.wnfetd(window, i)
                starts.append(low)
                ends.append(high)
        if not starts:
            continue
        expected_start = spiceypy.et2utc(min(starts), "ISOC", 0)
        expected_end = spiceypy.et2utc(max(ends), "ISOC", 0)
        coverage = spk_coverage(spk, project_root=REPO_ROOT)
        assert coverage.available, coverage.detail
        assert coverage.start_utc == expected_start, (spk.parent.name,
                                                      coverage.start_utc,
                                                      expected_start)
        assert coverage.end_utc == expected_end, (spk.parent.name,
                                                  coverage.end_utc,
                                                  expected_end)
        assert coverage.detail == spk.name, coverage.detail
        checked += 1
    if not checked:
        raise Skip("no cached SPK files to compare against")
    print(f"    [binary] {checked} SPKs match a direct spkcov, detail "
          f"unchanged")


def test_a_missing_file_and_a_blank_field_are_still_what_they_were():
    from gui_core.body_cards import spk_coverage

    blank = spk_coverage("")
    assert not blank.available and blank.detail == "no kernel selected", \
        blank.detail
    absent = spk_coverage(REPO_ROOT / "does" / "not" / "exist.bsp")
    assert not absent.available, absent.detail
    assert "is not on disk" in absent.detail, absent.detail
    print("    [edges] blank and missing unchanged")


# --------------------------------------------------------------------------
# 5. Nothing global moves
# --------------------------------------------------------------------------

def test_the_check_never_changes_the_working_directory():
    """A GUI process left chdir-ed opens its file dialogs in the wrong place.

    The committed metakernels' PATH_VALUES are relative to the repository
    root, so 'chdir there, furnsh, chdir back' is the obvious implementation
    and the one this asserts against: an exception between the two chdirs
    leaves the process somewhere else for the rest of its life.
    """
    from gui_core.body_cards import spk_coverage

    _require(JUSTITIA_MK, JUSTITIA_SPK, LSK, DE440S)
    before = os.getcwd()
    with tempfile.TemporaryDirectory() as scratch:
        os.chdir(scratch)
        try:
            elsewhere = os.getcwd()
            for target in (JUSTITIA_MK, JUSTITIA_SPK, "", "no/such/file.tm",
                           REPO_ROOT / "data" / "kernels" / "lsk"
                           / "naif0012.tls", JUSTITIA_DIR):
                spk_coverage(target, naif_id=JUSTITIA_ID,
                             project_root=REPO_ROOT)
                assert os.getcwd() == elsewhere, (
                    f"spk_coverage({target}) left the process in "
                    f"{os.getcwd()}")
        finally:
            os.chdir(before)
    assert os.getcwd() == before
    print("    [cwd] unchanged across six calls, including the failures")


def test_a_relative_kernel_path_resolves_from_the_project_root():
    """The card writes the path the cache wrote, and the cache writes it
    relative. Resolving it against the working directory answers a question
    about where the GUI was launched from."""
    from gui_core.body_cards import spk_coverage

    _require(JUSTITIA_MK, JUSTITIA_SPK, LSK, DE440S)
    relative = str(JUSTITIA_MK.relative_to(REPO_ROOT))
    before = os.getcwd()
    with tempfile.TemporaryDirectory() as scratch:
        os.chdir(scratch)
        try:
            coverage = spk_coverage(relative, naif_id=JUSTITIA_ID,
                                    project_root=REPO_ROOT)
        finally:
            os.chdir(before)
    assert coverage.available, (
        f"a relative kernel path was not resolved: {coverage.detail}")
    print(f"    [relative] {relative} resolved from the project root")


def test_the_check_does_not_furnish_the_metakernel():
    """Only a leapseconds kernel may enter this process's pool.

    `furnsh` is global and permanent. A GUI that furnished a metakernel on
    every press of Check coverage would accumulate a planetary ephemeris per
    press and would answer a later question with an earlier body's kernels
    still loaded.
    """
    from gui_core.body_cards import spk_coverage

    try:
        import spiceypy
    except ImportError as exc:                              # pragma: no cover
        raise Skip(f"spiceypy is unavailable here ({exc})")
    _require(JUSTITIA_MK, JUSTITIA_SPK, LSK, DE440S)
    spk_coverage(JUSTITIA_MK, naif_id=JUSTITIA_ID, project_root=REPO_ROOT)
    before = spiceypy.ktotal("ALL")
    for _ in range(3):
        spk_coverage(JUSTITIA_MK, naif_id=JUSTITIA_ID, project_root=REPO_ROOT)
    after = spiceypy.ktotal("ALL")
    assert after == before, (
        f"the kernel pool grew from {before} to {after} over three checks")
    assert spiceypy.ktotal("META") == 0, (
        f"{spiceypy.ktotal('META')} metakernel(s) were furnished")
    assert spiceypy.ktotal("SPK") == 0, (
        f"{spiceypy.ktotal('SPK')} SPK(s) were furnished")
    print(f"    [pool] {after} kernels loaded, 0 metakernels, 0 SPKs")


# --------------------------------------------------------------------------
# 6. The card
# --------------------------------------------------------------------------

def test_the_card_checks_coverage_with_the_id_and_the_project_root():
    """End to end, on the real card, offscreen."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        import PyQt6
    except ImportError as exc:                              # pragma: no cover
        raise Skip(f"PyQt6 is unavailable here ({exc})")
    plug = Path(PyQt6.__file__).parent / "Qt6" / "plugins"
    os.environ.setdefault("QT_PLUGIN_PATH", str(plug))
    os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(plug / "platforms"))
    from PyQt6.QtCore import QSettings
    from PyQt6.QtWidgets import QApplication

    _require(JUSTITIA_MK, JUSTITIA_SPK, LSK, DE440S)
    before = os.getcwd()
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

        manager.set_value("generic_body.spice_metakernel",
                          str(JUSTITIA_MK.relative_to(REPO_ROOT)))
        manager.set_value("generic_body.naif_id", JUSTITIA_ID)
        manager.set_value("generic_body.spin.period_hours", 33.128)
        manager.set_value("simulation.start_time", "2035-08-25T00:00:00")
        manager.set_value("simulation.duration_days", 3.75)
        manager.set_value("thermal.equilibration_orbits", 40)
        card._check_coverage()
        app.processEvents()
        note = card.coverage_note.text()

        assert note, "Check coverage said nothing at all"
        assert SPICE_ARCHITECTURE_ERROR not in note, note
        assert "Coverage UNKNOWN" not in note, note
        assert "covers" in note, note
        assert "including spin-up" in note, note

        # ...and the wrong ID on the same file is refused, so the check above
        # is not passing because the card ignores what it is given.
        manager.set_value("generic_body.naif_id", "2000269")
        card._check_coverage()
        app.processEvents()
        wrong = card.coverage_note.text()
        assert "Coverage UNKNOWN" in wrong, wrong
        assert "2000269" in wrong, wrong
    assert os.getcwd() == before, os.getcwd()
    print("    [card] verdict shown for a metakernel, wrong id still refused")


CHECKS = [
    test_the_committed_justitia_metakernel_reports_its_own_coverage,
    test_every_cached_metakernel_is_readable_and_none_blames_spice,
    test_a_continued_path_is_reassembled_not_truncated,
    test_a_missing_member_is_named_and_not_blamed_on_the_kernel,
    test_a_cloud_placeholder_member_is_called_a_cloud_placeholder,
    test_a_metakernel_with_no_spk_says_so,
    test_an_unresolved_path_symbol_says_which_symbol,
    test_a_text_kernel_is_identified_rather_than_called_corrupt,
    test_the_five_failures_are_five_different_sentences,
    test_a_naif_id_absent_from_the_kernels_is_reported_as_that,
    test_the_metakernel_answer_is_not_de440s_three_centuries,
    test_an_unparsable_naif_id_is_refused_rather_than_ignored,
    test_a_semicolon_suffixed_naif_id_is_accepted,
    test_a_bare_kernel_name_resolves_the_way_the_engine_resolves_it,
    test_a_path_symbol_count_mismatch_is_refused_not_truncated,
    test_a_redeclared_pool_name_replaces_rather_than_appends,
    test_an_empty_member_entry_is_not_reported_as_a_loaded_kernel,
    test_a_run_window_in_a_coverage_gap_is_not_reported_as_covered,
    test_a_hostile_or_absurd_metakernel_is_refused_quickly_and_quietly,
    test_the_binary_spk_answer_matches_a_direct_spice_read,
    test_a_missing_file_and_a_blank_field_are_still_what_they_were,
    test_the_check_never_changes_the_working_directory,
    test_a_relative_kernel_path_resolves_from_the_project_root,
    test_the_check_does_not_furnish_the_metakernel,
    test_the_card_checks_coverage_with_the_id_and_the_project_root,
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
