#!/usr/bin/env python3
"""The curated tier serves a body the live services correctly refuse.

**Why this file exists.** Pressing "Fetch kernel from HORIZONS" for (101955)
Bennu produced a message that was true, complete, and a dead end::

    HORIZONS was reached and REFUSED to build a kernel for 101955 Bennu.
    ... SPK creation is not available for pre-computed objects in the major
    body index

Bennu's HORIZONS record is ``*SPECIAL*``, served from a pre-computed kernel,
and no designation form changes that. DAMIT has no Bennu either -- it publishes
lightcurve inversions and Bennu's shape is a radar model. Both services were
right; the user was still stuck, with an 80.7 MB kernel and an 89 kB shape
model already in this repository and no way to reach either from the card.

The curated tier is that route. What has to be true of it:

1. **Order.** cache -> curated registry -> live service -> a failure that says
   which. A curated body must NEVER reach HORIZONS or DAMIT, and a
   non-curated body's path must be exactly what it was.
2. **Offline.** With the kernel already fetched, the whole flow must complete
   with no network call of any kind -- not even a HEAD request. That is not a
   nicety: ``fetch_kernels.download`` asks for a Content-Length BEFORE it
   looks at the local file, so "just call the downloader to see if it is
   there" would put a round trip in front of a body that needs none.
3. **One copy of the pin.** The download URL and its sha256 live in
   ``scripts/fetch_kernels.py``. A second copy in the GUI would be a hash that
   goes stale in one of two places, silently, which is what a pin exists to
   prevent.
4. **Nothing invented.** Every field of a registry entry has to resolve to a
   file or a citation that is actually here.

Run directly::

    python3 src/python/tests/test_gui_curated_sources.py
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "python"))
sys.path.insert(0, str(REPO_ROOT / "gui"))


class Skip(Exception):
    """A check that cannot run here, as distinct from one that failed."""


BENNU = "101955 Bennu (1999 RQ36)"


def _run_config(start="2026-03-01T00:00:00"):
    """A Bennu-shaped run window, with the spin-up reach-back that matters."""
    return {
        "simulation": {"start_time": start, "duration_days": 0.358},
        "thermal": {"equilibration_orbits": 40.0},
        "generic_body": {"spin": {"period_hours": 4.29746}},
    }


def _isolated_root(tmp, members=()):
    """A scratch project root holding `scripts/`, the metakernel and `members`.

    A REAL project root rather than an environment override, because
    ``bennu.tm``'s own ``PATH_VALUES`` is ``data/kernels`` relative to the
    project root -- so pointing ``PTM_SPICE_KERNELS`` somewhere else moves
    where the ``.tm`` is FOUND without moving where its members resolve. The
    first version of this file made that mistake and its "kernel missing"
    fixture silently resolved every member back to the developer's real tree,
    so it could not have failed.

    `members` are basenames copied out of the real `data/kernels` tree.
    """
    root = Path(tmp) / "root"
    (root / "scripts").mkdir(parents=True)
    shutil.copy(REPO_ROOT / "scripts" / "fetch_kernels.py",
                root / "scripts" / "fetch_kernels.py")
    kernels = root / "data" / "kernels"
    kernels.mkdir(parents=True)
    shutil.copy(REPO_ROOT / "data" / "kernels" / "bennu.tm",
                kernels / "bennu.tm")
    # The SUBDIRECTORY comes from the pinned table, not from a glob. `rglob`
    # does not follow symlinked directories, and a development checkout may
    # well have `data/kernels/{lsk,pck,spk}` symlinked to a shared tree to
    # avoid a second 1.2 GB copy -- against which every arm of this file
    # silently SKIPPED.
    from gui_core.curated_sources import fetch_kernels_module

    module = fetch_kernels_module(REPO_ROOT)
    where = {name: subdir for subdir, name, _u, _h
             in list(module.KERNELS_CORE)
             + [k for row in module.BODY_KERNELS.values() for k in row]}
    real = REPO_ROOT / "data" / "kernels"
    for name in members:
        subdir = where.get(name)
        if subdir is None:
            raise Skip(f"{name} is not in the pinned kernel table")
        source = real / subdir / name
        if not source.is_file():
            raise Skip(f"{source} is not fetched in this checkout")
        target = kernels / subdir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source, target)
    return root


ALL_BENNU_MEMBERS = ("naif0012.tls", "pck00011.tpc", "de440s.bsp",
                     "sb-101955-118_long.bsp")


class ExplodingHorizons:
    """A HORIZONS client that fails the test if it is ever asked anything."""

    def __init__(self, log):
        self._log = log

    def download_spk(self, *args, **kwargs):
        self._log.append(args)
        raise AssertionError(
            "HORIZONS was asked for a kernel for a CURATED body. It refuses "
            "this body permanently; asking is a slower route to a message the "
            "user can do nothing with.")


class ExplodingDamit:
    def __init__(self, log):
        self._log = log

    def search_asteroid(self, *args, **kwargs):
        self._log.append(args)
        raise AssertionError(
            "DAMIT was searched for a CURATED body. DAMIT publishes "
            "lightcurve inversions and this body's shape is a radar product, "
            "so its 'no shape model' is a true answer to the wrong question.")


# --------------------------------------------------------------------------
# 1. The table names only things that are here
# --------------------------------------------------------------------------

def test_every_curated_entry_resolves_to_files_that_exist():
    """A curated entry is a promise that two artifacts are reachable.

    An entry naming a metakernel that does not ship, or a shape that is not
    on disk, would be the unbacked button the body-first screen exists to
    remove -- worse than the service refusal it replaced, because it looks
    like it worked.
    """
    from gui_core.curated_sources import (CURATED_BODIES, curated_shape_path,
                                          fetch_kernels_module)

    assert CURATED_BODIES, "the curated registry is empty"
    module = fetch_kernels_module(REPO_ROOT)
    for entry in CURATED_BODIES:
        selector = entry.ephemeris.fetch_bodies
        assert selector in module.BODY_KERNELS, (
            f"{entry.name} names --bodies {selector!r}, which "
            f"scripts/fetch_kernels.py does not fetch (it has "
            f"{sorted(module.BODY_KERNELS)}). The card would offer a download "
            f"that cannot happen.")
        assert module.BODY_KERNELS[selector], (
            f"--bodies {selector} fetches no kernels at all")

        metakernel = REPO_ROOT / "data" / "kernels" / entry.ephemeris.metakernel
        assert metakernel.is_file(), (
            f"{entry.name}'s metakernel {metakernel} does not ship. The "
            f"registry points generic_body.spice_metakernel at it.")

        shape_path, present = curated_shape_path(entry, REPO_ROOT)
        assert present, (
            f"{entry.name}'s curated shape {shape_path} is not on disk")

        if entry.shape.provenance_relpath:
            sidecar = REPO_ROOT / entry.shape.provenance_relpath
            assert sidecar.is_file(), (
                f"{entry.name}'s shape provenance {sidecar} is not on disk; "
                f"the card cites it by path.")
        assert entry.why_curated.strip(), (
            f"{entry.name} has no reason recorded for being curated. That "
            f"sentence is what the user reads INSTEAD of the service's "
            f"refusal; without it the tier is unexplained magic.")
        assert entry.ephemeris.reference.strip(), (
            f"{entry.name}'s ephemeris carries no citation")
    print(f"    [registry] {len(CURATED_BODIES)} entr"
          f"{'y' if len(CURATED_BODIES) == 1 else 'ies'}: every metakernel, "
          f"shape, sidecar and --bodies selector resolves")


def test_the_curated_entry_agrees_with_the_shipped_configuration():
    """The registry and ``examples/quickstart_asteroid_bennu.yaml`` are one body.

    The shipped example is the verified artifact; the registry is a second
    statement of the same facts, and two statements of one fact are a place
    for them to diverge. If somebody retunes the example's mesh path or NAIF
    id and not this table, a curated card and a shipped example would model
    two different Bennus.
    """
    import yaml

    from gui_core.curated_sources import curated_body

    example = REPO_ROOT / "examples" / "quickstart_asteroid_bennu.yaml"
    if not example.is_file():
        raise Skip(f"{example} is not in this checkout")
    doc = yaml.safe_load(example.read_text()) or {}
    generic = doc.get("generic_body") or {}
    entry = curated_body(str(generic.get("name", "")))
    assert entry is not None, (
        f"{example.name} models {generic.get('name')!r}, which the curated "
        f"registry does not recognize -- so the card cannot reproduce the "
        f"example it ships beside.")

    assert str(generic.get("naif_id")) == entry.ephemeris.naif_id, (
        f"{example.name} uses NAIF id {generic.get('naif_id')!r}; the "
        f"registry says {entry.ephemeris.naif_id!r}. Small-body SPK ids exist "
        f"in two conventions and only the kernel's own works.")
    assert str(generic.get("spice_metakernel")) == entry.ephemeris.metakernel
    shape = generic.get("shape") or {}
    assert str(shape.get("mesh_file")) == entry.shape.relpath, (
        f"{example.name} loads {shape.get('mesh_file')!r}; the registry "
        f"offers {entry.shape.relpath!r}.")
    assert float(shape.get("mesh_scale")) == entry.shape.mesh_scale
    assert abs(float(shape.get("radius_km")) - entry.shape.radius_km) < 1e-9
    print(f"    [parity] the registry and {example.name} agree on the "
          f"metakernel, the NAIF id {entry.ephemeris.naif_id}, the mesh path, "
          f"the scale and the radius")


def test_the_pinned_hash_is_not_copied_into_the_gui():
    """The sha256 exists once, in the script that downloads with it.

    A pin is only a pin while there is one of it. This is the check that
    keeps the convenience of a registry from quietly becoming a second table
    of URLs and hashes that nobody updates together.
    """
    from gui_core.curated_sources import fetch_kernels_module

    module = fetch_kernels_module(REPO_ROOT)
    hashes = {sha for entries in module.BODY_KERNELS.values()
              for (_sub, _name, _url, sha) in entries if sha}
    urls = {url for entries in module.BODY_KERNELS.values()
            for (_sub, _name, url, _sha) in entries}
    assert hashes, "no pinned hashes to check"

    offenders = []
    for path in sorted((REPO_ROOT / "gui").rglob("*.py")):
        text = path.read_text(errors="replace")
        for value in hashes | urls:
            if value in text:
                offenders.append(f"{path.relative_to(REPO_ROOT)} contains "
                                 f"{value[:24]}...")
    assert not offenders, (
        "the GUI carries its own copy of a pinned download URL or sha256:\n  "
        + "\n  ".join(offenders)
        + "\nThe curated registry must READ scripts/fetch_kernels.py, not "
          "restate it -- a hash with two copies goes stale in one of them.")
    print(f"    [pin] {len(hashes)} pinned sha256 and {len(urls)} URL(s) exist "
          f"only in scripts/fetch_kernels.py; the GUI imports the table")


# --------------------------------------------------------------------------
# 2. Resolution order, with the negative controls that matter
# --------------------------------------------------------------------------

def test_a_curated_body_never_reaches_horizons():
    """The negative control. HORIZONS is not asked, at all, ever.

    Not "asked and the failure handled". Not asked. The client factory here
    raises on any call, so a fetch that touched it fails this check rather
    than passing with a swallowed exception.
    """
    from gui_core.body_fetch import fetch_spk

    calls = []
    result = fetch_spk(_run_config(), REPO_ROOT, name=BENNU, naif_id="101955",
                       client_factory=lambda: ExplodingHorizons(calls))
    assert not calls, "HORIZONS was called for a curated body"
    assert result.kind == "spk"
    assert "HORIZONS was not asked" in result.message, (
        f"the result does not SAY that HORIZONS was skipped, so a user has no "
        f"way to know which source answered: {result.message[:200]}")
    print("    [order] a curated body's SPK fetch never constructs a HORIZONS "
          "request; the result says so in as many words")


def test_a_curated_body_never_reaches_damit():
    from gui_core.body_fetch import fetch_damit_shape

    calls = []
    result = fetch_damit_shape(REPO_ROOT, name=BENNU, naif_id="2101955",
                               client_factory=lambda: ExplodingDamit(calls))
    assert not calls, "DAMIT was searched for a curated body"
    assert result.ok, f"the curated shape was not offered: {result.message}"
    assert "DAMIT was NOT searched" in " ".join(result.warnings)
    print("    [order] a curated body's shape comes from the registry and "
          "DAMIT is not searched")


def test_a_non_curated_body_is_unchanged_and_still_reaches_the_service():
    """The other half of the control: nothing else moved.

    A tier that also changed the path for every body it does not cover would
    be a change to seven cached asteroids disguised as a change to one.
    """
    from gui_core.body_fetch import fetch_damit_shape, fetch_spk
    from gui_core.curated_sources import curated_body

    for other in ("623 Chimaera (A907 BC)", "269 Justitia (A887 SA)",
                  "23871 Ousha (1998 RC76)"):
        assert curated_body(other) is None, (
            f"{other} matched the curated registry. Only bodies whose "
            f"ephemeris AND shape have been verified against an archive "
            f"belong in it.")

    asked = []

    class Recording:
        def download_spk(self, command, start, stop, dest, **kwargs):
            asked.append(command)
            raise RuntimeError("stub: no kernel produced")

        def search_asteroid(self, number):
            asked.append(number)
            return []

    with tempfile.TemporaryDirectory() as tmp:
        fetch_spk(_run_config(), tmp, name="623 Chimaera (A907 BC)",
                  naif_id="20000623", client_factory=lambda: Recording())
        assert asked, ("a non-curated body no longer reaches HORIZONS at all; "
                       "the curated tier was supposed to add a route, not "
                       "remove one")
        before = len(asked)
        fetch_damit_shape(tmp, name="623 Chimaera (A907 BC)",
                          naif_id="20000623",
                          client_factory=lambda: Recording())
        assert len(asked) > before, (
            "a non-curated body no longer reaches DAMIT")
    print(f"    [control] a non-curated body still reaches both services "
          f"(HORIZONS asked for {asked[0]!r}, DAMIT for {asked[-1]!r})")


def test_the_curated_source_wins_over_a_superseded_cached_kernel():
    """Bennu's body cache holds solution 76. The card must not wire it.

    ``data/horizons_cache/bodies/101955_Bennu_(1999_RQ36)/ephemeris.spk`` is
    JPL solution 76 (2015-2023), kept for provenance and superseded on
    2026-08-27. Left to the ordinary cache-first path a 2019 run would wire
    it, silently, and a 2026 run would fall off the end of it into the
    engine's unlogged Keplerian fallback -- 0.092 AU and up to 23.9 K wrong.

    The curated entry wins, AND the superseded file is named rather than left
    on disk with nothing said about it.
    """
    from gui_core.body_fetch import fetch_spk

    cached = (REPO_ROOT / "data" / "horizons_cache" / "bodies"
              / "101955_Bennu_(1999_RQ36)" / "ephemeris.spk")
    if not cached.is_file():
        raise Skip(f"{cached} is not in this checkout")

    # 2019 is INSIDE the superseded kernel's coverage, so this is the case
    # where the plain cache-first rule would have taken it.
    result = fetch_spk(_run_config("2019-01-10T00:00:00"), REPO_ROOT,
                       name=BENNU, client_factory=lambda: ExplodingHorizons([]))
    assert result.ok, f"the curated ephemeris was not wired: {result.message}"
    wired = result.values["generic_body.spice_metakernel"]
    assert wired == "bennu.tm", (
        f"the card wired {wired!r}, not the curated metakernel. Inside "
        f"2015-2023 the superseded cached kernel would also have 'covered' "
        f"the run, which is exactly why the order matters here.")
    disclosure = " ".join(result.warnings)
    assert "ephemeris.spk" in disclosure and "NOT USED" in disclosure, (
        f"the superseded cached kernel is on disk and the result does not "
        f"mention it. Two kernels for one body and no statement about which "
        f"ran is the silence this project keeps paying for. Warnings were: "
        f"{result.warnings}")
    print("    [supersede] inside 2015-2023, where the cached solution-76 "
          "kernel WOULD have passed a coverage check, the curated solution "
          "118 is wired and the cached file is named as NOT USED")


# --------------------------------------------------------------------------
# 3. Detection in both states, and the offer
# --------------------------------------------------------------------------

def test_the_fetched_kernel_is_detected_in_both_states():
    """Present and absent are two answers, and neither needs a network.

    Measured both ways in an isolated project root, so "present" is a file
    this check put there rather than an assumption about the machine.
    """
    from gui_core.curated_sources import curated_body, curated_kernel_status

    entry = curated_body(BENNU)
    with tempfile.TemporaryDirectory() as tmp:
        root = _isolated_root(tmp, members=("naif0012.tls", "pck00011.tpc",
                                            "de440s.bsp"))
        absent = curated_kernel_status(entry, root)
        assert not absent.ready, "an unfetched kernel was reported as ready"
        assert [k.name for k in absent.missing] == \
            [entry.ephemeris.spk_filename], absent.summary()
        assert absent.download_bytes > 1_000_000, (
            "the size of the missing download is not disclosed, so the card "
            "would ask for consent without saying to what")
        assert absent.can_download

        spk = root / "data" / "kernels" / "spk" / entry.ephemeris.spk_filename
        spk.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO_ROOT / "data" / "kernels" / "spk"
                    / entry.ephemeris.spk_filename, spk)
        present = curated_kernel_status(entry, root)
        assert present.ready, (
            f"a fetched kernel was not detected: {present.summary()}")
        assert not present.missing
    print(f"    [detect] absent -> not ready, "
          f"{absent.download_bytes / (1024 * 1024):.1f} MB disclosed; "
          f"present -> ready. Neither answer touches the network.")


def test_readiness_covers_every_kernel_the_metakernel_names():
    """"Ready" is `furnsh` succeeding, not the body's own SPK being present.

    MEASURED DEFECT, 2026-08-28. `bennu.tm` loads four kernels — leap
    seconds, planetary constants, `de440s.bsp` and the body SPK — and only
    the last is in `BODY_KERNELS["bennu"]`. A status built from that row
    reported `ready=True` and *"bennu.tm and its 1 pinned kernel are on this
    disk"*, so the card printed "already on this disk — no download and no
    network needed" on a tree where three of the four were missing, and the
    coverage check then refused. `furnsh` stops at the FIRST member it cannot
    open, so those three are not optional extras.

    Reproduced in an isolated project root: body SPK present, core kernels
    absent. Fails against the row-based status, passes against the
    metakernel-based one.
    """
    from gui_core.curated_sources import curated_body, curated_kernel_status

    entry = curated_body(BENNU)
    with tempfile.TemporaryDirectory() as tmp:
        root = _isolated_root(tmp, members=(entry.ephemeris.spk_filename,))
        status = curated_kernel_status(entry, root)
        assert not status.ready, (
            f"the body's own SPK is present and three shared kernels are "
            f"not, and this was reported as ready: {status.summary()}")
        missing = {k.name for k in status.missing}
        assert missing == {"naif0012.tls", "pck00011.tpc", "de440s.bsp"}, (
            f"the missing members were not identified: {missing}")
        assert status.can_download, (
            "all three are in KERNELS_CORE and are pinned, so a download "
            "fixes this and must be offered")
        assert all(k.url and k.sha256 for k in status.fetchable), (
            "a fetchable member reached the offer without a URL or a hash")

        # And the whole set present IS ready.
        full = _isolated_root(str(Path(tmp) / "full"),
                              members=ALL_BENNU_MEMBERS)
        assert curated_kernel_status(entry, full).ready
    print("    [ready] readiness follows bennu.tm's own KERNELS_TO_LOAD: the "
          "body SPK alone is NOT ready, and the three missing shared kernels "
          "are named and offered")


def test_a_download_is_not_offered_for_a_state_it_cannot_fix():
    """A download button that spends bandwidth and changes nothing.

    Three states are "not ready" and only one of them is a download. The
    first draft offered one for all three, including a modal reading
    *"Download ... now? 0 B"* when the metakernel itself was the missing
    piece. A cloud placeholder is worse than useless: `fetch_kernels.download`
    finds a full-size file, short-circuits to `verify`, and HASHES it — which
    hydrates the 80 MB the message told the user not to touch.
    """
    from gui_core.body_fetch import fetch_spk
    from gui_core.curated_sources import curated_body, curated_kernel_status

    entry = curated_body(BENNU)
    with tempfile.TemporaryDirectory() as tmp:
        # The metakernel absent: nothing to fetch, because the .tm is the
        # thing that says what to fetch.
        root = _isolated_root(tmp, members=ALL_BENNU_MEMBERS)
        (root / "data" / "kernels" / "bennu.tm").unlink()
        status = curated_kernel_status(entry, root)
        assert not status.ready
        assert not status.can_download, (
            "a download was offered for a missing metakernel, which a "
            "download does not supply — it ships with the repository")
        result = fetch_spk(_run_config(), root, name=BENNU,
                           client_factory=lambda: ExplodingHorizons([]))
        assert not result.offer, (
            f"an offer was made anyway: {result.offer[:160]}")
        assert result.offer_bytes == 0
        assert "would not fix this" in result.message, result.message[:200]
    print("    [offer] a missing metakernel is not answered with a download "
          "offer, and no '0 B' consent dialog is produced")


def test_a_missing_curated_kernel_is_an_offer_and_downloads_nothing():
    """80 MB is not something a button press may spend without asking.

    And the offer has to be a FLAG the card can branch on, not a phrase in the
    message: a caller that detected "this needs consent" by searching English
    would break the first time the wording improved.
    """
    from gui_core.body_fetch import fetch_spk
    from gui_core.curated_sources import curated_body

    with tempfile.TemporaryDirectory() as tmp:
        root = _isolated_root(tmp, members=("naif0012.tls", "pck00011.tpc",
                                            "de440s.bsp"))
        result = fetch_spk(_run_config(), root, name=BENNU,
                           client_factory=lambda: ExplodingHorizons([]))

        assert not result.ok
        assert result.offer, (
            "a missing curated kernel produced a plain failure, not an offer. "
            "The card has nothing to ask the user about, so the body stays "
            "unreachable -- which is the original defect.")
        assert result.offer_bytes > 1_000_000, (
            f"the offer discloses {result.offer_bytes} bytes")
        assert "--bodies bennu" in result.message, (
            "the offer does not name the command that does the same thing "
            "outside the window")
        assert "retried up to five times" in result.offer, (
            "the offer does not say how long a bad network can hold the "
            "window, and the download cannot be canceled from there")
        assert not result.values, "an offer wrote values into the card"
        # The one file that was missing, and only that one: `de440s.bsp` is
        # legitimately in this same directory because the fixture put it there.
        body_spk = (root / "data" / "kernels" / "spk"
                    / curated_body(BENNU).ephemeris.spk_filename)
        assert not body_spk.exists(), (
            f"{body_spk.name} was downloaded without consent")
    print(f"    [consent] a missing curated kernel is an OFFER carrying "
          f"{result.offer_bytes / (1024 * 1024):.1f} MB, the retry budget and "
          f"the `--bodies bennu` command; nothing is downloaded and no value "
          f"is written")


def test_consent_uses_the_projects_own_downloader_with_the_pinned_hash():
    """The download is ``fetch_kernels.download``'s, not a second one.

    Resume, bounded retry and the sha256 check are all in that function. A
    GUI-side reimplementation would be a second place for the hash check to
    be forgotten -- so this asserts the URL AND the pinned hash arrive at the
    downloader unchanged.
    """
    from gui_core.body_fetch import fetch_spk
    from gui_core.curated_sources import fetch_kernels_module

    module = fetch_kernels_module(REPO_ROOT)
    pinned = {name: (url, sha) for _s, name, url, sha
              in list(module.KERNELS_CORE) + module.BODY_KERNELS["bennu"]}
    seen = []

    class StubDownloader:
        @staticmethod
        def download(url, dest, sha):
            seen.append((url, dest, sha))
            print("  10.0 MB / 80.7 MB ( 12.4%)")
            Path(dest).write_bytes(b"stub")
            return True

    progress = []
    with tempfile.TemporaryDirectory() as tmp:
        # A SCRATCH project root, not REPO_ROOT. A completed download makes
        # `_curated_spk` record the install in `data/horizons_cache/
        # index.json`, which is a TRACKED file -- so running this arm against
        # the repository would leave a Bennu entry pointing at a temporary
        # directory in everybody's `git status`. It did, once, and that is how
        # `test_wiring_an_already_present_kernel_writes_no_tracked_file` and
        # the guard at the end of this file came to exist.
        root = _isolated_root(tmp, members=("naif0012.tls", "pck00011.tpc",
                                            "de440s.bsp"))
        fetch_spk(_run_config(), root, name=BENNU,
                  client_factory=lambda: ExplodingHorizons([]),
                  curated_download=True, downloader=StubDownloader,
                  progress=progress.append)

        assert len(seen) == 1, (
            f"the downloader was called {len(seen)} time(s); exactly one "
            f"member was missing")
        url, dest, sha = seen[0]
        want_url, want_sha = pinned["sb-101955-118_long.bsp"]
        assert url == want_url, f"URL changed on the way through: {url}"
        assert sha == want_sha, (
            f"the pinned sha256 did not reach the downloader ({sha!r} vs "
            f"{want_sha!r}). Without it the download is unverified.")
        # And it landed where the METAKERNEL will look for it, not merely
        # somewhere plausible: `bennu.tm` names $KERNELS/spk/<file> with
        # PATH_VALUES relative to the project root.
        expected_path = (root / "data" / "kernels" / "spk"
                         / "sb-101955-118_long.bsp")
        assert Path(dest) == expected_path, (
            f"the kernel was written to {dest}, not to {expected_path} where "
            f"furnsh will look for it")
    assert progress, ("no progress line reached the caller, so an 80 MB "
                      "download would look like a frozen window")
    print(f"    [download] the pinned URL and sha256 reach "
          f"fetch_kernels.download unchanged, the file lands where bennu.tm "
          f"resolves it, and {len(progress)} progress line(s) reach the "
          f"caller")


# --------------------------------------------------------------------------
# 4. Coverage, and the shape's provenance
# --------------------------------------------------------------------------

def test_the_bennu_metakernel_reports_its_published_coverage():
    """1900-2135, read out of the kernel rather than taken from the table.

    The registry states the coverage; this checks the STATEMENT against the
    file. A table that said 1900-2135 over a kernel that stopped in 2023
    would put a green label on the silent-fallback failure.
    """
    try:
        import spiceypy                                    # noqa: F401
    except ImportError as exc:
        raise Skip(f"spiceypy is not available here ({exc})")

    from gui_core.body_cards import spk_coverage
    from gui_core.curated_sources import curated_body, curated_kernel_status

    entry = curated_body(BENNU)
    status = curated_kernel_status(entry, REPO_ROOT)
    if not status.ready:
        raise Skip(f"Bennu's kernel is not fetched here: {status.summary()}")

    coverage = spk_coverage(entry.ephemeris.metakernel,
                            naif_id=entry.ephemeris.naif_id,
                            project_root=REPO_ROOT)
    assert coverage.available, f"coverage unreadable: {coverage.detail}"
    start_year = int(coverage.start_utc[:4])
    end_year = int(coverage.end_utc[:4])
    declared_start = int(entry.ephemeris.coverage_start[:4])
    declared_end = int(entry.ephemeris.coverage_end[:4])
    # The kernel begins 41 seconds before 1900-01-01 in UTC, so the year is
    # compared, not the timestamp -- an ET/UTC boundary is not a discrepancy.
    assert abs(start_year - declared_start) <= 1, (
        f"the registry says coverage starts {entry.ephemeris.coverage_start}; "
        f"the kernel says {coverage.start_utc}")
    assert abs(end_year - declared_end) <= 1, (
        f"the registry says coverage ends {entry.ephemeris.coverage_end}; "
        f"the kernel says {coverage.end_utc}")
    assert coverage.covers("2026-01-01T00:00:00", "2026-01-05T00:00:00")
    print(f"    [coverage] bennu.tm -> {coverage.start_utc} to "
          f"{coverage.end_utc} ({coverage.days:.0f} days) for body "
          f"{entry.ephemeris.naif_id}; the registry's "
          f"{entry.ephemeris.coverage_text()} is the same window")


def test_the_curated_shape_comes_with_its_provenance_and_its_git_status():
    """A shape that ships is a different statement from a shape that does not.

    ``.gitignore`` line 14 (``*.obj``, written for C object files) matches
    every asteroid shape model here -- PTM-303, and why a Justitia shape once
    went missing. Bennu's is force-added and TRACKED. Saying "untracked, back
    it up" over a file the repository carries would be wrong; saying "tracked"
    over one it does not would be worse. The registry records which, and the
    check confirms it against git.
    """
    import subprocess

    from gui_core.body_fetch import fetch_damit_shape
    from gui_core.curated_sources import CURATED_BODIES

    git = "/Library/Developer/CommandLineTools/usr/bin/git"
    for entry in CURATED_BODIES:
        result = fetch_damit_shape(REPO_ROOT, name=entry.name,
                                   client_factory=lambda: ExplodingDamit([]))
        assert result.ok, result.message
        assert result.values["generic_body.shape.mesh_file"] == \
            entry.shape.relpath
        assert result.values["generic_body.shape.mesh_scale"] == \
            entry.shape.mesh_scale
        blob = " ".join(result.warnings)
        assert entry.shape.archive in blob, (
            "the shape is offered without naming its archive")

        if not Path(git).exists():
            raise Skip("the CommandLineTools git is not on this machine")
        listed = subprocess.run(
            [git, "-C", str(REPO_ROOT), "ls-files", "--error-unmatch",
             entry.shape.relpath],
            capture_output=True, text=True)
        really_tracked = listed.returncode == 0
        assert really_tracked == entry.shape.tracked, (
            f"the registry says {entry.name}'s shape tracked="
            f"{entry.shape.tracked}; git says {really_tracked}. The card "
            f"tells the user whether to back this file up, and a wrong "
            f"answer either loses a file or sends them chasing one that is "
            f"already safe.")
        assert ("TRACKED" if really_tracked else "NOT tracked") in blob
    print("    [shape] the offered shape carries its archive id, its scale "
          "and a git-tracking claim that git itself agrees with")


# --------------------------------------------------------------------------
# 5. The unchanged contracts
# --------------------------------------------------------------------------

def test_the_curated_shape_brings_its_spin_and_can_still_disagree():
    """A mesh without its pole is a body pointing the wrong way, silently.

    `fetch_damit_shape` has always sent the shape and the spin together,
    because a shape model is defined in the body-fixed frame its own pole and
    period establish. The first draft of the curated path sent the mesh alone
    — and that ALSO disabled the divergence flag: `spin_consistency` compares
    `shape_values` against the card, finds no `generic_body.spin.*` in them,
    and returns `""`, which `_apply_fetch` renders as an affirmative "these
    agree". A missing warning that looks like a passed check.

    Two arms: the spin arrives, and a card carrying a different pole is
    FLAGGED rather than quietly overwritten-and-forgotten.
    """
    from gui_core.body_fetch import fetch_damit_shape, spin_consistency
    from gui_core.curated_sources import curated_body

    entry = curated_body(BENNU)
    result = fetch_damit_shape(REPO_ROOT, name=BENNU,
                               client_factory=lambda: ExplodingDamit([]))
    assert result.ok, result.message
    for key, want in (
            ("generic_body.spin.period_hours", entry.shape.spin_period_hours),
            ("generic_body.spin.pole_ra_deg", entry.shape.spin_pole_ra_deg),
            ("generic_body.spin.pole_dec_deg", entry.shape.spin_pole_dec_deg)):
        assert key in result.values, (
            f"the curated shape does not offer {key}; the mesh would be "
            f"oriented by whatever spin happens to be on the card")
        assert abs(float(result.values[key]) - want) < 1e-9

    # The divergence flag must still be able to fire. A pole 90 degrees away
    # is unambiguously a different solution.
    disagreeing = {"generic_body": {"spin": {
        "period_hours": 12.0, "pole_ra_deg": 0.0, "pole_dec_deg": 0.0}}}
    flag = spin_consistency(disagreeing, result.values)
    assert flag, (
        "a card with a 12 h period and a pole 65 degrees away was reported as "
        "agreeing with the radar model's own spin")
    assert "SPIN AND SHAPE DISAGREE" in flag

    agreeing = {"generic_body": {"spin": {
        "period_hours": entry.shape.spin_period_hours,
        "pole_ra_deg": entry.shape.spin_pole_ra_deg,
        "pole_dec_deg": entry.shape.spin_pole_dec_deg}}}
    assert spin_consistency(agreeing, result.values) == "", (
        "the matching spin was reported as a disagreement")
    print(f"    [spin] the radar shape brings period "
          f"{entry.shape.spin_period_hours:g} h and pole "
          f"({entry.shape.spin_pole_ra_deg:g}, "
          f"{entry.shape.spin_pole_dec_deg:g}), and a card holding a "
          f"different spin is still flagged")


def test_the_shipped_example_agrees_with_the_curated_spin():
    """The spin in the registry is the shipped configuration's own."""
    import yaml

    from gui_core.curated_sources import curated_body

    example = REPO_ROOT / "examples" / "quickstart_asteroid_bennu.yaml"
    if not example.is_file():
        raise Skip(f"{example} is not in this checkout")
    spin = (((yaml.safe_load(example.read_text()) or {}).get("generic_body")
             or {}).get("spin") or {})
    shape = curated_body(BENNU).shape
    for key, want in (("period_hours", shape.spin_period_hours),
                      ("pole_ra_deg", shape.spin_pole_ra_deg),
                      ("pole_dec_deg", shape.spin_pole_dec_deg)):
        assert abs(float(spin[key]) - want) < 1e-6, (
            f"{example.name} has {key} = {spin[key]}; the registry says "
            f"{want}. Two statements of one spin solution have drifted.")
    print("    [parity] the registry's spin is the shipped example's spin, "
          "to 1e-6")


def test_the_pinned_table_is_executed_from_the_installation_only():
    """``fetch_kernels_module`` runs code. It must not run the user's cwd.

    ``project_root`` reaches this module from ``BodyCard._project_root``,
    which falls back to ``Path.cwd()`` when the engine executable cannot be
    located -- and ``find_project_root`` returns ``exe.parent`` for an
    executable outside a ``build/bin`` layout. Loading
    ``<project_root>/scripts/fetch_kernels.py`` therefore meant executing a
    file from whatever directory the GUI happened to be launched in.
    Everywhere else in the module ``project_root`` selects DATA; this one call
    turns a path into code, so it is pinned to the installation.

    Discriminating: a scratch root is given a ``scripts/fetch_kernels.py``
    that raises on import. If the parameter still chose the file, this check
    fails with that exception.
    """
    from gui_core.curated_sources import fetch_kernels_module

    with tempfile.TemporaryDirectory() as tmp:
        hostile = Path(tmp) / "scripts"
        hostile.mkdir(parents=True)
        (hostile / "fetch_kernels.py").write_text(
            "raise SystemExit('a script from the working directory ran')\n")
        module = fetch_kernels_module(Path(tmp))
        assert hasattr(module, "BODY_KERNELS"), (
            "the installation's table was not the one loaded")
        assert "bennu" in module.BODY_KERNELS
        loaded = Path(module.__file__ or "")
        assert loaded.resolve() != (hostile / "fetch_kernels.py").resolve(), (
            f"the scratch root's script was executed: {loaded}")
    print("    [exec] the pinned table is executed from the installation, and "
          "a scripts/fetch_kernels.py in a passed-in root is not run")


def test_a_download_will_not_write_outside_the_kernel_directory():
    """A path parsed out of a metakernel is fine to read and not to write to.

    ``metakernel_members`` resolves a member anywhere on disk, on purpose --
    an absolute path is what every generated metakernel uses. That was safe
    while the only operations were ``stat`` and ``getfat``. Downloading turns
    the same parsed string into a file this code CREATES, and
    ``PTM_SPICE_KERNELS`` lets the ``.tm`` being parsed be one the user
    chose, so the destination is bounded to the kernel directory and the
    project tree.
    """
    from gui_core.curated_sources import curated_body, download_curated_kernels

    entry = curated_body(BENNU)
    wrote = []

    class StubDownloader:
        @staticmethod
        def download(url, dest, sha):
            wrote.append(dest)
            Path(dest).write_bytes(b"stub")
            return True

    with tempfile.TemporaryDirectory() as tmp:
        root = _isolated_root(tmp, members=("naif0012.tls", "pck00011.tpc",
                                            "de440s.bsp"))
        outside = Path(tmp) / "elsewhere"
        outside.mkdir()
        # A metakernel whose body-SPK member points outside the tree.
        (root / "data" / "kernels" / "bennu.tm").write_text(
            "KPL/MK\n\\begindata\n"
            "PATH_VALUES  = ( 'data/kernels' )\n"
            "PATH_SYMBOLS = ( 'KERNELS' )\n"
            "KERNELS_TO_LOAD = ( '$KERNELS/lsk/naif0012.tls'\n"
            "                    '$KERNELS/pck/pck00011.tpc'\n"
            "                    '$KERNELS/spk/de440s.bsp'\n"
            f"                    '{outside}/sb-101955-118_long.bsp' )\n"
            "\\begintext\n")
        ok, lines = download_curated_kernels(entry, root,
                                             downloader=StubDownloader)

    assert not wrote, (
        f"a file was written outside the kernel directory: {wrote}")
    assert not ok
    blob = " ".join(lines)
    assert "outside both the kernel directory and the project" in blob, blob
    print("    [write-bound] a metakernel member resolving outside the tree "
          "is reported, and no file is created there")


def test_no_curated_fetch_applies_anything_by_itself():
    """These functions RETURN values. They do not write configurations.

    The property that keeps a fetch started on one card out of another card's
    run, and it has to hold for the new path as well as the old one.
    """
    from gui_core.body_fetch import fetch_damit_shape, fetch_spk

    config = _run_config()
    before = repr(config)
    spk = fetch_spk(config, REPO_ROOT, name=BENNU,
                    client_factory=lambda: ExplodingHorizons([]))
    shape = fetch_damit_shape(REPO_ROOT, name=BENNU,
                              client_factory=lambda: ExplodingDamit([]))
    assert repr(config) == before, (
        "fetch_spk mutated the configuration it was handed")
    assert spk.values and shape.values, "nothing was offered to apply"
    print("    [purity] both curated fetches return values and mutate nothing")


def test_every_progress_line_reaches_the_caller_not_only_the_last():
    """A burst that arrives as one write must not lose the lines inside it.

    ``fetch_kernels.download`` prints progress with ``\\r`` and its summary
    lines with ``\\n``, and both can land in a single ``write`` -- Python
    buffers, and a redirected stdout is not line-buffered. The first draft of
    the splitter took ``max`` of the two ``find`` results, which looks
    equivalent to "the next delimiter" and is not: with a newline at 3 and a
    carriage return at 10 it cuts at 10 and silently swallows the line that
    ended at 3.

    Written as a DISCRIMINATING check: it fails against `max`, passes against
    `min`. Without that property it would be a test that agrees with whatever
    the code does.
    """
    from gui_core.curated_sources import _LineWriter

    seen = []
    writer = _LineWriter(seen.append)
    # Newline BEFORE carriage return in one write -- the ordering `max` gets
    # wrong. Three lines went in; three must come out, in order.
    writer.write("resuming at 40.0 MB\r  60.0 MB / 80.7 MB\rdone\n")
    writer.flush()
    assert seen == ["resuming at 40.0 MB", "60.0 MB / 80.7 MB", "done"], seen

    seen.clear()
    writer = _LineWriter(seen.append)
    writer.write("first\nsecond\rthird\n")
    writer.flush()
    assert seen == ["first", "second", "third"], (
        f"a mixed \\n / \\r burst lost a line: {seen}")

    # Partial writes are held, not emitted half-formed.
    seen.clear()
    writer = _LineWriter(seen.append)
    writer.write("  12.0 MB")
    assert seen == [], seen
    writer.write(" / 80.7 MB\n")
    assert seen == ["12.0 MB / 80.7 MB"], seen

    # A callback that raises must not break the download it is reporting on.
    def hostile(_line):
        raise RuntimeError("a progress handler blew up")

    _LineWriter(hostile).write("anything\n")
    print("    [progress] 3 of 3 lines survive a mixed \\n/\\r burst, a "
          "partial write is held, and a raising callback does not propagate")


def test_wiring_an_already_present_kernel_writes_no_tracked_file():
    """Pressing a button that changes nothing must change nothing.

    ``data/horizons_cache/index.json`` is TRACKED. ``_update_index`` writes it
    after a HORIZONS fetch, which is right -- a kernel just landed on disk and
    the index records where. The curated path's normal case installs NOTHING,
    and recording "I looked and it was already here" in a tracked file would
    leave every user's checkout dirty after a read.

    Found the hard way: the first offscreen acceptance run in this worktree
    modified that file, and `git status` showed it beside the source changes.
    """
    import subprocess

    from gui_core.body_fetch import fetch_spk
    from gui_core.curated_sources import curated_body, curated_kernel_status

    git = "/Library/Developer/CommandLineTools/usr/bin/git"
    if not Path(git).exists():
        raise Skip("the CommandLineTools git is not on this machine")
    if not curated_kernel_status(curated_body(BENNU), REPO_ROOT).ready:
        raise Skip("Bennu's kernel is not fetched in this checkout")

    def dirty():
        out = subprocess.run(
            [git, "-C", str(REPO_ROOT), "status", "--porcelain",
             "data/horizons_cache"],
            capture_output=True, text=True)
        return out.stdout.strip()

    before = dirty()
    result = fetch_spk(_run_config(), REPO_ROOT, name=BENNU,
                       client_factory=lambda: ExplodingHorizons([]))
    assert result.ok, result.message
    after = dirty()
    assert after == before, (
        f"wiring an already-present curated kernel wrote into the tracked "
        f"cache tree.\n  before: {before!r}\n  after:  {after!r}\n"
        f"Nothing landed on disk, so nothing may be recorded as having.")
    print("    [read-only] wiring a kernel that is already present leaves the "
          "tracked data/horizons_cache tree byte-identical")


def test_typing_a_curated_name_updates_the_note_and_touches_no_network():
    """The banner follows the Name field, and the Name field is keystrokes.

    A network request per keystroke is the defect the fetch buttons exist to
    avoid, and this note deliberately DOES update as the user types -- so the
    only thing that keeps it honest is that the work behind it is a table
    lookup and two ``stat`` calls. Pinned here with a socket that raises.
    """
    try:
        from PyQt6.QtWidgets import QApplication
    except ImportError as exc:
        raise Skip(f"PyQt6 is not available here ({exc})")
    if not os.environ.get("QT_QPA_PLATFORM"):
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
    app = QApplication.instance() or QApplication([])

    import socket

    from gui_core.config_manager import ConfigManager
    from gui_panels.body_cards_panel import AsteroidCard

    manager = ConfigManager()
    card = AsteroidCard(manager, "asteroid")
    card._project_root = lambda: REPO_ROOT
    card._active = True

    opened = []
    real_socket = socket.socket

    class NoNetwork(real_socket):
        def __init__(self, *args, **kwargs):
            opened.append("socket")
            raise AssertionError("a keystroke opened a socket")

    socket.socket = NoNetwork
    try:
        widget = card.widgets["generic_body.name"]
        for prefix in ("B", "Be", "Ben", "Benn", "Bennu"):
            widget.setText(prefix)
            app.processEvents()
        note = card.curated_note.text()
    finally:
        socket.socket = real_socket

    assert not opened, "typing reached the network"
    assert "solution 118" in note, (
        f"typing the body's name did not raise its curated provenance: "
        f"{note!r}")
    widget.setText("Bennx")
    app.processEvents()
    assert card.curated_note.text() == "", (
        "the curated note survived the name ceasing to match")
    print("    [keystroke] the curated note appears at 'Bennu' and clears at "
          "'Bennx', across 6 keystrokes, with no socket opened")


def test_the_kernel_selector_refactor_did_not_change_what_gets_fetched():
    """``select_kernels`` still chooses exactly what it chose before.

    ``BODY_KERNELS`` was introduced so the GUI could read the pinned table.
    That is a refactor of a script that decides what a user downloads, so the
    behavior is pinned here rather than assumed: every mode, against the
    filenames the script's own docstring names.
    """
    from gui_core.curated_sources import fetch_kernels_module

    module = fetch_kernels_module(REPO_ROOT)
    core = {name for _s, name, _u, _h in module.KERNELS_CORE}
    with tempfile.TemporaryDirectory() as tmp:
        def names(bodies, present=()):
            for tm in present:
                (Path(tmp) / tm).write_text("stub")
            got = {n for _s, n, _u, _h in module.select_kernels(bodies, tmp)}
            for tm in present:
                (Path(tmp) / tm).unlink()
            return got

        assert names("moon") == core
        assert names("europa") == core | {"jup365.bsp"}
        assert names("bennu") == core | {"sb-101955-118_long.bsp"}
        assert names("all") == core | {"jup365.bsp",
                                       "sb-101955-118_long.bsp"}
        assert names("auto") == core, "auto with no .tm present added kernels"
        assert names("auto", ["bennu.tm"]) == core | {
            "sb-101955-118_long.bsp"}
        assert names("auto", ["europa.tm"]) == core | {"jup365.bsp"}
        assert names("auto", ["europa.tm", "bennu.tm"]) == core | {
            "jup365.bsp", "sb-101955-118_long.bsp"}
    print(f"    [do-no-harm] select_kernels is unchanged across all 5 modes "
          f"and both auto-detections ({len(core)} core kernels)")


def test_the_card_swaps_its_labels_for_a_curated_body_and_back():
    """A card is reused across bodies; the curated labels must not stick.

    "Use JPL solution 118" left over the top of 623 Chimaera would be the same
    cross-card residue that ``_repair_body_name`` removes from
    ``simulation.body`` and ``_provenance`` removes from the field labels.
    """
    try:
        from PyQt6.QtWidgets import QApplication
    except ImportError as exc:
        raise Skip(f"PyQt6 is not available here ({exc})")
    if not os.environ.get("QT_QPA_PLATFORM"):
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
    app = QApplication.instance() or QApplication([])
    assert app is not None

    from gui_core.config_manager import ConfigManager
    from gui_panels.body_cards_panel import AsteroidCard

    manager = ConfigManager()
    card = AsteroidCard(manager, "asteroid")
    card._project_root = lambda: REPO_ROOT
    default_spk = card.spk_fetch_button.text()
    default_shape = card.damit_button.text()

    manager.set_value("generic_body.name", BENNU)
    card._refresh_curated()
    curated_spk = card.spk_fetch_button.text()
    assert curated_spk != default_spk, (
        "the ephemeris button still offers HORIZONS for a body HORIZONS "
        "refuses")
    assert "solution 118" in curated_spk, curated_spk
    assert "radar" in card.damit_button.text().lower()
    note = card.curated_note.text()
    assert "1900-2135" in note and "Farnocchia" in note, note
    assert "HORIZONS is not asked" in note, note

    manager.set_value("generic_body.name", "623 Chimaera (A907 BC)")
    manager.set_value("generic_body.naif_id", "20000623")
    card._refresh_curated()
    assert card.spk_fetch_button.text() == default_spk, (
        f"the curated label survived a change of body: "
        f"{card.spk_fetch_button.text()!r}")
    assert card.damit_button.text() == default_shape
    assert card.curated_note.text() == "", (
        f"a curated note survived onto a non-curated body: "
        f"{card.curated_note.text()!r}")
    print("    [card] the labels and the provenance note appear for a curated "
          "body and are removed again when the body changes")


CHECKS = [
    test_every_curated_entry_resolves_to_files_that_exist,
    test_the_curated_entry_agrees_with_the_shipped_configuration,
    test_the_pinned_hash_is_not_copied_into_the_gui,
    test_a_curated_body_never_reaches_horizons,
    test_a_curated_body_never_reaches_damit,
    test_a_non_curated_body_is_unchanged_and_still_reaches_the_service,
    test_the_curated_source_wins_over_a_superseded_cached_kernel,
    test_the_fetched_kernel_is_detected_in_both_states,
    test_readiness_covers_every_kernel_the_metakernel_names,
    test_a_download_is_not_offered_for_a_state_it_cannot_fix,
    test_a_missing_curated_kernel_is_an_offer_and_downloads_nothing,
    test_consent_uses_the_projects_own_downloader_with_the_pinned_hash,
    test_the_bennu_metakernel_reports_its_published_coverage,
    test_the_curated_shape_comes_with_its_provenance_and_its_git_status,
    test_the_curated_shape_brings_its_spin_and_can_still_disagree,
    test_the_shipped_example_agrees_with_the_curated_spin,
    test_the_pinned_table_is_executed_from_the_installation_only,
    test_a_download_will_not_write_outside_the_kernel_directory,
    test_no_curated_fetch_applies_anything_by_itself,
    test_every_progress_line_reaches_the_caller_not_only_the_last,
    test_wiring_an_already_present_kernel_writes_no_tracked_file,
    test_typing_a_curated_name_updates_the_note_and_touches_no_network,
    test_the_kernel_selector_refactor_did_not_change_what_gets_fetched,
    test_the_card_swaps_its_labels_for_a_curated_body_and_back,
]


def _tracked_state():
    """`git status --porcelain` for the paths these checks could write.

    A whole-suite guard, run before and after. One arm of this file DID write
    a Bennu entry into the tracked `data/horizons_cache/index.json` -- with a
    `/var/folders/...` path in it -- because it passed the repository as the
    project root while pointing the kernel directory at a temporary one. Every
    later `git status` then carried it. A test suite that dirties the working
    tree is a test suite that will eventually be committed by accident.
    """
    import subprocess
    git = "/Library/Developer/CommandLineTools/usr/bin/git"
    if not Path(git).exists():
        return None
    out = subprocess.run(
        [git, "-C", str(REPO_ROOT), "status", "--porcelain",
         "data", "examples", "configs"],
        capture_output=True, text=True)
    return out.stdout


def main():
    failed = skipped = 0
    before = _tracked_state()
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
    after = _tracked_state()
    if before is not None and after != before:
        failed += 1
        print("FAIL the suite left the working tree dirty. These checks must "
              "not write into the repository:\n"
              f"  before:\n{before or '    (clean)'}"
              f"  after:\n{after or '    (clean)'}")
    elif before is not None:
        print("    [hygiene] data/, examples/ and configs/ are in the same "
              "state as before the run")
    print(f"\n{total - failed - skipped}/{total} passed, {skipped} skipped, "
          f"{failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
