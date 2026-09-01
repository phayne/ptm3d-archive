#!/usr/bin/env python3
"""A fetch does not overwrite a kernel the repository ships.

WHAT WENT WRONG, on 2026-08-31, in one session of ordinary testing:

1. The asteroid card's SPK fetch installed its download over
   ``data/horizons_cache/bodies/269_Justitia_(A887_SA)/ephemeris.spk`` -- a
   TRACKED file -- replacing an 899-day kernel with a 762-day one that spanned
   NONE of the campaign windows the fourteen shipped Justitia configurations
   run in. Nothing compared the two. Nothing asked.
2. Two committed ``metakernel.tm`` files were regenerated to machine-absolute
   ``PATH_VALUES``. Each of those files' own header forbids exactly that
   ("Hand-written relative, NOT regenerated verbatim from
   SPKGenerator.create_metakernel() ... baking those in here would make this
   committed file resolve only on the machine that generated it"), and the
   regeneration needed no download at all: it happened on the CACHE-HIT path,
   where ``from_cache=True`` and the fetch is supposed to be a read.

GUI-FETCH-CLOBBERS-COMMITTED-KERNELS. The arms below are the four properties
that make each of those impossible, plus the negative control that keeps the
guard from costing anything on the ordinary path:

* a TRACKED destination is not overwritten -- the default saves beside it and
  points the configuration at the copy;
* an explicit Replace is STILL refused when it would lose coverage, with both
  spans named;
* a committed metakernel that resolves is left exactly as it is, cache hit
  included;
* an UNTRACKED destination behaves precisely as it did before.

Run directly::

    python3 src/python/tests/test_gui_kernel_guard.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "gui"))
sys.path.insert(0, str(REPO_ROOT / "src" / "python"))

BODY = "269 Justitia (A887 SA)"
BODY_DIR = "269_Justitia_(A887_SA)"

#: The two arcs from the incident, so the numbers in this file are the numbers
#: that happened rather than round ones invented for a test.
SHIPPED_SPAN = ("2035-01-01T00:00:00", "2037-06-19T00:00:00", 899.0)
FETCHED_SPAN = ("2032-01-01T00:00:00", "2034-02-01T00:00:00", 762.0)


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

def _config(start="2035-08-24T00:00:00", days=10.0, cycles=20.0, period=5.0):
    return {
        "simulation": {"start_time": start, "duration_days": days},
        "thermal": {"equilibration_orbits": cycles},
        "generic_body": {"spin": {"period_hours": period}},
    }


class _FakeCoverage:
    def __init__(self, available, start_utc="", end_utc="", days=0.0,
                 detail="stub"):
        self.available = available
        self.start_utc = start_utc
        self.end_utc = end_utc
        self.days = days
        self.detail = detail
        self.bodies = ()
        self.intervals = ()

    def covers(self, start_utc, end_utc):
        if not self.available:
            return None
        return bool(self.start_utc <= str(start_utc)[:19]
                    and str(end_utc)[:19] <= self.end_utc)


def _with_coverage_by_content(table, default):
    """Patch ``spk_coverage`` so DIFFERENT files answer DIFFERENTLY.

    The existing stub in ``test_gui_body_autopopulate`` returns one coverage
    for every path, which is right for its questions and useless for these:
    every arm here is about comparing the file on disk with the one that just
    arrived, and a stub that cannot tell them apart would report "no coverage
    lost" for the very clobber this file is about.

    Keyed on the file's first bytes rather than its name, because the point of
    the guard is that the new kernel may end up under a DIFFERENT name.
    """
    from gui_core import body_cards
    original = body_cards.spk_coverage

    def stub(path, metakernel=None, naif_id=None, project_root=None):
        try:
            head = Path(str(path)).read_bytes()[:64]
        except OSError:
            return default
        for marker, coverage in table.items():
            if marker in head:
                return coverage
        return default

    body_cards.spk_coverage = stub
    return lambda: setattr(body_cards, "spk_coverage", original)


class _SpkClient:
    """A HORIZONS stub that writes a distinguishable file."""

    def __init__(self, payload=b"DAF/SPK FETCHED stub payload"):
        self.payload = payload

    def download_spk(self, body_id, start, stop, output_path,
                     body_type="asteroid"):
        Path(output_path).write_bytes(self.payload)
        return output_path


def _git(argv, cwd):
    return subprocess.run(
        ["git", "-c", "user.email=test@example.invalid",
         "-c", "user.name=Test"] + argv,
        cwd=str(cwd), capture_output=True, text=True, timeout=60)


def _fixture_repo(root, with_metakernel=True, metakernel_text=None):
    """A real git repository holding a COMMITTED kernel for this body.

    A real repository, not a monkeypatched predicate: the thing being tested is
    whether the guard can tell that a file is part of the distribution, and a
    stubbed answer to that question would test the stub.
    """
    root = Path(root)
    body_dir = root / "data" / "horizons_cache" / "bodies" / BODY_DIR
    body_dir.mkdir(parents=True, exist_ok=True)
    spk = body_dir / "ephemeris.spk"
    spk.write_bytes(b"DAF/SPK SHIPPED stub payload")
    if with_metakernel:
        mk = body_dir / "metakernel.tm"
        mk.write_text(metakernel_text or _relative_metakernel())
    init = _git(["init", "-q"], root)
    if init.returncode != 0:                                 # pragma: no cover
        pytest.skip(f"git is unavailable here ({init.stderr.strip()})")
    _git(["add", "-A"], root)
    commit = _git(["commit", "-q", "-m", "ship the kernels"], root)
    if commit.returncode != 0:                               # pragma: no cover
        pytest.skip(f"git could not commit here ({commit.stderr.strip()})")
    return root


def _relative_metakernel():
    """The shape the committed metakernels actually have: RELATIVE paths."""
    return (
        "KPL/MK\n\n\\begintext\n\n"
        "  Paths below are RELATIVE to the repository root. Hand-written,\n"
        "  NOT regenerated from SPKGenerator.create_metakernel().\n\n"
        "\\begindata\n\n"
        "  PATH_VALUES  = ( 'data/horizons_cache/bodies/" + BODY_DIR + "' )\n"
        "  PATH_SYMBOLS = ( 'BODY' )\n\n"
        "  KERNELS_TO_LOAD = (\n"
        "            '$BODY/ephemeris.spk'\n"
        "  )\n\n"
        "\\begintext\n\nEnd of MK.\n")


def _fetch(root, **kwargs):
    from gui_core.body_fetch import fetch_spk
    return fetch_spk(_config(), str(root), name=BODY, naif_id="20000269",
                     client_factory=lambda: _SpkClient(), **kwargs)


def _covering_table():
    """Both kernels cover the run; the fetched one loses the shipped arc."""
    shipped = _FakeCoverage(True, *SHIPPED_SPAN)
    fetched = _FakeCoverage(True, "2035-01-01T00:00:00",
                            "2036-01-01T00:00:00", 365.0)
    return ({b"SHIPPED": shipped, b"FETCHED": fetched}, shipped, fetched)


# --------------------------------------------------------------------------
# The predicate
# --------------------------------------------------------------------------

def test_tracked_by_git_says_tracked_untracked_and_unknown_apart():
    from gui_core.body_fetch import tracked_by_git

    with tempfile.TemporaryDirectory() as tmp:
        root = _fixture_repo(tmp)
        spk = root / "data" / "horizons_cache" / "bodies" / BODY_DIR / "ephemeris.spk"
        tracked, _why = tracked_by_git(spk)
        assert tracked is True, "a committed file was not reported as tracked"

        loose = spk.with_name("scratch.spk")
        loose.write_bytes(b"x")
        assert tracked_by_git(loose)[0] is False, (
            "an untracked file inside the repository was reported as tracked")

    with tempfile.TemporaryDirectory() as plain:
        # A release tarball: git runs, and answers that this is not a
        # repository. That has to be a definite False, or a plain install
        # cannot fetch anything.
        loose = Path(plain) / "ephemeris.spk"
        loose.write_bytes(b"x")
        tracked, _why = tracked_by_git(loose)
        assert tracked is False, (
            f"outside any repository the answer must be a definite False, "
            f"not {tracked!r} -- otherwise a plain install can never fetch")


def test_an_unaskable_question_is_treated_as_protected():
    """"Could not look" is not "not tracked".

    With no git to ask, the guard must protect rather than proceed: going quiet
    exactly where it cannot check is the defect, not the remedy.
    """
    from gui_core import body_fetch

    original = body_fetch._git_program
    body_fetch._git_program = lambda: ""
    try:
        with tempfile.TemporaryDirectory() as tmp:
            loose = Path(tmp) / "ephemeris.spk"
            loose.write_bytes(b"x")
            tracked, why = body_fetch.tracked_by_git(loose)
        assert tracked is None, tracked
        assert "git could not be found" in why, why
    finally:
        body_fetch._git_program = original


def test_an_exit_one_that_is_not_gits_own_no_is_unknown_not_untracked():
    """The macOS shim, which is how this guard would have gone quiet.

    `shutil.which("git")` finds `/usr/bin/git`, the xcrun shim. With the
    Command Line Tools missing or moved it prints "xcrun: error: invalid
    active developer path" and exits 1 -- the SAME exit code `--error-unmatch`
    uses for "this path is untracked". Reading that as "not tracked" overwrites
    the committed kernel in place, which is the incident.
    """
    from gui_core import body_fetch

    def shim(stderr, code=1):
        class _Out:
            returncode = code
            stdout = ""
        _Out.stderr = stderr
        return lambda argv, **kwargs: _Out()

    real_run = subprocess.run
    try:
        subprocess.run = shim("xcrun: error: invalid active developer path\n")
        tracked, why = body_fetch.tracked_by_git(Path("/nowhere/ephemeris.spk"))
        assert tracked is None, (
            f"an xcrun shim failure was read as {tracked!r}; only git's own "
            f"'did not match' means untracked")
        assert "xcrun" in why, why

        subprocess.run = shim("error: pathspec 'x' did not match any file(s) "
                              "known to git\n")
        assert body_fetch.tracked_by_git(Path("/nowhere/x"))[0] is False, (
            "git's own words for 'untracked' must still read as untracked")
    finally:
        subprocess.run = real_run


# --------------------------------------------------------------------------
# The tracked destination
# --------------------------------------------------------------------------

def test_replace_is_refused_when_the_old_coverage_cannot_be_read():
    """The vacuous pass: the guard reproducing the incident it prevents.

    With no spiceypy -- or no leapseconds kernel, which is the state of any
    fresh checkout, `data/kernels/lsk` being untracked -- `spk_coverage`
    reports `available=False` for BOTH files. A loss check that returned "no
    loss" there would replace the shipped 899-day kernel and only afterward
    report COVERAGE UNKNOWN, with the dialog still promising it would be
    refused if coverage were dropped.
    """
    from gui_core.body_fetch import ON_TRACKED_REPLACE

    unreadable = _FakeCoverage(False, detail="no spiceypy")
    restore = _with_coverage_by_content({}, unreadable)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            root = _fixture_repo(tmp)
            spk = (root / "data" / "horizons_cache" / "bodies" / BODY_DIR
                   / "ephemeris.spk")
            before = spk.read_bytes()

            result = _fetch(root, refetch=True, on_tracked=ON_TRACKED_REPLACE)

            assert spk.read_bytes() == before, (
                "a shipped kernel was replaced on a machine that could not "
                "read either coverage")
            assert "could not be read" in result.refusal, result.refusal
    finally:
        restore()


def test_a_first_fetch_still_lands_on_the_canonical_name_when_git_is_mute():
    """Protecting a file that is not there would break every first fetch.

    With git unavailable the destination is unknown, but an ABSENT destination
    has nothing to protect. Landing under a dated name instead would leave
    `fetch_spk`'s cache-first check -- which tests the canonical
    `ephemeris.spk` -- looking at a path that never exists, so every press
    would re-download the kernel and park another copy.
    """
    from gui_core import body_fetch

    table, _s, _f = _covering_table()
    restore = _with_coverage_by_content(table, _FakeCoverage(False))
    original = body_fetch._git_program
    body_fetch._git_program = lambda: ""
    try:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = _fetch(root)
            spk = (root / "data" / "horizons_cache" / "bodies" / BODY_DIR
                   / "ephemeris.spk")
            assert spk.exists(), (
                f"the first fetch did not land on the canonical name; "
                f"{sorted(p.name for p in spk.parent.iterdir())}")
            assert not list(spk.parent.glob("ephemeris_fetched_*.spk"))
            said = " ".join(result.lines())
            assert "could not be checked" in said, (
                f"the check was skipped without saying so:\n{said}")
    finally:
        body_fetch._git_program = original
        restore()


def test_a_refused_replacement_does_not_park_a_second_identical_copy():
    """Each Yes on the card's offer re-fetches. It must not accumulate."""
    from gui_core.body_fetch import ON_TRACKED_REPLACE

    shipped = _FakeCoverage(True, *SHIPPED_SPAN)
    fetched = _FakeCoverage(True, *FETCHED_SPAN)
    restore = _with_coverage_by_content(
        {b"SHIPPED": shipped, b"FETCHED": fetched}, _FakeCoverage(False))
    try:
        with tempfile.TemporaryDirectory() as tmp:
            root = _fixture_repo(tmp)
            body_dir = (root / "data" / "horizons_cache" / "bodies" / BODY_DIR)
            first = _fetch(root, refetch=True, on_tracked=ON_TRACKED_REPLACE)
            second = _fetch(root, refetch=True, on_tracked=ON_TRACKED_REPLACE)
            copies = sorted(body_dir.glob("ephemeris_fetched_*.spk"))
            assert len(copies) == 1, (
                f"two refused replacements left {len(copies)} copies: "
                f"{[p.name for p in copies]}")
            assert first.paths["spk"] == second.paths["spk"]
    finally:
        restore()

def test_a_tracked_kernel_is_not_overwritten_and_the_copy_is_wired():
    """The default: save beside, and point the configuration at the copy."""
    table, _shipped, _fetched = _covering_table()
    restore = _with_coverage_by_content(table, _FakeCoverage(False))
    try:
        with tempfile.TemporaryDirectory() as tmp:
            root = _fixture_repo(tmp)
            body_dir = (root / "data" / "horizons_cache" / "bodies" / BODY_DIR)
            spk = body_dir / "ephemeris.spk"
            before = spk.read_bytes()

            result = _fetch(root, refetch=True)

            assert spk.read_bytes() == before, (
                "the tracked kernel was overwritten -- this is the clobber")
            beside = [p for p in body_dir.glob("ephemeris_fetched_*.spk")]
            assert len(beside) == 1, (
                f"expected exactly one kernel saved beside; found {beside}")
            assert beside[0].read_bytes().startswith(b"DAF/SPK FETCHED")
            assert not list(body_dir.glob("*.part")), (
                "the scratch download was left behind")

            wired = result.values.get("generic_body.spice_metakernel", "")
            assert str(spk) != wired, (
                "the configuration was pointed at the shipped kernel rather "
                "than at what was actually fetched")
            assert beside[0].stem in wired, (
                f"the configuration points at {wired!r}, which is neither the "
                f"kernel that was written nor a metakernel beside it")
            said = " ".join(result.lines())
            assert "part of the distribution" in said, said
            assert beside[0].name in said, said
    finally:
        restore()


def test_replacing_a_tracked_kernel_is_refused_when_coverage_would_be_lost():
    """The exact clobber: 899 days on disk, 762 days fetched, replace asked.

    Consent to replace is not consent to lose an arc, and the refusal names
    BOTH spans so the user can see what the trade actually was.
    """
    from gui_core.body_fetch import ON_TRACKED_REPLACE

    shipped = _FakeCoverage(True, *SHIPPED_SPAN)
    fetched = _FakeCoverage(True, *FETCHED_SPAN)
    restore = _with_coverage_by_content(
        {b"SHIPPED": shipped, b"FETCHED": fetched}, _FakeCoverage(False))
    try:
        with tempfile.TemporaryDirectory() as tmp:
            root = _fixture_repo(tmp)
            body_dir = (root / "data" / "horizons_cache" / "bodies" / BODY_DIR)
            spk = body_dir / "ephemeris.spk"
            before = spk.read_bytes()

            result = _fetch(root, refetch=True, on_tracked=ON_TRACKED_REPLACE)

            assert spk.read_bytes() == before, (
                "a replacement that loses coverage went through anyway")
            assert not result.ok
            assert "REFUSED" in result.refusal, result.refusal
            for moment in (SHIPPED_SPAN[0], SHIPPED_SPAN[1],
                           FETCHED_SPAN[0], FETCHED_SPAN[1]):
                assert moment in result.refusal, (
                    f"the refusal does not name {moment}; both spans have to "
                    f"be in it:\n{result.refusal}")
            assert "generic_body.ephemeris_source" not in result.values, (
                "a refused fetch still claimed SPICE")
            # The download is real and is kept, beside, rather than discarded.
            assert Path(result.paths["spk"]).exists()
            assert Path(result.paths["spk"]) != spk
    finally:
        restore()


def test_replacing_a_tracked_kernel_is_allowed_when_nothing_is_lost():
    """Consent, honored -- and the way back stated in the same breath."""
    from gui_core.body_fetch import ON_TRACKED_REPLACE

    shipped = _FakeCoverage(True, "2035-06-01T00:00:00",
                            "2035-12-01T00:00:00", 183.0)
    fetched = _FakeCoverage(True, "2034-01-01T00:00:00",
                            "2037-01-01T00:00:00", 1096.0)
    restore = _with_coverage_by_content(
        {b"SHIPPED": shipped, b"FETCHED": fetched}, _FakeCoverage(False))
    try:
        with tempfile.TemporaryDirectory() as tmp:
            root = _fixture_repo(tmp)
            spk = (root / "data" / "horizons_cache" / "bodies" / BODY_DIR
                   / "ephemeris.spk")

            result = _fetch(root, refetch=True, on_tracked=ON_TRACKED_REPLACE)

            assert spk.read_bytes().startswith(b"DAF/SPK FETCHED"), (
                "an explicit Replace that loses nothing was not carried out")
            said = " ".join(result.lines())
            assert "REPLACED" in said, said
            assert "git checkout --" in said, (
                f"a tracked file was replaced without saying how to get it "
                f"back:\n{said}")
    finally:
        restore()


def test_cancel_writes_nothing_at_all():
    from gui_core.body_fetch import ON_TRACKED_CANCEL

    table, _s, _f = _covering_table()
    restore = _with_coverage_by_content(table, _FakeCoverage(False))
    try:
        with tempfile.TemporaryDirectory() as tmp:
            root = _fixture_repo(tmp)
            body_dir = (root / "data" / "horizons_cache" / "bodies" / BODY_DIR)
            before = sorted(p.name for p in body_dir.iterdir())
            payload = (body_dir / "ephemeris.spk").read_bytes()

            result = _fetch(root, refetch=True, on_tracked=ON_TRACKED_CANCEL)

            assert sorted(p.name for p in body_dir.iterdir()) == before, (
                "Cancel left files behind")
            assert (body_dir / "ephemeris.spk").read_bytes() == payload
            assert not result.ok
            assert "NOT been touched" in result.message, result.message
    finally:
        restore()


def test_an_unknown_on_tracked_choice_is_refused_by_name():
    """Not silently defaulted: a misspelled choice would otherwise get the
    default's behavior under the user's own word for something else."""
    table, _s, _f = _covering_table()
    restore = _with_coverage_by_content(table, _FakeCoverage(False))
    try:
        with tempfile.TemporaryDirectory() as tmp:
            root = _fixture_repo(tmp)
            result = _fetch(root, refetch=True, on_tracked="overwrite")
            assert not result.ok
            assert "on_tracked" in result.message, result.message
            assert "save_beside" in result.message, result.message
    finally:
        restore()


# --------------------------------------------------------------------------
# The committed metakernel
# --------------------------------------------------------------------------

def test_a_cache_hit_does_not_rewrite_a_committed_metakernel():
    """The half that needed no download: a cache HIT rewrote two of them.

    The kernel on disk covers the run, so nothing is fetched at all -- and the
    committed, relative-path metakernel beside it must come out byte-identical.
    """
    shipped = _FakeCoverage(True, "2034-01-01T00:00:00",
                            "2037-01-01T00:00:00", 1096.0)
    restore = _with_coverage_by_content({b"SHIPPED": shipped},
                                        _FakeCoverage(False))
    try:
        with tempfile.TemporaryDirectory() as tmp:
            root = _fixture_repo(tmp)
            mk = (root / "data" / "horizons_cache" / "bodies" / BODY_DIR
                  / "metakernel.tm")
            before = mk.read_text()

            result = _fetch(root)

            assert result.from_cache, (
                "this arm is about the cache-hit path and did not take it")
            assert mk.read_text() == before, (
                "a committed metakernel was rewritten by a fetch that "
                "downloaded nothing:\n" + mk.read_text())
            assert "PATH_VALUES  = ( 'data/" in mk.read_text(), (
                "the committed metakernel's relative PATH_VALUES were "
                "replaced with machine-absolute ones, which its own header "
                "forbids")
            # `resolve()`: `body_cache_dir` resolves the project root, so on
            # macOS the answer is under /private/var and the fixture path is
            # under /var -- one directory, two spellings.
            wired = Path(result.values["generic_body.spice_metakernel"])
            assert wired.resolve() == mk.resolve(), (wired, mk)
    finally:
        restore()


def test_a_committed_metakernel_that_does_not_resolve_gets_a_local_variant():
    """Left alone, still -- and the configuration points at the local copy.

    A committed metakernel naming a kernel that is not on THIS disk cannot be
    used, but it is still not this fetch's file to rewrite.
    """
    broken = _relative_metakernel().replace("ephemeris.spk", "not_here.spk")
    shipped = _FakeCoverage(True, "2034-01-01T00:00:00",
                            "2037-01-01T00:00:00", 1096.0)
    restore = _with_coverage_by_content({b"SHIPPED": shipped},
                                        _FakeCoverage(False))
    try:
        with tempfile.TemporaryDirectory() as tmp:
            root = _fixture_repo(tmp, metakernel_text=broken)
            body_dir = (root / "data" / "horizons_cache" / "bodies" / BODY_DIR)
            mk = body_dir / "metakernel.tm"
            before = mk.read_text()

            result = _fetch(root)

            assert mk.read_text() == before, (
                "a committed metakernel was rewritten because it did not "
                "resolve; not resolving is not consent")
            local = body_dir / "metakernel_local.tm"
            assert local.exists(), (
                f"no machine-local metakernel was written; "
                f"{sorted(p.name for p in body_dir.iterdir())}")
            wired = Path(result.values["generic_body.spice_metakernel"])
            assert wired.resolve() == local.resolve(), (wired, local)
            said = " ".join(result.lines())
            assert "metakernel_local.tm" in said and "unchanged" in said, said
    finally:
        restore()


# --------------------------------------------------------------------------
# The negative control
# --------------------------------------------------------------------------

def test_an_untracked_destination_behaves_exactly_as_before():
    """The ordinary case, unchanged: install in place, no beside, no warning.

    Everyone pays for a guard that fires on the common path, and only four
    files in this repository need protecting.
    """
    table, _s, _f = _covering_table()
    restore = _with_coverage_by_content(table, _FakeCoverage(False))
    try:
        with tempfile.TemporaryDirectory() as tmp:
            # No `git init`: this is a plain directory, the shape a colleague's
            # working folder or a release tarball has.
            root = Path(tmp)
            result = _fetch(root)

            spk = (root / "data" / "horizons_cache" / "bodies" / BODY_DIR
                   / "ephemeris.spk")
            assert spk.exists() and spk.read_bytes().startswith(
                b"DAF/SPK FETCHED"), "the kernel was not installed in place"
            assert not list(spk.parent.glob("ephemeris_fetched_*.spk")), (
                "an untracked destination got the beside treatment")
            said = " ".join(result.lines())
            assert "part of the distribution" not in said, (
                f"the guard spoke up about a file nobody ships:\n{said}")
            assert result.ok, said
    finally:
        restore()


# --------------------------------------------------------------------------
# The card: the guard is not a dead end
# --------------------------------------------------------------------------

def _asteroid_card(scratch):
    """``(panel, card)`` -- one offscreen asteroid card, settings in `scratch`."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        import PyQt6
    except ImportError as exc:                               # pragma: no cover
        pytest.skip(f"PyQt6 is unavailable here ({exc})")
    plug = Path(PyQt6.__file__).parent / "Qt6" / "plugins"
    os.environ.setdefault("QT_PLUGIN_PATH", str(plug))
    os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(plug / "platforms"))
    from PyQt6.QtCore import QSettings
    from PyQt6.QtWidgets import QApplication
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    for scope in (QSettings.Scope.UserScope, QSettings.Scope.SystemScope):
        QSettings.setPath(QSettings.Format.IniFormat, scope, str(scratch))
    from gui_core.config_manager import ConfigManager
    from gui_panels.config_panel import ConfigurationPanel
    global _APP
    _APP = QApplication.instance() or QApplication([])
    # Through the whole panel, not `AsteroidCard(manager, "asteroid")` alone:
    # the card takes the landing panel as its third argument and this is the
    # one place that wires them, so building it by hand would be testing a
    # differently-constructed card than the window has.
    panel = ConfigurationPanel(ConfigManager())
    return panel, panel.tabs["card_asteroid"]


_APP = None


def _answering(button_name):
    """Patch ``QMessageBox.question``; returns ``(restore, asked)``."""
    from PyQt6.QtWidgets import QMessageBox
    original = QMessageBox.question
    asked = []

    def stub(parent, title, text, *args, **kwargs):
        asked.append((title, text))
        return getattr(QMessageBox.StandardButton, button_name)

    QMessageBox.question = staticmethod(stub)
    return (lambda: setattr(QMessageBox, "question", original)), asked


def test_the_card_offers_the_replacement_it_declined_to_make():
    """A guard with no way out is a feature nobody can use.

    The safe thing has already happened by the time this dialog appears, so
    the question is the way OUT of the guard rather than the way past it --
    and the default button is No, because the thing that went wrong was a
    replacement nobody chose.
    """
    from gui_core.body_fetch import FetchResult, ON_TRACKED_REPLACE

    with tempfile.TemporaryDirectory() as scratch:
        panel, card = _asteroid_card(scratch)
        result = FetchResult(
            ok=True, kind="spk",
            message="The kernel was saved as ephemeris_fetched_20260831.spk.",
            tracked_destination="/repo/data/horizons_cache/bodies/x/ephemeris.spk")

        # No: nothing is re-fetched, and the card says which file stands.
        restore, asked = _answering("No")
        calls = []
        card._on_fetch_spk = lambda **kw: calls.append(kw)
        try:
            handled = card._offer_tracked_replacement(result)
        finally:
            restore()
        assert handled is True
        assert calls == [], "answering No still fetched something"
        assert len(asked) == 1, asked
        title, text = asked[0]
        assert "ephemeris.spk" in text and "NOT been changed" in text, text
        assert "refused if it would drop coverage" in text, text
        assert "ephemeris.spk" in card.coverage_note.text(), (
            card.coverage_note.text())

        # Yes: a second fetch, and it carries the consent by NAME.
        restore, asked = _answering("Yes")
        calls = []
        try:
            card._offer_tracked_replacement(result)
        finally:
            restore()
        assert calls == [{"on_tracked": ON_TRACKED_REPLACE}], calls

        # No flag, no dialog -- the ordinary fetch is not interrupted.
        restore, asked = _answering("Yes")
        try:
            assert card._offer_tracked_replacement(
                FetchResult(ok=True, kind="spk", message="fine")) is False
        finally:
            restore()
        assert asked == [], "a dialog was raised for an ordinary fetch"
        panel.deleteLater()


def test_the_shipped_justitia_kernel_is_tracked_here():
    """The premise, on the real repository rather than a fixture.

    If this ever fails, the guard above is protecting nothing: it would mean
    the kernel the fourteen Justitia configurations read is no longer part of
    the distribution.
    """
    from gui_core.body_fetch import tracked_by_git

    spk = (REPO_ROOT / "data" / "horizons_cache" / "bodies" / BODY_DIR
           / "ephemeris.spk")
    if not spk.exists():                                     # pragma: no cover
        pytest.skip(f"{spk} is not on this disk")
    tracked, why = tracked_by_git(spk)
    if tracked is None:                                      # pragma: no cover
        pytest.skip(f"git could not be asked here: {why}")
    assert tracked is True, (
        f"{spk} is not tracked, so a fetch would overwrite it with no guard")


if __name__ == "__main__":                                   # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
