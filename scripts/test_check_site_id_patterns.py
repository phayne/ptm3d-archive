#!/usr/bin/env python3
"""Self-test for ``scripts/check_site_id_patterns.py``.

A hygiene gate that cannot fail is a hygiene gate nobody should trust, and this
project has already paid for that lesson twice: the deny-list check skips on a
machine with no local list (which is every CI runner), and a pull request
merged 61 site-identifier hits through it. So the shape-only check that
replaces it in CI gets its own gate, run as a CI step BEFORE the check itself,
exactly like ``scripts/test_check_american_english.py``.

What is pinned here:

* the checker finds a planted identifier shape, and does not find one in a
  clean file (so it can fail, and it does not fire on everything);
* both shapes are caught, not just the first;
* the exit codes: 0 clean, 1 on a hit, **2 when git cannot answer** -- the last
  one is the whole point, because reporting clean on a broken git is how a
  gate learns to say yes;
* the inline escape hatch and the ALLOW_PATHS table both actually exempt;
* the checker does not print the matching text, only the line and the shape;
* the checker's own two files are clean under the checker, with ALLOW_PATHS
  empty -- so the gate is not exempting itself to pass.

**Every shaped string in this file is assembled at run time from fragments.**
Written out whole, they would be literals in a tracked file, and the checker
would flag its own self-test -- correctly. Assembling them is not a trick to
get around the gate; it is the demonstration that the gate works on the file
that plants them.

Run directly::

    python3 scripts/test_check_site_id_patterns.py
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
import tempfile
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import check_site_id_patterns as C                            # noqa: E402

REPO_ROOT = HERE.parent

#: A catalog-shaped identifier, assembled so this file contains no literal of
#: that shape. It is not a real identifier: the region code and number are
#: invented, and the gate cannot tell the difference -- which is correct, since
#: shape is all it knows.
PLANTED_ID = "A03" + "ZZ" + "9999"

#: A site-filename-shaped coordinate field, likewise assembled. 89.999 S is not
#: a place this project works on; it is a shape.
PLANTED_COORD = "_8" + "9.999S" + "_"

#: A tag of the kind the tests are supposed to use instead -- invented, and
#: deliberately NOT of either shape.
SAFE_TAG = "TESTSITE01"


class Failure(AssertionError):
    pass


def _git():
    for git in ("/Library/Developer/CommandLineTools/usr/bin/git", "git"):
        try:
            subprocess.run([git, "--version"], capture_output=True, check=True)
            return git
        except (OSError, subprocess.CalledProcessError):
            continue
    return None


def _repo(tmp: Path, files: dict) -> Path:
    """A throwaway git repository with ``files`` written and staged."""
    git = _git()
    assert git, "no working git binary to build the fixture repository with"
    subprocess.run([git, "init", "-q", str(tmp)], capture_output=True,
                   check=True)
    for name, body in files.items():
        target = tmp / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body)
    subprocess.run([git, "-C", str(tmp), "add", "-A"], capture_output=True,
                   check=True)
    return tmp


def _run(argv, cwd=None):
    """``(exit code, stdout, stderr)`` from the checker's own main().

    ``SystemExit`` counts as an exit code, because to a CI step it IS one:
    argparse raises it for an unrecognized flag, and a harness that let it
    escape would report the strictness arm as a crash rather than as the
    exit 2 the shell actually sees.
    """
    out, err = io.StringIO(), io.StringIO()
    previous = os.getcwd()
    if cwd:
        os.chdir(cwd)
    try:
        with redirect_stdout(out), redirect_stderr(err):
            try:
                code = C.main(["check_site_id_patterns.py"] + argv)
            except SystemExit as exc:
                code = exc.code if isinstance(exc.code, int) else 1
    finally:
        os.chdir(previous)
    return code, out.getvalue(), err.getvalue()


# --------------------------------------------------------------------------

def check_a_planted_catalog_identifier_is_found():
    """The load-bearing arm: it can fail."""
    with tempfile.TemporaryDirectory() as td:
        root = _repo(Path(td), {
            "dirty.py": f'MESH = "{PLANTED_ID}_R3km.ply"\n',
            "clean.py": f'MESH = "{SAFE_TAG}_R3km.ply"\n',
        })
        hits = C.scan(C.tracked_files(str(root)), str(root))
        names = {p for p, _n, _w in hits}
        if "dirty.py" not in names:
            raise Failure("a planted catalog identifier was NOT found; the "
                          "checker cannot fail, so its passing says nothing")
        if "clean.py" in names:
            raise Failure("a file with an invented, non-matching tag was "
                          "flagged; the pattern is too broad to keep")


def check_a_planted_coordinate_field_is_found():
    """The second shape is live too, not decoration next to the first."""
    with tempfile.TemporaryDirectory() as td:
        root = _repo(Path(td), {
            "paths.md": f"the mesh is at core{PLANTED_COORD}R3km_linear.ply\n",
        })
        hits = C.scan(C.tracked_files(str(root)), str(root))
        if not any(p == "paths.md" for p, _n, _w in hits):
            raise Failure("the coordinate-field shape was not caught, so only "
                          "one of the two patterns is doing anything")


def check_the_exit_code_is_1_on_a_hit_and_0_when_clean():
    """A checker that prints hits and returns 0 cannot gate anything."""
    with tempfile.TemporaryDirectory() as td:
        root = _repo(Path(td), {"dirty.py": f"# {PLANTED_ID}\n"})
        code, out, _err = _run(["--root", str(root)], cwd=str(root))
        if code != 1:
            raise Failure(f"expected exit 1 on a hit, got {code}")
        if "dirty.py:1" not in out:
            raise Failure(f"the hit did not name the file and line: {out!r}")

    with tempfile.TemporaryDirectory() as td:
        root = _repo(Path(td), {"clean.py": f"# {SAFE_TAG}\n"})
        code, out, _err = _run(["--root", str(root)], cwd=str(root))
        if code != 0:
            raise Failure(f"expected exit 0 on a clean tree, got {code}: {out}")
        if "clean" not in out:
            raise Failure(f"a clean tree did not say so: {out!r}")


def check_git_failure_exits_2_and_never_reports_clean():
    """The failure mode this whole file exists for.

    The gate it replaces reports a SKIP when it cannot run, which in CI is
    indistinguishable from a pass. This one must be unable to do that.
    """
    with tempfile.TemporaryDirectory() as td:
        # A directory that is not a repository: git exits non-zero.
        code, out, err = _run(["--root", td], cwd=td)
        if code != 2:
            raise Failure(
                f"expected exit 2 when the tracked list cannot be obtained, "
                f"got {code}; stdout={out!r}")
        if "clean" in out:
            raise Failure("it reported clean with nothing to scan")
        if "will NOT report clean" not in err:
            raise Failure(f"the reason was not stated on stderr: {err!r}")

    # And the same through the API, so a future caller cannot get [] instead.
    with tempfile.TemporaryDirectory() as td:
        try:
            C.tracked_files(td)
        except C.GitUnavailable:
            pass
        else:
            raise Failure("tracked_files returned a list for a non-repository "
                          "instead of raising; an empty list reads as clean")


def check_a_site_shaped_FILENAME_is_found():
    """The likeliest leak of all: a file named after the run it came from.

    A first draft scanned contents only, and a tracked mesh whose filename
    carried both a catalog identifier and a coordinate field reported clean --
    doubly so, because a .ply is never opened. The contents here are innocent
    on purpose: the name is the whole hit.
    """
    mesh = f"{PLANTED_ID}{PLANTED_COORD}R3km_linear.ply"
    with tempfile.TemporaryDirectory() as td:
        root = _repo(Path(td), {
            mesh: "not really a mesh, and deliberately clean inside\n",
            f"notes_{PLANTED_ID}.md": "also clean inside\n",
            "fine.md": "clean name, clean contents\n",
        })
        hits = C.scan(C.tracked_files(str(root)), str(root))
        names = {p for p, _n, _w in hits}
        if mesh not in names:
            raise Failure("a .ply NAMED with an identifier shape was not "
                          "caught; a binary suffix must skip the CONTENTS, "
                          "never the name")
        if f"notes_{PLANTED_ID}.md" not in names:
            raise Failure("a text file named with an identifier shape was not "
                          "caught by its name")
        if "fine.md" in names:
            raise Failure("a clean file was flagged")
        for path, number, why in hits:
            if number == C.PATH_LINE and "NAME" not in why:
                raise Failure(f"a name hit did not say so: {why!r}")


def check_the_patterns_are_case_insensitive():
    """The deny list matches case-insensitively; this half must agree.

    Where they disagree is exactly where this half is the only gate that runs,
    so a lowercased filename or a ``str.lower()``-ed fixture key would ship.
    """
    lowered = PLANTED_ID.lower() + PLANTED_COORD.lower() + "r3km.ply"
    with tempfile.TemporaryDirectory() as td:
        root = _repo(Path(td), {"paths.md": f"see {lowered}\n"})
        hits = C.scan(C.tracked_files(str(root)), str(root))
        if not hits:
            raise Failure("a lowercased identifier shape was not caught; this "
                          "half and the deny list disagree on case")


def check_an_unreadable_named_path_exits_2():
    """Naming a path that cannot be read must not report clean.

    Explicit-path mode is how a pre-commit hook or a changed-files CI step
    would call this. A typo'd or renamed path used to be swallowed and counted
    as one clean file -- the same "learned to say yes" failure as a broken git,
    on a different road.
    """
    with tempfile.TemporaryDirectory() as td:
        code, out, err = _run(["--root", td, "does/not/exist.py"], cwd=td)
        if code != 2:
            raise Failure(f"expected exit 2 for an unreadable named path, "
                          f"got {code}; stdout={out!r}")
        if "clean" in out:
            raise Failure("it reported clean on a file it could not open")
        if "could not open" not in err and "could not be read" not in err:
            raise Failure(f"the reason was not stated: {err!r}")

    # And through the API, so a future caller cannot get [] instead.
    with tempfile.TemporaryDirectory() as td:
        try:
            C.check("does/not/exist.py", td)
        except C.Unreadable:
            pass
        else:
            raise Failure("check() returned a list for an unopenable path")


def check_one_line_with_both_shapes_is_reported_once():
    """The count has to mean what it says.

    A site-named path carries both shapes, which is the common case, and both
    patterns firing on it used to report the same line twice -- "2 lines in 1
    file" about one line.
    """
    with tempfile.TemporaryDirectory() as td:
        root = _repo(Path(td), {
            "notes.md": f"the mesh is {PLANTED_ID}{PLANTED_COORD}R3km.ply\n"})
        hits = [h for h in C.scan(C.tracked_files(str(root)), str(root))
                if h[1] != C.PATH_LINE]
        if len(hits) != 1:
            raise Failure(f"one line matching both shapes produced {len(hits)} "
                          f"reports: {hits}")


def check_spice_text_kernels_are_read_as_text():
    """.tf/.tpc are TEXT kernels, and a frame kernel can name a site frame.

    The deny-list check's own suffix table calls them binary; this one does
    not, and that difference is deliberate rather than an oversight.
    """
    for suffix in (".tf", ".tpc"):
        if suffix in C.BINARY_SUFFIXES:
            raise Failure(f"{suffix} is a SPICE TEXT kernel and must not be "
                          f"in BINARY_SUFFIXES")
    with tempfile.TemporaryDirectory() as td:
        root = _repo(Path(td), {
            "frames.tf": f"FRAME_{PLANTED_ID}_TOPO = -1000\n"})
        if not C.scan(C.tracked_files(str(root)), str(root)):
            raise Failure("an identifier shape inside a .tf text kernel was "
                          "not caught")


def check_the_inline_escape_hatch_exempts_one_line():
    with tempfile.TemporaryDirectory() as td:
        root = _repo(Path(td), {
            "note.md": f"a coincidence {PLANTED_ID}  <!-- {C.INLINE_ALLOW} -->\n"
                       f"and a real one {PLANTED_ID}\n",
        })
        hits = C.scan(C.tracked_files(str(root)), str(root))
        lines = sorted(n for _p, n, _w in hits)
        if lines != [2]:
            raise Failure(f"expected only line 2 to be reported, got {lines}")


def check_allow_paths_exempts_a_named_file():
    with tempfile.TemporaryDirectory() as td:
        root = _repo(Path(td), {"vendor.py": f"# {PLANTED_ID}\n"})
        saved = dict(C.ALLOW_PATHS)
        try:
            C.ALLOW_PATHS["vendor.py"] = "test"
            hits = C.scan(C.tracked_files(str(root)), str(root))
        finally:
            C.ALLOW_PATHS.clear()
            C.ALLOW_PATHS.update(saved)
        if hits:
            raise Failure(f"ALLOW_PATHS did not exempt the file it names: {hits}")


def check_the_report_never_prints_the_matching_text():
    """A gate against publishing an identifier must not publish it in the log."""
    with tempfile.TemporaryDirectory() as td:
        root = _repo(Path(td), {"dirty.py": f'SITE = "{PLANTED_ID}"\n'})
        _code, out, err = _run(["--root", str(root)], cwd=str(root))
        if PLANTED_ID in out or PLANTED_ID in err:
            raise Failure("the report quoted the matching text; a CI log is "
                          "readable by everyone who can read the build")


def check_the_checker_and_this_file_are_clean_with_no_exemptions():
    """The gate is not passing by exempting itself.

    ``ALLOW_PATHS`` is empty on purpose: a regex for a shape does not have that
    shape, and strings assembled at run time are not literals. If either file
    ever needs an exemption to pass, that is a design change and this arm is
    the one that will say so.
    """
    if C.ALLOW_PATHS:
        raise Failure(f"ALLOW_PATHS is not empty: {sorted(C.ALLOW_PATHS)}; "
                      f"every entry is a hole, and neither of the gate's own "
                      f"files should need one")
    for name in ("scripts/check_site_id_patterns.py",
                 "scripts/test_check_site_id_patterns.py"):
        hits = C.check(name, str(REPO_ROOT))
        if hits:
            raise Failure(f"{name} matches its own patterns at lines "
                          f"{[n for n, _w in hits]}")


def check_ordinary_text_does_not_match():
    """The patterns are narrow enough to survive contact with the repository.

    A gate that fires on normal prose gets switched off, and the switching-off
    is quiet. These are the near-misses worth pinning: a version-like token, a
    northern coordinate field, and the catalog prefix followed by punctuation
    rather than a region code.
    """
    benign = [
        "A03 is the catalog, not an identifier",
        "the prefix A03-something appears in prose",
        # Northern hemisphere, and so not the shape. The degree value is
        # invented: writing a REAL one here would pass this shape-only gate and
        # fail the deny-list one, which is what happened on the first draft.
        "core_89.999N_R3km_linear.ply",
        "a duration of 8.5 days",
        "version A0300 of the format",       # digits with no region code
        f"{SAFE_TAG}_00.000S_000.000E_R3km_linear.ply",
    ]
    with tempfile.TemporaryDirectory() as td:
        root = _repo(Path(td), {"prose.md": "\n".join(benign) + "\n"})
        hits = C.scan(C.tracked_files(str(root)), str(root))
        if hits:
            raise Failure(f"ordinary text was flagged at lines "
                          f"{[n for _p, n, _w in hits]}: too broad")


def check_a_usage_error_exits_2():
    code, _out, _err = _run(["--nonsense"])
    if code != 2:
        raise Failure(f"expected exit 2 on an unrecognized flag, got {code}")


CHECKS = [
    check_a_planted_catalog_identifier_is_found,
    check_a_planted_coordinate_field_is_found,
    check_a_site_shaped_FILENAME_is_found,
    check_the_patterns_are_case_insensitive,
    check_the_exit_code_is_1_on_a_hit_and_0_when_clean,
    check_git_failure_exits_2_and_never_reports_clean,
    check_an_unreadable_named_path_exits_2,
    check_one_line_with_both_shapes_is_reported_once,
    check_spice_text_kernels_are_read_as_text,
    check_the_inline_escape_hatch_exempts_one_line,
    check_allow_paths_exempts_a_named_file,
    check_the_report_never_prints_the_matching_text,
    check_the_checker_and_this_file_are_clean_with_no_exemptions,
    check_ordinary_text_does_not_match,
    check_a_usage_error_exits_2,
]


def main() -> int:
    failed = 0
    for fn in CHECKS:
        try:
            fn()
        except SystemExit as exc:
            # argparse exits on --nonsense; _run catches the code, so a
            # SystemExit escaping here is a real defect, not the usage arm.
            failed += 1
            print(f"FAIL {fn.__name__}: escaped SystemExit({exc.code})")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {fn.__name__}:\n  {exc}")
        else:
            print(f"PASS {fn.__name__}")
    print(f"\n[test_check_site_id_patterns] {len(CHECKS) - failed}/"
          f"{len(CHECKS)} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
