#!/usr/bin/env python3
"""Check the repository for British spellings in prose, comments and UI text.

PTM3D is written in American English: documentation, tutorials, code comments,
docstrings and anything a user can see. This checker is the ratchet that keeps
it that way, in the same spirit as ``scripts/check_github_math.py``.

It reports a British spelling anywhere in a text file, with two escape hatches
for the cases where the British form is the correct text:

1. ``ALLOW`` below -- an explicit (path, substring) table for external strings
   we do not own (OGC WKT unit names, quoted paper titles) and for code
   identifiers that cannot be renamed without breaking callers or saved data.
2. An end-of-line marker ``american-english: allow`` on the offending line, for
   one-off cases in files that do not yet have an ALLOW entry. For a run of
   lines -- documentation that has to quote the spellings it forbids -- bracket
   them with ``american-english: allow-begin`` and ``american-english:
   allow-end`` (in a comment, so an HTML comment in Markdown).

This file and its self-test exempt themselves: the rule table and the test
fixtures *are* lists of British spellings.

Jupyter notebooks are parsed as JSON, not scanned as text: only cell sources
and *textual* outputs are read. A prose spelling a notebook prints is a real
hit and is reported; base64 image data is never scanned, because random bytes
match these patterns and the hits move every time a figure is redrawn.

`scripts/test_check_american_english.py` pins all of that, including the exit
code. Run it after touching this file.

A word that is correct in American English must never enter ``RULES``. Two that
bite people: ``cancellation`` keeps its double L in American usage (only
``canceled``/``canceling`` drop one), and ``analyses`` is the ordinary plural of
``analysis``. Both are absent from the table on purpose.

Usage::

    python3 scripts/check_american_english.py            # whole tracked tree
    python3 scripts/check_american_english.py PATH ...   # specific paths
    python3 scripts/check_american_english.py --exclude docs/tutorials/notebooks/

Exits 1 if any un-allowed British spelling is found, and 2 on a usage error --
including an unrecognized flag. The argument parser is deliberately STRICT: a
hand-rolled loop used to treat anything that was not ``--exclude`` as a PATH, so
``--help`` was opened as a file, found missing, skipped, and reported
``check_american_english: clean`` with exit 0. Every typo'd flag "passed".
"""

import argparse
import json
import os
import re
import subprocess
import sys

# --- the rule table ---------------------------------------------------------
# (regex, suggestion). Matched case-insensitively with word boundaries; the
# regex body is the *interior* of the word, so a rule fires on inflections too.
RULES = [
    # -our -> -or
    (r"\w*colour\w*", "color"),
    (r"\w*behaviour\w*", "behavior"),
    (r"\w*favour\w*", "favor"),
    (r"\w*flavour\w*", "flavor"),
    (r"\w*honour\w*", "honor"),
    (r"\w*labour\w*", "labor"),
    (r"\w*neighbour\w*", "neighbor"),
    (r"\w*vapour\w*", "vapor"),
    (r"\w*odour\w*", "odor"),
    (r"\w*armour\w*", "armor"),
    (r"\w*endeavour\w*", "endeavor"),
    (r"(rigour|vigour|humour|rumour|harbour|savour|clamour|splendour|candour"
     r"|ardour|fervour|valour|tumour|parlour)s?", "-or"),
    # -re -> -er
    (r"\w*centre\w*", "center"),
    (r"\w*metres?", "meter"),
    (r"\w*litres?", "liter"),
    (r"\w*fibre\w*", "fiber"),
    (r"(theatre|calibre|sombre|sceptre|spectre|lustre|meagre|manoeuvre)\w*",
     "-er"),
    # -ise / -isation / -yse -> -ize / -ization / -yze
    # `optimis` carries a lookahead the rest of this row does not need:
    # bare `\w*optimis\w*` also matches OPTIMISTIC, which is correct American
    # English (opt-imis-tic). That is exactly the class of false positive the
    # module docstring warns against, and it fires on ordinary numerics prose
    # ("optimistic by 8.8x"). The verb forms are still caught.
    (r"\w*(normalis|initialis|optimis(?=e|ed|es|ing|ation|er|ers)"
     r"|minimis|maximis|summaris|recognis"
     r"|organis|discretis|parameteris|visualis|utilis|stabilis|synchronis"
     r"|serialis|randomis|quantis|linearis|penalis|prioritis|standardis"
     r"|specialis|generalis|centralis|modularis|vectoris|factoris|digitis"
     r"|sanitis|authoris|itemis|criticis|magnetis|ionis|equalis|idealis"
     r"|localis|globalis|regularis|rasteris|memoris|systematis|tokenis"
     r"|vaporis|harmonis|homogenis|customis|finalis|formalis|mobilis"
     r"|modernis|neutralis|oxidis|popularis|pressuris|publicis|scrutinis"
     r"|sterilis|symbolis|theoris|verbalis|marginalis|apologis|legitimis"
     r"|characteris(?=e|ed|es|ing|ation)|realis(?=e|ed|es|ing)"
     r"|polaris(?=e|ed|es|ing|ation)|emphasis(?=e|ed|es|ing)"
     r"|synthesis(?=e|ed|es|ing)|hypothesis(?=e|ed|es|ing))\w*", "-ize"),
    # NOTE: `analyses` is deliberately absent -- it is the ordinary American
    # plural of `analysis`. Only the verb forms are British here.
    (r"\w*(analyse(?!s)|analysed|analysing|analyser|paralys(e(?!s)|ed|ing)"
     r"|catalys(e(?!s)|ed|ing)|hydrolys(e(?!s)|ed|ing))\w*", "-yze"),
    # doubled consonant before a suffix
    (r"\w*(modelling|modelled|modeller|labelling|labelled|labeller"
     r"|cancelling|cancelled|travelling|travelled|traveller|signalling"
     r"|signalled|fuelling|fuelled|levelling|levelled|totalling|totalled"
     r"|counselling|counselled|channelling|channelled|dialling|dialled)", "one L"),
    # single consonant where American doubles it
    (r"(fulfil|fulfils|fulfilment|instal|skilful|skilfully|wilful|wilfully"
     r"|enrol|enrols|enrolment|appal|appals|distil|distils|instil|instils)",
     "double L"),
    # -gue / -ce / misc
    (r"\w*artefact\w*", "artifact"),
    (r"\w*catalogue\w*", "catalog"),
    (r"analogues?", "analog"),
    (r"(defence|offence|pretence|licence|practise|practised|practising)s?",
     "-nse / -ice"),
    (r"programmes?", "program"),
    (r"orientat(ed|es|ing|e)", "oriented / orient"),
    (r"\w*grey\w*", "gray"),
    (r"(aluminium|sulphur|sulphide|sulphate|sulphuric|sulphurous|caesium)\w*",
     "US element name"),
    (r"(storey|storeys|plough|ploughed|ploughing|draught|draughts|kerb|kerbs"
     r"|tyre|tyres|gaol|aeroplane|jewellery|pyjamas|gramme|grammes|cheque"
     r"|cheques|chequered|connexion|inflexion|cypher)", "US spelling"),
    (r"(judgement|acknowledgement|ageing|enquiry|enquiries|enquire|sceptic"
     r"|sceptical|scepticism|mould|moulded|moulding|smoulder|smouldering)s?",
     "US spelling"),
    (r"(whilst|amongst|spelt|learnt|dreamt|leapt)", "US usage"),
]

_COMPILED = [(re.compile(r"\b(?:" + pat + r")\b", re.IGNORECASE), hint)
             for pat, hint in RULES]

# --- deliberate exceptions --------------------------------------------------
# (path suffix, substring that must appear on the line, why).
# A line matches an entry only if BOTH the path and the substring match, so an
# allowlist entry cannot accidentally silence an unrelated hit in the same file.
ALLOW = [
    # OGC WKT / EPSG spell the SI base unit "metre". These strings are parsed
    # by GDAL and by other people's tools; they are not our prose.
    ("gui/gui_core/data_exporter.py", 'UNIT["metre",1]',
     "OGC WKT unit name"),
    ("src/python/ptm_cog.py", 'UNIT["metre",1]',
     "OGC WKT unit name"),
    # Quoted titles from the cited literature.
    ("docs/appendix-d-exosphere-model.md", "Murphy, D.M. & Koop",
     "quoted paper title"),
    ("docs/appendix-d-exosphere-model-full.md", "Murphy, D.M. & Koop",
     "quoted paper title"),
    # Code identifiers. Renaming these breaks callers or saved data, so the
    # British spelling stays until someone does a deliberate API change.
    ("src/cpp/gpu/bvh_gpu_pool.cpp", "honourPause",
     "function identifier; never renamed"),
    ("gui/gui_core/data_exporter.py", "self._cancelled",
     "attribute identifier; never renamed"),
    ("gui/gui_panels/visualization_panel.py", "was_cancelled",
     "local identifier; never renamed"),
    ("src/python/radiation_exchange_schematic.py", "centre_window",
     "local identifier; never renamed"),
    ("src/python/tests/test_results_api.py", "test_same_geometry_in_metres",
     "test function identifier; never renamed"),
    ("src/python/tests/test_results_api.py", "test_forced_mesh_units_are_honoured",
     "test function identifier; never renamed"),
]

INLINE_ALLOW = "american-english: allow"
BLOCK_BEGIN = "american-english: allow-begin"
BLOCK_END = "american-english: allow-end"

# Two files are, by construction, corpora of British spellings: this one (its
# rule table) and its self-test (its fixtures). Scanning either is meaningless,
# and allow-listing line by line would rot on the next edit.
SELF_EXEMPT = {
    os.path.abspath(__file__),
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "test_check_american_english.py"),
}

SKIP_EXTENSIONS = {
    ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".ttf", ".zip",
    ".bin", ".fits", ".npy", ".npz", ".mp4", ".gz",
    # SPICE kernels and other binary/vendor data
    ".bsp", ".bpc", ".tpc", ".tls", ".bds", ".tf", ".ti", ".tsc", ".tm",
}

SKIP_PREFIXES = ("third_party/", "data/spice/", "data/kernels/")


class GitUnavailable(RuntimeError):
    """git could not list the tracked files, so the default scan cannot run.

    NOT the same thing as a clean tree, and the difference used to be invisible:
    `tracked_files()` returned `[]` on every git failure, `main` printed
    "no files to check" and returned 0, and a gate that had scanned NOTHING was
    indistinguishable at the exit code from one that had scanned 1,100 files and
    found nothing wrong. On this machine that is a live path, not a hypothetical:
    `/usr/bin/git` is the xcrun shim and exits 1, and the checkout sits on
    OneDrive, where the working tree can be evicted out from under a command.
    """


def tracked_files():
    """Every tracked file, via the git binary that works on this machine.

    Raises :class:`GitUnavailable` -- with what each candidate said -- rather
    than returning an empty list. A caller that wants the old behavior has to
    ask for it in as many words.
    """
    attempts = []
    for git in ("/Library/Developer/CommandLineTools/usr/bin/git", "git"):
        try:
            out = subprocess.check_output([git, "ls-files"],
                                          stderr=subprocess.PIPE)
        except OSError as exc:
            attempts.append(f"{git}: {exc}")
            continue
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or b"").decode(errors="replace").strip()
            attempts.append(f"{git}: exit {exc.returncode}"
                            + (f" ({detail.splitlines()[0]})" if detail else ""))
            continue
        return [p for p in out.decode().split("\n") if p]
    raise GitUnavailable("; ".join(attempts) or "no git binary was tried")


def walk(paths):
    for p in paths:
        if os.path.isdir(p):
            for root, dirs, names in os.walk(p):
                dirs[:] = [d for d in dirs if d not in (".git", "__pycache__")]
                for n in names:
                    yield os.path.join(root, n)
        else:
            yield p


def allowed(path, line):
    for suffix, needle, _why in ALLOW:
        if path.endswith(suffix) and needle in line:
            return True
    return INLINE_ALLOW in line


# --- notebooks --------------------------------------------------------------
# A .ipynb is JSON, and a re-executed one carries megabytes of base64 image
# data on single lines. Scanning it raw finds "British spellings" in PNG bytes
# -- `...SfACrGReYZMCj...` matches `grey` -- and those hits move every time the
# figure is redrawn, so they can never be allow-listed away. Parse the JSON and
# read only the parts a human wrote or the kernel printed.
#
# Output MIME types are ALLOW-listed, not block-listed: a new binary bundle
# (image/webp, application/vnd.plotly.v1+json) must default to "not scanned".
TEXTUAL_MIME = ("text/plain", "text/html", "text/markdown", "text/latex")

# text/html legitimately carries <img src="data:image/png;base64,...">, so
# strip embedded data URIs even from the textual bundles.
_DATA_URI = re.compile(r"data:[\w.+-]+/[\w.+-]+;base64,[A-Za-z0-9+/=\s]*")


def _as_lines(value):
    """nbformat stores multi-line text as a list of strings, or as one string."""
    if isinstance(value, str):
        return value.split("\n")
    if isinstance(value, list):
        out = []
        for item in value:
            if isinstance(item, str):
                out.extend(item.split("\n"))
        return out
    return []


def notebook_lines(text):
    """Prose-bearing lines of a notebook, in file order.

    Yields the cell sources and the *textual* outputs -- a printed British
    spelling is a real hit and must still be caught. Image, PDF and any other
    non-textual output bundle is skipped entirely.

    Raises ValueError if the file is not a notebook we understand, so the
    caller can fall back rather than silently scanning nothing.
    """
    nb = json.loads(text)
    if not isinstance(nb, dict) or "cells" not in nb:
        raise ValueError("not an nbformat notebook")
    for cell in nb.get("cells", []):
        if not isinstance(cell, dict):
            continue
        for line in _as_lines(cell.get("source")):
            yield line
        for out in cell.get("outputs", []) or []:
            if not isinstance(out, dict):
                continue
            kind = out.get("output_type")
            if kind == "stream":
                for line in _as_lines(out.get("text")):
                    yield line
            elif kind in ("execute_result", "display_data"):
                data = out.get("data") or {}
                for mime in TEXTUAL_MIME:
                    if mime in data:
                        for line in _as_lines(data[mime]):
                            yield _DATA_URI.sub("", line)
            elif kind == "error":
                yield str(out.get("ename", ""))
                yield str(out.get("evalue", ""))
                for line in _as_lines(out.get("traceback")):
                    yield line


def _number_against_source(lines, raw):
    """Best-effort raw-file line numbers for text pulled out of a notebook.

    nbformat writes each source line as its own JSON string on its own file
    line, so the escaped text can be found in the raw file. The pointer only
    advances, because notebook order is file order. Falls back to 0 -- shown as
    a bare path -- when a line cannot be located, which is honest about the
    fact that the number is a convenience, not a parse position.
    """
    raw_lines = raw.split("\n")
    out, ptr = [], 0
    for line in lines:
        lineno = 0
        if line.strip():
            needle = json.dumps(line)[1:-1]
            for i in range(ptr, len(raw_lines)):
                if needle in raw_lines[i]:
                    lineno, ptr = i + 1, i
                    break
        out.append((lineno, line))
    return out


def check(path):
    if os.path.abspath(path) in SELF_EXEMPT:
        return []
    if os.path.splitext(path)[1].lower() in SKIP_EXTENSIONS:
        return []
    norm = path.replace(os.sep, "/")
    if any(norm.startswith(p) or ("/" + p) in norm for p in SKIP_PREFIXES):
        return []
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except (OSError, UnicodeError):
        return []
    if "\x00" in text[:4096]:          # binary payload with a text extension
        return []

    if norm.endswith(".ipynb"):
        try:
            lines = list(notebook_lines(text))
        except (ValueError, json.JSONDecodeError) as exc:
            # Never fall through to a raw scan of a notebook we could not
            # parse -- that is the base64 false-positive path. Fail loudly.
            print(f"{norm}: cannot parse as a notebook ({exc}); NOT scanned",
                  file=sys.stderr)
            return []
        numbered = _number_against_source(lines, text)
    else:
        numbered = list(enumerate(text.split("\n"), 1))

    found = []
    in_block = False
    for lineno, line in numbered:
        if BLOCK_BEGIN in line:
            in_block = True
            continue
        if BLOCK_END in line:
            in_block = False
            continue
        if in_block or allowed(norm, line):
            continue
        # Dedupe on the EXACT text, not a case-folded key: "Cancelled" and
        # "cancelled" on one line are two separate things to fix, and folding
        # them together hides the second one.
        seen = set()
        for rx, hint in _COMPILED:
            for m in rx.finditer(line):
                if m.group(0) in seen:
                    continue
                seen.add(m.group(0))
                found.append((lineno, m.group(0), hint, line.strip()[:120]))
    return found


def _parser():
    p = argparse.ArgumentParser(
        prog="check_american_english.py",
        description="Report British spellings in prose, comments and UI text.",
        epilog="Exit status: 0 clean, 1 British spelling(s) found, 2 usage "
               "error, 3 the tracked-file list could not be obtained (git "
               "failed, or reported no tracked files) so NOTHING was scanned. "
               "A path that begins with '-' must be given after '--'.")
    p.add_argument("paths", nargs="*", metavar="PATH",
                   help="files or directories to check "
                        "(default: every git-tracked file)")
    p.add_argument("--exclude", action="append", default=[], metavar="PREFIX",
                   help="skip ARGUMENTS whose path starts with PREFIX; "
                        "repeatable. Applied to the argument list, NOT to the "
                        "files found inside a directory argument -- so it "
                        "filters the default whole-tracked-tree scan, and it "
                        "cannot exclude a subdirectory of a PATH you name")
    return p


def main(argv):
    # STRICT. argparse exits 2 with a usage message on an unrecognized flag and
    # 0 on --help; the loop this replaced silently filed both under "a path to
    # check" and then printed "clean". A gate that cannot be misinvoked loudly
    # is a gate that reports success for arguments it never understood.
    #
    # `parse_intermixed_args`, not `parse_args`: the hand-rolled loop accepted
    # flags and paths in any order, and callers use that
    # (`... a.py --exclude docs/ b.py`). Plain `parse_args` rejects the trailing
    # path with the SAME exit code as a typo, which would turn this from a
    # strictness fix into a broken CI gate.
    args = _parser().parse_intermixed_args(argv[1:])
    excludes = args.exclude
    # The DEFAULT scan is the whole tracked tree, and "I could not get that
    # list" has to be told apart from "I got it and it was clean". Both used to
    # be exit 0 with a line on stderr; a wrapper reading only $? could not see
    # the difference, and neither could CI. Exit 3, its own code, so a caller
    # can act on it without parsing text.
    if args.paths:
        paths = args.paths
        scanning_tracked_tree = False
    else:
        scanning_tracked_tree = True
        try:
            paths = tracked_files()
        except GitUnavailable as exc:
            print(f"check_american_english: FAILED TO RUN -- the tracked-file "
                  f"list could not be obtained, so NOTHING was checked. "
                  f"git said: {exc}", file=sys.stderr)
            return 3
        if not paths:
            print("check_american_english: FAILED TO RUN -- git listed no "
                  "tracked files, so NOTHING was checked. Run this from inside "
                  "the checkout, or name the paths to scan.", file=sys.stderr)
            return 3
    if excludes:
        paths = [p for p in paths
                 if not any(p.replace(os.sep, "/").startswith(x)
                            for x in excludes if x)]
    if not paths:
        # Reachable two ways, and only one of them is the caller's own doing:
        # --exclude that filters everything out. That stays exit 0 -- the user
        # asked for it and the message says what happened -- but a DEFAULT scan
        # emptied by --exclude scanned nothing it was asked to, and says so.
        if scanning_tracked_tree:
            print("check_american_english: FAILED TO RUN -- --exclude removed "
                  "every tracked file, so NOTHING was checked.", file=sys.stderr)
            return 3
        print("check_american_english: no files to check "
              "(--exclude removed every path named)", file=sys.stderr)
        return 0
    total = 0
    for path in sorted(set(walk(paths))):
        for lineno, word, hint, line in check(path):
            total += 1
            print(f"{path}:{lineno}: '{word}' -> {hint}\n    {line}")
    if total:
        print(f"\n{total} British spelling(s) found. PTM3D uses American "
              f"English in docs, comments, docstrings and user-visible text.")
        print("If a hit is correct as written (a quoted title, an external "
              "format literal, a code identifier), add it to ALLOW in "
              "scripts/check_american_english.py or put "
              f"'{INLINE_ALLOW}' on the line.")
        return 1
    print("check_american_english: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
