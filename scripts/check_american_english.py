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

This file exempts itself: its rule table *is* a list of British spellings.

A word that is correct in American English must never enter ``RULES``. Two that
bite people: ``cancellation`` keeps its double L in American usage (only
``canceled``/``canceling`` drop one), and ``analyses`` is the ordinary plural of
``analysis``. Both are absent from the table on purpose.

Usage::

    python3 scripts/check_american_english.py            # whole tracked tree
    python3 scripts/check_american_english.py PATH ...   # specific paths
    python3 scripts/check_american_english.py --exclude docs/tutorials/notebooks/

Exits 1 if any un-allowed British spelling is found.
"""

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
    (r"\w*(normalis|initialis|optimis|minimis|maximis|summaris|recognis"
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
    ("scripts/jwst/gardening.py", '"Modelling secondary impacts"',
     "quoted section title from the source paper"),
    # Code identifiers. Renaming these breaks callers or saved data, so the
    # British spelling stays until someone does a deliberate API change.
    ("src/cpp/gpu/bvh_gpu_pool.cpp", "honourPause",
     "function identifier (PTM-198)"),
    ("gui/gui_core/data_exporter.py", "self._cancelled",
     "attribute identifier (PTM-198)"),
    ("gui/gui_panels/visualization_panel.py", "was_cancelled",
     "local identifier (PTM-198)"),
    ("src/python/radiation_exchange_schematic.py", "centre_window",
     "local identifier (PTM-198)"),
    ("src/python/tests/test_results_api.py", "test_same_geometry_in_metres",
     "test function identifier (PTM-198)"),
    ("src/python/tests/test_results_api.py", "test_forced_mesh_units_are_honoured",
     "test function identifier (PTM-198)"),
]

INLINE_ALLOW = "american-english: allow"
BLOCK_BEGIN = "american-english: allow-begin"
BLOCK_END = "american-english: allow-end"

# This checker's own rule table is, unavoidably, a list of British spellings.
SELF_PATH = os.path.abspath(__file__)

SKIP_EXTENSIONS = {
    ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".ttf", ".zip",
    ".bin", ".fits", ".npy", ".npz", ".mp4", ".gz",
    # SPICE kernels and other binary/vendor data
    ".bsp", ".bpc", ".tpc", ".tls", ".bds", ".tf", ".ti", ".tsc", ".tm",
}

SKIP_PREFIXES = ("third_party/", "data/spice/", "data/kernels/")


def tracked_files():
    """Every tracked file, via the git binary that works on this machine."""
    for git in ("/Library/Developer/CommandLineTools/usr/bin/git", "git"):
        try:
            out = subprocess.check_output([git, "ls-files"],
                                          stderr=subprocess.DEVNULL)
            return [p for p in out.decode().split("\n") if p]
        except (OSError, subprocess.CalledProcessError):
            continue
    return []


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


def check(path):
    if os.path.abspath(path) == SELF_PATH:
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
    found = []
    in_block = False
    for lineno, line in enumerate(text.split("\n"), 1):
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


def main(argv):
    args, excludes = [], []
    it = iter(argv[1:])
    for a in it:
        if a == "--exclude":
            excludes.append(next(it, ""))
        elif a.startswith("--exclude="):
            excludes.append(a.split("=", 1)[1])
        else:
            args.append(a)
    paths = args or tracked_files()
    if excludes:
        paths = [p for p in paths
                 if not any(p.replace(os.sep, "/").startswith(x)
                            for x in excludes if x)]
    if not paths:
        print("check_american_english: no files to check", file=sys.stderr)
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
