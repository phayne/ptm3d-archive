#!/usr/bin/env python3
"""Check Markdown math for constructs GitHub's renderer mangles.

GitHub renders `$...$` and `$$...$$` with MathJax, but the Markdown pipeline
applies CommonMark backslash escaping to the document first. A backslash
followed by ASCII punctuation is replaced by that punctuation alone, so LaTeX
macros built from backslash + punctuation are destroyed before MathJax sees
them:

    \\,  ->  ,        (a literal comma appears in the equation)
    \\;  ->  ;
    \\:  ->  :
    \\!  ->  !
    \\{  ->  {        (grouping braces instead of literal braces)
    \\|  ->  |        (single bar instead of a norm)
    \\%  ->  %        (starts a MathJax comment)
    \\_  ->  _        (starts a subscript)
    \\\\ ->  \\       (row separator lost, except at end of line)

Use the letter-only equivalents instead, which pass through unchanged:

    \\,  -> \\thinspace        \\;  -> \\thickspace
    \\:  -> \\medspace         \\!  -> \\negthinspace
    \\{  -> \\lbrace           \\}  -> \\rbrace
    \\|  -> \\lVert / \\rVert

For percent signs and underscores, move the character outside the math span.
For multi-row environments, put each `\\\\` at the end of its own line inside a
`$$` block, where it survives.

Usage:
    python3 scripts/check_github_math.py [PATH ...]     # default: docs/ and *.md

Exits 1 if any hostile sequence is found.
"""

import os
import re
import sys

FENCE = re.compile(r"^(```|~~~)")
CODESPAN = re.compile(r"`[^`]*`")
MATH_BLOCK = re.compile(r"\$\$(.+?)\$\$", re.S)
MATH_INLINE = re.compile(r"(?<!\$)\$(?!\$)((?:[^$\n]|\\\$)+?)(?<!\$)\$(?!\$)")
HOSTILE = re.compile(r"\\([!-/:-@\[-`{-~])")

SUGGESTION = {
    ",": r"\thinspace",
    ";": r"\thickspace",
    ":": r"\medspace",
    "!": r"\negthinspace",
    "{": r"\lbrace",
    "}": r"\rbrace",
    "|": r"\lVert or \rVert",
    "%": "move the percent sign outside the math span",
    "_": "move the underscore outside the math span",
    "\\": "put the row separator at the end of a line inside a $$ block",
}


def mask_code(text):
    """Blank fenced code blocks and inline code spans, preserving offsets."""
    out = []
    in_fence = False
    fence_tok = None
    for line in text.split("\n"):
        stripped = line.strip()
        m = FENCE.match(stripped)
        if m:
            tok = m.group(1)
            if not in_fence:
                in_fence, fence_tok = True, tok
            elif tok == fence_tok:
                in_fence = False
            out.append(" " * len(line))
            continue
        out.append(" " * len(line) if in_fence else CODESPAN.sub(
            lambda mm: " " * len(mm.group(0)), line))
    return "\n".join(out)


def math_spans(masked):
    """Yield (start, end) offsets of math content, blocks first."""
    spans = []
    taken = bytearray(len(masked))
    for m in MATH_BLOCK.finditer(masked):
        spans.append((m.start(1), m.end(1)))
        for i in range(max(0, m.start() - 1), min(len(masked), m.end() + 1)):
            taken[i] = 1
    for m in MATH_INLINE.finditer(masked):
        if any(taken[i] for i in range(m.start(), m.end())):
            continue
        spans.append((m.start(1), m.end(1)))
    return spans


def scan(path):
    text = open(path, encoding="utf-8", errors="replace").read()
    masked = mask_code(text)
    findings = []
    for start, end in math_spans(masked):
        for m in HOSTILE.finditer(masked[start:end]):
            off = start + m.start()
            # A row separator survives when it is the last thing on its line.
            if m.group(1) == "\\" and text[off + 2:].lstrip(" \t").startswith("\n"):
                continue
            line = text.count("\n", 0, off) + 1
            findings.append((line, "\\" + m.group(1),
                             text[max(0, off - 25):off + 25].replace("\n", " ")))
    return sorted(findings)


def collect(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            for dp, dn, fn in os.walk(p):
                dn[:] = [d for d in dn if d != ".git"]
                files += [os.path.join(dp, f) for f in fn if f.endswith(".md")]
        elif p.endswith(".md"):
            files.append(p)
    return sorted(files)


def main(argv):
    paths = argv[1:] or ["docs", "README.md", "INSTALL.md", "CHANGELOG.md"]
    files = collect([p for p in paths if os.path.exists(p)])
    total = 0
    for f in files:
        for line, tok, ctx in scan(f):
            total += 1
            print(f"{f}:{line}: {tok} -> use {SUGGESTION.get(tok[1], 'a letter-only macro')}"
                  f"   ...{ctx}...")
    if total:
        print(f"\n{total} sequence(s) GitHub's Markdown pipeline will mangle "
              f"in {len(files)} file(s) scanned.")
        return 1
    print(f"OK: no GitHub-hostile math sequences in {len(files)} file(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
