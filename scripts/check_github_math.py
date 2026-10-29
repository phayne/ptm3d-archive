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

Two further mistakes are reported:

    \\lbracec_j, \\thinspaceF     a letter-only macro glued to the next letter is
                                one undefined control word (write `\\lbrace c_j`);
                                checked in every math form, including ```math
                                fences and $`...`$, which are otherwise safe
    $|x|$ in a table row         the bare `|` ends the table cell; use \\lvert x \\rvert

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
# Letter-only replacements glued to a following letter form an undefined control word.
GLUED = re.compile(r"\\(thinspace|thickspace|medspace|negthinspace|negmedspace|negthickspace"
                   r"|lbrace|rbrace|lVert|rVert|lvert|rvert)(?=[A-Za-z])")
SAFE_MATH = re.compile(r"(?ms)^[ \t]*```math[ \t]*\n(.*?)^[ \t]*```|\$`([^`]+)`\$")

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
    "|": "\\lvert ... \\rvert (a bare | ends the table cell)",
    "glued": "a space after the control word (\\lbrace c_j, not \\lbracec_j)",
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
        line_start = text.rfind("\n", 0, start) + 1
        if (text[line_start:start].lstrip().startswith("|") and "|" in masked[start:end]
                and "\n" not in masked[start:end]):
            findings.append((text.count("\n", 0, start) + 1, "|",
                             text[max(0, start - 25):end + 25].replace("\n", " ")))
    # Glued control words are wrong in every math form, the verbatim ones included.
    regions = [(s, e) for s, e in math_spans(masked)]
    regions += [(m.start(m.lastindex), m.end(m.lastindex)) for m in SAFE_MATH.finditer(text)]
    for start, end in regions:
        for m in GLUED.finditer(text[start:end]):
            off = start + m.start()
            findings.append((text.count("\n", 0, off) + 1, "glued",
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
    paths = argv[1:] or ["docs", "README.md", "INSTALL.md", "CHANGELOG.md",
                         "src/cpp/exosphere/README.md"]
    files = collect([p for p in paths if os.path.exists(p)])
    total = 0
    for f in files:
        for line, tok, ctx in scan(f):
            total += 1
            key = tok if tok in ("|", "glued") else tok[1]
            print(f"{f}:{line}: {tok} -> use {SUGGESTION.get(key, 'a letter-only macro')}"
                  f"   ...{ctx}...")
    if total:
        print(f"\n{total} sequence(s) GitHub's Markdown pipeline will mangle "
              f"in {len(files)} file(s) scanned.")
        return 1
    print(f"OK: no GitHub-hostile math sequences in {len(files)} file(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
