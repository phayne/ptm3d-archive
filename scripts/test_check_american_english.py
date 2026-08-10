#!/usr/bin/env python3
"""Self-test for scripts/check_american_english.py. No pytest, no deps.

The cases that matter, in the order they cost us something:

  * A notebook is JSON with megabytes of base64 image data on single lines.
    Scanning it raw reports British spellings that are really PNG bytes --
    `...SfACrGReYZMCj...` matches `grey` -- and those hits move whenever the
    figure is redrawn, so they can never be allow-listed away. Four of them
    were live on 2026-08-09.
  * The opposite failure is worse and quieter: "fixing" that by not reading
    notebooks at all. A British spelling a notebook *prints* is a real hit.
    Several cases below exist only to prove the scanner still reads.
  * `cancellation` and `analyses` are already correct American English.
    A naive rule "corrects" them, which is a regression, not a fix.
  * The exit code has to be non-zero when there are hits. A gate that prints
    and returns 0 trains people to ignore it.

Usage:  python3 scripts/test_check_american_english.py
"""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(HERE, "check_american_english.py")

spec = importlib.util.spec_from_file_location("chk", TARGET)
chk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chk)

_fail = 0


def check(cond, what):
    global _fail
    print(("ok:   " if cond else "FAIL: ") + what)
    if not cond:
        _fail += 1


def write(tmp, name, text):
    path = os.path.join(tmp, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


def hits(path):
    return [h[1] for h in chk.check(path)]


def notebook(cells):
    return json.dumps({"cells": cells, "metadata": {}, "nbformat": 4,
                       "nbformat_minor": 5}, indent=1)


def md_cell(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code_cell(source, outputs=None):
    return {"cell_type": "code", "metadata": {}, "execution_count": 1,
            "source": source, "outputs": outputs or []}


# A real base64 PNG prefix plus the exact byte run that produced the live
# false positive. "GReY" is inside it; there is no British spelling here.
B64_WITH_GREY = (
    "iVBORw0KGgoAAAANSUhEUgAABdsAAAGHCAYAAABI74s6AAAAOXRFWHRTb2Z0d2FyZQBN"
    "YXRwbG90bGliIHZlcnNpb24zLjcuNSwgaHR0cH"
    "vDyurZeemymEI4EqXLZe2ZjOStkrfC61SfACrGReYZMCj6b0BZKIhi"
    "cmiaIH8P3DevZrDyGy6c5SDvYVtu8XJn6aWu0XRKEgrEYExeej9VTmDQWKxVdTD3"
)

tmp = tempfile.mkdtemp(prefix="ae_selftest_")

# --- 1. plain text ----------------------------------------------------------
p = write(tmp, "clean.md", "The color of the gray sphere, normalized.\n")
check(hits(p) == [], "a clean markdown file reports nothing")

p = write(tmp, "dirty.md", "The colour of the grey sphere.\n")
check(sorted(h.lower() for h in hits(p)) == ["colour", "grey"],
      "a British spelling in markdown is reported")

p = write(tmp, "behaviour.py", "# legacy behaviour is normalised here\n")
check(sorted(h.lower() for h in hits(p)) == ["behaviour", "normalised"],
      "British spellings in a Python comment are reported")

# --- 2. words that are ALREADY American -------------------------------------
p = write(tmp, "correct.md",
          "Cancellation keeps both L's. These analyses are plural.\n"
          "The run was canceled and is not canceling again.\n")
check(hits(p) == [],
      "`cancellation` and `analyses` are NOT flagged (already American)")

# --- 3. notebooks: the base64 regression ------------------------------------
nb = notebook([code_cell(
    ["import matplotlib.pyplot as plt\n", "plt.plot([1, 2, 3])\n"],
    [{"output_type": "display_data", "metadata": {},
      "data": {"image/png": B64_WITH_GREY,
               "text/plain": ["<Figure size 640x480 with 1 Axes>"]}}])])
p = write(tmp, "nb_image.ipynb", nb)
check(hits(p) == [],
      "base64 image/png payload containing 'GReY' is NOT reported")
check("GReY" in B64_WITH_GREY,
      "  (guard: the payload really does contain the byte run that fired)")

# the same bytes, scanned as if it were a plain text file, DO fire --
# this proves the case above is the notebook parser working, not a weak rule
p_raw = write(tmp, "payload.txt", B64_WITH_GREY + "\n")
check(len(hits(p_raw)) > 0,
      "  (guard: those same bytes DO fire when not parsed as a notebook)")

# --- 4. notebooks: the scanner must still read ------------------------------
p = write(tmp, "nb_md.ipynb",
          notebook([md_cell(["# Colour and behaviour\n"])]))
check(sorted(h.lower() for h in hits(p)) == ["behaviour", "colour"],
      "British spelling in a markdown cell IS reported")

p = write(tmp, "nb_code.ipynb",
          notebook([code_cell(["# normalise the flux\n", "x = 1\n"])]))
check([h.lower() for h in hits(p)] == ["normalise"],
      "British spelling in a code cell comment IS reported")

p = write(tmp, "nb_stream.ipynb", notebook([code_cell(
    ["print('x')\n"],
    [{"output_type": "stream", "name": "stdout",
      "text": ["computing the colour scale\n"]}])]))
check([h.lower() for h in hits(p)] == ["colour"],
      "British spelling PRINTED in a stream output IS reported")

p = write(tmp, "nb_plain.ipynb", notebook([code_cell(
    ["r\n"],
    [{"output_type": "execute_result", "execution_count": 1, "metadata": {},
      "data": {"text/plain": ["Run(mesh centred at origin)"]}}])]))
check([h.lower() for h in hits(p)] == ["centred"],
      "British spelling in a text/plain result IS reported")

p = write(tmp, "nb_err.ipynb", notebook([code_cell(
    ["boom\n"],
    [{"output_type": "error", "ename": "ValueError",
      "evalue": "unrecognised body", "traceback": ["  colour failed\n"]}])]))
check(sorted(h.lower() for h in hits(p)) == ["colour", "unrecognised"],
      "British spelling in an error output IS reported")

# --- 5. notebooks: non-textual and embedded payloads stay out ---------------
p = write(tmp, "nb_pdf.ipynb", notebook([code_cell(
    ["f\n"],
    [{"output_type": "display_data", "metadata": {},
      "data": {"application/pdf": B64_WITH_GREY}}])]))
check(hits(p) == [], "application/pdf payload is not scanned")

p = write(tmp, "nb_unknown_mime.ipynb", notebook([code_cell(
    ["f\n"],
    [{"output_type": "display_data", "metadata": {},
      "data": {"application/vnd.plotly.v1+json": B64_WITH_GREY}}])]))
check(hits(p) == [],
      "an unknown output MIME type defaults to NOT scanned")

p = write(tmp, "nb_datauri.ipynb", notebook([code_cell(
    ["f\n"],
    [{"output_type": "display_data", "metadata": {},
      "data": {"text/html": [
          '<img src="data:image/png;base64,' + B64_WITH_GREY + '">']}}])]))
check(hits(p) == [],
      "a base64 data URI inside text/html is stripped before scanning")

p = write(tmp, "nb_html.ipynb", notebook([code_cell(
    ["f\n"],
    [{"output_type": "display_data", "metadata": {},
      "data": {"text/html": ["<b>the colour scale</b>"]}}])]))
check([h.lower() for h in hits(p)] == ["colour"],
      "  (guard: real prose in text/html IS still reported)")

# --- 6. a notebook we cannot parse is refused, not raw-scanned --------------
p = write(tmp, "nb_broken.ipynb", "{not json at all " + B64_WITH_GREY)
check(hits(p) == [],
      "an unparseable .ipynb is not raw-scanned (no base64 false positives)")

# --- 7. escape hatches ------------------------------------------------------
p = write(tmp, "inline.md", "the colour scale  <!-- american-english: allow -->\n")
check(hits(p) == [], "an inline allow marker suppresses the line")

p = write(tmp, "block.md",
          "colour one\n"
          "<!-- american-english: allow-begin -->\n"
          "colour two\ncolour three\n"
          "<!-- american-english: allow-end -->\n"
          "colour four\n")
check(len(hits(p)) == 2, "allow-begin/allow-end suppresses only the block")

check(chk.check(TARGET) == [],
      "the checker exempts its own rule table")

# This file is a corpus of British spellings by construction. If it is not
# exempt, the gate fails on its own fixtures -- which it did, once.
check(chk.check(os.path.abspath(__file__)) == [],
      "the checker exempts this self-test's fixtures")

# --- 8. exit codes (end to end, no pipeline to confuse $?) ------------------
clean = write(tmp, "exit_clean.md", "color, gray, normalized\n")
dirty = write(tmp, "exit_dirty.md", "colour\n")
rc_clean = subprocess.run([sys.executable, TARGET, clean],
                          capture_output=True).returncode
rc_dirty = subprocess.run([sys.executable, TARGET, dirty],
                          capture_output=True).returncode
check(rc_clean == 0, f"exit code is 0 on a clean tree (got {rc_clean})")
check(rc_dirty != 0,
      f"exit code is NON-ZERO when a British spelling is present "
      f"(got {rc_dirty})")

out = subprocess.run([sys.executable, TARGET, dirty], capture_output=True)
check(b"colour" in out.stdout and out.returncode == 1,
      "the run that PRINTS a hit also FAILS (prints and returns 0 is the bug)")

if _fail == 0:
    print("\nALL AMERICAN-ENGLISH CHECKER TESTS PASSED")
    sys.exit(0)
print(f"\n{_fail} AMERICAN-ENGLISH CHECKER TEST(S) FAILED")
sys.exit(1)
