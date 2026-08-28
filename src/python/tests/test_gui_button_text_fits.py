#!/usr/bin/env python3
"""Text that is drawn on a button fits on that button -- in both directions.

**Why this file exists, round one (width).** The run-type buttons showed their
summary as a second line of ``setText``, and a QPushButton does not wrap: it
draws each newline-separated line on ONE line and elides whatever does not fit.
Measured on 2026-08-27, in the button's own 12 pt bold font, at the button's own
minimum width:

===============  =========  ==========  =======
run type         one line   available   verdict
===============  =========  ==========  =======
time_series        962 px      156 px   cut off
integrated        1348 px      156 px   cut off
sweep             1122 px      156 px   cut off
===============  =========  ==========  =======

All three. The reader saw a fragment ending in an ellipsis.

**Why it exists, round two (HEIGHT) -- and why round one's gate did not catch
it.** The width fix shipped, and the text was still cut off in Paul's window,
vertically. This file passed throughout, because it measured
``button.minimumHeight()`` on a button built in a bare ``QApplication`` -- and
the application the user runs has a STYLESHEET. ``ModernStyle.get_stylesheet()``
carries ``QPushButton { padding: 8px 16px; min-height: 28px; }``, and
``QStyleSheetStyle`` applies a stylesheet box by OVERWRITING the widget's
minimum size. Measured 2026-08-28:

=========================  ===========  ==========  =======
                           no stylesheet  styled     verdict
=========================  ===========  ==========  =======
button minimumHeight            96 px      46 px    overwritten
title label height              19 px       9 px    needs 16
summary label height            16 px       9 px    needs 16
=========================  ===========  ==========  =======

So the constructor's ``setMinimumHeight(96)`` was discarded in the real
application and kept in the test. **The stylesheet is therefore applied here.**
It is an input to the geometry, exactly as the font is, and leaving it out is
what made this gate agree with a window it did not resemble.

**How this can be checked without pixels.** ``VisualizationPanel`` crashes
headless, so nothing here can be confirmed by looking at a window. What CAN be
measured offscreen is what Qt itself uses to lay the text out: ``QFontMetrics``
on the widget's own ``QFont``, and -- for the height -- the geometry the layout
actually hands each child label after a real ``show()``. That is not a
substitute for the pixels; it is the input to them. **Paul's click-test is the
visual acceptance.** What is proved here is that the numbers Qt lays out from
leave room for the text, at every width tested.

Run directly::

    python3 src/python/tests/test_gui_button_text_fits.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "python"))
sys.path.insert(0, str(REPO_ROOT / "gui"))


class Skip(Exception):
    """A check that cannot run here, as distinct from one that failed."""


_APP = None

#: Widths the run-type column plausibly takes, in px. The narrowest is the
#: button's own ``minimumWidth`` (180) plus the margins its container adds --
#: below that the window cannot be squeezed -- and the widest is a maximized
#: window on this machine. A wrapping label is TALLEST at the NARROWEST width,
#: so the low end is the one that matters; the rest are there because a fit
#: that held only at one width would be an accident.
PROBE_WIDTHS = (200, 320, 480, 720, 1100, 1600)


def _qt():
    """A QApplication with the REAL stylesheet on it.

    The stylesheet is not decoration here. See the module docstring: it changes
    the button's minimum height from 96 to 46 and is the reason the height
    defect was invisible to this file for a day.
    """
    global _APP
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        import PyQt6
    except ImportError as exc:                              # pragma: no cover
        raise Skip(f"PyQt6 is unavailable here ({exc})")
    plug = Path(PyQt6.__file__).parent / "Qt6" / "plugins"
    os.environ.setdefault("QT_PLUGIN_PATH", str(plug))
    os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(plug / "platforms"))
    from PyQt6.QtWidgets import QApplication
    if _APP is None:
        _APP = QApplication.instance() or QApplication([])
        from gui_core.styles import ModernStyle
        _APP.setStyleSheet(ModernStyle.get_stylesheet())
    return _APP


def _fit(button, text):
    """``(fits, detail)`` for `text` drawn as `button`'s summary, by metrics.

    The HORIZONTAL half of the contract, measured at the button's minimum
    width -- the narrowest it can ever be squeezed to, and so the widest the
    wrapped paragraph can ever be forced to be. Two ways a wrapped paragraph
    fails, and both are checked:

    * it is TALLER than the room the button reserves for it;
    * a single WORD is wider than the line, which word wrap cannot break, so
      that word is elided however tall the button is.
    """
    from PyQt6.QtCore import QRect, Qt
    from PyQt6.QtGui import QFontMetrics

    pad = button.TEXT_PADDING_PX
    avail_w = button.minimumWidth() - pad
    title_h = QFontMetrics(button.font()).height()
    # `required_height` is what the widget itself will now demand at this
    # width, so the room really available is measured against that rather than
    # against a constant that the stylesheet is free to overwrite.
    avail_h = (button.required_height(button.minimumWidth())
               - pad - title_h - button.LABEL_SPACING_PX)

    metrics = QFontMetrics(button.subtitle_font())
    box = metrics.boundingRect(QRect(0, 0, avail_w, 10_000),
                               int(Qt.TextFlag.TextWordWrap), text)
    longest = max((metrics.horizontalAdvance(w) for w in text.split()),
                  default=0)
    detail = (f"{len(text)} chars wrap to {box.width()}x{box.height()} px in "
              f"{avail_w}x{avail_h} px (longest word {longest} px)")
    return (box.height() <= avail_h and longest <= avail_w), detail


def _laid_out(width, factory=None):
    """Build the real run-type column at `width` and return what Qt gave it.

    ``[Measured, ...]`` -- PLAIN NUMBERS, read off after a genuine ``show()``,
    so they are the geometry the layout produced and not the geometry the
    widget asked for. Nothing live is returned: the host widget owns the
    buttons and dies with this call, and handing a caller a button whose parent
    has been destroyed raises "wrapped C/C++ object has been deleted" from the
    failure message that was trying to describe it.

    `factory` builds one button from a run type, so a caller can measure a
    DELIBERATELY BROKEN button and prove this check would catch it.
    """
    from PyQt6.QtCore import QRect, Qt
    from PyQt6.QtGui import QFontMetrics
    from PyQt6.QtWidgets import QVBoxLayout, QWidget

    app = _qt()
    from gui_core.body_cards import RUN_TYPES
    from gui_panels.body_cards_panel import _BigButton

    if factory is None:
        def factory(run_type):
            return _BigButton(run_type.title, run_type.summary)

    host = QWidget()
    box = QVBoxLayout(host)
    buttons = []
    for run_type in RUN_TYPES:
        button = factory(run_type)
        box.addWidget(button)
        buttons.append((run_type, button))
    box.addStretch()
    host.resize(width, 1200)
    host.show()
    app.processEvents()

    out = []
    for run_type, button in buttons:
        summary, title = button.subtitle_label, button.title_label
        metrics = QFontMetrics(summary.font())
        need = metrics.boundingRect(
            QRect(0, 0, summary.width(), 10_000),
            int(Qt.TextFlag.TextWordWrap), run_type.summary).height()
        out.append(Measured(
            key=run_type.key,
            button_w=button.width(), button_h=button.height(),
            title_h=title.height(),
            title_need=QFontMetrics(title.font()).height(),
            summary_h=summary.height(), summary_need=need))
    host.close()
    return out


class Measured:
    """One button's laid-out geometry, as numbers that outlive the widget."""

    def __init__(self, key, button_w, button_h, title_h, title_need,
                 summary_h, summary_need):
        self.key = key
        self.button_w = button_w
        self.button_h = button_h
        self.title_h = title_h
        self.title_need = title_need
        self.summary_h = summary_h
        self.summary_need = summary_need

    @property
    def clipped(self):
        return (self.title_h < self.title_need
                or self.summary_h < self.summary_need)

    def describe(self, width):
        return (f"{self.key} at host width {width}: button "
                f"{self.button_w}x{self.button_h} gave the title "
                f"{self.title_h} px (needs {self.title_need}) and the summary "
                f"{self.summary_h} px (needs {self.summary_need})")


def test_every_run_type_summary_fits_its_button():
    """The horizontal contract, at the button's minimum width."""
    _qt()
    from gui_core.body_cards import RUN_TYPES
    from gui_panels.body_cards_panel import _BigButton

    failures = []
    for run_type in RUN_TYPES:
        button = _BigButton(run_type.title, run_type.summary)
        assert button.subtitle_label is not None, (
            f"the {run_type.key} button has no wrapping summary label, so its "
            f"summary is a second line of setText and will be elided")
        fits, detail = _fit(button, run_type.summary)
        if not fits:
            failures.append(f"{run_type.key}: {detail}\n"
                            f"        {run_type.summary!r}")
        else:
            print(f"    [fit] {run_type.key}: {detail}")
    assert not failures, "\n  ".join(
        ["a run-type summary does not fit its button at the button's minimum "
         "size, so it is cut off:"] + failures)


def test_run_type_buttons_are_tall_enough_at_every_plausible_width():
    """The VERTICAL contract, on the geometry the layout really produced.

    This is the check that was missing while Paul was looking at clipped text.
    It asserts against ``label.height()`` -- what Qt handed the label -- rather
    than against any number the widget asked for, because the defect was
    precisely that the number the widget asked for was thrown away.
    """
    _qt()
    failures = []
    for width in PROBE_WIDTHS:
        rows = _laid_out(width)
        failures += [row.describe(width) for row in rows if row.clipped]
        print(f"    [height] host {width:4d} px: all {len(rows)} run types "
              f"have room for title and summary "
              f"(summary {rows[0].summary_h}/{rows[0].summary_need} px)")
    assert not failures, "\n  ".join(
        ["a run-type button is too SHORT for its own text, so the text is cut "
         "off vertically:"] + failures)


def test_the_height_check_catches_a_button_that_does_not_re_assert():
    """The negative control for the HEIGHT check -- the round-two defect.

    A check that has only ever been seen to pass has not been tested, and the
    previous version of this file is proof: it passed for a day while the text
    was clipped in front of the user. So here is the exact broken button --
    one that sets its minimum height once, in the constructor, and lets the
    application stylesheet overwrite it -- and the height check above must
    report every one of them as clipped.
    """
    _qt()
    from gui_panels.body_cards_panel import _BigButton

    class _NoReassert(_BigButton):
        """`_BigButton` as it was: a constant, applied once, then discarded."""

        def _apply_required_height(self):
            pass                                   # the defect, reproduced

    def factory(run_type):
        button = _NoReassert(run_type.title, run_type.summary)
        button.setMinimumHeight(_BigButton.DESIGN_MIN_HEIGHT)
        return button

    rows = _laid_out(520, factory=factory)
    clipped = [row for row in rows if row.clipped]
    assert len(clipped) == len(rows), (
        "the height check did NOT catch a button that lets the stylesheet "
        "overwrite its minimum height -- so it would not have caught the "
        "defect it exists for. Reported clipped: "
        f"{[row.key for row in clipped]} of {[row.key for row in rows]}")
    print(f"    [control] all {len(clipped)} run types are reported clipped "
          f"without the re-assert: {clipped[0].describe(520)}")


def test_the_stylesheet_does_not_shrink_the_button_below_its_text():
    """The regression this round is about, named directly.

    With ``ModernStyle`` applied, a ``_BigButton`` must still be at least its
    design height. Before the fix it was 46 px -- ``min-height: 28px`` plus the
    stylesheet's padding and border -- because QStyleSheetStyle OVERWRITES the
    widget's minimum size rather than combining with it.
    """
    app = _qt()
    from PyQt6.QtWidgets import QVBoxLayout, QWidget
    from gui_core.body_cards import RUN_TYPES
    from gui_panels.body_cards_panel import _BigButton

    assert app.styleSheet(), (
        "this check is meaningless without the application stylesheet, which "
        "is the thing that used to discard the button's minimum height")

    host = QWidget()
    box = QVBoxLayout(host)
    button = _BigButton(RUN_TYPES[0].title, RUN_TYPES[0].summary)
    box.addWidget(button)
    box.addStretch()
    host.resize(520, 600)
    host.show()
    app.processEvents()

    floor = _BigButton.DESIGN_MIN_HEIGHT
    assert button.minimumHeight() >= floor, (
        f"the application stylesheet shrank the button's minimum height to "
        f"{button.minimumHeight()} px, below the {floor} px its text is laid "
        f"out for -- the text will be clipped vertically")
    assert button.height() >= floor, (
        f"the button was laid out {button.height()} px tall, below the "
        f"{floor} px floor")
    print(f"    [stylesheet] styled button holds {button.height()} px "
          f"(floor {floor}, minimumHeight {button.minimumHeight()})")
    host.close()


def test_a_long_summary_makes_the_button_grow_rather_than_clip():
    """The floor is a floor, not the answer.

    A summary too long for 96 px must push the button taller. Without this, a
    future edit to a run-type summary would be clipped again and the height
    check above would still pass, because it only ever sees text that fits.
    """
    _qt()
    from gui_panels.body_cards_panel import _BigButton

    short = _BigButton("Probe", "Short.")
    long_text = ("The same configuration run once per point of a grid, for "
                 "every terrain crossed with every latitude the shipped "
                 "configurations use, which is a great many words on purpose "
                 "so that this button cannot possibly fit them in its design "
                 "height and must therefore grow downward to hold them all.")
    tall = _BigButton("Probe", long_text)

    narrow = tall.minimumWidth()
    assert short.required_height(narrow) == _BigButton.DESIGN_MIN_HEIGHT, (
        f"a short summary should sit at the design floor, got "
        f"{short.required_height(narrow)}")
    grown = tall.required_height(narrow)
    assert grown > _BigButton.DESIGN_MIN_HEIGHT, (
        f"a {len(long_text)}-character summary did not make the button grow: "
        f"required_height({narrow}) = {grown}, floor "
        f"{_BigButton.DESIGN_MIN_HEIGHT}")
    # ...and narrower is taller, because the text wraps to more lines.
    assert tall.required_height(narrow) >= tall.required_height(narrow * 4), (
        "a narrower button did not need at least as much height as a wide "
        "one, so the height is not being derived from the wrapped text")
    print(f"    [growth] long summary needs {grown} px at {narrow} px wide, "
          f"{tall.required_height(narrow * 4)} px at {narrow * 4} px")


def test_the_fit_check_fails_on_text_that_does_not_fit():
    """The negative control.

    A check that has only ever been seen to pass has not been tested. These
    are the three summaries as they were written on 2026-08-27, and every one
    of them must be reported as not fitting -- otherwise this file would go on
    passing after somebody put a paragraph back.

    They are VERBATIM historical quotations, kept for their LENGTH. The third
    still reads "every Sutton terrain" because that is what the summary said on
    2026-08-27, before the label sweep of 2026-08-28 -- it is not a label this
    GUI shows anywhere, it is the measured text. Do not rewrite it to match the
    current wording: the replacement is shorter, and shortening the strings a
    negative control is built on is how the control quietly stops controlling.
    """
    _qt()
    from gui_panels.body_cards_panel import _BigButton

    was = [
        ("One temperature field per output sample, resolving the diurnal or "
         "seasonal cycle and letting any face be plotted against time."),
        ("The same simulation, but only the minimum, mean and maximum over "
         "the whole window are written. Every per-sample file is skipped, "
         "which is most of the disk a long run uses."),
        ("The same configuration run once per point of a grid -- for Europa, "
         "every Sutton terrain crossed with every latitude the shipped "
         "configurations use."),
    ]
    button = _BigButton("Probe", "probe")
    passed = [text for text in was if _fit(button, text)[0]]
    assert not passed, (
        "the fit check accepts text that was measured as overflowing:\n  "
        + "\n  ".join(repr(t) for t in passed))
    # ...and a single word longer than the line is caught even though it is
    # only one line tall.
    fits, detail = _fit(button, "a " + "x" * 60)
    assert not fits, f"an unbreakable 60-character word was accepted: {detail}"
    print(f"    [control] all 3 previous summaries and an unbreakable word "
          f"are reported as not fitting")


def test_a_summary_button_still_reports_its_title_and_can_be_clicked():
    """The two things the restructure could have broken, held to.

    ``text()`` is read by ``test_gui_body_first`` to check which run types are
    offered, and the child labels would swallow the mouse press and stop the
    flow dead at the run-type screen if they were not transparent to it.
    """
    _qt()
    from PyQt6.QtCore import Qt
    from gui_panels.body_cards_panel import _BigButton

    button = _BigButton("Parameter grid", "One run per point of a grid.")
    assert button.text() == "Parameter grid", button.text()
    assert "\n" not in button.text(), (
        f"a button title carries a second line again: {button.text()!r}")
    for label in (button.title_label, button.subtitle_label):
        assert label.testAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents), (
            "a label inside the button would swallow the click, so the "
            "run-type screen would not advance")

    seen = []
    button.clicked.connect(lambda _c=False: seen.append(True))
    button.click()
    assert seen == [True], "the button did not emit clicked()"

    # a body button -- no summary -- is unchanged: Qt draws its own text.
    plain = _BigButton("Europa")
    assert plain.subtitle_label is None
    assert plain.text() == "Europa"
    assert plain.required_height(200) == _BigButton.DESIGN_MIN_HEIGHT
    print("    [button] title readable, click delivered, body buttons "
          "unchanged")


CHECKS = [
    test_every_run_type_summary_fits_its_button,
    test_run_type_buttons_are_tall_enough_at_every_plausible_width,
    test_the_height_check_catches_a_button_that_does_not_re_assert,
    test_the_stylesheet_does_not_shrink_the_button_below_its_text,
    test_a_long_summary_makes_the_button_grow_rather_than_clip,
    test_the_fit_check_fails_on_text_that_does_not_fit,
    test_a_summary_button_still_reports_its_title_and_can_be_clicked,
]


def main():
    try:
        _qt()
    except Skip as exc:
        print(f"SKIP all: {exc}")
        return 0
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
