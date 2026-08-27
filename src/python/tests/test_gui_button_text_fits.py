#!/usr/bin/env python3
"""Text that is drawn on a button fits on that button.

**Why this file exists.** The run-type buttons showed their summary as a
second line of ``setText``, and a QPushButton does not wrap: it draws each
newline-separated line on ONE line and elides whatever does not fit. Measured
on 2026-08-27, in the button's own 12 pt bold font, at the button's own
minimum width:

===============  =========  ==========  =======
run type         one line   available   verdict
===============  =========  ==========  =======
time_series        962 px      156 px   cut off
integrated        1348 px      156 px   cut off
sweep             1122 px      156 px   cut off
===============  =========  ==========  =======

All three. The reader saw a fragment ending in an ellipsis.

**How this can be checked without pixels.** ``VisualizationPanel`` crashes
headless, so nothing here can be confirmed by looking at a window. What CAN be
measured offscreen is exactly what Qt itself uses to lay the text out:
``QFontMetrics`` on the same ``QFont`` the widget builds. That is not a
substitute for the pixels -- it is the input to them.

**The width this measures is the MINIMUM.** In a real window these buttons
stretch to the panel, several hundred pixels; ``minimumWidth`` is the narrowest
they can ever be squeezed to. Asserting at the minimum means the text fits at
every width the button can take, so this cannot pass on a wide developer window
and fail on a narrow one.

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


def _qt():
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
    _APP = QApplication.instance() or QApplication([])
    return _APP


def _fit(button, text):
    """``(fits, detail)`` for `text` drawn as `button`'s summary.

    Two ways a wrapped paragraph fails to fit, and both are checked:

    * it is TALLER than the room under the title, and the tail is clipped;
    * a single WORD is wider than the line, which word wrap cannot break, so
      that word is elided however tall the button is.
    """
    from PyQt6.QtCore import QRect, Qt
    from PyQt6.QtGui import QFontMetrics

    pad = button.TEXT_PADDING_PX
    avail_w = button.minimumWidth() - pad
    title_h = QFontMetrics(button.font()).height()
    avail_h = button.minimumHeight() - pad - title_h

    metrics = QFontMetrics(button.subtitle_font())
    box = metrics.boundingRect(QRect(0, 0, avail_w, 10_000),
                               int(Qt.TextFlag.TextWordWrap), text)
    longest = max((metrics.horizontalAdvance(w) for w in text.split()),
                  default=0)
    detail = (f"{len(text)} chars wrap to {box.width()}x{box.height()} px in "
              f"{avail_w}x{avail_h} px (longest word {longest} px)")
    return (box.height() <= avail_h and longest <= avail_w), detail


def test_every_run_type_summary_fits_its_button():
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


def test_the_fit_check_fails_on_text_that_does_not_fit():
    """The negative control.

    A check that has only ever been seen to pass has not been tested. These
    are the three summaries as they were written on 2026-08-27, and every one
    of them must be reported as not fitting -- otherwise this file would go on
    passing after somebody put a paragraph back.
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
    print("    [button] title readable, click delivered, body buttons "
          "unchanged")


CHECKS = [
    test_every_run_type_summary_fits_its_button,
    test_the_fit_check_fails_on_text_that_does_not_fit,
    test_a_summary_button_still_reports_its_title_and_can_be_clicked,
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
