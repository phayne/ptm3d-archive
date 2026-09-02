"""A record of what the Flux cubes group does when you click its checkbox.

Three fixes have been made to that checkbox and the options still do not
appear when it is ticked on one person's machine, while every automated check
passes on another. That means the automated checks are not looking at what his
window is doing, so this writes down what the window actually does, in order,
with timestamps, and puts it in a file that can be read afterward.

It records:

* the checkbox's own signals, and what the checkbox says its state is at the
  moment each one arrives;
* every time the group or one of its option rows is shown or hidden, and which
  function asked for it;
* every read and write of the ``flux_cubes.enabled`` setting;
* every re-description of the group, and the value it was given to render;
* every scroll, with the position it aimed for and the position it reached;
* the state of everything half a second AFTER the click, which is what catches
  a control that changes back on its own.

Where it goes
-------------
Two places, so it is hard to lose:

* ``~/ptm3d_fluxcube_trace.log`` -- appended to, one line per event;
* standard error, which is the terminal the program was started from.

It is OFF by default, and was on by default from 2026-08-31 to 2026-09-02
while the checkbox was being chased. Set ``PTM_GUI_TRACE=1`` in the environment
to turn it on::

    PTM_GUI_TRACE=1 python3 gui/run_gui.py

Off means OFF: with the variable unset, `enabled()` is false, nothing is
written to the log or to standard error, and
``body_cards_panel._build_flux_cube_group`` does not even CONNECT the two
observer slots -- so ``switch.receivers(switch.toggled)`` is zero and the
group costs nothing at all. That property is what
``test_the_click_chain_is_ordered_by_a_guarantee_not_by_luck`` reads, and it
is now the DEFAULT-tested path rather than the exception.

A value other than the accepted spellings is a WARNING, not a silent OFF: a
user who writes ``PTM_GUI_TRACE=yes`` (which works) or ``PTM_GUI_TRACE=on``
(which works) or ``PTM_GUI_TRACE=please`` (which does not) is told which
happened, once, at import.
"""

from __future__ import annotations

import os
import sys
import time

from pathlib import Path

#: The one string to search the log for.
PREFIX = "[FLUXCUBE-TRACE]"

#: Where the record is written. Fixed rather than derived so it can be named
#: in an instruction without the reader having to work anything out.
LOG_PATH = Path.home() / "ptm3d_fluxcube_trace.log"

#: The setting this is all about.
KEY = "flux_cubes.enabled"

#: The spellings that turn the record on. Anything else is off, and anything
#: else that is not plainly an "off" spelling says so rather than being
#: quietly ignored -- a switch that does nothing and reports nothing is the
#: defect this whole module was written to investigate.
_ON = ("1", "true", "yes", "on")
_OFF = ("", "0", "false", "no", "off")


def _read_switch(raw):
    if raw is None:
        return False
    value = raw.strip().lower()
    if value in _ON:
        return True
    if value in _OFF:
        return False
    print(f"{PREFIX} PTM_GUI_TRACE={raw!r} is not a value this understands; "
          f"the record is OFF. Use one of {', '.join(_ON)} to turn it on.",
          file=sys.stderr)
    return False


_ENABLED = _read_switch(os.environ.get("PTM_GUI_TRACE"))
_started = False
_handle = None


def enabled():
    return _ENABLED


def _stream():
    """The log file, opened once and kept open. Never raises.

    A trace that stops the program it is tracing would be worse than no trace,
    so every failure here falls back to standard error alone.
    """
    global _handle, _started
    if not _ENABLED:
        return None
    if _handle is None:
        try:
            _handle = open(LOG_PATH, "a", buffering=1, encoding="utf-8")
        except OSError as exc:
            print(f"{PREFIX} could not open {LOG_PATH} ({exc}); writing to "
                  f"standard error only", file=sys.stderr)
            _handle = False          # remember the failure, do not retry
    if not _started:
        _started = True
        _emit_raw(f"===== started {time.strftime('%Y-%m-%d %H:%M:%S')} "
                  f"pid {os.getpid()} =====")
    return _handle or None


def _emit_raw(line):
    text = f"{PREFIX} {line}"
    print(text, file=sys.stderr)
    if _handle:
        try:
            _handle.write(text + "\n")
        except OSError:
            pass


def _caller(skip=3):
    """Which function asked for this, as file:line and name.

    The immediate caller is usually a helper, so the two frames above it are
    recorded as well; that is what makes "who hid the group" answerable.

    Walks three frames with `sys._getframe` rather than building the whole
    stack with `traceback.extract_stack`. The setting is read on every card
    load, and reading a value should not cost a full stack walk each time.
    """
    parts = []
    try:
        frame = sys._getframe(skip)
    except (ValueError, AttributeError):
        return "?"
    for _ in range(3):
        if frame is None:
            break
        parts.append(f"{Path(frame.f_code.co_filename).name}:"
                     f"{frame.f_lineno} {frame.f_code.co_name}")
        frame = frame.f_back
    return " <- ".join(parts)


def note(event, **fields):
    """Write one line. `event` is a short name, `fields` become key=value."""
    if not _ENABLED:
        return
    _stream()
    stamp = time.strftime("%H:%M:%S") + f".{int(time.time() * 1000) % 1000:03d}"
    body = " ".join(f"{k}={v!r}" for k, v in fields.items())
    _emit_raw(f"{stamp} {event:<22s} {body}")


def note_with_caller(event, **fields):
    """As `note`, and also record which function asked."""
    if not _ENABLED:
        return
    fields = dict(fields)
    fields["asked_by"] = _caller(skip=3)
    note(event, **fields)


def describe_widget(card, widget, label):
    """Everything worth knowing about one control, in one string.

    Not just "is it hidden": a control can be shown, unhidden, and outside the
    part of the card the window is displaying, which is what happened in the
    previous round. Both questions are answered here.
    """
    if widget is None:
        return f"{label}=missing"
    try:
        shown = widget.isVisibleTo(card)
    except Exception:                                        # noqa: BLE001
        shown = "?"
    on_screen = "?"
    y = "?"
    try:
        from PyQt6.QtCore import QPoint
        from PyQt6.QtWidgets import QScrollArea
        areas = card.findChildren(QScrollArea)
        if areas:
            viewport = areas[0].viewport()
            y = widget.mapTo(viewport, QPoint(0, 0)).y()
            on_screen = (y + widget.height() > 0) and (y < viewport.height())
    except Exception:                                        # noqa: BLE001
        pass
    return f"{label}(shown={shown},on_screen={on_screen},y={y})"


def snapshot(card, when):
    """The whole group's state in one line, for `when`.

    Called at the click and again half a second later. If the two disagree,
    something changed the control after the click, and the log says what the
    difference is without anybody having to reproduce it.
    """
    if not _ENABLED or card is None:
        return
    try:
        switch = card.widgets.get(KEY)
        checked = switch.isChecked() if switch is not None else "?"
        configured = card.config_manager.get_value(KEY, None)
        parts = [describe_widget(card, getattr(card, "flux_group", None),
                                 "group"),
                 describe_widget(card, getattr(card, "flux_body", None),
                                 "options_block")]
        for name in ("flux_cubes.mode", "flux_cubes.spacing_m",
                     "flux_cubes.radius_m"):
            parts.append(describe_widget(card, card.widgets.get(name),
                                         name.split(".", 1)[1]))
        note(f"state-{when}", checkbox_checked=checked,
             setting_value=configured, detail=" ".join(parts))
    except Exception as exc:                                 # noqa: BLE001
        note(f"state-{when}", error=repr(exc))
