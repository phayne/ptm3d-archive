"""A record of what the 3D canvas does when the pointer moves over it.

Why this exists
---------------
The latitude/longitude readout and the "Select Face Mode" highlight both failed
on one person's screen while 14 of 14 automated checks passed on another. The
checks drove a ``pyvista.Plotter(off_screen=True)``, because
``pyvistaqt.QtInteractor`` was believed to segfault offscreen; the running
program is a ``QtInteractor`` in a real window on a Retina display. The two
disagree, and the only instrument that can see the difference is the machine
with the screen.

So this writes down what the real canvas actually does, per pointer move, in
terms that settle the question without anyone having to reproduce anything:

* which interactor object the observers are attached to, and their ids;
* the render window's size in VTK display pixels against Qt's logical size and
  device pixel ratio -- the pair whose disagreement was the defect;
* the raw event position VTK was handed, and the display position the pick
  actually used after correction;
* the cell the picker returns for BOTH, so one line shows whether the
  correction is what made the difference;
* which dataset the ray hit and how many cells it has;
* whether the mesh carries the per-face coordinate arrays, and what
  ``describe_mesh_frame`` decided about its frame;
* the exact text put on the readout.

The switch
----------
The SAME switch as :mod:`gui.gui_core.flux_trace`, and deliberately not a new
one: ``PTM_GUI_TRACE=1``. The decision is imported from that module so there is
one spelling, one warning about an unrecognized value, and one thing to
remember::

    PTM_GUI_TRACE=1 python3 gui/run_gui.py

Off means OFF. With the variable unset nothing is opened, nothing is written,
and the per-move path costs one ``if``.

Where it goes
-------------
Two places, so it is hard to lose:

* ``~/ptm3d_pointer_trace.log`` -- appended to, one line per event;
* standard error, which is the terminal the program was started from.

A separate file from the flux-cube record because they are separate
investigations and each is read by searching for its own prefix; mixing them
makes both harder to read.

⚠️ The line-writing machinery below is deliberately a near-copy of
:mod:`gui.gui_core.flux_trace`'s, and that duplication is known. Only the
SWITCH is shared, which is the part that has to have one answer. Folding both
onto a common writer means editing `flux_trace`, whose own gate is currently
red for unrelated reasons, and this branch is a fix for a pick defect; it is
the right cleanup and it belongs in its own change.
"""

from __future__ import annotations

import os
import sys
import time

from pathlib import Path

try:                                    # the GUI package, normal case
    from . import flux_trace
except ImportError:                     # pragma: no cover - direct import
    import flux_trace                                           # type: ignore

#: The one string to search the log for.
PREFIX = "[POINTER-TRACE]"

#: Where the record is written. Fixed rather than derived so it can be named in
#: an instruction without the reader having to work anything out.
LOG_PATH = Path.home() / "ptm3d_pointer_trace.log"

#: Pointer moves arrive in the thousands per second. Every one of the first
#: `_VERBATIM` is written; after that one in `_THEREAFTER`. A move that CHANGES
#: the answer -- onto a facet, off the mesh, onto a different facet -- is always
#: written, whatever the count, because those are the transitions that carry the
#: information and throttling them away would defeat the record.
_VERBATIM = 40
_THEREAFTER = 50

_handle = None
_started = False
_moves = 0


def enabled():
    """True when ``PTM_GUI_TRACE`` is on. Decided by :mod:`flux_trace`."""
    return flux_trace.enabled()


def _stream():
    """The log file, opened once and kept open. Never raises.

    A trace that stops the program it is tracing would be worse than no trace,
    so every failure here falls back to standard error alone.
    """
    global _handle, _started
    if not enabled():
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
        _emit_raw("===== send this whole file back; it is the measurement =====")
    return _handle or None


def _emit_raw(line):
    text = f"{PREFIX} {line}"
    print(text, file=sys.stderr)
    if _handle:
        try:
            _handle.write(text + "\n")
        except OSError:
            pass


def note(event, **fields):
    """Write one line. `event` is a short name, `fields` become key=value."""
    if not enabled():
        return
    _stream()
    stamp = time.strftime("%H:%M:%S") + f".{int(time.time() * 1000) % 1000:03d}"
    body = " ".join(f"{k}={v!r}" for k, v in fields.items())
    _emit_raw(f"{stamp} {event:<18s} {body}")


def should_record_move(changed):
    """Whether THIS pointer move goes in the record.

    Every transition, every one of the first `_VERBATIM` moves, then one in
    `_THEREAFTER`.

    Only reached when the trace is ON: the caller short-circuits on
    `enabled()` first, so with the trace off nothing is counted and the whole
    pointer path costs one comparison per move. The count is therefore a count
    of TRACED moves, which is what the thinning is about.
    """
    global _moves
    _moves += 1
    if changed:
        return True
    if _moves <= _VERBATIM:
        return True
    return _moves % _THEREAFTER == 0
