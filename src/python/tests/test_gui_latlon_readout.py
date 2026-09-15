#!/usr/bin/env python3
"""The 3D viewer shows the latitude and longitude under the pointer -- and
shows NOTHING when the mesh cannot supply one.

⛔⛔ THE WHOLE RISK OF THIS FEATURE IS A PLAUSIBLE WRONG NUMBER. A coordinate
display is trusted by eye; there is no residual to inspect and no gate that
fires on a readout that is simply 40 m or 90 degrees off. Three separate ways to
produce one, all of which have existed in this file:

1. **A numeric fallback.** The previous status line read
   ``self.mesh['latitude'][cell_id] if 'latitude' in self.mesh.array_names
   else 0``. A mesh without the arrays reported ``(0.00, 0.00)`` -- a perfectly
   plausible coordinate, in the Gulf of Guinea.
2. **A frame that is not a body frame.** ``asin(z/r)`` returns a number for any
   mesh. On a local tangent-plane patch built about its own center it is not a
   latitude, and nothing about the number says so.
3. **The wrong facet.** ``vtkCellPicker.GetCellId`` is an index into whatever
   dataset the ray hit, and its tolerance decides how far along the ray it will
   accept a cell. MEASURED on the 2,397,890-facet reference run: with a cell
   locator attached and the tolerance of 0.005 the shipped picker used, the
   returned cell sits **39.8 m** -- about eleven facets -- from the point the ray
   actually hit. At tolerance 0 the same picks land **1.1 m** away.

The coordinates themselves are NOT computed here. They are the ones the engine
wrote into ``mesh.ply`` -- ``asin(z/r)`` and ``atan2(y, x)`` over the face
centroid, ``src/cpp/main/ptm_thermal_sim.cpp:13636`` -- i.e. planetocentric
latitude and east-positive longitude in the mesh's own body-fixed frame. Section
2 checks the number the user sees against a conversion written out longhand in
this file from the facet's own vertices.

⭐ AND IT HAS TO KEEP WORKING AFTER AN ANIMATION EXPORT. ``export_animation``
calls ``_reinit_plotter``, which replaces the plotter with one carrying a
DIFFERENT VTK interactor. The observers stayed bound to the retired one while
``picking_active``, ``picking_enabled`` and the "Select Face Mode" button all
still read as on, so face selection silently stopped working
(PTM-GUI-PICKING-DEAD-AFTER-EXPORT, found 2026-09-15, fixed here). A readout
bound the same way would inherit that defect on the first export.

Section 1 is ``ast`` over the panel source: no PyQt6, no VTK, no display, and it
cannot be skipped. Section 2 drives the real widget against real VTK, firing
real ``MouseMoveEvent``s through the real interactor.

⚠️ WHAT SECTION 2 SUBSTITUTES: ``pyvistaqt.QtInteractor`` SEGFAULTS under
``QT_QPA_PLATFORM=offscreen`` (rc=139, re-measured 2026-09-15), so the panel
module's ``QtInteractor`` symbol is replaced by a real
``pyvista.Plotter(off_screen=True)`` carrying a real ``QWidget`` as its
``.interactor``. Its ``iren.interactor`` is a real
``vtkGenericRenderWindowInteractor`` with the full observer API, so the binding,
the observer firing, the picker and the renderer are all genuine. NOT covered:
the on-screen appearance and position of the readout label, and a physical mouse.

⛔ Every fixture is synthetic. No site identifier and no real site coordinate
appears in this file, under the 2026-09-03 standing ruling -- which matters more
here than anywhere, because this file is about coordinates.

Run directly::

    python3 src/python/tests/test_gui_latlon_readout.py
"""

from __future__ import annotations

import ast
import math
import os
import re
import struct
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "python"))
sys.path.insert(0, str(REPO_ROOT / "gui"))

PANEL_SOURCE = REPO_ROOT / "gui" / "gui_panels" / "visualization_panel.py"

BODY_RADIUS_M = 1737400.0

#: Invented round numbers for the synthetic patch, south of the 70 deg polar
#: threshold and away from the pole. Not a site.
FIXTURE_LAT_DEG = -80.0
FIXTURE_LON_DEG = 20.0

#: What the readout line looks like, as a user reads it.
READOUT_RE = re.compile(
    r"lat\s+([0-9.]+)\s*°\s*([NS])\s+lon\s+([0-9.]+)\s*°\s*E")

_QAPP = None

#: What the environment said about the Qt platform BEFORE `_harness` forces
#: "offscreen" into it with `setdefault`. Section 3 needs a real window, so it
#: has to know whether the caller asked for a headless platform or whether
#: this file imposed one.
_PLATFORM_AS_INVOKED = os.environ.get("QT_QPA_PLATFORM")


class Skip(Exception):
    """A check that cannot run here, as distinct from one that failed."""


FAILURES: list[str] = []
SKIPS: list[str] = []
PASSES: list[str] = []


def check(name, fn, *args):
    try:
        detail = fn(*args)
    except Skip as exc:
        SKIPS.append(f"{name}: {exc}")
        print(f"SKIP {name} :: {exc}", flush=True)
    except AssertionError as exc:
        FAILURES.append(f"{name}: {exc}")
        print(f"FAIL {name} :: {exc}", flush=True)
    except Exception as exc:                                    # noqa: BLE001
        FAILURES.append(f"{name}: {type(exc).__name__}: {exc}")
        print(f"FAIL {name} :: {type(exc).__name__}: {exc}", flush=True)
    else:
        PASSES.append(name)
        print(f"PASS {name} :: {detail or ''}", flush=True)


# ---------------------------------------------------------------------------
# Section 1 -- the invariants, over the source.
# ---------------------------------------------------------------------------

def _panel_tree():
    return ast.parse(PANEL_SOURCE.read_text(encoding="utf-8"))


def _widget_class():
    for node in ast.walk(_panel_tree()):
        if isinstance(node, ast.ClassDef) and node.name == "PyVistaWidget":
            return node
    raise AssertionError("class PyVistaWidget is gone from the panel source")


def _methods(cls):
    return {n.name: n for n in cls.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def _source_of(node):
    return ast.get_source_segment(
        PANEL_SOURCE.read_text(encoding="utf-8"), node) or ""


def _assigns_self_attr(fn, attr):
    for node in ast.walk(fn):
        if not isinstance(node, ast.Assign):
            continue
        for tgt in node.targets:
            if (isinstance(tgt, ast.Attribute) and tgt.attr == attr
                    and isinstance(tgt.value, ast.Name)
                    and tgt.value.id == "self"):
                return True
    return False


def _calls_self_method(fn, name):
    for node in ast.walk(fn):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == name
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "self"):
            return True
    return False


def check_every_plotter_rebind_repoints_the_pointer_observers():
    """The invariant, not the patch.

    A new plotter is a new VTK interactor. Observers left on the old one still
    exist, still have ids and never fire again, and every flag the UI shows
    still reads "on". Anything that rebinds ``self.plotter`` must re-point them.
    """
    cls = _widget_class()
    offenders = []
    for name, fn in _methods(cls).items():
        if name in ("__init__", "init_ui"):
            continue
        if not _assigns_self_attr(fn, "plotter"):
            continue
        if not _calls_self_method(fn, "_bind_interactor_observers"):
            offenders.append(name)
    assert not offenders, (
        f"these PyVistaWidget methods rebind self.plotter without re-pointing "
        f"the pointer observers: {sorted(offenders)}. The coordinate readout "
        f"freezes and face selection stops working, with nothing in the UI "
        f"saying so (PTM-GUI-PICKING-DEAD-AFTER-EXPORT).")
    rebinders = sorted(n for n, f in _methods(cls).items()
                       if n not in ("__init__", "init_ui")
                       and _assigns_self_attr(f, "plotter"))
    assert rebinders, (
        "no method rebinds self.plotter any more -- this check is vacuous and "
        "must be re-aimed at whatever replaced _reinit_plotter")
    return f"{len(rebinders)} rebinder(s) checked: {rebinders}"


def check_a_rebind_removes_the_observers_it_replaces():
    """Otherwise every rebind stacks another copy on the interactor.

    The previous code added a fresh pair on every toggle of face-selection mode
    and removed none, so N toggles ran the pick N times per mouse move -- at
    537 ms each on a production mesh.
    """
    methods = _methods(_widget_class())
    fn = methods.get("_unbind_interactor_observers")
    assert fn is not None, (
        "PyVistaWidget._unbind_interactor_observers is gone, so nothing can "
        "remove an observer that was added to a retired interactor")
    assert "RemoveObserver" in _source_of(fn), (
        "_unbind_interactor_observers never calls RemoveObserver")
    binder = methods.get("_bind_interactor_observers")
    assert binder is not None, "PyVistaWidget._bind_interactor_observers is gone"
    assert _calls_self_method(binder, "_unbind_interactor_observers"), (
        "_bind_interactor_observers does not unbind first, so each call leaves "
        "another live observer behind")
    return "bind unbinds first, and unbind really calls RemoveObserver"


def check_no_coordinate_has_a_numeric_fallback():
    """A coordinate must never fall back to a number.

    Pinned as a property of the whole panel source, not of one method: the
    defect was an inline conditional expression whose ``else`` was ``0``, and
    the next one will be somewhere else in the file.
    """
    tree = _panel_tree()
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.IfExp):
            continue
        body = ast.dump(node.body)
        if "'latitude'" not in body and "'longitude'" not in body:
            continue
        if isinstance(node.orelse, ast.Constant) and isinstance(
                node.orelse.value, (int, float)):
            offenders.append(getattr(node, "lineno", "?"))
    assert not offenders, (
        f"a latitude or longitude falls back to a literal number at line(s) "
        f"{offenders}. (0.00, 0.00) is a plausible coordinate and there is "
        f"nothing on screen to distinguish it from a measured one.")
    src = PANEL_SOURCE.read_text(encoding="utf-8")
    assert "COORD_READOUT_NO_FRAME" in src, (
        "there is no text for 'this mesh has no georeferenced frame', so the "
        "refusal case has nowhere to go")
    return "no numeric fallback, and a refusal string exists"


def check_the_picker_keeps_its_tolerance_at_zero_and_its_locator():
    """Both halves were measured, and dropping either is a defect.

    Without the locator a single pick costs 537 ms on the 2,397,890-facet
    reference run. With the locator but the old 0.005 tolerance, the cell it
    returns is 39.8 m from the ray's hit point.
    """
    fn = _methods(_widget_class()).get("_build_hover_picker")
    assert fn is not None, "PyVistaWidget._build_hover_picker is gone"
    src = _source_of(fn)
    assert "vtkCellLocator" in src, (
        "the hover picker has no vtkCellLocator; a brute-force pick measured "
        "537 ms per mouse move on a production mesh")
    assert "AddLocator" in src, "the locator is built but never attached"
    assert re.search(r"SetTolerance\(\s*0(\.0+)?\s*\)", src), (
        "the hover picker's tolerance is not 0; at the 0.005 the previous "
        "picker used, the locator returns a cell about eleven facets from the "
        "point the ray hit")
    return "locator attached, tolerance 0"


def check_the_readout_is_not_drawn_into_the_scene():
    """A VTK text actor would appear in every exported animation frame.

    The 2026-09-15 pass established that the exported GIF is byte-identical
    across that change; a readout rendered into the scene would break that and
    put a coordinate on a deliverable without being asked.
    """
    methods = _methods(_widget_class())
    for name in ("face_coordinate_text", "_reset_coordinate_readout",
                 "on_mouse_move"):
        fn = methods.get(name)
        assert fn is not None, f"PyVistaWidget.{name} is gone"
        src = _source_of(fn)
        assert "add_text" not in src, (
            f"{name} draws the readout into the 3D scene with add_text, so it "
            f"would be burned into every exported animation frame")
    return "the readout is a Qt widget, not a scene actor"


def check_the_pick_corrects_the_event_position_for_the_canvas():
    """⛔⛔ EVERY pick must go through the canvas-scale correction.

    An interactor event position is NOT a VTK display position when the two
    are sized differently, which on macOS they always are: the Cocoa render
    window is sized in Qt's LOGICAL points while `pyvistaqt` multiplies the
    event position by the screen's device pixel ratio. Feeding one to
    `vtkCellPicker.Pick` misses by exactly that factor, which is how the
    readout and the Select Face Mode highlight both died on a Retina screen
    while fourteen offscreen checks passed.

    Both the hover readout and click selection have to be covered, so the
    correction belongs in the ONE place both reach.
    """
    methods = _methods(_widget_class())
    for name in ("_display_position", "_event_position_scale",
                 "_canvas_widget", "_pick_cell_at_display"):
        assert name in methods, f"PyVistaWidget.{name} is gone"
    pick = methods.get("_pick_face_at")
    assert pick is not None, "PyVistaWidget._pick_face_at is gone"
    assert _calls_self_method(pick, "_display_position"), (
        "_pick_face_at does not correct the event position for the canvas; a "
        "raw interactor position misses the mesh by the device pixel ratio on "
        "every HiDPI screen")
    move = methods.get("on_mouse_move")
    assert move is not None, "PyVistaWidget.on_mouse_move is gone"
    assert (_calls_self_method(move, "_display_position")
            or _calls_self_method(move, "_pick_face_at")), (
        "on_mouse_move picks without the canvas correction")
    click = methods.get("on_left_click")
    assert click is not None, "PyVistaWidget.on_left_click is gone"
    assert _calls_self_method(click, "_pick_face_at"), (
        "click selection does not go through the corrected pick, so it keeps "
        "the HiDPI offset the hover readout no longer has")
    return "hover and click both pick through _display_position"


def check_the_canvas_scale_is_measured_not_assumed():
    """The correction must come from the two sizes, not from a platform test.

    `sys.platform == "darwin"` would be right today and wrong the moment VTK's
    Cocoa backend starts reporting device pixels, or on any other backend that
    does. The ratio of the render window's size to the canvas's logical size
    times its device pixel ratio is 1 exactly when no correction is needed.
    """
    methods = _methods(_widget_class())
    fn = methods.get("_event_position_scale")
    assert fn is not None, "PyVistaWidget._event_position_scale is gone"
    assert "_event_pixel_ratio" in methods, (
        "PyVistaWidget._event_pixel_ratio is gone; the correction would then "
        "be taken from the SCREEN's ratio rather than the one the Qt adapter "
        "applied, and those differ between installed pyvistaqt copies")
    src = (_source_of(fn) + _source_of(methods["_canvas_widget"])
           + _source_of(methods["_event_pixel_ratio"]))
    for banned in ("sys.platform", "platform.system", "darwin", "Darwin"):
        assert banned not in src, (
            f"the canvas scale is decided by {banned!r} rather than measured "
            f"from the render window and the canvas")
    assert "GetSize" in src, "the render window's size is never read"
    assert "devicePixelRatioF" in src, "the device pixel ratio is never read"

    # And the arithmetic, on the three shapes that matter. A stand-in `self`
    # keeps this runnable with no display at all.
    import types

    class _FakeWin:
        def __init__(self, size):
            self._s = size

        def GetSize(self):
            return self._s

    class _FakeCanvas:
        """A canvas that can report the screen's ratio and the ratio the Qt
        adapter actually APPLIED as different numbers, because two installed
        copies of `pyvistaqt` disagree about that under PyQt6 and only one of
        them scales."""

        def __init__(self, win, w, h, ratio, applied=None):
            self._win, self._w, self._h, self._r = win, w, h, ratio
            if applied is not None:
                self._applied = applied
                self._getPixelRatio = lambda: self._applied

        def GetRenderWindow(self):
            return self._win

        def width(self):
            return self._w

        def height(self):
            return self._h

        def devicePixelRatioF(self):
            return self._r

    try:
        import gui_panels.visualization_panel as panel
    except Exception as exc:                                    # noqa: BLE001
        raise Skip(f"the visualization panel is not importable ({exc})")

    def scale_for(rw_size, logical, ratio, applied=None):
        win = _FakeWin(rw_size)
        canvas = _FakeCanvas(win, logical[0], logical[1], ratio, applied)
        fake = types.SimpleNamespace(
            plotter=types.SimpleNamespace(ren_win=win, interactor=canvas))
        # The three helpers are exercised, not stubbed: they are the halves
        # that decide WHICH widget and WHICH ratio the correction reads, and a
        # stub would make this arm agree with itself.
        fake._canvas_widget = lambda: panel.PyVistaWidget._canvas_widget(fake)
        fake._event_pixel_ratio = (
            lambda c: panel.PyVistaWidget._event_pixel_ratio(fake, c))
        assert fake._canvas_widget() is canvas, (
            "_canvas_widget did not identify the widget owning the plotter's "
            "render window")
        return panel.PyVistaWidget._event_position_scale(fake)

    # The MEASURED live case: a 900x619 logical canvas at device pixel ratio
    # 2.0 whose Cocoa render window reports 900x619, on the `pyvistaqt` this
    # GUI runs, whose `_getPixelRatio` returns 2.0 under PyQt6.
    cocoa = scale_for((900, 619), (900, 619), 2.0, applied=2.0)
    assert cocoa == (0.5, 0.5), (
        f"a Retina Cocoa canvas gives scale {cocoa}, not (0.5, 0.5)")
    # ⛔⛔ THE REGRESSION THIS ARM EXISTS FOR. Other installed copies of
    # `pyvistaqt` gate the scaling on `PyQtImpl in ["PyQt5", "PySide2",
    # "PySide6"]` and so apply NOTHING under PyQt6. There the event position
    # is already a display position, and a correction taken from the screen's
    # ratio would halve a position that was right -- breaking a working viewer
    # on exactly the screens this change is meant to fix.
    unscaled_adapter = scale_for((900, 619), (900, 619), 2.0, applied=1.0)
    assert unscaled_adapter == (1.0, 1.0), (
        f"a Retina canvas whose adapter does NOT scale gives "
        f"{unscaled_adapter}, not (1.0, 1.0); this halves a correct position "
        f"and breaks picking where it currently works")
    # A backend whose render window really is in device pixels.
    device_px = scale_for((1800, 1238), (900, 619), 2.0, applied=2.0)
    assert device_px == (1.0, 1.0), (
        f"a backend whose render window is already in device pixels gives "
        f"{device_px}, not (1.0, 1.0); that backend must not be touched")
    plain = scale_for((900, 619), (900, 619), 1.0, applied=1.0)
    assert plain == (1.0, 1.0), f"a non-HiDPI canvas gives {plain}"
    # No `_getPixelRatio` at all: fall back to the public ratio.
    no_private = scale_for((900, 619), (900, 619), 2.0)
    assert no_private == (0.5, 0.5), (
        f"with no `_getPixelRatio` to read, the public device pixel ratio "
        f"must be used; got {no_private}")
    absurd = scale_for((900, 619), (2, 2), 1.0, applied=1.0)
    assert absurd == (1.0, 1.0), (
        f"an incredible ratio gives {absurd}; it must fall back to 1 rather "
        f"than guess")
    return ("0.5 on a scaling Retina adapter, 1.0 on a non-scaling one, on "
            "device-pixel backends and on non-HiDPI")


def check_the_resting_readout_is_its_own_state():
    """"Off the mesh" must not be what a fresh mesh says.

    Paul's report was "it just says the pointer is off the mesh even when it
    is directly on the mesh". That sentence could not distinguish a pointer
    handler that never ran from one that ran and missed, because the window
    printed the same words for both. Whatever the cause, those are different
    states and have to read differently.
    """
    cls = _widget_class()
    names = set()
    for node in cls.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    names.add(tgt.id)
    for needed in ("COORD_READOUT_WAITING", "COORD_READOUT_OFF_MESH",
                   "COORD_READOUT_NO_FRAME", "COORD_READOUT_IDLE",
                   "COORD_READOUT_FAILED"):
        assert needed in names, f"PyVistaWidget.{needed} is gone"
    texts = {}
    for node in cls.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name.startswith("COORD_READOUT_"):
                try:
                    texts[name] = ast.literal_eval(node.value)
                except ValueError:
                    pass
    assert len(set(texts.values())) == len(texts), (
        f"two readout states share a string: {texts}")
    reset = _methods(cls).get("_reset_coordinate_readout")
    assert reset is not None, "PyVistaWidget._reset_coordinate_readout is gone"
    assert "seen_pointer" in {a.arg for a in reset.args.args}, (
        "_reset_coordinate_readout cannot tell a pointer that missed from a "
        "pointer that has never arrived, so it says the same thing for both")
    src = _source_of(reset)
    assert "COORD_READOUT_WAITING" in src, (
        "the resting text is still the off-the-mesh text")
    handler = _source_of(_methods(cls)["on_mouse_move"])
    assert "COORD_READOUT_FAILED" in handler, (
        "a pointer handler that raises leaves the previous text on screen, "
        "which reads as a pointer that wandered off the mesh")
    return f"{len(texts)} readout states, all distinct"


# ---------------------------------------------------------------------------
# Section 2 -- the behavior, against real VTK.
# ---------------------------------------------------------------------------

def _write_mesh_ply(path, verts, tris):
    """Write the PLY layout ``PyVistaWidget.read_binary_mesh`` requires.

    Per-face latitude and longitude are computed here the way the engine writes
    them (``asin(z/r)``, ``atan2(y, x)`` over the face centroid), because that
    is what a real ``mesh.ply`` carries.
    """
    import numpy as np

    cen = verts[tris].mean(axis=1)
    rad = np.linalg.norm(cen, axis=1)
    lat = np.degrees(np.arcsin(cen[:, 2] / np.maximum(rad, 1e-30)))
    lon = np.degrees(np.arctan2(cen[:, 1], cen[:, 0]))
    normals = np.zeros_like(verts)
    normals[:, 2] = 1.0
    with open(path, "wb") as f:
        f.write(b"ply\nformat binary_little_endian 1.0\n")
        f.write(f"element vertex {len(verts)}\n".encode())
        for prop in ("x", "y", "z", "nx", "ny", "nz"):
            f.write(f"property double {prop}\n".encode())
        f.write(f"element face {len(tris)}\n".encode())
        f.write(b"property list uchar int vertex_indices\n")
        f.write(b"property float latitude\nproperty float longitude\n")
        f.write(b"end_header\n")
        np.column_stack([verts, normals]).astype("<f8").tofile(f)
        for tri, la, lo in zip(tris, lat, lon):
            f.write(struct.pack("<B3iff", 3, int(tri[0]), int(tri[1]),
                                int(tri[2]), float(la), float(lo)))


def _grid(n, half):
    import numpy as np
    u = np.linspace(-half, half, n)
    X, Y = np.meshgrid(u, u)
    tris = []
    for j in range(n - 1):
        for i in range(n - 1):
            a = j * n + i
            tris.append((a, a + n, a + 1))
            tris.append((a + 1, a + n, a + n + 1))
    return X, Y, np.asarray(tris, dtype=np.int32)


def _write_run(root, body_centered=True, n=40, half=1500.0):
    """A results directory with one field and three timesteps.

    ``body_centered=False`` writes the SAME terrain about its own center -- a
    local tangent-plane patch, which is the mesh class that must get no
    coordinate at all.
    """
    import numpy as np

    root = Path(root)
    (root / "temperature").mkdir(parents=True, exist_ok=True)
    X, Y, tris = _grid(n, half)
    elev = 120.0 * np.sin(X / 700.0) * np.cos(Y / 900.0)

    if body_centered:
        lat0 = math.radians(FIXTURE_LAT_DEG)
        lon0 = math.radians(FIXTURE_LON_DEG)
        east = np.array([-math.sin(lon0), math.cos(lon0), 0.0])
        north = np.array([-math.sin(lat0) * math.cos(lon0),
                          -math.sin(lat0) * math.sin(lon0), math.cos(lat0)])
        up = np.array([math.cos(lat0) * math.cos(lon0),
                       math.cos(lat0) * math.sin(lon0), math.sin(lat0)])
        verts = (up * BODY_RADIUS_M
                 + X.ravel()[:, None] * east
                 + Y.ravel()[:, None] * north
                 + elev.ravel()[:, None] * up).astype(np.float64)
    else:
        verts = np.column_stack(
            [X.ravel(), Y.ravel(), elev.ravel()]).astype(np.float64)

    _write_mesh_ply(root / "mesh.ply", verts, tris)

    nf = len(tris)
    temps = (200.0 + 60.0 * np.linspace(0.0, 1.0, nf)).astype("<f4")
    for step in range(3):
        with open(root / "temperature" / f"temperature_{step:04d}.bin",
                  "wb") as f:
            f.write(struct.pack("<I", nf))
            f.write((temps + 3.0 * step).astype("<f4").tobytes())
    with open(root / "time_steps.txt", "w") as f:
        f.write("# step\ttime\tlabel\n")
        for step in range(3):
            f.write(f"{step}\t{step * 3600.0}\tstep {step}\n")
    return root


def _harness():
    """A real PyVistaWidget on a real offscreen plotter, or ``Skip``."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    # ⛔ This section needs the canvas, so it must NOT take the disable path.
    os.environ.pop("PTM_GUI_DISABLE_3D", None)
    try:
        import PyQt6
        from PyQt6.QtWidgets import QApplication, QWidget
    except Exception as exc:                                    # noqa: BLE001
        raise Skip(f"PyQt6 is not importable here ({exc})")
    # ⛔ Without these Qt does not raise when it cannot find its offscreen
    # platform plugin: it calls abort(), taking the buffered stdout with it, and
    # the whole file reports nothing at all.
    plug = Path(PyQt6.__file__).parent / "Qt6" / "plugins"
    os.environ.setdefault("QT_PLUGIN_PATH", str(plug))
    os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(plug / "platforms"))
    try:
        import pyvista as pv
    except Exception as exc:                                    # noqa: BLE001
        raise Skip(f"pyvista is not importable here ({exc})")
    global _QAPP
    if QApplication.instance() is None:
        try:
            _QAPP = QApplication([])
        except Exception as exc:                                # noqa: BLE001
            raise Skip(f"no Qt platform plugin could be initialized ({exc})")
    else:
        _QAPP = QApplication.instance()
    try:
        import gui_panels.visualization_panel as panel
    except Exception as exc:                                    # noqa: BLE001
        raise Skip(f"the visualization panel is not importable ({exc})")
    if not panel.PYVISTA_AVAILABLE:
        raise Skip("the panel reports PyVista unavailable (PTM_GUI_DISABLE_3D?)")

    class OffscreenInteractor(pv.Plotter):
        """Stand-in for QtInteractor, which segfaults offscreen. Its
        ``iren.interactor`` is a real vtkGenericRenderWindowInteractor."""

        def __init__(self, parent=None, **kwargs):
            super().__init__(off_screen=True, window_size=(480, 360))
            self.interactor = QWidget(parent)

    panel.QtInteractor = OffscreenInteractor
    return panel


def _loaded(panel, run_dir):
    widget = panel.PyVistaWidget()
    assert widget.load_results(str(run_dir)), f"{run_dir} did not load"
    widget.plotter.render()
    return widget


def _move_pointer(widget, x, y):
    """Fire a real MouseMoveEvent at a screen position, through the real
    interactor, so the shipped observer runs."""
    iren = widget.plotter.iren.interactor
    iren.SetEventPosition(int(x), int(y))
    iren.InvokeEvent("MouseMoveEvent")
    return widget.coord_readout_label.text()


def _center_of_canvas(widget):
    w, h = widget.plotter.window_size
    return w // 2, h // 2


def check_the_readout_names_the_facet_under_the_pointer(tmp):
    """The number on screen must equal a conversion computed here, longhand,
    from that facet's own vertices."""
    import numpy as np
    panel = _harness()
    widget = _loaded(panel, _write_run(Path(tmp) / "body", body_centered=True))
    x, y = _center_of_canvas(widget)
    text = _move_pointer(widget, x, y)

    match = READOUT_RE.search(text)
    assert match, f"the readout does not show a coordinate: {text!r}"
    shown_lat = float(match.group(1)) * (1 if match.group(2) == "N" else -1)
    shown_lon = float(match.group(3))

    face_id = widget.hover_face_id
    assert face_id is not None, "no facet was picked"
    verts = widget.mesh_data["vertices"][widget.mesh_data["faces"][face_id]]
    cx, cy, cz = verts.mean(axis=0)
    r = math.sqrt(cx * cx + cy * cy + cz * cz)
    hand_lat = math.degrees(math.asin(cz / r))
    hand_lon = math.degrees(math.atan2(cy, cx)) % 360.0

    assert abs(shown_lat - hand_lat) < 5e-4, (
        f"the readout shows latitude {shown_lat} for facet {face_id}; its own "
        f"vertices give {hand_lat}")
    assert abs(shown_lon - hand_lon) < 5e-4, (
        f"the readout shows longitude {shown_lon} for facet {face_id}; its own "
        f"vertices give {hand_lon}")
    assert "planetocentric" in text and "east-positive" in text, (
        f"the readout does not state the convention on screen: {text!r}")
    return (f"facet {face_id}: shown ({shown_lat:.5f}, {shown_lon:.5f}), "
            f"hand ({hand_lat:.5f}, {hand_lon:.5f})")


def check_the_picked_facet_is_the_one_under_the_ray(tmp):
    """The cell the picker returns must be the cell the ray hit, not one along
    the ray within a generous tolerance."""
    import numpy as np
    panel = _harness()
    widget = _loaded(panel, _write_run(Path(tmp) / "accuracy",
                                       body_centered=True, n=60))
    worst = 0.0
    w, h = widget.plotter.window_size
    sampled = 0
    for fx in (0.35, 0.45, 0.5, 0.55, 0.65):
        for fy in (0.4, 0.5, 0.6):
            _move_pointer(widget, w * fx, h * fy)
            face_id = widget.hover_face_id
            if face_id is None:
                continue
            hit = np.array(widget._hover_picker.GetPickPosition())
            verts = widget.mesh_data["vertices"][
                widget.mesh_data["faces"][face_id]]
            worst = max(worst, float(np.linalg.norm(hit - verts.mean(axis=0))))
            sampled += 1
    assert sampled >= 5, f"only {sampled} of 15 sample points hit the mesh"
    # One facet of this fixture is about 76 m across; two facet half-widths is
    # the honest bound on "the centroid of the facet the ray hit".
    edge = 2 * 1500.0 / 59
    assert worst < 1.5 * edge, (
        f"the picked facet's centroid is up to {worst:.1f} m from the point the "
        f"ray hit, on a mesh whose facets are {edge:.1f} m across")
    return f"{sampled} picks, worst centroid-to-hit {worst:.1f} m (facet {edge:.1f} m)"


def check_the_pointer_off_the_mesh_shows_no_coordinate(tmp):
    panel = _harness()
    widget = _loaded(panel, _write_run(Path(tmp) / "offmesh",
                                       body_centered=True))
    _move_pointer(widget, *_center_of_canvas(widget))
    assert READOUT_RE.search(widget.coord_readout_label.text()), (
        "the pointer never landed on the mesh, so this check is vacuous")
    text = _move_pointer(widget, 2, 2)                   # canvas corner
    assert not READOUT_RE.search(text), (
        f"the readout still shows a coordinate with the pointer off the mesh: "
        f"{text!r}")
    assert text == panel.PyVistaWidget.COORD_READOUT_OFF_MESH, (
        f"off-mesh text is {text!r}, expected the dedicated one")
    return f"off-mesh text {text!r}"


def check_a_mesh_with_no_body_frame_gets_no_coordinate(tmp):
    """A local tangent-plane patch must show the refusal, not a number.

    The SAME terrain as the other fixtures, written about its own center. Every
    facet still carries a latitude and longitude in the PLY -- the engine
    computes them unconditionally -- and every one of them is meaningless.
    """
    panel = _harness()
    widget = _loaded(panel, _write_run(Path(tmp) / "local",
                                       body_centered=False))
    assert widget._mesh_frame is not None, "no frame was decided at all"
    assert not widget._mesh_frame.usable, (
        f"a local tangent-plane patch was accepted as a body frame: "
        f"{widget._mesh_frame.reason}")
    text = _move_pointer(widget, *_center_of_canvas(widget))
    assert not READOUT_RE.search(text), (
        f"a coordinate was shown for a mesh with no body frame: {text!r}")
    assert text == panel.PyVistaWidget.COORD_READOUT_NO_FRAME, (
        f"expected the no-frame text, got {text!r}")
    return f"refused: {widget._mesh_frame.reason}"


def check_a_mesh_without_coordinate_arrays_refuses(tmp):
    """The defect this replaces returned (0.00, 0.00)."""
    panel = _harness()
    widget = _loaded(panel, _write_run(Path(tmp) / "noarrays",
                                       body_centered=True))
    widget.mesh.cell_data.remove("latitude")
    text = widget.face_coordinate_text(0)
    assert not READOUT_RE.search(text), (
        f"a coordinate was produced from a mesh with no latitude array: "
        f"{text!r}")
    assert "0.00" not in text, (
        f"the readout fell back to a number: {text!r}")
    return f"refused: {text!r}"


def check_the_readout_still_works_after_the_plotter_is_replaced(tmp):
    """``_reinit_plotter`` in isolation -- the mechanism, before the export."""
    panel = _harness()
    widget = _loaded(panel, _write_run(Path(tmp) / "reinit",
                                       body_centered=True))
    before = _move_pointer(widget, *_center_of_canvas(widget))
    assert READOUT_RE.search(before), f"no coordinate before the rebind: {before!r}"
    old_iren = widget.plotter.iren.interactor

    widget._reinit_plotter()
    widget.load_timestep_data(0)
    widget.plotter.render()
    new_iren = widget.plotter.iren.interactor
    assert new_iren is not old_iren, (
        "_reinit_plotter did not actually replace the interactor, so this "
        "check is vacuous")
    bound = {id(iren) for iren, _ in widget._interactor_observers}
    assert bound == {id(new_iren)}, (
        f"the pointer observers are bound to {bound}, not to the new "
        f"interactor {id(new_iren)}")

    after = _move_pointer(widget, *_center_of_canvas(widget))
    assert READOUT_RE.search(after), (
        f"the readout stopped updating after the plotter was replaced: "
        f"{after!r}")
    return "observers re-pointed, readout live"


def check_the_readout_survives_an_animation_export(tmp):
    """The reported trap, end to end, through the real export."""
    panel = _harness()
    widget = _loaded(panel, _write_run(Path(tmp) / "export",
                                       body_centered=True))
    before = _move_pointer(widget, *_center_of_canvas(widget))
    assert READOUT_RE.search(before), f"no coordinate before the export: {before!r}"
    old_iren = widget.plotter.iren.interactor

    gif = Path(tmp) / "export" / "anim.gif"
    try:
        ok = widget.export_animation(str(gif), format_type="gif", fps=4)
    except Exception as exc:                                    # noqa: BLE001
        raise Skip(f"the animation export could not run here ({exc})")
    assert ok, "export_animation reported failure"
    assert widget.plotter.iren.interactor is not old_iren, (
        "the export did not replace the plotter, so this check is vacuous")

    after = _move_pointer(widget, *_center_of_canvas(widget))
    assert READOUT_RE.search(after), (
        f"the coordinate readout is dead after an animation export: {after!r} "
        f"(PTM-GUI-PICKING-DEAD-AFTER-EXPORT)")
    widget.select_face_btn.setChecked(True)
    widget.toggle_face_selection(True)
    assert widget.picking_active, (
        "face selection reports inactive after an export")
    return "readout and face selection both live after the export"


def check_a_replaced_canvas_has_not_been_pointed_at(tmp):
    """A new canvas has seen no pointer, whatever the retired one saw.

    `_reinit_plotter` builds a DIFFERENT `QtInteractor`. If the widget still
    believes a pointer has arrived, a move over empty space on the new canvas
    changes neither the facet (already none) nor the first-event branch, and
    the readout sits on its resting text forever -- the two states running
    together again, which is the defect this change exists to separate.
    """
    panel = _harness()
    widget = _loaded(panel, _write_run(Path(tmp) / "rebuild_seen",
                                       body_centered=True))
    _move_pointer(widget, *_center_of_canvas(widget))
    assert widget._pointer_seen, "the pointer was not registered before"
    widget._reinit_plotter()
    assert not widget._pointer_seen, (
        "the widget still believes a pointer has arrived after the canvas was "
        "replaced, so the new one can never leave its resting text")
    assert (widget.coord_readout_label.text()
            == panel.PyVistaWidget.COORD_READOUT_WAITING), (
        f"the replaced canvas rests on "
        f"{widget.coord_readout_label.text()!r}")
    return "a replaced canvas starts unpointed-at"


def check_a_failed_pointer_handler_says_so_and_stops_saying_so(tmp):
    """Two halves of the same rule, and the second is the one that gets
    forgotten: a failure has to be VISIBLE, and a recovery has to CLEAR it.

    The failure text is latched behind a "report once" flag so a moving
    pointer cannot flood the console. Latched and never cleared, a transient
    failure leaves "the pointer handler failed" on screen over a readout that
    is working, and a later real failure prints nothing at all.
    """
    panel = _harness()
    widget = _loaded(panel, _write_run(Path(tmp) / "handler_fail",
                                       body_centered=True))
    good = _move_pointer(widget, *_center_of_canvas(widget))
    assert READOUT_RE.search(good), f"no coordinate to start from: {good!r}"

    real_pick = widget._pick_cell_at_display

    def boom(*_a, **_k):
        raise RuntimeError("synthetic pointer failure")

    widget._pick_cell_at_display = boom
    failed = _move_pointer(widget, 3, 3)
    assert failed == panel.PyVistaWidget.COORD_READOUT_FAILED, (
        f"a pointer handler that raised left {failed!r} on screen, which "
        f"reads as a pointer that wandered off the mesh")

    widget._pick_cell_at_display = real_pick
    recovered = _move_pointer(widget, *_center_of_canvas(widget))
    assert READOUT_RE.search(recovered), (
        f"the readout still says {recovered!r} after the handler recovered; "
        f"the failure text is latched")
    assert not widget._hover_error_reported, (
        "the report-once latch survived the recovery, so the NEXT real "
        "failure would print nothing")
    return "failure is shown, recovery clears it, and the latch resets"


def check_the_hover_highlight_is_the_facet_under_the_pointer(tmp):
    """Selection mode still highlights the picked facet, and does it without
    walking the whole mesh.

    ``extract_cells`` measured 10.3 ms on the 2,397,890-facet reference run
    against 2.2 ms for the one-triangle build, on a path that runs for every
    pointer move.
    """
    import numpy as np
    panel = _harness()
    widget = _loaded(panel, _write_run(Path(tmp) / "highlight",
                                       body_centered=True))
    widget.toggle_face_selection(True)
    assert widget.picking_active, "selection mode did not arm"
    _move_pointer(widget, *_center_of_canvas(widget))
    face_id = widget.hover_face_id
    assert face_id is not None, "no facet was picked"
    assert widget.hover_highlight_actor is not None, (
        "selection mode picked a facet but drew no highlight")
    cell = widget._single_face_polydata(face_id)
    assert cell is not None and cell.n_cells == 1, (
        f"the highlight mesh has {None if cell is None else cell.n_cells} "
        f"cells, expected exactly 1")
    expected = widget.mesh_data["vertices"][widget.mesh_data["faces"][face_id]]
    assert np.allclose(np.asarray(cell.points), expected), (
        "the highlight triangle is not the picked facet's own three vertices")
    fn = _methods(_widget_class())["update_hover_highlight"]
    # A CALL, over the AST -- not a substring. The method's own comment names
    # `extract_cells` to say why it is not used, and a text search cannot tell
    # the two apart.
    called = [n.func.attr for n in ast.walk(fn)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    assert "extract_cells" not in called, (
        "update_hover_highlight calls extract_cells, which walks the whole "
        "mesh; that is an O(N) call on a per-pointer-move path")
    return f"facet {face_id} highlighted from its own 3 vertices"


def check_toggling_selection_mode_does_not_stack_observers(tmp):
    panel = _harness()
    widget = _loaded(panel, _write_run(Path(tmp) / "toggle",
                                       body_centered=True))
    for _ in range(5):
        widget.toggle_face_selection(True)
        widget.toggle_face_selection(False)
    assert len(widget._interactor_observers) == 2, (
        f"{len(widget._interactor_observers)} observers are bound after five "
        f"toggles; each one runs the pick again on every mouse move")
    return "2 observers after five toggles"


# ---------------------------------------------------------------------------
# Section 3 -- the REAL canvas.
#
# ⛔⛔⛔ THIS SECTION EXISTS BECAUSE SECTION 2 PASSED 14/14 WHILE THE FEATURE
# DID NOT WORK.
#
# Section 2 drives `iren.SetEventPosition(x, y)` directly. That writes a VTK
# DISPLAY position straight into the interactor and skips the one step the
# running program actually performs: `pyvistaqt`'s translation of a Qt mouse
# position into a VTK one. On macOS those differ by the device pixel ratio --
# VTK's Cocoa render window is sized in Qt's LOGICAL points while
# `QVTKRenderWindowInteractor._setEventInformation` multiplies the position by
# the screen's ratio -- so on a Retina display every pick landed at twice the
# intended place and returned nothing. An offscreen plotter has a ratio of 1,
# which is why Section 2 could never see it. A harness that supplies the
# quantity under test is not a test of it.
#
# So this section builds a REAL `pyvistaqt.QtInteractor` in a REAL window and
# sends REAL `QMouseEvent`s through `QApplication.sendEvent`, which is the path
# a physical mouse takes. ⭐ `QtInteractor` does NOT segfault when a genuine
# window system is available -- the rc=139 recorded on 2026-09-15 was under
# `QT_QPA_PLATFORM=offscreen`, not a property of the class. It is still fragile
# enough to be worth isolating, so the whole section runs in a SUBPROCESS: a
# crash there is a reported failure with its output, not an `abort()` that
# takes this file's buffered stdout with it.
#
# Where there is no window system -- CI, ssh, `QT_QPA_PLATFORM=offscreen` --
# the worker says so and the section SKIPS out loud. It does not pretend.
# ---------------------------------------------------------------------------

#: The worker prints exactly one of these as its last line.
_WORKER_OK = "REAL-CANVAS-OK"
_WORKER_SKIP = "REAL-CANVAS-SKIP"


def _real_canvas_worker():
    """Run the real-window arms. Called in a child process only.

    Prints one `PASS`/`FAIL` line per arm and then `_WORKER_OK` or
    `_WORKER_SKIP`; exits 1 if any arm failed.
    """
    import numpy as np

    failures = []

    def arm(name, fn):
        try:
            detail = fn()
        except Skip as exc:
            print(f"SKIP {name} :: {exc}", flush=True)
        except Exception as exc:                                # noqa: BLE001
            failures.append(name)
            print(f"FAIL {name} :: {type(exc).__name__}: {exc}", flush=True)
        else:
            print(f"PASS {name} :: {detail or ''}", flush=True)

    plat = os.environ.get("QT_QPA_PLATFORM", "")
    if plat in ("offscreen", "minimal", "vnc"):
        print(f"{_WORKER_SKIP} QT_QPA_PLATFORM={plat!r} cannot show a real "
              f"window", flush=True)
        return 0
    try:
        import PyQt6
        from PyQt6.QtWidgets import QApplication, QMainWindow
        from PyQt6.QtCore import QPoint, Qt, QEvent
        from PyQt6.QtGui import QMouseEvent
    except Exception as exc:                                    # noqa: BLE001
        print(f"{_WORKER_SKIP} PyQt6 is not importable ({exc})", flush=True)
        return 0
    plug = Path(PyQt6.__file__).parent / "Qt6" / "plugins"
    os.environ.setdefault("QT_PLUGIN_PATH", str(plug))
    os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(plug / "platforms"))
    os.environ.pop("PTM_GUI_DISABLE_3D", None)
    try:
        app = QApplication.instance() or QApplication([])
    except Exception as exc:                                    # noqa: BLE001
        print(f"{_WORKER_SKIP} no Qt platform plugin ({exc})", flush=True)
        return 0
    if not app.screens():
        print(f"{_WORKER_SKIP} Qt reports no screens", flush=True)
        return 0
    try:
        import gui_panels.visualization_panel as panel
    except Exception as exc:                                    # noqa: BLE001
        print(f"{_WORKER_SKIP} the panel is not importable ({exc})", flush=True)
        return 0
    if not panel.PYVISTA_AVAILABLE:
        print(f"{_WORKER_SKIP} the panel reports PyVista unavailable",
              flush=True)
        return 0
    if not hasattr(panel.QtInteractor, "GetRenderWindow"):
        print(f"{_WORKER_SKIP} the panel's QtInteractor is a stand-in",
              flush=True)
        return 0

    tmp = Path(tempfile.mkdtemp(prefix="ptm_latlon_realcanvas_"))
    run = _write_run(tmp / "run")

    win = QMainWindow()
    widget = panel.PyVistaWidget()
    win.setCentralWidget(widget)
    win.resize(900, 700)
    win.show()
    app.processEvents()
    if not widget.load_results(str(run)):
        print(f"{_WORKER_SKIP} the fixture did not load", flush=True)
        return 0
    app.processEvents()
    widget.plotter.render()
    app.processEvents()

    # ⛔ The canvas is found HERE, with no help from the panel. An arm that
    # calls a helper the fix introduced can only pass on a tree that has the
    # fix, which makes it a tautology rather than a test of the behavior --
    # the exact mistake recorded against this feature's COG sibling.
    canvas = None
    for cand in (widget.plotter, getattr(widget.plotter, "interactor", None)):
        if cand is not None and callable(getattr(cand, "GetRenderWindow", None)):
            canvas = cand
            break
    if canvas is None:
        print("FAIL real_canvas_is_identified :: no widget owns the plotter's "
              "render window", flush=True)
        return 1
    ratio = canvas.devicePixelRatioF()
    rw = widget.plotter.ren_win.GetSize()
    print(f"canvas {canvas.width()}x{canvas.height()} ratio {ratio} "
          f"render window {rw[0]}x{rw[1]}", flush=True)

    def send_move(lx, ly):
        QApplication.sendEvent(canvas, QMouseEvent(
            QEvent.Type.MouseMove, QPoint(int(lx), int(ly)).toPointF(),
            Qt.MouseButton.NoButton, Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier))
        app.processEvents()

    cx, cy = canvas.width() // 2, canvas.height() // 2

    def the_readout_names_the_facet_under_a_real_mouse():
        """⭐ THE DISCRIMINATING ARM. A real Qt mouse move at the middle of a
        real canvas must produce the coordinate of the facet in the middle of
        the mesh -- the fixture is centered on the canvas, so that is a number
        this file knows without asking the code under test."""
        send_move(cx, cy)
        text = widget.coord_readout_label.text()
        m = READOUT_RE.search(text)
        assert m, (f"the readout says {text!r} after a real mouse move at the "
                   f"center of a real canvas; expected a coordinate")
        lat = float(m.group(1)) * (-1.0 if m.group(2) == "S" else 1.0)
        lon = float(m.group(3))
        assert abs(lat - FIXTURE_LAT_DEG) < 0.02, (
            f"latitude {lat} is not the center of the patch "
            f"({FIXTURE_LAT_DEG})")
        assert abs(((lon - FIXTURE_LON_DEG + 180.0) % 360.0) - 180.0) < 0.02, (
            f"longitude {lon} is not the center of the patch "
            f"({FIXTURE_LON_DEG})")
        return f"{text}"

    def select_face_mode_highlights_under_a_real_mouse():
        """Paul's own symptom, verbatim: "Hover does not work in select face
        mode". The highlight never touches latitude or longitude, so this is a
        second, independent witness that the PICK is what failed."""
        widget.select_face_btn.click()
        app.processEvents()
        assert widget.picking_active, "Select Face Mode did not arm"
        widget.hover_face_id = None
        send_move(cx, cy)
        assert widget.hover_face_id is not None, (
            "no facet is under the pointer at the center of the canvas")
        assert widget.hover_highlight_actor is not None, (
            f"facet {widget.hover_face_id} is under the pointer and the "
            f"hover highlight was not drawn")
        assert widget.pick_status_label.text().startswith("Face "), (
            f"the status line says {widget.pick_status_label.text()!r}")
        widget.select_face_btn.click()
        app.processEvents()
        return f"facet {widget.hover_face_id} highlighted"

    def the_event_position_is_not_a_vtk_display_position():
        """The measurement that names the defect, made in THIS file.

        On a canvas whose device pixel ratio is not 1, the position VTK is
        handed after a Qt mouse move and the position VTK's display space
        actually uses are different numbers. Both picks are done here with the
        panel's own picker and renderer -- shipped on every tree, before and
        after the fix -- so this arm measures the same thing either way and
        fails on the tree that does not correct for it.
        """
        if abs(ratio - 1.0) < 1e-9:
            raise Skip("this screen's device pixel ratio is 1, so the two "
                       "positions are the same number; only a HiDPI screen "
                       "can exercise this")
        send_move(cx, cy)
        raw = tuple(widget.plotter.iren.interactor.GetEventPosition())
        # Derived here from the two sizes, not from the code under test.
        sx = rw[0] / (canvas.width() * ratio)
        sy = rw[1] / (canvas.height() * ratio)
        corrected = (int(round(raw[0] * sx)), int(round(raw[1] * sy)))
        assert raw != corrected, (
            f"the event position {raw} and the display position {corrected} "
            f"coincide on a ratio-{ratio} canvas, so this arm proves nothing")

        def pick(px, py):
            widget._hover_picker.Pick(px, py, 0, widget.plotter.renderer)
            cid = widget._hover_picker.GetCellId()
            return None if cid is None or cid < 0 else int(cid)

        assert pick(*raw) is None, (
            f"picking the raw event position {raw} returned a facet, so this "
            f"arm cannot tell a corrected pick from an uncorrected one")
        hit = pick(*corrected)
        assert hit is not None, (
            f"picking the display position {corrected} returned nothing")
        assert widget.hover_face_id == hit, (
            f"the panel settled on facet {widget.hover_face_id} while the "
            f"display position {corrected} holds facet {hit}; the panel is "
            f"picking somewhere other than where the pointer is")
        return (f"raw {raw} picks nothing, display {corrected} picks {hit}, "
                f"and the panel agrees")

    def the_resting_readout_does_not_claim_the_pointer_is_off_the_mesh():
        """Before any pointer event there is nothing to be off OF. The two
        states shared one string, which is how a dead handler read as a
        wandering pointer."""
        w2 = panel.PyVistaWidget()
        win2 = QMainWindow()
        win2.setCentralWidget(w2)
        win2.resize(700, 500)
        win2.show()
        app.processEvents()
        assert w2.load_results(str(run)), "the fixture did not load"
        app.processEvents()
        resting = w2.coord_readout_label.text()
        w2.close()
        win2.close()
        assert resting != panel.PyVistaWidget.COORD_READOUT_OFF_MESH, (
            f"a freshly loaded mesh already says {resting!r}, so a pointer "
            f"handler that never runs is indistinguishable from one that runs "
            f"and misses -- which is exactly how this defect was reported")
        return resting

    arm("test_real_canvas_readout_names_the_facet_under_a_real_mouse",
        the_readout_names_the_facet_under_a_real_mouse)
    arm("test_real_canvas_select_face_mode_highlights_under_a_real_mouse",
        select_face_mode_highlights_under_a_real_mouse)
    arm("test_real_canvas_event_position_is_not_a_display_position",
        the_event_position_is_not_a_vtk_display_position)
    arm("test_real_canvas_resting_readout_is_not_the_off_mesh_text",
        the_resting_readout_does_not_claim_the_pointer_is_off_the_mesh)

    widget.close()
    win.close()
    print(_WORKER_OK if not failures else "REAL-CANVAS-FAILED", flush=True)
    return 1 if failures else 0


def check_the_real_canvas_arms(_tmp=None):
    """Run Section 3 in a child process and report what it found."""
    import subprocess

    env = dict(os.environ)
    if _PLATFORM_AS_INVOKED is None:
        env.pop("QT_QPA_PLATFORM", None)
    else:
        env["QT_QPA_PLATFORM"] = _PLATFORM_AS_INVOKED
    env["PTM_LATLON_REAL_CANVAS_WORKER"] = "1"
    env.pop("PTM_GUI_DISABLE_3D", None)
    proc = subprocess.run([sys.executable, str(Path(__file__).resolve())],
                          env=env, capture_output=True, text=True, timeout=600)
    out = (proc.stdout or "") + (proc.stderr or "")
    tail = "\n".join(l for l in out.splitlines()
                      if l.startswith(("PASS ", "FAIL ", "SKIP ", "canvas ",
                                       "REAL-CANVAS")))
    if _WORKER_SKIP in out:
        raise Skip(next(l for l in out.splitlines()
                        if l.startswith(_WORKER_SKIP))[len(_WORKER_SKIP):]
                   .strip() or "no real window is available here")
    if _WORKER_OK not in out:
        raise AssertionError(
            f"the real-canvas worker exited {proc.returncode}:\n"
            f"{tail or out[-2000:]}")
    n_pass = sum(1 for l in out.splitlines() if l.startswith("PASS "))
    n_skip = sum(1 for l in out.splitlines() if l.startswith("SKIP "))
    verdict = f"{n_pass} real-canvas arms passed, {n_skip} skipped :: {tail}"
    if proc.returncode != 0:
        # ⛔ Every arm reported its own result BEFORE the banner, so a nonzero
        # exit after it is a crash in TEARDOWN -- a live VTK render window
        # being destroyed, which this project has on record crashing -- and
        # not a statement about the feature. Saying "the readout is broken"
        # because the window died on the way out would be a false alarm; so
        # would swallowing it. It is named, loudly, in the verdict.
        verdict += (f" [⚠️ the worker crashed with rc={proc.returncode} AFTER "
                    f"every arm reported; that is a teardown crash of the VTK "
                    f"window, not a readout failure, and it is worth a look]")
    return verdict


# ---------------------------------------------------------------------------

def main():
    tmp = tempfile.mkdtemp(prefix="ptm_latlon_gate_")
    check("test_every_plotter_rebind_repoints_the_pointer_observers",
          check_every_plotter_rebind_repoints_the_pointer_observers)
    check("test_a_rebind_removes_the_observers_it_replaces",
          check_a_rebind_removes_the_observers_it_replaces)
    check("test_no_coordinate_has_a_numeric_fallback",
          check_no_coordinate_has_a_numeric_fallback)
    check("test_the_picker_keeps_its_tolerance_at_zero_and_its_locator",
          check_the_picker_keeps_its_tolerance_at_zero_and_its_locator)
    check("test_the_readout_is_not_drawn_into_the_scene",
          check_the_readout_is_not_drawn_into_the_scene)
    check("test_the_readout_names_the_facet_under_the_pointer",
          check_the_readout_names_the_facet_under_the_pointer, tmp)
    check("test_the_picked_facet_is_the_one_under_the_ray",
          check_the_picked_facet_is_the_one_under_the_ray, tmp)
    check("test_the_pointer_off_the_mesh_shows_no_coordinate",
          check_the_pointer_off_the_mesh_shows_no_coordinate, tmp)
    check("test_a_mesh_with_no_body_frame_gets_no_coordinate",
          check_a_mesh_with_no_body_frame_gets_no_coordinate, tmp)
    check("test_a_mesh_without_coordinate_arrays_refuses",
          check_a_mesh_without_coordinate_arrays_refuses, tmp)
    check("test_the_readout_still_works_after_the_plotter_is_replaced",
          check_the_readout_still_works_after_the_plotter_is_replaced, tmp)
    check("test_the_readout_survives_an_animation_export",
          check_the_readout_survives_an_animation_export, tmp)
    check("test_the_hover_highlight_is_the_facet_under_the_pointer",
          check_the_hover_highlight_is_the_facet_under_the_pointer, tmp)
    check("test_toggling_selection_mode_does_not_stack_observers",
          check_toggling_selection_mode_does_not_stack_observers, tmp)
    check("test_a_replaced_canvas_has_not_been_pointed_at",
          check_a_replaced_canvas_has_not_been_pointed_at, tmp)
    check("test_a_failed_pointer_handler_says_so_and_stops_saying_so",
          check_a_failed_pointer_handler_says_so_and_stops_saying_so, tmp)
    check("test_the_pick_corrects_the_event_position_for_the_canvas",
          check_the_pick_corrects_the_event_position_for_the_canvas)
    check("test_the_canvas_scale_is_measured_not_assumed",
          check_the_canvas_scale_is_measured_not_assumed)
    check("test_the_resting_readout_is_its_own_state",
          check_the_resting_readout_is_its_own_state)
    check("test_the_real_canvas_arms", check_the_real_canvas_arms)

    total = len(PASSES) + len(FAILURES) + len(SKIPS)
    print(f"\n[test_gui_latlon_readout] {len(PASSES)}/{total} passed"
          + (f", {len(SKIPS)} skipped" if SKIPS else "")
          + (f", {len(FAILURES)} failed" if FAILURES else ""), flush=True)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    if os.environ.get("PTM_LATLON_REAL_CANVAS_WORKER") == "1":
        sys.exit(_real_canvas_worker())
    sys.exit(main())
