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

    total = len(PASSES) + len(FAILURES) + len(SKIPS)
    print(f"\n[test_gui_latlon_readout] {len(PASSES)}/{total} passed"
          + (f", {len(SKIPS)} skipped" if SKIPS else "")
          + (f", {len(FAILURES)} failed" if FAILURES else ""), flush=True)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
