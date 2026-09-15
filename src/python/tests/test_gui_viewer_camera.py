#!/usr/bin/env python3
"""The 3D viewer's camera must survive an animation export, and "Reset View"
must mean the view the results opened with.

⛔ THE TWO DEFECTS THIS EXISTS TO STOP, BOTH REPORTED FROM REAL USE:

1. **"Reset view doesn't work properly -- it should reset to the same view as
   when opening the results."** ``PyVistaWidget.reset_camera`` called
   ``plotter.reset_camera()``, which is a DIFFERENT operation: it refits the
   camera DISTANCE to the bounds of the visible actors and keeps whatever
   direction, roll and view-up the user last orbited to. Measured on a
   2,397,890-facet lunar polar run, after a 35 deg azimuth / 12 deg elevation
   orbit it returned a camera 10.0 km from the opening camera, still tilted,
   covering 25.4% of the canvas against the opening view's 52.7%.

2. **"After outputting an animation, the visualization goes dark, i.e. no mesh
   or results visible."** NOT missing actors and NOT missing scalars -- both
   were measured intact after the export (3 actors, ``main_mesh`` visible,
   mapper array ``temperature``, 2,397,890 cells). It is the CAMERA.
   ``export_animation`` ends by calling ``_reinit_plotter``, which replaces
   ``self.plotter`` with a brand new one whose renderer is empty and whose
   camera is PyVista's default -- position (0, 0, 1), focal point (0, 0, 0).
   ``update_display`` then decided whether to preserve the current camera from
   ``self._has_loaded_data``, a flag on the WIDGET rather than on the plotter,
   so it read that default camera as "the user's view" and re-imposed it over
   the framing ``add_mesh`` had just computed. The mesh ended up 1,734 km away
   inside a 1,239 km-wide field of view: 1.8% of the canvas lit, against 52.7%
   before the export. That is the "dark".

⭐ THEY ARE TWO DEFECTS, NOT ONE -- but the first is why the second could not be
walked back. After the export, pressing Reset View put the camera on the
OPPOSITE side of a south-polar heightfield (z = -1,717 km, above the surface,
view-up (0, 1, 0)) from the opening view (z = -1,746 km, below it, view-up
(1, 0, 0)), so the user had no way back to the view they started from. Fixing
either one alone leaves the report half-standing.

Section 1 is ``ast`` over the panel source, so it needs neither PyQt6 nor VTK
nor a display and CANNOT be skipped. It pins the INVARIANT rather than the
patch: any method that rebinds ``self.plotter`` is handing ``update_display``
a plotter that has never framed anything, so it must clear the flag that says
otherwise. That is the whole of defect 2 stated as a rule.

Section 2 drives the real widget against real VTK and asserts on the RENDERED
CANVAS -- the md5 of the offscreen pixels -- because "the same view" is a claim
about what is on screen, not about a dict this module could write and then read
back. It skips loudly, naming what is missing, when pyvista or PyQt6 are not
installed.

⚠️ WHAT SECTION 2 SUBSTITUTES, stated so no one reads it as broader than it is:
``pyvistaqt.QtInteractor`` SEGFAULTS under ``QT_QPA_PLATFORM=offscreen`` on the
development machine (rc=139, no traceback), so the panel module's
``QtInteractor`` symbol is replaced by a real ``pyvista.Plotter(off_screen=True)``
carrying a real ``QWidget`` as its ``.interactor``. Both are
``pyvista.plotting.plotter.BasePlotter`` subclasses and share the camera and
actor semantics under test. The widget, its layout, ``update_display``,
``_reinit_plotter``, ``export_animation`` and ``reset_camera`` are the shipped
code, unmodified. What is therefore NOT covered: anything that needs a live GL
context -- whether the retired canvas widget is really invisible on screen,
whether Qt's stacking order puts the new canvas on top, and the interactive
orbit itself, which here is driven through the plotter instead of the mouse.

Run directly::

    python3 src/python/tests/test_gui_viewer_camera.py
"""

from __future__ import annotations

import ast
import hashlib
import os
import struct
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "python"))
sys.path.insert(0, str(REPO_ROOT / "gui"))

PANEL_SOURCE = REPO_ROOT / "gui" / "gui_panels" / "visualization_panel.py"

#: The widget attribute that tells ``update_display`` a camera worth keeping is
#: already on screen.
LOADED_FLAG = "_has_loaded_data"
#: The widget attribute holding the view the results opened with.
HOME_STATE = "_home_camera_state"


#: The one QApplication, held so it is not garbage collected mid-run.
_QAPP = None


class Skip(Exception):
    """A check that cannot run here, as distinct from one that failed."""


FAILURES: list[str] = []
SKIPS: list[str] = []
PASSES: list[str] = []


def check(name, fn):
    try:
        detail = fn()
    except Skip as exc:
        SKIPS.append(f"{name}: {exc}")
        print(f"SKIP {name} :: {exc}", flush=True)
    except AssertionError as exc:
        FAILURES.append(f"{name}: {exc}")
        print(f"FAIL {name} :: {exc}", flush=True)
    except Exception as exc:                                    # noqa: BLE001
        # Anything else -- a pyvista API change, imageio missing its ffmpeg
        # plugin, a full disk. Recorded as a failure rather than allowed out of
        # `main`, where it would take the remaining checks and the summary line
        # with it and leave a partial transcript that reads like a pass.
        FAILURES.append(f"{name}: {type(exc).__name__}: {exc}")
        print(f"FAIL {name} :: {type(exc).__name__}: {exc}", flush=True)
    else:
        PASSES.append(name)
        print(f"PASS {name} :: {detail or ''}", flush=True)


# ---------------------------------------------------------------------------
# Section 1 -- the invariant, over the source. Needs nothing installed.
# ---------------------------------------------------------------------------

def _panel_tree():
    return ast.parse(PANEL_SOURCE.read_text(encoding="utf-8"))


def _pyvista_widget_class():
    for node in ast.walk(_panel_tree()):
        if isinstance(node, ast.ClassDef) and node.name == "PyVistaWidget":
            return node
    raise AssertionError("class PyVistaWidget is gone from the panel source")


def _methods(cls):
    return {n.name: n for n in cls.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


#: "no particular value" -- distinct from ``None``, which is a value these
#: checks need to assert LITERALLY. With ``None`` as the sentinel, asking
#: "does `load_results` set the home view to None" degrades to "does it assign
#: the home view at all", and a `load_results` that RE-CAPTURED the view
#: instead of clearing it would pass.
_ANY_VALUE = object()


def _assigns_self_attr(fn, attr, value=_ANY_VALUE):
    """True if `fn`'s body assigns `self.<attr>` (optionally to `value`)."""
    for node in ast.walk(fn):
        if not isinstance(node, ast.Assign):
            continue
        for tgt in node.targets:
            if (isinstance(tgt, ast.Attribute) and tgt.attr == attr
                    and isinstance(tgt.value, ast.Name) and tgt.value.id == "self"):
                if value is _ANY_VALUE:
                    return True
                if isinstance(node.value, ast.Constant) and node.value.value is value:
                    return True
    return False


def check_plotter_rebind_clears_the_flag():
    """Every method that replaces the plotter must say the new one is unframed.

    This is defect 2 as a rule. ``init_ui`` is exempt: it builds the first
    plotter on a widget that has never loaded anything, so the flag is already
    absent.
    """
    cls = _pyvista_widget_class()
    offenders = []
    for name, fn in _methods(cls).items():
        if name in ("__init__", "init_ui"):
            continue
        if not _assigns_self_attr(fn, "plotter"):
            continue
        if not _assigns_self_attr(fn, LOADED_FLAG, value=False):
            offenders.append(name)
    assert not offenders, (
        f"these PyVistaWidget methods rebind self.plotter without clearing "
        f"self.{LOADED_FLAG}: {sorted(offenders)}. update_display will read the "
        f"new plotter's DEFAULT camera as the user's view and re-impose it, "
        f"which is what left the canvas dark after an animation export.")
    rebinders = sorted(n for n, f in _methods(cls).items()
                       if n not in ("__init__", "init_ui")
                       and _assigns_self_attr(f, "plotter"))
    assert rebinders, (
        "no method rebinds self.plotter any more -- this check has become "
        "vacuous and must be re-aimed at whatever replaced _reinit_plotter")
    return f"{len(rebinders)} rebinder(s) checked: {rebinders}"


def check_new_results_forget_the_previous_home_view():
    """``load_results`` must clear BOTH camera flags, together.

    A home view left over from another mesh is worse than having none: Reset
    View would fly to a place the new dataset does not occupy.
    """
    fn = _methods(_pyvista_widget_class()).get("load_results")
    assert fn is not None, "PyVistaWidget.load_results is gone"
    assert _assigns_self_attr(fn, LOADED_FLAG, value=False), (
        f"load_results no longer clears self.{LOADED_FLAG}")
    assert _assigns_self_attr(fn, HOME_STATE, value=None), (
        f"load_results does not clear self.{HOME_STATE}, so opening a second "
        f"results directory keeps the first one's Reset View target")
    return "both camera flags cleared on a new results directory"


# ---------------------------------------------------------------------------
# Section 2 -- the behavior, against real VTK, asserted on rendered pixels.
# ---------------------------------------------------------------------------

def _write_run(root, z0, sign, n=40, half=2000.0):
    """A minimal results directory: one mesh, one field, one timestep.

    The PLY is written in the layout ``PyVistaWidget.read_binary_mesh``
    requires -- double x/y/z/nx/ny/nz, and faces carrying a uchar count, three
    int indices and per-face float latitude/longitude. A plain VTK-written PLY
    is NOT readable by that method, so this writer is not a convenience.
    """
    import numpy as np

    root = Path(root)
    (root / "temperature").mkdir(parents=True, exist_ok=True)
    u = np.linspace(-half, half, n)
    X, Y = np.meshgrid(u, u)
    Z = sign * (z0 + 120.0 * np.sin(X / 700.0) * np.cos(Y / 900.0))
    verts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]).astype(np.float64)

    tris = []
    for j in range(n - 1):
        for i in range(n - 1):
            a = j * n + i
            tris.append((a, a + n, a + 1))
            tris.append((a + 1, a + n, a + n + 1))
    tris = np.asarray(tris, dtype=np.int32)

    cen = verts[tris].mean(axis=1)
    rad = np.linalg.norm(cen, axis=1)
    lat = np.degrees(np.arcsin(cen[:, 2] / rad)).astype(np.float32)
    lon = np.degrees(np.arctan2(cen[:, 1], cen[:, 0])).astype(np.float32)
    normals = np.zeros_like(verts)
    normals[:, 2] = float(sign)

    with open(root / "mesh.ply", "wb") as f:
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

    nf = len(tris)
    temps = (200.0 + 60.0 * np.linspace(0.0, 1.0, nf)).astype("<f4")
    for step in range(3):
        with open(root / "temperature" / f"temperature_{step:04d}.bin", "wb") as f:
            f.write(struct.pack("<I", nf))
            f.write((temps + 3.0 * step).astype("<f4").tobytes())
    with open(root / "time_steps.txt", "w") as f:
        f.write("# step\ttime\tlabel\n")
        for step in range(3):
            f.write(f"{step}\t{step * 3600.0}\tstep {step}\n")
    return root


def _viewer_harness():
    """A real PyVistaWidget on a real offscreen plotter, or ``Skip``."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    # ⛔ This section needs the canvas, so it must NOT take the disable path.
    os.environ.pop("PTM_GUI_DISABLE_3D", None)
    try:
        import PyQt6
        from PyQt6.QtWidgets import QApplication, QWidget
    except Exception as exc:                                    # noqa: BLE001
        raise Skip(f"PyQt6 is not importable here ({exc})")
    # ⛔ Without these, Qt cannot find its offscreen platform plugin and
    # `QApplication([])` does not raise -- it calls abort(). SIGABRT is not
    # catchable by the `except Skip` below, and it takes the buffered stdout
    # with it, so the whole file reports NOTHING: not a pass, not a failure,
    # not a skip. Measured: rc=134, empty stdout. Same three lines as
    # `test_gui_batch_checkpoint.py`.
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
            # ⛔ Held in a module global on purpose. A QApplication that is only
            # a local goes out of scope and is collected, and the next QWidget
            # aborts the process with "Must construct a QApplication before a
            # QWidget" -- no traceback, no failure report, just an exit.
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
        """Stand-in for QtInteractor, which segfaults offscreen. See module
        docstring: same BasePlotter camera and actor semantics."""

        def __init__(self, parent=None, **kwargs):
            super().__init__(off_screen=True, window_size=(480, 360))
            self.interactor = QWidget(parent)

    panel.QtInteractor = OffscreenInteractor
    return panel


def _canvas_md5(widget):
    import numpy as np
    widget.plotter.render()
    img = np.asarray(widget.plotter.screenshot(return_img=True))[..., :3]
    background = np.array([30, 30, 30])           # the panel's '#1e1e1e'
    covered = float((np.abs(img.astype(np.int16) - background).max(axis=-1) > 8).mean())
    return hashlib.md5(img.tobytes()).hexdigest(), round(covered, 6)


def _camera(widget):
    cam = widget.plotter.renderer.camera
    return (tuple(round(v, 3) for v in cam.position),
            tuple(round(v, 3) for v in cam.focal_point),
            tuple(round(v, 6) for v in cam.up),
            round(float(cam.view_angle), 6))


def _orbit(widget, azimuth, elevation):
    cam = widget.plotter.camera
    cam.Azimuth(azimuth)
    cam.Elevation(elevation)
    cam.OrthogonalizeViewUp()
    widget.plotter.renderer.reset_camera_clipping_range()
    widget.plotter.render()


def _loaded_viewer(panel, run_dir):
    widget = panel.PyVistaWidget()
    assert widget.load_results(str(run_dir)), f"{run_dir} did not load"
    assert widget.mesh is not None, f"{run_dir} produced no mesh"
    return widget


def check_reset_view_returns_to_the_opening_view(tmp):
    panel = _viewer_harness()
    run = _write_run(Path(tmp) / "south", 1737400.0, -1)
    view = _loaded_viewer(panel, run)
    assert view.detected_hemisphere == "south", (
        f"fixture is meant to exercise the south-polar camera branch, got "
        f"{view.detected_hemisphere!r}")
    opening, opening_cov = _canvas_md5(view)
    opening_cam = _camera(view)

    _orbit(view, 35.0, 12.0)
    orbited, _ = _canvas_md5(view)
    assert orbited != opening, (
        "the orbit did not change the canvas, so the rest of this check is "
        "vacuous")

    view.reset_camera()
    after, after_cov = _canvas_md5(view)
    assert after == opening, (
        f"Reset View did not return to the opening view: canvas {after[:12]} "
        f"(coverage {after_cov}) vs opening {opening[:12]} (coverage "
        f"{opening_cov}); camera {_camera(view)} vs {opening_cam}")
    return f"canvas md5 {opening[:12]}, coverage {opening_cov}"


def check_reset_view_survives_a_field_switch(tmp):
    panel = _viewer_harness()
    run = _write_run(Path(tmp) / "south_fields", 1737400.0, -1)
    view = _loaded_viewer(panel, run)
    opening_cam = _camera(view)

    view.set_timestep(2)
    _orbit(view, -50.0, -20.0)
    assert _camera(view) != opening_cam, "the orbit did not move the camera"
    view.reset_camera()
    # ⛔ The CAMERA, not the canvas: a different timestep paints different
    # colors on the same geometry, so the pixels are expected to differ and an
    # image comparison here would fail for a reason that is not the bug. The
    # claim under test is about where the camera is.
    assert _camera(view) == opening_cam, (
        f"Reset View after a timestep change landed on {_camera(view)} instead "
        f"of the opening view {opening_cam}; the recorded home view is supposed "
        f"to outlive field and timestep changes")
    return "home view outlived a timestep change"


def check_canvas_is_not_dark_after_an_animation_export(tmp):
    panel = _viewer_harness()
    run = _write_run(Path(tmp) / "south_export", 1737400.0, -1)
    view = _loaded_viewer(panel, run)
    _orbit(view, 20.0, 8.0)
    before, before_cov = _canvas_md5(view)
    before_cam = _camera(view)

    ok = view.export_animation(str(Path(tmp) / "anim.gif"),
                               format_type="gif", fps=5)
    assert ok, "export_animation reported failure on the fixture run"

    after, after_cov = _canvas_md5(view)
    actors = view.plotter.renderer.actors
    assert "main_mesh" in actors, (
        f"the mesh actor is gone after the export; actors={sorted(actors)}")
    array = actors["main_mesh"].GetMapper().GetArrayName()
    assert array == view.current_field, (
        f"the mesh actor lost its scalars after the export: mapper array "
        f"{array!r}, field {view.current_field!r}")
    assert after_cov >= 0.9 * before_cov, (
        f"the canvas went dark after the animation export: mesh covered "
        f"{before_cov} of it before and {after_cov} after; the actor and its "
        f"scalars are intact, so this is the camera")
    assert _camera(view) == before_cam, (
        f"the export did not restore the view it was shot from: "
        f"{_camera(view)} vs {before_cam}")
    assert after == before, (
        f"the canvas after the export ({after[:12]}) is not the canvas before "
        f"it ({before[:12]})")
    return f"coverage {before_cov} -> {after_cov}, camera restored"


def check_reset_view_still_works_after_an_export(tmp):
    panel = _viewer_harness()
    run = _write_run(Path(tmp) / "south_reset_after", 1737400.0, -1)
    view = _loaded_viewer(panel, run)
    opening, _ = _canvas_md5(view)
    _orbit(view, 20.0, 8.0)
    assert view.export_animation(str(Path(tmp) / "anim2.gif"),
                                 format_type="gif", fps=5)
    view.reset_camera()
    after, after_cov = _canvas_md5(view)
    assert after == opening, (
        f"after an animation export, Reset View landed on {after[:12]} "
        f"(coverage {after_cov}) instead of the opening view {opening[:12]}; "
        f"this is the combination the user was stuck in")
    return "recoverable after an export"


def check_a_second_dataset_does_not_inherit_the_first_view(tmp):
    """Two real datasets, ~3,475 km apart, loaded into one widget."""
    panel = _viewer_harness()
    south = _write_run(Path(tmp) / "pair_south", 1737400.0, -1, n=40)
    north = _write_run(Path(tmp) / "pair_north", 1737400.0, +1, n=30, half=900.0)
    view = _loaded_viewer(panel, south)
    first_cells = view.mesh.n_cells
    _orbit(view, 40.0, 15.0)

    assert view.load_results(str(north)), "the second dataset did not load"
    # A silent PLY read failure leaves the FIRST mesh in place and makes what
    # follows pass vacuously. Guard it.
    assert view.mesh.n_cells != first_cells, (
        f"the second dataset's mesh did not replace the first "
        f"({view.mesh.n_cells} cells both times); the rest of this check "
        f"would be vacuous")
    assert view.detected_hemisphere == "north", (
        f"the second fixture is meant to exercise the other camera branch, "
        f"got {view.detected_hemisphere!r}")

    view.reset_camera()
    focal = tuple(round(float(v), 1) for v in _camera(view)[1])
    center = tuple(round(float(v), 1) for v in view.mesh.center)
    assert focal == center, (
        f"Reset View is aimed at {focal}, not at the new dataset's mesh "
        f"center {center} -- a home view from the previous dataset survived")
    _, coverage = _canvas_md5(view)
    assert coverage > 0.10, (
        f"the new dataset covers only {coverage} of the canvas after Reset "
        f"View, so the camera is not framing it")
    return f"focal {focal}, coverage {coverage}"


def check_a_terrain_preview_owns_its_own_reset_view(tmp):
    """Reset View must frame what is DRAWN, not a results mesh that is gone.

    ``show_mesh_preview`` clears the canvas and draws a bare DEM, deliberately
    without touching ``self.mesh`` or ``self.mesh_data``. The two geometries
    are nowhere near each other -- a results mesh sits at a body radius
    (~1,737 km here), a DEM preview in local meters -- so a Reset View aimed at
    the wrong one is a blank canvas, not a slightly-off view.

    ``VisualizationPanel.preview_terrain`` refuses to draw over a loaded
    results directory today, which makes this sequence unreachable through the
    UI. It is tested anyway, at the widget, because that guard is one edit away
    from moving and the failure it would let through is silent.
    """
    import numpy as np

    panel = _viewer_harness()
    from gui_core.mesh_preview import MeshPreview

    run = _write_run(Path(tmp) / "preview_then_reset", 1737400.0, -1)
    view = _loaded_viewer(panel, run)
    results_focal = tuple(round(float(v), 1) for v in view.mesh.center)

    n = 12
    u = np.linspace(-300.0, 300.0, n)
    gx, gy = np.meshgrid(u, u)
    verts = np.column_stack([gx.ravel(), gy.ravel(),
                             (40.0 * np.sin(gx / 90.0)).ravel()])
    faces = []
    for j in range(n - 1):
        for i in range(n - 1):
            a = j * n + i
            faces.append((a, a + n, a + 1))
            faces.append((a + 1, a + n, a + n + 1))
    preview = MeshPreview(vertices=verts, faces=np.asarray(faces, dtype=np.int64),
                          elevation=verts[:, 2].copy(), mode="surface",
                          units="meters")
    message = view.show_mesh_preview(preview)
    assert message == "", f"the preview refused to draw: {message}"

    _orbit(view, 30.0, 10.0)
    view.reset_camera()
    focal = tuple(round(float(v), 1) for v in _camera(view)[1])
    preview_center = tuple(round(float(v), 1) for v in verts.mean(axis=0))
    assert focal != results_focal, (
        f"Reset View is aimed at the RESULTS mesh {results_focal} while a "
        f"terrain preview is on screen; the preview is centered near "
        f"{preview_center} and would not be visible at all")
    _, coverage = _canvas_md5(view)
    assert coverage > 0.05, (
        f"the terrain preview covers only {coverage} of the canvas after "
        f"Reset View, so the camera is not framing what is drawn")
    return f"focal {focal}, coverage {coverage}"


def main():
    print("=" * 78)
    print("Section 1 -- the invariant (ast over the panel source; unskippable)")
    print("=" * 78)
    check("a method that rebinds the plotter clears the framed flag",
          check_plotter_rebind_clears_the_flag)
    check("a new results directory forgets the previous home view",
          check_new_results_forget_the_previous_home_view)

    print()
    print("=" * 78)
    print("Section 2 -- the behavior (real VTK; asserted on rendered pixels)")
    print("=" * 78)
    with tempfile.TemporaryDirectory(prefix="ptm_viewer_camera_") as tmp:
        for name, fn in (
            ("Reset View returns to the opening view",
             check_reset_view_returns_to_the_opening_view),
            ("Reset View survives a timestep change",
             check_reset_view_survives_a_field_switch),
            ("the canvas is not dark after an animation export",
             check_canvas_is_not_dark_after_an_animation_export),
            ("Reset View still works after an export",
             check_reset_view_still_works_after_an_export),
            ("a second dataset does not inherit the first view",
             check_a_second_dataset_does_not_inherit_the_first_view),
            ("a terrain preview owns its own Reset View",
             check_a_terrain_preview_owns_its_own_reset_view),
        ):
            check(name, lambda fn=fn: fn(tmp))

    print()
    print(f"{len(PASSES)} passed, {len(FAILURES)} failed, {len(SKIPS)} skipped")
    for line in SKIPS:
        print(f"  SKIPPED: {line}")
    for line in FAILURES:
        print(f"  FAILED:  {line}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
