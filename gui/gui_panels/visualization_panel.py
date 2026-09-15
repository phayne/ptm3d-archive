"""
Visualization Panel for PTM3D GUI
3D mesh visualization using PyVista and 2D plots using Matplotlib
Enhanced with face selection via click or lat/lon input, and data export
"""

import json
import logging
import numpy as np
import os
from pathlib import Path
import re
import struct
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QSplitter,
    QLabel, QSlider, QPushButton, QComboBox, QGroupBox, QGridLayout,
    QSpinBox, QDoubleSpinBox, QCheckBox, QFrame, QLineEdit, QFileDialog,
    QMessageBox, QDialog, QDialogButtonBox, QProgressDialog,
    QRadioButton, QButtonGroup, QApplication
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QValidator

#: ⛔ THE ONE THING THAT CANNOT BE BUILT WITHOUT A DISPLAY, AND THE REASON THE
#: WHOLE WINDOW COULD NOT BE TESTED.
#:
#: `pyvistaqt.QtInteractor` creates a native VTK render window, which needs a
#: real GL context; under `QT_QPA_PLATFORM=offscreen` it SEGFAULTS the process
#: (`pyvistaqt/rwi.py:386`, measured 2026-09-01, rc=139, no Python traceback
#: without `faulthandler`). Every offscreen harness in this repository has
#: therefore driven `ConfigurationPanel` directly and never built
#: `PTM3DMainWindow` -- so nothing has ever tested the real window's own
#: wiring, and three rounds of a flux-cube defect got past green suites
#: (2026-08-31a, 08-31b, 09-01) because the harness was a sub-widget.
#:
#: Setting `PTM_GUI_DISABLE_3D=1` takes the SAME path this module already
#: takes on a machine with no PyVista installed -- a path guarded at every one
#: of its ~20 use sites and exercised in production -- so the window builds and
#: everything except the 3D canvas is the real thing. It is opt-in by an
#: explicitly named variable, it is never set by the application, and it says
#: so out loud: a silently degraded window is exactly the class of defect this
#: exists to catch.
DISABLE_3D_CANVAS = os.environ.get("PTM_GUI_DISABLE_3D", "") == "1"

# The one shared decision about what a mesh's Cartesian frame means. Imported
# from `ptm.results` rather than reimplemented here: `gui_core.data_exporter`
# already reaches that module for the projection helpers, and a second copy of
# "is this mesh body-centered" is exactly the kind of duplicate catalog this
# file has had deleted from it twice.
from gui_core._ptm_bootstrap import ensure_ptm_importable  # noqa: E402

ensure_ptm_importable()
from ptm.results import describe_mesh_frame  # noqa: E402

# PyVista for 3D visualization
try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    print("Warning: PyVista not available. 3D visualization disabled.")

if DISABLE_3D_CANVAS and PYVISTA_AVAILABLE:
    PYVISTA_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "[visualization] PTM_GUI_DISABLE_3D=1 is set: the 3D canvas is NOT "
        "being created and every 3D view in this window is inert, even though "
        "PyVista is installed and would work. This exists so a headless test "
        "can build the real main window; if you are looking at a running "
        "application with no 3D view, this variable is why.")
    print("Warning: PTM_GUI_DISABLE_3D=1 -- 3D visualization disabled on "
          "purpose. Unset it to get the 3D view back.")

# Matplotlib for 2D plots
try:
    import matplotlib
    matplotlib.use('QtAgg')
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: Matplotlib not available. 2D plotting disabled.")


# --- Fields that carry sentinel values -------------------------------------
#
# ⭐ WHY THIS SECTION EXISTS: 0.0 AND +inf MEAN OPPOSITE THINGS IN THE SAME
# ARRAY, AND AN AUTOSCALED COLOR BAR CANNOT TELL THEM APART.
#
# `shadow_depth` is the height above the local surface at which the Sun is not
# even partially visible -- the umbral ceiling. The engine writes three things
# that are not measurements into the same float32 array, and says so itself in
# `statistics/shadow_depth.json` under "sentinel_encoding":
#
#     0.0    the Sun is visible standing on the facet -- CLEAR AT THE SURFACE.
#     +inf   still umbral at the search ceiling `h_max_m` -- DEEPER THAN THE
#            RUN LOOKED. Inside a permanently shadowed region this is the
#            MAJORITY value BY DESIGN; it is not a failure, and the engine
#            deliberately invents no number for it.
#     NaN    not evaluated. In the avg/max/min files it additionally marks a
#            facet with no resolved, nonzero epoch at all, because those
#            statistics are taken over resolved nonzero epochs only.
#
# A color range taken as (min, max) over such an array is (0, +inf): every
# facet renders one flat color, and the two OPPOSITE meanings -- sunlit at your
# feet, and shadowed deeper than five meters -- become the same pixel. So the
# rule here is: scale to the FINITE range, and give each sentinel its own
# off-scale color plus a plain-text legend that names it.
#
# The five functions below are pure -- numpy in, plain data out, no Qt and no
# PyVista -- so the decision they make is testable on a machine that cannot
# open a render window, which is every machine this repository's GUI suites run
# on. See `src/python/tests/test_gui_field_registry.py`.

#: The shadow-depth products, per-step series and statistics alike. These are
#: the ones that carry BOTH sentinels.
SHADOW_DEPTH_FIELDS = frozenset((
    'shadow_depth', 'shadow_depth_avg', 'shadow_depth_max', 'shadow_depth_min'))

#: The two companion fractions. No +inf -- they are bounded [0, 1] -- but the
#: engine leaves them NaN on a face with no evaluated epoch at all
#: (`shadow_depth_pass.cpp`), so they need the NaN color and the legend too.
SHADOW_FRACTION_FIELDS = frozenset((
    'shadow_lit_frac', 'shadow_unresolved_frac'))

#: Everything the off-scale treatment applies to.
SHADOW_SENTINEL_FIELDS = SHADOW_DEPTH_FIELDS | SHADOW_FRACTION_FIELDS

#: `effective_albedo` (PTM-107(a)) carries the OTHER sentinel, on its own. It is
#: NaN wherever the face was never lit during the integration window, because
#: the ratio it reports has no denominator there and the engine writes no number
#: rather than inventing one. No +inf -- the quantity is bounded -- so these
#: fields take the NaN color and the legend, and NOT the above-range color.
#: The population is not a rounding detail: 26,057 of 2,397,890 faces on the
#: reference 1 h lunar polar run, exactly the count that run's own log reports
#: as never lit.
UNDEFINED_RATIO_FIELDS = frozenset(('effective_albedo',))

#: Every field whose array can hold NaN meaning "no value here". The NaN color,
#: the legend pane and the never-autoscale rule key off THIS set; the
#: above-range color keys off `SHADOW_SENTINEL_FIELDS`, because those are the
#: only arrays that can hold +inf.
NAN_BEARING_FIELDS = SHADOW_SENTINEL_FIELDS | UNDEFINED_RATIO_FIELDS

#: The two indirect radiosity bands, per-step series and statistics alike.
#: They carry no sentinel; they carry a CONVENTION that the numbers do not show
#: and that the fields next to them in the menu do not share -- see
#: :func:`field_convention_note`.
INDIRECT_BAND_FIELDS = frozenset(('indirect_solar', 'indirect_ir'))

#: The sidecar the shadow-depth pass writes beside its statistics. It carries
#: `h_max_m`, the search ceiling the +inf sentinel refers to.
SHADOW_DEPTH_SIDECAR = 'shadow_depth.json'

#: ⛔ NO SINGLE OFF-SCALE COLOR IS SAFE, WHICH IS WHY THIS IS A LIST.
#:
#: The obvious implementation picks one color and asserts it is not in any
#: colormap. Measured against the 39 entries of
#: `VisualizationPanel.AVAILABLE_COLORMAPS`, that assertion is false for every
#: candidate: magenta is EXACTLY the top of `cool` and within 0.029 of a point
#: on `hsv`; white is in 25 of them; black in 9; red in 7. And the toolbar's
#: Colormap dropdown writes a per-field override, so a user can put any of
#: those under a shadow-depth field. A +inf facet drawn the same magenta as the
#: top of a `cool` ramp reads as "exactly at the ceiling" -- the one misreading
#: this whole treatment exists to prevent.
#:
#: So the color is chosen AGAINST the colormap in use, first match wins, and
#: when every candidate collides `off_scale_above_color` says so rather than
#: picking one quietly. Ordered by how many of the 39 offered maps each
#: collides with, measured: magenta 2, cyan 3, red 7, black 9, white 25. No
#: single offered map rules out more than 3 of the 5 today -- `hsv`, the worst,
#: still leaves black and white -- so the all-collide branch is not reachable
#: from the dropdown alone. It is kept because `set_cmap_override` accepts any
#: matplotlib colormap name, not only the curated list.
OFF_SCALE_ABOVE_CANDIDATES = (
    ('magenta', (1.0, 0.0, 1.0)),
    ('cyan',    (0.0, 1.0, 1.0)),
    ('red',     (1.0, 0.0, 0.0)),
    ('black',   (0.0, 0.0, 0.0)),
    ('white',   (1.0, 1.0, 1.0)),
)

#: How far a ramp color must stay from the sentinel color, as a Euclidean
#: distance in unit RGB (max possible sqrt(3) = 1.73). 0.25 keeps `Reds`'
#: darkest step (0.248 from pure red) on the wrong side, deliberately: at that
#: separation the two are not reliably distinguishable on a shaded 3D surface.
OFF_SCALE_MIN_RGB_DISTANCE = 0.25

#: NaN -- "no value here". Mid-gray, and NOT chosen against the colormap: it
#: marks absence rather than an extreme, so being readable as "not data" is the
#: whole requirement and it is stated in the legend either way.
#:
#: ⚠️ MEASURED LIMITATION, stated rather than assumed away. Because this color
#: is fixed, it CAN collide with the ramp -- the thing
#: `off_scale_above_color` exists to prevent for the other sentinel. Over the
#: 39 entries of `VisualizationPanel.AVAILABLE_COLORMAPS`, 9 come within
#: `OFF_SCALE_MIN_RGB_DISTANCE` of it: `gray`/`gray_r` (0.0000),
#: `cividis`/`cividis_r` (0.0300), `terrain` (0.1052), `Purples`/`Purples_r`
#: (0.2106), `gist_earth` (0.2351), `cubehelix` (0.2462). Two consequences,
#: one handled here and one not:
#:   * A NaN-bearing field must not DEFAULT to one of those 9. That is why
#:     `effective_albedo` -- whose obvious ramp for a reflectance is `gray` --
#:     defaults to `Blues_r` (0.3670) instead.
#:   * The shadow-depth family and `ice_depth` already default to `cividis_r`,
#:     0.0300 away, and the shadow-depth statistics on the reference run are ~52%
#:     NaN. Those facets are drawn in a gray the middle of their own ramp very
#:     nearly matches. That is a real defect, it is NOT introduced here, and
#:     fixing it means choosing this color against the ramp the way
#:     `off_scale_above_color` does -- which changes what those fields look
#:     like. Owed ledger row: PTM-GUI-NAN-GRAY-COLLIDES-WITH-RAMP.
OFF_SCALE_NAN_COLOR = '#808080'


def off_scale_above_color(cmap_name):
    """``(color_name, warning)`` for the +inf sentinel under ``cmap_name``.

    The first candidate no sampled ramp color comes within
    ``OFF_SCALE_MIN_RGB_DISTANCE`` of. ``warning`` is ``''`` when one was
    found, and a sentence naming the problem when every candidate collides.
    The warning is returned, not logged and dropped, because it goes in front
    of the user: a sentinel that cannot be told from the ramp must say so
    rather than look like data.

    Falls back to the first candidate, and says so, when matplotlib is not
    importable or the colormap is unknown -- there is then no ramp to compare
    against, and refusing to draw would be worse than drawing with a caveat.
    """
    default = OFF_SCALE_ABOVE_CANDIDATES[0][0]
    if not MATPLOTLIB_AVAILABLE or not cmap_name:
        return default, ""
    try:
        ramp = plt.get_cmap(cmap_name)(np.linspace(0.0, 1.0, 256))[:, :3]
    except Exception:                                        # noqa: BLE001
        return default, (
            f"The colormap \"{cmap_name}\" could not be sampled, so the "
            f"off-scale color was not checked against it; if the ramp itself "
            f"contains {default}, the two are not distinguishable.")
    for name, rgb in OFF_SCALE_ABOVE_CANDIDATES:
        distance = np.sqrt(((ramp - np.asarray(rgb)) ** 2).sum(axis=1)).min()
        if distance >= OFF_SCALE_MIN_RGB_DISTANCE:
            return name, ""
    return default, (
        f"The colormap \"{cmap_name}\" contains every color this view can use "
        f"for the off-scale sentinel, so \"deeper than the search ceiling\" is "
        f"drawn in {default}, which also appears on the ramp. Choose another "
        f"colormap to tell the two apart.")


def nan_color_collision_warning(cmap_name):
    """A sentence when ``cmap_name`` itself contains the NaN gray, or ``''``.

    ⭐ THE DISCLOSURE THE FIXED NaN COLOR OWES. `off_scale_above_color` picks
    the +inf color AGAINST the ramp and complains when it cannot; the NaN color
    is a constant and cannot do either, so the collision it is measured to have
    -- `cividis_r`, the default for the whole shadow-depth family and for
    `ice_depth`, is 0.0300 away, and those maps are ~52% NaN on the reference
    run --
    would otherwise be visible only to somebody reading this file. A facet
    drawn in a gray the middle of its own color bar also uses reads as data.

    This does not change any color. It says, in front of the user, that two
    things on screen are hard to tell apart, which is the difference between a
    known limitation and a silent one. The fix is to choose the NaN color
    against the ramp as well: PTM-GUI-NAN-GRAY-COLLIDES-WITH-RAMP.
    """
    if not MATPLOTLIB_AVAILABLE or not cmap_name:
        return ""
    try:
        from matplotlib.colors import to_rgb
        ramp = plt.get_cmap(cmap_name)(np.linspace(0.0, 1.0, 256))[:, :3]
        nan_rgb = np.asarray(to_rgb(OFF_SCALE_NAN_COLOR))
    except Exception:                                        # noqa: BLE001
        return ""
    distance = float(np.sqrt(((ramp - nan_rgb) ** 2).sum(axis=1)).min())
    if distance >= OFF_SCALE_MIN_RGB_DISTANCE:
        return ""
    return (
        f"Caution: the colormap \"{cmap_name}\" itself contains the gray this "
        f"view uses for \"no value\" (within {distance:.2f} in RGB), so a facet "
        f"with no value and a facet in the middle of the color bar are hard to "
        f"tell apart. Choose another colormap to separate them.")


def finite_data_range(data):
    """``(min, max)`` over the FINITE entries of ``data``, or ``None``.

    ``np.min``/``np.max`` propagate NaN and return ``inf`` -- which is exactly
    how one sentinel turns a color bar into one flat color. This ignores both.
    ``None`` when nothing finite is left: that is the honest answer and it is
    distinct from ``(0.0, 0.0)``.
    """
    if data is None:
        return None
    arr = np.asarray(data, dtype=np.float64)
    if arr.size == 0:
        return None
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None
    return (float(finite.min()), float(finite.max()))


def shadow_depth_ceiling_m(results_dir):
    """``h_max_m`` from ``statistics/shadow_depth.json``, or ``None``.

    ``None`` for a run that wrote no sidecar, an unreadable one, or one whose
    ceiling is not a positive number. The caller then falls back to the finite
    range of the data, which is always available. Nothing is guessed here: the
    ceiling is the engine's own number or it is absent.
    """
    if not results_dir:
        return None
    path = Path(results_dir) / 'statistics' / SHADOW_DEPTH_SIDECAR
    try:
        with open(path, 'r') as handle:
            sidecar = json.load(handle)
    except (OSError, ValueError):
        return None
    if not isinstance(sidecar, dict):
        return None
    ceiling = sidecar.get('h_max_m')
    try:
        ceiling = float(ceiling)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(ceiling) or ceiling <= 0:
        return None
    return ceiling


def shadow_depth_clim(data_range, ceiling_m):
    """The color range for a shadow-depth field: ``(0.0, ceiling)``.

    The bottom is pinned at EXACTLY 0 because 0 is a meaning ("the Sun is
    visible standing here"), not a small number. The avg/max/min files exclude
    zero epochs, so their own minimum is the bisection's finest step (~0.01 m);
    an autoscaled bar would paint a one-centimeter shadow the same color as no
    shadow at all.

    The top is the engine's search ceiling when the sidecar gave one, so the
    bar means the same thing in every run of the same configuration, and the
    finite maximum otherwise (or whichever is larger, so no real value is ever
    pushed off-scale by a sidecar that disagrees with its own data). ``None``
    when neither is available -- there is nothing to scale to, and the caller
    must not invent one.

    One consequence, measured rather than assumed: over the default
    ``h_max_m = 5`` the 256-step ramp gives 19.5 mm per color step, which is
    the engine's own final bisection bracket at the default ladder
    (5 / (16 * 2**4)). So the bar resolves the search exactly as finely as the
    search resolves itself, and no finer -- 0.0 and the smallest reportable
    shadow, 0.0098 m, land in the same step. Distinguishing those two is what
    ``shadow_lit_frac`` is for; distinguishing 0.0 from +inf, which is the
    opposite-meaning pair, is what the off-scale color is for.
    """
    top = None
    if ceiling_m is not None and np.isfinite(ceiling_m) and ceiling_m > 0:
        top = float(ceiling_m)
    if data_range is not None and np.isfinite(data_range[1]):
        if top is None or data_range[1] > top:
            top = float(data_range[1])
    if top is None or not np.isfinite(top) or top <= 0:
        return None
    return (0.0, top)


def sentinel_legend(field_name, ceiling_m=None, above_color=None, warning=""):
    """One plain sentence naming what each off-scale color means, or ``''``.

    This is the legend the color bar cannot draw: the bar spans the finite
    range, and the sentinels sit outside it by construction. Qualitative on
    purpose -- it states meanings, not counts, so it does not change from
    timestep to timestep and carries no number the engine did not write.

    ``above_color`` is the name :func:`off_scale_above_color` actually chose,
    so the sentence names the color on screen rather than a color this module
    hoped for; ``warning`` is that function's complaint, appended verbatim when
    it has one.
    """
    ceiling = 'the search ceiling'
    if ceiling_m is not None and np.isfinite(ceiling_m) and ceiling_m > 0:
        ceiling = f"{ceiling_m:g} m"
    above = above_color or OFF_SCALE_ABOVE_CANDIDATES[0][0]
    if field_name == 'shadow_depth':
        text = (
            "Shadow depth: the height above the local surface at which the Sun "
            "is not even partially visible. 0 m, the bottom of the color bar, "
            f"means the Sun is visible standing on the facet. {above.capitalize()} "
            f"means still in shadow above {ceiling} -- deeper than this run "
            f"looked, NOT {ceiling}. Gray means not evaluated.")
    elif field_name in SHADOW_DEPTH_FIELDS:
        text = (
            "Shadow depth: the height above the local surface at which the Sun "
            "is not even partially visible, reduced over the output window. "
            "0 m, the bottom of the color bar, means the Sun is visible "
            "standing on the facet. This statistic covers resolved, nonzero "
            "epochs only, so gray means the facet had none of them -- open "
            "\"Fraction of Epochs Umbral Above the Search Ceiling\" to see "
            "where those facets are.")
    elif field_name == 'shadow_unresolved_frac':
        text = (
            "Fraction of the evaluated epochs at which the facet was still in "
            f"shadow above {ceiling}. 1.0 is a facet that was never resolved: "
            "a permanently shadowed floor, by design, not a failed search. "
            "Gray means the facet had no evaluated epoch at all.")
    elif field_name == 'shadow_lit_frac':
        text = (
            "Fraction of the evaluated epochs at which the Sun was visible "
            "standing on the facet -- shadow depth exactly 0. Gray means the "
            "facet had no evaluated epoch at all.")
    elif field_name in UNDEFINED_RATIO_FIELDS:
        text = (
            "Effective albedo: the flux-weighted albedo the surface boundary "
            "condition actually absorbed with, reduced over the integration "
            "window. Dimensionless. Gray means the facet was never lit during "
            "the window, so there is no incident flux to form the ratio from "
            "and the engine wrote no number rather than inventing one -- a "
            "legitimate population, not a gap in the data.")
    else:
        return ""
    return f"{text} {warning}".strip() if warning else text


def field_convention_note(field_name):
    """A sentence about a convention the numbers do not show, or ``''``.

    Not a sentinel legend: nothing is off-scale and no color needs naming. The
    two indirect radiosity bands are stored as ABSORBED flux at the facet,
    while ``direct_solar``, ``primary_reflected`` and ``primary_ir`` -- sitting
    directly above them in the same menu, in the same units, on the same mesh
    -- are stored INCIDENT. The engine states that choice at its own writer and
    nothing on screen shows it, so a reader who compares the two numbers is
    misled by a view that is working correctly.

    Statistics share their base field's note: ``indirect_ir_max`` is the same
    quantity reduced, and the convention is what it is.
    """
    base = str(field_name or "")
    for suffix in ('_avg', '_max', '_min'):
        if base.endswith(suffix):
            base = base[:-len(suffix)]
            break
    if base in INDIRECT_BAND_FIELDS:
        return (
            "Indirect bands are ABSORBED flux at the facet. Direct Solar, "
            "Primary Reflected and Primary IR in this same menu are INCIDENT "
            "flux, so the two are not directly comparable. The two-band "
            "radiosity solve produces exactly these absorbed band fluxes, and "
            "their sum is the value coupled into the thermal boundary "
            "condition.")
    return ""


class PyVistaWidget(QWidget):
    """3D visualization widget using PyVista"""

    face_selected = pyqtSignal(int, dict)  # face_id, data dict

    #: What the coordinate readout says with the pointer off the mesh, and what
    #: it says when the mesh's frame does not support a coordinate at all. The
    #: second one is not a failure message -- it is the correct answer for a
    #: local tangent-plane patch, and it has to be distinguishable from a
    #: coordinate that merely has not been computed yet.
    COORD_READOUT_IDLE = "lat / lon: move the pointer over the mesh"
    COORD_READOUT_OFF_MESH = "lat / lon: \u2014 (pointer is off the mesh)"
    COORD_READOUT_NO_FRAME = (
        "lat / lon: unavailable \u2014 this mesh has no georeferenced frame")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.mesh = None
        self.current_data = None
        self.current_field = "temperature"
        self.current_timestep = 0
        self.results_dir = None
        self.timesteps = []
        self.mesh_data = None
        self.detected_hemisphere = 'global'  # 'north', 'south', or 'global'

        # The camera state the results opened with -- what "Reset View" means.
        # Captured once per results directory (see `update_display`), cleared
        # by `load_results` so a new dataset can never inherit the previous
        # one's framing. `None` means nothing has been displayed yet.
        self._home_camera_state = None

        # Hover highlighting state
        self.hover_face_id = None
        self.hover_highlight_actor = None
        self.selected_face_id = None

        # -- coordinate readout ------------------------------------------
        # What `ptm.results.describe_mesh_frame` decided about THIS mesh's
        # Cartesian frame. `None` until a mesh is loaded; `.usable` False means
        # no latitude and longitude are shown at all, because a number computed
        # on an assumption about the frame is worse than no number.
        self._mesh_frame = None
        # A `vtkCellPicker` with a `vtkCellLocator` over the results mesh.
        # ⭐ MEASURED on the 2,397,890-facet reference run: the picker WITHOUT a
        # locator takes 537 ms per pick, which is what the shipped hover path
        # did on every mouse move -- under 2 picks per second. With the locator
        # it is 0.054 ms, a factor of ~9,900, and no throttle is needed.
        # ⛔ The tolerance MUST stay small. At the 0.005 the old picker used,
        # the locator returns a cell 39.8 m (about 11 facets) from the point the
        # ray actually hit; at tolerance 0 the same picks land 1.1 m away,
        # which is inside a facet.
        self._hover_picker = None
        self._hover_locator = None
        # (interactor, observer_id) pairs, so a rebind REMOVES the old
        # observers instead of stacking a second copy on a new interactor.
        # `export_animation` replaces the plotter, and the observers used to
        # stay bound to the retired one while every flag still read "on"
        # (PTM-GUI-PICKING-DEAD-AFTER-EXPORT).
        self._interactor_observers = []

        # Color scale state
        self.fixed_color_scale = True
        self.color_range_min = None
        self.color_range_max = None
        self.global_data_range = None  # Will be computed from all timesteps

        # Per-field colormap overrides. Empty by default — falls back to
        # VisualizationPanel.FIELD_INFO[field][1]. The toolbar's Colormap
        # dropdown writes here; survives field switches so each field
        # remembers its last user-chosen palette.
        self.cmap_overrides = {}

        # Synthetic ("sun") lighting state. When enabled, the default
        # PyVista headlight is replaced by a directional sun light at
        # (sun_azimuth_deg, sun_elevation_deg) on the local-tangent
        # frame at the mesh centroid, plus a soft ambient fill.
        # Defaults are populated from the mesh latitudes on first load
        # (see _set_default_sun_from_latitude).
        self.synthetic_lighting_enabled = False
        self.sun_azimuth_deg = 0.0      # 0 = local north, 90 = east, 180 = south
        self.sun_elevation_deg = 30.0   # 0 = horizon, 90 = zenith
        self.sun_intensity = 1.0
        self.ambient_intensity = 0.2
        self.mesh_mean_lat_deg = 0.0

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if PYVISTA_AVAILABLE:
            # Create PyVista Qt interactor
            self.plotter = QtInteractor(self)
            self.plotter.set_background('#1e1e1e')
            self.plotter.add_axes()
            layout.addWidget(self.plotter.interactor, 1)

            # Face picking state
            self.picking_enabled = False
            self.picking_active = False

            # Add control buttons overlay at top
            controls_layout = QHBoxLayout()
            controls_layout.setContentsMargins(5, 5, 5, 0)

            # Select Face toggle button
            self.select_face_btn = QPushButton("Select Face Mode")
            self.select_face_btn.setCheckable(True)
            self.select_face_btn.setMaximumWidth(140)
            self.select_face_btn.setStyleSheet("""
                QPushButton {
                    background-color: #404040;
                    color: white;
                    border: 1px solid #606060;
                    border-radius: 4px;
                    padding: 5px 10px;
                }
                QPushButton:checked {
                    background-color: #ff9800;
                    color: black;
                    border: 1px solid #ffb74d;
                }
                QPushButton:hover {
                    background-color: #505050;
                }
                QPushButton:checked:hover {
                    background-color: #ffa726;
                }
            """)
            self.select_face_btn.clicked.connect(self.toggle_face_selection)
            controls_layout.addWidget(self.select_face_btn)

            # Status label
            self.pick_status_label = QLabel("")
            self.pick_status_label.setStyleSheet("color: #888888; font-size: 11px;")
            controls_layout.addWidget(self.pick_status_label)

            controls_layout.addStretch()

            # Color scale controls
            self.fixed_scale_check = QCheckBox("Fixed Scale")
            self.fixed_scale_check.setChecked(True)
            self.fixed_scale_check.setStyleSheet("color: #e0e0e0; font-size: 11px;")
            self.fixed_scale_check.stateChanged.connect(self.on_fixed_scale_changed)
            controls_layout.addWidget(self.fixed_scale_check)

            # Min value input
            self.scale_min_label = QLabel("Min:")
            self.scale_min_label.setStyleSheet("color: #888888; font-size: 11px;")
            self.scale_min_label.setVisible(False)
            controls_layout.addWidget(self.scale_min_label)

            self.scale_min_input = QDoubleSpinBox()
            self.scale_min_input.setRange(-1000, 10000)
            # 3 decimals, not 1: the menu now carries dimensionless fields --
            # `effective_albedo` spans 0.137-0.290 on the reference run, and the
            # two shadow fractions span 0-1. At one decimal those boxes read
            # 0.1 and 0.3 for the whole map, and editing either one snapped the
            # other to the rounded value. Temperature reads 24.277 instead of
            # 24.3, which is the same number with more of it shown.
            self.scale_min_input.setDecimals(3)
            self.scale_min_input.setValue(0)
            self.scale_min_input.setMaximumWidth(70)
            self.scale_min_input.setVisible(False)
            self.scale_min_input.valueChanged.connect(self.on_color_range_changed)
            controls_layout.addWidget(self.scale_min_input)

            # Max value input
            self.scale_max_label = QLabel("Max:")
            self.scale_max_label.setStyleSheet("color: #888888; font-size: 11px;")
            self.scale_max_label.setVisible(False)
            controls_layout.addWidget(self.scale_max_label)

            self.scale_max_input = QDoubleSpinBox()
            self.scale_max_input.setRange(-1000, 10000)
            self.scale_max_input.setDecimals(3)
            self.scale_max_input.setValue(400)
            self.scale_max_input.setMaximumWidth(70)
            self.scale_max_input.setVisible(False)
            self.scale_max_input.valueChanged.connect(self.on_color_range_changed)
            controls_layout.addWidget(self.scale_max_input)

            layout.insertLayout(0, controls_layout)

            # -- coordinate readout, bottom-right corner of the canvas ----
            #
            # A Qt row under the canvas, NOT a `plotter.add_text` overlay and
            # NOT a floating child raised over the interactor. Two reasons, both
            # deliberate: a VTK text actor is rendered INTO the scene, so it
            # would appear in every exported animation frame and change the
            # deliverable; and `QtInteractor.interactor` wraps a native render
            # window, over which sibling Qt widgets do not composite reliably on
            # macOS. This paints, always, and is readable without a display.
            readout_row = QHBoxLayout()
            readout_row.setContentsMargins(5, 0, 8, 3)
            readout_row.addStretch()
            self.coord_readout_label = QLabel(self.COORD_READOUT_IDLE)
            self.coord_readout_label.setObjectName("coordinateReadout")
            self.coord_readout_label.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse)
            self.coord_readout_label.setStyleSheet(
                "color: #c8c8c8; font-size: 11px; "
                "font-family: Menlo, Monaco, monospace; "
                "background-color: rgba(30, 30, 30, 200); "
                "padding: 2px 6px; border-radius: 3px;")
            readout_row.addWidget(self.coord_readout_label)
            layout.addLayout(readout_row)

        else:
            # Fallback message
            label = QLabel("PyVista not available.\nInstall with: pip install pyvista pyvistaqt")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet("color: #ff9800; font-size: 14px;")
            layout.addWidget(label)

    # -- terrain preview ---------------------------------------------------
    #
    # THIS METHOD IS NOT COVERED BY ANY TEST AND CANNOT BE. Constructing this
    # widget headless takes the process down with a `qFatal` from VTK, so the
    # only verification of what follows is a click test in the running
    # application. Everything that DECIDES anything -- which reader, which
    # elevation rule, which datum, whether the mesh is too big for a surface --
    # was moved into `gui_core.mesh_preview`, which imports no Qt and is
    # measured by `src/python/tests/test_gui_mesh_preview.py`. What is left
    # here is the pyvista call, and every part of it is guarded: this method
    # RETURNS a message and never raises, because an exception out of a Qt slot
    # is an application abort.

    #: Colormap for elevation. `terrain` is the one in AVAILABLE_COLORMAPS
    #: whose name says what it is being used for.
    PREVIEW_CMAP = "terrain"

    def show_mesh_preview(self, preview):
        """Draw a `mesh_preview.MeshPreview`, colored by elevation.

        Returns "" on success or a sentence describing what stopped it. It
        does not raise, and it does not touch `self.mesh` / `self.mesh_data`:
        those belong to a loaded RESULTS directory, and a preview that quietly
        replaced them would make the field selector describe a mesh that is no
        longer on screen.
        """
        if not PYVISTA_AVAILABLE:
            return ("PyVista is not installed, so no mesh can be drawn. "
                    "Install it with: pip install pyvista pyvistaqt")
        plotter = getattr(self, "plotter", None)
        if plotter is None:
            return "The 3D view is not available, so no mesh was drawn."
        if preview is None or preview.refusal or preview.vertices is None:
            return preview.refusal if preview is not None else \
                "No mesh was read, so nothing was drawn."
        try:
            points = np.asarray(preview.vertices, dtype=np.float64)
            if preview.mode == "surface" and preview.faces is not None \
                    and len(preview.faces):
                faces = np.asarray(preview.faces, dtype=np.int64)
                cells = np.column_stack(
                    [np.full(len(faces), 3, dtype=np.int64), faces]).ravel()
                poly = pv.PolyData(points, cells)
            else:
                poly = pv.PolyData(points)
            unit = {"meters": "m", "kilometers": "km"}.get(preview.units, "")
            title = f"Elevation ({unit})" if unit else "Elevation"
            if preview.elevation is not None \
                    and len(preview.elevation) == len(points):
                poly[title] = np.asarray(preview.elevation, dtype=np.float64)
            plotter.clear()
            kwargs = {"cmap": self.PREVIEW_CMAP, "show_edges": False}
            if title in poly.array_names:
                kwargs["scalars"] = title
                kwargs["scalar_bar_args"] = {"title": title}
            if preview.mode != "surface":
                kwargs["render_points_as_spheres"] = False
                kwargs["point_size"] = 2.0
            plotter.add_mesh(poly, **kwargs)
            plotter.reset_camera()
            # The preview is now what "opening" means for this canvas, so it
            # owns the Reset View target. Without this, Reset View would fly to
            # the home view of a RESULTS mesh that is no longer drawn -- a
            # lunar-radius camera over a DEM in local meters, i.e. a blank
            # canvas. `preview_terrain` refuses to draw over a loaded results
            # directory today, so that sequence is not currently reachable;
            # this keeps it unreachable if that guard ever moves.
            self._home_camera_state = self.capture_camera_state()
            return ""
        except Exception as exc:                             # noqa: BLE001
            # Anything at all: a VTK allocation failure, a pyvista API change,
            # a mesh whose indices do not address its own vertices. The pane
            # says so and the application stays up.
            return (f"The terrain preview could not be drawn "
                    f"({type(exc).__name__}: {exc}). The configuration is "
                    f"unaffected.")

    def load_results(self, results_dir):
        """Load simulation results from directory"""
        self.results_dir = Path(results_dir)
        if not self.results_dir.exists():
            return False

        # Reset color range when loading new results
        self.global_data_range = None

        # Reset camera tracking so first load gets default view centered on mesh
        self._has_loaded_data = False
        # ... and forget the previous mesh's coordinate frame with it, so a
        # results directory whose mesh fails to load cannot show the PREVIOUS
        # run's coordinates under the pointer.
        self._mesh_frame = None
        self._hover_picker = None
        self._hover_locator = None
        self.hover_face_id = None
        # ⛔ `load_mesh` -- and with it `_reset_coordinate_readout` -- only runs
        # when a mesh.ply is there. Without this, opening a results directory
        # that has none leaves the PREVIOUS run's coordinate on screen.
        self._reset_coordinate_readout()
        # ... and forget the previous dataset's home view along with it. A
        # stale camera from another mesh is worse than no camera: "Reset View"
        # would fly to a place this mesh does not occupy.
        self._home_camera_state = None

        # Load mesh
        mesh_file = self.results_dir / "mesh.ply"
        if mesh_file.exists():
            self.load_mesh(mesh_file)

        # Load timesteps
        time_file = self.results_dir / "time_steps.txt"
        if time_file.exists():
            self.load_timesteps(time_file)

        # Load initial data
        if len(self.timesteps) > 0:
            self.load_timestep_data(0)

        return True

    def load_mesh(self, mesh_file):
        """Load mesh from PLY file"""
        if not PYVISTA_AVAILABLE:
            return

        try:
            # Read PLY mesh data
            vertices, faces, normals, latitudes, longitudes = self.read_binary_mesh(mesh_file)

            # Create PyVista mesh
            n_faces = len(faces)
            cells = np.column_stack([
                np.full(n_faces, 3, dtype=np.int64),
                faces
            ]).ravel()

            self.mesh = pv.PolyData(vertices, cells)

            # Store face metadata
            self.mesh['latitude'] = latitudes
            self.mesh['longitude'] = longitudes

            # Store for later use
            self.mesh_data = {
                'vertices': vertices,
                'faces': faces,
                'normals': normals,
                'latitudes': latitudes,
                'longitudes': longitudes
            }

            # Decide, from the MESH, whether a latitude and longitude read off
            # it mean anything. This is the whole risk of the coordinate
            # readout: `asin(z/r)` returns a number for any mesh, including a
            # local tangent-plane patch where it is not a latitude.
            self._mesh_frame = describe_mesh_frame(self.mesh_data)
            self._build_hover_picker()

            # Detect hemisphere for camera orientation
            self.detected_hemisphere = self.detect_hemisphere(latitudes)

            # Pick default sun position from the mesh's mean latitude.
            # Doesn't override existing user choices on subsequent loads
            # within the same session unless the user clicks "Use
            # latitude default" in the lighting dialog.
            self._set_default_sun_from_latitude(latitudes)

            # Bind (or re-bind) the hover observers now that there is something
            # to hover over.
            self._bind_interactor_observers()
            self._reset_coordinate_readout()

            self.update_display()

        except Exception as e:
            print(f"Error loading mesh: {e}")

    def detect_hemisphere(self, latitudes):
        """Detect which hemisphere the mesh is predominantly in.

        Args:
            latitudes: Array of face latitudes in degrees

        Returns:
            str: 'north', 'south', or 'global' (spans both hemispheres significantly)
        """
        if latitudes is None or len(latitudes) == 0:
            return 'global'

        mean_lat = np.mean(latitudes)
        min_lat = np.min(latitudes)
        max_lat = np.max(latitudes)
        lat_range = max_lat - min_lat

        # If mesh spans more than 60 degrees of latitude, consider it global
        if lat_range > 60:
            # But still bias toward the hemisphere with more coverage
            if mean_lat < -15:
                return 'south'
            elif mean_lat > 15:
                return 'north'
            return 'global'

        # For regional meshes, use mean latitude
        if mean_lat < -5:
            return 'south'
        elif mean_lat > 5:
            return 'north'
        else:
            return 'global'

    def set_hemisphere_camera(self):
        """Set camera orientation based on detected hemisphere.

        For southern hemisphere meshes, orients the view so that south pole
        is "up" (looking from below the mesh with inverted z-axis).
        """
        if not PYVISTA_AVAILABLE or not self.plotter or not self.mesh:
            return

        hemisphere = getattr(self, 'detected_hemisphere', 'global')

        # Get mesh center and bounds for camera positioning
        bounds = self.mesh.bounds
        center = self.mesh.center
        x_range = bounds[1] - bounds[0]
        y_range = bounds[3] - bounds[2]
        z_range = bounds[5] - bounds[4]
        max_dim = max(x_range, y_range, z_range)

        if hemisphere == 'south':
            # View from below: camera at negative z, looking up, with +X as "up"
            camera_pos = (center[0], center[1], center[2] - max_dim * 2)
            focal_point = center
            view_up = (1, 0, 0)  # +X points up in screen for south polar projection
            self.plotter.camera_position = [camera_pos, focal_point, view_up]
        elif hemisphere == 'north':
            # Standard view from above: camera at positive z, looking down
            camera_pos = (center[0], center[1], center[2] + max_dim * 2)
            focal_point = center
            view_up = (0, 1, 0)  # Y-up
            self.plotter.camera_position = [camera_pos, focal_point, view_up]
        else:
            # Global mesh - use default isometric view
            self.plotter.reset_camera()
            self.plotter.view_isometric()

    def read_binary_mesh(self, mesh_file):
        """Read binary PLY mesh file"""
        with open(mesh_file, 'rb') as f:
            # Parse header
            header = self.parse_ply_header(f)

            is_little_endian = (header['format'] == 'binary_little_endian')
            endian = '<' if is_little_endian else '>'

            # Read vertices
            vertex_dtype = []
            for prop_type, prop_name in header['vertex_props']:
                if prop_type == 'float':
                    vertex_dtype.append((prop_name, f'{endian}f4'))
                elif prop_type == 'double':
                    vertex_dtype.append((prop_name, f'{endian}f8'))

            vertex_dtype = np.dtype(vertex_dtype)
            vertex_buffer = f.read(header['num_vertices'] * vertex_dtype.itemsize)
            vertex_array = np.frombuffer(vertex_buffer, dtype=vertex_dtype)

            vertices = np.empty((header['num_vertices'], 3), dtype=np.float64)
            vertices[:, 0] = vertex_array['x']
            vertices[:, 1] = vertex_array['y']
            vertices[:, 2] = vertex_array['z']

            normals = None
            if 'nx' in vertex_array.dtype.names:
                normals = np.empty((header['num_vertices'], 3), dtype=np.float64)
                normals[:, 0] = vertex_array['nx']
                normals[:, 1] = vertex_array['ny']
                normals[:, 2] = vertex_array['nz']

            # Read faces
            faces = np.empty((header['num_faces'], 3), dtype=np.int32)
            latitudes = np.empty(header['num_faces'], dtype=np.float32)
            longitudes = np.empty(header['num_faces'], dtype=np.float32)

            for i in range(header['num_faces']):
                count_byte = f.read(1)
                idx_data = f.read(12)
                faces[i] = struct.unpack(f'{endian}III', idx_data)
                lat_data = f.read(4)
                latitudes[i] = struct.unpack(f'{endian}f', lat_data)[0]
                lon_data = f.read(4)
                longitudes[i] = struct.unpack(f'{endian}f', lon_data)[0]

        return vertices, faces, normals, latitudes, longitudes

    def parse_ply_header(self, f):
        """Parse PLY file header"""
        header = {
            'format': 'ascii',
            'num_vertices': 0,
            'num_faces': 0,
            'vertex_props': [],
            'face_props': []
        }
        current_element = None

        while True:
            line = f.readline()
            if isinstance(line, bytes):
                line = line.decode('ascii')
            line = line.strip()

            if line == 'end_header':
                break

            parts = line.split()
            if not parts:
                continue

            if parts[0] == 'format':
                header['format'] = parts[1]
            elif parts[0] == 'element':
                current_element = parts[1]
                if current_element == 'vertex':
                    header['num_vertices'] = int(parts[2])
                elif current_element == 'face':
                    header['num_faces'] = int(parts[2])
            elif parts[0] == 'property':
                if current_element == 'vertex' and len(parts) >= 3:
                    header['vertex_props'].append((parts[1], parts[2]))

        return header

    def load_timesteps(self, time_file):
        """Load timestep metadata"""
        self.timesteps = []
        with open(time_file, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    self.timesteps.append({
                        'time': float(parts[0]),
                        'label': parts[1]
                    })

    def load_timestep_data(self, timestep_index):
        """Load data for a specific timestep"""
        if not self.results_dir or not self.mesh:
            return

        self.current_timestep = timestep_index
        n_faces = self.mesh.n_cells

        # Load requested field
        data = self.load_field_data(self.current_field, timestep_index, n_faces)
        if data is not None:
            self.current_data = data
            self.mesh[self.current_field] = data
            self.update_display()

    def load_field_data(self, field_name, timestep_index, expected_count):
        """Load a specific data field for a timestep, or a per-face annual
        statistic (in which case timestep_index is ignored).

        Statistics live in <results>/statistics/<field_name>.bin (single file,
        no timestep) and are recognized by
        `VisualizationPanel.is_stats_field`. That rule used to be spelled out a
        second time here, and the copies drifted the moment a scalar product
        that is not `ice_depth` arrived: `shadow_lit_frac` was listed in the
        menu and then looked for under `shadow_lit_frac/shadow_lit_frac_0000.bin`,
        a directory no run writes. One rule, one place.
        """
        # Statistics path: single file, no timestep index
        if VisualizationPanel.is_stats_field(field_name):
            stats_path = self.results_dir / 'statistics' / f"{field_name}.bin"
            if not stats_path.exists():
                return None
            return self.read_binary_data(stats_path, expected_count)

        # Per-timestep series path
        field_dirs = {
            'temperature': 'temperature',
            'direct_solar': 'direct_solar',
            'illumination': 'illumination',
            'primary_reflected': 'primary_reflected',
            'primary_ir': 'primary_ir'
        }

        dir_name = field_dirs.get(field_name, field_name)
        data_dir = self.results_dir / dir_name

        if not data_dir.exists():
            return None

        file_path = data_dir / f"{dir_name}_{timestep_index:04d}.bin"
        if not file_path.exists():
            return None

        return self.read_binary_data(file_path, expected_count)

    def read_binary_data(self, file_path, expected_count):
        """Read binary data file"""
        try:
            with open(file_path, 'rb') as f:
                # Try header format first
                header = f.read(4)
                if len(header) == 4:
                    count = struct.unpack('<I', header)[0]
                    expected_size = count * 4

                    # Validate
                    f.seek(0, 2)
                    file_size = f.tell()

                    if file_size == 4 + expected_size and count == expected_count:
                        # Header format
                        f.seek(4)
                        data = np.frombuffer(f.read(), dtype='<f4')
                        return data
                    else:
                        # Try headerless format
                        f.seek(0)
                        data = np.frombuffer(f.read(), dtype='<f4')
                        if len(data) == expected_count:
                            return data

            return np.zeros(expected_count, dtype=np.float32)

        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return np.zeros(expected_count, dtype=np.float32)

    def update_display(self):
        """Update the 3D display"""
        if not PYVISTA_AVAILABLE or not self.mesh:
            return

        # Save camera position before clearing to preserve user's view
        camera_position = self.plotter.camera_position

        self.plotter.clear()

        # Flag to check if we should restore camera (not on first load)
        # Use getattr to check the actual value, not just existence
        restore_camera = camera_position is not None and getattr(self, '_has_loaded_data', False)

        # Determine scalar field and colormap
        scalar_field = self.current_field
        if scalar_field in self.mesh.array_names:
            scalars = scalar_field
        else:
            scalars = None

        # Choose colormap based on field: user override, then FIELD_INFO,
        # then the 'temperature => plasma, else viridis' fallback. That order
        # used to be written out twice -- here and in `get_effective_cmap`,
        # whose docstring said it "mirrors the resolution order in
        # update_display()". The legend pane resolves the color of the
        # off-scale sentinel from `get_effective_cmap`, so two copies is two
        # answers and a legend that can name a color the view did not draw.
        cmap = self.get_effective_cmap()

        clim = self.effective_clim()

        # Add mesh to plotter with horizontal colorbar at bottom
        mesh_kwargs = {
            'scalars': scalars,
            'cmap': cmap,
            'show_edges': False,
            'lighting': True,
            'smooth_shading': True,
            'pickable': True,
            'name': 'main_mesh',
            'scalar_bar_args': {
                'title': self.get_field_label(),
                'color': 'white',
                'title_font_size': 12,
                'label_font_size': 10,
                'vertical': False,
                'position_x': 0.2,
                'position_y': 0.02,
                'width': 0.6,
                'height': 0.06,
                'n_labels': 5,
                'fmt': '%.1f'
            }
        }
        if clim is not None:
            mesh_kwargs['clim'] = clim

        # A sentinel-bearing field gives its off-scale values their own colors.
        # VTK maps +inf to the above-range color and NaN to the NaN color
        # directly, so NOTHING is substituted in the array: a probed facet
        # still reports `inf`, not a number this panel made up. Scoped to the
        # fields that actually carry sentinels -- setting a NaN color for every
        # field would change how every other field renders its gaps, which is a
        # separate decision and not this one.
        #
        # The two sets are DIFFERENT and deliberately so: `effective_albedo` is
        # NaN where a facet was never lit, so it needs the NaN color, but it
        # can never be +inf and an above-range color for it would name a
        # population that does not exist.
        if self.current_field in NAN_BEARING_FIELDS:
            mesh_kwargs['nan_color'] = OFF_SCALE_NAN_COLOR
        if self.current_field in SHADOW_DEPTH_FIELDS:
            above, _warning = off_scale_above_color(cmap)
            mesh_kwargs['above_color'] = above

        self.plotter.add_mesh(self.mesh, **mesh_kwargs)

        # Apply synthetic sun lighting if enabled (replaces default
        # headlight with a directional sun + soft ambient fill).
        self._apply_synthetic_lighting()

        # Re-add selection highlight if there was one
        if self.selected_face_id is not None:
            self.update_selection_highlight(self.selected_face_id)

        # Add timestamp label in upper-right corner (smaller font)
        if self.timesteps and 0 <= self.current_timestep < len(self.timesteps):
            ts_info = self.timesteps[self.current_timestep]
            ts_label = ts_info.get('label', f"Timestep {self.current_timestep}")
            self.plotter.add_text(
                ts_label,
                position='upper_right',
                font_size=9,
                color='white',
                name='timestamp_label'
            )

        # Add light source indicator (optional)
        if hasattr(self, 'sun_position'):
            self.plotter.add_point_labels(
                [self.sun_position], ['Sun'],
                point_size=20, font_size=12, text_color='yellow'
            )

        # Restore camera position if this isn't the first load
        if restore_camera:
            self.plotter.camera_position = camera_position
        else:
            # First load - set camera based on detected hemisphere
            self.set_hemisphere_camera()
            # This IS "the view when opening the results", so record it here,
            # at the one place it is established. Switching field or timestep
            # comes back through this method with `_has_loaded_data` already
            # true and takes the branch above, so the home view survives both
            # -- which is what Reset View is for.
            self._home_camera_state = self.capture_camera_state()
            # Mark that we've loaded data, so future updates preserve camera
            self._has_loaded_data = True

    def toggle_face_selection(self, checked):
        """Toggle face selection mode on/off"""
        if not PYVISTA_AVAILABLE or not self.mesh:
            self.select_face_btn.setChecked(False)
            self.pick_status_label.setText("Load results first")
            return

        if checked:
            self.enable_hover_picking()
            if self.picking_active:
                self.pick_status_label.setText("Hover to highlight, click to select")
            else:
                self.pick_status_label.setText("Selection mode failed")
                self.select_face_btn.setChecked(False)
        else:
            self.disable_hover_picking()
            self.pick_status_label.setText("")
            # Remove any hover highlight
            self.clear_hover_highlight()

    # -- hover coordinate readout ------------------------------------------

    def _build_hover_picker(self):
        """Build a `vtkCellPicker` backed by a `vtkCellLocator` over the mesh.

        Both halves are load-bearing and both were MEASURED on the
        2,397,890-facet reference run:

        * Without the locator, one pick costs **537 ms**. The shipped hover path
          did exactly that on every `MouseMoveEvent`, so face-selection mode was
          under two picks per second on a production mesh. With the locator it
          is **0.054 ms** -- fast enough that no throttle is needed, which is
          why there is none.
        * Tolerance has to stay at 0. With the locator attached and the old
          0.005 tolerance, the returned cell sits **39.8 m** (about 11 facets)
          from the point the ray actually hit; at 0 the same picks land
          **1.1 m** away, inside one facet. A coordinate readout that names a
          facet 40 m from the cursor is the "plausible but wrong" answer this
          feature exists to avoid.
        """
        self._hover_picker = None
        self._hover_locator = None
        if not PYVISTA_AVAILABLE or self.mesh is None:
            return
        try:
            import vtk
            locator = vtk.vtkCellLocator()
            locator.SetDataSet(self.mesh)
            locator.BuildLocator()
            picker = vtk.vtkCellPicker()
            picker.SetTolerance(0.0)
            picker.AddLocator(locator)
        except Exception as e:                                  # noqa: BLE001
            # Announced, not swallowed: falling back to a picker with no
            # locator would make the viewer feel broken rather than say
            # anything. The pair is already None.
            print(f"[Readout] could not build the accelerated picker ({e}); "
                  f"the coordinate readout is off.")
            return
        self._hover_locator = locator
        self._hover_picker = picker

    def _unbind_interactor_observers(self):
        """Drop every observer this widget added, from whichever interactor it
        was added to. Called before every (re)bind."""
        for iren, observer_id in self._interactor_observers:
            try:
                iren.RemoveObserver(observer_id)
            except Exception:                                   # noqa: BLE001
                pass
        self._interactor_observers = []

    def _bind_interactor_observers(self):
        """Point the hover observers at the CURRENT interactor.

        ⛔ `export_animation` calls `_reinit_plotter`, which replaces
        `self.plotter` with a new `QtInteractor` -- a different `iren` object.
        The observers stayed on the retired one while `picking_active`,
        `picking_enabled` and the "Select Face Mode" button all still read as
        on, so face selection silently stopped working after any animation
        export (PTM-GUI-PICKING-DEAD-AFTER-EXPORT). Anything that rebinds
        `self.plotter` must call this.

        It also fixes a second, older defect: the previous code added a fresh
        pair of observers on every toggle of face-selection mode and never
        removed any, so N toggles ran the pick N times per mouse move.
        """
        self._unbind_interactor_observers()
        if not PYVISTA_AVAILABLE or not getattr(self, 'plotter', None):
            return
        if self.mesh is None:
            return
        try:
            iren = self.plotter.iren.interactor
            self._interactor_observers = [
                (iren, iren.AddObserver('MouseMoveEvent', self.on_mouse_move)),
                (iren, iren.AddObserver('LeftButtonPressEvent',
                                        self.on_left_click)),
            ]
        except Exception as e:                                  # noqa: BLE001
            print(f"[Readout] could not attach the pointer observers: {e}")
            self._interactor_observers = []

    def _reset_coordinate_readout(self):
        """Put the readout back to its resting text for the current mesh."""
        label = getattr(self, 'coord_readout_label', None)
        if label is None:
            return
        frame = self._mesh_frame
        if frame is not None and not frame.usable:
            label.setText(self.COORD_READOUT_NO_FRAME)
            label.setToolTip(frame.reason)
        elif self.mesh is None:
            label.setText(self.COORD_READOUT_IDLE)
            label.setToolTip("")
        else:
            label.setText(self.COORD_READOUT_OFF_MESH)
            label.setToolTip(frame.label if frame is not None else "")

    def face_coordinate_text(self, face_id):
        """The readout line for one facet, or the refusal that replaces it.

        The latitude and longitude are the ones the ENGINE wrote into
        `mesh.ply` -- `asin(z/r)` and `atan2(y, x)` over the face centroid,
        `src/cpp/main/ptm_thermal_sim.cpp:13636` -- so this converts nothing and
        assumes nothing about the frame. That is planetocentric latitude and
        east-positive longitude, the IAU convention, and the readout says so on
        screen rather than only in a document.

        ⛔ It NEVER falls back to a number. The previous status line read
        `self.mesh['latitude'][cell_id] if 'latitude' in ... else 0`, so a mesh
        without the arrays reported (0.00, 0.00) -- a perfectly plausible
        coordinate in the Gulf of Guinea.
        """
        frame = self._mesh_frame
        if frame is None or not frame.usable:
            return self.COORD_READOUT_NO_FRAME
        if self.mesh is None or face_id is None or face_id < 0:
            return self.COORD_READOUT_OFF_MESH
        names = self.mesh.array_names
        if 'latitude' not in names or 'longitude' not in names:
            return ("lat / lon: unavailable \u2014 this mesh carries no "
                    "per-face coordinates")
        try:
            lat = float(self.mesh['latitude'][face_id])
            lon = float(self.mesh['longitude'][face_id])
        except Exception:                                       # noqa: BLE001
            return self.COORD_READOUT_OFF_MESH
        if not (np.isfinite(lat) and np.isfinite(lon)):
            return self.COORD_READOUT_OFF_MESH
        hemi = 'N' if lat >= 0 else 'S'
        return (f"lat {abs(lat):8.4f}\u00b0 {hemi}   "
                f"lon {lon % 360.0:8.4f}\u00b0 E   "
                f"\u00b7 {frame.label}  \u00b7 facet {face_id}")

    def _pick_face_at(self, x, y):
        """Cell id of the results mesh under screen position (x, y), or None.

        ⛔ `picker.GetCellId()` is an index into whatever dataset the ray hit,
        which is not necessarily the results mesh: the Sun label and any other
        pickable actor in the scene have their own cells, and reading a face
        latitude at another dataset's cell index is a wrong coordinate with no
        symptom. The dataset is checked.
        """
        picker = self._hover_picker
        if picker is None or self.mesh is None:
            return None
        if not getattr(self, 'plotter', None):
            return None
        picker.Pick(x, y, 0, self.plotter.renderer)
        cell_id = picker.GetCellId()
        if cell_id is None or not (0 <= cell_id < self.mesh.n_cells):
            return None
        dataset = picker.GetDataSet()
        if dataset is None:
            return None
        try:
            if dataset.GetNumberOfCells() != self.mesh.n_cells:
                return None
        except Exception:                                       # noqa: BLE001
            return None
        return int(cell_id)

    def enable_hover_picking(self):
        """Enable hover-based face picking with highlighting"""
        if not PYVISTA_AVAILABLE or not self.mesh:
            return

        try:
            # The observers are bound whenever a mesh is loaded, because the
            # coordinate readout follows the pointer with or without selection
            # mode. This only arms the highlight-and-select half.
            if not self._interactor_observers:
                self._bind_interactor_observers()
            self.picking_active = bool(self._interactor_observers)
            self.picking_enabled = self.picking_active
            if self.picking_active:
                # The readout follows the pointer with selection mode off, so
                # `hover_face_id` already names the facet under it. Clearing it
                # makes the next move redraw, rather than take the "same facet,
                # do nothing" path and leave the user with no highlight until
                # they cross a facet boundary.
                self.hover_face_id = None
            else:
                print("Hover picking setup failed: no pointer observers could "
                      "be attached")

        except Exception as e:
            print(f"Hover picking setup failed: {e}")
            self.picking_active = False

    def disable_hover_picking(self):
        """Disable hover picking mode"""
        if not PYVISTA_AVAILABLE:
            return

        try:
            # The observers STAY bound: the coordinate readout is not part of
            # selection mode and keeps following the pointer. Only the
            # highlight-and-select behavior is switched off.
            self.picking_active = False
            self.clear_hover_highlight()
        except Exception as e:
            print(f"Error disabling hover picking: {e}")

    def on_mouse_move(self, obj, event):
        """Follow the pointer: update the coordinate readout, and -- only in
        selection mode -- the hover highlight.

        The readout is NOT gated on `picking_active`. It is a coordinate
        display, not a selection tool, and it has to change as the pointer moves
        over the mesh whether or not the user is selecting facets.
        """
        if not self.mesh:
            return

        try:
            iren = self.plotter.iren.interactor
            x, y = iren.GetEventPosition()
            cell_id = self._pick_face_at(x, y)

            if cell_id is not None and cell_id != self.hover_face_id:
                self.hover_face_id = cell_id
                label = getattr(self, 'coord_readout_label', None)
                if label is not None:
                    label.setText(self.face_coordinate_text(cell_id))
                if self.picking_active:
                    self.update_hover_highlight(cell_id)
                    self.pick_status_label.setText(
                        f"Face {cell_id} - Click to select")

            elif cell_id is None and self.hover_face_id is not None:
                self.hover_face_id = None
                self._reset_coordinate_readout()
                if self.picking_active:
                    self.clear_hover_highlight()
                    self.pick_status_label.setText(
                        "Hover to highlight, click to select")

        except Exception as e:                                  # noqa: BLE001
            # A mouse-move handler must not raise -- PyQt6 aborts the process on
            # an exception out of a slot -- but it must not be silent either: a
            # dead readout looks exactly like a pointer that is off the mesh.
            # Reported once, then suppressed so a moving pointer cannot flood
            # the console.
            if not getattr(self, '_hover_error_reported', False):
                self._hover_error_reported = True
                print(f"[Readout] the pointer handler failed and the "
                      f"coordinate readout is now stale: {e}")

    def on_left_click(self, obj, event):
        """Handle left click for face selection"""
        if not self.picking_active or not self.mesh:
            return

        try:
            iren = self.plotter.iren.interactor
            x, y = iren.GetEventPosition()
            # The same locator-backed picker the readout uses. The previous
            # code built a fresh brute-force `vtkCellPicker` per click, which
            # measured 537 ms on the 2,397,890-facet reference run and landed
            # 3.6 m from the ray's hit point; this one is 0.054 ms and 1.1 m.
            cell_id = self._pick_face_at(x, y)

            if cell_id is not None:
                self.selected_face_id = cell_id
                self._emit_face_selection(cell_id)
                # Keep hover highlight but add selection indicator
                self.update_selection_highlight(cell_id)

        except Exception as e:
            print(f"Click selection error: {e}")

    def _single_face_polydata(self, face_id):
        """One facet as its own tiny mesh, or None.

        O(1) in the size of the results mesh. `extract_cells` is O(N) and this
        is called for every pointer move.
        """
        if self.mesh_data is None:
            return None
        faces = self.mesh_data['faces']
        if face_id is None or face_id < 0 or face_id >= len(faces):
            return None
        points = np.asarray(
            self.mesh_data['vertices'][faces[face_id]], dtype=np.float64)
        return pv.PolyData(points, np.array([3, 0, 1, 2], dtype=np.int64))

    def update_hover_highlight(self, face_id):
        """Update the hover highlight to show a specific face"""
        if not PYVISTA_AVAILABLE or not self.mesh or face_id is None:
            return

        try:
            # Remove old hover highlight if exists
            if self.hover_highlight_actor is not None:
                self.plotter.remove_actor(self.hover_highlight_actor, render=False)
                self.hover_highlight_actor = None

            # The one triangle, built from its own three vertices.
            # ⛔ NOT `self.mesh.extract_cells([face_id])`: that walks the whole
            # dataset and measured 10.3 ms on the 2,397,890-facet reference run
            # against 2.2 ms here, on a path that runs for every pointer move.
            # Same three points, same triangle.
            cell = self._single_face_polydata(face_id)
            if cell is not None and cell.n_cells > 0:
                # Add as wireframe with bright color
                self.hover_highlight_actor = self.plotter.add_mesh(
                    cell,
                    color='cyan',
                    style='wireframe',
                    line_width=3,
                    render=False,
                    name='hover_highlight',
                    pickable=False
                )
                self.plotter.render()

        except Exception as e:
            print(f"Hover highlight error: {e}")

    def clear_hover_highlight(self):
        """Clear the hover highlight"""
        if not PYVISTA_AVAILABLE:
            return

        try:
            if self.hover_highlight_actor is not None:
                self.plotter.remove_actor(self.hover_highlight_actor, render=False)
                self.hover_highlight_actor = None
                self.plotter.render()
        except Exception:
            pass

    def update_selection_highlight(self, face_id):
        """Update the permanent selection highlight"""
        if not PYVISTA_AVAILABLE or not self.mesh or face_id is None:
            return

        try:
            # Remove old selection marker
            try:
                self.plotter.remove_actor('selected_face_marker', render=False)
            except Exception:
                pass

            # The same one-triangle build the hover highlight uses, for the
            # same reason: `extract_cells` walks the whole dataset.
            cell = self._single_face_polydata(face_id)
            if cell is not None and cell.n_cells > 0:
                # Add solid highlight with different color from hover
                self.plotter.add_mesh(
                    cell,
                    color='yellow',
                    style='surface',
                    opacity=0.7,
                    render=False,
                    name='selected_face_marker',
                    pickable=False
                )

                # Also add face center marker -- the centroid of the three
                # points already built above, rather than a second lookup.
                self.plotter.add_points(
                    np.asarray([np.asarray(cell.points).mean(axis=0)]),
                    color='yellow',
                    point_size=12,
                    render=False,
                    name='selected_center_marker'
                )

                self.plotter.render()

        except Exception as e:
            print(f"Selection highlight error: {e}")

    def enable_face_picking(self):
        """Enable face picking on the mesh via left-click (legacy method)"""
        # Redirect to hover-based picking
        self.enable_hover_picking()

    def disable_face_picking(self):
        """Disable face picking mode (legacy method)"""
        self.disable_hover_picking()

    def on_fixed_scale_changed(self, state):
        """Handle fixed scale checkbox change"""
        self.fixed_color_scale = (state == Qt.CheckState.Checked.value)

        # Show/hide manual range inputs. These four widgets only exist when
        # the 3D canvas was built (see `init_ui`), so they are reached the same
        # guarded way as every other canvas-only control in this module --
        # which also lets a headless test drive the state transition below.
        show_inputs = not self.fixed_color_scale
        for name in ('scale_min_label', 'scale_min_input',
                     'scale_max_label', 'scale_max_input'):
            widget = getattr(self, name, None)
            if widget is not None:
                widget.setVisible(show_inputs)

        # If switching to manual, populate with the current range -- AND adopt
        # it as the manual range. Seeding only the spin boxes left
        # `color_range_min/max` at None, so the boxes displayed numbers the
        # view was not using and `update_display` fell through to PyVista's
        # autoscale until the user happened to edit one of them.
        if show_inputs:
            self._adopt_manual_range()

        # Update display
        self.update_display()

    def _adopt_manual_range(self):
        """Seed the manual range and the spin boxes from the CURRENT field.

        ⛔ A MANUAL RANGE BELONGS TO THE FIELD IT WAS TAKEN FROM. Without this
        being re-run on a field switch, a range adopted on one field silently
        governs the next one: untick "Fixed Scale" on Temperature (adopting,
        say, 24-369 K) and then select Shadow Depth, and every real shadow
        depth -- 0 to 5 m -- collapses onto the bottom color while every +inf
        is drawn above-range. That is exactly the flat-color misread the
        sentinel treatment exists to prevent, reached the long way round. It
        bites the dimensionless fields hardest: an albedo map spanning 0.14 to
        0.29 under a temperature range is one flat color.

        The spin boxes are read BACK after being set, and that is deliberate:
        they have finite decimals and a finite range, so the number the widget
        holds is not always the number handed in. Adopting what the widget
        shows keeps the one invariant that matters -- the boxes say what the
        view is using -- even where the widget cannot represent the data
        exactly.
        """
        data_range = self._global_range()
        if data_range is None:
            self.color_range_min = None
            self.color_range_max = None
            return
        adopted = []
        for name, value in (('scale_min_input', data_range[0]),
                            ('scale_max_input', data_range[1])):
            widget = getattr(self, name, None)
            if widget is not None:
                widget.blockSignals(True)
                widget.setValue(value)
                widget.blockSignals(False)
                value = widget.value()
            adopted.append(value)
        self.color_range_min, self.color_range_max = adopted

    def on_color_range_changed(self):
        """Handle manual color range change"""
        if not self.fixed_color_scale:
            self.color_range_min = self.scale_min_input.value()
            self.color_range_max = self.scale_max_input.value()
            self.update_display()

    def compute_global_data_range(self):
        """Color range for the current field, over FINITE values only.

        Per-step series: sampled across timesteps, as before. Statistics: the
        single file, which previously fell through to None and left the color
        bar to VTK's own autoscale -- identical numbers for an all-finite
        field, and the difference that matters for one that is not.

        ⛔ FINITE ONLY, and that is the whole point. `np.min`/`np.max` over a
        `shadow_depth` array return `(0.0, +inf)`, because +inf is the engine's
        "still umbral at the search ceiling" sentinel and inside a permanently
        shadowed region it is the majority value by design. Handing that to
        PyVista as a `clim` paints every facet one flat color -- the sunlit
        ones and the deeply shadowed ones alike. See the module header.
        """
        if not self.results_dir:
            return None

        field_name = self.current_field
        ceiling_m = shadow_depth_ceiling_m(self.results_dir)

        # Statistics: one file, no timestep index.
        if VisualizationPanel.is_stats_field(field_name):
            stats_path = self.results_dir / 'statistics' / f"{field_name}.bin"
            if not stats_path.is_file():
                return None
            data = self.read_binary_data(
                stats_path, self.mesh.n_cells if self.mesh else 0)
            data_range = finite_data_range(data)
            if field_name in SHADOW_DEPTH_FIELDS:
                return shadow_depth_clim(data_range, ceiling_m)
            return data_range

        field_dirs = {
            'temperature': 'temperature',
            'direct_solar': 'direct_solar',
            'illumination': 'illumination',
            'primary_reflected': 'primary_reflected',
            'primary_ir': 'primary_ir'
        }

        dir_name = field_dirs.get(field_name, field_name)
        data_dir = self.results_dir / dir_name

        if not data_dir.exists():
            return None

        global_min = float('inf')
        global_max = float('-inf')

        # Sample a subset of files to compute range (for performance)
        files = sorted(data_dir.glob(f"{dir_name}_*.bin"))
        # Sample every Nth file if there are many
        step = max(1, len(files) // 10)
        sampled_files = files[::step]

        for file_path in sampled_files:
            data = self.read_binary_data(file_path, self.mesh.n_cells if self.mesh else 0)
            file_range = finite_data_range(data)
            if file_range is not None:
                global_min = min(global_min, file_range[0])
                global_max = max(global_max, file_range[1])

        data_range = None
        if global_min < float('inf') and global_max > float('-inf'):
            data_range = (global_min, global_max)
        if field_name in SHADOW_DEPTH_FIELDS:
            return shadow_depth_clim(data_range, ceiling_m)
        return data_range

    def effective_clim(self):
        """The color range this view will hand PyVista, or ``None``.

        Fixed scale: the global range across timesteps. Manual scale: the two
        spin-box values, when both are set. Extracted from `update_display` so
        the decision can be driven without a render window -- it is the one
        place a sentinel can silently get back onto an autoscaled bar.

        ⛔ A SENTINEL-BEARING FIELD NEVER RETURNS None HERE. With `clim=None`
        PyVista autoscales to `(nanmin, nanmax)`, which for a `shadow_depth`
        array is `(0, +inf)`. Measured through the real `vtkLookupTable` at
        that range: 0.0, 0.5, 2.5, 5.0 AND +inf all map to the SAME bottom
        color -- the whole field goes flat and the sentinel becomes
        indistinguishable from "lit at the surface", which is its opposite.
        That was reachable by unticking "Fixed Scale" before touching either
        spin box.
        """
        if self.fixed_color_scale:
            return self._global_range()
        if self.color_range_min is not None and self.color_range_max is not None:
            return (self.color_range_min, self.color_range_max)
        if self.current_field in NAN_BEARING_FIELDS:
            return self._global_range()
        return None

    def _global_range(self):
        """`compute_global_data_range`, computed once per field.

        Cached in `global_data_range`, which `set_field` clears -- so switching
        field recomputes and redrawing does not. Without this the sentinel
        branch above re-read a 9.6-million-byte statistics file on every frame
        of an animation.
        """
        if self.global_data_range is None:
            self.global_data_range = self.compute_global_data_range()
        return self.global_data_range

    def get_field_label(self):
        """Color-bar title for the current field.

        `VisualizationPanel.FIELD_INFO` is the one catalog; this used to carry
        a five-entry copy of the per-step half of it, so every statistic and
        every product added since -- ice_depth, the shadow-depth family -- was
        titled with its raw file-name key. The catalog's strings for those five
        are character-identical to the copy this replaces, so no bar that was
        already labeled changes. The raw key is still the fallback for a field
        no catalog knows, which is honest: it names the file.
        """
        info = VisualizationPanel.FIELD_INFO.get(self.current_field)
        if info is not None:
            return info[0]
        return self.current_field

    def set_timestep(self, index):
        """Set current timestep"""
        if 0 <= index < len(self.timesteps):
            self.load_timestep_data(index)

    def set_field(self, field_name):
        """Set current data field"""
        self.current_field = field_name
        # Reset global range when field changes
        self.global_data_range = None
        # ...and re-adopt the manual range, which is per-field. See
        # `_adopt_manual_range`: carrying one field's range onto the next is
        # how a 0-5 m shadow depth ends up drawn on a 24-369 K bar.
        if not self.fixed_color_scale:
            self._adopt_manual_range()
        if self.mesh and self.results_dir:
            self.load_timestep_data(self.current_timestep)

    def get_effective_cmap(self, field_name=None):
        """Return the colormap currently used for `field_name` (or current).
        Mirrors the resolution order in update_display(): user override,
        FIELD_INFO default, then 'plasma' for temperature-ish fields,
        else 'viridis'."""
        if field_name is None:
            field_name = self.current_field
        override = self.cmap_overrides.get(field_name)
        if override:
            return override
        info = VisualizationPanel.FIELD_INFO.get(field_name)
        if info is not None:
            return info[1]
        if isinstance(field_name, str) and 'temperature' in field_name.lower():
            return 'plasma'
        return 'viridis'

    # ---------- Synthetic ("sun") lighting -------------------------------

    def _set_default_sun_from_latitude(self, latitudes):
        """Set sun_azimuth/elevation defaults from the mesh's mean
        latitude, using the typical-noon-sun-elevation rule
        elevation = max(5°, 90° − |mean_lat|). Only applied on first
        mesh load per session — subsequent loads keep the user's
        previously chosen sun position so panning between datasets
        doesn't reset it."""
        if latitudes is None or len(latitudes) == 0:
            return
        mean_lat = float(np.mean(latitudes))
        self.mesh_mean_lat_deg = mean_lat
        if not getattr(self, '_sun_defaults_set', False):
            self.sun_elevation_deg = max(5.0, 90.0 - abs(mean_lat))
            # Sun is on the equator side of the observer's meridian at
            # local noon. Northern-hemisphere observer sees sun toward
            # the south; southern-hemisphere observer sees it toward
            # the north.
            self.sun_azimuth_deg = 180.0 if mean_lat > 0 else 0.0
            self._sun_defaults_set = True

    def set_synthetic_lighting(self, enabled, azimuth_deg=None,
                               elevation_deg=None, intensity=None,
                               ambient=None):
        """Update synthetic-lighting parameters and redraw."""
        self.synthetic_lighting_enabled = bool(enabled)
        if azimuth_deg   is not None: self.sun_azimuth_deg   = float(azimuth_deg)
        if elevation_deg is not None: self.sun_elevation_deg = float(elevation_deg)
        if intensity     is not None: self.sun_intensity     = float(intensity)
        if ambient       is not None: self.ambient_intensity = float(ambient)
        if self.mesh:
            self.update_display()

    def _apply_synthetic_lighting(self):
        """Replace the plotter's lights with a directional sun light + a
        soft ambient fill positioned by (azimuth, elevation) in the
        mesh-centroid local tangent frame. No-op when disabled or when
        no mesh is present."""
        if not PYVISTA_AVAILABLE or not self.plotter or not self.mesh:
            return
        if not self.synthetic_lighting_enabled:
            # Restore the default 3-light kit so the scene reverts to
            # PyVista's normal lighting when the user turns this off.
            try:
                self.plotter.remove_all_lights()
                self.plotter.enable_lightkit()
            except Exception:
                pass
            return

        # Local tangent frame at the mesh centroid (which sits roughly
        # on the body sphere, so centroid normalized = local up).
        try:
            centroid = np.asarray(self.mesh.center, dtype=float)
        except Exception:
            return
        r = np.linalg.norm(centroid)
        if r < 1.0:
            # Mesh centered at body origin — fall back to global +z up.
            up = np.array([0.0, 0.0, 1.0])
            centroid = up * 1.0
        else:
            up = centroid / r

        # Project geographic-north (+z) onto tangent plane → local north.
        # If centroid is near a pole this projection vanishes, so fall
        # back to a perpendicular direction.
        n_geo = np.array([0.0, 0.0, 1.0])
        north = n_geo - np.dot(n_geo, up) * up
        if np.linalg.norm(north) < 1e-6:
            north = np.array([1.0, 0.0, 0.0]) - np.dot([1.0, 0.0, 0.0], up) * up
        north = north / np.linalg.norm(north)
        east = np.cross(up, north)
        east = east / max(np.linalg.norm(east), 1e-12)

        az = np.deg2rad(self.sun_azimuth_deg)
        el = np.deg2rad(self.sun_elevation_deg)
        # Direction TOWARD the sun, in world coords:
        sun_dir = (np.cos(el) * (np.cos(az) * north + np.sin(az) * east)
                   + np.sin(el) * up)

        # Place light far enough that VTK treats it as effectively
        # parallel rays across the mesh.
        try:
            bounds = self.mesh.bounds
            mesh_extent = max(bounds[1]-bounds[0],
                              bounds[3]-bounds[2],
                              bounds[5]-bounds[4])
        except Exception:
            mesh_extent = r
        D = max(mesh_extent * 50.0, 1.0)
        light_pos = centroid + D * sun_dir
        focal = centroid

        try:
            self.plotter.remove_all_lights()
            sun = pv.Light(position=tuple(light_pos),
                           focal_point=tuple(focal),
                           color='white',
                           intensity=self.sun_intensity)
            sun.light_type = 'scene light'
            sun.positional = False
            self.plotter.add_light(sun)
            # Soft ambient fill so the night side isn't pure black.
            ambient_pos = centroid - D * up
            fill = pv.Light(position=tuple(ambient_pos),
                            focal_point=tuple(focal),
                            color='white',
                            intensity=self.ambient_intensity)
            fill.light_type = 'scene light'
            self.plotter.add_light(fill)
        except Exception as e:
            print(f"[Lighting] failed to apply synthetic lights: {e}")

    def set_cmap_override(self, field_name, cmap):
        """Override the colormap for `field_name`. Pass cmap=None or '' to
        clear the override (revert to FIELD_INFO default). Triggers a
        redraw if the override applies to the currently displayed field."""
        if not field_name:
            return
        if cmap:
            self.cmap_overrides[field_name] = cmap
        else:
            self.cmap_overrides.pop(field_name, None)
        if field_name == self.current_field and self.mesh:
            self.update_display()

    def capture_camera_state(self):
        """Every field of the current view, as a plain dict, or ``None``.

        A camera is more than a position: ``plotter.reset_camera()`` refits the
        DISTANCE to the visible bounds and leaves direction, roll, view-up,
        view angle and parallel scale exactly where the user last dragged
        them. Restoring a view therefore needs all of it, not a distance.

        Read through ``plotter.renderer.camera``, never ``plotter.camera``:
        the latter's getter RESETS an unset camera to PyVista's own default
        as a side effect of being read (``plotting/plotter.py`` ~1623), so
        merely measuring a fresh plotter would move it.
        """
        if not PYVISTA_AVAILABLE or not getattr(self, 'plotter', None):
            return None
        try:
            cam = self.plotter.renderer.camera
            return {
                'position': tuple(float(v) for v in cam.position),
                'focal_point': tuple(float(v) for v in cam.focal_point),
                'view_up': tuple(float(v) for v in cam.up),
                'view_angle': float(cam.view_angle),
                'parallel_projection': bool(cam.parallel_projection),
                'parallel_scale': float(cam.parallel_scale),
            }
        except Exception as e:
            # Never raise out of a Qt slot: PyQt6 aborts the process.
            print(f"[Camera] could not capture the current view: {e}")
            return None

    def apply_camera_state(self, state):
        """Put the camera back exactly where `capture_camera_state` found it.

        Returns True if the view was applied. The clipping range is NOT
        restored from the capture -- it is re-derived from the actors that are
        on screen now, because a stale near/far pair clips away a mesh that is
        otherwise correctly framed.
        """
        if not PYVISTA_AVAILABLE or not getattr(self, 'plotter', None):
            return False
        if not state:
            return False
        try:
            self.plotter.camera_position = [
                state['position'], state['focal_point'], state['view_up']]
            cam = self.plotter.renderer.camera
            cam.view_angle = state['view_angle']
            cam.parallel_projection = state['parallel_projection']
            cam.parallel_scale = state['parallel_scale']
            self.plotter.renderer.reset_camera_clipping_range()
            self.plotter.render()
            return True
        except Exception as e:
            print(f"[Camera] could not restore the view: {e}")
            return False

    def reset_camera(self):
        """Return to the view the results opened with.

        This is the "Reset View" the user presses, and Paul's specification for
        it is literal: "it should reset to the same view as when opening the
        results". It used to call `plotter.reset_camera()`, which is a
        different operation -- that refits the camera DISTANCE to the visible
        bounds and keeps whatever direction and roll the view was orbited to.
        Measured on the 2,397,890-facet reference run: after a 35 deg azimuth /
        12 deg elevation orbit, `plotter.reset_camera()` returned a camera
        10.0 km away from the opening camera, still tilted, and covering 25.4%
        of the canvas against the opening view's 52.7%.

        The fallbacks below announce themselves rather than quietly doing
        something else: a Reset View that silently reframes is the defect this
        replaces.
        """
        # `getattr`, not `self.plotter`: this is a Qt slot, and PyQt6 aborts
        # the whole process on an unhandled exception inside one.
        if not PYVISTA_AVAILABLE or not getattr(self, 'plotter', None):
            return
        state = getattr(self, '_home_camera_state', None)
        if state is not None and self.apply_camera_state(state):
            return
        if self.mesh is not None:
            print("[Camera] Reset View: no opening view had been recorded, so "
                  "the default view for this mesh is being rebuilt instead.")
            self.set_hemisphere_camera()
            self._home_camera_state = self.capture_camera_state()
            return
        print("[Camera] Reset View: no mesh is loaded, so the view was only "
              "refit to whatever is on screen.")
        self.plotter.reset_camera()

    def export_image(self, filename):
        """Export current view as image"""
        if PYVISTA_AVAILABLE and self.plotter:
            self.plotter.screenshot(filename)

    def export_animation(self, filename, format_type="gif", fps=5, progress_callback=None):
        """Export animation of all timesteps.

        Args:
            filename: Output file path
            format_type: "gif" or "mp4"
            fps: Frames per second
            progress_callback: Optional callback(current, total) for progress updates

        Returns:
            True if successful, False otherwise
        """
        if not PYVISTA_AVAILABLE or not self.mesh or not self.results_dir:
            return False

        n_timesteps = len(self.timesteps)
        if n_timesteps == 0:
            return False

        try:
            # Store current timestep to restore later
            original_timestep = self.current_timestep

            # Store camera position to maintain consistent view
            camera_position = self.plotter.camera_position
            # The FULL view, for putting the canvas back afterwards. The
            # triplet above is enough to hold the framing steady between
            # frames, but not to restore it after `_reinit_plotter` replaces
            # the plotter: the replacement starts on PyVista's default camera,
            # which is what left the canvas dark.
            camera_state = self.capture_camera_state()

            if format_type.lower() == "gif":
                # Use PyVista's GIF export
                self.plotter.open_gif(filename)
            else:
                # Use PyVista's movie export (requires ffmpeg)
                self.plotter.open_movie(filename, framerate=fps)

            # Iterate through all timesteps
            for i in range(n_timesteps):
                # Load and display timestep
                self.load_timestep_data(i)

                # Restore camera position to ensure consistent framing
                self.plotter.camera_position = camera_position

                # Ensure complete render before capturing frame
                self.plotter.render()

                # Write frame
                self.plotter.write_frame()

                # Report progress
                if progress_callback:
                    progress_callback(i + 1, n_timesteps)

            # Mute the plotter before closing to prevent any additional frame capture
            # by ensuring render state is stable
            self.plotter.render()

            # Close the animation file
            self.plotter.mwriter.close()  # Close writer directly to avoid extra frame

            # Reinitialize plotter after close
            self._reinit_plotter()

            # Restore original timestep. `_reinit_plotter` has cleared
            # `_has_loaded_data`, so this redraw frames the new plotter on the
            # mesh and re-records the home view instead of inheriting the
            # empty plotter's default camera.
            self.load_timestep_data(original_timestep)

            # Put the user's own framing back. Without this the canvas returns
            # to the opening view rather than the view the animation was shot
            # from, which is still correct but is not what the user left.
            if camera_state is not None and not self.apply_camera_state(camera_state):
                print("[Animation] the view before the export could not be "
                      "restored; the canvas is showing the default view for "
                      "this mesh.")

            return True

        except Exception as e:
            print(f"Animation export error: {e}")
            # Try to recover
            try:
                # Try to close writer if it exists
                if hasattr(self.plotter, 'mwriter') and self.plotter.mwriter:
                    try:
                        self.plotter.mwriter.close()
                    except Exception:
                        pass
                self._reinit_plotter()
                self.load_timestep_data(original_timestep if 'original_timestep' in dir() else 0)
                # Same reason as the success path: a replaced plotter starts on
                # PyVista's default camera, and a failed export must not leave
                # the canvas dark either.
                if 'camera_state' in dir() and camera_state is not None:
                    self.apply_camera_state(camera_state)
            except Exception:
                pass
            return False

    def _reinit_plotter(self):
        """Reinitialize the plotter after closing for animation export.

        ⛔ The replacement plotter is a DIFFERENT object with an empty renderer
        and PyVista's default camera -- position (0, 0, 1), focal point
        (0, 0, 0), view angle 30 deg. `update_display` decides whether to keep
        the current camera from `_has_loaded_data`, which is a property of the
        WIDGET, not of the plotter; left set, the first redraw after this
        method read the new plotter's default camera as "the user's view" and
        re-imposed it over the framing `add_mesh` had just computed. On the
        2,397,890-facet reference run that put the mesh 1,734 km away inside a
        1,239 km-wide field of view -- 1.8% of the canvas lit against 52.7%
        before the export, which is the "goes dark" the user reported.
        Clearing the flag here is the fix: the next redraw reframes on the
        mesh. `export_animation` then restores the user's own pre-export view
        on top of that.
        """
        if not PYVISTA_AVAILABLE:
            return

        # Get the layout
        layout = self.layout()

        # Remove old plotter widget
        if hasattr(self, 'plotter') and self.plotter:
            try:
                retired = self.plotter.interactor
                layout.removeWidget(retired)
                # `QLayout.removeWidget` does not hide the widget and does not
                # reparent it: it stays a visible child of this widget at its
                # last geometry, so without this the retired canvas is still
                # painted underneath the new one. It is deliberately NOT closed
                # or deleted here -- tearing down a live VTK render window is
                # not something this change can test without a display, and a
                # crash on export would be worse than the widget it retires.
                # See PTM-GUI-RETIRED-INTERACTOR-LEAK.
                retired.hide()
            except Exception:
                pass

        # The new plotter has never displayed anything, so the next
        # `update_display` must frame it rather than preserve a camera it
        # never had.
        self._has_loaded_data = False

        # Create new plotter
        self.plotter = QtInteractor(self)
        self.plotter.set_background('#1e1e1e')
        self.plotter.add_axes()

        # Add back to layout (index 1, after controls)
        layout.insertWidget(1, self.plotter.interactor, 1)

        # The new plotter has a DIFFERENT `iren`. Without this the hover
        # observers stay on the retired one, the coordinate readout freezes and
        # face selection stops working, while `picking_active`,
        # `picking_enabled` and the "Select Face Mode" button all still read as
        # on (PTM-GUI-PICKING-DEAD-AFTER-EXPORT).
        self._bind_interactor_observers()
        self.hover_face_id = None
        self.hover_highlight_actor = None
        self._reset_coordinate_readout()

    def find_face_by_latlon(self, lat, lon):
        """Find the nearest face to given lat/lon coordinates"""
        if self.mesh_data is None:
            return None

        lats = self.mesh_data['latitudes']
        lons = self.mesh_data['longitudes']

        # Calculate angular distance (simple approximation)
        dlat = lats - lat
        dlon = lons - lon
        # Handle longitude wraparound
        dlon = np.where(dlon > 180, dlon - 360, dlon)
        dlon = np.where(dlon < -180, dlon + 360, dlon)

        dist = np.sqrt(dlat**2 + dlon**2)
        face_id = int(np.argmin(dist))

        return face_id

    def highlight_face(self, face_id):
        """Highlight a specific face on the mesh (called from external sources like 2D plots)"""
        if not PYVISTA_AVAILABLE or not self.mesh or face_id is None:
            return

        # Use the selection highlight method
        self.selected_face_id = face_id
        self.update_selection_highlight(face_id)

    def on_surface_picked(self, point):
        """Handle surface point picking - find the cell containing the picked point"""
        if point is None or self.mesh is None:
            return

        try:
            # Find the cell that contains this point
            cell_id = self.mesh.find_closest_cell(point)
            if cell_id is not None and cell_id >= 0:
                self._emit_face_selection(cell_id)
        except Exception as e:
            print(f"Surface picking error: {e}")

    def on_cell_picked(self, cell):
        """Handle cell (face) picking (rectangle selection mode)"""
        if cell is None:
            return

        # Handle different return types from PyVista versions
        if hasattr(cell, 'GetCellId'):
            face_id = cell.GetCellId()
        elif isinstance(cell, (int, np.integer)):
            face_id = int(cell)
        elif hasattr(cell, 'n_cells') and cell.n_cells > 0:
            # It's a mesh subset - find the original cell ID
            # PyVista returns the picked cell as a new mesh
            face_id = self.find_cell_in_mesh(cell)
        else:
            print(f"Unknown cell type: {type(cell)}")
            return

        if face_id is None or face_id < 0:
            return

        self._emit_face_selection(face_id)

    def on_click_position(self, point):
        """Handle click position fallback for face selection"""
        if point is None or self.mesh is None:
            return

        # Find closest cell to click point
        try:
            cell_id = self.mesh.find_closest_cell(point)
            if cell_id >= 0:
                self._emit_face_selection(cell_id)
        except Exception as e:
            print(f"Click position handling: {e}")

    def find_cell_in_mesh(self, picked_cell):
        """Find cell ID of picked cell in original mesh"""
        if picked_cell is None or self.mesh is None:
            return None

        try:
            # Get center of picked cell
            center = picked_cell.cell_centers().points[0]
            # Find closest cell in original mesh
            return self.mesh.find_closest_cell(center)
        except Exception:
            return None

    def _emit_face_selection(self, face_id):
        """Emit face selection signal with data"""
        if face_id < 0 or face_id >= self.mesh.n_cells:
            return

        # Gather data for this face
        data = {
            'face_id': face_id,
            'latitude': float(self.mesh['latitude'][face_id]) if 'latitude' in self.mesh.array_names else None,
            'longitude': float(self.mesh['longitude'][face_id]) if 'longitude' in self.mesh.array_names else None
        }

        # Add current field value
        if self.current_field in self.mesh.array_names:
            data[self.current_field] = float(self.mesh[self.current_field][face_id])

        # Highlight the selected face
        self.highlight_face(face_id)

        self.face_selected.emit(face_id, data)

    def refresh(self):
        """Refresh the display"""
        self.update_display()

    def cleanup(self):
        """Clean up resources"""
        if PYVISTA_AVAILABLE and hasattr(self, 'plotter'):
            self.plotter.close()


class PlotWidget(QWidget):
    """2D plotting widget using Matplotlib with face selection and export"""

    face_selection_requested = pyqtSignal(int)  # Request to highlight face in 3D

    def __init__(self, parent=None):
        super().__init__(parent)
        self.results_dir = None
        self.timesteps = []
        self.timestep_labels = []
        self.current_face = None
        self.current_face_lat = None
        self.current_face_lon = None
        self.mesh_data = None
        self.current_plot_data = {}  # Store data for export
        self.current_timestep = 0  # Track current timestep for distribution plots

        # Time axis settings
        self.use_time_axis = False  # False = timestep index, True = actual time
        self.time_unit = "s"  # Current time unit: s, min, hr, d
        self.time_unit_label = "Time (s)"  # Label for x-axis
        self.time_scale = 1.0  # Multiplier to convert timestep values to display unit

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        if not MATPLOTLIB_AVAILABLE:
            label = QLabel("Matplotlib not available.\nInstall with: pip install matplotlib")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet("color: #ff9800; font-size: 14px;")
            layout.addWidget(label)
            return

        # === Face Selection Controls ===
        selection_group = QGroupBox("Location Selection")
        selection_layout = QGridLayout()

        # Lat/Lon input
        selection_layout.addWidget(QLabel("Latitude:"), 0, 0)
        self.lat_input = QDoubleSpinBox()
        self.lat_input.setRange(-90.0, 90.0)
        self.lat_input.setDecimals(4)
        self.lat_input.setValue(0.0)
        self.lat_input.setSuffix("°")
        selection_layout.addWidget(self.lat_input, 0, 1)

        selection_layout.addWidget(QLabel("Longitude:"), 0, 2)
        self.lon_input = QDoubleSpinBox()
        self.lon_input.setRange(-180.0, 360.0)
        self.lon_input.setDecimals(4)
        self.lon_input.setValue(0.0)
        self.lon_input.setSuffix("°")
        selection_layout.addWidget(self.lon_input, 0, 3)

        self.find_btn = QPushButton("Find Location")
        self.find_btn.clicked.connect(self.on_find_location)
        selection_layout.addWidget(self.find_btn, 0, 4)

        # Face ID display
        selection_layout.addWidget(QLabel("Face ID:"), 1, 0)
        self.face_id_label = QLabel("--")
        self.face_id_label.setStyleSheet("color: #2196f3; font-weight: bold;")
        selection_layout.addWidget(self.face_id_label, 1, 1)

        # Actual lat/lon of selected face
        selection_layout.addWidget(QLabel("Actual:"), 1, 2)
        self.actual_latlon_label = QLabel("--")
        self.actual_latlon_label.setStyleSheet("color: #4caf50;")
        selection_layout.addWidget(self.actual_latlon_label, 1, 3, 1, 2)

        selection_group.setLayout(selection_layout)
        layout.addWidget(selection_group)

        # === Plot Type Selection ===
        plot_control_layout = QHBoxLayout()

        plot_control_layout.addWidget(QLabel("Plot Type:"))
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems([
            "Temperature Time Series",
            "All Fields Time Series",
            "Temperature Distribution",
            "Temperature vs Latitude",
            "Global Statistics"
        ])
        self.plot_type_combo.currentTextChanged.connect(self.on_plot_type_changed)
        plot_control_layout.addWidget(self.plot_type_combo)

        # Time axis toggle
        plot_control_layout.addWidget(QLabel("  X-Axis:"))
        self.time_axis_btn = QPushButton("Timestep")
        self.time_axis_btn.setCheckable(True)
        self.time_axis_btn.setMaximumWidth(100)
        self.time_axis_btn.setToolTip("Toggle between timestep index and actual time units")
        self.time_axis_btn.clicked.connect(self.toggle_time_axis)
        plot_control_layout.addWidget(self.time_axis_btn)

        plot_control_layout.addStretch()

        # Export buttons
        self.export_data_btn = QPushButton("Export Data (ASCII)")
        self.export_data_btn.clicked.connect(self.export_data_ascii)
        plot_control_layout.addWidget(self.export_data_btn)

        self.export_plot_btn = QPushButton("Export Plot (PNG)")
        self.export_plot_btn.clicked.connect(self.export_plot_image)
        plot_control_layout.addWidget(self.export_plot_btn)

        layout.addLayout(plot_control_layout)

        # === Matplotlib Figure ===
        self.figure = Figure(figsize=(10, 6), dpi=100)
        self.figure.patch.set_facecolor('#1e1e1e')
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setStyleSheet("background-color: #252526; color: white;")

        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 1)

        # Initialize with default plot
        self.init_empty_plot()

    def init_empty_plot(self):
        """Show empty plot with instructions"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self.style_axis(ax)
        ax.text(0.5, 0.5, "Click on 3D mesh or enter Lat/Lon to select a location",
                ha='center', va='center', color='#888888', fontsize=14,
                transform=ax.transAxes)
        ax.set_title("Temperature Time Series")
        self.canvas.draw()

    def style_axis(self, ax):
        """Apply dark theme styling to axis"""
        ax.set_facecolor('#252526')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_color('#404040')

    def load_results(self, results_dir, mesh_data=None):
        """Load results for plotting"""
        self.results_dir = Path(results_dir)
        self.mesh_data = mesh_data

        # Load timesteps with labels
        time_file = self.results_dir / "time_steps.txt"
        if time_file.exists():
            self.timesteps = []
            self.timestep_labels = []
            with open(time_file, 'r') as f:
                for line in f:
                    if line.startswith('#'):
                        continue
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        self.timesteps.append(float(parts[0]))
                        self.timestep_labels.append(parts[1])

        # Calculate appropriate time unit for this dataset
        self.calculate_time_unit()

        # Reset time axis mode and update button
        self.use_time_axis = False
        if hasattr(self, 'time_axis_btn'):
            self.time_axis_btn.setText("Timestep")
            self.time_axis_btn.setChecked(False)

        # Reset selection
        self.current_face = None
        self.init_empty_plot()

    def on_find_location(self):
        """Find face at specified lat/lon"""
        if self.mesh_data is None:
            QMessageBox.warning(self, "No Data", "Please load simulation results first.")
            return

        lat = self.lat_input.value()
        lon = self.lon_input.value()

        # Find nearest face
        face_id = self.find_nearest_face(lat, lon)
        if face_id is not None:
            self.select_face(face_id)
            self.face_selection_requested.emit(face_id)

    def find_nearest_face(self, lat, lon):
        """Find the nearest face to given lat/lon"""
        if self.mesh_data is None:
            return None

        lats = self.mesh_data['latitudes']
        lons = self.mesh_data['longitudes']

        dlat = lats - lat
        dlon = lons - lon
        dlon = np.where(dlon > 180, dlon - 360, dlon)
        dlon = np.where(dlon < -180, dlon + 360, dlon)

        dist = np.sqrt(dlat**2 + dlon**2)
        return int(np.argmin(dist))

    def select_face(self, face_id, face_data=None):
        """Select a face and update display"""
        self.current_face = face_id

        # Get face lat/lon
        if face_data and 'latitude' in face_data:
            self.current_face_lat = face_data['latitude']
            self.current_face_lon = face_data['longitude']
        elif self.mesh_data is not None and face_id < len(self.mesh_data['latitudes']):
            self.current_face_lat = self.mesh_data['latitudes'][face_id]
            self.current_face_lon = self.mesh_data['longitudes'][face_id]

        # Update UI
        self.face_id_label.setText(str(face_id))
        if self.current_face_lat is not None:
            self.actual_latlon_label.setText(
                f"({self.current_face_lat:.4f}°, {self.current_face_lon:.4f}°)"
            )
            # Update input fields to match
            self.lat_input.blockSignals(True)
            self.lon_input.blockSignals(True)
            self.lat_input.setValue(self.current_face_lat)
            self.lon_input.setValue(self.current_face_lon)
            self.lat_input.blockSignals(False)
            self.lon_input.blockSignals(False)

        # Update plot
        self.update_plot()

    def show_face_timeseries(self, face_id, face_data=None):
        """Show time series for a specific face (called from 3D view)"""
        self.select_face(face_id, face_data)

    def on_plot_type_changed(self, plot_type):
        """Handle plot type change"""
        self.update_plot()

    def toggle_time_axis(self):
        """Toggle between timestep index and actual time units"""
        self.use_time_axis = not self.use_time_axis

        if self.use_time_axis:
            # Calculate and set appropriate time unit
            self.calculate_time_unit()
            self.time_axis_btn.setText(self.time_unit_label.replace("Time ", "").strip("()"))
            self.time_axis_btn.setChecked(True)
        else:
            self.time_axis_btn.setText("Timestep")
            self.time_axis_btn.setChecked(False)

        # Refresh the current plot
        self.update_plot()

    def calculate_time_unit(self):
        """Calculate the best time unit based on timestep interval.

        Automatically selects seconds, minutes, hours, or days based on
        the typical interval between timesteps.
        """
        if len(self.timesteps) < 2:
            # Default to seconds if we can't calculate
            self.time_unit = "s"
            self.time_unit_label = "Time (s)"
            self.time_scale = 1.0
            return

        # Calculate average timestep interval (timesteps are in seconds)
        intervals = []
        for i in range(1, len(self.timesteps)):
            intervals.append(self.timesteps[i] - self.timesteps[i - 1])
        avg_interval = sum(intervals) / len(intervals)

        # Total duration
        total_duration = self.timesteps[-1] - self.timesteps[0]

        # Choose unit based on total duration and interval
        # Goal: have x-axis values in a reasonable range (not too many digits)
        if total_duration < 120:  # Less than 2 minutes
            self.time_unit = "s"
            self.time_unit_label = "Time (s)"
            self.time_scale = 1.0
        elif total_duration < 7200:  # Less than 2 hours
            self.time_unit = "min"
            self.time_unit_label = "Time (min)"
            self.time_scale = 1.0 / 60.0
        elif total_duration < 172800:  # Less than 2 days
            self.time_unit = "hr"
            self.time_unit_label = "Time (hr)"
            self.time_scale = 1.0 / 3600.0
        else:  # 2 days or more
            self.time_unit = "d"
            self.time_unit_label = "Time (days)"
            self.time_scale = 1.0 / 86400.0

    def get_time_axis_values(self, n_points=None):
        """Get x-axis values based on current time axis mode.

        Args:
            n_points: Number of data points (defaults to len(self.timesteps))

        Returns:
            tuple: (x_values, x_label) where x_values is a list/array and x_label is the axis label
        """
        if n_points is None:
            n_points = len(self.timesteps)

        if self.use_time_axis and len(self.timesteps) >= n_points:
            # Use actual time values, scaled to current unit
            # Subtract first timestep to start from 0
            t0 = self.timesteps[0] if self.timesteps else 0
            x_values = [(self.timesteps[i] - t0) * self.time_scale for i in range(n_points)]
            x_label = self.time_unit_label
        else:
            # Use timestep index
            x_values = list(range(n_points))
            x_label = "Timestep"

        return x_values, x_label

    def set_timestep(self, timestep):
        """Set current timestep and update relevant plots"""
        self.current_timestep = timestep
        # Update plots that depend on timestep (distribution and vs latitude)
        plot_type = self.plot_type_combo.currentText() if hasattr(self, 'plot_type_combo') else ""
        if plot_type in ["Temperature Distribution", "Temperature vs Latitude"]:
            self.update_plot()

    def update_plot(self):
        """Update the current plot"""
        if not MATPLOTLIB_AVAILABLE or not self.results_dir:
            return

        plot_type = self.plot_type_combo.currentText()

        if plot_type == "Temperature Time Series":
            self.plot_temperature_timeseries()
        elif plot_type == "All Fields Time Series":
            self.plot_all_fields_timeseries()
        elif plot_type == "Temperature Distribution":
            self.plot_temperature_distribution()
        elif plot_type == "Temperature vs Latitude":
            self.plot_temperature_vs_latitude()
        elif plot_type == "Global Statistics":
            self.plot_global_statistics()

    def plot_temperature_timeseries(self):
        """Plot temperature time series for selected face"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self.style_axis(ax)

        if self.current_face is None:
            ax.text(0.5, 0.5, "Select a location on the 3D mesh\nor enter Lat/Lon coordinates",
                    ha='center', va='center', color='#888888', fontsize=14,
                    transform=ax.transAxes)
            ax.set_title("Temperature Time Series")
            self.canvas.draw()
            return

        temp_dir = self.results_dir / "temperature"
        if not temp_dir.exists():
            ax.text(0.5, 0.5, "Temperature data not found",
                    ha='center', va='center', color='#ff5722', fontsize=14,
                    transform=ax.transAxes)
            self.canvas.draw()
            return

        # Load temperature data for this face
        files = sorted(temp_dir.glob("temperature_*.bin"))
        temps = []

        for i, f in enumerate(files):
            data = self.read_binary_data(f)
            if data is not None and self.current_face < len(data):
                temps.append(data[self.current_face])

        # Get x-axis label (even if no data, for consistent labeling)
        _, x_label = self.get_time_axis_values(len(files) if files else 1)

        if temps:
            # Get x-axis values based on time axis mode
            x_values, x_label = self.get_time_axis_values(len(temps))

            ax.plot(x_values, temps, color='#ff5722', linewidth=2, marker='o',
                    markersize=3, label='Temperature')
            ax.fill_between(x_values, min(temps), temps, alpha=0.3, color='#ff5722')

            # Store for export (always include both timestep and time info)
            self.current_plot_data = {
                'title': f'Temperature Time Series - Face {self.current_face}',
                'columns': ['Timestep', 'Time_Label', f'Time_{self.time_unit}', 'Temperature_K'],
                'data': []
            }
            t0 = self.timesteps[0] if self.timesteps else 0
            for i, temp in enumerate(temps):
                label = self.timestep_labels[i] if i < len(self.timestep_labels) else ""
                time_val = (self.timesteps[i] - t0) * self.time_scale if i < len(self.timesteps) else i
                self.current_plot_data['data'].append([i, label, time_val, temp])

            # Add statistics
            ax.axhline(np.mean(temps), color='#4caf50', linestyle='--',
                       linewidth=1, label=f'Mean: {np.mean(temps):.1f} K')
            ax.axhline(np.max(temps), color='#f44336', linestyle=':',
                       linewidth=1, label=f'Max: {np.max(temps):.1f} K')
            ax.axhline(np.min(temps), color='#2196f3', linestyle=':',
                       linewidth=1, label=f'Min: {np.min(temps):.1f} K')

            ax.legend(loc='upper right', facecolor='#252526', edgecolor='#404040',
                      labelcolor='white')

        title = f"Temperature Time Series - Face {self.current_face}"
        if self.current_face_lat is not None:
            title += f" ({self.current_face_lat:.2f}°, {self.current_face_lon:.2f}°)"
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel('Temperature (K)')
        ax.grid(True, alpha=0.3, color='#404040')

        self.figure.tight_layout()
        self.canvas.draw()

    def plot_all_fields_timeseries(self):
        """Plot all available fields for selected face"""
        self.figure.clear()

        if self.current_face is None:
            ax = self.figure.add_subplot(111)
            self.style_axis(ax)
            ax.text(0.5, 0.5, "Select a location first",
                    ha='center', va='center', color='#888888', fontsize=14,
                    transform=ax.transAxes)
            self.canvas.draw()
            return

        fields = [
            ('temperature', 'Temperature (K)', '#ff5722'),
            ('illumination', 'Illumination (W/m²)', '#4caf50'),
            ('direct_solar', 'Direct Solar (W/m²)', '#ffeb3b'),
            ('primary_reflected', 'Reflected (W/m²)', '#2196f3'),
        ]

        axes = self.figure.subplots(2, 2)
        self.current_plot_data = {
            'title': f'All Fields Time Series - Face {self.current_face}',
            'columns': ['Timestep'],
            'data': []
        }

        # Get x-axis label once for all subplots
        _, x_label = self.get_time_axis_values()

        for ax_idx, (field, label, color) in enumerate(fields):
            ax = axes[ax_idx // 2, ax_idx % 2]
            self.style_axis(ax)

            field_dir = self.results_dir / field
            if field_dir.exists():
                files = sorted(field_dir.glob(f"{field}_*.bin"))
                values = []
                for f in files:
                    data = self.read_binary_data(f)
                    if data is not None and self.current_face < len(data):
                        values.append(data[self.current_face])

                if values:
                    # Get x-axis values based on time axis mode
                    x_values, _ = self.get_time_axis_values(len(values))
                    ax.plot(x_values, values, color=color, linewidth=1.5)
                    ax.fill_between(x_values, 0, values, alpha=0.3, color=color)

            ax.set_title(label, fontsize=10)
            ax.set_xlabel(x_label, fontsize=9)
            ax.grid(True, alpha=0.3, color='#404040')

        self.figure.suptitle(f'Face {self.current_face} ({self.current_face_lat:.2f}°, {self.current_face_lon:.2f}°)',
                             color='white', fontsize=12)
        self.figure.tight_layout()
        self.canvas.draw()

    def plot_temperature_distribution(self):
        """Plot temperature distribution histogram for current timestep"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self.style_axis(ax)

        temp_dir = self.results_dir / "temperature"
        if temp_dir.exists():
            files = sorted(temp_dir.glob("temperature_*.bin"))
            if files:
                # Use current timestep, clamped to valid range
                timestep_idx = min(self.current_timestep, len(files) - 1)
                timestep_idx = max(0, timestep_idx)
                data_file = files[timestep_idx]
                data = self.read_binary_data(data_file)
                if data is not None and len(data) > 0:
                    n, bins, patches = ax.hist(data, bins=50, color='#ff5722',
                                               alpha=0.8, edgecolor='none')

                    # Store for export
                    self.current_plot_data = {
                        'title': f'Temperature Distribution (Timestep {timestep_idx})',
                        'columns': ['Bin_Center_K', 'Count'],
                        'data': []
                    }
                    bin_centers = (bins[:-1] + bins[1:]) / 2
                    for bc, count in zip(bin_centers, n):
                        self.current_plot_data['data'].append([bc, count])

                    # Mark selected face if any
                    if self.current_face is not None and self.current_face < len(data):
                        face_temp = data[self.current_face]
                        ax.axvline(face_temp, color='yellow', linewidth=2,
                                   label=f'Selected: {face_temp:.1f} K')
                        ax.legend(facecolor='#252526', edgecolor='#404040', labelcolor='white')

                    # Show timestep label if available
                    ts_label = ""
                    if timestep_idx < len(self.timestep_labels):
                        ts_label = f" - {self.timestep_labels[timestep_idx]}"
                    ax.set_title(f'Temperature Distribution (Timestep {timestep_idx}{ts_label})')

        ax.set_xlabel('Temperature (K)')
        ax.set_ylabel('Number of Faces')
        ax.grid(True, alpha=0.3, color='#404040')

        self.figure.tight_layout()
        self.canvas.draw()

    def plot_temperature_vs_latitude(self):
        """Plot temperature vs latitude scatter for current timestep"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self.style_axis(ax)

        if self.mesh_data is None:
            ax.text(0.5, 0.5, "Mesh data not loaded",
                    ha='center', va='center', color='#888888', fontsize=14,
                    transform=ax.transAxes)
            self.canvas.draw()
            return

        temp_dir = self.results_dir / "temperature"
        if temp_dir.exists():
            files = sorted(temp_dir.glob("temperature_*.bin"))
            if files:
                # Use current timestep, clamped to valid range
                timestep_idx = min(self.current_timestep, len(files) - 1)
                timestep_idx = max(0, timestep_idx)
                data_file = files[timestep_idx]
                temps = self.read_binary_data(data_file)
                lats = self.mesh_data['latitudes']

                if temps is not None and len(temps) == len(lats):
                    ax.scatter(lats, temps, s=1, alpha=0.3, c='#2196f3')

                    # Store for export (sampled to reduce size)
                    self.current_plot_data = {
                        'title': 'Temperature vs Latitude',
                        'columns': ['Latitude_deg', 'Temperature_K'],
                        'data': []
                    }
                    # Sample every Nth point to keep export reasonable
                    step = max(1, len(lats) // 10000)
                    for i in range(0, len(lats), step):
                        self.current_plot_data['data'].append([lats[i], temps[i]])

                    # Mark selected face
                    if self.current_face is not None and self.current_face < len(temps):
                        ax.scatter([lats[self.current_face]], [temps[self.current_face]],
                                   s=100, c='yellow', marker='*', zorder=10,
                                   label=f'Selected Face {self.current_face}')
                        ax.legend(facecolor='#252526', edgecolor='#404040', labelcolor='white')

                    # Title with timestep info
                    ts_label = ""
                    if timestep_idx < len(self.timestep_labels):
                        ts_label = f" - {self.timestep_labels[timestep_idx]}"
                    ax.set_title(f'Temperature vs Latitude (Timestep {timestep_idx}{ts_label})')

        if not ax.get_title():
            ax.set_title('Temperature vs Latitude')
        ax.set_xlabel('Latitude (degrees)')
        ax.set_ylabel('Temperature (K)')
        ax.grid(True, alpha=0.3, color='#404040')

        self.figure.tight_layout()
        self.canvas.draw()

    def plot_global_statistics(self):
        """Plot global min/max/mean temperature over time"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self.style_axis(ax)

        x_label = "Timestep"  # Default

        temp_dir = self.results_dir / "temperature"
        if temp_dir.exists():
            files = sorted(temp_dir.glob("temperature_*.bin"))
            mins, maxs, means = [], [], []

            for f in files:
                data = self.read_binary_data(f)
                if data is not None and len(data) > 0:
                    mins.append(np.min(data))
                    maxs.append(np.max(data))
                    means.append(np.mean(data))

            if mins:
                # Get x-axis values based on time axis mode
                x_values, x_label = self.get_time_axis_values(len(mins))

                ax.fill_between(x_values, mins, maxs, alpha=0.3, color='#ff5722',
                                label='Min-Max Range')
                ax.plot(x_values, means, color='#ff5722', linewidth=2, label='Mean')
                ax.plot(x_values, mins, color='#2196f3', linewidth=1, linestyle='--', label='Min')
                ax.plot(x_values, maxs, color='#f44336', linewidth=1, linestyle='--', label='Max')

                # Store for export (include both timestep and time)
                self.current_plot_data = {
                    'title': 'Global Temperature Statistics',
                    'columns': ['Timestep', f'Time_{self.time_unit}', 'Min_K', 'Max_K', 'Mean_K'],
                    'data': []
                }
                t0 = self.timesteps[0] if self.timesteps else 0
                for i in range(len(mins)):
                    time_val = (self.timesteps[i] - t0) * self.time_scale if i < len(self.timesteps) else i
                    self.current_plot_data['data'].append([i, time_val, mins[i], maxs[i], means[i]])

                ax.legend(loc='upper right', facecolor='#252526', edgecolor='#404040',
                          labelcolor='white')

        ax.set_title('Global Temperature Statistics')
        ax.set_xlabel(x_label)
        ax.set_ylabel('Temperature (K)')
        ax.grid(True, alpha=0.3, color='#404040')

        self.figure.tight_layout()
        self.canvas.draw()

    def read_binary_data(self, file_path):
        """Read binary data file"""
        try:
            with open(file_path, 'rb') as f:
                header = f.read(4)
                if len(header) == 4:
                    count = struct.unpack('<I', header)[0]
                    f.seek(0, 2)
                    file_size = f.tell()

                    if file_size == 4 + count * 4:
                        f.seek(4)
                        return np.frombuffer(f.read(), dtype='<f4')
                    else:
                        f.seek(0)
                        return np.frombuffer(f.read(), dtype='<f4')

            return None
        except Exception:
            return None

    def export_data_ascii(self):
        """Export current plot data as ASCII file"""
        if not self.current_plot_data or not self.current_plot_data.get('data'):
            QMessageBox.warning(self, "No Data", "No plot data available to export.")
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Data",
            str(Path.home() / "ptm3d_plot_data.dat"),
            "Data Files (*.dat *.txt *.csv);;All Files (*)"
        )

        if not filename:
            return

        try:
            with open(filename, 'w') as f:
                # Write header
                f.write(f"# {self.current_plot_data['title']}\n")
                f.write(f"# Generated by PTM3D GUI\n")
                if self.current_face is not None:
                    f.write(f"# Face ID: {self.current_face}\n")
                    if self.current_face_lat is not None:
                        f.write(f"# Location: ({self.current_face_lat:.6f}, {self.current_face_lon:.6f})\n")
                f.write("#\n")

                # Write column headers
                f.write("# " + "\t".join(self.current_plot_data['columns']) + "\n")

                # Write data
                for row in self.current_plot_data['data']:
                    line = "\t".join(f"{v:.6g}" if isinstance(v, float) else str(v) for v in row)
                    f.write(line + "\n")

            QMessageBox.information(self, "Export Complete",
                                    f"Data exported to:\n{filename}")

        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export data:\n{str(e)}")

    def export_plot_image(self):
        """Export current plot as image"""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Plot",
            str(Path.home() / "ptm3d_plot.png"),
            "PNG Images (*.png);;PDF Files (*.pdf);;SVG Files (*.svg);;All Files (*)"
        )

        if not filename:
            return

        try:
            self.figure.savefig(filename, dpi=150, facecolor='#1e1e1e',
                                edgecolor='none', bbox_inches='tight')
            QMessageBox.information(self, "Export Complete",
                                    f"Plot exported to:\n{filename}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export plot:\n{str(e)}")

    def update_plots(self):
        """Update plots (called when results are loaded)"""
        self.update_plot()


# Match plain decimals OR scientific notation: 0.001, 3.3e-6, 1.5E+02, etc.
# Permissive partial-match (`^...$` not anchored to require the user finish)
# is left to QValidator.Intermediate; we only insist on Acceptable when the
# whole text parses as float.
_SCI_NUMBER_RE = re.compile(
    r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$"
)
_SCI_PARTIAL_RE = re.compile(
    r"^[+-]?(\d+\.?\d*|\.\d+)?([eE][+-]?\d*)?$"
)


class ScientificDoubleSpinBox(QDoubleSpinBox):
    """QDoubleSpinBox that accepts scientific notation in user input
    (e.g. ``3.3e-6``) and renders very small / very large values in ``%g``
    format. Plain decimals like ``0.0000036`` still work.

    Drop-in replacement: same .value()/.setValue() API.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        # 9 decimals lets the spinner step by ``setSingleStep`` even at
        # microdegree scale without round-trip rounding errors.
        self.setDecimals(9)

    def validate(self, text, pos):
        s = text.strip()
        # Strip trailing suffix (set via setSuffix) so the user sees / types
        # "3.3e-6 degrees" but parsing doesn't choke.
        suf = self.suffix()
        if suf and s.endswith(suf.strip()):
            s = s[: -len(suf.strip())].strip()
        if not s:
            return (QValidator.State.Intermediate, text, pos)
        if _SCI_NUMBER_RE.match(s):
            return (QValidator.State.Acceptable, text, pos)
        if _SCI_PARTIAL_RE.match(s):
            return (QValidator.State.Intermediate, text, pos)
        return (QValidator.State.Invalid, text, pos)

    def valueFromText(self, text):
        s = text.strip()
        suf = self.suffix()
        if suf and s.endswith(suf.strip()):
            s = s[: -len(suf.strip())].strip()
        try:
            return float(s)
        except ValueError:
            return self.minimum()

    def textFromValue(self, value):
        # Use general format: switches to scientific automatically below
        # ~1e-4 so 3.3e-6 displays compactly. Trailing zeros suppressed.
        return f"{value:g}"


class ExportDataDialog(QDialog):
    """Dialog for configuring and executing data export in GeoTIFF/HDF5/COG formats."""

    #: Radio-button ids in `format_btn_group`. 0-3 predate this dialog's COG
    #: support and are matched positionally in `_start_export`; named here so a
    #: fifth format is added in one place instead of four.
    FORMAT_ID_GEOTIFF = 0
    FORMAT_ID_HDF5 = 1
    FORMAT_ID_BOTH = 2
    FORMAT_ID_VTU = 3
    FORMAT_ID_COG = 4

    def __init__(self, results_dir, mesh_data, timesteps, parent=None):
        super().__init__(parent)
        self.results_dir = Path(results_dir)
        self.mesh_data = mesh_data
        self.timesteps = timesteps
        self.export_worker = None

        # Check VTK availability
        try:
            import vtk
            self.vtk_available = True
        except ImportError:
            self.vtk_available = False

        # Import availability and detection functions
        from gui_core.data_exporter import (
            SCIPY_AVAILABLE, RASTERIO_AVAILABLE, H5PY_AVAILABLE,
            COG_AVAILABLE, COG_UNAVAILABLE_REASON,
            GridResolutionEstimator, FIELD_INFO, BODY_RADII,
            detect_body_from_mesh, choose_projection, POLAR_LAT_THRESHOLD,
            detect_available_statistics, STATISTICS_FIELDS, STAT_LABELS
        )
        self.scipy_available = SCIPY_AVAILABLE
        self.rasterio_available = RASTERIO_AVAILABLE
        self.h5py_available = H5PY_AVAILABLE
        self.cog_available = COG_AVAILABLE
        self.cog_unavailable_reason = COG_UNAVAILABLE_REASON
        self.body_radii = BODY_RADII

        # Auto-detect body from mesh vertices
        self.detected_body = detect_body_from_mesh(mesh_data) if mesh_data else None
        if self.detected_body is None:
            self.detected_body = 'Moon'  # fallback
        self.body_radius = BODY_RADII.get(self.detected_body, 1737400.0)

        # Auto-detect projection
        self.projection = 'eqc'  # fallback
        if mesh_data and 'latitudes' in mesh_data:
            self.projection = choose_projection(mesh_data['latitudes'])
            lats = mesh_data['latitudes']
            if self.projection == 'stere':
                min_lat = float(np.nanmin(lats))
                max_lat = float(np.nanmax(lats))
                self.pole_lat = 90.0 if min_lat > POLAR_LAT_THRESHOLD else -90.0

        # Auto-detect resolution
        self.auto_resolution = 0.05  # fallback
        # The mesh's OWN median face spacing, before the estimator rounds it to
        # one of `NICE_RESOLUTIONS`. Kept because that table's floor is 0.001
        # deg -- 30.3 m on the Moon -- so on a meter-scale site mesh the
        # "auto-detected" number is coarser than the data by a large factor and
        # nothing said so. The dialog now says so. See
        # PTM-EXPORT-NICE-RESOLUTION-FLOOR.
        self.raw_face_spacing_deg = None
        if SCIPY_AVAILABLE and mesh_data:
            self.auto_resolution, self.raw_face_spacing_deg = \
                GridResolutionEstimator.estimate_resolution(
                    mesh_data['latitudes'], mesh_data['longitudes'],
                    return_raw=True)

        # Detect available fields
        self.available_fields = []
        for field_name in FIELD_INFO:
            field_dir = self.results_dir / field_name
            if field_dir.exists() and any(field_dir.glob("*.bin")):
                self.available_fields.append(field_name)

        # Detect available statistics
        self.available_statistics = detect_available_statistics(self.results_dir)

        self.setWindowTitle("Export Simulation Data")
        self.setMinimumWidth(480)
        self.init_ui()

    def init_ui(self):
        from gui_core.data_exporter import FIELD_INFO

        layout = QVBoxLayout(self)

        # Format selection
        format_group = QGroupBox("Format")
        format_layout = QVBoxLayout()

        self.format_btn_group = QButtonGroup(self)
        format_row = QHBoxLayout()

        self.geotiff_radio = QRadioButton("GeoTIFF")
        self.hdf5_radio = QRadioButton("HDF5")
        self.both_radio = QRadioButton("Both")
        self.vtu_radio = QRadioButton("VTU/PVD")
        self.cog_radio = QRadioButton("COG")
        self.cog_radio.setToolTip(
            "Cloud Optimized GeoTIFF: the same product "
            "scripts/ptm3d_stats_to_cog.py writes. Single-band Float32, GDAL "
            "COG layout, 512 px internal tiles, AVERAGE overview chain. One "
            "file per statistic, and one file per timestep for a time series."
        )
        self.both_radio.setChecked(True)

        self.format_btn_group.addButton(self.geotiff_radio, 0)
        self.format_btn_group.addButton(self.hdf5_radio, 1)
        self.format_btn_group.addButton(self.both_radio, 2)
        self.format_btn_group.addButton(self.vtu_radio, 3)
        self.format_btn_group.addButton(self.cog_radio, self.FORMAT_ID_COG)

        if not self.vtk_available:
            self.vtu_radio.setEnabled(False)
            self.vtu_radio.setToolTip("VTK not available (pip install vtk)")
        if not self.cog_available:
            self.cog_radio.setEnabled(False)
            self.cog_radio.setToolTip(
                f"COG export unavailable: {self.cog_unavailable_reason}")

        format_row.addWidget(self.geotiff_radio)
        format_row.addWidget(self.hdf5_radio)
        format_row.addWidget(self.both_radio)
        format_row.addWidget(self.vtu_radio)
        format_row.addWidget(self.cog_radio)
        format_row.addStretch()
        format_layout.addLayout(format_row)

        # Warnings for missing dependencies
        warnings = []
        if not self.rasterio_available:
            warnings.append("rasterio not available (pip install rasterio)")
            self.geotiff_radio.setEnabled(False)
            self.both_radio.setEnabled(False)
            self.hdf5_radio.setChecked(True)
        if not self.h5py_available:
            warnings.append("h5py not available (pip install h5py)")
            self.hdf5_radio.setEnabled(False)
            if not self.rasterio_available:
                self.both_radio.setEnabled(False)
            elif self.both_radio.isChecked():
                self.geotiff_radio.setChecked(True)
        if not self.scipy_available:
            warnings.append("scipy not available - regridding disabled (pip install scipy)")
        if not self.vtk_available:
            warnings.append("vtk not available - VTU/PVD export disabled (pip install vtk)")
        if not self.cog_available:
            warnings.append(
                f"COG export disabled: {self.cog_unavailable_reason}")

        if warnings:
            warn_label = QLabel("\n".join(warnings))
            warn_label.setStyleSheet("color: #ff9800; font-size: 11px; font-style: italic;")
            format_layout.addWidget(warn_label)

        format_group.setLayout(format_layout)
        layout.addWidget(format_group)

        # Fields selection
        fields_group = QGroupBox("Fields")
        fields_layout = QVBoxLayout()

        self.field_checks = {}
        default_checked = {'temperature', 'illumination', 'direct_solar'}
        for field_name in self.available_fields:
            info = FIELD_INFO.get(field_name, {})
            label = info.get('label', field_name)
            units = info.get('units', '')
            cb = QCheckBox(f"{label} ({units})" if units else label)
            cb.setChecked(field_name in default_checked)
            cb.stateChanged.connect(self._update_estimate)
            fields_layout.addWidget(cb)
            self.field_checks[field_name] = cb

        if not self.available_fields:
            if self.available_statistics:
                # Stats-only run (chunked workflow with skip_time_series=true).
                # The Statistics section below is the user's path forward.
                hint = QLabel(
                    "Time-series fields not present "
                    "(skip_time_series=true). Statistics still available "
                    "below."
                )
                hint.setStyleSheet("color: #888888; font-style: italic;")
                hint.setWordWrap(True)
                fields_layout.addWidget(hint)
            else:
                no_fields_label = QLabel(
                    "No data fields found in results directory."
                )
                no_fields_label.setStyleSheet(
                    "color: #ff5252; font-style: italic;"
                )
                fields_layout.addWidget(no_fields_label)

        fields_group.setLayout(fields_layout)
        layout.addWidget(fields_group)

        # Statistics selection
        self.stat_checks = {}
        if self.available_statistics:
            from gui_core.data_exporter import STATISTICS_FIELDS, STAT_LABELS

            stats_group = QGroupBox("Statistics (avg/max/min over integration window)")
            stats_layout = QVBoxLayout()

            # Group by base field, preserving insertion order
            seen_fields = []
            for base_field, _st in self.available_statistics:
                if base_field not in seen_fields:
                    seen_fields.append(base_field)

            default_stat_fields = {'temperature', 'ice_depth'}
            for base_field in seen_fields:
                field_info = STATISTICS_FIELDS.get(base_field, {})
                field_label = field_info.get('label', base_field)
                units = field_info.get('units', '')
                units_str = f" ({units})" if units else ""
                is_scalar = field_info.get('scalar', False)

                row = QHBoxLayout()
                if is_scalar:
                    # Scalar fields (e.g. ice_depth): single checkbox, no
                    # avg/max/min split.
                    key = (base_field, None)
                    if key in self.available_statistics:
                        cb = QCheckBox(f"{field_label}{units_str}")
                        cb.setChecked(base_field in default_stat_fields)
                        cb.stateChanged.connect(self._update_estimate)
                        row.addWidget(cb)
                        self.stat_checks[key] = cb
                else:
                    row.addWidget(QLabel(f"{field_label}{units_str}:"))
                    for stat_type in ['avg', 'max', 'min']:
                        key = (base_field, stat_type)
                        if key in self.available_statistics:
                            stat_label = STAT_LABELS[stat_type]
                            cb = QCheckBox(stat_label)
                            cb.setChecked(base_field in default_stat_fields)
                            cb.stateChanged.connect(self._update_estimate)
                            row.addWidget(cb)
                            self.stat_checks[key] = cb

                row.addStretch()
                stats_layout.addLayout(row)

            stats_group.setLayout(stats_layout)
            layout.addWidget(stats_group)

        # Connect format change to toggle grid group
        self.format_btn_group.buttonClicked.connect(self._on_format_changed)

        # Grid & Projection
        grid_group = QGroupBox("Grid & Projection")
        self.grid_group = grid_group
        grid_layout = QGridLayout()

        # Auto-detected body
        radius_km = self.body_radius / 1000.0
        body_label = QLabel(f"Body: {self.detected_body} (R = {radius_km:.1f} km)")
        body_label.setStyleSheet("font-weight: bold;")
        grid_layout.addWidget(body_label, 0, 0, 1, 2)

        # Projection type
        if self.projection == 'stere':
            pole_str = "North" if getattr(self, 'pole_lat', 90) > 0 else "South"
            proj_text = f"Polar stereographic ({pole_str} pole, +proj=stere)"
        else:
            proj_text = "Equirectangular (+proj=eqc)"
        proj_label = QLabel(f"CRS: {proj_text}")
        proj_label.setStyleSheet("color: #888888; font-size: 11px;")
        grid_layout.addWidget(proj_label, 0, 2)

        grid_layout.addWidget(QLabel("Resolution:"), 1, 0)
        # Resolution spinbox + unit selector live in a small horizontal row
        # so we can switch between "degrees" (the canonical regridder unit)
        # and "m/px" (convenient for science users). The spinbox accepts
        # scientific notation (e.g. 3.3e-6) for sub-microdegree resolutions.
        res_row = QHBoxLayout()
        res_row.setContentsMargins(0, 0, 0, 0)

        self.resolution_spin = ScientificDoubleSpinBox()
        # Range covers 1 m/px on Moon (~3.3e-6\u00b0) up to 10\u00b0 (regional eqc).
        self.resolution_spin.setRange(1e-7, 100.0)
        self.resolution_spin.setSingleStep(0.001)
        self.resolution_spin.setValue(self.auto_resolution)
        self.resolution_spin.valueChanged.connect(self._on_resolution_changed)
        res_row.addWidget(self.resolution_spin, 1)

        self.resolution_unit_combo = QComboBox()
        self.resolution_unit_combo.addItems(["degrees", "m/px"])
        self.resolution_unit_combo.setCurrentText("degrees")
        self.resolution_unit_combo.currentTextChanged.connect(
            self._on_resolution_unit_changed
        )
        res_row.addWidget(self.resolution_unit_combo, 0)

        # Internal source-of-truth (always degrees). Updated in the
        # value/unit-change handlers; read by _start_export.
        self._resolution_degrees = float(self.auto_resolution)

        grid_layout.addLayout(res_row, 1, 1)

        # Auto-detected display: degrees + meters at the body's surface.
        auto_pixel_m = self._get_body_radius() * np.radians(self.auto_resolution)
        auto_label = QLabel(
            f"(auto-detected: {self.auto_resolution:g}\u00b0  "
            f"\u2248 {auto_pixel_m:.2f} m/px)"
        )
        auto_label.setStyleSheet("color: #888888; font-size: 11px;")
        grid_layout.addWidget(auto_label, 1, 2)

        self.grid_size_label = QLabel("")
        self.grid_size_label.setStyleSheet("color: #888888; font-size: 11px;")
        grid_layout.addWidget(self.grid_size_label, 2, 0, 1, 3)

        # The estimator rounds to a fixed table whose floor is 0.001 deg. On a
        # meter-scale site mesh that is coarser than the data by a large factor,
        # and until now nothing said so -- the dialog only showed the rounded
        # number. It does not change the number: it names the mesh's own spacing
        # beside it so the user can choose.
        self.spacing_label = QLabel("")
        self.spacing_label.setStyleSheet("color: #ff9800; font-size: 11px;")
        self.spacing_label.setWordWrap(True)
        self.spacing_label.setVisible(False)
        if self.raw_face_spacing_deg:
            spacing_m = self._get_body_radius() * np.radians(
                self.raw_face_spacing_deg)
            auto_m = self._get_body_radius() * np.radians(self.auto_resolution)
            if self.auto_resolution > 1.5 * self.raw_face_spacing_deg:
                self.spacing_label.setText(
                    f"This mesh's median face spacing is "
                    f"{self.raw_face_spacing_deg:.3g}\u00b0 "
                    f"\u2248 {spacing_m:.2f} m. The auto-detected resolution "
                    f"above is rounded to a fixed table whose finest entry is "
                    f"0.001\u00b0 \u2248 {auto_m:.1f} m, so it is coarser "
                    f"than the mesh. Switch the unit to m/px to export at the "
                    f"mesh's own scale."
                )
                self.spacing_label.setVisible(True)
        grid_layout.addWidget(self.spacing_label, 3, 0, 1, 3)

        # One projection choice, and the minimum one: which meridian the output
        # CRS is centered on. Default OFF, which is what every GUI export has
        # done to date (the mesh's own midpoint). Turning it on reproduces
        # `scripts/ptm3d_stats_to_cog.py --fixed-meridian 0`, whose reason is on
        # record: many GIS tools lock the session CRS to the first layer loaded
        # and silently reuse it, so site-centered rasters from different sites
        # landed tens of km apart. The note below fires when it is off and COG
        # is selected, rather than leaving that silent.
        self.shared_crs_check = QCheckBox(
            "Shared CRS: central meridian 0\u00b0")
        self.shared_crs_check.setChecked(False)
        self.shared_crs_check.stateChanged.connect(self._on_shared_crs_changed)
        # ⛔ It only reaches the raster on the POLAR STEREOGRAPHIC branch.
        # `MeshRegridder` carries `center_lon` into `_init_stereo`, but
        # `_init_eqc` builds the grid from the raw longitude bounds and both
        # raster writers then recompute the equirectangular central meridian
        # from those bounds -- so on an equatorial mesh the checkbox would
        # change the README and the PROJ string while the pixels stayed where
        # they were. Offering an inert control that makes the documentation
        # disagree with the data is worse than not offering it.
        if self.projection == 'stere':
            self.shared_crs_check.setToolTip(
                "Center the output projection on 0\u00b0 instead of this "
                "mesh's own midpoint, so rasters from different sites share "
                "one CRS and overlay correctly in a GIS session. Recommended "
                "for COGs that will be loaded beside other sites."
            )
        else:
            self.shared_crs_check.setEnabled(False)
            self.shared_crs_check.setToolTip(
                "Only available for polar stereographic output. An "
                "equirectangular raster is built on its own longitude bounds, "
                "so forcing a central meridian here would change the stated "
                "CRS without moving a pixel."
            )
        grid_layout.addWidget(self.shared_crs_check, 4, 0, 1, 2)

        self.crs_note_label = QLabel("")
        self.crs_note_label.setStyleSheet("color: #888888; font-size: 11px;")
        self.crs_note_label.setWordWrap(True)
        grid_layout.addWidget(self.crs_note_label, 5, 0, 1, 3)

        self.per_timestep_check = QCheckBox("One GeoTIFF per timestep (smaller files)")
        self.per_timestep_check.setChecked(True)
        self.per_timestep_check.setToolTip(
            "Write individual GeoTIFF files per timestep instead of one "
            "large multi-band file. Recommended for large meshes."
        )
        grid_layout.addWidget(self.per_timestep_check, 6, 0, 1, 3)

        grid_group.setLayout(grid_layout)
        layout.addWidget(grid_group)

        # Output directory
        output_group = QGroupBox("Output")
        output_layout = QGridLayout()

        output_layout.addWidget(QLabel("Directory:"), 0, 0)
        self.output_dir_edit = QLineEdit()
        default_out = str(self.results_dir / "exports")
        self.output_dir_edit.setText(default_out)
        self.output_dir_edit.textChanged.connect(self._update_estimate)
        output_layout.addWidget(self.output_dir_edit, 0, 1)

        browse_btn = QPushButton("Browse...")
        browse_btn.setMaximumWidth(80)
        browse_btn.clicked.connect(self._browse_output)
        output_layout.addWidget(browse_btn, 0, 2)

        self.estimate_label = QLabel("")
        self.estimate_label.setStyleSheet("color: #888888; font-size: 11px;")
        output_layout.addWidget(self.estimate_label, 1, 0, 1, 3)

        output_group.setLayout(output_layout)
        layout.addWidget(output_group)

        # Buttons
        self.button_box = QDialogButtonBox()
        self.export_btn = self.button_box.addButton(
            "Export", QDialogButtonBox.ButtonRole.AcceptRole
        )
        self.cancel_btn = self.button_box.addButton(
            QDialogButtonBox.StandardButton.Cancel
        )
        self.export_btn.clicked.connect(self._start_export)
        self.cancel_btn.clicked.connect(self.reject)
        layout.addWidget(self.button_box)

        # Initialize displays
        self._update_grid_info()
        self._on_format_changed()
        self._update_estimate()

    def _get_body_radius(self):
        """Get detected body radius in meters."""
        return self.body_radius

    def _meters_per_degree(self):
        """Great-circle meters per degree at the body's surface."""
        return self._get_body_radius() * np.radians(1.0)

    def _spin_value_to_degrees(self, value):
        """Convert the current spinbox value to degrees, honoring the unit
        combo. The regridder always works in degrees."""
        unit = self.resolution_unit_combo.currentText()
        if unit == "m/px":
            return float(value) / self._meters_per_degree()
        return float(value)

    def _degrees_to_spin_value(self, degrees):
        """Inverse of _spin_value_to_degrees, for switching units without
        changing the underlying physical resolution."""
        unit = self.resolution_unit_combo.currentText()
        if unit == "m/px":
            return float(degrees) * self._meters_per_degree()
        return float(degrees)

    def _on_resolution_changed(self, value):
        """User edited the spinbox — recompute canonical degrees + refresh."""
        self._resolution_degrees = self._spin_value_to_degrees(value)
        self._update_grid_info()

    def _on_resolution_unit_changed(self, _new_unit):
        """User switched between degrees and m/px — preserve the physical
        resolution by converting the displayed value, not by reinterpreting
        the same numeric input under a different unit."""
        # Block the valueChanged signal while we update the display so we
        # don't double-convert.
        new_display = self._degrees_to_spin_value(self._resolution_degrees)
        self.resolution_spin.blockSignals(True)
        try:
            self.resolution_spin.setValue(new_display)
        finally:
            self.resolution_spin.blockSignals(False)
        self._update_grid_info()

    def _update_grid_info(self):
        """Update grid size display when resolution changes."""
        if not self.mesh_data:
            return
        res = self._resolution_degrees
        lats = self.mesh_data['latitudes']
        lons = self.mesh_data['longitudes']
        body_radius = self._get_body_radius()

        if self.projection == 'stere':
            # For stereographic, grid is in projected meters
            from gui_core.data_exporter import _forward_polar_stereo
            pole_lat = getattr(self, 'pole_lat', 90.0)
            choice = self._center_lon_choice()
            center_lon = self._site_center_lon() if choice is None else choice
            x, y = _forward_polar_stereo(lats, lons, pole_lat, center_lon, body_radius)
            # Pixel size in meters from degree-resolution mapped to projected space
            pixel_m = body_radius * np.radians(res)
            x_range = float(np.nanmax(x) - np.nanmin(x))
            y_range = float(np.nanmax(y) - np.nanmin(y))
            n_x = int(np.ceil(x_range / pixel_m)) + 1
            n_y = int(np.ceil(y_range / pixel_m)) + 1
            self.grid_size_label.setText(
                f"Grid size: {n_y} \u00d7 {n_x} pixels "
                f"(~{pixel_m:.0f} m/px, stereographic)"
            )
        else:
            lat_range = np.ptp(lats)
            lon_range = np.ptp(lons)
            n_lat = int(np.ceil(lat_range / res)) + 1
            n_lon = int(np.ceil(lon_range / res)) + 1
            # Pixel size in meters at center latitude
            center_lat = (np.nanmin(lats) + np.nanmax(lats)) / 2
            pixel_m = body_radius * np.radians(res)
            pixel_m_lon = pixel_m * np.cos(np.radians(center_lat))
            self.grid_size_label.setText(
                f"Grid size: {n_lat} \u00d7 {n_lon} pixels "
                f"(~{pixel_m:.0f} \u00d7 {pixel_m_lon:.0f} m/px at center)"
            )
        self._update_estimate()

    def _update_estimate(self):
        """Update estimated output size."""
        selected_fields = [f for f, cb in self.field_checks.items() if cb.isChecked()]
        selected_stats = [
            key for key, cb in self.stat_checks.items() if cb.isChecked()
        ]
        n_fields = len(selected_fields)
        n_timesteps = len(self.timesteps)
        n_statistics = len(selected_stats)

        if n_fields == 0 and n_statistics == 0:
            self.estimate_label.setText("Estimated size: --")
            self.export_btn.setEnabled(False)
            return

        self.export_btn.setEnabled(True)

        res = self._resolution_degrees
        lats = self.mesh_data['latitudes']
        lons = self.mesh_data['longitudes']
        n_faces = len(lats)

        if self.projection == 'stere':
            from gui_core.data_exporter import _forward_polar_stereo
            pole_lat = getattr(self, 'pole_lat', 90.0)
            choice = self._center_lon_choice()
            center_lon = self._site_center_lon() if choice is None else choice
            body_radius = self._get_body_radius()
            x, y = _forward_polar_stereo(lats, lons, pole_lat, center_lon, body_radius)
            pixel_m = body_radius * np.radians(res)
            n_lat = int(np.ceil((np.nanmax(y) - np.nanmin(y)) / pixel_m)) + 1
            n_lon = int(np.ceil((np.nanmax(x) - np.nanmin(x)) / pixel_m)) + 1
        else:
            n_lat = int(np.ceil(np.ptp(lats) / res)) + 1
            n_lon = int(np.ceil(np.ptp(lons) / res)) + 1

        total_bytes = 0
        fmt_id = self.format_btn_group.checkedId()

        if fmt_id == 0 or fmt_id == 2:  # GeoTIFF or Both
            # Multi-band GeoTIFF: ~4 bytes per pixel per band, compressed ~50%
            total_bytes += n_fields * n_timesteps * n_lat * n_lon * 4 * 0.5
            # Statistics: single-band GeoTIFFs
            total_bytes += n_statistics * n_lat * n_lon * 4 * 0.5

        if fmt_id == 1 or fmt_id == 2:  # HDF5 or Both
            # Unstructured: n_fields * n_timesteps * n_faces * 4 bytes, compressed ~50%
            total_bytes += n_fields * n_timesteps * n_faces * 4 * 0.5
            # Regridded: n_fields * n_timesteps * n_lat * n_lon * 4 bytes, compressed ~50%
            total_bytes += n_fields * n_timesteps * n_lat * n_lon * 4 * 0.5
            # Statistics: per-face + regridded
            total_bytes += n_statistics * n_faces * 4 * 0.5
            total_bytes += n_statistics * n_lat * n_lon * 4 * 0.5

        if fmt_id == self.FORMAT_ID_COG:
            # One single-band COG per timestep per field, plus one per
            # statistic. LZW on a smooth field plus the overview chain runs
            # near half the raw size, same assumption the GeoTIFF rows use.
            total_bytes += n_fields * n_timesteps * n_lat * n_lon * 4 * 0.5
            total_bytes += n_statistics * n_lat * n_lon * 4 * 0.5

        if fmt_id == 3:  # VTU/PVD
            # VTU: mesh geometry + per-field data per timestep, compressed ~50%
            mesh_bytes = n_faces * 3 * 3 * 4  # vertices (3 verts * 3 coords * float32)
            total_bytes += n_timesteps * (mesh_bytes + n_fields * n_faces * 4) * 0.5
            # Statistics VTU files
            total_bytes += (mesh_bytes + n_statistics * n_faces * 4) * 0.5

        # Format size
        if total_bytes < 1024:
            size_str = f"{total_bytes:.0f} B"
        elif total_bytes < 1024**2:
            size_str = f"{total_bytes / 1024:.1f} KB"
        elif total_bytes < 1024**3:
            size_str = f"{total_bytes / 1024**2:.1f} MB"
        else:
            size_str = f"{total_bytes / 1024**3:.2f} GB"

        self.estimate_label.setText(f"Estimated size: ~{size_str}")

    def _on_format_changed(self, button=None):
        """Toggle Grid & Projection group based on selected format."""
        fmt_id = self.format_btn_group.checkedId()
        is_vtu = (fmt_id == self.FORMAT_ID_VTU)
        is_cog = (fmt_id == self.FORMAT_ID_COG)
        self.grid_group.setEnabled(not is_vtu)
        # A COG holds one band. There is no multi-band COG to choose, so the
        # control is pinned rather than accepting a setting it would ignore.
        self.per_timestep_check.setEnabled(not is_cog and not is_vtu)
        if is_cog:
            self.per_timestep_check.setToolTip(
                "A COG holds one band, so a time series is always one file "
                "per timestep.")
        else:
            self.per_timestep_check.setToolTip(
                "Write individual GeoTIFF files per timestep instead of one "
                "large multi-band file. Recommended for large meshes.")
        self._refresh_crs_note()
        self._update_estimate()

    def _on_shared_crs_changed(self, _state=None):
        self._refresh_crs_note()
        self._update_grid_info()

    def _refresh_crs_note(self):
        """Say which meridian the output CRS will be centered on.

        Never silent: the site-centered default is the one that composes badly
        across sites, so when a COG is about to be written on it the dialog says
        so instead of leaving the user to find out in a GIS session.
        """
        if not hasattr(self, 'crs_note_label'):
            return
        is_cog = (self.format_btn_group.checkedId() == self.FORMAT_ID_COG)
        if self.projection != 'stere':
            self.crs_note_label.setStyleSheet(
                "color: #888888; font-size: 11px;")
            self.crs_note_label.setText(
                "Equirectangular output is centered on this mesh's own "
                "longitude bounds.")
            return
        if self.shared_crs_check.isChecked():
            self.crs_note_label.setStyleSheet(
                "color: #888888; font-size: 11px;")
            self.crs_note_label.setText(
                "Central meridian 0\u00b0. Rasters from different sites share "
                "one CRS and overlay correctly.")
            return
        center_lon = self._site_center_lon()
        text = (f"Central meridian {center_lon:.4f}\u00b0 (this mesh's own "
                f"midpoint).")
        if is_cog:
            text += (" Rasters written this way do NOT share a CRS with other "
                     "sites; a GIS session that locks onto the first layer's "
                     "CRS will place them wrongly.")
            self.crs_note_label.setStyleSheet(
                "color: #ff9800; font-size: 11px;")
        else:
            self.crs_note_label.setStyleSheet(
                "color: #888888; font-size: 11px;")
        self.crs_note_label.setText(text)

    def _site_center_lon(self):
        """The mesh midpoint meridian the regridder would pick on its own."""
        if not self.mesh_data or 'longitudes' not in self.mesh_data:
            return 0.0
        lon = np.asarray(self.mesh_data['longitudes'], dtype=np.float64)
        if np.ptp(lon) > 300:
            lon = lon % 360.0
        return float((np.nanmin(lon) + np.nanmax(lon)) / 2)

    def _center_lon_choice(self):
        """What to hand `MeshRegridder(center_lon=...)`: 0.0, or None.

        Never 0.0 on the equirectangular branch, where the regridder accepts it
        and the raster writers ignore it -- see the control's own comment.
        """
        if self.projection != 'stere':
            return None
        return 0.0 if self.shared_crs_check.isChecked() else None

    def _browse_output(self):
        """Browse for output directory."""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Output Directory",
            self.output_dir_edit.text(),
            QFileDialog.Option.ShowDirsOnly
        )
        if directory:
            self.output_dir_edit.setText(directory)

    def _start_export(self):
        """Start the export process."""
        from gui_core.data_exporter import DataExportWorker

        selected_fields = [f for f, cb in self.field_checks.items() if cb.isChecked()]
        selected_stats = [
            key for key, cb in self.stat_checks.items() if cb.isChecked()
        ]

        if not selected_fields and not selected_stats:
            QMessageBox.warning(
                self, "Nothing Selected",
                "Please select at least one field or statistic to export."
            )
            return

        output_dir = self.output_dir_edit.text().strip()
        if not output_dir:
            QMessageBox.warning(self, "No Output", "Please specify an output directory.")
            return

        # Determine formats
        fmt_id = self.format_btn_group.checkedId()
        formats = []
        if fmt_id == self.FORMAT_ID_GEOTIFF:
            formats = ['geotiff']
        elif fmt_id == self.FORMAT_ID_HDF5:
            formats = ['hdf5']
        elif fmt_id == self.FORMAT_ID_VTU:
            formats = ['vtu']
        elif fmt_id == self.FORMAT_ID_COG:
            formats = ['cog']
        else:
            formats = ['geotiff', 'hdf5']

        config = {
            'results_dir': str(self.results_dir),
            'mesh_data': self.mesh_data,
            'timesteps': self.timesteps,
            'fields': selected_fields,
            'formats': formats,
            'resolution': self._resolution_degrees,
            'output_dir': output_dir,
            'geotiff_per_timestep': self.per_timestep_check.isChecked(),
            'body_radius': self._get_body_radius(),
            'statistics': selected_stats,
            'center_lon': self._center_lon_choice(),
        }

        # Disable controls during export
        self.export_btn.setEnabled(False)
        self.cancel_btn.setText("Cancel Export")

        # Create progress dialog
        n_timesteps = len(self.timesteps)
        n_fields = len(selected_fields)
        n_statistics = len(selected_stats)
        total_steps = 1 + (n_fields * n_timesteps) + n_fields + n_statistics

        self.progress = QProgressDialog("Starting export...", "Cancel", 0, total_steps, self)
        self.progress.setWindowTitle("Exporting Data")
        self.progress.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress.setMinimumDuration(0)
        self.progress.canceled.connect(self._cancel_export)
        self.progress.show()

        # Start worker thread
        self.export_worker = DataExportWorker(config)
        self.export_worker.progress_updated.connect(self._on_progress)
        self.export_worker.status_updated.connect(self._on_status)
        self.export_worker.export_finished.connect(self._on_finished)
        self.export_worker.start()

    def _cancel_export(self):
        """Cancel the running export."""
        if self.export_worker:
            self.export_worker.cancel()

    def _on_progress(self, current, total):
        """Handle progress update."""
        if hasattr(self, 'progress') and self.progress:
            self.progress.setMaximum(total)
            self.progress.setValue(current)

    def _on_status(self, message):
        """Handle status update."""
        if hasattr(self, 'progress') and self.progress:
            self.progress.setLabelText(message)

    def _on_finished(self, success, message):
        """Handle export completion."""
        if hasattr(self, 'progress') and self.progress:
            self.progress.close()

        if success:
            QMessageBox.information(self, "Export Complete", message)
            self.accept()
        else:
            QMessageBox.warning(self, "Export", message)
            self.export_btn.setEnabled(True)
            self.cancel_btn.setText("Cancel")


class VisualizationPanel(QWidget):
    """Combined visualization panel with 3D view and 2D plots"""

    face_selected = pyqtSignal(int, dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.results_dir = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # What the terrain preview is showing, and every reservation about it:
        # the datum the elevation is measured from, an assumed length unit, and
        # whether the mesh was thinned to be drawn at all. PLAIN TEXT, and
        # above the view rather than in a tooltip -- a preview whose caveats
        # are only discoverable by hovering is a preview whose caveats nobody
        # reads.
        self.preview_note = QLabel("")
        self.preview_note.setWordWrap(True)
        self.preview_note.setTextFormat(Qt.TextFormat.PlainText)
        self.preview_note.setStyleSheet(
            "color: #90caf9; font-size: 11px; padding: 4px;")
        self.preview_note.setVisible(False)
        layout.addWidget(self.preview_note)

        # ⭐ WHAT THE OFF-SCALE COLORS MEAN, FOR THE FIELD NOW ON SCREEN.
        #
        # The color bar spans the finite range; `shadow_depth`'s two sentinels
        # sit OUTSIDE it by construction, so the bar cannot label them and a
        # magenta facet is unreadable without this sentence. PLAIN TEXT, above
        # the view, not a tooltip -- "0 m" and "+inf" are opposite meanings and
        # a legend nobody hovers over is a legend nobody reads. Empty, and
        # hidden, for every field that carries no sentinel, which is all of
        # them but the six in `SHADOW_SENTINEL_FIELDS`.
        self.field_note = QLabel("")
        self.field_note.setWordWrap(True)
        self.field_note.setTextFormat(Qt.TextFormat.PlainText)
        self.field_note.setStyleSheet(
            "color: #ffcc80; font-size: 11px; padding: 4px;")
        self.field_note.setVisible(False)
        layout.addWidget(self.field_note)

        # Create tab widget for 3D/2D views
        self.tab_widget = QTabWidget()

        # 3D View Tab
        self.view_3d = PyVistaWidget()
        self.view_3d.face_selected.connect(self.on_face_selected)
        self.tab_widget.addTab(self.view_3d, "3D Visualization")

        # 2D Plots Tab
        self.plots_2d = PlotWidget()
        self.plots_2d.face_selection_requested.connect(self.on_face_selection_from_plots)
        self.tab_widget.addTab(self.plots_2d, "2D Plots")

        layout.addWidget(self.tab_widget)

        # Controls panel at bottom
        self.create_controls()
        layout.addWidget(self.controls_widget)

    # Known field keys and their display labels. Per-timestep series (loaded
    # from <results>/<key>/<key>_NNNN.bin) plus statistics rasters (loaded from
    # <results>/statistics/<key>.bin — single file each, no timestep index).
    # Statistics keys are recognized by `is_stats_field`: the trailing
    # _max/_avg/_min, or membership of SCALAR_STATS_FIELDS below.
    #
    # ⭐ WHY THIS IS STILL A HAND-WRITTEN LITERAL, given that
    # `ptm.results.STATISTICS_FIELDS` carries the same key set. Deriving it
    # was considered and rejected, for reasons worth not re-deriving:
    #   * This module must import with `src/python` absent from the path --
    #     that is how the GUI is distributed -- so a derived catalog needs a
    #     hand-written fallback anyway, and the duplication survives with a
    #     second code path on top of it.
    #   * The shared registry carries no colormap, and colormap choice is a
    #     viewer decision with measured constraints (see OFF_SCALE_NAN_COLOR).
    #     A derivation would trade this dict for a field->colormap dict of the
    #     same length.
    #   * The labels here are not the shared registry's: they carry the unit
    #     inline and an "Annual Max"/"Max"/"Mean" prefix per statistic. Deriving
    #     them would rewrite ~20 strings a user already reads today.
    #   * The gate that binds the two (`src/python/tests/test_gui_field_registry.py`)
    #     reads this literal with `ast`, so it needs neither PyQt6 nor a display
    #     and CANNOT be skipped. An import-based check can be, and a skipped
    #     gate reads as a pass.
    # What was fixed instead is the thing duplication actually cost: the gate
    # now proves BOTH directions against every catalog `ptm.results` publishes,
    # and separately proves that every product in a real run directory is
    # offered. Owed ledger row for the remaining duplication:
    # PTM-GUI-FIELD-REGISTRY-DUP, narrowed, not closed.
    # Curated set of matplotlib colormaps for the toolbar Colormap dropdown.
    # Order: perceptually-uniform sequential first, then classic sequentials,
    # cyclic, diverging, qualitative, then their _r reversed counterparts.
    # Any cmap matplotlib/PyVista supports can be passed directly to
    # PyVistaWidget.set_cmap_override(); these are just the convenient ones.
    AVAILABLE_COLORMAPS = [
        # perceptually uniform
        'viridis', 'plasma', 'inferno', 'magma', 'cividis',
        'viridis_r', 'plasma_r', 'inferno_r', 'magma_r', 'cividis_r',
        # classic sequential
        'gray', 'gray_r', 'hot', 'cool', 'turbo', 'jet',
        'Blues', 'Greens', 'Oranges', 'Reds', 'Purples',
        'Blues_r', 'Greens_r', 'Oranges_r', 'Reds_r', 'Purples_r',
        # diverging
        'coolwarm', 'bwr', 'seismic', 'RdBu_r', 'RdYlBu_r', 'PiYG',
        # cyclic
        'twilight', 'twilight_shifted', 'hsv',
        # terrain-flavored
        'terrain', 'gist_earth', 'ocean', 'cubehelix',
    ]

    FIELD_INFO = {
        # Per-step time series
        'temperature':       ('Temperature (K)',                       'plasma'),
        'illumination':      ('Total Illumination (W/m²)',             'viridis'),
        'direct_solar':      ('Direct Solar (W/m²)',                   'inferno'),
        'primary_reflected': ('Primary Reflected (W/m²)',              'cividis'),
        'primary_ir':        ('Primary IR (W/m²)',                     'magma'),
        # Statistics: per-face annual aggregates (no timestep)
        'temperature_max':       ('Annual Max Temperature (K)',        'plasma'),
        'temperature_avg':       ('Annual Mean Temperature (K)',       'plasma'),
        'temperature_min':       ('Annual Min Temperature (K)',        'plasma'),
        'illumination_max':      ('Max Illumination (W/m²)',           'viridis'),
        'illumination_avg':      ('Mean Illumination (W/m²)',          'viridis'),
        'illumination_min':      ('Min Illumination (W/m²)',           'viridis'),
        'direct_solar_max':      ('Max Direct Solar (W/m²)',           'inferno'),
        'direct_solar_avg':      ('Mean Direct Solar (W/m²)',          'inferno'),
        'direct_solar_min':      ('Min Direct Solar (W/m²)',           'inferno'),
        'primary_reflected_max': ('Max Primary Reflected (W/m²)',      'cividis'),
        'primary_reflected_avg': ('Mean Primary Reflected (W/m²)',     'cividis'),
        'primary_reflected_min': ('Min Primary Reflected (W/m²)',      'cividis'),
        'primary_ir_max':        ('Max Primary IR (W/m²)',             'magma'),
        'primary_ir_avg':        ('Mean Primary IR (W/m²)',            'magma'),
        'primary_ir_min':        ('Min Primary IR (W/m²)',             'magma'),
        # The two indirect (multiply-scattered) radiosity bands, written by a
        # two-band radiosity run as both a per-step series and avg/max/min
        # maps. The ramp follows the BAND, not the component, so a reader can
        # put direct and indirect solar side by side without relearning the
        # bar: solar-reflected keeps `cividis` (as Primary Reflected does) and
        # thermal-IR keeps `magma` (as Primary IR does). These carry no
        # sentinel -- measured on the reference 1 h run, 0 NaN and 0 +inf in all
        # six statistics and in the 169 frames of each series -- but they DO
        # carry a convention their neighbors do not: see
        # `field_convention_note`.
        'indirect_solar':        ('Indirect Solar (W/m²)',             'cividis'),
        'indirect_solar_max':    ('Max Indirect Solar (W/m²)',         'cividis'),
        'indirect_solar_avg':    ('Mean Indirect Solar (W/m²)',        'cividis'),
        'indirect_solar_min':    ('Min Indirect Solar (W/m²)',         'cividis'),
        'indirect_ir':           ('Indirect IR (W/m²)',                'magma'),
        'indirect_ir_max':       ('Max Indirect IR (W/m²)',            'magma'),
        'indirect_ir_avg':       ('Mean Indirect IR (W/m²)',           'magma'),
        'indirect_ir_min':       ('Min Indirect IR (W/m²)',            'magma'),
        # Shadow depth: the height above the local surface at which the Sun is
        # not even partially visible (the umbral ceiling), meters, measured
        # along the body radial. Reversed cividis, matching ice_depth: deeper
        # reads darker. The two sentinels in these arrays are NOT on this ramp
        # -- see the module header and `sentinel_legend`.
        'shadow_depth':          ('Shadow Depth (m)',                  'cividis_r'),
        'shadow_depth_max':      ('Max Shadow Depth (m)',              'cividis_r'),
        'shadow_depth_avg':      ('Mean Shadow Depth (m)',             'cividis_r'),
        'shadow_depth_min':      ('Min Shadow Depth (m)',              'cividis_r'),
        # Schorghofer & Taylor (2007) depth-resolved ice-stability output.
        # Single per-face value (no avg/max/min split). File: ice_depth.bin.
        'ice_depth':             ('Ice Stability Depth (m)',           'cividis_r'),
        # The two populations the shadow-depth statistics deliberately exclude,
        # as fractions of the evaluated epochs. Dimensionless, 0 to 1. Single
        # per-face value, like ice_depth.
        'shadow_lit_frac':       ('Fraction of Epochs Lit at the Surface',
                                                                       'cividis'),
        'shadow_unresolved_frac': ('Fraction of Epochs Umbral Above the '
                                   'Search Ceiling',                   'magma'),
        # PTM-107(a): the flux-weighted albedo the surface boundary condition
        # absorbed with, over the integration window. Dimensionless, so no unit
        # in the title -- same as the two fractions above. Single per-face
        # value; file: statistics/effective_albedo.bin.
        # ⛔ NOT `gray`, which is what a reflectance map wants to be: the fixed
        # NaN color is mid-gray and a `gray` ramp contains it EXACTLY (measured
        # 0.0000), so the never-lit facets would be indistinguishable from a
        # mid-range albedo -- and on the reference 1 h run there are 26,057 of
        # them.
        # `Blues_r` is 0.3670 away, clears `OFF_SCALE_MIN_RGB_DISTANCE`, and
        # still reads the right way round for a reflectance (bright facets
        # bright). See the `OFF_SCALE_NAN_COLOR` note at the top of this file.
        'effective_albedo':      ('Effective Albedo',                  'Blues_r'),
    }

    #: Statistics products that are ONE value per face with no avg/max/min
    #: split, so the suffix rule in `is_stats_field` cannot see them.
    #: `ptm.results.STATISTICS_FIELDS` marks these with `'scalar': True`. The
    #: two lists are held together by `src/python/tests/test_gui_field_registry.py`
    #: rather than by an import, so this panel keeps working in a checkout
    #: where `src/python` is not on the path -- and so that adding a scalar
    #: product to the engine fails a test here instead of silently producing a
    #: field the viewer cannot offer. `effective_albedo` is one: a single
    #: per-face ratio with no avg/max/min split, like `ice_depth`.
    SCALAR_STATS_FIELDS = frozenset((
        'ice_depth',
        'shadow_lit_frac',
        'shadow_unresolved_frac',
        'effective_albedo',
    ))

    # Field keys that come from the statistics directory rather than a
    # per-timestep series: all names ending in _max / _avg / _min, plus the
    # scalar single-value products above.
    @staticmethod
    def is_stats_field(field_key):
        return (field_key.endswith('_max') or field_key.endswith('_avg') or
                field_key.endswith('_min') or
                field_key in VisualizationPanel.SCALAR_STATS_FIELDS)

    def create_controls(self):
        """Create visualization control panel"""
        self.controls_widget = QFrame()
        self.controls_widget.setFrameStyle(QFrame.Shape.StyledPanel)
        layout = QHBoxLayout(self.controls_widget)
        layout.setContentsMargins(10, 5, 10, 5)

        # Field selector
        layout.addWidget(QLabel("Field:"))
        self.field_combo = QComboBox()
        self.field_combo.setMinimumWidth(160)
        self.field_combo.currentTextChanged.connect(self._on_field_combo_changed)
        layout.addWidget(self.field_combo)

        layout.addWidget(QLabel("  "))

        # Colormap selector. Per-field; remembers the last user choice
        # for each field via PyVistaWidget.cmap_overrides.
        layout.addWidget(QLabel("Colormap:"))
        self.cmap_combo = QComboBox()
        self.cmap_combo.setMinimumWidth(120)
        for cm in self.AVAILABLE_COLORMAPS:
            self.cmap_combo.addItem(cm)
        self.cmap_combo.currentTextChanged.connect(self._on_cmap_combo_changed)
        layout.addWidget(self.cmap_combo)

        layout.addWidget(QLabel("  "))

        # Timestep slider
        layout.addWidget(QLabel("Timestep:"))
        self.timestep_slider = QSlider(Qt.Orientation.Horizontal)
        self.timestep_slider.setMinimum(0)
        self.timestep_slider.setMaximum(0)
        self.timestep_slider.valueChanged.connect(self.on_timestep_slider_changed)
        layout.addWidget(self.timestep_slider, 1)

        self.timestep_label = QLabel("0/0")
        layout.addWidget(self.timestep_label)

        layout.addWidget(QLabel(" "))  # Spacer

        # Playback controls
        self.play_btn = QPushButton("Play")
        self.play_btn.setMaximumWidth(60)
        self.play_btn.clicked.connect(self.toggle_playback)
        layout.addWidget(self.play_btn)

        # Animation timer
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.advance_timestep)
        self.is_playing = False

        layout.addWidget(QLabel(" | "))

        # View options
        self.show_edges_check = QCheckBox("Show Edges")
        self.show_edges_check.stateChanged.connect(self.on_show_edges_changed)
        layout.addWidget(self.show_edges_check)

        # Synthetic sun lighting toggle + access to az/el dialog. Default
        # sun position is computed from the loaded mesh's mean latitude
        # (typical-noon-sun-elevation rule); the dialog lets the user
        # tweak az/el or restore that default.
        self.sun_light_check = QCheckBox("Sun Light")
        self.sun_light_check.setToolTip(
            "Replace default lighting with a directional sun positioned\n"
            "at typical-noon elevation for the mesh's mean latitude.")
        self.sun_light_check.stateChanged.connect(self._on_sun_light_toggled)
        layout.addWidget(self.sun_light_check)

        sun_btn = QPushButton("Sun…")
        sun_btn.setMaximumWidth(50)
        sun_btn.setToolTip("Adjust sun azimuth / elevation")
        sun_btn.clicked.connect(self._open_sun_dialog)
        layout.addWidget(sun_btn)

        reset_btn = QPushButton("Reset View")
        reset_btn.setMaximumWidth(80)
        reset_btn.clicked.connect(self.reset_camera)
        layout.addWidget(reset_btn)

        layout.addWidget(QLabel(" | "))

        # Export Image button
        export_img_btn = QPushButton("Export Image")
        export_img_btn.setMaximumWidth(100)
        export_img_btn.clicked.connect(self.export_current_image)
        layout.addWidget(export_img_btn)

        # Export Animation button
        export_anim_btn = QPushButton("Export Animation")
        export_anim_btn.setMaximumWidth(120)
        export_anim_btn.clicked.connect(self.show_export_animation_dialog)
        layout.addWidget(export_anim_btn)

        # Export Data button
        export_data_btn = QPushButton("Export Data")
        export_data_btn.setMaximumWidth(100)
        export_data_btn.clicked.connect(self.show_export_data_dialog)
        layout.addWidget(export_data_btn)

    # -- terrain preview ----------------------------------------------------
    #
    # NOT COVERED BY ANY TEST. See PyVistaWidget.show_mesh_preview: this widget
    # cannot be constructed headless. The decisions are in
    # `gui_core.mesh_preview` and are tested there; this is the wiring.

    def preview_terrain(self, config, project_root=None):
        """Show the mesh a configuration would run on, colored by elevation.

        Returns the sentence it put in the pane, so a caller can log it. Never
        raises: it is reached from a configuration-changed slot.

        A loaded RESULTS directory wins. Drawing a bare terrain over a
        temperature field would leave the field selector, the timestep slider
        and the color bar all describing a mesh that is no longer on screen --
        so the preview declines and says why, rather than silently replacing
        what the user is looking at.
        """
        try:
            from gui_core.mesh_preview import mesh_preview_for_config
            if self.results_dir:
                message = (
                    "A results directory is loaded, so the terrain preview was "
                    "not drawn over it. Close or reload results to preview a "
                    "mesh.")
                self._set_preview_note(message)
                return message
            preview = mesh_preview_for_config(config, project_root)
            message = self.view_3d.show_mesh_preview(preview)
            self._set_preview_note(message or preview.summary())
            if not message:
                self.tab_widget.setCurrentWidget(self.view_3d)
            return message or preview.summary()
        except Exception as exc:                             # noqa: BLE001
            message = (f"The terrain preview failed ({type(exc).__name__}: "
                       f"{exc}). The configuration is unaffected.")
            try:
                self._set_preview_note(message)
            except Exception:                                # noqa: BLE001
                pass
            return message

    def _set_preview_note(self, text):
        note = getattr(self, "preview_note", None)
        if note is None:
            return
        note.setText(str(text or ""))
        note.setVisible(bool(text))

    def _update_field_note(self, field_key):
        """Put the off-scale legend for `field_key` in the pane, or clear it.

        The color it names is the one `off_scale_above_color` chose for the
        colormap CURRENTLY in effect for this field -- including the user's own
        override from the Colormap dropdown -- so the sentence and the screen
        cannot disagree, and its complaint about a colormap that swallows every
        candidate is carried through rather than dropped.

        Returns the sentence it showed (``''`` when there is none), so a caller
        -- and a test -- can read the decision rather than re-deriving it.
        """
        note = getattr(self, "field_note", None)
        above, warning = (OFF_SCALE_ABOVE_CANDIDATES[0][0], "")
        # ⛔ SHADOW_DEPTH_FIELDS, not SHADOW_SENTINEL_FIELDS. The two fraction
        # fields are bounded [0, 1] and cannot hold +inf, so nothing of theirs
        # is ever drawn above range -- and `off_scale_above_color`'s complaint
        # is about a population they do not have. Under a colormap that
        # swallows every candidate, running it here put "deeper than the search
        # ceiling is drawn in magenta" into the legend of a field that has no
        # such facets.
        if field_key in SHADOW_DEPTH_FIELDS:
            above, warning = off_scale_above_color(
                self.view_3d.get_effective_cmap(field_key))
        text = sentinel_legend(
            field_key, shadow_depth_ceiling_m(self.results_dir),
            above_color=above, warning=warning)
        # A field with no sentinel can still need a sentence: the indirect
        # bands are absorbed flux where their neighbors are incident. `or`,
        # not an append -- no field has both today, and a pane holding two
        # unrelated paragraphs is a pane nobody reads.
        text = text or field_convention_note(field_key)
        # ⭐ And the one complaint the fixed NaN color cannot make for itself.
        # Appended, not substituted: the sentinel legend still has to say what
        # the colors MEAN before this says they are hard to tell apart.
        if field_key in NAN_BEARING_FIELDS:
            collision = nan_color_collision_warning(
                self.view_3d.get_effective_cmap(field_key))
            if collision:
                text = f"{text} {collision}".strip()
        if note is not None:
            note.setText(text)
            note.setVisible(bool(text))
        return text

    def load_results(self, results_dir):
        """Load simulation results"""
        self.results_dir = results_dir
        self._set_preview_note("")

        # Detect available fields
        self._populate_field_combo(results_dir)

        # Load into 3D view
        self.view_3d.load_results(results_dir)

        # Load into 2D plots
        self.plots_2d.load_results(results_dir, self.view_3d.mesh_data)

        # Update slider range
        n_timesteps = len(self.view_3d.timesteps)
        self.timestep_slider.setMaximum(max(0, n_timesteps - 1))
        self.timestep_label.setText(f"0/{n_timesteps}")

    def _populate_field_combo(self, results_dir):
        """Detect available data fields and populate the field selector.

        Two sources are scanned:
          1. Per-timestep series:  <results>/<field_key>/<field_key>_NNNN.bin
          2. Statistics:           <results>/statistics/<field_key>.bin
        Items from (1) come first; if both are present we still list both
        so the user can switch between the time-series view and the
        annual aggregates.
        """
        from pathlib import Path
        results_path = Path(results_dir)

        self.field_combo.blockSignals(True)
        self.field_combo.clear()

        available = []
        stats_dir = results_path / "statistics"

        for field_key, (label, _cmap) in self.FIELD_INFO.items():
            if self.is_stats_field(field_key):
                # Statistics file: <results>/statistics/<key>.bin
                bin_path = stats_dir / f"{field_key}.bin"
                if bin_path.is_file():
                    available.append((field_key, label))
            else:
                # Per-timestep series: <results>/<key>/<key>_NNNN.bin
                field_dir = results_path / field_key
                if field_dir.is_dir() and any(field_dir.glob("*.bin")):
                    available.append((field_key, label))

        for field_key, label in available:
            self.field_combo.addItem(label, field_key)

        # Select best default: temperature time-series if available; else
        # temperature_max as a fallback statistic; else first field.
        default_idx = 0
        for preferred in ("temperature", "temperature_max"):
            for i, (field_key, _label) in enumerate(available):
                if field_key == preferred:
                    default_idx = i
                    break
            else:
                continue
            break

        if available:
            self.field_combo.setCurrentIndex(default_idx)
            self.view_3d.set_field(available[default_idx][0])
            self._sync_cmap_combo_to_field(available[default_idx][0])
            self._update_field_note(available[default_idx][0])
        else:
            self._update_field_note(None)

        self.field_combo.blockSignals(False)

    def _on_field_combo_changed(self, label_text):
        """Handle field selector change."""
        field_key = self.field_combo.currentData()
        if field_key:
            self.view_3d.set_field(field_key)
            # Sync the colormap dropdown to the new field's effective cmap
            # without triggering set_cmap_override.
            self._sync_cmap_combo_to_field(field_key)
            self._update_field_note(field_key)

    def _sync_cmap_combo_to_field(self, field_key):
        """Update the Colormap dropdown to show the cmap currently in use
        for `field_key`, without firing the change handler."""
        if not hasattr(self, 'cmap_combo') or self.cmap_combo is None:
            return
        cmap = self.view_3d.get_effective_cmap(field_key)
        idx = self.cmap_combo.findText(cmap)
        self.cmap_combo.blockSignals(True)
        if idx >= 0:
            self.cmap_combo.setCurrentIndex(idx)
        else:
            # Cmap is something not in our curated list — add it as a one-off
            # so the user can see what's active. Stays available until the
            # GUI restarts.
            self.cmap_combo.addItem(cmap)
            self.cmap_combo.setCurrentIndex(self.cmap_combo.count() - 1)
        self.cmap_combo.blockSignals(False)

    def _on_cmap_combo_changed(self, cmap):
        """Handle Colormap dropdown change — apply override for the
        currently-selected field and redraw."""
        if not cmap:
            return
        field_key = self.field_combo.currentData()
        if field_key:
            self.view_3d.set_cmap_override(field_key, cmap)
            # The off-scale sentinel color is chosen against the ramp, so it
            # can move when the ramp does; the legend must move with it or it
            # names a color that is no longer on screen.
            self._update_field_note(field_key)
            if self.view_3d.current_timestep >= 0:
                self.view_3d.load_timestep_data(self.view_3d.current_timestep)

    def set_timestep(self, index):
        """Set current timestep"""
        self.timestep_slider.setValue(index)

    def set_field(self, field_name):
        """Set data field to visualize.

        Called both from the panel's own field_combo (via
        _on_field_combo_changed → view_3d.set_field) and from external
        callers such as the main toolbar's field selector. In the
        external case we must keep the panel's own field_combo and
        cmap_combo in sync, otherwise the user-facing colormap dropdown
        ends up writing overrides to the wrong field.
        """
        self.view_3d.set_field(field_name)
        # Sync the internal field_combo (signals blocked so we don't
        # re-enter this code path).
        if hasattr(self, 'field_combo') and self.field_combo is not None:
            for i in range(self.field_combo.count()):
                if self.field_combo.itemData(i) == field_name:
                    if self.field_combo.currentIndex() != i:
                        self.field_combo.blockSignals(True)
                        self.field_combo.setCurrentIndex(i)
                        self.field_combo.blockSignals(False)
                    break
        # Sync the colormap dropdown to this field's effective cmap.
        if hasattr(self, 'cmap_combo') and self.cmap_combo is not None:
            self._sync_cmap_combo_to_field(field_name)
        self._update_field_note(field_name)

    def on_timestep_slider_changed(self, value):
        """Handle timestep slider change"""
        self.view_3d.set_timestep(value)
        self.plots_2d.set_timestep(value)  # Update 2D plots as well
        n_total = len(self.view_3d.timesteps)
        self.timestep_label.setText(f"{value}/{n_total}")

    def toggle_playback(self):
        """Toggle animation playback"""
        if self.is_playing:
            self.animation_timer.stop()
            self.play_btn.setText("Play")
            self.is_playing = False
        else:
            self.animation_timer.start(200)  # 5 fps
            self.play_btn.setText("Pause")
            self.is_playing = True

    def advance_timestep(self):
        """Advance to next timestep in animation"""
        current = self.timestep_slider.value()
        max_val = self.timestep_slider.maximum()
        if current < max_val:
            self.timestep_slider.setValue(current + 1)
        else:
            self.timestep_slider.setValue(0)  # Loop

    def on_show_edges_changed(self, state):
        """Handle show edges checkbox"""
        pass

    # ---------- Synthetic sun lighting callbacks --------------------------

    def _on_sun_light_toggled(self, state):
        """Toolbar 'Sun Light' checkbox -> apply or revert lighting."""
        enabled = bool(state)
        # Use whatever az/el the view_3d has stored (latitude-default on
        # first mesh load, or the user's most recent dialog choice).
        self.view_3d.set_synthetic_lighting(
            enabled,
            azimuth_deg=self.view_3d.sun_azimuth_deg,
            elevation_deg=self.view_3d.sun_elevation_deg)

    def _open_sun_dialog(self):
        """Open a small dialog to adjust sun azimuth/elevation/intensity
        and reset to the latitude-derived default. Live preview: every
        slider change re-renders immediately."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Sun Lighting")
        layout = QGridLayout(dlg)

        v = self.view_3d
        # Mesh-mean-latitude readout for context.
        info_lbl = QLabel(
            f"Mesh mean latitude: {v.mesh_mean_lat_deg:+.2f}°   "
            f"(typical noon elevation: {max(0.0, 90.0 - abs(v.mesh_mean_lat_deg)):.1f}°)")
        layout.addWidget(info_lbl, 0, 0, 1, 3)

        enable_cb = QCheckBox("Enable synthetic sun light")
        enable_cb.setChecked(v.synthetic_lighting_enabled)
        layout.addWidget(enable_cb, 1, 0, 1, 3)

        # Azimuth: 0 = local north (toward geographic north pole projection),
        # 90 = east, 180 = south, 270 = west. Range 0..360°.
        layout.addWidget(QLabel("Azimuth (°N→E):"), 2, 0)
        az_spin = QDoubleSpinBox()
        az_spin.setRange(0.0, 360.0)
        az_spin.setSingleStep(5.0)
        az_spin.setDecimals(1)
        az_spin.setValue(v.sun_azimuth_deg)
        layout.addWidget(az_spin, 2, 1)
        az_slider = QSlider(Qt.Orientation.Horizontal)
        az_slider.setRange(0, 360)
        az_slider.setValue(int(round(v.sun_azimuth_deg)))
        layout.addWidget(az_slider, 2, 2)

        # Elevation: 0 = horizon, 90 = zenith. Allow slight negative
        # values so the user can investigate sub-horizon geometries.
        layout.addWidget(QLabel("Elevation (° above horizon):"), 3, 0)
        el_spin = QDoubleSpinBox()
        el_spin.setRange(-10.0, 90.0)
        el_spin.setSingleStep(1.0)
        el_spin.setDecimals(1)
        el_spin.setValue(v.sun_elevation_deg)
        layout.addWidget(el_spin, 3, 1)
        el_slider = QSlider(Qt.Orientation.Horizontal)
        el_slider.setRange(-10, 90)
        el_slider.setValue(int(round(v.sun_elevation_deg)))
        layout.addWidget(el_slider, 3, 2)

        # Sun + ambient intensity.
        layout.addWidget(QLabel("Sun intensity:"), 4, 0)
        sun_spin = QDoubleSpinBox()
        sun_spin.setRange(0.0, 5.0)
        sun_spin.setSingleStep(0.1)
        sun_spin.setDecimals(2)
        sun_spin.setValue(v.sun_intensity)
        layout.addWidget(sun_spin, 4, 1, 1, 2)

        layout.addWidget(QLabel("Ambient fill:"), 5, 0)
        amb_spin = QDoubleSpinBox()
        amb_spin.setRange(0.0, 2.0)
        amb_spin.setSingleStep(0.05)
        amb_spin.setDecimals(2)
        amb_spin.setValue(v.ambient_intensity)
        layout.addWidget(amb_spin, 5, 1, 1, 2)

        # Reset to latitude-derived defaults.
        reset_btn = QPushButton("Use latitude default")
        reset_btn.setToolTip(
            "Reset azimuth/elevation to the typical-noon position for the\n"
            "mesh's mean latitude.")
        layout.addWidget(reset_btn, 6, 0, 1, 3)

        # Close button.
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        bb.rejected.connect(dlg.accept)
        bb.accepted.connect(dlg.accept)
        layout.addWidget(bb, 7, 0, 1, 3)

        # --- live preview wiring (sliders <-> spinboxes <-> view) ----------
        applying = {'busy': False}
        def apply():
            if applying['busy']:
                return
            applying['busy'] = True
            try:
                v.set_synthetic_lighting(
                    enable_cb.isChecked(),
                    azimuth_deg=az_spin.value(),
                    elevation_deg=el_spin.value(),
                    intensity=sun_spin.value(),
                    ambient=amb_spin.value())
                # Keep the toolbar checkbox in sync.
                self.sun_light_check.blockSignals(True)
                self.sun_light_check.setChecked(enable_cb.isChecked())
                self.sun_light_check.blockSignals(False)
            finally:
                applying['busy'] = False

        def link_slider_spin(slider, spin):
            slider.valueChanged.connect(lambda val: spin.setValue(float(val)))
            spin.valueChanged.connect(lambda val: slider.setValue(int(round(val))))
        link_slider_spin(az_slider, az_spin)
        link_slider_spin(el_slider, el_spin)

        enable_cb.toggled.connect(apply)
        az_spin.valueChanged.connect(apply)
        el_spin.valueChanged.connect(apply)
        sun_spin.valueChanged.connect(apply)
        amb_spin.valueChanged.connect(apply)

        def reset_to_default():
            mean_lat = v.mesh_mean_lat_deg
            az_spin.setValue(180.0 if mean_lat > 0 else 0.0)
            el_spin.setValue(max(5.0, 90.0 - abs(mean_lat)))
        reset_btn.clicked.connect(reset_to_default)

        dlg.setMinimumWidth(420)
        dlg.exec()

    def on_face_selected(self, face_id, data):
        """Handle face selection from 3D view"""
        self.face_selected.emit(face_id, data)
        self.plots_2d.show_face_timeseries(face_id, data)
        self.tab_widget.setCurrentWidget(self.plots_2d)

    def on_face_selection_from_plots(self, face_id):
        """Handle face selection request from 2D plots"""
        # Highlight in 3D view
        self.view_3d.highlight_face(face_id)
        # Switch to 3D tab briefly to show selection
        # self.tab_widget.setCurrentWidget(self.view_3d)

    def reset_camera(self):
        """Reset camera to default view"""
        self.view_3d.reset_camera()

    def export_image(self, filename):
        """Export current visualization"""
        self.view_3d.export_image(filename)

    def refresh(self):
        """Refresh visualization"""
        self.view_3d.refresh()
        self.plots_2d.update_plots()

    def export_current_image(self):
        """Export current 3D view as image"""
        if not self.view_3d.mesh or not self.view_3d.results_dir:
            QMessageBox.warning(self, "No Data", "Please load simulation results first.")
            return

        # Get current field and timestep for default filename
        field = self.view_3d.current_field
        timestep = self.view_3d.current_timestep
        default_name = f"ptm3d_{field}_t{timestep:04d}.png"

        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Image",
            str(Path.cwd() / default_name),
            "PNG Images (*.png);;JPEG Images (*.jpg *.jpeg);;TIFF Images (*.tiff *.tif);;All Files (*)"
        )

        if not filename:
            return

        try:
            self.view_3d.export_image(filename)
            QMessageBox.information(self, "Export Complete",
                                   f"Image exported to:\n{filename}")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed",
                               f"Failed to export image:\n{str(e)}")

    def show_export_animation_dialog(self):
        """Show dialog to configure and export animation"""
        if not self.view_3d.mesh or not self.view_3d.results_dir:
            QMessageBox.warning(self, "No Data", "Please load simulation results first.")
            return

        n_timesteps = len(self.view_3d.timesteps)
        if n_timesteps == 0:
            QMessageBox.warning(self, "No Timesteps", "No timesteps available for animation.")
            return

        # Create export dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Export Animation")
        dialog.setMinimumWidth(400)
        layout = QVBoxLayout(dialog)

        # Format selection
        format_group = QGroupBox("Animation Format")
        format_layout = QGridLayout()

        format_layout.addWidget(QLabel("Format:"), 0, 0)
        format_combo = QComboBox()
        format_combo.addItems(["GIF (.gif)", "MP4 Video (.mp4)"])
        format_layout.addWidget(format_combo, 0, 1)

        format_layout.addWidget(QLabel("FPS:"), 1, 0)
        fps_spin = QSpinBox()
        fps_spin.setRange(1, 60)
        fps_spin.setValue(5)
        format_layout.addWidget(fps_spin, 1, 1)

        format_group.setLayout(format_layout)
        layout.addWidget(format_group)

        # Info label
        info_label = QLabel(f"Animation will include {n_timesteps} frames\n"
                           f"Current field: {self.view_3d.current_field}")
        info_label.setStyleSheet("color: #888888; font-style: italic;")
        layout.addWidget(info_label)

        # Note about MP4
        note_label = QLabel("Note: MP4 export requires ffmpeg to be installed.")
        note_label.setStyleSheet("color: #ff9800; font-size: 11px;")
        layout.addWidget(note_label)

        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        # Get selected format
        format_idx = format_combo.currentIndex()
        format_type = "gif" if format_idx == 0 else "mp4"
        fps = fps_spin.value()

        # Get output filename
        default_name = f"ptm3d_animation_{self.view_3d.current_field}"
        if format_type == "gif":
            file_filter = "GIF Images (*.gif);;All Files (*)"
            default_name += ".gif"
        else:
            file_filter = "MP4 Videos (*.mp4);;All Files (*)"
            default_name += ".mp4"

        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Animation",
            str(Path.cwd() / default_name),
            file_filter
        )

        if not filename:
            return

        # Stop playback if running
        if self.is_playing:
            self.toggle_playback()

        # Show progress dialog
        progress = QProgressDialog("Exporting animation...", "Cancel", 0, n_timesteps, self)
        progress.setWindowTitle("Exporting")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.show()

        was_cancelled = False

        def update_progress(current, total):
            nonlocal was_cancelled
            progress.setValue(current)
            progress.setLabelText(f"Rendering frame {current}/{total}...")
            QApplication.processEvents()
            if progress.wasCanceled():
                was_cancelled = True

        # Export animation
        success = self.view_3d.export_animation(
            filename,
            format_type=format_type,
            fps=fps,
            progress_callback=update_progress
        )

        progress.close()

        if was_cancelled:
            QMessageBox.information(self, "Canceled", "Animation export was canceled.")
            # Try to delete partial file
            try:
                Path(filename).unlink(missing_ok=True)
            except Exception:
                pass
        elif success:
            QMessageBox.information(self, "Export Complete",
                                   f"Animation exported to:\n{filename}")
        else:
            QMessageBox.critical(self, "Export Failed",
                               "Failed to export animation.\n"
                               "For MP4 format, ensure ffmpeg is installed.")

    def show_export_data_dialog(self):
        """Show dialog for exporting data in GeoTIFF/HDF5 formats"""
        if not self.view_3d.mesh or not self.view_3d.results_dir:
            QMessageBox.warning(self, "No Data", "Please load simulation results first.")
            return

        if not self.view_3d.timesteps:
            QMessageBox.warning(self, "No Timesteps", "No timestep data available to export.")
            return

        dialog = ExportDataDialog(
            results_dir=str(self.view_3d.results_dir),
            mesh_data=self.view_3d.mesh_data,
            timesteps=self.view_3d.timesteps,
            parent=self
        )
        dialog.exec()

    def cleanup(self):
        """Clean up resources"""
        self.animation_timer.stop()
        self.view_3d.cleanup()
