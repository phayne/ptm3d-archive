"""Figures from a run directory: the renderers, the colour scales, four plots.

The tutorial notebooks carry two byte-identical copies of an 84-line
orthographic globe renderer and two copies of a 69-line style-and-normalization
block. Nothing else in the tree has either. This module is where they live now,
plus the four canned plots that cover what those notebooks actually draw.

Two halves, and the split matters:

*Renderers* (:func:`globe`, :func:`flux_norm`, :func:`sym_norm`,
:func:`clip_extend`, :func:`vertex_values_from_faces`) take arrays and an axes.
They know nothing about run directories and are the pieces a figure script
composes by hand.

*Canned plots* (:func:`plot_diurnal`, :func:`plot_map`, :func:`plot_globes`,
:func:`plot_histogram`) take a :class:`ptm.results.Results` and return a
figure. They are the one-liners, and each is also reachable as a method on
``Results``.

Matplotlib only -- no cartopy, no PyVista. Axis labels come from the run's own
:attr:`~ptm.results.Results.units_style`, so a figure made from a Results
configured for the GUI's ``W/m²`` spelling is labelled that way.

Figure convention, project-wide: no data tables, no annotation boxes and no
notes drawn on a figure. Numbers belong in the caller's own output.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "clip_extend",
    "colorbar",
    "default_norm",
    "face_unit_vectors",
    "flux_norm",
    "globe",
    "globe_grid",
    "mesh_angular_radius",
    "panel_label",
    "plot_diurnal",
    "plot_globes",
    "plot_histogram",
    "plot_map",
    "plot_shape",
    "plot_sweep_curves",
    "plot_sweep_residuals",
    "scatter_map",
    "shape",
    "styled",
    "sweep_colors",
    "sym_norm",
    "vertex_values_from_faces",
    "view_basis",
]

#: Sequential, perceptually uniform, monotone in lightness, survives greyscale.
#: Never a rainbow: its uneven lightness invents structure that is not there.
CMAP_FIELD = "inferno"
#: Diverging, for a signed difference drawn about zero.
CMAP_DIFF = "RdBu_r"

_GRATICULE_COLOR = "#7b7a72"
_LIMB_COLOR = "#52514e"


def styled(variant=None):
    """Apply :func:`ptm.pubstyle.apply_style`, once, and return matplotlib.pyplot.

    Every plot below calls this. It is idempotent and importing this module
    does not apply anything, so a caller that has its own rcParams can simply
    not use the canned plots.
    """
    import matplotlib.pyplot as plt
    from . import pubstyle
    pubstyle.apply_style(variant)
    return plt


# --- colour scales ----------------------------------------------------------

def flux_norm(values, lo_pct=1.0, hi_pct=100.0):
    """A colour scale linear in ``T**4`` between two percentiles of ``values``.

    Radiative equilibrium makes surface temperature the fourth root of absorbed
    flux, so a temperature field is a compressed picture of the flux field.
    Colouring by ``T**4`` undoes exactly that compression; the colorbar still
    reads in kelvin, its ticks are simply not evenly spaced.

    Uses ``FuncNorm``. On matplotlib older than 3.4, which has no ``FuncNorm``,
    it falls back to ``PowerNorm(4)`` -- which stretches the top of the range
    in the same direction but is not exactly linear in ``T**4`` -- and says so
    through :mod:`warnings` rather than substituting silently.
    """
    values = np.asarray(values, dtype=np.float64)
    vmin = float(np.percentile(values, lo_pct))
    vmax = float(np.percentile(values, hi_pct))
    try:
        from matplotlib.colors import FuncNorm
    except ImportError:                                   # matplotlib < 3.4
        import warnings
        import matplotlib as mpl
        from matplotlib.colors import PowerNorm
        warnings.warn(
            f"matplotlib {mpl.__version__} has no FuncNorm (added in 3.4); "
            f"using PowerNorm(gamma=4), which is not exactly linear in T**4",
            stacklevel=2)
        return PowerNorm(4.0, vmin=vmin, vmax=vmax)
    return FuncNorm((lambda t: np.clip(t, 0.0, None) ** 4,
                     lambda y: np.clip(y, 0.0, None) ** 0.25),
                    vmin=vmin, vmax=vmax)


def _flux_by_default(what, moment='max', step=None):
    """Whether to colour this request linear in ``T**4`` unless told otherwise.

    Only a *peak* temperature statistic. The point of :func:`flux_norm` is to
    undo the fourth-root compression of a field that is everywhere near its own
    radiative equilibrium, and ``temperature_max`` is that field: every face is
    shown at its own local noon.

    A single output step is not, and neither is ``min`` or ``avg``. On a Moon
    snapshot spanning 67 to 385 K, colouring linear in ``T**4`` gives the whole
    night hemisphere -- most of the visible disk -- 2.1% of the colour range,
    so it renders as one flat black. Measured on
    ``quickstart_moon_diurnal`` step 120; the same field on a linear scale
    separates the night side over a third of the bar.

    An explicit ``flux_scaled=`` always wins; this only chooses the default.
    """
    if not isinstance(what, str) or not what.startswith("temperature"):
        return False
    return step is None and moment == 'max'


def default_norm(values, what, moment='max', step=None, flux_scaled=None,
                 norm=None):
    """The colour scale for a request, unless the caller already chose one.

    ``norm`` wins if given; then an explicit ``flux_scaled``; then
    :func:`_flux_by_default`. Every canned plot that colours a per-face field
    resolves its scale here, so ``plot_map``, ``plot_globes``,
    :func:`globe_grid` and ``plot_shape`` cannot drift apart about what a
    temperature snapshot should look like.
    """
    if norm is not None:
        return norm
    if flux_scaled is None:
        flux_scaled = _flux_by_default(what, moment, step)
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    if not finite.size:
        return None
    if flux_scaled:
        return flux_norm(finite)
    from matplotlib.colors import Normalize
    return Normalize(float(finite.min()), float(finite.max()))


def sym_norm(values, pct=99.0):
    """Limits symmetric about zero for a signed field, robust to outliers."""
    from matplotlib.colors import Normalize
    m = float(np.percentile(np.abs(np.asarray(values)), pct))
    if not m > 0:
        m = float(np.max(np.abs(values))) or 1.0
    return Normalize(vmin=-m, vmax=m)


def clip_extend(values, norm):
    """Which ends of ``norm`` clip ``values``: the colorbar's ``extend=``."""
    values = np.asarray(values)
    below = bool(np.any(values < norm.vmin))
    above = bool(np.any(values > norm.vmax))
    return {(False, False): "neither", (True, False): "min",
            (False, True): "max", (True, True): "both"}[(below, above)]


# --- orthographic globe -----------------------------------------------------

def face_unit_vectors(lat_deg, lon_deg):
    """Unit vectors from per-face latitude and longitude in degrees."""
    la = np.radians(np.asarray(lat_deg, dtype=float))
    lo = np.radians(np.asarray(lon_deg, dtype=float))
    return np.stack([np.cos(la) * np.cos(lo),
                     np.cos(la) * np.sin(lo),
                     np.sin(la)], axis=1)


def vertex_values_from_faces(tris, face_values, n_vertices):
    """Average a per-face field onto the mesh vertices.

    Each icosphere vertex is shared by five or six triangles. Raises on a
    vertex that no face references, because the gouraud interpolation below
    assumes a closed mesh and would otherwise divide by zero silently.
    """
    total = np.zeros(n_vertices)
    count = np.zeros(n_vertices)
    for k in range(3):
        np.add.at(total, tris[:, k], face_values)
        np.add.at(count, tris[:, k], 1.0)
    if count.min() == 0:
        raise ValueError(
            "the mesh has a vertex that no face references, so it is not "
            "closed; globe rendering interpolates across vertices and cannot "
            "handle that")
    return total / count


def view_basis(view_lat, view_lon):
    """Orthonormal ``(toward viewer, right, up)`` for an orthographic view."""
    la, lo = np.radians(view_lat), np.radians(view_lon)
    toward = np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo),
                       np.sin(la)])
    right = np.cross([0.0, 0.0, 1.0], toward)
    if np.linalg.norm(right) < 1e-9:          # looking straight down a pole
        right = np.array([0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    return toward, right, np.cross(toward, right)


def _draw_arc(ax, pts, toward, right, up, color, lw):
    depth = pts @ toward
    x, y = pts @ right, pts @ up
    ax.plot(np.where(depth > 0, x, np.nan), np.where(depth > 0, y, np.nan),
            color=color, linewidth=lw, alpha=0.55, zorder=3)


def _graticule(ax, toward, right, up, step=30.0, color=_GRATICULE_COLOR,
               lw=0.4):
    """Parallels and meridians, drawn only where they face the viewer."""
    for lat0 in np.arange(-60.0, 61.0, step):
        lon = np.linspace(-180.0, 180.0, 361)
        _draw_arc(ax, face_unit_vectors(np.full_like(lon, lat0), lon),
                  toward, right, up, color, lw)
    for lon0 in np.arange(-180.0, 180.0, step):
        lat = np.linspace(-90.0, 90.0, 181)
        _draw_arc(ax, face_unit_vectors(lat, np.full_like(lat, lon0)),
                  toward, right, up, color, lw)


def globe(ax, xyz, tris, vertex_values, view_lat, view_lon, cmap=CMAP_FIELD,
          norm=None, title="", graticule_step=30.0, mark=None):
    """Draw an orthographic globe of a per-vertex field on ``ax``.

    Triangles whose three vertices all face the viewer are projected onto the
    view plane and filled by ``tripcolor(..., shading='gouraud')``, which
    interpolates linearly across each triangle. Nothing is resampled onto a
    raster: what is drawn is the mesh's own field.

    ``xyz`` are vertex positions (any length unit -- they are normalized here),
    ``tris`` the ``(n_faces, 3)`` vertex indices, ``mark`` an optional
    ``(lat, lon)`` drawn as a cross when it faces the viewer.

    Returns the ``tripcolor`` artist, for a colorbar.
    """
    import matplotlib.pyplot as plt

    xyz = np.asarray(xyz, dtype=np.float64)
    xyz = xyz / np.linalg.norm(xyz, axis=1, keepdims=True)
    toward, right, up = view_basis(view_lat, view_lon)
    px, py, depth = xyz @ right, xyz @ up, xyz @ toward
    front = np.all(depth[tris] > 0.0, axis=1)
    if not front.any():
        raise ValueError(
            f"no triangle faces the viewer at lat={view_lat}, lon={view_lon}")
    art = ax.tripcolor(px, py, tris[front], vertex_values, shading="gouraud",
                       cmap=cmap, norm=norm, rasterized=True)
    if graticule_step:
        _graticule(ax, toward, right, up, step=graticule_step)
    ax.add_artist(plt.Circle((0, 0), 1.0, fill=False, lw=0.8,
                             color=_LIMB_COLOR, zorder=4))
    if mark is not None:
        m = face_unit_vectors(np.array([mark[0]]), np.array([mark[1]]))[0]
        if m @ toward > 0:
            ax.plot(m @ right, m @ up, marker="+", ms=9, mew=1.6,
                    color="#0b0b0b", zorder=5)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=11)
    return art


# --- a real shape model -------------------------------------------------------
# globe() normalizes the vertices onto a unit sphere, which is the right thing
# for an icosphere of a round body and destroys a spinning top, a contact binary
# or any DAMIT/radar shape. This draws the facets where they actually are.


def _face_normals(vertices, tris):
    """Outward unit normals, one per triangle.

    Oriented by the face centroid rather than by the winding order: shape files
    in this tree come from several archives (DAMIT, the PDS radar models,
    the icosphere generator) and their winding is not uniformly outward. Every
    body drawn this way is star-shaped about its centre of figure, so the
    centroid direction settles the sign.
    """
    corners = vertices[tris]
    normals = np.cross(corners[:, 1] - corners[:, 0],
                       corners[:, 2] - corners[:, 0])
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.where(lengths > 0, lengths, 1.0)
    outward = corners.mean(axis=1) - vertices.mean(axis=0)
    flip = np.sum(normals * outward, axis=1) < 0
    normals[flip] *= -1.0
    return normals


def _view_direction(elev, azim):
    """The unit vector from the body toward the camera, for matplotlib's
    ``view_init(elev, azim)`` convention."""
    e, a = np.radians(elev), np.radians(azim)
    return np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])


def shape(ax, vertices, faces, face_values, cmap=CMAP_FIELD, norm=None,
          elev=20.0, azim=35.0, title="", edgecolor="none", linewidth=0.0,
          cull_backfaces=True, axis_labels=("x", "y", "z (spin axis)"),
          unit=""):
    """Draw a per-face field on the actual facets of a shape model.

    ``ax`` must be a 3-D axes (``fig.add_subplot(projection='3d')``).
    ``vertices`` are in the mesh's own length unit and are *not* rescaled --
    that is the whole difference from :func:`globe`, which normalizes them onto
    a unit sphere and turns Bennu into a ball. The three axes are given equal
    limits and an equal box aspect, so the body's real proportions survive.

    ``cull_backfaces`` drops the facets pointing away from the camera before
    drawing. Matplotlib's 3-D renderer is a painter's algorithm, not a z-buffer,
    so without culling the far side is painted first and every so often shows
    through a near facet. Culling removes that for a star-shaped body; a genuine
    concavity can still mis-sort, and there is no fix for that inside
    matplotlib.

    Drawn with ``antialiased=False``, which is not cosmetic: antialiasing
    blends each facet's edge with the white canvas behind it, and on Bennu's
    2,692 facets that pale hairline covers the body in a wireframe that is not
    in the data. Pass ``edgecolor='k'`` with a ``linewidth`` to draw the facets
    deliberately.

    Returns the ``Poly3DCollection``, which carries the colour mapping, so
    :func:`colorbar` works on it directly.
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces)
    values = np.asarray(face_values, dtype=np.float64)
    if faces.shape[0] != values.shape[0]:
        raise ValueError(
            f"the mesh has {faces.shape[0]} faces but {values.shape[0]} values "
            f"were given")

    keep = np.ones(faces.shape[0], dtype=bool)
    if cull_backfaces:
        keep = _face_normals(vertices, faces) @ _view_direction(elev, azim) > 0
        if not keep.any():
            raise ValueError(
                f"no facet faces the camera at elev={elev}, azim={azim}")

    collection = Poly3DCollection(vertices[faces[keep]], cmap=cmap, norm=norm,
                                  edgecolors=edgecolor, linewidths=linewidth,
                                  antialiased=False)
    collection.set_array(values[keep])
    ax.add_collection3d(collection)

    span = float(np.abs(vertices).max())
    ax.set_xlim(-span, span)
    ax.set_ylim(-span, span)
    ax.set_zlim(-span, span)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    if axis_labels is not None:
        suffix = f" ({unit})" if unit else ""
        for setter, name in zip((ax.set_xlabel, ax.set_ylabel, ax.set_zlabel),
                                axis_labels):
            setter(f"{name}{suffix}")
    if title:
        ax.set_title(title, fontsize=11)
    return collection


# --- resolving what to plot -------------------------------------------------

def mesh_angular_radius(directions):
    """Half-angle of the cone containing every face direction, in degrees.

    ``directions`` are unit vectors -- :attr:`ptm.results.Results.face_unit_vectors`.
    About 90 for a hemisphere and 180 for a closed body; a few degrees for a
    site patch. Latitude and longitude spans do not measure this: the fig-8
    polar core spans 3 degrees of latitude and 119 of longitude while covering
    a 1.6-degree cap, because longitudes fan out at a pole.
    """
    directions = np.asarray(directions, dtype=np.float64)
    axis = directions.mean(axis=0)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:                      # faces surround the center evenly
        return 180.0
    cosines = directions @ (axis / norm)
    return float(np.degrees(np.arccos(np.clip(cosines.min(), -1.0, 1.0))))


def _require_global_mesh(run, min_angular_radius_deg=30.0):
    """Refuse to draw a globe of a mesh that covers a small patch of one.

    A site mesh -- an A3GT crater, the fig-8 core+ring -- projects to a speck
    near the limb, or to nothing at all from three of the four default views.
    That is a correct rendering of the wrong request, and it looks exactly like
    a rendering failure, so it fails here instead.
    """
    radius = mesh_angular_radius(run.face_unit_vectors)
    if radius < min_angular_radius_deg:
        raise ValueError(
            f"this mesh covers a cap of angular radius {radius:.1f} deg -- it "
            f"is a site patch, not a body, and an orthographic globe of it is "
            f"a speck on an empty disk. Use plot_map(), which projects the "
            f"patch itself.")


def _values_and_label(run, what, moment, step):
    """Turn a ``(field, moment|step)`` request into ``(values, label)``.

    ``step`` and ``moment`` are alternatives: a step asks for one output
    timestep, a moment for a statistic over all of them. ``step`` wins when
    both are given, which is why every caller's ``step`` documentation says so
    -- ``moment`` has a default, so a sentinel could not tell an explicit
    ``moment='max'`` from the default one.

    ``what`` may be an array, in which case the caller supplies the label.
    """
    if isinstance(what, np.ndarray):
        return what, ""
    field = what or "temperature"
    if step is not None:
        values = run.field(field, step)
        return values, f"{run.axis_label(field)}"
    values = run.stat(field, moment)
    return values, run.axis_label(field, moment)


def _nonlinear_ticks(norm, n=6):
    """Tick values evenly spaced *along the bar* for a non-linear norm.

    A ``FuncNorm`` linear in ``T**4`` maps equal temperature steps to very
    unequal bar positions, so matplotlib's default locator -- which chooses
    round numbers in data space -- crowds several labels into the bottom
    millimetre of the colorbar and prints them on top of each other. Placing
    the ticks uniformly in normalized space and rounding the *labels* instead
    puts one legible tick per interval.

    Returns ``None`` for a linear norm, where the default locator is better.
    """
    inverse = getattr(norm, "inverse", None)
    if inverse is None:
        return None
    from matplotlib.colors import Normalize
    if type(norm) is Normalize:
        return None
    try:
        values = np.asarray(inverse(np.linspace(0.0, 1.0, n)), dtype=float)
    except Exception:                                          # noqa: BLE001
        return None
    if not np.all(np.isfinite(values)):
        return None
    step = max(abs(values[-1] - values[0]) / (n - 1), 1e-12)
    decimals = max(0, 1 - int(np.floor(np.log10(step))))
    rounded = np.unique(np.round(values, decimals))
    return rounded if rounded.size >= 2 else None


def colorbar(fig, artist, ax, label="", values=None, norm=None, **kwargs):
    """A colorbar with legible ticks and the correct clipping arrows.

    ``ax`` may be one axes or a sequence of them, and a sequence is how a
    shared colorbar is made: one bar beside the whole row or column, sized by
    the group rather than by one panel.

    Two things this does that ``fig.colorbar`` does not. It sets ``extend``
    from the data, so the arrows appear exactly when values fall outside the
    norm -- pass ``values`` (any array, or the concatenation of several panels'
    arrays) to get that. And on a non-linear norm it places the ticks evenly
    along the *bar* instead of at round numbers in data space: with
    :func:`flux_norm` on a field spanning day and night, matplotlib's default
    locator stacks three labels on top of each other in the bottom millimetre.

    ``**kwargs`` go straight to ``fig.colorbar`` (``fraction``, ``pad``,
    ``orientation``, ``shrink``).
    """
    extend = (clip_extend(values, norm)
              if (norm is not None and values is not None) else "neither")
    bar = fig.colorbar(artist, ax=ax, extend=extend, **kwargs)
    if norm is not None:
        ticks = _nonlinear_ticks(norm)
        if ticks is not None:
            bar.set_ticks(ticks)
    if label:
        bar.set_label(label)
    return bar


# Forwarded rather than imported at module level: ptm.pubstyle imports
# matplotlib on import, and this module deliberately does not.
def panel_label(ax, label, x=0.02, y=0.97, **kwargs):
    """Place a panel label like ``'(a)'`` in the top-left corner of ``ax``.

    See :func:`ptm.pubstyle.panel_label`; this is the same function, reachable
    from the module a figure is already importing.
    """
    from . import pubstyle
    return pubstyle.panel_label(ax, label, x=x, y=y, **kwargs)


# --- unprojected per-face map -------------------------------------------------
# plot_map() regrids onto a raster and needs scipy. Every tutorial figure that
# shows a field over the whole body instead scatters the face centers directly:
# no interpolation, no extra dependency, and each dot is one solved face. That
# is twelve lines of axis setup repeated six times across the notebooks, and it
# is the single most-copied plotting block in them.


def _face_marker_size(n_faces):
    """A dot size that tiles the map without merging into a slab.

    Scaled so the dots cover a roughly constant fraction of a 7-inch-wide
    equirectangular axes: 16 pt^2 at ico3 (1,280 faces), 7 at ico4, 5 at ico5,
    floored at 1 so a 500k-face mesh still draws something.
    """
    return float(np.clip(16.0 * (1280.0 / max(n_faces, 1)) ** 0.5, 1.0, 24.0))


def scatter_map(ax, lat, lon, values, cmap=CMAP_FIELD, norm=None, size=None,
                title="", xlabel=True, ylabel=True, grid=True, **kwargs):
    """Draw a per-face field at its own face centers on ``ax``.

    One dot per face at its centre, no regridding and no scipy: what is drawn
    is the mesh's own values. ``lat``/``lon`` are per-face degrees --
    :attr:`ptm.results.Results.lat` and :attr:`~ptm.results.Results.lon`, in
    that order -- and ``lon`` is on ``[-180, 180]``, which is what the axis is
    set to.

    ``xlabel``/``ylabel`` may be ``False`` to leave the axis bare, which is
    what the inner panels of a shared-axis grid want. Returns the ``scatter``
    artist, for :func:`colorbar`.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    values = np.asarray(values)
    if lat.shape != lon.shape or lat.shape[0] != values.shape[0]:
        raise ValueError(
            f"lat, lon and values must have one entry per face; got "
            f"{lat.shape}, {lon.shape}, {values.shape}")
    # Argument order is (lat, lon) to match Results.lat / Results.lon and the
    # rest of this package, while scatter's own order is (x=lon, y=lat).
    # Swapping them draws a plausible-looking wrong figure, so it is caught.
    if np.nanmax(np.abs(lat)) > 90.0 + 1e-6:
        raise ValueError(
            f"latitude out of range (max |lat| = {np.nanmax(np.abs(lat)):.1f} "
            f"deg): the arguments are (ax, lat, lon, values), and these look "
            f"transposed")

    if size is None:
        size = _face_marker_size(values.shape[0])
    artist = ax.scatter(lon, lat, c=values, cmap=cmap, norm=norm, s=size,
                        linewidths=0, **kwargs)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xticks(range(-180, 181, 60))
    ax.set_yticks(range(-90, 91, 30))
    if xlabel:
        ax.set_xlabel("Longitude (deg)")
    if ylabel:
        ax.set_ylabel("Latitude (deg)")
    if grid:
        ax.grid(True, linewidth=0.4)
    if title:
        ax.set_title(title, fontsize=11)
    return artist


# --- a grid of globes ---------------------------------------------------------
# plot_globes() draws one field from several views. Every comparison figure in
# the tutorials needs the other axis too: two fields down the rows (temperature
# and what topography changed; albedo and the temperature it produces), or two
# output steps across one row (before ingress and mid-eclipse). Rows share a
# colour scale and a colorbar; columns share a viewpoint.

#: The four views the tutorials use when none are named.
DEFAULT_VIEWS = ((0.0, 0.0, "0 deg lon"), (0.0, 180.0, "180 deg lon"),
                 (90.0, 0.0, "north pole"), (-90.0, 0.0, "south pole"))


def _normalize_views(views):
    views = DEFAULT_VIEWS if views is None else views
    return [(float(v[0]), float(v[1]), (v[2] if len(v) > 2 else ""))
            for v in views]


def _row_values(run, row, n_views):
    """The value array for each view in one row of :func:`globe_grid`.

    ``what`` may be a single field name or array (the same field seen from
    every view), or a sequence of one per view -- which is how a row compares
    two output steps side by side.
    """
    what = row.get("what", "temperature")
    moment, step = row.get("moment", "max"), row.get("step")
    is_sequence = (isinstance(what, (list, tuple))
                   or (isinstance(what, np.ndarray) and what.ndim == 2))
    if not is_sequence:
        values, label = _values_and_label(run, what, moment, step)
        return [values] * n_views, label
    if len(what) != n_views:
        raise ValueError(
            f"this row lists {len(what)} fields but the grid has {n_views} "
            f"views; a per-view list must have exactly one entry per view")
    resolved = [_values_and_label(run, w, moment, step) for w in what]
    labels = [lab for _, lab in resolved if lab]
    return [v for v, _ in resolved], (labels[0] if labels else "")


def _row_norm(row, stacked):
    """One norm per row, so every panel in it is on the same colour scale."""
    return default_norm(stacked, row.get("what"), row.get("moment", "max"),
                        row.get("step"), row.get("flux_scaled"),
                        row.get("norm"))


def globe_grid(run, rows, views=None, save=None, show=False, figsize=None,
               suptitle=None, panel_labels=False, graticule_step=30.0):
    """A grid of orthographic globes: one row per field, one column per view.

    ::

        plots.globe_grid(run,
            rows=[{"what": "temperature", "moment": "max"},
                  {"what": delta, "label": r"$\\Delta T$ (K)",
                   "cmap": plots.CMAP_DIFF, "norm": plots.sym_norm(delta)}],
            views=[(90, 0, "north pole"), (-90, 0, "south pole")])

    Each entry of ``rows`` is a mapping. ``what`` is a field name, an array, or
    a list of either with one entry per view (that is how one row compares two
    output steps). The optional keys are ``moment``, ``step``, ``label``,
    ``cmap``, ``norm``, ``flux_scaled``, ``mark`` (a ``(lat, lon)`` cross) and
    ``titles``.

    Every panel in a row shares one norm and one colorbar, because that is what
    makes a row comparable; different rows are independent. Column titles are
    drawn on the top row only. Returns the figure.
    """
    plt = styled()

    views = _normalize_views(views)
    rows = [dict(r) for r in rows]
    if not rows:
        raise ValueError("globe_grid needs at least one row")

    mesh = run.mesh
    _require_global_mesh(run)

    n_rows, n_cols = len(rows), len(views)
    if figsize is None:
        figsize = _globe_grid_figsize(n_rows, n_cols, bool(suptitle))
    # Constrained layout, not the default: a globe's axes has aspect='equal',
    # so its box shrinks inside whatever cell it is given and the leftover
    # becomes a band of white between the rows -- measured at 0.8 in on a
    # 2x2 grid, which reads as a missing panel. Constrained layout also keeps
    # each row's colorbar the height of that row, and places the suptitle, so
    # the suptitle below must not be given an explicit y.
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False,
                             constrained_layout=True)

    letters = iter("abcdefghijklmnopqrstuvwxyz")
    for i, row in enumerate(rows):
        per_view, default_label = _row_values(run, row, n_cols)
        vertex_fields = [vertex_values_from_faces(mesh["faces"], v,
                                                  mesh["n_vertices"])
                         for v in per_view]
        stacked = np.concatenate(vertex_fields)
        norm = _row_norm(row, stacked)
        cmap = row.get("cmap", CMAP_FIELD)
        titles = row.get("titles")
        artist = None
        for j, (view_lat, view_lon, view_title) in enumerate(views):
            title = (titles[j] if titles is not None
                     else (view_title if i == 0 else ""))
            artist = globe(axes[i][j], mesh["vertices"], mesh["faces"],
                           vertex_fields[j], view_lat, view_lon, cmap=cmap,
                           norm=norm, title=title,
                           graticule_step=graticule_step, mark=row.get("mark"))
            if panel_labels:
                panel_label(axes[i][j], f"({next(letters)})")
        colorbar(fig, artist, list(axes[i]), row.get("label", default_label),
                 stacked, norm, fraction=0.02, pad=0.02)

    if suptitle:
        fig.suptitle(suptitle)
    return _finish(fig, save, show)


#: Inches a globe grid loses off its width to the colorbar, its label and the
#: figure margins. Measured on the 2x2 and 1x4 grids the tutorials draw.
_GLOBE_CBAR_IN = 0.9
#: Inches reserved for the column titles, and for a suptitle when there is one.
_GLOBE_TITLE_IN = 0.35
_GLOBE_SUPTITLE_IN = 0.45


def _globe_grid_figsize(n_rows, n_cols, has_suptitle, column_in=3.4):
    """A canvas on which the square globes fill their cells.

    The width is the free parameter and the height follows from it: an
    aspect='equal' axes shrinks its box to a square inside whatever cell it is
    given, so a cell taller than it is wide turns the surplus into white space
    between the rows -- 0.8 in of it on a 2x2 grid, which reads as a missing
    panel rather than as margin.
    """
    width = column_in * n_cols
    cell = (width - _GLOBE_CBAR_IN) / n_cols
    height = n_rows * cell + _GLOBE_TITLE_IN
    if has_suptitle:
        height += _GLOBE_SUPTITLE_IN
    return (width, height)


def _finish(fig, save=None, show=False, dpi=200):
    """Save and/or show, and return the figure either way."""
    if save is not None:
        fig.savefig(save, dpi=dpi)
    if show:
        import matplotlib.pyplot as plt
        plt.show()
    return fig


# --- canned plots -----------------------------------------------------------

def plot_diurnal(run, field='temperature', face=None, lat=None, lon=None,
                 ax=None, save=None, show=False, color=None, label=None):
    """One face's time series against elapsed hours.

    ``face=`` an index, or ``lat=``/``lon=`` in degrees for the nearest face.
    """
    plt = styled()
    from . import pubstyle

    index = run.resolve_face(face, lat, lon)
    hours, values = run.series(field, face=index)

    if ax is None:
        fig, ax = plt.subplots(figsize=(pubstyle.DOUBLE_W, 0.42 * pubstyle.DOUBLE_W))
    else:
        fig = ax.figure
    ax.plot(hours, values, color=color or pubstyle.COLORS["ptm"], label=label)
    ax.set_xlabel("Elapsed time (hours)")
    ax.set_ylabel(run.axis_label(field))
    ax.set_title(f"Face {index} at {run.lat[index]:.1f}, {run.lon[index]:.1f} deg")
    ax.set_xlim(hours.min(), hours.max())
    if label:
        ax.legend()
    fig.tight_layout()
    return _finish(fig, save, show)


def plot_map(run, what='temperature', moment='max', step=None, resolution=None,
             cmap=CMAP_FIELD, norm=None, flux_scaled=None, linear=True,
             ax=None, save=None, show=False, label=None):
    """A regridded lat/lon (or polar stereographic) map of a per-face field.

    The projection is chosen by :class:`~ptm.results.MeshRegridder` from the
    mesh's own latitude range: equirectangular for a global or mid-latitude
    mesh, polar stereographic for one that lies beyond 70 degrees. Needs scipy.

    ``linear=True`` (the default) uses the regridder's barycentric path, which
    fills every pixel inside the convex hull of the face centers. The
    nearest-neighbour path (``linear=False``) masks any pixel more than 1.5
    grid cells from a face center, which on a coarse global mesh leaves a
    notched, broken-looking polar edge -- correct, but it reads as a bug.

    ``flux_scaled`` colours by :func:`flux_norm` instead of linearly. The
    default follows :func:`_flux_by_default`: peak temperature yes, a single
    step or ``min``/``avg`` no. Passing ``step`` draws that output step and
    ``moment`` is not used.
    """
    plt = styled()
    from . import pubstyle

    values, axis_label = _values_and_label(run, what, moment, step)
    axis_label = label or axis_label
    grid, regridder = run.to_grid(values, resolution=resolution, linear=linear)

    # A regridded map is masked outside the mesh's hull, so the scale comes
    # from the finite pixels -- which is what default_norm() looks at.
    norm = default_norm(grid, what, moment, step, flux_scaled, norm)

    if ax is None:
        fig, ax = plt.subplots(figsize=(pubstyle.DOUBLE_W,
                                        0.5 * pubstyle.DOUBLE_W))
    else:
        fig = ax.figure

    if regridder.projection == 'eqc':
        extent = [regridder.lon_grid[0], regridder.lon_grid[-1],
                  regridder.lat_grid[0], regridder.lat_grid[-1]]
        image = ax.imshow(grid, origin="lower", extent=extent, cmap=cmap,
                          norm=norm, aspect="auto", interpolation="nearest")
        ax.set_xlabel("Longitude (deg)")
        ax.set_ylabel("Latitude (deg)")
    else:
        extent = [regridder.x_grid[0] / 1e3, regridder.x_grid[-1] / 1e3,
                  regridder.y_grid[0] / 1e3, regridder.y_grid[-1] / 1e3]
        image = ax.imshow(grid, origin="lower", extent=extent, cmap=cmap,
                          norm=norm, aspect="equal", interpolation="nearest")
        ax.set_xlabel("Polar stereographic x (km)")
        ax.set_ylabel("Polar stereographic y (km)")

    finite = grid[np.isfinite(grid)]
    colorbar(fig, image, ax, axis_label, finite, norm, pad=0.02, fraction=0.046)
    fig.tight_layout()
    return _finish(fig, save, show)


def plot_globes(run, what='temperature', moment='max', step=None, views=None,
                cmap=CMAP_FIELD, norm=None, flux_scaled=None, mark_subsolar=False,
                save=None, show=False, label=None):
    """A row of orthographic globes of a per-face field.

    One row of :func:`globe_grid`, which is where the drawing lives; use that
    directly for more than one field or more than one output step.

    ``views`` is a sequence of ``(lat, lon)`` or ``(lat, lon, title)``. The
    default is the four the tutorials use: 0 deg, 180 deg, and both poles. With
    ``mark_subsolar=True`` and a ``step``, the sub-solar point is read from
    ``body_positions.csv`` and marked.

    ``flux_scaled`` colours by :func:`flux_norm`; the default follows
    :func:`_flux_by_default` -- peak temperature yes, a single step no, because
    a snapshot spanning day and night has almost all of its faces in the bottom
    two percent of a ``T**4`` bar. Passing ``step`` draws that output step and
    ``moment`` is not used.
    """
    mark = (run.subsolar_latlon(step)
            if (mark_subsolar and step is not None) else None)
    row = {"what": what, "moment": moment, "step": step, "cmap": cmap,
           "norm": norm, "flux_scaled": flux_scaled, "mark": mark}
    if label:
        row["label"] = label
    return globe_grid(run, [row], views=views, save=save, show=show)


def plot_shape(run, what='temperature', moment='max', step=None, views=None,
               cmap=CMAP_FIELD, norm=None, flux_scaled=None, ax=None,
               axes_off=None, save=None, show=False, label=None, figsize=None,
               suptitle=None):
    """The run's own facets, coloured by a per-face field.

    For a body whose shape matters -- a DAMIT model, a radar shape, a contact
    binary. :func:`plot_globes` normalizes the vertices onto a sphere, which is
    correct for an icosphere and wrong for anything else.

    ``views`` is a sequence of ``(elev, azim)`` or ``(elev, azim, title)`` in
    degrees; the default is one view. All views share one colour scale and one
    colorbar. Passing ``ax`` draws a single view on an existing 3-D axes and
    ``views`` is then a single view.

    ``axes_off`` hides the 3-D box, ticks and axis labels. The default is to
    keep them for a single view -- where the km scale is worth having -- and
    drop them for several, because a 3-D axes draws its tick labels outside its
    own bounding box and three of them side by side overprint each other. Set
    it explicitly either way.

    Axis labels carry the mesh's length unit from
    :attr:`ptm.results.Results.mesh_units`, which is settled by the engine's
    own rule rather than assumed.
    """
    plt = styled()

    values, axis_label = _values_and_label(run, what, moment, step)
    axis_label = label or axis_label
    mesh = run.mesh

    if views is None:
        views = [(20.0, 35.0, "")]
    views = [(float(v[0]), float(v[1]), (v[2] if len(v) > 2 else ""))
             for v in views]
    if ax is not None and len(views) != 1:
        raise ValueError("ax= draws one view; pass a single view or drop ax=")
    if axes_off is None:
        axes_off = len(views) > 1

    norm = default_norm(values, what, moment, step, flux_scaled, norm)

    if ax is None:
        if figsize is None:
            figsize = ((4.2 if axes_off else 5.0) * len(views), 4.4)
        fig = plt.figure(figsize=figsize)
        axes = [fig.add_subplot(1, len(views), i + 1, projection="3d")
                for i in range(len(views))]
        # Explicit margins, not tight/constrained layout: neither reserves room
        # for a 3-D axes' tick labels, which are drawn outside its bounding
        # box, and the left panel's z label is clipped against the canvas.
        fig.subplots_adjust(left=(0.02 if axes_off else 0.09),
                            right=0.88, bottom=0.04, top=0.90,
                            wspace=(0.02 if axes_off else 0.24))
    else:
        fig, axes = ax.figure, [ax]

    artist = None
    for panel, (elev, azim, title) in zip(axes, views):
        artist = shape(panel, mesh["vertices"], mesh["faces"], values,
                       cmap=cmap, norm=norm, elev=elev, azim=azim, title=title,
                       unit=run.mesh_units,
                       axis_labels=None if axes_off else ("x", "y",
                                                          "z (spin axis)"))
        if axes_off:
            panel.set_axis_off()
    colorbar(fig, artist, axes, axis_label, values, norm, shrink=0.6, pad=0.02)
    if suptitle:
        fig.suptitle(suptitle)
    return _finish(fig, save, show)


def plot_histogram(run, what='temperature', moment='max', step=None, bins=60,
                   area_weighted=False, ax=None, save=None, show=False,
                   label=None, color=None):
    """Distribution of a per-face field.

    ``area_weighted=True`` weights each face by its area, which is what a map
    shows; the unweighted version is what a face count shows, and on an
    icosphere the two are nearly the same. Areas come from
    :attr:`~ptm.results.Results.face_areas_m2`, so the mesh's length unit is
    already accounted for.

    Passing ``step`` draws that output step and ``moment`` is not used.
    """
    plt = styled()
    from . import pubstyle

    values, axis_label = _values_and_label(run, what, moment, step)
    axis_label = label or axis_label
    weights = run.face_areas_m2 if area_weighted else None

    if ax is None:
        fig, ax = plt.subplots(figsize=(pubstyle.COLUMN_W * 1.6,
                                        pubstyle.COLUMN_W * 1.1))
    else:
        fig = ax.figure
    ax.hist(values, bins=bins, weights=weights,
            color=color or pubstyle.COLORS["ptm"], edgecolor="none")
    ax.set_xlabel(axis_label)
    ax.set_ylabel("Surface area (m$^2$)" if area_weighted else "Faces")
    fig.tight_layout()
    return _finish(fig, save, show)


# --- sweeps -------------------------------------------------------------------
# A sweep axis is an *ordered magnitude*, not a set of unrelated categories, so
# the curves are coloured from a sequential ramp in grid order rather than from
# a categorical cycle: neighbouring parameter values then look neighbouring, the
# ordering survives greyscale, and no curve's colour depends on how many other
# curves are drawn.
#
# The span stops short of viridis' pale end because that end vanishes on a white
# page: measured against the #fcfcfb surface, six curves over [0.15, 0.90] put
# three of them under a 3:1 contrast ratio and the lightest at 1.49:1, whereas
# [0.02, 0.72] leaves one at 2.19:1 and keeps the worst adjacent-pair colour
# separation above the CVD target (deutan OKLab dE 9.1, protan/tritan >= 8.0).
# The legend is always drawn, which is the relief a sub-3:1 mark requires.

CMAP_SWEEP = "viridis"
_SWEEP_RAMP_SPAN = (0.02, 0.72)
#: Above this many curves a legend stops being readable and becomes a colorbar.
SWEEP_LEGEND_MAX = 8


def sweep_colors(n, cmap=CMAP_SWEEP, span=_SWEEP_RAMP_SPAN):
    """``n`` colours from a sequential ramp, in grid order."""
    import matplotlib.pyplot as plt
    ramp = plt.get_cmap(cmap)
    if n == 1:
        return [ramp(0.5)]
    lo, hi = span
    return [ramp(lo + (hi - lo) * i / (n - 1)) for i in range(n)]


def _curve_list(sweep, curves, kwargs):
    """Accept either a Sweep or an explicit list of :class:`ptm.sweep.Curve`."""
    if curves is not None:
        return list(curves)
    if not hasattr(sweep, "curves"):
        raise TypeError(
            "pass a Sweep, or curves=[Curve(...), ...]; got "
            f"{type(sweep).__name__}")
    return sweep.curves(**kwargs)


def _sweep_axis_labels(sweep, xlabel, ylabel, field):
    """Default the axis labels, taking the y label from the run's own catalog.

    ``Results.axis_label`` knows the units and the run's unit spelling, so
    ``temperature`` becomes ``Temperature (K)`` rather than ``Temperature``.
    """
    if xlabel is None:
        xlabel = "Local solar time (h)"
    if ylabel is None:
        ylabel = field.replace("_", " ").title()
        try:
            done = sweep.completed()
            if done:
                ylabel = done[0].results.axis_label(field)
        except Exception:                                       # noqa: BLE001
            pass
    return xlabel, ylabel


def _bare_value_labels(curves, legend_title):
    """With a legend *title* naming the axis, the entries need only the value.

    ``thermal inertia = 50`` six times under a heading that already says
    "Thermal inertia (SI)" is the same word printed seven times.
    """
    if not legend_title:
        return curves
    keys = {k for c in curves for k in c.params}
    if len(keys) != 1:
        return curves
    key = next(iter(keys))
    out = []
    for curve in curves:
        value = curve.params[key]
        text = f"{value:g}" if isinstance(value, float) else str(value)
        out.append(curve._replace(label=text))
    return out


def _sweep_legend(ax, title):
    """A legend beside the axes, never on top of the curves.

    Inside the axes, a six-entry legend lands on the family it describes --
    measured, on the first draw of this figure. Outside costs width and
    collides with nothing.
    """
    return ax.legend(title=title, frameon=False, fontsize="small",
                     loc="upper left", bbox_to_anchor=(1.01, 1.0),
                     borderaxespad=0.0, handlelength=1.6)


def plot_sweep_curves(sweep=None, curves=None, field='temperature',
                      xlabel=None, ylabel=None, title=None, cmap=CMAP_SWEEP,
                      ax=None, save=None, show=False, legend_title=None,
                      **kwargs):
    """One curve per grid point.

    ``sweep`` is a :class:`ptm.sweep.Sweep`; anything in ``**kwargs`` goes to
    :meth:`~ptm.sweep.Sweep.curves`, so ``of=``, ``lat=``/``lon=`` and
    ``field=`` all work here. Alternatively pass ``curves=`` directly.

    Up to :data:`SWEEP_LEGEND_MAX` curves get a legend; beyond that the ramp
    becomes a colorbar, because a fifteen-entry legend is a data table and this
    project does not put data tables on figures.
    """
    plt = styled()
    from . import pubstyle

    kwargs.setdefault("field", field)
    curves = _curve_list(sweep, curves, kwargs)
    if not curves:
        raise RuntimeError("no completed points to plot")
    curves = _bare_value_labels(curves, legend_title)
    colors = sweep_colors(len(curves), cmap=cmap)
    xlabel, ylabel = _sweep_axis_labels(sweep, xlabel, ylabel, field)

    if ax is None:
        fig, ax = plt.subplots(figsize=(pubstyle.DOUBLE_W,
                                        0.45 * pubstyle.DOUBLE_W))
    else:
        fig = ax.figure

    for curve, color in zip(curves, colors):
        ax.plot(curve.x, curve.y, color=color, label=curve.label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.set_xlim(min(float(np.min(c.x)) for c in curves),
                max(float(np.max(c.x)) for c in curves))

    if len(curves) <= SWEEP_LEGEND_MAX:
        _sweep_legend(ax, legend_title)
    else:
        _sweep_colorbar(fig, ax, curves, cmap, legend_title)
    fig.tight_layout()
    return _finish(fig, save, show)


def _sweep_colorbar(fig, ax, curves, cmap, label):
    """A colorbar standing in for an unreadably long legend.

    Only meaningful when the grid has one numeric axis; with more than one the
    curves are ordered but not on a single scale, so the ramp is left
    unlabelled rather than given a misleading axis.
    """
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import ListedColormap, Normalize

    keys = {k for c in curves for k in c.params}
    numeric = [k for k in sorted(keys)
               if all(isinstance(c.params.get(k), (int, float))
                      and not isinstance(c.params.get(k), bool)
                      for c in curves)]
    if len(numeric) != 1:
        return None
    key = numeric[0]
    values = np.array([float(c.params[key]) for c in curves])
    lo, hi = _SWEEP_RAMP_SPAN
    norm = Normalize(vmin=values.min(), vmax=values.max())
    # The ramp is sampled over a sub-span, so the bar must be too, or the bar
    # shows colours no curve was drawn in.
    sub = plt.get_cmap(cmap)(np.linspace(lo, hi, 256))
    mappable = ScalarMappable(norm=norm, cmap=ListedColormap(sub))
    bar = fig.colorbar(mappable, ax=ax, fraction=0.04, pad=0.02)
    bar.set_label(label or key.split(".")[-1].replace("_", " "))
    return bar


def plot_sweep_residuals(sweep=None, obs_x=None, obs_y=None, obs_sigma=None,
                         curves=None, field='temperature', xlabel=None,
                         ylabel=None, obs_label="observed", cmap=CMAP_SWEEP,
                         title=None, save=None, show=False, legend_title=None,
                         **kwargs):
    """Model curves against observations, with model-minus-observation below.

    ``obs_x``/``obs_y`` are the observations, ``obs_sigma`` their uncertainty
    (a scalar or per-point). Each model curve is interpolated onto ``obs_x``
    for the residual panel; a model whose x-range does not cover an
    observation contributes nothing there rather than an extrapolation, and
    says so.

    Two stacked panels sharing one x-axis -- never two y-scales on one axes.
    """
    plt = styled()
    from . import pubstyle

    if obs_x is None or obs_y is None:
        raise ValueError("plot_sweep_residuals needs obs_x and obs_y")
    obs_x = np.asarray(obs_x, dtype=float)
    obs_y = np.asarray(obs_y, dtype=float)

    kwargs.setdefault("field", field)
    curves = _curve_list(sweep, curves, kwargs)
    if not curves:
        raise RuntimeError("no completed points to plot")
    curves = _bare_value_labels(curves, legend_title)
    colors = sweep_colors(len(curves), cmap=cmap)
    xlabel, ylabel = _sweep_axis_labels(sweep, xlabel, ylabel, field)

    fig, (top, bottom) = plt.subplots(
        2, 1, sharex=True, figsize=(pubstyle.DOUBLE_W, 0.75 * pubstyle.DOUBLE_W),
        gridspec_kw={"height_ratios": [2.2, 1.0]})

    # A marker per residual is right for a handful of observations and is
    # clutter for hundreds: at 192 samples x 5 curves the markers merge into a
    # band and hide the trend they were meant to show.
    marker = "o" if obs_x.size <= 40 else None

    for curve, color in zip(curves, colors):
        top.plot(curve.x, curve.y, color=color, label=curve.label)
        x = np.asarray(curve.x, dtype=float)
        order = np.argsort(x)
        inside = (obs_x >= x[order][0]) & (obs_x <= x[order][-1])
        if not inside.all():
            import warnings
            warnings.warn(
                f"{(~inside).sum()} observation(s) lie outside the x-range of "
                f"curve {curve.label!r}; they are left out of its residual "
                f"rather than extrapolated", stacklevel=2)
        model = np.interp(obs_x[inside], x[order],
                          np.asarray(curve.y, dtype=float)[order])
        bottom.plot(obs_x[inside], model - obs_y[inside], color=color,
                    marker=marker, markersize=3.5, linewidth=1.2)

    ink = pubstyle.COLORS["neutral"]
    if obs_sigma is None:
        top.plot(obs_x, obs_y, linestyle="none", marker="o", markersize=4.5,
                 color=ink, label=obs_label, zorder=5)
    else:
        top.errorbar(obs_x, obs_y, yerr=obs_sigma, linestyle="none", marker="o",
                     markersize=4.5, color=ink, ecolor=ink, elinewidth=1.0,
                     capsize=2.5, label=obs_label, zorder=5)
        bottom.fill_between(obs_x, -np.asarray(obs_sigma) * np.ones_like(obs_x),
                            np.asarray(obs_sigma) * np.ones_like(obs_x),
                            color=pubstyle.COLORS["envelope"], zorder=0)
    bottom.axhline(0.0, color=ink, linewidth=1.0, zorder=1)

    top.set_ylabel(ylabel)
    if title:
        top.set_title(title)
    bottom.set_xlabel(xlabel)
    # The residual carries the observable's own units, so the label borrows
    # whatever is in parentheses on the top panel rather than inventing one.
    units = ylabel[ylabel.rfind("(") + 1:ylabel.rfind(")")] \
        if "(" in ylabel and ylabel.rfind(")") > ylabel.rfind("(") else ""
    bottom.set_ylabel(f"Model $-$ obs ({units})" if units else "Model $-$ obs")
    bottom.set_xlim(float(obs_x.min()), float(obs_x.max()))

    if len(curves) < SWEEP_LEGEND_MAX:
        _sweep_legend(top, legend_title)
    else:
        _sweep_colorbar(fig, [top, bottom], curves, cmap, legend_title)
        handles, labels = top.get_legend_handles_labels()
        top.legend(handles=handles[-1:], labels=labels[-1:], frameon=False,
                   fontsize="small", loc="upper left",
                   bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
    fig.tight_layout()
    return _finish(fig, save, show)
