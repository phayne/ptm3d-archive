"""Preview geometry for a mesh, colored by elevation. NO Qt, NO VTK.

WHY THIS IS A SEPARATE MODULE. ``VisualizationPanel`` cannot be exercised on
this machine: constructing it headless takes the VTK render window down with a
``qFatal``, and nothing that is only reachable through it can be tested at all.
So everything that DECIDES anything -- which reader, which elevation rule, what
reference radius, whether the mesh is too big to draw as a surface -- lives
here, where it is measured without a window. The panel keeps only the
``pyvista`` calls, and those are guarded so a failure prints a message into the
pane instead of aborting the application.

THE ELEVATION RULE, and why it is what it is.

1. **If the file carries its own per-vertex ``elevation`` property, that is
   the elevation.** Every DEM mesh this project runs is written by
   ``dtm_to_ply_delaunay`` (``scripts/dem/``), whose PLY header is::

       property double x / y / z
       property float  nx / ny / nz
       property double latitude
       property double longitude
       property double elevation

   Measured on ``sunlit_radcore_r50.ply`` (70,149 vertices, the mesh the
   manuscript's Fig. 8 runs on): ``|v| - elevation`` is **1737400.0 m with a
   standard deviation of 1.6e-10 m** over every vertex. In other words the
   tool's own elevation IS radial distance minus the Moon's reference radius,
   to the last bit of a double. Reading the property rather than recomputing it
   means the preview shows the number the DEM was built with, and the
   geometric rule below is checkable AGAINST it rather than merely plausible.

2. **Otherwise, for a BODY-CENTERED mesh, elevation is ``|v| - R_ref``.** A
   mesh is body-centered when its vertices all sit at a similar distance from
   the origin -- the fractional spread of ``|v|`` is small. That covers both
   cases the project has: a DEM patch, whose vertices are a few kilometers of
   relief on a 1737 km sphere, and a whole body (an icosphere, or a radar shape
   model), whose vertices are the body's own surface. ``R_ref`` is:

   * the body's catalog mean radius, when a body was named AND the mesh really
     is at that radius (within :data:`RADIUS_AGREEMENT`). Then the preview's
     elevation is the same datum the DEM tool used, and case 1 and case 2 agree
     to floating point -- which is exactly what
     ``test_gui_mesh_preview.py::test_the_geometric_rule_reproduces_a_real_dem``
     measures.
   * otherwise the mesh's own mean radius, which makes the elevation RELIEF
     ABOUT THE MESH MEAN and not height above a datum. It is a different
     quantity and it is labeled as one -- see :attr:`MeshPreview.datum`.

3. **Otherwise the mesh is in a local frame and elevation is ``z``.** A local
   patch has the origin inside it, so ``|v|`` runs from nearly zero to half the
   diagonal and rule 2 would report distance-from-the-middle as height. The
   assumption that ``+z`` is up is STATED in :attr:`MeshPreview.datum` rather
   than hidden, because it is the one rule here that the file cannot confirm.

UNITS. Asteroid ``.obj`` shape models are in KILOMETERS and lunar/Europa DEM
meshes are in METERS (``~/.claude/ptm-project-knowledge.md``, "Mesh frame is km
OR m depending on body"). Nothing in either format says which. So the unit is
MEASURED against the named body's catalog radius when there is one, and
reported as unknown when there is not -- never assumed. A frame slip here would
mis-scale a color bar by 1000x, which is the same class of error as the offset
scale bug the illumination work paid for.

SIZE. A preview is a picture, not a simulation. Above
:data:`PREVIEW_FACE_CAP` faces the surface is not drawn: the vertices are
sub-sampled to at most :data:`PREVIEW_POINT_CAP` and returned as a POINT CLOUD,
which renders in a moment and still shows the terrain and its elevation range.
That is a different picture and it says so (:attr:`MeshPreview.mode`); it is
not a decimation, and calling it one would be the quiet substitution this
project keeps paying for.

REFUSALS ARE VALUES, NOT EXCEPTIONS. :func:`read_mesh_preview` always returns a
:class:`MeshPreview`; a file that cannot be read comes back with ``refusal``
set and everything else empty. The caller is a Qt slot, and an exception out of
a Qt slot is an application abort.
"""

from __future__ import annotations

import math
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "PREVIEW_FACE_CAP",
    "PREVIEW_POINT_CAP",
    "RADIUS_AGREEMENT",
    "BODY_RADII_M",
    "MeshPreview",
    "elevation_field",
    "mesh_preview_for_config",
    "preview_mesh_path",
    "read_mesh_preview",
]


#: Faces above which the preview stops drawing a surface. Chosen from the
#: meshes this project actually opens: the A3GT A03MP026 3 km core is 2,397,890
#: faces and the moon3d flagship is 509,337, so the cap has to sit below both
#: or the "big mesh" path is never exercised on a real file. 500,000 also keeps
#: the point-cloud fallback under a second on this machine.
PREVIEW_FACE_CAP = 500_000

#: Points the fallback draws. The A3GT core has 899,154 vertices; drawing all
#: of them is neither necessary to see the terrain nor free.
PREVIEW_POINT_CAP = 400_000

#: How close the mesh's mean radius must be to a body's catalog radius before
#: the catalog radius is used as the elevation datum. Generous on purpose: a
#: regional DEM at a pole sits a few km below the mean radius, and a radar
#: shape model of a lumpy asteroid is not a sphere at all. What it must exclude
#: is a UNIT slip, which is a factor of 1000.
RADIUS_AGREEMENT = 0.30

#: Volumetric mean radii, meters. The same numbers as
#: ``gui_core.facet_scale._BODY_RADII_M`` and ``ptm.results.BODY_RADII``;
#: duplicated for the same reason ``facet_scale`` duplicates them -- this module
#: must import in a checkout where the ``ptm`` package is not on the path.
#: The duplication is CHECKED rather than trusted:
#: ``test_gui_mesh_preview.test_the_body_radii_agree_with_the_facet_scale_table``
#: fails if the two tables drift, which is the whole hazard of a third copy.
BODY_RADII_M = {
    "Europa": 1560800.0,
    "Moon": 1737400.0,
    "Mercury": 2439700.0,
    "Mars": 3389500.0,
    "Enceladus": 252100.0,
    "Titan": 2574730.0,
    "Ceres": 473000.0,
}

#: Fractional spread of |v| below which the mesh is treated as body-centered.
#: A DEM patch is ~0.004 (7 km of relief on 1737 km); a whole icosphere is 0; a
#: radar shape model of Bennu is ~0.2. A mesh in a LOCAL frame has vertices
#: near the origin, so its spread is of order 2. There is a decade of daylight
#: between the two populations.
_BODY_CENTERED_SPREAD = 0.75

#: PLY scalar type names to (struct code, size). ``list`` is handled apart.
_PLY_SCALAR = {
    "char": ("b", 1), "int8": ("b", 1),
    "uchar": ("B", 1), "uint8": ("B", 1),
    "short": ("h", 2), "int16": ("h", 2),
    "ushort": ("H", 2), "uint16": ("H", 2),
    "int": ("i", 4), "int32": ("i", 4),
    "uint": ("I", 4), "uint32": ("I", 4),
    "float": ("f", 4), "float32": ("f", 4),
    "double": ("d", 8), "float64": ("d", 8),
}


def _fmt(value):
    """A length with enough digits to be a number rather than "0.0".

    An asteroid shape model is in KILOMETERS, so its relief is hundredths of a
    unit; the same one-decimal format that reads well for a lunar DEM in meters
    renders Bennu's whole topography as "-0.0 to 0.0". The precision follows
    the magnitude instead of being fixed.
    """
    magnitude = abs(float(value))
    if magnitude >= 100:
        return f"{value:,.0f}"
    if magnitude >= 1:
        return f"{value:,.2f}"
    return f"{value:,.4f}"


@dataclass
class MeshPreview:
    """What the pane needs to draw a mesh, plus what it must say about it.

    ``vertices`` is an ``(n, 3)`` float array and ``faces`` an ``(m, 3)``
    integer array indexing it; in ``points`` mode ``faces`` is empty and only
    the vertices are drawn. ``elevation`` is per VERTEX and in the same length
    unit as the vertices, so a renderer can hand both straight to a scalar bar.
    """

    #: numpy ``(n, 3)`` float64, or None. Annotated ``object`` rather than left
    #: bare: an un-annotated assignment in a dataclass body is a CLASS
    #: attribute, not a field, so it would silently sit outside ``__init__``,
    #: ``__repr__`` and ``__eq__`` while the three below it are fields. One
    #: kind of member per class.
    vertices: object = None
    #: numpy ``(m, 3)`` int64, or None.
    faces: object = None
    #: numpy ``(n,)`` float64, or None. Per VERTEX.
    elevation: object = None
    #: "surface" or "points". See PREVIEW_FACE_CAP.
    mode: str = "surface"
    #: "meters", "kilometers", or "the file's own units".
    units: str = "the file's own units"
    #: One sentence naming the datum the elevation is measured from.
    datum: str = ""
    #: Where the elevation came from: "file", "radius", or "vertical".
    elevation_source: str = ""
    n_vertices_total: int = 0
    n_faces_total: int = 0
    n_vertices_shown: int = 0
    n_faces_shown: int = 0
    #: "ply-binary", "ply-ascii", "obj", or "".
    reader: str = ""
    path: str = ""
    #: Non-empty when nothing could be read. Everything else is then empty.
    refusal: str = ""
    #: Statements the pane must display: downsampling, an assumed unit, a
    #: datum that is not a body radius. Never silent.
    notes: list = field(default_factory=list)

    def elevation_range(self):
        """``(low, high)`` of the elevation field, or ``(None, None)``."""
        if self.elevation is None or len(self.elevation) == 0:
            return (None, None)
        return (float(self.elevation.min()), float(self.elevation.max()))

    def summary(self):
        """One paragraph for the pane. Says every reservation out loud."""
        if self.refusal:
            return self.refusal
        low, high = self.elevation_range()
        unit = {"meters": "m", "kilometers": "km"}.get(self.units, "")
        span = ("elevation not determined" if low is None else
                f"elevation {_fmt(low)} to {_fmt(high)} {unit}".rstrip())
        shown = (f"{self.n_faces_shown:,} faces"
                 if self.mode == "surface"
                 else f"{self.n_vertices_shown:,} points")
        head = (f"{Path(self.path).name}: {self.n_vertices_total:,} vertices, "
                f"{self.n_faces_total:,} faces ({self.reader}); drawing "
                f"{shown}. {span}. {self.datum}")
        return " ".join([head] + list(self.notes))


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def _numpy():
    """numpy, or None. Imported here so the module loads without it."""
    try:
        import numpy
    except ImportError:                                  # pragma: no cover
        return None
    return numpy


def _refuse(path, message):
    return MeshPreview(path=str(path), refusal=message)


def _declared_count_fits(path, count, item_bytes):
    """False when a header's element count cannot fit in the file.

    A PLY header is just text and this reader opens files nobody in this
    project wrote. ``element vertex 999999999999`` makes the read below ask
    CPython for a 24 TB buffer BEFORE it can discover the file is 4 kB; that is
    a MemoryError (caught, but only after the allocation attempt) or, on a
    machine with overcommit, a swap storm. The header is checked against the
    file's actual size first, which costs one ``stat``.

    Deliberately a LOWER bound and not an exact one: the count is compared
    against the whole file rather than against what remains after the header,
    so no legitimate file is ever refused by rounding.
    """
    try:
        size = os.path.getsize(path)
    except OSError:
        return True
    return count * max(1, int(item_bytes)) <= size


def _is_dataless(path):
    """A cloud placeholder: real size, zero blocks on this disk.

    ``preflight.is_dataless``, not a copy of it -- the launch check and the
    preview must agree about which files are readable, for the same reason
    :func:`_resolve` delegates. The inline fallback is for a checkout where the
    GUI package is not importable. It matters here because opening one of these
    BLOCKS while OneDrive hydrates it, which on the GUI thread is a frozen
    window rather than an error message.
    """
    try:
        from gui_core.preflight import is_dataless
        return is_dataless(path)
    except ImportError:                                  # pragma: no cover
        try:
            st = os.stat(path)
        except OSError:
            return False
        return st.st_blocks == 0 and st.st_size > 0


def _parse_ply_header(handle):
    """``(format, elements)`` for an open binary PLY handle, or ``(None, why)``.

    ``elements`` is a list of ``(name, count, [(type, prop_name), ...])`` in
    file order, with a list property recorded as
    ``("list", count_type, value_type, prop_name)``.
    """
    magic = handle.readline().strip()
    if magic != b"ply":
        return None, "this file does not begin with the PLY magic number"
    fmt = None
    elements = []
    while True:
        raw = handle.readline()
        if not raw:
            return None, "the PLY header ended before 'end_header'"
        line = raw.decode("ascii", "replace").strip()
        if line == "end_header":
            break
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "format":
            fmt = parts[1] if len(parts) > 1 else None
        elif parts[0] == "element" and len(parts) >= 3:
            try:
                elements.append((parts[1], int(parts[2]), []))
            except ValueError:
                return None, f"unreadable element count in {line!r}"
        elif parts[0] == "property" and elements:
            if parts[1] == "list" and len(parts) >= 5:
                elements[-1][2].append(("list", parts[2], parts[3], parts[4]))
            elif len(parts) >= 3:
                elements[-1][2].append((parts[1], parts[2]))
    if fmt is None:
        return None, "the PLY header names no format"
    return fmt, elements


def _scalar_dtype(properties, endian):
    """A numpy dtype for a run of fixed-size scalar properties, or None."""
    numpy = _numpy()
    fields = []
    for prop in properties:
        if prop[0] == "list":
            return None
        code = _PLY_SCALAR.get(prop[0])
        if code is None:
            return None
        fields.append((prop[1], endian + numpy.dtype(code[0]).str[1:]))
    return numpy.dtype(fields)


def _read_ply_binary(path, handle, elements, endian):
    numpy = _numpy()
    vertices = faces = stored_elevation = None
    n_vertices = n_faces = 0
    for name, count, properties in elements:
        if name == "vertex":
            dtype = _scalar_dtype(properties, endian)
            if dtype is None:
                return _refuse(path, "this PLY's vertex element carries a list "
                                     "property, which this reader does not "
                                     "parse. Nothing was drawn.")
            if not _declared_count_fits(path, count, dtype.itemsize):
                return _refuse(path, f"this PLY's header claims {count:,} "
                                     f"vertices, which cannot fit in a file "
                                     f"this size. The header is not "
                                     f"trustworthy; nothing was drawn.")
            block = handle.read(count * dtype.itemsize)
            if len(block) < count * dtype.itemsize:
                return _refuse(path, f"this PLY claims {count} vertices and "
                                     f"the file ends early. Nothing was "
                                     f"drawn.")
            table = numpy.frombuffer(block, dtype=dtype, count=count)
            names = table.dtype.names
            if not {"x", "y", "z"} <= set(names):
                return _refuse(path, "this PLY's vertex element has no x, y, z "
                                     "properties. Nothing was drawn.")
            vertices = numpy.empty((count, 3), dtype=numpy.float64)
            vertices[:, 0] = table["x"]
            vertices[:, 1] = table["y"]
            vertices[:, 2] = table["z"]
            if "elevation" in names:
                stored_elevation = numpy.array(table["elevation"],
                                               dtype=numpy.float64)
            n_vertices = count
        elif name == "face":
            faces, why = _read_ply_binary_faces(handle, count, properties,
                                                endian)
            if faces is None:
                return _refuse(path, why)
            n_faces = count
        else:
            dtype = _scalar_dtype(properties, endian)
            if dtype is None:
                # An unknown element with a list in it: its size cannot be
                # computed without walking it, and walking past the faces is
                # not worth it. Everything needed has usually been read by now.
                break
            handle.seek(count * dtype.itemsize, os.SEEK_CUR)
    if vertices is None:
        return _refuse(path, "this PLY has no vertex element. Nothing was "
                             "drawn.")
    preview = MeshPreview(path=str(path), reader="ply-binary")
    preview.vertices = vertices
    preview.faces = faces
    preview.elevation = stored_elevation
    preview.n_vertices_total = n_vertices
    preview.n_faces_total = n_faces
    return preview


def _read_ply_binary_faces(handle, count, properties, endian):
    """``(faces, why)`` -- an ``(m, 3)`` array, or ``(None, refusal)``.

    Assumes the list length is the SAME for every face, reads the whole block
    in one call, and then CHECKS that assumption on every row. A quad or a
    polygon soup is refused by name rather than silently mis-indexed, which is
    what a fixed stride would do to it.
    """
    numpy = _numpy()
    if not properties or properties[0][0] != "list":
        return None, ("this PLY's face element does not begin with a list "
                      "property, so its faces cannot be read. Nothing was "
                      "drawn.")
    _kind, count_type, value_type, _name = properties[0]
    count_code = _PLY_SCALAR.get(count_type)
    value_code = _PLY_SCALAR.get(value_type)
    if count_code is None or value_code is None:
        return None, (f"this PLY's face list is {count_type}/{value_type}, "
                      f"which this reader does not parse. Nothing was drawn.")
    if count == 0:
        return numpy.zeros((0, 3), dtype=numpy.int64), ""
    # Peek at the first count byte to fix the stride, then rewind.
    start = handle.tell()
    head = handle.read(count_code[1])
    if len(head) < count_code[1]:
        return None, "this PLY's face block is empty. Nothing was drawn."
    per_face = struct.unpack(endian + count_code[0], head)[0]
    handle.seek(start)
    if per_face != 3:
        return None, (f"this PLY's first face has {per_face} vertices. The "
                      f"preview draws triangles only, and re-triangulating a "
                      f"mesh to look at it would show a different mesh. "
                      f"Nothing was drawn.")
    trailing = _scalar_dtype(properties[1:], endian) if len(properties) > 1 \
        else numpy.dtype([])
    if trailing is None:
        return None, ("this PLY's face element carries a second list property, "
                      "so its stride is not fixed and this reader does not "
                      "parse it. Nothing was drawn.")
    fields = [("n", endian + numpy.dtype(count_code[0]).str[1:]),
              ("v", endian + numpy.dtype(value_code[0]).str[1:], (3,))]
    fields.extend(trailing.descr)
    dtype = numpy.dtype(fields)
    if not _declared_count_fits(handle.name, count, dtype.itemsize):
        return None, (f"this PLY's header claims {count:,} faces, which cannot "
                      f"fit in a file this size. The header is not "
                      f"trustworthy; nothing was drawn.")
    block = handle.read(count * dtype.itemsize)
    if len(block) < count * dtype.itemsize:
        return None, (f"this PLY claims {count} faces and the file ends "
                      f"early. Nothing was drawn.")
    table = numpy.frombuffer(block, dtype=dtype, count=count)
    if not bool((table["n"] == 3).all()):
        return None, ("this PLY mixes triangles with other polygons. The "
                      "preview draws triangles only. Nothing was drawn.")
    return numpy.array(table["v"], dtype=numpy.int64), ""


def _read_ply_ascii(path, handle, elements):
    numpy = _numpy()
    vertices = faces = stored_elevation = None
    n_vertices = n_faces = 0
    for name, count, properties in elements:
        if name == "vertex":
            names = [p[1] for p in properties if p[0] != "list"]
            try:
                index = {key: names.index(key) for key in ("x", "y", "z")}
            except ValueError:
                return _refuse(path, "this PLY's vertex element has no x, y, z "
                                     "properties. Nothing was drawn.")
            elevation_at = names.index("elevation") \
                if "elevation" in names else None
            rows = numpy.empty((count, 3), dtype=numpy.float64)
            stored = (numpy.empty(count, dtype=numpy.float64)
                      if elevation_at is not None else None)
            for row in range(count):
                parts = handle.readline().split()
                if len(parts) < len(names):
                    return _refuse(path, f"this PLY claims {count} vertices "
                                         f"and the file ends early. Nothing "
                                         f"was drawn.")
                try:
                    rows[row] = (float(parts[index["x"]]),
                                 float(parts[index["y"]]),
                                 float(parts[index["z"]]))
                    if stored is not None:
                        stored[row] = float(parts[elevation_at])
                except ValueError:
                    return _refuse(path, f"vertex {row} of this PLY is not "
                                         f"numeric. Nothing was drawn.")
            vertices, stored_elevation, n_vertices = rows, stored, count
        elif name == "face":
            rows = numpy.empty((count, 3), dtype=numpy.int64)
            for row in range(count):
                parts = handle.readline().split()
                if not parts:
                    return _refuse(path, f"this PLY claims {count} faces and "
                                         f"the file ends early. Nothing was "
                                         f"drawn.")
                try:
                    per_face = int(parts[0])
                except ValueError:
                    return _refuse(path, f"face {row} of this PLY is not "
                                         f"numeric. Nothing was drawn.")
                if per_face != 3:
                    return _refuse(path, f"face {row} of this PLY has "
                                         f"{per_face} vertices. The preview "
                                         f"draws triangles only. Nothing was "
                                         f"drawn.")
                rows[row] = [int(parts[1]), int(parts[2]), int(parts[3])]
            faces, n_faces = rows, count
        else:
            for _ in range(count):
                handle.readline()
    if vertices is None:
        return _refuse(path, "this PLY has no vertex element. Nothing was "
                             "drawn.")
    preview = MeshPreview(path=str(path), reader="ply-ascii")
    preview.vertices = vertices
    preview.faces = faces
    preview.elevation = stored_elevation
    preview.n_vertices_total = n_vertices
    preview.n_faces_total = n_faces
    return preview


def _read_obj(path):
    """Wavefront OBJ: ``v`` and triangular ``f``. Shape models are in km.

    Only the first index of an ``f`` token is read (``a/b/c`` is
    vertex/texture/normal), and OBJ indices are ONE-BASED -- an off-by-one here
    would fold the mesh, which is why it has its own test.
    """
    numpy = _numpy()
    vertices, faces = [], []
    skipped = 0
    try:
        with open(path, "r", errors="replace") as handle:
            for line in handle:
                if line.startswith("v "):
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            vertices.append((float(parts[1]), float(parts[2]),
                                             float(parts[3])))
                        except ValueError:
                            return _refuse(path, "this OBJ has a non-numeric "
                                                 "vertex. Nothing was drawn.")
                elif line.startswith("f "):
                    parts = line.split()[1:]
                    if len(parts) != 3:
                        skipped += 1
                        continue
                    try:
                        faces.append(tuple(int(p.split("/")[0]) - 1
                                           for p in parts))
                    except ValueError:
                        return _refuse(path, "this OBJ has a non-numeric "
                                             "face. Nothing was drawn.")
    except OSError as exc:
        return _refuse(path, f"this OBJ could not be read ({exc}). Nothing "
                             f"was drawn.")
    if not vertices:
        return _refuse(path, "this OBJ has no vertices. Nothing was drawn.")
    preview = MeshPreview(path=str(path), reader="obj")
    preview.vertices = numpy.array(vertices, dtype=numpy.float64)
    preview.faces = numpy.array(faces, dtype=numpy.int64) if faces \
        else numpy.zeros((0, 3), dtype=numpy.int64)
    preview.n_vertices_total = len(vertices)
    preview.n_faces_total = len(faces)
    if skipped:
        preview.notes.append(
            f"{skipped:,} non-triangular faces in this OBJ were left out of "
            f"the preview; the preview draws triangles only.")
    return preview


# ---------------------------------------------------------------------------
# The elevation rule
# ---------------------------------------------------------------------------

def elevation_field(vertices, body=None, stored=None, radius_m=None):
    """``(elevation, source, units, datum)`` for a vertex array.

    The whole rule from the module docstring, in one place and with no I/O, so
    it can be measured against a real DEM's own stored elevation.

    ``radius_m`` is an explicit mean radius in METERS and wins over the body
    catalog. It is how a GENERIC body gets a datum at all: there is no catalog
    row for Bennu, but ``generic_body.shape.radius_km`` is on the card.

    ``source`` is ``"file"``, ``"radius"`` or ``"vertical"``; ``units`` is
    ``"meters"``, ``"kilometers"`` or ``"the file's own units"``; ``datum`` is
    the sentence the pane shows.
    """
    numpy = _numpy()
    if numpy is None:                                    # pragma: no cover
        return None, "", "the file's own units", ""
    radius = numpy.sqrt((numpy.asarray(vertices, dtype=numpy.float64) ** 2)
                        .sum(axis=1))
    if len(radius) == 0:
        return None, "", "the file's own units", ""
    mean_radius = float(radius.mean())
    body = str(body or "").strip()
    catalog = BODY_RADII_M.get(body.title())
    if radius_m and float(radius_m) > 0:
        catalog = float(radius_m)
        body = body or "this body"

    # UNITS FIRST, because the datum comparison is a comparison of lengths.
    units = "the file's own units"
    catalog_in_file_units = None
    if catalog and mean_radius > 0:
        if abs(mean_radius - catalog) / catalog <= RADIUS_AGREEMENT:
            units, catalog_in_file_units = "meters", catalog
        elif abs(mean_radius - catalog / 1000.0) / (catalog / 1000.0) \
                <= RADIUS_AGREEMENT:
            units, catalog_in_file_units = "kilometers", catalog / 1000.0

    spread = ((float(radius.max()) - float(radius.min())) / mean_radius
              if mean_radius > 0 else float("inf"))
    body_centered = spread <= _BODY_CENTERED_SPREAD

    if stored is not None and len(stored) == len(radius):
        datum = ("Elevation is this mesh's own per-vertex elevation property, "
                 "the value the DEM was built with.")
        if catalog_in_file_units is not None:
            residual = float(numpy.abs(radius - stored
                                       - catalog_in_file_units).max())
            tolerance = 1e-6 * catalog_in_file_units
            if residual <= tolerance:
                datum += (f" It is radial distance minus the {body} reference "
                          f"radius, {_fmt(catalog_in_file_units)} "
                          f"{'km' if units == 'kilometers' else 'm'} -- "
                          f"checked on every vertex of this file.")
            else:
                datum += (f" It is NOT radial distance minus the {body} "
                          f"reference radius: those disagree by up to "
                          f"{_fmt(residual)}, so the file is using some other "
                          f"datum and the preview shows the file's.")
        return numpy.asarray(stored, dtype=numpy.float64), "file", units, datum

    if not body_centered:
        datum = ("Elevation is the mesh's own z coordinate. This mesh is not "
                 "centered on a body -- its vertices surround the origin -- so "
                 "it is read as a local patch with z up. That is an "
                 "assumption about the file, not something the file states.")
        return (numpy.asarray(vertices, dtype=numpy.float64)[:, 2].copy(),
                "vertical", units, datum)

    if catalog_in_file_units is not None:
        datum = (f"Elevation is radial distance minus the {body} reference "
                 f"radius, {_fmt(catalog_in_file_units)} "
                 f"{'km' if units == 'kilometers' else 'm'}.")
        return radius - catalog_in_file_units, "radius", units, datum

    datum = (f"Elevation is radial distance minus THIS MESH'S own mean radius, "
             f"{_fmt(mean_radius)}. No body radius was available to use as a "
             f"datum, so this is relief about the mesh mean and not height "
             f"above a reference surface.")
    return radius - mean_radius, "radius", units, datum


# ---------------------------------------------------------------------------
# The entry point
# ---------------------------------------------------------------------------

def read_mesh_preview(path, body=None, face_cap=PREVIEW_FACE_CAP,
                      point_cap=PREVIEW_POINT_CAP, radius_m=None):
    """Read `path` and return a :class:`MeshPreview`. NEVER raises."""
    try:
        return _read_mesh_preview(path, body, face_cap, point_cap, radius_m)
    except Exception as exc:                             # noqa: BLE001
        # The caller is a Qt slot. An exception out of one is an application
        # abort, and this reader parses files nobody in this project wrote.
        return _refuse(path, f"this mesh could not be read ({type(exc).__name__}"
                             f": {exc}). Nothing was drawn.")


def _read_mesh_preview(path, body, face_cap, point_cap, radius_m=None):
    numpy = _numpy()
    if numpy is None:                                    # pragma: no cover
        return _refuse(path, "numpy is not installed, so no mesh can be read.")
    if not path:
        return _refuse(path, "No mesh file is selected.")
    path = Path(path)
    if not path.exists():
        return _refuse(path, f"{path} is not on disk, so there is nothing to "
                             f"preview. The launch check names where the run "
                             f"will look for it.")
    if _is_dataless(path):
        return _refuse(path, f"{path} is present but DATALESS -- a cloud "
                             f"placeholder whose contents are not on this "
                             f"disk. Reading it would block this window while "
                             f"it downloads, so nothing was drawn. Hydrate it "
                             f"in Finder first.")

    suffix = path.suffix.lower()
    if suffix == ".obj":
        preview = _read_obj(path)
    elif suffix == ".ply":
        with open(path, "rb") as handle:
            fmt, elements = _parse_ply_header(handle)
            if fmt is None:
                return _refuse(path, f"{path.name} is not a PLY this reader "
                                     f"understands: {elements}. Nothing was "
                                     f"drawn.")
            if fmt.startswith("binary"):
                endian = "<" if "little" in fmt else ">"
                preview = _read_ply_binary(path, handle, elements, endian)
            elif fmt == "ascii":
                handle.close()
                with open(path, "r", errors="replace") as text:
                    while True:
                        line = text.readline()
                        if not line or line.strip() == "end_header":
                            break
                    preview = _read_ply_ascii(path, text, elements)
            else:
                return _refuse(path, f"{path.name} declares PLY format "
                                     f"{fmt!r}, which this reader does not "
                                     f"parse. Nothing was drawn.")
    else:
        return _refuse(path, f"{path.name} is a {suffix or 'extensionless'} "
                             f"file. The preview reads PLY and OBJ. The ENGINE "
                             f"also loads legacy VTK meshes; those are not "
                             f"previewed here, and not previewing one says "
                             f"nothing about whether the run will load it.")
    if preview.refusal:
        return preview

    # INDEX RANGE, before anything hands these to VTK. A face index outside
    # the vertex array is not a Python error downstream -- `pv.PolyData` passes
    # the connectivity straight to VTK, which reads the point array at that
    # offset. An out-of-bounds read there is a SEGFAULT, not an exception, so
    # the try/except around the render call cannot catch it and the whole
    # application goes down. This reader parses files nobody in this project
    # wrote, so the check belongs here, where it is testable.
    if preview.faces is not None and len(preview.faces):
        highest = int(preview.faces.max())
        lowest = int(preview.faces.min())
        if lowest < 0 or highest >= preview.n_vertices_total:
            return _refuse(path,
                           f"{path.name} has a face referring to vertex "
                           f"{highest if highest >= preview.n_vertices_total else lowest}, "
                           f"and it has {preview.n_vertices_total:,} vertices. "
                           f"The file's connectivity does not address its own "
                           f"geometry; nothing was drawn.")

    elevation, source, units, datum = elevation_field(
        preview.vertices, body=body, stored=preview.elevation,
        radius_m=radius_m)
    preview.elevation = elevation
    preview.elevation_source = source
    preview.units = units
    preview.datum = datum
    if units == "the file's own units":
        preview.notes.append(
            "The length unit of this mesh could not be established from a body "
            "radius, so the elevation numbers are in whatever unit the file "
            "uses -- meters for a lunar or Europa DEM, kilometers for an "
            "asteroid shape model.")

    _apply_caps(preview, face_cap, point_cap)
    return preview


def _apply_caps(preview, face_cap, point_cap):
    """Thin a mesh too large to draw, and SAY which picture is being shown."""
    numpy = _numpy()
    faces = preview.faces
    if faces is not None and len(faces) and len(faces) <= face_cap:
        preview.mode = "surface"
        preview.n_faces_shown = len(faces)
        preview.n_vertices_shown = len(preview.vertices)
        return
    if faces is None or not len(faces):
        preview.mode = "points"
        preview.faces = numpy.zeros((0, 3), dtype=numpy.int64)
        preview.notes.append(
            "This mesh carries no triangles, so the preview draws its "
            "vertices as points.")
    else:
        preview.mode = "points"
        preview.notes.append(
            f"This mesh has {len(faces):,} faces, above the "
            f"{face_cap:,}-face preview limit, so the preview draws its "
            f"VERTICES as points rather than the surface. It is a thinned "
            f"sample of the same geometry, not a decimated mesh: it shows the "
            f"terrain and its elevation range, and it is not what the run "
            f"solves on.")
        preview.faces = numpy.zeros((0, 3), dtype=numpy.int64)

    n = len(preview.vertices)
    if n > point_cap:
        stride = int(math.ceil(n / float(point_cap)))
        keep = slice(None, None, stride)
        preview.vertices = numpy.ascontiguousarray(preview.vertices[keep])
        if preview.elevation is not None:
            preview.elevation = numpy.ascontiguousarray(
                preview.elevation[keep])
        preview.notes.append(
            f"Every {stride}th vertex is drawn ({len(preview.vertices):,} of "
            f"{n:,}); the elevation range quoted is that of the drawn sample.")
    preview.n_vertices_shown = len(preview.vertices)
    preview.n_faces_shown = 0


# ---------------------------------------------------------------------------
# What a configuration asks to preview
# ---------------------------------------------------------------------------

def preview_mesh_path(config, project_root=None):
    """``(path, why)`` -- the mesh a configuration would have the preview show.

    ``mesh.import_file`` first, because it is what the ENGINE runs on when it
    is set (``ptm_thermal_sim.cpp:17036`` decides "custom mesh" on that key
    alone). ``generic_body.shape.mesh_file`` next, which is the same question
    on a generic-body card. A run with neither is a procedural icosphere, and
    an icosphere has no topography to color -- so it gets a refusal that says
    that, not an empty pane.

    ``mesh.occluder_file`` is deliberately NOT offered. The ring is added to
    the shadow BVH and is "NOT thermally solved / radiosity-coupled / output"
    (``ptm_thermal_sim.cpp:16405-16410``); previewing it beside the core would
    show terrain the run does not solve on.
    """
    config = config or {}
    mesh = config.get("mesh") or {}
    imported = str(mesh.get("import_file", "") or "").strip()
    if imported:
        return _resolve(imported, project_root), "mesh.import_file"
    shape = ((config.get("generic_body") or {}).get("shape") or {})
    shape_file = str(shape.get("mesh_file", "") or "").strip()
    if shape_file:
        return _resolve(shape_file, project_root), "generic_body.shape.mesh_file"
    return None, ("This run has no mesh file: it builds a procedural icosphere "
                  "from mesh.resolution, which is a smooth sphere with no "
                  "topography to color. Pick a terrain or a shape model to see "
                  "one.")


def _resolve(value, project_root):
    """Absolutize the way the engine's ``PTM_DATA_ROOT`` contract does.

    ``preflight.resolve_data_path``, NOT a copy of it. ``mesh.import_file`` and
    ``generic_body.shape.mesh_file`` are both data-root fields
    (``preflight._ENGINE_DATA_ROOT_FIELDS``), and a preview that resolved them
    differently from the launch check would draw a mesh the run does not open --
    or refuse one the run loads happily. The Europa terrain card already learned
    this: it uses the pre-flight's resolver for the same key so it cannot say
    "on disk" about a file the run then fails to find.

    The fallback is used only where the GUI package is not importable, which is
    the same condition the rest of this module is built to survive.
    """
    root = Path(project_root) if project_root \
        else Path(__file__).resolve().parents[2]
    try:
        from gui_core.preflight import resolve_data_path
        return resolve_data_path(value, root)
    except ImportError:                                  # pragma: no cover
        path = Path(value).expanduser()
        if path.is_absolute():
            return path
        data_root = str(os.environ.get("PTM_DATA_ROOT", "") or "").strip()
        if data_root and str(value).startswith("data/"):
            return Path(data_root).expanduser() / str(value)[len("data/"):]
        return root / path


def mesh_preview_for_config(config, project_root=None, face_cap=PREVIEW_FACE_CAP,
                            point_cap=PREVIEW_POINT_CAP):
    """The preview a configuration asks for, refusal and all. Never raises.

    A GENERIC body has no row in :data:`BODY_RADII_M`, so its datum comes from
    ``generic_body.shape.radius_km`` -- the same number the ENGINE builds the
    body with. Without it the elevation is relief about the mesh's own mean
    radius, which is a different quantity and is labeled as one.
    """
    path, why = preview_mesh_path(config, project_root)
    if path is None:
        return _refuse("", why)
    config = config or {}
    body = str((config.get("simulation") or {}).get("body", "") or "")
    shape = ((config.get("generic_body") or {}).get("shape") or {})
    try:
        radius_km = float(shape.get("radius_km", 0.0) or 0.0)
    except (TypeError, ValueError):
        radius_km = 0.0
    radius_m = radius_km * 1000.0 if radius_km > 0 else None
    return read_mesh_preview(path, body=body, face_cap=face_cap,
                             point_cap=point_cap, radius_m=radius_m)
