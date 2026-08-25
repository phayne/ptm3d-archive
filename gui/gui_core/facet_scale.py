"""Facet scale, and the beaming rule that depends on it.

`radiation.beaming` turns on the Smith (1967) sub-facet roughness treatment.
The engine enforces exactly one thing about it -- `beaming: true` requires
`rms_slope_deg > 0` (`ptm_thermal_sim.cpp:13592`) -- and **nothing** about the
scale of the mesh it is applied to. Grepping `src/cpp/` for a facet-scale check
returns nothing. So if the GUI offers the switch, the GUI is the only thing
between a user and double-counted roughness: a mesh whose facets are already AT
the roughness scale resolves the roughness geometrically, and adding a
sub-facet model on top counts it twice.

`docs/radiation-exchange.md` section 8.4 gives the criterion as a length
comparison: facet scale against the thermal-isolation scale of roughly a
millimeter to a centimeter. Measured on the meshes this project actually runs:

    sunlit_radcore_r50                  140,091 faces     237 m     ON
    haworth_radcore_r32c                 99,665 faces     181 m     ON
    A3GT A03MP026 3 km core           2,397,890 faces    3.46 m     ON
    moon3d_c...NEW_500k_clean           509,337 faces    1.96 mm    OFF

The boundaries below are Ada's proposal from `docs/design/gui-update-plan.md`
section 4.4, approved as DP-4: refuse below 0.05 m (one decade above the top of
the millimeter-to-centimeter band, so the 1.96 mm mesh is refused with margin),
warn between 0.05 m and 1 m, allow at or above 1 m (so the 3.46 m A3GT core
passes cleanly), and lunar bodies only, because w = 32.5 deg is calibrated on
lunar regolith and section 8.4 declines to carry it to icy or small bodies.

**Not an engine-side guard.** ENG-BEAM-SCALE-1 in `BUGS.md` records that a CLI
user still has nothing between them and the same mistake. This module is the
GUI's half only.

Unit assumption, stated because it decides an answer: an imported mesh is read
in its own frame and assumed to be in METERS, which is the lunar and Europa DEM
convention (asteroid `.obj` are in kilometers, but beaming is lunar-only here).
If a lunar mesh were nevertheless in kilometers, its facet scale would read 1000x
too small and beaming would be REFUSED -- the conservative direction. The label
says which assumption was used.

No Qt import: this is testable without a display.
"""

import math
import struct
from dataclasses import dataclass
from pathlib import Path

#: Below this facet scale the mesh is at or near the roughness scale itself and
#: a sub-facet model would double-count. Refuse.
FACET_SCALE_DISABLE_BELOW_M = 0.05
#: Between the two, allow but require the user to acknowledge the number.
FACET_SCALE_WARN_BELOW_M = 1.0
#: Bodies whose regolith the Smith slope distribution is calibrated for.
BEAMING_BODIES = ("Moon",)
#: Above this face count, sample rather than sum every triangle. The plan
#: (section 4.4) proposed 5,000,000; measurement moved it down. Summing all
#: 2,397,890 faces of the A3GT A03MP026 core took 9.9 s in this reader, which is
#: not something to do on the GUI thread every time a field changes. Sampling
#: 5,000 evenly spaced faces of the same mesh reproduces the documented 3.46 m.
_AREA_SAMPLE_ABOVE = 200_000
_AREA_SAMPLE_COUNT = 5_000

#: Volumetric mean radii, meters. Same numbers as ptm.results.BODY_RADII;
#: duplicated rather than imported so this module stays importable from a
#: checkout where the ptm package has not been bootstrapped yet.
_BODY_RADII_M = {
    "Europa": 1560800.0,
    "Moon": 1737400.0,
    "Mercury": 2439700.0,
    "Mars": 3389500.0,
    "Enceladus": 252100.0,
    "Titan": 2574730.0,
    "Ceres": 473000.0,
}


@dataclass
class FacetScale:
    """sqrt(face area), as a headline mean AND a fine-end percentile.

    ``meters`` is sqrt(MEAN area) -- the number docs/radiation-exchange.md 8.4
    tabulates, kept so the two agree. ``fine_meters`` is sqrt(the 5th-percentile
    area), and it is what the rule is applied to.

    The mean alone fails open on a MULTI-RESOLUTION mesh, which is not a
    contrived case: `scripts/a3gt_v2_extended_mesh.py` builds the production
    A3GT v2 mesh as one PLY whose resolution falls off from the center, and
    measured on the real A03MP026 3 km core the mean is 3.47 m while the 5th
    percentile is 0.71 m -- a factor of 4.9. Construct the extreme and the mean
    hides it completely: 99% of facets at 1 mm plus 1% at 1 m reports 70.7 cm,
    passing a rule written to refuse millimeter facets with margin.

    Beaming double-counts on the facets that ALREADY resolve the roughness, so
    the question is whether a meaningful fraction of the mesh sits at that
    scale, not what the average is.
    """
    meters: float = None          # None = not determinable
    fine_meters: float = None     # sqrt(5th-percentile area); the gate
    n_faces: int = None
    source: str = "unknown"       # icosphere | ply | ply-sampled | obj | unknown
    detail: str = ""

    def describe(self):
        if self.meters is None:
            return f"facet scale unknown ({self.detail})"
        spread = ""
        if self.fine_meters is not None and \
                self.fine_meters < 0.9 * self.meters:
            spread = (f", but the finest 5% are "
                      f"{_fmt_length(self.fine_meters)} -- this mesh is "
                      f"multi-resolution")
        return (f"facet scale {_fmt_length(self.meters)} "
                f"({self.n_faces:,} faces, {self.source}"
                + (f"; {self.detail}" if self.detail else "") + ")" + spread)


@dataclass
class BeamingPolicy:
    """Whether the GUI may offer the beaming switch, and what to say."""
    allowed: bool = False
    needs_acknowledgment: bool = False
    reason: str = ""
    scale: FacetScale = None


def _fmt_length(m):
    if m < 0.01:
        return f"{m * 1000.0:.3g} mm"
    if m < 1.0:
        return f"{m * 100.0:.3g} cm"
    if m < 1000.0:
        return f"{m:.3g} m"
    return f"{m / 1000.0:.4g} km"


def _percentile(values, q):
    """Nearest-rank percentile of `values`. Small samples, so no numpy."""
    ordered = sorted(values)
    if not ordered:
        return 0.0
    i = int(q * (len(ordered) - 1))
    return ordered[max(0, min(len(ordered) - 1, i))]


def _clean_body(config):
    body = str((config.get("simulation", {}) or {}).get("body", "") or "")
    return body[4:] if body.startswith("[H] ") else body


# --------------------------------------------------------------------------
# PLY geometry. Header-only readers already exist in preflight (face count);
# this one needs the vertices, so it parses the element/property layout.
# --------------------------------------------------------------------------

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


def _parse_ply_header(handle):
    """Return (format, elements, header_bytes) or raise ValueError.

    elements is a list of (name, count, properties); a property is either
    ('scalar', type, name) or ('list', count_type, item_type, name).
    """
    if handle.readline().strip() != b"ply":
        raise ValueError("not a PLY file")
    fmt = None
    elements = []
    while True:
        raw = handle.readline()
        if not raw:
            raise ValueError("PLY header ended without end_header")
        line = raw.decode("ascii", errors="replace").strip()
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "format":
            fmt = parts[1]
        elif parts[0] == "element":
            elements.append((parts[1], int(parts[2]), []))
        elif parts[0] == "property":
            if not elements:
                raise ValueError("PLY property outside any element")
            if parts[1] == "list":
                elements[-1][2].append(("list", parts[2], parts[3], parts[4]))
            else:
                elements[-1][2].append(("scalar", parts[1], parts[2]))
        elif parts[0] == "end_header":
            break
    if fmt is None:
        raise ValueError("PLY header carries no format line")
    return fmt, elements, handle.tell()


#: {(path, mtime_ns, size): (mean_area, n_faces, n_sampled)}. The Radiation tab
#: recomputes on every field change; without this, a 100 MB PLY would be re-read
#: each time. Keyed on mtime and size so an edited mesh is re-measured.
_AREA_CACHE = {}
_AREA_CACHE_MAX = 8


def _obj_face_areas(path):
    """(sampled per-face areas, n_faces, n_sampled) for a Wavefront OBJ.

    The engine loads OBJ (mesh_utils.cpp:1264) and the Mesh tab's file filter
    offers it, so refusing it here made beaming permanently unreachable for a
    legitimate coarse lunar OBJ. The format is line-oriented: `v x y z` and
    `f a b c` with 1-based indices, optionally `f a/vt/vn ...`.

    RESERVOIR sampling, not "sum the first N then stride". OBJ has no header
    face count, so a stride cannot be chosen up front -- and taking the first
    200,000 faces exhaustively is a BIASED sample of a mesh whose resolution
    varies with position, which every A3GT-class mesh's does. Measured on
    byte-identical geometry written to both formats (245,000 faces at 1 mm then
    5,000 at 1 m): the prefix-then-stride reader reported 2.49 cm and REFUSED
    while the PLY path reported 14.1 cm and allowed -- a 5.7x disagreement that
    flipped the launch gate on file extension alone. A reservoir is uniform over
    the whole file in one pass.
    """
    import random

    verts = []
    reservoir = []
    n_faces = 0
    rng = random.Random(20260825)          # deterministic: same file, same answer
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            if line.startswith("v "):
                parts = line.split()
                verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif line.startswith("f "):
                n_faces += 1
                idx = []
                for token in line.split()[1:]:
                    i = int(token.split("/")[0])
                    # OBJ indices are 1-based; a negative one counts back from
                    # the end of the vertex list seen SO FAR.
                    idx.append(i - 1 if i > 0 else len(verts) + i)
                area = 0.0
                for k in range(1, len(idx) - 1):
                    area += _triangle_area(verts[idx[0]], verts[idx[k]],
                                           verts[idx[k + 1]])
                if len(reservoir) < _AREA_SAMPLE_COUNT:
                    reservoir.append(area)
                else:
                    j = rng.randrange(n_faces)
                    if j < _AREA_SAMPLE_COUNT:
                        reservoir[j] = area
    if not reservoir:
        raise ValueError("OBJ file declares no face")
    return reservoir, n_faces, len(reservoir)


def _ply_mean_face_area(path):
    """(sampled per-face areas in the mesh's own units, n_faces, n_sampled)."""
    stat = Path(path).stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    if key in _AREA_CACHE:
        return _AREA_CACHE[key]
    result = (_obj_face_areas(path)
              if str(path).lower().endswith(".obj")
              else _ply_mean_face_area_uncached(path))
    if len(_AREA_CACHE) >= _AREA_CACHE_MAX:
        _AREA_CACHE.clear()
    _AREA_CACHE[key] = result
    return result


def _ply_mean_face_area_uncached(path):
    """Read the geometry. An unsupported layout is reported, never guessed at."""
    with open(path, "rb") as fh:
        fmt, elements, _offset = _parse_ply_header(fh)
        if fmt == "ascii":
            return _ply_mean_face_area_ascii(fh, elements)
        if fmt != "binary_little_endian":
            raise ValueError(f"unsupported PLY format {fmt!r}")
        return _ply_mean_face_area_binary(fh, elements)


def _element_struct(properties):
    """(struct format, size) for an element with only scalar properties."""
    fmt = "<"
    size = 0
    for prop in properties:
        if prop[0] != "scalar":
            return None, None
        code, width = _PLY_SCALAR[prop[1]]
        fmt += code
        size += width
    return fmt, size


def _xyz_indices(properties):
    names = [p[2] for p in properties if p[0] == "scalar"]
    try:
        return names.index("x"), names.index("y"), names.index("z")
    except ValueError:
        raise ValueError("PLY vertex element has no x/y/z properties")


def _triangle_area(a, b, c):
    ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
    cx = uy * vz - uz * vy
    cy = uz * vx - ux * vz
    cz = ux * vy - uy * vx
    return 0.5 * math.sqrt(cx * cx + cy * cy + cz * cz)


def _sample_stride(n_faces):
    if n_faces <= _AREA_SAMPLE_ABOVE:
        return 1, n_faces
    stride = max(1, n_faces // _AREA_SAMPLE_COUNT)
    return stride, len(range(0, n_faces, stride))


def _ply_mean_face_area_binary(fh, elements):
    vertices = None
    for name, count, properties in elements:
        if name == "vertex":
            fmt, size = _element_struct(properties)
            if fmt is None:
                raise ValueError("PLY vertex element carries a list property")
            ix, iy, iz = _xyz_indices(properties)
            raw = fh.read(size * count)
            if len(raw) < size * count:
                raise ValueError("PLY vertex block is truncated")
            unpack = struct.Struct(fmt).unpack_from
            vertices = [None] * count
            for i in range(count):
                row = unpack(raw, i * size)
                vertices[i] = (row[ix], row[iy], row[iz])
        elif name == "face":
            if vertices is None:
                raise ValueError("PLY face element precedes the vertex element")
            return _accumulate_binary_faces(fh, count, properties, vertices)
        else:
            fmt, size = _element_struct(properties)
            if fmt is None:
                raise ValueError(
                    f"cannot skip PLY element {name!r}: it carries a list property")
            fh.seek(size * count, 1)
    raise ValueError("PLY has no face element")


def _accumulate_binary_faces(fh, n_faces, properties, vertices):
    lists = [p for p in properties if p[0] == "list"]
    if not lists:
        raise ValueError("PLY face element carries no vertex-index list")
    # A meshlab-style face element carries trailing scalars (red/green/blue)
    # after the index list. The engine's own reader handles that
    # (mesh_utils.cpp locates the vertex_indices property by name and skips the
    # rest), so refusing it here would refuse beaming on a mesh that runs fine,
    # for a reason about this reader rather than about the mesh. What must NOT
    # happen is reading only the index list and leaving the stream misaligned:
    # a misread record that happens to decode n == 3 yields a WRONG mean area
    # silently, which could report a mesh as coarser than it is and allow
    # beaming on one the rule exists to refuse.
    #
    # So: the index list must come FIRST (an unknown-length list ahead of it
    # cannot be skipped without decoding it, and no PTM3D-relevant exporter
    # writes one), and everything after it must be fixed-width scalars.
    index_prop = lists[0]
    if properties.index(index_prop) != 0:
        raise ValueError(
            "PLY face element puts a property before the vertex-index list "
            f"({[p[-1] for p in properties]}); this reader cannot compute face "
            f"areas from that layout")
    trailing = properties[1:]
    if any(p[0] != "scalar" for p in trailing):
        raise ValueError(
            "PLY face element carries a second list after the vertex-index "
            f"list ({[p[-1] for p in properties]}); this reader cannot compute "
            f"face areas from that layout")
    trailing_bytes = sum(_PLY_SCALAR[p[1]][1] for p in trailing)
    _kind, count_type, item_type, _name = index_prop
    ccode, cwidth = _PLY_SCALAR[count_type]
    icode, iwidth = _PLY_SCALAR[item_type]
    stride, _expected = _sample_stride(n_faces)
    block_start = fh.tell()

    if stride > 1:
        seeked = _seek_sampled_triangles(fh, n_faces, stride, block_start,
                                         ccode, cwidth, icode, iwidth,
                                         trailing_bytes, vertices)
        if seeked is not None:
            return seeked
        # Not a uniform triangle block. Fall back to streaming rather than
        # reporting an area computed from records at the wrong offsets.
        fh.seek(block_start)

    areas = []
    for f in range(n_faces):
        head = fh.read(cwidth)
        if len(head) < cwidth:
            raise ValueError("PLY face block is truncated")
        n = struct.unpack("<" + ccode, head)[0]
        body = fh.read(iwidth * n)
        if len(body) < iwidth * n:
            raise ValueError("PLY face block is truncated")
        if trailing_bytes and len(fh.read(trailing_bytes)) < trailing_bytes:
            raise ValueError("PLY face block is truncated")
        if f % stride:
            continue
        idx = struct.unpack("<" + icode * n, body)
        # A polygon of n vertices is a fan of n-2 triangles; PTM3D meshes are
        # triangulated, so this is n == 3 in practice.
        area = 0.0
        for k in range(1, n - 1):
            area += _triangle_area(vertices[idx[0]], vertices[idx[k]],
                                   vertices[idx[k + 1]])
        areas.append(area)
    if not areas:
        raise ValueError("PLY face block yielded no usable triangle")
    return areas, n_faces, len(areas)


def _seek_sampled_triangles(fh, n_faces, stride, block_start,
                            ccode, cwidth, icode, iwidth, trailing_bytes,
                            vertices):
    """Sample by seeking, valid only if every face record is a triangle.

    Returns (mean_area, n_faces, n_sampled), or None when the block is not a
    uniform run of 3-vertex records -- in which case the caller streams. Every
    record it touches is re-checked for n == 3, so a mixed-arity block is
    detected rather than silently misread at the wrong offsets.
    """
    record = cwidth + 3 * iwidth + trailing_bytes
    fh.seek(0, 2)
    if fh.tell() - block_start < n_faces * record:
        return None
    unpack_head = struct.Struct("<" + ccode).unpack
    unpack_idx = struct.Struct("<" + icode * 3).unpack

    areas = []
    for f in range(0, n_faces, stride):
        fh.seek(block_start + f * record)
        raw = fh.read(record)
        if len(raw) < record or unpack_head(raw[:cwidth])[0] != 3:
            return None
        a, b, c = unpack_idx(raw[cwidth:cwidth + 3 * iwidth])
        areas.append(_triangle_area(vertices[a], vertices[b], vertices[c]))
    if not areas:
        return None
    return areas, n_faces, len(areas)


def _ply_mean_face_area_ascii(fh, elements):
    vertices = []
    areas = []
    n_faces = 0
    for name, count, properties in elements:
        if name == "vertex":
            ix, iy, iz = _xyz_indices(properties)
            for _ in range(count):
                row = fh.readline().split()
                vertices.append((float(row[ix]), float(row[iy]), float(row[iz])))
        elif name == "face":
            n_faces = count
            stride, _expected = _sample_stride(count)
            for f in range(count):
                row = fh.readline().split()
                if f % stride:
                    continue
                # Trailing per-face properties (colors) follow the index list
                # on the same line; slicing to 1+n ignores them, which is
                # correct here because an ASCII record is line-delimited.
                n = int(row[0])
                idx = [int(v) for v in row[1:1 + n]]
                area = 0.0
                for k in range(1, n - 1):
                    area += _triangle_area(vertices[idx[0]], vertices[idx[k]],
                                           vertices[idx[k + 1]])
                areas.append(area)
        else:
            for _ in range(count):
                fh.readline()
    if not areas:
        raise ValueError("PLY face block yielded no usable triangle")
    return areas, n_faces, len(areas)


# --------------------------------------------------------------------------

def facet_scale(config, project_root=None):
    """sqrt(mean face area) for the mesh this config selects."""
    mesh = (config.get("mesh", {}) or {}) if isinstance(config, dict) else {}
    import_file = str(mesh.get("import_file", "") or "")

    if import_file:
        # Resolve exactly the way preflight (and therefore the run YAML) does,
        # so a relative `data/...` reference under PTM_DATA_ROOT is found here
        # too. A bare `project_root / path` reported "does not exist" for the
        # documented deployment, and the policy then refused for the wrong
        # reason.
        try:
            from gui_core.preflight import resolve_data_path
            path = Path(resolve_data_path(import_file, project_root))
        except Exception:
            path = Path(import_file)
            if project_root and not path.is_absolute():
                path = Path(project_root) / path
        if not path.exists():
            return FacetScale(detail=f"{path} does not exist")
        # This runs from mesh.import_file's per-keystroke textChanged, on the
        # GUI thread, BEFORE any pre-flight. Opening an evicted OneDrive
        # placeholder here triggers a synchronous hydration of a
        # multi-tens-of-megabyte file and hangs the window -- the exact hazard
        # preflight guards every other mesh path against. stat() alone never
        # hydrates, so the check itself is free.
        try:
            from gui_core.preflight import is_dataless
            if is_dataless(path):
                return FacetScale(
                    detail=f"{path.name} is a cloud placeholder (dataless); "
                           f"reading it here would hang the window")
        except ImportError:
            pass
        if not path.name.lower().endswith((".ply", ".obj")):
            return FacetScale(
                detail=f"{path.name} is neither PLY nor OBJ; only those two "
                       f"geometry formats are read here")
        is_obj = path.name.lower().endswith(".obj")
        try:
            areas, n_faces, sampled = _ply_mean_face_area(path)
        except (OSError, ValueError, IndexError, KeyError, struct.error) as exc:
            return FacetScale(detail=f"could not read {path.name}: {exc}")
        mean_area = sum(areas) / len(areas) if areas else 0.0
        if not mean_area or mean_area <= 0:
            return FacetScale(n_faces=n_faces,
                              detail=f"{path.name} has zero mean face area")
        try:
            scale = float(mesh.get("scale_factor", 1.0) or 1.0)
        except (TypeError, ValueError):
            return FacetScale(
                n_faces=n_faces,
                detail=f"mesh.scale_factor is "
                       f"{mesh.get('scale_factor')!r}, not a number")
        fmt_name = "OBJ" if is_obj else "PLY"
        detail = f"assuming the {fmt_name} is in meters"
        source = "obj" if is_obj else "ply"
        if sampled < (n_faces or 0):
            source += "-sampled"
            detail = f"sampled {sampled:,} of {n_faces:,} faces; " + detail
        return FacetScale(meters=math.sqrt(mean_area) * abs(scale),
                          fine_meters=math.sqrt(_percentile(areas, 0.05))
                          * abs(scale),
                          n_faces=n_faces, source=source, detail=detail)

    body = _clean_body(config)
    topographic = bool(mesh.get("use_topographic", False))
    radius = _BODY_RADII_M.get(body)
    if radius is None:
        return FacetScale(detail=f"no reference radius for body {body!r}")
    try:
        res = int(mesh.get("resolution", 5))
    except (TypeError, ValueError):
        return FacetScale(detail="mesh.resolution is not an integer")
    if not 0 <= res <= 12:
        return FacetScale(detail=f"mesh.resolution {res} is out of range")
    n_faces = 20 * (4 ** res)
    total_area = 4.0 * math.pi * radius * radius
    if topographic:
        # A topographic run loads data/topo/<body>/topo_<body>_ico<N>.bin,
        # which is an ico-N mesh: MEASURED on this tree, ico4 -> 5,120 faces and
        # ico5 -> 20,480, both exactly 20*4^N (the engine's own
        # "topographic mesh with ... faces" line). The count is therefore the
        # same as the smooth branch. The displacement adds a little SURFACE
        # AREA, so the true facet scale is marginally smaller than this
        # estimate -- irrelevant against a 5 cm threshold when a lunar ico6
        # facet is 21 km, and said rather than assumed.
        return FacetScale(
            meters=math.sqrt(total_area / n_faces),
            fine_meters=math.sqrt(total_area / n_faces), n_faces=n_faces,
            source="icosphere-topographic",
            detail=f"{body} topography grid, resolution {res}; area from the "
                   f"smooth sphere, so the true scale is slightly smaller")
    return FacetScale(meters=math.sqrt(total_area / n_faces),
                      fine_meters=math.sqrt(total_area / n_faces),
                      n_faces=n_faces, source="icosphere",
                      detail=f"{body} sphere, resolution {res}")


def beaming_policy(config, project_root=None):
    """Whether the GUI may offer `radiation.beaming`, and what to tell the user."""
    body = _clean_body(config)

    # Body first, and only then the mesh. Measuring the facet scale decodes
    # every vertex of the PLY, and this runs from mesh.import_file's
    # per-keystroke textChanged; a Europa or asteroid project with a large
    # imported mesh was paying a multi-million-vertex decode on the GUI thread
    # for an answer that is "no" on body grounds regardless.
    if body not in BEAMING_BODIES:
        return BeamingPolicy(
            allowed=False, scale=FacetScale(detail="not measured: wrong body"),
            reason=(f"Beaming is offered for {', '.join(BEAMING_BODIES)} only. "
                    f"The Smith slope distribution is calibrated on lunar "
                    f"regolith (w = 32.5 deg, Jhoti et al. 2024), and "
                    f"docs/radiation-exchange.md section 8.4 declines to carry "
                    f"it to icy or small bodies. Selected body: "
                    f"{body or 'none'}."))

    scale = facet_scale(config, project_root)

    if scale.meters is None:
        # NOT a refusal. "This mesh is at the roughness scale" and "I could not
        # read this mesh" are different statements, and the module docstring
        # says so; treating them alike locked beaming out of every format this
        # reader does not parse, including ones the engine loads happily. Ask
        # instead: the acknowledgment tier is exactly "you decide, on the
        # record".
        return BeamingPolicy(
            allowed=True, needs_acknowledgment=True, scale=scale,
            reason=("The facet scale could not be MEASURED here, which is not "
                    "the same as the mesh being too fine -- this reader parses "
                    "PLY and OBJ geometry and derives icosphere and "
                    "topographic meshes from the resolution. "
                    f"{scale.detail.capitalize()}. Confirm yourself that this "
                    f"mesh's facets are far larger than the roughness scale "
                    f"(millimeters to centimeters) before enabling beaming."))

    # The GATE is the FINE END, not the mean. sqrt(mean area) is dominated by
    # the largest facets, so a mesh that is 99% millimeter-scale with a coarse
    # minority reports 70.7 cm and sails through a rule written to refuse
    # millimeter facets with margin. Beaming double-counts on the facets that
    # already resolve the roughness, so what matters is whether a meaningful
    # fraction of the mesh sits at that scale.
    #
    # Measured on the real meshes, sqrt(mean) vs sqrt(5th percentile):
    #   sunlit_radcore_r50   234 m  /  169 m   (ratio 1.4, near-uniform)
    #   A3GT A03MP026 core  3.47 m  / 0.708 m  (ratio 4.9, multi-resolution)
    #   A3GT ring            146 m  / 25.1 m   (ratio 5.8, by construction)
    # None has a millimeter-scale subpopulation, so no validated mesh is newly
    # refused; the A3GT core moves from allow to the acknowledgment tier, which
    # is the honest reading of 5% of its facets being sub-meter.
    fine = scale.fine_meters if scale.fine_meters is not None else scale.meters

    if fine < FACET_SCALE_DISABLE_BELOW_M:
        return BeamingPolicy(
            allowed=False, scale=scale,
            reason=(f"{scale.describe()}. The finest facets are at or near the "
                    f"roughness scale itself, so this mesh already resolves the "
                    f"roughness geometrically there; a sub-facet model on top "
                    f"would count it twice. The limit is "
                    f"{_fmt_length(FACET_SCALE_DISABLE_BELOW_M)} and the "
                    f"finest 5% of this mesh are {_fmt_length(fine)}."))

    if fine < FACET_SCALE_WARN_BELOW_M or scale.meters < FACET_SCALE_WARN_BELOW_M:
        return BeamingPolicy(
            allowed=True, needs_acknowledgment=True, scale=scale,
            reason=(f"{scale.describe()}. Between "
                    f"{_fmt_length(FACET_SCALE_DISABLE_BELOW_M)} and "
                    f"{_fmt_length(FACET_SCALE_WARN_BELOW_M)} the mesh is close "
                    f"enough to the roughness scale that some of the roughness "
                    f"may already be resolved. Confirm that this mesh does not "
                    f"resolve sub-facet slopes before enabling beaming."))

    return BeamingPolicy(allowed=True, scale=scale,
                         reason=f"{scale.describe()}.")
