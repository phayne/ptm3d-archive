"""Readers and regridding for PTM3D output products.

This is the single implementation of "read a PTM3D run directory": the
per-face binary reader, the ``mesh.ply`` reader, statistics discovery, and the
unstructured-mesh-to-raster regridder. It imports numpy only (scipy is
optional and only needed for regridding), has no Qt and no matplotlib, and has
no import side effects, so it is safe to import from a script, a notebook or
the GUI alike.

It was previously carried twice: here (as ``src/python/ptm_regridder.py``) and
in ``gui/gui_core/data_exporter.py``, with a source comment on this side asking
that the two be kept in sync by hand. They had already diverged -- this copy
had gained a ``center_lon`` argument and a barycentric ``regrid_linear`` path
that the GUI copy never received. Both now import from here.

Two output binary formats exist in the tree and they are NOT the same:

* ``statistics/*.bin`` and the per-timestep ``*.bin``: ``int32`` face count
  (4 bytes) followed by ``n`` little-endian ``float32``. :func:`read_binary_data`
  reads these, including the header-less legacy variant.
* per-face albedo maps under ``data/albedo_maps/``: ``size_t`` count (8 bytes)
  followed by ``n`` ``float64``. Read by :func:`ptm.readers.read_albedo_map`.

:class:`Results`, at the bottom of this module, is the object over a whole run
directory that composes these functions with the readers in ``ptm.readers``.
"""

import hashlib
import struct
from pathlib import Path

import numpy as np

from . import readers

try:
    from scipy.spatial import cKDTree, Delaunay
    from scipy.interpolate import LinearNDInterpolator
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# --- Constants -------------------------------------------------------------

FIELD_INFO = {
    'temperature': {'label': 'Temperature', 'units': 'K'},
    'illumination': {'label': 'Total Illumination', 'units': 'W/m^2'},
    'direct_solar': {'label': 'Direct Solar', 'units': 'W/m^2'},
    'primary_reflected': {'label': 'Primary Reflected', 'units': 'W/m^2'},
    'primary_ir': {'label': 'Primary IR', 'units': 'W/m^2'},
}

BODY_RADII = {
    'Europa':    1560800.0,
    'Moon':      1737400.0,
    'Mercury':   2439700.0,
    'Mars':      3389500.0,
    'Enceladus':  252100.0,
    'Titan':     2574730.0,
    'Ceres':      473000.0,
}

STAT_TYPES = ['avg', 'max', 'min']
STAT_LABELS = {'avg': 'Average', 'max': 'Maximum', 'min': 'Minimum'}

STATISTICS_FIELDS = {
    'temperature':       {'label': 'Temperature',         'units': 'K'},
    'illumination':      {'label': 'Total Illumination',  'units': 'W/m^2'},
    'direct_solar':      {'label': 'Direct Solar',        'units': 'W/m^2'},
    'primary_reflected': {'label': 'Primary Reflected',   'units': 'W/m^2'},
    'primary_ir':        {'label': 'Primary IR',          'units': 'W/m^2'},
    # scalar=True: single per-face value (no avg/max/min split).
    'ice_depth':         {'label': 'Ice Depth', 'units': 'm', 'scalar': True},
}

NICE_RESOLUTIONS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.025, 0.05,
                    0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 5.0]

POLAR_LAT_THRESHOLD = 70.0

# The catalogs above spell flux units 'W/m^2'. The GUI writes the same units
# into GeoTIFF/HDF5 tags and into on-screen labels as 'W/m²'. Those strings end
# up in exported files, so the two spellings are a real difference and not a
# detail to unify silently; this returns whichever is asked for, from the one
# catalog.
_UNITS_UNICODE = {'W/m^2': 'W/m²'}


def field_catalog(catalog, units_style='ascii'):
    """Return ``catalog`` (FIELD_INFO or STATISTICS_FIELDS) with the unit
    strings rendered in ``units_style``: 'ascii' (``W/m^2``, as defined above)
    or 'unicode' (``W/m²``). The returned dict is a fresh copy."""
    if units_style not in ('ascii', 'unicode'):
        raise ValueError(f"units_style must be 'ascii' or 'unicode', got {units_style!r}")
    out = {}
    for key, info in catalog.items():
        info = dict(info)
        if units_style == 'unicode':
            info['units'] = _UNITS_UNICODE.get(info.get('units'), info.get('units'))
        out[key] = info
    return out


# --- Projection helpers ----------------------------------------------------

def _forward_polar_stereo(lat, lon, pole_lat, center_lon, body_radius):
    lat_r = np.radians(np.asarray(lat, dtype=np.float64))
    dlon_r = np.radians(np.asarray(lon, dtype=np.float64) - center_lon)

    if pole_lat > 0:
        rho = 2.0 * body_radius * np.tan(np.pi / 4.0 - lat_r / 2.0)
        x = rho * np.sin(dlon_r)
        y = -rho * np.cos(dlon_r)
    else:
        rho = 2.0 * body_radius * np.tan(np.pi / 4.0 + lat_r / 2.0)
        x = rho * np.sin(dlon_r)
        y = rho * np.cos(dlon_r)
    return x, y


def _inverse_polar_stereo(x, y, pole_lat, center_lon, body_radius):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    rho = np.sqrt(x**2 + y**2)

    if pole_lat > 0:
        lat = np.degrees(np.pi / 2.0 - 2.0 * np.arctan(rho / (2.0 * body_radius)))
        lon = center_lon + np.degrees(np.arctan2(x, -y))
    else:
        lat = np.degrees(-np.pi / 2.0 + 2.0 * np.arctan(rho / (2.0 * body_radius)))
        lon = center_lon + np.degrees(np.arctan2(x, y))
    return lat, lon


def choose_projection(latitudes):
    min_lat = float(np.nanmin(latitudes))
    max_lat = float(np.nanmax(latitudes))
    if min_lat > POLAR_LAT_THRESHOLD:
        return 'stere'
    elif max_lat < -POLAR_LAT_THRESHOLD:
        return 'stere'
    return 'eqc'


def detect_body_from_mesh(mesh_data):
    vertices = mesh_data.get('vertices')
    if vertices is None or len(vertices) == 0:
        return None

    vertices = np.asarray(vertices)
    radii = np.sqrt(np.sum(vertices**2, axis=1))
    mesh_radius = float(np.median(radii))

    best_body = None
    best_diff = float('inf')
    for name, radius_m in BODY_RADII.items():
        diff = abs(radius_m / 1000.0 - mesh_radius)
        if diff < best_diff:
            best_diff = diff
            best_body = name

    expected_km = BODY_RADII[best_body] / 1000.0
    if best_diff / expected_km < 0.05:
        return best_body
    return None


class GridResolutionEstimator:
    """Estimates optimal grid resolution from unstructured mesh face spacing."""

    @staticmethod
    def estimate_resolution(latitudes, longitudes):
        if not SCIPY_AVAILABLE:
            lat_range = np.ptp(latitudes)
            lon_range = np.ptp(longitudes)
            n_faces = len(latitudes)
            area = lat_range * lon_range
            raw = np.sqrt(area / n_faces)
            return GridResolutionEstimator._round_to_nice(raw)

        points = np.column_stack([longitudes, latitudes])
        tree = cKDTree(points)
        distances, _ = tree.query(points, k=2, workers=-1)
        raw_resolution = float(np.median(distances[:, 1]))
        return GridResolutionEstimator._round_to_nice(raw_resolution)

    @staticmethod
    def _round_to_nice(value):
        if value <= 0:
            return 0.01
        best = NICE_RESOLUTIONS[0]
        best_ratio = abs(np.log(value / best))
        for nice in NICE_RESOLUTIONS[1:]:
            ratio = abs(np.log(value / nice))
            if ratio < best_ratio:
                best = nice
                best_ratio = ratio
        return best


# --- Regridder -------------------------------------------------------------

class MeshRegridder:
    """Regrids unstructured face data to a regular projected grid.

    Auto-selects equirectangular vs polar stereographic from the latitude
    range, uses KDTree k=3 inverse-distance weighting.
    """

    def __init__(self, latitudes, longitudes, resolution, body_radius=1737400.0,
                 center_lon=None):
        """Build the regridding KDTree and (optionally cached) Delaunay.

        Args:
            latitudes, longitudes: per-face center coords (degrees).
            resolution: grid resolution. Degrees for eqc; converted to
                ``body_radius * radians(resolution)`` metres for stereo.
            body_radius: planetary body radius (m).
            center_lon: central meridian to use for the projection.
                ``None`` (default) auto-picks the mesh midpoint (a
                site-centered projection). Passing a fixed value lets
                multiple regridder instances share the same CRS - useful
                when several site COGs need to be loaded together in a
                GIS tool.
        """
        if not SCIPY_AVAILABLE:
            raise RuntimeError("scipy is required for regridding")

        self.resolution = resolution
        self.body_radius = body_radius
        self._forced_center_lon = center_lon

        lon = np.array(longitudes, dtype=np.float64)
        lat = np.array(latitudes, dtype=np.float64)
        if np.ptp(lon) > 300:
            lon = lon % 360.0

        min_lat = float(np.nanmin(lat))
        max_lat = float(np.nanmax(lat))

        if min_lat > POLAR_LAT_THRESHOLD:
            self.projection = 'stere'
            self.pole_lat = 90.0
        elif max_lat < -POLAR_LAT_THRESHOLD:
            self.projection = 'stere'
            self.pole_lat = -90.0
        else:
            self.projection = 'eqc'
            self.pole_lat = None

        if self.projection == 'stere':
            self._init_stereo(lat, lon)
        else:
            self._init_eqc(lat, lon)

    def _init_eqc(self, lat, lon):
        resolution = self.resolution
        if self._forced_center_lon is not None:
            self.center_lon = float(self._forced_center_lon)
        else:
            self.center_lon = float((np.nanmin(lon) + np.nanmax(lon)) / 2)

        lat_min = np.floor(np.nanmin(lat) / resolution) * resolution
        lat_max = np.ceil(np.nanmax(lat) / resolution) * resolution
        lon_min = np.floor(np.nanmin(lon) / resolution) * resolution
        lon_max = np.ceil(np.nanmax(lon) / resolution) * resolution

        self.lat_grid = np.arange(lat_min, lat_max + resolution * 0.5, resolution)
        self.lon_grid = np.arange(lon_min, lon_max + resolution * 0.5, resolution)
        self.n_lat = len(self.lat_grid)
        self.n_lon = len(self.lon_grid)

        valid_mask = np.isfinite(lon) & np.isfinite(lat)
        source_points = np.column_stack([lon[valid_mask], lat[valid_mask]])
        tree = cKDTree(source_points)

        LON, LAT = np.meshgrid(self.lon_grid, self.lat_grid)
        grid_points = np.column_stack([LON.ravel(), LAT.ravel()])

        self.distances, indices = tree.query(grid_points, k=3, workers=-1)
        valid_indices = np.where(valid_mask)[0]
        self.neighbor_faces = valid_indices[indices]

        self.zero_mask = (self.distances == 0.0)
        self.has_any_zeros = np.any(self.zero_mask)

        self._outside_mask = (
            self.distances[:, 0] > 1.5 * resolution
        ).reshape(self.n_lat, self.n_lon)

        # Cache for optional barycentric (LinearND) regrid path
        self._valid_face_indices = valid_indices
        self._source_points_2d = source_points
        self._grid_points_2d = grid_points
        self._delaunay = None

        self.x_grid = None
        self.y_grid = None

    def _init_stereo(self, lat, lon):
        if self._forced_center_lon is not None:
            center_lon = float(self._forced_center_lon)
        else:
            center_lon = float((np.nanmin(lon) + np.nanmax(lon)) / 2)
        self.center_lon = center_lon

        metric_res = self.body_radius * np.radians(self.resolution)

        valid_mask = np.isfinite(lon) & np.isfinite(lat)
        fx, fy = _forward_polar_stereo(
            lat[valid_mask], lon[valid_mask],
            self.pole_lat, center_lon, self.body_radius
        )

        x_min = np.floor(np.nanmin(fx) / metric_res) * metric_res
        x_max = np.ceil(np.nanmax(fx) / metric_res) * metric_res
        y_min = np.floor(np.nanmin(fy) / metric_res) * metric_res
        y_max = np.ceil(np.nanmax(fy) / metric_res) * metric_res

        self.x_grid = np.arange(x_min, x_max + metric_res * 0.5, metric_res)
        self.y_grid = np.arange(y_min, y_max + metric_res * 0.5, metric_res)
        self.n_lon = len(self.x_grid)
        self.n_lat = len(self.y_grid)

        source_points = np.column_stack([fx, fy])
        tree = cKDTree(source_points)

        X, Y = np.meshgrid(self.x_grid, self.y_grid)
        grid_points = np.column_stack([X.ravel(), Y.ravel()])

        self.distances, indices = tree.query(grid_points, k=3, workers=-1)
        valid_indices = np.where(valid_mask)[0]
        self.neighbor_faces = valid_indices[indices]

        self.zero_mask = (self.distances == 0.0)
        self.has_any_zeros = np.any(self.zero_mask)

        self._outside_mask = (
            self.distances[:, 0] > 1.5 * metric_res
        ).reshape(self.n_lat, self.n_lon)

        # Cache for optional barycentric (LinearND) regrid path
        self._valid_face_indices = valid_indices
        self._source_points_2d = source_points
        self._grid_points_2d = grid_points
        self._delaunay = None

        grid_lats, grid_lons = _inverse_polar_stereo(
            X.ravel(), Y.ravel(),
            self.pole_lat, center_lon, self.body_radius
        )
        self.lat_grid = grid_lats.reshape((self.n_lat, self.n_lon))
        self.lon_grid = grid_lons.reshape((self.n_lat, self.n_lon))

    def regrid(self, face_values):
        vals = face_values[self.neighbor_faces]
        valid_vals = np.isfinite(vals)

        if self.has_any_zeros:
            choose = self.zero_mask.argmax(axis=1)
            has_zero = self.zero_mask.any(axis=1)
            out = np.full(self.distances.shape[0], np.nan, dtype=np.float32)
            out[has_zero] = vals[
                np.arange(vals.shape[0])[has_zero], choose[has_zero]
            ].astype(np.float32)

            rem = ~has_zero
            if np.any(rem):
                d = self.distances[rem]
                v = vals[rem]
                vv = valid_vals[rem]
                w = 1.0 / (d**2 + 1e-12)
                w[~vv] = 0.0
                ws = w.sum(axis=1)
                good = ws > 0
                out_rem = np.full(rem.sum(), np.nan, dtype=np.float32)
                if np.any(good):
                    wn = w[good] / ws[good][:, None]
                    out_rem[good] = np.nansum(
                        wn * v[good], axis=1
                    ).astype(np.float32)
                out[rem] = out_rem
        else:
            w = 1.0 / (self.distances**2 + 1e-12)
            w[~valid_vals] = 0.0
            ws = w.sum(axis=1)
            out = np.full(self.distances.shape[0], np.nan, dtype=np.float32)
            good = ws > 0
            if np.any(good):
                wn = w[good] / ws[good][:, None]
                out[good] = np.nansum(
                    wn * vals[good], axis=1
                ).astype(np.float32)

        result = out.reshape((self.n_lat, self.n_lon))
        result[self._outside_mask] = np.nan
        return result

    def regrid_linear(self, face_values):
        """Barycentric (piecewise-linear) regrid via Delaunay over face centroids.

        Unlike ``regrid()`` (KDTree k=3 inverse-distance weighting + a tight
        nearest-centroid mask), this returns a value at every grid point
        inside the *convex hull* of the face centroids - filling all the
        internal "gaps" that arise when the projected face spacing is larger
        than the output pixel size. Pixels outside the convex hull are NaN.

        First call builds the Delaunay AND caches per-grid-point simplex
        membership + barycentric weights. Subsequent calls reuse all of
        that, so the per-field cost is a single indexing + 3-way dot
        product - dominant cost on the second-and-later fields is just
        the array allocation.
        """
        if not SCIPY_AVAILABLE:
            raise RuntimeError("scipy is required for regridding")

        if self._delaunay is None:
            tri = Delaunay(self._source_points_2d)
            self._delaunay = tri

            simplex = tri.find_simplex(self._grid_points_2d)
            inside = simplex >= 0
            self._lin_inside_flat = inside

            # Per-inside-pixel barycentric weights and vertex indices.
            inside_simplex = simplex[inside]
            T = tri.transform[inside_simplex]            # (M, 3, 2)
            grid_in = self._grid_points_2d[inside]       # (M, 2)
            b01 = np.einsum('ijk,ik->ij',
                            T[:, :2, :], grid_in - T[:, 2, :])
            b2 = 1.0 - b01.sum(axis=1)
            self._lin_weights = np.column_stack([b01, b2]).astype(np.float64)
            self._lin_vertex_idx = tri.simplices[inside_simplex]

        vals = np.asarray(face_values, dtype=np.float64)[self._valid_face_indices]
        vert_vals = vals[self._lin_vertex_idx]                      # (M, 3)
        out = np.full(self._grid_points_2d.shape[0], np.nan,
                      dtype=np.float32)
        out[self._lin_inside_flat] = np.einsum(
            'ij,ij->i', self._lin_weights, vert_vals
        ).astype(np.float32)
        return out.reshape((self.n_lat, self.n_lon))

    def get_grid_shape(self):
        return (self.n_lat, self.n_lon)


# --- File I/O --------------------------------------------------------------

def read_binary_data(file_path, expected_count):
    """Read a PTM3D per-face binary file (uint32 count + float32 array)."""
    try:
        with open(file_path, 'rb') as f:
            header = f.read(4)
            if len(header) == 4:
                count = struct.unpack('<I', header)[0]
                f.seek(0, 2)
                file_size = f.tell()

                if file_size == 4 + count * 4 and count == expected_count:
                    f.seek(4)
                    return np.frombuffer(f.read(), dtype='<f4').copy()
                else:
                    f.seek(0)
                    data = np.frombuffer(f.read(), dtype='<f4').copy()
                    if len(data) == expected_count:
                        return data
        return None
    except Exception:
        return None


def read_ply_mesh_minimal(mesh_path):
    """Minimal PLY reader (ASCII + binary_little_endian).

    Returns a dict with vertices, faces, face_centers, face_normals, face_areas,
    face_latitudes, face_longitudes. lat/lon are computed from face-centroid
    Cartesian coordinates (assumes body-centered mesh in km or m).
    """
    mesh_path = Path(mesh_path)
    vertices = []
    faces = []

    _ply_type_map = {
        'float': '<f4', 'float32': '<f4',
        'double': '<f8', 'float64': '<f8',
        'int': '<i4', 'int32': '<i4',
        'uint': '<u4', 'uint32': '<u4',
        'uchar': '<u1', 'uint8': '<u1',
        'char': '<i1', 'int8': '<i1',
        'short': '<i2', 'int16': '<i2',
        'ushort': '<u2', 'uint16': '<u2',
    }

    with open(mesh_path, "rb") as f:
        is_binary = False
        n_vertices = 0
        n_faces = 0
        vertex_props = []
        face_scalar_props = []
        current_element = None
        while True:
            line = f.readline().decode("ascii", errors="replace").strip()
            if line.startswith("element vertex"):
                n_vertices = int(line.split()[-1])
                current_element = 'vertex'
            elif line.startswith("element face"):
                n_faces = int(line.split()[-1])
                current_element = 'face'
            elif line.startswith("property list"):
                pass
            elif line.startswith("property") and current_element == 'vertex':
                parts = line.split()
                vertex_props.append((parts[2], _ply_type_map.get(parts[1], '<f4')))
            elif line.startswith("property") and current_element == 'face':
                parts = line.split()
                face_scalar_props.append((parts[2], _ply_type_map.get(parts[1], '<f4')))
            elif "binary_little_endian" in line:
                is_binary = True
            elif line == "end_header":
                break

        if is_binary:
            vertex_dtype = (np.dtype(vertex_props) if vertex_props
                            else np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4')]))
            vertex_data = np.frombuffer(
                f.read(n_vertices * vertex_dtype.itemsize), dtype=vertex_dtype
            )
            vertices = np.column_stack(
                [vertex_data['x'], vertex_data['y'], vertex_data['z']]
            ).astype(np.float64)
            face_extra_bytes = sum(np.dtype(dt).itemsize for _, dt in face_scalar_props)
            faces = np.zeros((n_faces, 3), dtype=np.int32)
            for i in range(n_faces):
                count = struct.unpack('<B', f.read(1))[0]
                idx = np.frombuffer(f.read(count * 4), dtype="<i4")
                faces[i] = idx[:3]
                if face_extra_bytes > 0:
                    f.read(face_extra_bytes)
        else:
            for _ in range(n_vertices):
                parts = f.readline().decode().split()
                vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])
            vertices = np.array(vertices, dtype=np.float64)

            for _ in range(n_faces):
                parts = f.readline().decode().split()
                n = int(parts[0])
                faces.append([int(parts[j + 1]) for j in range(min(n, 3))])
            faces = np.array(faces, dtype=np.int32)

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)

    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    face_centers = (v0 + v1 + v2) / 3.0
    edge1 = v1 - v0
    edge2 = v2 - v0
    cross = np.cross(edge1, edge2)
    norms = np.linalg.norm(cross, axis=1, keepdims=True)
    face_normals = cross / np.maximum(norms, 1e-30)
    face_areas = 0.5 * norms.ravel()

    r = np.linalg.norm(face_centers, axis=1)
    face_latitudes = np.degrees(np.arcsin(
        np.clip(face_centers[:, 2] / np.maximum(r, 1e-30), -1, 1)
    ))
    face_longitudes = np.degrees(np.arctan2(face_centers[:, 1], face_centers[:, 0]))

    return {
        "n_vertices": len(vertices),
        "n_faces": len(faces),
        "vertices": vertices,
        "faces": faces,
        "face_centers": face_centers,
        "face_normals": face_normals,
        "face_areas": face_areas,
        "face_latitudes": face_latitudes,
        "face_longitudes": face_longitudes,
    }


# --- Stats binary discovery ------------------------------------------------

def detect_available_statistics(stats_dir):
    """Return a list of (base_field, stat_type) tuples for files that exist
    and contain non-NaN data. ``stat_type`` is None for scalar fields like
    ``ice_depth`` (file is ``<base>.bin`` with no _avg/_max/_min suffix).
    """
    stats_dir = Path(stats_dir)
    if not stats_dir.is_dir():
        return []

    def _file_has_data(filepath):
        if not filepath.is_file() or filepath.stat().st_size <= 4:
            return False
        try:
            with open(filepath, 'rb') as f:
                header = f.read(4)
                if len(header) != 4:
                    return False
                count = struct.unpack('<I', header)[0]
                data = np.frombuffer(f.read(), dtype='<f4')
                return len(data) == count and not np.all(np.isnan(data))
        except Exception:
            return False

    available = []
    for base_field, info in STATISTICS_FIELDS.items():
        if info.get('scalar'):
            if _file_has_data(stats_dir / f"{base_field}.bin"):
                available.append((base_field, None))
            continue
        for stat_type in STAT_TYPES:
            if _file_has_data(stats_dir / f"{base_field}_{stat_type}.bin"):
                available.append((base_field, stat_type))
    return available


# --- A run directory as an object ------------------------------------------
#
# Everything above is a function over a path. Results is the object over a run
# directory that composes them, so that "open this run and show me a picture"
# is a line of code rather than the 250 lines of readers the four tutorial
# notebooks each re-derive. It adds no reader that is not already above or in
# ptm.readers, and it computes no physics.

class Results:
    """A finished PTM3D run directory.

    ::

        from ptm import Results
        run = Results("output/quickstart_moon_diurnal")
        run.stat("temperature", "max")          # per-face array
        t, T = run.series("temperature", lat=0, lon=0)
        run.plot_diurnal(lat=0, lon=0, save="diurnal.png")

    The constructor reads only ``metadata.txt``; the mesh, the time axis and
    the body positions load on first use and are then cached.

    Two things a run directory does not record are settled here rather than
    guessed at every call site:

    ``mesh_units``
        ``mesh.ply`` carries no length unit and is metres for a topographic
        run, kilometres for a plain icosphere. :attr:`mesh_units` states which,
        by the engine's own rule (:func:`ptm.readers.infer_mesh_length_unit`),
        and :attr:`face_centers_m` / :attr:`face_areas_m2` are normalized. Pass
        ``mesh_units=`` or ``body_radius_km=`` for a mesh the rule refuses.
    ``units_style``
        Flux units are spelled ``W/m^2`` by the catalogs in this module and
        ``W/m²`` by the GUI, and the GUI's spelling is written into exported
        GeoTIFF and HDF5 tags. Results carries the choice
        (``units_style='ascii'`` or ``'unicode'``) instead of picking one, and
        :meth:`units_of` and every axis label follow it.
    """

    #: The per-step field directories the engine writes on a default run, in
    #: the order ``metadata.txt`` lists them. This is a preferred *ordering*,
    #: not the set: :attr:`step_fields` discovers what is actually on disk,
    #: because a two-band radiosity run also writes ``indirect_solar`` and
    #: ``indirect_ir``, and a hard-coded list drops them silently.
    STEP_FIELD_ORDER = ('illumination', 'direct_solar', 'primary_reflected',
                        'primary_ir', 'indirect_solar', 'indirect_ir',
                        'temperature')

    #: Labels and units for products ``STATISTICS_FIELDS`` does not carry.
    #: ``effective_albedo.bin`` is written by every run and the two ``indirect``
    #: bands by every two-band radiosity run, yet none of the three appears in
    #: that catalog, so ``detect_available_statistics`` does not report them.
    #: Kept separate rather than added to the catalog, because the GUI builds
    #: its field menu from the catalog and gaining three entries there is a
    #: change to the GUI, not to this class.
    EXTRA_FIELD_INFO = {
        'indirect_solar': {'label': 'Indirect Solar', 'units': 'W/m^2'},
        'indirect_ir': {'label': 'Indirect IR', 'units': 'W/m^2'},
        'effective_albedo': {'label': 'Effective Albedo', 'units': '',
                             'scalar': True},
    }

    def __init__(self, run_dir, units_style='ascii', mesh_units=None,
                 body_radius_km=None):
        self.path = Path(run_dir).expanduser().resolve()
        if not self.path.is_dir():
            raise FileNotFoundError(f"{self.path} is not a directory")
        if units_style not in ('ascii', 'unicode'):
            raise ValueError(
                f"units_style must be 'ascii' or 'unicode', got {units_style!r}")
        self.units_style = units_style
        self._forced_mesh_units = mesh_units
        self._body_radius_km = body_radius_km

        self.metadata = self._read_metadata()
        #: What ``provenance.txt`` says, or ``{}`` on a run written before the
        #: engine recorded any (see :meth:`provenance`).
        self.engine_provenance = self._read_engine_provenance()
        if not self.metadata and not (self.path / "mesh.ply").is_file():
            raise FileNotFoundError(
                f"{self.path} has neither metadata.txt nor mesh.ply; it does not "
                f"look like a PTM3D run directory")

        self._mesh = None
        self._mesh_unit_report = None
        self._times = None
        self._positions = None
        self._regridders = {}

    def __repr__(self):
        return (f"<Results {self.path.name}: {self.n_faces} faces, "
                f"{len(self.step_fields)} per-step field(s), "
                f"{self.metadata.get('time_steps', '?')} steps>")

    # -- metadata.txt --------------------------------------------------------

    def _read_engine_provenance(self):
        """``provenance.txt``, written by the engine since schema
        ``ptm3d-provenance-1``.

        Absent on every run produced before the engine recorded its own
        identity, which is why nothing here raises: an old run directory is a
        run directory with less to say about itself, not a broken one.

        Values stay strings. ``engine_binary_md5`` is a hex digest and
        ``mesh_faces`` is a count, but mixing types in one flat mapping is how
        a caller ends up comparing an int to the string the engine wrote; the
        two fields that are genuinely typed are surfaced by
        :meth:`provenance` instead.
        """
        path = self.path / "provenance.txt"
        if not path.is_file():
            return {}
        out = {}
        for line in path.read_text().splitlines():
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            out[key.strip()] = value.strip()
        return out

    def _read_metadata(self):
        path = self.path / "metadata.txt"
        if not path.is_file():
            return {}
        out = {}
        for line in path.read_text().splitlines():
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key, value = key.strip(), value.strip()
            if value in ("true", "false"):
                out[key] = (value == "true")
            elif key == "components":
                out[key] = tuple(v for v in value.split(",") if v)
            else:
                try:
                    out[key] = int(value)
                except ValueError:
                    out[key] = value
        return out

    # -- mesh ----------------------------------------------------------------

    @property
    def mesh(self):
        """``read_ply_mesh_minimal`` output for this run's ``mesh.ply``.

        The per-face ``latitude``/``longitude`` properties stored in the file
        are not re-read: this dict's computed values reproduce them to float32
        rounding (measured max difference 3.6e-06 deg in latitude, 7.6e-06 deg
        in longitude on ``quickstart_moon_diurnal``).
        """
        if self._mesh is None:
            path = self.path / "mesh.ply"
            if not path.is_file():
                raise FileNotFoundError(f"{path} not found")
            self._mesh = read_ply_mesh_minimal(path)
            declared = self.metadata.get("faces")
            if declared is not None and declared != self._mesh["n_faces"]:
                raise ValueError(
                    f"{self.path}: metadata.txt says faces={declared} but "
                    f"mesh.ply has {self._mesh['n_faces']}; the run directory "
                    f"is inconsistent")
        return self._mesh

    @property
    def n_faces(self):
        faces = self.metadata.get("faces")
        return int(faces) if faces is not None else int(self.mesh["n_faces"])

    @property
    def lat(self):
        """Per-face latitude, degrees."""
        return self.mesh["face_latitudes"]

    @property
    def lon(self):
        """Per-face longitude, degrees on [-180, 180]."""
        return self.mesh["face_longitudes"]

    @property
    def mesh_unit_report(self):
        """The full :class:`ptm.readers.MeshUnitReport` for this mesh."""
        if self._mesh_unit_report is None:
            mesh = self.mesh
            if self._forced_mesh_units is not None:
                if self._forced_mesh_units not in ("m", "km"):
                    raise ValueError("mesh_units must be 'm' or 'km', got "
                                     f"{self._forced_mesh_units!r}")
                mag = np.linalg.norm(mesh["face_centers"], axis=1)
                self._mesh_unit_report = readers.MeshUnitReport(
                    unit=self._forced_mesh_units, median=float(np.median(mag)),
                    lo=float(mag.min()), hi=float(mag.max()),
                    corroborated=False, body=None)
            else:
                self._mesh_unit_report = readers.infer_mesh_length_unit(
                    mesh["face_centers"], vertices=mesh["vertices"],
                    body_radius_km=self._body_radius_km)
        return self._mesh_unit_report

    @property
    def mesh_units(self):
        """``'m'`` or ``'km'`` -- the unit ``mesh.ply``'s coordinates are in."""
        return self.mesh_unit_report.unit

    @property
    def _to_m(self):
        return 1000.0 if self.mesh_units == "km" else 1.0

    @property
    def face_centers_m(self):
        """Face centers in metres, whatever the mesh file's unit is."""
        return self.mesh["face_centers"] * self._to_m

    @property
    def face_areas_m2(self):
        """Face areas in square metres, whatever the mesh file's unit is."""
        return self.mesh["face_areas"] * (self._to_m ** 2)

    @property
    def face_unit_vectors(self):
        """Unit vectors of the face centers -- unit-independent by definition."""
        centers = self.mesh["face_centers"]
        return centers / np.linalg.norm(centers, axis=1, keepdims=True)

    # -- fields and statistics ------------------------------------------------

    @property
    def step_fields(self):
        """Per-step field names this run wrote, discovered from the directory.

        A directory counts when it holds at least one ``<name>/<name>_NNNN.bin``.
        Known names come first in :data:`STEP_FIELD_ORDER`; anything else the
        engine grows follows, sorted, rather than being dropped.
        """
        found = set()
        for entry in self.path.iterdir():
            if not entry.is_dir():
                continue
            if next(entry.glob(f"{entry.name}_[0-9]*.bin"), None) is not None:
                found.add(entry.name)
        ordered = [n for n in self.STEP_FIELD_ORDER if n in found]
        return tuple(ordered + sorted(found.difference(ordered)))

    @property
    def stats(self):
        """``[(field, moment), ...]`` present under ``statistics/``.

        ``moment`` is ``'avg'``/``'max'``/``'min'``, or ``None`` for a scalar
        product such as ``ice_depth`` or ``effective_albedo``.

        ``detect_available_statistics`` is asked first, then the directory is
        scanned for anything it does not know about -- it iterates a fixed
        catalog, so on a two-band radiosity run it silently omits
        ``indirect_solar_*``, ``indirect_ir_*`` and ``effective_albedo``, all
        of which are on disk.
        """
        stats_dir = self.path / "statistics"
        known = detect_available_statistics(stats_dir)
        if not stats_dir.is_dir():
            return known
        seen = set(known)
        extra = []
        for path in stats_dir.glob("*.bin"):
            if path.stat().st_size <= 4:
                continue
            stem = path.stem
            base, _, suffix = stem.rpartition("_")
            entry = ((base, suffix) if base and suffix in STAT_TYPES
                     else (stem, None))
            if entry not in seen:
                seen.add(entry)
                extra.append(entry)
        return known + sorted(extra)

    def steps(self, field='temperature'):
        """Sorted step indices for which ``field`` has a per-step file."""
        directory = self.path / field
        if not directory.is_dir():
            raise FileNotFoundError(
                f"{directory} not found; this run wrote {self.step_fields}")
        out = []
        for path in directory.glob(f"{field}_*.bin"):
            token = path.stem[len(field) + 1:]
            if token.isdigit():
                out.append(int(token))
        return sorted(out)

    def _read(self, path):
        values = read_binary_data(path, self.n_faces)
        if values is None:
            raise ValueError(
                f"{path}: not a readable per-face binary of {self.n_faces} "
                f"values (expected a 4-byte int32 count then that many "
                f"float32, or a headerless array of the same length)")
        return values

    def stat(self, field='temperature', moment='max'):
        """A per-face statistic, e.g. ``stat('temperature', 'max')``.

        ``moment=None`` reads the scalar form ``statistics/<field>.bin``.
        """
        name = field if moment is None else f"{field}_{moment}"
        return self._read(self.path / "statistics" / f"{name}.bin")

    def field(self, field='temperature', step=0):
        """One per-step per-face field, e.g. ``field('temperature', 120)``."""
        return self._read(self.path / field / f"{field}_{int(step):04d}.bin")

    def albedo_map(self, path):
        """Read a per-face albedo map and check it against this run's mesh.

        Separate from :meth:`field` because it is a different binary layout --
        ``size_t`` count and float64 -- and lives outside the run directory,
        under ``data/albedo_maps/``.
        """
        return readers.read_albedo_map(path, expected_count=self.n_faces)

    # -- time ----------------------------------------------------------------

    def times(self):
        """The output time axis as a :class:`ptm.readers.TimeAxis`.

        ``.et`` is SPICE ET (seconds past J2000), which is what column 1 of
        ``time_steps.txt`` actually holds; ``.elapsed_s`` is that column rebased
        to the first output step, which is what its header claims it is.
        """
        if self._times is None:
            self._times = readers.read_time_steps(self.path / "time_steps.txt")
        return self._times

    @property
    def elapsed_hours(self):
        return self.times().elapsed_s / 3600.0

    def body_positions(self):
        """``body_positions.csv`` as a structured array (see readers)."""
        if self._positions is None:
            self._positions = readers.read_body_positions(
                self.path / "body_positions.csv")
        return self._positions

    def subsolar_latlon(self, step=0):
        """Body-fixed sub-solar (lat, lon) in degrees at ``step``."""
        return readers.subsolar_latlon(self.body_positions(), step)

    # -- selection -------------------------------------------------------------

    def face_at(self, lat, lon):
        """Index of the face nearest ``(lat, lon)``.

        Returns just the index; :meth:`face_at_verbose` also returns the
        angular separation, which is worth printing when the mesh is coarse.
        """
        return self.face_at_verbose(lat, lon)[0]

    def face_at_verbose(self, lat, lon):
        """``(index, separation_degrees)`` for the face nearest ``(lat, lon)``."""
        return readers.find_nearest_face(lat, lon, self.lat, self.lon)

    def faces_in_band(self, lat_min, lat_max):
        """Indices of faces whose center latitude is within ``[lat_min, lat_max]``."""
        lat = self.lat
        return np.where((lat >= lat_min) & (lat <= lat_max))[0]

    def resolve_face(self, face=None, lat=None, lon=None):
        """A face index from either spec: ``face=`` or ``lat=``/``lon=``.

        The one place that decides what "which face?" means, so every entry
        point that accepts both spellings agrees on it.
        """
        if face is not None:
            if lat is not None or lon is not None:
                raise ValueError("give either face= or lat=/lon=, not both")
            return int(face)
        if lat is None or lon is None:
            raise ValueError("give face=, or both lat= and lon=")
        return self.face_at(lat, lon)

    # -- series ----------------------------------------------------------------

    def series(self, field='temperature', face=None, lat=None, lon=None,
               steps=None):
        """One face's time series: ``(elapsed_hours, values)``.

        The single most repeated pattern in the notebooks and the survey
        scripts. ``face=`` takes an index, ``lat=``/``lon=`` take degrees and go
        through :meth:`face_at`.
        """
        index = self.resolve_face(face, lat, lon)
        if steps is None:
            steps = self.steps(field)
        values = np.array([self.field(field, s)[index] for s in steps])
        hours = self.elapsed_hours
        if len(hours) >= len(steps):
            hours = hours[list(steps)]
        else:
            raise ValueError(
                f"time_steps.txt has {len(hours)} rows but {len(steps)} "
                f"{field} files exist; the run directory is inconsistent")
        return hours, values

    def stack(self, field='temperature', steps=None, faces=None):
        """``(n_steps, n_faces)`` for a per-step field. Reads every step.

        ``faces`` restricts the columns as each step is read, so a band of a
        large mesh costs the band rather than the mesh: the fig-8 core is
        140,091 faces, and 240 steps of it is 134 MB before the selection.
        """
        if steps is None:
            steps = self.steps(field)
        if faces is None:
            return np.stack([self.field(field, s) for s in steps])
        faces = np.asarray(faces)
        return np.stack([self.field(field, s)[faces] for s in steps])

    def local_solar_time_curve(self, field='temperature', lat_max=10.0,
                               n_bins=48, steps=None,
                               noon_field='direct_solar'):
        """Noon-referenced diurnal curve over a latitude band.

        Returns ``(local_solar_time_hours, mean, p25, p75, n_faces_used)``.
        ``steps=None`` uses every output step; the asteroid notebooks pass the
        second of two rotations, and that choice stays with the caller.
        """
        if steps is None:
            steps = self.steps(field)
        band = self.faces_in_band(-abs(lat_max), abs(lat_max))
        if band.size == 0:
            raise ValueError(f"no faces within +/-{lat_max} deg of the equator")
        values = self.stack(field, steps, faces=band)
        reference = self.stack(noon_field, steps, faces=band)
        centers, mean, p25, p75 = readers.local_solar_time_bins(
            values, reference, n_bins=n_bins)
        return centers, mean, p25, p75, int(band.size)

    # -- regridding ------------------------------------------------------------

    def to_grid(self, values, resolution=None, linear=False):
        """Regrid a per-face field to a raster: ``(grid, regridder)``.

        Uses :class:`MeshRegridder`, which picks equirectangular or polar
        stereographic from the latitude range. ``resolution=None`` asks
        :class:`GridResolutionEstimator` for one. Needs scipy
        (``pip install 'ptm[regrid]'``); nothing here silently falls back to a
        coarser method.
        """
        if not SCIPY_AVAILABLE:
            raise RuntimeError(
                "regridding needs scipy: pip install 'ptm[regrid]'")
        values = np.asarray(values)
        if values.shape[0] != self.n_faces:
            raise ValueError(f"expected {self.n_faces} face values, got "
                             f"{values.shape[0]}")
        if resolution is None:
            resolution = GridResolutionEstimator.estimate_resolution(
                self.lat, self.lon)
        key = float(resolution)
        if key not in self._regridders:
            self._regridders[key] = MeshRegridder(
                self.lat, self.lon, key,
                body_radius=float(np.median(np.linalg.norm(
                    self.face_centers_m, axis=1))))
        regridder = self._regridders[key]
        grid = (regridder.regrid_linear(values) if linear
                else regridder.regrid(values))
        return grid, regridder

    # -- units -----------------------------------------------------------------

    def field_catalog(self):
        """The field catalog in this run's :attr:`units_style`.

        ``STATISTICS_FIELDS`` plus :data:`EXTRA_FIELD_INFO`, so a field that
        exists on disk always has a label and a unit.
        """
        merged = dict(STATISTICS_FIELDS)
        merged.update(self.EXTRA_FIELD_INFO)
        return field_catalog(merged, units_style=self.units_style)

    def units_of(self, field):
        """The unit string for ``field``, spelled in this run's units style."""
        info = self.field_catalog().get(field)
        return info["units"] if info else ""

    def label_of(self, field):
        """A human label for ``field`` (``'Temperature'``, ...)."""
        info = self.field_catalog().get(field)
        return info["label"] if info else field.replace("_", " ").title()

    def axis_label(self, field, moment=None):
        """``'Maximum Temperature (K)'`` -- an axis label for a field/moment."""
        label = self.label_of(field)
        if moment:
            label = f"{STAT_LABELS.get(moment, moment.title())} {label}"
        units = self.units_of(field)
        return f"{label} ({units})" if units else label

    # -- provenance --------------------------------------------------------------

    def provenance(self):
        """What this run records about itself, and what it does not.

        Since engine schema ``ptm3d-provenance-1`` the engine writes
        ``provenance.txt`` (version, git SHA, build type, GPU flag, compiler,
        the executable's own md5, the full command line, the body, the mesh
        source and the mesh length unit it actually used) and a verbatim
        ``config_used.yaml``. Those fields are reported here under ``engine``,
        and the corresponding ``not_recorded`` entries disappear.

        On a run made before that -- every run in the archive -- there is only
        ``metadata.txt``: face count, step counts, exported components. The
        missing identity is listed in ``not_recorded`` rather than inferred,
        because a plausible-looking guess about which binary produced a run is
        worse than an admission that the run does not say.

        Two things stay honest even when the record exists. ``config_used.yaml``
        is the configuration file as read, not a dump of the resolved
        configuration -- ``engine['command_line']`` carries any CLI override on
        top of it. And ``engine_binary_md5`` is the md5 of the *executable*: if
        ``engine['engine_shared_libraries']`` is ``"true"``, a change confined
        to one shared project library would not move it, and that appears in
        ``not_recorded`` too.
        """
        meta = dict(self.metadata)
        engine = dict(self.engine_provenance)
        not_recorded = []

        config_path, config_sha256 = None, None
        recorded_copy = engine.get("config_copy")
        if recorded_copy and recorded_copy != "none":
            # The engine named the file, so a driver that copied a second YAML
            # in cannot make this ambiguous.
            candidate = self.path / recorded_copy
            if candidate.is_file():
                config_path = candidate
            else:
                not_recorded.append(
                    f"config (provenance.txt names {recorded_copy}, which is "
                    f"not in the run directory)")
        if config_path is None and not recorded_copy:
            yamls = sorted(p for p in self.path.glob("*.yaml"))
            if len(yamls) == 1:
                config_path = yamls[0]
            elif len(yamls) > 1:
                not_recorded.append(
                    f"config (the run directory holds {len(yamls)} YAML files; "
                    f"none is marked as the one used)")
            else:
                not_recorded.append(
                    "config (this run predates provenance.txt; the engine wrote "
                    "no copy of the YAML it ran)")
        if config_path is not None:
            config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()

        if not engine:
            not_recorded.append(
                "engine binary md5 / version / git SHA (this run predates "
                "provenance.txt)")
            not_recorded.append("command line (this run predates provenance.txt)")
            not_recorded.append("body name (infer it from the mesh radius if needed)")
        elif engine.get("engine_shared_libraries") == "true":
            not_recorded.append(
                "complete engine identity: engine_binary_md5 covers the "
                "executable only, and this build links shared project "
                "libraries, so a change confined to one of them would not "
                "move it")
        not_recorded.append("field units (this package's catalog supplies them)")

        started = engine.get("run_started_utc")
        heartbeat = self.path / "heartbeat.txt"
        if started is None and heartbeat.is_file():
            text = heartbeat.read_text().strip()
            if text:
                started = text.split()[0]

        try:
            times = self.times()
            span = {"n_steps": len(times.et),
                    "first_utc": times.utc[0] if times.utc else None,
                    "last_utc": times.utc[-1] if times.utc else None,
                    "elapsed_hours": float(times.elapsed_s[-1] / 3600.0),
                    "cadence_hours": (float(np.median(np.diff(times.elapsed_s))
                                            / 3600.0)
                                      if len(times.et) > 1 else None)}
        except (FileNotFoundError, ValueError) as exc:
            span = None
            not_recorded.append(f"time axis ({exc})")

        report = self.mesh_unit_report
        # The engine states the unit it actually used; this reader infers one
        # from the mesh. Disagreement means one of the two is wrong about the
        # geometry, which is worth saying out loud rather than quietly
        # preferring either.
        engine_unit = engine.get("mesh_length_unit")
        # The engine spells the unit out; this package uses 'm'/'km'. Compare
        # the normalized forms, or every run reports a disagreement that is
        # only a difference of vocabulary.
        engine_unit_short = {"meters": "m", "kilometers": "km"}.get(engine_unit)
        if engine_unit_short and engine_unit_short != report.unit:
            not_recorded.append(
                f"an agreed mesh length unit: the engine recorded "
                f"{engine_unit!r}, this reader infers {report.unit!r}")

        return {
            "run_dir": str(self.path),
            "n_faces": self.n_faces,
            "metadata": meta,
            "engine": engine,
            "body": engine.get("body"),
            "engine_version": engine.get("engine_version"),
            "engine_git_sha": engine.get("engine_git_sha"),
            "engine_binary_md5": engine.get("engine_binary_md5"),
            "command_line": engine.get("command_line"),
            "mesh_units": report.unit,
            "mesh_units_recorded_by_engine": engine_unit,
            "mesh_units_corroborated_by": report.body,
            "mesh_median_radius_m": report.median * self._to_m,
            "step_fields": self.step_fields,
            "statistics": self.stats,
            "time": span,
            "started_utc": started,
            "config_file": str(config_path) if config_path else None,
            "config_sha256": config_sha256,
            "config_md5_recorded": (engine.get("config_md5")
                                    if engine.get("config_md5") not in (None, "none")
                                    else None),
            "units_style": self.units_style,
            "not_recorded": tuple(not_recorded),
        }

    # -- canned plots ----------------------------------------------------------
    # Thin delegations, so that importing ptm.results never imports matplotlib.

    def plot_diurnal(self, **kwargs):
        """Time series at one face. See :func:`ptm.plots.plot_diurnal`."""
        from . import plots
        return plots.plot_diurnal(self, **kwargs)

    def plot_map(self, *args, **kwargs):
        """Regridded lat/lon map. See :func:`ptm.plots.plot_map`."""
        from . import plots
        return plots.plot_map(self, *args, **kwargs)

    def plot_globes(self, *args, **kwargs):
        """Orthographic globe set. See :func:`ptm.plots.plot_globes`."""
        from . import plots
        return plots.plot_globes(self, *args, **kwargs)

    def plot_histogram(self, *args, **kwargs):
        """Distribution of a per-face field. See :func:`ptm.plots.plot_histogram`."""
        from . import plots
        return plots.plot_histogram(self, *args, **kwargs)
