"""
Headless regridder and mesh I/O for PTM3D output products.

Pure-Python (numpy + scipy + rasterio). No Qt / no GUI imports, so this module
is safe to import from CLI scripts.

NOTE: the projection helpers, GridResolutionEstimator, and MeshRegridder are
intentionally byte-for-byte equivalent to the implementations in
``gui/gui_core/data_exporter.py``. If either side changes, keep both in sync.
The PLY reader is the same minimal reader used by
``gui/viewshed_core/viewshed_data_loader.py``.
"""

import struct
from pathlib import Path

import numpy as np

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
