"""
Data Exporter for PTM3D GUI
Handles exporting simulation data in GeoTIFF and HDF5 formats.
Includes regridding from unstructured triangular mesh to regular grid.
Automatically selects equirectangular or polar stereographic projection.
"""

import struct
from pathlib import Path
from datetime import datetime

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

# Optional dependency guards
try:
    from scipy.spatial import cKDTree
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False

try:
    import vtk
    from vtk.util import numpy_support
    VTK_AVAILABLE = True
except ImportError:
    VTK_AVAILABLE = False


# Field metadata
FIELD_INFO = {
    'temperature': {'label': 'Temperature', 'units': 'K'},
    'illumination': {'label': 'Total Illumination', 'units': 'W/m²'},
    'direct_solar': {'label': 'Direct Solar', 'units': 'W/m²'},
    'primary_reflected': {'label': 'Primary Reflected', 'units': 'W/m²'},
    'primary_ir': {'label': 'Primary IR', 'units': 'W/m²'},
}

# Planetary body radii in meters
BODY_RADII = {
    'Europa': 1560800.0,
    'Moon': 1737400.0,
    'Mercury': 2439700.0,
    'Mars': 3389500.0,
    'Enceladus': 252100.0,
    'Titan': 2574730.0,
    'Ceres': 473000.0,
}

# Statistics metadata
STAT_TYPES = ['avg', 'max', 'min']
STAT_LABELS = {'avg': 'Average', 'max': 'Maximum', 'min': 'Minimum'}

STATISTICS_FIELDS = {
    'temperature': {'label': 'Temperature', 'units': 'K'},
    'illumination': {'label': 'Total Illumination', 'units': 'W/m²'},
    'direct_solar': {'label': 'Direct Solar', 'units': 'W/m²'},
    'primary_reflected': {'label': 'Primary Reflected', 'units': 'W/m²'},
    'primary_ir': {'label': 'Primary IR', 'units': 'W/m²'},
    # scalar=True: single per-face value (no avg/max/min split). On disk as
    # ice_depth.bin (Schorghofer J(z)=J0 stability depth, not a time-average).
    'ice_depth': {'label': 'Ice Depth', 'units': 'm', 'scalar': True},
}

# Nice resolution values for rounding
NICE_RESOLUTIONS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.025, 0.05,
                    0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 5.0]

# Latitude threshold for switching to polar stereographic projection.
# Data where ALL faces are within this many degrees of a pole uses stere.
POLAR_LAT_THRESHOLD = 70.0


def detect_body_from_mesh(mesh_data):
    """Detect planetary body from mesh vertex distances from origin.

    The PTM3D simulation stores mesh vertices at the body radius (in km).
    By computing the median vertex distance, we can match to a known body.

    Returns:
        str or None: Body name if confidently detected, else None
    """
    vertices = mesh_data.get('vertices')
    if vertices is None or len(vertices) == 0:
        return None

    vertices = np.asarray(vertices)
    radii = np.sqrt(np.sum(vertices**2, axis=1))
    mesh_radius = float(np.median(radii))

    # Try matching as km (PTM3D convention)
    best_body = None
    best_diff = float('inf')
    for name, radius_m in BODY_RADII.items():
        diff = abs(radius_m / 1000.0 - mesh_radius)
        if diff < best_diff:
            best_diff = diff
            best_body = name

    # Accept if within 5% of known body radius
    expected_km = BODY_RADII[best_body] / 1000.0
    if best_diff / expected_km < 0.05:
        return best_body

    return None


def _forward_polar_stereo(lat, lon, pole_lat, center_lon, body_radius):
    """Forward polar stereographic projection for a sphere.

    Args:
        lat, lon: Arrays of geographic coordinates (degrees)
        pole_lat: +90 for north pole, -90 for south pole
        center_lon: Central meridian (degrees)
        body_radius: Sphere radius (meters)

    Returns:
        (x, y) projected coordinates in meters
    """
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
    """Inverse polar stereographic projection for a sphere.

    Args:
        x, y: Projected coordinates (meters)
        pole_lat: +90 for north pole, -90 for south pole
        center_lon: Central meridian (degrees)
        body_radius: Sphere radius (meters)

    Returns:
        (lat, lon) in degrees
    """
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
    """Choose projection based on latitude range of the data.

    Returns 'stere' if all data is within ~20 degrees of a pole,
    otherwise 'eqc'.
    """
    min_lat = float(np.nanmin(latitudes))
    max_lat = float(np.nanmax(latitudes))

    if min_lat > POLAR_LAT_THRESHOLD:
        return 'stere'
    elif max_lat < -POLAR_LAT_THRESHOLD:
        return 'stere'
    return 'eqc'


class GridResolutionEstimator:
    """Estimates optimal grid resolution from unstructured mesh face spacing."""

    @staticmethod
    def estimate_resolution(latitudes, longitudes):
        """Compute optimal grid resolution from face center spacing.

        Args:
            latitudes: Array of face center latitudes (degrees)
            longitudes: Array of face center longitudes (degrees)

        Returns:
            float: Estimated resolution in degrees, rounded to a nice value
        """
        if not SCIPY_AVAILABLE:
            # Fallback: rough estimate from coordinate range
            lat_range = np.ptp(latitudes)
            lon_range = np.ptp(longitudes)
            n_faces = len(latitudes)
            area = lat_range * lon_range
            raw = np.sqrt(area / n_faces)
            return GridResolutionEstimator._round_to_nice(raw)

        points = np.column_stack([longitudes, latitudes])
        tree = cKDTree(points)

        # Query k=2: self + nearest neighbor
        distances, _ = tree.query(points, k=2, workers=-1)
        # Column 1 is the nearest neighbor distance (column 0 is self = 0)
        nearest_distances = distances[:, 1]

        raw_resolution = float(np.median(nearest_distances))
        return GridResolutionEstimator._round_to_nice(raw_resolution)

    @staticmethod
    def _round_to_nice(value):
        """Round a value to the nearest 'nice' resolution."""
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


class MeshRegridder:
    """Regrids unstructured face data to a regular projected grid.

    Automatically selects equirectangular projection for mid-latitude data
    and polar stereographic for data within ~20 deg of either pole.
    Uses KDTree-based k=3 inverse-distance weighting.
    """

    def __init__(self, latitudes, longitudes, resolution, body_radius=1737400.0):
        """Build the regridding index.

        Args:
            latitudes: Face center latitudes (N_faces,)
            longitudes: Face center longitudes (N_faces,)
            resolution: Grid resolution in degrees
            body_radius: Planetary body radius in meters
        """
        if not SCIPY_AVAILABLE:
            raise RuntimeError("scipy is required for regridding")

        self.resolution = resolution
        self.body_radius = body_radius

        lon = np.array(longitudes, dtype=np.float64)
        lat = np.array(latitudes, dtype=np.float64)
        if np.ptp(lon) > 300:
            lon = lon % 360.0

        # Determine projection from latitude range
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
        """Initialize equirectangular grid."""
        resolution = self.resolution
        self.center_lon = float((np.nanmin(lon) + np.nanmax(lon)) / 2)

        lat_min = np.floor(np.nanmin(lat) / resolution) * resolution
        lat_max = np.ceil(np.nanmax(lat) / resolution) * resolution
        lon_min = np.floor(np.nanmin(lon) / resolution) * resolution
        lon_max = np.ceil(np.nanmax(lon) / resolution) * resolution

        self.lat_grid = np.arange(lat_min, lat_max + resolution * 0.5, resolution)
        self.lon_grid = np.arange(lon_min, lon_max + resolution * 0.5, resolution)
        self.n_lat = len(self.lat_grid)
        self.n_lon = len(self.lon_grid)

        # KDTree in degree space
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

        # Mask grid points outside the mesh boundary:
        # if nearest face center is > 1.5x the cell spacing, it's outside
        self._outside_mask = (
            self.distances[:, 0] > 1.5 * resolution
        ).reshape(self.n_lat, self.n_lon)

        # Stereographic attributes unused
        self.x_grid = None
        self.y_grid = None

    def _init_stereo(self, lat, lon):
        """Initialize polar stereographic grid."""
        center_lon = float((np.nanmin(lon) + np.nanmax(lon)) / 2)
        self.center_lon = center_lon

        # Metric resolution from degree resolution
        metric_res = self.body_radius * np.radians(self.resolution)

        # Convert face centers to stereographic
        valid_mask = np.isfinite(lon) & np.isfinite(lat)
        fx, fy = _forward_polar_stereo(
            lat[valid_mask], lon[valid_mask],
            self.pole_lat, center_lon, self.body_radius
        )

        # Grid bounds in projected space (snapped to resolution)
        x_min = np.floor(np.nanmin(fx) / metric_res) * metric_res
        x_max = np.ceil(np.nanmax(fx) / metric_res) * metric_res
        y_min = np.floor(np.nanmin(fy) / metric_res) * metric_res
        y_max = np.ceil(np.nanmax(fy) / metric_res) * metric_res

        self.x_grid = np.arange(x_min, x_max + metric_res * 0.5, metric_res)
        self.y_grid = np.arange(y_min, y_max + metric_res * 0.5, metric_res)
        self.n_lon = len(self.x_grid)   # width (columns)
        self.n_lat = len(self.y_grid)   # height (rows)

        # KDTree in projected space for correct nearest-neighbors at poles
        source_points = np.column_stack([fx, fy])
        tree = cKDTree(source_points)

        X, Y = np.meshgrid(self.x_grid, self.y_grid)
        grid_points = np.column_stack([X.ravel(), Y.ravel()])

        self.distances, indices = tree.query(grid_points, k=3, workers=-1)
        valid_indices = np.where(valid_mask)[0]
        self.neighbor_faces = valid_indices[indices]

        self.zero_mask = (self.distances == 0.0)
        self.has_any_zeros = np.any(self.zero_mask)

        # Mask grid points outside the mesh boundary:
        # if nearest face center is > 1.5x the cell spacing, it's outside
        self._outside_mask = (
            self.distances[:, 0] > 1.5 * metric_res
        ).reshape(self.n_lat, self.n_lon)

        # 2D lat/lon grids for HDF5 coordinate storage
        grid_lats, grid_lons = _inverse_polar_stereo(
            X.ravel(), Y.ravel(),
            self.pole_lat, center_lon, self.body_radius
        )
        self.lat_grid = grid_lats.reshape((self.n_lat, self.n_lon))
        self.lon_grid = grid_lons.reshape((self.n_lat, self.n_lon))

    def regrid(self, face_values):
        """Interpolate face values to the regular grid.

        Args:
            face_values: 1D array of per-face values (N_faces,)

        Returns:
            2D array of shape (n_lat, n_lon) with NaN for missing data
        """
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

    def get_grid_shape(self):
        """Return (n_lat, n_lon) tuple."""
        return (self.n_lat, self.n_lon)


class GeoTIFFExporter:
    """Exports regridded data as GeoTIFF files."""

    @staticmethod
    def _make_profile(n_bands, regridder):
        """Build a rasterio profile from regridder grid info.

        Dispatches to equirectangular or polar stereographic based on
        the regridder's projection attribute.
        """
        if regridder.projection == 'stere':
            return GeoTIFFExporter._make_stereo_profile(n_bands, regridder)
        else:
            return GeoTIFFExporter._make_eqc_profile(n_bands, regridder)

    @staticmethod
    def _make_eqc_profile(n_bands, regridder):
        """Build profile for equirectangular projection."""
        lat_array = regridder.lat_grid
        lon_array = regridder.lon_grid
        body_radius = regridder.body_radius
        n_lat = regridder.n_lat
        n_lon = regridder.n_lon

        dlon = float(lon_array[1] - lon_array[0])
        dlat = float(lat_array[1] - lat_array[0])

        west_deg = float(lon_array[0]) - dlon / 2
        east_deg = float(lon_array[-1]) + dlon / 2
        south_deg = float(lat_array[0]) - dlat / 2
        north_deg = float(lat_array[-1]) + dlat / 2

        center_lon = (west_deg + east_deg) / 2
        center_lat = (south_deg + north_deg) / 2

        deg2rad = np.pi / 180.0
        cos_ts = np.cos(center_lat * deg2rad)

        west_m = body_radius * cos_ts * (west_deg - center_lon) * deg2rad
        east_m = body_radius * cos_ts * (east_deg - center_lon) * deg2rad
        south_m = body_radius * south_deg * deg2rad
        north_m = body_radius * north_deg * deg2rad

        transform = from_bounds(west_m, south_m, east_m, north_m, n_lon, n_lat)

        # Use WKT to avoid PROJ database lookup (conda env conflicts)
        wkt = (
            'PROJCS["Equirectangular",'
            'GEOGCS["GCS_Sphere",'
            f'DATUM["D_Sphere",SPHEROID["Sphere",{body_radius},0]],'
            'PRIMEM["Greenwich",0],'
            'UNIT["degree",0.0174532925199433]],'
            'PROJECTION["Equidistant_Cylindrical"],'
            f'PARAMETER["standard_parallel_1",{center_lat}],'
            f'PARAMETER["central_meridian",{center_lon}],'
            'PARAMETER["false_easting",0],'
            'PARAMETER["false_northing",0],'
            'UNIT["metre",1]]'
        )
        crs = CRS.from_wkt(wkt)

        return {
            'driver': 'GTiff',
            'dtype': 'float32',
            'width': n_lon,
            'height': n_lat,
            'count': n_bands,
            'crs': crs,
            'transform': transform,
            'nodata': np.nan,
            'compress': 'lzw',
            'predictor': 2,
            'tiled': True,
            'blockxsize': 256,
            'blockysize': 256,
        }

    @staticmethod
    def _make_stereo_profile(n_bands, regridder):
        """Build profile for polar stereographic projection."""
        x_grid = regridder.x_grid
        y_grid = regridder.y_grid
        body_radius = regridder.body_radius

        dx = float(x_grid[1] - x_grid[0])
        dy = float(y_grid[1] - y_grid[0])

        west = float(x_grid[0]) - dx / 2
        east = float(x_grid[-1]) + dx / 2
        south = float(y_grid[0]) - dy / 2
        north = float(y_grid[-1]) + dy / 2

        transform = from_bounds(
            west, south, east, north, regridder.n_lon, regridder.n_lat
        )

        # Use WKT to avoid PROJ database lookup (conda env conflicts)
        wkt = (
            'PROJCS["Polar_Stereographic",'
            'GEOGCS["GCS_Sphere",'
            f'DATUM["D_Sphere",SPHEROID["Sphere",{body_radius},0]],'
            'PRIMEM["Greenwich",0],'
            'UNIT["degree",0.0174532925199433]],'
            'PROJECTION["Polar_Stereographic"],'
            f'PARAMETER["latitude_of_origin",{regridder.pole_lat}],'
            f'PARAMETER["central_meridian",{regridder.center_lon}],'
            'PARAMETER["scale_factor",1],'
            'PARAMETER["false_easting",0],'
            'PARAMETER["false_northing",0],'
            'UNIT["metre",1]]'
        )
        crs = CRS.from_wkt(wkt)

        return {
            'driver': 'GTiff',
            'dtype': 'float32',
            'width': regridder.n_lon,
            'height': regridder.n_lat,
            'count': n_bands,
            'crs': crs,
            'transform': transform,
            'nodata': np.nan,
            'compress': 'lzw',
            'predictor': 2,
            'tiled': True,
            'blockxsize': 256,
            'blockysize': 256,
        }

    @staticmethod
    def _write_tags(dst, field_name):
        """Write metadata tags to an open rasterio dataset."""
        info = FIELD_INFO.get(field_name, {})
        if not info:
            # Statistics field with avg/max/min suffix (e.g. 'temperature_avg').
            parts = field_name.rsplit('_', 1)
            if len(parts) == 2 and parts[1] in STAT_LABELS:
                base_info = STATISTICS_FIELDS.get(parts[0], {})
                if base_info:
                    info = {
                        'label': f"{base_info['label']} ({STAT_LABELS[parts[1]]})",
                        'units': base_info.get('units', ''),
                    }
        if not info:
            # Scalar statistics field (e.g. 'ice_depth' — no _avg/_max/_min).
            base_info = STATISTICS_FIELDS.get(field_name, {})
            if base_info:
                info = {
                    'label': base_info.get('label', field_name),
                    'units': base_info.get('units', ''),
                }
        dst.update_tags(
            field=field_name,
            units=info.get('units', ''),
            description=info.get('label', field_name),
            created=datetime.now().isoformat(),
            software='PTM3D',
        )

    @staticmethod
    def export_field(output_path, data_3d, regridder, field_name,
                     time_labels=None):
        """Write a multi-band GeoTIFF file for one field.

        Args:
            output_path: Output file path
            data_3d: Array of shape (N_timesteps, N_lat, N_lon)
            regridder: MeshRegridder instance (provides CRS and grid info)
            field_name: Field name for metadata
            time_labels: Optional list of UTC time label strings
        """
        if not RASTERIO_AVAILABLE:
            raise RuntimeError("rasterio is required for GeoTIFF export")

        n_bands, n_lat, n_lon = data_3d.shape
        data_flipped = data_3d[:, ::-1, :]

        profile = GeoTIFFExporter._make_profile(n_bands, regridder)

        with rasterio.open(str(output_path), 'w', **profile) as dst:
            for band_idx in range(n_bands):
                dst.write(data_flipped[band_idx], band_idx + 1)
                if time_labels and band_idx < len(time_labels):
                    dst.set_band_description(band_idx + 1, time_labels[band_idx])
            GeoTIFFExporter._write_tags(dst, field_name)

    @staticmethod
    def export_single_timestep(output_path, data_2d, regridder, field_name,
                               time_label=None):
        """Write a single-band GeoTIFF file for one timestep.

        Args:
            output_path: Output file path
            data_2d: Array of shape (N_lat, N_lon)
            regridder: MeshRegridder instance (provides CRS and grid info)
            field_name: Field name for metadata
            time_label: Optional UTC time label string
        """
        if not RASTERIO_AVAILABLE:
            raise RuntimeError("rasterio is required for GeoTIFF export")

        n_lat, n_lon = data_2d.shape
        data_flipped = data_2d[::-1, :]

        profile = GeoTIFFExporter._make_profile(1, regridder)

        with rasterio.open(str(output_path), 'w', **profile) as dst:
            dst.write(data_flipped, 1)
            if time_label:
                dst.set_band_description(1, time_label)
            GeoTIFFExporter._write_tags(dst, field_name)

    @staticmethod
    def export_statistic(output_path, data_2d, regridder, base_field, stat_type):
        """Write a single-band GeoTIFF for one statistic field.

        Delegates to export_single_timestep for file creation, then uses
        the same tag format as time-series exports for maximum compatibility.

        Args:
            output_path: Output file path
            data_2d: Array of shape (N_lat, N_lon)
            regridder: MeshRegridder instance
            base_field: Base field name (e.g. 'temperature')
            stat_type: Statistic type ('avg', 'max', 'min', or None for
                scalar single-band fields like ice_depth).
        """
        info = STATISTICS_FIELDS.get(base_field, {})
        if stat_type is None:
            field_name = base_field
            units = info.get('units', '')
            label = info.get('label', base_field)
            band_label = f"{label} ({units})" if units else label
        else:
            stat_label = STAT_LABELS.get(stat_type, stat_type)
            field_name = f"{base_field}_{stat_type}"
            band_label = f"{info.get('label', base_field)} ({stat_label})"
        GeoTIFFExporter.export_single_timestep(
            output_path, data_2d, regridder, field_name, band_label
        )


class HDF5Exporter:
    """Exports simulation data to a single HDF5 file."""

    @staticmethod
    def export(output_path, mesh_data, timesteps, field_data,
               regridded_data=None, regridder=None,
               statistics_data=None, regridded_statistics=None):
        """Write HDF5 file with mesh, time, unstructured, and regridded data.

        Args:
            output_path: Output file path
            mesh_data: Dict with 'vertices', 'faces', 'latitudes', 'longitudes'
            timesteps: List of dicts with 'time' and 'label' keys
            field_data: Dict of {field_name: (N_timesteps, N_faces) array}
            regridded_data: Optional dict of {field_name: (N_timesteps, N_lat, N_lon) array}
            regridder: Optional MeshRegridder instance (for grid coordinates)
            statistics_data: Optional dict of {(base_field, stat_type): 1D face array}
            regridded_statistics: Optional dict of {(base_field, stat_type): 2D grid array}
        """
        if not H5PY_AVAILABLE:
            raise RuntimeError("h5py is required for HDF5 export")

        with h5py.File(str(output_path), 'w') as f:
            # Root attributes
            f.attrs['created'] = datetime.now().isoformat()
            f.attrs['software'] = 'PTM3D'
            f.attrs['format_version'] = '1.0'

            # Mesh group
            mesh_grp = f.create_group('mesh')
            mesh_grp.create_dataset(
                'latitudes',
                data=mesh_data['latitudes'].astype(np.float32),
                compression='gzip', compression_opts=4
            )
            mesh_grp.create_dataset(
                'longitudes',
                data=mesh_data['longitudes'].astype(np.float32),
                compression='gzip', compression_opts=4
            )
            mesh_grp.create_dataset(
                'vertices',
                data=mesh_data['vertices'].astype(np.float64),
                compression='gzip', compression_opts=4
            )
            mesh_grp.create_dataset(
                'faces',
                data=mesh_data['faces'].astype(np.int32),
                compression='gzip', compression_opts=4
            )
            mesh_grp.attrs['n_faces'] = len(mesh_data['latitudes'])
            mesh_grp.attrs['n_vertices'] = len(mesh_data['vertices'])

            # Time group
            time_grp = f.create_group('time')
            elapsed = np.array([ts['time'] for ts in timesteps], dtype=np.float64)
            time_grp.create_dataset('elapsed_seconds', data=elapsed)
            time_grp['elapsed_seconds'].attrs['units'] = 'seconds'

            labels = [ts['label'] for ts in timesteps]
            dt = h5py.string_dtype()
            time_grp.create_dataset('utc_labels', data=labels, dtype=dt)
            time_grp.attrs['n_timesteps'] = len(timesteps)

            # Unstructured data group
            data_grp = f.create_group('data')
            for field_name, data_array in field_data.items():
                info = FIELD_INFO.get(field_name, {})
                n_faces = data_array.shape[1]
                ds = data_grp.create_dataset(
                    field_name,
                    data=data_array.astype(np.float32),
                    chunks=(1, n_faces),
                    compression='gzip', compression_opts=4
                )
                ds.attrs['units'] = info.get('units', '')
                ds.attrs['description'] = info.get('label', field_name)

            # Regridded data group
            if regridded_data and regridder:
                reg_grp = f.create_group('regridded')

                # Coordinate arrays (1D for eqc, 2D for stereo)
                reg_grp.create_dataset('latitudes', data=regridder.lat_grid)
                reg_grp['latitudes'].attrs['units'] = 'degrees_north'
                reg_grp.create_dataset('longitudes', data=regridder.lon_grid)
                reg_grp['longitudes'].attrs['units'] = 'degrees_east'

                if regridder.projection == 'stere':
                    reg_grp.create_dataset('x', data=regridder.x_grid)
                    reg_grp['x'].attrs['units'] = 'meters'
                    reg_grp.create_dataset('y', data=regridder.y_grid)
                    reg_grp['y'].attrs['units'] = 'meters'

                reg_grp.attrs['projection'] = regridder.projection
                reg_grp.attrs['resolution_degrees'] = regridder.resolution
                reg_grp.attrs['body_radius_m'] = regridder.body_radius
                reg_grp.attrs['n_lat'] = regridder.n_lat
                reg_grp.attrs['n_lon'] = regridder.n_lon

                # Projection string
                if regridder.projection == 'stere':
                    reg_grp.attrs['proj4'] = (
                        f'+proj=stere +lat_0={regridder.pole_lat} '
                        f'+lat_ts={regridder.pole_lat} '
                        f'+lon_0={regridder.center_lon} '
                        f'+R={regridder.body_radius} +units=m +no_defs'
                    )
                else:
                    center_lat = float(
                        (regridder.lat_grid[0] + regridder.lat_grid[-1]) / 2
                    )
                    reg_grp.attrs['proj4'] = (
                        f'+proj=eqc +lat_ts={center_lat} +lat_0=0 '
                        f'+lon_0={regridder.center_lon} '
                        f'+R={regridder.body_radius} +units=m +no_defs'
                    )

                for field_name, data_array in regridded_data.items():
                    info = FIELD_INFO.get(field_name, {})
                    ds = reg_grp.create_dataset(
                        field_name,
                        data=data_array.astype(np.float32),
                        chunks=(1, regridder.n_lat, regridder.n_lon),
                        compression='gzip', compression_opts=4
                    )
                    ds.attrs['units'] = info.get('units', '')
                    ds.attrs['description'] = info.get('label', field_name)

            # Statistics group (per-face, no time dimension)
            if statistics_data:
                stats_grp = f.create_group('statistics')
                stats_grp.attrs['description'] = (
                    'Per-face statistics computed over integration window'
                )
                for (base_field, stat_type), face_data in statistics_data.items():
                    info = STATISTICS_FIELDS.get(base_field, {})
                    if stat_type is None:
                        # Scalar field (e.g. ice_depth): no _<stat> suffix.
                        key = base_field
                        description = info.get('label', base_field)
                        stat_attr = 'value'
                    else:
                        key = f"{base_field}_{stat_type}"
                        stat_label = STAT_LABELS.get(stat_type, stat_type)
                        description = f"{info.get('label', base_field)} ({stat_label})"
                        stat_attr = stat_type
                    ds = stats_grp.create_dataset(
                        key,
                        data=face_data.astype(np.float32),
                        compression='gzip', compression_opts=4
                    )
                    ds.attrs['units'] = info.get('units', '')
                    ds.attrs['description'] = description
                    ds.attrs['statistic'] = stat_attr
                    ds.attrs['base_field'] = base_field

            # Regridded statistics group
            if regridded_statistics and regridder:
                reg_stats = f.create_group('regridded_statistics')
                reg_stats.attrs['projection'] = regridder.projection
                reg_stats.attrs['resolution_degrees'] = regridder.resolution
                for (base_field, stat_type), grid_data in regridded_statistics.items():
                    info = STATISTICS_FIELDS.get(base_field, {})
                    if stat_type is None:
                        key = base_field
                        description = info.get('label', base_field)
                        stat_attr = 'value'
                    else:
                        key = f"{base_field}_{stat_type}"
                        stat_label = STAT_LABELS.get(stat_type, stat_type)
                        description = f"{info.get('label', base_field)} ({stat_label})"
                        stat_attr = stat_type
                    ds = reg_stats.create_dataset(
                        key,
                        data=grid_data.astype(np.float32),
                        compression='gzip', compression_opts=4
                    )
                    ds.attrs['units'] = info.get('units', '')
                    ds.attrs['description'] = description
                    ds.attrs['statistic'] = stat_attr
                    ds.attrs['base_field'] = base_field


def write_export_readme(output_dir, regridder, fields, timesteps,
                        formats, per_timestep=False, statistics=None):
    """Write a README.txt describing the exported data and projection.

    Args:
        output_dir: Path to export directory
        regridder: MeshRegridder instance (or None if regridding was skipped)
        fields: List of exported field names
        timesteps: List of timestep dicts with 'time' and 'label' keys
        formats: List of format strings ('geotiff', 'hdf5')
        per_timestep: Whether GeoTIFFs were written per-timestep
        statistics: Optional list of (base_field, stat_type) tuples
    """
    output_dir = Path(output_dir)
    lines = []

    lines.append("PTM3D Data Export")
    lines.append("=================")
    lines.append("")
    lines.append(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("")

    # Body and projection
    lines.append("Projection")
    lines.append("----------")

    if regridder is None:
        lines.append("No regridding was performed (scipy unavailable).")
    elif regridder.projection == 'stere':
        pole_str = "North" if regridder.pole_lat > 0 else "South"
        lines.append(f"Polar stereographic projection ({pole_str} pole).")
        lines.append("")
        proj4 = (
            f"+proj=stere +lat_0={regridder.pole_lat} "
            f"+lat_ts={regridder.pole_lat} "
            f"+lon_0={regridder.center_lon} "
            f"+R={regridder.body_radius} +units=m +no_defs"
        )
        lines.append(f"PROJ string:")
        lines.append(f"\t{proj4}")
        lines.append("")
        lines.append(f"Body radius: {regridder.body_radius:.1f} m "
                      f"({regridder.body_radius / 1000:.1f} km)")
        lines.append(f"Grid resolution: {regridder.resolution} degrees "
                      f"(~{regridder.body_radius * np.radians(regridder.resolution):.1f} m/pixel)")
        lines.append(f"Grid size: {regridder.n_lat} x {regridder.n_lon} pixels")
        lines.append(f"Central meridian: {regridder.center_lon:.4f} degrees")
        lines.append("")
        lines.append("Pixel values are in projected meters (easting, northing).")
        lines.append("Latitude and longitude for each pixel can be recovered via")
        lines.append("the inverse stereographic transform or from the PROJ string above.")
    else:
        center_lat = float(
            (regridder.lat_grid[0] + regridder.lat_grid[-1]) / 2
        )
        proj4 = (
            f"+proj=eqc +lat_ts={center_lat} +lat_0=0 "
            f"+lon_0={regridder.center_lon} "
            f"+R={regridder.body_radius} +units=m +no_defs"
        )
        lines.append("Equirectangular (simple cylindrical) projection.")
        lines.append("")
        lines.append(f"PROJ string:")
        lines.append(f"\t{proj4}")
        lines.append("")
        lines.append(f"Body radius: {regridder.body_radius:.1f} m "
                      f"({regridder.body_radius / 1000:.1f} km)")
        lines.append(f"Grid resolution: {regridder.resolution} degrees")
        lat_pixel = regridder.body_radius * np.radians(regridder.resolution)
        lon_pixel = lat_pixel * np.cos(np.radians(center_lat))
        lines.append(f"Pixel size at center latitude ({center_lat:.2f} deg): "
                      f"~{lat_pixel:.1f} x {lon_pixel:.1f} m")
        lines.append(f"Grid size: {regridder.n_lat} x {regridder.n_lon} pixels")
        lines.append(f"Latitude range: {float(regridder.lat_grid[0]):.4f} to "
                      f"{float(regridder.lat_grid[-1]):.4f} degrees")
        lines.append(f"Longitude range: {float(regridder.lon_grid[0]):.4f} to "
                      f"{float(regridder.lon_grid[-1]):.4f} degrees")
        lines.append(f"True-scale latitude: {center_lat:.4f} degrees")
        lines.append(f"Central meridian: {regridder.center_lon:.4f} degrees")

    lines.append("")
    lines.append("")

    # Fields
    lines.append("Data Fields")
    lines.append("-----------")
    for field_name in fields:
        info = FIELD_INFO.get(field_name, {})
        label = info.get('label', field_name)
        units = info.get('units', '')
        lines.append(f"* {label} ({field_name})")
        if units:
            lines.append(f"  Units: {units}")
    lines.append("")
    lines.append(f"Timesteps: {len(timesteps)}")
    if len(timesteps) > 0:
        lines.append(f"First: {timesteps[0]['label']}")
        lines.append(f"Last:  {timesteps[-1]['label']}")

    lines.append("")
    lines.append("")

    # Products
    lines.append("Products")
    lines.append("--------")

    if 'geotiff' in formats:
        if per_timestep:
            lines.append("* {field_name}/{field_name}_NNNN.tif")
            lines.append("")
            lines.append("  Per-timestep GeoTIFF files. Each file is a single-band")
            lines.append("  Float32 raster. NoData value is NaN. Compression: deflate.")
        else:
            lines.append("* {field_name}.tif")
            lines.append("")
            lines.append("  Multi-band GeoTIFF, one band per timestep. Float32.")
            lines.append("  NoData value is NaN. Compression: deflate.")
            lines.append("  Band descriptions contain UTC time labels.")
        lines.append("")

    if 'hdf5' in formats:
        lines.append("* ptm3d_data.h5")
        lines.append("")
        lines.append("  HDF5 file with hierarchical structure:")
        lines.append("    /mesh/          - Original unstructured mesh (vertices, faces, lat/lon)")
        lines.append("    /time/          - Elapsed seconds and UTC labels")
        lines.append("    /data/{field}   - Unstructured face data (N_timesteps, N_faces)")
        lines.append("    /regridded/     - Regridded data on regular grid")
        lines.append("    /regridded/{field} - (N_timesteps, N_lat, N_lon)")
        if statistics:
            lines.append("    /statistics/    - Per-face statistics (avg/max/min)")
            lines.append("    /regridded_statistics/ - Regridded statistics")
        lines.append("  Compression: gzip level 4.")
        lines.append("")

    if statistics:
        if 'geotiff' in formats:
            lines.append("* statistics/{field}_{stat}.tif")
            lines.append("")
            lines.append("  Single-band GeoTIFF files for per-face statistics.")
            lines.append("  Float32, NoData = NaN, compression: deflate.")
            lines.append("")

        lines.append("")
        lines.append("Statistics")
        lines.append("----------")
        lines.append("Per-face statistics computed over the integration window:")
        lines.append("")
        for base_field, stat_type in statistics:
            info = STATISTICS_FIELDS.get(base_field, {})
            label = info.get('label', base_field)
            units = info.get('units', '')
            units_str = f" ({units})" if units else ""
            if stat_type is None:
                lines.append(f"* {label}{units_str}")
            else:
                stat_label = STAT_LABELS.get(stat_type, stat_type)
                lines.append(f"* {label} - {stat_label}{units_str}")
        lines.append("")

    lines.append("")
    lines.append("Software")
    lines.append("--------")
    lines.append("These data products were created by PTM3D (Planetary Thermal Model 3D).")
    lines.append("")

    readme_path = output_dir / "README.txt"
    readme_path.write_text("\n".join(lines), encoding='utf-8')
    return readme_path


def read_binary_data(file_path, expected_count):
    """Read a PTM3D binary data file.

    Args:
        file_path: Path to .bin file
        expected_count: Expected number of face values

    Returns:
        numpy float32 array or None if file can't be read
    """
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


def detect_available_statistics(results_dir):
    """Detect which statistics files are available and contain valid data.

    Args:
        results_dir: Path to the results directory

    Returns:
        List of (base_field, stat_type) tuples for files with non-NaN data.
    """
    stats_dir = Path(results_dir) / 'statistics'
    if not stats_dir.is_dir():
        return []

    def _file_has_data(filepath):
        """True iff the binary stats file is well-formed and not all-NaN."""
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
            # Single per-face value, file is <base>.bin (no _avg/_max/_min).
            # Selection key uses stat_type=None to flag the scalar path.
            if _file_has_data(stats_dir / f"{base_field}.bin"):
                available.append((base_field, None))
            continue
        for stat_type in STAT_TYPES:
            if _file_has_data(stats_dir / f"{base_field}_{stat_type}.bin"):
                available.append((base_field, stat_type))
    return available


class DataExportWorker(QThread):
    """Background worker for exporting simulation data."""

    progress_updated = pyqtSignal(int, int)    # current, total
    status_updated = pyqtSignal(str)           # status message
    export_finished = pyqtSignal(bool, str)    # success, message

    def __init__(self, config, parent=None):
        """Initialize export worker.

        Args:
            config: Dict with keys:
                - results_dir: Path to results directory
                - mesh_data: Dict with vertices, faces, latitudes, longitudes
                - timesteps: List of {time, label} dicts
                - fields: List of field names to export
                - formats: List of format strings ('geotiff', 'hdf5')
                - resolution: Grid resolution in degrees
                - output_dir: Output directory path
                - geotiff_per_timestep: bool, if True write one file per timestep
                - body_radius: Planetary body radius in meters
        """
        super().__init__(parent)
        self.config = config
        self._cancelled = False

    def cancel(self):
        """Request cancellation."""
        self._cancelled = True

    def run(self):
        """Execute the export."""
        try:
            self._do_export()
        except Exception as e:
            self.export_finished.emit(False, f"Export failed: {str(e)}")

    def _do_export(self):
        results_dir = Path(self.config['results_dir'])
        mesh_data = self.config['mesh_data']
        timesteps = self.config['timesteps']
        fields = self.config['fields']
        formats = self.config['formats']
        resolution = self.config['resolution']
        output_dir = Path(self.config['output_dir'])
        per_timestep = self.config.get('geotiff_per_timestep', False)
        body_radius = self.config.get('body_radius', 1737400.0)
        statistics = self.config.get('statistics', [])

        output_dir.mkdir(parents=True, exist_ok=True)

        n_faces = len(mesh_data['latitudes'])
        n_timesteps = len(timesteps)
        n_fields = len(fields)
        n_statistics = len(statistics)

        do_geotiff = 'geotiff' in formats and RASTERIO_AVAILABLE
        do_hdf5 = 'hdf5' in formats and H5PY_AVAILABLE
        needs_regrid = (do_geotiff or do_hdf5) and SCIPY_AVAILABLE

        # For per-timestep GeoTIFF, we don't need to accumulate grid data
        # unless HDF5 also needs it
        keep_grid_data = do_hdf5

        # Calculate total steps for progress
        total_steps = 1 + (n_fields * n_timesteps) + n_fields + n_statistics
        current_step = 0

        # Build regridder (not needed for VTU-only export)
        regridder = None
        if needs_regrid:
            self.status_updated.emit("Building spatial index for regridding...")
            regridder = MeshRegridder(
                mesh_data['latitudes'],
                mesh_data['longitudes'],
                resolution,
                body_radius=body_radius
            )
        current_step += 1
        self.progress_updated.emit(current_step, total_steps)

        if self._cancelled:
            self._cleanup(output_dir)
            self.export_finished.emit(False, "Export cancelled")
            return

        # Read and regrid all data, one field at a time
        all_face_data = {}
        all_grid_data = {}
        output_files = []

        for field_name in fields:
            if self._cancelled:
                self._cleanup(output_dir)
                self.export_finished.emit(False, "Export cancelled")
                return

            self.status_updated.emit(f"Processing {field_name}...")
            field_dir = results_dir / field_name

            face_array = np.full((n_timesteps, n_faces), np.nan, dtype=np.float32)
            grid_array = None
            if regridder and keep_grid_data:
                grid_array = np.full(
                    (n_timesteps, regridder.n_lat, regridder.n_lon),
                    np.nan, dtype=np.float32
                )

            # Create per-timestep output directory if needed
            if do_geotiff and per_timestep and regridder:
                tiff_dir = output_dir / field_name
                tiff_dir.mkdir(parents=True, exist_ok=True)

            for t in range(n_timesteps):
                if self._cancelled:
                    self._cleanup(output_dir)
                    self.export_finished.emit(False, "Export cancelled")
                    return

                file_path = field_dir / f"{field_name}_{t:04d}.bin"
                if file_path.exists():
                    data = read_binary_data(str(file_path), n_faces)
                    if data is not None:
                        face_array[t] = data
                        if regridder:
                            grid_2d = regridder.regrid(data)
                            if keep_grid_data:
                                grid_array[t] = grid_2d
                            # Write per-timestep GeoTIFF immediately
                            if do_geotiff and per_timestep:
                                tiff_path = tiff_dir / f"{field_name}_{t:04d}.tif"
                                time_label = timesteps[t]['label']
                                GeoTIFFExporter.export_single_timestep(
                                    tiff_path, grid_2d, regridder,
                                    field_name, time_label
                                )

                current_step += 1
                self.progress_updated.emit(current_step, total_steps)

            all_face_data[field_name] = face_array
            if grid_array is not None:
                all_grid_data[field_name] = grid_array

            # Write multi-band GeoTIFF (only if not per-timestep mode)
            if do_geotiff and not per_timestep and regridder:
                self.status_updated.emit(f"Writing GeoTIFF for {field_name}...")
                # Need to regrid all if we didn't keep grid data
                if grid_array is None:
                    grid_array = np.full(
                        (n_timesteps, regridder.n_lat, regridder.n_lon),
                        np.nan, dtype=np.float32
                    )
                    for t in range(n_timesteps):
                        file_path = field_dir / f"{field_name}_{t:04d}.bin"
                        if file_path.exists():
                            data = read_binary_data(str(file_path), n_faces)
                            if data is not None:
                                grid_array[t] = regridder.regrid(data)
                tiff_path = output_dir / f"{field_name}.tif"
                time_labels = [ts['label'] for ts in timesteps]
                GeoTIFFExporter.export_field(
                    tiff_path, grid_array, regridder,
                    field_name, time_labels
                )
                output_files.append(str(tiff_path))

            if do_geotiff and per_timestep and regridder:
                tiff_dir = output_dir / field_name
                output_files.append(f"{tiff_dir}/ ({n_timesteps} files)")

            current_step += 1
            self.progress_updated.emit(current_step, total_steps)

        if self._cancelled:
            self._cleanup(output_dir)
            self.export_finished.emit(False, "Export cancelled")
            return

        # Export statistics (single-band per-face data)
        all_stat_face_data = {}
        all_stat_grid_data = {}

        if statistics:
            stats_dir = results_dir / 'statistics'

            for base_field, stat_type in statistics:
                if self._cancelled:
                    self._cleanup(output_dir)
                    self.export_finished.emit(False, "Export cancelled")
                    return

                # Scalar fields (e.g. ice_depth) carry stat_type=None and
                # live on disk as <base>.bin / <base>.tif. Non-scalar fields
                # use the stat_type-suffixed filenames.
                is_scalar = stat_type is None
                if is_scalar:
                    field_label = STATISTICS_FIELDS.get(
                        base_field, {}
                    ).get('label', base_field)
                    self.status_updated.emit(
                        f"Processing statistic: {field_label}..."
                    )
                    filepath = stats_dir / f"{base_field}.bin"
                    tiff_name = f"{base_field}.tif"
                else:
                    stat_label = STAT_LABELS.get(stat_type, stat_type)
                    self.status_updated.emit(
                        f"Processing statistic: {base_field} {stat_label}..."
                    )
                    filepath = stats_dir / f"{base_field}_{stat_type}.bin"
                    tiff_name = f"{base_field}_{stat_type}.tif"

                face_data = read_binary_data(str(filepath), n_faces)

                if face_data is not None:
                    stat_key = (base_field, stat_type)
                    all_stat_face_data[stat_key] = face_data

                    if regridder:
                        grid_2d = regridder.regrid(face_data)

                        # Skip all-NaN grids (no valid data after regridding)
                        if np.all(np.isnan(grid_2d)):
                            tag = (
                                field_label if is_scalar
                                else f"{base_field} {stat_label}"
                            )
                            self.status_updated.emit(
                                f"Skipping {tag}: all NaN after regridding"
                            )
                        else:
                            all_stat_grid_data[stat_key] = grid_2d

                            if do_geotiff:
                                stat_tiff_dir = output_dir / 'statistics'
                                stat_tiff_dir.mkdir(parents=True, exist_ok=True)
                                tiff_path = stat_tiff_dir / tiff_name
                                GeoTIFFExporter.export_statistic(
                                    tiff_path, grid_2d, regridder,
                                    base_field, stat_type
                                )

                current_step += 1
                self.progress_updated.emit(current_step, total_steps)

            if do_geotiff and all_stat_grid_data and regridder:
                stat_tiff_dir = output_dir / 'statistics'
                output_files.append(
                    f"{stat_tiff_dir}/ ({len(all_stat_grid_data)} files)"
                )

        # Write HDF5 (single file with all fields and statistics)
        if 'hdf5' in formats and H5PY_AVAILABLE:
            self.status_updated.emit("Writing HDF5 file...")
            h5_path = output_dir / "ptm3d_data.h5"
            HDF5Exporter.export(
                h5_path, mesh_data, timesteps,
                all_face_data,
                regridded_data=all_grid_data if regridder else None,
                regridder=regridder,
                statistics_data=(
                    all_stat_face_data if all_stat_face_data else None
                ),
                regridded_statistics=(
                    all_stat_grid_data if all_stat_grid_data else None
                ),
            )
            output_files.append(str(h5_path))

        # Write VTU/PVD (unstructured mesh, one VTU per timestep + PVD collection)
        if 'vtu' in formats and VTK_AVAILABLE:
            self.status_updated.emit("Writing VTU/PVD files...")
            vtu_dir = output_dir / "vtu"
            vtu_dir.mkdir(parents=True, exist_ok=True)

            # Build VTK unstructured grid from mesh
            import vtk
            from vtk.util import numpy_support

            vertices = mesh_data['vertices']
            faces = mesh_data['faces']
            n_verts = len(vertices)

            points = vtk.vtkPoints()
            points.SetNumberOfPoints(n_verts)
            for i in range(n_verts):
                points.SetPoint(i, float(vertices[i, 0]),
                                float(vertices[i, 1]),
                                float(vertices[i, 2]))

            cells = vtk.vtkCellArray()
            for f in faces:
                cells.InsertNextCell(3, [int(f[0]), int(f[1]), int(f[2])])

            # Write one VTU per timestep with all selected fields
            vtu_files = []
            for t in range(n_timesteps):
                if self._cancelled:
                    self._cleanup(output_dir)
                    self.export_finished.emit(False, "Export cancelled")
                    return

                grid = vtk.vtkUnstructuredGrid()
                grid.SetPoints(points)
                grid.SetCells(vtk.VTK_TRIANGLE, cells)

                # Add lat/lon as cell data
                for coord_name in ('latitudes', 'longitudes'):
                    if coord_name in mesh_data:
                        arr = numpy_support.numpy_to_vtk(
                            mesh_data[coord_name].astype(np.float64),
                            deep=True
                        )
                        arr.SetName(coord_name)
                        grid.GetCellData().AddArray(arr)

                # Add field data for this timestep
                for field_name in fields:
                    if field_name in all_face_data:
                        face_vals = all_face_data[field_name][t]
                        vtk_arr = numpy_support.numpy_to_vtk(
                            face_vals.astype(np.float64), deep=True
                        )
                        vtk_arr.SetName(field_name)
                        grid.GetCellData().AddArray(vtk_arr)

                vtu_path = vtu_dir / f"ptm3d_{t:04d}.vtu"
                writer = vtk.vtkXMLUnstructuredGridWriter()
                writer.SetFileName(str(vtu_path))
                writer.SetInputData(grid)
                writer.SetDataModeToAppended()
                writer.SetCompressorTypeToZLib()
                writer.Write()
                vtu_files.append(vtu_path)

                self.status_updated.emit(
                    f"Writing VTU timestep {t + 1}/{n_timesteps}..."
                )

            # Write PVD collection file
            pvd_path = output_dir / "ptm3d_timeseries.pvd"
            pvd_lines = [
                '<?xml version="1.0"?>',
                '<VTKFile type="Collection" version="0.1">',
                '  <Collection>',
            ]
            for t, vtu_path in enumerate(vtu_files):
                time_val = float(t)
                if t < len(timesteps) and 'time' in timesteps[t]:
                    time_val = float(timesteps[t]['time'])
                rel_path = f"vtu/{vtu_path.name}"
                pvd_lines.append(
                    f'    <DataSet timestep="{time_val}" '
                    f'file="{rel_path}"/>'
                )
            pvd_lines.append('  </Collection>')
            pvd_lines.append('</VTKFile>')
            pvd_path.write_text('\n'.join(pvd_lines))
            output_files.append(str(pvd_path))

            # Write statistics VTU files if available
            if all_stat_face_data:
                stat_grid = vtk.vtkUnstructuredGrid()
                stat_grid.SetPoints(points)
                stat_grid.SetCells(vtk.VTK_TRIANGLE, cells)

                for coord_name in ('latitudes', 'longitudes'):
                    if coord_name in mesh_data:
                        arr = numpy_support.numpy_to_vtk(
                            mesh_data[coord_name].astype(np.float64),
                            deep=True
                        )
                        arr.SetName(coord_name)
                        stat_grid.GetCellData().AddArray(arr)

                for (base_field, stat_type), face_data in all_stat_face_data.items():
                    vtk_arr = numpy_support.numpy_to_vtk(
                        face_data.astype(np.float64), deep=True
                    )
                    arr_name = (
                        base_field if stat_type is None
                        else f"{base_field}_{stat_type}"
                    )
                    vtk_arr.SetName(arr_name)
                    stat_grid.GetCellData().AddArray(vtk_arr)

                stat_vtu_path = output_dir / "ptm3d_statistics.vtu"
                writer = vtk.vtkXMLUnstructuredGridWriter()
                writer.SetFileName(str(stat_vtu_path))
                writer.SetInputData(stat_grid)
                writer.SetDataModeToAppended()
                writer.SetCompressorTypeToZLib()
                writer.Write()
                output_files.append(str(stat_vtu_path))

            output_files.append(f"{vtu_dir}/ ({len(vtu_files)} files)")

        # Write README
        readme_path = write_export_readme(
            output_dir, regridder, fields, timesteps,
            formats, per_timestep=per_timestep,
            statistics=statistics if statistics else None
        )
        output_files.append(str(readme_path))

        # Summary
        file_list = "\n".join(f"  - {f}" for f in output_files)
        if 'vtu' in formats and not do_geotiff and not do_hdf5:
            summary = f"Export complete! (VTU/PVD format)\n\nFiles created:\n{file_list}"
        else:
            proj_name = 'polar stereographic' if (regridder and regridder.projection == 'stere') else 'equirectangular'
            summary = f"Export complete! (projection: {proj_name})\n\nFiles created:\n{file_list}"
        self.export_finished.emit(True, summary)

    def _cleanup(self, output_dir):
        """Remove partial output files on cancellation."""
        try:
            for f in output_dir.glob("*.tif"):
                f.unlink(missing_ok=True)
            h5_path = output_dir / "ptm3d_data.h5"
            if h5_path.exists():
                h5_path.unlink(missing_ok=True)
            vtu_dir = output_dir / "vtu"
            if vtu_dir.exists():
                for f in vtu_dir.glob("*.vtu"):
                    f.unlink(missing_ok=True)
            pvd_path = output_dir / "ptm3d_timeseries.pvd"
            if pvd_path.exists():
                pvd_path.unlink(missing_ok=True)
        except Exception:
            pass
