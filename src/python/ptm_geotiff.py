"""
PTM3D GeoTIFF Reader

Reads GeoTIFF files exported by the PTM3D GUI and provides convenient
access to data arrays, geographic coordinates, and time series extraction.

Handles both equirectangular and polar stereographic projections
automatically by parsing the embedded CRS.

Usage:
    from ptm_geotiff import read_geotiff, read_geotiff_folder, sample_points

    # Single multi-band GeoTIFF
    data = read_geotiff("temperature.tif")
    print(data['values'].shape)   # (n_timesteps, n_lat, n_lon)
    print(data['latitudes'].shape) # (n_lat,) for eqc, (n_lat, n_lon) for stere

    # Folder of per-timestep GeoTIFFs
    data = read_geotiff_folder("temperature/")

    # Extract time series at specific lat/lon points
    pts = [(61.5, 180.2), (62.0, 179.8)]
    ts = sample_points(data, pts)  # (n_points, n_timesteps)
"""

import re
import numpy as np
from pathlib import Path

try:
    import rasterio
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False


# ---------------------------------------------------------------------------
# CRS parsing
# ---------------------------------------------------------------------------

def _parse_wkt_params(wkt):
    """Extract projection parameters from a WKT CRS string.

    Returns dict with keys:
        projection: 'eqc' or 'stere' or 'unknown'
        body_radius: float (meters)
        central_meridian: float (degrees)
        standard_parallel_1: float (degrees, eqc only)
        latitude_of_origin: float (degrees, stere only)
    """
    params = {}

    proj_match = re.search(r'PROJECTION\["([^"]+)"\]', wkt)
    if proj_match:
        name = proj_match.group(1)
        if 'Cylindrical' in name or 'Equidistant' in name or 'Equirectangular' in name:
            params['projection'] = 'eqc'
        elif 'Stereographic' in name:
            params['projection'] = 'stere'
        else:
            params['projection'] = 'unknown'
    else:
        params['projection'] = 'unknown'

    spheroid_match = re.search(r'SPHEROID\["[^"]*",([0-9.eE+\-]+)', wkt)
    if spheroid_match:
        params['body_radius'] = float(spheroid_match.group(1))

    for pm in re.finditer(r'PARAMETER\["([^"]+)",([0-9.eE+\-]+)\]', wkt):
        name = pm.group(1).lower()
        value = float(pm.group(2))
        if 'central_meridian' in name:
            params['central_meridian'] = value
        elif 'standard_parallel' in name:
            params['standard_parallel_1'] = value
        elif 'latitude_of_origin' in name:
            params['latitude_of_origin'] = value

    return params


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------

def _pixel_coords(transform, width, height):
    """Return 1-D arrays of pixel-center x and y in the CRS.

    Uses the affine transform directly (no PROJ database needed).
    y is returned in file order (north-to-south for north-up grids).
    """
    x = transform.c + (np.arange(width) + 0.5) * transform.a
    y = transform.f + (np.arange(height) + 0.5) * transform.e
    return x, y


def _xy_to_latlon_eqc(x, y, body_radius, central_meridian, standard_parallel):
    """Convert equirectangular projected metres to lat/lon degrees."""
    cos_ts = np.cos(np.radians(standard_parallel))
    lat = np.degrees(np.asarray(y, dtype=np.float64) / body_radius)
    lon = central_meridian + np.degrees(
        np.asarray(x, dtype=np.float64) / (body_radius * cos_ts)
    )
    return lat, lon


def _xy_to_latlon_stere(x, y, body_radius, latitude_of_origin, central_meridian):
    """Convert polar stereographic projected metres to lat/lon degrees."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    rho = np.sqrt(x**2 + y**2)

    if latitude_of_origin > 0:  # north pole
        lat = np.degrees(np.pi / 2 - 2 * np.arctan(rho / (2 * body_radius)))
        lon = central_meridian + np.degrees(np.arctan2(x, -y))
    else:  # south pole
        lat = np.degrees(-np.pi / 2 + 2 * np.arctan(rho / (2 * body_radius)))
        lon = central_meridian + np.degrees(np.arctan2(x, y))

    return lat, lon


def _latlon_to_xy_eqc(lat, lon, body_radius, central_meridian, standard_parallel):
    """Convert lat/lon degrees to equirectangular projected metres."""
    cos_ts = np.cos(np.radians(standard_parallel))
    x = body_radius * cos_ts * np.radians(np.asarray(lon) - central_meridian)
    y = body_radius * np.radians(np.asarray(lat))
    return x, y


def _latlon_to_xy_stere(lat, lon, body_radius, latitude_of_origin, central_meridian):
    """Convert lat/lon degrees to polar stereographic projected metres."""
    lat_r = np.radians(np.asarray(lat, dtype=np.float64))
    dlon_r = np.radians(np.asarray(lon, dtype=np.float64) - central_meridian)

    if latitude_of_origin > 0:
        rho = 2 * body_radius * np.tan(np.pi / 4 - lat_r / 2)
        x = rho * np.sin(dlon_r)
        y = -rho * np.cos(dlon_r)
    else:
        rho = 2 * body_radius * np.tan(np.pi / 4 + lat_r / 2)
        x = rho * np.sin(dlon_r)
        y = rho * np.cos(dlon_r)

    return x, y


def _compute_coordinates(transform, width, height, crs_params):
    """Compute geographic and projected coordinate arrays.

    Returns:
        (lat, lon, x_proj, y_proj) where:
        - lat, lon: geographic coordinates (1D for eqc, 2D for stere)
        - x_proj, y_proj: 1D arrays of projected CRS coordinates (metres),
          in ascending order (matching the flipped data arrays)
    """
    x, y = _pixel_coords(transform, width, height)

    # Flip y to ascending order (file stores north-up, we want south-to-north)
    y = y[::-1]

    proj = crs_params.get('projection', 'unknown')
    R = crs_params.get('body_radius', 1737400.0)
    lon0 = crs_params.get('central_meridian', 0.0)

    if proj == 'eqc':
        lat_ts = crs_params.get('standard_parallel_1', 0.0)
        lat, lon = _xy_to_latlon_eqc(x, y, R, lon0, lat_ts)
        # x -> lon (1-D), y -> lat (1-D) for a regular rectangular grid
        return lat, lon, x, y
    elif proj == 'stere':
        lat_0 = crs_params.get('latitude_of_origin', 90.0)
        X, Y = np.meshgrid(x, y)
        lat, lon = _xy_to_latlon_stere(X, Y, R, lat_0, lon0)
        return lat, lon, x, y
    else:
        # Fallback: return projected coordinates as both
        return y, x, x, y


# ---------------------------------------------------------------------------
# Main reader functions
# ---------------------------------------------------------------------------

def read_geotiff(path):
    """Read a PTM3D GeoTIFF file.

    Handles both multi-band (all timesteps) and single-band files.

    Args:
        path: Path to a .tif file

    Returns:
        dict with keys:
            values:      ndarray, shape (n_bands, n_lat, n_lon) or (n_lat, n_lon)
            latitudes:   ndarray, 1D (eqc) or 2D (stere)
            longitudes:  ndarray, 1D (eqc) or 2D (stere)
            x:           ndarray, 1D projected x coordinates (metres, ascending)
            y:           ndarray, 1D projected y coordinates (metres, ascending)
            time_labels: list of str (band descriptions, typically UTC times)
            field:       str (field name from tags, e.g. 'temperature')
            units:       str (e.g. 'K', 'W/m²')
            projection:  str ('eqc' or 'stere')
            body_radius: float (metres)
            crs_wkt:     str (full WKT string)
    """
    if not RASTERIO_AVAILABLE:
        raise RuntimeError("rasterio is required: pip install rasterio")

    path = Path(path)
    with rasterio.open(str(path)) as ds:
        # Read all bands and flip to latitude-ascending order
        data = ds.read()  # (n_bands, height, width), row 0 = north
        data = data[:, ::-1, :]  # flip to row 0 = south (lat ascending)

        # Replace nodata with NaN
        if ds.nodata is not None:
            data = data.astype(np.float32)
            data[data == ds.nodata] = np.nan

        # Parse CRS
        wkt = ds.crs.to_wkt() if ds.crs else ''
        crs_params = _parse_wkt_params(wkt)

        # Compute coordinates
        lat, lon, x_proj, y_proj = _compute_coordinates(
            ds.transform, ds.width, ds.height, crs_params
        )

        # Band descriptions (time labels)
        time_labels = list(ds.descriptions) if ds.descriptions else []
        time_labels = [t if t else '' for t in time_labels]

        # Tags
        tags = ds.tags()
        field = tags.get('field', '')
        units = tags.get('units', '')

    # Squeeze single-band to 2D
    if data.shape[0] == 1:
        data = data[0]

    return {
        'values': data,
        'latitudes': lat,
        'longitudes': lon,
        'x': x_proj,
        'y': y_proj,
        'time_labels': time_labels,
        'field': field,
        'units': units,
        'projection': crs_params.get('projection', 'unknown'),
        'body_radius': crs_params.get('body_radius', 0.0),
        'crs_wkt': wkt,
    }


def read_geotiff_folder(folder_path, pattern="*.tif"):
    """Read all per-timestep GeoTIFF files in a folder into a 3D array.

    Files are sorted by name, which matches the PTM3D naming convention
    ({field}_{NNNN}.tif).

    Args:
        folder_path: Path to folder containing .tif files
        pattern: Glob pattern for files (default "*.tif")

    Returns:
        dict with same keys as read_geotiff(), except:
            values: ndarray of shape (n_timesteps, n_lat, n_lon)
            time_labels: list of str collected from each file's band description
    """
    if not RASTERIO_AVAILABLE:
        raise RuntimeError("rasterio is required: pip install rasterio")

    folder = Path(folder_path)
    files = sorted(folder.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' in {folder}")

    # Read the first file for coordinates and metadata
    first = read_geotiff(files[0])
    shape_2d = first['values'].shape[-2:]  # (n_lat, n_lon)

    # Pre-allocate output array
    n_files = len(files)
    values = np.full((n_files, *shape_2d), np.nan, dtype=np.float32)
    time_labels = []

    # Read first file's data
    v = first['values']
    values[0] = v if v.ndim == 2 else v[0]
    tl = first['time_labels']
    time_labels.append(tl[0] if tl else '')

    # Read remaining files
    for i, f in enumerate(files[1:], start=1):
        with rasterio.open(str(f)) as ds:
            band = ds.read(1)[::-1, :]  # flip to lat-ascending
            if ds.nodata is not None:
                band = band.astype(np.float32)
                band[band == ds.nodata] = np.nan
            values[i] = band
            desc = ds.descriptions
            time_labels.append(desc[0] if desc and desc[0] else '')

    return {
        'values': values,
        'latitudes': first['latitudes'],
        'longitudes': first['longitudes'],
        'x': first['x'],
        'y': first['y'],
        'time_labels': time_labels,
        'field': first['field'],
        'units': first['units'],
        'projection': first['projection'],
        'body_radius': first['body_radius'],
        'crs_wkt': first['crs_wkt'],
    }


# ---------------------------------------------------------------------------
# Point sampling
# ---------------------------------------------------------------------------

def _find_nearest_pixel(lat_target, lon_target, latitudes, longitudes,
                        crs_params=None, transform=None, width=None):
    """Find the (row, col) of the nearest pixel to a target lat/lon.

    For equirectangular grids (1D lat/lon), uses simple 1D lookups.
    For stereographic grids (2D lat/lon), projects the target point and
    uses the affine transform for direct index computation.
    """
    if latitudes.ndim == 1:
        # Equirectangular: independent lat and lon axes
        row = int(np.argmin(np.abs(latitudes - lat_target)))
        col = int(np.argmin(np.abs(longitudes - lon_target)))
        return row, col
    else:
        # Stereographic: project target to (x, y), then pixel index
        if crs_params and transform:
            R = crs_params.get('body_radius', 1737400.0)
            lat_0 = crs_params.get('latitude_of_origin', 90.0)
            lon0 = crs_params.get('central_meridian', 0.0)
            tx, ty = _latlon_to_xy_stere(
                lat_target, lon_target, R, lat_0, lon0
            )
            # Inverse affine: (col, row) from (x, y)
            col = (float(tx) - transform.c) / transform.a - 0.5
            row = (float(ty) - transform.f) / transform.e - 0.5
            # Flip row index (file is north-up, our array is south-up)
            row = (latitudes.shape[0] - 1) - row
            row = int(np.clip(np.round(row), 0, latitudes.shape[0] - 1))
            col = int(np.clip(np.round(col), 0, latitudes.shape[1] - 1))
            return row, col

        # Fallback: brute-force search over 2D grid
        dist = (latitudes - lat_target)**2 + (longitudes - lon_target)**2
        idx = np.nanargmin(dist)
        row, col = np.unravel_index(idx, latitudes.shape)
        return int(row), int(col)


def sample_point(data, lat, lon):
    """Extract the time series at the nearest pixel to (lat, lon).

    Args:
        data: dict returned by read_geotiff() or read_geotiff_folder()
        lat: target latitude (degrees)
        lon: target longitude (degrees)

    Returns:
        1D ndarray of shape (n_timesteps,), or a scalar for single-timestep data
    """
    crs_params = _parse_wkt_params(data.get('crs_wkt', ''))

    # Reconstruct transform from the CRS parameters for stereographic lookup
    transform = None
    if crs_params.get('projection') == 'stere' and data.get('crs_wkt'):
        # We need the original transform; re-open the file is not ideal,
        # so fall back to brute-force 2D search
        pass

    row, col = _find_nearest_pixel(
        lat, lon, data['latitudes'], data['longitudes'], crs_params
    )

    vals = data['values']
    if vals.ndim == 2:
        return vals[row, col]
    else:
        return vals[:, row, col]


def sample_points(data, points):
    """Extract time series at multiple lat/lon points.

    Args:
        data: dict returned by read_geotiff() or read_geotiff_folder()
        points: list of (lat, lon) tuples

    Returns:
        2D ndarray of shape (n_points, n_timesteps)
        For single-timestep data, shape is (n_points,)
    """
    results = []
    for lat, lon in points:
        results.append(sample_point(data, lat, lon))
    return np.array(results)


def get_coordinate_at_pixel(data, row, col):
    """Get the (lat, lon) at a specific pixel index.

    Args:
        data: dict returned by read_geotiff() or read_geotiff_folder()
        row: row index (0 = southernmost for lat-ascending data)
        col: column index

    Returns:
        (lat, lon) tuple in degrees
    """
    lats = data['latitudes']
    lons = data['longitudes']
    if lats.ndim == 1:
        return float(lats[row]), float(lons[col])
    else:
        return float(lats[row, col]), float(lons[row, col])
