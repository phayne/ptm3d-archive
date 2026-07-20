"""
PTM3D HDF5 Reader

Reads HDF5 files exported by the PTM3D GUI and provides convenient access
to both regridded (regular lat/lon grid) and unstructured (per-face) data,
along with geographic coordinates and time series extraction.

The HDF5 format contains:
  /mesh/       - Original unstructured triangular mesh geometry
  /time/       - Timestep labels and elapsed seconds
  /data/       - Per-face unstructured data for each field
  /regridded/  - Regridded data on a regular lat/lon grid (if exported)

Usage:
    from ptm_hdf5 import read_hdf5, read_hdf5_field, list_hdf5_fields, sample_points

    # List available fields
    fields = list_hdf5_fields("simulation.h5")

    # Read one field (returns dict compatible with ptm_geotiff reader)
    data = read_hdf5_field("simulation.h5", "temperature")
    print(data['values'].shape)   # (n_timesteps, n_lat, n_lon)

    # Read everything (all fields, mesh, unstructured + regridded)
    full = read_hdf5("simulation.h5")

    # Extract time series at specific lat/lon points
    pts = [(61.5, 180.2), (62.0, 179.8)]
    ts = sample_points(data, pts)  # (n_points, n_timesteps)
"""

import numpy as np
from pathlib import Path

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False


def _require_h5py():
    if not H5PY_AVAILABLE:
        raise RuntimeError("h5py is required: pip install h5py")


# ---------------------------------------------------------------------------
# Field listing
# ---------------------------------------------------------------------------

def list_hdf5_fields(path):
    """List the available data fields in a PTM3D HDF5 file.

    Returns field names found in both /data/ (unstructured) and
    /regridded/ (gridded) groups.

    Args:
        path: Path to a .h5 file

    Returns:
        dict with keys:
            unstructured: list of str (field names in /data/)
            regridded:    list of str (field names in /regridded/, excluding
                          coordinate datasets)
    """
    _require_h5py()
    coord_names = {'latitudes', 'longitudes', 'x', 'y'}
    with h5py.File(str(path), 'r') as f:
        unstruct = list(f['data'].keys()) if 'data' in f else []
        regrid = []
        if 'regridded' in f:
            regrid = [k for k in f['regridded'].keys() if k not in coord_names]
        return {'unstructured': unstruct, 'regridded': regrid}


# ---------------------------------------------------------------------------
# Single-field reader (GeoTIFF-compatible dict)
# ---------------------------------------------------------------------------

def read_hdf5_field(path, field):
    """Read a single field from a PTM3D HDF5 file.

    Returns a dict with the same keys as ptm_geotiff.read_geotiff_folder(),
    making it a drop-in replacement for plotting and analysis code.

    Uses the regridded data if available, otherwise falls back to
    unstructured face data.

    Args:
        path: Path to a .h5 file
        field: Field name (e.g. 'temperature', 'illumination')

    Returns:
        dict with keys:
            values:      ndarray (n_timesteps, n_lat, n_lon) or (n_timesteps, n_faces)
            latitudes:   ndarray, 1D (eqc) or 2D (stere) for regridded;
                         1D (n_faces,) for unstructured
            longitudes:  ndarray, same as latitudes
            x:           ndarray, 1D projected x (metres), or empty array
            y:           ndarray, 1D projected y (metres), or empty array
            time_labels: list of str (UTC time labels)
            elapsed_seconds: ndarray (n_timesteps,) of float64
            field:       str (field name)
            units:       str (e.g. 'K', 'W/m2')
            projection:  str ('eqc', 'stere', or '')
            body_radius: float (metres)
            source:      str ('regridded' or 'unstructured')
    """
    _require_h5py()
    path = Path(path)

    with h5py.File(str(path), 'r') as f:
        # Time information
        time_labels, elapsed = _read_time(f)

        # Try regridded first
        if 'regridded' in f and field in f['regridded']:
            rg = f['regridded']
            values = rg[field][:]
            lats = rg['latitudes'][:]
            lons = rg['longitudes'][:]
            x_proj = rg['x'][:] if 'x' in rg else np.array([])
            y_proj = rg['y'][:] if 'y' in rg else np.array([])
            units = rg[field].attrs.get('units', '')
            if isinstance(units, bytes):
                units = units.decode()
            proj = rg.attrs.get('projection', '')
            if isinstance(proj, bytes):
                proj = proj.decode()
            body_r = float(rg.attrs.get('body_radius_m', 0.0))
            source = 'regridded'

        elif 'data' in f and field in f['data']:
            dg = f['data']
            values = dg[field][:]
            lats = f['mesh']['latitudes'][:] if 'mesh' in f else np.array([])
            lons = f['mesh']['longitudes'][:] if 'mesh' in f else np.array([])
            x_proj = np.array([])
            y_proj = np.array([])
            units = dg[field].attrs.get('units', '')
            if isinstance(units, bytes):
                units = units.decode()
            proj = ''
            body_r = 0.0
            source = 'unstructured'

        else:
            available = list_hdf5_fields(path)
            raise KeyError(
                f"Field '{field}' not found. Available: "
                f"unstructured={available['unstructured']}, "
                f"regridded={available['regridded']}"
            )

    return {
        'values': values,
        'latitudes': lats,
        'longitudes': lons,
        'x': x_proj,
        'y': y_proj,
        'time_labels': time_labels,
        'elapsed_seconds': elapsed,
        'field': field,
        'units': units,
        'projection': proj,
        'body_radius': body_r,
        'source': source,
    }


# ---------------------------------------------------------------------------
# Full file reader
# ---------------------------------------------------------------------------

def read_hdf5(path):
    """Read all data from a PTM3D HDF5 file.

    Returns a dict with the complete file contents organized hierarchically:
    mesh geometry, time arrays, per-field unstructured and regridded data,
    and projection metadata.

    Args:
        path: Path to a .h5 file

    Returns:
        dict with keys:
            mesh: dict with 'latitudes', 'longitudes', 'vertices', 'faces'
            time_labels:      list of str (UTC labels)
            elapsed_seconds:  ndarray (n_timesteps,)
            fields: dict mapping field_name -> {
                'values':      regridded ndarray (n_t, n_lat, n_lon) or None
                'face_values': unstructured ndarray (n_t, n_faces) or None
                'units': str
                'description': str
            }
            latitudes:   ndarray (regridded grid coords, or empty)
            longitudes:  ndarray (regridded grid coords, or empty)
            x:           ndarray (projected x, metres, or empty)
            y:           ndarray (projected y, metres, or empty)
            projection:  str ('eqc', 'stere', or '')
            body_radius: float (metres)
            resolution:  float (grid resolution in degrees, or 0)
            proj4:       str (PROJ4 string, or '')
            format_version: str
    """
    _require_h5py()
    path = Path(path)
    coord_names = {'latitudes', 'longitudes', 'x', 'y'}

    with h5py.File(str(path), 'r') as f:
        # Root attributes
        fmt_ver = _decode(f.attrs.get('format_version', ''))

        # Mesh
        mesh = {}
        if 'mesh' in f:
            mg = f['mesh']
            mesh = {
                'latitudes': mg['latitudes'][:] if 'latitudes' in mg else np.array([]),
                'longitudes': mg['longitudes'][:] if 'longitudes' in mg else np.array([]),
                'vertices': mg['vertices'][:] if 'vertices' in mg else np.array([]),
                'faces': mg['faces'][:] if 'faces' in mg else np.array([]),
            }

        # Time
        time_labels, elapsed = _read_time(f)

        # Unstructured data (/data/)
        unstruct = {}
        if 'data' in f:
            for name in f['data']:
                ds = f['data'][name]
                unstruct[name] = {
                    'data': ds[:],
                    'units': _decode(ds.attrs.get('units', '')),
                    'description': _decode(ds.attrs.get('description', '')),
                }

        # Regridded data (/regridded/)
        regridded = {}
        grid_lats = np.array([])
        grid_lons = np.array([])
        x_proj = np.array([])
        y_proj = np.array([])
        proj = ''
        body_r = 0.0
        resolution = 0.0
        proj4 = ''

        if 'regridded' in f:
            rg = f['regridded']
            grid_lats = rg['latitudes'][:] if 'latitudes' in rg else np.array([])
            grid_lons = rg['longitudes'][:] if 'longitudes' in rg else np.array([])
            x_proj = rg['x'][:] if 'x' in rg else np.array([])
            y_proj = rg['y'][:] if 'y' in rg else np.array([])
            proj = _decode(rg.attrs.get('projection', ''))
            body_r = float(rg.attrs.get('body_radius_m', 0.0))
            resolution = float(rg.attrs.get('resolution_degrees', 0.0))
            proj4 = _decode(rg.attrs.get('proj4', ''))

            for name in rg:
                if name not in coord_names:
                    ds = rg[name]
                    regridded[name] = {
                        'data': ds[:],
                        'units': _decode(ds.attrs.get('units', '')),
                        'description': _decode(ds.attrs.get('description', '')),
                    }

        # Merge into per-field dict
        all_fields = set(list(unstruct.keys()) + list(regridded.keys()))
        fields = {}
        for name in sorted(all_fields):
            u = unstruct.get(name)
            r = regridded.get(name)
            fields[name] = {
                'values': r['data'] if r else None,
                'face_values': u['data'] if u else None,
                'units': (r or u or {}).get('units', ''),
                'description': (r or u or {}).get('description', ''),
            }

    return {
        'mesh': mesh,
        'time_labels': time_labels,
        'elapsed_seconds': elapsed,
        'fields': fields,
        'latitudes': grid_lats,
        'longitudes': grid_lons,
        'x': x_proj,
        'y': y_proj,
        'projection': proj,
        'body_radius': body_r,
        'resolution': resolution,
        'proj4': proj4,
        'format_version': fmt_ver,
    }


# ---------------------------------------------------------------------------
# Point sampling
# ---------------------------------------------------------------------------

def sample_point(data, lat, lon):
    """Extract the time series at the nearest grid pixel to (lat, lon).

    Works with dicts returned by read_hdf5_field(). For unstructured data
    (no regular grid), finds the nearest face center.

    Args:
        data: dict returned by read_hdf5_field()
        lat: target latitude (degrees)
        lon: target longitude (degrees)

    Returns:
        1D ndarray of shape (n_timesteps,), or scalar for single-timestep data
    """
    lats = data['latitudes']
    lons = data['longitudes']
    vals = data['values']

    if lats.ndim == 1 and vals.ndim == 3:
        # Regular 1D grid (eqc regridded): independent lat/lon axes
        row = int(np.argmin(np.abs(lats - lat)))
        col = int(np.argmin(np.abs(lons - lon)))
        return vals[:, row, col]

    elif lats.ndim == 2 and vals.ndim == 3:
        # 2D coordinate grid (stere regridded): brute-force 2D search
        dist = (lats - lat)**2 + (lons - lon)**2
        idx = np.nanargmin(dist)
        row, col = np.unravel_index(idx, lats.shape)
        return vals[:, int(row), int(col)]

    elif lats.ndim == 1 and vals.ndim == 2:
        # Unstructured face data: nearest face center
        dist = (lats - lat)**2 + (lons - lon)**2
        face = int(np.argmin(dist))
        return vals[:, face]

    else:
        raise ValueError(
            f"Unexpected array shapes: latitudes {lats.shape}, values {vals.shape}"
        )


def sample_points(data, points):
    """Extract time series at multiple lat/lon points.

    Args:
        data: dict returned by read_hdf5_field()
        points: list of (lat, lon) tuples

    Returns:
        2D ndarray of shape (n_points, n_timesteps)
    """
    results = [sample_point(data, lat, lon) for lat, lon in points]
    return np.array(results)


def get_coordinate_at_pixel(data, row, col):
    """Get the (lat, lon) at a specific pixel index in the regridded grid.

    Args:
        data: dict returned by read_hdf5_field() (regridded source)
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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _decode(val):
    """Decode bytes to str if needed (h5py attribute values)."""
    if isinstance(val, bytes):
        return val.decode()
    return str(val) if val is not None else ''


def _read_time(f):
    """Read time group from an open h5py File, returning (labels, elapsed)."""
    time_labels = []
    elapsed = np.array([])
    if 'time' in f:
        tg = f['time']
        if 'utc_labels' in tg:
            raw = tg['utc_labels'][:]
            time_labels = [_decode(s) for s in raw]
        if 'elapsed_seconds' in tg:
            elapsed = tg['elapsed_seconds'][:]
    return time_labels, elapsed
