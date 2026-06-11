"""
Cloud Optimized GeoTIFF (COG) exporter for PTM3D per-face statistics.

Writes single-band float32 COGs that match the format colleagues outside the
PTM3D pipeline expect: GDAL ``COG`` driver layout, 512x512 internal tiles,
LZW compression, 5 overview levels (AVERAGE resampling), NaN nodata, embedded
``STATISTICS_*`` band tags, ``OVERVIEW_RESAMPLING=AVERAGE`` in the
``IMAGE_STRUCTURE`` metadata domain, and PTM3D provenance metadata.

Implementation is a 2-pass build:
  1. Create a tiled GTiff with stats + metadata baked in, then shell out to
     ``gdaladdo -r average`` to build the overview chain. The CLI write path
     places the ``<Item name="RESAMPLING" sample="0">AVERAGE</Item>`` tag
     into the main IFD's ``GDALMetadata`` (Python's ``BuildOverviews``
     appends it at the file tail, which the COG translate step then misses).
  2. ``gdal_translate -of COG -co OVERVIEWS=FORCE_USE_EXISTING`` (CLI, not
     the Python ``gdal.Translate`` wrapper which silently drops the tag)
     consumes those overviews. Result: ``OVERVIEW_RESAMPLING=AVERAGE`` ends
     up in ``IMAGE_STRUCTURE`` metadata of the final COG, matching the
     example file.

Format reverse-engineered from
``/Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/moon/artemis/a3gt/
 ptm3d/Malapert_PTM3D-site07/cog/{ice_depth,temperature_max}.tif``.
"""

from datetime import datetime
import os
import shutil
import subprocess
import tempfile

import numpy as np

try:
    from osgeo import gdal
    GDAL_AVAILABLE = True
    # Allow exceptions instead of silent failures
    gdal.UseExceptions()
except ImportError:
    GDAL_AVAILABLE = False

from ptm_regridder import FIELD_INFO, STATISTICS_FIELDS, STAT_LABELS


def _find_gdal_tool(name):
    """Locate a GDAL CLI tool (``gdaladdo`` / ``gdal_translate``).

    Prefers the system install (``/usr/local/bin``, ``/opt/homebrew/bin``)
    because GDAL >= 3.12 is what propagates the
    ``OVERVIEW_RESAMPLING=AVERAGE`` IMAGE_STRUCTURE tag during COG translate
    - older versions (e.g. the GDAL 3.4 in the europa_viz conda env)
    silently drop it. The interpreter's own conda env tool is the fallback
    when no system GDAL is found.
    """
    candidates = [
        f'/usr/local/bin/{name}',
        f'/opt/homebrew/bin/{name}',
    ]
    py_dir = os.path.dirname(os.path.abspath(__import__('sys').executable))
    candidates.append(os.path.join(py_dir, name))
    for path in candidates:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    found = shutil.which(name)
    if found:
        return found
    raise RuntimeError(f"could not find {name!r} on PATH")


# --- WKT helpers -----------------------------------------------------------

def _stereo_wkt(pole_lat, center_lon, body_radius):
    return (
        'PROJCS["Polar_Stereographic",'
        'GEOGCS["GCS_Sphere",'
        f'DATUM["D_Sphere",SPHEROID["Sphere",{body_radius},0]],'
        'PRIMEM["Greenwich",0],'
        'UNIT["degree",0.0174532925199433]],'
        'PROJECTION["Polar_Stereographic"],'
        f'PARAMETER["latitude_of_origin",{pole_lat}],'
        f'PARAMETER["central_meridian",{center_lon}],'
        'PARAMETER["scale_factor",1],'
        'PARAMETER["false_easting",0],'
        'PARAMETER["false_northing",0],'
        'UNIT["metre",1]]'
    )


def _eqc_wkt(center_lat, center_lon, body_radius):
    return (
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


def _geo_from_regridder(regridder):
    """Return ``(wkt, geo_transform, width, height)``.

    ``geo_transform`` is the 6-element affine GDAL expects:
    ``(ulx, dx, 0, uly, 0, -dy)`` with rows running north -> south.
    """
    if regridder.projection == 'stere':
        x_grid = regridder.x_grid
        y_grid = regridder.y_grid
        dx = float(x_grid[1] - x_grid[0])
        dy = float(y_grid[1] - y_grid[0])
        west = float(x_grid[0]) - dx / 2
        north = float(y_grid[-1]) + dy / 2
        wkt = _stereo_wkt(regridder.pole_lat, regridder.center_lon,
                          regridder.body_radius)
        return wkt, (west, dx, 0.0, north, 0.0, -dy), regridder.n_lon, regridder.n_lat

    # equirectangular
    lat_array = regridder.lat_grid
    lon_array = regridder.lon_grid
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
    body_radius = regridder.body_radius
    west = body_radius * cos_ts * (west_deg - center_lon) * deg2rad
    east = body_radius * cos_ts * (east_deg - center_lon) * deg2rad
    south = body_radius * south_deg * deg2rad
    north = body_radius * north_deg * deg2rad
    dx = (east - west) / regridder.n_lon
    dy = (north - south) / regridder.n_lat
    wkt = _eqc_wkt(center_lat, center_lon, body_radius)
    return wkt, (west, dx, 0.0, north, 0.0, -dy), regridder.n_lon, regridder.n_lat


# --- Field metadata --------------------------------------------------------

def field_metadata(field_name):
    """Return ``(units, file_description, band_description)`` for a field."""
    info = FIELD_INFO.get(field_name, {})
    if info:
        units = info.get('units', '')
        label = info.get('label', field_name)
        band = f"{label} ({units})" if units else label
        return units, label, band

    parts = field_name.rsplit('_', 1)
    if len(parts) == 2 and parts[1] in STAT_LABELS:
        base = STATISTICS_FIELDS.get(parts[0], {})
        if base:
            stat = STAT_LABELS[parts[1]]
            units = base.get('units', '')
            label = f"{base.get('label', parts[0])} ({stat})"
            return units, label, label

    base = STATISTICS_FIELDS.get(field_name, {})
    if base:
        units = base.get('units', '')
        label = base.get('label', field_name)
        band = f"{label} ({units})" if units else label
        return units, label, band

    return '', field_name, field_name


# --- Statistics ------------------------------------------------------------

def _compute_stats_tags(data):
    """Compute STATISTICS_* tag dict from a 2D float32 array with NaN nodata."""
    flat = data.ravel()
    valid_mask = np.isfinite(flat)
    n_total = flat.size
    n_valid = int(valid_mask.sum())
    valid_pct = 100.0 * n_valid / n_total if n_total else 0.0

    tags = {'STATISTICS_VALID_PERCENT': f"{valid_pct:.2f}"}
    if n_valid > 0:
        valid = flat[valid_mask].astype(np.float64)
        tags.update({
            'STATISTICS_MINIMUM': f"{float(valid.min()):.14g}",
            'STATISTICS_MAXIMUM': f"{float(valid.max()):.14g}",
            'STATISTICS_MEAN':    f"{float(valid.mean()):.14g}",
            'STATISTICS_STDDEV':  f"{float(valid.std()):.14g}",
        })
    return tags


# --- Overview levels -------------------------------------------------------

def _pick_overview_levels(width, height, blocksize):
    """Pick power-of-two overview levels until the smaller dim < blocksize.

    Matches the example COG's 5 levels on 10003x10018 with blocksize 512:
    2, 4, 8, 16, 32.
    """
    levels = []
    factor = 2
    while min(width, height) // factor >= max(blocksize // 2, 1):
        levels.append(factor)
        factor *= 2
        if factor > 64:
            break
    return levels


# --- Main export -----------------------------------------------------------

# Default nodata for out-of-mesh / invalid pixels. A numeric sentinel
# (float32 -FLT_MAX) rather than NaN: every GIS reliably honors a numeric
# nodata as transparent, whereas float NaN nodata is silently rendered as the
# color-ramp minimum by many viewers (e.g. ice_depth 0 -> spurious "ice at the
# surface" in the rectangular corners outside the circular mesh). Pass
# nodata_value=float('nan') to restore the legacy NaN convention.
DEFAULT_NODATA = float(np.finfo(np.float32).min)  # -3.4028235e+38


def export_field(output_path, face_values, regridder, field_name,
                 *, blocksize=512, compress='LZW',
                 overview_resampling='AVERAGE',
                 software='PTM3D',
                 interpolation='linear',
                 nodata_value=DEFAULT_NODATA):
    """Regrid ``face_values`` and write a COG to ``output_path``.

    Args:
        output_path: Destination .tif path.
        face_values: 1D per-face values (length must equal regridder face count).
        regridder: Built ``MeshRegridder`` instance.
        field_name: PTM3D stats field name, e.g. ``temperature_max``,
            ``ice_depth``.
        blocksize: Internal tile size (default 512 - matches example COGs).
        compress: TIFF compression (default LZW).
        overview_resampling: Resampling for overview levels (default AVERAGE).
        software: ``software`` tag written to the file.

    Returns ``(grid, stats_tags)``.
    """
    if not GDAL_AVAILABLE:
        raise RuntimeError("osgeo.gdal is required for COG export")

    if interpolation == 'linear':
        grid = regridder.regrid_linear(face_values)
    elif interpolation == 'idw':
        grid = regridder.regrid(face_values)
    else:
        raise ValueError(f"unknown interpolation mode: {interpolation!r}")
    grid_flipped = grid[::-1, :]  # north-up GeoTIFF row order

    wkt, geo_transform, width, height = _geo_from_regridder(regridder)
    units, file_desc, band_desc = field_metadata(field_name)
    stats_tags = _compute_stats_tags(grid_flipped)
    created = datetime.now().isoformat()

    # Replace non-finite (out-of-mesh / invalid) pixels with the nodata
    # sentinel AFTER stats are computed from the NaN mask, so a finite sentinel
    # can never inflate the valid-percent or skew the min/max/mean tags. When
    # nodata_value is itself NaN (legacy mode) this is a no-op.
    if np.isfinite(nodata_value):
        grid_flipped = np.where(
            np.isfinite(grid_flipped), grid_flipped, np.float32(nodata_value)
        ).astype(np.float32, copy=False)

    output_path = str(output_path)
    output_dir = os.path.dirname(output_path) or '.'
    os.makedirs(output_dir, exist_ok=True)

    # --- Pass 1: tiled GTiff with overviews + all metadata baked in --------
    with tempfile.NamedTemporaryFile(
        suffix='.tif', prefix='_ptm_cog_', dir=output_dir, delete=False
    ) as tmp:
        tmp_path = tmp.name

    try:
        gtiff_opts = [
            f'COMPRESS={compress}',
            'TILED=YES',
            f'BLOCKXSIZE={blocksize}',
            f'BLOCKYSIZE={blocksize}',
            'BIGTIFF=IF_SAFER',
        ]
        drv = gdal.GetDriverByName('GTiff')
        ds = drv.Create(tmp_path, width, height, 1, gdal.GDT_Float32,
                        options=gtiff_opts)
        try:
            ds.SetGeoTransform(geo_transform)
            ds.SetProjection(wkt)

            band = ds.GetRasterBand(1)
            band.WriteArray(grid_flipped)
            band.SetNoDataValue(float(nodata_value))
            band.SetDescription(band_desc)
            for k, v in stats_tags.items():
                band.SetMetadataItem(k, v)

            # File-level provenance tags
            ds.SetMetadataItem('field', field_name)
            ds.SetMetadataItem('units', units)
            ds.SetMetadataItem('description', file_desc)
            ds.SetMetadataItem('created', created)
            ds.SetMetadataItem('software', software)

            ds.FlushCache()
        finally:
            ds = None

        # Build overviews via the gdaladdo CLI (the Python BuildOverviews
        # equivalent writes the GDALMetadata tag at the file tail, which
        # the downstream COG translate then fails to pick up - see
        # module-level docstring).
        levels = _pick_overview_levels(width, height, blocksize)
        if levels:
            gdaladdo = _find_gdal_tool('gdaladdo')
            subprocess.check_call(
                [gdaladdo, '-r', overview_resampling.lower(), tmp_path,
                 *[str(L) for L in levels]],
                stdout=subprocess.DEVNULL,
            )

        # --- Pass 2: gdal_translate -of COG -co OVERVIEWS=FORCE_USE_EXISTING
        # (CLI, not Python gdal.Translate - the Python wrapper drops the
        # OVERVIEW_RESAMPLING tag on the way through).
        gdal_translate = _find_gdal_tool('gdal_translate')
        subprocess.check_call(
            [gdal_translate, '-of', 'COG', '-q',
             '-co', f'COMPRESS={compress}',
             '-co', f'BLOCKSIZE={blocksize}',
             '-co', 'OVERVIEWS=FORCE_USE_EXISTING',
             '-co', f'OVERVIEW_COMPRESS={compress}',
             '-co', 'NUM_THREADS=ALL_CPUS',
             '-co', 'BIGTIFF=IF_SAFER',
             tmp_path, output_path],
            stdout=subprocess.DEVNULL,
        )
    finally:
        for ext in ('', '.ovr', '.aux.xml'):
            p = tmp_path + ext
            if os.path.isfile(p):
                try:
                    os.remove(p)
                except OSError:
                    pass

    return grid_flipped, stats_tags
