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
``<campaign-root>/
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

from ptm.results import (FIELD_INFO, STATISTICS_FIELDS, STAT_LABELS,
                         depth_of_field, field_info, nonfinite_census)


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

    # `field_info`, not `STATISTICS_FIELDS`: that catalog is fixed, and a
    # subsurface temperature map is named for the depth it holds, so the set of
    # them belongs to the run. One resolver keeps this raster and the GUI's own
    # export describing the same file the same way.
    parts = field_name.rsplit('_', 1)
    if len(parts) == 2 and parts[1] in STAT_LABELS:
        base = field_info(parts[0])
        if base:
            stat = STAT_LABELS[parts[1]]
            units = base.get('units', '')
            label = f"{base.get('label', parts[0])} ({stat})"
            if 'depth_m' in base:
                label = (f"Temperature at {base['depth_m']:g} m depth "
                         f"({stat})")
            return units, label, label

    base = field_info(field_name)
    if base:
        units = base.get('units', '')
        label = base.get('label', field_name)
        if 'depth_m' in base:
            label = f"Temperature at {base['depth_m']:g} m depth"
        band = f"{label} ({units})" if units else label
        return units, label, band

    return '', field_name, field_name


def depth_band_tags(field_name, manifest_entries=None):
    """Band metadata naming the depth a subsurface map holds, or ``{}``.

    Three separate facts, and they are not the same number:

    ``depth_m``
        the depth these pixels hold, AS PRECISELY AS THE RUN RECORDED IT: the
        manifest's resolved depth when there is a manifest, and otherwise the
        three decimals the file name carries. The two differ, and by more than
        rounding noise -- a real lunar run resolved a 0.05 m request to
        0.052724 m and named the file ``z0p053m``, so the name is 276 um
        coarser than the number. The name is a naming convention; this is the
        measurement. Always present for a depth map.
    ``depth_requested_m``
        the depth that was asked for. Present only when the run wrote
        ``statistics/depth_map_z.json``.
    ``depth_residual_m``
        resolved minus requested. Nonzero exactly when the run moved the
        request to the nearest depth it solves on, which is the default -- so
        a reader of the raster can tell a moved depth from one that was asked
        for, WITHOUT the run directory in front of them. A GeoTIFF outlives
        the directory it came from, and a depth that silently moved is the
        kind of thing that gets quoted as the depth requested.

    `manifest_entries` is :func:`ptm.results.depth_entries` output. Without it
    only ``depth_m`` is reported, at the file name's precision: an absent
    manifest yields fewer tags and a coarser one, never invented ones.
    """
    base = field_name.rsplit('_', 1)[0] if field_name.rsplit('_', 1)[-1] \
        in STAT_LABELS else field_name
    depth_m = depth_of_field(base)
    if depth_m is None:
        return {}
    tags = {'depth_m': f"{depth_m:.3f}"}
    entry = (manifest_entries or {}).get(base)
    if isinstance(entry, dict):
        if 'resolved_m' in entry:
            tags['depth_m'] = f"{float(entry['resolved_m']):.9g}"
        if 'requested_m' in entry:
            tags['depth_requested_m'] = f"{float(entry['requested_m']):.9g}"
        if 'residual_m' in entry:
            tags['depth_residual_m'] = f"{float(entry['residual_m']):.9g}"
        if 'on_node' in entry:
            tags['depth_on_solved_node'] = \
                'true' if entry['on_node'] else 'false'
    return tags


# --- Statistics ------------------------------------------------------------

#: Every tag name `_compute_stats_tags` can produce. Named once so
#: `export_field` can refuse to let a caller's extra band metadata collide with
#: one, rather than one of them quietly winning.
_STATS_TAG_NAMES = ('STATISTICS_VALID_PERCENT', 'STATISTICS_MINIMUM',
                    'STATISTICS_MAXIMUM', 'STATISTICS_MEAN',
                    'STATISTICS_STDDEV')


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


# --- Sentinels -------------------------------------------------------------
#
# ⛔⛔ A raster has ONE nodata value and PTM3D products carry TWO sentinels that
# do not mean the same thing:
#
#   NaN    "not evaluated" / "no value here" -- `effective_albedo` on a facet
#          that was never lit (26,057 of 2,397,890 on the reference lunar polar
#          run), `shadow_depth_{avg,max,min}` on a facet with no resolved epoch
#          (1,240,686 of 2,397,890 on the same run).
#   +inf   "still umbral at the search ceiling" -- a MEASUREMENT: the Sun is not
#          even partially visible anywhere below `h_max_m` above this facet.
#          The per-step `shadow_depth` series carries it on 0.3% to 27% of
#          facets depending on the epoch.
#
# Both have to become nodata in the main band, because there is no finite number
# that means either one. What must never happen is the third outcome, and both
# regrid paths produce it unless they are stopped:
#
#   * `MeshRegridder.regrid` (IDW) zeroes the weight of every NON-FINITE
#     neighbor, so a `+inf` facet is DROPPED and its pixel is filled from the
#     finite facets around it. "The Sun is never visible below 5 m" comes back
#     as a depth in meters.
#   * `MeshRegridder.regrid_linear` (barycentric, the default here) multiplies
#     `+inf` by a barycentric weight, so `+inf` SPREADS across every triangle
#     that touches such a facet and silently erases the finite neighbors.
#
# The policy below: split the two classes before regridding, regrid an
# indicator for the `+inf` population on the same grid, and stamp the main band
# to nodata wherever that indicator says the facet was unresolved -- so an
# unresolved region can never be reported as a finite value -- then write the
# indicator beside the main raster as `<name>_unresolved.tif` so the second
# meaning survives as its own layer instead of being conflated with "no value".

#: Band metadata naming the face-level census. Written on every band, so a
#: reader holding only the GeoTIFF can tell an absent measurement from a
#: measured absence.
_CENSUS_TAG_NAMES = ('PTM_FACES_TOTAL', 'PTM_FACES_FINITE', 'PTM_FACES_NAN',
                     'PTM_FACES_POS_INF', 'PTM_FACES_NEG_INF',
                     'PTM_SENTINEL_POLICY')

#: Threshold on the regridded 0/1 indicator. A pixel whose interpolated
#: indicator reaches this is stamped nodata in the main band. 0.5 = "the
#: unresolved facets outweigh the resolved ones at this pixel".
UNRESOLVED_STAMP_THRESHOLD = 0.5

#: Suffix of the companion layer.
UNRESOLVED_SUFFIX = '_unresolved'


def unresolved_mask_path(output_path):
    """Where the companion `+inf` layer for ``output_path`` goes."""
    output_path = os.fspath(output_path)
    stem, ext = os.path.splitext(output_path)
    return f"{stem}{UNRESOLVED_SUFFIX}{ext or '.tif'}"


def _census_tags(census):
    parts = ['NaN (not evaluated) is nodata in this band']
    if census.pos_inf:
        parts.append(
            '+inf (umbral above the search ceiling) is nodata here too and is '
            'written separately as the companion '
            + UNRESOLVED_SUFFIX.lstrip('_') + ' layer, stamped out of this '
            'band so it can never read as a finite value')
    if census.neg_inf:
        parts.append(
            '-inf is nodata and has NO meaning in any PTM3D product: it is '
            'counted and dropped, and deliberately NOT placed in the '
            + UNRESOLVED_SUFFIX.lstrip('_') + ' layer, which is a statement '
            'about solar visibility')
    if not (census.pos_inf or census.neg_inf):
        parts.append('this field carries no infinite sentinel')
    policy = '; '.join(parts)
    return {
        'PTM_FACES_TOTAL': str(census.total),
        'PTM_FACES_FINITE': str(census.finite),
        'PTM_FACES_NAN': str(census.nan),
        'PTM_FACES_POS_INF': str(census.pos_inf),
        'PTM_FACES_NEG_INF': str(census.neg_inf),
        'PTM_SENTINEL_POLICY': policy,
    }


def psr_mask_values(direct_solar_max, threshold):
    """Three-state permanent-shadow classification, one value per face.

    ``1.0`` permanently shadowed, ``0.0`` not permanently shadowed, ``NaN``
    NO DATA -- this facet carried no value, so it is neither.

    ⛔ THE DEFECT THIS EXISTS TO STOP. The rule was written
    ``psr = dmax < threshold`` in ``scripts/ptm3d_stats_to_cog.py``, and in
    IEEE arithmetic ``NaN < 1.0`` is ``False``. A facet with no value -- never
    evaluated, or a zero-contribution face in a merged product -- therefore
    came out ``False``: stamped **not permanently shadowed**. That is the exact
    inversion of the quantity the layer exists to report, decided silently, for
    the population least able to contradict it. An unknown may not be forced
    into either class.

    The NaN leaves here as a NaN and :func:`export_field` turns a non-finite
    pixel into the raster's declared nodata -- ``DEFAULT_NODATA``
    (``-3.4028235e+38``) by default, or NaN under the legacy convention.
    Neither can be confused with ``0.0`` or ``1.0``, so a reader that ignores
    the nodata declaration still does not silently receive one of the two
    classes; it receives a number obviously outside a 0/1 mask.

    ⛔ Pass the result to :func:`export_field` as ``unresolved_faces`` as well,
    or ``--interpolation idw`` puts a class back: the inverse-distance path
    zeroes the weight of a non-finite neighbor and fills the pixel from the
    resolved facets around it. Measured on a synthetic polar patch, 100 of 100
    unresolved facets came back holding ``0.0``.

    ⛔ ``+inf`` and ``-inf`` are no data here too, deliberately. Neither is a
    flux, so neither licenses a classification: ``+inf < threshold`` would
    quietly answer "sunlit" for a corrupt facet and ``-inf < threshold``
    "permanently shadowed", each as confidently as a measurement.

    A facet that HAS a value is classified exactly as before, so a source
    carrying no non-finite face produces a bit-identical mask.

    This lives here rather than in the converter script so the rule can be
    imported -- and therefore gated -- with numpy alone, without the raster
    stack the script refuses to start without.
    """
    dmax = np.asarray(direct_solar_max, dtype=np.float64)
    resolved = np.isfinite(dmax)
    out = np.full(dmax.shape, np.nan, dtype=np.float64)
    out[resolved] = dmax[resolved] < threshold
    return out


def _regrid(regridder, face_values, interpolation):
    if interpolation == 'linear':
        return regridder.regrid_linear(face_values)
    if interpolation == 'idw':
        return regridder.regrid(face_values)
    raise ValueError(f"unknown interpolation mode: {interpolation!r}")


# --- Main export -----------------------------------------------------------

# Default nodata for out-of-mesh / invalid pixels. A numeric sentinel
# (float32 -FLT_MAX) rather than NaN: every GIS reliably honors a numeric
# nodata as transparent, whereas float NaN nodata is silently rendered as the
# color-ramp minimum by many viewers (e.g. ice_depth 0 -> spurious "ice at the
# surface" in the rectangular corners outside the circular mesh). Pass
# nodata_value=float('nan') to restore the legacy NaN convention.
DEFAULT_NODATA = float(np.finfo(np.float32).min)  # -3.4028235e+38


def _write_raster(output_path, grid_flipped, regridder, field_name, *,
                  blocksize=512, compress='LZW', overview_resampling='AVERAGE',
                  software='PTM3D', nodata_value=DEFAULT_NODATA,
                  extra_tags=None, band_desc=None, file_desc=None, units=None):
    """Write one already-regridded, north-up band as a COG.

    Split out of :func:`export_field` so the companion `+inf` layer is written
    by exactly the same code path -- same grid, same CRS, same overview chain,
    same nodata -- rather than by a second, drifting writer. `grid_flipped` is
    in GeoTIFF row order (north first) and still carries non-finite values; the
    nodata substitution happens here, AFTER the statistics tags are computed
    from the finite mask, so a finite sentinel can never inflate the
    valid-percent or skew the min/max/mean.

    Returns ``(grid_with_nodata_filled, stats_tags)``.
    """
    extra_tags = dict(extra_tags or {})
    wkt, geo_transform, width, height = _geo_from_regridder(regridder)
    default_units, default_file_desc, default_band_desc = \
        field_metadata(field_name)
    units = default_units if units is None else units
    file_desc = default_file_desc if file_desc is None else file_desc
    band_desc = default_band_desc if band_desc is None else band_desc
    stats_tags = _compute_stats_tags(grid_flipped)
    created = datetime.now().isoformat()

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
            for k, v in extra_tags.items():
                band.SetMetadataItem(str(k), str(v))
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
            leftover = tmp_path + ext
            if os.path.isfile(leftover):
                try:
                    os.remove(leftover)
                except OSError:
                    pass

    return grid_flipped, stats_tags


def export_field(output_path, face_values, regridder, field_name,
                 *, blocksize=512, compress='LZW',
                 overview_resampling='AVERAGE',
                 software='PTM3D',
                 interpolation='linear',
                 nodata_value=DEFAULT_NODATA,
                 extra_tags=None,
                 unresolved_faces=None):
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
        unresolved_faces: optional per-face boolean, ``True`` where the facet
            carries NO value. Same remedy the `+inf` population gets, for a
            population only the CALLER can recognize -- a derived layer such as
            the PSR mask encodes its classes as plain 0/1 floats, so nothing
            here could tell an unresolved facet from a resolved one by looking
            at the values. ⛔ Without it, `interpolation='idw'` silently
            reinstates the defect: `MeshRegridder.regrid` zeroes the weight of
            a non-finite neighbor, so the unresolved facet is DROPPED and its
            pixel is filled from the resolved facets around it. Measured on a
            synthetic polar patch, 100 of 100 unresolved facets came back
            holding a class. ``None`` (the default) leaves every existing
            caller's raster bit-for-bit as it was.

            ⚠️ On the barycentric path this stamp adds nothing, because a
            non-finite vertex already NaNs every Delaunay triangle it touches
            -- which blanks MORE than the unresolved facets cover: measured
            2.45x (12,674 extra nodata pixels for 100 facets covering 5,171)
            on the synthetic polar patch in
            ``src/python/tests/test_psr_mask_nodata.py``. Resolved facets are
            lost that way, but no unresolved one is ever answered, which is the
            safe direction. Bounding it needs a regridder that can drop a
            vertex from the triangulation; that is the open
            ``PTM-REGRID-SENTINEL-UNSAFE``, not something this function can do.

            ⚠️ Deliberately OPT-IN, not defaulted to ``~isfinite(face_values)``.
            Defaulting it would close the same hole for every NaN-carrying
            field (`effective_albedo`, `shadow_depth_*`, `ice_depth`, and any
            `*_avg` from a merged run) -- but only under `idw`, and only by
            CHANGING rasters that have already shipped. That is a product
            decision, not a refactor, and it is not this function's to make.
        extra_tags: extra band metadata, ``{name: str}``. Used for the depth a
            subsurface temperature map holds -- see :func:`depth_band_tags`.
            Written on the BAND, beside the ``STATISTICS_*`` tags, because it
            describes what the pixels ARE. It cannot overwrite one of those:
            they are applied last, and a collision raises rather than
            silently winning, since a statistic quietly replaced by a depth
            would be a wrong number in the same field name.
    Returns ``(grid, stats_tags)``.
    """
    extra_tags = dict(extra_tags or {})
    clash = sorted(set(extra_tags) & set(_STATS_TAG_NAMES + _CENSUS_TAG_NAMES))
    if clash:
        raise ValueError(
            f"extra_tags would overwrite the computed statistics tag(s) "
            f"{clash} on band 1 of {output_path}")
    if not GDAL_AVAILABLE:
        raise RuntimeError("osgeo.gdal is required for COG export")

    # No dtype cast: the regrid paths do their own, and forcing float64 here
    # would copy a 2.4M-facet float32 map for nothing.
    face_values = np.asarray(face_values)
    census = nonfinite_census(face_values)
    extra_tags.update(_census_tags(census))

    if census.pos_inf or census.neg_inf:
        # Split the sentinel classes BEFORE regridding. See the module's
        # "Sentinels" section: left in, an infinite value is either dropped
        # (IDW, so its pixel is filled from finite neighbors) or smeared
        # (barycentric, so it erases them). Neither is acceptable for a value
        # that means "the Sun is not visible anywhere below the ceiling".
        #
        # ⛔ Only `+inf` is the umbral-ceiling sentinel. `-inf` has no meaning
        # in any PTM3D product -- it would be an underflow or a bad fill -- so
        # it is removed from the interpolation and counted, but it is NOT put
        # into the companion layer, whose whole content is a statement about
        # solar visibility. Labeling an underflow as "the Sun was never
        # visible" would be inventing a measurement.
        main_values = np.where(np.isfinite(face_values), face_values, np.nan)
        grid = _regrid(regridder, main_values, interpolation)
        if census.pos_inf:
            indicator = _regrid(regridder,
                                np.isposinf(face_values).astype(np.float64),
                                interpolation)
            stamp = (np.isfinite(indicator)
                     & (indicator >= UNRESOLVED_STAMP_THRESHOLD))
            grid = np.where(stamp, np.nan, grid).astype(np.float32, copy=False)
        else:
            indicator = None
    else:
        indicator = None
        grid = _regrid(regridder, face_values, interpolation)

    if unresolved_faces is not None:
        # Regrid a 0/1 indicator of the caller's unresolved population on the
        # same grid and stamp the band out wherever those facets outweigh the
        # resolved ones -- the same rule, and the same threshold, the `+inf`
        # population gets above. On the barycentric path the non-finite values
        # have already blanked at least this region, so this is decisive only
        # for `idw`; it is applied on both so the two paths cannot disagree
        # about which pixels are answerable.
        flags = np.asarray(unresolved_faces, dtype=bool)
        if flags.shape != np.shape(face_values):
            raise ValueError(
                f"unresolved_faces has shape {flags.shape}, face_values "
                f"{np.shape(face_values)}; they must describe the same faces")
        if flags.any():
            unresolved_grid = _regrid(regridder, flags.astype(np.float64),
                                      interpolation)
            stamp = (np.isfinite(unresolved_grid)
                     & (unresolved_grid >= UNRESOLVED_STAMP_THRESHOLD))
            grid = np.where(stamp, np.nan, grid).astype(np.float32, copy=False)

    grid_flipped = grid[::-1, :]  # north-up GeoTIFF row order

    grid_flipped, stats_tags = _write_raster(
        output_path, grid_flipped, regridder, field_name,
        blocksize=blocksize, compress=compress,
        overview_resampling=overview_resampling, software=software,
        nodata_value=nodata_value, extra_tags=extra_tags)

    if indicator is not None:
        # The second meaning, as its own layer. 1 = the Sun was still hidden
        # above the search ceiling here, 0 = it was not, nodata = outside the
        # mesh. Written through this same function so it is a real COG with the
        # same grid, CRS and overviews, and can be laid straight over the band
        # it explains.
        _write_raster(
            unresolved_mask_path(output_path),
            np.where(np.isfinite(indicator), indicator,
                     np.nan).astype(np.float32)[::-1, :],
            regridder, f"{field_name}{UNRESOLVED_SUFFIX}",
            blocksize=blocksize, compress=compress,
            overview_resampling=overview_resampling, software=software,
            nodata_value=nodata_value,
            band_desc=('Fraction of the facets behind this pixel that were '
                       'still umbral above the search ceiling (1 = all)'),
            file_desc=(f"{field_metadata(field_name)[1]} -- unresolved "
                       f"(umbral above the search ceiling) fraction"),
            units='',
            extra_tags={'PTM_SENTINEL_POLICY':
                        'companion layer for the +inf population of '
                        f'{field_name}; dimensionless fraction in [0, 1]'},
        )

    return grid_flipped, stats_tags
