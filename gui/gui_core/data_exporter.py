"""
Data Exporter for PTM3D GUI
Handles exporting simulation data in GeoTIFF and HDF5 formats.
Includes regridding from unstructured triangular mesh to regular grid.
Automatically selects equirectangular or polar stereographic projection.
"""

from pathlib import Path
from datetime import datetime

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from ._ptm_bootstrap import ensure_ptm_importable

ensure_ptm_importable()

# The mesh/statistics readers, the projection helpers and the regridder used to
# be duplicated here, with a comment on the other copy asking that the two be
# kept byte-for-byte identical by hand. They had already drifted: the shared
# copy gained a `center_lon` argument and a barycentric `regrid_linear` path
# that never reached this file. There is one implementation now.
#
# Several of these names are unused *in this module* and imported to preserve
# its public surface: gui_panels/visualization_panel.py imports BODY_RADII,
# GridResolutionEstimator, detect_body_from_mesh, choose_projection,
# POLAR_LAT_THRESHOLD and _forward_polar_stereo from here.
from ptm.results import (                                          # noqa: E402,F401
    BODY_RADII,
    NICE_RESOLUTIONS,
    POLAR_LAT_THRESHOLD,
    SCIPY_AVAILABLE,
    STAT_LABELS,
    STAT_TYPES,
    GridResolutionEstimator,
    MeshRegridder,
    _forward_polar_stereo,
    _inverse_polar_stereo,
    choose_projection,
    detect_body_from_mesh,
    field_catalog,
    nonfinite_census,
    read_binary_data,
)
from ptm.results import FIELD_INFO as _FIELD_INFO_ASCII            # noqa: E402
from ptm.results import STATISTICS_FIELDS as _STATS_FIELDS_ASCII   # noqa: E402
from ptm.results import detect_available_statistics as _detect_stats_in_dir  # noqa: E402
from ptm.results import field_info as _results_field_info           # noqa: E402
# Either per-step naming generation (BUGS.md PTM-664).
from ptm.readers import frame_path as _frame_path                   # noqa: E402

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

# The Cloud Optimized GeoTIFF writer. NOT a second raster implementation: this
# is the same `src/python/ptm_cog.py` that `scripts/ptm3d_stats_to_cog.py` has
# shipped production COGs with since 2026-05, reached here so the GUI's
# "Export Data" offers the identical product instead of growing its own.
# It needs `osgeo.gdal` for the first pass AND the system `gdaladdo` /
# `gdal_translate` for the overview chain, so availability is not just an
# import.
try:
    import ptm_cog
    from ptm.results import depth_entries as _depth_entries
    COG_AVAILABLE = ptm_cog.GDAL_AVAILABLE
    try:
        ptm_cog._find_gdal_tool('gdaladdo')
        ptm_cog._find_gdal_tool('gdal_translate')
        COG_UNAVAILABLE_REASON = '' if COG_AVAILABLE else 'osgeo.gdal is not importable'
    except RuntimeError as _exc:
        COG_AVAILABLE = False
        COG_UNAVAILABLE_REASON = str(_exc)
except ImportError as _exc:                                       # pragma: no cover
    ptm_cog = None
    COG_AVAILABLE = False
    COG_UNAVAILABLE_REASON = str(_exc)


# Field metadata. Same catalog as ptm.results, with the flux units spelled
# 'W/m²' rather than 'W/m^2': these strings are written into GeoTIFF and HDF5
# tags and shown in the UI, so the spelling is part of this module's output and
# is requested explicitly instead of being carried by a second copy of the dict.
FIELD_INFO = field_catalog(_FIELD_INFO_ASCII, units_style='unicode')
STATISTICS_FIELDS = field_catalog(_STATS_FIELDS_ASCII, units_style='unicode')


def field_info(base_field):
    """Label and units for a statistics field, in this module's unit spelling.

    Wraps `ptm.results.field_info` so a SUBSURFACE TEMPERATURE MAP gets a label
    and a unit here too. Those fields are named for the depth they hold, so no
    fixed catalog can carry them; without this they reached the export dialog
    as a bare `temperature_z0p298m` with `units = ''`, while the raster
    exporter labeled the same file "Temperature at 0.298 m depth" in kelvin.

    The fixed catalog is consulted FIRST because it is the unicode-spelled one
    (`W/m²`, which is what this module writes into GeoTIFF and HDF5 tags). A
    depth map is in kelvin, where the two spellings agree.
    """
    info = STATISTICS_FIELDS.get(base_field)
    if info:
        return dict(info)
    return _results_field_info(base_field)

# BODY_RADII, STAT_TYPES, STAT_LABELS, NICE_RESOLUTIONS, POLAR_LAT_THRESHOLD,
# detect_body_from_mesh, choose_projection, _forward_polar_stereo,
# _inverse_polar_stereo, GridResolutionEstimator, MeshRegridder and
# read_binary_data are imported above from ptm.results and re-exported here,
# because gui_panels/visualization_panel.py imports them from this module.














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
                base_info = field_info(parts[0])
                if base_info:
                    info = {
                        'label': f"{base_info['label']} ({STAT_LABELS[parts[1]]})",
                        'units': base_info.get('units', ''),
                    }
        if not info:
            # Scalar statistics field (e.g. 'ice_depth' — no _avg/_max/_min).
            base_info = field_info(field_name)
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
        info = field_info(base_field)
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
                    info = field_info(base_field)
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
                    info = field_info(base_field)
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
                        formats, per_timestep=False, statistics=None,
                        cog_nodata=None):
    """Write a README.txt describing the exported data and projection.

    Args:
        output_dir: Path to export directory
        regridder: MeshRegridder instance (or None if regridding was skipped)
        fields: List of exported field names
        timesteps: List of timestep dicts with 'time' and 'label' keys
        formats: List of format strings ('geotiff', 'hdf5')
        per_timestep: Whether GeoTIFFs were written per-timestep
        statistics: Optional list of (base_field, stat_type) tuples
        cog_nodata: The nodata value the COGs were written with, when 'cog' is
            in ``formats``. Named here because a raster has ONE nodata value
            and PTM3D carries two distinct sentinels -- see the COG section.
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

    if 'cog' in formats:
        lines.append("* cog/statistics/{field}.tif")
        lines.append("* cog/{field}/{field}_NNNN.tif")
        lines.append("")
        lines.append("  Cloud Optimized GeoTIFFs. Single-band Float32, GDAL COG")
        lines.append("  layout, 512x512 internal tiles, LZW, overview chain")
        lines.append("  resampled by AVERAGE. A COG holds one band, so a time")
        lines.append("  series is one file per timestep.")
        lines.append("")
        nodata_str = ('NaN' if cog_nodata is None or cog_nodata != cog_nodata
                      else f"{cog_nodata:.7e}")
        lines.append(f"  NoData: {nodata_str}")
        lines.append("")
        lines.append("  Sentinels. A raster has ONE nodata value and PTM3D")
        lines.append("  writes two sentinels that do not mean the same thing:")
        lines.append("")
        lines.append("    NaN   this facet was not evaluated / carries no")
        lines.append("          value (for example effective_albedo on a facet")
        lines.append("          that was never lit).")
        lines.append("    +inf  the Sun was still not visible at the top of the")
        lines.append("          shadow-depth search ceiling. This is a")
        lines.append("          measurement, not an absence.")
        lines.append("")
        lines.append("  Both become nodata in the main band, because no finite")
        lines.append("  number means either one. The +inf population is written")
        lines.append("  separately as {field}_unresolved.tif -- the fraction of")
        lines.append("  the facets behind each pixel that were unresolved -- and")
        lines.append("  is stamped out of the main band, so an unresolved area")
        lines.append("  can never be read back as a finite depth. Every band")
        lines.append("  carries the face-level counts as PTM_FACES_TOTAL,")
        lines.append("  PTM_FACES_FINITE, PTM_FACES_NAN, PTM_FACES_POS_INF and")
        lines.append("  PTM_FACES_NEG_INF, with PTM_SENTINEL_POLICY naming the")
        lines.append("  rule that was applied.")
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
            info = field_info(base_field)
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


def detect_available_statistics(results_dir):
    """Which statistics files a RUN directory has, and which hold real data.

    Deliberately not the same signature as
    ``ptm.results.detect_available_statistics``, which takes the
    ``statistics/`` directory itself. This one takes the run directory, because
    that is what ``gui_panels/visualization_panel.py`` passes. Returns the same
    list of ``(base_field, stat_type)`` pairs, with ``stat_type`` None for
    scalar fields such as ``ice_depth``.
    """
    return _detect_stats_in_dir(Path(results_dir) / 'statistics')




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
                - center_lon: central meridian for the output projection, in
                  degrees, or None for the mesh's own midpoint (site-centered)
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
        cog_interpolation = self.config.get('cog_interpolation', 'linear')
        cog_nodata = self.config.get(
            'cog_nodata', ptm_cog.DEFAULT_NODATA if ptm_cog else float('nan'))

        do_geotiff = 'geotiff' in formats and RASTERIO_AVAILABLE
        do_hdf5 = 'hdf5' in formats and H5PY_AVAILABLE
        do_cog = 'cog' in formats and COG_AVAILABLE
        needs_regrid = (do_geotiff or do_hdf5 or do_cog) and SCIPY_AVAILABLE

        # A format that was asked for and cannot be produced has to say so.
        # Silently writing four of the five products the dialog offered is the
        # class of failure this project keeps paying for.
        if 'cog' in formats and not do_cog:
            self.export_finished.emit(
                False, "COG export was requested but is unavailable: "
                       f"{COG_UNAVAILABLE_REASON or 'unknown reason'}")
            return
        if do_cog and not SCIPY_AVAILABLE:
            self.export_finished.emit(
                False, "COG export was requested but scipy is missing, so the "
                       "mesh cannot be regridded onto a raster.")
            return

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
                body_radius=body_radius,
                center_lon=self.config.get('center_lon'),
            )
        current_step += 1
        self.progress_updated.emit(current_step, total_steps)

        if self._cancelled:
            self._cleanup(output_dir)
            self.export_finished.emit(False, "Export canceled")
            return

        # Read and regrid all data, one field at a time
        all_face_data = {}
        all_grid_data = {}
        output_files = []

        for field_name in fields:
            if self._cancelled:
                self._cleanup(output_dir)
                self.export_finished.emit(False, "Export canceled")
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

            # A COG holds ONE band, so a time series is always one file per
            # timestep -- there is no multi-band COG to choose instead.
            if do_cog and regridder:
                cog_dir = output_dir / 'cog' / field_name
                cog_dir.mkdir(parents=True, exist_ok=True)

            for t in range(n_timesteps):
                if self._cancelled:
                    self._cleanup(output_dir)
                    self.export_finished.emit(False, "Export canceled")
                    return

                file_path = _frame_path(field_dir, field_name, t)
                if file_path.exists():
                    data = read_binary_data(str(file_path), n_faces)
                    if data is not None:
                        face_array[t] = data
                        if regridder and (do_geotiff or do_hdf5):
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
                        if do_cog and regridder:
                            # Face values, not the grid above: `export_field`
                            # owns the regrid so it can separate the sentinel
                            # classes BEFORE interpolating them together.
                            census = nonfinite_census(data)
                            if (census.finite == 0 and census.pos_inf == 0
                                    and census.neg_inf == 0):
                                # All-NaN timestep: an all-nodata COG costs two
                                # GDAL subprocess calls and says nothing. Said
                                # out loud rather than skipped silently.
                                self.status_updated.emit(
                                    f"Skipping COG for {field_name} timestep "
                                    f"{t}: no facet carries a value")
                            else:
                                ptm_cog.export_field(
                                    cog_dir / f"{field_name}_{t:04d}.tif",
                                    data, regridder, field_name,
                                    interpolation=cog_interpolation,
                                    nodata_value=cog_nodata,
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
                        file_path = _frame_path(field_dir, field_name, t)
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

            if do_cog and regridder:
                output_files.append(
                    f"{output_dir / 'cog' / field_name}/ ({n_timesteps} COGs)")

            current_step += 1
            self.progress_updated.emit(current_step, total_steps)

        if self._cancelled:
            self._cleanup(output_dir)
            self.export_finished.emit(False, "Export canceled")
            return

        # Export statistics (single-band per-face data)
        all_stat_face_data = {}
        all_stat_grid_data = {}
        cog_stat_names = []

        if statistics:
            stats_dir = results_dir / 'statistics'
            # What the run recorded about each subsurface depth, so a COG band
            # can carry the depth it actually holds rather than the one its
            # file name rounds to. Empty when the run wrote no manifest.
            depth_manifest = _depth_entries(stats_dir) if do_cog else {}

            for base_field, stat_type in statistics:
                if self._cancelled:
                    self._cleanup(output_dir)
                    self.export_finished.emit(False, "Export canceled")
                    return

                # Scalar fields (e.g. ice_depth) carry stat_type=None and
                # live on disk as <base>.bin / <base>.tif. Non-scalar fields
                # use the stat_type-suffixed filenames.
                is_scalar = stat_type is None
                if is_scalar:
                    field_label = field_info(base_field).get('label', base_field)
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
                    field_name_on_disk = (
                        base_field if is_scalar
                        else f"{base_field}_{stat_type}")

                    if do_cog and regridder:
                        census = nonfinite_census(face_data)
                        # ⛔ NOT `census.finite == 0` alone. A field where every
                        # facet is +inf is fully MEASURED -- the Sun was never
                        # visible below the ceiling anywhere -- and skipping it
                        # writes neither the band nor its companion layer, so
                        # the measurement is unrecoverable from the export.
                        # That is the exact conflation the sentinel policy
                        # exists to prevent.
                        if (census.finite == 0 and census.pos_inf == 0
                                and census.neg_inf == 0):
                            self.status_updated.emit(
                                f"Skipping COG for {field_name_on_disk}: "
                                f"no facet carries a value or a sentinel")
                        else:
                            cog_stats_dir = output_dir / 'cog' / 'statistics'
                            cog_stats_dir.mkdir(parents=True, exist_ok=True)
                            ptm_cog.export_field(
                                cog_stats_dir / f"{field_name_on_disk}.tif",
                                face_data, regridder, field_name_on_disk,
                                interpolation=cog_interpolation,
                                nodata_value=cog_nodata,
                                extra_tags=ptm_cog.depth_band_tags(
                                    field_name_on_disk, depth_manifest),
                            )
                            cog_stat_names.append(field_name_on_disk)

                    if regridder and (do_geotiff or do_hdf5):
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

            if cog_stat_names:
                output_files.append(
                    f"{output_dir / 'cog' / 'statistics'}/ "
                    f"({len(cog_stat_names)} COGs)"
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
                    self.export_finished.emit(False, "Export canceled")
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
            statistics=statistics if statistics else None,
            cog_nodata=cog_nodata if do_cog else None,
        )
        output_files.append(str(readme_path))

        # Summary
        file_list = "\n".join(f"  - {f}" for f in output_files)
        if 'vtu' in formats and not do_geotiff and not do_hdf5:
            summary = f"Export complete! (VTU/PVD format)\n\nFiles created:\n{file_list}"
        else:
            proj_name = 'polar stereographic' if (regridder and regridder.projection == 'stere') else 'equirectangular'
            # ⛔ Only on the stereographic branch. `regridder.center_lon` is
            # what the EQUIRECTANGULAR grid was centered on, but both raster
            # writers recompute that meridian from the grid bounds, so quoting
            # it here would state a CRS the files do not carry.
            if regridder is not None and regridder.projection == 'stere':
                proj_name += f", central meridian {regridder.center_lon:.4f} deg"
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
            cog_root = output_dir / "cog"
            if cog_root.exists():
                for f in cog_root.rglob("*.tif"):
                    f.unlink(missing_ok=True)
            vtu_dir = output_dir / "vtu"
            if vtu_dir.exists():
                for f in vtu_dir.glob("*.vtu"):
                    f.unlink(missing_ok=True)
            pvd_path = output_dir / "ptm3d_timeseries.pvd"
            if pvd_path.exists():
                pvd_path.unlink(missing_ok=True)
        except Exception:
            pass
