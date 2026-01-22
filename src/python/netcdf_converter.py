#!/usr/bin/env python3
"""
NetCDF Converter for PTM3D outputs

Converts VTU/PVD statistics (avg, max, min) and optional HDF5 outputs into a single NetCDF file.

Usage examples:
  # Convert statistics PVD to NetCDF
  python netcdf_converter.py --input output/your_sim/vtk/statistics.pvd --output output/your_sim/netcdf/statistics.nc

  # Convert a directory containing statistics VTU files
  python netcdf_converter.py --input output/your_sim/vtk --output output/your_sim/netcdf/statistics.nc --format vtu

  # Convert HDF5 file to NetCDF
  python netcdf_converter.py --input output/your_sim/hdf5/statistics.h5 --output output/your_sim/netcdf/statistics.nc --format hdf5

Notes:
- Requires `netCDF4` Python package. If not installed, the script will exit with a helpful message.
- For VTU inputs, the converter reads CellData arrays (e.g., temperature_avg, illumination_avg, ice_depth) and writes them as NetCDF variables.
- Ice depth is written once and shared across statistics timesteps.
"""

import sys
import os
import argparse
from pathlib import Path


def require_netcdf4():
    try:
        import netCDF4  # noqa: F401
        return True
    except Exception:
        print("ERROR: netCDF4 package is required. Install via: pip install netCDF4")
        return False


def require_vtk():
    try:
        import vtk  # noqa: F401
        from vtk.util import numpy_support  # noqa: F401
        return True
    except ImportError as e:
        print(f"ERROR: VTK package is required for VTU/PVD conversion.")
        print(f"Import error: {e}")
        print(f"Install via: conda install -c conda-forge vtk")
        return False


def require_scipy():
    try:
        import scipy  # noqa: F401
        from scipy.spatial import cKDTree  # noqa: F401
        return True
    except Exception:
        print("ERROR: SciPy is required for gridded output. Install via: pip install scipy")
        return False


def read_pvd_timesteps(pvd_path: Path):
    from xml.etree import ElementTree as ET
    tree = ET.parse(str(pvd_path))
    root = tree.getroot()
    datasets = []
    for ds in root.findall('.//DataSet'):
        file_attr = ds.get('file')
        timestep_attr = ds.get('timestep')
        if file_attr:
            datasets.append((float(timestep_attr) if timestep_attr is not None else None, pvd_path.parent / file_attr))
    return datasets


def read_vtu_cell_arrays(vtu_path: Path):
    import vtk
    from vtk.util import numpy_support
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(vtu_path))
    reader.Update()
    ugrid = reader.GetOutput()
    cell_data = ugrid.GetCellData()
    arrays = {}
    for i in range(cell_data.GetNumberOfArrays()):
        arr = cell_data.GetArray(i)
        name = arr.GetName()
        arrays[name] = numpy_support.vtk_to_numpy(arr)
    return arrays


def read_vtu_geometry(vtu_path: Path):
    """Read geometry from a VTU file.
    Returns dict with points (N,3), faces (M,3) connectivity, and optional lat/lon per face.
    """
    import vtk
    from vtk.util import numpy_support

    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(vtu_path))
    reader.Update()
    ugrid = reader.GetOutput()

    # Points
    pts_vtk = ugrid.GetPoints().GetData()
    points = numpy_support.vtk_to_numpy(pts_vtk)  # shape (N,3)

    # Cells (assume triangles)
    cells_vtk = ugrid.GetCells().GetData()
    cells_np = numpy_support.vtk_to_numpy(cells_vtk)
    # VTK cell array format: [n0, i0_0, i0_1, i0_2, n1, i1_0, ...]
    faces = []
    idx = 0
    while idx < len(cells_np):
        nverts = int(cells_np[idx]); idx += 1
        verts = cells_np[idx:idx+nverts]; idx += nverts
        if nverts == 3:
            faces.append(verts.tolist())
        else:
            # For non-tri cells, take first three vertices to form a triangle (fallback)
            if nverts >= 3:
                faces.append(verts[:3].tolist())
    import numpy as np
    faces = np.array(faces, dtype=np.int32)

    # Optional latitude/longitude from CellData
    lat = lon = None
    cell_data = ugrid.GetCellData()
    for i in range(cell_data.GetNumberOfArrays()):
        arr = cell_data.GetArray(i)
        name = (arr.GetName() or '').lower()
        if name == 'latitude':
            lat = numpy_support.vtk_to_numpy(arr)
        elif name == 'longitude':
            lon = numpy_support.vtk_to_numpy(arr)

    return {
        'points': points,
        'faces': faces,
        'latitude': lat,
        'longitude': lon,
    }


def merge_statistics_arrays(vtu_dir_or_pvd: Path, format_hint: str = 'auto'):
    """Return dict of arrays per stat: {'avg': {...}, 'max': {...}, 'min': {...}} and shared arrays.
    If input is a PVD, it will load referenced VTU files.
    If input is a directory, it will look for statistics_{avg,max,min}.vtu.
    """
    stats = {'avg': {}, 'max': {}, 'min': {}}
    shared = {}

    if format_hint == 'auto':
        if vtu_dir_or_pvd.suffix.lower() == '.pvd':
            format_hint = 'pvd'
        else:
            format_hint = 'vtu'

    if format_hint == 'pvd':
        datasets = read_pvd_timesteps(vtu_dir_or_pvd)
        # Map timesteps 0,1,2 to avg,max,min
        ts_map = {0.0: 'avg', 1.0: 'max', 2.0: 'min'}
        for ts, f in datasets:
            if ts in ts_map:
                arrays = read_vtu_cell_arrays(f)
                stats[ts_map[ts]] = arrays
    elif format_hint == 'vtu':
        base = vtu_dir_or_pvd
        for st in ['avg', 'max', 'min']:
            f = base / f'statistics_{st}.vtu'
            if f.exists():
                arrays = read_vtu_cell_arrays(f)
                stats[st] = arrays
            else:
                print(f"WARNING: Missing {f}")
    else:
        raise ValueError("Unsupported format for VTU merge")

    # Identify shared arrays (present in all stats and identical)
    if stats['avg'] and stats['max'] and stats['min']:
        keys = set(stats['avg'].keys()) & set(stats['max'].keys()) & set(stats['min'].keys())
        for k in keys:
            a0, a1, a2 = stats['avg'][k], stats['max'][k], stats['min'][k]
            try:
                import numpy as np
                if np.array_equal(a0, a1) and np.array_equal(a0, a2):
                    shared[k] = a0
                    # Also keep per-stat versions for completeness
            except Exception:
                pass
    return stats, shared


def write_netcdf_from_vtu(stats, shared, output_nc: Path, mode: str = 'grid'):
    import numpy as np
    import netCDF4 as nc
    from pathlib import Path as _Path
    # griddata no longer used; we use cKDTree-based IDW for speed
    
    # Try to locate a VTU file to read geometry from
    geometry_source = None
    geom_hint = os.environ.get('NETCDF_GEOM_SRC')
    if geom_hint:
        candidate = _Path(geom_hint)
        if candidate.exists():
            geometry_source = candidate

    # Determine face count
    example = next(iter(stats['avg'].values())) if stats['avg'] else None
    if example is None:
        raise RuntimeError("No arrays found in 'avg' statistics for VTU conversion")
    n_faces = example.shape[0]

    ds = nc.Dataset(str(output_nc), 'w')
    
    # Original unstructured mesh dimension
    ds.createDimension('nele', n_faces)
    ds.createDimension('stat', 3)

    # Attempt to read geometry
    points = faces = lat = lon = None
    try:
        base = _Path(output_nc).parent.parent
        candidates = []
        if geometry_source:
            candidates.append(geometry_source)
        candidates += [
            base / 'vtk' / 'statistics_avg.vtu',
            base / 'vtk' / 'statistics_max.vtu',
            base / 'vtk' / 'statistics_min.vtu',
        ]
        for c in candidates:
            if c and _Path(c).exists():
                geom = read_vtu_geometry(_Path(c))
                points, faces, lat, lon = geom['points'], geom['faces'], geom['latitude'], geom['longitude']
                break
    except Exception as e:
        print(f"WARNING: Could not read geometry: {e}")
        points = faces = lat = lon = None

    # Compute lat/lon from centroids if not already present; also compute elevation
    has_geometry = points is not None and faces is not None and faces.shape[0] == n_faces
    elevation_face = None
    reference_radius = None
    if has_geometry:
        fverts = faces.astype(np.int64)
        centroids = points[fverts].mean(axis=1)  # (nele, 3)
        x, y, z = centroids[:, 0], centroids[:, 1], centroids[:, 2]
        if lat is None or lon is None:
            lon = np.degrees(np.arctan2(y, x)).astype('float64')
            lat = np.degrees(np.arctan2(z, np.sqrt(x * x + y * y))).astype('float64')
        # Compute elevation relative to median node radius
        rn = np.sqrt(points[:, 0]**2 + points[:, 1]**2 + points[:, 2]**2)
        reference_radius = float(np.nanmedian(rn))
        rc = np.sqrt(x**2 + y**2 + z**2)
        elevation_face = (rc - reference_radius).astype('float64')

    # ========== PART 1: UGRID Unstructured Mesh (for advanced tools) ==========
    write_mesh = has_geometry and (mode in ('mesh', 'both'))
    if write_mesh:
        ds.createDimension('nnode', points.shape[0])
        ds.createDimension('nvertex', 3)

        # Node coordinates
        node_x = ds.createVariable('node_x', 'f8', ('nnode',))
        node_x.units = 'meters'
        node_x.long_name = 'node x-coordinate (Cartesian ECEF)'
        node_x[:] = points[:, 0].astype('float64')
        
        node_y = ds.createVariable('node_y', 'f8', ('nnode',))
        node_y.units = 'meters'
        node_y.long_name = 'node y-coordinate (Cartesian ECEF)'
        node_y[:] = points[:, 1].astype('float64')
        
        node_z = ds.createVariable('node_z', 'f8', ('nnode',))
        node_z.units = 'meters'
        node_z.long_name = 'node z-coordinate (Cartesian ECEF)'
        node_z[:] = points[:, 2].astype('float64')

        # Face connectivity
        conn = ds.createVariable('face_nodes', 'i4', ('nele', 'nvertex'))
        conn.long_name = 'face vertex indices (0-based)'
        conn.cf_role = 'face_node_connectivity'
        conn.start_index = 0
        conn[:, :] = faces.astype('int32')

        # Face centroid coordinates (1D arrays for unstructured mesh)
        face_lat = ds.createVariable('face_lat', 'f8', ('nele',))
        face_lat.units = 'degrees_north'
        face_lat.standard_name = 'latitude'
        face_lat.long_name = 'latitude of face centroid'
        face_lat.valid_min = -90.0
        face_lat.valid_max = 90.0
        face_lat[:] = lat.astype('float64')

        face_lon = ds.createVariable('face_lon', 'f8', ('nele',))
        face_lon.units = 'degrees_east'
        face_lon.standard_name = 'longitude'
        face_lon.long_name = 'longitude of face centroid'
        face_lon.valid_min = -180.0
        face_lon.valid_max = 180.0
        face_lon[:] = lon.astype('float64')

        # Face elevation (centroid-based)
        if elevation_face is not None:
            felev = ds.createVariable('elevation', 'f8', ('nele',))
            felev.units = 'meters'
            felev.standard_name = 'surface_altitude'
            felev.long_name = 'face centroid elevation relative to reference sphere'
            felev[:] = elevation_face.astype('float64')
            _attach_ugrid_metadata(felev)

        # Mesh topology variable
        mesh = ds.createVariable('mesh', 'i4', ())
        mesh.cf_role = 'mesh_topology'
        mesh.long_name = 'Topology data for unstructured triangular mesh'
        mesh.topology_dimension = 2
        mesh.node_coordinates = 'node_x node_y node_z'
        mesh.face_node_connectivity = 'face_nodes'
        mesh.face_coordinates = 'face_lat face_lon'

    # ========== PART 2: Gridded Data (for Panoply) ==========
    gridded_vars = {}
    write_grid = has_geometry and (mode in ('grid', 'both'))
    if write_grid:
        # Determine grid resolution based on data extent
        lat_min, lat_max = float(np.nanmin(lat)), float(np.nanmax(lat))
        lon_min, lon_max = float(np.nanmin(lon)), float(np.nanmax(lon))
        
        # For polar regions, use finer resolution (adjust as needed)
        # Target ~1000x1000 grid for balance between resolution and file size
        n_lat = min(1000, int((lat_max - lat_min) * 100))  # ~0.01 deg resolution
        n_lon = min(1000, int((lon_max - lon_min) * 100))
        n_lat = max(100, n_lat)  # Minimum 100 points
        n_lon = max(100, n_lon)
        
        # Create regular grid
        lat_grid = np.linspace(lat_min, lat_max, n_lat)
        lon_grid = np.linspace(lon_min, lon_max, n_lon)
        
        # Create grid dimensions
        ds.createDimension('latitude', n_lat)
        ds.createDimension('longitude', n_lon)
        
        # Create coordinate variables (Panoply requires these exact names)
        lat_var = ds.createVariable('latitude', 'f8', ('latitude',))
        lat_var.units = 'degrees_north'
        lat_var.standard_name = 'latitude'
        lat_var.long_name = 'latitude'
        lat_var.axis = 'Y'
        lat_var[:] = lat_grid
        
        lon_var = ds.createVariable('longitude', 'f8', ('longitude',))
        lon_var.units = 'degrees_east'
        lon_var.standard_name = 'longitude'
        lon_var.long_name = 'longitude'
        lon_var.axis = 'X'
        lon_var[:] = lon_grid
        
        # Create 2D meshgrid for interpolation
        LON, LAT = np.meshgrid(lon_grid, lat_grid)
        grid_points = np.column_stack([LON.ravel(), LAT.ravel()])
        
        print(f"Regridding {n_faces} faces to {n_lat}x{n_lon} grid...")
        print(f"Building spatial index for fast k=3 IDW interpolation...")
        
        # OPTIMIZATION: Build KDTree once for all variables
        from scipy.spatial import cKDTree
        source_points = np.column_stack([lon, lat])
        
        # Filter valid points (non-NaN coordinates)
        valid_coord_mask = np.isfinite(lon) & np.isfinite(lat)
        if not np.any(valid_coord_mask):
            print("ERROR: No valid coordinates found in mesh")
            write_grid = False
        else:
            # Build spatial index on valid source points
            tree = cKDTree(source_points[valid_coord_mask])
            
            # Query k=3 nearest neighbors for all grid points at once (vectorized)
            # distances, indices shapes: (n_lat*n_lon, k)
            distances, indices = tree.query(grid_points, k=3, workers=-1)
            valid_indices = np.where(valid_coord_mask)[0]  # map from tree-space to face indices
            neighbor_faces = valid_indices[indices]
            maxd = np.nanmax(distances)
            print(f"Spatial index built. Max distance to nearest face: {maxd:.6f} degrees")

            def regrid_variable(data_1d):
                """Interpolate to 2D grid using k=3 inverse-distance weighting (vectorized)."""
                # Gather neighbor values (n_points, k)
                vals = data_1d[neighbor_faces]
                # Mask invalid neighbor values
                valid_vals = np.isfinite(vals)

                # Handle exact matches: where any distance==0, pick that neighbor's value
                zero_mask = (distances == 0.0)
                if np.any(zero_mask):
                    # For rows with any zero distance, choose the first zero-distance neighbor
                    choose = zero_mask.argmax(axis=1)
                    has_zero = zero_mask.any(axis=1)
                    out = np.full(distances.shape[0], np.nan, dtype=np.float32)
                    out[has_zero] = vals[np.arange(vals.shape[0])[has_zero], choose[has_zero]].astype(np.float32)
                    # For remaining rows, perform IDW on non-zero distances
                    rem = ~has_zero
                    if np.any(rem):
                        d = distances[rem]
                        v = vals[rem]
                        vv = valid_vals[rem]
                        w = 1.0 / (d**2 + 1e-12)
                        w[~vv] = 0.0
                        ws = w.sum(axis=1)
                        # Avoid division by zero
                        good = ws > 0
                        out_rem = np.full(rem.sum(), np.nan, dtype=np.float32)
                        if np.any(good):
                            wn = (w[good] / ws[good][:, None])
                            out_rem[good] = np.nansum(wn * v[good], axis=1).astype(np.float32)
                        out[rem] = out_rem
                else:
                    # Pure IDW path
                    w = 1.0 / (distances**2 + 1e-12)
                    w[~valid_vals] = 0.0
                    ws = w.sum(axis=1)
                    out = np.full(distances.shape[0], np.nan, dtype=np.float32)
                    good = ws > 0
                    if np.any(good):
                        wn = (w[good] / ws[good][:, None])
                        out[good] = np.nansum(wn * vals[good], axis=1).astype(np.float32)

                return out.reshape((n_lat, n_lon))

            # Gridded elevation variable
            if elevation_face is not None:
                vge = ds.createVariable('elevation', 'f4', ('latitude', 'longitude'), fill_value=np.nan)
                vge.units = 'meters'
                vge.standard_name = 'surface_altitude'
                vge.long_name = 'elevation relative to reference sphere'
                vge.coordinates = 'latitude longitude'
                vge.grid_mapping = 'crs'
                vge.setncattr('missing_value', np.float32(np.nan))
                vge[:, :] = regrid_variable(elevation_face)
        
        # Store gridded versions for later writing
        gridded_vars = {}

    # ========== Global Attributes ==========
    # Global attributes
    if write_mesh and write_grid:
        ds.Conventions = 'CF-1.8, UGRID-1.0'
        ds.featureType = 'mesh'
    elif write_mesh:
        ds.Conventions = 'CF-1.8, UGRID-1.0'
        ds.featureType = 'mesh'
    else:
        ds.Conventions = 'CF-1.8'
    ds.setncattr('mesh_topology', 'unstructured_triangular_mesh')
    ds.setncattr('geometry_source', str(geometry_source) if geometry_source else 'auto')
    if has_geometry:
        ds.setncattr('geospatial_lat_min', float(np.nanmin(lat)))
        ds.setncattr('geospatial_lat_max', float(np.nanmax(lat)))
        ds.setncattr('geospatial_lon_min', float(np.nanmin(lon)))
        ds.setncattr('geospatial_lon_max', float(np.nanmax(lon)))
        ds.setncattr('geospatial_vertical_min', 0.0)
        ds.setncattr('geospatial_vertical_max', 0.0)
        ds.setncattr('geospatial_vertical_positive', 'up')

    # Stat labels
    stat_labels = ds.createVariable('stat_labels', 'S1', ('stat',))
    stat_labels.setncattr('description', 'Statistics labels for index 0..2')
    stat_labels.long_name = 'avg,max,min'

    # ========== Write Data Variables ==========
    
    # Helper for unstructured mesh variables
    def _attach_ugrid_metadata(var):
        if write_mesh:
            var.coordinates = 'face_lat face_lon'
            var.mesh = 'mesh'
            var.location = 'face'
    
    # Helper for gridded variables
    def _create_vars(name, data_1d, stat_suffix=''):
        """Create variables depending on mode: mesh, grid, or both.
        Returns the grid variable when created, else None."""
        # Skip coordinate variables to avoid name conflicts
        if name.lower() in ('latitude', 'longitude', 'lat', 'lon'):
            return None

        grid_var = None

        # Mesh variable naming: if only mesh, keep clean name; if both, append _mesh
        if write_mesh:
            mesh_vname = f'{name}{stat_suffix}' if (mode == 'mesh') else f'{name}{stat_suffix}_mesh'
            v_ugrid = ds.createVariable(mesh_vname, 'f4', ('nele',), fill_value=np.nan)
            v_ugrid[:] = data_1d.astype('float32')
            v_ugrid.setncattr('missing_value', np.float32(np.nan))
            _attach_ugrid_metadata(v_ugrid)

        if write_grid:
            grid_name = f'{name}{stat_suffix}' if stat_suffix else name
            v_grid = ds.createVariable(grid_name, 'f4', ('latitude', 'longitude'), fill_value=np.nan)
            v_grid.coordinates = 'latitude longitude'
            v_grid.grid_mapping = 'crs'
            v_grid.setncattr('missing_value', np.float32(np.nan))
            gridded_data = regrid_variable(data_1d)
            v_grid[:, :] = gridded_data
            grid_var = v_grid

        return grid_var

    # Write shared arrays (constant across stats)
    print("Writing shared variables...")
    for name, arr in shared.items():
        var = _create_vars(name, arr)
        if var:
            var.setncattr('shared_across_stats', 'true')

    # Write per-stat arrays
    print("Writing per-statistic variables...")
    for st_i, st_name in enumerate(['avg', 'max', 'min']):
        for name, arr in stats[st_name].items():
            if name in shared:
                continue  # Already written as shared variable
            # Use stat suffix in variable name to ensure uniqueness
            var = _create_vars(name, arr, stat_suffix=f'_{st_name}')
            if var:
                var.setncattr('stat_index', st_i)
                var.setncattr('statistic', st_name)
    
    # Add coordinate reference system (optional, for GIS tools)
    if write_grid and 'latitude' in ds.dimensions:
        crs = ds.createVariable('crs', 'i4', ())
        crs.grid_mapping_name = 'latitude_longitude'
        crs.longitude_of_prime_meridian = 0.0
        if reference_radius is not None:
            crs.semi_major_axis = float(reference_radius)
        else:
            crs.semi_major_axis = 1737400.0
        crs.inverse_flattening = 0.0  # Sphere
        crs.spatial_ref = 'GEOGCS["Moon 2000",DATUM["D_Moon_2000",SPHEROID["Moon_2000_IAU_IAG",1737400.0,0.0]],PRIMEM["Greenwich",0],UNIT["Decimal_Degree",0.0174532925199433]]'

    ds.sync()
    ds.close()
    # Update geospatial vertical range based on elevation
    if elevation_face is not None:
        try:
            ds.setncattr('geospatial_vertical_min', float(np.nanmin(elevation_face)))
            ds.setncattr('geospatial_vertical_max', float(np.nanmax(elevation_face)))
            ds.setncattr('geospatial_vertical_positive', 'up')
            ds.setncattr('reference_radius_m', float(reference_radius) if reference_radius is not None else np.nan)
        except Exception:
            pass

    if write_mesh and write_grid:
        print(f"Wrote NetCDF (mesh + grid): {output_nc}")
        print(f"  - Mesh faces: {n_faces}")
        print(f"  - Grid size: {n_lat} x {n_lon}")
    elif write_mesh:
        print(f"Wrote NetCDF (mesh only): {output_nc}")
        print(f"  - Mesh faces: {n_faces}")
    elif write_grid:
        print(f"Wrote NetCDF (grid only): {output_nc}")
        print(f"  - Grid size: {n_lat} x {n_lon}")


def convert_hdf5_to_netcdf(h5_path: Path, output_nc: Path):
    import h5py
    import netCDF4 as nc
    import numpy as np

    with h5py.File(str(h5_path), 'r') as hf:
        ds = nc.Dataset(str(output_nc), 'w')
        # Simple heuristic: assume datasets are 1D face arrays or 2D (face, time/stat)
        # Create dimensions on the fly
        dims_created = {}

        def ensure_dim(name, size):
            if name not in dims_created:
                ds.createDimension(name, size)
                dims_created[name] = size

        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                data = obj[...]
                # Determine dims
                if data.ndim == 1:
                    ensure_dim('face', data.shape[0])
                    v = ds.createVariable(name.replace('/', '_'), 'f4', ('face',))
                    v[:] = data.astype('float32')
                elif data.ndim == 2:
                    ensure_dim('face', data.shape[0])
                    ensure_dim('axis1', data.shape[1])
                    v = ds.createVariable(name.replace('/', '_'), 'f4', ('face', 'axis1'))
                    v[:] = data.astype('float32')
                else:
                    # Fallback: store as raw bytes string
                    v = ds.createVariable(name.replace('/', '_'), 'f8', tuple())
                    v.assignValue(np.nan)

        hf.visititems(visit)
        ds.sync()
        ds.close()
    print(f"Wrote NetCDF: {output_nc}")


def main():
    ap = argparse.ArgumentParser(description='Convert PTM3D VTU/PVD/HDF5 outputs to NetCDF')
    ap.add_argument('--input', required=True, help='Input file or directory (PVD, VTU dir, or HDF5)')
    ap.add_argument('--output', required=True, help='Output NetCDF file path')
    ap.add_argument('--format', choices=['auto', 'pvd', 'vtu', 'hdf5'], default='auto', help='Input format')
    ap.add_argument('--mode', choices=['grid', 'mesh', 'both'], default='grid',
                    help='Output type: grid (Panoply-friendly), mesh (UGRID), or both (writes two files)')
    args = ap.parse_args()

    if not require_netcdf4():
        sys.exit(1)

    input_path = Path(args.input)
    output_nc = Path(args.output)
    output_nc.parent.mkdir(parents=True, exist_ok=True)

    fmt = args.format
    if fmt == 'auto':
        if input_path.is_file():
            if input_path.suffix.lower() == '.pvd':
                fmt = 'pvd'
            elif input_path.suffix.lower() in ('.h5', '.hdf5'):
                fmt = 'hdf5'
            else:
                print('ERROR: Unknown file type for auto format')
                sys.exit(2)
        else:
            fmt = 'vtu'

    # Robust fallback handling for mis-specified PVD path (e.g., root passed instead of vtk/statistics.pvd)
    if fmt == 'pvd':
        if not (input_path.is_file() and input_path.exists()):
            # Attempt to auto-locate statistics.pvd
            search_roots = []
            if input_path.is_dir():
                search_roots.append(input_path)
            search_roots.append(input_path.parent)
            candidates = []
            for r in search_roots:
                if r:
                    candidates.append(r / 'vtk' / 'statistics.pvd')
                    candidates.append(r / 'statistics.pvd')
            found = None
            for c in candidates:
                if c.exists():
                    found = c
                    break
            if found:
                print(f"INFO: Auto-located PVD file at: {found}")
                input_path = found
            else:
                # Try switching to VTU directory mode if statistics_{avg,max,min}.vtu exist
                possible_dirs = []
                if input_path.is_dir():
                    possible_dirs.append(input_path / 'vtk')
                    possible_dirs.append(input_path)
                else:
                    possible_dirs.append(input_path.parent / 'vtk')
                    possible_dirs.append(input_path.parent)
                switched = None
                for d in possible_dirs:
                    if d.exists() and (d / 'statistics_avg.vtu').exists():
                        fmt = 'vtu'
                        input_path = d
                        switched = d
                        print(f"INFO: Falling back to VTU directory mode at: {d}")
                        break
                if not switched and not found:
                    print(f"ERROR: PVD file not found and no VTU statistics directory discovered.")
                    print("Searched candidates:")
                    for c in candidates:
                        print(f"  {c}")
                    sys.exit(5)

    if fmt in ('pvd', 'vtu'):
        if not require_vtk():
            sys.exit(1)
        stats, shared = merge_statistics_arrays(input_path, format_hint=fmt)
        if not stats['avg']:
            print('ERROR: No statistics arrays found for VTU/PVD conversion')
            sys.exit(3)
        # Validate SciPy if grid output requested
        if args.mode in ('grid', 'both'):
            if not require_scipy():
                sys.exit(1)
        if args.mode == 'both':
            # Write separate files to keep sizes manageable
            write_netcdf_from_vtu(stats, shared, output_nc, mode='grid')
            mesh_out = output_nc.with_name(output_nc.stem + '_mesh.nc')
            write_netcdf_from_vtu(stats, shared, mesh_out, mode='mesh')
        else:
            write_netcdf_from_vtu(stats, shared, output_nc, mode=args.mode)
    elif fmt == 'hdf5':
        convert_hdf5_to_netcdf(input_path, output_nc)
    else:
        print('ERROR: Unsupported format')
        sys.exit(4)


if __name__ == '__main__':
    main()
