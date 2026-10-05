"""PTM3D surface temperature adapter for the O&S ice shell model.

Reads PTM3D binary temperature output and provides the same interface
as OSSurfaceTemperature, allowing drop-in replacement in IceShellModel.
"""

import os
import struct
import numpy as np


class PTM3DSurfaceTemperature:
    """Adapter using PTM3D surface temperatures in the O&S shell model.

    The per-face temperature is the ARITHMETIC time mean of the snapshots
    (see from_binary_files); a ``statistics/temperature_avg.bin`` product
    is the same statistic.

    Provides the same interface as OSSurfaceTemperature:
      - temperature(theta)     -> zonally-averaged T_s at colatitude theta
      - temperature_2d(theta, lam)  -> interpolated onto regular grid
    """

    def __init__(self):
        self._face_lat = None    # degrees, [-90, 90]
        self._face_lon = None    # degrees, [0, 360]
        self._face_temp = None   # Kelvin, per face (diurnal mean)
        self._face_colat = None  # radians, [0, pi]

    @classmethod
    def from_binary_files(cls, temperature_dir, mesh_file=None,
                          n_timesteps=None):
        """Load from PTM3D binary output directory.

        Parameters
        ----------
        temperature_dir : str
            Directory containing temperature_XXXX.bin files.
        mesh_file : str, optional
            PLY mesh file with face lat/lon. If None, looks for mesh.ply
            in the parent directory of temperature_dir.
        n_timesteps : int, optional
            Number of timesteps to average. If None, uses all found.

        Returns
        -------
        PTM3DSurfaceTemperature
            Initialized adapter.
        """
        obj = cls()

        # Load mesh for lat/lon
        if mesh_file is None:
            parent = os.path.dirname(os.path.abspath(temperature_dir))
            mesh_file = os.path.join(parent, 'mesh.ply')

        obj._face_lat, obj._face_lon = _load_ply_face_latlon(mesh_file)
        obj._face_colat = np.radians(90.0 - obj._face_lat)
        n_faces = len(obj._face_lat)

        # Load and average temperature files
        temp_files = sorted([
            f for f in os.listdir(temperature_dir)
            if f.startswith('temperature_') and f.endswith('.bin')
        ])

        if n_timesteps is not None:
            temp_files = temp_files[:n_timesteps]

        if not temp_files:
            raise FileNotFoundError(
                f"No temperature_*.bin files found in {temperature_dir}")

        # Accumulate the ARITHMETIC time mean <T>. This is the shell's top
        # boundary condition by project decision (2026-10-04): the conductive
        # shell responds to the time-mean temperature, not to the
        # radiative-equivalent (<T^4>)^(1/4), which is warmer and shifts
        # t20 + 2 t22 by ~0.28 km (physics review, 2026-10-03).
        temp_sum = np.zeros(n_faces, dtype=np.float64)
        count = 0
        unreadable = []

        for fname in temp_files:
            fpath = os.path.join(temperature_dir, fname)
            data = _read_binary_timestep(fpath, n_faces)
            if data is not None:
                temp_sum += data.astype(np.float64)
                count += 1
            else:
                unreadable.append(fname)

        if count == 0:
            raise RuntimeError("No valid temperature files could be read")
        if unreadable:
            print(f"  WARNING: {len(unreadable)} of {len(temp_files)} temperature "
                  f"files unreadable or wrong size and EXCLUDED from the mean "
                  f"(first: {unreadable[0]})")

        obj._face_temp = (temp_sum / count).astype(np.float64)
        print(f"Loaded {count} timesteps, {n_faces} faces")
        print(f"  T range: {obj._face_temp.min():.1f} - {obj._face_temp.max():.1f} K")

        return obj

    @classmethod
    def from_arrays(cls, temperatures, latitudes, longitudes):
        """Load from numpy arrays directly.

        Parameters
        ----------
        temperatures : ndarray, shape (n_faces,)
            Surface temperature per face [K].
        latitudes : ndarray, shape (n_faces,)
            Face center latitudes [degrees, -90 to 90].
        longitudes : ndarray, shape (n_faces,)
            Face center longitudes [degrees, 0 to 360].
        """
        obj = cls()
        obj._face_temp = np.asarray(temperatures, dtype=np.float64)
        obj._face_lat = np.asarray(latitudes, dtype=np.float64)
        obj._face_lon = np.asarray(longitudes, dtype=np.float64)
        obj._face_colat = np.radians(90.0 - obj._face_lat)
        return obj

    def symmetrize_ns(self):
        """Remove north-south asymmetry by averaging hemisphere pairs.

        For each face at colatitude theta, finds the nearest face at
        colatitude (pi - theta) with the same longitude and averages
        their temperatures.  This corrects for seasonal effects that
        should cancel over a full orbit.

        Modifies the object in place and invalidates cached lookups.
        """
        from scipy.spatial import cKDTree

        # Build Cartesian coords on unit sphere for mirror-matching
        colat = self._face_colat
        lon_rad = np.radians(self._face_lon)
        x = np.sin(colat) * np.cos(lon_rad)
        y = np.sin(colat) * np.sin(lon_rad)
        z = np.cos(colat)

        # Mirror points: reflect z -> -z  (colat -> pi - colat)
        x_m, y_m, z_m = x, y, -z
        tree = cKDTree(np.column_stack([x, y, z]))
        _, idx = tree.query(np.column_stack([x_m, y_m, z_m]))

        # Average each face with its mirror partner
        self._face_temp = 0.5 * (self._face_temp + self._face_temp[idx])

        # Invalidate cached lookups
        if hasattr(self, '_zonal_theta'):
            del self._zonal_theta
            del self._zonal_temp

        T = self._face_temp
        print(f"  Symmetrized: T range {T.min():.1f} - {T.max():.1f} K")

    def temperature(self, theta):
        """Zonally-averaged temperature at colatitude theta.

        Uses a pre-computed interpolation table for efficiency with
        multi-dimensional input arrays.

        Parameters
        ----------
        theta : float or ndarray
            Colatitude [rad], range [0, pi]. Can be any shape.

        Returns
        -------
        float or ndarray
            Zonally-averaged surface temperature [K], same shape as input.
        """
        if not hasattr(self, '_zonal_theta'):
            self._build_zonal_lookup()

        theta_arr = np.asarray(theta, dtype=float)
        scalar = theta_arr.ndim == 0
        theta_flat = np.atleast_1d(theta_arr).ravel()

        result = np.interp(theta_flat, self._zonal_theta, self._zonal_temp)
        result = result.reshape(theta_arr.shape) if not scalar else float(result[0])
        return result

    def _build_zonal_lookup(self, n_bins=360):
        """Pre-compute zonally-averaged temperature on a regular grid."""
        bin_edges = np.linspace(0, np.pi, n_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_temps = np.zeros(n_bins)

        for i in range(n_bins):
            mask = ((self._face_colat >= bin_edges[i]) &
                    (self._face_colat < bin_edges[i + 1]))
            if np.any(mask):
                bin_temps[i] = np.mean(self._face_temp[mask])
            else:
                bin_temps[i] = np.nan

        valid = ~np.isnan(bin_temps)
        if not np.all(valid) and np.any(valid):
            bin_temps[~valid] = np.interp(
                bin_centers[~valid], bin_centers[valid], bin_temps[valid])

        self._zonal_theta = bin_centers
        self._zonal_temp = bin_temps

    def temperature_2d(self, theta, lam, seam_pad_deg=30.0):
        """Interpolate the per-face temperature onto a (theta, lam) grid.

        Linear (Delaunay) interpolation in (colatitude, longitude) is not
        periodic in longitude: without help, grid points within about one
        face of lam = +-180 deg are interpolated across the whole sphere or
        fall back to nearest neighbor (the "seam" artifact, about -0.006 km
        in b11c; physics review 2026-10-03). Faces within ``seam_pad_deg`` of
        the seam are therefore replicated at lon -/+ 360 deg before the
        triangulation, so every caller gets a seam-free field.

        Points still outside the convex hull (only the polar caps beyond the
        outermost face centers) are filled by nearest neighbor; the count is
        reported, never silent.

        Parameters
        ----------
        theta : ndarray, shape (n_theta,)
            Colatitude [rad].
        lam : ndarray, shape (n_lam,)
            Longitude [rad].
        seam_pad_deg : float
            Width of the periodic padding band on each side of the seam.

        Returns
        -------
        ndarray, shape (n_theta, n_lam)
            Interpolated surface temperature [K].
        """
        from scipy.interpolate import griddata

        face_colat = self._face_colat
        face_lon = ((np.radians(self._face_lon) + np.pi) % (2 * np.pi)) - np.pi
        T = self._face_temp
        pad = np.radians(seam_pad_deg)
        east = face_lon > np.pi - pad      # replicate at lon - 2 pi
        west = face_lon < -np.pi + pad     # replicate at lon + 2 pi
        pts_colat = np.concatenate([face_colat, face_colat[east], face_colat[west]])
        pts_lon = np.concatenate([face_lon, face_lon[east] - 2 * np.pi,
                                  face_lon[west] + 2 * np.pi])
        vals = np.concatenate([T, T[east], T[west]])

        THETA, LAM = np.meshgrid(theta, lam, indexing='ij')
        grid_T = griddata((pts_colat, pts_lon), vals, (THETA, LAM),
                          method='linear', fill_value=np.nan)
        nan_mask = np.isnan(grid_T)
        if np.any(nan_mask):
            grid_T_nn = griddata((pts_colat, pts_lon), vals, (THETA, LAM),
                                 method='nearest')
            grid_T[nan_mask] = grid_T_nn[nan_mask]
            rows = np.unique(np.where(nan_mask)[0])
            print(f"  temperature_2d: {int(nan_mask.sum())} grid points outside the "
                  f"face hull filled by nearest neighbor (rows {rows.min()}-{rows.max()} "
                  f"of {len(theta)})")
        return grid_T

    def temperature_per_face(self):
        """Return raw per-face data.

        Returns
        -------
        dict with keys: 'temperature', 'latitude', 'longitude', 'colatitude'
        """
        return {
            'temperature': self._face_temp.copy(),
            'latitude': self._face_lat.copy(),
            'longitude': self._face_lon.copy(),
            'colatitude': self._face_colat.copy(),
        }


def _read_binary_timestep(file_path, expected_count):
    """Read a PTM3D binary timestep file.

    Format: [uint32 num_faces][float32 data[num_faces]]
    """
    try:
        fsize = os.path.getsize(file_path)
        if fsize < 4:
            return None

        with open(file_path, 'rb') as f:
            count = struct.unpack('<I', f.read(4))[0]
            if count > 0 and (fsize - 4) >= count * 4:
                data = np.frombuffer(f.read(count * 4), dtype='<f4', count=count)
                if data.size == expected_count:
                    return data
            # Fallback: headerless
            if fsize == expected_count * 4:
                f.seek(0)
                return np.frombuffer(f.read(), dtype='<f4', count=expected_count)
    except Exception:
        pass
    return None


def _load_ply_face_latlon(ply_path):
    """Load face latitude/longitude from a PTM3D PLY mesh file.

    The PLY format stores face properties including latitude and longitude.

    Returns
    -------
    latitudes : ndarray
        Face center latitudes [degrees].
    longitudes : ndarray
        Face center longitudes [degrees].
    """
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"Mesh file not found: {ply_path}")

    latitudes = []
    longitudes = []

    with open(ply_path, 'rb') as f:
        # Parse header
        header_lines = []
        n_vertices = 0
        n_faces = 0
        is_binary = False
        lat_idx = -1
        lon_idx = -1

        while True:
            line = f.readline().decode('ascii', errors='replace').strip()
            header_lines.append(line)

            if line.startswith('element vertex'):
                n_vertices = int(line.split()[-1])
            elif line.startswith('element face'):
                n_faces = int(line.split()[-1])
            elif line.startswith('format binary'):
                is_binary = True
            elif line == 'end_header':
                break

        if not is_binary:
            raise NotImplementedError("Only binary PLY files are supported")

        # Parse vertex properties for size calculation
        vertex_props = []
        in_vertex = False
        for line in header_lines:
            if line.startswith('element vertex'):
                in_vertex = True
                continue
            if in_vertex and line.startswith('property'):
                parts = line.split()
                if parts[1] != 'list':
                    vertex_props.append(parts[1])
            elif in_vertex and line.startswith('element'):
                in_vertex = False

        # Parse face properties (skip list properties like vertex_indices)
        face_props = []  # list of (dtype, name) for scalar properties only
        face_list_prop = None  # (count_type, index_type) for the list property
        in_face = False
        for line in header_lines:
            if line.startswith('element face'):
                in_face = True
                continue
            if in_face and line.startswith('property'):
                parts = line.split()
                if parts[1] == 'list':
                    # property list <count_type> <index_type> <name>
                    face_list_prop = (parts[2], parts[3])
                elif len(parts) >= 3:
                    face_props.append((parts[1], parts[2]))
            elif in_face and line.startswith('element'):
                in_face = False

        vertex_size = sum(_ply_type_size(t) for t in vertex_props)
        f.read(n_vertices * vertex_size)

        # Determine list property sizes
        list_count_size = _ply_type_size(face_list_prop[0]) if face_list_prop else 1
        list_count_fmt = _ply_format(face_list_prop[0]) if face_list_prop else '<B'
        list_index_size = _ply_type_size(face_list_prop[1]) if face_list_prop else 4

        # Read faces
        for _ in range(n_faces):
            # Read vertex index list
            n_verts = struct.unpack(list_count_fmt, f.read(list_count_size))[0]
            f.read(n_verts * list_index_size)

            # Read scalar face properties
            face_data = {}
            for dtype, name in face_props:
                size = _ply_type_size(dtype)
                val = struct.unpack(_ply_format(dtype), f.read(size))[0]
                face_data[name] = val

            latitudes.append(face_data.get('latitude', 0.0))
            longitudes.append(face_data.get('longitude', 0.0))

    return np.array(latitudes), np.array(longitudes)


def _ply_type_size(dtype):
    """Size in bytes of a PLY data type."""
    sizes = {
        'float': 4, 'float32': 4, 'double': 8, 'float64': 8,
        'int': 4, 'int32': 4, 'uint': 4, 'uint32': 4,
        'short': 2, 'int16': 2, 'ushort': 2, 'uint16': 2,
        'char': 1, 'int8': 1, 'uchar': 1, 'uint8': 1,
    }
    return sizes.get(dtype, 4)


def _ply_format(dtype):
    """Struct format string for a PLY data type."""
    formats = {
        'float': '<f', 'float32': '<f', 'double': '<d', 'float64': '<d',
        'int': '<i', 'int32': '<i', 'uint': '<I', 'uint32': '<I',
        'short': '<h', 'int16': '<h', 'ushort': '<H', 'uint16': '<H',
        'char': '<b', 'int8': '<b', 'uchar': '<B', 'uint8': '<B',
    }
    return formats.get(dtype, '<f')
