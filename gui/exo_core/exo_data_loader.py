"""
Data loader for Exosphere GUI.
Handles binary I/O, PLY mesh loading, field/timestep discovery, and caching.
"""

import csv
import struct
from functools import lru_cache
from pathlib import Path

import numpy as np

from .exo_tooltips import FIELD_INFO


# ---------------------------------------------------------------------------
# Binary data I/O
# ---------------------------------------------------------------------------

def read_binary_field(filepath):
    """Read a binary field file (uint32 header + float32 array).

    Args:
        filepath: Path to .bin file.

    Returns:
        numpy float32 array, or None on error.
    """
    filepath = Path(filepath)
    if not filepath.is_file():
        return None
    try:
        with open(filepath, "rb") as f:
            header = f.read(4)
            if len(header) < 4:
                return None
            count = struct.unpack("<I", header)[0]
            data = np.frombuffer(f.read(), dtype=np.float32)
            if len(data) != count:
                return None
            return data
    except Exception:
        return None


def write_binary_field(filepath, data):
    """Write a binary field file (uint32 header + float32 array).

    Args:
        filepath: Path to output .bin file.
        data: numpy array (will be cast to float32).
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(data, dtype=np.float32)
    with open(filepath, "wb") as f:
        f.write(struct.pack("<I", len(arr)))
        f.write(arr.tobytes())


# ---------------------------------------------------------------------------
# PLY mesh loading
# ---------------------------------------------------------------------------

def _parse_ply_header(f):
    """Parse PLY header, return metadata dict."""
    magic = f.readline().decode("ascii", errors="replace").strip()
    if magic != "ply":
        raise ValueError(f"Not a PLY file (magic: {magic!r})")

    info = {
        "format": "binary_little_endian",
        "num_vertices": 0,
        "num_faces": 0,
        "vertex_properties": [],
        "face_properties": [],
        "header_size": 0,
    }
    current_element = None

    while True:
        line = f.readline().decode("ascii", errors="replace").strip()
        if line == "end_header":
            info["header_size"] = f.tell()
            break
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "format":
            info["format"] = parts[1]
        elif parts[0] == "element":
            current_element = parts[1]
            count = int(parts[2])
            if current_element == "vertex":
                info["num_vertices"] = count
            elif current_element == "face":
                info["num_faces"] = count
        elif parts[0] == "property":
            if current_element == "vertex":
                if parts[1] == "list":
                    # vertex list property (rare)
                    info["vertex_properties"].append(
                        ("list", parts[2], parts[3], parts[4])
                    )
                else:
                    info["vertex_properties"].append((parts[1], parts[2]))
            elif current_element == "face":
                if parts[1] == "list":
                    info["face_properties"].append(
                        ("list", parts[2], parts[3], parts[4])
                    )
                else:
                    info["face_properties"].append((parts[1], parts[2]))

    return info


_PLY_TYPE_MAP = {
    "float": np.float32,
    "float32": np.float32,
    "double": np.float64,
    "float64": np.float64,
    "int": np.int32,
    "int32": np.int32,
    "uint": np.uint32,
    "uint32": np.uint32,
    "short": np.int16,
    "int16": np.int16,
    "ushort": np.uint16,
    "uint16": np.uint16,
    "char": np.int8,
    "int8": np.int8,
    "uchar": np.uint8,
    "uint8": np.uint8,
}


def load_ply_mesh(filepath):
    """Load a PLY mesh file.

    Returns:
        dict with keys:
            vertices: (N, 3) float64 array
            faces: (M, 3) int32 array (triangle indices)
            face_centers: (M, 3) float64 array
            num_vertices: int
            num_faces: int
            face_latitudes: (M,) float64 array in degrees
            face_longitudes: (M,) float64 array in degrees [0, 360)
        or None on error.
    """
    filepath = Path(filepath)
    if not filepath.is_file():
        return None

    try:
        with open(filepath, "rb") as f:
            info = _parse_ply_header(f)

            if info["format"] != "binary_little_endian":
                raise ValueError(
                    f"Unsupported PLY format: {info['format']}. "
                    "Only binary_little_endian is supported."
                )

            nv = info["num_vertices"]
            nf = info["num_faces"]

            # Build vertex dtype
            vertex_fields = []
            for prop in info["vertex_properties"]:
                if prop[0] == "list":
                    raise ValueError("List properties on vertices not supported")
                dtype = _PLY_TYPE_MAP.get(prop[0], np.float32)
                vertex_fields.append((prop[1], dtype))

            vertex_dtype = np.dtype(vertex_fields)
            vertex_data = np.frombuffer(f.read(nv * vertex_dtype.itemsize), dtype=vertex_dtype)

            # Extract x, y, z
            vertices = np.column_stack([
                vertex_data["x"].astype(np.float64),
                vertex_data["y"].astype(np.float64),
                vertex_data["z"].astype(np.float64),
            ])

            # Read faces (variable-length list: count_type + count * index_type)
            # Determine types from face list property
            list_prop = None
            scalar_face_props = []
            for prop in info["face_properties"]:
                if prop[0] == "list" and list_prop is None:
                    list_prop = prop  # ("list", count_type, index_type, name)
                elif prop[0] != "list":
                    scalar_face_props.append(prop)

            if list_prop is None:
                raise ValueError("No face vertex_indices list property found")

            count_dtype = _PLY_TYPE_MAP.get(list_prop[1], np.uint8)
            index_dtype = _PLY_TYPE_MAP.get(list_prop[2], np.int32)

            faces = np.zeros((nf, 3), dtype=np.int32)
            # Check for extra scalar face properties (e.g., lat, lon)
            extra_face_data = {prop[1]: np.zeros(nf, dtype=np.float64) for prop in scalar_face_props}

            for i in range(nf):
                count_bytes = f.read(np.dtype(count_dtype).itemsize)
                n_verts = np.frombuffer(count_bytes, dtype=count_dtype)[0]
                indices = np.frombuffer(
                    f.read(int(n_verts) * np.dtype(index_dtype).itemsize),
                    dtype=index_dtype,
                )
                # Only take first 3 for triangles
                faces[i] = indices[:3].astype(np.int32)
                # Read scalar face properties
                for prop in scalar_face_props:
                    dt = _PLY_TYPE_MAP.get(prop[0], np.float32)
                    val = np.frombuffer(f.read(np.dtype(dt).itemsize), dtype=dt)[0]
                    extra_face_data[prop[1]][i] = float(val)

        # Compute face centers
        face_centers = np.mean(vertices[faces], axis=1)

        # Compute lat/lon from face centers (spherical coordinates)
        x, y, z = face_centers[:, 0], face_centers[:, 1], face_centers[:, 2]
        r = np.sqrt(x**2 + y**2 + z**2)
        r = np.maximum(r, 1e-30)  # avoid division by zero

        latitudes = np.degrees(np.arcsin(z / r))
        longitudes = np.degrees(np.arctan2(y, x))
        longitudes[longitudes < 0] += 360.0

        return {
            "vertices": vertices,
            "faces": faces,
            "face_centers": face_centers,
            "num_vertices": nv,
            "num_faces": nf,
            "face_latitudes": latitudes,
            "face_longitudes": longitudes,
        }

    except Exception as e:
        print(f"Error loading PLY mesh: {e}")
        return None


# ---------------------------------------------------------------------------
# Field and timestep discovery
# ---------------------------------------------------------------------------

def discover_fields(directory):
    """Discover available data fields in a directory.

    Scans for subdirectories that contain .bin files matching the
    pattern {field_name}/{field_name}_NNNN.bin.

    Args:
        directory: Path to data directory.

    Returns:
        dict mapping field_name -> list of (timestep_index, filepath) tuples,
        sorted by timestep index.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return {}

    fields = {}
    for subdir in sorted(directory.iterdir()):
        if not subdir.is_dir():
            continue
        field_name = subdir.name
        bins = sorted(subdir.glob(f"{field_name}_*.bin"))
        if not bins:
            # Also check for legacy naming without the field prefix
            bins = sorted(subdir.glob("*.bin"))
        if not bins:
            continue

        timesteps = []
        for binfile in bins:
            stem = binfile.stem
            # Extract timestep number from suffix (e.g., "temperature_K_0042" -> 42)
            parts = stem.rsplit("_", 1)
            if len(parts) == 2:
                try:
                    ts_idx = int(parts[1])
                    timesteps.append((ts_idx, binfile))
                except ValueError:
                    pass
        timesteps.sort(key=lambda x: x[0])
        if timesteps:
            fields[field_name] = timesteps

    return fields


def discover_timestep_count(directory):
    """Get the number of timesteps available in a directory.

    Uses the first discovered field to determine count.
    """
    fields = discover_fields(directory)
    if not fields:
        return 0
    first_field = next(iter(fields.values()))
    return len(first_field)


def detect_mode(directory):
    """Detect whether output is diurnal or multiscale.

    Multiscale output typically has temperature_mean_K, epoch numbering, etc.

    Returns:
        "multiscale" or "diurnal"
    """
    directory = Path(directory)
    # Multiscale mode produces temperature_mean_K as averaged field
    if (directory / "temperature_mean_K").is_dir():
        return "multiscale"
    # Also check for evolution summary
    if (directory / "evolution_summary.csv").is_file():
        return "multiscale"
    return "diurnal"


def get_available_field_names(directory):
    """Get sorted list of available field names with metadata.

    Returns:
        list of dicts with keys: name, label, units, category, colormap
    """
    fields = discover_fields(directory)
    result = []
    for name in sorted(fields.keys()):
        info = FIELD_INFO.get(name, {
            "label": name.replace("_", " ").title(),
            "units": "",
            "category": "unknown",
            "colormap": "viridis",
        })
        result.append({"name": name, **info})
    return result


# ---------------------------------------------------------------------------
# Time step info
# ---------------------------------------------------------------------------

def load_timestep_times(directory):
    """Load timestep time values if available.

    Looks for time_steps.txt in the directory.

    Returns:
        numpy array of time values in seconds, or None if not found.
    """
    ts_file = Path(directory) / "time_steps.txt"
    if not ts_file.is_file():
        return None
    try:
        times = []
        with open(ts_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    times.append(float(line))
        return np.array(times)
    except Exception:
        return None


def compute_epoch_times(n_epochs, cycles_per_epoch=100, europa_day=3.5512):
    """Compute time values for multi-timescale epochs.

    Args:
        n_epochs: Number of epochs.
        cycles_per_epoch: Diurnal cycles per epoch (default: 100).
        europa_day: Europa rotation period in days (default: 3.5512).

    Returns:
        dict with:
            seconds: numpy array of times in seconds
            days: numpy array in days
            years: numpy array in years
    """
    epoch_duration_days = cycles_per_epoch * europa_day
    epoch_duration_seconds = epoch_duration_days * 86400.0
    seconds = np.arange(n_epochs) * epoch_duration_seconds
    return {
        "seconds": seconds,
        "days": seconds / 86400.0,
        "years": seconds / (365.25 * 86400.0),
    }


# ---------------------------------------------------------------------------
# Atmospheric mass CSV
# ---------------------------------------------------------------------------

def load_atmospheric_mass(directory):
    """Load atmospheric mass balance data from CSV.

    Looks for atmosphere/atmospheric_mass.csv.

    Returns:
        dict with arrays: timestep, total_emission, total_deposition,
        total_escape, atmospheric_mass (all numpy arrays), or None.
    """
    csv_path = Path(directory) / "atmosphere" / "atmospheric_mass.csv"
    if not csv_path.is_file():
        return None
    try:
        data = {"timestep": [], "total_emission": [], "total_deposition": [],
                "total_escape": [], "atmospheric_mass": []}
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                for key in data:
                    if key in row:
                        data[key].append(float(row[key]))
        return {k: np.array(v) for k, v in data.items() if v}
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Cached data loading
# ---------------------------------------------------------------------------

class ExoDataCache:
    """LRU-cached data loader for field arrays.

    Caches up to max_entries field arrays to avoid redundant disk reads.
    """

    def __init__(self, max_entries=200):
        self._cache = {}
        self._access_order = []
        self._max_entries = max_entries

    def clear(self):
        """Clear the entire cache."""
        self._cache.clear()
        self._access_order.clear()

    def get_field(self, filepath):
        """Load a field array, using cache if available.

        Args:
            filepath: Path to .bin file.

        Returns:
            numpy float32 array, or None on error.
        """
        key = str(filepath)
        if key in self._cache:
            # Move to end (most recently used)
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]

        data = read_binary_field(filepath)
        if data is not None:
            self._cache[key] = data
            self._access_order.append(key)
            # Evict oldest if over limit
            while len(self._cache) > self._max_entries:
                oldest = self._access_order.pop(0)
                self._cache.pop(oldest, None)
        return data

    def get_field_at_timestep(self, directory, field_name, timestep_idx):
        """Load a specific field at a specific timestep.

        Args:
            directory: Data directory path.
            field_name: Name of the field (subdirectory name).
            timestep_idx: Zero-based timestep index.

        Returns:
            numpy float32 array, or None.
        """
        directory = Path(directory)
        # Try zero-padded and non-padded naming
        for fmt in [f"{field_name}_{timestep_idx:04d}.bin",
                     f"{field_name}_{timestep_idx:05d}.bin",
                     f"{field_name}_{timestep_idx}.bin"]:
            filepath = directory / field_name / fmt
            if filepath.is_file():
                return self.get_field(filepath)
        return None

    def load_timeseries(self, directory, field_name, face_indices=None):
        """Load a field across all timesteps.

        Args:
            directory: Data directory.
            field_name: Field name.
            face_indices: Optional list/array of face indices to extract.
                         If None, returns full arrays.

        Returns:
            If face_indices is None:
                list of numpy arrays (one per timestep)
            If face_indices is given:
                numpy array of shape (n_timesteps, len(face_indices))
        """
        fields = discover_fields(directory)
        if field_name not in fields:
            return None

        timesteps = fields[field_name]
        if face_indices is not None:
            face_indices = np.asarray(face_indices)
            result = np.zeros((len(timesteps), len(face_indices)), dtype=np.float32)
            for i, (ts_idx, filepath) in enumerate(timesteps):
                data = self.get_field(filepath)
                if data is not None:
                    valid = face_indices < len(data)
                    result[i, valid] = data[face_indices[valid]]
            return result
        else:
            result = []
            for ts_idx, filepath in timesteps:
                data = self.get_field(filepath)
                result.append(data)
            return result

    @property
    def cache_size(self):
        """Number of cached entries."""
        return len(self._cache)

    @property
    def memory_estimate_mb(self):
        """Estimated memory usage in MB."""
        total_bytes = sum(
            arr.nbytes for arr in self._cache.values() if arr is not None
        )
        return total_bytes / (1024 * 1024)


# ---------------------------------------------------------------------------
# Face lookup utilities
# ---------------------------------------------------------------------------

def find_nearest_face(face_centers, lat_deg, lon_deg):
    """Find the mesh face nearest to a given lat/lon.

    Args:
        face_centers: (N, 3) array of face center coordinates.
        lat_deg: Target latitude in degrees.
        lon_deg: Target longitude in degrees.

    Returns:
        (face_index, distance_km) tuple.
    """
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    # Convert target to Cartesian (unit sphere)
    target = np.array([np.cos(lat) * np.cos(lon),
                       np.cos(lat) * np.sin(lon),
                       np.sin(lat)])

    # Normalize face centers to unit vectors
    norms = np.linalg.norm(face_centers, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-30)
    unit_centers = face_centers / norms

    # Dot product -> angular distance
    dots = np.clip(unit_centers @ target, -1.0, 1.0)
    angles = np.arccos(dots)
    idx = np.argmin(angles)

    # Distance on surface (approximate, using mean radius)
    mean_radius = np.mean(norms) / 1000.0  # km
    dist_km = angles[idx] * mean_radius

    return int(idx), float(dist_km)
