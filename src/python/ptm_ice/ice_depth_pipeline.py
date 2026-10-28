"""Ice Depth Pipeline

Reads PTM3D temperature outputs (precomputed statistics VTU, binary timeseries, or
legacy VTK ASCII), aggregates per-face statistics (mean / max / min), and computes
ice depths using the vectorized solver (`compute_ice_depths`) provided in
`ice_table_vectorized.py`.

Design goals:
- Prefer already exported PTM3D statistics (statistics_avg.vtu, statistics_max.vtu,
    statistics_min.vtu) when present for zero recomputation.
- Memory efficient streaming for large face counts (>= 1e7) via chunked read when
    computing statistics from raw binary timeseries.
- Optional use of Numba backend if available (leverages existing dispatcher).
- Graceful degradation: if h5py missing, skip HDF5 output.
- Avoid loading full time-face matrix when not required (two-pass mean/max).

Binary format assumptions (temperature_XXXX.bin):
- Each file contains contiguous float32 or float64 temperatures for all faces.
- File naming pattern: temperature_<index>.bin (>= 6 digits from current engines,
  4 from older ones), read in NUMERIC index order
- All files same dtype & length; first file determines face count.

Legacy VTK assumptions:
- One VTK file per timestep OR combined file containing multiple SCALARS blocks.
- We parse only lines after 'SCALARS temperature_' and before next 'SCALARS' or EOF.
- Temperatures stored as ASCII floats, one per line.

Outputs:
- ice_depth.csv: columns: face_index, T_mean, T_max, T_min(optional), ice_depth
- Optional HDF5: dataset /ice_depth (float32), /T_mean, /T_max, /T_min (if available)
- Optional VTU + PVD: `ice_depth.vtu` containing added CellData arrays (ice_depth, ice_T_mean, ice_T_max, ice_T_min) and a one-step `ice_depth.pvd` referencing it for ParaView consumption.

CLI usage example:
    python -m ptm_ice.ice_depth_pipeline \
        --source-dir /path/to/output/temperature \
        --format binary \
        --timesteps 720 \
        --dtype float32 \
        --backend auto \
        --out-dir /path/to/ice_depth_results

Parameter mapping to solver:
    T_avg -> mean temperature (K)
    T_peak -> max temperature (K)
    Other physical parameters exposed with defaults; adjust via CLI.

"""
from __future__ import annotations
import os
import re
import math
import csv
import argparse
from typing import Iterable, Tuple, List, Optional, Dict
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

try:  # optional tqdm progress bar
    from tqdm import tqdm  # type: ignore
    HAS_TQDM = True
except Exception:  # pragma: no cover
    HAS_TQDM = False

try:
    import h5py  # type: ignore
    HAS_H5PY = True
except Exception:
    HAS_H5PY = False

try:
    import meshio  # type: ignore
    HAS_MESHIO = True
except Exception:  # pragma: no cover
    HAS_MESHIO = False

try:
    import vtk  # type: ignore
    HAS_VTK = True
except Exception:  # pragma: no cover
    HAS_VTK = False

from .ice_table_vectorized import compute_ice_depths, NUMBA_AVAILABLE
if NUMBA_AVAILABLE:
    try:
        import numba  # type: ignore
        from numba import prange, set_num_threads  # type: ignore
    except Exception:  # pragma: no cover
        numba = None

# ---------------------------------------------------------------------------
# Numba-assisted parallel accumulation (faces axis), optional
# ---------------------------------------------------------------------------
if NUMBA_AVAILABLE:
    @numba.njit(parallel=True)
    def _accumulate_face_stats_parallel(T_sum: np.ndarray, T_max: np.ndarray, data: np.ndarray) -> None:  # pragma: no cover (numba compiled)
        for i in prange(data.size):
            val = data[i]
            T_sum[i] += val
            if val > T_max[i]:
                T_max[i] = val


# ---------------------------------------------------------------------------
# VTU (XML) statistics reader (precomputed mean / max / min)
# ---------------------------------------------------------------------------

def _parse_vtu_celldata_fast(file_path: str, target_name: Optional[str] = None, *, progress: bool = False) -> Optional[np.ndarray]:
    """Fast path: extract CellData DataArray from VTU with appended or inline ascii/binary encoding.

        Supports:
            - format="appended" with offsets (uncompressed)
            - format="appended" with offsets + compressor="vtkZLibDataCompressor" (zlib blocks)
            - format="ascii"
            - format="binary" (base64 with block header) for Float32/Float64
    Returns None if unsupported or parsing fails so caller can fallback.
    """
    try:
        with open(file_path, 'rb') as f:
            raw = f.read()
    except Exception:
        return None
    # Quick rejection if file clearly huge and not worth scanning header (we still try)
    header_scan_limit = min(len(raw), 4_000_000)
    app_idx = raw.find(b'<AppendedData')
    header = raw[:app_idx if app_idx != -1 else header_scan_limit]
    piece_cells_match = re.search(rb'NumberOfCells="(\d+)"', header)
    if not piece_cells_match:
        return None
    n_cells = int(piece_cells_match.group(1))
    celldata_start = header.find(b'<CellData')
    if celldata_start == -1:
        return None
    celldata_end = header.find(b'</CellData>', celldata_start)
    if celldata_end == -1:
        return None
    celldata_block = header[celldata_start:celldata_end]

    # Collect DataArray tags and associated content boundaries for ascii/binary
    tag_iter = list(re.finditer(rb'<DataArray[^>]*>', celldata_block))
    if not tag_iter:
        return None
    arrays_meta = []
    for m in tag_iter:
        tag = m.group(0)
        name_m = re.search(rb'Name="([^"]+)"', tag)
        fmt_m = re.search(rb'format="([^"]+)"', tag)
        off_m = re.search(rb'offset="(\d+)"', tag)
        type_m = re.search(rb'type="([^"]+)"', tag)
        ncomp_m = re.search(rb'NumberOfComponents="(\d+)"', tag)
        if not (fmt_m and type_m):
            continue
        fmt = fmt_m.group(1).decode('ascii')
        vtype = type_m.group(1).decode('ascii')
        name = name_m.group(1).decode('utf-8') if name_m else 'unnamed'
        ncomp = int(ncomp_m.group(1)) if ncomp_m else 1
        offset = int(off_m.group(1)) if off_m else None
        arrays_meta.append((name, fmt, vtype, ncomp, offset))

    # Filter by target name if provided
    selected: Optional[Tuple[str,str,str,int,Optional[int]]] = None
    if target_name:
        for meta in arrays_meta:
            if meta[0] == target_name:
                selected = meta
                break
    if selected is None:
        selected = arrays_meta[0]
    name, fmt, vtype, ncomp, offset = selected

    type_map = {
        'Float64': '<f8', 'Float32': '<f4'
    }
    if vtype not in type_map:
        return None  # Only support float types fast path
    np_dt = np.dtype(type_map[vtype])

    # format handlers
    if fmt == 'ascii':
        # Extract numeric tokens between opening and closing tag within celldata_block slice
        # Reconstruct full slice from raw (we rely on celldata_block only)
        # Simpler: find closing tag from current tag position relative to celldata_block
        start_rel = m.start()  # last iter variable m would be final; recompute properly
        # Re-find the tag for selected name to get its relative position
        tag_pattern = rf'<DataArray[^>]*Name="{name}"[^>]*>'.encode('utf-8')
        tag_pos = celldata_block.find(tag_pattern)
        if tag_pos == -1:
            return None
        close_pos = celldata_block.find(b'</DataArray>', tag_pos)
        if close_pos == -1:
            return None
        data_slice = celldata_block[tag_pos:close_pos]
        # Remove tag portion
        gt_idx = data_slice.find(b'>')
        if gt_idx == -1:
            return None
        payload = data_slice[gt_idx+1:]
        # Split tokens
        vals = []
        for tok in re.findall(rb'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+', payload):
            try:
                vals.append(float(tok.decode('ascii')))
            except Exception:
                pass
        arr = np.asarray(vals, dtype='float64')
        if arr.size < n_cells:
            return None
        return arr[:n_cells]
    elif fmt == 'appended' and offset is not None:
        # Determine header integer size and compressor
        header_type_match = re.search(rb'header_type="(UInt64|UInt32)"', header)
        header_type = header_type_match.group(1).decode('ascii') if header_type_match else 'UInt32'
        int_nbytes = 8 if header_type == 'UInt64' else 4
        compressor_match = re.search(rb'compressor="([^"]+)"', header)
        compressor = compressor_match.group(1).decode('ascii') if compressor_match else None
        app_tag_start = raw.find(b'<AppendedData')
        if app_tag_start == -1:
            return None
        app_gt = raw.find(b'>', app_tag_start)
        if app_gt == -1:
            return None
        us_idx = raw.find(b'_', app_gt)
        if us_idx == -1:
            return None
        appended_start = us_idx + 1
        pos = appended_start + offset
        if pos + int_nbytes > len(raw):
            return None
        # Uncompressed appended: first integer is payload length
        if not compressor or compressor.lower() == 'none':
            block_len = int.from_bytes(raw[pos:pos+int_nbytes], 'little')
            payload_start = pos + int_nbytes
            payload_end = payload_start + block_len
            if payload_end > len(raw):
                return None
            payload = memoryview(raw)[payload_start:payload_end]
            arr = np.frombuffer(payload, dtype=np_dt)
        else:
            # Compressed appended (zlib): layout per VTK XML: [nblocks][block_size][last_block]
            # followed by [nblocks x compressed_sizes] and concatenated compressed blocks
            import zlib  # local import to avoid overhead if not needed
            cursor = pos
            # number of blocks
            nblocks = int.from_bytes(raw[cursor:cursor+int_nbytes], 'little')
            cursor += int_nbytes
            if nblocks <= 0 or cursor + 2*int_nbytes > len(raw):
                return None
            block_size = int.from_bytes(raw[cursor:cursor+int_nbytes], 'little'); cursor += int_nbytes
            last_block = int.from_bytes(raw[cursor:cursor+int_nbytes], 'little'); cursor += int_nbytes
            # compressed sizes per block
            comp_sizes = []
            for _ in range(nblocks):
                if cursor + int_nbytes > len(raw):
                    return None
                comp_sizes.append(int.from_bytes(raw[cursor:cursor+int_nbytes], 'little'))
                cursor += int_nbytes
            # Decompress each block
            out_chunks: List[bytes] = []
            for bi in range(nblocks):
                csz = comp_sizes[bi]
                if cursor + csz > len(raw):
                    return None
                comp_blob = bytes(raw[cursor:cursor+csz])
                cursor += csz
                try:
                    decomp = zlib.decompress(comp_blob)
                except Exception:
                    return None
                # Validate uncompressed size per block
                expected_u = last_block if bi == (nblocks - 1) else block_size
                if expected_u > 0 and len(decomp) != expected_u:
                    # Some VTK variants include a leading 4-byte length; strip if present
                    if len(decomp) - int_nbytes == expected_u:
                        decomp = decomp[int_nbytes:]
                    else:
                        # Fallback: accept as-is to avoid hard failure
                        pass
                out_chunks.append(decomp)
            payload = b''.join(out_chunks)
            arr = np.frombuffer(memoryview(payload), dtype=np_dt)
        expected = n_cells * ncomp
        if arr.size >= expected:
            arr = arr[:expected]
        if ncomp > 1:
            try:
                arr = arr.reshape((n_cells, ncomp))[:, 0]
            except Exception:
                arr = arr.ravel()[:n_cells]
        return arr.astype('float64')
    elif fmt == 'binary':
        # Need to locate inline base64 between tags in full file (not just celldata_block because payload may be outside slice)
        # Simple approach: find opening tag in raw then closing tag
        tag_regex = rf'<DataArray[^>]*Name="{name}"[^>]*>'.encode('utf-8')
        tag_pos = raw.find(tag_regex)
        if tag_pos == -1:
            return None
        close_pos = raw.find(b'</DataArray>', tag_pos)
        if close_pos == -1:
            return None
        gt_idx = raw.find(b'>', tag_pos)
        if gt_idx == -1 or gt_idx > close_pos:
            return None
        b64_payload = raw[gt_idx+1:close_pos].strip()
        # Base64 may include newlines/spaces
        b64_payload = b''.join(b64_payload.split())
        import base64
        try:
            decoded = base64.b64decode(b64_payload)
        except Exception:
            return None
        # First header_len bytes length header (UInt32 default)
        header_type_match = re.search(rb'header_type="(UInt64|UInt32)"', header)
        header_type = header_type_match.group(1).decode('ascii') if header_type_match else 'UInt32'
        header_len = 8 if header_type == 'UInt64' else 4
        if len(decoded) <= header_len:
            return None
        arr_buf = decoded[header_len:]
        arr = np.frombuffer(arr_buf, dtype=np_dt)
        expected = n_cells * ncomp
        if arr.size >= expected:
            arr = arr[:expected]
        if ncomp > 1:
            try:
                arr = arr.reshape((n_cells, ncomp))[:,0]
            except Exception:
                arr = arr.ravel()[:n_cells]
        return arr.astype('float64')
    else:
        return None


def _parse_vtu_celldata_vtk(file_path: str, target_name: Optional[str] = None, *, progress: bool = False) -> Optional[np.ndarray]:
    """Fast VTU CellData reader using VTK-Python library.

    Returns None if not available or parsing fails.
    """
    if not HAS_VTK:
        return None
    try:
        from vtk.util.numpy_support import vtk_to_numpy  # type: ignore
        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(file_path)
        reader.Update()
        output = reader.GetOutput()
        cell_data = output.GetCellData()
        if cell_data.GetNumberOfArrays() == 0:
            return None
        if target_name:
            arr_vtk = cell_data.GetArray(target_name)
            if arr_vtk is None:
                return None
        else:
            arr_vtk = cell_data.GetArray(0)
        arr = vtk_to_numpy(arr_vtk)
        # If multi-component, take first component
        if arr.ndim > 1:
            arr = arr[:, 0]
        return arr.astype('float64')
    except Exception:
        return None


_DATAARRAY_TAG_RE = re.compile(r"<DataArray[^>]*>")
_NAME_ATTR_RE = re.compile(r"Name=\"([^\"]+)\"")

def _parse_vtu_scalar(file_path: str, target_name: Optional[str] = None, *, progress: bool = False, fast_only: bool = False) -> np.ndarray:
    """Parse a VTU file and extract a scalar DataArray by name.

    Attempts ASCII line parse first; if that fails (UnicodeDecodeError or no arrays)
    and meshio is available, falls back to meshio which supports appended/binary VTU.
    """
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        lines = []  # trigger meshio fallback

    if lines:
        in_cell = False
        collecting = False
        current_name = None
        buffer_vals: List[float] = []
        found_arrays: Dict[str, np.ndarray] = {}
        for line in lines:
            ls = line.strip()
            if ls.startswith('<CellData'):
                in_cell = True
                continue
            if in_cell and ls.startswith('</CellData'):
                break
            if not in_cell:
                continue
            if '<DataArray' in ls:
                if collecting and current_name and buffer_vals:
                    found_arrays[current_name] = np.asarray(buffer_vals, dtype='float64')
                collecting = True
                buffer_vals = []
                m = _NAME_ATTR_RE.search(ls)
                current_name = m.group(1) if m else 'unnamed'
                continue
            if collecting and '</DataArray>' in ls:
                if current_name and buffer_vals:
                    found_arrays[current_name] = np.asarray(buffer_vals, dtype='float64')
                collecting = False
                current_name = None
                buffer_vals = []
                continue
            if collecting:
                for tok in ls.split():
                    try:
                        buffer_vals.append(float(tok))
                    except ValueError:
                        pass
        if collecting and current_name and buffer_vals:
            found_arrays[current_name] = np.asarray(buffer_vals, dtype='float64')
        if found_arrays:
            if target_name:
                if target_name not in found_arrays:
                    raise ValueError(f"Requested DataArray '{target_name}' not found in {file_path}. Available: {list(found_arrays.keys())}")
                return found_arrays[target_name]
            first_key = sorted(found_arrays.keys())[0]
            return found_arrays[first_key]
    # Try fast path(s) first unless disabled
    try:
        fast = _parse_vtu_celldata_fast(file_path, target_name=target_name, progress=progress)
        if fast is not None:
            return fast
    except Exception:
        pass
    # Try VTK-Python reader (fast, robust)
    try:
        vtk_arr = _parse_vtu_celldata_vtk(file_path, target_name=target_name, progress=progress)
        if vtk_arr is not None:
            return vtk_arr
    except Exception:
        pass
    if fast_only:
        raise ValueError(f"Fast VTU parse failed for {file_path}; retry without --stats-fast to use meshio fallback.")
    # Fallback: meshio
    if not HAS_MESHIO:
        raise ValueError(f"Failed ASCII parse and meshio not installed for {file_path}")
    # Timed, progressive feedback for potentially large binary VTU
    import sys, threading, time
    file_size = os.path.getsize(file_path)
    start_time = time.time()
    stop_event = threading.Event()

    def _progress_printer():  # pragma: no cover
        last_report = -1
        while not stop_event.is_set():
            elapsed = time.time() - start_time
            # Print every ~10 seconds
            if int(elapsed) // 10 > last_report:
                last_report = int(elapsed) // 10
                print(f"[meshio] Loading {os.path.basename(file_path)} | size={file_size/1e6:.1f} MB | elapsed={elapsed:.0f}s", file=sys.stderr, flush=True)
            time.sleep(2.0)
    if progress:
        t = threading.Thread(target=_progress_printer, daemon=True)
        t.start()
        print(f"Reading binary VTU with meshio: {os.path.basename(file_path)} (size={file_size/1e6:.1f} MB)...", file=sys.stderr, flush=True)
    mesh = meshio.read(file_path)
    stop_event.set()
    if progress:
        print(f"[meshio] Completed {os.path.basename(file_path)} in {(time.time()-start_time):.1f}s, cells={sum(len(block.data) for block in mesh.cells)}", file=sys.stderr, flush=True)
    # Gather cell data arrays (may be list per cell block)
    # Flatten blocks if multiple; choose by target_name or first
    aggregated: Dict[str, np.ndarray] = {}
    for name, data_list in mesh.cell_data.items():  # meshio: dict[str, List[np.ndarray]]
        # Concatenate across cell blocks
        try:
            concatenated = np.concatenate(data_list).astype('float64')
        except Exception:
            continue
        aggregated[name] = concatenated
    if not aggregated:
        raise ValueError(f"No cell_data arrays available via meshio in {file_path}")
    if target_name:
        if target_name not in aggregated:
            raise ValueError(f"Requested DataArray '{target_name}' not in VTU. Available: {list(aggregated.keys())}")
        return aggregated[target_name]
    first_key = sorted(aggregated.keys())[0]
    return aggregated[first_key]

def statistics_available(source_dir: str) -> bool:
    return all(os.path.isfile(os.path.join(source_dir, f)) for f in (
        'statistics_avg.vtu', 'statistics_max.vtu'))

def load_statistics_arrays(source_dir: str, *, progress: bool = False, fast_only: bool = False) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Load precomputed mean / max / optional min temperature arrays from statistics VTU files.

    Files expected:
        statistics_avg.vtu  -> mean temperatures (CellData array named 'temperature')
        statistics_max.vtu  -> max temperatures (CellData array named 'temperature')
        statistics_min.vtu  -> (optional) min temperatures (CellData array named 'temperature')
    Returns (T_mean, T_max, T_min_or_None)
    Automatically converts Celsius to Kelvin if negative temperatures detected.
    """
    import sys
    avg_path = os.path.join(source_dir, 'statistics_avg.vtu')
    max_path = os.path.join(source_dir, 'statistics_max.vtu')
    min_path = os.path.join(source_dir, 'statistics_min.vtu')
    if not (os.path.isfile(avg_path) and os.path.isfile(max_path)):
        raise FileNotFoundError("Required statistics_avg.vtu or statistics_max.vtu missing")
    # Each statistics VTU file contains multiple CellData arrays (temperature, illumination, etc.)
    # We need to explicitly request the 'temperature' array
    T_mean = _parse_vtu_scalar(avg_path, target_name='temperature', progress=progress, fast_only=fast_only)
    T_max = _parse_vtu_scalar(max_path, target_name='temperature', progress=progress, fast_only=fast_only)
    T_min: Optional[np.ndarray] = None
    if os.path.isfile(min_path):
        try:
            T_min = _parse_vtu_scalar(min_path, target_name='temperature', progress=progress, fast_only=fast_only)
        except Exception:
            T_min = None
    if T_mean.shape != T_max.shape or (T_min is not None and T_min.shape != T_mean.shape):
        raise ValueError("Statistics VTU arrays have inconsistent shapes")
    # Auto-detect Celsius and convert to Kelvin
    if np.any(T_mean < 0) or np.any(T_max < 0) or (T_min is not None and np.any(T_min < 0)):
        print(f"[WARNING] Negative temperatures detected (min T_mean={T_mean.min():.2f}, T_max={T_max.min():.2f}); assuming Celsius and converting to Kelvin.", file=sys.stderr, flush=True)
        T_mean = T_mean + 273.15
        T_max = T_max + 273.15
        if T_min is not None:
            T_min = T_min + 273.15
    # Sanity check: temperatures should be physically reasonable
    if np.any(T_mean < 2.7) or np.any(T_mean > 1000.0) or np.any(T_max < 2.7) or np.any(T_max > 1000.0):
        print(f"[WARNING] Unreasonable temperatures detected. Check that 'temperature' array contains actual temperatures, not other data.", file=sys.stderr, flush=True)
        print(f"          T_mean range: {T_mean.min():.2f} - {T_mean.max():.2f} K", file=sys.stderr, flush=True)
        print(f"          T_max range:  {T_max.min():.2f} - {T_max.max():.2f} K", file=sys.stderr, flush=True)
    # Diagnostic output
    print(f"Loaded statistics for {T_mean.size:,} faces:", file=sys.stderr, flush=True)
    print(f"  T_mean: {T_mean.min():.2f} K .. {T_mean.max():.2f} K (mean={T_mean.mean():.2f} K)", file=sys.stderr, flush=True)
    print(f"  T_max:  {T_max.min():.2f} K .. {T_max.max():.2f} K (mean={T_max.mean():.2f} K)", file=sys.stderr, flush=True)
    if T_min is not None:
        print(f"  T_min:  {T_min.min():.2f} K .. {T_min.max():.2f} K (mean={T_min.mean():.2f} K)", file=sys.stderr, flush=True)
    # Check if mean ≈ max everywhere (bad statistics)
    diff = np.abs(T_max - T_mean)
    if np.percentile(diff, 95) < 0.1:
        print(f"[WARNING] T_mean ≈ T_max for >95% of faces (max diff={diff.max():.3e} K). Statistics files may contain single-timestep snapshots instead of aggregates.", file=sys.stderr, flush=True)
        print(f"          Ice depth solver expects time-averaged mean and peak temperatures; results may be physically incorrect.", file=sys.stderr, flush=True)
    return T_mean, T_max, T_min

# ---------------------------------------------------------------------------
# Binary temperature reader (streaming)
# ---------------------------------------------------------------------------

def _enumerate_binary_files(source_dir: str) -> List[str]:
    # NUMERIC order: 4-digit names from older engines roll over to 5 digits at
    # frame 10,000 and sort out of order alphabetically.
    names = [n for n in os.listdir(source_dir) if re.match(r"temperature_\d+\.bin$", n)]
    names.sort(key=lambda n: int(n[len("temperature_"):-4]))
    return [os.path.join(source_dir, n) for n in names]

def infer_binary_layout(first_file: str, dtype: str) -> int:
    dt = np.dtype(dtype)
    size_bytes = os.path.getsize(first_file)
    if size_bytes % dt.itemsize != 0:
        raise ValueError(f"Binary file size {size_bytes} not divisible by dtype size {dt.itemsize}")
    return size_bytes // dt.itemsize

def stream_binary_mean_max(source_dir: str, timesteps: int, dtype: str = 'float32',
                           chunk_faces: int = 1_000_000,
                           threads: int = 1,
                           progress: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    files = _enumerate_binary_files(source_dir)
    if not files:
        raise FileNotFoundError(f"No temperature_XXXX.bin files in {source_dir}")
    if timesteps > 0:
        files = files[:timesteps]
    n_steps = len(files)
    n_faces = infer_binary_layout(files[0], dtype)

    # Streaming accumulation to avoid storing full matrix
    T_sum = np.zeros(n_faces, dtype='float64')
    T_max = np.full(n_faces, -np.inf, dtype='float64')

    # Read faces in chunks across timesteps (opt for fewer large IO ops)
    dt = np.dtype(dtype)
    iterator = files
    if progress:
        if HAS_TQDM:
            iterator = tqdm(files, desc="Streaming binary", unit="file")
        else:
            print(f"Streaming {len(files)} binary temperature files (no tqdm installed)...")

    # Parallelization strategy: parallelize accumulation across faces using numba if available; IO remains sequential.
    use_numba_parallel = NUMBA_AVAILABLE and threads > 1 and numba is not None
    if use_numba_parallel:
        try:
            set_num_threads(threads)
        except Exception:
            pass
    for path in iterator:
        data = np.fromfile(path, dtype=dt)
        if data.size != n_faces:
            raise ValueError(f"File {path} length mismatch: {data.size} vs {n_faces}")
        if use_numba_parallel:
            _accumulate_face_stats_parallel(T_sum, T_max, data.astype('float64'))
        else:
            T_sum += data
            np.maximum(T_max, data, out=T_max)

    T_mean = T_sum / n_steps
    return T_mean.astype('float64'), T_max.astype('float64')

# ---------------------------------------------------------------------------
# Legacy VTK ASCII reader
# ---------------------------------------------------------------------------
VTK_SCALARS_RE = re.compile(r"SCALARS\s+temperature_\d+\s+\w+")

def _extract_temperature_block(lines: List[str], start_idx: int) -> List[float]:
    vals: List[float] = []
    i = start_idx
    # Skip optional LOOKUP_TABLE line if present
    if i < len(lines) and lines[i].strip().startswith('LOOKUP_TABLE'):
        i += 1
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith('SCALARS'):
            break
        # Terminating sections that may appear
        if line.startswith('POINT_DATA') or line.startswith('CELL_DATA'):  # new section
            break
        try:
            vals.append(float(line))
        except ValueError:
            # Non-numeric line -> end of block
            break
        i += 1
    return vals

def read_vtk_temperatures(file_path: str) -> List[np.ndarray]:
    with open(file_path, 'r') as f:
        lines = f.readlines()
    temps: List[np.ndarray] = []
    for idx, line in enumerate(lines):
        if VTK_SCALARS_RE.search(line):
            block = _extract_temperature_block(lines, idx + 1)
            temps.append(np.asarray(block, dtype='float64'))
    if not temps:
        raise ValueError(f"No temperature blocks found in {file_path}")
    # Validate consistent face counts
    n_faces = temps[0].size
    for arr in temps:
        if arr.size != n_faces:
            raise ValueError("Inconsistent face count among temperature blocks")
    return temps

def stream_vtk_mean_max(source_dir: str, timesteps: int = 0, *, threads: int = 1, progress: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    files = [os.path.join(source_dir, f) for f in sorted(os.listdir(source_dir)) if f.endswith('.vtk')]
    if not files:
        # Check for common mistake: user specified --format vtk but directory has .vtu files
        vtu_files = [f for f in os.listdir(source_dir) if f.endswith('.vtu')]
        if vtu_files:
            raise FileNotFoundError(
                f"No .vtk files in {source_dir}. Found {len(vtu_files)} .vtu files instead. "
                "If you have statistics_avg.vtu and statistics_max.vtu, omit --format to auto-detect. "
                "Otherwise, use --format binary for temperature_XXXX.bin files."
            )
        raise FileNotFoundError(f"No .vtk files in {source_dir}")
    if timesteps > 0:
        files = files[:timesteps]

    T_sum: Optional[np.ndarray] = None
    T_max: Optional[np.ndarray] = None
    count = 0
    iterator = files
    if progress:
        if HAS_TQDM:
            iterator = tqdm(files, desc="Streaming VTK", unit="file")
        else:
            print(f"Streaming {len(files)} VTK files (no tqdm installed)...")
    for fp in iterator:
        blocks = read_vtk_temperatures(fp)  # may contain multiple timesteps
        for arr in blocks:
            if T_sum is None:
                T_sum = np.zeros_like(arr, dtype='float64')
                T_max = np.full_like(arr, -np.inf, dtype='float64')
            T_sum += arr
            np.maximum(T_max, arr, out=T_max)
            count += 1
    if T_sum is None or T_max is None:
        raise RuntimeError("Failed to accumulate temperatures from VTK files")
    T_mean = T_sum / count
    return T_mean, T_max

# ---------------------------------------------------------------------------
# Solver integration
# ---------------------------------------------------------------------------

def compute_ice_depth_from_stats(T_mean: np.ndarray, T_max: np.ndarray, *,
                                 backend: str = 'auto',
                                 J0: float = 1e-5, delta: float = 0.05,
                                 l: float = 2.83e6, zs: float = 4000.0,
                                 Q: float = 1e-3, k: float = 205.0,
                                 chunk_size: int = 500_000) -> np.ndarray:
    if T_mean.shape != T_max.shape:
        raise ValueError("T_mean and T_max shape mismatch")
    # Resolve 'auto' backend selection
    if backend == 'auto':
        backend = 'numba' if NUMBA_AVAILABLE else 'numpy'
    depths, _ = compute_ice_depths(
        tavg=T_mean,
        tmax=T_max,
        J0=J0,
        delta=delta,
        l=l,
        zs=zs,
        Q=Q,
        k=k,
        backend=backend,
        chunk_size=chunk_size,
    )
    return depths

# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_csv(out_path: str, T_mean: np.ndarray, T_max: np.ndarray, depths: np.ndarray, T_min: Optional[np.ndarray] = None) -> None:
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        header = ['face', 'T_mean', 'T_max']
        if T_min is not None:
            header.append('T_min')
        header.append('ice_depth')
        w.writerow(header)
        for i in range(depths.size):
            row = [i, f"{T_mean[i]:.6f}", f"{T_max[i]:.6f}"]
            if T_min is not None:
                row.append(f"{T_min[i]:.6f}")
            row.append(f"{depths[i]:.8f}")
            w.writerow(row)


def write_hdf5(out_path: str, T_mean: np.ndarray, T_max: np.ndarray, depths: np.ndarray, T_min: Optional[np.ndarray] = None) -> None:
    if not HAS_H5PY:
        raise RuntimeError("h5py not available")
    with h5py.File(out_path, 'w') as h5:
        h5.create_dataset('ice_depth', data=depths.astype('float32'), compression='gzip')
        h5.create_dataset('T_mean', data=T_mean.astype('float32'), compression='gzip')
        h5.create_dataset('T_max', data=T_max.astype('float32'), compression='gzip')
        if T_min is not None:
            h5.create_dataset('T_min', data=T_min.astype('float32'), compression='gzip')

# ---------------------------------------------------------------------------
# VTU / PVD writers (for ParaView visualization)
# ---------------------------------------------------------------------------

def _format_dataarray(name: str, data: np.ndarray, dtype: str = 'Float64') -> List[str]:
    """Create XML lines for a CellData DataArray in ASCII format."""
    # Single line of space-separated values (ParaView handles large lines; chunking optional)
    values_line = ' '.join(f"{v:.12g}" for v in data.astype('float64'))
    return [f'    <DataArray type="{dtype}" Name="{name}" format="ascii">', f'      {values_line}', '    </DataArray>']

def write_vtu_with_ice_depth(template_vtu: str, out_vtu: str, depths: np.ndarray,
                              T_mean: np.ndarray, T_max: np.ndarray, T_min: Optional[np.ndarray]) -> None:
    """Create an output VTU with ice depth & temperature stats.

    If meshio is available, use it to robustly load binary/appended VTU and write a
    fresh VTU containing new CellData arrays. Otherwise attempt in-place ASCII injection.
    """
    if not os.path.isfile(template_vtu):
        raise FileNotFoundError(f"Template VTU not found: {template_vtu}")
    if not (depths.size == T_mean.size == T_max.size and (T_min is None or T_min.size == depths.size)):
        raise ValueError("Inconsistent array sizes for VTU export")
    if HAS_MESHIO:
        mesh = meshio.read(template_vtu)
        # Determine total cell count
        total_cells = sum(len(block.data) for block in mesh.cells)
        if total_cells != depths.size:
            raise ValueError(f"Template cell count {total_cells} != depths size {depths.size}")
        # Split depths across cell blocks preserving order
        split_arrays: List[np.ndarray] = []
        start = 0
        for block in mesh.cells:
            end = start + len(block.data)
            split_arrays.append(depths[start:end])
            start = end
        def split_stat(stat: np.ndarray) -> List[np.ndarray]:
            parts = []
            s = 0
            for block in mesh.cells:
                e = s + len(block.data)
                parts.append(stat[s:e])
                s = e
            return parts
        cell_data: Dict[str, List[np.ndarray]] = {
            'ice_depth': split_arrays,
            'ice_T_mean': split_stat(T_mean),
            'ice_T_max': split_stat(T_max),
        }
        if T_min is not None:
            cell_data['ice_T_min'] = split_stat(T_min)
        mesh.cell_data.update(cell_data)
        meshio.write(out_vtu, mesh, file_format='vtu')
        return
    # ASCII fallback injection
    with open(template_vtu, 'r') as f:
        lines = f.readlines()
    cell_start = None
    cell_end = None
    for i, ln in enumerate(lines):
        if '<CellData' in ln:
            cell_start = i
        if cell_start is not None and '</CellData>' in ln:
            cell_end = i
            break
    new_arrays: List[str] = []
    new_arrays.extend(_format_dataarray('ice_depth', depths))
    new_arrays.extend(_format_dataarray('ice_T_mean', T_mean))
    new_arrays.extend(_format_dataarray('ice_T_max', T_max))
    if T_min is not None:
        new_arrays.extend(_format_dataarray('ice_T_min', T_min))
    if cell_start is not None and cell_end is not None:
        existing = '\n'.join(lines[cell_start:cell_end+1])
        filtered = []
        for arr in new_arrays:
            name_match = re.search(r'Name="([^"]+)"', arr)
            if name_match and name_match.group(1) in existing:
                continue
            filtered.append(arr)
        insertion_index = cell_end
        lines[insertion_index:insertion_index] = [l + '\n' for l in filtered]
    else:
        piece_idx = None
        for i, ln in enumerate(lines):
            if '<Piece' in ln:
                piece_idx = i
                break
        if piece_idx is None:
            raise RuntimeError("Could not locate <Piece> element in template VTU (ASCII fallback)")
        block = ['  <CellData>\n'] + [l + '\n' for l in new_arrays] + ['  </CellData>\n']
        lines[piece_idx+1:piece_idx+1] = block
    with open(out_vtu, 'w') as f:
        f.writelines(lines)

def write_pvd(pvd_path: str, vtu_files: List[str], times: Optional[List[float]] = None) -> None:
    """Write a minimal PVD file referencing one or more VTU files."""
    if times is None:
        times = [0.0] * len(vtu_files)
    if len(times) != len(vtu_files):
        raise ValueError("times length must match vtu_files length")
    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
        '  <Collection>'
    ]
    for t, fp in zip(times, vtu_files):
        rel = os.path.basename(fp)
        lines.append(f'    <DataSet timestep="{t:.6g}" group="" part="0" file="{rel}"/>')
    lines.extend(['  </Collection>', '</VTKFile>'])
    with open(pvd_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

# ---------------------------------------------------------------------------
# High-level pipeline
# ---------------------------------------------------------------------------

def run_pipeline(source_dir: str, fmt: str, timesteps: int, out_dir: str, *,
                 dtype: str = 'float32', backend: str = 'auto',
                 J0: float = 1e-5, delta: float = 0.05, l: float = 2.83e6,
                 zs: float = 4000.0, Q: float = 1e-3, k: float = 205.0,
                 chunk_size: int = 500_000, hdf5: bool = False,
                 prefer_statistics: bool = True,
                 pvd: bool = False,
                 mesh_vtu: Optional[str] = None,
                 pvd_time: Optional[float] = None,
                 threads: int = 1,
                 progress: bool = False,
                 fast_stats: bool = False) -> str:
    os.makedirs(out_dir, exist_ok=True)
    T_min: Optional[np.ndarray] = None
    # Attempt statistics first if requested and format not forcing raw
    if prefer_statistics and statistics_available(source_dir):
        if progress:
            print("Loading precomputed statistics from VTU files...")
        try:
            T_mean, T_max, T_min = load_statistics_arrays(source_dir, progress=progress, fast_only=fast_stats)
            if progress:
                print(f"Statistics loaded: {T_mean.size} faces")
        except Exception as e:
            if fast_stats and 'Fast VTU parse failed' in str(e):
                raise RuntimeError(
                    f"Fast statistics parse failed: {e}.\n"
                    "Rerun without --stats-fast to allow meshio (slower) fallback, or provide raw binary temperature files."
                )
            print(f"Warning: failed to load statistics VTU files ({e}); attempting raw fallback")
            T_mean = T_max = None  # trigger fallback
    else:
        T_mean = T_max = None

    if T_mean is None or T_max is None:
        # Runtime format fallback if statistics failed and fmt is None
        if fmt is None:
            bin_files = _enumerate_binary_files(source_dir)
            if bin_files:
                fmt = 'binary'
            else:
                vtk_files = [f for f in os.listdir(source_dir) if f.endswith('.vtk')]
                if vtk_files:
                    fmt = 'vtk'
                else:
                    raise RuntimeError(
                        "Statistics unavailable and no raw temperature files detected (binary or .vtk). "
                        "Provide temperature_XXXX.bin or legacy .vtk files, or fix statistics VTU encoding."
                    )
            if progress:
                print(f"Auto-detected raw format: {fmt}")
        if fmt == 'binary':
            T_mean, T_max = stream_binary_mean_max(source_dir, timesteps, dtype=dtype, threads=threads, progress=progress)
        elif fmt == 'vtk':
            T_mean, T_max = stream_vtk_mean_max(source_dir, timesteps, threads=threads, progress=progress)
        else:
            raise ValueError(f"Unsupported format: {fmt}")

    # Attempt to configure numba threads for solver if applicable
    resolved_backend = backend
    if resolved_backend == 'auto':
        resolved_backend = 'numba' if NUMBA_AVAILABLE else 'numpy'
    if NUMBA_AVAILABLE and threads > 0 and resolved_backend == 'numba':
        try:
            set_num_threads(threads)
        except Exception:
            pass
    if progress:
        print(f"Computing ice depths using backend={resolved_backend} (threads={threads})...")
    depths = compute_ice_depth_from_stats(
        T_mean, T_max, backend=backend, J0=J0, delta=delta,
        l=l, zs=zs, Q=Q, k=k, chunk_size=chunk_size)

    csv_path = os.path.join(out_dir, 'ice_depth.csv')
    write_csv(csv_path, T_mean, T_max, depths, T_min)

    if hdf5 and HAS_H5PY:
        h5_path = os.path.join(out_dir, 'ice_depth.h5')
        write_hdf5(h5_path, T_mean, T_max, depths, T_min)

    if pvd:
        if mesh_vtu is None:
            raise ValueError("mesh_vtu template path required when pvd=True")
        out_vtu = os.path.join(out_dir, 'ice_depth.vtu')
        write_vtu_with_ice_depth(mesh_vtu, out_vtu, depths, T_mean, T_max, T_min)
        # Single aggregated result -> one time entry; use provided pvd_time or timesteps count if >0
        time_val = pvd_time if pvd_time is not None else (float(timesteps) if timesteps > 0 else 0.0)
        pvd_path = os.path.join(out_dir, 'ice_depth.pvd')
        write_pvd(pvd_path, [out_vtu], [time_val])
    return csv_path

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compute ice depths from PTM temperature outputs")
    p.add_argument('--source-dir', required=True, help='Directory with temperature files (binary or VTK)')
    p.add_argument('--format', choices=['binary', 'vtk'], default=None, help='Input format: binary or vtk. If omitted, auto-detects statistics VTU files (statistics_avg.vtu, statistics_max.vtu).')
    p.add_argument('--timesteps', type=int, default=0, help='Limit timesteps (0=all)')
    p.add_argument('--dtype', choices=['float32', 'float64'], default='float32', help='Binary dtype')
    p.add_argument('--backend', choices=['auto', 'numpy', 'numba'], default='auto', help='Solver backend')
    p.add_argument('--out-dir', required=True, help='Output directory')
    p.add_argument('--chunk-size', type=int, default=500000, help='Chunk size for solver')
    p.add_argument('--threads', type=int, default=1, help='Thread count for parallel accumulation & numba backend')
    p.add_argument('--progress', action='store_true', help='Show progress bars (requires tqdm) or simple status messages')
    p.add_argument('--hdf5', action='store_true', help='Write HDF5 outputs if h5py available')
    # Physical parameters (advanced overrides)
    p.add_argument('--J0', type=float, default=1e-5)
    p.add_argument('--delta', type=float, default=0.05)
    p.add_argument('--l', type=float, default=2.83e6)
    p.add_argument('--zs', type=float, default=4000.0)
    p.add_argument('--Q', type=float, default=1e-3)
    p.add_argument('--k', type=float, default=205.0)
    p.add_argument('--no-statistics', action='store_true', help='Disable use of statistics_avg/max/min.vtu even if present')
    p.add_argument('--pvd', action='store_true', help='Also write ice_depth.vtu and ice_depth.pvd (requires --mesh-vtu)')
    p.add_argument('--mesh-vtu', type=str, default=None, help='Template mesh VTU file to inject ice depth CellData arrays')
    p.add_argument('--pvd-time', type=float, default=None, help='Timestamp value for PVD entry (default: timesteps or 0)')
    p.add_argument('--stats-fast', action='store_true', help='Force fast statistics VTU reader (skip meshio); fails if unsupported encoding.')
    return p


def main(argv: Optional[List[str]] = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    
    # Auto-detect format if not specified
    fmt = args.format
    if fmt is None:
        # Check for statistics VTU files first (preferred fast path)
        if statistics_available(args.source_dir):
            fmt = 'binary'  # arbitrary choice; run_pipeline will use statistics regardless
        else:
            # Check for binary temperature files
            binary_files = _enumerate_binary_files(args.source_dir)
            if binary_files:
                fmt = 'binary'
            else:
                # Check for legacy VTK files
                vtk_files = [f for f in os.listdir(args.source_dir) if f.endswith('.vtk')]
                if vtk_files:
                    fmt = 'vtk'
                else:
                    raise ValueError(
                        f"Could not auto-detect format in {args.source_dir}. "
                        "Expected statistics_avg.vtu/statistics_max.vtu, temperature_XXXX.bin, or *.vtk files. "
                        "Please specify --format binary or --format vtk explicitly."
                    )
    
    csv_path = run_pipeline(
        source_dir=args.source_dir,
        fmt=fmt,
        timesteps=args.timesteps,
        out_dir=args.out_dir,
        dtype=args.dtype,
        backend=args.backend,
        J0=args.J0, delta=args.delta, l=args.l, zs=args.zs, Q=args.Q, k=args.k,
        chunk_size=args.chunk_size,
        hdf5=args.hdf5,
        prefer_statistics=not args.no_statistics,
        pvd=args.pvd,
        mesh_vtu=args.mesh_vtu,
        pvd_time=args.pvd_time,
        threads=args.threads,
        progress=args.progress,
        fast_stats=args.stats_fast,
    )
    print(f"Ice depth CSV written: {csv_path}")
    if args.hdf5 and HAS_H5PY:
        print("HDF5 output written as well.")
    if args.pvd:
        print("PVD + VTU outputs written (ice_depth.pvd / ice_depth.vtu)")

if __name__ == '__main__':  # pragma: no cover
    main()
