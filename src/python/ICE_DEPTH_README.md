# PTM Ice Table Depth Calculator

Automated pipeline for computing ice stability depths from PTM3D temperature outputs.

## Overview

This module provides efficient tooling to:
1. **Read** PTM3D temperature time series (binary `.bin` or legacy VTK ASCII)
2. **Aggregate** per-face mean and max temperatures across all timesteps
3. **Compute** ice table depths using a vectorized Newton solver
4. **Export** results to CSV (and optionally HDF5)

Designed to handle large-scale datasets (≥10⁷ mesh faces) via chunked processing and optional Numba acceleration.

### New Parallel & Progress Features

- `--threads`: Specifies thread count for Numba-based solver and parallel face-wise accumulation of mean/max (binary & VTK). Set to your physical core / logical CPU count (e.g. 48) for maximum throughput. Defaults to 1.
- `--progress`: Enables progress bars (requires `tqdm`) or simple status messages if `tqdm` is not installed.
- Internally, if Numba is available and `--threads > 1`, face-wise accumulation uses a parallel loop (`prange`). IO remains sequential to avoid excessive disk contention.
- The solver backend (`--backend auto`) will choose Numba if installed; thread count is applied via `numba.set_num_threads()` before computation.

## Installation

Requires Python ≥3.8 and NumPy:

```bash
cd src/python
pip install -e .
```

**Optional dependencies:**
- `numba` for parallel JIT acceleration (~3-5× speedup)
- `h5py` for HDF5 output format
- `pytest` for running tests

```bash
pip install numba h5py pytest
```

## Quick Start

### Python API

```python
from ptm_ice.ice_depth_pipeline import run_pipeline

csv_path = run_pipeline(
    source_dir='/path/to/temperature_files',
    fmt='binary',           # or 'vtk'
    timesteps=720,          # 0 = all files
    out_dir='/output/dir',
    dtype='float32',
    backend='auto',         # 'numpy', 'numba', or 'auto'
    chunk_size=1_000_000,   # faces per chunk
    hdf5=True,              # requires h5py
)
```

### Command Line

```bash
python -m ptm_ice.ice_depth_pipeline \
    --source-dir /path/to/ptm_output/temperature \
    --format binary \
    --timesteps 720 \
    --dtype float32 \
    --backend auto \
    --out-dir /path/to/ice_results \
    --chunk-size 1000000 \
    --threads 48 \
    --progress \
    --hdf5
        --pvd --mesh-vtu /path/to/mesh_template.vtu  # optional ParaView export
```

**Physical parameters** (optional overrides):
```bash
    --J0 1e-5 \
    --delta 0.05 \
    --l 2.83e6 \
    --zs 4000.0 \
    --Q 1e-3 \
    --k 205.0
```

## Input Formats

### Binary (temperature_XXXX.bin)

- Each file contains contiguous float32/float64 temperatures for all faces
- File naming: `temperature_0000.bin`, `temperature_0001.bin`, ...
- All files same dtype and length

**Example structure:**
```
ptm_output/
  temperature/
    temperature_0000.bin
    temperature_0001.bin
    ...
    temperature_0719.bin
```

### Legacy VTK ASCII

- One or more `.vtk` files containing `SCALARS temperature_*` blocks
- Parser extracts all temperature fields from each file
- Useful for legacy PTM outputs before binary format

**Example structure:**
```
vtk_output/
  mesh_temp_0000.vtk
  mesh_temp_0001.vtk
  ...
```

## Output Formats

### CSV (ice_depth.csv)

```csv
face,T_mean,T_max,ice_depth
0,95.234567,120.456789,0.00123456
1,92.123456,118.234567,0.00234567
2,89.876543,115.987654,nan
...
```

- `face`: Face index (0-based)
- `T_mean`: Mean temperature across all timesteps (K)
- `T_max`: Maximum temperature across all timesteps (K)
- `ice_depth`: Ice table depth (m), or `nan` if no solution

### HDF5 (ice_depth.h5)

When `--hdf5` flag is set and `h5py` is available:

```
/ice_depth   [n_faces] float32 (gzip compressed)
/T_mean      [n_faces] float32 (gzip compressed)
/T_max       [n_faces] float32 (gzip compressed)
"""

### VTU + PVD (ParaView visualization)

When `--pvd` is provided (along with `--mesh-vtu <template.vtu>`), the pipeline writes:

```
ice_depth.vtu  # UnstructuredGrid with new CellData arrays:
                             #   ice_depth, ice_T_mean, ice_T_max, (ice_T_min if available)
ice_depth.pvd  # Single timestep collection referencing ice_depth.vtu
```

The template VTU should be a mesh produced by PTM3D (geometry + topology). The pipeline *injects* new scalar arrays into its `<CellData>` block without altering points or cells, enabling seamless visualization alongside other PTM outputs.

**Example:**
```bash
python -m ptm_ice.ice_depth_pipeline \
    --source-dir output/europa_high_res/temperature \
    --format binary \
    --out-dir output/europa_high_res/ice_depth \
    --pvd --mesh-vtu output/europa_high_res/mesh_template.vtu
```

Load `ice_depth.pvd` in ParaView and choose `ice_depth` or `ice_T_mean` from the CellData drop-down.

**Notes:**
- If statistics VTU files (`statistics_avg.vtu`, `statistics_max.vtu`, optional `statistics_min.vtu`) are present, they are used automatically to avoid recomputation.
- Progress bars appear only when `--progress` is passed. Install with `pip install tqdm` for rich display.
- For very large datasets (>1e7 faces), prefer a moderate chunk size (e.g. 1e6) and sufficient threads (but avoid oversubscribing beyond logical CPUs).
- Parallel face accumulation reduces wall time for temperature statistics; disk throughput may still dominate. Consider running from SSDs for best performance.
- Use `--no-statistics` to force raw aggregation from binary/VTK time series.
- Provide `--pvd-time <value>` to set a custom (scalar) time in the PVD for the single result.
"""
```

## Physical Model

The solver computes ice stability depth `z` from the energy balance:

```
f(z) = (2 * z * J0) / (μ * l) - A * p_v(T(z)) / √T(z) = 0
```

where:
- `T(z) = T_avg + ΔT * exp(-z/z_s) + (Q/k) * z` (thermal profile)
- `p_v(T)` is saturation vapor pressure (Clausius-Clapeyron)
- `J0` solar flux, `l` latent heat, `z_s` skin depth, `Q` internal heat, `k` conductivity

See `ice_table.py` and `ice_table_vectorized.py` for detailed physics.

## Performance

| Face count | Backend | Time (approx) | Memory |
|------------|---------|---------------|--------|
| 10³        | numpy   | < 1 s         | < 10 MB |
| 10⁴        | numpy   | ~ 2 s         | ~ 20 MB |
| 10⁵        | numpy   | ~ 20 s        | ~ 50 MB |
| 10⁶        | numba   | ~ 10 s        | ~ 100 MB |
| 10⁷        | numba   | ~ 60 s        | ~ 500 MB |

**Notes:**
- Numba backend provides ~3-5× speedup via parallel JIT
- Memory scales primarily with `chunk_size` (default 500k)
- IO time dominates for small face counts; solver time for large counts

## Testing

```bash
cd src/python
pytest ptm_ice/test_ice_depth_pipeline.py -v
```

**Test coverage:**
- Binary reader mean/max aggregation
- VTK ASCII parser
- End-to-end pipeline
- CSV writer

## Examples

See `ice_depth_pipeline_demo.ipynb` for interactive demonstration:
- Synthetic binary temperature generation
- Pipeline execution
- Results visualization
- Parameter exploration

## Module Reference

### Core Functions

#### `stream_binary_mean_max(source_dir, timesteps, dtype='float32')`
Read binary files and compute mean/max per face.

**Returns:** `(T_mean, T_max)` arrays

#### `stream_vtk_mean_max(source_dir, timesteps=0)`
Parse VTK files and compute mean/max per face.

**Returns:** `(T_mean, T_max)` arrays

#### `compute_ice_depth_from_stats(T_mean, T_max, *, backend='auto', ...)`
Apply ice depth solver to aggregated temperature statistics.

**Returns:** `depths` array (same shape as inputs)

#### `run_pipeline(source_dir, fmt, timesteps, out_dir, **kwargs)`
High-level pipeline orchestrator.

**Returns:** CSV output path

### CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--source-dir` | str | **required** | Directory with temperature files |
| `--format` | str | **required** | Input format: `binary` or `vtk` |
| `--timesteps` | int | 0 | Limit timesteps (0=all) |
| `--dtype` | str | float32 | Binary dtype: `float32` or `float64` |
| `--backend` | str | auto | Solver backend: `numpy`, `numba`, `auto` |
| `--out-dir` | str | **required** | Output directory |
| `--chunk-size` | int | 500000 | Faces per chunk |
| `--hdf5` | flag | False | Write HDF5 output if available |
| `--J0` | float | 1e-5 | Solar flux parameter |
| `--delta` | float | 0.05 | Temperature delta parameter |
| `--l` | float | 2.83e6 | Latent heat (J/kg) |
| `--zs` | float | 4000.0 | Skin depth (m) |
| `--Q` | float | 1e-3 | Internal heat flux (W/m²) |
| `--k` | float | 205.0 | Thermal conductivity (W/m/K) |
| `--no-statistics` | flag | False | Disable use of precomputed statistics VTU files |
| `--pvd` | flag | False | Write `ice_depth.vtu` and `ice_depth.pvd` for ParaView |
| `--mesh-vtu` | str | None | Template mesh VTU path required when using `--pvd` |
| `--pvd-time` | float | timesteps/0 | Custom time value for PVD entry |

## Troubleshooting

**"No temperature_XXXX.bin files found"**
- Check `--source-dir` path
- Verify file naming pattern: `temperature_0000.bin`, `temperature_0001.bin`, etc.

**"Mesh size mismatch"**
- Binary files must all have same length (same face count)
- Check for truncated or corrupted files

**"Numba not available, falling back to numpy"**
- Install `numba` for acceleration: `pip install numba`
- Or use `--backend numpy` explicitly

**"h5py not available"**
- Install h5py: `pip install h5py`
- Or remove `--hdf5` flag to use CSV only

**Large memory usage**
- Reduce `--chunk-size` (e.g., 100000 for ~40 MB chunks)
- Ensure binary files are not accidentally duplicated in memory

## Integration with PTM3D

The module is designed to work with standard PTM3D outputs:

1. Run PTM3D thermal simulation
2. Locate temperature output directory (e.g., `output/europa_high_res/temperature/`)
3. Run pipeline on that directory
4. Ice depths are now available for downstream analysis

**Typical workflow:**
```bash
# 1. Run PTM3D simulation
./build/bin/ptm_thermal_sim europa_high_res.yaml

# 2. Compute ice depths
python -m ptm_ice.ice_depth_pipeline \
    --source-dir output/europa_high_res/temperature \
    --format binary \
    --timesteps 0 \
    --backend auto \
    --out-dir output/europa_high_res/ice_depth \
    --hdf5

# 3. Analyze results
# CSV: output/europa_high_res/ice_depth/ice_depth.csv
# HDF5: output/europa_high_res/ice_depth/ice_depth.h5
```

## References

- Original physics: Notebook `ice_table.ipynb`
- Vectorized solver: `ice_table_vectorized.py`
- C++ PTM3D: `src/cpp/thermal/thermal_model.cpp`

## License

Same as parent PTM3D project.
