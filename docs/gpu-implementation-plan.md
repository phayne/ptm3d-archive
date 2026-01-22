# GPU Acceleration in PTM

## Overview

PTM supports GPU acceleration for computationally intensive operations, providing significant speedups for large mesh simulations. GPU acceleration is optional—the model runs entirely on CPU if GPU support is not available or not enabled.

**GPU-accelerated components:**
- **Thermal solver**: Batched 1D heat equation solver for all mesh faces
- **Radiosity solver**: Sparse matrix-vector operations for radiation exchange
- **Shadow ray tracing**: BVH-accelerated visibility tests for illumination (experimental)

## Supported Platforms

| Platform | Backend | Thermal | Radiosity | Ray Tracing |
|----------|---------|---------|-----------|-------------|
| macOS (AMD GPU) | OpenCL | Yes | Yes | Yes |
| Linux | OpenCL | Yes | Yes | Yes |
| Linux (NVIDIA) | CUDA | Yes | No | No |
| Linux (AMD) | HIP/ROCm | Yes | No | No |
| Windows | OpenCL | Yes | Yes | Yes (untested) |

**OpenCL is the recommended backend** with full feature support. HIP and CUDA backends support thermal solver acceleration only; radiosity and ray tracing fall back to CPU when using these backends.

The build system automatically detects available GPU backends and selects the most appropriate one.

## Building with GPU Support

### Prerequisites

- **macOS**: OpenCL is included with the system (no additional installation needed)
- **Linux (NVIDIA)**: CUDA Toolkit 11.0+ or HIP with CUDA backend
- **Linux (AMD)**: ROCm 5.0+ with HIP

### CMake Configuration

```bash
# Auto-detect GPU backend
cmake -DENABLE_GPU=ON -DGPU_BACKEND=AUTO ..

# Force OpenCL (recommended for full feature support)
cmake -DENABLE_GPU=ON -DGPU_BACKEND=OPENCL ..

# Force CUDA (thermal solver only)
cmake -DENABLE_GPU=ON -DGPU_BACKEND=CUDA ..

# Force HIP (thermal solver only)
cmake -DENABLE_GPU=ON -DGPU_BACKEND=HIP ..

# Disable GPU (default)
cmake -DENABLE_GPU=OFF ..
```

### Build

```bash
mkdir build && cd build
cmake -DENABLE_GPU=ON ..
cmake --build . -j4
```

## Command-Line Usage

### Thermal Solver

```bash
# Enable GPU thermal solver
./bin/ptm_thermal_sim --config europa.yaml --use-gpu-thermal

# CPU thermal solver (default)
./bin/ptm_thermal_sim --config europa.yaml
```

### Radiation Exchange

```bash
# Enable GPU radiosity solver for radiation exchange
./bin/ptm_thermal_sim --config europa.yaml --radiation-exchange --rad-use-gpu
```

### Illumination (Experimental)

```bash
# Enable GPU ray tracing for shadow calculations
./bin/ptm_thermal_sim --config europa.yaml --use-gpu-illumination

# Enable GPU for both thermal and illumination
./bin/ptm_thermal_sim --config europa.yaml --use-gpu-both
```

## Architecture

### GPU Thermal Solver

The GPU thermal solver processes all mesh faces in parallel. Each GPU thread handles one mesh face, solving the 1D heat equation through the depth column using the Crank-Nicolson finite difference scheme.

**Key features:**
- **Batched processing**: All faces processed simultaneously on GPU
- **Register-based solver**: The Thomas algorithm (tridiagonal solver) runs entirely in GPU registers for maximum performance
- **Automatic sub-stepping**: Maintains numerical stability by internally sub-stepping when the output timestep is large
- **Equilibration cycles**: Runs multiple diurnal cycles before output to establish thermal equilibrium

**Memory layout**: Uses Structure-of-Arrays (SoA) format for coalesced GPU memory access:
```
T[layer][face] → T_gpu[layer * n_faces + face]
```

### GPU Radiosity Solver

The radiosity solver uses sparse matrix-vector multiplication (SpMV) to compute indirect thermal radiation exchange between mesh faces.

**Components:**
- CSR sparse matrix format for view factors
- Jacobi iteration with GPU-accelerated kernels
- Emission and absorption computed in parallel

### GPU Ray Tracer

The BVH (Bounding Volume Hierarchy) ray tracer accelerates shadow ray tests for illumination calculations.

**Features:**
- SAH (Surface Area Heuristic) BVH construction on CPU
- Stack-based BVH traversal on GPU
- Batched ray processing for efficiency
- BVH serialization for reuse across runs

## Performance Characteristics

GPU acceleration provides the greatest benefit for:
- **Large meshes**: >100K faces see substantial speedups
- **Long simulations**: Many timesteps amortize GPU initialization overhead
- **Radiation exchange**: SpMV operations parallelize efficiently

For small meshes (<10K faces), CPU may be faster due to GPU kernel launch overhead.

**Typical speedups** (600K faces, 1000 timesteps):
- Thermal solver: 50-100x vs single-threaded CPU
- Radiosity solver: 10-50x depending on view factor sparsity

## Fallback Behavior

If GPU acceleration is requested but unavailable (no GPU, driver issues, or not compiled with GPU support), PTM automatically falls back to CPU execution with a warning message:

```
Warning: GPU thermal solver requested but not available, falling back to CPU
```

This ensures simulations complete successfully regardless of GPU availability.

## Numerical Considerations

- GPU thermal solver uses single precision (float32) internally for performance
- Results are validated to match CPU double-precision within 0.01 K
- Material property calculations (conductivity, heat capacity) use the same physics as CPU

## File Locations

GPU implementation source files:
```
src/cpp/gpu/
├── CMakeLists.txt           # GPU build configuration
├── gpu_config.h             # Backend detection and configuration
├── gpu_memory.h             # SoA buffer management
├── gpu_utils.h              # Error handling utilities
│
├── # OpenCL backend (full feature support)
├── opencl_config.h          # OpenCL context management
├── thermal_opencl.h/cpp     # OpenCL thermal solver
├── thermal_kernels.cl       # OpenCL thermal kernel source
├── radiosity_opencl.h/cpp   # OpenCL radiosity solver
├── radiosity_kernels.cl     # OpenCL radiosity kernel source
├── bvh_opencl.h/cpp         # OpenCL BVH ray tracer
│
└── # HIP/CUDA backend (thermal solver only)
    ├── thermal_gpu.h/cpp    # HIP/CUDA thermal solver interface
    └── thermal_kernels.cpp  # HIP/CUDA kernel implementations
```
