# Claude Code Project Notes

## Build System

CMake is located at `/usr/local/bin/cmake` (not in default PATH).

### Quick Build Commands

```bash
# Configure (run from project root)
cd /Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/europa/heat3d/ptm
mkdir -p build && cd build
/usr/local/bin/cmake .. -DCMAKE_BUILD_TYPE=Release

# Build all
/usr/local/bin/cmake --build . -j4

# Build specific target
/usr/local/bin/cmake --build . --target ptm_thermal_sim -j4
```

### Key Build Targets

- `ptm_thermal_sim` - Main thermal simulation executable (in `bin/`)
- `radiation` - Radiation exchange library (view factors, H-matrix)
- `thermal` - Thermal model library
- `benchmark_hmatrix` - H-matrix performance benchmark
- `test_bowl_crater` - Bowl crater radiation test

## Project Structure

```
src/cpp/
├── main/ptm_thermal_sim.cpp    # Main driver (~6500 lines, monolithic)
├── radiation/                   # View factors, H-matrix, radiosity solver
│   ├── radiation_config.h       # RadiationConfig struct
│   ├── radiation_exchange.h/cpp # RadiationExchangeManager
│   ├── view_factor.h/cpp        # Sparse view factor matrix
│   └── hmatrix_view_factor.h/cpp # H-matrix compressed view factors
├── thermal/                     # 1D heat conduction
│   ├── thermal_model.h/cpp      # Model, Profile, Config classes
│   ├── spectral_equilibration.h/cpp # FFT-based equilibration solver
│   └── planetary_properties.h   # Planet class with thermal properties
├── gpu/                         # GPU acceleration (optional, Phase 1 complete)
│   ├── gpu_config.h             # Backend detection (HIP/CUDA/OpenCL)
│   ├── gpu_memory.h             # SoA buffer management
│   ├── gpu_utils.h              # Error handling, timing
│   ├── thermal_gpu.h/cpp        # HIP/CUDA thermal solver (Phase 2: kernels)
│   ├── thermal_kernels.cpp      # HIP/CUDA kernel implementations
│   ├── opencl_config.h          # OpenCL context and device management
│   ├── thermal_opencl.h/cpp     # OpenCL thermal solver
│   ├── thermal_kernels.cl       # OpenCL kernel source
│   └── CMakeLists.txt           # HIP/CUDA/OpenCL detection and build
├── illumination/                # Solar flux with SPICE ephemeris
│   └── illumination_model.h/cpp # IlluminationModel class
├── mesh/                        # Mesh I/O and utilities
└── tools/                       # Config parser, utilities
    └── config_parser.h/cpp      # SimulationConfig struct
```

## Key Classes and Where to Find Them

| Class | File | Purpose |
|-------|------|---------|
| `SimulationConfig` | `tools/config_parser.h` | All command-line/YAML config options |
| `RadiationConfig` | `radiation/radiation_config.h` | Radiation exchange parameters |
| `RadiationExchangeManager` | `radiation/radiation_exchange.h` | High-level radiation API |
| `HMatrixViewFactor` | `radiation/hmatrix_view_factor.h` | H-matrix compressed view factors |
| `thermal::Model` | `thermal/thermal_model.h` | 1D thermal conduction solver |
| `thermal::Profile` | `thermal/thermal_model.h` | Depth-temperature profile |
| `IlluminationModel` | `illumination/illumination_model.h` | SPICE-based solar flux |

## Important Code Locations in ptm_thermal_sim.cpp

| Line Range | Function/Section |
|------------|------------------|
| ~570-600 | Forward declarations |
| ~1028-1133 | `runThermalModelWithMatchedTimes()` |
| ~1135-1242 | `runThermalModelWithWarmStart()` (warm-start optimization) |
| ~1942-2266 | `runThermalModelsForAllFaces()` |
| ~2271-2389 | `runThermalModelsForSelectedFaces()` (selective re-run optimization) |
| ~6032-6080 | Radiation exchange config setup |
| ~6118-6330 | Radiation iteration loop (thermal-radiation coupling) |

## Command-Line Flags

### Radiation Exchange
- `--radiation-exchange` - Enable radiation exchange
- `--rad-hmatrix` - Use H-matrix compression
- `--rad-view-factor-file FILE` - Save/load view factors
- `--rad-verbose` - Print detailed progress
- `--rad-coupling-iter N` - Max radiation-thermal coupling iterations (default: 10, adaptive)
- `--rad-coupling-temp-tol K` - Temperature convergence threshold in K (default: 0.5)
- `--rad-coupling-flux-tol W` - Flux change threshold in W/m² (default: 0.1)
- `--rad-coupling-face-frac F` - Fraction of faces still changing to stop (default: 0.01 = 1%)

### Thermal
- `--thermal-threads N` - Number of OpenMP threads
- `--equilibration-steps N` - Equilibration time steps

## User Preferences

- User has AMD GPU (not NVIDIA), so OptiX is not available
- GPU acceleration deferred; focus on CPU optimizations
- Prefers optimizations enabled by default

## Recent Optimizations (January 2026)

### Radiation-Thermal Coupling (ptm_thermal_sim.cpp:6118-6500)
1. **Adaptive iteration count**: Loop terminates early when:
   - Temperature changes < `coupling_temp_threshold` (default 0.5 K)
   - < `coupling_face_fraction` (default 1%) of faces still changing
   - No faces need update
   - Max iterations from `max_coupling_iterations` (default 10)

2. **Selective thermal re-run**: Only recomputes faces where indirect flux changed > threshold

3. **Flux-based first iteration prediction**: On first iteration, estimates which faces will see significant temperature changes based on ratio of indirect flux to total absorbed flux. Faces where indirect < 1% of total are skipped.

4. **Warm-start thermal solver**: Uses previous iteration temperatures as initial condition

All optimizations are enabled by default. Thresholds configurable via command-line flags.

**Important**: When radiation exchange is enabled (`--radiation-exchange`), full illumination arrays are required for iterative coupling. Streaming mode is automatically disabled and illumination data is extracted to RAM, which may require significant memory (~15 GB for 600K faces × 1000 timesteps).

5. **Progress bar**: Each radiation iteration shows a progress bar for indirect flux computation across timesteps

### Thermal Model (thermal_model.cpp)
- **Heat capacity polynomial**: Uses Horner's method with SIMD (AVX-512/AVX2) for efficient cp(T) evaluation
- **Fused material updates**: Combined k(T), cp(T), and kappa updates use FMA instructions
- **FMA (Fused Multiply-Add)**: Replaces separate mul+add in SIMD paths for better performance and accuracy
- Grid coupling coefficients (g1, g2) pre-computed once during profile initialization
- Thread-local memory pools for Thomas algorithm buffers

### Spectral Equilibration (spectral_equilibration.cpp)
- **FFT-based analytical solver**: Replaces thousands of diurnal time-stepping cycles with ~3-5 iterations
- **Physics**: Decomposes surface flux S(t) into Fourier harmonics, solves heat equation analytically for each
- **Skin depth scaling**: Each harmonic n attenuates with δ_n = δ_1/√n
- **Iterative refinement**: Updates linearized radiative BC and effective properties until convergence
- **Config options**:
  - `USE_SPECTRAL_EQUILIBRATION = true` - Enable spectral equilibration
  - `SPECTRAL_HARMONICS = 16` - Number of Fourier modes
  - `SPECTRAL_MAX_ITER = 5` - Max property-update iterations

### GPU Acceleration Module (gpu/) - Phase 3 Complete
GPU-accelerated thermal solver with register-based Thomas algorithm.

**Phase 1 (Infrastructure)**:
- **Backend abstraction** (`gpu_config.h`): Portable HIP/CUDA/OpenCL API with auto-detection
- **SoA memory layout** (`gpu_memory.h`): Structure-of-Arrays buffers for coalesced GPU access
- **Utility classes** (`gpu_utils.h`): Error handling, timing, memory estimation
- **Solver interface** (`thermal_gpu.h`): `ThermalGPUSolver` class with CPU fallback

**Phase 2 (Kernels)** - `thermal_kernels.cpp`:
- **`combinedThermalKernel`**: Main solver - material update + Crank-Nicolson + Thomas algorithm
- **`materialUpdateKernel`**: k(T), cp(T), kappa updates with Horner's method
- **`thermalSolverKernel`**: Separate solver kernel for flexibility
- **Register-based TDMA**: Entire depth column (up to 64 layers) kept in GPU registers
- **Linearized radiative BC**: 4εσT³ approximation avoids Newton iterations on GPU

**OpenCL Backend** (works on macOS with AMD GPUs):
- **`opencl_config.h`**: OpenCL context, device detection, error handling
- **`thermal_opencl.h/cpp`**: `ThermalOpenCLSolver` class with same interface as HIP/CUDA
- **`thermal_kernels.cl`**: OpenCL kernel source (also embedded in .cpp for distribution)
- Auto-detects platforms/devices, prefers GPU over CPU
- Uses private memory (registers) for Thomas algorithm arrays

**Phase 3 (Integration)**:
- Main executable integration in `ptm_thermal_sim.cpp`
- `runThermalModelsForAllFaces_GPU()` handles both HIP/CUDA and OpenCL backends
- Conditional compilation selects appropriate solver at build time
- Automatic fallback to CPU if GPU unavailable at runtime

**Build options**:
```bash
cmake -DENABLE_GPU=ON -DGPU_BACKEND=AUTO ..    # Auto-detect (HIP > CUDA > OpenCL)
cmake -DENABLE_GPU=ON -DGPU_BACKEND=HIP ..     # Force HIP backend
cmake -DENABLE_GPU=ON -DGPU_BACKEND=CUDA ..    # Force CUDA backend
cmake -DENABLE_GPU=ON -DGPU_BACKEND=OPENCL ..  # Force OpenCL backend (macOS)
```

**Runtime flags**:
- `--use-gpu-thermal` - Enable GPU thermal solver (warns if not compiled with GPU support)
- `--use-gpu-illumination` - Enable GPU ray tracing for illumination (experimental)
- `--use-gpu-both` - Enable GPU for both thermal and illumination
- `--rad-use-gpu` - Enable GPU radiosity solver for radiation exchange

**Integration**: `runThermalModelsForAllFaces_GPU()` in ptm_thermal_sim.cpp (~lines 2539-2760)

### GPU Radiosity Solver (gpu/radiosity_opencl.cpp) - Implemented
GPU-accelerated radiosity solver using sparse matrix-vector multiply (SpMV) for view factor operations.

**Files**:
- `radiosity_opencl.h` - RadiosityOpenCLSolver class declaration
- `radiosity_opencl.cpp` - Implementation with embedded kernel source
- `radiosity_kernels.cl` - Standalone kernel file (for reference)

**Kernels**:
- `spmv_kernel` - CSR sparse matrix-vector multiply
- `emission_kernel` - Thermal emission (ε σ T^4)
- `solar_source_kernel` - Reflected solar (ρ Q_solar)
- `jacobi_kernel` - Jacobi iteration update with convergence tracking
- `indirect_flux_kernel` - Compute absorbed indirect flux

**Implementation Notes**:
- Avoids `CL_CHECK` macro (from `opencl_config.h`) which can cause SIGILL crashes on macOS
  when exceptions cross shared library boundaries with aggressive optimization
- Uses inline lambda-based error checking instead for all OpenCL calls
- GPU radiosity validated: 0.00014% relative error vs CPU solver on test cases

### GPU BVH Ray Tracer (gpu/bvh_opencl.cpp) - Implemented
GPU-accelerated shadow ray tracing for illumination model using custom SAH-built BVH.

**Files**:
- `bvh_opencl.h` - BVHBuilder and BVHRayTracer class declarations
- `bvh_opencl.cpp` - Implementation with embedded OpenCL kernel

**Features**:
- CPU-side BVH construction using Surface Area Heuristic (SAH)
- SoA (Structure-of-Arrays) memory layout for coalesced GPU access
- Stack-based BVH traversal (32-deep stack in private memory)
- Möller-Trumbore ray-triangle intersection
- Slab method for ray-box intersection
- BVH serialization to disk for reuse

**Implementation Notes**:
- Uses non-throwing error macros (`CL_CHECK_RETURN`) instead of `CL_CHECK` to avoid
  SIGILL crashes when exceptions cross shared library boundaries on macOS
- Thread-safe via mutex (gpu_ray_tracer_mutex_) - serializes all GPU ray trace calls
- Falls back to CPU Embree ray tracing on error

**Batched Ray Tracing** (January 2026):
The GPU illumination now uses a three-phase batched approach via `calculateDirectSolarBatchWithFiniteDiskGPU()`:
1. **Phase 1**: Generate all rays for all faces in parallel (CPU, OpenMP)
   - Pre-compute sun directions, ray origins, and disk samples per face
   - Filter eclipse-blocked rays on CPU (no GPU dispatch needed)
   - Compact rays into SoA buffers
2. **Phase 2**: Dispatch all rays to GPU in single kernel launch
   - Single `traceShadowRays()` call with all rays
   - Massive parallelism on GPU (all rays processed simultaneously)
3. **Phase 3**: Process results in parallel (CPU, OpenMP)
   - Distribute occlusion results back to per-face flux calculations

**Performance Notes**:
- For small meshes (<100K faces), CPU Embree is typically faster due to lower overhead
- GPU becomes beneficial for very large meshes or when CPU is already busy
- Results match CPU Embree within ~0.001% (float vs double precision difference)
- The batched approach provides ~250x speedup over per-face GPU dispatch

**Runtime flag**: `--use-gpu-illumination` (experimental)

### H-Matrix View Factors (hmatrix_view_factor.cpp)
- Randomized SVD for faster compression (default)
- **AVX-512 16-wide ray packets** (auto-detected at runtime, falls back to AVX2 8-wide)
- Batched ray queries via Embree `rtcIntersect16`/`rtcIntersect8`
- Cluster-level backface culling
- Binary serialization for save/load (.hmat format, magic=0x484D4154)

**Runtime flag**: `--rad-hmatrix` (enables H-matrix compression for large meshes)

### Sparse View Factor Format (view_factor.cpp)
- Magic number 0x5646464D ("VFFM") added for format detection
- Both formats auto-detected by `RadiationExchangeManager::loadViewFactors()`

## External Dependencies

- **Embree 4**: Ray tracing (Intel, CPU-only)
- **SPICE/CSPICE**: Ephemeris data (`/Users/paha3326/software/cspice/`)
- **Eigen 3**: Linear algebra
- **CFITSIO**: FITS file I/O
- **OpenMP**: Parallelization

## Testing

```bash
# Run H-matrix benchmark
./bin/benchmark_hmatrix

# Run bowl crater test
./bin/test_bowl_crater
```

## Documentation

### Design Documents (docs/)

- **[thermal-efficiency-plan.md](docs/thermal-efficiency-plan.md)** - Original GPU acceleration technical spec
  - Phase A: Spectral initialization (FFT-based equilibration)
  - Phase B: Batched Crank-Nicolson transient solver
  - SoA memory layout, HIP portability strategy

- **[gpu-implementation-plan.md](docs/gpu-implementation-plan.md)** - Detailed GPU implementation roadmap
  - Phase 1: Infrastructure (SoA buffers, CMake integration)
  - Phase 2: Register-heavy TDMA kernel for thermal solver
  - Phase 3: Integration with existing CPU code
  - Phase 4: Validation and benchmarking
  - Includes expected speedups (50-100x for large meshes)
