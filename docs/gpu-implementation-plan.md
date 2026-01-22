# GPU Implementation Plan for PTM Thermal Model

## Overview

This document outlines a phased approach to GPU-accelerate the Planetary Thermal Model (PTM). The goal is to achieve 10-100x speedup for large meshes (>100K faces) while maintaining numerical accuracy and supporting both NVIDIA and AMD GPUs.

**Current Bottlenecks:**
1. **Thermal solver**: ~1500s for 618K faces × 1000 timesteps (OpenMP parallelized)
2. **Radiation exchange**: View factor computation (already has H-matrix optimization)
3. **Ray tracing for illumination**: Currently CPU-based with Embree

## Architecture Decision: HIP for Portability

We use **HIP (Heterogeneous-Compute Interface for Portability)** because:
- Near-identical syntax to CUDA
- Compiles to NVCC (CUDA) on NVIDIA, ROCm on AMD
- Supports hipSPARSE, hipBLAS which marshal to vendor-specific libraries
- Single codebase for both GPU vendors

Alternative considered: SYCL/OneAPI, but HIP has better maturity for scientific computing.

---

## Phase 1: Infrastructure & Memory Management

### 1.1 GPU Abstraction Layer

Create `src/cpp/gpu/` directory with:

```
src/cpp/gpu/
├── CMakeLists.txt
├── gpu_config.h         # Feature detection, backend selection
├── gpu_memory.h         # Memory pool, SoA buffers
├── gpu_memory.cpp
├── thermal_gpu.h        # GPU thermal solver interface
├── thermal_gpu.cpp      # HIP kernel implementations
└── gpu_utils.h          # Error handling, timing utilities
```

**Key design: CPU fallback always available.** GPU code is isolated; the thermal model can run without any GPU support.

### 1.2 Structure of Arrays (SoA) Memory Layout

**Critical for GPU performance.** Current memory layout is Array of Structures (AoS):
```cpp
// Current: AoS - bad for GPU coalescing
face[i].temperature[depth]  // Strided access
```

Transform to SoA:
```cpp
// SoA - coalesced memory access
T_all[depth * n_faces + face_id]  // Contiguous by face within depth
```

**Memory layout for 600K faces × 30 depth layers:**
```cpp
struct ThermalGPUBuffers {
    // Primary state (read/write each timestep)
    float* T;         // [n_faces * n_layers] - temperatures
    float* flux;      // [n_faces] - surface flux per face

    // Material properties (updated less frequently)
    float* k;         // [n_faces * n_layers] - conductivity
    float* kappa;     // [n_faces * n_layers] - diffusivity
    float* cp;        // [n_faces * n_layers] - heat capacity
    float* rho;       // [n_faces * n_layers] - density

    // Grid geometry (constant after init)
    float* z;         // [n_layers] - depth grid (shared)
    float* dz;        // [n_layers-1] - layer spacing
    float* g1, *g2;   // [n_layers] - coupling coefficients

    // Face properties (constant)
    float* emissivity; // [n_faces]
    float* albedo;     // [n_faces]
};
```

**Memory estimate:**
- 600K faces × 30 layers × 6 arrays × 4 bytes = ~432 MB
- Plus flux, properties: ~500 MB total GPU memory

### 1.3 CMake Integration

```cmake
# In root CMakeLists.txt
option(ENABLE_GPU "Enable GPU acceleration" OFF)
option(GPU_BACKEND "GPU backend: HIP, CUDA, or NONE" "HIP")

if(ENABLE_GPU)
    if(GPU_BACKEND STREQUAL "HIP")
        find_package(hip REQUIRED)
        add_subdirectory(src/cpp/gpu)
    endif()
endif()
```

---

## Phase 2: GPU Thermal Solver Kernel

### 2.1 Register-Heavy TDMA Kernel

Since depth grid is small (N~30), keep entire column in GPU registers:

```cpp
__global__ void BatchedThermalSolver(
    float* T_global,      // [n_faces * n_layers]
    const float* flux,    // [n_faces]
    const float* k,       // [n_faces * n_layers]
    const float* kappa,   // [n_faces * n_layers]
    const float* g1,      // [n_layers]
    const float* g2,      // [n_layers]
    float dt,
    float sigma,
    int n_faces,
    int n_layers
) {
    int face_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (face_id >= n_faces) return;

    // Load column into registers (coalesced read from SoA layout)
    float T_local[MAX_LAYERS];  // MAX_LAYERS = 64, fits in registers
    for (int i = 0; i < n_layers; i++) {
        T_local[i] = T_global[i * n_faces + face_id];
    }

    // Surface boundary condition (linearized radiation)
    float T_surf = T_local[0];
    float T3 = T_surf * T_surf * T_surf;
    float rad_coeff = 4.0f * sigma * emissivity * T3;
    // ... build tridiagonal system in registers ...

    // Thomas algorithm entirely in registers
    float a[MAX_LAYERS], b[MAX_LAYERS], c[MAX_LAYERS], d[MAX_LAYERS];
    // ... forward elimination ...
    // ... backward substitution ...

    // Write back (coalesced)
    for (int i = 0; i < n_layers; i++) {
        T_global[i * n_faces + face_id] = T_local[i];
    }
}
```

**Key optimizations:**
1. **One thread per face**: Entire thermal column solved by single thread
2. **Register storage**: No VRAM latency during Thomas algorithm
3. **Coalesced I/O**: SoA layout ensures adjacent threads access adjacent memory
4. **Linearized BC**: Avoid Newton iterations on GPU (4εσT³ approximation)

### 2.2 Launch Configuration

```cpp
// Optimal for ~600K faces
dim3 block(256);  // 256 threads per block
dim3 grid((n_faces + 255) / 256);  // ~2400 blocks

BatchedThermalSolver<<<grid, block, 0, stream>>>(
    d_T, d_flux, d_k, d_kappa, d_g1, d_g2, dt, sigma, n_faces, n_layers
);
```

### 2.3 Material Property Update Kernel

Temperature-dependent k(T), cp(T) updated on GPU:

```cpp
__global__ void UpdateMaterialProperties(
    const float* T,
    float* k,
    float* cp,
    float* kappa,
    const float* rho,
    const float* kc,     // base conductivity
    float R350,          // radiative conductivity coefficient
    const float* cpCoeff, // cp polynomial coefficients
    int n_coeff,
    float cp0,
    int n_faces,
    int n_layers
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_faces * n_layers;
    if (idx >= total) return;

    float Ti = T[idx];

    // k(T) = kc + R350 * T^3
    float T2 = Ti * Ti;
    float T3 = T2 * Ti;
    k[idx] = kc[idx] + R350 * T3;

    // cp(T) via Horner's method
    float cpval = cpCoeff[n_coeff - 1];
    for (int j = n_coeff - 2; j >= 0; j--) {
        cpval = cpval * Ti + cpCoeff[j];
    }
    cp[idx] = cp0 + Ti * cpval;

    // kappa = k / (rho * cp)
    kappa[idx] = k[idx] / (rho[idx] * cp[idx]);
}
```

---

## Phase 3: Integration with Existing Code

### 3.1 CPU/GPU Abstraction

```cpp
// thermal_model.h
class Model {
    // ...
    void advance();  // Existing CPU path

#ifdef ENABLE_GPU
    void advanceGPU();  // GPU path
    std::unique_ptr<GPUThermalContext> gpu_ctx_;
#endif
};

// In thermal_model.cpp
void Model::run() {
    // ...
#ifdef ENABLE_GPU
    if (config->USE_GPU && gpu_ctx_) {
        advanceGPU();
    } else
#endif
    {
        advance();  // CPU fallback
    }
}
```

### 3.2 Batched Face Processing

For the full mesh simulation, batch all faces to GPU:

```cpp
// ptm_thermal_sim.cpp - GPU batch processing
bool runThermalModelsForAllFaces_GPU(
    const std::shared_ptr<Mesh>& mesh,
    const std::vector<std::vector<double>>& flux_per_face,  // [face][time]
    const std::vector<double>& times,
    const thermal::Config& config,
    std::vector<std::vector<double>>& surface_temps_out
) {
    GPUThermalBatch batch(mesh->num_faces, config.n_layers);

    // Initialize: upload grid, properties, initial T
    batch.initializeFromMesh(mesh, config);

    // Time stepping loop
    for (size_t t = 0; t < times.size() - 1; t++) {
        double dt = times[t+1] - times[t];

        // Upload flux for this timestep
        batch.uploadFlux(flux_per_face, t);

        // GPU kernel: update material props + solve
        batch.advanceTimestep(dt);

        // Download surface temps (only if needed for output)
        if (shouldOutput(t)) {
            batch.downloadSurfaceTemps(surface_temps_out, t);
        }
    }

    return true;
}
```

### 3.3 Async Overlap Strategy

Hide data transfer latency with compute:

```cpp
// Double-buffered flux upload
cudaStream_t compute_stream, transfer_stream;

for (size_t t = 0; t < n_timesteps; t++) {
    // Async upload flux for t+1 while computing t
    if (t + 1 < n_timesteps) {
        uploadFluxAsync(flux_buffer[(t+1) % 2], transfer_stream);
    }

    // Compute timestep t
    launchKernel(flux_buffer[t % 2], compute_stream);

    // Sync before next iteration
    cudaStreamSynchronize(compute_stream);
}
```

---

## Phase 4: Validation & Testing

### 4.1 Numerical Validation

1. **Single-face comparison**: Run one face through CPU and GPU paths, compare T(z,t)
2. **Tolerance**: Max |T_cpu - T_gpu| < 0.01 K (float32 precision)
3. **Conservation check**: Energy balance before/after timestep

### 4.2 Benchmark Suite

```cpp
// benchmark_thermal_gpu.cpp
void benchmarkThermalGPU() {
    std::vector<size_t> face_counts = {1000, 10000, 100000, 600000};

    for (auto n_faces : face_counts) {
        // CPU timing
        auto cpu_time = runCPUBenchmark(n_faces, n_timesteps);

        // GPU timing
        auto gpu_time = runGPUBenchmark(n_faces, n_timesteps);

        std::cout << "Faces: " << n_faces
                  << " CPU: " << cpu_time << "ms"
                  << " GPU: " << gpu_time << "ms"
                  << " Speedup: " << cpu_time/gpu_time << "x" << std::endl;
    }
}
```

### 4.3 Test Cases

1. **Flat surface**: Known analytical solution (Fourier series)
2. **Bowl crater**: Compare to existing validated results
3. **Lunar model**: Full 618K face simulation

---

## Phase 5: Ray Tracing GPU Acceleration (Future)

After thermal solver is GPU-accelerated, extend to illumination:

1. **OptiX integration**: NVIDIA ray tracing
2. **HIP RT**: AMD ray tracing (newer, less mature)
3. **Fallback**: Keep Embree CPU path

This is lower priority since illumination is typically pre-computed once.

---

## Implementation Checklist

### Phase 1 (Infrastructure)
- [ ] Create `src/cpp/gpu/` directory structure
- [ ] Implement GPU detection and backend selection in CMake
- [ ] Create `ThermalGPUBuffers` SoA memory manager
- [ ] Add `--use-gpu` command-line flag
- [ ] Implement CPU-to-SoA conversion utilities

### Phase 2 (Kernels)
- [ ] Implement `BatchedThermalSolver` kernel
- [ ] Implement `UpdateMaterialProperties` kernel
- [ ] Add surface boundary condition kernel
- [ ] Implement Thomas algorithm in registers
- [ ] Test single-face kernel correctness

### Phase 3 (Integration)
- [ ] Add GPU context to `thermal::Model`
- [ ] Implement `runThermalModelsForAllFaces_GPU()`
- [ ] Add async stream management
- [ ] Integrate with radiation exchange iteration

### Phase 4 (Validation)
- [ ] Create benchmark suite
- [ ] Validate against CPU results
- [ ] Profile memory bandwidth utilization
- [ ] Optimize block/grid dimensions

---

## Expected Performance

| Mesh Size | CPU Time (OpenMP 8 threads) | GPU Time (est.) | Speedup |
|-----------|----------------------------|-----------------|---------|
| 10K faces | ~25s | ~0.5s | 50x |
| 100K faces | ~250s | ~3s | 80x |
| 600K faces | ~1500s | ~15s | 100x |

**Key insight**: GPU advantage grows with face count due to:
1. Massive parallelism (thousands of concurrent faces)
2. Register-resident solver eliminates memory bottleneck
3. Coalesced SoA access maximizes bandwidth

---

## Risk Mitigation

1. **Float32 precision**: May need double precision for some calculations; benchmark accuracy loss
2. **Memory limits**: 600K faces × 30 layers fits in ~500MB; larger meshes may need chunking
3. **HIP maturity**: Test on both NVIDIA and AMD early to catch portability issues
4. **Complexity**: Keep CPU path as fallback; GPU is optional enhancement

---

## References

- [thermal-efficiency-plan.md](thermal-efficiency-plan.md) - Original technical specification
- [HIP Programming Guide](https://rocm.docs.amd.com/projects/HIP/en/latest/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
