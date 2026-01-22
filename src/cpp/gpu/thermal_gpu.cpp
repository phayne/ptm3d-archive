/**
 * @file thermal_gpu.cpp
 * @brief GPU thermal solver implementation
 *
 * This file contains:
 * - CPU fallback implementations (always available)
 * - GPU kernel launch wrappers (when PTM_GPU_ENABLED)
 *
 * Phase 2: Full kernel implementations in thermal_kernels.cpp
 */

#include "thermal_gpu.h"
#include "thermal_kernels.h"
#include <iostream>
#include <cmath>
#include <algorithm>

namespace gpu {

// ============================================================================
// ThermalGPUSolver implementation
// ============================================================================

ThermalGPUSolver::ThermalGPUSolver(size_t n_faces, size_t n_layers,
                                   const ThermalGPUConfig& config)
    : n_faces_(n_faces)
    , n_layers_(n_layers)
    , config_(config)
{
}

ThermalGPUSolver::~ThermalGPUSolver() {
#if PTM_GPU_ENABLED
    for (auto& stream : streams_) {
        if (stream) {
            gpuStreamDestroy(stream);
        }
    }
#endif
}

bool ThermalGPUSolver::initialize(const GPUConfig& gpu_config) {
    if (initialized_) {
        return true;
    }

    bool use_gpu = gpu_config.isAvailable();

    if (use_gpu) {
        // Check memory requirements
        size_t required = estimateThermalMemory(n_faces_, n_layers_);
        if (required > gpu_config.available_memory) {
            if (config_.verbose) {
                std::cerr << "Warning: Mesh requires " << (required / (1024*1024))
                          << " MB but only " << (gpu_config.available_memory / (1024*1024))
                          << " MB available. Using CPU fallback." << std::endl;
            }
            use_gpu = false;
        }
    }

    // Allocate buffers
    buffers_.allocate(n_faces_, n_layers_, use_gpu);

#if PTM_GPU_ENABLED
    if (use_gpu) {
        // Create streams for async operations
        streams_.resize(config_.streams);
        for (int i = 0; i < config_.streams; ++i) {
            GPU_CHECK(gpuStreamCreate(&streams_[i]));
        }
    }
#endif

    // Create timer
    timer_ = std::make_unique<GPUTimer>();

    initialized_ = true;

    if (config_.verbose) {
        std::cout << "ThermalGPUSolver initialized:"
                  << " faces=" << n_faces_
                  << ", layers=" << n_layers_
                  << ", GPU=" << (use_gpu ? "yes" : "no (CPU fallback)")
                  << ", memory=" << (buffers_.totalBytes() / (1024*1024)) << " MB"
                  << std::endl;
    }

    return true;
}

void ThermalGPUSolver::uploadMeshData(
    const std::vector<float>& z_grid,
    const std::vector<float>& dz,
    const std::vector<float>& g1,
    const std::vector<float>& g2,
    const std::vector<float>& rho,
    const std::vector<float>& kc,
    const std::vector<float>& emissivity,
    const std::vector<float>& albedo,
    const std::vector<float>& T_deep,
    const std::vector<float>& Qb)
{
    if (!initialized_) {
        throw std::runtime_error("ThermalGPUSolver not initialized");
    }

    // Upload grid geometry
    buffers_.z.copyFromHost(z_grid);
    buffers_.dz.copyFromHost(dz);
    buffers_.g1.copyFromHost(g1);
    buffers_.g2.copyFromHost(g2);

    // Upload per-face/layer properties
    buffers_.rho.copyFromHost(rho);
    buffers_.kc.copyFromHost(kc);

    // Upload surface properties
    buffers_.emissivity.copyFromHost(emissivity);
    buffers_.albedo.copyFromHost(albedo);

    // Upload boundary conditions
    buffers_.T_deep.copyFromHost(T_deep);
    buffers_.Qb.copyFromHost(Qb);
}

void ThermalGPUSolver::setMaterialParams(
    float cp0,
    const std::vector<float>& cp_coeffs,
    float R350,
    float sigma)
{
    buffers_.cp0 = cp0;
    buffers_.R350 = R350;
    buffers_.sigma = sigma;
    buffers_.n_cp_coeffs = static_cast<int>(cp_coeffs.size());

    if (!cp_coeffs.empty()) {
        buffers_.cp_coeffs = DeviceBuffer<float>(cp_coeffs.size(), buffers_.isOnGPU());
        buffers_.cp_coeffs.copyFromHost(cp_coeffs);
    }
}

void ThermalGPUSolver::uploadInitialTemperature(const std::vector<float>& T_initial) {
    if (!initialized_) {
        throw std::runtime_error("ThermalGPUSolver not initialized");
    }
    buffers_.T.copyFromHost(T_initial);
}

void ThermalGPUSolver::uploadFlux(const std::vector<float>& flux) {
    if (!initialized_) {
        throw std::runtime_error("ThermalGPUSolver not initialized");
    }
    buffers_.flux.copyFromHost(flux);
}

void ThermalGPUSolver::advanceTimestep(float dt) {
    if (!initialized_) {
        throw std::runtime_error("ThermalGPUSolver not initialized");
    }

#if PTM_GPU_ENABLED
    if (buffers_.isOnGPU()) {
        timer_->start(streams_[0]);
        launchMaterialUpdateKernel();
        launchThermalKernel(dt);
        timer_->stop(streams_[0]);
        last_kernel_time_ms_ = timer_->elapsedMs();
        return;
    }
#endif

    // CPU fallback implementation
    timer_->start();
    // TODO: Implement CPU fallback thermal solver
    // For now, this is a placeholder - the actual CPU path uses
    // the existing thermal::Model class
    timer_->stop();
    last_kernel_time_ms_ = timer_->elapsedMs();
}

void ThermalGPUSolver::downloadSurfaceTemps(std::vector<float>& T_surface) {
    if (!initialized_) {
        throw std::runtime_error("ThermalGPUSolver not initialized");
    }

    // Surface temps are layer 0, which is at indices [0, n_faces)
    T_surface.resize(n_faces_);

    // Download just the surface layer
    std::vector<float> T_all;
    buffers_.T.copyToHost(T_all);

    // Extract surface (layer 0)
    std::copy(T_all.begin(), T_all.begin() + n_faces_, T_surface.begin());
}

void ThermalGPUSolver::downloadAllTemps(std::vector<float>& T_all) {
    if (!initialized_) {
        throw std::runtime_error("ThermalGPUSolver not initialized");
    }
    buffers_.T.copyToHost(T_all);
}

void ThermalGPUSolver::synchronize() {
#if PTM_GPU_ENABLED
    if (buffers_.isOnGPU()) {
        for (auto& stream : streams_) {
            if (stream) {
                gpuStreamSynchronize(stream);
            }
        }
    }
#endif
}

// ============================================================================
// GPU Kernel Launch Methods
// ============================================================================

void ThermalGPUSolver::launchMaterialUpdateKernel() {
#if PTM_GPU_ENABLED
    kernels::LaunchParams params = kernels::computeLaunchParams(
        n_faces_ * n_layers_, config_.block_size);
    params.stream = streams_.empty() ? nullptr : streams_[0];

    kernels::launchMaterialUpdateKernel(
        buffers_.T.data(),
        buffers_.k.data(),
        buffers_.cp.data(),
        buffers_.kappa.data(),
        buffers_.rho.data(),
        buffers_.kc.data(),
        buffers_.R350,
        buffers_.cp_coeffs.data(),
        buffers_.n_cp_coeffs,
        buffers_.cp0,
        static_cast<int>(n_faces_),
        static_cast<int>(n_layers_),
        params);
#endif
}

void ThermalGPUSolver::launchThermalKernel(float dt) {
#if PTM_GPU_ENABLED
    kernels::LaunchParams params = kernels::computeLaunchParams(
        n_faces_, config_.block_size);
    params.stream = streams_.empty() ? nullptr : streams_[0];

    // Use combined kernel for efficiency (material update + solve in one launch)
    kernels::launchCombinedThermalKernel(
        buffers_.T.data(),
        buffers_.k.data(),
        buffers_.cp.data(),
        buffers_.kappa.data(),
        buffers_.rho.data(),
        buffers_.kc.data(),
        buffers_.flux.data(),
        buffers_.emissivity.data(),
        buffers_.z.data(),
        buffers_.T_deep.data(),
        buffers_.Qb.data(),
        buffers_.R350,
        buffers_.cp_coeffs.data(),
        buffers_.n_cp_coeffs,
        buffers_.cp0,
        buffers_.sigma,
        dt,
        static_cast<int>(n_faces_),
        static_cast<int>(n_layers_),
        params);
#else
    (void)dt;
#endif
}

// ============================================================================
// High-level batch processing function
// ============================================================================

bool runBatchedThermalGPU(
    size_t n_faces,
    size_t n_layers,
    const std::vector<std::vector<float>>& flux_per_timestep,
    const std::vector<double>& times,
    const std::vector<float>& z_grid,
    const std::vector<float>& rho,
    const std::vector<float>& kc,
    const std::vector<float>& emissivity,
    const std::vector<float>& T_initial,
    float cp0,
    const std::vector<float>& cp_coeffs,
    float R350,
    std::vector<std::vector<float>>& surface_temps_out,
    const ThermalGPUConfig& config)
{
    // Initialize GPU
    GPUConfig gpu_config = initializeGPU();

    if (config.verbose) {
        printDeviceInfo(gpu_config);
    }

    // Create solver
    ThermalGPUSolver solver(n_faces, n_layers, config);

    if (!solver.initialize(gpu_config)) {
        std::cerr << "Failed to initialize GPU thermal solver" << std::endl;
        return false;
    }

    // Compute grid derivatives (dz, g1, g2)
    std::vector<float> dz(n_layers > 0 ? n_layers - 1 : 0);
    std::vector<float> g1(n_layers, 0.0f);
    std::vector<float> g2(n_layers, 0.0f);

    for (size_t i = 0; i < n_layers - 1; ++i) {
        dz[i] = z_grid[i + 1] - z_grid[i];
    }

    // Simple coupling coefficients (to be refined)
    for (size_t i = 1; i < n_layers - 1; ++i) {
        float dz_minus = z_grid[i] - z_grid[i - 1];
        float dz_plus = z_grid[i + 1] - z_grid[i];
        g1[i] = 2.0f / (dz_minus * (dz_minus + dz_plus));
        g2[i] = 2.0f / (dz_plus * (dz_minus + dz_plus));
    }

    // Upload mesh data
    std::vector<float> albedo(n_faces, 0.0f);  // Placeholder
    std::vector<float> T_deep(n_faces, T_initial.empty() ? 200.0f : T_initial[n_faces * (n_layers - 1)]);
    std::vector<float> Qb(n_faces, 0.0f);

    solver.uploadMeshData(z_grid, dz, g1, g2, rho, kc, emissivity, albedo, T_deep, Qb);
    solver.setMaterialParams(cp0, cp_coeffs, R350);
    solver.uploadInitialTemperature(T_initial);

    // Prepare output
    size_t n_timesteps = times.size();
    surface_temps_out.resize(n_timesteps);

    // Time stepping loop
    for (size_t t = 0; t < n_timesteps - 1; ++t) {
        float dt = static_cast<float>(times[t + 1] - times[t]);

        // Upload flux for this timestep
        solver.uploadFlux(flux_per_timestep[t]);

        // Advance
        solver.advanceTimestep(dt);

        // Download surface temps
        solver.downloadSurfaceTemps(surface_temps_out[t]);
    }

    // Final timestep
    if (n_timesteps > 0) {
        solver.downloadSurfaceTemps(surface_temps_out[n_timesteps - 1]);
    }

    solver.synchronize();

    if (config.verbose) {
        std::cout << "GPU thermal simulation complete" << std::endl;
    }

    return true;
}

}  // namespace gpu
