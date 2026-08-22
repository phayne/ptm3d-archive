#pragma once

/**
 * @file thermal_gpu.h
 * @brief GPU-accelerated thermal solver interface
 *
 * Provides batched thermal computation across all mesh faces.
 * Uses Structure of Arrays layout for coalesced memory access.
 */

#include "gpu_config.h"
#include "gpu_memory.h"
#include "gpu_utils.h"
#include <cstddef>
#include <vector>
#include <memory>
#include "../constants/stefan_boltzmann.h"  // ptm::kStefanBoltzmann (the single definition)

namespace gpu {

/**
 * @brief Configuration for GPU thermal solver
 */
struct ThermalGPUConfig {
    int block_size = 256;            // Threads per block
    bool use_shared_memory = false;  // Use shared memory for grid arrays
    bool async_transfers = true;     // Use async memory transfers
    int streams = 2;                 // Number of CUDA/HIP streams for overlap
    float convergence_tol = 0.1f;    // Temperature convergence tolerance [K]
    bool verbose = false;
};

/**
 * @brief GPU-accelerated batched thermal solver
 *
 * Processes all mesh faces in parallel on GPU. Each GPU thread handles
 * one face's entire depth column using register-based Thomas algorithm.
 *
 * Usage:
 *   1. Create ThermalGPUSolver with mesh parameters
 *   2. Call initialize() to allocate GPU buffers
 *   3. Call uploadMeshData() to transfer grid, properties
 *   4. For each timestep:
 *      a. uploadFlux() with current surface flux
 *      b. advanceTimestep() to run GPU kernel
 *   5. Call downloadResults() to get final temperatures
 */
class ThermalGPUSolver {
public:
    /**
     * @brief Constructor
     *
     * @param n_faces Number of mesh faces
     * @param n_layers Number of depth layers per face (typically 20-30)
     * @param config Solver configuration
     */
    ThermalGPUSolver(size_t n_faces, size_t n_layers,
                     const ThermalGPUConfig& config = ThermalGPUConfig());

    ~ThermalGPUSolver();

    // No copy
    ThermalGPUSolver(const ThermalGPUSolver&) = delete;
    ThermalGPUSolver& operator=(const ThermalGPUSolver&) = delete;

    /**
     * @brief Initialize GPU resources
     *
     * Allocates device memory, creates streams. Call once before use.
     *
     * @param gpu_config GPU device configuration from initializeGPU()
     * @return true if initialization succeeded
     */
    bool initialize(const GPUConfig& gpu_config);

    /**
     * @brief Upload mesh geometry and material properties
     *
     * Call once after initialize(), before time stepping.
     *
     * @param z_grid Depth grid [n_layers] (shared across faces)
     * @param dz Layer spacing [n_layers-1]
     * @param g1 Coupling coefficient 1 [n_layers]
     * @param g2 Coupling coefficient 2 [n_layers]
     * @param rho Density [n_faces * n_layers] in SoA layout
     * @param kc Base conductivity [n_faces * n_layers] in SoA layout
     * @param emissivity Surface emissivity [n_faces]
     * @param albedo Surface albedo [n_faces]
     * @param T_deep Deep boundary temperature [n_faces]
     * @param Qb Bottom heat flux [n_faces]
     */
    void uploadMeshData(
        const std::vector<float>& z_grid,
        const std::vector<float>& dz,
        const std::vector<float>& g1,
        const std::vector<float>& g2,
        const std::vector<float>& rho,
        const std::vector<float>& kc,
        const std::vector<float>& emissivity,
        const std::vector<float>& albedo,
        const std::vector<float>& T_deep,
        const std::vector<float>& Qb);

    /**
     * @brief Set material property parameters
     *
     * @param cp0 Base heat capacity [J/kg/K]
     * @param cp_coeffs Polynomial coefficients for cp(T)
     * @param R350 Radiative conductivity coefficient for k(T)
     * @param sigma Stefan-Boltzmann constant
     */
    void setMaterialParams(
        float cp0,
        const std::vector<float>& cp_coeffs,
        float R350,
        float sigma = static_cast<float>(ptm::kStefanBoltzmann));

    /**
     * @brief Upload initial temperature profile
     *
     * @param T_initial Initial temperatures [n_faces * n_layers] in SoA layout
     */
    void uploadInitialTemperature(const std::vector<float>& T_initial);

    /**
     * @brief Upload surface flux for current timestep
     *
     * @param flux Absorbed surface flux [n_faces] in W/m^2
     */
    void uploadFlux(const std::vector<float>& flux);

    /**
     * @brief Advance thermal solution by one timestep
     *
     * Runs GPU kernel to:
     * 1. Update material properties k(T), cp(T), kappa(T)
     * 2. Apply surface boundary condition (linearized radiation)
     * 3. Solve Crank-Nicolson system using Thomas algorithm
     * 4. Apply bottom boundary condition
     *
     * @param dt Timestep [seconds]
     */
    void advanceTimestep(float dt);

    /**
     * @brief Download current surface temperatures
     *
     * @param T_surface Output vector [n_faces]
     */
    void downloadSurfaceTemps(std::vector<float>& T_surface);

    /**
     * @brief Download full temperature profiles
     *
     * @param T_all Output temperatures [n_faces * n_layers] in SoA layout
     */
    void downloadAllTemps(std::vector<float>& T_all);

    /**
     * @brief Synchronize all GPU operations
     *
     * Blocks until all queued kernels and transfers complete.
     */
    void synchronize();

    /**
     * @brief Get timing statistics from last timestep
     *
     * @return Kernel execution time in milliseconds
     */
    float getLastKernelTimeMs() const { return last_kernel_time_ms_; }

    /**
     * @brief Check if solver is initialized and ready
     */
    bool isReady() const { return initialized_; }

    /**
     * @brief Get number of faces
     */
    size_t numFaces() const { return n_faces_; }

    /**
     * @brief Get number of layers
     */
    size_t numLayers() const { return n_layers_; }

private:
    size_t n_faces_;
    size_t n_layers_;
    ThermalGPUConfig config_;

    bool initialized_ = false;
    float last_kernel_time_ms_ = 0.0f;

    // GPU buffers (SoA layout)
    ThermalSoABuffers buffers_;

    // Streams for async operations
#if PTM_GPU_ENABLED
    std::vector<gpuStream_t> streams_;
#endif

    // Timing
    std::unique_ptr<GPUTimer> timer_;

    // Launch thermal kernel
    void launchThermalKernel(float dt);

    // Launch material update kernel
    void launchMaterialUpdateKernel();
};

/**
 * @brief Run batched thermal simulation on GPU
 *
 * High-level function that handles the entire thermal simulation
 * for all mesh faces over multiple timesteps.
 *
 * @param n_faces Number of mesh faces
 * @param n_layers Depth layers per face
 * @param flux_per_timestep Surface flux [n_timesteps][n_faces]
 * @param times Time points [n_timesteps]
 * @param z_grid Depth grid [n_layers]
 * @param rho Density [n_faces * n_layers] SoA
 * @param kc Base conductivity [n_faces * n_layers] SoA
 * @param emissivity Surface emissivity [n_faces]
 * @param T_initial Initial temperature [n_faces * n_layers] SoA
 * @param cp0 Base heat capacity
 * @param cp_coeffs cp(T) polynomial coefficients
 * @param R350 Radiative conductivity coefficient
 * @param surface_temps_out Output surface temps [n_timesteps][n_faces]
 * @param config Solver configuration
 * @return true on success
 */
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
    const ThermalGPUConfig& config = ThermalGPUConfig());

}  // namespace gpu
