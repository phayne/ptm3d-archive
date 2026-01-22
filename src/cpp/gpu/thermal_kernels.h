#pragma once

/**
 * @file thermal_kernels.h
 * @brief GPU kernel declarations for thermal solver
 *
 * Contains kernel function declarations and launch wrappers.
 * Actual kernel implementations are in thermal_kernels.cpp (HIP)
 * or thermal_kernels.cu (CUDA).
 */

#include "gpu_config.h"

namespace gpu {
namespace kernels {

// Maximum depth layers supported in register-based solver
// Larger grids fall back to shared memory or CPU
constexpr int MAX_REGISTER_LAYERS = 64;

// Kernel launch parameters
struct LaunchParams {
    int block_size = 256;
    int grid_size = 0;       // Computed from n_faces
    size_t shared_mem = 0;   // Shared memory per block
#if PTM_GPU_ENABLED
    gpuStream_t stream = nullptr;
#endif
};

/**
 * @brief Compute launch parameters for given face count
 */
inline LaunchParams computeLaunchParams(size_t n_faces, int block_size = 256) {
    LaunchParams params;
    params.block_size = block_size;
    params.grid_size = static_cast<int>((n_faces + block_size - 1) / block_size);
    return params;
}

#if PTM_GPU_ENABLED

/**
 * @brief Update temperature-dependent material properties
 *
 * Computes k(T), cp(T), and kappa = k/(rho*cp) for all grid points.
 *
 * @param T Temperature array [n_faces * n_layers] SoA
 * @param k Output conductivity [n_faces * n_layers]
 * @param cp Output heat capacity [n_faces * n_layers]
 * @param kappa Output diffusivity [n_faces * n_layers]
 * @param rho Density [n_faces * n_layers]
 * @param kc Base conductivity [n_faces * n_layers]
 * @param R350 Radiative conductivity coefficient
 * @param cp_coeffs cp(T) polynomial coefficients
 * @param n_cp_coeffs Number of coefficients
 * @param cp0 Base heat capacity
 * @param n_faces Number of faces
 * @param n_layers Number of depth layers
 * @param params Launch parameters
 */
void launchMaterialUpdateKernel(
    const float* T,
    float* k,
    float* cp,
    float* kappa,
    const float* rho,
    const float* kc,
    float R350,
    const float* cp_coeffs,
    int n_cp_coeffs,
    float cp0,
    int n_faces,
    int n_layers,
    const LaunchParams& params);

/**
 * @brief Solve surface boundary condition (radiative + absorbed flux)
 *
 * Computes surface temperature using linearized Stefan-Boltzmann.
 * T_new = T_old + (Q_absorbed - eps*sigma*T^4) / (k/dz + 4*eps*sigma*T^3)
 *
 * @param T Temperature array [n_faces * n_layers]
 * @param flux Absorbed surface flux [n_faces]
 * @param k Conductivity [n_faces * n_layers]
 * @param emissivity Surface emissivity [n_faces]
 * @param dz0 First layer thickness
 * @param sigma Stefan-Boltzmann constant
 * @param dt Timestep
 * @param n_faces Number of faces
 * @param params Launch parameters
 */
void launchSurfaceBCKernel(
    float* T,
    const float* flux,
    const float* k,
    const float* emissivity,
    float dz0,
    float sigma,
    float dt,
    int n_faces,
    const LaunchParams& params);

/**
 * @brief Main thermal solver kernel - batched Crank-Nicolson with Thomas algorithm
 *
 * Each thread handles one face's entire depth column using register storage.
 * This is the core computational kernel.
 *
 * Algorithm per thread:
 * 1. Load T column into registers
 * 2. Build tridiagonal system (Crank-Nicolson)
 * 3. Solve with Thomas algorithm in registers
 * 4. Write back to global memory
 *
 * @param T Temperature [n_faces * n_layers] SoA layout
 * @param k Conductivity [n_faces * n_layers]
 * @param kappa Diffusivity [n_faces * n_layers]
 * @param rho Density [n_faces * n_layers]
 * @param cp Heat capacity [n_faces * n_layers]
 * @param flux Surface flux [n_faces]
 * @param emissivity Surface emissivity [n_faces]
 * @param z Depth grid [n_layers]
 * @param T_deep Deep boundary temperature [n_faces]
 * @param Qb Bottom heat flux [n_faces]
 * @param sigma Stefan-Boltzmann constant
 * @param dt Timestep
 * @param n_faces Number of faces
 * @param n_layers Number of depth layers
 * @param params Launch parameters
 */
void launchThermalSolverKernel(
    float* T,
    const float* k,
    const float* kappa,
    const float* rho,
    const float* cp,
    const float* flux,
    const float* emissivity,
    const float* z,
    const float* T_deep,
    const float* Qb,
    float sigma,
    float dt,
    int n_faces,
    int n_layers,
    const LaunchParams& params);

/**
 * @brief Combined kernel: material update + thermal solve in one launch
 *
 * More efficient than separate kernels for small grids by reducing
 * launch overhead and keeping data in registers.
 */
void launchCombinedThermalKernel(
    float* T,
    float* k,
    float* cp,
    float* kappa,
    const float* rho,
    const float* kc,
    const float* flux,
    const float* emissivity,
    const float* z,
    const float* T_deep,
    const float* Qb,
    float R350,
    const float* cp_coeffs,
    int n_cp_coeffs,
    float cp0,
    float sigma,
    float dt,
    int n_faces,
    int n_layers,
    const LaunchParams& params);

#endif  // PTM_GPU_ENABLED

}  // namespace kernels
}  // namespace gpu
