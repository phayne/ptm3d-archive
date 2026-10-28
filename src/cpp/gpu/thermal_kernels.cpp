/**
 * @file thermal_kernels.cpp
 * @brief GPU kernel implementations for thermal solver
 *
 * This file is compiled with HIP (hipcc) when GPU support is enabled.
 * Contains the actual __global__ kernel implementations.
 *
 * Key design principles:
 * 1. One thread per face - each thread owns a complete depth column
 * 2. Register-based storage - entire T profile kept in registers
 * 3. Coalesced memory access - SoA layout ensures adjacent threads access adjacent memory
 * 4. Linearized surface BC - avoids Newton iterations on GPU
 */

#include "thermal_kernels.h"
#include "gpu_config.h"

#if PTM_GPU_ENABLED

#include <cmath>

namespace gpu {
namespace kernels {

// ============================================================================
// Device helper functions
// ============================================================================

/**
 * @brief Horner's method for polynomial evaluation on GPU
 * cp(T) = cp0 + T * (c0 + T * (c1 + T * c2 ...))
 */
__device__ __forceinline__
float evalPolynomial(float T, const float* coeffs, int n_coeffs, float base) {
    if (n_coeffs == 0) return base;

    float result = coeffs[n_coeffs - 1];
    for (int j = n_coeffs - 2; j >= 0; --j) {
        result = result * T + coeffs[j];
    }
    return base + T * result;
}

/**
 * @brief Compute k(T) = kc * (1 + R350 * T^3)   [Hayne et al. (2017)]
 *
 * this returned `kc + R350*T^3` -- ADDITIVE, not multiplicative -- and
 * its own doc comment stated that wrong formula, so the two agreed with each
 * other and with nothing else. Every other path in PTM3D uses the
 * multiplicative Hayne form: `thermal_model.cpp` (CPU profile/grid/batch),
 * `thermal_kernels.cl`, and the LIVE OpenCL kernel string in
 * `thermal_opencl.cpp::getEmbeddedKernelSource()`. With R350 = chi/350^3 =
 * 6.297e-8 and kc ~ 1.5e-3 W/m/K, the correct k(350 K) is 5.6e-3; the additive
 * form gave 1.5e-3 + 2.7 = 2.70 W/m/K, ~480x too large, and ~700x at lunar noon.
 *
 * NOT RUNTIME-TESTED. This translation unit is compiled only by the HIP and
 * CUDA backends (`src/cpp/gpu/CMakeLists.txt:69,:106` configure_file(... .cu
 * COPYONLY)); the macOS/OpenCL build used for every PTM3D result to date never
 * compiles it, so no baseline exists to regress and the fix cannot be executed
 * on this machine. It is a textual correction to match the three live
 * implementations. See docs/KNOWN_LIMITATIONS.md.
 */
__device__ __forceinline__
float computeK(float T, float kc, float R350) {
    float T2 = T * T;
    float T3 = T2 * T;
    return kc * (1.0f + R350 * T3);
}

// ============================================================================
// Material Update Kernel
// ============================================================================

__global__
void materialUpdateKernel(
    const float* __restrict__ T,
    float* __restrict__ k,
    float* __restrict__ cp,
    float* __restrict__ kappa,
    const float* __restrict__ rho,
    const float* __restrict__ kc,
    float R350,
    const float* __restrict__ cp_coeffs,
    int n_cp_coeffs,
    float cp0,
    int n_faces,
    int n_layers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_faces * n_layers;

    if (idx >= total) return;

    float Ti = T[idx];

    // k(T) = kc * (1 + R350 * T^3)   [was stated additively in the HIP/CUDA path]
    float ki = computeK(Ti, kc[idx], R350);
    k[idx] = ki;

    // cp(T) via Horner's method
    float cpi = evalPolynomial(Ti, cp_coeffs, n_cp_coeffs, cp0);
    cp[idx] = cpi;

    // kappa = k / (rho * cp)
    float rhoi = rho[idx];
    kappa[idx] = ki / (rhoi * cpi);
}

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
    const LaunchParams& params)
{
    int total = n_faces * n_layers;
    int grid = (total + params.block_size - 1) / params.block_size;

    #if PTM_GPU_HIP
    hipLaunchKernelGGL(materialUpdateKernel,
        dim3(grid), dim3(params.block_size), 0, params.stream,
        T, k, cp, kappa, rho, kc, R350, cp_coeffs, n_cp_coeffs, cp0,
        n_faces, n_layers);
    #elif PTM_GPU_CUDA
    materialUpdateKernel<<<grid, params.block_size, 0, params.stream>>>(
        T, k, cp, kappa, rho, kc, R350, cp_coeffs, n_cp_coeffs, cp0,
        n_faces, n_layers);
    #endif
}

// ============================================================================
// Surface Boundary Condition Kernel
// ============================================================================

__global__
void surfaceBCKernel(
    float* __restrict__ T,
    const float* __restrict__ flux,
    const float* __restrict__ k,
    const float* __restrict__ emissivity,
    float dz0,
    float sigma,
    float dt,
    int n_faces)
{
    int face = blockIdx.x * blockDim.x + threadIdx.x;
    if (face >= n_faces) return;

    // Surface is at layer 0: index = 0 * n_faces + face = face
    float T_surf = T[face];
    float k_surf = k[face];
    float eps = emissivity[face];
    float Q_in = flux[face];

    // Linearized radiative BC
    float T2 = T_surf * T_surf;
    float T3 = T2 * T_surf;
    float T4 = T2 * T2;

    // Radiative emission
    float Q_rad = eps * sigma * T4;

    // Net flux
    float Q_net = Q_in - Q_rad;

    // Linearized update: dT/dt = Q_net / (rho * cp * dz)
    // Using linearized radiation: d(sigma*T^4)/dT = 4*sigma*T^3
    float rad_coeff = 4.0f * eps * sigma * T3;
    float cond_coeff = k_surf / dz0;

    // Implicit linearization for stability
    float denom = cond_coeff + rad_coeff;
    if (denom > 1e-10f) {
        float dT = Q_net * dt / denom;
        // Limit temperature change for stability
        dT = fminf(fmaxf(dT, -10.0f), 10.0f);
        T[face] = T_surf + dT;
    }
}

void launchSurfaceBCKernel(
    float* T,
    const float* flux,
    const float* k,
    const float* emissivity,
    float dz0,
    float sigma,
    float dt,
    int n_faces,
    const LaunchParams& params)
{
    #if PTM_GPU_HIP
    hipLaunchKernelGGL(surfaceBCKernel,
        dim3(params.grid_size), dim3(params.block_size), 0, params.stream,
        T, flux, k, emissivity, dz0, sigma, dt, n_faces);
    #elif PTM_GPU_CUDA
    surfaceBCKernel<<<params.grid_size, params.block_size, 0, params.stream>>>(
        T, flux, k, emissivity, dz0, sigma, dt, n_faces);
    #endif
}

// ============================================================================
// Main Thermal Solver Kernel - Register-based Thomas Algorithm
// ============================================================================

/**
 * This kernel implements the Crank-Nicolson scheme with Thomas algorithm
 * entirely in GPU registers. Each thread handles one face's depth column.
 *
 * The key insight is that for small grids (N <= 64), we can keep the entire
 * tridiagonal system in registers, avoiding all intermediate memory access.
 *
 * Crank-Nicolson discretization:
 *   (1 + r/2) * T_new[i] - r/4 * (T_new[i-1] + T_new[i+1])
 * = (1 - r/2) * T_old[i] + r/4 * (T_old[i-1] + T_old[i+1])
 *
 * where r = kappa * dt / dz^2
 */
__global__
void thermalSolverKernel(
    float* __restrict__ T_global,
    const float* __restrict__ k,
    const float* __restrict__ kappa,
    const float* __restrict__ rho,
    const float* __restrict__ cp,
    const float* __restrict__ flux,
    const float* __restrict__ emissivity,
    const float* __restrict__ z,
    const float* __restrict__ T_deep,
    const float* __restrict__ Qb,
    float sigma,
    float dt,
    int n_faces,
    int n_layers)
{
    int face = blockIdx.x * blockDim.x + threadIdx.x;
    if (face >= n_faces) return;

    // Limit to max register size
    if (n_layers > MAX_REGISTER_LAYERS) return;

    // =========================================================================
    // 1. Load temperature column into registers (coalesced read)
    // =========================================================================
    float T_local[MAX_REGISTER_LAYERS];
    float kappa_local[MAX_REGISTER_LAYERS];

    #pragma unroll 8
    for (int i = 0; i < n_layers && i < MAX_REGISTER_LAYERS; ++i) {
        int idx = i * n_faces + face;  // SoA layout
        T_local[i] = T_global[idx];
        kappa_local[i] = kappa[idx];
    }

    // Load boundary conditions
    float T_bot = T_deep[face];
    float Q_bot = Qb[face];
    float eps = emissivity[face];
    float Q_surf = flux[face];

    // =========================================================================
    // 2. Build tridiagonal system in registers
    // =========================================================================
    // Crank-Nicolson: A * T_new = B * T_old + source
    // Tridiagonal system: a[i]*T[i-1] + b[i]*T[i] + c[i]*T[i+1] = d[i]

    float a[MAX_REGISTER_LAYERS];  // Lower diagonal
    float b[MAX_REGISTER_LAYERS];  // Main diagonal
    float c[MAX_REGISTER_LAYERS];  // Upper diagonal
    float d[MAX_REGISTER_LAYERS];  // RHS

    // Interior nodes (i = 1 to n_layers-2)
    #pragma unroll 4
    for (int i = 1; i < n_layers - 1 && i < MAX_REGISTER_LAYERS - 1; ++i) {
        float dz_minus = z[i] - z[i-1];
        float dz_plus = z[i+1] - z[i];
        float dz_avg = 0.5f * (dz_minus + dz_plus);

        // Fourier number
        float r_minus = kappa_local[i] * dt / (dz_minus * dz_avg);
        float r_plus = kappa_local[i] * dt / (dz_plus * dz_avg);

        // Crank-Nicolson coefficients (implicit part)
        a[i] = -0.5f * r_minus;
        c[i] = -0.5f * r_plus;
        b[i] = 1.0f + 0.5f * (r_minus + r_plus);

        // RHS: explicit part
        d[i] = T_local[i] * (1.0f - 0.5f * (r_minus + r_plus))
             + 0.5f * r_minus * T_local[i-1]
             + 0.5f * r_plus * T_local[i+1];
    }

    // =========================================================================
    // 3. Surface boundary condition (i = 0)
    // =========================================================================
    {
        float dz0 = z[1] - z[0];
        float k0 = k[face];  // k at surface

        // Linearized radiative BC
        float T0 = T_local[0];
        float T0_3 = T0 * T0 * T0;
        float T0_4 = T0_3 * T0;

        // Coefficients for surface node
        float r0 = kappa_local[0] * dt / (dz0 * dz0);

        // Radiative coupling coefficient
        float h_rad = 4.0f * eps * sigma * T0_3;  // Linearized
        float Bi = h_rad * dz0 / k0;  // Biot number

        a[0] = 0.0f;  // No node below surface
        c[0] = -0.5f * r0;
        b[0] = 1.0f + 0.5f * r0 + 0.5f * Bi * r0;

        // RHS with flux source
        float Q_rad = eps * sigma * T0_4;
        float Q_net = Q_surf - Q_rad + h_rad * T0;  // Linearization correction
        d[0] = T0 * (1.0f - 0.5f * r0 - 0.5f * Bi * r0)
             + 0.5f * r0 * T_local[1]
             + r0 * dz0 / k0 * Q_net;
    }

    // =========================================================================
    // 4. Bottom boundary condition (i = n_layers - 1)
    // =========================================================================
    {
        int i = n_layers - 1;
        float dz_last = z[i] - z[i-1];

        if (fabsf(Q_bot) > 1e-10f) {
            // Flux boundary condition
            float r = kappa_local[i] * dt / (dz_last * dz_last);
            a[i] = -0.5f * r;
            c[i] = 0.0f;
            b[i] = 1.0f + 0.5f * r;
            d[i] = T_local[i] * (1.0f - 0.5f * r)
                 + 0.5f * r * T_local[i-1]
                 + dt * Q_bot / (rho[i * n_faces + face] * cp[i * n_faces + face] * dz_last);
        } else {
            // Fixed temperature boundary
            a[i] = 0.0f;
            c[i] = 0.0f;
            b[i] = 1.0f;
            d[i] = T_bot;
        }
    }

    // =========================================================================
    // 5. Thomas algorithm (forward elimination) - entirely in registers
    // =========================================================================
    #pragma unroll 4
    for (int i = 1; i < n_layers && i < MAX_REGISTER_LAYERS; ++i) {
        float w = a[i] / b[i-1];
        b[i] = b[i] - w * c[i-1];
        d[i] = d[i] - w * d[i-1];
    }

    // =========================================================================
    // 6. Back substitution - entirely in registers
    // =========================================================================
    T_local[n_layers - 1] = d[n_layers - 1] / b[n_layers - 1];

    #pragma unroll 4
    for (int i = n_layers - 2; i >= 0; --i) {
        T_local[i] = (d[i] - c[i] * T_local[i+1]) / b[i];
    }

    // =========================================================================
    // 7. Write back to global memory (coalesced write)
    // =========================================================================
    #pragma unroll 8
    for (int i = 0; i < n_layers && i < MAX_REGISTER_LAYERS; ++i) {
        int idx = i * n_faces + face;
        T_global[idx] = T_local[i];
    }
}

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
    const LaunchParams& params)
{
    #if PTM_GPU_HIP
    hipLaunchKernelGGL(thermalSolverKernel,
        dim3(params.grid_size), dim3(params.block_size), 0, params.stream,
        T, k, kappa, rho, cp, flux, emissivity, z, T_deep, Qb,
        sigma, dt, n_faces, n_layers);
    #elif PTM_GPU_CUDA
    thermalSolverKernel<<<params.grid_size, params.block_size, 0, params.stream>>>(
        T, k, kappa, rho, cp, flux, emissivity, z, T_deep, Qb,
        sigma, dt, n_faces, n_layers);
    #endif
}

// ============================================================================
// Combined Kernel - Material Update + Thermal Solve
// ============================================================================

/**
 * Combined kernel that does material property update and thermal solve
 * in a single kernel launch. More efficient for reducing launch overhead.
 */
__global__
void combinedThermalKernel(
    float* __restrict__ T_global,
    float* __restrict__ k_global,
    float* __restrict__ cp_global,
    float* __restrict__ kappa_global,
    const float* __restrict__ rho,
    const float* __restrict__ kc,
    const float* __restrict__ flux,
    const float* __restrict__ emissivity,
    const float* __restrict__ z,
    const float* __restrict__ T_deep,
    const float* __restrict__ Qb,
    float R350,
    const float* __restrict__ cp_coeffs,
    int n_cp_coeffs,
    float cp0,
    float sigma,
    float dt,
    int n_faces,
    int n_layers)
{
    int face = blockIdx.x * blockDim.x + threadIdx.x;
    if (face >= n_faces) return;
    if (n_layers > MAX_REGISTER_LAYERS) return;

    // Load and update properties in registers
    float T_local[MAX_REGISTER_LAYERS];
    float k_local[MAX_REGISTER_LAYERS];
    float cp_local[MAX_REGISTER_LAYERS];
    float kappa_local[MAX_REGISTER_LAYERS];
    float rho_local[MAX_REGISTER_LAYERS];

    // Load temperature and compute properties
    #pragma unroll 8
    for (int i = 0; i < n_layers && i < MAX_REGISTER_LAYERS; ++i) {
        int idx = i * n_faces + face;
        float Ti = T_global[idx];
        T_local[i] = Ti;

        float kc_i = kc[idx];
        float rho_i = rho[idx];
        rho_local[i] = rho_i;

        // k(T) = kc * (1 + R350 * T^3)   [was stated additively in the HIP/CUDA path]
        float ki = computeK(Ti, kc_i, R350);
        k_local[i] = ki;

        // cp(T)
        float cpi = evalPolynomial(Ti, cp_coeffs, n_cp_coeffs, cp0);
        cp_local[i] = cpi;

        // kappa = k / (rho * cp)
        kappa_local[i] = ki / (rho_i * cpi);
    }

    // Load boundary conditions
    float T_bot = T_deep[face];
    float Q_bot = Qb[face];
    float eps = emissivity[face];
    float Q_surf = flux[face];

    // Build and solve tridiagonal system (same as thermalSolverKernel)
    float a[MAX_REGISTER_LAYERS];
    float b[MAX_REGISTER_LAYERS];
    float c[MAX_REGISTER_LAYERS];
    float d[MAX_REGISTER_LAYERS];

    // Interior nodes
    #pragma unroll 4
    for (int i = 1; i < n_layers - 1 && i < MAX_REGISTER_LAYERS - 1; ++i) {
        float dz_minus = z[i] - z[i-1];
        float dz_plus = z[i+1] - z[i];
        float dz_avg = 0.5f * (dz_minus + dz_plus);

        float r_minus = kappa_local[i] * dt / (dz_minus * dz_avg);
        float r_plus = kappa_local[i] * dt / (dz_plus * dz_avg);

        a[i] = -0.5f * r_minus;
        c[i] = -0.5f * r_plus;
        b[i] = 1.0f + 0.5f * (r_minus + r_plus);

        d[i] = T_local[i] * (1.0f - 0.5f * (r_minus + r_plus))
             + 0.5f * r_minus * T_local[i-1]
             + 0.5f * r_plus * T_local[i+1];
    }

    // Surface BC
    {
        float dz0 = z[1] - z[0];
        float T0 = T_local[0];
        float T0_3 = T0 * T0 * T0;
        float T0_4 = T0_3 * T0;

        float r0 = kappa_local[0] * dt / (dz0 * dz0);
        float h_rad = 4.0f * eps * sigma * T0_3;
        float Bi = h_rad * dz0 / k_local[0];

        a[0] = 0.0f;
        c[0] = -0.5f * r0;
        b[0] = 1.0f + 0.5f * r0 + 0.5f * Bi * r0;

        float Q_rad = eps * sigma * T0_4;
        float Q_net = Q_surf - Q_rad + h_rad * T0;
        d[0] = T0 * (1.0f - 0.5f * r0 - 0.5f * Bi * r0)
             + 0.5f * r0 * T_local[1]
             + r0 * dz0 / k_local[0] * Q_net;
    }

    // Bottom BC
    {
        int i = n_layers - 1;
        float dz_last = z[i] - z[i-1];

        if (fabsf(Q_bot) > 1e-10f) {
            float r = kappa_local[i] * dt / (dz_last * dz_last);
            a[i] = -0.5f * r;
            c[i] = 0.0f;
            b[i] = 1.0f + 0.5f * r;
            d[i] = T_local[i] * (1.0f - 0.5f * r)
                 + 0.5f * r * T_local[i-1]
                 + dt * Q_bot / (rho_local[i] * cp_local[i] * dz_last);
        } else {
            a[i] = 0.0f;
            c[i] = 0.0f;
            b[i] = 1.0f;
            d[i] = T_bot;
        }
    }

    // Thomas algorithm - forward elimination
    #pragma unroll 4
    for (int i = 1; i < n_layers && i < MAX_REGISTER_LAYERS; ++i) {
        float w = a[i] / b[i-1];
        b[i] = b[i] - w * c[i-1];
        d[i] = d[i] - w * d[i-1];
    }

    // Back substitution
    T_local[n_layers - 1] = d[n_layers - 1] / b[n_layers - 1];

    #pragma unroll 4
    for (int i = n_layers - 2; i >= 0; --i) {
        T_local[i] = (d[i] - c[i] * T_local[i+1]) / b[i];
    }

    // Write back to global memory
    #pragma unroll 8
    for (int i = 0; i < n_layers && i < MAX_REGISTER_LAYERS; ++i) {
        int idx = i * n_faces + face;
        T_global[idx] = T_local[i];
        k_global[idx] = k_local[i];
        cp_global[idx] = cp_local[i];
        kappa_global[idx] = kappa_local[i];
    }
}

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
    const LaunchParams& params)
{
    #if PTM_GPU_HIP
    hipLaunchKernelGGL(combinedThermalKernel,
        dim3(params.grid_size), dim3(params.block_size), 0, params.stream,
        T, k, cp, kappa, rho, kc, flux, emissivity, z, T_deep, Qb,
        R350, cp_coeffs, n_cp_coeffs, cp0, sigma, dt, n_faces, n_layers);
    #elif PTM_GPU_CUDA
    combinedThermalKernel<<<params.grid_size, params.block_size, 0, params.stream>>>(
        T, k, cp, kappa, rho, kc, flux, emissivity, z, T_deep, Qb,
        R350, cp_coeffs, n_cp_coeffs, cp0, sigma, dt, n_faces, n_layers);
    #endif
}

}  // namespace kernels
}  // namespace gpu

#endif  // PTM_GPU_ENABLED
