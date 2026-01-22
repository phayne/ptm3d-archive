/**
 * @file thermal_kernels.cl
 * @brief OpenCL kernels for PTM thermal solver
 *
 * NOTE: This file is provided for reference. The actual kernel source used
 * by the solver is embedded in thermal_opencl.cpp via getEmbeddedKernelSource().
 * The embedded version uses Newton-Raphson surface boundary conditions which
 * provides better numerical stability than the linearized approach here.
 *
 * Implements GPU-accelerated thermal diffusion solver using Crank-Nicolson
 * with Thomas algorithm. Each work item handles one mesh face's depth column.
 *
 * Memory layout: Structure of Arrays (SoA)
 * - T[layer * n_faces + face] for coalesced access
 */

// Maximum depth layers supported in private memory
// Larger grids need alternative implementation
#define MAX_LAYERS 64

// Stefan-Boltzmann constant (can be overridden at compile time)
#ifndef SIGMA
#define SIGMA 5.67051196e-8f
#endif

/**
 * @brief Compute temperature-dependent thermal conductivity k(T)
 *
 * k(T) = kc * (1 + R350 * T^3)
 * Accounts for radiative conductivity at high temperatures.
 */
inline float compute_k(float T, float kc, float R350) {
    float T3 = T * T * T;
    return kc * (1.0f + R350 * T3);
}

/**
 * @brief Compute temperature-dependent heat capacity cp(T)
 *
 * Uses Horner's method for polynomial evaluation.
 * cp(T) = cp0 * (c0 + c1*T + c2*T^2 + ...)
 */
inline float compute_cp(float T,
                        float cp0,
                        __constant float* cp_coeffs,
                        int n_coeffs) {
    // Polynomial: cp = cp0 + T * (c0 + T * (c1 + T * (c2 + ...)))
    // Horner form: result = c[n-1]; then result = result*T + c[i] for i=n-2..0
    // Final: cp = cp0 + T * result
    if (n_coeffs == 0) {
        return cp0;
    }

    // Horner's method: evaluate polynomial from highest to lowest coefficient
    float result = cp_coeffs[n_coeffs - 1];
    for (int i = n_coeffs - 2; i >= 0; --i) {
        result = result * T + cp_coeffs[i];
    }
    // IMPORTANT: cp = cp0 + T * result, NOT cp0 * result!
    return cp0 + T * result;
}

/**
 * @brief Material properties update kernel
 *
 * Computes k(T), cp(T), and kappa = k/(rho*cp) for all grid points.
 * Each work item handles one (face, layer) pair.
 */
__kernel void materialUpdateKernel(
    __global const float* T,      // Temperature [n_faces * n_layers]
    __global float* k,            // Output: conductivity
    __global float* cp,           // Output: heat capacity
    __global float* kappa,        // Output: diffusivity
    __global const float* rho,    // Density
    __global const float* kc,     // Base conductivity
    float R350,                   // Radiative conductivity coefficient
    __constant float* cp_coeffs,  // cp(T) polynomial coefficients
    int n_cp_coeffs,
    float cp0,
    int n_faces,
    int n_layers)
{
    int idx = get_global_id(0);
    int total = n_faces * n_layers;

    if (idx >= total) return;

    float T_val = T[idx];
    float kc_val = kc[idx];
    float rho_val = rho[idx];

    float k_val = compute_k(T_val, kc_val, R350);
    float cp_val = compute_cp(T_val, cp0, cp_coeffs, n_cp_coeffs);

    k[idx] = k_val;
    cp[idx] = cp_val;
    kappa[idx] = k_val / (rho_val * cp_val);
}

/**
 * @brief Surface boundary condition kernel
 *
 * Applies linearized radiative boundary condition at the surface.
 * Uses implicit scheme for stability.
 */
__kernel void surfaceBCKernel(
    __global float* T,                // Temperature [n_faces * n_layers]
    __global const float* flux,       // Absorbed flux [n_faces]
    __global const float* k,          // Conductivity
    __global const float* emissivity, // Surface emissivity
    __global const float* z,          // Depth grid
    float sigma,
    float dt,
    int n_faces)
{
    int face = get_global_id(0);
    if (face >= n_faces) return;

    // Surface layer index (layer 0)
    int surf_idx = face;  // layer 0 * n_faces + face

    float T_surf = T[surf_idx];
    float Q_abs = flux[face];
    float eps = emissivity[face];
    float k_surf = k[surf_idx];

    // Get subsurface temperature for conductive flux
    int sub_idx = n_faces + face;  // layer 1 * n_faces + face
    float T_sub = T[sub_idx];
    float dz0 = z[1] - z[0];

    // Linearized radiation: 4*eps*sigma*T^3
    float T3 = T_surf * T_surf * T_surf;
    float rad_coeff = 4.0f * eps * sigma * T3;

    // Energy balance:
    // rho*cp*dz*(T_new - T_old)/dt = Q_abs - eps*sigma*T^4 + k*(T_sub - T_surf)/dz
    // Linearize around current T:
    // T_new = T_old + dt/(rho*cp*dz) * (Q_abs - eps*sigma*T^4 + k*(T_sub - T)/dz)

    // Simplified implicit surface update
    float Q_rad = eps * sigma * T_surf * T3;  // eps*sigma*T^4
    float Q_cond = k_surf * (T_sub - T_surf) / dz0;

    // Update with implicit radiation term
    float denom = 1.0f + dt * (rad_coeff + k_surf / dz0) / dz0;
    float numer = T_surf + dt * (Q_abs - Q_rad + Q_cond) / dz0;

    T[surf_idx] = numer / denom;
}

/**
 * @brief Main thermal solver kernel - Crank-Nicolson with Thomas algorithm
 *
 * Each work item handles one face's entire depth column.
 * Uses private memory for the tridiagonal system.
 *
 * Algorithm:
 * 1. Load temperature column into private memory
 * 2. Build Crank-Nicolson tridiagonal system
 * 3. Solve with Thomas algorithm
 * 4. Write results back to global memory
 */
__kernel void thermalSolverKernel(
    __global float* T_global,         // Temperature [n_faces * n_layers]
    __global const float* k,          // Conductivity
    __global const float* kappa,      // Diffusivity
    __global const float* rho,        // Density
    __global const float* cp,         // Heat capacity
    __global const float* flux,       // Surface flux [n_faces]
    __global const float* emissivity, // Surface emissivity [n_faces]
    __global const float* z,          // Depth grid [n_layers]
    __global const float* T_deep,     // Deep boundary temp [n_faces]
    __global const float* Qb,         // Bottom heat flux [n_faces]
    float sigma,
    float dt,
    int n_faces,
    int n_layers)
{
    int face = get_global_id(0);
    if (face >= n_faces) return;

    // Limit to supported layer count
    if (n_layers > MAX_LAYERS) {
        // Fall back to simpler explicit method for very deep grids
        return;
    }

    // Private arrays for tridiagonal system
    float T_local[MAX_LAYERS];
    float a[MAX_LAYERS];  // Sub-diagonal
    float b[MAX_LAYERS];  // Main diagonal
    float c[MAX_LAYERS];  // Super-diagonal
    float d[MAX_LAYERS];  // RHS

    // Load temperature column (coalesced reads)
    for (int i = 0; i < n_layers; ++i) {
        T_local[i] = T_global[i * n_faces + face];
    }

    // Load material properties for this face
    float eps = emissivity[face];
    float Q_surf = flux[face];
    float T_bottom = T_deep[face];
    float Q_bottom = Qb[face];

    // Build tridiagonal system using Crank-Nicolson discretization
    // ∂T/∂t = κ * ∂²T/∂z²

    // Surface layer (i=0): radiative + conductive BC
    {
        float dz = z[1] - z[0];
        float dz2 = dz * dz;
        float kappa_0 = kappa[face];  // layer 0
        float k_0 = k[face];

        // Fourier number
        float Fo = kappa_0 * dt / dz2;

        // Radiative linearization coefficient
        float T3 = T_local[0] * T_local[0] * T_local[0];
        float rad = 4.0f * eps * sigma * T3 * dz / k_0;

        // Crank-Nicolson: average of explicit and implicit
        a[0] = 0.0f;
        b[0] = 1.0f + Fo + 0.5f * rad;
        c[0] = -Fo;

        // RHS: explicit part + source
        float Q_rad_old = eps * sigma * T_local[0] * T3;
        d[0] = (1.0f - Fo) * T_local[0] + Fo * T_local[1]
             + dt * (Q_surf - Q_rad_old) / (rho[face] * cp[face] * dz);
    }

    // Interior layers - flux-based formulation matching CPU code
    for (int i = 1; i < n_layers - 1; ++i) {
        int idx = i * n_faces + face;
        int idx_minus = (i - 1) * n_faces + face;

        float dz_minus = z[i] - z[i - 1];
        float dz_plus = z[i + 1] - z[i];
        float dz_avg = 0.5f * (dz_minus + dz_plus);

        // Grid coefficients (matching CPU g1, g2 formulation)
        float g1 = 1.0f / (dz_minus * dz_avg);  // coupling to i-1
        float g2 = 1.0f / (dz_plus * dz_avg);   // coupling to i+1

        // Flux-based: use k[i-1] for lower interface, k[i] for upper
        float alpha = g1 * k[idx_minus];
        float beta = g2 * k[idx];
        float inv_rhocp = dt / (rho[idx] * cp[idx]);

        a[i] = -0.5f * inv_rhocp * alpha;
        b[i] = 1.0f + 0.5f * inv_rhocp * (alpha + beta);
        c[i] = -0.5f * inv_rhocp * beta;

        // RHS: Crank-Nicolson explicit part
        d[i] = T_local[i] + 0.5f * inv_rhocp * (
            alpha * T_local[i - 1]
            - (alpha + beta) * T_local[i]
            + beta * T_local[i + 1]);
    }

    // Bottom layer (i=n_layers-1): fixed temperature or flux BC
    {
        int i = n_layers - 1;
        int idx = i * n_faces + face;

        float dz = z[i] - z[i - 1];
        float dz2 = dz * dz;
        float kappa_n = kappa[idx];
        float k_n = k[idx];

        float Fo = kappa_n * dt / dz2;

        // Fixed temperature BC
        a[i] = -0.5f * Fo;
        b[i] = 1.0f + 0.5f * Fo;
        c[i] = 0.0f;

        // RHS with bottom BC
        d[i] = 0.5f * Fo * T_local[i - 1]
             + (1.0f - 0.5f * Fo) * T_local[i]
             + 0.5f * Fo * T_bottom;

        // Add bottom heat flux if specified
        if (Q_bottom != 0.0f) {
            d[i] += dt * Q_bottom / (rho[idx] * cp[idx] * dz);
        }
    }

    // Thomas algorithm (TDMA) - forward sweep
    for (int i = 1; i < n_layers; ++i) {
        float m = a[i] / b[i - 1];
        b[i] = b[i] - m * c[i - 1];
        d[i] = d[i] - m * d[i - 1];
    }

    // Back substitution
    T_local[n_layers - 1] = d[n_layers - 1] / b[n_layers - 1];
    for (int i = n_layers - 2; i >= 0; --i) {
        T_local[i] = (d[i] - c[i] * T_local[i + 1]) / b[i];
    }

    // Write results back (coalesced writes)
    for (int i = 0; i < n_layers; ++i) {
        T_global[i * n_faces + face] = T_local[i];
    }
}

/**
 * @brief Combined kernel: material update + thermal solve
 *
 * More efficient for reducing kernel launch overhead.
 * Each work item handles one face's complete thermal update.
 */
__kernel void combinedThermalKernel(
    __global float* T_global,
    __global float* k_global,
    __global float* cp_global,
    __global float* kappa_global,
    __global const float* rho,
    __global const float* kc,
    __global const float* flux,
    __global const float* emissivity,
    __global const float* z,
    __global const float* T_deep,
    __global const float* Qb,
    float R350,
    __constant float* cp_coeffs,
    int n_cp_coeffs,
    float cp0,
    float sigma,
    float dt,
    int n_faces,
    int n_layers)
{
    int face = get_global_id(0);
    if (face >= n_faces) return;

    if (n_layers > MAX_LAYERS) return;

    // Private arrays
    float T_local[MAX_LAYERS];
    float k_local[MAX_LAYERS];
    float cp_local[MAX_LAYERS];
    float kappa_local[MAX_LAYERS];
    float a[MAX_LAYERS];
    float b[MAX_LAYERS];
    float c[MAX_LAYERS];
    float d[MAX_LAYERS];

    // Step 1: Load temperatures and compute material properties
    for (int i = 0; i < n_layers; ++i) {
        int idx = i * n_faces + face;
        float T_val = T_global[idx];
        float kc_val = kc[idx];
        float rho_val = rho[idx];

        T_local[i] = T_val;
        k_local[i] = compute_k(T_val, kc_val, R350);
        cp_local[i] = compute_cp(T_val, cp0, cp_coeffs, n_cp_coeffs);
        kappa_local[i] = k_local[i] / (rho_val * cp_local[i]);
    }

    // Load boundary conditions
    float eps = emissivity[face];
    float Q_surf = flux[face];
    float T_bottom = T_deep[face];
    float Q_bottom = Qb[face];

    // Step 2: Build tridiagonal system

    // Surface layer (i=0)
    {
        float dz = z[1] - z[0];
        float dz2 = dz * dz;
        float Fo = kappa_local[0] * dt / dz2;

        float T3 = T_local[0] * T_local[0] * T_local[0];
        float rad = 4.0f * eps * sigma * T3 * dz / k_local[0];

        a[0] = 0.0f;
        b[0] = 1.0f + Fo + 0.5f * rad;
        c[0] = -Fo;

        float Q_rad_old = eps * sigma * T_local[0] * T3;
        float rho_0 = rho[face];
        d[0] = (1.0f - Fo) * T_local[0] + Fo * T_local[1]
             + dt * (Q_surf - Q_rad_old) / (rho_0 * cp_local[0] * dz);
    }

    // Interior layers - flux-based formulation matching CPU code
    for (int i = 1; i < n_layers - 1; ++i) {
        float dz_minus = z[i] - z[i - 1];
        float dz_plus = z[i + 1] - z[i];
        float dz_avg = 0.5f * (dz_minus + dz_plus);

        // Grid coefficients (matching CPU g1, g2 formulation)
        float g1 = 1.0f / (dz_minus * dz_avg);  // coupling to i-1
        float g2 = 1.0f / (dz_plus * dz_avg);   // coupling to i+1

        // Flux-based: use k[i-1] for lower interface, k[i] for upper
        float alpha = g1 * k_local[i - 1];
        float beta = g2 * k_local[i];
        float inv_rhocp = dt / (rho[i * n_faces + face] * cp_local[i]);

        a[i] = -0.5f * inv_rhocp * alpha;
        b[i] = 1.0f + 0.5f * inv_rhocp * (alpha + beta);
        c[i] = -0.5f * inv_rhocp * beta;

        // RHS: Crank-Nicolson explicit part
        d[i] = T_local[i] + 0.5f * inv_rhocp * (
            alpha * T_local[i - 1]
            - (alpha + beta) * T_local[i]
            + beta * T_local[i + 1]);
    }

    // Bottom layer
    {
        int i = n_layers - 1;
        int idx = i * n_faces + face;
        float dz = z[i] - z[i - 1];
        float dz2 = dz * dz;
        float Fo = kappa_local[i] * dt / dz2;

        a[i] = -0.5f * Fo;
        b[i] = 1.0f + 0.5f * Fo;
        c[i] = 0.0f;

        d[i] = 0.5f * Fo * T_local[i - 1]
             + (1.0f - 0.5f * Fo) * T_local[i]
             + 0.5f * Fo * T_bottom;

        if (Q_bottom != 0.0f) {
            float rho_n = rho[idx];
            d[i] += dt * Q_bottom / (rho_n * cp_local[i] * dz);
        }
    }

    // Step 3: Thomas algorithm - forward sweep
    for (int i = 1; i < n_layers; ++i) {
        float m = a[i] / b[i - 1];
        b[i] = b[i] - m * c[i - 1];
        d[i] = d[i] - m * d[i - 1];
    }

    // Back substitution
    T_local[n_layers - 1] = d[n_layers - 1] / b[n_layers - 1];
    for (int i = n_layers - 2; i >= 0; --i) {
        T_local[i] = (d[i] - c[i] * T_local[i + 1]) / b[i];
    }

    // Step 4: Write results back
    for (int i = 0; i < n_layers; ++i) {
        int idx = i * n_faces + face;
        T_global[idx] = T_local[i];
        k_global[idx] = k_local[i];
        cp_global[idx] = cp_local[i];
        kappa_global[idx] = kappa_local[i];
    }
}

/**
 * @brief Persistent thermal kernel - processes ALL timesteps on GPU
 *
 * This kernel keeps all data on GPU and processes all timesteps for maximum efficiency.
 * Data transfer is minimized: upload flux once, download surface temps once.
 *
 * Memory layout:
 * - T: [n_layers * n_faces] (SoA)
 * - flux_all: [n_timesteps * n_faces] (timestep-major for coalesced reads)
 * - output_temps: [n_timesteps * n_faces] (timestep-major)
 */
__kernel void persistentThermalKernel(
    __global float* T_global,          // Temperature [n_layers * n_faces]
    __global float* k_global,          // Conductivity [n_layers * n_faces]
    __global float* cp_global,         // Heat capacity [n_layers * n_faces]
    __global float* kappa_global,      // Diffusivity [n_layers * n_faces]
    __global const float* rho,         // Density [n_layers * n_faces]
    __global const float* kc,          // Base conductivity [n_layers * n_faces]
    __global const float* flux_all,    // Absorbed flux [n_timesteps * n_faces]
    __global float* output_temps,      // Output surface temps [n_timesteps * n_faces]
    __global const float* emissivity,  // Emissivity [n_faces]
    __global const float* z,           // Depth grid [n_layers]
    __global const float* T_deep,      // Deep temperature [n_faces]
    __global const float* Qb,          // Bottom heat flux [n_faces]
    float R350,                        // Radiative conductivity coefficient
    __constant float* cp_coeffs,       // cp(T) polynomial coefficients
    int n_cp_coeffs,
    float cp0,
    float sigma,                       // Stefan-Boltzmann constant
    float dt,                          // Timestep
    int n_faces,
    int n_layers,
    int n_timesteps)
{
    int face = get_global_id(0);
    if (face >= n_faces) return;
    if (n_layers > MAX_LAYERS) return;

    // Private arrays - stored in GPU registers
    float T_local[MAX_LAYERS];
    float k_local[MAX_LAYERS];
    float cp_local[MAX_LAYERS];
    float kappa_local[MAX_LAYERS];
    float a[MAX_LAYERS];
    float b[MAX_LAYERS];
    float c_coef[MAX_LAYERS];
    float d[MAX_LAYERS];

    // Load initial temperature profile from global memory
    for (int i = 0; i < n_layers; ++i) {
        int idx = i * n_faces + face;
        T_local[i] = T_global[idx];
    }

    // Load boundary conditions (constant for all timesteps)
    float eps = emissivity[face];
    float T_bottom = T_deep[face];
    float Q_bottom = Qb[face];

    // Pre-load base conductivity and density (constant)
    float kc_local[MAX_LAYERS];
    float rho_local[MAX_LAYERS];
    for (int i = 0; i < n_layers; ++i) {
        int idx = i * n_faces + face;
        kc_local[i] = kc[idx];
        rho_local[i] = rho[idx];
    }

    // ========== TIMESTEP LOOP - ALL TIMESTEPS ON GPU ==========
    for (int t = 0; t < n_timesteps; ++t) {

        // Get flux for this timestep (coalesced read: timestep-major layout)
        float Q_surf = flux_all[t * n_faces + face];

        // Step 1: Update material properties k(T), cp(T), kappa(T)
        for (int i = 0; i < n_layers; ++i) {
            float T_val = T_local[i];
            k_local[i] = compute_k(T_val, kc_local[i], R350);
            cp_local[i] = compute_cp(T_val, cp0, cp_coeffs, n_cp_coeffs);
            kappa_local[i] = k_local[i] / (rho_local[i] * cp_local[i]);
        }

        // Step 2: Build tridiagonal system (Crank-Nicolson)

        // Surface layer (i=0)
        {
            float dz = z[1] - z[0];
            float dz2 = dz * dz;
            float Fo = kappa_local[0] * dt / dz2;

            float T3 = T_local[0] * T_local[0] * T_local[0];
            float rad = 4.0f * eps * sigma * T3 * dz / k_local[0];

            a[0] = 0.0f;
            b[0] = 1.0f + Fo + 0.5f * rad;
            c_coef[0] = -Fo;

            float Q_rad_old = eps * sigma * T_local[0] * T3;
            d[0] = (1.0f - Fo) * T_local[0] + Fo * T_local[1]
                 + dt * (Q_surf - Q_rad_old) / (rho_local[0] * cp_local[0] * dz);
        }

        // Interior layers - flux-based formulation matching CPU code
        // Uses k[i-1] for lower interface, k[i] for upper interface
        for (int i = 1; i < n_layers - 1; ++i) {
            float dz_minus = z[i] - z[i - 1];
            float dz_plus = z[i + 1] - z[i];
            float dz_avg = 0.5f * (dz_minus + dz_plus);

            // Grid coefficients (matching CPU g1, g2 formulation)
            float g1 = 1.0f / (dz_minus * dz_avg);  // coupling to i-1
            float g2 = 1.0f / (dz_plus * dz_avg);   // coupling to i+1

            // Flux-based: use k[i-1] for lower interface, k[i] for upper
            float alpha = g1 * k_local[i - 1];
            float beta = g2 * k_local[i];
            float inv_rhocp = dt / (rho_local[i] * cp_local[i]);

            a[i] = -0.5f * inv_rhocp * alpha;
            b[i] = 1.0f + 0.5f * inv_rhocp * (alpha + beta);
            c_coef[i] = -0.5f * inv_rhocp * beta;

            // RHS: Crank-Nicolson explicit part
            d[i] = T_local[i] + 0.5f * inv_rhocp * (
                alpha * T_local[i - 1]
                - (alpha + beta) * T_local[i]
                + beta * T_local[i + 1]);
        }

        // Bottom layer
        {
            int i = n_layers - 1;
            float dz = z[i] - z[i - 1];
            float dz2 = dz * dz;
            float Fo = kappa_local[i] * dt / dz2;

            a[i] = -0.5f * Fo;
            b[i] = 1.0f + 0.5f * Fo;
            c_coef[i] = 0.0f;

            d[i] = 0.5f * Fo * T_local[i - 1]
                 + (1.0f - 0.5f * Fo) * T_local[i]
                 + 0.5f * Fo * T_bottom;

            if (Q_bottom != 0.0f) {
                d[i] += dt * Q_bottom / (rho_local[i] * cp_local[i] * dz);
            }
        }

        // Step 3: Thomas algorithm - forward sweep
        for (int i = 1; i < n_layers; ++i) {
            float m = a[i] / b[i - 1];
            b[i] = b[i] - m * c_coef[i - 1];
            d[i] = d[i] - m * d[i - 1];
        }

        // Back substitution
        T_local[n_layers - 1] = d[n_layers - 1] / b[n_layers - 1];
        for (int i = n_layers - 2; i >= 0; --i) {
            T_local[i] = (d[i] - c_coef[i] * T_local[i + 1]) / b[i];
        }

        // Step 4: Store surface temperature for this timestep
        output_temps[t * n_faces + face] = T_local[0];

    } // End timestep loop

    // Write final temperature profile back to global memory
    for (int i = 0; i < n_layers; ++i) {
        int idx = i * n_faces + face;
        T_global[idx] = T_local[i];
        k_global[idx] = k_local[i];
        cp_global[idx] = cp_local[i];
        kappa_global[idx] = kappa_local[i];
    }
}

/**
 * @brief Simple vector addition kernel for testing
 */
__kernel void testKernel(
    __global float* result,
    __global const float* a,
    __global const float* b,
    int n)
{
    int i = get_global_id(0);
    if (i < n) {
        result[i] = a[i] + b[i];
    }
}
