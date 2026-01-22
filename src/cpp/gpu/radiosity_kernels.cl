/**
 * @file radiosity_kernels.cl
 * @brief OpenCL kernels for PTM radiosity solver
 *
 * Implements GPU-accelerated radiosity equation solver using sparse
 * matrix-vector multiplication (SpMV) with CSR format for view factors.
 *
 * Memory layout: Contiguous arrays for coalesced access
 */

// Stefan-Boltzmann constant (can be overridden at compile time)
#ifndef SIGMA
#define SIGMA 5.67051196e-8f
#endif

/**
 * @brief Sparse matrix-vector multiply (CSR format)
 *
 * Computes y = A * x where A is stored in CSR format.
 * Each work item handles one row (one face).
 *
 * @param values Non-zero values of the matrix
 * @param col_indices Column index for each non-zero
 * @param row_ptrs Row pointers (size = n_rows + 1)
 * @param x Input vector
 * @param y Output vector
 * @param n_rows Number of rows (faces)
 */
__kernel void spmv_kernel(
    __global const float* values,
    __global const int* col_indices,
    __global const int* row_ptrs,
    __global const float* x,
    __global float* y,
    int n_rows)
{
    int row = get_global_id(0);
    if (row >= n_rows) return;

    int row_start = row_ptrs[row];
    int row_end = row_ptrs[row + 1];

    float sum = 0.0f;
    for (int j = row_start; j < row_end; ++j) {
        sum += values[j] * x[col_indices[j]];
    }
    y[row] = sum;
}

/**
 * @brief Compute thermal emission: E = epsilon * sigma * T^4
 *
 * Each work item handles one face.
 *
 * @param temperatures Surface temperatures [K]
 * @param emissivity Emissivity for each face
 * @param emission Output: thermal emission [W/m^2]
 * @param n_faces Number of faces
 */
__kernel void emission_kernel(
    __global const float* temperatures,
    __global const float* emissivity,
    __global float* emission,
    int n_faces)
{
    int i = get_global_id(0);
    if (i >= n_faces) return;

    float T = temperatures[i];
    float T2 = T * T;
    float T4 = T2 * T2;
    emission[i] = emissivity[i] * SIGMA * T4;
}

/**
 * @brief Compute reflected solar source: rho * Q_solar
 *
 * Each work item handles one face.
 *
 * @param solar_flux Incident solar flux [W/m^2]
 * @param reflectivity Reflectivity (1 - emissivity)
 * @param solar_source Output: reflected solar flux
 * @param n_faces Number of faces
 */
__kernel void solar_source_kernel(
    __global const float* solar_flux,
    __global const float* reflectivity,
    __global float* solar_source,
    int n_faces)
{
    int i = get_global_id(0);
    if (i >= n_faces) return;

    solar_source[i] = reflectivity[i] * solar_flux[i];
}

/**
 * @brief Initialize radiosity with emission + solar source
 *
 * B_0 = E + rho * Q_solar
 *
 * @param emission Thermal emission [W/m^2]
 * @param solar_source Reflected solar [W/m^2]
 * @param radiosity Output: initial radiosity
 * @param n_faces Number of faces
 */
__kernel void init_radiosity_kernel(
    __global const float* emission,
    __global const float* solar_source,
    __global float* radiosity,
    int n_faces)
{
    int i = get_global_id(0);
    if (i >= n_faces) return;

    radiosity[i] = emission[i] + solar_source[i];
}

/**
 * @brief Jacobi iteration update
 *
 * B_new = E + rho * Q_solar + rho * (F * B_old)
 *
 * Also computes change = |B_new - B_old| for convergence check.
 *
 * @param emission Thermal emission [W/m^2]
 * @param solar_source Reflected solar [W/m^2]
 * @param reflectivity Reflectivity for each face
 * @param F_B Result of SpMV: F * B_old
 * @param radiosity_old Current radiosity
 * @param radiosity_new Output: updated radiosity
 * @param change Output: |B_new - B_old| for convergence
 * @param n_faces Number of faces
 */
__kernel void jacobi_kernel(
    __global const float* emission,
    __global const float* solar_source,
    __global const float* reflectivity,
    __global const float* F_B,
    __global const float* radiosity_old,
    __global float* radiosity_new,
    __global float* change,
    int n_faces)
{
    int i = get_global_id(0);
    if (i >= n_faces) return;

    // B_new = E + rho * Q_solar + rho * (F * B)
    float B_new = emission[i] + solar_source[i] + reflectivity[i] * F_B[i];

    radiosity_new[i] = B_new;
    change[i] = fabs(B_new - radiosity_old[i]);
}

/**
 * @brief Compute absorbed indirect flux from final radiosity
 *
 * Q_indirect = epsilon * (F * B)
 *
 * This gives the radiation absorbed by each face from all other faces.
 *
 * @param emissivity Emissivity (= absorptivity by Kirchhoff)
 * @param F_B Result of final SpMV: F * B_final
 * @param indirect_flux Output: absorbed indirect flux [W/m^2]
 * @param n_faces Number of faces
 */
__kernel void indirect_flux_kernel(
    __global const float* emissivity,
    __global const float* F_B,
    __global float* indirect_flux,
    int n_faces)
{
    int i = get_global_id(0);
    if (i >= n_faces) return;

    // By Kirchhoff's law, absorptivity = emissivity
    indirect_flux[i] = emissivity[i] * F_B[i];
}

/**
 * @brief Partial reduction for max finding (optional optimization)
 *
 * Reduces the change array to find maximum per work group.
 * The final max is computed on CPU from work group results.
 *
 * @param change Per-element change values
 * @param partial_max Output: max within each work group
 * @param n Number of elements
 */
__kernel void reduce_max_kernel(
    __global const float* change,
    __global float* partial_max,
    __local float* scratch,
    int n)
{
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int group_size = get_local_size(0);
    int group_id = get_group_id(0);

    // Load into local memory
    scratch[lid] = (gid < n) ? change[gid] : 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Parallel reduction for max
    for (int stride = group_size / 2; stride > 0; stride /= 2) {
        if (lid < stride) {
            scratch[lid] = fmax(scratch[lid], scratch[lid + stride]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write result
    if (lid == 0) {
        partial_max[group_id] = scratch[0];
    }
}

/**
 * @brief Simple vector copy kernel
 *
 * dst = src
 */
__kernel void copy_kernel(
    __global const float* src,
    __global float* dst,
    int n)
{
    int i = get_global_id(0);
    if (i >= n) return;
    dst[i] = src[i];
}

/**
 * @brief Test kernel for validation
 *
 * result = a + b
 */
__kernel void test_add_kernel(
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
