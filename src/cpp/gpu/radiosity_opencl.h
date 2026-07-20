#pragma once

/**
 * @file radiosity_opencl.h
 * @brief OpenCL-based radiosity solver for PTM radiation exchange
 *
 * Provides GPU acceleration for the radiosity equation solver using
 * sparse matrix-vector multiplication (SpMV) with CSR format.
 * Works on macOS with AMD GPUs via OpenCL.
 */

#ifdef PTM_GPU_OPENCL

#include "opencl_config.h"
#include <vector>
#include <memory>
#include <string>

namespace gpu {
namespace opencl {

/**
 * @brief Configuration for OpenCL radiosity solver
 */
struct RadiosityOpenCLConfig {
    size_t work_group_size = 256;      // Work items per work group
    int max_iterations = 100;          // Max Jacobi iterations
    double convergence_tolerance = 1e-4;  // Convergence tolerance (safe for float precision)
    int convergence_check_interval = 10;  // Check convergence every N iterations
    bool warm_start = true;            // Warm start from previous solution
    bool verbose = false;
};

/**
 * @brief OpenCL-accelerated radiosity solver
 *
 * Solves the radiosity equation: B = E + rho * F * B + rho * Q_solar
 * where:
 *   B = radiosity (emitted + reflected radiation)
 *   E = thermal emission (epsilon * sigma * T^4)
 *   rho = reflectivity (1 - epsilon)
 *   F = view factor matrix (sparse, CSR format)
 *   Q_solar = incident solar flux
 *
 * Uses Jacobi iteration with GPU-accelerated sparse matrix-vector multiply.
 */
class RadiosityOpenCLSolver {
public:
    /**
     * @brief Constructor
     * @param n_faces Number of mesh faces
     * @param config Solver configuration
     */
    RadiosityOpenCLSolver(size_t n_faces, const RadiosityOpenCLConfig& config = RadiosityOpenCLConfig());

    ~RadiosityOpenCLSolver();

    // No copy
    RadiosityOpenCLSolver(const RadiosityOpenCLSolver&) = delete;
    RadiosityOpenCLSolver& operator=(const RadiosityOpenCLSolver&) = delete;

    /**
     * @brief Initialize OpenCL resources
     * @return true if initialization succeeded
     */
    bool initialize();

    /**
     * @brief Upload CSR view factor matrix (call once at start)
     *
     * @param values Non-zero values of view factor matrix
     * @param col_indices Column indices for each non-zero
     * @param row_ptrs Row pointers (size = n_faces + 1)
     */
    void uploadViewFactors(
        const std::vector<double>& values,
        const std::vector<int>& col_indices,
        const std::vector<int>& row_ptrs);

    /**
     * @brief Upload face properties (call once at start)
     *
     * @param emissivity Emissivity for each face
     * @param reflectivity Reflectivity (1 - emissivity) for each face
     */
    void uploadFaceProperties(
        const std::vector<float>& emissivity,
        const std::vector<float>& reflectivity);

    /**
     * @brief Solve radiosity equation for one timestep
     *
     * @param temperatures Surface temperatures [K] for all faces
     * @param solar_flux Incident solar flux [W/m^2] for all faces
     * @param indirect_flux_out Output: absorbed indirect flux [W/m^2] for all faces
     */
    void solve(
        const std::vector<float>& temperatures,
        const std::vector<float>& solar_flux,
        std::vector<float>& indirect_flux_out);

    /**
     * @brief Wait for all operations to complete
     */
    void synchronize();

    /**
     * @brief Get number of iterations in last solve
     */
    int getLastIterations() const { return last_iterations_; }

    /**
     * @brief Get residual from last solve
     */
    double getLastResidual() const { return last_residual_; }

    /**
     * @brief Check if last solve converged
     */
    bool didConverge() const { return converged_; }

    /**
     * @brief Check if solver is ready
     */
    bool isReady() const { return initialized_; }

    /**
     * @brief Get timing of last solve in milliseconds
     */
    float getLastSolveTimeMs() const { return last_solve_time_ms_; }

    /**
     * @brief Get device information
     */
    const OpenCLDeviceInfo& getDeviceInfo() const;

private:
    // Create OpenCL buffers
    void createBuffers();

    // Build kernel program
    void buildKernels();

    // Release resources
    void release();

    // Compute max residual from change buffer (CPU reduction)
    double computeMaxResidual();

    size_t n_faces_;
    size_t nnz_;  // Number of non-zeros in view factor matrix
    RadiosityOpenCLConfig config_;

    // OpenCL context
    std::unique_ptr<OpenCLContext> context_;

    // OpenCL program and kernels
    cl_program program_ = nullptr;
    cl_kernel spmv_kernel_ = nullptr;           // Scalar CSR SpMV (one thread per row)
    cl_kernel spmv_vector_kernel_ = nullptr;    // CSR-Vector SpMV (one warp per row)
    cl_kernel max_reduction_kernel_ = nullptr;  // Parallel max reduction
    cl_kernel jacobi_kernel_ = nullptr;         // Jacobi update
    cl_kernel emission_kernel_ = nullptr;       // Compute thermal emission
    cl_kernel solar_source_kernel_ = nullptr;   // Compute reflected solar
    cl_kernel indirect_flux_kernel_ = nullptr;  // Compute absorbed indirect flux

    // CSR matrix buffers (view factors)
    cl_mem buf_csr_values_ = nullptr;      // float [nnz]
    cl_mem buf_csr_col_indices_ = nullptr; // int [nnz]
    cl_mem buf_csr_row_ptrs_ = nullptr;    // int [n_faces + 1]

    // State buffers
    cl_mem buf_radiosity_ = nullptr;       // float [n_faces] - current radiosity
    cl_mem buf_radiosity_new_ = nullptr;   // float [n_faces] - next iteration
    cl_mem buf_F_B_ = nullptr;             // float [n_faces] - SpMV result (F * B)

    // Input/output buffers
    cl_mem buf_temperatures_ = nullptr;    // float [n_faces]
    cl_mem buf_solar_flux_ = nullptr;      // float [n_faces]
    cl_mem buf_emission_ = nullptr;        // float [n_faces] - epsilon * sigma * T^4
    cl_mem buf_solar_source_ = nullptr;    // float [n_faces] - rho * Q_solar
    cl_mem buf_indirect_flux_ = nullptr;   // float [n_faces] - output

    // Face property buffers
    cl_mem buf_emissivity_ = nullptr;      // float [n_faces]
    cl_mem buf_reflectivity_ = nullptr;    // float [n_faces]

    // Convergence checking
    cl_mem buf_change_ = nullptr;          // float [n_faces] - |B_new - B_old|
    cl_mem buf_partial_max_ = nullptr;     // float [n_work_groups] - partial max for reduction
    size_t n_reduction_groups_ = 0;        // Number of work groups for reduction

    // Host-side buffer for convergence reduction
    std::vector<float> change_host_;

    bool initialized_ = false;
    bool view_factors_uploaded_ = false;
    bool properties_uploaded_ = false;
    bool has_previous_solution_ = false;  // For warm starting

    // Convergence tracking
    int last_iterations_ = 0;
    double last_residual_ = 0.0;
    bool converged_ = false;
    float last_solve_time_ms_ = 0.0f;
};

/**
 * @brief Get embedded kernel source code
 *
 * Returns the radiosity kernel source as a string, avoiding the need
 * to distribute a separate .cl file.
 */
std::string getRadiosityKernelSource();

}  // namespace opencl
}  // namespace gpu

#endif  // PTM_GPU_OPENCL
