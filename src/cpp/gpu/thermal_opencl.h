#pragma once

/**
 * @file thermal_opencl.h
 * @brief OpenCL-based thermal solver for PTM
 *
 * Provides GPU acceleration via OpenCL, which works on:
 * - macOS with AMD GPUs
 * - Linux with AMD/NVIDIA/Intel GPUs
 * - Windows with various GPU vendors
 */

#ifdef PTM_GPU_OPENCL

#include "opencl_config.h"
#include <vector>
#include <memory>
#include <string>

namespace gpu {
namespace opencl {

/**
 * @brief Configuration for OpenCL thermal solver
 */
struct ThermalOpenCLConfig {
    size_t work_group_size = 256;    // Work items per work group
    bool verbose = false;
    std::string kernel_path = "";    // Path to .cl file (optional, uses embedded)
};

/**
 * @brief OpenCL-accelerated thermal solver
 *
 * Implements the same algorithm as the HIP/CUDA version but using OpenCL.
 * Works on macOS with AMD GPUs.
 */
class ThermalOpenCLSolver {
public:
    /**
     * @brief Constructor
     * @param n_faces Number of mesh faces
     * @param n_layers Number of depth layers per face
     * @param config Solver configuration
     */
    ThermalOpenCLSolver(size_t n_faces, size_t n_layers,
                        const ThermalOpenCLConfig& config = ThermalOpenCLConfig());

    ~ThermalOpenCLSolver();

    // No copy
    ThermalOpenCLSolver(const ThermalOpenCLSolver&) = delete;
    ThermalOpenCLSolver& operator=(const ThermalOpenCLSolver&) = delete;

    /**
     * @brief Initialize OpenCL resources
     * @return true if initialization succeeded
     */
    bool initialize();

    /**
     * @brief Upload mesh geometry and material properties
     */
    void uploadMeshData(
        const std::vector<float>& z_grid,
        const std::vector<float>& rho,
        const std::vector<float>& kc,
        const std::vector<float>& emissivity,
        const std::vector<float>& T_deep,
        const std::vector<float>& Qb);

    /**
     * @brief Set material property parameters
     */
    void setMaterialParams(
        float cp0,
        const std::vector<float>& cp_coeffs,
        float R350,
        float sigma = 5.67051196e-8f);

    /**
     * @brief Upload initial temperature profile
     */
    void uploadInitialTemperature(const std::vector<float>& T_initial);

    /**
     * @brief Upload surface flux for current timestep
     */
    void uploadFlux(const std::vector<float>& flux);

    /**
     * @brief Advance thermal solution by one timestep
     */
    void advanceTimestep(float dt);

    /**
     * @brief Download current surface temperatures
     */
    void downloadSurfaceTemps(std::vector<float>& T_surface);

    /**
     * @brief Download full temperature profiles
     */
    void downloadAllTemps(std::vector<float>& T_all);

    /**
     * @brief Wait for all operations to complete
     */
    void synchronize();

    /**
     * @brief Solve ALL timesteps in a single GPU kernel launch
     *
     * This persistent kernel approach minimizes CPU-GPU data transfer:
     * - Upload all flux data once
     * - Run all timesteps on GPU
     * - Download all surface temperatures once
     *
     * Expected 8-12x speedup over per-timestep approach.
     *
     * @param all_flux Absorbed flux for all timesteps [n_timesteps * n_faces]
     *                 Layout: timestep-major (flux[t * n_faces + face])
     * @param output_temps Output surface temperatures [n_timesteps * n_faces]
     *                     Layout: timestep-major
     * @param n_timesteps Number of timesteps to process
     * @param dt Timestep duration (seconds)
     */
    void solvePersistentAllTimesteps(
        const std::vector<float>& all_flux,
        std::vector<float>& output_temps,
        int n_timesteps,
        float dt);

    /**
     * @brief Run n_equil_cycles of the persistent kernel using the same
     * flux trajectory without reading output back between cycles, then run
     * one final "output" cycle and read its surface temperatures into
     * output_temps. Flux buffer is uploaded once; output buffer is
     * downloaded once. Equivalent to calling solvePersistentAllTimesteps
     * (n_equil_cycles + 1) times with the same arguments but avoids the
     * wasted per-cycle 2 * n_timesteps * n_faces * sizeof(float) of PCIe
     * traffic during equilibration.
     */
    void solvePersistentMultipleCycles(
        const std::vector<float>& all_flux,
        std::vector<float>& output_temps,
        int n_timesteps,
        float dt,
        int n_equil_cycles);

    /**
     * @brief Get timing of last kernel execution
     */
    float getLastKernelTimeMs() const { return last_kernel_time_ms_; }

    /**
     * @brief Check if solver is ready
     */
    bool isReady() const { return initialized_; }

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

    size_t n_faces_;
    size_t n_layers_;
    ThermalOpenCLConfig config_;

    // OpenCL context
    std::unique_ptr<OpenCLContext> context_;

    // OpenCL program and kernels
    cl_program program_ = nullptr;
    cl_kernel material_kernel_ = nullptr;
    cl_kernel thermal_kernel_ = nullptr;
    cl_kernel combined_kernel_ = nullptr;
    cl_kernel persistent_kernel_ = nullptr;

    // Device buffers
    cl_mem buf_T_ = nullptr;           // Temperature [n_faces * n_layers]
    cl_mem buf_k_ = nullptr;           // Conductivity
    cl_mem buf_cp_ = nullptr;          // Heat capacity
    cl_mem buf_kappa_ = nullptr;       // Diffusivity
    cl_mem buf_rho_ = nullptr;         // Density
    cl_mem buf_kc_ = nullptr;          // Base conductivity
    cl_mem buf_flux_ = nullptr;        // Surface flux [n_faces]
    cl_mem buf_emissivity_ = nullptr;  // Surface emissivity
    cl_mem buf_z_ = nullptr;           // Depth grid [n_layers]
    cl_mem buf_T_deep_ = nullptr;      // Deep boundary temp
    cl_mem buf_Qb_ = nullptr;          // Bottom heat flux
    cl_mem buf_cp_coeffs_ = nullptr;   // cp polynomial coefficients

    // Material parameters
    float cp0_ = 0.0f;
    float R350_ = 0.0f;
    float sigma_ = 5.67051196e-8f;
    int n_cp_coeffs_ = 0;

    bool initialized_ = false;
    float last_kernel_time_ms_ = 0.0f;
};

/**
 * @brief Get embedded kernel source code
 *
 * Returns the thermal kernel source as a string, avoiding the need
 * to distribute a separate .cl file.
 */
std::string getEmbeddedKernelSource();

/**
 * @brief Check if OpenCL is available on this system
 */
bool isOpenCLAvailable();

/**
 * @brief Print available OpenCL platforms and devices
 */
void printOpenCLInfo(std::ostream& out = std::cout);

}  // namespace opencl
}  // namespace gpu

#endif  // PTM_GPU_OPENCL
