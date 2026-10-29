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
#include "thermal/eq_extrapolation.h"
#include <cstddef>
#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include "../constants/stefan_boltzmann.h"  // ptm::kStefanBoltzmann (the single definition)

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
     * @param n_faces Number of mesh faces (treated as MAX/CAPACITY when used
     *                with setActiveFaceCount; long-lived buffers are sized to
     *                this value)
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
     * @brief Initialize OpenCL resources (default first-available GPU)
     * @return true if initialization succeeded
     */
    bool initialize();

    /**
     * @brief Initialize OpenCL resources pinned to a specific device.
     *
     * Used by ThermalGPUDevicePool to route a solver instance to one of
     * several enumerated GPUs. The single-device default initialize()
     * above is unchanged so legacy runs stay byte-identical.
     */
    bool initialize(const OpenCLDeviceInfo& device_handle);

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
     *
     * `cond_model` selects the regolith conductivity law: 0 = Hayne et al.
     * (2017) k = kc*(1+R350*T^3) (the historical default), 1 = Martinez &
     * Siegler (2021) Eq. 10. It is INTENTIONALLY not defaulted: a defaulted
     * argument would let a new GPU entry point silently run Hayne while the
     * config asked for Martinez, which is precisely the failure mode of
     * the built-in-body property override. Make the compiler ask.
     */
    void setMaterialParams(
        float cp0,
        const std::vector<float>& cp_coeffs,
        float R350,
        float sigma,
        int cond_model);

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
     * @brief Upload full temperature profiles into the GPU buffer.
     *
     * Inverse of downloadAllTemps. T_all must have size n_faces × n_layers
     * in the same SoA layout (layer-major: T[layer * n_faces + face]).
     * Used by the chunked-run workflow to warm-start a chunk from the
     * final thermal state of the previous chunk.
     */
    void uploadAllTemps(const std::vector<float>& T_all);

    /**
     * @brief Wait for all operations to complete
     */
    void synchronize();

    /**
     * @brief Print the per-run surface-BC certificate (max achieved
     *        |f| / (4 eps sigma T^3) in kelvin). Prints only when the running
     *        maximum has grown since the last print, so a chunked or
     *        multi-cycle run reports every new worst case exactly once.
     */
    void reportSurfaceBCCertificate();

    /**
     * @brief Optional subsurface temperature-map accumulator, handed to the
     * persistent kernel.
     *
     * Everything here is in the frame of the kernel call it is passed to: the
     * accumulation window is expressed in the SAME timestep indices the kernel
     * sees (relative to that call, then offset by t_offset for a self-chunked
     * dispatch), exactly like the Schorghofer accumulator's accum_start/end.
     *
     * `sum`, `vmax` and `vmin` are n_depths x n_faces_active float buffers,
     * depth-major. The caller owns sizing and INITIALIZATION: sum to 0,
     * vmax to -INFINITY, vmin to +INFINITY. They are uploaded, accumulated
     * into, and read back.
     *
     * `node_j` and `weight` are the plan's per-depth interpolation weights
     * (thermal::depthmap::Plan). They are never derived here, because the
     * refusal surface that produces them (a depth below the deepest node, a
     * name collision after snapping) belongs with the plan, not with a solver.
     */
    struct DepthAccumIO {
        std::vector<float>* sum  = nullptr;
        std::vector<float>* vmax = nullptr;
        std::vector<float>* vmin = nullptr;
        const std::vector<int>*   node_j = nullptr;
        const std::vector<float>* weight = nullptr;
        int accum_start = 0;
        int accum_end   = -1;   // -1 = n_timesteps
        // Set to true by the solver if the accumulators could not be read back
        // off the device. The caller MUST check it: on a failed read the host
        // buffers keep their seeds, and a seeded buffer merges without
        // complaint into a product whose per-face sample count is still
        // correct -- so nothing downstream would notice an all-sentinel max
        // map. `solvePersistent*` returns void, so this is the channel.
        bool* readback_failed = nullptr;
        bool valid() const {
            return sum && vmax && vmin && node_j && weight &&
                   !node_j->empty() && node_j->size() == weight->size();
        }
        std::size_t n_depths() const { return node_j ? node_j->size() : 0; }
    };

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
        float dt,
        // Optional Schorghofer & Taylor (2007) accumulator. When non-null,
        // ⟨E(T(z,t))⟩ is summed per face × layer over relative timesteps in
        // [accum_start, accum_end). Buffer must be n_layers × n_faces float
        // (caller responsible for sizing/zeroing). Pass nullptr to disable.
        std::vector<float>* avg_E_inout = nullptr,
        int accum_start = 0,
        int accum_end   = -1 /* -1 = n_timesteps */,
        // Optional subsurface depth-map accumulator; see DepthAccumIO.
        const DepthAccumIO* depth = nullptr);

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
        int n_equil_cycles,
        // Optional Schorghofer & Taylor (2007) accumulator. When non-null,
        // ⟨E(T(z,t))⟩ is accumulated only during the FINAL (output) cycle,
        // for relative timesteps in [accum_start, accum_end). Equilibration
        // cycles (cycles 0..n_equil_cycles-1) skip accumulation.
        std::vector<float>* avg_E_inout = nullptr,
        int accum_start = 0,
        int accum_end   = -1 /* -1 = n_timesteps */,
        // Pass-stride RRE extrapolation of the equilibration state. Pass
        // nullptr (the default) or an Options with enabled == false and this
        // function takes LITERALLY the pre-existing code path: no readback, no
        // upload, no branch inside the cycle loop beyond one bool test.
        //
        // Only ever supplied at the two EQUILIBRATION call sites. The output
        // cycle and the coupling re-thermal must never receive it: the output
        // window is not a fixed-point iteration at all, and the re-thermal
        // dispatches one pass per call (n_equil_cycles = 0), so a ring buffer
        // could never fill there anyway.
        const ptm::eqextrap::Options* extrap = nullptr,
        ptm::eqextrap::Stats* extrap_stats   = nullptr,
        // Optional subsurface depth-map accumulator; see DepthAccumIO. Like
        // avg_E_inout it is accumulated ONLY during the final (output) cycle.
        const DepthAccumIO* depth = nullptr);

    /**
     * @brief Set the active face count for the next call.
     *
     * Lets a single solver instance (sized for `n_faces` capacity at
     * construction) be reused for batches smaller than that capacity.
     * The next kernel launch will operate on the first `n` faces only;
     * upload and download helpers expect/return data sized to `n`.
     *
     * Why: re-creating a ThermalOpenCLSolver per batch is the failure
     * mode that wedged the macOS+AMD OpenCL driver during the
     * radiation-coupling re-thermal phase (see
     * memory/gpu_hang_amd_macos.md). Long-lived per-device solvers
     * eliminate context create/destroy churn.
     *
     * Asserts n <= n_faces (capacity). Pass 0 to revert to capacity.
     */
    void setActiveFaceCount(size_t n);

    /// Active face count (== capacity unless setActiveFaceCount was called).
    size_t activeFaceCount() const noexcept { return n_faces_active_; }

    /// Capacity (the n_faces passed to the constructor).
    size_t capacity() const noexcept { return n_faces_; }

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
    // Reduce the per-face surface-BC certificate into max_surf_resid_K_.
    void reduceSurfaceBCResidual(cl_command_queue queue);

    // Build kernel program
    void buildKernels();

    // Release resources
    void release();

    size_t n_faces_;          // capacity (max batch size; sets long-lived buffer sizes)
    size_t n_faces_active_;   // current active face count for kernel launches
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
    cl_mem buf_n_substeps_frozen_ = nullptr;  // per-face frozen substep count (R2 fix)
    cl_mem buf_surf_resid_ = nullptr;  // per-face surface-BC residual certificate [K]
    cl_mem buf_surf_iters_ = nullptr;  // per-face max surface-Newton step count
    cl_mem buf_cp_coeffs_ = nullptr;   // cp polynomial coefficients

    // Material parameters
    float cp0_ = 0.0f;
    float R350_ = 0.0f;
    // Conductivity law passed to every kernel (0 = Hayne 2017, 1 = Martinez 2021).
    int   cond_model_ = 0;
    float sigma_ = static_cast<float>(ptm::kStefanBoltzmann);
    int n_cp_coeffs_ = 0;

    bool initialized_ = false;
    float last_kernel_time_ms_ = 0.0f;
    // Per-run maximum of the surface Newton's achieved
    // |f| / (4 eps sigma T^3) [K], reduced from buf_surf_resid_.
    float max_surf_resid_K_ = 0.0f;
    float reported_surf_resid_K_ = -1.0f;
    int       max_surf_iters_    = 0;
    long long faces_at_iter_cap_ = 0;
};

/**
 * @brief Get embedded kernel source code
 *
 * Returns the thermal kernel source as a string, avoiding the need
 * to distribute a separate .cl file.
 */
std::string getEmbeddedKernelSource();

/**
 * @brief In-process override of the per-dispatch self-chunk ceiling.
 *
 * The consumer gpuMaxTimestepsPerDispatch() (thermal_opencl.cpp) honors a
 * positive override set here BEFORE consulting its static env/profile cache.
 * This lets the in-run auto-calibration inject the ceiling it measured for the
 * loaded run's grid without a disk round-trip — necessary because the probe
 * itself dispatches the persistent kernel, which would otherwise latch a
 * conservative ceiling into that static cache and defeat the calibrated value.
 *
 * A non-positive @p max_timesteps_per_dispatch clears the override. This is a
 * no-op when PTM_MAX_TIMESTEPS_PER_DISPATCH is set in the environment — the
 * operator's explicit ceiling always wins. Thread-safe (atomic).
 */
void setDispatchCeilingOverride(int max_timesteps_per_dispatch);

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
