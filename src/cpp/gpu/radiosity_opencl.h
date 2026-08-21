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
#include <cstdint>

namespace gpu {
namespace opencl {

/**
 * @brief Configuration for OpenCL radiosity solver
 */
struct RadiosityOpenCLConfig {
    size_t work_group_size = 256;      // Work items per work group
    int max_iterations = 100;          // Max Jacobi iterations
    double convergence_tolerance = 1e-4;  // Convergence tolerance (safe for float precision)
    // Check convergence every N Jacobi iterations. 0 = auto, resolved per
    // matrix by resolveConvergenceCheckInterval() below; > 0 pins it.
    int convergence_check_interval = 0;
    bool warm_start = true;            // Warm start from previous solution
    bool verbose = false;
};

/**
 * @brief Resolve the Jacobi convergence-check interval for one matrix.
 *
 * A convergence check costs one device->host round trip (a reduction kernel
 * plus a blocking read of the per-work-group partial maxima). Not checking
 * costs Jacobi iterations the solve did not need: the loop can only stop at a
 * multiple of the interval, so a solve that has already stagnated at iteration
 * 2 still runs to 10. Which of the two dominates is a property of the MATRIX,
 * not of the mesh's face count, so this compares the two costs directly:
 *
 *     t_traversal (per iteration) ~= nnz / kSpmvThroughputNnzPerSec
 *     t_check     (per check)     ~= kConvergenceReadSeconds
 *
 * and checks every iteration only when one traversal costs at least as much as
 * one check. Both constants are MEASURED on this machine's Radeon Pro W6800X,
 * 2026-08-15, and are the provenance of the threshold:
 *
 *   * 509,337-face moon3d CSR, nnz = 495,163,704: 24.2 ms per Jacobi traversal
 *     (=> 2.05e10 nnz/s) and 3.7 ms of fixed per-solve cost INCLUDING one
 *     convergence read -- so 3.7 ms is an upper bound on the read alone. The
 *     shipped interval of 10 made the GPU sweep 1.74x SLOWER than the CPU
 *     while the solve reported resid = 0 at iteration 10.
 *   * 12,800-face bowl crater, sparse VF (~2e6 nnz): 0.38 ms per traversal --
 *     64x cheaper, while the read is a latency and barely moves with size.
 *     Forcing a check every iteration there measured +29 % wall clock.
 *
 * The resulting threshold, nnz > 7.6e7, sits 6.5x below the flagship and 38x
 * above the crater: both measured arms are far from it, on the side their own
 * measurement demands. Deliberately a function of nnz and not of wall clock --
 * a timing-dependent interval would make the iteration count, and therefore
 * the bits of the answer, vary run to run.
 *
 * @param configured `RadiosityOpenCLConfig::convergence_check_interval`.
 * @param nnz Stored non-zeros of the view-factor matrix on this device.
 * @return Iterations between convergence checks (>= 1).
 */
int resolveConvergenceCheckInterval(int configured, size_t nnz);

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
     * @brief Initialize OpenCL resources on the first available GPU.
     * @return true if initialization succeeded
     */
    bool initialize();

    /**
     * @brief Initialize OpenCL resources pinned to a specific enumerated GPU.
     *
     * Used by RadiosityGPUDevicePool to spin up one solver per device. The
     * default-device path (`initialize()` above) is byte-identical to legacy
     * behavior so single-device callers are unaffected.
     *
     * @param device_handle Device handle from OpenCLContext::enumerateGPUs().
     * @return true if initialization succeeded.
     */
    bool initialize(const OpenCLDeviceInfo& device_handle);

    /**
     * @brief Upload CSR view factor matrix (call once at start)
     *
     * float32 IS the device format: the SpMV kernel reads a `__global const
     * float*`, so this overload hands the caller's array straight to
     * clEnqueueWriteBuffer with no host-side copy of the values, on both the
     * single-buffer and the chunked (PTM-25) path. A caller that already holds
     * float32 -- the PTM_HMATRIX_VF_REPLACE CSR is float32 on disk AND float32
     * in the CPU SparseVFOperator -- must use this overload, not the fp64 one.
     *
     * @param values Non-zero values of the view factor matrix (float32)
     * @param col_indices Column indices for each non-zero
     * @param row_ptrs Row pointers (size = n_faces + 1)
     */
    void uploadViewFactors(
        const std::vector<float>& values,
        const std::vector<int>& col_indices,
        const std::vector<int64_t>& row_ptrs);

    /**
     * @brief fp64 overload for callers holding a ViewFactorMatrix.
     *
     * ViewFactorMatrix stores values in float64 and the device stores float32,
     * so somebody has to narrow; this is the only place left on the upload path
     * that does. The narrowing granularity is unchanged from before the overload
     * split -- whole-matrix on the single-buffer path (bounded by the ~2 GB
     * per-buffer budget), per chunk on the chunked path -- so a huge float64
     * matrix never needs a second full copy of itself resident.
     *
     * float32 -> float64 -> float32 is value-preserving, so widening a float32
     * CSR just to come back through here is pure cost with no numerical effect:
     * a float32 caller should use the overload above.
     *
     * @param values Non-zero values of the view factor matrix (float64)
     * @param col_indices Column indices for each non-zero
     * @param row_ptrs Row pointers (size = n_faces + 1)
     */
    void uploadViewFactors(
        const std::vector<double>& values,
        const std::vector<int>& col_indices,
        const std::vector<int64_t>& row_ptrs);

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
     * @brief Upload the view-factor row normalization (call once at start)
     *
     * row_norm[i] = 1 / max(1, sum_j F_ij). Applied INSIDE the Jacobi iteration
     * and to the final absorbed flux, exactly as the CPU solver does, so a mesh
     * carrying sum_j F_ij > 1 rows runs the same operator on both paths. Never
     * calling this leaves the buffer at its 1.0 seed (identity), which is
     * bitwise the pre-row-norm kernel.
     *
     * @param row_norm Per-face 1/max(1, sum_j F_ij); must have n_faces entries.
     * @throws std::runtime_error on a size mismatch (never silently partial).
     */
    void uploadRowNorm(const std::vector<float>& row_norm);

    /**
     * @brief Upload the VISIBLE-band face properties (call once at start)
     *
     * The band-separated solve runs the same operator F against two different
     * property pairs: IR uses (eps, 1-eps) from uploadFaceProperties(), visible
     * uses (1-A, A) from this call. Uploading the visible pair to its OWN device
     * buffers is what lets solveTwoBandFused() advance both bands in one CSR
     * traversal; the legacy route instead re-uploaded the property pair twice per
     * timestep and ran two separate solves.
     *
     * Calling this also allocates the visible band's state buffers, so a caller
     * that never asks for two-band keeps exactly the legacy device footprint.
     *
     * @param emissivity_vis Visible-band absorptivity 1 - A_i (n_faces entries).
     * @param reflectivity_vis Visible-band reflectivity A_i (n_faces entries).
     * @throws std::runtime_error on a size mismatch (never silently partial).
     */
    void uploadVisibleProperties(const std::vector<float>& emissivity_vis,
                                 const std::vector<float>& reflectivity_vis);

    /// True once uploadVisibleProperties() has run, i.e. solveTwoBandFused() is usable.
    bool hasVisibleProperties() const { return visible_properties_uploaded_; }

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
     * @brief Band-separated (two-band) solve, FUSED over one CSR traversal.
     *
     * Solves both bands against the shared view-factor matrix:
     *   IR      : B_ir  = E + (1-eps) * norm * (F B_ir),    E = eps sigma T^4
     *   Visible : B_vis = A Q + A     * norm * (F B_vis)
     * and returns each band's absorbed indirect flux
     *   Q_ir  = eps   * norm * (F B_ir)
     *   Q_vis = (1-A) * norm * (F B_vis).
     *
     * This is the GPU counterpart of RadiositySolver::solveTwoBandJacobiFused():
     * while BOTH bands are still iterating, one width-2 SpMV reads each CSR row
     * once and accumulates both dot products, and one Jacobi launch advances both
     * band vectors. The instant a band meets the stopping criterion it is FROZEN
     * (its radiosity kept, its matvec skipped) and the slower band finishes on the
     * single-band kernels -- so a timestep costs max(iters_ir, iters_vis) + 1
     * traversals instead of (iters_ir + 1) + (iters_vis + 1).
     *
     * Per band the arithmetic is unchanged from two separate solve() calls: same
     * cold start, same update expression, same row-norm-in-kernel placement, same
     * per-row summation order and tree reduction, same convergence quantity and
     * the same check interval. The fused result is therefore expected to be
     * BIT-IDENTICAL to the unfused route, which is what test_two_band_rownorm's
     * B1f section asserts.
     *
     * @param temperatures Surface temperatures [K] (IR band source).
     * @param solar_flux Incident solar flux [W/m^2], already carrying any
     *        direct-beam source-albedo prescale (PTM-106) the caller applies.
     * @param indirect_ir_out Output: absorbed indirect IR flux [W/m^2].
     * @param indirect_vis_out Output: absorbed indirect visible flux [W/m^2].
     * @throws std::runtime_error if uploadVisibleProperties() has not run.
     */
    void solveTwoBandFused(
        const std::vector<float>& temperatures,
        const std::vector<float>& solar_flux,
        std::vector<float>& indirect_ir_out,
        std::vector<float>& indirect_vis_out);

    /**
     * @brief Wait for all operations to complete
     */
    void synchronize();

    /**
     * @brief Get number of iterations in last solve
     */
    int getLastIterations() const { return last_iterations_; }

    /// Jacobi iterations the IR band ran in the last solveTwoBandFused().
    int getLastIterationsIR() const { return last_iterations_ir_; }
    /// Jacobi iterations the visible band ran in the last solveTwoBandFused().
    int getLastIterationsVis() const { return last_iterations_vis_; }

    /**
     * @brief CSR traversals (matvecs) the last solve actually dispatched.
     *
     * The measured cost of a radiosity sweep at campaign scale is the number of
     * times the matrix is read, and that is NOT the iteration count: every solve
     * also pays one FINAL matvec to turn the converged radiosity into absorbed
     * flux, and the fused two-band path shares one traversal between the bands
     * while both iterate. Counting dispatches here (once per logical matvec,
     * regardless of how many chunk launches it takes) is the only honest way to
     * state a before/after traversal ratio -- deriving it from getLastIterations()
     * undercounts every solve by one and the unfused two-band route by two.
     */
    long getLastTraversals() const { return last_traversals_; }

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

    // Shared implementation of both uploadViewFactors() overloads. ValueT is
    // float (the device format -- written by reference, zero host copies) or
    // double (narrowed, at the granularity documented on the public overload).
    // Defined in the .cpp; both instantiations are used from that same TU.
    template <typename ValueT>
    void uploadViewFactorsT(const std::vector<ValueT>& values,
                            const std::vector<int>& col_indices,
                            const std::vector<int64_t>& row_ptrs);

    // Build the chunked CSR sub-buffers (PTM-25) when the matrix exceeds the
    // per-buffer chunk budget. Fails loud if a single row exceeds the budget.
    // float32 values are written to each chunk buffer directly out of the
    // caller's array; float64 values are narrowed one chunk at a time.
    template <typename ValueT>
    void buildChunkedCsrT(const std::vector<ValueT>& values,
                          const std::vector<int>& col_indices,
                          const std::vector<int64_t>& row_ptrs,
                          size_t max_chunk_nnz);

    // Enqueue y = F * x, dispatching to the single-buffer or chunked SpMV path.
    void enqueueSpMV(cl_mem x, cl_mem y, size_t global_size, size_t local_size);

    // Enqueue y1 = F * x1 and y2 = F * x2 in ONE structural traversal (width-2),
    // dispatching to the single-buffer or chunked kernel. Per band the per-row
    // stride and tree reduction are identical to enqueueSpMV(), so each output is
    // bit-identical to the single-band kernel's.
    void enqueueSpMV2(cl_mem x1, cl_mem x2, cl_mem y1, cl_mem y2);

    // Allocate the visible band's state buffers + the 4-segment band scratch.
    // Called from uploadVisibleProperties() so a gray-only caller never pays for
    // them. Idempotent.
    void createTwoBandBuffers();

    // Reduce the first `n_segments` segments of the residual scratch in ONE
    // kernel launch and ONE device->host read, writing their maxima to out[0..
    // n_segments). Same reduction kernel, work-group partitioning and host-side
    // ordering the pre-scratch single-quantity reduction used, so each value is
    // bit-identical to reducing that quantity's own n_faces-long array.
    //   n_segments = 2 : gray / single band  (|dB|, |B|)
    //   n_segments = 4 : fused two-band      (|dB_ir|, |B_ir|, |dB_vis|, |B_vis|)
    void computeScratchMaxima(int n_segments, double* out);

public:
    // Benchmark the isolated SpMV kernel (single-buffer path only): run the
    // chosen kernel n_reps times back-to-back with clFinish fences, return total
    // milliseconds. mode 0 = scalar (one work-item/row), 1 = vector (one
    // wavefront/row, coalesced). Uses the uploaded CSR + buf_radiosity_ as x.
    // THROWS if the matrix went through the chunked upload path (PTM-25) or no
    // CSR is resident: the single-buffer kernel arguments would otherwise be NULL
    // cl_mem handles, which OpenCL accepts, producing a timing measured against
    // null pointers instead of an error.
    double benchmarkSpMV(int n_reps, int mode);
private:

    // Release the chunked CSR sub-buffers and reset the chunked state (PTM-25).
    void releaseChunkBuffers();

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
    cl_kernel spmv_chunk_kernel_ = nullptr;     // Chunked CSR SpMV (PTM-25 workaround)
    cl_kernel spmv_vector_chunk_kernel_ = nullptr;  // Chunked CSR-Vector SpMV (coalesced)
    cl_kernel spmv_vector2_kernel_ = nullptr;       // Width-2 CSR-Vector SpMV (both bands)
    cl_kernel spmv_vector2_chunk_kernel_ = nullptr; // Width-2 chunked CSR-Vector SpMV
    cl_kernel jacobi2_kernel_ = nullptr;            // Fused two-band Jacobi update
    cl_kernel indirect_flux2_kernel_ = nullptr;     // Fused two-band absorbed flux

    // CSR matrix buffers (view factors)
    cl_mem buf_csr_values_ = nullptr;      // float [nnz]
    cl_mem buf_csr_col_indices_ = nullptr; // int [nnz]
    cl_mem buf_csr_row_ptrs_ = nullptr;    // int [n_faces + 1]

    // Chunked CSR (PTM-25): used only when a single values/col_indices buffer
    // would reach the 2^31-byte GPU addressing wall. The matrix is split on ROW
    // boundaries into sub-buffers each below the wall; the SpMV runs one launch
    // per chunk. Below the wall csr_chunks_ is empty and the single-buffer path
    // above is used unchanged (byte-identical).
    struct CsrChunk {
        cl_mem values = nullptr;      // float [nnz_chunk]  (chunk-local)
        cl_mem col_indices = nullptr; // int [nnz_chunk]    (GLOBAL column indices)
        cl_mem row_ptrs = nullptr;    // int [n_rows+1]     (chunk-local, starts at 0)
        int row_offset = 0;           // global row index of this chunk's first row
        int n_rows = 0;               // rows owned by this chunk
        size_t nnz = 0;               // non-zeros in this chunk
    };
    std::vector<CsrChunk> csr_chunks_;
    bool use_chunked_csr_ = false;

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
    cl_mem buf_row_norm_ = nullptr;        // float [n_faces] - 1/max(1, sum_j F_ij), seeded to 1

    // Visible-band state, allocated only when uploadVisibleProperties() runs
    // (createTwoBandBuffers). The IR band reuses the buffers above, so the gray
    // and IR arithmetic is literally the same code on the same memory.
    cl_mem buf_radiosity_vis_ = nullptr;      // float [n_faces]
    cl_mem buf_radiosity_vis_new_ = nullptr;  // float [n_faces]
    cl_mem buf_F_B_vis_ = nullptr;            // float [n_faces]
    cl_mem buf_emissivity_vis_ = nullptr;     // float [n_faces] - 1 - A
    cl_mem buf_reflectivity_vis_ = nullptr;   // float [n_faces] - A
    cl_mem buf_indirect_flux_vis_ = nullptr;  // float [n_faces]
    // Persistent all-zero source for the visible band's cold start.
    // ⛔ Do NOT replace this with clEnqueueFillBuffer: on this machine's
    // AMD Radeon PRO W6800X Duo (Apple OpenCL 1.2) clEnqueueFillBuffer returns
    // CL_SUCCESS and leaves the buffer holding GARBAGE (measured 2026-08-16 with
    // a standalone probe: a buffer seeded to 12345.0 came back as -1.57e6, and
    // the resulting solve started the visible band at |B| ~ 1e21 and needed 4x
    // the Jacobi iterations to decay back down). A device-to-device copy from a
    // buffer zeroed once by clEnqueueWriteBuffer is the reliable mechanism, and
    // it is also structurally identical to how the unfused route cold-starts
    // (clEnqueueCopyBuffer from its zero-valued emission buffer).
    cl_mem buf_zeros_ = nullptr;              // float [n_faces], all 0.0f

    // Residual scratch, allocated for every solver in createBuffers():
    //   gray / single band : [0*seg,1*seg) |dB|      [1*seg,2*seg) |B|
    //   fused two-band     : + [2*seg,3*seg) |dB_vis| [3*seg,4*seg) |B_vis|
    // seg = n_reduction_groups_ * work_group_size >= n_faces, so every segment
    // starts on a work-group boundary and one reduction launch over the segments
    // in use yields every maximum with a SINGLE device->host read -- the same one
    // round trip per check the old single-quantity reduction paid. The
    // [n_faces, seg) tail of each segment is zero-filled at creation and never
    // written, so the per-group partial maxima match the old reduction exactly.
    // Carrying |B| alongside |dB| is what makes a RELATIVE stopping criterion
    // free: it is reduced in the same launch as the absolute one.
    cl_mem buf_resid_scratch_ = nullptr;   // float [4 * seg]
    cl_mem buf_partial_max4_ = nullptr;    // float [4 * n_reduction_groups_]
    size_t band_segment_ = 0;              // seg

    // Convergence checking
    size_t n_reduction_groups_ = 0;        // Number of work groups for reduction

    bool initialized_ = false;
    bool view_factors_uploaded_ = false;
    bool properties_uploaded_ = false;
    bool visible_properties_uploaded_ = false;
    bool has_previous_solution_ = false;  // For warm starting

    // Convergence tracking
    int last_iterations_ = 0;
    int last_iterations_ir_ = 0;
    int last_iterations_vis_ = 0;
    long last_traversals_ = 0;   // CSR matvecs dispatched by the last solve
    double last_residual_ = 0.0;
    bool converged_ = false;
    float last_solve_time_ms_ = 0.0f;
    // Announce the resolved check interval once per solver (per device in a
    // pool), not once per timestep.
    bool logged_check_interval_ = false;
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
