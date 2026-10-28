/**
 * @file radiosity_opencl.cpp
 * @brief OpenCL radiosity solver implementation
 */

#ifdef PTM_GPU_OPENCL

#include "radiosity_opencl.h"
#include <fstream>
#include <sstream>
#include <chrono>
#include <type_traits>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <atomic>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "../constants/stefan_boltzmann.h"  // ptm::kStefanBoltzmann

namespace gpu {
namespace opencl {

// Embedded kernel source for radiosity solver
std::string getRadiosityKernelSource() {
    return R"CL_SOURCE(
// OpenCL kernels for PTM radiosity solver
// Implements Jacobi iteration for radiosity equation: B = E + rho * F * B

// Stefan-Boltzmann. Checked against ptm::kStefanBoltzmann by the static_assert
// after this function; OpenCL C cannot include the C++ header.
#define SIGMA_SB 5.670374419e-8f

// Sparse matrix-vector multiply (CSR format) - Scalar version
// Each work item handles one row of the matrix
// Simple but has warp divergence for variable row lengths
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

// Chunked SpMV (PTM-25 workaround for the 2^31-byte single-buffer wall).
// Identical math to spmv_kernel but on ONE row-chunk whose values/col_indices
// buffers each stay below 2^31 bytes, so the int j byte offset cannot overflow.
// Chunk-local row_ptrs start at 0; col_indices remain GLOBAL indices into the
// full radiosity vector x[n_faces]. Bit-identical to spmv_kernel (same per-row
// summation order).
__kernel void spmv_chunk_kernel(
    __global const float* values,
    __global const int* col_indices,
    __global const int* row_ptrs,
    __global const float* x,
    __global float* y,
    int row_offset,
    int n_rows_chunk)
{
    int lr = get_global_id(0);
    if (lr >= n_rows_chunk) return;

    int row_start = row_ptrs[lr];
    int row_end = row_ptrs[lr + 1];

    float sum = 0.0f;
    for (int j = row_start; j < row_end; ++j) {
        sum += values[j] * x[col_indices[j]];
    }
    y[row_offset + lr] = sum;
}

// Chunked CSR-Vector SpMV (PTM-25 chunking + wavefront-per-row coalescing).
// One work-group per chunk-local row; the wavefront strides the row so global
// loads coalesce (~5x the scalar kernel's bandwidth). Same per-row tree
// reduction as spmv_vector_kernel, so single-buffer and chunked vector paths
// are bit-identical for a given row.
__kernel void spmv_vector_chunk_kernel(
    __global const float* values,
    __global const int* col_indices,
    __global const int* row_ptrs,   /* chunk-local, starts at 0 */
    __global const float* x,        /* full radiosity [n_faces] */
    __global float* y,              /* full output [n_faces] */
    int row_offset,
    int n_rows_chunk,
    __local float* shared_sum)
{
    int lr = get_group_id(0);
    int lid = get_local_id(0);
    int warp = get_local_size(0);
    if (lr >= n_rows_chunk) return;

    int row_start = row_ptrs[lr];
    int row_end = row_ptrs[lr + 1];

    float sum = 0.0f;
    for (int j = row_start + lid; j < row_end; j += warp) {
        sum += values[j] * x[col_indices[j]];
    }
    shared_sum[lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = warp / 2; s > 0; s >>= 1) {
        if (lid < s) shared_sum[lid] += shared_sum[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) y[row_offset + lr] = shared_sum[0];
}

// CSR-Vector SpMV: One work group per row
// Better warp utilization - threads cooperatively process row elements
// Work group size should match wavefront/warp size (64 for AMD, 32 for NVIDIA)
__kernel void spmv_vector_kernel(
    __global const float* values,
    __global const int* col_indices,
    __global const int* row_ptrs,
    __global const float* x,
    __global float* y,
    int n_rows,
    __local float* shared_sum)
{
    int row = get_group_id(0);
    int lid = get_local_id(0);
    int warp_size = get_local_size(0);

    if (row >= n_rows) return;

    int row_start = row_ptrs[row];
    int row_end = row_ptrs[row + 1];

    // Threads cooperatively process row elements
    float sum = 0.0f;
    for (int j = row_start + lid; j < row_end; j += warp_size) {
        sum += values[j] * x[col_indices[j]];
    }

    // Store partial sum in local memory
    shared_sum[lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Tree reduction within work group
    for (int s = warp_size / 2; s > 0; s >>= 1) {
        if (lid < s) {
            shared_sum[lid] += shared_sum[lid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // First thread writes result
    if (lid == 0) {
        y[row] = shared_sum[0];
    }
}

// Width-2 CSR-Vector SpMV: y1 = F*x1 AND y2 = F*x2 in ONE structural traversal.
// The two bands of a band-separated radiosity solve share the operator F and
// differ only in their sources and albedos, so reading each row's (col, val)
// once and accumulating two dot products halves the CSR traffic of the sweep.
// Per band the stride (row_start + lid, step warp_size) and the tree reduction
// are IDENTICAL to spmv_vector_kernel, so each output is bit-identical to what
// the single-band kernel produces for the same x -- the fusion is a traversal
// optimization, not a numerical change. `shared` must hold 2 * warp_size floats
// (band 1 in [0, warp), band 2 in [warp, 2*warp)).
__kernel void spmv_vector2_kernel(
    __global const float* values,
    __global const int* col_indices,
    __global const int* row_ptrs,
    __global const float* x1,
    __global const float* x2,
    __global float* y1,
    __global float* y2,
    int n_rows,
    __local float* shared)
{
    int row = get_group_id(0);
    int lid = get_local_id(0);
    int warp_size = get_local_size(0);

    if (row >= n_rows) return;

    int row_start = row_ptrs[row];
    int row_end = row_ptrs[row + 1];

    float sum1 = 0.0f;
    float sum2 = 0.0f;
    for (int j = row_start + lid; j < row_end; j += warp_size) {
        float v = values[j];
        int c = col_indices[j];
        sum1 += v * x1[c];
        sum2 += v * x2[c];
    }

    shared[lid] = sum1;
    shared[warp_size + lid] = sum2;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = warp_size / 2; s > 0; s >>= 1) {
        if (lid < s) {
            shared[lid] += shared[lid + s];
            shared[warp_size + lid] += shared[warp_size + lid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        y1[row] = shared[0];
        y2[row] = shared[warp_size];
    }
}

// Width-2 chunked CSR-Vector SpMV (PTM-25 chunking + band fusion). Same relation
// to spmv_vector_chunk_kernel that spmv_vector2_kernel has to spmv_vector_kernel:
// chunk-local row_ptrs, GLOBAL column indices, disjoint output row ranges.
__kernel void spmv_vector2_chunk_kernel(
    __global const float* values,
    __global const int* col_indices,
    __global const int* row_ptrs,   /* chunk-local, starts at 0 */
    __global const float* x1,       /* full radiosity [n_faces] */
    __global const float* x2,
    __global float* y1,             /* full output [n_faces] */
    __global float* y2,
    int row_offset,
    int n_rows_chunk,
    __local float* shared)          /* 2 * warp floats */
{
    int lr = get_group_id(0);
    int lid = get_local_id(0);
    int warp = get_local_size(0);
    if (lr >= n_rows_chunk) return;

    int row_start = row_ptrs[lr];
    int row_end = row_ptrs[lr + 1];

    float sum1 = 0.0f;
    float sum2 = 0.0f;
    for (int j = row_start + lid; j < row_end; j += warp) {
        float v = values[j];
        int c = col_indices[j];
        sum1 += v * x1[c];
        sum2 += v * x2[c];
    }
    shared[lid] = sum1;
    shared[warp + lid] = sum2;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = warp / 2; s > 0; s >>= 1) {
        if (lid < s) {
            shared[lid] += shared[lid + s];
            shared[warp + lid] += shared[warp + lid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        y1[row_offset + lr] = shared[0];
        y2[row_offset + lr] = shared[warp];
    }
}

// Parallel max reduction kernel
// Each work group produces one partial max
// Two-stage approach: first pass reduces to partial maxes, second pass to final
__kernel void max_reduction_kernel(
    __global const float* input,
    __global float* partial_max,
    int n,
    __local float* shared)
{
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int wg_size = get_local_size(0);
    int group_id = get_group_id(0);

    // Load value (or 0 if out of bounds)
    float val = (gid < n) ? input[gid] : 0.0f;
    shared[lid] = val;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Tree reduction for max
    for (int s = wg_size / 2; s > 0; s >>= 1) {
        if (lid < s) {
            shared[lid] = fmax(shared[lid], shared[lid + s]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // First thread in group writes partial result
    if (lid == 0) {
        partial_max[group_id] = shared[0];
    }
}

// Compute thermal emission: E = epsilon * sigma * T^4
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
    emission[i] = emissivity[i] * SIGMA_SB * T4;
}

// Compute reflected solar source: rho * Q_solar
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

// Jacobi iteration update: B_new = E + solar + rho * (rownorm * (F * B_old))
//
// row_norm[i] = 1 / max(1, sum_j F_ij) is the SAME energy-conservation
// renormalization the CPU Jacobi applies to every matvec result
// (radiation_exchange.cpp solveJacobi / solveTwoBandFused). Applying it here,
// inside the iteration, is what makes the GPU Jacobi operator identical to the
// CPU one on a mesh with sum_j F_ij > 1 rows -- previously the GPU applied the
// row norm only to the FINAL flux, so its iteration matrix was rho*F rather
// than rho*(norm.F) and the visible band could gain > 1. Where every row sum is
// <= 1 the buffer is exactly 1.0f and `1.0f * x == x` bitwise, so this is
// byte-identical to the pre-row-norm kernel on such meshes.
// The two convergence quantities go into the shared residual scratch as two
// work-group-aligned segments of `seg` floats -- |dB| then |B| -- so ONE
// reduction launch and ONE device->host read serve both. |B| is what a RELATIVE
// stopping criterion needs; reducing it in the same launch is why testing
// max|dB| / max|B| costs the solve nothing extra over testing max|dB| alone.
__kernel void jacobi_kernel(
    __global const float* emission,
    __global const float* solar_source,
    __global const float* reflectivity,
    __global const float* F_B,
    __global const float* row_norm,
    __global const float* radiosity_old,
    __global float* radiosity_new,
    __global float* resid_scratch,
    int seg,
    int n_faces)
{
    int i = get_global_id(0);
    if (i >= n_faces) return;

    float B_new = emission[i] + solar_source[i] + reflectivity[i] * (row_norm[i] * F_B[i]);
    radiosity_new[i] = B_new;
    float diff = B_new - radiosity_old[i];
    resid_scratch[i] = (diff >= 0.0f) ? diff : -diff;
    resid_scratch[seg + i] = (B_new >= 0.0f) ? B_new : -B_new;
}

// Fused two-band Jacobi update: ONE launch advances both band vectors.
//
//   B_ir_new  = E     + (1-eps) * (rownorm * (F B_ir))
//   B_vis_new = A Q   + A       * (rownorm * (F B_vis))
//
// Term for term this is jacobi_kernel run with each band's own (source, refl)
// pair -- the IR band's zero solar source and the visible band's zero emission
// are simply not added, and `x + 0.0f == x` exactly, so each band's update is
// bit-identical to the corresponding single-band launch.
//
// `active_mask` implements the per-band FREEZE that the CPU fused solver uses:
// bit 0 = IR still iterating, bit 1 = visible still iterating. A frozen band's
// radiosity and scratch entries are left untouched, so it holds exactly the
// value its own standalone solve would have stopped at.
//
// The four convergence quantities go into the same scratch buffer jacobi_kernel
// uses, as four work-group-aligned segments of `seg` floats -- |dB_ir|, |B_ir|,
// |dB_vis|, |B_vis| -- so a single reduction launch and a single device->host
// read serve both bands and both stopping criteria (absolute and relative).
__kernel void jacobi2_kernel(
    __global const float* ir_source,      /* E = eps sigma T^4 */
    __global const float* ir_refl,        /* 1 - eps */
    __global const float* ir_F_B,
    __global const float* vis_source,     /* A * Q_solar */
    __global const float* vis_refl,       /* A */
    __global const float* vis_F_B,
    __global const float* row_norm,
    __global const float* B_ir_old,
    __global float* B_ir_new,
    __global const float* B_vis_old,
    __global float* B_vis_new,
    __global float* band_scratch,
    int seg,
    int active_mask,
    int n_faces)
{
    int i = get_global_id(0);
    if (i >= n_faces) return;

    float nrm = row_norm[i];

    if (active_mask & 1) {
        float b = ir_source[i] + ir_refl[i] * (nrm * ir_F_B[i]);
        B_ir_new[i] = b;
        float d = b - B_ir_old[i];
        band_scratch[i] = (d >= 0.0f) ? d : -d;
        band_scratch[seg + i] = (b >= 0.0f) ? b : -b;
    }
    if (active_mask & 2) {
        float b = vis_source[i] + vis_refl[i] * (nrm * vis_F_B[i]);
        B_vis_new[i] = b;
        float d = b - B_vis_old[i];
        band_scratch[2 * seg + i] = (d >= 0.0f) ? d : -d;
        band_scratch[3 * seg + i] = (b >= 0.0f) ? b : -b;
    }
}

// Fused two-band absorbed indirect flux. Per band identical to
// indirect_flux_kernel (same absorptivity * (rownorm * F B) product), evaluated
// for both bands off the single width-2 final traversal.
__kernel void indirect_flux2_kernel(
    __global const float* ir_absorb,   /* eps */
    __global const float* ir_F_B,
    __global const float* vis_absorb,  /* 1 - A */
    __global const float* vis_F_B,
    __global const float* row_norm,
    __global float* ir_out,
    __global float* vis_out,
    int n_faces)
{
    int i = get_global_id(0);
    if (i >= n_faces) return;

    float nrm = row_norm[i];
    ir_out[i]  = ir_absorb[i]  * (nrm * ir_F_B[i]);
    vis_out[i] = vis_absorb[i] * (nrm * vis_F_B[i]);
}

// Compute absorbed indirect flux: Q_indirect = epsilon * (rownorm * (F * B))
// Same row normalization as jacobi_kernel; the host no longer post-multiplies.
__kernel void indirect_flux_kernel(
    __global const float* emissivity,
    __global const float* F_B,
    __global const float* row_norm,
    __global float* indirect_flux,
    int n_faces)
{
    int i = get_global_id(0);
    if (i >= n_faces) return;

    indirect_flux[i] = emissivity[i] * (row_norm[i] * F_B[i]);
}
)CL_SOURCE";
}

static_assert(ptm::kStefanBoltzmann == 5.670374419e-8,
              "ptm::kStefanBoltzmann changed. Update the `#define SIGMA_SB` in "
              "getRadiosityKernelSource(), then update this assert.");

// Convergence-check interval, auto-resolved from the cost of a traversal
// against the cost of a check. The two constants below are measurements, not
// tuning: see the derivation and provenance in the header's doc comment for
// resolveConvergenceCheckInterval(). Changing either changes results at scale,
// so change them only with a new measurement.
namespace {
// Jacobi SpMV throughput on the Radeon Pro W6800X, from the 495,163,704-nnz
// moon3d CSR at 24.2 ms per traversal (2026-08-15).
constexpr double kSpmvThroughputNnzPerSec = 2.05e10;
// Upper bound on one device->host convergence read: the whole fixed per-solve
// cost of the same arm, which already includes one read (2026-08-15).
constexpr double kConvergenceReadSeconds = 3.7e-3;
// Interval used when a traversal is cheaper than a check. This is the shipped
// historical value, kept so that small meshes stay bit-for-bit on the behavior
// they were validated against.
constexpr int kCheapTraversalInterval = 10;
}  // namespace

int resolveConvergenceCheckInterval(int configured, size_t nnz) {
    if (configured > 0) return configured;   // pinned by the caller
    const double t_traversal = static_cast<double>(nnz) / kSpmvThroughputNnzPerSec;
    return (t_traversal >= kConvergenceReadSeconds) ? 1 : kCheapTraversalInterval;
}

namespace {
// Float32 floor on the RELATIVE residual. The reduction that produces
// max|B_new - B_old| is a float32 sum: at B ~ 1e3 W/m^2 one ulp is 6.1e-5, so
// max|dB| / max|B| cannot resolve below ~6e-8 no matter how converged the solve
// is. 1e-6 sits an order of magnitude above that floor. This replaces the old
// absolute floor of 1e-5 W/m^2, which was the same guard expressed in the wrong
// units: on a bright surface it demanded a residual ~900x tighter than the
// configured relative tolerance, and on a dark one it let the solve stop early.
constexpr double kFloat32RelResidualFloor = 1e-6;

// Restore the legacy ABSOLUTE stopping test (max|dB| < tol, floored at 1e-5
// W/m^2). Exists so the criterion change can be measured against the route it
// replaces on one binary, and so a future divergence can be bisected without a
// rebuild -- not because the absolute test is still a supported physics option.
// Announced once, because a quietly different convergence rule is exactly the
// kind of silent substitution that produced this defect.
bool useLegacyAbsoluteCriterion() {
    static const bool on = []() {
        if (!std::getenv("PTM_RAD_GPU_ABS_CRITERION")) return false;
        std::cout << "  [Radiosity GPU] PTM_RAD_GPU_ABS_CRITERION set: using the LEGACY "
                     "ABSOLUTE stopping test max|dB| < max(tol, 1e-5) W/m^2 instead of the "
                     "CPU's relative max|dB|/max|B| < tol. At B ~ 900 W/m^2 that bar is "
                     "~900x stricter, so the solve does strictly more Jacobi iterations for "
                     "the same configured tolerance. Diagnostic only.\n";
        return true;
    }();
    return on;
}

// The one definition of "this band has converged", shared by the gray and the
// fused two-band GPU paths and matching RadiositySolver::solveJacobi() /
// solveTwoBandJacobiFused() line for line:
//     rel = (max|B| > 1e-10) ? max|dB| / max|B| : max|dB|;  converged iff rel < tol
// The 1e-10 guard is the CPU's, verbatim: it keeps an all-zero band (a fully
// shadowed visible band, say) from dividing by zero and instead scores it on the
// absolute change, which for an all-zero band is also zero.
struct StopTest {
    double residual;   // the quantity compared against the tolerance
    bool   converged;
};
StopTest radiosityStopTest(double max_change, double max_val, double tolerance) {
    if (useLegacyAbsoluteCriterion()) {
        const double tol_abs = std::max(tolerance, 1e-5);
        return {max_change, max_change < tol_abs};
    }
    const double tol_rel = std::max(tolerance, kFloat32RelResidualFloor);
    const double rel = (max_val > 1e-10) ? max_change / max_val : max_change;
    return {rel, rel < tol_rel};
}
}  // namespace

// RadiosityOpenCLSolver implementation

RadiosityOpenCLSolver::RadiosityOpenCLSolver(size_t n_faces,
                                             const RadiosityOpenCLConfig& config)
    : n_faces_(n_faces)
    , nnz_(0)
    , config_(config)
{
    // Diagnostic override for the convergence-check interval. Exists so the
    // auto rule above can be falsified at a new scale without a rebuild; it
    // changes how many Jacobi iterations run, so it is never silent.
    if (const char* e = std::getenv("PTM_RAD_GPU_CHECK_INTERVAL")) {
        const long long v = std::atoll(e);
        if (v > 0 && v <= 1000000) {
            config_.convergence_check_interval = static_cast<int>(v);
            std::cout << "  [Radiosity GPU] PTM_RAD_GPU_CHECK_INTERVAL=" << v
                      << ": pinning the convergence-check interval (overrides the "
                         "auto rule).\n";
        } else {
            std::cerr << "  [Radiosity GPU] WARNING: ignoring "
                         "PTM_RAD_GPU_CHECK_INTERVAL='" << e
                      << "' (must be a positive integer <= 1000000); the "
                         "configured/auto interval stands.\n";
        }
    }
}

RadiosityOpenCLSolver::~RadiosityOpenCLSolver() {
    release();
}

bool RadiosityOpenCLSolver::initialize() {
    if (initialized_) return true;

    // Create OpenCL context
    context_ = std::make_unique<OpenCLContext>();
    if (!context_->initialize(true)) {  // Prefer GPU
        if (config_.verbose) {
            std::cerr << "Failed to initialize OpenCL context for radiosity solver\n";
        }
        return false;
    }

    if (config_.verbose) {
        printDeviceInfo(context_->deviceInfo());
    }

    try {
        buildKernels();
        createBuffers();
    } catch (const std::exception& e) {
        if (config_.verbose) {
            std::cerr << "OpenCL initialization error: " << e.what() << "\n";
        }
        release();
        return false;
    }

    initialized_ = true;

    if (config_.verbose) {
        std::cout << "RadiosityOpenCLSolver initialized: " << n_faces_ << " faces\n";
    }

    return true;
}

bool RadiosityOpenCLSolver::initialize(const OpenCLDeviceInfo& device_handle) {
    if (initialized_) return true;

    context_ = std::make_unique<OpenCLContext>();
    if (!context_->initialize(device_handle)) {
        if (config_.verbose) {
            std::cerr << "Failed to initialize OpenCL context on device "
                      << device_handle.name << "\n";
        }
        return false;
    }

    if (config_.verbose) {
        printDeviceInfo(context_->deviceInfo());
    }

    try {
        buildKernels();
        createBuffers();
    } catch (const std::exception& e) {
        if (config_.verbose) {
            std::cerr << "OpenCL initialization error on " << device_handle.name
                      << ": " << e.what() << "\n";
        }
        release();
        return false;
    }

    initialized_ = true;

    if (config_.verbose) {
        std::cout << "RadiosityOpenCLSolver initialized on "
                  << device_handle.name
                  << ": " << n_faces_ << " faces\n";
    }

    return true;
}

void RadiosityOpenCLSolver::buildKernels() {
    std::string source = getRadiosityKernelSource();

    // Build program manually (not using context_->buildProgram to avoid exception issues
    // with CL_CHECK macro across shared library boundaries on macOS)
    const char* src = source.c_str();
    size_t len = source.size();

    cl_int err;
    program_ = clCreateProgramWithSource(context_->context(), 1, &src, &len, &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error(std::string("clCreateProgramWithSource failed: ") +
                                 clErrorString(err));
    }

    std::string options = "-cl-fast-relaxed-math -cl-mad-enable";
    cl_device_id device = context_->device();
    err = clBuildProgram(program_, 1, &device, options.c_str(), nullptr, nullptr);
    if (err != CL_SUCCESS) {
        // Get build log
        size_t log_size;
        clGetProgramBuildInfo(program_, device, CL_PROGRAM_BUILD_LOG,
                             0, nullptr, &log_size);
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(program_, device, CL_PROGRAM_BUILD_LOG,
                             log_size, log.data(), nullptr);

        std::string msg = std::string("clBuildProgram failed: ") + clErrorString(err) +
                         "\nBuild log:\n" + log.data();
        clReleaseProgram(program_);
        program_ = nullptr;
        throw std::runtime_error(msg);
    }

    // Create kernels
    spmv_kernel_ = clCreateKernel(program_, "spmv_kernel", &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error(std::string("Failed to create spmv_kernel: ") + clErrorString(err));
    }

    spmv_vector_kernel_ = clCreateKernel(program_, "spmv_vector_kernel", &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error(std::string("Failed to create spmv_vector_kernel: ") + clErrorString(err));
    }

    max_reduction_kernel_ = clCreateKernel(program_, "max_reduction_kernel", &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error(std::string("Failed to create max_reduction_kernel: ") + clErrorString(err));
    }

    emission_kernel_ = clCreateKernel(program_, "emission_kernel", &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error(std::string("Failed to create emission_kernel: ") + clErrorString(err));
    }

    solar_source_kernel_ = clCreateKernel(program_, "solar_source_kernel", &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error(std::string("Failed to create solar_source_kernel: ") + clErrorString(err));
    }

    jacobi_kernel_ = clCreateKernel(program_, "jacobi_kernel", &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error(std::string("Failed to create jacobi_kernel: ") + clErrorString(err));
    }

    indirect_flux_kernel_ = clCreateKernel(program_, "indirect_flux_kernel", &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error(std::string("Failed to create indirect_flux_kernel: ") + clErrorString(err));
    }

    spmv_chunk_kernel_ = clCreateKernel(program_, "spmv_chunk_kernel", &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error(std::string("Failed to create spmv_chunk_kernel: ") + clErrorString(err));
    }

    spmv_vector_chunk_kernel_ = clCreateKernel(program_, "spmv_vector_chunk_kernel", &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error(std::string("Failed to create spmv_vector_chunk_kernel: ") + clErrorString(err));
    }

    spmv_vector2_kernel_ = clCreateKernel(program_, "spmv_vector2_kernel", &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error(std::string("Failed to create spmv_vector2_kernel: ") + clErrorString(err));
    }

    spmv_vector2_chunk_kernel_ = clCreateKernel(program_, "spmv_vector2_chunk_kernel", &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error(std::string("Failed to create spmv_vector2_chunk_kernel: ") + clErrorString(err));
    }

    jacobi2_kernel_ = clCreateKernel(program_, "jacobi2_kernel", &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error(std::string("Failed to create jacobi2_kernel: ") + clErrorString(err));
    }

    indirect_flux2_kernel_ = clCreateKernel(program_, "indirect_flux2_kernel", &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error(std::string("Failed to create indirect_flux2_kernel: ") + clErrorString(err));
    }

    // Query optimal work group size from CSR-Vector kernel
    // Use preferred work group size multiple (wavefront/warp size)
    size_t preferred_wg = 0;
    err = clGetKernelWorkGroupInfo(spmv_vector_kernel_, device,
                                   CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
                                   sizeof(preferred_wg), &preferred_wg, nullptr);
    if (err == CL_SUCCESS && preferred_wg > 0) {
        // Use wavefront/warp size for CSR-Vector kernel (64 on AMD, 32 on NVIDIA)
        // For other kernels, keep the configured work_group_size
        if (config_.verbose) {
            std::cout << "  Preferred work group size multiple: " << preferred_wg << "\n";
        }
    }
}

void RadiosityOpenCLSolver::createBuffers() {
    cl_context ctx = context_->context();
    cl_int err;

    auto check = [](cl_int e, const char* name) {
        if (e != CL_SUCCESS) {
            throw std::runtime_error(std::string("Failed to create buffer ") + name +
                                     ": " + clErrorString(e));
        }
    };

    // State buffers (will be used during solve)
    buf_radiosity_ = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                   n_faces_ * sizeof(float), nullptr, &err);
    check(err, "radiosity");

    buf_radiosity_new_ = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                       n_faces_ * sizeof(float), nullptr, &err);
    check(err, "radiosity_new");

    buf_F_B_ = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                             n_faces_ * sizeof(float), nullptr, &err);
    check(err, "F_B");

    buf_temperatures_ = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
                                      n_faces_ * sizeof(float), nullptr, &err);
    check(err, "temperatures");

    buf_solar_flux_ = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
                                    n_faces_ * sizeof(float), nullptr, &err);
    check(err, "solar_flux");

    buf_emission_ = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                  n_faces_ * sizeof(float), nullptr, &err);
    check(err, "emission");

    buf_solar_source_ = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                      n_faces_ * sizeof(float), nullptr, &err);
    check(err, "solar_source");

    buf_indirect_flux_ = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
                                       n_faces_ * sizeof(float), nullptr, &err);
    check(err, "indirect_flux");

    buf_emissivity_ = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
                                    n_faces_ * sizeof(float), nullptr, &err);
    check(err, "emissivity");

    buf_reflectivity_ = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
                                      n_faces_ * sizeof(float), nullptr, &err);
    check(err, "reflectivity");

    // View-factor row normalization 1/max(1, sum_j F_ij), applied inside the
    // Jacobi iteration and to the final flux. Seeded to 1.0 (identity) so a
    // caller that never calls uploadRowNorm() gets exactly the legacy kernel
    // arithmetic (`1.0f * x == x`).
    buf_row_norm_ = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
                                   n_faces_ * sizeof(float), nullptr, &err);
    check(err, "row_norm");
    {
        const std::vector<float> ones(n_faces_, 1.0f);
        err = clEnqueueWriteBuffer(context_->queue(), buf_row_norm_, CL_TRUE, 0,
                                   n_faces_ * sizeof(float), ones.data(),
                                   0, nullptr, nullptr);
        check(err, "row_norm(seed=1)");
    }

    // Partial max buffer for GPU reduction
    // Number of work groups = ceil(n_faces / work_group_size)
    n_reduction_groups_ = (n_faces_ + config_.work_group_size - 1) / config_.work_group_size;
    // Residual scratch: up to FOUR work-group-aligned segments of `seg` floats.
    //   gray / single band : [0] |dB|      [1] |B|
    //   fused two-band     : [0] |dB_ir|   [1] |B_ir|   [2] |dB_vis|  [3] |B_vis|
    // seg = n_reduction_groups_ * work_group_size >= n_faces, so every segment
    // starts on a work-group boundary and ONE reduction launch over the segments
    // in use yields all the maxima with a SINGLE device->host read. The
    // [n_faces, seg) tail of each segment is zero-filled here and never written
    // (the Jacobi kernels guard on i >= n_faces), so the per-group partial maxima
    // are identical to reducing each quantity's own n_faces-long array -- which
    // is what the pre-scratch code did for |dB| alone.
    band_segment_ = n_reduction_groups_ * config_.work_group_size;
    const size_t scratch_len = 4 * band_segment_;
    buf_resid_scratch_ = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                        scratch_len * sizeof(float), nullptr, &err);
    check(err, "resid_scratch");
    {
        // ⛔ clEnqueueWriteBuffer, NOT clEnqueueFillBuffer: on this machine's AMD
        // Radeon PRO W6800X Duo (Apple OpenCL 1.2) clEnqueueFillBuffer returns
        // CL_SUCCESS and leaves the buffer holding garbage (measured 2026-08-16).
        const std::vector<float> zeros(scratch_len, 0.0f);
        err = clEnqueueWriteBuffer(context_->queue(), buf_resid_scratch_, CL_TRUE, 0,
                                   scratch_len * sizeof(float), zeros.data(),
                                   0, nullptr, nullptr);
        check(err, "resid_scratch(zero)");
    }
    buf_partial_max4_ = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                       4 * n_reduction_groups_ * sizeof(float),
                                       nullptr, &err);
    check(err, "partial_max4");

    // CSR buffers will be created when uploadViewFactors is called
    // (since we don't know nnz yet)
}

// Both public overloads are thin: the value type is the ONLY thing that differs,
// and it is resolved with `if constexpr` inside the one implementation so the two
// entry points cannot drift apart. Narrowing, where it is needed at all, happens
// at exactly the same granularity it always did -- whole-matrix on the
// single-buffer path (which is capped at the ~2 GB per-buffer budget anyway) and
// PER CHUNK on the chunked path, so a huge float64 matrix does not suddenly need
// a whole extra float32 copy of itself resident.
void RadiosityOpenCLSolver::uploadViewFactors(
    const std::vector<double>& values,
    const std::vector<int>& col_indices,
    const std::vector<int64_t>& row_ptrs)
{
    uploadViewFactorsT(values, col_indices, row_ptrs);
}

void RadiosityOpenCLSolver::uploadViewFactors(
    const std::vector<float>& values,
    const std::vector<int>& col_indices,
    const std::vector<int64_t>& row_ptrs)
{
    uploadViewFactorsT(values, col_indices, row_ptrs);
}

template <typename ValueT>
void RadiosityOpenCLSolver::uploadViewFactorsT(
    const std::vector<ValueT>& values,
    const std::vector<int>& col_indices,
    const std::vector<int64_t>& row_ptrs)
{
    if (!initialized_) {
        throw std::runtime_error("RadiosityOpenCLSolver not initialized");
    }

    nnz_ = values.size();

    if (nnz_ == 0) {
        if (config_.verbose) {
            std::cout << "Warning: Empty view factor matrix uploaded\n";
        }
        view_factors_uploaded_ = true;
        return;
    }

    // choose the single-buffer path or the chunked path. The SpMV kernel
    // addresses the CSR values/col_indices buffers with a signed-int32 byte
    // offset; on a 32-bit-addressing device (W6800X/macOS reports
    // CL_DEVICE_ADDRESS_BITS=32) a single indexed buffer >= 2^31 bytes overflows
    // that offset and the driver aborts the PROCESS (uncatchable SIGABRT) once
    // the kernel runs. Below the per-chunk budget we use the single-buffer path
    // unchanged (byte-identical); above it we split the matrix on row boundaries
    // into sub-buffers each below the wall (buildChunkedCsr, which fails loud if
    // a single row cannot fit a chunk).
    {
        // Release any chunk buffers from a previous upload.
        releaseChunkBuffers();

        cl_device_id dev = context_->device();
        cl_uint address_bits = 64;
        cl_ulong max_alloc = 0;
        clGetDeviceInfo(dev, CL_DEVICE_ADDRESS_BITS, sizeof(address_bits),
                        &address_bits, nullptr);
        clGetDeviceInfo(dev, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_alloc),
                        &max_alloc, nullptr);
        const unsigned long long elem =
            static_cast<unsigned long long>(std::max(sizeof(float), sizeof(int)));
        const unsigned long long addr_wall =
            (address_bits <= 32) ? (1ull << 31) : ~0ull;
        unsigned long long wall = addr_wall;
        if (max_alloc != 0 && static_cast<unsigned long long>(max_alloc) < wall)
            wall = static_cast<unsigned long long>(max_alloc);

        // Per-chunk budget: stay under the measured-safe point (530M nnz =
        // 2.120 GB completed; 540M = 2.160 GB aborted; hard wall 2^31 =
        // 2.147 GB). Default 2.0 GB/buffer leaves ~120 MB margin.
        // PTM_RADIOSITY_CHUNK_NNZ lowers the per-chunk nnz cap (oracle/testing).
        unsigned long long budget_bytes = 2000000000ull;
        if (budget_bytes > wall) budget_bytes = wall;
        size_t max_chunk_nnz = static_cast<size_t>(budget_bytes / elem);
        if (const char* e = std::getenv("PTM_RADIOSITY_CHUNK_NNZ")) {
            unsigned long long v = std::strtoull(e, nullptr, 10);
            if (v > 0 && v < static_cast<unsigned long long>(max_chunk_nnz))
                max_chunk_nnz = static_cast<size_t>(v);
        }

        if (nnz_ > max_chunk_nnz) {
            use_chunked_csr_ = true;
            buildChunkedCsrT(values, col_indices, row_ptrs, max_chunk_nnz);
            view_factors_uploaded_ = true;
            return;
        }
    }

    cl_context ctx = context_->context();
    cl_command_queue queue = context_->queue();
    cl_int err;

    // The device's CSR value format is float32. When the caller already holds
    // float32 there is nothing to convert and nothing to stage -- the write
    // below reads straight out of `values`. A float64 caller (ViewFactorMatrix)
    // is narrowed into `staged` here; this path is only reached below the
    // per-buffer budget (~500 M nnz, ~2 GB of float32), so the transient is
    // bounded and unchanged from before the overload split.
    std::vector<float> staged;
    const float* values_f = nullptr;
    if constexpr (std::is_same_v<ValueT, float>) {
        values_f = values.data();
    } else {
        staged.resize(nnz_);
        for (size_t i = 0; i < nnz_; ++i) staged[i] = static_cast<float>(values[i]);
        values_f = staged.data();
    }

    // Helper for error checking without CL_CHECK macro (avoids SIGILL issue)
    auto checkErr = [](cl_int e, const char* op) {
        if (e != CL_SUCCESS) {
            throw std::runtime_error(std::string(op) + " failed: " + clErrorString(e));
        }
    };

    // Create CSR buffers
    buf_csr_values_ = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
                                    nnz_ * sizeof(float), nullptr, &err);
    checkErr(err, "clCreateBuffer(csr_values)");

    buf_csr_col_indices_ = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
                                         nnz_ * sizeof(int), nullptr, &err);
    checkErr(err, "clCreateBuffer(csr_col_indices)");

    buf_csr_row_ptrs_ = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
                                      row_ptrs.size() * sizeof(int), nullptr, &err);
    checkErr(err, "clCreateBuffer(csr_row_ptrs)");

    // The GPU SpMV kernel indexes row_ptrs as int32. This single-buffer path is
    // only taken when nnz_ <= the per-chunk budget (< 2^31), so every row
    // pointer fits int32; convert the int64 host array down. (Values above the
    // budget go through buildChunkedCsr instead.)
    std::vector<int> row_ptrs32(row_ptrs.size());
    for (size_t i = 0; i < row_ptrs.size(); ++i) {
        row_ptrs32[i] = static_cast<int>(row_ptrs[i]);
    }

    // Upload CSR data
    err = clEnqueueWriteBuffer(queue, buf_csr_values_, CL_FALSE, 0,
                               nnz_ * sizeof(float),
                               values_f, 0, nullptr, nullptr);
    checkErr(err, "clEnqueueWriteBuffer(csr_values)");

    err = clEnqueueWriteBuffer(queue, buf_csr_col_indices_, CL_FALSE, 0,
                               nnz_ * sizeof(int),
                               col_indices.data(), 0, nullptr, nullptr);
    checkErr(err, "clEnqueueWriteBuffer(csr_col_indices)");

    err = clEnqueueWriteBuffer(queue, buf_csr_row_ptrs_, CL_TRUE, 0,
                               row_ptrs32.size() * sizeof(int),
                               row_ptrs32.data(), 0, nullptr, nullptr);
    checkErr(err, "clEnqueueWriteBuffer(csr_row_ptrs)");

    view_factors_uploaded_ = true;

    if (config_.verbose) {
        size_t mem_mb = (nnz_ * (sizeof(float) + sizeof(int)) +
                        row_ptrs.size() * sizeof(int)) / (1024 * 1024);
        std::cout << "Uploaded view factor matrix to GPU: "
                  << nnz_ << " non-zeros (" << mem_mb << " MB)\n";
    }
}

template <typename ValueT>
void RadiosityOpenCLSolver::buildChunkedCsrT(
    const std::vector<ValueT>& values,
    const std::vector<int>& col_indices,
    const std::vector<int64_t>& row_ptrs,
    size_t max_chunk_nnz)
{
    cl_context ctx = context_->context();
    cl_command_queue queue = context_->queue();
    cl_int err;
    auto checkErr = [](cl_int e, const char* op) {
        if (e != CL_SUCCESS) {
            throw std::runtime_error(std::string(op) + " failed: " + clErrorString(e));
        }
    };

    const size_t n = n_faces_;
    // Partition rows greedily so each chunk's nnz <= max_chunk_nnz. Row-boundary
    // chunks keep the chunk-local row_ptrs (and thus the SpMV byte offset) below
    // the wall. col_indices stay GLOBAL (index the full radiosity vector).
    size_t row = 0;
    while (row < n) {
        const size_t chunk_row_begin = row;
        const size_t base = static_cast<size_t>(row_ptrs[chunk_row_begin]);

        size_t next = chunk_row_begin;
        while (next < n) {
            const size_t nnz_if_add =
                static_cast<size_t>(row_ptrs[next + 1]) - base;
            if (nnz_if_add > max_chunk_nnz) {
                if (next == chunk_row_begin) {
                    // A single row exceeds the budget: it cannot be split on a row
                    // boundary. Fail loud (2^31-wall backstop) rather than abort.
                    const size_t row_nnz =
                        static_cast<size_t>(row_ptrs[next + 1]) -
                        static_cast<size_t>(row_ptrs[next]);
                    throw std::runtime_error(
                        "RadiosityOpenCLSolver: row " + std::to_string(next) +
                        " has " + std::to_string(row_nnz) +
                        " non-zeros, exceeding the per-chunk budget of " +
                        std::to_string(max_chunk_nnz) +
                        " nnz (2^31-byte GPU addressing wall). This "
                        "matrix cannot run on this GPU without a wider SpMV "
                        "kernel; use CPU radiosity.");
                }
                break;
            }
            ++next;
        }

        const size_t chunk_row_end = next;
        const int chunk_rows = static_cast<int>(chunk_row_end - chunk_row_begin);
        const size_t chunk_nnz =
            static_cast<size_t>(row_ptrs[chunk_row_end]) - base;

        CsrChunk chunk;
        chunk.row_offset = static_cast<int>(chunk_row_begin);
        chunk.n_rows = chunk_rows;
        chunk.nnz = chunk_nnz;

        // Chunk-local row_ptrs (subtract base so the first entry is 0). base can
        // exceed int32 for a large matrix, so subtract in int64 first; each
        // resulting value is <= chunk_nnz <= max_chunk_nnz < 2^31, so int is safe.
        std::vector<int> local_rowptr(static_cast<size_t>(chunk_rows) + 1);
        for (int r = 0; r <= chunk_rows; ++r) {
            const int64_t local =
                row_ptrs[chunk_row_begin + static_cast<size_t>(r)] -
                static_cast<int64_t>(base);
            local_rowptr[static_cast<size_t>(r)] = static_cast<int>(local);
        }

        // float32 caller: write this chunk straight out of the caller's array,
        // exactly as the column-index slice below has always been written -- no
        // staging copy at all. float64 caller: narrow THIS CHUNK only (<=
        // max_chunk_nnz floats, ~2 GB), which is what this function has always
        // done; converting the whole matrix up front instead would add a second
        // full copy of a matrix that is in the chunked path precisely because it
        // is enormous.
        std::vector<float> chunk_staged;
        const float* chunk_vals = nullptr;
        if constexpr (std::is_same_v<ValueT, float>) {
            chunk_vals = values.data() + base;
        } else {
            chunk_staged.resize(chunk_nnz);
            for (size_t k = 0; k < chunk_nnz; ++k) {
                chunk_staged[k] = static_cast<float>(values[base + k]);
            }
            chunk_vals = chunk_staged.data();
        }

        chunk.values = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
                                      chunk_nnz * sizeof(float), nullptr, &err);
        checkErr(err, "clCreateBuffer(chunk values)");
        chunk.col_indices = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
                                           chunk_nnz * sizeof(int), nullptr, &err);
        checkErr(err, "clCreateBuffer(chunk col_indices)");
        chunk.row_ptrs = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
                                        (static_cast<size_t>(chunk_rows) + 1) * sizeof(int),
                                        nullptr, &err);
        checkErr(err, "clCreateBuffer(chunk row_ptrs)");

        err = clEnqueueWriteBuffer(queue, chunk.values, CL_FALSE, 0,
                                   chunk_nnz * sizeof(float),
                                   chunk_vals, 0, nullptr, nullptr);
        checkErr(err, "clEnqueueWriteBuffer(chunk values)");
        err = clEnqueueWriteBuffer(queue, chunk.col_indices, CL_FALSE, 0,
                                   chunk_nnz * sizeof(int),
                                   col_indices.data() + base, 0, nullptr, nullptr);
        checkErr(err, "clEnqueueWriteBuffer(chunk col_indices)");
        err = clEnqueueWriteBuffer(queue, chunk.row_ptrs, CL_TRUE, 0,
                                   (static_cast<size_t>(chunk_rows) + 1) * sizeof(int),
                                   local_rowptr.data(), 0, nullptr, nullptr);
        checkErr(err, "clEnqueueWriteBuffer(chunk row_ptrs)");

        csr_chunks_.push_back(chunk);
        row = chunk_row_end;
    }

    if (config_.verbose) {
        std::cout << "Uploaded CHUNKED view factor matrix to GPU: " << nnz_
                  << " non-zeros in " << csr_chunks_.size()
                  << " chunk(s) (budget " << max_chunk_nnz << " nnz/chunk)\n";
    }
}

void RadiosityOpenCLSolver::releaseChunkBuffers()
{
    for (auto& ch : csr_chunks_) {
        if (ch.values) clReleaseMemObject(ch.values);
        if (ch.col_indices) clReleaseMemObject(ch.col_indices);
        if (ch.row_ptrs) clReleaseMemObject(ch.row_ptrs);
    }
    csr_chunks_.clear();
    use_chunked_csr_ = false;
}

double RadiosityOpenCLSolver::benchmarkSpMV(int n_reps, int mode)
{
    cl_command_queue queue = context_->queue();
    cl_int err;
    auto ck = [](cl_int e, const char* op) {
        if (e != CL_SUCCESS)
            throw std::runtime_error(std::string(op) + ": " + clErrorString(e));
    };
    const int n_faces = static_cast<int>(n_faces_);
    auto tick = []() {
        return std::chrono::duration_cast<std::chrono::microseconds>(
                   std::chrono::high_resolution_clock::now().time_since_epoch())
                   .count() / 1000.0;
    };

    // This benchmark drives the SINGLE-BUFFER kernels directly, so it needs the
    // single-buffer CSR to exist. On a matrix above the per-chunk budget the CSR
    // lives in csr_chunks_ and buf_csr_values_/_col_indices_/_row_ptrs_ are all
    // still nullptr -- and OpenCL PERMITS a NULL cl_mem kernel argument, so the
    // kernels would run against null pointers and this function would return a
    // plausible millisecond number measured on nothing. That is exactly the trap
    // for anyone benchmarking SpMV at campaign scale (>500M nnz), which is the
    // only regime where it matters. Refuse loudly instead.
    if (use_chunked_csr_) {
        throw std::runtime_error(
            "benchmarkSpMV: this matrix (" + std::to_string(nnz_) +
            " nnz) was uploaded through the CHUNKED CSR path, so the "
            "single-buffer CSR buffers this benchmark drives do not exist. "
            "Benchmarking it here would time the kernels against null buffers "
            "and report a meaningless number. Use the solver's own per-solve "
            "timing (getLastSolveTimeMs / getLastTraversals) at this scale.");
    }
    if (!view_factors_uploaded_ || !buf_csr_values_ || !buf_csr_col_indices_ ||
        !buf_csr_row_ptrs_) {
        throw std::runtime_error(
            "benchmarkSpMV: no single-buffer CSR is resident on the device "
            "(uploadViewFactors has not run, or ran with an empty matrix).");
    }

    if (mode == 0) {
        const size_t local_size = config_.work_group_size;
        const size_t gs = ((n_faces_ + local_size - 1) / local_size) * local_size;
        // Every argument is checked: an unchecked clSetKernelArg turns a wrong
        // buffer or a stale kernel into a silently wrong timing, not an error.
        ck(clSetKernelArg(spmv_kernel_, 0, sizeof(cl_mem), &buf_csr_values_), "arg0");
        ck(clSetKernelArg(spmv_kernel_, 1, sizeof(cl_mem), &buf_csr_col_indices_), "arg1");
        ck(clSetKernelArg(spmv_kernel_, 2, sizeof(cl_mem), &buf_csr_row_ptrs_), "arg2");
        ck(clSetKernelArg(spmv_kernel_, 3, sizeof(cl_mem), &buf_radiosity_), "arg3");
        ck(clSetKernelArg(spmv_kernel_, 4, sizeof(cl_mem), &buf_F_B_), "arg4");
        ck(clSetKernelArg(spmv_kernel_, 5, sizeof(int), &n_faces), "arg5");
        clFinish(queue);
        const double t0 = tick();
        for (int r = 0; r < n_reps; ++r) {
            err = clEnqueueNDRangeKernel(queue, spmv_kernel_, 1, nullptr,
                                         &gs, &local_size, 0, nullptr, nullptr);
            ck(err, "scalar spmv");
        }
        clFinish(queue);
        return tick() - t0;
    } else {
        const size_t warp = 64;  // AMD wavefront: coalesced cooperative row reads
        const size_t gs = n_faces_ * warp;
        ck(clSetKernelArg(spmv_vector_kernel_, 0, sizeof(cl_mem), &buf_csr_values_), "arg0");
        ck(clSetKernelArg(spmv_vector_kernel_, 1, sizeof(cl_mem), &buf_csr_col_indices_), "arg1");
        ck(clSetKernelArg(spmv_vector_kernel_, 2, sizeof(cl_mem), &buf_csr_row_ptrs_), "arg2");
        ck(clSetKernelArg(spmv_vector_kernel_, 3, sizeof(cl_mem), &buf_radiosity_), "arg3");
        ck(clSetKernelArg(spmv_vector_kernel_, 4, sizeof(cl_mem), &buf_F_B_), "arg4");
        ck(clSetKernelArg(spmv_vector_kernel_, 5, sizeof(int), &n_faces), "arg5");
        ck(clSetKernelArg(spmv_vector_kernel_, 6, warp * sizeof(float), nullptr), "arg6");
        clFinish(queue);
        const double t0 = tick();
        for (int r = 0; r < n_reps; ++r) {
            err = clEnqueueNDRangeKernel(queue, spmv_vector_kernel_, 1, nullptr,
                                         &gs, &warp, 0, nullptr, nullptr);
            ck(err, "vector spmv");
        }
        clFinish(queue);
        return tick() - t0;
    }
}

void RadiosityOpenCLSolver::enqueueSpMV(cl_mem x, cl_mem y,
                                        size_t global_size, size_t local_size)
{
    cl_command_queue queue = context_->queue();
    cl_int err;
    auto checkErr = [](cl_int e, const char* op) {
        if (e != CL_SUCCESS) {
            throw std::runtime_error(std::string(op) + " failed: " + clErrorString(e));
        }
    };
    const int n_faces = static_cast<int>(n_faces_);

    // CSR-Vector SpMV: one wavefront (64) per row, cooperative coalesced reads +
    // local tree reduction -- ~5x the scalar kernel's bandwidth (measured). The
    // vector reduction sums each row in a different float order than the scalar
    // spmv_kernel, so production GPU results differ from the old scalar path at
    // ~float epsilon (both are valid float32 SpMV; the scalar kernel is retained
    // for benchmarkSpMV). single-buffer and chunked vector paths are bit-
    // identical to each other for a given row (same stride + reduction).
    (void)global_size; (void)local_size;
    const size_t warp = 64;
    const size_t local_bytes = warp * sizeof(float);

    // One logical traversal of the matrix, however many chunk launches it takes.
    ++last_traversals_;

    if (!use_chunked_csr_) {
        const size_t gs = n_faces_ * warp;
        err = clSetKernelArg(spmv_vector_kernel_, 0, sizeof(cl_mem), &buf_csr_values_);
        checkErr(err, "clSetKernelArg(spmv_vector,csr_values)");
        err = clSetKernelArg(spmv_vector_kernel_, 1, sizeof(cl_mem), &buf_csr_col_indices_);
        checkErr(err, "clSetKernelArg(spmv_vector,csr_col_indices)");
        err = clSetKernelArg(spmv_vector_kernel_, 2, sizeof(cl_mem), &buf_csr_row_ptrs_);
        checkErr(err, "clSetKernelArg(spmv_vector,csr_row_ptrs)");
        err = clSetKernelArg(spmv_vector_kernel_, 3, sizeof(cl_mem), &x);
        checkErr(err, "clSetKernelArg(spmv_vector,x)");
        err = clSetKernelArg(spmv_vector_kernel_, 4, sizeof(cl_mem), &y);
        checkErr(err, "clSetKernelArg(spmv_vector,y)");
        err = clSetKernelArg(spmv_vector_kernel_, 5, sizeof(int), &n_faces);
        checkErr(err, "clSetKernelArg(spmv_vector,n_faces)");
        err = clSetKernelArg(spmv_vector_kernel_, 6, local_bytes, nullptr);
        checkErr(err, "clSetKernelArg(spmv_vector,shared)");
        err = clEnqueueNDRangeKernel(queue, spmv_vector_kernel_, 1, nullptr,
                                     &gs, &warp, 0, nullptr, nullptr);
        checkErr(err, "clEnqueueNDRangeKernel(spmv_vector_kernel)");
        return;
    }

    // Chunked path + vector coalescing: one wavefront per chunk-local row.
    for (const CsrChunk& ch : csr_chunks_) {
        err = clSetKernelArg(spmv_vector_chunk_kernel_, 0, sizeof(cl_mem), &ch.values);
        checkErr(err, "clSetKernelArg(spmv_vchunk,values)");
        err = clSetKernelArg(spmv_vector_chunk_kernel_, 1, sizeof(cl_mem), &ch.col_indices);
        checkErr(err, "clSetKernelArg(spmv_vchunk,col_indices)");
        err = clSetKernelArg(spmv_vector_chunk_kernel_, 2, sizeof(cl_mem), &ch.row_ptrs);
        checkErr(err, "clSetKernelArg(spmv_vchunk,row_ptrs)");
        err = clSetKernelArg(spmv_vector_chunk_kernel_, 3, sizeof(cl_mem), &x);
        checkErr(err, "clSetKernelArg(spmv_vchunk,x)");
        err = clSetKernelArg(spmv_vector_chunk_kernel_, 4, sizeof(cl_mem), &y);
        checkErr(err, "clSetKernelArg(spmv_vchunk,y)");
        err = clSetKernelArg(spmv_vector_chunk_kernel_, 5, sizeof(int), &ch.row_offset);
        checkErr(err, "clSetKernelArg(spmv_vchunk,row_offset)");
        err = clSetKernelArg(spmv_vector_chunk_kernel_, 6, sizeof(int), &ch.n_rows);
        checkErr(err, "clSetKernelArg(spmv_vchunk,n_rows)");
        err = clSetKernelArg(spmv_vector_chunk_kernel_, 7, local_bytes, nullptr);
        checkErr(err, "clSetKernelArg(spmv_vchunk,shared)");
        const size_t gsz = static_cast<size_t>(ch.n_rows) * warp;
        err = clEnqueueNDRangeKernel(queue, spmv_vector_chunk_kernel_, 1, nullptr,
                                     &gsz, &warp, 0, nullptr, nullptr);
        checkErr(err, "clEnqueueNDRangeKernel(spmv_vector_chunk_kernel)");
    }
}

void RadiosityOpenCLSolver::enqueueSpMV2(cl_mem x1, cl_mem x2, cl_mem y1, cl_mem y2)
{
    cl_command_queue queue = context_->queue();
    cl_int err;
    auto checkErr = [](cl_int e, const char* op) {
        if (e != CL_SUCCESS) {
            throw std::runtime_error(std::string(op) + " failed: " + clErrorString(e));
        }
    };
    const int n_faces = static_cast<int>(n_faces_);
    const size_t warp = 64;
    const size_t local_bytes = 2 * warp * sizeof(float);  // two band accumulators

    // ONE traversal serving both bands -- that is the whole point of the fusion,
    // and it must be counted as one so the before/after traversal ratio is real.
    ++last_traversals_;

    if (!use_chunked_csr_) {
        const size_t gs = n_faces_ * warp;
        err = clSetKernelArg(spmv_vector2_kernel_, 0, sizeof(cl_mem), &buf_csr_values_);
        checkErr(err, "clSetKernelArg(spmv_vector2,csr_values)");
        err = clSetKernelArg(spmv_vector2_kernel_, 1, sizeof(cl_mem), &buf_csr_col_indices_);
        checkErr(err, "clSetKernelArg(spmv_vector2,csr_col_indices)");
        err = clSetKernelArg(spmv_vector2_kernel_, 2, sizeof(cl_mem), &buf_csr_row_ptrs_);
        checkErr(err, "clSetKernelArg(spmv_vector2,csr_row_ptrs)");
        err = clSetKernelArg(spmv_vector2_kernel_, 3, sizeof(cl_mem), &x1);
        checkErr(err, "clSetKernelArg(spmv_vector2,x1)");
        err = clSetKernelArg(spmv_vector2_kernel_, 4, sizeof(cl_mem), &x2);
        checkErr(err, "clSetKernelArg(spmv_vector2,x2)");
        err = clSetKernelArg(spmv_vector2_kernel_, 5, sizeof(cl_mem), &y1);
        checkErr(err, "clSetKernelArg(spmv_vector2,y1)");
        err = clSetKernelArg(spmv_vector2_kernel_, 6, sizeof(cl_mem), &y2);
        checkErr(err, "clSetKernelArg(spmv_vector2,y2)");
        err = clSetKernelArg(spmv_vector2_kernel_, 7, sizeof(int), &n_faces);
        checkErr(err, "clSetKernelArg(spmv_vector2,n_faces)");
        err = clSetKernelArg(spmv_vector2_kernel_, 8, local_bytes, nullptr);
        checkErr(err, "clSetKernelArg(spmv_vector2,shared)");
        err = clEnqueueNDRangeKernel(queue, spmv_vector2_kernel_, 1, nullptr,
                                     &gs, &warp, 0, nullptr, nullptr);
        checkErr(err, "clEnqueueNDRangeKernel(spmv_vector2_kernel)");
        return;
    }

    for (const CsrChunk& ch : csr_chunks_) {
        err = clSetKernelArg(spmv_vector2_chunk_kernel_, 0, sizeof(cl_mem), &ch.values);
        checkErr(err, "clSetKernelArg(spmv_v2chunk,values)");
        err = clSetKernelArg(spmv_vector2_chunk_kernel_, 1, sizeof(cl_mem), &ch.col_indices);
        checkErr(err, "clSetKernelArg(spmv_v2chunk,col_indices)");
        err = clSetKernelArg(spmv_vector2_chunk_kernel_, 2, sizeof(cl_mem), &ch.row_ptrs);
        checkErr(err, "clSetKernelArg(spmv_v2chunk,row_ptrs)");
        err = clSetKernelArg(spmv_vector2_chunk_kernel_, 3, sizeof(cl_mem), &x1);
        checkErr(err, "clSetKernelArg(spmv_v2chunk,x1)");
        err = clSetKernelArg(spmv_vector2_chunk_kernel_, 4, sizeof(cl_mem), &x2);
        checkErr(err, "clSetKernelArg(spmv_v2chunk,x2)");
        err = clSetKernelArg(spmv_vector2_chunk_kernel_, 5, sizeof(cl_mem), &y1);
        checkErr(err, "clSetKernelArg(spmv_v2chunk,y1)");
        err = clSetKernelArg(spmv_vector2_chunk_kernel_, 6, sizeof(cl_mem), &y2);
        checkErr(err, "clSetKernelArg(spmv_v2chunk,y2)");
        err = clSetKernelArg(spmv_vector2_chunk_kernel_, 7, sizeof(int), &ch.row_offset);
        checkErr(err, "clSetKernelArg(spmv_v2chunk,row_offset)");
        err = clSetKernelArg(spmv_vector2_chunk_kernel_, 8, sizeof(int), &ch.n_rows);
        checkErr(err, "clSetKernelArg(spmv_v2chunk,n_rows)");
        err = clSetKernelArg(spmv_vector2_chunk_kernel_, 9, local_bytes, nullptr);
        checkErr(err, "clSetKernelArg(spmv_v2chunk,shared)");
        const size_t gsz = static_cast<size_t>(ch.n_rows) * warp;
        err = clEnqueueNDRangeKernel(queue, spmv_vector2_chunk_kernel_, 1, nullptr,
                                     &gsz, &warp, 0, nullptr, nullptr);
        checkErr(err, "clEnqueueNDRangeKernel(spmv_vector2_chunk_kernel)");
    }
}

void RadiosityOpenCLSolver::uploadFaceProperties(
    const std::vector<float>& emissivity,
    const std::vector<float>& reflectivity)
{
    if (!initialized_) {
        throw std::runtime_error("RadiosityOpenCLSolver not initialized");
    }

    cl_command_queue queue = context_->queue();
    cl_int err;

    auto checkErr = [](cl_int e, const char* op) {
        if (e != CL_SUCCESS) {
            throw std::runtime_error(std::string(op) + " failed: " + clErrorString(e));
        }
    };

    err = clEnqueueWriteBuffer(queue, buf_emissivity_, CL_FALSE, 0,
                               emissivity.size() * sizeof(float),
                               emissivity.data(), 0, nullptr, nullptr);
    checkErr(err, "clEnqueueWriteBuffer(emissivity)");

    err = clEnqueueWriteBuffer(queue, buf_reflectivity_, CL_TRUE, 0,
                               reflectivity.size() * sizeof(float),
                               reflectivity.data(), 0, nullptr, nullptr);
    checkErr(err, "clEnqueueWriteBuffer(reflectivity)");

    properties_uploaded_ = true;
}

void RadiosityOpenCLSolver::uploadRowNorm(const std::vector<float>& row_norm)
{
    if (!initialized_) {
        throw std::runtime_error("RadiosityOpenCLSolver not initialized");
    }
    // A wrong-length row norm would silently renormalize the wrong faces (or
    // leave the tail at the 1.0 seed), which is exactly the class of silent
    // substitution this project refuses. Fail loud.
    if (row_norm.size() != n_faces_) {
        throw std::runtime_error("uploadRowNorm: row_norm has " +
                                 std::to_string(row_norm.size()) + " entries but the solver "
                                 "was built for " + std::to_string(n_faces_) + " faces");
    }

    cl_int err = clEnqueueWriteBuffer(context_->queue(), buf_row_norm_, CL_TRUE, 0,
                                      n_faces_ * sizeof(float), row_norm.data(),
                                      0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        throw std::runtime_error(std::string("clEnqueueWriteBuffer(row_norm) failed: ") +
                                 clErrorString(err));
    }
}

void RadiosityOpenCLSolver::createTwoBandBuffers()
{
    if (buf_radiosity_vis_) return;   // already allocated

    cl_context ctx = context_->context();
    cl_command_queue queue = context_->queue();
    cl_int err;
    auto check = [](cl_int e, const char* name) {
        if (e != CL_SUCCESS) {
            throw std::runtime_error(std::string("Failed to create buffer ") + name +
                                     ": " + clErrorString(e));
        }
    };

    auto make = [&](const char* name) {
        cl_mem b = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                  n_faces_ * sizeof(float), nullptr, &err);
        check(err, name);
        return b;
    };

    buf_radiosity_vis_ = make("radiosity_vis");
    buf_radiosity_vis_new_ = make("radiosity_vis_new");
    buf_F_B_vis_ = make("F_B_vis");
    buf_indirect_flux_vis_ = make("indirect_flux_vis");

    // All-zero cold-start source, written ONCE with clEnqueueWriteBuffer.
    // See the header note: clEnqueueFillBuffer is a silent no-op on this stack.
    buf_zeros_ = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
                                n_faces_ * sizeof(float), nullptr, &err);
    check(err, "zeros");
    {
        const std::vector<float> zeros(n_faces_, 0.0f);
        err = clEnqueueWriteBuffer(queue, buf_zeros_, CL_TRUE, 0,
                                   n_faces_ * sizeof(float), zeros.data(),
                                   0, nullptr, nullptr);
        check(err, "zeros(seed=0)");
    }

    buf_emissivity_vis_ = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
                                         n_faces_ * sizeof(float), nullptr, &err);
    check(err, "emissivity_vis");
    buf_reflectivity_vis_ = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
                                           n_faces_ * sizeof(float), nullptr, &err);
    check(err, "reflectivity_vis");
    // The residual scratch and its partial-max buffer are allocated in
    // createBuffers() for every solver -- the gray path uses segments 0-1, the
    // fused two-band path 0-3.
}

void RadiosityOpenCLSolver::uploadVisibleProperties(
    const std::vector<float>& emissivity_vis,
    const std::vector<float>& reflectivity_vis)
{
    if (!initialized_) {
        throw std::runtime_error("RadiosityOpenCLSolver not initialized");
    }
    // Same refusal as uploadRowNorm: a wrong-length property vector would leave
    // part of the mesh on whatever the buffer happened to hold, i.e. a different
    // albedo on some faces with no signal at all.
    if (emissivity_vis.size() != n_faces_ || reflectivity_vis.size() != n_faces_) {
        throw std::runtime_error(
            "uploadVisibleProperties: got " + std::to_string(emissivity_vis.size()) +
            " emissivity and " + std::to_string(reflectivity_vis.size()) +
            " reflectivity entries but the solver was built for " +
            std::to_string(n_faces_) + " faces");
    }

    createTwoBandBuffers();

    cl_command_queue queue = context_->queue();
    cl_int err = clEnqueueWriteBuffer(queue, buf_emissivity_vis_, CL_FALSE, 0,
                                      n_faces_ * sizeof(float), emissivity_vis.data(),
                                      0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        throw std::runtime_error(std::string("clEnqueueWriteBuffer(emissivity_vis) failed: ") +
                                 clErrorString(err));
    }
    err = clEnqueueWriteBuffer(queue, buf_reflectivity_vis_, CL_TRUE, 0,
                               n_faces_ * sizeof(float), reflectivity_vis.data(),
                               0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        throw std::runtime_error(std::string("clEnqueueWriteBuffer(reflectivity_vis) failed: ") +
                                 clErrorString(err));
    }

    visible_properties_uploaded_ = true;
}

void RadiosityOpenCLSolver::computeScratchMaxima(int n_segments, double* out)
{
    cl_command_queue queue = context_->queue();
    cl_int err;
    auto checkErr = [](cl_int e, const char* op) {
        if (e != CL_SUCCESS) {
            throw std::runtime_error(std::string(op) + " failed: " + clErrorString(e));
        }
    };

    const size_t local_size = config_.work_group_size;
    const size_t total = static_cast<size_t>(n_segments) * band_segment_;
    const int n_total = static_cast<int>(total);
    const size_t local_mem_size = local_size * sizeof(float);

    err = clSetKernelArg(max_reduction_kernel_, 0, sizeof(cl_mem), &buf_resid_scratch_);
    checkErr(err, "clSetKernelArg(max_reduction,input)");
    err = clSetKernelArg(max_reduction_kernel_, 1, sizeof(cl_mem), &buf_partial_max4_);
    checkErr(err, "clSetKernelArg(max_reduction,partial_max)");
    err = clSetKernelArg(max_reduction_kernel_, 2, sizeof(int), &n_total);
    checkErr(err, "clSetKernelArg(max_reduction,n)");
    err = clSetKernelArg(max_reduction_kernel_, 3, local_mem_size, nullptr);
    checkErr(err, "clSetKernelArg(max_reduction,shared)");

    err = clEnqueueNDRangeKernel(queue, max_reduction_kernel_, 1, nullptr,
                                 &total, &local_size, 0, nullptr, nullptr);
    checkErr(err, "clEnqueueNDRangeKernel(max_reduction)");

    // ONE blocking read for every maximum this check needs -- the same single
    // round trip per check the single-quantity reduction used to pay.
    const size_t n_partials = static_cast<size_t>(n_segments) * n_reduction_groups_;
    std::vector<float> partials(n_partials);
    err = clEnqueueReadBuffer(queue, buf_partial_max4_, CL_TRUE, 0,
                              n_partials * sizeof(float),
                              partials.data(), 0, nullptr, nullptr);
    checkErr(err, "clEnqueueReadBuffer(partial_max)");

    for (int seg = 0; seg < n_segments; ++seg) {
        float m = 0.0f;
        const size_t base = static_cast<size_t>(seg) * n_reduction_groups_;
        for (size_t i = 0; i < n_reduction_groups_; ++i) {
            m = std::max(m, partials[base + i]);
        }
        out[seg] = static_cast<double>(m);
    }
}

void RadiosityOpenCLSolver::solve(
    const std::vector<float>& temperatures,
    const std::vector<float>& solar_flux,
    std::vector<float>& indirect_flux_out)
{
    if (!initialized_) {
        throw std::runtime_error("RadiosityOpenCLSolver not initialized");
    }

    if (!view_factors_uploaded_) {
        throw std::runtime_error("View factors not uploaded to GPU");
    }

    if (!properties_uploaded_) {
        throw std::runtime_error("Face properties not uploaded to GPU");
    }

    auto start = std::chrono::high_resolution_clock::now();
    last_traversals_ = 0;

    cl_command_queue queue = context_->queue();
    int n_faces = static_cast<int>(n_faces_);
    const int seg = static_cast<int>(band_segment_);
    cl_int err;

    // Helper for error checking without CL_CHECK macro (avoids SIGILL issue)
    auto checkErr = [](cl_int e, const char* op) {
        if (e != CL_SUCCESS) {
            throw std::runtime_error(std::string(op) + " failed: " + clErrorString(e));
        }
    };

    // Calculate global/local sizes
    size_t global_size = ((n_faces_ + config_.work_group_size - 1) /
                          config_.work_group_size) * config_.work_group_size;
    size_t local_size = config_.work_group_size;

    // Upload temperatures and solar flux
    err = clEnqueueWriteBuffer(queue, buf_temperatures_, CL_FALSE, 0,
                               temperatures.size() * sizeof(float),
                               temperatures.data(), 0, nullptr, nullptr);
    checkErr(err, "clEnqueueWriteBuffer(temperatures)");

    err = clEnqueueWriteBuffer(queue, buf_solar_flux_, CL_TRUE, 0,
                               solar_flux.size() * sizeof(float),
                               solar_flux.data(), 0, nullptr, nullptr);
    checkErr(err, "clEnqueueWriteBuffer(solar_flux)");

    // Compute thermal emission: E = eps * sigma * T^4
    {
        int arg = 0;
        err = clSetKernelArg(emission_kernel_, arg++, sizeof(cl_mem), &buf_temperatures_);
        checkErr(err, "clSetKernelArg(emission_kernel,temperatures)");
        err = clSetKernelArg(emission_kernel_, arg++, sizeof(cl_mem), &buf_emissivity_);
        checkErr(err, "clSetKernelArg(emission_kernel,emissivity)");
        err = clSetKernelArg(emission_kernel_, arg++, sizeof(cl_mem), &buf_emission_);
        checkErr(err, "clSetKernelArg(emission_kernel,emission)");
        err = clSetKernelArg(emission_kernel_, arg++, sizeof(int), &n_faces);
        checkErr(err, "clSetKernelArg(emission_kernel,n_faces)");

        err = clEnqueueNDRangeKernel(queue, emission_kernel_, 1, nullptr,
                                     &global_size, &local_size, 0, nullptr, nullptr);
        checkErr(err, "clEnqueueNDRangeKernel(emission_kernel)");
    }

    // Compute solar source: rho * Q_solar
    {
        int arg = 0;
        err = clSetKernelArg(solar_source_kernel_, arg++, sizeof(cl_mem), &buf_solar_flux_);
        checkErr(err, "clSetKernelArg(solar_source_kernel,solar_flux)");
        err = clSetKernelArg(solar_source_kernel_, arg++, sizeof(cl_mem), &buf_reflectivity_);
        checkErr(err, "clSetKernelArg(solar_source_kernel,reflectivity)");
        err = clSetKernelArg(solar_source_kernel_, arg++, sizeof(cl_mem), &buf_solar_source_);
        checkErr(err, "clSetKernelArg(solar_source_kernel,solar_source)");
        err = clSetKernelArg(solar_source_kernel_, arg++, sizeof(int), &n_faces);
        checkErr(err, "clSetKernelArg(solar_source_kernel,n_faces)");

        err = clEnqueueNDRangeKernel(queue, solar_source_kernel_, 1, nullptr,
                                     &global_size, &local_size, 0, nullptr, nullptr);
        checkErr(err, "clEnqueueNDRangeKernel(solar_source_kernel)");
    }

    // Initialize radiosity: B = E + rho * Q_solar (or warm start from previous)
    if (!config_.warm_start || !has_previous_solution_) {
        // Cold start: Initialize radiosity buffer to emission
        // Note: Solar source will be added through first Jacobi iteration
        err = clEnqueueCopyBuffer(queue, buf_emission_, buf_radiosity_,
                                  0, 0, n_faces_ * sizeof(float),
                                  0, nullptr, nullptr);
        checkErr(err, "clEnqueueCopyBuffer(emission->radiosity)");
        has_previous_solution_ = true;
    }
    // else: warm start - buf_radiosity_ already contains previous solution

    // If no view factors (empty matrix), just return thermal emission
    if (nnz_ == 0) {
        // indirect_flux = 0 (no radiation exchange)
        indirect_flux_out.assign(n_faces_, 0.0f);
        last_iterations_ = 0;
        last_residual_ = 0.0;
        converged_ = true;

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        last_solve_time_ms_ = duration.count() / 1000.0f;
        return;
    }

    // SpMV kernel args are set inside enqueueSpMV (which also dispatches to the
    // chunked path when the matrix exceeds the single-buffer wall).

    // Jacobi iteration with periodic convergence checking. A check costs one
    // device->host round trip; not checking costs iterations the solve did not
    // need. resolveConvergenceCheckInterval() weighs the two for THIS matrix
    // (see its doc comment for the measurements behind the threshold).
    converged_ = false;
    last_residual_ = std::numeric_limits<double>::max();
    // The interval the shipped rule asks for on THIS matrix. PTM_RAD_GPU_CHECK_INTERVAL
    // (handled in the constructor) has already pinned config_.convergence_check_interval
    // if it was set, so it arrives here through the "pinned" branch of the resolver.
    const int resolved_interval =
        resolveConvergenceCheckInterval(config_.convergence_check_interval, nnz_);

    // Diagnostic per-iteration trace (PTM_RAD_ITER_TRACE=1). Forces a check every
    // iteration and reports max|dB| (the GPU criterion), max|B| and their ratio
    // (the CPU criterion) so the two stopping rules can be scored on one solve.
    // PTM_RAD_ITER_TRACE_FULL=1 additionally ignores the stopping rule and runs
    // all max_iterations, exposing the full decay curve past the shipped stop.
    static const bool iter_trace      = (std::getenv("PTM_RAD_ITER_TRACE") != nullptr);
    static const bool iter_trace_full = (std::getenv("PTM_RAD_ITER_TRACE_FULL") != nullptr);
    // Atomic: the multi-GPU pool submits solves on several device threads at once.
    static std::atomic<long> trace_solve_id{0};
    const long this_solve = iter_trace ? trace_solve_id.fetch_add(1) + 1 : 0;
    // The trace needs every sweep, so it overrides the resolved interval. That also
    // moves the STOPPING point (the solve now exits the iteration it converges on
    // rather than the next multiple of resolved_interval), which is why it is
    // announced below rather than applied quietly.
    const int check_interval = iter_trace ? 1 : resolved_interval;

    if (!logged_check_interval_) {
        logged_check_interval_ = true;
        std::cout << "  [Radiosity GPU] convergence check every " << resolved_interval
                  << " Jacobi iteration" << (resolved_interval == 1 ? "" : "s")
                  << " ("
                  << (config_.convergence_check_interval > 0
                          ? "pinned"
                          : (resolved_interval == 1
                                 ? "auto: a traversal of this matrix costs more than a check"
                                 : "auto: a check costs more than a traversal of this matrix"))
                  << "; nnz = " << nnz_ << ")\n";
        if (check_interval != resolved_interval) {
            std::cout << "  [Radiosity GPU] PTM_RAD_ITER_TRACE=1: overriding that to a check "
                         "every iteration, so each solve stops on the iteration it converges "
                         "rather than at a multiple of " << resolved_interval
                      << ". Iteration counts and timings under the trace are NOT shipped cost.\n";
        }
    }

    for (int iter = 0; iter < config_.max_iterations; ++iter) {
        // SpMV: F_B = F * radiosity (single-buffer or chunked)
        enqueueSpMV(buf_radiosity_, buf_F_B_, global_size, local_size);

        // Jacobi update: B_new = E + rho*Q_solar + rho*F*B
        {
            int arg = 0;
            err = clSetKernelArg(jacobi_kernel_, arg++, sizeof(cl_mem), &buf_emission_);
            checkErr(err, "clSetKernelArg(jacobi_kernel,emission)");
            err = clSetKernelArg(jacobi_kernel_, arg++, sizeof(cl_mem), &buf_solar_source_);
            checkErr(err, "clSetKernelArg(jacobi_kernel,solar_source)");
            err = clSetKernelArg(jacobi_kernel_, arg++, sizeof(cl_mem), &buf_reflectivity_);
            checkErr(err, "clSetKernelArg(jacobi_kernel,reflectivity)");
            err = clSetKernelArg(jacobi_kernel_, arg++, sizeof(cl_mem), &buf_F_B_);
            checkErr(err, "clSetKernelArg(jacobi_kernel,F_B)");
            err = clSetKernelArg(jacobi_kernel_, arg++, sizeof(cl_mem), &buf_row_norm_);
            checkErr(err, "clSetKernelArg(jacobi_kernel,row_norm)");
            err = clSetKernelArg(jacobi_kernel_, arg++, sizeof(cl_mem), &buf_radiosity_);
            checkErr(err, "clSetKernelArg(jacobi_kernel,radiosity)");
            err = clSetKernelArg(jacobi_kernel_, arg++, sizeof(cl_mem), &buf_radiosity_new_);
            checkErr(err, "clSetKernelArg(jacobi_kernel,radiosity_new)");
            err = clSetKernelArg(jacobi_kernel_, arg++, sizeof(cl_mem), &buf_resid_scratch_);
            checkErr(err, "clSetKernelArg(jacobi_kernel,resid_scratch)");
            err = clSetKernelArg(jacobi_kernel_, arg++, sizeof(int), &seg);
            checkErr(err, "clSetKernelArg(jacobi_kernel,seg)");
            err = clSetKernelArg(jacobi_kernel_, arg++, sizeof(int), &n_faces);
            checkErr(err, "clSetKernelArg(jacobi_kernel,n_faces)");

            err = clEnqueueNDRangeKernel(queue, jacobi_kernel_, 1, nullptr,
                                         &global_size, &local_size, 0, nullptr, nullptr);
            checkErr(err, "clEnqueueNDRangeKernel(jacobi_kernel)");
        }

        // Swap radiosity buffers
        std::swap(buf_radiosity_, buf_radiosity_new_);

        // Check convergence periodically (reduces PCIe sync overhead)
        // Also check on last iteration to get final residual
        if ((iter + 1) % check_interval == 0 || iter == config_.max_iterations - 1) {
            // ONE reduction launch, ONE read: max|dB| and max|B| together.
            double m[2] = {0.0, 0.0};
            computeScratchMaxima(2, m);
            const double max_change = m[0];
            const double max_val = m[1];
            const StopTest st = radiosityStopTest(max_change, max_val,
                                                  config_.convergence_tolerance);

            last_residual_ = st.residual;
            last_iterations_ = iter + 1;

            if (iter_trace) {
                const double rel = (max_val > 1e-10) ? max_change / max_val : max_change;
                std::fprintf(stderr,
                             "[rad-trace] gpu solve=%ld iter=%d dB_abs=%.6e B_max=%.6e rel=%.6e\n",
                             this_solve, iter + 1, max_change, max_val, rel);
            }

            if (st.converged && !iter_trace_full) {
                converged_ = true;
                break;
            }
        }
    }

    // Compute final SpMV to get F * B_final (single-buffer or chunked)
    enqueueSpMV(buf_radiosity_, buf_F_B_, global_size, local_size);

    // Compute indirect flux: Q_indirect = eps * (F * B)
    {
        int arg = 0;
        err = clSetKernelArg(indirect_flux_kernel_, arg++, sizeof(cl_mem), &buf_emissivity_);
        checkErr(err, "clSetKernelArg(indirect_flux_kernel,emissivity)");
        err = clSetKernelArg(indirect_flux_kernel_, arg++, sizeof(cl_mem), &buf_F_B_);
        checkErr(err, "clSetKernelArg(indirect_flux_kernel,F_B)");
        err = clSetKernelArg(indirect_flux_kernel_, arg++, sizeof(cl_mem), &buf_row_norm_);
        checkErr(err, "clSetKernelArg(indirect_flux_kernel,row_norm)");
        err = clSetKernelArg(indirect_flux_kernel_, arg++, sizeof(cl_mem), &buf_indirect_flux_);
        checkErr(err, "clSetKernelArg(indirect_flux_kernel,indirect_flux)");
        err = clSetKernelArg(indirect_flux_kernel_, arg++, sizeof(int), &n_faces);
        checkErr(err, "clSetKernelArg(indirect_flux_kernel,n_faces)");

        err = clEnqueueNDRangeKernel(queue, indirect_flux_kernel_, 1, nullptr,
                                     &global_size, &local_size, 0, nullptr, nullptr);
        checkErr(err, "clEnqueueNDRangeKernel(indirect_flux_kernel)");
    }

    // Download indirect flux
    indirect_flux_out.resize(n_faces_);
    err = clEnqueueReadBuffer(queue, buf_indirect_flux_, CL_TRUE, 0,
                              n_faces_ * sizeof(float),
                              indirect_flux_out.data(), 0, nullptr, nullptr);
    checkErr(err, "clEnqueueReadBuffer(indirect_flux)");

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    last_solve_time_ms_ = duration.count() / 1000.0f;
}

void RadiosityOpenCLSolver::solveTwoBandFused(
    const std::vector<float>& temperatures,
    const std::vector<float>& solar_flux,
    std::vector<float>& indirect_ir_out,
    std::vector<float>& indirect_vis_out)
{
    if (!initialized_) {
        throw std::runtime_error("RadiosityOpenCLSolver not initialized");
    }
    if (!view_factors_uploaded_) {
        throw std::runtime_error("View factors not uploaded to GPU");
    }
    if (!properties_uploaded_) {
        throw std::runtime_error("Face properties not uploaded to GPU");
    }
    if (!visible_properties_uploaded_) {
        throw std::runtime_error(
            "solveTwoBandFused: visible-band properties were never uploaded "
            "(uploadVisibleProperties). Refusing to run a two-band solve against "
            "the IR albedo on both bands.");
    }

    auto start = std::chrono::high_resolution_clock::now();
    last_traversals_ = 0;

    cl_command_queue queue = context_->queue();
    const int n_faces = static_cast<int>(n_faces_);
    const int seg = static_cast<int>(band_segment_);
    cl_int err;
    auto checkErr = [](cl_int e, const char* op) {
        if (e != CL_SUCCESS) {
            throw std::runtime_error(std::string(op) + " failed: " + clErrorString(e));
        }
    };

    const size_t global_size = ((n_faces_ + config_.work_group_size - 1) /
                                config_.work_group_size) * config_.work_group_size;
    const size_t local_size = config_.work_group_size;

    // ---- Sources -------------------------------------------------------------
    // IR band source: E = eps sigma T^4 (emission_kernel, IR emissivity).
    // Visible band source: A Q (solar_source_kernel driven by the VISIBLE
    // reflectivity A, not the IR one). Both are exactly the buffers the unfused
    // route builds for its two separate solves.
    err = clEnqueueWriteBuffer(queue, buf_temperatures_, CL_FALSE, 0,
                               temperatures.size() * sizeof(float),
                               temperatures.data(), 0, nullptr, nullptr);
    checkErr(err, "clEnqueueWriteBuffer(temperatures)");
    err = clEnqueueWriteBuffer(queue, buf_solar_flux_, CL_TRUE, 0,
                               solar_flux.size() * sizeof(float),
                               solar_flux.data(), 0, nullptr, nullptr);
    checkErr(err, "clEnqueueWriteBuffer(solar_flux)");

    {
        int arg = 0;
        err = clSetKernelArg(emission_kernel_, arg++, sizeof(cl_mem), &buf_temperatures_);
        checkErr(err, "clSetKernelArg(emission_kernel,temperatures)");
        err = clSetKernelArg(emission_kernel_, arg++, sizeof(cl_mem), &buf_emissivity_);
        checkErr(err, "clSetKernelArg(emission_kernel,emissivity)");
        err = clSetKernelArg(emission_kernel_, arg++, sizeof(cl_mem), &buf_emission_);
        checkErr(err, "clSetKernelArg(emission_kernel,emission)");
        err = clSetKernelArg(emission_kernel_, arg++, sizeof(int), &n_faces);
        checkErr(err, "clSetKernelArg(emission_kernel,n_faces)");
        err = clEnqueueNDRangeKernel(queue, emission_kernel_, 1, nullptr,
                                     &global_size, &local_size, 0, nullptr, nullptr);
        checkErr(err, "clEnqueueNDRangeKernel(emission_kernel)");
    }
    {
        int arg = 0;
        err = clSetKernelArg(solar_source_kernel_, arg++, sizeof(cl_mem), &buf_solar_flux_);
        checkErr(err, "clSetKernelArg(solar_source_kernel,solar_flux)");
        err = clSetKernelArg(solar_source_kernel_, arg++, sizeof(cl_mem), &buf_reflectivity_vis_);
        checkErr(err, "clSetKernelArg(solar_source_kernel,reflectivity_vis)");
        err = clSetKernelArg(solar_source_kernel_, arg++, sizeof(cl_mem), &buf_solar_source_);
        checkErr(err, "clSetKernelArg(solar_source_kernel,solar_source)");
        err = clSetKernelArg(solar_source_kernel_, arg++, sizeof(int), &n_faces);
        checkErr(err, "clSetKernelArg(solar_source_kernel,n_faces)");
        err = clEnqueueNDRangeKernel(queue, solar_source_kernel_, 1, nullptr,
                                     &global_size, &local_size, 0, nullptr, nullptr);
        checkErr(err, "clEnqueueNDRangeKernel(solar_source_kernel)");
    }

    // ---- Cold start ----------------------------------------------------------
    // Matches the unfused route exactly: the IR band starts at its emission
    // buffer, the visible band at ITS emission buffer -- which the unfused route
    // fills by running emission_kernel on temps = 0, i.e. all zeros. Two-band
    // callers force warm_start off (a shared radiosity buffer made warm starting
    // meaningless across bands), so this always runs; with per-band buffers a
    // warm start would now be well defined, but changing that would change
    // results and is deliberately not done here.
    err = clEnqueueCopyBuffer(queue, buf_emission_, buf_radiosity_,
                              0, 0, n_faces_ * sizeof(float), 0, nullptr, nullptr);
    checkErr(err, "clEnqueueCopyBuffer(emission->radiosity_ir)");
    err = clEnqueueCopyBuffer(queue, buf_zeros_, buf_radiosity_vis_,
                              0, 0, n_faces_ * sizeof(float), 0, nullptr, nullptr);
    checkErr(err, "clEnqueueCopyBuffer(zeros->radiosity_vis)");
    has_previous_solution_ = true;

    if (nnz_ == 0) {
        indirect_ir_out.assign(n_faces_, 0.0f);
        indirect_vis_out.assign(n_faces_, 0.0f);
        last_iterations_ = last_iterations_ir_ = last_iterations_vis_ = 0;
        last_residual_ = 0.0;
        converged_ = true;
        last_solve_time_ms_ = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - start).count() / 1000.0f;
        return;
    }

    // ---- Fused Jacobi --------------------------------------------------------
    converged_ = false;
    last_residual_ = std::numeric_limits<double>::max();
    const int resolved_interval =
        resolveConvergenceCheckInterval(config_.convergence_check_interval, nnz_);

    static const bool iter_trace      = (std::getenv("PTM_RAD_ITER_TRACE") != nullptr);
    static const bool iter_trace_full = (std::getenv("PTM_RAD_ITER_TRACE_FULL") != nullptr);
    static std::atomic<long> trace_solve_id_2b{0};
    const long this_solve = iter_trace ? trace_solve_id_2b.fetch_add(1) + 1 : 0;
    const int check_interval = iter_trace ? 1 : resolved_interval;

    if (!logged_check_interval_) {
        logged_check_interval_ = true;
        std::cout << "  [Radiosity GPU] convergence check every " << resolved_interval
                  << " Jacobi iteration" << (resolved_interval == 1 ? "" : "s")
                  << " ("
                  << (config_.convergence_check_interval > 0
                          ? "pinned"
                          : (resolved_interval == 1
                                 ? "auto: a traversal of this matrix costs more than a check"
                                 : "auto: a check costs more than a traversal of this matrix"))
                  << "; nnz = " << nnz_ << "); FUSED two-band: one width-2 traversal "
                     "serves both bands while both iterate.\n";
        if (check_interval != resolved_interval) {
            std::cout << "  [Radiosity GPU] PTM_RAD_ITER_TRACE=1: overriding that to a check "
                         "every iteration, so each solve stops on the iteration it converges "
                         "rather than at a multiple of " << resolved_interval
                      << ". Iteration counts and timings under the trace are NOT shipped cost.\n";
        }
    }

    bool conv_ir = false, conv_vis = false;
    int it_ir = 0, it_vis = 0;
    double res_ir = std::numeric_limits<double>::max();
    double res_vis = std::numeric_limits<double>::max();

    for (int iter = 0; iter < config_.max_iterations && !(conv_ir && conv_vis); ++iter) {
        const int mask = (conv_ir ? 0 : 1) | (conv_vis ? 0 : 2);

        if (mask == 3) {
            enqueueSpMV2(buf_radiosity_, buf_radiosity_vis_, buf_F_B_, buf_F_B_vis_);
        } else if (mask == 1) {
            enqueueSpMV(buf_radiosity_, buf_F_B_, global_size, local_size);
        } else {
            enqueueSpMV(buf_radiosity_vis_, buf_F_B_vis_, global_size, local_size);
        }

        {
            int arg = 0;
            err = clSetKernelArg(jacobi2_kernel_, arg++, sizeof(cl_mem), &buf_emission_);
            checkErr(err, "clSetKernelArg(jacobi2,ir_source)");
            err = clSetKernelArg(jacobi2_kernel_, arg++, sizeof(cl_mem), &buf_reflectivity_);
            checkErr(err, "clSetKernelArg(jacobi2,ir_refl)");
            err = clSetKernelArg(jacobi2_kernel_, arg++, sizeof(cl_mem), &buf_F_B_);
            checkErr(err, "clSetKernelArg(jacobi2,ir_F_B)");
            err = clSetKernelArg(jacobi2_kernel_, arg++, sizeof(cl_mem), &buf_solar_source_);
            checkErr(err, "clSetKernelArg(jacobi2,vis_source)");
            err = clSetKernelArg(jacobi2_kernel_, arg++, sizeof(cl_mem), &buf_reflectivity_vis_);
            checkErr(err, "clSetKernelArg(jacobi2,vis_refl)");
            err = clSetKernelArg(jacobi2_kernel_, arg++, sizeof(cl_mem), &buf_F_B_vis_);
            checkErr(err, "clSetKernelArg(jacobi2,vis_F_B)");
            err = clSetKernelArg(jacobi2_kernel_, arg++, sizeof(cl_mem), &buf_row_norm_);
            checkErr(err, "clSetKernelArg(jacobi2,row_norm)");
            err = clSetKernelArg(jacobi2_kernel_, arg++, sizeof(cl_mem), &buf_radiosity_);
            checkErr(err, "clSetKernelArg(jacobi2,B_ir_old)");
            err = clSetKernelArg(jacobi2_kernel_, arg++, sizeof(cl_mem), &buf_radiosity_new_);
            checkErr(err, "clSetKernelArg(jacobi2,B_ir_new)");
            err = clSetKernelArg(jacobi2_kernel_, arg++, sizeof(cl_mem), &buf_radiosity_vis_);
            checkErr(err, "clSetKernelArg(jacobi2,B_vis_old)");
            err = clSetKernelArg(jacobi2_kernel_, arg++, sizeof(cl_mem), &buf_radiosity_vis_new_);
            checkErr(err, "clSetKernelArg(jacobi2,B_vis_new)");
            err = clSetKernelArg(jacobi2_kernel_, arg++, sizeof(cl_mem), &buf_resid_scratch_);
            checkErr(err, "clSetKernelArg(jacobi2,resid_scratch)");
            err = clSetKernelArg(jacobi2_kernel_, arg++, sizeof(int), &seg);
            checkErr(err, "clSetKernelArg(jacobi2,seg)");
            err = clSetKernelArg(jacobi2_kernel_, arg++, sizeof(int), &mask);
            checkErr(err, "clSetKernelArg(jacobi2,active_mask)");
            err = clSetKernelArg(jacobi2_kernel_, arg++, sizeof(int), &n_faces);
            checkErr(err, "clSetKernelArg(jacobi2,n_faces)");
            err = clEnqueueNDRangeKernel(queue, jacobi2_kernel_, 1, nullptr,
                                         &global_size, &local_size, 0, nullptr, nullptr);
            checkErr(err, "clEnqueueNDRangeKernel(jacobi2_kernel)");
        }

        if (!conv_ir)  { std::swap(buf_radiosity_, buf_radiosity_new_);         it_ir  = iter + 1; }
        if (!conv_vis) { std::swap(buf_radiosity_vis_, buf_radiosity_vis_new_); it_vis = iter + 1; }

        if ((iter + 1) % check_interval == 0 || iter == config_.max_iterations - 1) {
            double dB_ir = 0.0, B_ir = 0.0, dB_vis = 0.0, B_vis = 0.0;
            double m[4] = {0.0, 0.0, 0.0, 0.0};
            computeScratchMaxima(4, m);
            dB_ir = m[0]; B_ir = m[1]; dB_vis = m[2]; B_vis = m[3];

            if (iter_trace) {
                const double rel_ir  = (B_ir  > 1e-10) ? dB_ir  / B_ir  : dB_ir;
                const double rel_vis = (B_vis > 1e-10) ? dB_vis / B_vis : dB_vis;
                std::fprintf(stderr,
                             "[rad-trace] gpu-2band-ir solve=%ld iter=%d dB_abs=%.6e "
                             "B_max=%.6e rel=%.6e%s\n",
                             this_solve, iter + 1, dB_ir, B_ir, rel_ir,
                             conv_ir ? " (frozen)" : "");
                std::fprintf(stderr,
                             "[rad-trace] gpu-2band-vis solve=%ld iter=%d dB_abs=%.6e "
                             "B_max=%.6e rel=%.6e%s\n",
                             this_solve, iter + 1, dB_vis, B_vis, rel_vis,
                             conv_vis ? " (frozen)" : "");
            }

            // Stopping criterion, per band, evaluated only for bands still
            // iterating -- the GPU counterpart of the CPU fused solver's
            // `if (!conv_ir)` / `if (!conv_vis)` blocks. Each band is scored on
            // ITS OWN max|dB| and max|B|; scoring both against a shared maximum
            // would tie the dim band's stopping point to the bright one's scale.
            // A frozen band keeps the residual it stopped on; it is never
            // re-scored against a scratch segment nothing writes any more.
            const double tol = config_.convergence_tolerance;
            if (!conv_ir) {
                const StopTest st = radiosityStopTest(dB_ir, B_ir, tol);
                res_ir = st.residual;
                if (st.converged && !iter_trace_full) conv_ir = true;
            }
            if (!conv_vis) {
                const StopTest st = radiosityStopTest(dB_vis, B_vis, tol);
                res_vis = st.residual;
                if (st.converged && !iter_trace_full) conv_vis = true;
            }
        }
    }

    last_iterations_ir_  = it_ir;
    last_iterations_vis_ = it_vis;
    // Aggregate getter keeps the unfused route's meaning: the visible band is the
    // one the legacy code path reported last, and test_two_band_rownorm B1e reads
    // exactly that.
    last_iterations_ = it_vis;
    // A two-band solve has converged only when BOTH bands have, and its residual
    // is the worse of the two -- the same rule solveTwoBandJacobiFused() adopted
    // under the coupling-mode control, rather than silently reporting the second band alone.
    converged_ = conv_ir && conv_vis;
    last_residual_ = std::max(res_ir, res_vis);

    // ---- Final traversal + absorbed flux, both bands in one pass -------------
    enqueueSpMV2(buf_radiosity_, buf_radiosity_vis_, buf_F_B_, buf_F_B_vis_);
    {
        int arg = 0;
        err = clSetKernelArg(indirect_flux2_kernel_, arg++, sizeof(cl_mem), &buf_emissivity_);
        checkErr(err, "clSetKernelArg(indirect_flux2,ir_absorb)");
        err = clSetKernelArg(indirect_flux2_kernel_, arg++, sizeof(cl_mem), &buf_F_B_);
        checkErr(err, "clSetKernelArg(indirect_flux2,ir_F_B)");
        err = clSetKernelArg(indirect_flux2_kernel_, arg++, sizeof(cl_mem), &buf_emissivity_vis_);
        checkErr(err, "clSetKernelArg(indirect_flux2,vis_absorb)");
        err = clSetKernelArg(indirect_flux2_kernel_, arg++, sizeof(cl_mem), &buf_F_B_vis_);
        checkErr(err, "clSetKernelArg(indirect_flux2,vis_F_B)");
        err = clSetKernelArg(indirect_flux2_kernel_, arg++, sizeof(cl_mem), &buf_row_norm_);
        checkErr(err, "clSetKernelArg(indirect_flux2,row_norm)");
        err = clSetKernelArg(indirect_flux2_kernel_, arg++, sizeof(cl_mem), &buf_indirect_flux_);
        checkErr(err, "clSetKernelArg(indirect_flux2,ir_out)");
        err = clSetKernelArg(indirect_flux2_kernel_, arg++, sizeof(cl_mem), &buf_indirect_flux_vis_);
        checkErr(err, "clSetKernelArg(indirect_flux2,vis_out)");
        err = clSetKernelArg(indirect_flux2_kernel_, arg++, sizeof(int), &n_faces);
        checkErr(err, "clSetKernelArg(indirect_flux2,n_faces)");
        err = clEnqueueNDRangeKernel(queue, indirect_flux2_kernel_, 1, nullptr,
                                     &global_size, &local_size, 0, nullptr, nullptr);
        checkErr(err, "clEnqueueNDRangeKernel(indirect_flux2_kernel)");
    }

    indirect_ir_out.resize(n_faces_);
    indirect_vis_out.resize(n_faces_);
    err = clEnqueueReadBuffer(queue, buf_indirect_flux_, CL_FALSE, 0,
                              n_faces_ * sizeof(float),
                              indirect_ir_out.data(), 0, nullptr, nullptr);
    checkErr(err, "clEnqueueReadBuffer(indirect_flux_ir)");
    err = clEnqueueReadBuffer(queue, buf_indirect_flux_vis_, CL_TRUE, 0,
                              n_faces_ * sizeof(float),
                              indirect_vis_out.data(), 0, nullptr, nullptr);
    checkErr(err, "clEnqueueReadBuffer(indirect_flux_vis)");

    last_solve_time_ms_ = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - start).count() / 1000.0f;
}

void RadiosityOpenCLSolver::synchronize() {
    if (context_) {
        context_->finish();
    }
}

const OpenCLDeviceInfo& RadiosityOpenCLSolver::getDeviceInfo() const {
    static OpenCLDeviceInfo empty;
    return context_ ? context_->deviceInfo() : empty;
}

void RadiosityOpenCLSolver::release() {
    // Ensure all GPU operations are finished before releasing resources
    if (context_) {
        context_->finish();
    }

    // Release kernels
    if (spmv_kernel_) {
        clReleaseKernel(spmv_kernel_);
        spmv_kernel_ = nullptr;
    }
    if (spmv_vector_kernel_) {
        clReleaseKernel(spmv_vector_kernel_);
        spmv_vector_kernel_ = nullptr;
    }
    if (max_reduction_kernel_) {
        clReleaseKernel(max_reduction_kernel_);
        max_reduction_kernel_ = nullptr;
    }
    if (emission_kernel_) {
        clReleaseKernel(emission_kernel_);
        emission_kernel_ = nullptr;
    }
    if (solar_source_kernel_) {
        clReleaseKernel(solar_source_kernel_);
        solar_source_kernel_ = nullptr;
    }
    if (jacobi_kernel_) {
        clReleaseKernel(jacobi_kernel_);
        jacobi_kernel_ = nullptr;
    }
    if (indirect_flux_kernel_) {
        clReleaseKernel(indirect_flux_kernel_);
        indirect_flux_kernel_ = nullptr;
    }
    if (spmv_chunk_kernel_) {
        clReleaseKernel(spmv_chunk_kernel_);
        spmv_chunk_kernel_ = nullptr;
    }
    if (spmv_vector_chunk_kernel_) {
        clReleaseKernel(spmv_vector_chunk_kernel_);
        spmv_vector_chunk_kernel_ = nullptr;
    }
    if (spmv_vector2_kernel_) {
        clReleaseKernel(spmv_vector2_kernel_);
        spmv_vector2_kernel_ = nullptr;
    }
    if (spmv_vector2_chunk_kernel_) {
        clReleaseKernel(spmv_vector2_chunk_kernel_);
        spmv_vector2_chunk_kernel_ = nullptr;
    }
    if (jacobi2_kernel_) {
        clReleaseKernel(jacobi2_kernel_);
        jacobi2_kernel_ = nullptr;
    }
    if (indirect_flux2_kernel_) {
        clReleaseKernel(indirect_flux2_kernel_);
        indirect_flux2_kernel_ = nullptr;
    }

    // Release program
    if (program_) {
        clReleaseProgram(program_);
        program_ = nullptr;
    }

    // Release CSR buffers
    if (buf_csr_values_) { clReleaseMemObject(buf_csr_values_); buf_csr_values_ = nullptr; }
    if (buf_csr_col_indices_) { clReleaseMemObject(buf_csr_col_indices_); buf_csr_col_indices_ = nullptr; }
    if (buf_csr_row_ptrs_) { clReleaseMemObject(buf_csr_row_ptrs_); buf_csr_row_ptrs_ = nullptr; }

    // Release chunked CSR sub-buffers
    releaseChunkBuffers();

    // Release state buffers
    if (buf_radiosity_) { clReleaseMemObject(buf_radiosity_); buf_radiosity_ = nullptr; }
    if (buf_radiosity_new_) { clReleaseMemObject(buf_radiosity_new_); buf_radiosity_new_ = nullptr; }
    if (buf_F_B_) { clReleaseMemObject(buf_F_B_); buf_F_B_ = nullptr; }

    // Release input/output buffers
    if (buf_temperatures_) { clReleaseMemObject(buf_temperatures_); buf_temperatures_ = nullptr; }
    if (buf_solar_flux_) { clReleaseMemObject(buf_solar_flux_); buf_solar_flux_ = nullptr; }
    if (buf_emission_) { clReleaseMemObject(buf_emission_); buf_emission_ = nullptr; }
    if (buf_solar_source_) { clReleaseMemObject(buf_solar_source_); buf_solar_source_ = nullptr; }
    if (buf_indirect_flux_) { clReleaseMemObject(buf_indirect_flux_); buf_indirect_flux_ = nullptr; }

    // Release property buffers
    if (buf_emissivity_) { clReleaseMemObject(buf_emissivity_); buf_emissivity_ = nullptr; }
    if (buf_reflectivity_) { clReleaseMemObject(buf_reflectivity_); buf_reflectivity_ = nullptr; }
    if (buf_row_norm_) { clReleaseMemObject(buf_row_norm_); buf_row_norm_ = nullptr; }

    // Release the visible band's state (only allocated on a two-band solver)
    if (buf_radiosity_vis_) { clReleaseMemObject(buf_radiosity_vis_); buf_radiosity_vis_ = nullptr; }
    if (buf_radiosity_vis_new_) { clReleaseMemObject(buf_radiosity_vis_new_); buf_radiosity_vis_new_ = nullptr; }
    if (buf_F_B_vis_) { clReleaseMemObject(buf_F_B_vis_); buf_F_B_vis_ = nullptr; }
    if (buf_emissivity_vis_) { clReleaseMemObject(buf_emissivity_vis_); buf_emissivity_vis_ = nullptr; }
    if (buf_reflectivity_vis_) { clReleaseMemObject(buf_reflectivity_vis_); buf_reflectivity_vis_ = nullptr; }
    if (buf_indirect_flux_vis_) { clReleaseMemObject(buf_indirect_flux_vis_); buf_indirect_flux_vis_ = nullptr; }
    if (buf_zeros_) { clReleaseMemObject(buf_zeros_); buf_zeros_ = nullptr; }
    if (buf_partial_max4_) { clReleaseMemObject(buf_partial_max4_); buf_partial_max4_ = nullptr; }
    band_segment_ = 0;
    visible_properties_uploaded_ = false;

    // Release convergence buffer
    if (buf_resid_scratch_) { clReleaseMemObject(buf_resid_scratch_); buf_resid_scratch_ = nullptr; }

    // Release context
    context_.reset();

    initialized_ = false;
    view_factors_uploaded_ = false;
    properties_uploaded_ = false;
    has_previous_solution_ = false;
}

}  // namespace opencl
}  // namespace gpu

#endif  // PTM_GPU_OPENCL
