/**
 * @file radiosity_opencl.cpp
 * @brief OpenCL radiosity solver implementation
 */

#ifdef PTM_GPU_OPENCL

#include "radiosity_opencl.h"
#include <fstream>
#include <sstream>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <atomic>
#include <algorithm>
#include <cmath>
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
__kernel void jacobi_kernel(
    __global const float* emission,
    __global const float* solar_source,
    __global const float* reflectivity,
    __global const float* F_B,
    __global const float* row_norm,
    __global const float* radiosity_old,
    __global float* radiosity_new,
    __global float* change,
    int n_faces)
{
    int i = get_global_id(0);
    if (i >= n_faces) return;

    float B_new = emission[i] + solar_source[i] + reflectivity[i] * (row_norm[i] * F_B[i]);
    radiosity_new[i] = B_new;
    float diff = B_new - radiosity_old[i];
    change[i] = (diff >= 0.0f) ? diff : -diff;
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

// RadiosityOpenCLSolver implementation

RadiosityOpenCLSolver::RadiosityOpenCLSolver(size_t n_faces,
                                             const RadiosityOpenCLConfig& config)
    : n_faces_(n_faces)
    , nnz_(0)
    , config_(config)
{
    change_host_.resize(n_faces);

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

    buf_change_ = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                n_faces_ * sizeof(float), nullptr, &err);
    check(err, "change");

    // Partial max buffer for GPU reduction
    // Number of work groups = ceil(n_faces / work_group_size)
    n_reduction_groups_ = (n_faces_ + config_.work_group_size - 1) / config_.work_group_size;
    buf_partial_max_ = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                      n_reduction_groups_ * sizeof(float), nullptr, &err);
    check(err, "partial_max");

    // CSR buffers will be created when uploadViewFactors is called
    // (since we don't know nnz yet)
}

void RadiosityOpenCLSolver::uploadViewFactors(
    const std::vector<double>& values,
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

    // PTM-25: choose the single-buffer path or the chunked path. The SpMV kernel
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
            buildChunkedCsr(values, col_indices, row_ptrs, max_chunk_nnz);
            view_factors_uploaded_ = true;
            return;
        }
    }

    cl_context ctx = context_->context();
    cl_command_queue queue = context_->queue();
    cl_int err;

    // Convert double values to float
    std::vector<float> values_f(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        values_f[i] = static_cast<float>(values[i]);
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
                               values_f.data(), 0, nullptr, nullptr);
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

void RadiosityOpenCLSolver::buildChunkedCsr(
    const std::vector<double>& values,
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
                    // boundary. Fail loud (PTM-25 backstop) rather than abort.
                    const size_t row_nnz =
                        static_cast<size_t>(row_ptrs[next + 1]) -
                        static_cast<size_t>(row_ptrs[next]);
                    throw std::runtime_error(
                        "RadiosityOpenCLSolver: row " + std::to_string(next) +
                        " has " + std::to_string(row_nnz) +
                        " non-zeros, exceeding the per-chunk budget of " +
                        std::to_string(max_chunk_nnz) +
                        " nnz (2^31-byte GPU addressing wall, PTM-25). This "
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

        // Convert this chunk's values slice to float.
        std::vector<float> vals_f(chunk_nnz);
        for (size_t k = 0; k < chunk_nnz; ++k) {
            vals_f[k] = static_cast<float>(values[base + k]);
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
                                   vals_f.data(), 0, nullptr, nullptr);
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

    if (mode == 0) {
        const size_t local_size = config_.work_group_size;
        const size_t gs = ((n_faces_ + local_size - 1) / local_size) * local_size;
        clSetKernelArg(spmv_kernel_, 0, sizeof(cl_mem), &buf_csr_values_);
        clSetKernelArg(spmv_kernel_, 1, sizeof(cl_mem), &buf_csr_col_indices_);
        clSetKernelArg(spmv_kernel_, 2, sizeof(cl_mem), &buf_csr_row_ptrs_);
        clSetKernelArg(spmv_kernel_, 3, sizeof(cl_mem), &buf_radiosity_);
        clSetKernelArg(spmv_kernel_, 4, sizeof(cl_mem), &buf_F_B_);
        clSetKernelArg(spmv_kernel_, 5, sizeof(int), &n_faces);
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
        clSetKernelArg(spmv_vector_kernel_, 0, sizeof(cl_mem), &buf_csr_values_);
        clSetKernelArg(spmv_vector_kernel_, 1, sizeof(cl_mem), &buf_csr_col_indices_);
        clSetKernelArg(spmv_vector_kernel_, 2, sizeof(cl_mem), &buf_csr_row_ptrs_);
        clSetKernelArg(spmv_vector_kernel_, 3, sizeof(cl_mem), &buf_radiosity_);
        clSetKernelArg(spmv_vector_kernel_, 4, sizeof(cl_mem), &buf_F_B_);
        clSetKernelArg(spmv_vector_kernel_, 5, sizeof(int), &n_faces);
        clSetKernelArg(spmv_vector_kernel_, 6, warp * sizeof(float), nullptr);
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

    // Chunked path (PTM-25) + vector coalescing: one wavefront per chunk-local row.
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

    cl_command_queue queue = context_->queue();
    int n_faces = static_cast<int>(n_faces_);
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
    // chunked path when the matrix exceeds the single-buffer wall, PTM-25).

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

    // Ensure tolerance is safe for single-precision (float has ~7 significant digits)
    const double effective_tolerance = std::max(config_.convergence_tolerance, 1e-5);

    for (int iter = 0; iter < config_.max_iterations; ++iter) {
        // SpMV: F_B = F * radiosity (single-buffer or chunked, PTM-25)
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
            err = clSetKernelArg(jacobi_kernel_, arg++, sizeof(cl_mem), &buf_change_);
            checkErr(err, "clSetKernelArg(jacobi_kernel,change)");
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
            // Download change buffer and compute max on CPU
            double max_change = computeMaxResidual();

            last_residual_ = max_change;
            last_iterations_ = iter + 1;

            if (iter_trace) {
                const double max_b = computeMaxOfBuffer(buf_radiosity_);
                const double rel = (max_b > 1e-10) ? max_change / max_b : max_change;
                std::fprintf(stderr,
                             "[rad-trace] gpu solve=%ld iter=%d dB_abs=%.6e B_max=%.6e rel=%.6e\n",
                             this_solve, iter + 1, max_change, max_b, rel);
            }

            if (max_change < effective_tolerance && !iter_trace_full) {
                converged_ = true;
                break;
            }
        }
    }

    // Compute final SpMV to get F * B_final (single-buffer or chunked, PTM-25)
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

double RadiosityOpenCLSolver::computeMaxResidual() {
    return computeMaxOfBuffer(buf_change_);
}

double RadiosityOpenCLSolver::computeMaxOfBuffer(cl_mem buf) {
    // Use GPU parallel reduction to find max change
    // Two-stage approach:
    // 1. Each work group reduces to one partial max
    // 2. Download partial maxes and find final max on CPU
    //
    // This is much faster than downloading the entire change array
    // (n_reduction_groups_ floats vs n_faces_ floats)

    cl_command_queue queue = context_->queue();
    cl_int err;

    // Helper for error checking
    auto checkErr = [](cl_int e, const char* op) {
        if (e != CL_SUCCESS) {
            throw std::runtime_error(std::string(op) + " failed: " + clErrorString(e));
        }
    };

    // Stage 1: GPU reduction - each work group produces one partial max
    int n_faces = static_cast<int>(n_faces_);
    size_t local_size = config_.work_group_size;
    size_t global_size = n_reduction_groups_ * local_size;
    size_t local_mem_size = local_size * sizeof(float);

    err = clSetKernelArg(max_reduction_kernel_, 0, sizeof(cl_mem), &buf);
    checkErr(err, "clSetKernelArg(max_reduction,input)");
    err = clSetKernelArg(max_reduction_kernel_, 1, sizeof(cl_mem), &buf_partial_max_);
    checkErr(err, "clSetKernelArg(max_reduction,partial_max)");
    err = clSetKernelArg(max_reduction_kernel_, 2, sizeof(int), &n_faces);
    checkErr(err, "clSetKernelArg(max_reduction,n)");
    err = clSetKernelArg(max_reduction_kernel_, 3, local_mem_size, nullptr);
    checkErr(err, "clSetKernelArg(max_reduction,shared)");

    err = clEnqueueNDRangeKernel(queue, max_reduction_kernel_, 1, nullptr,
                                 &global_size, &local_size, 0, nullptr, nullptr);
    checkErr(err, "clEnqueueNDRangeKernel(max_reduction)");

    // Stage 2: Download partial maxes (small array) and find final max on CPU
    std::vector<float> partial_maxes(n_reduction_groups_);
    err = clEnqueueReadBuffer(queue, buf_partial_max_, CL_TRUE, 0,
                              n_reduction_groups_ * sizeof(float),
                              partial_maxes.data(), 0, nullptr, nullptr);
    checkErr(err, "clEnqueueReadBuffer(partial_max)");

    float max_change = 0.0f;
    for (size_t i = 0; i < n_reduction_groups_; ++i) {
        max_change = std::max(max_change, partial_maxes[i]);
    }

    return static_cast<double>(max_change);
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

    // Release program
    if (program_) {
        clReleaseProgram(program_);
        program_ = nullptr;
    }

    // Release CSR buffers
    if (buf_csr_values_) { clReleaseMemObject(buf_csr_values_); buf_csr_values_ = nullptr; }
    if (buf_csr_col_indices_) { clReleaseMemObject(buf_csr_col_indices_); buf_csr_col_indices_ = nullptr; }
    if (buf_csr_row_ptrs_) { clReleaseMemObject(buf_csr_row_ptrs_); buf_csr_row_ptrs_ = nullptr; }

    // Release chunked CSR sub-buffers (PTM-25)
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

    // Release convergence buffer
    if (buf_change_) { clReleaseMemObject(buf_change_); buf_change_ = nullptr; }
    if (buf_partial_max_) { clReleaseMemObject(buf_partial_max_); buf_partial_max_ = nullptr; }

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
