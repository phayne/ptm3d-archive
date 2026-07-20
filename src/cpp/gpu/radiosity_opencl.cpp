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
#include <algorithm>
#include <cmath>

namespace gpu {
namespace opencl {

// Embedded kernel source for radiosity solver
std::string getRadiosityKernelSource() {
    return R"CL_SOURCE(
// OpenCL kernels for PTM radiosity solver
// Implements Jacobi iteration for radiosity equation: B = E + rho * F * B

#define SIGMA_SB 5.67051196e-8f

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

// Jacobi iteration update: B_new = E + solar + rho * (F * B_old)
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

    float B_new = emission[i] + solar_source[i] + reflectivity[i] * F_B[i];
    radiosity_new[i] = B_new;
    float diff = B_new - radiosity_old[i];
    change[i] = (diff >= 0.0f) ? diff : -diff;
}

// Compute absorbed indirect flux: Q_indirect = epsilon * (F * B)
__kernel void indirect_flux_kernel(
    __global const float* emissivity,
    __global const float* F_B,
    __global float* indirect_flux,
    int n_faces)
{
    int i = get_global_id(0);
    if (i >= n_faces) return;

    indirect_flux[i] = emissivity[i] * F_B[i];
}
)CL_SOURCE";
}

// RadiosityOpenCLSolver implementation

RadiosityOpenCLSolver::RadiosityOpenCLSolver(size_t n_faces,
                                             const RadiosityOpenCLConfig& config)
    : n_faces_(n_faces)
    , nnz_(0)
    , config_(config)
{
    change_host_.resize(n_faces);
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
    const std::vector<int>& row_ptrs)
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
                               row_ptrs.size() * sizeof(int),
                               row_ptrs.data(), 0, nullptr, nullptr);
    checkErr(err, "clEnqueueWriteBuffer(csr_row_ptrs)");

    view_factors_uploaded_ = true;

    if (config_.verbose) {
        size_t mem_mb = (nnz_ * (sizeof(float) + sizeof(int)) +
                        row_ptrs.size() * sizeof(int)) / (1024 * 1024);
        std::cout << "Uploaded view factor matrix to GPU: "
                  << nnz_ << " non-zeros (" << mem_mb << " MB)\n";
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

    // Set up SpMV kernel args (these don't change during iteration)
    err = clSetKernelArg(spmv_kernel_, 0, sizeof(cl_mem), &buf_csr_values_);
    checkErr(err, "clSetKernelArg(spmv_kernel,csr_values)");
    err = clSetKernelArg(spmv_kernel_, 1, sizeof(cl_mem), &buf_csr_col_indices_);
    checkErr(err, "clSetKernelArg(spmv_kernel,csr_col_indices)");
    err = clSetKernelArg(spmv_kernel_, 2, sizeof(cl_mem), &buf_csr_row_ptrs_);
    checkErr(err, "clSetKernelArg(spmv_kernel,csr_row_ptrs)");
    // Arg 3 (input x) and 4 (output y) will be set per iteration
    err = clSetKernelArg(spmv_kernel_, 5, sizeof(int), &n_faces);
    checkErr(err, "clSetKernelArg(spmv_kernel,n_faces)");

    // Jacobi iteration with periodic convergence checking
    // Checking every iteration causes PCIe latency overhead; check every N iterations
    converged_ = false;
    last_residual_ = std::numeric_limits<double>::max();
    const int check_interval = std::max(1, config_.convergence_check_interval);

    // Ensure tolerance is safe for single-precision (float has ~7 significant digits)
    const double effective_tolerance = std::max(config_.convergence_tolerance, 1e-5);

    for (int iter = 0; iter < config_.max_iterations; ++iter) {
        // SpMV: F_B = F * radiosity
        err = clSetKernelArg(spmv_kernel_, 3, sizeof(cl_mem), &buf_radiosity_);
        checkErr(err, "clSetKernelArg(spmv_kernel,radiosity)");
        err = clSetKernelArg(spmv_kernel_, 4, sizeof(cl_mem), &buf_F_B_);
        checkErr(err, "clSetKernelArg(spmv_kernel,F_B)");

        err = clEnqueueNDRangeKernel(queue, spmv_kernel_, 1, nullptr,
                                     &global_size, &local_size, 0, nullptr, nullptr);
        checkErr(err, "clEnqueueNDRangeKernel(spmv_kernel)");

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

            if (max_change < effective_tolerance) {
                converged_ = true;
                break;
            }
        }
    }

    // Compute final SpMV to get F * B_final
    err = clSetKernelArg(spmv_kernel_, 3, sizeof(cl_mem), &buf_radiosity_);
    checkErr(err, "clSetKernelArg(spmv_kernel,radiosity,final)");
    err = clSetKernelArg(spmv_kernel_, 4, sizeof(cl_mem), &buf_F_B_);
    checkErr(err, "clSetKernelArg(spmv_kernel,F_B,final)");

    err = clEnqueueNDRangeKernel(queue, spmv_kernel_, 1, nullptr,
                                 &global_size, &local_size, 0, nullptr, nullptr);
    checkErr(err, "clEnqueueNDRangeKernel(spmv_kernel,final)");

    // Compute indirect flux: Q_indirect = eps * (F * B)
    {
        int arg = 0;
        err = clSetKernelArg(indirect_flux_kernel_, arg++, sizeof(cl_mem), &buf_emissivity_);
        checkErr(err, "clSetKernelArg(indirect_flux_kernel,emissivity)");
        err = clSetKernelArg(indirect_flux_kernel_, arg++, sizeof(cl_mem), &buf_F_B_);
        checkErr(err, "clSetKernelArg(indirect_flux_kernel,F_B)");
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

    err = clSetKernelArg(max_reduction_kernel_, 0, sizeof(cl_mem), &buf_change_);
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

    // Release program
    if (program_) {
        clReleaseProgram(program_);
        program_ = nullptr;
    }

    // Release CSR buffers
    if (buf_csr_values_) { clReleaseMemObject(buf_csr_values_); buf_csr_values_ = nullptr; }
    if (buf_csr_col_indices_) { clReleaseMemObject(buf_csr_col_indices_); buf_csr_col_indices_ = nullptr; }
    if (buf_csr_row_ptrs_) { clReleaseMemObject(buf_csr_row_ptrs_); buf_csr_row_ptrs_ = nullptr; }

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
