#pragma once

/**
 * @file gpu_config.h
 * @brief GPU backend configuration and feature detection
 *
 * Provides compile-time and runtime detection of GPU capabilities.
 * Supports HIP (AMD/NVIDIA portable), CUDA, and OpenCL backends.
 */

#include <algorithm>
#include <string>
#include <cstdint>

// Backend detection macros
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_NVIDIA__)
    #define PTM_GPU_HIP 1
    #include <hip/hip_runtime.h>
    #define PTM_GPU_ENABLED 1
#elif defined(__CUDACC__)
    #define PTM_GPU_CUDA 1
    #include <cuda_runtime.h>
    #define PTM_GPU_ENABLED 1
#elif defined(PTM_GPU_OPENCL) && PTM_GPU_OPENCL
    // OpenCL backend - headers included in opencl_config.h
    #define PTM_GPU_ENABLED 1
#else
    #define PTM_GPU_ENABLED 0
#endif

// Portable type aliases
#if PTM_GPU_ENABLED
    #if PTM_GPU_HIP
        using gpuError_t = hipError_t;
        using gpuStream_t = hipStream_t;
        using gpuEvent_t = hipEvent_t;
        using gpuDeviceProp_t = hipDeviceProp_t;

        #define gpuSuccess hipSuccess
        #define gpuGetLastError hipGetLastError
        #define gpuGetErrorString hipGetErrorString
        #define gpuMalloc hipMalloc
        #define gpuFree hipFree
        #define gpuMemcpy hipMemcpy
        #define gpuMemcpyAsync hipMemcpyAsync
        #define gpuMemcpyHostToDevice hipMemcpyHostToDevice
        #define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
        #define gpuMemset hipMemset
        #define gpuDeviceSynchronize hipDeviceSynchronize
        #define gpuStreamCreate hipStreamCreate
        #define gpuStreamDestroy hipStreamDestroy
        #define gpuStreamSynchronize hipStreamSynchronize
        #define gpuEventCreate hipEventCreate
        #define gpuEventDestroy hipEventDestroy
        #define gpuEventRecord hipEventRecord
        #define gpuEventSynchronize hipEventSynchronize
        #define gpuEventElapsedTime hipEventElapsedTime
        #define gpuGetDeviceCount hipGetDeviceCount
        #define gpuGetDevice hipGetDevice
        #define gpuSetDevice hipSetDevice
        #define gpuGetDeviceProperties hipGetDeviceProperties

    #elif PTM_GPU_CUDA
        using gpuError_t = cudaError_t;
        using gpuStream_t = cudaStream_t;
        using gpuEvent_t = cudaEvent_t;
        using gpuDeviceProp_t = cudaDeviceProp;

        #define gpuSuccess cudaSuccess
        #define gpuGetLastError cudaGetLastError
        #define gpuGetErrorString cudaGetErrorString
        #define gpuMalloc cudaMalloc
        #define gpuFree cudaFree
        #define gpuMemcpy cudaMemcpy
        #define gpuMemcpyAsync cudaMemcpyAsync
        #define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
        #define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
        #define gpuMemset cudaMemset
        #define gpuDeviceSynchronize cudaDeviceSynchronize
        #define gpuStreamCreate cudaStreamCreate
        #define gpuStreamDestroy cudaStreamDestroy
        #define gpuStreamSynchronize cudaStreamSynchronize
        #define gpuEventCreate cudaEventCreate
        #define gpuEventDestroy cudaEventDestroy
        #define gpuEventRecord cudaEventRecord
        #define gpuEventSynchronize cudaEventSynchronize
        #define gpuEventElapsedTime cudaEventElapsedTime
        #define gpuGetDeviceCount cudaGetDeviceCount
        #define gpuGetDevice cudaGetDevice
        #define gpuSetDevice cudaSetDevice
        #define gpuGetDeviceProperties cudaGetDeviceProperties
    #endif
#endif

namespace gpu {

/**
 * @brief GPU backend type enumeration
 */
enum class Backend {
    None,    // No GPU support compiled
    HIP,     // HIP (AMD ROCm or NVIDIA via HIP)
    CUDA,    // Native CUDA
    OpenCL   // OpenCL (portable, works on macOS with AMD)
};

/**
 * @brief Runtime GPU configuration and capabilities
 */
struct GPUConfig {
    Backend backend = Backend::None;
    int device_id = 0;
    std::string device_name;
    size_t total_memory = 0;          // bytes
    size_t available_memory = 0;      // bytes (approximate)
    int compute_capability_major = 0;
    int compute_capability_minor = 0;
    int max_threads_per_block = 0;
    int max_blocks_per_sm = 0;
    int num_sms = 0;                  // Streaming multiprocessors
    int warp_size = 0;
    bool supports_managed_memory = false;
    bool supports_concurrent_kernels = false;

    // Computed launch parameters
    int optimal_block_size = 256;     // Threads per block
    int max_active_blocks = 0;        // For occupancy

    /**
     * @brief Check if GPU is available and usable
     */
    bool isAvailable() const { return backend != Backend::None; }

    /**
     * @brief Get backend name as string
     */
    std::string backendName() const {
        switch (backend) {
            case Backend::HIP: return "HIP";
            case Backend::CUDA: return "CUDA";
            case Backend::OpenCL: return "OpenCL";
            default: return "None";
        }
    }
};

/**
 * @brief Initialize GPU and detect capabilities
 *
 * @param device_id GPU device to use (default 0)
 * @return GPUConfig with detected capabilities, or backend=None if unavailable
 *
 * @note For OpenCL backend, this returns a minimal config since OpenCL uses
 *       its own initialization path via ThermalOpenCLSolver::initialize().
 */
inline GPUConfig initializeGPU(int device_id = 0) {
    GPUConfig config;

#if defined(PTM_GPU_HIP) && PTM_GPU_HIP
    int device_count = 0;
    gpuError_t err = gpuGetDeviceCount(&device_count);

    if (err != gpuSuccess || device_count == 0) {
        return config;  // No GPU available
    }

    if (device_id >= device_count) {
        device_id = 0;  // Fall back to first device
    }

    err = gpuSetDevice(device_id);
    if (err != gpuSuccess) {
        return config;
    }

    gpuDeviceProp_t props;
    err = gpuGetDeviceProperties(&props, device_id);
    if (err != gpuSuccess) {
        return config;
    }

    config.backend = Backend::HIP;
    config.device_name = props.name;
    config.total_memory = props.totalGlobalMem;
    config.num_sms = props.multiProcessorCount;
    config.max_threads_per_block = props.maxThreadsPerBlock;
    config.warp_size = props.warpSize;
    // HIP uses gcnArch for AMD, or major/minor for NVIDIA via HIP
    #if defined(__HIP_PLATFORM_AMD__)
        config.compute_capability_major = props.gcnArch / 100;
        config.compute_capability_minor = props.gcnArch % 100;
    #else
        config.compute_capability_major = props.major;
        config.compute_capability_minor = props.minor;
    #endif
    config.supports_managed_memory = props.managedMemory;
    config.supports_concurrent_kernels = props.concurrentKernels;

    config.device_id = device_id;

    // Compute optimal block size (multiple of warp size, <= 256 for register pressure)
    config.optimal_block_size = std::min(256, config.max_threads_per_block);
    config.optimal_block_size = (config.optimal_block_size / config.warp_size) * config.warp_size;

    // Estimate available memory (leave some headroom)
    config.available_memory = static_cast<size_t>(config.total_memory * 0.85);

#elif defined(PTM_GPU_CUDA) && PTM_GPU_CUDA
    int device_count = 0;
    gpuError_t err = gpuGetDeviceCount(&device_count);

    if (err != gpuSuccess || device_count == 0) {
        return config;  // No GPU available
    }

    if (device_id >= device_count) {
        device_id = 0;  // Fall back to first device
    }

    err = gpuSetDevice(device_id);
    if (err != gpuSuccess) {
        return config;
    }

    gpuDeviceProp_t props;
    err = gpuGetDeviceProperties(&props, device_id);
    if (err != gpuSuccess) {
        return config;
    }

    config.backend = Backend::CUDA;
    config.device_name = props.name;
    config.total_memory = props.totalGlobalMem;
    config.num_sms = props.multiProcessorCount;
    config.max_threads_per_block = props.maxThreadsPerBlock;
    config.warp_size = props.warpSize;
    config.compute_capability_major = props.major;
    config.compute_capability_minor = props.minor;
    config.supports_managed_memory = props.managedMemory;
    config.supports_concurrent_kernels = props.concurrentKernels;

    config.device_id = device_id;

    // Compute optimal block size (multiple of warp size, <= 256 for register pressure)
    config.optimal_block_size = std::min(256, config.max_threads_per_block);
    config.optimal_block_size = (config.optimal_block_size / config.warp_size) * config.warp_size;

    // Estimate available memory (leave some headroom)
    config.available_memory = static_cast<size_t>(config.total_memory * 0.85);

#elif defined(PTM_GPU_OPENCL) && PTM_GPU_OPENCL
    // OpenCL uses its own initialization path via ThermalOpenCLSolver
    // Return a minimal config indicating OpenCL is available
    config.backend = Backend::OpenCL;
    config.device_id = device_id;
    config.optimal_block_size = 256;  // Default work group size
#endif

    return config;
}

/**
 * @brief Check if GPU support was compiled in
 */
constexpr bool isGPUCompiled() {
#if PTM_GPU_ENABLED
    return true;
#else
    return false;
#endif
}

/**
 * @brief Get compile-time backend
 */
constexpr Backend getCompiledBackend() {
#if PTM_GPU_HIP
    return Backend::HIP;
#elif PTM_GPU_CUDA
    return Backend::CUDA;
#elif PTM_GPU_OPENCL
    return Backend::OpenCL;
#else
    return Backend::None;
#endif
}

}  // namespace gpu
