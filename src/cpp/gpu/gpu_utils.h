#pragma once

/**
 * @file gpu_utils.h
 * @brief GPU utility functions: error handling, timing, diagnostics
 */

#include "gpu_config.h"
#include <iostream>
#include <sstream>
#include <chrono>
#include <stdexcept>

namespace gpu {

/**
 * @brief GPU error checking macro
 *
 * Usage: GPU_CHECK(gpuMalloc(&ptr, size));
 */
#if PTM_GPU_ENABLED
#define GPU_CHECK(call)                                                     \
    do {                                                                    \
        gpuError_t err = (call);                                            \
        if (err != gpuSuccess) {                                            \
            std::ostringstream oss;                                         \
            oss << "GPU error at " << __FILE__ << ":" << __LINE__ << "\n"   \
                << "  " << #call << "\n"                                    \
                << "  Error: " << gpuGetErrorString(err);                   \
            throw std::runtime_error(oss.str());                            \
        }                                                                   \
    } while (0)

#define GPU_CHECK_LAST()                                                    \
    do {                                                                    \
        gpuError_t err = gpuGetLastError();                                 \
        if (err != gpuSuccess) {                                            \
            std::ostringstream oss;                                         \
            oss << "GPU error at " << __FILE__ << ":" << __LINE__ << "\n"   \
                << "  Error: " << gpuGetErrorString(err);                   \
            throw std::runtime_error(oss.str());                            \
        }                                                                   \
    } while (0)
#else
#define GPU_CHECK(call) do { (void)(call); } while(0)
#define GPU_CHECK_LAST() do {} while(0)
#endif

/**
 * @brief RAII wrapper for GPU stream
 */
class GPUStream {
public:
    GPUStream() {
#if PTM_GPU_ENABLED
        GPU_CHECK(gpuStreamCreate(&stream_));
        valid_ = true;
#endif
    }

    ~GPUStream() {
#if PTM_GPU_ENABLED
        if (valid_) {
            gpuStreamDestroy(stream_);
        }
#endif
    }

    // No copy
    GPUStream(const GPUStream&) = delete;
    GPUStream& operator=(const GPUStream&) = delete;

    // Move
    GPUStream(GPUStream&& other) noexcept {
#if PTM_GPU_ENABLED
        stream_ = other.stream_;
        valid_ = other.valid_;
        other.valid_ = false;
#endif
    }

    GPUStream& operator=(GPUStream&& other) noexcept {
#if PTM_GPU_ENABLED
        if (this != &other) {
            if (valid_) {
                gpuStreamDestroy(stream_);
            }
            stream_ = other.stream_;
            valid_ = other.valid_;
            other.valid_ = false;
        }
#endif
        return *this;
    }

#if PTM_GPU_ENABLED
    gpuStream_t get() const { return stream_; }
    operator gpuStream_t() const { return stream_; }
#endif

    void synchronize() {
#if PTM_GPU_ENABLED
        if (valid_) {
            GPU_CHECK(gpuStreamSynchronize(stream_));
        }
#endif
    }

private:
#if PTM_GPU_ENABLED
    gpuStream_t stream_ = nullptr;
#endif
    bool valid_ = false;
};

/**
 * @brief GPU timer using events for accurate kernel timing
 */
class GPUTimer {
public:
    GPUTimer() {
#if PTM_GPU_ENABLED
        GPU_CHECK(gpuEventCreate(&start_));
        GPU_CHECK(gpuEventCreate(&stop_));
        valid_ = true;
#endif
    }

    ~GPUTimer() {
#if PTM_GPU_ENABLED
        if (valid_) {
            gpuEventDestroy(start_);
            gpuEventDestroy(stop_);
        }
#endif
    }

    // No copy
    GPUTimer(const GPUTimer&) = delete;
    GPUTimer& operator=(const GPUTimer&) = delete;

#if PTM_GPU_ENABLED
    void start(gpuStream_t stream = nullptr) {
        GPU_CHECK(gpuEventRecord(start_, stream));
    }

    void stop(gpuStream_t stream = nullptr) {
        GPU_CHECK(gpuEventRecord(stop_, stream));
    }
#else
    void start() { cpu_start_ = std::chrono::high_resolution_clock::now(); }
    void stop() { cpu_stop_ = std::chrono::high_resolution_clock::now(); }
#endif

    /**
     * @brief Get elapsed time in milliseconds
     *
     * Must call stop() before this. Blocks until stop event completes.
     */
    float elapsedMs() {
#if PTM_GPU_ENABLED
        GPU_CHECK(gpuEventSynchronize(stop_));
        float ms = 0.0f;
        GPU_CHECK(gpuEventElapsedTime(&ms, start_, stop_));
        return ms;
#else
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            cpu_stop_ - cpu_start_);
        return duration.count() / 1000.0f;
#endif
    }

private:
#if PTM_GPU_ENABLED
    gpuEvent_t start_ = nullptr;
    gpuEvent_t stop_ = nullptr;
#else
    std::chrono::high_resolution_clock::time_point cpu_start_, cpu_stop_;
#endif
    bool valid_ = false;
};

/**
 * @brief Print GPU device information
 */
inline void printDeviceInfo(const GPUConfig& config, std::ostream& out = std::cout) {
    out << "GPU Configuration:\n";
    out << "  Backend: " << config.backendName() << "\n";

    if (!config.isAvailable()) {
        out << "  Status: No GPU available\n";
        return;
    }

    out << "  Device: " << config.device_name << " (ID " << config.device_id << ")\n";
    out << "  Memory: " << (config.total_memory / (1024*1024)) << " MB total, "
        << (config.available_memory / (1024*1024)) << " MB available\n";
    out << "  Compute: " << config.compute_capability_major << "."
        << config.compute_capability_minor << "\n";
    out << "  SMs: " << config.num_sms << ", Warp size: " << config.warp_size << "\n";
    out << "  Max threads/block: " << config.max_threads_per_block << "\n";
    out << "  Optimal block size: " << config.optimal_block_size << "\n";
    out << "  Managed memory: " << (config.supports_managed_memory ? "yes" : "no") << "\n";
    out << "  Concurrent kernels: " << (config.supports_concurrent_kernels ? "yes" : "no") << "\n";
}

/**
 * @brief Compute grid dimensions for kernel launch
 *
 * @param total_threads Total work items (e.g., number of faces)
 * @param block_size Threads per block
 * @return Number of blocks (grid size)
 */
inline int computeGridSize(int total_threads, int block_size) {
    return (total_threads + block_size - 1) / block_size;
}

/**
 * @brief Estimate memory requirement for thermal batch processing
 *
 * @param n_faces Number of mesh faces
 * @param n_layers Depth layers per face
 * @return Estimated GPU memory in bytes
 */
inline size_t estimateThermalMemory(size_t n_faces, size_t n_layers) {
    // Per-layer arrays: T, k, kappa, cp, rho, kc = 6 arrays
    size_t layer_arrays = 6 * n_faces * n_layers * sizeof(float);

    // Per-face arrays: flux, emissivity, albedo, T_surface, Qb, T_deep = 6 arrays
    size_t face_arrays = 6 * n_faces * sizeof(float);

    // Grid arrays: z, dz, g1, g2 = ~4 * n_layers
    size_t grid_arrays = 4 * n_layers * sizeof(float);

    // Add 20% overhead for alignment and temporaries
    return static_cast<size_t>((layer_arrays + face_arrays + grid_arrays) * 1.2);
}

/**
 * @brief Check if mesh fits in GPU memory
 *
 * @param config GPU configuration
 * @param n_faces Number of mesh faces
 * @param n_layers Depth layers per face
 * @return true if mesh fits, false otherwise
 */
inline bool meshFitsInGPU(const GPUConfig& config, size_t n_faces, size_t n_layers) {
    if (!config.isAvailable()) return false;
    size_t required = estimateThermalMemory(n_faces, n_layers);
    return required < config.available_memory;
}

}  // namespace gpu
