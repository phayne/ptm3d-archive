#pragma once

/**
 * @file gpu_memory.h
 * @brief GPU memory management with Structure of Arrays (SoA) layout
 *
 * Provides efficient memory allocation and data transfer utilities
 * optimized for coalesced GPU memory access patterns.
 */

#include "gpu_config.h"
#include <algorithm>
#include <cstddef>
#include <vector>
#include <memory>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include "../constants/stefan_boltzmann.h"  // ptm::kStefanBoltzmann (the single definition)

namespace gpu {

/**
 * @brief RAII wrapper for GPU device memory
 *
 * Automatically frees memory on destruction. Supports both
 * GPU-allocated memory and provides CPU fallback when GPU unavailable.
 */
template<typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;

    /**
     * @brief Allocate device buffer
     * @param count Number of elements
     * @param use_gpu If false, allocate on host instead
     */
    explicit DeviceBuffer(size_t count, bool use_gpu = true)
        : count_(count), use_gpu_(use_gpu && isGPUCompiled()) {
        if (count_ == 0) return;

        if (use_gpu_) {
#if PTM_GPU_ENABLED
            gpuError_t err = gpuMalloc(&device_ptr_, count_ * sizeof(T));
            if (err != gpuSuccess) {
                throw std::runtime_error(
                    std::string("GPU allocation failed: ") + gpuGetErrorString(err));
            }
#endif
        } else {
            host_fallback_.resize(count_);
        }
    }

    ~DeviceBuffer() {
        free();
    }

    // Move semantics
    DeviceBuffer(DeviceBuffer&& other) noexcept
        : device_ptr_(other.device_ptr_)
        , host_fallback_(std::move(other.host_fallback_))
        , count_(other.count_)
        , use_gpu_(other.use_gpu_) {
        other.device_ptr_ = nullptr;
        other.count_ = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            free();
            device_ptr_ = other.device_ptr_;
            host_fallback_ = std::move(other.host_fallback_);
            count_ = other.count_;
            use_gpu_ = other.use_gpu_;
            other.device_ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    // No copy
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    /**
     * @brief Get raw device pointer
     */
    T* data() {
        if (use_gpu_) {
            return device_ptr_;
        }
        return host_fallback_.data();
    }

    const T* data() const {
        if (use_gpu_) {
            return device_ptr_;
        }
        return host_fallback_.data();
    }

    /**
     * @brief Copy data from host to device
     */
    void copyFromHost(const T* host_data, size_t count) {
        if (count > count_) {
            throw std::runtime_error("Copy exceeds buffer size");
        }
        if (use_gpu_) {
#if PTM_GPU_ENABLED
            gpuError_t err = gpuMemcpy(device_ptr_, host_data,
                                       count * sizeof(T), gpuMemcpyHostToDevice);
            if (err != gpuSuccess) {
                throw std::runtime_error(
                    std::string("GPU memcpy H2D failed: ") + gpuGetErrorString(err));
            }
#endif
        } else {
            std::memcpy(host_fallback_.data(), host_data, count * sizeof(T));
        }
    }

    void copyFromHost(const std::vector<T>& host_data) {
        copyFromHost(host_data.data(), host_data.size());
    }

    /**
     * @brief Copy data from device to host
     */
    void copyToHost(T* host_data, size_t count) const {
        if (count > count_) {
            throw std::runtime_error("Copy exceeds buffer size");
        }
        if (use_gpu_) {
#if PTM_GPU_ENABLED
            gpuError_t err = gpuMemcpy(host_data, device_ptr_,
                                       count * sizeof(T), gpuMemcpyDeviceToHost);
            if (err != gpuSuccess) {
                throw std::runtime_error(
                    std::string("GPU memcpy D2H failed: ") + gpuGetErrorString(err));
            }
#endif
        } else {
            std::memcpy(host_data, host_fallback_.data(), count * sizeof(T));
        }
    }

    void copyToHost(std::vector<T>& host_data) const {
        if (host_data.size() < count_) {
            host_data.resize(count_);
        }
        copyToHost(host_data.data(), count_);
    }

    /**
     * @brief Async copy from host (requires stream)
     */
#if PTM_GPU_ENABLED
    void copyFromHostAsync(const T* host_data, size_t count, gpuStream_t stream) {
        if (count > count_) {
            throw std::runtime_error("Copy exceeds buffer size");
        }
        if (use_gpu_) {
            gpuError_t err = gpuMemcpyAsync(device_ptr_, host_data,
                                            count * sizeof(T),
                                            gpuMemcpyHostToDevice, stream);
            if (err != gpuSuccess) {
                throw std::runtime_error(
                    std::string("GPU async memcpy H2D failed: ") + gpuGetErrorString(err));
            }
        } else {
            std::memcpy(host_fallback_.data(), host_data, count * sizeof(T));
        }
    }

    void copyToHostAsync(T* host_data, size_t count, gpuStream_t stream) const {
        if (count > count_) {
            throw std::runtime_error("Copy exceeds buffer size");
        }
        if (use_gpu_) {
            gpuError_t err = gpuMemcpyAsync(host_data, device_ptr_,
                                            count * sizeof(T),
                                            gpuMemcpyDeviceToHost, stream);
            if (err != gpuSuccess) {
                throw std::runtime_error(
                    std::string("GPU async memcpy D2H failed: ") + gpuGetErrorString(err));
            }
        } else {
            std::memcpy(host_data, host_fallback_.data(), count * sizeof(T));
        }
    }
#endif

    /**
     * @brief Set all elements to zero
     */
    void zero() {
        if (use_gpu_) {
#if PTM_GPU_ENABLED
            gpuMemset(device_ptr_, 0, count_ * sizeof(T));
#endif
        } else {
            std::memset(host_fallback_.data(), 0, count_ * sizeof(T));
        }
    }

    size_t size() const { return count_; }
    size_t bytes() const { return count_ * sizeof(T); }
    bool isOnGPU() const { return use_gpu_; }
    bool empty() const { return count_ == 0; }

private:
    void free() {
        if (use_gpu_ && device_ptr_) {
#if PTM_GPU_ENABLED
            gpuFree(device_ptr_);
#endif
            device_ptr_ = nullptr;
        }
        host_fallback_.clear();
        host_fallback_.shrink_to_fit();
    }

    T* device_ptr_ = nullptr;
    std::vector<T> host_fallback_;
    size_t count_ = 0;
    bool use_gpu_ = false;
};

/**
 * @brief Structure of Arrays (SoA) buffer for thermal model
 *
 * Organizes data for coalesced GPU memory access:
 * - T[layer][face] layout: T_all[layer * n_faces + face_id]
 * - Adjacent threads access adjacent memory locations
 */
struct ThermalSoABuffers {
    // Dimensions
    size_t n_faces = 0;
    size_t n_layers = 0;

    // Primary state arrays [n_faces * n_layers]
    DeviceBuffer<float> T;           // Temperature
    DeviceBuffer<float> k;           // Conductivity
    DeviceBuffer<float> kappa;       // Diffusivity
    DeviceBuffer<float> cp;          // Heat capacity
    DeviceBuffer<float> rho;         // Density
    DeviceBuffer<float> kc;          // Base conductivity (for k(T))

    // Surface arrays [n_faces]
    DeviceBuffer<float> flux;        // Surface absorbed flux
    DeviceBuffer<float> emissivity;  // Surface emissivity
    DeviceBuffer<float> albedo;      // Surface albedo
    DeviceBuffer<float> T_surface;   // Surface temperature (output)

    // Grid geometry [n_layers] - shared across all faces
    DeviceBuffer<float> z;           // Depth grid
    DeviceBuffer<float> dz;          // Layer spacing [n_layers - 1]
    DeviceBuffer<float> g1;          // Coupling coefficient 1
    DeviceBuffer<float> g2;          // Coupling coefficient 2

    // Bottom boundary [n_faces]
    DeviceBuffer<float> Qb;          // Bottom heat flux
    DeviceBuffer<float> T_deep;      // Deep boundary temperature

    // Material property coefficients (small, constant)
    DeviceBuffer<float> cp_coeffs;   // cp polynomial coefficients
    float cp0 = 0.0f;                // Base heat capacity
    float R350 = 0.0f;               // Radiative conductivity coefficient
    int n_cp_coeffs = 0;

    // Physical constants
    float sigma = static_cast<float>(ptm::kStefanBoltzmann);  // Stefan-Boltzmann

    /**
     * @brief Allocate all buffers
     *
     * @param faces Number of mesh faces
     * @param layers Number of depth layers per face
     * @param use_gpu Allocate on GPU (true) or CPU fallback (false)
     */
    void allocate(size_t faces, size_t layers, bool use_gpu = true) {
        n_faces = faces;
        n_layers = layers;
        size_t total = faces * layers;

        // State arrays
        T = DeviceBuffer<float>(total, use_gpu);
        k = DeviceBuffer<float>(total, use_gpu);
        kappa = DeviceBuffer<float>(total, use_gpu);
        cp = DeviceBuffer<float>(total, use_gpu);
        rho = DeviceBuffer<float>(total, use_gpu);
        kc = DeviceBuffer<float>(total, use_gpu);

        // Surface arrays
        flux = DeviceBuffer<float>(faces, use_gpu);
        emissivity = DeviceBuffer<float>(faces, use_gpu);
        albedo = DeviceBuffer<float>(faces, use_gpu);
        T_surface = DeviceBuffer<float>(faces, use_gpu);

        // Grid arrays (shared)
        z = DeviceBuffer<float>(layers, use_gpu);
        dz = DeviceBuffer<float>(layers > 0 ? layers - 1 : 0, use_gpu);
        g1 = DeviceBuffer<float>(layers, use_gpu);
        g2 = DeviceBuffer<float>(layers, use_gpu);

        // Bottom boundary
        Qb = DeviceBuffer<float>(faces, use_gpu);
        T_deep = DeviceBuffer<float>(faces, use_gpu);
    }

    /**
     * @brief Get linear index for (face, layer)
     *
     * SoA layout: all faces at layer 0, then all faces at layer 1, etc.
     * This ensures coalesced access when threads iterate over faces.
     */
    size_t index(size_t face, size_t layer) const {
        return layer * n_faces + face;
    }

    /**
     * @brief Total GPU memory used (bytes)
     */
    size_t totalBytes() const {
        return T.bytes() + k.bytes() + kappa.bytes() + cp.bytes() +
               rho.bytes() + kc.bytes() + flux.bytes() + emissivity.bytes() +
               albedo.bytes() + T_surface.bytes() + z.bytes() + dz.bytes() +
               g1.bytes() + g2.bytes() + Qb.bytes() + T_deep.bytes() +
               cp_coeffs.bytes();
    }

    /**
     * @brief Check if buffers are allocated on GPU
     */
    bool isOnGPU() const {
        return T.isOnGPU();
    }
};

/**
 * @brief Convert AoS temperature data to SoA layout
 *
 * Input: temps[face][layer] - Array of Structures
 * Output: soa_T[layer * n_faces + face] - Structure of Arrays
 *
 * @param temps_aos AoS data [n_faces][n_layers]
 * @param soa_T Output SoA buffer (must be pre-allocated)
 * @param n_faces Number of faces
 * @param n_layers Number of layers
 */
inline void convertAoStoSoA(
    const std::vector<std::vector<double>>& temps_aos,
    std::vector<float>& soa_T,
    size_t n_faces,
    size_t n_layers)
{
    soa_T.resize(n_faces * n_layers);

    for (size_t f = 0; f < n_faces; ++f) {
        const auto& face_temps = temps_aos[f];
        size_t face_layers = std::min(face_temps.size(), n_layers);
        for (size_t l = 0; l < face_layers; ++l) {
            soa_T[l * n_faces + f] = static_cast<float>(face_temps[l]);
        }
        // Zero-pad if face has fewer layers
        for (size_t l = face_layers; l < n_layers; ++l) {
            soa_T[l * n_faces + f] = 0.0f;
        }
    }
}

/**
 * @brief Convert SoA temperature data back to AoS layout
 *
 * @param soa_T SoA data [n_layers * n_faces]
 * @param temps_aos Output AoS [n_faces][n_layers]
 * @param n_faces Number of faces
 * @param n_layers Number of layers
 */
inline void convertSoAtoAoS(
    const std::vector<float>& soa_T,
    std::vector<std::vector<double>>& temps_aos,
    size_t n_faces,
    size_t n_layers)
{
    temps_aos.resize(n_faces);

    for (size_t f = 0; f < n_faces; ++f) {
        temps_aos[f].resize(n_layers);
        for (size_t l = 0; l < n_layers; ++l) {
            temps_aos[f][l] = static_cast<double>(soa_T[l * n_faces + f]);
        }
    }
}

/**
 * @brief Extract surface temperatures from SoA buffer
 *
 * @param soa_T Full SoA temperature buffer
 * @param surface_temps Output surface temperatures [n_faces]
 * @param n_faces Number of faces
 */
inline void extractSurfaceTemps(
    const std::vector<float>& soa_T,
    std::vector<float>& surface_temps,
    size_t n_faces)
{
    surface_temps.resize(n_faces);
    // Layer 0 is at indices [0, n_faces)
    std::memcpy(surface_temps.data(), soa_T.data(), n_faces * sizeof(float));
}

}  // namespace gpu
