#pragma once

/**
 * @file bvh_gpu_pool.h
 * @brief Pool of BVHRayTracer instances spread across multiple GPUs.
 *
 * Owns N BVHRayTracer instances (one per device). The same mesh + BVH is
 * replicated on every device. A traceShadowRays() call partitions the
 * input ray arrays into N approximately-equal contiguous slices and
 * dispatches them concurrently via std::async; each device serializes
 * its own work via a per-device mutex.
 *
 * Single-device use (N=1) routes through the same call path used by the
 * legacy code, so byte-identical results are preserved when the user
 * does not opt in to multi-GPU.
 */

#ifdef PTM_GPU_OPENCL

#include "opencl_config.h"
#include "bvh_opencl.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace gpu {
namespace opencl {

class BVHGPUPool {
public:
    /// Construct a pool over the given device handles. Tracers are NOT
    /// initialized eagerly — call initialize() to build them.
    explicit BVHGPUPool(std::vector<OpenCLDeviceInfo> devices,
                       int inter_batch_pause_ms = 0,
                       bool verbose = false);
    ~BVHGPUPool();

    BVHGPUPool(const BVHGPUPool&) = delete;
    BVHGPUPool& operator=(const BVHGPUPool&) = delete;

    /// Build one BVHRayTracer per device. Drops devices that fail to
    /// initialize. Returns true if at least one device succeeded.
    bool initialize();

    /// Upload mesh to every healthy device. Returns false on any device
    /// failure (the device is dropped from the pool).
    bool uploadMesh(const std::vector<std::array<double, 3>>& vertices,
                    const std::vector<std::array<int, 3>>& indices);
    bool uploadMesh(const std::vector<std::array<float, 3>>& vertices,
                    const std::vector<std::array<uint32_t, 3>>& indices);

    /// Upload BVH to every healthy device. Same drop-on-failure semantics.
    bool uploadBVH(const BVHBuilder& bvh);

    /// True iff at least one device is initialized AND has mesh+BVH uploaded.
    bool isReady() const noexcept;

    /// Number of currently-healthy devices.
    size_t numDevices() const noexcept;

    /// Optional access to per-slot device metadata (for diagnostics).
    const OpenCLDeviceInfo& deviceAt(size_t i) const;

    /// Aggregated stats summed across all healthy devices.
    /// (Wall time is the *sum* across devices; for parallel wall time
    /// use the model's own wall-clock timers.)
    BVHRayTracer::Stats getAggregatedStats() const;

    /// Trace shadow rays across all healthy devices in parallel.
    /// Output `occluded_out` is sized to match the input ray count.
    /// Slice boundaries are static: each device gets a contiguous range.
    void traceShadowRays(const std::vector<float>& ray_org_x,
                         const std::vector<float>& ray_org_y,
                         const std::vector<float>& ray_org_z,
                         const std::vector<float>& ray_dir_x,
                         const std::vector<float>& ray_dir_y,
                         const std::vector<float>& ray_dir_z,
                         std::vector<uint8_t>& occluded_out);

    /// Per-ray tmax variant (for view-factor visibility).
    void traceShadowRays(const std::vector<float>& ray_org_x,
                         const std::vector<float>& ray_org_y,
                         const std::vector<float>& ray_org_z,
                         const std::vector<float>& ray_dir_x,
                         const std::vector<float>& ray_dir_y,
                         const std::vector<float>& ray_dir_z,
                         const std::vector<float>& ray_tmax,
                         std::vector<uint8_t>& occluded_out);

    /// Convenience overload for AoS ray data (matches BVHRayTracer's
    /// existing convenience signature). Internally splits, then partitions.
    void traceShadowRays(const std::vector<std::array<double, 3>>& origins,
                         const std::vector<std::array<double, 3>>& directions,
                         std::vector<uint8_t>& occluded_out);

private:
    struct DeviceSlot {
        OpenCLDeviceInfo            handle{};
        std::unique_ptr<BVHRayTracer> tracer;
        std::mutex                  mtx;
        std::chrono::steady_clock::time_point last_finish_{};
        bool                        healthy = true;
    };

    int  inter_batch_pause_ms_;
    bool verbose_;
    std::vector<std::unique_ptr<DeviceSlot>> slots_;

    /// Returns indices of healthy slots in stable enumeration order.
    std::vector<size_t> healthyIndices() const;
};

}  // namespace opencl
}  // namespace gpu

#endif  // PTM_GPU_OPENCL
