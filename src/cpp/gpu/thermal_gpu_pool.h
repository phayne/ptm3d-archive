#pragma once

/**
 * @file thermal_gpu_pool.h
 * @brief Pool of ThermalOpenCLSolver instances spread across multiple GPUs.
 *
 * Owns N solvers (one per device). Submitted batches are routed to the next
 * idle device and run concurrently via std::async. Each device serializes
 * its own batches with a per-device mutex; cross-device batches run in
 * parallel.
 *
 * Single-device behaviour (N=1) routes through the same call path used by
 * the legacy single-GPU code, so byte-identical results are preserved when
 * the user does not opt in.
 */

#ifdef PTM_GPU_OPENCL

#include "opencl_config.h"
#include "thermal_opencl.h"

#include <chrono>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace gpu {
namespace opencl {

/**
 * @brief Configuration for ThermalGPUDevicePool
 */
struct ThermalGPUPoolConfig {
    // Selected device handles (one entry = one device the pool will own).
    // If empty, pool falls back to the legacy single first-available GPU.
    std::vector<OpenCLDeviceInfo> devices;

    // Optional pause inserted after each kernel completes on a given device,
    // before the next batch is enqueued on that same device. Useful as a
    // mitigation for the macOS+AMD `gpusKillClientExt` watchdog hang at long
    // sustained kernel cadence (see memory/gpu_hang_amd_macos.md).
    int inter_batch_pause_ms = 0;

    bool verbose = false;
};

/**
 * @brief Multi-device pool of ThermalOpenCLSolver instances.
 *
 * Each call to submit() returns a future that resolves when the worker has
 * finished the batch on its assigned device. The user-supplied callable
 * receives a reference to a freshly-created ThermalOpenCLSolver scoped to
 * one device for the duration of the batch.
 */
class ThermalGPUDevicePool {
public:
    /// One worker per entry in cfg.devices. Constructor does not initialize
    /// any solvers eagerly — the caller passes per-batch parameters
    /// (n_faces_in_batch, n_layers, ThermalOpenCLConfig) at submit time.
    explicit ThermalGPUDevicePool(ThermalGPUPoolConfig cfg);
    ~ThermalGPUDevicePool();

    ThermalGPUDevicePool(const ThermalGPUDevicePool&) = delete;
    ThermalGPUDevicePool& operator=(const ThermalGPUDevicePool&) = delete;

    /// Number of devices the pool will dispatch across.
    /// 1 if the pool is empty / single-device fallback.
    size_t numDevices() const noexcept { return cfg_.devices.size(); }

    /// Returns the device handle assigned to a given pool slot. Useful for
    /// printing per-batch which physical GPU did the work.
    const OpenCLDeviceInfo& deviceAt(size_t i) const { return cfg_.devices.at(i); }

    /**
     * @brief Submit a batch of work to be executed on the next-idle device.
     *
     * The work callable is `bool(ThermalOpenCLSolver& solver, size_t device_idx)`.
     * The callable owns the full lifecycle of `solver` for the batch — it
     * uploads inputs, runs the kernel, downloads outputs, and returns true
     * on success. The pool guarantees:
     *   - `solver` is initialized on a single, well-defined device.
     *   - No two batches run on the *same* device concurrently
     *     (per-device mutex).
     *   - Batches on *different* devices run in parallel.
     *   - Optional inter_batch_pause_ms sleep is honoured between
     *     consecutive batches on the same device.
     *
     * @param n_faces_in_batch Faces solver should size its buffers for.
     * @param n_layers         Depth layers per face.
     * @param solver_config    Forwarded to ThermalOpenCLSolver constructor.
     * @param work             Callback that does the per-batch work.
     * @return future<bool> — true if the batch ran cleanly.
     */
    std::future<bool> submit(
        size_t n_faces_in_batch,
        size_t n_layers,
        const ThermalOpenCLConfig& solver_config,
        std::function<bool(ThermalOpenCLSolver&, size_t)> work);

private:
    struct DeviceSlot {
        OpenCLDeviceInfo handle;
        std::mutex       mtx;        // serializes batches on this device
        std::chrono::steady_clock::time_point last_finish_{};
        bool             healthy = true;
    };

    /// Round-robin index of the next device to assign. Wrap with mtx_.
    size_t pickNextDeviceIdx();

    ThermalGPUPoolConfig            cfg_;
    std::vector<std::unique_ptr<DeviceSlot>> slots_;
    std::mutex                      mtx_;          // protects next_idx_
    size_t                          next_idx_ = 0;
};

}  // namespace opencl
}  // namespace gpu

#endif  // PTM_GPU_OPENCL
