#pragma once

/**
 * @file thermal_gpu_pool.h
 * @brief Pool of long-lived ThermalOpenCLSolver instances spread across multiple GPUs.
 *
 * Owns N solvers (one per device), eagerly initialized at pool construction
 * and reused across every batch submitted to the pool. Submitted batches are
 * routed to the next idle device and run concurrently via std::async. Each
 * device serializes its own batches with a per-device mutex; cross-device
 * batches run in parallel.
 *
 * Why long-lived solvers: re-creating an OpenCL context per batch is the
 * failure mode that wedged the macOS+AMD driver during the radiation-coupling
 * re-thermal phase (see memory/gpu_hang_amd_macos.md and
 * memory/multi_gpu_thermal.md). The pool now creates each device's context
 * once and reuses it for every batch on that device.
 *
 * Single-device behavior (N=1) is unchanged: callers continue to bypass the
 * pool entirely on the legacy single-GPU path, so byte-identical results are
 * preserved when the user does not opt in.
 */

#ifdef PTM_GPU_OPENCL

#include "opencl_config.h"
#include "thermal_opencl.h"

#include <chrono>
#include <cstddef>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <vector>

namespace gpu {
namespace opencl {

/**
 * @brief Configuration for ThermalGPUDevicePool
 */
struct ThermalGPUPoolConfig {
    // Selected device handles (one entry = one device the pool will own).
    std::vector<OpenCLDeviceInfo> devices;

    // Maximum batch size the pool needs to support. Each device's solver is
    // allocated once with this capacity; submitted batches must be <= this.
    size_t max_n_faces = 0;

    // Depth layers per face (kernel + buffer dimension).
    size_t n_layers = 0;

    // Per-solver OpenCL configuration (work-group size, kernel path).
    ThermalOpenCLConfig solver_config;

    // Optional pause inserted after each kernel completes on a given device,
    // before the next batch is enqueued on that same device. Useful as a
    // mitigation for the macOS+AMD `gpusKillClientExt` watchdog hang at long
    // sustained kernel cadence.
    int inter_batch_pause_ms = 0;

    bool verbose = false;
};

/**
 * @brief Multi-device pool of long-lived ThermalOpenCLSolver instances.
 *
 * Each call to submit() returns a future that resolves when the worker has
 * finished the batch on its assigned device. The user-supplied callable
 * receives a reference to the persistent ThermalOpenCLSolver pinned to one
 * device. The solver's active face count is set to `n_faces_in_batch` before
 * the callable is invoked; the callable must size its uploaded data to match.
 */
class ThermalGPUDevicePool {
public:
    /// Constructs and eagerly initializes one solver per entry in
    /// cfg.devices. Throws std::runtime_error if any solver fails to
    /// initialize on its device. Must outlive every future submit() returns.
    explicit ThermalGPUDevicePool(ThermalGPUPoolConfig cfg);
    ~ThermalGPUDevicePool();

    ThermalGPUDevicePool(const ThermalGPUDevicePool&) = delete;
    ThermalGPUDevicePool& operator=(const ThermalGPUDevicePool&) = delete;

    /// Number of devices the pool dispatches across.
    size_t numDevices() const noexcept { return slots_.size(); }

    /// Per-device max face count (cfg.max_n_faces). submit() rejects
    /// larger batches; callers should size their BATCH_SIZE accordingly.
    size_t capacity() const noexcept { return cfg_.max_n_faces; }

    /// Layer count the pool was sized for (cfg.n_layers). Callers can
    /// verify this matches their solver's expected n_layers before
    /// reusing a shared pool across different code paths.
    size_t nLayers() const noexcept { return cfg_.n_layers; }

    /// Returns the device handle assigned to a given pool slot.
    const OpenCLDeviceInfo& deviceAt(size_t i) const { return slots_.at(i)->handle; }

    /**
     * @brief Submit a batch of work to be executed on the next-idle device.
     *
     * The work callable is `bool(ThermalOpenCLSolver& solver, size_t device_idx)`.
     * Before the callable runs the pool calls `solver.setActiveFaceCount(n_faces_in_batch)`
     * so kernel launches operate on the correct prefix of the device buffers.
     *
     * The pool guarantees:
     *   - `solver` is the persistent solver pinned to one device.
     *   - No two batches run on the *same* device concurrently
     *     (per-device mutex).
     *   - Batches on *different* devices run in parallel.
     *   - Optional inter_batch_pause_ms sleep is honored between
     *     consecutive batches on the same device.
     *
     * @param n_faces_in_batch Number of faces in this batch (<= max_n_faces).
     * @param work             Callback that does the per-batch work.
     * @return future<bool> — true if the batch ran cleanly.
     */
    std::future<bool> submit(
        size_t n_faces_in_batch,
        std::function<bool(ThermalOpenCLSolver&, size_t)> work);

private:
    struct DeviceSlot {
        OpenCLDeviceInfo handle;
        std::unique_ptr<ThermalOpenCLSolver> solver;
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
