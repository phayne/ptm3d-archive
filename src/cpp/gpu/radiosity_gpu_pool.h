#pragma once

/**
 * @file radiosity_gpu_pool.h
 * @brief Pool of RadiosityOpenCLSolver instances spread across multiple GPUs.
 *
 * One persistent solver per device, instantiated and pre-loaded with the
 * static view-factor matrix and face properties at pool init. Submitted
 * batches are routed to the next-idle device; per-device solves run
 * serially under a per-device mutex, while different devices run in
 * parallel.
 *
 * Per-timestep radiosity solves (B = E + rho*F*B + rho*Q_solar) are
 * mutually independent — they share only the CSR view-factor matrix F
 * and the per-face emissivity/reflectivity vectors, all of which are
 * uploaded once per device at init. Each batch's input (T, Q_solar) and
 * output (Q_indirect) is per-timestep state, transferred at solve time.
 *
 * Single-device behavior is the legacy single-GPU path used directly
 * (no pool); the pool is only instantiated when n_devices > 1.
 */

#ifdef PTM_GPU_OPENCL

#include "opencl_config.h"
#include "radiosity_opencl.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace gpu {
namespace opencl {

/**
 * @brief Configuration for RadiosityGPUDevicePool.
 */
struct RadiosityGPUPoolConfig {
    /// Selected device handles; one entry = one device the pool will own.
    /// If empty, submit() will fail loudly. The single-device fallback
    /// path lives in RadiositySolver and does not use this pool.
    std::vector<OpenCLDeviceInfo> devices;

    /// Optional pause inserted after each kernel completes on a given
    /// device, before the next batch is enqueued on that same device.
    /// Useful as a mitigation for the macOS+AMD `gpusKillClientExt`
    /// watchdog hang at long sustained kernel cadence
    /// (see memory/gpu_hang_amd_macos.md).
    int inter_batch_pause_ms = 0;

    bool verbose = false;
};

/**
 * @brief Multi-device pool of RadiosityOpenCLSolver instances.
 *
 * Lifecycle:
 *   1. Construct with config naming N devices.
 *   2. initialize(n_faces, solver_cfg) — spins up one solver per device,
 *      pinned to its device via OpenCLContext::initialize(handle).
 *   3. uploadStaticData(...) — replicates the CSR view-factor matrix and
 *      per-face emissivity/reflectivity to every device.
 *   4. submit(work) — async dispatch of a batch to the next-idle device.
 *      The work callback receives a reference to that device's persistent
 *      RadiosityOpenCLSolver (already loaded with static data) and is
 *      free to call solve() against it.
 */
class RadiosityGPUDevicePool {
public:
    using BatchCallback = std::function<bool(RadiosityOpenCLSolver&, size_t)>;

    explicit RadiosityGPUDevicePool(RadiosityGPUPoolConfig cfg);
    ~RadiosityGPUDevicePool();

    RadiosityGPUDevicePool(const RadiosityGPUDevicePool&) = delete;
    RadiosityGPUDevicePool& operator=(const RadiosityGPUDevicePool&) = delete;

    /// Spin up one persistent RadiosityOpenCLSolver per device, pinned to
    /// its device. Must be called once before uploadStaticData() / submit().
    /// Returns false if any device fails to initialize.
    bool initialize(size_t n_faces, const RadiosityOpenCLConfig& solver_cfg);

    /// Replicate the CSR view-factor matrix, face properties and the
    /// view-factor row normalization to every initialized device. Returns
    /// false if any upload fails.
    ///
    /// @param row_norm Per-face 1/max(1, sum_j F_ij), applied inside each
    ///        device's Jacobi iteration. Pass an empty vector to leave every
    ///        device at the 1.0 identity seed (legacy arithmetic).
    /// @param emissivity_vis Visible-band absorptivity 1 - A_i. Pass empty (the
    ///        default) for a gray run; passing it enables the FUSED two-band
    ///        solve on every device, which shares one CSR traversal between the
    ///        bands. Like row_norm this is operator-level state: a device that
    ///        missed it would run a different route from its peers, so a partial
    ///        upload fails the whole pool rather than producing a mixed fleet.
    /// @param reflectivity_vis Visible-band reflectivity A_i (paired with the above).
    ///
    /// @param csr_values float32 -- the device's own CSR value format. Passed
    ///        by reference to every device's uploadViewFactors(), so a caller
    ///        that already holds float32 (the PTM_HMATRIX_VF_REPLACE CSR) pays
    ///        no host copy of the matrix at all, on any device.
    bool uploadStaticData(
        const std::vector<float>& csr_values,
        const std::vector<int>& csr_col_indices,
        const std::vector<int64_t>& csr_row_ptrs,
        const std::vector<float>& emissivity,
        const std::vector<float>& reflectivity,
        const std::vector<float>& row_norm = {},
        const std::vector<float>& emissivity_vis = {},
        const std::vector<float>& reflectivity_vis = {});

    /// fp64 adapter for callers holding a ViewFactorMatrix. Each device narrows
    /// the matrix itself (RadiosityOpenCLSolver's fp64 overload) rather than the
    /// pool staging one float32 copy for the whole fleet, so the host transient
    /// stays at one device's worth instead of the whole matrix -- unchanged from
    /// before this overload split existed.
    bool uploadStaticData(
        const std::vector<double>& csr_values,
        const std::vector<int>& csr_col_indices,
        const std::vector<int64_t>& csr_row_ptrs,
        const std::vector<float>& emissivity,
        const std::vector<float>& reflectivity,
        const std::vector<float>& row_norm = {},
        const std::vector<float>& emissivity_vis = {},
        const std::vector<float>& reflectivity_vis = {});

    /// Number of devices the pool dispatches across.
    size_t numDevices() const noexcept { return slots_.size(); }

    /// Device handle for a given pool slot (for printing which physical GPU
    /// did the work).
    const OpenCLDeviceInfo& deviceAt(size_t i) const {
        return slots_.at(i)->handle;
    }

    /**
     * @brief Submit a batch of work to the next-idle device.
     *
     * The work callable is `bool(RadiosityOpenCLSolver&, size_t device_idx)`.
     * The pool guarantees:
     *   - `solver` is the persistent solver for one specific device,
     *     already pre-loaded with static data.
     *   - No two batches run on the *same* device concurrently.
     *   - Batches on *different* devices run in parallel.
     *   - Optional inter_batch_pause_ms sleep is honored between
     *     consecutive batches on the same device.
     *
     * @return future<bool> — true if the batch ran cleanly.
     */
    std::future<bool> submit(BatchCallback work);

private:
    struct DeviceSlot {
        OpenCLDeviceInfo handle;
        std::unique_ptr<RadiosityOpenCLSolver> solver;
        std::mutex       mtx;        // serializes batches on this device
        std::chrono::steady_clock::time_point last_finish_{};
        bool             healthy = true;
    };

    /// Round-robin index of the next device to assign. Wrap with mtx_.
    size_t pickNextDeviceIdx();

    /// Shared body of both uploadStaticData() overloads. `upload_csr` is the
    /// only part that differs between them (which uploadViewFactors overload to
    /// call); everything else -- the per-device loop, the fail-loud handling and
    /// the property/row-norm/visible-band uploads -- is one implementation, so
    /// the two entry points cannot drift apart.
    bool uploadStaticDataCommon(
        const std::function<void(RadiosityOpenCLSolver&)>& upload_csr,
        const std::vector<float>& emissivity,
        const std::vector<float>& reflectivity,
        const std::vector<float>& row_norm,
        const std::vector<float>& emissivity_vis,
        const std::vector<float>& reflectivity_vis);

    RadiosityGPUPoolConfig                    cfg_;
    std::vector<std::unique_ptr<DeviceSlot>>  slots_;
    std::mutex                                mtx_;          // protects next_idx_
    size_t                                    next_idx_ = 0;
    bool                                      initialized_ = false;
    bool                                      static_data_uploaded_ = false;
};

}  // namespace opencl
}  // namespace gpu

#endif  // PTM_GPU_OPENCL
