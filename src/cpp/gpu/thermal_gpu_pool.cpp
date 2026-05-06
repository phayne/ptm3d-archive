#ifdef PTM_GPU_OPENCL

#include "thermal_gpu_pool.h"

#include <iostream>

namespace gpu {
namespace opencl {

ThermalGPUDevicePool::ThermalGPUDevicePool(ThermalGPUPoolConfig cfg)
    : cfg_(std::move(cfg))
{
    slots_.reserve(cfg_.devices.size());
    for (const auto& dev : cfg_.devices) {
        auto slot = std::make_unique<DeviceSlot>();
        slot->handle = dev;
        slots_.push_back(std::move(slot));
    }
    if (cfg_.verbose && !slots_.empty()) {
        std::cout << "[GPU Pool] Pool created with " << slots_.size()
                  << " device(s):" << std::endl;
        for (size_t i = 0; i < slots_.size(); ++i) {
            std::cout << "  [" << i << "] " << slots_[i]->handle.name
                      << " (" << (slots_[i]->handle.global_mem_size / (1024 * 1024))
                      << " MB)" << std::endl;
        }
    }
}

ThermalGPUDevicePool::~ThermalGPUDevicePool() = default;

size_t ThermalGPUDevicePool::pickNextDeviceIdx() {
    std::lock_guard<std::mutex> lk(mtx_);
    // Skip devices marked unhealthy. Caller already knows slots_ is non-empty.
    for (size_t tried = 0; tried < slots_.size(); ++tried) {
        size_t idx = (next_idx_ + tried) % slots_.size();
        if (slots_[idx]->healthy) {
            next_idx_ = (idx + 1) % slots_.size();
            return idx;
        }
    }
    // All unhealthy — caller will get a failed future from submit()'s work.
    next_idx_ = (next_idx_ + 1) % slots_.size();
    return next_idx_;
}

std::future<bool> ThermalGPUDevicePool::submit(
    size_t n_faces_in_batch,
    size_t n_layers,
    const ThermalOpenCLConfig& solver_config,
    std::function<bool(ThermalOpenCLSolver&, size_t)> work)
{
    if (slots_.empty()) {
        // No multi-device pool requested — caller should not have invoked
        // submit() in this case. Return an already-failed future to make
        // misuse loud.
        std::promise<bool> p;
        p.set_value(false);
        std::cerr << "[GPU Pool] submit() called on empty pool" << std::endl;
        return p.get_future();
    }

    const size_t idx = pickNextDeviceIdx();
    DeviceSlot* slot = slots_[idx].get();
    const int   pause_ms = cfg_.inter_batch_pause_ms;
    const bool  verbose  = cfg_.verbose;

    return std::async(std::launch::async,
        [slot, idx, n_faces_in_batch, n_layers, solver_config,
         work = std::move(work), pause_ms, verbose]() -> bool {
            std::lock_guard<std::mutex> lk(slot->mtx);

            // Honour inter-batch pause on the *same* device.
            if (pause_ms > 0) {
                auto now = std::chrono::steady_clock::now();
                auto since = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now - slot->last_finish_).count();
                if (slot->last_finish_.time_since_epoch().count() != 0
                    && since < pause_ms) {
                    std::this_thread::sleep_for(
                        std::chrono::milliseconds(pause_ms - since));
                }
            }

            ThermalOpenCLSolver solver(n_faces_in_batch, n_layers, solver_config);
            if (!solver.initialize(slot->handle)) {
                std::cerr << "[GPU Pool] Solver init failed on device "
                          << idx << " (" << slot->handle.name << ")\n";
                slot->healthy = false;
                slot->last_finish_ = std::chrono::steady_clock::now();
                return false;
            }

            bool ok = false;
            try {
                ok = work(solver, idx);
            } catch (const std::exception& e) {
                std::cerr << "[GPU Pool] Worker threw on device " << idx
                          << ": " << e.what() << "\n";
                ok = false;
            } catch (...) {
                std::cerr << "[GPU Pool] Worker threw unknown exception on device "
                          << idx << "\n";
                ok = false;
            }

            slot->last_finish_ = std::chrono::steady_clock::now();

            if (!ok) {
                // Don't permanently kill the device on a single failure — the
                // caller may still want to retry on it. Mark only init
                // failures as unhealthy (handled above).
                if (verbose) {
                    std::cerr << "[GPU Pool] Batch on device " << idx
                              << " (" << slot->handle.name << ") returned false\n";
                }
            }
            return ok;
        });
}

}  // namespace opencl
}  // namespace gpu

#endif  // PTM_GPU_OPENCL
