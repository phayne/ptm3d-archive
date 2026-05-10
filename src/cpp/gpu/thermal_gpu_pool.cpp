#ifdef PTM_GPU_OPENCL

#include "thermal_gpu_pool.h"

#include <iostream>
#include <stdexcept>

namespace gpu {
namespace opencl {

ThermalGPUDevicePool::ThermalGPUDevicePool(ThermalGPUPoolConfig cfg)
    : cfg_(std::move(cfg))
{
    if (cfg_.devices.empty()) {
        throw std::runtime_error(
            "ThermalGPUDevicePool: no devices supplied (cfg.devices is empty)");
    }
    if (cfg_.max_n_faces == 0 || cfg_.n_layers == 0) {
        throw std::runtime_error(
            "ThermalGPUDevicePool: max_n_faces and n_layers must be > 0");
    }

    slots_.reserve(cfg_.devices.size());
    for (const auto& dev : cfg_.devices) {
        auto slot = std::make_unique<DeviceSlot>();
        slot->handle = dev;
        slot->solver = std::make_unique<ThermalOpenCLSolver>(
            cfg_.max_n_faces, cfg_.n_layers, cfg_.solver_config);
        if (!slot->solver->initialize(slot->handle)) {
            throw std::runtime_error(
                "ThermalGPUDevicePool: solver init failed on device "
                + slot->handle.name);
        }
        slots_.push_back(std::move(slot));
    }

    if (cfg_.verbose) {
        std::cout << "[GPU Pool] Pool created with " << slots_.size()
                  << " device(s) (max_n_faces=" << cfg_.max_n_faces
                  << ", n_layers=" << cfg_.n_layers << "):" << std::endl;
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
    std::function<bool(ThermalOpenCLSolver&, size_t)> work)
{
    if (slots_.empty()) {
        std::promise<bool> p;
        p.set_value(false);
        std::cerr << "[GPU Pool] submit() called on empty pool" << std::endl;
        return p.get_future();
    }
    if (n_faces_in_batch > cfg_.max_n_faces) {
        std::promise<bool> p;
        p.set_value(false);
        std::cerr << "[GPU Pool] submit(): batch size " << n_faces_in_batch
                  << " exceeds pool capacity " << cfg_.max_n_faces << std::endl;
        return p.get_future();
    }

    const size_t idx = pickNextDeviceIdx();
    DeviceSlot* slot = slots_[idx].get();
    const int   pause_ms = cfg_.inter_batch_pause_ms;
    const bool  verbose  = cfg_.verbose;

    return std::async(std::launch::async,
        [slot, idx, n_faces_in_batch, work = std::move(work),
         pause_ms, verbose]() -> bool {
            std::lock_guard<std::mutex> lk(slot->mtx);

            // Honour inter-batch pause on the *same* device.
            if (pause_ms > 0
                && slot->last_finish_.time_since_epoch().count() != 0) {
                auto now = std::chrono::steady_clock::now();
                auto since = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now - slot->last_finish_).count();
                if (since < pause_ms) {
                    std::this_thread::sleep_for(
                        std::chrono::milliseconds(pause_ms - since));
                }
            }

            if (!slot->healthy || !slot->solver) {
                std::cerr << "[GPU Pool] Device " << idx << " ("
                          << slot->handle.name << ") is unhealthy; skipping batch\n";
                slot->last_finish_ = std::chrono::steady_clock::now();
                return false;
            }

            // The pool's persistent solver is sized for cfg.max_n_faces;
            // narrow it for this batch so kernel launches use the right
            // n_faces (stride and global work size).
            slot->solver->setActiveFaceCount(n_faces_in_batch);

            bool ok = false;
            try {
                ok = work(*slot->solver, idx);
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

            if (!ok && verbose) {
                std::cerr << "[GPU Pool] Batch on device " << idx
                          << " (" << slot->handle.name << ") returned false\n";
            }
            return ok;
        });
}

}  // namespace opencl
}  // namespace gpu

#endif  // PTM_GPU_OPENCL
