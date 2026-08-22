#ifdef PTM_GPU_OPENCL

#include "radiosity_gpu_pool.h"

#include <iostream>

namespace gpu {
namespace opencl {

RadiosityGPUDevicePool::RadiosityGPUDevicePool(RadiosityGPUPoolConfig cfg)
    : cfg_(std::move(cfg))
{
    slots_.reserve(cfg_.devices.size());
    for (const auto& dev : cfg_.devices) {
        auto slot = std::make_unique<DeviceSlot>();
        slot->handle = dev;
        slots_.push_back(std::move(slot));
    }
    if (cfg_.verbose && !slots_.empty()) {
        std::cout << "[Radiosity Pool] Pool created with " << slots_.size()
                  << " device(s):" << std::endl;
        for (size_t i = 0; i < slots_.size(); ++i) {
            std::cout << "  [" << i << "] " << slots_[i]->handle.name
                      << " (" << (slots_[i]->handle.global_mem_size / (1024 * 1024))
                      << " MB)" << std::endl;
        }
    }
}

RadiosityGPUDevicePool::~RadiosityGPUDevicePool() = default;

bool RadiosityGPUDevicePool::initialize(size_t n_faces,
                                         const RadiosityOpenCLConfig& solver_cfg) {
    if (initialized_) return true;
    if (slots_.empty()) {
        std::cerr << "[Radiosity Pool] initialize() called on empty pool\n";
        return false;
    }

    for (size_t i = 0; i < slots_.size(); ++i) {
        DeviceSlot* slot = slots_[i].get();
        slot->solver = std::make_unique<RadiosityOpenCLSolver>(n_faces, solver_cfg);
        if (!slot->solver->initialize(slot->handle)) {
            std::cerr << "[Radiosity Pool] Solver init failed on device " << i
                      << " (" << slot->handle.name << ")\n";
            slot->healthy = false;
            slot->solver.reset();
            return false;
        }
        if (cfg_.verbose) {
            std::cout << "[Radiosity Pool] Device " << i << " ready: "
                      << slot->handle.name << "\n";
        }
    }

    initialized_ = true;
    return true;
}

bool RadiosityGPUDevicePool::uploadStaticData(
    const std::vector<float>& csr_values,
    const std::vector<int>& csr_col_indices,
    const std::vector<int64_t>& csr_row_ptrs,
    const std::vector<float>& emissivity,
    const std::vector<float>& reflectivity,
    const std::vector<float>& row_norm,
    const std::vector<float>& emissivity_vis,
    const std::vector<float>& reflectivity_vis) {

    return uploadStaticDataCommon(
        [&](RadiosityOpenCLSolver& s) {
            s.uploadViewFactors(csr_values, csr_col_indices, csr_row_ptrs);
        },
        emissivity, reflectivity, row_norm, emissivity_vis, reflectivity_vis);
}

bool RadiosityGPUDevicePool::uploadStaticData(
    const std::vector<double>& csr_values,
    const std::vector<int>& csr_col_indices,
    const std::vector<int64_t>& csr_row_ptrs,
    const std::vector<float>& emissivity,
    const std::vector<float>& reflectivity,
    const std::vector<float>& row_norm,
    const std::vector<float>& emissivity_vis,
    const std::vector<float>& reflectivity_vis) {

    return uploadStaticDataCommon(
        [&](RadiosityOpenCLSolver& s) {
            s.uploadViewFactors(csr_values, csr_col_indices, csr_row_ptrs);
        },
        emissivity, reflectivity, row_norm, emissivity_vis, reflectivity_vis);
}

bool RadiosityGPUDevicePool::uploadStaticDataCommon(
    const std::function<void(RadiosityOpenCLSolver&)>& upload_csr,
    const std::vector<float>& emissivity,
    const std::vector<float>& reflectivity,
    const std::vector<float>& row_norm,
    const std::vector<float>& emissivity_vis,
    const std::vector<float>& reflectivity_vis) {

    if (!initialized_) {
        std::cerr << "[Radiosity Pool] uploadStaticData() called before initialize()\n";
        return false;
    }

    for (size_t i = 0; i < slots_.size(); ++i) {
        DeviceSlot* slot = slots_[i].get();
        if (!slot->healthy || !slot->solver) continue;

        // uploadViewFactors can throw (chunk/VRAM fail-loud, PTM-25). Every
        // device holds the full CSR replica, so a failure on any device means
        // the pool cannot serve this matrix -- fail loud rather than run a
        // partially-loaded pool.
        try {
            upload_csr(*slot->solver);
            slot->solver->uploadFaceProperties(emissivity, reflectivity);
            // Row norm is part of the operator, not of the per-batch state: a
            // device that missed it would run rho*F while its peers run
            // rho*(norm.F), i.e. a different physics per GPU. Same fail-loud
            // treatment as the CSR.
            if (!row_norm.empty()) slot->solver->uploadRowNorm(row_norm);
            // Same reasoning as the row norm: the visible-band property pair
            // decides whether this device runs the fused or the unfused two-band
            // route. A device that missed it would silently take the slow route
            // while its peers took the fast one -- same answer, but a per-device
            // difference that nothing would report.
            if (!emissivity_vis.empty() || !reflectivity_vis.empty()) {
                slot->solver->uploadVisibleProperties(emissivity_vis, reflectivity_vis);
            }
        } catch (const std::exception& e) {
            std::cerr << "[Radiosity Pool] Static-data upload failed on device " << i
                      << " (" << slot->handle.name << "): " << e.what() << "\n";
            slot->healthy = false;
            return false;
        }

        if (cfg_.verbose) {
            std::cout << "[Radiosity Pool] Static data uploaded to device " << i
                      << " (" << slot->handle.name << ")\n";
        }
    }

    static_data_uploaded_ = true;
    return true;
}

size_t RadiosityGPUDevicePool::pickNextDeviceIdx() {
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

std::future<bool> RadiosityGPUDevicePool::submit(BatchCallback work) {
    if (slots_.empty() || !initialized_) {
        std::promise<bool> p;
        p.set_value(false);
        std::cerr << "[Radiosity Pool] submit() called before initialize() "
                     "or on empty pool\n";
        return p.get_future();
    }

    const size_t idx = pickNextDeviceIdx();
    DeviceSlot* slot = slots_[idx].get();
    const int   pause_ms = cfg_.inter_batch_pause_ms;
    const bool  verbose  = cfg_.verbose;

    return std::async(std::launch::async,
        [slot, idx, work = std::move(work), pause_ms, verbose]() -> bool {
            std::lock_guard<std::mutex> lk(slot->mtx);

            if (!slot->solver) {
                std::cerr << "[Radiosity Pool] Device " << idx
                          << " has no solver; skipping batch\n";
                return false;
            }

            // Honor inter-batch pause on the *same* device.
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

            bool ok = false;
            try {
                ok = work(*slot->solver, idx);
            } catch (const std::exception& e) {
                std::cerr << "[Radiosity Pool] Worker threw on device " << idx
                          << ": " << e.what() << "\n";
                ok = false;
            } catch (...) {
                std::cerr << "[Radiosity Pool] Worker threw unknown exception "
                          << "on device " << idx << "\n";
                ok = false;
            }

            slot->last_finish_ = std::chrono::steady_clock::now();

            if (!ok && verbose) {
                std::cerr << "[Radiosity Pool] Batch on device " << idx
                          << " (" << slot->handle.name << ") returned false\n";
            }
            return ok;
        });
}

}  // namespace opencl
}  // namespace gpu

#endif  // PTM_GPU_OPENCL
