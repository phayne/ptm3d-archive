#ifdef PTM_GPU_OPENCL

#include "bvh_gpu_pool.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace gpu {
namespace opencl {

BVHGPUPool::BVHGPUPool(std::vector<OpenCLDeviceInfo> devices,
                       int inter_batch_pause_ms,
                       bool verbose)
    : inter_batch_pause_ms_(inter_batch_pause_ms),
      verbose_(verbose)
{
    slots_.reserve(devices.size());
    for (auto& dev : devices) {
        auto slot = std::make_unique<DeviceSlot>();
        slot->handle = std::move(dev);
        slots_.push_back(std::move(slot));
    }
}

BVHGPUPool::~BVHGPUPool() = default;

bool BVHGPUPool::initialize() {
    if (slots_.empty()) {
        std::cerr << "[BVH Pool] initialize() called on empty pool" << std::endl;
        return false;
    }

    size_t n_ok = 0;
    for (size_t i = 0; i < slots_.size(); ++i) {
        auto& slot = *slots_[i];
        if (!slot.healthy) continue;

        slot.tracer = std::make_unique<BVHRayTracer>();
        if (!slot.tracer->initialize(slot.handle)) {
            std::cerr << "[BVH Pool] Tracer init failed on device " << i
                      << " (" << slot.handle.name << "); dropping" << std::endl;
            slot.tracer.reset();
            slot.healthy = false;
            continue;
        }
        ++n_ok;
    }

    if (verbose_ && n_ok > 0) {
        std::cout << "[BVH Pool] Initialized " << n_ok << " device(s):" << std::endl;
        for (size_t i = 0; i < slots_.size(); ++i) {
            const auto& slot = *slots_[i];
            if (!slot.healthy) continue;
            std::cout << "  [" << i << "] " << slot.handle.name << " ("
                      << (slot.handle.global_mem_size / (1024 * 1024)) << " MB)"
                      << std::endl;
        }
    }

    return n_ok > 0;
}

bool BVHGPUPool::uploadMesh(const std::vector<std::array<double, 3>>& vertices,
                            const std::vector<std::array<int, 3>>& indices) {
    bool all_ok = true;
    for (size_t i = 0; i < slots_.size(); ++i) {
        auto& slot = *slots_[i];
        if (!slot.healthy || !slot.tracer) continue;
        try {
            slot.tracer->uploadMesh(vertices, indices);
        } catch (const std::exception& e) {
            std::cerr << "[BVH Pool] Mesh upload failed on device " << i
                      << " (" << slot.handle.name << "): " << e.what()
                      << "; dropping" << std::endl;
            slot.tracer.reset();
            slot.healthy = false;
            all_ok = false;
        }
    }
    return all_ok;
}

bool BVHGPUPool::uploadMesh(const std::vector<std::array<float, 3>>& vertices,
                            const std::vector<std::array<uint32_t, 3>>& indices) {
    bool all_ok = true;
    for (size_t i = 0; i < slots_.size(); ++i) {
        auto& slot = *slots_[i];
        if (!slot.healthy || !slot.tracer) continue;
        try {
            slot.tracer->uploadMesh(vertices, indices);
        } catch (const std::exception& e) {
            std::cerr << "[BVH Pool] Mesh upload failed on device " << i
                      << " (" << slot.handle.name << "): " << e.what()
                      << "; dropping" << std::endl;
            slot.tracer.reset();
            slot.healthy = false;
            all_ok = false;
        }
    }
    return all_ok;
}

bool BVHGPUPool::uploadBVH(const BVHBuilder& bvh) {
    bool all_ok = true;
    for (size_t i = 0; i < slots_.size(); ++i) {
        auto& slot = *slots_[i];
        if (!slot.healthy || !slot.tracer) continue;
        try {
            slot.tracer->uploadBVH(bvh);
        } catch (const std::exception& e) {
            std::cerr << "[BVH Pool] BVH upload failed on device " << i
                      << " (" << slot.handle.name << "): " << e.what()
                      << "; dropping" << std::endl;
            slot.tracer.reset();
            slot.healthy = false;
            all_ok = false;
        }
    }
    return all_ok;
}

bool BVHGPUPool::isReady() const noexcept {
    for (const auto& slot_ptr : slots_) {
        if (slot_ptr && slot_ptr->healthy && slot_ptr->tracer
            && slot_ptr->tracer->isAvailable()) {
            return true;
        }
    }
    return false;
}

size_t BVHGPUPool::numDevices() const noexcept {
    size_t n = 0;
    for (const auto& slot_ptr : slots_) {
        if (slot_ptr && slot_ptr->healthy) ++n;
    }
    return n;
}

const OpenCLDeviceInfo& BVHGPUPool::deviceAt(size_t i) const {
    if (i >= slots_.size()) {
        throw std::out_of_range("BVHGPUPool::deviceAt");
    }
    return slots_[i]->handle;
}

BVHRayTracer::Stats BVHGPUPool::getAggregatedStats() const {
    BVHRayTracer::Stats agg;
    for (const auto& slot_ptr : slots_) {
        if (!slot_ptr || !slot_ptr->tracer) continue;
        const auto& s = slot_ptr->tracer->getStats();
        agg.rays_traced   += s.rays_traced;
        agg.total_time_ms += s.total_time_ms;
        agg.upload_time_ms += s.upload_time_ms;
        agg.kernel_time_ms += s.kernel_time_ms;
    }
    return agg;
}

std::vector<size_t> BVHGPUPool::healthyIndices() const {
    std::vector<size_t> out;
    out.reserve(slots_.size());
    for (size_t i = 0; i < slots_.size(); ++i) {
        if (slots_[i] && slots_[i]->healthy && slots_[i]->tracer) {
            out.push_back(i);
        }
    }
    return out;
}

namespace {

// Static partitioning: split [0, n_rays) into k contiguous slices
// of size floor(n/k) or ceil(n/k). Returns slice_starts of length k+1.
inline std::vector<size_t> sliceStarts(size_t n_rays, size_t k) {
    std::vector<size_t> starts(k + 1, 0);
    if (k == 0 || n_rays == 0) {
        starts.assign(k + 1, 0);
        return starts;
    }
    size_t base = n_rays / k;
    size_t rem  = n_rays % k;
    size_t pos  = 0;
    for (size_t i = 0; i < k; ++i) {
        starts[i] = pos;
        pos += base + (i < rem ? 1 : 0);
    }
    starts[k] = n_rays;
    return starts;
}

inline void honourPause(BVHGPUPool*, // unused, kept for symmetry
                        std::chrono::steady_clock::time_point& last_finish,
                        int pause_ms) {
    if (pause_ms <= 0) return;
    if (last_finish.time_since_epoch().count() == 0) return;
    auto now = std::chrono::steady_clock::now();
    auto since = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - last_finish).count();
    if (since < pause_ms) {
        std::this_thread::sleep_for(
            std::chrono::milliseconds(pause_ms - since));
    }
}

}  // namespace

void BVHGPUPool::traceShadowRays(const std::vector<float>& ray_org_x,
                                 const std::vector<float>& ray_org_y,
                                 const std::vector<float>& ray_org_z,
                                 const std::vector<float>& ray_dir_x,
                                 const std::vector<float>& ray_dir_y,
                                 const std::vector<float>& ray_dir_z,
                                 std::vector<uint8_t>& occluded_out)
{
    const size_t n_rays = ray_org_x.size();
    occluded_out.assign(n_rays, 0);
    if (n_rays == 0) return;

    auto active = healthyIndices();
    if (active.empty()) {
        std::cerr << "[BVH Pool] traceShadowRays: no healthy devices" << std::endl;
        return;
    }

    const auto starts = sliceStarts(n_rays, active.size());

    // Each healthy device traces its own contiguous slice.
    auto worker = [&, this](size_t slot_idx, size_t s, size_t e) -> bool {
        auto& slot = *slots_[slot_idx];
        std::lock_guard<std::mutex> lk(slot.mtx);
        honourPause(this, slot.last_finish_, inter_batch_pause_ms_);

        // Per-slice copies (avoids changing BVHRayTracer's existing API).
        // Cost: O(n_rays / k) memcpy per device, dwarfed by kernel time.
        const size_t n = e - s;
        std::vector<float> ox(ray_org_x.begin() + s, ray_org_x.begin() + e);
        std::vector<float> oy(ray_org_y.begin() + s, ray_org_y.begin() + e);
        std::vector<float> oz(ray_org_z.begin() + s, ray_org_z.begin() + e);
        std::vector<float> dx(ray_dir_x.begin() + s, ray_dir_x.begin() + e);
        std::vector<float> dy(ray_dir_y.begin() + s, ray_dir_y.begin() + e);
        std::vector<float> dz(ray_dir_z.begin() + s, ray_dir_z.begin() + e);
        std::vector<uint8_t> occ(n, 0);

        try {
            slot.tracer->traceShadowRays(ox, oy, oz, dx, dy, dz, occ);
        } catch (const std::exception& ex) {
            std::cerr << "[BVH Pool] traceShadowRays failed on device "
                      << slot_idx << " (" << slot.handle.name << "): "
                      << ex.what() << std::endl;
            slot.last_finish_ = std::chrono::steady_clock::now();
            return false;
        }

        // Splice results back into the caller's contiguous output buffer.
        std::copy(occ.begin(), occ.end(), occluded_out.begin() + s);

        slot.last_finish_ = std::chrono::steady_clock::now();
        return true;
    };

    if (active.size() == 1) {
        // Single-device fast path: no async overhead.
        worker(active[0], starts[0], starts[1]);
        return;
    }

    std::vector<std::future<bool>> futures;
    futures.reserve(active.size());
    for (size_t k = 0; k < active.size(); ++k) {
        const size_t slot_idx = active[k];
        const size_t s = starts[k];
        const size_t e = starts[k + 1];
        if (s == e) continue;  // empty slice (rare; e.g. n_rays < k)
        futures.emplace_back(std::async(std::launch::async, worker, slot_idx, s, e));
    }
    for (auto& f : futures) (void)f.get();
}

void BVHGPUPool::traceShadowRays(const std::vector<float>& ray_org_x,
                                 const std::vector<float>& ray_org_y,
                                 const std::vector<float>& ray_org_z,
                                 const std::vector<float>& ray_dir_x,
                                 const std::vector<float>& ray_dir_y,
                                 const std::vector<float>& ray_dir_z,
                                 const std::vector<float>& ray_tmax,
                                 std::vector<uint8_t>& occluded_out)
{
    const size_t n_rays = ray_org_x.size();
    occluded_out.assign(n_rays, 0);
    if (n_rays == 0) return;

    auto active = healthyIndices();
    if (active.empty()) {
        std::cerr << "[BVH Pool] traceShadowRays(tmax): no healthy devices" << std::endl;
        return;
    }

    const auto starts = sliceStarts(n_rays, active.size());

    auto worker = [&, this](size_t slot_idx, size_t s, size_t e) -> bool {
        auto& slot = *slots_[slot_idx];
        std::lock_guard<std::mutex> lk(slot.mtx);
        honourPause(this, slot.last_finish_, inter_batch_pause_ms_);

        const size_t n = e - s;
        std::vector<float> ox(ray_org_x.begin() + s, ray_org_x.begin() + e);
        std::vector<float> oy(ray_org_y.begin() + s, ray_org_y.begin() + e);
        std::vector<float> oz(ray_org_z.begin() + s, ray_org_z.begin() + e);
        std::vector<float> dx(ray_dir_x.begin() + s, ray_dir_x.begin() + e);
        std::vector<float> dy(ray_dir_y.begin() + s, ray_dir_y.begin() + e);
        std::vector<float> dz(ray_dir_z.begin() + s, ray_dir_z.begin() + e);
        std::vector<float> tm(ray_tmax.begin()  + s, ray_tmax.begin()  + e);
        std::vector<uint8_t> occ(n, 0);

        try {
            slot.tracer->traceShadowRays(ox, oy, oz, dx, dy, dz, tm, occ);
        } catch (const std::exception& ex) {
            std::cerr << "[BVH Pool] traceShadowRays(tmax) failed on device "
                      << slot_idx << " (" << slot.handle.name << "): "
                      << ex.what() << std::endl;
            slot.last_finish_ = std::chrono::steady_clock::now();
            return false;
        }

        std::copy(occ.begin(), occ.end(), occluded_out.begin() + s);
        slot.last_finish_ = std::chrono::steady_clock::now();
        return true;
    };

    if (active.size() == 1) {
        worker(active[0], starts[0], starts[1]);
        return;
    }

    std::vector<std::future<bool>> futures;
    futures.reserve(active.size());
    for (size_t k = 0; k < active.size(); ++k) {
        const size_t slot_idx = active[k];
        const size_t s = starts[k];
        const size_t e = starts[k + 1];
        if (s == e) continue;
        futures.emplace_back(std::async(std::launch::async, worker, slot_idx, s, e));
    }
    for (auto& f : futures) (void)f.get();
}

void BVHGPUPool::traceShadowRays(const std::vector<std::array<double, 3>>& origins,
                                 const std::vector<std::array<double, 3>>& directions,
                                 std::vector<uint8_t>& occluded_out)
{
    const size_t n = origins.size();
    if (directions.size() != n) {
        std::cerr << "[BVH Pool] origins/directions size mismatch: "
                  << origins.size() << " vs " << directions.size() << std::endl;
        occluded_out.clear();
        return;
    }

    // Convert AoS double → SoA float once for the whole batch; the per-device
    // sub-slices reuse the converted floats inside the SoA traceShadowRays path.
    std::vector<float> ox(n), oy(n), oz(n), dx(n), dy(n), dz(n);
    for (size_t i = 0; i < n; ++i) {
        ox[i] = static_cast<float>(origins[i][0]);
        oy[i] = static_cast<float>(origins[i][1]);
        oz[i] = static_cast<float>(origins[i][2]);
        dx[i] = static_cast<float>(directions[i][0]);
        dy[i] = static_cast<float>(directions[i][1]);
        dz[i] = static_cast<float>(directions[i][2]);
    }
    traceShadowRays(ox, oy, oz, dx, dy, dz, occluded_out);
}

}  // namespace opencl
}  // namespace gpu

#endif  // PTM_GPU_OPENCL
