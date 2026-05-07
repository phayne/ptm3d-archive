#pragma once

/**
 * @file cwbvh_builder.h
 * @brief Thin wrapper around tinybvh::BVH8_CWBVH for the PTM3D BVH ray tracer.
 *
 * Produces a Ylitie 2017 compressed wide BVH8 (80-byte nodes + 48-byte
 * triangles) ready to upload to GPU. We use only the CPU-side builder;
 * the OpenCL traversal kernel is custom (see cwbvh_kernels.cl).
 *
 * Layout of `nodesData()` (per node, 80 bytes = 5×16):
 *   node[0]: parent AABB origin (3× fp32) + (ex, ey, ez, imask) packed in .w
 *   node[1]: child_base_idx (uint32 in .x) + tri_base_idx (uint32 in .y)
 *            + 8 child meta bytes packed across .z and .w
 *   node[2..4]: 48 bytes = 6 × 8-byte slots, one per (qlox, qloy, qloz,
 *               qhix, qhiy, qhiz). Slot i holds child i's quantized
 *               byte for that axis-min/max.
 *
 * Layout of `trisData()` (default tinybvh layout, 48 bytes = 3×16 per tri):
 *   tri[0] = v2 - v0  (Möller-Trumbore e2)
 *   tri[1] = v1 - v0  (Möller-Trumbore e1)
 *   tri[2] = v0 in .xyz, original triangle index in .w (uint32 as float bits)
 */

#ifdef PTM_GPU_OPENCL

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace gpu {
namespace opencl {

class CWBVHBuilderImpl;  // pimpl: keeps tinybvh out of the public header

class CWBVHBuilder {
public:
    CWBVHBuilder();
    ~CWBVHBuilder();

    CWBVHBuilder(const CWBVHBuilder&) = delete;
    CWBVHBuilder& operator=(const CWBVHBuilder&) = delete;

    /// Build from float vertices + uint32 triangle indices.
    /// The caller's arrays remain unchanged; tinybvh copies what it needs.
    /// Returns true on success.
    bool build(const std::vector<std::array<float, 3>>& vertices,
               const std::vector<std::array<uint32_t, 3>>& indices);

    /// Build from double vertices + signed-int indices (convenience overload
    /// matching the legacy BVHBuilder API).
    bool build(const std::vector<std::array<double, 3>>& vertices,
               const std::vector<std::array<int, 3>>& indices);

    /// Pointer to the packed CWBVH nodes (80 B/node, layout above).
    const float* nodesData() const noexcept;
    size_t       nodesBytes() const noexcept;
    size_t       numNodes() const noexcept;  // = nodesBytes() / 80

    /// Pointer to the packed triangle data (48 B/tri, layout above).
    const float* trisData() const noexcept;
    size_t       trisBytes() const noexcept;
    size_t       numTriangles() const noexcept;

    /// Disk cache via tinybvh's BVH8_CWBVH::Save/Load. The cache file is
    /// versioned by tinybvh; mismatched versions or expectedTris cause
    /// load() to return false (rebuild from scratch).
    bool save(const std::string& filename) const;
    bool load(const std::string& filename, size_t expected_triangles);

    bool isEmpty() const noexcept;

private:
    std::unique_ptr<CWBVHBuilderImpl> impl_;
};

}  // namespace opencl
}  // namespace gpu

#endif  // PTM_GPU_OPENCL
