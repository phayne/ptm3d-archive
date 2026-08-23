#ifdef PTM_GPU_OPENCL

#include "cwbvh_builder.h"

// Single TU defining tinybvh implementations.
#define TINYBVH_IMPLEMENTATION
#include "tiny_bvh.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace gpu {
namespace opencl {

class CWBVHBuilderImpl {
public:
    tinybvh::BVH8_CWBVH bvh;
    std::vector<tinybvh::bvhvec4> verts_packed;  // (x, y, z, 0)
    std::vector<uint32_t>         indices_flat;  // 3 indices per triangle
    size_t                        num_triangles = 0;
};

CWBVHBuilder::CWBVHBuilder() : impl_(new CWBVHBuilderImpl()) {}
CWBVHBuilder::~CWBVHBuilder() = default;

bool CWBVHBuilder::build(const std::vector<std::array<float, 3>>& vertices,
                          const std::vector<std::array<uint32_t, 3>>& indices)
{
    if (indices.empty() || vertices.empty()) return false;

    // Pack vertices into bvhvec4 (x, y, z, 0).
    impl_->verts_packed.resize(vertices.size());
    for (size_t i = 0; i < vertices.size(); ++i) {
        impl_->verts_packed[i] = tinybvh::bvhvec4(
            vertices[i][0], vertices[i][1], vertices[i][2], 0.0f);
    }

    // Flatten triangle indices.
    impl_->indices_flat.resize(indices.size() * 3);
    for (size_t i = 0; i < indices.size(); ++i) {
        impl_->indices_flat[i * 3 + 0] = indices[i][0];
        impl_->indices_flat[i * 3 + 1] = indices[i][1];
        impl_->indices_flat[i * 3 + 2] = indices[i][2];
    }
    impl_->num_triangles = indices.size();

    try {
        impl_->bvh.Build(impl_->verts_packed.data(),
                         impl_->indices_flat.data(),
                         static_cast<uint32_t>(indices.size()));
    } catch (const std::exception& e) {
        std::cerr << "[CWBVHBuilder] tinybvh::Build threw: " << e.what()
                  << std::endl;
        return false;
    }

    if (impl_->bvh.bvh8Data == nullptr || impl_->bvh.usedBlocks == 0) {
        std::cerr << "[CWBVHBuilder] tinybvh::Build produced empty BVH"
                  << std::endl;
        return false;
    }
    return true;
}

bool CWBVHBuilder::build(const std::vector<std::array<double, 3>>& vertices,
                         const std::vector<std::array<int, 3>>& indices)
{
    std::vector<std::array<float, 3>> v_f(vertices.size());
    for (size_t i = 0; i < vertices.size(); ++i) {
        v_f[i] = {{ static_cast<float>(vertices[i][0]),
                    static_cast<float>(vertices[i][1]),
                    static_cast<float>(vertices[i][2]) }};
    }
    std::vector<std::array<uint32_t, 3>> i_u(indices.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        i_u[i] = {{ static_cast<uint32_t>(indices[i][0]),
                    static_cast<uint32_t>(indices[i][1]),
                    static_cast<uint32_t>(indices[i][2]) }};
    }
    return build(v_f, i_u);
}

const float* CWBVHBuilder::nodesData() const noexcept {
    return reinterpret_cast<const float*>(impl_->bvh.bvh8Data);
}

size_t CWBVHBuilder::nodesBytes() const noexcept {
    return static_cast<size_t>(impl_->bvh.usedBlocks) * 16;
}

size_t CWBVHBuilder::numNodes() const noexcept {
    // 5 bvhvec4 blocks per node = 80 bytes.
    return static_cast<size_t>(impl_->bvh.usedBlocks) / 5;
}

const float* CWBVHBuilder::trisData() const noexcept {
    return reinterpret_cast<const float*>(impl_->bvh.bvh8Tris);
}

size_t CWBVHBuilder::trisBytes() const noexcept {
    // Default tinybvh layout (no CWBVH_COMPRESSED_TRIS): 3 bvhvec4 = 48 B/tri.
    return impl_->num_triangles * 48;
}

size_t CWBVHBuilder::numTriangles() const noexcept {
    return impl_->num_triangles;
}

bool CWBVHBuilder::save(const std::string& filename) const {
    try {
        impl_->bvh.Save(filename.c_str());
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[CWBVHBuilder] Save failed: " << e.what() << std::endl;
        return false;
    }
}

bool CWBVHBuilder::load(const std::string& filename, size_t expected_triangles) {
    try {
        const bool ok = impl_->bvh.Load(filename.c_str(),
                                         static_cast<uint32_t>(expected_triangles));
        if (ok) {
            impl_->num_triangles = expected_triangles;
        }
        return ok;
    } catch (const std::exception& e) {
        std::cerr << "[CWBVHBuilder] Load failed: " << e.what() << std::endl;
        return false;
    }
}

bool CWBVHBuilder::isEmpty() const noexcept {
    return impl_->bvh.bvh8Data == nullptr || impl_->bvh.usedBlocks == 0;
}

}  // namespace opencl
}  // namespace gpu

#endif  // PTM_GPU_OPENCL
