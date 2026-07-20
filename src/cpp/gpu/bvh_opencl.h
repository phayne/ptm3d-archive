#pragma once

/**
 * @file bvh_opencl.h
 * @brief OpenCL-accelerated BVH ray tracer for illumination model
 *
 * Provides GPU-accelerated shadow testing and visibility queries using
 * a custom BVH built with Surface Area Heuristic (SAH).
 * Falls back to CPU Embree when OpenCL is unavailable.
 */

#ifdef PTM_GPU_OPENCL

#include "opencl_config.h"
#include <vector>
#include <array>
#include <memory>
#include <string>
#include <cstdint>

namespace gpu {
namespace opencl {

/**
 * @brief BVH node for CPU construction
 *
 * Internal nodes have tri_count = 0 and valid left/right children.
 * Leaf nodes have tri_count > 0 and store triangle indices.
 */
struct BVHNode {
    std::array<float, 3> min_bound;
    std::array<float, 3> max_bound;
    uint32_t left_child;      // Index of left child (0 = invalid for leaves)
    uint32_t right_child;     // Index of right child (0 = invalid for leaves)
    uint32_t tri_start;       // First triangle index in tri_indices array (for leaves)
    uint32_t tri_count;       // Triangle count (0 = internal node, >0 = leaf)
};

/**
 * @brief BVH builder configuration
 */
struct BVHConfig {
    int max_triangles_per_leaf = 4;    // Max triangles before subdivision
    int max_depth = 32;                 // Maximum BVH depth
    bool use_sah = true;                // Use Surface Area Heuristic
    int sah_bins = 12;                  // Number of SAH bins
    float traversal_cost = 1.0f;        // Cost of BVH node traversal
    float intersection_cost = 1.0f;     // Cost of triangle intersection
    bool verbose = false;               // Print build statistics
};

/**
 * @brief CPU-side BVH builder using Surface Area Heuristic
 *
 * Builds a high-quality BVH that can be uploaded to GPU for ray tracing.
 * The BVH can be serialized to disk for reuse across sessions.
 */
class BVHBuilder {
public:
    BVHBuilder() = default;

    /**
     * @brief Build BVH from mesh vertices and indices
     *
     * @param vertices Vertex positions (float precision for GPU)
     * @param indices Triangle vertex indices
     * @param config Build configuration
     */
    void build(const std::vector<std::array<float, 3>>& vertices,
               const std::vector<std::array<uint32_t, 3>>& indices,
               const BVHConfig& config = BVHConfig());

    /**
     * @brief Build BVH from double-precision vertices (converts internally)
     */
    void build(const std::vector<std::array<double, 3>>& vertices,
               const std::vector<std::array<int, 3>>& indices,
               const BVHConfig& config = BVHConfig());

    // Accessors
    const std::vector<BVHNode>& nodes() const { return nodes_; }
    const std::vector<uint32_t>& triIndices() const { return tri_indices_; }
    size_t numNodes() const { return nodes_.size(); }
    size_t numTriangles() const { return num_triangles_; }
    bool isEmpty() const { return nodes_.empty(); }

    /**
     * @brief Write BVH to binary file for later reuse
     *
     * File format:
     * - Magic: "BVH1" (4 bytes)
     * - num_nodes (uint32)
     * - num_triangles (uint32)
     * - num_tri_indices (uint32)
     * - nodes data (BVHNode array)
     * - tri_indices data (uint32 array)
     */
    bool write(const std::string& filename) const;

    /**
     * @brief Read BVH from binary file
     * @return true if successful, false if file not found or corrupted
     */
    bool read(const std::string& filename);

private:
    std::vector<BVHNode> nodes_;
    std::vector<uint32_t> tri_indices_;  // Reordered triangle indices for leaves
    size_t num_triangles_ = 0;

    // Build helpers
    struct BuildPrimitive {
        std::array<float, 3> centroid;
        std::array<float, 3> min_bound;
        std::array<float, 3> max_bound;
        uint32_t index;
    };

    uint32_t buildRecursive(std::vector<BuildPrimitive>& primitives,
                            size_t start, size_t end,
                            int depth, const BVHConfig& config);

    // SAH evaluation
    struct SAHBucket {
        int count = 0;
        std::array<float, 3> min_bound = {{1e30f, 1e30f, 1e30f}};
        std::array<float, 3> max_bound = {{-1e30f, -1e30f, -1e30f}};
    };

    float evaluateSAH(const std::vector<BuildPrimitive>& primitives,
                      size_t start, size_t end,
                      int axis, float split_pos,
                      const std::array<float, 3>& node_min,
                      const std::array<float, 3>& node_max,
                      const BVHConfig& config);

    static float surfaceArea(const std::array<float, 3>& min_b,
                             const std::array<float, 3>& max_b);

    static constexpr uint32_t BVH_MAGIC = 0x31485642; // "BVH1"
};

/**
 * @brief GPU ray tracer using custom BVH
 *
 * Traces shadow rays on GPU using OpenCL. For each ray, determines
 * if it is blocked by any geometry (occluded) or reaches the light (visible).
 */
class BVHRayTracer {
public:
    BVHRayTracer();
    ~BVHRayTracer();

    // No copy
    BVHRayTracer(const BVHRayTracer&) = delete;
    BVHRayTracer& operator=(const BVHRayTracer&) = delete;

    /**
     * @brief Initialize OpenCL context and compile kernels (default first-available GPU)
     * @return true if initialization succeeded
     */
    bool initialize();

    /**
     * @brief Initialize OpenCL resources pinned to a specific device.
     *
     * Used by BVHGPUPool to route a tracer instance to one of several
     * enumerated GPUs. The single-device default initialize() is unchanged
     * so legacy runs stay byte-identical.
     */
    bool initialize(const OpenCLDeviceInfo& device_handle);

    /**
     * @brief Check if GPU ray tracer is available
     */
    bool isAvailable() const { return initialized_; }

    /**
     * @brief Upload mesh vertices and indices to GPU
     *
     * Call once per mesh. Converts to float and SoA format.
     */
    void uploadMesh(const std::vector<std::array<float, 3>>& vertices,
                    const std::vector<std::array<uint32_t, 3>>& indices);

    /**
     * @brief Upload mesh from double-precision arrays (converts internally)
     */
    void uploadMesh(const std::vector<std::array<double, 3>>& vertices,
                    const std::vector<std::array<int, 3>>& indices);

    /**
     * @brief Upload BVH to GPU
     *
     * Converts BVH nodes to Structure-of-Arrays (SoA) format for
     * efficient GPU memory access.
     */
    void uploadBVH(const BVHBuilder& bvh);

    /**
     * @brief Trace shadow rays to determine occlusion
     *
     * For each ray, returns 1 if blocked (occluded), 0 if visible.
     * This is optimized for shadow testing with early exit on first hit.
     *
     * @param ray_org_x,y,z Ray origin components (SoA format)
     * @param ray_dir_x,y,z Ray direction components (SoA format)
     * @param occluded_out Output: 1 if ray is blocked, 0 if visible
     */
    void traceShadowRays(const std::vector<float>& ray_org_x,
                         const std::vector<float>& ray_org_y,
                         const std::vector<float>& ray_org_z,
                         const std::vector<float>& ray_dir_x,
                         const std::vector<float>& ray_dir_y,
                         const std::vector<float>& ray_dir_z,
                         std::vector<uint8_t>& occluded_out);

    /**
     * @brief Trace shadow rays with per-ray tmax (for finite-distance visibility like view factors)
     *
     * Same as traceShadowRays but each ray terminates at ray_tmax[i] along its direction.
     * Useful for view-factor visibility where tmax = distance-to-target - epsilon.
     */
    void traceShadowRays(const std::vector<float>& ray_org_x,
                         const std::vector<float>& ray_org_y,
                         const std::vector<float>& ray_org_z,
                         const std::vector<float>& ray_dir_x,
                         const std::vector<float>& ray_dir_y,
                         const std::vector<float>& ray_dir_z,
                         const std::vector<float>& ray_tmax,
                         std::vector<uint8_t>& occluded_out);

    /**
     * @brief Convenience overload for AoS ray data
     */
    void traceShadowRays(const std::vector<std::array<double, 3>>& origins,
                         const std::vector<std::array<double, 3>>& directions,
                         std::vector<uint8_t>& occluded_out);

    /**
     * @brief Get OpenCL device information
     */
    const OpenCLDeviceInfo& getDeviceInfo() const;

    /**
     * @brief Get performance statistics
     */
    struct Stats {
        size_t rays_traced = 0;
        double total_time_ms = 0;
        double upload_time_ms = 0;
        double kernel_time_ms = 0;
    };
    const Stats& getStats() const { return stats_; }
    void resetStats() { stats_ = Stats(); }

private:
    void buildKernels();
    void createBuffers(size_t num_triangles, size_t max_rays);
    bool ensureRayBuffers(size_t num_rays);  // Returns false on error
    void release();

    std::unique_ptr<OpenCLContext> context_;
    cl_program program_ = nullptr;
    cl_kernel shadow_kernel_ = nullptr;

    // Mesh buffers (triangle vertices in SoA)
    cl_mem buf_tri_v0_x_ = nullptr;
    cl_mem buf_tri_v0_y_ = nullptr;
    cl_mem buf_tri_v0_z_ = nullptr;
    cl_mem buf_tri_v1_x_ = nullptr;
    cl_mem buf_tri_v1_y_ = nullptr;
    cl_mem buf_tri_v1_z_ = nullptr;
    cl_mem buf_tri_v2_x_ = nullptr;
    cl_mem buf_tri_v2_y_ = nullptr;
    cl_mem buf_tri_v2_z_ = nullptr;

    // BVH buffers (SoA layout)
    cl_mem buf_node_min_x_ = nullptr;
    cl_mem buf_node_min_y_ = nullptr;
    cl_mem buf_node_min_z_ = nullptr;
    cl_mem buf_node_max_x_ = nullptr;
    cl_mem buf_node_max_y_ = nullptr;
    cl_mem buf_node_max_z_ = nullptr;
    cl_mem buf_node_left_ = nullptr;
    cl_mem buf_node_right_ = nullptr;
    cl_mem buf_node_tri_start_ = nullptr;
    cl_mem buf_node_tri_count_ = nullptr;
    cl_mem buf_tri_indices_ = nullptr;

    // Ray buffers
    cl_mem buf_ray_org_x_ = nullptr;
    cl_mem buf_ray_org_y_ = nullptr;
    cl_mem buf_ray_org_z_ = nullptr;
    cl_mem buf_ray_dir_x_ = nullptr;
    cl_mem buf_ray_dir_y_ = nullptr;
    cl_mem buf_ray_dir_z_ = nullptr;
    cl_mem buf_ray_inv_dir_x_ = nullptr;
    cl_mem buf_ray_inv_dir_y_ = nullptr;
    cl_mem buf_ray_inv_dir_z_ = nullptr;
    cl_mem buf_ray_tmax_ = nullptr;
    cl_mem buf_occluded_ = nullptr;

    size_t num_nodes_ = 0;
    size_t num_triangles_ = 0;
    size_t max_rays_ = 0;
    size_t work_group_size_ = 256;
    bool initialized_ = false;
    bool mesh_uploaded_ = false;
    bool bvh_uploaded_ = false;

    Stats stats_;
};

/**
 * @brief Get the OpenCL kernel source code for BVH ray tracing
 *
 * The kernel is embedded as a string for easy distribution without
 * external .cl files.
 */
std::string getBVHKernelSource();

}  // namespace opencl
}  // namespace gpu

#endif  // PTM_GPU_OPENCL
