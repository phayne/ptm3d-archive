/**
 * @file bvh_opencl.cpp
 * @brief Implementation of OpenCL-accelerated BVH ray tracer
 */

#ifdef PTM_GPU_OPENCL

#include "bvh_opencl.h"
#include <fstream>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <iostream>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace gpu {
namespace opencl {

//=============================================================================
// Embedded OpenCL kernel source
//=============================================================================

std::string getBVHKernelSource() {
    return R"CL(
// Shadow ray kernel - returns 1 if blocked, 0 if visible
// Uses early exit on first hit for maximum performance
// Note: Inline functions removed to improve macOS AMD driver compatibility
__kernel void shadow_ray_kernel(
    // BVH nodes (SoA)
    __global const float* node_min_x,
    __global const float* node_min_y,
    __global const float* node_min_z,
    __global const float* node_max_x,
    __global const float* node_max_y,
    __global const float* node_max_z,
    __global const uint* node_left,
    __global const uint* node_right,
    __global const uint* node_tri_start,
    __global const uint* node_tri_count,
    // Triangle vertices (SoA)
    __global const float* tri_v0_x,
    __global const float* tri_v0_y,
    __global const float* tri_v0_z,
    __global const float* tri_v1_x,
    __global const float* tri_v1_y,
    __global const float* tri_v1_z,
    __global const float* tri_v2_x,
    __global const float* tri_v2_y,
    __global const float* tri_v2_z,
    // Triangle indices for leaf nodes
    __global const uint* tri_indices,
    // Rays (SoA)
    __global const float* ray_org_x,
    __global const float* ray_org_y,
    __global const float* ray_org_z,
    __global const float* ray_dir_x,
    __global const float* ray_dir_y,
    __global const float* ray_dir_z,
    __global const float* ray_inv_dir_x,
    __global const float* ray_inv_dir_y,
    __global const float* ray_inv_dir_z,
    // Per-ray max t along direction (1e10 = effectively infinite)
    __global const float* ray_tmax_in,
    // Output
    __global uchar* occluded,
    // Ray count
    int num_rays)
{
    int ray_id = get_global_id(0);
    if (ray_id >= num_rays) return;

    // Load ray data
    float org_x = ray_org_x[ray_id];
    float org_y = ray_org_y[ray_id];
    float org_z = ray_org_z[ray_id];
    float dir_x = ray_dir_x[ray_id];
    float dir_y = ray_dir_y[ray_id];
    float dir_z = ray_dir_z[ray_id];
    float inv_x = ray_inv_dir_x[ray_id];
    float inv_y = ray_inv_dir_y[ray_id];
    float inv_z = ray_inv_dir_z[ray_id];

    float ray_tmin = 0.001f;  // Small offset to avoid self-intersection
    float ray_tmax = ray_tmax_in[ray_id];

    // Stack-based BVH traversal (smaller stack for compatibility)
    uint stack[32];
    int stack_ptr = 0;
    stack[stack_ptr++] = 0;  // Start at root node

    while (stack_ptr > 0) {
        uint node_idx = stack[--stack_ptr];

        // === Inline ray-box intersection (slab method) ===
        float t1, t2;
        float box_tmin = ray_tmin;
        float box_tmax = ray_tmax;

        t1 = (node_min_x[node_idx] - org_x) * inv_x;
        t2 = (node_max_x[node_idx] - org_x) * inv_x;
        box_tmin = max(box_tmin, min(t1, t2));
        box_tmax = min(box_tmax, max(t1, t2));

        t1 = (node_min_y[node_idx] - org_y) * inv_y;
        t2 = (node_max_y[node_idx] - org_y) * inv_y;
        box_tmin = max(box_tmin, min(t1, t2));
        box_tmax = min(box_tmax, max(t1, t2));

        t1 = (node_min_z[node_idx] - org_z) * inv_z;
        t2 = (node_max_z[node_idx] - org_z) * inv_z;
        box_tmin = max(box_tmin, min(t1, t2));
        box_tmax = min(box_tmax, max(t1, t2));

        // Skip if box not hit
        if (box_tmin > box_tmax) {
            continue;
        }

        uint tri_count = node_tri_count[node_idx];

        if (tri_count > 0) {
            // Leaf node - test triangles
            uint tri_start = node_tri_start[node_idx];

            for (uint i = 0; i < tri_count; i++) {
                uint tri_idx = tri_indices[tri_start + i];

                // === Inline Moller-Trumbore ray-triangle intersection ===
                float v0x = tri_v0_x[tri_idx], v0y = tri_v0_y[tri_idx], v0z = tri_v0_z[tri_idx];
                float v1x = tri_v1_x[tri_idx], v1y = tri_v1_y[tri_idx], v1z = tri_v1_z[tri_idx];
                float v2x = tri_v2_x[tri_idx], v2y = tri_v2_y[tri_idx], v2z = tri_v2_z[tri_idx];

                // Edge vectors
                float e1x = v1x - v0x, e1y = v1y - v0y, e1z = v1z - v0z;
                float e2x = v2x - v0x, e2y = v2y - v0y, e2z = v2z - v0z;

                // h = cross(dir, e2)
                float hx = dir_y * e2z - dir_z * e2y;
                float hy = dir_z * e2x - dir_x * e2z;
                float hz = dir_x * e2y - dir_y * e2x;

                float a = e1x * hx + e1y * hy + e1z * hz;

                // Parallel check (skip if nearly parallel)
                if (fabs(a) < 1e-8f) continue;

                float f = 1.0f / a;
                float sx = org_x - v0x, sy = org_y - v0y, sz = org_z - v0z;
                float u = f * (sx * hx + sy * hy + sz * hz);

                if (u < 0.0f || u > 1.0f) continue;

                // q = cross(s, e1)
                float qx = sy * e1z - sz * e1y;
                float qy = sz * e1x - sx * e1z;
                float qz = sx * e1y - sy * e1x;

                float v = f * (dir_x * qx + dir_y * qy + dir_z * qz);

                if (v < 0.0f || u + v > 1.0f) continue;

                float hit_t = f * (e2x * qx + e2y * qy + e2z * qz);

                if (hit_t >= ray_tmin && hit_t <= ray_tmax) {
                    // Early exit - any hit blocks the ray
                    occluded[ray_id] = 1;
                    return;
                }
            }
        } else {
            // Internal node - intersect both children's AABBs and push in
            // front-to-back order (farther first; nearer popped next).
            // Shadow rays benefit ~30-40% from this on incoherent / oblique
            // sun directions because the early-out fires sooner.
            uint left_idx  = node_left[node_idx];
            uint right_idx = node_right[node_idx];

            // Left child slab test
            float lt1, lt2, l_tmin = ray_tmin, l_tmax = ray_tmax;
            lt1 = (node_min_x[left_idx] - org_x) * inv_x;
            lt2 = (node_max_x[left_idx] - org_x) * inv_x;
            l_tmin = max(l_tmin, min(lt1, lt2));
            l_tmax = min(l_tmax, max(lt1, lt2));
            lt1 = (node_min_y[left_idx] - org_y) * inv_y;
            lt2 = (node_max_y[left_idx] - org_y) * inv_y;
            l_tmin = max(l_tmin, min(lt1, lt2));
            l_tmax = min(l_tmax, max(lt1, lt2));
            lt1 = (node_min_z[left_idx] - org_z) * inv_z;
            lt2 = (node_max_z[left_idx] - org_z) * inv_z;
            l_tmin = max(l_tmin, min(lt1, lt2));
            l_tmax = min(l_tmax, max(lt1, lt2));
            bool l_hit = (l_tmin <= l_tmax) && (left_idx != 0);

            // Right child slab test
            float rt1, rt2, r_tmin = ray_tmin, r_tmax = ray_tmax;
            rt1 = (node_min_x[right_idx] - org_x) * inv_x;
            rt2 = (node_max_x[right_idx] - org_x) * inv_x;
            r_tmin = max(r_tmin, min(rt1, rt2));
            r_tmax = min(r_tmax, max(rt1, rt2));
            rt1 = (node_min_y[right_idx] - org_y) * inv_y;
            rt2 = (node_max_y[right_idx] - org_y) * inv_y;
            r_tmin = max(r_tmin, min(rt1, rt2));
            r_tmax = min(r_tmax, max(rt1, rt2));
            rt1 = (node_min_z[right_idx] - org_z) * inv_z;
            rt2 = (node_max_z[right_idx] - org_z) * inv_z;
            r_tmin = max(r_tmin, min(rt1, rt2));
            r_tmax = min(r_tmax, max(rt1, rt2));
            bool r_hit = (r_tmin <= r_tmax) && (right_idx != 0);

            if (stack_ptr < 30) {
                if (l_hit && r_hit) {
                    // Push farther first so nearer is popped next.
                    if (l_tmin < r_tmin) {
                        stack[stack_ptr++] = right_idx;
                        stack[stack_ptr++] = left_idx;
                    } else {
                        stack[stack_ptr++] = left_idx;
                        stack[stack_ptr++] = right_idx;
                    }
                } else if (l_hit) {
                    stack[stack_ptr++] = left_idx;
                } else if (r_hit) {
                    stack[stack_ptr++] = right_idx;
                }
            }
        }
    }

    // No hit - ray is visible
    occluded[ray_id] = 0;
}
)CL";
}

//=============================================================================
// BVHBuilder implementation
//=============================================================================

void BVHBuilder::build(const std::vector<std::array<float, 3>>& vertices,
                       const std::vector<std::array<uint32_t, 3>>& indices,
                       const BVHConfig& config) {
    nodes_.clear();
    tri_indices_.clear();
    num_triangles_ = indices.size();

    if (num_triangles_ == 0) {
        return;
    }

    // Build primitives with centroids and bounds
    std::vector<BuildPrimitive> primitives(num_triangles_);

    for (size_t i = 0; i < num_triangles_; ++i) {
        const auto& tri = indices[i];
        const auto& v0 = vertices[tri[0]];
        const auto& v1 = vertices[tri[1]];
        const auto& v2 = vertices[tri[2]];

        primitives[i].index = static_cast<uint32_t>(i);

        // Compute bounding box
        primitives[i].min_bound = {{
            std::min({v0[0], v1[0], v2[0]}),
            std::min({v0[1], v1[1], v2[1]}),
            std::min({v0[2], v1[2], v2[2]})
        }};
        primitives[i].max_bound = {{
            std::max({v0[0], v1[0], v2[0]}),
            std::max({v0[1], v1[1], v2[1]}),
            std::max({v0[2], v1[2], v2[2]})
        }};

        // Compute centroid
        primitives[i].centroid = {{
            (v0[0] + v1[0] + v2[0]) / 3.0f,
            (v0[1] + v1[1] + v2[1]) / 3.0f,
            (v0[2] + v1[2] + v2[2]) / 3.0f
        }};
    }

    // Reserve space for nodes (roughly 2N-1 for N triangles)
    nodes_.reserve(2 * num_triangles_);
    tri_indices_.reserve(num_triangles_);

    // Build recursively
    buildRecursive(primitives, 0, num_triangles_, 0, config);

    if (config.verbose) {
        std::cout << "BVH built: " << nodes_.size() << " nodes, "
                  << num_triangles_ << " triangles" << std::endl;
    }
}

void BVHBuilder::build(const std::vector<std::array<double, 3>>& vertices,
                       const std::vector<std::array<int, 3>>& indices,
                       const BVHConfig& config) {
    // Convert double to float
    std::vector<std::array<float, 3>> float_vertices(vertices.size());
    for (size_t i = 0; i < vertices.size(); ++i) {
        float_vertices[i] = {{
            static_cast<float>(vertices[i][0]),
            static_cast<float>(vertices[i][1]),
            static_cast<float>(vertices[i][2])
        }};
    }

    // Convert int to uint32_t
    std::vector<std::array<uint32_t, 3>> uint_indices(indices.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        uint_indices[i] = {{
            static_cast<uint32_t>(indices[i][0]),
            static_cast<uint32_t>(indices[i][1]),
            static_cast<uint32_t>(indices[i][2])
        }};
    }

    build(float_vertices, uint_indices, config);
}

uint32_t BVHBuilder::buildRecursive(std::vector<BuildPrimitive>& primitives,
                                     size_t start, size_t end,
                                     int depth, const BVHConfig& config) {
    uint32_t node_idx = static_cast<uint32_t>(nodes_.size());
    nodes_.emplace_back();
    BVHNode& node = nodes_.back();

    size_t count = end - start;

    // Compute bounds of all primitives in this node
    node.min_bound = {{1e30f, 1e30f, 1e30f}};
    node.max_bound = {{-1e30f, -1e30f, -1e30f}};

    for (size_t i = start; i < end; ++i) {
        for (int j = 0; j < 3; ++j) {
            node.min_bound[j] = std::min(node.min_bound[j], primitives[i].min_bound[j]);
            node.max_bound[j] = std::max(node.max_bound[j], primitives[i].max_bound[j]);
        }
    }

    // Make leaf if few primitives or max depth
    if (count <= static_cast<size_t>(config.max_triangles_per_leaf) ||
        depth >= config.max_depth) {
        // Leaf node
        node.tri_start = static_cast<uint32_t>(tri_indices_.size());
        node.tri_count = static_cast<uint32_t>(count);
        node.left_child = 0;
        node.right_child = 0;

        for (size_t i = start; i < end; ++i) {
            tri_indices_.push_back(primitives[i].index);
        }
        return node_idx;
    }

    // Find best split using SAH
    int best_axis = 0;
    float best_pos = 0;
    float best_cost = 1e30f;

    if (config.use_sah) {
        // Compute centroid bounds
        std::array<float, 3> centroid_min = {{1e30f, 1e30f, 1e30f}};
        std::array<float, 3> centroid_max = {{-1e30f, -1e30f, -1e30f}};

        for (size_t i = start; i < end; ++i) {
            for (int j = 0; j < 3; ++j) {
                centroid_min[j] = std::min(centroid_min[j], primitives[i].centroid[j]);
                centroid_max[j] = std::max(centroid_max[j], primitives[i].centroid[j]);
            }
        }

        // Try each axis
        for (int axis = 0; axis < 3; ++axis) {
            float extent = centroid_max[axis] - centroid_min[axis];
            if (extent < 1e-6f) continue;

            // Initialize bins
            std::vector<SAHBucket> buckets(config.sah_bins);

            // Fill bins
            for (size_t i = start; i < end; ++i) {
                float rel_pos = (primitives[i].centroid[axis] - centroid_min[axis]) / extent;
                int bucket_idx = std::min(static_cast<int>(rel_pos * config.sah_bins),
                                         config.sah_bins - 1);
                buckets[bucket_idx].count++;
                for (int j = 0; j < 3; ++j) {
                    buckets[bucket_idx].min_bound[j] = std::min(buckets[bucket_idx].min_bound[j],
                                                                primitives[i].min_bound[j]);
                    buckets[bucket_idx].max_bound[j] = std::max(buckets[bucket_idx].max_bound[j],
                                                                primitives[i].max_bound[j]);
                }
            }

            // Evaluate each split position
            for (int split = 1; split < config.sah_bins; ++split) {
                // Left side
                int left_count = 0;
                std::array<float, 3> left_min = {{1e30f, 1e30f, 1e30f}};
                std::array<float, 3> left_max = {{-1e30f, -1e30f, -1e30f}};

                for (int b = 0; b < split; ++b) {
                    left_count += buckets[b].count;
                    for (int j = 0; j < 3; ++j) {
                        left_min[j] = std::min(left_min[j], buckets[b].min_bound[j]);
                        left_max[j] = std::max(left_max[j], buckets[b].max_bound[j]);
                    }
                }

                // Right side
                int right_count = 0;
                std::array<float, 3> right_min = {{1e30f, 1e30f, 1e30f}};
                std::array<float, 3> right_max = {{-1e30f, -1e30f, -1e30f}};

                for (int b = split; b < config.sah_bins; ++b) {
                    right_count += buckets[b].count;
                    for (int j = 0; j < 3; ++j) {
                        right_min[j] = std::min(right_min[j], buckets[b].min_bound[j]);
                        right_max[j] = std::max(right_max[j], buckets[b].max_bound[j]);
                    }
                }

                if (left_count == 0 || right_count == 0) continue;

                // Compute SAH cost
                float parent_area = surfaceArea(node.min_bound, node.max_bound);
                float left_area = surfaceArea(left_min, left_max);
                float right_area = surfaceArea(right_min, right_max);

                float cost = config.traversal_cost +
                            config.intersection_cost * (
                                left_count * left_area / parent_area +
                                right_count * right_area / parent_area
                            );

                if (cost < best_cost) {
                    best_cost = cost;
                    best_axis = axis;
                    best_pos = centroid_min[axis] + extent * split / config.sah_bins;
                }
            }
        }
    }

    // Fall back to median split if SAH didn't find good split
    if (best_cost >= static_cast<float>(count) * config.intersection_cost) {
        // Find longest axis
        float max_extent = 0;
        for (int axis = 0; axis < 3; ++axis) {
            float extent = node.max_bound[axis] - node.min_bound[axis];
            if (extent > max_extent) {
                max_extent = extent;
                best_axis = axis;
            }
        }

        // Use median position
        std::nth_element(primitives.begin() + start,
                        primitives.begin() + start + count / 2,
                        primitives.begin() + end,
                        [best_axis](const BuildPrimitive& a, const BuildPrimitive& b) {
                            return a.centroid[best_axis] < b.centroid[best_axis];
                        });
        best_pos = primitives[start + count / 2].centroid[best_axis];
    }

    // Partition primitives
    auto mid = std::partition(primitives.begin() + start, primitives.begin() + end,
                              [best_axis, best_pos](const BuildPrimitive& p) {
                                  return p.centroid[best_axis] < best_pos;
                              });
    size_t mid_idx = mid - primitives.begin();

    // Handle degenerate cases
    if (mid_idx == start || mid_idx == end) {
        mid_idx = start + count / 2;
    }

    // Mark as internal node
    node.tri_count = 0;
    node.tri_start = 0;

    // Build children (update node reference after recursive calls since vector may reallocate)
    uint32_t left_idx = buildRecursive(primitives, start, mid_idx, depth + 1, config);
    uint32_t right_idx = buildRecursive(primitives, mid_idx, end, depth + 1, config);

    nodes_[node_idx].left_child = left_idx;
    nodes_[node_idx].right_child = right_idx;

    return node_idx;
}

float BVHBuilder::surfaceArea(const std::array<float, 3>& min_b,
                              const std::array<float, 3>& max_b) {
    float dx = max_b[0] - min_b[0];
    float dy = max_b[1] - min_b[1];
    float dz = max_b[2] - min_b[2];
    return 2.0f * (dx * dy + dy * dz + dz * dx);
}

bool BVHBuilder::write(const std::string& filename) const {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    // Write header
    file.write(reinterpret_cast<const char*>(&BVH_MAGIC), sizeof(BVH_MAGIC));

    uint32_t num_nodes = static_cast<uint32_t>(nodes_.size());
    uint32_t num_tris = static_cast<uint32_t>(num_triangles_);
    uint32_t num_indices = static_cast<uint32_t>(tri_indices_.size());

    file.write(reinterpret_cast<const char*>(&num_nodes), sizeof(num_nodes));
    file.write(reinterpret_cast<const char*>(&num_tris), sizeof(num_tris));
    file.write(reinterpret_cast<const char*>(&num_indices), sizeof(num_indices));

    // Write nodes
    file.write(reinterpret_cast<const char*>(nodes_.data()),
               nodes_.size() * sizeof(BVHNode));

    // Write triangle indices
    file.write(reinterpret_cast<const char*>(tri_indices_.data()),
               tri_indices_.size() * sizeof(uint32_t));

    return file.good();
}

bool BVHBuilder::read(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    // Read and verify magic
    uint32_t magic;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (magic != BVH_MAGIC) {
        return false;
    }

    uint32_t num_nodes, num_tris, num_indices;
    file.read(reinterpret_cast<char*>(&num_nodes), sizeof(num_nodes));
    file.read(reinterpret_cast<char*>(&num_tris), sizeof(num_tris));
    file.read(reinterpret_cast<char*>(&num_indices), sizeof(num_indices));

    // Read nodes
    nodes_.resize(num_nodes);
    file.read(reinterpret_cast<char*>(nodes_.data()),
              num_nodes * sizeof(BVHNode));

    // Read triangle indices
    tri_indices_.resize(num_indices);
    file.read(reinterpret_cast<char*>(tri_indices_.data()),
              num_indices * sizeof(uint32_t));

    num_triangles_ = num_tris;

    return file.good();
}

//=============================================================================
// BVHRayTracer implementation
//=============================================================================

BVHRayTracer::BVHRayTracer() = default;

BVHRayTracer::~BVHRayTracer() {
    release();
}

bool BVHRayTracer::initialize() {
    if (initialized_) return true;

    try {
        context_ = std::make_unique<OpenCLContext>();
        if (!context_->initialize(true)) {
            std::cerr << "Warning: Failed to initialize OpenCL for BVH ray tracer" << std::endl;
            return false;
        }

        buildKernels();

        // Get optimal work group size
        work_group_size_ = std::min(context_->deviceInfo().max_work_group_size, size_t(256));

        initialized_ = true;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error initializing BVH ray tracer: " << e.what() << std::endl;
        return false;
    }
}

bool BVHRayTracer::initialize(const OpenCLDeviceInfo& device_handle) {
    if (initialized_) return true;

    try {
        context_ = std::make_unique<OpenCLContext>();
        if (!context_->initialize(device_handle)) {
            std::cerr << "Warning: Failed to initialize OpenCL for BVH ray tracer on device "
                      << device_handle.name << std::endl;
            return false;
        }

        buildKernels();

        work_group_size_ = std::min(context_->deviceInfo().max_work_group_size, size_t(256));

        initialized_ = true;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error initializing BVH ray tracer on " << device_handle.name
                  << ": " << e.what() << std::endl;
        return false;
    }
}

void BVHRayTracer::buildKernels() {
    std::string source = getBVHKernelSource();

    // Build program manually (not using context_->buildProgram to avoid exception issues
    // with CL_CHECK macro across shared library boundaries on macOS)
    const char* src = source.c_str();
    size_t len = source.size();

    cl_int err;
    program_ = clCreateProgramWithSource(context_->context(), 1, &src, &len, &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error(std::string("clCreateProgramWithSource failed: ") +
                                 clErrorString(err));
    }

    // Use simpler build options to avoid driver issues on some platforms
    std::string options = "-cl-fast-relaxed-math";
    cl_device_id device = context_->device();
    err = clBuildProgram(program_, 1, &device, options.c_str(), nullptr, nullptr);
    if (err != CL_SUCCESS) {
        // Get build log
        size_t log_size;
        clGetProgramBuildInfo(program_, device, CL_PROGRAM_BUILD_LOG,
                             0, nullptr, &log_size);
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(program_, device, CL_PROGRAM_BUILD_LOG,
                             log_size, log.data(), nullptr);

        std::string msg = std::string("clBuildProgram failed: ") + clErrorString(err) +
                         "\nBuild log:\n" + log.data();
        clReleaseProgram(program_);
        program_ = nullptr;
        throw std::runtime_error(msg);
    }

    // Create kernel
    shadow_kernel_ = clCreateKernel(program_, "shadow_ray_kernel", &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error(std::string("Failed to create shadow_ray_kernel: ") +
                                 clErrorString(err));
    }
}

void BVHRayTracer::uploadMesh(const std::vector<std::array<float, 3>>& vertices,
                              const std::vector<std::array<uint32_t, 3>>& indices) {
    if (!initialized_) {
        std::cerr << "BVHRayTracer::uploadMesh: Not initialized" << std::endl;
        return;
    }

    num_triangles_ = indices.size();

    std::cerr << "[BVHRayTracer] Uploading mesh: " << num_triangles_ << " triangles, "
              << vertices.size() << " vertices" << std::endl;

    // Convert to SoA format
    std::vector<float> v0_x(num_triangles_), v0_y(num_triangles_), v0_z(num_triangles_);
    std::vector<float> v1_x(num_triangles_), v1_y(num_triangles_), v1_z(num_triangles_);
    std::vector<float> v2_x(num_triangles_), v2_y(num_triangles_), v2_z(num_triangles_);

    for (size_t i = 0; i < num_triangles_; ++i) {
        const auto& tri = indices[i];
        v0_x[i] = vertices[tri[0]][0];
        v0_y[i] = vertices[tri[0]][1];
        v0_z[i] = vertices[tri[0]][2];
        v1_x[i] = vertices[tri[1]][0];
        v1_y[i] = vertices[tri[1]][1];
        v1_z[i] = vertices[tri[1]][2];
        v2_x[i] = vertices[tri[2]][0];
        v2_y[i] = vertices[tri[2]][1];
        v2_z[i] = vertices[tri[2]][2];
    }

    std::cerr << "[BVHRayTracer] SoA conversion complete" << std::endl;

    // Create and upload buffers
    cl_context ctx = context_->context();
    cl_command_queue queue = context_->queue();
    cl_int err;

    size_t buf_size = num_triangles_ * sizeof(float);

    std::cerr << "[BVHRayTracer] Buffer size: " << (buf_size / (1024*1024.0)) << " MB per array" << std::endl;

    // Release existing buffers
    if (buf_tri_v0_x_) clReleaseMemObject(buf_tri_v0_x_);
    if (buf_tri_v0_y_) clReleaseMemObject(buf_tri_v0_y_);
    if (buf_tri_v0_z_) clReleaseMemObject(buf_tri_v0_z_);
    if (buf_tri_v1_x_) clReleaseMemObject(buf_tri_v1_x_);
    if (buf_tri_v1_y_) clReleaseMemObject(buf_tri_v1_y_);
    if (buf_tri_v1_z_) clReleaseMemObject(buf_tri_v1_z_);
    if (buf_tri_v2_x_) clReleaseMemObject(buf_tri_v2_x_);
    if (buf_tri_v2_y_) clReleaseMemObject(buf_tri_v2_y_);
    if (buf_tri_v2_z_) clReleaseMemObject(buf_tri_v2_z_);

    // Use non-throwing error checks (macOS shared library exception issue)
    #define CL_CHECK_RETURN(call, msg) \
        do { \
            cl_int _err = (call); \
            if (_err != CL_SUCCESS) { \
                std::cerr << "OpenCL error in " << msg << ": " << clErrorString(_err) << std::endl; \
                return; \
            } \
        } while (0)

    std::cerr << "[BVHRayTracer] Creating triangle buffers..." << std::endl;

    buf_tri_v0_x_ = clCreateBuffer(ctx, CL_MEM_READ_ONLY, buf_size, nullptr, &err);
    CL_CHECK_RETURN(err, "clCreateBuffer v0_x");
    buf_tri_v0_y_ = clCreateBuffer(ctx, CL_MEM_READ_ONLY, buf_size, nullptr, &err);
    CL_CHECK_RETURN(err, "clCreateBuffer v0_y");
    buf_tri_v0_z_ = clCreateBuffer(ctx, CL_MEM_READ_ONLY, buf_size, nullptr, &err);
    CL_CHECK_RETURN(err, "clCreateBuffer v0_z");
    buf_tri_v1_x_ = clCreateBuffer(ctx, CL_MEM_READ_ONLY, buf_size, nullptr, &err);
    CL_CHECK_RETURN(err, "clCreateBuffer v1_x");
    buf_tri_v1_y_ = clCreateBuffer(ctx, CL_MEM_READ_ONLY, buf_size, nullptr, &err);
    CL_CHECK_RETURN(err, "clCreateBuffer v1_y");
    buf_tri_v1_z_ = clCreateBuffer(ctx, CL_MEM_READ_ONLY, buf_size, nullptr, &err);
    CL_CHECK_RETURN(err, "clCreateBuffer v1_z");
    buf_tri_v2_x_ = clCreateBuffer(ctx, CL_MEM_READ_ONLY, buf_size, nullptr, &err);
    CL_CHECK_RETURN(err, "clCreateBuffer v2_x");
    buf_tri_v2_y_ = clCreateBuffer(ctx, CL_MEM_READ_ONLY, buf_size, nullptr, &err);
    CL_CHECK_RETURN(err, "clCreateBuffer v2_y");
    buf_tri_v2_z_ = clCreateBuffer(ctx, CL_MEM_READ_ONLY, buf_size, nullptr, &err);
    CL_CHECK_RETURN(err, "clCreateBuffer v2_z");

    std::cerr << "[BVHRayTracer] Uploading triangle data..." << std::endl;

    CL_CHECK_RETURN(clEnqueueWriteBuffer(queue, buf_tri_v0_x_, CL_FALSE, 0, buf_size, v0_x.data(), 0, nullptr, nullptr), "write v0_x");
    CL_CHECK_RETURN(clEnqueueWriteBuffer(queue, buf_tri_v0_y_, CL_FALSE, 0, buf_size, v0_y.data(), 0, nullptr, nullptr), "write v0_y");
    CL_CHECK_RETURN(clEnqueueWriteBuffer(queue, buf_tri_v0_z_, CL_FALSE, 0, buf_size, v0_z.data(), 0, nullptr, nullptr), "write v0_z");
    CL_CHECK_RETURN(clEnqueueWriteBuffer(queue, buf_tri_v1_x_, CL_FALSE, 0, buf_size, v1_x.data(), 0, nullptr, nullptr), "write v1_x");
    CL_CHECK_RETURN(clEnqueueWriteBuffer(queue, buf_tri_v1_y_, CL_FALSE, 0, buf_size, v1_y.data(), 0, nullptr, nullptr), "write v1_y");
    CL_CHECK_RETURN(clEnqueueWriteBuffer(queue, buf_tri_v1_z_, CL_FALSE, 0, buf_size, v1_z.data(), 0, nullptr, nullptr), "write v1_z");
    CL_CHECK_RETURN(clEnqueueWriteBuffer(queue, buf_tri_v2_x_, CL_FALSE, 0, buf_size, v2_x.data(), 0, nullptr, nullptr), "write v2_x");
    CL_CHECK_RETURN(clEnqueueWriteBuffer(queue, buf_tri_v2_y_, CL_FALSE, 0, buf_size, v2_y.data(), 0, nullptr, nullptr), "write v2_y");
    CL_CHECK_RETURN(clEnqueueWriteBuffer(queue, buf_tri_v2_z_, CL_FALSE, 0, buf_size, v2_z.data(), 0, nullptr, nullptr), "write v2_z");

    std::cerr << "[BVHRayTracer] Finishing queue..." << std::endl;

    err = clFinish(queue);
    CL_CHECK_RETURN(err, "clFinish");

    mesh_uploaded_ = true;
    std::cerr << "[BVHRayTracer] Mesh upload complete" << std::endl;

    #undef CL_CHECK_RETURN
}

void BVHRayTracer::uploadMesh(const std::vector<std::array<double, 3>>& vertices,
                              const std::vector<std::array<int, 3>>& indices) {
    // Convert to float
    std::vector<std::array<float, 3>> float_vertices(vertices.size());
    for (size_t i = 0; i < vertices.size(); ++i) {
        float_vertices[i] = {{
            static_cast<float>(vertices[i][0]),
            static_cast<float>(vertices[i][1]),
            static_cast<float>(vertices[i][2])
        }};
    }

    // Convert to uint32_t
    std::vector<std::array<uint32_t, 3>> uint_indices(indices.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        uint_indices[i] = {{
            static_cast<uint32_t>(indices[i][0]),
            static_cast<uint32_t>(indices[i][1]),
            static_cast<uint32_t>(indices[i][2])
        }};
    }

    uploadMesh(float_vertices, uint_indices);
}

void BVHRayTracer::uploadBVH(const BVHBuilder& bvh) {
    if (!initialized_) {
        throw std::runtime_error("BVHRayTracer not initialized");
    }

    const auto& nodes = bvh.nodes();
    const auto& tri_indices = bvh.triIndices();
    num_nodes_ = nodes.size();

    if (num_nodes_ == 0) {
        throw std::runtime_error("Cannot upload empty BVH");
    }

    // Convert to SoA format
    std::vector<float> min_x(num_nodes_), min_y(num_nodes_), min_z(num_nodes_);
    std::vector<float> max_x(num_nodes_), max_y(num_nodes_), max_z(num_nodes_);
    std::vector<uint32_t> left(num_nodes_), right(num_nodes_);
    std::vector<uint32_t> tri_start(num_nodes_), tri_count(num_nodes_);

    for (size_t i = 0; i < num_nodes_; ++i) {
        const auto& node = nodes[i];
        min_x[i] = node.min_bound[0];
        min_y[i] = node.min_bound[1];
        min_z[i] = node.min_bound[2];
        max_x[i] = node.max_bound[0];
        max_y[i] = node.max_bound[1];
        max_z[i] = node.max_bound[2];
        left[i] = node.left_child;
        right[i] = node.right_child;
        tri_start[i] = node.tri_start;
        tri_count[i] = node.tri_count;
    }

    cl_context ctx = context_->context();
    cl_command_queue queue = context_->queue();
    cl_int err;

    // Release existing BVH buffers
    if (buf_node_min_x_) clReleaseMemObject(buf_node_min_x_);
    if (buf_node_min_y_) clReleaseMemObject(buf_node_min_y_);
    if (buf_node_min_z_) clReleaseMemObject(buf_node_min_z_);
    if (buf_node_max_x_) clReleaseMemObject(buf_node_max_x_);
    if (buf_node_max_y_) clReleaseMemObject(buf_node_max_y_);
    if (buf_node_max_z_) clReleaseMemObject(buf_node_max_z_);
    if (buf_node_left_) clReleaseMemObject(buf_node_left_);
    if (buf_node_right_) clReleaseMemObject(buf_node_right_);
    if (buf_node_tri_start_) clReleaseMemObject(buf_node_tri_start_);
    if (buf_node_tri_count_) clReleaseMemObject(buf_node_tri_count_);
    if (buf_tri_indices_) clReleaseMemObject(buf_tri_indices_);

    size_t node_size_float = num_nodes_ * sizeof(float);
    size_t node_size_uint = num_nodes_ * sizeof(uint32_t);
    size_t tri_idx_size = tri_indices.size() * sizeof(uint32_t);

    std::cerr << "[BVHRayTracer] BVH buffer sizes: nodes=" << (node_size_float / (1024.0)) << " KB, "
              << "indices=" << (tri_idx_size / (1024.0)) << " KB" << std::endl;

    // Use non-throwing error checks (macOS shared library exception issue)
    #define CL_CHECK_RETURN(call, msg) \
        do { \
            cl_int _err = (call); \
            if (_err != CL_SUCCESS) { \
                std::cerr << "OpenCL error in " << msg << ": " << clErrorString(_err) << std::endl; \
                return; \
            } \
        } while (0)

    std::cerr << "[BVHRayTracer] Creating BVH buffers..." << std::endl;

    // Create buffers
    buf_node_min_x_ = clCreateBuffer(ctx, CL_MEM_READ_ONLY, node_size_float, nullptr, &err);
    CL_CHECK_RETURN(err, "clCreateBuffer node_min_x");
    buf_node_min_y_ = clCreateBuffer(ctx, CL_MEM_READ_ONLY, node_size_float, nullptr, &err);
    CL_CHECK_RETURN(err, "clCreateBuffer node_min_y");
    buf_node_min_z_ = clCreateBuffer(ctx, CL_MEM_READ_ONLY, node_size_float, nullptr, &err);
    CL_CHECK_RETURN(err, "clCreateBuffer node_min_z");
    buf_node_max_x_ = clCreateBuffer(ctx, CL_MEM_READ_ONLY, node_size_float, nullptr, &err);
    CL_CHECK_RETURN(err, "clCreateBuffer node_max_x");
    buf_node_max_y_ = clCreateBuffer(ctx, CL_MEM_READ_ONLY, node_size_float, nullptr, &err);
    CL_CHECK_RETURN(err, "clCreateBuffer node_max_y");
    buf_node_max_z_ = clCreateBuffer(ctx, CL_MEM_READ_ONLY, node_size_float, nullptr, &err);
    CL_CHECK_RETURN(err, "clCreateBuffer node_max_z");
    buf_node_left_ = clCreateBuffer(ctx, CL_MEM_READ_ONLY, node_size_uint, nullptr, &err);
    CL_CHECK_RETURN(err, "clCreateBuffer node_left");
    buf_node_right_ = clCreateBuffer(ctx, CL_MEM_READ_ONLY, node_size_uint, nullptr, &err);
    CL_CHECK_RETURN(err, "clCreateBuffer node_right");
    buf_node_tri_start_ = clCreateBuffer(ctx, CL_MEM_READ_ONLY, node_size_uint, nullptr, &err);
    CL_CHECK_RETURN(err, "clCreateBuffer node_tri_start");
    buf_node_tri_count_ = clCreateBuffer(ctx, CL_MEM_READ_ONLY, node_size_uint, nullptr, &err);
    CL_CHECK_RETURN(err, "clCreateBuffer node_tri_count");
    buf_tri_indices_ = clCreateBuffer(ctx, CL_MEM_READ_ONLY, tri_idx_size, nullptr, &err);
    CL_CHECK_RETURN(err, "clCreateBuffer tri_indices");

    std::cerr << "[BVHRayTracer] Uploading BVH data..." << std::endl;

    // Upload data
    CL_CHECK_RETURN(clEnqueueWriteBuffer(queue, buf_node_min_x_, CL_FALSE, 0, node_size_float, min_x.data(), 0, nullptr, nullptr), "write node_min_x");
    CL_CHECK_RETURN(clEnqueueWriteBuffer(queue, buf_node_min_y_, CL_FALSE, 0, node_size_float, min_y.data(), 0, nullptr, nullptr), "write node_min_y");
    CL_CHECK_RETURN(clEnqueueWriteBuffer(queue, buf_node_min_z_, CL_FALSE, 0, node_size_float, min_z.data(), 0, nullptr, nullptr), "write node_min_z");
    CL_CHECK_RETURN(clEnqueueWriteBuffer(queue, buf_node_max_x_, CL_FALSE, 0, node_size_float, max_x.data(), 0, nullptr, nullptr), "write node_max_x");
    CL_CHECK_RETURN(clEnqueueWriteBuffer(queue, buf_node_max_y_, CL_FALSE, 0, node_size_float, max_y.data(), 0, nullptr, nullptr), "write node_max_y");
    CL_CHECK_RETURN(clEnqueueWriteBuffer(queue, buf_node_max_z_, CL_FALSE, 0, node_size_float, max_z.data(), 0, nullptr, nullptr), "write node_max_z");
    CL_CHECK_RETURN(clEnqueueWriteBuffer(queue, buf_node_left_, CL_FALSE, 0, node_size_uint, left.data(), 0, nullptr, nullptr), "write node_left");
    CL_CHECK_RETURN(clEnqueueWriteBuffer(queue, buf_node_right_, CL_FALSE, 0, node_size_uint, right.data(), 0, nullptr, nullptr), "write node_right");
    CL_CHECK_RETURN(clEnqueueWriteBuffer(queue, buf_node_tri_start_, CL_FALSE, 0, node_size_uint, tri_start.data(), 0, nullptr, nullptr), "write node_tri_start");
    CL_CHECK_RETURN(clEnqueueWriteBuffer(queue, buf_node_tri_count_, CL_FALSE, 0, node_size_uint, tri_count.data(), 0, nullptr, nullptr), "write node_tri_count");
    CL_CHECK_RETURN(clEnqueueWriteBuffer(queue, buf_tri_indices_, CL_FALSE, 0, tri_idx_size, tri_indices.data(), 0, nullptr, nullptr), "write tri_indices");

    std::cerr << "[BVHRayTracer] Finishing BVH upload..." << std::endl;

    err = clFinish(queue);
    CL_CHECK_RETURN(err, "clFinish");

    bvh_uploaded_ = true;
    std::cerr << "[BVHRayTracer] BVH upload complete" << std::endl;

    #undef CL_CHECK_RETURN
}

bool BVHRayTracer::ensureRayBuffers(size_t num_rays) {
    if (num_rays <= max_rays_) return true;

    cl_context ctx = context_->context();
    cl_int err;

    // Release existing ray buffers
    if (buf_ray_org_x_) clReleaseMemObject(buf_ray_org_x_);
    if (buf_ray_org_y_) clReleaseMemObject(buf_ray_org_y_);
    if (buf_ray_org_z_) clReleaseMemObject(buf_ray_org_z_);
    if (buf_ray_dir_x_) clReleaseMemObject(buf_ray_dir_x_);
    if (buf_ray_dir_y_) clReleaseMemObject(buf_ray_dir_y_);
    if (buf_ray_dir_z_) clReleaseMemObject(buf_ray_dir_z_);
    if (buf_ray_inv_dir_x_) clReleaseMemObject(buf_ray_inv_dir_x_);
    if (buf_ray_inv_dir_y_) clReleaseMemObject(buf_ray_inv_dir_y_);
    if (buf_ray_inv_dir_z_) clReleaseMemObject(buf_ray_inv_dir_z_);
    if (buf_ray_tmax_) clReleaseMemObject(buf_ray_tmax_);
    if (buf_occluded_) clReleaseMemObject(buf_occluded_);

    // Allocate with some headroom
    max_rays_ = num_rays + num_rays / 4;

    size_t buf_size_float = max_rays_ * sizeof(float);
    size_t buf_size_uchar = max_rays_ * sizeof(uint8_t);

    // Use non-throwing error check macro
    #define CL_CHECK_FALSE(call, msg) \
        do { \
            cl_int _err = (call); \
            if (_err != CL_SUCCESS) { \
                std::cerr << "OpenCL error in " << msg << ": " << clErrorString(_err) << std::endl; \
                return false; \
            } \
        } while (0)

    buf_ray_org_x_ = clCreateBuffer(ctx, CL_MEM_READ_ONLY, buf_size_float, nullptr, &err);
    CL_CHECK_FALSE(err, "clCreateBuffer ray_org_x");
    buf_ray_org_y_ = clCreateBuffer(ctx, CL_MEM_READ_ONLY, buf_size_float, nullptr, &err);
    CL_CHECK_FALSE(err, "clCreateBuffer ray_org_y");
    buf_ray_org_z_ = clCreateBuffer(ctx, CL_MEM_READ_ONLY, buf_size_float, nullptr, &err);
    CL_CHECK_FALSE(err, "clCreateBuffer ray_org_z");
    buf_ray_dir_x_ = clCreateBuffer(ctx, CL_MEM_READ_ONLY, buf_size_float, nullptr, &err);
    CL_CHECK_FALSE(err, "clCreateBuffer ray_dir_x");
    buf_ray_dir_y_ = clCreateBuffer(ctx, CL_MEM_READ_ONLY, buf_size_float, nullptr, &err);
    CL_CHECK_FALSE(err, "clCreateBuffer ray_dir_y");
    buf_ray_dir_z_ = clCreateBuffer(ctx, CL_MEM_READ_ONLY, buf_size_float, nullptr, &err);
    CL_CHECK_FALSE(err, "clCreateBuffer ray_dir_z");
    buf_ray_inv_dir_x_ = clCreateBuffer(ctx, CL_MEM_READ_ONLY, buf_size_float, nullptr, &err);
    CL_CHECK_FALSE(err, "clCreateBuffer ray_inv_dir_x");
    buf_ray_inv_dir_y_ = clCreateBuffer(ctx, CL_MEM_READ_ONLY, buf_size_float, nullptr, &err);
    CL_CHECK_FALSE(err, "clCreateBuffer ray_inv_dir_y");
    buf_ray_inv_dir_z_ = clCreateBuffer(ctx, CL_MEM_READ_ONLY, buf_size_float, nullptr, &err);
    CL_CHECK_FALSE(err, "clCreateBuffer ray_inv_dir_z");
    buf_ray_tmax_ = clCreateBuffer(ctx, CL_MEM_READ_ONLY, buf_size_float, nullptr, &err);
    CL_CHECK_FALSE(err, "clCreateBuffer ray_tmax");
    buf_occluded_ = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, buf_size_uchar, nullptr, &err);
    CL_CHECK_FALSE(err, "clCreateBuffer occluded");

    #undef CL_CHECK_FALSE
    return true;
}

void BVHRayTracer::traceShadowRays(const std::vector<float>& ray_org_x,
                                   const std::vector<float>& ray_org_y,
                                   const std::vector<float>& ray_org_z,
                                   const std::vector<float>& ray_dir_x,
                                   const std::vector<float>& ray_dir_y,
                                   const std::vector<float>& ray_dir_z,
                                   std::vector<uint8_t>& occluded_out) {
    // Backward-compatible: infinite tmax (original shadow-to-sun semantics).
    std::vector<float> inf_tmax(ray_org_x.size(), 1e10f);
    traceShadowRays(ray_org_x, ray_org_y, ray_org_z,
                    ray_dir_x, ray_dir_y, ray_dir_z,
                    inf_tmax, occluded_out);
}

void BVHRayTracer::traceShadowRays(const std::vector<float>& ray_org_x,
                                   const std::vector<float>& ray_org_y,
                                   const std::vector<float>& ray_org_z,
                                   const std::vector<float>& ray_dir_x,
                                   const std::vector<float>& ray_dir_y,
                                   const std::vector<float>& ray_dir_z,
                                   const std::vector<float>& ray_tmax,
                                   std::vector<uint8_t>& occluded_out) {
    if (!initialized_ || !mesh_uploaded_ || !bvh_uploaded_) {
        std::cerr << "BVHRayTracer not ready (init=" << initialized_ << ", mesh=" << mesh_uploaded_
                  << ", bvh=" << bvh_uploaded_ << ")" << std::endl;
        // Return all as occluded as fallback
        occluded_out.assign(ray_org_x.size(), 1);
        return;
    }

    size_t num_rays = ray_org_x.size();
    if (num_rays == 0) {
        occluded_out.clear();
        return;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    // Ensure buffers are large enough
    if (!ensureRayBuffers(num_rays)) {
        std::cerr << "Failed to allocate ray buffers" << std::endl;
        occluded_out.assign(num_rays, 1);
        return;
    }

    // Compute inverse directions for slab intersection
    std::vector<float> inv_dir_x(num_rays), inv_dir_y(num_rays), inv_dir_z(num_rays);
    for (size_t i = 0; i < num_rays; ++i) {
        // Add small epsilon to avoid division by zero
        inv_dir_x[i] = 1.0f / (std::abs(ray_dir_x[i]) < 1e-8f ?
                               (ray_dir_x[i] >= 0 ? 1e-8f : -1e-8f) : ray_dir_x[i]);
        inv_dir_y[i] = 1.0f / (std::abs(ray_dir_y[i]) < 1e-8f ?
                               (ray_dir_y[i] >= 0 ? 1e-8f : -1e-8f) : ray_dir_y[i]);
        inv_dir_z[i] = 1.0f / (std::abs(ray_dir_z[i]) < 1e-8f ?
                               (ray_dir_z[i] >= 0 ? 1e-8f : -1e-8f) : ray_dir_z[i]);
    }

    cl_command_queue queue = context_->queue();
    size_t buf_size_float = num_rays * sizeof(float);

    // Use non-throwing error check macro
    #define CL_CHECK_RETURN(call, msg) \
        do { \
            cl_int _err = (call); \
            if (_err != CL_SUCCESS) { \
                std::cerr << "OpenCL error in " << msg << ": " << clErrorString(_err) << std::endl; \
                occluded_out.assign(num_rays, 1); \
                return; \
            } \
        } while (0)

    // Upload ray data
    auto upload_start = std::chrono::high_resolution_clock::now();

    CL_CHECK_RETURN(clEnqueueWriteBuffer(queue, buf_ray_org_x_, CL_FALSE, 0, buf_size_float, ray_org_x.data(), 0, nullptr, nullptr), "write ray_org_x");
    CL_CHECK_RETURN(clEnqueueWriteBuffer(queue, buf_ray_org_y_, CL_FALSE, 0, buf_size_float, ray_org_y.data(), 0, nullptr, nullptr), "write ray_org_y");
    CL_CHECK_RETURN(clEnqueueWriteBuffer(queue, buf_ray_org_z_, CL_FALSE, 0, buf_size_float, ray_org_z.data(), 0, nullptr, nullptr), "write ray_org_z");
    CL_CHECK_RETURN(clEnqueueWriteBuffer(queue, buf_ray_dir_x_, CL_FALSE, 0, buf_size_float, ray_dir_x.data(), 0, nullptr, nullptr), "write ray_dir_x");
    CL_CHECK_RETURN(clEnqueueWriteBuffer(queue, buf_ray_dir_y_, CL_FALSE, 0, buf_size_float, ray_dir_y.data(), 0, nullptr, nullptr), "write ray_dir_y");
    CL_CHECK_RETURN(clEnqueueWriteBuffer(queue, buf_ray_dir_z_, CL_FALSE, 0, buf_size_float, ray_dir_z.data(), 0, nullptr, nullptr), "write ray_dir_z");
    CL_CHECK_RETURN(clEnqueueWriteBuffer(queue, buf_ray_inv_dir_x_, CL_FALSE, 0, buf_size_float, inv_dir_x.data(), 0, nullptr, nullptr), "write ray_inv_dir_x");
    CL_CHECK_RETURN(clEnqueueWriteBuffer(queue, buf_ray_inv_dir_y_, CL_FALSE, 0, buf_size_float, inv_dir_y.data(), 0, nullptr, nullptr), "write ray_inv_dir_y");
    CL_CHECK_RETURN(clEnqueueWriteBuffer(queue, buf_ray_inv_dir_z_, CL_FALSE, 0, buf_size_float, inv_dir_z.data(), 0, nullptr, nullptr), "write ray_inv_dir_z");
    CL_CHECK_RETURN(clEnqueueWriteBuffer(queue, buf_ray_tmax_, CL_FALSE, 0, buf_size_float, ray_tmax.data(), 0, nullptr, nullptr), "write ray_tmax");

    cl_int err = clFinish(queue);
    CL_CHECK_RETURN(err, "clFinish after ray upload");
    auto upload_end = std::chrono::high_resolution_clock::now();

    // Set kernel arguments
    int arg_idx = 0;
    CL_CHECK_RETURN(clSetKernelArg(shadow_kernel_, arg_idx++, sizeof(cl_mem), &buf_node_min_x_), "setarg node_min_x");
    CL_CHECK_RETURN(clSetKernelArg(shadow_kernel_, arg_idx++, sizeof(cl_mem), &buf_node_min_y_), "setarg node_min_y");
    CL_CHECK_RETURN(clSetKernelArg(shadow_kernel_, arg_idx++, sizeof(cl_mem), &buf_node_min_z_), "setarg node_min_z");
    CL_CHECK_RETURN(clSetKernelArg(shadow_kernel_, arg_idx++, sizeof(cl_mem), &buf_node_max_x_), "setarg node_max_x");
    CL_CHECK_RETURN(clSetKernelArg(shadow_kernel_, arg_idx++, sizeof(cl_mem), &buf_node_max_y_), "setarg node_max_y");
    CL_CHECK_RETURN(clSetKernelArg(shadow_kernel_, arg_idx++, sizeof(cl_mem), &buf_node_max_z_), "setarg node_max_z");
    CL_CHECK_RETURN(clSetKernelArg(shadow_kernel_, arg_idx++, sizeof(cl_mem), &buf_node_left_), "setarg node_left");
    CL_CHECK_RETURN(clSetKernelArg(shadow_kernel_, arg_idx++, sizeof(cl_mem), &buf_node_right_), "setarg node_right");
    CL_CHECK_RETURN(clSetKernelArg(shadow_kernel_, arg_idx++, sizeof(cl_mem), &buf_node_tri_start_), "setarg node_tri_start");
    CL_CHECK_RETURN(clSetKernelArg(shadow_kernel_, arg_idx++, sizeof(cl_mem), &buf_node_tri_count_), "setarg node_tri_count");
    CL_CHECK_RETURN(clSetKernelArg(shadow_kernel_, arg_idx++, sizeof(cl_mem), &buf_tri_v0_x_), "setarg tri_v0_x");
    CL_CHECK_RETURN(clSetKernelArg(shadow_kernel_, arg_idx++, sizeof(cl_mem), &buf_tri_v0_y_), "setarg tri_v0_y");
    CL_CHECK_RETURN(clSetKernelArg(shadow_kernel_, arg_idx++, sizeof(cl_mem), &buf_tri_v0_z_), "setarg tri_v0_z");
    CL_CHECK_RETURN(clSetKernelArg(shadow_kernel_, arg_idx++, sizeof(cl_mem), &buf_tri_v1_x_), "setarg tri_v1_x");
    CL_CHECK_RETURN(clSetKernelArg(shadow_kernel_, arg_idx++, sizeof(cl_mem), &buf_tri_v1_y_), "setarg tri_v1_y");
    CL_CHECK_RETURN(clSetKernelArg(shadow_kernel_, arg_idx++, sizeof(cl_mem), &buf_tri_v1_z_), "setarg tri_v1_z");
    CL_CHECK_RETURN(clSetKernelArg(shadow_kernel_, arg_idx++, sizeof(cl_mem), &buf_tri_v2_x_), "setarg tri_v2_x");
    CL_CHECK_RETURN(clSetKernelArg(shadow_kernel_, arg_idx++, sizeof(cl_mem), &buf_tri_v2_y_), "setarg tri_v2_y");
    CL_CHECK_RETURN(clSetKernelArg(shadow_kernel_, arg_idx++, sizeof(cl_mem), &buf_tri_v2_z_), "setarg tri_v2_z");
    CL_CHECK_RETURN(clSetKernelArg(shadow_kernel_, arg_idx++, sizeof(cl_mem), &buf_tri_indices_), "setarg tri_indices");
    CL_CHECK_RETURN(clSetKernelArg(shadow_kernel_, arg_idx++, sizeof(cl_mem), &buf_ray_org_x_), "setarg ray_org_x");
    CL_CHECK_RETURN(clSetKernelArg(shadow_kernel_, arg_idx++, sizeof(cl_mem), &buf_ray_org_y_), "setarg ray_org_y");
    CL_CHECK_RETURN(clSetKernelArg(shadow_kernel_, arg_idx++, sizeof(cl_mem), &buf_ray_org_z_), "setarg ray_org_z");
    CL_CHECK_RETURN(clSetKernelArg(shadow_kernel_, arg_idx++, sizeof(cl_mem), &buf_ray_dir_x_), "setarg ray_dir_x");
    CL_CHECK_RETURN(clSetKernelArg(shadow_kernel_, arg_idx++, sizeof(cl_mem), &buf_ray_dir_y_), "setarg ray_dir_y");
    CL_CHECK_RETURN(clSetKernelArg(shadow_kernel_, arg_idx++, sizeof(cl_mem), &buf_ray_dir_z_), "setarg ray_dir_z");
    CL_CHECK_RETURN(clSetKernelArg(shadow_kernel_, arg_idx++, sizeof(cl_mem), &buf_ray_inv_dir_x_), "setarg ray_inv_dir_x");
    CL_CHECK_RETURN(clSetKernelArg(shadow_kernel_, arg_idx++, sizeof(cl_mem), &buf_ray_inv_dir_y_), "setarg ray_inv_dir_y");
    CL_CHECK_RETURN(clSetKernelArg(shadow_kernel_, arg_idx++, sizeof(cl_mem), &buf_ray_inv_dir_z_), "setarg ray_inv_dir_z");
    CL_CHECK_RETURN(clSetKernelArg(shadow_kernel_, arg_idx++, sizeof(cl_mem), &buf_ray_tmax_), "setarg ray_tmax");
    CL_CHECK_RETURN(clSetKernelArg(shadow_kernel_, arg_idx++, sizeof(cl_mem), &buf_occluded_), "setarg occluded");
    int num_rays_int = static_cast<int>(num_rays);
    CL_CHECK_RETURN(clSetKernelArg(shadow_kernel_, arg_idx++, sizeof(int), &num_rays_int), "setarg num_rays");

    // Launch kernel
    auto kernel_start = std::chrono::high_resolution_clock::now();

    size_t global_size = ((num_rays + work_group_size_ - 1) / work_group_size_) * work_group_size_;
    CL_CHECK_RETURN(clEnqueueNDRangeKernel(queue, shadow_kernel_, 1, nullptr,
                                    &global_size, &work_group_size_, 0, nullptr, nullptr), "enqueue kernel");
    err = clFinish(queue);
    CL_CHECK_RETURN(err, "clFinish after kernel");

    auto kernel_end = std::chrono::high_resolution_clock::now();

    // Read results
    occluded_out.resize(num_rays);
    CL_CHECK_RETURN(clEnqueueReadBuffer(queue, buf_occluded_, CL_TRUE, 0,
                                 num_rays * sizeof(uint8_t), occluded_out.data(),
                                 0, nullptr, nullptr), "read occluded");

    auto end_time = std::chrono::high_resolution_clock::now();

    // Update stats
    stats_.rays_traced += num_rays;
    stats_.upload_time_ms += std::chrono::duration<double, std::milli>(upload_end - upload_start).count();
    stats_.kernel_time_ms += std::chrono::duration<double, std::milli>(kernel_end - kernel_start).count();
    stats_.total_time_ms += std::chrono::duration<double, std::milli>(end_time - start_time).count();

    #undef CL_CHECK_RETURN
}

void BVHRayTracer::traceShadowRays(const std::vector<std::array<double, 3>>& origins,
                                   const std::vector<std::array<double, 3>>& directions,
                                   std::vector<uint8_t>& occluded_out) {
    size_t num_rays = origins.size();

    // Convert AoS to SoA
    std::vector<float> org_x(num_rays), org_y(num_rays), org_z(num_rays);
    std::vector<float> dir_x(num_rays), dir_y(num_rays), dir_z(num_rays);

    for (size_t i = 0; i < num_rays; ++i) {
        org_x[i] = static_cast<float>(origins[i][0]);
        org_y[i] = static_cast<float>(origins[i][1]);
        org_z[i] = static_cast<float>(origins[i][2]);
        dir_x[i] = static_cast<float>(directions[i][0]);
        dir_y[i] = static_cast<float>(directions[i][1]);
        dir_z[i] = static_cast<float>(directions[i][2]);
    }

    traceShadowRays(org_x, org_y, org_z, dir_x, dir_y, dir_z, occluded_out);
}

const OpenCLDeviceInfo& BVHRayTracer::getDeviceInfo() const {
    if (!context_) {
        static OpenCLDeviceInfo empty_info;
        return empty_info;
    }
    return context_->deviceInfo();
}

void BVHRayTracer::release() {
    if (shadow_kernel_) {
        clReleaseKernel(shadow_kernel_);
        shadow_kernel_ = nullptr;
    }
    if (program_) {
        clReleaseProgram(program_);
        program_ = nullptr;
    }

    // Release mesh buffers
    if (buf_tri_v0_x_) { clReleaseMemObject(buf_tri_v0_x_); buf_tri_v0_x_ = nullptr; }
    if (buf_tri_v0_y_) { clReleaseMemObject(buf_tri_v0_y_); buf_tri_v0_y_ = nullptr; }
    if (buf_tri_v0_z_) { clReleaseMemObject(buf_tri_v0_z_); buf_tri_v0_z_ = nullptr; }
    if (buf_tri_v1_x_) { clReleaseMemObject(buf_tri_v1_x_); buf_tri_v1_x_ = nullptr; }
    if (buf_tri_v1_y_) { clReleaseMemObject(buf_tri_v1_y_); buf_tri_v1_y_ = nullptr; }
    if (buf_tri_v1_z_) { clReleaseMemObject(buf_tri_v1_z_); buf_tri_v1_z_ = nullptr; }
    if (buf_tri_v2_x_) { clReleaseMemObject(buf_tri_v2_x_); buf_tri_v2_x_ = nullptr; }
    if (buf_tri_v2_y_) { clReleaseMemObject(buf_tri_v2_y_); buf_tri_v2_y_ = nullptr; }
    if (buf_tri_v2_z_) { clReleaseMemObject(buf_tri_v2_z_); buf_tri_v2_z_ = nullptr; }

    // Release BVH buffers
    if (buf_node_min_x_) { clReleaseMemObject(buf_node_min_x_); buf_node_min_x_ = nullptr; }
    if (buf_node_min_y_) { clReleaseMemObject(buf_node_min_y_); buf_node_min_y_ = nullptr; }
    if (buf_node_min_z_) { clReleaseMemObject(buf_node_min_z_); buf_node_min_z_ = nullptr; }
    if (buf_node_max_x_) { clReleaseMemObject(buf_node_max_x_); buf_node_max_x_ = nullptr; }
    if (buf_node_max_y_) { clReleaseMemObject(buf_node_max_y_); buf_node_max_y_ = nullptr; }
    if (buf_node_max_z_) { clReleaseMemObject(buf_node_max_z_); buf_node_max_z_ = nullptr; }
    if (buf_node_left_) { clReleaseMemObject(buf_node_left_); buf_node_left_ = nullptr; }
    if (buf_node_right_) { clReleaseMemObject(buf_node_right_); buf_node_right_ = nullptr; }
    if (buf_node_tri_start_) { clReleaseMemObject(buf_node_tri_start_); buf_node_tri_start_ = nullptr; }
    if (buf_node_tri_count_) { clReleaseMemObject(buf_node_tri_count_); buf_node_tri_count_ = nullptr; }
    if (buf_tri_indices_) { clReleaseMemObject(buf_tri_indices_); buf_tri_indices_ = nullptr; }

    // Release ray buffers
    if (buf_ray_org_x_) { clReleaseMemObject(buf_ray_org_x_); buf_ray_org_x_ = nullptr; }
    if (buf_ray_org_y_) { clReleaseMemObject(buf_ray_org_y_); buf_ray_org_y_ = nullptr; }
    if (buf_ray_org_z_) { clReleaseMemObject(buf_ray_org_z_); buf_ray_org_z_ = nullptr; }
    if (buf_ray_dir_x_) { clReleaseMemObject(buf_ray_dir_x_); buf_ray_dir_x_ = nullptr; }
    if (buf_ray_dir_y_) { clReleaseMemObject(buf_ray_dir_y_); buf_ray_dir_y_ = nullptr; }
    if (buf_ray_dir_z_) { clReleaseMemObject(buf_ray_dir_z_); buf_ray_dir_z_ = nullptr; }
    if (buf_ray_inv_dir_x_) { clReleaseMemObject(buf_ray_inv_dir_x_); buf_ray_inv_dir_x_ = nullptr; }
    if (buf_ray_inv_dir_y_) { clReleaseMemObject(buf_ray_inv_dir_y_); buf_ray_inv_dir_y_ = nullptr; }
    if (buf_ray_inv_dir_z_) { clReleaseMemObject(buf_ray_inv_dir_z_); buf_ray_inv_dir_z_ = nullptr; }
    if (buf_ray_tmax_) { clReleaseMemObject(buf_ray_tmax_); buf_ray_tmax_ = nullptr; }
    if (buf_occluded_) { clReleaseMemObject(buf_occluded_); buf_occluded_ = nullptr; }

    context_.reset();
    initialized_ = false;
    mesh_uploaded_ = false;
    bvh_uploaded_ = false;
    max_rays_ = 0;
}

}  // namespace opencl
}  // namespace gpu

#endif  // PTM_GPU_OPENCL
